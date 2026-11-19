"""
Defect image uploader module.
Ported from Databricks DLM Extraction Pipeline (Cell 16).

Indexes images under the NSV deliverable folder, uploads breach images
to the central CDN API endpoint, and updates image_url / image_urls.
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

from src.common.logger import get_logger
from src.common.fs_utils import path_exists, walk_files

logger = get_logger(__name__)

DEFAULT_UPLOAD_URL = "https://nhaidhms-stage-api.dic.org.in/defects/images/upload"


class ImageUploader:
    """
    Manages indexing and uploading of pothole and cracking defect images.
    """

    def __init__(
        self,
        deliverable_dir: str,
        upload_url: str = DEFAULT_UPLOAD_URL,
        max_workers: int = 4,
        retries: int = 2,
        retry_delay_sec: float = 1.0,
    ):
        self.deliverable_dir = deliverable_dir
        self.upload_url = upload_url.rstrip('/')
        if not self.upload_url.endswith('/defects/images/upload'):
            self.upload_url = f"{self.upload_url}/defects/images/upload"
        self.max_workers = max_workers
        self.retries = retries
        self.retry_delay_sec = retry_delay_sec
        self.session = requests.Session()
        self.image_index: Dict[str, Dict[str, str]] = {'cracking': {}, 'pothole': {}}
        self._index_images()

    def _index_images(self) -> None:
        """
        Scan deliverable directory and build filename/stem to absolute path mapping.
        """
        if not path_exists(self.deliverable_dir):
            logger.warning("Deliverable directory does not exist: %s", self.deliverable_dir)
            return

        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        count = 0
        all_files = walk_files(self.deliverable_dir, list(valid_exts))
        for full_path in all_files:
            f = os.path.basename(full_path)
            full_lower = full_path.lower()
            kind = 'cracking' if 'crack' in full_lower else ('pothole' if 'pothole' in full_lower else None)
            if kind is None:
                continue
            stem = os.path.splitext(f)[0]
            self.image_index[kind].setdefault(f, full_path)
            self.image_index[kind].setdefault(stem, full_path)
            count += 1

        logger.info(
            "Indexed %d image files (cracking: %d, pothole: %d)",
            count,
            len(set(self.image_index['cracking'].values())),
            len(set(self.image_index['pothole'].values())),
        )

    def find_image(self, name: str, kind: str) -> Optional[str]:
        """
        Find path of image by name or stem. Checks specified kind first, then other.
        """
        if not name:
            return None
        name = str(name).strip()
        stem = os.path.splitext(name)[0]
        other = 'pothole' if kind == 'cracking' else 'cracking'

        return (
            self.image_index[kind].get(name) or
            self.image_index[kind].get(stem) or
            self.image_index[other].get(name) or
            self.image_index[other].get(stem)
        )

    def upload_one_image(self, path: str, kind: str, display_defect_id: str) -> Tuple[bool, str]:
        """
        Upload single image file to the CDN.
        """
        safe_id = re.sub(r'[^A-Za-z0-9._-]', '-', display_defect_id)[:128]
        last_error = ''

        for attempt in range(1, self.retries + 1):
            try:
                if path.startswith("s3://"):
                    import pyarrow.fs as pafs
                    fs, path_in_fs = pafs.FileSystem.from_uri(path)
                    with fs.open_input_stream(path_in_fs) as fh:
                        img_bytes = fh.read()
                    files = {'image': (os.path.basename(path), io.BytesIO(img_bytes), 'image/jpeg')}
                else:
                    with open(path, 'rb') as fh:
                        files = {'image': (os.path.basename(path), io.BytesIO(fh.read()), 'image/jpeg')}

                data = {'defect_type': kind, 'display_defect_id': safe_id}
                r = self.session.post(self.upload_url, files=files, data=data, timeout=30)

                if r.status_code == 201:
                    cdn_url = (r.json().get('data') or {}).get('url', '')
                    return True, cdn_url
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_error = str(e)[:200]

            if attempt < self.retries:
                time.sleep(self.retry_delay_sec)

        return False, last_error

    def process_table_images(
        self,
        table_name: str,
        df: pd.DataFrame,
        defect_type: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Find images for breach rows in df, upload them, and update image_url / image_urls.
        """
        summary = {
            'table': table_name,
            'defect_type': defect_type,
            'images_to_upload': 0,
            'no_file_found': 0,
            'uploaded': 0,
            'failed': 0,
            'rows_updated': 0,
        }

        if df is None or df.empty or 'image_name' not in df.columns:
            return df, summary

        out = df.copy()
        if 'image_url' not in out.columns:
            out['image_url'] = None
        if 'image_urls' not in out.columns:
            out['image_urls'] = None

        has_name = out['image_name'].notna() & (out['image_name'] != '')
        has_did = out['display_defect_id'].notna() & (out['display_defect_id'] != '')
        needs_url = out['image_url'].isna() | (out['image_url'] == '')

        todo_mask = has_name & has_did & needs_url
        todo = out[todo_mask].groupby('image_name')['display_defect_id'].first().reset_index()

        summary['images_to_upload'] = len(todo)
        if todo.empty:
            return out, summary

        tasks = []
        no_file = []
        for _, r in todo.iterrows():
            img_name = r['image_name']
            did = r['display_defect_id']
            img_path = self.find_image(img_name, defect_type)
            if img_path and path_exists(img_path):
                tasks.append((img_name, img_path, did))
            else:
                no_file.append(img_name)

        summary['no_file_found'] = len(no_file)
        url_map: Dict[str, str] = {}

        if tasks:
            def _worker(item):
                n, p, d = item
                ok, res = self.upload_one_image(p, defect_type, d)
                return n, ok, res

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(_worker, tasks))

            for n, ok, res in results:
                if ok:
                    url_map[n] = res
                    summary['uploaded'] += 1
                else:
                    logger.warning("Image upload failed for %s: %s", n, res)
                    summary['failed'] += 1

        if url_map:
            updated_count = 0
            for idx, row in out.iterrows():
                name = row.get('image_name')
                if name in url_map:
                    cdn_url = url_map[name]
                    out.at[idx, 'image_url'] = cdn_url
                    existing_urls = row.get('image_urls')
                    if not existing_urls or pd.isna(existing_urls):
                        out.at[idx, 'image_urls'] = [cdn_url]
                    updated_count += 1
            summary['rows_updated'] = updated_count

        return out, summary
