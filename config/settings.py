"""
Configuration and runtime settings for the DLM Extraction Pipeline.

Migrated from Notebook Cell 2.
Replaces Databricks widgets (dbutils.widgets) with command-line arguments and .env support.
Supports S3 production paths and local test execution.
"""

import argparse
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Dict, Any

from src.common.logger import get_logger

logger = get_logger(__name__)


def parse_env_file(path: str = ".env") -> Dict[str, str]:
    """
    Parse .env file supporting both standard KEY=VAL and bash parameter expansions:
    KEY="${KEY:-DEFAULT}" or KEY="${OTHER_KEY}/suffix".
    """
    env: Dict[str, str] = {}
    if not os.path.exists(path):
        return env

    def resolve_bash_val(val: str, current_env: Dict[str, str]) -> str:
        v = val.strip().strip('"').strip("'")

        def eval_expr(expr: str) -> str:
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                var_name = var_name.strip()
                if var_name in current_env and current_env[var_name]:
                    return current_env[var_name]
                if var_name in os.environ and os.environ[var_name]:
                    return os.environ[var_name]
                return resolve_bash_val(default, current_env)
            else:
                var_name = expr.strip()
                return current_env.get(var_name, os.getenv(var_name, ""))

        while "${" in v:
            start = v.find("${")
            depth = 0
            end = -1
            for i in range(start, len(v)):
                if v[i:i + 2] == "${":
                    depth += 1
                elif v[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                break
            inner = v[start + 2:end]
            resolved = eval_expr(inner)
            v = v[:start] + resolved + v[end + 1:]
        return v

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, val = line.split("=", 1)
                    k = k.strip()
                    env[k] = resolve_bash_val(val, env)
    except Exception as e:
        logger.warning("Error reading .env file %s: %s", path, e)

    return env


@dataclass
class PipelineSettings:
    """Settings and resolved paths for a single UPC pipeline run."""
    zone: str = "B"
    phase: str = "2"
    cycle: str = "1"
    upc: str = "N_07024_01001_GJ"
    aws_region: str = "ap-south-1"
    bucket: str = "dhms-new-stgae"

    # Infrastructure S3 Roots
    raw_s3_root: str = ""
    bronze_s3_root: str = ""
    bronze_nsv_root: str = ""
    bronze_nsv_backup_root: str = ""
    silver_s3_root: str = ""
    output_root: str = ""
    nsv_downloads_root: str = ""
    work_dir: str = ""
    tracker_s3: str = ""
    batch_log_s3: str = ""

    # Optional local data root override (for testing/local execution)
    data_root: Optional[str] = None

    # Pipeline runtime paths
    tracker_excel_path: str = ""
    tracker_sheet_name: str = ""
    d1_dir: str = ""
    d2_dir: str = ""
    d3_dir: str = ""
    d4_dir: str = ""
    backup_base_dir: str = ""
    deliverable_dir: str = ""
    image_upload_base_url: str = "https://nhaidhms-stage-api.dic.org.in"
    silver_output_dir: str = "./output"
    validation_report_dir: str = "./output/validation_reports"
    exports_dir: str = "./output/exports"
    log_level: str = "INFO"
    log_file: Optional[str] = None
    json_log_file: Optional[str] = "./output/logs/execution.jsonl"
    run_date: date = field(default_factory=lambda: datetime.today().date())

    def resolve_paths(self) -> None:
        """Resolve dynamic partition and folder hierarchy for NSV and Backup."""
        # 1. Tracker path
        self.tracker_excel_path = self.tracker_s3
        self.tracker_sheet_name = f"Zone_{self.zone}"

        # 2. Local vs S3 resolution
        if self.data_root and os.path.exists(self.data_root):
            # Local data structure override
            nsv_root = os.path.join(
                self.data_root,
                "Source=NSV",
                f"Zone={self.zone}",
                f"Phase={self.phase}",
                f"Cycle={self.cycle}",
                f"UPC={self.upc}",
            )
            nsv_date = find_latest_date_partition(nsv_root)
            upc_base = os.path.join(nsv_root, nsv_date) if nsv_date else nsv_root
            deliverable_base = find_deliverable_parent(upc_base)

            self.d1_dir = os.path.join(deliverable_base, "Deliverable=Deliverable - 1")
            self.d2_dir = os.path.join(deliverable_base, "Deliverable=Deliverable - 2")
            self.d3_dir = os.path.join(deliverable_base, "Deliverable=Deliverable - 3")
            self.d4_dir = os.path.join(deliverable_base, "Deliverable=Deliverable - 4")

            backup_root = os.path.join(
                self.data_root,
                "Source=NSV_Backup",
                f"Zone={self.zone}",
                f"Phase={self.phase}",
                f"Cycle={self.cycle}",
                f"UPC={self.upc}",
            )
            backup_date = find_latest_date_partition(backup_root)
            if backup_date:
                backup_date_path = os.path.join(backup_root, backup_date)
                self.backup_base_dir = find_lane_parent(backup_date_path) or backup_date_path
            else:
                self.backup_base_dir = backup_root

            self.deliverable_dir = os.path.join(
                self.data_root,
                f"Zone_{self.zone}",
                f"Phase_{self.phase}",
                f"Cycle_{self.cycle}",
                self.upc,
                "deliverable",
            )
        else:
            # S3 production layout
            # Bronze input layout:
            #   <BRONZE_S3_ROOT>/Source=NSV/Zone=<Z>/Phase=<P>/Cycle=<C>/UPC=<UPC>/...
            #   <BRONZE_S3_ROOT>/Source=NSV_Backup/Zone=<Z>/Phase=<P>/Cycle=<C>/UPC=<UPC>/...
            nsv_base = self.bronze_nsv_root or os.path.join(self.bronze_s3_root, "Source=NSV")
            nsv_upc_root = os.path.join(
                nsv_base,
                f"Zone={self.zone}",
                f"Phase={self.phase}",
                f"Cycle={self.cycle}",
                f"UPC={self.upc}",
            )
            nsv_date = find_latest_date_partition(nsv_upc_root)
            upc_base = os.path.join(nsv_upc_root, nsv_date) if nsv_date else nsv_upc_root
            deliv_parent = find_deliverable_parent(upc_base)

            self.d1_dir = os.path.join(deliv_parent, "Deliverable=Deliverable - 1")
            self.d2_dir = os.path.join(deliv_parent, "Deliverable=Deliverable - 2")
            self.d3_dir = os.path.join(deliv_parent, "Deliverable=Deliverable - 3")
            self.d4_dir = os.path.join(deliv_parent, "Deliverable=Deliverable - 4")

            backup_base = self.bronze_nsv_backup_root or os.path.join(self.bronze_s3_root, "Source=NSV_Backup")
            backup_upc_root = os.path.join(
                backup_base,
                f"Zone={self.zone}",
                f"Phase={self.phase}",
                f"Cycle={self.cycle}",
                f"UPC={self.upc}",
            )
            backup_date = find_latest_date_partition(backup_upc_root)
            if backup_date:
                backup_date_path = os.path.join(backup_upc_root, backup_date)
                self.backup_base_dir = find_lane_parent(backup_date_path) or backup_date_path
            else:
                self.backup_base_dir = backup_upc_root

            self.deliverable_dir = os.path.join(
                self.nsv_downloads_root,
                f"Zone_{self.zone}",
                f"Phase_{self.phase}",
                f"Cycle_{self.cycle}",
                self.upc,
                "deliverable",
            )

        # Output paths
        if self.data_root:
            local_out = os.path.abspath("./output")
            self.silver_output_dir = os.path.join(
                local_out,
                f"Zone={self.zone}",
                f"Phase={self.phase}",
                f"Cycle={self.cycle}",
                f"UPC={self.upc}",
                "dlm_kpi_new",
            )
            self.validation_report_dir = os.path.join(local_out, "validation_reports")
            self.exports_dir = os.path.join(local_out, "exports")
        else:
            if self.silver_s3_root:
                upc_partition = f"Zone={self.zone}/Phase={self.phase}/Cycle={self.cycle}/UPC={self.upc}"
                self.silver_output_dir = f"{self.silver_s3_root.rstrip('/')}/{upc_partition}/dlm_kpi_new"
                # Validation results saved in s3://dhms-new-stgae/DHMS/NSV/Silver/Zone=<Z>/Phase=<P>/Cycle=<C>/UPC=<UPC>
                self.validation_report_dir = f"{self.silver_s3_root.rstrip('/')}/{upc_partition}"
                self.exports_dir = f"{self.silver_s3_root.rstrip('/')}/{upc_partition}/exports"
            elif self.work_dir:
                self.validation_report_dir = os.path.join(self.work_dir, "validation_reports")
                self.exports_dir = os.path.join(self.work_dir, "exports")

    @classmethod
    def from_cli(cls, args: Any, env_file: str = ".env") -> "PipelineSettings":
        """
        Construct PipelineSettings from CLI args merged with .env file.
        """
        env = parse_env_file(env_file)

        zone = getattr(args, "zone", None) or env.get("ZONE", "B")
        phase = getattr(args, "phase", None) or env.get("PHASE", "2")
        cycle = getattr(args, "cycle", None) or env.get("CYCLE", "1")
        upc = getattr(args, "upc", None) or env.get("UPC", "N_07024_01001_GJ")
        data_root = getattr(args, "data_root", None) or env.get("DATA_ROOT", None)

        settings = cls(
            zone=str(zone),
            phase=str(phase),
            cycle=str(cycle),
            upc=str(upc),
            aws_region=env.get("AWS_REGION", "ap-south-1"),
            bucket=env.get("BUCKET", "dhms-new-stgae"),
            raw_s3_root=env.get("RAW_S3_ROOT", ""),
            bronze_s3_root=env.get("BRONZE_S3_ROOT", ""),
            bronze_nsv_root=env.get("BRONZE_NSV_ROOT", ""),
            bronze_nsv_backup_root=env.get("BRONZE_NSV_BACKUP_ROOT", ""),
            silver_s3_root=env.get("SILVER_S3_ROOT", ""),
            output_root=env.get("OUTPUT_ROOT", ""),
            nsv_downloads_root=env.get("NSV_DOWNLOADS_ROOT", ""),
            work_dir=env.get("WORK_DIR", ""),
            tracker_s3=env.get("TRACKER_S3", ""),
            batch_log_s3=env.get("BATCH_LOG_S3", ""),
            data_root=data_root,
            image_upload_base_url=env.get("IMAGE_UPLOAD_BASE_URL", "https://nhaidhms-stage-api.dic.org.in"),
            log_level=env.get("LOG_LEVEL", "INFO"),
            log_file=env.get("LOG_FILE", None),
            json_log_file=env.get("JSON_LOG_FILE", "./output/logs/execution.jsonl"),
        )
        settings.resolve_paths()
        return settings


from src.common.fs_utils import path_exists, list_dir, is_dir, walk_files

def find_latest_date_partition(base_path: str) -> Optional[str]:
    """Find the latest year=*/month=*/day=* partition in base_path."""
    try:
        years = sorted([y for y in list_dir(base_path) if y.startswith("year=")], reverse=True)
        for y in years:
            yp = f"{base_path}/{y}" if base_path.startswith("s3://") else os.path.join(base_path, y)
            months = sorted([m for m in list_dir(yp) if m.startswith("month=")], reverse=True)
            for m in months:
                mp = f"{yp}/{m}" if base_path.startswith("s3://") else os.path.join(yp, m)
                days = sorted([d for d in list_dir(mp) if d.startswith("day=")], reverse=True)
                for d in days:
                    return f"{y}/{m}/{d}"
    except Exception as e:
        logger.debug(f"Direct traversal error in find_latest_date_partition: {e}")

    # Fallback for S3 / nested structures: inspect objects under base_path
    try:
        import re
        all_files = walk_files(base_path)
        date_parts = set()
        for f in all_files:
            m = re.search(r"(year=\d+/month=\d+/day=\d+)", f)
            if m:
                date_parts.add(m.group(1))
        if date_parts:
            return sorted(date_parts, reverse=True)[0]
    except Exception as e:
        logger.debug(f"Regex fallback error in find_latest_date_partition: {e}")

    return None


def find_deliverable_parent(base: str) -> str:
    """Find the directory containing Deliverable=* partitions."""
    try:
        children = sorted(list_dir(base))
        if any(c.startswith("Deliverable=") for c in children):
            return base
        for child in children:
            child_path = f"{base}/{child}" if base.startswith("s3://") else os.path.join(base, child)
            sub_children = list_dir(child_path)
            if any(gc.startswith("Deliverable=") for gc in sub_children):
                return child_path
            found = find_deliverable_parent(child_path)
            if found != child_path and any(gc.startswith("Deliverable=") for gc in list_dir(found)):
                return found
    except Exception:
        pass

    # Fallback: inspect actual files found under base
    try:
        all_files = walk_files(base)
        for f in all_files:
            if "Deliverable=Deliverable -" in f:
                idx = f.find("Deliverable=Deliverable -")
                return f[:idx].rstrip("/")
    except Exception:
        pass

    return base


def find_lane_parent(base: str) -> Optional[str]:
    """Find the directory containing Lane=* partitions."""
    try:
        children = sorted(list_dir(base))
        if any(c.startswith("Lane=") for c in children):
            return base
        for child in children:
            child_path = f"{base}/{child}" if base.startswith("s3://") else os.path.join(base, child)
            found = find_lane_parent(child_path)
            if found:
                return found
    except Exception:
        pass

    # Fallback: inspect actual files found under base
    try:
        all_files = walk_files(base)
        for f in all_files:
            if "/Lane=" in f:
                idx = f.find("/Lane=")
                return f[:idx]
    except Exception:
        pass

    return None
