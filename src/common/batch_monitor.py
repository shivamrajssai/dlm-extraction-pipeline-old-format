"""
Batch Monitoring and Lifecycle Accounting for DLM Extraction Pipeline.
Maintains execution state and summary statistics across all UPCs in a batch invocation
conforming to Deliverable 7 of the Observability Architecture.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import os
import time
from datetime import datetime, timezone
import pandas as pd

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UPCExecutionRecord:
    upc: str
    zone: str
    phase: str
    cycle: int
    status: str  # SUCCESS | FAILED | SKIPPED
    start_time: str
    end_time: Optional[str] = None
    duration_ms: float = 0.0
    tables_written: int = 0
    records_written: int = 0
    validation_pass: int = 0
    validation_fail: int = 0
    validation_skip: int = 0
    error_message: Optional[str] = None
    excel_export: Optional[str] = None
    csv_export_dir: Optional[str] = None


class BatchMonitor:
    """
    Tracks and aggregates batch-level metrics across all processed UPCs.
    """

    def __init__(self, batch_id: Optional[str] = None, batch_file: Optional[str] = None):
        self.batch_id = batch_id or f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self.batch_file = batch_file
        self.start_time: float = time.time()
        self.start_iso: str = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[float] = None
        self.end_iso: Optional[str] = None
        self.total_upcs_received: int = 0
        self.records: Dict[str, UPCExecutionRecord] = {}

    def set_total_upcs(self, count: int) -> None:
        self.total_upcs_received = count

    def start_upc(self, upc: str, zone: str, phase: str, cycle: int) -> None:
        rec = UPCExecutionRecord(
            upc=upc,
            zone=zone,
            phase=str(phase),
            cycle=int(cycle),
            status="RUNNING",
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self.records[upc] = rec

    def complete_upc(
        self,
        upc: str,
        duration_ms: float,
        tables_written: int = 0,
        records_written: int = 0,
        validation_summary: Optional[Dict[str, int]] = None,
        excel_export: Optional[str] = None,
        csv_export_dir: Optional[str] = None,
    ) -> None:
        val_sum = validation_summary or {}
        if upc in self.records:
            rec = self.records[upc]
            rec.status = "SUCCESS"
            rec.end_time = datetime.now(timezone.utc).isoformat()
            rec.duration_ms = duration_ms
            rec.tables_written = tables_written
            rec.records_written = records_written
            rec.validation_pass = val_sum.get("PASS", 0)
            rec.validation_fail = val_sum.get("FAIL", 0)
            rec.validation_skip = val_sum.get("SKIP", 0)
            rec.excel_export = excel_export
            rec.csv_export_dir = csv_export_dir

    def fail_upc(self, upc: str, duration_ms: float, error_message: str) -> None:
        if upc in self.records:
            rec = self.records[upc]
            rec.status = "FAILED"
            rec.end_time = datetime.now(timezone.utc).isoformat()
            rec.duration_ms = duration_ms
            rec.error_message = error_message[:500]

    def skip_upc(self, upc: str, zone: str, phase: str, cycle: int, reason: str) -> None:
        rec = UPCExecutionRecord(
            upc=upc,
            zone=zone,
            phase=str(phase),
            cycle=int(cycle),
            status="SKIPPED",
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            error_message=reason,
        )
        self.records[upc] = rec

    def finish_batch(self) -> Dict[str, Any]:
        self.end_time = time.time()
        self.end_iso = datetime.now(timezone.utc).isoformat()
        total_duration_ms = (self.end_time - self.start_time) * 1000.0

        completed = sum(1 for r in self.records.values() if r.status == "SUCCESS")
        failed = sum(1 for r in self.records.values() if r.status == "FAILED")
        skipped = sum(1 for r in self.records.values() if r.status == "SKIPPED")

        summary = {
            "batch_id": self.batch_id,
            "batch_file": self.batch_file,
            "started_at": self.start_iso,
            "finished_at": self.end_iso,
            "duration_ms": total_duration_ms,
            "total_upcs_received": self.total_upcs_received,
            "upcs_started": len(self.records),
            "upcs_completed": completed,
            "upcs_failed": failed,
            "upcs_skipped": skipped,
            "upcs_not_processed": max(0, self.total_upcs_received - len(self.records)),
            "total_tables_written": sum(r.tables_written for r in self.records.values()),
            "total_records_written": sum(r.records_written for r in self.records.values()),
        }
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                "upc", "zone", "phase", "cycle", "status", "start_time", "end_time",
                "duration_ms", "tables_written", "records_written", "validation_pass",
                "validation_fail", "validation_skip", "error_message",
            ])
        return pd.DataFrame([asdict(r) for r in self.records.values()])

    def save_batch_summary(self, csv_path: str, json_path: Optional[str] = None) -> None:
        df = self.to_dataframe()
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        if csv_path.startswith("s3://"):
            from src.common.fs_utils import write_file_bytes
            write_file_bytes(csv_path, csv_bytes)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, "wb") as f:
                f.write(csv_bytes)
        logger.info("Saved batch summary CSV to %s", csv_path)

        if json_path:
            summary = self.finish_batch()
            json_bytes = json.dumps(summary, indent=2).encode("utf-8")
            if json_path.startswith("s3://"):
                from src.common.fs_utils import write_file_bytes
                write_file_bytes(json_path, json_bytes)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
                with open(json_path, "wb") as f:
                    f.write(json_bytes)
            logger.info("Saved run summary JSON to %s", json_path)
