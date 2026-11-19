"""
Issue Reporter for DLM Extraction Pipeline.
Maintains structured records for both Technical Exceptions and Business Validation Discrepancies
conforming to Deliverable 6 of the Observability Architecture.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import hashlib
import os
import traceback
import uuid
from datetime import datetime, timezone
import pandas as pd

from src.common.run_context import get_current_context, RunContext
from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IssueRecord:
    issue_id: str
    issue_type: str  # TECHNICAL_EXCEPTION | BUSINESS_VALIDATION_DISCREPANCY
    run_id: str
    batch_id: Optional[str]
    upc: Optional[str]
    zone: Optional[str]
    phase: Optional[str]
    cycle: Optional[int]
    stage_id: Optional[str]
    stage_name: Optional[str]
    module: Optional[str]
    function: Optional[str]
    issue_description: str
    expected_behaviour: str
    actual_behaviour: str
    exception_type: Optional[str] = None
    technical_error_message: Optional[str] = None
    root_cause: str = "Under investigation"
    business_impact: Optional[str] = None
    required_action: Optional[str] = None
    status: str = "OPEN"
    traceback_ref: Optional[str] = None
    event_fingerprint: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class IssueReporter:
    """
    Collects, manages, and exports structured issues encountered during pipeline execution.
    """

    def __init__(self, tracebacks_dir: Optional[str] = None):
        self.issues: List[IssueRecord] = []
        self.tracebacks_dir = tracebacks_dir or "./output/tracebacks"

    def _compute_fingerprint(self, parts: List[Any]) -> str:
        raw = "|".join(str(p or "") for p in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def record_technical_exception(
        self,
        exc: Exception,
        stage_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        module: Optional[str] = None,
        function: Optional[str] = None,
        issue_description: Optional[str] = None,
        expected_behaviour: Optional[str] = None,
        actual_behaviour: Optional[str] = None,
        business_impact: Optional[str] = None,
        required_action: Optional[str] = None,
        ctx: Optional[RunContext] = None,
    ) -> IssueRecord:
        """Record a technical runtime exception with optional traceback file."""
        c = ctx or get_current_context()
        stg_id = stage_id or c.stage_id or "UNKNOWN"
        stg_name = stage_name or c.stage_name or stg_id

        exc_type = type(exc).__name__
        exc_msg = str(exc)
        tb_str = traceback.format_exc()

        # Save traceback file
        tb_ref = None
        try:
            safe_upc = (c.upc or "unknown").replace("/", "_")
            tb_dir = os.path.join(self.tracebacks_dir, c.run_id, f"upc-{safe_upc}")
            os.makedirs(tb_dir, exist_ok=True)
            tb_path = os.path.join(tb_dir, f"{stg_id}.txt")
            with open(tb_path, "w", encoding="utf-8") as f:
                f.write(tb_str)
            tb_ref = tb_path
        except Exception as e:
            logger.debug("Could not write traceback file: %s", e)

        issue_id = f"ISS-{uuid.uuid4().hex[:8].upper()}"
        fingerprint = self._compute_fingerprint([c.upc, stg_id, exc_type, exc_msg[:100]])

        rec = IssueRecord(
            issue_id=issue_id,
            issue_type="TECHNICAL_EXCEPTION",
            run_id=c.run_id,
            batch_id=c.batch_id,
            upc=c.upc,
            zone=c.zone,
            phase=c.phase,
            cycle=c.cycle,
            stage_id=stg_id,
            stage_name=stg_name,
            module=module or "main",
            function=function or "",
            issue_description=issue_description or f"Pipeline encountered {exc_type} during {stg_name}.",
            expected_behaviour=expected_behaviour or f"{stg_name} should complete without error.",
            actual_behaviour=actual_behaviour or f"Raised {exc_type}: {exc_msg}",
            exception_type=exc_type,
            technical_error_message=exc_msg,
            root_cause="Under investigation",
            business_impact=business_impact or "The UPC output may be incomplete or missing.",
            required_action=required_action or "Inspect error details and rerun the UPC after resolution.",
            status="OPEN",
            traceback_ref=tb_ref,
            event_fingerprint=fingerprint,
        )
        self.issues.append(rec)
        logger.warning(f"Recorded technical issue {issue_id}: {rec.issue_description}")
        return rec

    def record_validation_discrepancy(
        self,
        check_name: str,
        scope: str,
        failed_count: int,
        checked_count: int,
        detail: str = "",
        section: str = "Validation",
        ctx: Optional[RunContext] = None,
    ) -> IssueRecord:
        """Record a business rule or schema validation failure."""
        c = ctx or get_current_context()
        issue_id = f"ISS-{uuid.uuid4().hex[:8].upper()}"
        fingerprint = self._compute_fingerprint([c.upc, section, check_name, scope])

        rec = IssueRecord(
            issue_id=issue_id,
            issue_type="BUSINESS_VALIDATION_DISCREPANCY",
            run_id=c.run_id,
            batch_id=c.batch_id,
            upc=c.upc,
            zone=c.zone,
            phase=c.phase,
            cycle=c.cycle,
            stage_id="VALIDATE",
            stage_name="Validation Checks",
            module="src.validation.validator",
            function="validate_all",
            issue_description=f"Validation '{check_name}' failed for {scope} ({failed_count}/{checked_count} records failed).",
            expected_behaviour=f"All {checked_count} records in {scope} should satisfy '{check_name}'.",
            actual_behaviour=f"{failed_count} records failed validation. {detail}".strip(),
            exception_type="ValidationError",
            technical_error_message=detail or f"Failed count: {failed_count}",
            root_cause="Under investigation",
            business_impact="Quality audit flagged data inconsistency in output deliverable.",
            required_action="Review failing records against standard specifications.",
            status="OPEN",
            traceback_ref=None,
            event_fingerprint=fingerprint,
        )
        self.issues.append(rec)
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all recorded issues to a pandas DataFrame."""
        if not self.issues:
            return pd.DataFrame(columns=[
                "issue_id", "issue_type", "run_id", "batch_id", "upc", "zone",
                "phase", "cycle", "stage_id", "stage_name", "module", "function",
                "issue_description", "expected_behaviour", "actual_behaviour",
                "exception_type", "technical_error_message", "root_cause",
                "business_impact", "required_action", "status", "traceback_ref",
                "event_fingerprint", "created_at",
            ])
        return pd.DataFrame([asdict(i) for i in self.issues])

    def save_issue_report(self, output_path: str) -> str:
        """Export all recorded issues to a CSV file (local or S3)."""
        df = self.to_dataframe()
        if output_path.startswith("s3://"):
            from src.common.fs_utils import write_file_bytes
            write_file_bytes(output_path, df.to_csv(index=False).encode("utf-8"))
        else:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df.to_csv(output_path, index=False)
        logger.info(f"Saved issue report with {len(df)} records to {output_path}")
        return output_path
