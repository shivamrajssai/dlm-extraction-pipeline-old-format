"""
Custom Log Formatters for DLM Extraction Pipeline.
Supports structured JSON-Lines for automated telemetry and clean readable console output.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict

from src.common.run_context import get_current_context


class JsonLinesFormatter(logging.Formatter):
    """
    Formats log records into machine-readable JSON-Lines (NDJSON).
    Incorporates run_id, batch_id, upc, stage_id, duration_ms, and record metrics.
    """

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_current_context()

        # Build base event structure conforming to Deliverable 5 specification
        event: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "run_id": getattr(record, "run_id", ctx.run_id),
            "batch_id": getattr(record, "batch_id", ctx.batch_id),
            "upc": getattr(record, "upc", ctx.upc),
            "zone": getattr(record, "zone", ctx.zone),
            "phase": getattr(record, "phase", ctx.phase),
            "cycle": getattr(record, "cycle", ctx.cycle),
            "stage_id": getattr(record, "stage_id", ctx.stage_id),
            "stage_name": getattr(record, "stage_name", ctx.stage_name),
            "module": record.name,
            "function": record.funcName,
            "level": record.levelname,
            "activity": record.getMessage(),
            "execution_status": getattr(record, "execution_status", "INFO" if record.levelno < logging.WARNING else record.levelname),
            "duration_ms": getattr(record, "duration_ms", None),
            "records_read": getattr(record, "records_read", None),
            "records_processed": getattr(record, "records_processed", None),
            "records_rejected": getattr(record, "records_rejected", None),
            "records_written": getattr(record, "records_written", None),
            "details": getattr(record, "details", None),
        }

        # Attach error object if exception info is present or error dict passed
        if record.exc_info:
            exc_type = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            exc_msg = str(record.exc_info[1]) if record.exc_info[1] else ""
            event["error"] = {
                "exception_type": exc_type,
                "message": exc_msg,
                "traceback_ref": getattr(record, "traceback_ref", None),
            }
        elif hasattr(record, "error") and record.error:
            event["error"] = record.error

        # Attach validation object if validation metrics passed
        if hasattr(record, "validation") and record.validation:
            event["validation"] = record.validation

        # Remove keys with None values in optional blocks, but keep standard fields
        return json.dumps(event, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Formats log records cleanly for operator console output.
    Format: YYYY-MM-DD HH:MM:SS [LEVEL] [STAGE|UPC] message
    """

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_current_context()
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Context prefix
        parts = []
        if ctx.stage_id:
            parts.append(ctx.stage_id)
        if ctx.upc:
            parts.append(ctx.upc)
        ctx_tag = f"[{':'.join(parts)}]" if parts else f"[{record.name}:{record.lineno}]"

        msg = record.getMessage()

        # Append duration or records if passed in extra
        extra_info = []
        if hasattr(record, "duration_ms") and record.duration_ms is not None:
            extra_info.append(f"duration={record.duration_ms:.1f}ms")
        if hasattr(record, "records_processed") and record.records_processed is not None:
            extra_info.append(f"processed={record.records_processed}")
        if hasattr(record, "execution_status") and record.execution_status not in ("INFO", None):
            extra_info.append(f"status={record.execution_status}")

        if extra_info:
            msg = f"{msg} ({', '.join(extra_info)})"

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return f"{ts} [{level}] {ctx_tag} {msg}"
