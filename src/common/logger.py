"""
Logger module for the DLM Extraction Pipeline.

Provides centralized, thread-safe logging configured with:
  1. Operator Console Handler (readable formatting with stage context)
  2. Telemetry File Handler (machine-readable JSON-Lines format)
"""

import logging
import os
import sys
from typing import Optional, Union, Any, Dict

from src.common.log_formatter import ConsoleFormatter, JsonLinesFormatter
from src.common.run_context import get_current_context

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "dlm_pipeline",
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    json_log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    level: Optional[Union[int, str]] = None,
) -> logging.Logger:
    """
    Configure and return a standard logger instance with console and JSON-lines file handlers.
    """
    if level is not None:
        log_level = level

    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    target_logger = logging.getLogger(name)
    target_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-initialization
    if target_logger.hasHandlers():
        target_logger.handlers.clear()

    # 1. Console Handler (stdout) with ConsoleFormatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ConsoleFormatter())
    target_logger.addHandler(console_handler)

    # 2. Traditional Text File Handler (if log_file specified)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(ConsoleFormatter())
        target_logger.addHandler(file_handler)

    # 3. JSON-Lines Telemetry File Handler
    if json_log_file:
        j_dir = os.path.dirname(json_log_file)
        if j_dir and not os.path.exists(j_dir):
            os.makedirs(j_dir, exist_ok=True)
        json_handler = logging.FileHandler(json_log_file, encoding="utf-8")
        json_handler.setLevel(log_level)
        json_handler.setFormatter(JsonLinesFormatter())
        target_logger.addHandler(json_handler)

    target_logger.propagate = False
    return target_logger


def log_stage_event(
    target_logger: logging.Logger,
    level: int,
    activity: str,
    execution_status: str = "SUCCESS",
    duration_ms: Optional[float] = None,
    records_read: Optional[int] = None,
    records_processed: Optional[int] = None,
    records_rejected: Optional[int] = None,
    records_written: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a structured telemetry log event conforming to the Deliverable 5 standard format.
    """
    ctx = get_current_context()
    extra: Dict[str, Any] = {
        "run_id": ctx.run_id,
        "batch_id": ctx.batch_id,
        "upc": ctx.upc,
        "zone": ctx.zone,
        "phase": ctx.phase,
        "cycle": ctx.cycle,
        "stage_id": ctx.stage_id,
        "stage_name": ctx.stage_name,
        "execution_status": execution_status,
        "duration_ms": duration_ms,
        "records_read": records_read,
        "records_processed": records_processed,
        "records_rejected": records_rejected,
        "records_written": records_written,
        "details": details,
        "error": error,
        "validation": validation,
    }
    target_logger.log(level, activity, extra=extra)


# Default application-wide logger
logger = setup_logger("dlm_pipeline")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return configured logger instance."""
    if name:
        return logging.getLogger(name)
    return logger
