"""
Run and Stage Context Management for DLM Extraction Pipeline.
Provides thread-safe context propagation across pipeline stages using contextvars.
"""

from dataclasses import dataclass, field
from contextvars import ContextVar
from typing import Optional, Dict, Any
import time
import uuid
from datetime import datetime, timezone


@dataclass
class RunContext:
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    batch_id: Optional[str] = None
    upc: Optional[str] = None
    zone: Optional[str] = None
    phase: Optional[str] = None
    cycle: Optional[int] = None
    stage_id: Optional[str] = None
    stage_name: Optional[str] = None
    stage_start_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "upc": self.upc,
            "zone": self.zone,
            "phase": self.phase,
            "cycle": self.cycle,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
        }


# Context variable for thread-safe context propagation
_CURRENT_CONTEXT: ContextVar[RunContext] = ContextVar(
    "current_run_context", default=RunContext()
)


def get_current_context() -> RunContext:
    """Retrieve the current thread-safe run context."""
    return _CURRENT_CONTEXT.get()


def init_run_context(
    run_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    upc: Optional[str] = None,
    zone: Optional[str] = None,
    phase: Optional[str] = None,
    cycle: Optional[int] = None,
) -> RunContext:
    """Initialize or update the active run context."""
    ctx = RunContext(
        run_id=run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}",
        batch_id=batch_id,
        upc=upc,
        zone=zone,
        phase=str(phase) if phase is not None else None,
        cycle=int(cycle) if cycle is not None else None,
    )
    _CURRENT_CONTEXT.set(ctx)
    return ctx


class stage_scope:
    """
    Context manager to track execution of a specific pipeline stage.
    Times the stage and updates stage_id / stage_name in the active context.
    """

    def __init__(self, stage_id: str, stage_name: str):
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.start_time: float = 0.0
        self.duration_ms: float = 0.0
        self.prev_stage_id: Optional[str] = None
        self.prev_stage_name: Optional[str] = None

    def __enter__(self):
        ctx = get_current_context()
        self.prev_stage_id = ctx.stage_id
        self.prev_stage_name = ctx.stage_name
        ctx.stage_id = self.stage_id
        ctx.stage_name = self.stage_name
        ctx.stage_start_time = time.time()
        self.start_time = ctx.stage_start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.time() - self.start_time) * 1000.0
        ctx = get_current_context()
        ctx.stage_id = self.prev_stage_id
        ctx.stage_name = self.prev_stage_name
        ctx.stage_start_time = None
        return False
