"""Progress tracking for pipeline runs using Server-Sent Events (SSE)."""

import asyncio
import json
import time
from datetime import datetime
from uuid import uuid4
from typing import AsyncGenerator

from app.core.logging_config import get_agent_logger

logger = get_agent_logger("ProgressTracker")


class PipelineProgress:
    """Tracks progress of a pipeline run and emits events for SSE."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: asyncio.Queue = asyncio.Queue()
        self.current_phase = ""
        self.current_step = ""
        self.phases_completed = 0
        self.total_phases = 7  # Updated dynamically by supervisor
        self.status = "pending"
        self.started_at = datetime.now().isoformat()
        self.logs: list[dict] = []
        self.result: dict | None = None

    def emit(self, event_type: str, data: dict):
        """Emit a progress event."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            **data,
        }
        self.logs.append(event)
        self.events.put_nowait(event)

    def phase_start(self, phase_name: str, phase_number: int):
        self.current_phase = phase_name
        self.emit("phase_start", {
            "phase": phase_name,
            "phase_number": phase_number,
            "total_phases": self.total_phases,
            "progress_pct": int((phase_number - 1) / self.total_phases * 100),
        })

    def phase_complete(self, phase_name: str, phase_number: int, details: dict = None):
        self.phases_completed = phase_number
        self.emit("phase_complete", {
            "phase": phase_name,
            "phase_number": phase_number,
            "total_phases": self.total_phases,
            "progress_pct": int(phase_number / self.total_phases * 100),
            "details": details or {},
        })

    def step_start(self, step_name: str):
        self.current_step = step_name
        self.emit("step_start", {
            "phase": self.current_phase,
            "step": step_name,
        })

    def step_complete(self, step_name: str, details: dict = None):
        self.emit("step_complete", {
            "phase": self.current_phase,
            "step": step_name,
            "details": details or {},
        })

    def log_message(self, message: str, level: str = "info"):
        self.emit("log", {
            "message": message,
            "level": level,
            "phase": self.current_phase,
            "step": self.current_step,
        })

    def pipeline_complete(self, result: dict):
        self.status = "completed"
        self.result = result
        self.emit("pipeline_complete", {
            "progress_pct": 100,
            "summary": result.get("summary", {}),
        })
        # Send sentinel to close SSE stream
        self.events.put_nowait(None)

    def pipeline_failed(self, error: str):
        self.status = "failed"
        self.emit("pipeline_failed", {
            "error": error,
        })
        self.events.put_nowait(None)

    async def event_stream(self) -> AsyncGenerator[str, None]:
        """Generate SSE events for streaming to client."""
        while True:
            event = await self.events.get()
            if event is None:
                # Send final event and close
                yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"


# Global registry of active progress trackers
_progress_trackers: dict[str, PipelineProgress] = {}


def create_progress_tracker(run_id: str) -> PipelineProgress:
    tracker = PipelineProgress(run_id)
    _progress_trackers[run_id] = tracker
    return tracker


def get_progress_tracker(run_id: str) -> PipelineProgress | None:
    return _progress_trackers.get(run_id)


def remove_progress_tracker(run_id: str):
    _progress_trackers.pop(run_id, None)
