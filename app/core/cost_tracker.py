"""Cost tracking for LLM API calls across the pipeline.

Tracks token usage and estimated costs per agent/phase.
Provides per-run cost summaries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# Approximate pricing (per 1K tokens) — update as needed
MODEL_PRICING = {
    "gpt-5.2": {"input": 0.005, "output": 0.015},
    "gpt-5.4-mini": {"input": 0.0003, "output": 0.0012},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
    # Fallback
    "default": {"input": 0.003, "output": 0.010},
}


@dataclass
class LLMCallRecord:
    """A single LLM API call record."""
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: float = 0.0
    duration_seconds: float = 0.0

    @property
    def estimated_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["default"])
        return (
            (self.input_tokens / 1000) * pricing["input"]
            + (self.output_tokens / 1000) * pricing["output"]
        )


class CostTracker:
    """Thread-safe cost tracker for a pipeline run."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id
        self._records: list[LLMCallRecord] = []
        self._lock = threading.Lock()

    def record_call(
        self, agent: str, model: str,
        input_tokens: int = 0, output_tokens: int = 0,
        duration_seconds: float = 0.0,
    ):
        """Record a single LLM API call."""
        record = LLMCallRecord(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=time.time(),
            duration_seconds=duration_seconds,
        )
        with self._lock:
            self._records.append(record)

    def get_summary(self) -> dict:
        """Get cost summary grouped by agent."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
                "by_agent": {},
            }

        by_agent: dict[str, dict] = {}
        total_cost = 0.0
        total_input = 0
        total_output = 0

        for r in records:
            cost = r.estimated_cost
            total_cost += cost
            total_input += r.input_tokens
            total_output += r.output_tokens

            if r.agent not in by_agent:
                by_agent[r.agent] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "models_used": set(),
                }
            by_agent[r.agent]["calls"] += 1
            by_agent[r.agent]["input_tokens"] += r.input_tokens
            by_agent[r.agent]["output_tokens"] += r.output_tokens
            by_agent[r.agent]["cost_usd"] += cost
            by_agent[r.agent]["models_used"].add(r.model)

        # Convert sets to lists for JSON serialization
        for agent_data in by_agent.values():
            agent_data["models_used"] = sorted(agent_data["models_used"])
            agent_data["cost_usd"] = round(agent_data["cost_usd"], 6)

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_calls": len(records),
            "by_agent": by_agent,
        }


# Global tracker per run (replaced for each pipeline execution)
_active_trackers: dict[str, CostTracker] = {}


def get_cost_tracker(run_id: str) -> CostTracker:
    """Get or create a cost tracker for a run."""
    if run_id not in _active_trackers:
        _active_trackers[run_id] = CostTracker(run_id)
    return _active_trackers[run_id]


def remove_cost_tracker(run_id: str):
    """Remove a cost tracker when the run completes."""
    _active_trackers.pop(run_id, None)
