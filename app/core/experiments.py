"""Experimentation framework for A/B comparison of pipeline runs.

Stores run configurations and results for systematic comparison.
Enables tracking what changed (prompt, threshold, model) and measuring impact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from uuid import uuid4


@dataclass
class ExperimentConfig:
    """Configuration snapshot for a pipeline run."""
    quality_threshold: float = 0.7
    max_questions_per_chunk: int = 15
    dedup_threshold: float = 0.92
    similarity_threshold: float = 0.75
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    enable_multihop: bool = True
    generation_model: str = ""
    validation_model: str = ""
    personas_enabled: list[str] = field(default_factory=list)
    query_styles: list[str] = field(default_factory=list)
    custom_notes: str = ""


@dataclass
class ExperimentResult:
    """Result metrics from a pipeline run."""
    total_chunks: int = 0
    total_questions_generated: int = 0
    after_dedup: int = 0
    after_validation: int = 0
    rejected: int = 0
    avg_quality_score: float = 0.0
    criteria_averages: dict = field(default_factory=dict)
    # Advanced metrics averages
    avg_claim_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    # Cost
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    # Timing
    total_time_seconds: float = 0.0


@dataclass
class Experiment:
    """A single experiment run with config + results."""
    experiment_id: str = field(default_factory=lambda: str(uuid4())[:8])
    run_id: str = ""
    name: str = ""
    timestamp: float = field(default_factory=time.time)
    config: ExperimentConfig = field(default_factory=ExperimentConfig)
    result: ExperimentResult = field(default_factory=ExperimentResult)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentStore:
    """Persistent experiment storage (JSON file-based)."""

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or Path("output/experiments.json")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._experiments: dict[str, Experiment] = {}
        self._load()

    def _load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = Experiment(
                        experiment_id=exp_data.get("experiment_id", ""),
                        run_id=exp_data.get("run_id", ""),
                        name=exp_data.get("name", ""),
                        timestamp=exp_data.get("timestamp", 0),
                        config=ExperimentConfig(**exp_data.get("config", {})),
                        result=ExperimentResult(**exp_data.get("result", {})),
                        tags=exp_data.get("tags", []),
                    )
                    self._experiments[exp.experiment_id] = exp
            except Exception:
                self._experiments = {}

    def _save(self):
        data = {
            "experiments": [e.to_dict() for e in self._experiments.values()]
        }
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def record(self, experiment: Experiment):
        """Record an experiment result."""
        self._experiments[experiment.experiment_id] = experiment
        self._save()

    def list_experiments(self) -> list[dict]:
        """List all experiments (most recent first)."""
        exps = sorted(self._experiments.values(), key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in exps]

    def get_experiment(self, experiment_id: str) -> dict | None:
        exp = self._experiments.get(experiment_id)
        return exp.to_dict() if exp else None

    def compare(self, id_a: str, id_b: str) -> dict | None:
        """Compare two experiments side by side."""
        a = self._experiments.get(id_a)
        b = self._experiments.get(id_b)
        if not a or not b:
            return None

        def _diff(va, vb):
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                return {"a": va, "b": vb, "delta": round(vb - va, 4)}
            return {"a": va, "b": vb}

        result_a = asdict(a.result)
        result_b = asdict(b.result)

        comparison = {
            "experiment_a": {"id": a.experiment_id, "name": a.name, "run_id": a.run_id},
            "experiment_b": {"id": b.experiment_id, "name": b.name, "run_id": b.run_id},
            "config_diff": {},
            "result_diff": {},
        }

        # Config diff
        config_a = asdict(a.config)
        config_b = asdict(b.config)
        for key in config_a:
            if config_a[key] != config_b.get(key):
                comparison["config_diff"][key] = {"a": config_a[key], "b": config_b.get(key)}

        # Result diff
        for key in result_a:
            comparison["result_diff"][key] = _diff(result_a[key], result_b.get(key))

        return comparison

    def delete(self, experiment_id: str) -> bool:
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]
            self._save()
            return True
        return False
