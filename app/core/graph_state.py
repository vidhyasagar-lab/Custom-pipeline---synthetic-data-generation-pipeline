"""LangGraph state definition for the Chatbot Synthetic Data Generation pipeline.

Pipeline (multi-hop enabled):
    Agent 1: Semantic Chunker → [Agent 2: Question Generator ‖ Agent 2.5: Multi-Hop Generator]
    → Merge → Agent 3: Deduplicator → Agent 4: RAG Answer Generator → Agent 5: Quality Validator
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """State passed between LangGraph nodes in the pipeline."""

    # Run metadata
    run_id: str
    status: str
    started_at: str
    errors: Annotated[list[str], operator.add]

    # Document(s) — supports single or multiple
    document_path: str
    document_paths: list[str]
    raw_text: str

    # Agent 1: Semantic Chunker outputs
    chunks: list[dict]
    chunk_embeddings: list[list[float]]  # Reusable embeddings for chunks (avoids re-embedding)

    # Agent 2: Question Generator outputs
    qa_pairs: list[dict]

    # Agent 2.5: Multi-Hop Generator outputs
    multihop_pairs: list[dict]       # chunk pairs selected for multi-hop
    multihop_qa_pairs: list[dict]    # generated multi-hop QA pairs

    # Agent 3: Deduplicator outputs
    duplicates_removed: int

    # Agent 4: RAG Answer Generator outputs
    qa_triples: list[dict]

    # Agent 5: Quality Validator outputs
    validated_triples: list[dict]
    rejected_triples: list[dict]

    # Knowledge Graph (P1.1)
    knowledge_graph: dict

    # Advanced Metrics (P1.2, P1.3, P2.6, P2.8)
    advanced_metrics_summary: dict

    # Cost tracking (P3.12)
    cost_summary: dict

    # Tracking
    quality: dict
    phase_timings: Annotated[list[dict], operator.add]

    # Config
    config_quality_threshold: float
    config_max_retries: int


def create_initial_state(
    run_id: str,
    document_path: str = "",
    document_paths: list[str] | None = None,
    quality_threshold: float = 0.7,
    max_retries: int = 3,
    **kwargs,
) -> GraphState:
    """Create the initial state for a pipeline run."""
    return GraphState(
        run_id=run_id,
        status="running",
        started_at=datetime.now().isoformat(),
        errors=[],
        document_path=document_path,
        document_paths=document_paths or ([document_path] if document_path else []),
        raw_text="",
        chunks=[],
        chunk_embeddings=[],
        qa_pairs=[],
        multihop_pairs=[],
        multihop_qa_pairs=[],
        qa_triples=[],
        duplicates_removed=0,
        validated_triples=[],
        rejected_triples=[],
        knowledge_graph={},
        advanced_metrics_summary={},
        cost_summary={},
        quality={
            "total_generated": 0,
            "after_dedup": 0,
            "after_validation": 0,
            "rejected": 0,
            "avg_quality_score": 0.0,
        },
        phase_timings=[],
        config_quality_threshold=quality_threshold,
        config_max_retries=max_retries,
    )
