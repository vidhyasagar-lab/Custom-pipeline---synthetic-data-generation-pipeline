"""Data models for the Chatbot Synthetic Data Generation pipeline.

Output format: Conversational training data (ShareGPT/OpenAI style)
with metadata for each conversation including source chunks, retrieval info,
and quality scores.
"""

from __future__ import annotations

import typing as t
from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    """Metadata for a semantic chunk."""
    chunk_id: int
    doc_chunk_id: int = 0
    source_file: str = ""
    doc_index: int = 0
    sentence_start: int = 0
    sentence_end: int = 0
    num_sentences: int = 0
    char_length: int = 0


class RetrievalDetail(BaseModel):
    """Details about a retrieved chunk (hybrid retrieval)."""
    chunk_id: int
    source_file: str = ""
    rrf_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None


class ConversationMessage(BaseModel):
    """A single message in a conversation (OpenAI/ShareGPT format)."""
    role: str  # "user" or "assistant"
    content: str


class ConversationMetadata(BaseModel):
    """Metadata attached to each conversation for traceability."""
    source_chunk_id: int = -1
    source_file: str = ""
    question_type: str = ""
    difficulty: str = ""
    quality_score: float = 0.0
    validation_scores: dict = Field(default_factory=dict)
    retrieved_chunk_ids: list[int] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieval_details: list[RetrievalDetail] = Field(default_factory=list)
    cross_document: bool = False


class Conversation(BaseModel):
    """A single multi-turn conversation for chatbot training."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[ConversationMessage] = Field(default_factory=list)
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)


class QATriple(BaseModel):
    """Internal QA triple (used during pipeline processing)."""
    question: str
    answer: str
    follow_up_q: str = ""
    follow_up_a: str = ""
    context: list[str] = Field(default_factory=list)
    question_type: str = ""
    difficulty: str = ""
    source_chunk_id: int = -1
    retrieved_chunk_ids: list[int] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieval_details: list[dict] = Field(default_factory=list)
    quality_score: float = 0.0
    validation_scores: dict = Field(default_factory=dict)


class PipelineRunRequest(BaseModel):
    document_path: t.Optional[str] = None
    document_paths: list[str] = Field(default_factory=list)
    quality_threshold: float = 0.7
    max_retries: int = 3
    enable_multihop: bool = True
    # New: Persona & diversity controls
    enable_personas: t.Optional[bool] = None
    personas: list[str] = Field(default_factory=list)
    query_styles: list[str] = Field(default_factory=list)
    # New: Advanced metrics & KG
    enable_advanced_metrics: t.Optional[bool] = None
    enable_knowledge_graph: t.Optional[bool] = None
    # New: Experimentation
    experiment_name: str = ""


class PipelineRunResponse(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: t.Optional[datetime] = None
    completed_at: t.Optional[datetime] = None
    conversations: list[Conversation] = Field(default_factory=list)
    qa_triples: list[QATriple] = Field(default_factory=list)
    quality: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
