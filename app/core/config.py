"""Core configuration for the Synthetic Data Generation pipeline."""

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import field_validator
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Azure OpenAI (Generation model)
    openai_api_type: str = "azure"
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_model_name: str = os.getenv("AZURE_OPENAI_GPT54MINI_MODEL_NAME", "gpt-5.2")
    azure_openai_deployment_name: str = os.getenv("AZURE_OPENAI_GPT54MINI_DEPLOYMENT_NAME", "gpt-5.2")

    # Azure OpenAI (Validation model — separate model to avoid self-evaluation bias)
    azure_openai_validation_endpoint: str = os.getenv("AZURE_OPENAI_VALIDATION_ENDPOINT", "")
    azure_openai_validation_api_key: str = os.getenv("AZURE_OPENAI_VALIDATION_API_KEY", "")
    azure_openai_validation_api_version: str = os.getenv("AZURE_OPENAI_VALIDATION_API_VERSION", "2025-04-01-preview")
    azure_openai_validation_model_name: str = os.getenv("AZURE_OPENAI_VALIDATION_MODEL_NAME", "gpt-5.4-mini")
    azure_openai_validation_deployment_name: str = os.getenv("AZURE_OPENAI_VALIDATION_DEPLOYMENT_NAME", "gpt-5.4-mini")

    # Embedding Model
    azure_openai_embedding_endpoint: str = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", "")
    azure_openai_embedding_api_key: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY", "")
    azure_openai_embedding_api_version: str = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2025-01-01-preview")
    azure_openai_embedding_model_name: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
    azure_openai_embedding_deployment_name: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

    # Pipeline Configuration
    quality_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.7"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "2000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "300"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    max_concurrent_calls: int = int(os.getenv("MAX_CONCURRENT_CALLS", "5"))

    # Tunable thresholds (previously hardcoded)
    max_questions_per_chunk: int = int(os.getenv("MAX_QUESTIONS_PER_CHUNK", "15"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
    dedup_threshold: float = float(os.getenv("DEDUP_THRESHOLD", "0.92"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.6"))
    sparse_weight: float = float(os.getenv("SPARSE_WEIGHT", "0.4"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

    # Multi-Hop Configuration
    enable_multihop: bool = os.getenv("ENABLE_MULTIHOP", "true").lower() in ("true", "1", "yes")
    multihop_similarity_min: float = float(os.getenv("MULTIHOP_SIMILARITY_MIN", "0.4"))
    multihop_similarity_max: float = float(os.getenv("MULTIHOP_SIMILARITY_MAX", "0.82"))
    max_multihop_pairs: int = int(os.getenv("MAX_MULTIHOP_PAIRS", "50"))
    multihop_questions_per_pair: int = int(os.getenv("MULTIHOP_QUESTIONS_PER_PAIR", "3"))

    # Knowledge Graph
    enable_knowledge_graph: bool = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() in ("true", "1", "yes")

    # Advanced Metrics
    enable_advanced_metrics: bool = os.getenv("ENABLE_ADVANCED_METRICS", "true").lower() in ("true", "1", "yes")
    noise_sensitivity_sample_rate: float = float(os.getenv("NOISE_SENSITIVITY_SAMPLE_RATE", "0.1"))

    # Personas & Query Diversity
    enable_personas: bool = os.getenv("ENABLE_PERSONAS", "true").lower() in ("true", "1", "yes")
    active_personas: list[str] = os.getenv("ACTIVE_PERSONAS", "beginner,expert,impatient,curious_student,non_native_speaker").split(",")
    active_query_styles: list[str] = os.getenv("ACTIVE_QUERY_STYLES", "conversational,web_search,formal").split(",")
    active_query_lengths: list[str] = os.getenv("ACTIVE_QUERY_LENGTHS", "short,medium,long").split(",")

    # Cost Tracking
    enable_cost_tracking: bool = os.getenv("ENABLE_COST_TRACKING", "true").lower() in ("true", "1", "yes")

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    cache_dir: Path = base_dir / "cache"

    @field_validator("quality_threshold")
    @classmethod
    def validate_quality_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("quality_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("max_concurrent_calls")
    @classmethod
    def validate_max_concurrent(cls, v):
        if v < 1:
            raise ValueError("max_concurrent_calls must be >= 1")
        return v

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
