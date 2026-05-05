"""FastAPI application entry point for Synthetic Data Generation pipeline.
Orchestrated by LangGraph StateGraph with comprehensive logging.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging_config import setup_logging, get_agent_logger

# Initialize structured logging
setup_logging()
logger = get_agent_logger("Application")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()

    # Ensure directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    logger.set_step("startup")
    logger.info("=" * 60)
    logger.info("Synthetic Data Generation Pipeline - Starting")
    agent_count = 6 if settings.enable_multihop else 5
    logger.info(f"  Orchestrator: LangGraph StateGraph ({agent_count} agents)")
    logger.info(f"  Multi-hop generation: {'ENABLED' if settings.enable_multihop else 'DISABLED'}")
    logger.info(f"  Generation Model: {settings.azure_openai_model_name}")
    logger.info(f"  Generation Endpoint: {settings.azure_openai_endpoint}")
    logger.info(f"  Validation Model: {settings.azure_openai_validation_model_name}")
    logger.info(f"  Validation Endpoint: {settings.azure_openai_validation_endpoint}")
    logger.info(f"  Data dir: {settings.data_dir}")
    logger.info(f"  Cache dir: {settings.cache_dir}")
    logger.info(f"  Question generation: exhaustive (up to {settings.max_questions_per_chunk} per chunk)")
    if settings.enable_multihop:
        logger.info(f"  Multi-hop similarity band: [{settings.multihop_similarity_min}, {settings.multihop_similarity_max}]")
        logger.info(f"  Max multi-hop pairs: {settings.max_multihop_pairs}")
        logger.info(f"  Multi-hop questions per pair: {settings.multihop_questions_per_pair}")
    logger.info(f"  Quality threshold: {settings.quality_threshold}")
    logger.info(f"  Similarity threshold: {settings.similarity_threshold}")
    logger.info(f"  Dedup threshold: {settings.dedup_threshold}")
    logger.info(f"  Max concurrent: {settings.max_concurrent_calls}")
    logger.info("=" * 60)

    yield

    logger.set_step("shutdown")
    logger.info("Synthetic Data Generation Pipeline - Shutting down")


app = FastAPI(
    title="Synthetic Data Generation Pipeline",
    description=(
        "5-Agent pipeline for synthetic chatbot training data generation. "
        "Orchestrated by LangGraph StateGraph with hybrid retrieval (FAISS + BM25), "
        "using LangChain with Azure OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    """Serve the frontend HTML."""
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
