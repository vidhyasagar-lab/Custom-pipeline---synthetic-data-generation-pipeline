# SynthGen Studio

**Agentic pipeline for generating high-quality synthetic chatbot training data from enterprise documents.**

Built with LangGraph + LangChain + Azure OpenAI. Processes `.docx` and `.pdf` documents through a multi-agent pipeline that chunks, generates diverse questions, retrieves context via hybrid search, produces grounded answers, and validates quality — outputting production-ready conversational training data in ShareGPT/OpenAI format.

---

## Table of Contents

- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Agent Details](#agent-details)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Output Format](#output-format)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Supervisor (LangGraph StateGraph)            │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  Agent 1      │──▶│  KG Builder  │──▶│  Agent 2             │    │
│  │  Semantic     │   │  (Optional)  │   │  Question Generator  │    │
│  │  Chunker      │   │              │   │  + Personas          │    │
│  └──────────────┘   └──────────────┘   └──────────┬───────────┘    │
│                                                    │                │
│                                          ┌─────────▼─────────┐     │
│                                          │  Agent 2.5         │     │
│                                          │  Multi-Hop Gen     │     │
│                                          │  (Optional)        │     │
│                                          └─────────┬─────────┘     │
│                                                    │                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────▼───────────┐    │
│  │  Agent 5      │◀──│  Agent 4     │◀──│  Agent 3             │    │
│  │  Quality      │   │  RAG Answer  │   │  Deduplicator        │    │
│  │  Validator    │   │  Generator   │   │  (FAISS)             │    │
│  └──────────────┘   └──────────────┘   └──────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**7 configurable agents** orchestrated by a LangGraph StateGraph. The pipeline is fully async, supports multiple documents, and provides real-time progress via SSE.

---

## Pipeline Flow

```
START → Semantic Chunker → [Knowledge Graph Builder] → Question Generator
      → [Multi-Hop Generator] → Deduplicator → Answer Generator
      → Quality Validator → END
```

Nodes in `[ ]` are optional and toggled via configuration. Deduplication runs **before** answer generation to avoid wasting LLM calls on duplicate questions.

| Phase | Agent | What Happens |
|-------|-------|-------------|
| 1 | **Semantic Chunker** | Loads documents, splits into semantically coherent chunks using embedding similarity breaks, extracts heading hierarchy |
| 2 | **Knowledge Graph Builder** | Extracts entities/keyphrases/topics from each chunk via LLM, builds relationship edges using Jaccard similarity |
| 3 | **Question Generator** | Generates up to 15 diverse questions per chunk across 6 types, 3 difficulties, 5 personas, 3 styles. Two-stage: generate → noise filter |
| 4 | **Multi-Hop Generator** | Finds related-but-distinct chunk pairs (FAISS similarity band or KG edges), extracts bridge concepts, generates synthesis questions |
| 5 | **Deduplicator** | Embeds all questions, FAISS nearest-neighbor search, removes duplicates above threshold (default 0.92) |
| 6 | **Answer Generator** | Hybrid retrieval (FAISS dense + BM25 sparse + RRF fusion), generates grounded answers constrained to retrieved context |
| 7 | **Quality Validator** | LLM-as-Judge scoring on 5-6 criteria with rubric examples, borderline retry, optional RAGAS-inspired advanced metrics |

---

## Agent Details

### Agent 1: Semantic Chunker

Splits documents at **semantic boundaries** rather than fixed character counts.

- Embeds all sentences using Azure OpenAI `text-embedding-ada-002`
- Computes cosine similarity between consecutive sentence pairs
- Creates a chunk boundary when similarity drops below threshold (default `0.75`)
- Enforces min (200 chars) and max (3000 chars) chunk sizes
- Extracts heading hierarchy from `.docx` files (Heading 1/2/3 styles)
- **SHA-256 content-hash caching** — identical files skip re-processing entirely

### Knowledge Graph Builder (Optional)

Builds a semantic relationship graph over chunks for smarter multi-hop pair selection.

- **Node properties** (LLM-extracted per chunk): named entities, keyphrases, topics
- **Edge types**: `shared_entity`, `shared_keyphrase`, `hierarchical`
- **Edge strength**: Jaccard similarity — `J(A, B) = |A ∩ B| / |A ∪ B|`
- Serialized to dict and passed through LangGraph state
- Supports `from_dict()` reconstruction for downstream use

### Agent 2: Question Generator

Exhaustive question generation with persona-driven diversity.

- **6 question types**: factual, explanatory, procedural, comparative, inferential, scenario-based
- **3 difficulty levels**: simple, moderate, complex
- **5 personas**: beginner, expert, impatient, curious student, non-native speaker
- **3 query styles**: conversational, web search, formal
- **3 query lengths**: short (<10 words), medium (10-20), long (20-40)
- Each chunk gets a randomly assigned persona/style/length scenario injected into the generation prompt
- Two-stage pipeline: exhaustive generation → noise filter (removes vague, generic, self-referencing questions)
- Follow-up Q&A pairs generated for multi-turn conversations

### Agent 2.5: Multi-Hop Generator (Optional)

Generates questions that require synthesizing information from **two different chunks**.

- **Pair selection**: FAISS similarity band `[0.4, 0.82]` or KG relationship strength band
  - Below 0.4 = unrelated, above 0.82 = too similar (same topic)
  - Cross-document pairs prioritized
- **Bridge concept extraction**: LLM identifies shared entities, themes, or causal links between two passages
- **Question generation**: Questions must require BOTH passages to answer fully
- Bridge types: `shared_entity`, `complementary`, `causal`, `comparative`

### Agent 3: Deduplicator

FAISS-based semantic deduplication **before** answer generation.

- Embeds all question texts
- Finds top-10 nearest neighbors per question
- Marks pairs with similarity > `0.92` as duplicates
- **O(n log n)** complexity via FAISS indexing vs O(n²) naive approach

### Agent 4: RAG Answer Generator

Generates context-grounded answers using hybrid retrieval.

- **Dense retrieval**: FAISS `IndexFlatIP` with normalized vectors (cosine similarity)
  - Reuses chunk embeddings from Agent 1 (no re-embedding)
- **Sparse retrieval**: BM25Okapi with NLTK tokenization + query term expansion
- **Fusion**: Reciprocal Rank Fusion (RRF) — `score = Σ (w / (k + rank))`
  - Default weights: dense `0.6`, sparse `0.4`, k=`60`
- Retrieves top-5 chunks, builds context with source attribution
- LLM generates answer constrained to **only** use retrieved context
- Tracks retrieved chunk IDs, RRF scores, and source files per answer

### Agent 5: Quality Validator

LLM-as-Judge with rubric-graded scoring using a **separate validation model** to avoid self-evaluation bias.

**Standard QA scoring** (5 criteria, weighted sum):

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Faithfulness | 0.30 | Answer grounded in context (not hallucinated) |
| Relevance | 0.25 | Answer addresses the question |
| Completeness | 0.20 | Covers key points from context |
| Tone | 0.15 | Sounds like a helpful chatbot |
| Follow-up Quality | 0.10 | Follow-up is natural and answerable |

**Multi-hop QA** adds a 6th criterion:

| Criterion | Weight | What It Measures |
|-----------|--------|-----------------|
| Multi-hop Validity | 0.20 | Genuinely synthesizes from both passages |

- Overall score computed **server-side** (not by LLM)
- Acceptance threshold: `0.7` (configurable)
- **Borderline retry**: triples scoring `[threshold - 0.15, threshold)` are re-evaluated
- Each criterion has rubric examples at 0.0, 0.2, 0.5, 0.8, 1.0 score levels

**Advanced metrics** (optional, sampled on 30% of validated triples):
- Claim-based faithfulness (decompose answer → verify each claim)
- Context precision (% of retrieved chunks actually relevant)
- Context recall (% of answer statements attributable to context)
- Answer relevancy (reverse question generation + similarity)
- Noise sensitivity (inject irrelevant chunk, measure answer drift)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Semantic chunking** | Embedding-based boundary detection instead of fixed character splits |
| **Hybrid retrieval** | FAISS + BM25 + RRF fusion for both semantic and keyword matching |
| **Persona diversity** | 5 personas × 3 styles × 3 lengths = 45 query scenario combinations |
| **Multi-hop questions** | Cross-chunk synthesis questions with explicit bridge concept extraction |
| **Knowledge graph** | Entity/keyphrase relationship graph for smarter pair selection |
| **FAISS dedup** | Semantic deduplication before answer generation saves LLM cost |
| **Rubric-graded validation** | LLM-as-Judge with score-level examples, borderline retry |
| **Separate validation model** | Generation and validation use different models to avoid self-bias |
| **Chunk caching** | SHA-256 content-hash caching skips re-processing identical files |
| **Embedding reuse** | Chunks embedded once, reused across dedup, retrieval, and multi-hop |
| **Multi-document** | Process multiple documents in one run with cross-document Q&A |
| **Real-time progress** | SSE streaming with per-phase progress, live logs |
| **Experiment tracking** | A/B comparison of pipeline runs with config snapshots |
| **Export** | JSON and Excel export of results |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph StateGraph |
| LLM Framework | LangChain + Azure OpenAI |
| Dense Retrieval | FAISS (`faiss-cpu`) |
| Sparse Retrieval | BM25Okapi (`rank-bm25`) |
| Embeddings | Azure OpenAI `text-embedding-ada-002` |
| NLP | NLTK (sentence tokenization) |
| Document Parsing | `python-docx`, PyMuPDF |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS SPA |
| Configuration | Pydantic Settings + `.env` |
| Package Management | UV |

---

## Project Structure

```
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── agents/
│   │   ├── supervisor.py          # LangGraph orchestrator
│   │   ├── document_processor.py  # Agent 1: Semantic chunking
│   │   ├── knowledge_graph.py     # KG builder (optional)
│   │   ├── question_generator.py  # Agent 2: Question generation
│   │   ├── multihop_generator.py  # Agent 2.5: Multi-hop questions
│   │   ├── deduplicator.py        # Agent 3: FAISS deduplication
│   │   ├── answer_generator.py    # Agent 4: Hybrid RAG answers
│   │   ├── quality_validator.py   # Agent 5: LLM-as-Judge
│   │   └── advanced_metrics.py    # RAGAS-inspired metrics
│   ├── api/
│   │   └── routes.py              # REST API endpoints
│   ├── core/
│   │   ├── config.py              # Pydantic settings from .env
│   │   ├── graph_state.py         # LangGraph TypedDict state
│   │   ├── llm.py                 # Cached LLM factory functions
│   │   ├── logging_config.py      # Structured agent-aware logging
│   │   ├── progress.py            # SSE progress tracking
│   │   ├── personas.py            # Persona/style/distribution
│   │   ├── cost_tracker.py        # Token usage cost tracking
│   │   └── experiments.py         # Experiment A/B framework
│   └── models/
│       ├── schemas.py             # API request/response models
│       └── llm_responses.py       # Pydantic LLM output validation
├── frontend/
│   ├── index.html                 # SPA entry point
│   ├── styles.css                 # UI styles
│   └── app.js                     # Frontend logic
├── data/                          # Drop documents here
├── cache/                         # Chunk/embedding cache
├── output/                        # Experiment results
├── start.py                       # One-click start script
├── pyproject.toml                 # Dependencies (UV)
└── .env                           # Environment variables
```

---

## Setup

### Prerequisites

- **Python 3.13+**
- **UV** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Azure OpenAI** access with deployed models

### Installation

```bash
# Clone and enter the project
cd "Synthetic Data Generation"

# Install dependencies with UV
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Running

```bash
# Option 1: One-click start
python start.py

# Option 2: Direct uvicorn
uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# Option 3: Custom port
python start.py --port 9000
```

Open **http://localhost:8080** in your browser.

---

## Configuration

### Required Environment Variables

```env
# Generation Model (Azure OpenAI)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_GPT54MINI_MODEL_NAME=gpt-4o
AZURE_OPENAI_GPT54MINI_DEPLOYMENT_NAME=gpt-4o

# Validation Model (separate endpoint recommended)
AZURE_OPENAI_VALIDATION_ENDPOINT=https://your-validation-endpoint.openai.azure.com/
AZURE_OPENAI_VALIDATION_API_KEY=your-validation-key
AZURE_OPENAI_VALIDATION_API_VERSION=2025-04-01-preview
AZURE_OPENAI_VALIDATION_MODEL_NAME=gpt-4o-mini
AZURE_OPENAI_VALIDATION_DEPLOYMENT_NAME=gpt-4o-mini

# Embedding Model
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.openai.azure.com/
AZURE_OPENAI_EMBEDDING_API_KEY=your-embedding-key
AZURE_OPENAI_EMBEDDING_API_VERSION=2025-01-01-preview
AZURE_OPENAI_EMBEDDING_MODEL_NAME=text-embedding-ada-002
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
```

### Pipeline Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `QUALITY_THRESHOLD` | `0.7` | Minimum score to accept a QA triple (0.0–1.0) |
| `MAX_QUESTIONS_PER_CHUNK` | `15` | Max questions generated per chunk |
| `SIMILARITY_THRESHOLD` | `0.75` | Semantic chunking break point (lower = more chunks) |
| `DEDUP_THRESHOLD` | `0.92` | Question similarity to consider duplicate |
| `DENSE_WEIGHT` | `0.6` | RRF weight for FAISS dense retrieval |
| `SPARSE_WEIGHT` | `0.4` | RRF weight for BM25 sparse retrieval |
| `MAX_CONCURRENT_CALLS` | `5` | Parallel LLM calls per batch |

### Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MULTIHOP` | `true` | Enable multi-hop question generation |
| `ENABLE_KNOWLEDGE_GRAPH` | `true` | Build KG for multi-hop pair selection |
| `ENABLE_ADVANCED_METRICS` | `true` | Compute RAGAS-inspired metrics on validated triples |
| `ENABLE_PERSONAS` | `true` | Inject persona/style/length diversity into questions |
| `ENABLE_COST_TRACKING` | `true` | Track token usage and estimated costs |

### Multi-Hop Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTIHOP_SIMILARITY_MIN` | `0.4` | Lower bound of similarity band |
| `MULTIHOP_SIMILARITY_MAX` | `0.82` | Upper bound of similarity band |
| `MAX_MULTIHOP_PAIRS` | `50` | Max chunk pairs to process |
| `MULTIHOP_QUESTIONS_PER_PAIR` | `3` | Questions generated per pair |

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate` | Run pipeline synchronously (blocks until complete) |
| `POST` | `/generate/async` | Start pipeline asynchronously, returns `run_id` |
| `GET` | `/status/{run_id}` | Get status and results of a pipeline run |
| `GET` | `/results/{run_id}/export?format=json\|excel` | Export results as JSON or Excel |
| `GET` | `/progress/{run_id}` | SSE stream of real-time progress events |
| `GET` | `/progress/{run_id}/logs` | Get accumulated progress logs |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-document` | Upload a single document |
| `POST` | `/upload-documents` | Upload multiple documents |

### Experiments

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/experiments` | List all experiment runs |
| `GET` | `/experiments/{id}` | Get experiment details |
| `GET` | `/experiments/compare/{id_a}/{id_b}` | Compare two experiments |
| `DELETE` | `/experiments/{id}` | Delete an experiment |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with config summary |
| `GET` | `/graph-info` | Pipeline graph structure info |
| `DELETE` | `/cache` | Clear chunk/embedding cache |

### Example: Generate Request

```bash
curl -X POST http://localhost:8080/api/v1/generate/async \
  -H "Content-Type: application/json" \
  -d '{
    "document_paths": ["data/document.docx"],
    "quality_threshold": 0.7,
    "max_retries": 3,
    "enable_multihop": true,
    "enable_knowledge_graph": true,
    "enable_advanced_metrics": true,
    "enable_personas": true,
    "experiment_name": "baseline-run"
  }'
```

---

## Frontend

The SynthGen Studio UI is a single-page application served at `http://localhost:8080`.

**Pages:**

| Page | Description |
|------|-------------|
| **Dashboard** | Overview metrics, recent runs, system health |
| **Generate** | Upload documents, configure pipeline, start runs with real-time progress |
| **Results** | Browse completed runs, view conversations, export data |
| **Experiments** | Compare pipeline configurations and results side by side |

**Features:**
- Drag-and-drop document upload
- Real-time pipeline progress with phase tracking and live logs
- Toggle multi-hop, KG, personas, and advanced metrics from the UI
- JSON/Excel export of results

---

## Output Format

The pipeline outputs conversational training data in **ShareGPT/OpenAI format**:

```json
{
  "conversations": [
    {
      "id": "e31002f6-0",
      "messages": [
        {"role": "user", "content": "What are the primary indications for this drug?"},
        {"role": "assistant", "content": "The primary indications include..."},
        {"role": "user", "content": "Are there any age-related dosage adjustments?"},
        {"role": "assistant", "content": "Yes, for patients over 65..."}
      ],
      "metadata": {
        "source_file": "Document_1_Drug_Product_Monograph.docx",
        "question_type": "factual",
        "difficulty": "moderate",
        "quality_score": 0.87,
        "validation_scores": {
          "faithfulness": 0.95,
          "relevance": 0.90,
          "completeness": 0.85,
          "tone": 0.80,
          "follow_up_quality": 0.75
        },
        "retrieved_sources": ["Document_1_Drug_Product_Monograph.docx"],
        "cross_document": false,
        "hop_count": 1
      }
    }
  ],
  "quality": {
    "total_generated": 150,
    "after_dedup": 132,
    "after_validation": 98,
    "rejected": 34,
    "avg_quality_score": 0.82
  },
  "summary": {
    "documents_processed": 2,
    "total_chunks": 45,
    "total_conversations": 98,
    "multihop_conversations": 12,
    "cross_document_conversations": 8
  }
}
```

Each conversation includes full provenance: source chunks, retrieval details, quality scores, and question metadata — enabling filtering and analysis downstream.

---

## License

Private / Internal Use
