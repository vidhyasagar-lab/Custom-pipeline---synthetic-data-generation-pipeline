"""Agent 4: Deduplicator (LangGraph Node)
Removes duplicate/near-duplicate QA pairs using FAISS approximate nearest
neighbor search — O(n log n) instead of O(n²) pairwise comparison.

Runs BEFORE answer generation to avoid wasting LLM calls on duplicate questions.
"""

import time
import asyncio

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.llm import get_azure_embeddings
from app.core.logging_config import get_agent_logger
from app.core.graph_state import GraphState

logger = get_agent_logger("Agent4:Deduplicator")


async def deduplicator_node(state: GraphState) -> dict:
    """LangGraph node: Remove semantically duplicate QA pairs using FAISS.
    
    Operates on qa_pairs (pre-answer) to save LLM calls during answer generation.
    """
    try:
        with logger.phase("Deduplication"):
            settings = get_settings()
            threshold = settings.dedup_threshold

            qa_pairs = state.get("qa_pairs", [])

            if not qa_pairs:
                raise ValueError("No QA pairs to deduplicate. Run Question Generator first.")

            logger.info(f"Input: {len(qa_pairs)} QA pairs (pre-answer dedup)")
            logger.info(f"Dedup threshold: {threshold}")

            # --- Step 1: Embed all questions ---
            with logger.step("Embed questions"):
                try:
                    start = time.time()
                    embeddings_model = get_azure_embeddings()
                    questions = [q["question"] for q in qa_pairs]

                    all_embeddings = []
                    batch_size = 50
                    batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
                    
                    # Parallel embedding: up to 3 concurrent API calls
                    concurrent = min(len(batches), 3)
                    for i in range(0, len(batches), concurrent):
                        group = batches[i:i + concurrent]
                        tasks = [embeddings_model.aembed_documents(b) for b in group]
                        try:
                            results = await asyncio.gather(*tasks)
                            for batch_embs in results:
                                all_embeddings.extend(batch_embs)
                            logger.info(f"  Embedded batches {i + 1}-{min(i + concurrent, len(batches))}/{len(batches)}")
                        except Exception as e:
                            logger.error(f"  Embedding batch group failed: {e}")
                            raise RuntimeError(f"Question embedding failed: {e}") from e

                    embed_time = time.time() - start
                    logger.info(f"Embedded {len(questions)} questions in {embed_time:.1f}s")

                except (RuntimeError, ValueError):
                    raise
                except Exception as e:
                    logger.error(f"Embedding step failed: {e}")
                    raise RuntimeError(f"Dedup embedding failed: {e}") from e

            # --- Step 2: FAISS-based duplicate detection (O(n log n)) ---
            with logger.step("Find duplicates (FAISS)"):
                try:
                    start = time.time()
                    emb_array = np.array(all_embeddings, dtype=np.float32)

                    # Normalize for cosine similarity via inner product
                    faiss.normalize_L2(emb_array)

                    dim = emb_array.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    index.add(emb_array)

                    # Search for top-k nearest neighbors per question
                    k = min(10, len(qa_pairs))
                    scores, indices = index.search(emb_array, k)

                    to_remove = set()
                    duplicate_pairs = []

                    for i in range(len(qa_pairs)):
                        if i in to_remove:
                            continue
                        for j_pos in range(1, k):  # Skip self (position 0)
                            j = int(indices[i][j_pos])
                            if j < 0 or j in to_remove or j <= i:
                                continue
                            sim = float(scores[i][j_pos])
                            if sim > threshold:
                                to_remove.add(j)
                                duplicate_pairs.append((i, j, sim))

                    dedup_time = time.time() - start
                    logger.info(f"FAISS dedup done in {dedup_time:.1f}s")
                    logger.info(f"  Duplicate pairs found: {len(duplicate_pairs)}")
                    logger.info(f"  Items to remove: {len(to_remove)}")

                    for idx, (i, j, sim) in enumerate(duplicate_pairs[:5]):
                        logger.info(f"  Example dup {idx + 1}: sim={sim:.3f}")
                        logger.info(f"    Q1: {qa_pairs[i]['question'][:80]}...")
                        logger.info(f"    Q2: {qa_pairs[j]['question'][:80]}...")

                except Exception as e:
                    logger.error(f"FAISS dedup failed: {e}")
                    raise RuntimeError(f"Dedup FAISS search failed: {e}") from e

            # --- Step 3: Filter ---
            with logger.step("Filter duplicates"):
                deduplicated = [q for i, q in enumerate(qa_pairs) if i not in to_remove]
                removed_count = len(qa_pairs) - len(deduplicated)

                logger.info(f"Deduplication complete:")
                logger.info(f"  Before: {len(qa_pairs)}")
                logger.info(f"  After: {len(deduplicated)}")
                logger.info(f"  Removed: {removed_count} ({removed_count / max(len(qa_pairs), 1):.1%})")

            return {
                "qa_pairs": deduplicated,
                "duplicates_removed": removed_count,
                "phase_timings": [{
                    "phase": "deduplication",
                    "method": "FAISS approximate nearest neighbor (pre-answer)",
                    "before": len(qa_pairs),
                    "after": len(deduplicated),
                    "removed": removed_count,
                    "threshold": threshold,
                    "duration_seconds": round(embed_time + dedup_time, 2),
                }],
            }

    except Exception as e:
        logger.error(f"Deduplicator node failed: {e}")
        raise
