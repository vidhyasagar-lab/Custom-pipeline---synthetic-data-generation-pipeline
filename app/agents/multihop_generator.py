"""Agent 2.5: Multi-Hop Question Generator (LangGraph Node)
Generates cross-chunk multi-hop questions that require synthesizing
information from TWO different passages to answer.

Uses FAISS to find semantically related (but not identical) chunk pairs,
then uses LLM to identify bridge concepts and generate multi-hop questions.
"""

import json
import time
import asyncio

import numpy as np
import faiss

from app.core.config import get_settings
from app.core.llm import get_question_llm
from app.core.logging_config import get_agent_logger
from app.core.graph_state import GraphState
from app.agents.knowledge_graph import KnowledgeGraph

logger = get_agent_logger("Agent2.5:MultiHop")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

BRIDGE_CONCEPT_PROMPT = """You are analyzing two text passages to find bridge concepts — shared entities, themes, or ideas that connect them.

PASSAGE A:
{chunk_a}

PASSAGE B:
{chunk_b}

Identify 1-3 bridge concepts that connect these two passages. A bridge concept is:
- A shared entity (person, organization, process, term) mentioned in BOTH passages
- A shared theme or topic that both passages discuss from different angles
- A relationship where one passage provides context or detail that complements the other

If no meaningful bridge concepts exist, respond with an empty array.

Respond ONLY with a JSON object:
{{
  "bridge_concepts": ["concept1", "concept2"],
  "connection_type": "shared_entity|complementary|causal|comparative|none",
  "reasoning": "brief explanation of how the passages connect"
}}

Return ONLY valid JSON, no other text."""

MULTIHOP_QUESTION_PROMPT = """You are generating multi-hop training data for a customer-facing chatbot. Multi-hop questions require combining information from TWO different passages to answer fully.

CRITICAL CONSTRAINT: You MUST generate questions ONLY from the information in the two passages below. Do NOT use your own knowledge. Every question MUST require information from BOTH passages to answer completely.

<1-hop>
PASSAGE A:
{chunk_a}
</1-hop>

<2-hop>
PASSAGE B:
{chunk_b}
</2-hop>

BRIDGE CONCEPTS connecting these passages: {bridge_concepts}
CONNECTION TYPE: {connection_type}

RULES:
1. Generate up to {max_questions} multi-hop questions that REQUIRE combining facts from BOTH passages.
2. Each question must be unanswerable from either passage alone — it MUST need both.
3. Questions must sound like a real person typing into a chatbot (casual, natural language).
4. NEVER reference "the text", "passage A/B", "the document", or any source material.
5. For each question, provide a brief reasoning path showing which info comes from which passage.
6. For each question, generate a natural follow-up.

Question types for multi-hop:
- "comparison": Comparing aspects described across the two passages
- "synthesis": Combining facts from both to form a complete picture
- "causal_chain": A causes B (from passage A), B leads to C (from passage B)
- "bridge_reasoning": Using a shared concept to connect different facts

Respond ONLY with a JSON array:
[
  {{
    "question": "natural conversational question requiring both passages",
    "follow_up_q": "natural follow-up after getting the answer",
    "question_type": "multihop",
    "reasoning_type": "comparison|synthesis|causal_chain|bridge_reasoning",
    "difficulty": "complex",
    "hop_reasoning": "Passage A provides X, Passage B provides Y, combining them answers the question"
  }}
]

Return ONLY valid JSON array."""


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK PAIR SELECTION (FAISS similarity band)
# ═══════════════════════════════════════════════════════════════════════════════

def find_multihop_chunk_pairs(
    chunks: list[dict],
    chunk_embeddings: list[list[float]],
    similarity_min: float,
    similarity_max: float,
    max_pairs: int,
) -> list[tuple[int, int, float]]:
    """Find chunk pairs within the similarity band [min, max] using FAISS.

    Chunks that are too similar (>max) are likely about the same topic.
    Chunks that are too dissimilar (<min) likely have no meaningful connection.
    The sweet spot produces related-but-different pairs ideal for multi-hop.

    Returns list of (idx_a, idx_b, similarity_score) tuples.
    """
    if len(chunks) < 2 or not chunk_embeddings:
        return []

    embeddings = np.array(chunk_embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    n = len(embeddings)
    dim = embeddings.shape[1]

    # Use FAISS for efficient similarity search
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Search for top-k neighbors per chunk
    k = min(n, 20)
    scores, indices = index.search(embeddings, k)

    # Collect pairs in similarity band
    pairs = []
    seen = set()

    for i in range(n):
        for j_pos in range(k):
            j = int(indices[i][j_pos])
            sim = float(scores[i][j_pos])

            if i == j:
                continue

            pair_key = (min(i, j), max(i, j))
            if pair_key in seen:
                continue

            if similarity_min <= sim <= similarity_max:
                # Skip pairs from the same document section to get true cross-document hops
                doc_a = chunks[i].get("metadata", {}).get("source_file", "")
                doc_b = chunks[j].get("metadata", {}).get("source_file", "")
                # Prefer cross-document pairs, but allow same-document if different sections
                priority = 0 if doc_a != doc_b else 1

                pairs.append((i, j, sim, priority))
                seen.add(pair_key)

    # Sort: cross-document first, then by similarity (descending within band)
    pairs.sort(key=lambda x: (x[3], -x[2]))

    # Return top max_pairs
    return [(i, j, sim) for i, j, sim, _ in pairs[:max_pairs]]


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE CONCEPT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_bridge_concepts(
    llm,
    chunk_a: dict,
    chunk_b: dict,
    max_retries: int = 2,
) -> dict | None:
    """Extract bridge concepts connecting two chunks using LLM."""
    text_a = chunk_a["page_content"][:3000]
    text_b = chunk_b["page_content"][:3000]

    prompt = BRIDGE_CONCEPT_PROMPT.format(chunk_a=text_a, chunk_b=text_b)

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            result = json.loads(content)

            if not isinstance(result, dict):
                raise ValueError("Response is not a JSON object")

            bridge = result.get("bridge_concepts", [])
            conn_type = result.get("connection_type", "none")

            if conn_type == "none" or not bridge:
                return None

            return result

        except (json.JSONDecodeError, ValueError):
            if attempt < max_retries:
                await asyncio.sleep(1)
                continue
            return None
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"Bridge extraction failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HOP QUESTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_multihop_questions(
    llm,
    chunk_a: dict,
    chunk_b: dict,
    bridge_info: dict,
    max_questions: int = 3,
    max_retries: int = 2,
) -> list[dict]:
    """Generate multi-hop questions for a chunk pair given bridge concepts."""
    text_a = chunk_a["page_content"][:3000]
    text_b = chunk_b["page_content"][:3000]

    bridge_concepts = ", ".join(bridge_info.get("bridge_concepts", []))
    connection_type = bridge_info.get("connection_type", "complementary")

    prompt = MULTIHOP_QUESTION_PROMPT.format(
        chunk_a=text_a,
        chunk_b=text_b,
        bridge_concepts=bridge_concepts,
        connection_type=connection_type,
        max_questions=max_questions,
    )

    id_a = chunk_a.get("metadata", {}).get("chunk_id", "?")
    id_b = chunk_b.get("metadata", {}).get("chunk_id", "?")

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            questions = json.loads(content)

            if not isinstance(questions, list):
                raise ValueError("Response is not a JSON array")

            valid = []
            for q in questions[:max_questions]:
                if not isinstance(q, dict):
                    continue
                question_text = q.get("question", "").strip()
                if not question_text or len(question_text) < 15:
                    continue

                # Skip questions referencing documents/passages
                lower_q = question_text.lower()
                if any(ref in lower_q for ref in [
                    "passage a", "passage b", "the text", "the document",
                    "the passage", "the chunk", "according to",
                ]):
                    continue

                q["question_type"] = "multihop"
                q["source_chunk_ids"] = [id_a, id_b]
                q["source_chunks"] = [text_a[:500], text_b[:500]]
                q["hop_count"] = 2
                q["bridge_concepts"] = bridge_info.get("bridge_concepts", [])
                q["connection_type"] = connection_type
                valid.append(q)

            logger.info(f"  Pair ({id_a}, {id_b}): {len(valid)} multi-hop questions")
            return valid

        except (json.JSONDecodeError, ValueError):
            if attempt < max_retries:
                await asyncio.sleep(1)
                continue
            logger.warning(f"  Pair ({id_a}, {id_b}): Parse failed after retries")
            return []
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"  Pair ({id_a}, {id_b}): Failed: {e}")
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE: PROCESS A SINGLE PAIR
# ═══════════════════════════════════════════════════════════════════════════════

async def _process_single_pair(
    llm,
    chunk_a: dict,
    chunk_b: dict,
    similarity: float,
    max_questions: int,
) -> tuple[dict | None, list[dict]]:
    """Process a single chunk pair: extract bridges then generate questions."""
    id_a = chunk_a.get("metadata", {}).get("chunk_id", "?")
    id_b = chunk_b.get("metadata", {}).get("chunk_id", "?")

    # Step 1: Extract bridge concepts
    bridge_info = await extract_bridge_concepts(llm, chunk_a, chunk_b)

    if bridge_info is None:
        logger.info(f"  Pair ({id_a}, {id_b}): No bridge concepts found, skipping")
        return None, []

    # Step 2: Generate multi-hop questions
    questions = await generate_multihop_questions(
        llm, chunk_a, chunk_b, bridge_info, max_questions=max_questions,
    )

    pair_record = {
        "chunk_id_a": id_a,
        "chunk_id_b": id_b,
        "similarity": similarity,
        "bridge_concepts": bridge_info.get("bridge_concepts", []),
        "connection_type": bridge_info.get("connection_type", ""),
        "questions_generated": len(questions),
    }

    return pair_record, questions


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODE
# ═══════════════════════════════════════════════════════════════════════════════

async def multihop_generator_node(state: GraphState) -> dict:
    """LangGraph node: Generate multi-hop questions from cross-chunk pairs.

    Inserted between question_generator and answer_generator.
    Multi-hop QA pairs are merged into the main qa_pairs list.
    """
    try:
        with logger.phase("Multi-Hop Question Generation"):

            settings = get_settings()
            chunks = state.get("chunks", [])
            chunk_embeddings = state.get("chunk_embeddings", [])
            existing_qa = state.get("qa_pairs", [])

            if len(chunks) < 2:
                logger.info("Less than 2 chunks — skipping multi-hop generation")
                return {
                    "multihop_pairs": [],
                    "multihop_qa_pairs": [],
                    "phase_timings": [{"phase": "multihop_generator", "duration": 0.0}],
                }

            if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
                logger.warning("Chunk embeddings missing or mismatched — skipping multi-hop")
                return {
                    "multihop_pairs": [],
                    "multihop_qa_pairs": [],
                    "phase_timings": [{"phase": "multihop_generator", "duration": 0.0}],
                }

            # --- Step 1: Find chunk pairs ---
            with logger.step("Find chunk pairs"):
                start = time.time()
                kg_data = state.get("knowledge_graph", {})

                if kg_data and settings.enable_knowledge_graph:
                    # Use Knowledge Graph for semantically richer pair selection
                    logger.info("Using Knowledge Graph for pair selection")
                    kg = KnowledgeGraph.from_dict(kg_data)
                    kg_pairs = kg.get_multi_hop_pairs(
                        min_strength=settings.multihop_similarity_min,
                        max_strength=settings.multihop_similarity_max,
                        max_pairs=settings.max_multihop_pairs,
                        prefer_cross_document=True,
                    )
                    # Convert KG pairs to index-based pairs
                    chunk_id_to_idx = {c.get("metadata", {}).get("chunk_id", ""): i for i, c in enumerate(chunks)}
                    pairs = []
                    for src_id, tgt_id, strength in kg_pairs:
                        idx_a = chunk_id_to_idx.get(src_id)
                        idx_b = chunk_id_to_idx.get(tgt_id)
                        if idx_a is not None and idx_b is not None:
                            pairs.append((idx_a, idx_b, strength))
                    logger.info(f"KG provided {len(pairs)} candidate pairs")
                else:
                    # Fallback: FAISS similarity band
                    logger.info("Using FAISS similarity band for pair selection")
                    pairs = find_multihop_chunk_pairs(
                        chunks=chunks,
                        chunk_embeddings=chunk_embeddings,
                        similarity_min=settings.multihop_similarity_min,
                        similarity_max=settings.multihop_similarity_max,
                        max_pairs=settings.max_multihop_pairs,
                    )

                pair_time = time.time() - start
                logger.info(f"Found {len(pairs)} candidate chunk pairs in {pair_time:.1f}s")

            if not pairs:
                logger.info("No suitable chunk pairs found for multi-hop questions")
                return {
                    "multihop_pairs": [],
                    "multihop_qa_pairs": [],
                    "phase_timings": [{"phase": "multihop_generator", "duration": pair_time}],
                }

            # --- Step 2: Initialize LLM ---
            with logger.step("Initialize LLM"):
                llm = get_question_llm()
                logger.info(f"LLM: {settings.azure_openai_model_name} (temp=0.7)")

            # --- Step 3: Process pairs (bridge extraction + question generation) ---
            with logger.step("Generate multi-hop questions"):
                start = time.time()
                max_q = settings.multihop_questions_per_pair
                concurrent = settings.max_concurrent_calls

                all_pair_records = []
                all_multihop_qa = []

                # Process in concurrent batches
                for i in range(0, len(pairs), concurrent):
                    batch = pairs[i:i + concurrent]
                    batch_num = i // concurrent + 1
                    total_batches = (len(pairs) - 1) // concurrent + 1

                    logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} pairs)...")

                    tasks = [
                        _process_single_pair(
                            llm,
                            chunks[idx_a],
                            chunks[idx_b],
                            sim,
                            max_q,
                        )
                        for idx_a, idx_b, sim in batch
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"  Pair processing error: {result}")
                            continue
                        pair_record, questions = result
                        if pair_record:
                            all_pair_records.append(pair_record)
                        all_multihop_qa.extend(questions)

                    logger.info(f"  Running total: {len(all_multihop_qa)} multi-hop questions")

                    if i + concurrent < len(pairs):
                        await asyncio.sleep(1)

                gen_time = time.time() - start

            # --- Step 4: Merge into existing qa_pairs ---
            with logger.step("Merge with single-hop questions"):
                merged_qa = list(existing_qa) + all_multihop_qa
                logger.info(f"Single-hop: {len(existing_qa)} | Multi-hop: {len(all_multihop_qa)} | Merged: {len(merged_qa)}")

            # --- Summary ---
            total_time = pair_time + gen_time
            pairs_with_questions = sum(1 for p in all_pair_records if p["questions_generated"] > 0)

            reasoning_types = {}
            for q in all_multihop_qa:
                rt = q.get("reasoning_type", "unknown")
                reasoning_types[rt] = reasoning_types.get(rt, 0) + 1

            logger.info(f"Summary:")
            logger.info(f"  Candidate pairs: {len(pairs)}")
            logger.info(f"  Pairs with bridge concepts: {len(all_pair_records)}")
            logger.info(f"  Pairs producing questions: {pairs_with_questions}")
            logger.info(f"  Total multi-hop questions: {len(all_multihop_qa)}")
            logger.info(f"  Reasoning types: {reasoning_types}")
            logger.info(f"  Total time: {total_time:.1f}s")

            return {
                "qa_pairs": merged_qa,
                "multihop_pairs": all_pair_records,
                "multihop_qa_pairs": all_multihop_qa,
                "phase_timings": [{
                    "phase": "multihop_generator",
                    "duration": round(total_time, 2),
                    "candidate_pairs": len(pairs),
                    "pairs_with_bridges": len(all_pair_records),
                    "questions_generated": len(all_multihop_qa),
                    "reasoning_types": reasoning_types,
                }],
            }

    except Exception as e:
        logger.error(f"Multi-hop generation failed: {e}")
        return {
            "multihop_pairs": [],
            "multihop_qa_pairs": [],
            "errors": [f"Multi-hop generation failed: {e}"],
            "phase_timings": [{"phase": "multihop_generator", "duration": 0.0, "error": str(e)}],
        }
