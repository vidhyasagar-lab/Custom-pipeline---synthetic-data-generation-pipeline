"""Agent 2: Question Generator (LangGraph Node)
Generates ALL possible conversational questions for each chunk.
Questions are human-like, natural chatbot conversations — no noise, no filler.
Uses a strict quality filter to remove vague, generic, or unanswerable questions.
"""

import json
import time
import asyncio

from app.core.config import get_settings
from app.core.llm import get_question_llm
from app.core.logging_config import get_agent_logger
from app.core.graph_state import GraphState
from app.core.personas import generate_scenarios, QueryScenario, DEFAULT_DISTRIBUTION

logger = get_agent_logger("Agent2:QuestionGen")

QUESTION_PROMPT = """You are generating training data for a customer-facing chatbot. Your job is to generate the most important and diverse questions a real human user might type into a chatbot based on the text below.

CRITICAL CONSTRAINT: You MUST generate questions ONLY from the information present in the TEXT CHUNK below. Do NOT use your own knowledge, training data, or any external information whatsoever. Every question must be directly answerable from the provided text alone. If a concept appears in the text, ask about it using only the context the text provides — do NOT enrich, expand, or supplement with outside knowledge.

TEXT CHUNK:
{chunk_text}

RULES — READ CAREFULLY:
1. Generate up to {max_questions} high-quality questions that cover the key facts, concepts, definitions, processes, comparisons, and relationships in the text. Focus on QUALITY and DIVERSITY — each question should ask something meaningfully different.
2. STRICTLY use ONLY the provided text as your source. Do NOT inject any facts, definitions, examples, or context from your own knowledge. If the text does not mention something, do NOT ask about it.
3. Questions MUST sound like a real person typing into a chat box. Use casual, natural language:
   - GOOD: "What's the difference between X and Y?"
   - GOOD: "Can you explain how X works?"
   - GOOD: "Hey, I'm confused about X — what does it mean?"
   - BAD: "Describe the mechanism by which X operates." (too academic)
   - BAD: "What is mentioned in the text about X?" (references the text directly)
3. NEVER reference "the text", "the passage", "the document", "the chunk", or "according to" — the chatbot user doesn't know about any document.
4. NO noise questions:
   - No vague questions ("What is important?", "Tell me more")
   - No questions that can't be answered from the text
   - No yes/no questions unless they lead to explanation
   - No redundant/overlapping questions — each must ask something distinct
5. For each question, generate a natural follow-up a user would ask AFTER getting the answer. The follow-up must also be answerable strictly from the provided text.
6. Assign question type and difficulty accurately.

Question types (use ALL that apply to the content):
- "factual": Direct questions about specific facts, numbers, names, definitions
- "explanatory": "Can you explain...", "What does ... mean?", "Why does..."
- "procedural": "How do I...", "What are the steps to...", "What's the process for..."
- "comparative": "What's the difference between...", "How does X compare to..."
- "inferential": Questions requiring reasoning or connecting ideas from the text
- "scenario": "What would happen if...", "In a situation where..."

Respond ONLY with a JSON array:
[
  {{
    "question": "natural conversational question",
    "follow_up_q": "natural follow-up after getting the answer",
    "question_type": "factual|explanatory|procedural|comparative|inferential|scenario",
    "difficulty": "simple|moderate|complex"
  }}
]

Generate up to {max_questions} questions maximum. Prioritize the most useful and diverse questions. Return ONLY valid JSON array."""

NOISE_FILTER_PROMPT = """You are a strict quality filter for chatbot training data. Review these questions and REMOVE any that are:

1. Vague or generic (could apply to any topic)
2. Reference a document, text, passage, or source
3. Too academic or formal (not how a real person would type to a chatbot)
4. Redundant (asks the same thing as another question, just rephrased)
5. Unanswerable from the original context
6. Yes/no questions that don't lead to useful explanations
7. Questions that are just noise or filler

QUESTIONS TO REVIEW:
{questions_json}

Return ONLY the questions that pass ALL quality checks, as a JSON array with the same structure.
If ALL questions are good, return them all. If none pass, return an empty array [].
Return ONLY valid JSON array, no other text."""

# Scenario injection block (inserted before rules when personas are enabled)
SCENARIO_INJECTION = """
ADDITIONAL SCENARIO CONSTRAINTS:
{scenario_instruction}

Apply these persona/style/length constraints to ALL questions you generate.
"""


async def generate_questions_for_chunk(
    llm,
    chunk: dict,
    max_retries: int = 3,
    scenario: QueryScenario | None = None,
) -> list[dict]:
    """Generate ALL possible questions for a single chunk using LLM.
    
    If a scenario is provided, injects persona/style/length instructions.
    """
    try:
        chunk_text = chunk["page_content"]
        chunk_id = chunk["metadata"]["chunk_id"]

        if not chunk_text or len(chunk_text.strip()) < 50:
            logger.warning(f"  Chunk {chunk_id}: Too short ({len(chunk_text)} chars), skipping")
            return []

        settings = get_settings()
        max_q = settings.max_questions_per_chunk
        prompt = QUESTION_PROMPT.format(chunk_text=chunk_text[:4000], max_questions=max_q)

        # Inject scenario instructions if personas are enabled
        if scenario:
            scenario_block = SCENARIO_INJECTION.format(
                scenario_instruction=scenario.to_prompt_instruction()
            )
            prompt = prompt + scenario_block

        for attempt in range(max_retries + 1):
            try:
                response = await llm.ainvoke(prompt)
                content = response.content.strip()

                # Strip markdown code fences if present
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                questions = json.loads(content)

                if not isinstance(questions, list):
                    raise ValueError("Response is not a JSON array")

                # Basic validation: remove clearly bad entries
                valid_questions = []
                for q in questions[:max_q]:  # Hard cap per chunk
                    if not isinstance(q, dict):
                        continue
                    question_text = q.get("question", "").strip()
                    if not question_text or len(question_text) < 10:
                        continue
                    # Skip questions that reference the document
                    lower_q = question_text.lower()
                    if any(ref in lower_q for ref in [
                        "the text", "the passage", "the document", "the chunk",
                        "mentioned in", "according to the", "based on the reading",
                        "the article", "this section", "the paragraph",
                    ]):
                        continue
                    # Skip overly vague questions
                    if lower_q in [
                        "what is important?", "tell me more.", "can you explain?",
                        "what else?", "anything else?", "what do you think?",
                    ]:
                        continue

                    q["source_chunk_id"] = chunk_id
                    q["source_chunk"] = chunk_text[:500]
                    if scenario:
                        q["persona"] = scenario.persona.name
                        q["query_style"] = scenario.style.name
                        q["query_length"] = scenario.length
                    valid_questions.append(q)

                logger.info(f"  Chunk {chunk_id}: {len(valid_questions)} questions (raw: {len(questions)})")
                return valid_questions

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    logger.warning(f"  Chunk {chunk_id}: Parse error (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(1)
                    continue
                logger.error(f"  Chunk {chunk_id}: Failed to parse after {max_retries + 1} attempts: {e}")
                return []

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"  Chunk {chunk_id}: Error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"  Chunk {chunk_id}: Failed after {max_retries + 1} attempts: {e}")
                return []

    except Exception as e:
        logger.error(f"  Question generation failed for chunk {chunk.get('metadata', {}).get('chunk_id', '?')}: {e}")
        return []


async def _filter_single_batch(llm, batch: list[dict], max_retries: int = 2) -> list[dict]:
    """Filter a single batch of questions for noise. Used for parallel execution."""
    batch_for_review = [
        {"question": q["question"], "follow_up_q": q.get("follow_up_q", ""),
         "question_type": q.get("question_type", ""), "difficulty": q.get("difficulty", "")}
        for q in batch
    ]

    prompt = NOISE_FILTER_PROMPT.format(questions_json=json.dumps(batch_for_review, indent=2))

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            kept = json.loads(content)
            if not isinstance(kept, list):
                raise ValueError("Filter response is not a JSON array")

            # Match back to original questions (by question text)
            kept_texts = {k.get("question", "").strip().lower() for k in kept}
            return [q for q in batch if q["question"].strip().lower() in kept_texts]

        except (json.JSONDecodeError, ValueError):
            if attempt < max_retries:
                continue
            return batch  # If filter fails, keep all

        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"Noise filter failed for batch, keeping all: {e}")
            return batch


async def filter_noise_questions(
    llm,
    questions: list[dict],
    batch_size: int = 30,
) -> list[dict]:
    """Use LLM to filter out noise questions — parallel batch processing."""
    try:
        if len(questions) <= 5:
            return questions

        # Create all filter tasks
        batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
        
        settings = get_settings()
        concurrent = settings.max_concurrent_calls

        filtered = []
        for i in range(0, len(batches), concurrent):
            group = batches[i:i + concurrent]
            tasks = [_filter_single_batch(llm, batch) for batch in group]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Noise filter batch error: {result}")
                    # On failure, keep corresponding batch
                    idx = results.index(result)
                    if idx < len(group):
                        filtered.extend(group[idx])
                elif isinstance(result, list):
                    filtered.extend(result)

            if i + concurrent < len(batches):
                await asyncio.sleep(1)

        return filtered

    except Exception as e:
        logger.error(f"Noise filter failed entirely: {e}")
        return questions


async def question_generator_node(state: GraphState) -> dict:
    """LangGraph node: Generate ALL possible conversational questions for each chunk."""
    try:
        with logger.phase("Question Generation"):

            settings = get_settings()
            chunks = state.get("chunks", [])

            if not chunks:
                raise ValueError("No chunks available. Run Semantic Chunker first.")

            num_chunks = len(chunks)
            logger.info(f"Chunks: {num_chunks}")
            max_q = settings.max_questions_per_chunk
            logger.info(f"Mode: Comprehensive \u2014 up to {max_q} questions per chunk")

            # --- Initialize LLM ---
            with logger.step("Initialize LLM"):
                try:
                    llm = get_question_llm()
                    logger.info(f"LLM: {settings.azure_openai_model_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM: {e}")
                    raise RuntimeError(f"LLM initialization failed: {e}") from e

            # --- Generate questions in batches ---
            with logger.step("Generate questions (exhaustive)"):
                start = time.time()
                all_qa_pairs = []
                batch_size = settings.max_concurrent_calls
                failed_chunks = 0

                # Generate scenarios for persona/style diversity
                scenarios = None
                if settings.enable_personas:
                    scenarios = generate_scenarios(
                        num_scenarios=num_chunks,
                        personas=settings.active_personas,
                        styles=settings.active_query_styles,
                        lengths=settings.active_query_lengths,
                    )
                    logger.info(f"Personas enabled: {len(set(s.persona.name for s in scenarios))} personas, "
                                f"{len(set(s.style.name for s in scenarios))} styles")

                for i in range(0, num_chunks, batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (num_chunks - 1) // batch_size + 1

                    logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

                    tasks = [
                        generate_questions_for_chunk(
                            llm, chunk,
                            scenario=scenarios[i + idx] if scenarios and (i + idx) < len(scenarios) else None,
                        )
                        for idx, chunk in enumerate(batch)
                    ]

                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                    except Exception as e:
                        logger.error(f"  Batch {batch_num} gather failed: {e}")
                        failed_chunks += len(batch)
                        continue

                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"  Batch error: {result}")
                            failed_chunks += 1
                        elif isinstance(result, list):
                            all_qa_pairs.extend(result)
                        else:
                            failed_chunks += 1

                    logger.info(f"  Running total: {len(all_qa_pairs)} questions generated")

                    if i + batch_size < num_chunks:
                        await asyncio.sleep(1)

                gen_elapsed = time.time() - start

            # --- Noise filter pass ---
            with logger.step("Noise filter"):
                pre_filter = len(all_qa_pairs)
                logger.info(f"Running noise filter on {pre_filter} questions...")

                try:
                    start = time.time()
                    all_qa_pairs = await filter_noise_questions(llm, all_qa_pairs)
                    filter_elapsed = time.time() - start
                    noise_removed = pre_filter - len(all_qa_pairs)
                    logger.info(f"Noise filter complete in {filter_elapsed:.1f}s: removed {noise_removed} noisy questions")
                except Exception as e:
                    logger.warning(f"Noise filter failed, keeping all questions: {e}")
                    filter_elapsed = 0
                    noise_removed = 0

            # --- Summary ---
            with logger.step("Summary"):
                type_counts = {}
                difficulty_counts = {}
                for q in all_qa_pairs:
                    qt = q.get("question_type", "unknown")
                    diff = q.get("difficulty", "unknown")
                    type_counts[qt] = type_counts.get(qt, 0) + 1
                    difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

                logger.info(f"Question generation complete in {gen_elapsed:.1f}s")
                logger.info(f"  Total questions (after filter): {len(all_qa_pairs)}")
                logger.info(f"  Noise removed: {noise_removed}")
                logger.info(f"  Failed chunks: {failed_chunks}")
                logger.info(f"  Type distribution: {type_counts}")
                logger.info(f"  Difficulty distribution: {difficulty_counts}")

            return {
                "qa_pairs": all_qa_pairs,
                "phase_timings": [{
                    "phase": "question_generation",
                    "mode": f"comprehensive (up to {settings.max_questions_per_chunk} per chunk)",
                    "total_questions": len(all_qa_pairs),
                    "pre_filter_count": pre_filter,
                    "noise_removed": noise_removed,
                    "failed_chunks": failed_chunks,
                    "type_distribution": type_counts,
                    "difficulty_distribution": difficulty_counts,
                    "generation_seconds": round(gen_elapsed, 2),
                    "filter_seconds": round(filter_elapsed, 2),
                }],
            }

    except Exception as e:
        logger.error(f"Question generator node failed: {e}")
        raise
