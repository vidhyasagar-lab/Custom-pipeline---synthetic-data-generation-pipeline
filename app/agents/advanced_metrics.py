"""Advanced validation metrics inspired by RAGAS.

Implements:
- Claim-based faithfulness: Decompose answer → atomic claims → verify each against context
- Answer relevancy: Generate questions from answer, check if they match the original
- Context quality metrics: Context Precision and Context Recall
- Noise sensitivity: Test if irrelevant context changes the answer
"""

from __future__ import annotations

import asyncio
import json

from app.core.llm import get_validation_llm
from app.core.logging_config import get_agent_logger

logger = get_agent_logger("AdvancedMetrics")

# ═══════════════════════════════════════════════════════════════════════════════
# P1.2: CLAIM-BASED FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════════════

CLAIM_DECOMPOSITION_PROMPT = """Decompose this answer into independent, atomic factual claims.
Each claim should be a single verifiable statement.

ANSWER: {answer}

Return a JSON array of claim strings:
["claim 1", "claim 2", ...]

Rules:
- Each claim must be a single, self-contained factual statement
- Break compound sentences into separate claims
- Omit opinions, hedging, and filler ("I think", "Based on", etc.)
- Keep only substantive factual claims (skip greetings, meta-statements)
- Maximum 15 claims

Return ONLY valid JSON array."""

CLAIM_VERIFICATION_PROMPT = """For each claim below, determine if it is supported by the given context.

CONTEXT:
{context}

CLAIMS:
{claims_json}

For each claim, respond with a JSON array of objects:
[
  {{"claim": "the claim text", "verdict": "supported" | "not_supported", "reason": "brief reason"}}
]

Rules:
- "supported": The claim can be directly inferred from the context
- "not_supported": The claim cannot be found in or inferred from the context
- Be strict: if the context doesn't explicitly contain the information, mark as not_supported

Return ONLY valid JSON array."""


async def compute_claim_faithfulness(
    llm,
    answer: str,
    context: list[str],
    max_retries: int = 2,
) -> dict:
    """Decompose answer into claims and verify each against context.

    Returns:
        {
            "claim_faithfulness_score": float (0-1),
            "total_claims": int,
            "supported_claims": int,
            "unsupported_claims": list[str],
            "claims": list[dict],
        }
    """
    try:
        # Step 1: Decompose answer into claims
        prompt = CLAIM_DECOMPOSITION_PROMPT.format(answer=answer)
        for attempt in range(max_retries + 1):
            try:
                response = await llm.ainvoke(prompt)
                content = response.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                claims = json.loads(content.strip())
                if isinstance(claims, list) and claims:
                    break
                claims = []
            except Exception:
                if attempt == max_retries:
                    return {"claim_faithfulness_score": 0.0, "total_claims": 0,
                            "supported_claims": 0, "unsupported_claims": [], "claims": []}
                await asyncio.sleep(1)

        if not claims:
            return {"claim_faithfulness_score": 1.0, "total_claims": 0,
                    "supported_claims": 0, "unsupported_claims": [], "claims": []}

        # Step 2: Verify claims against context
        ctx_text = "\n\n".join(context)[:6000]
        claims_json = json.dumps(claims, indent=2)
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            context=ctx_text, claims_json=claims_json
        )

        for attempt in range(max_retries + 1):
            try:
                response = await llm.ainvoke(prompt)
                content = response.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                verdicts = json.loads(content.strip())
                if isinstance(verdicts, list):
                    break
                verdicts = []
            except Exception:
                if attempt == max_retries:
                    verdicts = []
                await asyncio.sleep(1)

        supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
        total = len(verdicts) if verdicts else len(claims)
        unsupported = [v.get("claim", "") for v in verdicts if v.get("verdict") != "supported"]

        score = supported / total if total > 0 else 0.0

        return {
            "claim_faithfulness_score": round(score, 3),
            "total_claims": total,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "claims": verdicts,
        }

    except Exception as e:
        logger.error(f"Claim faithfulness computation failed: {e}")
        return {"claim_faithfulness_score": 0.0, "total_claims": 0,
                "supported_claims": 0, "unsupported_claims": [], "claims": []}


# ═══════════════════════════════════════════════════════════════════════════════
# P1.3: CONTEXT QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

CONTEXT_PRECISION_PROMPT = """Given the question and answer, evaluate each retrieved context chunk.
Is each chunk relevant to answering the question?

QUESTION: {question}
ANSWER: {answer}

RETRIEVED CHUNKS:
{chunks_text}

For each chunk (numbered), respond with a JSON array:
[
  {{"chunk_number": 1, "relevant": true, "reason": "brief reason"}},
  {{"chunk_number": 2, "relevant": false, "reason": "brief reason"}}
]

A chunk is "relevant" if it contains information that was actually used or needed
to produce the answer. Be strict.

Return ONLY valid JSON array."""

CONTEXT_RECALL_PROMPT = """Given the answer, identify which statements in the answer are
attributable to the retrieved context.

ANSWER: {answer}

RETRIEVED CONTEXT:
{context}

For each sentence/statement in the answer, determine if it can be attributed to
the retrieved context.

Return a JSON object:
{{
  "total_statements": <int>,
  "attributed_statements": <int>,
  "unattributed": ["statement that isn't in context", ...]
}}

Return ONLY valid JSON."""


async def compute_context_precision(
    llm, question: str, answer: str, retrieved_chunks: list[str], max_retries: int = 2
) -> dict:
    """Context Precision: What fraction of retrieved chunks were actually relevant?

    Higher = better retrieval (less noise in retrieved context).
    """
    if not retrieved_chunks:
        return {"context_precision": 0.0, "relevant_chunks": 0, "total_chunks": 0}

    chunks_text = "\n\n".join(
        f"--- Chunk {i+1} ---\n{c[:1000]}" for i, c in enumerate(retrieved_chunks)
    )
    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=question, answer=answer, chunks_text=chunks_text
    )

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            results = json.loads(content.strip())
            if isinstance(results, list):
                relevant = sum(1 for r in results if r.get("relevant"))
                return {
                    "context_precision": round(relevant / len(retrieved_chunks), 3),
                    "relevant_chunks": relevant,
                    "total_chunks": len(retrieved_chunks),
                    "details": results,
                }
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(1)

    return {"context_precision": 0.0, "relevant_chunks": 0, "total_chunks": len(retrieved_chunks)}


async def compute_context_recall(
    llm, answer: str, context: list[str], max_retries: int = 2
) -> dict:
    """Context Recall: What fraction of answer statements are attributable to retrieved context?

    Higher = better coverage (the retrieval found all needed information).
    """
    if not context:
        return {"context_recall": 0.0, "total_statements": 0, "attributed_statements": 0}

    ctx_text = "\n\n".join(context)[:6000]
    prompt = CONTEXT_RECALL_PROMPT.format(answer=answer, context=ctx_text)

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            result = json.loads(content.strip())
            total = result.get("total_statements", 0)
            attributed = result.get("attributed_statements", 0)
            recall = attributed / total if total > 0 else 0.0
            return {
                "context_recall": round(recall, 3),
                "total_statements": total,
                "attributed_statements": attributed,
                "unattributed": result.get("unattributed", []),
            }
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(1)

    return {"context_recall": 0.0, "total_statements": 0, "attributed_statements": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# P2.6: NOISE SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

async def compute_noise_sensitivity(
    llm, question: str, answer: str, context: list[str],
    noise_chunk: str = "The weather in Paris is typically mild in spring with temperatures averaging 15°C.",
    max_retries: int = 2,
) -> dict:
    """Noise Sensitivity: Does adding an irrelevant chunk change the answer?

    Generates a new answer with the noise chunk injected into context,
    then compares semantic similarity between original and noisy answer.
    Lower sensitivity = more robust.
    """
    from app.models.llm_responses import parse_llm_response, AnswerResponse
    from app.agents.answer_generator import ANSWER_PROMPT

    noisy_context = context + [f"[Source: noise_test]\n{noise_chunk}"]
    ctx_text = "\n\n---\n\n".join(noisy_context)

    prompt = ANSWER_PROMPT.format(
        context=ctx_text[:8000],
        question=question,
        follow_up_q="No follow-up question.",
    )

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            parsed = parse_llm_response(response.content.strip(), AnswerResponse)
            if parsed is None:
                continue
            noisy_answer = parsed.answer

            # Simple similarity: check if answers are substantively different
            original_words = set(answer.lower().split())
            noisy_words = set(noisy_answer.lower().split())
            if not original_words or not noisy_words:
                return {"noise_sensitivity": 0.0, "noise_changed_answer": False}

            overlap = len(original_words & noisy_words) / len(original_words | noisy_words)
            changed = overlap < 0.7  # Significantly different

            return {
                "noise_sensitivity": round(1.0 - overlap, 3),
                "noise_changed_answer": changed,
                "original_answer_preview": answer[:100],
                "noisy_answer_preview": noisy_answer[:100],
            }
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(1)

    return {"noise_sensitivity": 0.0, "noise_changed_answer": False}


# ═══════════════════════════════════════════════════════════════════════════════
# P2.8: ANSWER RELEVANCY (Reverse Metric)
# ═══════════════════════════════════════════════════════════════════════════════

REVERSE_QUESTION_PROMPT = """Given this answer, generate 3 questions that this answer would be a good response to.

ANSWER: {answer}

Return a JSON array of 3 question strings:
["question 1", "question 2", "question 3"]

Rules:
- Questions should be natural and conversational
- Each question should be meaningfully different
- Questions should capture the core information in the answer

Return ONLY valid JSON array."""


async def compute_answer_relevancy(
    llm, question: str, answer: str, max_retries: int = 2,
) -> dict:
    """Answer Relevancy: Generate questions from the answer and check if they
    semantically match the original question.

    If the answer is relevant, questions generated from it should be similar
    to the original question. This catches faithful-but-irrelevant answers.
    """
    prompt = REVERSE_QUESTION_PROMPT.format(answer=answer)

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
            generated_qs = json.loads(content.strip())
            if not isinstance(generated_qs, list):
                continue

            # Compute word-level overlap between original and generated questions
            orig_words = set(question.lower().split())
            similarities = []
            for gq in generated_qs:
                if not isinstance(gq, str):
                    continue
                gq_words = set(gq.lower().split())
                if orig_words and gq_words:
                    overlap = len(orig_words & gq_words) / len(orig_words | gq_words)
                    similarities.append(overlap)

            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

            return {
                "answer_relevancy": round(avg_sim, 3),
                "generated_questions": generated_qs[:3],
                "question_similarities": [round(s, 3) for s in similarities],
            }
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(1)

    return {"answer_relevancy": 0.0, "generated_questions": [], "question_similarities": []}


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH COMPUTATION (called from quality_validator)
# ═══════════════════════════════════════════════════════════════════════════════

async def compute_advanced_metrics(triple: dict, sample_noise: bool = False) -> dict:
    """Compute all advanced metrics for a single QA triple.

    Called from quality_validator for each triple. Returns a dict of
    advanced metrics to merge into the triple's validation_scores.
    """
    llm = get_validation_llm()

    question = triple.get("question", "")
    answer = triple.get("answer", "")
    context = triple.get("context", [])

    tasks = [
        compute_claim_faithfulness(llm, answer, context),
        compute_context_precision(llm, question, answer, context),
        compute_context_recall(llm, answer, context),
        compute_answer_relevancy(llm, question, answer),
    ]

    if sample_noise:
        tasks.append(compute_noise_sensitivity(llm, question, answer, context))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    metrics = {}
    labels = ["claim_faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    if sample_noise:
        labels.append("noise_sensitivity")

    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            logger.warning(f"Advanced metric '{label}' failed: {result}")
        elif isinstance(result, dict):
            # Flatten nested dicts into top-level metrics
            metrics.update(result)

    return metrics
