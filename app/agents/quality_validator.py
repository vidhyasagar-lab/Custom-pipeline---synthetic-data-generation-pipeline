"""Agent 5: Quality Validator (LangGraph Node)
Validates QA pairs for faithfulness (answer grounded in context)
and chatbot tone quality using a separate validation LLM (gpt-5.4-mini).
Calculates overall_score server-side from component scores.
"""

import json
import time
import asyncio

from app.core.config import get_settings
from app.core.llm import get_validation_llm
from app.core.logging_config import get_agent_logger
from app.core.graph_state import GraphState
from app.models.llm_responses import ValidationScores, MultihopValidationScores, parse_llm_response
from app.agents.advanced_metrics import compute_advanced_metrics

logger = get_agent_logger("Agent5:QualityValidator")

VALIDATION_PROMPT = """You are a quality judge for chatbot training data. Evaluate this QA pair:

CONTEXT (retrieved from source document):
{context}

QUESTION: {question}
ANSWER: {answer}

FOLLOW-UP QUESTION: {follow_up_q}
FOLLOW-UP ANSWER: {follow_up_a}

Score this QA pair on these criteria (0.0 to 1.0 each):

1. **faithfulness**: Is the answer fully supported by the context?
   - 1.0: Every claim directly traceable to context, no additions
   - 0.8: All key claims supported, minor elaboration within reason
   - 0.5: Some claims supported but includes unsupported inferences
   - 0.2: Major claims not found in context, significant hallucination
   - 0.0: Answer contradicts or is unrelated to context

2. **relevance**: Does the answer actually address the question?
   - 1.0: Directly and completely answers what was asked
   - 0.8: Answers the question with minor tangential info
   - 0.5: Partially addresses the question, misses key aspects
   - 0.2: Mostly off-topic with some relevant fragments
   - 0.0: Completely irrelevant to the question

3. **completeness**: Does the answer cover the key points from the context?
   - 1.0: All important points from context relevant to the question are included
   - 0.8: Most key points covered, minor omissions
   - 0.5: Covers about half of relevant points
   - 0.2: Missing most important points
   - 0.0: Essentially empty or trivial response

4. **tone**: Does the answer sound like a helpful chatbot?
   - 1.0: Natural, conversational, helpful, well-structured
   - 0.8: Mostly conversational with minor formality issues
   - 0.5: Mix of conversational and academic/robotic
   - 0.2: Mostly academic/robotic with little personality
   - 0.0: Completely robotic, copy-pasted, or inappropriate tone

5. **follow_up_quality**: Is the follow-up Q&A coherent and useful?
   - 1.0: Natural continuation, adds value, well-answered
   - 0.8: Good continuation with minor issues
   - 0.5: Somewhat related but forced or weakly answered
   - 0.2: Barely related or poorly answered
   - 0.0: Nonsensical or completely disconnected

Respond ONLY with a JSON object:
{{
  "faithfulness": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "tone": 0.0,
  "follow_up_quality": 0.0,
  "reasoning": "brief explanation"
}}

Return ONLY valid JSON, no other text."""

MULTIHOP_VALIDATION_PROMPT = """You are a quality judge for chatbot training data. This is a MULTI-HOP question that should require synthesizing information from MULTIPLE source passages.

CONTEXT (from multiple source passages):
{context}

QUESTION: {question}
ANSWER: {answer}

FOLLOW-UP QUESTION: {follow_up_q}
FOLLOW-UP ANSWER: {follow_up_a}

Score this multi-hop QA pair on these criteria (0.0 to 1.0 each):

1. **faithfulness**: Is the answer fully supported by the context?
   - 1.0: Every claim directly traceable to context passages
   - 0.5: Some claims supported but includes unsupported inferences
   - 0.0: Answer contradicts or is unrelated to context

2. **relevance**: Does the answer actually address the question?
   - 1.0: Directly and completely answers what was asked
   - 0.5: Partially addresses the question
   - 0.0: Completely irrelevant

3. **completeness**: Does the answer cover the key points from the context?
   - 1.0: All important points from both passages included
   - 0.5: Covers about half of relevant points
   - 0.0: Missing most important points

4. **tone**: Does the answer sound like a helpful chatbot?
   - 1.0: Natural, conversational, helpful
   - 0.5: Mix of conversational and robotic
   - 0.0: Completely robotic or inappropriate

5. **follow_up_quality**: Is the follow-up Q&A coherent and useful?
   - 1.0: Natural continuation that adds value
   - 0.5: Somewhat related but forced
   - 0.0: Nonsensical or disconnected

6. **multi_hop_validity**: Does the answer genuinely synthesize information from MULTIPLE passages?
   - 1.0: Clearly combines distinct information from different passages, impossible to answer from one alone
   - 0.8: Mostly requires both passages, minor overlap
   - 0.5: Uses both passages but could partially be answered from one
   - 0.2: Mostly answerable from a single passage
   - 0.0: Entirely answerable from one passage, no real synthesis

Respond ONLY with a JSON object:
{{
  "faithfulness": 0.0,
  "relevance": 0.0,
  "completeness": 0.0,
  "tone": 0.0,
  "follow_up_quality": 0.0,
  "multi_hop_validity": 0.0,
  "reasoning": "brief explanation"
}}

Return ONLY valid JSON, no other text."""


def compute_overall_score(scores: dict, is_multihop: bool = False) -> float:
    """Compute weighted overall score server-side.
    
    Standard weights: faithfulness(0.3) + relevance(0.25) + completeness(0.2) + tone(0.15) + follow_up_quality(0.1)
    Multi-hop weights: faithfulness(0.25) + relevance(0.2) + completeness(0.15) + tone(0.1) + follow_up_quality(0.1) + multi_hop_validity(0.2)
    """
    if is_multihop:
        return (
            scores.get("faithfulness", 0.0) * 0.25
            + scores.get("relevance", 0.0) * 0.2
            + scores.get("completeness", 0.0) * 0.15
            + scores.get("tone", 0.0) * 0.1
            + scores.get("follow_up_quality", 0.0) * 0.1
            + scores.get("multi_hop_validity", 0.0) * 0.2
        )
    return (
        scores.get("faithfulness", 0.0) * 0.3
        + scores.get("relevance", 0.0) * 0.25
        + scores.get("completeness", 0.0) * 0.2
        + scores.get("tone", 0.0) * 0.15
        + scores.get("follow_up_quality", 0.0) * 0.1
    )


async def validate_qa_pair(llm, triple: dict, max_retries: int = 2) -> dict | None:
    """Validate a single QA pair using LLM-as-Judge.
    Uses multi-hop prompt for multihop questions to validate cross-passage synthesis.
    """
    try:
        context = "\n".join(triple.get("context", []))
        is_multihop = triple.get("question_type") == "multihop"

        prompt_template = MULTIHOP_VALIDATION_PROMPT if is_multihop else VALIDATION_PROMPT
        prompt = prompt_template.format(
            context=context[:6000],
            question=triple.get("question", ""),
            answer=triple.get("answer", ""),
            follow_up_q=triple.get("follow_up_q", ""),
            follow_up_a=triple.get("follow_up_a", ""),
        )

        for attempt in range(max_retries + 1):
            try:
                response = await llm.ainvoke(prompt)
                content = response.content.strip()

                model_class = MultihopValidationScores if is_multihop else ValidationScores
                parsed = parse_llm_response(content, model_class)
                if parsed is None:
                    raise ValueError("Validation response failed Pydantic validation")

                return parsed.model_dump()

            except (ValueError,) as e:
                if attempt < max_retries:
                    logger.warning(f"  Validation parse error (attempt {attempt + 1}): {e}")
                    continue
                logger.error(f"  Validation parse failed after {max_retries + 1} attempts")
                return None

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"  Validation LLM error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"  Validation failed: {e}")
                return None

    except Exception as e:
        logger.error(f"  Unexpected validation error: {e}")
        return None


async def quality_validator_node(state: GraphState) -> dict:
    """LangGraph node: Validate QA triples for quality using separate validation model."""
    try:
        with logger.phase("Quality Validation"):

            settings = get_settings()
            triples = state.get("qa_triples", [])
            threshold = state.get("config_quality_threshold", 0.7)

            if not triples:
                raise ValueError("No triples to validate. Run Answer Generator first.")

            logger.info(f"Input: {len(triples)} deduplicated triples")
            logger.info(f"Quality threshold: {threshold}")

            # --- Validate in batches ---
            with logger.step("Validate QA pairs"):
                try:
                    llm = get_validation_llm()
                    logger.info(f"Using validation model: {settings.azure_openai_validation_model_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize validation LLM: {e}")
                    raise RuntimeError(f"Validation LLM initialization failed: {e}") from e

                start = time.time()
                validated = []
                rejected = []
                failed_validation = 0
                all_scores = []
                batch_size = settings.max_concurrent_calls

                for i in range(0, len(triples), batch_size):
                    batch = triples[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(triples) - 1) // batch_size + 1

                    logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} triples)...")

                    tasks = [validate_qa_pair(llm, t) for t in batch]

                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                    except Exception as e:
                        logger.error(f"  Batch {batch_num} gather failed: {e}")
                        failed_validation += len(batch)
                        for triple in batch:
                            triple["quality_score"] = 0.0
                            triple["validation_scores"] = {}
                            rejected.append(triple)
                        continue

                    for triple, result in zip(batch, results):
                        if isinstance(result, Exception) or result is None:
                            failed_validation += 1
                            triple["quality_score"] = 0.0
                            triple["validation_scores"] = {}
                            rejected.append(triple)
                            continue

                        is_multihop = triple.get("question_type") == "multihop"

                        # Compute overall score server-side (don't trust LLM math)
                        component_scores = {
                            "faithfulness": result.get("faithfulness", 0.0),
                            "relevance": result.get("relevance", 0.0),
                            "completeness": result.get("completeness", 0.0),
                            "tone": result.get("tone", 0.0),
                            "follow_up_quality": result.get("follow_up_quality", 0.0),
                        }
                        if is_multihop:
                            component_scores["multi_hop_validity"] = result.get("multi_hop_validity", 0.0)
                        overall = compute_overall_score(component_scores, is_multihop=is_multihop)
                        all_scores.append(overall)

                        triple["quality_score"] = overall
                        triple["validation_scores"] = component_scores
                        triple["validation_reasoning"] = result.get("reasoning", "")

                        if overall >= threshold:
                            validated.append(triple)
                        else:
                            rejected.append(triple)

                    logger.info(f"  Validated: {len(validated)}, Rejected: {len(rejected)}")

                    if i + batch_size < len(triples):
                        await asyncio.sleep(1)

                elapsed = time.time() - start

            # --- Retry borderline rejects ---
            borderline_min = max(threshold - 0.15, 0.3)
            borderline = [t for t in rejected if borderline_min <= t.get("quality_score", 0) < threshold]

            recovered = 0
            if borderline:
                with logger.step(f"Retry borderline ({len(borderline)} triples, score {borderline_min:.2f}-{threshold:.2f})"):
                    retry_start = time.time()

                    for i in range(0, len(borderline), batch_size):
                        batch = borderline[i:i + batch_size]
                        tasks = [validate_qa_pair(llm, t) for t in batch]
                        try:
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                        except Exception:
                            continue

                        for triple, result in zip(batch, results):
                            if isinstance(result, Exception) or result is None:
                                continue
                            is_multihop = triple.get("question_type") == "multihop"
                            component_scores = {
                                "faithfulness": result.get("faithfulness", 0.0),
                                "relevance": result.get("relevance", 0.0),
                                "completeness": result.get("completeness", 0.0),
                                "tone": result.get("tone", 0.0),
                                "follow_up_quality": result.get("follow_up_quality", 0.0),
                            }
                            if is_multihop:
                                component_scores["multi_hop_validity"] = result.get("multi_hop_validity", 0.0)
                            retry_score = compute_overall_score(component_scores, is_multihop=is_multihop)

                            # Use the better of the two scores
                            if retry_score >= threshold:
                                triple["quality_score"] = retry_score
                                triple["validation_scores"] = component_scores
                                triple["validation_reasoning"] = result.get("reasoning", "")
                                rejected.remove(triple)
                                validated.append(triple)
                                all_scores.append(retry_score)
                                recovered += 1

                        if i + batch_size < len(borderline):
                            await asyncio.sleep(1)

                    retry_elapsed = time.time() - retry_start
                    elapsed += retry_elapsed
                    logger.info(f"Retry recovered {recovered}/{len(borderline)} borderline triples in {retry_elapsed:.1f}s")

            # --- Summary ---
            with logger.step("Summary"):
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

                criteria_avgs = {}
                for t in validated + rejected:
                    scores = t.get("validation_scores", {})
                    for k, v in scores.items():
                        if k not in criteria_avgs:
                            criteria_avgs[k] = []
                        criteria_avgs[k].append(v)

                criteria_summary = {k: round(sum(v) / len(v), 3) for k, v in criteria_avgs.items() if v}

                logger.info(f"Quality validation complete in {elapsed:.1f}s")
                logger.info(f"  Total evaluated: {len(triples)}")
                logger.info(f"  Passed (>= {threshold}): {len(validated)}")
                logger.info(f"  Rejected (< {threshold}): {len(rejected)}")
                logger.info(f"  Failed to validate: {failed_validation}")
                logger.info(f"  Average overall score: {avg_score:.3f}")
                logger.info(f"  Criteria averages: {criteria_summary}")

            # --- Advanced Metrics (RAGAS-inspired) ---
            advanced_metrics_summary = {}
            if settings.enable_advanced_metrics and validated:
                with logger.step("Advanced metrics (claim faithfulness, context quality, answer relevancy)"):
                    adv_start = time.time()
                    sample_rate = settings.noise_sensitivity_sample_rate

                    # Sample a subset for advanced metrics (can be expensive)
                    import random
                    sample_size = max(1, min(len(validated), int(len(validated) * 0.3)))
                    sample_triples = random.sample(validated, sample_size)
                    logger.info(f"Computing advanced metrics on {sample_size}/{len(validated)} validated triples")

                    adv_results = []
                    for i in range(0, len(sample_triples), batch_size):
                        batch = sample_triples[i:i + batch_size]
                        tasks = [
                            compute_advanced_metrics(
                                triple=t,
                                sample_noise=(random.random() < sample_rate),
                            )
                            for t in batch
                        ]
                        try:
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            for t, r in zip(batch, results):
                                if isinstance(r, dict):
                                    t["advanced_metrics"] = r
                                    adv_results.append(r)
                        except Exception as e:
                            logger.warning(f"Advanced metrics batch failed: {e}")

                        if i + batch_size < len(sample_triples):
                            await asyncio.sleep(1)

                    # Compute averages
                    if adv_results:
                        metric_keys = ["claim_faithfulness_score", "context_precision", "context_recall", "answer_relevancy"]
                        for key in metric_keys:
                            values = [r.get(key, 0.0) for r in adv_results if key in r]
                            if values:
                                advanced_metrics_summary[key] = round(sum(values) / len(values), 3)
                        advanced_metrics_summary["sample_size"] = len(adv_results)

                    adv_elapsed = time.time() - adv_start
                    logger.info(f"Advanced metrics computed in {adv_elapsed:.1f}s: {advanced_metrics_summary}")

            return {
                "validated_triples": validated,
                "rejected_triples": rejected,
                "advanced_metrics_summary": advanced_metrics_summary,
                "quality": {
                    "total_generated": len(triples),
                    "after_dedup": len(triples),
                    "after_validation": len(validated),
                    "rejected": len(rejected),
                    "avg_quality_score": round(avg_score, 3),
                    "criteria_averages": criteria_summary,
                },
                "phase_timings": [{
                    "phase": "quality_validation",
                    "model": settings.azure_openai_validation_model_name,
                    "total_evaluated": len(triples),
                    "passed": len(validated),
                    "rejected": len(rejected),
                    "failed_validation": failed_validation,
                    "avg_score": round(avg_score, 3),
                    "duration_seconds": round(elapsed, 2),
                }],
            }

    except Exception as e:
        logger.error(f"Quality validator node failed: {e}")
        raise
