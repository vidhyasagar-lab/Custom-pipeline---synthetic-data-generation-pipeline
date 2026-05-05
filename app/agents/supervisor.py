"""Supervisor Agent — LangGraph StateGraph Orchestrator
Custom Pipeline for Chatbot Synthetic Data Generation.

Graph Structure (multi-hop enabled):
    START → semantic_chunker → question_generator → multihop_generator
          → deduplicator → answer_generator → quality_validator → END

Graph Structure (multi-hop disabled):
    START → semantic_chunker → question_generator → deduplicator
          → answer_generator → quality_validator → END

Key optimization: Deduplication runs BEFORE answer generation to avoid
wasting LLM calls generating answers for duplicate questions.
"""

import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langgraph.graph import StateGraph, END, START

from app.core.config import get_settings
from app.core.logging_config import get_agent_logger, setup_logging
from app.core.graph_state import GraphState, create_initial_state
from app.core.progress import PipelineProgress
from app.agents.document_processor import semantic_chunker_node
from app.agents.question_generator import question_generator_node
from app.agents.multihop_generator import multihop_generator_node
from app.agents.answer_generator import answer_generator_node
from app.agents.deduplicator import deduplicator_node
from app.agents.quality_validator import quality_validator_node
from app.agents.knowledge_graph import build_knowledge_graph

logger = get_agent_logger("Supervisor")


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH NODE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

async def knowledge_graph_node(state: GraphState) -> dict:
    """LangGraph node: Build knowledge graph from chunks."""
    try:
        chunks = state.get("chunks", [])
        if not chunks:
            return {"knowledge_graph": {}, "phase_timings": [{"phase": "knowledge_graph", "duration": 0.0}]}

        start = time.time()
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        kg = await build_knowledge_graph(chunks)
        elapsed = time.time() - start

        kg_dict = kg.to_dict()
        logger.info(f"KG built in {elapsed:.1f}s: {kg_dict['stats']['num_nodes']} nodes, "
                     f"{kg_dict['stats']['num_relationships']} relationships")

        return {
            "knowledge_graph": kg_dict,
            "phase_timings": [{"phase": "knowledge_graph", "duration": round(elapsed, 2)}],
        }
    except Exception as e:
        logger.error(f"Knowledge graph build failed (non-fatal): {e}")
        return {"knowledge_graph": {}, "phase_timings": [{"phase": "knowledge_graph", "duration": 0.0}]}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline_graph():
    """Build the LangGraph StateGraph for chatbot synthetic data generation.

    If multi-hop is enabled, inserts the multihop_generator between
    question_generator and answer_generator using conditional routing.
    """
    logger.set_step("build_graph")
    settings = get_settings()
    multihop_enabled = settings.enable_multihop
    kg_enabled = settings.enable_knowledge_graph

    agent_count = 5 + (1 if multihop_enabled else 0) + (1 if kg_enabled else 0)
    logger.info(f"Building LangGraph pipeline ({agent_count}-Agent, multihop={'ON' if multihop_enabled else 'OFF'}, KG={'ON' if kg_enabled else 'OFF'})...")

    workflow = StateGraph(GraphState)

    # ─── Add nodes ────────────────────────────────────────────────────────
    workflow.add_node("semantic_chunker", semantic_chunker_node)
    if kg_enabled:
        workflow.add_node("knowledge_graph_builder", knowledge_graph_node)
    workflow.add_node("question_generator", question_generator_node)
    if multihop_enabled:
        workflow.add_node("multihop_generator", multihop_generator_node)
    workflow.add_node("deduplicator", deduplicator_node)
    workflow.add_node("answer_generator", answer_generator_node)
    workflow.add_node("quality_validator", quality_validator_node)

    logger.info(f"Nodes added: {agent_count} agents")

    # ─── Edges ────────────────────────────────────────────────────────────
    workflow.add_edge(START, "semantic_chunker")

    if kg_enabled:
        workflow.add_edge("semantic_chunker", "knowledge_graph_builder")
        workflow.add_edge("knowledge_graph_builder", "question_generator")
    else:
        workflow.add_edge("semantic_chunker", "question_generator")

    if multihop_enabled:
        workflow.add_edge("question_generator", "multihop_generator")
        workflow.add_edge("multihop_generator", "deduplicator")
    else:
        workflow.add_edge("question_generator", "deduplicator")

    workflow.add_edge("deduplicator", "answer_generator")
    workflow.add_edge("answer_generator", "quality_validator")
    workflow.add_edge("quality_validator", END)

    flow_parts = ["START", "chunker"]
    if kg_enabled:
        flow_parts.append("KG")
    flow_parts.append("questions")
    if multihop_enabled:
        flow_parts.append("multihop")
    flow_parts.extend(["dedup", "answers", "quality", "END"])
    logger.info(f"Edges: {' → '.join(flow_parts)}")

    compiled = workflow.compile()
    logger.info("Graph compiled successfully")

    return compiled


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

class Supervisor:
    """Supervisor Agent: Orchestrates the LangGraph pipeline."""

    def __init__(self):
        setup_logging()
        self.settings = get_settings()
        self.graph = build_pipeline_graph()

    async def execute_pipeline(
        self,
        document_path: str | None = None,
        document_paths: list[str] | None = None,
        quality_threshold: float = 0.7,
        max_retries: int = 3,
        progress: PipelineProgress | None = None,
        **kwargs,
    ) -> dict:
        """Execute the full LangGraph pipeline."""
        run_id = str(uuid4())

        try:
            with logger.phase(f"Pipeline Execution (run_id={run_id[:8]}...)"):

                # --- Step 1: Create initial state ---
                with logger.step("Initialize pipeline state"):
                    try:
                        initial_state = create_initial_state(
                            run_id=run_id,
                            document_path=document_path or "",
                            document_paths=document_paths or [],
                            quality_threshold=quality_threshold,
                            max_retries=max_retries,
                        )
                    except Exception as e:
                        logger.error(f"Failed to create initial state: {e}")
                        raise RuntimeError(f"Pipeline state initialization failed: {e}") from e

                    logger.info(f"Run ID: {run_id}")
                    logger.info(f"Config: threshold={quality_threshold}, max_retries={max_retries}")
                    if document_paths:
                        logger.info(f"Documents: {len(document_paths)} file(s)")
                        for dp in document_paths:
                            logger.info(f"  - {dp}")
                    elif document_path:
                        logger.info(f"Document: {document_path}")
                    else:
                        logger.info("Document: auto-detect from data/ directory")

                # --- Step 2: Execute the graph ---
                with logger.step("Execute LangGraph"):
                    logger.info("Invoking compiled StateGraph...")
                    start_time = time.time()

                    if progress:
                        progress.log_message("Starting pipeline execution...")

                    try:
                        final_state = None

                        # Build phase_map with sequential numbering based on active nodes
                        phase_list = [("semantic_chunker", "Semantic Chunking")]
                        if settings.enable_knowledge_graph:
                            phase_list.append(("knowledge_graph_builder", "Knowledge Graph"))
                        phase_list.append(("question_generator", "Question Generation"))
                        if settings.enable_multihop:
                            phase_list.append(("multihop_generator", "Multi-Hop Generation"))
                        phase_list.append(("deduplicator", "Deduplication"))
                        phase_list.append(("answer_generator", "Answer Generation"))
                        phase_list.append(("quality_validator", "Quality Validation"))

                        phase_map = {node: (label, idx + 1) for idx, (node, label) in enumerate(phase_list)}
                        total_phases = len(phase_list)

                        if progress:
                            progress.total_phases = total_phases

                        async for event in self.graph.astream(initial_state, stream_mode="updates"):
                            for node_name, node_output in event.items():
                                if node_name in phase_map and progress:
                                    phase_name, phase_num = phase_map[node_name]
                                    progress.phase_complete(phase_name, phase_num, {
                                        "node": node_name,
                                        "keys_updated": list(node_output.keys()) if isinstance(node_output, dict) else [],
                                    })
                                if isinstance(node_output, dict):
                                    if final_state is None:
                                        final_state = dict(initial_state)
                                    final_state.update(node_output)

                        if final_state is None:
                            raise RuntimeError("Pipeline produced no output from any node")

                        total_time = time.time() - start_time
                        logger.info(f"Graph execution completed in {total_time:.1f}s")

                    except Exception as e:
                        total_time = time.time() - start_time
                        logger.error(f"Graph execution FAILED after {total_time:.1f}s: {e}")
                        if progress:
                            progress.pipeline_failed(str(e))
                        raise

                # --- Step 3: Build response with conversational format ---
                with logger.step("Build response"):
                    try:
                        validated = final_state.get("validated_triples", [])
                        quality = final_state.get("quality", {})
                        phase_timings = final_state.get("phase_timings", [])

                        # Build conversations (ShareGPT/OpenAI training format)
                        conversations = []
                        for t in validated:
                            messages = [
                                {"role": "user", "content": t.get("question", "")},
                                {"role": "assistant", "content": t.get("answer", "")},
                            ]
                            if t.get("follow_up_q") and t.get("follow_up_a"):
                                messages.append({"role": "user", "content": t["follow_up_q"]})
                                messages.append({"role": "assistant", "content": t["follow_up_a"]})

                            retrieved_sources = t.get("retrieved_sources", [])
                            is_cross_doc = len(set(retrieved_sources)) > 1

                            conv = {
                                "id": f"{run_id[:8]}-{len(conversations)}",
                                "messages": messages,
                                "metadata": {
                                    "source_chunk_id": t.get("source_chunk_id", -1),
                                    "source_chunk_ids": t.get("source_chunk_ids", []),
                                    "source_file": retrieved_sources[0] if retrieved_sources else "",
                                    "question_type": t.get("question_type", ""),
                                    "reasoning_type": t.get("reasoning_type", ""),
                                    "hop_count": t.get("hop_count", 1),
                                    "difficulty": t.get("difficulty", ""),
                                    "quality_score": t.get("quality_score", 0.0),
                                    "validation_scores": t.get("validation_scores", {}),
                                    "retrieved_chunk_ids": t.get("retrieved_chunk_ids", []),
                                    "retrieved_sources": retrieved_sources,
                                    "retrieval_details": t.get("retrieval_details", []),
                                    "cross_document": is_cross_doc,
                                },
                            }
                            conversations.append(conv)

                        cross_doc_count = sum(1 for c in conversations if c["metadata"]["cross_document"])
                        multihop_count = sum(1 for c in conversations if c["metadata"]["question_type"] == "multihop")
                        doc_paths_used = final_state.get("document_paths", [])

                        response = {
                            "run_id": run_id,
                            "status": "completed",
                            "started_at": initial_state["started_at"],
                            "completed_at": datetime.now().isoformat(),
                            "total_time_seconds": round(total_time, 2),
                            "format": "conversational",
                            "conversations": conversations,
                            "qa_triples": validated,
                            "quality": quality,
                            "phase_timings": phase_timings,
                            "retrieval_method": "hybrid (FAISS dense + BM25 sparse with RRF)",
                            "dedup_info": {
                                "duplicates_removed": final_state.get("duplicates_removed", 0),
                                "before_dedup": len(final_state.get("qa_pairs", [])) + final_state.get("duplicates_removed", 0),
                                "after_dedup": len(final_state.get("qa_pairs", [])),
                            },
                            "summary": {
                                "documents_processed": len(doc_paths_used),
                                "document_names": [Path(p).name for p in doc_paths_used],
                                "total_chunks": len(final_state.get("chunks", [])),
                                "total_questions_generated": len(final_state.get("qa_pairs", [])),
                                "total_answers_generated": len(final_state.get("qa_triples", [])),
                                "after_dedup": len(final_state.get("deduplicated_triples", [])),
                                "total_triples": len(validated),
                                "total_conversations": len(conversations),
                                "cross_document_conversations": cross_doc_count,
                                "multihop_conversations": multihop_count,
                                "multihop_pairs_found": len(final_state.get("multihop_pairs", [])),
                                "rejected": len(final_state.get("rejected_triples", [])),
                                "avg_quality_score": quality.get("avg_quality_score", 0.0),
                            },
                            "errors": final_state.get("errors", []),
                            "advanced_metrics": final_state.get("advanced_metrics_summary", {}),
                            "knowledge_graph_stats": final_state.get("knowledge_graph", {}).get("stats", {}),
                            "cost_summary": final_state.get("cost_summary", {}),
                        }

                    except Exception as e:
                        logger.error(f"Failed to build response: {e}")
                        raise RuntimeError(f"Response building failed: {e}") from e

                    logger.info("=" * 60)
                    logger.info("PIPELINE EXECUTION SUMMARY")
                    logger.info("=" * 60)
                    logger.info(f"  Run ID: {run_id}")
                    logger.info(f"  Status: COMPLETED")
                    logger.info(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
                    logger.info(f"  Documents: {response['summary']['documents_processed']}")
                    logger.info(f"  Chunks: {response['summary']['total_chunks']}")
                    logger.info(f"  Questions generated: {response['summary']['total_questions_generated']}")
                    logger.info(f"  Answers generated: {response['summary']['total_answers_generated']}")
                    logger.info(f"  After dedup: {response['summary']['after_dedup']}")
                    logger.info(f"  Final conversations: {response['summary']['total_conversations']}")
                    logger.info(f"  Cross-document: {response['summary']['cross_document_conversations']}")
                    logger.info(f"  Multi-hop: {response['summary']['multihop_conversations']}")
                    logger.info(f"  Rejected: {response['summary']['rejected']}")
                    logger.info(f"  Avg quality: {response['summary']['avg_quality_score']:.3f}")
                    logger.info(f"  Retrieval: hybrid (FAISS + BM25)")
                    logger.info("=" * 60)

                    if progress:
                        progress.pipeline_complete(response)

                    return response

        except Exception as e:
            logger.error(f"Pipeline execution failed (run_id={run_id[:8]}): {e}")
            raise
