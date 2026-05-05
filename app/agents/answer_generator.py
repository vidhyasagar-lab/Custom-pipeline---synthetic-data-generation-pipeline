"""Agent 3: RAG Answer Generator (LangGraph Node)
For each question, uses hybrid retrieval (FAISS dense + BM25 sparse) across
all chunks from all documents, then generates a grounded chatbot-style answer.
Supports cross-document retrieval when multiple documents are uploaded.
"""

import json
import re
import time
import asyncio

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.core.llm import get_answer_llm, get_azure_embeddings
from app.core.logging_config import get_agent_logger
from app.core.graph_state import GraphState
from app.models.llm_responses import AnswerResponse, parse_llm_response

logger = get_agent_logger("Agent3:AnswerGen")

# Common stop words to filter from query expansion
_STOP_WORDS = frozenset({
    "what", "how", "why", "when", "where", "who", "which", "can", "could",
    "would", "should", "does", "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "did", "will", "shall",
    "may", "might", "must", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "about", "into", "through", "during", "before", "after", "above",
    "between", "and", "but", "or", "not", "no", "if", "then", "so", "that",
    "this", "it", "its", "you", "your", "me", "my", "we", "our", "they",
    "their", "tell", "explain", "describe", "hey", "please", "thanks",
    "difference", "mean", "work", "works", "know", "thing", "things",
})


def _extract_key_terms(question: str) -> list[str]:
    """Extract key terms from a question for BM25 query expansion.
    
    Filters out common stop words and short terms, keeping domain-specific
    keywords that improve lexical retrieval.
    """
    tokens = re.findall(r'\w+', question.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]

ANSWER_PROMPT = """You are a helpful chatbot assistant. Answer the user's question based ONLY on the provided context. Be concise, natural, and helpful — like a real chatbot conversation.

CRITICAL CONSTRAINT: You MUST use ONLY the information in the CONTEXT below. Do NOT use your own knowledge, training data, or any external information under any circumstances. If the context does not contain the answer, say so — do NOT fill gaps with outside knowledge. Every fact, detail, and claim in your answer must come directly from the provided context.

CONTEXT (retrieved from source documents):
{context}

USER QUESTION: {question}

FOLLOW-UP QUESTION: {follow_up_q}

Instructions:
1. Answer the main question in a helpful, conversational chatbot tone
2. Keep the answer concise but complete (2-5 sentences typically)
3. Use ONLY facts and details explicitly stated in the context above — do NOT add, infer, or supplement with your own knowledge
4. If the context doesn't contain enough information, say "Based on the available information, I don't have enough details to fully answer that." Do NOT guess or fill in from your training data.
5. Also answer the follow-up question based strictly on the same context
6. Do NOT make up information not in the context

Respond ONLY with a JSON object:
{{
  "answer": "your answer to the main question",
  "follow_up_a": "your answer to the follow-up question"
}}

Return ONLY valid JSON, no other text."""


def _tokenize(text: str) -> list[str]:
    """Tokenizer for BM25 using NLTK word_tokenize for better accuracy."""
    try:
        from nltk.tokenize import word_tokenize
        return [w.lower() for w in word_tokenize(text) if w.isalnum()]
    except Exception:
        return re.findall(r'\w+', text.lower())


class HybridIndex:
    """FAISS (dense) + BM25 (sparse) hybrid retrieval index.

    Combines dense embedding similarity with BM25 keyword matching
    using Reciprocal Rank Fusion (RRF) for score aggregation.
    Works across all chunks from all documents.
    """

    def __init__(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        faiss_index: faiss.IndexFlatIP,
        bm25: BM25Okapi,
    ):
        self.chunks = chunks
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.bm25 = bm25

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        rrf_k: int = 60,
        expanded_keywords: list[str] | None = None,
    ) -> list[dict]:
        """Hybrid search: FAISS dense + BM25 sparse with RRF fusion.
        
        If expanded_keywords are provided, they are appended to the BM25 query
        for improved lexical recall (query expansion).
        """
        try:
            settings = get_settings()
            dense_weight = dense_weight or settings.dense_weight
            sparse_weight = sparse_weight or settings.sparse_weight
            n_chunks = len(self.chunks)
            if n_chunks == 0:
                logger.warning("HybridIndex.search called with empty index")
                return []

            fetch_k = min(n_chunks, top_k * 3)

            # --- Dense retrieval (FAISS) ---
            try:
                query_vec = np.array([query_embedding], dtype=np.float32)
                faiss.normalize_L2(query_vec)
                dense_scores, dense_indices = self.faiss_index.search(query_vec, fetch_k)
                dense_scores = dense_scores[0]
                dense_indices = dense_indices[0]
            except Exception as e:
                logger.error(f"FAISS dense search failed: {e}")
                dense_scores, dense_indices = np.array([]), np.array([])

            # Build dense rank map
            dense_rank = {}
            for rank, idx in enumerate(dense_indices):
                if idx >= 0:
                    dense_rank[int(idx)] = rank + 1

            # --- Sparse retrieval (BM25) ---
            try:
                query_tokens = _tokenize(query_text)
                if expanded_keywords:
                    query_tokens.extend([k.lower() for k in expanded_keywords if k.isalnum()])
                bm25_scores = self.bm25.get_scores(query_tokens)
                bm25_ranked = np.argsort(bm25_scores)[::-1][:fetch_k]
            except Exception as e:
                logger.error(f"BM25 sparse search failed: {e}")
                bm25_scores = np.zeros(n_chunks)
                bm25_ranked = np.array([])

            sparse_rank = {}
            for rank, idx in enumerate(bm25_ranked):
                if bm25_scores[idx] > 0:
                    sparse_rank[int(idx)] = rank + 1

            # --- Reciprocal Rank Fusion ---
            all_candidates = set(dense_rank.keys()) | set(sparse_rank.keys())
            if not all_candidates:
                logger.warning("No candidates from either dense or sparse search")
                return []

            rrf_scores = {}

            for idx in all_candidates:
                score = 0.0
                if idx in dense_rank:
                    score += dense_weight * (1.0 / (rrf_k + dense_rank[idx]))
                if idx in sparse_rank:
                    score += sparse_weight * (1.0 / (rrf_k + sparse_rank[idx]))
                rrf_scores[idx] = score

            sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            top_results = sorted_candidates[:top_k]

            return [
                {
                    "chunk": self.chunks[idx],
                    "rrf_score": round(score, 6),
                    "dense_rank": dense_rank.get(idx),
                    "sparse_rank": sparse_rank.get(idx),
                }
                for idx, score in top_results
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed unexpectedly: {e}")
            return []


async def build_hybrid_index(chunks: list[dict], precomputed_embeddings: list[list[float]] | None = None) -> HybridIndex:
    """Build a FAISS + BM25 hybrid index from all chunks (cross-document).
    
    If precomputed_embeddings are provided (from Agent 1), reuses them instead
    of re-embedding all chunks — saving API calls and time.
    """
    try:
        logger.set_step("build_index")

        texts = [c["page_content"] for c in chunks]

        if not texts:
            raise ValueError("No chunk texts to build index from")

        # --- Dense embeddings (reuse if available) ---
        if precomputed_embeddings and len(precomputed_embeddings) == len(chunks):
            logger.info(f"Reusing {len(precomputed_embeddings)} pre-computed chunk embeddings from Agent 1")
            all_embeddings = precomputed_embeddings
        else:
            if precomputed_embeddings:
                logger.warning(f"Embedding count mismatch ({len(precomputed_embeddings)} vs {len(chunks)}), re-embedding")
            
            embeddings_model = get_azure_embeddings()
            logger.info(f"Building FAISS index over {len(chunks)} chunks (fresh embeddings)...")
            all_embeddings = []
            batch_size = 50
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

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
                    raise RuntimeError(f"FAISS embedding failed: {e}") from e

        emb_array = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_array)

        dim = emb_array.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(emb_array)
        logger.info(f"  FAISS index built: {faiss_index.ntotal} vectors, dim={dim}")

        # --- BM25 index ---
        logger.info("Building BM25 index...")
        tokenized_corpus = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"  BM25 index built: {len(tokenized_corpus)} documents")

        # Log cross-document stats
        doc_sources = set()
        for c in chunks:
            src = c.get("metadata", {}).get("source_file", "unknown")
            doc_sources.add(src)
        logger.info(f"  Cross-document index covers {len(doc_sources)} document(s): {sorted(doc_sources)}")

        return HybridIndex(chunks, emb_array, faiss_index, bm25)

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Failed to build hybrid index: {e}")
        raise RuntimeError(f"Hybrid index build failed: {e}") from e


async def generate_answer(
    llm,
    embeddings_model,
    hybrid_index: HybridIndex,
    qa_pair: dict,
    top_k: int = 5,
    max_retries: int = 2,
) -> dict | None:
    """Generate an answer for a single QA pair using hybrid retrieval."""
    question = qa_pair.get("question", "")
    follow_up_q = qa_pair.get("follow_up_q", "")

    try:
        if not question:
            logger.warning("Empty question received, skipping")
            return None

        # Hybrid retrieval
        try:
            query_emb = await embeddings_model.aembed_query(question)
        except Exception as e:
            logger.error(f"  Query embedding failed: {e}")
            return None

        try:
            results = hybrid_index.search(
                query_emb, query_text=question, top_k=top_k,
                expanded_keywords=_extract_key_terms(question),
            )
        except Exception as e:
            logger.error(f"  Hybrid search failed: {e}")
            return None

        if not results:
            logger.warning(f"  No retrieval results for question: {question[:60]}...")
            return None

        # Build context with source attribution
        context_parts = []
        for r in results:
            src = r["chunk"]["metadata"].get("source_file", "unknown")
            text = r["chunk"]["page_content"]
            context_parts.append(f"[Source: {src}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        retrieved_chunk_ids = [r["chunk"]["metadata"]["chunk_id"] for r in results]
        retrieved_sources = [r["chunk"]["metadata"].get("source_file", "unknown") for r in results]

        prompt = ANSWER_PROMPT.format(
            context=context[:8000],
            question=question,
            follow_up_q=follow_up_q or "No follow-up question.",
        )

        for attempt in range(max_retries + 1):
            try:
                response = await llm.ainvoke(prompt)
                content = response.content.strip()

                parsed_response = parse_llm_response(content, AnswerResponse)
                if parsed_response is None:
                    raise ValueError("Failed to parse/validate answer response")

                return {
                    "question": question,
                    "answer": parsed_response.answer,
                    "follow_up_q": follow_up_q,
                    "follow_up_a": parsed_response.follow_up_a,
                    "context": [r["chunk"]["page_content"] for r in results],
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_sources": retrieved_sources,
                    "retrieval_scores": [r["rrf_score"] for r in results],
                    "retrieval_details": [
                        {
                            "chunk_id": r["chunk"]["metadata"]["chunk_id"],
                            "source_file": r["chunk"]["metadata"].get("source_file", "unknown"),
                            "rrf_score": r["rrf_score"],
                            "dense_rank": r["dense_rank"],
                            "sparse_rank": r["sparse_rank"],
                        }
                        for r in results
                    ],
                    "question_type": qa_pair.get("question_type", "unknown"),
                    "difficulty": qa_pair.get("difficulty", "unknown"),
                    "source_chunk_id": qa_pair.get("source_chunk_id", -1),
                }

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    logger.warning(f"  Parse error (attempt {attempt + 1}): {e}")
                    continue
                logger.error(f"  Answer parse failed after {max_retries + 1} attempts for: {question[:60]}...")
                return None

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"  LLM error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error(f"  Answer gen failed for question: {e}")
                return None

    except Exception as e:
        logger.error(f"  Unexpected error generating answer: {e}")
        return None


async def answer_generator_node(state: GraphState) -> dict:
    """LangGraph node: Generate RAG-grounded answers using hybrid retrieval."""
    try:
        with logger.phase("RAG Answer Generation (Hybrid)"):

            settings = get_settings()
            chunks = state.get("chunks", [])
            qa_pairs = state.get("qa_pairs", [])

            if not qa_pairs:
                raise ValueError("No questions to answer. Run Question Generator first.")
            if not chunks:
                raise ValueError("No chunks available for retrieval. Run Semantic Chunker first.")

            logger.info(f"Questions to answer: {len(qa_pairs)}")
            logger.info(f"Chunks for retrieval: {len(chunks)}")

            # Log cross-document info
            doc_sources = set(c.get("metadata", {}).get("source_file", "?") for c in chunks)
            logger.info(f"Documents in index: {len(doc_sources)} — {sorted(doc_sources)}")

            # --- Build hybrid index (FAISS + BM25) ---
            # Get pre-computed embeddings from Agent 1 (avoids re-embedding)
            chunk_embeddings = state.get("chunk_embeddings", [])

            with logger.step("Build hybrid index"):
                try:
                    start = time.time()
                    hybrid_index = await build_hybrid_index(chunks, precomputed_embeddings=chunk_embeddings)
                    index_time = time.time() - start
                    logger.info(f"Hybrid index built in {index_time:.1f}s ({len(chunks)} chunks)")
                except Exception as e:
                    logger.error(f"Failed to build hybrid index: {e}")
                    raise RuntimeError(f"Hybrid index build failed: {e}") from e

            # --- Generate answers ---
            with logger.step("Generate answers"):
                try:
                    llm = get_answer_llm()
                    embeddings_model = get_azure_embeddings()
                    logger.info("Using answer LLM (temperature=0.0 for faithfulness)")
                except Exception as e:
                    logger.error(f"Failed to initialize LLM/embeddings: {e}")
                    raise RuntimeError(f"LLM/Embeddings initialization failed: {e}") from e

                start = time.time()
                all_triples = []
                failed = 0
                batch_size = settings.max_concurrent_calls

                for i in range(0, len(qa_pairs), batch_size):
                    batch = qa_pairs[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(qa_pairs) - 1) // batch_size + 1

                    logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} questions)...")

                    tasks = [
                        generate_answer(llm, embeddings_model, hybrid_index, qa)
                        for qa in batch
                    ]

                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                    except Exception as e:
                        logger.error(f"  Batch {batch_num} gather failed: {e}")
                        failed += len(batch)
                        continue

                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"  Error: {result}")
                            failed += 1
                        elif result is None:
                            failed += 1
                        else:
                            all_triples.append(result)

                    logger.info(f"  Running total: {len(all_triples)} answers generated")

                    if i + batch_size < len(qa_pairs):
                        await asyncio.sleep(1)

                elapsed = time.time() - start

            # --- Summary ---
            with logger.step("Summary"):
                logger.info(f"Answer generation complete in {elapsed:.1f}s")
                logger.info(f"  Total answers: {len(all_triples)}")
                logger.info(f"  Failed: {failed}")
                logger.info(f"  Success rate: {len(all_triples) / max(len(qa_pairs), 1):.1%}")

                cross_doc = 0
                for t in all_triples:
                    sources = set(t.get("retrieved_sources", []))
                    if len(sources) > 1:
                        cross_doc += 1
                if len(doc_sources) > 1:
                    logger.info(f"  Cross-document retrievals: {cross_doc}/{len(all_triples)}")

            return {
                "qa_triples": all_triples,
                "phase_timings": [{
                    "phase": "answer_generation",
                    "retrieval_method": "hybrid (FAISS + BM25 with RRF)",
                    "total_answers": len(all_triples),
                    "failed": failed,
                    "cross_document_retrievals": cross_doc,
                    "index_time_seconds": round(index_time, 2),
                    "generation_time_seconds": round(elapsed, 2),
                }],
            }

    except Exception as e:
        logger.error(f"Answer generator node failed: {e}")
        raise
