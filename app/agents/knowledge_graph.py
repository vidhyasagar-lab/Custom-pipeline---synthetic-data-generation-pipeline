"""Knowledge Graph builder for semantic chunk relationships.

Extracts entities and keyphrases from chunks, builds a graph of relationships
using Jaccard similarity on shared entities. Used by the multi-hop generator
to find principled cross-chunk connections instead of relying on raw embedding
similarity alone.

Inspired by RAGAS testset generation architecture:
  Documents → Nodes (with extracted properties) → Relationships → KG
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.llm import get_azure_llm
from app.core.logging_config import get_agent_logger

logger = get_agent_logger("KnowledgeGraph")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KGNode:
    """A node in the knowledge graph (corresponds to a chunk)."""
    chunk_id: int
    source_file: str = ""
    entities: list[str] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    page_content: str = ""
    # Hierarchical info
    parent_section: str = ""
    heading_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "entities": self.entities,
            "keyphrases": self.keyphrases,
            "topics": self.topics,
            "parent_section": self.parent_section,
            "heading_path": self.heading_path,
        }


@dataclass
class KGRelationship:
    """An edge between two KG nodes."""
    source_id: int
    target_id: int
    relationship_type: str  # "shared_entity", "shared_keyphrase", "hierarchical", "semantic"
    shared_properties: list[str] = field(default_factory=list)
    strength: float = 0.0  # Jaccard similarity or other metric

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "shared_properties": self.shared_properties,
            "strength": self.strength,
        }


@dataclass
class KnowledgeGraph:
    """Knowledge graph built from document chunks."""
    nodes: dict[int, KGNode] = field(default_factory=dict)
    relationships: list[KGRelationship] = field(default_factory=list)

    def add_node(self, node: KGNode):
        self.nodes[node.chunk_id] = node

    def add_relationship(self, rel: KGRelationship):
        self.relationships.append(rel)

    def get_node(self, chunk_id: int) -> KGNode | None:
        return self.nodes.get(chunk_id)

    def get_relationships_for(self, chunk_id: int) -> list[KGRelationship]:
        return [r for r in self.relationships
                if r.source_id == chunk_id or r.target_id == chunk_id]

    def get_multi_hop_pairs(
        self,
        min_strength: float = 0.1,
        max_strength: float = 0.8,
        max_pairs: int = 50,
        prefer_cross_document: bool = True,
    ) -> list[tuple[int, int, float]]:
        """Get chunk pairs suitable for multi-hop questions.

        Filters by relationship strength band and prioritizes cross-document
        pairs for richer multi-hop synthesis.

        Returns list of (source_id, target_id, strength) tuples.
        """
        candidates = []
        for rel in self.relationships:
            if min_strength <= rel.strength <= max_strength:
                candidates.append((rel.source_id, rel.target_id, rel.strength))

        if prefer_cross_document:
            cross_doc = []
            same_doc = []
            for src, tgt, strength in candidates:
                src_node = self.nodes.get(src)
                tgt_node = self.nodes.get(tgt)
                if src_node and tgt_node and src_node.source_file != tgt_node.source_file:
                    cross_doc.append((src, tgt, strength))
                else:
                    same_doc.append((src, tgt, strength))
            # Prioritize cross-doc, fill rest with same-doc
            result = cross_doc[:max_pairs]
            remaining = max_pairs - len(result)
            if remaining > 0:
                result.extend(same_doc[:remaining])
            return result

        # Sort by strength (middle of band is best for multi-hop)
        mid = (min_strength + max_strength) / 2
        candidates.sort(key=lambda x: -abs(x[2] - mid))
        return candidates[:max_pairs]

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Reconstruct a KnowledgeGraph from its serialized dict form."""
        kg = cls()
        for chunk_id_str, node_data in data.get("nodes", {}).items():
            node = KGNode(
                chunk_id=int(chunk_id_str) if str(chunk_id_str).isdigit() else chunk_id_str,
                source_file=node_data.get("source_file", ""),
                entities=node_data.get("entities", []),
                keyphrases=node_data.get("keyphrases", []),
                topics=node_data.get("topics", []),
                parent_section=node_data.get("parent_section", ""),
                heading_path=node_data.get("heading_path", []),
            )
            kg.add_node(node)
        for rel_data in data.get("relationships", []):
            rel = KGRelationship(
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                relationship_type=rel_data.get("relationship_type", ""),
                shared_properties=rel_data.get("shared_properties", []),
                strength=rel_data.get("strength", 0.0),
            )
            kg.add_relationship(rel)
        return kg

    def to_dict(self) -> dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "stats": {
                "num_nodes": len(self.nodes),
                "num_relationships": len(self.relationships),
                "relationship_types": dict(self._type_counts()),
            },
        }

    def _type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.relationships:
            counts[r.relationship_type] = counts.get(r.relationship_type, 0) + 1
        return counts


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY / KEYPHRASE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_PROMPT = """Extract named entities and key phrases from this text.

TEXT:
{text}

Return a JSON object with:
{{
  "entities": ["entity1", "entity2", ...],
  "keyphrases": ["phrase1", "phrase2", ...],
  "topics": ["topic1", "topic2", ...]
}}

Rules:
- entities: Named entities (people, organizations, products, technologies, standards, locations, dates)
- keyphrases: Important technical terms, concepts, or multi-word expressions
- topics: High-level topic categories (1-3 words each)
- Normalize casing (title case for entities, lowercase for keyphrases/topics)
- Remove duplicates
- Maximum 20 entities, 15 keyphrases, 5 topics

Return ONLY valid JSON."""


async def extract_properties(llm, chunk: dict, max_retries: int = 2) -> dict:
    """Extract entities, keyphrases, and topics from a single chunk using LLM."""
    text = chunk.get("page_content", "")
    if len(text.strip()) < 50:
        return {"entities": [], "keyphrases": [], "topics": []}

    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:3000])

    for attempt in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # Strip markdown fences
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            import json
            data = json.loads(content)
            return {
                "entities": [str(e).strip() for e in data.get("entities", []) if e],
                "keyphrases": [str(k).strip().lower() for k in data.get("keyphrases", []) if k],
                "topics": [str(t).strip().lower() for t in data.get("topics", []) if t],
            }
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.warning(f"Entity extraction failed for chunk {chunk.get('metadata', {}).get('chunk_id', '?')}: {e}")
            return {"entities": [], "keyphrases": [], "topics": []}


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIONSHIP BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def build_relationships(kg: KnowledgeGraph, min_similarity: float = 0.05) -> None:
    """Build relationships between KG nodes based on shared properties.

    Creates edges for:
    - shared_entity: Nodes sharing named entities (Jaccard on entity sets)
    - shared_keyphrase: Nodes sharing keyphrases (Jaccard on keyphrase sets)
    - hierarchical: Nodes from same document section
    """
    node_ids = list(kg.nodes.keys())
    n = len(node_ids)

    for i in range(n):
        for j in range(i + 1, n):
            node_a = kg.nodes[node_ids[i]]
            node_b = kg.nodes[node_ids[j]]

            # Entity-based relationship
            entities_a = {e.lower() for e in node_a.entities}
            entities_b = {e.lower() for e in node_b.entities}
            entity_sim = _jaccard_similarity(entities_a, entities_b)

            if entity_sim >= min_similarity:
                shared = sorted(entities_a & entities_b)
                kg.add_relationship(KGRelationship(
                    source_id=node_ids[i],
                    target_id=node_ids[j],
                    relationship_type="shared_entity",
                    shared_properties=shared,
                    strength=entity_sim,
                ))

            # Keyphrase-based relationship
            kp_a = set(node_a.keyphrases)
            kp_b = set(node_b.keyphrases)
            kp_sim = _jaccard_similarity(kp_a, kp_b)

            if kp_sim >= min_similarity:
                shared_kp = sorted(kp_a & kp_b)
                kg.add_relationship(KGRelationship(
                    source_id=node_ids[i],
                    target_id=node_ids[j],
                    relationship_type="shared_keyphrase",
                    shared_properties=shared_kp,
                    strength=kp_sim,
                ))

            # Hierarchical (same section in same document)
            if (node_a.source_file == node_b.source_file
                    and node_a.parent_section
                    and node_a.parent_section == node_b.parent_section):
                kg.add_relationship(KGRelationship(
                    source_id=node_ids[i],
                    target_id=node_ids[j],
                    relationship_type="hierarchical",
                    shared_properties=[node_a.parent_section],
                    strength=0.5,
                ))


# ═══════════════════════════════════════════════════════════════════════════════
# KG CONSTRUCTION (MAIN ENTRY POINT)
# ═══════════════════════════════════════════════════════════════════════════════

async def build_knowledge_graph(chunks: list[dict]) -> KnowledgeGraph:
    """Build a knowledge graph from document chunks.

    1. Extract entities/keyphrases per chunk (LLM-based, parallel)
    2. Build relationships using Jaccard similarity on shared properties
    3. Return KG with nodes and edges

    This KG replaces raw embedding similarity for multi-hop pair selection.
    """
    logger.set_step("build_kg")
    start = time.time()

    settings = get_settings()
    kg = KnowledgeGraph()
    llm = get_azure_llm()

    logger.info(f"Building knowledge graph from {len(chunks)} chunks...")

    # --- Step 1: Extract properties in parallel batches ---
    batch_size = settings.max_concurrent_calls
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks = [extract_properties(llm, chunk) for chunk in batch]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"KG extraction batch failed: {e}")
            results = [{"entities": [], "keyphrases": [], "topics": []}] * len(batch)

        for chunk, result in zip(batch, results):
            if isinstance(result, Exception):
                result = {"entities": [], "keyphrases": [], "topics": []}

            metadata = chunk.get("metadata", {})
            node = KGNode(
                chunk_id=metadata.get("chunk_id", i),
                source_file=metadata.get("source_file", ""),
                entities=result.get("entities", []),
                keyphrases=result.get("keyphrases", []),
                topics=result.get("topics", []),
                page_content=chunk.get("page_content", "")[:200],
                parent_section=metadata.get("parent_section", ""),
                heading_path=metadata.get("heading_path", []),
            )
            kg.add_node(node)

        logger.info(f"  Extracted properties for chunks {i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)}")

        if i + batch_size < len(chunks):
            await asyncio.sleep(0.5)

    # --- Step 2: Build relationships ---
    logger.info("Building relationships from shared entities/keyphrases...")
    build_relationships(kg)

    elapsed = time.time() - start
    stats = kg.to_dict()["stats"]
    logger.info(f"Knowledge graph built in {elapsed:.1f}s: "
                f"{stats['num_nodes']} nodes, {stats['num_relationships']} relationships")
    logger.info(f"  Relationship types: {stats['relationship_types']}")

    return kg
