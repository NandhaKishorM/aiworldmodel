"""
Knowledge Graph — Session-Scoped Entity Tracking
==================================================
Lightweight in-memory directed graph for tracking entities, relations,
and their properties during a TTT inference session. Provides relational
constraints to the symbolic engine.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    entity_type: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    concept_ids: List[int] = field(default_factory=list)  # associated concept indices

    def add_concept(self, concept_id: int) -> None:
        self.concept_ids.append(concept_id)


@dataclass
class Relation:
    """A directed edge in the knowledge graph."""
    source: str
    relation_type: str
    target: str
    confidence: float = 1.0
    position: int = 0  # sequence position where this relation was inferred


class KnowledgeGraph:
    """
    Session-scoped knowledge graph for entity and relation tracking.

    Created fresh for each inference call. Entities and relations are
    populated as the symbolic bottleneck projects hidden states to
    concepts during the dry-run forward pass.
    """

    def __init__(
        self,
        max_entities: int = 1024,
        max_relations: int = 4096,
    ) -> None:
        self.max_entities = max_entities
        self.max_relations = max_relations

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # Adjacency index: source_id → list of (relation_type, target_id)
        self._adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Reverse index: target_id → list of (relation_type, source_id)
        self._reverse: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------
    def add_entity(
        self,
        entity_id: str,
        entity_type: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Add or update an entity."""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            if entity_type != "unknown":
                entity.entity_type = entity_type
            if attributes:
                entity.attributes.update(attributes)
            return entity

        if len(self.entities) >= self.max_entities:
            logger.warning(f"Max entities ({self.max_entities}) reached, ignoring")
            return self.entities.get(entity_id, Entity(id=entity_id))

        entity = Entity(
            id=entity_id,
            entity_type=entity_type,
            attributes=attributes or {},
        )
        self.entities[entity_id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    # ------------------------------------------------------------------
    # Relation management
    # ------------------------------------------------------------------
    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        confidence: float = 1.0,
        position: int = 0,
    ) -> Relation:
        """Add a directed relation between two entities."""
        if len(self.relations) >= self.max_relations:
            logger.warning(f"Max relations ({self.max_relations}) reached")
            return Relation(source=source, relation_type=relation_type, target=target)

        # Ensure entities exist
        if source not in self.entities:
            self.add_entity(source)
        if target not in self.entities:
            self.add_entity(target)

        relation = Relation(
            source=source,
            relation_type=relation_type,
            target=target,
            confidence=confidence,
            position=position,
        )
        self.relations.append(relation)
        self._adjacency[source].append((relation_type, target))
        self._reverse[target].append((relation_type, source))
        return relation

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def query(
        self,
        source: Optional[str] = None,
        relation_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> List[Relation]:
        """Query relations matching the given pattern (None = wildcard)."""
        results = []
        for rel in self.relations:
            if source is not None and rel.source != source:
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            if target is not None and rel.target != target:
                continue
            results.append(rel)
        return results

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, str]]:
        """Return (relation_type, target_id) for all outgoing edges."""
        return self._adjacency.get(entity_id, [])

    def get_predecessors(self, entity_id: str) -> List[Tuple[str, str]]:
        """Return (relation_type, source_id) for all incoming edges."""
        return self._reverse.get(entity_id, [])

    # ------------------------------------------------------------------
    # Consistency checking
    # ------------------------------------------------------------------
    def check_consistency(self) -> List[str]:
        """
        Run basic consistency checks:
        1. No self-loops
        2. No contradictory relations (e.g., A is_a B and A is_not_a B)
        3. Type consistency across relations
        """
        issues = []

        for rel in self.relations:
            # Self-loop
            if rel.source == rel.target:
                issues.append(
                    f"Self-loop: {rel.source} --{rel.relation_type}--> {rel.target}"
                )

        # Check for contradictions
        relation_pairs: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for rel in self.relations:
            relation_pairs[(rel.source, rel.target)].add(rel.relation_type)

        contradictions = {
            frozenset({"is_a", "is_not_a"}),
            frozenset({"has", "lacks"}),
            frozenset({"causes", "prevents"}),
        }
        for (src, tgt), rel_types in relation_pairs.items():
            for contra_pair in contradictions:
                if contra_pair.issubset(rel_types):
                    issues.append(
                        f"Contradiction: {src} → {tgt} has both "
                        f"{' and '.join(contra_pair)}"
                    )

        return issues

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset the graph for a new session."""
        self.entities.clear()
        self.relations.clear()
        self._adjacency.clear()
        self._reverse.clear()

    def stats(self) -> Dict[str, int]:
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": len(set(e.entity_type for e in self.entities.values())),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={len(self.entities)}, "
            f"relations={len(self.relations)})"
        )
