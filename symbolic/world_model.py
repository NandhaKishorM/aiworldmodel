"""
Symbolic World Model — Rule Registry & Symbolic State
=======================================================
Defines the symbolic state representation and the rule protocol
for evaluating transitions between discrete concept states.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


# ======================================================================
# Symbolic State
# ======================================================================

@dataclass
class SymbolicState:
    """
    Represents a discrete symbolic state at a single sequence position.

    Attributes
    ----------
    concept_id   : integer concept index from the Gumbel-Softmax output
    concept_dist : soft probability distribution over all concepts (differentiable)
    position     : token position in the sequence
    attributes   : optional key-value attributes (entity properties, etc.)
    history      : list of prior concept_ids (for temporal reasoning)
    """
    concept_id: int
    concept_dist: Optional[torch.Tensor] = None  # (K,) soft assignment
    position: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    history: List[int] = field(default_factory=list)

    def push_to_history(self) -> None:
        """Append current concept_id to history."""
        self.history.append(self.concept_id)


# ======================================================================
# Rule Protocol
# ======================================================================

class Rule(ABC):
    """
    Abstract constraint rule for the symbolic world model.

    Rules evaluate whether a transition (state_t → state_t+1) is
    logically valid. Returns 0.0 for valid transitions, > 0.0 for
    violations (magnitude indicates severity).
    """

    def __init__(self, weight: float = 1.0, name: str = ""):
        self.weight = weight
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(
        self,
        state_t: SymbolicState,
        state_t1: SymbolicState,
    ) -> float:
        """Return violation magnitude (0 = valid)."""
        ...

    @abstractmethod
    def evaluate_soft(
        self,
        z_t: torch.Tensor,
        z_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable evaluation using soft concept distributions.

        z_t, z_t1 : (K,) Gumbel-Softmax soft assignments
        Returns: scalar tensor (violation magnitude)
        """
        ...


# ======================================================================
# Built-in Rules
# ======================================================================

class TypeConsistencyRule(Rule):
    """
    Entities should maintain type-consistent concept assignments.

    If concept groups are defined (e.g., {0,1,2} are 'person' concepts),
    switching to a concept in a different group is a violation.
    """

    def __init__(
        self,
        concept_groups: Optional[Dict[str, Set[int]]] = None,
        weight: float = 1.0,
    ):
        super().__init__(weight=weight, name="TypeConsistency")
        # Default: each concept is its own group (no constraint)
        self.concept_groups = concept_groups or {}
        self._build_group_map()

    def _build_group_map(self) -> None:
        """Map each concept_id to its group name."""
        self.id_to_group: Dict[int, str] = {}
        for group_name, concept_ids in self.concept_groups.items():
            for cid in concept_ids:
                self.id_to_group[cid] = group_name

    def evaluate(self, state_t: SymbolicState, state_t1: SymbolicState) -> float:
        g_t = self.id_to_group.get(state_t.concept_id, None)
        g_t1 = self.id_to_group.get(state_t1.concept_id, None)
        if g_t is not None and g_t1 is not None and g_t != g_t1:
            return self.weight
        return 0.0

    def evaluate_soft(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        """
        Soft version: for each group, compute probability mass in group at t
        and probability mass outside group at t+1. Product = violation.
        """
        violation = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
        for group_name, concept_ids in self.concept_groups.items():
            ids = list(concept_ids)
            mask = torch.zeros(z_t.shape[-1], device=z_t.device, dtype=z_t.dtype)
            mask[ids] = 1.0

            # Prob of being in group at t
            p_in_t = (z_t * mask).sum()
            # Prob of being outside group at t+1
            p_out_t1 = (z_t1 * (1 - mask)).sum()

            violation = violation + p_in_t * p_out_t1

        return violation * self.weight


class TemporalOrderRule(Rule):
    """
    Temporal ordering: certain concept transitions must be monotonic.

    E.g., concept 10 (past) should not follow concept 15 (future)
    if temporal ordering is defined.
    """

    def __init__(
        self,
        ordered_concepts: Optional[List[int]] = None,
        weight: float = 0.8,
    ):
        super().__init__(weight=weight, name="TemporalOrder")
        self.ordered_concepts = ordered_concepts or []
        self._build_order_map()

    def _build_order_map(self) -> None:
        self.concept_order: Dict[int, int] = {}
        for rank, cid in enumerate(self.ordered_concepts):
            self.concept_order[cid] = rank

    def evaluate(self, state_t: SymbolicState, state_t1: SymbolicState) -> float:
        order_t = self.concept_order.get(state_t.concept_id, -1)
        order_t1 = self.concept_order.get(state_t1.concept_id, -1)
        if order_t >= 0 and order_t1 >= 0 and order_t1 < order_t:
            return self.weight * (order_t - order_t1) / max(len(self.ordered_concepts), 1)
        return 0.0

    def evaluate_soft(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        if not self.ordered_concepts:
            return torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)

        device = z_t.device
        dtype = z_t.dtype
        K = z_t.shape[-1]

        # Build order vector
        order_vec = torch.full((K,), -1.0, device=device, dtype=dtype)
        for cid, rank in self.concept_order.items():
            if cid < K:
                order_vec[cid] = float(rank)

        mask = (order_vec >= 0).float()

        # Expected order at t and t+1
        expected_t = (z_t * order_vec * mask).sum()
        expected_t1 = (z_t1 * order_vec * mask).sum()

        # Violation: t+1 order < t order
        violation = torch.relu(expected_t - expected_t1)
        return violation * self.weight


class MutualExclusionRule(Rule):
    """
    Certain concept pairs cannot co-occur in adjacent positions.

    E.g., (alive, dead) or (increase, decrease) should not appear
    consecutively unless a valid transition is defined.
    """

    def __init__(
        self,
        exclusion_pairs: Optional[List[Tuple[int, int]]] = None,
        weight: float = 1.0,
    ):
        super().__init__(weight=weight, name="MutualExclusion")
        self.exclusion_pairs = exclusion_pairs or []

    def evaluate(self, state_t: SymbolicState, state_t1: SymbolicState) -> float:
        for a, b in self.exclusion_pairs:
            if (state_t.concept_id == a and state_t1.concept_id == b) or \
               (state_t.concept_id == b and state_t1.concept_id == a):
                return self.weight
        return 0.0

    def evaluate_soft(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        violation = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
        for a, b in self.exclusion_pairs:
            K = z_t.shape[-1]
            if a < K and b < K:
                # P(concept=a at t) * P(concept=b at t+1)
                v1 = z_t[..., a] * z_t1[..., b]
                v2 = z_t[..., b] * z_t1[..., a]
                violation = violation + v1 + v2
        return violation * self.weight


class ClinicalContraindicationRule(Rule):
    """
    Medical logic: If Symptom/Condition A is present, Treatment B is contraindicated.
    """
    def __init__(
        self,
        contraindication_pairs: Optional[List[Tuple[int, int]]] = None,
        weight: float = 2.0,
    ):
        super().__init__(weight=weight, name="ClinicalContraindication")
        self.pairs = contraindication_pairs or []

    def evaluate(self, state_t: SymbolicState, state_t1: SymbolicState) -> float:
        for cond, treat in self.pairs:
            if state_t.concept_id == cond and state_t1.concept_id == treat:
                return self.weight
        return 0.0

    def evaluate_soft(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        violation = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
        for cond, treat in self.pairs:
            K = z_t.shape[-1]
            if cond < K and treat < K:
                # P(condition at t) * P(treatment at t+1)
                violation = violation + (z_t[..., cond] * z_t1[..., treat])
        return violation * self.weight


class PhysicsConservationRule(Rule):
    """
    Physics logic: Certain continuous quantities (mapped to concepts) must be conserved.
    Penalizes states that suddenly create or destroy probability mass for these concepts.
    """
    def __init__(
        self,
        conserved_concepts: Optional[List[int]] = None,
        weight: float = 1.5,
    ):
        super().__init__(weight=weight, name="PhysicsConservation")
        self.conserved = conserved_concepts or []

    def evaluate(self, state_t: SymbolicState, state_t1: SymbolicState) -> float:
        v = 0.0
        for c in self.conserved:
            c_t = 1.0 if state_t.concept_id == c else 0.0
            c_t1 = 1.0 if state_t1.concept_id == c else 0.0
            v += abs(c_t - c_t1)
        return v * self.weight

    def evaluate_soft(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        violation = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
        for c in self.conserved:
            K = z_t.shape[-1]
            if c < K:
                # Penalty for absolute change in probability mass of a conserved concept
                violation = violation + torch.abs(z_t[..., c] - z_t1[..., c])
        return violation * self.weight


# ======================================================================
# Symbolic World Model (Rule Registry)
# ======================================================================

class SymbolicWorldModel:
    """
    Registry of symbolic rules that define valid state transitions.

    The world model evaluates whether a sequence of symbolic states
    is logically consistent, generating violation scores that become
    the symbolic loss signal for TTT.
    """

    def __init__(self) -> None:
        self.rules: List[Rule] = []
        self._rule_names: Set[str] = set()

    def add_rule(self, rule: Rule) -> None:
        """Register a new constraint rule."""
        if rule.name in self._rule_names:
            logger.warning(f"Rule '{rule.name}' already registered, replacing")
            self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)
        self._rule_names.add(rule.name)
        logger.debug(f"Registered rule: {rule.name} (weight={rule.weight})")

    def evaluate_transition(
        self,
        state_t: SymbolicState,
        state_t1: SymbolicState,
    ) -> Dict[str, float]:
        """Evaluate all rules on a state transition, return per-rule violations."""
        violations = {}
        for rule in self.rules:
            v = rule.evaluate(state_t, state_t1)
            if v > 0:
                violations[rule.name] = v
        return violations

    def evaluate_transition_soft(
        self,
        z_t: torch.Tensor,
        z_t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Differentiable evaluation over all rules.

        Returns:
            total_violation : scalar tensor (sum of weighted violations)
            details         : dict of per-rule violation magnitudes
        """
        total = torch.tensor(0.0, device=z_t.device, dtype=z_t.dtype)
        details: Dict[str, float] = {}
        for rule in self.rules:
            v = rule.evaluate_soft(z_t, z_t1)
            total = total + v
            details[rule.name] = v.item()
        return total, details

    def evaluate_sequence_soft(
        self,
        z_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Evaluate all consecutive pairs in a concept sequence.

        z_sequence : (seq_len, K) soft concept assignments
        Returns:
            total_loss : scalar tensor
            step_details : list of per-step violation dicts
        """
        total_loss = torch.tensor(0.0, device=z_sequence.device, dtype=z_sequence.dtype)
        step_details = []
        for t in range(z_sequence.shape[0] - 1):
            step_loss, details = self.evaluate_transition_soft(
                z_sequence[t], z_sequence[t + 1]
            )
            total_loss = total_loss + step_loss
            step_details.append(details)
        return total_loss, step_details

    @classmethod
    def from_config(cls, config: Dict) -> SymbolicWorldModel:
        """Build world model from YAML config dict."""
        model = cls()
        rules_config = config.get("symbolic", {}).get("default_rules", [])

        for rule_cfg in rules_config:
            rule_type = rule_cfg["type"]
            weight = rule_cfg.get("weight", 1.0)

            if rule_type == "type_consistency":
                model.add_rule(TypeConsistencyRule(weight=weight))
            elif rule_type == "temporal_order":
                model.add_rule(TemporalOrderRule(weight=weight))
            elif rule_type == "mutual_exclusion":
                pairs = rule_cfg.get("pairs", [])
                # Convert string pairs to index pairs (symbolic; use hashing)
                index_pairs = []
                for pair in pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        if isinstance(pair[0], int):
                            index_pairs.append(tuple(pair))
                        else:
                            # Hash string labels to concept indices
                            index_pairs.append((
                                hash(pair[0]) % 128,
                                hash(pair[1]) % 128,
                            ))
                model.add_rule(MutualExclusionRule(
                    exclusion_pairs=index_pairs, weight=weight
                ))
            elif rule_type == "clinical_contraindication":
                pairs = rule_cfg.get("pairs", [])
                index_pairs = []
                for pair in pairs:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        idx_pair = (
                            pair[0] if isinstance(pair[0], int) else hash(pair[0]) % 128,
                            pair[1] if isinstance(pair[1], int) else hash(pair[1]) % 128,
                        )
                        index_pairs.append(idx_pair)
                model.add_rule(ClinicalContraindicationRule(
                    contraindication_pairs=index_pairs, weight=weight
                ))
            elif rule_type == "physics_conservation":
                concepts = rule_cfg.get("concepts", [])
                index_concepts = [
                    c if isinstance(c, int) else hash(c) % 128 
                    for c in concepts
                ]
                model.add_rule(PhysicsConservationRule(
                    conserved_concepts=index_concepts, weight=weight
                ))
            else:
                logger.warning(f"Unknown rule type: {rule_type}, skipping")

        logger.info(f"World model loaded with {len(model.rules)} rules")
        return model

    def __repr__(self) -> str:
        rules_str = ", ".join(f"{r.name}(w={r.weight})" for r in self.rules)
        return f"SymbolicWorldModel(rules=[{rules_str}])"
