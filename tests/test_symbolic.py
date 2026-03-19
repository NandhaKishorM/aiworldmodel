"""
Tests — Symbolic Engine (World Model + Constraint Engine)
==========================================================
Validates rule evaluation, hinge loss, and differentiability.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from symbolic.world_model import (
    SymbolicState,
    SymbolicWorldModel,
    TypeConsistencyRule,
    TemporalOrderRule,
    MutualExclusionRule,
)
from symbolic.constraint_engine import ConstraintEngine, ConstraintResult
from symbolic.knowledge_graph import KnowledgeGraph


# ======================================================================
# World Model Rules
# ======================================================================

class TestTypeConsistencyRule:

    def test_valid_same_group(self):
        rule = TypeConsistencyRule(
            concept_groups={"person": {0, 1, 2}, "object": {3, 4, 5}},
            weight=1.0,
        )
        s_t = SymbolicState(concept_id=0)
        s_t1 = SymbolicState(concept_id=1)
        assert rule.evaluate(s_t, s_t1) == 0.0

    def test_violation_different_group(self):
        rule = TypeConsistencyRule(
            concept_groups={"person": {0, 1, 2}, "object": {3, 4, 5}},
            weight=1.0,
        )
        s_t = SymbolicState(concept_id=0)
        s_t1 = SymbolicState(concept_id=4)
        assert rule.evaluate(s_t, s_t1) == 1.0

    def test_unknown_concepts_no_violation(self):
        rule = TypeConsistencyRule(
            concept_groups={"person": {0, 1}},
            weight=1.0,
        )
        s_t = SymbolicState(concept_id=10)
        s_t1 = SymbolicState(concept_id=20)
        assert rule.evaluate(s_t, s_t1) == 0.0

    def test_soft_evaluation(self):
        rule = TypeConsistencyRule(
            concept_groups={"A": {0, 1}, "B": {2, 3}},
            weight=1.0,
        )
        # z_t firmly in group A, z_t1 firmly in group B → violation
        z_t = torch.tensor([0.5, 0.5, 0.0, 0.0])
        z_t1 = torch.tensor([0.0, 0.0, 0.5, 0.5])
        v = rule.evaluate_soft(z_t, z_t1)
        assert v.item() > 0

        # z_t and z_t1 both in group A → no violation
        z_t = torch.tensor([0.5, 0.5, 0.0, 0.0])
        z_t1 = torch.tensor([0.4, 0.6, 0.0, 0.0])
        v = rule.evaluate_soft(z_t, z_t1)
        assert v.item() == pytest.approx(0.0, abs=1e-6)


class TestTemporalOrderRule:

    def test_valid_forward_order(self):
        rule = TemporalOrderRule(ordered_concepts=[10, 20, 30], weight=1.0)
        s_t = SymbolicState(concept_id=10)
        s_t1 = SymbolicState(concept_id=20)
        assert rule.evaluate(s_t, s_t1) == 0.0

    def test_violation_reverse_order(self):
        rule = TemporalOrderRule(ordered_concepts=[10, 20, 30], weight=1.0)
        s_t = SymbolicState(concept_id=30)
        s_t1 = SymbolicState(concept_id=10)
        assert rule.evaluate(s_t, s_t1) > 0

    def test_non_ordered_concepts_valid(self):
        rule = TemporalOrderRule(ordered_concepts=[10, 20], weight=1.0)
        s_t = SymbolicState(concept_id=99)
        s_t1 = SymbolicState(concept_id=50)
        assert rule.evaluate(s_t, s_t1) == 0.0


class TestMutualExclusionRule:

    def test_excluded_pair_violation(self):
        rule = MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0)
        s_t = SymbolicState(concept_id=0)
        s_t1 = SymbolicState(concept_id=1)
        assert rule.evaluate(s_t, s_t1) == 1.0

    def test_excluded_pair_reverse_violation(self):
        rule = MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0)
        s_t = SymbolicState(concept_id=1)
        s_t1 = SymbolicState(concept_id=0)
        assert rule.evaluate(s_t, s_t1) == 1.0

    def test_non_excluded_pair_valid(self):
        rule = MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0)
        s_t = SymbolicState(concept_id=0)
        s_t1 = SymbolicState(concept_id=2)
        assert rule.evaluate(s_t, s_t1) == 0.0

    def test_soft_evaluation_differentiable(self):
        rule = MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0)
        z_t = torch.tensor([0.9, 0.1, 0.0, 0.0], requires_grad=True)
        z_t1 = torch.tensor([0.1, 0.9, 0.0, 0.0], requires_grad=True)
        v = rule.evaluate_soft(z_t, z_t1)

        assert v.item() > 0
        v.backward()
        assert z_t.grad is not None
        assert z_t1.grad is not None


# ======================================================================
# Constraint Engine
# ======================================================================

class TestConstraintEngine:

    @pytest.fixture
    def engine(self):
        wm = SymbolicWorldModel()
        wm.add_rule(MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0))
        return ConstraintEngine(world_model=wm, epsilon=0.01)

    def test_no_violation_below_epsilon(self, engine):
        # Both positions have same concept → no exclusion violation
        z = torch.zeros(5, 4)
        z[:, 2] = 1.0  # all positions = concept 2
        result = engine.evaluate(z)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-5)
        assert result.num_violations == 0

    def test_violation_above_epsilon(self, engine):
        z = torch.zeros(2, 4)
        z[0, 0] = 1.0  # concept 0 at position 0
        z[1, 1] = 1.0  # concept 1 at position 1 → exclusion violated
        result = engine.evaluate(z)
        assert result.loss.item() > 0
        assert result.num_violations > 0

    def test_differentiable_loss(self, engine):
        z = torch.zeros(3, 4, requires_grad=True)
        # Create a violation pattern
        z_data = torch.zeros(3, 4)
        z_data[0, 0] = 1.0
        z_data[1, 1] = 1.0
        z_data[2, 0] = 1.0
        z = z_data.clone().requires_grad_(True)

        result = engine.evaluate(z)
        result.loss.backward()
        assert z.grad is not None

    def test_hinge_margin(self):
        wm = SymbolicWorldModel()
        wm.add_rule(MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=0.005))
        engine = ConstraintEngine(world_model=wm, epsilon=0.01)

        z = torch.zeros(2, 4)
        z[0, 0] = 1.0
        z[1, 1] = 1.0
        result = engine.evaluate(z)
        # violation (0.005*2=0.01) equals epsilon → hinge should be ~0
        assert result.loss.item() == pytest.approx(0.0, abs=0.002)


# ======================================================================
# Knowledge Graph
# ======================================================================

class TestKnowledgeGraph:

    def test_add_and_query(self):
        kg = KnowledgeGraph()
        kg.add_entity("alice", "person")
        kg.add_entity("bob", "person")
        kg.add_relation("alice", "knows", "bob")

        results = kg.query(source="alice")
        assert len(results) == 1
        assert results[0].target == "bob"

    def test_consistency_self_loop(self):
        kg = KnowledgeGraph()
        kg.add_entity("x", "thing")
        kg.add_relation("x", "is_a", "x")
        issues = kg.check_consistency()
        assert any("Self-loop" in i for i in issues)

    def test_consistency_contradiction(self):
        kg = KnowledgeGraph()
        kg.add_relation("a", "is_a", "b")
        kg.add_relation("a", "is_not_a", "b")
        issues = kg.check_consistency()
        assert any("Contradiction" in i for i in issues)

    def test_clear(self):
        kg = KnowledgeGraph()
        kg.add_entity("a")
        kg.add_relation("a", "r", "b")
        kg.clear()
        assert kg.stats()["num_entities"] == 0
        assert kg.stats()["num_relations"] == 0


# ======================================================================
# World Model Config Loading
# ======================================================================

class TestWorldModelConfig:

    def test_from_config(self):
        config = {
            "symbolic": {
                "default_rules": [
                    {"type": "type_consistency", "weight": 1.0},
                    {"type": "mutual_exclusion", "weight": 1.0, "pairs": [[0, 1]]},
                ]
            }
        }
        wm = SymbolicWorldModel.from_config(config)
        assert len(wm.rules) == 2
