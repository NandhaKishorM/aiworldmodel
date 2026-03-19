"""
Tests — Gumbel-Softmax Bottleneck
====================================
Validates differentiability, temperature annealing, and shape correctness.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.neuro_symbolic_bottleneck import (
    GumbelSoftmaxProjection,
    ConceptCodebook,
    ProjectionResult,
)


class TestConceptCodebook:
    """Tests for the concept codebook module."""

    def test_shape(self):
        K, d = 64, 128
        codebook = ConceptCodebook(num_concepts=K, hidden_dim=d)
        h = torch.randn(2, 10, d)
        logits = codebook(h)
        assert logits.shape == (2, 10, K)

    def test_normalization(self):
        codebook = ConceptCodebook(num_concepts=32, hidden_dim=64, normalize=True)
        V = codebook.get_codebook()
        norms = V.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_no_normalization(self):
        codebook = ConceptCodebook(num_concepts=32, hidden_dim=64, normalize=False)
        V = codebook.get_codebook()
        norms = V.norm(dim=-1)
        # Should NOT all be 1.0
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


class TestGumbelSoftmaxProjection:
    """Tests for the Gumbel-Softmax projection layer."""

    @pytest.fixture
    def projection(self):
        return GumbelSoftmaxProjection(
            num_concepts=64,
            hidden_dim=128,
            tau_start=1.0,
            tau_min=0.1,
            tau_anneal_rate=0.5,
        )

    def test_output_shape(self, projection):
        h = torch.randn(2, 10, 128)
        result = projection(h)

        assert isinstance(result, ProjectionResult)
        assert result.soft_z.shape == (2, 10, 64)
        assert result.hard_z.shape == (2, 10, 64)
        assert result.concept_ids.shape == (2, 10)
        assert result.logits.shape == (2, 10, 64)
        assert result.entropy.shape == (2, 10)

    def test_soft_z_sums_to_one(self, projection):
        h = torch.randn(1, 5, 128)
        result = projection(h)
        sums = result.soft_z.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_hard_z_is_one_hot(self, projection):
        h = torch.randn(1, 5, 128)
        result = projection(h)
        # Each hard_z row should sum to 1
        sums = result.hard_z.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # Each hard_z row should have max = 1
        maxes = result.hard_z.max(dim=-1).values
        assert torch.allclose(maxes, torch.ones_like(maxes), atol=1e-5)

    def test_gradient_flow(self, projection):
        """Critical test: gradients must flow through Gumbel-Softmax."""
        h = torch.randn(1, 5, 128, requires_grad=True)
        result = projection(h)

        # Sum soft_z and backward
        loss = result.soft_z.sum()
        loss.backward()

        # Gradients must reach the input
        assert h.grad is not None
        assert h.grad.abs().sum() > 0

        # Gradients must reach codebook
        for param in projection.codebook.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_gradient_flow_through_hard(self, projection):
        """Straight-through estimator must pass gradients."""
        h = torch.randn(1, 5, 128, requires_grad=True)
        result = projection(h)

        # Use hard_z (should still get gradients via straight-through)
        loss = result.hard_z.sum()
        loss.backward()

        assert h.grad is not None
        assert h.grad.abs().sum() > 0

    def test_temperature_annealing(self, projection):
        assert projection.tau == 1.0

        tau1 = projection.anneal_temperature()
        assert tau1 == 0.5

        tau2 = projection.anneal_temperature()
        assert tau2 == 0.25

        tau3 = projection.anneal_temperature()
        assert tau3 == 0.125

        # Should clamp at tau_min
        tau4 = projection.anneal_temperature()
        assert tau4 == 0.1

    def test_temperature_reset(self, projection):
        projection.anneal_temperature()
        projection.anneal_temperature()
        projection.reset_temperature()
        assert projection.tau == 1.0

    def test_low_temperature_approaches_one_hot(self):
        """As τ→0, soft_z output should be sharper (lower entropy)."""
        proj = GumbelSoftmaxProjection(
            num_concepts=64, hidden_dim=128, tau_start=1.0, tau_min=0.01
        )
        torch.manual_seed(42)
        h = torch.randn(1, 5, 128)

        # Compute entropy directly from soft_z (reflects temperature)
        torch.manual_seed(0)
        result_high = proj(h, tau=5.0)
        sz_high = result_high.soft_z
        ent_high = -(sz_high * (sz_high + 1e-10).log()).sum(dim=-1).mean()

        torch.manual_seed(0)
        result_low = proj(h, tau=0.01)
        sz_low = result_low.soft_z
        ent_low = -(sz_low * (sz_low + 1e-10).log()).sum(dim=-1).mean()

        assert ent_low < ent_high

    def test_concept_utilization(self, projection):
        h = torch.randn(1, 100, 128)
        result = projection(h)
        util = projection.concept_utilization(result.concept_ids)

        assert "active_concepts" in util
        assert "utilization_ratio" in util
        assert "perplexity" in util
        assert util["active_concepts"] > 0
        assert 0 <= util["utilization_ratio"] <= 1


class TestProjectionWithDifferentDim:
    """Test projection when input dim != codebook dim."""

    def test_with_projection_layer(self):
        proj = GumbelSoftmaxProjection(num_concepts=32, hidden_dim=64)
        proj.set_projection(input_dim=128)

        h = torch.randn(1, 5, 128, requires_grad=True)
        result = proj(h)

        assert result.soft_z.shape == (1, 5, 32)

        # Gradients still flow
        result.soft_z.sum().backward()
        assert h.grad is not None
