"""
Tests — TTT Engine (Loss, Optimizer, Adapter Isolation)
=========================================================
Validates the TTT loop mechanics without requiring the full model.
Uses mock/small models to test gradient isolation and loss behavior.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lora_adapter import LoRALayer, LoRAInjector
from models.neuro_symbolic_bottleneck import GumbelSoftmaxProjection
from symbolic.world_model import SymbolicWorldModel, MutualExclusionRule
from symbolic.constraint_engine import ConstraintEngine
from ttt.loss import SelfSupervisedLoss, SymbolicConsistencyLoss, TTTLoss
from ttt.optimizer import FastWeightOptimizer


# ======================================================================
# LoRA Adapter Tests
# ======================================================================

class TestLoRALayer:

    def test_shape(self):
        lora = LoRALayer(d_in=128, d_out=256, rank=8)
        x = torch.randn(2, 10, 128)
        out = lora(x)
        assert out.shape == (2, 10, 256)

    def test_zero_init_output(self):
        """B is zero-initialized, so initial output should be near zero."""
        lora = LoRALayer(d_in=128, d_out=256, rank=8)
        x = torch.randn(2, 10, 128)
        out = lora(x)
        assert out.abs().max().item() < 1e-6

    def test_reset(self):
        lora = LoRALayer(d_in=64, d_out=64, rank=4)
        # Manually set non-zero B
        lora.B.data.fill_(1.0)
        assert lora.adapter_norm > 0

        lora.reset()
        x = torch.randn(1, 5, 64)
        out = lora(x)
        assert out.abs().max().item() < 1e-6

    def test_gradient_flow(self):
        lora = LoRALayer(d_in=64, d_out=64, rank=4)
        x = torch.randn(1, 5, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.A.grad is not None
        assert lora.B.grad is not None


class TestLoRAInjector:

    @pytest.fixture
    def mock_model(self):
        """Simple model with named linear layers to inject into."""
        model = nn.ModuleDict({
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": nn.ModuleDict({
                        "q_proj": nn.Linear(64, 64),
                        "k_proj": nn.Linear(64, 64),
                        "v_proj": nn.Linear(64, 64),
                    }),
                    "mlp": nn.ModuleDict({
                        "gate_proj": nn.Linear(64, 128),
                    }),
                })
                for _ in range(2)
            ])
        })
        return model

    def test_injection_count(self, mock_model):
        injector = LoRAInjector(
            model=mock_model,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        # 2 layers × 2 target modules = 4 adapters
        assert len(injector.adapters) == 4

    def test_adapter_only_trainable(self, mock_model):
        # Freeze base
        for p in mock_model.parameters():
            p.requires_grad = False

        injector = LoRAInjector(mock_model, rank=4, target_modules=["q_proj"])
        params = injector.get_parameters()

        # All adapter params should require grad
        for p in params:
            assert p.requires_grad

        # Base model params should NOT require grad
        for p in mock_model.parameters():
            assert not p.requires_grad

    def test_reset_all(self, mock_model):
        injector = LoRAInjector(mock_model, rank=4, target_modules=["q_proj"])
        # Manually modify adapter
        for adapter in injector.adapters.values():
            adapter.B.data.fill_(1.0)

        injector.reset_all()

        for adapter in injector.adapters.values():
            assert adapter.B.data.abs().max().item() < 1e-6

    def test_remove_all(self, mock_model):
        injector = LoRAInjector(mock_model, rank=4, target_modules=["q_proj"])
        assert len(injector.adapters) > 0

        injector.remove_all()
        assert len(injector.adapters) == 0


# ======================================================================
# Loss Tests
# ======================================================================

class TestSelfSupervisedLoss:

    def test_output_shape(self):
        loss_fn = SelfSupervisedLoss(mask_ratio=0.3)
        logits = torch.randn(1, 20, 100)  # (batch, seq, vocab)
        input_ids = torch.randint(0, 100, (1, 20))
        loss, num_masked = loss_fn(logits, input_ids)

        assert loss.dim() == 0  # scalar
        assert num_masked > 0

    def test_differentiable(self):
        loss_fn = SelfSupervisedLoss(mask_ratio=0.5)
        logits = torch.randn(1, 20, 100, requires_grad=True)
        input_ids = torch.randint(0, 100, (1, 20))
        loss, _ = loss_fn(logits, input_ids)

        loss.backward()
        assert logits.grad is not None


class TestSymbolicConsistencyLoss:

    def test_differentiable(self):
        wm = SymbolicWorldModel()
        wm.add_rule(MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0))
        ce = ConstraintEngine(world_model=wm, epsilon=0.01)
        loss_fn = SymbolicConsistencyLoss(ce)

        proj = GumbelSoftmaxProjection(num_concepts=4, hidden_dim=32)
        h = torch.randn(1, 5, 32, requires_grad=True)
        result = proj(h)

        loss, cr = loss_fn(result)
        loss.backward()  # Should not raise
        assert h.grad is not None


class TestTTTLoss:

    def test_combined_loss(self):
        wm = SymbolicWorldModel()
        wm.add_rule(MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0))
        ce = ConstraintEngine(world_model=wm, epsilon=0.01)

        ttt_loss = TTTLoss(
            constraint_engine=ce,
            lambda_sym=0.5,
            mask_ratio=0.15,
        )

        proj = GumbelSoftmaxProjection(num_concepts=4, hidden_dim=32)
        h = torch.randn(1, 10, 32)
        projection_result = proj(h)

        logits = torch.randn(1, 10, 100, requires_grad=True)
        input_ids = torch.randint(0, 100, (1, 10))

        result = ttt_loss(logits, input_ids, projection_result)

        assert result.total_loss.dim() == 0
        assert result.self_sup_loss.dim() == 0
        assert result.symbolic_loss.dim() == 0

        result.total_loss.backward()
        assert logits.grad is not None


# ======================================================================
# Fast Weight Optimizer Tests
# ======================================================================

class TestFastWeightOptimizer:

    def test_step_updates_params(self):
        params = [torch.randn(4, 4, requires_grad=True)]
        opt = FastWeightOptimizer(params, optimizer_type="adam", learning_rate=0.1)

        initial = params[0].data.clone()
        loss = (params[0] ** 2).sum()
        metrics = opt.step(loss)

        # Params should have changed
        assert not torch.equal(params[0].data, initial)
        assert "grad_norm" in metrics
        assert "param_norm" in metrics
        assert metrics["step"] == 1

    def test_reset(self):
        params = [torch.randn(4, 4, requires_grad=True)]
        opt = FastWeightOptimizer(params, optimizer_type="sgd")

        loss = (params[0] ** 2).sum()
        opt.step(loss)
        assert opt.step_count == 1

        opt.reset()
        assert opt.step_count == 0


# ======================================================================
# Integration: Gradient Isolation
# ======================================================================

class TestGradientIsolation:
    """
    Critical test: verify that TTT only updates adapter params,
    leaving base model completely untouched.
    """

    def test_base_model_frozen(self):
        # Simulate base model
        base = nn.Linear(64, 64)
        for p in base.parameters():
            p.requires_grad = False

        base_snapshot = {n: p.clone() for n, p in base.named_parameters()}

        # LoRA adapter
        adapter = LoRALayer(d_in=64, d_out=64, rank=4)
        adapter.B.data.fill_(0.01)  # small non-zero

        # Simulate forward: base + adapter
        x = torch.randn(1, 5, 64)
        base_out = base(x)
        adapter_out = adapter(x)
        combined = base_out + adapter_out

        # Loss and backward
        loss = combined.sum()
        loss.backward()

        # Base params: no grad, unchanged
        for name, param in base.named_parameters():
            assert param.grad is None
            assert torch.equal(param.data, base_snapshot[name])

        # Adapter params: have grad
        assert adapter.A.grad is not None
        assert adapter.B.grad is not None
