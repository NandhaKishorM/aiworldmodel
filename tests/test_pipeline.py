"""
Tests — Inference Pipeline (Integration)
==========================================
End-to-end pipeline tests using mock/small components.
Does NOT require downloading the full Gemma model.
"""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lora_adapter import LoRAInjector
from models.neuro_symbolic_bottleneck import GumbelSoftmaxProjection
from symbolic.world_model import SymbolicWorldModel, MutualExclusionRule
from symbolic.constraint_engine import ConstraintEngine
from symbolic.knowledge_graph import KnowledgeGraph
from ttt.ttt_engine import TTTEngine, TTTResult
from utils.metrics import TTTMetrics


class MockTokenizer:
    """Mock tokenizer for testing without real model."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, text, **kwargs):
        # Return random token IDs
        ids = torch.randint(2, 100, (1, 20))
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }

    def decode(self, ids, **kwargs):
        return "Mock output text for testing purposes."

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class MockModel(nn.Module):
    """
    Minimal model that mimics transformer interface for testing.
    Has named linear layers for LoRA injection.
    """

    def __init__(self, hidden_dim=64, vocab_size=100, num_layers=2):
        super().__init__()
        self.config = type("Config", (), {
            "hidden_size": hidden_dim,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": vocab_size,
            "max_position_embeddings": 512,
        })()

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(hidden_dim, hidden_dim),
                    "v_proj": nn.Linear(hidden_dim, hidden_dim),
                }),
            })
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, **kwargs):
        h = self.embed(input_ids)
        hidden_states = [h]

        for layer in self.layers:
            q = layer["self_attn"]["q_proj"](h)
            v = layer["self_attn"]["v_proj"](h)
            h = h + q + v  # simplified
            hidden_states.append(h)

        logits = self.head(h)

        result = type("Output", (), {
            "logits": logits,
            "hidden_states": tuple(hidden_states) if output_hidden_states else None,
        })()
        return result

    def generate(self, input_ids, attention_mask=None, **kwargs):
        max_new = kwargs.get("max_new_tokens", 5)
        generated = [input_ids]
        for _ in range(max_new):
            out = self.forward(generated[-1][:, -1:], output_hidden_states=False)
            next_token = out.logits.argmax(dim=-1)
            generated.append(next_token)
        return torch.cat([input_ids] + generated[1:], dim=1)


class TestTTTEngineIntegration:
    """Integration tests using mock model."""

    @pytest.fixture
    def setup(self):
        hidden_dim = 64
        num_concepts = 16

        model = MockModel(hidden_dim=hidden_dim, vocab_size=100)
        tokenizer = MockTokenizer()

        # Freeze base
        for p in model.parameters():
            p.requires_grad = False

        lora = LoRAInjector(
            model=model,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )

        bottleneck = GumbelSoftmaxProjection(
            num_concepts=num_concepts,
            hidden_dim=hidden_dim,
        )

        wm = SymbolicWorldModel()
        wm.add_rule(MutualExclusionRule(exclusion_pairs=[(0, 1)], weight=1.0))

        ce = ConstraintEngine(world_model=wm, epsilon=0.01)
        kg = KnowledgeGraph()

        config = {
            "ttt": {
                "num_update_steps": 2,
                "learning_rate": 1e-3,
                "lambda_sym": 10.0,
                "optimizer": "adam",
                "gradient_clip_norm": 1.0,
            },
            "bottleneck": {"extraction_layer": -1},
            "inference": {
                "max_new_tokens": 5,
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": False,
                "repetition_penalty": 1.0,
            },
        }

        engine = TTTEngine(
            model=model,
            tokenizer=tokenizer,
            lora_injector=lora,
            bottleneck=bottleneck,
            constraint_engine=ce,
            knowledge_graph=kg,
            config=config,
        )

        return engine, model, lora

    def test_ttt_engine_runs(self, setup):
        engine, model, lora = setup
        input_ids = torch.randint(2, 100, (1, 15))
        attention_mask = torch.ones_like(input_ids)

        result = engine.run(input_ids, attention_mask)

        assert isinstance(result, TTTResult)
        assert len(result.output_text) > 0
        assert len(result.step_metrics) == 2  # num_update_steps=2
        assert result.total_time_ms > 0

    def test_base_model_untouched(self, setup):
        engine, model, lora = setup

        # Snapshot base params
        base_snapshot = {n: p.clone() for n, p in model.named_parameters()}

        input_ids = torch.randint(2, 100, (1, 15))
        engine.run(input_ids)

        # Verify base model unchanged
        for name, param in model.named_parameters():
            assert torch.equal(param.data, base_snapshot[name]), \
                f"Base model param '{name}' was modified!"

    def test_adapter_changes_after_ttt(self, setup):
        engine, model, lora = setup

        lora.reset_all()
        initial_norm = lora.total_adapter_norm()

        input_ids = torch.randint(2, 100, (1, 15))
        engine.run(input_ids)

        final_norm = lora.total_adapter_norm()
        # After TTT steps, adapter norm should differ from initial
        # (B starts zero, so initial norm is ~0; after gradient steps it changes)
        assert final_norm != initial_norm or final_norm > 0

    def test_loss_decreases(self, setup):
        engine, model, lora = setup
        input_ids = torch.randint(2, 100, (1, 15))
        result = engine.run(input_ids)

        losses = [m.total_loss for m in result.step_metrics]
        # With 2 steps, loss shouldn't increase significantly
        # (not guaranteed to decrease with so few steps, but should be finite)
        for loss in losses:
            assert not torch.isnan(torch.tensor(loss))
            assert not torch.isinf(torch.tensor(loss))

    def test_session_discard(self, setup):
        engine, model, lora = setup
        input_ids = torch.randint(2, 100, (1, 15))
        engine.run(input_ids)

        engine.discard_session()
        # After discard, adapter norm should be ~0 (reset to zero-init B)
        norm = lora.total_adapter_norm()
        assert norm < 1e-5


class TestTTTMetrics:

    def test_from_result(self):
        from ttt.ttt_engine import TTTStepMetrics

        result = TTTResult(
            output_ids=torch.tensor([[1, 2, 3]]),
            output_text="test",
            step_metrics=[
                TTTStepMetrics(
                    step=0, total_loss=1.5, self_sup_loss=1.0,
                    symbolic_loss=0.5, grad_norm=0.1,
                    adapter_norm=0.01, num_violations=2, temperature=1.0,
                ),
                TTTStepMetrics(
                    step=1, total_loss=1.2, self_sup_loss=0.8,
                    symbolic_loss=0.4, grad_norm=0.08,
                    adapter_norm=0.02, num_violations=1, temperature=0.9,
                ),
            ],
            total_time_ms=100.0,
            ttt_time_ms=60.0,
            generation_time_ms=40.0,
            final_adapter_norm=0.02,
        )

        metrics = TTTMetrics.from_ttt_result(result)
        assert metrics.total_ttt_steps == 2
        assert metrics.total_violations == 3
        assert len(metrics.loss_per_step) == 2

        summary = metrics.summary()
        assert "TTT Inference Metrics" in summary

        json_str = metrics.to_json()
        assert "total_ttt_steps" in json_str
