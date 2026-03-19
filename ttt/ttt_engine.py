"""
TTT Engine — Test-Time Training Orchestrator
================================================
Orchestrates the full TTT loop:

1. Reset LoRA adapters → zero-effect
2. Dry-run forward pass → extract hidden states
3. Gumbel-Softmax projection → symbolic concepts
4. Compute L_TTT = L_self_sup + λ · L_sym
5. Backprop into fast weights only
6. Repeat N steps
7. Final forward pass with adapted weights → generate output

The base model (W_base) is NEVER modified.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from models.lora_adapter import LoRAInjector
from models.neuro_symbolic_bottleneck import GumbelSoftmaxProjection, ProjectionResult
from symbolic.constraint_engine import ConstraintEngine, ConstraintResult
from symbolic.knowledge_graph import KnowledgeGraph
from ttt.loss import TTTLoss, TTTLossResult
from ttt.optimizer import FastWeightOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TTTStepMetrics:
    """Metrics from a single TTT gradient step."""
    step: int
    total_loss: float
    self_sup_loss: float
    symbolic_loss: float
    grad_norm: float
    adapter_norm: float
    num_violations: int
    temperature: float


@dataclass
class TTTResult:
    """Complete result from a TTT inference session."""
    output_ids: torch.Tensor                    # generated token IDs
    output_text: str                            # decoded output text
    step_metrics: List[TTTStepMetrics]          # per-step TTT metrics
    total_time_ms: float                        # wall clock time
    ttt_time_ms: float                          # time spent in TTT loop
    generation_time_ms: float                   # time spent in final generation
    final_adapter_norm: float                   # adapter norm after TTT
    concept_utilization: Dict[str, Any] = field(default_factory=dict)
    kg_stats: Dict[str, int] = field(default_factory=dict)


class TTTEngine:
    """
    Full Test-Time Training engine.

    Manages the dual-speed system:
    - System 1 (Base Model): frozen transformer weights
    - System 2 (Symbolic Engine): non-differentiable logic via differentiable proxy
    - Fast Weights (LoRA): updated during the TTT loop
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        lora_injector: LoRAInjector,
        bottleneck: GumbelSoftmaxProjection,
        constraint_engine: ConstraintEngine,
        knowledge_graph: KnowledgeGraph,
        config: Dict,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lora_injector = lora_injector
        self.bottleneck = bottleneck
        self.constraint_engine = constraint_engine
        self.knowledge_graph = knowledge_graph
        self.config = config

        ttt_cfg = config.get("ttt", {})
        self.num_update_steps = ttt_cfg.get("num_update_steps", 3)
        self.lambda_sym = ttt_cfg.get("lambda_sym", 10.0)
        self.learning_rate = ttt_cfg.get("learning_rate", 1e-4)
        self.optimizer_type = ttt_cfg.get("optimizer", "adam")
        self.gradient_clip_norm = ttt_cfg.get("gradient_clip_norm", 1.0)

        bottleneck_cfg = config.get("bottleneck", {})
        self.extraction_layer = bottleneck_cfg.get("extraction_layer", -1)

        # TTT loss function
        self.ttt_loss_fn = TTTLoss(
            constraint_engine=constraint_engine,
            lambda_sym=self.lambda_sym,
        )

        # Device
        self.device = next(model.parameters()).device

        logger.info(
            f"TTTEngine initialized: {self.num_update_steps} steps, "
            f"lr={self.learning_rate}, λ_sym={self.lambda_sym}"
        )

    # ------------------------------------------------------------------
    # Main TTT + Generation flow
    # ------------------------------------------------------------------
    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_kwargs: Optional[Dict] = None,
    ) -> TTTResult:
        """
        Execute the full TTT loop and generate output.

        Parameters
        ----------
        input_ids       : (1, seq_len) tokenized input
        attention_mask  : (1, seq_len) mask
        generation_kwargs : kwargs for model.generate()

        Returns
        -------
        TTTResult with generated text and all metrics
        """
        total_start = time.perf_counter()

        # === Phase 1: TTT Adaptation ===
        ttt_start = time.perf_counter()
        step_metrics = self._ttt_loop(input_ids, attention_mask)
        ttt_time = (time.perf_counter() - ttt_start) * 1000

        # === Phase 2: Final Generation with Adapted Weights ===
        gen_start = time.perf_counter()
        output_ids, output_text = self._generate(
            input_ids, attention_mask, generation_kwargs or {}
        )
        gen_time = (time.perf_counter() - gen_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        # Concept utilization stats
        concept_util = {}
        if step_metrics:
            with torch.no_grad():
                hidden = self._extract_hidden_states(input_ids, attention_mask)
                proj = self.bottleneck(hidden)
                concept_util = self.bottleneck.concept_utilization(proj.concept_ids)

        return TTTResult(
            output_ids=output_ids,
            output_text=output_text,
            step_metrics=step_metrics,
            total_time_ms=total_time,
            ttt_time_ms=ttt_time,
            generation_time_ms=gen_time,
            final_adapter_norm=self.lora_injector.total_adapter_norm(),
            concept_utilization=concept_util,
            kg_stats=self.knowledge_graph.stats(),
        )

    # ------------------------------------------------------------------
    # TTT Loop
    # ------------------------------------------------------------------
    def _ttt_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> List[TTTStepMetrics]:
        """
        Perform N gradient steps on fast weights using the TTT objective.

        Steps:
        1. Reset adapters → zero effect
        2. For each step:
           a. Forward pass (with adapters active)
           b. Extract hidden states → Gumbel-Softmax projection
           c. Compute L_TTT
           d. Backprop into adapter params only
           e. Anneal Gumbel temperature
        """
        # Reset for new session if memory is not persistent
        if not self.config.get("ttt", {}).get("persistent_memory", False):
            self.lora_injector.reset_all()
            self.bottleneck.reset_temperature()
            self.knowledge_graph.clear()

        # Build optimizer for adapter params only
        adapter_params = self.lora_injector.get_parameters()
        optimizer = FastWeightOptimizer(
            parameters=adapter_params,
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            gradient_clip_norm=self.gradient_clip_norm,
        )

        step_metrics: List[TTTStepMetrics] = []

        # Enable gradient computation for adapters
        for p in adapter_params:
            p.requires_grad_(True)

        for step_idx in range(self.num_update_steps):
            # --- Dry-run forward pass ---
            self.model.eval()

            # Forward pass with LoRA hooks active (adapters contribute ΔW)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            logits = outputs.logits  # (1, seq_len, vocab)
            hidden_states = outputs.hidden_states  # tuple of layer outputs

            # Extract hidden state from target layer
            layer_idx = self.extraction_layer
            if layer_idx < 0:
                layer_idx = len(hidden_states) + layer_idx
            h = hidden_states[layer_idx]  # (1, seq_len, d)

            # --- Gumbel-Softmax projection ---
            self.bottleneck.train()
            projection = self.bottleneck(h, tau=self.bottleneck.tau)

            # --- Compute TTT loss ---
            loss_result: TTTLossResult = self.ttt_loss_fn(
                logits=logits,
                input_ids=input_ids,
                projection_result=projection,
                attention_mask=attention_mask,
            )

            # --- Gradient step on fast weights ---
            opt_metrics = optimizer.step(loss_result.total_loss)

            # --- Temperature annealing ---
            new_tau = self.bottleneck.anneal_temperature()

            # --- Record metrics ---
            metrics = TTTStepMetrics(
                step=step_idx,
                total_loss=loss_result.total_loss.item(),
                self_sup_loss=loss_result.self_sup_loss.item(),
                symbolic_loss=loss_result.symbolic_loss.item(),
                grad_norm=opt_metrics["grad_norm"],
                adapter_norm=self.lora_injector.total_adapter_norm(),
                num_violations=(
                    loss_result.constraint_result.num_violations
                    if loss_result.constraint_result else 0
                ),
                temperature=new_tau,
            )
            step_metrics.append(metrics)

            logger.info(
                f"TTT step {step_idx}: L_total={metrics.total_loss:.4f} "
                f"(L_ss={metrics.self_sup_loss:.4f}, "
                f"L_sym={metrics.symbolic_loss:.4f}), "
                f"grad_norm={metrics.grad_norm:.4f}, "
                f"adapter_norm={metrics.adapter_norm:.4f}, "
                f"τ={metrics.temperature:.3f}"
            )

        return step_metrics

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_kwargs: Dict,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate output with adapted weights: y = Transformer(x; W_base + BA)

        LoRA hooks remain active, so the model uses W_base + ΔW_fast.
        """
        self.model.eval()

        inf_cfg = self.config.get("inference", {})
        gen_config = {
            "max_new_tokens": inf_cfg.get("max_new_tokens", 512),
            "temperature": inf_cfg.get("temperature", 0.7),
            "top_p": inf_cfg.get("top_p", 0.9),
            "top_k": inf_cfg.get("top_k", 50),
            "do_sample": inf_cfg.get("do_sample", True),
            "repetition_penalty": inf_cfg.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_config.update(generation_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_config,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[:, input_ids.shape[1]:]
        output_text = self.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        )

        return output_ids, output_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract hidden states from the target layer."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        layer_idx = self.extraction_layer
        if layer_idx < 0:
            layer_idx = len(outputs.hidden_states) + layer_idx
        return outputs.hidden_states[layer_idx]

    def discard_session(self) -> None:
        """
        End TTT session: reset fast weights, clear KG.
        Base model remains pristine.
        """
        self.lora_injector.reset_all()
        self.knowledge_graph.clear()
        self.bottleneck.reset_temperature()
        logger.info("TTT session discarded — base model untouched")

    def save_session(self, path: str) -> None:
        """Save adapter state for episodic memory / offline consolidation."""
        self.lora_injector.save_adapters(path)
        logger.info(f"TTT session saved to {path}")
