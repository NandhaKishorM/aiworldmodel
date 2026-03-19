"""
Neuro-Symbolic TTT Inference Pipeline
========================================
End-to-end pipeline that loads all components, handles tokenization,
runs the TTT loop, and returns generated text.

Usage:
    pipeline = NeuroSymbolicTTTPipeline.from_config("config/ttt_config.yaml")
    result = pipeline.generate("Your prompt here")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from models.base_model import BaseModelLoader
from models.lora_adapter import LoRAInjector
from models.neuro_symbolic_bottleneck import GumbelSoftmaxProjection
from symbolic.constraint_engine import ConstraintEngine
from symbolic.knowledge_graph import KnowledgeGraph
from symbolic.world_model import SymbolicWorldModel
from ttt.ttt_engine import TTTEngine, TTTResult
from utils.logging_utils import setup_logging, get_logger

logger = logging.getLogger(__name__)


class NeuroSymbolicTTTPipeline:
    """
    Production inference pipeline for Neuro-Symbolic TTT.

    Components:
    - BaseModelLoader   → frozen Gemma-3-1B-IT
    - LoRAInjector      → fast weights (A, B)
    - GumbelSoftmax     → continuous → discrete projection
    - SymbolicWorldModel → rule registry
    - ConstraintEngine  → differentiable symbolic loss
    - KnowledgeGraph    → session entity tracker
    - TTTEngine         → orchestrates everything
    """

    def __init__(
        self,
        config: Dict,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.config = config
        self._model = model
        self._tokenizer = tokenizer
        self._initialized = False

        self.base_loader: Optional[BaseModelLoader] = None
        self.lora_injector: Optional[LoRAInjector] = None
        self.bottleneck: Optional[GumbelSoftmaxProjection] = None
        self.world_model: Optional[SymbolicWorldModel] = None
        self.constraint_engine: Optional[ConstraintEngine] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.ttt_engine: Optional[TTTEngine] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: Optional[str] = None,
    ) -> NeuroSymbolicTTTPipeline:
        """Load pipeline from YAML config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if device:
            config["model"]["device"] = device

        # Setup logging from config
        log_cfg = config.get("logging", {})
        setup_logging(
            level=log_cfg.get("level", "INFO"),
            fmt=log_cfg.get("format", "rich"),
            log_file=log_cfg.get("log_file"),
        )

        pipeline = cls(config=config)
        pipeline.initialize()
        return pipeline

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Load and wire all components."""
        if self._initialized:
            logger.warning("Pipeline already initialized")
            return

        init_start = time.perf_counter()

        model_cfg = self.config.get("model", {})
        lora_cfg = self.config.get("lora", {})
        bn_cfg = self.config.get("bottleneck", {})
        sym_cfg = self.config.get("symbolic", {})
        ttt_cfg = self.config.get("ttt", {})

        # 1. Load base model
        if self._model is None or self._tokenizer is None:
            self.base_loader = BaseModelLoader(
                model_name=model_cfg.get("name", "unsloth/gemma-3-1b-it"),
                torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
                device=model_cfg.get("device", "auto"),
            )
            self._model, self._tokenizer = self.base_loader.load()
        else:
            logger.info("Using pre-loaded model and tokenizer")

        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        hidden_dim = self._model.config.hidden_size

        # 2. Inject LoRA adapters
        target_layers = lora_cfg.get("target_layers", "all")
        if isinstance(target_layers, list):
            target_layers = [int(x) for x in target_layers]

        self.lora_injector = LoRAInjector(
            model=self._model,
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 32.0),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            target_layers=target_layers,
            dropout=lora_cfg.get("dropout", 0.0),
        )

        num_adapter_params = self.lora_injector.num_trainable_params()
        logger.info(f"LoRA adapters: {num_adapter_params:,} trainable params")

        # 3. Gumbel-Softmax bottleneck
        self.bottleneck = GumbelSoftmaxProjection(
            num_concepts=bn_cfg.get("num_concepts", 128),
            hidden_dim=hidden_dim,
            tau_start=bn_cfg.get("tau_start", 1.0),
            tau_min=bn_cfg.get("tau_min", 0.1),
            tau_anneal_rate=bn_cfg.get("tau_anneal_rate", 0.9),
            hard=bn_cfg.get("hard_sampling", False),
            normalize=bn_cfg.get("codebook_normalize", True),
        ).to(device=device, dtype=dtype)

        # 4. Symbolic world model
        self.world_model = SymbolicWorldModel.from_config(self.config)
        logger.info(f"World model: {self.world_model}")

        # 5. Constraint engine
        self.constraint_engine = ConstraintEngine(
            world_model=self.world_model,
            epsilon=ttt_cfg.get("epsilon", 0.01),
        )

        # 6. Knowledge graph
        kg_cfg = sym_cfg.get("knowledge_graph", {})
        self.knowledge_graph = KnowledgeGraph(
            max_entities=kg_cfg.get("max_entities", 1024),
            max_relations=kg_cfg.get("max_relations", 4096),
        )

        # 7. TTT engine
        self.ttt_engine = TTTEngine(
            model=self._model,
            tokenizer=self._tokenizer,
            lora_injector=self.lora_injector,
            bottleneck=self.bottleneck,
            constraint_engine=self.constraint_engine,
            knowledge_graph=self.knowledge_graph,
            config=self.config,
        )

        self._initialized = True
        init_time = (time.perf_counter() - init_start) * 1000
        logger.info(f"Pipeline initialized in {init_time:.0f}ms")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        ttt_enabled: Optional[bool] = None,
        **generation_kwargs,
    ) -> TTTResult:
        """
        Generate text with optional TTT adaptation.

        Parameters
        ----------
        prompt          : input text
        ttt_enabled     : override config ttt_enabled flag
        **generation_kwargs : passed to model.generate()

        Returns
        -------
        TTTResult with output text, metrics, and timing
        """
        assert self._initialized, "Call .initialize() first"

        # Should we run TTT?
        do_ttt = ttt_enabled if ttt_enabled is not None else \
            self.config.get("inference", {}).get("ttt_enabled", True)

        # Tokenize
        inputs = self._tokenize(prompt)
        input_ids = inputs["input_ids"].to(next(self._model.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(self._model.parameters()).device)

        logger.info(
            f"Input: {input_ids.shape[1]} tokens, "
            f"TTT={'enabled' if do_ttt else 'disabled'}"
        )

        if do_ttt:
            # Full TTT pipeline
            result = self.ttt_engine.run(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_kwargs=generation_kwargs,
            )
        else:
            # Standard inference (bypass TTT)
            result = self._standard_generate(
                input_ids, attention_mask, generation_kwargs
            )

        return result

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize prompt, applying chat template if available."""
        # Try chat template first (Gemma IT models use this)
        try:
            chat_input = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                chat_input,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = prompt

        return self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("model", {}).get(
                "max_position_embeddings", 8192
            ),
        )

    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_kwargs: Dict,
    ) -> TTTResult:
        """Standard inference without TTT (bypass mode)."""
        start = time.perf_counter()

        inf_cfg = self.config.get("inference", {})
        gen_config = {
            "max_new_tokens": inf_cfg.get("max_new_tokens", 512),
            "temperature": inf_cfg.get("temperature", 0.7),
            "top_p": inf_cfg.get("top_p", 0.9),
            "top_k": inf_cfg.get("top_k", 50),
            "do_sample": inf_cfg.get("do_sample", True),
            "repetition_penalty": inf_cfg.get("repetition_penalty", 1.1),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        gen_config.update(generation_kwargs)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_config,
            )

        new_tokens = output_ids[:, input_ids.shape[1]:]
        output_text = self._tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        )

        elapsed = (time.perf_counter() - start) * 1000
        return TTTResult(
            output_ids=output_ids,
            output_text=output_text,
            step_metrics=[],
            total_time_ms=elapsed,
            ttt_time_ms=0.0,
            generation_time_ms=elapsed,
            final_adapter_norm=0.0,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def reset_session(self) -> None:
        """Discard TTT adaptations and reset to base model behavior."""
        if self.ttt_engine:
            self.ttt_engine.discard_session()

    def save_session(self, path: str) -> None:
        """Save current adapter state for offline consolidation."""
        if self.ttt_engine:
            self.ttt_engine.save_session(path)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """Free all resources."""
        if self.lora_injector:
            self.lora_injector.remove_all()
        if self.base_loader:
            self.base_loader.unload()
        self._initialized = False
        logger.info("Pipeline unloaded")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"NeuroSymbolicTTTPipeline(status={status})"
