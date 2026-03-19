"""
Base Model Loader — Frozen Gemma-3-1B-IT
=========================================
Loads the pre-trained transformer with all parameters frozen.
Exposes hidden-state extraction for the neuro-symbolic bottleneck.
Optimized for T4 Colab (16 GB VRAM, BF16 support).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


class BaseModelLoader:
    """Loads and freezes the base Gemma-3-1B-IT model."""

    def __init__(
        self,
        model_name: str = "unsloth/gemma-3-1b-it",
        torch_dtype: str = "bfloat16",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.device = self._resolve_device(device)

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                logger.info(f"Auto-detected GPU: {gpu_name} ({vram_gb:.1f} GB)")
                return dev
            logger.warning("No CUDA device found — falling back to CPU")
            return torch.device("cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Download / cache and return (model, tokenizer)."""
        logger.info(f"Loading model: {self.model_name} (dtype={self.torch_dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device.type == "cpu":
            self.model = self.model.to(self.device)

        self._freeze_all_parameters()
        self.model.eval()

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded — {num_params / 1e6:.1f}M params (all frozen)")
        return self.model, self.tokenizer

    def _freeze_all_parameters(self) -> None:
        """Freeze every parameter in the base model."""
        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Run a forward pass and return hidden states for requested layers.

        Parameters
        ----------
        input_ids : (batch, seq_len) token ids
        attention_mask : optional mask
        layer_indices : list of layer indices to return (default: all + final)

        Returns
        -------
        dict mapping layer index → hidden state tensor (batch, seq_len, d)
        """
        assert self.model is not None, "Call .load() first"

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        all_hidden = outputs.hidden_states  # tuple of (num_layers + 1) tensors

        if layer_indices is None:
            layer_indices = list(range(len(all_hidden)))

        result: Dict[int, torch.Tensor] = {}
        for idx in layer_indices:
            actual_idx = idx if idx >= 0 else len(all_hidden) + idx
            if 0 <= actual_idx < len(all_hidden):
                result[idx] = all_hidden[actual_idx]
            else:
                logger.warning(f"Layer index {idx} out of range, skipping")
        return result

    # ------------------------------------------------------------------
    # Forward pass with LoRA-augmented weights
    # ------------------------------------------------------------------
    def forward_with_adapters(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard forward pass returning logits.
        LoRA hooks (if injected) are automatically active.
        """
        assert self.model is not None, "Call .load() first"
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_model_config(self) -> Dict:
        """Return key model dimensions for downstream modules."""
        assert self.model is not None, "Call .load() first"
        cfg = self.model.config
        return {
            "hidden_dim": cfg.hidden_size,
            "num_layers": cfg.num_hidden_layers,
            "num_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            "vocab_size": cfg.vocab_size,
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", 8192),
        }

    @property
    def hidden_dim(self) -> int:
        assert self.model is not None, "Call .load() first"
        return self.model.config.hidden_size

    def unload(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU cache cleared")
