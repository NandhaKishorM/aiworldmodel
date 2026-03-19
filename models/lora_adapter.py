"""
LoRA Adapter — Fast Weights for Test-Time Training
====================================================
Implements low-rank adaptation matrices A ∈ ℝ^{r×d_in} and B ∈ ℝ^{d_out×r}.
The adapter output ΔW·x = B(Ax) is added to the frozen layer output via
forward hooks. Only A and B are updated during TTT; base weights stay frozen.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    A single LoRA adapter: ΔW = (α/r) · B @ A

    Parameters
    ----------
    d_in   : input dimension of the target linear layer
    d_out  : output dimension of the target linear layer
    rank   : low-rank dimension r
    alpha  : scaling factor
    dropout: dropout applied before the LoRA path
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A: ℝ^{r × d_in}  — projects input down to rank
        self.A = nn.Parameter(torch.empty(rank, d_in))
        # B: ℝ^{d_out × r} — projects back up
        self.B = nn.Parameter(torch.empty(d_out, rank))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-uniform for A, zero-init for B (standard LoRA convention)."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def reset(self) -> None:
        """Re-initialize to zero-effect state (call at start of each TTT session)."""
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute adapter contribution: scaling · B @ A @ x.T transposed back.

        x : (batch, seq_len, d_in)
        returns: (batch, seq_len, d_out)
        """
        x = self.dropout(x)
        # x @ A^T  → (batch, seq_len, rank)
        # result @ B^T → (batch, seq_len, d_out)
        return self.scaling * (x @ self.A.T @ self.B.T)

    @property
    def adapter_norm(self) -> float:
        """Frobenius norm of ΔW = B @ A for monitoring."""
        with torch.no_grad():
            delta_w = self.B @ self.A
            return delta_w.norm().item()


class LoRAInjector:
    """
    Injects LoRA adapters into a frozen pre-trained model via forward hooks.

    The injector finds target modules (e.g., q_proj, v_proj) and attaches
    hooks that add the LoRA output to each module's forward result.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        target_modules: Optional[List[str]] = None,
        target_layers: str | List[int] = "all",
        dropout: float = 0.0,
    ) -> None:
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.target_layers = target_layers
        self.dropout = dropout

        self.adapters: Dict[str, LoRALayer] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self._inject()

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def _inject(self) -> None:
        """Walk the model tree, find target modules, create LoRA adapters."""
        injected = 0
        for full_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self._should_inject(full_name):
                continue

            d_in = module.in_features
            d_out = module.out_features

            adapter = LoRALayer(
                d_in=d_in,
                d_out=d_out,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            ).to(device=next(module.parameters()).device, dtype=next(module.parameters()).dtype)

            self.adapters[full_name] = adapter

            # Forward hook: add adapter output to module output
            hook = module.register_forward_hook(
                self._make_hook(adapter)
            )
            self._hooks.append(hook)
            injected += 1

        logger.info(
            f"Injected {injected} LoRA adapters (rank={self.rank}, "
            f"alpha={self.alpha}) into: {self.target_modules}"
        )

    def _should_inject(self, full_name: str) -> bool:
        """Check if this module name matches target modules and target layers."""
        # Check module name suffix
        name_parts = full_name.split(".")
        module_name = name_parts[-1]
        if module_name not in self.target_modules:
            return False

        # Check layer index
        if self.target_layers == "all":
            return True

        # Extract layer index from name like "model.layers.5.self_attn.q_proj"
        for part in name_parts:
            if part.isdigit():
                layer_idx = int(part)
                if isinstance(self.target_layers, list):
                    return layer_idx in self.target_layers
        return True

    @staticmethod
    def _make_hook(adapter: LoRALayer):
        """Create a forward hook closure for the given adapter."""
        def hook_fn(module: nn.Module, input: Tuple, output: torch.Tensor) -> torch.Tensor:
            # input[0] is the input tensor to the linear layer
            x = input[0]
            lora_out = adapter(x)
            return output + lora_out
        return hook_fn

    # ------------------------------------------------------------------
    # Parameter access (for optimizer)
    # ------------------------------------------------------------------
    def get_parameters(self) -> List[nn.Parameter]:
        """Return all adapter parameters — the ONLY trainable params for TTT."""
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        return params

    def get_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Return named adapter parameters for debugging."""
        named = []
        for name, adapter in self.adapters.items():
            for pname, param in adapter.named_parameters():
                named.append((f"lora.{name}.{pname}", param))
        return named

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def reset_all(self) -> None:
        """Reset all adapters to zero-effect state (session start)."""
        for adapter in self.adapters.values():
            adapter.reset()
        logger.debug("All LoRA adapters reset to initial state")

    def get_adapter_norms(self) -> Dict[str, float]:
        """Return Frobenius norm of each adapter's ΔW for monitoring."""
        return {name: adapter.adapter_norm for name, adapter in self.adapters.items()}

    def total_adapter_norm(self) -> float:
        """Aggregate adapter norm across all injected layers."""
        return sum(adapter.adapter_norm for adapter in self.adapters.values())

    def num_trainable_params(self) -> int:
        """Count total trainable parameters across all adapters."""
        return sum(p.numel() for p in self.get_parameters())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def remove_all(self) -> None:
        """Remove all hooks and adapters (restore original model behavior)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.adapters.clear()
        logger.info("All LoRA hooks removed")

    def save_adapters(self, path: str) -> None:
        """Save adapter state dicts for episodic memory / offline consolidation."""
        state = {name: adapter.state_dict() for name, adapter in self.adapters.items()}
        torch.save(state, path)
        logger.info(f"Adapter states saved to {path}")

    def load_adapters(self, path: str) -> None:
        """Load previously saved adapter states."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        for name, adapter_state in state.items():
            if name in self.adapters:
                self.adapters[name].load_state_dict(adapter_state)
        logger.info(f"Adapter states loaded from {path}")
