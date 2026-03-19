"""
Fast Weight Optimizer — Adapter-Only Parameter Updates
========================================================
Thin wrapper around PyTorch optimizers that ensures ONLY the LoRA
adapter parameters (A, B matrices) are updated during TTT.
The base model remains completely frozen.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FastWeightOptimizer:
    """
    Manages gradient-based updates for LoRA fast weights during TTT.

    Supports:
    - Adam / SGD optimization
    - Gradient clipping
    - N-step updates with metrics tracking
    - Zero-grad and state reset for session boundaries
    """

    def __init__(
        self,
        parameters: List[nn.Parameter],
        optimizer_type: str = "adam",
        learning_rate: float = 1e-4,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        self.learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm

        if not self.parameters:
            raise ValueError("No trainable parameters provided to FastWeightOptimizer")

        # Build optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters,
                lr=learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self._step_count = 0

        logger.info(
            f"FastWeightOptimizer: {optimizer_type.upper()}, "
            f"lr={learning_rate}, grad_clip={gradient_clip_norm}, "
            f"params={sum(p.numel() for p in self.parameters)}"
        )

    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """
        Perform one gradient step on the fast weights.

        Parameters
        ----------
        loss : scalar loss tensor (must have grad_fn)

        Returns
        -------
        Dict with gradient norm, parameter norm, and loss value
        """
        self.optimizer.zero_grad()

        # Backward pass — gradients flow only to adapter parameters
        loss.backward(retain_graph=False)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters,
            max_norm=self.gradient_clip_norm,
        ).item()

        # Optimizer step (updates A and B matrices ONLY)
        self.optimizer.step()

        self._step_count += 1

        # Parameter norm for monitoring
        param_norm = sum(p.data.norm().item() ** 2 for p in self.parameters) ** 0.5

        return {
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "loss": loss.item(),
            "step": self._step_count,
        }

    def zero_grad(self) -> None:
        """Clear all gradients."""
        self.optimizer.zero_grad()

    def reset(self) -> None:
        """Reset optimizer state (call at session start)."""
        self.optimizer.state.clear()
        self._step_count = 0
        self.zero_grad()

    @property
    def step_count(self) -> int:
        return self._step_count
