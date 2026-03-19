"""
TTT Loss Functions — Self-Supervised + Symbolic Consistency
=============================================================
Implements the test-time training objective:

  L_TTT = L_self_sup(x_test) + λ · L_sym(z_t, z_{t+1})

L_self_sup: Masked token prediction (no external labels needed)
L_sym:      Hinge loss from symbolic constraint violations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from symbolic.constraint_engine import ConstraintEngine, ConstraintResult
from models.neuro_symbolic_bottleneck import ProjectionResult

logger = logging.getLogger(__name__)


@dataclass
class TTTLossResult:
    """Detailed breakdown of the TTT loss computation."""
    total_loss: torch.Tensor
    self_sup_loss: torch.Tensor
    symbolic_loss: torch.Tensor
    constraint_result: Optional[ConstraintResult] = None
    mask_ratio_actual: float = 0.0
    num_masked_tokens: int = 0


class SelfSupervisedLoss(nn.Module):
    """
    Self-supervised loss via standard Causal Language Modeling (CLM).

    Predicts the next token using standard cross-entropy on the input sequence.
    This provides a completely stable gradient target across TTT steps,
    unlike random masking which causes loss to bounce wildly.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute standard auto-regressive loss.

        Returns
        -------
        (loss_scalar, num_valid_tokens)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Shift for next-token prediction alignment:
        # logits[t] predicts token at position t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        labels = shift_labels.clone()
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
            labels[~shift_mask.bool()] = self.ignore_index

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        num_valid = (labels != self.ignore_index).sum().item()
        return loss, int(num_valid)


class SymbolicConsistencyLoss(nn.Module):
    """
    Wrapper around the ConstraintEngine that produces a differentiable
    loss from symbolic rule violations on projected concept sequences.
    """

    def __init__(
        self,
        constraint_engine: ConstraintEngine,
    ) -> None:
        super().__init__()
        self.constraint_engine = constraint_engine

    def forward(
        self,
        projection_result: ProjectionResult,
    ) -> Tuple[torch.Tensor, ConstraintResult]:
        """
        Compute symbolic consistency loss.

        Uses the soft_z (Gumbel-Softmax output) to maintain
        differentiability through the symbolic evaluation.

        Parameters
        ----------
        projection_result : output from GumbelSoftmaxProjection

        Returns
        -------
        (loss_tensor, constraint_result)
        """
        soft_z = projection_result.soft_z  # (batch, seq_len, K)

        result = self.constraint_engine.evaluate(
            z_sequence=soft_z,
            return_details=True,
        )
        return result.loss, result


class TTTLoss(nn.Module):
    """
    Combined Test-Time Training loss:

      L_TTT = L_self_sup + λ · L_sym

    This is the objective that drives fast-weight updates at test time.
    """

    def __init__(
        self,
        constraint_engine: ConstraintEngine,
        lambda_sym: float = 10.0,
    ) -> None:
        super().__init__()
        self.lambda_sym = lambda_sym

        self.self_sup_loss = SelfSupervisedLoss()
        self.symbolic_loss = SymbolicConsistencyLoss(constraint_engine)

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        projection_result: ProjectionResult,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> TTTLossResult:
        """
        Compute the full TTT objective.

        Parameters
        ----------
        logits           : (batch, seq_len, vocab) model output
        input_ids        : (batch, seq_len) original tokens
        projection_result: Gumbel-Softmax projection of hidden states
        attention_mask   : optional

        Returns
        -------
        TTTLossResult with all loss components and diagnostics
        """
        # Self-supervised loss (masked token prediction)
        l_self_sup, num_masked = self.self_sup_loss(
            logits, input_ids, attention_mask
        )

        # Symbolic consistency loss
        l_sym, constraint_result = self.symbolic_loss(projection_result)

        # Combined TTT objective
        total_loss = l_self_sup + self.lambda_sym * l_sym

        return TTTLossResult(
            total_loss=total_loss,
            self_sup_loss=l_self_sup,
            symbolic_loss=l_sym,
            constraint_result=constraint_result,
            mask_ratio_actual=num_masked / max(input_ids.numel(), 1),
            num_masked_tokens=num_masked,
        )
