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
    Self-supervised loss via random token masking.

    Randomly masks a fraction of input tokens and computes cross-entropy
    between the model's predictions at masked positions and the original
    tokens. No external labels required — the input IS the label.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute masked self-supervised loss.

        Parameters
        ----------
        logits      : (batch, seq_len, vocab_size) model output logits
        input_ids   : (batch, seq_len) original token IDs
        attention_mask : (batch, seq_len) which tokens to consider
        mask_indices : pre-computed mask (if None, randomly generated)

        Returns
        -------
        (loss_scalar, num_masked_tokens)
        """
        batch_size, seq_len, vocab_size = logits.shape

        if mask_indices is None:
            mask_indices = self._generate_mask(
                batch_size, seq_len, attention_mask, logits.device
            )

        # Shift for next-token prediction alignment:
        # logits[t] predicts token at position t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = mask_indices[:, 1:].contiguous()

        # Apply mask: only compute loss on masked positions
        labels = shift_labels.clone()
        labels[~shift_mask] = self.ignore_index

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        num_masked = shift_mask.sum().item()
        return loss, int(num_masked)

    def _generate_mask(
        self,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate random boolean mask for token positions."""
        mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio

        # Don't mask padding or special tokens
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        # Ensure at least 1 token is masked
        if mask.sum() == 0:
            valid = attention_mask.bool() if attention_mask is not None else \
                torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            valid_positions = valid.nonzero(as_tuple=False)
            if len(valid_positions) > 0:
                idx = valid_positions[0]
                mask[idx[0], idx[1]] = True

        return mask


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
        lambda_sym: float = 0.5,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.lambda_sym = lambda_sym

        self.self_sup_loss = SelfSupervisedLoss(mask_ratio=mask_ratio)
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
