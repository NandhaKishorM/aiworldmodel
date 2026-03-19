"""
Constraint Engine — Differentiable Symbolic Violation Evaluator
================================================================
Wraps the SymbolicWorldModel to produce differentiable hinge-loss
signals from sequences of Gumbel-Softmax concept projections.

Math:
  L_sym = max(0, C(z_t, z_{t+1}) - ε)

Where C is the aggregated constraint violation from all rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from symbolic.world_model import SymbolicWorldModel, SymbolicState

logger = logging.getLogger(__name__)


@dataclass
class ConstraintResult:
    """Output of the constraint engine evaluation."""
    loss: torch.Tensor                          # scalar hinge loss (differentiable)
    total_violation: float                      # raw violation magnitude
    num_violations: int                         # count of violations > ε
    per_step_details: List[Dict[str, float]]    # per-transition breakdown
    per_rule_totals: Dict[str, float] = field(default_factory=dict)


class ConstraintEngine:
    """
    Evaluates symbolic constraints on concept sequences and produces
    differentiable hinge-loss for TTT backpropagation.
    """

    def __init__(
        self,
        world_model: SymbolicWorldModel,
        epsilon: float = 0.01,
        reduction: str = "mean",
    ) -> None:
        """
        Parameters
        ----------
        world_model : the symbolic rule registry
        epsilon     : hinge loss margin — violations below ε are ignored
        reduction   : "mean" or "sum" over sequence positions
        """
        self.world_model = world_model
        self.epsilon = epsilon
        self.reduction = reduction

    def evaluate(
        self,
        z_sequence: torch.Tensor,
        return_details: bool = True,
    ) -> ConstraintResult:
        """
        Evaluate symbolic constraints across a sequence of concept distributions.

        Parameters
        ----------
        z_sequence : (batch, seq_len, K) or (seq_len, K) soft concept assignments
        return_details : whether to compute per-step breakdowns

        Returns
        -------
        ConstraintResult with differentiable loss
        """
        # Handle batch dim
        if z_sequence.dim() == 3:
            # Process first batch element (TTT typically batch_size=1)
            z_seq = z_sequence[0]
        else:
            z_seq = z_sequence

        seq_len = z_seq.shape[0]
        device = z_seq.device
        dtype = z_seq.dtype

        if seq_len < 2:
            return ConstraintResult(
                loss=torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True),
                total_violation=0.0,
                num_violations=0,
                per_step_details=[],
            )

        # Evaluate all consecutive pairs
        step_losses: List[torch.Tensor] = []
        all_details: List[Dict[str, float]] = []
        per_rule_totals: Dict[str, float] = {}
        num_violations = 0

        for t in range(seq_len - 1):
            z_t = z_seq[t]
            z_t1 = z_seq[t + 1]

            # Differentiable evaluation
            violation, details = self.world_model.evaluate_transition_soft(z_t, z_t1)

            # Hinge loss: max(0, C - ε)
            step_loss = F.relu(violation - self.epsilon)
            step_losses.append(step_loss)

            if step_loss.item() > 0:
                num_violations += 1

            # Accumulate per-rule totals
            for rule_name, v in details.items():
                per_rule_totals[rule_name] = per_rule_totals.get(rule_name, 0.0) + v

            if return_details:
                all_details.append(details)

        # Aggregate
        if step_losses:
            stacked = torch.stack(step_losses)
            if self.reduction == "mean":
                total_loss = stacked.mean()
            else:
                total_loss = stacked.sum()
        else:
            total_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        return ConstraintResult(
            loss=total_loss,
            total_violation=sum(l.item() for l in step_losses),
            num_violations=num_violations,
            per_step_details=all_details,
            per_rule_totals=per_rule_totals,
        )

    def evaluate_hard(
        self,
        concept_ids: torch.Tensor,
    ) -> Dict[str, int]:
        """
        Non-differentiable evaluation using hard concept IDs.
        For logging and diagnostics only.

        concept_ids : (seq_len,) integer concept indices
        """
        if concept_ids.dim() > 1:
            concept_ids = concept_ids[0]

        total_violations: Dict[str, int] = {}
        for t in range(len(concept_ids) - 1):
            state_t = SymbolicState(concept_id=concept_ids[t].item(), position=t)
            state_t1 = SymbolicState(concept_id=concept_ids[t + 1].item(), position=t + 1)

            violations = self.world_model.evaluate_transition(state_t, state_t1)
            for rule_name in violations:
                total_violations[rule_name] = total_violations.get(rule_name, 0) + 1

        return total_violations
