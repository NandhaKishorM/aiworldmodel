"""
Neuro-Symbolic Bottleneck — Gumbel-Softmax Concept Projection
================================================================
Projects continuous hidden states h_t ∈ ℝ^d onto a discrete symbolic
vocabulary of K concept vectors using the Gumbel-Softmax trick, enabling
differentiable discrete sampling for the symbolic constraint engine.

Math:
  π = h_t · V^T                         (logits over K concepts)
  z_i = exp((π_i + g_i) / τ) / Σ_j ...  (Gumbel-Softmax sample)

As τ → 0, z converges to a one-hot vector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ProjectionResult:
    """Output of the Gumbel-Softmax projection."""
    soft_z: torch.Tensor          # (batch, seq_len, K) — differentiable soft assignment
    hard_z: torch.Tensor          # (batch, seq_len, K) — one-hot (straight-through)
    concept_ids: torch.Tensor     # (batch, seq_len) — argmax concept indices
    logits: torch.Tensor          # (batch, seq_len, K) — raw logits before Gumbel
    entropy: torch.Tensor         # (batch, seq_len) — per-position assignment entropy


class ConceptCodebook(nn.Module):
    """
    Learnable codebook of K concept embedding vectors.

    V ∈ ℝ^{K × d} where each row is a concept prototype vector.
    """

    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.normalize = normalize

        # Codebook: K concept vectors
        self.embeddings = nn.Parameter(torch.empty(num_concepts, hidden_dim))
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Xavier-uniform init for balanced initial logit magnitudes."""
        nn.init.xavier_uniform_(self.embeddings)

    def get_codebook(self) -> torch.Tensor:
        """Return (optionally L2-normalized) concept vectors."""
        if self.normalize:
            return F.normalize(self.embeddings, p=2, dim=-1)
        return self.embeddings

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute logits: π = h · V^T

        h : (batch, seq_len, d)
        returns: (batch, seq_len, K)
        """
        V = self.get_codebook()
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h @ V.T


class GumbelSoftmaxProjection(nn.Module):
    """
    Full Gumbel-Softmax projection layer with temperature annealing.

    Maps continuous hidden states → discrete concept assignments
    while maintaining differentiability for backprop through the
    symbolic constraint engine.
    """

    def __init__(
        self,
        num_concepts: int = 128,
        hidden_dim: int = 1152,
        tau_start: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal_rate: float = 0.9,
        hard: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.tau = tau_start
        self.tau_start = tau_start
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        self.hard = hard

        # Concept codebook
        self.codebook = ConceptCodebook(
            num_concepts=num_concepts,
            hidden_dim=hidden_dim,
            normalize=normalize,
        )

        # Optional projection if hidden_dim doesn't match
        self.projection: Optional[nn.Linear] = None

        logger.info(
            f"GumbelSoftmax projection: K={num_concepts}, d={hidden_dim}, "
            f"τ={tau_start}→{tau_min}"
        )

    def set_projection(self, input_dim: int) -> None:
        """Add a linear projection if model hidden dim differs from codebook dim."""
        if input_dim != self.codebook.hidden_dim:
            self.projection = nn.Linear(
                input_dim, self.codebook.hidden_dim, bias=False
            )
            logger.info(
                f"Added projection: {input_dim} → {self.codebook.hidden_dim}"
            )

    def forward(
        self,
        h: torch.Tensor,
        tau: Optional[float] = None,
    ) -> ProjectionResult:
        """
        Project hidden states to concept space via Gumbel-Softmax.

        Parameters
        ----------
        h   : (batch, seq_len, d) hidden states from transformer
        tau : override temperature (uses internal state if None)

        Returns
        -------
        ProjectionResult with soft_z, hard_z, concept_ids, logits, entropy
        """
        if self.projection is not None:
            h = self.projection(h)

        tau = tau or self.tau

        # Compute logits over concepts: (batch, seq_len, K)
        logits = self.codebook(h)

        # Gumbel-Softmax sampling
        if self.training or tau > 0:
            soft_z = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            soft_z = F.softmax(logits, dim=-1)

        # Straight-through hard sample (for symbolic engine)
        hard_z = self._straight_through(soft_z)

        # Argmax concept IDs (non-differentiable, for rule evaluation)
        concept_ids = logits.argmax(dim=-1)

        # Per-position entropy for monitoring concept utilization
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        return ProjectionResult(
            soft_z=soft_z,
            hard_z=hard_z,
            concept_ids=concept_ids,
            logits=logits,
            entropy=entropy,
        )

    @staticmethod
    def _straight_through(soft_z: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator: hard one-hot in forward,
        soft gradient in backward.
        """
        index = soft_z.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(soft_z).scatter_(-1, index, 1.0)
        # Gradient flows through soft_z
        return (hard - soft_z).detach() + soft_z

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------
    def anneal_temperature(self) -> float:
        """Decay temperature by anneal_rate, clamped to tau_min."""
        self.tau = max(self.tau_min, self.tau * self.tau_anneal_rate)
        return self.tau

    def reset_temperature(self) -> None:
        """Reset temperature to initial value (session start)."""
        self.tau = self.tau_start

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def concept_utilization(self, concept_ids: torch.Tensor) -> dict:
        """
        Compute concept usage statistics for monitoring codebook collapse.

        concept_ids: (batch, seq_len)
        """
        flat = concept_ids.flatten()
        K = self.codebook.num_concepts
        counts = torch.bincount(flat, minlength=K).float()
        total = flat.numel()

        # Perplexity: exp(entropy of usage distribution)
        probs = counts / total
        log_probs = (probs + 1e-10).log()
        perplexity = (-(probs * log_probs).sum()).exp().item()

        return {
            "active_concepts": (counts > 0).sum().item(),
            "total_concepts": K,
            "utilization_ratio": (counts > 0).sum().item() / K,
            "perplexity": perplexity,
            "top_5_concepts": counts.topk(5).indices.tolist(),
        }
