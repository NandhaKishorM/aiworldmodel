"""
TTT Metrics — Telemetry and Monitoring
=========================================
Dataclasses and utilities for tracking TTT performance,
symbolic violations, adapter dynamics, and latency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TTTMetrics:
    """Aggregated metrics from a TTT inference session."""

    # Per-step metrics
    loss_per_step: List[float] = field(default_factory=list)
    self_sup_loss_per_step: List[float] = field(default_factory=list)
    symbolic_loss_per_step: List[float] = field(default_factory=list)
    grad_norm_per_step: List[float] = field(default_factory=list)
    adapter_norm_per_step: List[float] = field(default_factory=list)
    violations_per_step: List[int] = field(default_factory=list)
    temperature_per_step: List[float] = field(default_factory=list)

    # Aggregate
    total_ttt_steps: int = 0
    total_time_ms: float = 0.0
    ttt_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    final_adapter_norm: float = 0.0
    total_violations: int = 0

    # Concept utilization
    active_concepts: int = 0
    concept_perplexity: float = 0.0

    @classmethod
    def from_ttt_result(cls, result) -> TTTMetrics:
        """Build metrics from a TTTResult object."""
        metrics = cls()
        for step in result.step_metrics:
            metrics.loss_per_step.append(step.total_loss)
            metrics.self_sup_loss_per_step.append(step.self_sup_loss)
            metrics.symbolic_loss_per_step.append(step.symbolic_loss)
            metrics.grad_norm_per_step.append(step.grad_norm)
            metrics.adapter_norm_per_step.append(step.adapter_norm)
            metrics.violations_per_step.append(step.num_violations)
            metrics.temperature_per_step.append(step.temperature)

        metrics.total_ttt_steps = len(result.step_metrics)
        metrics.total_time_ms = result.total_time_ms
        metrics.ttt_time_ms = result.ttt_time_ms
        metrics.generation_time_ms = result.generation_time_ms
        metrics.final_adapter_norm = result.final_adapter_norm
        metrics.total_violations = sum(metrics.violations_per_step)

        if result.concept_utilization:
            metrics.active_concepts = result.concept_utilization.get("active_concepts", 0)
            metrics.concept_perplexity = result.concept_utilization.get("perplexity", 0.0)

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Pretty-print summary for console output."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║        TTT Inference Metrics             ║",
            "╠══════════════════════════════════════════╣",
        ]

        if self.loss_per_step:
            lines.append(f"║  Total Steps       : {self.total_ttt_steps:<19}║")
            lines.append(f"║  Initial Loss      : {self.loss_per_step[0]:<19.4f}║")
            lines.append(f"║  Final Loss        : {self.loss_per_step[-1]:<19.4f}║")
            improvement = (
                (self.loss_per_step[0] - self.loss_per_step[-1])
                / max(self.loss_per_step[0], 1e-8) * 100
            )
            lines.append(f"║  Loss Improvement  : {improvement:<18.1f}%║")
        else:
            lines.append(f"║  TTT Disabled (standard inference)       ║")

        lines.append(f"║  Total Violations  : {self.total_violations:<19}║")
        lines.append(f"║  Adapter Norm      : {self.final_adapter_norm:<19.4f}║")
        lines.append(f"║  Active Concepts   : {self.active_concepts:<19}║")
        lines.append(f"║  Concept Perplexity: {self.concept_perplexity:<19.2f}║")
        lines.append(f"╠══════════════════════════════════════════╣")
        lines.append(f"║  TTT Time          : {self.ttt_time_ms:<18.0f}ms║")
        lines.append(f"║  Gen Time          : {self.generation_time_ms:<18.0f}ms║")
        lines.append(f"║  Total Time        : {self.total_time_ms:<18.0f}ms║")
        lines.append(f"╚══════════════════════════════════════════╝")

        return "\n".join(lines)
