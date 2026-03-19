# Neuro-Symbolic Test-Time Training (NS-TTT)

> A dual-speed neuro-symbolic transformer that adapts at inference time using symbolic constraint violations as loss signals. Built on **Gemma-3-1B-IT** with LoRA fast weights and Gumbel-Softmax concept projection.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-red)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Prompt x_test                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │  Base Model   │    │  LoRA Fast Weights (A, B)        │   │
│  │  (Frozen)     │◄──►│  Updated via TTT gradient steps  │   │
│  │  Gemma 1B     │    │  Δy = (α/r) · B @ A @ x         │   │
│  └──────┬───────┘    └──────────────────────────────────┘   │
│         │                                                    │
│         ▼ hidden states h_t                                  │
│  ┌──────────────────────────────┐                           │
│  │  Gumbel-Softmax Bottleneck   │                           │
│  │  π = h_t · V^T               │                           │
│  │  z = softmax((π + g) / τ)    │                           │
│  └──────────────┬───────────────┘                           │
│                 │ soft concept z_t                            │
│                 ▼                                             │
│  ┌──────────────────────────────┐                           │
│  │  Symbolic World Model        │                           │
│  │  Rules: type consistency,    │  C(z_t, z_{t+1}) > 0     │
│  │  temporal order, exclusion   │  = constraint violation    │
│  └──────────────┬───────────────┘                           │
│                 │                                             │
│                 ▼                                             │
│  ┌──────────────────────────────┐                           │
│  │  L_TTT = L_ss + λ · L_sym   │                           │
│  │  ∇_{A,B} L_TTT → update     │                           │
│  └──────────────────────────────┘                           │
│                                                             │
│  Final: y = Transformer(x; W_base + BA)                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/aiworldmodel.git
cd aiworldmodel
pip install -r requirements.txt
```

### Single Prompt

```bash
python main.py --prompt "The patient shows fever and productive cough. Diagnosis:"
```

### Interactive Mode

```bash
python main.py --interactive
```

### Standard Inference (No TTT)

```bash
python main.py --prompt "Explain quantum computing" --no-ttt
```

### Google Colab (T4 GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/aiworldmodel/blob/main/notebooks/ttt_demo.ipynb)

---

## Project Structure

```
├── config/ttt_config.yaml          # All hyperparameters
├── models/
│   ├── base_model.py               # Frozen Gemma-3-1B-IT loader
│   ├── lora_adapter.py             # LoRA fast weights (A, B matrices)
│   └── neuro_symbolic_bottleneck.py # Gumbel-Softmax concept projection
├── symbolic/
│   ├── world_model.py              # Rule registry + symbolic states
│   ├── constraint_engine.py        # Differentiable hinge loss
│   └── knowledge_graph.py          # Session-scoped entity tracking
├── ttt/
│   ├── loss.py                     # L_self_sup + L_sym
│   ├── optimizer.py                # Fast-weight SGD/Adam
│   └── ttt_engine.py               # Full TTT orchestration loop
├── inference/pipeline.py           # End-to-end inference API
├── utils/
│   ├── logging_utils.py            # Structured logging
│   └── metrics.py                  # TTT telemetry
├── tests/                          # pytest test suite
├── notebooks/ttt_demo.ipynb        # Colab demo notebook
└── main.py                         # CLI entry point
```

## Configuration

All hyperparameters are in `config/ttt_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.rank` | 16 | LoRA low-rank dimension |
| `lora.alpha` | 32 | LoRA scaling factor |
| `bottleneck.num_concepts` | 128 | Symbolic vocabulary size K |
| `bottleneck.tau_start` | 1.0 | Initial Gumbel-Softmax temperature |
| `ttt.num_update_steps` | 3 | Gradient steps at test time |
| `ttt.learning_rate` | 1e-4 | Fast-weight learning rate |
| `ttt.lambda_sym` | 0.5 | Symbolic loss weight |
| `ttt.epsilon` | 0.01 | Hinge loss margin |

## Key Equations

**Gumbel-Softmax Projection:**
```
π = h_t · V^T
z_i = exp((π_i + g_i) / τ) / Σ_j exp((π_j + g_j) / τ)
```

**TTT Objective:**
```
L_TTT = L_self_sup(x_test) + λ · max(0, C(z_t, z_{t+1}) - ε)
```

**Fast Weight Update:**
```
θ_fast^(n+1) = θ_fast^(n) - α · ∇_{θ_fast} L_TTT
```

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_bottleneck.py -v

# With coverage
python -m pytest tests/ -v --cov=. --cov-report=term-missing
```

## Adding Custom Rules

```yaml
# In config/ttt_config.yaml
symbolic:
  default_rules:
    - type: "mutual_exclusion"
      weight: 1.0
      pairs:
        - ["healthy", "diseased"]
        - ["increasing", "decreasing"]
```

Or programmatically:

```python
from symbolic.world_model import SymbolicWorldModel, MutualExclusionRule

world_model = SymbolicWorldModel()
world_model.add_rule(MutualExclusionRule(
    exclusion_pairs=[(10, 25), (30, 45)],
    weight=1.5,
))
```

## Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| T4 Colab | 16 GB | ✅ Recommended (BF16) |
| RTX 3060 | 12 GB | ✅ Works well |
| CPU | 8+ GB RAM | ⚠️ Slow but functional |

## License

MIT
