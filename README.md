# 🛰️ Intelligent Orbital Debris Removal Planner

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/AESS-Sustainability%20Hackathon%202026-orange.svg)](#)

**AI-Driven Fuel-Optimal Debris Collection with RAG Operational Advisory**

*Team ta5abes — Track 4: Sustainable Space Systems & Orbital Lifecycle*

---

## 🌍 Problem Statement

Earth's orbital environment is approaching a critical tipping point. ESA tracks over **36,500 debris objects** larger than 10 cm, with an estimated **1 million+ fragments** larger than 1 cm — each capable of catastrophically disabling an operational satellite.

The core unsolved problem is **autonomous planning of fuel-optimal removal missions** that efficiently sequence multi-target collection paths. Manual planning is slow, fuel-wasteful, and non-scalable.

## 💡 Our Solution

A dual-AI mission planning system combining:

| Component | Function | Technology |
|---|---|---|
| **RL Core Agent** | Plans fuel-optimal multi-target debris collection sequences | PPO (Stable-Baselines3) + Custom Gymnasium Env |
| **RAG Advisory** | On-demand operational guidance from NASA/ESA/IADC documents | BM25 retrieval over chunked document corpus |

## 📊 Key Results

Performance comparison across **100 episodes** with 8 debris targets in LEO (1,200 m/s fuel budget):

| Planning Strategy | Avg Delta-V (m/s) | Full-Clear Rate | Fuel Efficiency |
|---|---:|---:|---|
| **Random Baseline** | 1065.7 | 44.0% | Baseline |
| **Greedy Nearest-Neighbor** | 590.5 | 100.0% | +44.6% vs random |
| **Risk-Weighted Nearest** | 709.7 | 100.0% | +33.4% vs random |
| **RL Agent (PPO)** | **683.2** | **100.0%** | **+35.9% vs random** |

> **Key result**: The RL agent achieves a **35.9% reduction in fuel consumption** compared to random baseline planning, directly translating to longer mission lifetimes and more debris cleared per spacecraft.

*Note: Values are from simulation. See `results/` for reproducible outputs.*

## 🏗️ System Architecture

```
TLE Debris Data → Orbital Simulator → RL Agent (PPO) → Optimized Path → Mission Output
                                                                    ↓
                                        RAG Advisory ← On-demand queries
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Mission Planning Pipeline                        │
├──────────────┬──────────────────┬───────────────────────────────────┤
│  Data Input  │  RL Environment  │  Planning & Output               │
│              │                  │                                   │
│  • Debris    │  • State: pos,   │  • PPO agent selects next target │
│    catalog   │    fuel, targets  │  • Reward: -ΔV + risk bonus     │
│  • Risk      │  • Action: pick  │  • Output: optimal sequence     │
│    scores    │    next target    │  • Viz: 2D polar + 3D orbital   │
│  • Fuel      │  • Done: cleared │  • RAG: operational advisory    │
│    budget    │    or fuel=0      │                                  │
└──────────────┴──────────────────┴───────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or uv package manager

### Setup (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Baseline Evaluation

```powershell
python -m simulation.evaluate --episodes 100
```

### Train RL Agent

```powershell
# Quick training (~5 min)
python -m simulation.train --timesteps 50000

# Full training (~30 min, better convergence)
python -m simulation.train --timesteps 500000
```

### Evaluate Trained Model

```powershell
python -m simulation.evaluate --episodes 100 --model-path results\models\ppo_debris.zip
```

### Generate Visualizations

```powershell
# Mission path plots for all baseline policies
python -m simulation.visualize --all-policies

# With trained PPO model
python -m simulation.visualize --policy ppo --model-path results\models\ppo_debris.zip

# Generate comparison charts from evaluation data
python -m simulation.plot_results
```

### Query RAG Advisory

```powershell
# Single query
python -m rag.rag_system --docs docs --query "What should we do when delta-v budget is low?"

# Demo mode (runs pre-built example queries)
python -m rag.rag_system --docs docs --demo
```

## 📁 Repository Structure

| Path | Purpose |
|---|---|
| `simulation/orbit_env.py` | Custom Gymnasium debris-removal environment |
| `simulation/train.py` | PPO training with TensorBoard logging & checkpoints |
| `simulation/evaluate.py` | Baseline + RL policy evaluation with statistics |
| `simulation/visualize.py` | 2D polar + 3D orbital mission path visualization |
| `simulation/plot_results.py` | Publication-quality comparison charts |
| `simulation/policies.py` | Hand-crafted baseline policies (random, nearest, risk-weighted) |
| `simulation/callbacks.py` | Custom SB3 training callbacks |
| `simulation/scenario.py` | Mission scenario definitions and presets |
| `rag/rag_system.py` | Zero-dependency RAG advisory with BM25 scoring |
| `docs/` | Mission assumptions, IADC guidelines, best practices |
| `results/` | Evaluation data, charts, trained models, visualizations |

## ⚙️ Current Model Assumptions

1. **Orbital geometry**: 2D circular-orbit representation using angular positions
2. **Transfer cost**: Simplified delta-v estimate derived from angular separation (20 + 1.5 × angular_distance)
3. **Single-capture actions**: Each action targets exactly one debris object
4. **Fuel model**: Aggregate delta-v budget (m/s), no separate fuel mass tracking
5. **Termination**: Episode ends when all targets cleared, fuel depleted, or max steps reached
6. **Risk weighting**: Higher-risk objects receive stronger reward bonus
7. **Observation space**: [cos(θ), sin(θ), fuel_fraction] + per-target [cos(θ), sin(θ), risk, active]

## 🔮 Future Work

- [ ] Full 3D orbital mechanics using poliastro
- [ ] Lambert solver for precise transfer delta-v calculations
- [ ] Real TLE catalog integration (Space-Track.org API)
- [ ] Multi-spacecraft coordination
- [ ] Curriculum learning with progressive difficulty
- [ ] Integration with ESA Space Debris Office catalog API
- [ ] Atmospheric drag and perturbation modeling

## 🤖 AI Usage Disclosure

This project was developed primarily through team effort during a 24-hour hackathon. AI assistance was limited to:
- **Concept document**: Initial structural suggestions for the concept document — all content drafted and reviewed by the team.
- **Code implementation**: Minor code scaffolding and debugging support.

**Crucially, the core reinforcement learning model architecture, training logic, and orbital physics environment were designed, implemented, and trained entirely by the team.** All technical results and performance metrics are the direct result of our own model training and validation.

All technical decisions, architectural choices, and quantitative claims are the responsibility of team ta5abes. Baseline vs. optimized comparison values represent simulated results reproducible from the submitted code.

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the AESS Sustainability Hackathon 2026 by team ta5abes* 🚀
