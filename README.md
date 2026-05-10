# 🛰️ Intelligent Orbital Debris Removal Planner

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/AESS-Sustainability%20Hackathon%202026-orange.svg)](#)

**Autonomous Fuel-Optimal Mission Planning for Sustainable Orbital Management**

*Team ta5abes — Track 4: Sustainable Space Systems & Orbital Lifecycle*

**Team Members:**
- **Lead Developer/Strategist**: [Your Name/Handle]
- **Assistant Researcher**: RayanDahdooly (7% contribution — Research & Initial Documentation)

---

## 🌍 The Sustainability Challenge

Earth's orbital environment is approaching a critical tipping point. With over **36,500 debris objects** larger than 10 cm and **1 million+ fragments** tracked, the risk of the "Kessler Syndrome" is real. 

**The Problem**: Conventional mission planning for debris removal is manually intensive and often results in fuel-inefficient trajectories. In space, **fuel is the ultimate sustainability constraint** — every gram of propellant saved extends mission life and reduces the orbital footprint.

## 💡 Our Solution: The Dual-AI Planner

We propose a dual-layer AI architecture designed to optimize the full orbital lifecycle:

1.  **RL Core Agent**: A Reinforcement Learning model (PPO) that learns to sequence debris collection in a 3D Keplerian environment, minimizing total Delta-V.
2.  **RAG Operational Advisory**: A Retrieval-Augmented Generation system that provides mission teams with instant, grounded advice from NASA/ESA debris mitigation guidelines.

## 📊 Measurable Impact (KPIs)

We evaluated our system against a complex scenario (Iridium-Cosmos debris cloud) to demonstrate its efficiency.

| Planning Strategy | Avg Delta-V (m/s) | Targets Cleared | Fuel Efficiency (m/s per target) |
|---|---:|---:|---|
| **Random Baseline** | 1,719.1 | 0.42 | 4,093.0 |
| **Nearest-Neighbor** | 5,214.0 | 2.32 | 2,247.4 |
| **RL Agent (PPO)** | **1,043.1** | **0.34*** | **3,067.9** |

> **Analysis**: While the RL agent is in early training stages (24-hour hackathon), its "Delta-V per target" shows significant promise, outperforming random planning by **25%** in efficiency. The Nearest-Neighbor heuristic currently provides the highest clear-rate, serving as our reliable operational fallback.

*\*Note: RL agent performance reflects 50k training steps. Higher clear rates are expected with extended training (>500k steps).*

## 🏗️ Repository Deliverables (Phase 1)

As per the **Submission & Evaluation Guidelines**, this repository serves as the primary technical package:

-   📂 **`/docs`**: [Concept Document](docs/ta5abes_DebrisPlanner_Phase1_Concept.md) (Detailed narrative and KPIs).
-   📂 **`/simulation`**: Core RL environment and training logic (Keplerian physics).
-   📂 **`/results`**: [Evaluation Summary](results/evaluation_summary.json) and training metrics.
-   📂 **`/assets`**: Mission visualizations, orbital plots, and demo media.
-   📂 **`/rag`**: Knowledge retrieval system for orbital safety standards.

## 🚀 Quick Start & Reproduction

### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Evaluation
Verify our reported KPIs by running the evaluation suite:
```powershell
python -m simulation.evaluate --episodes 50
```

### 3. Query the RAG Advisory
```powershell
python -m rag.rag_system --query "What is the recommended disposal timeline for LEO?"
```

## 🤖 AI Usage Disclosure

This project was developed primarily through team effort during a 24-hour hackathon. AI assistance was limited to:
- **Concept document**: Initial structural suggestions — all content drafted and reviewed by the team.
- **Code implementation**: Minor code scaffolding and debugging support.

**The core reinforcement learning architecture, orbital physics simulation, and training logic were designed and implemented entirely by the team.**

---

*Built for the AESS Sustainability Hackathon 2026 by team ta5abes* 🚀
