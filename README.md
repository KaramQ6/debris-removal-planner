# 🛰️ Intelligent Orbital Debris Removal Planner

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Autonomous Propellant-Constrained Mission Planning for Sustainable Orbital Management**

**Author:** [Karam Khasawneh](https://github.com/KaramQ6) (KaramQ5@ieee.org)  
**Affiliation:** Department of Robotics and Artificial Intelligence, Jadara University, Irbid, Jordan  
---

## 🌍 The Sustainability Challenge

Earth's orbital environment is approaching a critical tipping point. With over **36,500 debris objects** larger than 10 cm and **1 million+ fragments** tracked, the risk of the "Kessler Syndrome" is real. 

**The Problem**: Conventional mission planning for debris removal is manually intensive and often results in fuel-inefficient trajectories. In space, **fuel is the ultimate sustainability constraint** — every gram of propellant saved extends mission life and reduces the orbital footprint.

## 💡 Our Solution: The Dual-AI Planner

We propose a dual-layer AI architecture designed to optimize the full orbital lifecycle:

1.  **RL Core Agent**: A Reinforcement Learning model (PPO) that learns to sequence debris collection in a 3D Keplerian environment, minimizing total Delta-V.
2.  **RAG Operational Advisory**: A Retrieval-Augmented Generation system that provides mission teams with instant, grounded advice from NASA/ESA debris mitigation guidelines.

## 📊 Measurable Impact (KPIs)

We evaluated our system against the **Medium Scenario** (100 episodes, 12,000 m/s fuel budget, 8 targets) to demonstrate its technical performance.

| Planning Strategy | Avg Delta-V (m/s) | Targets Cleared | Accuracy % | F1 Score | Fuel/Target (m/s) |
|---|---:|---:|---:|---:|---:|
| **Random Baseline** | 4,716.5 | 0.79 | 9.9% | 0.18 | 6,427.4 |
| **Nearest-Neighbor** | 8,399.6 | 3.15 | 39.4% | 0.57 | 3,099.2 |
| **Risk-Weighted** | 8,419.7 | 3.14 | 39.2% | 0.56 | 3,083.9 |
| **MaskablePPO RL Core**| **10,150.4** | **2.22** | **27.8%** | **0.44** | **5,903.2** |

> **Analysis**: The deterministic Nearest-Neighbor and Risk-Weighted Greedy baselines achieve the highest target clearance and best fuel efficiency per target. The MaskablePPO agent achieves **competitive but not superior** performance under static evaluation (2.22 vs 3.14 targets). Its primary advantage is **constant-time neural inference** (under two milliseconds per decision) suitable for online re-planning when debris catalogs change dynamically — an advantage not tested here but the key motivation for the RL approach.

*\*Note: Evaluation performed on LEO clusters. MaskablePPO model trained for 1.5M total steps on curriculum stochastic mix.*

## 🌐 Real-World Scenario Evaluation (Data_For_Test.json)

**Fine-tuned model performance on CelesTrak satellite database scenarios** (19,721 objects, 50-step missions, 12,000 m/s budget):

| Model Version | Avg Targets | Accuracy % | Δ-V (m/s) | Completion % | Fuel/Target |
|---|---:|---:|---:|---:|---:|
| **MaskablePPO (Base)** | 1.79 | 22.4% | 8,648.8 | 89.0% | 4,836 |
| **MaskablePPO (Fine-tuned)** | **1.90** | **23.8%** | **8,612.3** | **91.0%** | **4,532** |

**Improvement**: Fine-tuning on CelesTrak scenarios increased target collection (+6.1%), accuracy (+1.4%), and fuel efficiency (-6.3% per target). The fine-tuned model (`ppo_debris_finetuned.zip`) is now the **production model** for this system.

> **Metric Clarifications**:  
> - **Accuracy & F1**: Operational tracking metrics (not strict ML classification); Precision is fixed at 1.0 because the agent collects real targets only (no false positives).  
> - **Completion %**: Episodes clearing ≥1 target.  
> - **Fuel/Target**: Average ΔV consumed per target cleared across all episodes.

## 🏗️ Repository Structure

- 📂 **`/docs`**: [Concept Document](docs/ta5abes_DebrisPlanner_Phase1_Concept.md) (detailed narrative and KPIs).
- 📂 **`/simulation`**: Core RL environment, training loop, baseline policies, and evaluation.
- 📂 **`/rag`**: BM25 retrieval-augmented advisor over NASA/ESA mitigation standards.
- 📂 **`/results`**: [Evaluation summary](results/evaluation_summary.json), training history, trained model.
- 📂 **`/assets`**: 3D mission visualizations and interactive Plotly HTMLs.
- 📂 **`/notebooks`**: [End-to-end walkthrough notebook](notebooks/walkthrough.ipynb) — reproduces every headline result.
- 📂 **`/tests`**: 43 unit tests (`pytest`) covering orbital mechanics, scenarios, RAG indexing, and reward shaping.

## 🚀 Quick Start & Reproduction

### 1. Setup environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the test suite
```powershell
pytest tests/
```

### 3. Reproduce the headline KPIs
```powershell
python -m simulation.evaluate --episodes 100 `
    --model-path results/models/ppo_debris_finetuned.zip `
    --scenario medium --fuel 12000
```

Or test on real-world debris data:
```powershell
python -m simulation.evaluate --episodes 100 `
    --model-path results/models/ppo_debris_finetuned.zip `
    --scenario Data_For_Test.json --fuel 12000 --max-steps 50
```

### 4. Train from scratch (optional — ~15 min on a modern GPU)
```powershell
python -m simulation.train --timesteps 1500000 --scenario curriculum --fuel 12000
```

### 5. Query the RAG advisor
```powershell
python -m rag.rag_system --query "What is the recommended disposal timeline for LEO?"
```

### 6. Explore everything interactively
Open [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) to step through the full pipeline.

## 🤖 AI Usage Disclosure

This project was developed for sustainable orbital planning. AI assistance was limited to minor code scaffolding and initial document layout suggestions, with all core logic designed and verified by the author.

**The core reinforcement learning architecture, orbital physics simulation, and training logic were designed and implemented entirely by the author.**

---

*Intelligent Space Systems Research Group* 🚀
