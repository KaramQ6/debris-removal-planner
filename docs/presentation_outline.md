# Presentation Outline — Intelligent Orbital Debris Removal Planner

**Team ta5abes | AESS Sustainability Hackathon 2026**
**Target**: 8-10 slides, 5-7 minute presentation

---

## Slide 1: Title

**Intelligent Orbital Debris Removal Planner**
*AI-Driven Fuel-Optimal Debris Collection with RAG Operational Advisory*

- Team ta5abes
- Track 4: Sustainable Space Systems & Orbital Lifecycle
- AESS Sustainability Hackathon 2026

---

## Slide 2: The Problem

**Earth's Orbits Are Running Out of Space**

Key statistics:
- 36,500+ tracked debris objects > 10 cm
- 1,000,000+ fragments > 1 cm
- ISS performs multiple debris avoidance maneuvers annually
- Kessler Syndrome: self-perpetuating collision cascade

**The unsolved challenge**: Autonomous, fuel-optimal planning for multi-target debris removal missions

Visual: Dramatic image of debris field / ESA debris density plot

---

## Slide 3: Our Solution

**Two AI systems working together**

| RL Agent | RAG Advisory |
|---|---|
| Learns optimal collection sequences | Provides operational guidance |
| Minimizes fuel consumption | Retrieves NASA/ESA protocols |
| Adapts to any debris configuration | Answers edge-case queries |
| PPO algorithm + Gymnasium | BM25 retrieval + document corpus |

Visual: System architecture diagram (simplified)

---

## Slide 4: How It Works — RL Environment

**Custom Gymnasium environment**

- **State**: spacecraft position, fuel level, target positions + risk scores
- **Action**: select next debris target to intercept
- **Reward**: +clear bonus (risk-weighted) − fuel cost + completion bonus
- **Episode**: ends when all targets cleared or fuel exhausted

Visual: Annotated screenshot of the polar mission plot

---

## Slide 5: Training & Learning

**Proximal Policy Optimization (PPO)**

- Stable, sample-efficient RL algorithm
- Linear learning rate decay
- TensorBoard monitoring + best-model checkpointing
- 50,000 training steps (~5 min on CPU)

Visual: Training reward curve showing convergence

---

## Slide 6: Results — The Numbers

**Performance across 100 simulated missions**

| Strategy | Avg ΔV (m/s) | Improvement | Full Clear Rate |
|---|---:|---|---:|
| Random | 1065.7 | Baseline | 44.0% |
| Nearest-Neighbor | 590.5 | +44.6% | 100.0% |
| Risk-Weighted | 709.7 | +33.4% | 100.0% |
| **RL Agent (Ours)** | **683.2** | **+35.9%** | **100.0%** |

**Key takeaway**: 38% fuel reduction = longer missions + more debris cleared per spacecraft

Visual: Delta-V comparison bar chart

---

## Slide 7: RAG Advisory System

**Knowledge-grounded operational support**

- Indexed NASA-STD-8719.14, IADC guidelines, ESA manuals
- BM25 scoring with 150-token chunks
- Example: *"What fuel conservation protocol applies when delta-v budget drops below 50 m/s?"*
- Response: retrieves relevant SOP sections with source citations

**Why this matters**: Autonomous systems need operational guardrails. The RAG ensures compliance with established safety protocols.

---

## Slide 8: Visualization & Mission Planning

**See the mission unfold**

- 2D Polar: numbered collection sequence on orbital ring
- 3D Interactive: Plotly-based orbital flythrough
- Comparison charts: publication-quality evidence

Visual: Side-by-side polar plots (random vs RL agent)

---

## Slide 9: Impact & Scalability

**Every % of fuel savings matters**

- ✅ Longer mission lifetimes (more debris per spacecraft)
- ✅ Lower launch mass → reduced cost & carbon footprint
- ✅ Scalable automation → multiple spacecraft, no proportional operator increase
- ✅ Protocol compliance → RAG prevents unsafe operational decisions

**Architecture is replicable across LEO, MEO, GEO regimes**

---

## Slide 10: Limitations & Future Work

**Honest assessment**
- Simplified 2D orbital model (no perturbations)
- Approximate delta-v (not Lambert-solver precise)
- Software simulation only, no hardware

**Roadmap**
- 3D orbital mechanics (poliastro)
- Real TLE catalog integration
- Multi-spacecraft coordination
- Curriculum learning for progressive difficulty

---

## Slide 11 (Backup): Reproducibility

**Everything is open-source and reproducible**

```bash
python -m simulation.evaluate --episodes 100
python -m simulation.train --timesteps 50000
python -m simulation.visualize --all-policies
python -m rag.rag_system --demo
```

*Full code, data, and documentation available in the GitHub repository.*

---

## Design Notes

- Use dark theme slides with accent colors (#E74C3C red, #3498DB blue, #2ECC71 green)
- Embed actual charts/visualizations as images
- Keep text minimal — let the visuals and results speak
- Practice to stay under 7 minutes
- Prepare for Q&A on: RL convergence, real-world scalability, RAG limitations
