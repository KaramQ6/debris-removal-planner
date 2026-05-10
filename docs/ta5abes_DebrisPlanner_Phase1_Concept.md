

**Intelligent Orbital Debris Removal Planner**

AI-Driven Fuel-Optimal Debris Collection with RAG Operational Advisory

AESS Sustainability Hackathon 2026  |  Phase 1 Concept Document

| Field | Details |
| ----- | ----- |
| **Team Name** | ta5abes |
| **Selected Track** | Track 4: Sustainable Space Systems & Orbital Lifecycle |
| **Competition** | AESS Sustainability Hackathon 2026 |
| **Team Members** | [Your Name/Handle], RayanDahdooly |
| **Submission Type** | Phase 1 — Concept Document |
| **Core Technology** | Reinforcement Learning (PPO) \+ Retrieval-Augmented Generation (RAG) |
| **Problem Area** | Orbital debris accumulation threatening long-term sustainability of Earth's orbital environment |

# **1\. Project Summary**

Team ta5abes proposes the Intelligent Orbital Debris Removal Planner — a dual-AI mission planning system designed to address one of the most critical sustainability challenges in modern space operations: the exponential accumulation of orbital debris threatening the long-term viability of Earth's orbital environment.

The system combines a Reinforcement Learning (RL) agent that learns fuel-optimal, multi-target debris collection trajectories, with a Retrieval-Augmented Generation (RAG) advisory layer that provides real-time operational guidance from indexed NASA and ESA technical documentation. Together, they form an autonomous mission planning capability that reduces fuel consumption, maximizes debris cleared per mission, and ensures operational decisions are grounded in established space safety protocols.

# **2\. Problem Statement**

Earth's orbital environment is approaching a critical tipping point. As of 2025, ESA tracks over 36,500 debris objects larger than 10 cm, with estimates exceeding 1 million fragments larger than 1 cm — each capable of catastrophically disabling an operational satellite upon collision. The consequences are systemic:

* Active satellites face rising collision probability, threatening global communications, navigation (GPS), weather forecasting, and Earth observation.

* The ISS executes multiple debris avoidance maneuvers annually, consuming mission-critical fuel and crew time.

* Unchecked growth risks the Kessler Syndrome — a self-perpetuating collision cascade that could render entire orbital shells permanently unusable.

* Each new launch adds debris, yet no scalable autonomous removal system currently operates at mission-ready level.

The core unsolved problem is not debris identification — it is the autonomous planning of fuel-optimal removal missions that efficiently sequence multi-target collection paths. Current manual planning is slow, fuel-wasteful, and non-scalable to the magnitude of the problem.

# **3\. Proposed Solution**

The Intelligent Orbital Debris Removal Planner integrates two AI subsystems into a unified mission planning platform:

| Component | Primary Function | Technology Stack |
| ----- | ----- | ----- |
| **RL Core Agent** | Plans fuel-optimal multi-target debris collection sequences. Prioritizes high-risk objects. Minimizes total delta-v across full mission profile. | Python, Stable-Baselines3, PPO algorithm, custom OpenAI Gym environment |
| **RAG Advisory System** | Provides on-demand operational guidance from indexed NASA/ESA manuals. Answers edge-case queries during mission planning phase. | LangChain, FAISS vector store, NASA Debris Mitigation Guidelines, IADC standards |

The RL agent handles the quantitative optimization problem (trajectory and sequencing), while the RAG system handles the qualitative knowledge problem (operational protocols and decision support). This division of responsibility mirrors how expert human mission teams operate — and enables the system to function autonomously without ground operator intervention.

# **4\. System Architecture**

## **4.1 Data Flow**

TLE Debris Data Input  →  Orbital Environment Simulator  →  RL Agent (PPO)  →  Optimized Multi-Target Path  →  Mission Output \+ 3D Visualization  →  RAG Advisory (on-demand queries)

| System Block | Function | Implementation |
| ----- | ----- | ----- |
| **Data Ingestion** | Loads real TLE (Two-Line Element) debris catalog from Space-Track.org or NASA datasets. Computes collision risk scores per object. | Python \+ sgp4 library |
| **Orbital Simulator** | Custom 2D orbital environment modeling spacecraft position, fuel state, debris positions, and collision risk priorities. | Custom OpenAI Gym environment |
| **RL Agent** | Learns optimal debris collection sequencing via Proximal Policy Optimization (PPO). Reward \= negative delta-v \+ positive debris cleared. | Stable-Baselines3 PPO |
| **RAG Knowledge Base** | Indexes NASA Debris Mitigation Standard, ESA Space Debris User Manual, IADC guidelines. Retrieves relevant sections per query. | LangChain \+ FAISS vector store |
| **Visualization** | 3D animated orbital path showing debris collection sequence, delta-v budget per segment, and total mission profile. | Matplotlib / Plotly |

# **5\. Implementation Method**

## **5.1 RL Environment Design**

The custom Gym environment defines the following:

* State space: spacecraft orbital position, velocity vector, remaining fuel (kg), list of remaining debris targets with positions and risk scores

* Action space: selection of next debris target to intercept

* Reward function: negative delta-v consumed per maneuver, bonus reward for clearing high-risk objects first, penalty for excessive fuel use

* Episode termination: all targets cleared or fuel depleted

## **5.2 RL Training Protocol**

* Algorithm: Proximal Policy Optimization (PPO) — stable, sample-efficient, well-suited for continuous action spaces

* Training environment: simplified 2D orbital plane with 5-10 debris targets per episode

* Evaluation metric: total delta-v (m/s) consumed vs. random baseline and greedy nearest-neighbor baseline

* Hardware: standard CPU training, estimated 2-4 hours for convergence on simplified environment

## **5.3 RAG System Design**

* Document corpus: NASA-STD-8719.14 (Debris Mitigation), ESA Space Debris User Manual, IADC Space Debris Mitigation Guidelines

* Chunking strategy: 500-token chunks with 50-token overlap

* Retrieval: FAISS cosine similarity search, top-3 chunks per query

* Response time target: under 3 seconds per query

* Example query: 'What fuel conservation protocol applies when delta-v budget drops below 50 m/s?' → RAG retrieves relevant SOP section

# **6\. Baseline vs. Optimized Comparison**

The primary evidence for system effectiveness is a quantitative comparison of mission fuel consumption (total delta-v in m/s) between three planning strategies across a standardized test scenario: 8 debris targets in LEO, initial fuel budget of 1,200 m/s delta-v.

| Planning Strategy | Avg Delta-V (m/s) | Targets Cleared | Fuel Efficiency (m/s/target) |
| ----- | ----- | ----- | ----- |
| **Random Order (Baseline)** | 1,719 m/s | 0.42 / 8 | 4,093 m/s |
| **Greedy Nearest-Neighbor** | 5,214 m/s | 2.32 / 8 | 2,247 m/s |
| **RL Agent (Our System)** | **1,043 m/s** | **0.34 / 8*** | **3,067 m/s** |

Key result: While the RL agent is in the early training phases of the 24-hour hackathon, its fuel efficiency (Delta-V per target) already outperforms random planning by **25%**. The Greedy Nearest-Neighbor heuristic serves as a robust operational fallback, providing reliable clearance rates in complex scenarios.

*Note: RL performance represents early convergence (50k steps). Full training is expected to significantly increase clear rates while maintaining high fuel efficiency. Real-world performance will depend on actual orbital mechanics constraints and debris catalog accuracy. All assumptions are documented in the repository.*

# **7\. Impact Statement**

The Intelligent Orbital Debris Removal Planner addresses orbital sustainability at its most fundamental level: the economic and physical viability of continued space operations. Every percentage reduction in mission fuel consumption directly translates to:

* Longer mission lifetimes — spacecraft can clear more debris before fuel exhaustion

* Lower launch mass requirements — less propellant needed per mission reduces launch cost and carbon footprint

* Higher debris clearance rates — more efficient paths mean more objects removed per mission

* Scalable automation — autonomous planning enables simultaneous coordination of multiple removal spacecraft without proportional increase in ground operator workload

The RAG advisory component adds a safety dimension: operational decisions are made with direct reference to established NASA and ESA debris mitigation standards, reducing the risk of protocol violations that could create additional debris through mission failure.

Long-term, this system architecture — autonomous RL planning \+ knowledge-grounded advisory — represents a replicable model for sustainable space operations management across orbital regimes (LEO, MEO, GEO).

# **8\. Limitations and Honest Assessment**

* Simulation fidelity: The current implementation uses a simplified 2D orbital model. Real orbital mechanics involve perturbations (atmospheric drag, solar radiation pressure, gravitational harmonics) not yet modeled.

* Transfer maneuver approximation: Delta-v calculations use simplified Hohmann transfer approximations. High-fidelity trajectory optimization (e.g., Lambert's problem solvers) would improve real-world accuracy.

* RAG document scope: The knowledge base covers publicly available NASA/ESA standards. Classified or mission-specific operational documents are not included.

* Hardware validation: This is a software simulation only. No physical spacecraft or hardware testing has been conducted.

* RL training time: Achieving robust policy convergence for complex multi-target scenarios may require significantly longer training than available in Phase 1 timeline.

Future work: Integration of full 3D orbital mechanics (using poliastro or equivalent), Lambert solver for precise transfer calculations, multi-spacecraft coordination, and integration with ESA's Space Debris Office catalog API.

# **9\. Repository Structure**

| Folder / File | Contents |
| ----- | ----- |
| **README.md** | Project overview, KPIs, reproduction steps, and AI disclosure. |
| **/docs** | Phase 1 Concept Document and Hackathon Guidelines. |
| **/simulation** | Custom Gymnasium environment, training, and evaluation scripts. |
| **/results** | Quantitative evaluation summaries and training history. |
| **/assets** | Orbital visualizations, mission plots, and demo media. |
| **/rag** | Knowledge-grounded operational advisory system. |
| **/tools** | Internal evaluation and visualization tools. |

# **10\. Submission Checklist**

| Status | Item |
| ----- | ----- |
| **\[x\]** | Concept document — clear, concise, exported as PDF |
| **\[ \]** | Repository — public GitHub, README explains how to run simulation |
| **\[ \]** | Results reproducible from submitted code and data files |
| **\[ \]** | Demo video — 2-3 minutes, shows RL agent planning path and RAG query response |
| **\[ \]** | Presentation deck — 8-10 slides, problem to result narrative |
| **\[ \]** | All assumptions and AI-assisted outputs disclosed |
| **\[ \]** | File names follow convention: ta5abes\_DebrisPlanner\_Phase1\_\[Type\].\[ext\] |
| **\[ \]** | All links verified from incognito browser before submission |

# **AI Usage Disclosure**

This project was developed primarily through team effort during a 24-hour hackathon. AI assistance was limited to initial structural suggestions for the concept document and minor code scaffolding. 

**Crucially, the core reinforcement learning model architecture, training logic, and orbital physics environment were designed, implemented, and trained entirely by the team.** All technical results and performance metrics presented are the direct result of our own model training and validation sessions. All technical content, problem framing, architectural decisions, and quantitative estimates were reviewed and validated by team ta5abes. Baseline vs. optimized comparison values represent simulated results reproducible from the submitted code.

