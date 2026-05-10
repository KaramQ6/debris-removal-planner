# System Architecture — Intelligent Orbital Debris Removal Planner

## High-Level Architecture

```mermaid
graph TB
    subgraph Input["📡 Data Input"]
        TLE["TLE Debris Catalog<br/>(Space-Track.org)"]
        RISK["Collision Risk<br/>Assessments"]
        FUEL["Mission Fuel Budget<br/>(delta-v)"]
    end

    subgraph SimCore["🔧 Simulation Core"]
        ENV["Gymnasium Environment<br/>(orbit_env.py)"]
        SCEN["Mission Scenarios<br/>(scenario.py)"]
        SCEN --> ENV
    end

    subgraph RLAgent["🤖 RL Planning Agent"]
        PPO["PPO Algorithm<br/>(Stable-Baselines3)"]
        TRAIN["Training Pipeline<br/>(train.py)"]
        EVAL["Evaluation Engine<br/>(evaluate.py)"]
        CB["Metrics Callbacks<br/>(callbacks.py)"]
        TRAIN --> PPO
        CB --> TRAIN
        PPO --> EVAL
    end

    subgraph Baselines["📏 Baseline Policies"]
        RND["Random Policy"]
        NN["Nearest-Neighbor"]
        RW["Risk-Weighted NN"]
    end

    subgraph RAGSys["📚 RAG Advisory System"]
        DOCS["Document Corpus<br/>(NASA/ESA/IADC)"]
        CHUNK["BM25 Indexer<br/>+ Chunking"]
        QUERY["Query Interface"]
        DOCS --> CHUNK
        CHUNK --> QUERY
    end

    subgraph Output["📊 Output & Visualization"]
        PATH["Optimized Mission Path"]
        POLAR["2D Polar Plot"]
        ORBIT3D["3D Orbital View"]
        CHARTS["Comparison Charts"]
        REPORT["Evaluation Report"]
    end

    Input --> SimCore
    SimCore --> RLAgent
    SimCore --> Baselines
    RLAgent --> Output
    Baselines --> Output
    RAGSys -.->|"On-demand<br/>advisory"| Output

    style Input fill:#E8F5E9,stroke:#4CAF50
    style SimCore fill:#E3F2FD,stroke:#2196F3
    style RLAgent fill:#FFF3E0,stroke:#FF9800
    style Baselines fill:#F3E5F5,stroke:#9C27B0
    style RAGSys fill:#FCE4EC,stroke:#E91E63
    style Output fill:#ECEFF1,stroke:#607D8B
```

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant D as Debris Catalog
    participant E as Gym Environment
    participant A as PPO Agent
    participant R as RAG Advisory
    participant V as Visualization

    D->>E: Load target positions, risks
    E->>A: Observation (pos, fuel, targets)
    loop Each Step
        A->>E: Action (target index)
        E->>A: Reward, next observation
        Note over E,A: Reward = -ΔV + risk_bonus
    end
    A->>V: Trajectory data
    V->>V: Generate plots
    R-->>A: On-demand guidance
```

## Component Details

| Component | File | Responsibility |
|---|---|---|
| **Environment** | `simulation/orbit_env.py` | State management, physics, reward computation |
| **Scenarios** | `simulation/scenario.py` | Target generation, mission constraints |
| **PPO Training** | `simulation/train.py` | Model training with callbacks and checkpoints |
| **Evaluation** | `simulation/evaluate.py` | Multi-policy comparison with statistics |
| **Baselines** | `simulation/policies.py` | Random, nearest-neighbor, risk-weighted policies |
| **Visualization** | `simulation/visualize.py` | 2D polar, 3D orbit, interactive Plotly HTML |
| **Charts** | `simulation/plot_results.py` | Delta-V comparison, distributions, reward curves |
| **RAG System** | `rag/rag_system.py` | BM25 document retrieval with structured responses |
| **Knowledge Base** | `docs/*.md, *.txt` | NASA/ESA/IADC reference documents |
