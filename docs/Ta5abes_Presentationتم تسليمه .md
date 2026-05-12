 # TA5ABESTA5ABES

```
Hackathon
```
##### Sustainable Space Systems & Orbital Lifecycle (Challenge 4)

###### INTELLIGENT ORBITAL DEBRIS REMOVAL PLANNERINTELLIGENT ORBITAL DEBRIS REMOVAL PLANNER


### ProbeLem Statement

```
Hackathon
```
Over 36,500 tracked objects and 1M+ untraceable

fragments threatening space infrastructure

Approaching the "Kessler Syndrome" threshold ,one

collision could trigger a chain reaction

Every year, collision risk increases ~5% in LEO


**Fuel consumption (Delta-V)**

**Collision risk**

**Orbital mechanics**

**Mission duration**

**ΔV = √(μ/r₁) · (√(2r₂/(r₁+r₂)) − 1)**

**Even small orbital transfers are computationally expensive.**

### Why This Problem Is Hard Mission

### planning must balance:

```
Hackathon
```

### Proposed Solution

```
Hackathon
```
```
TLE Orbital Data
↓
3D Orbital Simulator (Keplerian + J2)
↓
RL Agent (PPO — 1.7M Training Steps)
↓
Fuel-Optimal Mission Planner
↓
RAG Safety Advisor (NASA/ESA)
↓
Safe Debris Collection Plan
```
**Our system autonomously learns efficient debris collection**

**strategies while following aerospace safety guidelines.**

###### Core idea.


### Technical Implementation

```
Hackathon
```
#### Technologies: Python, PyTorch, Gymnasium

#### Algorithms: PPO (RL), FAISS (RAG)

#### Physics: Two-Body Keplerian + J2 Perturbation

#### Protocol: API-based real-time TLE data fetching


### System Architecture & Technical Design

```
Hackathon
```

## Reinforcement Learning Core

```
Hackathon
```
State Space: Satellite position · Fuel remaining · Debris coordinates

Actions: Select next debris target · Perform orbital transfer

Reward Function:R = **α** ·D − **β** ·ΔV − **γ** ·C − **δ** ·T

```
Maximize debris collected
Minimize fuel usage (Delta-V)
Avoid risky trajectories
Minimize time
```
Trained for 1.7 Million steps on high-precision 3D Keplerian physics


### RAG Advisory System

```
Hackathon
```
###### Knowledge Sources:

**NASA-STD-8719.**

**ESA Debris Mitigation Guidelines**

**IADC Recommendations**

###### Purpose: Reduce hallucinations · Provide operational

###### safety guidance · Assist mission operators


**Key Numbers:
86.3% Delta-V reduction vs. random baseline
45% higher fuel efficiency per target vs. random
1,043 m/s average operational Delta-V
Stable convergence after 1.7M training steps
*Agent is conservative, prioritizes low-cost, safe intercepts**

### Results

```
Hackathon
```

SDG 9 — Industry, Innovation & Infrastructure Our

AI-powered planner builds smart space

infrastructure for sustainable orbital management

SDG 13 — Climate Action Protecting climate-

monitoring satellites by reducing collision risk in

LEO

### SDG Alignment

```
Hackathon
```

### Impact & Scalability

```
Hackathon
```
###### Impact:

###### Cleaner orbital environments

###### Reduced collision probability

###### Longer satellite mission lifetimes

###### Millions saved in fuel per mission

###### Future Work:

###### Multi-agent debris removal fleet

###### High-fidelity physics (atmospheric drag + solar radiation pressure)

###### Real-time ESA/NASA live TLE integration


### Challenges

```
Hackathon
```
###### Sim-to-real gap in orbital mechanics

###### Fragments < 1cm untrackable (industry-wide

###### hardware limitation)

###### Simplified physics assumptions in current model

###### Radar uncertainty & maneuver noise modeling


### Conclusion

```
Hackathon
```
##### AI combining RL optimization + aerospace safety enables

##### autonomous, fuel-efficient debris removal

##### 86.3% fuel reduction proves the approach is viable and

##### scalable

##### "Protecting our orbits today ensures our access to the stars

##### tomorrow."


