# Intelligent Orbital Debris Removal Planner

Starter implementation for **Phase 1** of team **ta5abes**:
- A Gymnasium orbital debris environment
- PPO training entrypoint
- Baseline + RL-ready evaluation script
- Lightweight RAG-style advisory prototype over local docs

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run baseline comparison:

```powershell
python -m simulation.evaluate --episodes 100
```

Train PPO policy:

```powershell
python -m simulation.train --timesteps 50000
```

Evaluate trained model:

```powershell
python -m simulation.evaluate --episodes 100 --model-path results\models\ppo_debris.zip
```

Query advisory docs:

```powershell
python -m rag.rag_system --docs docs --query "What should we do when delta-v budget is low?"
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `simulation\orbit_env.py` | Custom debris-removal mission environment |
| `simulation\train.py` | PPO training script (Stable-Baselines3) |
| `simulation\evaluate.py` | Baseline and optional RL policy evaluation |
| `rag\rag_system.py` | Local-document retrieval advisory prototype |
| `docs\` | Assumptions and reference excerpts used by RAG |
| `results\` | Saved models and evaluation summaries |

## Current model assumptions

1. Simplified 2D circular-orbit representation using angular transfers.
2. Delta-v is approximated by angular separation, not full Lambert optimization.
3. One servicing spacecraft and one target captured per step.
4. Mission ends when all targets are cleared, fuel is exhausted, or max steps reached.
