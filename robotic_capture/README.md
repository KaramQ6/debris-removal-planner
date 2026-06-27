# 🤖 Robotic Capture & Detumbling of Tumbling Debris (Paper 2)

**Status:** early scaffolding · **Target venue:** top-tier robotics (ICRA / IROS / RA-L)

A *separate, self-contained* research project for the second paper. It studies **autonomous
capture and detumbling of a tumbling debris target using a free-floating space manipulator**,
trained with reinforcement learning under unknown target inertia.

> ⚠️ **Relationship to the old paper.** This folder is intentionally isolated from the published
> orbital-sequencing work that lives in the repository root (`simulation/`, `rag/`, `docs/`).
> That work is **not touched**. The only conceptual link is a thin one: the old high-level planner
> *selects which target*; this project studies *how the manipulator physically captures it*.
> Because the two share almost no method or text, paper 2 is unambiguously new work
> (no self-plagiarism risk).

## Why this is a robotics paper (and the old one was not)
The old environment *teleports* the chaser onto the target's orbit and only subtracts a Δv scalar —
there is **no control**. This project adds the missing layer: closed-loop **contact-rich control**
of a manipulator on a **free-floating base** (where arm motion reacts back on the base — the
defining challenge of space robotics).

## Planned contribution (to be locked in `docs/literature_review.md`)
Learned capture + detumbling of a tumbling target with **unknown inertia**, on a free-floating
base, via RL + domain randomization, benchmarked against **strong model-based baselines**
(impedance / MPC), reporting contact forces and base disturbance.

## Structure
- `docs/literature_review.md` — Phase 0: prior-art gap analysis + locked novelty.
- `docs/contribution.md` — one-paragraph claim + paper outline.
- `docs/paper/` — LaTeX manuscript (later).
- `sim/` — MuJoCo free-floating manipulator + tumbling target Gymnasium environment.
- `sim/assets/` — MJCF (`.xml`) physics models.
- `tests/` — self-checks (`pytest`).
- `notes/` — working notes.

## Setup
```powershell
# from repo root, reusing the existing .venv
pip install -r robotic_capture/requirements.txt
pytest robotic_capture/tests/
```

## Roadmap
0. ✅ Lock novelty + literature gap.
1. ✅ Build the simulator (free-floating base + 3-DOF arm + tumbling target + contact).
2. ✅ Formulate the control MDP (state / action / reward).
3. ✅ Strong model-based baseline — **ARC** (force-tracking, inertia-agnostic) in `control/baselines.py`.
4. ◐ Learned policy + domain randomization (`train.py`, SAC; `--no-domain-randomize` for the ablation).
5. ◐ Evaluation + sweeps (`eval.py`): **headline = base-attitude disturbance vs target inertia**
   (chaser ACS fuel), with spin reduction as a parity check; bootstrap CIs → `results/`.

Run: `python -m robotic_capture.train --timesteps 120000 --out runs/sac_dr`
then `python -m robotic_capture.eval --policy runs/sac_dr.zip`.

> **Framing note (post-ARC pivot):** ARC is inertia-agnostic *by design*, so the claim is **not**
> "model-based fails under unknown inertia." The contribution is lower induced base disturbance
> (ACS fuel) and kinematics-only sensing (no force sensor). See `docs/contribution.md`.
