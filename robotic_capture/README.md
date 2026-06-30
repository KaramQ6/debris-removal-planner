# 🤖 Robotic Capture & Detumbling of Tumbling Debris (Paper 2)

**Status:** manuscript in progress — env, baseline, 3-seed results & figures done; drafting `docs/paper/main.tex` · **Target venue:** top-tier robotics (ICRA / IROS / RA-L)

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

## Latest results (3 seeds, 95% bootstrap CIs)
Regenerate with `python -m robotic_capture.make_results` → `results/eval_tables_full.md`.

**Headline — base-attitude disturbance rejection** (peak induced base rate, rad/s; lower is better,
target inertia held at 1×). The base-penalised policy stays flat while ARC tracks contact force only:

| injected (rad/s) | RL (ours) | RL ablation (w_b=0) | ARC | zero-control |
|---|---|---|---|---|
| 0.05 | **0.023** | 0.035 | 0.123 | 0.052 |
| 0.12 | **0.025** | 0.036 | 0.128 | 0.122 |

→ **≈80% lower peak base disturbance than ARC (~5×)** across the sweep; the ablation degrades but does
**not** collapse to ARC's level, so the base-attitude reward is a contributing (not sole) mechanism.

**Inertia parity check** (no injected disturbance) — confirms ARC is inertia-agnostic *by design*, so
this is parity, **not** a "model-based fails under unknown inertia" claim. The RL–ARC base-disturbance
**gap grows with target mass** (ratio ARC/RL): **2.8× @ 0.5× → 6.4× @ 2×**; RL detumble success stays
100% while ARC drops to 67% at ≥1.5× (attributed to horizon/gain tuning, not inertia ignorance).

## Structure
- `docs/literature_review.md` — Phase 0: prior-art gap analysis + locked novelty.
- `docs/contribution.md` — one-paragraph claim + paper outline (post-ARC pivot).
- `docs/paper/` — LaTeX manuscript (`main.tex`) + `figures/` (Overleaf-ready paths).
- `sim/free_flyer_env.py` — MuJoCo free-floating manipulator + tumbling target Gym env; `sim/assets/` MJCF models.
- `control/baselines.py` — `ARCController` (baseline), `ZeroController` (reference), `PolicyController` (RL adapter).
- `train.py` — SAC training (`--base-disturbance`, `--w-base`, `--domain-randomize`) → `runs/`.
- `eval.py` — paired sweeps (disturbance / inertia / spin / seeds) with bootstrap CIs → `results/`.
- `make_results.py` — regenerates all three Results tables (pooled over 3 seeds) → `results/eval_tables_full.md`.
- `plot_base_disturbance.py` · `plot_disturbance_vs_inertia.py` · `plot_learning_curve.py` · `render_scene.py`
  · `plot_system_diagram.py` — paper figures → `docs/paper/figures/`.
- `tests/` — self-checks (`pytest`). `notes/` — working notes. `runs/` (git-ignored) models/logs. `results/` tables/CSVs.

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
4. ✅ Learned policy + domain randomization — 3 seeds main (`sac_dist_seed{0,1,2}`, w_base=0.3) +
   3 seeds ablation (`sac_dist_ablate_seed{0,1,2}`, w_base=0), all `base_disturbance=0.8`, 120k steps.
5. ✅ Evaluation + sweeps (`eval.py`): headline base-attitude disturbance rejection, inertia parity,
   and gap-vs-mass; bootstrap CIs + figures → `results/` and `docs/paper/figures/`.
6. ◐ Writing the manuscript (`docs/paper/main.tex`) — results & figures in; drafting + Overleaf export.

Run (train one seed, evaluate, regenerate tables):
```powershell
python -m robotic_capture.train --timesteps 120000 --w-base 0.3 --seed 0 --out robotic_capture/runs/sac_dist_seed0
python -m robotic_capture.eval --policy robotic_capture/runs/sac_dist_seed0.zip
python -m robotic_capture.make_results   # pools the 3 trained seeds into the Results tables
```

> **Framing note (post-ARC pivot):** ARC is inertia-agnostic *by design*, so the claim is **not**
> "model-based fails under unknown inertia." The contribution is lower induced base disturbance
> (ACS fuel) and kinematics-only sensing (no force sensor). See `docs/contribution.md`.
