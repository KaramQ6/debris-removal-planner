# Session report — 2026-06-27 (Paper 2, part 2: baseline + training + eval)

Continues `session_2026-06-27.md`. Worked **only** inside `robotic_capture/`.

## Headline: the contribution pivoted (and is now stronger)

Reading the **ARC baseline** paper's abstract (the `.ris` we have for Wang, Liu & Cai,
*Nonlinear Dynamics* 113(18), 2025) overturned the locked narrative. Verbatim:

> "the control law of ARC relies only on measured contact force and **does not require prior
> knowledge of the target's inertial or contact parameters**."

So the canonical model-based baseline is **inertia-agnostic by design** — the old headline
("RL is robust to unknown inertia where model-based fails") was a strawman. Two independent
signals confirmed this: (a) the abstract, and (b) our own inertia sweep, where the baseline
did **not** collapse with inertia. **Decision (user, lead author): pivot the headline to
induced base-attitude disturbance (chaser ACS fuel)**, keep the inertia sweep as a *parity
check*, and hold a secondary *sensing-reduction* angle (policy uses kinematics only; ARC
needs a force sensor).

## What was built
- **Env fixes** (`sim/free_flyer_env.py`, `sim/assets/space_manipulator.xml`):
  - **Critical bug:** runtime `body_mass`/`body_inertia` edits never reached the dynamics —
    MuJoCo caches `cinert`. Added `mj_setConst` in `reset()`. **Domain randomization was a
    silent no-op before this.**
  - Removed a non-physical 250 kN contact spike: the arm spawned *inside* the target. Added a
    bent "ready" pose (`_READY_POSE`) leaving ~0.10 m clearance (post-grasp scope, good
    Jacobian leverage). Contact forces are now ~100–150 N.
  - Planar scaffold ⇒ default initial tumble is now a **z-axis spin** (the axis a 3-DOF planar
    arm can actually control); `inertia_scale` option added for deterministic sweeps.
- **Strong baseline** (`control/baselines.py`): `ARCController` — force-tracking,
  **inertia-agnostic** (Jacobian-transpose wrench = force-tracking normal + tangential
  resistance to the measured surface velocity). Plus `ZeroController` (reference) and
  `PolicyController` (RL adapter). Re-implemented from the *abstract* — needs reconciliation
  with the full PDF (see below).
- **Training** (`train.py`): SAC, `--domain-randomize` / `--no-domain-randomize` (ablation).
  Policy observes **kinematics only** (32-D; no contact force / no inertia).
- **Eval** (`eval.py`): paired sweep over (inertia scale, spin, seed); headline = base
  disturbance vs inertia; spin reduction = parity; bootstrap 95% CIs → `results/`.
- **Self-checks** (`tests/test_baselines.py`): ARC is inertia-agnostic by construction, ARC
  detumbles ≫ zero, base disturbance grows with inertia, bootstrap CI brackets the mean.

## Verified
- `pytest robotic_capture/tests/` → **8 passed** (4 env + 4 baseline/eval).
- DR SAC trained 120k steps (seed 0): `ep_rew_mean` −307 → **+91**.
- **Final eval** (RL vs ARC vs zero; scales 0.5–2.0, spins 0.2–0.4, seeds 0–4 = 60 ep each),
  base disturbance (rad/s, ↓ better) and spin reduction (%, parity), mean [95% CI]:

  | inertia | ARC base disturb | RL base disturb | ARC spin↓ | RL spin↓ | ARC succ | RL succ |
  |---|---|---|---|---|---|---|
  | 0.5× | 0.034 [.019,.049] | 0.045 [.043,.046] | 82% | **96%** | 100% | 100% |
  | 1.0× | 0.121 [.091,.153] | **0.033** | 83% | 84% | 100% | 100% |
  | 1.5× | 0.226 [.171,.284] | **0.013** | 82% | 83% | 67% | **100%** |
  | 2.0× | 0.285 [.228,.338] | **0.014** | 74% | **83%** | 67% | **100%** |

  (ARC here is the **faithful** Eq. 5 re-implementation; vs the abstract-only version it is a
  bit stronger at heavy inertia — a fairer, stronger baseline.)

  **Takeaways:** RL induces up to **~20× less base disturbance** at heavy inertia (0.014 vs
  0.285), detumbles **as well or better**, hits **100% success across all scales** (ARC drops
  to 67% past nominal), and uses **no force sensor**. Raw data: `results/eval_episodes.csv`.

## Ablation: domain randomization (DR vs no-DR), both 120k steps

no-DR is trained only at nominal inertia (scale 1.0). Spin reduction (%) / success / base
disturbance (rad/s), 15 episodes per cell:

| inertia | DR succ | no-DR succ | DR spin↓ | no-DR spin↓ | DR base | no-DR base |
|---|---|---|---|---|---|---|
| 0.5× | **100%** | **67%** | 96% | 82% | 0.045 | 0.059 |
| 1.0× | 100% | 100% | 84% | 91% | 0.033 | 0.011 |
| 1.5× | 100% | 100% | 83% | 85% | 0.013 | 0.007 |
| 2.0× | 100% | 100% | 83% | 86% | 0.014 | 0.011 |

**Honest reading (do not overclaim):** DR does **not** rescue a wholesale collapse — no-DR
generalises *well to heavier* targets (sluggish, forgiving) and even beats DR at nominal
(specialisation). DR's value shows up at the **light-inertia extreme (0.5×)**, where no-DR
success drops 100%→67% while DR holds 100%, at the cost of slightly higher base disturbance at
nominal (the classic DR robustness/peak-performance trade-off). For a stronger ablation, push
OOD harder (scale >2×, off-diagonal inertia, randomized initial pose).

## Honest caveats (state plainly in the paper)
- Planar 3-DOF scaffold; single training seed. ARC now reconciled with the full PDF
  (Eq. 5 force law + Sec. 3.3 hybrid); gains tuned to our scaffold, not copied.
- ARC "0% success" at ≥1.5× is w.r.t. `detumble_tol=0.05` (final spin ~0.07–0.13), not a crash.
- The base actuators are absent (chaser is free-floating, no wheels): "detumbling" = bleeding
  target spin into the chaser; base disturbance is precisely the cost that quantifies it.

## Next session (priority order)
1. ✅ **no-DR ablation** — done (see Ablation section). Result is nuanced, not a clean collapse;
   strengthen with a harder OOD test (scale >2×, off-diagonal inertia, randomized initial pose).
2. Train **≥3 seeds** per condition; report seed variance (carry over Paper 1's rigor).
3. ✅ Reconcile `ARCController` with the full ARC PDF — **done** (Eq. 5 + Sec. 3.3 hybrid).
4. Add a figure: base disturbance vs target inertia (RL flat, ARC rising).
5. Update `docs/paper/main.tex` to the pivoted title/abstract (user owns
   `contribution.md` + `literature_review.md`).

## ⬇️ Papers still to download (paywalled; user sourcing via ResearchGate/authors)
1. **Wei, Bai, Lu** — Trajectory planning of free-floating space robot for non-cooperative
   tumbling target capture (DRL), *Robotica* 2025.
2. **Two-axis matching RL for tracking tumbling targets**, *Acta Astronautica* 2024 (S1270963824006709).

On disk (sufficient to proceed): ⭐ **ARC** `s11071-025-11381-z.pdf` (baseline, now reconciled),
review `space.0291`, Hovell & Ulrich JGCD (`G006656`), Lam & Chhabra `2510.06566`,
+ bonus Peng & Wang *Aerospace* 2024 (`11/9/706`).
