"""Regenerate every Results table for Paper 2 (pooled over 3 seeds) into one markdown file.

Reuses the committed ``eval.evaluate`` / ``eval.bootstrap_ci`` so the paper's tables are
reproducible from the repo (not a one-off scratch script). Runs three paired sweeps:

  1. **Disturbance rejection** (headline + ablation): inertia fixed at 1.0x, disturbance swept.
  2. **Inertia parity** (no injected disturbance): inertia swept, disturbance 0.
  3. **Gap vs target mass**: inertia swept at a fixed mid disturbance (0.5 N.m).

Pools the 3 main seeds into ``rl`` and the 3 ablation seeds into ``rl_ablate``. Writes
``results/eval_tables_full.md``. The headline figure is produced separately by
``plot_base_disturbance.py``.

Usage (from repo root, with the project .venv; needs the user site-packages on PYTHONPATH):

    python -m robotic_capture.make_results
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv
from robotic_capture.control.baselines import ARCController, ZeroController, PolicyController
from robotic_capture.eval import evaluate, bootstrap_ci

RUNS = "robotic_capture/runs"
MAIN = [f"{RUNS}/sac_dist_seed{i}" for i in range(3)]
ABLATE = [f"{RUNS}/sac_dist_ablate_seed{i}" for i in range(3)]
SPINS, SEEDS, TOL, MAXSTEPS = [0.2, 0.3, 0.4], list(range(5)), 0.05, 500
DISTS = [0.3, 0.4, 0.5, 0.7]      # ~0.05, 0.07, 0.09, 0.12 rad/s open-loop
SCALES = [0.5, 1.0, 1.5, 2.0]
SLOPE = 0.175                     # rad/s open-loop base spin per N.m (calibration)
OUT = "robotic_capture/results/eval_tables_full.md"


def _named(paths: list[str], prefix: str) -> dict:
    """Load policies as distinctly-named controllers so rows can be pooled by prefix."""
    ctrls = {}
    for i, p in enumerate(paths):
        # SECURITY: SAC.load unpickles the zip -- only load trusted, self-produced run files.
        pc = PolicyController(SAC.load(p))
        pc.name = f"{prefix}{i}"
        ctrls[pc.name] = pc
    return ctrls


def _ci(rows, pred, metric) -> tuple[float, float, float]:
    return bootstrap_ci([r[metric] for r in rows if pred(r)])


def _fmt(t) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def _succ(rows, pred) -> float:
    v = [r["success"] for r in rows if pred(r)]
    return 100 * float(np.mean(v)) if v else float("nan")


def main() -> None:
    env = FreeFlyerCaptureEnv(domain_randomize=False)
    env.reset()
    arc, zero = ARCController(env.model), ZeroController()
    rl, ablate = _named(MAIN, "rl_s"), _named(ABLATE, "abl_s")
    is_rl = lambda c: c.startswith("rl_s")
    is_abl = lambda c: c.startswith("abl_s")
    lines: list[str] = []

    # Sweep 1 -- disturbance rejection (headline + ablation), inertia = 1.0
    r1 = evaluate({"zero": zero, "arc": arc, **rl, **ablate}, scales=[1.0], spins=SPINS,
                  seeds=SEEDS, dists=DISTS, detumble_tol=TOL, max_steps=MAXSTEPS)
    lines += ["## Sweep 1 -- Disturbance rejection (inertia=1.0x): peak base disturbance (rad/s)",
              "",
              "| inj (rad/s) | RL (ours) | RL ablate (w_b=0) | ARC | zero |",
              "|---|---|---|---|---|"]
    for d in DISTS:
        at = lambda r, d=d: r["base_disturbance"] == d
        lines.append(
            f"| {d*SLOPE:.2f} "
            f"| {_fmt(_ci(r1, lambda r, d=d: at(r) and is_rl(r['controller']), 'base_disturb_peak'))} "
            f"| {_fmt(_ci(r1, lambda r, d=d: at(r) and is_abl(r['controller']), 'base_disturb_peak'))} "
            f"| {_fmt(_ci(r1, lambda r, d=d: at(r) and r['controller'] == 'arc', 'base_disturb_peak'))} "
            f"| {_fmt(_ci(r1, lambda r, d=d: at(r) and r['controller'] == 'zero', 'base_disturb_peak'))} |")

    # Sweep 2 -- inertia parity (no injected disturbance)
    r2 = evaluate({"zero": zero, "arc": arc, **rl}, scales=SCALES, spins=SPINS, seeds=SEEDS,
                  dists=[0.0], detumble_tol=TOL, max_steps=MAXSTEPS)
    lines += ["", "## Sweep 2 -- Inertia parity (no injected disturbance)",
              "",
              "| inertia | RL base | ARC base | RL spin% | ARC spin% | RL succ | ARC succ |",
              "|---|---|---|---|---|---|---|"]
    for s in SCALES:
        at = lambda r, s=s: r["inertia_scale"] == s
        lines.append(
            f"| {s:g}x "
            f"| {_fmt(_ci(r2, lambda r, s=s: at(r) and is_rl(r['controller']), 'base_disturb_peak'))} "
            f"| {_fmt(_ci(r2, lambda r, s=s: at(r) and r['controller'] == 'arc', 'base_disturb_peak'))} "
            f"| {_ci(r2, lambda r, s=s: at(r) and is_rl(r['controller']), 'spin_reduction_pct')[0]:.0f}% "
            f"| {_ci(r2, lambda r, s=s: at(r) and r['controller'] == 'arc', 'spin_reduction_pct')[0]:.0f}% "
            f"| {_succ(r2, lambda r, s=s: at(r) and is_rl(r['controller'])):.0f}% "
            f"| {_succ(r2, lambda r, s=s: at(r) and r['controller'] == 'arc'):.0f}% |")

    # Sweep 3 -- gap vs target mass at a fixed mid disturbance (0.5 N.m)
    r3 = evaluate({"zero": zero, "arc": arc, **rl}, scales=SCALES, spins=SPINS, seeds=SEEDS,
                  dists=[0.5], detumble_tol=TOL, max_steps=MAXSTEPS)
    lines += ["", "## Sweep 3 -- Gap vs target mass at disturbance 0.5 N.m (~0.09 rad/s)",
              "",
              "| inertia | RL base peak | ARC base peak | ratio ARC/RL |",
              "|---|---|---|---|"]
    for s in SCALES:
        at = lambda r, s=s: r["inertia_scale"] == s
        rl_m = _ci(r3, lambda r, s=s: at(r) and is_rl(r['controller']), 'base_disturb_peak')[0]
        arc_m = _ci(r3, lambda r, s=s: at(r) and r['controller'] == 'arc', 'base_disturb_peak')[0]
        lines.append(f"| {s:g}x | {rl_m:.3f} | {arc_m:.3f} | {arc_m / rl_m:.1f}x |")

    text = "\n".join(lines) + "\n"
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
