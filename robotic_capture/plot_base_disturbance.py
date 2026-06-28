"""Headline figure: base-attitude disturbance REJECTION vs injected disturbance.

Three lines on one axis -- **RL (ours)** (the disturbance-trained SAC policy, pooled over
its training seeds), **ARC** (the model-based baseline, which has no base-attitude term),
and **zero** (the do-nothing reference, whose base spin equals the injected disturbance).
Reuses the paired evaluation sweep (``eval.evaluate``) at fixed nominal inertia so every
controller sees the same (disturbance, spin, seed) episodes, then aggregates the rejection
metric -- peak induced base angular velocity (rad/s; proxy for chaser ACS fuel) -- per
injected disturbance level with a bootstrap 95% CI band. Writes
``results/fig_base_disturbance.pdf``.

The injected disturbance is a constant base torque (N·m); the x-axis is shown in the
calibrated open-loop base spin it produces (rad/s) -- see ``DIST_SLOPE`` below and the
calibration in the session notes (~0.175 rad/s of open-loop base spin per N·m).

Usage (from repo root, with the project .venv):

    python -m robotic_capture.plot_base_disturbance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv
from robotic_capture.control.baselines import ARCController, ZeroController, PolicyController
from robotic_capture.eval import evaluate, bootstrap_ci

# Calibration: open-loop (zero-control) base spin induced per N·m of injected base torque
# at nominal inertia (measured over a zero-action sweep). Used only to label the x-axis in
# the reviewer-friendly rad/s units; the underlying sweep variable is the torque itself.
DIST_SLOPE = 0.175  # rad/s per N·m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies", nargs="+",
                   default=["robotic_capture/runs/sac_dist_seed0",
                            "robotic_capture/runs/sac_dist_seed1",
                            "robotic_capture/runs/sac_dist_seed2"],
                   help="disturbance-trained SAC policy paths (one per seed); pooled into the RL line")
    p.add_argument("--dists", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.7],
                   help="injected base disturbance torques (N·m); ~0.05-0.12 rad/s open-loop")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0],
                   help="target inertia scale(s); headline fixes this at nominal (1.0)")
    p.add_argument("--spins", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--detumble-tol", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out", type=str,
                   default="robotic_capture/results/fig_base_disturbance.pdf")
    args = p.parse_args()

    # ARC needs a model handle; the policies are loaded as named controllers so the
    # shared sweep tags their rows distinctly (rl_seed0/1/2) before pooling.
    model_env = FreeFlyerCaptureEnv(domain_randomize=False)
    model_env.reset()
    controllers = {"zero": ZeroController(), "arc": ARCController(model_env.model)}
    for i, path in enumerate(args.policies):
        # SECURITY: SAC.load unpickles the zip -- only load trusted, self-produced run files.
        pc = PolicyController(SAC.load(path))
        pc.name = f"rl_seed{i}"  # rollout() reads controller.name for the row label
        controllers[pc.name] = pc
    print(f"Sweeping {list(controllers)} over dists={args.dists} scales={args.scales} "
          f"spins={args.spins} seeds={args.seeds}")

    rows = evaluate(controllers, scales=args.scales, spins=args.spins, seeds=args.seeds,
                    dists=args.dists, detumble_tol=args.detumble_tol, max_steps=args.max_steps)

    def line(keep) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-disturbance (mean, ci_low, ci_high) of peak base disturbance for matching ctrls."""
        mean, lo, hi = [], [], []
        for d in args.dists:
            vals = [r["base_disturb_peak"] for r in rows
                    if r["base_disturbance"] == d and keep(r["controller"])]
            m, l, h = bootstrap_ci(vals)
            mean.append(m); lo.append(l); hi.append(h)
        return np.array(mean), np.array(lo), np.array(hi)

    series = {
        "RL (ours)": (line(lambda c: c.startswith("rl_")), "#1f77b4", "o"),
        "ARC (baseline)": (line(lambda c: c == "arc"), "#d62728", "s"),
        "zero (reference)": (line(lambda c: c == "zero"), "#7f7f7f", "^"),
    }

    x = np.asarray(args.dists, dtype=float) * DIST_SLOPE  # injected disturbance in rad/s
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for label, ((mean, lo, hi), color, marker) in series.items():
        ax.plot(x, mean, marker=marker, color=color, lw=1.9, ms=5.5, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.set_xlabel("injected base disturbance (rad/s, open-loop)")
    ax.set_ylabel("peak base disturbance (rad/s)")
    ax.set_title("Base-attitude disturbance rejection")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\nWrote {out}  (RL pooled over {len(args.policies)} seeds)")
    print("inj(rad/s)  RL(ours)           ARC                zero")
    for j, v in enumerate(x):
        cells = "  ".join(f"{m[j]:.3f}[{l[j]:.3f},{h[j]:.3f}]"
                          for (m, l, h), _, _ in series.values())
        print(f"{v:>7.2f}    {cells}")


if __name__ == "__main__":
    main()
