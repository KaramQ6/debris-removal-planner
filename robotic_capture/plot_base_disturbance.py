"""Figure: induced base-attitude disturbance vs target inertia scale.

Three lines on one axis -- **RL-DR** (the domain-randomized SAC policy, pooled over
its training seeds), **ARC** (the model-based baseline), and **zero** (the do-nothing
reference). Reuses the paired evaluation sweep (``eval.evaluate``) so every controller
sees the same (inertia scale, spin, seed) episodes, then aggregates the headline metric
-- final base angular velocity (rad/s; proxy for chaser ACS fuel) -- per inertia scale
with a bootstrap 95% CI band. Writes ``results/fig_base_disturbance.pdf``.

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dr-policies", nargs="+",
                   default=["robotic_capture/runs/sac_dr",
                            "robotic_capture/runs/sac_dr_seed1",
                            "robotic_capture/runs/sac_dr_seed2"],
                   help="DR-trained SAC policy paths (one per seed); pooled into the RL-DR line")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
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
    for i, path in enumerate(args.dr_policies):
        pc = PolicyController(SAC.load(path))
        pc.name = f"rl_seed{i}"  # rollout() reads controller.name for the row label
        controllers[pc.name] = pc
    print(f"Sweeping {list(controllers)} over scales={args.scales} "
          f"spins={args.spins} seeds={args.seeds}")

    rows = evaluate(controllers, scales=args.scales, spins=args.spins, seeds=args.seeds,
                    detumble_tol=args.detumble_tol, max_steps=args.max_steps)

    def line(keep) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-scale (mean, ci_low, ci_high) of base disturbance for matching controllers."""
        mean, lo, hi = [], [], []
        for s in args.scales:
            vals = [r["base_disturb_final"] for r in rows
                    if r["inertia_scale"] == s and keep(r["controller"])]
            m, l, h = bootstrap_ci(vals)
            mean.append(m); lo.append(l); hi.append(h)
        return np.array(mean), np.array(lo), np.array(hi)

    series = {
        "RL-DR (ours)": (line(lambda c: c.startswith("rl_")), "#1f77b4", "o"),
        "ARC (baseline)": (line(lambda c: c == "arc"), "#d62728", "s"),
        "zero (reference)": (line(lambda c: c == "zero"), "#7f7f7f", "^"),
    }

    x = np.asarray(args.scales, dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for label, ((mean, lo, hi), color, marker) in series.items():
        ax.plot(x, mean, marker=marker, color=color, lw=1.9, ms=5.5, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.set_xlabel(r"target inertia scale ($\times$ nominal)")
    ax.set_ylabel("induced base disturbance (rad/s)")
    ax.set_title("Induced base-attitude disturbance vs target inertia")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"\nWrote {out}  (RL-DR pooled over {len(args.dr_policies)} seeds)")
    print("scale  RL-DR              ARC                zero")
    for j, s in enumerate(x):
        cells = "  ".join(f"{m[j]:.3f}[{l[j]:.3f},{h[j]:.3f}]"
                          for (m, l, h), _, _ in series.values())
        print(f"{s:>4g}   {cells}")


if __name__ == "__main__":
    main()
