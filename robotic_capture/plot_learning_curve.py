"""Learning-curve figure: episode return vs training steps, pooled over 3 seeds.

Parses the SB3 stdout logs of the three disturbance-trained main runs
(``runs/sac_dist_seed{0,1,2}.log``) for the ``ep_rew_mean`` / ``total_timesteps``
rollout pairs, interpolates each seed onto a common step grid, and plots the
across-seed mean with a min--max band. No model loading or simulation is needed --
the curve is reconstructed entirely from the committed training logs. Writes
``docs/paper/figures/fig_learning_curve.pdf``.

Usage (from the robotic_capture/ dir, with the project .venv):

    python -m robotic_capture.plot_learning_curve
    python plot_learning_curve.py            # also works from this directory
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

# SB3 logs each rollout block as two table rows we care about, in this order:
#   |    ep_rew_mean     | -395     |
#   |    total_timesteps | 2000     |
_REW = re.compile(r"\|\s*ep_rew_mean\s*\|\s*([-\d.eE+]+)\s*\|")
_STEP = re.compile(r"\|\s*total_timesteps\s*\|\s*([-\d.eE+]+)\s*\|")


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, ep_rew_mean) arrays parsed from one SB3 stdout log."""
    steps, rews, pending = [], [], None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _REW.search(line)
        if m:
            pending = float(m.group(1))
            continue
        m = _STEP.search(line)
        if m and pending is not None:
            steps.append(float(m.group(1)))
            rews.append(pending)
            pending = None
    return np.asarray(steps), np.asarray(rews)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", nargs="+",
                   default=[f"runs/sac_dist_seed{i}.log" for i in range(3)],
                   help="SB3 stdout logs, one per training seed")
    p.add_argument("--points", type=int, default=200, help="common-grid resolution")
    p.add_argument("--out", type=str, default="docs/paper/figures/fig_learning_curve.pdf")
    args = p.parse_args()

    curves = [parse_log(Path(l)) for l in args.logs]
    curves = [(s, r) for s, r in curves if s.size > 1]
    if not curves:
        raise SystemExit("No usable (steps, reward) pairs parsed -- check --logs paths.")

    # Common grid over the steps all seeds actually cover (no extrapolation), then
    # interpolate each seed onto it so we can take an across-seed mean / band.
    lo = max(s.min() for s, _ in curves)
    hi = min(s.max() for s, _ in curves)
    grid = np.linspace(lo, hi, args.points)
    stack = np.vstack([np.interp(grid, s, r) for s, r in curves])
    mean, ymin, ymax = stack.mean(0), stack.min(0), stack.max(0)

    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for i, (s, r) in enumerate(curves):
        ax.plot(s / 1e3, r, color="#1f77b4", lw=0.8, alpha=0.25,
                label="per-seed" if i == 0 else None)
    ax.fill_between(grid / 1e3, ymin, ymax, color="#1f77b4", alpha=0.18, lw=0,
                    label="min--max (3 seeds)")
    ax.plot(grid / 1e3, mean, color="#1f77b4", lw=2.0, label="mean (3 seeds)")
    ax.set_xlabel("training steps ($\\times 10^3$)")
    ax.set_ylabel("episode return (ep\\_rew\\_mean)")
    ax.set_title("Training: detumbling policy (SAC, base-disturbance DR)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}  (pooled over {len(curves)} seeds, "
          f"grid {lo:.0f}-{hi:.0f} steps, final mean return {mean[-1]:.1f})")


if __name__ == "__main__":
    main()
