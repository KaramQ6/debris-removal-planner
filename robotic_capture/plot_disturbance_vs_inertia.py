"""Inertia-parity figure: peak base disturbance vs target inertia scale, RL vs ARC.

Companion to the headline disturbance-rejection figure. Shows the *contact-induced*
base disturbance with **no injected disturbance** as target inertia is swept
0.5x--2x: the learned policy stays low and nearly flat while ARC's force-tracking law
lets it rise steeply -- the inertia-parity story of Table~\\ref{tab:parity}.

The values mirror the committed parity table (Sweep 2 of ``results/eval_tables_full.md``,
which is the paper's Table~II), so the figure and table are guaranteed consistent and
no model loading / simulation is needed to redraw.

ponytail: numbers are mirrored from the parity table rather than recomputed, so a redraw
can never drift from the table. Upgrade path if the eval changes: re-run
``python -m robotic_capture.make_results`` and copy the new Sweep-2 means/CIs into ``DATA``.

Usage (from the robotic_capture/ dir, with the project .venv):

    python plot_disturbance_vs_inertia.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

# Inertia scale -> (mean, ci_low, ci_high) of peak base disturbance (rad/s), no injected
# disturbance. Source: results/eval_tables_full.md Sweep 2 == Table II (tab:parity).
SCALES = [0.5, 1.0, 1.5, 2.0]
DATA = {
    "RL (ours)":      {0.5: (0.018, 0.015, 0.020), 1.0: (0.022, 0.019, 0.025),
                       1.5: (0.038, 0.033, 0.043), 2.0: (0.046, 0.040, 0.053)},
    "ARC (baseline)": {0.5: (0.051, 0.044, 0.059), 1.0: (0.121, 0.091, 0.153),
                       1.5: (0.246, 0.179, 0.320), 2.0: (0.294, 0.234, 0.353)},
}
STYLE = {"RL (ours)": ("#1f77b4", "o"), "ARC (baseline)": ("#d62728", "s")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str,
                   default="docs/paper/figures/fig_disturbance_vs_inertia.pdf")
    args = p.parse_args()

    x = np.asarray(SCALES, dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for label, series in DATA.items():
        mean = np.array([series[s][0] for s in SCALES])
        lo = np.array([series[s][0] - series[s][1] for s in SCALES])
        hi = np.array([series[s][2] - series[s][0] for s in SCALES])
        color, marker = STYLE[label]
        ax.errorbar(x, mean, yerr=[lo, hi], color=color, marker=marker, lw=1.9, ms=5.5,
                    capsize=3, label=label)
    ax.set_xlabel("target inertia scale ($\\times$ nominal)")
    ax.set_ylabel("peak base disturbance (rad/s)")
    ax.set_title("Inertia parity (no injected disturbance)")
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}  (RL vs ARC over inertia {SCALES}, 95% CI error bars from Table II)")


if __name__ == "__main__":
    main()
