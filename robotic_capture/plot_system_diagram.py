"""Method/system overview figure: the closed-loop control diagram.

Draws the RL control loop -- free-floating MuJoCo environment, kinematic observation,
SAC policy, joint-torque action -- with the injected chaser-side base disturbance and the
reward channel, as a vector diagram. Pure matplotlib (no LaTeX/TikZ package needed), saved
to ``docs/paper/figures/fig_system.pdf`` so it matches the other figures.

Usage (from the robotic_capture/ dir, with the project .venv):

    python plot_system_diagram.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, GREEN, RED = "#dce6f2", "#e6f0dd", "#f3dada"


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec="#333333", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, p0, p1, color="#333333", style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="docs/paper/figures/fig_system.pdf")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    _box(ax, 0.4, 1.7, 4.6, 2.05,
         "Free-floating environment (MuJoCo)\n"
         "chaser base + planar 3-DOF arm\n"
         "+ tumbling non-cooperative target\n"
         "compliant contact,  zero-$g$", BLUE)
    _box(ax, 6.6, 2.15, 3.0, 1.3, "SAC policy $\\pi$\n$[256,256]$ MLP", GREEN)

    # obs (top, env -> policy) and action (bottom, policy -> env)
    _arrow(ax, (5.0, 3.15), (6.6, 3.15))
    ax.text(5.8, 3.38, "obs $s\\in\\mathbb{R}^{32}$ (kinematics only)",
            ha="center", va="bottom", fontsize=8)
    _arrow(ax, (6.6, 2.45), (5.0, 2.45))
    ax.text(5.8, 2.22, "torque $\\tau$ ($\\pm 5$ N$\\cdot$m)", ha="center", va="top", fontsize=8)

    # injected base disturbance (top -> env)
    _box(ax, 0.7, 4.5, 3.9, 0.7,
         "base disturbance $w_{\\mathrm{ext}}$  (ACS/thruster-noise proxy)", RED)
    _arrow(ax, (2.65, 4.5), (2.65, 3.75), color="#b22222", lw=1.6)

    # reward channel (training signal), routed below everything as a dashed feedback path
    dash = dict(color="#555555", lw=1.2, ls=(0, (4, 3)))
    ax.plot([4.1, 4.1], [1.7, 1.0], **dash)
    ax.plot([4.1, 8.1], [1.0, 1.0], **dash)
    _arrow(ax, (8.1, 1.0), (8.1, 2.15), color="#555555", lw=1.2, ls=(0, (4, 3)))
    ax.text(6.1, 0.78, "reward $r$ (Eq. 1)", ha="center", va="top", fontsize=8,
            color="#555555")

    fig.tight_layout(pad=0.2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
