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

    # Compact wide-banner layout so the two-column figure* packs onto the page with text
    # (a tall diagram floats alone and leaves a large white gap).
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    _box(ax, 0.4, 0.7, 4.4, 1.8,
         "Free-floating environment (MuJoCo)\n"
         "chaser base + planar 3-DOF arm\n"
         "+ tumbling target,  compliant contact", BLUE)
    _box(ax, 6.7, 1.05, 3.0, 1.1, "SAC policy $\\pi$\n$[256,256]$ MLP", GREEN)

    # obs (top, env -> policy) and action (bottom, policy -> env)
    _arrow(ax, (4.8, 1.95), (6.7, 1.95))
    ax.text(5.75, 2.12, "obs $s\\in\\mathbb{R}^{32}$ (kinematics only)",
            ha="center", va="bottom", fontsize=8)
    _arrow(ax, (6.7, 1.25), (4.8, 1.25))
    ax.text(5.75, 1.08, "torque $\\tau$ ($\\pm 5$ N$\\cdot$m)", ha="center", va="top", fontsize=8)

    # injected base disturbance (label + red arrow into the env top)
    ax.text(2.6, 3.12, "base disturbance $w_{\\mathrm{ext}}$  (ACS/thruster-noise proxy)",
            ha="center", va="top", fontsize=8, color="#b22222")
    _arrow(ax, (2.6, 2.82), (2.6, 2.5), color="#b22222", lw=1.6)

    # reward channel (training signal), routed below as a dashed feedback path
    dash = dict(color="#555555", lw=1.2, ls=(0, (4, 3)))
    ax.plot([4.2, 4.2], [0.7, 0.4], **dash)
    ax.plot([4.2, 8.2], [0.4, 0.4], **dash)
    _arrow(ax, (8.2, 0.4), (8.2, 1.05), color="#555555", lw=1.2, ls=(0, (4, 3)))
    ax.text(6.0, 0.28, "reward $r$ (Eq. 1)", ha="center", va="top", fontsize=8, color="#555555")

    fig.tight_layout(pad=0.2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
