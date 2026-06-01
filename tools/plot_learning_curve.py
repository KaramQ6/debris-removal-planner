"""Plot publication-quality learning curves from training_history.json.

Produces two subplots:
  1. Episode reward vs training episode (rolling mean, window=100)
  2. Targets cleared vs training episode (rolling mean, window=100)

Output saved to ``assets/learning_curve.png`` at 300 DPI.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORY_PATH = os.path.join(ROOT, "results", "training_history.json")
ASSETS_DIR = os.path.join(ROOT, "assets")
OUTPUT_PATH = os.path.join(ASSETS_DIR, "learning_curve.png")


def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean using cumsum for efficiency (no scipy needed)."""
    cs = np.cumsum(data)
    cs = np.insert(cs, 0, 0.0)
    return (cs[window:] - cs[:-window]) / window


def main() -> None:
    if not os.path.isfile(HISTORY_PATH):
        sys.exit(f"ERROR: {HISTORY_PATH} not found")

    print(f"Loading {HISTORY_PATH} …")
    with open(HISTORY_PATH, "r") as f:
        data = json.load(f)

    rewards = np.array(data["episode_rewards"], dtype=np.float64)
    cleared = np.array(data.get("episode_cleared", []), dtype=np.float64)

    has_cleared = len(cleared) == len(rewards)
    n_episodes = len(rewards)
    window = 100

    print(f"  Episodes: {n_episodes:,}")
    print(f"  Rolling window: {window}")

    # Compute rolling averages
    reward_smooth = rolling_mean(rewards, window)
    x_smooth = np.arange(window, n_episodes + 1)  # align x-axis

    if has_cleared:
        cleared_smooth = rolling_mean(cleared, window)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    n_panels = 2 if has_cleared else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.5 * n_panels),
                             constrained_layout=True, dpi=150)
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: Reward ---
    ax = axes[0]
    # Light raw data in background (subsample for speed)
    step = max(1, n_episodes // 5000)
    ax.scatter(np.arange(0, n_episodes, step), rewards[::step],
               s=0.15, alpha=0.08, color="#6C757D", rasterized=True, zorder=1)
    ax.plot(x_smooth, reward_smooth, color="#0D6EFD", linewidth=1.4, zorder=2,
            label=f"Rolling mean (w={window})")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curve — Episode Reward", fontweight="bold", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel 2: Targets cleared ---
    if has_cleared:
        ax2 = axes[1]
        ax2.scatter(np.arange(0, n_episodes, step), cleared[::step],
                    s=0.15, alpha=0.08, color="#6C757D", rasterized=True, zorder=1)
        ax2.plot(x_smooth, cleared_smooth, color="#198754", linewidth=1.4, zorder=2,
                 label=f"Rolling mean (w={window})")
        ax2.set_xlabel("Training Episode")
        ax2.set_ylabel("Targets Cleared")
        ax2.set_title("Learning Curve — Targets Cleared per Episode",
                       fontweight="bold", fontsize=11)
        ax2.legend(frameon=False, fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
