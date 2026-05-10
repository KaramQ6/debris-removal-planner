"""Generate publication-quality charts from evaluation and training results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. pip install matplotlib"
        ) from exc


def plot_delta_v_comparison(results: dict, output_path: Path) -> None:
    """Bar chart comparing average delta-v across policies."""
    plt = _ensure_matplotlib()

    names = list(results.keys())
    avg_dvs = [results[n]["avg_delta_v"] for n in names]
    std_dvs = [results[n].get("std_delta_v", 0) for n in names]

    colors = {
        "random": "#E74C3C",
        "nearest": "#F39C12",
        "risk_weighted": "#3498DB",
        "ppo": "#2ECC71",
    }
    bar_colors = [colors.get(n, "#95A5A6") for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, avg_dvs, yerr=std_dvs, capsize=5,
                  color=bar_colors, edgecolor="white", linewidth=1.5,
                  alpha=0.9)

    # Add value labels on bars
    for bar, val in zip(bars, avg_dvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold",
                fontsize=12)

    ax.set_ylabel("Average Total Delta-V (m/s)", fontsize=13)
    ax.set_xlabel("Planning Policy", fontsize=13)
    ax.set_title("Mission Fuel Consumption by Planning Strategy",
                 fontsize=15, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add improvement annotation
    if "random" in results and "ppo" in results:
        baseline = results["random"]["avg_delta_v"]
        ppo_dv = results["ppo"]["avg_delta_v"]
        improvement = (baseline - ppo_dv) / baseline * 100
        ax.annotate(
            f"RL Agent: {improvement:.1f}% fuel reduction\nvs random baseline",
            xy=(names.index("ppo"), ppo_dv),
            xytext=(names.index("ppo") + 0.3, baseline * 0.85),
            arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2),
            fontsize=11, color="#2ECC71", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#2ECC71", alpha=0.9),
        )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_delta_v_distribution(results: dict, output_path: Path) -> None:
    """Box plot showing delta-v distribution per policy."""
    plt = _ensure_matplotlib()

    names = list(results.keys())
    data = [results[n].get("all_delta_v", []) for n in names]

    colors = {
        "random": "#E74C3C",
        "nearest": "#F39C12",
        "risk_weighted": "#3498DB",
        "ppo": "#2ECC71",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))

    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(colors.get(name, "#95A5A6"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Total Delta-V (m/s)", fontsize=13)
    ax.set_xlabel("Planning Policy", fontsize=13)
    ax.set_title("Delta-V Distribution Across Episodes",
                 fontsize=15, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_training_reward_curve(history_path: Path, output_path: Path) -> None:
    """Line chart of training episode rewards over time."""
    plt = _ensure_matplotlib()

    data = json.loads(history_path.read_text(encoding="utf-8"))
    rewards = data.get("episode_rewards", [])

    if not rewards:
        print("  No training reward data found — skipping curve plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color="#3498DB", linewidth=0.5)

    # Smoothed line (rolling average)
    window = max(1, len(rewards) // 50)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:], smoothed,
            color="#2C3E50", linewidth=2, label=f"Rolling avg (w={window})"
        )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Episode Reward", fontsize=13)
    ax.set_title("PPO Training Reward Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_clear_rate_comparison(results: dict, output_path: Path) -> None:
    """Horizontal bar chart of full-clear rates."""
    plt = _ensure_matplotlib()

    names = list(results.keys())
    rates = [results[n]["full_clear_rate"] * 100 for n in names]

    colors = {
        "random": "#E74C3C",
        "nearest": "#F39C12",
        "risk_weighted": "#3498DB",
        "ppo": "#2ECC71",
    }
    bar_colors = [colors.get(n, "#95A5A6") for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(names, rates, color=bar_colors, edgecolor="white",
                   linewidth=1.5, alpha=0.9, height=0.5)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold", fontsize=12)

    ax.set_xlabel("Full Mission Clear Rate (%)", fontsize=13)
    ax.set_title("Mission Success Rate by Policy",
                 fontsize=15, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate charts from evaluation results."
    )
    parser.add_argument(
        "--eval-results",
        type=str,
        default=r"results\evaluation_summary.json",
    )
    parser.add_argument(
        "--training-history",
        type=str,
        default=r"results\training_history.json",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = Path(args.eval_results)
    history_path = Path(args.training_history)

    print("Generating charts...")

    if eval_path.exists():
        results = json.loads(eval_path.read_text(encoding="utf-8"))
        plot_delta_v_comparison(results, output_dir / "delta_v_comparison.png")
        plot_delta_v_distribution(results, output_dir / "delta_v_distribution.png")
        plot_clear_rate_comparison(results, output_dir / "clear_rate_comparison.png")
    else:
        print(f"  Evaluation results not found at {eval_path} — run evaluate first.")

    if history_path.exists():
        plot_training_reward_curve(
            history_path, output_dir / "training_reward_curve.png"
        )
    else:
        print(f"  Training history not found at {history_path} — run train first.")

    print("Done.")


if __name__ == "__main__":
    main()
