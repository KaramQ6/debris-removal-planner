"""Evaluate baseline and trained planning policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .orbit_env import OrbitDebrisEnv
from .policies import nearest_neighbor_policy, random_policy, risk_weighted_policy


POLICY_NAMES = ["random", "nearest", "risk_weighted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and trained planning policies."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=6000.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument(
        "--scenario", type=str, default="medium",
        help="Scenario preset name or path to Celestrak JSON file."
    )
    parser.add_argument(
        "--output", type=str, default=r"results\evaluation_summary.json"
    )
    return parser.parse_args()


def run_policy(
    name: str,
    episodes: int,
    seed: int,
    targets: int,
    fuel: float,
    max_steps: int,
    model_path: str = "",
    scenario_generator: callable | None = None,
) -> dict[str, object]:
    """Run *episodes* rollouts of a named policy and return aggregate metrics."""
    rng = np.random.default_rng(seed)
    all_delta_v: list[float] = []
    all_cleared: list[int] = []
    all_fuel_remaining: list[float] = []
    full_clear_count = 0
    sample_trajectories: list[list[dict]] = []

    model = None
    if name == "ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError(
                "sb3-contrib is required to evaluate a trained MaskablePPO model. "
                "Install: pip install sb3-contrib"
            ) from exc
        model = MaskablePPO.load(model_path)

    for i in range(episodes):
        env = OrbitDebrisEnv(
            scenario_generator=scenario_generator,
            seed=seed + i,
            target_count=targets,
            fuel_budget=fuel,
            max_steps=max_steps,
        )
        obs, _ = env.reset(seed=seed + i)
        done = False
        info: dict = {}

        while not done:
            if name == "random":
                action = random_policy(env, rng)
            elif name == "nearest":
                action = nearest_neighbor_policy(env)
            elif name == "risk_weighted":
                action = risk_weighted_policy(env)
            elif name == "ppo":
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)  # type: ignore[union-attr]
                action = int(action)
            else:
                raise ValueError(f"Unknown policy name: {name}")

            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        dv = float(info["total_delta_v"])
        cleared = int(info["cleared"])
        fuel_rem = float(info["fuel_remaining"])

        all_delta_v.append(dv)
        all_cleared.append(cleared)
        all_fuel_remaining.append(fuel_rem)
        if cleared == int(info["total_targets"]):
            full_clear_count += 1

        # Save first 3 trajectories for visualization
        if i < 3:
            sample_trajectories.append(info.get("trajectory", []))

    # Calculate advanced metrics
    avg_cleared = float(np.mean(all_cleared))
    total_targets = targets
    accuracy = avg_cleared / total_targets

    # Precision/Recall/F1: Operational tracking metrics (not strict ML classification)
    # In this RL task, every target collected = True Positive, every missed = False Negative.
    # Note: Precision is fixed at 1.0 because the problem has no False Positives
    # (agent can't "hallucinate" debris; it only collects real targets).
    # F1 = Recall in this context, so these are best interpreted as task performance signals.
    tp = sum(all_cleared)
    fn = episodes * targets - tp
    precision = 1.0 if tp + fn == 0 else tp / (tp + 0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Completion rates: percentage of episodes reaching target thresholds
    completion_rate = sum(1 for c in all_cleared if c >= 1) / episodes
    full_completion_rate = sum(1 for c in all_cleared if c == total_targets) / episodes

    # Fuel efficiency: ΔV per target cleared (only for episodes where targets were cleared)
    cleared_episodes = [c for c in all_cleared if c > 0]
    dv_for_cleared = [all_delta_v[i] for i, c in enumerate(all_cleared) if c > 0]
    fuel_per_target = (
        float(np.mean([dv / c for dv, c in zip(dv_for_cleared, cleared_episodes)]))
        if len(cleared_episodes) > 0 else float('inf')
    )

    # Regression metrics for fuel usage (Actual vs Idealized Budget)
    # Note: MAE/MSE/RMSE will often be near 0 if agent consumes the full budget
    fuel_usage = [fuel - r for r in all_fuel_remaining]
    y_true = [fuel] * episodes
    mae = float(np.mean(np.abs(np.array(y_true) - np.array(fuel_usage))))
    mse = float(np.mean((np.array(y_true) - np.array(fuel_usage))**2))
    rmse = float(np.sqrt(mse))

    return {
        "avg_delta_v": float(np.mean(all_delta_v)),
        "std_delta_v": float(np.std(all_delta_v)),
        "min_delta_v": float(np.min(all_delta_v)),
        "max_delta_v": float(np.max(all_delta_v)),
        "avg_cleared": avg_cleared,
        "avg_fuel_remaining": float(np.mean(all_fuel_remaining)),
        "full_clear_rate": full_clear_count / episodes,
        "completion_rate": completion_rate,
        "full_completion_rate": full_completion_rate,
        "fuel_per_target": fuel_per_target,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "episodes": episodes,
        "all_delta_v": all_delta_v,
        "all_cleared": [int(c) for c in all_cleared],
        "sample_trajectories": sample_trajectories,
    }


def main() -> None:
    args = parse_args()
    results: dict[str, dict] = {}

    import functools
    from .scenario import SCENARIO_PRESETS, from_celestrak_json
    
    scenario_path = Path(args.scenario)
    if scenario_path.exists() and scenario_path.suffix.lower() == '.json':
        scenario_generator = functools.partial(from_celestrak_json, filepath=scenario_path)
    else:
        scenario_generator = SCENARIO_PRESETS.get(args.scenario, SCENARIO_PRESETS["medium"])

    for idx, name in enumerate(POLICY_NAMES):
        print(f"Evaluating {name} policy ({args.episodes} episodes)...")
        results[name] = run_policy(
            name=name,
            episodes=args.episodes,
            seed=args.seed + idx * 1000,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
            scenario_generator=scenario_generator,
        )

    if args.model_path:
        print(f"Evaluating PPO model ({args.episodes} episodes)...")
        results["ppo"] = run_policy(
            name="ppo",
            episodes=args.episodes,
            seed=args.seed + 3000,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
            model_path=args.model_path,
            scenario_generator=scenario_generator,
        )

    # Print comparison table
    print()
    print(
        f"{'Policy':<15} {'Delta-V':>10} {'Cleared':>8} {'Acc %':>7} {'F1':>6} {'RMSE':>8} {'Fuel Left':>10}"
    )
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<15} {m['avg_delta_v']:>10.1f} {m['avg_cleared']:>8.2f} "
            f"{100.0 * m['accuracy']:>6.1f}% {m['f1_score']:>6.2f} {m['rmse']:>8.1f} "
            f"{m['avg_fuel_remaining']:>10.1f}"
        )

    # Improvement metrics
    if "random" in results and len(results) > 1:
        baseline_dv = results["random"]["avg_delta_v"]
        print()
        print("Improvements over random baseline:")
        for name, m in results.items():
            if name == "random":
                continue
            improvement = (baseline_dv - m["avg_delta_v"]) / baseline_dv * 100
            print(f"  {name}: {improvement:+.1f}% delta-v reduction")

    # Save results (exclude large arrays for readability)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for name, m in results.items():
        save_data[name] = {
            k: v
            for k, v in m.items()
            if k not in ("sample_trajectories",)
        }

    output_path.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
