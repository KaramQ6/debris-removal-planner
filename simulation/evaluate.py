from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .orbit_env import OrbitDebrisEnv
from .policies import nearest_neighbor_policy, random_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and trained planning policies.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=1200.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output", type=str, default=r"results\evaluation_summary.json")
    return parser.parse_args()


def run_policy(
    name: str,
    episodes: int,
    seed: int,
    targets: int,
    fuel: float,
    max_steps: int,
    model_path: str = "",
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    total_delta_v = []
    total_cleared = []
    full_clear_count = 0

    model = None
    if name == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required to evaluate a trained PPO model."
            ) from exc
        model = PPO.load(model_path)

    for i in range(episodes):
        env = OrbitDebrisEnv(
            seed=seed + i,
            target_count=targets,
            fuel_budget=fuel,
            max_steps=max_steps,
        )
        obs, _ = env.reset(seed=seed + i)
        done = False
        info = {}

        while not done:
            if name == "random":
                action = random_policy(env, rng)
            elif name == "nearest":
                action = nearest_neighbor_policy(env)
            elif name == "ppo":
                action, _ = model.predict(obs, deterministic=True)  # type: ignore[union-attr]
                action = int(action)
            else:
                raise ValueError(f"Unknown policy name: {name}")

            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        total_delta_v.append(float(info["total_delta_v"]))
        total_cleared.append(float(info["cleared"]))
        if int(info["cleared"]) == int(info["total_targets"]):
            full_clear_count += 1

    return {
        "avg_delta_v": float(np.mean(total_delta_v)),
        "avg_cleared": float(np.mean(total_cleared)),
        "full_clear_rate": full_clear_count / episodes,
    }


def main() -> None:
    args = parse_args()
    results = {
        "random": run_policy(
            name="random",
            episodes=args.episodes,
            seed=args.seed,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
        ),
        "nearest": run_policy(
            name="nearest",
            episodes=args.episodes,
            seed=args.seed + 1000,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
        ),
    }

    if args.model_path:
        results["ppo"] = run_policy(
            name="ppo",
            episodes=args.episodes,
            seed=args.seed + 2000,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
            model_path=args.model_path,
        )

    print("| Policy | Avg Delta-V (m/s) | Avg Cleared Targets | Full-Clear Rate |")
    print("| --- | ---: | ---: | ---: |")
    for name, metrics in results.items():
        print(
            f"| {name} | {metrics['avg_delta_v']:.2f} | {metrics['avg_cleared']:.2f} | "
            f"{100.0 * metrics['full_clear_rate']:.1f}% |"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()

