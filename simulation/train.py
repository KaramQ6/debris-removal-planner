from __future__ import annotations

import argparse
from pathlib import Path

from .orbit_env import OrbitDebrisEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on debris-removal environment.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=1200.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=r"results\models\ppo_debris")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for training. Install dependencies from requirements.txt."
        ) from exc

    env = OrbitDebrisEnv(
        seed=args.seed,
        target_count=args.targets,
        fuel_budget=args.fuel,
        max_steps=args.max_steps,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        gamma=0.99,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    obs, _ = env.reset(seed=args.seed + 999)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    print("Training complete.")
    print(f"Model saved to: {output_path}.zip")
    print(
        f"One deterministic rollout -> cleared: {info['cleared']}/{info['total_targets']}, "
        f"delta-v: {info['total_delta_v']:.2f} m/s, fuel remaining: {info['fuel_remaining']:.2f} m/s"
    )


if __name__ == "__main__":
    main()

