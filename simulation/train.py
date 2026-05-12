"""Train MaskablePPO on the debris-removal Gymnasium environment."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on debris-removal environment."
    )
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=12000.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenario", type=str, default="curriculum",
        help="Scenario preset name (easy, medium, hard, shakti, curriculum) or path to Celestrak JSON file."
    )
    parser.add_argument(
        "--output", type=str, default=r"results\models\ppo_debris"
    )
    parser.add_argument(
        "--load-model", type=str, default="",
        help="Path to an existing model .zip to resume training from."
    )
    return parser.parse_args()


def linear_schedule(initial_lr: float):
    """Return a callable that linearly decays learning rate to 0."""

    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr

    return _schedule


def mask_fn(env):
    """Extract action masks from the unwrapped environment."""
    return env.action_masks()


def main() -> None:
    args = parse_args()
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise ImportError(
            "sb3-contrib and stable-baselines3 are required for training. "
            "Install: pip install sb3-contrib stable-baselines3"
        ) from exc

    import functools
    from .callbacks import EpisodeMetricsCallback
    from .orbit_env import OrbitDebrisEnv
    from .scenario import SCENARIO_PRESETS, from_celestrak_json
    
    scenario_path = Path(args.scenario)
    if scenario_path.exists() and scenario_path.suffix.lower() == '.json':
        scenario_generator = functools.partial(from_celestrak_json, filepath=scenario_path)
    else:
        scenario_generator = SCENARIO_PRESETS.get(args.scenario, SCENARIO_PRESETS["medium"])

    def make_env(rank: int, seed: int = 0):
        """Helper to create a single environment instance."""
        def _init():
            env = OrbitDebrisEnv(
                scenario_generator=scenario_generator,
                seed=seed + rank,
                target_count=args.targets,
                fuel_budget=args.fuel,
                max_steps=args.max_steps,
            )
            env = ActionMasker(env, mask_fn)
            env = Monitor(env)
            return env
        set_random_seed(seed)
        return _init

    # Parallel environments. 4 keeps RAM usage well under 1 GiB while still
    # delivering ~900 fps on a modern GPU — bumping higher previously caused
    # transient allocator failures on Windows.
    num_cpu = 8
    train_env = SubprocVecEnv([make_env(i, args.seed) for i in range(num_cpu)])

    # Separate eval environment
    raw_eval_env = OrbitDebrisEnv(
        scenario_generator=scenario_generator,
        seed=args.seed + 999,
        target_count=args.targets,
        fuel_budget=args.fuel,
        max_steps=args.max_steps,
    )
    masked_eval_env = ActionMasker(raw_eval_env, mask_fn)
    eval_env = Monitor(masked_eval_env)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Callbacks
    metrics_cb = EpisodeMetricsCallback(
        output_path=str(output_path.parent.parent / "training_history.json"), verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str((output_path.parent / "best_model").resolve()),
        log_path=str((output_path.parent / "eval_logs").resolve()),
        eval_freq=max(1000, 10000 // num_cpu),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    # EvalCallback in SB3 does not forward action masks during predict(), so
    # its "best_model" can mis-rank checkpoints. Independent periodic snapshots
    # let us pick the true late-training policy after the run.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 50_000 // num_cpu),
        save_path=str((output_path.parent / "checkpoints").resolve()),
        name_prefix="ppo_ckpt",
    )

    tb_log = None

    if args.load_model:
        print(f"Loading existing model from: {args.load_model}")
        model = MaskablePPO.load(
            args.load_model,
            env=train_env,
            tensorboard_log=tb_log,
            device="auto",
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=linear_schedule(3e-4),
            n_steps=2048,
            batch_size=256, # Increased for GPU speed
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=args.seed,
            device="auto",
            tensorboard_log=tb_log,
        )

    print(f"Training MaskablePPO for {args.timesteps:,} timesteps ...")
    print(f"  Targets: {args.targets}  |  Fuel budget: {args.fuel} m/s")
    print(f"  Device: {model.device}")
    print(f"  Model output: {output_path}")
    if tb_log:
        print(f"  TensorBoard logs: {tb_log}")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[metrics_cb, eval_cb, checkpoint_cb],
        progress_bar=True,
    )
    model.save(str(output_path))

    # Quick deterministic rollout
    obs, _ = eval_env.reset(seed=args.seed + 888)
    done = False
    while not done:
        action_masks = masked_eval_env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _, terminated, truncated, info = eval_env.step(int(action))
        done = terminated or truncated

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Model saved to: {output_path}.zip")
    print(f"Best model saved to: {output_path.parent / 'best_model'}")
    print(
        f"Rollout -> cleared: {info['cleared']}/{info['total_targets']}, "
        f"delta-v: {info['total_delta_v']:.2f} m/s, "
        f"fuel remaining: {info['fuel_remaining']:.2f} m/s"
    )
    if metrics_cb.summary:
        s = metrics_cb.summary
        print(
            f"Training avg -> reward: {s['mean_reward']:.2f}, "
            f"delta-v: {s['mean_delta_v']:.2f}, "
            f"cleared: {s['mean_cleared']:.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
