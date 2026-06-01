"""Evaluate baseline and trained planning policies under dynamic perturbations."""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .orbit_env import OrbitDebrisEnv
from .policies import (
    branch_and_bound_policy,
    nearest_neighbor_policy,
    random_policy,
    risk_weighted_policy,
)
from .scenario import SCENARIO_PRESETS, DebrisTarget, from_celestrak_json


class DynamicOrbitDebrisEnv(OrbitDebrisEnv):
    """Subclass of OrbitDebrisEnv that introduces dynamic orbital perturbations on targets."""

    def __init__(
        self,
        scenario_generator: Callable[..., Any] | None = None,
        *,
        max_targets: int = 12,
        seed: int | None = None,
        target_count: int = 8,
        fuel_budget: float = 6000.0,
        max_steps: int = 50,
        drift_std: float = 0.05,
        replacement_prob: float = 0.05,
    ) -> None:
        self.drift_std = drift_std
        self.replacement_prob = replacement_prob
        super().__init__(
            scenario_generator,
            max_targets=max_targets,
            seed=seed,
            target_count=target_count,
            fuel_budget=fuel_budget,
            max_steps=max_steps,
        )

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if not (terminated or truncated):
            # 1. Apply atmospheric drag SMA decay and secular perturbations
            for i in range(len(self._targets)):
                if self._active[i]:
                    t = self._targets[i]
                    
                    # Drift elements (drag reduces SMA slightly)
                    delta_sma = self.np_random.normal(-0.05, 0.02)
                    delta_ecc = self.np_random.normal(0.0, 0.0001)
                    delta_inc = self.np_random.normal(0.0, 0.02)
                    delta_raan = self.np_random.normal(0.0, 0.1)
                    delta_arg_p = self.np_random.normal(0.0, 0.1)
                    delta_nu = self.np_random.normal(0.2, 0.05)  # True anomaly motion
                    
                    new_sma = max(6471.0, t.sma_km + delta_sma)  # perigee > 100km altitude
                    new_ecc = max(0.0, min(0.9, t.eccentricity + delta_ecc))
                    new_inc = max(0.0, min(180.0, t.inclination_deg + delta_inc))
                    new_raan = (t.raan_deg + delta_raan) % 360
                    new_arg_p = (t.arg_periapsis_deg + delta_arg_p) % 360
                    new_nu = (t.true_anomaly_deg + delta_nu) % 360

                    self._targets[i] = DebrisTarget(
                        target_id=t.target_id,
                        sma_km=new_sma,
                        eccentricity=new_ecc,
                        inclination_deg=new_inc,
                        raan_deg=new_raan,
                        arg_periapsis_deg=new_arg_p,
                        true_anomaly_deg=new_nu,
                        target_type=t.target_type,
                        age_days=t.age_days + 1.0,
                        risk=t.risk,
                        name=t.name,
                    )

            # 2. Dynamic target replacement: simulates tracking system updates
            if self.np_random.random() < self.replacement_prob:
                active_indices = np.flatnonzero(self._active)
                if len(active_indices) > 0:
                    replace_idx = self.np_random.choice(active_indices)
                    
                    # Spawn a fresh new debris target in LEO catalog bounds
                    new_sma = float(self.np_random.uniform(7000.0, 7300.0))
                    new_ecc = float(self.np_random.uniform(0.0, 0.05))
                    new_inc = float(self.np_random.uniform(28.0, 98.0))
                    new_raan = float(self.np_random.uniform(0.0, 360.0))
                    new_arg_p = float(self.np_random.uniform(0.0, 360.0))
                    new_nu = float(self.np_random.uniform(0.0, 360.0))
                    
                    self._targets[replace_idx] = DebrisTarget(
                        target_id=int(replace_idx),
                        sma_km=new_sma,
                        eccentricity=new_ecc,
                        inclination_deg=new_inc,
                        raan_deg=new_raan,
                        arg_periapsis_deg=new_arg_p,
                        true_anomaly_deg=new_nu,
                        target_type="FRAG",
                        age_days=100.0,
                        risk=0.02,
                        name=f"DYNAMIC-NEW-{replace_idx:02d}",
                    )

            # Rebuild observation with perturbed target configurations
            obs = self._build_observation()

        return obs, reward, terminated, truncated, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and trained planning policies under dynamic perturbations."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=12000.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument(
        "--scenario", type=str, default="medium",
        help="Scenario preset name or path to Celestrak JSON file."
    )
    parser.add_argument(
        "--output", type=str, default=r"results\dynamic_evaluation_summary.json"
    )
    return parser.parse_args()


def run_dynamic_policy(
    name: str,
    episodes: int,
    seed: int,
    targets: int,
    fuel: float,
    max_steps: int,
    model_path: str = "",
    scenario_generator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run dynamic evaluation rollouts of a named policy and return aggregate metrics."""
    rng = np.random.default_rng(seed)
    all_delta_v: list[float] = []
    all_cleared: list[int] = []
    all_fuel_remaining: list[float] = []
    full_clear_count = 0

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
        # Instantiate the dynamic environment
        env = DynamicOrbitDebrisEnv(
            scenario_generator=scenario_generator,
            seed=seed + i,
            target_count=targets,
            fuel_budget=fuel,
            max_steps=max_steps,
            drift_std=0.05,
            replacement_prob=0.05,
        )
        obs, _ = env.reset(seed=seed + i)
        done = False
        info: dict = {}

        import time
        start_time = time.perf_counter()
        
        while not done:
            if name == "random":
                action = random_policy(env, rng)
            elif name == "nearest":
                action = nearest_neighbor_policy(env)
            elif name == "risk_weighted":
                action = risk_weighted_policy(env)
            elif name == "branch_and_bound":
                action = branch_and_bound_policy(env)
            elif name == "ppo":
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
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

    avg_cleared = float(np.mean(all_cleared))
    accuracy = avg_cleared / targets

    # Completion rates
    completion_rate = sum(1 for c in all_cleared if c >= 1) / episodes
    
    # Fuel efficiency per target
    cleared_episodes = [c for c in all_cleared if c > 0]
    dv_for_cleared = [all_delta_v[k] for k, c in enumerate(all_cleared) if c > 0]
    fuel_per_target = (
        float(np.mean([dv / c for dv, c in zip(dv_for_cleared, cleared_episodes)]))
        if len(cleared_episodes) > 0 else float('inf')
    )

    return {
        "avg_delta_v": float(np.mean(all_delta_v)),
        "std_delta_v": float(np.std(all_delta_v)),
        "avg_cleared": avg_cleared,
        "avg_fuel_remaining": float(np.mean(all_fuel_remaining)),
        "completion_rate": completion_rate,
        "fuel_per_target": fuel_per_target,
        "accuracy": accuracy,
        "episodes": episodes,
        "all_delta_v": all_delta_v,
        "all_cleared": [int(c) for c in all_cleared],
    }


def main() -> None:
    args = parse_args()
    results: dict[str, dict] = {}

    scenario_path = Path(args.scenario)
    if scenario_path.exists() and scenario_path.suffix.lower() == '.json':
        scenario_generator = functools.partial(from_celestrak_json, filepath=scenario_path)
    else:
        scenario_generator = SCENARIO_PRESETS.get(args.scenario, SCENARIO_PRESETS["medium"])

    policy_names = ["random", "nearest", "risk_weighted", "branch_and_bound"]
    if args.model_path:
        policy_names.append("ppo")

    for idx, name in enumerate(policy_names):
        print(f"Evaluating {name} policy under dynamic perturbations ({args.episodes} episodes)...")
        results[name] = run_dynamic_policy(
            name=name,
            episodes=args.episodes,
            seed=args.seed + idx * 1000,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
            model_path=args.model_path,
            scenario_generator=scenario_generator,
        )

    # Print dynamic comparison table
    print()
    print("=" * 70)
    print(" DYNAMIC PERTURBATION EVALUATION RESULTS")
    print("=" * 70)
    print(
        f"{'Policy':<18} {'Delta-V (m/s)':>15} {'Cleared':>10} {'Acc %':>10} {'Fuel/Target':>15}"
    )
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<18} {m['avg_delta_v']:>15.1f} {m['avg_cleared']:>10.2f} "
            f"{100.0 * m['accuracy']:>9.1f}% {m['fuel_per_target']:>15.1f}"
        )
    print("=" * 70)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {}
    for name, m in results.items():
        save_data[name] = {
            k: v
            for k, v in m.items()
            if k not in ("all_delta_v", "all_cleared")
        }
        
    output_path.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    print(f"Dynamic evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
