"""Run a trained model on a scenario and save a 3D interactive visualization."""

import argparse
import functools
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from .orbit_env import OrbitDebrisEnv
from .scenario import SCENARIO_PRESETS, from_celestrak_json
from .visualize import generate_plotly_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="Last_30_day.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/demo_visual.html")
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if scenario_path.exists() and scenario_path.suffix.lower() == '.json':
        scenario_generator = functools.partial(from_celestrak_json, filepath=scenario_path)
    else:
        scenario_generator = SCENARIO_PRESETS.get(args.scenario, SCENARIO_PRESETS["medium"])

    env = OrbitDebrisEnv(scenario_generator=scenario_generator, seed=args.seed)
    model = MaskablePPO.load(args.model_path)

    obs, _ = env.reset(seed=args.seed)
    done = False
    
    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    print(f"Episode complete. Cleared: {info['cleared']}/{info['total_targets']}")
    print(f"Total Delta-V: {info['total_delta_v']:.2f} m/s")

    # generate_plotly_3d expects a dict with 'trajectory', 'policy', 'scenario', 'total_delta_v', 'targets'
    data = {
        "trajectory": info["trajectory"],
        "policy": "ppo",
        "scenario": args.scenario,
        "total_delta_v": info["total_delta_v"],
        "targets": [
            {
                "sma_km": t.sma_km,
                "eccentricity": t.eccentricity,
                "inclination_deg": t.inclination_deg,
                "raan_deg": t.raan_deg,
                "arg_periapsis_deg": t.arg_periapsis_deg,
                "true_anomaly_deg": t.true_anomaly_deg,
                "target_type": t.target_type,
                "age_days": t.age_days,
                "risk": t.risk,
                "name": t.name
            }
            for t in env._scenario.targets
        ]
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_plotly_3d(data, output_path)


if __name__ == "__main__":
    main()
