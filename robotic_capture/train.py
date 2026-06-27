"""Train a SAC detumbling policy on the free-floating capture environment.

Two configurations support the paper's ablation:

* ``--domain-randomize`` (default on): target mass/inertia randomized 0.5x-2x each
  episode -- the robustness condition.
* ``--no-domain-randomize``: fixed nominal inertia -- the ablation that should collapse
  out-of-distribution robustness.

The policy observes kinematics only (no contact-force/inertia in the observation), so a
trained policy is "sensing-light" relative to the force-feedback ARC baseline. The reward
(defined in the env) penalises target spin, end-effector distance, **induced base angular
velocity** (the headline metric), and control effort.

Usage (from repo root, with the project .venv):

    python -m robotic_capture.train --timesteps 150000 --out runs/sac_dr
    python -m robotic_capture.train --no-domain-randomize --out runs/sac_nodr
    python -m robotic_capture.train --timesteps 2000 --out runs/smoke   # quick wiring check
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv


def make_env(domain_randomize: bool, max_steps: int, seed: int) -> Monitor:
    env = FreeFlyerCaptureEnv(domain_randomize=domain_randomize, max_steps=max_steps)
    env.reset(seed=seed)
    return Monitor(env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--domain-randomize", dest="dr", action="store_true", default=True)
    p.add_argument("--no-domain-randomize", dest="dr", action="store_false")
    p.add_argument("--out", type=str, default="robotic_capture/runs/sac_dr")
    args = p.parse_args()

    env = make_env(args.dr, args.max_steps, args.seed)
    # Small MLP: the state is low-dimensional (32-D). SAC is sample-efficient for
    # continuous torque control. Defaults are deliberately close to SB3 stock to keep
    # the result reproducible and the script short.
    model = SAC(
        "MlpPolicy", env, seed=args.seed, verbose=1,
        learning_rate=3e-4, buffer_size=200_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=1000, policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved policy to {out}.zip  (domain_randomize={args.dr})")


if __name__ == "__main__":
    main()
