from __future__ import annotations

import numpy as np

from .orbit_env import OrbitDebrisEnv


def random_policy(env: OrbitDebrisEnv, rng: np.random.Generator) -> int:
    actions = env.valid_actions()
    if len(actions) == 0:
        return 0
    return int(rng.choice(actions))


def nearest_neighbor_policy(env: OrbitDebrisEnv) -> int:
    actions = env.valid_actions()
    if len(actions) == 0:
        return 0
    best_action = min(actions, key=lambda a: env.delta_v_to_target(int(a)))
    return int(best_action)

