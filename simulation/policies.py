"""Hand-crafted baseline policies for debris-removal mission planning."""

from __future__ import annotations

import numpy as np

from .orbit_env import OrbitDebrisEnv


def random_policy(env: OrbitDebrisEnv, rng: np.random.Generator) -> int:
    """Select a random valid target."""
    actions = env.valid_actions()
    if len(actions) == 0:
        return 0
    return int(rng.choice(actions))


def nearest_neighbor_policy(env: OrbitDebrisEnv) -> int:
    """Always intercept the closest remaining target (greedy by distance)."""
    actions = env.valid_actions()
    if len(actions) == 0:
        return 0
    best_action = min(actions, key=lambda a: env.delta_v_to_target(int(a)))
    return int(best_action)


def risk_weighted_policy(env: OrbitDebrisEnv, *, distance_weight: float = 0.6) -> int:
    """Balance proximity and collision risk.

    Score = distance_weight * (1 - norm_cost) + (1 - distance_weight) * risk

    Higher scores are preferred.  Default *distance_weight=0.6* favours nearby
    targets while still giving meaningful priority to high-risk objects.
    """
    actions = env.valid_actions()
    if len(actions) == 0:
        return 0

    costs = np.array([env.delta_v_to_target(int(a)) for a in actions])
    max_cost = costs.max() if costs.max() > 0 else 1.0
    norm_costs = costs / max_cost  # 0 = closest, 1 = farthest

    risks = np.array([env._targets[int(a)].risk for a in actions])

    scores = distance_weight * (1.0 - norm_costs) + (1.0 - distance_weight) * risks
    return int(actions[np.argmax(scores)])
