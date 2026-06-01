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


def branch_and_bound_policy(env: OrbitDebrisEnv) -> int:
    """Exactly solve the remaining target sequence using recursive depth-first branch-and-bound.

    Returns the first action of the optimal sequence that clears the maximum targets
    with minimum fuel consumption.
    """
    targets = env._targets
    active = env._active

    best_sequence: list[int] = []
    best_cleared_count = -1
    best_total_dv = float("inf")

    def search(
        current_sma: float,
        current_ecc: float,
        current_inc: float,
        current_raan: float,
        current_arg_p: float,
        remaining_fuel: float,
        cleared_indices: list[int],
        current_dv: float,
    ) -> None:
        nonlocal best_sequence, best_cleared_count, best_total_dv

        cleared_count = len(cleared_indices)
        # Keep track of the sequence that clears the most targets,
        # and among those, consumes the least total delta-v.
        if (cleared_count > best_cleared_count) or (
            cleared_count == best_cleared_count and current_dv < best_total_dv
        ):
            best_cleared_count = cleared_count
            best_total_dv = current_dv
            best_sequence = list(cleared_indices)

        if cleared_count == len(targets):
            return

        for idx in range(len(targets)):
            if not active[idx] or idx in cleared_indices:
                continue

            t = targets[idx]
            # env._delta_v_cost returns cost in m/s
            dv = env._delta_v_cost(
                current_sma,
                current_ecc,
                current_inc,
                current_raan,
                current_arg_p,
                t.sma_km,
                t.eccentricity,
                t.inclination_deg,
                t.raan_deg,
                t.arg_periapsis_deg,
            )

            if dv <= remaining_fuel:
                # Prune branch if remaining targets + current cleared count cannot exceed best count
                remaining_active_count = len(targets) - cleared_count - 1
                if cleared_count + 1 + remaining_active_count < best_cleared_count:
                    continue
                if (
                    cleared_count + 1 + remaining_active_count == best_cleared_count
                    and current_dv + dv >= best_total_dv
                ):
                    continue

                search(
                    t.sma_km,
                    t.eccentricity,
                    t.inclination_deg,
                    t.raan_deg,
                    t.arg_periapsis_deg,
                    remaining_fuel - dv,
                    cleared_indices + [idx],
                    current_dv + dv,
                )

    # Run the recursive search from the chaser spacecraft's current state
    search(
        env._sp_sma,
        env._sp_ecc,
        env._sp_inc,
        env._sp_raan,
        env._sp_arg_p,
        env._fuel_remaining,
        [],
        0.0,
    )

    if len(best_sequence) > 0:
        return int(best_sequence[0])
    return int(nearest_neighbor_policy(env))

