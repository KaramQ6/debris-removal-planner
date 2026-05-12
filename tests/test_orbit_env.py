"""Tests for the OrbitDebrisEnv Gymnasium environment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulation.orbit_env import OrbitDebrisEnv
from simulation.scenario import default_scenario


@pytest.fixture
def env():
    return OrbitDebrisEnv(
        scenario_generator=default_scenario,
        seed=42,
        target_count=5,
        fuel_budget=20_000.0,
        max_steps=30,
    )


class TestDeltaVCost:
    def test_zero_cost_same_orbit(self):
        cost = OrbitDebrisEnv._delta_v_cost(
            7000.0, 0.0, 30.0, 100.0, 0.0,
            7000.0, 0.0, 30.0, 100.0, 0.0,
        )
        assert cost < 1.0  # numerical noise only

    def test_hohmann_low_altitude_change(self):
        """Raising SMA by ~100 km should cost ~50 m/s — a textbook Hohmann."""
        cost = OrbitDebrisEnv._delta_v_cost(
            7000.0, 0.0, 30.0, 0.0, 0.0,
            7100.0, 0.0, 30.0, 0.0, 0.0,
        )
        # Loose bracket — the env uses a generalized approximation.
        assert 20.0 < cost < 200.0

    def test_plane_change_dominates(self):
        """90-degree plane change should be expensive (>5 km/s in LEO)."""
        cost = OrbitDebrisEnv._delta_v_cost(
            7000.0, 0.0, 0.0, 0.0, 0.0,
            7000.0, 0.0, 90.0, 0.0, 0.0,
        )
        assert cost > 5000.0


class TestEnvironment:
    def test_observation_shape_matches_space(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_action_mask_marks_only_active(self, env):
        env.reset()
        mask = env.action_mask()
        assert mask.shape == (env.max_targets,)
        assert mask[: len(env._targets)].all()
        # Padded slots beyond actual targets must be False.
        assert not mask[len(env._targets):].any()

    def test_invalid_action_penalized_but_not_terminal(self, env):
        env.reset()
        # Clear target 0 first.
        env.step(0)
        # Now action 0 is invalid (already cleared) — should be penalized.
        _, reward, terminated, truncated, _ = env.step(0)
        assert reward < 0
        assert not terminated

    def test_fuel_exhaustion_terminates(self, env):
        """Step with insufficient fuel should terminate the episode."""
        tight = OrbitDebrisEnv(
            scenario_generator=default_scenario,
            seed=0,
            target_count=5,
            fuel_budget=1.0,  # absurdly low
            max_steps=30,
        )
        tight.reset()
        _, reward, terminated, _, _ = tight.step(0)
        assert terminated
        assert reward <= 0

    def test_clear_reward_positive_for_cheap_intercept(self, env):
        env.reset()
        # Pick the cheapest target — should net positive reward.
        valid = env.valid_actions()
        cheapest = min(valid, key=lambda a: env.delta_v_to_target(int(a)))
        _, reward, _, _, info = env.step(int(cheapest))
        assert reward > 0
        assert info["cleared"] == 1

    def test_full_clear_grants_completion_bonus(self):
        """Completing all targets must trigger the terminal bonus."""
        e = OrbitDebrisEnv(
            scenario_generator=default_scenario,
            seed=99,
            target_count=3,
            fuel_budget=50_000.0,
            max_steps=20,
        )
        e.reset()
        for _ in range(3):
            valid = e.valid_actions()
            assert len(valid) > 0
            cheapest = min(valid, key=lambda a: e.delta_v_to_target(int(a)))
            _, reward, terminated, _, info = e.step(int(cheapest))
        assert info["cleared"] == 3
        assert terminated
        # Completion bonus is +50 → final step reward should be well above 25.
        assert reward > 50.0

    def test_observation_normalized_range(self, env):
        obs, _ = env.reset()
        assert obs.min() >= -1.0 - 1e-6
        assert obs.max() <= 1.0 + 1e-6

    def test_trajectory_logged(self, env):
        env.reset()
        valid = env.valid_actions()
        env.step(int(valid[0]))
        _, _, _, _, info = env.step(int(env.valid_actions()[0]))
        assert len(info["trajectory"]) >= 1
        first = info["trajectory"][0]
        assert "delta_v" in first
        assert "target_id" in first
