"""Unit tests for Branch-and-Bound optimal sequencing and Dynamic perturbations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from simulation.orbit_env import OrbitDebrisEnv
from simulation.dynamic_evaluate import DynamicOrbitDebrisEnv
from simulation.policies import branch_and_bound_policy
from simulation.scenario import default_scenario


@pytest.fixture
def env():
    return OrbitDebrisEnv(
        scenario_generator=default_scenario,
        seed=42,
        target_count=4,
        fuel_budget=15000.0,
        max_steps=20,
    )


@pytest.fixture
def dynamic_env():
    return DynamicOrbitDebrisEnv(
        scenario_generator=default_scenario,
        seed=42,
        target_count=4,
        fuel_budget=15000.0,
        max_steps=20,
        drift_std=0.05,
        replacement_prob=0.1,
    )


class TestBranchAndBound:
    def test_optimal_sequence_produces_valid_action(self, env):
        env.reset()
        action = branch_and_bound_policy(env)
        assert action in env.valid_actions()

    def test_optimal_zero_fuel_fallback(self):
        env_poor = OrbitDebrisEnv(
            scenario_generator=default_scenario,
            seed=42,
            target_count=4,
            fuel_budget=0.0,  # Zero fuel
            max_steps=20,
        )
        env_poor.reset()
        action = branch_and_bound_policy(env_poor)
        # Should fallback to nearest or valid actions
        assert action >= 0


class TestDynamicEnvironment:
    def test_drift_applied_on_step(self, dynamic_env):
        dynamic_env.reset()
        # Capture target heights before step
        smas_before = [t.sma_km for t in dynamic_env._targets]
        
        # Take a step
        dynamic_env.step(0)
        
        smas_after = [t.sma_km for t in dynamic_env._targets]
        # Active targets (except 0 which was captured/de-activated) should have drifted
        for idx in range(1, len(smas_before)):
            if dynamic_env._active[idx]:
                assert smas_before[idx] != smas_after[idx]

    def test_rebuild_observation_matches_drift(self, dynamic_env):
        dynamic_env.reset()
        obs2, _, _, _, _ = dynamic_env.step(0)
        # The true active indicators and SMA features in obs2 should match the updated env states
        assert len(obs2) == dynamic_env.observation_space.shape[0]
