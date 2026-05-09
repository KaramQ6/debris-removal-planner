from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario import MissionScenario, default_scenario


class OrbitDebrisEnv(gym.Env[np.ndarray, int]):
    """
    Simplified 2D debris-removal environment.

    Action: index of next debris target to intercept.
    Reward: high-risk clear bonus minus approximated transfer delta-v cost.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: MissionScenario | None = None,
        *,
        max_targets: int = 10,
        seed: int | None = None,
        target_count: int = 8,
        fuel_budget: float = 1200.0,
        max_steps: int = 50,
    ) -> None:
        super().__init__()
        if max_targets < 1:
            raise ValueError("max_targets must be >= 1")

        self.max_targets = max_targets
        self._seed = seed
        self._target_count = target_count
        self._fuel_budget = fuel_budget
        self._max_steps = max_steps
        self._scenario = scenario or default_scenario(
            target_count=target_count,
            fuel_budget=fuel_budget,
            max_steps=max_steps,
            seed=seed,
        )
        if len(self._scenario.targets) > self.max_targets:
            raise ValueError("Scenario target count exceeds max_targets")

        self.action_space = spaces.Discrete(self.max_targets)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 + self.max_targets * 3,), dtype=np.float32
        )
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self._targets = list(self._scenario.targets)
        self._active = np.ones(len(self._targets), dtype=bool)
        self._spacecraft_angle_deg = self._scenario.start_angle_deg
        self._fuel_remaining = self._scenario.fuel_budget
        self._steps = 0
        self._total_delta_v = 0.0
        self._cleared = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self._scenario = default_scenario(
                target_count=self._target_count,
                fuel_budget=self._fuel_budget,
                max_steps=self._max_steps,
                seed=seed,
            )
        self._reset_state()
        return self._build_observation(), self._build_info()

    @staticmethod
    def _angular_distance_deg(a: float, b: float) -> float:
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)

    @classmethod
    def _delta_v_cost(cls, origin_deg: float, target_deg: float) -> float:
        # Coarse transfer approximation for early-stage planning.
        return 20.0 + 1.5 * cls._angular_distance_deg(origin_deg, target_deg)

    def valid_actions(self) -> np.ndarray:
        return np.flatnonzero(self._active)

    def delta_v_to_target(self, action: int) -> float:
        if action < 0 or action >= len(self._targets):
            return float("inf")
        if not self._active[action]:
            return float("inf")
        return self._delta_v_cost(self._spacecraft_angle_deg, self._targets[action].angle_deg)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1

        reward = 0.0
        terminated = False

        if action < 0 or action >= len(self._targets) or not self._active[action]:
            reward -= 5.0
        else:
            target = self._targets[action]
            delta_v = self._delta_v_cost(self._spacecraft_angle_deg, target.angle_deg)

            if delta_v > self._fuel_remaining:
                reward -= 20.0
                terminated = True
                self._fuel_remaining = 0.0
            else:
                self._fuel_remaining -= delta_v
                self._spacecraft_angle_deg = target.angle_deg
                self._active[action] = False
                self._total_delta_v += delta_v
                self._cleared += 1
                reward += 10.0 + 10.0 * target.risk - 0.05 * delta_v

        if self._cleared == len(self._targets):
            terminated = True
            reward += 15.0

        if self._fuel_remaining <= 0.0:
            terminated = True

        truncated = self._steps >= self._scenario.max_steps and not terminated
        return self._build_observation(), reward, terminated, truncated, self._build_info()

    def _build_observation(self) -> np.ndarray:
        obs = np.full(self.observation_space.shape, -1.0, dtype=np.float32)

        angle_rad = np.deg2rad(self._spacecraft_angle_deg)
        obs[0] = float(np.cos(angle_rad))
        obs[1] = float(2.0 * (self._fuel_remaining / self._scenario.fuel_budget) - 1.0)

        for i in range(self.max_targets):
            offset = 2 + i * 3
            if i < len(self._targets):
                target = self._targets[i]
                obs[offset] = float(np.cos(np.deg2rad(target.angle_deg)))
                obs[offset + 1] = float(2.0 * target.risk - 1.0)
                obs[offset + 2] = 1.0 if self._active[i] else -1.0

        return obs

    def _build_info(self) -> dict[str, Any]:
        return {
            "cleared": self._cleared,
            "total_targets": len(self._targets),
            "fuel_remaining": self._fuel_remaining,
            "total_delta_v": self._total_delta_v,
            "remaining_targets": int(np.count_nonzero(self._active)),
            "steps": self._steps,
        }

