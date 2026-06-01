"""Custom Gymnasium environment for 3D orbital debris removal mission planning."""

from __future__ import annotations

import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario import MissionScenario, default_scenario


class OrbitDebrisEnv(gym.Env[np.ndarray, int]):
    """
    Realistic 3D debris-removal environment with Eccentricity support.

    **Observation** (per slot, padded to ``max_targets``):
        Spacecraft (11 features): [cos(nu), sin(nu), cos(raan), sin(raan), cos(inc), sin(inc), cos(arg_p), sin(arg_p), sma_norm, eccentricity, fuel_fraction]
        Target N (12 features):   [cos(nu), sin(nu), cos(raan), sin(raan), cos(inc), sin(inc), cos(arg_p), sin(arg_p), sma_norm, eccentricity, risk, active]

    **Action**: index of next debris target to intercept.

    **Reward**: risk-weighted clear bonus − delta-v cost + completion bonus.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_generator: Callable[..., MissionScenario] | None = None,
        *,
        max_targets: int = 12,
        seed: int | None = None,
        target_count: int = 8,
        fuel_budget: float = 6000.0,
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
        self._scenario_generator = scenario_generator or default_scenario
        try:
            self._scenario = self._scenario_generator(
                target_count=target_count,
                fuel_budget=fuel_budget,
                max_steps=max_steps,
                seed=seed,
            )
        except TypeError:
            self._scenario = self._scenario_generator(
                target_count=target_count,
                seed=seed,
            )
        if len(self._scenario.targets) > self.max_targets:
            raise ValueError("Scenario target count exceeds max_targets")

        self.action_space = spaces.Discrete(self.max_targets)
        
        # 11 spacecraft features + 12 features per target slot
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(11 + self.max_targets * 12,),
            dtype=np.float32,
        )
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self._targets = list(self._scenario.targets)
        self._active = np.ones(len(self._targets), dtype=bool)
        
        self._sp_sma = self._scenario.start_sma_km
        self._sp_ecc = self._scenario.start_eccentricity
        self._sp_inc = self._scenario.start_inclination_deg
        self._sp_raan = self._scenario.start_raan_deg
        self._sp_arg_p = self._scenario.start_arg_periapsis_deg
        self._sp_nu = self._scenario.start_true_anomaly_deg
        
        self._fuel_remaining = self._scenario.fuel_budget
        self._steps = 0
        self._total_delta_v = 0.0
        self._cleared = 0
        self._trajectory: list[dict[str, Any]] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        new_seed = int(self.np_random.integers(0, 2**31))
        try:
            self._scenario = self._scenario_generator(
                target_count=self._target_count,
                fuel_budget=self._fuel_budget,
                max_steps=self._max_steps,
                seed=new_seed,
            )
        except TypeError:
            # Fallback if generator doesn't accept fuel_budget/max_steps
            self._scenario = self._scenario_generator(
                target_count=self._target_count,
                seed=new_seed,
            )
        self._reset_state()
        return self._build_observation(), self._build_info()

    @classmethod
    def _delta_v_cost(
        cls, 
        sma1: float, ecc1: float, inc1: float, raan1: float, arg_p1: float,
        sma2: float, ecc2: float, inc2: float, raan2: float, arg_p2: float
    ) -> float:
        """
        Approximate Delta-V cost (in m/s) between two arbitrary eccentric 3D orbits.
        Uses a generalized formulation incorporating shape, size, and orientation changes.
        """
        mu = 398600.4418  # Earth standard gravitational parameter in km^3/s^2
        
        # 1. Delta-V for Coplanar size and shape change
        # Average velocity approximation
        v1 = math.sqrt(mu / sma1)
        v2 = math.sqrt(mu / sma2)
        
        # Standard Hohmann logic applied to average circular equivalents
        v_tx1 = math.sqrt(mu * (2/sma1 - 2/(sma1+sma2)))
        v_tx2 = math.sqrt(mu * (2/sma2 - 2/(sma1+sma2)))
        dv_size = abs(v_tx1 - v1) + abs(v2 - v_tx2)
        
        # Eccentricity change penalty (approximate)
        # DeltaV to change eccentricity keeping SMA constant requires velocity impulse at periapsis/apoapsis
        dv_ecc = 0.5 * (v1 + v2) * abs(ecc2 - ecc1)

        # 2. Delta-V for Orientation Change (Inc, RAAN)
        inc1_rad = math.radians(inc1)
        inc2_rad = math.radians(inc2)
        raan1_rad = math.radians(raan1)
        raan2_rad = math.radians(raan2)
        
        # Angle between orbital planes using spherical law of cosines
        cos_theta = (math.cos(inc1_rad) * math.cos(inc2_rad) +
                     math.sin(inc1_rad) * math.sin(inc2_rad) * math.cos(raan2_rad - raan1_rad))
        # Clip to handle floating point inaccuracies
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)
        
        # Plane change is cheapest at apoapsis (slowest point)
        r_apoapsis = max(sma1 * (1 + ecc1), sma2 * (1 + ecc2))
        v_apoapsis = math.sqrt(mu * (2/r_apoapsis - 1/max(sma1, sma2)))
        dv_plane = 2 * v_apoapsis * math.sin(theta / 2)
        
        # Argument of periapsis change (apsidal rotation)
        arg_p_diff = abs(math.radians((arg_p2 - arg_p1 + 180) % 360 - 180))
        # Apsidal rotation cost is proportional to eccentricity
        dv_apsidal = 2 * v1 * max(ecc1, ecc2) * math.sin(arg_p_diff / 2)

        total_dv_km_s = dv_size + dv_ecc + dv_plane + dv_apsidal
        return total_dv_km_s * 1000.0

    def valid_actions(self) -> np.ndarray:
        return np.flatnonzero(self._active)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_targets, dtype=bool)
        for i in range(len(self._targets)):
            if self._active[i] and self.delta_v_to_target(i) <= self._fuel_remaining:
                mask[i] = True
        if not np.any(mask):
            for i in range(len(self._targets)):
                if self._active[i]:
                    mask[i] = True
        return mask

    def action_masks(self) -> np.ndarray:
        return self.action_mask()

    def delta_v_to_target(self, action: int) -> float:
        if action < 0 or action >= len(self._targets) or not self._active[action]:
            return float("inf")
        t = self._targets[action]
        return self._delta_v_cost(
            self._sp_sma, self._sp_ecc, self._sp_inc, self._sp_raan, self._sp_arg_p,
            t.sma_km, t.eccentricity, t.inclination_deg, t.raan_deg, t.arg_periapsis_deg
        )

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1
        reward = 0.0
        terminated = False

        if action < 0 or action >= len(self._targets) or not self._active[action]:
            reward -= 5.0
        else:
            target = self._targets[action]
            delta_v = self.delta_v_to_target(action)

            if delta_v > self._fuel_remaining:
                # Smaller penalty: encourage exploration of expensive transfers
                # without panic-collapsing the policy.
                reward -= 5.0
                terminated = True
                self._fuel_remaining = 0.0
            else:
                self._fuel_remaining -= delta_v

                prev_sma = self._sp_sma
                prev_ecc = self._sp_ecc
                prev_inc = self._sp_inc
                prev_raan = self._sp_raan
                prev_arg_p = self._sp_arg_p

                self._sp_sma = target.sma_km
                self._sp_ecc = target.eccentricity
                self._sp_inc = target.inclination_deg
                self._sp_raan = target.raan_deg
                self._sp_arg_p = target.arg_periapsis_deg
                self._sp_nu = target.true_anomaly_deg

                self._active[action] = False
                self._total_delta_v += delta_v
                self._cleared += 1

                # Stronger clear bonus (justifies expensive plane-changes in
                # high-inclination clusters) + LEGEND risk weighting.
                # Fuel cost scaled so a max-budget burn (~12000 m/s) costs ~12.
                reward += 25.0 + 30.0 * target.risk - 0.001 * delta_v

                self._trajectory.append(
                    {
                        "step": self._steps,
                        "from_sma": prev_sma,
                        "from_ecc": prev_ecc,
                        "from_inc": prev_inc,
                        "from_raan": prev_raan,
                        "from_arg_p": prev_arg_p,
                        "to_sma": target.sma_km,
                        "to_ecc": target.eccentricity,
                        "to_inc": target.inclination_deg,
                        "to_raan": target.raan_deg,
                        "to_arg_p": target.arg_periapsis_deg,
                        "target_id": target.target_id,
                        "target_name": target.name,
                        "delta_v": delta_v,
                        "risk": target.risk,
                        "fuel_after": self._fuel_remaining,
                    }
                )

        if self._cleared == len(self._targets):
            terminated = True
            reward += 50.0
            fuel_fraction = self._fuel_remaining / self._scenario.fuel_budget
            reward += 30.0 * fuel_fraction

        if self._fuel_remaining <= 0.0:
            terminated = True

        truncated = self._steps >= self._scenario.max_steps and not terminated
        return (
            self._build_observation(),
            reward,
            terminated,
            truncated,
            self._build_info(),
        )

    def _normalize_sma(self, sma: float) -> float:
        """Normalize SMA roughly around LEO bounds (6000-8000 km) to [-1, 1]."""
        return max(-1.0, min(1.0, (sma - 7000.0) / 1000.0))

    def _build_observation(self) -> np.ndarray:
        obs = np.full(self.observation_space.shape, -1.0, dtype=np.float32)

        # Spacecraft state: [cos(nu), sin(nu), cos(raan), sin(raan), cos(inc), sin(inc), cos(arg_p), sin(arg_p), sma_norm, ecc, fuel_fraction]
        nu_rad = math.radians(self._sp_nu)
        raan_rad = math.radians(self._sp_raan)
        inc_rad = math.radians(self._sp_inc)
        arg_p_rad = math.radians(self._sp_arg_p)
        
        obs[0] = float(math.cos(nu_rad))
        obs[1] = float(math.sin(nu_rad))
        obs[2] = float(math.cos(raan_rad))
        obs[3] = float(math.sin(raan_rad))
        obs[4] = float(math.cos(inc_rad))
        obs[5] = float(math.sin(inc_rad))
        obs[6] = float(math.cos(arg_p_rad))
        obs[7] = float(math.sin(arg_p_rad))
        obs[8] = self._normalize_sma(self._sp_sma)
        obs[9] = float(self._sp_ecc)
        if self._scenario.fuel_budget > 0.0:
            obs[10] = float(2.0 * (self._fuel_remaining / self._scenario.fuel_budget) - 1.0)
        else:
            obs[10] = -1.0


        # Per-target features: [cos(nu), sin(nu), cos(raan), sin(raan), cos(inc), sin(inc), cos(arg_p), sin(arg_p), sma_norm, ecc, risk, active]
        for i in range(self.max_targets):
            offset = 11 + i * 12
            if i < len(self._targets):
                t = self._targets[i]
                t_nu = math.radians(t.true_anomaly_deg)
                t_raan = math.radians(t.raan_deg)
                t_inc = math.radians(t.inclination_deg)
                t_arg_p = math.radians(t.arg_periapsis_deg)
                
                obs[offset] = float(math.cos(t_nu))
                obs[offset + 1] = float(math.sin(t_nu))
                obs[offset + 2] = float(math.cos(t_raan))
                obs[offset + 3] = float(math.sin(t_raan))
                obs[offset + 4] = float(math.cos(t_inc))
                obs[offset + 5] = float(math.sin(t_inc))
                obs[offset + 6] = float(math.cos(t_arg_p))
                obs[offset + 7] = float(math.sin(t_arg_p))
                obs[offset + 8] = self._normalize_sma(t.sma_km)
                obs[offset + 9] = float(t.eccentricity)
                obs[offset + 10] = float(2.0 * t.risk - 1.0)
                obs[offset + 11] = 1.0 if self._active[i] else -1.0

        return obs

    def _build_info(self) -> dict[str, Any]:
        return {
            "cleared": self._cleared,
            "total_targets": len(self._targets),
            "fuel_remaining": self._fuel_remaining,
            "total_delta_v": self._total_delta_v,
            "remaining_targets": int(np.count_nonzero(self._active)),
            "steps": self._steps,
            "trajectory": list(self._trajectory),
        }
