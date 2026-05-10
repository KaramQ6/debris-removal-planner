"""Mission scenario definitions and debris target data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DebrisTarget:
    """Single debris object in the mission scenario."""

    target_id: int
    angle_deg: float
    risk: float  # 0.0 – 1.0 collision probability indicator
    altitude_km: float = 400.0  # orbital altitude (LEO default)
    inclination_deg: float = 51.6  # orbital inclination (ISS-like default)
    name: str = ""


@dataclass(frozen=True)
class MissionScenario:
    """Full scenario specifying targets, fuel, and constraints."""

    targets: tuple[DebrisTarget, ...]
    fuel_budget: float  # total delta-v available (m/s)
    max_steps: int
    start_angle_deg: float
    name: str = "default"


def default_scenario(
    target_count: int = 8,
    fuel_budget: float = 1200.0,
    max_steps: int = 50,
    seed: int | None = None,
) -> MissionScenario:
    """Generate a random scenario with *target_count* debris objects."""
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 360.0, size=target_count)
    risks = rng.uniform(0.1, 1.0, size=target_count)
    altitudes = rng.uniform(350.0, 800.0, size=target_count)
    inclinations = rng.uniform(28.0, 98.0, size=target_count)

    targets = tuple(
        DebrisTarget(
            target_id=i,
            angle_deg=float(angles[i]),
            risk=float(risks[i]),
            altitude_km=float(altitudes[i]),
            inclination_deg=float(inclinations[i]),
            name=f"DBR-{i:04d}",
        )
        for i in range(target_count)
    )
    start_angle = float(rng.uniform(0.0, 360.0))
    return MissionScenario(
        targets=targets,
        fuel_budget=fuel_budget,
        max_steps=max_steps,
        start_angle_deg=start_angle,
        name=f"random_{target_count}t",
    )


# ---------------------------------------------------------------------------
# Scenario presets for curriculum learning / reproducible benchmarks
# ---------------------------------------------------------------------------

def easy_scenario(seed: int | None = None) -> MissionScenario:
    """5 targets, generous fuel budget — good for early training."""
    return default_scenario(target_count=5, fuel_budget=1500.0, max_steps=30, seed=seed)


def medium_scenario(seed: int | None = None) -> MissionScenario:
    """8 targets, standard budget — matches concept document baseline."""
    return default_scenario(target_count=8, fuel_budget=1200.0, max_steps=50, seed=seed)


def hard_scenario(seed: int | None = None) -> MissionScenario:
    """12 targets, tight fuel budget — stress test for trained policies."""
    return default_scenario(target_count=12, fuel_budget=1000.0, max_steps=60, seed=seed)


SCENARIO_PRESETS: dict[str, type[MissionScenario] | callable] = {
    "easy": easy_scenario,
    "medium": medium_scenario,
    "hard": hard_scenario,
}
