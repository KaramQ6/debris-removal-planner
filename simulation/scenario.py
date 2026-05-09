from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DebrisTarget:
    target_id: int
    angle_deg: float
    risk: float


@dataclass(frozen=True)
class MissionScenario:
    targets: tuple[DebrisTarget, ...]
    fuel_budget: float
    max_steps: int
    start_angle_deg: float


def default_scenario(
    target_count: int = 8,
    fuel_budget: float = 1200.0,
    max_steps: int = 50,
    seed: int | None = None,
) -> MissionScenario:
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 360.0, size=target_count)
    risks = rng.uniform(0.1, 1.0, size=target_count)
    targets = tuple(
        DebrisTarget(target_id=i, angle_deg=float(angles[i]), risk=float(risks[i]))
        for i in range(target_count)
    )
    start_angle = float(rng.uniform(0.0, 360.0))
    return MissionScenario(
        targets=targets,
        fuel_budget=fuel_budget,
        max_steps=max_steps,
        start_angle_deg=start_angle,
    )

