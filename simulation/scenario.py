"""Mission scenario definitions and debris target data structures."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def calculate_legend_risk(target_type: str, age_days: float) -> float:
    """Approximate time-dependent risk based on NASA LEGEND (Sept 2025 ODQN)."""
    if target_type == "S/C":
        # Spacecraft: max prob 0.044, single exponential decay
        return 0.044 * (1.0 - math.exp(-age_days / 3650.0))
    elif target_type == "R/B":
        # Rocket Body: max 0.019, 78% of explosions in first year
        base = 0.019
        early_factor = 0.78 * (1.0 - math.exp(-age_days / 100.0))
        late_factor = 0.22 * (1.0 - math.exp(-age_days / 2000.0))
        return base * (early_factor + late_factor)
    elif target_type == "SOZ":
        # SOZ Units (Proton): Gaussian peaking at 10.4 years (3810 days) with max 0.57
        peak = 3810.0
        sigma = 800.0
        return 0.57 * math.exp(-0.5 * ((age_days - peak) / sigma)**2)
    else:
        # Generic fragment
        return 0.01


@dataclass(frozen=True)
class DebrisTarget:
    """Single debris object in the 3D mission scenario."""
    target_id: int
    sma_km: float               # Semi-major axis (km)
    eccentricity: float         # Orbital eccentricity [0.0, 1.0)
    inclination_deg: float      # Orbital inclination (degrees)
    raan_deg: float             # Right Ascension of the Ascending Node (degrees)
    arg_periapsis_deg: float    # Argument of Periapsis (degrees)
    true_anomaly_deg: float     # Current position in the orbit (degrees)
    target_type: str            # "S/C", "R/B", "SOZ", "FRAG"
    age_days: float             # Age since launch/event
    risk: float                 # 0.0 – 1.0 collision/explosion probability
    name: str = ""


@dataclass(frozen=True)
class MissionScenario:
    """Full scenario specifying 3D targets, fuel, and constraints."""
    targets: tuple[DebrisTarget, ...]
    fuel_budget: float  # total delta-v available (m/s)
    max_steps: int
    start_sma_km: float
    start_eccentricity: float
    start_inclination_deg: float
    start_raan_deg: float
    start_arg_periapsis_deg: float
    start_true_anomaly_deg: float
    name: str = "default"


def _generate_cloud(
    rng: np.random.Generator,
    count: int,
    base_sma: float,
    base_ecc: float,
    base_inc: float,
    sma_var: float,
    ecc_var: float,
    inc_var: float,
    types: list[str],
    prefix: str,
) -> list[DebrisTarget]:
    
    smas = rng.normal(base_sma, sma_var, size=count)
    eccs = np.clip(rng.normal(base_ecc, ecc_var, size=count), 0.0, 0.99)
    incs = rng.normal(base_inc, inc_var, size=count)
    raans = rng.uniform(0.0, 360.0, size=count)
    args_p = rng.uniform(0.0, 360.0, size=count)
    anomalies = rng.uniform(0.0, 360.0, size=count)
    
    targets = []
    for i in range(count):
        t_type = rng.choice(types)
        # Randomize age between 100 and 5000 days
        age = float(rng.uniform(100, 5000))
        # Ensure minimum orbital height isn't inside Earth (6371 km)
        perigee = smas[i] * (1.0 - eccs[i])
        if perigee < 6471.0: # Ensure at least 100km altitude
            eccs[i] = (smas[i] - 6471.0) / smas[i]

        risk = calculate_legend_risk(t_type, age)
        # Add random scatter to risk to represent cross-sectional collision probability
        risk = np.clip(risk + float(rng.uniform(0.01, 0.1)), 0.0, 1.0)

        targets.append(DebrisTarget(
            target_id=0, # Will be set later
            sma_km=float(smas[i]),
            eccentricity=float(eccs[i]),
            inclination_deg=float(incs[i]),
            raan_deg=float(raans[i]),
            arg_periapsis_deg=float(args_p[i]),
            true_anomaly_deg=float(anomalies[i]),
            target_type=t_type,
            age_days=age,
            risk=float(risk),
            name=f"{prefix}-{i:04d}",
        ))
    return targets


def default_scenario(
    target_count: int = 8,
    fuel_budget: float = 6000.0,
    max_steps: int = 50,
    seed: int | None = None,
) -> MissionScenario:
    """Iridium-Cosmos realistic cloud (2009 event). Circular orbits at ~780km."""
    rng = np.random.default_rng(seed)
    # Iridium/Cosmos altitude ~778.6 km -> SMA ~7149.6 km, Inc ~72.5
    cloud = _generate_cloud(
        rng, count=target_count,
        base_sma=7149.6, base_ecc=0.001, base_inc=72.5,
        sma_var=10.0, ecc_var=0.005, inc_var=0.5,
        types=["FRAG", "S/C", "R/B"], prefix="IR-COS"
    )
    
    # Re-id
    final_targets = []
    for i, t in enumerate(cloud):
        final_targets.append(DebrisTarget(
            target_id=i, sma_km=t.sma_km, eccentricity=t.eccentricity,
            inclination_deg=t.inclination_deg, raan_deg=t.raan_deg,
            arg_periapsis_deg=t.arg_periapsis_deg, true_anomaly_deg=t.true_anomaly_deg,
            target_type=t.target_type, age_days=t.age_days, risk=t.risk, name=t.name
        ))

    return MissionScenario(
        targets=tuple(final_targets),
        fuel_budget=fuel_budget,
        max_steps=max_steps,
        start_sma_km=7149.6,
        start_eccentricity=0.0,
        start_inclination_deg=72.5,
        start_raan_deg=float(rng.uniform(0.0, 360.0)),
        start_arg_periapsis_deg=0.0,
        start_true_anomaly_deg=float(rng.uniform(0.0, 360.0)),
        name=f"iridium_cosmos_{target_count}t",
    )


def fengyun_scenario(seed: int | None = None, target_count: int = 8) -> MissionScenario:
    """Fengyun-1C ASAT event (2007). Altitude 850km, Sun-synchronous (98.8 deg)."""
    rng = np.random.default_rng(seed)
    cloud = _generate_cloud(
        rng, count=target_count,
        base_sma=7221.0, base_ecc=0.002, base_inc=98.8,
        sma_var=20.0, ecc_var=0.01, inc_var=0.2,
        types=["FRAG"], prefix="FY1C"
    )
    for i, t in enumerate(cloud):
        cloud[i] = object.__setattr__(t, 'target_id', i) or t
        
    # Python dataclass frozen hack, but safer to recreate:
    cloud = [DebrisTarget(i, t.sma_km, t.eccentricity, t.inclination_deg, t.raan_deg, 
                          t.arg_periapsis_deg, t.true_anomaly_deg, t.target_type, 
                          t.age_days, t.risk, t.name) for i, t in enumerate(cloud)]

    return MissionScenario(
        targets=tuple(cloud), fuel_budget=7000.0, max_steps=50,
        start_sma_km=7221.0, start_eccentricity=0.0, start_inclination_deg=98.8,
        start_raan_deg=float(rng.uniform(0, 360)), start_arg_periapsis_deg=0.0, start_true_anomaly_deg=0.0,
        name=f"fengyun_{target_count}t"
    )


def mission_shakti_scenario(seed: int | None = None, target_count: int = 8) -> MissionScenario:
    """Mission Shakti (2019). Highly eccentric orbits due to velocity kicks."""
    rng = np.random.default_rng(seed)
    # Target was 283km (SMA 6654 circular). Kicks pushed apogee to ~1730km (SMA = 7377, ecc = ~0.1)
    cloud = _generate_cloud(
        rng, count=target_count,
        base_sma=7377.0, base_ecc=0.098, base_inc=28.2, # ISRO microsat-R inclination is roughly 28 deg
        sma_var=100.0, ecc_var=0.03, inc_var=2.0,
        types=["FRAG"], prefix="SHAKTI"
    )
    cloud = [DebrisTarget(i, t.sma_km, t.eccentricity, t.inclination_deg, t.raan_deg, 
                          t.arg_periapsis_deg, t.true_anomaly_deg, t.target_type, 
                          t.age_days, t.risk, t.name) for i, t in enumerate(cloud)]

    return MissionScenario(
        targets=tuple(cloud), fuel_budget=8000.0, max_steps=60,
        start_sma_km=6654.0, start_eccentricity=0.0, start_inclination_deg=28.2,
        start_raan_deg=float(rng.uniform(0, 360)), start_arg_periapsis_deg=0.0, start_true_anomaly_deg=0.0,
        name=f"mission_shakti_{target_count}t"
    )


import json
from pathlib import Path

def from_celestrak_json(filepath: Path | str, target_count: int = 12, seed: int | None = None) -> MissionScenario:
    """Load orbital parameters from Celestrak JSON format."""
    rng = np.random.default_rng(seed)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Shuffle data to get random targets each time
    indices = rng.permutation(len(data))
    selected = [data[i] for i in indices[:min(target_count, len(data))]]
    
    mu = 398600.4418
    targets = []
    for i, obj in enumerate(selected):
        mean_motion = obj.get("MEAN_MOTION", 15.0)
        n_rad_s = mean_motion * 2.0 * math.pi / 86400.0
        sma = (mu / (n_rad_s ** 2)) ** (1/3)
        
        name = obj.get("OBJECT_NAME", f"OBJ-{i}")
        target_type = "S/C"
        if "DEB" in name:
            target_type = "FRAG"
        elif "R/B" in name:
            target_type = "R/B"
            
        targets.append(DebrisTarget(
            target_id=i,
            sma_km=sma,
            eccentricity=obj.get("ECCENTRICITY", 0.0),
            inclination_deg=obj.get("INCLINATION", 0.0),
            raan_deg=obj.get("RA_OF_ASC_NODE", 0.0),
            arg_periapsis_deg=obj.get("ARG_OF_PERICENTER", 0.0),
            true_anomaly_deg=obj.get("MEAN_ANOMALY", 0.0),
            target_type=target_type,
            age_days=float(rng.uniform(100, 5000)),
            risk=np.clip(calculate_legend_risk(target_type, 3000.0) + float(rng.uniform(0.01, 0.1)), 0.0, 1.0),
            name=name
        ))
    
    start_sma = targets[0].sma_km - 100.0
    start_inc = targets[0].inclination_deg
    start_raan = float(rng.uniform(0, 360))
    start_arg_p = 0.0
    start_nu = 0.0
    
    return MissionScenario(
        targets=tuple(targets), fuel_budget=8000.0, max_steps=50,
        start_sma_km=start_sma, start_eccentricity=0.0, start_inclination_deg=start_inc,
        start_raan_deg=start_raan, start_arg_periapsis_deg=start_arg_p, start_true_anomaly_deg=start_nu,
        name=f"json_custom_{len(targets)}t"
    )

def easy_scenario(seed: int | None = None) -> MissionScenario:
    return default_scenario(target_count=5, fuel_budget=8000.0, max_steps=30, seed=seed)

def medium_scenario(seed: int | None = None) -> MissionScenario:
    return fengyun_scenario(seed=seed, target_count=8)

def hard_scenario(seed: int | None = None) -> MissionScenario:
    return mission_shakti_scenario(seed=seed, target_count=12)

SCENARIO_PRESETS: dict[str, type[MissionScenario] | callable] = {
    "easy": easy_scenario,
    "medium": medium_scenario,
    "hard": hard_scenario,
    "iridium_cosmos": default_scenario,
    "fengyun": fengyun_scenario,
    "shakti": mission_shakti_scenario,
}
