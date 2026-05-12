"""Tests for scenario generation and LEGEND risk modeling."""

from __future__ import annotations

import math

import pytest

from simulation.scenario import (
    SCENARIO_PRESETS,
    calculate_legend_risk,
    curriculum_scenario,
    default_scenario,
    fengyun_scenario,
    mission_shakti_scenario,
)


class TestLegendRisk:
    def test_spacecraft_grows_with_age(self):
        early = calculate_legend_risk("S/C", age_days=100.0)
        late = calculate_legend_risk("S/C", age_days=10_000.0)
        assert late > early
        assert late <= 0.044

    def test_rocket_body_capped(self):
        late = calculate_legend_risk("R/B", age_days=10_000.0)
        assert late <= 0.019

    def test_soz_peaks_near_decade(self):
        peak = calculate_legend_risk("SOZ", age_days=3810.0)
        before = calculate_legend_risk("SOZ", age_days=1000.0)
        after = calculate_legend_risk("SOZ", age_days=8000.0)
        assert peak > before
        assert peak > after
        assert math.isclose(peak, 0.57, rel_tol=1e-3)

    def test_unknown_type_returns_baseline(self):
        assert calculate_legend_risk("FRAG", age_days=1000.0) == 0.01


class TestScenarios:
    @pytest.mark.parametrize("name", list(SCENARIO_PRESETS.keys()))
    def test_preset_callable(self, name):
        scenario = SCENARIO_PRESETS[name](seed=42)
        assert len(scenario.targets) > 0
        assert scenario.fuel_budget > 0
        assert scenario.max_steps > 0

    def test_default_scenario_uses_iridium_geometry(self):
        s = default_scenario(target_count=8, seed=0)
        assert len(s.targets) == 8
        assert math.isclose(s.start_sma_km, 7149.6, abs_tol=1.0)
        assert math.isclose(s.start_inclination_deg, 72.5, abs_tol=0.1)

    def test_fengyun_scenario_uses_sso_inclination(self):
        s = fengyun_scenario(seed=1)
        assert math.isclose(s.start_inclination_deg, 98.8, abs_tol=0.1)

    def test_shakti_scenario_has_eccentric_targets(self):
        s = mission_shakti_scenario(seed=2)
        avg_ecc = sum(t.eccentricity for t in s.targets) / len(s.targets)
        assert avg_ecc > 0.05

    def test_perigee_above_atmosphere(self):
        """Generated targets must not pass through Earth's atmosphere."""
        for name in ("iridium_cosmos", "fengyun", "shakti"):
            s = SCENARIO_PRESETS[name](seed=7)
            for t in s.targets:
                perigee = t.sma_km * (1.0 - t.eccentricity)
                assert perigee > 6471.0, f"{name}: perigee {perigee} below 100 km altitude"

    def test_curriculum_returns_valid_scenario(self):
        """All curriculum branches should produce usable scenarios."""
        for seed in range(20):
            s = curriculum_scenario(seed=seed)
            assert len(s.targets) > 0
            assert s.fuel_budget > 0

    def test_reproducible_with_seed(self):
        a = default_scenario(seed=123)
        b = default_scenario(seed=123)
        assert len(a.targets) == len(b.targets)
        for ta, tb in zip(a.targets, b.targets):
            assert math.isclose(ta.sma_km, tb.sma_km)
            assert math.isclose(ta.inclination_deg, tb.inclination_deg)
