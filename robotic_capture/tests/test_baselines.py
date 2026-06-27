"""Self-checks for the model-based baselines and the evaluation harness.

Key invariants:
- ARC actually detumbles (reduces target spin far more than doing nothing).
- ARC is inertia-agnostic by construction (its constructor takes no target inertia).
- The headline trend holds: induced base disturbance grows with target inertia.
- The bootstrap CI and rollout metrics are well-formed.
"""

import inspect

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv  # noqa: E402
from robotic_capture.control.baselines import ARCController, ZeroController  # noqa: E402
from robotic_capture.eval import rollout, bootstrap_ci  # noqa: E402


@pytest.fixture(scope="module")
def env():
    return FreeFlyerCaptureEnv(domain_randomize=False, detumble_tol=0.05)


def test_arc_constructor_is_inertia_agnostic():
    """ARC must not accept any target inertia/mass parameter (its whole premise)."""
    params = set(inspect.signature(ARCController.__init__).parameters)
    assert not (params & {"nominal_inertia", "inertia", "target_mass", "mass"})


def test_arc_detumbles_more_than_zero(env):
    z = rollout(env, ZeroController(), inertia_scale=1.0, spin=0.3, seed=0, detumble_tol=0.05)
    a = rollout(env, ARCController(env.model), inertia_scale=1.0, spin=0.3, seed=0, detumble_tol=0.05)
    assert a["spin_reduction_pct"] > 50.0, "ARC failed to detumble"
    assert a["spin_reduction_pct"] > z["spin_reduction_pct"] + 30.0, "ARC no better than doing nothing"


def test_base_disturbance_grows_with_inertia(env):
    """The headline result: heavier targets dump more momentum into the chaser base."""
    arc = ARCController(env.model)
    light = rollout(env, arc, inertia_scale=0.5, spin=0.3, seed=0, detumble_tol=0.05)
    heavy = rollout(env, arc, inertia_scale=2.0, spin=0.3, seed=0, detumble_tol=0.05)
    assert heavy["base_disturb_final"] > light["base_disturb_final"], \
        "base disturbance should increase with target inertia"


def test_bootstrap_ci_brackets_mean():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000, rng=np.random.default_rng(0))
    assert lo <= mean <= hi
    assert mean == pytest.approx(3.0)
