"""Self-checks for the free-floating capture environment.

The critical check is `test_free_floating_coupling`: from rest, applying arm torque must
induce base motion. If that fails, the free-floating dynamics (the whole point of the paper)
are broken.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv  # noqa: E402


def _make(**kw):
    return FreeFlyerCaptureEnv(domain_randomize=False, **kw)


def test_spaces_and_step():
    env = _make()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert env.observation_space.contains(obs2)
    assert np.isfinite(r)
    assert {"target_angvel_norm", "base_angvel_norm", "contact_force"} <= info.keys()


def test_free_floating_coupling():
    """Arm torque from rest must move the base (momentum conservation coupling)."""
    env = _make()
    env.reset(seed=1, options={"target_angvel": [0.0, 0.0, 0.0]})
    assert env._info()["base_angvel_norm"] < 1e-6  # truly at rest
    for _ in range(50):
        env.step([1.0, 0.0, 0.0])  # torque on joint 1 only
    assert env._info()["base_angvel_norm"] > 1e-4, "base did not react to arm motion"


def test_domain_randomization_varies_mass():
    env = FreeFlyerCaptureEnv(domain_randomize=True)
    masses = set()
    for s in range(8):
        _, info = env.reset(seed=s)
        masses.add(round(info["target_mass"], 3))
    assert len(masses) > 1, "domain randomization did not vary target mass"


def test_no_randomization_fixed_mass():
    env = _make()
    _, i0 = env.reset(seed=0)
    _, i1 = env.reset(seed=7)
    assert i0["target_mass"] == pytest.approx(i1["target_mass"])
