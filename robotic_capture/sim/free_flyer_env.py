"""Free-floating space-manipulator capture & detumbling environment (Paper 2).

A 3-DOF arm on a 6-DOF free-floating chaser base must reach a tumbling, non-cooperative
target and detumble it. Gravity is zero (orbital free-fall), so arm motion reacts back on the
base -- the dynamic coupling that defines space robotics. Two disturbance axes are randomized at
reset: target mass/inertia (the inertia *parity* check) and a constant chaser-side base torque
(``base_disturbance``) standing in for ACS/thruster noise -- the paper's headline is that a
base-penalised policy *rejects* the latter where ARC's force-tracking law does not.

This is the simulation scaffold for the contribution in ``docs/contribution.md``: a planar 3-DOF
arm and a diagonal-inertia rigid target -- intentionally minimal so the control/MDP and reward can
be validated before scaling up fidelity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import mujoco
except ImportError:  # pragma: no cover - exercised only without the optional dep
    mujoco = None

_MODEL_PATH = Path(__file__).parent / "assets" / "space_manipulator.xml"

# Arm "ready" joint angles (rad) for j1,j2,j3 -- bent elbow, end-effector near the
# grasp face. Tuned to the target position in the MJCF (see reset()).
_READY_POSE = np.array([0.8, -1.6, 0.8])


class FreeFlyerCaptureEnv(gym.Env):
    """Gymnasium env for contact-rich detumbling with a free-floating manipulator.

    Action: 3 normalized joint torques in ``[-1, 1]`` (scaled to the MJCF ``ctrlrange``).
    Observation: base pose+twist, joint state, target relative pose + angular velocity,
    and the end-effector-to-grasp vector (32-D).
    Reward (scaffold, detumbling-oriented): penalizes target spin, end-effector distance,
    induced base angular velocity, and control effort, with a terminal bonus once detumbled.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        *,
        domain_randomize: bool = True,
        max_steps: int = 500,
        frame_skip: int = 5,
        tumble_max: float = 0.5,
        detumble_tol: float = 0.02,
        base_disturbance: float = 0.0,
        w_base: float = 0.3,
        seed: int | None = None,
    ) -> None:
        if mujoco is None:
            raise RuntimeError(
                "mujoco is not installed. Run: pip install -r robotic_capture/requirements.txt"
            )
        super().__init__()
        # Load via string, not path: MuJoCo's C++ file opener mishandles non-ASCII
        # (e.g. Arabic) paths on Windows; Python reads the unicode path fine.
        self.model = mujoco.MjModel.from_xml_string(_MODEL_PATH.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)

        self.domain_randomize = domain_randomize
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.tumble_max = tumble_max          # rad/s, per-axis initial spin bound
        self.detumble_tol = detumble_tol      # rad/s, "detumbled" threshold
        self.base_disturbance = base_disturbance  # N*m, max ACS/thruster torque on the base (0 = off)
        self._dist_torque = np.zeros(3)       # per-episode disturbance wrench (set in reset)

        # reward weights. w_base is the base-attitude penalty -- the disturbance-rejection
        # channel; the ablation (criterion 4) sets it to 0 to isolate that mechanism.
        self.w_tumble, self.w_dist, self.w_ctrl = 1.0, 0.5, 0.01
        self.w_base = w_base

        self.ctrl_max = float(self.model.actuator_ctrlrange[:, 1].max())

        # cache addresses (robust across mujoco versions)
        self._base_dof = self._dof_adr("base_free")
        self._base_bid = self._body_id("base")
        self._tgt_dof = self._dof_adr("target_free")
        self._tgt_qpos = self._qpos_adr("target_free")
        self._ee_sid = self._site_id("ee")
        self._grasp_sid = self._site_id("grasp")
        self._tgt_bid = self._body_id("target")
        self._nominal_mass = float(self.model.body_mass[self._tgt_bid])
        self._nominal_inertia = self.model.body_inertia[self._tgt_bid].copy()

        n_act = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        obs_dim = self._obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._steps = 0
        if seed is not None:
            self.reset(seed=seed)

    # -- address helpers --------------------------------------------------
    def _dof_adr(self, joint: str) -> int:
        return int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)])

    def _qpos_adr(self, joint: str) -> int:
        return int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)])

    def _site_id(self, site: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)

    def _body_id(self, body: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)

    # -- core API ---------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Domain randomization over target inertia (the inertia parity check).
        # ponytail: scales the diagonal principal inertia only; off-diagonal/products
        # of inertia are a future-fidelity upgrade, not needed to validate the MDP.
        # An explicit ``inertia_scale`` in ``options`` overrides both (used by the
        # evaluation sweep to test a controller at a *fixed*, known scale).
        if options is not None and "inertia_scale" in options:
            scale = float(options["inertia_scale"])
        elif self.domain_randomize:
            scale = float(self.np_random.uniform(0.5, 2.0))
        else:
            scale = 1.0
        self._inertia_scale = scale
        self.model.body_mass[self._tgt_bid] = self._nominal_mass * scale
        self.model.body_inertia[self._tgt_bid] = self._nominal_inertia * scale
        # CRITICAL: MuJoCo caches inertial quantities (cinert) derived from body_mass/
        # body_inertia; mutating them at runtime does NOT reach the mass matrix until
        # mj_setConst recomputes the cache. Without this, domain randomization over
        # inertia is silently a no-op in the dynamics.
        mujoco.mj_setConst(self.model, self.data)

        # Arm "ready" pose: a bent elbow that places the end-effector a small clearance
        # from the grasp face (post-grasp scope) and -- unlike a fully extended arm --
        # gives the Jacobian leverage to apply in-plane contact forces.
        self.data.qpos[7:10] = _READY_POSE

        # Initial tumble (rad/s). The 3-DOF arm is planar (all hinges about z), so the
        # well-posed detumbling axis is z; default tumble is a z-spin of randomized
        # magnitude. ``target_angvel`` in options overrides with an explicit 3-vector.
        if options is not None and "target_angvel" in options:
            omega = np.asarray(options["target_angvel"], dtype=float)
        else:
            omega = np.array([0.0, 0.0, self.np_random.uniform(-self.tumble_max, self.tumble_max)])
        self.data.qvel[self._tgt_dof + 3 : self._tgt_dof + 6] = omega

        # Chaser-side base-attitude disturbance (proxy for unmodeled ACS/thruster noise):
        # a constant external torque on the base about the planar detumbling axis (z),
        # random sign per episode. This is the headline gap -- ARC's force-tracking law has
        # no term that responds to it, so the base spins up; the base-penalised policy must
        # reject it via arm reaction. Magnitude: fixed by ``base_disturbance`` in options
        # (evaluation sweep), domain-randomized in [0, base_disturbance] during training,
        # or the constructor default otherwise.
        if options is not None and "base_disturbance" in options:
            mag = float(options["base_disturbance"])
        elif self.domain_randomize and self.base_disturbance > 0.0:
            mag = float(self.np_random.uniform(0.0, self.base_disturbance))
        else:
            mag = self.base_disturbance
        sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        self._dist_torque = np.array([0.0, 0.0, sign * mag])

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.data.ctrl[:] = a * self.ctrl_max
        # Persistent base disturbance torque (sampled per episode in reset()).
        self.data.xfrc_applied[self._base_bid, 3:6] = self._dist_torque
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._steps += 1

        info = self._info()
        reward = (
            -self.w_tumble * info["target_angvel_norm"]
            - self.w_dist * info["ee_grasp_dist"]
            - self.w_base * info["base_angvel_norm"]
            - self.w_ctrl * float(np.sum(a**2))
        )
        terminated = bool(info["target_angvel_norm"] < self.detumble_tol)
        if terminated:
            reward += 100.0
        truncated = bool(self._steps >= self.max_steps)
        return self._obs(), float(reward), terminated, truncated, info

    # -- observation / info ----------------------------------------------
    def _obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        base_pos = qpos[0:3]
        base_quat = qpos[3:7]
        base_vel = qvel[self._base_dof : self._base_dof + 6]
        joint_qpos = qpos[7:10]
        joint_qvel = qvel[6:9]
        tgt_pos = qpos[self._tgt_qpos : self._tgt_qpos + 3]
        tgt_quat = qpos[self._tgt_qpos + 3 : self._tgt_qpos + 7]
        tgt_angvel = qvel[self._tgt_dof + 3 : self._tgt_dof + 6]
        ee_to_grasp = self.data.site_xpos[self._grasp_sid] - self.data.site_xpos[self._ee_sid]
        return np.concatenate(
            [
                base_pos, base_quat, base_vel,        # 13
                joint_qpos, joint_qvel,               # 6
                tgt_pos - base_pos, tgt_quat, tgt_angvel,  # 10
                ee_to_grasp,                          # 3
            ]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        tgt_angvel = self.data.qvel[self._tgt_dof + 3 : self._tgt_dof + 6]
        base_angvel = self.data.qvel[self._base_dof + 3 : self._base_dof + 6]
        ee_to_grasp = self.data.site_xpos[self._grasp_sid] - self.data.site_xpos[self._ee_sid]
        return {
            "target_angvel_norm": float(np.linalg.norm(tgt_angvel)),
            "base_angvel_norm": float(np.linalg.norm(base_angvel)),
            "ee_grasp_dist": float(np.linalg.norm(ee_to_grasp)),
            "contact_force": self._net_contact_force(),
            "ncon": int(self.data.ncon),
            "target_mass": float(self.model.body_mass[self._tgt_bid]),
            "inertia_scale": float(getattr(self, "_inertia_scale", 1.0)),
            "base_disturbance_torque": float(np.linalg.norm(self._dist_torque)),
        }

    def _net_contact_force(self) -> float:
        """Sum of contact normal-force magnitudes (N) over all active contacts."""
        if self.data.ncon == 0:
            return 0.0
        c6 = np.zeros(6, dtype=float)
        total = 0.0
        for i in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, i, c6)
            total += float(np.linalg.norm(c6[:3]))
        return total


if __name__ == "__main__":
    # Quick random rollout: confirms the env runs and prints detumbling metrics.
    env = FreeFlyerCaptureEnv(domain_randomize=True)
    obs, info = env.reset(seed=0)
    print(f"obs dim={obs.shape[0]}  act dim={env.action_space.shape[0]}  "
          f"target_mass={info['target_mass']:.1f}kg  start_spin={info['target_angvel_norm']:.3f}rad/s")
    ret = 0.0
    for _ in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ret += r
        if term or trunc:
            break
    print(f"steps={env._steps}  return={ret:.1f}  end_spin={info['target_angvel_norm']:.3f}rad/s  "
          f"base_disturbance={info['base_angvel_norm']:.4f}rad/s  contacts={info['ncon']}")
