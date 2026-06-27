"""Model-based baselines for contact detumbling (the *strong* comparison).

The headline baseline is :class:`ARCController`, a re-implementation of **Active
Resistance Control** (Wang, Liu & Cai, *Nonlinear Dynamics* 113(18):24937-24965,
2025; doi:10.1007/s11071-025-11381-z). Its two defining properties, taken from the
paper's method description, are:

* it **tracks a desired contact force** so the arm does not separate from the target
  after the initial collision (force feedback, not position control); and
* its control law **uses only measured contact force and relative motion -- no target
  inertial or contact parameters** (no mass/inertia identification).

That second property is why ARC is the *right* baseline for an unknown-inertia study:
it is already inertia-agnostic, so the honest comparison is not "RL is robust where
the baseline fails" but the base-attitude disturbance (chaser ACS fuel) that ARC
ignores and the learned policy is rewarded to minimise. ARC also requires a force/
torque sensor, whereas our policy observes kinematics only -- the sensing-reduction
angle (see ``docs/contribution.md``).

The end-effector wrench is mapped to joint torques by the transpose of the
end-effector Jacobian (``mj_jacSite``).

.. note::
   Reconciled against the full PDF (Wang et al. 2025): the normal-direction force law is
   ARC Eq. (5) -- a two-phase, force-feedback-only law (phase 1: ``beta1 * max(Fc)`` to prevent
   post-collision separation; phase 2: incremental tracking ``u += beta2 * (Fc* - Fc)``) -- and
   the detumbling uses the paper's force/position **hybrid** strategy (Sec. 3.3): ARC on the
   contact normal, a motion controller on the tangential directions. Gains here are tuned to
   our MuJoCo scaffold, not copied from the paper's two-ball example.

All controllers share ``act(env, obs) -> action`` and an optional ``reset()``.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - env import already guards this
    mujoco = None


class ZeroController:
    """Apply zero torque. Physical reference (free target keeps spinning), not a baseline.

    Included so the comparison table has a "do nothing" lower bound, making the spin
    reduction of the real controllers interpretable.
    """

    name = "zero"

    def reset(self) -> None:  # noqa: D401 - trivial
        pass

    def act(self, env, obs) -> np.ndarray:  # noqa: ARG002 - obs unused by design
        return np.zeros(env.action_space.shape, dtype=np.float32)


class PolicyController:
    """Adapter wrapping a trained Stable-Baselines3 policy as a controller."""

    name = "rl"

    def __init__(self, model, *, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, env, obs) -> np.ndarray:  # noqa: ARG002 - env unused (obs-only policy)
        action, _ = self.model.predict(np.asarray(obs), deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


class ARCController:
    """Active Resistance Control baseline: force-tracking, inertia-agnostic detumbling.

    Control law (per step, all quantities measured -- no target inertia/mass used):

    1. **Reach** until contact: a Cartesian PD pulls the end-effector to the grasp point.
    2. **ARC normal force** once in contact (Eq. 5, force feedback only):
       phase 1 (first ``ts_steps`` of contact) ``u = beta1 * max(Fc)`` prevents post-collision
       separation; phase 2 ``u += beta2 * (f_desired - Fc)`` incrementally tracks the desired
       contact force. ``u`` acts along the inward contact normal.
    3. **Tangential motion controller** (the hybrid strategy's other half, Sec. 3.3):
       a force opposing the tangential contact-point surface velocity ``omega_target x r``
       bleeds the target's spin. Uses measured angular velocity (kinematic), never inertia.

    Args:
        model: a ``mujoco.MjModel`` (resolves ids and the actuator scale).
        f_desired: desired contact normal force to track (N).
        beta1: phase-1 anti-separation gain (Eq. 6 requires ``beta1 >= 0.5``).
        beta2: phase-2 incremental force-tracking gain (Eq. 6 requires ``0 < beta2 < 1``).
        ts_steps: contact steps before switching from phase 1 to phase 2.
        u_max: anti-windup clamp on the integral normal force (N).
        kp_reach, kd_reach: pre-contact Cartesian impedance stiffness / damping.
        k_resist: gain on the tangential resisting force (N s/m).
        k_joint_damp: joint-space derivative damping for numerical stability.
        contact_gate: end-effector-to-grasp distance (m) below which contact mode engages.
    """

    name = "arc"

    def __init__(
        self,
        model,
        *,
        f_desired: float = 20.0,
        beta1: float = 0.5,
        beta2: float = 0.5,
        ts_steps: int = 10,
        u_max: float = 200.0,
        kp_reach: float = 40.0,
        kd_reach: float = 8.0,
        k_resist: float = 8.0,
        k_joint_damp: float = 0.5,
        contact_gate: float = 0.15,
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise RuntimeError("mujoco is required for ARCController")
        self.model = model
        self.f_desired = f_desired
        self.beta1 = beta1
        self.beta2 = beta2
        self.ts_steps = ts_steps
        self.u_max = u_max
        self.kp_reach = kp_reach
        self.kd_reach = kd_reach
        self.k_resist = k_resist
        self.k_joint_damp = k_joint_damp
        self.contact_gate = contact_gate
        self._u = 0.0          # integral normal-force state (Eq. 5 phase 2)
        self._max_fc = 0.0     # running max contact force (Eq. 5 phase 1)
        self._contact_steps = 0

        self._ee_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self._grasp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp")
        self._tgt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._arm_dofs = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in ("j1", "j2", "j3")],
            dtype=int,
        )
        self._ctrl_max = float(model.actuator_ctrlrange[:, 1].max())
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._vel6 = np.zeros(6)

    def reset(self) -> None:
        self._u = 0.0
        self._max_fc = 0.0
        self._contact_steps = 0

    def act(self, env, obs) -> np.ndarray:  # noqa: ARG002 - obs unused (privileged state)
        m, d = env.model, env.data
        x_ee = d.site_xpos[self._ee_sid]
        x_grasp = d.site_xpos[self._grasp_sid]
        x_tgt = d.xpos[self._tgt_bid]

        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self._ee_sid)
        J_arm = self._jacp[:, self._arm_dofs]              # (3, 3)
        v_ee = self._jacp @ d.qvel                          # (3,) world frame

        dist = float(np.linalg.norm(x_grasp - x_ee))
        f_meas = env._net_contact_force()                   # measured contact force (N)
        in_contact = dist < self.contact_gate or f_meas > 1e-6

        if not in_contact:
            # (1) Reach: pull the end-effector onto the grasp point.
            self._contact_steps = 0
            f_ee = self.kp_reach * (x_grasp - x_ee) - self.kd_reach * v_ee
        else:
            # Inward normal: from end-effector toward the target centre.
            n = x_tgt - x_ee
            n = n / (np.linalg.norm(n) + 1e-9)
            # (2) ARC normal force law, Eq. (5) -- force feedback only, no inertia.
            self._max_fc = max(self._max_fc, f_meas)
            self._contact_steps += 1
            if self._contact_steps <= self.ts_steps:        # phase 1: prevent separation
                self._u = self.beta1 * self._max_fc
            else:                                            # phase 2: track desired force
                self._u += self.beta2 * (self.f_desired - f_meas)
            self._u = float(np.clip(self._u, 0.0, self.u_max))   # anti-windup
            f_normal = self._u * n
            # (3) Tangential motion controller: oppose the contact-point surface velocity.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY,
                                     self._tgt_bid, self._vel6, 0)
            omega_t = self._vel6[:3]                         # measured angular velocity
            r = x_ee - x_tgt
            v_surf = np.cross(omega_t, r)                    # contact-point velocity
            v_tan = v_surf - (v_surf @ n) * n               # tangential component
            f_resist = -self.k_resist * v_tan - self.kd_reach * (v_ee - (v_ee @ n) * n)
            f_ee = f_normal + f_resist

        tau = J_arm.T @ f_ee - self.k_joint_damp * d.qvel[self._arm_dofs]
        return np.clip(tau / self._ctrl_max, -1.0, 1.0).astype(np.float32)
