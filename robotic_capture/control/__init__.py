"""Controllers for the free-floating capture/detumbling task.

Exposes the model-based baselines and a thin adapter for trained RL policies, all
sharing one call signature: ``controller.act(env, obs) -> np.ndarray`` (normalized
action in ``[-1, 1]^nu``). This lets the evaluation harness score every controller
the same way (see ``robotic_capture/eval.py``).
"""

from .baselines import ARCController, ZeroController, PolicyController

__all__ = ["ARCController", "ZeroController", "PolicyController"]
