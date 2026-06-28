"""Scene-snapshot figure: the free-floating capture scaffold at the post-grasp ready pose.

Renders the MuJoCo model (free-floating chaser base + 3-DOF arm + tumbling target) offscreen
at the arm's ready pose and annotates the three bodies. Writes
``docs/paper/figures/fig_scene.pdf``. The offscreen framebuffer is enlarged in-memory, so the
committed MJCF (``sim/assets/space_manipulator.xml``) is left unchanged.

ponytail: fixed camera and label positions are tuned to this specific scene geometry; if the
MJCF body layout changes, re-tune ``cam`` / the annotate anchors. Needs a working GL context
(``mujoco.Renderer``); fails loudly if offscreen rendering is unavailable.

Usage (from the robotic_capture/ dir, with the project .venv):

    python render_scene.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import matplotlib
matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt

MJCF = Path(__file__).parent / "sim" / "assets" / "space_manipulator.xml"
READY = [0.8, -1.6, 0.8]  # arm joint angles (rad); matches FreeFlyerCaptureEnv reset()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="docs/paper/figures/fig_scene.pdf")
    args = p.parse_args()

    # Enlarge the offscreen buffer in-memory (default is 640x480) without touching the file.
    xml = MJCF.read_text(encoding="utf-8").replace(
        '<compiler angle="radian"/>',
        '<compiler angle="radian"/>\n  <visual><global offwidth="1200" offheight="900"/></visual>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[7:10] = READY
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, height=560, width=1200)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.92, 0.0, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 3.5, 90, -55
    r.update_scene(d, cam)
    img = r.render()

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.imshow(img)
    ax.axis("off")
    lab = dict(color="white", fontsize=9, ha="center",
               bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="white", lw=0.5, alpha=0.65))
    arr = dict(arrowstyle="->", color="white", lw=1.0)
    ax.annotate("free-floating\nchaser base", xy=(0.28, 0.5), xytext=(0.17, 0.10),
                xycoords="axes fraction", textcoords="axes fraction", arrowprops=arr, **lab)
    ax.annotate("3-DOF arm", xy=(0.5, 0.45), xytext=(0.5, 0.10),
                xycoords="axes fraction", textcoords="axes fraction", arrowprops=arr, **lab)
    ax.annotate("tumbling\ntarget", xy=(0.7, 0.5), xytext=(0.84, 0.10),
                xycoords="axes fraction", textcoords="axes fraction", arrowprops=arr, **lab)
    fig.tight_layout(pad=0.1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}  ({img.shape[1]}x{img.shape[0]} render at ready pose)")


if __name__ == "__main__":
    main()
