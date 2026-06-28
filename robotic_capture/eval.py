"""Evaluation harness: detumbling metrics with bootstrap confidence intervals.

Runs one or more controllers over a grid of (inertia scale, injected base disturbance,
initial spin, seed) episodes on the *same* configurations, so the comparison is paired and
fair. The pivoted headline metric is **base-attitude disturbance rejection** (peak induced
base angular velocity, a proxy for chaser ACS fuel) as a function of the injected disturbance
torque; the inertia sweep is reported as the parity check. Writes a Markdown table and a
per-episode CSV to ``results/``.

Usage (from repo root, with the project .venv):

    # headline: disturbance sweep at nominal inertia (calibrated torques ~0.05-0.12 rad/s)
    python -m robotic_capture.eval --policy runs/sac.zip --scales 1.0 --dists 0.3 0.4 0.5 0.7
    # parity check: inertia sweep, no added disturbance
    python -m robotic_capture.eval --policy runs/sac.zip --scales 0.5 1.0 1.5 2.0 --dists 0.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv
from robotic_capture.control.baselines import ARCController, ZeroController, PolicyController


def rollout(env: FreeFlyerCaptureEnv, controller, *, inertia_scale: float,
            spin: float, seed: int, detumble_tol: float,
            base_disturbance: float = 0.0) -> dict:
    """Run one episode and return its metrics."""
    obs, info = env.reset(seed=seed,
                          options={"inertia_scale": inertia_scale,
                                   "target_angvel": [0.0, 0.0, spin],
                                   "base_disturbance": base_disturbance})
    if hasattr(controller, "reset"):
        controller.reset()
    dt = env.frame_skip * float(env.model.opt.timestep)
    start = info["target_angvel_norm"]
    min_spin = start
    base_peak = base_int = contact_int = contact_peak = 0.0
    detumble_step = None
    steps = 0
    while True:
        obs, _, term, trunc, info = env.step(controller.act(env, obs))
        steps += 1
        spin_now = info["target_angvel_norm"]
        min_spin = min(min_spin, spin_now)
        base_peak = max(base_peak, info["base_angvel_norm"])
        base_int += info["base_angvel_norm"] * dt
        contact_peak = max(contact_peak, info["contact_force"])
        contact_int += info["contact_force"] * dt
        if detumble_step is None and spin_now < detumble_tol:
            detumble_step = steps
        if term or trunc:
            break
    return {
        "controller": controller.name,
        "inertia_scale": inertia_scale,
        "base_disturbance": base_disturbance,
        "spin": spin,
        "seed": seed,
        "start_spin": start,
        "end_spin": info["target_angvel_norm"],
        "min_spin": min_spin,
        "spin_reduction_pct": 100.0 * (start - min_spin) / start if start > 0 else 0.0,
        "success": float(min_spin < detumble_tol),
        "detumble_time_s": (detumble_step * dt) if detumble_step else float("nan"),
        "base_disturb_final": info["base_angvel_norm"],
        "base_disturb_peak": base_peak,
        "base_disturb_integral": base_int,   # proxy for ACS reorientation effort
        "contact_force_peak": contact_peak,
        "contact_force_mean": contact_int / (steps * dt) if steps else 0.0,
    }


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 10000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 3
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = rng or np.random.default_rng(0)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def _fmt(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def evaluate(controllers: dict, *, scales, spins, seeds, dists, detumble_tol: float,
             max_steps: int) -> list[dict]:
    """Run every controller over the (scale, disturbance, spin, seed) grid on a shared env."""
    env = FreeFlyerCaptureEnv(domain_randomize=False, max_steps=max_steps,
                              detumble_tol=detumble_tol)
    rows = []
    for name, ctrl in controllers.items():
        for scale in scales:
            for dist in dists:
                for spin in spins:
                    for seed in seeds:
                        rows.append(rollout(env, ctrl, inertia_scale=scale, spin=spin,
                                            seed=seed, detumble_tol=detumble_tol,
                                            base_disturbance=dist))
        n = len(scales) * len(dists) * len(spins) * len(seeds)
        print(f"  ran {name}: {n} episodes")
    return rows


def headline_table(rows: list[dict], scales) -> str:
    """Markdown table: base-attitude disturbance (headline) + spin reduction (parity)."""
    controllers = sorted({r["controller"] for r in rows})
    lines = [
        "### Headline: induced base-attitude disturbance vs target inertia",
        "",
        "Final base angular velocity (rad/s; proxy for chaser ACS fuel) and spin reduction "
        "(%, parity check), mean [95% bootstrap CI].",
        "",
        "| inertia scale | controller | base disturbance (rad/s) | spin reduction (%) | success |",
        "|---|---|---|---|---|",
    ]
    for scale in scales:
        for c in controllers:
            sub = [r for r in rows if r["controller"] == c and r["inertia_scale"] == scale]
            if not sub:
                continue
            bd = bootstrap_ci([r["base_disturb_final"] for r in sub])
            red = bootstrap_ci([r["spin_reduction_pct"] for r in sub])
            succ = np.mean([r["success"] for r in sub]) * 100
            lines.append(f"| {scale:g} | {c} | {_fmt(*bd)} | {_fmt(*red)} | {succ:.0f}% |")
    return "\n".join(lines)


def disturbance_table(rows: list[dict], dists) -> str:
    """Markdown table: base-attitude disturbance REJECTION vs injected disturbance torque.

    This is the pivoted headline: at matched target inertia, how well each controller keeps
    the chaser base steady as the injected ACS/thruster-proxy torque grows. Peak base
    angular velocity (rad/s) is the rejection metric; spin reduction (%) is the parity check.
    """
    controllers = sorted({r["controller"] for r in rows})
    lines = [
        "### Headline: base-attitude disturbance rejection vs injected torque",
        "",
        "Peak base angular velocity (rad/s; lower = better rejection) and spin reduction "
        "(%, parity check), mean [95% bootstrap CI].",
        "",
        "| disturbance (N·m) | controller | base disturbance peak (rad/s) | spin reduction (%) | success |",
        "|---|---|---|---|---|",
    ]
    for dist in dists:
        for c in controllers:
            sub = [r for r in rows if r["controller"] == c and r["base_disturbance"] == dist]
            if not sub:
                continue
            bd = bootstrap_ci([r["base_disturb_peak"] for r in sub])
            red = bootstrap_ci([r["spin_reduction_pct"] for r in sub])
            succ = np.mean([r["success"] for r in sub]) * 100
            lines.append(f"| {dist:g} | {c} | {_fmt(*bd)} | {_fmt(*red)} | {succ:.0f}% |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=str, default=None, help="path to a trained SB3 .zip")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--dists", type=float, nargs="+", default=[0.0],
                   help="injected base disturbance torques (N·m). Calibration: ~0.175 rad/s "
                        "open-loop base spin per N·m, so {0.3,0.4,0.5,0.7} ~ {0.05..0.12} rad/s.")
    p.add_argument("--spins", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--detumble-tol", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out", type=str, default="robotic_capture/results")
    args = p.parse_args()

    model_env = FreeFlyerCaptureEnv(domain_randomize=False)
    model_env.reset()
    controllers = {"zero": ZeroController(), "arc": ARCController(model_env.model)}
    if args.policy:
        from stable_baselines3 import SAC
        # SECURITY: SAC.load unpickles the policy zip -- only load trusted, self-produced
        # run files, never a model zip from an untrusted source (pickle = arbitrary code exec).
        controllers["rl"] = PolicyController(SAC.load(args.policy))

    print(f"Evaluating {list(controllers)} over scales={args.scales} dists={args.dists} "
          f"spins={args.spins} seeds={args.seeds}")
    rows = evaluate(controllers, scales=args.scales, spins=args.spins, seeds=args.seeds,
                    dists=args.dists, detumble_tol=args.detumble_tol, max_steps=args.max_steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_episodes.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Disturbance headline when a real disturbance sweep is present; inertia parity table
    # whenever more than one inertia scale is tested. Both can appear in one run.
    sections = []
    if any(d != 0.0 for d in args.dists):
        sections.append(disturbance_table(rows, args.dists))
    if len(args.scales) > 1:
        sections.append(headline_table(rows, args.scales))
    if not sections:                       # fallback: at least emit the inertia table
        sections.append(headline_table(rows, args.scales))
    table = "\n\n".join(sections)
    (out / "eval_table.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nWrote {csv_path} and {out / 'eval_table.md'}")


if __name__ == "__main__":
    main()
