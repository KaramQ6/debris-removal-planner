"""Evaluation harness: detumbling metrics with bootstrap confidence intervals.

Runs one or more controllers over a grid of (inertia scale, initial spin, seed) episodes
on the *same* configurations, so the comparison is paired and fair. Reports the paper's
headline metric -- induced **base-attitude disturbance** (a proxy for chaser ACS fuel) --
as a function of target inertia, alongside spin reduction (the parity check) and contact
force. Writes a Markdown table and a per-episode CSV to ``results/``.

Usage (from repo root, with the project .venv):

    python -m robotic_capture.eval                          # ARC + zero baselines
    python -m robotic_capture.eval --policy runs/sac.zip    # add the trained RL policy
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from robotic_capture.sim.free_flyer_env import FreeFlyerCaptureEnv
from robotic_capture.control.baselines import ARCController, ZeroController, PolicyController


def rollout(env: FreeFlyerCaptureEnv, controller, *, inertia_scale: float,
            spin: float, seed: int, detumble_tol: float) -> dict:
    """Run one episode and return its metrics."""
    obs, info = env.reset(seed=seed,
                          options={"inertia_scale": inertia_scale,
                                   "target_angvel": [0.0, 0.0, spin]})
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


def evaluate(controllers: dict, *, scales, spins, seeds, detumble_tol: float,
             max_steps: int) -> list[dict]:
    """Run every controller over the full (scale, spin, seed) grid on one shared env."""
    env = FreeFlyerCaptureEnv(domain_randomize=False, max_steps=max_steps,
                              detumble_tol=detumble_tol)
    rows = []
    for name, ctrl in controllers.items():
        for scale in scales:
            for spin in spins:
                for seed in seeds:
                    rows.append(rollout(env, ctrl, inertia_scale=scale, spin=spin,
                                        seed=seed, detumble_tol=detumble_tol))
        n = len(scales) * len(spins) * len(seeds)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=str, default=None, help="path to a trained SB3 .zip")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
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
        controllers["rl"] = PolicyController(SAC.load(args.policy))

    print(f"Evaluating {list(controllers)} over scales={args.scales} "
          f"spins={args.spins} seeds={args.seeds}")
    rows = evaluate(controllers, scales=args.scales, spins=args.spins, seeds=args.seeds,
                    detumble_tol=args.detumble_tol, max_steps=args.max_steps)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_episodes.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    table = headline_table(rows, args.scales)
    (out / "eval_table.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nWrote {csv_path} and {out / 'eval_table.md'}")


if __name__ == "__main__":
    main()
