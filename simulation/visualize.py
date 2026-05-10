"""Mission visualization — 3D orbital plots using Keplerian elements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .orbit_env import OrbitDebrisEnv
from .policies import nearest_neighbor_policy, random_policy, risk_weighted_policy


def get_cartesian(sma: float, ecc: float, inc: float, raan: float, arg_p: float, nu: float) -> tuple[float, float, float]:
    """Convert true Keplerian orbital elements to ECI Cartesian coordinates."""
    inc_rad = math.radians(inc)
    raan_rad = math.radians(raan)
    arg_p_rad = math.radians(arg_p)
    nu_rad = math.radians(nu)
    
    # Distance in the orbital plane (radius)
    r = sma * (1.0 - ecc**2) / (1.0 + ecc * math.cos(nu_rad))
    
    # Perifocal coordinates
    x_peri = r * math.cos(nu_rad)
    y_peri = r * math.sin(nu_rad)
    
    # Rotation to ECI
    cw, sw = math.cos(arg_p_rad), math.sin(arg_p_rad)
    cO, sO = math.cos(raan_rad), math.sin(raan_rad)
    ci, si = math.cos(inc_rad), math.sin(inc_rad)
    
    R11 = cO * cw - sO * ci * sw
    R12 = -cO * sw - sO * ci * cw
    R21 = sO * cw + cO * ci * sw
    R22 = -sO * sw + cO * ci * cw
    R31 = si * sw
    R32 = si * cw
    
    x = R11 * x_peri + R12 * y_peri
    y = R21 * x_peri + R22 * y_peri
    z = R31 * x_peri + R32 * y_peri
    
    return x, y, z


def _run_single_episode(
    policy_name: str,
    seed: int = 42,
    targets: int = 8,
    fuel: float = 6000.0,
    max_steps: int = 50,
    model_path: str = "",
    scenario_type: str = "medium",
) -> dict[str, Any]:
    """Run one episode and return scenario + trajectory data."""
    # Use preset logic instead of custom targets/fuel if specified, except generic
    from .scenario import SCENARIO_PRESETS
    
    scenario_func = SCENARIO_PRESETS.get(scenario_type, SCENARIO_PRESETS["medium"])
    scenario = scenario_func(seed=seed)
    
    env = OrbitDebrisEnv(
        scenario_generator=lambda **kwargs: scenario,
        seed=seed,
        max_targets=max(12, len(scenario.targets))
    )
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    model = None
    if policy_name == "ppo":
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path)
        except ImportError:
            print("Warning: sb3_contrib not installed, skipping PPO.")

    done = False
    info: dict = {}
    while not done:
        if policy_name == "random":
            action = random_policy(env, rng)
        elif policy_name == "nearest":
            action = nearest_neighbor_policy(env)
        elif policy_name == "risk_weighted":
            action = risk_weighted_policy(env)
        elif policy_name == "ppo" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = random_policy(env, rng)

        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    targets_data = []
    for t in env._targets:
        targets_data.append({
            "id": t.target_id,
            "name": t.name,
            "sma_km": t.sma_km,
            "eccentricity": t.eccentricity,
            "inclination_deg": t.inclination_deg,
            "raan_deg": t.raan_deg,
            "arg_periapsis_deg": t.arg_periapsis_deg,
            "true_anomaly_deg": t.true_anomaly_deg,
            "risk": t.risk,
            "target_type": getattr(t, "target_type", "FRAG"),
            "age_days": getattr(t, "age_days", 0.0),
        })

    return {
        "policy": policy_name,
        "scenario": scenario.name,
        "start_sma_km": env._scenario.start_sma_km,
        "start_eccentricity": env._scenario.start_eccentricity,
        "start_inclination_deg": env._scenario.start_inclination_deg,
        "start_raan_deg": env._scenario.start_raan_deg,
        "start_arg_periapsis_deg": env._scenario.start_arg_periapsis_deg,
        "start_true_anomaly_deg": env._scenario.start_true_anomaly_deg,
        "fuel_budget": env._scenario.fuel_budget,
        "targets": targets_data,
        "trajectory": info.get("trajectory", []),
        "total_delta_v": info["total_delta_v"],
        "cleared": info["cleared"],
        "total_targets": info["total_targets"],
        "fuel_remaining": info["fuel_remaining"],
    }


def plot_polar_mission(data: dict, output_path: Path) -> None:
    """Fallback 2D plot (skipped in 3D upgrade)."""
    pass


def plot_3d_orbit(data: dict, output_path: Path) -> None:
    """3D orbital view using matplotlib (static)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    earth_r = 6371.0  # Earth radius in km

    # Draw Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = earth_r * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_r * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color="#3498DB")

    # Draw orbital rings for each target
    theta = np.linspace(0, 360, 200)
    for t in data["targets"]:
        ring_x, ring_y, ring_z = [], [], []
        for th in theta:
            rx, ry, rz = get_cartesian(
                t["sma_km"], t["eccentricity"], t["inclination_deg"], 
                t["raan_deg"], t["arg_periapsis_deg"], th
            )
            ring_x.append(rx)
            ring_y.append(ry)
            ring_z.append(rz)
        ax.plot(ring_x, ring_y, ring_z, color="#BDC3C7", linewidth=0.5, alpha=0.3, linestyle="--")

        # Plot actual target position
        tx, ty, tz = get_cartesian(
            t["sma_km"], t["eccentricity"], t["inclination_deg"], 
            t["raan_deg"], t["arg_periapsis_deg"], t["true_anomaly_deg"]
        )
        risk_color = plt.cm.RdYlGn_r(t["risk"])
        size = 40 + 120 * t["risk"]
        ax.scatter([tx], [ty], [tz], c=[risk_color], s=size, edgecolors="white", linewidth=0.8, zorder=5)

    # Draw spacecraft trajectory
    trajectory = data["trajectory"]
    if trajectory:
        path_x, path_y, path_z = [], [], []
        # Start point
        sx, sy, sz = get_cartesian(
            data["start_sma_km"], data.get("start_eccentricity", 0.0), 
            data["start_inclination_deg"], data["start_raan_deg"], 
            data.get("start_arg_periapsis_deg", 0.0), data["start_true_anomaly_deg"]
        )
        path_x.append(sx)
        path_y.append(sy)
        path_z.append(sz)
        
        ax.scatter([sx], [sy], [sz], c="#2ECC71", s=200, marker="^", edgecolors="white", linewidth=2, zorder=10)

        for hop in trajectory:
            hx, hy, hz = get_cartesian(
                hop["to_sma"], hop.get("to_ecc", 0.0), hop["to_inc"], 
                hop["to_raan"], hop.get("to_arg_p", 0.0), hop.get("to_nu", 0.0)
            )
            path_x.append(hx)
            path_y.append(hy)
            path_z.append(hz)
        ax.plot(path_x, path_y, path_z, color="#E74C3C", linewidth=2.5, alpha=0.8)

    limit = 10000
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_title(
        f"Realistic 3D Orbital Path — {data['policy'].upper()} Policy\n"
        f"Scenario: {data['scenario']} | ΔV Consumed: {data['total_delta_v']:.1f} m/s",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3D plot: {output_path}")


def generate_plotly_3d(data: dict, output_path: Path) -> None:
    """Interactive 3D orbit visualization using Plotly (HTML)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    fig = go.Figure()
    earth_r = 6371.0

    # Earth sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    fig.add_trace(go.Surface(
        x=earth_r * np.cos(u) * np.sin(v),
        y=earth_r * np.sin(u) * np.sin(v),
        z=earth_r * np.cos(v),
        colorscale=[[0, "#3498DB"], [1, "#2980B9"]],
        showscale=False, opacity=0.8, name="Earth",
    ))

    # Debris targets & Rings
    theta = np.linspace(0, 360, 100)
    for t in data["targets"]:
        ring_x, ring_y, ring_z = [], [], []
        for th in theta:
            rx, ry, rz = get_cartesian(
                t["sma_km"], t["eccentricity"], t["inclination_deg"], 
                t["raan_deg"], t["arg_periapsis_deg"], th
            )
            ring_x.append(rx)
            ring_y.append(ry)
            ring_z.append(rz)
            
        fig.add_trace(go.Scatter3d(
            x=ring_x, y=ring_y, z=ring_z,
            mode="lines", line=dict(color="rgba(150, 150, 150, 0.5)", width=1, dash="dash"),
            showlegend=False, hoverinfo="skip"
        ))
        
        # Add Debris Cloud effect (point cloud around target)
        cloud_size = 20
        cloud_rng = np.random.default_rng(int(t["sma_km"]))
        offsets = cloud_rng.normal(0, 15.0, (cloud_size, 3))
        
        # Current position of target (using true_anomaly_deg)
        tx, ty, tz = get_cartesian(
            t["sma_km"], t["eccentricity"], t["inclination_deg"], 
            t["raan_deg"], t["arg_periapsis_deg"], t["true_anomaly_deg"]
        )
        
        fig.add_trace(go.Scatter3d(
            x=tx + offsets[:, 0], y=ty + offsets[:, 1], z=tz + offsets[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="white", opacity=0.4),
            name=f"Cloud: {t['name']}",
            showlegend=False, hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter3d(
            x=[tx], y=[ty], z=[tz],
            mode="markers+text",
            marker=dict(size=5 + 10 * t["risk"], color="#F1C40F"),
            text=[t["name"]], textposition="top center", name=t['name'],
            hoverinfo="text",
            hovertext=[(
                f"<b>{t['name']}</b><br>"
                f"Type: {t['target_type']}<br>"
                f"Risk: {t['risk']*100:.1f}%<br>"
                f"SMA: {t['sma_km']:.1f} km"
            )]
        ))

    # Trajectory
    trajectory = data["trajectory"]
    if trajectory:
        sx, sy, sz = get_cartesian(
            data["start_sma_km"], data.get("start_eccentricity", 0.0), 
            data["start_inclination_deg"], data["start_raan_deg"], 
            data.get("start_arg_periapsis_deg", 0.0), data["start_true_anomaly_deg"]
        )
        px, py, pz = [sx], [sy], [sz]
        labels = ["START"]
        
        fig.add_trace(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode="markers", marker=dict(size=12, color="#2ECC71", symbol="diamond"),
            name="Spacecraft Start",
        ))

        for i, hop in enumerate(trajectory):
            hx, hy, hz = get_cartesian(
                hop["to_sma"], hop.get("to_ecc", 0.0), hop["to_inc"], 
                hop["to_raan"], hop.get("to_arg_p", 0.0), hop.get("to_nu", 0.0)
            )
            px.append(hx)
            py.append(hy)
            pz.append(hz)
            labels.append(f"Step {i+1}: ΔV={hop['delta_v']:.1f}")

        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="lines+markers",
            line=dict(color="#E74C3C", width=4),
            marker=dict(size=4, color="#E74C3C"),
            text=labels,
            hoverinfo="text",
            name="Transfer Path",
        ))

    fig.update_layout(
        title=dict(text=f"Interactive 3D Orbital Mission — {data['policy'].upper()} Policy ({data['scenario']})<br><sub>Total ΔV: {data['total_delta_v']:.1f} m/s</sub>", font=dict(size=16)),
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)", aspectmode="data"),
        width=1000, height=800, template="plotly_dark",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"  Saved interactive 3D: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize debris-removal missions.")
    parser.add_argument("--policy", type=str, default="nearest",
                        choices=["random", "nearest", "risk_weighted", "ppo"])
    parser.add_argument("--scenario", type=str, default="medium",
                        choices=["easy", "medium", "hard", "iridium_cosmos", "fengyun", "shakti"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=6000.0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--all-policies", action="store_true",
                        help="Generate visualizations for all baseline policies.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = ["random", "nearest", "risk_weighted"] if args.all_policies else [args.policy]
    if args.model_path:
        policies.append("ppo")

    for policy in policies:
        print(f"\nGenerating 3D visualizations for {policy} policy on {args.scenario} scenario...")
        data = _run_single_episode(
            policy_name=policy,
            seed=args.seed,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
            model_path=args.model_path if policy == "ppo" else "",
            scenario_type=args.scenario,
        )

        traj_path = output_dir / f"trajectory_{policy}_{args.scenario}.json"
        traj_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        plot_3d_orbit(data, output_dir / f"mission_3d_{policy}_{args.scenario}.png")
        generate_plotly_3d(data, output_dir / f"mission_interactive_{policy}_{args.scenario}.html")

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
