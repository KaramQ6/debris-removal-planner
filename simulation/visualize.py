"""Mission visualization — 2D polar and 3D orbital plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .orbit_env import OrbitDebrisEnv
from .policies import nearest_neighbor_policy, random_policy, risk_weighted_policy


def _run_single_episode(
    policy_name: str,
    seed: int = 42,
    targets: int = 8,
    fuel: float = 1200.0,
    max_steps: int = 50,
    model_path: str = "",
) -> dict[str, Any]:
    """Run one episode and return scenario + trajectory data."""
    env = OrbitDebrisEnv(
        seed=seed, target_count=targets, fuel_budget=fuel, max_steps=max_steps
    )
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    model = None
    if policy_name == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)

    done = False
    info: dict = {}
    while not done:
        if policy_name == "random":
            action = random_policy(env, rng)
        elif policy_name == "nearest":
            action = nearest_neighbor_policy(env)
        elif policy_name == "risk_weighted":
            action = risk_weighted_policy(env)
        elif policy_name == "ppo":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    # Gather scenario data for plotting
    targets_data = []
    for t in env._targets:
        targets_data.append({
            "id": t.target_id,
            "name": t.name,
            "angle_deg": t.angle_deg,
            "risk": t.risk,
            "altitude_km": t.altitude_km,
        })

    return {
        "policy": policy_name,
        "start_angle_deg": env._scenario.start_angle_deg,
        "fuel_budget": env._scenario.fuel_budget,
        "targets": targets_data,
        "trajectory": info.get("trajectory", []),
        "total_delta_v": info["total_delta_v"],
        "cleared": info["cleared"],
        "total_targets": info["total_targets"],
        "fuel_remaining": info["fuel_remaining"],
    }


def plot_polar_mission(data: dict, output_path: Path) -> None:
    """2D polar plot showing debris positions and collection path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        print("matplotlib required for polar plot. pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Plot debris targets
    for t in data["targets"]:
        angle_rad = math.radians(t["angle_deg"])
        radius = 1.0  # Normalized orbital ring
        risk_color = plt.cm.RdYlGn_r(t["risk"])
        size = 80 + 200 * t["risk"]
        ax.scatter(
            angle_rad, radius,
            c=[risk_color], s=size, zorder=5, edgecolors="white",
            linewidth=1.5, alpha=0.9,
        )
        ax.annotate(
            t["name"],
            xy=(angle_rad, radius),
            xytext=(angle_rad, radius + 0.15),
            fontsize=7, ha="center", va="center",
            color="#2C3E50", fontweight="bold",
        )

    # Plot spacecraft start position
    start_rad = math.radians(data["start_angle_deg"])
    ax.scatter(
        start_rad, 1.0, c="#2ECC71", s=200, marker="^", zorder=10,
        edgecolors="white", linewidth=2, label="Spacecraft Start",
    )

    # Draw collection path
    trajectory = data["trajectory"]
    if trajectory:
        path_angles = [start_rad]
        for hop in trajectory:
            path_angles.append(math.radians(hop["to_angle"]))

        for i in range(len(path_angles) - 1):
            a1, a2 = path_angles[i], path_angles[i + 1]
            ax.annotate(
                "",
                xy=(a2, 1.0), xytext=(a1, 1.0),
                arrowprops=dict(
                    arrowstyle="->", color="#3498DB",
                    lw=2, connectionstyle="arc3,rad=0.1",
                ),
            )
            # Step number
            mid_angle = (a1 + a2) / 2
            ax.text(
                mid_angle, 1.12, str(i + 1),
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="#3498DB",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#3498DB", alpha=0.8),
            )

    # Styling
    ax.set_ylim(0, 1.4)
    ax.set_rticks([])
    ax.set_title(
        f"Mission Path — {data['policy'].upper()} Policy\n"
        f"Cleared: {data['cleared']}/{data['total_targets']} | "
        f"ΔV: {data['total_delta_v']:.1f} m/s | "
        f"Fuel left: {data['fuel_remaining']:.1f} m/s",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))

    # Color bar for risk
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Collision Risk", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved polar plot: {output_path}")


def plot_3d_orbit(data: dict, output_path: Path) -> None:
    """3D orbital view using matplotlib (static)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for 3D plot. pip install matplotlib")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    earth_r = 0.3
    x = earth_r * np.outer(np.cos(u), np.sin(v))
    y = earth_r * np.outer(np.sin(u), np.sin(v))
    z = earth_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3, color="#3498DB")

    # Draw orbital ring
    theta = np.linspace(0, 2 * np.pi, 200)
    orbit_r = 1.0
    ax.plot(
        orbit_r * np.cos(theta), orbit_r * np.sin(theta),
        np.zeros_like(theta),
        color="#BDC3C7", linewidth=1, alpha=0.5, linestyle="--",
    )

    # Plot debris
    for t in data["targets"]:
        angle_rad = math.radians(t["angle_deg"])
        x_t = orbit_r * math.cos(angle_rad)
        y_t = orbit_r * math.sin(angle_rad)
        z_t = 0.0
        risk_color = plt.cm.RdYlGn_r(t["risk"])
        size = 40 + 120 * t["risk"]
        ax.scatter(
            [x_t], [y_t], [z_t],
            c=[risk_color], s=size, depthshade=False,
            edgecolors="white", linewidth=0.8, zorder=5,
        )

    # Plot spacecraft start
    start_rad = math.radians(data["start_angle_deg"])
    ax.scatter(
        [orbit_r * math.cos(start_rad)],
        [orbit_r * math.sin(start_rad)],
        [0.0],
        c="#2ECC71", s=200, marker="^", depthshade=False,
        edgecolors="white", linewidth=2, zorder=10,
    )

    # Draw trajectory path
    trajectory = data["trajectory"]
    if trajectory:
        path_x = [orbit_r * math.cos(start_rad)]
        path_y = [orbit_r * math.sin(start_rad)]
        path_z = [0.0]
        for hop in trajectory:
            a = math.radians(hop["to_angle"])
            path_x.append(orbit_r * math.cos(a))
            path_y.append(orbit_r * math.sin(a))
            path_z.append(0.0)
        ax.plot(path_x, path_y, path_z, color="#E74C3C", linewidth=2.5, alpha=0.8)

    ax.set_title(
        f"3D Orbital View — {data['policy'].upper()} Policy\n"
        f"ΔV: {data['total_delta_v']:.1f} m/s | "
        f"Cleared: {data['cleared']}/{data['total_targets']}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 0.3])

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 3D plot: {output_path}")


def generate_plotly_3d(data: dict, output_path: Path) -> None:
    """Interactive 3D orbit visualization using Plotly (HTML)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly required for interactive 3D. pip install plotly")
        return

    fig = go.Figure()

    # Orbital ring
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta), y=np.sin(theta), z=np.zeros_like(theta),
        mode="lines", line=dict(color="gray", width=2, dash="dash"),
        name="Orbit",
    ))

    # Earth sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    er = 0.3
    fig.add_trace(go.Surface(
        x=er * np.cos(u) * np.sin(v),
        y=er * np.sin(u) * np.sin(v),
        z=er * np.cos(v),
        colorscale=[[0, "#3498DB"], [1, "#2980B9"]],
        showscale=False, opacity=0.4, name="Earth",
    ))

    # Debris targets
    for t in data["targets"]:
        a = math.radians(t["angle_deg"])
        risk_pct = int(t["risk"] * 100)
        fig.add_trace(go.Scatter3d(
            x=[math.cos(a)], y=[math.sin(a)], z=[0],
            mode="markers+text",
            marker=dict(
                size=6 + 10 * t["risk"],
                color=f"rgb({int(255*t['risk'])}, {int(255*(1-t['risk']))}, 50)",
            ),
            text=[t["name"]],
            textposition="top center",
            name=f"{t['name']} (risk={risk_pct}%)",
        ))

    # Spacecraft start
    sr = math.radians(data["start_angle_deg"])
    fig.add_trace(go.Scatter3d(
        x=[math.cos(sr)], y=[math.sin(sr)], z=[0],
        mode="markers", marker=dict(size=12, color="#2ECC71", symbol="diamond"),
        name="Spacecraft Start",
    ))

    # Trajectory path
    trajectory = data["trajectory"]
    if trajectory:
        px = [math.cos(sr)]
        py = [math.sin(sr)]
        pz = [0.0]
        labels = ["START"]
        for i, hop in enumerate(trajectory):
            a = math.radians(hop["to_angle"])
            px.append(math.cos(a))
            py.append(math.sin(a))
            pz.append(0.0)
            labels.append(f"Step {i+1}: ΔV={hop['delta_v']:.1f}")

        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="lines+markers",
            line=dict(color="#E74C3C", width=5),
            marker=dict(size=4, color="#E74C3C"),
            text=labels,
            name="Mission Path",
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Interactive Orbital Mission — {data['policy'].upper()}<br>"
                f"<sub>ΔV: {data['total_delta_v']:.1f} m/s | "
                f"Cleared: {data['cleared']}/{data['total_targets']} | "
                f"Fuel left: {data['fuel_remaining']:.1f} m/s</sub>"
            ),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.3),
        ),
        width=1000, height=700,
        template="plotly_dark",
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"  Saved interactive 3D: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize debris-removal missions.")
    parser.add_argument("--policy", type=str, default="nearest",
                        choices=["random", "nearest", "risk_weighted", "ppo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--fuel", type=float, default=1200.0)
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
        print(f"\nGenerating visualizations for {policy} policy...")
        data = _run_single_episode(
            policy_name=policy,
            seed=args.seed,
            targets=args.targets,
            fuel=args.fuel,
            max_steps=args.max_steps,
            model_path=args.model_path if policy == "ppo" else "",
        )

        # Save trajectory data
        traj_path = output_dir / f"trajectory_{policy}.json"
        traj_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Generate plots
        plot_polar_mission(data, output_dir / f"mission_polar_{policy}.png")
        plot_3d_orbit(data, output_dir / f"mission_3d_{policy}.png")
        generate_plotly_3d(data, output_dir / f"mission_interactive_{policy}.html")

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
