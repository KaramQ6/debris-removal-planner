"""Real-time training dashboard for Debris Removal RL Agent."""

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_history(history_path: Path):
    if not history_path.exists():
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def create_dashboard(history_path: Path, output_path: Path):
    history = load_history(history_path)
    if not history or not history.get("episode_rewards"):
        print("Waiting for training data...")
        return False

    rewards = history["episode_rewards"]
    cleared = history["episode_cleared"]
    delta_v = history["episode_delta_vs"]
    episodes = np.arange(len(rewards))

    # Calculate moving averages for smoothness
    window = min(50, len(rewards))
    reward_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    cleared_ma = np.convolve(cleared, np.ones(window)/window, mode='valid')

    # Create Dashboard with Subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Training Reward (Learning Progress)", "Targets Cleared per Mission", "Delta-V Consumed (m/s)"),
        vertical_spacing=0.1
    )

    # 1. Reward Curve
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards, mode='markers', marker=dict(size=2, color='rgba(52, 152, 219, 0.3)'), name="Reward", showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=episodes[window-1:], y=reward_ma, mode='lines', line=dict(color='#3498DB', width=3), name="Moving Avg (50)"),
        row=1, col=1
    )

    # 2. Cleared Targets
    fig.add_trace(
        go.Scatter(x=episodes, y=cleared, mode='markers', marker=dict(size=2, color='rgba(46, 204, 113, 0.3)'), name="Cleared", showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=episodes[window-1:], y=cleared_ma, mode='lines', line=dict(color='#2ECC71', width=3), name="Cleared Avg"),
        row=2, col=1
    )

    # 3. Delta-V
    fig.add_trace(
        go.Scatter(x=episodes, y=delta_v, mode='lines', line=dict(color='#E74C3C', width=1, shape='spline'), name="Delta-V"),
        row=3, col=1
    )

    # Aesthetics
    fig.update_layout(
        height=900,
        title_text=f"🛰️ Debris Removal Training Dashboard (Episodes: {len(rewards)})",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
    )

    fig.write_html(str(output_path))
    return True

if __name__ == "__main__":
    hist_path = Path("results/training_history.json")
    out_path = Path("results/training_dashboard.html")
    
    print(f"Monitoring {hist_path}...")
    print(f"Dashboard will be saved to {out_path}")
    
    while True:
        if create_dashboard(hist_path, out_path):
            print(f"[{time.strftime('%H:%M:%S')}] Dashboard updated.")
        time.sleep(30) # Update every 30 seconds
