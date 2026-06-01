"""Evaluate greedy baselines on CelesTrak real-world data for paper Table III."""
import sys, json, functools
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.evaluate import run_policy
from simulation.scenario import from_celestrak_json

DATA_PATH = Path("data/Data_For_Test.json")
OUTPUT_PATH = Path("results/celestrak_baseline_comparison.json")

scenario_gen = functools.partial(from_celestrak_json, filepath=DATA_PATH)

results = {}
for name in ["random", "nearest", "risk_weighted"]:
    print(f"Evaluating {name} on CelesTrak (100 episodes)...")
    results[name] = run_policy(
        name=name,
        episodes=100,
        seed=42,
        targets=8,
        fuel=12000.0,
        max_steps=50,
        scenario_generator=scenario_gen,
    )
    r = results[name]
    print(f"  Cleared: {r['avg_cleared']:.2f}, DV: {r['avg_delta_v']:.1f}, "
          f"Rate: {r['avg_cleared']/8*100:.1f}%, FPT: {r['fuel_per_target']:.1f}")

# Save compact results
save = {}
for name, m in results.items():
    save[name] = {
        "avg_cleared": m["avg_cleared"],
        "clearance_rate": f"{m['avg_cleared']/8*100:.1f}%",
        "avg_delta_v": round(m["avg_delta_v"], 1),
        "fuel_per_target": round(m["fuel_per_target"], 1),
        "std_delta_v": round(m["std_delta_v"], 1),
    }
OUTPUT_PATH.write_text(json.dumps(save, indent=2), encoding="utf-8")
print(f"\nResults saved to {OUTPUT_PATH}")
