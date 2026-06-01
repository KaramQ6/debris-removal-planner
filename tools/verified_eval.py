"""Re-run the LEO preset evaluation with statistical tests and verified numbers.
Produces results/verified_leo_evaluation.json with exact numbers for Table II.
"""
import sys, json, functools
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.evaluate import run_policy

EPISODES = 100
SEED = 7
TARGETS = 8
FUEL = 12000.0
MAX_STEPS = 50

results = {}
for name in ["random", "nearest", "risk_weighted", "ppo"]:
    print(f"Evaluating {name} (LEO presets, {EPISODES} episodes, seed={SEED})...")
    model_path = "results/models/ppo_debris.zip" if name == "ppo" else ""
    r = run_policy(
        name=name,
        episodes=EPISODES,
        seed=SEED,
        targets=TARGETS,
        fuel=FUEL,
        max_steps=MAX_STEPS,
        model_path=model_path,
    )
    results[name] = r
    c = np.array(r["all_cleared"])
    d = np.array(r["all_delta_v"])
    print(f"  Cleared: {c.mean():.2f} +/- {c.std():.2f}")
    print(f"  DeltaV:  {d.mean():.1f} +/- {d.std():.1f}")
    print(f"  FPT:     {r['fuel_per_target']:.1f}")

# Statistical tests: Wilcoxon signed-rank for paired matched scenarios
print("\n=== Statistical Comparisons (Wilcoxon signed-rank, two-sided) ===")
ppo_cleared = np.array(results["ppo"]["all_cleared"])
ppo_dv = np.array(results["ppo"]["all_delta_v"])
stat_results = {}

for name in ["random", "nearest", "risk_weighted"]:
    b_cleared = np.array(results[name]["all_cleared"])
    b_dv = np.array(results[name]["all_delta_v"])
    
    w_c, p_c = stats.wilcoxon(ppo_cleared, b_cleared, alternative='two-sided')
    w_d, p_d = stats.wilcoxon(ppo_dv, b_dv, alternative='two-sided')
    
    # Wilcoxon matched-pair rank-biserial correlation
    diff = ppo_cleared - b_cleared
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        ranks = stats.rankdata(np.abs(non_zero_diff))
        signs = np.sign(non_zero_diff)
        w_pos = np.sum(ranks[signs > 0])
        w_neg = np.sum(ranks[signs < 0])
        r_c = (w_pos - w_neg) / (w_pos + w_neg)
    else:
        r_c = 0.0
        
    print(f"\n  PPO vs {name}:")
    print(f"    Cleared: W={w_c:.1f}, p={p_c:.4e}, r={r_c:.3f}")
    print(f"    DeltaV:  W={w_d:.1f}, p={p_d:.4e}")
    
    stat_results[f"ppo_vs_{name}"] = {
        "cleared_W": float(w_c), "cleared_p": float(p_c), "cleared_r": float(r_c),
        "delta_v_W": float(w_d), "delta_v_p": float(p_d),
    }

# Confidence intervals (bootstrap 95% CI)
print("\n=== 95% Bootstrap Confidence Intervals ===")
ci_results = {}
for name in ["random", "nearest", "risk_weighted", "ppo"]:
    c = np.array(results[name]["all_cleared"])
    boots = [np.random.choice(c, size=len(c), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci_results[name] = {"cleared_ci_low": round(lo, 2), "cleared_ci_high": round(hi, 2)}
    print(f"  {name}: {c.mean():.2f} [{lo:.2f}, {hi:.2f}]")

# Save everything
output = {}
for name in ["random", "nearest", "risk_weighted", "ppo"]:
    r = results[name]
    c = np.array(r["all_cleared"])
    d = np.array(r["all_delta_v"])
    output[name] = {
        "avg_cleared": round(c.mean(), 2),
        "std_cleared": round(c.std(), 2),
        "avg_delta_v": round(d.mean(), 1),
        "std_delta_v": round(d.std(), 1),
        "fuel_per_target": round(r["fuel_per_target"], 1),
        "clearance_rate": f"{c.mean()/TARGETS*100:.1f}%",
        "ci_95": ci_results[name],
    }

output["statistical_tests"] = stat_results
output["evaluation_params"] = {
    "episodes": EPISODES, "seed": SEED, "targets": TARGETS,
    "fuel": FUEL, "max_steps": MAX_STEPS, "scenario": "LEO_medium_preset"
}

Path("results/verified_leo_evaluation.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8"
)
print("\nSaved to results/verified_leo_evaluation.json")
