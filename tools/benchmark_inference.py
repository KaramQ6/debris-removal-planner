"""Benchmark single-step inference latency of the trained MaskablePPO model.

Loads the model, generates random observations and action masks matching the
OrbitDebrisEnv observation/action spaces, and times 10,000 calls to
``model.predict()``.  Results are saved to ``results/inference_benchmark.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

# Prefer finetuned model, fall back to base
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_debris_finetuned.zip")
if not os.path.isfile(MODEL_PATH):
    MODEL_PATH = os.path.join(MODELS_DIR, "ppo_debris.zip")
if not os.path.isfile(MODEL_PATH):
    sys.exit(f"ERROR: No model found in {MODELS_DIR}")

OUTPUT_PATH = os.path.join(RESULTS_DIR, "inference_benchmark.json")

# ---------------------------------------------------------------------------
# Environment geometry (must match OrbitDebrisEnv defaults)
# ---------------------------------------------------------------------------
MAX_TARGETS = 12
TARGET_COUNT = 8
OBS_DIM = 11 + MAX_TARGETS * 12  # 155


def main() -> None:
    from sb3_contrib import MaskablePPO

    print(f"Loading model from {MODEL_PATH} …")
    model = MaskablePPO.load(MODEL_PATH)

    rng = np.random.default_rng(42)
    n_iters = 10_000

    # Pre-generate random observations and action masks
    observations = rng.uniform(-1.0, 1.0, size=(n_iters, OBS_DIM)).astype(np.float32)
    masks = np.zeros((n_iters, MAX_TARGETS), dtype=bool)
    for i in range(n_iters):
        # At least one action must be valid; activate a random subset of targets
        n_valid = rng.integers(1, TARGET_COUNT + 1)
        valid_indices = rng.choice(TARGET_COUNT, size=n_valid, replace=False)
        masks[i, valid_indices] = True

    # Warm-up (exclude from timing)
    for _ in range(100):
        model.predict(observations[0], action_masks=masks[0], deterministic=True)

    # Timed run
    latencies_us: list[float] = []
    print(f"Running {n_iters} inference calls …")
    for i in range(n_iters):
        t0 = time.perf_counter()
        model.predict(observations[i], action_masks=masks[i], deterministic=True)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    arr = np.array(latencies_us)
    mean_us = float(np.mean(arr))
    std_us = float(np.std(arr))
    median_us = float(np.median(arr))
    p95_us = float(np.percentile(arr, 95))
    p99_us = float(np.percentile(arr, 99))

    results = {
        "model_path": os.path.basename(MODEL_PATH),
        "n_iterations": n_iters,
        "mean_latency_us": round(mean_us, 2),
        "std_latency_us": round(std_us, 2),
        "median_latency_us": round(median_us, 2),
        "p95_latency_us": round(p95_us, 2),
        "p99_latency_us": round(p99_us, 2),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Inference Benchmark Results ===")
    print(f"  Model        : {results['model_path']}")
    print(f"  Iterations   : {n_iters:,}")
    print(f"  Mean latency : {mean_us:.2f} ± {std_us:.2f} µs")
    print(f"  Median       : {median_us:.2f} µs")
    print(f"  P95          : {p95_us:.2f} µs")
    print(f"  P99          : {p99_us:.2f} µs")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
