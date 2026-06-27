# Session report — 2026-06-27 (Paper 2, part 3: 3-seed training + headline figure)

Continues `session_2026-06-27_part2.md`. Worked **only** inside `robotic_capture/`.
Closes "Next session" items #2 (≥3 seeds) and #4 (base-disturbance figure) from part 2.

## What was done
- **Trained seeds 1 & 2** for both conditions (seed 0 already existed), 120k steps each,
  matching seed 0's single-env config exactly so seeds stay comparable. Ran all 4 in parallel
  (32-core box; MuJoCo env-step is the CPU bottleneck, so per-process torch threads were capped
  at 6). Outputs: `runs/sac_{dr,nodr}_seed{1,2}.zip` + per-run `train_*_seed*.log`.
- **Headline figure** (`plot_base_disturbance.py`, new module): pooled paired sweep over
  scales {0.5,1,1.5,2} × spins {0.2,0.3,0.4} × seeds 0-4, with the 3 DR policies pooled into one
  RL-DR line. Reuses `eval.evaluate` / `eval.bootstrap_ci`. → `results/fig_base_disturbance.pdf`.

## Verified
- All 4 runs exit 0; final `ep_rew_mean`: DR seed1 90.6 / seed2 90.4 (seed0 was ~+91);
  no-DR seed1 92.1 / seed2 92.2 (seed0 ~+92.6). **Seed variance is small** — the result is stable.
- Figure base disturbance (rad/s), mean [95% bootstrap CI], RL-DR pooled over 3 seeds:

  | inertia | RL-DR | ARC | zero |
  |---|---|---|---|
  | 0.5× | 0.026 [.022,.030] | 0.034 [.019,.049] | 0.000 |
  | 1.0× | **0.016** [.012,.020] | 0.121 [.091,.153] | 0.000 |
  | 1.5× | **0.015** [.013,.017] | 0.226 [.171,.284] | 0.000 |
  | 2.0× | **0.020** [.017,.023] | 0.285 [.228,.338] | 0.000 |

  RL-DR is flat & low; ARC rises ~8-19× higher with inertia. Cleaner than the single-seed
  table in part 2 (pooling 3 seeds tightened the RL CIs).

## Next session
1. Optional: seed-variance band (per-seed means ± spread) as a separate panel, if a reviewer
   asks for it explicitly — the pooled CI already covers it for the headline.
2. **Throughput:** next full training round, feed SAC from `SubprocVecEnv` (~8-16 envs) to cut
   wall-time — env-step is the limiter, GPU sat ~36% (tiny MLP). Switch ALL seeds together.
3. Drop the figure into `docs/paper/main.tex` and update the pivoted abstract/title.
