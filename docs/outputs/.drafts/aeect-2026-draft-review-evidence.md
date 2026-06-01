# Evidence Notes: aeect-2026-draft

## Inspected Sources
- Paper draft: `AEECT_2026_Paper_Draft.md`
- Local guideline excerpts:
  - `iadc_guidelines_excerpt.md`
  - `nasa_esa_reference_excerpt.txt`
  - `debris_mitigation_best_practices.md`

## Key Claims and Observations (Paper)

### Abstract / Intro
- Claims a "novel hybrid dual-AI framework" with MaskablePPO + BM25 RAG advisor.
- Reports **"86.3% Delta-V saving over random baselines"** (Abstract).
- Cites debris population numbers: **36,500 objects >10 cm** and **1 million fragments 1–10 cm** in LEO (Intro, ref [12]).

### Contributions
- Action-masked PPO for multi-target sequencing; invalid actions masked by fuel and target state.
- 3D Keplerian simulator with four-component Delta-V approximation (size, eccentricity, plane, apsidal).
- BM25-based RAG advisor over NASA-STD-8719.14 and IADC guidelines.

### Methods Details
- Observation space: 155 dimensions (11 spacecraft + 12 targets × 12 features).
- Action masking sets invalid logits to −∞; masked softmax ensures zero probability.
- Reward: intercept bonus (+25), risk weighting (+30·r_i), fuel penalty (−0.001·ΔV), terminal bonuses.

### Experimental Results (Tables II–V)
- **Table II (LEO preset, 100 episodes):**
  - Random baseline Avg ΔV 5345.3±4521; targets cleared 0.93.
  - Nearest-Neighbor Avg ΔV 8964.2±2384; targets cleared 2.86.
  - Risk-Weighted Greedy Avg ΔV 8429.1±2402; targets cleared 2.89.
  - MaskablePPO Avg ΔV 8292.6±2506; targets cleared 2.36.
  - Fuel efficiency (m/s/target) is reported but interpretation unclear relative to targets cleared.
- Text claims: "MaskablePPO consumes significantly less fuel than greedy Nearest-Neighbor" and **"56.2% target clearance improvement over random baseline"**.
- **Table III (Celestrak, 19,721 objects):**
  - PPO v4 base: targets 1.79; task accuracy 22.4%; avg ΔV 8648.8.
  - PPO finetuned: targets 1.90; task accuracy 23.8%; avg ΔV 8612.3.
  - Text claims: 6.1% higher target collection and 6.3% fuel-per-target improvement.
- **Table IV (RAG queries):** sample queries with BM25 scores and excerpt summaries.
- **Table V (continuous low-thrust):** single transfer simulation (600→800 km, 53°→54°) with TOF 48 h, 29.8 revolutions, ΔV 3456 m/s.

### Reproducibility/Availability Statements
- No code repository or dataset link provided in the draft.
- Training hyperparameters (PPO settings, network architecture), simulator source, random seeds, and data splits are not specified.
- Real-world dataset usage (Celestrak) mentioned without direct URL, filtering criteria, or preprocessing steps.

## Cross-checks Against Local Guideline Excerpts (RAG Evidence)
- `iadc_guidelines_excerpt.md` confirms **25-year** post-mission disposal guidance (supports Table IV query about "LEO disposal timeline").
- `nasa_esa_reference_excerpt.txt` states fuel margin for collision avoidance and end-of-life operations (partial support for Table IV "conjunction fuel protocol").
- `debris_mitigation_best_practices.md` mentions prioritizing high-mass objects and fuel margin guidance (supports Table IV "high-risk debris prioritization").
- These are secondary excerpts, not the full standards, limiting verification of compliance claims.

## Potential Consistency Issues Observed
- Abstract claim of **86.3% Delta-V saving over random** conflicts with Table II values (MaskablePPO ΔV 8292.6 vs random 5345.3 → higher, not lower).
- Fuel efficiency (m/s/target) for MaskablePPO (3513.8) is worse than Risk-Weighted Greedy (2916.6), yet text asserts superior fuel efficiency.
- “56.2% target clearance improvement” appears inconsistent with Table II (2.36 vs 0.93 targets cleared suggests ~154% improvement).

## Additional Notes
- F1-score and task accuracy are used but definitions for sequential planning tasks are thin; precision fixed to 1.0 by definition, which reduces interpretability.
- Claims of “high-fidelity” simulator are not validated against astrodynamics benchmarks or error analysis.
