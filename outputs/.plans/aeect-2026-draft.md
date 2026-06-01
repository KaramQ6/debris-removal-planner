# Audit Plan: aeect-2026-draft

## Audit Scope
- **Target Paper**: `docs/AEECT_2026_Paper_Draft.md` (and `docs/AEECT_2026_Paper.tex`)
- **Target Repository**: `c:\Users\ASUS\OneDrive\المستندات\GitHub\debris-removal-planner` (debris-removal-planner)

## Claims & Codebases to Inspect
1. **Action Masking Logic**:
   - Claim: The PPO agent masks out invalid actions including already visited debris and targets exceeding fuel capacity.
   - Code location: `simulation/orbit_env.py` (specifically `action_mask()` and `action_masks()`).
2. **Risk-Weighted Greedy Baseline**:
   - Claim: Greedily intercepts based on a weighted linear combination of proximity and risk.
   - Code location: `simulation/policies.py` (specifically `risk_weighted_policy()`).
3. **RAG Advisory Layer**:
   - Claim: Integrates BM25 with term-frequency vector cosine similarity to advisory operators, operating as a parallel decoupled layer.
   - Code location: `rag/rag_system.py` (specifically `SimpleRAGAdvisor`).
4. **Celestrak target count & Evaluation Metadata**:
   - Claim: Evaluated on Celestrak dataset with accurate target counts (8 targets) and matching metrics.
   - Code location: `simulation/evaluate.py`, `results/finetuned_evaluation_summary.json`, `results/test_evaluation_summary.json`.
5. **Ablation baseline stats**:
   - Claim: Unmasked PPO fails due to sparse reward deadlocks (average cleared targets = 0.15).
   - Code location: `results/unmasked_ppo_evaluation_summary.json`.

## Verification Methods
- Surgical code analysis of the matching Python scripts.
- Execute linter checks using `tools/verify_paper.py`.
- Verify regenerated JSON evaluation summaries in `results/` for mathematical consistency.
