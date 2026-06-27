# Contribution Statement & Paper Outline (Paper 2) — REVISED

> **Why this revision exists:** the v1 headline ("robust to unknown target inertia, unlike the
> model-based baseline") is contradicted by the baseline's own published abstract — ARC's control
> law is explicitly designed to need no prior knowledge of target inertia — and by our own inertia
> sweep, where ARC does not degrade. See `literature_review.md` §3 for the corrected gap. The pivot
> below keeps the same infrastructure (env, training, eval) and the same baseline; only the framing
> and the headline metric change.

## One-paragraph claim (the thing reviewers must believe)
We present a reinforcement-learning controller for the **post-grasp detumbling** of a
non-cooperative tumbling target captured by a **free-floating space manipulator**, evaluated against
**ARC (active resistance control; Wang et al., 2025)** — a force-feedback baseline whose control law
is, by design, agnostic to the target's mass and inertia. Our inertia sweep (0.5×–2× nominal)
confirms this design goal: ARC's detumbling performance does not collapse as inertia error grows,
and neither does our policy's. The two controllers diverge once an **unmodeled disturbance is
injected into the chaser's own base attitude** — a stand-in for unmodeled ACS/thruster noise, not
target-side uncertainty. ARC's control law tracks contact force only; it has no term that responds
to base-attitude error, and the resulting attitude excursion (and the ACS fuel needed to correct it)
grows with target mass in our experiments. Our policy is trained to reject this disturbance directly,
reducing peak induced base-attitude error by **[X]%** relative to ARC across the tested disturbance
range (0.05–0.12 rad/s) at matched target mass. We frame this as the first comparison of a learned
and a model-based contact-detumbling controller under **chaser-side** disturbance, a setting the
model-based literature does not address and the pre-contact RL literature only addresses before
contact is established.

*(`[X]%` is a placeholder — pull the actual delta from the experiment table before this goes in the
abstract. If the delta is small or noisy, say so; do not round up to make a cleaner number.)*

## Why this is top-tier shaped
- **Physical novelty**, not a reward tweak: contact + free-floating coupling + chaser-side disturbance.
- **Strong, correctly-characterized baseline** (ARC) — fixes Paper 1's core weakness, and we no longer
  mischaracterize what the baseline does or doesn't handle.
- **Disturbance narrative is the genuinely contested one.** The inertia narrative was not contested —
  ARC's abstract already claims it. Nobody currently claims to handle base-attitude disturbance during
  contact-phase detumbling; that's the actual gap (see lit review).
- **Space-specific metric** (base-attitude disturbance = chaser ACS fuel) reviewers in OOS care about,
  now correctly positioned as the headline rather than a secondary metric.
- **Honest parity result on inertia** strengthens credibility instead of weakening it: it pre-empts the
  "isn't ARC already inertia-robust?" reviewer question by answering it directly, in the paper, with data.

## Falsifiable success criteria (decide before writing the abstract)
1. **Headline.** Under the base-attitude disturbance sweep (0.05–0.12 rad/s), the learned policy
   reduces peak (and cumulative) induced base-attitude error / ACS correction fuel by ≥ X% relative
   to ARC at matched target mass. Report the gap **as a function of target mass**, since the baseline's
   degradation was observed to scale with mass — a flat-X% claim would hide that structure.
2. **Parity check, not a baseline-failure claim.** Under the inertia sweep (0.5×–2× nominal, no added
   base disturbance), both ARC and the learned policy maintain detumble success and comparable
   contact-force tracking. State this explicitly as *confirming* ARC's claimed inertia-agnosticism,
   not contradicting it — this is what rules out "the gain is just inertia mismatch" as an alternative
   explanation for criterion 1.
3. **Conditional secondary claim — verify before including.** If the policy's observation space does
   *not* include a direct contact-force measurement (ARC's law requires one), report contact-force
   tracking parity with ARC without that sensor as a secondary, sensing-requirement contribution.
   Confirm the actual observation space before writing this into the contribution — do not assume it.
4. **Ablation.** Remove the disturbance-specific reward/observation channel → the base-attitude
   rejection advantage collapses toward ARC's level. This isolates the mechanism actually responsible
   for criterion 1's result (replaces the old "remove domain randomization → robustness collapses"
   ablation, which depended on the now-retracted inertia-collapse framing).
5. **Blocking prerequisite — fix before re-running anything.** Confirm whether the inertia
   domain-randomization call in MuJoCo (`mj_setConst`) was actually taking effect. If it was a no-op,
   any current results describing training as "via domain randomization over inertia" are unverified.
   Fix it, re-run, and report plainly what was and wasn't randomized — including if DR over inertia
   turns out not to matter once it's actually active, since ARC's own robustness may mean it doesn't.

## Paper outline (ICRA/IROS, 6–8 pages)
1. **Introduction** — debris + non-cooperative capture; the gap is chaser-side disturbance rejection
   during contact, not target-side inertia (ARC already covers that).
2. **Related Work** — Clusters A/B/C from `literature_review.md`; position against 2510.06566,
   Wei 2025, and ARC (Wang et al., *Nonlinear Dynamics* 2025) — characterized correctly this time.
3. **Problem Formulation** — free-floating dynamics, contact model, chaser-base disturbance model, MDP.
4. **Method** — policy, observation space (confirm force-sensing assumption), training, what is and
   isn't domain-randomized.
5. **Baselines** — ARC, re-implemented per its published control law (force-feedback, no target
   parameter ID — verify the re-implementation matches this, not a generic impedance controller).
6. **Experiments** — (a) base-attitude disturbance sweep [headline], (b) inertia sweep [parity check],
   (c) ablations.
7. **Limitations & Sim-to-real** — contact-model fidelity; disturbance model is a proxy for ACS noise,
   not validated against real ACS hardware; no hardware (state plainly).
8. **Conclusion**.

## Explicitly out of scope (to keep it tight)
- Orbital sequencing / target selection (that is Paper 1; one framing sentence only).
- Multi-arm, dual-arm, tether-net capture modalities.
- Real-hardware validation (named as future work, not claimed).
- **Inertia-robustness as the primary contribution** — superseded by the pivot above. Retained only
  as the parity ablation in criterion 2; do not lead the abstract or intro with it.

## Open items to close before drafting the abstract
- [ ] Pull the actual base-attitude-disturbance delta (criterion 1) from the results table.
- [ ] Confirm the RL observation space re: contact-force sensing (criterion 3).
- [ ] Verify/fix `mj_setConst` for inertia DR; re-run if it was a no-op (criterion 5).
- [ ] Re-implement-check ARC against its published control law, not a generic impedance baseline.
