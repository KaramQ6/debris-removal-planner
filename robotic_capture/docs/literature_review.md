# Phase 0 — Literature Review & Novelty Lock (Paper 2) — REVISED

*Scope: RL for free-floating space-manipulator capture and detumbling of non-cooperative
tumbling debris. Prepared to lock a defensible contribution for a top-tier robotics venue
(ICRA / IROS / RA-L).*

> **Correction note (read first):** the original §3 gap statement and the comparison table below
> mischaracterized ARC (the active-resistance-control baseline) as assuming known target dynamics
> and failing under unknown inertia. ARC's published abstract states the opposite — its control law
> "relies only on measured contact force and does not require prior knowledge of the target's
> inertial or contact parameters," specifically to eliminate target parameter identification
> (Wang, Liu & Cai, *Nonlinear Dynamics* 113, 2025, doi:10.1007/s11071-025-11381-z — verified
> directly against the publisher abstract). Our own inertia-sweep data corroborate this: ARC does
> not collapse as inertia error grows. The table and gap statement below are corrected accordingly.
> The genuine, uncontested gap is **chaser-side base-attitude disturbance during contact**, not
> target-side inertia.

---

## 1. The field, in three clusters

**Cluster A — High-level mission planning / target sequencing** *(this is Paper 1's territory, NOT ours).*
RL (incl. MaskablePPO), MCTS, greedy, and OR pick *which* debris to visit and the orbit-transfer
order. Recent: a Feb-2026 MaskablePPO multi-debris planner with co-elliptic transfers + refueling
claims up to ~2× greedy clearance — **directly relevant to Paper 1, flag for that paper's revision.**

**Cluster B — Pre-contact RL guidance / tracking** *(crowded).*
The bulk of RL space-manipulator papers. A free-floating arm tracks a capture point on a tumbling
target while minimizing **base disturbance** (the arm-base dynamic coupling) and avoiding
self-collision. Examples: TD3 on a 7-DOF KUKA on a free-floating base for *obstacle-free guidance*
(arXiv 2510.06566); TD3 tumbling-target capture trajectory planning with base-pose disturbance in
the reward (Wei et al., *Robotica* 2025); "two-axis matching" RL to track *fast* tumbling targets
(2024). **Key fact: these minimize base disturbance only during the pre-grasp approach phase, before
contact dynamics are engaged. None of them address disturbance once the arm and target are coupled
through contact.**

**Cluster C — Contact & detumbling** *(thinner, mostly model-based).*
Post-grasp stabilization of a tumbling target. Largely classical control: ARC ("active resistance
control" for contact detumbling of non-cooperative spacecraft, Wang et al., *Nonlinear Dynamics*
2025) — force-feedback, explicitly inertia-agnostic by design (see correction note above), but its
control law has no term addressing disturbance originating from the **chaser's own base**; it also
requires a direct contact-force measurement. On the RL side: a soft-capture phase study using
**tactile feedback** (arXiv 2409.12273) touches the contact moment but does not clearly model
free-floating base dynamics or report a model-based baseline.

---

## 2. Closest competitors (what they do / what they miss)

| Work | Phase | Free-float base | Contact dyn. | Inertia handling | Base-attitude disturbance rejection | Sensing | Gap they leave |
|---|---|:--:|:--:|---|---|---|---|
| Obstacle-free guidance, TD3 (arXiv 2510.06566) | pre-contact guidance | ✅ | ❌ | n/a | n/a | not stated | no grasp/detumble |
| Tumbling capture planning, TD3 (Wei 2025, *Robotica*) | pre-grasp trajectory | ✅ | ❌ | n/a | pre-grasp only (reward term) | not stated | no contact phase |
| Two-axis matching RL (2024) | track fast tumbler | ✅ | ❌ | n/a | pre-grasp only | not stated | tracking only |
| Soft-capture + tactile DRL (arXiv 2409.12273) | soft-capture contact | unclear | partial | not addressed | not addressed | tactile (contact) | weak base model, no model-based baseline |
| **ARC** (Wang et al. 2025, *Nonlinear Dynamics*) | detumbling | model-based | ✅ | **robust by design** (force-feedback, no target-parameter ID needed) | **not addressed** — control law assumes contact-force tracking is sufficient | requires measured contact force (F/T) | no rejection of chaser-side base-attitude disturbance; sensing-dependent |

**Pattern:** RL handles *reaching* the target, minimizing base disturbance only before contact.
Contact/detumbling is handled by ARC, which is already robust to target-side inertia uncertainty but
has no mechanism for chaser-side disturbance during contact, and depends on contact-force sensing.
**Chaser-side base-attitude disturbance during the contact phase is the gap that is actually open** —
target-side inertia uncertainty is not.

---

## 3. The gap (locked — corrected)

> The **post-grasp, contact-rich detumbling** of a tumbling target with a **free-floating
> manipulator**, under **disturbance originating from the chaser's own base** (unmodeled ACS/thruster
> noise, not target-side parameter uncertainty), evaluated against a **correctly-characterized
> model-based baseline** (ARC) and reported on **contact forces + base-attitude disturbance /
> induced ACS fuel** — is under-explored. ARC is, by its own design, robust to unknown target inertia
> via force feedback; it does not address disturbances originating from the chaser base, and its
> control law depends on direct contact-force measurement. Pre-contact RL guidance work minimizes
> base disturbance, but only before contact is established. We address the post-contact case.

This gap is defensible because it is *physical* (contact + chaser-base coupling), not just "another
reward function," and because it is genuinely uncontested in the literature — unlike the inertia
angle, which ARC's own abstract already claims to have solved.

---

## 4. Candidate novelty angles (ranked — corrected)

**A. ⭐ Recommended — Chaser-side base-attitude disturbance rejection during contact detumbling.**
A learned policy that, after grasp, rejects an unmodeled disturbance to the chaser's own base
attitude while detumbling the target — distinct from target-side inertia uncertainty, which ARC
already handles by design. Benchmarked against ARC. Headline result: the learned policy reduces
induced base-attitude error / ACS correction fuel relative to ARC, with the gap growing with target
mass. *Why it wins:* clean story, genuinely open gap, strong and correctly-characterized baseline,
space-specific metric (ACS fuel) reviewers in OOS care about. *Risk:* the disturbance model must be
a credible proxy for real ACS/thruster noise — justify the 0.05–0.12 rad/s range explicitly.

**B. Unified approach→grasp→detumble policy (single agent across phases).**
Most papers do one phase. A single policy (or options/hierarchy) spanning all three, with
phase-aware reward. *Why:* "full pipeline" appeal. *Risk:* reviewers may see each phase as
incremental unless one phase is clearly novel — pair with (A) as the novel phase, as before.

**C. Target-inertia parity check (demoted from headline to ablation).**
Was the original headline; now a supporting result. Report that the learned policy matches ARC's
already-claimed inertia-agnosticism across 0.5×–2× nominal inertia, with no added base disturbance.
*Why keep it:* pre-empts the obvious reviewer question ("doesn't ARC already handle this?") and rules
out inertia mismatch as an alternative explanation for the (A) result. *Why it can't lead:* ARC's
abstract already claims this property, and our own data confirm rather than challenge it.

**D. Reduced sensing requirement (conditional secondary angle).**
ARC's law requires measured contact force. If the policy's observation space does not include a
direct contact-force measurement, matching ARC's performance without that sensor is a real,
low-cost secondary contribution. *Confirm the actual observation space before claiming this* — do
not assume it from the architecture description alone.

**E. Thin hierarchical link to Paper 1 (selection → capture).**
High-level MaskablePPO selects target; low-level captures. *Why:* continuity. *Risk:* this is the
exact "weak integration" that hurt Paper 1 — keep as framing, **not** the core claim.

**Decision:** Lead with **A** (chaser-side base-attitude disturbance rejection), framed inside **B**
(the novel detumbling phase of a full pipeline), report **C** explicitly as a parity/fairness check
rather than a failure result, add **D** only after confirming the observation space, and drop **E**
to a single framing sentence.

---

## 5. Papers to obtain (PDFs the user must download — paywalled / blocked)

> I could not fetch these (paywall / 403). Priority ⭐ = needed to position the contribution.
> ARC's abstract is now verified (see correction note); the full text is still needed to check
> whether it discusses chaser base dynamics at all, and to confirm the re-implementation matches
> its actual control law rather than a generic impedance controller.

1. ⭐ **Review — Autonomous Space Robotic Manipulators for OOS & ADR**, *Space: Science & Technology* 2024. doi:10.34133/space.0291 — authoritative gap/positioning source.
2. ⭐ **Active resistance control: a contact control method for detumbling non-cooperative spacecraft by a robotic arm**, Wang, Liu & Cai, *Nonlinear Dynamics* 113, 24937–24965 (2025). doi:10.1007/s11071-025-11381-z — **our primary model-based baseline; abstract verified, full text still needed.**
3. ⭐ **Wei, Bai, Lu — Trajectory planning of free-floating space robot for non-cooperative tumbling target capture (DRL/TD3)**, *Robotica* 2025 — closest pre-grasp competitor.
4. **Two-axis matching RL for tracking tumbling targets**, *Acta Astronautica* 2024 (S1270963824006709).
5. **Laboratory Experimentation of Spacecraft Robotic Capture Using DRL Guidance**, *AIAA JGCD*, doi:10.2514/1.G006656 — real-hardware credibility reference.

Open-access (I can read these; listed for your library): arXiv 2510.06566, arXiv 2409.12273,
arXiv 2209.01434, arXiv 2403.07125 (tether-net), arXiv 1710.06537 (dynamics randomization),
arXiv 2602.17685 (MaskablePPO multi-debris — **relevant to Paper 1**), Frontiers 10.3389/fcteg.2024.1394668,
MDPI Aerospace 10/9/778 and 10/1/13.

---

## 6. Sources (links)
- https://spj.science.org/doi/10.34133/space.0291
- https://link.springer.com/article/10.1007/s11071-025-11381-z
- https://www.cambridge.org/core/journals/robotica/article/abs/trajectory-planning-of-freefloating-space-robot-for-noncooperative-tumbling-target-capture-based-on-deep-reinforcement-learning/465B066C4811D176A929F58022B3161B
- https://www.sciencedirect.com/science/article/abs/pii/S1270963824006709
- https://arc.aiaa.org/doi/abs/10.2514/1.G006656
- https://arxiv.org/abs/2510.06566
- https://arxiv.org/abs/2409.12273
- https://arxiv.org/pdf/2209.01434
- https://arxiv.org/pdf/2403.07125
- https://arxiv.org/pdf/1710.06537
- https://arxiv.org/html/2602.17685
- https://www.frontiersin.org/journals/control-engineering/articles/10.3389/fcteg.2024.1394668/full
- https://www.mdpi.com/2226-4310/10/9/778
- https://www.mdpi.com/2226-4310/10/1/13
