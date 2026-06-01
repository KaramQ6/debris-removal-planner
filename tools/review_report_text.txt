MANUSCRIPT REVIEW REPORT
Title: Autonomous Propellant-Constrained Multi-Target Space Debris Removal Planning via
Action-Masked Deep Reinforcement Learning and RAG Operational Advisory
Authors: Karam Khasawneh
Field: Aerospace Engineering / Space Debris Removal and Orbital Trajectory Optimization
Date: May 26, 2026
Session: 22f84c29-46d4-4811-9259-7f1a0a1d3d1c
OVERALL ASSESSMENT
Overall Quality
65/100
Recommendation: Major Revisions
AI Confidence Level: 91%
CATEGORY SCORES
Methodology
58/100
Statistics
72/100
Citations
55/100
Novelty
61/100
Logic
62/100
Ethics
82/100
Reproducibility
82/100
Writing
62/100
Structure
35/100
STRUCTURAL ANALYSIS
Completeness
35/100
Missing sections: discussion, abstract, methods, results
[WARNING] Missing required section: Discussion
[WARNING] Missing required section: Abstract
[CRITICAL] Missing required section: Methods
[CRITICAL] Missing required section: Results
[INFO] No figures or tables detected in the manuscript.
Detected Sections
introduction · 521 words (p. 1)
literature_review · 4,181 words (p. 2)
acknowledgments · 46 words (p. 7)
references · 617 words (p. 7)
Total words: 5,365 | Figures: 0 | Tables: 0
MAJOR CONCERNS
1. [MAJOR] (p. 1) The abstract frames the system as competitive with greedy heuristics, but the reported
results show the learned policy clears fewer targets than both greedy baselines in both evaluation settings.
This is not a direct contradiction, but the wording risks overstating competitiveness without clarifying that
Page 1 of 9

performance is inferior on the main metric.
2. [MAJOR] (p. 4) The paper claims the learned policy enables online adaptability under dynamic catalog
updates, but the results explicitly say this was not validated. The manuscript presents this as a practical
advantage despite lacking empirical evidence in the experiments.
3. [MAJOR] (p. 4) The introduction states the framework addresses complex dynamic target catalogs and
variable perturbations, yet the evaluation is conducted mainly in static rollout settings and the paper admits
dynamic perturbation testing is still future work. The objective is broader than the validated analysis.
4. [MAJOR] (p. 6) The conclusion says the agent is suitable for online re-planning scenarios, but the paper
does not provide a formal online replanning experiment or stress test under live updates. This is a
conclusion overreach relative to the presented data.
5. [MAJOR] (p. 6) The continuous low-thrust section presents a preliminary feasibility result, but it is based
on a single simulated transfer with no comparative baseline, sensitivity analysis, or validation against
high-fidelity dynamics. The claim of compatibility with discrete sequencing outputs is therefore only weakly
supported.
6. [MAJOR] (p. 1) The RAG advisory is described as supporting regulatory compliance decisions, but the
paper does not evaluate retrieval precision, clause accuracy, or operator decision impact. The retrieval
examples show plausible snippets, yet they do not demonstrate compliance correctness.
7. [MAJOR] (p. 4) The manuscript reports a nonparametric two-sided Mann-Whitney U test, which is
reasonable for comparing independent rollout outcomes, but it should clarify whether the test assumptions of
independence were met across the 100 rollouts and whether paired or repeated-scenario structure existed.
8. [MAJOR] (p. 4) P-values are reported only as thresholds for the main comparisons, and no adjustment for
multiple pairwise testing is mentioned despite comparing MaskablePPO against several baselines; this can
inflate false-positive risk.
9. [MAJOR] (p. 4) Sample size justification and statistical power are not provided for the rollout experiments,
so it is difficult to judge whether the study was adequately powered for the observed differences or for the
null findings relative to greedy baselines.
10. [MAJOR] (p. 4) The sample size for evaluation is partially stated, but justification is incomplete. The
paper reports 100 rollouts and 30 queries, yet provides no rationale for why these numbers are sufficient to
estimate performance with desired precision or detect meaningful differences.
11. [MAJOR] (p. 4) The comparison includes relevant control/baseline methods, which strengthens the
benchmarking. However, the baselines are narrow and mostly heuristic, and there is no ablation separating
the effects of the action mask, reward shaping, environment fidelity, and RAG layer on overall performance.
12. [MAJOR] (p. 4) Confounding is only partly addressed. Although the manuscript compares methods on
the same scenarios, it does not clearly control for initialization, scenario difficulty, target ordering, or
training/evaluation leakage between synthetic and CelesTrak-derived cases, all of which could materially
affect the reported advantage.
13. [MAJOR] (p. 2) The methodological validity is mixed. The masked RL formulation is coherent for discrete
target sequencing, but several orbital mechanics approximations are described as intermediate-fidelity
surrogates and are acknowledged as such, meaning results should not be interpreted as high-fidelity mission
validation.
14. [MAJOR] (p. 4) A power calculation or sample size justification is absent. Statistical testing is reported,
but the manuscript does not explain the number of rollouts needed to achieve adequate power or the
uncertainty expected for the performance metrics.
15. [MAJOR] (p. 5) The paper transparently notes one important limitation of the RAG module: no labeled
retrieval evaluation was performed. That omission leaves the advisory component insufficiently validated for
the claimed operator-support role.
16. [MAJOR] (p. 6) Generalizability is limited because the policy is trained and evaluated on a specific
surrogate environment, fixed fuel budgets, and a small set of target counts and mission settings. The
reported gains may not transfer to different orbital regimes, larger catalogs, perturbation-rich dynamics, or
operational constraints not modeled here.
17. [MAJOR] (p. 6) The manuscript itself acknowledges key limitations, but they are under-discussed relative
to the strength of the claims. In particular, online adaptability under dynamic perturbations and formal RAG
Page 2 of 9

precision remain unverified despite being central to the abstracted contribution.
18. [MAJOR] (p. 1) The manuscript cites core background claims on debris growth and collision cascading
appropriately, but several technical claims in the method section are asserted without citation, especially the
custom 3D surrogate dynamics and reward design.
19. [MAJOR] (p. 1) The paper contains a likely internal inconsistency in the reported clearance performance:
the abstract claims 2.36 targets cleared on average, while Table II and the later ablation section report 2.22
and 2.36 for MaskablePPO respectively, suggesting unstable or inconsistent reporting.
20. [MAJOR] (p. 3) The action-masking formalism is central to the method, but the manuscript does not cite
the specific implementation or theoretical justification for masking invalid actions in PPO beyond a general
preprint, and the claim that masking guarantees physically valid sampling is stronger than the cited support.
21. [MAJOR] (p. 7) The reference list appears mostly complete and internally formatted in a consistent
IEEE-like style, but there are duplication issues: Stable-Baselines3 is listed twice as [8] and [16], and some
references are not uniformly formatted (e.g., differing treatment of arXiv and conference items).
22. [MAJOR] (p. 7) There is a potentially fabricated or inconsistent reference pattern in the citation of
NASA-STD-8719.14B and the IADC guideline versioning, because the manuscript mixes standard/report
naming conventions without clear publication details and with ambiguous revision metadata.
23. [MAJOR] (p. 1) The RL novelty is limited because the core ingredients·PPO, action masking, and
debris-sequencing MDP formulation·are each well precedented in both reinforcement learning and active
debris removal planning. The manuscript adds a different simulator and action constraints, but not a
fundamentally new learning paradigm.
24. [MAJOR] (p. 1) The RAG component is conceptually useful but not especially novel: it uses standard
BM25 retrieval over two guideline documents, and the authors explicitly state that no formal retrieval
evaluation was conducted. This reads more as an engineering integration than a new methodological
contribution.
25. [MAJOR] (p. 4) The paper is transparent that the learned policy does not beat the strongest greedy
baselines in the reported static evaluations. That substantially limits the practical novelty of the RL
contribution, because the main empirical claim is competitive execution speed rather than superior solution
quality.
26. [MAJOR] (p. 4) The authors acknowledge an important gap in validation: adaptability under dynamic
perturbations is not yet demonstrated, and retrieval quality has not been benchmarked against labeled
ground truth. These omissions weaken the support for the paper·s claimed operational value.
27. [MAJOR] (p. 4) The main reproducibility barrier is that the manuscript does not provide a clearly
enumerated inclusion/exclusion protocol for constructing the synthetic scenarios or selecting the
CelesTrak-derived missions, leaving scenario sampling partly underspecified.
28. [MAJOR] (p. 5) Although the dataset source is identified, the paper does not fully specify the exact
CelesTrak subset, query date, filtering rules, or the exact objects used in each run, so the real-world
evaluation cannot be reconstructed precisely from the manuscript alone.
29. [MAJOR] (p. 5) The manuscript acknowledges that the retrieval component was not formally
benchmarked against labeled ground truth, so the reported RAG performance is limited to latency and
illustrative examples rather than validated retrieval quality.
30. [MAJOR] (p. 5) The manuscript states that formal evaluation of the retrieval component was not
conducted, which is an important transparency limitation and should be highlighted as an incomplete
validation of the advisory subsystem.
31. [MAJOR] (p. 1) The abstract is broadly aligned with the paper, but it overstates the system's success by
emphasizing competitiveness while the results show the learned policy underperforms greedy baselines on
the main clearance metric. This weakens abstract accuracy and calibration.
32. [MAJOR] (p. 1) The manuscript generally follows a clear sectioned structure, but several paragraphs
repeat wording or are duplicated across page breaks, which disrupts flow and makes the exposition feel
unpolished. This is especially noticeable in the abstract and around section transitions.
33. [MAJOR] (p. 1) There are noticeable grammar and formatting issues that impede smooth reading,
including duplicated fragments, broken line wraps, and inconsistent mathematical typesetting. These issues
do not usually obscure meaning, but they reduce professionalism and make some formulas hard to parse.
Page 3 of 9

34. [MAJOR] (p. 4) The results discussion is generally understandable and logically ordered, but the paper
should better distinguish empirical findings from interpretive claims. For example, the narrative emphasizes
deployment advantages even where benchmark performance is weaker than baselines.
MINOR CONCERNS
1. (p. 5) The manuscript states that the RL agent improves from about 1.0 to 2.5 cleared targets during
training, but the final evaluated performance on CelesTrak is only 1.90 targets cleared. The relationship
between training convergence and final generalization is not clearly explained.
2. (p. 3) The reward design includes a strong terminal bonus and risk weighting, but the paper does not
analyze whether the reward shaping biases the policy toward clearing risk-weighted easy targets rather than
optimizing mission utility under fuel constraints. This leaves a logical gap in the optimization argument.
3. (p. 4) The manuscript reports some confidence intervals, which is good practice, but effect sizes are
largely absent; readers are left without standardized magnitude measures for the statistical comparisons
beyond means, SDs, and bootstrap intervals.
4. (p. 4) The manuscript uses terms that may overstate inferential strength, such as claiming statistical
superiority from a nonparametric comparison without discussing practical relevance or the direction of the
main performance metric relative to baseline heuristics.
5. (p. 4) The manuscript uses a computational simulation and benchmark comparison design rather than an
experimental or clinical study. It evaluates a trained policy against heuristic baselines in synthetic and
catalog-derived scenarios, which is appropriate for the stated planning problem but limits causal inference.
6. (p. 6) The reproducibility section is a strength because it specifies software stack, random seeds, and
hardware. This improves transparency and supports independent replication, although full reproducibility
would also benefit from pinned dataset snapshots and exact scenario-generation code.
7. (p. 2) The literature review is reasonably cited for RL, action masking, RAG, and ADR planning, but
several foundational astrodynamics formulas are presented without direct support to primary sources beyond
a general textbook citation, which weakens traceability for the custom Delta-V surrogate.
8. (p. 2) The manuscript relies heavily on a small set of self-referential or closely related citations, especially
[9] and [16], with the same author list appearing twice in the references and the article citing its own
proposed method extensively. This suggests a moderate-to-high self-citation rate.
9. (p. 4) The RAG advisor section is minimally supported by citations to RAG and standards, but the paper
does not cite any retrieval-evaluation methodology or benchmark source while claiming operational
usefulness from BM25 ranking and cosine tie-breaking.
10. (p. 4) The experimental section lacks citations for the statistical testing choices and for several
quantitative claims about benchmark scenarios, making it difficult to assess whether the comparison protocol
is standard and reproducible.
11. (p. 7) Several references are older than 10 years in a fast-moving field and should be justified as
foundational rather than current: Kessler 1978, Vallado 2013, Petropoulos 2004, and Liou 2004. These are
acceptable as classic references, but the manuscript would benefit from more recent ADR and RL planning
literature.
12. (p. 2) The manuscript cites some credible, field-appropriate sources, but it does not sufficiently represent
alternative perspectives on ADR planning, such as operations research, mixed-integer optimization, or
classical trajectory optimization literature beyond a few broad references.
13. (p. 1) The manuscript·s main novelty claim is a hybrid framework that combines action-masked PPO for
multi-target debris sequencing with a separate BM25-based RAG advisor for regulatory compliance. The RL
core is positioned as a constant-time re-planning planner, while the RAG layer is framed as an
operator-facing decision-support component.
14. (p. 6) The manuscript·s strongest novelty is likely the packaging of multiple known elements into a single
operationally oriented system: masked RL for target ordering, a 3D Keplerian surrogate, local regulatory
retrieval, and a low-thrust feasibility add-on. This is an incremental systems contribution rather than a
transformative algorithmic advance.
15. (p. 2) The related-work positioning is directionally reasonable but not fully convincing, because the
comparison table mainly describes broad limitations rather than directly contrasting against closest prior
Page 4 of 9

multi-target ADR RL and sequence-planning papers. The novelty argument would be stronger with a sharper
direct comparison to the most relevant state of the art.
16. (p. 5) The manuscript provides a fairly detailed algorithmic description, including state/action design,
masking logic, reward shaping, retrieval scoring, hyperparameters, and environment setup, which supports
replication of the main computational workflow.
17. (p. 5) The source code is explicitly stated to be available in a public GitHub repository, which
substantially improves reproducibility of both the RL and RAG components.
18. (p. 6) The manuscript specifies software versions and computational environment details, including
operating system, Python version, core libraries, random seed, and hardware, which is helpful for
reproducing experiments.
19. (p. 4) The experimental section reports exact p-values, Mann-Whitney U statistics, and bootstrap
confidence intervals for several comparisons, improving statistical transparency.
20. (p. 7) No protocol registration or pre-registration is mentioned, which is not unusual for engineering work
but still reduces transparency about whether analyses were specified in advance.
21. (p. 6) The manuscript gives some computational environment details, but it does not specify critical
implementation choices such as exact package build hashes, dependency lockfiles, GPU driver/CUDA
versions, or the exact repository commit used for the reported results.
22. (p. 7) Material and equipment specifications are only partially complete: the CPU and GPU model are
named, but no details are provided on memory, storage, or other system configuration that could affect
runtime benchmarking.
23. (p. 7) The manuscript includes clear declarations for funding, conflicts of interest, and author
contributions, which supports transparency. It also provides a public code repository to aid reproducibility.
24. (p. 5) A public repository is provided for the codebase and evaluation scripts, which is a strong
reproducibility feature.
25. (p. 7) The paper does not contain an ethics committee, IRB, or human-subjects approval statement. This
is likely acceptable for a physical sciences simulation paper, but the absence should be explicitly clarified as
not involving human subjects or personal data.
26. (p. 1) No informed consent statement is present. Because the work appears to be simulation-based and
does not involve identifiable human participants, this is probably not applicable, but the manuscript should
state that no human subjects were involved.
27. (p. 1) No animal ethics or IACUC statement is present. This is not applicable to the described research,
but the manuscript should explicitly indicate that no animal experimentation was performed.
28. (p. 4) The RAG component uses NASA and IADC documents, but the manuscript does not describe
specific privacy or anonymization measures for any data. Since the inputs are public orbital catalogs and
standards documents, privacy risk appears low; still, the manuscript should state that no personal data were
collected or stored.
29. (p. 1) A clinical trial registration statement is not applicable because the manuscript is not a clinical or
biomedical study.
30. (p. 3) The technical writing is accessible to a specialist audience, but the manuscript occasionally relies
on dense jargon and long explanatory insertions that slow readability. In places, definitions are
overexplained in a way that interrupts the mathematical presentation.
31. (p. 5) Sentence structure varies reasonably well overall, but some sentences become excessively long
and packed with clauses, reducing clarity. Shorter sentences would improve readability in methodological
and results sections.
32. (p. 5) Terminology is mostly consistent, but the manuscript alternates among several near-synonyms and
naming conventions for the same components, such as MaskablePPO, RL Agent, trained policy, and
MaskablePPO (Fine-tuned). More disciplined terminology would make comparisons easier to follow.
33. (p. 2) The figures/tables are represented only by captions and embedded tables in the excerpt, and the
captions are informative but sometimes verbose or redundant. They generally convey the content, though
the prose would benefit from tighter, more standardized table titles and notes.
Page 5 of 9

STRENGTHS
· The manuscript is internally transparent in several places, explicitly reporting that greedy baselines
outperform the RL policy on the main clearance metric.
· It includes useful reproducibility details such as training hyperparameters, hardware, random seed, and
code release information.
· Provides nonparametric comparisons and bootstrap confidence intervals for key RL outcomes.
· Explicitly discloses at least one important validation gap for the RAG system, avoiding overclaiming there.
· Clear computational benchmarking framework with multiple baselines and reported statistical comparisons.
· Good reproducibility signaling through explicit hyperparameters, software versions, random seed, and code
release statement.
· Core claims about debris hazard, ADR motivation, PPO, and RAG are generally supported with relevant
citations.
· The manuscript includes a reasonably broad reference list spanning astrodynamics, RL, and space debris
policy sources.
· Combines sequencing, action masking, and operator guidance into one coherent workflow.
· Provides reproducibility details, baseline comparisons, and public code release.
· Public code repository is explicitly provided.
· Software versions, random seed, hardware, and training hyperparameters are reported.
· Funding, conflict of interest, and author contribution statements are present.
· A public code repository and detailed hyperparameters are provided to support reproducibility.
· The manuscript has a clear high-level organization with conventional sectioning from introduction through
methods, results, and conclusion.
· The writing is generally understandable to a technical aerospace/AI audience, and the paper provides useful
definitional context for key concepts and metrics.
RECOMMENDED IMPROVEMENTS
1. Tone down claims of competitiveness and online adaptability unless supported by dynamic replanning
experiments and formal comparisons against stronger baselines.
2. Add dedicated validation for the RAG component and the low-thrust feasibility claim, including labeled
retrieval metrics and multi-case continuous-dynamics evaluation.
3. Report corrected p-values or a multiple-comparison procedure (for example, Holm or
Benjamini-Hochberg) for all pairwise baseline tests.
4. Add standardized effect sizes and a brief power/sample-size rationale for the rollout study, and clarify
whether evaluation rollouts were independent or repeated across matched scenarios.
5. Add an a priori sample-size or power justification for rollout evaluations and retrieval-query experiments,
including effect sizes and uncertainty targets.
6. Strengthen validation with broader ablations and stress tests: separate action masking, reward shaping,
simulator fidelity, and RAG retrieval quality; then evaluate on perturbed missions and held-out catalog
snapshots.
7. Add citations for the custom orbital surrogate equations, action masking rationale, and statistical test
selection, and clearly distinguish foundational formulas from novel approximations.
8. Deduplicate repeated references, standardize all entries to a single citation style, and add more recent
ADR optimization and dynamic planning literature to improve diversity and currency.
9. Add a direct ablation and benchmark against the closest prior ADR sequencing RL methods, especially
those cited in the related work, using identical scenarios and metrics.
10. Demonstrate the claimed operational advantages with stronger evidence: dynamic catalog updates,
formal RAG retrieval evaluation, and end-to-end mission utility rather than only latency.
11. Add a replication appendix listing the exact CelesTrak snapshot date, filtering criteria, sampled target
IDs, and scenario-generation procedure, plus the repository commit hash and dependency lockfile.
12. Include a predefined evaluation protocol for the synthetic and real-world scenarios, and report formal
RAG metrics such as precision@k, recall@k, and MRR on a labeled regulatory test set.
Page 6 of 9

13. Add an ethics/compliance note stating that the study used only simulation and public orbital data, with no
human or animal subjects and therefore no IRB/IACUC approval required.
14. Expand the data availability section to specify exactly which datasets, code, trained weights, and
preprocessing scripts are publicly available and under what access conditions.
15. Revise the abstract and conclusion so that performance claims match the reported results more
precisely, especially where the RL agent underperforms the greedy baselines.
16. Perform a line edit to remove duplicated fragments, normalize terminology, and shorten overly long
sentences and formula explanations for smoother readability.
EDITORIAL SUMMARY
May 26, 2026
Karam Khasawneh
Department of Robotics and Artificial Intelligence
Jadara University
Irbid, Jordan
KaramQ5@ieee.org
Dear Karam Khasawneh,
Thank you for submitting your manuscript, ·Autonomous Propellant-Constrained Multi-Target Space Debris
Removal Planning via Action-Masked Deep Reinforcement Learning and RAG Operational Advisory.· The
paper addresses an important problem in aerospace engineering by combining action-masked deep
reinforcement learning with a retrieval-augmented advisory layer for propellant-constrained debris removal
planning, and it offers a computational framework that is potentially useful for sequence optimization and
mission support.
After review, the referees found several substantive issues that need to be addressed before the work can be
considered further. The main concern is calibration of claims relative to the evidence presented: the
manuscript·s abstract, introduction, and conclusion at times describe the method as competitive, adaptive, and
suitable for online replanning, yet the reported evaluations do not validate dynamic catalog updates, live
replanning, or superiority over greedy baselines on the principal clearance metric. In addition, the continuous
low-thrust section appears preliminary and would benefit from clearer framing as a feasibility demonstration
rather than a validated extension. Reviewers also noted an internal inconsistency in the reported clearance
performance, as well as several instances where statistical reporting is incomplete, including the lack of
multiple-comparison adjustment, effect sizes, and sample size or power justification. Methodologically, the
benchmarking design is appropriate for a simulation study, but the current baseline set is narrow, confounding
is not fully controlled, and key ablations separating the contributions of masking, reward shaping, environment
fidelity, and the RAG layer are missing. Citation coverage also needs strengthening in several technical
sections, particularly for custom surrogate dynamics, masking justification, and foundational astrodynamics
claims.
The manuscript is promising in its overall direction and has several strengths, including a generally clear
algorithmic description, public code availability, and useful reproducibility details such as software versions,
seeds, and hardware. However, the paper would benefit from more rigorous validation, more careful statistical
presentation, clearer scenario-selection criteria, and more precise alignment between claims and
demonstrated results. The writing and formatting issues identified by the reviewers should also be corrected
to improve readability and professionalism.
Given the scope of the revisions needed, I recommend **major revisions**. I encourage you to revise the
manuscript carefully and resubmit with more conservative claims, stronger experimental justification, clearer
statistical reporting, and improved consistency throughout the text.
Best regards,
CitedEvidence:
Page 7 of 9

INDIVIDUAL REVIEWER NOTES
Logical Consistency
The manuscript is technically ambitious and reasonably structured, but its strongest limitations are claims that extend
beyond the experiments actually reported. The main quantitative result is that the RL policy underperforms simple greedy
methods on target clearance, so the conclusions should be more carefully hedged and aligned with the evidence.
Statistical Analysis
The statistical reporting is generally functional and includes some good practices such as nonparametric testing and
bootstrap confidence intervals. However, the manuscript would benefit from clearer design details, multiple-comparison
control, effect sizes, and a power rationale to support stronger inference from the RL evaluation results.
Methodology
This is a solid computational benchmarking paper with a coherent RL formulation and useful reproducibility details, but it
remains a surrogate-level study rather than a validated mission-planning system. The main methodological gaps are the
absence of power justification, limited control of confounders, and incomplete validation of both the RL and RAG
components under realistic operational variability.
Citation Integrity
The manuscript has a credible citation base for its broad domain framing, but the reference practice shows moderate
weaknesses in completeness, diversity, and consistency. The biggest concerns are duplicated citations, several uncited
or lightly supported methodological claims, and some potentially inconsistent reference formatting; overall, the citation
quality is serviceable but not yet robust for a fast-moving aerospace RL/ADR topic.
Novelty Assessment
The manuscript presents a coherent engineering integration of masked PPO planning, a 3D orbital surrogate, and a local
RAG compliance advisor, but the individual technical ingredients are largely established in the literature. Its novelty is
therefore incremental rather than significant: useful as a system-level prototype, but not a transformative advance in ADR
planning. The strongest claims are not yet fully supported because the policy does not outperform the best heuristics and
some operational benefits remain unvalidated.
Reproducibility V2
The manuscript is moderately to strongly reproducible because it shares code, provides detailed model and
hyperparameter descriptions, and reports software/hardware versions and statistical tests. However, exact reconstruction
would still require some guesswork around scenario generation, dataset subsampling, and the RAG evaluation protocol,
so the work is not fully turnkey from the manuscript alone.
Ethical Compliance
This is a technically detailed and generally transparent simulation study in aerospace engineering. Core disclosure items
such as funding, conflicts of interest, author contributions, and code availability are present, but the manuscript would be
stronger with an explicit statement that no human or animal subjects were involved and with clearer documentation of the
data and retrieval-evaluation limitations.
Writing Quality
The manuscript is readable and technically coherent, with a clear structure and appropriate domain terminology for
aerospace/AI readers. However, it needs substantive prose polishing: duplicated text, long dense passages, and slightly
overstated claims reduce clarity and abstract fidelity. The overall writing quality is adequate, but not yet fully
publication-polished.
Editor Overview
May 26, 2026
Karam Khasawneh
Department of Robotics and Artificial Intelligence
Jadara University
Irbid, Jordan
KaramQ5@ieee.org
Dear Karam Khasawneh,
Page 8 of 9

Thank you for submitting your manuscript, ·Autonomous Propellant-Constrained Multi-Target Space Debris Removal
Planning via Action-Masked Deep Reinforcement Learning and RAG Operational Advisory.· The paper addresses an
important problem in aerospace engineering by combining action-masked deep reinforcement learning with a
retrieval-augmented advisory layer for propellant-constrained debris removal planning, and it offers a computational
framework that is potentially useful for sequence optimization and mission support.
After review, the referees found several substantive issues that need to be addressed before the work can be considered
further. The main concern is calibration of claims relative to the evidence presented: the manuscript·s abstract, introduction,
and conclusion at times describe the method as competitive, adaptive, and suitable for online replanning, yet the reported
evaluations do not validate dynamic catalog updates, live replanning, or superiority over greedy baselines on the principal
clearance metric. In addition, the continuous low-thrust section appears preliminary and would benefit from clearer
framing as a feasibility demonstration rather than a validated extension. Reviewers also noted an internal inconsistency in
the reported clearance performance, as well as several instances where statistical reporting is incomplete, including the
lack of multiple-comparison adjustment, effect sizes, and sample size or power justification. Methodologically, the
benchmarking design is appropriate for a simulation study, but the current baseline set is narrow, confounding is not fully
controlled, and key ablations separating the contributions of masking, reward shaping, environment fidelity, and the RAG
layer are missing. Citation coverage also needs strengthening in several technical sections, particularly for custom
surrogate dynamics, masking justification, and foundational astrodynamics claims.
The manuscript is promising in its overall direction and has several strengths, including a generally clear algorithmic
description, public code availability, and useful reproducibility details such as software versions, seeds, and hardware.
However, the paper would benefit from more rigorous validation, more careful statistical presentation, clearer
scenario-selection criteria, and more precise alignment between claims and demonstrated results. The writing and
formatting issues identified by the reviewers should also be corrected to improve readability and professionalism.
Given the scope of the revisions needed, I recommend **major revisions**. I encourage you to revise the manuscript
carefully and resubmit with more conservative claims, stronger experimental justification, clearer statistical reporting, and
improved consistency throughout the text.
Best regards,
CitedEvidence:
Generated by GPT Researcher Manuscript Reviewer | 2026-05-26 15:43 UTC
Page 9 of 9

