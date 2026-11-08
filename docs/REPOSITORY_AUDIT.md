# DroPhenix — repository audit

**Date:** 2026-11-07 · **Tree:** `D:\DroPhenix\DroPhenix-main` at commit `633c3f4`
**Nature of this audit:** static inspection plus targeted execution of the project's own code
against constructed inputs with known truth. **This is not an experimental validation certificate.**
No module, script or data file was modified.

Companion documents: `MODULE_INVENTORY.md` (what exists), `BUG_AND_RISK_REGISTER.md` (86 verified
findings), `ARCHITECTURE_PROPOSAL.md` (what to build), `SCIENTIFIC_VALIDATION_PLAN.md` (how to
prove it), `PRIORITISED_DEVELOPMENT_PLAN.md` (in what order).

---

## 1. The ten questions, answered

### 1. What currently exists?

A 50-module scientific Python library (28,911 lines), a 33-page Streamlit application
(`app.py`, 7,983 lines), a headless CLI, 44 validation suites with a master runner, 18 reference
dataset directories, a NAR-format manuscript with figures, and an extensive internal documentation
set including two prior self-audits. Full inventory in `MODULE_INVENTORY.md`.

This is not a prototype. It is a substantial, unusually self-critical research codebase that
already does several things most behavioural-analysis software does not.

### 2. What actually works?

**Verified by execution:** `validation/run_all.py` reports **42 suites, 0 failures, exit 0**, with
≥663 explicit checks. Eight suites are anchored to published raw data — about 40,000 individual
flies across lifespan, physiological decline, larval locomotion, grooming and climbing — and
independent cross-checks run against `lifelines`, `statsmodels`, `astropy` `LombScargle`
(agreement to 1e-12) and exact combinatorics.

**What that establishes, precisely:** the mathematics of the audited paths agrees with reference
implementations and with published values. **What it does not establish:** that the paths a user
actually reaches are the audited ones. Four of them are not, and that is where every P0 defect
lives.

### 3. What is scientifically correct?

The following are correct, externally cross-checked, and must survive any refactor:

- Kaplan–Meier, log-rank with BH correction, median survival CI, Cox — on the `statsmodels`
  numerical core, matched against `lifelines` and the Freireich 1963 trial.
- Log-logistic dose-response fitting with profile-likelihood intervals — agrees with R's `drc`
  `ryegrass` fit to ≤ 9e-6.
- Cosinor estimation — matches `astropy` `LombScargle` coefficient for coefficient to 1e-12.
- The hierarchical clustered-inference machinery: ICC against the closed-form one-way
  random-effects identity to 1e-10, beta-binomial against `statsmodels` GLM, and the measured
  5.2-fold type-I inflation of the naive fly-as-replicate test.
- `min_attainable_p` and the underpowered-by-design verdict, verified by brute force. The 2026-09
  audit's correction here is a model of the right behaviour: it established that **no** exact
  two-sided test can reach p < 0.05 with three vials per group, declined to substitute a test that
  would have appeared to, and reported the design floor instead.
- The PI affine identity to the published performance index, to 1.4e-14.
- `control_prior.py`'s operating characteristics, including the unflattering measured result that
  robust borrowing takes type-I error from 2.9% to 16.4% at ten points of control drift.
- `larval.py`'s refusals: mm/s without spatial calibration, and any video whose container reports
  no frame rate.
- The refusal of free-running circadian period from LD recordings.
- `docs/METHODS_PROVENANCE.md`'s separation of established methods from house-defined scores
  (CRI, TCS, MCMS, SDQ, ECS, MSCI, MDR labels), which pre-empts the commonest reviewer objection.

### 4. What contains known or newly discovered defects?

86 verified findings: **26 P0** (a wrong scientific number on a path reachable today, silently),
**39 P1** (invalid or miscalibrated inference), **6 P2** (claims in the project's own documents its
own data do not support), **19 P3** (crashes, unreachable features, unread safeguards). A further
21 of the external reviewer's claims were **refuted** and 12 were **already fixed** — both recorded,
because acting on a refuted finding would damage correct code.

**The dominant pattern, found four separate times:** a correct safeguard exists and is undone,
bypassed, or never read.

| | The safeguard | How it is defeated |
|---|---|---|
| 1 | `dam_reader.read_monitor` writes NaN for faulty monitor minutes, and the reader's docstring names zero-filling as corruption mode #1 | `dams.load_dams()` applies `.fillna(0.0)` one function later. 60 missing minutes become a 60-minute sleep bout with `Minutes_missing = 0`. Every production path calls it. |
| 2 | `motor_decline.compute_msci` uses an exact permutation p because the asymptotic p is 0.0 at n = 3 — a fix the suite asserts in three checks | `app.py:859` defines `_compute_msci_inline`, whose comment says *"bypasses module function"*. It is the only MSCI path in the UI. It prints p = 0.0, drops `min_groups` to 2, omits the caveat, and adds a cross-genotype pooled row. |
| 3 | `climbing_video.py` has a confirmable per-vial scoring line, a 5-frame median, explicit unusable rows, and separately stored detected and operator tap times | None of it was back-ported to the legacy stack (`auto_recording`, `fly_detector`, `video_analyzer`, `cv_pipeline`, `recorder`, `recording`, `auto_session`), which neither internal video audit covered — and which three app pages and two public APIs still run on. |
| 4 | `data_triage.py` qualifies a design before analysis: confounding, replication, clustering, coverage | Its only caller in the project is `dam_pipeline.py:631`. **It never runs on climbing data at all.** `provenance.py` is in the same position: never imported, so every export is unattributed. |

The lab knows the right answers. The defect is distribution, not knowledge.

**The second pattern** is the one the external reviewer identified correctly and generally: an
identifier that defines the experimental unit is discarded upstream, so the method downstream cannot
be correct however well implemented. Its sharpest instance is quantitative — run on DroPhenix's own
demo data, which holds **36 real vials**, `data_triage` reports **3 units**, pools twelve unrelated
vials into each, measures genotype variance as between-vial variance, and tells the user to inflate
standard errors by ~380% on the strength of it.

**The most serious single finding** is not an identifier problem and was not in the external review:
`hierarchical.py:345` clips negative Hessian variances to zero, producing `z = inf`, `p = 0.0` and a
zero-width odds-ratio interval **while reporting `converged = True`**. It is undetectable from the
output, and the reviewer's own proposed safeguard would not catch it.

### 5. Which modules are duplicated?

- **Six competing climbing detection/scoring implementations** (`climbing_video`,
  `climbing_simple`, `climbing_video_analyzer`, `climbing_band`, `fly_detector`, `fly_blob`) plus a
  backend selector. They encode different assumptions, and — measured — different biological
  conclusions from the same footage.
- **Two camera-recording implementations** (`recording.py`, `recorder.py`) with *contradictory*
  backend configuration: one removes DirectShow from the OpenCV registry, and all four camera opens
  across both then request it. Live recording cannot start on Windows.
- **Two MSCI implementations**, one audited and unused, one inline in the UI and wrong (above).
- `multi_assay.process_lifespan` was a third median-survival implementation; the 2026-09 audit
  already delegated it to `survival.median_survival_table`. That is the correct precedent.

### 6. What should be retained?

Everything in §3, plus the architecture's distinctive asset: **refusal as a first-class result**.
`stats_engine`'s design floor, the LD period refusal, `larval`'s frame-rate refusal, `trajectory`'s
horizon refusal, `normalizer`'s censored-t50 suppression, `climbing_band`'s reason codes. Scattered
through the codebase as a habit, this is the single most publishable property of the platform. It
should become a framework primitive.

Retain also the 42 suites, the eight published-data anchors, the method-provenance separation, and
the two prior audit documents — including their records of their own errors.

### 7. What should be refactored?

- `app.py` — 33 pages and 7,983 lines are the only composition layer in the project. Of 50 modules,
  only 20 import another project module at all, mostly function-locally. Design-aware routing has
  nowhere to live. Retire page by page behind an API, keeping the app runnable throughout.
- The six detectors into one interface with several validated backends and a **separate** scorer
  that takes its denominator from experimental metadata.
- `normalizer.py`'s nine hardcoded `groupby` lists into a `DesignSpec`-driven analysis key.
- The two recorders into one, with real capture timestamps retained.
- Per-fly assay comparisons (`grooming`, `larval`, `ymaze`) routed through `hierarchical.py`, which
  already exists and is imported by neither.

### 8. What should be redesigned?

- **`translational_score.py`** — not repaired but redesigned. Five components largely derived from
  one climbing measurement are not five lines of evidence; missing data must never score; and
  compounds with different evidence sets must not share a scale. Rebuild as a transparent
  multi-domain evidence profile with uncertainty, as an exploratory prioritisation index.
- **`trajectory_cluster.py`** — the distance/linkage pairing, the barycentre, the stability
  estimate, and the clustering unit are each wrong, and it runs on the wrong time axis entirely
  (assay seconds, not age — the exact category error `motor_decline.py` is emphatic about avoiding).
- **The `drug_ranking` composite** — a fixed-effect CI and p reported beside an I² of 90% and
  `assays_consistent = False` is not a small bug: p = 3.0×10⁻⁷ where random effects give 0.107.
- **The missing-data representation across the platform** — a detector finding no flies, a vial in
  which no fly climbed, an unscheduled observation and a refused calculation are four different
  scientific facts that currently share one representation or collapse into zero.

### 9. What is missing for a professional desktop and web platform?

A shared data model with declared experimental units; typed missingness; a provenance manifest that
survives export; an API; a frontend; packaging; a container; pytest; and — scientifically the
largest gap — **an annotated detector benchmark**. The only fly-count ground truth in the repository
is 57 hand-counted cylinder-timepoints, which are aggregate counts per cylinder, not per-fly
positions. Precision, recall, F1, count bias and sensitivity to stationary or overlapping flies
**cannot be computed from them**, and the measured phenotype-dependent recall (0.88 healthy → 0.35
severe) is strong reason to expect the last two to be poor.

The best detector evidence in the project is `validate_fly_blob.py` — real footage against N loaded
in a sealed cylinder, 110 cells over 22 held-out vials fitted on 19 separate ones, MAE 2.69 against
8.38 for the shipped threshold — and its own docstring correctly explains why that still cannot
separate detector error from a definitional difference. Until a per-fly annotated benchmark exists,
**the only defensible detector claim is a count MAE of 2.69 flies with a signed bias of −1.60**, and
no precision, recall or F1 claim may appear in the manuscript.

### 10. What is the most efficient implementation order?

`PRIORITISED_DEVELOPMENT_PLAN.md`. The ordering principle: fix what produces a wrong number on a
live path, then make the wrong number impossible to produce again, then build. Three sequencing
constraints are non-obvious and are enforced there — the Video Analysis page's three broken
signatures must be repaired *after* the denominator and timepoint fixes, the typed-missingness fix
must precede the integer-cast fix it would otherwise crash, and the experimental-unit fix must
precede the five defects it repairs.

---

## 2. Scientific and engineering risks

| Risk | Exposure | Mitigation |
|---|---|---|
| **A detector blind to the phenotype under study** | Measured recall 0.88 healthy → 0.35 severe; confidence-based filtering preferentially deletes the impaired cohort. Biases every disease comparison towards the null. | Merge-splitting; stratified annotated benchmark; never pool F1 across motion states. |
| **A passing test suite read as validation** | 42 suites pass with 86 defects live; five pass by testing the adjacent case, which reads as coverage of the defect. | Keep the four claim levels separate in every document; add the ten named regression tests; never cite suite pass as validation. |
| **A house score cited as a method** | CRI, TCS, MCMS, SDQ, ECS, MSCI, MDR labels. | Already mitigated by `METHODS_PROVENANCE.md`; carry that separation into the manuscript and the UI. |
| **Manuscript claims the data do not support** | Six found (P2), including the wild-type reference arm sleeping above the project's own plausibility ceiling with no mention, and a dimorphism claim false in one of two genotypes. | Re-derive every reported number from the current code before submission; make the plausibility gate per-group. |
| **Unattributed outputs** | `provenance.py` never imported; 3 of 11 manifest items exist. | Manifest sidecar on every export, enforced at the API boundary. |
| **Silent data repair presented as raw** | The "Preview" a user checks already has negatives zeroed, over-counts capped, duplicates averaged; no later page records it. | Raw/QC/corrected separation; corrected-cell flags carried into every page and export. |
| **Unrecoverable acquisition metadata** | Measured 1.6× video time compression; only nominal fps and requested duration stored, so the error cannot be detected after the fact. | Store real capture timestamps per frame. |
| **Over-claim on prediction** | Trajectory forecasting has calibrated intervals on held-out timepoints but not held-out cohorts, and endpoints currently derive from refused days. | Research-preview labelling; cohort-level conformal calibration; filter refused days. |
| **Single-maintainer bus factor** | One 7,983-line monolith, no packaging, no pytest, no CI. | Packaging, pytest, CI as part of M2. |

---

## 3. Publication readiness

**Not established, and no claim of it should be made yet.** The expert's assessment is correct and
this audit sharpens it.

What is already strong enough for a methods paper: the breadth of assays, the design-aware
qualification concept, the refusal behaviour, the published-data anchors, the method-provenance
discipline, and the hierarchical/survivor-bias/forecasting methodology as *research* contributions.

What blocks submission: the 26 P0 defects (several of which would change numbers already in the
manuscript), the absence of an annotated detector benchmark behind any accuracy claim, the six
unsupported document claims, and the fact that the application's own UI bypasses audited code in at
least two places.

On venue, the expert's reading is right: present DroPhenix as a **general Drosophila phenomics
platform** with neurodegeneration as one biological demonstration, not as an ALS tool — the NAR Web
Server Issue requires disease-agnostic applications, accepts standalone software with a container,
and requires a presubmission proposal. That proposal should not be sent until the P0 corrections
land and the detector benchmark exists.

---

## 4. What this audit did not do

- It did not run the application end to end against real videos, raw DAM monitor files, or a
  complete biological experiment.
- It did not assess detector accuracy on real footage beyond the count-level evidence already in
  the repository.
- It did not verify the manuscript's figures against the current code (recommended before
  submission; six claims in the supporting documents already failed that test).
- It assigned no numerical accuracy or validation score, because none is warranted.
- It changed no code. Every finding is reproducible from the commands recorded in the panel reports.

---

*Produced read-only, 2026-11-07. Baseline `validation/run_all.py`: 42 suites, 0 failures, exit 0.*
