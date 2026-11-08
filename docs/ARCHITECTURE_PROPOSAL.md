# DroPhenix — architecture proposal

**Date:** 2026-11-07 · **Status:** proposal, awaiting the project owner's approval.
**Scope:** `D:\DroPhenix\DroPhenix-main` at commit `633c3f4`.
Companion documents: `MODULE_INVENTORY.md`, `REPOSITORY_AUDIT.md`,
`BUG_AND_RISK_REGISTER.md`, `SCIENTIFIC_VALIDATION_PLAN.md`.

No code has been changed. This document says what should be built, in what order, and — for each
decision — why the science requires it rather than why it would look tidier.

---

## 1. The one problem the architecture has to solve

Nearly every open scientific defect in this codebase has the same shape:

> **An identifier that defines the experimental unit is discarded upstream, so the statistical
> method downstream cannot be correct however well it is implemented.**

Concretely:

- `design_schema.py` canonicalises `FlyID`, `VialID`, `Age`, `ExperimentID`, `BatchID`.
  `normalizer.py` then groups endpoints by `Genotype × Sex × Treatment × Replicate`. Age, vial
  and fly are gone before any statistic is computed. Everything downstream —
  `motor_decline.py`, `trajectory.py`, `trajectory_cluster.py` — therefore fits **group means**
  and reports uncertainty that does not describe the experiment.
- `hierarchical.py` contains the correct clustered-inference machinery, measured at a 5.2-fold
  type-I inflation when flies are treated as independent. It is imported by exactly two modules,
  and by none of `grooming`, `larval`, `ymaze` or `normalizer` — the per-fly paths that commit
  the error.
- `video_analyzer.py` sets the climbing denominator from the number of flies the detector found
  rather than the number loaded into the tube, so a detection failure becomes a biological
  measurement.

This is not a code-cleanliness problem. It is a measurement-model problem, and it is the reason
the architecture must change before more features are added.

**The architectural commitment:** the experimental unit is a first-class, declared property of
every dataset and every analysis, carried end to end, and no analysis may aggregate across an
identifier without declaring that it did so.

## 2. What is already right and must be preserved

The repository is not a prototype in need of replacement. It already does several things that
most behavioural-analysis software does not, and the migration must not lose them.

| Existing strength | Where | Why it must survive |
|---|---|---|
| **Refusal instead of a confident number** | `stats_engine.min_attainable_p`, `dam_reader` period refusal under LD, `larval` refusing an invented frame rate, `trajectory` refusing beyond horizon, `normalizer` suppressing a censored t50 mean | This is the single most publishable property of the platform. It must become a framework primitive, not a per-module habit. |
| **Published-data anchors** | 8 suites against Zenodo/Dryad datasets totalling ~40,000 flies | Reproducibility evidence a reviewer can rerun. |
| **Independent-implementation cross-checks** | `lifelines`, `statsmodels`, `astropy` `LombScargle`, `scipy.permutation_test`, closed forms | Distinguishes "our code runs" from "our maths is right". |
| **House-metric provenance** | `docs/METHODS_PROVENANCE.md` separates established methods (KM, log-rank, Hill, cosinor, GEE) from house-defined scores (CRI, TCS, MCMS, SDQ, ECS, MSCI, MDR labels) | Prevents the most common reviewer objection: a house score cited as if it were a literature method. |
| **Design qualification before analysis** | `data_triage.py`, `dam_planner.py`, `circadian_climbing` order/phase separability | Correct scientific sequence: establish what the design can answer *before* testing. |
| **Audit honesty** | `docs/MODULE_AUDIT_2026-09.md` records its own two wrong fixes (AICc K, t½ interval) and one wrong worked example | A culture worth keeping; the register in this proposal continues it. |

The 2026-09 internal audit already fixed 31 findings and left 3 explicitly open. The external
expert review overlaps it substantially and confirms several fixes. The remaining work is
smaller than the external review alone implies — which is exactly why each finding has been
re-verified against the current code rather than copied forward.

## 3. Target package structure

Incremental extraction, not a rewrite. Each module moves only when its replacement passes a
regression test demonstrating identical numbers (or a documented, approved correction).

```
DroPhenix/
├── packages/
│   ├── drocore/              # the contract layer — no assay logic
│   │   ├── schemas/          # entities §4; pandera/pydantic validation
│   │   ├── units/            # ExperimentalUnit: the declared analysis key
│   │   ├── qc/               # QCStatus states; never a repaired value
│   │   ├── refusal/          # Refusal(reason, what_would_fix_it) primitive
│   │   └── provenance/       # sidecar manifest writer
│   ├── phenotyping/
│   │   ├── video/            # capture, geometry, calibration
│   │   ├── detect/           # ONE detector interface, N validated backends
│   │   ├── climbing/         # counts -> PI, AUC, t50
│   │   ├── dams/             # reader, sleep, circadian
│   │   ├── ymaze/ grooming/ larval/
│   ├── statistics/
│   │   ├── core/             # a-priori test selection, effect sizes, power floors
│   │   ├── hierarchical/     # beta-binomial, GEE, GLMM
│   │   ├── survival/         # KM, log-rank, RMST, Cox + PH diagnostics, frailty
│   │   ├── longitudinal/     # replicate-level mixed models, cosinor
│   │   └── dose_response/    # LL.x, binomial DR, hormesis
│   ├── therapeutic/          # rescue, selectivity, toxicity, ranking
│   ├── evidence/             # cross-assay profile, discordance, forecasting
│   └── reporting/            # tables, figures, QC report, manifest
├── services/api/             # FastAPI; the ONLY way a UI reaches science
├── apps/{desktop,web}/       # React + TypeScript; Tauri shell for desktop
├── tests/{unit,integration,scientific,benchmark,regression}/
├── benchmarks/{videos,annotations,synthetic,reference_results}/
└── docs/
```

`app.py` (7,983 lines, 33 pages) is retired page by page: each page becomes a thin client of an
API endpoint. It stays runnable throughout so the lab is never without a working tool.

## 4. The canonical data model

Entities, with the experimental-unit question answered explicitly for each assay rather than left
to the analysis that happens to run.

| Entity | Identity | Notes |
|---|---|---|
| Project | `ProjectID` | |
| Experiment | `ExperimentID` | One independent experimental run. The outermost replication unit. |
| Batch | `BatchID` | Rearing/handling batch; a random effect, not a nuisance to average over. |
| Strain / Genotype | `StrainID`, `Genotype` | Genotype records driver and responder separately (GAL4/UAS), plus background. |
| Cohort | `CohortID` | A group defined by genotype × sex × treatment × age at start. |
| Vial | `VialID` | Housing **and** the usual experimental unit for group climbing. |
| Fly | `FlyID` | Present only when individuals are genuinely tracked. Never synthesised. |
| Treatment / Dose | `Compound`, `Dose`, `Route`, `Schedule` | Vehicle is a treatment level, not an absence. |
| Assay | `Assay` | climbing, dams, survival, ymaze, grooming, larval, flight |
| Observation | `(unit, Assay, Time)` → `RawValue` | Immutable. |
| DerivedPhenotype | → `DerivedValue` | Carries the function, parameters and inputs that produced it. |
| Analysis | model, formula, unit declaration | Carries convergence state and diagnostics. |
| Result | estimate, CI, adjusted p, diagnostics, **refusals** | A refusal is a result. |
| Provenance | versions, checksums, seeds, parameters, OS, commit | Sidecar file, not `DataFrame.attrs`. |

### 4.1 The `ExperimentalUnit` declaration

Every analysis function must declare, in its signature, which identifier is the independent unit
and which identifiers it aggregates over. The framework refuses an analysis whose declared unit is
absent from the data.

| Assay | Default independent unit | Repeated within unit |
|---|---|---|
| Group climbing (counts above a line) | `VialID` within `ExperimentID` | timepoints, trials, ages |
| Individually tracked climbing | `FlyID` | timepoints, ages |
| DAM activity/sleep | `FlyID` (one channel) | minutes, days |
| Lifespan | `FlyID` for event times; `VialID` for shared conditions (frailty term) | — |
| Y-maze | `FlyID` | turns (the sequence is the datum) |
| Grooming | `FlyID`, clustered in `VialID` | bouts |
| Larval | larva, clustered in clutch/plate | frames |
| Dose-response | the unit at which the dose was applied — usually `VialID` | flies in that vial |

**Rule:** a video file is not a replicate; a vial is not automatically independent of its
cohort; a minute is never a replicate. The schema stores what the experiment was, and the
analysis declares what it treats as independent — the two are then checked against each other.

### 4.2 Three data states, never merged

| State | Meaning | Permitted operation |
|---|---|---|
| `RAW` | exactly as observed or decoded | read only; never overwritten |
| `QC` | a flag plus a reason, attached to a raw value | filter, report, stratify |
| `CORRECTED` | a value that differs from raw | only with an explicit, logged, owner-approved policy; always alongside the raw value and the reason |

This single rule retires the whole class of defects where an invalid count was clipped, a
duplicate was averaged and rounded, a missing measurement became zero, or a missing SEM became
zero.

### 4.3 Missingness is typed

`NaN` is not sufficient. The platform distinguishes:

- `NOT_MEASURED` — no observation was scheduled or attempted
- `MEASUREMENT_FAILED` — attempted, instrument or detector failed (e.g. zero flies detected)
- `BELOW_THRESHOLD` — a valid zero (e.g. all flies genuinely below the line)
- `CENSORED_ADMIN` / `CENSORED_LOST` / `CENSORED_ESCAPED` — distinct survival censoring causes
- `REFUSED` — the platform declined to produce a number, with the reason

A detector finding no flies and a vial in which no fly climbed are different scientific facts and
must never share a representation.

## 5. Design-aware statistical routing

Today the caller picks a test. In the target, the caller states the **question** and the design;
the engine selects the family, explains the choice, and reports what would invalidate it.

| Design feature detected | Family selected | Why |
|---|---|---|
| Binomial counts clustered in vials | beta-binomial or binomial GLMM; GEE for population-average | PI is a rescaled binomial proportion; flies in a vial are not independent |
| Repeated ages on the same unit | mixed model with unit random effects | within-unit correlation; informative dropout flagged separately |
| Time-to-event with censoring | KM + log-rank; RMST difference with CI; Cox **only** with PH diagnostics | PH is an assumption, not a default |
| Dose applied at vial level | nonlinear DR on vial-level responses, binomial variance | flies in a vial share the dose application |
| Few independent clusters (< ~10) | refuse sandwich SEs; use small-sample corrections or exact methods | cluster-robust SEs are badly calibrated at 3 vials |
| Attainable p floor above α | **refuse** with the required n stated | already implemented; becomes a framework primitive |
| Complete confounding of two factors | **block** the question | no model recovers absent design information |

Every result object carries: effect size, CI, adjusted p, the model formula, the declared unit,
convergence state, diagnostics, and the list of assumptions not satisfied. A non-converged fit can
never silently become a result.

## 6. Detector architecture — one interface, several validated backends

The six competing climbing implementations are not deleted. They become backends behind:

```python
class FlyDetector(Protocol):
    def detect(self, frame, roi, calibration) -> Detections: ...
    # Detections: positions, n_detected, confidence, QC flags, and
    # explicitly NOT a climbing percentage.
```

and a separate scorer that turns detections into endpoints using **`N_start` from experimental
metadata** as the denominator, never the detected count. `N_detected`, `N_above`, `N_below` and
`N_start` are all stored; a mismatch is a QC flag, not a silent adjustment.

Backend selection must be made on measured performance over an annotated benchmark, with the
selection rule fitted on data disjoint from the data used to evaluate it.

### 6.1 The benchmark that does not yet exist

The repository's only fly-count ground truth is 57 hand-counted cylinder-timepoints
(`data/validation/hand_count_holdout.csv`, `hand_count_template.csv`). These are aggregate counts,
so they support a count-agreement check and nothing more. **Precision, recall, F1, count bias and
sensitivity to stationary or overlapping flies cannot be computed from them.** Until a per-fly
annotated benchmark exists, no detector-accuracy claim may be made — including in the manuscript.

Required benchmark strata, two independent annotators, per-fly positions:
empty vial · sparse · crowded · **stationary flies** · graduated vial · poor lighting · late
timepoints (impaired locomotion) · partial visibility · varied camera position.

The stationary-fly stratum is the scientifically critical one: every background-subtraction and
persistence-filter backend in this codebase can preferentially lose the immobile fly, which is
precisely the neurodegenerative phenotype the platform exists to measure. A detector that is
accurate on healthy flies and blind to impaired ones would bias every disease comparison towards
the null.

## 7. Evidence engine — the flagship, with guardrails

The scientific contribution worth publishing is **not** another composite score. It is a
transparent cross-assay evidence profile with explicit discordance detection.

Design constraints, each derived from a verified defect:

1. **Evidence domains must be genuinely distinct.** CRI, TSW, SDQ, MDR and the trajectory label
   are largely functions of the same climbing measurement; combining them does not create five
   independent lines of evidence. Domains: motor function · survival · activity/circadian ·
   other behavioural function · toxicity and selectivity · reproducibility across experiments.
2. **Missing evidence is missing.** It may never be scored (the current "Undetermined" → 0.2
   mapping makes absence of data contribute as if measured).
3. **Only comparable evidence sets are ranked.** Compounds are compared on a pre-specified common
   domain set; otherwise separate profiles are reported and not ranked.
4. **Discordance is an output, not an error.** Climbing improved, survival unchanged, mild WT
   impairment at high dose is a *finding* about the compound.
5. **Survivor bias is a declared check**, not an afterthought: any rescue claim from an ageing
   cohort carries the balanced-panel and dropout-sensitivity result beside it.
6. **No tier label without an interval.** "Strong translational candidate" requires uncertainty,
   data completeness and reproducibility, not a weighted sum above a threshold.

## 8. One engine, two interfaces

| Layer | Choice | Reason |
|---|---|---|
| Scientific engine | Python (existing modules, migrated) | preserves ~29,000 lines of audited science and 42 validation suites |
| API | FastAPI | one contract for both interfaces |
| Frontend | React + TypeScript | shared between desktop and web |
| Desktop shell | Tauri + local Python service | local video processing, offline, no upload of unpublished data |
| Local store | SQLite + Parquet | offline projects, columnar observations |
| Web store | PostgreSQL | multi-user projects |
| Jobs | separate worker process; **no Redis until a measured need** | long video and simulation jobs |
| Deployment | Docker/Podman | NAR standalone-software requirement |
| Tests | pytest (to be adopted; today a custom runner) | property-based and regression tests alongside the scientific suites |

**Non-negotiable:** no statistical or image-analysis algorithm is reimplemented in TypeScript.
Both interfaces call the same Python functions. A number shown in the browser and the same number
shown on the desktop must come from one implementation.

The 42 existing validation suites keep running against the migrated packages. A migration that
changes a number must either (a) prove the number was wrong and record the correction, or (b) be
rejected.

## 9. Migration sequence

Ordered so that each stage removes a class of wrong numbers before the next stage builds on it.

| Stage | Work | Gate before proceeding |
|---|---|---|
| **M0** | Freeze baseline: 42 suites, all pass, exit 0, recorded with versions (done, 2026-11-07) | baseline recorded |
| **M1** | P0 numerical and data-integrity corrections (see `BUG_AND_RISK_REGISTER.md`) | each fix has a failing-before/passing-after regression test |
| **M2** | `drocore`: schemas, `ExperimentalUnit`, typed missingness, QC states, refusal primitive, provenance sidecar | every assay reads/writes through it; no identifier loss, proven by a round-trip test |
| **M3** | Detector interface + annotated benchmark; retire the denominator defect | precision/recall/F1/bias reported per stratum, stationary-fly stratum included |
| **M4** | Statistical engine: design-aware routing, clustered inference wired into the per-fly assays, PH diagnostics, binomial dose-response | simulation study: nominal type-I error under vial clustering; known EC50/LC50 recovered |
| **M5** | Longitudinal rebuild on replicate-level data (motor decline, trajectory, survivor bias) | known individual and group slopes recovered; interval coverage on held-out cohorts |
| **M6** | Evidence engine with the §7 guardrails | discordance detected on constructed cases; no score from incomparable evidence sets |
| **M7** | FastAPI + React + Tauri; retire `app.py` page by page | both interfaces reproduce the benchmark dataset identically |
| **M8** | Release engineering: container, docs, example workflows, licence | independent reproduction of every manuscript figure from the repository |

## 10. Decisions needed from the project owner

These are scientific choices, not implementation details, and I will not make them unilaterally.

1. **The a-priori test for PI.** `stats_engine.auto_test` still selects a test conditional on a
   Shapiro–Wilk pre-test, which does not preserve α. PI is a rescaled binomial proportion. The
   principled options are a rank test or a binomial/beta-binomial GLM as the declared default.
   Which becomes the platform default? (Open since the 2026-09 audit, item 3.4.)
2. **Vial identifiers in the per-fly assay schemas.** Routing grooming, larval and Y-maze through
   `hierarchical.py` requires `VialID` in those inputs. This is a data-collection change for the
   lab, not only a code change. Do we require it, or report the design effect as a caveat?
3. **Terminal immobility policy.** Rename `likely_dead` to `suspected_terminal_immobility`, keep
   three states (alive / confirmed dead / indeterminate), and **never auto-exclude** the third —
   confirm this, since it changes numbers the DAM report already produced.
4. **Denominator policy.** `N_start` from the loading record becomes the only climbing
   denominator. This will change published-looking percentages wherever the detector under- or
   over-counted. Confirm, and confirm whether historical outputs should be recomputed.
5. **Licence.** An OSI licence is required by NAR Genomics and Bioinformatics; the Web Server
   Issue has its own requirements. Future commercial use constrains the choice. Decide before
   any public release.
6. **Scope discipline for v1.0.** The expert's recommendation, which I endorse: a small number of
   genuinely validated assays (climbing, DAM, survival) plus everything else labelled
   *research preview*, rather than many partially validated assays presented equally.

---

*Read-only audit output. No repository file other than this document was created or modified.*
