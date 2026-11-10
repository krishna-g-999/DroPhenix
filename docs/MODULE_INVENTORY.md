# DroPhenix — module inventory

**Date:** 2026-11-07 · **Method:** read-only inspection of `D:\DroPhenix\DroPhenix-main` at
commit `633c3f4`. Line counts from `wc -l`, public function counts and summaries from the AST,
import edges from a grep over every import statement including function-local ones.
**No file in the repository was modified to produce this document.**

This is the WP0 deliverable: every module has a documented purpose, an input/output contract
summary, a declared owner layer in the target architecture, and a status. It is the companion to
`REPOSITORY_AUDIT.md` (what exists and what is wrong) and `BUG_AND_RISK_REGISTER.md` (verified
defects).

---

## 1. Scale

| Quantity | Value |
|---|---|
| Python modules in `modules/` | 50 (excluding `__init__.py`) |
| Lines in `modules/` | 28,911 |
| Lines in `app.py` (Streamlit monolith) | 7,983 |
| Streamlit pages in `app.py` | 33 |
| CLI entry point | `drophenix_cli.py` (29,411 bytes) |
| Validation suites | 44 `validate_*.py` + `run_all.py` master runner |
| Published reference datasets wired into a suite | 8 |
| Largest module | `circadian_climbing.py` (2,233 lines) |
| Modules over 1,000 lines | 5 (`circadian_climbing`, `climbing_video`, `dose_response`, `climbing_band`, `trajectory`, `stats_engine`) |

## 2. Entry points

| Entry point | Kind | Notes |
|---|---|---|
| `app.py` | Streamlit application | The only integration layer. 33 pages, each wiring one or more assay modules directly to the UI. |
| `drophenix_cli.py` | Command line | Headless path; used for the autonomous DAM run analysis. |
| `validation/run_all.py` | Validation runner | Executes every suite, reports consolidated pass/fail, reports `SKIPPED` per suite on a missing optional dependency. |
| `.claude/launch.json` | Dev-server config | Launches the Streamlit app for preview. |

**Architectural finding (see `REPOSITORY_AUDIT.md` §3):** the modules are almost entirely flat.
Of 50 modules, only 20 import another project module at all, and most of those imports are
function-local. There is no composition layer between the algorithms and the UI, so the
"design-aware routing" the scientific plan requires has nowhere to live today except inside
`app.py`.

## 3. Inventory

Status legend — **Keep**: scientifically sound, retain as is or with minor work. **Correct**:
retain the algorithm, fix a verified defect first. **Consolidate**: duplicate or competing
implementation, must be unified behind one interface. **Research preview**: the method is
legitimate but its claims exceed its current validation; must ship labelled as exploratory.
**Harden**: not a scientific risk, engineering work only.

### 3.1 Experimental design, ingestion and provenance

| Module | Lines | Purpose | Input → Output | Target layer | Status |
|---|---|---|---|---|---|
| `design_schema.py` | 635 | Extended design schema for longitudinal, chronobiological and dose-response designs (FlyID, VialID, Age, ZT, TrialIndex, Dose, Compound, Batch, ExperimentID) | raw column names → canonicalised design columns + confounding check | `drocore/schemas` | Keep — but must become *authoritative*, not advisory |
| `data_parser.py` | 547 | Scientific data ingestion, validation and format normalisation (CSV/Excel/PDF) | file → canonical counts table | `drocore/validation` | **Correct** — silent repair of invalid observations |
| `data_triage.py` | 446 | Go/no-go qualification: which questions can this dataset answer (confounding, replication, ceilings, clustering, plausibility, coverage) | design table → blocked/warned question list | `drocore/validation` | Keep — scope its checks to the intended analysis |
| `provenance.py` | 56 | Records the software and library versions behind every output | — → provenance dict in `DataFrame.attrs` | `drocore/provenance` | **Correct** — `.attrs` does not survive CSV export |
| `normalizer.py` | 490 | Core normalisation and the primary climbing endpoints (PI, AUC, t50) | counts table → endpoint table | `phenotyping/climbing` | **Correct** — drops Age/VialID/FlyID/ExperimentID |
| `synthetic.py` | 272 | Literature-calibrated synthetic data for all four assays | parameters → synthetic datasets | `tests/scientific` | **Correct** — does not generate the structures it is used to test |

### 3.2 Climbing video — acquisition

| Module | Lines | Purpose | Target layer | Status |
|---|---|---|---|---|
| `recording.py` | 366 | Thread-safe camera recording for the Live Recording page | `phenotyping/video/capture` | **Correct** — camera-backend conflict, clock drift |
| `recorder.py` | 358 | Camera management and timed capture for automated assays | `phenotyping/video/capture` | **Consolidate** with `recording.py` |
| `auto_recording.py` | 200 | Adaptive recording (motion-triggered) | `phenotyping/video/capture` | **Correct** — background-subtractor channel mismatch; image-fraction scoring line |
| `auto_session.py` | 143 | Bridges `RecordingSession` raw counts into the normalizer/metrics pipeline | `phenotyping/climbing` | **Correct** — integer truncation of fractional measurements |
| `protocol.py` | 290 | The recording-protocol specification and a pre-experiment setup check | `drocore/experiment` | Keep — the protocol/validation distinction is correct and should survive into the manuscript |
| `protocol_guide.py` | 210 | Downloadable templates and recording instructions | `apps/*` | **Harden** — a download labelled PDF that serves a PNG |

### 3.3 Climbing video — geometry, detection and scoring

Six detection/scoring implementations coexist. This is the single largest consolidation task.

| Module | Lines | Detection principle | Known limitation | Status |
|---|---|---|---|---|
| `fly_blob.py` | 296 | Crocker–Grier size-matched band-pass blob detection | temporal-median background can absorb a stationary fly | **Keep as the preferred backend** — needs ground-truth benchmark |
| `fly_detector.py` | 513 | Otsu thresholding | dark vial markings; no-detection indistinguishable from zero climbing | **Correct** then **Consolidate** |
| `climbing_video.py` | 1,985 | Temporal background subtraction; canonical scoring path | stationary flies fade into background; tap-time default | **Correct** |
| `climbing_simple.py` | 402 | Darkest horizontal feature inside an operator box | depends entirely on operator ROI | **Consolidate** |
| `climbing_video_analyzer.py` | 615 | Adaptive thresholding, operator-defined ROI | graduations mistaken for flies | **Consolidate** |
| `climbing_band.py` | 1,230 | Blob detection plus persistence filtering, band-and-zone scoring | persistence filter can remove the impaired phenotype | **Research preview** — documented as not yet usable |
| `climbing_detectors.py` | 787 | Chooses a backend from footage characteristics | its decision thresholds are not independently validated | **Correct** — training/test independence unverified |
| `cylinder_finder.py` | 308 | Locates cylinders from the operator's drawn band | depends on consistent band geometry | Keep |
| `climbing_layout.py` | 358 | Hand-drawn vial layouts, aligned across a session, refuses when unsafe | — | Keep |
| `calibration_card.py` | 291 | ArUco card: pixels → millimetres, printer-scale and plane checks | single scale factor, no intrinsics or rectification | **Correct** (uncertainty) |
| `video_analyzer.py` | 462 | Frame-level analysis: ROIs, timepoint sampling, QC output | denominator from detected rather than loaded count; one frame per timepoint | **Correct** — release-blocking |
| `cv_pipeline.py` | 55 | Thin wrapper over `VideoAnalyzer` for single and batch runs | treats each video file as a sequential replicate | **Correct** |

### 3.4 Activity, sleep and circadian

| Module | Lines | Purpose | Target layer | Status |
|---|---|---|---|---|
| `dam_reader.py` | 574 | Reads raw DAMSystem monitor files; monitor faults → missing, not sleep; restart and terminal-immobility detection; refuses period under LD | `phenotyping/dams` | **Correct** — death-detection naming and exclusion policy |
| `dams.py` | 773 | Sleep and circadian metrics engine (5-min rule, chi-square periodogram, Lomb–Scargle) | `phenotyping/dams` | **Correct** — missing bins in the observation denominator; bout statistics at coarse resolution |
| `dam_pipeline.py` | 674 | Raw monitor files → design-aware, verified DAM analysis | `statistics` + `reporting` | Keep — biological-unit audit required |
| `dam_planner.py` | 328 | Plans a monitor run before it happens (prevents the age/monitor confound) | `drocore/experiment` | Keep |
| `dam_figures.py` | 410 | Sleep-literature-format figures from precomputed statistics | `reporting` | Keep |
| `dam_report.py` | 429 | Word report from a `DamRun`, with a check that every reported value traces to a source table | `reporting` | Keep — traceability is not validity |
| `circadian_climbing.py` | 2,233 | Time-of-day cosinor analysis of climbing, with circadian/testing-order separation | `statistics/longitudinal` | Keep — strongest design-aware module; repeated-measures model to extend |
| `control_prior.py` | 915 | Robust meta-analytic-predictive priors for historical controls | `statistics` | **Research preview** — drift operating characteristics must be shown |

### 3.5 Statistics

| Module | Lines | Purpose | Target layer | Status |
|---|---|---|---|---|
| `stats_engine.py` | 1,104 | Test selection, effect sizes, attainable-p floor, underpowered-by-design refusal | `statistics/core` | **Correct** — normality-gated router; omnibus/post-hoc naming |
| `hierarchical.py` | 813 | Beta-binomial, GEE and mixed models for vial-clustered climbing counts; ICC, DEFF, ESS | `statistics/hierarchical` | **Correct** — cluster-count safeguards, convergence gating |
| `survival.py` | 625 | Kaplan–Meier, log-rank with BH, median survival CI, RMST, Cox (statsmodels core) | `statistics/survival` | **Correct** — cohort expansion, missing event status, PH diagnostics |
| `survivor_bias.py` | 729 | Survivor-conditional vs balanced-panel vs mixed-model vs pattern-mixture trajectories | `statistics/longitudinal` | **Research preview** — estimands and dropout assumptions must be explicit |
| `dose_response.py` | 1,567 | LL.2/LL.3/LL.4 Hill fits, profile-likelihood intervals, Brain–Cousens hormesis, mortality-confounding diagnostics | `therapeutic/dose_response` | **Correct** — LC50 definition is the highest-severity numerical defect found |

### 3.6 Longitudinal, prediction and therapeutic interpretation

| Module | Lines | Purpose | Target layer | Status |
|---|---|---|---|---|
| `motor_decline.py` | 886 | Motor Decline Rate, AICc model choice, decline CIs, trajectory phenotyping, motor–survival coupling | `statistics/longitudinal` | **Correct** — fits group means, not replicate trajectories |
| `trajectory.py` | 1,179 | Hierarchical disease-trajectory forecasting: curve families, empirical-Bayes population prior, forward validation, conformal intervals, refusal beyond horizon | `evidence/forecast` | **Research preview** — exact-fit model weighting; refused days reaching endpoints |
| `trajectory_cluster.py` | 480 | DTW trajectory clustering, cluster-count selection, archetype labels, PCA/UMAP | `evidence/exploratory` | **Research preview** — Ward with DTW, pseudo-barycentre, clusters condition means |
| `metrics.py` | 654 | Composite metrics: CRI, SDQ, TSW (with interaction contrast), ECS | `therapeutic` | **Correct** — TSW direction; component-set comparability |
| `multi_assay.py` | 308 | Multi-Assay Composite Motor Score across climbing, survival, flight, activity | `evidence` | **Correct** — cross-coverage ranking |
| `translational_score.py` | 430 | Translational Candidate Score across CRI/TSW/SDQ/MDR/trajectory | `therapeutic` | **Redesign** — missing data scored as evidence; correlated components |
| `drug_ranking.py` | 369 | Cross-strain, cross-assay compound ranking; Hedges' g, inverse-variance and DerSimonian–Laird pooling, Q/τ²/I² | `therapeutic` | **Correct** — shared-control correlation; small-k intervals |

### 3.7 Other behavioural assays

| Module | Lines | Purpose | Target layer | Status |
|---|---|---|---|---|
| `ymaze.py` | 548 | Turn bias, switch rate, arm sequences, alternation, handedness, bias-corrected alternation and switchiness | `phenotyping/ymaze` | **Correct** — long-format FlyID; raw-alternation interpretation |
| `grooming.py` | 238 | Grooming time, bouts, mean duration, rate; interval merging | `phenotyping/grooming` | **Correct** — denominator wording; clustered comparisons |
| `larval.py` | 396 | Larval path length, displacement, speed, tortuosity, pauses, speed–curvature exponent | `phenotyping/larval` | **Correct** — object identity; clustered comparisons |

### 3.8 Presentation

| Module | Lines | Purpose | Status |
|---|---|---|---|
| `visualizer.py` | 617 | Publication figures: Wong (2011) palette, 300 DPI, raw-data overlays | **Correct** — zero substitution for missing metric and missing SEM |
| `styles.py` | 298 | Branding, sidebar, scoped CSS | **Harden** — unescaped interpolation into `unsafe_allow_html`; CWD-relative logo path |

## 4. Internal dependency edges

The complete set of intra-project import edges (20 of 50 modules participate):

```
auto_recording      -> recording
auto_session        -> normalizer, metrics
calibration_card    -> climbing_video
circadian_climbing  -> design_schema, hierarchical
climbing_band       -> fly_blob
climbing_detectors  -> climbing_video, climbing_video_analyzer, fly_blob
climbing_layout     -> climbing_video
cv_pipeline         -> video_analyzer, fly_detector
dam_pipeline        -> dam_reader, dams, drug_ranking, stats_engine, data_triage
dam_planner         -> design_schema
dams                -> stats_engine
data_parser         -> design_schema          (NOTE: bare `from design_schema import ...`)
data_triage         -> hierarchical, dam_reader
drug_ranking        -> data_parser, normalizer, ymaze, dams, survival
grooming            -> stats_engine
larval              -> stats_engine
multi_assay         -> survival
protocol            -> climbing_video, calibration_card
recorder            -> fly_detector
synthetic           -> ymaze, dams
video_analyzer      -> fly_detector
ymaze               -> stats_engine
```

Two observations carried into `REPOSITORY_AUDIT.md`:

1. `hierarchical.py` is imported by only two modules (`circadian_climbing`, `data_triage`) and by
   neither `grooming`, `larval`, `ymaze` nor `normalizer` — the per-fly assays that most need it.
   The clustering correction exists but is not wired into the paths that commit the error.
2. `data_parser.py:38` imports `design_schema` without the `modules.` prefix, so it resolves only
   when `modules/` happens to be on `sys.path`. Fragile under packaging.

## 5. Validation coverage

44 suites, run by `validation/run_all.py`. Eight are anchored to published raw data:

| Module | Published anchor | Scale |
|---|---|---|
| `survival.py` | Lifespan heritability, *Am Nat* 2013 (Zenodo 4996358) | 39,581 individual flies |
| `motor_decline.py`, `survival`, MSCI | Physiological decline, *Evolution* 2016 (Zenodo 4954719) | 10 populations, geotaxis + mortality across age |
| `larval.py` | Speed–curvature power law, *Biol Lett* 2016 (Zenodo 5010896) | 123 larvae, 7 fps tracks |
| `grooming.py` | Nf1 grooming, *PLOS Genet* 2020 (Zenodo 3964901) | 108 flies + 1,096 scored bouts |
| climbing velocity | FreeClimber, *J Exp Biol* 2021 | 4,093 particle detections, real w1118 recording |
| `dose_response.py` | `drc` `ryegrass` (Inderjit et al. 2002) | 24 obs, agrees to ≤ 9e-6 |
| `stats_engine.py` | PlantGrowth (Dobson 1983) | exact |
| `survival.py` | Freireich et al. 1963 6-MP | exact |

Independent-implementation cross-checks in use: `lifelines` (KM/median), `statsmodels`
(GLM binomial, survival core), `astropy` `LombScargle` (cosinor coefficients, agreement to
1e-12), `scipy.stats.permutation_test` (exact enumeration), plus closed-form identities.

**Not backed by any real dataset:** `hierarchical.py`, `survivor_bias.py`, `trajectory.py`,
`control_prior.py` (closed forms and generative models only — these are methods modules), the
`dams.py` ≥5-min bout rule (literature anchors only, no raw beam counts), and
`circadian_climbing.py` beyond the astropy cross-check.

**Detector accuracy is the largest measurement gap.** The only fly-count ground truth in the
repository is `data/validation/hand_count_holdout.csv` (39 cylinder-timepoint hand counts) and
`hand_count_template.csv` (18). These are aggregate counts per cylinder per timepoint, not
per-fly annotations, so they support a count-agreement check but **cannot** yield precision,
recall, F1 or sensitivity to stationary and overlapping flies — the metrics a detector-accuracy
claim requires.

---

*Produced read-only. No repository file was modified. Companion documents:
`REPOSITORY_AUDIT.md`, `ARCHITECTURE_PROPOSAL.md`, `BUG_AND_RISK_REGISTER.md`,
`SCIENTIFIC_VALIDATION_PLAN.md`.*
