# DroPhenix — scientific validation plan

**Date:** 2026-11-07 · **Status:** proposal, awaiting approval.
Companion: `BUG_AND_RISK_REGISTER.md` (the defects this plan must catch), `REPOSITORY_AUDIT.md`.

---

## 0. The four claims, kept separate

Every statement about DroPhenix must declare which of these it is. Conflating them is the single
most likely way this project misleads a reviewer or itself.

| Level | Claim | Evidence that establishes it | Currently |
|---|---|---|---|
| **L1** | The software executes | the suite runs, exit 0 | **Established** — 42 suites, 0 failures |
| **L2** | The mathematics is correct | agreement with closed forms and independent implementations | **Established for the audited paths**: `lifelines`, `statsmodels`, `astropy` (1e-12), R `drc` (9e-6), exact combinatorics, closed-form ICC (1e-10) |
| **L3** | The statistics are valid for the design | simulation with known truth, calibrated type-I error and interval coverage | **Partially** — strong for `hierarchical`; the two rows below that the plan named are now measured, one meeting its target (meta-analysis) and one failing it with the cause diagnosed (dose-response interval, N95); still absent for several paths |
| **L4** | The measurement reflects the biology | annotated ground truth; independent biological endpoints; reproduction across batches and laboratories | **Not established.** The detector gap is the binding constraint |

**A passing suite is L1 and sometimes L2. It is never L3 and never L4.** This audit found 42 suites
passing with 86 defects live, and five suites passing *because they test the adjacent case* — which
is worse than no coverage, because the suite reads as coverage of the defect.

---

## Layer 1 — software and arithmetic correctness

Each item below is a regression test that **fails on the current tree** and must pass after the
corresponding correction. These are not aspirational; the register names the measured failure for
each.

| Test | Asserts | Catches |
|---|---|---|
| Daily-cohort conservation | 10 start, 1 death, 1 censor → exactly 10 rows; `deaths + censored ≤ start` per day asserted; S(5) = 0.9000 | N17, N18 |
| Event-status integrity | a missing or non-binary `Event` raises a row-level error, never becomes "censored" | N19, and the strictly larger coerced class |
| Missing-minute preservation | `load_dams` on a 60-NaN frame preserves 60 NaNs; `sleep_per_fly` through it returns sleep 0.0, `Minutes_missing` 60, `Coverage_frac` 0.958 | N1 |
| Hessian gating | a beta-binomial fit with an indefinite Hessian returns NaN SEs and `inference_valid = False`, never `p = 0.0` with `converged = True` | N16 |
| Convergence surfacing | `fit_mixed_logit` on 3 vials with no between-vial variance returns `fit_warnings` containing the `ConvergenceWarning` and `inference_valid = False`; `compare_time_axes` with one axis failing returns `better_axis = None` | N38 |
| LC50 by definition | fitted curve with `d = 0.90, b = 2, e = 10` reports LC50 = 8.944, not 10.010; at `d = 0.80` the reported value equals the dose where fitted survival = 0.5 | N14, N15 |
| Denominator fixity | N_start = 30, detected 25 → reported 40.0%, not 48.0%; detected 40 → QC flag, denominator still 30 | N5 |
| Zero detections are not zero climbing | a blank frame yields `MEASUREMENT_FAILED`, not `PI = 0.0`; a vial with all flies below the line yields `PI = −100.0` | N8 |
| Timepoint accounting | a 25 s clip asked for `[10,20,30,60,120]` returns 5 rows, 3 marked unusable with a reason | N7 |
| Refused days excluded | `derive_endpoints` on a prediction containing refused days excludes them from `window_days`; onset and healthspan CIs do not inflate | N29 |
| Model-averaging sanity | no curve family with `n_params ≥ n` receives non-zero AICc weight | N26 |
| TCS missingness | `Trajectory = NaN` yields `n_components = 4`, never a 0.2 contribution | N32a |
| TCS weights | `weights={"CRI": 1.0}` yields a weight vector summing to 1.0 with CRI = 1.0 | N32b |
| MCMS comparability | two rows with equal PI and unequal assay coverage return equal tiers or an explicit incompleteness label, and are not ranked together | N33 |
| Composite heterogeneity | four effects 1.20 / −0.10 / 0.90 / 0.05 (var 0.04) yield `ci_lo < 0 < ci_hi`, `p > 0.05`, and a note naming I² | N34 |
| Y-maze identity | six long-format entries without `FlyID` are refused, not split into six flies; invalid arm labels are refused, not scored 100% alternation | N20 |
| Unit identity | triage on the demo data reports 36 units or refuses the ICC with `ambiguous_unit`; it never derives a design effect from `Replicate` alone | N4 |
| No silent repair | a negative count, a count above n, and differing-n duplicates are each flagged and preserved, never rewritten | N11, N12 |
| MSCI path | `_compute_msci_inline` absent; at n = 3 with |ρ| = 1 the UI shows p = 0.333 and "Underpowered by construction" with a non-empty caveat | N2 |
| Longitudinal CI | the three-vial / three-age table yields a decline-rate CI within 10% of a replicate-level mixed model | N23 |
| Three-age honesty | a 3-age design returns either a finite k with `AICc = NaN`, or `fit_model = "insufficient_data_for_model_selection"`; `fit_note` names `n ≤ K` | N24 |
| Time-axis guard | `cluster_trajectories` on a seconds-axis frame refuses with `requires_age_data`; on the age table `time_axis == "Age"` and WT-vehicle carries no "Rescue" archetype | N31 |
| Provenance survival | an export writes a sidecar containing version, git commit, dependency versions, OS, parameters, input checksum, timestamp and seeds; a round-trip through CSV recovers all of them | N74 |
| Identifier round-trip | a dataset carrying `ExperimentID, BatchID, VialID, FlyID, Age, Sex, Dose` reaches every endpoint table with all seven intact, or the aggregation that dropped one is declared | N13, N75 |

**Plus the ten tests the panel named for the suites that currently mask defects**, listed in the
register's "Why 42 passing suites did not catch any of this".

---

## Layer 2 — measurement validation

### 2.1 The detector benchmark (the binding constraint on publication)

**What exists:** 57 hand-counted cylinder-timepoints (`data/validation/hand_count_holdout.csv`,
`hand_count_template.csv`) — aggregate counts per cylinder, not positions. `validate_fly_blob.py`
is the strongest detector evidence in the project: real footage against N loaded in a sealed
cylinder, 110 cells over 22 held-out vials fitted on 19 separate ones, MAE **2.69** against **8.38**
for the shipped threshold. Its own docstring correctly explains why a per-cell total still cannot
separate detector error from a definitional difference, and cannot distinguish 25 correct detections
from 23 correct plus 2 false positives.

**What is therefore unmeasured:** precision, recall, F1, the `above/below` split that the assay
actually reports, stationary-fly sensitivity, overlap sensitivity. The measured phenotype-dependent
recall (0.88 healthy → 0.35 severe, N3) is strong reason to expect the last two to be poor.

**Required benchmark.** ~200 frames: 4 genotypes (**at least one strongly impaired line is
non-negotiable**) × 5 timepoints × 2 lighting conditions × ≥5 vials. Every fly's centroid marked by
**two independent annotators** with a third adjudicating. Report inter-annotator agreement **first**,
as the ceiling on any claim derived from the set.

| Metric | Reported |
|---|---|
| Precision, recall, F1 | at a one-fly-width matching radius |
| Count MAE and **signed** bias | separately — stability without accuracy is the failure mode already measured |
| `n_above` error | against the annotated scoring line, separately from total count error |
| Detection failure rate | per stratum |

**Stratified by crowding and by motion state, never pooled** — a pooled F1 would hide exactly the
confound in N3. Both detectors on identical frames. `validate_fly_blob.py`'s hold-out discipline
retained.

**Until this exists:** the only defensible claim is count MAE 2.69 with signed bias −1.60 on 110
held-out cells. No precision, recall or F1 figure may appear in the manuscript.

### 2.2 Calibration metrology

A perfectly flat, correctly printed card under barrel distortion k1 = −0.25 yields 6.83% scale error
and two simultaneous wrong diagnoses; at k1 = −0.45, 22.6%. The clean-card floor is 1.41%. This
propagates into `mean_height_mm`, the output promoted as cross-laboratory comparable.

Required: camera intrinsics and distortion correction on the lab's own cameras; a measured
uncertainty attached to `px_per_mm`; and an uncertainty budget for `mean_height_mm` against a
physical ruler at several positions in the field of view.

### 2.3 Acquisition timing

Measured: a camera asked for 24 fps delivering 15 produces a 1.6× time compression, so a requested
t = 10 s samples the frame captured at 16.0 s. Only nominal fps and requested duration are stored,
so this is **unrecoverable after the fact**. Required: real per-frame capture timestamps, and a test
that reconstructs true sample times from a recording made with a deliberately mismatched camera.

### 2.4 Activity and sleep

`docs/REFERENCE_DATASETS.md` §2a is now **unblocked** (N67). Bin width is readable per file from the
ZT increment, corroborated by the maximum cell value; day 1 = baseline, 2 = deprivation,
3 = recovery, with deprived-group sleep falling to 766.9 / 622.7 / 275.4 min on day 2 while controls
stay flat, and rebound of +187 / +141 / +165 min against +54 / +34 / +58 for controls. Column counts
match the paper's Figure 1 n.

| Suite | Validates | Limitation to state in the suite |
|---|---|---|
| sleep-homeostasis directional | per-day summary arithmetic under a file-declared bin width; the **direction** of homeostatic rebound | Column labels repeat within a file (flies must be keyed by position); absolute level 18.0–21.6 h/day exceeds every published DAM total-sleep value, so this is **not** an absolute sleep-amount benchmark |

**~~Still required~~ — CLOSED 2026-11-08.** Zenodo 3403481 was downloaded (Selkrig et al. 2018,
*Sci Rep* 8:10646, CC-BY 4.0) and is twelve genuine raw DAMSystem monitor files. Suite
`validate_dam_raw_microbiome`, 41/41. The ≥5-min rule now reproduces an implementation written
from the published definition and sharing no code with `modules/dams.py`, on all 379 mapped flies,
to **0.0500 min/day** — half of the last decimal place the module publishes, so exactly. The
criterion is shown to bite (1 / 5 / 25 min give 1,252 / 1,190 / 1,036 min/day on the same flies),
the agreement is not a tie at zero, and bad-status rows are confirmed to arrive as NaN rather than
as stillness.

That suite also produced the strongest **statistical** validation in the project, because the
paper's conclusion is negative — the microbiome has a *limited* influence on sleep — and the
per-fly test in this repository contradicts it: Kruskal-Wallis **p = 0.015** on 377 flies, against
**p = 0.372** at the replicate cohort (n = 19) and **p = 0.230** at the monitor (n = 16), with a
measured between-monitor ICC of 0.118 and a design effect of 4.60. The published conclusion is
recovered only when clustering is respected.

Also closed: the §2a sleep-homeostasis suite now exists (`validate_sleep_homeostasis`, 32/32) and
its characterisation was re-derived from the files rather than read across from the document —
which corrected the document on two points (see the register).

**What is still NOT established.** No annotated ground truth for sleep exists in any public
dataset small enough to use, so the sleep RULE is validated and the sleep MEASUREMENT is not.
Zenodo 5109970 (48.9 GB) remains the only candidate and remains too large to pull here.

**Also required:** make `plausibility_report` per genotype × sex rather than pooled-median (N63),
and re-run it over the existing published results — on which 31–33% of flies currently sit outside
the project's own plausible range while the gate reports OK.

### 2.5 Other assays

| Assay | Reference measurement | Status |
|---|---|---|
| Larval | published speed–curvature exponent, 123 larvae | **Done**, 19/19 |
| Grooming | 1,096 hand-scored bouts, body-part annotated | **Done**, 21/21, with the genotype-coding contradiction in the source correctly left unresolved rather than resolved circularly |
| Y-maze | hand-verified turn sequences | **Not done** — the only public individuality datasets are 2.1 TB and 12.6 TB. A locally scored sequence set of ~20 flies would suffice |
| Climbing velocity | FreeClimber, 4,093 detections | **Done**, 17/17 — but establishes nothing about DroPhenix's own detection, as that suite's docstring states |
| Climbing vs age, second independent benchmark | Thiem et al. 2024, per-fly escape climbing, 4 ages, typed failures | **Done 2026-11-08**, 43/43. The data guide was downloaded from the same Zenodo record, so the figure panels are labelled and the dataset the reference document had written off is usable. The paper's own Figure 1B summary — eight values, mean and SEM — is reproduced exactly. Three source-data notes raised and left unresolved (N92, N93, N94) |

---

## Layer 3 — statistical validity

Simulation with known truth, reporting **calibration**, not point recovery. Each entry names the
defect it would have caught.

| Analysis | Question | Target | Catches |
|---|---|---|---|
| Hierarchical | type-I error under vial clustering at ICC 0.05–0.20, 3–12 vials per group | nominal 5% ± simulation error, per route | N37a; extends the existing suite, the strongest in the project |
| Beta-binomial | fraction of non-identified fits that return a p-value | **0** | N16 |
| GEE | sandwich-SE calibration at 3, 5, 10, 20 clusters | refuse where miscalibrated, rather than report | N37a |
| Survival | KM/log-rank recovery; **PH violation detection rate** under a constructed crossing-hazard design | detection ≥ 80% | N40 |
| Survival | vial-frailty coverage at ICC > 0 | nominal 95% | N40 |
| RMST | coverage of the between-group difference interval | nominal 95% | N40 |
| Dose-response | EC50 and LC50 recovery and interval coverage under **binomial** error, at `d` of 1.0, 0.90, 0.80 and `b` of 1, 2, 4 | coverage ≥ 93%; LC50 bias < 1% | N14, N36 — **DONE 2026-11-08, `validate_statistical_calibration`. The coverage target is NOT met: 0.95 at b = 1, 0.80–0.86 at b = 2, 0.47–0.79 at b = 4, and it does not improve with design. Cause measured as the variance model, not the design or the algebra; see N95. Bias meets the target on an adequate design (+0.32% at 20 doses × 300) and not on an assay-sized one (+2.6%).** |
| Hormesis | type-I of the Brain–Cousens F test under Gaussian and binomial residuals | already measured at 5.0–5.2% — **pin it as a regression** | N58 |
| Meta-analysis | DL vs HKSJ coverage at k = 2–6 and τ² of 0, 0.1, 0.3 | HKSJ ≥ 93% where DL gives 88.2% | N35 — **DONE 2026-11-08, `validate_statistical_calibration`, target met. DL worst 0.776 at k = 2, τ² = 0.3; HKSJ ≥ 0.949 in all twelve cells and never worse than DL; width cost 1.32×–6.48×, worst at k = 2, where HKSJ over-covers at 1.000 (owner decision 10).** |
| Meta-analysis | shared-control covariance effect on between-drug differences | the measured 1.34× widening reproduced; the latent 0.80× case constructed and refused | N59 |
| Motor decline | individual **and** group slope recovery from replicate-level data; CI coverage | coverage ≥ 93%; CI within 10% of the mixed-model width | N23 |
| Survivor bias | informative-dropout detection rate **and** false-alarm rate under a null, over ≥ 400 replications | type-I ≤ 6% (currently **12.5%** on a replicated sweep) | N39c |
| Survivor bias | marginal vs conditional R² computed correctly, on a design with substantial between-animal variance | the 35× overstatement does not reproduce | N39d |
| Survivor bias | bootstrap interval on the slope ratio replaces the 0.85/1.15 verdict | interval reported; no categorical verdict without it | N39e |
| Trajectory | interval coverage on **held-out cohorts**, not later timepoints of the same cohort | nominal 95% | N28 |
| Trajectory | mandatory comparison against last-observation, train-mean, linear-extrapolation and population-curve baselines | already implemented — keep and extend to the cohort split | — |
| Circadian | period-estimate resolution as a function of record length | refuse a period claim the record cannot support; currently a true 25 h returns 24.00 h at 3 and 5 days | N45 |
| Circadian | cluster-robust rhythm-test calibration at realistic cluster counts | nominal 5% (currently **9%**) | N50 |
| Clustering | recovery of known subtypes from **replicate-level** trajectories; stability by ARI | recovered above chance, with stability reported | N30 |
| TCS / ranking | does the ranking recover known efficacy under missingness and toxicity | known-best compound ranked first with its interval; incomparable evidence sets refused | N32, N33, N34 |
| Synthetic generator | does the generator produce the structures the simulations above require | persistent FlyIDs, vial membership, dropout, correct `rho`; 0 FlyID collisions | N61 — **this is a prerequisite for the rows above it** |

**Ordering note:** N61 comes first. Several Layer-3 claims currently rest on a generator that does
not produce the structure being tested — independent per-assay draws sharing only group labels, no
individual longitudinal cohort, 86 colliding FlyIDs, and a `rho` of 0.36 realising as 0.71.

---

## Layer 4 — biological validation

The most important layer and the one no amount of simulation substitutes for.

1. **Known-direction biological contrasts.** Wild-type versus FUS and TDP-43 models; treatment
   versus vehicle; dose-dependent response. The test is whether DroPhenix recovers credible
   differences **without forcing the expected result** — so the analysis plan, the endpoint and the
   experimental unit must be pre-registered in the project before the data are analysed.
2. **Concordance across independent endpoints.** A compound that improves climbing is not
   neuroprotective. Required: motor function, survival, toxicity, and where available aggregation or
   pathology and a molecular endpoint — reported as an evidence profile with its discordances
   visible, not as a score.
3. **Independent reproducibility.** The same workflow on an independent batch, and on another
   laboratory's or a public dataset. Report measurement agreement, effect-size agreement, interval
   coverage, between-batch variability, failure rates, and analysis reproducibility from the
   manifest alone.
4. **Predictive modules on genuinely independent data.** Trajectory forecasting validated on cohorts
   not used for prior estimation or calibration. Until then it stays labelled research preview.

---

## Governance

- Every suite states, in its own output, which of L1–L4 it establishes.
- A fix lands only with a regression test that **fails before it and passes after**.
- A suite may not pass by testing the adjacent case. Each of the five identified instances gets the
  test the panel named.
- A refuted finding is recorded as refuted; correct code is not changed to satisfy a review.
- Convergence warnings, boundary fits and refusals are outputs, never suppressed. Four blanket
  `simplefilter("ignore")` blocks are removed as part of M1.
- No number enters the manuscript that has not been re-derived from the current tree. Six claims in
  the project's own documents already fail that test.

---

*Read-only audit output, 2026-11-07. Baseline at the time: 42 suites, 0 failures, exit 0.
P0 Blocks A–D and the published-data validation of 2026-11-08 have since been implemented; the
sections above are annotated where that changed their status.*
