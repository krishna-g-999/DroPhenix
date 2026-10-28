# Climbing Run Analysis — design for the autonomous pipeline (hand-scored counts)

**Status: PROPOSAL, 2026-09-20. Nothing here is implemented.** Awaiting approval.
Companion to the DAM pipeline (`docs/DAM_RUN_ANALYSIS.md`), which sets the contract:
raw file → quality and refusals → design read from the data → pre-specified tests
with FDR → sensitivity → figures → report verified cell by cell → a suite proving
the pipeline reproduces a known analysis.

## 0. What was inspected

| Piece | State | Consequence for the design |
|---|---|---|
| `modules/data_parser.py` | Canonical long schema (Genotype, Sex, Treatment, Replicate, Time, Count, n) plus optional design keys incl. `VialID`, `TrialIndex`, `Age`, `Dose`; reads CSV/Excel; wide→long | Reuse as the loader; add the lab's own sheet layout once the sheet is re-supplied |
| `modules/hierarchical.py` | ICC (Ridout 1999), beta-binomial regression (Williams 1975), GEE sandwich, mixed logit, `pseudoreplication_report`; direction of the clustering correction asserted in validation | The primary test engine already exists and is validated |
| `modules/normalizer.py` | PI, PI time course, AUC, t50 | Endpoints exist |
| `modules/motor_decline.py` | Decline over age; AICc model selection; validated on FreeClimber and Shahrestani/Rose 2016 (Zenodo 4954719) | Reuse when `Age` is present |
| `modules/dose_response.py` | Log-logistic, profile-likelihood EC50 | Reuse when `Dose` is present |
| `modules/data_triage.py`, `design_schema.py` | Confounding, replication, saturation, coverage checks; unit of replication picked from the columns | Reuse as the refusal layer |
| `stats_engine.py` | Rank tests a priori, exact min-attainable p, BH-FDR, van Elteren, Hedges' g | Reuse |
| CLI `triage`, `check` | Diagnostics only; no analysis, no report | The pipeline is what is missing between these and a report |
| **Ground truth** | `Testing data/Climbing assay/Paraquat climbing assay 010626_new set/` with `CLIMBING ASSAY PARAQUAT.xlsx` and the tidy `paraquat_ground_truth.csv` (1,080 hand-scored cells) **is no longer on disk** (folder absent since 31 Aug 2026). 144 of the cells survive as the `hand` column of `results/climbing_video/headtohead_simple.csv` | The real-data validation (section 6B) needs the sheet re-supplied |

Facts about the lab's design, from that sheet when it was available (recorded in
`docs/VIDEO_PHASE1_DESIGN.md`): one recording = one tap of ~2 min; 5 taps per vial;
3 replicate vials per condition; counts of flies above the 40 mL line at 12 times
(10–120 s); ~20 flies per vial; **ICC of taps within a vial = 0.573** (design effect
3.29), so five taps carry about 1.5 independent trials' worth of information;
tap-to-tap spread of a perfect counter median 3 flies.

## 1. Inputs

1. **Long CSV / Excel** in the canonical schema (the `templates/climbing_template.csv`
   layout), with optional `VialID`, `TrialIndex` (tap), `Age`, `Dose`, `Batch`, `Date`.
2. **The lab's own scoring sheet** — a parser written to its layout once it is
   re-supplied, so people upload what they already keep.
3. **The video batch table** from `drophenix_cli.py batch` (same schema, with its
   quality flags carried through and shown, never dropped).

The pipeline decides the **unit of replication** from the columns, in this order:
`VialID` → `Replicate` (within Genotype × Sex × Treatment × Age) → refuse inference.
Taps (`TrialIndex`) are repeated measures of a vial, never units.

## 2. Quality pass and refusals (before any test)

| Check | Action |
|---|---|
| `Count > n`, negative, non-integer | Row refused, listed |
| Time grid not shared across vials; missing times | Reported; endpoints computed only on the shared grid |
| Fewer than 5 flies in a vial | Warning; vial kept, flagged |
| One vial per group | **No inferential statistics for that contrast**; descriptive only, stated |
| Saturation (every vial in a group at 0 % or 100 % at the score time) | Warning: no power at that endpoint; AUC/t50 used descriptively |
| Treatment confounded with Batch/Date/Age (`data_triage`) | Contrast labelled confounded, excluded from the primary family (as monitor-clean in DAM) |
| Tap count differs between vials | Reported; per-vial mean over taps used, n taps shown |
| Score time not in the data | Refuse rather than interpolate; ask for the protocol time |

## 3. Endpoints (pre-specified, the published set)

| Endpoint | Definition | Role |
|---|---|---|
| **Fraction climbed at the score time** (and PI = 2·fraction − 1) | Protocol time *t_score*; default the earliest time ≥ 10 s present, always printed | **Primary** |
| AUC of the fraction-climbed time course | Trapezoid over the shared grid (existing `calculate_auc`) | Secondary |
| t50 | Time to half the plateau (existing `calculate_t50`); NaN when never reached, never imputed | Secondary |
| Decline rate over age | `motor_decline.fit_decay` with AICc; only when `Age` has ≥ 3 levels | When applicable |
| Dose–response | `dose_response` when `Dose` has ≥ 4 levels | When applicable |

Per-vial values are the mean over that vial's taps; the number of taps and the
within-vial tap SD are carried in the per-vial table so the irreducible spread is
visible next to the between-vial spread.

## 4. Statistics

**Why not a t-test / ANOVA on PI.** PI is a rescaled binomial proportion: its variance
depends on its mean and flies in a vial are not independent (ICC 0.573 for taps, and
a non-zero fly-level ICC is expected). Both are stated in `hierarchical.py` and both
are ignored by the conventional analysis.

**Primary test — beta-binomial regression, likelihood-ratio test** (Williams 1975;
`hierarchical.fit_betabinomial`), response = flies above the line out of *n* at
*t_score*, one observation per vial (taps averaged into the count), covariates =
the contrast factor plus Sex (stratification, as in DAM) and any blocking factor
present. It uses the within-vial information while estimating, not assuming, the
intra-vial correlation.

**Assumption-free companion — exact permutation of vial labels** on the vial-level
fraction, always reported beside it with its **minimum attainable p** (3 vs 3 vials
→ 0.10 two-sided). This is the honest statement of what the lab's design can do
without a model. When the two disagree the report says which assumption the
difference rests on.

**Small-cluster control (must be measured, section 6A).** GEE sandwich SEs are
anti-conservative with 3 clusters per arm; the beta-binomial LRT may be too. The
type I error of the primary test at 3, 4, 5 vials per group and ICC 0.05–0.6 will be
measured by simulation before it is shipped. If it exceeds 0.06 at the lab's design,
the primary switches to a small-sample-corrected variant (Bartlett-corrected LRT or a
parametric bootstrap of the LRT) and the design doc records the measured numbers.

**Effect sizes.** Difference in fraction climbed with a 95 % CI from the model;
PI difference; Hedges' g on vial-level PI (for `drug_ranking` pooling across strains),
with its wide CI at n = 3 shown, not hidden.

**Families and FDR.** One family per question (genotype; treatment within genotype;
age), BH-FDR at 5 % across the contrasts of the **primary endpoint** only; AUC and t50
reported as secondary, uncorrected, labelled — the same two-tier structure as DAM.

**Sensitivity in place of outlier removal.** With 3 vials per group a MAD-based
outlier rule is degenerate. Instead every verdict is recomputed **leaving each vial
out in turn**; a verdict that depends on one vial is labelled so. The tap-level ICC
and design effect are reported so the reader sees the clustering the data actually
contain.

## 5. Outputs (the DAM contract)

`per_vial.csv`, `timecourse.csv`, `contrasts_*.csv` (primary/secondary, family, q,
min attainable p, leave-one-out verdict), `quality.csv`; figures: time-course curves
with SEM ribbons per group (sexes in separate panels), endpoint bars with individual
vials, forest plot of the primary effects, decline curve when Age is present; a Word
report built from the run object and **read back by `verify()`**; app page
**Climbing Run Analysis** beside Climbing Curves; CLI `climb-run`.

## 6. Validation plan (the suite that must pass before it is announced)

A. **Simulation with known truth**: clustered binomial vials (ICC 0.05, 0.2, 0.6;
   3–6 vials/group; 15–25 flies) — effect recovered; **type I error of the primary
   test measured at the lab's 3-vials design** and printed; the vial-permutation p
   never below its attainable minimum; a confounded design refused.
B. **Real data**: the paraquat sheet (1,080 cells) — the pipeline reproduces the hand
   PI per vial and time, the tap ICC 0.573 and design effect 3.29, and the condition
   ordering WT untreated > WT 5 mM > WT 10 mM, TDP-43 < WT; **FreeClimber** and
   **Shahrestani/Rose 2016** already used by `validate_real_aging_benchmark.py` for
   the age model.
C. **Report**: builds and every cell traces (as `verify_dam_docx` / `dam_report.verify`).
D. **Refusals**: one vial per group → no p; Count > n → row refused; missing score
   time → refused.

## 7. What is needed from the laboratory

1. **The paraquat scoring sheet** (`CLIMBING ASSAY PARAQUAT.xlsx`) — it is no longer
   under `Testing data/`. Any further hand-scored climbing sheets, ideally with sex
   and age, strengthen section 6B.
2. **The protocol score time** (10 s? 30 s?) and the line height used, so the default
   is the lab's rule rather than mine.
3. Confirmation that the scoring rule is **net** (flies above the line at time *t*,
   up minus down), as established for the video work.

## 8. What this will and will not claim

It will claim to reproduce the hand analysis of the sheets it is validated on, with
vial-level inference and measured type I error at the lab's design. It will not
claim that 3 vials per group can establish a small effect: the minimum attainable
permutation p and the model-based p are both printed so that the reader sees which
one any significance rests on.
