# DroPhenix — Inversion Analysis (Pre-Mortem)

**Question asked in inverted form:** *It is 2028. The DroPhenix trajectory-prediction paper has
been retracted, or was rejected without review, or — worse — was published and a lab that trusted
it wasted two years and forty thousand flies chasing a compound that never worked. What went
wrong?*

This document enumerates the answers. Every entry is a concrete, checkable failure mode with a
mechanism, a detection test, and a mitigation. Entries are graded:

| Grade | Meaning |
|---|---|
| **FATAL** | Invalidates the central claim. Paper is wrong, not merely incomplete. |
| **MAJOR** | Biases estimates or inflates confidence enough to change conclusions. |
| **MINOR** | Degrades quality or reviewer confidence; not conclusion-changing. |

Status codes refer to the state of the codebase **as audited on 2026-08-03, before the uplift**;
file:line citations throughout point at that state. What has since been built in response is
described in `docs/UPLIFT_PLAN.md`, which also carries the measured results. Codes:
**OPEN** (present and unaddressed) · **PARTIAL** · **CLOSED** · **BY-DESIGN** (accepted, disclosed).

---

## Part A — Failures of the measurement itself

*Before any statistics: does the number mean what we say it means?*

### A1. The Performance Index is a binomial proportion analysed as a Gaussian variate — **FATAL**

`PI = (2·Count/n − 1)·100` (`modules/normalizer.py:47-56`) is a linear rescaling of the
proportion `p = Count/n`. Its sampling variance is therefore `4·10⁴·p(1−p)/n` — it depends on the
mean. Variance is maximal at `p = 0.5` and collapses to zero at both ends.

Every test in `stats_engine.py` (Welch's t, ANOVA, Mann–Whitney) assumes variance that does not
depend on the mean. Applying them to PI means:

- Groups near 50% climbing are systematically judged *more variable* than groups near 0% or 100%,
  independent of any biology.
- Levene's test (`stats_engine.py:243`) will flag "unequal variance" as an artefact of unequal
  means, then the router (`stats_engine.py:543-562`) silently downgrades to a rank test — a
  cascade in which a purely structural property of proportions changes which hypothesis test is
  reported.
- With `n = 20` flies per vial, PI takes only 21 distinct values. Normality tests on 3–5 such
  values have essentially no power, so the router's "all normal" branch is a coin flip.

**Detection:** plot residual variance against fitted mean; it will be an inverted parabola.
Simulate under a true null with `p` varying across groups and count false positives.

**Mitigation:** model the counts, not the index — binomial GLM/GLMM with a logit link, plus an
overdispersion term (beta-binomial or observation-level random effect) because flies in a vial are
not independent Bernoulli trials. Report PI as a *presentation* scale only, back-transformed from
the model. **Status: OPEN.**

### A2. Ceiling and floor truncation destroys the early and late trajectory — **MAJOR**

Young wild-type flies climb at ~95–100%. The measurement cannot resolve differences above the
ceiling, so the most informative early window — where a drug would ideally act — is compressed to
noise. At the other end, terminal flies sit at 0% and the assay cannot distinguish "barely alive"
from "dead an hour later".

Fitting an exponential decay through a censored ceiling biases the decay constant `k` downward
(the curve is held flat at the start), and biases the plateau upward.

**Detection:** count the fraction of vials at `Count == n` or `Count == 0`. If >10% of the
early cells are saturated, the early slope is not identified.

**Mitigation:** (i) work on the logit scale where the ceiling is at +∞ rather than a hard bound;
(ii) prefer a graded endpoint — height reached, or latency to cross the line — over pass/fail
when the ceiling is being hit; (iii) refuse to report an early slope when saturation exceeds a
threshold, rather than reporting a biased one. **Status: OPEN.**

### A3. The experimental unit is the vial, not the fly — **FATAL**

Flies in a vial share food, share the tap that startles them, share CO₂ exposure history,
temperature, humidity and handling. Their climbing outcomes are positively correlated. The
independent replicate is therefore the **vial**, and 20 flies in a vial carry substantially less
than 20 flies' worth of information.

The design effect is `DEFF = 1 + (m − 1)·ICC` for `m` flies per vial. At `m = 20` and a modest
`ICC = 0.05`, `DEFF ≈ 1.95` — the effective sample size is roughly *half* the nominal one, and a
naive standard error is too small by ~40%. At `ICC = 0.15`, `DEFF ≈ 3.85`.

**Detection:** fit a random-intercept model and report the ICC. Compare the naive p-value with the
cluster-corrected one; the ratio is the size of the error being made.

**Mitigation:** vial-level random intercept (or GEE with vial clusters and robust SEs). Report ICC
and effective sample size as a standard output, not an optional diagnostic. **Status: OPEN.**

### A4. The statistics page does not test replicates at all — **FATAL**

This is worse than A3 and is a property of the current wiring, not of the design. The Statistical
Analysis page passes `base_metrics["pi"]` (`app.py:1684`, `app.py:1728`), which is
`calculate_pi()`'s **group-level summary** — exactly one row per Genotype × Sex × Treatment
(`modules/normalizer.py:148-159`). The replicate-level table `pi_raw` exists
(`normalizer.py:354`) and is not used.

So when a user groups by Treatment, each "observation" entering the t-test is one
genotype×sex cell mean. With two genotypes and two sexes, a treatment comparison is a t-test on
n = 4 versus n = 4 — and those four numbers are means of means, with the within-cell variance
already discarded.

The same collapsed table feeds `sdq_permutation_test` (`app.py:809`, `app.py:1809`), which then
splits by Sex within Genotype × Treatment and finds exactly **one** male value and **one** female
value. Ten thousand permutations are drawn from a two-element pool. The module's own docstring
flags this (`stats_engine.py:657-658`) and the app ignores the warning.

**Detection:** assert that the number of rows entering any test equals the number of independent
vials. This is a unit test, not a judgement call.

**Mitigation:** route all inference through replicate-level data; make the group-summary table
display-only and structurally incapable of reaching a test function. **Status: OPEN.**

### A5. Repeated measurement changes the animal — **MAJOR**

The user's design tests the same flies morning, afternoon and evening, and across ages. Two
distinct artefacts are conceivable, and they act in opposite directions to the effects of interest:

1. **Within-session fatigue** — consecutive tap-and-climb trials reducing performance across trials.
2. **Across-session habituation** — repeated iterative geotaxis over days reducing the climbing
   index independently of ageing.

**The published evidence argues against both, and this must be stated plainly because it cuts
against the design's premise.** Gargano et al. (2005, *Exp Gerontol* 40:386-395), the paper that
introduced the RING assay, reports in its abstract: *"performance in RING assays is not influenced
by the density of flies being tested, the time of day, or repeated testing."* That is a published
null for all three of the confounds this section is about, and for the time-of-day effect the
design sets out to measure.

Three things follow, none of which is "abandon the design":

- The null was established in **wild-type** flies under one laboratory's conditions. Whether it
  holds in a neurodegeneration model with compromised motor circuits, or across a wider age range,
  or under toxin exposure, is genuinely open. A disease genotype could show a time-of-day
  modulation that wild-type does not — that is a real hypothesis.
- Because a strong prior null exists, the design must be powered and analysed to make a *negative*
  result interpretable. Finding no time-of-day effect would replicate Gargano; finding one is a
  claim against an existing published null and needs correspondingly strong evidence.
- Recording trial and session index costs nothing and converts the assumption into a check. The
  danger is not that fatigue certainly exists; it is that if it exists in this protocol and is not
  recorded, it is inseparable from ageing.

If morning is always assayed first and evening last, a monotonic within-day decline would be read
as a circadian trough. Counterbalancing removes that possibility rather than relying on the
literature null holding under new conditions.

**Detection:** include trial index within session and cumulative-sessions-to-date as covariates and
test whether they carry signal. Run a counterbalanced subset (evening-first) and check the
time-of-day estimate is invariant to order.

**Mitigation:** (i) counterbalance testing order across genotypes and across days — this must be
designed in, it cannot be fixed afterwards; (ii) include a *cross-sectional* arm of
never-before-tested flies at each age to identify any habituation component; (iii) model trial
index and session count explicitly. **Status: OPEN — and this is a design requirement, not only a
software one.**

*Note on how this entry was written:* an earlier draft of this document cited Gargano et al. 2005
as evidence that repeated RING testing *causes* fatigue and habituation. That was the reverse of
what the paper reports. The error was caught by checking the primary abstract rather than working
from memory, and is recorded here because it is precisely the failure mode entry E4 warns about.

### A6. Density, sex, mating status and handling confounds — **MAJOR**

Climbing depends on flies-per-vial (crowding), sex, mating status, time since CO₂ anaesthesia,
time since food change, and vial position in the rack (light gradient). Unequal group sizes mean
unequal density, so "unequal N" is not merely a statistical inconvenience — it is a *biological*
confound that correlates with treatment whenever attrition differs by group.

**Detection:** regress performance on `n` within control groups; a non-zero slope means density is
acting.

**Mitigation:** carry `n` as a covariate rather than only as a denominator; record and model rack
position, days-since-CO₂, and days-since-transfer. **Status: OPEN (n is a denominator only).**

---

## Part B — Failures of the statistical model

### B1. Survivor bias in every aging trajectory — **FATAL for the trajectory claim**

This is the single most dangerous failure in the entire project.

In an aging cohort, the weakest flies die first. The mean climbing score of the *survivors* at day
40 is therefore an average over a progressively selected, progressively fitter subpopulation. The
observed cohort mean can be flat — or even rise — while every individual fly is declining. Any
decline constant `k` fitted to cohort means is biased toward zero, and the bias grows with age and
is *larger in the sicker genotype*, because that genotype loses more of its frail flies.

The direction of the error is exactly the direction that manufactures false drug efficacy: a
neurotoxic genotype that kills frail flies early will appear to have a *shallower* motor decline
among survivors.

The literature makes this concrete. In longitudinal single-fly tracking of wild-type *Drosophila*,
~64% of flies show locomotor impairment and impairments appear predominantly in **the last 4–6 days
of life (~8% of lifespan)**, with >80% impaired within 48 h of death (Gaitanidis et al. 2019,
*Aging* 11:1850-1873). Motor collapse in flies is largely a **terminal** event, phase-locked to
death, not a smooth function of chronological age.

**Consequences:**
- Fitting `PI(age)` — which is what `compute_mdr_table` does (`modules/motor_decline.py:343`) —
  models the wrong time axis for the dominant biological process.
- Comparing genotypes with different lifespans at the same chronological age compares animals at
  entirely different points in their terminal trajectory.

**Detection:**
- Compare the cohort-mean trajectory with the trajectory of the balanced subset of flies that
  survive to the final timepoint. Divergence is the bias, measured directly.
- Fit on a **time-to-death** axis and compare predictive performance against the chronological-age
  axis. If time-to-death wins, age-based decline constants are confounded.

**Mitigation:**
- Inverse-probability-of-censoring weighting on the longitudinal fit.
- A joint longitudinal–survival model with a shared random effect (frailty), so that mortality
  informs the trajectory rather than silently truncating it.
- Report both axes (chronological age *and* aligned time-to-death) as standard output.
- Report a **survivor-bias diagnostic** with every decline constant: how much `k` changes when
  estimated on survivors-only versus IPCW-corrected.

**Status: OPEN. This is the highest-value scientific contribution available in this project,
precisely because the fly field routinely fits cohort means over chronological age.**

### B2. Motor-decline constants are reported with no uncertainty at all — **MAJOR**

`compute_mdr_table` calls `curve_fit` and discards the returned covariance matrix on the same line:
`popt, _ = curve_fit(...)` (`modules/motor_decline.py:148`). Consequently `k` has no standard
error, no confidence interval and no p-value; `t_half = ln2/k` (`:159`) has no interval; and
`delta_k = k_vehicle − k` and `Rescue_pct` (`:428-431`) — the numbers a user would put in a
figure and a grant — are bare point estimates.

Model choice is a hard `R² ≥ 0.5` threshold (`:192-201`) rather than an information criterion, and
trajectory classes ("Sustained Rescue", "Progressive Decline") are assigned by fixed cut-points on
those uncertainty-free point estimates (`:462-490`). A `k` estimated from three noisy points and a
`k` estimated from twelve clean ones receive the same label with the same confidence.

Additionally the fit is performed on group-mean PI per age (`:279`), so between-vial scatter — the
only honest source of error — is destroyed before fitting, and each age contributes one unweighted
point.

**Mitigation:** retain the covariance; propagate to `k`, `t½`, `Δk` and `Rescue%` by delta method
*and* by bootstrap over vials; fit on replicate-level data with vial random effects; replace the R²
threshold with AICc; refuse to emit a trajectory class when the interval spans the cut-point.
**Status: OPEN.**

### B3. Multiplicity across the full analysis surface — **MAJOR**

BH-FDR is implemented (`stats_engine.py:157`) and applied *within* a comparison family. But the
platform now spans six assays × multiple genotypes × treatments × sexes × ages × times of day, and
the Drug Ranking page composites across all of them. The family that matters for a screening claim
is the whole surface, and nothing tracks it.

**Mitigation:** a global registry of every test performed in a session, with a session-level FDR
report and an explicit statement of which family each q-value belongs to. **Status: OPEN.**

### B4. Meta-analytic pooling of dependent effect sizes — **MAJOR**

Drug Ranking pools Hedges' *g* across strains by inverse-variance / DerSimonian–Laird
(`modules/drug_ranking.py:91`, `:269`), then composites across assays. Two independence assumptions
are violated:

- The same vehicle control is typically shared by several treatment arms, so their *g* values are
  correlated (shared denominator and shared control mean). Ignoring this understates the pooled SE.
- Assay-level scores from the same flies (climbing and lifespan on one cohort) are strongly
  correlated; compositing them as independent evidence double-counts.

**Detection:** simulate a screen with a shared control and compare nominal versus empirical CI
coverage of the pooled estimate.

**Mitigation:** covariance-aware pooling (multivariate meta-analysis, or at minimum a correction
for shared controls) and an explicit correlation matrix across assays. **Status: OPEN.**

### B5. Cox models fitted without clustering — **MINOR-to-MAJOR**

`PHReg.from_formula(...).fit()` (`modules/survival.py:461-462`) is called with no `groups=`, no
cluster-robust covariance and no frailty term. When flies are housed in vials — and they always are
— deaths within a vial are correlated (a bad food batch kills a whole vial). Standard errors on
hazard ratios are consequently too small.

**Mitigation:** cluster-robust sandwich SEs by vial, or a shared-frailty Cox model.
**Status: OPEN.**

---

## Part C — Failures of the prediction claim

*The moment we say "predict", the evidentiary bar changes: a fitted curve is a description, a
forecast is a testable claim.*

### C1. Validating a forecaster with a random split — **FATAL**

Randomly holding out timepoints from within the same flies' trajectories leaks the future into the
training set: the model sees day 30 and day 40 while predicting day 35, which is interpolation
dressed as forecasting. Reported error will be several-fold too optimistic.

**Mitigation:** validation must be **forward-in-time** (train on `age ≤ T`, test on `age > T`) and
additionally **leave-one-cohort-out** and **leave-one-genotype-out**, reporting error as a function
of forecast horizon. The prototype at
`Paper Mining/modules/trajectory_model.py:231` already does forward-in-time correctly; that
property must be preserved and extended, not lost in integration. **Status: PARTIAL.**

### C2. Accuracy without calibration — **MAJOR**

A model can have good RMSE and badly wrong intervals. If the 95% prediction interval contains the
truth 60% of the time, every downstream power calculation and go/no-go decision is wrong. Accuracy
and calibration are separate claims and both must be reported.

**Mitigation:** empirical coverage of 50%/80%/95% intervals as a function of horizon; a calibration
plot; and the continuous ranked probability score (CRPS), which scores the whole predictive
distribution rather than only its centre. **Status: OPEN.**

### C8. The calibrator is itself pseudoreplicated — **MAJOR** (found 2026-08-07, FIXED)

Not predicted in the original pre-mortem, and it should have been: it is failure mode A1
(pseudoreplication) reappearing inside the machinery built to fix C2.

`build_conformal_calibrator` collected one standardised residual per held-out timepoint and gated
the correction on having at least 20 **residuals**. But one cohort contributes a residual at every
held-out age, and a cohort that declines faster than the prior expects sits below its prediction at
all of them. Measured on simulated panels the residuals correlate within cohort at **ICC = 0.683**
(design effect 3.05), so 240 residuals from 60 cohorts carry about 79 independent observations, not
240. The exchangeable unit that split-conformal validity rests on is the **cohort**.

The consequence was not conservatism, it was harm. Coverage of a nominal 95% interval, measured
against a known truth:

| calibration cohorts | residuals | empirical coverage |
|---|---|---|
| none (uncalibrated) | – | 0.882 |
| 6 | 24 | 0.882 (correction did nothing) |
| 10 | 40 | **0.751** (worse than not calibrating) |
| 20 | 80 | 0.819 |
| 60 | 240 | 0.973 |

A user with ten cohorts cleared the gate and received intervals labelled 95% that covered 75% —
precisely the defect the module's own docstring exists to denounce.

**Mitigation (applied):** gate on distinct cohorts per horizon bucket, not residuals.
`MIN_COHORTS_95 = 40`, because a two-sided 95% conformal interval takes the ⌈(n+1)·0.975⌉-th order
statistic of the calibration residuals and that order statistic exists only for n ≥ 39; below it the
upper quantile is an interpolation inside the sample and the finite-sample guarantee does not hold.
`MIN_COHORTS_50 = 12`, imposed for stability rather than validity (the conformal bound there is
n ≥ 3). Too few cohorts now **refuse** the correction and fall back to the uncorrected interval with
a stated reason, rather than returning a confident wrong number. `ConformalCalibrator` carries
`cohorts_per_bucket` alongside `n_per_bucket` so the distinction is visible. Regression-tested at 6,
10 and 20 cohorts in `validation/validate_trajectory_prediction.py` §F; the 60-cohort path is
unchanged and still reaches 0.949. **Status: FIXED.**

### C3. No baseline — **FATAL for the novelty claim**

A trajectory model that does not beat trivial baselines is not a method. The mandatory comparators:

1. **Last observation carried forward.**
2. **Group mean at the target age** (i.e. ignore the individual entirely).
3. **Population curve** — the average trajectory for that genotype, ignoring this cohort's data.
4. **Simple linear extrapolation** of the last two observed points.

If the hierarchical model does not beat all four at the horizons claimed, the honest report is that
it does not. **Status: OPEN.**

### C4. Model-form uncertainty dominates extrapolation — **MAJOR**

Predicting day 40 from days 3–14 is almost entirely a statement about curve *shape*. Exponential,
logistic and Gompertz families can be indistinguishable over the observed window and diverge
wildly beyond it. AICc model averaging (`trajectory_model.py:170-177`) is the right instinct, but
AICc weights computed from 3–4 points are themselves extremely noisy, and the current code silently
falls back from AICc to AIC when `n ≤ p + 1` (`:100-103`).

**Mitigation:** report the per-family divergence explicitly (the spread across families *is* the
honest uncertainty); cap the extrapolation horizon at which any prediction is issued; and let the
population prior — not the individual's 3 points — carry the shape at long horizons.
**Status: PARTIAL (flagged honestly, not yet solved).**

### C5. The population prior does not exist — **FATAL for the trajectory module as currently built**

`Paper Mining/scripts/06_fit_trajectories.py` declares its own status at line 23: **"SKELETON"**,
and `empirical_bayes_predict()` returns `{"status": "SKELETON -- posterior fit not yet
implemented"}`. Its input, `outputs/literature_master.xlsx`, does not exist; scripts 02 and 03 have
never produced their outputs, so 04, 05 and 06 have never run past their input guards.
`data/extracted/`, `data/verified/` and `data/synthesized/` are all empty.

So the component that is supposed to make far extrapolation trustworthy is currently absent, and
nothing in `DroPhenix-main` consumes it.

**Mitigation:** build the hierarchical layer against data that actually exists — the lab's own
historical control cohorts — rather than against an unbuilt literature corpus. **Status: OPEN.**

### C6. Distribution shift between models — **MAJOR**

A prior learned on TDP-43 flies applied to α-synuclein flies, or on one genetic background applied
to another, or on 25 °C data applied to 29 °C data, will be confidently wrong. Transfer must be
measured, not assumed.

**Mitigation:** leave-one-disease-out transfer test with the failure reported as a headline number,
not buried; and a prior-data conflict statistic that refuses to borrow when the new cohort is
incompatible (see D3). **Status: OPEN.**

### C7. The prediction target is not decision-relevant — **MAJOR**

"PI at day 30 = 42 ± 15" is not what anyone decides on. The decisions are: *will this compound
delay the onset of impairment?*, *by how many days?*, *will this fly cohort still be measurable at
day 35?* A forecast of a scale-free index at an arbitrary day answers none of them.

**Mitigation:** report derived, interpretable quantities with intervals — predicted **onset age**
of impairment (first crossing of a defined threshold), predicted **healthspan** (time above
threshold), and predicted **days gained** versus vehicle. **Status: OPEN.**

---

## Part D — Failures specific to the new designs

### D1. Time-of-day effects confounded with everything — **MAJOR**

*Drosophila* is crepuscular: activity peaks at dawn and dusk with a midday siesta, and locomotor
performance varies with circadian phase. Standard protocols control this by testing at a fixed time
of day. The moment the design deliberately varies time of day, three confounds open:

- **Order effects** — see A5. Time of day is perfectly collinear with cumulative fatigue unless
  order is counterbalanced.
- **Light history** — a morning assay follows lights-on; an evening assay follows a full photophase.
  Light exposure itself modulates activity.
- **Feeding state** — flies fed at a fixed time are at different post-prandial states in the morning
  and evening.

There is also a genuine scientific opportunity here, and it needs the right model: circadian
*amplitude* may collapse before circadian *mean* does in neurodegeneration, making time-of-day
amplitude an earlier biomarker than mean climbing. That is a testable hypothesis, and it requires a
cosinor model with amplitude and acrophase estimated per genotype with confidence intervals — not
three separate t-tests on three time points.

**Detection:** counterbalanced order subset; test for a time-of-day × order interaction.

**Mitigation:** cosinor regression (MESOR, amplitude, acrophase) nested inside the hierarchical
model, with trial order as a covariate. Three time points per day is the bare minimum to identify a
24 h cosine (3 parameters, 3 points, zero residual df) — so amplitude is only identified by pooling
across flies or days. This must be stated, not glossed. **Status: OPEN (no time-of-day column
exists anywhere in the parser).**

### D2. Historical control borrowing that injects bias — **FATAL if done naively**

Using previous experiments' controls to reduce the number of flies in the current control arm is
scientifically and ethically attractive (fewer animals, more treatment arms per experiment). It is
also the fastest way to manufacture a false positive.

If the historical controls differ from what the current controls would have been — fly stock drift,
a new food batch, a different incubator, a different season, a different experimenter — then
borrowing shifts the control estimate and the treatment effect absorbs the difference. Type I error
can rise far above 5% and the inflation is invisible in the analysis output.

**Detection:** *quantify it, do not argue about it.* Simulate the whole procedure across a grid of
true drift between historical and current controls, and report type I error and power at each drift
level. The honest deliverable is a table showing exactly how much bias the method tolerates before
type I error exceeds a stated bound.

**Mitigation:** a **robust meta-analytic-predictive prior** (Schmidli et al. 2014, *Biometrics*
70:1023-1032): derive the predictive distribution for a new experiment's control parameter from a
random-effects meta-analysis of past control arms — so between-experiment heterogeneity `τ²` is
estimated rather than assumed away — then mix it with a weakly informative component so that a
conflicting current control can overwhelm the prior. Report the prior's **effective sample size**
in flies, which is the honest statement of what is being borrowed. **Status: OPEN (no historical
control machinery exists; no Bayesian machinery of any kind exists in the repo).**

### D3. Prior–data conflict passing silently — **FATAL**

Even a robust prior fails if nobody checks whether it conflicts with the data. A borrowed control
that is 15 percentage points off must trigger a visible, blocking warning, not a footnote.

**Mitigation:** a formal prior–data conflict statistic computed and displayed on every borrowed
analysis, with a hard refusal threshold above which the analysis reverts to the concurrent control
only. **Status: OPEN.**

### D4. Toxin dose-response failures — **MAJOR**

- **Too few doses.** A four-parameter log-logistic curve has four parameters; fitting it to three
  doses is not an estimate. EC50 confidence intervals from the delta method are badly wrong when
  the curve is not well-identified, and near the boundary they can include impossible values.
- **Hormesis.** Low-dose stimulation followed by high-dose inhibition is common in toxicology. A
  monotonic Hill fit will average it away and report a misleading EC50.
- **Toxicity confounded with mortality.** A toxin that kills flies looks like it "reduces climbing"
  only among the survivors — this is B1 again, in a new costume, and it is the standard failure
  mode of behavioural toxicology. The behavioural endpoint must be interpreted jointly with the
  mortality endpoint or it is uninterpretable.
- **Nominal versus delivered dose.** Concentration in food is not dose ingested. Flies eat less of
  bitter or aversive compounds, so the nominal high dose may deliver less compound than the mid
  dose. Feeding must be measured or the possibility disclosed.

**Mitigation:** profile-likelihood intervals for EC50/LC50; an explicit Brain–Cousens hormesis test
against the monotonic model; joint reporting of behavioural and mortality dose-response with a
survivors-only-versus-corrected diagnostic; a required field for whether dose is nominal or
measured. **Status: OPEN (no dose, concentration or compound column exists in the parser; zero
occurrences of EC50/LC50/Hill anywhere in the repo).**

---

## Part E — Failures of validation and reporting

### E1. Validating simulations against the parameters that generated them — **MAJOR**

Of 12 validation suites, **2** use real published raw data (`validate_stats.py` on PlantGrowth;
`validate_survival.py` on Freireich 1963). Of 18 reference datasets, **1** is real raw data
(Freireich) and 1 uses real published parameters (Konopka–Benzer periods) with simulated traces.
The remaining 16 are simulated from published summary parameters and then checked against those
same parameters.

That is a valid test of *implementation* — it proves the code computes what it claims — but it is
circular as a test of *scientific validity*, and a methods reviewer will say so. It cannot detect a
mis-specified model, because the simulator and the estimator share the same assumptions.

**Mitigation:** (i) state the distinction explicitly in the manuscript rather than presenting all
suites as equivalent; (ii) add **simulation-based calibration** — check that posterior/interval
coverage is nominal across the prior predictive, which *can* detect mis-specification; (iii)
prioritise obtaining real raw data (the lab's own cohorts are the most accessible source, and they
have the enormous advantage of being the actual target population). **Status: PARTIAL —
`PROVENANCE.md` labels each dataset honestly; the manuscript does not carry the distinction
forward.**

### E2. A benchmark denominator that excludes the disagreements — **MAJOR (integrity)**

`comparison_report.md` contains 26 metric rows: 24 tagged `yes` and 2 tagged `info`.
`validation/comparison_study.py:257-258` computes the total from rows tagged `yes`/`NO` only, so
`info` rows are excluded from the denominator *by construction*. The headline "24/24 metrics agree,
r = 1.0000" is therefore arithmetically guaranteed regardless of what the excluded rows contain.

One excluded row is a real numerical disagreement (Cox HR under Efron ties: 4.5231 vs lifelines
4.81687). It is defensible on the merits — DroPhenix uses Breslow ties and matches R's Breslow
result and the published value exactly — but a denominator that cannot count disagreements is not a
benchmark, and a reviewer who notices will distrust everything else.

**Mitigation:** report `24 agree / 2 method-difference / 26 total`, with the two differences named
in the main text. The underlying science survives this change intact; the framing does not.
**Status: OPEN.**

### E3. Unreported dependencies and unreachable code — **MINOR**

`lifelines`, `astropy` and `pyBOAT` are named in the manuscript abstract as benchmark comparators
but are absent from `requirements.txt`, so the reported comparison cannot be reproduced from a
clean install. A `"Live Recording"` page exists in `PAGE_MAP` (`app.py:3397`) but not in `PAGES`
(`app.py:154-173`), so it is unreachable. `app.py:297-555` is a ~258-line dead string literal of a
superseded home page. `_msci_interpret` and `_render_mdr_plotly` are each defined twice
(`app.py:927`/`1451`, `app.py:941`/`1311`), the later shadowing the earlier.

**Mitigation:** pin the benchmark dependencies in a `requirements-validation.txt`; delete dead
code; resolve duplicate definitions. **Status: OPEN.**

### E4. Reference integrity — **MAJOR if unfixed at submission**

The manuscript carries a self-flagged caveat (`manuscript.md:12`) that pagination and DOIs are to
be confirmed, and reference 5 (`manuscript.md:161`) has no author list. A previous pass already
caught one fabricated attribution (a spontaneous-alternation paper attributed to the wrong first
author). That is evidence the failure mode is live, not hypothetical.

**Mitigation:** every reference verified against the primary record before submission, with the
verification recorded. No reference cited from memory. **Status: OPEN.**

---

## Part F — Failures of the publication strategy

### F1. The target journal does not publish this kind of paper — **FATAL to the stated goal**

*Nature Methods* publishes methods that are new and demonstrably better, benchmarked against
existing tools on real data, usually with a biological finding the method made possible. A web
application that applies standard statistics to standard assays — however carefully — is a
*Bioinformatics Applications Note* or a *NAR Web Server* paper, and will be desk-rejected by
*Nature Methods* without review. The existing manuscript is already correctly aimed at NAR Web
Server.

Being blunt about what would change that: the paper would need a methodological result that is true
independent of the software. The candidates present in this project are:

1. **Survivor-bias-corrected motor trajectories.** A demonstration that the standard practice —
   fitting cohort-mean climbing against chronological age — systematically underestimates decline,
   with the magnitude of the bias quantified, and a corrected estimator that recovers the truth in
   simulation and changes conclusions on real data. This is a genuine, checkable, field-wide claim.
2. **Validated historical control borrowing for invertebrate behavioural screens.** Quantified
   fly savings at controlled type I error — an experimental-design result with a direct 3R benefit.
3. **Terminal-decline alignment.** If time-to-death predicts motor state substantially better than
   chronological age (as Gaitanidis et al. 2019 implies), then age-matched comparison of genotypes
   with different lifespans is confounded — which would affect a large existing literature.

Each is a *finding*, not a feature. Any one of them, established on real data with the software as
the vehicle, is a far stronger paper than a feature list. **Status: strategic — see
`docs/UPLIFT_PLAN.md`.**

### F2. Claiming novelty that already exists — **FATAL (credibility)**

Joint longitudinal–survival models, MAP priors, cosinor rhythmometry, four-parameter log-logistic
dose-response and binomial GLMMs are all established methodology with decades of literature. The
contribution here is their **correct application to a field that does not currently use them**, and
the demonstration of *how much difference it makes*. Claiming to have invented any of them would be
fatal, and is unnecessary — the applied result is the contribution.

**Mitigation:** every method cited to its primary source; novelty claimed only for the application,
the quantified magnitude of the correction, and the integrated implementation. **Status: policy.**

### F3. No real data — **FATAL**

Every failure mode above is currently demonstrated on simulated data. Simulation proves the
estimator recovers the truth *under its own assumptions*; only real data proves the assumptions
resemble reality. The lab's incoming cohorts — young and old, control and mutant, across times of
day, with toxin arms — are exactly the data required, and no substitute exists.

**Status: BLOCKING. No lab-generated raw data files were found anywhere on this machine at the time
of writing. The ingestion, analysis and validation path must be built and tested against
specification-conformant simulated data so that it runs the moment real data arrives — but no
claim about biology can be made until it does.**

---

## Consolidated risk register

| ID | Failure | Grade | Status |
|---|---|---|---|
| A1 | Binomial proportion analysed as Gaussian | FATAL | OPEN |
| A2 | Ceiling/floor truncation biases slopes | MAJOR | OPEN |
| A3 | Vial is the experimental unit; flies are not independent | FATAL | OPEN |
| A4 | Statistics page tests cell means, not replicates | FATAL | OPEN |
| A5 | Repeated testing causes fatigue/habituation | MAJOR | OPEN |
| A6 | Density, sex, handling confounds uncontrolled | MAJOR | OPEN |
| B1 | Survivor bias in aging trajectories | FATAL | OPEN |
| B2 | Decline constants reported without uncertainty | MAJOR | OPEN |
| B3 | Multiplicity across the whole analysis surface | MAJOR | OPEN |
| B4 | Pooling of dependent effect sizes | MAJOR | OPEN |
| B5 | Cox fitted without vial clustering | MAJOR | OPEN |
| C1 | Random split leaks the future | FATAL | PARTIAL |
| C2 | Accuracy reported without calibration | MAJOR | OPEN |
| C3 | No trivial baselines to beat | FATAL | OPEN |
| C4 | Model-form uncertainty dominates extrapolation | MAJOR | PARTIAL |
| C5 | Population prior does not exist (skeleton) | FATAL | OPEN |
| C6 | Distribution shift across disease models | MAJOR | OPEN |
| C7 | Prediction target not decision-relevant | MAJOR | OPEN |
| C8 | Conformal calibrator gated on correlated residuals | MAJOR | FIXED |
| D1 | Time-of-day confounded with order and light history | MAJOR | OPEN |
| D2 | Naive historical borrowing inflates type I error | FATAL | OPEN |
| D3 | Prior–data conflict passing silently | FATAL | OPEN |
| D4 | Dose-response: few doses, hormesis, mortality confound | MAJOR | OPEN |
| E1 | Simulations validated against their own parameters | MAJOR | PARTIAL |
| E2 | Benchmark denominator excludes disagreements | MAJOR | OPEN |
| E3 | Unreported dependencies, dead and unreachable code | MINOR | OPEN |
| E4 | Reference integrity not verified | MAJOR | OPEN |
| F1 | Target journal mismatch | FATAL | STRATEGIC |
| F2 | Overclaiming methodological novelty | FATAL | POLICY |
| F3 | No real data | FATAL | BLOCKING |

---

## The inverted conclusion

To make this project fail, one would do exactly what is currently being done: analyse proportions
as Gaussians, treat flies as independent, test collapsed cell means, fit decline curves to
survivor-biased cohort means over chronological age, report decline constants without intervals,
extrapolate from a prior that was never built, validate simulations against their own parameters,
and count a benchmark with a denominator that cannot express disagreement.

Every one of these is fixable, and each fix is independently publishable-grade work. The order that
follows is therefore not arbitrary: measurement model before statistical model, statistical model
before prediction, prediction before borrowing, and real data before any biological claim.
