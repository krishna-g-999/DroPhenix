# DroPhenix uplift — what was found, what was built, what the lab must do next

Companion to `docs/INVERSION_ANALYSIS.md`, which enumerates the failure modes this work
addresses. Read that first; this document is the response to it.

---

## 1. What the audit found

The pre-existing platform is competent software with a serious statistical hole underneath it.
Six assays, actuarial survival analysis, circadian periodograms and meta-analytic drug ranking were
all implemented and benchmarked against seven independent tools. But every inferential path in it
made the same three assumptions, and all three are false for climbing data:

| Assumption made | Reality |
|---|---|
| The Performance Index is a Gaussian variate | It is a linear rescaling of a binomial proportion; its variance depends on its mean |
| Flies are independent observations | Flies share a vial; the vial is the experimental unit |
| The trajectory of a cohort mean is the trajectory of a fly | It is the trajectory of a *survivor*, and the selection strengthens with age |

Two findings were worse than the assumptions themselves:

- **The Statistical Analysis page never tested replicates.** It was fed
  `base_metrics["pi"]`, the group-level summary produced by
  `modules/normalizer.py:148-159` — one row per Genotype × Sex × Treatment. Every "n" entering
  every test was a mean of means. The replicate-level table existed and was not used. The sex-
  dimorphism permutation test drew 10,000 permutations from a pool of two numbers, and the module's
  own docstring said so.
- **Motor-decline constants were reported with no uncertainty at all.** `curve_fit` returns the
  parameter covariance; `modules/motor_decline.py:148` discarded it on the same line
  (`popt, _ = curve_fit(...)`). Decay constant `k`, half-life, `Δk` and "Rescue %" — the numbers a
  user puts in a figure — had no standard error, no interval and no test, and trajectory classes
  ("Sustained Rescue", "Progressive Decline") were assigned by fixed cut-points on those bare
  point estimates.

Two reporting-integrity problems were also found and fixed:

- The benchmark headline "24/24 metrics agree" used a denominator computed from rows tagged
  `yes`/`NO` only, excluding the `info` rows *by construction* — so it could not express a
  disagreement. It now reads **24 agree / 0 disagree / 2 documented method differences, out of 26**,
  with both differences named in the report.
- A citation error introduced during this very work: Gargano et al. (2005) was cited as evidence
  that repeated RING testing causes fatigue and habituation. The paper reports the **opposite** —
  see §4.

---

## 2. What was built

Seven new modules, each with a runnable validation suite. All are wired into
`validation/run_all.py`.

### `modules/hierarchical.py` — the measurement model
Beta-binomial regression with a logit link (the intra-vial correlation ρ is a fitted parameter, not
an assumption), population-averaged GEE with cluster-robust standard errors as an independent
cross-check, a mixed model on the empirical logit for random slopes, and the ICC / design effect /
effective-sample-size diagnostic.

The headline result, from `validation/validate_hierarchical.py`:

> Under a **true null** with 10 vials per arm, 20 flies per vial and ICC = 0.10, the naive
> fly-as-independent test rejects **25.9% of the time** — a 5.2× inflation of the nominal 5%. The
> vial-level test rejects 4.1% and the beta-binomial 5.7%.

`pseudoreplication_report()` runs both analyses on the user's own data and shows the two p-values
side by side. It refuses, rather than returning a number, when a treatment has only one vial —
because then the treatment effect and the vial effect are the same parameter.

Validated against: the exact one-way random-effects ICC identity (agreement to 1e-10), scipy's
`f_oneway` F statistic, and statsmodels `GLM(Binomial)` in the ρ→0 limit (max coefficient
difference 1.7e-4).

### `modules/survivor_bias.py` — the estimand problem
Four estimators of the same trajectory, computed side by side so their disagreement is visible:
survivor mean (the conventional curve), balanced panel, random-slope mixed model, and a Little
(1993) pattern-mixture reweighting to the full starting cohort. Plus an informative-dropout test, a
time-to-death axis, and a formal comparison of which time axis fits better.

From `validation/validate_survivor_bias.py`, against a generative model with a known true slope of
−0.0120/day:

| Estimator | Estimate | Error |
|---|---|---|
| Cohort mean (conventional) | −0.00781/day | **35% understated** |
| Balanced panel | −0.00808/day | 33% understated |
| Mixed model | −0.01222/day | 2% |
| Pattern mixture | −0.01174/day | 2% |

And the scenario that matters most — **two genotypes with identical true decline but different
lifespans**:

> Cohort-mean slopes: −0.00852/day (long-lived) vs −0.00190/day (short-lived) — an apparent
> 4.5-fold difference in decline rate, entirely manufactured by differential mortality.
> Mixed-model slopes: −0.01202 vs −0.01131 — gap reduced from 0.0066 to 0.0007.

That is a false drug effect, quantified. Any manipulation that shortens lifespan will appear
neuroprotective under a cohort-mean climbing analysis.

The module also declines to cry wolf: with the frailty coupling switched off it correctly reports
no informative dropout (ρ = +0.002, p = 0.97). Notably, a *raw* mean-value-versus-death-age
correlation gives ρ = −0.38 (p = 4e-15) in that same null case — a pure artefact of long-lived
flies being observed at later, lower-performing ages. The module uses the age-adjusted intercept
specifically to avoid it.

### `modules/trajectory.py` — forecasting with a real population prior
Four curve families on an unconstrained, bounded parameter scale; AICc model averaging when there
is no prior; a two-stage empirical-Bayes population prior estimated by EM from completed cohorts;
forward-in-time validation only; CRPS; mandatory comparison against four trivial baselines; and
split-conformal interval calibration.

From `validation/validate_trajectory_prediction.py`, forecasting days 30–45 from only days 5, 10
and 15:

> MAE **0.0958 with** the population prior vs **0.1516 without** — a 37% reduction. Beats
> last-observation-carried-forward in 98% of cohorts, the train mean in 100%, and linear
> extrapolation of the last two points in 83%.

Three genuine bugs were found by this suite and fixed:

1. **Collapsing variance.** With three ages and a three-parameter curve, the fit passes through
   every point, the residual sum of squares is numerically zero, and the parameter covariance
   collapses — the model reported a perfect fit and near-zero intervals in exactly the sparse case
   it exists to serve. Fixed by falling back to the population's residual SD when there are no
   residual degrees of freedom.
2. **An EM absorbing state.** Initialising the between-cohort covariance from the
   moment-corrected value (raw minus mean within-cohort covariance) can go negative for weakly
   identified parameters; clipping it to zero implies total shrinkage, total shrinkage implies zero
   apparent spread, and Σ stays at zero forever. The symptom was a prior claiming every cohort was
   identical (estimated SDs 0.016/0.059 against a truth of 0.25/0.40). Fixed by initialising from
   the raw covariance, which overestimates, and letting EM shrink down.
3. **Systematic over-confidence.** Even after those fixes, nominal-95% intervals covered 70%.
   Split-conformal calibration on held-out cohorts brings this to **95.6%**. The calibrator refuses
   cohorts that were also used to build the prior, since in-sample residuals would under-correct.

### `modules/control_prior.py` — historical control arms
Robust meta-analytic-predictive priors (Schmidli et al. 2014). Deterministic throughout: the
between-experiment variance τ is integrated by quadrature, the mean analytically, and the
conjugate normal-mixture updates are closed form — so operating characteristics with 8,000
replicates run in about 1.5 seconds and results are exactly reproducible.

From `validation/validate_control_prior.py`, with 12 historical control arms of 100 flies:

> A concurrent control of **25 flies instead of 100** keeps type I error at or below 10% across
> drift of ±5%, **saving 75 flies per experiment**. At zero drift, power rises from **39.7% to
> 64.6%**.

And the honest half of the same table:

> At +10% drift the type I error rises to **16.4%** for the robust prior and **39.3%** for a
> non-robust MAP prior, against 2.9% for concurrent-only.

That is the cost of borrowing, measured rather than argued about. The prior-data conflict check
detects a 15% drift 81% of the time and a 25% drift 99.8% of the time, while flagging only 3.0% of
genuinely exchangeable experiments. Under conflict the robust posterior collapses onto the
concurrent control (0.474 vs 0.470 concurrent-only, against a historical prior of 0.768) — it
switches itself off, which a non-robust prior does not.

The effective sample size was validated against its exact analytic limit: as τ→0 the prior must
carry exactly the information of the pooled historical flies, and it returns 726 against 720
(0.8% error).

### `modules/dose_response.py` — toxins
LL.4/LL.3/LL.2 log-logistic fitting with controls retained in the fit, profile-likelihood EC50
intervals, ECx, a Brain–Cousens hormesis test, relative potency with a parallelism test, and
mortality-adjusted behavioural toxicity.

Validated against **real published data**: the `ryegrass` ferulic-acid dataset from the `drc`
package, with the raw values read out of the package's own binary distribution and the reference
estimates cross-checked against a printed `summary(drm(..., LL.4()))`. Agreement to ≤9e-06 on every
parameter.

Two results worth noting. Profile-likelihood intervals are **4.5× asymmetric** for EC50 and 12.7×
for EC90, and in a shallow-slope case the delta method returns a **negative dose** as its lower
bound while the profile interval does not. And in a simulation where climbing failure and death are
driven by a *single* latent toxic burden — no separate behavioural mechanism whatsoever — the
survivor-only data still yield a clean-looking sigmoid with EC50 = 13.06 against LC50 = 15.99. The
module flags the ratio and refuses to call it a behavioural effect.

### `modules/circadian_climbing.py` — time of day
Exact linear-reparameterised cosinor with MESOR, amplitude and acrophase intervals by three
independent routes (delta method, parametric bootstrap, Bingham elliptical joint region);
population-mean cosinor treating each fly's (β, γ) as the unit of replication; and a
fatigue-versus-rhythm separator that **refuses to attribute an effect to circadian phase** when
testing order is collinear with time of day.

It also surfaced a real interpretive trap: between-fly phase scatter damps the population amplitude
by exactly exp(−(ωσ_φ)²/2), so a disease genotype with a lower population amplitude may have
perfectly intact individual rhythms that are merely **desynchronised from each other**. The module
reports a synchronisation index alongside amplitude so the two cannot be confused.

### `modules/design_schema.py` — the design layer
Recognises FlyID, VialID, ExperimentID, Batch, TimeOfDay/ZT, SessionIndex, TrialIndex, Compound,
Dose, DoseUnit, Date, Temperature and RackPosition; infers whether the design is longitudinal,
repeated-measures or nested; identifies the correct unit of replication; and runs confounding
checks between every design factor and every nuisance variable. Complete confounding is reported as
**blocking** — no analysis can separate those effects, and the fix is in the design, not the
statistics.

It also fixed a latent data-loss bug: the parser's row-uniqueness key was
(Genotype, Sex, Treatment, Replicate, Time, Age). Adding time-of-day or fly identity without
extending that key would have **silently averaged** the same vial's morning and evening
measurements into one row.

---

## 3. What the lab must record — the data specification

This is the most important section, because the analyses above are only as good as the columns they
receive, and several of them cannot run at all without information that is easy to record at the
bench and impossible to reconstruct afterwards.

### Required for every climbing row

| Column | Why it is needed |
|---|---|
| `Genotype`, `Treatment`, `Sex` | grouping |
| `VialID` | **the experimental unit.** Without it, every p-value is anticonservative and the ICC cannot be estimated. A unique string per physical vial, stable across the whole experiment. |
| `Count_climbed`, `n_total_flies` | the binomial numerator and denominator. Record the actual number in the vial *at that timepoint*, not the number originally set up. |
| `Time` | seconds after the tap |

### Required for trajectory work

| Column | Why it is needed |
|---|---|
| `Age` | days post-eclosion. The decline axis. |
| `FlyID` | only if individual flies are tracked. Enables true repeated measures. |
| **`DeathAge`** | **the single highest-value extra column in this entire specification.** Without it, informative-dropout testing, pattern-mixture correction and time-to-death alignment are all unavailable, and the survivor-bias magnitude cannot be estimated — only assumed to be zero. Record the day each vial's flies die, even approximately. |

### Required for the time-of-day design

| Column | Why it is needed |
|---|---|
| `TimeOfDay` | **record the actual clock time** (`09:15`), not the category "Morning". Categories are mapped to conventional Zeitgeber midpoints, and the acrophase then inherits the width of the whole testing window. |
| `SessionIndex` | how many times this vial has been tested before. Converts the fatigue/habituation assumption into a testable covariate. |
| `TrialIndex` | trial number within the session, if multiple taps are scored. |
| lights-on time | not a column — a single facility setting, but a wrong value rotates every phase estimate. |

### Required for toxins

| Column | Why it is needed |
|---|---|
| `Compound`, `Dose`, `DoseUnit` | numeric dose, not a label. `Dose = 0` for vehicle controls — keep them, they inform the upper asymptote. |
| mortality at each dose (`n_dead`) | behavioural toxicity among survivors is uninterpretable without it. **State whether `n_total_flies` counts flies scored (survivors) or flies exposed at the start** — the numbers cannot distinguish the two, and getting it wrong changes every mortality rate and hence the LC50. The shipped template counts survivors. |

### Required for historical control borrowing

| Column | Why it is needed |
|---|---|
| `ExperimentID` | which experiment a row belongs to. This is what makes between-experiment heterogeneity estimable, and it is what turns a pile of past controls into a usable prior. |
| `Batch` | food batch / incubator, if it varies. Drift is the mechanism that makes borrowing dangerous. |
| `Date` | for detecting seasonal drift. |

Templates for all of these are in `templates/`.

### Design requirements that no software can fix afterwards

1. **At least two vials per treatment arm.** With one vial per arm the treatment and vial effects
   are a single parameter. This is not a precision issue; it is an identifiability wall, and the
   software will refuse.
2. **Counterbalance testing order against time of day.** If genotype A is always tested in the
   morning and genotype B in the afternoon, the two effects are completely confounded and no
   analysis separates them. Rotate the order across days and across genotypes.
3. **Include never-before-tested flies at each age**, if repeated testing is used. This is the only
   way to identify a habituation component, should one exist.
4. **Keep a concurrent control even when borrowing historical ones.** It is what detects drift, and
   without it the prior-data conflict check — the entire safety mechanism — cannot run.
5. **Record `n` per vial at each timepoint**, not once at setup. Density affects climbing, and
   attrition changes density differentially between groups.

---

## 4. A published result that argues against part of the design

Gargano JW, Martin I, Bhandari P, Grotewiel MS (2005) *Rapid iterative negative geotaxis (RING): a
new method for assessing age-related locomotor decline in Drosophila*, Exp Gerontol 40:386-395 —
the paper that introduced the RING assay — states in its abstract:

> "We show that performance in RING assays is not influenced by the density of flies being tested,
> the time of day, or repeated testing."

That is a published null for the time-of-day effect the morning/afternoon/evening design sets out
to measure, and for two of the confounds that design would introduce. It must be confronted rather
than worked around. Three consequences:

- The null was established in **wild-type** flies, in one laboratory, over one age range. Whether
  it holds in a neurodegeneration model with compromised motor circuits, at advanced age, or under
  toxin exposure, is genuinely open. That is a legitimate hypothesis and worth testing.
- Because a prior null exists, the experiment must be powered so that a *negative* result is
  interpretable. Replicating Gargano is a perfectly good outcome; claiming a time-of-day effect is
  a claim against an existing published null and needs correspondingly strong evidence.
- The circadian module's `design_advisor` should be run **before** the experiment. On a
  morning/afternoon/evening schedule (approximately ZT2/ZT8/ZT11) it reports a 15-hour unsampled
  gap and phase isotropy of 0.10 — meaning sensitivity to a rhythm varies roughly ten-fold
  depending on when that rhythm happens to peak. A fourth, later timepoint would buy more than more
  flies would.

This entry exists partly as a record: an earlier draft of `docs/INVERSION_ANALYSIS.md` cited this
paper as evidence *for* fatigue and habituation effects — the reverse of what it says. The error was
caught by fetching the primary abstract instead of working from memory.

---

## 5. Publication strategy — an honest assessment

**Nature Methods will desk-reject a web application.** That journal publishes methods that are new,
demonstrably better than the alternatives on real data, and usually accompanied by a biological
finding the method made possible. A platform that applies standard statistics to standard assays —
however carefully, and however well benchmarked — is a *Bioinformatics* Applications Note or a *NAR*
Web Server paper. The existing manuscript is already correctly aimed at NAR Web Server, and that
remains the right target for the platform as a platform.

What would change the calculus is a **methodological result that is true independently of the
software**. Three are now in hand, and each is a finding rather than a feature:

1. **Survivor bias systematically flattens Drosophila motor-decline curves, and the bias is larger
   in shorter-lived genotypes.** Two genotypes with identical true decline appeared 4.5× different
   under cohort-mean analysis. The fly ageing literature routinely fits cohort means against
   chronological age; if that bias is as large in real data as it is in simulation, it affects a
   substantial body of published work. This is checkable, and the lab's own old-versus-young,
   control-versus-mutant cohorts are exactly the data that would check it.
2. **Historical control arms can cut control-fly use by 75% at controlled type I error** — with the
   operating characteristics simulated across realistic drift, and the drift level at which the
   method becomes unsafe stated explicitly. This is an experimental-design result with a direct 3R
   benefit, and the "flies are not wasted" motivation is exactly the right framing.
3. **Terminal-decline alignment.** If time-to-death predicts motor state better than chronological
   age — as Gaitanidis et al. (2019) implies, with impairment concentrated in the last 4–6 days of
   life — then age-matched comparison of genotypes with different lifespans is confounded as a
   matter of course.

Any one of these, established on **real data**, with the software as the vehicle rather than the
subject, is a considerably stronger paper than a feature list. Result 1 is the strongest candidate:
it is a correction to standard practice, it is quantifiable, and it changes conclusions.

**What must not be claimed.** Joint longitudinal-survival modelling, MAP priors, cosinor
rhythmometry, four-parameter log-logistic dose-response and binomial GLMMs are all established
methodology with decades of literature. The contribution here is their correct application to a
field that does not currently use them, and the demonstration of how much difference it makes.
Claiming to have invented any of them would be fatal on discovery, and is unnecessary — the applied
result is the contribution.

---

## 6. What is still open

| Item | Status |
|---|---|
| **Real laboratory data** | **BLOCKING.** No lab-generated raw data files were found anywhere on this machine. Every result above is simulation or published-benchmark. The ingestion and analysis path is built and tested so it runs the moment data arrives, but no biological claim can be made until it does. |
| R `lme4`/`nlme` cross-validation | Blocked: the network proxy performs TLS interception and both `pip` and `conda` installs fail with certificate errors. The hierarchical module is instead validated against an exact analytic identity, scipy, and statsmodels. |
| Multiplicity across the whole analysis surface | Open (`INVERSION_ANALYSIS.md` B3). BH-FDR is applied within families; there is no session-level registry of every test performed. |
| Dependent effect sizes in drug ranking | Open (B4). Arms sharing a vehicle control have correlated Hedges' *g*; the pooled standard error is currently understated. |
| Cox models without vial clustering | Open (B5). `modules/survival.py:461` fits without `groups=` or a frailty term. |
| Manuscript update | The manuscript still describes the pre-uplift platform and does not disclose the hierarchical, survivor-bias or dose-response limitations that have now been addressed. It needs rewriting around the new results. |
| Reference verification | Open (E4). Every reference must be checked against the primary record before submission — see §4 for why this is not a formality. |
