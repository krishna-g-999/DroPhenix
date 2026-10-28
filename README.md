# DroPhenix Analytics v3.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drophenix.streamlit.app)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

> **A web-based, statistically rigorous platform for multi-assay Drosophila melanogaster phenotyping and cross-strain compound ranking**

**New in v3.0:** five additional assays beyond climbing — **lifespan/survival** (Kaplan–Meier, log-rank, Cox), **Y-maze** (alternation + handedness), **activity/sleep (DAMS)** (5-min sleep rule + chi-square circadian periodogram), **larval crawling**, and **grooming** — plus a **cross-strain drug-ranking** engine (Hedges' *g* + inverse-variance / DerSimonian–Laird meta-analysis handling unequal sample sizes), a runnable validation suite benchmarked against published data, a head-to-head comparison vs R `survival`/lifelines/statsmodels/scipy/pingouin/astropy/pyBOAT (24 of 26 comparisons agree, 0 disagree, 2 documented method differences), per-assay data templates, and the NAR manuscript.

**New in v3.1 — the statistical layer underneath.** Climbing counts are now modelled as what they are: **binomial proportions clustered in vials**, with the vial as the experimental unit. Treating flies as independent replicates inflates the false-positive rate 5.2-fold under a true null, and the platform now measures that inflation on your own data rather than committing it. Alongside this: **survivor-bias correction** for ageing trajectories (the conventional cohort-mean curve understates decline by 35% when frail flies die first, and can make two genotypes with identical decline look 4.5× apart), **calibrated trajectory forecasting** with empirical-Bayes population priors, **robust historical-control priors** that cut control-fly use by up to 75% at controlled type I error, **toxin dose-response** with profile-likelihood EC50 intervals and mortality-confounding diagnostics, and **time-of-day cosinor analysis** of climbing. See [`docs/UPLIFT_PLAN.md`](docs/UPLIFT_PLAN.md).

Developed at the **Department of Bioinformatics, SSSIHL, Puttaparthi, India**  
PI: Prof. Venketesh Sivaramakrishnan · Scholar: Krishnasalini Gunanathan

---

## Live Server

🔗 **https://drophenix.streamlit.app**  
No installation required · No login · Works in any browser

---

## Statement of Need

Drosophila climbing (negative geotaxis) assays are the primary motor readout in fly models of ALS, Parkinson's, and Huntington's disease. Existing analysis workflows rely on endpoint Performance Index (PI) calculated manually in spreadsheets — a method that:

- Discards longitudinal trajectory information
- Cannot quantify sex-differential drug responses
- Provides no measure of WT-selective therapeutic safety
- Offers no standardised kinetic disease-progression metric

DroPhenix addresses all four gaps through a suite of purpose-built composite metrics.

---

## Novel Metrics

| Metric | Formula | Biological meaning |
|--------|---------|-------------------|
| **CRI** — Composite Rescue Index | `CRI = 0.4·f(PI) + 0.4·f(AUC) + 0.2·f(t50)` × 100 | Integrates endpoint, kinetic, and latency rescue into one score (0–100%) |
| **SDQ** — Sex Dimorphism Quotient | `SDQ = \|PI_M − PI_F\| / max(\|PI_M\|, \|PI_F\|)` | Quantifies sex-differential drug response; critical for translational validity |
| **TSW** — Therapeutic Selectivity Window | `TSW = ΔPI_disease / \|ΔPI_WT\|` | Measures disease-selective rescue; TSW > 1.0 = safe window |
| **MDR** — Motor Decline Rate | `PI(age) = PI₀ · e^{−k·age}` | Exponential decline of climbing PI **over age (days)** — negative geotaxis is age-dependent (Gargano et al. 2005). k = decline rate; Δk = treatment rescue. Requires an `Age` column; the per-second assay timecourse is used for climbing curves/AUC/t₅₀, not for decline. |

---

## Features

- **Data Input** — Upload CSV/Excel or load built-in demo datasets; per-assay templates in `templates/`
- **Climbing** — Longitudinal curves (SEM ribbons), Performance Index, AUC, t50
- **Survival Analysis** — Kaplan–Meier (Greenwood CIs), median survival, log-rank (BH-FDR pairwise), Cox proportional hazards
- **Y-maze** — Spontaneous alternation (vs chance), turn-bias/handedness with Monte-Carlo individuality test, exploration
- **Activity Monitoring (DAMS)** — Sleep (≥5-min rule: total/day/night, bouts) and circadian period (chi-square periodogram + Lomb–Scargle)
- **DAM Run Analysis** — raw TriKinetics monitor files + a channel map → a complete, design-aware analysis without hand-holding: the recording is checked day by day and an analysis window proposed (failed lamp days, flicker, gaps and partial days excluded for a stated reason), the design is read from the channel map, contrasts are labelled monitor-clean or confounded, the published DAM parameter set is tested sex-stratified with FDR within each family, every family is rerun without the flagged outliers, a second timepoint can be compared, and a Word report is written and verified cell by cell. Validated against two complete recordings, whose hand analysis it reproduces to floating-point precision ([`docs/DAM_RUN_ANALYSIS.md`](docs/DAM_RUN_ANALYSIS.md)); also `drophenix_cli.py dam-run`
- **Larval Crawling** — Video centroid tracking → speed (mm/s), path, tortuosity
- **Grooming** — %-time grooming, bout count/duration, group statistics
- **Novel Metrics** — CRI, SDQ, TSW, and MDR (decline **over age**) with 300 DPI export
- **Statistical Engine** — Normality/variance-gated ANOVA / Kruskal-Wallis, BH-FDR, effect sizes, bootstrap CI
- **Clustering** — DTW-based trajectory clustering, dendrogram, PCA/UMAP
- **Drug Ranking (cross-strain)** — Hedges' *g* + inverse-variance / DerSimonian–Laird meta-analysis across **unequal-N strains** → Therapeutic Score
- **Translational Scoring** — Tier I–V compound ranking (TCS)
- **Video / Live Recording** — Webcam / MP4 with automated fly counting (OpenCV)
- **Local command line** (`drophenix_cli.py`) — score video on the lab machine, without a browser. Recordings are ~120 MB each and a batch takes ~45 minutes, so uploading them to a hosted app is slow, often blocked, and sends unpublished data off-site for no benefit. Do the heavy part locally, get a kilobyte CSV, analyse it anywhere: `manifest` → `inspect` → `batch` → `check`
- **Video scoring** — score a climbing assay directly from a phone recording: automatic cylinder and scoring-line detection, sub-pixel frame registration, per-vial counts across the whole climbing time-course, and three independent quality gates that flag an unreliable result rather than reporting it silently ([`docs/VIDEO_SCORING.md`](docs/VIDEO_SCORING.md))
- **Hierarchical statistics** — binomial/beta-binomial modelling of climbing counts with the vial as the experimental unit, cluster-robust GEE, intraclass correlation, design effect, and a diagnostic that shows the naive and cluster-correct p-values side by side
- **Survivor-bias correction** — four estimators of the same ageing trajectory (survivor mean, balanced panel, mixed model, pattern mixture), an informative-dropout test, and a time-to-death axis
- **Trajectory forecasting** — empirical-Bayes population priors from completed cohorts, forward-in-time validation against four trivial baselines, split-conformal interval calibration, and decision-relevant endpoints (onset age, healthspan, days gained)
- **Historical control arms** — robust meta-analytic-predictive priors with prior-data conflict detection, effective sample size in flies, and simulated operating characteristics
- **Dose-response** — log-logistic fitting with profile-likelihood EC50 intervals, hormesis testing, and mortality-confounding diagnostics
- **Validation & Benchmarking** — `validation/run_all.py` (41 suites, 1,067 checks vs published data and analytic ground truth) and `comparison_study.py` (vs 7 established tools)
- **Export** — CSV, PNG, SVG, PDF, ZIP bundle

---

## Installation (local)

```bash
# Clone repository
git clone https://github.com/krishna-g-999/drophenix.git
cd drophenix

# Create environment (conda recommended)
conda create -n drophenix python=3.11
conda activate drophenix

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

---

## Repository Structure

```
drophenix/
├── app.py                  # Main Streamlit application
├── modules/
│   ├── data_parser.py      # Input validation and parsing
│   ├── normalizer.py       # PI, AUC, t50 computation
│   ├── metrics.py          # CRI, SDQ, TSW, ECS
│   ├── motor_decline.py    # MDR exponential decay, MSCI
│   ├── stats_engine.py     # Statistical tests, bootstrap CI
│   ├── climbing_video.py   # Score a climbing assay from video (see docs/)
│   ├── design_schema.py    # Experimental-design detection, confounding checks
│   ├── hierarchical.py     # Beta-binomial / GEE / mixed models, ICC, design effect
│   ├── survivor_bias.py    # Selection-corrected ageing trajectories
│   ├── trajectory.py       # Empirical-Bayes forecasting + conformal calibration
│   ├── control_prior.py    # Robust MAP priors for historical control arms
│   ├── dose_response.py    # Log-logistic dose-response, EC50/LC50, hormesis
│   ├── circadian_climbing.py # Time-of-day cosinor analysis of climbing
│   ├── survival.py         # Kaplan-Meier, log-rank, Cox PH
│   ├── trajectory_cluster.py # DTW clustering, PCA/UMAP
│   ├── translational_score.py # TCS compound ranking
│   ├── visualizer.py       # Plotly figure factory
│   ├── recording.py        # Webcam / video capture
│   └── video_analyzer.py   # OpenCV fly detection
├── docs/
│   ├── INVERSION_ANALYSIS.md # Pre-mortem: every way the platform can fail
│   ├── UPLIFT_PLAN.md      # Findings, results, lab data specification
│   └── VIDEO_SCORING.md    # Video pipeline: design, evidence, recording guidance
├── protocols/
│   └── protocol_guide.py   # Vial setup & data template guide
├── assets/
│   └── Drophenix_Logo.png
├── requirements.txt
├── packages.txt
└── validation/
    ├── run_all.py          # Master runner — 41 suites, exit 0 = all passed
    └── comparison_study.py # Head-to-head vs 7 established tools
```

---

## Validation

A runnable validation suite under `validation/` benchmarks each analysis module
against **published raw data** (reproducing the papers' reported statistics) and
hand-checked analytics. Run the whole suite with:

```bash
python validation/run_all.py        # exit 0 = every suite passed
```

| Suite | Benchmark (published source / ground truth) | Key reproduced result |
|-------|------------------------------|-----------------------|
| `validate_stats` | PlantGrowth one-way ANOVA (Dobson 1983 / R) | F(2,27)=4.846, p=0.01591; Kruskal H=7.99 |
| `validate_survival` | Freireich 1963 6-MP leukemia trial | log-rank χ²=16.79; Cox HR=4.52; medians 23/8 wk |
| `validate_ymaze` | de Bivort 2015; Kim 2017 | turn-bias mean ≈0.50, 23.5% extreme; 50% alternation chance |
| `validate_dams` | Konopka & Benzer 1971; Sokolove–Bushell 1978 | periods 24/19/28 h recovered; exact 5-min sleep rule |
| `validate_video` | synthetic ground-truth climbing clips | exact fly counts and PI recovered (8/4→33.3; 3/9→−50) |
| `validate_larval` | known-velocity synthetic clip; lit. speed | exact speed recovered; WT ~1 mm/s |
| `validate_grooming` | Qiao et al. 2018 (eLife) | exact bout maths; WT ~13% time grooming |
| `validate_metrics` | analytic ground truth (5 sample sets) | MDR k-recovery Pearson r=0.989, median R²=0.99; SDQ/TSW exact; CRI ordinal |
| `validate_trajectory` | known shape families (4 seeds) | DTW clustering purity = 1.0 (2 clusters) |
| `validate_drug_ranking` | statsmodels meta-analysis | inverse-variance/DL pooling exact; ranking A>B>C |
| `validate_synthetic` | unified cohort coherence | disease-selective drug effect on all four assays |
| `validate_hierarchical` | exact one-way random-effects ICC identity; statsmodels GLM(Binomial) | ICC matches the closed form to 1e-10; naive fly-level test rejects **25.9% under a true null** (5.2× inflation) while the clustered tests hold 4–6% |
| `validate_survivor_bias` | generative model with a known true slope | cohort mean understates decline by **35%**; mixed model within 2%; two genotypes with identical decline appear 4.5× apart under cohort-mean analysis |
| `validate_trajectory_prediction` | known curve parameters; analytic CRPS | population prior cuts long-horizon MAE **37%**; split-conformal calibration restores 95% coverage from 0.70 to **0.956** |
| `validate_control_prior` | numerical integration; exact τ→0 ESS limit | closed forms match integration to 1e-8; ESS 726 vs 720 analytic; **75 control flies saved per experiment** at type I ≤ 10% |
| `validate_dose_response` | **drc `ryegrass` ferulic acid data** (real published raw data) | every LL.4 parameter matches the published fit to ≤9×10⁻⁶ |
| `validate_circadian_climbing` | exact linear cosinor identity; astropy Lomb–Scargle | noise-free recovery to 1e-10; refuses to attribute an effect to circadian phase when order is collinear |
| `validate_climbing_video` | synthetic footage with per-frame ground truth; registration ablation on **real laboratory recordings** | exact counts and 0 px scoring-line error on synthetic; on real video, sub-pixel registration cuts count variation from **89% to 15%** |

**What these suites do and do not establish.** They are of three kinds, and they
are not equivalent evidence. *Analytic* checks (ICC, CRPS, cosinor, prior ESS,
curve recovery) establish that the implementation is correct. *Published-data*
checks (Freireich 1963, PlantGrowth, drc ryegrass) establish agreement with the
literature. *Simulation* checks establish that an estimator works under its own
assumptions — necessary, but unable to detect a model mis-specified for real
flies. No amount of simulation substitutes for laboratory data. See
`docs/INVERSION_ANALYSIS.md`.

Downloadable per-assay data templates are in [`templates/`](templates/) (see
`templates/README.md`) and from the ⬇ button on each module's page.

### Design-aware analysis

Five columns unlock the hierarchical, trajectory, circadian, dose-response and
control-borrowing analyses, and none can be reconstructed after the experiment:
`VialID` (the experimental unit — flies in a vial are not independent),
`DeathAge` (without it, survivor bias can only be assumed to be zero),
`TimeOfDay` (actual clock time, not "Morning"), `SessionIndex` (testing order),
and `ExperimentID` (what makes past control arms borrowable). Full specification
in [`docs/UPLIFT_PLAN.md`](docs/UPLIFT_PLAN.md) §3.

### Benchmark vs established tools

`validation/comparison_study.py` runs DroPhenix head-to-head against installed,
peer-reviewed reference tools on identical benchmark data (report:
`validation/comparison_report.md`). **24 of 26 comparisons agree within
tolerance; 0 disagree; 2 are documented method differences or comparators that
could not be installed here.** Every row is counted in the denominator and both
non-agreeing rows are named in the report — a benchmark whose denominator cannot
express a disagreement is not a benchmark.

| Domain | Reference tool | Result |
|--------|----------------|--------|
| Survival (KM median, log-rank χ²/p) | **lifelines** | exact agreement |
| Survival (Cox HR) | lifelines / Klein-Moeschberger 2003 | DroPhenix Breslow ties = published 4.523 (lifelines Efron = 4.817; expected ties-method difference) |
| ANOVA / Kruskal / Hedges g | **scipy**, **pingouin** | exact agreement |
| Meta-analysis pooling (I², DL) | **statsmodels** | exact agreement |
| Circadian period | **astropy** Lomb-Scargle | 24.00 vs 23.96 h |

---

## Citation

If you use DroPhenix in your research, please cite:

> Gunanathan K, Sivaramakrishnan V (2026).  
> **DroPhenix: A web-based analytics platform for quantitative Drosophila motor phenotyping.**  
> *Nucleic Acids Research*, Web Server Issue.  
> DOI: [pending]

---

## License

DroPhenix is released under the **GNU General Public License v3.0**.  
See [LICENSE](LICENSE) for details.

---

## Contact

- **Krishnasalini Gunanathan** — PhD Scholar, Dept. of Bioinformatics, SSSIHL  
- **Prof. Venketesh Sivaramakrishnan** — Principal Investigator  
- Sri Sathya Sai Institute of Higher Learning, Puttaparthi 515134, Andhra Pradesh, India