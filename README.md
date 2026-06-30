# DroPhenix Analytics v3.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drophenix.streamlit.app)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

> **A web-based, statistically rigorous platform for multi-assay Drosophila melanogaster phenotyping and cross-strain compound ranking**

**New in v3.0:** five additional assays beyond climbing — **lifespan/survival** (Kaplan–Meier, log-rank, Cox), **Y-maze** (alternation + handedness), **activity/sleep (DAMS)** (5-min sleep rule + chi-square circadian periodogram), **larval crawling**, and **grooming** — plus a **cross-strain drug-ranking** engine (Hedges' *g* + inverse-variance / DerSimonian–Laird meta-analysis handling unequal sample sizes), a runnable validation suite benchmarked against published data, a head-to-head comparison vs R `survival`/lifelines/statsmodels/scipy/pingouin/astropy/pyBOAT (24/24 metrics agree), per-assay data templates, and the NAR manuscript.

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
- **Larval Crawling** — Video centroid tracking → speed (mm/s), path, tortuosity
- **Grooming** — %-time grooming, bout count/duration, group statistics
- **Novel Metrics** — CRI, SDQ, TSW, and MDR (decline **over age**) with 300 DPI export
- **Statistical Engine** — Normality/variance-gated ANOVA / Kruskal-Wallis, BH-FDR, effect sizes, bootstrap CI
- **Clustering** — DTW-based trajectory clustering, dendrogram, PCA/UMAP
- **Drug Ranking (cross-strain)** — Hedges' *g* + inverse-variance / DerSimonian–Laird meta-analysis across **unequal-N strains** → Therapeutic Score
- **Translational Scoring** — Tier I–V compound ranking (TCS)
- **Video / Live Recording** — Webcam / MP4 with automated fly counting (OpenCV)
- **Validation & Benchmarking** — `validation/run_all.py` (12 suites vs published data) and `comparison_study.py` (vs 7 established tools)
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
│   ├── trajectory_cluster.py # DTW clustering, PCA/UMAP
│   ├── translational_score.py # TCS compound ranking
│   ├── visualizer.py       # Plotly figure factory
│   ├── recording.py        # Webcam / video capture
│   └── video_analyzer.py   # OpenCV fly detection
├── protocols/
│   └── protocol_guide.py   # Vial setup & data template guide
├── assets/
│   └── Drophenix_Logo.png
├── requirements.txt
├── packages.txt
└── validate_drophenix_v2.py  # Independent validation suite
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

Downloadable per-assay data templates are in [`templates/`](templates/) (see
`templates/README.md`) and from the ⬇ button on each module's page.

### Benchmark vs established tools

`validation/comparison_study.py` runs DroPhenix head-to-head against installed,
peer-reviewed reference tools on identical benchmark data (report:
`validation/comparison_report.md`). **15/15 metrics agree within tolerance.**

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