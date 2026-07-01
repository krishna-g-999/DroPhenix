# Supplementary Data

**DroPhenix: an integrated, statistically rigorous web platform for multi-assay *Drosophila* phenotyping and actuarial cross-strain compound ranking**

Gunanathan K. and Sivaramakrishnan V.

> All values below are produced by the runnable scripts shipped with the software and are exactly reproducible: `validation/run_all.py` (per-module validation), `validation/comparison_study.py` (cross-tool benchmark; writes `validation/comparison_report.md`), and `validation/demo_unequal_n.py` (unequal-N demonstration).

---

## Supplementary Table S1. Complete head-to-head comparison with established tools

Every quantitative comparison performed, on identical benchmark data (Freireich 1963 6-MP trial; PlantGrowth; a known 24-h rhythm; a fixed effect-size set). "Agree" = within the stated tolerance. Two method-note rows (marked *note*) are reported for completeness and are not counted as discrepancies.

| Domain | Metric | DroPhenix | Reference tool | Reference | Agree |
|---|---|---|---|---|---|
| Survival | Median 6-MP (weeks) | 23.0 | lifelines | 23.0 | yes |
| Survival | Median Placebo (weeks) | 8.0 | lifelines | 8.0 | yes |
| Survival | Log-rank χ² | 16.7929 | lifelines | 16.79294 | yes |
| Survival | Log-rank p-value | 4.2×10⁻⁵ | lifelines | 4.2×10⁻⁵ | yes |
| Survival | Cox HR Breslow (vs published 4.523) | 4.5231 | Klein–Moeschberger 2003 | 4.523 | yes |
| Survival | Cox HR Efron (tie-method *note*) | 4.5231 | lifelines (Efron) | 4.81687 | *note* |
| Statistics | One-way ANOVA F | 4.8461 | scipy | 4.84609 | yes |
| Statistics | ANOVA p-value | 0.01591 | scipy | 0.01591 | yes |
| Statistics | Kruskal–Wallis H | 7.9882 | scipy | 7.98823 | yes |
| Statistics | One-way ANOVA F | 4.8461 | pingouin | 4.84609 | yes |
| Statistics | Hedges' g (trt2 vs ctrl) | 0.91404 | pingouin | 0.91404 | yes |
| Meta-analysis | Pooled effect (fixed) | 0.41034 | statsmodels | 0.41034 | yes |
| Meta-analysis | Pooled effect (DL random) | 0.42883 | statsmodels | 0.42883 | yes |
| Meta-analysis | Heterogeneity I² (%) | 27.72586 | statsmodels | 27.72586 | yes |
| Circadian | Dominant period (h) [true 24] | 24.0 | ground truth | 24.0 | yes |
| Circadian | Dominant period (h) | 24.0 | astropy (Lomb–Scargle) | 23.96404 | yes |
| R: survival | Median 6-MP | 23.0 | R `survfit` | 23.0 | yes |
| R: survival | Median Placebo | 8.0 | R `survfit` | 8.0 | yes |
| R: survival | Log-rank χ² | 16.7929 | R `survdiff` | 16.79294 | yes |
| R: survival | Log-rank p-value | 4.2×10⁻⁵ | R `survdiff` | 4.2×10⁻⁵ | yes |
| R: survival | Cox HR (Breslow) | 4.5231 | R `coxph(breslow)` | 4.52307 | yes |
| R: stats | ANOVA F | 4.8461 | R `aov` | 4.84609 | yes |
| R: stats | ANOVA p-value | 0.01591 | R `aov` | 0.01591 | yes |
| R: stats | Kruskal–Wallis H | 7.9882 | R `kruskal.test` | 7.98823 | yes |
| Circadian | Dominant period (h) | 24.0 | pyBOAT (wavelet) | 23.9598 | yes |
| Survival | scikit-survival (*note*) | — | scikit-survival | not installable (no C toolchain / no Python 3.13 wheel) | *note* |

**24/24 quantitative comparisons agree within tolerance (Pearson r = 1.0000 across DroPhenix vs reference values).** The two *note* rows are the Efron tie-method value and the un-buildable scikit-survival package; survival is independently cross-validated against R `survival` (Breslow) and lifelines (Efron).

---

## Supplementary Table S2. Per-module validation against published benchmarks / analytic ground truth

Each row is an automated test in `validation/` (all pass).

| Module / script | Benchmark (source) | Reproduced result |
|---|---|---|
| `validate_stats` | PlantGrowth ANOVA/Kruskal (Dobson 1983) | F=4.846, p=0.01591; H=7.988; Cohen's d, Hedges' g, BH hand-checked |
| `validate_survival` | Freireich 1963 6-MP leukemia trial | median 23/8 wk; log-rank χ²=16.793; Gehan–Breslow 13.458; Cox HR 4.523 |
| `validate_ymaze` | Buchanan/de Bivort 2015; Lewis 2017 | turn-bias mean 0.50; 23.5% extreme; alternation chance 50%; individuality sig vs null |
| `validate_dams` | Konopka & Benzer 1971; Sokolove–Bushell 1978; Hendricks/Shaw 2000 | periods 24/19/28 h; 100% sensitivity, 0% false-positive; exact 4↔5-min sleep boundary |
| `validate_video` | synthetic ground-truth climbing clips | exact fly counts and PI (8/4→33.3; 3/9→−50) |
| `validate_larval` | known-velocity clip; Gargano 2005 | exact recovered speed; WT ≈1 mm s⁻¹ |
| `validate_grooming` | Qiao 2018 | exact bout arithmetic; WT ≈13% time grooming |
| `validate_metrics` | analytic ground truth (5 sample sets) | MDR k-recovery Pearson r=0.989, median R²=0.99; SDQ, TSW exact; CRI ordinal |
| `validate_trajectory` | known shape families (4 seeds) | DTW clustering purity = 1.0 (2 clusters) |
| `validate_drug_ranking` | statsmodels meta-analysis | inverse-variance/DL pooling exact; ranking A>B>C; unequal-N flagged |
| `validate_synthetic` | unified cohort coherence | disease-selective drug effect on all four assays |
| `demo_unequal_n` | Monte-Carlo simulation | see Supplementary Note S1 |

---

## Supplementary Note S1. Unequal-N cross-strain compound comparison

**Worked example** — one compound vs vehicle in four disease strains of unequal size. Per-strain effect is Hedges' *g*; precision weight wᵢ = 1/Var(gᵢ).

| Strain | n (per arm) | Hedges' g | Var(g) | Weight (%) |
|---|---|---|---|---|
| TDP43 | 3 | 0.394 | 0.446 | 19.6 |
| SOD1 | 4 | 0.074 | 0.379 | 23.1 |
| FUS | 5 | 0.371 | 0.335 | 26.1 |
| HTT-Q93 | 7 | 0.840 | 0.280 | 31.2 |

Naïve equal-weight mean of *g* = 0.420; inverse-variance pooled (fixed) = 0.453; DerSimonian–Laird random-effects pooled = 0.453 (Q = 0.943, τ² = 0.000, I² = 0.0%). The larger strain carries proportionally more weight; the small n = 3 strain is down-weighted rather than dominating or being discarded.

**Estimator efficiency (simulation)** — 3000 simulations, true g = 0.7, strain sizes {3, 7, 5, 4, 9}:

| Estimator | RMSE | Sampling variance | 95% CI coverage |
|---|---|---|---|
| Naïve equal-weight mean | 0.334 | 0.111 | — |
| Inverse-variance pooled | 0.267 | 0.066 | 95.8% |

Inverse-variance pooling reduces RMSE by **19.9%** and is well-calibrated (coverage ≈ nominal), confirming it is the efficient, correct estimator under unequal N (Figure 2).

---

## Supplementary Note S2. Literature-referenced reference datasets (≥3 per assay)

To exercise every module on data tied to the literature, the repository ships
`reference_datasets/` with three datasets per assay, in each module's upload-template
format, generated and analysed by `reference_datasets/build_and_run.py` (provenance in
`reference_datasets/PROVENANCE.md`, results in `reference_datasets/RESULTS.md`). Survival
sample 1 is the **real** Freireich et al. (1963) raw 6-MP data; circadian sample 1 uses the
**real** Konopka & Benzer (1971) free-running periods (24/19/28 h). The remaining datasets
are simulated from text-verified published parameters (alternation chance 50%, Lewis 2017;
turn-bias mean 0.50 and 23.5% extreme, Buchanan/de Bivort 2015; wild-type lifespan median
≈50 d, Linford 2013; larval speed ≈1 mm s⁻¹; grooming ≈13% of time, Qiao 2018), each cited
in PROVENANCE.md and clearly labelled as simulated (machine-readable raw supplements were
not retrievable for those assays). **All 18 datasets reproduce the cited published
expectation** when run through DroPhenix.

---

## Supplementary Figures

Analysis results produced live by the DroPhenix modules on the literature-referenced reference datasets (Supplementary Note S2). Colour-blind-safe (Okabe-Ito) palette; single Arial font.

**Figure S1. Climbing (negative geotaxis).** Endpoint Performance Index (PI = 2·%climbed − 100) by group for three disease-model datasets (PD α-synuclein, ALS TDP-43, HD polyQ). In every dataset wild-type PI exceeds the disease-model vehicle and the drug partially rescues.

**Figure S2. Lifespan.** Kaplan–Meier survival curves; sample 1 is the real Freireich et al. (1963) 6-MP trial (median 23 vs 8 weeks), samples 2–3 are Gompertz fly cohorts (wild-type median ≈50 d; disease shorter; drug extends).

**Figure S3. Y-maze.** Mean spontaneous alternation (± SEM) per genotype against the 50% chance line (Lewis 2017); wild-type alternates above chance and above the memory-mutant cohorts.

**Figure S4. DAMS circadian period.** Chi-square (Sokolove–Bushell) periodograms for wild-type, per-short and per-long, recovering the Konopka & Benzer (1971) periods of 24, 19 and 28 h respectively; dashed line = Bonferroni-corrected significance threshold.

**Figure S5. Larval crawling.** Mean crawling speed (± SEM) by genotype for three disease contrasts; wild-type ≈1 mm s⁻¹, disease models slower, drug partially restored.

**Figure S6. Grooming.** Percentage of time grooming (± SEM) by genotype; wild-type ≈13% (Qiao 2018), reduced in the disease models and partially restored by drug.

---

## Supplementary Methods: formula summary

| Quantity | Formula |
|---|---|
| Performance Index | PI = 2·(climbed/total)·100 − 100 |
| Kaplan–Meier | Ŝ(t) = Π_{tᵢ≤t}(1 − dᵢ/nᵢ); Greenwood Var = Ŝ(t)²·Σ dᵢ/[nᵢ(nᵢ−dᵢ)] |
| Log-rank (Mantel–Cox) | χ² = (Σ(O₁ⱼ − E₁ⱼ))² / Σ V₁ⱼ; df = G−1 |
| Cox PH | h(t\|x)=h₀(t)·exp(βᵀx); HR = exp(β); Breslow ties |
| χ² periodogram | Q_P = N′·K·Σ_h(X̄_h−X̄)² / Σ(Xᵢ−X̄)² ~ χ²_{p−1}; Bonferroni across periods |
| Sleep | bout = ≥5 consecutive min of zero activity |
| Spontaneous alternation | %alt = 100·(alternation triads)/(entries−2); chance = 50% |
| Turn bias | b = R/(R+L); individuality: observed MAD vs binomial-null MAD |
| Motor Decline Rate | PI(age) = PI₀·e^(−k·age); Δk = k_vehicle − k_drug |
| Hedges' g | g = J·(x̄_t−x̄_c)/s_p, J = 1 − 3/[4(n₁+n₂)−9] |
| Var(g) | J²·[(n₁+n₂)/(n₁n₂) + g²/(2(n₁+n₂−2))] |
| Inverse-variance pooling | g_pooled = Σwᵢgᵢ/Σwᵢ, wᵢ = 1/Var(gᵢ), Var = 1/Σwᵢ |
| DerSimonian–Laird | τ² = max(0,(Q−(k−1))/C); wᵢ* = 1/(Var(gᵢ)+τ²); I² = max(0,(Q−(k−1))/Q) |
