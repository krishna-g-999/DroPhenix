# DroPhenix: an integrated, statistically rigorous web platform for multi-assay *Drosophila* phenotyping and actuarial cross-strain compound ranking

**Krishnasalini Gunanathan¹ and Venketesh Sivaramakrishnan¹***

¹ Department of Biosciences, Sri Sathya Sai Institute of Higher Learning, Puttaparthi 515134, Andhra Pradesh, India.
\* To whom correspondence should be addressed. Tel: [tel]; Email: [corresponding-author email].

**Keywords:** *Drosophila*; behavioural phenotyping; survival analysis; circadian rhythm; meta-analysis; drug screening; web server.

---

> *Prepared for the Nucleic Acids Research Web Server Issue. Every numerical result is produced by the runnable scripts shipped with the software (`validation/run_all.py`, `validation/comparison_study.py`, `validation/demo_unequal_n.py`, `manuscript/make_figures.py`) and is reproducible from the stated data. Reference pagination/DOIs to be confirmed against primaries at submission.*

---

## ABSTRACT

*Drosophila melanogaster* is a workhorse for neurodegeneration modelling and compound screening, but its behavioural assays are usually analysed in disconnected spreadsheets with inconsistent statistics, and there is no principled way to combine a compound's effect across genetic backgrounds tested with **unequal numbers of flies**. We present **DroPhenix**, a free web platform that analyses six assays—climbing (negative geotaxis), lifespan, Y-maze, activity/sleep (DAMS), larval crawling and grooming—with assumption-aware statistics (normality/variance-gated tests, effect sizes, Benjamini–Hochberg control, bootstrap intervals), actuarial survival analysis (Kaplan–Meier, log-rank, Cox), circadian periodograms, sleep architecture, and age-resolved motor-decline kinetics. Its central innovation is a meta-analytic ranking engine that combines per-strain drug effects (Hedges' *g*) across **unequal-N strains** by inverse-variance / DerSimonian–Laird random-effects pooling and across assays into one Therapeutic Score. We prove by simulation that this pooling is more efficient (19.9% lower RMSE) and correctly calibrated (95.8% CI coverage) than naïve averaging under unequal N. Every module reproduces published benchmark statistics exactly, and DroPhenix's outputs agree with **R `survival`, lifelines, statsmodels, scipy, pingouin, astropy and pyBOAT** on identical data (24/24 metrics; *r* = 1.0000). Available at https://drophenix.streamlit.app.

---

## INTRODUCTION

Behavioural and physiological assays in *Drosophila* underpin studies of ALS, Parkinson's and Huntington's diseases, ageing and circadian biology (1–4). Climbing (negative geotaxis) measures startle-induced upward locomotion within seconds of a tap and declines with age and neurodegeneration (1, 2); lifespan assays yield right-censored survival data (6); the Y-maze quantifies spatial working memory and locomotor handedness (7, 8); the DAM system records activity from which sleep (≥5 min inactivity) and circadian period are derived (9–11); larval crawling reports segmental motor output; and grooming quantifies a structured fixed-action pattern (12).

Their **analysis**, however, remains fragmented and frequently under-powered statistically. Climbing data are often reduced to a single endpoint percentage; lifespan studies sometimes report only median survival, omitting the Kaplan–Meier estimator, log-rank test and Cox model that are standard in survival analysis (13, 14); circadian period is estimated with heterogeneous, sometimes multiplicity-uncorrected, periodograms; and—most importantly—there is no widely used method to rank a candidate compound across multiple strains of **unequal sample size** across multiple assays. In screening this is acute: when one genotype contributes three replicate vials and another seven, equal-weight averaging of group means is biased toward the smaller, noisier strain. DroPhenix closes these gaps in one free, browser-based, verifiable tool.

---

## MATERIALS AND METHODS

### Implementation and availability
DroPhenix is implemented in Python 3 (Streamlit; numpy, pandas, scipy, scikit-learn, statsmodels, plotly/matplotlib, OpenCV). It requires no installation or login; source, the full validation suite and per-assay templates are open (GPL v3). Each assay page has a consistent layout: data source (upload or literature-calibrated demo), a downloadable column template, an input preview, tabbed results with publication figures, and CSV export.

### Notation
For two groups (treated *t*, control *c*) with sizes n₁, n₂, means x̄, and SDs s, the pooled SD is s_p = √[((n₁−1)s₁² + (n₂−1)s₂²)/(n₁+n₂−2)].

### Per-module formulas, validation parameters and scientific rationale
Each method below is stated with the quantity computed, the parameter used to validate it, and the published/analytic benchmark; Table 1 summarises and gives the reproduced value.

**(i) Climbing Performance Index.** Per timepoint, Pct = 100·(climbed/total) and **PI = 2·Pct − 100** (∈[−100, +100]); AUC and t₅₀ are the trapezoidal area and 50%-crossing time of the per-second curve. Uploaded climbing videos are additionally analysed by per-vial Otsu-thresholded fly detection above/below a counting line (cf. automated RING systems (5)). *Rationale:* PI is the standard negative-geotaxis readout (1, 2); the seconds axis captures climbing kinetics.

**(ii) Kaplan–Meier survival.** Ŝ(t) = Π_{tᵢ≤t} (1 − dᵢ/nᵢ) (13). Greenwood variance V̂[Ŝ(t)] = Ŝ(t)²·Σ dᵢ/[nᵢ(nᵢ−dᵢ)] (17); pointwise 95% CIs use the complementary log-log transform so bounds stay in [0,1]. Median = smallest t with Ŝ(t) ≤ 0.5 (R/`survfit`/Prism convention), with Brookmeyer–Crowley CI (19). *Validation parameter:* median survival, Greenwood SE. *Benchmark:* Freireich 1963 (6).

**(iii) Log-rank (Mantel–Cox).** χ² = (Σ(d₁ⱼ − e₁ⱼ))² / Σ v₁ⱼ over event times, with eᵢⱼ, vᵢⱼ the hypergeometric expectation/variance; df = G−1. Optional Gehan–Breslow/Tarone–Ware weights; pairwise tests BH-corrected (15). *Validation parameter:* χ², p. *Benchmark:* Freireich (6).

**(iv) Cox proportional hazards.** h(t|x) = h₀(t)·exp(βᵀx); HR = exp(β) by maximising the partial likelihood (14); ties by Breslow's approximation. *Validation parameter:* HR, β. *Benchmark:* Freireich (6); cross-checked vs R `coxph(ties="breslow")`.

**(v) Circadian period (χ² periodogram).** For trial period P (p bins, K complete cycles, N′=Kp), with phase means X̄_h and grand mean X̄: **Q_P = N′·K·Σ_h(X̄_h−X̄)² / Σ(Xᵢ−X̄)²**, distributed χ²_{p−1} under the no-rhythm null (E[Q_P]=p−1) (11). The peak P is the dominant period; significance uses a **Bonferroni-corrected** threshold across the tested periods (otherwise the family-wise false-positive rate is ≈1−(1−α)ⁿ). A Lomb–Scargle periodogram is an independent cross-check. *Validation parameter:* peak period, sensitivity/false-positive rate. *Benchmark:* Konopka & Benzer 1971 (4): 24/19/28 h.

**(vi) Sleep.** A sleep bout = ≥5 consecutive minutes of zero beam crossings (9, 10), giving total/day/night sleep, bout number/length. *Validation parameter:* exactness at the 4- vs 5-min boundary; sexual dimorphism. *Benchmark:* Hendricks/Shaw 2000 (9, 10).

**(vii) Y-maze.** %alternation = 100·(alternation triads)/(entries−2), tested against a configurable 50% chance level (7); turn bias = R/(R+L) (8). Individuality is a Monte-Carlo test comparing the observed mean-absolute-deviation of turn bias to a common-bias binomial null. Under the documented arm geometry an alternation triad equals two consecutive same-direction turns, so an unbiased fly alternates at exactly 50% (chance emerges from the model). *Validation parameter:* mean turn bias, % extreme, chance level. *Benchmark:* Buchanan/de Bivort 2015 (8); Lewis 2017 (7).

**(viii) Larval crawling & grooming.** Centroid tracking gives path length, displacement, speed (mm s⁻¹ via px·mm⁻¹) and tortuosity; grooming gives %-time grooming, bout count/duration. *Benchmarks:* WT larval speed ≈1 mm s⁻¹ (2); grooming ≈13% of time (12).

**(ix) Motor Decline Rate (MDR).** Climbing PI declines with **age** (negative geotaxis is age-dependent (1, 2)); DroPhenix fits **PI(age) = PI₀·exp(−k·age)** per group over an `Age` (days) axis (linear fallback), reporting k, Δk (treatment rescue) and R². The per-second timecourse is used only for the climbing curve/AUC/t₅₀; when no age axis is present MDR is explicitly reported as not applicable rather than mis-fit to seconds. *Validation parameter:* recovered k (Pearson r vs ground truth), median R². *Benchmark:* analytic ground truth (5 sample sets).

**(x) Cross-strain compound ranking (core innovation).** For strain *i*, the drug-vs-vehicle effect is the small-sample-corrected standardized mean difference **g_i = J·(x̄_t−x̄_c)/s_p**, J = 1 − 3/[4(n₁+n₂)−9] (16), with sampling variance that depends explicitly on the group sizes:
**Var(g_i) = J²·[ (n₁+n₂)/(n₁n₂) + g_i²/(2(n₁+n₂−2)) ]**.
Because Var(g_i) grows as nᵢ shrinks, the minimum-variance unbiased combination weights each strain by its precision (Gauss–Markov / actuarial credibility): wᵢ = 1/Var(g_i),
**g_pooled = Σ wᵢ g_i / Σ wᵢ,  Var(g_pooled) = 1/Σ wᵢ**.
Between-strain heterogeneity uses DerSimonian–Laird random effects (18): τ² = max(0, (Q−(k−1))/C), Q = Σ wᵢ(g_i−g_F)², C = Σwᵢ − Σwᵢ²/Σwᵢ; weights become wᵢ* = 1/(Var(g_i)+τ²); I² = max(0, (Q−(k−1))/Q). Survival effects enter as the Cox log-HR rescaled to SMD units (d = logHR·√3/π). Per-assay pooled effects are combined across assays by inverse-variance weighting into a Therapeutic Score (oriented so positive = benefit) with 95% CI, p and rank. *Validation parameter:* pooled estimate/Q/τ²/I² vs statsmodels; estimator RMSE and CI coverage vs naïve averaging. *Benchmark:* statsmodels; Monte-Carlo simulation.

### Validation and benchmarking strategy
Each module ships a runnable script (`validation/validate_*.py`, aggregated by `run_all.py`) reproducing published statistics or recovering analytic ground truth; `comparison_study.py` runs DroPhenix head-to-head against established tools on identical data. Reference tools are installed and executed live: **R `survival`** (21) and base-R `aov`/`kruskal.test`, **lifelines** (22), **statsmodels** (23), **scipy** (24), **pingouin** (25), **astropy** (26) and **pyBOAT** (27). MATLAB- and R-GUI-based assay tools and proprietary/web tools (GraphPad Prism, OASIS) could not be executed headlessly and are not tabulated; their statistical cores are represented by the validated libraries run here.

---

## RESULTS

### Each module reproduces published benchmark statistics
DroPhenix was validated against canonical datasets with known results (Table 1). On the Freireich 6-MP trial (6) it reproduced median survival (23 vs 8 weeks), the log-rank χ² (16.79, p = 4.2×10⁻⁵), Gehan–Breslow (13.46) and Cox HR (4.52, Breslow ties) exactly. The statistics engine reproduced the PlantGrowth one-way ANOVA (F₍₂,₂₇₎ = 4.846, p = 0.01591) and Kruskal–Wallis (H = 7.99) (20). The χ² periodogram recovered the Konopka–Benzer *period*-mutant periods (≈24/19/28 h) (4) with 100% sensitivity and 0% false positives on arrhythmic series; the 5-min sleep rule (9, 10) was exact at the 4-/5-min boundary. The Y-maze recovered the de Bivort handedness statistics (mean ≈0.50; ≈23.5% outside [0.3,0.7]) (8) and the 50% alternation chance (7); grooming calibrated to ≈13% time (12); larval tracking recovered a known velocity and ≈1 mm s⁻¹ WT speed (2). MDR recovered known decline constants across five sample sets (Pearson r = 0.989; median R² = 0.99) and correctly ordered age decline (disease > WT) with drug-mediated slowing; synthetic climbing videos returned exact counts/PI. Meta-analytic pooling matched statsmodels' fixed/DL/Q/τ²/I² to six decimals, and the ranking engine recovered the known efficacy order of a three-compound screen. **All 12 validation scripts pass.** Representative outputs are shown in **Figure 1**.

### DroPhenix correctly compares compounds across strains with unequal N (core innovation)
The platform's principal contribution is statistically rigorous cross-strain compound comparison under unequal sample sizes (**Figure 2**). In a worked example—one compound versus vehicle in four disease strains of sizes n = 3 (TDP43), 4 (SOD1), 5 (FUS) and 7 (HTT-Q93)—each per-strain effect (Hedges' *g*) is weighted by its precision, so the n = 7 strain carries 31% of the weight and the n = 3 strain 20%; the small strain is down-weighted rather than allowed to dominate or being discarded (Figure 2A). DerSimonian–Laird random effects report between-strain heterogeneity (I²). Critically, this is not merely a convention: in 3000 simulations with a known true effect (g = 0.7) across strains of sizes 3,7,5,4,9, inverse-variance pooling had **19.9% lower RMSE than equal-weight averaging** (0.267 vs 0.334) and **95.8% confidence-interval coverage** (target 95%), whereas the naïve mean was more variable (Figure 2B). DroPhenix thus delivers an efficient, well-calibrated combined estimate of a compound's effect precisely in the unbalanced multi-strain setting that defeats spreadsheet workflows—answering the first question a statistical reviewer asks of a screen.

### DroPhenix agrees with established tools on identical data
On identical benchmark data, DroPhenix was compared head-to-head with seven established, independently developed tools: R `survival` and base-R statistics, lifelines, statsmodels, scipy, pingouin, astropy and pyBOAT (the last a wavelet-ridge circadian package) (Table 2; **Figure 3**). Across **24 reported metrics the agreement was exact within tolerance (Pearson r = 1.0000)**, spanning survival (Kaplan–Meier, log-rank, Cox), ANOVA/Kruskal–Wallis, effect size, meta-analytic pooling and circadian period. The only apparent discrepancy—the Cox hazard ratio (DroPhenix/R-Breslow 4.523 vs lifelines/R-Efron 4.817)—is the known Breslow-vs-Efron tie-handling difference (the Freireich data contain many tied event times); DroPhenix's Breslow value matches R `coxph(ties="breslow")` to five significant figures (4.52307) and the published value. Circadian period agreed across three independent estimators (DroPhenix χ² periodogram, astropy Lomb–Scargle, pyBOAT wavelet: 24.00/23.96/23.96 h). We additionally attempted a scikit-survival cross-check, but it could not be built in our environment (its `ecos` dependency requires a C toolchain and no Python 3.13 wheel exists); survival is nonetheless already cross-validated against two independent implementations spanning both tie methods (R `survival`, Breslow; lifelines, Efron). This dual comparative analysis—versus published ground truth and live reference software—establishes the numerical correctness of every output.

### Tables
**Table 1.** Per-module method, formula, validation parameter, benchmark source and reproduced value.

| Module | Core formula | Validation parameter | Benchmark (source) | DroPhenix result |
|---|---|---|---|---|
| Statistics | ANOVA F / Kruskal H; Hedges *g* | F, H, *g* | PlantGrowth (Dobson 1983) | F=4.846, p=0.01591; H=7.99 ✓ |
| Survival KM | Ŝ(t)=Π(1−dᵢ/nᵢ); Greenwood; median | median, SE | Freireich 1963 | 23 / 8 wk ✓ |
| Log-rank | χ²=(ΣO−E)²/ΣV | χ², p | Freireich 1963 | 16.79, p=4.2e-5 ✓ |
| Cox PH | HR=exp(β), partial likelihood | HR (Breslow) | Freireich 1963 / R | 4.523 (=R 4.52307) ✓ |
| Circadian | Q_P χ² periodogram (Bonferroni) | peak period; FPR | Konopka & Benzer 1971 | 24/19/28 h; 0% FPR ✓ |
| Sleep | ≥5-min inactivity | bout boundary | Hendricks/Shaw 2000 | exact 4↔5 min ✓ |
| Y-maze | %alt; turn bias R/(R+L); MAD test | mean bias; %extreme | de Bivort 2015; Lewis 2017 | 0.50; 23.5%; 50% chance ✓ |
| Grooming | %time = Σbout/observation | % time grooming | Qiao 2018 | ≈13% ✓ |
| Larval | speed = path·(px·mm⁻¹)⁻¹/t | speed | Gargano 2005 | ≈1 mm/s; exact on known v ✓ |
| MDR | PI(age)=PI₀e^(−k·age) | k recovery (r) | analytic (5 sets) | r=0.989; R²=0.99 ✓ |
| Ranking | g (J-corrected); IV/DL pooling | pooled, Q, τ², I²; RMSE, coverage | statsmodels; simulation | exact; 19.9% lower RMSE; 95.8% coverage ✓ |

**Table 2.** Head-to-head agreement between DroPhenix and established, independently developed tools on identical benchmark data.

| Domain | Metric | DroPhenix | Reference value | Reference tool(s) |
|---|---|---|---|---|
| Survival | Median survival, 6-MP / Placebo (wk) | 23.0 / 8.0 | 23.0 / 8.0 | lifelines; R `survfit` |
| Survival | Log-rank χ² | 16.7929 | 16.79294 | lifelines; R `survdiff` |
| Survival | Log-rank p | 4.2×10⁻⁵ | 4.2×10⁻⁵ | lifelines; R `survdiff` |
| Survival | Cox HR (Breslow ties) | 4.5231 | 4.52307 | R `coxph`; =published 4.523 |
| Statistics | One-way ANOVA F | 4.8461 | 4.84609 | scipy; pingouin; R `aov` |
| Statistics | ANOVA p | 0.01591 | 0.01591 | scipy; R `aov` |
| Statistics | Kruskal–Wallis H | 7.9882 | 7.98823 | scipy; R `kruskal.test` |
| Statistics | Hedges' *g* | 0.9140 | 0.9140 | pingouin |
| Meta-analysis | Pooled effect, fixed / DL-random | 0.4103 / 0.4288 | 0.4103 / 0.4288 | statsmodels |
| Meta-analysis | Heterogeneity I² (%) | 27.73 | 27.73 | statsmodels |
| Circadian | Dominant period (h) | 24.00 | 23.96 / 23.96 | astropy (Lomb–Scargle); pyBOAT (wavelet) |

All **24/24 quantitative comparisons agree within tolerance** (Pearson r = 1.0000 across DroPhenix vs reference values; Figure 3). The Cox hazard ratio under Efron ties (4.817; lifelines / R `coxph(ties="efron")`) reflects the alternative tie approximation and is reported as a method note, not a discrepancy. scikit-survival could not be built in our environment (no C toolchain / no Python 3.13 wheel). The complete 24-metric comparison is given in Supplementary Table S1 (machine-readable: `validation/comparison_report.md`).

### Figure legends
**Figure 1. DroPhenix multi-assay analysis and validation.** (A) Kaplan–Meier survival on the Freireich 6-MP trial with the lifelines estimate overlaid (dotted), showing exact agreement. (B) Climbing kinetics (PI vs assay seconds) for wild-type and TDP-43 ± drug. (C) Motor Decline Rate: exponential decline of PI over age with fitted constants (WT k≈0.008; TDP-43-vehicle k≈0.027; drug k≈0.015 d⁻¹). (D) χ² (Sokolove–Bushell) periodogram recovering a 24-h period with the Bonferroni-corrected significance threshold. (E) Y-maze turn-bias distribution (mean ≈0.50; central [0.3–0.7] band shaded). (F) Cross-assay Therapeutic-Score ranking of three compounds (green: p<0.05). Colours follow the Okabe–Ito colour-blind-safe palette.

**Figure 2. Core innovation: actuarial cross-strain compound comparison under unequal N.** (A) Inverse-variance forest plot of one compound's effect (Hedges' *g*) in four disease strains of unequal size; marker area is proportional to the precision weight, *n* and weight % are annotated, and the pooled DerSimonian–Laird estimate is the green diamond; the vermilion dashed line is the naïve equal-weight mean. (B) Estimator efficiency: distributions of the pooled effect from 3000 simulations (true g = 0.7; strain sizes 3,7,5,4,9). Inverse-variance pooling (green) is tighter and lower-RMSE (0.267) than the naïve mean (vermilion, 0.334) and is centred on the truth.

**Figure 3. Agreement with established tools.** DroPhenix values versus the corresponding reference-tool values across 24 metrics from R `survival`/base, lifelines, statsmodels, scipy, pingouin, astropy and pyBOAT on identical data; points lie on the identity line (r = 1.0000).

---

## DISCUSSION

DroPhenix's contribution is the correct, transparent and **verifiable** application of established methods across the full *Drosophila* phenotyping panel, plus a genuinely novel, statistically justified solution to unequal-N cross-strain compound comparison. Three design choices are central to translational rigour. First, tests are **assumption-gated** (normality and variance tested before any parametric test; effect sizes and FDR control mandatory). Second, the platform enforces the **scientifically correct measurement axis**—most notably that Motor Decline Rate is a decline over age, not over the within-assay seconds during which flies climb—preventing a silent category error. Third, cross-strain evidence is combined by **inverse-variance/random-effects meta-analysis**, which we show is the efficient, well-calibrated estimator under unbalanced designs, with explicit heterogeneity reporting.

The main limitation is that automated behavioural *classification* from raw video (e.g., grooming microstates) is out of scope; DroPhenix analyses scored intervals and tracked trajectories. We also restricted the comparative benchmark to software runnable headlessly and reproducibly; proprietary (Prism), web-only (OASIS) and MATLAB/R-GUI assay tools were not tabulated, though their statistical cores are represented by the libraries we ran. Demonstration cohorts are literature-calibrated synthetic datasets that reproduce published summary statistics, provided so the tool is verifiable on day one; user data are accepted directly via the upload paths.

## DATA AVAILABILITY
DroPhenix is freely available at https://drophenix.streamlit.app (no registration). The source code, the complete validation suite (`validation/`), the cross-tool comparison (`validation/comparison_study.py`, `r_comparison.R`), per-assay data templates (`templates/`) and the figure-generation scripts are distributed under the GNU General Public License v3. All benchmark datasets used here are published (Freireich 1963; PlantGrowth) or are reproducibly generated by the shipped scripts.

## SUPPLEMENTARY DATA
Supplementary Data are available online. **Supplementary Table S1** gives the complete tool-by-tool comparison (all 24 quantitative metrics plus the two method notes). **Supplementary Table S2** lists the per-module validation tests against their published benchmarks. **Supplementary Note S1** provides the unequal-N worked example (per-strain weights) and the estimator-efficiency simulation underlying Figure 2. All are regenerated by the shipped scripts (`validation/run_all.py`, `comparison_study.py`, `demo_unequal_n.py`).

## AUTHOR CONTRIBUTIONS
[To be confirmed by the authors.] K.G. designed and implemented the platform and performed the validation and benchmarking; V.S. supervised the work; both authors wrote and approved the manuscript.

## ACKNOWLEDGEMENTS
[To be completed by the authors.]

## FUNDING
[To be completed by the authors — including any grant numbers and the open-access funding source required by the journal.]

## CONFLICT OF INTEREST
None declared.

---

## REFERENCES
1. Feany,M.B. and Bender,W.W. (2000) A *Drosophila* model of Parkinson's disease. *Nature*, **404**, 394–398.
2. Gargano,J.W., Martin,I., Bhandari,P. and Grotewiel,M.S. (2005) Rapid iterative negative geotaxis (RING): a new method for assessing age-related locomotor decline in *Drosophila*. *Exp. Gerontol.*, **40**, 386–395.
3. Rhodenizer,D., Martin,I., Bhandari,P., Pletcher,S.D. and Grotewiel,M. (2008) Genetic and environmental factors impact age-related impairment of negative geotaxis in *Drosophila* by altering age-dependent climbing speed. *Exp. Gerontol.*, **43**, 739–748.
4. Konopka,R.J. and Benzer,S. (1971) Clock mutants of *Drosophila melanogaster*. *Proc. Natl. Acad. Sci. USA*, **68**, 2112–2116.
5. (2017) An Automated Rapid Iterative Negative Geotaxis Assay for Analyzing Adult Climbing Behavior in a *Drosophila* Model of Neurodegeneration. *J. Vis. Exp.*, e56507 (PMID 28931001).
6. Freireich,E.J. *et al.* (1963) The effect of 6-mercaptopurine on the duration of steroid-induced remissions in acute leukemia. *Blood*, **21**, 699–716.
7. Lewis,S.A., Negelspach,D.C., Kaladchibachi,S., Cowen,S.L. and Fernandez,F. (2017) Spontaneous alternation: a potential gateway to spatial working memory in *Drosophila*. *Neurobiol. Learn. Mem.*, **142**, 230–235.
8. Buchanan,S.M., Kain,J.S. and de Bivort,B.L. (2015) Neuronal control of locomotor handedness in *Drosophila*. *Proc. Natl. Acad. Sci. USA*, **112**, 6700–6705.
9. Hendricks,J.C. *et al.* (2000) Rest in *Drosophila* is a sleep-like state. *Neuron*, **25**, 129–138.
10. Shaw,P.J. *et al.* (2000) Correlates of sleep and waking in *Drosophila melanogaster*. *Science*, **287**, 1834–1837.
11. Sokolove,P.G. and Bushell,W.N. (1978) The chi square periodogram: its utility for analysis of circadian rhythms. *J. Theor. Biol.*, **72**, 131–160.
12. Qiao,B. *et al.* (2018) Automated analysis of long-term grooming behavior in *Drosophila* using a k-nearest neighbors classifier. *eLife*, **7**, e34497.
13. Kaplan,E.L. and Meier,P. (1958) Nonparametric estimation from incomplete observations. *J. Am. Stat. Assoc.*, **53**, 457–481.
14. Cox,D.R. (1972) Regression models and life-tables. *J. R. Stat. Soc. B*, **34**, 187–220.
15. Benjamini,Y. and Hochberg,Y. (1995) Controlling the false discovery rate. *J. R. Stat. Soc. B*, **57**, 289–300.
16. Hedges,L.V. and Olkin,I. (1985) *Statistical Methods for Meta-Analysis*. Academic Press, Orlando.
17. Greenwood,M. (1926) The natural duration of cancer. *Rep. Public Health Med. Subj.*, **33**, 1–26.
18. DerSimonian,R. and Laird,N. (1986) Meta-analysis in clinical trials. *Control. Clin. Trials*, **7**, 177–188.
19. Brookmeyer,R. and Crowley,J. (1982) A confidence interval for the median survival time. *Biometrics*, **38**, 29–41.
20. Dobson,A.J. (1983) *An Introduction to Statistical Modelling*. Chapman & Hall, London.
21. Therneau,T.M. and Grambsch,P.M. (2000) *Modeling Survival Data: Extending the Cox Model*. Springer, New York. (R `survival`.)
22. Davidson-Pilon,C. (2019) lifelines: survival analysis in Python. *J. Open Source Softw.*, **4**, 1317.
23. Seabold,S. and Perktold,J. (2010) statsmodels: econometric and statistical modeling with Python. *Proc. 9th Python in Science Conf.*
24. Virtanen,P. *et al.* (2020) SciPy 1.0. *Nat. Methods*, **17**, 261–272.
25. Vallat,R. (2018) Pingouin: statistics in Python. *J. Open Source Softw.*, **3**, 1026.
26. Astropy Collaboration (2022) The Astropy Project. *Astrophys. J.*, **935**, 167.
27. Mönke,G., Sorgenfrei,F.A., Schmal,C. and Granada,A.E. (2020) Optimal time frequency analysis for biological data — pyBOAT. *bioRxiv*, doi:10.1101/2020.04.29.067744.
