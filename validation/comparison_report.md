# DroPhenix benchmark vs established tools

All reference tools were installed and executed on identical data (Freireich 1963 6-MP trial; PlantGrowth; a known 24 h rhythm; a fixed set of effect sizes). Values are reproducible.

| Domain        | Metric                                                                                                                            | DroPhenix   | Reference tool          | Reference   | Abs diff   | Agree   |
|:--------------|:----------------------------------------------------------------------------------------------------------------------------------|:------------|:------------------------|:------------|:-----------|:--------|
| Survival      | Median 6-MP (weeks)                                                                                                               | 23.0        | lifelines               | 23.0        | 0.0        | yes     |
| Survival      | Median Placebo (weeks)                                                                                                            | 8.0         | lifelines               | 8.0         | 0.0        | yes     |
| Survival      | Log-rank chi-square                                                                                                               | 16.7929     | lifelines               | 16.79294    | 4e-05      | yes     |
| Survival      | Log-rank p-value                                                                                                                  | 4e-05       | lifelines               | 4e-05       | 0.0        | yes     |
| Survival      | Cox HR Breslow (vs published 4.523)                                                                                               | 4.5231      | Klein-Moeschberger 2003 | 4.523       | 0.0001     | yes     |
| Survival      | Cox HR Efron (ties-method difference)                                                                                             | 4.5231      | lifelines               | 4.81687     | 0.29377    | info    |
| Stats         | One-way ANOVA F                                                                                                                   | 4.8461      | scipy                   | 4.84609     | 1e-05      | yes     |
| Stats         | ANOVA p-value                                                                                                                     | 0.01591     | scipy                   | 0.01591     | 0.0        | yes     |
| Stats         | Kruskal-Wallis H                                                                                                                  | 7.9882      | scipy                   | 7.98823     | 3e-05      | yes     |
| Stats         | One-way ANOVA F                                                                                                                   | 4.8461      | pingouin                | 4.84609     | 1e-05      | yes     |
| Stats         | Hedges g (trt2 vs ctrl)                                                                                                           | 0.91404     | pingouin                | 0.91404     | 0.0        | yes     |
| Meta-analysis | Pooled (fixed)                                                                                                                    | 0.41034     | statsmodels             | 0.41034     | 0.0        | yes     |
| Meta-analysis | Pooled (DL random)                                                                                                                | 0.42883     | statsmodels             | 0.42883     | 0.0        | yes     |
| Meta-analysis | Heterogeneity I^2 (%)                                                                                                             | 27.72586    | statsmodels             | 27.72586    | 0.0        | yes     |
| Circadian     | Dominant period (h) [true 24]                                                                                                     | 24.0        | ground truth            | 24.0        | 0.0        | yes     |
| Circadian     | Dominant period (h)                                                                                                               | 24.0        | astropy LombScargle     | 23.96404    | 0.03596    | yes     |
| R: survival   | Median 6-MP                                                                                                                       | 23.0        | R survfit               | 23.0        | 0.0        | yes     |
| R: survival   | Median Placebo                                                                                                                    | 8.0         | R survfit               | 8.0         | 0.0        | yes     |
| R: survival   | Log-rank chi-square                                                                                                               | 16.7929     | R survdiff              | 16.79294    | 4e-05      | yes     |
| R: survival   | Log-rank p-value                                                                                                                  | 4e-05       | R survdiff              | 4e-05       | 0.0        | yes     |
| R: survival   | Cox HR (Breslow)                                                                                                                  | 4.5231      | R coxph(breslow)        | 4.52307     | 3e-05      | yes     |
| R: stats      | ANOVA F                                                                                                                           | 4.8461      | R aov                   | 4.84609     | 1e-05      | yes     |
| R: stats      | ANOVA p-value                                                                                                                     | 0.01591     | R aov                   | 0.01591     | 0.0        | yes     |
| R: stats      | Kruskal-Wallis H                                                                                                                  | 7.9882      | R kruskal.test          | 7.98823     | 3e-05      | yes     |
| Circadian     | Dominant period (h)                                                                                                               | 24.0        | pyBOAT wavelet          | 23.9598     | 0.0402     | yes     |
| Survival      | scikit-survival not installable here (no C toolchain / no py3.13 wheel); survival already cross-checked vs R survival + lifelines | -           | scikit-survival         | -           | -          | info    |

**Agreement: 24 of 26 comparisons agree within tolerance; 0 disagree; 2 are documented method differences or unavailable comparators.**

Every row is counted in the denominator. Rows tagged `info` are method differences (for example, Breslow versus Efron handling of tied event times, where both answers are correct for their respective conventions) or comparators that could not be installed in this environment. They are reported here rather than excluded, because a benchmark whose denominator cannot express a disagreement is not a benchmark.
