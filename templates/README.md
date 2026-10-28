# DroPhenix data templates

One template per assay. Keep the **header row exactly as shown**; add your rows
below. `Genotype`, `Treatment`, `Sex` are the grouping columns used everywhere
(use a consistent vehicle/control label, e.g. `Vehicle`, across assays so the
Drug Ranking module can combine them). Group sizes may be unequal — the
statistics handle that.

| File | Assay | Required columns | Notes |
|------|-------|------------------|-------|
| `climbing_template.csv` | Climbing (negative geotaxis) | Genotype, Treatment, Sex, Replicate, Time, Count_climbed, n_total_flies | One row per (vial × timepoint). Time in seconds; Count_climbed = flies above the line; n_total_flies = total in vial. PI = 2·(Count/n)·100 − 100. |
| `lifespan_event_template.csv` | Lifespan (event-level) | Genotype, Treatment, Sex, Time, Event | One row per fly. Time = day of death or censoring; Event = 1 death, 0 censored (alive at end / lost). |
| `lifespan_cohort_template.csv` | Lifespan (daily counts) | Genotype, Treatment, Sex, Day, Dead_count, Censored_count, Total_start | One row per scoring day per cohort. Censored_count optional. Expanded internally to event-level. |
| `ymaze_template.csv` | Y-maze | Genotype, Treatment, Sex, FlyID, Sequence | Sequence = ordered arm-entry string using 3 arm labels (e.g. `A`,`B`,`C`), e.g. `ABCACB`. (Long per-entry rows with `Order`,`Arm` also accepted.) |
| `dams_template.csv` | Activity monitoring (DAMS) | Genotype, Treatment, Sex, FlyID, Time_min, Activity | One row per fly per 1-min bin. Activity = beam crossings. Sleep = ≥5 min of zero activity. Several days needed for circadian period. |
| `larval_template.csv` | Larval crawling (summary) | Genotype, Treatment, Sex, FlyID, mean_speed_mm_s, path_length_mm, displacement_mm, tortuosity | One row per larva. (Or track a video directly in the app to generate these.) |
| `grooming_template.csv` | Grooming (bout-level) | Genotype, Treatment, Sex, FlyID, Bout_start_s, Bout_end_s, Observation_s | One row per scored grooming bout. Observation_s = total scored time per fly. (Per-fly summary with `Total_grooming_s` also accepted.) |

**Drug Ranking** does not need its own template — it combines the four assay
files above (climbing, lifespan, ymaze, dams) by Genotype × Treatment × Sex.

## Design-aware templates (longitudinal, time-of-day, toxin, historical controls)

These add the optional columns that unlock the hierarchical, trajectory,
circadian, dose-response and control-borrowing analyses. They are backwards
compatible: a file without them parses exactly as before.

| File | Use | Key added columns |
|------|-----|-------------------|
| `climbing_longitudinal_template.csv` | Same vials followed across ages and times of day | `VialID`, `Age`, `TimeOfDay`, `SessionIndex`, `TrialIndex`, `DeathAge`, `ExperimentID` |
| `toxin_dose_response_template.csv` | Toxin or drug dose series | `Compound`, `Dose`, `DoseUnit`, `n_dead` |

**A trap in the toxin template worth stating explicitly.** `n_total_flies` and
`n_dead` can be read two ways, and the numbers alone cannot say which you meant:

- `n_total_flies` = flies **scored** (survivors), so flies exposed = `n_total_flies + n_dead`
- `n_total_flies` = flies **exposed** at the start, so survivors = `n_total_flies - n_dead`

The shipped template uses the **first** reading: `n_total_flies` falls with dose
because dead flies are no longer scored. The Dose-Response page asks which
convention your file uses rather than guessing. Whichever you adopt, use it
consistently — mixing the two silently changes every mortality rate and hence
the LC50.
| `historical_controls_template.csv` | Past control arms, for borrowing | `ExperimentID`, `Batch`, `Date` |

Five columns carry most of the value, and none can be reconstructed after the
fact:

- **`VialID`** — the experimental unit. Flies in a vial are not independent, so
  without this the intraclass correlation cannot be estimated and every p-value
  is anticonservative. Use a unique string per physical vial, stable across the
  whole experiment.
- **`DeathAge`** — the age at which the flies in that vial died. This is the
  single highest-value optional column: without it, survivor bias in an ageing
  trajectory cannot be measured, only assumed to be zero, and time-to-death
  alignment is unavailable.
- **`TimeOfDay`** — record the **actual clock time** (`09:15`), not the category
  "Morning". Categories are mapped to conventional Zeitgeber midpoints, and the
  estimated circadian phase then inherits the width of the whole testing window.
- **`SessionIndex`** — how many times this vial has been tested before. Converts
  an assumption about fatigue and habituation into a testable covariate.
- **`ExperimentID`** — which experiment each row belongs to. This is what makes
  between-experiment heterogeneity estimable, and therefore what turns a pile of
  past control arms into a usable prior.

`n_total_flies` should be the number **alive in that vial at that timepoint**,
not the number originally set up. Density affects climbing, and attrition changes
density differentially between groups.

Two design rules that no software can repair afterwards: use **at least two
vials per treatment arm** (with one, the treatment and vial effects are a single
parameter and cannot be separated), and **counterbalance testing order against
time of day** (otherwise the two are completely confounded). See
`docs/UPLIFT_PLAN.md` §3 for the full specification.

Example rows in each file are illustrative synthetic values, not measured data —
replace them with your own.
