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

Example rows in each file are illustrative synthetic values, not measured data —
replace them with your own.
