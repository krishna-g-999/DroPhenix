# DAM Run Analysis — raw monitor files to a verified report

**Page:** *DAM Run Analysis* (beside *Activity Monitoring (DAMS)*).
**CLI:** `python drophenix_cli.py dam-run`.
**Module:** `modules/dam_pipeline.py`, `modules/dam_figures.py`, `modules/dam_report.py`.
**Validation:** `validation/validate_dam_pipeline.py` (part of `validation/run_all.py`).

This is the analysis behind `docs/DAM_RESULTS_REPORT.docx`, made reusable. It
does not know any genotype or treatment name in advance; it reads the design from
the channel map. Its acceptance test is that, given only the laboratory's raw
files and map, it reproduces that report's windows, flies, outliers and every
effect size and p-value.

## What you supply

1. **Raw TriKinetics files**, one per monitor (`Monitor_1.txt`, `Monitor_2.txt`, …),
   exported at **1-min bins**. A coarser export (for example 60 min) is refused:
   the ≥5-min sleep rule cannot be applied to it.
2. **A channel map** CSV with columns `Monitor, Channel, Genotype, Sex, Treatment`
   (template: `templates/dams_template.csv`). Monitor names need only share the
   first number with the file name (`Monitor_1_25th day` matches `Monitor_1.txt`).
3. Optionally a **second run** (another timepoint of the same design) with its
   own files and, if different, its own map.

## What it does, in order

| Step | Rule | Why |
|---|---|---|
| Recording quality | Per-monitor light channel table; the monitor whose light channel changes state most often supplies Zeitgeber time; the monitors' agreement is reported and a stuck sensor named. | On the 11–15 Aug 2026 run one sensor was stuck on; using it would have put the whole night in "day". |
| Photoperiod | Measured as the median lights-on span ≥60 min. | The two validation recordings ran on different photoperiods; assuming 12:12 mis-scored day and night sleep on one of them. |
| Analysis window | A day is usable when it has 1,440 min of records, light hours within 0.5 h of the median over complete days, ≤3 light transitions and no gap >30 min. The window is the longest run of consecutive usable days. Fewer than 3 days is allowed with a warning; none at all is refused. The window can be overridden. | On the two validation recordings this recovers the windows chosen by hand and excludes the days on which the light cycle failed or flickered. |
| Per-fly parameters | Total / day / night sleep, bout number and length (day, night), longest bout, night sleep latency, waking activity, total activity; phase-normalised sleep and activity per light- and dark-hour. Dead flies are detected by the reader. | The parameter set of Pfeiffenberger 2010, Chiu 2010 and ShinyR-DAM. |
| Outliers | Modified z (Iglewicz & Hoaglin) >3.5 versus the group median on total sleep and total activity, MAD pooled per run. Flagged flies are **kept** in the primary analysis; every family is rerun without them as a sensitivity analysis. | Pre-specified for the report; a fly is never silently dropped. |
| Design | Reference genotype and treatment guessed from common names (wt, control, ctrl, w1118, …; untreated, vehicle, dmso, …) and overridable. A contrast is **monitor-clean** when both arms occupy the same set of monitors, otherwise **confounded** and excluded from the pooled primary analysis. | On a map in which a genotype or a treatment arm sits entirely in one monitor, this is detected and the contrast is labelled confounded without anyone having to say so. |
| Tests | van Elteren stratified Wilcoxon (exact, stratified by sex), Hedges' g pooled by inverse variance with a DerSimonian–Laird random-effects check, BH-FDR within each family (primary tier); per-sex Mann–Whitney uncorrected (secondary tier, labelled). | The methods validated in `validate_dam_parameters.py` and `validate_stats.py`. |
| Two runs | Unpaired stratified contrasts on phase-normalised parameters, and a pairing test reported so the reader knows why unpaired. | In the validation recordings the tubes did not hold the same individual flies at the two timepoints (within-group ρ ≈ 0), which the pairing test detects. |
| Report | Word report from the run object; `verify()` reads it back and traces every numeric cell to the tables. | Hand-typed numbers went wrong six times in September 2026. |

## Reading the output

* **Answers** — one line per family: how many parameters pass FDR 5 %, in which
  direction, with the caveat if the contrast is confounded or the window short.
* **Figures** — headline effect-size figure, sleep profiles (by genotype, by
  treatment, by run), bar charts with individual flies, age comparison, outlier map.
* **Tables** — per-fly parameters, day-by-day recording quality, every contrast
  family, all-flies and without-outliers.
* **Report & downloads** — the verified `.docx`, all CSVs and PNGs as a zip.

A result marked *confounded* is reported, but it is not evidence for or against
the treatment: the arms sat in different monitors and monitor effects cannot be
separated from the biology in that design. A window shorter than 3 days is
analysed with a warning that per-fly estimates will be noisy.

## What is and is not established

The pipeline reproduces the hand analysis of the only two recordings available
to floating-point precision (`validate_dam_pipeline.py`, sections C–D). It has
not been run on any other laboratory's data, and the window and outlier rules
are the ones pre-specified for that report — not the only defensible choices.
Nothing in this module is a new statistical method.

## Command line

```bash
python drophenix_cli.py dam-run --monitors Monitor_1.txt Monitor_2.txt --map channel_map.csv --name "Day 25" --second D15/Monitor_1.txt D15/Monitor_2.txt --second-name "Day 15" --out results/my_run
```

Options: `--ref-genotype`, `--ref-treatment`, `--no-stratify`,
`--window YYYY-MM-DD:YYYY-MM-DD` (first:last day), `--second-map`.
