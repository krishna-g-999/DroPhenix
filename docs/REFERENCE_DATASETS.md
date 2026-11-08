# Real research datasets for DroPhenix validation

**2026-09-08.** Every dataset found, whether it downloaded, and what it does or could validate.
All are openly licensed (CC0 or CC-BY) and cited to their paper.

Current state: **49 suites, exit 0** (was 36 suites / 896 checks when
this document was written on 2026-09-08; P0 Blocks A-D and the two real
sleep datasets have been added since).

---

## 1. Downloaded, wired into a validation suite

| Module validated | Dataset | Scale | Suite | Result |
|---|---|---|---|---|
| **survival.py** (lifespan) | Heritability of lifespan is largely sex-limited in *Drosophila*, *Am Nat* 2013, [doi:10.1086/673296](https://doi.org/10.1086/673296) · data [doi:10.5061/dryad.q412q](https://doi.org/10.5061/dryad.q412q) · [Zenodo 4996358](https://zenodo.org/records/4996358) | **39,581 individual flies**, 50 hemiclonal lines, 200 groups, recorded BOTH as individual death ages and as daily death counts | `validate_real_lifespan_benchmark` | 15/15 |
| **motor_decline.py** (MDR), survival, MSCI | Patterns of physiological decline due to age and selection, *Evolution* 2016, [doi:10.1111/evo.13065](https://doi.org/10.1111/evo.13065) · data [doi:10.5061/dryad.qb509](https://doi.org/10.5061/dryad.qb509) · [Zenodo 4954719](https://zenodo.org/records/4954719) | 10 populations, geotaxis + mortality across age, cohorts of 850–5,000 | `validate_real_aging_benchmark` | 28/28 |
| **`larval.py` trajectory differentiation** | Speed–curvature power law, *Biol Lett* 2016, [doi:10.1098/rsbl.2016.0597](https://doi.org/10.1098/rsbl.2016.0597) · [Zenodo 5010896](https://zenodo.org/records/5010896) | 123 larvae, 3 conditions, 7 fps centroid tracks | `validate_larval_powerlaw` | 19/19 |
| **`grooming.py`, both input layouts** | Developmental loss of neurofibromin drives excessive grooming, *PLOS Genet* 2020, [doi:10.1371/journal.pgen.1008920](https://doi.org/10.1371/journal.pgen.1008920) · [Zenodo 3964901](https://zenodo.org/records/3964901) | 108 flies (summary) + 1,096 scored bouts (bout-level), body part annotated | `validate_grooming_nf1` | 21/21 |
| **climbing velocity read-out** | FreeClimber, *J Exp Biol* 2021, [doi:10.1242/jeb.229377](https://doi.org/10.1242/jeb.229377) · [GitHub](https://github.com/adamspierer/FreeClimber) | 4,093 particle detections, 3 vials, real w1118 recording | `validate_freeclimber_benchmark` | 17/17 |
| dose-response | `drc` `ryegrass` (Inderjit, Streibig & Olofsdotter 2002) | 24 obs | `validate_dose_response` | agrees ≤ 9e-6 |
| stats engine | PlantGrowth (Dobson 1983) | 30 obs | `validate_stats` | exact |
| survival (textbook) | Freireich et al. 1963 6-MP trial | 42 patients (human) | `validate_survival` | exact |

### What the lifespan benchmark found

- Both input layouts of the same experiment agree: **99.0 %** of the 200 groups give an *identical* median, and the maximum disagreement is 2 days — one scoring interval, which is the finest possible difference when deaths are scored every other day.
- KM medians match `lifelines` on **195 of 200** real groups. Every one of the 5 disagreements is an **exact `S(t) = 0.5` tie**, which is the deviation documented in `_km_median` (DroPhenix takes the earlier time). The documentation was accurate.
- **A real defect surfaced only because of the scale of this data:** the sex log-rank over 39,581 flies gives χ² = 10,625, and `round(p, 6)` printed the p-value as exactly **0.0**. Same defect class as the one fixed in `motor_decline` on 2026-09-06. `survival.py` now reports p-values at 3 significant figures and never returns 0.

---

## 2. Downloaded, characterised, NOT yet wired in — needs your input

### 2a. Sleep / DAM — ~~blocked~~ **RESOLVED 2026-11-07, ready to wire in**

> **Resolution.** The ambiguity below was mine, not the data's, and it is settled
> from the files themselves — no email to the authors needed.
>
> **The bin width is not constant across protocols, and it is readable per file.**
> Two independent confirmations agree: the increment of the `ZT` column, and the
> maximum cell value (minutes of sleep in a bin cannot exceed the bin width).
>
> | File set | ZT increment | Bin width | Bins/day | Max cell | Rows | Days |
> |---|---|---|---|---|---|---|
> | `6h deprivation/*` | 1.0 | **60 min** | 24 | 60.0 | 72 | 3 |
> | `12h deprivation/*` | 1.0 | **60 min** | 24 | 60.0 | 72 | 3 |
> | `24h deprivation/*` | 0.5 | **30 min** | 48 | 30.0 | 144 | 3 |
>
> The "physically impossible 1,083 min/day" was the artefact of applying a
> 30-minute width to a **60-minute-bin** file. At 60 min/bin the daily cap is
> 1,440 and 1,083 is unremarkable. Day index must come from the ZT wrap
> (a diff of −23.0 or −23.5), not from the row index.
>
> **Day 1 = baseline, day 2 = deprivation, day 3 = recovery**, and this is
> visible in the data without assuming it — mean sleep per fly per day, 20 s
> stimulation protocol:
>
> | Protocol | Group | Baseline | Deprivation | Recovery | n flies |
> |---|---|---|---|---|---|
> | 6 h | Control | 1082.7 | 1180.0 | 1136.7 | 44 |
> | 6 h | Deprived | 937.9 | **766.9** | 1125.1 | 62 |
> | 12 h | Control | 1141.7 | 1175.7 | 1175.9 | 32 |
> | 12 h | Deprived | 1105.8 | **622.7** | 1246.4 | 64 |
> | 24 h | Control | 1239.0 | 1283.8 | 1296.9 | 32 |
> | 24 h | Deprived | 1035.6 | **275.4** | 1200.3 | 32 |
>
> Deprivation is unambiguous on day 2 in every deprived group and absent in
> every control. Rebound against each group's own baseline: **+187.2** (6 h),
> **+140.6** (12 h), **+164.7** (24 h) for deprived, against +54.0, +34.2, +57.9
> for controls. The column counts (44, 32, 32) match the n reported in that
> paper's Figure 1, which corroborates the mapping independently.
>
> **Two cautions that belong in the suite, not in a footnote.**
>
> 1. **Column labels repeat within a file.**
>    `6h deprivation/Control_20sec_SleepData.txt` has header
>    `ZT I1..I14 I1 I3 I4 …` — replicate cohorts concatenated with reused
>    individual labels. A parser keyed on the column *name* will collide flies.
>    Key them by **position**, with a cohort-qualified synthetic id. This is the
>    same identifier-reuse trap the register tracks for `Replicate` (N4).
> 2. **Not an absolute sleep-amount anchor.** 1,083–1,297 min/day is 18.0–21.6 h
>    asleep, above every published DAM total-sleep value for *Drosophila*
>    (roughly 500–700 min/day female, 900–1,100 male) and above this
>    repository's own `PLAUSIBLE_SLEEP_MIN_PER_DAY = (300, 1150)`. Either these
>    files record binned inactivity rather than ≥5-min-rule sleep, or the
>    paradigm inflates scored immobility. So the suite may validate (a) the
>    per-day summary arithmetic under a **file-declared** bin width and (b) the
>    **direction** of homeostatic rebound — and must not cite the dataset as an
>    absolute sleep-amount benchmark. It still cannot validate the ≥5-min bout
>    rule itself; that needs raw beam counts (see 2b).
>
> The original text is kept below as the record of what was wrong and why.

### 2a (original). Sleep / DAM — **blocked on an ambiguity I could not resolve**

**Homeostatic control of deep sleep**, *eLife* 2024, [doi:10.7554/eLife.91355](https://doi.org/10.7554/eLife.91355) · data [doi:10.5061/dryad.qnk98sfpp](https://doi.org/10.5061/dryad.qnk98sfpp) · [Zenodo 10069769](https://zenodo.org/records/10069769)
Local: `reference_datasets/sleep_homeostasis/dataset.zip` (35 `.txt` files)

Format: column 1 is Zeitgeber time, remaining columns are individual flies (I1…I32), values are minutes of sleep per bin. Sleep-deprivation design with 6 h / 12 h / 24 h protocols, deprived vs undisturbed controls.

**Why I stopped.** The files do not share a bin width, and my reading of one gives a physically impossible result: the 6 h file has 24 bins/day, which at 30 min/bin caps a day at 720 minutes, yet mean sleep computes to **1,083 min/day**. The 24 h files segment cleanly (48 bins/day, 1,239–1,297 min/day, plausible). I also cannot tell from the README which day is baseline, which is deprivation and which is recovery — and my first attempt gave a *rebound in the wrong direction* for the 24 h protocol, which is a sign the mapping is wrong, not a finding.

**What I need from you (one minute of your expertise):** for each protocol, the bin width and which day index is the recovery day. With that, the suite writes itself — the ground truth is that deprived flies show rebound (more sleep on recovery than controls), which is a clean directional test of the sleep-summary path.

**Also note:** this dataset is *already-scored sleep*, not raw beam counts. It can validate the sleep **summary** arithmetic but **not** the ≥5-min-of-zero-activity bout rule, which needs raw DAM monitor files. For that, see 2b.

### ~~2b. Raw DAM monitor files~~ — **DONE 2026-11-08**, see below

**Zenodo 3403481 was downloaded and wired in.** `reference_datasets/dam_microbiome/RawData.zip`
(3.3 MB, CC-BY 4.0) contains `RawData/Figure5/Monitor*.txt`: twelve genuine raw
DAMSystem monitor files, 10,030 one-minute rows x 32 channels each, plus the
experiment's own channel map as an `.xlsx`. Suite `validate_dam_raw_microbiome`,
41/41.

**What it closed.** The >=5-min inactivity rule — the definition under which
every sleep number in this platform is computed — had no validation beyond its
own unit tests, because every sleep dataset here was already-scored sleep. It is
now compared against an implementation written from the published definition
(Hendricks 2000; Shaw 2000) that shares no code with `modules/dams.py`, on 379
real flies: **max disagreement 0.0500 min/day**, which is half of the last
decimal place the module publishes. The criterion is shown to bite (1 / 5 / 25
min give 1,252 / 1,190 / 1,036 min/day on the same flies), the comparison is
shown not to be a tie at zero sleep, and bad-status rows are confirmed to reach
the analysis as NaN rather than as stillness.

**What it also closed, unexpectedly.** The paper's result is NEGATIVE — a
*limited* influence of the microbiome on sleep — and this repository's per-fly
test contradicts it: Kruskal-Wallis **p = 0.015** on 377 flies. Aggregating to
the replicate cohort, the unit the experiment randomised, gives **p = 0.372**
(n = 19); to the monitor, **p = 0.230** (n = 16). Measured between-monitor
ICC 0.118, mean cluster 31.4, **DEFF 4.60** — 379 flies carrying the information
of 82. The published conclusion is recovered only when clustering is respected,
which is principle 5 on real data with a known right answer.

**And one defect it exposed.** `design_schema.check_confounding` reported
`severity = 0.000` and the verdict "effects are separable" while returning
Cramer's V = 0.91 in the same call, because severity measures only *complete*
confounding — the fraction of factor levels confined to one nuisance level — and
each group here spans four to six monitors. Nine of the twelve monitors in fact
hold a single group. Fixed by adding `cluster_confounded`, `nuisance_purity` and
`n_nuisance_levels_single_factor`, and by rewriting that verdict to name the
cost and the unit of analysis. `confounded`, `severity` and `blocking` are
unchanged, so no existing suite's meaning moved.

**What remains open.** No annotated ground truth for sleep exists in any of
this, so the rule is validated and the *measurement* is not. Zenodo 5109970
(48.9 GB ethoscope) is still the only candidate for that and is still too large
to pull here.

### 2b (original). Raw DAM monitor files

**Sensory processing during sleep — ethoscope data**, [Zenodo 5109970](https://zenodo.org/records/5109970) — 48.9 GB. Too large to pull here; a single monitor file would be enough. This is the dataset that could validate the 5-min rule itself.
**The Drosophila microbiome has a limited influence on sleep**, *Sci Rep* 2018, [doi:10.1038/s41598-018-28764-5](https://doi.org/10.1038/s41598-018-28764-5) · [Zenodo 3403481](https://zenodo.org/records/3403481) — 3.3 MB `RawData.zip`, likely DAM-format. **Downloaded but not yet opened.**

### ~~2c. Larval locomotion~~ — **DONE**, see §1

**The speed–curvature power law in *Drosophila* larval locomotion**, *Biol Lett* 2016, [doi:10.1098/rsbl.2016.0597](https://doi.org/10.1098/rsbl.2016.0597) · data [doi:10.5061/dryad.37sb7](https://doi.org/10.5061/dryad.37sb7) · [Zenodo 5010896](https://zenodo.org/records/5010896)
Local: `reference_datasets/larval_powerlaw/*.mat` (41 larvae per condition)

**Completed 2026-09-08.** `larval.speed_curvature_exponent()` implements the published method and reproduces every reported value — see §1 and `validate_larval_powerlaw`.

### ~~2d. Grooming~~ — **DONE**, see §1

**Completed 2026-09-08.** Both layouts validated. One thing to flag to the data's authors:

**A genotype-coding contradiction in the source data.** The two sheets annotate opposite codings —
`Fig_1C`: "genotype 1 = wCS10, genotype 2 = Nf1P1"; `Fig_1F`: "Genotype 1 = nf1P1, Genotype 2 =
wCS10". Under Fig_1F's own annotation, nf1P1 grooms **less** than wCS10 in every segment and
essentially every body part (per-fly totals 229/125/119 vs 350/462/445), which is the opposite of
the direction the paper reports for that figure. The file's two genotype columns (`genotype` and
`genotype_wcs10First`) are exact relabellings of each other, so the ambiguity cannot be settled
from the file. **I did not resolve it by choosing the mapping that agrees with the paper** — that
would make the check circular. Fig_1F is therefore used only for mechanics, and the directional
claim rests on Fig_1C alone. Worth an email to the corresponding author if you want Fig_1F's
direction usable.

---

## 3. Found but not usable

| Dataset | Why not |
|---|---|
| Vaccaro et al. startle-induced geotaxis + lifespan ([Zenodo 4997909](https://zenodo.org/records/4997909)) | Ships as proprietary GraphPad `.pzf` binaries (header `PCFFGRA4`). Not machine-readable. **If you have Prism, exporting `SING.zip` to CSV would give a second independent climbing-vs-age benchmark.** |
| Anything hosted only on Dryad | Dryad returns **401/403** to scripted download (bearer token required). The Zenodo mirrors worked. If a dataset is Dryad-only, you downloading it in a browser is the fastest path. |
| Thiem et al. 2024 escape climbing ([Zenodo 10629747](https://zenodo.org/records/10629747)) | Downloaded, but the sheets are unlabelled figure-panel exports (`Unnamed: 0…`). Would need the "Data guide.docx" read alongside. Low priority — Shahrestani already covers climbing-vs-age. |
| Y-maze individuality ([Zenodo 18342692](https://zenodo.org/records/18342692), [20339311](https://zenodo.org/records/20339311)) | 2.1 TB and 12.6 TB. Not feasible here. |

---

## 4. Still simulation-only after all this

Honest list of what no real dataset currently backs:

- `hierarchical.py`, `survivor_bias.py`, `trajectory.py`, `control_prior.py` — validated against closed forms and generative models with known truth. These are *methods* modules; a real dataset would test them only indirectly.
- `grooming.py` — calibration anchor only (2d is the fix).
- ~~`dams.py` sleep bout rule~~ — **CLOSED 2026-11-08.** Both §2a and §2b are
  now downloaded, wired in and passing: `validate_sleep_homeostasis` (32/32,
  real scored sleep, rebound direction, definition sensitivity) and
  `validate_dam_raw_microbiome` (41/41, real raw beam counts, the >=5-min rule
  itself against an independent implementation of the published definition).
  What is still simulation-only here is the sleep **measurement**, which needs
  annotated ground truth that no public dataset provides.
- `circadian_climbing.py` — `astropy` cross-check and simulation.

---

## How to fetch these yourself

Zenodo allows direct download; substitute the record id and URL-encoded filename:

```bash
curl -sL -o out.xlsx "https://zenodo.org/api/records/<ID>/files/<FILENAME>/content"
```

Dryad does not — use the Zenodo mirror if one exists, or download in a browser.
