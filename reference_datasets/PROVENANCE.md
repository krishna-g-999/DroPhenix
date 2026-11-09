# Reference-dataset provenance

REAL = verified raw data or exact published parameters. simulated = generated from text-verified published parameters (authorised fallback), cited below.

| Module | Dataset | Type | Citation | Calibration parameters |
|---|---|---|---|---|
| Climbing | sample1_PD_aSyn | simulated | Feany & Bender 2000 Nature 404:394 (a-syn PD model: progressive climbing loss) | endpoint pass fraction WT=0.88, disease=0.4, drug=0.66 |
| Climbing | sample2_ALS_TDP43 | simulated | Gargano et al. 2005 Exp Gerontol 40:386 (RING; locomotor decline) | endpoint pass fraction WT=0.88, disease=0.3, drug=0.58 |
| Climbing | sample3_HD_Q93 | simulated | Drosophila HD polyQ models: climbing impairment rescued by treatment | endpoint pass fraction WT=0.88, disease=0.35, drug=0.55 |
| Survival | sample1_Freireich1963 | REAL raw data | Freireich EJ et al. 1963 Blood 21:699-716 | verified raw event-level data |
| Survival | sample2_fly_lifespan | simulated | Linford NJ et al. 2013 J Vis Exp e50068 | Gompertz; target medians WT=50, disease=30, drug=42 d |
| Survival | sample3_fly_lifespan | simulated | Linford NJ et al. 2013 J Vis Exp e50068 | Gompertz; target medians WT=52, disease=34, drug=45 d |
| Y-maze | sample1_amnesiac | simulated | Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700 | chance 50%; turn bias Beta(4.6,4.6); WT rho=0.34, disease rho=0.02 |
| Y-maze | sample2_rut2080 | simulated | Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700 | chance 50%; turn bias Beta(4.6,4.6); WT rho=0.36, disease rho=0.05 |
| Y-maze | sample3_dnc1 | simulated | Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700 | chance 50%; turn bias Beta(4.6,4.6); WT rho=0.3, disease rho=0.0 |
| DAMS | sample1_period_mutants | REAL published periods | Konopka RJ & Benzer S 1971 PNAS 68:2112-2116 | activity simulated with the published free-running periods 24/19/28 h |
| DAMS | sample2_sleep | simulated | Hendricks JC 2000 Neuron 25:129; Shaw PJ 2000 Science 287:1834 | 5-min inactivity sleep rule |
| DAMS | sample3_sleep | simulated | Hendricks JC 2000 Neuron 25:129; Shaw PJ 2000 Science 287:1834 | 5-min inactivity sleep rule |
| Larval | sample1_ALS_TDP43 | simulated | Gargano et al. 2005 (locomotion); larval crawling literature | WT mean speed ~1.0 mm/s |
| Larval | sample2_PD_Pink1 | simulated | Gargano et al. 2005 (locomotion); larval crawling literature | WT mean speed ~1.0 mm/s |
| Larval | sample3_HD_Q93 | simulated | Gargano et al. 2005 (locomotion); larval crawling literature | WT mean speed ~1.0 mm/s |
| Grooming | sample1_dCREB2 | simulated | Qiao B et al. 2018 eLife 7:e34497 | WT ~13% time grooming |
| Grooming | sample2_FoxP | simulated | Qiao B et al. 2018 eLife 7:e34497 | WT ~13% time grooming |
| Grooming | sample3_Cul3 | simulated | Qiao B et al. 2018 eLife 7:e34497 | WT ~13% time grooming |

## Added 2026-11-08 — real raw and real scored sleep recordings

Both entries below are REAL published data, not simulated from parameters, and
both close a gap that `docs/SCIENTIFIC_VALIDATION_PLAN.md` named explicitly.

| Module | Dataset | Type | Citation | What it establishes |
|---|---|---|---|---|
| `dams.py` — the **>=5-min bout rule itself** | `dam_microbiome/RawData.zip` → `RawData/Figure5/Monitor{26,27,36,37,46,47,81,82,83,84,85,86}.txt` | **REAL raw beam counts** (CC-BY 4.0) | Selkrig J, Mohammad F, Ng SH, et al. (2018) *Sci Rep* 8:10646, doi:10.1038/s41598-018-28764-5; data doi:10.5281/zenodo.3403481 | Twelve raw DAMSystem monitors, 10,030 one-minute rows each, 32 channels, 14–21 Sep 2017, 12:12 LD. The dataset's own channel map assigns 379 of the 384 channels to germ-free, conventionalised or conventionally reared. Suite `validate_sleep_homeostasis`'s sibling `validate_dam_raw_microbiome`: the rule reproduces an independent implementation of the published definition on all 379 flies to within the module's own 1-decimal rounding (max 0.0500 min/day). The paper's conclusion is **negative** — a limited influence of the microbiome on sleep — which makes it a benchmark for *not* manufacturing an effect. |
| `dams.py` / `dam_reader.py` — per-day summary, rebound direction, definition sensitivity | `sleep_homeostasis/dataset.zip` (35 `.txt`) | **REAL scored sleep** (CC0) | Jagannath A, Sengupta T, et al. (2024) *eLife* 13:RP91355, doi:10.7554/eLife.91355; data doi:10.5061/dryad.qnk98sfpp, Zenodo 10069769 | 33 sleep files. Figure 1: DAM-style, 6 h / 12 h (60-min bins) and 24 h (30-min bins) deprivation, 3 days. Figures 2–5: ethoscope, the **same flies scored under both a 5-min and a 25-min sleep definition** — six paired files, 0 violating cells out of ~35,000, the 25-min criterion recovering 36–48% of 5-min-rule sleep. Suite `validate_sleep_homeostasis`. |

**Two corrections these datasets forced, recorded so the record is not read as it was written.**

1. `docs/REFERENCE_DATASETS.md` §2a cautioned that the eLife dataset is "not an
   absolute sleep-amount anchor" at 1,083–1,297 min/day. That is true of the
   **Figure 1** files (pooled median 1,145.0 min/day over 331 control flies,
   48.6% above the repository's 1,150 ceiling) and **false of the ethoscope
   files**, where all 128 undisturbed flies at the 5-min criterion sit inside
   (300, 1150) with a pooled median of 776.3. The ethoscope files are a usable
   absolute anchor; the Figure 1 files are not.
2. The ceiling itself is **method-dependent, and 1,150 is not an upper bound on
   published values**. Same criterion, same species: single-beam DAM
   (Selkrig 2018) gives a median of 1,188 min/day with healthy activity and
   almost nothing lost to death truncation, while position tracking
   (eLife 2024 ethoscope) gives 730–847. A fly awake but not crossing the beam
   scores as asleep. The flag text in `dam_reader.py` was corrected to stop
   claiming the value exceeds every published figure; the numeric bound is
   **not** changed, because that is a calibration decision for the owner.

## Added 2026-11-09 — FreeClimber's own example, for a detector comparison

| Module | Dataset | Type | Citation | What it establishes |
|---|---|---|---|---|
| `freeclimber_detector.py` — the reimplementation of FreeClimber's DETECTION | `freeclimber/w1118_m_2_1.h264` (+ `.mp4`, `example.cfg`, `raw.csv`, `filtered.csv`, `slopes.csv`) | **REAL example recording and its published outputs** (MIT) | Spierer AN, Yoon D, Zhu C-T, Rand DM (2021) *J Exp Biol* 224:jeb229377, doi:10.1242/jeb.229377; https://github.com/adamspierer/FreeClimber | A reimplementation used as a reference has to be shown faithful BEFORE it judges anything, and the published program cannot be run here (it pins `trackpy==0.4.2`, which does not build on Python 3.13, and its own `tp.batch(invert=True)` call was removed after 0.4.x). These files let the reimplementation be checked against FreeClimber's own answer: 4,273 located spots against 4,289, 4,091 survivors against 4,093, per-vial totals 1343/1364/1384 against 1344/1364/1385, max per-frame difference 1 spot. Suite `validate_freeclimber_detection`. |

The MIT licence requires its copyright and permission notice to travel with the
files, so both are reproduced in `freeclimber/NOTICE.md` alongside them.
