# Climbing-video analysis — status, code, and engineering history

**Date:** 2026-09-22 · **Branch:** `fix/climbing-video-audit` · **Rig:** BOROSIL 50 ml
graduated cylinders, black band drawn at the 40 ml mark, 3 cylinders per recording.

This is a handover document. It says what the code is, where it lives, what was
tried, what the numbers were, and what is still wrong. Every figure quoted here
was measured on this project's own recordings; none is carried over from
literature or from another rig.

---

## 1. One-paragraph summary

A second scoring path (`modules/climbing_band.py`) was built for the graduated-
cylinder rig after the existing pipeline was found placing the Day-3 scoring line
on the cotton plug while reporting itself confident. The new path finds the drawn
band by row-darkness coverage, counts flies above it with a size-matched blob
detector, and reports a bracket rather than a point estimate. **Against 480
operator hand counts it reaches MAE 4.5 flies with a 3-fly human reproducibility
floor, and it compresses the dynamic range: it reads too high on near-empty tubes
and too low on full ones.** It preserves genotype ordering but not effect size, so
it is not yet usable for quantifying a drug effect. The biology in this dataset is
sound and comes from the hand counts, not the tool.

---

## 2. Where the code lives

### 2.1 Committed and standalone

| Path | Lines | What it is |
|---|---|---|
| `modules/climbing_band.py` | 741 | The estimator. No session dependencies. |
| `scripts/validate_climbing_band.py` | 62 | CLI: hand counts + tool CSV → MAE report |
| `scripts/experiments/n_from_video_mass.py` | 317 | Failed N-from-video experiment, kept as a record |
| `data/validation/hand_count_template.csv` | — | 18 development counts (Day 3/7) |
| `data/validation/hand_count_holdout.csv` | — | 30 blind hold-out counts (Day 3/7) |
| `app.py` → "Climbing (Simple)" page | ~290 | Score tab + Validate tab |

Commits on the branch:

```
0b462f0  fix(climbing): count the whole tube above the band, not a fixed 250 px
6062c84  experiment(climbing): estimate N from video by mass -- does not beat the stated loading
076f14a  feat(climbing): band-and-zone scorer, measured against hand counts and NOT yet usable
7eae413  fix(climbing-video): repair four defects found by audit, add multi-tap scoring
```

### 2.2 NOT committed — session-scratchpad only

**These will be lost when the temp directory is cleared.** Everything the analysis
actually did this month lives here:

| File | Lines | What it does |
|---|---|---|
| `hd_full_run.py` | 142 | Batch-score a folder (87 videos), automatic geometry |
| `d15_run.py` | 136 | Parse the operator's `.xlsx`, score, pair, report MAE |
| `d15_figures.py` | 244 | Figures 1–4 |
| `cross_day_conditions.py` | 93 | Figure 6, per-condition across days |
| `cross_day.py` | 108 | Figure 5 + card-sheet extraction |
| `proto_mass.py` | 300 | N-from-video prototype (superseded by the committed copy) |
| `holdout_zone_ab.py` | 32 | Zone-height A/B on the hold-out |

Three files in `scripts/` (`report_hd_day15.py`, `report_hd_cards.py`,
`report_hd_cross_day.py`) are copies of the above but **still import from the
scratchpad path** and will break.

### 2.3 Public API of `climbing_band.py`

```python
BAND_DARK = 80          # the drawn band is marker ink, nearly black
FLY_DARK  = 140         # fly bodies against backlit glass (median grey 151)
ZONE_FULL = None        # count the whole tube above the band
NOISE_FLOOR_FLIES = 3   # measured human repeat spread on this rig

find_black_band(gray, col_start, col_end, dark_thresh, min_coverage) -> int | None
count_flies_above_band(gray, background, band_y, c0, c1, zone_height_px) -> (n, centroids, crop)
count_flies_below_band(gray, background, band_y, c0, c1, zone_height_px) -> (n, centroids, crop)
persistence_filter(per_frame, bin_px, static_frac) -> [moving counts]
detect_crossings(video, band_y, c0, c1, background, ...) -> DataFrame   # REJECTED, see §5.4
analyse_video(video, c0, c1, n_flies, timepoints, ...) -> DataFrame     # the entry point
validation_report(hand_df, tool_df, day_lookup) -> str
pi(count_above, n) -> float | None                                      # None when count > n
```

---

## 3. The algorithm, and why each piece is there

### 3.1 Find the band

Scan every image row; compute the fraction of pixels darker than `BAND_DARK`
across the tube's columns. The drawn band spans the full width; graduation ticks
span a fraction; printed characters have gaps. The first row-set exceeding 60%
coverage is the band. **Return `None` if nothing reaches it** — a guessed line
silently relocates the scoring criterion, which is worse than scoring nothing.

Measured on 36 cylinders against a line verified by eye: found on 21/36, and
**81% of those within 15 px, median error 4 px**.

### 3.2 Count above the band

Crop to the tube's columns, run `fly_blob` (size-matched, fly diameter =
0.07 × tube width), keep detections above the band row.

**Detection runs on the whole tube column; the row band only selects the result.**
Cropping to the zone first sets the detector's threshold from a crop dominated by
ink, and live flies get rejected — see §5.6.

### 3.3 Report a bracket, not a point

Two counts of the same zone bound the truth from opposite sides:

- **upper** = every detection above the band. Necessarily too high: the printed
  BOROSIL block is dark, fly-sized, and above the band on this rig.
- **lower** = detections that *move* (`persistence_filter`). Necessarily too low:
  a fly clinging motionless is discarded along with the print.

The estimate is the **midpoint**; the width is the uncertainty. On the
development set: upper MAE 3.47 (bias +2.53), lower MAE 2.80 (bias −2.67),
**midpoint MAE 2.20 (bias −0.07)**. The bias signs being equal and opposite is
the evidence the errors genuinely cancel.

### 3.4 The denominator

`analyse_video` never estimates N. It takes the operator's number. The estimator
itself is denominator-free — `(lower + upper)/2` never divides by anything — so
only the PI depends on N.

---

## 4. Validation history — every number, in order

| Dataset | n counts | Cylinders | Purpose |
|---|---|---|---|
| Day 3/7 development | 18 | 6 | Method selection (6 methods compared) |
| Day 3/7 hold-out | 30 | 10 | Blind; the first honest figure |
| Day 15 | 432 | 36 | Blind; largest, with genotype labels |
| Human repeat | 12 pairs | 4 | The reproducibility floor |

**Total: 480 hand counts.**

### 4.1 The headline numbers

```
Human vs human (same instant, two counts)   MAE 2.92   bias +2.58   SD 2.39
Tool, Day-15 hold-out (432 counts)          MAE 4.53   bias -1.63   within 3: 43%
Tool, Day-15 at t=10 s (protocol endpoint)  MAE 4.15   bias +1.54
Tool, Day-3/7 hold-out (30 counts)          MAE 6.52   bias -3.18
```

**The 3-fly noise floor is measured here, not assumed.** Two independent operator
counts of the same cylinder at the same instant differ by 2.92 flies on average,
max 5. The +2.58 bias between the two sessions suggests a slightly different
convention (flies on the band, flies in the plug).

### 4.2 The overfitting lesson

The development set gave **MAE 2.20**. That figure came from comparing six
methods on 15 readings and picking the winner. On the blind hold-out the same
estimator scored **6.53** — 3× worse. The gap is the cost of selecting on your
test data. **2.20 should never be quoted; 4.5–6.5 is the honest range.**

### 4.3 The structural finding: range compression

MAE hides the real problem. On the Day-15 set the error depends on the count:

```
proportional bias: slope -0.89 flies per fly (r = -0.62)
tool reads  +3.5 too high at  3 flies
tool reads  -7.2 too low  at 15 flies
```

Two mechanisms, a **floor** and a **ceiling**:

- **Floor (~5 flies)** — printed label counted as flies. CyO cylinders holding
  0–2 flies read 3–12.
- **Ceiling (~12–15)** — at the top of a well-climbed tube flies crowd, cling and
  enter the cotton plug; the persistence filter strips resting flies as "static".
  Wildtype cylinders at 17–20 read 10–15.

**Consequence:** the tool preserves genotype ordering but not effect size. On
Day 15 it would find Wildtype > Htt > CyO and would detect the withaferin rescue
— but report it as ΔPI ≈ 0.1 where the hand counts say **0.47**.

---

## 5. What was tried and rejected — with the number that killed it

Everything here is recorded in the source so it is not re-attempted.

### 5.1 Crossing gate (per-fly tracking through the band)

Track blob centroids frame-to-frame in a ±30 px strip; count up- and
down-crossings; report the net. Theory: static print generates no events, so it is
immune to label contamination.

**MAE 10.47, bias −10.47 — and it returns NEGATIVE counts** (down to −1.60 PI on
HTT). More down-crossings than up, in an assay where every fly starts below the
band. The tracker invents events; nothing in the method forbids it. Second-worst
of six methods.

### 5.2 Complementary estimate (N − count_below)

Count the *less crowded* side. Sound in principle.

**MAE 6.07, bias +5.67.** Defeated by crowding: on impaired cohorts the below-zone
finds 1–8 flies where 13–17 are packed, because touching bodies merge into one
blob. `complementary` inherits that directly.

### 5.3 Proportional closure — force `above + below = N`

`N × above/(above+below)`. **MAE 4.66, bias +4.55** — worse than doing nothing.
The two sensors' errors are not proportional (one over-counts from print, the
other under-counts from crowding), so rescaling smears one bias into the other.

Measured: `above + below` ranged **5 to 28** against a true 20 — violated in both
directions. It survives as a *diagnostic* (`accounted_for`), never a correction.

### 5.4 N estimated from the video by mass

Per-pixel high-percentile background, additive fly mass, conservation. Six
refinement steps, each measured against 8 cylinders whose true loading the
operator counted:

```
raw percentile background              MAE 206.8
+ skip t<2s, components >= 12 px       MAE  70.0
+ static-ink mask                      MAE  63.8
+ PER-FRAME REGISTRATION               MAE  11.8   <- the big one
+ pooled fly-size across the 3 tubes   MAE   8.95
+ conservation outlier rule            MAE   9.10
"assume the stated 20"                 MAE   2.62  <- wins
```

**Rejected.** But it produced the most transferable finding in the project:
**every Day-7 recording moves 3–7 px during the take**, sometimes as a step, and
per-frame registration took MAE from 64 to 12. The production counter does not
register. See §7.

It also surfaced a data problem no per-frame method could see: **IMG_6182 left
carries a 35-second epoch (t = 60–94 s) at three times its fly mass** — an
external disturbance in front of the tube.

### 5.5 Hard-bin persistence filter

Binning detection positions to a 4 px grid splits a genuinely static detection
across two adjacent bins when it sits near a boundary, so it never reaches the
static threshold. Measured on the label-facing cylinder: 4 px bin removed 2 of 6,
8 px removed 3, 12 px removed 5, 20 px removed all 6 *including real flies*.
Replaced with **radius matching**, which has no boundaries.

### 5.6 Cropping to the zone before detection

`fly_blob` sets its threshold from the background's band-pass response. A crop
dominated by the band and the printed block gives a null quantile so high that
live flies are rejected. Measured on IMG_6029 P1 (20 flies loaded, 3 above by
hand): the below-band zone returned **zero** detections, so `complementary` read
20 − 0 = 20; the above-zone returned 1 where the full-tube detector finds 6.
**Fixed:** detect on the whole column, filter by row.

### 5.7 Band located on frame 0

At the tap the cohort is heaped at the bottom, and a raft of flies is dark right
across the tube — precisely the signature `find_black_band` searches for.

```
IMG_6029 P1   frame0 898   t10 576   t60 576   -> 322 px too low
IMG_6029 P3   frame0 891   t10 586   t60 586   -> 305 px too low
IMG_6181 P2   frame0 743   t10 630   t60 630   -> 113 px too low
```

**Fixed:** the band is located on a 15-frame temporal median.

### 5.8 One threshold for band and flies

A threshold of 80 finds the band and **1 of 18 flies**. Measured sweep on a
wild-type cylinder where the reference count is 18:

```
fly threshold   t=10  t=20  t=30  t=60  t=120
          80       1     1     1     1      2   <- as originally specified
         120      13    12     7     9      8
         140      16    15    16    18     16   <- matches the reference
         160      23    22    28     0      2   <- unstable
         180       0     0     0     0      0   <- all one blob, area-filtered
```

**Fixed:** `BAND_DARK = 80`, `FLY_DARK = 140`. 140 is a sharp optimum, so it is a
per-rig calibration, not a constant of nature.

### 5.9 Fixed 250 px counting zone

Tuned on the Day-3/7 rig where it happened to reach the plug. On the 15th-Day rig
the camera is closer and band-to-plug is ~400 px, so good climbers sat above the
zone uncounted. Measured on C0689 left (hand count 17 at 60 s): the whole-tube
detector finds 15 above the band; a 250 px zone keeps **2**, 400 px keeps 15,
600 px keeps 15.

```
15th-Day hold-out (432)   MAE 7.38 -> 4.53   bias -7.16 -> -1.97
Day-3/7 hold-out  (30)    MAE 6.53 -> 6.52   bias -3.33 -> -3.18   (no regression)
```

**Fixed:** default is the whole tube above the band.

### 5.10 Relative band threshold

Tried because the absolute threshold refuses 42% of Day-3 cylinders. Found three
more bands (24/36 vs 21/36) but accuracy fell from **81% to 58%** within 15 px.
Rejected — attack the refusal rate with a better criterion, not a looser one.

---

## 6. Defects found and fixed in the pre-existing pipeline

Found by audit before any new code was written; all committed in `7eae413`.

| # | Defect | Consequence |
|---|---|---|
| 1 | Fly-size sliders inert — wrote `min_area_frac` while the detector read `min_area_w2` | An impossible window (min = max) returned 19 detections, same as wide open. The app told users to "adjust the size limits". |
| 2 | `flies_detected` set to `flies_above` | `Detected_total` == `Count_climbed`; `Confidence` became PI capped at 1, so a genuine PI 0.5 exported as `Low_confidence = True`; over-detection invisible (59 objects with 15 above N=20 passed) |
| A | `build_layout` returned a 4-tuple on failure, 2-tuple on success | `ValueError` swallowed by the caller; the informative reason discarded on every failure |
| B | Per-vial summary indexed timepoints by **position** | 27 of 204 vials had a wrong `above_120s`; Day-7 C0148 held a **60 s** value in a column read as 120 s |

Also fixed: a stale docstring describing an abandoned estimated-denominator
scheme; output paths hardcoded to a temporary scratch directory; a fixed 1.96 CI
multiplier (now *t*, intervals widen 5.5–6.4%); a silent fallback to 20 flies.

**Day-3 geometry failure.** `cylinder_finder.locate_cylinders` put the scoring
line on the cotton-plug/R-label region (median row 402, darkness 48) instead of
the drawn band (find_line median 582, depth 128). The two locators agreed on
**8%** of Day-3 cylinders and 100% of Day-7. On the third of Day-3 videos where
it declared `ok()`, agreement fell to **4%** — confidence and correctness were
anti-correlated. Had it been scored, every Day-3 vial would have read PI ≈ 0.

---

## 7. What is still wrong

### 7.1 The denominator was never recorded

Stated as 20 per cylinder; hand counting gave **14, 18, 20, 21, 22, 22, 23, 25**
(mean 20.6, range 14–25). Every PI computed against 20 is wrong by up to ±25%
relative. Several readings previously called "impossible over-detection" (21–23
above the line) were real.

On Day 15, N is pinned by the video for 4 of 36 cylinders (21, 21, 22, 22) and
assumed 20 for the other 32. **A per-vial loading manifest is the single highest-
value missing input.**

### 7.2 Range compression (§4.3) — the main accuracy blocker

Not yet fixed. Two sources, both characterised:
- printed labels facing the camera (**bench fix: rotate the cylinders**)
- crowding-merge and the cotton plug (**genuine algorithmic limit**)

### 7.3 Camera drift is not corrected in production

3–7 px per take on every Day-7 recording. Registration was the dominant fix in
the N experiment (MAE 64 → 12) and is **not** in `climbing_band.py`.

### 7.4 The confidence flag does not discriminate

On the Day-3/7 hold-out: HIGH 6.56, MODERATE 7.67, LOW 7.25. No separation. It
should be removed or rebuilt — a meaningless label invites filtering that does
nothing.

### 7.5 Band finder refuses 42% of Day-3 cylinders

Refusing is correct, but it means Day 3 is 58% scoreable and the surviving subset
may be biased.

### 7.6 No test coverage

`scripts/` and the app layer have **no tests**. All 28 validation suites passed
before the four defects in §6 were found and after they were fixed — the suite
could not distinguish the two states. Every fix in this project was verified by
scratch scripts that are not committed.

---

## 8. Limits that no algorithm can fix

- **Cotton plug.** A fly in the plug is in no pixel. Conservation makes it visible
  as a deficit; it cannot be counted.
- **Depth occlusion.** Two flies stacked along the viewing axis share projected
  area from a single camera.
- **Cohorts that never disperse.** N from video needs a dispersed moment; a heaped
  impaired cohort gives a lower bound only.
- **A 7 px printed character and a 7 px fly**, both dark, in the same place.

### 8.1 A measured caveat about still frames

I counted four cylinders from still frames and scored **MAE 7.0–9.6 against the
operator — worse than the tool.** On IMG_6183 right at 30 and 60 s the two
operator counts agreed exactly (17/17, 18/18) and I saw 7 and 8.

**Roughly half the flies above the band on this rig are not identifiable in a
single still frame**, even by a careful reader at 3× magnification. They are
visible when the video plays, because motion reveals them. This reframes the
tool's residual error: its snapshot detector works under the same handicap.

---

## 9. The biology (from hand counts, not the tool)

Day 15, 432 counts, 3 vials per condition:

- **Genotype ordering is unambiguous:** Wildtype > Htt > CyO in all four
  conditions at all twelve timepoints.
- **Withaferin rescues Htt climbing in males, as a late effect:** PI 0.38 → 0.85
  at 60 s (ΔPI +0.47, Hedges g = 2.47, Welch p = 0.020). At the 10 s endpoint the
  effect is only +0.10. The drug restores how far they climb, not how fast they
  start — a 10-s-only protocol would have called it marginal.
- **In females the effect is present but not established:** ΔPI +0.26 at 60 s
  (g = 0.93, p = 0.27) with n = 3.

With n = 3 per cell, read the effect sizes; the p-values are indicative only.

**Days 3 and 7 have no hand counts**, so their curves are the tool's and are
readable for ordering and direction only. Day 7 male preserves the expected
ordering; Day 3 shows no structure (it is the rig with the faintest band and
highest refusal rate).

---

## 10. What would move this forward, in cost order

1. **Rotate the cylinders so printed labels face away from the camera.** One
   minute at the bench; removes the largest single error source (+7 to +13 flies
   on affected cylinders).
2. **Record actual loading per vial.** Fixes every PI.
3. **Hand-count Day 7 male, three conditions, at 10/30/60 s** — 9 videos × 3
   cylinders × 3 timepoints = 81 counts. Turns the cross-day comparison from a
   caveat into a result.
4. **Wire the workflow into the repo** (§2.2): a batch command, automatic geometry
   in the page, the spreadsheet parser and per-vial N logic in the module,
   path-independent figure scripts. ~1 day; makes this month reproducible without
   an assistant.
5. **Add per-frame registration** to the production counter (§7.3).
6. **Then** attack crowding-merge — it is the genuine algorithmic limit, and items
   1–2 change what is left of the residual.

---

## 11. Reading guide for the output files

Written to `HD Climbing assay/_DroPhenix_analysis/`:

| File | Contents |
|---|---|
| `d15_analysis.csv` | Day 15: hand count and tool side by side, per cylinder per timepoint, with `N`, `N_status`, `PI_hand`, `PI_tool` |
| `hd_all_vials.csv` | Days 3/7: every cylinder, 12 timepoints, `pct_nominal` (÷20) and `pct_known` (÷ counted N, blank where unknown) |
| `cross_day_table.csv` | PI at 10 s and 60 s per day/sex/treatment/genotype |
| `human_repeat.csv`, `three_rater.csv` | Reproducibility measurements |
| `fig1`–`fig6` | Curves, endpoints, agreement, cross-day |
| `condition_map_day3_day7.csv` | Genotype/sex/treatment read from in-frame cards, **with a confidence column — the Day-3 medium/low rows need verification** |

**Column to trust:** `reported_count` (bracket midpoint) — it is what was measured
against hand counts. `count_lower`/`count_upper` give the interval.
**Column to treat as diagnostic only:** `complementary`, `accounted_for`,
`confidence`.

---

## 12. Bottom line

The estimator is honest, auditable, and roughly **1.5× noisier than the human it
would replace**, with a known and characterised range compression. The biology in
this dataset is real and rests on the operator's 480 hand counts.

**This paragraph previously said the tool was "suitable today for screening".
Section 13 withdraws that.** On the only independent set the slope of the reported
count against the hand count is 0.003 with a 95% interval of [−0.532, +0.539] —
which does not establish screening either. The software now refuses to report a
Performance Index rather than leaving that judgement to the reader.

The engineering history above exists so that the next person does not repeat the
crossing gate, the complementary estimate, proportional closure, or N-from-video.
Each is recorded with the number that killed it.


---

## 13. What the calibration audit found (2026-09-23), and what was done about it

Section 12 used to end by recommending the tool for screening. This section is
why it no longer does.

### 13.1 MAE was hiding two large errors that cancel

Day 15, 432 readings paired with hand counts:

```
calibration   reported = 6.08 + 0.199 x true      (ideal 0 + 1.000)

FLOOR      hand <= 1:  8.0 objects detected above an EMPTY tube,
                       of which 7.2 never move. Cylinder-weighted
                       reported floor 5.62 (8 cylinders, range 3.12-10.50)
SURVIVAL   hand >= 10: cylinder-weighted 0.22 of real flies survive
                       as separate detections (20 cylinders, range 0.00-0.56)
```

The tool invents about five or six objects that are not flies and catches
roughly one real fly in five. These are large errors in opposite directions and
they cancel into an MAE of 4.5. **Every MAE quoted for this estimator is the
residue of that cancellation**, which is why the F12/F13 change moved it by 0.01
and why chasing it was never going to mean anything.

Cost to the biology: the tool retains **20-27%** of the true genotype separation
in PI (Wildtype - Htt and Wildtype - CyO, at 10 s and 60 s). Ordering survives;
magnitude does not.

### 13.2 Phenotype bias is not a separate defect

Raw error by genotype looks disqualifying (Wildtype -5.69, Htt -1.88, CyO +2.69;
ANOVA p = 3.8e-52). It vanishes entirely once the true-count trend is removed:
residual means within 0.2 flies, p = 0.67 by reading and p = 0.96 by cylinder.
**The detector is count-aware, not phenotype-aware.** There is one defect wearing
three genotype masks, and fixing the calibration fixes all three.

Two checks that had to pass before any of the above could be believed:

  - **Left/middle/right pairing.** All six permutations were scored. Identity
    wins overall (MAE 4.538 against 4.99-5.94) and the per-video winners are
    scattered across five different permutations, which is what post-hoc fitting
    to noise looks like. The mapping is sound. C0665 alone fits a rotation much
    better (5.46 -> 2.54) and deserves an eye.
  - **Are the hand counts cumulative?** `fly_blob`'s docstring records that the
    paraquat laboratory's were. If Day 15's were too, every comparison here would
    be between two different quantities. They are not: 44 of 396 consecutive
    steps decrease, across 19 of 36 cylinders.

### 13.3 The obvious fix failed confirmation, and was not adopted

The floor is 89% static, and the static detector correctly identifies it -- the
lower bound reads 0.88 on an empty tube. The midpoint estimator then adds half of
it back by construction (`mid = upper - static/2`). That looks like a free fix.
It is not:

```
Day 15 (n=432)        MAE    bias   slope      Day 3/7 (n=30)      MAE    bias   slope
  upper              4.87   +2.39   0.272        upper            7.10   -0.03  -0.190
  midpoint SHIPPED   4.54   -1.63   0.199        midpoint         6.53   -3.33   0.003
  lower              6.58   -5.64   0.125        lower            7.23   -6.63   0.196
```

**The ranking reverses between the two sets.** Upper has the best slope on Day 15
and the worst -- and negative -- on Day 3/7. Adopting the Day-15 winner would be
selecting a method on the set it was found in, which is precisely the mistake
that took a development MAE of 2.20 to a held-out 6.53.

The Day-3/7 slope CI is [-0.532, +0.539] at n = 30. It excludes a faithful
instrument but cannot rank three candidates. **The blocker is hand counts, not
code.** The estimator is therefore FROZEN -- see `climbing_band.ESTIMATOR_SELECTION`.

Nor can the static filter simply be leaned on harder. It cannot distinguish a
permanent mark from a fly that is not moving, and **reduced movement is the
phenotype**: a filter tuned to delete things that stop will preferentially delete
affected animals and flatter the disease model invisibly. The gate before any
tuning there is a labelled measurement of stationary-fly sensitivity, which has
never been made. See the protective note on `persistence_filter`.

### 13.4 The QC flags run backwards

Applying the new reason codes to the 432 Day-15 readings:

```
code                     fires on   MAE flagged   MAE clean   Welch p
BRACKET_TOO_WIDE              56%          4.66        4.39     0.382
CONSERVATION_VIOLATED         58%          4.14        5.08     0.002
STATIC_ARTIFACT_EXCESS        79%          4.38        5.12     0.034
any code at all               93%          4.42        5.97     0.012
```

A reading that raises **no** flag is **worse** than one raising several. The old
`confidence` flag is inverted the same way (HIGH -> 5.15, MODERATE -> 4.28,
LOW -> 4.66) -- worse than the audit's finding that it fails to discriminate.

The reason is structural: error on this instrument is set by where a reading sits
on a compressed calibration curve, not by any per-reading condition. Quiet
readings are sparse ones, and sparse is where the additive floor dominates.
**You cannot QC your way out of a calibration problem.**

Consequence, and it is encoded in the module: the codes that **gate** are logical
(impossible count, absent denominator, wrong tube width, unvalidated estimator)
and do not rest on this correlation. The codes that merely **describe** are
REVIEW_REQUIRED only, and must never be used to select readings, drop outliers or
weight an average -- on this evidence that would select the worse ones.
`validate_climbing_band.py` has a test that locks the distinction.

### 13.5 What was changed

Three layers, because a reading carries three kinds of statement and they were
being emitted as one number:

| layer | what | gated? |
|---|---|---|
| 1 Observations | `count_above`, `count_below`, `static_removed`, `tube_w_px`, `band_y` | never -- a refusal must not hide its own evidence |
| 2 Estimate | `reported_count`, `pi_estimate`, `n_source`, `n_quality` | never, but never presented as a result |
| 3 Result | `pi`, with `pi_verdict` and `pi_reasons` | yes |

- **`VALIDATED_DOMAIN = None`.** No conditions have been established under which
  a PI may be reported, so today every reading is `NOT_REPORTABLE`. This is not a
  placeholder; it is the honest state, and opening the gate is a deliberate edit
  made when the evidence allows.
- **The exported PI used to bypass every check in the module.** `pi()` gained its
  bounds in F3 and then turned out to have no callers: what left the module was
  an inline `mid / max(n_flies, 1)`, which cannot refuse anything and turned a
  zero denominator into a division by one -- so a cylinder with no flies recorded
  produced a PI equal to the raw count. It now goes through `assess_reading`.
- **The denominator's provenance travels with the result** (`MANUAL | VIDEO |
  NOMINAL | UNKNOWN` x `VERIFIED | ESTIMATED | UNCERTAIN`). `UNKNOWN` cannot
  produce a PI at all; `NOMINAL` is REVIEW_REQUIRED, because on this dataset the
  loading stated as 20 was 14-25 in the operator's notes.
- **`validation/validate_climbing_band.py` now exists** -- 91 checks. The module
  had none, and previous reports of "28 suites passed" meant the *other* modules
  still passed. Nothing was checking this one.
- **The page no longer advertises MAE 2.20.** That was measured on the 15 readings
  the midpoint was *chosen* on. It now states the held-out 4.54 and 6.53, the
  floor, the survival fraction and the slope.

### 13.6 The one thing that would move this

Hand counts on Day 3/7, distributed across videos and cylinders rather than piled
into one recording -- the split that matters is at the **video/vial** level, since
timepoints within a cylinder are not independent. Target: Day 7 male, 3
conditions x 10/30/60 s (~81 counts), plus the 9 blank Set D rows in
`hand_count_holdout.csv`. That takes the confirmation set from 30 to ~120 and the
slope interval from about +/-0.53 to about +/-0.25 -- enough to rank the three
estimators and, if one holds, to open `VALIDATED_DOMAIN`.

The Set D blanks are worth keeping separate: they answer a different question
(*what does the instrument report when there are no flies?*), which is the
specificity test the floor measurement above is a proxy for.
