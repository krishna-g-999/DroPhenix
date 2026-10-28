# Phase 0 audit — `modules/climbing_video.py`

**Date:** 2026-08-08 · **Module:** `modules/climbing_video.py` (1,851 lines) ·
**Suite:** `validation/validate_climbing_video.py` (686 lines) ·
**Footage:** 15 recordings, `D:/DroPhenix/Testing data/Climbing assay/Drosophila _ climbing assay/`

Every number below was produced by a command run during this audit. Where a thing could not be
measured on the available data, it is listed as unmeasured rather than estimated. Nothing here is
a rewrite; this is inspection and measurement only.

---

## 1. What the pipeline currently does

Read from source, stage by stage.

**Background.** `build_background` samples 40 frames from `start_s=4.0` to the end and takes the
per-pixel **temporal median**. The median (not mean) is the right call: a fly that rests for a
quarter of the clip leaves no trace in a median but a permanent smudge in a mean. Sampling after
4 s excludes the tap and the operator's hands. The organising insight of the module is here — the
printed graduation scale is *static* and flies *move*, so subtracting a temporal median removes
ticks, numerals and label text in one step, which no intensity threshold can do.

**Registration.** `align_to_background` estimates a sub-pixel translation by `cv2.phaseCorrelate`
and removes it with an affine warp. Rotation and scale are not corrected. Displacements above
25 px are treated as a wild estimate and left uncorrected.

**Tap.** `detect_taps` downsamples to 240×135, takes mean absolute frame difference over 900
samples, and calls a burst a tap when motion exceeds a prominence criterion. `t0` is the **end** of
the first burst. `detect_framing_change` separately looks for mid-clip camera moves.

**Vial geometry.** `_activity_map` accumulates, per pixel, the number of frames differing from the
background by more than 25 grey levels. Column-summing gives a horizontal profile;
`_bands_from_profile` finds bands, merges fragments whose gap is under 40% of the median inter-band
gap, keeps the strongest `n_vials`, and re-derives each band's edges against its own peak.
Then — and this matters — **every band is replaced by a common width** (`common_w`, the median band
width + 12% padding, capped at 95% of the smallest centre-to-centre gap) centred on its own
centroid. Vertically, `_tube_extent` returns the span in which flies were *seen*, and a
**common height** (`common_h`, the median) is imposed upward from each vial's own base.
`_find_line` locates the darkest horizontal row in the middle 64% of each band.

**Detection and counting.** `count_in_vial` subtracts frame from background, thresholds at
`diff_threshold=26`, morphological open(2) then close(3), connected components, keeps blobs by area
(`min_area=4`, `max_area=260`, autoscaled by the square of apparent tube width) and aspect
(`max_aspect=6`). A blob larger than `1.6 ×` the frame's median blob area is split by area into
`round(area/typical)` flies. Blob centroid above `line_y` ⇒ climbed.

**Scoring.** `fly_heights` returns every fly's height above the region base — the good design
decision in the module, because it lets one pass over the video be re-scored under any criterion.
`score_heights` applies a `ScoringRule` (`drawn_line` / `height_fraction` / `height_mm`) and
**refuses** (returns `NaN`, not 0) when the criterion falls outside the measured region.

**Orchestration.** `analyze_climbing_video` counts at each scoring time over `average_frames=5`
consecutive frames and takes the median; then runs a stability pass of 25 samples between `t0+3` and
`t0+115` and reports, per vial, the median/min/max/CV of the detected total plus a
`_reliability_grade` string. `to_drophenix_table` emits `Genotype, Treatment, Sex, VialID,
Replicate, Time, Count_climbed, n_total_flies` and `analyze_batch` appends `DetectorReliability`.

### What it does *not* do

| Capability the brief requires | Present? |
|---|---|
| Per-vial numeric count uncertainty | **No** — a string grade only |
| Perspective / true vertical axis recovery | **No** |
| Frame-rate adequacy gate | **No** |
| Resolution gate | **No** |
| Camera-angle / tilt gate | **No** |
| Glare detection | **No** |
| Explicit occlusion model | **No** — area-splitting only |
| Climbing velocity (FreeClimber-style) | **No** — 0 occurrences of `velocity`/`slope`/`regression` |
| Refusal on unscoreable criterion | **Yes** |
| Refusal on camera movement mid-clip | **Yes** |
| Refusal on implausible count (>60) | **Yes** |

---

## 2. What happens when it is run

### 2.1 The footage is homogeneous — so several failure modes are untestable here

```
video           w x h      fps  frames  dur_s  codec
IMG_6030.MOV  1920x1080  29.97   3641   121.5   hevc
... (all 15 identical to within 40 frames) ...
```

All 15 are 1920×1080, 29.97 fps, ~121 s, HEVC. **Variable/low frame rate, low resolution and
rolling-shutter artefacts cannot be assessed on this dataset at all.** They must be simulated.

### 2.2 Overall yield: 18% of vials are "good"

From `batch15_qc.csv` (45 vials, 15 videos):

| reliability grade | vials |
|---|---|
| good | 8 |
| acceptable — supply the true total | 13 |
| poor — check the annotated frame | 8 |
| unusable — region or detection is wrong | 11 |
| unusable (no flies detected) | 5 |

Coefficient of variation of the detected total, over a quantity that is **physically constant**:
median **29%**, 75th percentile **124%**, maximum **2005%**.

The exported table carries 143 rows / 24 vials from **9 of 15 videos**; 36 of those rows are graded
"poor" and exported anyway (correctly labelled, but exported).

### 2.3 The measurement handed downstream has no uncertainty, and a denominator that moves

Numeric columns in `batch15_climbing_table.csv`: `Replicate, Time, Count_climbed, n_total_flies`.
There is **no uncertainty column of any kind**. The only quality signal is
`DetectorReliability`, a string, constant across all of a vial's timepoints.

Worse, no operator totals were supplied, so `n_total_flies` is the *detected* total at each
timepoint:

```
vials whose denominator changes across timepoints: 24 of 24
within-vial CV of n: median 14.8%, max 46.3%
largest within-vial swing: 22 flies

IMG_6042_R2  t= 5s  k=16  n=39      <- the number of flies in a sealed
             t=10s  k=17  n=18         cylinder cannot change
             t=15s  k=27  n=29
             t=20s  k=17  n=18
             t=30s  k=16  n=17
             t=60s  k=14  n=17
```

This violates the binomial model the downstream clustered analysis assumes: `n` is supposed to be a
known constant, and here it is a noisy measurement correlated with the numerator. A beta-binomial
or GEE fit will absorb this detector noise as **biological overdispersion**, inflating the estimated
ICC and hence the design effect — corrupting exactly the quantity DroPhenix exists to get right.
(The tool's own `estimate_icc` could not even be called on this table without an API mismatch, which
is a separate integration gap.)

### 2.4 The detection confidence score is 35% constant and blind to vertical geometry

```
conf = 0.35·(1 − width_cv/0.30) + 0.30·(1 − gap_cv/0.30) + 0.35·line_score
```

`detect_vials` assigns every cylinder the same `common_w`. Measured across all 15 recordings:

```
videos where all 3 vials share one width: 15 of 15
maximum within-video width CV:            0.0000
```

So `width_cv ≡ 0` **by construction** and the first term contributes a constant **0.35** whatever the
detection did. That is 35% of the score measuring nothing. It also means the module has thrown away
the one signal that would reveal camera perspective — under perspective the near cylinder is wider
on screen than the far one — while still deriving `px_per_mm` from that equalised width and applying
it to *vertical* distances.

Nothing in `conf` examines vertical geometry. Cylinders in one frame stand on one bench, so their
bases should agree; the line is drawn at one height, so `line_fraction` should agree. Measured:

| video | conf | base spread (px) | line-fraction spread | line outside region | needs_review | rows exported |
|---|---|---|---|---|---|---|
| IMG_6064 | 0.996 | 64 | 0.07 | no | **False** | 18 |
| IMG_6042 | 0.986 | 199 | 0.42 | no | **False** | 12 |
| IMG_6058 | **0.968** | **649** | **1.34** | **yes** | **False** | **11** |
| IMG_6046 | 0.953 | 270 | 0.54 | no | **False** | 18 |
| IMG_6034 | 0.892 | 451 | 6.22 | **yes** | **False** | 0 |
| IMG_6065 | 0.700 | 350 | 0.79 | **yes** | True | 12 |
| IMG_6063 | 0.467 | 140 | 0.58 | **yes** | True | 0 |

- **4 of 15 recordings have the scoring line physically outside the detected region.** Two of the
  four are not flagged.
- **8 of 15 have cylinder bases differing by more than 40 px** on a level bench.
- **IMG_6058 is the worst case**: bases 649 px apart, line outside the region, `line_fraction`
  running from −0.36 to +0.98 — reported at **confidence 0.968, `needs_review = False`**, and it
  contributed 11 rows to the statistics table.

**The check that would have caught this already exists in the code and is discarded.**
`climbing_video.py:704` computes `height_spread` and never uses it again.

Consequence for the exported dataset:

```
rows exported from videos with demonstrably broken geometry: 71 of 143 (50%)
vials exported from those videos:                            12 of 24
their DetectorReliability grades: acceptable 29, poor 24, good 18
```

Half the dataset, and 18 rows of it are graded **"good"**.

### 2.5 The height origin is estimated from the flies, so it is confounded with the phenotype

`_tube_extent` defines the region as *where flies were seen*, not where the glassware is. Two
consequences follow, and both are systematic rather than random:

1. The **zero point** of every height measurement is the lowest row any fly reached. A cohort that
   piles at the base sets a lower origin than one that does not.
2. The **region height** is the best climber's reach. A poorly climbing cohort gets a shorter tube
   and therefore a lower bar under a `height_fraction` rule. The module already knows this — the
   comment at line ~693 says so explicitly and imposes `common_h` as a partial fix — but the fix is
   within-video only. Across videos the detected region ran from **74 px** (IMG_6034) to **1079 px**
   (IMG_6032) for the same glassware.

IMG_6034 is the clean demonstration: region 74 px ≈ 19 mm tall, an 80 mm criterion far outside it,
every vial correctly refused as unscoreable — but at **confidence 0.892**, which would not prompt a
user to look.

### 2.6 The scoring clock is unvalidated, and at 5 s the answer doubles per second of error

Detected `t0` across the recordings ranges 0.27–1.20 s, suspiciously close to file start: the
operator is already tapping as recording begins, so the burst found may be the tail of the tap, the
hands withdrawing, or the phone being set down. **There is no ground truth for `t0` in this footage.**

Measured on the cleanest recording (IMG_6030, confidence 0.99, bases within 4 px), proportion
climbed averaged over the three cylinders:

```
t0 error (s)   t=5s    t=10s   t=20s
  -2.0         0.019   0.065   0.169
  -1.0         0.020   0.083   0.191
   0.0         0.022   0.096   0.198
  +1.0         0.067   0.110   0.230
  +2.0         0.074   0.137   0.233

sensitivity per second:  t=5s +0.024 (107% of the value at t0)
                         t=10s +0.013 (14%)
                         t=20s +0.019 (10%)
```

**At the 5 s scoring time, one second of tap-timing error changes the reported proportion by more
than 100% of its own value**, and a ±2 s window spans 0.019–0.074, a ~4× swing. No uncertainty for
`t0` is reported anywhere.

### 2.7 What the existing validation suite does and does not establish

The suite **passes in full** (run this session). It is honest in its docstring about not claiming
accuracy on real footage. Two qualifications:

- Section A is headed "Exact counts … with known ground truth", but the assertion is
  `abs(detected − truth) <= max(2, 0.15·n_true)` — **±15% or ±2 flies**, not exact. The heading
  overclaims relative to the test.
- The registration ablation reproduces cleanly and is the module's strongest result:
  `without registration [217, 170, 208, 374, 15, 18, 19, 15, 15, 13] (CV 114%)`,
  `with registration [17, 14, 17, 20, 15, 18, 19, 15, 15, 15] (CV 12%)`.
- The synthetic renderer models: static graduation scale, 1.2 px settling drift, Gaussian noise,
  random-walk motion, fly size scaled to tube width. It does **not** model: perspective/tilt,
  motion blur, glare, depth occlusion, variable frame rate, background heterogeneity, or flies
  piled on a reflective base — i.e. none of the phone-video conditions the brief targets.

---

## 3. Against FreeClimber

Grounded in the FreeClimber README (fetched this session) and the published description; where I
could not retrieve a figure I say so rather than supplying one.

| | FreeClimber (Spierer et al. 2021) | DroPhenix `climbing_video` |
|---|---|---|
| Background | static background subtraction | temporal **median** over 40 samples after the tap |
| Detection | `trackpy` particle location, filtered on eccentricity, circularity, spot mass, spot signal | threshold + morphology + connected components, filtered on area and aspect |
| Identity/tracking | per-frame spots, binned by vial | per-frame only, **no tracking** |
| Primary metric | **climbing velocity** = slope of the most linear window of mean vertical position vs time | **count above a criterion**, plus mean height |
| Vial layout | frame divided into vertical bins, white dividers | bands found from an activity profile, equalised width |
| Registration | not described | **sub-pixel phase correlation** |
| Parameters | user-tuned per video via GUI or `.cfg` | fixed defaults, autoscaled by apparent tube width |
| Refusal | none described | criterion-outside-region, camera-moved, implausible count, no-flies |
| Uncertainty | regression quality score per vial in `.slopes.csv` | reliability grade (string) |
| Rig | standardised: fixed drop, backlighting, synced capture, Raspberry Pi h264 | uncontrolled phone video |
| Stated limits | not callable as a module; tested only on macOS; quality-sensitive | see this document |

I could **not** retrieve a published quantitative agreement-with-manual figure for FreeClimber from
the sources available to me. Any comparison claim must therefore be made on shared data we run
ourselves, not cited.

### Adopt from FreeClimber

1. **Local-linear-regression velocity on mean vertical position.** It uses every fly and every
   frame rather than collapsing to a threshold crossing, it is insensitive to where anyone drew a
   line, and it is the field's comparable number. Its regression R² is also a *free, honest
   per-vial quality statistic* — exactly the kind of thing this module currently lacks.
2. **Richer per-blob shape descriptors** (eccentricity, circularity, integrated intensity) rather
   than area + aspect alone. These separate a fly from a tick fragment or a glare edge far better
   than area does, and they are cheap.
3. **Publishing the intermediate detections** (a `raw`/`filtered` spot table). It makes the
   detection auditable and lets a human re-score without re-running the video.

### Where FreeClimber wins today, honestly

- Velocity is a better-conditioned statistic than a single threshold count, and it does not depend
  on line placement — which is one of our largest error sources.
- Per-video parameter tuning through a GUI, tedious though it is, will beat fixed defaults on
  awkward footage. We have no tuning path at all.

### What DroPhenix must add that FreeClimber does not have

Perspective correction and a real-scale vertical axis; a propagated per-vial count uncertainty;
acquisition-quality gates (frame rate, resolution, angle, glare) with explicit refusal; and
cross-vial geometric consistency checks. This is the uncontrolled-phone-video gap, and it is where
the module should compete.

---

## 4. Prioritised upgrade plan

Ordered by measured harm per unit of work. P1 items all correspond to defects demonstrated above.

**P1-a — Emit a count uncertainty and stop moving the denominator.** (§2.3)
Per vial per timepoint, produce `Count_climbed`, `n_total_flies`, and an interval or variance for
each, from: frame-to-frame count dispersion at that timepoint, merged-blob share, and the
region/line geometry error. Fix `n` per vial for the whole clip (operator total where given,
otherwise a single robust estimate over the clip), never per timepoint. Verify the numbers reach
`hierarchical` in a form it consumes, and show that ignoring them changes the ICC.

**P1-b — Cross-vial geometric consistency check, and put vertical geometry into the confidence.**
(§2.4) Use the already-computed `height_spread`; add base-row agreement, `line_fraction` agreement,
and a hard invariant that `y_top ≤ line_y ≤ y_bottom`. Any violation refuses that vial. On this
dataset that reclassifies 4 recordings immediately and would have withheld 71 of 143 exported rows
pending review. Replace the constant `width_cv` term with something that measures.

**P1-c — Anchor the vertical axis to the glassware, not to the flies.** (§2.5) Detect the tube
base and top from static edges in the background image (the median frame contains the empty
cylinder), so the origin no longer depends on how well the cohort climbed. Fall back to the current
behaviour only when edge detection fails, and say so.

**P2-a — Perspective and true vertical.** Recover a homography per cylinder from its own silhouette
(two near-vertical glass edges plus base ellipse) or from the printed graduations, giving a real
vertical axis and a vertical `px_per_mm` distinct from the horizontal one. Report the residual tilt
as a quality term; refuse beyond a threshold to be determined by simulation. This also restores the
per-vial width signal that `common_w` currently destroys — keep the equalised width for *region
placement*, but keep the raw widths for *geometry inference*.

**P2-b — `t0` uncertainty, and a `t0`-free primary metric.** (§2.6) Report a `t0` interval from the
burst profile and propagate it; prefer scoring windows (e.g. mean over 4–6 s) to instants; and adopt
the FreeClimber velocity, which is far less sensitive to the exact origin than a 5 s count.

**P2-c — Acquisition gates with explicit refusal.** Minimum frame rate, minimum flies-per-frame
pixel area, maximum tilt, glare fraction, motion-blur estimate. **Thresholds must come from the
simulation in Phase 3, not from judgement** — I will not put numbers in the code that I have not
measured.

**P3-a — Better blob descriptors and an occlusion model** (adopt from FreeClimber), replacing
area-ratio splitting with something that reports how much of the count rests on unresolved merges.

**P3-b — Velocity as a secondary metric**, with its regression R² as a quality statistic.

**P3-c — Degraded-footage simulator** covering tilt, blur, shake, glare, low fps and depth
occlusion, with known ground truth — the instrument that makes every threshold above defensible.

### Open questions for you before Phase 1

1. **Human counts.** The single highest-value missing thing is a human count of a few frames from
   these videos. Without it, accuracy on real footage remains unestablished — stability is not
   accuracy, and a detector that misses the same three flies every frame is perfectly stable.
   Roughly one hour of your time would anchor the whole validation.
2. **Operator totals.** Were the flies-per-cylinder counts recorded for these 15 recordings? If so
   the denominator problem largely disappears for this dataset.
3. **Scope of P2-a.** Full per-cylinder homography is the largest single piece of work here. Is
   perspective correction worth it for your own recordings, or is the practical answer a stricter
   acquisition protocol plus refusal?
