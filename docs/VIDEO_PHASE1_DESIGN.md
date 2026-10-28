# Phase 1 design — measurement-quality-first climbing video

**Status: PROPOSAL. Nothing here is implemented.** Awaiting approval.
Grounded in `docs/VIDEO_AUDIT_2026-08.md` (Phase 0) and in the paraquat ground truth
(1,080 hand-scored cells, `Testing data/Climbing assay/Paraquat climbing assay 010626_new set/`).

Two rules govern every choice below.

1. **No threshold in the shipped code that I have not measured.** Where a number is needed
   and not yet measured, this document says `TBD-sim` and names the experiment that will fix it.
2. **Nothing leaves this module without an uncertainty.** The output is not a count; it is a
   count, an interval, and a verdict.

---

## 0. What the ground truth now permits

| | |
|---|---|
| paired comparisons | **1,080** (6 conditions × 3 replicates × 5 taps × 12 times) |
| for the four headline conditions | 720 cells, 12,480 fly-observations |
| known per cell | flies above the 40 mL line **and** flies in the vial |
| mapping | video *n* (1–5) = tap *n*, replicates 1 (left) + 2 (right); videos 6–10 = replicate 3 |
| **the benchmark that matters** | tap-to-tap spread of a *perfect* counter: **median 3 flies**, max 6 |
| clustering to reproduce | ICC of taps within a vial **0.573**, design effect **3.29** |

The last two lines are the point. A method is not "accurate" because its error is small in the
abstract; it is accurate when its error is small **relative to the irreducible variation of the
assay itself**, and when the clustering it reports matches what the animals actually do.

---

## 0b. THE BASELINE — measured 2026-08-11, and worse than expected

The current module, run on the paraquat recordings and scored against the hand counts.
Vial count supplied from the manifest (generous), module's own detected tap and drawn line.
**360 paired comparisons over 20 recordings.**

| | all cells | module ACCEPTED (conf > 0) | module REFUSED (conf 0.00) |
|---|---|---|---|
| recordings | 20 | 10 | 10 |
| bias (video − hand) | −6.59 flies | −3.36 | −9.07 |
| mean absolute error | 9.75 | 7.28 | 11.64 |
| within the 3-fly floor | 13.6% | 25.0% | 4.9% |
| correlation with truth | 0.203 | 0.040 | −0.075 |

**The refusal machinery is doing real work** — it flagged exactly the half of the recordings where
error is worst — but it is not sufficient, because the accepted half is still wrong.

### The finding that reframes the project

From 40 s on, once climbing has plateaued, on ACCEPTED recordings only:

```
truth proportion : mean 0.860, SD 0.177, range 0.40-1.00   <- real spread, not range restriction
video proportion : mean 0.626, SD 0.348
r (proportions)  : -0.279
```

and at the level that matters:

| | truth | video |
|---|---|---|
| TDP-43, 10 mM paraquat | **0.422** | 0.806 |
| WT, untreated | **0.927** | 0.676 |

**The module ranks the conditions backwards.** It reports paraquat-poisoned TDP-43 flies as
climbing *better* than untreated wild-type. Across the four vials with accepted recordings the
vial-level correlation is **−0.812**.

Caveat, stated because it matters: only four vials survived acceptance, and TDP 10 mM contributed
one of them, so −0.812 is not a stable estimate. The **condition-level inversion is large and not
marginal**, and that is what the claim rests on.

### Three mechanisms tested; two rejected

I proposed that settled flies were being absorbed into the temporal median, and that climbers were
sitting above the detected region top. **Both were tested on the cleanest recording and both are
wrong:**

- rebuilding the background from the first 12 s, first 20 s, or last 20 s changed the count by at
  most 1 (15, 15, 15, 14) — no absorption;
- opening the region from `y_top = 220` up to `y_top = 5` changed nothing, and zero blobs lay above
  the detected top — nothing is being clipped.

Visual inspection of the mask above the scoring line shows ~14 well-separated blobs for ~14 visible
flies, so per-visible-fly detection is working. The hand count for that vial is 18–19. The residual
is therefore **flies that are not visible in a single 2D view** — behind the printed label, on the
far side of the glass, or lost in the wall reflection.

That has a direct consequence for this design: **the dominant error at plateau is not detector
quality, it is occlusion**, and no improvement to thresholding or blob descriptors will recover a
fly the camera cannot see. §3 and §4 are re-prioritised accordingly — the visible-fraction model in
§4 moves from a refinement to the core of the work, and the early-time overcount (pooled
video/truth **1.169** at 10–20 s, with video proportions exceeding 1.0) is a separate defect in the
merged-blob splitting that must be fixed independently.

---

## 1. Acquisition robustness

**Keep, unchanged:** temporal-median background and sub-pixel phase-correlation registration.
Measured, twice, on real footage: detected total CV **114% → 12%**. This is the module's one
proven claim and nothing in the redesign touches it.

**Add — illumination normalisation.** The paraquat rig is white tile with a glare gradient and
grout lines; the first rig was a plain wall. A single global threshold (`diff_threshold = 26`)
cannot serve both. Proposal: estimate a smooth illumination field from the median background
(large-kernel morphological opening), divide it out, and threshold on the normalised residual.
Retain the absolute threshold as a floor.
*Test:* detection recall on the same frames before/after, on both rigs.

**Add — motion-blur and focus estimate.** Per sampled frame, variance of the Laplacian inside
each vial region, normalised by the region's contrast. Used two ways: as a covariate that
**widens the uncertainty** (blurred frames genuinely carry less information), and, past a
threshold, as a refusal. Threshold `TBD-sim`.

**Add — effective frame-rate check.** The two rigs differ (29.97 fixed HEVC vs 29.73–30.03
variable h264). Read presentation timestamps where available and compute the realised interval
distribution rather than trusting the declared `fps`. A fly climbing at ~10 mm/s crosses a
25 mm-wide field in 2.5 s, so the sampling requirement is set by how precisely the *scoring
instant* must be located, not by the motion itself — which makes it a function of the local slope
of the climbing curve. Minimum fps `TBD-sim`.

**Not adopted:** stabilisation beyond translation. Phase 0 measured drift on real footage at
~1 px; fitting rotation and scale to that invites overfitting. Revisit only if the tilt estimate
in §2 shows frame-to-frame rotation.

---

## 2. Geometry — the largest change, and the one with the most measured harm behind it

### 2.1 Anchor the vertical axis to the glass, not to the flies

Today `_tube_extent` defines the region as *where flies were seen*. Measured consequences: the
detected region ran 74 px to 1079 px for identical glassware, the origin moves with the cohort's
behaviour, and a poor climber gets a shorter tube and therefore a lower bar.

**Proposal.** The temporal median *is* a photograph of the empty cylinder. Detect the two
near-vertical glass edges by a vertical-gradient Hough/RANSAC line fit within each band, and the
base from the strong horizontal edge where glass meets the mat. The region becomes the glassware.
Fall back to the current fly-envelope behaviour only when edge detection fails — and say so in
the output rather than silently.

### 2.2 Use the printed graduations as an intrinsic vertical ruler

This is the single best opportunity in the new footage and it is currently unused. Every cylinder
in both datasets is a graduated 50 mL cylinder with printed 10/20/30/40/50 marks at a **known,
equal physical spacing**. Those marks are static, so they survive into the median background
cleanly.

Detecting the graduation ladder gives, per cylinder:

- **a vertical px/mm**, measured along the axis that heights are measured on — instead of today's
  scale inferred from tube *width* and applied to *height*;
- **a perspective/tilt read for free**: under a tilted camera the ladder's spacing changes
  monotonically with height. Fit spacing vs height; a significant slope *is* the perspective, and
  its magnitude is the quality metric. A constant spacing means the camera is square on;
- **a physically meaningful scoring criterion.** In this laboratory the hand-drawn line sits at
  the 40 mL mark in every recording. "Above the 40 mL graduation" is reproducible in any lab that
  owns the same glassware, in a way that "above where somebody drew a line" is not.

*Test:* recovered px/mm against the cylinder's catalogue geometry, and against the diameter-based
estimate, on all 41 paraquat recordings; agreement of the graduation-derived line row with the
detected drawn line.

### 2.3 Perspective

With two glass edges and the graduation ladder, a per-cylinder homography is over-determined.
**But I propose to start with the weaker, cheaper model** — a vertical scale that varies linearly
with height (which is what a small tilt produces) — and only fit a full homography if the residual
justifies it. Measured on the paraquat recordings, if the ladder spacing is constant to within
noise, the full homography is unnecessary work and I will say so rather than build it.

### 2.4 Cross-vial consistency — hard invariants, not scores

Phase 0 measured: 4 of 15 recordings with the scoring line **outside** the detected region (2
unflagged), 8 of 15 with cylinder bases differing by more than 40 px on a level bench, and 71 of
143 exported rows (50%) originating from such recordings, 18 of them graded "good".

Proposal — these become **refusals**, not confidence terms:

| invariant | rationale |
|---|---|
| `y_top ≤ line_y ≤ y_bottom` | a scoring line outside the tube is not a measurement |
| base rows agree within `TBD-sim` px after accounting for fitted bench tilt | they stand on one surface |
| region heights agree within `TBD-sim` % | same glassware |
| `line_fraction` agrees across cylinders within `TBD-sim` | one line, drawn with a ruler |

`height_spread` — already computed at `climbing_video.py:704` and thrown away — becomes the second
of these.

And the confidence score is rebuilt: the `width_cv` term is **provably a constant 0.35** (all three
vials share one width in 15 of 15 recordings, CV exactly 0.0000) and is removed. Replaced by terms
that vary: glass-edge fit residual, graduation-ladder regularity, tilt magnitude, line contrast,
spacing regularity.

### 2.5 t₀

Measured: at the 10 s scoring time on the first rig, one second of t₀ error moves the reported
proportion by **more than 100% of its own value**. The paraquat set makes this testable for the
first time — 41 recordings where the tap is the first event and the hand counts fix the clock.

Proposal: report t₀ as an **interval** from the motion-burst profile; propagate it through the
locally-estimated slope of the climbing curve into the count uncertainty; and prefer
**window-averaged** scores (e.g. 8–12 s) to instants, which is both more stable and closer to what
a human scorer with a stopwatch actually does.

---

## 3. Detection and counting

**Classical or learned?** I will not commit before measuring, and the honest constraint is this:
the ground truth we now hold is **counts, not positions**. That is enough to *evaluate* any
detector and to *calibrate* its bias, but not enough to *train* a learned detector — SLEAP/DLC/YOLO
all need annotated locations. So the plan is:

1. Improve the classical detector and measure it against the 1,080 cells.
2. Only if it plateaus above the 3-fly floor, spend the annotation effort a learned detector needs,
   and say plainly what that effort is.

**Improvements to the classical path**, adopted from FreeClimber and from Phase 0:

- richer per-blob descriptors — eccentricity, circularity, integrated darkness — instead of area
  and aspect alone. These separate a fly from a tick fragment or a glare edge, which area cannot;
- publish the per-blob detection table, so a human can re-score without re-running the video;
- **occlusion handled explicitly** rather than by area ratio. Today a blob above `1.6 × median` is
  split into `round(area/typical)` flies. Proposal: keep the split as the point estimate but carry
  the **ambiguity forward** — a blob of 2.4 typical areas is 2 or 3 flies, and that uncertainty
  belongs in the interval, not in a warning string.

---

## 4. The uncertainty model — the core of the redesign

Today the exported table's numeric columns are `Replicate, Time, Count_climbed, n_total_flies`.
No uncertainty, and a denominator that changed across timepoints in **24 of 24** vials.

**Proposed output, per vial per scoring time:**

| field | meaning |
|---|---|
| `Count_climbed` | point estimate of the numerator |
| `n_total_flies` | denominator — **one value per vial for the whole clip**, never per timepoint |
| `Count_se` | standard error of the numerator, on the count scale |
| `Count_lo`, `Count_hi` | interval |
| `logit_var` | measurement variance on the logit scale, for the clustered stage |
| `MeasurementQuality` | 0–1 composite |
| `Triage` | `usable` / `flagged` / `refused`, with a reason string |

**Four variance components, each individually measurable:**

1. **Frame dispersion.** Spread of the count over the frames averaged at that instant. Already
   computed and currently discarded.
2. **Merge ambiguity.** From the blob-area distribution: the count's sensitivity to the split rule.
3. **Line-position uncertainty.** The elegant one. The number of flies lying within ±δ of the
   scoring line *is* the count's sensitivity to where the line sits, and δ comes from the geometry
   fit. A vial with nobody near the line has a robust count; a vial with six flies straddling it
   does not — and today both report the same confident integer.
4. **t₀ uncertainty.** The t₀ interval times the local slope of the climbing curve, which the
   12-point time course measures directly.

Combined on the logit scale and handed to the clustered-binomial stage as a measurement-error term,
so the beta-binomial no longer absorbs detector noise as biological overdispersion.

**The decisive test** (Phase 3): the video-derived ICC must reproduce the hand-count value of
**0.573**. If ignoring the measurement error inflates it, that is the direct demonstration that the
uncertainty matters — which is the brief's propagation requirement, answered with real data
rather than argued.

---

## 5. Triage and refusal

Every rule below is testable and every threshold marked `TBD-sim` comes from Phase 3 simulation,
not judgement.

| check | refuse when | source |
|---|---|---|
| line inside region | `line_y` outside `[y_top, y_bottom]` | hard invariant |
| cross-vial base agreement | spread > `TBD-sim` px after bench-tilt fit | hard invariant |
| region height agreement | spread > `TBD-sim` % | hard invariant |
| fly size resolvable | median blob < `TBD-sim` px² | resolution gate |
| frame rate | effective fps < `TBD-sim` | temporal gate |
| tilt | ladder-spacing slope > `TBD-sim` | geometry gate |
| glare | saturated fraction inside a region > `TBD-sim` | acquisition gate |
| blur | normalised Laplacian variance < `TBD-sim` | acquisition gate |
| count plausibility | detected total > 60 | kept from today |
| camera moved | existing `detect_framing_change` | kept from today |
| criterion outside region | existing refusal | kept from today |

`flagged` (not refused) means the number is emitted with a widened interval and a reason. The
existing `DetectorReliability` string is kept for continuity but demoted: the numeric quality score
becomes the thing downstream code reads.

---

## 6. Compatibility

The public API and the exported schema are **additive only**. `analyze_climbing_video`,
`analyze_batch`, `to_drophenix_table`, `count_in_vial`, `fly_heights`, `score_heights` keep their
signatures and their existing columns; new columns are appended. The one behavioural change I am
proposing is that **`n_total_flies` stops varying within a vial** — and that is a bug fix, logged
as such, with the Phase 0 evidence attached.

Optional secondary metric: **FreeClimber-style velocity** — slope of the most linear window of mean
vertical position vs time — with its regression R² as a free quality statistic. Reported alongside,
never instead of, the count.

---

## 7. Validation plan (Phase 3), stated now so the design is falsifiable

1. **Against the hand counts.** 1,080 paired comparisons. Report bias, MAE, and the fraction within
   the 3-fly biological floor, per timepoint and per condition — including where it fails.
2. **Interval calibration.** Do the stated intervals cover the hand count at their nominal rate?
   Coverage plotted against horizon and against quality score.
3. **Clustering.** Video-derived ICC vs the hand-count 0.573, with and without the measurement-error
   term.
4. **Simulation with known ground truth**, extended to what the current renderer lacks: perspective,
   blur, shake, glare, depth occlusion, low and variable fps. This is what sets every `TBD-sim`.
5. **Cross-rig.** The two datasets differ in codec, background, lighting and frame-rate stability.
   Thresholds tuned on one must be shown on the other.
6. **Against FreeClimber**, on shared footage, reporting where it wins.
7. **Baseline.** The current module's numbers on the same 1,080 cells — measured *before* any
   rewrite, so the improvement claim is falsifiable. In progress.

---

## 8. What this design does not fix

- **No positional ground truth.** We can validate counts, not per-fly detection. A detector that
  misses two flies and invents two elsewhere scores perfectly here.
- **Depth is lost.** A fly behind the cylinder is not counted by any method in this class,
  FreeClimber included.
- **One laboratory, one operator, two rigs.** Cross-lab generalisation stays unproven until
  someone else's footage arrives.
- **Day 8 and Day 12 have no TDP-43 and no 10 mM**, so the longitudinal arm cannot be validated at
  the doses of interest.
- **The 5-tap structure is repeated measures on the same animals** — habituation and fatigue across
  taps are confounded with tap number in both the hand counts and the video. Neither method can
  separate them; the design would have to change.

---

## 0c. NEGATIVE RESULT — swapping the detector does not fix it (2026-08-12)

**Hypothesis.** `climbing_video_analyzer` over-detects 3.4x on graduated cylinders because
adaptive thresholding cannot separate a printed tick from a fly. Graduations are STATIC, so
replacing only the detector with a temporal-median background difference, inside the identical
user-defined ROI, should remove the excess. Geometry would then be vindicated and §3 could stay
a refinement.

**Controlled comparison.** Six single-cylinder recordings (WT untreated n=20, TDP 10 mM n=15),
four timepoints each, 24 paired measurements against the laboratory hand counts. Held constant:
recording, frames, region, scoring line, truth. Changed: one step.

**A bug surfaced mid-experiment and invalidated half of it.** `_find_line` takes the darkest
horizontal feature in the tube. On all three TDP recordings the blue bench mat at the tube base
(contrast 69) is darker than the hand-drawn line at ~54% height (contrast 57), so the line was
placed 330, 437 and 632 px too low — one at **−1096% of tube height, below the base**. The
software was scoring "above the 15 mL mark" while the laboratory scored "above 40 mL". Those
numbers were withdrawn, not reinterpreted, and the comparison re-run with the line corrected.
All three WT recordings were already correct (0 px change).

**Result, after correction:**

| | bias | MAE | within the 3-fly floor | within-condition r |
|---|---|---|---|---|
| adaptive threshold | +2.79 | 6.96 | 16.7% | TDP −0.09, WT −0.08 |
| median background | +7.83 | 15.00 | 37.5% | TDP +0.22, WT −0.40 |

**The hypothesis is refuted.** The median background is worse on MAE, not better. Removing the
graduations is necessary but not sufficient.

**And a trap worth recording.** The adaptive threshold's *pooled* r is **+0.92**, which looks
like a strong result. It is entirely between-group separation — WT hand counts average 18.1 and
TDP 6.3. Within condition the correlation is noise. (WT's hand counts run 17–19 of 20, SD 0.79,
so range restriction is severe there; TDP has SD 2.42 and still gives −0.09, which is the
meaningful one.) No pooled correlation should be quoted for this module without its
within-condition value beside it.

**Fixed as a consequence:** a scoring line outside the tube is now a hard refusal in
`detect_vials`, with the offending fraction named. On the first dataset IMG_6058 moves from
confidence 0.968 / `needs_review` False to **0.30 / True**. Both video suites still pass.

**Proposed, awaiting approval:** replace `_find_line`'s global-darkest rule with a search
restricted to the middle band of the tube, where a scoring line is actually drawn. Measured: it
changes nothing on the three recordings the current rule already gets right, and corrects all
three it gets wrong.

**What this does to the plan.** Two independent detectors, same geometry and truth, both fail
and neither can rank one vial against another. That makes the occlusion ceiling the blocking
question rather than one item among several: until we know how many flies are *visible* in a
frame, we cannot tell whether a detector is bad or whether the information is absent.
