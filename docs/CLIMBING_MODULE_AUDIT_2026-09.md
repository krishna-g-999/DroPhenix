# Scientific audit of the six climbing-video modules

**Date:** 2026-09-23 · **Scope:** `climbing_band`, `climbing_detectors`,
`climbing_layout`, `climbing_simple`, `climbing_video`, `climbing_video_analyzer`
(4,888 lines). Companion to `docs/MODULE_AUDIT_2026-09.md`, which covers the
non-video modules and explicitly excludes this pipeline.

## How to read this

**Every finding below was produced by running the shipped code**, not by reading
it. Each carries the input, the observed output, and the reason it matters. Where
a hypothesis failed to reproduce, that is recorded too — three did.

Findings are graded:

| Grade | Meaning |
|---|---|
| **SILENT** | Produces a wrong number with no warning. The dangerous class. |
| **LOUD** | Fails or refuses visibly. A defect, but self-announcing. |
| **COST** | Correct but expensive or fragile. |
| **SOUND** | Checked, no defect found. Recorded so it is not re-audited. |

---

## Summary

| # | Module | Finding | Grade |
|---|---|---|---|
| F1 | `climbing_band` | `find_black_band` median of a bimodal candidate set returns a row with no feature on it | **SILENT** |
| F2 | `climbing_band` | `_norm_pos` cannot match the bench sheet this repo ships | LOUD |
| F3 | `climbing_band` | `pi()` accepts a negative count and returns a negative PI | **SILENT** |
| F4 | `climbing_band` | `persistence_filter` is O(F²·n²) | COST |
| F5 | `climbing_video_analyzer` | `try_detect_reference_line` cannot see a line outside 25–75% of the ROI | LOUD |
| F6 | `climbing_simple` | `find_line` rejects the real line when the box is flush to the frame edge | LOUD |
| F7 | `climbing_layout` | `canvas_to_native` rounds each edge independently; box width drifts by 1 px | COST |
| F8 | `climbing_video` | `count_in_vial` merge correction inverts once merges exceed ~50% of blobs | **SILENT** |
| F9 | `climbing_video` | a merged blob is assigned wholly to one side of the line | **SILENT** |
| F12 | `climbing_video_analyzer` | fly size is taken from the ROI width; an over-wide box zeroes the count | **SILENT** |
| F13 | `fly_blob` | same width dependence in the production path | **SILENT** |

Four SILENT findings. All four can produce a confidently wrong scientific number.

---

## F1 — `climbing_band.find_black_band` returns a row with nothing on it

**Grade: SILENT.** `modules/climbing_band.py:98`

The function collects every row whose dark-pixel coverage exceeds `min_coverage`
and returns `int(np.median(rows))`. That is correct for one band. It is wrong for
two, and the median then lands in the gap between them.

**Input:** an 800×100 synthetic tube with two full-width dark features — one at
rows 100–105 (a taped label edge, a shelf, a second marker line) and the real
drawn band at rows 500–505.

**Observed:**

```
dark full-width rows : [100..105] and [500..505]
find_black_band      -> 302     coverage at row 302 = 0.00
```

The returned scoring line sits on clear glass, 200 px from either real feature.
Nothing in the return value signals this; the caller receives an `int` exactly as
it would for a clean single band.

**Why it matters here:** the Day-3 rig has both a drawn band and dark label
material on the same tube — that is the documented reason the *other* locator
failed on Day 3. This function has the same exposure by a different mechanism.

**Fix:** cluster the candidate rows by contiguity and return the median of the
**darkest contiguous run**, not of all candidates. Refuse (or report both) when
two runs are separated by more than one band-thickness and have comparable
darkness.

---

## F2 — the shipped validator cannot read the shipped bench sheet

**Grade: LOUD.** `modules/climbing_band.py:622`

`_norm_pos` normalises a cylinder label by extracting its digits. `data/validation/
hand_count_holdout.csv`, committed in this repo, labels cylinders `left`,
`middle`, `right` — deliberately, because the R1/R2/R3 pen marks are re-ordered
between recordings and were nearly mis-joined once already.

**Observed:**

```
_norm_pos('left')   -> 'LEFT'        _norm_pos('C1') -> 'P1'
_norm_pos('middle') -> 'MIDDLE'      _norm_pos('P1') -> 'P1'

validation_report(shipped hold-out sheet, tool CSV):
   hand-count rows supplied : 30 of 30 filled in
   matched to a tool reading: 0
   UNMATCHED ... : 30
```

The report is honest — it says 0 matched rather than inventing a number — so this
is LOUD, not SILENT. But the validation tool in this repo cannot validate the
validation data in this repo.

**Fix:** map `left|middle|right` → `P1|P2|P3` inside `_norm_pos`, and reject a
sheet whose positions resolve to fewer distinct values than it has cylinders.

---

## F3 — `pi()` returns a negative Performance Index

**Grade: SILENT.** `modules/climbing_band.py:451`

The function refuses when `count > n` — a sealed vial cannot hold more than it was
given — but applies no floor.

**Observed:**

```
pi(count= -5, n=20) -> -0.25
pi(count=  0, n=20) ->  0.0
pi(count= 21, n=20) -> None
```

**This is reachable, not hypothetical.** The crossing-gate estimator in the same
module (`detect_crossings`, kept as a documented failure) produced net counts down
to **−32**, and PI values to **−1.60**, on HTT Male cylinders. A negative
proportion is not a low score; it is a detector fault.

**Fix:** refuse on `count < 0` exactly as on `count > n`, with the same "this is a
fault, not a score" reasoning.

---

## F4 — `persistence_filter` is quadratic in frames *and* detections

**Grade: COST.** `modules/climbing_band.py:406`

Radius matching replaced hard bins for a good measured reason (bin boundaries
split static detections). The replacement compares every detection in every frame
against every detection in every other frame.

**Observed:**

```
12 frames x  8 detections -> 0.002 s
60 frames x 20 detections -> 0.302 s
60 frames x 40 detections -> 1.148 s
```

At 36 cylinders that is ~41 s of pure filtering per dataset, and it grows as the
square of the sampling density — precisely the direction anyone would want to move
to reduce noise.

**Fix:** build one `KDTree` per frame (already a dependency, used in
`detect_crossings`) and query, rather than a Python double loop. O(F²·n·log n).

---

## F5 — `try_detect_reference_line` cannot see a line near the ends of the ROI

**Grade: LOUD.** `modules/climbing_video_analyzer.py:531`

The search is restricted to `h//4 … 3h//4`.

**Observed**, with an unmistakable 6 px black line drawn across a uniform ROI:

```
true line at 0.15 of height -> None      <- missed
true line at 0.30           -> 0.28
true line at 0.50           -> 0.48
true line at 0.70           -> 0.68
true line at 0.85 of height -> None      <- missed
```

Returning `None` is the module's documented refusal, so the operator is told to
place the line by hand — LOUD, not SILENT. But a band at the 40 ml mark of a tube
whose ROI runs to the bench is at roughly 0.2 of ROI height, i.e. just outside the
window, and the operator is asked to do by hand what the software could do.

**Fix:** search the full ROI and rely on the existing 3× mean-strength test to
reject weak peaks; the band restriction is doing work the strength test already
does.

---

## F6 — `find_line` rejects the real line when the box touches the frame edge

**Grade: LOUD.** `modules/climbing_simple.py:162`

The function rejects candidates that are equally dark *outside* the tube, which is
a good rule — it removes the bench mat and the shelf edge. It samples that outside
strip at `box.x0 − 40 … box.x0 − 4` and the mirror on the right. When the box is
flush to the frame, those slices clamp to **inside the tube**, the real band
appears equally dark "outside", and it is rejected.

**Observed**, identical tube and identical band at row 400, only the frame margin
differing:

```
tube mid-frame, 60 px clear each side -> row 402  depth 165.2  correct
tube 10 px from the left edge         -> row 402  depth 165.2  correct
tube flush to both frame edges        -> row 180  depth   0.0  *** LINE MISSED ***
```

Row 180 is the search band's lower bound — a structural artefact, not a feature.
`is_clear()` returns False (depth 0 < 25), so a caller that checks is protected.
**Callers that use `fit.row` without checking `is_clear()` are not**, and both
`scripts/score_climbing_sp.py` and `scripts/pipeline/hd_score.py` do exactly that.

**Fix:** when either outside strip has fewer than ~10 usable columns, skip the
outside test rather than comparing against tube interior, and say so in `LineFit`.

---

## F7 — `canvas_to_native` rounds each edge independently

**Grade: COST.** `modules/climbing_layout.py:233`

**Observed:** the same 40-unit drawn box, mapped 700 → 1920, produces native
widths of **109 or 110 px** depending only on where it sits on the canvas.

Because fly diameter is `0.07 × box width`, that is a ~0.9% swing in the assumed
fly size from nothing but box position. Small — but it means the same operator
drawing the same box twice gets two different detector calibrations, which
undermines the inter-operator reproducibility measurement proposed for Phase 2.

**Fix:** map the origin and the *width* separately, then derive `x1 = x0 + w`.

---

## F8 — the merge correction inverts once merges become the majority

**Grade: SILENT.** `modules/climbing_video.py:859`

`count_in_vial` estimates one fly's footprint as the **median blob area in the
same frame**, then splits blobs larger than `1.6 × typical`. Its own docstring
states the assumption ("as long as most detections are single animals") and, two
paragraphs later, names the case that violates it: *"Undercounting from merges is
the dominant error at early scoring times, when flies are piled at the base."*

Those two statements cannot both hold. When most blobs are merged the median *is*
a merged blob, `typical` doubles, and the split silently stops firing.

**Observed**, synthetic tube, touching pairs plus singles:

```
1 merged pair + 6 singles (truth  8) -> n_total  8  typical 145  n_merged 1  err  +0
3 merged pairs+ 4 singles (truth 10) -> n_total 10  typical 145  n_merged 3  err  +0
5 merged pairs+ 2 singles (truth 12) -> n_total  7  typical 331  n_merged 0  err  -5
6 merged pairs+ 1 single  (truth 13) -> n_total  7  typical 331  n_merged 0  err  -6
```

The correction works perfectly up to 3/7 merged and then **collapses at 5/7**:
`typical` jumps 145 → 331, `n_merged` falls to 0, and the count reads 7 where 13
flies are present — a **46% undercount** reported with no flag at all.

`n_merged = 0` is the tell, and it reads identically to "no merges occurred."

**Why it matters here:** this is the t = 10 s piled-at-base condition on every
recording, and it is the same direction as the ceiling effect measured against
your hand counts (tool reads −7.2 at 15 flies).

**Fix:** estimate `typical` from a **pooled distribution across the whole
recording**, weighted toward late timepoints when the cohort is dispersed — the
same reasoning that made pooled fly-size calibration work in the N-from-video
experiment. Then flag `typical_area` when it departs from the pooled value, since
that departure is the diagnostic.

---

## F9 — a merged blob is assigned wholly to one side of the line

**Grade: SILENT.** `modules/climbing_video.py:940`

`if b["cy"] < vial.line_y: n_above += k` — the whole estimated count `k` follows
the blob's centroid.

**Observed:** one elongated blob spanning y = 385…415 with its centroid on the
line at y = 400, plus four singles below.

```
n_total 7   n_above 0   n_below 7   n_merged 1
```

At least one of the flies in that blob is above the line; none is counted there.
The error is largest exactly at the scoring boundary, which is where the
measurement is decided.

**Fix:** distribute `k` across the blob's vertical extent — for a blob spanning
`[y_top, y_bot]`, assign the fraction above the line proportionally. Not a
segmentation, but strictly better than all-or-nothing, and it degrades gracefully.

---

## F12 / F13 — fly size is taken from the drawn box, not the cylinder

**Grade: SILENT.** `modules/climbing_video_analyzer.py:274`, `modules/fly_blob.py:153`

Both detectors scale the expected fly from the width of the region they are given.
`climbing_video_analyzer` uses `min_area_w2 × w²`; `fly_blob` uses
`fly_d_frac × w`. This is *correct* when the region is the cylinder. The
"Climbing (Simple)" page asks the operator to type the columns by hand.

**Observed**, 20 flies in a 100 px cylinder, only the drawn box widening:

```
box 100 px wide (tube is 100) -> 19 of 20 flies   assumed fly 4.0 px
box 140 px wide               -> 19 of 20         assumed fly 5.6 px
box 200 px wide               -> 19 of 20         assumed fly 8.0 px
box 300 px wide               ->  0 of 20 flies   assumed fly 12.0 px

fly_blob.fly_diameter: 80 px -> 5.6    100 -> 7.0    150 -> 10.5    200 -> 14.0
```

A box drawn three times too wide reports **zero flies in a full tube**, with no
error and no warning. A box twice too wide doubles the assumed fly diameter — and
`fly_d_frac` is documented in `fly_blob` as sharply peaked (0.05 → MAE 9.11,
0.07 → 2.93, 0.10 → 9.82), so a modest drawing error moves the detector off its
calibrated optimum.

This is the same class as the already-fixed defect where fly size depended on box
*height*. The height dependence was removed; the width dependence is load-bearing
and cannot simply be removed — but it must not be silent.

**Fix:** measure the glass edges inside the drawn box and use the **detected tube
width**, not the box width; refuse when the box is more than ~1.3× the detected
tube. Failing that, show the assumed fly diameter in pixels next to the preview so
the operator can see it change.

---

## Checked and SOUND

Recorded so they are not re-audited:

- **`score_crossings` beyond the recording** returns `NaN` with an explanatory
  note, never 0. Verified on a 45 s clip requesting t = 60 and 120 s.
- **`score_heights`** refuses when the criterion lies outside the measured region,
  with the reasoning documented from a real case where wild-type flies scored 0%.
- **`align_to`** refuses on frame-size mismatch, on correlation below 0.08, and on
  a shift exceeding 25% of a vial width; it correlates on a padded vial box rather
  than the whole frame, for a documented measured reason (periodic tiling).
- **`analyze_frame_for_vial`** computes a two-sided signed discrepancy against
  `n_flies_expected`, so over-detection is as visible as under-detection.
- **`line_mask_px` defaults to 0** — flies on the line are *not* erased by default.
  My hypothesis that they were did not reproduce.
- **`recommend_backend`** refuses (`None`) on unstable framing and on the
  no-motion/no-clutter case, and explains why in prose.

### Hypotheses that did not reproduce

Three suspicions were wrong and are recorded as such:

1. Flies on the scoring line are deleted — **no**, `line_mask_px` is 0 by default.
2. A merged pair straddling the line mis-assigns both — **true, but my first test
   left a 16 px gap so no merge occurred**; it reproduced only after forcing the
   blobs to touch (F9).
3. `score_crossings` silently reuses the last grid sample for an in-range
   timepoint — **not observed**; the duration guard fires correctly.

---

## One concern that is not a defect

`recommend_backend` returns `blob` with **`confident=True`** on a probe reporting
**40 static marks** — eight times `STATIC_CLUTTER_LIMIT`. Its reasoning is sound
in isolation (clutter and no motion defeat the other two backends, so blob is the
only survivor). But the HD hold-out measured `blob` over-detecting by +7 to +13
flies on exactly the label-facing cylinders that generate that clutter. The
recommender is choosing the least-bad option and calling it confidence.

**Suggestion:** separate "which backend" from "how much to trust it". The first is
a choice; the second is a measurement the probe cannot currently make.

---

## What this means for the v2 architecture

The four SILENT findings share one shape: **a quantity is estimated from the
data in a way that is valid under an assumption the code does not check.**

- F1 assumes one dark band.
- F8 assumes most blobs are single flies.
- F9 assumes a blob is on one side of the line.
- F12/F13 assume the drawn box is the cylinder.

Each is reasonable; none is verified at runtime; all four fail in the direction
that matters for this assay. That is the argument for the reviewer's QC-vector
proposal stated concretely: the checks needed are not generic image-quality
metrics but **assumption checks specific to each estimator** —

```
band_is_unimodal          F1
typical_area_vs_pooled    F8
merged_fraction           F8
straddling_blob_fraction  F9
box_width_vs_tube_width   F12/F13
outside_strip_usable      F6
```

Those six are cheap, and each is the precondition of a number the pipeline already
reports.

## Recommended order

1. **F3** — one line; a negative PI must never leave the module.
2. **F12/F13** — show the assumed fly diameter in the UI today; detect tube width next.
3. **F8** — pooled `typical_area`; the largest silent numeric error found.
4. **F1** — contiguity clustering in `find_black_band`.
5. **F2** — so the shipped validator can read the shipped data.
6. **F9** — proportional assignment across the line.
7. **F6, F5, F7, F4** — bounded, self-announcing, or cost only.

None of these changes the accuracy conclusion in
`docs/CLIMBING_BAND_STATUS_2026-09.md`: the tool screens, it does not quantify.
F8 and F9 are plausible contributors to the measured ceiling effect, and fixing
them is testable against the existing 480 hand counts — but that test must be run
on a set not used to develop the fix.
