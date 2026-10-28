# Draw-the-vials: design proposal

**Status: PROPOSAL, awaiting approval.** Nothing here is implemented.

The user's suggestion: on upload, let the operator draw a box around each vial, label them,
and mark the region flies are counted above. This document says how much of the measured
error that removes, what it does not touch, and what the interface should be.

---

## 1. Would it work? Yes — and it targets the largest remaining error source

Every geometry failure measured in this project is one that a drawn box and a drawn line
eliminate outright. These are not hypothetical; each is a recorded observation.

| measured failure | recordings | what a hand-drawn layout does |
|---|---|---|
| scoring line found **outside the tube** — one at **−1096%** of tube height, i.e. below the base | 4 of 15 | eliminated |
| line placed 330, 437, 632 px too low: `_find_line` takes the darkest horizontal feature and the blue bench mat beats the drawn line | 3 of 3 TDP | eliminated |
| vial boxes placed on the **frame edges** while the cylinders sat in the middle | TDP 10 mM | eliminated |
| **3 cylinders detected where 2 exist** | TDP untreated | eliminated |
| cylinder bases **649 px apart** although they stand on one bench | IMG_6058 | eliminated |
| detected region ran **74 px to 1079 px** for identical glassware, because the region is the fly envelope and not the glass — so the measurement origin moves with the phenotype | across the set | eliminated |

On the first dataset, **71 of 143 exported rows (50%)** came from recordings with
demonstrably broken geometry, 18 of them graded "good". A drawn layout removes that entire
class.

And we know what the result looks like when the geometry is right, because it has been
measured: correct line + `threshold_masked` + crossing-based scoring gave **MAE 2.37 flies
on wild-type**, against a floor of 3 flies set by the laboratory's own repeat taps. That is
inside the noise of the assay itself.

### What it does NOT fix, stated plainly

- **Detection inside the box.** Printed graduations are still fly-sized and dark;
  `threshold_masked` handles them by masking their locations, and that is independent of
  who drew the box.
- **Flies that are not visible** — on the far side of the glass or behind printed text.
- **TDP 10 mM**, which is at MAE 3.78 and still above the floor; two of those recordings
  have almost nothing detectable at all and are already refused.
- **Within-condition ranking.** Tap-to-tap differences of 1–2 flies remain below what this
  resolves.

So the honest claim is: this removes the geometry class of errors, which was roughly half
the exported dataset. It does not make the detector better.

---

## 2. The thing that decides whether it is pleasant or tedious

The paraquat set has 41 recordings. Drawing 41 layouts is not a feature, it is a punishment.
So: **does the camera stay put between recordings in a session?** Measured, by phase
correlation against the first frame of each cluster, scaled to native resolution:

```
WT Untreated   +0.3  -1.5 | +4.9 -13.1 | +3.8 -16.1 | -140.3 +33.0 px
               (new rig)  | +2.6  +0.7 | +75.4  +3.3 | -47.5 +62.2 | +7.2 +6.7
TDP 10mM       +0.2  -0.6 | -0.7  -4.8 | +10.3  -0.4 | +16.5 -15.1 px
               (new rig)  | +0.4  -0.1 | +0.3  -0.3 | +29.1 -254.9 | -10.9 -58.4
```

**Mostly it does — but not always.** Many recordings sit within a few pixels of their
neighbour; a handful move by 75, 140, even 255 px. A vial here is ~150–190 px wide, so a
140 px shift moves the box off the cylinder entirely, while a 5 px shift is irrelevant.

This rules out "draw once, apply blindly". It equally rules out "draw every time". The
design below takes the third option, which the module already has the machinery for.

---

## 3. Proposed interface

### Step A — draw the vials, once per rig setup

A representative frame is shown on a drawable canvas (`streamlit-drawable-canvas`, already
installed and in `requirements.txt`, and never used by the current page). The operator drags
a rectangle around each cylinder. Boxes are auto-numbered **left to right**, matching the
`R1, R2, …` convention the rest of the pipeline uses, and each gets a row in a small table
for genotype, treatment, sex and **flies loaded**.

The canvas is displayed scaled to fit; every coordinate is converted back to native pixels
on the way out. This is the one place a drawing interface usually goes wrong and it will
have an explicit test.

### Step B — one line, not one line per vial

The operator drags a single horizontal line across the whole frame. It applies to every
vial, with an optional per-vial nudge.

This is deliberate. The line is drawn on the glassware with a ruler at one height, so the
vials' line positions are not independent quantities — and when the software treated them as
independent it produced a `line_fraction` spread of **1.34 across three cylinders in one
frame**. Encoding the physical constraint in the interface prevents the error rather than
detecting it afterwards.

A "snap to the nearest printed graduation" helper is offered, since this laboratory's line
sits on the 40 mL mark and graduations are the most reproducible landmark available.

### Step C — the tap, proposed rather than hunted

`t0` matters: one second of error changes the 10 s score by more than 100% of its value. The
operator should not have to scrub for it. The tap is found physically — the moment mean fly
height is at its minimum, which is measurable and unambiguous — and offered as a marker the
operator confirms or drags. Measured on the WT recording, mean height runs 0.17 at the tap,
rising to 0.67 by 12 s, so the minimum is sharp.

### Step D — flies loaded, per vial

A required field, not an optional one. When it was left blank, the denominator was taken
from the detector and **changed across timepoints in 24 of 24 vials** — up to 39 → 17 within
one vial, for a sealed cylinder whose fly count cannot change. This is the cheapest fix to a
real defect in the whole project.

### Step E — apply to the rest of the session, with alignment and a preview

The layout is saved as a small JSON. For every other recording in the session the software:

1. estimates the shift between that recording's background and the reference by phase
   correlation — the same routine already validated in this module (detected-total CV
   **114% → 12%**);
2. translates the boxes and the line by that shift;
3. shows the operator the aligned layout on a frame from that recording;
4. **refuses to apply silently** when the shift exceeds a fraction of the vial width, which
   is where the box would no longer sit on the cylinder. From the measurements, a 5 px shift
   is nothing and a 140 px shift is fatal; the exact threshold comes from a sweep, not from
   me picking a number.

On this dataset that turns 41 drawings into roughly 8, with the rest confirmed at a glance.

### Step F — the preview that already exists

Detections are overlaid before any run, as now, plus the count against the stated fly number
and the two-sided discrepancy flag. Nothing is scored until the operator has seen the boxes
with detections in them.

---

## 4. What this changes in the code

Additive. `climbing_video_analyzer` **already accepts a user-defined ROI** — the whole spec
was built around it. The only thing missing is that the region is typed in as four numbers
(`X start`, `Y start`, `Width`, `Height`) instead of drawn, which is why nobody would use
it. So this is largely an interface over machinery that exists and is tested:

- `VialConfig` already carries the ROI and `line_y_frac`;
- `climbing_detectors.count_with_backend` already takes any region;
- `score_crossings` already takes any region and does the protocol-correct scoring;
- `align_to_background` already estimates the shift needed for Step E.

New: a canvas page, a layout JSON with a version field, the coordinate-scaling test, and the
alignment-with-refusal step.

---

## 5. Honest expectation

With drawn geometry, `threshold_masked`, and crossing-based scoring, wild-type vials measured
**MAE 2.37 flies** — below the 3-fly biological floor. That is what this should deliver on
clean footage, and it would make the video route usable for real analysis rather than a
cross-check.

It will not rescue recordings where the flies cannot be seen. Those are already refused, and
should stay refused.
