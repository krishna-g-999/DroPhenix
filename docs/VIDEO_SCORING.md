# Scoring a climbing assay from video — design, evidence, and limits

What the pipeline does, why it is built this way, what it achieves on the
laboratory's own recordings, and — most importantly — what it cannot yet be
claimed to do.

Module: `modules/climbing_video.py` · Validation:
`validation/validate_climbing_video.py`

---

## 1. What the recordings actually look like

Fifteen recordings were supplied (`Testing data/Climbing assay/`). All fifteen
share a setup and all fifteen decode without trouble:

| Property | Value |
|---|---|
| Format | iPhone `.MOV`, HEVC/H.265 |
| Resolution / rate | 1920 × 1080, 29.97 fps |
| Duration | 120.5 – 121.7 s (~3,620 frames) |
| Scene | 3 × 50 ml graduated cylinders, cotton-plugged, on white paper against a styrofoam sheet |
| Scoring line | hand-drawn black band at the 40 ml graduation on each cylinder |
| Metadata | a handwritten card in shot (genotype, sex, treatment, day, date) |

Two structural facts shaped the whole design.

**Each recording is one trial with a full time-course, not one endpoint.**
Whole-frame motion analysis shows a single burst in the first 1–3 seconds — the
tap and the operator's hands withdrawing — and then near-total stillness for two
minutes. A frame montage confirms the biology: at 2 s every fly is on the base;
by 8 s they are spread up the tube; by 20–30 s several are above the line. One
video therefore yields climbing at 5, 10, 15, 20, 30 and 60 s, which is a
richer measurement than the single 10 s or 30 s count usually recorded by hand.

**Framing varies between recordings.** Camera distance, cylinder position in
frame and zoom all differ. Fixed regions are impossible; the regions must be
found per video, or drawn by the operator.

---

## 2. Why the existing detector could not be used

`fly_detector.py` was written against synthetic clips with a clean fixed
background. On a real frame containing roughly thirty flies **it detected
three**. The reasons are structural, not tuning:

1. **It thresholds the whole frame.** A phone frame contains a window, a
   curtain, a bench and a table. Otsu's method separates "bright window" from
   "everything else"; the flies end up inside the second class.
2. **Its `dark` mode counts dark pixels.** A graduated cylinder is covered in
   printed scale ticks and numerals. They are dark. No intensity threshold can
   tell them from a fly.

---

## 3. The design

### 3.1 Temporal median background — removes every static dark feature at once

The scale markings are **static**; the flies **move**. The median of ~40 frames
spread across the clip therefore reconstructs the cylinder complete with its
printed scale and *without* its flies. Subtracting it eliminates ticks,
numerals, the hand-drawn line and the label text in one operation, with no
tuning.

A median rather than a mean, deliberately: a fly resting for a quarter of the
clip leaves no trace in the median, but a permanent dark smudge in the mean.
Sampling starts after ~4 s so that the operator's hands are never baked into the
background.

### 3.2 Sub-pixel registration — the step that made the difference

The cylinders are tapped by hand and go on settling for the first ten seconds or
so. Phase correlation against the background measures the drift: about **one
pixel**, decaying to zero by t ≈ 15 s.

One pixel sounds negligible. It is not. A graduation tick is one to two pixels
wide, so a one-pixel misalignment makes every tick and numeral differ strongly
from the background and register as a fly — and it does so *exactly during the
first fifteen seconds*, which is when a climbing assay is scored.

Measured on `IMG_6030.MOV`, cylinder R1, at 5–90 s:

| | counts across the clip | variation |
|---|---|---|
| Without registration | 109, 117, 97, 151, 17, 20, 19, 16, 16, 13 | **89%** |
| With registration | 19, 14, 18, 22, 17, 20, 18, 16, 16, 13 | **15%** |

The number of flies in a sealed cylinder cannot change, so every bit of that
variation is detector error. The first sequence is not noisy measurement; it is
a fabricated sixfold overcount concentrated in the scoring window.

Registration is applied to the activity map as well as to counting, for the same
reason: an unregistered activity map marks the printed scale and the bench edge
as places where flies have been.

### 3.3 Locating the cylinders — let the flies show you

The most reliable cue for where a cylinder is turns out to be **where flies have
been**. Accumulating, over the clip, the pixels darker than the background
produces a map that is dense exactly inside the tubes, and needs no assumption
about glass, glare, labels or clutter.

Three refinements were each forced by an observed failure:

- **Fragment merging.** A vial holding fewer flies does not produce one
  continuous band — by chance no fly occupies some horizontal slice of it, and
  the band breaks in two. Selecting the strongest bands then picks *half* a
  vial and places the region off-centre, clipping real animals out of their own
  tube.
- **Common width.** The cylinders are physically identical, so their regions
  must be too. A quiet vial otherwise gets a narrower region than a busy one and
  loses the flies nearest the glass.
- **Expand-only refinement.** Re-deriving band edges from each band's own peak
  must never shrink a band, or it silently undoes the fragment merge by stopping
  at the very dip that caused the fragmentation.

The scoring line is then found within each region as the darkest sustained image
row, searched inset from the tube edges so the printed scale does not dominate.
On synthetic footage this recovers the true row **exactly** (0 px error in all
three tubes).

### 3.4 Three independent quality gates

No single check catches every failure, so there are three, and they are blind to
different things:

| Gate | Catches | Blind to |
|---|---|---|
| **Region confidence** — width consistency, spacing regularity, line contrast | mislocated or missing cylinders | a well-formed region on the wrong object |
| **Count stability** — the total in a sealed vial cannot change, so its variation across the clip is pure detector error | flickering, occlusion, misregistration | a detector that is consistently wrong |
| **Plausibility** — a climbing-assay cylinder holds tens of flies, not hundreds | counting label text, clutter or reflections | undercounting |

The third exists because one recording (`IMG_6079`) passed region confidence at
0.72 and reported 86, 63 and 168 flies per cylinder. Confidence in the *region*
and stability of the *count* were both blind to it.

### 3.5 The numerator and the denominator are not equally hard

The performance index needs flies-above-line and flies-total. Counting **above**
the line is comparatively easy: few animals, well separated, in the clean upper
tube. Counting **all** of them is much harder, because at early times they are
piled on the base, overlapping, against a bright reflective foot.

So the pipeline asks the operator for the total — they loaded the vial and they
know — and asks the vision system only for the numerator. The detected total is
still computed, and reported as a **cross-check**: when the two disagree by more
than 20%, the analysis says so rather than silently preferring one.

Visual inspection supports this split. On `IMG_6030` R3 at t = 30 s the pipeline
reports 5 flies above the line, and 5 is what a human counts on the annotated
frame; the total is slightly under-counted, with a few blurred animals near the
base missed.

---

## 4. Results across all fifteen recordings

Every video processed without error, in 76–99 s each.

| Outcome | Count | Videos |
|---|---|---|
| High region confidence (≥0.89) and plausible counts | **10** | 6030, 6034, 6042, 6046, 6053, 6058, 6064, 6073, 6074, 6075 |
| Correctly flagged for review by low confidence | **4** | 6032 (0.00 — no flies detected), 6052 (0.54), 6063 (0.53), 6065 (0.70) |
| Passed confidence but caught by the plausibility gate | **1** | 6079 |

Region confidence reached 0.99–1.00 on the best recordings. Per-vial count
variation ran from 11% to over 100%, and is reported per vial as a graded
reliability rather than a pass/fail flag — perfect stability is not achievable
on real footage, because a fly that walks behind the label card genuinely cannot
be counted in that frame.

**No recording was silently mis-scored.** Every failure was surfaced by at least
one gate.

---

## 5. What is validated, and what is not

`validation/validate_climbing_video.py` establishes:

- **Exact counts on synthetic footage** with per-frame ground truth, in a scene
  that reproduces the real failure modes — a printed graduation scale, and
  sub-pixel drift in the opening seconds.
- **Scoring line recovery** to 0 px, and regions that contain 100% of the
  animals they must score.
- **The registration ablation**, on synthetic *and* on real laboratory footage.
- **Refusal**: an unreadable file raises a clear error; a clip with no flies is
  flagged, not scored.
- **Schema conformance**: the exported table loads through `data_parser`, each
  cylinder becomes its own `VialID`, and the design layer correctly identifies
  the vial as the unit of replication.

**What is NOT established: accuracy on real footage.** No human count of these
recordings exists, so the only real-data claim made is *stability*, which is
necessary but not sufficient — a detector that misses the same three flies in
every frame is perfectly stable and still wrong. The visual check on one frame
is encouraging, not evidence.

**This is the one thing needed to close the gap, and it is cheap:** have someone
count flies above the line, and total flies, in three or four frames per video
for five videos. That is perhaps an hour of work and it converts the whole
pipeline from "plausible and self-consistent" to "validated against human
scoring" — which is what a methods reviewer will ask for, and rightly.

---

## 5a. A bug this work uncovered in the existing application

`modules/recording.py` set, at import time:

```python
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
```

intending to quieten backend probing when enumerating cameras. These variables are process-wide
and are read once, when OpenCV first builds its backend registry, so they govern **file** decoding
as much as camera capture — and priority 0 does not deprioritise a backend, it **removes** it. With
it set, even an explicit `cv2.VideoCapture(path, cv2.CAP_FFMPEG)` returns `isOpened() == False`.

Because `app.py` imports this module at startup, **every video file in the running application was
unreadable on Windows**, including on the pre-existing Video Analysis page, which has nothing to do
with cameras. The failure was invisible to every standalone test script, because those do not
import `modules.recording` — which is precisely why it survived.

Fixed by setting FFMPEG to `1`; MSMF and DSHOW are unchanged. FFMPEG cannot open an integer camera
index on Windows, so it never competes with MSMF for a camera and only ever wins for files, which
is what is wanted. Both `validate_video.py` and `validate_climbing_video.py` pass after the change.

This is the second instance of the same class of bug in this project — the first was
`cv2.VideoCapture(path, cv2.CAP_DSHOW)`, a *camera* backend used to open *files*. The general
lesson: video-backend faults do not show up in unit tests that import only the module under test.
Exercise the application's own import path.

## 5b. Working across laboratories

Three questions decide whether this is a tool for one bench or for the field.

### Does it need a standard rig? No — and that is deliberate

The cylinders are found from **where the flies have been**, not from the shape of the glassware.
That is why the approach transfers: it makes no assumption about tube diameter, glass thickness,
glare, labels, or background clutter, only that flies are the things that move. A Drosophila vial,
a graduated cylinder and a purpose-built RING rack all work the same way.

What is *not* assumed any more:

- **The number of cylinders.** `auto_n_vials` determines it. Different labs run one tube, three,
  or a whole rack, and requiring the number up front turns a small ask into a silent error —
  asking for three when four are present discards a real vial; asking for four when three are
  present invents one out of clutter. When a batch comes off one rig, stating the number is still
  better, and the batch runner says so when recordings disagree.
- **That there is a line drawn on the glass at all.** See below.

### Should you give labs a template? Yes — but as an option, never a requirement

A tool that only works on its author's protocol is not a tool. So the design is: **work on
whatever arrives, and reward good practice.** Two separate things are worth handing out:

1. **A recording protocol** (§6). Guidance, not a gate. Following it improves the numbers; not
   following it does not stop the analysis, it just widens the quality warnings.
2. **A calibration**, which is a single number, not a printed target. State the cylinder's outside
   diameter in millimetres — 25 mm for a 50 ml graduated cylinder, 23.5 mm for a standard
   Drosophila vial — and every height is reported in millimetres.

That second one is the important one, and it is worth being blunt about why. **"Flies above the
line" is not a comparable measurement.** It depends on where somebody drew a line and how far away
the camera was, so two labs running the same genotype can report different numbers with neither
being wrong. Height climbed in millimetres is a physical quantity that anybody can reproduce — and
it is what the RING assay was originally defined on: Gargano et al. (2005) score *distance
climbed*, not a binary pass.

### Three scoring rules, applied after the fact

Detection extracts the **height of every fly**, and the scoring rule is applied to those heights
afterwards. The video is processed once; the criterion is cheap to change:

| Rule | Use when | Comparable across labs? |
|---|---|---|
| `drawn_line` | there is a mark on the glass (your recordings) | no |
| `height_fraction` | there is no mark; score above a fraction of tube height | partly |
| `height_mm` | you want the standard RING criterion, e.g. 80 mm | **yes** |

Validation asserts these agree where they must: scoring by the drawn line reproduces direct
counting *exactly*, and a fraction rule matches the equivalent millimetre rule fly for fly. A
dataset can therefore be re-scored under a collaborator's criterion — or under all three side by
side — without touching the video again.

### Two corrections the millimetre work forced, and one measurement worth knowing

**The detected region is not the glassware.** `_tube_extent` returns the span in which flies were
*seen*. The base is found reliably — flies always reach the bottom, and on a real recording the
three bases agreed to within 4 px — but the top is simply wherever that vial's best climber got to.
On one recording the three apparent "tube heights" came out as 610, 559 and 525 px for three
identical cylinders: a 16% spread that is biology, not geometry.

Left uncorrected this quietly corrupts two things. A `height_fraction` criterion would mean a
different physical height in every vial. Worse, **the poorer a vial climbs the shorter its apparent
tube becomes, so a poorly performing vial would be scored against a lower bar** — a bias pointing
exactly the wrong way. A common height is now imposed, measured up from each vial's own
well-determined base.

It also means `tube_height_mm` calibration is a trap: entering the *catalogue* height of the
cylinder makes every distance wrong, because the marked region is shorter than the glassware.
**Use the diameter** — it is easier to state accurately, it is measured across a region the
detector determines robustly, and it now yields an identical px/mm for every cylinder in a shot.

**And a finding worth reporting on its own.** Once distances are in millimetres, the hand-drawn
scoring line on three cylinders *in the same recording* sits at 95.0, 94.2 and 86.1 mm above the
base — and on other recordings at 66/75/71 mm and 83/72/79 mm. That is 8–11 mm of spread between
tubes photographed side by side, from hand-drawing and camera perspective together.

So the drawn line is not even self-consistent across three cylinders in one shot, let alone between
laboratories. This is the strongest argument for the millimetre criterion available, and it comes
from the laboratory's own data rather than from principle. It also means the three replicates in
each recording have been scored against slightly different physical thresholds all along —
switching to `height_mm` removes that source of between-replicate variance at no cost.

### Multiple taps, and a failure this uncovered

RING means *Rapid Iterative Negative Geotaxis*: many protocols tap three to five times and score
after each. `detect_taps` finds every tap, so an iterative protocol is not silently analysed as one
long trial.

Discriminating a tap from flies moving needed care. Two obvious approaches fail: frame displacement
is a fraction of a pixel during a tap because the camera is stationary, and a
few-standard-deviations threshold reported *eleven* taps on a single-tap recording. What works is
**prominence** — a tap involves hands entering and leaving the frame and is by far the largest
intensity event in the clip. Scale-free, and it still catches every tap of an iterative protocol
because those resemble each other far more than they resemble flies.

**Your recordings are single-tap**, confirmed frame by frame: one burst inside the first second,
then two minutes undisturbed. That is well suited to trajectory work, because it yields a
continuous climbing curve rather than a handful of endpoints.

This also solved a mystery. `IMG_6032` failed every quality gate for no visible reason. It turns
out **the camera is moved closer at about 20 s.** The temporal-median background assumes a fixed
scene, so a mid-clip move makes it a blend of two framings, leaves ghost features everywhere, and
detection collapses. `detect_framing_change` now catches this directly — by comparing median frames
from windows across the clip, independently of tap detection — and the analysis leads with a
warning telling the user to trim the recording. Four of the five recordings checked are correctly
judged framing-stable; only `IMG_6032` is flagged.

### Batch mode — what it actually means

Instead of opening fifteen videos one at a time, re-typing the genotype and treatment for each and
merging fifteen CSVs by hand, you write the metadata once in a manifest
(`templates/video_manifest_template.csv`) and get **one analysis-ready table**:

```
video,Genotype,Treatment,Sex,Age,TimeOfDay,ExperimentID,n_per_vial
IMG_6030.MOV,GMR-GAL4;UAS-hH,Withaferin,Female,3,10:30,LAB2026_03,20
```

This is not only convenience. Hand-merging is where mislabelling happens, and a mislabelled
treatment column cannot be recovered from the data afterwards.

Each recording is scored independently — background, regions, tap and calibration are all
per-video, because framing and zoom change between recordings. Only the metadata and the scoring
rule are shared. Recordings that fail quality thresholds are **reported and excluded**, never
silently included: they appear in `excluded` with a reason, so scoring one by hand afterwards is a
deliberate act.

## 5c. Where to run this: not in a browser

The web app is the right place to explore a dataset and the wrong place to process video. Four
reasons, none of them specific to Streamlit:

| | |
|---|---|
| **Size** | One recording is 120 MB; a session is nearly 2 GB. Uploading that to a hosted app is slow, often blocked outright, and pointless when the file is already on the machine doing the work. |
| **Duration** | Fifteen recordings take about 45 minutes. Browser frameworks re-run their script on interaction and hosted instances time out. Long jobs do not belong behind a web request. |
| **Privacy** | Unpublished data leaves the laboratory the moment it is uploaded. |
| **Methods** | A command is quotable. *"Recordings were scored with `drophenix_cli.py batch --height-mm 80`"* is a reproducible methods sentence; *"we clicked through a web form"* is not. |

**The recommended split: heavy, private and slow locally; statistics anywhere.** `drophenix_cli.py`
does the video work on the lab machine and emits a small CSV. The video is gigabytes; the scored
table is kilobytes, and kilobytes go anywhere — the app, R, a notebook.

```bash
drophenix_cli.py manifest --folder ./videos --out manifest.csv
# fill in Genotype / Treatment / Sex / n_per_vial
drophenix_cli.py inspect ./videos/IMG_6030.MOV --vials 3 --diameter-mm 25 --annotate ./checks
drophenix_cli.py batch --folder ./videos --manifest manifest.csv \
    --vials 3 --diameter-mm 25 --height-mm 80 --out results/
drophenix_cli.py check results/climbing_table.csv
```

`inspect` exists because the single most useful thing a user can do is look at an annotated frame
before trusting any number. `check` runs the design and clustering diagnostics on the scored table.

Other options considered and why they lose: a **desktop app** (PyQt, or Streamlit bundled with
PyInstaller) solves size and privacy but is a large maintenance burden for one lab; a **notebook**
is fine for exploration but poor for routine scoring and easy to run out of order; a **napari
plugin** would give region editing for free and is the strongest future option if manual region
correction becomes routine. A CLI plus the existing app covers all of it today at a fraction of the
cost, and the module is already a library, so the CLI is a thin wrapper over code that is already
validated.

## 5d. What the lab's own data says about clustering

Running `check` on the scored table is the first confirmation on real data of the statistical
problem this whole project began with:

> **ICC = 0.289** across 24 vials of about 17 flies. **Design effect 5.55.** Standard errors are
> **135% larger** than a fly-level analysis would give, and **401 flies carry the information of 72
> independent ones.**

The simulations earlier in this project assumed an intraclass correlation of 0.10. The real value
is nearly three times that, so the cost of treating flies as independent replicates is
correspondingly worse than the simulated 5.2-fold inflation of the false-positive rate. This is not
a modelling preference; it is measured in the laboratory's own recordings.

## 6. Recording guidance

The pipeline works better with small changes at the bench, none of which cost
anything:

1. **Keep the camera still.** A tripod or a propped phone removes the residual
   drift entirely. Registration handles the ~1 px of settling; it will not save
   a handheld pan.
2. **Start recording before the tap** so the tap itself is in shot. The tap is
   how the pipeline finds t = 0, and scoring times are meaningless without it.
3. **Keep hands out of frame after the tap.** They are excluded from the
   background by starting sampling at 4 s, but a hand re-entering mid-clip
   corrupts both the background and the registration.
4. **Avoid backlighting.** A bright window behind the cylinders reduces contrast
   between dark flies and the tube. The styrofoam backing is doing the right
   job — keep the light in front.
5. **Record the number of flies loaded per cylinder.** This is the denominator,
   and it is the single most useful thing the software cannot work out for
   itself.
6. **Keep the label card out of the cylinder regions**, or place it well to one
   side. Handwriting inside a region is the most likely source of a spurious
   count.

---

## 7. Where this fits

A scored video becomes the canonical DroPhenix climbing table via
`to_drophenix_table`, with **each cylinder as its own `VialID`** — because the
vial is the experimental unit, and a video of three cylinders is three
replicates, not one. From there it flows directly into the hierarchical
(clustered) statistics, the trajectory forecaster and the drug-ranking engine,
carrying the full time-course rather than a single endpoint.

The honest summary of the claim the tool can make today: *the software will score
your videos and tell you when not to trust the result.* Extending that to *the
software scores your videos as accurately as a trained scorer* requires the
human counts described in §5, and should not be claimed before then.

---

## 8. The scoring line is proposed by the software and confirmed by the operator

**Behavioural change, 2026-08-19.** The Draw Vials page no longer takes the scoring
line from a bare slider set to 45% of frame height, and no longer imposes one line
across every cylinder. The software now measures a line in each cylinder, states how
confident it is, and **refuses to score** a cylinder whose line is ambiguous until a
person has looked at it.

### Why

The line decides every number this module produces — a fly is climbed or not according
to which side of it the animal sits. A misplaced line does not fail loudly; it returns
a full set of ordinary-looking counts that are wrong.

That failure is not hypothetical here. The first version of the darkest-row rule
searched the whole box, and on the paraquat rig it put the line on the **bench mat at
the 10 mL graduation, roughly 400 px low, on 12 of 12 vials** — while reporting
margins of 1.02–1.06. I had previously quoted two of those cases (67 vs 60 and 80 vs
68) as wins; rendering both lines on a real frame showed they were not. The margins
were the tell, and they are now surfaced instead of being discarded.

### What the proposal is

The darkest horizontal feature in the middle band of the box **that does not continue
past the glass**. The last clause is the whole discriminator: a mark drawn on a tube
stops at the glass, while a bench mat, a shelf or a skirting board runs across the
frame. Candidate rows are compared against the strip 40 px either side of the box and
rejected when the darkness outside exceeds 0.6× the darkness inside
(`line_outside_px`, `line_max_outside_ratio` in `climbing_simple.DEFAULTS`).

Two numbers are reported with every proposal:

| | meaning | threshold |
|---|---|---|
| **depth** | how much darker than the tube's brightest row | ≥ 25 |
| **margin** | depth ÷ the next distinct candidate's depth | ≥ 1.3 |

A margin near 1.0 means a coin toss between two features. That is precisely the
signature of the 400-px error, which is why it gates rather than merely informs.

### What the operator sees, and what is blocked

Each cylinder gets its own slider pre-set to the measured row, a verdict beside it,
and a preview with every line drawn where it will actually be used — **green where the
software is confident, red where it is not**. A red cylinder must be ticked off by
eye, or the line moved, before the *Score these cylinders* button will enable. The
`problems()` check still applies afterwards: a confirmed line above or below its own
tube is refused regardless of confirmation, because confirmation is not a licence.

This follows the published method rather than departing from it. Cao et al. 2017
(*J Vis Exp* 127:e56507) have the operator mark the vial baselines with a cursor;
nothing automatic here is better than that, and pretending otherwise is what produced
the 400-px error.

### One line, or one per cylinder?

Both, and the distinction is physical. The line really is ruled on the glass at a
single height, but the tubes are not photographed from dead centre, so the same height
lands on a different row in each. Measured independently, the paraquat pair gave
**542 and 538**; the Day7 trio gave **581, 558 and 590** — the same line seen from
three angles. The layout therefore stores the shared height as the median and a signed
per-vial offset (`line_nudge_px`), written by `climbing_layout.set_vial_lines`. The
round trip is exact by construction and is tested on 2000 random layouts with zero
drift, because a few pixels moves every fly sitting near the line.

The `line_nudge_px` column was removed from the labelling table in section 4, where it
silently reset the confirmed lines to zero.

### How often confirmation will actually be asked for

Measured on 36 vials drawn from all three rigs (`headtohead_simple.py`, seed
20260818), the proposal was clear on:

| rig | vials | clear | median depth | median margin |
|---|---|---|---|---|
| set 1 | 12 | 75% | 129.3 | 1.44 |
| Day7 | 12 | 67% | 69.6 | 2.06 |
| paraquat | 12 | 58% | 56.9 | 1.72 |

So roughly a quarter to two-fifths of cylinders will ask for a look. That is the
intended cost: those are the vials where the previous behaviour produced a confident
wrong answer.

### What this does not establish

It does not measure how often the proposal is *right* — that needs hand-marked lines,
which do not exist for these recordings. What is measured is the count accuracy the
new geometry produces, on the only set with hand counts:

| | n | bias | MAE | within 3 flies |
|---|---|---|---|---|
| new (per-cylinder line) | 144 | −4.77 | **8.42** | **41.0%** |
| old (shared middle-band line) | 144 | −7.27 | 8.58 | 29.2% |

The improvement in MAE is small; the improvement in the proportion of counts landing
within three flies is not. On the subset where the line is clear, MAE is 6.67 and
52% land within three — which is the case for the gate, since the gate is what
separates those vials from the rest. WT untreated (MAE 18.67) and TDP untreated
(14.78) remain unsolved and are not improved by this change.

Validated by `validation/validate_confirmable_line.py`.

---

## 9. Two defects found by the validation suite, 2026-08-20

Both were surfaced by `run_all.py` reporting failures, and in both cases the failing
check was pointing at something real that its own message described wrongly.

### 9.1 The scoring line manufactured phantom flies

**The failing checks said** the detector found 18 or 19 of 20 synthetic flies, and that
the horizontal-line guard was no longer load-bearing. Both readings were wrong, and the
scene had to be instrumented before the cause became visible.

Rendering the same clip **with and without** the drawn line separated two independent
problems:

| fly radius | area vs the 64 px² floor | with line | without line |
|---|---|---|---|
| 5 (the old scene) | 1.23× | 19 | 19 |
| 6 | 1.77× | **21** | 20 |
| 7 | 2.41× | **23** | 20 |
| 8 | 3.14× | **22** | 20 |

At the old fly size the shortfall was identical with and without the line, so it was
never a line-removal fault at all — the flies were simply too close to the minimum-area
floor for lossy mp4v compression to leave them intact. But once the flies were big
enough to survive the codec, the line was seen to **add** one to three detections.

**Cause.** `remove_horizontal_lines` zeroed the band recovered by a wide-kernel
opening, but a drawn line is antialiased and video compression smears it further, so
rows just outside that band survived. The vertical close that follows — there to rejoin
a fly the line had cut in two — was applied to every column in the band, and inflated
those leftover smears into fly-sized blobs. They landed **on** the line, which is the
worst possible place for a phantom: exactly where the scoring decision is made.

**Fix**, in `climbing_video_analyzer.detect_flies_in_roi`:

1. The band grows outwards over rows whose foreground fraction still reaches
   `line_edge_frac` (0.20), capped at `line_edge_max_px` (4) each way so a raft of
   flies resting at the line cannot be eaten. A row carrying flies rather than a line
   sits at about 0.10–0.15.
2. The vertical repair is restricted to **columns with foreground on both sides** of
   the removed band — what a bisected fly looks like, and what a leftover smear does
   not.
3. A blob touching the removed band has been truncated by it, so it is judged against
   a reduced floor (`line_touch_area_frac`, 0.5). Condemning an animal for the part of
   itself the line was covering is the same error in the opposite direction.

**Measured after the fix:** exactly 20 of 20 at every timepoint, with and without the
line, for every fly radius from 7 to 10. Real-footage section F is unchanged (59
objects detected where 20 flies are present; mean |above − hand| 7.9) — that shortfall
is printed graduations, not the drawn line, and this change does not claim to touch it.

**The test scene was also wrong**, and is fixed alongside: it drew radius-5 flies (5%
of vial width) in nine columns 20 px apart, against a module whose area window is built
on a fly being about a tenth of vial width. It could not be repaired by enlarging the
flies alone — at 20 px spacing a radius-9 disc touches its neighbours and counts
collapse to 13 of 20 — so the spacing was widened with them: five columns 33 px apart,
radius 10, a 4.9× area margin and 13 px of clear space.

### 9.2 The interface offered the wrong scoring rule, and called it the protocol

This is the more serious of the two, because nothing about it looked broken.

`climbing_video.py` and `climbing_detectors.py` both document the position correctly:

> the net of up- and down-crossings is simply THE NUMBER STANDING ABOVE THE LINE.
> That is `mode="net"`, and it is the default.

`analyze_climbing_video` duly defaults to `scoring_mode="net"`. But **both** pages that
expose the choice offered:

1. *"Flies that have CROSSED the line by then (published protocol)"* → `crossings` → `max_reached`
2. *"Flies ABOVE the line at that instant"* → `instantaneous` → `net`

Streamlit's radio defaults to the first option, so the interface scored with the
**ratchet** by default while describing it as the published protocol — the exact rule
the modules document as *not* the protocol. The caption beneath it made it worse by
describing the correct rule ("a fly that climbs back down is subtracted") while
attached to the option that does no such thing.

This is the tail of an earlier correction: when the default was moved from the ratchet
to `net`, the module changed and the interface did not.

**Fix.** Both pages now offer the two rules under names that say what they compute,
protocol first:

- *Flies STANDING above the line — up minus down (the published protocol)* → `net`
- *The MOST that were ever above the line at one moment (a ratchet)* → `max_reached`

and pass the canonical names directly. The ambiguous spellings no longer appear
anywhere in `app.py`. They remain accepted by the modules for compatibility and are
canonicalised on entry, which is now tested: an exported table records the rule that
was **computed**, never the older spelling, because "crossings" could be read as either
quantity.

**Anyone who scored videos through the interface before this change got the ratchet.**
Those counts are not wrong measurements of what they measured, but they are not the
protocol's quantity and are systematically at least as large. Re-score with the first
option to obtain protocol counts.

---

## 10. The paraquat set re-scored under the protocol's rule, 2026-08-20

`scripts/rescore_paraquat_net.py` scores all 60 usable recordings with
`mode="net"`, the confirmable per-vial scoring line, and `backend="threshold"`, and
writes `paraquat_scored_net.csv` beside the videos with a `.provenance.json` sidecar.
**Every row now carries `scoring_mode`, `backend` and `step_s`** — the absence of which
was what made the audit below necessary in the first place.

**672 paired cells · bias −3.61 · MAE 5.34 · within 3 flies 36.2%**

| Genotype | Treatment | cells | MAE | bias |
|---|---|---:|---:|---:|
| TDP | 10 mM | 36 | 3.94 | −3.56 |
| TDP | 5 mM | 132 | 3.81 | −0.90 |
| TDP | Untreated | 84 | 6.44 | +1.70 |
| WT | 10 mM | 132 | 3.48 | −3.32 |
| WT | 5 mM | 156 | 7.38 | −7.33 |
| WT | Untreated | 132 | 6.00 | −5.61 |

The line was clear on 77% of cells (median margin 2.02).

### Which stored tables had actually been scored with the ratchet? None.

No table recorded its rule, and for two the producing script no longer existed, so the
rule was read off the numbers: **a ratchet cannot decrease with time; a standing count
can.** The laboratory's hand counts are the control — non-decreasing in 22% of series.

| table | non-decreasing | rule |
|---|---|---|
| `paraquat_ground_truth` (hand, control) | 22% | protocol |
| `paraquat_final` | 2% | `net` |
| `headtohead_v2` / `headtohead_simple` | 8–17% | `net` |
| `day7_results` / `day7_layout_results` | 17% / 0% | `net` |
| `backend_remeasure` | 22–28% | `net` |

Every analysis script passed `mode="net"` explicitly. **The ratchet defect of §9.2 was
confined to the interface**, so no scripted result was affected by it.

`paraquat_final.csv` agrees with the re-score on 648 shared cells: identical in 64%,
median difference 0, max 4, MAE 5.42 against the re-score's 5.37. The two differ only
in the scoring-line rule. It is superseded, not wrong.

### Removed

Recorded with sizes and hashes in `DELETED_incorrect_outputs.json`:

- **`paraquat_drawn_results.csv`** — all four series report a frozen count (11 at all
  twelve timepoints) where the verified re-score of the same vial gives
  9, 14, 12, 7, 7, 6, 5, 10, 7, 7, 5, 6. A count that cannot move over two minutes is
  not a measurement.
- **`batch15_climbing_table.csv`, `batch15_qc.csv`, `batch15_excluded.csv`** — the
  denominator changes across timepoints in **24 of 24 sealed vials**, so every PI is
  wrong, and Genotype/Treatment/Sex are all `UNSPECIFIED`.

Hand counts, manifests and the raw recordings were not touched.

### The open question this exposed, which is not a software defect

`crossing_rule.csv` is the experiment that originally chose the ratchet. Read properly
it says something uncomfortable:

| rule | bias | MAE | within-series *r* vs hand |
|---|---|---:|---:|
| `instant` (standing count) | **−6.21** | 6.39 | −0.08 |
| `runmax` (ratchet) | −0.99 | **3.16** | **+0.69** |

The ratchet matches the hand counts about twice as well. Part of that is arithmetic:
the detector under-counts by roughly six flies, and a count that only ever accumulates
upward offsets a shortfall that only ever runs downward. But the ratchet also *tracks*
the hand counts within a series where the standing count does not (+0.69 against
−0.08, on six usable series), and bias cancellation alone does not explain that.

The re-score shows the same shortfall directly — mean detected count sits near 10
while the hand mean sits near 13.6 from 20 s onward.

**`net` is what the protocol specifies and what the software now does consistently.**
But the honest reading is that the ratchet was concealing a detector that misses about
a quarter of the animals. Selecting the rule that hides the shortfall is not a fix;
closing the shortfall is. That is the outstanding work on this module, and no accuracy
claim should be made for it until it is done.

`crossing_rule.csv` is deliberately kept: its inference was wrong, its measurements
are not, and it is the evidence for this question.

---

## 11. The detector rebuilt: find blobs the size of a fly, 2026-08-21

### Why thresholding could not be tuned into working

Adaptive thresholding asks of every pixel *is this darker than its neighbourhood?*
A printed graduation, a numeral, a scratch and a fly all answer yes. It then groups
surviving pixels with connected components, which welds touching flies into a single
object — worst exactly when the cohort is densest, which is when the 10-second score
is taken. Neither failure is about the threshold, so neither could be tuned away.

Measured on this laboratory's own footage, that detector reported **59 objects where
20 flies were present** on a graduated cylinder, and missed ~6 flies per frame on the
paraquat set.

### The method, and its source

The particle-tracking literature settled this for colloids in the 1990s (Crocker &
Grier), and **FreeClimber** (Spierer et al. 2021, *J Exp Biol* 224:jeb229377) brought
it to this exact assay: it hands frames to `trackpy` and filters candidate spots on
expected diameter, roundness and integrated brightness.

`modules/fly_blob.py` asks *is there a blob about one fly wide centred here?*

1. **Band-pass at the fly's scale.** Invert, blur at `σ = d/(2√2)` — the scale at
   which a blob of diameter `d` responds most strongly — and subtract a boxcar
   estimate of the local background over `2d+1`. Illumination gradients (long
   wavelength) and sensor noise (short wavelength) both vanish.
2. **Local maxima, not connected components.** A pixel is a candidate only if nothing
   stronger lies within one fly width.
3. **Mass and shape.** Integrated response over a disc, and eccentricity from the
   brightness-weighted second moments. A printed line is elongated and fails on shape.
4. **The empty tube is the null distribution.** The temporal median holds the
   graduations, numerals, scratches and noise, and no flies, so a high quantile of its
   band-pass response is the detection threshold — a chosen false-positive rate rather
   than a magic number.

### Results

Fitted on 19 vials, tested on 22 held-out vials from separate recordings, all six
conditions, 110 cells:

| target | `blob` | shipped `threshold` |
|---|---:|---:|
| flies loaded (sealed vial) | **2.69** | 7.74 |
| above-the-line hand count | **3.53** | 5.42 |
| within 3 flies (total) | **67.3%** | 32.7% |

**This partly resolves §10's open question.** The ratchet beat the standing count
against hand counts (3.16 vs 6.39), and §10 argued some of that was bias cancellation
— the detector missing ~6 flies while a ratchet only accumulates upward. With the
detector fixed, the **instantaneous** count reaches MAE 3.53, essentially the
ratchet's 3.16 and at the assay's own 3-fly repeat-tap floor. The ratchet's advantage
was largely covering for a detector that missed flies.

### Four things tried and rejected, with numbers

| approach | why it failed here | measured |
|---|---|---|
| temporal background subtraction | FreeClimber's own step. It measures *velocity in the first seconds* while flies move; we need a *count at 120 s* when flies cling motionless, so the median **contains** them and subtracting deletes them | MAE 14.97 vs 7.73; fitted weight 0 |
| automatic scale selection (Lindeberg) | selected radius came back a constant 0.0198 × box width for every vial — pinned to the σ floor, because small-scale noise wins the argmax | 3.61 vs 2.48 |
| rejecting marks by location | helps a detector finding noise (25.57 → 6.30) but with the band-pass correctly scaled it eats real flies | retained behind a flag, off |
| thresholding off the frame maximum | one very dark mark suppresses every fly in the vial | cost ~5 flies/frame |

Two bugs of my own are recorded in the module: the noise floor was computed *after*
clipping, so its MAD was identically zero and it never bound; and the shape window was
initially too small — the band-pass blurs a 3 px line into a ~10 px ridge, which reads
as eccentricity 1.75 inside a 0.6d patch and slipped under a 1.8 limit. At 1.0d the
line reads 2.9 and a fly sitting *on* the line 2.1, so the limit is 2.8.

### Limits, stated rather than left to be found

- **Touching flies still merge** below ~1.2 fly-widths. Local maxima *can* split them,
  but the fitted separation chose 1.2d because splitting closer pairs cost more in
  false positives than it recovered (0.6d → 3.66, 0.8d → 3.44, 1.2d → 2.93). The count
  is biased low when the cohort is dense.
- A one-pixel speck is detected — a matched filter responds to point sources — and is
  separated only by mass (1/20th of a fly's).
- An object several times fly-size fragments into rim detections (22 from 10 drawn).
- Ten graduation ticks leak 2 detections; 80% rejected, not all.
- The shape filter costs 0.26 flies of total accuracy (2.69 vs 2.43). It is enabled
  anyway because the false positives it removes sit **on the scoring line**.

### A behavioural change in the recommender

Cluttered **and** still footage was refused outright — correct when the only detectors
were a threshold (defeated by clutter) and background subtraction (defeated by
stillness). `blob` is defeated by neither and was fitted on exactly that footage, so it
now returns a measurement. A **moved camera** still refuses, because it corrupts the
temporal median `blob` draws its threshold from. The test guarding the old behaviour
was inverted rather than deleted, so the change of contract is visible in the suite.

### Not fixed by this

17 of 58 vials had to be excluded before fitting, because the automatic vial *finder*
put a box across most of the frame (aspect 1.32 against 4.60 for a real cylinder).
That is a geometry failure, not a detection one — and 28% of automatic geometry
failing is precisely why the shipped tool has the operator draw the boxes.

Validated by `validation/validate_fly_blob.py`.
