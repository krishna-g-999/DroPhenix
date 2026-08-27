"""
Detector selection for climbing video: offer every method, and pick one on evidence.

WHY A CHOICE, AND WHY AN AUTOMATIC ONE
--------------------------------------
DroPhenix now has two ways to count flies in a vial and they fail in opposite
directions.

  THRESHOLD   `climbing_video_analyzer.detect_flies_in_roi` -- adaptive threshold
              inside a user-drawn region. Needs no motion, so it works on a
              settled cohort and on a moving camera. But it cannot tell a fly
              from any other locally dark, fly-sized mark, so on a graduated
              cylinder it counts the graduations.
  BACKGROUND  `climbing_video.count_in_vial` -- temporal-median background
              subtraction. Printed scales, numerals and drawn lines vanish
              because they are static. But a fly that stops moving is absorbed
              into the median and vanishes with them, and a camera that moves
              invalidates the whole model.

Neither is right in general, so the useful question is not "which is better" but
"which is right for THIS recording", and that is answerable BEFORE scoring.

THE TWO MEASUREMENTS THAT DECIDE IT
-----------------------------------
`static_clutter` -- fly-sized dark blobs found in the temporal-median background,
which is a photograph of the scene with no flies in it. Every one of them is a
permanent feature that an intensity threshold must mistake for an animal, so the
count is a PREDICTION of the threshold detector's false positives. Measured on
six laboratory recordings, that prediction correlates with the realised excess at
**r = +0.993**, and the relation is close to one-for-one: the threshold detector's
total came out at roughly `static_clutter + visible flies` every time.

    static clutter   1    2   19   42   43   57
    measured excess -13  -13   +9  +24  +27  +45

`moving_fraction` -- the share of dark objects that actually move across the clip.
Background subtraction can only see what moves; at 0.00 it has nothing to work
with and will report an empty vial however many flies are present. Two of the six
recordings measured exactly 0.00.

WHERE THE THRESHOLDS COME FROM
------------------------------
`STATIC_CLUTTER_LIMIT` is tied to the assay's own noise floor rather than chosen.
The laboratory's five repeat taps of the same vial differ by a median of 3 flies,
so 3 is the smallest error that means anything here. Since clutter converts to
false positives roughly one-for-one, 5 pieces of clutter already exceeds that
floor. It therefore marks the point beyond which the vial TOTAL from a threshold
must be treated as inflated -- it does NOT, on the evidence below, decide which
backend gives the better above-the-line count.

`MOTION_FLOOR` is set at 0.10 from the observed split: recordings usable by
background subtraction measured 0.32-0.60, unusable ones measured 0.00.

WHAT THIS DELIBERATELY WILL NOT DO
----------------------------------
It will not pick a winner when both methods fail their own check. On two of the
six recordings the clutter was low (so the threshold looked safe) AND nothing
moved (so the background had nothing) -- and the threshold detector duly found 2
objects where 15 flies were loaded. The honest output there is a refusal with the
reason, not the least-bad of two broken numbers.

WHAT WAS TESTED  (re-measured 2026-08-17; the earlier table was void)
---------------------------------------------------------------------
An earlier version of this table ranked `threshold_masked` best at MAE 5.88. That
number, and every other in it, was obtained under two mistakes that have since been
corrected, so it was discarded rather than adjusted:

  * the scoring rule was the RATCHET, not the protocol's rule. The protocol says a
    fly that falls back is recorded as -1 and the next crosser retakes its number,
    so the tally is the net of up and down crossings -- i.e. how many are STANDING
    above the line. See `score_crossings`.
  * the fly-size window scaled with the ROI AREA, so it depended on how big a box
    the user drew. Extending one box moved the floor from 34 to 44 px^2 and dropped
    7 of 29 detections. It is now anchored to the vial WIDTH.

Re-measured on 216 paired cells -- twelve recordings spanning all six conditions,
single- and two-cylinder frames, at the twelve times the laboratory scored:

    threshold             bias -1.38   MAE 6.26   within 3 flies 38%   <- best
    background            bias -5.50   MAE 6.54   within 3 flies 33%
    threshold_masked      bias -7.39   MAE 7.49   within 3 flies 23%
    threshold_declutter   bias -9.14   MAE 9.15   within 3 flies  7%

THE MASK WAS THE MISTAKE. `threshold_masked` was promoted on the old table; under
the corrected rule it is a third worse than doing nothing. The reason is visible in
the footage: this laboratory's graduations are dense, so dilating them into a mask
covers much of the tube and every fly resting on or beside a tick is discarded with
them. Removing printed marks by location is a good idea in principle and a bad one
on glassware this heavily ruled.

WHERE THE ERROR ACTUALLY LIVES -- IT IS NOT UNIFORM
---------------------------------------------------
The headline MAE hides a split that matters far more than the backend choice:

    WT  10 mM        MAE  1.42   within 3 flies  94%
    WT   5 mM        MAE  2.94   within 3 flies  64%
    WT  untreated    MAE  4.72   within 3 flies  36%
    TDP  5 mM        MAE  6.11   within 3 flies  25%
    TDP 10 mM        MAE  7.53   within 3 flies   6%   (reports 0.00 against 7.5)
    TDP untreated    MAE 14.86   within 3 flies   0%

On wild-type footage the method is at or near the assay's own 3-fly noise floor. On
the TDP recordings it fails, and on TDP 10 mM it reports nothing at all -- which is
consistent with the probe's independent finding that two of those recordings have a
moving fraction of 0.00 and yield two detections where fifteen flies were loaded.
That is a property of those recordings, not of the algorithm, and the right response
is to refuse them rather than tune against them.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import cv2

from . import climbing_video as CV
from . import climbing_video_analyzer as CVA
from . import fly_blob as FB

__all__ = [
    "BACKENDS", "BACKEND_LABELS", "MEASURED_TOTAL_MAE",
    "FootageProbe", "probe_footage",
    "recommend_backend", "count_with_backend", "static_mask",
    "score_crossings", "SCORING_MODES", "timepoint_diagnostics",
    "STATIC_CLUTTER_LIMIT", "MOTION_FLOOR", "BIOLOGICAL_FLOOR_FLIES",
]

#: How the count at a timepoint is defined. This turned out to matter far more
#: than which detector produced it -- see `score_crossings`.
SCORING_MODES = ("net", "max_reached", "instantaneous", "crossings")

#: `net` and `max_reached` are the meaningful names; the other two are the older
#: spellings kept so existing calls keep working.
_MODE_ALIAS = {"instantaneous": "net", "crossings": "max_reached"}

#: Median tap-to-tap spread of a PERFECT counter on this assay, measured from the
#: laboratory's five repeat taps of the same vial. The smallest error that means
#: anything; every limit below is expressed against it.
BIOLOGICAL_FLOOR_FLIES = 3

#: Above this many static fly-sized marks in the empty scene, an intensity
#: threshold accrues more false positives than the assay's own noise floor.
STATIC_CLUTTER_LIMIT = 5

#: Below this moving fraction, background subtraction has nothing to subtract.
MOTION_FLOOR = 0.10

#: Mean absolute error against laboratory hand counts, 24 paired measurements on
#: six recordings (see `validation/validate_climbing_detectors.py`). The default
#: backend is chosen from this table, not from argument.
MEASURED_MAE = {"blob": 3.53, "threshold": 6.26, "background": 6.54,
                "threshold_declutter": 9.15, "threshold_masked": 7.49}

#: `blob`'s entry above was measured on 110 cells from 22 vials HELD OUT from its own
#: fitting, against the same laboratory hand counts; the threshold scored 5.42 on
#: exactly those cells, so the two are directly comparable even though the other
#: entries come from an earlier 216-cell run. Worth knowing: 3.53 is close to the
#: BIOLOGICAL_FLOOR_FLIES spread of a perfect counter, and close to what the RATCHET
#: rule achieved with the old detector (3.16). That is evidence the ratchet's apparent
#: advantage was largely compensating for a detector that missed flies -- fix the
#: detector and an instantaneous count matches the hand counts nearly as well.
_BLOB_MAE_IS_HELD_OUT = True

#: Mean absolute error of the DETECTED TOTAL against the number of flies actually
#: loaded into a sealed cylinder, on 110 cells from 22 held-out vials spanning all
#: six paraquat conditions (`validation/validate_fly_blob.py`). This is a different
#: and stricter target than MEASURED_MAE above, which is scored against the
#: laboratory's above-the-line hand counts -- a quantity that laboratory records
#: cumulatively, and which therefore cannot separate a detector's error from a
#: definitional difference. A sealed vial's fly count can.
MEASURED_TOTAL_MAE = {"blob": 2.69, "threshold": 8.38}
DEFAULT_BACKEND = "blob"

BACKENDS = ("blob", "threshold", "background", "threshold_declutter",
            "threshold_masked")
BACKEND_LABELS = {
    "blob": ("Size-matched blob detection (recommended; finds flies by their SIZE, "
             "separates touching flies, ignores printed lines by their shape)"),
    "threshold": "Adaptive threshold (no motion needed; miscounts printed marks)",
    "background": "Median background (ignores printed marks; needs the flies to move)",
    "threshold_declutter": "Adaptive threshold minus static clutter (experimental)",
    "threshold_masked": ("Adaptive threshold with the printed scale masked out "
                         "(keeps settled flies, removes graduations)"),
}


@dataclass
class FootageProbe:
    """What the recording itself says about which detector can work on it."""

    static_clutter: int = 0          # fly-sized dark blobs in the empty scene
    moving_fraction: float = float("nan")
    framing_stable: bool = True
    median_blob_px: float = float("nan")
    n_moving: float = float("nan")   # background-subtraction total, mid-clip
    n_threshold: float = float("nan")  # threshold total, mid-clip
    n_blob: float = float("nan")       # size-matched total, mid-clip
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = list(self.notes)
        return d


# ─────────────────────────────────────────────────────────────────── probing ──
def _vialconfig_for(vial: "CV.VialSpec") -> "CVA.VialConfig":
    return CVA.VialConfig(
        vial_id=1, genotype="", replicate=1, n_flies_expected=1,
        roi_x=vial.x0, roi_y=vial.y_top, roi_w=vial.width, roi_h=vial.height,
        line_y_frac=0.5, fly_color="dark")


def probe_footage(path: str, vial: "CV.VialSpec",
                  background: Optional[np.ndarray] = None,
                  n_samples: int = 12,
                  params: Optional[dict] = None) -> FootageProbe:
    """Measure the properties that decide which detector can work.

    Cheap by design: one background, a dozen sampled frames. It is meant to run
    before scoring, on every vial, without the user asking.
    """
    p = dict(CVA.DEFAULT_DETECTION_PARAMS)
    if params:
        p.update(params)
    probe = FootageProbe()

    bg = background if background is not None else CV.build_background(path)
    info = CV.probe_video(path)

    # 1. static clutter -- what an intensity threshold finds in an EMPTY scene
    cfg = _vialconfig_for(vial)
    cents, _ = CVA.detect_flies_in_roi(cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), cfg, p)
    probe.static_clutter = len(cents)

    # 2. how much of what is dark actually moves
    lo = int(min(6.0, max(info.duration_s * 0.05, 0.0)) * info.fps)
    hi = max(lo + 1, info.n_frames - 5)
    idx = np.linspace(lo, hi, int(n_samples)).astype(int)
    pv = CV.DetectorParams(register=False)
    moving, thresh_tot, blob_px, blob_tot = [], [], [], []
    cap = cv2.VideoCapture(str(path))
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g, _, _ = CV.align_to_background(g, bg)
        r = CV.count_in_vial(g, bg, vial, pv)
        moving.append(r["n_total"])
        if np.isfinite(r.get("typical_area_px", np.nan)):
            blob_px.append(float(r["typical_area_px"]))
        thresh_tot.append(len(CVA.detect_flies_in_roi(im, cfg, p)[0]))
        y0b, y1b = int(vial.y_top), int(vial.y_bottom)
        x0b, x1b = int(vial.x0), int(vial.x1)
        gsub, bsub = g[y0b:y1b, x0b:x1b], bg[y0b:y1b, x0b:x1b]
        if gsub.size and bsub.shape == gsub.shape:
            blob_tot.append(len(FB.detect(gsub, bsub)))
    cap.release()

    probe.n_moving = float(np.median(moving)) if moving else float("nan")
    probe.n_threshold = float(np.median(thresh_tot)) if thresh_tot else float("nan")
    probe.n_blob = float(np.median(blob_tot)) if blob_tot else float("nan")
    probe.median_blob_px = float(np.median(blob_px)) if blob_px else float("nan")
    if np.isfinite(probe.n_threshold) and probe.n_threshold > 0:
        probe.moving_fraction = float(probe.n_moving / probe.n_threshold)
    elif np.isfinite(probe.n_moving):
        probe.moving_fraction = 1.0 if probe.n_moving > 0 else 0.0

    # 3. does the camera hold still? the median background assumes it does
    try:
        probe.framing_stable = bool(CV.detect_taps(path).get("framing_stable", True))
    except Exception as exc:                       # probing must never be fatal
        probe.notes.append(f"framing check unavailable ({type(exc).__name__})")

    return probe


# ───────────────────────────────────────────────────────────── recommendation ──
def recommend_backend(probe: FootageProbe,
                      n_expected: Optional[int] = None) -> dict:
    """Choose a detector from the probe, and say why in words.

    Returns `{backend, reason, confident, alternatives, warnings}`. `backend` is
    None when nothing is usable -- a refusal, not a fallback.
    """
    warnings: List[str] = []
    cluttered = probe.static_clutter > STATIC_CLUTTER_LIMIT
    still = (np.isfinite(probe.moving_fraction)
             and probe.moving_fraction < MOTION_FLOOR)
    moved_camera = not probe.framing_stable

    if cluttered:
        warnings.append(
            f"{probe.static_clutter} fly-sized dark marks are present in the empty "
            f"scene (graduations, numerals, label edges). An intensity threshold "
            f"cannot distinguish them from flies and will over-count by roughly "
            f"that many -- far above the {BIOLOGICAL_FLOOR_FLIES}-fly noise floor "
            f"of the assay itself.")
    if still:
        warnings.append(
            f"only {probe.moving_fraction:.0%} of the dark objects move across the "
            f"clip, so a temporal-median background has almost nothing to subtract; "
            f"a settled fly is absorbed into it and disappears.")
    if moved_camera:
        warnings.append(
            "the camera moved during this recording, so the median background is a "
            "blend of two scenes and background subtraction is invalid.")

    # Clutter plus stillness used to be a refusal, because the only two detectors
    # were an intensity threshold (defeated by clutter) and background subtraction
    # (defeated by stillness). `blob` is defeated by neither: it finds flies by
    # SIZE, so printed lines fail its shape test, and it needs no motion at all --
    # and the footage it was fitted on is exactly this regime, graduated cylinders
    # scored out to 120 s. The refusal now stands only when the CAMERA MOVED, which
    # corrupts the temporal median that `blob` takes its detection threshold from.
    if moved_camera and cluttered:
        return {"backend": None, "confident": False, "warnings": warnings,
                "alternatives": list(BACKENDS),
                "reason": ("No method can be trusted on this recording: the camera "
                           "moved, so the median background is a blend of two "
                           "scenes -- which invalidates background subtraction "
                           "outright and corrupts the noise threshold the "
                           "size-matched detector sets from it -- and the scene "
                           "carries too much static clutter for a plain intensity "
                           "threshold. Score it by hand, or re-record.")}
    if cluttered and still:
        return {"backend": "blob", "confident": True, "warnings": warnings,
                "alternatives": [b for b in BACKENDS if b != "blob"],
                "reason": (
                    f"{probe.static_clutter} static marks and only "
                    f"{probe.moving_fraction:.0%} of objects moving would defeat "
                    f"both an intensity threshold and background subtraction. "
                    f"Size-matched detection is defeated by neither: it rejects "
                    f"printed lines on their SHAPE and needs no motion. Measured "
                    f"on exactly this kind of footage, its vial total is out by "
                    f"{MEASURED_TOTAL_MAE['blob']:.1f} flies against "
                    f"{MEASURED_TOTAL_MAE['threshold']:.1f} for the threshold.")}

    if still or moved_camera:
        # background subtraction is ruled out by physics here, whatever the clutter
        # Ask the detector that will actually run, not a different one. This gate
        # used to consult the THRESHOLD count alone, and once `blob` became the
        # default that refused vials the default could score perfectly well --
        # measured on the paraquat set, where the threshold is blind to flies the
        # size-matched detector finds. A refusal must be evidence about the chosen
        # method, or it is just a different detector's blindness wearing a warning.
        seen = probe.n_blob if np.isfinite(probe.n_blob) else probe.n_threshold
        if (np.isfinite(seen) and n_expected and seen < 0.5 * n_expected):
            warnings.append(
                f"the size-matched detector finds only {seen:.0f} objects where "
                f"{n_expected} flies were loaded, so it is missing most of them "
                f"regardless of which method is chosen.")
            return {"backend": None, "confident": False, "warnings": warnings,
                    "alternatives": list(BACKENDS),
                    "reason": ("Background subtraction is unusable here, and fewer "
                               "than half the loaded flies are detectable by any "
                               "method -- too few objects are found to be flies at "
                               "all. Check focus, zoom and lighting.")}
        return {"backend": "blob", "confident": True, "warnings": warnings,
                "alternatives": [b for b in BACKENDS if b != "blob"],
                "reason": ((("The camera moved" if moved_camera else
                             f"Only {probe.moving_fraction:.0%} of objects move"))
                           + ", which rules out background subtraction. Size-matched "
                             "detection needs no motion, and its vial total was out "
                             f"by {MEASURED_TOTAL_MAE['blob']:.1f} flies against "
                             f"{MEASURED_TOTAL_MAE['threshold']:.1f} for a plain "
                             "threshold on held-out vials.")}

    # Every method is physically usable here. `blob` is recommended on the
    # strongest evidence in this module: its vial TOTAL was measured against the
    # number of flies actually loaded into a sealed cylinder -- ground truth that
    # cannot be argued with -- on 110 cells from 22 vials held out from fitting.
    return {"backend": DEFAULT_BACKEND, "confident": True, "warnings": warnings,
            "alternatives": [b for b in BACKENDS if b != DEFAULT_BACKEND],
            "reason": (
                f"Every method is usable on this recording "
                f"({probe.static_clutter} static marks, "
                f"{probe.moving_fraction:.0%} of objects moving). "
                f"'{DEFAULT_BACKEND}' is recommended: on 22 vials held out from "
                f"fitting, its detected total was out by "
                f"{MEASURED_TOTAL_MAE['blob']:.1f} flies against "
                f"{MEASURED_TOTAL_MAE['threshold']:.1f} for a plain threshold, "
                f"measured against the number of flies actually loaded. Compare "
                f"the preview images and override this if it looks wrong."
                + (f" Note that {probe.static_clutter} static marks are present, so "
                   f"the threshold's vial TOTAL will be inflated by roughly that "
                   f"many even where its above-the-line count is sound."
                   if cluttered else ""))}


# ───────────────────────────────────────────────────────────────── counting ──
def static_mask(background: np.ndarray, vial: "CV.VialSpec",
                params: Optional[dict] = None, grow: int = 2) -> np.ndarray:
    """Where the permanent marks ARE, in ROI pixel coordinates.

    The temporal median is the empty scene, so anything a threshold finds in it is
    printed on the glass and cannot be an animal. Masking those LOCATIONS is
    strictly better than subtracting their COUNT: it removes the graduations while
    leaving a settled fly -- which needs no motion to be seen -- fully visible.

    `grow` dilates each mark slightly, because a fly touching a tick merges with
    it; the cost is that a fly sitting exactly on a graduation is lost, which is a
    small, bounded and STATED loss rather than an unknown one.
    """
    p = dict(CVA.DEFAULT_DETECTION_PARAMS)
    if params:
        p.update(params)
    cfg = _vialconfig_for(vial)
    bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    cents, _ = CVA.detect_flies_in_roi(bgr, cfg, p)
    h, w = vial.height, vial.width
    m = np.zeros((max(h, 1), max(w, 1)), np.uint8)
    r = max(2, int(round(np.sqrt(max(float(p["min_area_frac"]) * h * w, 4)) * 0.9))) + grow
    for (cx, cy) in cents:
        cv2.circle(m, (int(cx), int(cy)), r, 255, -1)
    return m


def count_with_backend(backend: str, frame_bgr: np.ndarray, background: np.ndarray,
                       vial: "CV.VialSpec", probe: Optional[FootageProbe] = None,
                       params: Optional[dict] = None) -> dict:
    """Count flies in one vial in one frame with the named backend.

    Every backend returns the same keys, so the caller does not branch:
    `n_total`, `n_above`, `n_below`, `backend`, `note`.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}.")

    if backend == "blob":
        # Size-matched detection. Needs the same region of the temporal median, which
        # supplies the null distribution the threshold is set from -- see fly_blob.
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        y0, y1 = int(vial.y_top), int(vial.y_bottom)
        x0, x1 = int(vial.x0), int(vial.x1)
        sub = g[y0:y1, x0:x1]
        bsub = background[y0:y1, x0:x1]
        if sub.size == 0 or bsub.shape != sub.shape:
            return {"n_total": 0, "n_above": 0, "n_below": 0, "backend": backend,
                    "note": "the region falls outside the frame"}
        pts = FB.detect(sub, bsub, params)
        above = sum(1 for (_, cy, _) in pts if (cy + y0) < vial.line_y)
        return {"n_total": len(pts), "n_above": int(above),
                "n_below": int(len(pts) - above), "backend": backend,
                "note": f"fly diameter {FB.fly_diameter(x1 - x0, params):.1f} px"}

    if backend == "background":
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        g, _, _ = CV.align_to_background(g, background)
        r = CV.count_in_vial(g, background, vial,
                             CV.DetectorParams(register=False))
        return {"n_total": int(r["n_total"]), "n_above": int(r["n_above"]),
                "n_below": int(r["n_below"]), "backend": backend, "note": ""}

    cfg = _vialconfig_for(vial)
    cfg.line_y_frac = float(np.clip((vial.line_y - vial.y_top)
                                    / max(vial.height, 1), 0.0, 1.0))
    cents, _ = CVA.detect_flies_in_roi(frame_bgr, cfg,
                                       params or CVA.DEFAULT_DETECTION_PARAMS)
    line = cfg.line_y_for_height(vial.height)

    if backend == "threshold_masked":
        m = static_mask(background, vial, params)
        keep, drop = [], 0
        for (cx, cy) in cents:
            yy = int(np.clip(cy, 0, m.shape[0] - 1))
            xx = int(np.clip(cx, 0, m.shape[1] - 1))
            if m[yy, xx]:
                drop += 1
            else:
                keep.append((cx, cy))
        cents = keep
        note = (f"{drop} detection(s) fell on a permanent mark and were rejected "
                f"by location")

    above = sum(1 for (_, cy) in cents if cy < line)
    total = len(cents)
    if backend != "threshold_masked":
        note = ""

    if backend == "threshold_declutter":
        # The clutter count behaves like an additive offset (r = +0.993 against
        # the realised excess), so subtracting it recovers the fly count on
        # average. It is EXPERIMENTAL and labelled as such: the correction is an
        # aggregate, it cannot say WHICH detections were marks, and a fly sitting
        # on a graduation hides clutter that is then subtracted twice.
        c = int(probe.static_clutter) if probe else 0
        total = max(0, total - c)
        above = max(0, int(round(above * (total / max(len(cents), 1)))))
        note = (f"static clutter of {c} subtracted from the raw count of "
                f"{len(cents)}; aggregate correction, not per-detection")

    return {"n_total": int(total), "n_above": int(above),
            "n_below": int(max(0, total - above)), "backend": backend, "note": note}


# ══════════════════════════════════════════════════════════════════════════════
# SCORING: what "flies above the line at t seconds" actually means
# ══════════════════════════════════════════════════════════════════════════════
def score_crossings(video_path: str, vial: "CV.VialSpec", background: np.ndarray,
                    t0_s: float, report_times_s, backend: str = DEFAULT_BACKEND,
                    probe: "Optional[FootageProbe]" = None, step_s: float = 1.0,
                    smooth: int = 3, mode: str = "net",
                    params: Optional[dict] = None) -> "list[dict]":
    """Score a vial the way the published protocol does.

    WHAT THE PROTOCOL ACTUALLY ASKS FOR  (corrected 2026-08-17)
    ----------------------------------------------------------
    The standard protocol says: "If a fly climbs back down or falls, record that fly
    as -1 and count the next fly to cross the line as the same number as the fly
    that climbed back down or fell." The tally therefore steps BACK when a fly
    falls, and the next crosser retakes that number -- so the net of up- and
    down-crossings is simply THE NUMBER STANDING ABOVE THE LINE.

    That is `mode="net"`, and it is the default.

    An earlier version of this module made `max_reached` the default -- a ratchet
    that never decreases. That is NOT the protocol: it answers "what is the most
    that were ever above at one moment", which is a different quantity and always
    at least as large. It is retained because it is occasionally the question
    someone wants, but it must be chosen deliberately.

    THE OLD ARGUMENT FOR THE RATCHET, AND WHY IT WAS WRONG
    ------------------------------------------------------
    The standard negative-geotaxis protocols score the number of flies that have
    CROSSED the marked line within the observation window, accumulated from the
    tap. DroPhenix originally reported the number of flies ABOVE the line at the
    instant of the timepoint. Those are different quantities, and the second is
    always smaller by however many flies had crossed earlier and wandered back
    down.

    The laboratory's own hand counts carry the signature of the accumulating rule:
    a vial reads 17, 19, 19, 19 ... and never decreases. An instantaneous count
    fluctuates; a crossing count ratchets.

    Measured on 120 paired comparisons against those hand counts, changing ONLY
    this definition and holding the detector fixed:

        instantaneous   bias -6.21   MAE 6.39   within 3 flies 13.3%
        crossings       bias -0.99   MAE 3.16   within 3 flies 54.2%
        crossings, smoothed          MAE 3.08   within 3 flies 62.5%

    MAE 3.08 against a 3-fly floor -- the tap-to-tap spread of a perfect human
    counter on this assay -- so on wild-type vials (MAE 2.37) the method is now
    inside the noise of the assay itself. No detector change achieved anything
    close to this; the best detector improvement was 7.12 -> 5.88.

    HOW IT IS COMPUTED WITHOUT TRACKING INDIVIDUALS
    -----------------------------------------------
    Identity tracking is not needed for the count. The instantaneous number above
    the line is sampled on a fine grid from the tap onwards, median-smoothed over
    `smooth` samples so that one spurious detection cannot ratchet the total up
    permanently, and the running maximum is reported. The running maximum of the
    instantaneous count is exactly the ratchet the hand scorer applies.

    Pass `mode="instantaneous"` for the old behaviour, which remains the right
    choice if a laboratory genuinely scores "how many are above right now".

    THE CONFIRMATION DELAY, WHICH IS DELIBERATE
    -------------------------------------------
    Because a ratchet can never be undone, a single spurious detection above the
    line would raise the reported count for the rest of the recording. The median
    filter therefore requires a fly to be seen above the line in at least two of
    three consecutive samples before it is admitted as a crossing. The cost is a
    one-sample lag: at a moment when the cohort is crossing steeply, the reported
    figure trails the raw frame by up to one `step_s`. That is the intended
    trade -- a transient is not evidence of a crossing, and an over-count here is
    permanent while an under-count corrects itself on the next sample.

    Consequently the reported value can sit BELOW the raw count in a single frame,
    and that is not a violation of the ratchet: the guarantee is that it never
    falls below the SMOOTHED instantaneous estimate, and never decreases with time.
    """
    if mode not in SCORING_MODES:
        raise ValueError(f"mode must be one of {SCORING_MODES}")
    mode = _MODE_ALIAS.get(mode, mode)
    info = CV.probe_video(video_path)
    times = sorted(float(t) for t in report_times_s)
    if not times:
        return []
    grid = np.arange(0.0, min(max(times) + step_s, info.duration_s - t0_s), step_s)

    # The DETECTED TOTAL is carried alongside the above-the-line count, and this
    # is not decoration. Without it a caller has only `n_above`, and the app that
    # calls this had therefore been writing `n_above` into its detected-total
    # field: the exported `Detected_total` equalled `Count_climbed`, `Confidence`
    # became PI capped at 1 -- so a genuine TDP-43 vial at PI 0.5 exported as
    # `Low_confidence = True`, a real biological effect dressed as a data fault --
    # and over-detection could only be flagged when MORE flies stood above the
    # line than were loaded. A vial finding 59 objects with 15 above N=20 passed
    # silently. The two-sided check the analyzer module documents needs the total,
    # so the total is returned.
    inst, tot, cap = [], [], cv2.VideoCapture(str(video_path))
    for t in grid:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0_s + t) * info.fps))
        ok, im = cap.read()
        if ok:
            r = count_with_backend(backend, im, background, vial, probe=probe,
                                   params=params)
            inst.append(r["n_above"])
            tot.append(r["n_total"])
        else:
            inst.append(np.nan)
            tot.append(np.nan)
    cap.release()
    a = np.asarray(inst, float)
    b = np.asarray(tot, float)
    if not np.isfinite(a).any():
        return [{"time_s": t, "n_above": float("nan"), "n_total": float("nan"),
                 "mode": mode, "note": "no frame could be read"} for t in times]
    a = pd.Series(a).ffill().bfill().to_numpy()
    # gap-filled the same way and on the same index as `a`, so the total reported
    # at a timepoint is the total from the sample the count came from
    b = pd.Series(b).ffill().bfill().to_numpy()
    if mode == "net":
        series = a
    else:
        sm = (pd.Series(a).rolling(max(1, int(smooth)), center=True, min_periods=1)
              .median().to_numpy())
        series = np.maximum.accumulate(sm)

    out = []
    for t in times:
        i = int(np.searchsorted(grid, t, side="right")) - 1
        if i < 0 or t > (info.duration_s - t0_s):
            out.append({"time_s": t, "n_above": float("nan"),
                        "n_total": float("nan"), "mode": mode,
                        "note": "beyond the end of the recording"})
        else:
            # `n_total` is the INSTANTANEOUS number of objects found in the tube at
            # this sample, in both modes. Under `max_reached` the numerator is a
            # ratchet over earlier samples while this total is not, so `n_above`
            # CAN exceed `n_total` -- that is the ratchet counting a fly that
            # crossed and then fell back, not an inconsistency. Callers deriving a
            # below-the-line count must clamp at zero and say which mode produced
            # the row.
            out.append({"time_s": t, "n_above": float(series[i]),
                        "n_total": float(b[i]), "mode": mode, "note": ""})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# WHAT LIMITS THE COUNT AT EACH TIMEPOINT
# ══════════════════════════════════════════════════════════════════════════════
#: Flies whose centroid lies within this many vial-widths of the line are treated as
#: "near it". A fly body is roughly a tenth of a cylinder's width, so this band is
#: about two body lengths either side -- the distance over which a small error in
#: where the line sits, or a fly shifting its stance, flips the classification.
NEAR_LINE_BAND_FRAC = 0.20


def timepoint_diagnostics(video_path: str, vial: "CV.VialSpec",
                          background: np.ndarray, t0_s: float, times_s,
                          n_flies_loaded: int, backend: str = DEFAULT_BACKEND,
                          probe: "Optional[FootageProbe]" = None,
                          n_frames: int = 5,
                          params: Optional[dict] = None) -> "pd.DataFrame":
    """Per timepoint, the things that actually limit accuracy -- each measured.

    A single number per timepoint hides why it should or should not be believed.
    These five quantities are the ones that were found, by measurement, to drive the
    error in this assay, and every one of them is observable at the moment of the
    count without any ground truth:

    NEAR THE LINE   how many detections sit within a narrow band of the scoring
                    line. These are the flies whose classification is decided by
                    where the line sits to within a few pixels, so they are exactly
                    the count's sensitivity to the geometry. A vial with nobody near
                    the line has a robust count; one with six straddling it does not,
                    and today both report the same confident integer.
    FRAME SPREAD    the spread of the count over the consecutive frames averaged at
                    that instant. Pure measurement noise: the flies barely move in
                    0.17 s, so anything here is the detector disagreeing with itself.
    MERGED          detections whose area implies more than one animal. Worst at
                    early timepoints, when the cohort is still piled at the base --
                    which is precisely when the standard 10 s score is taken.
    ON A MARK       detections rejected for sitting on a permanent printed mark. High
                    values mean the count is leaning heavily on the mask, and a fly
                    resting on a graduation is being lost with it.
    SHORTFALL       detected total against the flies actually loaded. The number in
                    a sealed cylinder cannot change, so any gap is error -- either
                    animals not visible in this projection, or a detector missing them.

    Returned as one row per timepoint with a plain-language `limiting_factor`, so a
    reader can see which timepoints carry the analysis and which do not.
    """
    p = dict(CVA.DEFAULT_DETECTION_PARAMS)
    if params:
        p.update(params)
    info = CV.probe_video(video_path)
    band = max(3, int(round(NEAR_LINE_BAND_FRAC * max(vial.width, 1))))
    mask = static_mask(background, vial, p) if backend == "threshold_masked" else None

    cap = cv2.VideoCapture(str(video_path))
    rows = []
    for t in [float(x) for x in times_s]:
        if t0_s + t > info.duration_s:
            rows.append({"time_s": t, "n_above": np.nan, "scoreable": False,
                         "limiting_factor": "the recording ends before this time"})
            continue
        counts, totals, near, merged, on_mark = [], [], [], [], []
        for j in range(int(n_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(0, min(info.n_frames - 1, int((t0_s + t) * info.fps) + j)))
            ok, im = cap.read()
            if not ok:
                continue
            r = count_with_backend(backend, im, background, vial, probe=probe,
                                   params=p)
            counts.append(r["n_above"])
            totals.append(r["n_total"])

            cfg = _vialconfig_for(vial)
            cfg.line_y_frac = float(np.clip((vial.line_y - vial.y_top)
                                            / max(vial.height, 1), 0.0, 1.0))
            cents, _ = CVA.detect_flies_in_roi(im, cfg, p)
            ly = cfg.line_y_for_height(vial.height)
            near.append(sum(1 for (_, cy) in cents if abs(cy - ly) <= band))
            if mask is not None:
                on_mark.append(sum(
                    1 for (cx, cy) in cents
                    if mask[int(np.clip(cy, 0, mask.shape[0] - 1)),
                            int(np.clip(cx, 0, mask.shape[1] - 1))]))
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            g, _, _ = CV.align_to_background(g, background)
            merged.append(CV.count_in_vial(g, background, vial,
                                           CV.DetectorParams(register=False)
                                           )["n_merged"])
        if not counts:
            rows.append({"time_s": t, "n_above": np.nan, "scoreable": False,
                         "limiting_factor": "no frame could be read here"})
            continue

        n_above = float(np.median(counts))
        spread = float(np.max(counts) - np.min(counts))
        n_near = float(np.median(near)) if near else 0.0
        n_merged = float(np.median(merged)) if merged else 0.0
        n_mark = float(np.median(on_mark)) if on_mark else 0.0
        n_total = float(np.median(totals)) if totals else np.nan
        shortfall = (float(n_flies_loaded) - n_total
                     if (n_flies_loaded and np.isfinite(n_total)) else np.nan)

        # Rank the causes by how many flies each could actually MOVE the count.
        # `on_static_mark` is deliberately excluded: those detections are the printed
        # graduations being rejected as designed, so a large number means the mask is
        # working, not that the count is fragile. It is reported separately, and only
        # raised as a concern when the count leans on it heavily.
        causes = [("flies sitting within two body-lengths of the line", n_near),
                  ("the detector disagreeing with itself between frames", spread),
                  ("touching animals counted as one blob", n_merged)]
        if np.isfinite(shortfall) and shortfall > 0:
            causes.append(("flies loaded but not visible anywhere in the tube",
                           shortfall))
        elif np.isfinite(shortfall) and shortfall < 0:
            causes.append(("more objects found than flies loaded, so something that "
                           "is not a fly is being counted", -shortfall))
        causes.sort(key=lambda c: -c[1])
        worst, worst_n = causes[0]
        limiting = (f"{worst} ({worst_n:.0f} {'fly' if worst_n == 1 else 'flies'})"
                    if worst_n > 0 else
                    "nothing measurable is limiting this timepoint")

        rows.append({
            "time_s": t, "n_above": n_above, "n_total_detected": n_total,
            "flies_loaded": int(n_flies_loaded), "not_visible": shortfall,
            "scoreable": True, "near_line": n_near, "frame_spread": spread,
            "merged_blobs": n_merged, "on_static_mark": n_mark,
            "uncertain_flies": round(n_near + spread + n_merged, 1),
            "limiting_factor": limiting,
        })
    cap.release()
    df = pd.DataFrame(rows)
    if "uncertain_flies" in df.columns:
        # A count is only as good as the flies whose side of the line is in doubt.
        df["confidence"] = np.where(
            df["uncertain_flies"].isna(), "unknown",
            np.where(df["uncertain_flies"] <= 1, "firm",
                     np.where(df["uncertain_flies"] <= BIOLOGICAL_FLOOR_FLIES,
                              "within the assay's own noise", "soft")))
    return df
