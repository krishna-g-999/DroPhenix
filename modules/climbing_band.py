"""Count flies above a drawn band. Three functions, no cylinder model.

THE IDEA, AND WHY IT IS RIGHT
-----------------------------
On this rig the operator draws a black band round each cylinder at the 40 ml
graduation. That band is the widest dark horizontal feature in the frame: it
spans the full width of the tube, while a graduation tick spans a fraction of it
and printed characters have gaps between them. So the band can be found by
asking one question of each image row -- what fraction of this row is dark? --
and nothing else about the cylinder needs to be modelled.

This replaces, for this rig, a pipeline that located the glassware, fitted a row
of identical tubes, and then looked for the line inside that geometry. That
pipeline failed here in the way complicated geometry does fail: on Day 3 it put
the scoring line on the cotton plug and reported itself confident, agreeing with
the correct answer on 4% of the cylinders where it claimed `ok()`.

MEASURED, on 36 cylinders across 12 recordings, against a line verified by eye:

    row coverage > 0.60, dark < 80      found 21/36   81% within 15 px   median 4 px
    same but threshold relative to the
    cylinder's own brightness           found 24/36   58% within 15 px   median 4 px

The relative variant was tried because the absolute one refuses on 42% of
cylinders -- nearly all of them Day 3, whose band is genuinely fainter (depth
128 against 208 on Day 7). It finds three more bands and gets far more of them
wrong, so it is NOT used. It is recorded here because the refusal rate is a real
cost and someone will reasonably want to attack it again; attack it with a
better criterion, not with a looser threshold.

WHAT THE ZONE DOES AND DOES NOT FIX
-----------------------------------
Counting only a strip above the band removes, for free, every dark feature below
it and everything more than `zone_height_px` above it -- the cotton plug, the
tube rim, the hand-written R1/R2/R3.

It does NOT remove the printed BOROSIL specification block. Measured on the same
cylinders, the static dark clutter sits 1 to 232 px above the band, so a 250 px
zone contains it on 100% of cylinders. That block is the dominant false positive
on this footage: tracked across 110 s, a median of 3 detections per cylinder
never move, which is 31% of the reported above-the-line count, and removing them
shifts PI by about 19 points.

`persistence_filter` therefore exists and is not optional on labelled glassware.
A fly does not hold the same pixel for two minutes; printed text does. The cost
is stated: a fly that genuinely clings motionless for the whole window is lost
with it, so the filter is a parameter and its effect is always reported.

THE REAL FIX IS AT THE BENCH: turn the printed labels away from the camera.
No software can separate a 7 px printed character from a 7 px fly when both are
dark and in the same place.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

#: Distinguishes "caller passed None deliberately" from "caller passed nothing".
#: `assess_reading(validated_domain=None)` must mean "there is no validated
#: domain", which is a real and currently true state, not "use the default".
_SENTINEL = object()

import cv2
import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from . import fly_blob as FB

__all__ = ["find_black_band", "tube_width_from_band", "detect_crossings",
           "count_flies_above_band",
           "count_flies_below_band", "persistence_filter",
           "analyse_video", "pi", "BAND_DARK", "FLY_DARK"]

#: THE BAND AND THE FLIES NEED DIFFERENT THRESHOLDS, AND THIS IS NOT A DETAIL.
#: The drawn band is marker ink and is nearly black; the flies are dark bodies
#: against backlit glass that reads a median grey of 151. One threshold cannot
#: serve both. Measured on IMG_6181 P1 (wild-type male, flies demonstrably
#: climbing; the size-matched blob detector counts 18 above the line at 10-30 s):
#:
#:     fly threshold    t=10   t=20   t=30   t=60   t=120
#:               80        1      1      1      1       2   <- finds no flies
#:              120       13     12      7      9       8
#:              140       16     15     16     18      16   <- matches the reference
#:              160       23     22     28      0       2   <- unstable
#:              180        0      0      0      0       0   <- all one blob, area-filtered
#:
#: 140 is a sharp optimum, so it is a per-rig calibration and not a constant of
#: nature. Expose it, preview it, and re-check it on any new lighting.
BAND_DARK = 80
FLY_DARK = 140

#: THE COUNTING ZONE IS "ABOVE THE LINE", NOT "250 PIXELS ABOVE THE LINE".
#: A fixed 250 px was tuned on the Day-3/7 rig, where it happened to reach the
#: cotton plug. On the 15th-Day rig the camera is closer and band-to-plug is
#: ~400 px, so the good climbers sat above the zone and were never counted.
#: Measured on C0689 left (wild-type, hand count 17 at 60 s): the blob detector
#: over the whole tube finds 15 above the band; a 250 px zone keeps 2 of them,
#: 400 px keeps 15, 600 px keeps 15. There is no value to fit -- the protocol
#: counts everything above the line, so None means "to the top of the tube".
ZONE_FULL = None


# ────────────────────────────────────────────────── the validation manifest ──
#
# WHAT THIS IS FOR. Three times now a change has been evaluated on the same
# recordings that suggested it, and twice that produced a number that did not
# survive contact with a held-out set (development MAE 2.20 -> held-out 6.53).
# The manifest states, in one place that the code itself reads, which sets have
# been used for what -- so that an estimator cannot be re-chosen on a set that
# has already seen it, and so that the software knows whether it is entitled to
# report a biological result at all.
#
# IT IS NOT DOCUMENTATION. `assess_reading` consults `VALIDATED_DOMAIN`, and
# while that is None every Performance Index is NOT_REPORTABLE. Opening the gate
# is a deliberate edit here, made when the evidence below says it may be.

#: The estimator currently shipped. Do not change this without changing the
#: manifest, and do not change the manifest from a set listed as development.
ESTIMATOR = "bracket midpoint: (count_upper + count_lower) / 2"

#: FROZEN. Measured 2026-09-23, and this is the whole reason for the freeze:
#: the ranking of the three available estimators REVERSES between the two sets.
#:
#:   set        n     estimator   MAE    bias    slope (ideal 1.000)
#:   Day 15   432     upper      4.87   +2.39    0.272
#:   Day 15   432     midpoint   4.54   -1.63    0.199   <- shipped
#:   Day 15   432     lower      6.58   -5.64    0.125
#:   Day 3/7   30     upper      7.10   -0.03   -0.190
#:   Day 3/7   30     midpoint   6.53   -3.33    0.003   <- shipped
#:   Day 3/7   30     lower      7.23   -6.63    0.196
#:
#: On Day 15 `upper` has the best slope and the best retention of the genotype
#: contrast; on Day 3/7 it has the worst, and negative. Picking the Day-15
#: winner would be selecting a method on the set it was found in. The Day-3/7
#: slope CI is [-0.532, +0.539] at n = 30 -- wide enough to exclude a faithful
#: instrument but not wide enough to rank three candidates. The blocker is hand
#: counts, not code.
ESTIMATOR_SELECTION = "FROZEN"

#: Sets that have been used to choose or tune something. An estimator may never
#: be selected on these again, whatever they show.
DEVELOPMENT_SETS = (
    "Day 15 (C0665-C0698, 432 paired readings) -- used for the floor/slope "
    "decomposition and for generating the upper-bound hypothesis",
    "IMG_6029 + IMG_6181 (15 readings) -- used to choose the midpoint over the "
    "upper and lower bounds in the first place",
    "paraquat set (19 fitted + 22 held out vials) -- fly_blob's parameters",
)

#: Sets held for confirmation. Small, which is the current blocker.
CONFIRMATION_SETS = (
    "Day 3/7 hand counts (hand_count_holdout.csv + hand_count_template.csv, "
    "30 readings paired with a scored cylinder, 6 videos)",
)

#: What is actually known about the relationship between the reported count and
#: a hand count. Both slopes exclude 1.0; neither establishes a usable one.
VALIDATION_EVIDENCE = {
    "day15": {"n": 432, "n_cylinders": 36, "slope": 0.199,
              "slope_ci95": (0.154, 0.243), "intercept": 6.08,
              "mae": 4.54, "bias": -1.63,
              "empty_tube_floor_cyl_weighted": 5.62,
              "contrast_retained_frac": (0.20, 0.27)},
    "day37": {"n": 30, "n_videos": 6, "slope": 0.003,
              "slope_ci95": (-0.532, 0.539), "intercept": 12.98,
              "mae": 6.53, "bias": -3.33},
}

#: THE SET OF CONDITIONS UNDER WHICH A PERFORMANCE INDEX MAY BE REPORTED AS A
#: BIOLOGICAL RESULT. None means "no such conditions have been established".
#:
#: That is the honest state today and it is not a placeholder: on the only
#: independent set, the reported count's slope against the hand count is 0.003
#: with a 95% interval of [-0.532, +0.539]. An instrument whose reading may bear
#: no relation to the quantity it names cannot publish that quantity. The raw
#: observations and the unvalidated estimate are still returned -- see the three
#: layers in `assess_reading` -- so nothing diagnostic is hidden by this.
#:
#: To open the gate: expand the confirmation set (the near-term target is Day 7
#: male, 3 conditions x 10/30/60 s, plus the blank Set D rows), re-measure the
#: slope, and if its interval supports a stated relationship, replace this with
#: the domain it holds over.
VALIDATED_DOMAIN = None


def find_black_band(gray: np.ndarray, col_start: int, col_end: int,
                    dark_thresh: int = BAND_DARK,
                    min_coverage: float = 0.60) -> Optional[int]:
    """Row of the drawn band, or None when no row is dark enough across the tube.

    Returning None is the point. A guessed line silently relocates the scoring
    criterion, which is worse than scoring nothing.
    """
    c0, c1 = int(col_start), int(col_end)
    if gray is None or gray.ndim != 2 or c1 - c0 < 4:
        return None
    coverage = (gray[:, c0:c1] < int(dark_thresh)).mean(axis=1)
    rows = np.flatnonzero(coverage > float(min_coverage))
    if rows.size == 0:
        return None
    return int(np.median(rows))



CROSSING_HALF_BAND_PX = 30
CROSSING_MATCH_PX = 15


def detect_crossings(video_path: str, band_y: int, col_start: int, col_end: int,
                     background: np.ndarray,
                     half_band: int = CROSSING_HALF_BAND_PX,
                     max_match_px: int = CROSSING_MATCH_PX,
                     params: Optional[dict] = None) -> pd.DataFrame:
    """Count flies crossing the band, frame by frame, as events rather than states.

    WHY A GATE AND NOT A SNAPSHOT.
    Both static zones on this rig are contaminated, and differently. Above the
    band sits the printed specification block, which the size-matched detector
    still accepts: measured, a median of 3 detections per cylinder never move,
    31% of the above-count. Below the band the flies of an impaired cohort pack
    together, touching bodies merge into one blob, and the detector finds 1-6
    objects where 15-17 animals are -- which is why `complementary` read 12-19 on
    HTT Male cylinders whose true above-count is a handful. No area threshold
    fixes both: loosening it to split the crowd fragments the label, tightening it
    to reject the label loses flies.

    A gate is immune to both. A fly crossing the band is moving, is briefly
    separated from the cluster, and is alone at the boundary. Static print
    generates no events at all, however dark it is, because it never moves.

    Only a +/- `half_band` strip around the band is examined, so the label block
    and the crowded floor are never in frame.

    Returns frame_number, time_sec, crossing_up, crossing_down, net_crossings,
    n_in_gate -- cumulative, one row per frame read.

    CAVEAT, AND IT IS NOT SMALL: this counts CHANGE from the first frame. It is
    the number above the band only if the cohort started entirely below it. The
    operator taps before recording starts, so that is usually true -- but not
    always: IMG_6184 P3 already had flies above the band in frame 0. Callers
    should carry `initial_above` and check it.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("cannot open video: %s" % video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError("video reports no frame rate: %s" % video_path)

    # THE GATE SELECTS WHAT IS TRACKED, NOT WHAT THE DETECTOR MAY SEE.
    # Cropping to the +/- half_band strip before detection was tried and returns
    # NOTHING: `fly_blob` sets its threshold from the background's band-pass
    # response, and inside a 60-row strip that background is dominated by the
    # drawn band itself, so the null quantile goes sky-high and every fly is
    # rejected. Measured on IMG_6181 P1 at t=20 s: 0 detections from the strip,
    # against 30 over the whole tube of which 7 lay inside those same strip rows.
    # Context width matters and is not linear -- mean detections falling inside
    # the gate, per frame, over 100 frames:
    #     whole tube (1080 rows)   8.35   7.7 ms/frame   <- used
    #     band +/- 300 (600 rows)  7.31   4.6 ms/frame
    #     band +/- 150 (300 rows)  1.61   2.2 ms/frame   <- threshold contaminated
    # So detection runs on the full tube column and the gate is applied to the
    # RESULT. About 28 s per cylinder on a two-minute clip.
    c0, c1 = int(col_start), int(col_end)
    r0 = max(0, int(band_y) - int(half_band))
    r1 = min(background.shape[0], int(band_y) + int(half_band))
    bfull = background[:, c0:c1]

    # A BARE FRAME-TO-FRAME COMPARISON MANUFACTURES CROSSINGS, AND BADLY.
    # Measured on IMG_6181 P1, hand-counted at 13/15/12 flies above the band:
    # pairing each detection with its nearest neighbour in the previous frame and
    # counting any sign change gave net crossings of 49, 65 and 112 -- a PI of
    # 5.60, and 132 up-crossings from a cohort of 20. Two causes, both structural:
    #   1. a fly resting AT the band jitters a pixel either side of it, and at
    #      30 fps that is dozens of sign changes from one motionless animal;
    #   2. nearest-neighbour matching pairs a fly below the band with a DIFFERENT
    #      fly above it, inventing a crossing from two stationary animals.
    # So each track carries a CONFIRMED side, and that side changes only when the
    # centroid is clearly past the band by `margin`. Jitter inside the margin
    # changes nothing, and a mis-paired track must travel the full margin to score.
    margin = max(2, int(round(0.5 * FB.fly_diameter(c1 - c0, params))))

    # STATIC OBJECTS INSIDE THE GATE MUST BE REMOVED BEFORE MATCHING, NOT AFTER.
    # The gate is supposed to be immune to static clutter because print does not
    # move -- but that argument only holds if the clutter is not allowed to take
    # part in the matching. Measured on IMG_6181 P1, the +/-30 px strip held a
    # steady 7.59 detections in EVERY frame, at t=5 s and at t=120 s alike: a
    # 60-row strip cannot contain seven live flies for two minutes, and those are
    # graduation ticks and mottle in the drawn band. Nearest-neighbour matching
    # then pairs a genuinely climbing fly with a tick 15 px away, the track never
    # moves, and the crossing is never scored. The gate saw 1 up-crossing where
    # the hand count says 12 flies had climbed by t=10 s.
    # So: sample the clip, find the positions that are occupied nearly always, and
    # drop detections that land on them.
    _probe_idx = np.linspace(0, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) - 1, 0),
                             15).astype(int)
    _seen = []
    for _i in _probe_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(_i))
        _ok, _fr = cap.read()
        if not _ok:
            continue
        _g = cv2.cvtColor(_fr, cv2.COLOR_BGR2GRAY)[:, c0:c1]
        if _g.shape != bfull.shape:
            continue
        _seen.append([(float(x), float(y)) for (x, y, _m) in FB.detect(_g, bfull, params)
                      if r0 <= y < r1])
    static_pts = []
    if _seen:
        _need = 0.8 * len(_seen)
        for _pt in {pt for f_ in _seen for pt in f_}:
            _hits = sum(1 for f_ in _seen
                        if any((a - _pt[0]) ** 2 + (b - _pt[1]) ** 2 <= 25.0
                               for (a, b) in f_))
            if _hits >= _need:
                static_pts.append(_pt)
    static_tree = KDTree(np.array(static_pts, float)) if static_pts else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    up = down = 0
    tracks: List[dict] = []       # {"pos": (x, y), "side": above/below/None, "miss"}
    rows, fno = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sub = grey[:, c0:c1]
        if sub.shape == bfull.shape and sub.size:
            pts = FB.detect(sub, bfull, params)
            # keep only what is inside the gate; y is already a full-frame row
            cur = np.array([[float(x), float(y)] for (x, y, _m) in pts
                            if r0 <= y < r1], float).reshape(-1, 2)
            if static_tree is not None and len(cur):
                _d, _ = static_tree.query(cur)
                cur = cur[np.asarray(_d) > 5.0]
        else:
            cur = np.empty((0, 2), float)

        # KDTree, not a nested loop: this runs on every frame of every vial
        if tracks and len(cur):
            tpos = np.array([t["pos"] for t in tracks], float)
            dist, idx = KDTree(tpos).query(cur,
                                           distance_upper_bound=float(max_match_px))
        else:
            dist = np.full(len(cur), np.inf)
            idx = np.zeros(len(cur), int)
        claimed = set()
        for k in range(len(cur)):
            y = float(cur[k][1])
            if y < band_y - margin:
                side = "above"
            elif y > band_y + margin:
                side = "below"
            else:
                side = None                       # inside the margin: no opinion
            ti = int(idx[k]) if np.isfinite(dist[k]) else -1
            if ti >= 0 and ti not in claimed and ti < len(tracks):
                claimed.add(ti)
                tr = tracks[ti]
                tr["pos"] = (float(cur[k][0]), y)
                tr["miss"] = 0
                if side is not None and tr["side"] is not None and side != tr["side"]:
                    if side == "above":
                        up += 1
                    else:
                        down += 1
                if side is not None:
                    tr["side"] = side
            else:
                tracks.append({"pos": (float(cur[k][0]), y), "side": side,
                               "miss": 0})
        for n_, tr in enumerate(tracks):
            if n_ not in claimed:
                tr["miss"] += 1
        tracks = [t for t in tracks if t["miss"] <= 5]
        rows.append({"frame_number": fno, "time_sec": fno / fps,
                     "crossing_up": up, "crossing_down": down,
                     "net_crossings": up - down, "n_in_gate": int(len(cur))})
        fno += 1
    cap.release()
    return pd.DataFrame(rows)


def tube_width_from_band(background: np.ndarray, band_y: int,
                         col_start: int, col_end: int,
                         dark_thresh: int = BAND_DARK) -> Optional[Tuple[int, int]]:
    """The cylinder's true columns, measured from the drawn band itself.

    WHY THIS EXISTS. Both detectors scale the expected fly from the width of the
    region they are handed -- `fly_blob` as `fly_d_frac * width`, the adaptive
    path as `min_area_w2 * width**2` -- and the page asks the operator to type
    that region by hand. Measured on a 100 px cylinder holding 20 flies, with
    only the drawn box widening:

        box 100 px (the tube) -> 19 of 20 found, assumed fly  4.0 px
        box 140 px            -> 19 of 20,       assumed fly  5.6 px
        box 200 px            -> 19 of 20,       assumed fly  8.0 px
        box 300 px            ->  0 of 20,       assumed fly 12.0 px

    A box three times too wide reports an empty tube, silently. `fly_d_frac` is
    documented in `fly_blob` as sharply peaked (0.05 -> MAE 9.11, 0.07 -> 2.93,
    0.10 -> 9.82), so even a modest drawing error moves the detector off its
    calibrated optimum.

    The band is the measurement. It is drawn ROUND the cylinder, so at the band
    row the dark run IS the glass, and its extent is the tube width -- no extra
    edge detection, no new assumption. Returns (x0, x1) in FULL-FRAME columns, or
    None when no run is found, so the caller can refuse rather than guess.
    """
    c0, c1 = int(col_start), int(col_end)
    if background is None or background.ndim != 2 or c1 - c0 < 4:
        return None
    lo, hi = _band_rows(background, band_y, c0, c1, dark_thresh)
    strip = background[max(0, lo):min(background.shape[0], hi + 1), c0:c1]
    if strip.size == 0:
        return None
    # darkest value each column takes anywhere in the band's own rows: a
    # specular highlight breaks the band on one row but rarely on all of them
    dark = strip.min(axis=0) < int(dark_thresh)
    if not dark.any():
        return None
    # longest contiguous run -- the tube. Anything else is a shadow or a
    # neighbouring tube clipped by a generous box.
    best_len = best_start = run_len = 0
    run_start = None
    for i, d in enumerate(np.append(dark, False)):
        if d and run_start is None:
            run_start, run_len = i, 1
        elif d:
            run_len += 1
        elif run_start is not None:
            if run_len > best_len:
                best_len, best_start = run_len, run_start
            run_start, run_len = None, 0
    if best_len < 4:
        return None
    return c0 + best_start, c0 + best_start + best_len


def _band_rows(background: np.ndarray, band_y: int, col_start: int, col_end: int,
               dark_thresh: int = BAND_DARK,
               min_coverage: float = 0.60) -> Tuple[int, int]:
    """First and last row the drawn band itself occupies.

    The below-band zone must not start ON the band. The band is a thick marker
    ring -- 7 to 29 rows on this rig -- and the blob detector answers "yes, a
    fly-sized dark blob" to each mottle in it, which would land a row of false
    positives in the below-count and drag `complementary` upward.
    """
    c0, c1 = int(col_start), int(col_end)
    cov = (background[:, c0:c1] < int(dark_thresh)).mean(axis=1)
    lo = hi = int(band_y)
    while lo > 0 and cov[lo - 1] > min_coverage:
        lo -= 1
    while hi < len(cov) - 1 and cov[hi + 1] > min_coverage:
        hi += 1
    return lo, hi


def _blob_count(gray: np.ndarray, background: np.ndarray,
                r0: int, r1: int, col_start: int, col_end: int,
                params: Optional[dict] = None):
    """Size-matched blob detection in one row band. Shared by both counters.

    DETECTION RUNS ON THE WHOLE TUBE; THE ROW BAND ONLY SELECTS THE RESULT.
    Cropping to the band first is the obvious implementation and it is wrong, for
    the same reason it was wrong in `detect_crossings`: `fly_blob` sets its
    threshold from the background's band-pass response, and a crop dominated by
    the drawn band and the printed block gives a null quantile so high that live
    flies are rejected. Measured on IMG_6029 P1 with the crop in place -- a
    cylinder holding 20 flies with 3 above the band by hand count -- the
    below-band zone returned ZERO detections, so `complementary` read 20 - 0 = 20,
    and the above-zone returned 1 where the full-tube detector finds 6.

    The whole column is a representative null: it is mostly clear glass, so the
    quantile reflects the noise a fly must beat rather than the ink beside it.
    Centroids come back in ZONE coordinates, as before.
    """
    c0, c1 = int(col_start), int(col_end)
    r0, r1 = max(0, int(r0)), min(gray.shape[0], int(r1))
    if r1 <= r0 or c1 - c0 < 4:
        return 0, [], gray[r0:r1, c0:c1]
    full = gray[:, c0:c1]
    bfull = background[:, c0:c1]
    if bfull.shape != full.shape:
        return 0, [], gray[r0:r1, c0:c1]
    pts = FB.detect(full, bfull, params)
    cents = [(int(x), int(y) - r0) for (x, y, _m) in pts if r0 <= y < r1]
    return len(cents), cents, gray[r0:r1, c0:c1]


def count_flies_above_band(gray: np.ndarray, background: np.ndarray, band_y: int,
                           col_start: int, col_end: int,
                           zone_height_px: Optional[int] = ZONE_FULL,
                           params: Optional[dict] = None
                           ) -> Tuple[int, List[Tuple[int, int]], np.ndarray]:
    """Flies above the band -- the whole tube above it unless a height is given.

    THE PLAIN INTENSITY THRESHOLD THAT USED TO LIVE HERE IS GONE, AND THE
    MEASUREMENT THAT REMOVED IT IS WORTH KEEPING. Against 15 hand-counted
    readings on IMG_6029 and IMG_6181:

        simple threshold, above-count        MAE 11.93   bias +7.93
        complementary from the same counter  MAE  5.00   bias +0.07
        size-matched blob, instantaneous     MAE  3.53   bias +2.87

    The decisive case is IMG_6029 P1, which carries a dark circular mark near the
    rim. The threshold counted it as roughly thirty flies (35 against a hand count
    of 3, an error of +32); the blob detector, which asks whether a blob is the
    size and shape of a fly rather than merely dark, reads 6 (+3). No amount of
    zone trimming fixes the first because the mark is inside the zone by
    construction.

    `background` is the temporal median of the same recording -- the tube with no
    flies in it -- which the detector uses as its null distribution.
    """
    top = 0 if zone_height_px is None else max(0, int(band_y) - int(zone_height_px))
    return _blob_count(gray, background, top, int(band_y), col_start, col_end,
                       params)


def count_flies_below_band(gray: np.ndarray, background: np.ndarray, band_y: int,
                           col_start: int, col_end: int,
                           zone_height_px: Optional[int] = ZONE_FULL,
                           params: Optional[dict] = None
                           ) -> Tuple[int, List[Tuple[int, int]], np.ndarray]:
    """Flies below the band -- the whole tube below it unless a height is given.

    Counting the side nobody asked about, because it is often the easier side.
    Whichever side holds fewer animals can actually be resolved; the crowded side
    hides flies behind each other. The two also fail differently -- the printed
    specification block sits above the band on this rig and inflates only the
    above-count -- so their disagreement is informative, which is what
    `interval_width` reports.

    A fly in the cotton plug is invisible to the above-count and is correctly
    counted by the complementary estimate, because it is not below the line.

    The band's own rows are skipped; see `_band_rows`.
    """
    _lo, hi = _band_rows(background, band_y, col_start, col_end)
    r0 = hi + 1
    r1 = gray.shape[0] if zone_height_px is None else r0 + int(zone_height_px)
    return _blob_count(gray, background, r0, r1, col_start, col_end, params)


def persistence_filter(per_frame: Sequence[Sequence[Tuple[int, int]]],
                       bin_px: int = 4,
                       static_frac: float = 0.8) -> List[int]:
    """Counts with never-moving detections removed, one per input frame.

    A detection cell occupied in `static_frac` of the frames is printed text, a
    scratch, or a permanent mark -- not an animal. Positions are binned rather
    than matched pairwise; the pairwise form is O(n^2) per cylinder and does not
    finish on a two-minute clip.

    DO NOT TUNE `static_frac` OR `bin_px` UNTIL STATIONARY-FLY SENSITIVITY HAS
    BEEN MEASURED. This filter is working: above an empty tube it identifies 7.2
    of 8.0 detections as static, and the lower bound it produces reads 0.88
    where no fly is present. The temptation is therefore to lean on it harder.
    It must be resisted, because the filter cannot tell a permanent mark from a
    fly that is not moving, and on this assay those are not equally disposable:

        - at 120 s most flies are clinging motionless to the glass, so late
          timepoints are exactly where the loss would be largest;
        - REDUCED MOVEMENT IS THE PHENOTYPE. A Huntingtin or ALS model fly that
          climbs and then stops is the signal, and a filter tuned to delete
          things that stop will delete the affected animals preferentially. It
          would flatter the disease model and would do so invisibly.

    The measured consequence of leaning on it is already known: the lower bound
    (this filter applied in full) scores MAE 6.58 bias -5.64 on Day 15 against
    the midpoint's 4.54 / -1.63, and its slope against hand counts is 0.125
    against 0.199. Turning it up makes the instrument worse, not better.

    The gate before any tuning here is a labelled set giving, separately:
        sensitivity to a STATIONARY FLY   (what fraction survive this filter)
        specificity to a STATIC ARTIFACT  (what fraction of print it removes)
    The second is known to be good. The first has never been measured, and
    without it a change here cannot be told from a change in the biology.
    """
    frames = [list(pts) for pts in per_frame]
    if not frames:
        return []
    # HARD BINS WERE TRIED HERE AND ARE WRONG. Rounding each position to a
    # `bin_px` grid splits a genuinely static detection across two adjacent bins
    # whenever it sits near a boundary -- x = 15.9 and x = 16.1 are the same
    # printed character and different cells -- so it never reaches `static_frac`
    # in any one bin. Measured on IMG_6184 P3, the cylinder whose BOROSIL label
    # faces the camera and whose detections are visibly ON that label:
    #     bin  4 px  removed 2 of 6      <- misses most of the label
    #     bin  8 px  removed 3 of 6
    #     bin 12 px  removed 5 of 6
    #     bin 20 px  removed 6 of 6      <- and would delete real flies too
    # Widening the bin only trades one error for the other. Matching within a
    # RADIUS has no boundaries to fall across. It is O(n^2) in detections per
    # frame, which was the reason it was rejected first time round -- but that
    # measurement came from the blob detector's much larger sets over 60
    # cylinders. Here it is a handful of points across a dozen frames.
    radius = float(bin_px)
    need = float(static_frac) * len(frames)
    out = []
    for pts in frames:
        moving = 0
        for (x, y) in pts:
            seen = 0
            for other in frames:
                if any((px - x) ** 2 + (py - y) ** 2 <= radius * radius
                       for (px, py) in other):
                    seen += 1
            if seen < need:
                moving += 1
        out.append(moving)
    return out


def pi(count_above: float, n_flies_in_vial: int) -> Optional[float]:
    """Performance Index, or None when the count is not a possible count.

    A sealed cylinder cannot hold more flies than were put in it, so a count
    above `n_flies_in_vial` is not a high PI -- it is a detection fault, and it
    must not be rendered as a number.

    THE SAME APPLIES BELOW ZERO, AND THAT BOUND USED TO BE MISSING. Only the
    upper bound was checked, so `pi(-5, 20)` returned -0.25: a negative
    proportion, which is not a low score but a fault reported as a score. That
    is reachable from inside this module -- `detect_crossings` (kept as a
    documented failure) produced net counts to -32 and PI to -1.60 on HTT Male
    cylinders, because nothing in a difference of two running totals forbids it.
    A count is a cardinality; if it is negative the estimator has failed and the
    honest output is no number.
    """
    n = int(n_flies_in_vial)
    if n <= 0 or count_above < 0 or count_above > n:
        return None
    return float(count_above) / n


# ──────────────────────────────────────── three layers, and what gates them ──
#
# A reading carries three different kinds of statement and they were previously
# emitted as one number. They are separated here because they have different
# evidence behind them and deserve different trust:
#
#   LAYER 1  OBSERVATIONS   what the algorithm saw. count_above, count_below,
#                           static_removed, tube_w_px, band_y. These are always
#                           reported, even when everything else refuses, because
#                           they are what a person needs in order to work out
#                           WHY it refused. Suppressing them would hide the
#                           diagnosis along with the result.
#   LAYER 2  ESTIMATE       the count and PI the estimator derives, with their
#                           provenance attached. Always computed, never
#                           presented as a biological result.
#   LAYER 3  RESULT         the Performance Index as a measurement of the
#                           animals. Gated. Emitted only when the reading passes
#                           every check AND the estimator is validated for the
#                           state the reading is in.
#
# Downstream analysis must consume Layer 3. `pi_estimate` exists for someone who
# knowingly wants the raw number; it is deliberately a different column name so
# that taking it is a choice rather than an accident.

VALID = "VALID"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
NOT_REPORTABLE = "NOT_REPORTABLE"

#: REVIEW_REQUIRED means a person may be able to resolve this -- redraw the box,
#: supply the real N, look at the recording. NOT_REPORTABLE means the data in
#: hand cannot support the measurement and no amount of looking will change it.
_SEVERITY = {VALID: 0, REVIEW_REQUIRED: 1, NOT_REPORTABLE: 2}

#: code -> (severity, what it means, where the threshold comes from)
REASONS: Dict[str, Tuple[str, str, str]] = {
    "INVALID_COUNT_RANGE": (
        NOT_REPORTABLE,
        "the count above the band is negative or exceeds the number of flies "
        "in the cylinder, so it is not a possible count",
        "arithmetic: a sealed cylinder holds what was put in it, and a "
        "cardinality is not negative"),
    "INVALID_DENOMINATOR": (
        NOT_REPORTABLE,
        "the number of flies in the cylinder is zero or negative",
        "arithmetic"),
    "DENOMINATOR_UNKNOWN": (
        NOT_REPORTABLE,
        "the number of flies in the cylinder was never recorded, so a "
        "proportion of them cannot be formed",
        "definition: PI is a proportion and has no meaning without its "
        "denominator"),
    "DENOMINATOR_UNCERTAIN": (
        REVIEW_REQUIRED,
        "the number of flies is assumed rather than counted",
        "measured: on this dataset the loading stated as 20 per cylinder was "
        "14-25 in the operator's own notes, so a nominal N is an assumption "
        "and moves the PI by up to 30%"),
    "TUBE_WIDTH_MISMATCH": (
        NOT_REPORTABLE,
        "the drawn box is far wider than the cylinder, so the detector is "
        "looking for a fly of the wrong size",
        "measured: widening the box on a 100 px cylinder holding 20 flies gave "
        "19 found at 100-200 px and 0 of 20 at 300 px, silently"),
    "ROI_OVERSIZED": (
        REVIEW_REQUIRED,
        "the drawn box is noticeably wider than the cylinder",
        "measured: the same experiment, at the width where the count starts to "
        "move"),
    "BRACKET_TOO_WIDE": (
        REVIEW_REQUIRED,
        "the two bounds disagree by more than 7 flies, so the midpoint carries "
        "real uncertainty",
        "the bracket's own reported width; 7 is the threshold already used by "
        "the LOW confidence flag"),
    "CONSERVATION_VIOLATED": (
        REVIEW_REQUIRED,
        "the two zones together do not account for the flies loaded",
        "a sealed cylinder: above + below must equal N. Measured, this sum ran "
        "5 to 28 against a true 20, in both directions"),
    "STATIC_ARTIFACT_EXCESS": (
        REVIEW_REQUIRED,
        "more than half of what was detected above the band never moved, so "
        "the reading rests mostly on objects that may be print rather than "
        "animals",
        "A STATED CONVENTION, NOT A MEASURED THRESHOLD. Measured context: above "
        "an empty tube 7.2 of 8.0 detections were judged static, and the "
        "midpoint keeps half of them, giving a floor of 5.62 objects where no "
        "fly is present. Half is chosen as the point where the reading is "
        "carried by the static population; it has not been validated"),
    "ESTIMATOR_NOT_VALIDATED": (
        NOT_REPORTABLE,
        "the relationship between what this estimator reports and a hand count "
        "has not been established on any independent set",
        "measured: on the only confirmation set the slope against hand counts "
        "is 0.003, 95% CI [-0.532, +0.539], n = 30. See VALIDATED_DOMAIN"),
}

# THE THREE EMPIRICAL CODES DO NOT PREDICT ERROR. THEY PREDICT IT BACKWARDS.
#
# Measured on Day 15, 432 readings with a hand count beside each, applying these
# codes to the stored columns (the instrument-level code excluded, since it
# fires on everything by design and would make the test vacuous):
#
#   code                     fires on   MAE flagged   MAE clean   Welch p
#   BRACKET_TOO_WIDE              56%          4.66        4.39     0.382
#   CONSERVATION_VIOLATED         58%          4.14        5.08     0.002
#   STATIC_ARTIFACT_EXCESS        79%          4.38        5.12     0.034
#   any code at all               93%          4.42        5.97     0.012
#
# A reading that raises NO flag is WORSE than one that raises several. The old
# `confidence` flag this sits beside is inverted in the same way (HIGH -> MAE
# 5.15, MODERATE -> 4.28, LOW -> 4.66), which is more than the audit found: it
# does not merely fail to discriminate, it points the wrong way.
#
# WHY, AND WHY IT IS NOT FIXED BY MOVING THE THRESHOLDS. The error on this
# instrument is dominated by the calibration, not by per-reading conditions:
# reported = 6.08 + 0.199 x true, so the error is essentially a function of the
# true count alone (+3.7 flies at 3, -5.9 at 15) and is smallest near a true
# count of about 7.6. Quiet readings -- few detections, narrow bracket, little
# static, hence no flags -- are the sparse ones, and sparse is where the
# additive floor dominates. No per-reading flag can predict an error that is set
# by where the reading sits on a compressed calibration curve. YOU CANNOT QC
# YOUR WAY OUT OF A CALIBRATION PROBLEM.
#
# WHAT THIS MEANS FOR THE CODES BELOW, and it is a real distinction:
#   - the NOT_REPORTABLE codes are LOGICAL. An impossible count, an absent
#     denominator, a box that makes the detector look for the wrong-sized
#     animal, an estimator with no established relationship to the truth --
#     none of these depend on the measurement above, and none is weakened by it.
#   - the REVIEW_REQUIRED codes are DESCRIPTIVE. They say truly what state a
#     reading is in ("this count rests mostly on objects that never moved"),
#     and that is worth telling an operator. They are NOT a quality filter and
#     must not be used to select readings, drop outliers, or weight an average:
#     on the evidence above that would select the WORSE readings.
# Anyone tempted to gate on them should re-run the measurement first. The script
# is scratchpad/qc_discriminates.py and it takes seconds.

#: Where the denominator came from, worst last.
N_SOURCES = ("MANUAL", "VIDEO", "NOMINAL", "UNKNOWN")
#: How well it is known, worst last.
N_QUALITIES = ("VERIFIED", "ESTIMATED", "UNCERTAIN")


class Assessment(NamedTuple):
    """The three layers of one reading, and why the third is or is not given."""

    verdict: str                        #: VALID / REVIEW_REQUIRED / NOT_REPORTABLE
    reasons: Tuple[str, ...]            #: reason codes, worst first
    pi_estimate: Optional[float]        #: LAYER 2 -- the number, always computed
    pi_reportable: Optional[float]      #: LAYER 3 -- the number only when VALID
    n_source: str
    n_quality: str

    def explain(self) -> str:
        """One human paragraph. Used by the page and by the CLI report."""
        if self.verdict == VALID:
            return "PI %.3f -- passes every check." % (self.pi_reportable or 0.0)
        head = ("PI: NOT REPORTABLE" if self.verdict == NOT_REPORTABLE
                else "PI: REVIEW REQUIRED")
        lines = [head]
        for c in self.reasons:
            sev, what, _why = REASONS[c]
            lines.append("  %-26s %s" % (c, what))
        return "\n".join(lines)


def assess_reading(count_above: float, n_flies: Optional[int], *,
                   n_source: str = "UNKNOWN",
                   n_quality: str = "UNCERTAIN",
                   bracket_width: Optional[float] = None,
                   static_removed: Optional[float] = None,
                   count_upper: Optional[float] = None,
                   accounted_for: Optional[float] = None,
                   box_over_tube: Optional[float] = None,
                   validated_domain: object = _SENTINEL) -> Assessment:
    """Decide what may be said about one reading, and why.

    Pure: no video, no files, no global state except the manifest, which can be
    overridden through `validated_domain` so the gate itself is testable.

    The order of the checks does not matter -- every applicable code is
    collected and the verdict is the worst severity among them, so a reading
    that fails in three ways says so three times rather than reporting only the
    first thing noticed.
    """
    domain = VALIDATED_DOMAIN if validated_domain is _SENTINEL else validated_domain
    codes: List[str] = []

    if n_source not in N_SOURCES:
        raise ValueError("n_source must be one of %r, got %r" % (N_SOURCES, n_source))
    if n_quality not in N_QUALITIES:
        raise ValueError("n_quality must be one of %r, got %r" % (N_QUALITIES, n_quality))

    # ── the denominator, before anything is divided by it
    if n_source == "UNKNOWN" or n_flies is None:
        codes.append("DENOMINATOR_UNKNOWN")
    elif int(n_flies) <= 0:
        codes.append("INVALID_DENOMINATOR")
    elif n_source == "NOMINAL" or n_quality == "UNCERTAIN":
        codes.append("DENOMINATOR_UNCERTAIN")

    # ── the numerator
    est: Optional[float] = None
    if n_flies is not None and int(n_flies) > 0:
        est = float(count_above) / int(n_flies)
        if count_above < 0 or count_above > int(n_flies):
            codes.append("INVALID_COUNT_RANGE")

    # ── geometry
    if box_over_tube is not None:
        if box_over_tube > 1.6:
            codes.append("TUBE_WIDTH_MISMATCH")
        elif box_over_tube > 1.25:
            codes.append("ROI_OVERSIZED")

    # ── the estimator's own declared uncertainty
    if bracket_width is not None and bracket_width > 7:
        codes.append("BRACKET_TOO_WIDE")
    if (accounted_for is not None and n_flies is not None
            and abs(float(accounted_for) - float(n_flies)) > 5):
        codes.append("CONSERVATION_VIOLATED")
    if (static_removed is not None and count_upper is not None
            and float(count_upper) > 0
            and float(static_removed) / float(count_upper) > 0.5):
        codes.append("STATIC_ARTIFACT_EXCESS")

    # ── and whether this instrument is entitled to report at all
    if domain is None:
        codes.append("ESTIMATOR_NOT_VALIDATED")

    verdict = VALID
    for c in codes:
        if _SEVERITY[REASONS[c][0]] > _SEVERITY[verdict]:
            verdict = REASONS[c][0]
    codes.sort(key=lambda c: -_SEVERITY[REASONS[c][0]])

    return Assessment(verdict=verdict, reasons=tuple(codes), pi_estimate=est,
                      pi_reportable=(est if verdict == VALID else None),
                      n_source=n_source, n_quality=n_quality)


def analyse_video(video_path: str, col_start: int, col_end: int, n_flies: int,
                  timepoints_sec: Sequence[float],
                  zone_height_px: Optional[int] = ZONE_FULL,
                  band_dark: int = BAND_DARK, fly_dark: int = FLY_DARK,
                  min_coverage: float = 0.60,
                  min_blob_area: int = 20, max_blob_area: int = 400,
                  t_zero_sec: float = 0.0,
                  drop_static: bool = False,
                  n_source: str = "NOMINAL",
                  n_quality: str = "UNCERTAIN") -> pd.DataFrame:
    """One cylinder, one column range. Returns a row per timepoint.

    `drop_static` reports the count with never-moving detections removed. Both
    counts are always returned, so the size of that correction is visible rather
    than folded silently into the result.

    `n_source` and `n_quality` say where the denominator came from. They default
    to NOMINAL/UNCERTAIN -- not to a flattering value -- because that is what an
    un-argued `n_flies=20` actually is, and because the operator's own notes on
    this dataset put the real loading at 14-25 against a stated 20. A caller who
    has counted the flies should say so; a caller who has not should not be
    given the benefit of the doubt by a default.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("cannot open video: %s" % video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError("video reports no frame rate: %s" % video_path)

    # THE BAND MUST NOT BE LOOKED FOR IN A FRAME THAT CONTAINS FLIES.
    # This read frame 0 and was badly wrong. At the tap the cohort is heaped in
    # the bottom of the cylinder, and a raft of flies is dark right across the
    # tube -- which is precisely the signature `find_black_band` searches for.
    # Measured, band row from frame 0 against the same cylinder later:
    #     IMG_6029 P1   frame0 898   t10 576   t60 576   -> 322 px too low
    #     IMG_6029 P3   frame0 891   t10 586   t60 586   -> 305 px too low
    #     IMG_6181 P2   frame0 743   t10 630   t60 630   -> 113 px too low
    #     IMG_6184 P1   frame0 677   t10 676   t60 676   ->   1 px (sparse HD
    #                                                      flies form no raft)
    # A temporal median over frames spread across the clip contains the glass,
    # the print and the band, and no flies -- any given fly is somewhere else in
    # most of the frames. This is NOT background subtraction for detection; the
    # counting step still works on the raw frame. It is only how the band is
    # located, and it costs a few sampled reads.
    n_probe = 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    span = total if total > 0 else int((max(timepoints_sec) + 1) * fps)
    probe = []
    for idx in np.linspace(0, max(span - 1, 0), n_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            probe.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    if not probe:
        cap.release()
        raise RuntimeError("could not read any frame from %s" % video_path)
    median_img = np.median(np.stack(probe), axis=0).astype(np.uint8)
    band_y = find_black_band(median_img, col_start, col_end, band_dark,
                             min_coverage)
    if band_y is None:
        cap.release()
        return pd.DataFrame([{
            "time_sec": np.nan, "band_y": np.nan, "count_above": np.nan,
            "pi": np.nan, "over_detection": True,
            "status": ("refused: no row reaches %.0f%% dark coverage, so the "
                       "band was not found" % (100 * min_coverage))}])

    # MEASURE THE TUBE, DO NOT TAKE THE OPERATOR'S BOX FOR IT.
    # Both detectors scale the expected fly from the width they are given, and
    # the page asks for that width by hand. Measured on a 100 px cylinder with
    # 20 flies, widening only the drawn box: 100 px -> 19 found, 200 px -> 19,
    # 300 px -> 0 of 20, silently, because the assumed fly grew from 4.0 to
    # 12.0 px. The band is drawn round the glass, so its dark run at the band
    # row IS the tube; use that and report the discrepancy.
    tube = tube_width_from_band(median_img, band_y, col_start, col_end, band_dark)
    box_w = int(col_end) - int(col_start)
    if tube is None:
        cap.release()
        return pd.DataFrame([{
            "time_sec": np.nan, "band_y": float(band_y), "count_above": np.nan,
            "pi": np.nan, "over_detection": True,
            "status": ("refused: the band was found at row %d but its dark run "
                       "does not give a tube width, so the fly size cannot be "
                       "calibrated" % band_y)}])
    det_c0, det_c1 = tube
    tube_w = det_c1 - det_c0
    box_excess = box_w / max(tube_w, 1)
    if box_excess > 1.6:
        cap.release()
        return pd.DataFrame([{
            "time_sec": np.nan, "band_y": float(band_y), "count_above": np.nan,
            "pi": np.nan, "over_detection": True,
            "status": ("refused: the drawn box is %d px wide but the cylinder "
                       "measures %d px (%.1fx). The detector sizes a fly from "
                       "the width it is given, so this box would look for a "
                       "%.1f px fly instead of %.1f px and undercount badly. "
                       "Draw the box to the glass."
                       % (box_w, tube_w, box_excess,
                          FB.fly_diameter(box_w), FB.fly_diameter(tube_w)))}])
    col_start, col_end = det_c0, det_c1

    # PRIMARY: A BRACKET, NOT A POINT.
    #
    # Two counts of the same zone bound the truth from opposite sides, and each
    # bound is wrong in a KNOWN direction, which is what makes them usable:
    #   upper = every detection above the band. Necessarily too high, because the
    #           printed specification block is dark, fly-sized and inside the zone.
    #   lower = detections that MOVE. Necessarily too low, because a fly clinging
    #           motionless to the glass is discarded along with the print.
    # The animal count cannot be outside that interval, and the midpoint is the
    # estimator. Measured against 15 hand counts on IMG_6029 and IMG_6181:
    #
    #     upper bound (all detections)        MAE 3.47   bias +2.53
    #     lower bound (movers only)           MAE 2.80   bias -2.67
    #     MIDPOINT                            MAE 2.20   bias -0.07   <- used
    #
    # 2.20 is inside the assay's own 3-fly repeat-tap noise floor, and the bias of
    # -0.07 says the two errors genuinely cancel rather than one dominating. The
    # bracket held the hand count on 11 of 15 readings, mean width 5.2 flies.
    #
    # Rejected on the same data, and recorded so nobody re-tries them:
    #     crossing gate (net_crossings)       MAE 10.47  bias -10.47, and it
    #         returns NEGATIVE counts -- see `detect_crossings`
    #     complementary (N - count_below)     MAE 6.07   bias +5.67, defeated by
    #         crowding: the below-count finds 1-8 flies where 13-17 are packed
    #     proportional closure N*a/(a+b)      MAE 4.66   bias +4.55; the two
    #         sensors' errors are not proportional, so forcing the sum to N
    #         smears the label bias into the crowding bias
    n_static_probe = 15
    probe_pts = []
    for idx in np.linspace(0, max(span - 1, 0), n_static_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        okp, frp = cap.read()
        if not okp:
            probe_pts.append([])
            continue
        gp = cv2.cvtColor(frp, cv2.COLOR_BGR2GRAY)
        _n, cp, _z = count_flies_above_band(gp, median_img, band_y,
                                            col_start, col_end, zone_height_px)
        probe_pts.append(cp)
    movers = persistence_filter(probe_pts, bin_px=5, static_frac=0.8)
    static_est = int(round(float(np.median(
        [len(pp) - mv for pp, mv in zip(probe_pts, movers)])))) if movers else 0

    rows = []
    for t in timepoints_sec:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((t_zero_sec + float(t)) * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hi, _c, _z = count_flies_above_band(grey, median_img, band_y,
                                            col_start, col_end, zone_height_px)
        below, _cb, _zb = count_flies_below_band(grey, median_img, band_y,
                                                 col_start, col_end,
                                                 zone_height_px)
        lo = max(0, int(hi) - static_est)
        mid = 0.5 * (lo + int(hi))
        width = int(hi) - lo
        comp = max(0, int(n_flies) - int(below))
        accounted = int(hi) + int(below)

        # A sealed cylinder holds exactly `n_flies`. The two zones should account
        # for all of them; when they do not, something is being missed or invented
        # and the reading is flagged rather than quietly reported. Measured, this
        # sum ran 5 to 28 against a true 20 -- violated in BOTH directions, which
        # is why it is a diagnostic and not a correction.
        conf = ("HIGH" if width <= 3 else
                "MODERATE" if width <= 7 else "LOW")

        # THE EXPORTED PI USED TO BYPASS EVERY CHECK IN THIS MODULE.
        # `pi()` gained its bounds in F3 and then turned out to have no callers:
        # the number that actually left the module was `mid / max(n_flies, 1)`,
        # computed inline, which cannot refuse anything. `max(..., 1)` even
        # turned a denominator of zero into a division by one, so a cylinder
        # with no flies recorded produced a PI equal to the raw count. The
        # assessment below is now the only route out.
        a = assess_reading(
            float(mid), int(n_flies),
            n_source=n_source, n_quality=n_quality,
            bracket_width=float(width), static_removed=float(static_est),
            count_upper=float(hi), accounted_for=float(accounted),
            box_over_tube=float(box_excess))

        rows.append({
            "time_sec": float(t), "band_y": int(band_y),
            "reported_count": round(float(mid), 1),
            # LAYER 3 -- the biological result. None unless every check passes.
            "pi": (round(a.pi_reportable, 4) if a.pi_reportable is not None
                   else None),
            "pi_verdict": a.verdict,
            "pi_reasons": ";".join(a.reasons),
            # LAYER 2 -- the estimate, with its provenance. Never a result.
            "pi_estimate": (round(a.pi_estimate, 4)
                            if a.pi_estimate is not None else None),
            "n_source": n_source, "n_quality": n_quality,
            "count_lower": int(lo), "count_upper": int(hi),
            "bracket_width": int(width),
            "confidence": conf,
            "static_removed": int(static_est),
            "count_above": int(hi),
            "count_below": int(below),
            "complementary": int(comp),          # diagnostic only -- see above
            "accounted_for": int(accounted),
            "accounted_gap": int(accounted - int(n_flies)),
            "n_flies": int(n_flies),
            "tube_x0": int(det_c0), "tube_x1": int(det_c1),
            "tube_w_px": int(tube_w), "box_w_px": int(box_w),
            "box_over_tube": round(float(box_excess), 2),
            "fly_diameter_px": round(float(FB.fly_diameter(tube_w)), 1),
            "zone_top": int(0 if zone_height_px is None else max(0, band_y - zone_height_px)),
            "zone_height_px": (-1 if zone_height_px is None else int(zone_height_px)),
            "status": "scored"})
    cap.release()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────── hand-count validation ──
#: The assay's own repeat-tap spread for a perfect human counter. An algorithm at
#: or below this is as accurate as the measurement it is replacing; chasing lower
#: is not useful science.
NOISE_FLOOR_FLIES = 3


def _norm_pos(v) -> str:
    """'P1', 'C1', 1, 'c1' -> 'P1'. The page writes C1..C3, the bench sheet P1..P3."""
    s = str(v).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    return ("P" + digits) if digits else s


def validation_report(hand: pd.DataFrame, tool: pd.DataFrame,
                      day_lookup=None) -> str:
    """Accuracy of the tool against hand counts. Returns the printable report.

    One implementation, used by both `scripts/validate_climbing_band.py` and the
    Validate tab, so the two cannot drift apart.

    `hand` needs video_filename, cylinder_position, timepoint_sec, hand_count.
    `tool` is the CSV the Climbing (Simple) page exports.
    `day_lookup` maps a video filename to "Day 3"/"Day 7"; without it the day
    split is reported as unavailable rather than guessed.
    """
    h = hand.copy()
    t = tool.copy()
    h.columns = [c.strip() for c in h.columns]
    t.columns = [c.strip() for c in t.columns]

    need = {"video_filename", "cylinder_position", "timepoint_sec", "hand_count"}
    missing = need - set(h.columns)
    if missing:
        return "Hand-count file is missing column(s): %s" % sorted(missing)

    h["_v"] = h.video_filename.astype(str).str.strip()
    h["_p"] = h.cylinder_position.map(_norm_pos)
    h["_t"] = pd.to_numeric(h.timepoint_sec, errors="coerce")
    h["_hand"] = pd.to_numeric(h.hand_count, errors="coerce")
    filled = h[h._hand.notna()]

    tv = "video" if "video" in t.columns else "video_filename"
    tp = "cylinder" if "cylinder" in t.columns else "cylinder_position"
    tt = "time_sec" if "time_sec" in t.columns else "timepoint_sec"
    tc = "count_above"
    for col, nm in ((tv, "video"), (tp, "cylinder"), (tt, "time"), (tc, "count")):
        if col not in t.columns:
            return "Tool file is missing the %s column (expected %r)" % (nm, col)
    if "status" in t.columns:
        t = t[t.status.astype(str) == "scored"]
    t["_v"] = t[tv].astype(str).str.strip()
    t["_p"] = t[tp].map(_norm_pos)
    t["_t"] = pd.to_numeric(t[tt], errors="coerce")
    t["_tool"] = pd.to_numeric(t[tc], errors="coerce")

    m = filled.merge(t[["_v", "_p", "_t", "_tool"]], on=["_v", "_p", "_t"],
                     how="left")
    paired = m[m._tool.notna()].copy()

    L = []
    L.append("=" * 66)
    L.append("CLIMBING (SIMPLE) -- ACCURACY AGAINST HAND COUNTS")
    L.append("=" * 66)
    L.append("hand-count rows supplied : %d of %d filled in" % (len(filled), len(h)))
    L.append("matched to a tool reading: %d" % len(paired))
    unmatched = len(filled) - len(paired)
    if unmatched:
        L.append("UNMATCHED (not scored, or refused, or a name/position mismatch): %d"
                 % unmatched)
        for _, r in m[m._tool.isna()].iterrows():
            L.append("    %s %s t=%s" % (r._v, r._p, r._t))
    if paired.empty:
        L.append("")
        L.append("No paired measurements -- accuracy CANNOT be computed.")
        return "\n".join(L)

    paired["_err"] = paired._tool - paired._hand
    paired["_abs"] = paired._err.abs()
    mae = paired._abs.mean()
    bias = paired._err.mean()

    L.append("")
    L.append("MAE overall            : %.2f flies/vial   (bias %+.2f, n = %d)"
             % (mae, bias, len(paired)))

    L.append("")
    L.append("MAE by day")
    if day_lookup:
        paired["_day"] = paired._v.map(lambda v: day_lookup.get(v, "unknown"))
        for day in ("Day 3", "Day 7"):
            s = paired[paired._day == day]
            L.append("    %-7s: %s" % (day, "%.2f  (n = %d)" % (s._abs.mean(), len(s))
                                       if len(s) else "no hand counts supplied"))
        unk = paired[~paired._day.isin(["Day 3", "Day 7"])]
        if len(unk):
            L.append("    unknown: %.2f  (n = %d)" % (unk._abs.mean(), len(unk)))
    else:
        L.append("    unavailable -- no day could be determined for these videos")

    L.append("")
    L.append("MAE by timepoint")
    for tp_s in (10, 30, 60):
        s = paired[paired._t == tp_s]
        L.append("    T=%-4d : %s" % (tp_s, "%.2f  (n = %d)" % (s._abs.mean(), len(s))
                                      if len(s) else "no hand counts supplied"))
    other = paired[~paired._t.isin([10, 30, 60])]
    if len(other):
        for tp_s, s in other.groupby("_t"):
            L.append("    T=%-4g : %.2f  (n = %d)" % (tp_s, s._abs.mean(), len(s)))

    L.append("")
    bad = paired[paired._abs > 5].sort_values("_abs", ascending=False)
    if bad.empty:
        L.append("Vials differing by more than 5 flies: none")
    else:
        L.append("Vials differing by more than 5 flies (%d):" % len(bad))
        for _, r in bad.iterrows():
            L.append("    %-16s %s t=%-4g  tool %3.0f  hand %3.0f  diff %+.0f"
                     % (r._v, r._p, r._t, r._tool, r._hand, r._err))

    L.append("")
    within = mae <= NOISE_FLOOR_FLIES
    L.append("Tool IS %swithin the assay noise floor of %d flies (MAE %.2f)."
             % ("" if within else "NOT ", NOISE_FLOOR_FLIES, mae))
    L.append("=" * 66)
    return "\n".join(L)
