"""Size-matched fly detection: find blobs the size of a fly, not pixels that are dark.

THE PROBLEM WITH THRESHOLDING, WHICH IS STRUCTURAL AND NOT A TUNING FAULT
-------------------------------------------------------------------------
Adaptive thresholding asks of every pixel: "is this darker than its neighbourhood?"
A printed graduation, a numeral, a scratch in the glass and a fly all answer yes. It
then groups surviving pixels with connected components, which welds touching flies
into a single object -- worst exactly when the cohort is densest, which is when the
early timepoints are scored. Neither failure can be tuned away, because neither is
about the threshold.

THE METHOD, AND WHERE IT COMES FROM
-----------------------------------
The particle-tracking literature settled this for colloids in the 1990s (Crocker and
Grier 1996), and FreeClimber (Spierer et al. 2021, J Exp Biol 224:jeb229377) brought
it to this exact assay: it hands frames to `trackpy`, a Python implementation of that
algorithm, and filters candidate spots on expected diameter, roundness and integrated
brightness ("minmass").

The question becomes "is there a blob of ABOUT THIS SIZE centred here?", which buys
three things at once:

  1. A printed line is elongated, so it responds weakly to an isotropic filter matched
     to a fly and fails the eccentricity test. It is rejected on SHAPE, not darkness.
  2. Touching flies are separated by LOCAL MAXIMA instead of merged by connected
     components. A pixel is a candidate only if nothing stronger lies within one fly
     width. Two flies in contact are two maxima.
  3. The band-pass removes illumination gradients (long wavelength) and sensor noise
     (short wavelength) in one step, so no global brightness threshold is needed.

The detection threshold is set from the EMPTY TUBE. The temporal median contains the
graduations, the numerals, the scratches and the noise, and no flies, so running the
same band-pass over it gives the distribution of responses produced by everything
that is not a fly. A high quantile of that is the threshold: a chosen false-positive
rate rather than a magic number. Thresholding relative to the darkest thing in the
frame is the obvious alternative and is much worse -- one very dark mark then
suppresses every fly in the vial, which cost 5 flies per frame when it was tried.

TWO IDEAS THAT WERE TRIED AND REJECTED, RECORDED SO THEY ARE NOT RE-TRIED
-------------------------------------------------------------------------
TEMPORAL BACKGROUND SUBTRACTION -- FreeClimber's own step, and its paper notes it is
what stops vial marks becoming false positives. It fails here. FreeClimber measures
VELOCITY over the first seconds, while every fly is moving; this assay needs a COUNT
at 120 s, when flies cling motionless to the glass. A temporal median then CONTAINS
them, and subtracting it deletes them. Measured: MAE 14.97 with full subtraction
against 7.73 with none; the fitted weight came out at exactly zero.

AUTOMATIC SCALE SELECTION -- a scale-normalised Laplacian over a range of sigma
should measure the fly's size rather than assume it (Lindeberg 1998). Implemented and
measured, it did not work on this footage: the selected radius came back as a
constant 0.0198 x box width for every vial, pinned to the bottom of the sigma range,
because small-scale noise wins the argmax. Held out it scored MAE 3.61 against this
module's 2.48.

REJECTING MARKS BY LOCATION -- the temporal median is a picture of the tube with the
flies averaged away, so a fly-sized blob in it is a permanent mark at a fixed
coordinate, and detections standing on one can be dropped. This helps a detector that
is finding noise (it took the scale-space variant from MAE 25.57 to 6.30) but with
the band-pass correctly scaled to the fly it is unnecessary and slightly harmful --
it starts eating real flies. It is retained behind `mask_marks`, defaulting to off,
because it is the right tool on rigs with heavier printing than this laboratory's.

WHAT WAS MEASURED, AND AGAINST WHAT
-----------------------------------
The target is the number of flies in a SEALED cylinder, which is ground truth beyond
dispute at every timepoint. The laboratory's above-the-line hand count is NOT used
for fitting: it is recorded cumulatively (a fly that has once crossed stays counted),
so it is a different quantity from an instantaneous detection -- see
docs/VIDEO_SCORING.md.

Parameters were fitted on 19 vials and tested on 22 held-out vials from separate
recordings, spanning all six paraquat conditions (110 held-out cells):

    band-pass (this module)    MAE 2.69   bias -1.60   within 3 flies 67.3%
    shipped adaptive threshold MAE 8.38   bias +5.71   within 3 flies 32.7%

MEASURED LIMITS, stated rather than left to be discovered:
  - Flies closer than about 1.2 fly-widths centre to centre are counted as one. The
    separation is a fitted parameter and larger values scored better on real footage
    (0.6d -> MAE 3.66, 0.8d -> 3.44, 0.95d -> 3.28, 1.2d -> 2.93), i.e. splitting
    close pairs cost more in false positives than it recovered. Touching flies are
    therefore NOT resolved, and the count is biased low when the cohort is dense.
  - A single-scale matched filter responds to point sources: a one-pixel speck is
    detected, though with about a twentieth of a fly's mass (456 against 9480).
  - An object several times a fly's size fragments into detections around its rim
    (a disc of radius 49 px gave 22 detections where 10 objects were drawn), so a
    large shadow or a smear inflates the count rather than being ignored.

Seventeen of 58 vials were excluded first, because the automatic vial finder had put
a box across most of the frame (aspect ratio 1.32 against 4.60 for a real cylinder).
Those are geometry failures, not detection failures, and 28% of automatic geometry
failing is the reason the shipped tool has the operator draw the boxes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import peak_local_max

__all__ = ["PARAMS", "MEASURED", "bandpass", "bandpass_raw", "mark_map",
           "detect", "fly_diameter"]

#: Fitted on 19 vials, tested on 22 held out. See the module docstring.
PARAMS: Dict[str, float] = {
    # A fly's apparent width as a fraction of the CYLINDER's width. The fitted value
    # is sharply peaked: 0.05 -> MAE 9.11, 0.06 -> 5.26, 0.07 -> 2.93, 0.085 -> 5.92,
    # 0.10 -> 9.82. It is the single most important parameter here, which is the
    # whole point of a size-matched detector.
    "fly_d_frac": 0.07,
    # The empty tube's own band-pass response is the null distribution; this quantile
    # of it is the detection threshold. Also sharply peaked (99.0 -> 10.59,
    # 99.5 -> 2.93, 99.8 -> 6.80).
    "null_q": 99.5,
    "thr_mult": 1.0,
    # Minimum separation between two maxima, in fly widths. Above 1.0 because a fly
    # is longer than it is wide and its band-pass response has shoulders.
    "sep_frac": 1.2,
    # Floor on the detection threshold, in robust standard deviations of the frame's
    # own band-pass noise. Guards the case where the empty tube is so featureless
    # that its quantile collapses -- see `detect`.
    "noise_mult": 6.0,
    "minmass_mult": 0.5,   # mass floor, as a multiple of threshold x disc area
    # Shape is judged over a WIDER window than mass. The band-pass blurs a 3 px drawn
    # line into a ridge about 10 px tall, so inside a 0.6d patch the line looks nearly
    # round -- measured eccentricity 1.75, which slipped under a 1.8 limit and let
    # every scoring line through. Widening the window to 1.2d separates them cleanly:
    # line 3.40, fly 1.00, and a fly sitting ON the line 2.55. Hence a limit of 2.8,
    # which rejects the line and keeps the fly standing on it -- the animal whose
    # classification matters most.
    # 1.0d is the fitted compromise. On this laboratory's footage the shape filter
    # barely moves the vial total at all (MAE 2.93 for every limit at or above 3.5),
    # so the cost of enabling it is small and measured: held-out MAE 2.69 against
    # 2.43 with shape ignored. It is enabled anyway, because the false positives it
    # removes sit ON the scoring line -- exactly where the above-or-below decision is
    # made -- so they damage the score far more than they damage the total.
    "ecc_window_frac": 1.0,
    "max_ecc": 2.8,        # reject anything longer than this x its width
    # Reject detections standing on a permanent mark. OFF: see the module docstring.
    "mask_marks": 0.0,
    "mark_q": 99.0,
    "mark_r_frac": 0.35,
}

#: Held-out performance, for callers that report their own uncertainty.
MEASURED = {"mae_total": 2.69, "bias_total": -1.60, "within3": 0.673,
            "n_held_out_cells": 110, "n_held_out_vials": 22,
            "compare_threshold_mae": 8.38}


def fly_diameter(vial_width_px: float, params: Optional[dict] = None) -> float:
    """The fly diameter this detector will look for, in pixels."""
    p = dict(PARAMS, **(params or {}))
    return max(float(p["fly_d_frac"]) * float(vial_width_px), 4.0)


def bandpass(gray: np.ndarray, d: float) -> np.ndarray:
    """Crocker-Grier band-pass for DARK features on a light background.

    The image is inverted so flies are bright, blurred at `sigma = d/(2*sqrt(2))` --
    the scale at which a blob of diameter `d` gives its strongest response -- and a
    boxcar estimate of the local background over `2d+1` is subtracted. What survives
    is structure at the fly's own scale and nothing else.
    """
    return np.clip(bandpass_raw(gray, d), 0, None)


def bandpass_raw(gray: np.ndarray, d: float) -> np.ndarray:
    """The band-pass BEFORE clipping negatives away.

    Needed for the noise estimate: `bandpass` clips at zero, so over half its values
    are exactly zero and its median absolute deviation is identically zero -- which
    silently disabled the noise floor in `detect` when it was first written. The
    unclipped response is roughly symmetric about zero wherever there is no fly, so
    its MAD is a sound robust estimate of the sensor noise.
    """
    inv = 255.0 - gray.astype(np.float32)
    sigma = max(d / (2.0 * np.sqrt(2.0)), 0.8)
    fine = cv2.GaussianBlur(inv, (0, 0), sigma)
    k = int(2 * round(d) + 1)
    return fine - cv2.blur(inv, (k, k))


def _disc(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def mark_map(background: np.ndarray, d: float,
             params: Optional[dict] = None) -> np.ndarray:
    """Where the permanent marks are, from the empty-tube picture.

    Off by default -- see the module docstring for what it costs and when it helps.
    """
    p = dict(PARAMS, **(params or {}))
    bp = bandpass(background, d)
    thr = float(np.percentile(bp, float(p["mark_q"])))
    m = np.zeros(background.shape, bool)
    if bp.max() <= max(thr, 1e-6):
        return m
    for (y, x) in peak_local_max(bp, min_distance=max(int(0.7 * d), 1),
                                 threshold_abs=max(thr, 1e-6)):
        r = max(int(float(p["mark_r_frac"]) * d), 2)
        m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    return m


def detect(frame_gray: np.ndarray, background_gray: np.ndarray,
           params: Optional[dict] = None, d: Optional[float] = None,
           marks: Optional[np.ndarray] = None
           ) -> List[Tuple[float, float, float]]:
    """Find the flies. Returns `(x, y, mass)` per fly, in crop coordinates.

    `frame_gray` and `background_gray` are the same region of the same recording:
    the frame to count, and the temporal median that shows the tube without flies.
    """
    p = dict(PARAMS, **(params or {}))
    if frame_gray.ndim != 2 or background_gray.shape != frame_gray.shape:
        raise ValueError("frame and background must be the same 2-D grey region; "
                         f"got {frame_gray.shape} and {background_gray.shape}")
    h, w = frame_gray.shape[:2]
    if d is None:
        d = fly_diameter(w, p)

    bp = bandpass(frame_gray, d)
    bpb = bandpass(background_gray, d)
    thr = float(p["thr_mult"]) * float(np.percentile(bpb, float(p["null_q"])))

    # A NOISE FLOOR, because the null can collapse. The threshold above is a high
    # quantile of the empty tube's response, which is the right null when the tube
    # has structure -- printing, scratches, texture -- as every real one does. But a
    # nearly featureless background drives that quantile towards zero, and then the
    # frame's own sensor noise clears it and is counted as flies. So the threshold is
    # also held above the frame's noise, estimated robustly: flies are sparse, so the
    # median absolute deviation of the band-pass is dominated by noise, not by them.
    # On real footage the empty-tube quantile is far above this and the floor never
    # binds; on a synthetic flat tube it is the only thing standing between the
    # detector and 20 phantom flies.
    raw = bandpass_raw(frame_gray, d)
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))
    thr = max(thr, med + float(p["noise_mult"]) * 1.4826 * mad, 1e-3)
    if bp.max() <= thr:
        return []
    if marks is None and float(p["mask_marks"]) > 0:
        marks = mark_map(background_gray, d, p)

    peaks = peak_local_max(bp, min_distance=max(int(float(p["sep_frac"]) * d), 1),
                           threshold_abs=thr)
    if not len(peaks):
        return []

    rr = max(int(0.6 * d), 2)
    disc = _disc(rr)
    minmass = float(p["minmass_mult"]) * thr * float(disc.sum())
    max_ecc = float(p["max_ecc"])
    out: List[Tuple[float, float, float]] = []
    for (py, px) in peaks:
        if marks is not None and marks[py, px]:
            continue
        y0, y1 = max(0, py - rr), min(h, py + rr + 1)
        x0, x1 = max(0, px - rr), min(w, px + rr + 1)
        dm = disc[y0 - (py - rr):y1 - (py - rr), x0 - (px - rr):x1 - (px - rr)]
        vals = bp[y0:y1, x0:x1] * dm
        tot = float(vals.sum())
        if tot < minmass:
            continue
        # brightness-weighted centroid and second moments -- sub-pixel position, and
        # the eccentricity that separates a fly from a line or a rim
        # position and mass come from the disc; SHAPE from a wider window, because a
        # blurred line only looks like a line if enough of it is in view
        ew = max(int(float(p["ecc_window_frac"]) * d), rr)
        ey0, ey1 = max(0, py - ew), min(h, py + ew + 1)
        ex0, ex1 = max(0, px - ew), min(w, px + ew + 1)
        wide = bp[ey0:ey1, ex0:ex1]
        wtot = float(wide.sum())
        if wtot <= 0:
            continue
        gy, gx = np.mgrid[ey0:ey1, ex0:ex1]
        cy = float((gy * wide).sum() / wtot)
        cx = float((gx * wide).sum() / wtot)
        vyy = float(((gy - cy) ** 2 * wide).sum() / wtot)
        vxx = float(((gx - cx) ** 2 * wide).sum() / wtot)
        vxy = float(((gy - cy) * (gx - cx) * wide).sum() / wtot)
        tr, det = vyy + vxx, vyy * vxx - vxy * vxy
        root = max(tr * tr / 4.0 - det, 0.0) ** 0.5
        l1, l2 = tr / 2.0 + root, max(tr / 2.0 - root, 1e-9)
        if (l1 / l2) ** 0.5 > max_ecc:
            continue
        dgy, dgx = np.mgrid[y0:y1, x0:x1]
        pcy = float((dgy * vals).sum() / tot)
        pcx = float((dgx * vals).sum() / tot)
        out.append((pcx, pcy, tot))
    return out
