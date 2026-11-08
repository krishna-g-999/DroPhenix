"""
DroPhenix — modules/fly_detector.py
Fly detection engine for Drosophila climbing assay videos.

Algorithm (per-vial):
1. Crop vial ROI from frame
2. Grayscale + Gaussian blur 5×5 (noise reduction)
3. Background subtraction:
   - Static bg_frame provided → cv2.absdiff (preferred; background captured post-tap)
   - No bg_frame → MOG2 Gaussian mixture model (stateful, needs warmup)
4. Morphological open (removes noise) then close (fills fly body gaps)
5. Binary threshold → foreground mask
6. cv2.findContours (RETR_EXTERNAL) → connected blobs
7. Area filter: blob_area ≥ min_area; a blob larger than one fly is
   counted by area (N3 merge correction) and one larger than
   debris_area_mult × that is discarded as debris
8. Centroid (cx, cy) from image moments
9. Flies above threshold line: cy < threshold_frac × vial_height
10. PI = (above − below) / N_start × 100, where N_start is the number of flies
    LOADED into the vial, taken from the loading record — not the number
    detected. See VialCount for why, and register finding N5.

Scientific basis:
- MOG2 background subtraction is standard for Drosophila motion detection
  (Nichols et al. 2012 J Vis Exp 61:e3795)
- Threshold line at 50% vial height follows the negative geotaxis assay
  protocol (Feany & Bender 2000 Nature 404:394; Gargano et al. 2005)
- Blob area range calibrated for USB webcam at 1280×720 HD resolution;
  scale min_area/max_area proportionally for other magnifications
- Morphological open kernel = 3×3 ellipse removes single-pixel noise
  while preserving fly-sized blobs (~4–6 px diameter at HD)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import warnings

from modules import missingness as _MS


def _round_or_nan(v, nd: int):
    """Round a finite value; leave a NaN as a NaN.

    `round(float('nan'), 2)` returns nan, but an int-typed 0 sentinel would not,
    and the historical code rounded a hard-coded 0.0. Keeping this explicit
    makes it impossible to reintroduce a rounded zero as a stand-in for an
    absent measurement.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if not math.isfinite(f) else round(f, nd)

try:
    import cv2
    import numpy as np
    _CV2 = True
except ImportError:
    cv2 = None
    np = None
    _CV2 = False

try:
    import pandas as pd
    _PD = True
except ImportError:
    pd = None
    _PD = False


# ─── Vial layout presets ──────────────────────────────────────────────────────
VIAL_PRESETS: Dict[str, dict] = {
    "Single vial (full frame)": {"n_vials": 1,  "cols": 1, "rows": 1},
    "2 vials (horizontal)":     {"n_vials": 2,  "cols": 2, "rows": 1},
    # Three cylinders in a row is this laboratory's own rig and is what
    # `climbing_video` defaults to; it was missing from this list, so a caller
    # asking for it fell through `VIAL_PRESETS.get(preset, ...)` to ONE vial
    # covering the whole frame -- every fly in one region, under one genotype.
    "3 vials (horizontal)":     {"n_vials": 3,  "cols": 3, "rows": 1},
    "4 vials (2×2 grid)":       {"n_vials": 4,  "cols": 2, "rows": 2},
    "6 vials (2×3 grid)":       {"n_vials": 6,  "cols": 3, "rows": 2},
    "8 vials (2×4 grid)":       {"n_vials": 8,  "cols": 4, "rows": 2},
    "12 vials (3×4 grid)":      {"n_vials": 12, "cols": 4, "rows": 3},
}


# ─── VialROI ──────────────────────────────────────────────────────────────────
@dataclass
class VialROI:
    """
    Region of interest for one vial in the video frame.

    Attributes
    ----------
    x, y   : top-left corner (pixels)
    w, h   : width and height (pixels)
    label  : display label (e.g. "FUS | Spermine | Male")
    vial_id: integer ID (1-indexed)
    genotype, sex, treatment : biological metadata
    expected_n : N_start — the number of flies LOADED into this vial, taken
                 from the loading record. This is the only legitimate
                 denominator for a climbing rate (register N5).

                 0 means "not supplied". It is not a default of 10: a vial
                 whose loading record is absent has no denominator, and
                 inventing one is the defect the N5 fix removes. With
                 expected_n = 0 the counts are still reported and the RATE is
                 refused (Missingness = REFUSED).
    """
    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 400
    label: str = ""
    vial_id: int = 1
    genotype: str = ""
    sex: str = ""
    treatment: str = ""
    # N5: was `= 10`. A silent default denominator of ten flies produced a
    # plausible-looking percentage for a vial nobody had recorded the loading of.
    expected_n: int = 0

    @property
    def n_start(self) -> int:
        """The loading-record N, under the name the data model uses."""
        return int(self.expected_n or 0)

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)

    @staticmethod
    def auto_grid(
        frame_w: int,
        frame_h: int,
        preset: str = "Single vial (full frame)",
        margin_px: int = 4,
    ) -> "List[VialROI]":
        """
        Divide the frame into an evenly-spaced grid. UNCALIBRATED.

        Register N77. This splits the frame geometrically, with no reference to
        the image, so it is correct only when the rack fills the frame edge to
        edge and is square to the camera. It rarely is: a rack sitting to one
        side of the frame, or a camera a few degrees off, puts a vial boundary
        through the middle of a tube, and every fly on the far side of that
        boundary is then attributed to the neighbouring vial -- and therefore to
        the neighbouring GENOTYPE. That is a mislabelling, not a measurement
        error, and no amount of downstream statistics recovers from it.

        Use `VialROI.from_frame`, which finds the tubes in the image, and keep
        this for the single-vial case and as an explicit fallback when
        detection reports `needs_review`. Either way the operator must confirm
        the regions; `climbing_layout.py` is the manual editor.
        """
        # An unrecognised preset used to fall back to a SINGLE full-frame vial,
        # silently: every fly in the frame counted into one region and reported
        # under one genotype. A layout nobody asked for is not a default.
        if preset not in VIAL_PRESETS:
            raise ValueError(
                f"Unknown vial preset {preset!r}. The layouts available are "
                f"{sorted(VIAL_PRESETS)}. This used to fall through to one "
                f"full-frame vial, which silently pooled every vial in the "
                f"frame into a single region and a single genotype."
            )
        cfg  = VIAL_PRESETS[preset]
        cols = cfg["cols"]
        rows = cfg["rows"]
        cw   = frame_w // cols
        rh   = frame_h // rows
        vials = []
        vid = 1
        for r in range(rows):
            for c in range(cols):
                vials.append(VialROI(
                    x      = c * cw + margin_px,
                    y      = r * rh + margin_px,
                    w      = max(1, cw - 2 * margin_px),
                    h      = max(1, rh - 2 * margin_px),
                    label  = f"Vial_{vid}",
                    vial_id= vid,
                ))
                vid += 1
        return vials


    @staticmethod
    def from_frame(
        frame,
        n_vials: int = 3,
        margin_px: int = 4,
        labels: "Optional[List[str]]" = None,
        rows: int = 1,
    ) -> dict:
        """
        Locate the vial columns IN THE IMAGE and return regions for them.

        Register N77. The flies are the signal: they are darker than the tube
        and they are inside their own tube, so the column-wise darkness profile
        of a single frame peaks over the tubes and falls in the gaps between
        them. The band-finding is `climbing_video._bands_from_profile`, the same
        routine that module uses on its activity profile -- reused rather than
        re-written, as the merge correction was for N3 -- so the fragment
        merging, the per-band refinement and the overlap splitting are shared.

        Unlike `climbing_video.detect_vials` this needs only ONE frame, because
        `fly_detector` is a per-frame detector and has no motion to work with.
        The price is that it cannot tell a tube from a dark vertical fixture, so
        it returns a confidence and `needs_review` rather than a promise:

          width consistency  -- the tubes are identical, so their detected
                                widths should be
          spacing regularity -- they stand in a row, so the gaps should be equal
          signal strength    -- the darkness in the bands must exceed the
                                darkness between them by a clear margin

        Returns {"vials", "confidence", "needs_review", "reason", "bands",
        "components", "profile"}. On refusal `vials` is empty: a wrong region
        produces a confidently wrong count, which is worse than no count.
        """
        out = {"vials": [], "confidence": 0.0, "needs_review": True,
               "reason": "", "bands": [], "components": {}, "profile": None}
        if not _CV2 or frame is None:
            out["reason"] = ("No frame, or OpenCV is unavailable, so the vials "
                             "cannot be located in the image.")
            return out
        if int(n_vials) < 1:
            out["reason"] = f"n_vials = {n_vials} is not a number of vials."
            return out

        arr = np.asarray(frame)
        gray = (cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr)
        gray = gray.astype(np.float32)
        H, W = gray.shape[:2]
        out["profile"] = None

        # Darkness relative to the frame's own bright level, so the profile does
        # not depend on exposure. A robust bright reference (the 90th
        # percentile) rather than the max, which one specular highlight sets.
        bright = float(np.percentile(gray, 90))
        dark = np.clip(bright - gray, 0.0, None)
        profile = dark.sum(axis=0).astype(np.float32)
        out["profile"] = profile

        if n_vials == 1:
            out.update({
                "vials": [VialROI(x=margin_px, y=margin_px,
                                  w=max(1, W - 2 * margin_px),
                                  h=max(1, H - 2 * margin_px),
                                  label=(labels[0] if labels else "Vial_1"),
                                  vial_id=1)],
                "confidence": 1.0, "needs_review": False,
                "reason": "One vial: the region is the frame.",
            })
            return out

        # Absolute guard before any relative thresholding: on a featureless
        # frame a relative threshold carves n_vials "tubes" out of sensor noise
        # and reports high confidence, because noise is as self-consistent as
        # anything else. This is the same guard `climbing_video.detect_vials`
        # makes for the same reason.
        if float(profile.max()) <= 0 or float(np.ptp(profile)) < 1e-6:
            out["reason"] = ("The frame has no darker regions at all, so there "
                             "is nothing to locate. Draw the regions manually.")
            return out
        contrast = float(np.ptp(profile)) / max(float(profile.max()), 1e-9)
        if contrast < 0.05:
            out["reason"] = (
                f"The column darkness profile is almost flat (contrast "
                f"{contrast:.3f}), so no tube boundaries are visible in this "
                f"frame. Either the tubes fill the frame, the flies are not "
                f"visible against the background, or this is not a rack. Draw "
                f"the regions manually."
            )
            return out

        try:
            from modules.climbing_video import _bands_from_profile
        except ImportError:                          # pragma: no cover
            from climbing_video import _bands_from_profile
        bands = _bands_from_profile(profile, int(n_vials),
                                    min_width=max(8, W // (4 * int(n_vials))))
        out["bands"] = [(int(a), int(b)) for a, b in bands]
        if len(bands) < n_vials:
            out["reason"] = (
                f"Found {len(bands)} candidate tube region(s) but expected "
                f"{n_vials}. Draw the regions manually, or pass the number of "
                f"tubes actually in the frame."
            )
            return out

        widths = np.array([b - a for a, b in bands], dtype=float)
        centres = np.array([(a + b) / 2.0 for a, b in bands], dtype=float)
        w_cv = float(np.std(widths) / max(float(np.mean(widths)), 1e-9))
        width_score = float(np.clip(1.0 - w_cv / 0.25, 0.0, 1.0))
        if len(centres) > 2:
            gaps = np.diff(centres)
            g_cv = float(np.std(gaps) / max(float(np.mean(gaps)), 1e-9))
            spacing_score = float(np.clip(1.0 - g_cv / 0.25, 0.0, 1.0))
        else:
            spacing_score = 1.0
        in_band = np.zeros(W, bool)
        for a, b in bands:
            in_band[a:b] = True
        inside = float(np.median(profile[in_band])) if in_band.any() else 0.0
        outside = (float(np.median(profile[~in_band]))
                   if (~in_band).any() else 0.0)
        signal_score = float(np.clip((inside - outside) /
                                     max(inside, 1e-9), 0.0, 1.0))
        conf = float(round(width_score * spacing_score * signal_score, 4))
        out["components"] = {"width_consistency": round(width_score, 4),
                             "spacing_regularity": round(spacing_score, 4),
                             "signal_strength": round(signal_score, 4),
                             "profile_contrast": round(contrast, 4)}

        rows = max(1, int(rows))
        rh = H // rows
        vials, vid = [], 1
        for r in range(rows):
            for i, (a, b) in enumerate(bands):
                x = int(a) + margin_px
                w = max(1, int(b - a) - 2 * margin_px)
                vials.append(VialROI(
                    x=x, y=r * rh + margin_px, w=w,
                    h=max(1, rh - 2 * margin_px),
                    label=(labels[vid - 1] if labels and len(labels) >= vid
                           else f"Vial_{vid}"),
                    vial_id=vid,
                ))
                vid += 1
        out.update({
            "vials": vials, "confidence": conf,
            "needs_review": bool(conf < 0.60),
            "reason": ("" if conf >= 0.60 else
                       f"Detection confidence {conf:.2f} is below 0.60 "
                       f"(width {width_score:.2f}, spacing "
                       f"{spacing_score:.2f}, signal {signal_score:.2f}). The "
                       f"regions are returned so they can be corrected, not "
                       f"used as they are."),
        })
        return out


# ─── VialCount ────────────────────────────────────────────────────────────────
@dataclass
class VialCount:
    """
    Per-vial fly count result for one timepoint.

    Attributes
    ----------
    n_above   : flies detected above the scoring line (climbed)
    n_below   : flies detected below the scoring line (did not climb)
    total     : N_detected — flies the detector found. A QC observable, NOT a
                denominator (register N5).
    expected_n: N_start — flies loaded, from the loading record. The denominator.
    pct_above : 100 × n_above / N_start. NaN when there is no denominator or no
                usable measurement — never 0.0, which is a real rate.
    pi        : Performance Index = 100 × (n_above − n_below) / N_start.
    QC_flag   : "ok" | "count_mismatch:..." | "no_flies_detected" | "empty_roi"
                | "n_start_not_supplied"
    Confidence: 0–1 detection confidence estimate
    missingness: one of modules.missingness.CODES — what a NaN here means
    unusable_reason: the specific condition, in words, for a human reading a CSV

    Why PI treats an undetected fly as a non-climber
    ------------------------------------------------
    With N_start as the denominator, n_above + n_below can be less than N_start.
    The convention adopted is the manual-scoring one: the observer counts the
    flies that crossed the line out of the number LOADED, so a fly that cannot
    be located did not demonstrably climb. That is an assumption, not an
    observation, so whenever N_detected ≠ N_start the row carries a
    `count_mismatch` flag recording both numbers. The alternative — dropping
    the vial — discards a real measurement; the previous behaviour — shrinking
    the denominator — silently moved the vial toward the healthy phenotype.
    """
    vial_id:    int   = 1
    label:      str   = ""
    genotype:   str   = ""
    sex:        str   = ""
    treatment:  str   = ""
    expected_n: int   = 0
    n_above:    int   = 0           # FIX: was missing
    n_below:    int   = 0           # FIX: was missing
    total:      int   = 0
    # N8: these default to NaN, not 0.0. `PI = 0.0` is the PI of a vial with
    # half its flies above the line, so it cannot double as "no measurement".
    pct_above:  float = float("nan")
    pi:         float = float("nan")
    QC_flag:    str   = "ok"        # FIX: was missing
    Confidence: float = 1.0         # FIX: was missing
    missingness: str  = _MS.OBSERVED
    unusable_reason: str = ""
    # N3: how much of `total` came from the area-based merge correction rather
    # than from one blob per fly, and how many blobs were discarded as debris.
    # A vial whose count rests largely on the correction is a vial to re-check,
    # and merges are commonest exactly where flies are piled at the bottom.
    n_merged:   int   = 0
    n_debris:   int   = 0
    # N76: the blobs these counts came from, in CROP coordinates, and the
    # scoring line the detector actually used on this crop. `annotate_frame`
    # used to re-detect in order to draw, and it re-detected WITHOUT the
    # background frame -- so in mode="bg" the QC overlay was a different
    # computation from the numbers printed on it, and an operator checking the
    # overlay was not checking the counts. Keeping them makes the overlay a
    # rendering of the measurement rather than a second opinion about it.
    # Excluded from to_dict(): they are pixel coordinates, not a result.
    blobs:      list  = field(default_factory=list, repr=False)
    line_y:     float = float("nan")

    def _counts_observed(self) -> bool:
        """Were the above/below counts themselves measured?

        True for OBSERVED and for REFUSED — a refusal withholds the derived
        RATE (no N_start, so no denominator) while the counts behind it are
        real. False for MEASUREMENT_FAILED and NOT_MEASURED, where there is no
        count to report.
        """
        return self.missingness in (_MS.OBSERVED, _MS.REFUSED)

    def to_dict(self) -> dict:      # FIX: was missing
        return {
            "VialID":     self.vial_id,
            "Label":      self.label,
            "Genotype":   self.genotype,
            "Sex":        self.sex,
            "Treatment":  self.treatment,
            # N8, the subtle half: when the measurement FAILED, `n_above = 0`
            # is not an observation of zero climbers — it is the absence of an
            # observation, and a downstream `Count = 0` against `n = N_start`
            # computes PI = -100, i.e. a maximally impaired vial, from a vial
            # that was never scored. (Measured: a six-vial cohort with one
            # unscored vial read -40.0 instead of -28.0 until this line.)
            # What the detector genuinely observed — that it found no blobs —
            # is in N_detected, which stays 0.
            # A REFUSED row is different: the counts were measured, only the
            # rate is refused for want of a denominator, so they are kept.
            "n_above":    (self.n_above if self._counts_observed()
                           else float("nan")),
            "n_below":    (self.n_below if self._counts_observed()
                           else float("nan")),
            # `Total` is retained under its historical name for readers that
            # already parse it, but it is the DETECTED count and is now
            # published under that name too, beside the denominator.
            "Total":      self.total,
            "N_detected": self.total,
            "N_start":    self.expected_n,
            "Pct":        _round_or_nan(self.pct_above, 2),
            "PI":         _round_or_nan(self.pi, 2),
            "QC_flag":    self.QC_flag,
            "Confidence": round(self.Confidence, 3),
            "N_merged":   self.n_merged,
            "N_debris":   self.n_debris,
            _MS.COLUMN:   self.missingness,
            "unusable_reason": self.unusable_reason,
        }


# ─── counts_to_dataframe ──────────────────────────────────────────────────────
def counts_to_dataframe(          # FIX: was a broken @dataclass stub
    counts: List[VialCount],
    timepoint: int = 0,
    replicate: int = 1,
):
    """
    Convert a list of VialCount objects to a DroPhenix-compatible DataFrame.

    Returns
    -------
    pd.DataFrame with columns: VialID, Label, Genotype, Sex, Treatment,
    n_above, n_below, Total, N_detected, N_start, Pct, PI, QC_flag, Confidence,
    Missingness, unusable_reason, Time, Replicate
    """
    if not _PD:
        raise ImportError("pandas required for counts_to_dataframe()")
    rows = [vc.to_dict() for vc in counts]
    df = pd.DataFrame(rows)
    df["Time"]      = timepoint
    df["Replicate"] = replicate
    return df


# ─── DetectorConfig ───────────────────────────────────────────────────────────
@dataclass
class DetectorConfig:
    """
    Configuration for FlyDetector.

    threshold_frac : fraction of vial height defining climb/no-climb line
                     0.50 = midpoint (standard negative geotaxis protocol)
    min_area       : minimum blob area in px² (removes pixel noise)
    max_area       : upper blob-area bound. With split_merged on this is a
                     DEBRIS ceiling, not a fly filter: a blob above it is
                     discarded as debris, while a blob merely larger than one
                     fly is split by area (see split_merged). It used to drop
                     every merged blob, which is register finding N3.
    split_merged   : count a blob clearly larger than one fly as the number of
                     flies its area implies (N3). Off reproduces the
                     uncorrected behaviour for a historical run.
    merge_split_factor : a blob is treated as merged above this multiple of the
                     estimated single-fly area (1.6, as in climbing_video).
    debris_area_mult : a blob above this multiple of the single-fly area is
                     debris -- a shadow, a finger, a label -- not a pile of
                     flies, and is discarded and counted separately.
    max_merge_multiplicity : cap on the flies one blob may be counted as, and
                     the absolute debris ceiling in units of max_area. Kept
                     equal to debris_area_mult so the two rules agree.
    use_bg_subtraction : True = MOG2 (stateful); overridden by static bg_frame
    bg_history     : MOG2 history parameter (frames)
    bg_var_threshold: MOG2 variance threshold
    use_morphology : apply morphological open+close to foreground mask
    morph_kernel_size: kernel size for morphological ops (px)
    vial_preset    : layout preset name from VIAL_PRESETS
    """
    threshold_frac:    float = 0.50
    min_area:          int   = 20
    max_area:          int   = 800
    split_merged:      bool  = True
    merge_split_factor: float = 1.6
    debris_area_mult:  float = 8.0
    # Equal to debris_area_mult by construction: a blob larger than this many
    # flies is classified as debris, so capping the count lower would silently
    # truncate a pile the detector has already decided is made of flies.
    max_merge_multiplicity: int = 8
    # Detection mode:
    #   'dark' : flies are darker than the vial -> Otsu INVERSE threshold,
    #            background-independent (robust default for backlit/light-vial
    #            setups; works whether flies move or stay still).
    #   'bg'   : background subtraction (use a clean/median background frame;
    #            best when the vial has dark fixed features).
    mode:              str   = "dark"
    use_bg_subtraction:bool  = True
    bg_history:        int   = 200
    bg_var_threshold:  float = 40.0
    use_morphology:    bool  = True
    morph_kernel_size: int   = 3
    vial_preset:       str   = "Single vial (full frame)"
    sample_interval_sec: int = 5
    vial_rois:         list  = field(default_factory=list)
    brightness:        float = 0.0
    contrast:          float = 1.0

    def auto_rois(self, frame_w: int, frame_h: int) -> List[VialROI]:
        return VialROI.auto_grid(frame_w, frame_h, self.vial_preset)


def _single_fly_area(areas, min_area: float, split_factor: float = 1.6,
                     n_iter: int = 4):
    """
    Estimate the footprint of ONE fly from the blob areas in a single frame.

    The estimate has to be robust to the merges it is used to detect, which is
    circular, so it is resolved by iteration. `climbing_video` uses the median
    and documents the assumption that most blobs are single animals. That holds
    when the flies are spread through the tube, but not in the condition this
    correction exists for: with 30 flies piled into the bottom tenth of a vial
    the measured median blob area is 162 px^2 against a true single-fly area of
    96 px^2, so the median would be a merged pair and every count would come out
    at roughly half.

    Measured on the same frames, the 25th percentile of blob areas was 96.0 px^2
    in all four motion states -- single flies are present even in the severest
    pile -- so the iteration starts there and then refines: classify blobs
    against the current estimate, re-estimate from the blobs judged single, and
    repeat. It converges in one step on these data and degrades gracefully when
    nearly every blob is merged, in which case the estimate stays at the low
    quantile rather than drifting upward with the merges.

    Returns (single_fly_area_px, n_iterations_used).
    """
    a = np.asarray([x for x in areas if np.isfinite(x) and x >= min_area],
                   dtype=float)
    if a.size == 0:
        return float(max(min_area, 1.0) * 3.0), 0
    est = float(np.quantile(a, 0.25))
    used = 0
    for i in range(int(n_iter)):
        used = i + 1
        singles = a[a <= split_factor * est]
        if singles.size == 0:
            break
        new = float(np.median(singles))
        if abs(new - est) <= 1e-6 * max(est, 1.0):
            est = new
            break
        est = new
    return float(max(est, float(min_area))), used


# ─── FlyDetector ─────────────────────────────────────────────────────────────
class FlyDetector:
    """
    Stateful per-frame fly detector using OpenCV background subtraction.

    Two modes
    ---------
    Static bg (preferred):
        Pass a bg_frame captured post-tap (flies at bottom).
        Uses cv2.absdiff — no warmup required, more accurate.

    MOG2 (fallback):
        No bg_frame. Uses Gaussian mixture model.
        Requires ~30 warmup frames before counts are reliable.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._mog2 = None

    def reset(self):
        self._mog2 = None

    def warmup(self, frames: list):
        """Feed warm-up frames to MOG2 (only needed in stateful mode)."""
        if not _CV2:
            return
        self._ensure_mog2()
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            self._mog2.apply(cv2.GaussianBlur(gray, (5, 5), 0))

    # ── Low-level single-vial detection ──────────────────────────────────────
    def detect_detailed(
        self,
        frame,
        bg_frame=None,
    ) -> dict:
        """
        Detect flies in a single (pre-cropped) vial frame, with merge-splitting.

        Register N3. The area filter used to be `min_area <= area <= max_area`,
        which DISCARDS a blob containing two or more touching flies instead of
        counting them -- the merge correction both `climbing_video` and
        `climbing_band` apply was absent here. Because flies pile up at the base
        of the tube when they cannot climb, the number of merges is a function
        of the phenotype being measured: measured recall on the production
        default was 0.88 for healthy flies spread through the tube, 0.53 for an
        impaired cohort in the bottom fifth and 0.35 for a severe one in the
        bottom tenth. That is a confound perfectly correlated with the
        independent variable, it biases every disease comparison toward the
        null, and it makes a `Confidence < 0.5` filter delete the impaired
        cohort preferentially.

        What is done. Blobs at or above `min_area` are kept. The area of ONE fly
        is estimated from the blobs in this frame (see `_single_fly_area`). A
        blob above `merge_split_factor` times that area is counted as
        `round(area / single-fly area)` flies, capped at
        `max_merge_multiplicity`; a blob above `debris_area_mult` times it is
        debris and is discarded and counted. `max_area` becomes a debris
        ceiling rather than a fly filter.

        What this is NOT. It is a count correction, not a segmentation: the
        flies in a merged blob are all attributed to the side of the scoring
        line that the blob's centroid falls on. For a blob straddling the line
        that is an approximation, which is why `n_merged` is returned on every
        vial record.

        Returns a dict with `total`, `above`, `below`, `blobs`, `detail`,
        `n_single`, `n_merged`, `n_debris`, `typical_area_px` and `merge_note`.
        """
        if not _CV2 or frame is None:
            return {"total": 0, "above": 0, "below": 0, "blobs": [], "detail": [],
                    "n_blobs": 0, "n_single": 0, "n_merged": 0, "n_debris": 0,
                    "typical_area_px": float("nan"), "split_at_px": float("nan"),
                    "debris_at_px": float("nan"), "typical_area_iterations": 0,
                    "line_y": 0, "merge_note": "",
                    "error": ("no frame was supplied" if frame is None
                              else "OpenCV is not available")}

        cfg  = self.config
        h, w = frame.shape[:2]
        liney = int(h * cfg.threshold_frac)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Foreground (fly) mask. Precedence:
        #   1. mode='bg' AND a background frame is supplied -> absdiff
        #   2. mode='dark' (default) -> Otsu INVERSE threshold (dark flies = 255),
        #      background-independent and robust to stationary flies
        #   3. MOG2 (stateful) if explicitly requested
        #   4. last-resort dark Otsu inverse threshold
        mode = getattr(cfg, "mode", "dark")
        if bg_frame is not None and mode == "bg":
            bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY) if len(bg_frame.shape) == 3 else bg_frame
            bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
            fg = cv2.absdiff(blur, bg_blur)
            _, fg = cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)
        elif mode == "dark":
            # dark flies on a lighter vial -> keep pixels darker than Otsu cut
            _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif cfg.use_bg_subtraction:
            self._ensure_mog2()
            fg = self._mog2.apply(blur)
        else:
            _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup
        if cfg.use_morphology:
            k  = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (cfg.morph_kernel_size, cfg.morph_kernel_size),
            )
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

        _, thresh = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cand = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < cfg.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cand.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area))

        typical, iters = _single_fly_area(
            [a for _, _, a in cand], float(cfg.min_area),
            float(getattr(cfg, "merge_split_factor", 1.6)),
        )
        split_at = float(getattr(cfg, "merge_split_factor", 1.6)) * typical
        cap = int(getattr(cfg, "max_merge_multiplicity", 8))
        # Two debris ceilings, and the TIGHTER one applies. The relative ceiling
        # (debris_area_mult x the estimated single-fly area) is the informative
        # one whenever the frame contains enough blobs to estimate that area.
        # The absolute one, from the operator-supplied `max_area`, is what
        # remains when it does not: with a single enormous blob in the frame --
        # a shadow, a hand, the vial rack itself -- the self-calibrating
        # estimate takes that blob's own area as the size of one fly, and
        # without an absolute bound a 19,800 px^2 rectangle is reported as one
        # fly. `max_area` is read as the largest plausible SINGLE fly, so the
        # largest plausible blob is `cap` of them.
        debris_at = min(float(getattr(cfg, "debris_area_mult", 8.0)) * typical,
                        float(cap) * float(cfg.max_area))
        do_split = bool(getattr(cfg, "split_merged", True))

        total = above = 0
        n_merged = n_debris = n_single = 0
        blobs = []
        detail = []
        for cx, cy, area in cand:
            if area > debris_at:
                n_debris += 1
                detail.append({"cx": cx, "cy": cy, "area": area, "count": 0,
                               "class": "debris"})
                continue
            if not do_split and area > cfg.max_area:
                # The uncorrected path, kept reachable so a historical run can
                # be reproduced exactly.
                detail.append({"cx": cx, "cy": cy, "area": area, "count": 0,
                               "class": "dropped_over_max_area"})
                continue
            if do_split and area > split_at:
                k = int(np.clip(round(area / max(typical, 1.0)), 2, cap))
                n_merged += 1
                klass = "merged"
            else:
                k = 1
                n_single += 1
                klass = "single"
            total += k
            if cy < liney:
                above += k
            blobs.append((cx, cy, area))
            detail.append({"cx": cx, "cy": cy, "area": area, "count": k,
                           "class": klass})

        return {
            "total": int(total),
            "above": int(above),
            "below": int(total - above),
            "blobs": blobs,
            "detail": detail,
            "n_blobs": len(blobs),
            "n_single": int(n_single),
            "n_merged": int(n_merged),
            "n_debris": int(n_debris),
            "typical_area_px": float(typical),
            "split_at_px": float(split_at),
            "debris_at_px": float(debris_at),
            "typical_area_iterations": int(iters),
            "line_y": int(liney),
            "merge_note": (
                "A blob larger than "
                f"{float(getattr(cfg, 'merge_split_factor', 1.6)):g}x the estimated "
                f"single-fly area ({typical:.0f} px^2) is counted as "
                "round(area / single-fly area) flies. This is a COUNT CORRECTION, "
                "not a segmentation: it recovers how many animals are in the blob "
                "without claiming where each one is, so every fly in a merged "
                "blob is scored on the side of the line its blob centroid falls "
                "on. A blob straddling the line is therefore attributed as a "
                "unit, and n_merged is reported so the reader can see how much of "
                "the total rests on the correction."
            ),
        }

    def detect(
        self,
        frame,
        bg_frame=None,
    ) -> Tuple[int, int, list]:
        """
        Backwards-compatible wrapper: (total, above, blobs).

        `detect_detailed` is the full result, including the merge and debris
        counts that register N3 requires to be visible.
        """
        d = self.detect_detailed(frame, bg_frame)
        return d["total"], d["above"], d["blobs"]

    # ── Multi-vial counting (FIX: was missing entirely) ───────────────────────
    def count_in_vials(
        self,
        frame,
        vials: List[VialROI],
        bg_frame=None,
    ) -> List[VialCount]:
        """
        Count flies in every vial ROI for one video frame.

        Parameters
        ----------
        frame    : full BGR video frame
        vials    : list of VialROI objects defining crop regions
        bg_frame : optional full background frame (captured post-tap)

        Returns
        -------
        List[VialCount] — one entry per vial
        """
        if not _CV2 or frame is None:
            # No frame, no OpenCV: nothing was measured. This used to return
            # VialCount defaults, which were Pct 0.0 / PI 0.0 — a fabricated
            # measurement of a frame that does not exist.
            why = ("no frame was supplied" if frame is None
                   else "OpenCV is not available in this interpreter")
            return [VialCount(
                vial_id=v.vial_id, label=v.label,
                genotype=v.genotype, sex=v.sex, treatment=v.treatment,
                expected_n=int(v.expected_n or 0), Confidence=0.0,
                QC_flag="not_measured",
                missingness=_MS.NOT_MEASURED,
                unusable_reason=f"no detection attempted: {why}",
            ) for v in vials]

        results: List[VialCount] = []

        for v in vials:
            # Crop vial from frame
            sl_y, sl_x = v.as_slice()
            vial_crop = frame[sl_y, sl_x]
            if vial_crop.size == 0:
                # N8: an ROI that lands outside the frame is a FAILED
                # measurement, not an empty vial. Pct/PI stay NaN by default.
                results.append(VialCount(
                    vial_id=v.vial_id, label=v.label,
                    genotype=v.genotype, sex=v.sex, treatment=v.treatment,
                    expected_n=v.expected_n, QC_flag="empty_roi",
                    Confidence=0.0,
                    missingness=_MS.MEASUREMENT_FAILED,
                    unusable_reason=(
                        f"ROI ({v.x},{v.y},{v.w},{v.h}) lies outside the "
                        f"{frame.shape[1]}x{frame.shape[0]} frame: nothing was "
                        f"measured for this vial"
                    ),
                ))
                continue

            # Crop bg_frame to same ROI if provided
            bg_crop = None
            if bg_frame is not None:
                try:
                    bg_crop = bg_frame[sl_y, sl_x]
                except Exception:
                    bg_crop = None

            det = self.detect_detailed(vial_crop, bg_crop)
            total, above = det["total"], det["above"]
            n_merged, n_debris = det["n_merged"], det["n_debris"]
            det_blobs = list(det.get("blobs", []))
            det_line_y = float(det.get("line_y", float("nan")))
            below  = total - above

            # ── N5: the denominator ───────────────────────────────────────────
            # N_start, from the loading record. NOT `total`, which is what the
            # detector happened to find, and NOT max(N_start, total), which
            # concealed an overcount behind a tidy 100%.
            n_start = int(v.expected_n or 0)
            nan = float("nan")
            pct = pi = nan
            reason = ""
            miss = _MS.OBSERVED

            if n_start <= 0:
                # No loading record -> no denominator. The counts above are real
                # observations and are kept; the rate is refused.
                qc = "n_start_not_supplied"
                conf = 0.0
                miss = _MS.REFUSED
                reason = (
                    "N_start (flies loaded) was not supplied for this vial, so "
                    "no climbing rate can be computed; set VialROI.expected_n "
                    "from the loading record"
                )
            elif total == 0:
                # N8: not 0% climbed. Nothing was scored.
                qc = "no_flies_detected"
                conf = 0.0
                miss = _MS.MEASUREMENT_FAILED
                reason = (
                    f"no flies detected in a vial loaded with {n_start}: "
                    f"measurement failed, not zero climbing"
                )
            else:
                pct = above / n_start * 100.0
                # PI must be the same statistic as Pct, re-centred:
                #   PI = 100 * (2*Pct/100 - 1) = 100 * (2*above - n_start)/n_start
                # Using the DETECTED `below` here instead would make PI and Pct
                # disagree about the denominator — in the worked example above,
                # (12-13)/30 = -3.3 against a Pct of 40%, which is the PI of a
                # vial that is nearly half climbed. The project's documented
                # definition, and `normalizer.normalise`, both use
                # (2*Count - n)/n, so an unlocated fly is a non-climber here
                # exactly as it is there.
                pi  = (2.0 * above - n_start) / n_start * 100.0
                if total != n_start:
                    # A mismatch is reported, never corrected. Both numbers are
                    # in the row so a reader can judge the vial themselves.
                    qc = f"count_mismatch:detected={total},n_start={n_start}"
                    conf = round(min(total, n_start) / max(total, n_start), 3)
                    reason = (
                        f"detected {total} of {n_start} loaded flies; the "
                        f"{abs(n_start - total)} unaccounted fl"
                        f"{'y is' if abs(n_start - total) == 1 else 'ies are'} "
                        f"{'treated as non-climbers' if total < n_start else 'surplus blobs — check the ROI and the area filter'}"
                    )
                else:
                    qc = "ok"
                    conf = 1.0
            if n_merged or n_debris:
                bits = []
                if n_merged:
                    bits.append(f"merged_blobs={n_merged}")
                if n_debris:
                    bits.append(f"debris_blobs={n_debris}")
                qc = (qc + ";" if qc not in ("", "ok") else "") + ",".join(bits)                     if qc != "ok" else ",".join(bits)
                extra = (
                    f"{n_merged} blob(s) were larger than one fly and were counted "
                    f"by area (N3 merge correction), so part of this vial's total "
                    f"rests on that correction rather than on a one-blob-one-fly "
                    f"detection" if n_merged else ""
                )
                if n_debris:
                    extra = (extra + "; " if extra else "") + (
                        f"{n_debris} blob(s) were too large to be flies and were "
                        f"discarded as debris -- check the ROI")
                reason = (reason + "; " if reason else "") + extra

            results.append(VialCount(
                vial_id    = v.vial_id,
                label      = v.label,
                genotype   = v.genotype,
                sex        = v.sex,
                treatment  = v.treatment,
                expected_n = n_start,
                n_above    = above,
                n_below    = below,
                total      = total,
                pct_above  = pct,
                pi         = pi,
                n_merged   = n_merged,
                n_debris   = n_debris,
                blobs      = det_blobs,
                line_y     = det_line_y,
                QC_flag    = qc,
                Confidence = conf,
                missingness = miss,
                unusable_reason = reason,
            ))

        return results

    # ── Annotated frame (FIX: was missing entirely) ───────────────────────────
    def annotate_frame(
        self,
        frame,
        vials: List[VialROI],
        counts: List[VialCount],
        show_roi_boxes: bool = True,
        bg_frame=None,
    ):
        """
        Draw detection overlays on the full frame for QC visualisation.

        Draws per-vial:
        - Coloured bounding box (green=ok, yellow=low, red=no detection)
        - Threshold line (cyan dashed)
        - Count label: "↑N_above  ↓N_below  PI:XX"

        Register N76. This used to re-detect the blobs in order to draw them,
        and it re-detected WITHOUT the background frame, so in `mode="bg"` the
        overlay was computed differently from the counts printed on top of it:
        an operator inspecting the overlay to check a count was looking at a
        second opinion, not at the measurement. It also drew the scoring line
        at `v.y + v.h * threshold_frac` while the detector had used the real
        crop height -- the same number only when the ROI lies entirely inside
        the frame, and off by half the overhang when it does not.

        The blobs and the line now come from the `VialCount` the measurement
        produced. Nothing is re-detected. If a record predates this and carries
        no blobs, the label says `overlay:re-detected` so the overlay is never
        silently a different computation again -- and that fallback takes the
        same `bg_frame` the counts were computed with, so even then it is not
        the computation the defect was.

        Returns annotated BGR frame copy.
        """
        if not _CV2 or frame is None:
            return frame

        out      = frame.copy()
        count_map = {c.vial_id: c for c in counts}

        for v in vials:
            vc = count_map.get(v.vial_id)
            if vc is None:
                continue

            # Box colour by QC. A vial with no usable measurement is red, so
            # the operator sees it on the frame rather than in a CSV later.
            if not _MS.is_observed(vc.missingness):
                box_col = (0, 0, 220)
            elif vc.QC_flag == "ok":
                box_col = (0, 220, 60)
            else:
                box_col = (0, 165, 255)

            if show_roi_boxes:
                cv2.rectangle(out, (v.x, v.y), (v.x + v.w, v.y + v.h), box_col, 2)

            # The scoring line, in the frame the detector measured. `vc.line_y`
            # is in CROP coordinates and is what the detector compared every
            # centroid against, so it is the only line consistent with the
            # counts -- including when the ROI overhangs the frame edge and the
            # crop is shorter than `v.h`.
            sl_y, sl_x = v.as_slice()
            redetected = False
            blobs = list(vc.blobs or [])
            crop_line = vc.line_y
            if not math.isfinite(float(crop_line)) or (not blobs and vc.total > 0):
                # A record from before N76, or one produced elsewhere. Say so
                # rather than quietly drawing something else.
                vial_crop = frame[sl_y, sl_x]
                if vial_crop.size > 0:
                    # Use the background if the caller has it: re-detecting
                    # WITHOUT it is the original N76 defect, and the fallback
                    # must not reintroduce it.
                    bg_crop = (bg_frame[sl_y, sl_x]
                               if bg_frame is not None else None)
                    d = self.detect_detailed(vial_crop, bg_crop)
                    blobs = list(d["blobs"])
                    crop_line = float(d["line_y"])
                    redetected = True
            if not math.isfinite(float(crop_line)):
                crop_line = float(int(v.h * self.config.threshold_frac))

            liney = v.y + int(crop_line)
            cv2.line(out, (v.x, liney), (v.x + v.w, liney), (0, 255, 255), 1)

            for (cx, cy, _) in blobs:
                col = (0, 255, 0) if cy < crop_line else (0, 0, 255)
                cv2.circle(out, (v.x + int(cx), v.y + int(cy)), 4, col, -1)

            # Text overlay
            lbl = v.label if v.label else f"V{v.vial_id}"
            if _MS.is_observed(vc.missingness) and math.isfinite(float(vc.pi)):
                pi_txt = f"PI:{vc.pi:.0f}"
            else:
                # Never print "PI:nan" or, worse, a zero: say what happened.
                pi_txt = str(vc.missingness)
            txt = f"{lbl} | up:{vc.n_above} dn:{vc.n_below} {pi_txt}"
            if vc.n_merged:
                txt += f" merged:{vc.n_merged}"
            if redetected:
                txt += " overlay:re-detected"
            cv2.putText(
                out, txt,
                (v.x + 4, max(v.y - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv2.LINE_AA,
            )

        return out

    # ── Legacy single-frame annotator (kept for backward compat) ─────────────
    def annotate(self, frame, blobs: list, liney: int):
        if not _CV2 or frame is None:
            return frame
        out = frame.copy()
        cv2.line(out, (0, liney), (out.shape[1], liney), (0, 255, 255), 2)
        for (cx, cy, _) in blobs:
            cv2.circle(
                out, (cx, cy), 4,
                (0, 255, 0) if cy < liney else (0, 0, 255), -1,
            )
        return out

    def _ensure_mog2(self):
        if self._mog2 is None and _CV2:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history        = self.config.bg_history,
                varThreshold   = self.config.bg_var_threshold,
                detectShadows  = False,
            )
