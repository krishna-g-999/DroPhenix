"""
Climbing-assay video analysis: from a phone recording to scored counts.

WHAT THIS MODULE IS FOR
-----------------------
A laboratory records a negative-geotaxis trial on a phone: several graduated
cylinders side by side against a pale background, a hand-drawn line on each,
flies tapped to the bottom, camera left running. This module turns that
recording into the canonical DroPhenix table -- flies above the line and total
flies, per cylinder, at each scoring time -- with the quality control needed to
know whether those numbers can be trusted.

WHY IT IS SEPARATE FROM `video_analyzer.py`
-------------------------------------------
`video_analyzer.py`/`fly_detector.py` were written against synthetic clips with
a clean, fixed background and a fixed vial layout. On real recordings they fail
badly: on a representative laboratory frame containing roughly thirty flies the
existing detector found three. Two reasons, both structural:

  1. It thresholds the WHOLE FRAME. A phone frame contains a window, a table,
     a curtain and a bench; Otsu then separates "bright window" from "everything
     else" and the flies are lost in the second class.
  2. In `dark` mode it treats every dark pixel as a candidate fly. A graduated
     cylinder is covered in printed scale ticks and numerals, which are dark and
     which no intensity threshold can distinguish from a fly.

The fix for (2) is the organising idea of this module: the scale markings are
STATIC and the flies MOVE. A temporal median across the clip therefore
reconstructs the cylinder complete with its printed scale and *without* its
flies. Subtracting it removes every fixed dark feature at once -- ticks,
numerals, the hand-drawn line, label text -- and leaves the animals.

THE ONE MEASUREMENT THAT MATTERS MOST
-------------------------------------
The performance index needs two numbers: how many flies crossed the line, and
how many flies there are. These are not equally hard.

Counting flies ABOVE the line is comparatively easy: they are few, spread out,
and in the clean upper part of the tube. Counting ALL flies is much harder,
because at early times they are piled on the base of the cylinder, overlapping,
against a bright reflective foot.

So the total is taken from the user wherever possible -- they loaded the vial
and they know how many flies went in -- and the vision system is asked only for
the numerator. The detected total is still computed and reported, as a CHECK on
the user's number rather than a replacement for it. When the two disagree the
module says so instead of silently preferring one.

WHAT IT REFUSES TO DO
---------------------
It does not report counts from a vial whose region or scoring line could not be
located confidently; it returns the low confidence and asks for a manual region.
It does not hide an unstable count: the total number of flies in a sealed
cylinder is physically constant, so the variability of the detected total across
frames is a direct, ground-truth-free measure of detector reliability, and it is
reported with every result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as _e:      # pragma: no cover
    raise ImportError("OpenCV (cv2) is required for climbing_video.") from _e


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class VialSpec:
    """
    One cylinder: its pixel region and the height of its scoring line.

    `line_y` is an ABSOLUTE image row. Rows above it (smaller y) count as
    climbed, because image y increases downwards.
    """

    x0: int
    x1: int
    y_top: int
    y_bottom: int
    line_y: int
    label: str = ""
    line_confidence: float = float("nan")
    source: str = "auto"          # "auto" | "manual"

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def height(self) -> int:
        return int(self.y_bottom - self.y_top)

    def line_fraction(self) -> float:
        """Height of the line as a fraction of tube height (0 = bottom, 1 = top)."""
        if self.height <= 0:
            return float("nan")
        return float((self.y_bottom - self.line_y) / self.height)

    def to_dict(self) -> dict:
        return {
            "label": self.label, "x0": self.x0, "x1": self.x1,
            "y_top": self.y_top, "y_bottom": self.y_bottom,
            "line_y": self.line_y, "line_fraction": round(self.line_fraction(), 4),
            "line_confidence": self.line_confidence, "source": self.source,
        }


@dataclass
class DetectorParams:
    """
    Fly-detection parameters, in pixels at the video's native resolution.

    Defaults were calibrated on 1920x1080 phone footage of 50 ml graduated
    cylinders, where an adult fly occupies roughly 5-40 px^2. `min_area` and
    `max_area` are rescaled automatically by `autoscale_for` when the footage
    has a different resolution or the cylinders a different apparent width,
    because a fixed pixel area is meaningless across zoom levels.
    """

    diff_threshold: int = 26        # intensity below background to call a pixel "fly"
    min_area: int = 4
    max_area: int = 260
    max_aspect: float = 6.0         # reject long thin artefacts (edges, scratches)
    open_kernel: int = 2
    close_kernel: int = 3
    roi_inset_frac: float = 0.06    # trim this fraction off each side of the tube
    line_mask_px: int = 0           # rows around the line to ignore (0 = none)
    split_merged: bool = True       # split blobs that are clearly >1 fly by area
    register: bool = True           # align each frame to the background first

    def autoscale_for(self, vial_width_px: float,
                      reference_width_px: float = 100.0) -> "DetectorParams":
        """Rescale area limits by the square of the apparent cylinder width."""
        if not np.isfinite(vial_width_px) or vial_width_px <= 0:
            return self
        s = (float(vial_width_px) / float(reference_width_px)) ** 2
        out = DetectorParams(**self.__dict__)
        out.min_area = max(2, int(round(self.min_area * s)))
        out.max_area = max(out.min_area + 4, int(round(self.max_area * s)))
        return out


@dataclass
class Calibration:
    """
    Pixels-to-millimetres conversion for one recording.

    WHY THIS MATTERS MORE THAN IT LOOKS. A count of "flies above the line" is
    not comparable between laboratories, or even between two recordings in the
    same laboratory, because it depends on where somebody drew the line and how
    far away the camera was. Height climbed IN MILLIMETRES is comparable to
    anyone -- it is a physical quantity, and it is what the RING assay was
    originally defined on (Gargano et al. 2005 score distance climbed, not a
    binary pass).

    Supply ONE known physical dimension and everything else follows:

      tube_width_mm     PREFERRED. See the warning below.
      tube_height_mm    the height of the region the detector marks out -- NOT
                        the physical height of the cylinder. The detected
                        region spans from the base to the highest point any fly
                        reached, so it is shorter than the glassware and it
                        depends on how well the flies climbed. Giving the
                        catalogue height of the tube here will make every
                        distance wrong. Use the diameter instead unless you have
                        measured the marked region itself, or
      tube_width_mm     the outside diameter of the cylinder (a 50 ml
                        graduated cylinder is about 25 mm; a standard
                        Drosophila vial is 23.5 mm), or
      px_per_mm         directly, if the setup is already calibrated.

    Diameter is usually the easier one to state accurately, and it is measured
    across the vial region, which the detector already knows.
    """

    tube_height_mm: Optional[float] = None
    tube_width_mm: Optional[float] = None
    px_per_mm: Optional[float] = None
    source: str = ""

    def resolve(self, vial: "VialSpec") -> Optional[float]:
        """Return px_per_mm for this vial, or None if it cannot be determined."""
        if self.px_per_mm and self.px_per_mm > 0:
            return float(self.px_per_mm)
        if self.tube_height_mm and self.tube_height_mm > 0 and vial.height > 0:
            return float(vial.height) / float(self.tube_height_mm)
        if self.tube_width_mm and self.tube_width_mm > 0 and vial.width > 0:
            return float(vial.width) / float(self.tube_width_mm)
        return None


@dataclass
class ScoringRule:
    """
    How to decide whether a fly has "climbed".

    Three modes, because laboratories genuinely differ and a tool that supports
    only one of them is a tool for one laboratory:

      "drawn_line"      use the dark line detected on the cylinder. This is what
                        the supplied recordings use.
      "height_fraction" a fraction of the tube's height, e.g. 0.5 for the
                        midpoint. Needs no mark on the glass at all.
      "height_mm"       a physical height above the base, e.g. 80 mm. This is
                        the standard RING criterion and the only one that is
                        directly comparable between laboratories. Requires a
                        `Calibration`.

    The rule is applied to stored fly HEIGHTS, not baked into the detection, so
    a dataset can be re-scored under a different criterion -- including a
    criterion another laboratory uses, for comparison -- without touching the
    video again.
    """

    mode: str = "drawn_line"
    fraction: Optional[float] = None
    height_mm: Optional[float] = None

    def describe(self) -> str:
        if self.mode == "height_fraction":
            return f"above {self.fraction:.0%} of tube height"
        if self.mode == "height_mm":
            return f"above {self.height_mm:g} mm from the base"
        return "above the marked line"

    def line_row(self, vial: "VialSpec", px_per_mm: Optional[float]) -> Tuple[int, str]:
        """Resolve this rule to an image row for a given vial."""
        if self.mode == "height_fraction":
            f = float(self.fraction if self.fraction is not None else 0.5)
            return int(round(vial.y_bottom - f * vial.height)), ""
        if self.mode == "height_mm":
            if not px_per_mm:
                return vial.line_y, (
                    "A height in millimetres was requested but the recording is not "
                    "calibrated. Supply the cylinder diameter or the tube height in mm. "
                    "Falling back to the marked line."
                )
            return int(round(vial.y_bottom - float(self.height_mm) * px_per_mm)), ""
        return vial.line_y, ""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ═══════════════════════════════════════════════════════════════════════════
# 2. VIDEO PROBING AND BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════
def probe_video(path: str) -> VideoInfo:
    """Read basic properties. Raises with a clear message if the file cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(
            f"Could not open '{path}'. If this is a phone recording, the file may use a "
            "codec your OpenCV build cannot decode (HEVC/H.265 is common on iPhone). "
            "Re-export as H.264 MP4 and try again."
        )
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    cap.release()
    return VideoInfo(str(path), w, h, fps, n, n / max(fps, 1e-6), codec)


#: Above this average gap between wanted frames, seeking to each one beats decoding
#: everything in between. Measured on a 12 096-frame 100 fps recording: a seek costs
#: about 0.2 s, while grabbing a frame without decoding it costs about 0.002 s, so
#: sequential reading wins whenever it grabs fewer than ~95 frames per frame kept.
#: The threshold is set well below that crossing point, because being wrong in the
#: sequential direction is the cheaper mistake.
_SEQUENTIAL_MAX_GAP = 60


def _read_frames(path: str, frame_indices: Sequence[int]) -> List[np.ndarray]:
    """Read the requested frames as grey images, by whichever route is cheaper.

    Seeking to every frame is the obvious implementation and is what this did. It
    is also catastrophic for densely-sampled reads: `detect_taps` asks for 900
    frames spread over a clip, and 900 seeks took 177 s on one Day-3 recording --
    about 2.5 minutes per video before anything had been measured.

    When the wanted frames are close together, it is far cheaper to walk the file
    once and `grab()` past the unwanted ones, which advances the decoder without
    paying to decode a full frame. The same 900 frames come back in 25.8 s, a 6.8x
    speed-up, and byte-identical: 900 of 900 frames matched exactly.

    Sequential reading is NOT always better -- asking for three frames from a long
    recording would decode the whole file -- so the route is chosen from the average
    gap between the wanted frames.
    """
    want = sorted(set(int(x) for x in frame_indices))
    if not want:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    out: List[np.ndarray] = []
    span = want[-1] - want[0]
    if span / max(len(want), 1) <= _SEQUENTIAL_MAX_GAP and len(want) > 1:
        pos, k = 0, 0
        while k < len(want):
            if pos == want[k]:
                ok, img = cap.read()
                if not ok:
                    break
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                k += 1
            else:
                if not cap.grab():
                    break
            pos += 1
    else:
        for i in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if ok:
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def build_background(
    path: str,
    start_s: float = 4.0,
    n_samples: int = 40,
) -> np.ndarray:
    """
    Temporal-median background: the scene as it would look with no flies in it.

    Sampling starts after `start_s` so that the tap, and the hands that perform
    it, are excluded -- a hand present in many sampled frames would be baked
    into the background and then subtracted out of every later frame, which is
    the failure mode of averaging the first few frames instead.

    A MEDIAN is used rather than a mean because a fly that sits still for a
    quarter of the clip still leaves no trace in the median, whereas it leaves a
    permanent dark smudge in the mean.
    """
    info = probe_video(path)
    lo = int(start_s * info.fps)
    hi = max(lo + 1, info.n_frames - 5)
    if hi <= lo:
        lo, hi = 0, max(1, info.n_frames - 1)
    idx = np.linspace(lo, hi, int(n_samples)).astype(int)
    frames = _read_frames(path, idx)
    if len(frames) < 3:
        raise RuntimeError(f"Only {len(frames)} frames could be read from '{path}'.")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def align_to_background(frame_gray: np.ndarray,
                        background: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Shift a frame into register with the background, to sub-pixel precision.

    THIS IS NOT OPTIONAL, AND IT IS THE SINGLE MOST IMPORTANT STEP IN THE
    MODULE. The cylinders are tapped by hand and go on settling for the first
    ten seconds or so. The measured drift on real footage is only about ONE
    PIXEL -- but the printed scale on a graduated cylinder consists of lines one
    to two pixels wide, so a one-pixel misalignment makes every tick, numeral
    and label edge differ strongly from the background and register as a fly.

    The effect is not subtle. On a real recording, without registration the
    detected total in one cylinder read 111, 109, 109 and 160 flies over the
    first twelve seconds and then settled to 15; with registration it read 15,
    12, 13, 15 and then 15. Since the number of flies in a sealed cylinder
    cannot change, the first sequence is pure artefact -- and it is
    concentrated exactly in the early seconds, which is when a climbing assay
    is normally scored.

    Sub-pixel translation is estimated by phase correlation and removed by an
    affine warp. Rotation and scale are not corrected: on tripod-free but
    stationary footage the residual is dominated by translation, and fitting
    more parameters to a one-pixel displacement invites overfitting.
    """
    bgf = np.float32(background)
    (dx, dy), _ = cv2.phaseCorrelate(bgf, np.float32(frame_gray))
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return frame_gray, 0.0, 0.0
    # Guard against a wild estimate (occlusion, a hand entering the frame):
    # a real settling drift is a few pixels, not tens.
    if abs(dx) > 25 or abs(dy) > 25:
        return frame_gray, float(dx), float(dy)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(frame_gray, M, (frame_gray.shape[1], frame_gray.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, float(dx), float(dy)


def detect_tap_time(path: str, n_samples: int = 400) -> dict:
    """
    Locate the tap: the burst of whole-frame motion when the cylinders are
    knocked down and the operator's hands withdraw.

    Scoring times are measured from the END of that burst, because that is when
    the flies are at the bottom and free to climb. Timing from frame zero would
    include however long the operator spent settling the camera.
    """
    info = probe_video(path)
    idx = np.linspace(0, max(info.n_frames - 2, 1), int(n_samples)).astype(int)
    frames = _read_frames(path, idx)
    if len(frames) < 5:
        return {"available": False, "reason": "too few frames"}

    small = [cv2.resize(f, (240, 135)) for f in frames]
    motion = np.array([float(np.mean(cv2.absdiff(small[i], small[i - 1])))
                       for i in range(1, len(small))])
    times = idx[1:len(motion) + 1] / max(info.fps, 1e-6)

    quiet = motion[motion <= np.percentile(motion, 70)]
    base, spread = float(np.median(quiet)), float(np.std(quiet) + 1e-9)
    thr = base + 6.0 * spread

    active = motion > thr
    # The tap is the first sustained burst in the first third of the clip.
    end_i, start_i = None, None
    for i, a in enumerate(active):
        if a and start_i is None:
            start_i = i
        elif not a and start_i is not None:
            if times[start_i] < info.duration_s / 3.0:
                end_i = i
                break
            start_i = None
    if start_i is None:
        return {"available": False, "reason": "no motion burst detected",
                "suggested_t0_s": 0.0, "motion": motion, "times": times}
    if end_i is None:
        end_i = len(motion) - 1

    return {
        "available": True,
        "burst_start_s": float(times[start_i]),
        "burst_end_s": float(times[min(end_i, len(times) - 1)]),
        "suggested_t0_s": float(times[min(end_i, len(times) - 1)]),
        "threshold": thr,
        "baseline": base,
        "motion": motion,
        "times": times,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. AUTOMATIC VIAL AND LINE DETECTION
# ═══════════════════════════════════════════════════════════════════════════
def _activity_map(path: str, background: np.ndarray, start_s: float = 4.0,
                  n_samples: int = 40, thr: int = 25) -> np.ndarray:
    """
    Accumulate, over the clip, the count of frames in which each pixel was
    darker than the background.

    The result is a map of where flies have been -- which is, by definition, the
    inside of the cylinders. This is the most reliable cue available for
    locating the tubes: it needs no assumption about glass, glare, labels or
    background clutter, only that flies are the things that move.
    """
    info = probe_video(path)
    lo = int(start_s * info.fps)
    hi = max(lo + 1, info.n_frames - 5)
    idx = np.linspace(lo, hi, int(n_samples)).astype(int)
    acc = np.zeros(background.shape, np.float32)
    for g in _read_frames(path, idx):
        # Register before differencing, for the same reason as in counting: a
        # drifting camera makes every static edge in the scene "move", and an
        # unregistered activity map would then mark the printed scale, the
        # label and the bench edge as places where flies have been -- which is
        # exactly where the cylinders are NOT.
        g, _, _ = align_to_background(g, background)
        acc += (cv2.subtract(background, g) > thr).astype(np.float32)
    return acc


def _bands_from_profile(profile: np.ndarray, n_vials: int,
                        rel_thresh: float = 0.15,
                        min_width: int = 20) -> List[Tuple[int, int]]:
    smooth = cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32),
                              (31, 1), 0).ravel()
    peak = float(smooth.max())
    if peak <= 0:
        return []
    on = smooth > peak * rel_thresh
    bands, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_width:
                bands.append((start, i))
            start = None
    if start is not None and len(on) - start >= min_width:
        bands.append((start, len(on)))
    # Merge fragments before selecting. A vial holding relatively few flies does
    # not produce one continuous activity band: by chance no fly occupies some
    # horizontal slice of it during the sampled frames, and the band breaks in
    # two. Selecting the strongest bands at that point picks a HALF of a vial
    # and places the region off-centre, which then clips real animals out of
    # their own tube. Fragments separated by less than half a typical band width
    # belong to the same cylinder.
    if len(bands) > 2:
        # The merge threshold must come from the LAYOUT, not from the tube width.
        # A fixed fraction of tube width merges adjacent CYLINDERS whenever the
        # tubes are closer together than half a tube -- which is common on a
        # rack of four or more, and silently reduces the vial count.
        #
        # The gaps between bands are bimodal: small ones are cracks inside one
        # vial, large ones are the spaces between vials. Splitting at a fraction
        # of the median gap separates the two without assuming either scale.
        gaps = np.array([bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)],
                        dtype=float)
        max_gap = max(8.0, 0.40 * float(np.median(gaps)))
        merged = [list(bands[0])]
        for a, b in bands[1:]:
            if a - merged[-1][1] <= max_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        bands = [(a, b) for a, b in merged]

    if len(bands) > n_vials:
        bands = sorted(bands, key=lambda b: -float(smooth[b[0]:b[1]].sum()))[:n_vials]
    bands = sorted(bands)

    # Refine each band against ITS OWN peak rather than the global maximum.
    # Vials do not contain equal numbers of flies, and a busy vial sets the
    # global threshold too high for a quiet one -- whose region is then clipped
    # inward, cutting off the very animals it is supposed to score. Each band's
    # edges are therefore re-derived by walking outward from its own peak until
    # activity falls below the same fraction of that band's peak.
    refined = []
    for a, b in bands:
        seg_peak = float(smooth[a:b].max()) if b > a else 0.0
        if seg_peak <= 0:
            refined.append((a, b))
            continue
        cut = seg_peak * rel_thresh
        i = int(a + np.argmax(smooth[a:b]))
        lo = i
        while lo > 0 and smooth[lo - 1] > cut:
            lo -= 1
        hi = i
        while hi < len(smooth) - 1 and smooth[hi + 1] > cut:
            hi += 1
        # Only ever EXPAND. Walking out from the peak would otherwise re-split a
        # band that fragment-merging had just joined: the walk stops at the
        # internal dip that caused the fragmentation in the first place, and the
        # merge is silently undone.
        refined.append((min(lo, a), max(hi + 1, b)))

    # Drop any overlap created by the widening, splitting at the midpoint.
    refined.sort()
    for j in range(len(refined) - 1):
        a0, b0 = refined[j]
        a1, b1 = refined[j + 1]
        if b0 > a1:
            mid = (b0 + a1) // 2
            refined[j] = (a0, mid)
            refined[j + 1] = (mid, b1)
    return [(int(a), int(b)) for a, b in refined if b - a >= min_width]


def _find_line(background: np.ndarray, x0: int, x1: int,
               inset: float = 0.18) -> Tuple[int, float]:
    """
    Find the scoring line inside one cylinder: the darkest sustained image row.

    The search is inset from the tube edges so that the printed scale down one
    side, and the glass edge highlight down the other, do not dominate the row
    average. `depth` is how much darker that row is than the tube's typical row
    and is used as the confidence in the result.
    """
    pad = int(inset * (x1 - x0))
    col = background[:, x0 + pad:x1 - pad].astype(np.float32)
    if col.size == 0:
        return -1, 0.0
    rows = cv2.GaussianBlur(col.mean(axis=1).reshape(-1, 1), (1, 5), 0).ravel()
    lo, hi = int(0.12 * len(rows)), int(0.92 * len(rows))
    seg = rows[lo:hi]
    if seg.size == 0:
        return -1, 0.0
    y = lo + int(np.argmin(seg))
    depth = float(np.median(seg) - rows[y])
    return int(y), depth


def _tube_extent(activity: np.ndarray, x0: int, x1: int,
                 rel: float = 0.08) -> Tuple[int, int]:
    """Vertical extent of the tube, from where flies have been seen."""
    col = activity[:, x0:x1].sum(axis=1)
    col = cv2.GaussianBlur(col.reshape(-1, 1).astype(np.float32), (1, 31), 0).ravel()
    peak = float(col.max())
    if peak <= 0:
        return 0, activity.shape[0]
    on = np.where(col > peak * rel)[0]
    if on.size == 0:
        return 0, activity.shape[0]
    return int(on.min()), int(on.max())


def detect_vials(
    path: str,
    n_vials: Optional[int] = 3,
    background: Optional[np.ndarray] = None,
    start_s: float = 4.0,
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """
    Locate the cylinders and their scoring lines automatically.

    Returns a dict with `vials`, an overall `confidence` in [0, 1], the
    component scores, and `needs_review`. Confidence combines three checks that
    the known geometry must satisfy:

      width consistency  -- the cylinders are identical, so their detected
                            widths should be too
      spacing regularity -- they stand in a row, so the gaps should be equal
      line strength      -- a hand-drawn line is much darker than the glass;
                            a weak "line" means none was found

    Detection is a convenience, not a guarantee. On a set of 15 laboratory
    recordings this located all three cylinders in 11; the remaining 4 differed
    in framing or zoom enough to defeat it. That is why `needs_review` exists
    and why every caller must offer a manual region editor. A wrong region
    produces a confidently wrong count, which is worse than no count.
    """
    if background is None:
        background = build_background(path, start_s=start_s)

    auto_note = ""
    if n_vials is None:
        est = auto_n_vials(path, background=background, start_s=start_s)
        activity = est["activity"]
        n_vials = int(est["n_vials"])
        auto_note = f"Number of cylinders determined automatically: {n_vials}."
        if n_vials < 1:
            return {
                "vials": [], "confidence": 0.0, "needs_review": True,
                "reason": ("No cylinders could be found from fly activity. Either the "
                           "recording shows no moving animals, or the tubes are outside "
                           "the frame. Draw the regions manually."),
                "background": background, "activity": activity,
            }
    else:
        activity = _activity_map(path, background, start_s=start_s)

    profile = activity.sum(axis=0)

    # Absolute guard before any relative thresholding. `_bands_from_profile`
    # thresholds at a fraction of the profile's own maximum, so on a clip with
    # no moving animals at all it would happily carve three "cylinders" out of
    # sensor noise and report high confidence, because noise is as
    # self-consistent as anything else. Require a real amount of movement first.
    n_sampled = max(int(activity.max()), 1)
    moving_px = float((activity > 0.15 * n_sampled).sum())
    if moving_px < 0.0002 * activity.size:
        return {
            "vials": [], "confidence": 0.0, "needs_review": True,
            "reason": ("Almost nothing moves in this recording, so the cylinders cannot be "
                       "located from fly activity. Check that the clip shows live flies and "
                       "that the camera was running, or draw the regions manually."),
            "background": background, "activity": activity,
        }

    bands = _bands_from_profile(profile, n_vials)

    if len(bands) < n_vials:
        return {
            "vials": [], "confidence": 0.0, "needs_review": True,
            "reason": (f"Found {len(bands)} candidate cylinder region(s) but expected "
                       f"{n_vials}. Draw the regions manually."),
            "background": background, "activity": activity,
        }

    # Flies rarely occupy the last few percent of the tube's width (they walk on
    # the glass, and the near/far walls project inward), so the band where
    # activity is seen is systematically NARROWER than the tube. Widen it a
    # little so that a fly against the wall is not clipped out of its own vial.
    # The cylinders are physically identical, so their regions must be too.
    # Activity-derived widths are not: a vial holding fewer flies produces a
    # narrower activity band and its region gets clipped inward, cutting off
    # the animals nearest the glass -- the ones most likely to be missed anyway.
    # Imposing a COMMON WIDTH, taken as the widest band and centred on each
    # band's own centroid, applies the geometry we already know instead of
    # inferring it separately three times from noisy evidence.
    pad_frac = 0.06
    raw_widths = [b - a for a, b in bands]
    centres = sorted(int(round((a + b) / 2)) for a, b in bands)

    # MEDIAN width, not maximum. One band that has swallowed a neighbour, or a
    # stray region of clutter, inflates the maximum and then the common-width
    # rule applies that inflated width to every cylinder -- observed on a real
    # recording, where four bands each 307 px wide overlapped each other
    # wholesale. The median is unaffected by a single bad band.
    common_w = int(round(float(np.median(raw_widths)) * (1.0 + 2 * pad_frac)))

    # And cap it so that neighbouring regions can never overlap: two regions
    # sharing pixels would count the same fly in both vials.
    if len(centres) > 1:
        min_gap = int(min(np.diff(centres)))
        common_w = min(common_w, max(8, int(0.95 * min_gap)))

    bands = []
    for c in centres:
        a = max(0, c - common_w // 2)
        b = min(background.shape[1], a + common_w)
        a = max(0, b - common_w)
        bands.append((a, b))

    vials: List[VialSpec] = []
    extents = [_tube_extent(activity, a, b) for a, b in bands]

    # The vertical extent needs the same treatment as the width, and for a
    # sharper reason. `_tube_extent` returns the span in which flies were SEEN,
    # which is not the tube: the base is found reliably (flies always reach the
    # bottom, and on real footage the three bases agreed to within 4 px), but
    # the top is simply wherever that vial's best climber got to. On a real
    # recording the three apparent "tube heights" came out as 610, 559 and 525
    # px for three identical cylinders -- a 16% spread that is biology, not
    # geometry.
    #
    # Left uncorrected this quietly corrupts two things: a `height_fraction`
    # criterion means a different physical height in every vial, and the worse
    # a vial climbs the shorter its apparent tube becomes, so a poorly
    # performing vial is scored against a lower bar. Imposing a common height,
    # measured up from each vial's own well-determined base, removes both.
    common_h = int(round(float(np.median([b - t for t, b in extents]))))
    height_spread = (max(b - t for t, b in extents) -
                     min(b - t for t, b in extents)) / max(common_h, 1)

    line_problems: List[str] = []
    for i, (a, b) in enumerate(bands):
        _, y_bot = extents[i]
        y_top = max(0, y_bot - common_h)
        line_y, depth = _find_line(background, a, b)
        if line_y < 0:
            line_y = int(y_top + 0.5 * (y_bot - y_top))
        # HARD INVARIANT: a scoring line outside the tube is not a measurement.
        # `_find_line` takes the darkest horizontal feature, and when a cylinder
        # stands low in frame the bench mat at its base can be darker than the
        # drawn line -- measured on three TDP 10 mM recordings, where the line
        # was placed 330, 437 and 632 px too low, one of them at -1096% of tube
        # height, i.e. BELOW the base. Each was accepted, and every count scored
        # against it was meaningless while looking perfectly ordinary.
        lab = labels[i] if labels and i < len(labels) else f"R{i + 1}"
        if not (y_top <= line_y <= y_bot):
            frac = (y_bot - line_y) / max(y_bot - y_top, 1)
            line_problems.append(
                f"cylinder {lab}: the detected scoring line is outside the tube "
                f"({frac:.0%} of its height); the darkest horizontal feature is not "
                f"the drawn line here. Set the line manually."
            )
            line_y = int(y_top + 0.5 * (y_bot - y_top))
        vials.append(VialSpec(x0=int(a), x1=int(b), y_top=int(y_top),
                              y_bottom=int(y_bot), line_y=int(line_y),
                              label=lab, line_confidence=float(depth),
                              source="auto"))

    # Direct check, in place of trusting a threshold on an activity statistic:
    # look inside the regions just found and see whether any animals are
    # actually detectable there. A recording of an empty bench still contains
    # sensor noise and residual registration error, and a purely relative
    # threshold will carve three plausible "cylinders" out of it. Counting is
    # the unambiguous test -- a clip with no flies cannot be scored, whatever
    # its activity profile looks like.
    _info = probe_video(path)
    _frames = _read_frames(path, [int(0.5 * _info.n_frames)])
    n_seen = 0
    if _frames:
        _g, _, _ = align_to_background(_frames[0], background)
        n_seen = sum(count_in_vial(_g, background, v,
                                   DetectorParams(register=False))["n_total"]
                     for v in vials)
    if n_seen < 1:
        return {
            "vials": vials, "confidence": 0.0, "needs_review": True,
            "reason": ("No flies could be detected inside the regions found. Either the "
                       "recording contains no live animals, the cylinders were located "
                       "wrongly, or the flies are too small at this zoom to resolve. "
                       "Draw the regions manually and check the annotated frame."
                       + ((" ALSO: " + " ".join(line_problems)) if line_problems else "")),
            "problems": ["no flies detected in any region"] + line_problems,
            "width_cv": np.nan, "gap_cv": np.nan, "line_score": 0.0,
            "background": background, "activity": activity,
        }

    widths = np.array([v.width for v in vials], float)
    centres = np.array([(v.x0 + v.x1) / 2 for v in vials], float)
    gaps = np.diff(centres)
    depths = np.array([v.line_confidence for v in vials], float)

    width_cv = float(np.std(widths) / max(np.mean(widths), 1e-9))
    gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-9)) if gaps.size else 0.0
    # A hand-drawn line is typically 80-150 grey levels darker than the glass;
    # below ~40 the "line" is more likely to be shading or a label edge.
    line_score = float(np.clip(np.min(depths) / 80.0, 0.0, 1.0))

    conf = float(np.clip(
        (1.0 - np.clip(width_cv / 0.30, 0, 1)) * 0.35 +
        (1.0 - np.clip(gap_cv / 0.30, 0, 1)) * 0.30 +
        line_score * 0.35, 0.0, 1.0))

    problems = []
    if auto_note:
        pass
    if width_cv > 0.30:
        problems.append(f"cylinder widths differ by more than expected (CV {width_cv:.2f})")
    if gaps.size and gap_cv > 0.30:
        problems.append(f"cylinder spacing is irregular (CV {gap_cv:.2f})")
    if float(np.min(depths)) < 40:
        problems.append(f"scoring line is faint in at least one cylinder "
                        f"(contrast {float(np.min(depths)):.0f} grey levels)")
    # A line found outside the tube is a hard geometric failure, not a soft score:
    # it forces review regardless of how regular the spacing and contrast look.
    problems.extend(line_problems)
    if line_problems:
        conf = min(conf, 0.30)

    return {
        "vials": vials,
        "confidence": conf,
        "needs_review": bool(conf < 0.70 or problems),
        "width_cv": width_cv,
        "gap_cv": gap_cv,
        "line_score": line_score,
        "problems": problems,
        "reason": "; ".join(problems) if problems else "",
        "auto_n_vials_note": auto_note,
        "n_vials": len(vials),
        "background": background,
        "activity": activity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FLY DETECTION AND COUNTING
# ═══════════════════════════════════════════════════════════════════════════
def count_in_vial(
    frame_gray: np.ndarray,
    background: np.ndarray,
    vial: VialSpec,
    params: Optional[DetectorParams] = None,
) -> dict:
    """
    Detect flies in one cylinder in one frame and split them by the scoring line.

    Method: subtract the frame from the temporal-median background, so that
    every pixel darker than the empty scene becomes positive; threshold; clean
    with a morphological open (removes single-pixel noise) then close (rejoins
    a fly split by a glare highlight); label connected components; keep those
    whose area and aspect ratio are consistent with a fly.

    Blobs far larger than one fly are two or more animals touching. When
    `split_merged` is on they are counted as `round(area / typical_area)`
    individuals rather than as one, using the median area of the unambiguous
    blobs in the same frame as the estimate of one fly's size. This is a
    correction, not a segmentation: it recovers the count without claiming to
    know where each animal is, and the number of merged blobs is reported so a
    user can see how much of the total rests on it.
    """
    p = params or DetectorParams()
    p = p.autoscale_for(vial.width)

    if p.register:
        frame_gray, _, _ = align_to_background(frame_gray, background)

    inset = int(p.roi_inset_frac * vial.width)
    x0, x1 = vial.x0 + inset, vial.x1 - inset
    y0, y1 = max(vial.y_top, 0), min(vial.y_bottom, frame_gray.shape[0])
    if x1 <= x0 or y1 <= y0:
        return {"n_total": 0, "n_above": 0, "n_below": 0, "blobs": [],
                "n_merged": 0, "error": "empty region"}

    bg = background[y0:y1, x0:x1]
    fr = frame_gray[y0:y1, x0:x1]
    diff = cv2.subtract(bg, fr)

    mask = (diff > p.diff_threshold).astype(np.uint8)
    if p.open_kernel > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((p.open_kernel, p.open_kernel), np.uint8))
    if p.close_kernel > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((p.close_kernel, p.close_kernel), np.uint8))

    if p.line_mask_px > 0:
        ly = vial.line_y - y0
        lo = max(0, ly - p.line_mask_px)
        hi = min(mask.shape[0], ly + p.line_mask_px + 1)
        mask[lo:hi, :] = 0

    n_lab, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)

    kept = []
    for i in range(1, n_lab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if a < p.min_area:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > p.max_aspect:
            continue
        kept.append({"area": a, "cx": float(cent[i][0]) + x0,
                     "cy": float(cent[i][1]) + y0, "w": w, "h": h})

    # A single fly's footprint is estimated as the MEDIAN blob area in this
    # frame. The median is robust to the merged blobs themselves: as long as
    # most detections are single animals, it estimates one fly's size correctly
    # even when a quarter of the blobs are pairs.
    areas = [b["area"] for b in kept]
    typical = float(np.median(areas)) if areas else float(p.min_area * 3)
    typical = max(typical, float(p.min_area))

    # Blobs meaningfully larger than one fly are touching animals. The split
    # threshold is a multiple of the typical size, NOT `max_area`: `max_area`
    # is autoscaled to the apparent zoom and is deliberately generous, so a
    # merged pair sits far below it and would otherwise be counted as one
    # animal. Undercounting from merges is the dominant error at early scoring
    # times, when flies are piled at the base of the tube.
    split_at = 1.6 * typical

    n_total = 0
    n_above = 0
    n_merged = 0
    for b in kept:
        if p.split_merged and b["area"] > split_at:
            k = int(max(1, round(b["area"] / max(typical, 1.0))))
            n_merged += 1
        else:
            k = 1
        b["count"] = k
        n_total += k
        if b["cy"] < vial.line_y:          # smaller y = higher in the image
            n_above += k

    return {
        "n_total": int(n_total),
        "n_above": int(n_above),
        "n_below": int(n_total - n_above),
        "n_blobs": len(kept),
        "n_merged": int(n_merged),
        "typical_area_px": typical,
        "blobs": kept,
    }


def fly_heights(
    frame_gray: np.ndarray,
    background: np.ndarray,
    vial: VialSpec,
    params: Optional[DetectorParams] = None,
    px_per_mm: Optional[float] = None,
) -> dict:
    """
    Return the HEIGHT of every detected fly in one vial, rather than a count.

    This is the pivot that makes the pipeline lab-agnostic. Counting "how many
    are above the line" bakes one laboratory's scoring criterion into the
    expensive step. Extracting heights instead means the video is processed
    ONCE and can then be scored under any criterion -- a drawn line, a height
    fraction, a physical height in millimetres, or all three side by side -- and
    re-scored later to match a collaborator's protocol without touching the
    video again.

    Heights are returned three ways:
      `heights_px`   pixels above the base of the vial region
      `heights_frac` fraction of the tube height, 0 at the base, 1 at the top
      `heights_mm`   millimetres above the base, when a calibration is available

    A merged blob contributes its height once per fly it is estimated to
    contain, so the distribution is not biased against crowded regions.
    """
    res = count_in_vial(frame_gray, background, vial, params)
    hs_px: List[float] = []
    for b in res.get("blobs", []):
        h = float(vial.y_bottom - b["cy"])
        for _ in range(int(b.get("count", 1))):
            hs_px.append(h)
    arr = np.array(hs_px, dtype=float)
    out = {
        "heights_px": arr,
        "heights_frac": arr / vial.height if vial.height > 0 else arr * np.nan,
        "heights_mm": (arr / px_per_mm) if px_per_mm else None,
        "n": int(arr.size),
        "n_merged": res.get("n_merged", 0),
        "blobs": res.get("blobs", []),
    }
    if arr.size:
        out["mean_height_frac"] = float(np.mean(out["heights_frac"]))
        out["median_height_frac"] = float(np.median(out["heights_frac"]))
        if px_per_mm:
            out["mean_height_mm"] = float(np.mean(out["heights_mm"]))
            out["median_height_mm"] = float(np.median(out["heights_mm"]))
    return out


def score_heights(heights_px: np.ndarray, vial: VialSpec,
                  rule: ScoringRule, px_per_mm: Optional[float] = None) -> dict:
    """Apply a scoring rule to already-extracted heights."""
    row, note = rule.line_row(vial, px_per_mm)
    cutoff_px = float(vial.y_bottom - row)
    arr = np.asarray(heights_px, dtype=float)

    # REFUSE when the criterion lies outside the region that was actually
    # measured. If the cutoff is above the top of the detected region, no fly
    # can ever qualify and the honest answer is "not scoreable", NOT "zero
    # climbed" -- but zero is what a naive comparison returns, and it is
    # indistinguishable from a genuinely paralysed cohort.
    #
    # This is not hypothetical. On a real recording of WILD-TYPE flies the
    # detected region came out only 19 mm tall, an 80 mm criterion sat far above
    # it, and every timepoint in every vial reported 0% climbed. A wild-type
    # cohort scoring zero should be unbelievable on its face; the software
    # reported it without comment.
    unscoreable = False
    if cutoff_px > vial.height or cutoff_px < 0:
        unscoreable = True
        as_mm = f"{cutoff_px / px_per_mm:.0f} mm" if px_per_mm else f"{cutoff_px:.0f} px"
        region_mm = (f"{vial.height / px_per_mm:.0f} mm" if px_per_mm
                     else f"{vial.height} px")
        note = (note + " " if note else "") + (
            f"The criterion ({rule.describe()}, i.e. {as_mm} above the base) lies OUTSIDE the "
            f"measured region, which spans only {region_mm}. No fly could qualify, so this "
            "vial cannot be scored under this rule -- it is reported as missing rather than "
            "as zero. Either the cylinders were not detected over their full height, or the "
            "criterion is wrong for this setup."
        )

    n_above = int(np.sum(arr >= cutoff_px)) if (arr.size and not unscoreable) else 0
    return {
        "n_above": np.nan if unscoreable else n_above,
        "n_total": int(arr.size),
        "cutoff_px": cutoff_px,
        "cutoff_row": int(row),
        "cutoff_mm": (cutoff_px / px_per_mm) if px_per_mm else None,
        "unscoreable": unscoreable,
        "rule": rule.describe(),
        "note": note,
    }


def auto_n_vials(path: str, background: Optional[np.ndarray] = None,
                 start_s: float = 4.0, max_vials: int = 12) -> dict:
    """
    Work out HOW MANY cylinders are in shot, instead of being told.

    Different laboratories run different numbers of tubes side by side -- one,
    three, six, a whole RING rack. Requiring the number up front is a small ask
    that becomes a silent error the moment someone gets it wrong: asking for
    three when four are present makes the detector discard a real vial, and
    asking for four when three are present makes it invent one out of clutter.

    Bands are found from fly activity, fragments merged, and a band is accepted
    when it carries at least 25% of the strongest band's activity. That
    threshold is doing real work: a vial in which every fly died carries almost
    no activity and is correctly NOT reported as a vial, which is the honest
    outcome -- there is nothing there to score.
    """
    if background is None:
        background = build_background(path, start_s=start_s)
    activity = _activity_map(path, background, start_s=start_s)
    profile = activity.sum(axis=0)

    bands = _bands_from_profile(profile, n_vials=max_vials)
    if not bands:
        return {"n_vials": 0, "bands": [], "reason": "no fly activity found",
                "background": background, "activity": activity}

    smooth = cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (31, 1), 0).ravel()
    strength = np.array([float(smooth[a:b].sum()) for a, b in bands])
    keep = strength >= 0.25 * strength.max()
    bands = [b for b, k in zip(bands, keep) if k]

    return {
        "n_vials": len(bands),
        "bands": bands,
        "strengths": strength[keep].tolist(),
        "background": background,
        "activity": activity,
    }


def detect_framing_change(path: str, after_s: float = 4.0,
                          n_windows: int = 6, threshold: float = 10.0) -> dict:
    """
    Detect whether the camera was moved or zoomed part-way through a recording.

    The whole pipeline rests on a temporal-median background, which assumes the
    scene is fixed. If someone repositions the phone mid-clip, the median
    becomes a blend of two framings: every static feature appears doubled, the
    subtraction leaves ghosts everywhere, and detection collapses. One of the
    recordings supplied with this project (IMG_6032) is exactly this case -- the
    camera is moved closer at about 20 s -- and it failed every quality gate for
    reasons that made no sense until the framing change was found.

    Method: split the clip (after the tap) into windows, take a median frame of
    each, and compare them. Only the flies should differ between windows, and
    flies occupy a small fraction of the frame, so a large difference means the
    SCENE changed rather than its occupants.

    This is deliberately separate from tap detection. Comparing frames either
    side of the opening tap would always look like a scene change, because the
    operator's hands are in frame before it and gone afterwards.
    """
    info = probe_video(path)
    start = max(0.0, float(after_s))
    if info.duration_s - start < 4.0:
        return {"stable": True, "changes": [], "reason": "clip too short to check"}

    edges = np.linspace(start, info.duration_s - 0.3, n_windows + 1)
    meds, mids = [], []
    for i in range(n_windows):
        lo, hi = edges[i], edges[i + 1]
        idx = np.linspace(lo * info.fps, hi * info.fps, 5).astype(int)
        frames = [cv2.resize(f, (320, 180)) for f in _read_frames(path, idx)]
        if len(frames) < 2:
            continue
        meds.append(np.median(np.stack(frames), axis=0).astype(np.float32))
        mids.append(float((lo + hi) / 2))

    if len(meds) < 2:
        return {"stable": True, "changes": [], "reason": "too few windows"}

    diffs = [float(np.mean(np.abs(meds[i + 1] - meds[i]))) for i in range(len(meds) - 1)]
    changes = []
    for i, d in enumerate(diffs):
        if d > threshold:
            changes.append({"t0_s": mids[i + 1], "between_s": (mids[i], mids[i + 1]),
                            "difference": d})
    return {
        "stable": not changes,
        "changes": changes,
        "window_differences": diffs,
        "window_midpoints": mids,
        "threshold": threshold,
    }


def detect_taps(path: str, n_samples: int = 900, min_gap_s: float = 5.0) -> dict:
    """
    Find EVERY tap in a recording, not just the first.

    The RING assay is Rapid ITERATIVE Negative Geotaxis: many protocols tap the
    rack three to five times in succession and score after each. Supporting only
    a single tap would silently analyse one trial of a five-trial experiment and
    treat the rest as one long observation.

    Bursts of whole-frame motion separated by at least `min_gap_s` are reported
    as separate taps, each with the time the disturbance ENDS -- which is when
    the flies are down and free to climb, and therefore the zero for that
    trial's scoring times.

    On the recordings supplied with this project, every clip returns exactly ONE
    tap in the first second followed by two minutes of stillness. That is a
    single-tap protocol with a long observation window, and it is well suited to
    trajectory work because it yields a continuous climbing curve rather than a
    handful of endpoints.
    """
    info = probe_video(path)
    idx = np.linspace(0, max(info.n_frames - 2, 1), int(n_samples)).astype(int)
    frames = _read_frames(path, idx)
    if len(frames) < 5:
        return {"available": False, "reason": "too few frames", "taps": []}

    small = [cv2.resize(f, (320, 180)) for f in frames]
    motion = np.array([float(np.mean(cv2.absdiff(small[i], small[i - 1])))
                       for i in range(1, len(small))])
    times = idx[1:len(motion) + 1] / max(info.fps, 1e-6)

    # Discriminating a tap from flies moving needs care, and two obvious ideas
    # do not work:
    #
    #   - Frame displacement (phase correlation). The camera is stationary and
    #     only the operator's hands and the tubes move, so the whole-frame
    #     displacement during a tap is a fraction of a pixel -- measured at
    #     0.02-0.9 px on real recordings, indistinguishable from noise.
    #   - A threshold at "a few standard deviations above quiet". Flies moving
    #     en masse clear it easily; on a real single-tap recording this reported
    #     ELEVEN taps.
    #
    # What actually separates them is PROMINENCE. A tap involves hands entering
    # and leaving the frame and is by far the largest intensity event in the
    # clip -- roughly 17 units against a fly-motion background near 1. Requiring
    # a burst to reach a substantial fraction of the clip's peak motion is
    # scale-free: it does not care about resolution, lighting or how many flies
    # there are, and it still catches every tap of an iterative protocol,
    # because those taps resemble each other far more than they resemble flies.
    quiet = motion[motion <= np.percentile(motion, 70)]
    base, spread = float(np.median(quiet)), float(np.std(quiet) + 1e-9)
    peak = float(motion.max())
    thr = max(base + 8.0 * spread, 0.30 * peak)

    taps = []
    start = None
    for i, a in enumerate(motion > thr):
        if a and start is None:
            start = i
        elif not a and start is not None:
            t_start, t_end = float(times[start]), float(times[i - 1])
            if not taps or t_start - taps[-1]["end_s"] >= min_gap_s:
                taps.append({"start_s": t_start, "end_s": t_end,
                             "t0_s": t_end, "peak": float(motion[start:i].max())})
            else:
                taps[-1]["end_s"] = t_end
                taps[-1]["t0_s"] = t_end
            start = None
    if start is not None:
        t_start, t_end = float(times[start]), float(times[-1])
        if not taps or t_start - taps[-1]["end_s"] >= min_gap_s:
            taps.append({"start_s": t_start, "end_s": t_end, "t0_s": t_end,
                         "peak": float(motion[start:].max())})

    # A burst in the last seconds is the operator picking the rack up, not a tap.
    taps = [t for t in taps if t["t0_s"] < info.duration_s - 3.0]

    # Distinguish a TAP from a CAMERA MOVE. Both produce a large motion burst,
    # but they differ in what they leave behind: after a tap the scene looks the
    # same as before it, whereas after someone repositions or zooms the camera
    # the framing is permanently different.
    #
    # This matters more than a mislabelled event. A camera move mid-recording
    # makes the temporal-median background a blend of two different framings, so
    # every static feature appears doubled and nothing can be detected reliably.
    # One of the recordings supplied with this project (IMG_6032) is exactly
    # this case -- the camera is moved closer at about 20 s -- and it was the
    # video that failed every quality gate for reasons that were otherwise
    # mysterious.
    framing = detect_framing_change(path, after_s=(taps[0]["t0_s"] + 2.0) if taps else 4.0)
    camera_moves = framing["changes"]
    real_taps = taps

    protocol = ("single tap with a continuous observation window"
                if len(real_taps) == 1 else
                f"iterative: {len(real_taps)} taps" if len(real_taps) > 1 else
                "no tap detected -- check that the tap is inside the recording")
    if camera_moves:
        when = ", ".join("{:.1f}s".format(m["t0_s"]) for m in camera_moves)
        protocol += ("; WARNING: the camera appears to have been moved or zoomed "
                     f"{len(camera_moves)} time(s) during the recording (at {when}). "
                     "The background model assumes a fixed framing, so counts either side "
                     "of the move are not comparable. Trim the recording to a single "
                     "framing before scoring.")

    return {
        "available": bool(real_taps),
        "taps": real_taps,
        "n_taps": len(real_taps),
        "camera_moves": camera_moves,
        "framing_stable": not camera_moves,
        "all_bursts": taps,
        "threshold": thr,
        "duration_s": info.duration_s,
        "protocol": protocol,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. WHOLE-VIDEO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def _reliability_grade(cv_pct: float, median_count: float) -> str:
    """
    Grade the detector's reliability on one vial from the variability of its
    total count.

    A pass/fail flag is the wrong instrument here. The number of flies in a
    sealed cylinder is constant, so ALL of the observed variation is error --
    but not all of it is avoidable error. A fly that walks behind the label
    card, or onto the far side of a neighbour, genuinely cannot be counted in
    that frame. Perfect stability is therefore not achievable on real footage,
    and a binary threshold either condemns usable data or flatters unusable
    data depending on where it is set.

    Observed on 15 laboratory recordings, per-vial variation ran from about 11%
    to over 100%; the grades below were set against that distribution.
    """
    if not np.isfinite(cv_pct) or median_count < 1:
        return "unusable (no flies detected)"
    if cv_pct <= 15:
        return "good"
    if cv_pct <= 30:
        return "acceptable - supply the true total"
    if cv_pct <= 60:
        return "poor - check the annotated frame"
    return "unusable - region or detection is wrong"


def analyze_climbing_video(
    path: str,
    vials: Optional[Sequence[VialSpec]] = None,
    n_vials: int = 3,
    scoring_times_s: Sequence[float] = (5, 10, 15, 20, 30, 60),
    t0_s: Optional[float] = None,
    n_total_flies: Optional[Dict[str, int]] = None,
    params: Optional[DetectorParams] = None,
    stability_samples: int = 25,
    average_frames: int = 5,
    plausible_max_flies: int = 60,
    calibration: Optional[Calibration] = None,
    scoring_rule: Optional[ScoringRule] = None,
    scoring_mode: str = "net",
    crossing_step_s: float = 1.0,
) -> dict:
    """
    Score a whole recording.

    Parameters
    ----------
    vials           : explicit regions. If omitted they are detected, and the
                      detection confidence is returned for the caller to act on.
    scoring_times_s : seconds AFTER the tap at which to count. A single video
                      yields the whole climbing time-course, so there is no
                      reason to keep only one endpoint.
    t0_s            : time of the tap. Detected automatically when omitted.
    n_total_flies   : {vial label: number loaded}. Supplied by the operator; used
                      as the denominator and compared against the detected total.
    plausible_max_flies : a climbing-assay cylinder holds tens of flies, not
                      hundreds. A detected total above this is reported as
                      implausible regardless of how self-consistent it is,
                      because a detector counting label text or bench clutter
                      can be perfectly repeatable and still wrong -- confidence
                      in the REGION and stability of the COUNT are both blind to
                      that failure, so a third, independent check is needed.
    average_frames  : count over this many consecutive frames at each scoring
                      time and take the median. A single frame can lose a fly to
                      a glare highlight or a wing flick; the median over a few
                      neighbouring frames is far steadier and costs nothing,
                      since flies move only a little in 0.2 s.

    Returns a dict with `counts` (tidy DataFrame), `qc`, `vials` and `info`.
    """
    info = probe_video(path)
    background = build_background(path)

    det = None
    if vials is None:
        det = detect_vials(path, n_vials=n_vials, background=background)
        vials = det["vials"]
        if not vials:
            return {"ok": False, "reason": det.get("reason", "no cylinders found"),
                    "detection": det, "info": info}

    tap = detect_taps(path)
    if t0_s is None:
        t0_s = float(tap["taps"][0]["t0_s"]) if tap.get("taps") else 0.0
    else:
        tap = dict(tap); tap["user_supplied_t0"] = float(t0_s)
    framing_ok = bool(tap.get("framing_stable", True))

    p = params or DetectorParams()
    # Registration is a whole-frame operation, so it is done ONCE per frame here
    # and switched off inside the per-vial counter, which would otherwise repeat
    # the same phase correlation for every cylinder.
    p_novial = replace(p, register=False)
    rule = scoring_rule or ScoringRule("drawn_line")
    rule_notes: List[str] = []
    drifts: List[Tuple[float, float]] = []

    cap = cv2.VideoCapture(str(path))

    def frames_at(t_s: float, k: int) -> List[np.ndarray]:
        f0 = int((t0_s + t_s) * info.fps)
        got = []
        for j in range(k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(info.n_frames - 1, f0 + j)))
            ok, img = cap.read()
            if ok:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if p.register:
                    g, dx, dy = align_to_background(g, background)
                    drifts.append((dx, dy))
                got.append(g)
        return got

    rows = []
    for t in scoring_times_s:
        if t0_s + t > info.duration_s:
            for v in vials:
                rows.append({"vial": v.label, "time_s": float(t), "usable": False,
                             "reason": f"video ends at {info.duration_s:.1f}s"})
            continue
        grabs = frames_at(float(t), average_frames)
        if not grabs:
            continue
        for v in vials:
            ppm = calibration.resolve(v) if calibration else None
            hh = [fly_heights(g, background, v, p_novial, px_per_mm=ppm) for g in grabs]
            scored = [score_heights(h["heights_px"], v, rule, ppm) for h in hh]
            unscoreable = any(r.get("unscoreable") for r in scored)
            above = (np.nan if unscoreable
                     else int(np.median([r["n_above"] for r in scored])))
            total = int(np.median([r["n_total"] for r in scored]))
            merged = int(np.median([h["n_merged"] for h in hh]))
            mean_frac = float(np.median([h.get("mean_height_frac", np.nan) for h in hh]))
            mean_mm = (float(np.median([h.get("mean_height_mm", np.nan) for h in hh]))
                       if ppm else np.nan)
            if scored and scored[0]["note"] and scored[0]["note"] not in rule_notes:
                rule_notes.append(scored[0]["note"])
            supplied = None
            if n_total_flies:
                # "__all__" lets a manifest state one total that applies to every
                # cylinder, without having to name labels the detector has not
                # assigned yet.
                supplied = n_total_flies.get(v.label, n_total_flies.get("__all__"))
            denom = supplied if supplied else total
            rows.append({
                "vial": v.label, "time_s": float(t), "usable": True,
                "n_above_detected": above,
                "n_total_detected": total,
                "n_total_supplied": supplied,
                "n_total_used": denom,
                "n_merged_blobs": merged,
                "unscoreable": bool(unscoreable),
                "proportion_climbed": ((above / denom) if (denom and not unscoreable)
                                       else np.nan),
                "PI": (((2.0 * above / denom) * 100.0 - 100.0)
                       if (denom and not unscoreable) else np.nan),
                # Height climbed is reported alongside the count because it is
                # the lab-independent quantity: it does not depend on where
                # anyone drew a line, and it uses every fly rather than
                # collapsing the cohort to a proportion.
                "mean_height_frac": mean_frac,
                "mean_height_mm": mean_mm,
                "scoring_rule": rule.describe(),
            })

    # ── the scoring rule: crossings, not a snapshot ──────────────────────────
    # The published negative-geotaxis protocols count the flies that have REACHED
    # OR PASSED the line within the window, accumulating from the tap; DroPhenix
    # originally reported how many were above it at the instant of the timepoint.
    # Those differ by every fly that crossed earlier and wandered back down.
    # Measured against 120 hand-scored cells, changing only this definition moved
    # the mean error from 6.4 flies to 3.1 -- against a 3-fly floor set by the
    # laboratory's own repeat taps. It is the largest single improvement in this
    # module and it is not a detector change. See `climbing_detectors`.
    #
    # Implemented here rather than delegated because `climbing_detectors` imports
    # this module, and a lazy import to dodge the cycle would hide the dependency.
    if scoring_mode not in ("net", "max_reached", "crossings", "instantaneous"):
        raise ValueError("scoring_mode must be 'net' (the protocol's up-minus-down "
                         "count, i.e. how many are standing above the line) or "
                         "'max_reached' (the most ever above at one moment)")
    scoring_mode = {"instantaneous": "net", "crossings": "max_reached"}.get(
        scoring_mode, scoring_mode)
    if scoring_mode == "max_reached" and rows and scoring_times_s:
        t_max = float(max(scoring_times_s))
        grid = np.arange(0.0, min(t_max + crossing_step_s,
                                  max(info.duration_s - t0_s, 0.0)),
                         float(crossing_step_s))
        fine: Dict[str, List[float]] = {v.label: [] for v in vials}
        for t in grid:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0_s + t) * info.fps))
            ok, img = cap.read()
            if not ok:
                for v in vials:
                    fine[v.label].append(np.nan)
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if p.register:
                g, dx, dy = align_to_background(g, background)
                drifts.append((dx, dy))
            for v in vials:
                ppm = calibration.resolve(v) if calibration else None
                hh = fly_heights(g, background, v, p_novial, px_per_mm=ppm)
                sc = score_heights(hh["heights_px"], v, rule, ppm)
                fine[v.label].append(np.nan if sc.get("unscoreable")
                                     else float(sc["n_above"]))
        for lab, vals in fine.items():
            a = pd.Series(vals, dtype=float).ffill().bfill()
            if not a.notna().any():
                continue
            # A ratchet cannot be undone, so one spurious detection would raise the
            # count for the rest of the clip; requiring a fly to appear in 2 of 3
            # consecutive samples costs one sample of lag and prevents that.
            ratchet = np.maximum.accumulate(
                a.rolling(3, center=True, min_periods=1).median().to_numpy())
            for i, row in enumerate(rows):
                if row.get("vial") != lab or not row.get("usable"):
                    continue
                k = int(np.searchsorted(grid, float(row["time_s"]), side="right")) - 1
                if k < 0 or k >= ratchet.size or pd.isna(row.get("n_above_detected")):
                    continue
                rows[i]["n_above_instantaneous"] = row["n_above_detected"]
                rows[i]["n_above_detected"] = float(ratchet[k])
                den = row.get("n_total_used")
                if den:
                    rows[i]["proportion_climbed"] = float(ratchet[k]) / den
                    rows[i]["PI"] = (2.0 * float(ratchet[k]) / den) * 100.0 - 100.0

    # ── stability QC: the number of flies in a sealed tube cannot change ─────
    qc_rows = []
    lo = t0_s + 3.0
    hi = min(info.duration_s - 0.5, t0_s + 115.0)
    if hi > lo:
        ts = np.linspace(lo, hi, int(stability_samples))
        per_vial: Dict[str, List[int]] = {v.label: [] for v in vials}
        for t in ts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info.fps))
            ok, img = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if p.register:
                g, dx, dy = align_to_background(g, background)
                drifts.append((dx, dy))
            for v in vials:
                per_vial[v.label].append(count_in_vial(g, background, v, p_novial)["n_total"])
        for lab, vals in per_vial.items():
            arr = np.array(vals, float)
            if arr.size < 3:
                continue
            _nt = n_total_flies or {}
            supplied = _nt.get(lab, _nt.get("__all__"))
            med = float(np.median(arr))
            # A near-zero median makes a coefficient of variation meaningless
            # (it explodes to 1e13 and looks like a catastrophic instability
            # when the real finding is simply that nothing was detected). Report
            # it as undefined and let the "no flies detected" message carry the
            # information instead.
            cv_pct = float(np.std(arr) / med * 100.0) if med >= 1.0 else float("nan")
            qc_rows.append({
                "vial": lab,
                "detected_total_median": med,
                "detected_total_min": float(arr.min()),
                "detected_total_max": float(arr.max()),
                "detected_total_cv_pct": cv_pct,
                "n_supplied": supplied,
                "supplied_vs_detected_diff": (med - supplied) if supplied else np.nan,
                "reliability": _reliability_grade(cv_pct, med),
                "stable": bool(np.isfinite(cv_pct) and cv_pct <= 15.0),
                "implausible_count": bool(med > plausible_max_flies),
            })
    cap.release()

    counts = pd.DataFrame(rows)
    if not counts.empty:
        counts["scoring_mode"] = scoring_mode
    qc = pd.DataFrame(qc_rows)

    warnings: List[str] = []
    if det is not None and det.get("needs_review"):
        warnings.append(
            f"Cylinder detection confidence is {det['confidence']:.2f}"
            + (f" ({det['reason']})." if det.get("reason") else ".")
            + " Check the regions on the annotated frame and correct them before "
              "using these counts."
        )
    if not qc.empty:
        for _, r in qc[qc["implausible_count"]].iterrows():
            warnings.append(
                f"Vial {r['vial']}: {r['detected_total_median']:.0f} objects were detected, "
                f"which is far more flies than a climbing-assay cylinder holds. The region "
                "is almost certainly not on a cylinder, or non-fly features (label text, "
                "bench clutter, reflections) are being counted. Do not use these numbers; "
                "set the region manually."
            )

        bad = qc[(~qc["stable"]) & (~qc["implausible_count"])]
        for _, r in bad.iterrows():
            if not np.isfinite(r["detected_total_cv_pct"]):
                warnings.append(
                    f"Vial {r['vial']}: essentially no flies were detected "
                    f"(median {r['detected_total_median']:.0f} across the clip). Either this "
                    "region is not on a cylinder, or the flies are too small or too "
                    "low-contrast at this zoom. Check the annotated frame and set the "
                    "region manually."
                )
                continue
            warnings.append(
                f"Vial {r['vial']}: the detected total varied from "
                f"{r['detected_total_min']:.0f} to {r['detected_total_max']:.0f} across the "
                f"clip ({r['detected_total_cv_pct']:.0f}% variation). The number of flies in a "
                "sealed cylinder is constant, so this variation is detector error. Treat the "
                "counts as unreliable and supply the true totals."
            )
        if n_total_flies:
            for _, r in qc.iterrows():
                d = r["supplied_vs_detected_diff"]
                if pd.notna(d) and r["n_supplied"] and abs(d) > 0.2 * r["n_supplied"]:
                    warnings.append(
                        f"Vial {r['vial']}: you supplied {r['n_supplied']:.0f} flies but the "
                        f"detector sees about {r['detected_total_median']:.0f}. A gap this large "
                        "means either flies are being missed (often piled at the base) or "
                        "something else is being counted. The supplied number is used as the "
                        "denominator; the numerator may be affected too."
                    )
    if not counts.empty and "n_merged_blobs" in counts:
        tot_merged = float(counts["n_merged_blobs"].sum())
        if tot_merged > 0.25 * max(float(counts["n_above_detected"].sum()), 1):
            warnings.append(
                "A large share of detections were merged blobs split by area. Counts at early "
                "times, when flies cluster at the bottom, are the least reliable."
            )

    if not framing_ok:
        when = ", ".join("{:.0f}s".format(m["t0_s"]) for m in tap.get("camera_moves", []))
        warnings.insert(0, (
            f"THE CAMERA WAS MOVED OR ZOOMED DURING THIS RECORDING (around {when}). Every "
            "number below is unreliable: the background model assumes a fixed framing, so a "
            "mid-clip move makes it a blend of two scenes and leaves ghost features "
            "everywhere. Trim the video to a single framing and score the pieces separately."
        ))
    if tap.get("n_taps", 0) > 1 and framing_ok:
        warnings.append(
            f"{tap['n_taps']} taps were detected, so this looks like an iterative protocol. "
            "Scoring times are measured from the FIRST tap; if each tap begins a separate "
            "trial, analyse the segments separately."
        )
    warnings.extend(rule_notes)
    if calibration is None:
        warnings.append(
            "No physical calibration supplied, so heights are reported only as a fraction "
            "of tube height. Give the cylinder diameter in mm (about 25 mm for a 50 ml "
            "graduated cylinder, 23.5 mm for a standard Drosophila vial) and climbing "
            "height will also be reported in millimetres, which is comparable between "
            "laboratories and zoom levels in a way that a count above a hand-drawn line "
            "is not."
        )

    return {
        "ok": True,
        "info": info,
        "calibration": calibration,
        "scoring_rule": rule,
        "vials": list(vials),
        "detection": det,
        "tap": tap,
        "t0_s": float(t0_s),
        "counts": counts,
        "qc": qc,
        "warnings": warnings,
        "background": background,
        "params": p,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. ANNOTATION AND EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def annotate_frame(
    path: str,
    vials: Sequence[VialSpec],
    background: np.ndarray,
    at_time_s: float,
    params: Optional[DetectorParams] = None,
    t0_s: float = 0.0,
) -> np.ndarray:
    """
    Render a frame with the regions, the scoring lines and every detection drawn
    on it. This is the single most useful output of the module: a user can
    confirm or reject the analysis by looking at one picture, which no summary
    statistic allows them to do.
    """
    info = probe_video(path)
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0_s + at_time_s) * info.fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame at {at_time_s}s.")

    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    for v in vials:
        res = count_in_vial(g, background, v, params)
        cv2.rectangle(vis, (v.x0, v.y_top), (v.x1, v.y_bottom), (255, 180, 0), 2)
        cv2.line(vis, (v.x0, v.line_y), (v.x1, v.line_y), (0, 200, 255), 2)
        for b in res["blobs"]:
            above = b["cy"] < v.line_y
            col = (0, 220, 0) if above else (0, 0, 235)
            r = max(4, int(np.sqrt(max(b["area"], 1))))
            cv2.circle(vis, (int(b["cx"]), int(b["cy"])), r, col, 1)
        cv2.putText(vis, f"{v.label}: {res['n_above']}/{res['n_total']}",
                    (v.x0, max(v.y_top - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 3)
        cv2.putText(vis, f"{v.label}: {res['n_above']}/{res['n_total']}",
                    (v.x0, max(v.y_top - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 1)
    cv2.putText(vis, f"t = {at_time_s:.0f}s after tap   (green = above line)",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(vis, f"t = {at_time_s:.0f}s after tap   (green = above line)",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1)
    return vis


def analyze_batch(
    manifest: "pd.DataFrame | str",
    video_dir: Optional[str] = None,
    scoring_times_s: Sequence[float] = (5, 10, 15, 20, 30, 60),
    calibration: Optional[Calibration] = None,
    scoring_rule: Optional[ScoringRule] = None,
    n_vials: Optional[int] = None,
    params: Optional[DetectorParams] = None,
    min_confidence: float = 0.70,
    progress=None,
) -> dict:
    """
    Score a whole folder of recordings in one run and return ONE analysis-ready
    table.

    "Batch mode" simply means this: instead of opening fifteen videos one at a
    time, re-entering the genotype and treatment for each and merging fifteen
    CSVs by hand, you write the metadata once in a manifest and the pipeline
    does the rest. That matters beyond convenience -- hand-merging is where
    mislabelling happens, and a mislabelled treatment column is not recoverable
    from the data afterwards.

    MANIFEST COLUMNS
      video       file name (or path) of the recording        [required]
      Genotype, Treatment, Sex                                [required]
      Age, TimeOfDay, ExperimentID, Batch, Compound, Dose     [optional]
      n_per_vial  flies loaded, if the same in every cylinder [optional]
      n_R1, n_R2, ...  flies loaded in a named cylinder       [optional]

    Every video is scored independently -- background, regions, tap time and
    calibration are all per-video, because framing and zoom change between
    recordings. What is shared is only the metadata and the scoring rule.

    A video whose region confidence falls below `min_confidence` is REPORTED AND
    EXCLUDED from the merged table rather than quietly contributing suspect
    counts. It appears in `excluded` with its reason, so nothing disappears
    silently; scoring it by hand afterwards is a deliberate act.
    """
    if isinstance(manifest, str):
        man = pd.read_csv(manifest)
    else:
        man = manifest.copy()

    if "video" not in man.columns:
        raise ValueError("The manifest needs a 'video' column naming each recording.")
    for req in ("Genotype", "Treatment", "Sex"):
        if req not in man.columns:
            raise ValueError(f"The manifest needs a '{req}' column.")

    base_dir = Path(video_dir) if video_dir else None
    tables, qc_rows, excluded, failed = [], [], [], []

    for i, row in man.reset_index(drop=True).iterrows():
        name = str(row["video"]).strip()
        path = Path(name)
        if base_dir and not path.is_absolute():
            path = base_dir / name
        if progress:
            try:
                progress((i + 1) / max(len(man), 1), f"{name} ({i + 1}/{len(man)})")
            except Exception:
                pass

        if not path.exists():
            failed.append({"video": name, "reason": "file not found"})
            continue

        try:
            res = analyze_climbing_video(
                str(path), n_vials=n_vials, scoring_times_s=scoring_times_s,
                calibration=calibration, scoring_rule=scoring_rule, params=params,
                n_total_flies=_flies_from_row(row),
            )
        except Exception as exc:
            failed.append({"video": name, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue

        if not res.get("ok"):
            failed.append({"video": name, "reason": res.get("reason", "analysis failed")[:200]})
            continue

        conf = res["detection"]["confidence"] if res.get("detection") else float("nan")
        qc = res["qc"].copy()
        qc.insert(0, "video", name)
        qc["confidence"] = conf
        qc_rows.append(qc)

        if np.isfinite(conf) and conf < min_confidence:
            excluded.append({
                "video": name, "confidence": round(float(conf), 3),
                "reason": (res["detection"].get("reason") or
                           f"region confidence {conf:.2f} below the {min_confidence:.2f} "
                           "threshold"),
            })
            continue
        if not res["qc"].empty and bool(res["qc"]["implausible_count"].any()):
            bad = res["qc"][res["qc"]["implausible_count"]]["vial"].tolist()
            excluded.append({"video": name, "confidence": round(float(conf), 3),
                             "reason": f"implausible counts in vial(s) {bad}"})
            continue

        tab = to_drophenix_table(
            res,
            genotype=str(row["Genotype"]), treatment=str(row["Treatment"]),
            sex=str(row["Sex"]),
            experiment_id=(str(row["ExperimentID"]) if "ExperimentID" in man.columns
                           and pd.notna(row.get("ExperimentID")) else None),
            age_days=(float(row["Age"]) if "Age" in man.columns
                      and pd.notna(row.get("Age")) else None),
            time_of_day=(str(row["TimeOfDay"]) if "TimeOfDay" in man.columns
                         and pd.notna(row.get("TimeOfDay")) else None),
        )
        for extra in ("Batch", "Compound", "Dose", "DoseUnit"):
            if extra in man.columns and pd.notna(row.get(extra)) and not tab.empty:
                tab[extra] = row[extra]

        # VIAL-LEVEL quality control. Passing the video-level confidence gate is
        # not enough: region detection can be excellent for the rack as a whole
        # while one cylinder inside it is unscoreable -- a reflection, a vial
        # whose flies never left the base, a region that clipped the glass. On a
        # real 15-video batch, 13 of the 33 vials that reached the merged table
        # were graded "poor" or "unusable" by the very QC the analysis had
        # already computed, and none of it was carried forward.
        #
        # Unusable vials are dropped; poor ones are kept but LABELLED, so the
        # judgement travels with the data into the statistics instead of being
        # discarded at the door.
        if not tab.empty and not res["qc"].empty:
            grade = dict(zip(res["qc"]["vial"], res["qc"]["reliability"]))
            suffix = tab["VialID"].str.rsplit("_", n=1).str[-1]
            tab["DetectorReliability"] = suffix.map(grade).fillna("unknown")
            drop = tab["DetectorReliability"].str.startswith("unusable")
            if drop.any():
                for lab in sorted(set(suffix[drop])):
                    excluded.append({
                        "video": name, "vial": lab,
                        "confidence": round(float(conf), 3),
                        "reason": f"vial graded '{grade.get(lab, '?')}'",
                    })
                tab = tab[~drop]

        # A row whose criterion fell outside the measured region is missing, not
        # zero, and must not be exported as a count.
        if not tab.empty and "Count_climbed" in tab.columns:
            tab = tab[tab["Count_climbed"].notna()]

        if not tab.empty:
            tables.append(tab)

    merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    qc_all = pd.concat(qc_rows, ignore_index=True) if qc_rows else pd.DataFrame()

    notes: List[str] = []
    if not merged.empty:
        notes.append(
            f"{len(tables)} of {len(man)} recordings scored, giving "
            f"{merged['VialID'].nunique()} vials and {len(merged)} rows. Each cylinder is a "
            "separate VialID, so these are independent replicates for the clustered "
            "statistics."
        )
    if excluded:
        notes.append(f"{len(excluded)} recording(s) excluded on quality grounds -- see "
                     "`excluded`. They are not silently dropped and can be scored by hand.")
    if failed:
        notes.append(f"{len(failed)} recording(s) could not be processed -- see `failed`.")

    # A batch normally comes off one rig, so the cylinder count should not vary
    # between recordings. When it does, automatic detection has miscounted on
    # the odd ones out -- and a miscount is not a small error: an invented vial
    # contributes fabricated replicates to a clustered analysis, where extra
    # replicates directly inflate confidence.
    if not qc_all.empty and "video" in qc_all.columns:
        per_video = qc_all.groupby("video")["vial"].nunique()
        if per_video.nunique() > 1:
            modal = int(per_video.mode().iloc[0])
            odd = per_video[per_video != modal]
            notes.append(
                f"Cylinder count is not consistent across the batch: most recordings gave "
                f"{modal}, but {', '.join(f'{k} gave {v}' for k, v in odd.items())}. "
                "If every recording is of the same rig, set the number of cylinders "
                "explicitly rather than letting it be detected, and check the odd ones."
            )

    return {
        "table": merged,
        "qc": qc_all,
        "excluded": pd.DataFrame(excluded),
        "failed": pd.DataFrame(failed),
        "n_requested": int(len(man)),
        "n_scored": len(tables),
        "notes": notes,
    }


def _flies_from_row(row) -> Optional[Dict[str, int]]:
    """Read per-vial fly totals from a manifest row (n_per_vial, or n_<label>)."""
    out: Dict[str, int] = {}
    for key, val in row.items():
        k = str(key)
        if k.lower().startswith("n_") and k.lower() != "n_per_vial" and pd.notna(val):
            try:
                out[k[2:]] = int(val)
            except (TypeError, ValueError):
                continue
    if out:
        return out
    if "n_per_vial" in row.index and pd.notna(row.get("n_per_vial")):
        try:
            n = int(row["n_per_vial"])
            # applied to every label the detector finds
            return {"__all__": n}
        except (TypeError, ValueError):
            return None
    return None


def to_drophenix_table(
    result: dict,
    genotype: str,
    treatment: str,
    sex: str,
    experiment_id: Optional[str] = None,
    age_days: Optional[float] = None,
    time_of_day: Optional[str] = None,
    vial_id_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert a scored video into the canonical DroPhenix climbing table, so that
    it flows straight into the hierarchical, trajectory and ranking modules.

    Each cylinder becomes its own `VialID`, which matters: the vial is the
    experimental unit, and a video of three cylinders is three replicates, not
    one. Every scoring time becomes a row, giving the full climbing curve.
    """
    counts = result.get("counts")
    if counts is None or counts.empty:
        return pd.DataFrame()

    usable = counts[counts.get("usable", True) == True].copy()  # noqa: E712

    # A row whose denominator is zero cannot express a proportion, and exporting
    # it makes the whole table unloadable downstream (`data_parser` rejects
    # n <= 0, correctly). It arises when no total was supplied and the detector
    # happened to find nothing in that vial at that instant -- a momentary
    # detection failure, not an observation of an empty cylinder. Dropping the
    # row is right: the vial's other timepoints stay, and the gap is visible.
    if "n_total_used" in usable.columns:
        n_zero = int((usable["n_total_used"].fillna(0) <= 0).sum())
        usable = usable[usable["n_total_used"].fillna(0) > 0]
        if n_zero:
            usable.attrs["dropped_zero_denominator"] = n_zero

    # Likewise a row that could not be scored under the chosen rule is missing,
    # not zero.
    if "n_above_detected" in usable.columns:
        usable = usable[usable["n_above_detected"].notna()]

    if usable.empty:
        return pd.DataFrame()

    stem = vial_id_prefix or Path(result["info"].path).stem
    out = pd.DataFrame({
        "Genotype": genotype,
        "Treatment": treatment,
        "Sex": sex,
        "VialID": [f"{stem}_{v}" for v in usable["vial"]],
        "Replicate": pd.factorize(usable["vial"])[0] + 1,
        "Time": usable["time_s"].astype(float),
        "Count_climbed": usable["n_above_detected"],
        "n_total_flies": usable["n_total_used"].astype("Int64"),
    })
    if experiment_id:
        out["ExperimentID"] = experiment_id
    if age_days is not None:
        out["Age"] = float(age_days)
    if time_of_day:
        out["TimeOfDay"] = time_of_day
    out["SourceVideo"] = Path(result["info"].path).name
    return out.reset_index(drop=True)
