"""
Core climbing-assay video analysis module (ROI-based, user-defined geometry).

Implements the specification in `CLAUDE_ClimbingVideoAnalyzer.md`.

ALGORITHM
---------
1. Extract the frame at a specified time (seconds since tap-down)
2. Crop to the vial ROI
3. Convert to greyscale
4. Gaussian blur to reduce sensor noise
5. Adaptive thresholding (dark flies on light, or light flies on dark)
6. Morphological open (remove noise) then close (fill gaps in fly bodies)
7. Find contours, filter by area (fly size expressed as a fraction of ROI area)
8. Count contours whose centroid lies ABOVE the reference line
9. Report count, PI, and a confidence derived from detected vs expected total

WHY ROI-BASED RATHER THAN AUTOMATIC
-----------------------------------
`modules/climbing_video.py` locates cylinders automatically from a temporal-median
background. That is the right design when it works, but on 15 laboratory recordings
only 8 of 45 vials were graded "good", and on a second rig automatic detection put
the regions on the frame edges. This module takes the opposite trade: the user
states the geometry, and the software does not guess. It is the fallback for
footage the automatic path cannot handle, and the two are complementary rather
than competing.

RELATION TO THE THRESHOLDING FAILURE DOCUMENTED IN `climbing_video.py`
---------------------------------------------------------------------
That module's docstring records that a GLOBAL intensity threshold cannot separate
a fly from the printed graduations on a cylinder, because both are dark. Adaptive
thresholding compares each pixel with its local neighbourhood instead, which
removes a broad lighting gradient but does NOT remove a locally dark feature: a
printed tick or numeral is still a dark blob of roughly fly size. On graduated
glassware this module should therefore be expected to OVER-detect, and the
two-sided discrepancy check below exists to make that visible rather than silent.
Whether it over-detects in practice is a measurement, not an assumption -- see
`validation/validate_climbing_video_analyzer.py`.

DEVIATIONS FROM THE SPECIFICATION, AND WHY
------------------------------------------
Each is a correctness fix; none changes the documented outputs.

1. **ROI is clipped to the frame, and the reference line follows the clip.**
   The spec crops with `frame[y:y+h, x:x+w]`, which silently returns a SMALLER
   array when the ROI runs past the frame edge, while `line_y_pixels` is still
   computed from the requested `roi_h`. The line would then sit at the wrong
   height, or outside the crop entirely, and every count would be wrong with no
   error raised. The ROI is now clipped explicitly and the line is placed as a
   fraction of the ACTUAL cropped height.

2. **A two-sided discrepancy flag.** The spec's critical rule 1 is "flag if > 30%
   discrepancy", but its `confidence = min(detected/expected, 1.0)` is capped, so
   detecting 40 flies where 20 are expected reports confidence 1.00 -- "good".
   Over-detection is precisely the failure mode adaptive thresholding is prone to
   here, so it must not be invisible. `Confidence` is kept exactly as specified
   for compatibility; `count_discrepancy` and `flagged` implement the rule.

3. **`run_full_analysis` is a generator**, as the spec's own app code requires
   (`for result, progress in run_full_analysis(session)`). Its docstring said it
   returned a DataFrame. The docstring is corrected; the behaviour is unchanged.
   `session.results` is also populated on exhaustion.

4. **Guards** on zero/absent fps, unreadable frames, and empty ROIs, so a failure
   yields `confidence = 0` and a reason rather than an exception (spec rule 7).

5. **An aspect-ratio filter (`max_aspect`, default 6).** Measured, not assumed: on
   the specification's own test video the detector returns **21** flies where 20
   were drawn. The extra detection sits at row 200 — it is the reference line
   itself, 200 px wide and 5 px tall, area 796 px, which falls inside the area
   window and so passes an area-only filter. This is not an artefact of the test:
   in a real assay the scoring line is drawn on the glass, so an area-only filter
   counts the criterion as an animal. Rejecting blobs longer than 6x their width
   removes it (21 -> 20) and cannot remove a fly, which is compact. The same guard
   exists in `climbing_video.py` for the same reason.

   Note that the specification's own unit test cannot catch this: it asserts
   `len(centroids) >= 10`, so a false positive passes silently. The validation
   suite added alongside this module checks the count in BOTH directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

__all__ = [
    "VialConfig", "FrameResult", "VideoAnalysisSession",
    "load_video_metadata", "get_frame_at_time", "get_frame_at_number",
    "detect_flies_in_roi", "analyze_frame_for_vial", "run_full_analysis",
    "try_detect_reference_line", "results_to_drophenix_csv",
    "DEFAULT_DETECTION_PARAMS", "DISCREPANCY_LIMIT",
]

#: Critical rule 1 -- flag when detected and expected differ by more than this.
DISCREPANCY_LIMIT = 0.30

DEFAULT_DETECTION_PARAMS: dict = {
    "blur_kernel": 5,              # Gaussian blur kernel size (odd)
    "adaptive_block": 21,          # Adaptive threshold block size (odd, > 3)
    "adaptive_c": 4,               # Adaptive threshold constant
    # Fly size is expressed against the VIAL WIDTH, not the ROI area. A fly spans
    # roughly a tenth of a cylinder's width, so its area is about 0.01*w^2; these
    # limits bracket that. The defaults reproduce the previous ROI-area behaviour
    # for a typical box (height about 5.4x width), so nothing silently shifts.
    "min_area_w2": 0.0016,         # Min fly area as a fraction of vial width squared
    "max_area_w2": 0.081,          # Max fly area as a fraction of vial width squared
    "size_basis": "width",         # "width" (correct) or "roi_area" (legacy)
    "min_area_frac": 0.0003,       # legacy: fraction of ROI AREA -- see size_basis
    "max_area_frac": 0.015,        # legacy: fraction of ROI AREA
    "morph_open": 2,               # Morphological open iterations
    "morph_close": 2,              # Morphological close iterations
    "confidence_threshold": 0.70,  # Below this -> flagged as low confidence
    "max_aspect": 6.0,             # Reject blobs longer than this x their width
    "remove_horizontal_lines": True,  # Strip the drawn scoring line before counting
    # A drawn line is antialiased, and lossy video smears it further, so the rows
    # either side of the line's solid core are part of the same mark. Rows adjacent
    # to the core whose foreground fraction reaches this are absorbed into it. A row
    # carrying flies rather than a line sits well below: twenty flies spread down a
    # tube put at most two or three in any one row, about 0.10-0.15 of the width.
    "line_edge_frac": 0.20,
    # ...but never grow the band by more than this many rows either side, so a dense
    # raft of flies resting AT the line cannot be eaten by the line removal.
    "line_edge_max_px": 4,
    # Area floor for a blob truncated by the removed line band, as a fraction of the
    # ordinary floor. Set to 0 to judge truncated blobs like any other.
    "line_touch_area_frac": 0.5,
}


# ─────────────────────────────────────────────────────────────── data model ──
@dataclass
class VialConfig:
    """Configuration for one vial in the video."""

    vial_id: int
    genotype: str
    replicate: int
    n_flies_expected: int
    roi_x: int              # ROI top-left X in the full frame
    roi_y: int              # ROI top-left Y in the full frame
    roi_w: int              # ROI width in pixels
    roi_h: int              # ROI height in pixels
    line_y_frac: float      # Reference line Y as a fraction of ROI height
    fly_color: str = "dark"  # 'dark' or 'light' flies
    sex: str = "Mixed"
    treatment: str = "vehicle"
    notes: str = ""

    @property
    def line_y_pixels(self) -> int:
        """Reference line Y in ROI pixel space, for the REQUESTED ROI height.

        When the ROI is clipped by the frame edge the effective line is computed
        from the clipped height instead -- see `line_y_for_height`.
        """
        return int(self.line_y_frac * self.roi_h)

    def line_y_for_height(self, height: int) -> int:
        """Reference line row for an ROI that is actually `height` pixels tall."""
        if height <= 0:
            return 0
        return int(np.clip(round(self.line_y_frac * height), 0, height - 1))

    def clipped_roi(self, frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) of this ROI clipped to the frame."""
        x0 = int(np.clip(self.roi_x, 0, max(frame_w - 1, 0)))
        y0 = int(np.clip(self.roi_y, 0, max(frame_h - 1, 0)))
        x1 = int(np.clip(self.roi_x + self.roi_w, x0, frame_w))
        y1 = int(np.clip(self.roi_y + self.roi_h, y0, frame_h))
        return x0, y0, x1, y1


@dataclass
class FrameResult:
    """Result of analysis for one vial at one timepoint."""

    vial_id: int
    time_sec: float
    frame_number: int
    flies_above: int
    flies_below: int
    flies_detected: int
    n_flies_expected: int
    pi: float               # Performance Index = flies_above / n_flies_expected
    confidence: float       # flies_detected / n_flies_expected, capped at 1.0
    low_confidence: bool    # True if confidence < confidence_threshold
    count_discrepancy: float = 0.0   # (detected - expected) / expected, SIGNED
    flagged: bool = False            # |discrepancy| > DISCREPANCY_LIMIT
    reason: str = ""                 # why a frame failed, when it did
    # WHICH TAP THIS MEASUREMENT BELONGS TO, 1-based.
    # The assay is normally run as three to five taps of the same vial with a
    # minute's recovery between them; each tap is a technical replicate of the
    # SAME animals, and its own T=0. Without this field a five-tap recording
    # collapses into one trial and the other four are silently discarded -- or,
    # worse, are read as one long observation after a single tap. Defaults to 1
    # so a single-tap recording behaves exactly as before.
    tap: int = 1


@dataclass
class VideoAnalysisSession:
    """Holds all configuration and results for one video analysis session."""

    video_path: str
    fps: float = 0.0
    total_frames: int = 0
    duration_sec: float = 0.0
    t_zero_frame: int = 0
    #: One tap-down frame per tap, in order. EMPTY means "one tap, at
    #: `t_zero_frame`" -- the single-tap behaviour this module has always had.
    #: When populated, `t_zero_frame` is ignored and every tap is scored.
    tap_zero_frames: List[int] = field(default_factory=list)
    timepoints_sec: List[float] = field(default_factory=lambda: [10, 20, 30, 60])
    vials: List[VialConfig] = field(default_factory=list)
    results: List[FrameResult] = field(default_factory=list)
    detection_params: dict = field(
        default_factory=lambda: dict(DEFAULT_DETECTION_PARAMS))


# ──────────────────────────────────────────────────────────────── video I/O ──
def load_video_metadata(video_path: str) -> dict:
    """Read fps, frame count, duration and frame size.

    Raises RuntimeError if the file cannot be opened, rather than returning
    zeros that would silently corrupt every later frame index.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": (total_frames / fps) if fps > 0 else 0.0,
        "width": width,
        "height": height,
    }


def get_frame_at_number(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Read one frame by index. Returns a BGR array, or None if unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_number)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def get_frame_at_time(video_path: str, time_sec: float,
                      t_zero_frame: int = 0, fps: float = 30.0) -> Optional[np.ndarray]:
    """Read the frame `time_sec` seconds after `t_zero_frame`."""
    if not fps or fps <= 0:
        return None
    return get_frame_at_number(video_path, int(t_zero_frame) + int(time_sec * fps))


# ──────────────────────────────────────────────────────────────── detection ──
def detect_flies_in_roi(frame: np.ndarray, vial: VialConfig,
                        params: Optional[dict] = None
                        ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Detect flies inside one vial's ROI.

    Returns
    -------
    (centroids, debug_image)
        `centroids` are (x, y) in ROI pixel space; `debug_image` is the ROI in
        BGR with the reference line and detections drawn on it.
    """
    p = dict(DEFAULT_DETECTION_PARAMS)
    if params:
        p.update(params)

    if frame is None or getattr(frame, "size", 0) == 0:
        return [], np.zeros((max(vial.roi_h, 1), max(vial.roi_w, 1), 3), np.uint8)

    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = vial.clipped_roi(fh, fw)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return [], np.zeros((max(vial.roi_h, 1), max(vial.roi_w, 1), 3), np.uint8)

    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    blur_k = int(p["blur_kernel"])
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    block = int(p["adaptive_block"])
    if block % 2 == 0 or block <= 3:
        block = 21
    block = min(block, (min(h, w) // 2) * 2 + 1)     # must not exceed the ROI
    block = max(block, 5)
    c_val = float(p["adaptive_c"])

    mode = cv2.THRESH_BINARY_INV if vial.fly_color == "dark" else cv2.THRESH_BINARY
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   mode, block, c_val)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                               iterations=int(p["morph_open"]))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel,
                               iterations=int(p["morph_close"]))

    # Remove long HORIZONTAL structures -- the drawn scoring line, a shelf edge, a
    # tube rim -- before contour finding. Opening with a wide, one-pixel-tall
    # kernel keeps only runs longer than the kernel, which is what a line is and
    # what a fly is not. Subtracting them preserves the body of a fly that happens
    # to be sitting ON the line, which rejecting the merged blob by aspect ratio
    # would discard along with the line: measured on synthetic footage, aspect
    # rejection alone lost 4 of 20 flies at the moment the cohort crossed.
    line_len = max(15, int(0.5 * w))
    removed_band: Optional[Tuple[int, int]] = None
    if p.get("remove_horizontal_lines", True) and line_len < w:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
        long_h = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, h_kernel)
        if int(long_h.max() if long_h.size else 0) > 0:
            # Zero the WHOLE band the line occupies, not just the pixels the
            # opening recovered. A drawn line is antialiased, so its faint upper
            # and lower edges survive a subtraction and reappear as fragments --
            # measured, they gave an EMPTY tube 3 detections.
            core = np.flatnonzero(long_h.max(axis=1) > 0)
            r_lo, r_hi = int(core.min()), int(core.max())
            # Grow outwards over the antialiased shoulder. Leaving those rows behind
            # is what made this defective: they survived the zeroing, and the
            # vertical repair below then inflated them into fly-sized blobs. With
            # flies sized as the module documents (about a tenth of vial width), the
            # drawn line ADDED 1-3 phantom detections at the moment the cohort
            # crossed it -- the worst possible place for a phantom, since it lands
            # on the wrong side of the scoring decision. Without the line the same
            # frames counted exactly 20 of 20.
            frac = (cleaned > 0).mean(axis=1)
            edge = float(p.get("line_edge_frac", 0.20))
            grow = int(p.get("line_edge_max_px", 4))
            lo_stop, hi_stop = max(0, r_lo - grow), min(h - 1, r_hi + grow)
            while r_lo > lo_stop and frac[r_lo - 1] >= edge:
                r_lo -= 1
            while r_hi < hi_stop and frac[r_hi + 1] >= edge:
                r_hi += 1

            # Subtracting the line cuts any fly SITTING on it into two pieces, each
            # below the minimum area, so both halves vanish and the animal is lost
            # exactly where the scoring decision is tightest. A vertical close
            # rejoins them -- but applied to the WHOLE ROI it also welds unrelated
            # noise together: measured, that turned 20 flies into 22 and gave an
            # empty tube 3 detections. So the repair is confined to a band around
            # the removed line AND to the columns where something actually sits on
            # both sides of it, which is what a fly cut in half looks like and what
            # a leftover smear of line does not.
            pad = max(3, (r_hi - r_lo) + 4)
            t0, t1 = max(0, r_lo - pad), min(h, r_hi + 1 + pad)
            top = cleaned[t0:r_lo]
            bot = cleaned[r_hi + 1:t1]
            zeros = np.zeros(w, np.uint8)
            cut = ((top.max(axis=0) if top.size else zeros) > 0) & \
                  ((bot.max(axis=0) if bot.size else zeros) > 0)
            cleaned[r_lo:r_hi + 1, :] = 0
            removed_band = (r_lo, r_hi)
            if cut.any() and t1 > t0:
                vk = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (1, max(3, (r_hi - r_lo) + 5)))
                band = cleaned[t0:t1]
                repaired = cv2.morphologyEx(band, cv2.MORPH_CLOSE, vk, iterations=1)
                cols = np.flatnonzero(cut)
                band[:, cols] = repaired[:, cols]

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # THE SIZE WINDOW MUST NOT DEPEND ON HOW BIG A BOX THE USER DREW.
    # It used to: `min_area_frac * (w * h)` grows with the box height, so a taller
    # rectangle raised the minimum blob size and silently discarded small flies.
    # Measured on one cylinder at a fixed instant, extending the box upward moved
    # the floor from 34 to 44 px^2 and the count from 29 detections (18 above the
    # line) to 22 (14 above) -- against a hand count of 19. For a tool whose whole
    # premise is "draw the box", that is disqualifying.
    #
    # A fly's size is set by the CYLINDER, which is about ten fly-widths across, so
    # the window is anchored to w^2 and is unaffected by where the box ends.
    if str(p.get("size_basis", "width")) == "roi_area":
        roi_area = float(w * h)
        min_area = float(p["min_area_frac"]) * roi_area
        max_area = float(p["max_area_frac"]) * roi_area
    else:
        w2 = float(w * w)
        min_area = float(p.get("min_area_w2", 0.0016)) * w2
        max_area = float(p.get("max_area_w2", 0.081)) * w2

    max_aspect = float(p.get("max_aspect", 6.0) or 0.0)
    # A blob lying against the band the scoring line was removed from has been cut
    # by that removal, so judging it against a whole fly's area is the wrong test:
    # the animal is condemned for the part of itself the line was covering, exactly
    # where the scoring decision is tightest. Its floor is reduced accordingly, but
    # only for blobs actually touching the band -- everything else is unaffected.
    touch_frac = float(p.get("line_touch_area_frac", 0.5) or 0.0)
    centroids: List[Tuple[int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        floor = min_area
        if removed_band is not None and touch_frac > 0:
            _, by, _, bh_ = cv2.boundingRect(contour)
            if by <= removed_band[1] + 1 and by + bh_ >= removed_band[0] - 1:
                floor = min_area * touch_frac
        if not (floor <= area <= max_area):
            continue
        if max_aspect > 0:
            _, _, bw, bh = cv2.boundingRect(contour)
            if max(bw, bh) / max(min(bw, bh), 1) > max_aspect:
                continue          # a drawn line, a tube edge, a scratch -- not a fly
        m = cv2.moments(contour)
        if m["m00"] > 0:
            centroids.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))

    # ── debug overlay ────────────────────────────────────────────────────────
    debug = roi.copy() if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    line_y = vial.line_y_for_height(h)
    cv2.line(debug, (0, line_y), (w - 1, line_y), (0, 255, 0), 2)
    for (cx, cy) in centroids:
        colour = (0, 0, 255) if cy < line_y else (255, 0, 0)   # red above, blue below
        cv2.circle(debug, (cx, cy), 5, colour, -1)
    above = sum(1 for (_, cy) in centroids if cy < line_y)
    below = len(centroids) - above
    cv2.putText(debug, f"Above: {above}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(debug, f"Below: {below}", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return centroids, debug


def analyze_frame_for_vial(video_path: str, vial: VialConfig, time_sec: float,
                           t_zero_frame: int, fps: float,
                           params: Optional[dict] = None) -> FrameResult:
    """Analyse one vial at one timepoint."""
    p = dict(DEFAULT_DETECTION_PARAMS)
    if params:
        p.update(params)
    n_expected = max(int(vial.n_flies_expected), 0)

    def _failed(frame_no: int, why: str) -> FrameResult:
        # Spec rule 7: a failed frame is confidence 0 with a reason, never skipped.
        return FrameResult(
            vial_id=vial.vial_id, time_sec=float(time_sec), frame_number=int(frame_no),
            flies_above=0, flies_below=0, flies_detected=0,
            n_flies_expected=n_expected, pi=0.0, confidence=0.0,
            low_confidence=True, count_discrepancy=-1.0, flagged=True, reason=why)

    if not fps or fps <= 0:
        return _failed(0, "video reports no frame rate")

    target_frame = int(t_zero_frame) + int(float(time_sec) * fps)
    frame = get_frame_at_number(video_path, target_frame)
    if frame is None:
        return _failed(target_frame, "frame could not be read")

    centroids, _ = detect_flies_in_roi(frame, vial, p)
    fh, fw = frame.shape[:2]
    _, y0, _, y1 = vial.clipped_roi(fh, fw)
    line_y = vial.line_y_for_height(y1 - y0)

    above = sum(1 for (_, cy) in centroids if cy < line_y)
    total_detected = len(centroids)
    below = total_detected - above

    denom = max(n_expected, 1)
    pi = above / denom
    confidence = min(total_detected / denom, 1.0)
    discrepancy = (total_detected - n_expected) / denom

    return FrameResult(
        vial_id=vial.vial_id,
        time_sec=float(time_sec),
        frame_number=target_frame,
        flies_above=int(above),
        flies_below=int(below),
        flies_detected=int(total_detected),
        n_flies_expected=n_expected,
        pi=round(pi, 4),
        confidence=round(confidence, 3),
        low_confidence=bool(confidence < float(p["confidence_threshold"])),
        count_discrepancy=round(float(discrepancy), 3),
        flagged=bool(abs(discrepancy) > DISCREPANCY_LIMIT),
        reason="",
    )


def run_full_analysis(session: VideoAnalysisSession) -> Iterator[Tuple[FrameResult, float]]:
    """Analyse every vial at every timepoint.

    This is a GENERATOR: it yields `(FrameResult, progress)` where progress runs
    from 0 to 1, so a Streamlit progress bar can advance during a long run.
    `session.results` is filled once the generator is exhausted.
    """
    all_results: List[FrameResult] = []
    # one T=0 per tap; an empty list is the single-tap case and keeps the old path
    zeros = list(session.tap_zero_frames) or [int(session.t_zero_frame)]
    total_jobs = max(len(zeros) * len(session.vials)
                     * len(session.timepoints_sec), 1)
    job_num = 0

    for tap_i, t_zero in enumerate(zeros, 1):
        for vial in session.vials:
            for t in session.timepoints_sec:
                result = analyze_frame_for_vial(
                    session.video_path, vial, float(t),
                    int(t_zero), session.fps, session.detection_params)
                result.tap = tap_i
                all_results.append(result)
                job_num += 1
                yield result, job_num / total_jobs

    session.results = all_results


# ─────────────────────────────────────────────────── reference-line finding ──
def try_detect_reference_line(frame: np.ndarray, roi_x: int, roi_y: int,
                              roi_w: int, roi_h: int) -> Optional[float]:
    """Look for a horizontal reference line drawn on the vial.

    Method: vertical Sobel gradient, summed across each row, smoothed; the
    strongest row in the middle half of the ROI is accepted only when it is at
    least three times the mean row strength. Returns the line as a fraction of
    ROI height, or None when no line stands out -- refusing is the point, since a
    guessed line silently relocates the scoring criterion.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    fh, fw = frame.shape[:2]
    x0 = int(np.clip(roi_x, 0, max(fw - 1, 0)))
    y0 = int(np.clip(roi_y, 0, max(fh - 1, 0)))
    x1 = int(np.clip(roi_x + roi_w, x0, fw))
    y1 = int(np.clip(roi_y + roi_h, y0, fh))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < 8:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h = gray.shape[0]

    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    row_strength = sobel_y.sum(axis=1)

    k = max(3, h // 20)
    smoothed = np.convolve(row_strength, np.ones(k) / k, mode="same")

    start, end = h // 4, (3 * h) // 4
    region = smoothed[start:end]
    if region.size == 0:
        return None

    peak_y = start + int(np.argmax(region))
    mean_strength = float(smoothed.mean())
    if mean_strength <= 0:
        return None
    if float(smoothed[peak_y]) > mean_strength * 3.0:
        return round(peak_y / h, 3)
    return None


# ───────────────────────────────────────────────────────────────── exporting ──
def results_to_drophenix_csv(session: VideoAnalysisSession) -> pd.DataFrame:
    """Convert results to the DroPhenix climbing-assay table.

    Column names were checked against `modules/data_parser.py`: `Time_sec`,
    `Count_climbed` and `n_total_flies` are all recognised aliases of `Time`,
    `Count` and `n`, and `Vial_ID` is an alias of `VialID`, so this table loads
    through `load_data` without renaming.
    """
    if not session.results:
        return pd.DataFrame()

    vial_map = {v.vial_id: v for v in session.vials}
    rows = []
    for r in session.results:
        vial = vial_map.get(r.vial_id)
        if vial is None:
            continue
        rows.append({
            "Genotype": vial.genotype,
            "Treatment": vial.treatment,
            "Sex": vial.sex,
            "Replicate": vial.replicate,
            "Vial_ID": r.vial_id,
            # the technical replicate: same vial, same flies, a later tap
            "Tap": r.tap,
            "Time_sec": r.time_sec,
            "Count_climbed": r.flies_above,
            "Count_below": r.flies_below,
            "Detected_total": r.flies_detected,
            "n_total_flies": r.n_flies_expected,
            "PI": r.pi,
            "Confidence": r.confidence,
            "Low_confidence": r.low_confidence,
            "Count_discrepancy": r.count_discrepancy,
            "Flagged": r.flagged,
            "Frame_number": r.frame_number,
            "Video_file": Path(session.video_path).name,
            "Analysis_notes": (vial.notes + (" | " + r.reason if r.reason else "")).strip(" |"),
        })
    return pd.DataFrame(rows)
