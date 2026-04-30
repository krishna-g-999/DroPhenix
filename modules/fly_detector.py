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
7. Area filter: min_area ≤ blob_area ≤ max_area  (fly size range ~20–800 px²)
8. Centroid (cx, cy) from image moments
9. Flies above threshold line: cy < threshold_frac × vial_height
10. PI = (above − below) / total × 100

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
import warnings

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
    expected_n : expected total fly count per vial (for PI denominator)
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
    expected_n: int = 10          # FIX: was missing — required by analyze_with_n()

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
        Automatically divide the frame into an evenly-spaced vial grid.
        margin_px removes inter-vial boundary pixels to reduce edge artefacts.
        """
        cfg  = VIAL_PRESETS.get(preset, {"n_vials": 1, "cols": 1, "rows": 1})
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


# ─── VialCount ────────────────────────────────────────────────────────────────
@dataclass
class VialCount:
    """
    Per-vial fly count result for one timepoint.

    Attributes
    ----------
    n_above   : flies above threshold line (climbed)
    n_below   : flies below threshold line (did not climb)
    total     : total detected flies
    pct_above : percentage above line (= n_above / total × 100)
    pi        : Performance Index = (n_above − n_below) / total × 100
    QC_flag   : "ok" | "low_count" | "high_count" | "no_background"
    Confidence: 0–1 detection confidence estimate
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
    pct_above:  float = 0.0
    pi:         float = 0.0         # FIX: was missing
    QC_flag:    str   = "ok"        # FIX: was missing
    Confidence: float = 1.0         # FIX: was missing

    def to_dict(self) -> dict:      # FIX: was missing
        return {
            "VialID":     self.vial_id,
            "Label":      self.label,
            "Genotype":   self.genotype,
            "Sex":        self.sex,
            "Treatment":  self.treatment,
            "n_above":    self.n_above,
            "n_below":    self.n_below,
            "Total":      self.total,
            "Pct":        round(self.pct_above, 2),
            "PI":         round(self.pi, 2),
            "QC_flag":    self.QC_flag,
            "Confidence": round(self.Confidence, 3),
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
    n_above, n_below, Total, Pct, PI, QC_flag, Confidence, Time, Replicate
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
    max_area       : maximum blob area in px² (removes merged blobs / debris)
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
    def detect(
        self,
        frame,
        bg_frame=None,
    ) -> Tuple[int, int, list]:
        """
        Detect flies in a single (pre-cropped) vial frame.

        Parameters
        ----------
        frame    : BGR vial frame (numpy array)
        bg_frame : optional static background for the same vial crop

        Returns
        -------
        (total, above, blobs)
        blobs = [(cx, cy, area), ...]
        """
        if not _CV2 or frame is None:
            return 0, 0, []

        cfg  = self.config
        h, w = frame.shape[:2]
        liney = int(h * cfg.threshold_frac)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background subtraction
        if bg_frame is not None:
            # Static: absolute difference from background
            bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY) if len(bg_frame.shape) == 3 else bg_frame
            bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
            fg = cv2.absdiff(blur, bg_blur)
            _, fg = cv2.threshold(fg, 25, 255, cv2.THRESH_BINARY)
        elif cfg.use_bg_subtraction:
            self._ensure_mog2()
            fg = self._mog2.apply(blur)
        else:
            _, fg = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

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

        total = above = 0
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (cfg.min_area <= area <= cfg.max_area):
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            total += 1
            if cy < liney:
                above += 1
            blobs.append((cx, cy, area))

        return total, above, blobs

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
            return [VialCount(vial_id=v.vial_id, label=v.label) for v in vials]

        results: List[VialCount] = []

        for v in vials:
            # Crop vial from frame
            sl_y, sl_x = v.as_slice()
            vial_crop = frame[sl_y, sl_x]
            if vial_crop.size == 0:
                results.append(VialCount(
                    vial_id=v.vial_id, label=v.label,
                    genotype=v.genotype, sex=v.sex, treatment=v.treatment,
                    expected_n=v.expected_n, QC_flag="empty_roi",
                ))
                continue

            # Crop bg_frame to same ROI if provided
            bg_crop = None
            if bg_frame is not None:
                try:
                    bg_crop = bg_frame[sl_y, sl_x]
                except Exception:
                    bg_crop = None

            total, above, blobs = self.detect(vial_crop, bg_crop)
            below  = total - above
            pct    = round(above / total * 100.0, 2) if total > 0 else 0.0
            pi     = round((above - below) / total * 100.0, 2) if total > 0 else 0.0

            # QC flags
            exp_n = v.expected_n if v.expected_n > 0 else None
            if total == 0:
                qc = "no_flies_detected"
                conf = 0.0
            elif exp_n and abs(total - exp_n) > exp_n * 0.5:
                qc   = "low_count" if total < exp_n * 0.5 else "high_count"
                conf = round(min(total, exp_n) / max(total, exp_n), 3)
            else:
                qc   = "ok"
                conf = min(1.0, round(total / max(exp_n, 1), 3)) if exp_n else 1.0

            results.append(VialCount(
                vial_id    = v.vial_id,
                label      = v.label,
                genotype   = v.genotype,
                sex        = v.sex,
                treatment  = v.treatment,
                expected_n = v.expected_n,
                n_above    = above,
                n_below    = below,
                total      = total,
                pct_above  = pct,
                pi         = pi,
                QC_flag    = qc,
                Confidence = conf,
            ))

        return results

    # ── Annotated frame (FIX: was missing entirely) ───────────────────────────
    def annotate_frame(
        self,
        frame,
        vials: List[VialROI],
        counts: List[VialCount],
        show_roi_boxes: bool = True,
    ):
        """
        Draw detection overlays on the full frame for QC visualisation.

        Draws per-vial:
        - Coloured bounding box (green=ok, yellow=low, red=no detection)
        - Threshold line (cyan dashed)
        - Count label: "↑N_above  ↓N_below  PI:XX"

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

            # Box colour by QC
            if vc.QC_flag == "ok":
                box_col = (0, 220, 60)
            elif vc.QC_flag == "no_flies_detected":
                box_col = (0, 0, 220)
            else:
                box_col = (0, 165, 255)

            if show_roi_boxes:
                cv2.rectangle(out, (v.x, v.y), (v.x + v.w, v.y + v.h), box_col, 2)

            # Threshold line
            liney = v.y + int(v.h * self.config.threshold_frac)
            cv2.line(out, (v.x, liney), (v.x + v.w, liney), (0, 255, 255), 1)

            # Re-detect blobs for this vial to draw centroids
            sl_y, sl_x = v.as_slice()
            vial_crop = out[sl_y, sl_x]
            if vial_crop.size > 0:
                _, _, blobs = self.detect(frame[sl_y, sl_x])
                for (cx, cy, _) in blobs:
                    col = (0, 255, 0) if cy < int(v.h * self.config.threshold_frac) else (0, 0, 255)
                    cv2.circle(out, (v.x + cx, v.y + cy), 4, col, -1)

            # Text overlay
            lbl = v.label if v.label else f"V{v.vial_id}"
            txt = f"{lbl} | up:{vc.n_above} dn:{vc.n_below} PI:{vc.pi:.0f}"
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
