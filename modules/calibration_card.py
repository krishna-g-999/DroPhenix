"""
A printable calibration card: put a known length in the shot, get millimetres out.

WHY
---
Every distance the software reports is in pixels until something in the frame
has a known real size. Two ways to supply that:

  Tell it the cylinder diameter.  Works, but relies on somebody remembering the
  number, entering it correctly, and the tubes being what they are assumed to be.
  It also gives one number and no way to check it.

  Put a printed card in the shot.  The scale is measured from the image itself,
  every time, with no one to remember anything -- and because the card carries
  FOUR markers at known separations, the scale can be computed several ways and
  the answers compared. Agreement is evidence the calibration is right; a
  disagreement means the card is tilted relative to the camera and says so.

The card also gives something the tube diameter cannot: a fixed reference that
does not move when the flies do, so camera drift can be measured directly rather
than inferred.

THE TWO WAYS THIS GOES WRONG, BOTH HANDLED
------------------------------------------
1. PRINTED AT THE WRONG SCALE. Printer dialogues default to "fit to page" and
   silently shrink by a few percent; every distance is then wrong by that much,
   consistently and invisibly. The card therefore carries a 100 mm rule to be
   checked with a ruler before use, and `detect_card` reports the implied
   marker size so a mis-scaled print shows up as a number that disagrees with
   the specification.

2. NOT IN THE PLANE OF THE CYLINDERS. Scale falls off with distance from the
   camera. A card propped behind the tubes reads a smaller millimetre-per-pixel
   than the tubes themselves, and every height comes out too large. The card
   must sit in the SAME plane as the front face of the cylinders -- beside them,
   not behind them -- and this is printed on the card.

Markers are ArUco (DICT_4X4_50), which OpenCV detects natively; nothing needs
installing beyond what the analysis already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as _e:  # pragma: no cover
    raise ImportError("OpenCV is required for calibration_card.") from _e

# ── card specification (millimetres) ────────────────────────────────────────
MARKER_MM = 40.0          # side length of each ArUco square
SPAN_X_MM = 120.0         # horizontal distance between marker CENTRES
SPAN_Y_MM = 130.0         # vertical distance between marker CENTRES
RULE_MM = 100.0           # the printed check rule
MARKER_IDS = (0, 1, 2, 3)  # top-left, top-right, bottom-left, bottom-right
_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class CardDetection:
    found: bool
    n_markers: int
    px_per_mm: float
    px_per_mm_estimates: Dict[str, float]
    disagreement_pct: float
    implied_marker_mm: float
    centres: Dict[int, Tuple[float, float]]
    warnings: List[str]

    def summary(self) -> str:
        if not self.found:
            return "No calibration card found in this frame."
        return (f"{self.n_markers}/4 markers, {self.px_per_mm:.3f} px/mm "
                f"(estimates disagree by {self.disagreement_pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# 1. MAKE THE CARD
# ═══════════════════════════════════════════════════════════════════════════
def make_card(out_path: str, page: str = "A4", dpi: int = 300) -> str:
    """
    Render the printable card as PDF (and PNG alongside).

    Print at 100% scale — "actual size", NOT "fit to page". The card carries a
    100 mm rule precisely so that this can be verified with a ruler in five
    seconds, before it costs an experiment.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    W, H = (210.0, 297.0) if page.upper() == "A4" else (215.9, 279.4)
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.invert_yaxis()

    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT)
    cx0, cy0 = W / 2 - SPAN_X_MM / 2, 78.0   # top row of markers
    centres = {0: (cx0, cy0), 1: (cx0 + SPAN_X_MM, cy0),
               2: (cx0, cy0 + SPAN_Y_MM), 3: (cx0 + SPAN_X_MM, cy0 + SPAN_Y_MM)}

    for mid, (mx, my) in centres.items():
        img = cv2.aruco.generateImageMarker(aruco_dict, mid, 400)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, zorder=3,
                  extent=(mx - MARKER_MM / 2, mx + MARKER_MM / 2,
                          my + MARKER_MM / 2, my - MARKER_MM / 2))
        ax.text(mx, my + MARKER_MM / 2 + 5, f"id {mid}", ha="center", va="top",
                fontsize=6, color="0.45")

    # separations, printed so a human can check them too
    ax.annotate("", xy=(centres[1][0], cy0 - MARKER_MM / 2 - 12),
                xytext=(centres[0][0], cy0 - MARKER_MM / 2 - 12),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25"))
    ax.text(W / 2, cy0 - MARKER_MM / 2 - 15, f"{SPAN_X_MM:.0f}.0 mm centre to centre",
            ha="center", va="bottom", fontsize=8, color="0.25")
    ax.annotate("", xy=(cx0 - MARKER_MM / 2 - 12, centres[2][1]),
                xytext=(cx0 - MARKER_MM / 2 - 12, cy0),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="0.25"))
    ax.text(cx0 - MARKER_MM / 2 - 15, (cy0 + centres[2][1]) / 2,
            f"{SPAN_Y_MM:.0f}.0 mm", ha="right", va="center", fontsize=8,
            color="0.25", rotation=90)

    # print-scale check rule
    ry = H - 52.0
    rx0 = W / 2 - RULE_MM / 2
    ax.add_patch(Rectangle((rx0, ry), RULE_MM, 3.2, fc="black"))
    for i in range(11):
        x = rx0 + i * RULE_MM / 10
        ax.plot([x, x], [ry - 2.6, ry], color="black", lw=1.0)
        if i % 5 == 0:
            ax.text(x, ry - 3.4, f"{i * 10}", ha="center", va="bottom", fontsize=6)
    ax.text(W / 2, ry + 8.5,
            "CHECK BEFORE USE: this bar must measure exactly 100.0 mm.\n"
            "If it does not, the page was scaled — reprint at 100% / \"actual size\".",
            ha="center", va="top", fontsize=8.5, color="#D55E00", weight="bold")

    ax.text(W / 2, 18, "DroPhenix calibration card", ha="center", va="center",
            fontsize=15, weight="bold")
    ax.text(W / 2, 32,
            "Place this card BESIDE the cylinders, upright, in the SAME PLANE as the front of the "
            "tubes — not behind them.\nScale falls off with distance from the camera, so a card set "
            "back reads too small and every height comes out too large.\n"
            "Keep all four markers fully in shot and unobstructed.",
            ha="center", va="center", fontsize=8, color="0.3")
    ax.text(W / 2, H - 12,
            f"ArUco DICT_4X4_50, ids 0-3 · marker side {MARKER_MM:.0f}.0 mm · "
            f"centres {SPAN_X_MM:.0f}.0 x {SPAN_Y_MM:.0f}.0 mm",
            ha="center", va="center", fontsize=7, color="0.5")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), format="pdf", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=dpi, facecolor="white")
    plt.close(fig)
    return str(out.with_suffix(".pdf"))


# ═══════════════════════════════════════════════════════════════════════════
# 2. FIND IT IN A FRAME
# ═══════════════════════════════════════════════════════════════════════════
def detect_card(frame: np.ndarray) -> CardDetection:
    """
    Locate the card in one frame and derive pixels per millimetre.

    The scale is computed independently from the horizontal separation, the
    vertical separation, and each marker's own side length. Those are different
    measurements of the same quantity, so their spread is a direct check: they
    agree when the card is flat and square to the camera, and diverge when it is
    tilted. A single number with no way to check it would hide exactly that.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return CardDetection(False, 0, float("nan"), {}, float("nan"),
                             float("nan"), {}, ["No ArUco markers detected."])

    ids = ids.ravel().tolist()
    centres: Dict[int, Tuple[float, float]] = {}
    sides: List[float] = []
    for c, i in zip(corners, ids):
        if i not in MARKER_IDS:
            continue
        pts = c.reshape(4, 2)
        centres[int(i)] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        sides += [float(np.linalg.norm(pts[k] - pts[(k + 1) % 4])) for k in range(4)]

    warnings: List[str] = []
    est: Dict[str, float] = {}

    def sep(a, b):
        if a in centres and b in centres:
            return float(np.hypot(centres[a][0] - centres[b][0],
                                  centres[a][1] - centres[b][1]))
        return None

    for label, (a, b), mm in (("top edge", (0, 1), SPAN_X_MM),
                              ("bottom edge", (2, 3), SPAN_X_MM),
                              ("left edge", (0, 2), SPAN_Y_MM),
                              ("right edge", (1, 3), SPAN_Y_MM)):
        d = sep(a, b)
        if d:
            est[label] = d / mm
    if sides:
        est["marker side"] = float(np.median(sides)) / MARKER_MM

    if not est:
        return CardDetection(False, len(centres), float("nan"), {}, float("nan"),
                             float("nan"), centres,
                             ["Markers found but none of the expected ids 0-3."])

    vals = np.array(list(est.values()), dtype=float)
    ppm = float(np.median(vals))
    spread = float((vals.max() - vals.min()) / max(ppm, 1e-9) * 100.0)

    if len(centres) < 4:
        warnings.append(
            f"Only {len(centres)} of 4 markers visible. The scale is still usable but cannot "
            "be cross-checked; make sure the whole card is in shot and unobstructed.")
    if spread > 5.0:
        warnings.append(
            f"The independent scale estimates disagree by {spread:.1f}%. That usually means "
            "the card is tilted relative to the camera, or is not in the same plane as the "
            "cylinders. Stand it upright, beside the tubes, facing the lens.")

    implied = float(np.median(sides)) / ppm if sides else float("nan")
    if np.isfinite(implied) and abs(implied - MARKER_MM) / MARKER_MM > 0.03:
        warnings.append(
            f"The markers measure {implied:.1f} mm on the page, but the card specifies "
            f"{MARKER_MM:.0f} mm. The page was probably printed scaled; reprint at 100% "
            "and check the 100 mm rule with a ruler.")

    return CardDetection(True, len(centres), ppm, est, spread, implied,
                         centres, warnings)


def calibration_from_video(path: str, at_time_s: float = 5.0,
                           n_frames: int = 5) -> dict:
    """
    Read the scale from a recording, using several frames and taking the median.

    Several frames rather than one because a single frame can lose a marker to
    motion blur or a passing hand, and a calibration that silently falls back to
    three markers is worse than one that says so.
    """
    from modules import climbing_video as CV

    info = CV.probe_video(path)
    idx = [int((at_time_s + i * 0.5) * info.fps) for i in range(n_frames)]
    cap = cv2.VideoCapture(str(path))
    dets: List[CardDetection] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(info.n_frames - 1, i)))
        ok, img = cap.read()
        if ok:
            dets.append(detect_card(img))
    cap.release()

    good = [d for d in dets if d.found and np.isfinite(d.px_per_mm)]
    if not good:
        return {"available": False,
                "reason": ("No calibration card found. Either it was not in shot, or it is "
                           "too small in frame to resolve — the markers need to be at least "
                           "about 40 px across. Fall back to entering the cylinder diameter."),
                "detections": dets}

    ppm = float(np.median([d.px_per_mm for d in good]))
    warns = sorted({w for d in good for w in d.warnings})
    return {
        "available": True,
        "px_per_mm": ppm,
        "n_frames_with_card": len(good),
        "n_frames_tried": len(dets),
        "disagreement_pct": float(np.median([d.disagreement_pct for d in good])),
        "warnings": warns,
        "calibration": _as_calibration(ppm),
    }


def _as_calibration(px_per_mm: float):
    from modules.climbing_video import Calibration
    return Calibration(px_per_mm=px_per_mm, source="printed calibration card")
