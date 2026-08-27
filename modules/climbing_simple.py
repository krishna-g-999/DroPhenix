"""
The simple pipeline: boxes in, line found by darkness, flies counted above it.

WHAT THIS IS, AND WHY IT EXISTS SEPARATELY
------------------------------------------
`climbing_video.py` grew a great deal of machinery to place the scoring geometry
automatically -- an activity-map band finder, a middle-band line search, one shared
line imposed across cylinders, and a layout that could be carried between recordings
by phase correlation. Each piece was added to fix a real, measured failure. Together
they were still wrong about half the geometry, and the correlation guard eventually
refused correct alignments.

This module does the three things the assay actually requires and nothing else:

  1. the operator states where the cylinders are;
  2. the scoring line is the DARKEST HORIZONTAL FEATURE inside each box;
  3. flies above that line are counted at each timepoint.

Step 2 is the part that was measured rather than assumed. The line is drawn in black
marker; the graduations are printed in brown. Inside a box around a cylinder the
marker wins on darkness every time, and by a wide margin:

    Day7  C0142   P1 depth 66 vs 31 next   P2 55 vs 34   P3 89 vs 32
    Paraquat      R1 depth 67 vs 60        R2 80 vs 68

and it is unaffected by the blue bench mat -- extending the box down over the mat
returned the identical row (529 and 532). The earlier `_find_line` failure, which put
the line 330-632 px too low on three recordings, was not a contrast problem at all:
it was searching a wrongly-placed region. Once the box is on the tube the problem
disappears, and the middle-band workaround built for it is unnecessary.

THE PUBLISHED METHOD DOES THE SAME
----------------------------------
Cao et al. 2017 (J Vis Exp 127:e56507), the automated RING assay, has the operator
mark the vial: "Set the upper and lower baselines of the vial precisely by using the
baseline icon on the toolbar and then using the cursor to mark the upper and lower
baselines on the image." Their software finds flies inside that. Automatic vial
detection is a convenience, not a requirement of the field-standard method.

That paper also scores a SNAPSHOT at 6 s after each tap and reports mean climbing
DISTANCE, not a count above a line. Both are produced here, so a result can be
compared with the literature as well as with a hand count.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No layout alignment between recordings, no shared-line constraint, no band
restriction on the line search, no automatic cylinder detection. If a box is wrong
the operator moves it; nothing downstream tries to rescue it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import cv2

__all__ = [
    "Box", "LineFit", "find_line", "detect_flies", "score_video",
    "DEFAULTS", "JOVE_SNAPSHOT_S",
]

#: Cao et al. score a single snapshot at this many seconds after each tap.
JOVE_SNAPSHOT_S = 6.0

DEFAULTS: dict = {
    # A fly spans roughly a tenth of a cylinder's width, so its area is ~0.01*w^2.
    # Anchored to WIDTH, never to box area: a size window that grows with the box
    # would mean a taller rectangle silently discards small flies.
    "min_area_w2": 0.0016,
    "max_area_w2": 0.081,
    "max_aspect": 6.0,        # a fly is compact; a drawn line or a scratch is not
    "blur": 5,
    "block": 21,              # adaptive-threshold neighbourhood
    "C": 4,
    "open_iter": 2,
    "close_iter": 2,
    "line_inset": 0.18,       # ignore this fraction of the box width at each edge
    "line_mask_px": 3,        # rows either side of the line excluded from counting
    # The line is searched only in the middle band of the box. Below the band sit
    # the sticky label, the tube foot and the bench mat, and on the paraquat rig
    # those are DARKER than the marker: with the box run to the tube base the
    # darkest-row rule put the line at the 10 mL mark on the label, 400+ px below
    # the real one, on 12 of 12 vials. Above the band sit the brand sticker and
    # the cotton plug. A scoring line is drawn where flies are expected to reach.
    "line_band_lo": 0.20,     # fraction of box height from the TOP
    "line_band_hi": 0.75,
    "line_min_depth": 25.0,   # a line fainter than this is not a line
    "line_min_margin": 1.3,   # ...and one no darker than its rival is a coin toss
    # A mark drawn on the tube STOPS at the glass. The bench mat's top edge, a shelf
    # and a skirting board all continue past it. Comparing darkness inside the box
    # with darkness in a strip just outside it separates the two: measured, the mat
    # edge that captured one vial at row 852 is equally dark outside the tube, while
    # the real marker at row 538 is not.
    "line_outside_px": 40,    # width of the comparison strip either side
    "line_max_outside_ratio": 0.6,   # reject if outside darkness > this x inside
}


@dataclass
class Box:
    """One cylinder, as the operator marked it, in native frame pixels."""

    x0: int
    y0: int
    x1: int
    y1: int
    label: str = ""
    n_flies: int = 0
    genotype: str = ""
    treatment: str = ""
    sex: str = "Mixed"
    tube_mm: Optional[float] = None    # cylinder height, for climbing distance in mm

    def norm(self) -> "Box":
        return Box(min(self.x0, self.x1), min(self.y0, self.y1),
                   max(self.x0, self.x1), max(self.y0, self.y1),
                   self.label, self.n_flies, self.genotype, self.treatment,
                   self.sex, self.tube_mm)

    @property
    def w(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def h(self) -> int:
        return int(self.y1 - self.y0)


@dataclass
class LineFit:
    """Where the scoring line is, and how confidently."""

    row: int                  # absolute frame row
    depth: float              # how much darker than the tube's brightest row
    runner_up_depth: float    # the next distinct candidate
    margin: float             # depth / runner_up -- 1.0 means a coin toss
    row_in_box: int

    def is_clear(self, min_depth: float = 25.0, min_margin: float = 1.3) -> bool:
        """A line worth trusting is both dark and clearly darker than its rival.

        Measured: on the paraquat rig the whole-box rule returned margins of
        1.02-1.06 while placing the line 400 px wrong, so a margin near 1.0 is the
        signature of a guess. Below `min_margin` the operator must place the line.
        """
        return self.depth >= min_depth and self.margin >= min_margin

    def verdict(self) -> str:
        if self.is_clear():
            return "clear"
        if self.depth < 25.0:
            return f"too faint to be a drawn line (depth {self.depth:.0f})"
        return (f"not distinguishable from another feature "
                f"(margin {self.margin:.2f}) -- place it by hand")


# ─────────────────────────────────────────────────────────────── the line ──
def find_line(background: np.ndarray, box: Box,
              params: Optional[dict] = None) -> LineFit:
    """PROPOSE the scoring line: the darkest horizontal feature in the middle band.

    This is a proposal, not a decision. The caller is expected to show it to the
    operator, because the published method (Cao et al. 2017) has a person mark the
    vial baselines and nothing here is better than that.

    WHY THE BAND, MEASURED
    ----------------------
    An earlier version searched the whole box, on the reasoning that the black
    marker must outcontrast the brown printed graduations. It does -- but only
    against the PRINTING. Run the box down to the tube base and the darkest thing
    inside it is the orange sticky label and the blue bench mat behind the foot.
    Rendered on a paraquat frame, that rule placed the line at the 10 mL mark while
    the true mark sat at 40 mL, and it did so on 12 of 12 vials with a margin of
    1.02-1.06 over the runner-up -- i.e. it was never confident, and the margin said
    so.

    Restricting the search to the middle band removes the label, the foot, the mat,
    the brand sticker and the cotton plug in one step, and leaves the marker as the
    only strongly dark horizontal feature.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    b = box.norm()
    inset = int(p["line_inset"] * b.w)
    xa, xb = b.x0 + inset, max(b.x0 + inset + 1, b.x1 - inset)
    strip = background[b.y0:b.y1, xa:xb]
    if strip.size == 0:
        return LineFit(b.y0, 0.0, 0.0, 1.0, 0)

    prof = strip.mean(axis=1).astype(np.float32)
    sm = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    depth = float(sm.max()) - sm

    h = len(depth)
    lo = int(np.clip(p["line_band_lo"] * h, 0, max(h - 2, 0)))
    hi = int(np.clip(p["line_band_hi"] * h, lo + 1, h))
    band = depth[lo:hi]
    if band.size == 0:
        return LineFit(b.y0 + h // 2, 0.0, 0.0, 1.0, h // 2)

    # Reject candidates that are equally dark OUTSIDE the tube: those are scene
    # edges (the mat, a shelf, a skirting board), not something drawn on glass.
    op = int(p["line_outside_px"])
    fh, fw = background.shape[:2]
    lx0, lx1 = max(0, b.x0 - op), max(1, b.x0 - 4)
    rx0, rx1 = min(fw - 1, b.x1 + 4), min(fw, b.x1 + op)
    outs = []
    if lx1 > lx0:
        outs.append(background[b.y0:b.y1, lx0:lx1].mean(axis=1).astype(np.float32))
    if rx1 > rx0:
        outs.append(background[b.y0:b.y1, rx0:rx1].mean(axis=1).astype(np.float32))
    if outs:
        o = np.mean(np.stack(outs), axis=0)
        o = cv2.GaussianBlur(o.reshape(-1, 1), (1, 9), 0).ravel()
        out_depth = float(o.max()) - o
    else:
        out_depth = np.zeros_like(depth)

    ratio_cap = float(p["line_max_outside_ratio"])
    ok_rows = np.arange(lo, hi)
    inside = depth[lo:hi]
    outside = out_depth[lo:hi] if out_depth.size == depth.size else np.zeros_like(inside)
    confined = outside <= ratio_cap * np.maximum(inside, 1e-6)
    if confined.any():
        cand = np.where(confined, inside, -np.inf)
        k = lo + int(np.argmax(cand))
    else:
        k = lo + int(np.argmax(inside))

    runner = 0.0
    order = lo + np.argsort(-np.where(confined, inside, -np.inf))
    for i in order:
        if abs(int(i) - k) > 15 and np.isfinite(depth[int(i)]):
            runner = float(depth[int(i)])
            break
    return LineFit(row=b.y0 + k, depth=float(depth[k]), runner_up_depth=runner,
                   margin=float(depth[k] / runner) if runner > 1e-6 else float("inf"),
                   row_in_box=k)


# ───────────────────────────────────────────────────────────── the flies ──
def detect_flies(frame_bgr: np.ndarray, box: Box, line_row: int,
                 params: Optional[dict] = None) -> dict:
    """Fly centroids inside the box, split by the line.

    Adaptive thresholding, because a cylinder has a lighting gradient down it and a
    global threshold cannot serve both ends. Long horizontal structures are removed
    first: the scoring line itself is dark, fly-sized in area, and would otherwise be
    counted as an animal -- measured, it added exactly one false fly on synthetic
    footage where 20 were drawn.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    b = box.norm()
    fh, fw = frame_bgr.shape[:2]
    x0, y0 = max(0, b.x0), max(0, b.y0)
    x1, y1 = min(fw, b.x1), min(fh, b.y1)
    if x1 <= x0 or y1 <= y0:
        return {"centroids": [], "above": 0, "below": 0, "total": 0,
                "heights_px": np.array([])}

    roi = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h, w = gray.shape[:2]

    k = int(p["blur"]) | 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    block = int(p["block"]) | 1
    block = max(5, min(block, (min(h, w) // 2) * 2 + 1))
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, float(p["C"]))

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, el, iterations=int(p["open_iter"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, el, iterations=int(p["close_iter"]))

    # remove the drawn line and any other long horizontal run
    run = max(15, int(0.5 * w))
    if run < w:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1))
        long_h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
        if long_h.max() > 0:
            rows = np.flatnonzero(long_h.max(axis=1) > 0)
            mask[int(rows.min()):int(rows.max()) + 1, :] = 0

    # and blank a narrow band at the stated line, so its edges cannot be flies
    lr = int(np.clip(line_row - y0, 0, h - 1))
    m = int(p["line_mask_px"])
    if m > 0:
        mask[max(0, lr - m):min(h, lr + m + 1), :] = 0

    w2 = float(w * w)
    min_a, max_a = p["min_area_w2"] * w2, p["max_area_w2"] * w2
    max_asp = float(p["max_aspect"])

    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    cents, heights = [], []
    for i in range(1, n):
        a = float(stats[i, cv2.CC_STAT_AREA])
        if not (min_a <= a <= max_a):
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max_asp > 0 and max(bw, bh) / max(min(bw, bh), 1) > max_asp:
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        cents.append((cx, cy))
        heights.append(float(h - cy))          # px above the bottom of the box

    above = sum(1 for _, cy in cents if cy < lr)
    return {"centroids": cents, "above": int(above),
            "below": int(len(cents) - above), "total": int(len(cents)),
            "heights_px": np.asarray(heights, float)}


# ────────────────────────────────────────────────────────────── the score ──
def score_video(video_path: str, boxes: Sequence[Box], t0_s: float,
                times_s: Sequence[float], n_frames: int = 5,
                params: Optional[dict] = None,
                background: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Score one recording: a row per box per timepoint.

    The count is the number STANDING above the line, which is what the protocol's
    up-minus-down tally comes to: "record that fly as -1 and count the next fly to
    cross as the same number" means the tally steps back when a fly falls.

    Mean climbing height is reported alongside, in millimetres where `tube_mm` is
    given -- that is the quantity Cao et al. report, and it uses every fly instead of
    collapsing the cohort to a proportion.
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        cap.release()
        raise RuntimeError("video reports no frame rate")

    if background is None:
        idx = np.linspace(int(4 * fps), max(int(4 * fps) + 1, total - 5), 30).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if ok:
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        background = np.median(np.stack(frames), axis=0).astype(np.uint8)

    fits = {b.label: find_line(background, b, p) for b in boxes}

    rows = []
    for t in [float(x) for x in times_s]:
        f0 = int((t0_s + t) * fps)
        if f0 >= total:
            for b in boxes:
                rows.append({"vial": b.label, "time_s": t, "n_above": np.nan,
                             "reason": "beyond the end of the recording"})
            continue
        grabs = []
        for j in range(int(n_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, f0 + j))
            ok, im = cap.read()
            if ok:
                grabs.append(im)
        for b in boxes:
            fit = fits[b.label]
            if not grabs:
                rows.append({"vial": b.label, "time_s": t, "n_above": np.nan,
                             "reason": "no frame could be read"})
                continue
            res = [detect_flies(g, b, fit.row, p) for g in grabs]
            above = float(np.median([r["above"] for r in res]))
            det = float(np.median([r["total"] for r in res]))
            hs = np.concatenate([r["heights_px"] for r in res]) if res else np.array([])
            mean_h = float(np.mean(hs)) if hs.size else np.nan
            bh = b.norm().h
            rows.append({
                "vial": b.label, "time_s": t,
                "n_above": above, "n_detected": det,
                "n_flies": b.n_flies,
                "fraction": (above / b.n_flies) if b.n_flies else np.nan,
                "mean_height_frac": (mean_h / bh) if bh else np.nan,
                "mean_height_mm": ((mean_h / bh) * b.tube_mm
                                   if (bh and b.tube_mm) else np.nan),
                "line_row": fit.row, "line_depth": round(fit.depth, 1),
                "line_margin": round(fit.margin, 2),
                "line_is_clear": fit.is_clear(),
                "reason": "" if fit.is_clear() else
                          f"the scoring line is not clearly the darkest feature "
                          f"(depth {fit.depth:.0f}, next {fit.runner_up_depth:.0f})",
            })
    cap.release()
    return pd.DataFrame(rows)
