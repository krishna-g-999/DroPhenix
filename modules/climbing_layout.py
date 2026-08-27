"""
Hand-drawn vial layouts: draw once, align to the rest of the session, refuse when unsafe.

WHY THIS EXISTS
---------------
Automatic geometry is where this pipeline failed hardest, and every failure was of a
kind a drawn box removes outright:

  * the scoring line was found OUTSIDE the tube in 4 of 15 recordings, once at
    -1096% of tube height (below the base);
  * on 3 of 3 TDP recordings `_find_line` chose the blue bench mat over the drawn
    line and sat 330, 437 and 632 px too low;
  * vial boxes were placed on the frame edges while the cylinders stood in the middle;
  * three cylinders were detected where two exist;
  * cylinder bases came out 649 px apart although they stand on one bench;
  * the detected region ran 74 px to 1079 px for identical glassware, because it is
    the fly-occupied envelope rather than the glass -- so the measurement origin
    moved with the phenotype being measured.

Half the exported rows of the first dataset (71 of 143) came from recordings with
geometry in that state, 18 of them graded "good".

WHAT IS DELIBERATELY CONSTRAINED
--------------------------------
ONE LINE FOR ALL VIALS. The line is drawn on the glassware with a ruler at a single
height, so the vials' line positions are not independent measurements. Treating them
as independent produced a `line_fraction` spread of 1.34 across three cylinders in one
frame. The constraint is encoded here rather than checked afterwards; a per-vial nudge
remains available for a cylinder that genuinely sits higher on the bench.

DRAW ONCE, ALIGN, BUT NEVER SILENTLY
------------------------------------
The camera does NOT reliably hold still between recordings in a session. On two
consecutive WT recordings the cylinder moved from x 1133-1311 to x 1201-1398 -- about
78 px against a 190 px vial, so a layout carried across unchanged would sit a third of
a tube off the glass.

That number was itself hard-won, and the way it was got wrong is worth recording. An
earlier measurement downscaled both frames to 480 px before correlating and reported
+0.7, +0.2 px with a confident response of 0.486 -- the rig stands against white
tiling, tiling is periodic, and the grout lines matched themselves. Whole-frame
correlation at full resolution got dx right (+77.8) but dy badly wrong (+145).
Restricting the correlation to a padded box around the drawn vials -- where the
glassware is and the repeating pattern is not -- gave +77.9, +12.1, which agrees with
where the cylinder actually is.

So `align_to` correlates on the GLASSWARE, not the frame, and refuses on two grounds:
when the shift exceeds a fraction of the vial width, and when the correlation response
is too low to trust. A wrong shift that looks certain is worse than no shift at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import cv2

from . import climbing_video as CV

__all__ = [
    "DrawnVial", "Layout", "canvas_to_native", "layout_from_canvas",
    "align_to", "MAX_SHIFT_FRAC_OF_WIDTH", "MIN_CORRELATION_RESPONSE",
    "LAYOUT_SCHEMA_VERSION",
]

LAYOUT_SCHEMA_VERSION = 1

#: A layout may be carried to another recording only if the estimated shift is below
#: this fraction of the vial width. Beyond it the box no longer sits on the cylinder.
#: Set from the measured spread: shifts of 0.2-16 px against a 150-190 px vial are
#: harmless, a 140 px shift is fatal.
MAX_SHIFT_FRAC_OF_WIDTH = 0.25

#: Phase correlation returns a response in [0, 1]. A low value means the match is
#: ambiguous, which is what a periodic background produces -- and a wrong shift that
#: looks certain is worse than no shift at all.
MIN_CORRELATION_RESPONSE = 0.08


@dataclass
class DrawnVial:
    """One cylinder, as the operator drew it, in NATIVE frame pixels."""

    x0: int
    y0: int
    x1: int
    y1: int
    label: str = ""
    genotype: str = ""
    treatment: str = "Untreated"
    sex: str = "Mixed"
    n_flies: int = 0            # flies loaded -- required, see Layout.problems()
    line_nudge_px: int = 0      # per-vial adjustment to the shared line

    @property
    def width(self) -> int:
        return int(abs(self.x1 - self.x0))

    @property
    def height(self) -> int:
        return int(abs(self.y1 - self.y0))

    def line_row(self, shared_line_y: int) -> int:
        return int(shared_line_y + self.line_nudge_px)

    def to_spec(self, shared_line_y: int) -> "CV.VialSpec":
        """The region in the form every scoring routine already accepts."""
        return CV.VialSpec(
            x0=int(min(self.x0, self.x1)), x1=int(max(self.x0, self.x1)),
            y_top=int(min(self.y0, self.y1)), y_bottom=int(max(self.y0, self.y1)),
            line_y=self.line_row(shared_line_y),
            label=self.label or "R?", source="manual")


@dataclass
class Layout:
    """A drawn layout: the vials, the scoring line, and its provenance.

    `line_y` is the shared height the line was ruled at; each vial's
    `line_nudge_px` is the offset that height lands at in that particular tube,
    because the tubes are not all photographed from dead centre. Use
    `set_vial_lines` to write operator-confirmed rows into both fields.
    """

    vials: List[DrawnVial] = field(default_factory=list)
    line_y: int = 0                       # shared, in native frame pixels
    frame_w: int = 0
    frame_h: int = 0
    reference_video: str = ""
    t0_s: float = 0.0
    schema: int = LAYOUT_SCHEMA_VERSION
    notes: List[str] = field(default_factory=list)

    # ── validity ────────────────────────────────────────────────────────────
    def problems(self) -> List[str]:
        """Everything that would make this layout unscoreable, in plain words."""
        out: List[str] = []
        if not self.vials:
            out.append("No vials were drawn.")
        for v in self.vials:
            if v.width < 10 or v.height < 10:
                out.append(f"{v.label or 'a vial'}: the box is too small to score "
                           f"({v.width}x{v.height} px).")
            top, bot = min(v.y0, v.y1), max(v.y0, v.y1)
            if not (top <= v.line_row(self.line_y) <= bot):
                out.append(
                    f"{v.label or 'a vial'}: the scoring line is outside the box. A "
                    f"line above or below the tube is not a measurement -- move the "
                    f"line, or extend the box to cover the whole cylinder.")
            if v.n_flies <= 0:
                out.append(
                    f"{v.label or 'a vial'}: the number of flies loaded is not set. "
                    f"Without it the denominator is taken from the detector, which "
                    f"changed across timepoints in 24 of 24 vials the last time it "
                    f"was relied on -- for sealed cylinders whose count cannot change.")
        for a, b in _overlaps(self.vials):
            out.append(f"{a} and {b} overlap, so a fly could be counted in both.")
        return out

    def ok(self) -> bool:
        return not self.problems()

    # ── persistence ─────────────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Layout":
        d = json.loads(text)
        got = int(d.get("schema", 0))
        if got != LAYOUT_SCHEMA_VERSION:
            raise ValueError(
                f"This layout was written by schema version {got}; this build reads "
                f"version {LAYOUT_SCHEMA_VERSION}. Re-draw it rather than risk boxes "
                f"landing in the wrong place.")
        vials = [DrawnVial(**v) for v in d.pop("vials", [])]
        return Layout(vials=vials, **{k: v for k, v in d.items() if k != "vials"})

    def specs(self) -> List["CV.VialSpec"]:
        return [v.to_spec(self.line_y) for v in self.vials]


def set_vial_lines(layout: "Layout", rows_by_label: Dict[str, int]) -> "Layout":
    """Carry an operator-confirmed line for each vial into the layout, exactly.

    The layout stores one shared `line_y` plus a per-vial `line_nudge_px`, because
    the line really is drawn on the glass at a single height with a ruler. But the
    tubes are not photographed from dead centre, so the same physical height lands
    on a slightly different row in each one: measured independently, the paraquat
    pair gave 542 and 538, and the Day7 trio gave 581, 558 and 590. Those are the
    same line seen from three angles, not three different lines.

    So the shared height is kept as the median and each vial carries its own offset.
    The round trip is exact by construction -- `v.line_row(layout.line_y)` returns
    precisely the row that was confirmed, with no rounding drift -- which matters
    because a line off by a few pixels moves every fly sitting near it.

    Labels absent from `rows_by_label` keep the shared line.
    """
    if not layout.vials:
        return layout
    rows = [int(rows_by_label[v.label]) for v in layout.vials
            if v.label in rows_by_label]
    if not rows:
        return layout
    base = int(np.median(rows))
    layout.line_y = base
    for v in layout.vials:
        if v.label in rows_by_label:
            v.line_nudge_px = int(rows_by_label[v.label]) - base
    return layout


def _overlaps(vials: List[DrawnVial]) -> List[Tuple[str, str]]:
    out = []
    for i, a in enumerate(vials):
        for b in vials[i + 1:]:
            ax0, ax1 = sorted((a.x0, a.x1))
            bx0, bx1 = sorted((b.x0, b.x1))
            ay0, ay1 = sorted((a.y0, a.y1))
            by0, by1 = sorted((b.y0, b.y1))
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                out.append((a.label or "a vial", b.label or "another vial"))
    return out


# ─────────────────────────────────────────────────────── canvas coordinates ──
def canvas_to_native(value: float, canvas_size: int, native_size: int) -> int:
    """Scale one coordinate from the displayed canvas back to native pixels.

    The canvas is shown at whatever width fits the browser, so every drawn
    coordinate is in display units and must be scaled back. Getting this wrong puts
    every box in the wrong place while looking entirely plausible, which is why it
    is a named function with its own test rather than an inline multiplication.
    """
    if canvas_size <= 0:
        return 0
    return int(round(float(value) * float(native_size) / float(canvas_size)))


def layout_from_canvas(objects: List[dict], canvas_w: int, canvas_h: int,
                       frame_w: int, frame_h: int, line_canvas_y: float,
                       reference_video: str = "", t0_s: float = 0.0) -> Layout:
    """Build a Layout from `streamlit-drawable-canvas` rectangles.

    Boxes are sorted LEFT TO RIGHT and labelled R1, R2 ... which is the convention
    the rest of the pipeline already uses, so a drawn layout and an automatically
    detected one name the same cylinder the same way.
    """
    rects = []
    for o in objects or []:
        if str(o.get("type", "")).lower() != "rect":
            continue
        left, top = float(o.get("left", 0)), float(o.get("top", 0))
        w = float(o.get("width", 0)) * float(o.get("scaleX", 1) or 1)
        h = float(o.get("height", 0)) * float(o.get("scaleY", 1) or 1)
        if w <= 0 or h <= 0:
            continue
        rects.append((
            canvas_to_native(left, canvas_w, frame_w),
            canvas_to_native(top, canvas_h, frame_h),
            canvas_to_native(left + w, canvas_w, frame_w),
            canvas_to_native(top + h, canvas_h, frame_h)))
    rects.sort(key=lambda r: r[0])

    vials = [DrawnVial(x0=x0, y0=y0, x1=x1, y1=y1, label=f"R{i}")
             for i, (x0, y0, x1, y1) in enumerate(rects, 1)]
    return Layout(vials=vials,
                  line_y=canvas_to_native(line_canvas_y, canvas_h, frame_h),
                  frame_w=int(frame_w), frame_h=int(frame_h),
                  reference_video=reference_video, t0_s=float(t0_s))


# ────────────────────────────────────────────── carrying it to another clip ──
def align_to(layout: Layout, reference_frame: np.ndarray,
             target_frame: np.ndarray,
             max_shift_frac: float = MAX_SHIFT_FRAC_OF_WIDTH) -> dict:
    """Translate a layout onto another recording, or refuse.

    Returns `{ok, layout, dx, dy, reason}`. The shift is estimated by phase
    correlation on the whole frame; only translation is applied, because the
    residual between two recordings of a stationary rig is dominated by it and
    fitting more parameters to a few pixels invites overfitting.
    """
    if reference_frame is None or target_frame is None:
        return {"ok": False, "layout": layout, "dx": 0.0, "dy": 0.0,
                "reason": "a frame could not be read from one of the recordings"}

    a = reference_frame if reference_frame.ndim == 2 else cv2.cvtColor(
        reference_frame, cv2.COLOR_BGR2GRAY)
    b = target_frame if target_frame.ndim == 2 else cv2.cvtColor(
        target_frame, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape:
        return {"ok": False, "layout": layout, "dx": 0.0, "dy": 0.0,
                "reason": (f"the recordings are different sizes ({a.shape[1]}x"
                           f"{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}); a layout "
                           f"cannot be carried between them")}

    # Correlate on the GLASSWARE, not the whole frame. The rig here stands against
    # white tiling, and tiling is periodic: measured on two recordings whose cylinder
    # genuinely moved from x 1133-1311 to x 1201-1398 (about +78 px), whole-frame
    # correlation returned dy = +145 px (response 0.044) and the same frames
    # downscaled returned +0.7, +0.2 px (response 0.486) -- a confident zero that was
    # simply the grout lines matching themselves. Restricted to a padded box around
    # the vials the estimate was +77.9, +12.1 px (response 0.168), which agrees with
    # where the cylinder actually is.
    xs = [c for v in layout.vials for c in (min(v.x0, v.x1), max(v.x0, v.x1))]
    ys = [c for v in layout.vials for c in (min(v.y0, v.y1), max(v.y0, v.y1))]
    if xs and ys:
        pad = int(0.6 * float(np.median([v.width for v in layout.vials]) or 0))
        x0 = int(np.clip(min(xs) - pad, 0, a.shape[1] - 2))
        x1 = int(np.clip(max(xs) + pad, x0 + 2, a.shape[1]))
        y0 = int(np.clip(min(ys) - pad, 0, a.shape[0] - 2))
        y1 = int(np.clip(max(ys) + pad, y0 + 2, a.shape[0]))
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]

    (dx, dy), response = cv2.phaseCorrelate(np.float32(a), np.float32(b))
    if response < MIN_CORRELATION_RESPONSE:
        return {"ok": False, "layout": layout, "dx": float(dx), "dy": float(dy),
                "reason": (f"the two recordings could not be matched confidently "
                           f"(correlation {response:.3f}, below "
                           f"{MIN_CORRELATION_RESPONSE:.2f}). A repeating background "
                           f"such as tiling can make a wrong shift look certain, so "
                           f"this layout is not carried over -- draw this recording "
                           f"separately.")}
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return {"ok": False, "layout": layout, "dx": 0.0, "dy": 0.0,
                "reason": "the shift between the recordings could not be estimated"}

    widths = [v.width for v in layout.vials] or [1]
    limit = float(max_shift_frac) * float(np.median(widths))
    shift = float(np.hypot(dx, dy))
    if shift > limit:
        return {"ok": False, "layout": layout, "dx": float(dx), "dy": float(dy),
                "reason": (f"the camera moved {shift:.0f} px between these "
                           f"recordings, which is more than {max_shift_frac:.0%} of a "
                           f"{np.median(widths):.0f} px vial. The boxes would no "
                           f"longer sit on the cylinders -- draw this recording "
                           f"separately.")}

    moved = Layout(
        vials=[DrawnVial(**{**asdict(v),
                            "x0": int(round(v.x0 + dx)), "x1": int(round(v.x1 + dx)),
                            "y0": int(round(v.y0 + dy)), "y1": int(round(v.y1 + dy))})
               for v in layout.vials],
        line_y=int(round(layout.line_y + dy)),
        frame_w=layout.frame_w, frame_h=layout.frame_h,
        reference_video=layout.reference_video, t0_s=layout.t0_s,
        notes=list(layout.notes) + [f"aligned by {dx:+.1f}, {dy:+.1f} px"])
    return {"ok": True, "layout": moved, "dx": float(dx), "dy": float(dy),
            "response": float(response),
            "reason": f"aligned by {dx:+.1f}, {dy:+.1f} px "
                      f"(match confidence {response:.2f})"}
