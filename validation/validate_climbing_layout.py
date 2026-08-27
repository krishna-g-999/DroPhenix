"""
Validation suite for modules/climbing_layout.py

WHAT IS BEING VALIDATED, AND WHY THESE THINGS
---------------------------------------------
A drawing interface fails silently. A box in the wrong place looks exactly like a box
in the right place, and the numbers it produces look entirely ordinary. So the two
things that can go wrong invisibly are tested first and hardest.

A. CANVAS -> NATIVE COORDINATES. The canvas is displayed scaled to fit a browser, so
   every drawn coordinate must be scaled back to native pixels. Get the factor wrong
   and every box lands somewhere plausible and wrong. Tested against exact arithmetic,
   including the identity case and a non-square aspect.

B. LEFT-TO-RIGHT LABELLING. Drawn boxes must be named R1, R2 ... in the same order the
   automatic detector uses, or a drawn layout and a detected one disagree about which
   cylinder is which while both look right.

C. THE INVARIANTS THAT MADE THIS PAGE NECESSARY. A scoring line outside the box, a box
   too small to score, overlapping boxes, and a missing fly count are each refused with
   a reason. These are exactly the failures measured on real footage: the line was
   found outside the tube in 4 of 15 recordings, once at -1096% of tube height.

D. ALIGNMENT AND ITS REFUSAL. A layout carried to another recording must move by the
   real shift, and must REFUSE when the shift is large enough to slide the boxes off
   the glass. Measured on the paraquat set, shifts between recordings in one session
   ran from 0.2 px to 255 px against a 150-190 px vial.

E. ROUND TRIP. A saved layout must reload identically, and a layout written by another
   schema version must be refused rather than silently mis-read.

Run: ./.venv/Scripts/python.exe validation/validate_climbing_layout.py
Exit 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from modules import climbing_layout as LAY  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _rect(left, top, w, h):
    return {"type": "rect", "left": left, "top": top, "width": w, "height": h,
            "scaleX": 1, "scaleY": 1}


# ═════════════════════════════════════════════════════════════════════ tests ══
def test_A_coordinates() -> None:
    print("\nA. Canvas coordinates scale back to native pixels exactly")
    check("a half-scale canvas doubles the coordinate",
          LAY.canvas_to_native(100, 700, 1400) == 200,
          f"{LAY.canvas_to_native(100, 700, 1400)}")
    check("an unscaled canvas is the identity",
          LAY.canvas_to_native(123, 640, 640) == 123, "")
    check("a zero-size canvas cannot divide by zero",
          LAY.canvas_to_native(50, 0, 1920) == 0, "")

    # a real case: 1920x1080 shown at 700 wide
    disp_w, disp_h = 700, 394
    lay = LAY.layout_from_canvas([_rect(70, 39.4, 140, 197)], disp_w, disp_h,
                                 1920, 1080, line_canvas_y=197)
    v = lay.vials[0]
    check("a box drawn at 10% of the canvas lands at 10% of the frame",
          abs(v.x0 - 192) <= 2 and abs(v.x1 - 576) <= 2,
          f"x {v.x0}-{v.x1}, expected about 192-576")
    check("the shared line scales on the HEIGHT, not the width",
          abs(lay.line_y - 540) <= 3, f"line_y {lay.line_y}, expected about 540")
    check("the frame size is recorded with the layout",
          (lay.frame_w, lay.frame_h) == (1920, 1080), "")


def test_B_left_to_right() -> None:
    print("\nB. Boxes are numbered left to right, matching the detector")
    objs = [_rect(400, 10, 60, 300), _rect(100, 10, 60, 300), _rect(250, 10, 60, 300)]
    lay = LAY.layout_from_canvas(objs, 700, 400, 700, 400, line_canvas_y=200)
    check("three boxes drawn in any order become R1, R2, R3 by position",
          [v.label for v in lay.vials] == ["R1", "R2", "R3"],
          f"{[v.label for v in lay.vials]}")
    check("and R1 is the leftmost",
          lay.vials[0].x0 < lay.vials[1].x0 < lay.vials[2].x0,
          f"{[v.x0 for v in lay.vials]}")
    check("a non-rectangle scribble is ignored",
          len(LAY.layout_from_canvas([{"type": "path", "left": 1, "top": 1}],
                                     700, 400, 700, 400, 200).vials) == 0, "")


def test_C_invariants() -> None:
    print("\nC. The failures that made this page necessary are refused")
    good = LAY.Layout(vials=[LAY.DrawnVial(100, 50, 200, 500, "R1", n_flies=20)],
                      line_y=300, frame_w=800, frame_h=600)
    check("a sound layout has no problems", good.ok(), str(good.problems()))

    outside = LAY.Layout(vials=[LAY.DrawnVial(100, 50, 200, 500, "R1", n_flies=20)],
                         line_y=560, frame_w=800, frame_h=600)
    check("a line below the box is refused", not outside.ok(),
          outside.problems()[0][:80] if outside.problems() else "")
    check("...and the reason names the line",
          any("scoring line is outside" in m for m in outside.problems()), "")

    above = LAY.Layout(vials=[LAY.DrawnVial(100, 50, 200, 500, "R1", n_flies=20)],
                       line_y=10, frame_w=800, frame_h=600)
    check("a line above the box is refused too", not above.ok(), "")

    noflies = LAY.Layout(vials=[LAY.DrawnVial(100, 50, 200, 500, "R1", n_flies=0)],
                         line_y=300, frame_w=800, frame_h=600)
    check("a missing fly count is refused",
          any("flies loaded" in m for m in noflies.problems()),
          noflies.problems()[0][:90] if noflies.problems() else "")

    tiny = LAY.Layout(vials=[LAY.DrawnVial(100, 50, 105, 55, "R1", n_flies=20)],
                      line_y=52, frame_w=800, frame_h=600)
    check("a box too small to score is refused",
          any("too small" in m for m in tiny.problems()), "")

    over = LAY.Layout(vials=[LAY.DrawnVial(100, 50, 300, 500, "R1", n_flies=20),
                             LAY.DrawnVial(250, 50, 400, 500, "R2", n_flies=20)],
                      line_y=300, frame_w=800, frame_h=600)
    check("overlapping boxes are refused, since a fly would be counted twice",
          any("overlap" in m for m in over.problems()),
          over.problems()[0][:80] if over.problems() else "")

    v = LAY.DrawnVial(100, 50, 200, 500, "R1", n_flies=20, line_nudge_px=-40)
    check("a per-vial nudge moves that vial's line only",
          v.to_spec(300).line_y == 260, f"{v.to_spec(300).line_y}")


def test_D_alignment() -> None:
    print("\nD. A layout carried to another recording moves, or refuses")
    rng = np.random.default_rng(11)
    ref = (rng.integers(40, 210, (400, 600)).astype(np.uint8))
    ref = cv2.GaussianBlur(ref, (7, 7), 0)
    lay = LAY.Layout(vials=[LAY.DrawnVial(200, 60, 320, 360, "R1", n_flies=20)],
                     line_y=200, frame_w=600, frame_h=400)

    for dx, dy in ((6, -4), (-9, 3)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        moved_img = cv2.warpAffine(ref, M, (600, 400), borderMode=cv2.BORDER_REFLECT)
        r = LAY.align_to(lay, ref, moved_img)
        check(f"a {dx:+d},{dy:+d} px shift is applied",
              r["ok"] and abs(r["dx"] - dx) < 2 and abs(r["dy"] - dy) < 2,
              f"estimated {r['dx']:+.1f},{r['dy']:+.1f}; {r['reason'][:50]}")
        if r["ok"]:
            check(f"...and the box follows it ({dx:+d},{dy:+d})",
                  abs(r["layout"].vials[0].x0 - (200 + dx)) <= 2
                  and abs(r["layout"].line_y - (200 + dy)) <= 2,
                  f"x0 {r['layout'].vials[0].x0}, line {r['layout'].line_y}")

    M = np.float32([[1, 0, 90], [0, 1, 0]])           # 90 px on a 120 px vial
    far = cv2.warpAffine(ref, M, (600, 400), borderMode=cv2.BORDER_REFLECT)
    r = LAY.align_to(lay, ref, far)
    check("a shift larger than a quarter of the vial width is REFUSED",
          not r["ok"], f"{r['reason'][:90]}")
    check("...and the refusal says how far it moved",
          "px" in r["reason"] and "draw this recording separately" in r["reason"],
          r["reason"][:100])

    small = np.zeros((100, 100), np.uint8)
    r2 = LAY.align_to(lay, ref, small)
    check("recordings of different sizes are refused",
          not r2["ok"] and "different sizes" in r2["reason"], r2["reason"][:70])
    r3 = LAY.align_to(lay, None, ref)
    check("a missing frame is refused rather than crashing",
          not r3["ok"], r3["reason"][:60])


def test_E_round_trip() -> None:
    print("\nE. A saved layout reloads identically, or is refused")
    lay = LAY.Layout(
        vials=[LAY.DrawnVial(100, 50, 200, 500, "R1", genotype="WT", n_flies=20),
               LAY.DrawnVial(300, 50, 400, 500, "R2", genotype="TDP", n_flies=15)],
        line_y=300, frame_w=800, frame_h=600, reference_video="a.mp4", t0_s=3.5)
    back = LAY.Layout.from_json(lay.to_json())
    check("every vial survives the round trip",
          [(v.label, v.x0, v.x1, v.genotype, v.n_flies) for v in back.vials]
          == [(v.label, v.x0, v.x1, v.genotype, v.n_flies) for v in lay.vials], "")
    check("the line, the frame size and the tap survive too",
          (back.line_y, back.frame_w, back.frame_h, back.t0_s)
          == (lay.line_y, lay.frame_w, lay.frame_h, lay.t0_s), "")
    check("the reloaded layout still validates", back.ok(), str(back.problems()))
    check("the vials convert to the specs the scoring routines take",
          [s.label for s in back.specs()] == ["R1", "R2"]
          and back.specs()[0].line_y == 300, "")

    bad = lay.to_json().replace('"schema": 1', '"schema": 99')
    check("a layout from a different schema version is refused, not mis-read",
          _raises(lambda: LAY.Layout.from_json(bad)), "ValueError raised")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/climbing_layout.py")
    print("=" * 78)
    test_A_coordinates()
    test_B_left_to_right()
    test_C_invariants()
    test_D_alignment()
    test_E_round_trip()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CLIMBING-LAYOUT VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
