"""
Validation suite for the operator-confirmable scoring line (Draw Vials, section 3b).

WHY THIS SUITE EXISTS
---------------------
The scoring line decides every number this module produces. A fly is climbed or not
climbed according to which side of it the animal sits, so a line in the wrong place
does not fail loudly -- it returns a full set of confident, ordinary-looking counts
that are wrong. That is the failure this page was rebuilt to stop, and it has already
happened on this project's own footage: the whole-box darkest-row rule put the line on
the bench mat at the 10 mL graduation, roughly 400 px low, on 12 of 12 paraquat vials,
while reporting margins of 1.02-1.06.

So the design is: the software PROPOSES, and says how sure it is; the operator
CONFIRMS. That is also what the published method does -- Cao et al. 2017 (J Vis Exp
127:e56507) have the operator mark the vial baselines with a cursor.

WHAT IS VALIDATED HERE
----------------------
A. THE ARITHMETIC IS EXACT. A confirmed row must come back out of the layout bit for
   bit. The layout stores a shared height plus a per-vial offset, so the round trip
   runs through a median and a subtraction; if that drifts by even a pixel, every fly
   sitting near the line can flip. Tested against closed-form equality, including the
   even-count median and negative offsets.

B. THE GATE ACTUALLY GATES. A proposal below the confidence thresholds must be
   reported as unclear, in the words the operator will act on. The thresholds are the
   ones measured on real footage, not chosen for convenience.

C. A CONFIRMED-BUT-IMPOSSIBLE LINE IS STILL REFUSED. Operator confirmation is not a
   licence to score a line above or below the tube. The per-vial offset must be
   validated against that vial's own box.

D. THE PROPOSAL IS HONEST ON REAL FOOTAGE. `find_line` is run on real recordings from
   both rigs and the clear/unclear split is REPORTED, not asserted. This suite does
   not claim an accuracy for the proposal; it only checks that the confidence flag
   tracks the thing it claims to track, and that no vial is silently marked clear
   when its line sits outside the glassware.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
It does not measure how often the proposal is right. That requires hand-marked lines,
which do not exist for these recordings. The accuracy that IS measured -- counts
against hand counts on the paraquat set -- lives in the head-to-head comparison, not
here.

Run: ./.venv/Scripts/python.exe validation/validate_confirmable_line.py
Exit 0 = all pass.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import climbing_layout as LAY  # noqa: E402
from modules import climbing_simple as CS  # noqa: E402

FAILURES: list[str] = []

# Real footage, if this machine has it. Absent = those checks are SKIPPED and said to
# be skipped; they are never quietly counted as passes.
RIGS = {
    "paraquat": r"D:/DroPhenix/Testing data/Climbing assay/"
                r"Paraquat climbing assay 010626_new set/**/*.MP4",
    "Day7": r"D:/DroPhenix/Testing data/Climbing assay/Day7/*.MP4",
}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def _layout(rows_and_boxes) -> LAY.Layout:
    vials = [LAY.DrawnVial(x0=x0, x1=x0 + 120, y0=y0, y1=y1, label=lab, n_flies=20)
             for lab, x0, y0, y1 in rows_and_boxes]
    return LAY.Layout(vials=vials, line_y=0, frame_w=1920, frame_h=1080,
                      reference_video="synthetic.mp4")


# ═════════════════════════════════════════════════════════════════════ tests ══
def test_A_exact_round_trip() -> None:
    print("\nA. A confirmed row comes back out of the layout exactly")

    lay = _layout([("R1", 100, 200, 900), ("R2", 400, 200, 900),
                   ("R3", 700, 200, 900)])
    rows = {"R1": 581, "R2": 558, "R3": 590}      # the real Day7 measurements
    LAY.set_vial_lines(lay, rows)
    got = {v.label: v.line_row(lay.line_y) for v in lay.vials}
    check("three vials round-trip exactly", got == rows, f"{got}")
    check("the shared height is the median of the three",
          lay.line_y == 581, f"line_y={lay.line_y}")
    check("the offsets are signed and sum sensibly",
          sorted(v.line_nudge_px for v in lay.vials) == [-23, 0, 9],
          f"{sorted(v.line_nudge_px for v in lay.vials)}")

    # an even count puts the median between two rows -- the round trip must survive it
    lay2 = _layout([("R1", 100, 200, 900), ("R2", 400, 200, 900)])
    rows2 = {"R1": 542, "R2": 538}                # the real paraquat pair
    LAY.set_vial_lines(lay2, rows2)
    got2 = {v.label: v.line_row(lay2.line_y) for v in lay2.vials}
    check("an even number of vials round-trips exactly (median falls between rows)",
          got2 == rows2, f"line_y={lay2.line_y}, {got2}")

    # brute force: the arithmetic must hold for arbitrary rows, not just tidy ones
    rng = np.random.default_rng(20260819)
    bad = 0
    for _ in range(2000):
        n = int(rng.integers(1, 7))
        labs = [f"R{i}" for i in range(1, n + 1)]
        want = {l: int(rng.integers(0, 1080)) for l in labs}
        L = _layout([(l, 100 + 130 * i, 0, 1080) for i, l in enumerate(labs)])
        LAY.set_vial_lines(L, want)
        if {v.label: v.line_row(L.line_y) for v in L.vials} != want:
            bad += 1
    check("2000 random layouts round-trip with zero drift", bad == 0,
          f"{bad} mismatches")

    # a label with no confirmed row must keep the shared line, not silently move
    lay3 = _layout([("R1", 100, 200, 900), ("R2", 400, 200, 900)])
    LAY.set_vial_lines(lay3, {"R1": 500})
    check("a vial with no confirmed row keeps the shared line",
          lay3.line_y == 500 and
          [v.line_row(lay3.line_y) for v in lay3.vials] == [500, 500],
          f"{[v.line_row(lay3.line_y) for v in lay3.vials]}")

    check("an empty layout is returned unchanged rather than raising",
          LAY.set_vial_lines(LAY.Layout(), {"R1": 10}).vials == [], "")


def test_B_the_gate_gates() -> None:
    print("\nB. The confidence gate flags what it claims to flag")

    clear = CS.LineFit(row=540, depth=90.0, runner_up_depth=30.0, margin=3.0,
                       row_in_box=340)
    check("a dark, unambiguous line is clear", clear.is_clear(), clear.verdict())

    # the exact failure measured on the paraquat rig: dark enough, but a coin toss
    coin = CS.LineFit(row=940, depth=88.0, runner_up_depth=84.0, margin=1.04,
                      row_in_box=740)
    check("a margin of 1.04 is REFUSED (this is the 400-px-wrong case)",
          not coin.is_clear(), coin.verdict())
    check("...and says so in words the operator can act on",
          "place it by hand" in coin.verdict(), coin.verdict())

    faint = CS.LineFit(row=500, depth=8.0, runner_up_depth=2.0, margin=4.0,
                       row_in_box=300)
    check("a feature too faint to be a drawn line is REFUSED",
          not faint.is_clear(), faint.verdict())
    check("...and is described as faint, not as ambiguous",
          "faint" in faint.verdict(), faint.verdict())

    # the boundary must be inclusive on both sides, or the threshold is not the
    # threshold it is documented to be
    edge = CS.LineFit(row=500, depth=25.0, runner_up_depth=25.0 / 1.3, margin=1.3,
                      row_in_box=300)
    check("the documented thresholds (depth 25, margin 1.3) are inclusive",
          edge.is_clear(), f"depth={edge.depth}, margin={edge.margin}")
    just_under = CS.LineFit(row=500, depth=25.0, runner_up_depth=20.0, margin=1.29,
                            row_in_box=300)
    check("a margin one hundredth below the threshold is refused",
          not just_under.is_clear(), just_under.verdict())


def test_C_confirmation_is_not_a_licence() -> None:
    print("\nC. A confirmed line that cannot be real is still refused")

    lay = _layout([("R1", 100, 200, 900), ("R2", 400, 200, 900)])
    LAY.set_vial_lines(lay, {"R1": 550, "R2": 545})
    check("two sane confirmed lines leave the layout scoreable",
          lay.ok(), f"{lay.problems()}")

    # the operator drags one line off the bottom of its tube
    LAY.set_vial_lines(lay, {"R1": 550, "R2": 1005})
    probs = lay.problems()
    check("a confirmed line below the tube base is refused",
          any("outside the box" in p for p in probs), f"{probs}")
    check("...and the refusal names the vial it belongs to",
          any(p.startswith("R2") for p in probs), f"{probs}")

    LAY.set_vial_lines(lay, {"R1": 30, "R2": 545})
    check("a confirmed line above the tube top is refused",
          any("outside the box" in p and p.startswith("R1") for p in lay.problems()),
          f"{lay.problems()}")

    # the per-vial offset must be checked against that vial's OWN box, not a shared one
    lay2 = _layout([("R1", 100, 100, 400), ("R2", 400, 700, 1000)])
    LAY.set_vial_lines(lay2, {"R1": 250, "R2": 850})
    check("vials at different heights are each judged against their own box",
          lay2.ok(), f"{lay2.problems()}")


def test_D_proposal_on_real_footage() -> None:
    print("\nD. The proposal on real recordings -- reported, not asserted")

    try:
        from modules import climbing_video as CVm
    except Exception as exc:                                   # pragma: no cover
        print(f"  [SKIP] climbing_video unavailable ({exc})")
        return

    any_rig = False
    for rig, pattern in RIGS.items():
        vids = sorted(glob.glob(pattern, recursive=True))[:3]
        if not vids:
            print(f"  [SKIP] {rig}: no recordings on this machine")
            continue
        any_rig = True
        for path in vids:
            try:
                info = CVm.probe_video(path)
                bg = CVm.build_background(path)
                det = CVm.detect_vials(path, background=bg)
                vs = det.get("vials") or []
            except Exception as exc:
                print(f"  [SKIP] {Path(path).name}: {exc!r}")
                continue
            if not vs:
                print(f"  [SKIP] {Path(path).name}: no cylinders located")
                continue
            for i, v in enumerate(vs, 1):
                pad = int(0.08 * v.width)
                box = CS.Box(x0=max(0, v.x0 - pad), y0=max(0, v.y_top - 40),
                             x1=min(info.width, v.x1 + pad),
                             y1=min(info.height, v.y_bottom + 20),
                             label=f"R{i}", n_flies=20)
                fit = CS.find_line(bg, box)
                inside = box.y0 <= fit.row <= box.y1
                print(f"        {rig:9s} {Path(path).name[:20]:20s} R{i}: "
                      f"row {fit.row:4d}  depth {fit.depth:5.1f}  "
                      f"margin {fit.margin:4.2f}  -> {fit.verdict()}")
                # the one thing that must never happen: confident AND impossible
                check(f"{Path(path).name} R{i}: not marked clear while outside the box",
                      inside or not fit.is_clear(),
                      f"row {fit.row} vs box {box.y0}-{box.y1}")

    if not any_rig:
        print("  [SKIP] no real footage on this machine; section D established nothing")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: the operator-confirmable scoring line")
    print("=" * 78)
    test_A_exact_round_trip()
    test_B_the_gate_gates()
    test_C_confirmation_is_not_a_licence()
    test_D_proposal_on_real_footage()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CONFIRMABLE-LINE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
