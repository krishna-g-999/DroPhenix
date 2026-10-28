"""
Validation suite for modules/calibration_card.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
The card is rendered at a KNOWN number of pixels per millimetre — a 300 dpi page
of stated width in mm — so the correct scale is fixed by construction and the
detector can be scored against it exactly, not against another estimate.

A. Exact recovery of the scale from the rendered card.
B. Robustness to what a phone actually does: distance (heavy downscaling),
   rotation, and defocus. A calibration that only works on a flat scan is
   useless at the bench.
C. THE SELF-CHECK. The card's value is not that it measures a scale but that it
   measures it FIVE independent ways, so a bad calibration announces itself.
   A tilted card must raise a warning rather than quietly returning a wrong
   number, and a mis-scaled print must be caught from the implied marker size.
D. Refusal: no card in frame is reported as absent, not guessed at.
E. The detected scale drives the same `Calibration` object the rest of the
   pipeline uses, so heights come out in millimetres.

Run: ./.venv/Scripts/python.exe validation/validate_calibration_card.py
Exit 0 = all pass.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from modules import calibration_card as CC  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def _render(tmp: Path):
    """Render the card and return (image, true px/mm)."""
    CC.make_card(str(tmp / "card"), page="A4", dpi=300)
    img = cv2.imread(str(tmp / "card.png"), cv2.IMREAD_GRAYSCALE)
    return img, img.shape[1] / 210.0          # A4 width is exactly 210 mm


def test_A_exact_scale() -> None:
    print("\nA. The scale is recovered exactly from the rendered card")
    tmp = Path(tempfile.mkdtemp(prefix="card_"))
    try:
        img, true_ppm = _render(tmp)
        d = CC.detect_card(img)
        check("all four markers found", d.found and d.n_markers == 4, f"{d.n_markers}/4")
        err = abs(d.px_per_mm - true_ppm) / true_ppm * 100
        check("px/mm recovered to better than 0.5%", err < 0.5,
              f"detected {d.px_per_mm:.4f} vs true {true_ppm:.4f} ({err:.3f}% error)")
        check("five independent estimates are produced",
              len(d.px_per_mm_estimates) == 5, f"{sorted(d.px_per_mm_estimates)}")
        check("the estimates agree with each other", d.disagreement_pct < 2.0,
              f"spread {d.disagreement_pct:.2f}%")
        check("implied marker size matches the specification",
              abs(d.implied_marker_mm - CC.MARKER_MM) < 1.0,
              f"{d.implied_marker_mm:.2f} mm vs {CC.MARKER_MM:.0f} mm specified")
        check("a clean card raises no warnings", not d.warnings, str(d.warnings)[:80])
        check("both PDF and PNG are written",
              (tmp / "card.pdf").exists() and (tmp / "card.png").exists(), "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_B_realistic_conditions() -> None:
    print("\nB. Survives what a phone camera actually does")
    tmp = Path(tempfile.mkdtemp(prefix="card_"))
    try:
        img, true_ppm = _render(tmp)

        for f in (0.25, 0.10, 0.05):
            small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            d = CC.detect_card(small)
            err = (abs(d.px_per_mm - true_ppm * f) / (true_ppm * f) * 100
                   if d.found else float("inf"))
            check(f"card only {small.shape[1]} px wide still calibrates", err < 2.0,
                  f"{d.n_markers}/4 markers, {err:.2f}% error")

        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 12, 1.0)
        rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
        rot = cv2.resize(rot, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
        d = CC.detect_card(rot)
        err = abs(d.px_per_mm - true_ppm * 0.15) / (true_ppm * 0.15) * 100
        check("a rotated card calibrates correctly", d.found and err < 1.0,
              f"12 degrees, {err:.2f}% error")

        blur = cv2.GaussianBlur(cv2.resize(img, None, fx=0.15, fy=0.15), (5, 5), 0)
        d = CC.detect_card(blur)
        err = abs(d.px_per_mm - true_ppm * 0.15) / (true_ppm * 0.15) * 100
        check("an out-of-focus card calibrates correctly", d.found and err < 1.0,
              f"{err:.2f}% error")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_C_self_check() -> None:
    print("\nC. The self-check: a bad calibration announces itself")
    tmp = Path(tempfile.mkdtemp(prefix="card_"))
    try:
        img, true_ppm = _render(tmp)
        H, W = img.shape

        # Card tilted away from the camera. The scale is then WRONG, and the
        # point of four markers is that the wrongness is visible.
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[0, 0], [W, 0], [W * 0.86, H], [W * 0.14, H]])
        tilt = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                                   (W, H), borderValue=255)
        d = CC.detect_card(cv2.resize(tilt, None, fx=0.15, fy=0.15,
                                      interpolation=cv2.INTER_AREA))
        check("a tilted card is detected as inconsistent", d.disagreement_pct > 5.0,
              f"estimates disagree by {d.disagreement_pct:.1f}%")
        check("and the user is told it is tilted",
              any("tilted" in w for w in d.warnings),
              d.warnings[0][:90] if d.warnings else "no warning raised")

        # A page printed at 94% -- the classic "fit to page" error. Every
        # distance would be 6% wrong, silently, and the only clue is that the
        # markers no longer measure what the card says they do.
        shrunk = cv2.resize(img, None, fx=0.94 * 0.15, fy=0.94 * 0.15,
                            interpolation=cv2.INTER_AREA)
        d2 = CC.detect_card(shrunk)
        implied_at_true_scale = d2.px_per_mm / (true_ppm * 0.15)
        check("a mis-scaled print shifts the scale as expected",
              abs(implied_at_true_scale - 0.94) < 0.02,
              f"detected scale is {implied_at_true_scale:.3f}x the correct one "
              f"(printed at 94%) -- which is why the 100 mm rule must be checked "
              f"with a ruler")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_D_refusal() -> None:
    print("\nD. No card means no calibration, not a guess")
    blank = np.full((600, 800), 235, np.uint8)
    d = CC.detect_card(blank)
    check("a blank frame yields no detection",
          not d.found and not np.isfinite(d.px_per_mm), "")
    check("and says why", bool(d.warnings), d.warnings[0][:70] if d.warnings else "")

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (600, 800), dtype=np.uint8)
    d2 = CC.detect_card(noise)
    check("random noise does not produce a spurious calibration",
          not d2.found or not np.isfinite(d2.px_per_mm),
          f"found={d2.found}")


def test_E_feeds_the_pipeline() -> None:
    print("\nE. The card's scale drives the same Calibration the pipeline uses")
    from modules.climbing_video import Calibration, VialSpec, ScoringRule, score_heights

    tmp = Path(tempfile.mkdtemp(prefix="card_"))
    try:
        img, true_ppm = _render(tmp)
        d = CC.detect_card(img)
        cal = CC._as_calibration(d.px_per_mm)
        check("a Calibration object is produced",
              isinstance(cal, Calibration) and cal.source == "printed calibration card",
              cal.source)

        v = VialSpec(x0=0, x1=int(round(25 * d.px_per_mm)), y_top=0,
                     y_bottom=int(round(150 * d.px_per_mm)), line_y=0, label="R1")
        ppm = cal.resolve(v)
        check("resolving the calibration returns the card's scale",
              abs(ppm - d.px_per_mm) < 1e-9, f"{ppm:.4f} px/mm")

        # a fly exactly 80 mm above the base must be scored by an 80 mm rule
        heights = np.array([79.0, 81.0]) * ppm
        s = score_heights(heights, v, ScoringRule("height_mm", height_mm=80.0), ppm)
        check("an 80 mm criterion separates flies at 79 and 81 mm",
              s["n_above"] == 1, f"{s['n_above']} of 2 above, cutoff "
                                 f"{s['cutoff_mm']:.1f} mm")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/calibration_card.py")
    print("=" * 78)
    test_A_exact_scale()
    test_B_realistic_conditions()
    test_C_self_check()
    test_D_refusal()
    test_E_feeds_the_pipeline()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CALIBRATION-CARD VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
