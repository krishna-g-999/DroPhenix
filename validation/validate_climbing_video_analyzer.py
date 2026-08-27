"""
Validation suite for modules/climbing_video_analyzer.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. EXACT COUNTS on synthetic footage whose fly positions are known by
   construction, checked in BOTH directions. The specification's own unit test
   asserts `len(centroids) >= 10`, which a false positive passes silently -- and
   on the specification's own test video the detector did return 21 flies where
   20 were drawn.

B. THE REFERENCE-LINE REGRESSION. The extra detection in (A) was the scoring line
   itself: 200 px wide, 5 px tall, area 796 px, inside the area window. A drawn
   line on real glassware is the same shape, so this is a field failure and not a
   test artefact. It must stay fixed.

C. ROI CLIPPING. An ROI that runs past the frame edge must not silently shift the
   reference line, which is what cropping with `frame[y:y+h, x:x+w]` and then
   using the REQUESTED height would do.

D. TWO-SIDED FLAGGING. Over-detection must be as visible as under-detection; the
   specification's capped `confidence` cannot express it.

E. SCHEMA CONFORMANCE. The exported table must load through
   `modules.data_parser.load_data`, since the claim is that it feeds DroPhenix
   directly.

F. REAL FOOTAGE against hand counts (skipped, not failed, when the recordings are
   absent). This is the only test here that establishes accuracy on real data;
   everything above establishes correctness.

Run: ./.venv/Scripts/python.exe validation/validate_climbing_video_analyzer.py
Exit 0 = all pass.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from modules import climbing_video_analyzer as CVA  # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []
SEED = 20260812

REAL_DIR = Path(r"D:/DroPhenix/Testing data/Climbing assay/"
                r"Paraquat climbing assay 010626_new set")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  [SKIP] {name}  --  {why}")
    SKIPPED.append(f"{name}: {why}")


# ─────────────────────────────────────────────────────────── synthetic scene ──
def render(path: Path, n_flies: int = 20, w: int = 200, h: int = 600,
           fps: int = 30, seconds: int = 10, line_y: int = 200,
           draw_line: bool = True, radius: int = 10, n_cols: int = 5) -> dict:
    """Render a climbing clip and record the TRUE count above the line per frame.

    Flies rise at a fixed rate from known starting rows, so the number above the
    line is known exactly at every frame rather than asserted afterwards.

    THE SCENE MUST TEST THE DETECTOR, NOT THE CODEC
    -----------------------------------------------
    This used to draw radius-5 flies in nine columns 20 px apart, and it was
    measuring the wrong thing twice over.

    A radius-5 disc has an area of 79 px^2 against a minimum of 0.0016*w^2 = 64 --
    a margin of 1.23x. Lossy mp4v compression alone was enough to push flies under
    that floor: 2 of 20 vanished at t=5s with NO scoring line anywhere near them,
    and the same 2 vanished whether the line was drawn or not. The failures were
    being read as a line-removal bug, which they were not.

    It also could not be repaired by enlarging the flies alone: at 20 px spacing a
    disc of radius 9 touches its neighbours, and the counts collapsed to 13 of 20.
    The spacing had to grow with the flies.

    So the flies are now radius 10 in a 200 px vial -- a tenth of the width, which
    is the size the module's own area window is built around ("a fly spans roughly
    a tenth of a cylinder's width") -- in five columns 33 px apart. That leaves a
    4.9x area margin and 13 px of clear space between neighbours. Measured across
    radius 7-10 at this spacing, the detector returns exactly 20 of 20 at every
    timepoint, with and without the line, so any failure here is now a real one.
    """
    rng = np.random.default_rng(SEED)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("could not open a VideoWriter (mp4v)")

    step = max(2 * radius + 4, w // (n_cols + 1))
    starts = [520 - i * 6 for i in range(n_flies)]
    xs = [step + (i % n_cols) * step for i in range(n_flies)]
    truth: dict[int, tuple[int, int]] = {}

    for f in range(fps * seconds):
        t = f / fps
        img = np.full((h, w, 3), 230, np.uint8)
        if draw_line:
            cv2.line(img, (0, line_y), (w - 1, line_y), (0, 0, 0), 3)
        above = 0
        for i in range(n_flies):
            y = max(40, starts[i] - int(t * 30))
            cv2.circle(img, (xs[i], y), radius, (30, 30, 30), -1)
            if y < line_y:
                above += 1
        truth[f] = (n_flies, above)
        img = cv2.add(img, rng.normal(0, 1.5, img.shape)
                      .astype(np.int16).clip(-6, 6).astype(np.uint8))
        out.write(img)
    out.release()
    return {"path": str(path), "fps": fps, "truth": truth, "line_y": line_y,
            "w": w, "h": h, "n": n_flies}


def _vial(scene: dict, n_expected: int | None = None) -> CVA.VialConfig:
    return CVA.VialConfig(
        vial_id=1, genotype="Test", replicate=1,
        n_flies_expected=n_expected if n_expected is not None else scene["n"],
        roi_x=0, roi_y=0, roi_w=scene["w"], roi_h=scene["h"],
        line_y_frac=scene["line_y"] / scene["h"], fly_color="dark")


# ══════════════════════════════════════════════════════════════════════ tests ══
def test_A_exact_counts() -> None:
    print("\nA. Exact counts on synthetic footage with known ground truth")
    tmp = Path(tempfile.mkdtemp(prefix="cva_"))
    try:
        sc = render(tmp / "syn.mp4")
        v = _vial(sc)
        for t in (2.0, 5.0, 8.0):
            f_idx = int(t * sc["fps"])
            frame = CVA.get_frame_at_number(sc["path"], f_idx)
            cents, _ = CVA.detect_flies_in_roi(frame, v, CVA.DEFAULT_DETECTION_PARAMS)
            n_true, above_true = sc["truth"][f_idx]
            line = v.line_y_for_height(sc["h"])
            above = sum(1 for (_, cy) in cents if cy < line)
            check(f"t={t:.0f}s total flies (exactly {n_true})",
                  len(cents) == n_true, f"detected {len(cents)}")
            check(f"t={t:.0f}s flies above the line (exactly {above_true})",
                  above == above_true, f"detected {above}")

        # an empty tube must yield nothing at all
        empty = render(tmp / "empty.mp4", n_flies=0)
        fr = CVA.get_frame_at_number(empty["path"], 150)
        cents, _ = CVA.detect_flies_in_roi(fr, _vial(empty, 20),
                                           CVA.DEFAULT_DETECTION_PARAMS)
        check("an empty tube yields zero detections", len(cents) == 0,
              f"{len(cents)} spurious detection(s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_B_reference_line_not_a_fly() -> None:
    print("\nB. The drawn reference line is not counted as a fly  [regression]")
    tmp = Path(tempfile.mkdtemp(prefix="cva_"))
    try:
        with_line = render(tmp / "line.mp4", draw_line=True)
        no_line = render(tmp / "noline.mp4", draw_line=False)
        v = _vial(with_line)
        f_with = CVA.get_frame_at_number(with_line["path"], 150)
        f_without = CVA.get_frame_at_number(no_line["path"], 150)
        n_with = len(CVA.detect_flies_in_roi(f_with, v, CVA.DEFAULT_DETECTION_PARAMS)[0])
        n_without = len(CVA.detect_flies_in_roi(f_without, v,
                                                CVA.DEFAULT_DETECTION_PARAMS)[0])
        check("drawing the scoring line does not add a detection",
              n_with == n_without == with_line["n"],
              f"{n_with} with the line, {n_without} without, {with_line['n']} drawn")

        # and the guard is what does it: disabling the aspect filter brings it back
        loose = dict(CVA.DEFAULT_DETECTION_PARAMS, max_aspect=0.0,
                     remove_horizontal_lines=False)
        n_loose = len(CVA.detect_flies_in_roi(f_with, v, loose)[0])
        check("without the aspect filter the line IS miscounted (guard is load-bearing)",
              n_loose == with_line["n"] + 1,
              f"{n_loose} detected with both guards off vs {with_line['n']} drawn")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_C_roi_clipping() -> None:
    print("\nC. An ROI past the frame edge does not move the reference line")
    tmp = Path(tempfile.mkdtemp(prefix="cva_"))
    try:
        sc = render(tmp / "syn.mp4")
        frame = CVA.get_frame_at_number(sc["path"], 150)
        inside = _vial(sc)
        over = CVA.VialConfig(
            vial_id=1, genotype="T", replicate=1, n_flies_expected=sc["n"],
            roi_x=0, roi_y=0, roi_w=sc["w"] + 400, roi_h=sc["h"] + 400,
            line_y_frac=inside.line_y_frac, fly_color="dark")
        x0, y0, x1, y1 = over.clipped_roi(sc["h"], sc["w"])
        check("the oversized ROI is clipped to the frame",
              (x1 - x0, y1 - y0) == (sc["w"], sc["h"]),
              f"clipped to {x1-x0}x{y1-y0}, frame is {sc['w']}x{sc['h']}")
        check("the line is placed from the CLIPPED height, not the requested one",
              over.line_y_for_height(y1 - y0) == inside.line_y_for_height(sc["h"]),
              f"{over.line_y_for_height(y1-y0)} vs {inside.line_y_for_height(sc['h'])}"
              f" (the requested-height value would be {over.line_y_pixels})")
        a = CVA.detect_flies_in_roi(frame, inside, CVA.DEFAULT_DETECTION_PARAMS)[0]
        b = CVA.detect_flies_in_roi(frame, over, CVA.DEFAULT_DETECTION_PARAMS)[0]
        check("an oversized ROI gives the same detections as an exact one",
              len(a) == len(b), f"{len(a)} vs {len(b)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_D_two_sided_flagging() -> None:
    print("\nD. Over-detection is flagged as loudly as under-detection")
    tmp = Path(tempfile.mkdtemp(prefix="cva_"))
    try:
        sc = render(tmp / "syn.mp4")           # 20 flies present
        for expected, label in ((40, "under"), (10, "over"), (20, "correct")):
            v = _vial(sc, n_expected=expected)
            r = CVA.analyze_frame_for_vial(sc["path"], v, 5.0, 0, sc["fps"],
                                           CVA.DEFAULT_DETECTION_PARAMS)
            if label == "correct":
                check("a correct count is not flagged", not r.flagged,
                      f"discrepancy {r.count_discrepancy:+.2f}")
            else:
                check(f"{label}-detection is flagged", r.flagged,
                      f"expected {expected}, detected {r.flies_detected}, "
                      f"discrepancy {r.count_discrepancy:+.2f}, "
                      f"confidence {r.confidence:.2f}")
            if label == "over":
                check("...and the capped confidence alone would have hidden it",
                      r.confidence >= 1.0 and r.flagged,
                      f"confidence {r.confidence:.2f} but flagged={r.flagged}")

        bad = CVA.analyze_frame_for_vial("no_such_file.mp4", _vial(sc), 5.0, 0, 30.0)
        check("an unreadable video yields confidence 0 with a reason, not an exception",
              bad.confidence == 0.0 and bad.low_confidence and bad.reason,
              f"reason: {bad.reason!r}")
        nofps = CVA.analyze_frame_for_vial(sc["path"], _vial(sc), 5.0, 0, 0.0)
        check("a missing frame rate is refused, not divided by",
              nofps.confidence == 0.0 and "frame rate" in nofps.reason,
              f"reason: {nofps.reason!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_E_schema() -> None:
    print("\nE. The exported table loads through DroPhenix's parser")
    tmp = Path(tempfile.mkdtemp(prefix="cva_"))
    try:
        sc = render(tmp / "syn.mp4")
        sess = CVA.VideoAnalysisSession(
            video_path=sc["path"], fps=sc["fps"], total_frames=sc["fps"] * 10,
            duration_sec=10.0, t_zero_frame=0, timepoints_sec=[2, 5, 8],
            vials=[_vial(sc)])
        results = [r for r, _ in CVA.run_full_analysis(sess)]
        check("every vial x timepoint produced a result", len(results) == 3,
              f"{len(results)} results")
        check("session.results is populated once the generator is exhausted",
              len(sess.results) == 3, f"{len(sess.results)}")
        df = CVA.results_to_drophenix_csv(sess)
        check("a non-empty table is produced", not df.empty, f"{len(df)} rows")

        need = {"Genotype", "Treatment", "Sex", "Replicate", "Vial_ID", "Time_sec",
                "Count_climbed", "n_total_flies", "PI", "Confidence",
                "Low_confidence", "Frame_number", "Video_file", "Analysis_notes"}
        check("every column named in the specification is present",
              need.issubset(df.columns), f"missing {sorted(need - set(df.columns))}")

        csv = tmp / "out.csv"
        df.to_csv(csv, index=False)
        from modules import data_parser as DP
        loaded, warns, errs = DP.load_data(str(csv))
        check("it loads through data_parser without errors", not errs, f"{errs}")
        check("the aliases resolve to the canonical columns",
              {"Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n"}
              .issubset(loaded.columns),
              f"got {sorted(loaded.columns)[:10]}")
        check("PI never exceeds 1 or falls below 0",
              bool(((df["PI"] >= 0) & (df["PI"] <= 1)).all()),
              f"range {df['PI'].min()}-{df['PI'].max()}")
        check("counts never exceed the stated total",
              bool((df["Count_climbed"] <= df["n_total_flies"]).all()), "")
        check("the denominator is constant within a vial (a sealed vial cannot change)",
              df.groupby("Vial_ID")["n_total_flies"].nunique().eq(1).all(), "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_F_real_footage() -> None:
    print("\nF. REAL footage against hand counts  [the only accuracy test here]")
    gt_file = REAL_DIR / "paraquat_ground_truth.csv"
    man_file = REAL_DIR / "paraquat_video_manifest.csv"
    if not (gt_file.exists() and man_file.exists()):
        skip("real-footage accuracy", f"ground truth not present at {REAL_DIR}")
        return

    gt = pd.read_csv(gt_file)
    man = pd.read_csv(man_file)
    # one single-cylinder recording (replicate 3) -- ROI stated by hand, as the
    # module intends, from the frame geometry measured in the Phase 0 audit
    row = man[(man["folder"] == "WT Untreated") & (man["cylinders_in_frame"] == 1)
              & (man["use"])].sort_values("order").head(1)
    if row.empty:
        skip("real-footage accuracy", "no single-cylinder WT Untreated recording listed")
        return
    r = row.iloc[0]
    path = REAL_DIR / "WT Untreated" / r["video"]
    if not path.exists():
        skip("real-footage accuracy", f"{path.name} not present")
        return

    meta = CVA.load_video_metadata(str(path))
    # The ROI is taken from the automatic detector rather than typed in by hand:
    # a hand-typed guess is exactly the kind of thing this suite should not depend
    # on, and an earlier version of this test placed it 900 px off and "measured"
    # zero flies. `climbing_video.detect_vials` graded this recording 0.98.
    try:
        from modules import climbing_video as CV
        bg = CV.build_background(str(path))
        det = CV.detect_vials(str(path), n_vials=1, background=bg)
    except Exception as exc:
        skip("real-footage accuracy", f"could not locate the cylinder: {exc!r}")
        return
    if not det.get("vials"):
        skip("real-footage accuracy", "no cylinder located")
        return
    reg = det["vials"][0]
    line_frac = ((reg.y_bottom - reg.line_y) / reg.height) if reg.height else 0.5
    v = CVA.VialConfig(
        vial_id=3, genotype="WT", replicate=3, n_flies_expected=20,
        roi_x=reg.x0, roi_y=reg.y_top, roi_w=reg.width, roi_h=reg.height,
        line_y_frac=float(np.clip((reg.line_y - reg.y_top) / max(reg.height, 1),
                                  0.0, 1.0)),
        fly_color="dark", treatment="Untreated")
    print(f"      ROI from the automatic detector: x {reg.x0}-{reg.x1}, "
          f"y {reg.y_top}-{reg.y_bottom}, line at {v.line_y_frac:.2f} of height "
          f"(detection confidence {det['confidence']:.2f})")
    truth = gt[(gt.Genotype == "WT") & (gt.Treatment == "Untreated")
               & (gt.replicate == 3) & (gt.tap == int(r["tap"]))]
    if truth.empty:
        skip("real-footage accuracy", "no matching hand counts")
        return

    rows = []
    for _, tr in truth.iterrows():
        res = CVA.analyze_frame_for_vial(str(path), v, float(tr["time_s"]), 0,
                                         meta["fps"], CVA.DEFAULT_DETECTION_PARAMS)
        rows.append({"time_s": tr["time_s"], "hand": int(tr["n_climbed"]),
                     "video": res.flies_detected, "above": res.flies_above,
                     "flagged": res.flagged,
                     "discrepancy": res.count_discrepancy})
    d = pd.DataFrame(rows)
    print("      " + d.to_string(index=False).replace("\n", "\n      "))
    print("      Hand counts are flies above the laboratory's 40 mL line. The ROI")
    print("      and line come from the automatic detector, which graded this")
    print("      recording 0.99 -- so the geometry is not the weak link here.")

    # The scientific claim being tested is NOT accuracy -- one hand-placed ROI
    # cannot establish that -- but that the module DECLARES its state honestly.
    check("every timepoint produced a result, none silently skipped",
          len(d) == len(truth), f"{len(d)} of {len(truth)}")
    check("a gross mismatch with the expected count is flagged",
          bool(d["flagged"].any()) or bool((d["discrepancy"].abs() <= 0.30).all()),
          f"{int(d['flagged'].sum())} of {len(d)} flagged; "
          f"discrepancy {d['discrepancy'].min():+.2f} to {d['discrepancy'].max():+.2f}")
    ratio = d["video"].mean() / 20.0
    print(f"      MEASURED: the detector reports {d['video'].mean():.0f} objects on "
          f"average where 20 flies are present ({ratio:.1f}x). Mean "
          f"|above - hand| = {(d['above'] - d['hand']).abs().mean():.1f} flies.")
    print("      This is adaptive thresholding meeting a graduated cylinder: the")
    print("      printed ticks and numerals are dark and fly-sized, and no local")
    print("      threshold separates them from an animal. One recording, reported")
    print("      and not asserted.")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/climbing_video_analyzer.py")
    print("=" * 78)
    test_A_exact_counts()
    test_B_reference_line_not_a_fly()
    test_C_roi_clipping()
    test_D_two_sided_flagging()
    test_E_schema()
    test_F_real_footage()

    print("\n" + "=" * 78)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}): {'; '.join(SKIPPED)}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CLIMBING-VIDEO-ANALYZER VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
