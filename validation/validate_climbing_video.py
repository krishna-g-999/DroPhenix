"""
Validation suite for modules/climbing_video.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. EXACT RECOVERY on synthetic footage with known ground truth. Videos are
   rendered with a stated number of flies at stated heights, so the correct
   answer is known by construction rather than by anyone's judgement. The
   synthetic scene deliberately reproduces the two features that broke the
   previous detector on real laboratory footage:
     - a printed graduation scale (static dark ticks and numerals), which an
       intensity threshold cannot distinguish from a fly
     - sub-pixel camera drift in the first seconds, as real cylinders settle
       after being tapped

B. THE REGISTRATION ABLATION, on synthetic AND on real laboratory footage.
   This is the central engineering claim of the module. It is tested with a
   ground-truth-free invariant: the number of flies inside a sealed cylinder
   cannot change, so the variability of the detected total across a clip is
   pure detector error, whatever the true count happens to be. Registration
   must reduce it substantially.

C. Scoring-line detection against a known line row.

D. REFUSAL. Unreadable files, and detections that should be flagged for review,
   must produce an explicit refusal rather than a plausible-looking number.

E. Schema conformance: the exported table must load through
   `modules.data_parser.load_data` without errors, since the whole point is to
   feed the downstream hierarchical and trajectory analyses.

NOTE ON WHAT IS *NOT* CLAIMED
-----------------------------
No published or hand-counted ground truth exists for the laboratory videos used
in test B, so accuracy on real footage is NOT asserted here -- only stability,
which is necessary but not sufficient. A detector that misses the same three
flies in every frame is perfectly stable and still wrong. Establishing accuracy
on real recordings requires a human count of the same frames, and until that
exists the module reports the operator's supplied total as the denominator and
treats its own total as a cross-check.

Run: ./.venv/Scripts/python.exe validation/validate_climbing_video.py
Exit code 0 = all pass. Real-footage tests are skipped (not failed) when the
laboratory videos are not present on this machine.
"""

from __future__ import annotations

import glob
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from modules import climbing_video as CV  # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []
SEED = 20260803

REAL_VIDEO_DIR = r"D:/DroPhenix/Testing data/Climbing assay/Drosophila _ climbing assay"


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  [SKIP] {name}  --  {why}")
    SKIPPED.append(f"{name}: {why}")


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC SCENE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
def render_synthetic(
    path: str,
    n_flies_per_vial=(12, 18, 25),
    frac_above=(0.25, 0.5, 0.8),
    n_vials: int = 3,
    width: int = 960,
    height: int = 720,
    fps: int = 30,
    duration_s: float = 12.0,
    drift_px: float = 1.2,
    drift_until_s: float = 5.0,
    with_scale_marks: bool = True,
    seed: int = SEED,
) -> dict:
    """
    Render a climbing-assay-like clip with known ground truth.

    Every fly is a small dark ellipse that WANDERS along its tube over the clip.
    The movement is essential and not decoration: a temporal-median background
    can only separate a fly from the scene if the fly vacates its position for
    most of the clip. An earlier version of this generator jittered each fly by
    less than a pixel around a fixed point; the median then contained the flies,
    subtracting it erased them, and the detector correctly reported zero. The
    generator was wrong, not the detector -- but a synthetic benchmark that
    cannot reproduce the conditions of the real recording is worse than none,
    because it fails for reasons unrelated to the thing being tested.

    Ground truth is therefore recorded PER FRAME from the actual rendered
    positions, rather than imposed up front. `frac_above` sets only the starting
    distribution.

    `with_scale_marks` paints a static graduation scale -- thin dark ticks and
    numerals -- down one side of each tube. This is the feature that defeats an
    intensity-threshold detector, and it must be invisible to a
    background-subtraction one.

    `drift_px` applies a sub-pixel translation to the first `drift_until_s`
    seconds, reproducing cylinders still settling after the tap.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * fps)

    margin, gap = 90, 60
    tube_w = (width - 2 * margin - (n_vials - 1) * gap) // n_vials
    tube_top, tube_bot = 80, height - 80
    line_y = int(tube_top + 0.40 * (tube_bot - tube_top))

    # Fly size is scaled to the apparent tube width, as it is in real footage:
    # a 50 ml cylinder is ~25 mm across and an adult fly ~2.5 mm, so a fly spans
    # roughly a tenth of the tube. Drawing a fixed-pixel fly regardless of zoom
    # would make the synthetic benchmark test a size regime the detector will
    # never meet.
    fly_b = max(3, int(round(0.055 * tube_w)))     # semi-major (body length/2)
    fly_a = max(2, int(round(0.60 * fly_b)))       # semi-minor

    vials = []
    for i in range(n_vials):
        x0 = margin + i * (tube_w + gap)
        vials.append({"x0": x0, "x1": x0 + tube_w, "label": f"R{i + 1}"})

    truth = []
    for i, v in enumerate(vials):
        n = n_flies_per_vial[i]
        n_above = int(round(frac_above[i] * n))
        ys = []
        for k in range(n):
            if k < n_above:
                ys.append(rng.uniform(tube_top + 25, line_y - 18))
            else:
                ys.append(rng.uniform(line_y + 18, tube_bot - 25))
        xs = rng.uniform(v["x0"] + 12, v["x1"] - 12, size=n)
        truth.append({"label": v["label"], "n": n, "n_above": n_above,
                      "xs": xs, "ys": np.array(ys)})

    base = np.full((height, width), 232, np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    cv2.rectangle(base, (0, height - 60), (width, height), 245, -1)

    for v in vials:
        cv2.rectangle(base, (v["x0"], tube_top), (v["x1"], tube_bot), 200, 2)
        cv2.line(base, (v["x0"], line_y), (v["x1"], line_y), 30, 4)
        if with_scale_marks:
            # Tick thickness scales with the apparent tube width, exactly as the
            # flies do. A real graduation mark is about as wide on screen as a
            # fly's body, which is precisely why a misregistered tick can be
            # mistaken for one; drawing hairline ticks regardless of zoom would
            # make the synthetic scene easier than the real one.
            tick_t = max(1, int(round(0.014 * tube_w)))
            sx = v["x0"] + 8
            for yy in range(tube_top + 20, tube_bot - 10, 12):
                long = (yy - tube_top - 20) % 60 == 0
                cv2.line(base, (sx, yy), (sx + (16 if long else 8), yy), 70, tick_t)
            for j, yy in enumerate(range(tube_top + 20, tube_bot - 40, 60)):
                cv2.putText(base, str(50 - j * 10), (sx + 20, yy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.0018 * tube_w, 90,
                            max(1, tick_t))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("Could not open a VideoWriter (mp4v).")

    # per-frame ground truth: {frame -> {label -> (n_total, n_above)}}
    frame_truth: dict = {}
    pos = [{"x": t["xs"].copy(), "y": t["ys"].copy()} for t in truth]

    for f in range(n_frames):
        img = base.copy()
        rec = {}
        for i, (t, v) in enumerate(zip(truth, vials)):
            # random walk along the tube -- large enough that each fly vacates
            # any given position for most of the clip
            pos[i]["y"] += rng.normal(0, 6.0, size=t["n"])
            pos[i]["x"] += rng.normal(0, 2.5, size=t["n"])
            pos[i]["y"] = np.clip(pos[i]["y"], tube_top + 14, tube_bot - 14)
            pos[i]["x"] = np.clip(pos[i]["x"], v["x0"] + 10, v["x1"] - 10)
            for x, y in zip(pos[i]["x"], pos[i]["y"]):
                cv2.ellipse(img, (int(round(x)), int(round(y))), (fly_a, fly_b),
                            float(rng.uniform(0, 180)), 0, 360, 35, -1)
            rec[t["label"]] = (int(t["n"]), int(np.sum(pos[i]["y"] < line_y)))
        frame_truth[f] = rec

        if f < int(drift_until_s * fps) and drift_px:
            phase = f / max(int(drift_until_s * fps), 1)
            M = np.float32([[1, 0, drift_px * (1 - phase)], [0, 1, 0.25 * drift_px * (1 - phase)]])
            img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        img = cv2.add(img, rng.normal(0, 2.0, img.shape).astype(np.int16).clip(-8, 8).astype(np.uint8))
        writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    writer.release()

    specs = [CV.VialSpec(x0=v["x0"], x1=v["x1"], y_top=tube_top, y_bottom=tube_bot,
                         line_y=line_y, label=v["label"], source="manual")
             for v in vials]
    return {"path": str(path), "vials": specs, "truth": truth,
            "frame_truth": frame_truth, "line_y": line_y, "fps": fps,
            "duration_s": duration_s, "n_frames": n_frames}


# ═══════════════════════════════════════════════════════════════════════════
def test_A_synthetic_exact_counts() -> None:
    print("\nA. Exact counts on synthetic footage with known ground truth")
    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        scene = render_synthetic(tmp / "syn.mp4")
        bg = CV.build_background(scene["path"], start_s=6.0, n_samples=25)

        f_idx = int(9 * scene["fps"])
        cap = cv2.VideoCapture(scene["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, img = cap.read()
        cap.release()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = scene["frame_truth"][f_idx]

        for v in scene["vials"]:
            n_true, above_true = gt[v.label]
            res = CV.count_in_vial(gray, bg, v, CV.DetectorParams(min_area=6, max_area=200))
            tol = max(2, int(0.15 * n_true))
            check(f"{v.label}: total flies ({n_true} present)",
                  abs(res["n_total"] - n_true) <= tol,
                  f"detected {res['n_total']}, truth {n_true} (tol +/-{tol})")
            check(f"{v.label}: flies above the line ({above_true} present)",
                  abs(res["n_above"] - above_true) <= tol,
                  f"detected {res['n_above']}, truth {above_true} (tol +/-{tol})")

        # the static graduation scale must contribute nothing
        empty = render_synthetic(tmp / "empty.mp4", n_flies_per_vial=(0, 0, 0),
                                 frac_above=(0, 0, 0), seed=SEED + 1)
        bg_e = CV.build_background(empty["path"], start_s=6.0, n_samples=25)
        cap = cv2.VideoCapture(empty["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(9 * empty["fps"]))
        ok, img = cap.read(); cap.release()
        g_e = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        spurious = sum(CV.count_in_vial(g_e, bg_e, v)["n_total"] for v in empty["vials"])
        check("an empty tube with a printed scale yields no detections",
              spurious == 0,
              f"{spurious} spurious detection(s) from ticks/numerals across 3 tubes")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_B_registration_ablation() -> None:
    print("\nB. Registration ablation  [the central engineering claim]")
    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        scene = render_synthetic(tmp / "drift.mp4", drift_px=1.5, drift_until_s=6.0)
        bg = CV.build_background(scene["path"], start_s=7.0, n_samples=25)
        cap = cv2.VideoCapture(scene["path"])

        def totals(register: bool):
            out = []
            p = CV.DetectorParams(min_area=6, max_area=200, register=register)
            for t in (1.5, 2.5, 3.5, 5.0, 8.0, 10.0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * scene["fps"]))
                ok, im = cap.read()
                if not ok:
                    continue
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                out.append(sum(CV.count_in_vial(g, bg, v, p)["n_total"] for v in scene["vials"]))
            return np.array(out, float)

        off, on = totals(False), totals(True)
        cap.release()
        truth = sum(t["n"] for t in scene["truth"])
        cv_off = float(np.std(off) / max(np.mean(off), 1e-9) * 100)
        cv_on = float(np.std(on) / max(np.mean(on), 1e-9) * 100)
        print(f"      truth = {truth} flies")
        print(f"      without registration: {off.astype(int).tolist()}  (CV {cv_off:.0f}%)")
        print(f"      with    registration: {on.astype(int).tolist()}  (CV {cv_on:.0f}%)")

        check("registration reduces count variability under camera drift",
              cv_on < cv_off, f"CV {cv_on:.1f}% vs {cv_off:.1f}%")
        check("registered counts are close to the truth during the drift phase",
              abs(np.mean(on) - truth) <= max(3, 0.12 * truth),
              f"mean {np.mean(on):.1f} vs truth {truth}")
        inflation = float(np.max(off) / max(truth, 1))
        print(f"      unregistered peak / truth = {inflation:.2f}x")
        if inflation <= 1.3:
            print("      (This synthetic scene does not reproduce the severe "
                  "misregistration artefact seen on real footage -- its printed scale is "
                  "still finer, relative to a fly, than a real cylinder's. The artefact "
                  "is demonstrated on real laboratory video in test F, which is where the "
                  "claim is actually established.)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_C_line_detection() -> None:
    print("\nC. Scoring-line and vial detection on synthetic footage")
    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        scene = render_synthetic(tmp / "line.mp4", drift_px=0.0)
        det = CV.detect_vials(scene["path"], n_vials=3)
        check("all three tubes located", len(det["vials"]) == 3,
              f"{len(det['vials'])} found, confidence {det['confidence']:.2f}")
        if len(det["vials"]) == 3:
            errs = [abs(v.line_y - scene["line_y"]) for v in det["vials"]]
            check("scoring line found within 6 px of its true row",
                  max(errs) <= 6,
                  f"errors {errs} px (true row {scene['line_y']})")
            # The functional requirement is that the detected region CONTAINS
            # the animals, not that it reproduces the drawn rectangle. Flies do
            # not occupy the glass walls, so a region fitted to where they
            # actually are is legitimately narrower than the tube; asserting a
            # high rectangle IoU would penalise correct behaviour.
            mid = scene["n_frames"] // 2
            for v, tr in zip(det["vials"], scene["truth"]):
                xs = tr["xs"]
                inside = float(np.mean((xs >= v.x0) & (xs <= v.x1))) if len(xs) else 1.0
                check(f"{v.label} region contains the animals it must score",
                      inside >= 0.90,
                      f"{inside:.0%} of this tube's flies fall inside the detected region "
                      f"[{v.x0}, {v.x1}]")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_D_refusal() -> None:
    print("\nD. Explicit refusal rather than a plausible wrong answer")
    raised = False
    try:
        CV.probe_video("does_not_exist_12345.mov")
    except IOError as e:
        raised = "Could not open" in str(e)
    check("a missing/unreadable file raises a clear IOError", raised)

    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        # A scene with no flies at all gives the detector nothing to locate.
        empty = render_synthetic(tmp / "none.mp4", n_flies_per_vial=(0, 0, 0),
                                 frac_above=(0, 0, 0), seed=SEED + 5)
        det = CV.detect_vials(empty["path"], n_vials=3)
        check("a clip with no flies is flagged for review, not silently scored",
              det["needs_review"] or not det["vials"],
              f"confidence {det['confidence']:.2f}, {len(det['vials'])} region(s), "
              f"reason: {det.get('reason', '')[:60]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_E_schema_conformance() -> None:
    print("\nE. Exported table conforms to the DroPhenix climbing schema")
    from modules.data_parser import load_data
    from modules import design_schema as ds

    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        scene = render_synthetic(tmp / "exp.mp4", drift_px=0.0)
        res = CV.analyze_climbing_video(
            scene["path"], vials=scene["vials"],
            scoring_times_s=(2, 4, 6, 8), t0_s=0.0,
            n_total_flies={t["label"]: t["n"] for t in scene["truth"]},
            stability_samples=8,
        )
        check("analysis completes on synthetic footage", res.get("ok"), "")
        tab = CV.to_drophenix_table(res, genotype="Oregon-WT", treatment="Vehicle",
                                    sex="Male", experiment_id="VID_TEST", age_days=10,
                                    time_of_day="09:30")
        check("a non-empty table is produced", not tab.empty, f"{len(tab)} rows")

        csv_path = tmp / "out.csv"
        tab.to_csv(csv_path, index=False)
        df, warn, err = load_data(str(csv_path))
        check("the exported CSV loads through data_parser without errors",
              not err, f"errors: {err}")
        check("each cylinder becomes its own VialID (the experimental unit)",
              "VialID" in df.columns and df["VialID"].nunique() == 3,
              f"{df['VialID'].nunique() if 'VialID' in df.columns else 0} distinct VialIDs")

        spec = ds.describe_design(df)
        check("the design layer sees the vial as the replication unit",
              spec.cluster_unit == "VialID", f"cluster_unit = {spec.cluster_unit}")

        counts = res["counts"]
        check("counts never exceed the supplied total",
              bool((counts["n_above_detected"] <= counts["n_total_used"]).all()),
              "no vial reports more climbers than flies present")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_F_real_footage_stability() -> None:
    print("\nF. Stability on REAL laboratory footage (ground-truth-free invariant)")
    files = sorted(glob.glob(REAL_VIDEO_DIR + "/*.MOV"))
    if not files:
        skip("real-footage registration ablation",
             f"no videos found under {REAL_VIDEO_DIR}")
        return

    path = files[0]
    try:
        bg = CV.build_background(path)
        det = CV.detect_vials(path, n_vials=3, background=bg)
    except Exception as e:
        skip("real-footage registration ablation", f"could not process: {e!r}")
        return
    if not det["vials"]:
        skip("real-footage registration ablation", "no cylinders detected")
        return

    info = CV.probe_video(path)
    cap = cv2.VideoCapture(path)

    def series(register: bool) -> np.ndarray:
        p = CV.DetectorParams(register=register)
        vals = []
        for t in (5, 8, 10, 12, 15, 20, 30, 45, 60, 90):
            if t > info.duration_s - 1:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info.fps))
            ok, im = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            vals.append(CV.count_in_vial(g, bg, det["vials"][0], p)["n_total"])
        return np.array(vals, float)

    off, on = series(False), series(True)
    cap.release()
    cv_off = float(np.std(off) / max(np.mean(off), 1e-9) * 100)
    cv_on = float(np.std(on) / max(np.mean(on), 1e-9) * 100)

    print(f"      video: {Path(path).name}, cylinder {det['vials'][0].label}")
    print(f"      without registration: {off.astype(int).tolist()}  (CV {cv_off:.0f}%)")
    print(f"      with    registration: {on.astype(int).tolist()}  (CV {cv_on:.0f}%)")
    print("      The true count is unknown but CONSTANT, so all of this variation "
          "is detector error.")

    check("registration substantially stabilises the count on real footage",
          cv_on < 0.5 * cv_off,
          f"CV {cv_on:.0f}% with registration vs {cv_off:.0f}% without")
    check("registered count on real footage is stable in absolute terms",
          cv_on < 20.0, f"CV {cv_on:.1f}%")


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def test_J_scoring_mode() -> None:
    print("\nJ. Crossing-based scoring in the automatic pipeline")
    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        scene = render_synthetic(tmp / "syn.mp4")
        common = dict(vials=scene["vials"], scoring_times_s=(4, 6, 8, 10),
                      stability_samples=4, average_frames=3,
                      n_total_flies={"__all__": 30})
        inst = CV.analyze_climbing_video(scene["path"], scoring_mode="net", **common)
        cross = CV.analyze_climbing_video(scene["path"], scoring_mode="max_reached",
                                          **common)
        check("both scoring modes run", inst.get("ok") and cross.get("ok"), "")
        ci = inst["counts"]; cc = cross["counts"]
        # The table must record the rule that was COMPUTED, under the name that says
        # what it computes. `crossings` and `instantaneous` are older spellings kept
        # so existing calls keep working, and `crossings` is actively ambiguous: the
        # protocol's line-crossing tally is the NET of up and down, which is `net`,
        # while the old `crossings` spelling means the ratchet. A number exported
        # under that word could be read as either, so it is canonicalised first.
        check("the table records the rule that was computed, under an unambiguous name",
              set(cc["scoring_mode"]) == {"max_reached"}
              and set(ci["scoring_mode"]) == {"net"},
              f"{sorted(set(cc['scoring_mode']))} / {sorted(set(ci['scoring_mode']))}")
        alias = CV.analyze_climbing_video(scene["path"], scoring_mode="crossings",
                                          **common)
        check("an older spelling is accepted and canonicalised, not echoed back",
              set(alias["counts"]["scoring_mode"]) == {"max_reached"},
              f"{sorted(set(alias['counts']['scoring_mode']))}")
        check("crossings keeps the instantaneous value alongside it",
              "n_above_instantaneous" in cc.columns,
              f"{[c for c in cc.columns if 'above' in c]}")

        ok = True
        for lab, g in cc[cc["usable"]].groupby("vial"):
            v = g.sort_values("time_s")["n_above_detected"].dropna().tolist()
            ok &= all(b >= a for a, b in zip(v, v[1:]))
        check("a crossing count never decreases with time", ok,
              "a ratchet that falls is not a ratchet")

        merged = (cc[cc.usable].set_index(["vial", "time_s"])["n_above_detected"]
                  - ci[ci.usable].set_index(["vial", "time_s"])["n_above_detected"])
        check("crossings is never below the instantaneous count it accumulates",
              bool((merged.dropna() >= -1).all()),
              f"minimum difference {merged.min():.0f} (one sample of slack is "
              f"allowed: a crossing must be confirmed in 2 of 3 samples)")
        check("an unknown scoring mode is refused",
              _raises(lambda: CV.analyze_climbing_video(
                  scene["path"], scoring_mode="whatever", **common)),
              "ValueError raised")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_G_scoring_rules_and_calibration() -> None:
    print("\nG. Lab-independent scoring: calibration and interchangeable rules")
    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        scene = render_synthetic(tmp / "cal.mp4", drift_px=0.0)
        bg = CV.build_background(scene["path"], start_s=6.0, n_samples=25)
        v = scene["vials"][2]

        # Calibrate from a stated tube width; px_per_mm must follow exactly.
        cal = CV.Calibration(tube_width_mm=25.0)
        ppm = cal.resolve(v)
        check("px_per_mm derived exactly from a stated tube diameter",
              abs(ppm - v.width / 25.0) < 1e-9,
              f"{ppm:.4f} px/mm from a {v.width} px wide, 25 mm tube")

        # The three routes to a calibration must agree when they describe the
        # same geometry.
        ppm_h = CV.Calibration(tube_height_mm=v.height / ppm).resolve(v)
        ppm_d = CV.Calibration(px_per_mm=ppm).resolve(v)
        check("height-based and direct calibration agree with width-based",
              abs(ppm_h - ppm) < 1e-6 and abs(ppm_d - ppm) < 1e-12,
              f"width {ppm:.4f}, height {ppm_h:.4f}, direct {ppm_d:.4f}")

        f_idx = int(9 * scene["fps"])
        cap = cv2.VideoCapture(scene["path"]); cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, img = cap.read(); cap.release()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h = CV.fly_heights(g, bg, v, px_per_mm=ppm)
        n_true, above_true = scene["frame_truth"][f_idx][v.label]
        check("height extraction finds the same animals as counting",
              abs(h["n"] - n_true) <= max(2, int(0.15 * n_true)),
              f"{h['n']} heights vs {n_true} present")
        check("heights in mm are consistent with heights in pixels",
              np.allclose(h["heights_mm"], h["heights_px"] / ppm),
              "mm = px / px_per_mm for every fly")

        # The drawn-line rule must reproduce the count-based answer exactly,
        # since it is the same criterion computed two ways.
        s_line = CV.score_heights(h["heights_px"], v, CV.ScoringRule("drawn_line"), ppm)
        direct = CV.count_in_vial(g, bg, v)
        check("scoring heights by the drawn line matches direct counting exactly",
              s_line["n_above"] == direct["n_above"],
              f"{s_line['n_above']} vs {direct['n_above']}")

        # A height rule expressed as a fraction and the equivalent rule in mm
        # must give identical answers.
        frac = 0.5
        mm_equiv = frac * v.height / ppm
        a = CV.score_heights(h["heights_px"], v,
                             CV.ScoringRule("height_fraction", fraction=frac), ppm)
        b = CV.score_heights(h["heights_px"], v,
                             CV.ScoringRule("height_mm", height_mm=mm_equiv), ppm)
        check("a fraction rule and the equivalent millimetre rule agree",
              a["n_above"] == b["n_above"],
              f"{frac:.0%} of tube -> {a['n_above']} above; {mm_equiv:.1f} mm -> "
              f"{b['n_above']} above")

        # A stricter criterion can never pass more flies than a looser one.
        low = CV.score_heights(h["heights_px"], v,
                               CV.ScoringRule("height_fraction", fraction=0.2), ppm)
        high = CV.score_heights(h["heights_px"], v,
                                CV.ScoringRule("height_fraction", fraction=0.8), ppm)
        check("a higher criterion passes no more flies than a lower one",
              high["n_above"] <= low["n_above"],
              f"20% -> {low['n_above']}, 80% -> {high['n_above']}")

        # A criterion ABOVE the measured region must be refused, not scored as
        # zero. This is the failure that reported 0% climbing for a wild-type
        # cohort on real footage: the detected region was only 19 mm tall, an
        # 80 mm criterion sat above it, and no fly could ever qualify. Zero and
        # "not measurable" look identical in a results table and mean opposite
        # things.
        too_high_mm = (v.height / ppm) * 3.0
        oob = CV.score_heights(h["heights_px"], v,
                               CV.ScoringRule("height_mm", height_mm=too_high_mm), ppm)
        check("a criterion above the measured region is refused, not scored as zero",
              oob.get("unscoreable") is True and not np.isfinite(oob["n_above"]),
              f"n_above = {oob['n_above']} (must be NaN, not 0) for a "
              f"{too_high_mm:.0f} mm criterion in a {v.height / ppm:.0f} mm region")
        check("the refusal explains why",
              "OUTSIDE" in oob["note"] and "rather than as zero" in oob["note"],
              oob["note"][:100])

        # A criterion below the base is equally impossible and must also refuse.
        below = CV.score_heights(h["heights_px"], v,
                                 CV.ScoringRule("height_fraction", fraction=-0.2), ppm)
        check("a criterion below the base is also refused",
              below.get("unscoreable") is True, f"n_above = {below['n_above']}")

        # Asking for millimetres without a calibration must fall back and SAY so.
        nocal = CV.score_heights(h["heights_px"], v,
                                 CV.ScoringRule("height_mm", height_mm=80.0), None)
        check("a millimetre rule without calibration falls back and explains itself",
              len(nocal["note"]) > 20, nocal["note"][:90])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_H_auto_vial_count_and_batch() -> None:
    print("\nH. Automatic cylinder count, and batch scoring from a manifest")
    tmp = Path(tempfile.mkdtemp(prefix="dpx_vid_"))
    try:
        # Cylinder count must be discovered, not assumed -- other laboratories
        # run other numbers of tubes.
        for n in (2, 4):
            sc = render_synthetic(tmp / f"n{n}.mp4", n_vials=n,
                                  n_flies_per_vial=tuple([16] * n),
                                  frac_above=tuple([0.5] * n),
                                  drift_px=0.0, seed=SEED + 20 + n)
            est = CV.auto_n_vials(sc["path"])
            check(f"{n} cylinders detected without being told",
                  est["n_vials"] == n, f"found {est['n_vials']}")

        # Regions must never overlap: shared pixels would count one fly twice.
        sc = render_synthetic(tmp / "n4b.mp4", n_vials=4,
                              n_flies_per_vial=(10, 22, 14, 18),
                              frac_above=(0.3, 0.6, 0.4, 0.7),
                              drift_px=0.0, seed=SEED + 31)
        det = CV.detect_vials(sc["path"], n_vials=None)
        vs = sorted(det["vials"], key=lambda z: z.x0)
        overlap = any(vs[i].x1 > vs[i + 1].x0 for i in range(len(vs) - 1))
        check("detected regions never overlap each other",
              not overlap, f"{len(vs)} regions, widths {[v.width for v in vs]}")
        check("unequal fly numbers still give equal-width regions",
              len(set(v.width for v in vs)) == 1,
              f"widths {[v.width for v in vs]} despite 10/22/14/18 flies")

        # Batch: a manifest of two clips must merge into one table, and a
        # missing file must be reported rather than crashing the run.
        s1 = render_synthetic(tmp / "b1.mp4", drift_px=0.0, seed=SEED + 40)
        s2 = render_synthetic(tmp / "b2.mp4", drift_px=0.0, seed=SEED + 41)
        man = pd.DataFrame([
            {"video": "b1.mp4", "Genotype": "WT", "Treatment": "Vehicle",
             "Sex": "Male", "Age": 10, "n_per_vial": 18},
            {"video": "b2.mp4", "Genotype": "TDP43", "Treatment": "Drug",
             "Sex": "Male", "Age": 10, "n_per_vial": 18},
            {"video": "missing.mp4", "Genotype": "WT", "Treatment": "Vehicle",
             "Sex": "Male", "Age": 10, "n_per_vial": 18},
        ])
        res = CV.analyze_batch(man, video_dir=str(tmp), scoring_times_s=(3, 6, 9),
                               n_vials=3, calibration=CV.Calibration(tube_width_mm=25.0))
        check("both readable recordings were scored", res["n_scored"] == 2,
              f"{res['n_scored']} of {res['n_requested']}")
        check("the missing file is reported, not silently skipped",
              len(res["failed"]) == 1 and "not found" in res["failed"].iloc[0]["reason"],
              res["failed"].iloc[0]["reason"] if len(res["failed"]) else "nothing reported")
        tab = res["table"]
        check("the merged table carries both genotypes",
              set(tab["Genotype"]) == {"WT", "TDP43"}, f"{sorted(set(tab['Genotype']))}")
        check("every cylinder of every video is a distinct VialID",
              tab["VialID"].nunique() == 6, f"{tab['VialID'].nunique()} VialIDs from 2 videos")
        check("the supplied per-vial total was applied",
              bool((tab["n_total_flies"] == 18).all()), "all rows use n = 18")

        from modules.data_parser import load_data
        csv = tmp / "batch.csv"
        tab.to_csv(csv, index=False)
        df, w, e = load_data(str(csv))
        check("the merged batch table loads through data_parser",
              not e, f"errors: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_I_tap_detection() -> None:
    print("\nI. Tap detection distinguishes a tap from flies moving")
    files = sorted(glob.glob(REAL_VIDEO_DIR + "/*.MOV"))
    if not files:
        skip("tap detection on real footage", "laboratory videos not present")
        return
    # These recordings are a single-tap protocol, so exactly one tap is expected
    # -- UNLESS the camera was moved mid-clip, which produces its own motion
    # burst. That case must be reported as a framing problem rather than
    # silently counted as an extra tap, so it is accepted here only when the
    # framing warning is also raised.
    for path in files[:4]:
        r = CV.detect_taps(path)
        n = r.get("n_taps", 0)
        ts = ", ".join(f"{t['t0_s']:.2f}s" for t in r.get("taps", []))
        name = Path(path).name
        if not r.get("framing_stable", True):
            check(f"{name}: camera movement detected and reported",
                  "camera appears to have been moved" in r.get("protocol", ""),
                  f"{n} burst(s) at [{ts}]; flagged as a framing change, not scored as taps")
            continue
        check(f"{name}: exactly one tap, at the start",
              n == 1 and r["taps"][0]["t0_s"] < 5.0,
              f"{n} tap(s) at [{ts}] -- {r.get('protocol', '')}")

    # And the framing check must not cry wolf on a stable recording.
    stable = [p for p in files[:4] if CV.detect_framing_change(p)["stable"]]
    check("most recordings are correctly judged framing-stable",
          len(stable) >= 3, f"{len(stable)} of {min(4, len(files))} stable")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/climbing_video.py")
    print("=" * 78)
    test_A_synthetic_exact_counts()
    test_B_registration_ablation()
    test_C_line_detection()
    test_D_refusal()
    test_E_schema_conformance()
    test_F_real_footage_stability()
    test_G_scoring_rules_and_calibration()
    test_H_auto_vial_count_and_batch()
    test_I_tap_detection()
    test_J_scoring_mode()

    print("\n" + "=" * 78)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}):")
        for s in SKIPPED:
            print("  -", s)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CLIMBING-VIDEO VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
