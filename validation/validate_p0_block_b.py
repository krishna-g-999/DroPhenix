"""
validate_p0_block_b.py
Regression suite for P0 Block B of the 2026-11 audit: the denominator and the
representation of a missing measurement.

Every check here FAILED before the Block B fixes. See
docs/BUG_AND_RISK_REGISTER.md findings N5, N8, N10, N70, N79 and
docs/PRIORITISED_DEVELOPMENT_PLAN.md section 1 Block B.

What Block B is about
---------------------
B1 N5  The climbing denominator was the DETECTED total, or `max(expected,
       detected)`. A detector that misses flies therefore shrinks the
       denominator by exactly the flies it missed, and because a missed fly in
       a negative-geotaxis assay is overwhelmingly a fly that did not climb,
       every miss moves the vial TOWARD the healthy phenotype. Worked example
       from the register: N_start = 30, 25 detected, 12 above the line, truth
       40.0% -- the detected denominator reports 48.0% and PI -4.0 against a
       true PI of -20.0, a 16-point shift toward normal. `max()` additionally
       hides an overcount, which is the one case where the operator most needs
       to see the number. The loading record's N_start is the only legitimate
       denominator; N_detected is a QC observable, never an adjustment.
B2 N8  Zero detections returned `Pct 0.0, PI 0.0`. But `PI = 0.0` is the PI of
       a vial with half its flies above the line, and `PI = -100.0` is the PI of
       a fully impaired vial, so a detector failure moved a vial +100 PI points
       toward normal. A measurement that did not happen is not a measurement of
       zero. The fix is typed missingness: NOT_MEASURED, MEASUREMENT_FAILED,
       BELOW_THRESHOLD, CENSORED_*, REFUSED -- a NaN says there is no value, not
       why, and only the reason determines whether a row may be excluded,
       imputed, or must stop the analysis.
B3 N79 `auto_session.raw_counts_to_drophenix` cast counts with `int()`, which
       truncates (6.7 -> 6, always downward, and counts are genuinely fractional
       here because the modern detectors take a median over a frame window and
       `climbing_band` applies a merge correction). A missing `n_above` raised
       KeyError and a NaN raised ValueError -- so fixing N8 to emit NaN would
       have crashed this function on the first failed vial. The two must land
       together.
B4 N8  The converters carried neither `QC_flag` nor `Confidence`, so the
       detector's own refusal signal was emitted and then discarded one function
       downstream. A converter must propagate the QC fields or refuse to convert.
B5 N70 Three signature mismatches made the Video Analysis page unusable, which
       is why N5 and N7 were latent there and live through `cv_pipeline`. The
       page is repaired only now, last, so the repair does not switch two
       defects on in one commit.
B6 N10 Sequential replicate numbers were auto-assigned per video file. Two
       camera angles of one vial, or two recordings of one cohort, are not two
       biological replicates; auto-numbering is pseudoreplication by
       construction. An explicit file -> replicate mapping is required.

Run:  python validation/validate_p0_block_b.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ROOT = Path(__file__).resolve().parent.parent

try:
    import cv2
    _CV2 = True
except Exception:                                   # pragma: no cover
    cv2 = None
    _CV2 = False


# ── helpers ───────────────────────────────────────────────────────────────────
def _check(name, got, expected, tol, results):
    """Numeric check. A None or non-finite value is a failure, never a pass."""
    try:
        ok = (got is not None) and np.isfinite(float(got)) \
            and abs(float(got) - expected) <= tol
    except (TypeError, ValueError):
        ok = False
    results.append((name, f"{expected:g}",
                    ("None" if got is None else f"{float(got):g}"
                     if _finite(got) else str(got)), ok))


def _finite(v):
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _bool_check(name, cond, expected, got, results):
    results.append((name, str(expected), str(got), bool(cond)))


def _nan_check(name, got, results):
    """A value that must be NaN -- the whole point of several of these fixes."""
    is_nan = (got is None) or (isinstance(got, float) and np.isnan(got)) or \
             (not isinstance(got, str) and pd.isna(got))
    results.append((name, "NaN", str(got), bool(is_nan)))


def _frame(above_n, below_n, W=300, H=400, r=5):
    """
    A single still vial frame: dark discs on a light vial, the polarity
    `DetectorConfig(mode='dark')` expects. Positions are spaced so morphological
    closing cannot merge two flies into one blob -- merge-splitting is a separate
    finding (N3, Block C) and must not contaminate a denominator test.
    """
    frame = np.full((H, W, 3), 235, np.uint8)
    line = H // 2
    for i in range(above_n):
        x = 30 + (i % 6) * 45
        y = 40 + (i // 6) * 50
        cv2.circle(frame, (x, min(y, line - 20)), r, (25, 25, 25), -1)
    for i in range(below_n):
        x = 30 + (i % 6) * 45
        y = line + 40 + (i // 6) * 50
        cv2.circle(frame, (x, min(y, H - 20)), r, (25, 25, 25), -1)
    return frame


def _detector():
    from modules.fly_detector import FlyDetector, DetectorConfig
    return FlyDetector(DetectorConfig(min_area=20, max_area=400,
                                      threshold_frac=0.5))


# ══════════════════════════════════════════════════════════════════════════════
# B1 -- N_start is the only denominator
# ══════════════════════════════════════════════════════════════════════════════
def _block_b1(results):
    if not _CV2:
        _bool_check("B1 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return
    from modules.fly_detector import VialROI

    det = _detector()

    # The register's worked example, exactly: 30 loaded, 25 detected, 12 above.
    # Truth 40.0%; the detected denominator reports 48.0%.
    frame = _frame(above_n=12, below_n=13)
    v = VialROI(x=0, y=0, w=300, h=400, vial_id=1, expected_n=30)
    vc = det.count_in_vials(frame, [v])[0]
    d = vc.to_dict()

    _check("B1 detector sees the 25 flies that are there", d.get("N_detected"),
           25, 0, results)
    _check("B1 N_start is carried from the loading record", d.get("N_start"),
           30, 0, results)
    _check("B1 Pct uses N_start (40.0, not the detected 48.0)", d.get("Pct"),
           40.0, 0.01, results)
    _check("B1 PI uses N_start (-20.0, not the detected -4.0)", d.get("PI"),
           -20.0, 0.01, results)
    _bool_check("B1 the count shortfall is flagged, never adjusted away",
                "mismatch" in str(d.get("QC_flag", "")),
                "mismatch flag", d.get("QC_flag"), results)

    # An OVERCOUNT must stay visible. `max(expected, detected)` used to raise the
    # denominator to the detected total, so a 12-blob frame in a 10-fly vial read
    # exactly 100% -- the one number that looks perfect while the ROI is wrong.
    frame_over = _frame(above_n=12, below_n=0)
    v_over = VialROI(x=0, y=0, w=300, h=400, vial_id=2, expected_n=10)
    d_over = det.count_in_vials(frame_over, [v_over])[0].to_dict()
    _check("B1 an overcount is reported as >100%, not capped at 100",
           d_over.get("Pct"), 120.0, 0.01, results)
    _bool_check("B1 the overcount is flagged",
                "mismatch" in str(d_over.get("QC_flag", "")),
                "mismatch flag", d_over.get("QC_flag"), results)

    # No loading record -> no denominator. The counts are real observations and
    # are kept; the RATE is refused, because inventing a denominator is the
    # defect, not the fix.
    v_none = VialROI(x=0, y=0, w=300, h=400, vial_id=3, expected_n=0)
    d_none = det.count_in_vials(frame, [v_none])[0].to_dict()
    _check("B1 counts are preserved when N_start is absent",
           d_none.get("N_detected"), 25, 0, results)
    _nan_check("B1 Pct is refused, not guessed, when N_start is absent",
               d_none.get("Pct"), results)
    _bool_check("B1 the refusal is typed REFUSED and names the loading record",
                str(d_none.get("Missingness")) == "REFUSED"
                and "N_start" in str(d_none.get("unusable_reason", "")),
                "REFUSED + reason", f"{d_none.get('Missingness')}",
                results)


# ══════════════════════════════════════════════════════════════════════════════
# B2 -- typed missingness
# ══════════════════════════════════════════════════════════════════════════════
def _block_b2(results):
    try:
        from modules import missingness as MS
    except Exception as exc:
        _bool_check("B2 the typed-missingness vocabulary exists",
                    False, "modules.missingness", type(exc).__name__, results)
        return

    needed = ["OBSERVED", "NOT_MEASURED", "MEASUREMENT_FAILED",
              "BELOW_THRESHOLD", "CENSORED_RIGHT", "REFUSED"]
    have = [n for n in needed if isinstance(getattr(MS, n, None), str)]
    _bool_check("B2 every required missingness code is defined",
                len(have) == len(needed), f"{len(needed)} codes",
                f"{len(have)} codes", results)
    _bool_check("B2 OBSERVED is distinguishable from every failure code",
                MS.OBSERVED not in {MS.NOT_MEASURED, MS.MEASUREMENT_FAILED,
                                    MS.BELOW_THRESHOLD, MS.REFUSED},
                "distinct", "distinct" if MS.OBSERVED not in
                {MS.NOT_MEASURED, MS.MEASUREMENT_FAILED} else "collides",
                results)

    if not _CV2:
        _bool_check("B2 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return
    from modules.fly_detector import VialROI

    det = _detector()
    blank = np.full((400, 300, 3), 235, np.uint8)      # nothing to detect
    v = VialROI(x=0, y=0, w=300, h=400, vial_id=1, expected_n=20)
    d = det.count_in_vials(blank, [v])[0].to_dict()

    _check("B2 a blank frame detects nothing", d.get("N_detected"), 0, 0, results)
    _nan_check("B2 zero detections give Pct NaN, never 0.0", d.get("Pct"), results)
    _nan_check("B2 zero detections give PI NaN, never 0.0 (= half climbed)",
               d.get("PI"), results)
    _bool_check("B2 zero detections are typed MEASUREMENT_FAILED",
                str(d.get("Missingness")) == MS.MEASUREMENT_FAILED,
                MS.MEASUREMENT_FAILED, d.get("Missingness"), results)
    _check("B2 the detector's own confidence is still 0", d.get("Confidence"),
           0.0, 1e-9, results)

    # The subtle half of N8, and the one that survived the first pass of this
    # fix: `n_above = 0` on a FAILED vial is not an observation of zero
    # climbers. Carried into the pipeline as Count = 0 against n = N_start it
    # computes PI = -100 -- a maximally impaired vial manufactured out of a
    # vial that was never scored. Measured on a six-vial cohort with one
    # unscored vial: group PI -40.0 against a truth of -28.0.
    _nan_check("B2 a failed vial reports no climber count, not 0",
               d.get("n_above"), results)
    _nan_check("B2 a failed vial reports no non-climber count either",
               d.get("n_below"), results)
    _check("B2 what the detector DID observe is kept: it found no blobs",
           d.get("N_detected"), 0, 0, results)

    # A REFUSED row is different: the counts were measured and are kept; only
    # the rate is withheld, for want of a denominator.
    v_ref = VialROI(x=0, y=0, w=300, h=400, vial_id=3, expected_n=0)
    d_ref = det.count_in_vials(_frame(above_n=12, below_n=13), [v_ref])[0].to_dict()
    _check("B2 a REFUSED row keeps the counts it did measure",
           d_ref.get("n_above"), 12, 0, results)

    # An ROI outside the frame is also a failed measurement, not an empty vial.
    v_bad = VialROI(x=5000, y=5000, w=50, h=50, vial_id=2, expected_n=20)
    d_bad = det.count_in_vials(blank, [v_bad])[0].to_dict()
    _bool_check("B2 an off-frame ROI is MEASUREMENT_FAILED, not an empty vial",
                str(d_bad.get("Missingness")) == MS.MEASUREMENT_FAILED,
                MS.MEASUREMENT_FAILED, d_bad.get("Missingness"), results)
    _nan_check("B2 an off-frame ROI yields no percentage", d_bad.get("Pct"),
               results)


# ══════════════════════════════════════════════════════════════════════════════
# B3 -- auto_session: rounding, NaN and the missing column
# ══════════════════════════════════════════════════════════════════════════════
def _block_b3(results):
    from modules import auto_session as AS
    from modules import missingness as MS

    registry = AS.build_vial_registry([
        {"vial_id": 1, "Genotype": "WT", "Sex": "Male", "Treatment": "Veh",
         "expected_n": 20},
        {"vial_id": 2, "Genotype": "WT", "Sex": "Male", "Treatment": "Veh",
         "expected_n": 20},
    ])

    # Fractional counts are real: a median over an even frame window and
    # `climbing_band`'s merge correction both produce them. `int()` truncated
    # 6.7 to 6 -- a systematic downward bias on every vial, every timepoint.
    raw = pd.DataFrame([
        {"VialID": 1, "Time": 10.5, "n_above": 6.7, "Total": 19.4,
         "Replicate": 1, "QC_flag": "ok", "Confidence": 0.9},
        {"VialID": 2, "Time": 10.5, "n_above": np.nan, "Total": np.nan,
         "Replicate": 1, "QC_flag": "no_flies_detected", "Confidence": 0.0},
    ])
    try:
        out = AS.raw_counts_to_drophenix(raw, registry, replicate=1)
        raised = None
    except Exception as exc:                            # pragma: no cover
        out, raised = None, f"{type(exc).__name__}"
    _bool_check("B3 a NaN count no longer crashes the converter",
                raised is None, "no exception", raised or "none", results)
    if out is None:
        return

    r1 = out[out["VialID"] == 1].iloc[0] if "VialID" in out.columns else out.iloc[0]
    _check("B3 6.7 rounds to 7, it is not truncated to 6", r1.get("Count"),
           7, 0, results)
    _check("B3 the unrounded count is preserved", r1.get("Count_raw"),
           6.7, 1e-9, results)
    _check("B3 the denominator is N_start from the loading record, not Total",
           r1.get("n"), 20, 0, results)
    _check("B3 10.5 s is preserved as 10.5, not truncated to 10",
           r1.get("Time"), 10.5, 1e-9, results)

    r2 = out[out["VialID"] == 2].iloc[0]
    _bool_check("B3 the failed vial is kept as a row, not dropped",
                len(out) == 2, "2 rows", len(out), results)
    _nan_check("B3 the failed vial's Count stays NaN", r2.get("Count"), results)
    _bool_check("B3 the failed vial is typed MEASUREMENT_FAILED",
                str(r2.get("Missingness")) == MS.MEASUREMENT_FAILED,
                MS.MEASUREMENT_FAILED, r2.get("Missingness"), results)

    # A missing column must be named, not surface as a bare KeyError four
    # frames up the stack.
    try:
        AS.raw_counts_to_drophenix(raw.drop(columns=["n_above"]), registry)
        msg = "no exception"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
    _bool_check("B3 a missing n_above column is refused by name",
                "n_above" in msg and "KeyError" not in msg,
                "ValueError naming n_above", msg[:34], results)

    # A vial with no loading record cannot have a denominator invented for it.
    reg_no_n = AS.build_vial_registry([
        {"vial_id": 1, "Genotype": "WT", "Sex": "Male", "Treatment": "Veh"},
    ])
    try:
        AS.raw_counts_to_drophenix(raw[raw.VialID == 1], reg_no_n)
        msg2 = "no exception"
    except Exception as exc:
        msg2 = f"{type(exc).__name__}: {exc}"
    _bool_check("B3 a vial with no N_start in the registry is refused",
                "N_start" in msg2 or "expected_n" in msg2,
                "refusal naming N_start", msg2[:34], results)

    # partial_update's live plot must not sum the denominators of vials that
    # were never measured -- that inflates n while the mean excludes them.
    part = AS.partial_update(raw, [
        {"vial_id": 1, "Genotype": "WT", "Sex": "Male", "Treatment": "Veh",
         "expected_n": 20},
        {"vial_id": 2, "Genotype": "WT", "Sex": "Male", "Treatment": "Veh",
         "expected_n": 20},
    ])
    _check("B3 the live summary counts only the vial actually measured",
           (part["n"].iloc[0] if not part.empty and "n" in part.columns else None),
           20, 0, results)
    _check("B3 the live summary reports how many vials failed",
           (part["n_failed"].iloc[0] if not part.empty
            and "n_failed" in part.columns else None),
           1, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# B4 -- the converters propagate the QC fields or refuse
# ══════════════════════════════════════════════════════════════════════════════
def _block_b4(results):
    from modules.video_analyzer import video_counts_to_drophenix
    from modules import missingness as MS

    raw = pd.DataFrame([
        {"VialID": 1, "Time": 10, "Genotype": "WT", "Sex": "Male",
         "Treatment": "Veh", "n_above": 12, "n_below": 13, "N_detected": 25,
         "N_start": 30, "Total": 25, "Pct": 40.0, "PI": -20.0,
         "QC_flag": "count_mismatch", "Confidence": 0.83,
         "Missingness": MS.OBSERVED, "Replicate": 1},
        {"VialID": 2, "Time": 10, "Genotype": "WT", "Sex": "Male",
         "Treatment": "Veh", "n_above": np.nan, "n_below": np.nan,
         "N_detected": 0, "N_start": 30, "Total": 0, "Pct": np.nan,
         "PI": np.nan, "QC_flag": "no_flies_detected", "Confidence": 0.0,
         "Missingness": MS.MEASUREMENT_FAILED, "Replicate": 1},
    ])
    out = video_counts_to_drophenix(raw)

    for col in ("QC_flag", "Confidence", "Missingness", "N_start", "N_detected"):
        _bool_check(f"B4 the converter carries {col} through",
                    col in out.columns, "present",
                    "present" if col in out.columns else "DROPPED", results)

    r1 = out[out["VialID"] == 1].iloc[0]
    _check("B4 n is N_start (30), not max(Total, Count) = 25", r1.get("n"),
           30, 0, results)
    r2 = out[out["VialID"] == 2].iloc[0]
    _nan_check("B4 a failed vial converts to Count NaN, not 0", r2.get("Count"),
               results)
    _bool_check("B4 the failed vial survives conversion as a row",
                len(out) == 2, "2 rows", len(out), results)

    # Whatever a producer wrote in n_above, a row typed as a failed measurement
    # cannot carry a count through this boundary.
    raw_zero = raw.copy()
    raw_zero.loc[raw_zero["VialID"] == 2, ["n_above", "n_below"]] = 0
    out_z = video_counts_to_drophenix(raw_zero)
    _nan_check("B4 a zero count on a MEASUREMENT_FAILED row is still refused",
               out_z[out_z["VialID"] == 2].iloc[0].get("Count"), results)

    # End to end: the metric a user actually reads. Five impaired vials of 30
    # loaded plus one unscored vial must average to the five, not to six.
    from modules.normalizer import normalise
    cohort = pd.DataFrame([
        {"VialID": i + 1, "Time": 30, "Genotype": "FUS", "Sex": "Male",
         "Treatment": "Veh", "Replicate": i + 1, "n_above": a, "n_below": b,
         "N_detected": a + b, "N_start": 30, "QC_flag": "ok",
         "Confidence": 1.0, "Missingness": MS.OBSERVED}
        for i, (a, b) in enumerate([(12, 13), (10, 12), (11, 14), (9, 11), (12, 12)])
    ] + [
        {"VialID": 6, "Time": 30, "Genotype": "FUS", "Sex": "Male",
         "Treatment": "Veh", "Replicate": 6, "n_above": 0, "n_below": 0,
         "N_detected": 0, "N_start": 30, "QC_flag": "no_flies_detected",
         "Confidence": 0.0, "Missingness": MS.MEASUREMENT_FAILED},
    ])
    nz = normalise(video_counts_to_drophenix(cohort))
    pis = pd.to_numeric(nz["PI"], errors="coerce")
    _check("B4 group PI is the truth (-28.0), not the pre-fix -7.0",
           pis.mean(), -28.0, 0.05, results)
    _check("B4 the unscored vial is not counted as a replicate",
           pis.notna().sum(), 5, 0, results)

    # A frame with no QC columns at all cannot be converted silently: the
    # absence of the detector's refusal signal is itself disqualifying.
    try:
        video_counts_to_drophenix(raw.drop(columns=["QC_flag", "Confidence"]))
        msg = "no exception"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
    _bool_check("B4 a frame with no QC columns is refused by name",
                "QC_flag" in msg, "refusal naming QC_flag", msg[:34], results)


# ══════════════════════════════════════════════════════════════════════════════
# B5 -- the Video Analysis page's three signature breaks (N70), fixed last
# ══════════════════════════════════════════════════════════════════════════════
def _block_b5(results):
    src = (_ROOT / "app.py").read_text(encoding="utf-8", errors="replace")

    _bool_check("B5 no call passes video_path= to VideoAnalyzer's constructor",
                "VideoAnalyzer(video_path=" not in src.replace(" ", ""),
                "absent", "present" if "VideoAnalyzer(video_path="
                in src.replace(" ", "") else "absent", results)
    _bool_check("B5 no call passes vial_configs= to VideoAnalyzer",
                "vial_configs=" not in src.replace(" ", ""),
                "absent", "present" if "vial_configs="
                in src.replace(" ", "") else "absent", results)
    _bool_check("B5 the page builds real VialROI objects with geometry",
                "VialROI(" in src, "VialROI(", "VialROI(" in src, results)
    _bool_check("B5 analyze() is called with a video path and vials",
                ".analyze(" in src and "sample_timepoints" in src,
                "explicit args", "explicit args" if "sample_timepoints" in src
                else "missing", results)


# ══════════════════════════════════════════════════════════════════════════════
# B6 -- auto-numbered replicates are refused (N10)
# ══════════════════════════════════════════════════════════════════════════════
def _block_b6(results):
    if not _CV2:
        _bool_check("B6 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return
    from modules.video_analyzer import BatchVideoAnalyzer
    from modules.cv_pipeline import CVPipeline
    from modules.fly_detector import VialROI

    vials = [VialROI(x=0, y=0, w=300, h=400, vial_id=1, expected_n=12)]

    try:
        BatchVideoAnalyzer().analyze_batch(["a.mp4", "b.mp4"], vials,
                                           sample_timepoints=[3])
        msg = "no exception"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
    _bool_check("B6 BatchVideoAnalyzer refuses to auto-number replicates",
                "replicate" in msg.lower() and "no exception" not in msg,
                "refusal", msg[:34], results)

    try:
        CVPipeline().batch_analyze(["a.mp4", "b.mp4"], vials, [3])
        msg2 = "no exception"
    except Exception as exc:
        msg2 = f"{type(exc).__name__}: {exc}"
    _bool_check("B6 CVPipeline refuses to auto-number replicates",
                "replicate" in msg2.lower() and "no exception" not in msg2,
                "refusal", msg2[:34], results)

    # An explicit mapping is what makes two camera angles of ONE cohort one
    # replicate. The mapping must be honoured, not renumbered.
    import inspect
    sig = inspect.signature(BatchVideoAnalyzer.analyze_batch)
    _bool_check("B6 the explicit mapping is part of the signature",
                "replicate_map" in sig.parameters, "replicate_map",
                list(sig.parameters)[-1], results)
    sig2 = inspect.signature(CVPipeline.batch_analyze)
    _bool_check("B6 CVPipeline takes the same explicit mapping",
                "replicate_map" in sig2.parameters, "replicate_map",
                list(sig2.parameters)[-1], results)


def main() -> int:
    results = []
    for label, fn in (("B1", _block_b1), ("B2", _block_b2), ("B3", _block_b3),
                      ("B4", _block_b4), ("B5", _block_b5), ("B6", _block_b6)):
        try:
            fn(results)
        except Exception as exc:
            results.append((f"{label} block raised {type(exc).__name__}",
                            "no exception", str(exc)[:38], False))

    print("=" * 96)
    print("DroPhenix P0 Block B regression suite -- denominator and missing-measurement representation")
    print("  N5 denominator | N8 typed missingness | N79 cast | N70 page signatures | N10 replicates")
    print("=" * 96)
    print(f"{'Check':<62}{'Expected':>16}{'Engine':>12}{'Pass':>6}")
    print("-" * 96)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<62}{str(exp)[:16]:>16}{str(got)[:12]:>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 96)
    print(f"{len(results)} checks")
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
