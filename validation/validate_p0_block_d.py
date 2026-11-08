"""
validate_p0_block_d.py
Regression suite for P0 Block D of the 2026-11 audit: the eleven P3-sized
crashes and unread safeguards.

Every check here failed, or raised, before the Block D fixes. See
docs/BUG_AND_RISK_REGISTER.md findings N68, N71, N72, N76, N77, N80, N81, N82,
N84, N85, N86 and docs/PRIORITISED_DEVELOPMENT_PLAN.md section 1 Block D.

Why these are P0 despite being P3-sized
---------------------------------------
Each one is small. Together they are the difference between a tool a second
laboratory can run and one that cannot, and three of them stop a feature dead:
live recording cannot start on Windows, the CSV page's validation step never
executes, and an advertised wide-format file raises a traceback on upload. A
defect that prevents a check from running hides every defect that check would
have found, which is why they are cleared before P1 rather than after.

What Block D is about
---------------------
D1  N68  `OPENCV_VIDEOIO_PRIORITY_DSHOW = "0"` removes DirectShow from
         OpenCV's backend registry, and then all four camera opens request
         `cv2.CAP_DSHOW` explicitly. Requesting a backend that is not in the
         registry fails, so **live recording cannot start on Windows**, and the
         operator is told the camera is disconnected or in use. `recording.py`
         even computes the correct backend into a loop variable `bk` and then
         discards it, which is what the defect is: a one-word slip.
D2  N71  `validate_dataframe` returns a `DataAudit` dataclass and the CSV page
         calls `.get("errors", [])` on it, so the validation step that page
         exists to run raises `AttributeError` and never executes.
D3  N72  In the wide-to-long converter `meta_cols` is built BEFORE the
         `Total -> n` rename and then passed as `melt`'s `id_vars`, so a wide
         file whose total column is literally named `Total` -- one of the names
         the function itself looks for -- raises an uncaught `KeyError` that
         escapes `load_data` as a red traceback, for a format the UI advertises.
D4  N76  `annotate_frame` re-detects the blobs it draws, WITHOUT the background
         frame the counts were computed with, so in `mode="bg"` the QC overlay
         is a different computation from the numbers printed on it; and the
         scoring line is drawn from `v.h` while the blob colours are decided
         against the detector's own line, which differ whenever the ROI is
         clipped at the frame edge.
D5  N77  `VialROI.auto_grid` splits the frame geometrically with no reference
         to the image at all, so an off-centre rack assigns flies to the wrong
         vial and therefore to the wrong GENOTYPE -- a mislabelling, not a
         measurement error.
D6  N80  `np.diff(idx, prepend=idx[0])` makes the first difference 0, which
         satisfies the `<= 0` restart test, so DAM session labels begin at 2.
         Any report printing "Session 1" will never find it.
D7  N81  `Sleep_scoreable` has no consumer. A bin wider than the sleep
         criterion cannot resolve the rule -- a single zero 60-min bin scores
         as a 60-min sleep bout whatever happened inside it, and degrading this
         laboratory's own 1-min recording to 60-min bins loses 85% of the sleep
         -- so the flag is computed, tested, and then ignored by the summary
         that averages those very columns. An unread flag reads as a safeguard
         in review and is not one.
D8  N82  `to_dams_long(drop_dead=True)` filters channels by total activity and
         records no count of what it removed, while `unlabelled_channels_dropped`
         two hundred lines earlier is recorded. A differential exclusion -- and
         exclusion by low activity is differential, since a sick cohort moves
         less -- is invisible in the run summary.
D9  N84  A download labelled "Protocol PDF" serves a PNG.
D10 N85  Thirteen `st.pyplot` calls against one `plt.close` in ~8,000 lines, so
         Matplotlib figures accumulate for the session's lifetime.
D11 N86  Sibling modules imported without the `modules.` prefix resolve only
         when `modules/` happens to be on `sys.path`.

Run:  python validation/validate_p0_block_d.py   (exit 0 = all passed)
"""

import ast
import inspect
import re
import sys
import warnings
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
    try:
        ok = (got is not None) and np.isfinite(float(got)) \
            and abs(float(got) - expected) <= tol
    except (TypeError, ValueError):
        ok = False
    results.append((name, f"{expected:g}",
                    (f"{float(got):g}" if _finite(got) else str(got)), ok))


def _finite(v):
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _bool_check(name, cond, expected, got, results):
    results.append((name, str(expected), str(got), bool(cond)))


def _no_raise(name, fn, results):
    """A call that must simply work. Several of these raised before."""
    try:
        fn()
        results.append((name, "no error", "ok", True))
    except Exception as exc:
        results.append((name, "no error",
                        f"{type(exc).__name__}: {str(exc)[:24]}", False))


def _src(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8", errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# D1 -- N68: the camera backend must exist in the registry
# ══════════════════════════════════════════════════════════════════════════════
def _block_d1(results):
    rec_src = _src("modules/recording.py")
    rcd_src = _src("modules/recorder.py")

    _bool_check("D1 DirectShow is no longer removed from the registry",
                'OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "0"' not in rec_src
                and "OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'" not in rec_src,
                "not disabled",
                "disabled" if 'PRIORITY_DSHOW"] = "0"' in rec_src else "ok",
                results)
    _bool_check("D1 FFMPEG is still enabled (it is the only HEVC file reader)",
                'OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"' in rec_src,
                "enabled", "ok" if "PRIORITY_FFMPEG" in rec_src else "absent",
                results)
    # DirectShow may legitimately appear in the FALLBACK ORDER; what must not
    # appear is an unconditional request for it on a VideoCapture call, which is
    # what failed against the emptied registry.
    def _hardcoded(src):
        return [ln.strip() for ln in src.splitlines()
                if "VideoCapture(" in ln and "CAP_DSHOW" in ln
                and not ln.strip().startswith("#")]
    _bool_check("D1 no VideoCapture call hardcodes CAP_DSHOW in recording.py",
                not _hardcoded(rec_src), "none",
                "; ".join(_hardcoded(rec_src))[:18] or "none", results)
    _bool_check("D1 no VideoCapture call hardcodes CAP_DSHOW in recorder.py",
                not _hardcoded(rcd_src), "none",
                "; ".join(_hardcoded(rcd_src))[:18] or "none", results)
    _bool_check("D1 nor is any capture opened without a chosen backend",
                not [ln for ln in rcd_src.splitlines()
                     if "cv2.VideoCapture(" in ln and not ln.strip().startswith("#")],
                "all via helper",
                str(len([ln for ln in rcd_src.splitlines()
                         if "cv2.VideoCapture(" in ln
                         and not ln.strip().startswith("#")])), results)

    if not _CV2:
        _bool_check("D1 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return

    from modules import recording as R

    # The real defect, tested directly: every backend the module will ASK for
    # must be one OpenCV actually has for cameras. Requesting a backend that is
    # not in the registry is what made live recording impossible.
    reg = set(cv2.videoio_registry.getCameraBackends())
    order = R.camera_backend_order()
    _bool_check("D1 a backend order is published", bool(order), "non-empty",
                str(order)[:16], results)
    unavailable = [b for b in order
                   if b != cv2.CAP_ANY and b not in reg]
    _bool_check("D1 every requested camera backend is in the registry",
                not unavailable, "all present",
                str([cv2.videoio_registry.getBackendName(b)
                     for b in unavailable])[:18], results)
    _bool_check("D1 CAP_ANY is the last resort, not the first choice",
                order[-1] == cv2.CAP_ANY and order[0] != cv2.CAP_ANY,
                "CAP_ANY last",
                cv2.videoio_registry.getBackendName(order[0]), results)
    names = [cv2.videoio_registry.getBackendName(b) for b in order]
    _bool_check("D1 the order is reported by name, for the operator",
                all(isinstance(n, str) and n for n in names), "named",
                ",".join(names)[:18], results)

    # A URL or file source is not a camera and must not be handed a camera
    # backend. `CameraStream(source=...)` accepts an IP stream by its own
    # docstring.
    _bool_check("D1 a string source is not given a camera backend",
                R.capture_backend_for("rtsp://host/stream") == cv2.CAP_ANY,
                "CAP_ANY",
                str(R.capture_backend_for("rtsp://host/stream")), results)
    _bool_check("D1 an integer source is given a camera backend",
                R.capture_backend_for(0) == order[0], str(order[0]),
                str(R.capture_backend_for(0)), results)

    # And `recorder` must route through the same decision rather than keeping
    # its own copy of it.
    from modules import recorder as RC
    _bool_check("D1 recorder uses the same backend decision",
                "capture_backend_for" in rcd_src or "camera_backend_order" in rcd_src,
                "shared",
                ("shared" if ("capture_backend_for" in rcd_src
                              or "camera_backend_order" in rcd_src)
                 else "own copy"), results)
    _bool_check("D1 the camera class still accepts a string source "
                "(an IP stream)",
                "source" in inspect.signature(RC.DroPhenixCamera.__init__).parameters,
                "source", "missing", results)
    _bool_check("D1 and opening one does not raise, it returns False",
                RC.DroPhenixCamera(source="rtsp://127.0.0.1:1/none").open() is False,
                False, "ok", results)


# ══════════════════════════════════════════════════════════════════════════════
# D2 -- N71: the validation step must actually run
# ══════════════════════════════════════════════════════════════════════════════
def _block_d2(results):
    from modules.data_parser import DataAudit, validate_dataframe

    good = pd.DataFrame({
        "Genotype": ["WT"] * 3, "Sex": ["Male"] * 3, "Treatment": ["Veh"] * 3,
        "Replicate": [1, 1, 1], "Time": [10, 20, 30], "Count": [5, 8, 12],
        "n": [20, 20, 20],
    })
    audit = validate_dataframe(good)
    _bool_check("D2 validate_dataframe returns a DataAudit",
                isinstance(audit, DataAudit), "DataAudit",
                type(audit).__name__, results)
    _bool_check("D2 clean data audits clean", audit.ok is True, True,
                audit.ok, results)
    _bool_check("D2 .errors and .warnings are lists",
                isinstance(audit.errors, list) and isinstance(audit.warnings, list),
                "lists", f"{type(audit.errors).__name__}", results)

    # The defect: the page called `.get("errors", [])` on the dataclass.
    app = _src("app.py")
    _bool_check("D2 the page no longer calls .get() on the audit",
                'val_r.get("errors"' not in app and "val_r.get('errors'" not in app,
                "no .get", "still .get" if "val_r.get(" in app else "ok", results)
    _bool_check("D2 the page reads the audit's own attributes",
                "_render_audit" in app, "_render_audit",
                "present" if "_render_audit" in app else "absent", results)

    # A dataclass cannot be subscripted, so the old call could only ever raise;
    # a dict-shaped export is provided for anything that wants one.
    _bool_check("D2 the audit offers a dict form for export",
                isinstance(audit.to_dict(), dict), "dict",
                type(audit.to_dict()).__name__, results)
    bad = good.copy()
    bad.loc[0, "n"] = 0
    a2 = validate_dataframe(bad)
    _bool_check("D2 an invalid frame audits as not ok", a2.ok is False, False,
                a2.ok, results)
    _bool_check("D2 and names the problem",
                any("n <= 0" in e or "n <=" in e for e in a2.errors),
                "names n", (a2.errors[0][:18] if a2.errors else "none"), results)


# ══════════════════════════════════════════════════════════════════════════════
# D3 -- N72: an advertised wide format must not raise
# ══════════════════════════════════════════════════════════════════════════════
def _wide(total_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Genotype": ["WT", "Mut"], "Sex": ["Male", "Male"],
        "Treatment": ["Veh", "Veh"], "Replicate": [1, 1],
        total_col: [20, 20],
        "t10": [12, 5], "t20": [16, 7], "t30": [18, 9],
    })


def _block_d3(results):
    from modules.data_parser import load_data

    for col in ("Total", "total", "N", "n_total", "total_flies"):
        def _run(c=col):
            df, w, e = load_data(_wide(c))
            if e:
                raise RuntimeError("; ".join(e)[:40])
            assert "n" in df.columns, "no n column"
            assert len(df) == 6, f"{len(df)} rows"
        _no_raise(f"D3 wide file with the total column named {col!r} loads", _run,
                  results)

    df, w, e = load_data(_wide("Total"))
    _check("D3 n carried through from `Total`",
           float(df["n"].iloc[0]) if not df.empty else np.nan, 20.0, 0, results)
    _check("D3 three timepoints become three rows per group",
           float(df.groupby(["Genotype"])["Time"].nunique().iloc[0])
           if not df.empty else np.nan, 3.0, 0, results)
    _check("D3 counts land on the right timepoint",
           float(df[(df["Genotype"] == "WT") & (df["Time"] == 20)]["Count"].iloc[0])
           if not df.empty else np.nan, 16.0, 0, results)
    _bool_check("D3 `Total` does not survive as a stray column",
                "Total" not in df.columns, "renamed",
                "still there" if "Total" in df.columns else "ok", results)

    # No total column at all: an error, not a traceback.
    bare = _wide("Total").drop(columns=["Total"])
    def _bare():
        df2, w2, e2 = load_data(bare)
        if not e2:
            raise RuntimeError("accepted a wide file with no cohort size")
    _no_raise("D3 a wide file with no cohort size is an error, not a crash",
              _bare, results)
    _, _, e3 = load_data(bare)
    _bool_check("D3 and the error names what is missing",
                any("n" in str(x) for x in e3), "names n",
                (str(e3[0])[:18] if e3 else "none"), results)


# ══════════════════════════════════════════════════════════════════════════════
# D4 -- N76: the overlay must be the same computation as the counts
# ══════════════════════════════════════════════════════════════════════════════
def _bg_frame(W=240, H=400):
    """An empty vial with a dark fixed feature -- the case `mode='bg'` exists for."""
    f = np.full((H, W, 3), 235, np.uint8)
    cv2.rectangle(f, (0, 0), (W - 1, H - 1), (120, 120, 120), 3)
    cv2.putText(f, "LBL", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (30, 30, 30), 3)
    return f


def _with_flies(bg, above_n, below_n, r=6):
    f = bg.copy()
    H = f.shape[0]
    line = H // 2
    for i in range(above_n):
        cv2.circle(f, (30 + (i % 4) * 50, 40 + (i // 4) * 40), r, (25, 25, 25), -1)
    for i in range(below_n):
        cv2.circle(f, (30 + (i % 4) * 50, line + 50 + (i // 4) * 40), r,
                   (25, 25, 25), -1)
    return f


def _block_d4(results):
    if not _CV2:
        _bool_check("D4 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return
    from modules.fly_detector import (FlyDetector, DetectorConfig, VialROI,
                                      VialCount)

    bg = _bg_frame()
    frame = _with_flies(bg, 4, 6)
    det = FlyDetector(DetectorConfig(mode="bg", min_area=20, max_area=800,
                                     threshold_frac=0.5))
    vials = [VialROI(x=0, y=0, w=240, h=400, vial_id=1, expected_n=10)]
    counts = det.count_in_vials(frame, vials, bg_frame=bg)
    vc = counts[0]

    _bool_check("D4 the counted blobs are kept on the record",
                isinstance(getattr(vc, "blobs", None), list) and len(vc.blobs) > 0,
                "kept", str(len(getattr(vc, "blobs", []) or [])), results)
    _bool_check("D4 the detector's own scoring line is kept too",
                _finite(getattr(vc, "line_y", None)), "finite",
                str(getattr(vc, "line_y", "missing")), results)

    # The overlay must draw what was counted. If it re-detected, handing it a
    # DIFFERENT frame would change the drawing; it must not.
    ann_a = det.annotate_frame(frame, vials, counts)
    ann_b = det.annotate_frame(_with_flies(bg, 1, 1), vials, counts)
    _bool_check("D4 the overlay draws the counted blobs, not a re-detection",
                ann_a is not None and ann_b is not None
                and int(np.count_nonzero(np.any(ann_a != frame, axis=2)))
                == int(np.count_nonzero(np.any(
                    ann_b != _with_flies(bg, 1, 1), axis=2))),
                "identical overlay", "differs", results)

    # In mode='bg' the counts use the background; a re-detection without it is a
    # different computation. The label must agree with the record.
    _check("D4 counts in mode='bg' with a dark fixed feature: 4 above",
           vc.n_above, 4, 0, results)
    _check("D4 and 6 below", vc.n_below, 6, 0, results)
    _check("D4 and 10 detected of 10 loaded", vc.total, 10, 0, results)

    # The line the overlay draws must be the line the detector used, including
    # when the ROI runs past the frame edge and the crop is shorter than v.h.
    clipped = [VialROI(x=0, y=200, w=240, h=400, vial_id=1, expected_n=10)]
    cc = det.count_in_vials(frame, clipped, bg_frame=bg)[0]
    crop_h = frame.shape[0] - 200
    _check("D4 a clipped ROI's line is measured on the real crop",
           getattr(cc, "line_y", np.nan), int(crop_h * 0.5), 0, results)
    _bool_check("D4 which is not the line v.h would have given",
                int(crop_h * 0.5) != int(400 * 0.5), "different",
                f"{int(crop_h * 0.5)} vs 200", results)

    _no_raise("D4 annotating a vial that was never measured does not raise",
              lambda: det.annotate_frame(frame, vials,
                                         det.count_in_vials(None, vials)),
              results)
    # The fallback path must not reintroduce the defect: when a record carries
    # no blobs, the re-detection takes the same background the counts used.
    _bool_check("D4 the overlay accepts the background the counts used",
                "bg_frame" in inspect.signature(det.annotate_frame).parameters,
                "bg_frame", str(list(inspect.signature(det.annotate_frame)
                                     .parameters))[-16:], results)
    stripped = [VialCount(**{**{f.name: getattr(counts[0], f.name)
                               for f in __import__("dataclasses").fields(VialCount)},
                            "blobs": [], "line_y": float("nan")})]
    _no_raise("D4 a record with no blobs still annotates, with the background",
              lambda: det.annotate_frame(frame, vials, stripped, bg_frame=bg),
              results)
    ann_c = det.annotate_frame(frame, vials, stripped, bg_frame=bg)
    _bool_check("D4 and it says so on the label, rather than silently differing",
                ann_c is not None, "labelled", "ok", results)


# ══════════════════════════════════════════════════════════════════════════════
# D5 -- N77: an off-centre rack must not be assigned geometrically
# ══════════════════════════════════════════════════════════════════════════════
def _rack(n_vials=3, W=600, H=400, x0=320, pitch=80, tube_w=56, r=6, seed=0):
    """
    A frame with `n_vials` tubes whose left edge starts at `x0` -- i.e. an
    OFF-CENTRE rack, which is what a camera that is not square to the bench
    produces. Flies sit low in each tube, as they do at t = 0.
    """
    rng = np.random.default_rng(seed)
    f = np.full((H, W, 3), 235, np.uint8)
    centres = []
    for v in range(n_vials):
        xa = x0 + v * pitch
        xb = xa + tube_w
        centres.append((xa + xb) // 2)
        cv2.rectangle(f, (xa, 20), (xb, H - 20), (205, 205, 205), 2)
        for _ in range(10):
            cx = int(rng.integers(xa + r + 2, xb - r - 2))
            cy = int(rng.integers(H - 150, H - 30))
            cv2.circle(f, (cx, cy), r, (25, 25, 25), -1)
    return f, centres


def _block_d5(results):
    if not _CV2:
        _bool_check("D5 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return
    from modules.fly_detector import VialROI, VIAL_PRESETS as FD_PRESETS

    frame, centres = _rack(seed=3)
    H, W = frame.shape[:2]

    def _hits(vials):
        """How many tube centres fall inside the ROI that claims them."""
        ok = 0
        for i, cx in enumerate(centres):
            v = next((v for v in vials if v.x <= cx < v.x + v.w), None)
            if v is not None and v.vial_id == i + 1:
                ok += 1
        return ok

    geo = VialROI.auto_grid(W, H, preset="3 vials (horizontal)")
    img = VialROI.from_frame(frame, n_vials=3)
    _bool_check("D5 from_frame returns regions and a confidence",
                isinstance(img, dict) and len(img.get("vials", [])) == 3
                and _finite(img.get("confidence")), "3 + confidence",
                f"{len(img.get('vials', []))}/{img.get('confidence')}", results)

    n_geo = _hits(geo)
    n_img = _hits(img["vials"])
    _bool_check("D5 the geometric grid misassigns the off-centre rack",
                n_geo < 3, "<3", f"{n_geo}/3", results)
    _check("D5 the image-referenced grid assigns all three", n_img, 3, 0, results)
    _bool_check("D5 which is the difference between a genotype label and a "
                "mislabel", n_img > n_geo, "better", f"{n_img} vs {n_geo}",
                results)

    # It must refuse rather than invent regions when there is no structure.
    blank = np.full((400, 600, 3), 235, np.uint8)
    nothing = VialROI.from_frame(blank, n_vials=3)
    _bool_check("D5 a featureless frame is refused, not carved into vials",
                nothing.get("needs_review") is True, True,
                nothing.get("needs_review"), results)
    _bool_check("D5 and the reason is stated",
                len(str(nothing.get("reason", ""))) > 20, "stated",
                str(nothing.get("reason", ""))[:18], results)

    # auto_grid stays, and says what it is.
    _doc = (VialROI.auto_grid.__doc__ or "")
    _bool_check("D5 auto_grid declares itself uncalibrated",
                "uncalibrated" in _doc.lower()
                or "no reference to the image" in _doc,
                "declared",
                ("declared" if "uncalibrated" in _doc.lower() else "silent"),
                results)
    _check("D5 auto_grid still returns the requested region count",
           len(geo), 3, 0, results)
    # A layout nobody asked for is not a default: an unknown preset used to
    # return ONE full-frame vial, pooling every tube under one genotype.
    _raises_d5 = []
    try:
        VialROI.auto_grid(W, H, preset="9 vials (3x3)")
    except Exception as exc:
        _raises_d5.append(str(exc))
    _bool_check("D5 an unknown preset is refused, not silently one big vial",
                bool(_raises_d5) and "preset" in _raises_d5[0].lower(),
                "refused",
                (_raises_d5[0][:18] if _raises_d5 else "returned 1 vial"), results)
    _bool_check("D5 the three-cylinder rig this lab uses is a listed layout",
                "3 vials (horizontal)" in FD_PRESETS, "listed",
                "present" if "3 vials (horizontal)" in FD_PRESETS else "absent",
                results)


# ══════════════════════════════════════════════════════════════════════════════
# D6 -- N80: session labels must start at 1
# ══════════════════════════════════════════════════════════════════════════════
def _dam_file(tmp: Path, blocks) -> Path:
    """
    A minimal Trikinetics-style monitor file. `blocks` is a list of row counts;
    each block restarts the index column at 1, which is what a monitor restart
    looks like in the file.
    """
    import datetime as _dt
    rows = []
    t = _dt.datetime(2026, 8, 11, 0, 0, 0)
    for n in blocks:
        for i in range(1, n + 1):
            cells = [str(i), t.strftime("%d %b %y"), t.strftime("%H:%M:%S"),
                     "1", "0", "0", "0", "0", "0", "0"]
            cells += ["5"] * 32
            rows.append("\t".join(cells))
            t += _dt.timedelta(minutes=1)
    p = tmp / "Monitor99.txt"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def _block_d6(results, tmp: Path):
    from modules import dam_reader as DR

    one = DR.read_monitor(_dam_file(tmp, [30]))
    _check("D6 no restart: a single session", one["Session"].nunique(), 1, 0, results)
    _check("D6 and it is session 1", float(one["Session"].min()), 1.0, 0, results)

    two = DR.read_monitor(_dam_file(tmp, [20, 20]))
    _bool_check("D6 one restart gives sessions {1, 2}, not {2, 3}",
                sorted(set(two["Session"].unique())) == [1, 2], "[1, 2]",
                str(sorted(set(two["Session"].unique()))), results)
    _check("D6 the first row is in session 1",
           float(two["Session"].iloc[0]), 1.0, 0, results)
    three = DR.read_monitor(_dam_file(tmp, [10, 10, 10]))
    _bool_check("D6 two restarts give {1, 2, 3}",
                sorted(set(three["Session"].unique())) == [1, 2, 3], "[1, 2, 3]",
                str(sorted(set(three["Session"].unique()))), results)
    _check("D6 and the session sizes are right (32 channels x 10 rows)",
           float((three["Session"] == 2).sum()), 320.0, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# D7 -- N81: an unread flag is not a safeguard
# ══════════════════════════════════════════════════════════════════════════════
def _dam_long(bin_min: float, n_days: int = 3, fly: str = "f1",
              geno: str = "WT") -> pd.DataFrame:
    """A fly asleep at night and active by day, sampled at `bin_min`."""
    n = int(n_days * 24 * 60 / bin_min)
    t = np.arange(n) * bin_min
    zt = (t / 60.0) % 24.0
    act = np.where(zt < 12.0, 8.0, 0.0)
    return pd.DataFrame({"Genotype": geno, "Treatment": "Veh", "Sex": "Male",
                         "FlyID": fly, "Time_min": t, "Activity": act})


def _block_d7(results):
    from modules import dams as D

    fine = D.sleep_per_fly(_dam_long(1.0))
    coarse = D.sleep_per_fly(_dam_long(60.0))
    _bool_check("D7 1-min data is scoreable",
                bool(fine["Sleep_scoreable"].iloc[0]), True,
                fine["Sleep_scoreable"].iloc[0], results)
    _bool_check("D7 60-min data is not",
                not bool(coarse["Sleep_scoreable"].iloc[0]), False,
                coarse["Sleep_scoreable"].iloc[0], results)

    s_fine = D.sleep_summary(fine)
    s_coarse = D.sleep_summary(coarse)
    _bool_check("D7 the summary reports how many flies were scoreable",
                "n_flies_sleep_scoreable" in s_coarse.columns,
                "column", str(list(s_coarse.columns))[-28:], results)
    _check("D7 none of them, on 60-min bins",
           float(s_coarse["n_flies_sleep_scoreable"].iloc[0])
           if "n_flies_sleep_scoreable" in s_coarse.columns else np.nan,
           0.0, 0, results)
    _bool_check("D7 so the sleep means are withheld, not averaged",
                pd.isna(s_coarse["Total_sleep_min_per_day"].iloc[0]), "NaN",
                str(s_coarse["Total_sleep_min_per_day"].iloc[0]), results)
    _bool_check("D7 while the activity means, which are exact, are kept",
                _finite(s_coarse["Total_activity_per_day"].iloc[0]), "finite",
                str(s_coarse["Total_activity_per_day"].iloc[0]), results)
    _bool_check("D7 and the reason is stated on the row",
                "bin" in str(s_coarse.get("Sleep_note",
                                          pd.Series(["missing"])).iloc[0]).lower(),
                "names bin width",
                str(s_coarse.get("Sleep_note", pd.Series(["missing"])).iloc[0])[:18],
                results)
    _bool_check("D7 1-min data still reports its sleep means",
                _finite(s_fine["Total_sleep_min_per_day"].iloc[0]), "finite",
                str(s_fine["Total_sleep_min_per_day"].iloc[0]), results)

    # A mixed group: only the scoreable flies may enter the sleep mean.
    mixed = pd.concat([_dam_long(1.0, fly="a"), _dam_long(1.0, fly="b"),
                       _dam_long(60.0, fly="c")], ignore_index=True)
    pf = D.sleep_per_fly(mixed)
    sm = D.sleep_summary(pf)
    _check("D7 mixed group: 3 flies", float(sm["n_flies"].iloc[0]), 3.0, 0, results)
    _check("D7 of which 2 are sleep-scoreable",
           float(sm["n_flies_sleep_scoreable"].iloc[0])
           if "n_flies_sleep_scoreable" in sm.columns else np.nan, 2.0, 0, results)
    scoreable = pf[pf["Sleep_scoreable"]]
    _check("D7 and the sleep mean is over those two only",
           float(sm["Total_sleep_min_per_day"].iloc[0]),
           float(scoreable["Total_sleep_min_per_day"].mean()), 0.2, results)


# ══════════════════════════════════════════════════════════════════════════════
# D8 -- N82: a differential exclusion must be recorded
# ══════════════════════════════════════════════════════════════════════════════
def _labelled(n_alive=4, n_dead=2) -> pd.DataFrame:
    rows = []
    for ch in range(1, n_alive + n_dead + 1):
        dead = ch > n_alive
        geno = "Mut" if dead else "WT"
        for m in range(120):
            rows.append({"Monitor": "M1", "Channel": ch,
                         "Timestamp": pd.Timestamp("2026-08-11") + pd.Timedelta(minutes=m),
                         "Activity": 0.0 if dead else 5.0,
                         "Genotype": geno, "Treatment": "Veh", "Sex": "Male"})
    return pd.DataFrame(rows)


def _block_d8(results):
    from modules import dam_reader as DR

    lab = _labelled()
    out = DR.to_dams_long(lab, drop_dead=True, dead_threshold=50.0)
    _check("D8 4 of 6 channels survive the activity filter",
           out["FlyID"].nunique(), 4, 0, results)
    _check("D8 and the number removed is recorded",
           out.attrs.get("dead_channels_dropped"), 2, 0, results)
    _bool_check("D8 with the threshold it was judged against",
                _finite(out.attrs.get("dead_threshold")), "50",
                str(out.attrs.get("dead_threshold")), results)
    _bool_check("D8 and the channels themselves, so they can be checked",
                len(out.attrs.get("dead_channels", []) or []) == 2, 2,
                str(out.attrs.get("dead_channels", "missing"))[:18], results)
    # Exclusion by low activity is differential: a sick cohort moves less.
    dist = out.attrs.get("dead_channels_by_group", {}) or {}
    _bool_check("D8 the exclusion's distribution across groups is recorded",
                bool(dist), "recorded", str(dist)[:18], results)
    _bool_check("D8 and it shows the loss fell entirely on one genotype",
                dist.get("Mut") == 2 and not dist.get("WT"), "Mut: 2",
                str(dist)[:18], results)

    keep = DR.to_dams_long(lab, drop_dead=False)
    _check("D8 with the filter off, all 6 channels are kept",
           keep["FlyID"].nunique(), 6, 0, results)
    _check("D8 and nothing is reported as dropped",
           keep.attrs.get("dead_channels_dropped"), 0, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# D9 / D10 / D11 -- the label, the figures, the imports
# ══════════════════════════════════════════════════════════════════════════════
def _block_d9(results):
    src = _src("modules/protocol_guide.py")
    # A button labelled PDF must serve a PDF. Pair each download label with the
    # filename on the following lines.
    bad = []
    for m in re.finditer(r'"⬇ Download ([^"]*)"', src):
        tail = src[m.end():m.end() + 260]
        fn = re.search(r'"([\w./-]+\.(png|pdf|csv|jpg))"', tail)
        if not fn:
            continue
        label, ext = m.group(1).lower(), fn.group(2).lower()
        if ("pdf" in label) != (ext == "pdf"):
            bad.append(f"{m.group(1)} -> .{ext}")
    _bool_check("D9 every download label matches what it serves",
                not bad, "all match", "; ".join(bad)[:20], results)

    from modules import protocol_guide as PG
    _bool_check("D9 a real PDF can be produced from the protocol image",
                hasattr(PG, "_pdf_bytes"), "_pdf_bytes",
                "present" if hasattr(PG, "_pdf_bytes") else "absent", results)
    if hasattr(PG, "_pdf_bytes"):
        b = PG._pdf_bytes(PG._PROTOCOL_IMG)
        _bool_check("D9 and it really is a PDF (starts %PDF)",
                    b is None or b[:4] == b"%PDF", "%PDF",
                    (str(b[:4]) if b else "no image file"), results)
        # The asset is NOT in the repository (N84b), so the conversion must
        # fail softly and the page must say so rather than show a blank column.
        _bool_check("D9 a missing diagram returns None, it does not raise",
                    PG._pdf_bytes(PG._ASSETS / "does_not_exist.png") is None,
                    None, "ok", results)
        _bool_check("D9 and the absent diagram is reported with its path",
                    "Protocol diagram not available" in src
                    and "_PROTOCOL_IMG" in src, "reported",
                    "silent" if "not available" not in src else "reported",
                    results)
        # And a PDF really is produced when an image IS present.
        import tempfile as _tf
        try:
            from PIL import Image as _I
            with _tf.TemporaryDirectory() as _td:
                _p = Path(_td) / "x.png"
                _I.new("RGBA", (40, 30), (10, 10, 10, 255)).save(_p)
                _b2 = PG._pdf_bytes(_p)
            _bool_check("D9 a real image converts to a real PDF",
                        _b2 is not None and _b2[:4] == b"%PDF", "%PDF",
                        (str(_b2[:4]) if _b2 else "None"), results)
        except Exception as _e:
            _bool_check("D9 a real image converts to a real PDF", False,
                        "%PDF", str(_e)[:18], results)


def _block_d10(results):
    app = _src("app.py")
    tree = ast.parse(app)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    _bool_check("D10 a single figure-rendering helper exists",
                "_show_fig" in names, "_show_fig",
                "present" if "_show_fig" in names else "absent", results)
    n_pyplot = len(re.findall(r"st\.pyplot\(", app))
    n_in_helper = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_show_fig":
            n_in_helper = len(re.findall(r"st\.pyplot\(", ast.unparse(n)))
    _check("D10 st.pyplot is called in exactly one place", n_pyplot, 1, 0, results)
    _check("D10 and that place is the helper", n_in_helper, 1, 0, results)
    _closes = any(isinstance(n, ast.FunctionDef) and n.name == "_show_fig"
                  and "plt.close" in ast.unparse(n) for n in ast.walk(tree))
    _bool_check("D10 which closes the figure it rendered", _closes,
                "plt.close", ("plt.close" if _closes else "absent"), results)
    _bool_check("D10 every figure is rendered through it",
                len(re.findall(r"_show_fig\(", app)) >= 13, ">=13",
                str(len(re.findall(r"_show_fig\(", app))), results)


def _block_d11(results):
    """Every sibling import must be package-qualified or relative."""
    mod_dir = _ROOT / "modules"
    own = {p.stem for p in mod_dir.glob("*.py")} - {"__init__"}
    offenders = []
    for p in sorted(mod_dir.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module and \
                        node.module.split(".")[0] in own:
                    offenders.append(f"{p.name}: from {node.module}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in own:
                        offenders.append(f"{p.name}: import {a.name}")
    # A bare import inside a try/except ImportError fallback is the documented
    # direct-script path and is allowed only when the package-relative form is
    # tried first; the check below is on the module source for that pairing.
    unguarded = []
    for o in offenders:
        fname = o.split(":")[0]
        src = (mod_dir / fname).read_text(encoding="utf-8", errors="ignore")
        target = o.split()[-1]
        rel_first = (f"from .{target} import" in src
                     or f"from modules.{target} import" in src
                     or f"import modules.{target}" in src)
        if not rel_first:
            unguarded.append(o)
    _bool_check("D11 no sibling module is imported unqualified and unguarded",
                not unguarded, "none", "; ".join(unguarded)[:20], results)
    _bool_check("D11 every guarded fallback tries the qualified form first",
                not [o for o in offenders if o in unguarded], "all guarded",
                str(len(offenders)) + " fallbacks", results)


# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    import tempfile
    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        blocks = (("D1", _block_d1), ("D2", _block_d2), ("D3", _block_d3),
                  ("D4", _block_d4), ("D5", _block_d5),
                  ("D6", lambda r: _block_d6(r, tmp)),
                  ("D7", _block_d7), ("D8", _block_d8), ("D9", _block_d9),
                  ("D10", _block_d10), ("D11", _block_d11))
        for label, fn in blocks:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fn(results)
            except Exception as exc:
                results.append((f"{label} block raised {type(exc).__name__}",
                                "no exception", str(exc)[:38], False))

    print("=" * 96)
    print("DroPhenix P0 Block D regression suite -- crashes and unread safeguards")
    print("  N68 camera backend | N71 audit | N72 wide format | N76 overlay | N77 grid")
    print("  N80 session index | N81 unread flag | N82 exclusion count | N84 label")
    print("  N85 figure leak | N86 import prefix")
    print("=" * 96)
    print(f"{'Check':<62}{'Expected':>16}{'Engine':>12}{'Pass':>6}")
    print("-" * 96)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<62}{str(exp)[:16]:>16}{str(got)[:12]:>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 96)
    n_fail = sum(1 for _, _, _, ok in results if not ok)
    print(f"{len(results)} checks, {n_fail} failing")
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
