"""
validate_p0_block_a.py
Regression suite for P0 Block A of the 2026-11 audit: the four bypassed safeguards.

Every check here FAILED on commit 633c3f4 and must pass after the Block A fixes.
See docs/BUG_AND_RISK_REGISTER.md findings N1, N2, N6, N7, N9, N73, N74 and
docs/PRIORITISED_DEVELOPMENT_PLAN.md section 1 Block A.

The common defect this suite guards against is NOT a missing algorithm. In each
case the correct behaviour already existed somewhere in the codebase and was
undone, bypassed or never read one layer away:

A. N1  `dams.load_dams()` applied `.fillna(0.0)` to activity, undoing the
       NaN-as-missing convention `dam_reader.read_monitor` exists to establish.
       A 60-minute monitor fault was scored as a 60-minute sleep bout.
B. N2  `app.py` defined `_compute_msci_inline`, which bypassed the audited
       `compute_msci` and reinstated the asymptotic Spearman p = 0.0 at n = 3
       that the module was fixed to stop.
C. N74 `provenance.py` was never imported by anything, and stored only to
       `DataFrame.attrs`, which a CSV export does not preserve.
D. N6  `auto_recording._analyse_video` trained the background subtractor on
       3-channel colour and applied it to 1-channel grayscale. OpenCV 4.13
       raises nothing: it reports the whole frame as foreground, which destroys
       the early timepoints and reads stationary flies at the top of the tube as
       0% climbed -- a phenotype inversion.
   N9  the scoring line was a fixed fraction of the FULL FRAME height with no
       ROI at all, so any bench object of fly-like area was counted.
   N7  a requested timepoint beyond the end of the video vanished silently, with
       no row and no warning -- while `climbing_video.py` has emitted an explicit
       `{"usable": False, "reason": "video ends at ..."}` row all along.

Run:  python validation/validate_p0_block_a.py   (exit 0 = all passed)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import dams as D                      # noqa: E402
from modules import motor_decline as MD            # noqa: E402
from modules import provenance as PV               # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent

try:
    import cv2
    _CV2 = True
except Exception:                                   # pragma: no cover
    cv2 = None
    _CV2 = False


# ── helpers ───────────────────────────────────────────────────────────────────
def _check(name, got, expected, tol, results):
    ok = (got is not None) and np.isfinite(got) and abs(float(got) - expected) <= tol
    results.append((name, expected, round(float(got), 4) if got is not None and np.isfinite(got) else None, ok))
    return ok


def _bool_check(name, ok, expectation, got_str, results):
    results.append((name, expectation, str(got_str)[:40], bool(ok)))
    return bool(ok)


def _fault_frame(n_min=1440, nan_from=600, nan_len=60, activity=5.0):
    """One fly, continuously active, with a block of invalid monitor minutes.

    This is the exact scenario in the reader's own docstring: status != OK rows
    that `dam_reader` writes as NaN precisely so they are not read as stillness.
    """
    act = np.full(n_min, activity, dtype=float)
    act[nan_from:nan_from + nan_len] = np.nan
    return pd.DataFrame({
        "Genotype": "WT",
        "Treatment": "Vehicle",
        "Sex": "Male",
        "FlyID": "F1",
        "Time_min": np.arange(n_min, dtype=float),
        "Activity": act,
    })


def _synthetic_clip(path, seconds=5.0, fps=20.0, w=320, h=480,
                    n_flies=10, stationary_top=True):
    """A clip with a bright background and dark fly-sized blobs.

    `stationary_top` places the flies in the top fifth of the frame and leaves
    them there, which is the condition that matters: a healthy cohort that has
    climbed and stopped moving. A detector that needs motion reports 0% climbed
    for exactly the animals that performed best.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    rng = np.random.default_rng(7)
    xs = rng.integers(40, w - 40, n_flies)
    n_frames = int(round(seconds * fps))
    for i in range(n_frames):
        frame = np.full((h, w, 3), 235, dtype=np.uint8)
        # the tube: a slightly darker vertical band, as a real vial wall reads
        frame[40:h - 40, 60:w - 60] = 225
        for j, x in enumerate(xs):
            if stationary_top:
                y = 70 + (j % 5) * 7
            else:
                y = int(h - 80 - (i / max(n_frames - 1, 1)) * (h - 200))
            cv2.circle(frame, (int(x), int(y)), 4, (25, 25, 25), -1)
        vw.write(frame)
    vw.release()
    return path


# ── A. N1 -- a missing monitor minute is not sleep ────────────────────────────
def _block_a1(results):
    df = _fault_frame()

    long = D.load_dams(df)
    n_nan = int(np.count_nonzero(~np.isfinite(long["Activity"].to_numpy(float))))
    _check("A1 load_dams preserves the 60 invalid minutes as NaN", n_nan, 60, 0, results)

    pf = D.sleep_per_fly(long)
    _bool_check("A1 sleep_per_fly returns a row through load_dams",
                not pf.empty, "1 row", len(pf), results)
    if pf.empty:
        return
    r = pf.iloc[0]

    # The fly never stops moving. Before the fix the 60 NaN minutes were zeros,
    # which `_zero_run_bouts` scored as one 60-minute bout.
    _check("A1 a monitor fault is NOT scored as sleep",
           r["Total_sleep_min_per_day"], 0.0, 1e-9, results)
    _check("A1 a monitor fault creates no sleep bout",
           r["N_bouts_per_day"], 0.0, 1e-9, results)
    _check("A1 a monitor fault creates no longest bout",
           r["Longest_bout_min"], 0.0, 1e-9, results)

    # The missing time must be reported, not absorbed.
    _check("A1 Minutes_missing reports the fault", r["Minutes_missing"], 60, 0, results)
    _bool_check("A1 Coverage_frac is reported", "Coverage_frac" in pf.columns,
                "column present", list(pf.columns)[-3:], results)
    if "Coverage_frac" in pf.columns:
        _check("A1 Coverage_frac = 1380/1440", r["Coverage_frac"], 1380.0 / 1440.0, 1e-4, results)

    # Rates must be per OBSERVED day, not per elapsed day: counting the missing
    # minutes in the denominator trades sleep-from-a-fault for a diluted rate.
    # 1380 valid minutes at 5 counts each = 6900 counts over 0.958 observed days.
    _check("A1 activity is normalised to observed, not elapsed, time",
           r["Total_activity_per_day"], 6900.0 / (1380.0 / 1440.0), 1.0, results)

    # Session must survive, or a bout can be scored across a monitor restart.
    df_s = _fault_frame()
    df_s["Session"] = np.where(df_s["Time_min"] < 700, 2, 3)
    long_s = D.load_dams(df_s)
    _bool_check("A1 load_dams preserves Session", "Session" in long_s.columns,
                "Session kept", list(long_s.columns), results)

    # A genuine sleep bout must still be found: the fix must not suppress sleep.
    df_q = _fault_frame()
    df_q.loc[100:139, "Activity"] = 0.0          # a real 40-minute quiescence
    pf_q = D.sleep_per_fly(D.load_dams(df_q))
    _check("A1 a real 40-min quiescence is still scored as sleep",
           pf_q.iloc[0]["Longest_bout_min"], 40.0, 1e-9, results)


# ── B. N2 -- the audited MSCI is the only MSCI ─────────────────────────────────
def _block_a2(results):
    src = (_ROOT / "app.py").read_text(encoding="utf-8", errors="replace")
    _bool_check("A2 the inline MSCI bypass is gone from app.py",
                "_compute_msci_inline" not in src, "absent",
                src.count("_compute_msci_inline"), results)
    n_calls = src.count("compute_msci(")
    _bool_check("A2 app.py calls the audited compute_msci at >=2 sites",
                n_calls >= 2, ">=2", n_calls, results)

    # The module must accept what the UI actually hands it: an aliased lifespan
    # column, and a frame with no Treatment column. The inline version's one
    # genuinely useful feature was this tolerance; it belongs in the module.
    mdr = pd.DataFrame({
        "Genotype": ["WT", "WT", "WT"],
        "Treatment": ["Vehicle", "DrugA", "DrugB"],
        "Sex": ["Male", "Male", "Male"],
        "k": [0.010, 0.020, 0.030],
    })
    ls_alias = pd.DataFrame({
        "Genotype": ["WT", "WT", "WT"],
        "Treatment": ["Vehicle", "DrugA", "DrugB"],
        "Sex": ["Male", "Male", "Male"],
        "MedianSurvival_days": [60.0, 50.0, 40.0],
    })
    tab, merged = MD.compute_msci(mdr, ls_alias)
    _bool_check("A2 compute_msci accepts the MedianSurvival_days alias",
                not tab.empty, "1 row", len(tab), results)
    if tab.empty:
        return
    row = tab.iloc[0]

    # Perfect negative rank correlation over three groups. The exact two-sided
    # permutation p is 2/3! = 1/3; scipy's asymptotic p is 0.0, and that 0.0 is
    # what the UI printed before this fix.
    _check("A2 rho = -1 recovered", abs(float(row["MSCI_Spearman_rho"])), 1.0, 1e-9, results)
    pcol = "p_Spearman" if "p_Spearman" in tab.columns else "p_value"
    # p is reported to 3 significant figures (0.333), deliberately: an exact p
    # of 4.96e-05 printed at 4 decimal places becomes "0.0", which is the very
    # thing this path was fixed to stop reporting.
    _check("A2 exact permutation p = 1/3, not the asymptotic 0.0",
           row[pcol], 1.0 / 3.0, 5e-4, results)
    _check("A2 p_min_attainable = 1/3 at n = 3",
           row["p_min_attainable"], 1.0 / 3.0, 5e-4, results)
    _bool_check("A2 n = 3 is labelled underpowered by construction",
                "nderpowered" in str(row.get("Interpretation", "")),
                "Underpowered...", str(row.get("Interpretation", ""))[:38], results)
    _bool_check("A2 the ecological-correlation caveat is on the row",
                len(str(row.get("Caveat", "")).strip()) > 20,
                "non-empty Caveat", str(row.get("Caveat", ""))[:38], results)
    _bool_check("A2 no MSCI row reports p = 0.0",
                not bool((pd.to_numeric(tab[pcol], errors="coerce") == 0.0).any()),
                "no zero p", list(pd.to_numeric(tab[pcol], errors="coerce")), results)
    _bool_check("A2 the merged frame is returned for plotting",
                isinstance(merged, pd.DataFrame) and not merged.empty,
                "non-empty", len(merged) if merged is not None else None, results)

    # No Treatment column at all -- the merge key must degrade, not raise.
    mdr_nt = mdr.drop(columns=["Treatment"]).assign(k=[0.010, 0.020, 0.030])
    ls_nt = ls_alias.drop(columns=["Treatment"]).rename(
        columns={"MedianSurvival_days": "Median_Survival_days"})
    try:
        tab_nt, _ = MD.compute_msci(mdr_nt, ls_nt)
        ok_nt = isinstance(tab_nt, pd.DataFrame)
    except Exception as exc:                        # pragma: no cover
        ok_nt = False
        tab_nt = str(exc)
    _bool_check("A2 a frame without Treatment does not raise", ok_nt,
                "no exception", type(tab_nt).__name__, results)

    # The pooled cross-genotype row the inline version added is the strongest
    # form of the ecological fallacy available here and must not come back.
    _bool_check("A2 no cross-genotype 'ALL' row is manufactured",
                "ALL" not in set(tab["Genotype"].astype(str)),
                "no ALL row", sorted(set(tab["Genotype"].astype(str))), results)


# ── C. N74 -- provenance that survives an export ───────────────────────────────
def _block_a3(results):
    for fn in ("manifest", "write_manifest"):
        _bool_check(f"A3 provenance.{fn} exists", hasattr(PV, fn), "callable",
                    hasattr(PV, fn), results)
    if not (hasattr(PV, "manifest") and hasattr(PV, "write_manifest")):
        return

    df = pd.DataFrame({"Genotype": ["WT"], "PI": [42.0]})
    man = PV.manifest(inputs={"counts.csv": df}, parameters={"min_time": 10.0},
                      seeds={"numpy": 0})

    required = ("drophenix_version", "git_commit", "python", "platform",
                "dependencies", "parameters", "input_checksums", "timestamp_utc",
                "seeds")
    missing = [k for k in required if k not in man]
    _bool_check("A3 the manifest carries all 9 required items", not missing,
                "none missing", missing or "ok", results)
    _bool_check("A3 the input checksum is recorded",
                bool(man.get("input_checksums", {}).get("counts.csv")),
                "sha256 present", str(man.get("input_checksums"))[:34], results)
    _bool_check("A3 the numerical dependencies are recorded",
                {"numpy", "pandas", "scipy"} <= set(man.get("dependencies", {})),
                "numpy/pandas/scipy", sorted(man.get("dependencies", {}))[:3], results)

    # The point of the finding: it must survive leaving the process.
    with tempfile.TemporaryDirectory() as td:
        p = PV.write_manifest(Path(td) / "out.csv", man)
        _bool_check("A3 a sidecar file is written next to the export",
                    Path(p).exists(), "file exists", Path(p).name, results)
        import json
        back = json.loads(Path(p).read_text(encoding="utf-8"))
        _bool_check("A3 the manifest round-trips through the filesystem",
                    back.get("input_checksums") == man["input_checksums"],
                    "checksums match", back.get("timestamp_utc"), results)

    # And the app must actually use it -- an unimported safeguard is not one.
    src = (_ROOT / "app.py").read_text(encoding="utf-8", errors="replace")
    imported = ("from modules.provenance import" in src
                or "from modules import provenance" in src)
    _bool_check("A3 app.py actually imports provenance", imported,
                "real import", imported, results)
    _bool_check("A3 the export path attaches a manifest",
                "manifest" in src and "MANIFEST" in src,
                "manifest in export", "MANIFEST" in src, results)


# ── D. N6 / N9 / N7 -- the legacy climbing path ───────────────────────────────
def _block_a4(results):
    if not _CV2:
        _bool_check("A4 SKIPPED (cv2 absent)", True, "cv2", "absent", results)
        return

    from modules import auto_recording as AR

    with tempfile.TemporaryDirectory() as td:
        clip = _synthetic_clip(Path(td) / "stationary_top.mp4", seconds=5.0, fps=20.0)

        # N6: the channel mismatch marked the whole frame as foreground, which
        # destroyed the early timepoints. A foreground fraction near 1.0 is the
        # signature, so the module must report it.
        out = AR._analyse_video(str(clip), sample_pts=[1, 2, 3],
                                roi=(40, 440))
        _bool_check("A4 _analyse_video accepts an ROI", "error" not in out,
                    "no error", out.get("error", "ok"), results)
        rows = out.get("results", [])
        fracs = [r.get("fgmask_fraction") for r in rows
                 if isinstance(r, dict) and r.get("fgmask_fraction") is not None]
        _bool_check("A4 the foreground fraction is reported per timepoint",
                    len(fracs) >= 1, ">=1 value", fracs[:3], results)
        if fracs:
            _bool_check("A4 the frame is not wholly foreground (channel match)",
                        max(fracs) < 0.5, "<0.50", round(max(fracs), 3), results)

        # N9: without an ROI the line was a fraction of the full frame, so the
        # reading is not a biological measurement and must say so.
        out_noroi = AR._analyse_video(str(clip), sample_pts=[1])
        rows_nr = out_noroi.get("results", [])
        _bool_check("A4 without an ROI the result refuses to be a measurement",
                    bool(rows_nr) and rows_nr[0].get("usable") is False,
                    "usable False", rows_nr[0] if rows_nr else None, results)
        _bool_check("A4 the refusal names the missing ROI",
                    bool(rows_nr) and "roi" in str(rows_nr[0].get("reason", "")).lower(),
                    "reason mentions ROI",
                    rows_nr[0].get("reason") if rows_nr else None, results)

        # The scientific point of N6/N8: a cohort that climbed and then stopped
        # moving is invisible to a background subtractor. That must come back as
        # a FAILED MEASUREMENT, never as 0% climbed -- which is what the old
        # code reported, moving every such vial +100 PI points toward normal.
        _bool_check("A4 a stationary cohort is a failed measurement, not 0% climbed",
                    all(r.get("pct") is None for r in rows if not r.get("usable")),
                    "no pct on failures",
                    [r.get("pct") for r in rows], results)
        _bool_check("A4 the failure says detection failed rather than zero climbing",
                    any("not zero climbing" in str(r.get("reason", "")) for r in rows),
                    "'not zero climbing'",
                    rows[0].get("reason") if rows else None, results)

        # N7: a timepoint past the end of the clip must be reported as an
        # explicit unusable row, exactly as climbing_video.py already does.
        # Flies that MOVE, so the detectable timepoints are genuinely usable and
        # the only unusable row is the one past the end.
        clip_m = _synthetic_clip(Path(td) / "moving.mp4", seconds=5.0, fps=20.0,
                                 stationary_top=False)
        out_short = AR._analyse_video(str(clip_m), sample_pts=[2, 4, 120],
                                      roi=(40, 440))
        rs = out_short.get("results", [])
        _check("A4 every requested timepoint yields a row", len(rs), 3, 0, results)
        past = [r for r in rs if isinstance(r, dict) and r.get("t_s") == 120]
        _bool_check("A4 the timepoint past the end is marked unusable",
                    bool(past) and past[0].get("usable") is False,
                    "usable False", past[0].get("reason") if past else None, results)
        _bool_check("A4 the reason names the video length",
                    bool(past) and "ends at" in str(past[0].get("reason", "")),
                    "'ends at ...'", past[0].get("reason") if past else None, results)
        _bool_check("A4 an unusable timepoint contributes no percentage",
                    bool(past) and past[0].get("pct") is None,
                    "pct None", past[0].get("pct") if past else None, results)

        summ = out_short.get("summary", {})
        _check("A4 the summary reports every requested timepoint",
               summ.get("n_requested"), 3, 0, results)
        _bool_check("A4 the summary counts the unusable timepoint",
                    int(summ.get("n_unusable", 0)) >= 1, ">=1",
                    summ.get("n_unusable"), results)
        _bool_check("A4 moving flies at detectable timepoints ARE usable",
                    any(r.get("usable") for r in rs), "at least one usable",
                    [r.get("usable") for r in rs], results)

    # video_analyzer must not drop a requested timepoint either.
    from modules.video_analyzer import VideoAnalyzer
    from modules.fly_detector import VialROI
    with tempfile.TemporaryDirectory() as td:
        clip = _synthetic_clip(Path(td) / "short.mp4", seconds=4.0, fps=20.0)
        va = VideoAnalyzer()
        vials = [VialROI(vial_id=1, x=60, y=40, w=200, h=400, expected_n=10)]
        df = va.analyze(str(clip), vials, sample_timepoints=[1, 2, 60])
        tps = sorted(set(pd.to_numeric(df["Time"], errors="coerce").dropna().astype(int))) \
            if not df.empty and "Time" in df.columns else []
        _bool_check("A4 video_analyzer returns a row for a timepoint past the end",
                    60 in tps, "60 present", tps, results)
        if not df.empty and "QC_flag" in df.columns and 60 in tps:
            qc = str(df.loc[pd.to_numeric(df["Time"]) == 60, "QC_flag"].iloc[0])
            _bool_check("A4 video_analyzer flags it rather than reporting a count",
                        "video_end" in qc or "unusable" in qc, "video_end flag", qc,
                        results)


def main() -> int:
    results = []
    for label, fn in (("A1", _block_a1), ("A2", _block_a2),
                      ("A3", _block_a3), ("A4", _block_a4)):
        try:
            fn(results)
        except Exception as exc:
            # A block that raises is a failure, not a crash of the suite: the
            # whole point is to record the pre-fix state of every block.
            results.append((f"{label} block raised {type(exc).__name__}",
                            "no exception", str(exc)[:38], False))

    print("=" * 96)
    print("DroPhenix P0 Block A regression suite -- the four bypassed safeguards")
    print("  N1 load_dams zero-fill | N2 inline MSCI bypass | N74 provenance | N6/N9/N7 legacy video")
    print("=" * 96)
    print(f"{'Check':<62}{'Expected':>16}{'Engine':>12}{'Pass':>6}")
    print("-" * 96)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<62}{str(exp):>16}{str(got):>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 96)
    print(f"{len(results)} checks")
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
