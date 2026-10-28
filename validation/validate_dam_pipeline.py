"""
Validation suite for modules/dam_pipeline.py, dam_figures.py and dam_report.py.

WHAT IS BEING VALIDATED
-----------------------
The pipeline is the analysis of docs/DAM_RESULTS_REPORT.docx made reusable, so
the strongest test available is that it REPRODUCES that analysis on the same
raw files with no hand-holding: the same windows, the same flies, the same
outliers, and the same effect sizes and p-values to floating-point precision.
Beyond that, the parts that must generalise are tested on synthetic designs
whose names it has never seen.

  A. Window proposal on synthetic light traces: a failed day, a partial day, a
     set-up day with flicker and a gap are each excluded for the stated reason,
     and the surviving run of days is the one a person would choose.
  B. Design detection on synthetic channel maps with unfamiliar names: a
     contrast is monitor-clean exactly when the map says so; reference levels
     are picked from the names (control / vehicle) and can be overridden.
  C. The pipeline on the laboratory's day-25 files reproduces the report's
     contrast tables: Wt vs Htt, withaferin pooled (clean and all), the
     no-outlier variant, and the within-Monitor-2 check -- g, p and q identical.
  D. The same for the day-15 raw files, and the two-run comparison reproduces
     contrasts_day.csv; the pairing test says unpaired, as found by hand.
  E. Refusal: a 60-min export is rejected as not sleep-scoreable; a recording
     with no usable day yields no window rather than a guess.
  F. The report builds from a run and verify() traces every numeric cell.
  G. Every figure function returns a figure for a run with unfamiliar names.

Run: ./.venv/Scripts/python.exe validation/validate_dam_pipeline.py
Exit 0 = all pass. Real-data tests SKIP when the files are absent.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from modules import dam_pipeline as P     # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []
DAM = Path(r"D:/DroPhenix/Testing data/DAM")
CMAP = DAM / "25th day" / "channel_map_25thday.csv"
D25 = [DAM / "25th day" / "Monitor_1_25th day.txt", DAM / "25th day" / "Monitor_2_25th day.txt"]
D15 = [DAM / "15th day" / "Monitor_1.txt", DAM / "15th day" / "Monitor_2.txt"]
D15H = [DAM / "25th day" / "15th day" / "Monitor_1_15th day.txt", DAM / "25th day" / "15th day" / "Monitor_2_15th day.txt"]
REP = Path(__file__).resolve().parents[1] / "results" / "dam_report" / "tables"


def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def skip(name, why):
    print(f"  [SKIP] {name}  --  {why}"); SKIPPED.append(name)


# ─────────────────────────────────────────────────────────────────────────────
def synth_lab(days=6, light_h=14.0, start="2026-01-01 09:00", n_ch=8, seed=0,
              fail_day=None, flicker_day=None, gap_day=None, monitors=("MonA", "MonB"),
              stuck_monitor=None, genos=("ctrl", "mutX"), trts=("vehicle", "drugQ")):
    """A labelled long frame like `load_run` returns, with known faults."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    n = int(days * 1440)
    ts = pd.date_range(t0, periods=n, freq="min")
    hour = ts.hour + ts.minute / 60.0
    light = ((hour >= 7.0) & (hour < 7.0 + light_h)).astype(int)
    date = ts.normalize()
    if fail_day is not None:                       # lamp stays off all day
        light[date == pd.Timestamp(fail_day)] = 0
    if flicker_day is not None:                    # six 1-min toggles one evening
        base = np.where(date == pd.Timestamp(flicker_day))[0]
        for k in range(6):
            light[base[0] + 21 * 60 + 20 * k: base[0] + 21 * 60 + 20 * k + 2] = 1
    keep = np.ones(n, bool)
    if gap_day is not None:                        # 3-hour recording gap
        base = np.where(date == pd.Timestamp(gap_day))[0]
        keep[base[0] + 600: base[0] + 780] = False
    rows = []
    ch_id = 0
    for mi, mon in enumerate(monitors):
        L = light.copy()
        if mon == stuck_monitor:
            L[:] = 1
        for c in range(1, n_ch + 1):
            ch_id += 1
            g = genos[(ch_id - 1) // 4 % len(genos)]
            tr = trts[(ch_id - 1) // 2 % len(trts)]
            sx = ["M", "F"][(ch_id - 1) % 2]
            rate = 0.08 if light[0] else 0.02
            act = rng.poisson(np.where(light == 1, 0.5, 0.1) * (1.6 if g == genos[1] else 1.0))
            rows.append(pd.DataFrame({"Monitor": mon, "Channel": c, "Timestamp": ts[keep],
                                      "Activity": act[keep].astype(float), "Light": L[keep],
                                      "Status": 1, "Session": 1, "Valid": True,
                                      "Genotype": g, "Sex": sx, "Treatment": tr}))
    lab = pd.concat(rows, ignore_index=True)
    return lab


def test_A_window():
    print("\nA. The window proposal excludes what it should, for the stated reason")
    # Days: 1 partial, 2 flicker, 3 lamp failed, 4-6 clean, 7 gap, 8 partial.
    lab = synth_lab(days=8, fail_day="2026-01-03", flicker_day="2026-01-02", gap_day="2026-01-07")
    q = P.quality_report(lab)
    d = q["days"].set_index("date")
    ok("A1: photoperiod measured as 14 h", abs((q["photoperiod_min"] or 0) - 840) < 5, f"{q['photoperiod_min']}")
    ok("A2: the partial first day is excluded as 'partial day'", "partial" in d.loc["2026-01-01", "exclude_reason"],
       d.loc["2026-01-01", "exclude_reason"])
    ok("A3: the flicker day is excluded for flicker", "toggled" in d.loc["2026-01-02", "exclude_reason"],
       d.loc["2026-01-02", "exclude_reason"])
    ok("A4: the failed-lamp day is excluded for light hours", "light 0.0 h" in d.loc["2026-01-03", "exclude_reason"],
       d.loc["2026-01-03", "exclude_reason"])
    ok("A5: the gap day is excluded for the gap", "gap of" in d.loc["2026-01-07", "exclude_reason"],
       d.loc["2026-01-07", "exclude_reason"])
    ok("A6: the proposed window is the longest CONSECUTIVE run of clean days",
       q["window"] == (pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-07")),
       f"{q['window'][0].date()}..{(q['window'][1] - pd.Timedelta(1, "D")).date()}" if q["window"] else "None")
    lab2 = synth_lab(days=5, stuck_monitor="MonB")
    q2 = P.quality_report(lab2)
    ok("A7: a stuck light sensor is named and the cycling monitor supplies ZT",
       q2["cycling"] == "MonA" and q2["agreement"] < 0.7 and any("stuck" in f for f in q2["faults"]),
       f"cycling={q2['cycling']}, agreement={q2['agreement']:.2f}")


def test_B_design():
    print("\nB. Design detection with unfamiliar names")
    lab = synth_lab(days=4, genos=("ctrl", "mutX"), trts=("vehicle", "drugQ"))
    tidy = P.DR.to_dams_long(lab, drop_dead=False)
    d = P.Design().resolve(tidy)
    ok("B1: reference genotype picked from the name ('ctrl')", d.ref_genotype == "ctrl", d.ref_genotype)
    ok("B2: reference treatment picked from the name ('vehicle')", d.ref_treatment == "vehicle", d.ref_treatment)
    d2 = P.Design(ref_genotype="mutX", ref_treatment="drugQ").resolve(tidy)
    ok("B3: references can be overridden", (d2.ref_genotype, d2.ref_treatment) == ("mutX", "drugQ"))
    # In synth_lab channels 1-4 of MonA are ctrl, 5-8 mutX; MonB the same -> every
    # genotype in both monitors -> clean. Now build a confounded map.
    run = P.analyse_run(lab, "s", d)
    g = run.contrasts["genotype"]; g = g[g.analysis == "primary"]
    ok("B4: interleaved layout -> genotype contrast is monitor-clean", bool(g.monitor_clean.all()))
    lab_conf = synth_lab(days=4, n_ch=4)      # 4 channels per monitor: MonA all ctrl, MonB all mutX
    run_c = P.analyse_run(lab_conf, "c", d)
    gc = run_c.contrasts["genotype"]; gc = gc[gc.analysis == "primary"]
    ok("B5: one genotype per monitor -> genotype contrast is flagged confounded", bool((~gc.monitor_clean).all()))
    tpc = run_c.contrasts["treatment_pooled_clean"]; tpa = run_c.contrasts["treatment_pooled_all"]
    ok("B6: treatment within a monitor stays clean even when genotype is confounded",
       (not tpc.empty) and bool(tpc[tpc.analysis == "primary"].monitor_clean.all()))
    ok("B7: the stratum is sex; van Elteren is the primary test",
       d.stratify_col == "Sex" and "van Elteren" in g.test.iloc[0], g.test.iloc[0])


def _cmp(pipe_df, csv, filt, cols=("hedges_g", "p", "q_BH", "significant")):
    rep = pd.read_csv(REP / csv); rep = rep[rep.analysis == "primary"]; rep = rep[filt(rep)]
    pp = pipe_df[pipe_df.analysis == "primary"]; pp = pp[filt(pp)]
    m = rep.merge(pp, on="metric", suffixes=("_rep", "_pipe"))
    out = {c: float((m[f"{c}_rep"].astype(float) - m[f"{c}_pipe"].astype(float)).abs().max()) for c in cols}
    out["n"] = len(m); out["n_rep"] = len(rep); out["n_pipe"] = len(pp)
    return out


def test_C_D_real():
    print("\nC/D. The pipeline reproduces the report on the laboratory's own raw files")
    if not (CMAP.exists() and all(f.exists() for f in D25) and all(f.exists() for f in D15) and REP.exists()):
        skip("real-data reproduction", "monitor files or report tables not present"); return None
    r25 = P.run_from_files([str(f) for f in D25], str(CMAP), "day 25")
    r15 = P.run_from_files([str(f) for f in D15], str(CMAP), "day 15")
    ok("C1: day-25 window proposed = 23..30 Aug (the hand-chosen clean window)",
       r25.window == (pd.Timestamp("2026-08-23"), pd.Timestamp("2026-08-31")),
       f"{r25.window[0].date()}..{r25.window[1].date()}")
    ok("C2: day-25 flies = 55, outliers = 4, photoperiod 12.0 h",
       len(r25.per_fly) == 55 and int(r25.per_fly.Outlier.sum()) == 4 and abs(r25.photoperiod_min - 721) < 2,
       f"{len(r25.per_fly)}, {int(r25.per_fly.Outlier.sum())}, {r25.photoperiod_min/60:.2f}")
    ok("C3: reference levels resolved to Wt / Untreated / Sex without being told",
       (r25.design.ref_genotype, r25.design.ref_treatment, r25.design.stratify_col) == ("Wt", "Untreated", "Sex"))
    wh = lambda d: (d.ref == "Wt") & (d.cmp == "Htt")
    for label, df, csv, filt in [
        ("C4: Wt vs Htt, day 25", r25.contrasts["genotype"], "contrasts_genotype_25thday.csv", wh),
        ("C5: withaferin pooled, monitor-clean, day 25", r25.contrasts["treatment_pooled_clean"], "contrasts_treatment_pooled_clean_25thday.csv", lambda d: d.metric.notna()),
        ("C6: withaferin pooled, all blocks, day 25", r25.contrasts["treatment_pooled_all"], "contrasts_treatment_pooled_all_25thday.csv", lambda d: d.metric.notna()),
        ("C7: Wt vs Htt without outliers, day 25", r25.contrasts_noout["genotype"], "contrasts_genotype_25thday_nooutliers.csv", wh),
        ("D1: Wt vs Htt, day 15", r15.contrasts["genotype"], "contrasts_genotype_15thday.csv", wh),
        ("D2: withaferin pooled, monitor-clean, day 15", r15.contrasts["treatment_pooled_clean"], "contrasts_treatment_pooled_clean_15thday.csv", lambda d: d.metric.notna()),
    ]:
        c = _cmp(df, csv, filt)
        ok(f"{label}: g, p, q and significance identical to the report",
           c["n"] == 11 and c["hedges_g"] < 1e-9 and c["p"] < 1e-12 and c["q_BH"] < 1e-9 and c["significant"] == 0,
           f"n={c['n']}, max|dg|={c['hedges_g']:.1e}, max|dp|={c['p']:.1e}, max|dq|={c['q_BH']:.1e}")
    m2 = r25.contrasts["genotype_within_monitor"]
    m2 = m2[m2.stratum.str.contains("Monitor_2") & m2.stratum.str.contains("Withaferin") & (m2.ref == "CyO") & (m2.cmp == "Htt")]
    c = _cmp(m2, "contrasts_genotype_monitor2_25thday.csv", lambda d: d.metric.notna(), cols=("hedges_g", "p"))
    ok("C8: the generic within-monitor family contains the hand-built Monitor-2 check, with identical g and p",
       c["n"] == 11 and c["hedges_g"] < 1e-9 and c["p"] < 1e-12, f"n={c['n']}, max|dg|={c['hedges_g']:.1e}, max|dp|={c['p']:.1e}")
    ok("D3: day-15 window proposed = 12..14 Aug; 58 flies; 6 outliers; 14.0 h light",
       r15.window == (pd.Timestamp("2026-08-12"), pd.Timestamp("2026-08-15")) and len(r15.per_fly) == 58
       and int(r15.per_fly.Outlier.sum()) == 6 and abs(r15.photoperiod_min - 842) < 3,
       f"{r15.window[0].date()}..{r15.window[1].date()}, {len(r15.per_fly)}, {int(r15.per_fly.Outlier.sum())}, {r15.photoperiod_min/60:.2f} h")
    ok("D4: the stuck light sensor on day 15 is reported and Monitor_1 is used for ZT",
       r15.quality["cycling"] == "Monitor_1" and any("stuck" in f for f in r15.quality["faults"]), r15.quality["cycling"])
    cmp = P.compare_runs(r15, r25)
    tp = cmp["timepoint"].assign(stratum=cmp["timepoint"].stratum.str.replace(" stratified", " sex-stratified"))
    c = _cmp(tp, "contrasts_day.csv", lambda d: d.stratum.str.startswith("Htt Withaferin"))
    ok("D5: day 15 -> day 25 (Htt + withaferin) identical to the report's contrasts_day.csv",
       c["n"] == 15 and c["hedges_g"] < 1e-9 and c["p"] < 1e-12 and c["q_BH"] < 1e-9,
       f"n={c['n']}, max|dg|={c['hedges_g']:.1e}, max|dp|={c['p']:.1e}")
    ok("D6: tubes are detected as UNPAIRED across the two runs", not cmp["pairing"].attrs.get("paired"),
       f"rho = {cmp['pairing'].within_group_rho.tolist()}")
    return r15, r25, cmp


def test_E_refusals():
    print("\nE. Refusals")
    if all(f.exists() for f in D15H) and CMAP.exists():
        lab = P.load_run([str(f) for f in D15H], str(CMAP))
        q = P.quality_report(lab)
        ok("E1: the 60-min export is flagged NOT sleep-scoreable", not q["sleep_scoreable"], f"bin={q['bin_min']}")
    else:
        skip("E1 hourly export", "file not present")
    lab = synth_lab(days=2, gap_day="2026-01-02")        # day 1 partial, day 2 has a 3-h gap -> nothing usable
    q = P.quality_report(lab)
    ok("E2: a recording with no usable day proposes NO window rather than a guess", q["window"] is None,
       str(q["window"]))
    lab = synth_lab(days=3)                              # day 1 partial, days 2-3 complete -> a 2-day window
    q = P.quality_report(lab)
    ok("E3: a window shorter than 3 days is allowed but carries a warning",
       q["window"] is not None and any("Only 2 consecutive usable" in f for f in q["faults"]),
       "; ".join(q["faults"]))


def test_F_report(runs):
    print("\nF. The report builds and every number in it traces to the tables")
    from modules import dam_report as R, dam_figures as F
    import io
    if runs is None:
        lab = synth_lab(days=4)
        run = P.analyse_run(lab, "synthetic", P.Design())
        run_b = cmp = None
    else:
        run, run_b, cmp = runs
    figs = {}
    b = io.BytesIO(); F.headline(run, run_b, cmp).savefig(b, format="png", dpi=80); figs["Figure 1."] = b.getvalue()
    with tempfile.TemporaryDirectory() as td:
        out = R.build(run, str(Path(td) / "r.docx"), run_b=run_b, cmp=cmp, figures=figs)
        ok_, rep = R.verify(out, run, run_b, cmp)
        ok("F1: report builds and verify() traces every numeric table cell", ok_, rep)


def test_G_figures():
    print("\nG. Every figure function works on a design with unfamiliar names")
    from modules import dam_figures as F
    lab = synth_lab(days=4, genos=("ctrl", "mutX", "mutY"), trts=("vehicle", "drugQ"), n_ch=12)
    run = P.analyse_run(lab, "synthetic", P.Design())
    lab2 = synth_lab(days=3, genos=("ctrl", "mutX", "mutY"), trts=("vehicle", "drugQ"), n_ch=12, seed=1, start="2026-02-01 09:00")
    run2 = P.analyse_run(lab2, "synthetic B", run.design)
    cmp = P.compare_runs(run, run2)
    n = 0
    for name, fn in [("headline", lambda: F.headline(run, run2, cmp)),
                     ("profiles by genotype", lambda: F.profiles(run, by="genotype")),
                     ("profiles by treatment", lambda: F.profiles(run, by="treatment")),
                     ("profiles by run", lambda: F.profiles(run, by="run", run_b=run2)),
                     ("bars", lambda: F.bars(run, ["Total_sleep_min_per_day", "Total_activity_per_day"])),
                     ("age bars", lambda: F.age_bars(run, run2, cmp)),
                     ("outlier map", lambda: F.outlier_map(run))]:
        try:
            fig = fn(); n += 1
            import matplotlib.pyplot as plt; plt.close(fig)
        except Exception as e:
            ok(f"G: {name}", False, repr(e)[:120])
    ok("G1: all seven figure functions returned a figure for ctrl/mutX/mutY x vehicle/drugQ", n == 7, f"{n} of 7")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/dam_pipeline.py (raw files -> verified report)")
    print("=" * 78)
    test_A_window()
    test_B_design()
    runs = test_C_D_real()
    test_E_refusals()
    test_F_report(runs)
    test_G_figures()
    print("\n" + "=" * 78)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}): {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL DAM-PIPELINE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
