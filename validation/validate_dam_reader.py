"""
Validation suite for modules/dam_reader.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
Synthetic DAMSystem monitor files are written with KNOWN contents, so the
correct answer is fixed by construction. Each test reproduces a fault that was
present in the first real monitor files this module was written against, because
every one of them is silent: the analysis completes and returns numbers.

A. Exact parsing: counts, timestamps, channels and light recovered byte for byte.
B. Status rows are MISSING, not zero -- and, decisively, do not fabricate sleep.
C. Monitor restarts (index resets) are detected and labelled.
D. Dead and empty tubes are identified rather than scored as perfect sleepers.
E. Light regime is classified, and free-running period is REFUSED under LD and
   refused when the recording is too short.
F. The channel map governs group labels; unlabelled and excluded tubes are
   dropped rather than silently pooled.
G. Plausibility: implausible sleep and a reversed male/female sleep difference
   are flagged against published reference ranges.

Run: ./.venv/Scripts/python.exe validation/validate_dam_reader.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import dam_reader as DR  # noqa: E402
from modules import dams as DM  # noqa: E402

FAILURES: list[str] = []
SEED = 20260804


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def write_monitor(path, counts, start=datetime(2026, 6, 27, 8, 0, 0),
                  light=None, status=None, index=None, monitor_no=1):
    """
    Write a DAMSystem-format file from an explicit (n_minutes x 32) count array.

    Writing the file rather than mocking the reader is deliberate: the format,
    not the parsing logic, is what breaks in practice.
    """
    n = counts.shape[0]
    if light is None:
        light = np.array([(1 if 0 <= ((start + timedelta(minutes=i)).hour) < 12 else 0)
                          for i in range(n)])
    if status is None:
        status = np.ones(n, dtype=int)
    if index is None:
        index = np.arange(1, n + 1)
    lines = []
    for i in range(n):
        t = start + timedelta(minutes=i)
        row = [str(index[i]), t.strftime("%d %b %y"), t.strftime("%H:%M:%S"),
               str(int(status[i])), "0", str(monitor_no), "0", "Ct", "0",
               str(int(light[i]))]
        row += [str(int(c)) for c in counts[i]]
        lines.append("\t".join(row))
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def _active(n_min, rng, level=3.0, n_ch=32):
    """Counts for flies that are awake and moving most of the time."""
    return rng.poisson(level, size=(n_min, n_ch))


# ─────────────────────────────────────────────────────────────────────────────
def test_A_exact_parsing() -> None:
    print("\nA. Exact parsing of a known monitor file")
    tmp = Path(tempfile.mkdtemp(prefix="dam_"))
    try:
        rng = np.random.default_rng(SEED)
        n = 300
        counts = rng.integers(0, 9, size=(n, 32))
        f = write_monitor(tmp / "M.txt", counts)
        long = DR.read_monitor(f, monitor="M")

        check("row count = minutes x channels",
              len(long) == n * 32, f"{len(long)} vs {n * 32}")
        wide = long.pivot(index="Timestamp", columns="Channel", values="Activity")
        check("every count recovered exactly",
              np.array_equal(wide.to_numpy(), counts.astype(float)),
              f"max |diff| = {np.abs(wide.to_numpy() - counts).max():.0f}")
        check("channels numbered 1..32",
              sorted(long["Channel"].unique()) == list(range(1, 33)), "")
        check("timestamps are one minute apart",
              set(long.drop_duplicates('Timestamp')["Timestamp"].diff()
                  .dropna().dt.total_seconds()) == {60.0}, "")
        check("light column read", long["Light"].notna().all(), "")

        bad = tmp / "short.txt"
        bad.write_text("1\t27 Jun 26\t08:00:00\t1\t0\n")
        raised = False
        try:
            DR.read_monitor(bad)
        except ValueError as e:
            raised = "does not look like a raw monitor file" in str(e)
        check("a file with too few columns is rejected with a clear message", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_B_status_rows_are_missing_not_zero() -> None:
    print("\nB. Status rows are missing, not zero  [they would otherwise fabricate sleep]")
    tmp = Path(tempfile.mkdtemp(prefix="dam_"))
    try:
        rng = np.random.default_rng(SEED + 1)
        n = 24 * 60
        counts = _active(n, rng, level=4.0)     # a continuously active fly
        status = np.ones(n, dtype=int)
        # ten minutes of monitor fault in the middle; the monitor writes zeros
        status[600:610] = 50
        counts[600:610, :] = 0
        f = write_monitor(tmp / "M.txt", counts, status=status)

        long = DR.read_monitor(f, monitor="M")
        n_nan = int(long["Activity"].isna().sum())
        check("faulty minutes become NaN",
              n_nan == 10 * 32, f"{n_nan} NaN cells (expected {10 * 32})")

        raw = DR.read_monitor(f, monitor="M", drop_bad_status=False)
        check("the raw zeros are still available for inspection",
              int(raw["Activity"].isna().sum()) == 0, "")

        # The decisive test: does the fault create sleep?
        cmap = pd.DataFrame([{"Monitor": "M", "Channel": c, "Genotype": "WT",
                              "Sex": "Male", "Age": "young"} for c in range(1, 33)])
        tidy_ok = DR.to_dams_long(DR.apply_channel_map(long, cmap), drop_dead=False)
        tidy_bad = DR.to_dams_long(DR.apply_channel_map(raw, cmap), drop_dead=False)
        s_ok = DM.sleep_per_fly(tidy_ok)["Total_sleep_min_per_day"].mean()
        s_bad = DM.sleep_per_fly(tidy_bad)["Total_sleep_min_per_day"].mean()
        print(f"      sleep with faults as MISSING: {s_ok:.1f} min/day")
        print(f"      sleep with faults as ZERO   : {s_bad:.1f} min/day")
        check("reading faults as zero fabricates sleep in an always-active fly",
              s_bad > s_ok + 5,
              f"{s_bad:.1f} vs {s_ok:.1f} min/day from ten faulty minutes")
        check("treating them as missing keeps an active fly nearly sleepless",
              s_ok < 30, f"{s_ok:.1f} min/day")

        pf = DM.sleep_per_fly(tidy_ok)
        check("missing minutes are counted and reported",
              "Minutes_missing" in pf.columns and int(pf["Minutes_missing"].iloc[0]) == 10,
              f"{pf['Minutes_missing'].iloc[0] if 'Minutes_missing' in pf else 'absent'}")
        check("waking activity survives missing data (was NaN for every fly)",
              pf["Waking_activity"].notna().all() and (pf["Waking_activity"] > 0).all(),
              f"{pf['Waking_activity'].isna().sum()} NaN of {len(pf)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_C_restart_detection() -> None:
    print("\nC. Monitor restarts are detected")
    tmp = Path(tempfile.mkdtemp(prefix="dam_"))
    try:
        rng = np.random.default_rng(SEED + 2)
        n = 600
        counts = _active(n, rng)
        idx = np.concatenate([np.arange(1, 401), np.arange(1, n - 399)])
        f = write_monitor(tmp / "M.txt", counts, index=idx)
        long = DR.read_monitor(f, monitor="M")
        check("two sessions identified",
              long["Session"].nunique() == 2, f"{long['Session'].nunique()}")
        a = DR.assess_recording(long)
        check("the restart is reported to the user",
              a.n_sessions == 2 and any("restarted" in w for w in a.warnings),
              f"{len(a.session_breaks)} break(s) recorded")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_D_dead_channels() -> None:
    print("\nD. Dead and empty tubes are identified")
    tmp = Path(tempfile.mkdtemp(prefix="dam_"))
    try:
        rng = np.random.default_rng(SEED + 3)
        n = 24 * 60
        counts = _active(n, rng)
        counts[:, 4] = 0          # empty tube
        counts[:, 9] = 0          # a second empty tube
        f = write_monitor(tmp / "M.txt", counts)
        long = DR.read_monitor(f, monitor="M")
        a = DR.assess_recording(long, dead_threshold=50.0)
        dead = a.dead_channels.get("M", [])
        check("both silent channels flagged", set(dead) == {5, 10}, f"flagged {dead}")
        check("the consequence is explained, not just the fact",
              any("maximal sleep" in w for w in a.warnings), "")

        cmap = pd.DataFrame([{"Monitor": "M", "Channel": c, "Genotype": "WT",
                              "Sex": "Male", "Age": "young"} for c in range(1, 33)])
        tidy = DR.to_dams_long(DR.apply_channel_map(long, cmap), drop_dead=True)
        check("dead tubes are excluded from the analysis table",
              tidy["FlyID"].nunique() == 30, f"{tidy['FlyID'].nunique()} of 32 kept")

        # A total-activity threshold CANNOT catch a fly that died part-way: its
        # total is healthy and only the shape of the record betrays it. This is
        # the failure mode that matters most in an ageing experiment, because
        # every post-death minute is scored as sleep.
        counts2 = _active(n, rng)
        counts2[400:, 7] = 0                     # channel 8 dies after ~7 hours
        f2 = write_monitor(tmp / "M2.txt", counts2)
        long2 = DR.read_monitor(f2, monitor="M2")
        a2 = DR.assess_recording(long2, dead_threshold=50.0)
        check("a mid-recording death is INVISIBLE to a total-activity threshold",
              8 not in a2.dead_channels.get("M2", []),
              f"total-activity flagged {a2.dead_channels.get('M2', [])}")
        rep = DR.death_report(long2, silence_h=12.0)
        dead_ch = sorted(rep["table"].loc[rep["table"]["likely_dead"], "Channel"])
        check("death detection finds it from the shape of the record",
              dead_ch == [8], f"flagged channels {dead_ch}")
        check("the consequence for sleep is stated",
              any("scored as sleep" in f for f in rep["flags"]), "")
        trunc = DR.truncate_at_death(long2, silence_h=12.0)
        kept8 = trunc[trunc.Channel == 8]
        check("truncation keeps the living hours and drops the rest",
              0 < len(kept8) < len(long2[long2.Channel == 8]),
              f"{len(kept8)} of {len(long2[long2.Channel == 8])} minutes kept for channel 8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_E_period_refusal() -> None:
    print("\nE. Circadian period is refused when it cannot be measured")
    tmp = Path(tempfile.mkdtemp(prefix="dam_"))
    try:
        rng = np.random.default_rng(SEED + 4)

        # (i) Light-dark, plenty long: entrained, so period is meaningless
        n = 7 * 24 * 60
        f = write_monitor(tmp / "ld.mp4.txt", _active(n, rng))
        a = DR.assess_recording(DR.read_monitor(f, monitor="M"))
        check("a 7-day LD recording is classified LD",
              a.light_regime == "LD", a.light_regime)
        check("period is refused under LD even with 7 days of data",
              not a.can_analyse_period and "entrained" in a.period_refusal,
              a.period_refusal[:90])

        # (ii) Constant darkness, long enough: period is allowed
        f2 = write_monitor(tmp / "dd.txt", _active(n, rng), light=np.zeros(n, int))
        a2 = DR.assess_recording(DR.read_monitor(f2, monitor="M"))
        check("constant darkness is classified DD", a2.light_regime == "DD", a2.light_regime)
        check("period is allowed in DD with 7 days",
              a2.can_analyse_period, a2.period_refusal[:80])

        # (iii) Constant darkness but far too short
        n3 = 40 * 60
        f3 = write_monitor(tmp / "short.txt", _active(n3, rng), light=np.zeros(n3, int))
        a3 = DR.assess_recording(DR.read_monitor(f3, monitor="M"))
        check("period is refused when the recording is under 3 days",
              not a3.can_analyse_period and "cycles" in a3.period_refusal,
              f"{a3.n_days:.1f} days")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_F_channel_map() -> None:
    print("\nF. The channel map governs labels; unlabelled tubes are dropped")
    tmp = Path(tempfile.mkdtemp(prefix="dam_"))
    try:
        rng = np.random.default_rng(SEED + 5)
        n = 12 * 60
        f = write_monitor(tmp / "M.txt", _active(n, rng))
        long = DR.read_monitor(f, monitor="M")

        rows = []
        for c in range(1, 33):
            if c <= 8:
                rows.append({"Monitor": "M", "Channel": c, "Genotype": "WT",
                             "Sex": "Male", "Age": "young", "Exclude": ""})
            elif c <= 16:
                rows.append({"Monitor": "M", "Channel": c, "Genotype": "TDP43",
                             "Sex": "Female", "Age": "old", "Exclude": ""})
            elif c <= 20:
                rows.append({"Monitor": "M", "Channel": c, "Genotype": "WT",
                             "Sex": "Male", "Age": "young", "Exclude": "yes"})
            else:
                rows.append({"Monitor": "M", "Channel": c, "Genotype": "",
                             "Sex": "", "Age": "", "Exclude": ""})
        cmap = pd.DataFrame(rows)

        lab = DR.apply_channel_map(long, cmap)
        kept = sorted(lab["Channel"].unique())
        check("only labelled, non-excluded channels survive",
              kept == list(range(1, 17)), f"kept {kept[:4]}...{kept[-2:]} ({len(kept)})")
        check("group labels attached correctly",
              set(lab.loc[lab.Channel <= 8, "Genotype"]) == {"WT"}
              and set(lab.loc[lab.Channel > 8, "Genotype"]) == {"TDP43"}, "")

        tidy = DR.to_dams_long(lab, drop_dead=False)
        check("one tube becomes one FlyID",
              tidy["FlyID"].nunique() == 16, f"{tidy['FlyID'].nunique()}")
        check("the schema dams.py needs is produced",
              {"Genotype", "Treatment", "Sex", "FlyID", "Time_min",
               "Activity"}.issubset(tidy.columns), f"{sorted(tidy.columns)[:6]}")

        tmpl = DR.channel_map_template(["M1", "M2"])
        check("a blank template is offered for both monitors",
              len(tmpl) == 64 and set(tmpl["Monitor"]) == {"M1", "M2"}, f"{len(tmpl)} rows")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_G_plausibility() -> None:
    print("\nG. Implausible results are flagged against published ranges")
    # Barely-moving flies: sleep will come out near the ceiling.
    quiet = pd.DataFrame({
        "Genotype": ["WT"] * 20, "Sex": ["Male"] * 10 + ["Female"] * 10,
        "Total_sleep_min_per_day": np.r_[np.full(20, 1290.0)],
        "Day_sleep_min_per_day": np.r_[np.full(10, 640.0), np.full(10, 700.0)],
        "Total_activity_per_day": np.full(20, 220.0),
    })
    rep = DR.plausibility_report(quiet)
    check("sleep above the published range is flagged",
          not rep["ok"] and any("above the" in f for f in rep["flags"]),
          f"median {rep['median_sleep_min_per_day']:.0f} min/day")
    check("the flag names the mechanisms that mimic sleep",
          any("died" in f or "beam" in f for f in rep["flags"]), "")
    check("a reversed male/female daytime sleep difference is flagged",
          any("channel map" in f for f in rep["flags"]),
          f"male - female = {rep.get('male_minus_female_day_sleep', float('nan')):+.0f} min/day")

    normal = pd.DataFrame({
        "Genotype": ["WT"] * 20, "Sex": ["Male"] * 10 + ["Female"] * 10,
        "Total_sleep_min_per_day": np.r_[np.full(10, 780.0), np.full(10, 620.0)],
        "Day_sleep_min_per_day": np.r_[np.full(10, 380.0), np.full(10, 210.0)],
        "Total_activity_per_day": np.full(20, 900.0),
    })
    rep2 = DR.plausibility_report(normal)
    check("a normal dataset raises no flags",
          rep2["ok"] and not rep2["flags"], f"{rep2['flags']}")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/dam_reader.py")
    print("=" * 78)
    test_A_exact_parsing()
    test_B_status_rows_are_missing_not_zero()
    test_C_restart_detection()
    test_D_dead_channels()
    test_E_period_refusal()
    test_F_channel_map()
    test_G_plausibility()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL DAM-READER VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
