"""
Validation suite for modules/dam_planner.py

WHAT IS BEING VALIDATED
-----------------------
The planner's job is to make a specific mistake impossible, so the tests are
built as matched pairs: a layout containing the fault and one without it. A
checker that rejects everything is as useless as one that accepts everything.

A. A planned layout crosses every factor with every monitor. This is the whole
   point: it decides whether a question can be answered at all.
B. THE REGRESSION TEST. The actual layout from the laboratory's first run --
   all old flies in Monitor 1, all young in Monitor 2 -- must be rejected, and
   rejected specifically on Age while Genotype and Sex pass.
C. Groups are interleaved across channel positions rather than blocked, and a
   blocked layout is flagged.
D. Capacity, replication and single-monitor limits are reported honestly.
E. The advice matches the goal: period requires constant darkness, sleep does
   not.
F. The emitted channel map is directly usable by `dam_reader.apply_channel_map`.

Run: ./.venv/Scripts/python.exe validation/validate_dam_planner.py
Exit 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import dam_planner as DP  # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []
REAL_MAP = Path(r"D:/DroPhenix/Testing data/DAM/channel_map_CONFIRMED.csv")

GROUPS_8 = [{"Genotype": g, "Sex": s, "Age": a}
            for a in ("old", "young")
            for g in ("Oregon-R", "TDP43")
            for s in ("Female", "Male")]


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def _confounded_map(n_channels: int = 32) -> pd.DataFrame:
    """The layout that broke the first run: age nested inside monitor."""
    rows = []
    for mon, age in (("M1", "old"), ("M2", "young")):
        for gi, (geno, sex) in enumerate([("Oregon-R", "Female"), ("Oregon-R", "Male"),
                                          ("TDP43", "Female"), ("TDP43", "Male")]):
            for c in range(gi * 8 + 1, gi * 8 + 9):
                rows.append({"Monitor": mon, "Channel": c, "Genotype": geno,
                             "Sex": sex, "Age": age, "Treatment": "Untreated"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def test_A_planned_layout_is_crossed() -> None:
    print("\nA. A planned layout crosses every factor with every monitor")
    plan = DP.plan_run(GROUPS_8, n_per_group=8, n_monitors=2, n_channels=32, seed=1)
    check("the plan is produced", plan["ok"], plan.get("reason", ""))
    cm = plan["channel_map"]
    check("all 64 tubes allocated", len(cm) == 64, f"{len(cm)} tubes")

    for f in ("Genotype", "Sex", "Age"):
        tab = pd.crosstab(cm[f], cm["Monitor"])
        balanced = bool((tab.min(axis=1) > 0).all())
        even = bool((tab.max(axis=1) - tab.min(axis=1) <= 1).all())
        check(f"'{f}' appears in every monitor", balanced, tab.to_dict())
        check(f"'{f}' is evenly split between monitors", even,
              f"{tab.to_dict()}")

    v = plan["verification"]
    check("verification reports every factor as crossed",
          set(v["answerable"]) == {"Genotype", "Sex", "Age"} and not v["blocked"],
          f"answerable {v['answerable']}, blocked {v['blocked']}")
    check("a clean plan raises no warnings", not plan["warnings"],
          str(plan["warnings"])[:100])


def test_B_rejects_the_real_mistake() -> None:
    print("\nB. The layout that broke the first run is rejected  [regression test]")
    bad = _confounded_map()
    v = DP.check_plan(bad)
    check("the confounded layout is rejected", not v["ok"], "")
    check("rejected specifically on Age",
          v["per_factor"].get("Age") == "confounded", str(v["per_factor"]))
    check("Genotype and Sex still pass, so the check is not blanket-rejecting",
          v["per_factor"].get("Genotype") == "crossed"
          and v["per_factor"].get("Sex") == "crossed", str(v["per_factor"]))
    check("the message says the analysis cannot repair it",
          any("no analysis will repair" in b for b in v["blocking"]),
          v["blocking"][0][:90] if v["blocking"] else "")

    good = DP.plan_run(GROUPS_8, seed=2)["channel_map"]
    v2 = DP.check_plan(good)
    check("the planned replacement passes the same check", v2["ok"], "")

    if REAL_MAP.exists():
        vr = DP.check_plan(str(REAL_MAP))
        check("the laboratory's real confirmed map is rejected on Age",
              vr["per_factor"].get("Age") == "confounded", str(vr["per_factor"]))
    else:
        SKIPPED.append("real confirmed channel map not present")
        print("  [SKIP] real laboratory map -- file not present")


def test_C_positions_interleaved() -> None:
    print("\nC. Groups are spread across channel positions, not blocked")
    cm = DP.plan_run(GROUPS_8, seed=3)["channel_map"]
    worst = 0
    for (lvl, mon), sub in cm.groupby(["Genotype", "Monitor"]):
        ch = np.sort(sub["Channel"].to_numpy())
        contiguous = (ch.max() - ch.min() + 1) == len(ch)
        worst = max(worst, int(contiguous))
    check("no genotype occupies a contiguous block", worst == 0, "")

    blocked = _confounded_map()
    v = DP.check_plan(blocked)
    check("a blocked layout is flagged",
          any("contiguous block" in w for w in v["warnings"]),
          v["warnings"][0][:90] if v["warnings"] else "no warning raised")

    # different seeds must give different layouts, or a position effect becomes
    # a reproducible artefact across every experiment
    a = DP.plan_run(GROUPS_8, seed=1)["channel_map"]
    b = DP.plan_run(GROUPS_8, seed=2)["channel_map"]
    same = a[["Monitor", "Channel", "Genotype", "Sex", "Age"]].equals(
        b[["Monitor", "Channel", "Genotype", "Sex", "Age"]])
    check("different seeds give different layouts", not same,
          "otherwise a position effect repeats across every run")


def test_D_limits_reported() -> None:
    print("\nD. Capacity, replication and single-monitor limits are reported")
    over = DP.plan_run(GROUPS_8, n_per_group=20, n_monitors=2, n_channels=32)
    check("an over-capacity request is refused, not truncated",
          not over["ok"] and "Reduce" in over.get("reason", ""),
          over.get("reason", "")[:90])

    one = DP.plan_run(GROUPS_8, n_per_group=4, n_monitors=1, n_channels=32)
    check("a single monitor is allowed but flagged",
          one["ok"] and any("single monitor" in w for w in one["warnings"]),
          one["warnings"][0][:80] if one["warnings"] else "no warning")

    small = DP.plan_run(GROUPS_8, n_per_group=4, n_monitors=2, n_channels=32)
    check("small groups are flagged against the 8-tube minimum",
          any("smallest group" in w.lower() for w in small["warnings"]),
          str(small["warnings"])[:90])

    odd = DP.plan_run(GROUPS_8, n_per_group=7, n_monitors=2, n_channels=32)
    check("an odd group size still crosses every factor",
          odd["ok"] and not odd["verification"]["blocked"],
          f"blocked: {odd['verification']['blocked']}")
    check("and the uneven split is explained",
          any("does not divide evenly" in w for w in odd["warnings"]),
          str(odd["warnings"])[:90])


def test_E_advice_matches_goal() -> None:
    print("\nE. The advice matches what is being measured")
    sleep = DP.run_advice("sleep")
    period = DP.run_advice("period")
    check("sleep advice permits light-dark",
          any("light-dark cycle is correct" in a for a in sleep), "")
    check("sleep advice forbids reporting a period",
          any("constant darkness" in a for a in sleep), "")
    check("period advice requires constant darkness",
          any("CONSTANT DARKNESS" in a for a in period), "")
    check("period advice explains why LD cannot work",
          any("period of the lamp" in a for a in period), "")
    check("period advice sets a minimum duration",
          any("5 days" in a for a in period), "")

    plan = DP.plan_run(GROUPS_8, goal="period", seed=4)
    check("the goal reaches the plan", any("DD" in a for a in plan["advice"]), "")


def test_F_map_is_usable() -> None:
    print("\nF. The emitted map is directly usable by the reader")
    from modules import dam_reader as DR

    cm = DP.plan_run(GROUPS_8, seed=5)["channel_map"]
    need = {"Monitor", "Channel", "Genotype", "Sex", "Age", "Treatment"}
    check("the map has the columns the reader needs",
          need.issubset(cm.columns), f"{sorted(cm.columns)}")
    check("channels are unique within each monitor",
          not cm.duplicated(["Monitor", "Channel"]).any(), "")
    check("no tube is left unlabelled",
          cm["Genotype"].astype(str).str.strip().ne("").all(), "")

    # a synthetic monitor read through the planned map
    rng = np.random.default_rng(7)
    n = 200
    rows = []
    for mon in ("M1", "M2"):
        for ch in range(1, 33):
            for t in range(n):
                rows.append({"Monitor": mon, "Channel": ch,
                             "Timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=t),
                             "Activity": float(rng.poisson(2)), "Light": 1,
                             "Status": 1, "Session": 1, "Valid": True})
    long = pd.DataFrame(rows)
    lab = DR.apply_channel_map(long, cm)
    check("every channel is labelled after the merge",
          lab["Genotype"].notna().all() and len(lab) == len(long),
          f"{len(lab)} of {len(long)} rows kept")
    tidy = DR.to_dams_long(lab, drop_dead=False)
    check("it converts to the analysis schema",
          {"Genotype", "Treatment", "Sex", "FlyID", "Time_min",
           "Activity"}.issubset(tidy.columns) and tidy["FlyID"].nunique() == 64,
          f"{tidy['FlyID'].nunique()} flies")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/dam_planner.py")
    print("=" * 78)
    test_A_planned_layout_is_crossed()
    test_B_rejects_the_real_mistake()
    test_C_positions_interleaved()
    test_D_limits_reported()
    test_E_advice_matches_goal()
    test_F_map_is_usable()

    print("\n" + "=" * 78)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}): {'; '.join(SKIPPED)}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL DAM-PLANNER VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
