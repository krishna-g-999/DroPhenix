"""
Validation suite for modules/data_triage.py

WHAT IS BEING VALIDATED
-----------------------
Triage exists to catch design and recording faults BEFORE analysis, so the tests
are built as matched PAIRS: a dataset containing the fault, and an otherwise
identical one without it. A checker that fires on everything is as useless as
one that fires on nothing, so every test asserts both that the fault is caught
and that the clean version passes.

The faults are the ones found in this laboratory's real data:
  A. a factor completely confounded with a nuisance variable (age x monitor)
  B. no replication -- one unit per group
  C. a bounded measure pinned at its ceiling (sleep at 88% of the day)
  D. clustering -- flies within a vial are not independent
  E. metadata never filled in, so nothing varies
  F. too few doses to identify a dose-response curve
  G. the two real datasets give the verdicts arrived at by hand

Run: ./.venv/Scripts/python.exe validation/validate_data_triage.py
Exit 0 = all pass. Real-data tests SKIP when the files are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import data_triage as T  # noqa: E402

FAILURES: list[str] = []
SKIPPED: list[str] = []
SEED = 20260805
DAM_DIR = Path(r"D:/DroPhenix/Testing data/DAM")
VIDEO_TABLE = Path(r"D:/DroPhenix/Testing data/Climbing assay/batch15_climbing_table.csv")


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def skip(name: str, why: str) -> None:
    print(f"  [SKIP] {name}  --  {why}")
    SKIPPED.append(name)


def _codes(rep, severity=None):
    return {f.code for f in rep.findings if severity is None or f.severity == severity}


def _climbing(n_vials_per_group=4, monitors_confounded=False, seed=SEED):
    """A plain climbing dataset: 2 genotypes x n vials x 4 timepoints."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi, geno in enumerate(("Oregon-R", "TDP43")):
        for v in range(n_vials_per_group):
            vial = f"{geno[:3]}_V{v}"
            batch = ("B1" if gi == 0 else "B2") if monitors_confounded else f"B{v % 2 + 1}"
            for t in (10, 20, 30, 60):
                rows.append({"Genotype": geno, "Sex": "Male", "Treatment": "Vehicle",
                             "VialID": vial, "Batch": batch, "Time": t,
                             "Count": int(rng.binomial(20, 0.6 - 0.15 * gi)), "n": 20})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def test_A_confounding() -> None:
    print("\nA. Complete confounding is a blocker; a crossed design is not")
    bad = T.triage(_climbing(monitors_confounded=True))
    good = T.triage(_climbing(monitors_confounded=False))
    check("confounded design blocks the factor",
          bad.answerable.get("Genotype") == "blocked" and "confounded" in _codes(bad, T.BLOCKER),
          bad.verdict())
    check("crossed design does NOT block it",
          good.answerable.get("Genotype") == "answerable"
          and "confounded" not in _codes(good),
          good.verdict())
    check("the blocker says the fix is in the design, not the analysis",
          any("fixed in the design" in f.message for f in bad.blockers), "")


def test_B_replication() -> None:
    print("\nB. One unit per group is a blocker")
    one = _climbing(n_vials_per_group=1)
    many = _climbing(n_vials_per_group=4)
    r1, r4 = T.triage(one), T.triage(many)
    check("a single vial per genotype blocks the comparison",
          r1.answerable.get("Genotype") == "blocked"
          and "no_replication" in _codes(r1, T.BLOCKER), r1.verdict())
    check("four vials per genotype does not",
          "no_replication" not in _codes(r4), r4.verdict())
    two = T.triage(_climbing(n_vials_per_group=2))
    check("two vials is allowed but flagged as low replication",
          "low_replication" in _codes(two, T.WARNING), two.verdict())


def test_C_ceiling() -> None:
    print("\nC. A measure pinned at its ceiling is flagged")
    rng = np.random.default_rng(SEED + 2)
    ceil = _climbing()
    ceil["Count"] = 20                      # every fly climbed: proportion = 1.0
    mid = _climbing()
    mid["Count"] = rng.integers(6, 15, len(mid))
    rc, rm = T.triage(ceil), T.triage(mid)
    check("a saturated proportion is flagged", "ceiling" in _codes(rc, T.WARNING), "")
    check("a mid-range proportion is not", "ceiling" not in _codes(rm), "")

    floor = _climbing()
    floor["Count"] = 0
    check("a floored proportion is flagged",
          "floor" in _codes(T.triage(floor), T.WARNING), "")

    sleepy = pd.DataFrame({"Genotype": ["WT"] * 20, "Sex": ["Male"] * 20,
                           "FlyID": [f"F{i}" for i in range(20)],
                           "Total_sleep_min_per_day": np.full(20, 1300.0)})
    check("sleep near 1440 min/day is flagged as a ceiling",
          "ceiling" in _codes(T.triage(sleepy), T.WARNING), "")


def test_D_clustering() -> None:
    print("\nD. Correlated flies within a vial are reported with a design effect")
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for geno in ("Oregon-R", "TDP43"):
        for v in range(8):
            pv = rng.beta(0.6 * 9, 0.4 * 9)      # strong between-vial spread
            for t in (10, 30):
                rows.append({"Genotype": geno, "Sex": "Male", "VialID": f"{geno}{v}",
                             "Time": t, "Count": int(rng.binomial(20, pv)), "n": 20})
    rep = T.triage(pd.DataFrame(rows))
    check("clustering is detected and quantified",
          "clustering" in _codes(rep, T.WARNING), "")
    msg = [f for f in rep.findings if f.code == "clustering"]
    check("the message states the effective sample size",
          bool(msg) and "information of about" in msg[0].message,
          msg[0].message[:80] if msg else "")

    indep = _climbing(seed=SEED + 4)          # binomial, no extra vial variance
    check("an unclustered dataset is not flagged",
          "clustering" not in _codes(T.triage(indep)), "")


def test_E_no_factors() -> None:
    print("\nE. Unlabelled metadata is reported as a blocker with its cause")
    blank = _climbing()
    blank["Genotype"] = "UNSPECIFIED"
    blank["Sex"] = "UNSPECIFIED"
    blank["Treatment"] = "UNSPECIFIED"
    rep = T.triage(blank)
    check("a dataset with no varying factor is blocked",
          "no_factors" in _codes(rep, T.BLOCKER), rep.verdict())
    check("the message names the constant columns rather than being cryptic",
          any("single value" in f.message for f in rep.blockers), "")
    check("a labelled dataset is not blocked this way",
          "no_factors" not in _codes(T.triage(_climbing())), "")


def test_F_coverage() -> None:
    print("\nF. Too few doses to identify a dose-response curve")
    rng = np.random.default_rng(SEED + 5)
    rows = []
    for d in (0, 10, 100):
        for v in range(3):
            rows.append({"Genotype": "WT", "Sex": "Male", "Dose": d,
                         "VialID": f"D{d}_{v}", "Time": 10,
                         "Count": int(rng.binomial(20, 0.6)), "n": 20})
    rep = T.triage(pd.DataFrame(rows))
    check("three doses blocks a four-parameter curve",
          "too_few_doses" in _codes(rep, T.BLOCKER), rep.verdict())

    rows6 = []
    for d in (0, 3, 10, 30, 100, 300):
        for v in range(3):
            rows6.append({"Genotype": "WT", "Sex": "Male", "Dose": d,
                          "VialID": f"D{d}_{v}", "Time": 10,
                          "Count": int(rng.binomial(20, 0.6)), "n": 20})
    check("six doses does not",
          "too_few_doses" not in _codes(T.triage(pd.DataFrame(rows6))), "")


def test_G_real_data() -> None:
    print("\nG. The two real datasets give the verdicts reached by hand")
    # The laboratory renames these files as timepoints accumulate (Monitor1.txt
    # became "Monitor1_7th day.txt" on 2026-09-08), so match by prefix rather
    # than by an exact name that a rename silently breaks.
    m1 = sorted(DAM_DIR.glob("Monitor1*.txt")) if DAM_DIR.exists() else []
    m2 = sorted(DAM_DIR.glob("Monitor2*.txt")) if DAM_DIR.exists() else []
    if not (m1 and m2):
        skip("real DAM triage", f"no Monitor1*/Monitor2* files in {DAM_DIR}")
    else:
        from modules import dam_reader as DR
        long = DR.read_monitors([str(m1[0]), str(m2[0])], monitors=["M1", "M2"])
        cmap_path = DAM_DIR / "channel_map_CONFIRMED.csv"
        cmap = pd.read_csv(cmap_path) if cmap_path.exists() else None
        rep = T.triage_dam(long, channel_map=cmap)
        print(f"      verdict: {rep.verdict()}")
        check("circadian period is blocked (light-dark, 2.7 days)",
              rep.answerable.get("circadian period") == "blocked", "")
        if cmap is not None:
            check("Age is blocked (confounded with monitor)",
                  rep.answerable.get("Age") == "blocked", "")
            check("Genotype and Sex remain answerable",
                  rep.answerable.get("Genotype") == "answerable"
                  and rep.answerable.get("Sex") == "answerable", "")
        check("fly deaths are reported", "deaths" in _codes(rep, T.WARNING), "")

    if not VIDEO_TABLE.exists():
        skip("real video-table triage", "scored table not present")
    else:
        from modules.data_parser import load_data
        df, _, err = load_data(str(VIDEO_TABLE))
        if err:
            skip("real video-table triage", f"could not load: {err}")
            return
        rep = T.triage(df)
        print(f"      verdict: {rep.verdict()}")
        check("clustering in the video-scored vials is detected",
              "clustering" in _codes(rep, T.WARNING), "")
        check("the unit of replication is the vial",
              rep.unit_col == "VialID", str(rep.unit_col))


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/data_triage.py")
    print("=" * 78)
    test_A_confounding()
    test_B_replication()
    test_C_ceiling()
    test_D_clustering()
    test_E_no_factors()
    test_F_coverage()
    test_G_real_data()

    print("\n" + "=" * 78)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}): {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL DATA-TRIAGE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
