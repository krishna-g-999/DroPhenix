"""
run_all.py — DroPhenix master validation runner.

Executes every per-module validation suite (each benchmarked against published
raw data / hand-checked analytics) and prints a consolidated pass/fail report.
This is the real, runnable validation suite for DroPhenix.

Suites:
  validate_stats     — stats_engine vs PlantGrowth ANOVA/Kruskal (Dobson 1983)
  validate_survival  — survival vs Freireich 1963 6-MP leukemia data
  validate_ymaze     — Y-maze vs de Bivort 2015 / Kim 2017 anchors
  validate_dams      — DAMS vs Konopka & Benzer 1971 periods + 5-min sleep rule
  validate_synthetic — unified synthetic cohort sanity (disease-selective shape)

Run:  python validation/run_all.py        (exit 0 = every suite passed)
"""

import importlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")   # keep the consolidated report clean
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUITES = ["validate_stats", "validate_survival", "validate_ymaze",
          "validate_dams", "validate_synthetic", "validate_drug_ranking",
          "validate_video", "validate_larval", "validate_grooming",
          "validate_metrics", "validate_trajectory", "demo_unequal_n"]


def main() -> int:
    results = {}
    for name in SUITES:
        mod = importlib.import_module(f"validation.{name}")
        try:
            rc = mod.main()
        except Exception as e:  # a suite that crashes is a failure
            print(f"[{name}] CRASHED: {e!r}")
            rc = 1
        results[name] = (rc == 0)
        print()

    print("#" * 70)
    print("# DroPhenix MASTER VALIDATION SUMMARY")
    print("#" * 70)
    for name, ok in results.items():
        print(f"  {name:<22} {'PASS' if ok else 'FAIL'}")
    all_ok = all(results.values())
    print("#" * 70)
    print("OVERALL:", "ALL SUITES PASSED" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
