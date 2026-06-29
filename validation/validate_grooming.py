"""
validate_grooming.py
Validation of modules/grooming.py.

A. ALGORITHMIC exactness — bout-level input with known totals must yield exact
   %time-grooming, bout count and mean bout duration.
B. CALIBRATION — the demo reproduces the published wild-type grooming level
   (~13% of time; Qiao et al., 2018, eLife 7:e34497), with the disease model
   grooming less and the drug partially restoring it.

Run:  python validation/validate_grooming.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import grooming as G  # noqa: E402


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = got is not None and np.isfinite(got) and abs(got - expected) <= tol
        results.append((name, expected, round(float(got), 3) if got is not None and np.isfinite(got) else None, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    # ── A. Exact bout-level computation ──────────────────────────────────────
    # One fly, 600 s observation, 3 bouts of 10/20/30 s => 60 s grooming = 10%.
    bouts = pd.DataFrame({
        "Genotype": ["WT"] * 3, "Treatment": ["Veh"] * 3, "Sex": ["M"] * 3,
        "FlyID": ["f1"] * 3,
        "Bout_start_s": [10, 100, 300], "Bout_end_s": [20, 120, 330],
        "Observation_s": [600, 600, 600]})
    pf = G.per_fly_metrics(bouts).iloc[0]
    chk("Pct time grooming (=10)", pf["Pct_time_grooming"], 10.0, 1e-9)
    chk("Total grooming s (=60)", pf["Total_grooming_s"], 60.0, 1e-9)
    chk("N bouts (=3)", pf["N_bouts"], 3, 0)
    chk("Mean bout s (=20)", pf["Mean_bout_s"], 20.0, 1e-9)
    chk("Grooming rate /min (=0.3)", pf["Grooming_rate_per_min"], 0.3, 1e-9)

    # ── B. Demo calibration ───────────────────────────────────────────────────
    res = G.run_grooming_analysis(G.demo_grooming())
    pct = res["per_fly"].groupby(["Genotype", "Treatment"])["Pct_time_grooming"].mean()
    wt = pct[("Oregon-WT", "Vehicle")]
    tdv = pct[("TDP43", "Vehicle")]
    tdd = pct[("TDP43", "Drug")]
    chk("Demo WT grooming ~13%", wt, 13.0, 3.0)
    boolchk("Demo TDP43 grooms less than WT", tdv < wt, "TDP43<WT", f"{tdv:.1f}<{wt:.1f}")
    boolchk("Demo drug increases grooming", tdd > tdv, "drug>veh", f"{tdd:.1f}>{tdv:.1f}")

    print("=" * 70)
    print("DroPhenix grooming.py validation (exact bouts + 13% calibration)")
    print("=" * 70)
    print(f"{'Check':<36}{'Expected':>14}{'Got':>12}{'Pass':>6}")
    print("-" * 70)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<36}{str(exp):>14}{str(got):>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 70)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
