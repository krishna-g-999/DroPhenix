"""
validate_ymaze.py
Validation of DroPhenix modules/ymaze.py.

Two complementary layers:

A. ALGORITHMIC exactness — hand-constructed arm sequences with analytically
   known answers (no ambiguity possible).

B. PUBLISHED-LITERATURE anchors — the engine must reproduce the quantitative
   findings reported for real Drosophila Y-maze data:
     * Turn-bias population mean ~= 0.50
         Buchanan, Kain & de Bivort (2015) PNAS 112:6700-6705 (mean
         indistinguishable from 0.5); Werkhoven et al. (2021) reported a grand
         mean of 0.496 across 183,496 flies.
     * ~23.5% of (Canton-S) flies turn right >70% or <30% of the time
         (Buchanan et al., 2015) -> reproduced by Beta(4.6, 4.6) turn bias.
     * Stable individual handedness: between-fly turn-bias variance exceeds the
         binomial sampling null (de Bivort individuality test).
     * Spontaneous-alternation chance level = 50% for an unbiased fly with
         independent turns (Kim et al., 2017, Neurobiol. Learn. Mem. 144:50-58
         report flies alternating ABOVE chance).

Run:  python validation/validate_ymaze.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import ymaze as Y  # noqa: E402


def _check(name, got, expected, tol, results, integer=False):
    ok = (got is not None) and np.isfinite(got) and abs(got - expected) <= tol
    g = int(got) if (integer and got is not None and np.isfinite(got)) else (
        round(float(got), 4) if got is not None and np.isfinite(got) else None)
    results.append((name, expected, g, ok))
    return ok


def _one_fly(seq):
    df = pd.DataFrame({"Genotype": ["G"], "Treatment": ["T"], "Sex": ["M"],
                       "FlyID": ["f1"], "Sequence": [seq]})
    return Y.per_fly_metrics(Y.load_ymaze(df)).iloc[0]


def main() -> int:
    results = []

    # ── A. Algorithmic exactness ──────────────────────────────────────────────
    r = _one_fly("ABCABC")          # all turns same direction -> 100% alternation
    _check("ABCABC alternation% (=100)", r["alternation_pct"], 100.0, 0.0, results)
    _check("ABCABC n_alternations (=4)", r["n_alternations"], 4, 0, results, integer=True)
    _check("ABCABC turn_bias (=1.0)", r["turn_bias"], 1.0, 0.0, results)

    r = _one_fly("ABAB")            # always returns -> 0% alternation
    _check("ABAB alternation% (=0)", r["alternation_pct"], 0.0, 0.0, results)

    r = _one_fly("ABCACB")          # 3 of 4 triads alternate -> 75%
    _check("ABCACB alternation% (=75)", r["alternation_pct"], 75.0, 0.0, results)

    # ── B1. Chance level = 50% for unbiased independent turns ────────────────
    rng = np.random.default_rng(0)
    rows = []
    for f in range(400):
        turns = ["R" if rng.random() < 0.5 else "L" for _ in range(200)]
        rows.append({"Genotype": "WT", "Treatment": "ctrl", "Sex": "M",
                     "FlyID": f, "Sequence": "".join(Y.turns_to_arm_sequence(turns))})
    res = Y.run_ymaze_analysis(pd.DataFrame(rows))
    alt = res["alternation"].iloc[0]
    tb = res["turn_bias"].iloc[0]
    _check("Unbiased alternation% (~50)", alt["Mean_alternation_pct"], 50.0, 1.5, results)
    _check("Unbiased turn-bias mean (~0.50)", tb["Mean_turn_bias"], 0.50, 0.02, results)

    # ── B2. de Bivort calibration: ~23.5% extreme, individuality significant ─
    rng = np.random.default_rng(7)
    rows = []
    for f in range(1500):
        b = rng.beta(4.6, 4.6)
        turns = ["R" if rng.random() < b else "L" for _ in range(200)]
        rows.append({"Genotype": "CS", "Treatment": "ctrl", "Sex": "M",
                     "FlyID": f, "Sequence": "".join(Y.turns_to_arm_sequence(turns))})
    tb = Y.run_ymaze_analysis(pd.DataFrame(rows))["turn_bias"].iloc[0]
    _check("Calibrated turn-bias mean (~0.50)", tb["Mean_turn_bias"], 0.50, 0.02, results)
    _check("Calibrated %extreme (pub 23.5)", tb["Pct_extreme_30_70"], 23.5, 3.0, results)
    _check("Individuality p (<0.05; true spread)", tb["Individuality_p"], 0.0, 0.05, results)

    # ── B3. Common-bias null: individuality NOT significant ──────────────────
    rng = np.random.default_rng(1)
    rows = []
    for f in range(400):
        turns = ["R" if rng.random() < 0.5 else "L" for _ in range(200)]
        rows.append({"Genotype": "WT", "Treatment": "ctrl", "Sex": "M",
                     "FlyID": f, "Sequence": "".join(Y.turns_to_arm_sequence(turns))})
    tb = Y.run_ymaze_analysis(pd.DataFrame(rows))["turn_bias"].iloc[0]
    null_ok = (tb["Individuality_p"] is not None) and (tb["Individuality_p"] > 0.05)
    results.append(("Common-bias individuality p (>0.05, NS)", ">0.05",
                    round(float(tb["Individuality_p"]), 4), null_ok))

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("DroPhenix ymaze.py validation (algorithmic + published de Bivort/Kim anchors)")
    print("=" * 80)
    print(f"{'Check':<46}{'Expected':>14}{'Engine':>12}{'Pass':>7}")
    print("-" * 80)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<46}{str(exp):>14}{str(got):>12}{'OK' if ok else 'FAIL':>7}")
    print("-" * 80)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
