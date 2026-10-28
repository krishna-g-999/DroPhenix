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
    # ── Handedness confound in the two derived scores (2026-09-07) ───────────
    # The established Drosophila Y-maze read-outs are turn bias and switchiness
    # (Buchanan, Kain & de Bivort 2015, PNAS 112:6700). Under this module's
    # geometry an alternation triad is two consecutive turns in the SAME
    # direction, so for i.i.d. turns with P(right) = b a MEMORYLESS fly
    # alternates at b^2 + (1-b)^2 and switches at 2b(1-b). Both raw scores
    # therefore move with handedness alone. The corrected columns must sit at
    # zero for memoryless flies at EVERY bias, and must still detect a fly with
    # genuine sequential dependence regardless of its handedness.
    _rng = np.random.default_rng(1)

    def _memoryless(b, n_fly=120, n_turn=60):
        alt, sw = [], []
        for i in range(n_fly):
            turns = _rng.choice(["R", "L"], size=n_turn, p=[b, 1 - b])
            arms = Y.turns_to_arm_sequence(turns)
            d = pd.DataFrame({"Genotype": "WT", "Treatment": "V", "Sex": "M",
                              "FlyID": f"f{i}", "Order": range(len(arms)), "Arm": arms})
            r = Y.per_fly_metrics(Y.load_ymaze(d)).iloc[0]
            alt.append(r["alternation_vs_chance"]); sw.append(r["switch_rate_vs_chance"])
        return float(np.mean(alt)), float(np.mean(sw))

    raw_spread = []
    for b in (0.5, 0.7, 0.9):
        a_res, s_res = _memoryless(b)
        _check(f"memoryless b={b}: alternation_vs_chance ~ 0", a_res, 0.0, 3.0, results)
        _check(f"memoryless b={b}: switch_rate_vs_chance ~ 0", s_res, 0.0, 0.03, results)
        turns = _rng.choice(["R", "L"], size=2000, p=[b, 1 - b])
        arms = Y.turns_to_arm_sequence(turns)
        d = pd.DataFrame({"Genotype": "WT", "Treatment": "V", "Sex": "M", "FlyID": "x",
                          "Order": range(len(arms)), "Arm": arms})
        raw_spread.append(float(Y.per_fly_metrics(Y.load_ymaze(d)).iloc[0]["alternation_pct"]))
    results.append(("raw alternation_pct DOES move with handedness (the confound)",
                    ">25 pts", f"{max(raw_spread)-min(raw_spread):.0f} pts",
                    (max(raw_spread) - min(raw_spread)) > 25))

    def _switchy(b, p_repeat=0.2, n_fly=120, n_turn=60):
        out = []
        for i in range(n_fly):
            t = ["R"]
            for _ in range(n_turn - 1):
                t.append(t[-1] if _rng.random() < p_repeat else ("L" if t[-1] == "R" else "R"))
            arms = Y.turns_to_arm_sequence(t)
            d = pd.DataFrame({"Genotype": "WT", "Treatment": "V", "Sex": "M",
                              "FlyID": f"f{i}", "Order": range(len(arms)), "Arm": arms})
            out.append(Y.per_fly_metrics(Y.load_ymaze(d)).iloc[0]["switch_rate_vs_chance"])
        return float(np.mean(out))

    s_unbiased, s_handed = _switchy(0.5), _switchy(0.9)
    results.append(("real sequential dependence IS detected (unbiased fly)",
                    ">0.2", round(s_unbiased, 3), s_unbiased > 0.2))
    results.append(("...and equally in a strongly handed fly",
                    "same +-0.05", f"{s_unbiased:.3f} vs {s_handed:.3f}",
                    abs(s_unbiased - s_handed) < 0.05))

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
