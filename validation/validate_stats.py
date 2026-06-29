"""
validate_stats.py
Validation of DroPhenix modules/stats_engine.py — the core statistical engine
that every assay (and the Phase-5 drug ranking) depends on.

Benchmarks against the canonical, published PlantGrowth dataset (Dobson, 1983,
"An Introduction to Statistical Modelling"; the standard R `aov` example):
    one-way ANOVA  : F(2,27) = 4.846, p = 0.01591
    Kruskal-Wallis : H = 7.9882, df = 2, p = 0.01842
Plus hand-checked Cohen's d, Hedges' g, and Benjamini-Hochberg FDR.

Run:  python validation/validate_stats.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import stats_engine as SE  # noqa: E402


# Published PlantGrowth raw data (dried weight of plants, 3 groups of 10).
CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]


def _check(name, got, expected, tol, results):
    ok = (got is not None) and np.isfinite(got) and abs(got - expected) <= tol
    results.append((name, expected, round(float(got), 5) if got is not None and np.isfinite(got) else None, ok))
    return ok


def main() -> int:
    results = []
    data = {"ctrl": np.array(CTRL), "trt1": np.array(TRT1), "trt2": np.array(TRT2)}

    # ── One-way ANOVA (published F = 4.846, p = 0.01591) ──────────────────────
    anova, pairwise = SE.anova_tukey(data)
    _check("ANOVA F (=4.846)", anova["F_statistic"], 4.846, 0.01, results)
    _check("ANOVA p (=0.01591)", anova["p_value"], 0.01591, 0.0005, results)

    # Post-hoc: trt2 vs trt1 should be the most significant pair (smallest p).
    pw = pairwise.copy()
    pw["pair"] = pw["Group A"] + "-" + pw["Group B"]
    most_sig = pw.loc[pw["p_raw"].idxmin(), "pair"]
    ok = set(most_sig.split("-")) == {"trt1", "trt2"}
    results.append(("Post-hoc most-significant pair = trt1/trt2", "trt1/trt2", most_sig, ok))

    # ── Kruskal-Wallis (published H = 7.9882, p = 0.01842) ────────────────────
    kw, _ = SE.kruskal_dunn(data)
    _check("Kruskal-Wallis H (=7.9882)", kw["H_statistic"], 7.9882, 0.01, results)
    _check("Kruskal-Wallis p (=0.01842)", kw["p_value"], 0.01842, 0.0005, results)

    # ── Effect sizes (hand-checked) ───────────────────────────────────────────
    a = np.array([1, 2, 3, 4, 5], dtype=float)   # mean 3, sd 1.5811
    b = np.array([3, 4, 5, 6, 7], dtype=float)   # mean 5, sd 1.5811
    # pooled sd = 1.5811; d = (3-5)/1.5811 = -1.2649
    _check("Cohen's d (=-1.2649)", SE.cohens_d(a, b), -1.2649, 0.001, results)
    # Hedges' g = d * (1 - 3/(4*N-9)), N=10 -> correction 1 - 3/31 = 0.90323
    _check("Hedges' g (=-1.1425)", SE.hedges_g(a, b), -1.1425, 0.002, results)

    # ── Benjamini-Hochberg FDR (hand-checked) ─────────────────────────────────
    # p = [0.01,0.02,0.03,0.04,0.05]; each *5/rank then monotone -> all 0.05
    adj = SE.bh_correction(np.array([0.01, 0.02, 0.03, 0.04, 0.05]))
    _check("BH-FDR uniform case -> 0.05", float(np.max(np.abs(adj - 0.05))), 0.0, 1e-9, results)
    # A clearly significant smallest p stays small: [0.001, 0.5, 0.5, 0.5]
    adj2 = SE.bh_correction(np.array([0.001, 0.5, 0.5, 0.5]))
    _check("BH-FDR smallest (0.001*4=0.004)", adj2[0], 0.004, 1e-9, results)

    # ── auto_test routing: 3 normal equal-variance groups -> parametric ───────
    res = SE.auto_test(data)
    routed_param = "ANOVA" in res["method"]
    results.append(("auto_test routes PlantGrowth to ANOVA", "ANOVA path", res["method"], routed_param))

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("DroPhenix stats_engine.py validation vs published PlantGrowth (Dobson 1983)")
    print("=" * 78)
    print(f"{'Check':<44}{'Expected':>14}{'Engine':>12}{'Pass':>7}")
    print("-" * 78)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<44}{str(exp):>14}{str(got):>12}{'OK' if ok else 'FAIL':>7}")
    print("-" * 78)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
