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

import warnings
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

    # ── Test choice is A PRIORI, not read off the data (audit S3 finding 3.4) ─
    # PlantGrowth is 3 normal, equal-variance groups, so the old data-dependent
    # router sent it to ANOVA. That is no longer what decides: the default is the
    # rank test, chosen in advance, and a parametric analysis is available only
    # by declaring it. These checks pin the POLICY, not the branch.
    res = SE.auto_test(data)
    results.append(("default is the rank test, chosen a priori",
                    "Kruskal-Wallis", res["method"][:15],
                    res["method"].startswith("Kruskal-Wallis")))
    results.append(("...and the choice is recorded as a priori",
                    "a priori (rank)", res["test_choice"],
                    res["test_choice"] == "a priori (rank)"))
    res_p = SE.auto_test(data, method="parametric")
    results.append(("declared parametric still gives the published ANOVA path",
                    "ANOVA path", res_p["method"][:15], "ANOVA" in res_p["method"]))
    _check("declared parametric reproduces published F (=4.846)",
           res_p["main_result"]["F_statistic"], 4.846, 0.01, results)
    # Normality/Levene are still computed and reported -- they just no longer route.
    results.append(("Shapiro/Levene still reported as diagnostics", "both present",
                    f"{len(res['normality'])} rows, levene={'p_value' in res['levene']}",
                    len(res["normality"]) == 3 and "p_value" in res["levene"]))
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter("always")
        SE.auto_test(data, method="auto")
        results.append(("method='auto' is deprecated and warns", "warns",
                        "warned" if _w else "SILENT", bool(_w)))

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("DroPhenix stats_engine.py validation vs published PlantGrowth (Dobson 1983)")
    # ── 2026-09-07 cross-checks against reference implementations ────────────
    import itertools as _it
    from math import comb as _comb
    from scipy import stats as _sps

    _rng = np.random.default_rng(0)

    # rank_biserial_r against the Kerby (2014) definition r = f - u computed
    # from first principles over every (a, b) pair.
    _worst = 0.0
    for _ in range(200):
        _a = _rng.normal(0, 1, int(_rng.integers(3, 9)))
        _b = _rng.normal(0.5, 1, int(_rng.integers(3, 9)))
        _pairs = [(x, y) for x in _a for y in _b]
        _f = np.mean([x > y for x, y in _pairs])
        _u = np.mean([x < y for x, y in _pairs])
        _worst = max(_worst, abs(SE.rank_biserial_r(_a, _b) - (_f - _u)))
    _check("rank_biserial_r == Kerby (2014) f - u", _worst, 0.0, 1e-12, results)
    _hi, _lo = np.array([10., 11, 12, 13]), np.array([1., 2, 3, 4])
    results.append(("rank_biserial_r sign agrees with Cohen's d", "both +",
                    f"r={SE.rank_biserial_r(_hi,_lo):+.1f} d={SE.cohens_d(_hi,_lo):+.1f}",
                    SE.rank_biserial_r(_hi, _lo) > 0 and SE.cohens_d(_hi, _lo) > 0))

    # permutation_test_means against scipy's EXACT enumeration.
    _worst = 0.0
    for _ in range(60):
        _a = _rng.normal(0, 1, int(_rng.integers(3, 8)))
        _b = _rng.normal(0.6, 1, int(_rng.integers(3, 8)))
        _mine = SE.permutation_test_means(_a, _b, n_perm=10 ** 9)["p_value"]
        _ref = _sps.permutation_test(
            (_a, _b), lambda x, y: np.mean(x) - np.mean(y),
            permutation_type="independent", alternative="two-sided",
            n_resamples=np.inf).pvalue
        _worst = max(_worst, abs(_mine - _ref))
    _check("permutation p == scipy.stats.permutation_test (exact)", _worst, 0.0, 1e-12, results)

    # min_attainable_p against brute force on the most extreme possible data.
    _bad = []
    for _n1, _n2 in [(3, 3), (3, 4), (4, 4), (5, 5), (4, 6)]:
        _brute = _sps.mannwhitneyu(np.arange(_n1) + 100.0, np.arange(_n2) * 1.0,
                                   alternative="two-sided").pvalue
        if abs(_brute - SE.min_attainable_p(_n1, _n2)) > 1e-12:
            _bad.append((_n1, _n2))
    results.append(("min_attainable_p == brute-force MWU floor (5 designs)",
                    "all match", _bad or "all match", not _bad))
    _check("3-vial design floor is 0.100", SE.min_attainable_p(3, 3), 0.1, 1e-12, results)
    _sig = SE.auto_test({"A": np.array([90., 92, 95]),
                         "B": np.array([-90., -92, -95])})["main_result"]["Significant"]
    results.append(("underpowered design is labelled, not called 'ns'",
                    "underpowered by design", _sig, _sig == "underpowered by design"))

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
