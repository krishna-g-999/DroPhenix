"""
validate_drug_ranking.py
Validation of DroPhenix modules/drug_ranking.py (cross-strain, cross-assay
compound ranking with unequal-N handling).

Layers:
A. Hedges' g — exact agreement with an independent formula computation on the
   same data (so unequal-N variance is verified, not just the point estimate).
B. Inverse-variance + DerSimonian-Laird pooling — agreement with the peer-
   reviewed statsmodels.stats.meta_analysis.combine_effects (fixed, random,
   Q, tau^2, I^2).
C. End-to-end ranking — on a synthetic screen with KNOWN efficacy order
   (Drug_A > Drug_B > Drug_C), the engine must recover that order, mark the
   effective drugs significant and the inert one non-significant, and use all
   four assays.

Run:  python validation/validate_drug_ranking.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import drug_ranking as DR     # noqa: E402
from modules import synthetic as SYN        # noqa: E402


def _check(name, got, expected, tol, results):
    ok = (got is not None) and np.isfinite(got) and abs(got - expected) <= tol
    results.append((name, expected, round(float(got), 5) if got is not None and np.isfinite(got) else None, ok))
    return ok


def _bool(name, ok, expectation, got, results):
    results.append((name, expectation, got, bool(ok)))
    return bool(ok)


def main() -> int:
    results = []

    # ── A. Hedges' g exact vs independent formula ─────────────────────────────
    rng = np.random.default_rng(3)
    c = rng.normal(0, 1, 30); t = rng.normal(0.6, 1.2, 42)   # unequal N, unequal var
    e = DR.hedges_g(c, t)
    n1, n2 = len(t), len(c)
    sp = np.sqrt(((n1 - 1) * np.var(t, ddof=1) + (n2 - 1) * np.var(c, ddof=1)) / (n1 + n2 - 2))
    d = (t.mean() - c.mean()) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g_ref = J * d
    var_ref = J ** 2 * ((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    _check("Hedges g vs formula", e["g"], g_ref, 1e-9, results)
    _check("Hedges g variance vs formula", e["var"], var_ref, 1e-9, results)

    # ── B. Pooling vs statsmodels ─────────────────────────────────────────────
    try:
        from statsmodels.stats.meta_analysis import combine_effects
        eff = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.45])
        var = np.array([0.04, 0.02, 0.06, 0.03, 0.05, 0.025])
        mine = DR.inverse_variance_pool(eff, var)
        sm = combine_effects(eff, var, method_re="dl")
        sf = sm.summary_frame()
        _check("Pool fixed vs statsmodels", mine["fixed"], sf.loc["fixed effect", "eff"], 1e-6, results)
        _check("Pool random(DL) vs statsmodels", mine["random"], sf.loc["random effect", "eff"], 1e-6, results)
        _check("Pool Q vs statsmodels", mine["Q"], sm.q, 1e-6, results)
        _check("Pool tau^2 vs statsmodels", mine["tau2"], sm.tau2, 1e-6, results)
        _check("Pool I^2 vs statsmodels", mine["I2"], sm.i2 * 100, 1e-3, results)
    except Exception as ex:
        results.append(("statsmodels meta cross-check", "available", repr(ex), False))

    # Inverse-variance down-weights the noisier (smaller-N) estimate: pooled
    # estimate must be closer to the precise study than a plain mean.
    pool2 = DR.inverse_variance_pool([0.2, 1.0], [0.01, 1.0])  # study1 precise, study2 noisy
    _bool("IV weights toward precise study", pool2["fixed"] < 0.4, "<0.4 (not 0.6 mean)",
          round(pool2["fixed"], 3), results)

    # ── C. End-to-end ranking on known efficacy ──────────────────────────────
    res = DR.rank_compounds(SYN.generate_screen())
    comp = res["composite"]
    order = list(comp["drug"])
    _bool("Ranking order A>B>C", order == ["Drug_A", "Drug_B", "Drug_C"],
          "[A,B,C]", str(order), results)
    _bool("All 4 assays used", set(res["per_assay"]["assay"]) ==
          {"climbing", "ymaze", "dams", "survival"}, "4 assays",
          str(sorted(set(res["per_assay"]["assay"]))), results)
    pa = comp.set_index("drug")["p_value"]
    _bool("Drug_A significant", pa["Drug_A"] < 0.05, "p<0.05", round(float(pa["Drug_A"]), 4), results)
    _bool("Drug_C not significant", pa["Drug_C"] > 0.05, "p>0.05", round(float(pa["Drug_C"]), 4), results)
    # unequal-N must be detected and flagged
    flagged = (res["imbalance"]["flag"].str.contains("unequal").any())
    _bool("Unequal-N detected & flagged", flagged, "flagged", str(flagged), results)

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("DroPhenix drug_ranking.py validation (Hedges g, DL meta-analysis, ranking)")
    print("=" * 80)
    print(f"{'Check':<40}{'Expected':>18}{'Got':>14}{'Pass':>7}")
    print("-" * 80)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<40}{str(exp):>18}{str(got):>14}{'OK' if ok else 'FAIL':>7}")
    print("-" * 80)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
