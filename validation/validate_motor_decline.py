"""
validate_motor_decline.py
Validation of modules/motor_decline.py — Motor Decline Rate (MDR) and MSCI.

Regression tests for docs/MODULE_AUDIT_2026-09.md findings S2 2.2-2.4 and
S1 1.1. Each asserts a scientific property, not a return shape.

A. MODEL SUPPORT (audit 2.2)
   The two-parameter form PI0*exp(-k*t) with PI0 >= 0 cannot produce a negative
   value, so it could not fit any genotype whose PI crosses zero — the severely
   affected ones. A1 recovers k and the asymptote from data that goes negative.
   A2 confirms the two-parameter form is correctly declared inapplicable there.

B. MODEL SELECTION (audit 2.3)
   Selection is by AICc, which penalises parameters, not by a bare "R2 >= 0.5"
   threshold that never compared the two models. Data generated as a straight
   line must select linear; data generated as a decaying exponential must select
   exponential. Both fits are still reported either way, so k stays available
   for cross-group comparison.

C. UNCERTAINTY (audit 2.4)
   `popt, _ = curve_fit(...)` discarded the covariance, so k, t_half, delta_k
   and Rescue_pct had NO error at all.
   C1. k_se is finite and shrinks as the noise falls.
   C2. FREQUENTIST COVERAGE: over 300 simulated experiments with a known true
       k, the reported 95% CI must contain the truth close to 95% of the time.
       This is the check that a standard error is actually calibrated rather
       than merely present.
   C3. delta_k_se propagates from both groups' fits (variances add).
   C4. t_half_se follows the delta method exactly: ln2 * SE(k) / k^2.

D. MSCI PERMUTATION P-VALUE (audit 1.1)
   D1. n = 3 groups: the exact two-sided p for |rho| = 1 is 2/3! = 0.3333, not
       the 0.0 that scipy's asymptotic t-approximation produced. The row must
       be labelled underpowered, because 0.333 > 0.05 means NO result at n = 3
       can ever reach significance.
   D2. n = 8 groups, perfect monotone coupling: p = 2/8! = 4.96e-05, and it is
       reported at 3 significant figures so it never prints as "0.0".
   D3. A p-value of exactly 0 is never returned.

Run:  python validation/validate_motor_decline.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import motor_decline as MD  # noqa: E402


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 5) if got is not None and np.isfinite(got) else got, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    def between(name, got, lo, hi):
        ok = got is not None and np.isfinite(got) and lo <= float(got) <= hi
        results.append((name, f"[{lo}, {hi}]",
                        round(float(got), 4) if got is not None and np.isfinite(got) else got, ok))

    ages = np.arange(0.0, 61.0, 7.5)      # 9 ages, a realistic longitudinal design

    # ── A. Model support: PI that crosses zero ───────────────────────────────
    rng = np.random.default_rng(3)
    truth = dict(A=95.0, k=0.055, C=-55.0)          # 40 -> -55 over 60 days
    y = truth["A"] * np.exp(-truth["k"] * ages) + truth["C"] + rng.normal(0, 1.5, len(ages))
    fa = MD.fit_decay(ages, y)
    boolchk("A1 negative-PI series is fitted at all (k not NaN)",
            np.isfinite(fa["k"]), "finite k", fa["k"])
    chk("A1 k recovered (true 0.055)", fa["k"], 0.055, 0.010)
    chk("A1 asymptote recovered (true -55)", fa["PI_asymptote"], -55.0, 5.0)
    boolchk("A1 selects the offset exponential",
            fa["fit_model"] == "exponential_offset", "exponential_offset", fa["fit_model"])
    boolchk("A2 two-parameter form declared inapplicable for negative PI",
            "not applicable" in fa["fit_note"], "noted", fa["fit_note"][:46])

    # ── B. Model selection on AICc ───────────────────────────────────────────
    rng = np.random.default_rng(5)
    lin_y = 90.0 - 1.0 * ages + rng.normal(0, 2.0, len(ages))
    fl = MD.fit_decay(ages, lin_y)
    boolchk("B1 straight-line data selects linear",
            fl["fit_model"] == "linear", "linear", fl["fit_model"])
    chk("B1 linear slope recovered (true -1.0)", fl["linear_slope"], -1.0, 0.15)
    boolchk("B1 k is still reported for cross-group comparison",
            np.isfinite(fl["k"]), "finite k", fl["k"])
    boolchk("B1 but the selected model is flagged non-exponential",
            fl["selected_model_is_exponential"] is False, False,
            fl["selected_model_is_exponential"])

    exp_y = 90.0 * np.exp(-0.05 * ages) + rng.normal(0, 1.0, len(ages))
    fe = MD.fit_decay(ages, exp_y)
    boolchk("B2 exponential data selects an exponential",
            str(fe["fit_model"]).startswith("exp"), "exp*", fe["fit_model"])
    chk("B2 k recovered (true 0.050)", fe["k"], 0.050, 0.008)
    boolchk("B2 AICc of both models is reported",
            np.isfinite(fe["AICc_exp"]) and np.isfinite(fe["AICc_linear"]),
            "both finite", f"{fe['AICc_exp']} / {fe['AICc_linear']}")

    # ── C. Uncertainty ───────────────────────────────────────────────────────
    rng = np.random.default_rng(11)
    clean = MD.fit_decay(ages, 90 * np.exp(-0.05 * ages) + rng.normal(0, 0.5, len(ages)))
    noisy = MD.fit_decay(ages, 90 * np.exp(-0.05 * ages) + rng.normal(0, 6.0, len(ages)))
    boolchk("C1 k_se is finite (covariance is kept, not discarded)",
            np.isfinite(clean["k_se"]), "finite", clean["k_se"])
    boolchk("C1 k_se grows with measurement noise",
            noisy["k_se"] > clean["k_se"], "noisy > clean",
            f"{noisy['k_se']} > {clean['k_se']}")

    # C2. Frequentist coverage of the reported 95% CI.
    TRUE_K, SIGMA, N_SIM = 0.050, 3.0, 300
    rng = np.random.default_rng(2026)
    covered = n_fit = 0
    for _ in range(N_SIM):
        yy = 90.0 * np.exp(-TRUE_K * ages) + rng.normal(0, SIGMA, len(ages))
        f = MD.fit_decay(ages, yy)
        if not np.isfinite(f.get("k_CI_lower", np.nan)):
            continue
        n_fit += 1
        if f["k_CI_lower"] <= TRUE_K <= f["k_CI_upper"]:
            covered += 1
    coverage = covered / n_fit if n_fit else np.nan
    # Nonlinear-least-squares Wald intervals under-cover slightly at this n;
    # the band is deliberately 0.90-0.99 rather than a point check at 0.95, and
    # the measured value is printed so the shortfall is visible, not hidden.
    between(f"C2 95% CI coverage of true k, {N_SIM} sims at n={len(ages)} ages",
            coverage, 0.90, 0.99)
    boolchk("C2 (fits that converged)", n_fit >= 0.95 * N_SIM,
            f">= {int(0.95 * N_SIM)}", n_fit)

    # C3/C4. Propagation through delta_k and t_half.
    rng = np.random.default_rng(1)
    rows = []
    for treat, k in (("Vehicle", 0.055), ("Drug", 0.030)):
        for age in ages:
            rows.append({"Genotype": "TDP43", "Treatment": treat, "Sex": "M", "Age": age,
                         "PI_mean": 95 * np.exp(-k * age) - 40 + rng.normal(0, 1.2)})
    tab = MD.compute_mdr_table(pd.DataFrame(rows)).set_index("Treatment")
    d = tab.loc["Drug"]
    boolchk("C3 delta_k_se is reported", np.isfinite(d["delta_k_se"]),
            "finite", d["delta_k_se"])
    expected_dse = float(np.sqrt(d["k_se"] ** 2 + tab.loc["Vehicle", "k_se"] ** 2))
    chk("C3 delta_k_se = sqrt(var_treated + var_vehicle)",
        d["delta_k_se"], expected_dse, 1e-6)
    boolchk("C3 delta_k CI contains the true 0.025",
            d["delta_k_CI_lower"] <= 0.025 <= d["delta_k_CI_upper"], "contains 0.025",
            f"[{d['delta_k_CI_lower']}, {d['delta_k_CI_upper']}]")
    boolchk("C3 Rescue_pct now carries a standard error",
            np.isfinite(d["Rescue_pct_se"]), "finite", d["Rescue_pct_se"])
    chk("C4 t_half_se = ln2 * SE(k) / k^2 (delta method)",
        d["t_half_se"], float(np.log(2) * d["k_se"] / d["k"] ** 2), 5e-3)
    # The QUOTED t_half interval is the k interval transformed endpoint-wise,
    # not a symmetric Wald interval: t_half = ln2/k is convex and decreasing, so
    # t_half +/- z*SE undercovers (measured 0.865 vs nominal 0.95). Transforming
    # a monotone map preserves the k interval's coverage exactly.
    chk("C4 t_half_CI_lower = ln2 / k_CI_upper (endpoint transform)",
        d["t_half_CI_lower"], float(np.log(2) / d["k_CI_upper"]), 0.02)
    chk("C4 t_half_CI_upper = ln2 / k_CI_lower",
        d["t_half_CI_upper"], float(np.log(2) / d["k_CI_lower"]), 0.02)

    TRUE_TH = float(np.log(2) / TRUE_K)
    rng = np.random.default_rng(7)
    ck = ct = nn = 0
    for _ in range(300):
        yy = 90.0 * np.exp(-TRUE_K * ages) + rng.normal(0, SIGMA, len(ages))
        f = MD.fit_decay(ages, yy)
        if not np.isfinite(f.get("k_CI_lower", np.nan)):
            continue
        nn += 1
        ck += f["k_CI_lower"] <= TRUE_K <= f["k_CI_upper"]
        ct += f["t_half_CI_lower"] <= TRUE_TH <= f["t_half_CI_upper"]
    boolchk("C4 transformed t_half CI inherits the k CI's coverage exactly",
            abs(ck - ct) <= 1, "identical", f"k {ck}/{nn} vs t_half {ct}/{nn}")

    # AICc must follow Burnham & Anderson (2002): K counts sigma^2, so K = p+1.
    # The correction is nonlinear in K, so the wrong convention changes the
    # DIFFERENCE between models with different p and can flip the selection.
    for p_ in (2, 3):
        K = p_ + 1
        n_ = 9
        ref = n_ * np.log(10.0 / n_) + 2 * K + 2 * K * (K + 1) / (n_ - K - 1)
        chk(f"C5 AICc matches Burnham-Anderson at p={p_} (K=p+1)",
            MD._aic(10.0, n_, p_), ref, 1e-9)

    # ── D. MSCI permutation p-value ──────────────────────────────────────────
    def _msci(ks, ls):
        m = pd.DataFrame([{"Genotype": "G", "Treatment": f"t{i}", "Sex": "M",
                           "k": k, "k_se": 0.001} for i, k in enumerate(ks)])
        l = pd.DataFrame([{"Genotype": "G", "Treatment": f"t{i}", "Sex": "M",
                           "Median_Survival_days": d} for i, d in enumerate(ls)])
        return MD.compute_msci(m, l)[0].iloc[0]

    r3 = _msci([0.01, 0.02, 0.03], [50, 40, 30])
    # Tolerances below match the module's documented 3-significant-figure
    # reporting of p-values (0.333, 4.96e-05), not full float precision.
    chk("D1 n=3, |rho|=1: exact p = 2/3! = 0.333", r3["p_Spearman"], 1 / 3, 5e-4)
    boolchk("D1 regression: NOT the asymptotic p = 0.0",
            float(r3["p_Spearman"]) > 0.05, "> 0.05", r3["p_Spearman"])
    boolchk("D1 n=3 is labelled underpowered by construction",
            "Underpowered by construction" in r3["Interpretation"],
            "underpowered", r3["Interpretation"][:40])
    boolchk("D1 verdict is No, not Yes", r3["Significant"] == "No", "No", r3["Significant"])
    chk("D1 p_min_attainable at n=3 = 0.333", r3["p_min_attainable"], 1 / 3, 5e-4)

    r8 = _msci([0.010, 0.012, 0.015, 0.018, 0.021, 0.025, 0.030, 0.036],
               [80, 76, 71, 66, 60, 54, 47, 40])
    chk("D2 n=8, |rho|=1: exact p = 2/8! = 4.96e-05",
        r8["p_Spearman"], 2 / 40320, 1e-7)
    boolchk("D2 small p is not rounded away to 0.0",
            float(r8["p_Spearman"]) > 0, "> 0", r8["p_Spearman"])
    boolchk("D2 a genuine coupling IS detected at adequate n",
            r8["Significant"] == "Yes", "Yes", r8["Significant"])
    boolchk("D3 caveat about ecological correlation is carried",
            "survivor_bias" in str(r8["Caveat"]), "present", str(r8["Caveat"])[:40])

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 82)
    print("DroPhenix motor_decline.py validation (model support, AICc selection,")
    print("retained covariance with CI coverage, exact MSCI permutation p)")
    print("=" * 82)
    print(f"{'Check':<56}{'Expected':>12}{'Got':>10}{'Pass':>4}")
    print("-" * 82)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<56}{str(exp):>12}{str(got):>10}{'OK' if ok else 'FAIL':>4}")
    print("-" * 82)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
