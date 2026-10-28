"""
Validation suite for modules/hierarchical.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. ICC ANOVA estimator vs an EXACT ANALYTIC IDENTITY.
   For equal cluster sizes the Ridout-Demetrio-Firth ANOVA estimator reduces
   algebraically to the classical one-way random-effects ICC computed on the
   expanded 0/1 fly-level data. We compute the latter independently with
   scipy's one-way ANOVA machinery and require agreement to 1e-10. This is a
   closed-form check, not a tolerance-based one.

B. ICC recovery across repeated simulation (unbiasedness).
   Beta-binomial data with known intraclass correlation rho; the mean of the
   ANOVA estimates must recover rho.

C. Beta-binomial regression coefficient recovery.
   Data simulated with known log-odds coefficients; ML must recover them, and
   the Wald intervals must cover at the nominal rate.

D. Beta-binomial reduces to the binomial GLM when there is no overdispersion.
   With rho -> 0 the coefficients must match statsmodels GLM(Binomial) -- an
   independent, peer-reviewed implementation.

E. TYPE I ERROR UNDER CLUSTERING -- the result that matters.
   Under a TRUE NULL (no treatment effect) with clustered data, the naive
   fly-as-independent test must reject far more than 5% of the time, and the
   cluster-aware analyses must reject at approximately 5%. This demonstrates
   the concrete harm of the analysis DroPhenix previously performed.

F. Empirical-logit mixed model recovers a known random-slope trajectory.

Run: ./.venv/Scripts/python.exe validation/validate_hierarchical.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import hierarchical as H  # noqa: E402

FAILURES: list[str] = []
SEED = 20260803


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def simulate_betabinomial(rng, p, rho, n_clusters, m):
    """Draw cluster-level counts from a beta-binomial with mean p and ICC rho."""
    if rho <= 0:
        return rng.binomial(m, p, size=n_clusters)
    s = (1.0 - rho) / rho
    pv = rng.beta(p * s, (1.0 - p) * s, size=n_clusters)
    return rng.binomial(m, pv)


# ─────────────────────────────────────────────────────────────────────────────
def test_A_icc_analytic_identity() -> None:
    print("\nA. ICC ANOVA estimator vs exact analytic one-way random-effects ICC")
    rng = np.random.default_rng(SEED)
    m, k = 20, 12
    counts = simulate_betabinomial(rng, 0.6, 0.12, k, m)
    totals = np.full(k, m)

    ours = H.estimate_icc(counts, totals)

    # Independent computation on expanded 0/1 data
    groups = []
    for y in counts:
        groups.append(np.concatenate([np.ones(int(y)), np.zeros(m - int(y))]))
    all_y = np.concatenate(groups)
    grand = all_y.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb_ref = ssb / (k - 1)
    msw_ref = ssw / (len(all_y) - k)
    icc_ref = (msb_ref - msw_ref) / (msb_ref + (m - 1) * msw_ref)

    check("m0 equals cluster size for balanced design",
          abs(ours["m0"] - m) < 1e-10, f"m0={ours['m0']:.12f} vs m={m}")
    check("MSB matches one-way ANOVA between-group mean square",
          abs(ours["MSB"] - msb_ref) < 1e-12, f"{ours['MSB']:.12f} vs {msb_ref:.12f}")
    check("MSW matches one-way ANOVA within-group mean square",
          abs(ours["MSW"] - msw_ref) < 1e-12, f"{ours['MSW']:.12f} vs {msw_ref:.12f}")
    check("ICC matches classical one-way random-effects ICC exactly",
          abs(ours["icc_raw"] - icc_ref) < 1e-10,
          f"{ours['icc_raw']:.12f} vs {icc_ref:.12f}")

    # Cross-check the ANOVA decomposition itself against scipy's F statistic
    F_scipy = stats.f_oneway(*groups).statistic
    F_ours = ours["MSB"] / ours["MSW"]
    check("MSB/MSW equals scipy f_oneway F statistic",
          abs(F_ours - F_scipy) < 1e-9, f"{F_ours:.9f} vs {F_scipy:.9f}")


def test_B_icc_recovery() -> None:
    print("\nB. ICC recovery over repeated simulation (unbiasedness)")
    rng = np.random.default_rng(SEED + 1)
    for true_rho in (0.0, 0.05, 0.15, 0.30):
        ests = []
        for _ in range(400):
            counts = simulate_betabinomial(rng, 0.55, true_rho, 20, 25)
            r = H.estimate_icc(counts, np.full(20, 25))
            if np.isfinite(r["icc_raw"]):
                ests.append(r["icc_raw"])
        mean_est = float(np.mean(ests))
        # Tolerance widens with rho because the estimator's variance grows
        tol = max(0.02, 0.15 * max(true_rho, 0.05))
        check(f"mean ICC estimate recovers true rho={true_rho:.2f}",
              abs(mean_est - true_rho) < tol,
              f"mean={mean_est:.4f} (n=400 sims, tol={tol:.3f})")


def test_C_betabinomial_recovery() -> None:
    print("\nC. Beta-binomial regression recovers known coefficients")
    rng = np.random.default_rng(SEED + 2)
    true_b0, true_b1, true_rho = 0.4, -0.9, 0.10
    m, k_per = 25, 40

    est_b1, covered = [], 0
    n_sims = 120
    for _ in range(n_sims):
        rows = []
        for grp, lp in (("Ctrl", true_b0), ("Drug", true_b0 + true_b1)):
            p = float(expit(lp))
            counts = simulate_betabinomial(rng, p, true_rho, k_per, m)
            for j, y in enumerate(counts):
                rows.append({"Grp": grp, "VialID": f"{grp}{j}", "Count": int(y), "n": m})
        df = pd.DataFrame(rows)
        try:
            fit = H.fit_betabinomial(df, "C(Grp)")
        except Exception:
            continue
        key = [i for i in fit.params.index if i != "Intercept"][0]
        est_b1.append(float(fit.params[key]))
        lo = float(fit.conf_int.loc[key, "lower"])
        hi = float(fit.conf_int.loc[key, "upper"])
        if lo <= true_b1 <= hi:
            covered += 1

    mean_b1 = float(np.mean(est_b1))
    cov_rate = covered / len(est_b1)
    check("slope recovered (true beta1 = -0.9)",
          abs(mean_b1 - true_b1) < 0.06, f"mean={mean_b1:.4f} over {len(est_b1)} sims")
    check("95% Wald interval coverage is near nominal",
          0.90 <= cov_rate <= 0.99, f"coverage={cov_rate:.3f}")

    # rho recovery on one large dataset
    rows = []
    for grp, lp in (("Ctrl", true_b0), ("Drug", true_b0 + true_b1)):
        counts = simulate_betabinomial(rng, float(expit(lp)), true_rho, 200, m)
        for j, y in enumerate(counts):
            rows.append({"Grp": grp, "VialID": f"{grp}{j}", "Count": int(y), "n": m})
    fit = H.fit_betabinomial(pd.DataFrame(rows), "C(Grp)")
    check("dispersion rho recovered on a large sample (true 0.10)",
          abs(fit.rho - true_rho) < 0.03, f"rho_hat={fit.rho:.4f} +/- {fit.rho_se:.4f}")


def test_D_reduces_to_binomial_glm() -> None:
    print("\nD. Beta-binomial reduces to statsmodels GLM(Binomial) when rho -> 0")
    import statsmodels.api as sm

    rng = np.random.default_rng(SEED + 3)
    m = 30
    rows = []
    for grp, p in (("Ctrl", 0.62), ("Drug", 0.41)):
        for j in range(60):
            rows.append({"Grp": grp, "Count": int(rng.binomial(m, p)), "n": m})
    df = pd.DataFrame(rows)

    fit = H.fit_betabinomial(df, "C(Grp)")

    import patsy
    X = patsy.dmatrix("C(Grp)", df, return_type="dataframe")
    endog = np.column_stack([df["Count"].to_numpy(), (df["n"] - df["Count"]).to_numpy()])
    glm = sm.GLM(endog, np.asarray(X), family=sm.families.Binomial()).fit()

    diffs = np.abs(fit.params.to_numpy() - glm.params)
    check("rho estimated near zero for truly binomial data",
          fit.rho < 0.02, f"rho_hat={fit.rho:.5f}")
    check("coefficients match statsmodels GLM(Binomial)",
          float(diffs.max()) < 0.01,
          f"max |diff| = {float(diffs.max()):.6f}; ours={np.round(fit.params.to_numpy(),5)}, "
          f"statsmodels={np.round(glm.params,5)}")


def test_E_type_I_error() -> None:
    print("\nE. Type I error under clustering -- naive vs cluster-aware")
    rng = np.random.default_rng(SEED + 4)
    m, k_per, n_sims = 20, 10, 1500
    p_true, rho_true = 0.60, 0.10
    alpha = 0.05

    rej_naive = rej_vial = rej_bb = 0
    n_bb = 0
    for i in range(n_sims):
        ca = simulate_betabinomial(rng, p_true, rho_true, k_per, m)
        cb = simulate_betabinomial(rng, p_true, rho_true, k_per, m)

        # NAIVE: pool flies, 2x2 chi-square (what a fly-level test does)
        ya, yb = ca.sum(), cb.sum()
        na = nb = k_per * m
        tab = np.array([[ya, na - ya], [yb, nb - yb]])
        try:
            p_naive = stats.chi2_contingency(tab, correction=False)[1]
        except Exception:
            continue
        rej_naive += int(p_naive < alpha)

        # CLUSTER-AWARE 1: Welch t-test on vial-level empirical logits
        za, _ = H.empirical_logit(ca, np.full(k_per, m))
        zb, _ = H.empirical_logit(cb, np.full(k_per, m))
        p_vial = stats.ttest_ind(za, zb, equal_var=False).pvalue
        rej_vial += int(p_vial < alpha)

        # CLUSTER-AWARE 2: beta-binomial Wald (subsample for runtime)
        if i % 5 == 0:
            rows = []
            for lab, cc in (("A", ca), ("B", cb)):
                for j, y in enumerate(cc):
                    rows.append({"Grp": lab, "Count": int(y), "n": m})
            try:
                f = H.fit_betabinomial(pd.DataFrame(rows), "C(Grp)")
                key = [x for x in f.pvalues.index if x != "Intercept"][0]
                rej_bb += int(float(f.pvalues[key]) < alpha)
                n_bb += 1
            except Exception:
                pass

    t_naive = rej_naive / n_sims
    t_vial = rej_vial / n_sims
    t_bb = rej_bb / n_bb if n_bb else float("nan")

    # Binomial 99% envelope for a nominal 5% test
    se = np.sqrt(alpha * (1 - alpha) / n_sims)
    hi = alpha + 2.576 * se

    print(f"      naive fly-level chi-square : {t_naive:.3%} rejection under the null")
    print(f"      vial-level t on emp. logit : {t_vial:.3%}")
    print(f"      beta-binomial Wald         : {t_bb:.3%}  (n={n_bb})")
    print(f"      nominal alpha = {alpha:.0%}, upper 99% bound for a calibrated test = {hi:.3%}")

    check("naive fly-level test is anticonservative under clustering",
          t_naive > 0.12,
          f"{t_naive:.3%} -- inflated {t_naive / alpha:.1f}x above nominal 5%")
    check("vial-level test controls type I error at ~5%",
          t_vial <= hi + 0.015, f"{t_vial:.3%} (bound {hi:.3%})")
    check("beta-binomial controls type I error at ~5%",
          np.isfinite(t_bb) and t_bb <= hi + 0.03, f"{t_bb:.3%}")

    # The same demonstration surfaced through the user-facing report
    rows = []
    for lab in ("Vehicle", "Drug"):
        cc = simulate_betabinomial(rng, p_true, rho_true, k_per, m)
        for j, y in enumerate(cc):
            rows.append({"Treatment": lab, "VialID": f"{lab}{j}", "Count": int(y), "n": m})
    rep = H.pseudoreplication_report(pd.DataFrame(rows), "Treatment", "VialID")
    check("pseudoreplication_report runs and reports ICC + design effect",
          rep["usable"] and np.isfinite(rep["icc"]) and np.isfinite(rep["design_effect"]),
          f"ICC={rep.get('icc'):.3f}, DEFF={rep.get('design_effect'):.2f}")

    # It must refuse a confounded design rather than return a number
    bad = pd.DataFrame([
        {"Treatment": "Vehicle", "VialID": "V1", "Count": 12, "n": 20},
        {"Treatment": "Drug", "VialID": "V2", "Count": 18, "n": 20},
    ])
    rep_bad = H.pseudoreplication_report(bad, "Treatment", "VialID")
    check("refuses a design with one vial per treatment",
          not rep_bad["usable"] and "one" in rep_bad.get("reason", "").lower(),
          rep_bad.get("reason", "")[:90])


def test_F_mixed_slope_recovery() -> None:
    print("\nF. Empirical-logit mixed model recovers a known age slope")
    rng = np.random.default_rng(SEED + 5)
    ages = np.array([5, 15, 25, 35, 45], dtype=float)
    true_slope = -0.06          # logit units per day
    true_intercept = 2.2
    m, n_vials = 25, 30

    rows = []
    for v in range(n_vials):
        b0 = rng.normal(0, 0.35)          # vial random intercept
        b1 = rng.normal(0, 0.012)         # vial random slope
        for a in ages:
            lp = (true_intercept + b0) + (true_slope + b1) * a
            p = float(expit(lp))
            rows.append({
                "VialID": f"V{v}", "Age": a, "n": m,
                "Count": int(rng.binomial(m, p)),
            })
    df = pd.DataFrame(rows)

    res = H.fit_mixed_logit(df, fixed_rhs="Age", group_col="VialID", random_slope="Age")
    fe = res["fixed_effects"]
    slope = float(fe.loc["Age", "coef"])
    lo, hi = float(fe.loc["Age", "ci_lower"]), float(fe.loc["Age", "ci_upper"])

    check("age slope recovered on the logit scale (true -0.06)",
          abs(slope - true_slope) < 0.012, f"slope={slope:.5f} [{lo:.5f}, {hi:.5f}]")
    check("95% CI for the slope covers the truth",
          lo <= true_slope <= hi, f"[{lo:.5f}, {hi:.5f}]")
    check("random-intercept variance is estimated as positive",
          np.isfinite(res["random_intercept_var"]) and res["random_intercept_var"] > 0,
          f"var={res['random_intercept_var']:.4f}, model ICC={res['icc_model']:.3f}")

    # Direction of the clustering correction depends on WHERE the predictor
    # varies. This is a real and frequently-misunderstood distinction, so both
    # directions are asserted explicitly.
    #
    #   WITHIN-cluster predictor (Age: every vial measured at every age)
    #     -> the vial random intercept absorbs between-vial variation, so the
    #        mixed-model SE is SMALLER than the naive one. Clustering buys
    #        precision here.
    #   BETWEEN-cluster predictor (Treatment: assigned to whole vials)
    #     -> the naive SE is too small because it counts flies as independent
    #        replicates of the treatment. Clustering costs precision here, and
    #        that cost is the correction.
    import statsmodels.api as sm

    z, _ = H.empirical_logit(df["Count"], df["n"])
    X = sm.add_constant(df["Age"].to_numpy())
    se_naive_within = float(sm.OLS(z, X).fit().bse[1])
    se_mixed_within = float(fe.loc["Age", "std_err"])
    check("within-vial predictor (Age): clustering IMPROVES precision",
          se_mixed_within < se_naive_within,
          f"naive SE={se_naive_within:.5f} vs mixed SE={se_mixed_within:.5f} "
          f"({se_naive_within / se_mixed_within:.2f}x tighter once vial variance is removed)")

    # Now the between-vial case on the same generative model: assign whole
    # vials to a treatment that has NO true effect.
    rng2 = np.random.default_rng(SEED + 6)
    rows2 = []
    for v in range(n_vials):
        b0 = rng2.normal(0, 0.35)
        arm = "A" if v < n_vials // 2 else "B"
        for a in ages:
            p = float(expit((true_intercept + b0) + true_slope * a))
            rows2.append({"VialID": f"W{v}", "Arm": arm, "Age": a, "n": m,
                          "Count": int(rng2.binomial(m, p))})
    df2 = pd.DataFrame(rows2)

    z2, _ = H.empirical_logit(df2["Count"], df2["n"])
    arm_dummy = (df2["Arm"] == "B").astype(float).to_numpy()
    X2 = sm.add_constant(np.column_stack([arm_dummy, df2["Age"].to_numpy()]))
    se_naive_between = float(sm.OLS(z2, X2).fit().bse[1])

    res2 = H.fit_mixed_logit(df2, fixed_rhs="C(Arm) + Age", group_col="VialID")
    arm_row = [i for i in res2["fixed_effects"].index if "Arm" in i][0]
    se_mixed_between = float(res2["fixed_effects"].loc[arm_row, "std_err"])

    check("between-vial predictor (Arm): naive SE is too small",
          se_naive_between < se_mixed_between,
          f"naive SE={se_naive_between:.5f} vs mixed SE={se_mixed_between:.5f} "
          f"({se_mixed_between / se_naive_between:.2f}x larger once vials are respected)")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/hierarchical.py")
    print("=" * 78)
    test_A_icc_analytic_identity()
    test_B_icc_recovery()
    test_C_betabinomial_recovery()
    test_D_reduces_to_binomial_glm()
    test_E_type_I_error()
    test_F_mixed_slope_recovery()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL HIERARCHICAL VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
