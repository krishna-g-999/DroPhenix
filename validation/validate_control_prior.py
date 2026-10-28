"""
Validation suite for modules/control_prior.py  (robust MAP historical-control priors)

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. Vectorised simulation kernels vs the readable single-case reference
   implementations -- agreement to 1e-12. The fast path used for operating
   characteristics must be algebraically identical to the documented one.

B. Conjugate mixture update vs BRUTE-FORCE NUMERICAL INTEGRATION on a dense
   grid. The closed form is checked against the definition of a posterior, not
   against another closed form.

C. Effective sample size vs an EXACT ANALYTIC LIMIT. As the between-experiment
   heterogeneity tau -> 0, the MAP prior must carry exactly the information of
   the pooled historical flies, so ESS -> sum(n_historical). This is a closed-form
   target, not a tolerance guess.

D. The MAP prior must be WIDER than the posterior for the historical mean,
   because a new experiment is a new draw from the between-experiment
   distribution. Omitting that tau^2 term is the classic error; this test would
   catch it.

E. Prior-data conflict diagnostic: calibrated type I error when the new control
   really is exchangeable with history, and power when it is not.

F. Self-limiting behaviour: under strong conflict the robust posterior must
   collapse onto the concurrent-only estimate.

G. OPERATING CHARACTERISTICS -- the result that decides whether borrowing is
   safe. Type I error is measured across a grid of drift and reported honestly,
   including the regime where borrowing IS harmful.

H. Fly savings: `recommended_design` must return a design that meets its own
   type I constraint, and must refuse when no design does.

Run: ./.venv/Scripts/python.exe validation/validate_control_prior.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import control_prior as CP  # noqa: E402

FAILURES: list[str] = []
SEED = 20260803


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def _history(n_arms=8, n_each=100, p=0.75, tau=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    ps = expit(logit(p) + rng.normal(0, tau, n_arms))
    n = np.full(n_arms, n_each)
    y = rng.binomial(n, ps)
    return y, n


# ─────────────────────────────────────────────────────────────────────────────
def test_A_vectorised_matches_reference() -> None:
    print("\nA. Vectorised kernels match the single-case reference implementations")
    y_hist, n_hist = _history()
    prior = CP.build_map_prior(y_hist, n_hist)["prior"].compress(3, seed=SEED)
    rob = CP.robustify(prior, 0.8)

    rng = np.random.default_rng(SEED + 1)
    yc = rng.binomial(50, 0.72, size=200)
    yt = rng.binomial(80, 0.60, size=200)

    y_c, v_c = CP._arm_logit_summaries(yc, np.full(200, 50))
    y_t, v_t = CP._arm_logit_summaries(yt, np.full(200, 80))

    w, m, v = CP._vectorised_posterior(rob, y_c, v_c)
    pb_vec = CP._vectorised_prob_better(w, m, v, y_t, v_t)
    cp_vec = CP._vectorised_conflict_p(CP.informative_component(rob), y_c, v_c)

    max_post_diff = 0.0
    max_pb_diff = 0.0
    max_cp_diff = 0.0
    for i in range(200):
        ref_post = CP.update_posterior(rob, yc[i], 50)
        max_post_diff = max(max_post_diff,
                            float(np.max(np.abs(np.sort(ref_post.means) - np.sort(m[i])))))
        pb_ref = CP.prob_treatment_better(ref_post, yt[i], 80)
        max_pb_diff = max(max_pb_diff, abs(pb_ref - pb_vec[i]))
        cp_ref = CP.prior_data_conflict(rob, yc[i], 50)["p_value"]
        max_cp_diff = max(max_cp_diff, abs(cp_ref - cp_vec[i]))

    check("vectorised posterior means match reference",
          max_post_diff < 1e-12, f"max |diff| = {max_post_diff:.3e}")
    check("vectorised P(treat > control) matches reference",
          max_pb_diff < 1e-12, f"max |diff| = {max_pb_diff:.3e}")
    check("vectorised conflict p-value matches reference",
          max_cp_diff < 1e-12, f"max |diff| = {max_cp_diff:.3e}")


def test_B_conjugate_update_vs_numerical_integration() -> None:
    print("\nB. Closed-form conjugate update vs brute-force numerical integration")
    y_hist, n_hist = _history()
    prior = CP.build_map_prior(y_hist, n_hist)["prior"].compress(3, seed=SEED)
    rob = CP.robustify(prior, 0.8)

    yc, nc = 68, 100
    post = CP.update_posterior(rob, yc, nc)

    y0, v0 = CP._arm_logit_summaries([yc], [nc])
    y0, s2 = float(y0[0]), float(v0[0])

    grid = np.linspace(-6, 8, 40001)
    unnorm = rob.pdf(grid) * stats.norm.pdf(y0, grid, np.sqrt(s2))
    norm_const = integrate.trapezoid(unnorm, grid)
    num_mean = integrate.trapezoid(grid * unnorm, grid) / norm_const
    num_var = integrate.trapezoid((grid - num_mean) ** 2 * unnorm, grid) / norm_const

    check("posterior mean matches numerical integration",
          abs(post.mean - num_mean) < 1e-8,
          f"closed form {post.mean:.10f} vs numerical {num_mean:.10f}")
    check("posterior variance matches numerical integration",
          abs(post.var - num_var) < 1e-8,
          f"closed form {post.var:.10f} vs numerical {num_var:.10f}")

    # P(treat > control) against a 2-D numerical integral
    yt, nt = 55, 100
    yt_l, vt_l = CP._arm_logit_summaries([yt], [nt])
    yt_l, st2 = float(yt_l[0]), float(vt_l[0])
    pb_closed = CP.prob_treatment_better(post, yt, nt)
    dens_c = post.pdf(grid)
    dens_c = dens_c / integrate.trapezoid(dens_c, grid)
    surv = stats.norm.sf(grid, yt_l, np.sqrt(st2))
    pb_num = float(integrate.trapezoid(dens_c * surv, grid))
    check("P(treat > control) matches numerical integration",
          abs(pb_closed - pb_num) < 1e-8,
          f"closed form {pb_closed:.10f} vs numerical {pb_num:.10f}")


def test_C_ess_analytic_limit() -> None:
    print("\nC. Effective sample size vs the exact tau -> 0 analytic limit")
    # Identical arms + a near-zero heterogeneity prior forces tau -> 0, where
    # the MAP predictive variance is exactly the pooled sampling variance, so
    # ESS must equal the total number of historical flies.
    n_arms, n_each, p = 6, 120, 0.70
    y = np.full(n_arms, int(round(p * n_each)))
    n = np.full(n_arms, n_each)

    built = CP.build_map_prior(y, n, tau_prior_scale=1e-4, tau_max=1e-3)
    ess = CP.prior_effective_sample_size(built["prior"])
    total = float(n.sum())
    rel = abs(ess["ess_flies"] - total) / total

    check("ESS approaches total historical flies as tau -> 0",
          rel < 0.02,
          f"ESS = {ess['ess_flies']:.1f} vs {total:.0f} historical flies "
          f"(relative error {rel:.3%})")

    # With real heterogeneity the ESS must be substantially LESS -- borrowing
    # from disagreeing experiments is worth less, and the prior must say so.
    y_h, n_h = _history(n_arms=6, n_each=120, p=0.70, tau=0.35)
    built_h = CP.build_map_prior(y_h, n_h)
    ess_h = CP.prior_effective_sample_size(built_h["prior"])["ess_flies"]
    check("heterogeneous history yields a much smaller ESS",
          ess_h < 0.5 * total,
          f"ESS = {ess_h:.1f} vs {total:.0f} flies "
          f"(tau = {built_h['tau_mean']:.3f})")


def test_D_map_wider_than_mean_posterior() -> None:
    print("\nD. MAP predictive is wider than the posterior for the historical mean")
    y_hist, n_hist = _history(tau=0.25)
    built = CP.build_map_prior(y_hist, n_hist)
    prior = built["prior"]

    # posterior SD for mu alone (i.e. omitting the tau^2 predictive term)
    y, v = CP._arm_logit_summaries(y_hist, n_hist)
    tau = built["tau_mean"]
    w = 1.0 / (v + tau ** 2)
    se_mu = float(np.sqrt(1.0 / w.sum()))

    check("predictive SD exceeds the SD of the historical mean",
          prior.sd > se_mu * 1.2,
          f"MAP predictive SD = {prior.sd:.4f} vs SE of the mean = {se_mu:.4f} "
          f"(ratio {prior.sd / se_mu:.2f}); the difference is the tau^2 term")
    check("estimated between-experiment tau is positive and finite",
          0 < built["tau_mean"] < 5, f"tau = {built['tau_mean']:.4f}")

    # A single historical arm must be refused outright.
    try:
        CP.build_map_prior([70], [100])
        refused = False
        msg = "no exception raised"
    except ValueError as exc:
        refused = True
        msg = str(exc)[:80]
    check("refuses to build a prior from one historical arm", refused, msg)


def test_E_conflict_calibration_and_power() -> None:
    print("\nE. Prior-data conflict: calibration under exchangeability, power under drift")
    y_hist, n_hist = _history(n_arms=10, n_each=100, p=0.75, tau=0.20)
    built = CP.build_map_prior(y_hist, n_hist)
    prior = built["prior"]
    p_hist = float(expit(prior.mean))

    rng = np.random.default_rng(SEED + 5)
    n_c, n_sim = 100, 6000

    # Under exchangeability the new control is a NEW DRAW from the
    # between-experiment distribution, not from the historical mean.
    tau = built["tau_mean"]
    theta_new = rng.normal(prior.mean, tau, size=n_sim)
    y_exch = rng.binomial(n_c, expit(theta_new))
    yl, vl = CP._arm_logit_summaries(y_exch, np.full(n_sim, n_c))
    p_exch = CP._vectorised_conflict_p(prior, yl, vl)
    rate_exch = float(np.mean(p_exch < 0.05))

    check("flag rate under true exchangeability is near the nominal 5%",
          rate_exch < 0.11, f"{rate_exch:.2%} (nominal 5%)")

    # Under a large drift the diagnostic must fire often.
    for drift, want in ((0.15, 0.45), (0.25, 0.80)):
        p_drift = float(np.clip(p_hist - drift, 0.02, 0.98))
        y_d = rng.binomial(n_c, p_drift, size=n_sim)
        yl, vl = CP._arm_logit_summaries(y_d, np.full(n_sim, n_c))
        rate = float(np.mean(CP._vectorised_conflict_p(prior, yl, vl) < 0.05))
        check(f"detects a drift of {drift:.0%} (control {p_hist:.0%} -> {p_drift:.0%})",
              rate >= want, f"flag rate {rate:.1%} (required >= {want:.0%})")


def test_F_self_limiting_under_conflict() -> None:
    print("\nF. Robust prior switches itself off under conflict")
    y_hist, n_hist = _history(n_arms=8, n_each=100, p=0.75, tau=0.15)
    prior = CP.build_map_prior(y_hist, n_hist)["prior"]
    rob = CP.robustify(prior, 0.8)
    p_hist = float(expit(prior.mean))

    n_c = 100
    # concordant control: borrowing should pull the estimate toward history
    y_ok = int(round(p_hist * n_c))
    post_ok = CP.update_posterior(rob, y_ok, n_c)

    # strongly conflicting control
    y_bad = int(round(max(p_hist - 0.30, 0.05) * n_c))
    post_bad = CP.update_posterior(rob, y_bad, n_c)
    concurrent_only = float(y_bad) / n_c

    gap = abs(float(expit(post_bad.mean)) - concurrent_only)
    check("under conflict the posterior collapses onto the concurrent control",
          gap < 0.05,
          f"posterior {expit(post_bad.mean):.3f} vs concurrent-only {concurrent_only:.3f} "
          f"(gap {gap:.3f}); historical prior was {p_hist:.3f}")

    # and under agreement it is TIGHTER than the concurrent data alone
    y_l, v_l = CP._arm_logit_summaries([y_ok], [n_c])
    check("under agreement the posterior is tighter than concurrent data alone",
          post_ok.sd < float(np.sqrt(v_l[0])),
          f"posterior SD {post_ok.sd:.4f} vs concurrent-only SD {float(np.sqrt(v_l[0])):.4f}")

    # a non-robust prior must NOT self-limit -- this is what robustification buys
    post_bad_full = CP.update_posterior(prior, y_bad, n_c)
    check("a non-robustified prior does NOT self-limit (so robustification matters)",
          abs(float(expit(post_bad_full.mean)) - concurrent_only) > gap,
          f"full-MAP posterior {expit(post_bad_full.mean):.3f} stays "
          f"{abs(float(expit(post_bad_full.mean)) - concurrent_only):.3f} away from the "
          f"concurrent value, vs {gap:.3f} for the robust prior")


def test_G_operating_characteristics() -> None:
    print("\nG. Operating characteristics across drift  [the decisive result]")
    y_hist, n_hist = _history(n_arms=10, n_each=100, p=0.75, tau=0.15)
    p_hist = float(expit(CP.build_map_prior(y_hist, n_hist)["prior"].mean))

    grid = [round(p_hist + d, 3) for d in (-0.10, -0.05, 0.0, 0.05, 0.10)]
    grid = [float(np.clip(g, 0.05, 0.95)) for g in grid]

    oc = CP.operating_characteristics(
        y_hist, n_hist, n_control=40, n_treat=100,
        p_control_true_grid=grid, true_effect_logit=0.0,
        robust_weight=0.8, n_sim=8000, seed=SEED + 7,
    )
    piv = oc.pivot(index="p_control_true", columns="method", values="rejection_rate")
    print("\n      Type I error (true effect = 0), historical control rate "
          f"= {p_hist:.3f}:")
    print("      " + piv.round(4).to_string().replace("\n", "\n      "))
    flags = oc[oc.method == "robust_map"].set_index("p_control_true")["conflict_flag_rate"]
    print("\n      conflict-flag rate (robust MAP): "
          + ", ".join(f"{k:.2f}:{v:.1%}" for k, v in flags.items()))

    at_hist = piv.loc[min(piv.index, key=lambda x: abs(x - p_hist))]
    check("at zero drift, borrowing does not inflate type I error",
          at_hist["robust_map"] <= at_hist["concurrent_only"] + 0.02,
          f"robust_map {at_hist['robust_map']:.3f} vs concurrent_only "
          f"{at_hist['concurrent_only']:.3f}")

    worst_rob = float(piv["robust_map"].max())
    worst_full = float(piv["full_map"].max())
    check("robustification reduces the worst-case type I error vs full MAP",
          worst_rob <= worst_full + 1e-9,
          f"worst robust_map {worst_rob:.3f} vs worst full_map {worst_full:.3f}")

    check("type I error DOES inflate under drift (reported, not hidden)",
          worst_rob > at_hist["concurrent_only"],
          f"worst-case {worst_rob:.3f} at drift, vs {at_hist['concurrent_only']:.3f} "
          f"for concurrent-only -- this is the cost of borrowing and it is why "
          f"operating characteristics must be simulated before adopting it")

    # Power must be materially better than concurrent-only at zero drift,
    # otherwise borrowing buys nothing.
    oc_pw = CP.operating_characteristics(
        y_hist, n_hist, n_control=40, n_treat=100,
        p_control_true_grid=[p_hist], true_effect_logit=0.8,
        robust_weight=0.8, n_sim=8000, seed=SEED + 8,
    )
    pw = oc_pw.set_index("method")["rejection_rate"]
    print(f"\n      Power at zero drift (true logit effect 0.8): "
          f"concurrent_only {pw['concurrent_only']:.3f}, robust_map {pw['robust_map']:.3f}")
    check("borrowing increases power at zero drift",
          pw["robust_map"] > pw["concurrent_only"],
          f"{pw['robust_map']:.3f} vs {pw['concurrent_only']:.3f}")


def test_H_recommended_design() -> None:
    print("\nH. Design recommendation honours its own type I constraint")
    y_hist, n_hist = _history(n_arms=12, n_each=100, p=0.75, tau=0.10)
    rec = CP.recommended_design(
        y_hist, n_hist, n_control_standard=100, n_treat=100,
        max_type_I=0.10, tolerable_drift=0.05, n_sim=3000, seed=SEED + 9,
    )
    print("      " + rec["verdict"].replace("\n", "\n      ")[:400])
    tab = rec["search_table"]
    print("\n      " + tab.round(4).to_string(index=False).replace("\n", "\n      "))

    if rec["recommended_n_control"] is not None:
        row = tab[tab["n_control"] == rec["recommended_n_control"]].iloc[0]
        check("recommended design meets the stated type I bound",
              bool(row["worst_type_I"] <= rec["max_type_I_allowed"]),
              f"n_control={rec['recommended_n_control']}, worst type I "
              f"{row['worst_type_I']:.4f} <= {rec['max_type_I_allowed']:.3f}")
        check("recommended design retains a concurrent control",
              rec["recommended_n_control"] >= 10,
              f"n_control = {rec['recommended_n_control']} (never zero, by design)")
    else:
        check("refusal is accompanied by an explanation",
              "NOT recommended" in rec["verdict"], rec["verdict"][:100])

    # An impossible constraint must be refused, not fudged.
    rec_bad = CP.recommended_design(
        y_hist, n_hist, n_control_standard=100, n_treat=100,
        max_type_I=0.001, tolerable_drift=0.15, n_sim=2000, seed=SEED + 10,
    )
    check("refuses when no design can meet the constraint",
          rec_bad["recommended_n_control"] is None,
          f"returned {rec_bad['recommended_n_control']}; "
          f"verdict starts: {rec_bad['verdict'][:70]}")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/control_prior.py  (robust MAP historical controls)")
    print("=" * 78)
    test_A_vectorised_matches_reference()
    test_B_conjugate_update_vs_numerical_integration()
    test_C_ess_analytic_limit()
    test_D_map_wider_than_mean_posterior()
    test_E_conflict_calibration_and_power()
    test_F_self_limiting_under_conflict()
    test_G_operating_characteristics()
    test_H_recommended_design()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CONTROL-PRIOR VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
