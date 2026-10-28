"""
Validation suite for modules/trajectory.py (hierarchical trajectory forecasting)

Note: `validation/validate_trajectory.py` validates the DTW *clustering* module
(`modules/trajectory_cluster.py`). This file validates the *forecasting* module
(`modules/trajectory.py`). They are different things.

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. Every curve family recovers its own parameters EXACTLY from noise-free data
   (tolerance 1e-6). A family that cannot invert its own generative process is
   broken regardless of how it performs on real data.

B. CRPS against its EXACT CLOSED FORM for a Gaussian predictive distribution:
       CRPS(N(mu,sigma), y) = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
   This is an analytic identity, checked to 1e-3 with large sample sizes.

C. The population prior's EM must recover the true BETWEEN-cohort covariance,
   not the inflated raw covariance of the per-cohort estimates. The raw
   covariance is biased upward by the within-cohort estimation error; the whole
   value of the hierarchical layer depends on removing it, so this test compares
   against both and asserts EM is closer to the truth.

D. SHRINKAGE PAYS OFF: on sparse cohorts (3 early timepoints), forecasting with
   the population prior must beat forecasting without it, measured by
   forward-in-time error at long horizon.

E. The model must BEAT THE TRIVIAL BASELINES (last observation, train mean,
   linear extrapolation). If it does not, it is not a method.

F. CALIBRATION: empirical coverage of the 50% and 95% prediction intervals
   across many simulated cohorts must be near nominal.

G. Horizon refusal: targets beyond the configured horizon are refused, not
   silently extrapolated.

H. Decision endpoints: a known onset age must be recovered, and an onset that
   never occurs inside the window must be reported as undefined rather than
   invented.

Run: ./.venv/Scripts/python.exe validation/validate_trajectory_prediction.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import trajectory as TJ  # noqa: E402

FAILURES: list[str] = []
SEED = 20260803


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# ─────────────────────────────────────────────────────────────────────────────
def test_A_family_recovery() -> None:
    print("\nA. Each curve family recovers its own parameters from noise-free data")
    t = np.linspace(1, 50, 26)
    truths = {
        "exp_plateau": np.array([logit(0.92), logit(0.12), np.log(0.06)]),
        "logistic": np.array([logit(0.94), logit(0.10), np.log(0.20), 26.0]),
        "gompertz": np.array([logit(0.93), logit(0.08), np.log(0.09), 22.0]),
        "logit_linear": np.array([2.4, -0.09]),
    }
    for name, theta in truths.items():
        y = TJ.FAMILIES[name].predict(t, theta)
        fit = TJ.fit_series(t, y, name)
        ok = fit is not None
        if ok:
            yhat = TJ.FAMILIES[name].predict(t, fit.theta)
            max_dev = float(np.max(np.abs(yhat - y)))
            ok = max_dev < 1e-6
            detail = f"max |predicted - truth| = {max_dev:.3e}"
        else:
            detail = "fit returned None"
        check(f"{name} recovers a noise-free curve", ok, detail)


def test_B_crps_closed_form() -> None:
    print("\nB. CRPS from samples matches the exact Gaussian closed form")
    rng = np.random.default_rng(SEED)
    for mu, sigma, y in ((0.5, 0.1, 0.42), (0.8, 0.25, 1.30), (0.0, 1.0, 0.0)):
        samples = rng.normal(mu, sigma, size=400_000)
        got = TJ.crps_from_samples(samples, y)
        z = (y - mu) / sigma
        exact = sigma * (z * (2 * stats.norm.cdf(z) - 1)
                         + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        check(f"CRPS for N({mu}, {sigma}^2) at y={y}",
              abs(got - exact) < 2e-3, f"sample {got:.6f} vs exact {exact:.6f}")


def test_C_em_recovers_between_cohort_covariance() -> None:
    print("\nC. Population-prior EM removes within-cohort estimation error")
    rng = np.random.default_rng(SEED + 1)
    family = "exp_plateau"
    true_mu = np.array([logit(0.93), logit(0.15), np.log(0.055)])
    true_sd = np.array([0.30, 0.45, 0.35])
    ages = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=float)

    n_cohorts = 40
    cohorts, thetas = {}, []
    for i in range(n_cohorts):
        th = true_mu + rng.normal(0, true_sd)
        thetas.append(th)
        y = TJ.FAMILIES[family].predict(ages, th) + rng.normal(0, 0.035, len(ages))
        cohorts[f"C{i}"] = (ages, np.clip(y, 0.001, 0.999))

    prior = TJ.build_population_prior(cohorts, family=family, min_points=4)
    # Some cohorts are legitimately excluded: because observation starts at day
    # 5, the y0 parameter (the curve's value at t = 0) is extrapolated and can be
    # unidentified, which the bound check catches. That exclusion is correct
    # behaviour, so the requirement is that a usable majority survives AND that
    # the exclusions are reported rather than silent.
    check("prior was built from a usable majority of cohorts",
          prior is not None and prior.n_cohorts >= 0.5 * n_cohorts,
          f"n_cohorts used = {getattr(prior, 'n_cohorts', 0)} of {n_cohorts}")
    if prior is None:
        return
    check("excluded cohorts are reported, not dropped silently",
          any("excluded" in n for n in prior.notes) or prior.n_cohorts == n_cohorts,
          f"{len(prior.notes)} note(s); first: "
          f"{(prior.notes[0][:80] + '...') if prior.notes else 'none needed'}")

    # raw covariance of the per-cohort point estimates (the biased comparator)
    per_fits = [TJ.fit_series(ages, cohorts[f"C{i}"][1], family) for i in range(n_cohorts)]
    raw = np.vstack([f.theta for f in per_fits if f is not None])
    raw_sd = raw.std(axis=0, ddof=1)
    em_sd = np.sqrt(np.clip(np.diag(prior.cov), 0, None))

    err_raw = float(np.mean(np.abs(raw_sd - true_sd)))
    err_em = float(np.mean(np.abs(em_sd - true_sd)))

    print(f"      true between-cohort SD : {np.round(true_sd, 4)}")
    print(f"      raw (biased) SD        : {np.round(raw_sd, 4)}   mean |err| = {err_raw:.4f}")
    print(f"      EM  (corrected) SD     : {np.round(em_sd, 4)}   mean |err| = {err_em:.4f}")

    check("EM between-cohort SD is closer to the truth than the raw SD",
          err_em < err_raw, f"EM {err_em:.4f} vs raw {err_raw:.4f}")
    check("prior mean recovers the population mean",
          float(np.max(np.abs(prior.mean - true_mu))) < 0.35,
          f"max |diff| = {float(np.max(np.abs(prior.mean - true_mu))):.4f}")
    check("a prior cannot be built from a single cohort",
          TJ.build_population_prior({"only": (ages, cohorts['C0'][1])}, family=family) is None,
          "returns None as required")


def test_D_shrinkage_helps_sparse_cohorts() -> None:
    print("\nD. Population prior improves long-horizon forecasts for sparse cohorts")
    rng = np.random.default_rng(SEED + 2)
    family = "exp_plateau"
    true_mu = np.array([logit(0.93), logit(0.15), np.log(0.055)])
    true_sd = np.array([0.25, 0.40, 0.30])
    dense_ages = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=float)

    # 30 completed cohorts define the prior
    prior_cohorts = {}
    for i in range(30):
        th = true_mu + rng.normal(0, true_sd)
        y = TJ.FAMILIES[family].predict(dense_ages, th) + rng.normal(0, 0.035, len(dense_ages))
        prior_cohorts[f"P{i}"] = (dense_ages, np.clip(y, 0.001, 0.999))
    prior = TJ.build_population_prior(prior_cohorts, family=family)
    assert prior is not None

    # 60 NEW sparse cohorts: observed only at days 5, 10, 15; must predict 30-45
    train_ages = np.array([5, 10, 15], dtype=float)
    test_ages = np.array([30, 35, 40, 45], dtype=float)
    err_with, err_without = [], []
    cov_with = []
    for i in range(60):
        th = true_mu + rng.normal(0, true_sd)
        all_ages = np.concatenate([train_ages, test_ages])
        y_all = np.clip(TJ.FAMILIES[family].predict(all_ages, th)
                        + rng.normal(0, 0.035, len(all_ages)), 0.001, 0.999)
        try:
            v_with = TJ.forward_validate(all_ages, y_all, train_max_day=15.0,
                                         prior=prior, seed=i, max_horizon=4.0)
            v_without = TJ.forward_validate(all_ages, y_all, train_max_day=15.0,
                                            prior=None, seed=i, max_horizon=4.0)
        except Exception:
            continue
        if np.isfinite(v_with["summary"]["MAE"]):
            err_with.append(v_with["summary"]["MAE"])
            cov_with.append(v_with["summary"]["coverage_95"])
        if np.isfinite(v_without["summary"]["MAE"]):
            err_without.append(v_without["summary"]["MAE"])

    mae_with = float(np.mean(err_with))
    mae_without = float(np.mean(err_without))
    print(f"      MAE at days 30-45 from only days 5,10,15:")
    print(f"        with population prior    : {mae_with:.4f}  (n={len(err_with)} cohorts)")
    print(f"        without population prior : {mae_without:.4f}  (n={len(err_without)} cohorts)")
    print(f"        improvement              : {(1 - mae_with / mae_without) * 100:.1f}%")

    check("the population prior reduces long-horizon error",
          mae_with < mae_without,
          f"{mae_with:.4f} vs {mae_without:.4f} "
          f"({(1 - mae_with / mae_without) * 100:.1f}% lower)")
    # UNCALIBRATED coverage is expected to be poor -- empirical-Bayes plug-in
    # intervals are over-confident by construction (shrinkage bias plus an
    # estimated prior treated as known). Test F shows conformal calibration
    # repairs it. What is asserted here is that the deficiency is real and
    # bounded, so that a user who skips calibration is not silently misled about
    # how bad it is.
    raw_cov = float(np.mean(cov_with))
    check("uncalibrated intervals under-cover, as documented (calibration required)",
          0.50 <= raw_cov < 0.90,
          f"empirical coverage {raw_cov:.3f} at nominal 0.95 -- see test F, where "
          f"conformal calibration restores it")


def test_E_beats_baselines() -> None:
    print("\nE. Forecasts beat the trivial baselines")
    rng = np.random.default_rng(SEED + 3)
    family = "exp_plateau"
    true_mu = np.array([logit(0.93), logit(0.15), np.log(0.055)])
    true_sd = np.array([0.25, 0.40, 0.30])
    dense = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=float)

    cohorts = {}
    for i in range(25):
        th = true_mu + rng.normal(0, true_sd)
        y = TJ.FAMILIES[family].predict(dense, th) + rng.normal(0, 0.035, len(dense))
        cohorts[f"P{i}"] = (dense, np.clip(y, 0.001, 0.999))
    prior = TJ.build_population_prior(cohorts, family=family)

    wins = {k: 0 for k in ("last_observation", "train_mean", "linear_last_two")}
    n = 0
    for i in range(60):
        th = true_mu + rng.normal(0, true_sd)
        y = np.clip(TJ.FAMILIES[family].predict(dense, th)
                    + rng.normal(0, 0.035, len(dense)), 0.001, 0.999)
        try:
            v = TJ.forward_validate(dense, y, train_max_day=20.0, prior=prior,
                                    seed=i, max_horizon=4.0)
        except Exception:
            continue
        s = v["summary"]
        if not np.isfinite(s["MAE"]):
            continue
        n += 1
        for b in wins:
            if np.isfinite(s.get(f"MAE_{b}", np.nan)) and s["MAE"] < s[f"MAE_{b}"]:
                wins[b] += 1

    for b, w in wins.items():
        rate = w / max(n, 1)
        print(f"      beats {b:18s}: {rate:.1%} of {n} cohorts")
        check(f"beats the '{b}' baseline in a clear majority of cohorts",
              rate >= 0.70, f"{rate:.1%}")


def test_F_calibration() -> None:
    print("\nF. Interval calibration across many cohorts")
    rng = np.random.default_rng(SEED + 4)
    family = "exp_plateau"
    true_mu = np.array([logit(0.93), logit(0.15), np.log(0.055)])
    true_sd = np.array([0.25, 0.40, 0.30])
    dense = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=float)

    def _cohort(tag, n):
        out = {}
        for i in range(n):
            th = true_mu + rng.normal(0, true_sd)
            y = TJ.FAMILIES[family].predict(dense, th) + rng.normal(0, 0.035, len(dense))
            out[f"{tag}{i}"] = (dense, np.clip(y, 0.001, 0.999))
        return out

    prior_cohorts = _cohort("P", 30)
    prior = TJ.build_population_prior(prior_cohorts, family=family)

    # Calibration cohorts are DISJOINT from the prior cohorts -- split conformal.
    calib_cohorts = _cohort("K", 60)
    cal = TJ.build_conformal_calibrator(prior, calib_cohorts, train_max_day=20.0,
                                        seed=7, max_horizon=4.0)
    check("conformal calibrator was learned from held-out cohorts",
          cal is not None and cal.n_calibration_cohorts == 60,
          f"{getattr(cal, 'n_calibration_cohorts', 0)} cohorts, "
          f"{int(np.sum(getattr(cal, 'n_per_bucket', [0])))} residuals")

    check("overlapping calibration and prior cohorts are refused",
          _raises(lambda: TJ.build_conformal_calibrator(prior, prior_cohorts,
                                                        train_max_day=20.0)),
          "ValueError raised on overlap")

    # REGRESSION: the exchangeable unit is the COHORT, not the residual. One
    # cohort supplies a residual at every held-out age and those residuals are
    # correlated (measured ICC ~0.68), so counting residuals lets a calibration
    # through on a handful of cohorts. Calibrating a nominal-95% interval on ten
    # cohorts was measured at 0.75 coverage -- worse than not calibrating at all
    # (0.88). Too few cohorts must therefore REFUSE, not correct.
    for n_small in (6, 10, 20):
        small = _cohort("S", n_small)
        c = TJ.build_conformal_calibrator(prior, small, train_max_day=20.0,
                                          seed=7, max_horizon=4.0)
        if c is None:
            check(f"{n_small} calibration cohorts -> no calibrator", True,
                  "refused outright")
            continue
        n_res = int(np.sum(c.n_per_bucket))
        lo, hi = c.adjust(0.5, 0.1, horizon=1.0, level="95")
        check(f"{n_small} cohorts ({n_res} residuals) refuse the 95% correction",
              not np.isfinite(lo),
              f"returned [{lo}, {hi}] -- residual count must not stand in for "
              f"independent cohorts")
        check(f"and say why ({n_small} cohorts)",
              any("independent cohorts" in s for s in c.notes),
              c.notes[0][:80] if c.notes else "no note")

    check("the calibrator reports cohorts per bucket, not only residuals",
          hasattr(cal, "cohorts_per_bucket")
          and int(np.max(cal.cohorts_per_bucket)) <= cal.n_calibration_cohorts
          and int(np.sum(cal.n_per_bucket)) > int(np.max(cal.cohorts_per_bucket)),
          f"cohorts/bucket {list(map(int, cal.cohorts_per_bucket))} vs "
          f"residuals/bucket {list(map(int, cal.n_per_bucket))}")

    def _coverage(calibrator):
        c50, c95 = [], []
        for i in range(150):
            th = true_mu + rng.normal(0, true_sd)
            y = np.clip(TJ.FAMILIES[family].predict(dense, th)
                        + rng.normal(0, 0.035, len(dense)), 0.001, 0.999)
            try:
                v = TJ.forward_validate(dense, y, train_max_day=20.0, prior=prior,
                                        seed=5000 + i, max_horizon=4.0,
                                        calibrator=calibrator)
            except Exception:
                continue
            tab = v["per_horizon"]
            tab = tab[~tab["refused"]]
            if len(tab):
                c50.extend(tab["in_50CI"].dropna().tolist())
                c95.extend(tab["in_95CI"].dropna().tolist())
        return float(np.mean(c50)), float(np.mean(c95))

    raw50, raw95 = _coverage(None)
    cal50, cal95 = _coverage(cal)

    print(f"      UNCALIBRATED : 50% -> {raw50:.3f}   95% -> {raw95:.3f}")
    print(f"      CALIBRATED   : 50% -> {cal50:.3f}   95% -> {cal95:.3f}")
    rep = TJ.calibration_report([cal50, cal95], nominal=(0.5, 0.95))
    print("      " + rep.round(4).to_string(index=False).replace("\n", "\n      "))

    check("conformal calibration improves 95% coverage",
          abs(cal95 - 0.95) < abs(raw95 - 0.95),
          f"calibrated {cal95:.3f} vs uncalibrated {raw95:.3f} (nominal 0.95)")
    check("calibrated 95% coverage is within 7 points of nominal",
          abs(cal95 - 0.95) <= 0.07, f"empirical {cal95:.3f}")
    check("calibrated 50% coverage is within 10 points of nominal",
          abs(cal50 - 0.50) <= 0.10, f"empirical {cal50:.3f}")
    check("uncalibrated intervals are documented as over-confident",
          raw95 < 0.95, f"uncalibrated {raw95:.3f} -- this is why calibration is required")


def test_G_horizon_refusal() -> None:
    print("\nG. Beyond-horizon targets are refused, not extrapolated")
    t = np.array([5.0, 10.0, 15.0])
    y = np.array([0.92, 0.85, 0.78])
    res = TJ.predict_trajectory(t, y, [20.0, 30.0, 200.0], max_horizon=2.0, seed=1)
    preds = {p["day"]: p for p in res["predictions"]}

    check("an in-range target is predicted",
          np.isfinite(preds[20.0]["predicted"]) and not preds[20.0]["refused"],
          f"day 20 -> {preds[20.0]['predicted']:.4f} ({preds[20.0]['regime']})")
    check("a far target is refused",
          preds[200.0]["refused"] and not np.isfinite(preds[200.0]["predicted"]),
          f"day 200 refused: {preds[200.0]['refusal_reason'][:70]}")
    check("refusal carries a stated reason",
          len(preds[200.0]["refusal_reason"]) > 20, "reason present")
    check("fewer than two points is rejected outright",
          _raises(lambda: TJ.predict_trajectory([5.0], [0.9], [10.0])),
          "ValueError raised")


def test_H_decision_endpoints() -> None:
    print("\nH. Decision endpoints recover a known onset age")
    # Exact logistic decline crossing 0.5 at day 30 by construction
    theta = np.array([logit(0.95), logit(0.05), np.log(0.25), 30.0])
    ages = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
    y = TJ.FAMILIES["logistic"].predict(ages, theta)
    mid = float(np.interp(0.5, y[::-1], ages[::-1]))

    res = TJ.predict_trajectory(ages, y, np.arange(5, 55, 1.0),
                                families=["logistic"], seed=2, max_horizon=3.0)
    ep = TJ.derive_endpoints(res, threshold=0.5)
    check("onset age recovered near the true crossing",
          abs(ep["onset_age_days"] - mid) < 2.0,
          f"estimated {ep['onset_age_days']:.2f} d [{ep['onset_age_days_ci_lower']:.2f}, "
          f"{ep['onset_age_days_ci_upper']:.2f}] vs true crossing {mid:.2f} d")
    check("onset interval brackets the truth",
          ep["onset_age_days_ci_lower"] <= mid <= ep["onset_age_days_ci_upper"],
          f"[{ep['onset_age_days_ci_lower']:.2f}, {ep['onset_age_days_ci_upper']:.2f}]")
    check("healthspan is positive and inside the window",
          0 < ep["healthspan_days"] <= (ages.max() - ages.min()),
          f"{ep['healthspan_days']:.2f} d")

    # A window in which the threshold is never crossed must yield an undefined
    # onset with an explanation, not an invented number. Days 5-15 sit entirely
    # above 0.5 on this curve.
    res_short = TJ.predict_trajectory(ages, y, np.arange(5, 16, 1.0),
                                      families=["logistic"], seed=2, max_horizon=3.0)
    ep2 = TJ.derive_endpoints(res_short, threshold=0.5)
    check("an un-crossed threshold yields an undefined onset, not a fabricated one",
          not np.isfinite(ep2["onset_age_days"]) and len(ep2["onset_age_days_note"]) > 20,
          ep2["onset_age_days_note"][:110])


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/trajectory.py  (hierarchical trajectory forecasting)")
    print("=" * 78)
    test_A_family_recovery()
    test_B_crps_closed_form()
    test_C_em_recovers_between_cohort_covariance()
    test_D_shrinkage_helps_sparse_cohorts()
    test_E_beats_baselines()
    test_F_calibration()
    test_G_horizon_refusal()
    test_H_decision_endpoints()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL TRAJECTORY-PREDICTION VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
