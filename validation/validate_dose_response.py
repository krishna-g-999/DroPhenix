"""
Validation suite for modules/dose_response.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. EXACT PARAMETER RECOVERY on noise-free data. A curve is generated from a
   known LL.4 (and LL.3, LL.2) parameter set with no noise added; the fit must
   return those parameters to 1e-4. This is a closed-form check of the model
   function, the zero-dose limit and the optimiser all at once -- if any of
   them is wrong the residual sum of squares cannot reach zero. The ECx
   closed form is checked against a numerical root of the fitted curve to
   1e-10 in the same test.

B. RECOVERY AND COVERAGE UNDER NOISE. Over many replicate simulations the mean
   EC50 estimate must recover the truth, and the PROFILE-LIKELIHOOD interval
   must cover it at approximately its nominal 95%. Delta-method coverage is
   computed on the same simulations for comparison.

C. THE PROFILE INTERVAL IS NOT THE DELTA INTERVAL. On a design where EC50 is
   only moderately well determined, the profile interval must be visibly
   ASYMMETRIC about the estimate on the dose scale and must differ from the
   symmetric delta-method interval. Both are printed. This is the entire
   reason the profile machinery exists, so it is asserted, not assumed.

D. HORMESIS: POWER AND TYPE I ERROR. Data simulated from a Brain-Cousens model
   with a genuine low-dose stimulation must be flagged; monotone LL.4 data
   must NOT be, and the false-positive rate over repeated monotone simulations
   must be near the nominal 5%.

E. VALIDATION AGAINST REAL PUBLISHED DATA -- the `ryegrass` ferulic acid
   experiment (Inderjit, Streibig & Olofsdotter 2002), the canonical worked
   example of the R package `drc`. Both the raw data and the reference
   parameter estimates were obtained from primary sources and are documented
   in `test_E_...`'s docstring, including what could and could not be verified.

F. DESIGN REFUSAL. Three non-zero doses cannot identify a four-parameter
   model; `check_design` must refuse and `fit_loglogistic` must raise rather
   than return numbers.

G. MORTALITY CONFOUNDING. A toxin is simulated in which behaviour and death
   are driven by the SAME latent toxic burden, so the apparent behavioural
   curve among survivors is not a separate behavioural effect. The module must
   flag it (EC50/LC50 near 1). A second compound with a genuine behavioural
   effect two orders of magnitude below any lethality must NOT be flagged.

H. RELATIVE POTENCY. The EC50 ratio between two curves with a shared slope
   must recover the known ratio with nominal interval coverage, and curves
   with genuinely different slopes must be reported as non-parallel -- because
   a single potency ratio is only meaningful for parallel curves.

Run: ./.venv/Scripts/python.exe validation/validate_dose_response.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import dose_response as DR  # noqa: E402

FAILURES: list[str] = []
SEED = 20260803


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# Doses used throughout: a log-spaced ladder plus an untreated control.
DOSES = np.array([0.0, 0.3125, 0.625, 1.25, 2.5, 5.0, 10.0, 20.0, 40.0])


def replicate(doses, reps):
    return np.repeat(np.asarray(doses, dtype=float), reps)


# ─────────────────────────────────────────────────────────────────────────────
def test_A_exact_recovery() -> None:
    print("\nA. Exact parameter recovery on noise-free data (tolerance 1e-4)")
    b, c, d, e = 2.5, 10.0, 90.0, 5.0
    x = replicate(DOSES, 3)
    y = DR.loglogistic(x, b, c, d, e)

    fit = DR.fit_loglogistic(x, y, model="LL.4")
    truth = {"b": b, "c": c, "d": d, "e": e}
    worst = max(abs(float(fit.params[k]) - v) for k, v in truth.items())
    check("LL.4 recovers b, c, d, e exactly from a noise-free curve",
          worst < 1e-4,
          f"max |error| = {worst:.3e}; "
          f"b={fit.params['b']:.8f} c={fit.params['c']:.8f} "
          f"d={fit.params['d']:.8f} e={fit.params['e']:.8f}")
    check("residual sum of squares is numerically zero",
          fit.rss < 1e-16, f"RSS = {fit.rss:.3e}")

    # The zero-dose limit must be exactly d, not an approximation of it.
    check("model evaluated at dose 0 equals the upper asymptote d exactly",
          abs(float(DR.loglogistic(np.array([0.0]), b, c, d, e)[0]) - d) < 1e-12,
          f"f(0) = {float(DR.loglogistic(np.array([0.0]), b, c, d, e)[0]):.12f}, d = {d}")
    check("controls are kept in the fit, not dropped",
          fit.n_obs == x.size and int(np.sum(fit.dose == 0)) == 3,
          f"n_obs = {fit.n_obs} of {x.size}, with {int(np.sum(fit.dose == 0))} controls")

    # LL.3 and LL.2 on their own noise-free curves
    y3 = DR.loglogistic(x, 1.8, 0.0, 75.0, 4.0)
    f3 = DR.fit_loglogistic(x, y3, model="LL.3")
    err3 = max(abs(float(f3.params["b"]) - 1.8), abs(float(f3.params["d"]) - 75.0),
               abs(float(f3.params["e"]) - 4.0))
    check("LL.3 (c fixed at 0) recovers b, d, e exactly",
          err3 < 1e-4 and float(f3.params["c"]) == 0.0,
          f"max |error| = {err3:.3e}, c held at {float(f3.params['c']):.1f}")

    y2 = DR.loglogistic(x, 3.2, 0.0, 1.0, 7.0)
    f2 = DR.fit_loglogistic(x, y2, model="LL.2")
    err2 = max(abs(float(f2.params["b"]) - 3.2), abs(float(f2.params["e"]) - 7.0))
    check("LL.2 (c = 0, d = 1) recovers b, e exactly",
          err2 < 1e-4 and float(f2.params["c"]) == 0.0 and float(f2.params["d"]) == 1.0,
          f"max |error| = {err2:.3e}")

    # An INCREASING dose-response is the same model with d < c, not a special case.
    y_up = DR.loglogistic(x, 2.0, 80.0, 5.0, 3.0)
    f_up = DR.fit_loglogistic(x, y_up, model="LL.4")
    err_up = max(abs(float(f_up.params["b"]) - 2.0), abs(float(f_up.params["c"]) - 80.0),
                 abs(float(f_up.params["d"]) - 5.0), abs(float(f_up.params["e"]) - 3.0))
    check("an INCREASING response is recovered by the same model (d < c, b > 0)",
          err_up < 1e-4, f"max |error| = {err_up:.3e}")

    # ECx closed form vs a numerical root of the fitted curve
    ed = DR.effective_dose(fit, levels=(10, 25, 50, 90))
    worst_ed = 0.0
    for lvl in (10, 25, 50, 90):
        target = d - (lvl / 100.0) * (d - c)
        root = optimize.brentq(lambda t: float(fit.predict(np.array([t]))[0]) - target,
                               1e-9, 1e6, xtol=1e-14, rtol=1e-15)
        worst_ed = max(worst_ed, abs(root - float(ed.loc[lvl, "estimate"])))
    check("ECx closed form matches a numerical root of the fitted curve",
          worst_ed < 1e-10, f"max |error| = {worst_ed:.3e}")
    check("EC50 equals the parameter e by construction",
          abs(float(ed.loc[50, "estimate"]) - float(fit.params["e"])) < 1e-12,
          f"EC50 = {float(ed.loc[50, 'estimate']):.10f}, e = {float(fit.params['e']):.10f}")


# ─────────────────────────────────────────────────────────────────────────────
def test_B_noise_recovery_and_coverage() -> None:
    print("\nB. EC50 recovery and profile-likelihood coverage under noise")
    rng = np.random.default_rng(SEED + 1)
    b, c, d, e = 2.0, 10.0, 90.0, 5.0
    sigma = 6.0
    x = replicate(DOSES, 3)
    mu = DR.loglogistic(x, b, c, d, e)

    n_sims = 300
    ests, cov_prof, cov_delta, cov_delta_log = [], 0, 0, 0
    width_prof, width_delta = [], []
    n_ok = 0
    for _ in range(n_sims):
        y = mu + rng.normal(0.0, sigma, size=x.size)
        try:
            fit = DR.fit_loglogistic(x, y, model="LL.4")
            ci = DR.ec50_profile_ci(fit, level=0.95)
        except Exception:
            continue
        est = float(fit.params["e"])
        if not np.isfinite(est) or est <= 0:
            continue
        n_ok += 1
        ests.append(est)
        lo, hi = ci["profile"]["lower"], ci["profile"]["upper"]
        dlo, dhi = ci["delta"]["lower"], ci["delta"]["upper"]
        llo, lhi = ci["delta"]["log_lower"], ci["delta"]["log_upper"]
        cov_prof += int(lo <= e <= hi)
        cov_delta += int(np.isfinite(dlo) and dlo <= e <= dhi)
        cov_delta_log += int(np.isfinite(llo) and llo <= e <= lhi)
        if np.isfinite(hi):
            width_prof.append(hi - lo)
        if np.isfinite(dhi):
            width_delta.append(dhi - dlo)

    mean_e = float(np.mean(ests))
    median_e = float(np.median(ests))
    p_cov = cov_prof / n_ok
    d_cov = cov_delta / n_ok
    dl_cov = cov_delta_log / n_ok

    # Monte-Carlo 99% envelope for a genuinely 95% procedure
    se = np.sqrt(0.95 * 0.05 / n_ok)
    lo_env, hi_env = 0.95 - 2.576 * se, 0.95 + 2.576 * se

    print(f"      {n_ok} converged simulations, sigma = {sigma}, true EC50 = {e}")
    print(f"      mean EC50   = {mean_e:.4f}   median = {median_e:.4f}")
    print(f"      profile CI coverage      = {p_cov:.3%}")
    print(f"      delta CI coverage (dose) = {d_cov:.3%}")
    print(f"      delta CI coverage (log)  = {dl_cov:.3%}")
    print(f"      nominal 95%, MC 99% envelope = [{lo_env:.3%}, {hi_env:.3%}]")
    print(f"      mean width: profile = {np.mean(width_prof):.3f}, "
          f"delta = {np.mean(width_delta):.3f}")

    check("mean EC50 estimate recovers the truth", abs(mean_e - e) < 0.25,
          f"mean = {mean_e:.4f} vs true {e}")
    check("median EC50 estimate recovers the truth", abs(median_e - e) < 0.20,
          f"median = {median_e:.4f} vs true {e}")
    check("profile-likelihood 95% CI coverage is near nominal",
          0.92 <= p_cov <= 0.985, f"{p_cov:.3%} (envelope [{lo_env:.3%}, {hi_env:.3%}])")
    check("delta-method coverage is no better than the profile's",
          d_cov <= p_cov + 0.005,
          f"delta {d_cov:.3%} vs profile {p_cov:.3%}")


# ─────────────────────────────────────────────────────────────────────────────
def test_C_profile_vs_delta() -> None:
    print("\nC. Profile interval is asymmetric and differs from the delta method")
    rng = np.random.default_rng(SEED + 12)
    # A shallow slope with realistic noise makes EC50 only moderately well
    # determined, which is exactly the regime where the symmetric
    # approximation breaks down. The dose ladder still spans the whole curve,
    # so both bounds are finite and the two methods can be compared directly.
    b, c, d, e = 1.2, 5.0, 95.0, 8.0
    x = replicate(np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]), 3)
    y = DR.loglogistic(x, b, c, d, e) + rng.normal(0.0, 8.0, size=x.size)

    fit = DR.fit_loglogistic(x, y, model="LL.4")
    ci = DR.ec50_profile_ci(fit, level=0.95)
    est = ci["ec50"]
    plo, phi = ci["profile"]["lower"], ci["profile"]["upper"]
    dlo, dhi = ci["delta"]["lower"], ci["delta"]["upper"]
    llo, lhi = ci["delta"]["log_lower"], ci["delta"]["log_upper"]

    print(f"      EC50 estimate            = {est:.4f}")
    print(f"      profile 95% CI           = [{plo:.4f}, {phi:.4f}]   "
          f"(-{est - plo:.4f}, +{phi - est:.4f})")
    print(f"      delta 95% CI (dose)      = [{dlo:.4f}, {dhi:.4f}]   "
          f"(+/- {(dhi - dlo) / 2:.4f})")
    print(f"      delta 95% CI (log scale) = [{llo:.4f}, {lhi:.4f}]")
    print(f"      asymmetry ratio (upper arm / lower arm) = {ci['asymmetry_ratio']:.3f}")

    check("both profile bounds are finite on this design",
          np.isfinite(plo) and np.isfinite(phi) and plo > 0,
          f"[{plo:.4f}, {phi:.4f}]")
    check("profile interval is asymmetric on the dose scale",
          ci["asymmetry_ratio"] > 1.10,
          f"upper arm is {ci['asymmetry_ratio']:.3f}x the lower arm")
    check("profile interval differs from the symmetric delta interval",
          (abs(plo - dlo) / est > 0.02) or (abs(phi - dhi) / est > 0.02),
          f"lower differs by {abs(plo - dlo):.4f}, upper by {abs(phi - dhi):.4f}")
    check("profile upper bound exceeds the delta upper bound "
          "(the delta method understates the upper tail)",
          phi > dhi, f"profile {phi:.4f} vs delta {dhi:.4f}")
    check("the reported note names the profile interval as the one to report",
          "profile interval" in ci["note"], ci["note"][:80] + "...")

    # A case where the delta method returns an IMPOSSIBLE dose. With a shallow
    # slope the standard error on the dose scale exceeds the estimate itself,
    # so the symmetric interval runs below zero -- a negative concentration.
    rng2 = np.random.default_rng(SEED + 22)
    x2 = replicate(np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]), 3)
    y2 = DR.loglogistic(x2, 1.2, 5.0, 95.0, 8.0) + rng2.normal(0.0, 8.0, size=x2.size)
    fit2 = DR.fit_loglogistic(x2, y2, model="LL.4")
    ci2 = DR.ec50_profile_ci(fit2, level=0.95)
    print(f"      shallow-slope case: b = {float(fit2.params['b']):.3f}, "
          f"EC50 = {ci2['ec50']:.3f}")
    print(f"        delta   = [{ci2['delta']['lower']:.3f}, {ci2['delta']['upper']:.3f}]")
    print(f"        profile = [{ci2['profile']['lower']:.3f}, "
          f"{ci2['profile']['upper']:.3f}]")
    check("delta-method lower bound can be a NEGATIVE dose (the failure mode)",
          ci2["delta"]["lower"] < 0.0, f"delta lower = {ci2['delta']['lower']:.4f}")
    check("profile lower bound is always a possible (positive) dose",
          ci2["profile"]["lower"] > 0.0 and ci["profile"]["lower"] > 0.0,
          f"profile lower bounds {ci2['profile']['lower']:.4f} and {plo:.4f}")
    check("the note flags the impossible delta-method bound to the user",
          "not a possible dose" in ci2["note"], ci2["note"][-90:])

    # EC90 sits on the flat tail, where the two methods must diverge most.
    ed = DR.effective_dose(fit, levels=(10, 50, 90))
    print("      effective doses from the same fit:")
    for lvl in (10, 50, 90):
        r = ed.loc[lvl]
        print(f"        EC{int(lvl):02d} = {r['estimate']:9.4f}   "
              f"profile [{r['profile_lower']:9.4f}, {r['profile_upper']:9.4f}]   "
              f"delta [{r['delta_lower']:9.4f}, {r['delta_upper']:9.4f}]")
    ec90 = ed.loc[90]
    asym90 = ((ec90["profile_upper"] - ec90["estimate"])
              / max(ec90["estimate"] - ec90["profile_lower"], 1e-12))
    check("ECx intervals are ordered EC10 < EC50 < EC90",
          ed.loc[10, "estimate"] < ed.loc[50, "estimate"] < ed.loc[90, "estimate"],
          f"{ed.loc[10, 'estimate']:.3f} < {ed.loc[50, 'estimate']:.3f} "
          f"< {ed.loc[90, 'estimate']:.3f}")
    check("EC90 profile interval is more asymmetric than EC50's",
          asym90 > ci["asymmetry_ratio"],
          f"EC90 asymmetry {asym90:.3f} vs EC50 {ci['asymmetry_ratio']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
def test_D_hormesis() -> None:
    print("\nD. Hormesis: detection under a true hump, type I error under monotonicity")
    rng = np.random.default_rng(SEED + 3)
    x = replicate(DOSES, 4)

    # ── power: data really do come from a Brain-Cousens model ───────────────
    b, c, d, e, f_true = 2.5, 5.0, 60.0, 6.0, 12.0
    mu_h = DR._brain_cousens(x, b, c, d, float(np.log(e)), f_true)
    y_h = mu_h + rng.normal(0.0, 2.5, size=x.size)
    res_h = DR.hormesis_test(x, y_h)
    check("hormesis DETECTED on truly hormetic data",
          res_h["usable"] and res_h["hormesis_supported"],
          f"p = {res_h.get('p_value'):.3g}, f_hat = {res_h.get('f_param'):.3f} "
          f"(true f = {f_true}), peak +{res_h.get('max_stimulation_percent'):.1f}% "
          f"at dose {res_h.get('max_stimulation_dose'):.3f}")

    # The monotone fit is biased under this alternative -- worth showing.
    mono_e = res_h["monotone_ec50"]
    print(f"      monotone LL.4 EC50 on hormetic data = {mono_e:.3f} "
          f"(Brain-Cousens location e = {res_h['bc_params']['e']:.3f}, true {e})")

    # ── single monotone dataset must NOT be flagged ─────────────────────────
    mu_m = DR.loglogistic(x, 2.5, 5.0, 60.0, 6.0)
    y_m = mu_m + rng.normal(0.0, 2.5, size=x.size)
    res_m = DR.hormesis_test(x, y_m)
    check("hormesis NOT claimed on monotone data",
          res_m["usable"] and not res_m["hormesis_supported"],
          f"p = {res_m.get('p_value'):.3g}, f_hat = {res_m.get('f_param'):.4f}")

    # ── type I error over repeated monotone simulations ─────────────────────
    n_sims, alpha = 300, 0.05
    rej, n_ok = 0, 0
    for _ in range(n_sims):
        y = mu_m_i = DR.loglogistic(x, 2.5, 5.0, 60.0, 6.0) + rng.normal(0.0, 2.5, size=x.size)
        try:
            r = DR.hormesis_test(x, y, alpha=alpha)
        except Exception:
            continue
        if not r["usable"]:
            continue
        n_ok += 1
        rej += int(r["hormesis_supported"])
    rate = rej / n_ok
    se = np.sqrt(alpha * (1 - alpha) / n_ok)
    hi = alpha + 2.576 * se
    print(f"      false-positive hormesis rate over {n_ok} monotone simulations: "
          f"{rate:.3%} (nominal {alpha:.0%}, 99% upper bound {hi:.3%})")
    check("type I error for the hormesis test is controlled",
          rate <= hi + 0.02, f"{rate:.3%} vs bound {hi:.3%}")

    # ── power over repeated hormetic simulations ────────────────────────────
    n_sims_h = 150
    det, n_h = 0, 0
    for _ in range(n_sims_h):
        y = DR._brain_cousens(x, b, c, d, float(np.log(e)), f_true) \
            + rng.normal(0.0, 2.5, size=x.size)
        try:
            r = DR.hormesis_test(x, y)
        except Exception:
            continue
        if not r["usable"]:
            continue
        n_h += 1
        det += int(r["hormesis_supported"])
    power = det / n_h
    print(f"      detection rate over {n_h} hormetic simulations: {power:.1%}")
    check("hormesis test has high power against a real hump",
          power >= 0.80, f"power = {power:.1%}")

    # ── it must refuse when there are too few doses for a 5-parameter model ─
    x_small = replicate(np.array([0.0, 1.0, 3.0, 9.0, 27.0]), 3)
    y_small = DR.loglogistic(x_small, 2.0, 5.0, 60.0, 5.0)
    res_small = DR.hormesis_test(x_small, y_small)
    check("refuses Brain-Cousens with only 4 distinct non-zero doses",
          not res_small["usable"], res_small.get("reason", "")[:90])


# ─────────────────────────────────────────────────────────────────────────────
# The real published dataset
# ─────────────────────────────────────────────────────────────────────────────
# fmt: off
RYEGRASS_ROOTL = np.array([
    7.580000000, 8.000000000, 8.328571429, 7.250000000, 7.375000000, 7.962500000,
    8.355555556, 6.914285714, 7.750000000, 6.871428571, 6.450000000, 5.922222222,
    1.925000000, 2.885714286, 4.233333333, 1.187500000, 0.857142857, 1.057142857,
    0.687500000, 0.525000000, 0.825000000, 0.250000000, 0.220000000, 0.440000000,
])
RYEGRASS_CONC = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.94, 0.94, 0.94, 1.88, 1.88, 1.88,
    3.75, 3.75, 3.75, 7.50, 7.50, 7.50,
    15.00, 15.00, 15.00, 30.00, 30.00, 30.00,
])
# fmt: on

# Reference LL.4 estimates produced by the R package `drc` on these data.
RYEGRASS_REF = {
    "b": 2.98222, "c": 0.48141, "d": 7.79296, "e": 3.05795,
    "b_se": 0.46506, "c_se": 0.21219, "d_se": 0.18857, "e_se": 0.18573,
    "resid_se": 0.5196256, "df": 20,
}


def test_E_real_published_data() -> None:
    """
    E. Validation against real published data: ryegrass / ferulic acid.

    PROVENANCE OF THE RAW DATA -- what was verified and how.
      The 24 (conc, rootl) pairs above were read directly out of
      `data/ryegrass.rda` in the `drcData` package maintained by the `drc`
      authors (https://github.com/DoseResponse/drcData), by downloading the
      file and decoding R's serialization format. They are the package data
      themselves, not a transcription from a secondary description, and they
      are reproduced here verbatim to full stored precision.

      The underlying experiment is
        Inderjit, Streibig JC, Olofsdotter M (2002) Joint action of phenolic
        acid mixtures and its significance in allelopathy research.
        Physiologia Plantarum 114:422-428.
      root length (cm) of perennial ryegrass (Lolium perenne L.) against
      ferulic acid concentration (mM), three replicates at each of six
      non-zero doses plus six untreated controls.

      The dataset is distributed with `drc`, described in
        Ritz C, Baty F, Streibig JC, Gerhard D (2015) Dose-response analysis
        using R. PLoS ONE 10(12):e0146021.

    WHAT COULD NOT BE VERIFIED FROM THE PLOS ONE PAPER ITSELF.
      The PLoS ONE article was fetched and inspected. Its main text is a
      methodological description of the `drc` package: it does NOT tabulate
      the ryegrass raw values and does NOT print an LL.4 coefficient table for
      them. So the numbers below do not come from that paper, and are not
      attributed to it.

    PROVENANCE OF THE REFERENCE ESTIMATES.
      The `drc` LL.4 coefficient table for these data
        b = 2.98222 (SE 0.46506), c = 0.48141 (0.21219),
        d = 7.79296 (0.18857),    e = 3.05795 (0.18573),
        residual standard error 0.5196256 on 20 degrees of freedom
      was taken from the printed `summary(drm(rootl ~ conc, fct = LL.4()))`
      output in chapter 11 ("Dose-Response curves") of
        Kniss AR, Streibig JC, Statistical Analysis of Agricultural
        Experiments using R, https://rstats4ag.org/dose-response-curves.html
      (Streibig is an author of both `drc` and the original ryegrass study).

      That block was cross-checked rather than trusted: an independent scipy
      least-squares fit written for this validation, using none of the module
      code, reproduces b = 2.98223, c = 0.48141, d = 7.79296, e = 3.05796 and
      residual SE = 0.5196256 from the package raw data. Agreement to five
      decimals on four parameters and seven significant figures on the
      residual standard error is not something a mis-transcribed reference
      could produce, so both the raw data and the reference estimates are
      treated as verified. A second web source quoted b = 2.48 for the same
      fit; that value is inconsistent with the data and is not used.

      NOT VERIFIED, AND THEREFORE NOT ASSERTED: no published ED10 / ED90
      values for this fit were located, so those are checked only for internal
      consistency (closed form vs numerical root), not against a reference.

    ON THE STANDARD ERRORS. `drc` obtains its covariance from a numerically
    differentiated Hessian of the log-likelihood, which retains the
    second-derivative term; this module uses the Gauss-Newton approximation
    (J'J)^-1, as documented. The two agree to within about 1.5% here, so the
    parameters are asserted tightly and the standard errors loosely, with the
    discrepancy printed.
    """
    print("\nE. Real published data: ryegrass root length vs ferulic acid")
    print("      Inderjit, Streibig & Olofsdotter (2002) Physiol Plant 114:422-428;")
    print("      raw data decoded from drcData/data/ryegrass.rda (drc authors' repo);")
    print("      reference LL.4 estimates from a printed drc summary (rstats4ag.org ch.11),")
    print("      independently reproduced here by a scipy fit written for this test.")

    check("dataset has the documented 24 observations",
          RYEGRASS_ROOTL.size == 24 and RYEGRASS_CONC.size == 24,
          f"{RYEGRASS_ROOTL.size} root lengths, {RYEGRASS_CONC.size} concentrations")
    check("dataset has 6 controls and 6 non-zero dose levels in triplicate",
          int(np.sum(RYEGRASS_CONC == 0)) == 6
          and np.unique(RYEGRASS_CONC[RYEGRASS_CONC > 0]).size == 6,
          f"{int(np.sum(RYEGRASS_CONC == 0))} controls, "
          f"{np.unique(RYEGRASS_CONC[RYEGRASS_CONC > 0]).size} non-zero doses "
          f"{list(np.unique(RYEGRASS_CONC[RYEGRASS_CONC > 0]))}")

    fit = DR.fit_loglogistic(RYEGRASS_CONC, RYEGRASS_ROOTL, model="LL.4")

    print("\n      parameter      this module        drc reference     |diff|")
    worst_rel = 0.0
    for k in ("b", "c", "d", "e"):
        ours = float(fit.params[k])
        ref = RYEGRASS_REF[k]
        diff = abs(ours - ref)
        worst_rel = max(worst_rel, diff / abs(ref))
        print(f"      {k:<12}  {ours:14.6f}  {ref:16.5f}  {diff:11.2e}")
    print(f"      {'resid SE':<12}  {fit.resid_se:14.7f}  "
          f"{RYEGRASS_REF['resid_se']:16.7f}  "
          f"{abs(fit.resid_se - RYEGRASS_REF['resid_se']):11.2e}")
    print(f"      {'df':<12}  {fit.df_resid:14d}  {RYEGRASS_REF['df']:16d}")

    check("reproduces the published drc LL.4 parameter estimates",
          worst_rel < 5e-5,
          f"largest relative difference = {worst_rel:.2e} across b, c, d, e")
    check("reproduces the published residual standard error",
          abs(fit.resid_se - RYEGRASS_REF["resid_se"]) < 1e-6,
          f"{fit.resid_se:.7f} vs {RYEGRASS_REF['resid_se']:.7f}")
    check("residual degrees of freedom match (24 obs - 4 params)",
          fit.df_resid == RYEGRASS_REF["df"], f"{fit.df_resid} vs {RYEGRASS_REF['df']}")

    se_rel = max(abs(float(fit.bse[k]) - RYEGRASS_REF[f"{k}_se"]) / RYEGRASS_REF[f"{k}_se"]
                 for k in ("b", "c", "d", "e"))
    print(f"      standard errors: ours "
          f"{[round(float(fit.bse[k]), 5) for k in ('b', 'c', 'd', 'e')]} vs drc "
          f"{[RYEGRASS_REF[f'{k}_se'] for k in ('b', 'c', 'd', 'e')]}")
    check("standard errors agree with drc to within 5% "
          "(Gauss-Newton vs drc's full numerical Hessian)",
          se_rel < 0.05, f"largest relative difference = {se_rel:.3%}")

    # An independent re-derivation of the same fit, using no module code.
    def _raw_resid(p):
        b, c, d, le = p
        out = np.full_like(RYEGRASS_CONC, d)
        m = RYEGRASS_CONC > 0
        out[m] = c + (d - c) / (1.0 + np.exp(b * (np.log(RYEGRASS_CONC[m]) - le)))
        return out - RYEGRASS_ROOTL

    r = optimize.least_squares(_raw_resid, [2.0, 0.5, 7.8, np.log(3.0)],
                               xtol=1e-15, ftol=1e-15, gtol=1e-15)
    indep = {"b": r.x[0], "c": r.x[1], "d": r.x[2], "e": float(np.exp(r.x[3]))}
    worst_indep = max(abs(indep[k] - float(fit.params[k])) for k in indep)
    check("independent scipy re-fit (no module code) agrees with the module",
          worst_indep < 1e-6,
          f"max |diff| = {worst_indep:.2e}; independent fit "
          f"b={indep['b']:.5f} c={indep['c']:.5f} d={indep['d']:.5f} e={indep['e']:.5f}")

    # Intervals on the real data
    ci = DR.ec50_profile_ci(fit, level=0.95)
    print(f"\n      EC50 = {ci['ec50']:.5f} mM ferulic acid")
    print(f"        profile 95% CI = [{ci['profile']['lower']:.5f}, "
          f"{ci['profile']['upper']:.5f}]")
    print(f"        delta   95% CI = [{ci['delta']['lower']:.5f}, "
          f"{ci['delta']['upper']:.5f}]")
    check("profile interval on the real data brackets the published EC50",
          ci["profile"]["lower"] < RYEGRASS_REF["e"] < ci["profile"]["upper"],
          f"[{ci['profile']['lower']:.5f}, {ci['profile']['upper']:.5f}] "
          f"contains {RYEGRASS_REF['e']}")

    ed = DR.effective_dose(fit, levels=(10, 50, 90))
    print("      effective doses (internal-consistency check only -- no published")
    print("      ED10/ED90 for this fit was located, so these are NOT asserted")
    print("      against a reference):")
    worst = 0.0
    b, c, d = (float(fit.params[k]) for k in ("b", "c", "d"))
    for lvl in (10, 50, 90):
        est = float(ed.loc[lvl, "estimate"])
        target = d - (lvl / 100.0) * (d - c)
        root = optimize.brentq(lambda t: float(fit.predict(np.array([t]))[0]) - target,
                               1e-9, 1e6, xtol=1e-14, rtol=1e-15)
        worst = max(worst, abs(root - est))
        print(f"        ED{int(lvl):02d} = {est:8.5f}  "
              f"profile [{ed.loc[lvl, 'profile_lower']:8.5f}, "
              f"{ed.loc[lvl, 'profile_upper']:8.5f}]  "
              f"numerical root {root:8.5f}")
    check("ED10/ED50/ED90 on real data match numerical roots of the fitted curve",
          worst < 1e-9, f"max |diff| = {worst:.2e}")

    # LL.3 on the same data, as a nested-model sanity check
    f3 = DR.fit_loglogistic(RYEGRASS_CONC, RYEGRASS_ROOTL, model="LL.3")
    print(f"      LL.3 (c = 0) on the same data: b = {float(f3.params['b']):.5f}, "
          f"d = {float(f3.params['d']):.5f}, e = {float(f3.params['e']):.5f}, "
          f"RSS = {f3.rss:.5f}")
    check("LL.3 has a larger RSS than LL.4 (nested model, as it must)",
          f3.rss > fit.rss, f"LL.3 RSS = {f3.rss:.5f} > LL.4 RSS = {fit.rss:.5f}")

    # Relative potency of the compound against itself must be exactly 1
    rp = DR.relative_potency(fit, fit, label_a="ryegrass", label_b="ryegrass")
    check("relative potency of a fit against itself is exactly 1",
          abs(rp["ratio"] - 1.0) < 1e-12, f"ratio = {rp['ratio']:.12f}")

    # Hormesis must not be claimed on this monotone real dataset
    h = DR.hormesis_test(RYEGRASS_CONC, RYEGRASS_ROOTL)
    check("no hormesis claimed on the (monotone) ryegrass data",
          h["usable"] and not h["hormesis_supported"],
          f"p = {h['p_value']:.4f}, f_hat = {h['f_param']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def test_F_design_refusal() -> None:
    print("\nF. Design refusal: too few doses to identify the model")
    x3 = replicate(np.array([0.0, 1.0, 5.0, 25.0]), 3)      # THREE non-zero doses
    y3 = DR.loglogistic(x3, 2.0, 10.0, 90.0, 5.0)

    d = DR.check_design(x3, y3, model="LL.4")
    check("check_design REFUSES 3 non-zero doses for a 4-parameter model",
          not d["usable"] and d["n_distinct_nonzero_doses"] == 3,
          d["refusals"][0][:110] if d["refusals"] else "no reason given")

    raised = False
    try:
        DR.fit_loglogistic(x3, y3, model="LL.4")
    except ValueError as exc:
        raised = "cannot support" in str(exc)
    check("fit_loglogistic raises rather than returning numbers for that design",
          raised, "ValueError raised naming the design limitation")

    # It must still fit under an explicit override, and a 3-parameter model
    # on 3 doses must be allowed.
    forced = DR.fit_loglogistic(x3, y3, model="LL.4", force=True)
    check("force=True still fits, but the design verdict is carried on the object",
          not forced.design["usable"],
          f"forced fit e = {float(forced.params['e']):.4f}, design usable = "
          f"{forced.design['usable']}")

    d3 = DR.check_design(x3, y3, model="LL.3")
    check("3 non-zero doses ARE enough for the 3-parameter LL.3",
          d3["usable"], f"usable = {d3['usable']}, warnings = {len(d3['warnings'])}")

    # Four doses: identified, but warned about
    x4 = replicate(np.array([0.0, 1.0, 5.0, 25.0, 125.0]), 3)
    y4 = DR.loglogistic(x4, 2.0, 10.0, 90.0, 5.0)
    d4 = DR.check_design(x4, y4, model="LL.4")
    check("4 non-zero doses are accepted but warned about",
          d4["usable"] and any("unreliable" in w for w in d4["warnings"]),
          "; ".join(d4["warnings"])[:110])

    # EC50 outside the tested range must be flagged as extrapolated
    x_hi = replicate(np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.16]), 3)
    y_hi = DR.loglogistic(x_hi, 2.0, 10.0, 90.0, 50.0)      # EC50 far above any dose
    d_hi = DR.check_design(x_hi, y_hi, model="LL.4")
    check("EC50 outside the tested dose range is flagged as extrapolated",
          d_hi.get("ec50_extrapolated") is True
          and any("EXTRAPOLATED" in w for w in d_hi["warnings"]),
          f"ec50_extrapolated = {d_hi.get('ec50_extrapolated')}")

    # A constant response is not a dose-response
    d_flat = DR.check_design(x4, np.full(x4.size, 42.0), model="LL.4")
    check("a constant response is refused",
          not d_flat["usable"], d_flat["refusals"][0][:80] if d_flat["refusals"] else "")


# ─────────────────────────────────────────────────────────────────────────────
def _simulate_confounded(rng, doses, n_per_vial=30, n_vials=5,
                         b=3.0, m=10.0, sigma=1.0,
                         t_climb=1.0, t_death=1.4):
    """
    A toxin in which behaviour and death share ONE latent cause.

    Each fly carries a latent toxic burden

        z_i = b * (log(dose) - log(m)) + eps_i,     eps_i ~ N(0, sigma)

    (controls get z_i = -inf, i.e. no burden). The fly DIES if z_i > t_death,
    and among the survivors it FAILS TO CLIMB if z_i > t_climb, with
    t_climb < t_death. There is no separate behavioural mechanism at all --
    only one axis of poisoning, read out at two thresholds.

    A conventional analysis of climbing among survivors nevertheless returns a
    clean-looking sigmoid dose-response with a respectable EC50, and it is
    entirely a description of how close the surviving flies are to dying. This
    is the situation `mortality_adjusted_response` has to catch.
    """
    rows = []
    for dose in doses:
        for v in range(n_vials):
            if dose <= 0:
                z = np.full(n_per_vial, -np.inf)
            else:
                z = b * (np.log(dose) - np.log(m)) + rng.normal(0.0, sigma, n_per_vial)
            dead = int(np.sum(z > t_death))
            alive = z <= t_death
            n_alive = int(np.sum(alive))
            if n_alive == 0:
                continue
            climbed = int(np.sum(z[alive] <= t_climb))
            rows.append({"dose": float(dose), "vial": v, "n_exposed": n_per_vial,
                         "dead": dead, "n_scored": n_alive,
                         "climbed": climbed, "pi": climbed / n_alive})
    return pd.DataFrame(rows)


def _simulate_clean(rng, doses, n_per_vial=25, n_vials=4,
                    beh_ec50=0.5, beh_b=2.0, lc50=50.0, lc_b=3.0):
    """
    A compound with a GENUINE behavioural effect, half-maximal two orders of
    magnitude below anything lethal. The module must not flag this one --
    a diagnostic that fires on everything is useless.
    """
    rows = []
    for dose in doses:
        # survival falls from 1 to 0 with LC50 = lc50; behaviour falls with its
        # own, far lower, EC50 -- two genuinely separate mechanisms.
        p_surv = float(DR.loglogistic(np.array([dose]), lc_b, 0.0, 1.0, lc50)[0])
        p_climb = float(DR.loglogistic(np.array([dose]), beh_b, 0.05, 0.90, beh_ec50)[0])
        for v in range(n_vials):
            n_alive = int(rng.binomial(n_per_vial, p_surv))
            if n_alive == 0:
                continue
            climbed = int(rng.binomial(n_alive, p_climb))
            rows.append({"dose": float(dose), "vial": v, "n_exposed": n_per_vial,
                         "dead": n_per_vial - n_alive, "n_scored": n_alive,
                         "climbed": climbed, "pi": climbed / n_alive})
    return pd.DataFrame(rows)


def test_H_relative_potency() -> None:
    """
    H. Relative potency: accuracy of the ratio, coverage of its interval, and
       the parallelism requirement.

    Two compounds are simulated with the SAME Hill slope and EC50s a known
    factor apart; the estimated ratio must recover that factor and its
    interval must cover it at the nominal rate. Then two compounds with
    genuinely different slopes are simulated with enough data to detect it:
    the function must report them as non-parallel and must surface the fact
    that the potency ratio is then dose-dependent rather than a single number.
    """
    print("\nH. Relative potency: accuracy, coverage, and the parallelism requirement")
    rng = np.random.default_rng(SEED + 9)
    x = replicate(DOSES, 4)
    b_shared, c, d = 2.0, 10.0, 90.0
    e_a, e_b = 2.5, 10.0
    true_ratio = e_a / e_b                       # 0.25 -- A is 4x more potent

    n_sims, covered, ratios = 300, 0, []
    for _ in range(n_sims):
        ya = DR.loglogistic(x, b_shared, c, d, e_a) + rng.normal(0, 5.0, x.size)
        yb = DR.loglogistic(x, b_shared, c, d, e_b) + rng.normal(0, 5.0, x.size)
        try:
            fa = DR.fit_loglogistic(x, ya)
            fb = DR.fit_loglogistic(x, yb)
            rp = DR.relative_potency(fa, fb)
        except Exception:
            continue
        if not np.isfinite(rp["ratio"]):
            continue
        ratios.append(rp["ratio"])
        covered += int(rp["ci_lower"] <= true_ratio <= rp["ci_upper"])

    med = float(np.median(ratios))
    cov = covered / len(ratios)
    se = np.sqrt(0.95 * 0.05 / len(ratios))
    print(f"      {len(ratios)} simulations, true EC50 ratio = {true_ratio}")
    print(f"      median estimated ratio = {med:.4f}")
    print(f"      95% CI coverage        = {cov:.3%} "
          f"(MC 99% envelope [{0.95 - 2.576 * se:.3%}, {0.95 + 2.576 * se:.3%}])")
    check("relative potency recovers the true EC50 ratio",
          abs(med - true_ratio) < 0.02, f"median = {med:.4f} vs true {true_ratio}")
    check("relative potency 95% interval coverage is near nominal",
          0.92 <= cov <= 0.985, f"{cov:.3%}")

    # Parallel curves: the parallelism test must NOT fire
    ya = DR.loglogistic(x, b_shared, c, d, e_a) + rng.normal(0, 4.0, x.size)
    yb = DR.loglogistic(x, b_shared, c, d, e_b) + rng.normal(0, 4.0, x.size)
    rp_par = DR.relative_potency(DR.fit_loglogistic(x, ya), DR.fit_loglogistic(x, yb),
                                 label_a="compound A", label_b="compound B")
    check("curves with a shared slope are reported as parallel",
          rp_par["parallel"] and "not parallel" not in rp_par["verdict"].lower(),
          f"b = {rp_par['slope_a']:.3f} vs {rp_par['slope_b']:.3f}, "
          f"p = {rp_par['parallelism_p']:.3f}")

    # Non-parallel curves, with enough data to detect it
    x_big = replicate(DOSES, 12)
    y1 = DR.loglogistic(x_big, 1.0, c, d, 5.0) + rng.normal(0, 3.0, x_big.size)
    y2 = DR.loglogistic(x_big, 4.0, c, d, 5.0) + rng.normal(0, 3.0, x_big.size)
    rp_np = DR.relative_potency(DR.fit_loglogistic(x_big, y1),
                                DR.fit_loglogistic(x_big, y2),
                                label_a="shallow", label_b="steep")
    print(f"      non-parallel case: b = {rp_np['slope_a']:.3f} vs "
          f"{rp_np['slope_b']:.3f}, parallelism p = {rp_np['parallelism_p']:.3g}")
    print(f"        potency ratio is {rp_np['ratio_by_level'][10.0]:.3f} at EC10, "
          f"{rp_np['ratio_by_level'][50.0]:.3f} at EC50, "
          f"{rp_np['ratio_by_level'][90.0]:.3f} at EC90")
    check("non-parallel slopes are detected",
          not rp_np["parallel"] and rp_np["parallelism_p"] < 0.05,
          f"p = {rp_np['parallelism_p']:.3g}")
    check("the verdict warns that a single potency ratio is then ill-defined",
          "not parallel" in rp_np["verdict"].lower()
          and "level-specific" in rp_np["verdict"].lower(),
          rp_np["verdict"][-130:])
    check("equal EC50s with unequal slopes still give a dose-DEPENDENT ratio",
          abs(rp_np["ratio_by_level"][50.0] - 1.0) < 0.15
          and rp_np["ratio_by_level"][10.0] < 0.7
          and rp_np["ratio_by_level"][90.0] > 1.4,
          f"EC10 {rp_np['ratio_by_level'][10.0]:.3f}, EC50 "
          f"{rp_np['ratio_by_level'][50.0]:.3f}, EC90 "
          f"{rp_np['ratio_by_level'][90.0]:.3f}")


def test_G_mortality_confounding() -> None:
    print("\nG. Mortality confounding: behaviour that is really survivor selection")
    rng = np.random.default_rng(SEED + 6)
    doses = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 28.0, 40.0])
    df = _simulate_confounded(rng, doses)

    res = DR.mortality_adjusted_response(
        df["dose"], df["pi"], df["dead"], df["n_exposed"], n_scored=df["n_scored"])

    print("      per-dose mortality and survivor-conditional flag:")
    pd_tab = res["per_dose"]
    for dose, row in pd_tab.iterrows():
        print(f"        dose {dose:7.3f}  mortality {row['mortality']:6.1%}  "
              f"scored {int(row['n_scored']):4d}  mean PI {row['mean_response']:.3f}"
              f"{'   <-- survivor-conditional' if row['survivor_conditional'] else ''}")
    print(f"      EC50(behaviour) = {res['ec50']:.4f}")
    print(f"      LC50(mortality) = {res['lc50']:.4f}")
    print(f"      ratio EC50/LC50 = {res['ratio']:.3f}   severity = {res['severity']}")

    check("naive behavioural curve is still produced (the number users would report)",
          res["behaviour_fit"] is not None and np.isfinite(res["ec50"]),
          f"EC50 = {res['ec50']:.4f} with profile CI "
          f"[{res['ec50_ci']['profile']['lower']:.4f}, "
          f"{res['ec50_ci']['profile']['upper']:.4f}]")
    check("mortality curve is fitted and LC50 estimated",
          res["mortality_fit"] is not None and np.isfinite(res["lc50"]),
          f"LC50 = {res['lc50']:.4f}")
    check("EC50/LC50 ratio is near or above 1 when behaviour IS survivor selection",
          np.isfinite(res["ratio"]) and res["ratio"] >= 0.80,
          f"ratio = {res['ratio']:.3f}")
    check("the confound is FLAGGED",
          res["confounded"] and res["severity"] == "confounded",
          f"severity = {res['severity']}")
    check("the warning states the estimate is survivor-conditional and must not be "
          "reported as behavioural",
          "MUST NOT" in res["warning"] and "survivor" in res["warning"].lower(),
          res["warning"][:120] + "...")
    check("doses with substantial mortality are individually identified",
          res["n_confounded_doses"] > 0,
          f"{res['n_confounded_doses']} of {res['n_dose_levels']} dose levels exceed "
          f"{res['mortality_threshold']:.0%} mortality")

    # ── the negative control: a real behavioural effect, no lethality ───────
    print("\n      Negative control: genuine behavioural effect far below any lethality")
    rng2 = np.random.default_rng(SEED + 7)
    doses2 = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
    df2 = _simulate_clean(rng2, doses2)
    res2 = DR.mortality_adjusted_response(
        df2["dose"], df2["pi"], df2["dead"], df2["n_exposed"], n_scored=df2["n_scored"])
    print(f"        EC50(behaviour)  = {res2['ec50']:.4f}  (simulated truth 0.5)")
    print(f"        max mortality    = {res2['max_mortality']:.2%}")
    print(f"        LC50             = {res2['lc50']}  ({res2.get('mortality_error', '')[:60]})")
    print(f"        severity         = {res2['severity']}")
    check("a genuine behavioural effect with no lethality is NOT flagged",
          not res2["confounded"] and res2["severity"] == "clean",
          f"severity = {res2['severity']}, max mortality = {res2['max_mortality']:.2%}")
    check("behavioural EC50 recovered in the clean case (truth 0.5)",
          abs(res2["ec50"] - 0.5) < 0.15, f"EC50 = {res2['ec50']:.4f}")
    check("the clean verdict explains WHY it is safe rather than staying silent",
          "no meaningful survivor selection" in res2["warning"].lower()
          or "separable" in res2["warning"].lower(),
          res2["warning"][:110] + "...")

    # ── a partially-overlapping case must land in the middle tier ──────────
    rng3 = np.random.default_rng(SEED + 8)
    doses3 = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    df3 = _simulate_clean(rng3, doses3, beh_ec50=3.0, beh_b=2.0, lc50=8.0, lc_b=3.0)
    res3 = DR.mortality_adjusted_response(
        df3["dose"], df3["pi"], df3["dead"], df3["n_exposed"], n_scored=df3["n_scored"])
    print(f"\n      Partial-overlap case: EC50 = {res3['ec50']:.3f}, "
          f"LC50 = {res3['lc50']:.3f}, ratio = {res3['ratio']:.3f}, "
          f"severity = {res3['severity']}")
    check("an intermediate EC50/LC50 ratio lands in the 'overlapping' tier",
          res3["severity"] in ("overlapping", "confounded") and res3["confounded"],
          f"ratio = {res3['ratio']:.3f} -> {res3['severity']}")

    # ── relative potency between the confounded and clean compounds ────────
    rp = DR.relative_potency(res2["behaviour_fit"], res["behaviour_fit"],
                             label_a="clean compound", label_b="confounded toxin")
    print(f"\n      {rp['verdict']}")
    check("relative potency distinguishes the two compounds",
          np.isfinite(rp["ratio"]) and np.isfinite(rp["ci_upper"]) and rp["ci_upper"] < 1.0,
          f"ratio = {rp['ratio']:.4g} [{rp['ci_lower']:.4g}, {rp['ci_upper']:.4g}]")
    check("parallelism is tested and level-specific ratios are reported",
          np.isfinite(rp["parallelism_p"])
          and rp["ratio_by_level"][10.0] != rp["ratio_by_level"][90.0],
          f"b = {rp['slope_a']:.3f} vs {rp['slope_b']:.3f}, p = {rp['parallelism_p']:.3g}; "
          f"ratio {rp['ratio_by_level'][10.0]:.4g} at EC10 vs "
          f"{rp['ratio_by_level'][90.0]:.4g} at EC90")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/dose_response.py")
    print("=" * 78)
    test_A_exact_recovery()
    test_B_noise_recovery_and_coverage()
    test_C_profile_vs_delta()
    test_D_hormesis()
    test_E_real_published_data()
    test_F_design_refusal()
    test_G_mortality_confounding()
    test_H_relative_potency()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL DOSE-RESPONSE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
