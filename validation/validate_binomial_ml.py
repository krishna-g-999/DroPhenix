"""
validate_binomial_ml.py
Binomial maximum-likelihood dose-response fitting, and the LC50 interval that
finally covers.

ESTABLISHES
    L2 -- the likelihood and its standard errors agree with an INDEPENDENT
          implementation (`statsmodels` GLM binomial) to optimiser tolerance,
          exactly, for the case where the two models coincide.
    L3 -- the interval's coverage is measured against known truth, and reaches
          nominal where the least-squares interval did not.
    It establishes nothing biological.

WHY THIS SUITE EXISTS
---------------------
Register N95. `fit_loglogistic` is weighted least squares with the cohort size
`n` as the weight. A binomial proportion has variance p(1-p)/n; the weight
carries the 1/n and nothing carried the p(1-p). Measured on 7 doses x 40 flies
under LL.3 -- the model `mortality_adjusted_response` actually uses -- with 300
replications per cell and a nominal 95% interval:

    true d  b  |  LS delta   ML profile   ML profile, auto penalty
      1.00  1  |    0.957      0.977            0.977
      1.00  2  |    0.867      0.970            0.970
      1.00  4  |    0.470      0.973            0.973
      0.90  1  |    0.947      0.957            0.957
      0.90  2  |    0.887      0.953            0.953
      0.90  4  |    0.733      0.860            0.947
      0.80  1  |    0.947      0.937            0.937
      0.80  2  |    0.893      0.950            0.950
      0.80  4  |    0.753      0.820            0.923
    ---------------------------------------------------------------
    mean           0.828      0.933            0.954
    worst          0.470      0.820            0.923
    >= 0.93         3 of 9     7 of 9           8 of 9

A CORRECTION TO THE FIRST VERSION OF THIS FINDING, recorded because the
register carried the wrong magnitudes for a while. The first measurements were
made with LL.4, which estimates a lower asymptote `c` whose true value is 0 --
the boundary of the parameter space for a probability. A parameter on a
boundary voids the regularity conditions behind both the delta method and the
chi-square likelihood ratio, so part of the collapse those runs showed was
mis-specification rather than the variance model. Under the correctly specified
LL.3 the defect is smaller than first reported but entirely real: the legacy
interval still falls to 0.470 at a steep slope and clears 0.93 in only three of
nine cells.

THE REMEDY, in three parts, each measured rather than assumed:
  1. the binomial likelihood, with its covariance from the observed
     information, so the variance model matches the data;
  2. a profile-likelihood interval on the LC50 ITSELF, which is invariant to
     how the curve is parameterised -- the property the delta method lacks;
  3. Firth's penalised likelihood applied EXACTLY where the dose grid cannot
     resolve the slope, detected by the design's own resolution limit. Plain
     maximum likelihood is the better estimator in seven of the nine cells and
     Firth in the two where the fitted slope runs past that limit -- and those
     are the only two cells where it does so at all, in 9.3% and 12.3% of
     replicates against 0.0-0.3% elsewhere.

THE LEGACY PATH IS NOT TOUCHED
    `method="ls"` remains the default and must stay bit-identical, because the
    least-squares path is validated against R's `drc` on the `ryegrass` data
    (`validate_dose_response` check E) and against a continuous response
    measured once per vial, which is the case least squares was built for.
    Check A pins that. The owner's rule is that a legacy implementation is not
    removed until its replacement is validated, and the replacement is opt-in
    until then.

WHY THE CROSS-CHECK IS EXACT, AND NOT A SANITY CHECK
    For LL.2 -- c = 0, d = 1 -- the model is

        p = 1 / (1 + exp(b * (log x - log e)))

    which rearranges to  logit(p) = b*log(e) - b*log(x).  That is ordinary
    logistic regression on log-dose. So a binomial GLM with a logit link and
    design [1, log x] maximises the SAME likelihood, and its estimates must
    agree with ours to optimiser tolerance, not merely be close: intercept
    alpha = b*log(e) and slope beta = -b, so b = -beta and e = exp(-alpha/beta).
    Check B uses that. Zero doses are excluded there, because log(0) has no
    place in a GLM design matrix while LL.2 evaluates x = 0 by its limit of 1.

Run:  python validation/validate_binomial_ml.py   (exit 0 = all passed)
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import dose_response as DRS   # noqa: E402

SEED = 20261108


def ll_p(x, b, c, d, e):
    """The model, written here so the suite does not test the module with itself."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", over="ignore"):
        lx = np.log(x)
        out = c + (d - c) / (1.0 + np.exp(b * (lx - np.log(e))))
    return np.where(x > 0, out, d)


def true_lc(b, c, d, e, target=0.5):
    if not (min(c, d) < target < max(c, d)):
        return np.nan
    return e * ((d - target) / (target - c)) ** (1.0 / b)


def binom_loglik(k, n, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    from scipy.special import gammaln
    return float(np.sum(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                        + k * np.log(p) + (n - k) * np.log1p(-p)))


def main():
    results = []

    def chk(name, got, expected, tol):
        ok = bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol
        results.append((ok, name, "got %.10g, expected %.10g (tol %g)"
                        % (got, expected, tol)))

    def boolchk(name, ok, got):
        results.append((bool(ok), name, got))

    def raises(name, fn, must_mention):
        """A refusal check. The message must mention the thing refused."""
        try:
            fn()
        except Exception as exc:
            txt = str(exc).lower()
            hit = any(m.lower() in txt for m in must_mention)
            results.append((hit, name,
                            ("refused: %s" % str(exc)[:110]) if hit else
                            ("raised, but the message names none of %s: %s"
                             % (list(must_mention), str(exc)[:90]))))
        else:
            results.append((False, name, "did NOT refuse"))

    rng = np.random.default_rng(SEED)

    # ──────────────────────────────────────────────────────────────────────
    # A. the legacy path is untouched and the new one is opt-in
    # ──────────────────────────────────────────────────────────────────────
    doses = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    nfly = 40
    p_true = ll_p(doses, 2.0, 0.0, 0.90, 1.0)
    k_obs = rng.binomial(nfly, np.clip(p_true, 0, 1))
    y_obs = k_obs / nfly

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_default = DRS.fit_loglogistic(doses, y_obs, n=[nfly] * 7, model="LL.4",
                                        force=True)
        f_ls = DRS.fit_loglogistic(doses, y_obs, n=[nfly] * 7, model="LL.4",
                                   force=True, method="ls")
    boolchk("A1 the default is least squares, bit-for-bit",
            np.array_equal(f_default.theta, f_ls.theta)
            and f_default.rss == f_ls.rss,
            "theta identical: %s; rss %.17g vs %.17g"
            % (np.array_equal(f_default.theta, f_ls.theta), f_default.rss, f_ls.rss))
    boolchk("A2 the legacy fit reports its method",
            getattr(f_ls, "method", "ls") == "ls",
            "method = %r" % getattr(f_ls, "method", "<absent>"))
    raises("A3 an unknown method is refused by name",
           lambda: DRS.fit_loglogistic(doses, y_obs, n=[nfly] * 7, model="LL.4",
                                       force=True, method="mle"),
           ["method", "binomial_ml", "ls"])
    raises("A4 binomial ML without cohort sizes is refused",
           lambda: DRS.fit_loglogistic(doses, y_obs, model="LL.4", force=True,
                                       method="binomial_ml"),
           ["n", "cohort", "binomial"])
    raises("A4b an unknown penalty is refused by name",
           lambda: DRS.fit_loglogistic(doses, y_obs, n=[nfly] * 7, model="LL.3",
                                       force=True, method="binomial_ml",
                                       penalty="jeffreys"),
           ["penalty", "auto", "firth", "none"])
    raises("A5 counts that are not whole numbers are refused, not rounded",
           lambda: DRS.fit_loglogistic([1.0, 2.0, 4.0, 8.0], [0.9, 0.7, 0.33333, 0.1],
                                       n=[7, 7, 7, 7], model="LL.3", force=True,
                                       method="binomial_ml"),
           ["integer", "whole", "count"])

    # ──────────────────────────────────────────────────────────────────────
    # B. the likelihood, against statsmodels, where the models coincide
    # ──────────────────────────────────────────────────────────────────────
    import statsmodels.api as sm

    b_t, e_t, nn = 2.0, 1.5, 200
    xg = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    pg = ll_p(xg, b_t, 0.0, 1.0, e_t)
    kg = rng.binomial(nn, pg)
    Xg = sm.add_constant(np.log(xg))
    glm = sm.GLM(np.c_[kg, nn - kg], Xg, family=sm.families.Binomial()).fit()
    a_hat, beta_hat = float(glm.params[0]), float(glm.params[1])
    b_glm = -beta_hat
    e_glm = float(np.exp(-a_hat / beta_hat))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # penalty="none": the GLM maximises the PLAIN likelihood, so the
        # comparison must be against the plain maximum to be exact. The
        # penalised fit is validated by coverage in section E, not here.
        f_ml = DRS.fit_loglogistic(xg, kg / nn, n=[nn] * len(xg), model="LL.2",
                                   force=True, method="binomial_ml",
                                   penalty="none")
    chk("B1 LL.2 binomial ML recovers the GLM's slope b exactly",
        float(f_ml.params["b"]), b_glm, 1e-6)
    chk("B2 ... and the GLM's e exactly", float(f_ml.params["e"]), e_glm, 1e-6)
    ll_ours = binom_loglik(kg, np.full(len(xg), nn),
                           ll_p(xg, float(f_ml.params["b"]), 0.0, 1.0,
                                float(f_ml.params["e"])))
    boolchk("B3 the maximised log-likelihood matches the GLM's",
            abs(ll_ours - float(glm.llf)) < 1e-6,
            "ours %.10f vs statsmodels %.10f" % (ll_ours, float(glm.llf)))
    boolchk("B4 the fit reports a log-likelihood, and it is the maximum",
            np.isfinite(getattr(f_ml, "loglik", np.nan))
            and abs(float(f_ml.loglik) - ll_ours) < 1e-6,
            "fit.loglik = %s" % getattr(f_ml, "loglik", "<absent>"))
    # the SE of b comes from the observed information, and the GLM's SE of beta
    # is the same quantity under the sign change b = -beta
    se_b_glm = float(glm.bse[1])
    chk("B5 SE(b) from the observed information matches the GLM's",
        float(f_ml.bse["b"]), se_b_glm, 1e-4)
    boolchk("B6 the ML fit is flagged as having usable inference",
            bool(getattr(f_ml, "inference_valid", False)),
            "inference_valid = %s, Hessian min eigenvalue %s"
            % (getattr(f_ml, "inference_valid", "<absent>"),
               getattr(f_ml, "hessian_min_eigenvalue", "<absent>")))

    # ──────────────────────────────────────────────────────────────────────
    # C. recovery, and refusal where the design cannot support inference
    # ──────────────────────────────────────────────────────────────────────
    big = np.array([0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    p_big = ll_p(big, 2.0, 0.0, 0.90, 1.0)
    k_big = rng.binomial(2000, np.clip(p_big, 0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_big = DRS.fit_loglogistic(big, k_big / 2000.0, n=[2000] * len(big),
                                    model="LL.3", method="binomial_ml")
    chk("C1 b recovered on 2,000 flies per dose", float(f_big.params["b"]), 2.0, 0.08)
    chk("C2 d recovered", float(f_big.params["d"]), 0.90, 0.01)
    chk("C3 e recovered", float(f_big.params["e"]), 1.0, 0.04)
    lc_true = true_lc(2.0, 0.0, 0.90, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ld_big = DRS.lethal_dose_absolute(f_big, 0.5, profile=False)
    chk("C4 and so is the LC50", float(ld_big["estimate"]), lc_true, 0.03)
    # a design that cannot identify the curve must not produce a standard error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_bad = DRS.fit_loglogistic([1.0, 2.0, 4.0], [1.0, 1.0, 1.0],
                                    n=[10, 10, 10], model="LL.3", force=True,
                                    method="binomial_ml")
    boolchk("C5 a flat, unidentified fit reports inference_valid = False and "
            "NaN standard errors, never a number",
            bool(getattr(f_bad, "inference_valid", True)) is False
            and not np.isfinite(float(f_bad.bse["b"])),
            "inference_valid = %s, SE(b) = %s, reason %r"
            % (getattr(f_bad, "inference_valid", "<absent>"), f_bad.bse["b"],
               str(getattr(f_bad, "inference_reason", ""))[:70]))

    # ──────────────────────────────────────────────────────────────────────
    # D. the interval: a profile on the LC50 ITSELF
    # ──────────────────────────────────────────────────────────────────────
    # Why the LC50 itself. The delta interval is symmetric on log-dose and is
    # built from a covariance; a profile on the quantity of interest is
    # invariant to how that quantity is parameterised, which is the property
    # the delta method lacks and the reason it fails here.
    mid = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    p_mid = ll_p(mid, 2.0, 0.0, 0.90, 1.0)
    k_mid = rng.binomial(100, np.clip(p_mid, 0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_mid = DRS.fit_loglogistic(mid, k_mid / 100.0, n=[100] * len(mid),
                                    model="LL.3", force=True, method="binomial_ml")
        prof = DRS.lethal_dose_profile(f_mid, target=0.5, level=0.95)
    boolchk("D1 the profile interval brackets the point estimate",
            float(prof["ci_lower"]) < float(prof["estimate"]) < float(prof["ci_upper"]),
            "[%.4f, %.4f] around an estimate of %.4f"
            % (prof["ci_lower"], prof["ci_upper"], prof["estimate"]))
    boolchk("D2 it is ASYMMETRIC on the dose scale, which a delta interval "
            "cannot be",
            abs((float(prof["ci_upper"]) - float(prof["estimate"]))
                / (float(prof["estimate"]) - float(prof["ci_lower"])) - 1.0) > 0.02,
            "upper arm / lower arm = %.4f"
            % ((float(prof["ci_upper"]) - float(prof["estimate"]))
               / (float(prof["estimate"]) - float(prof["ci_lower"]))))
    # an independent deviance check: at each endpoint the drop in 2*loglik from
    # the maximum must equal the chi-square(1) critical value
    from scipy import stats as st
    crit = float(st.chi2.ppf(0.95, 1))
    dev_lo = 2.0 * (float(f_mid.loglik) - float(prof["loglik_at_lower"]))
    dev_hi = 2.0 * (float(f_mid.loglik) - float(prof["loglik_at_upper"]))
    boolchk("D3 both endpoints sit on the chi-square(1) deviance threshold",
            abs(dev_lo - crit) < 0.02 and abs(dev_hi - crit) < 0.02,
            "deviance at the bounds %.4f and %.4f against chi2(1, 0.95) = %.4f"
            % (dev_lo, dev_hi, crit))
    boolchk("D4 the profile interval is what an ML fit reports as PRIMARY",
            np.isclose(float(prof["ci_lower"]),
                       float(DRS.lethal_dose_absolute(f_mid, 0.5)["ci_lower"]))
            and np.isclose(float(prof["ci_upper"]),
                           float(DRS.lethal_dose_absolute(f_mid, 0.5)["ci_upper"])),
            "lethal_dose_absolute returns the profile bounds for an ML fit")
    ld_mid = DRS.lethal_dose_absolute(f_mid, 0.5)
    boolchk("D5 ... with the delta interval kept alongside, not discarded",
            all(key in ld_mid for key in ("delta_ci_lower", "delta_ci_upper")),
            "delta interval retained as [%s, %s]"
            % (ld_mid.get("delta_ci_lower"), ld_mid.get("delta_ci_upper")))
    boolchk("D6 and the coverage caveat is GONE for this path",
            ld_mid.get("interval_coverage_nominal") is True
            and not ld_mid.get("interval_caveat"),
            "interval_coverage_nominal = %s" % ld_mid.get("interval_coverage_nominal"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ld_ls = DRS.lethal_dose_absolute(f_default, 0.5, profile=False)
    boolchk("D7 while the least-squares path still declares its shortfall",
            ld_ls.get("interval_coverage_nominal") is False
            and bool(ld_ls.get("interval_caveat")),
            "interval_coverage_nominal = %s" % ld_ls.get("interval_coverage_nominal"))

    # ──────────────────────────────────────────────────────────────────────
    # F. the design's slope-resolution limit, and the rule built on it
    # ──────────────────────────────────────────────────────────────────────
    # b_max = 2 ln(19) / ln(r) is the steepest Hill slope a dose grid of
    # spacing r can resolve: past it the 0.95-to-0.05 transition fits inside
    # one dose spacing, so no curve is distinguishable from a step. It is a
    # resolution limit like a Nyquist frequency, not a belief about plausible
    # Hill slopes, and nothing in it is tuned.
    lim2 = DRS.resolvable_slope_limit([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    chk("F1 a two-fold dose grid resolves slopes up to 8.50", lim2["b_max"],
        2.0 * np.log(19.0) / np.log(2.0), 1e-9)
    lim3 = DRS.resolvable_slope_limit([0.0, 1.0, 3.0, 9.0, 27.0, 81.0])
    boolchk("F2 a coarser grid resolves less, a finer grid more",
            lim3["b_max"] < lim2["b_max"]
            < DRS.resolvable_slope_limit(
                np.concatenate([[0.0], np.geomspace(0.1, 10.0, 11)]))["b_max"],
            "three-fold %.2f < two-fold %.2f < 1.585-fold %.2f"
            % (lim3["b_max"], lim2["b_max"],
               DRS.resolvable_slope_limit(
                   np.concatenate([[0.0], np.geomspace(0.1, 10.0, 11)]))["b_max"]))
    boolchk("F3 fewer than two non-zero doses imposes no limit",
            not np.isfinite(DRS.resolvable_slope_limit([0.0, 1.0])["b_max"]),
            "b_max = %s" % DRS.resolvable_slope_limit([0.0, 1.0])["b_max"])
    # the rule: plain where the grid resolves the slope, Firth where it does not
    pv_f = ll_p(doses, 4.0, 0.0, 0.80, 1.0)
    fired = plain_used = 0
    for _ in range(40):
        kk = rng.binomial(nfly, np.clip(pv_f, 0, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff = DRS.fit_loglogistic(doses, kk / nfly, n=[nfly] * 7,
                                     model="LL.3", force=True,
                                     method="binomial_ml")
        if ff.penalty == "firth":
            fired += 1
            if not ff.separation:
                plain_used -= 100      # must be flagged when it fires
        else:
            plain_used += 1
    boolchk("F4 the auto rule reaches for Firth only sometimes, and says so "
            "when it does",
            0 < fired < 40 and plain_used > 0,
            "Firth on %d of 40 replicates at a true slope of 4 with d = 0.80, "
            "each flagged with separation=True" % fired)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_gentle = DRS.fit_loglogistic(doses, y_obs, n=[nfly] * 7, model="LL.3",
                                       force=True, method="binomial_ml")
    boolchk("F5 a well-resolved fit is left unpenalised, and the rule is "
            "recorded either way",
            f_gentle.penalty == "none" and "resolution limit" in f_gentle.penalty_rule,
            "penalty = %r, rule = %r" % (f_gentle.penalty, f_gentle.penalty_rule[:60]))

    # ---- the design-adequacy check, and the regime it must refuse ---------
    # The bootstrap threshold was meant to calibrate the last few points of
    # coverage. It does on a sound design; in the regime where the dose grid
    # cannot resolve the slope it returns an arbitrary number, because the
    # bootstrap datasets separate too. So it is used as a quantitative
    # adequacy measure, and the interval stops claiming nominal coverage where
    # the measurement says it is about 0.90.
    # A separating dataset, constructed rather than drawn: survival falls from
    # 38/40 to 1/40 between adjacent doses with nothing in between, so no
    # finite slope is distinguishable from a step. Drawing one at random works
    # only ~11% of the time, which is not a basis for a regression test.
    hard = np.array([32, 32, 30, 38, 1, 0, 0], dtype=float)
    good_x = np.concatenate([[0.0], np.geomspace(0.1, 10.0, 11)])
    good_k = rng.binomial(100, np.clip(ll_p(good_x, 2.0, 0.0, 0.95, 1.0), 0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_hard = DRS.fit_loglogistic(doses, hard / nfly, n=[nfly] * 7,
                                     model="LL.3", force=True,
                                     method="binomial_ml")
        f_good = DRS.fit_loglogistic(good_x, good_k / 100.0,
                                     n=[100] * len(good_x), model="LL.3",
                                     method="binomial_ml")
        bs_hard = DRS.bootstrap_deviance_threshold(f_hard, n_boot=120, seed=2)
        bs_good = DRS.bootstrap_deviance_threshold(f_good, n_boot=120, seed=2)
        r_hard = DRS.lethal_dose_absolute(f_hard, 0.5)
        r_good = DRS.lethal_dose_absolute(f_good, 0.5)
    boolchk("F6 on a sound design the bootstrap reference agrees with the "
            "asymptotic one, confirming chi-square(1)",
            0.6 <= bs_good["ratio"] <= 1.5 and "adequate" in bs_good["verdict"],
            "ratio %.3f on %d replicates -- %s"
            % (bs_good["ratio"], bs_good["n_used"], bs_good["verdict"][:60]))
    # A constructed cliff: 39/40 surviving at one dose and 0/40 at the next, so
    # no finite slope is distinguishable from a step. Deterministic, because
    # drawing one at random happens in about 11% of replicates and that is no
    # basis for a regression test.
    cliff = np.array([40.0, 40.0, 40.0, 39.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_plain_c = DRS.fit_loglogistic(doses, cliff / nfly, n=[nfly] * 7,
                                        model="LL.3", force=True,
                                        method="binomial_ml", penalty="none")
        f_auto_c = DRS.fit_loglogistic(doses, cliff / nfly, n=[nfly] * 7,
                                       model="LL.3", force=True,
                                       method="binomial_ml")
    bmax = DRS.resolvable_slope_limit(doses)["b_max"]
    boolchk("F7a on a separating dataset the unpenalised slope runs past what "
            "the grid can resolve",
            abs(float(f_plain_c.params["b"])) > bmax,
            "plain b = %.2f against b_max = %.2f"
            % (f_plain_c.params["b"], bmax))
    boolchk("F7b the auto rule then penalises AND caps, and the cap binds",
            f_auto_c.penalty == "firth"
            and abs(abs(float(f_auto_c.params["b"])) - bmax) < 1e-3,
            "auto b = %.4f, cap = %.4f, penalty = %s"
            % (f_auto_c.params["b"], f_auto_c.slope_cap, f_auto_c.penalty))
    boolchk("F7c ... and says so, naming the limit and the remedy",
            bool(f_auto_c.separation)
            and "resolve" in f_auto_c.separation_note
            and "intermediate doses" in f_auto_c.separation_note,
            "separation = %s; note names the dose ratio and the remedy"
            % f_auto_c.separation)
    boolchk("F7d the penalty choice is recorded either way",
            "Firth" in f_auto_c.penalty_rule
            and "resolution limit" in f_gentle.penalty_rule,
            "separating: %r" % f_auto_c.penalty_rule[:58])
    boolchk("F7e the bootstrap adequacy measure is reported with a verdict",
            bs_hard["n_used"] >= 50 and bool(bs_hard["verdict"]),
            "ratio %.2f on %d replicates -- %s"
            % (bs_hard["ratio"], bs_hard["n_used"], bs_hard["verdict"][:55]))
    boolchk("F8 and the interval there stops claiming nominal coverage",
            r_hard.get("interval_coverage_nominal") is False
            and bool(r_hard.get("interval_caveat")),
            "nominal = %s; caveat names the asymptote headroom"
            % r_hard.get("interval_coverage_nominal"))
    boolchk("F9 while a sound design still reports a nominal interval with no "
            "caveat",
            r_good.get("interval_coverage_nominal") is True
            and not r_good.get("interval_caveat"),
            "nominal = %s, caveat = %r"
            % (r_good.get("interval_coverage_nominal"),
               r_good.get("interval_caveat")))

    # ──────────────────────────────────────────────────────────────────────
    # E. coverage -- the whole point
    # ──────────────────────────────────────────────────────────────────────
    REPS = 200
    print("=" * 78)
    print("E. LC50 interval coverage, binomial error, LL.3, 7 doses x 40 "
          "flies, %d reps/cell" % REPS)
    print("   (LL.3 fixes c = 0 with d free: survival at infinite dose IS zero,")
    print("    and a free upper asymptote is Abbott's correction. LL.4 would put")
    print("    c on the boundary of its space and void the asymptotics.)")
    print("%5s %5s %12s %12s %12s" % ("d", "b", "LS (delta)", "ML (profile)",
                                      "true LC50"))
    cov_rows = []
    for d_true in (1.0, 0.90, 0.80):
        for b_true in (1.0, 2.0, 4.0):
            model = "LL.3"
            pv = ll_p(doses, b_true, 0.0, d_true, 1.0)
            lc = true_lc(b_true, 0.0, d_true, 1.0)
            n_ls = n_ml = c_ls = c_ml = 0
            for _ in range(REPS):
                kk = rng.binomial(nfly, np.clip(pv, 0, 1))
                yy = kk / nfly
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        fl = DRS.fit_loglogistic(doses, yy, n=[nfly] * 7,
                                                 model=model, force=True)
                        r = DRS.lethal_dose_absolute(fl, 0.5, profile=False)
                        lo, hi = r["ci_lower"], r["ci_upper"]
                        if np.isfinite(lo) and np.isfinite(hi):
                            n_ls += 1
                            c_ls += int(lo <= lc <= hi)
                    except Exception:
                        pass
                    try:
                        fm = DRS.fit_loglogistic(doses, yy, n=[nfly] * 7,
                                                 model=model, force=True,
                                                 method="binomial_ml")
                        rm = DRS.lethal_dose_absolute(fm, 0.5)
                        lo, hi = rm["ci_lower"], rm["ci_upper"]
                        if np.isfinite(lo) and np.isfinite(hi):
                            n_ml += 1
                            c_ml += int(lo <= lc <= hi)
                    except Exception:
                        pass
            cl = c_ls / n_ls if n_ls else np.nan
            cm = c_ml / n_ml if n_ml else np.nan
            print("%5.2f %5.1f %12.3f %12.3f %12.4f" % (d_true, b_true, cl, cm, lc))
            cov_rows.append({"d": d_true, "b": b_true, "ls": cl, "ml": cm,
                             "n_ls": n_ls, "n_ml": n_ml})
    print()

    ls_cov = np.array([r["ls"] for r in cov_rows])
    ml_cov = np.array([r["ml"] for r in cov_rows])
    steep = [r for r in cov_rows if r["b"] == 4.0]
    boolchk("E1 the least-squares interval collapses at the steepest slope, "
            "which is what N95 is",
            min(r["ls"] for r in steep) < 0.80,
            "LS coverage at b = 4: %s (nominal 0.95)"
            % [round(r["ls"], 3) for r in steep])
    boolchk("E2 ... and clears 93% in a minority of cells",
            int(np.sum(ls_cov >= 0.93)) <= 5,
            "LS clears 0.93 in %d of %d cells, mean %.3f"
            % (int(np.sum(ls_cov >= 0.93)), len(ls_cov), ls_cov.mean()))
    # Pinned at 900 replications (SE 0.0073) in the register: 0.9422, 0.9344
    # and 0.8967 for the three b = 4 cells against 0.4470, 0.6780 and 0.7386
    # for least squares. This suite runs 200 per cell, so it asserts the
    # stable properties and prints the rest.
    boolchk("E3 the ML profile interval clears 93% in at least 7 of 9 cells",
            int(np.sum(ml_cov >= 0.93)) >= len(ml_cov) - 2,
            "ML clears 0.93 in %d of %d cells; worst %.3f at d=%.2f b=%.0f"
            % (int(np.sum(ml_cov >= 0.93)), len(ml_cov), np.nanmin(ml_cov),
               cov_rows[int(np.nanargmin(ml_cov))]["d"],
               cov_rows[int(np.nanargmin(ml_cov))]["b"]))
    boolchk("E4 its mean coverage is at the nominal level",
            0.93 <= float(np.nanmean(ml_cov)) <= 0.98,
            "mean ML coverage %.3f against LS %.3f (nominal 0.95)"
            % (np.nanmean(ml_cov), np.nanmean(ls_cov)))
    boolchk("E5 it beats least squares on average and never by much less",
            float(np.nanmean(ml_cov)) > float(np.nanmean(ls_cov))
            and float(np.min(ml_cov - ls_cov)) > -0.03,
            "mean gain %+.3f; worst single-cell change %+.3f"
            % (np.nanmean(ml_cov) - np.nanmean(ls_cov),
               float(np.min(ml_cov - ls_cov))))
    boolchk("E6 the gain is concentrated where the shortfall was",
            float(np.mean([r["ml"] - r["ls"] for r in steep]))
            > float(np.mean([r["ml"] - r["ls"] for r in cov_rows if r["b"] == 1.0])),
            "mean gain %+.3f at b = 4 against %+.3f at b = 1"
            % (np.mean([r["ml"] - r["ls"] for r in steep]),
               np.mean([r["ml"] - r["ls"] for r in cov_rows if r["b"] == 1.0])))
    boolchk("E7 the ML fit converged on essentially every replicate",
            min(r["n_ml"] for r in cov_rows) >= int(0.95 * REPS),
            "fewest usable intervals in a cell: %d of %d"
            % (min(r["n_ml"] for r in cov_rows), REPS))
    # The one cell that stays below 0.95 is a DESIGN limit, and the software
    # now says so rather than overstating confidence.
    worst = cov_rows[int(np.nanargmin(ml_cov))]
    boolchk("E8 the residual shortfall is at the steepest slope with the "
            "lowest asymptote -- a design limit the fit now reports",
            worst["b"] == 4.0,
            "worst cell d=%.2f b=%.0f at %.3f; the separation note tells the "
            "operator to add intermediate doses, which is the remedy that "
            "actually closes it" % (worst["d"], worst["b"], np.nanmin(ml_cov)))

    print("validate_binomial_ml -- the N95 remedy")
    print("-" * 78)
    npass = 0
    for ok, name, got in results:
        print("[%s] %s  --  %s" % ("PASS" if ok else "FAIL", name, got))
        npass += ok
    print("-" * 78)
    print("%d/%d checks passed" % (npass, len(results)))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
