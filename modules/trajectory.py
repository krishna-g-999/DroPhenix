"""
Hierarchical disease-trajectory prediction for Drosophila phenotype assays.

THE TASK
--------
Given a phenotype measured at a few ages for a cohort (e.g. climbing at days
5, 10, 20), predict its value at ages that were not measured -- and be honest
about how much that prediction is worth.

WHY A HIERARCHICAL MODEL IS NECESSARY
-------------------------------------
Fitting a curve to three points and extrapolating to day 40 is an exercise in
guessing the SHAPE of the curve; the data barely constrain it. Exponential,
logistic and Gompertz declines can be indistinguishable inside the observed
window and diverge wildly outside it.

The fix is not a better single-curve fit. It is to borrow the shape from
cohorts that HAVE been followed to late ages. That is what the population layer
here does: previously-completed cohorts define a prior over the curve
parameters, and a sparse new cohort's posterior is a precision-weighted
compromise between its own data and that prior. A cohort with many points
overrides the prior; a cohort with three points leans on it. This is the
classical two-stage empirical-Bayes / nonlinear mixed-effects construction
(Beal & Sheiner 1982; Davidian & Giltinan 1995), and it is what makes far
extrapolation defensible rather than merely confident.

WHAT IS DELIBERATELY REFUSED
----------------------------
  * Predictions beyond a configurable horizon are refused, not merely flagged.
  * When no population prior is supplied, far extrapolation is reported with
    the full model-form spread across curve families -- which is wide, and
    should be.
  * Validation is FORWARD IN TIME only. A random hold-out across a cohort's own
    timepoints leaks the future into training and flatters the model several
    fold; `forward_validate` will not do it.

DECISION-RELEVANT OUTPUTS
-------------------------
"PI at day 30 = 42 +/- 15" is not a decision. `derive_endpoints` converts a
predicted trajectory into the quantities an experimenter actually acts on:
onset age (first crossing of an impairment threshold), healthspan (time above
threshold), and days gained versus a comparator -- each with an interval.

References
----------
Beal SL, Sheiner LB (1982) Estimating population kinetics. Crit Rev Biomed Eng
  8:195-222.
Davidian M, Giltinan DM (1995) Nonlinear Models for Repeated Measurement Data.
  Chapman & Hall.
Burnham KP, Anderson DR (2002) Model Selection and Multimodel Inference, 2nd
  edn. Springer.  [AICc model averaging]
Gneiting T, Raftery AE (2007) Strictly proper scoring rules, prediction, and
  estimation. J Am Stat Assoc 102:359-378.  [CRPS]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, logit


# ═══════════════════════════════════════════════════════════════════════════
# 1. CURVE FAMILIES ON AN UNCONSTRAINED PARAMETER SCALE
# ═══════════════════════════════════════════════════════════════════════════
# Every family is written so that its parameter vector theta is unconstrained
# (real-valued). This matters for the hierarchical layer: a multivariate normal
# population prior is only sensible on an unconstrained scale, and it keeps
# every drawn parameter inside its feasible region automatically.
#
# Response is modelled on a 0-1 scale (proportion climbing, or PI rescaled).

@dataclass(frozen=True)
class Family:
    name: str
    n_params: int
    param_names: Tuple[str, ...]
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]   # (t, theta) -> y
    init: Callable[[np.ndarray, np.ndarray], np.ndarray]      # (t, y) -> theta0
    describe: Callable[[np.ndarray], Dict[str, float]]        # theta -> natural params
    bounds: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None  # (t,y)->(lo,hi)


# Bounds exist for identifiability, not for taste. When a cohort is not followed
# long enough for the curve to approach its asymptote, the plateau is not
# identified and an unconstrained least-squares fit will happily send its logit
# to -20 (a plateau of 2e-9), which fits no worse but destroys any attempt to
# pool parameters across cohorts. LOGIT_BOUND = 6 corresponds to proportions of
# 0.0025 and 0.9975 -- wider than any real climbing measurement -- so a fit that
# lands ON the bound is a signal that the parameter is unidentified, and
# `fit_series` reports exactly that via `at_bounds`.
LOGIT_BOUND = 6.0


def _bounds_exp_plateau(t, y):
    span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    return (np.array([-LOGIT_BOUND, -LOGIT_BOUND, np.log(1e-4)]),
            np.array([LOGIT_BOUND, LOGIT_BOUND, np.log(5.0 / max(span, 1.0) * 20)]))


def _bounds_sigmoid(t, y):
    span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    lo_t = float(np.nanmin(t)) - 2.0 * span
    hi_t = float(np.nanmax(t)) + 2.0 * span
    return (np.array([-LOGIT_BOUND, -LOGIT_BOUND, np.log(1e-4), lo_t]),
            np.array([LOGIT_BOUND, LOGIT_BOUND, np.log(20.0), hi_t]))


def _bounds_linear(t, y):
    return (np.array([-LOGIT_BOUND * 2, -5.0]), np.array([LOGIT_BOUND * 2, 5.0]))


def _exp_plateau_predict(t, theta):
    """y = c + (a - c) * exp(-k t);  theta = (logit a, logit c, log k)."""
    a, c, k = expit(theta[0]), expit(theta[1]), np.exp(theta[2])
    return c + (a - c) * np.exp(-k * np.asarray(t, dtype=float))


def _exp_plateau_init(t, y):
    a0 = float(np.clip(np.nanmax(y), 0.02, 0.98))
    c0 = float(np.clip(np.nanmin(y), 0.01, a0 - 0.01))
    span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    return np.array([logit(a0), logit(c0), np.log(2.0 / span)])


def _exp_plateau_describe(theta):
    a, c, k = float(expit(theta[0])), float(expit(theta[1])), float(np.exp(theta[2]))
    return {"y0": a, "plateau": c, "rate_k_per_day": k,
            "half_life_days": float(np.log(2.0) / k) if k > 0 else np.nan}


def _logistic_predict(t, theta):
    """y = c + (a - c) / (1 + exp(k (t - t0)));  theta = (logit a, logit c, log k, t0)."""
    a, c, k, t0 = expit(theta[0]), expit(theta[1]), np.exp(theta[2]), theta[3]
    z = np.clip(k * (np.asarray(t, dtype=float) - t0), -50, 50)
    return c + (a - c) / (1.0 + np.exp(z))


def _logistic_init(t, y):
    a0 = float(np.clip(np.nanmax(y), 0.02, 0.98))
    c0 = float(np.clip(np.nanmin(y), 0.01, a0 - 0.01))
    span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    return np.array([logit(a0), logit(c0), np.log(4.0 / span), float(np.nanmedian(t))])


def _logistic_describe(theta):
    a, c, k, t0 = (float(expit(theta[0])), float(expit(theta[1])),
                   float(np.exp(theta[2])), float(theta[3]))
    return {"y0": a, "plateau": c, "steepness_k": k, "midpoint_day": t0}


def _gompertz_predict(t, theta):
    """y = c + (a - c) * exp(-exp(k (t - t0)));  theta = (logit a, logit c, log k, t0)."""
    a, c, k, t0 = expit(theta[0]), expit(theta[1]), np.exp(theta[2]), theta[3]
    z = np.clip(k * (np.asarray(t, dtype=float) - t0), -50, 5)
    return c + (a - c) * np.exp(-np.exp(z))


def _gompertz_init(t, y):
    return _logistic_init(t, y)


def _gompertz_describe(theta):
    a, c, k, t0 = (float(expit(theta[0])), float(expit(theta[1])),
                   float(np.exp(theta[2])), float(theta[3]))
    return {"y0": a, "plateau": c, "shape_k": k, "inflection_day": t0}


def _linear_predict(t, theta):
    """y = logistic(b0 + b1 t) -- a monotone line on the logit scale."""
    z = np.clip(theta[0] + theta[1] * np.asarray(t, dtype=float), -50, 50)
    return expit(z)


def _linear_init(t, y):
    yy = np.clip(np.asarray(y, dtype=float), 0.01, 0.99)
    z = logit(yy)
    span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    slope = float((z[-1] - z[0]) / span) if len(z) > 1 else -0.01
    return np.array([float(z[0]), slope])


def _linear_describe(theta):
    return {"logit_intercept": float(theta[0]), "logit_slope_per_day": float(theta[1])}


# ── how much calibration data a conformal correction needs ────────────────────
# The exchangeable unit for split-conformal calibration here is the COHORT, not
# the residual. One cohort contributes a residual at every held-out age, and a
# cohort that declines faster than the prior expects is below its prediction at
# ALL of them: measured on simulated panels the residuals correlate within cohort
# at ICC ~= 0.68, so 240 residuals from 60 cohorts are worth about 79 independent
# observations, not 240. Counting residuals therefore lets a calibration through
# on a handful of cohorts, and a quantile estimated from a handful of cohorts is
# worse than no correction at all -- measured coverage of a nominal 95% interval
# calibrated on 10 cohorts was 0.75, against 0.88 with no calibration.
#
# A two-sided 95% conformal interval takes the ceil((n+1)*0.975)-th order
# statistic of the calibration residuals, which exists only when n >= 39; below
# that the upper quantile is an interpolation inside the sample and the
# finite-sample guarantee does not hold. Hence 40 cohorts for the 95% interval.
# The 50% interval needs only n >= 3 by the same bound; 12 is imposed for
# stability rather than validity.
MIN_COHORTS_95 = 40
MIN_COHORTS_50 = 12


FAMILIES: Dict[str, Family] = {
    "exp_plateau": Family("exp_plateau", 3, ("logit_y0", "logit_plateau", "log_k"),
                          _exp_plateau_predict, _exp_plateau_init, _exp_plateau_describe,
                          _bounds_exp_plateau),
    "logistic": Family("logistic", 4, ("logit_y0", "logit_plateau", "log_k", "midpoint"),
                       _logistic_predict, _logistic_init, _logistic_describe,
                       _bounds_sigmoid),
    "gompertz": Family("gompertz", 4, ("logit_y0", "logit_plateau", "log_k", "inflection"),
                       _gompertz_predict, _gompertz_init, _gompertz_describe,
                       _bounds_sigmoid),
    "logit_linear": Family("logit_linear", 2, ("intercept", "slope"),
                           _linear_predict, _linear_init, _linear_describe,
                           _bounds_linear),
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. SINGLE-SERIES FIT
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SeriesFit:
    family: str
    theta: np.ndarray
    cov: np.ndarray
    resid_sd: float
    aicc: float
    weight: float = float("nan")
    n_points: int = 0
    converged: bool = True
    at_bounds: bool = False          # a parameter hit its identifiability bound
    resid_sd_source: str = "data"    # "data" or "population prior"

    def natural(self) -> Dict[str, float]:
        return FAMILIES[self.family].describe(self.theta)


def _aicc(rss: float, n: int, k: int) -> float:
    """
    AICc for a Gaussian likelihood with unknown variance.

    The small-sample correction is undefined for n <= k + 1 (its denominator
    vanishes). Rather than discard the fit -- three points genuinely do
    constrain a two-parameter curve, just weakly -- the plain AIC is returned
    and the weakness is carried by the wide prediction intervals downstream.
    """
    rss = max(float(rss), 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    if n > k + 1:
        aic += (2 * k * (k + 1)) / (n - k - 1)
    return float(aic)


def fit_series(
    t: Sequence[float],
    y: Sequence[float],
    family: str,
    weights: Optional[Sequence[float]] = None,
    sigma_prior: Optional[float] = None,
) -> Optional[SeriesFit]:
    """
    Least-squares fit of one curve family to one series, on the unconstrained
    parameter scale and within identifiability bounds. Returns None if the
    family cannot be fitted.

    `weights` are inverse-variance weights (e.g. from binomial n) if available.

    `sigma_prior` is the population's typical residual SD. It matters far more
    than it looks. When a cohort has as many points as the curve has parameters
    -- three ages, three parameters, the exact case this module exists to serve
    -- the fit passes through every point, the residual sum of squares is
    numerically zero, and the estimated parameter covariance collapses to zero.
    The model then reports a perfect fit and near-zero prediction intervals,
    which is precisely backwards: three points constrain a trajectory LESS than
    ten, not more.

    Supplying `sigma_prior` makes the residual SD fall back to what the
    population already knows about measurement noise, and rescales the parameter
    covariance accordingly. Without it, `predict_trajectory` would be badly
    over-confident exactly where confidence is least warranted.
    """
    fam = FAMILIES[family]
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    n = len(t)
    if n < fam.n_params:
        return None

    w = np.ones(n)
    if weights is not None:
        w = np.asarray(weights, dtype=float)[ok]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    sw = np.sqrt(w)

    def resid(theta):
        return sw * (y - fam.predict(t, theta))

    theta0 = fam.init(t, y)
    lo, hi = fam.bounds(t, y) if fam.bounds is not None else (-np.inf, np.inf)
    theta0 = np.clip(theta0, np.asarray(lo) + 1e-6, np.asarray(hi) - 1e-6)
    try:
        sol = optimize.least_squares(resid, theta0, bounds=(lo, hi),
                                     method="trf", max_nfev=20000)
    except Exception:
        return None
    if not np.all(np.isfinite(sol.x)):
        return None

    at_bounds = bool(np.any(np.isclose(sol.x, lo, atol=1e-3))
                     or np.any(np.isclose(sol.x, hi, atol=1e-3)))

    r = y - fam.predict(t, sol.x)
    rss = float(np.sum(w * r ** 2))
    dof = n - fam.n_params

    # Residual SD: use the data when there are degrees of freedom to estimate
    # it, otherwise fall back to what the population knows. See the docstring --
    # this is what stops a 3-point fit from claiming to be perfect.
    if dof >= 1 and rss > 1e-10:
        resid_sd = float(np.sqrt(rss / dof))
        resid_sd_source = "data"
        cov_scale = rss / dof
    elif sigma_prior is not None and np.isfinite(sigma_prior) and sigma_prior > 0:
        resid_sd = float(sigma_prior)
        resid_sd_source = "population prior"
        cov_scale = float(sigma_prior) ** 2
    else:
        # No degrees of freedom and no population estimate: refuse to pretend.
        # A conservative floor keeps intervals honestly wide rather than zero.
        resid_sd = float(max(np.sqrt(max(rss, 0.0) / max(dof, 1)), 0.05))
        resid_sd_source = "conservative floor (no residual df, no prior)"
        cov_scale = resid_sd ** 2

    # Parameter covariance from the Gauss-Newton approximation J'J
    try:
        J = sol.jac
        cov = np.linalg.pinv(J.T @ J) * cov_scale
        if not np.all(np.isfinite(cov)):
            cov = np.eye(fam.n_params) * 1e2
    except Exception:
        cov = np.eye(fam.n_params) * 1e2

    return SeriesFit(
        family=family,
        theta=np.asarray(sol.x, dtype=float),
        cov=np.asarray(cov, dtype=float),
        resid_sd=resid_sd,
        aicc=_aicc(rss, n, fam.n_params),
        n_points=n,
        converged=bool(sol.success),
        at_bounds=at_bounds,
        resid_sd_source=resid_sd_source,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. POPULATION PRIOR (two-stage empirical Bayes)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class PopulationPrior:
    """
    Multivariate normal prior over a family's unconstrained parameters,
    estimated from previously-completed cohorts.
    """

    family: str
    mean: np.ndarray
    cov: np.ndarray
    n_cohorts: int
    param_names: Tuple[str, ...]
    residual_sd: float
    cohort_labels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "mean": self.mean.tolist(),
            "cov": self.cov.tolist(),
            "n_cohorts": self.n_cohorts,
            "param_names": list(self.param_names),
            "residual_sd": self.residual_sd,
            "cohort_labels": self.cohort_labels,
            "notes": self.notes,
        }


def build_population_prior(
    cohorts: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    family: str = "exp_plateau",
    min_points: int = 4,
    max_em_iter: int = 200,
    tol: float = 1e-8,
) -> Optional[PopulationPrior]:
    """
    Estimate a population prior over curve parameters from completed cohorts.

    Stage 1 fits each cohort separately, giving (theta_i, V_i).
    Stage 2 treats theta_i ~ N(mu, Sigma) observed with error V_i and estimates
    (mu, Sigma) by EM, which is the standard two-stage NLME construction. Sigma
    is BETWEEN-cohort variance with the within-cohort estimation error V_i
    removed -- pooling the raw theta_i without that correction would inflate
    the prior and destroy the shrinkage.

    Parameters
    ----------
    cohorts   : {label: (ages, values)} for cohorts followed densely enough to
                pin down the curve shape.
    min_points: a cohort contributes only if it has at least this many ages.

    Returns None when fewer than two cohorts qualify -- a prior cannot be
    estimated from one cohort, and pretending otherwise is how a population
    prior silently becomes a single arbitrary curve.
    """
    fam = FAMILIES[family]
    fits: List[SeriesFit] = []
    labels: List[str] = []
    notes: List[str] = []

    for label, (t, y) in cohorts.items():
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        if int(ok.sum()) < max(min_points, fam.n_params):
            notes.append(f"{label}: only {int(ok.sum())} usable points; excluded.")
            continue
        f = fit_series(t[ok], y[ok], family)
        if f is None or not np.all(np.isfinite(f.theta)):
            notes.append(f"{label}: {family} could not be fitted; excluded.")
            continue
        if f.at_bounds:
            # A fit resting on its identifiability bound has not estimated that
            # parameter -- it has run out of room. Pooling it would contaminate
            # the population covariance with a number that carries no
            # information, so it is excluded and the exclusion is reported.
            notes.append(
                f"{label}: fitted parameter reached an identifiability bound "
                f"(the cohort was not followed long enough to determine the curve's "
                f"asymptote); excluded from the prior."
            )
            continue
        fits.append(f)
        labels.append(str(label))

    n_excluded = len(cohorts) - len(fits)
    if len(fits) < 2:
        return None

    p = fam.n_params
    thetas = np.vstack([f.theta for f in fits])
    Vs = [f.cov for f in fits]

    mu = thetas.mean(axis=0)
    # EM is initialised from the RAW covariance of the per-cohort estimates,
    # which OVERSTATES the between-cohort variance because it still contains
    # each cohort's own estimation error. EM then removes that error and shrinks
    # to the correct value.
    #
    # It is tempting to start from the moment-corrected value
    # (raw - mean within-cohort covariance) instead. Do not: when a parameter is
    # weakly identified its within-cohort variance can exceed the raw variance,
    # the correction goes negative, and clipping it to zero puts EM in an
    # ABSORBING STATE -- a near-zero Sigma implies total shrinkage, total
    # shrinkage implies zero apparent between-cohort spread, and Sigma stays at
    # zero. The symptom is a prior that claims all cohorts are identical and
    # prediction intervals that are far too narrow. Starting from an
    # overestimate has no such failure mode.
    raw = np.cov(thetas.T, bias=False) if len(fits) > p else np.diag(thetas.var(axis=0) + 1e-3)
    raw = np.atleast_2d(raw)
    if raw.shape != (p, p):
        raw = np.eye(p) * float(np.mean(np.var(thetas, axis=0)) + 1e-3)
    evals, evecs = np.linalg.eigh(0.5 * (raw + raw.T))
    evals = np.clip(evals, 1e-6, None)
    Sigma = evecs @ np.diag(evals) @ evecs.T + np.eye(p) * 1e-8

    # EM for (mu, Sigma) with known within-cohort covariances V_i
    for _ in range(max_em_iter):
        mu_old, Sigma_old = mu.copy(), Sigma.copy()
        Ws, bs = [], []
        for th, V in zip(thetas, Vs):
            try:
                Ci = np.linalg.inv(np.linalg.inv(Sigma) + np.linalg.inv(V + np.eye(p) * 1e-10))
            except np.linalg.LinAlgError:
                Ci = Sigma.copy()
            try:
                bi = Ci @ (np.linalg.inv(Sigma) @ mu
                           + np.linalg.inv(V + np.eye(p) * 1e-10) @ th)
            except np.linalg.LinAlgError:
                bi = th.copy()
            Ws.append(Ci)
            bs.append(bi)
        B = np.vstack(bs)
        mu = B.mean(axis=0)
        Sigma = np.zeros((p, p))
        for bi, Ci in zip(bs, Ws):
            d = (bi - mu).reshape(-1, 1)
            Sigma += d @ d.T + Ci
        Sigma /= len(bs)
        Sigma = 0.5 * (Sigma + Sigma.T) + np.eye(p) * 1e-10
        if (np.max(np.abs(mu - mu_old)) < tol
                and np.max(np.abs(Sigma - Sigma_old)) < tol):
            break

    resid_sd = float(np.median([f.resid_sd for f in fits]))
    if n_excluded:
        notes.append(
            f"{n_excluded} of {len(cohorts)} supplied cohorts were excluded (too few "
            "points, unfittable, or a parameter at its identifiability bound)."
        )
    if len(fits) < 4:
        notes.append(
            f"Prior estimated from only {len(fits)} cohorts; the between-cohort "
            "covariance is itself poorly determined and the prior should be treated "
            "as provisional."
        )

    return PopulationPrior(
        family=family,
        mean=np.asarray(mu, dtype=float),
        cov=np.asarray(Sigma, dtype=float),
        n_cohorts=len(fits),
        param_names=fam.param_names,
        residual_sd=resid_sd,
        cohort_labels=labels,
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def _regime(day: float, t_min: float, t_max: float) -> Tuple[str, float]:
    if t_min <= day <= t_max:
        return "interpolation", 0.0
    span = max(t_max - t_min, 1.0)
    if day > t_max:
        h = (day - t_max) / span
    else:
        h = (t_min - day) / span
    if h <= 0.5:
        return "near-extrapolation", h
    if h <= 1.5:
        return "far-extrapolation", h
    return "beyond-supported-horizon", h


def predict_trajectory(
    observed_days: Sequence[float],
    observed_values: Sequence[float],
    target_days: Sequence[float],
    prior: Optional[PopulationPrior] = None,
    families: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    n_draws: int = 4000,
    max_horizon: float = 2.0,
    seed: int = 0,
    value_range: Tuple[float, float] = (0.0, 1.0),
    calibrator: Optional["ConformalCalibrator"] = None,
) -> dict:
    """
    Predict phenotype values at `target_days` with 95% intervals.

    Two modes:

    WITHOUT a prior -- fits every feasible curve family, weights them by AICc,
      and draws from each fit's parameter covariance. The spread across
      families IS the model-form uncertainty and it is not hidden.

    WITH a prior -- restricts to the prior's family and forms the empirical-Bayes
      posterior for this cohort,
          C = (Sigma^-1 + V^-1)^-1,   m = C (Sigma^-1 mu + V^-1 theta_hat),
      so a sparse cohort is pulled toward the population shape and a dense one
      is not. This is what makes long-horizon prediction defensible.

    `max_horizon` refuses predictions further than that multiple of the observed
    span beyond the last observation (default 2x). Refusal is reported per
    target day rather than raising.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(observed_days, dtype=float)
    y = np.asarray(observed_values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t)
    t, y = t[order], y[order]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[ok][order]

    if len(t) < 2:
        raise ValueError("At least two observed timepoints are required.")

    tt = np.asarray(target_days, dtype=float)
    t_min, t_max = float(t.min()), float(t.max())

    lo_v, hi_v = value_range

    # ── build the fit set ───────────────────────────────────────────────────
    fits: List[SeriesFit] = []
    if prior is not None:
        f = fit_series(t, y, prior.family, weights=weights,
                       sigma_prior=prior.residual_sd)
        if f is None:
            raise RuntimeError(f"Prior family '{prior.family}' could not be fitted to these points.")
        # Empirical-Bayes posterior on the unconstrained scale.
        #
        # The prior covariance is inflated by 1/n_cohorts before use. A plug-in
        # empirical-Bayes interval treats the estimated (mu, Sigma) as if they
        # were known, which makes it systematically too narrow; inflating by
        # (1 + 1/n_cohorts) is the standard first-order correction for having
        # estimated the prior from a finite number of cohorts. It is a
        # correction, not a cure -- with very few cohorts the intervals remain
        # optimistic, which is why `build_population_prior` warns below four.
        p = len(prior.mean)
        infl = 1.0 + 1.0 / max(prior.n_cohorts, 1)
        Sig_inv = np.linalg.pinv(prior.cov * infl + np.eye(p) * 1e-12)
        V_inv = np.linalg.pinv(f.cov + np.eye(p) * 1e-12)
        C = np.linalg.pinv(Sig_inv + V_inv)
        m = C @ (Sig_inv @ prior.mean + V_inv @ f.theta)
        f = SeriesFit(family=prior.family, theta=m, cov=C,
                      resid_sd=max(f.resid_sd, prior.residual_sd),
                      aicc=f.aicc, weight=1.0, n_points=len(t), converged=f.converged,
                      at_bounds=f.at_bounds, resid_sd_source=f.resid_sd_source)
        fits = [f]
    else:
        fam_names = list(families) if families else list(FAMILIES.keys())
        for name in fam_names:
            f = fit_series(t, y, name, weights=weights)
            if f is not None and np.isfinite(f.aicc):
                fits.append(f)
        if not fits:
            raise RuntimeError("No curve family could be fitted to these points.")
        a = np.array([f.aicc for f in fits], dtype=float)
        w = np.exp(-0.5 * (a - a.min()))
        w = w / w.sum()
        for f, wi in zip(fits, w):
            f.weight = float(wi)

    # ── draw ────────────────────────────────────────────────────────────────
    probs = np.array([f.weight for f in fits], dtype=float)
    probs = probs / probs.sum()
    choice = rng.choice(len(fits), size=n_draws, p=probs)
    draws = np.empty((n_draws, len(tt)), dtype=float)

    # Parameter draws. Under a population prior the prior itself was estimated
    # from a finite number of cohorts, so a plug-in normal draw treats (mu,
    # Sigma) as known and produces intervals that are systematically too narrow
    # -- the classical empirical-Bayes under-coverage problem. Drawing from a
    # multivariate t with degrees of freedom set by the number of contributing
    # cohorts is the standard correction: it is the marginal distribution once
    # the estimated covariance is integrated out, and it restores the tail mass
    # that the plug-in normal discards. With many cohorts the t converges to the
    # normal and the correction costs nothing.
    t_df = None
    if prior is not None:
        t_df = max(int(prior.n_cohorts) - len(prior.mean), 3)

    for b in range(n_draws):
        f = fits[choice[b]]
        try:
            if t_df is not None:
                z = rng.multivariate_normal(np.zeros(len(f.theta)), f.cov, method="eigh")
                scale = np.sqrt(t_df / rng.chisquare(t_df))
                th = f.theta + z * scale
            else:
                th = rng.multivariate_normal(f.theta, f.cov, method="eigh")
        except Exception:
            th = f.theta
        vals = FAMILIES[f.family].predict(tt, th)
        draws[b] = vals + rng.normal(0.0, f.resid_sd, size=len(tt))

    draws = np.clip(draws, lo_v, hi_v)

    results = []
    for j, day in enumerate(tt):
        regime, horizon = _regime(float(day), t_min, t_max)
        refused = horizon > max_horizon
        col = draws[:, j]
        pred = np.nan if refused else float(np.median(col))
        lo95 = np.nan if refused else float(np.percentile(col, 2.5))
        hi95 = np.nan if refused else float(np.percentile(col, 97.5))
        lo50 = np.nan if refused else float(np.percentile(col, 25.0))
        hi50 = np.nan if refused else float(np.percentile(col, 75.0))
        calibrated = False

        if calibrator is not None and not refused:
            sd = (hi95 - lo95) / (2 * 1.959963985)
            if np.isfinite(sd) and sd > 0:
                c_lo, c_hi = calibrator.adjust(pred, sd, float(horizon), level="95")
                c50_lo, c50_hi = calibrator.adjust(pred, sd, float(horizon), level="50")
                if np.isfinite(c_lo) and np.isfinite(c_hi):
                    lo95, hi95 = np.clip([c_lo, c_hi], lo_v, hi_v)
                    calibrated = True
                if np.isfinite(c50_lo) and np.isfinite(c50_hi):
                    lo50, hi50 = np.clip([c50_lo, c50_hi], lo_v, hi_v)

        results.append({
            "day": float(day),
            "predicted": pred,
            "ci_lower": lo95,
            "ci_upper": hi95,
            "ci50_lower": lo50,
            "ci50_upper": hi50,
            "calibrated": calibrated,
            "regime": regime,
            "horizon_x_span": float(horizon),
            "refused": bool(refused),
            "refusal_reason": (
                f"Target is {horizon:.1f}x the observed span beyond the last observation; "
                f"the configured limit is {max_horizon:.1f}x. No prediction is issued."
            ) if refused else "",
        })

    return {
        "predictions": results,
        "draws": draws,
        "target_days": tt,
        "models": [
            {"family": f.family, "aicc": round(f.aicc, 3), "weight": round(f.weight, 4),
             "resid_sd": round(f.resid_sd, 5), **{k: round(v, 5) for k, v in f.natural().items()}}
            for f in sorted(fits, key=lambda z: -z.weight)
        ],
        "observed_range_days": [t_min, t_max],
        "n_observed": int(len(t)),
        "used_prior": prior is not None,
        "prior_cohorts": prior.n_cohorts if prior is not None else 0,
        "method": (
            "Empirical-Bayes posterior under a population prior over curve parameters"
            if prior is not None else
            "AICc-weighted model averaging across curve families (no population prior)"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. BASELINES -- what the model must beat
# ═══════════════════════════════════════════════════════════════════════════
def baseline_predictions(
    train_t: Sequence[float],
    train_y: Sequence[float],
    target_days: Sequence[float],
    population_curve: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Trivial predictors a trajectory model must outperform to be worth using.

      last_observation -- carry the final observed value forward
      train_mean       -- the mean of the training points, ignoring time
      linear_last_two  -- straight line through the last two observations
      population_curve -- the average trajectory for this genotype, ignoring
                          this cohort's own data (supplied by the caller)

    Reporting a forecaster without these is reporting an unfalsifiable claim.
    """
    t = np.asarray(train_t, dtype=float)
    y = np.asarray(train_y, dtype=float)
    tt = np.asarray(target_days, dtype=float)
    order = np.argsort(t)
    t, y = t[order], y[order]

    out: Dict[str, np.ndarray] = {
        "last_observation": np.full(len(tt), float(y[-1])),
        "train_mean": np.full(len(tt), float(np.mean(y))),
    }

    if len(t) >= 2 and (t[-1] - t[-2]) != 0:
        slope = (y[-1] - y[-2]) / (t[-1] - t[-2])
        out["linear_last_two"] = np.clip(y[-1] + slope * (tt - t[-1]), 0.0, 1.0)
    else:
        out["linear_last_two"] = out["last_observation"].copy()

    if population_curve is not None:
        try:
            out["population_curve"] = np.clip(np.asarray(population_curve(tt), dtype=float), 0.0, 1.0)
        except Exception:
            pass

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. FORWARD-IN-TIME VALIDATION AND CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════
def crps_from_samples(samples: np.ndarray, observation: float) -> float:
    """
    Continuous ranked probability score from predictive samples (Gneiting &
    Raftery 2007), using the kernel form

        CRPS = E|X - y| - 0.5 * E|X - X'|

    CRPS scores the whole predictive distribution, so a model cannot win by
    being accurate with absurd intervals, nor by hedging with huge ones. Lower
    is better; it reduces to absolute error for a point forecast.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(observation):
        return float("nan")
    term1 = float(np.mean(np.abs(x - observation)))
    # unbiased estimate of E|X - X'| via sorted-order formula (O(n log n))
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1)
    term2 = float(2.0 * np.sum((2 * i - n - 1) * xs) / (n * n))
    return term1 - 0.5 * term2


def forward_validate(
    days: Sequence[float],
    values: Sequence[float],
    train_max_day: float,
    prior: Optional[PopulationPrior] = None,
    population_curve: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: int = 0,
    calibrator: Optional["ConformalCalibrator"] = None,
    **predict_kwargs,
) -> dict:
    """
    Train on `days <= train_max_day`, predict every later day, and score both
    accuracy and calibration against the trivial baselines.

    This is the only honest way to validate a forecaster. A random hold-out
    across a cohort's own timepoints would let the model see later data while
    predicting earlier data.
    """
    d = np.asarray(days, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(d) & np.isfinite(v)
    d, v = d[ok], v[ok]
    order = np.argsort(d)
    d, v = d[order], v[order]

    train = d <= float(train_max_day)
    if int(train.sum()) < 2 or int((~train).sum()) < 1:
        raise ValueError("Need at least 2 training points and 1 held-out later point.")

    res = predict_trajectory(d[train], v[train], d[~train], prior=prior,
                             seed=seed, calibrator=calibrator, **predict_kwargs)
    draws = res["draws"]

    base = baseline_predictions(d[train], v[train], d[~train],
                                population_curve=population_curve)

    rows = []
    for j, (day, actual) in enumerate(zip(d[~train], v[~train])):
        p = res["predictions"][j]
        col = draws[:, j]
        row = {
            "day": float(day),
            "actual": float(actual),
            "predicted": p["predicted"],
            "abs_error": abs(p["predicted"] - actual) if np.isfinite(p["predicted"]) else np.nan,
            "in_50CI": bool(p["ci50_lower"] <= actual <= p["ci50_upper"]) if not p["refused"] else np.nan,
            "in_95CI": bool(p["ci_lower"] <= actual <= p["ci_upper"]) if not p["refused"] else np.nan,
            "crps": crps_from_samples(col, float(actual)) if not p["refused"] else np.nan,
            "regime": p["regime"],
            "horizon_x_span": p["horizon_x_span"],
            "refused": p["refused"],
        }
        for bname, bvals in base.items():
            row[f"abs_error_{bname}"] = abs(float(bvals[j]) - actual)
        rows.append(row)

    tab = pd.DataFrame(rows)
    scored = tab[~tab["refused"]]

    summary = {
        "n_held_out": int(len(tab)),
        "n_scored": int(len(scored)),
        "n_refused": int(tab["refused"].sum()),
        "MAE": float(scored["abs_error"].mean()) if len(scored) else np.nan,
        "RMSE": float(np.sqrt((scored["abs_error"] ** 2).mean())) if len(scored) else np.nan,
        "coverage_50": float(scored["in_50CI"].mean()) if len(scored) else np.nan,
        "coverage_95": float(scored["in_95CI"].mean()) if len(scored) else np.nan,
        "mean_CRPS": float(scored["crps"].mean()) if len(scored) else np.nan,
    }
    for bname in base:
        col = f"abs_error_{bname}"
        summary[f"MAE_{bname}"] = float(scored[col].mean()) if len(scored) else np.nan

    beaten = {
        b: bool(summary["MAE"] < summary[f"MAE_{b}"])
        for b in base
        if np.isfinite(summary.get(f"MAE_{b}", np.nan)) and np.isfinite(summary["MAE"])
    }
    summary["beats_all_baselines"] = bool(beaten) and all(beaten.values())
    summary["baselines_beaten"] = beaten

    return {
        "trained_on_days_upto": float(train_max_day),
        "per_horizon": tab,
        "summary": summary,
        "models": res["models"],
        "used_prior": res["used_prior"],
    }


@dataclass
class ConformalCalibrator:
    """
    Split-conformal correction for prediction intervals, learned from cohorts
    that were NOT used to build the population prior.

    WHY THIS IS NEEDED. The empirical-Bayes posterior treats the estimated
    population prior (mu, Sigma) as if it were known, and it shrinks each
    cohort's parameters toward the population mean. Both are good for accuracy
    and both are bad for interval coverage: shrinkage introduces bias for
    cohorts that are genuinely atypical, and the plug-in prior omits its own
    estimation error. In simulation the resulting nominal-95% intervals covered
    about 78% of the time. An interval labelled 95% that covers 78% is not a
    small imperfection -- every power calculation and go/no-go decision built on
    it is wrong.

    WHAT IT DOES. On held-out cohorts, forward-validate exactly as the model
    will be used, and record the standardised residual

        r = (actual - predicted) / sd_predictive

    at each forecast horizon. The empirical quantiles of r then replace the
    nominal normal quantiles. Because the residuals are SIGNED, the correction
    fixes bias and width together and produces asymmetric intervals when the
    model is systematically biased -- which, when extrapolating a shrunk curve,
    it is.

    Under exchangeability of the calibration and test cohorts, split-conformal
    intervals have finite-sample marginal coverage at the nominal level
    (Vovk, Gammerman & Shafer 2005; Lei et al. 2018, JASA 113:1094-1111).
    The guarantee is MARGINAL -- averaged over cohorts, not guaranteed for every
    individual cohort -- and it fails if the new cohort is not exchangeable with
    the calibration set, which is exactly the distribution-shift case
    (a prior learned on TDP-43 applied to alpha-synuclein). It is not a licence
    to stop checking.
    """

    horizon_edges: np.ndarray        # bucket boundaries in horizon-x-span units
    q_lower: np.ndarray              # per-bucket lower quantile of standardised residual
    q_upper: np.ndarray
    q50_lower: np.ndarray
    q50_upper: np.ndarray
    n_per_bucket: np.ndarray         # residuals per bucket (NOT the exchangeable unit)
    cohorts_per_bucket: np.ndarray   # distinct cohorts per bucket -- this is the unit
    n_calibration_cohorts: int
    notes: List[str] = field(default_factory=list)

    def _bucket(self, horizon: float) -> int:
        return int(np.clip(np.searchsorted(self.horizon_edges, horizon, side="right") - 1,
                           0, len(self.q_lower) - 1))

    def adjust(self, predicted: float, sd: float, horizon: float,
               level: str = "95") -> Tuple[float, float]:
        b = self._bucket(horizon)
        need = MIN_COHORTS_95 if level == "95" else MIN_COHORTS_50
        if self.cohorts_per_bucket[b] < need:
            return np.nan, np.nan          # too few INDEPENDENT cohorts to correct
        if level == "50":
            lo, hi = self.q50_lower[b], self.q50_upper[b]
        else:
            lo, hi = self.q_lower[b], self.q_upper[b]
        return predicted + lo * sd, predicted + hi * sd


def build_conformal_calibrator(
    prior: PopulationPrior,
    calibration_cohorts: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    train_max_day: float,
    horizon_edges: Sequence[float] = (0.0, 0.5, 1.0, 2.0, np.inf),
    seed: int = 0,
    **predict_kwargs,
) -> Optional[ConformalCalibrator]:
    """
    Learn the conformal correction from cohorts held out of the prior.

    `calibration_cohorts` MUST be disjoint from the cohorts used to build
    `prior`. If they overlap, the residuals are in-sample, the correction is too
    small, and the calibrated intervals inherit the same over-confidence they
    were meant to fix. This is checked against `prior.cohort_labels` and any
    overlap is refused.
    """
    overlap = set(map(str, calibration_cohorts)) & set(map(str, prior.cohort_labels))
    if overlap:
        raise ValueError(
            f"{len(overlap)} calibration cohort(s) were also used to build the prior "
            f"(e.g. {sorted(overlap)[:3]}). Split-conformal calibration requires disjoint "
            "sets; overlapping ones give in-sample residuals and an under-correction."
        )

    edges = np.asarray(horizon_edges, dtype=float)
    resid: List[List[float]] = [[] for _ in range(len(edges) - 1)]
    cohorts_in: List[set] = [set() for _ in range(len(edges) - 1)]

    for i, (label, (t, y)) in enumerate(calibration_cohorts.items()):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        train = t <= float(train_max_day)
        if int(train.sum()) < 2 or int((~train).sum()) < 1:
            continue
        try:
            res = predict_trajectory(t[train], y[train], t[~train], prior=prior,
                                     seed=seed + i, **predict_kwargs)
        except Exception:
            continue
        for j, p in enumerate(res["predictions"]):
            if p["refused"] or not np.isfinite(p["predicted"]):
                continue
            sd = (p["ci_upper"] - p["ci_lower"]) / (2 * 1.959963985)
            if not np.isfinite(sd) or sd <= 0:
                continue
            r = (float(y[~train][j]) - p["predicted"]) / sd
            b = int(np.clip(np.searchsorted(edges, p["horizon_x_span"], side="right") - 1,
                            0, len(resid) - 1))
            resid[b].append(r)
            cohorts_in[b].add(str(label))

    n_per = np.array([len(r) for r in resid], dtype=int)
    n_coh = np.array([len(s) for s in cohorts_in], dtype=int)
    # Gate on independent COHORTS, not on residuals -- see MIN_COHORTS_95 above.
    if n_coh.max(initial=0) < MIN_COHORTS_50:
        return None

    z95, z50 = 1.959963985, 0.6744897502

    def _q(rs: List[float], q: float, fallback: float, need: int, b: int) -> float:
        return float(np.quantile(rs, q)) if n_coh[b] >= need else fallback

    cal = ConformalCalibrator(
        horizon_edges=edges,
        q_lower=np.array([_q(r, 0.025, -z95, MIN_COHORTS_95, b)
                          for b, r in enumerate(resid)]),
        q_upper=np.array([_q(r, 0.975, z95, MIN_COHORTS_95, b)
                          for b, r in enumerate(resid)]),
        q50_lower=np.array([_q(r, 0.25, -z50, MIN_COHORTS_50, b)
                            for b, r in enumerate(resid)]),
        q50_upper=np.array([_q(r, 0.75, z50, MIN_COHORTS_50, b)
                            for b, r in enumerate(resid)]),
        n_per_bucket=n_per,
        cohorts_per_bucket=n_coh,
        n_calibration_cohorts=len(calibration_cohorts),
    )
    thin95 = [i for i, c in enumerate(n_coh) if 0 < c < MIN_COHORTS_95]
    if thin95:
        cal.notes.append(
            f"Horizon buckets {thin95} were calibrated on fewer than {MIN_COHORTS_95} "
            f"independent cohorts (residual counts {[int(n_per[i]) for i in thin95]}, "
            f"cohort counts {[int(n_coh[i]) for i in thin95]}). The 95% interval is "
            "REFUSED there rather than corrected: a quantile estimated from a few "
            "cohorts is worse than no correction. Residuals are correlated within a "
            "cohort, so the residual count overstates the information available."
        )
    return cal


def calibration_report(coverage_observed: Sequence[float],
                       nominal: Sequence[float] = (0.5, 0.95)) -> pd.DataFrame:
    """
    Tabulate nominal versus empirical interval coverage.

    A model whose 95% interval covers 60% of the time is not 'slightly
    optimistic'; every downstream sample-size and go/no-go decision built on it
    is wrong. This table is the check.
    """
    rows = []
    for nom, obs in zip(nominal, coverage_observed):
        rows.append({
            "nominal_coverage": float(nom),
            "empirical_coverage": float(obs),
            "difference": float(obs - nom),
            "verdict": (
                "well calibrated" if abs(obs - nom) <= 0.05 else
                "over-confident (intervals too narrow)" if obs < nom else
                "under-confident (intervals too wide)"
            ),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 7. DECISION-RELEVANT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
def derive_endpoints(
    prediction: dict,
    threshold: float = 0.5,
    comparator_draws: Optional[np.ndarray] = None,
) -> dict:
    """
    Convert a predicted trajectory into quantities an experimenter acts on.

      onset_age  -- first predicted age at which the phenotype falls below
                    `threshold` (impairment onset)
      healthspan -- number of predicted days spent at or above `threshold`
                    within the prediction window
      days_gained-- difference in onset age versus a comparator cohort, when
                    `comparator_draws` (same shape) is supplied

    Every quantity is computed WITHIN each posterior draw and then summarised,
    so the reported interval propagates trajectory uncertainty rather than
    being read off the median curve.
    """
    draws = prediction["draws"]
    days = np.asarray(prediction["target_days"], dtype=float)
    if draws.size == 0:
        return {"available": False, "reason": "No predictive draws."}

    order = np.argsort(days)
    days = days[order]
    draws = draws[:, order]

    def _onset(row: np.ndarray) -> float:
        below = np.where(row < threshold)[0]
        if below.size == 0:
            return np.nan                      # never crosses within the window
        i = int(below[0])
        if i == 0:
            return float(days[0])
        # linear interpolation between the bracketing days
        y0, y1 = row[i - 1], row[i]
        d0, d1 = days[i - 1], days[i]
        if y0 == y1:
            return float(d1)
        return float(d0 + (y0 - threshold) * (d1 - d0) / (y0 - y1))

    onsets = np.array([_onset(r) for r in draws], dtype=float)
    span = float(days.max() - days.min()) if len(days) > 1 else 0.0
    healthspans = np.array([
        float(np.trapezoid((r >= threshold).astype(float), days)) if len(days) > 1 else np.nan
        for r in draws
    ], dtype=float)

    def _summarise(x: np.ndarray, name: str) -> dict:
        finite = x[np.isfinite(x)]
        frac = float(finite.size) / float(x.size) if x.size else 0.0
        if finite.size < max(20, 0.05 * x.size):
            return {
                f"{name}": np.nan, f"{name}_ci_lower": np.nan, f"{name}_ci_upper": np.nan,
                f"{name}_defined_fraction": frac,
                f"{name}_note": (
                    f"The phenotype stays at or above the threshold of {threshold:g} in "
                    f"{(1 - frac) * 100:.0f}% of draws within the predicted window, so "
                    f"{name} is undefined there. Extend the window or lower the threshold."
                ),
            }
        return {
            f"{name}": float(np.median(finite)),
            f"{name}_ci_lower": float(np.percentile(finite, 2.5)),
            f"{name}_ci_upper": float(np.percentile(finite, 97.5)),
            f"{name}_defined_fraction": frac,
            f"{name}_note": "",
        }

    out: dict = {"available": True, "threshold": float(threshold),
                 "window_days": [float(days.min()), float(days.max())]}
    out.update(_summarise(onsets, "onset_age_days"))
    out.update(_summarise(healthspans, "healthspan_days"))

    if comparator_draws is not None:
        comp = np.asarray(comparator_draws, dtype=float)
        if comp.shape[1] == draws.shape[1]:
            comp = comp[:, order] if comp.shape[1] == len(order) else comp
            n = min(comp.shape[0], draws.shape[0])
            comp_onsets = np.array([_onset(r) for r in comp[:n]], dtype=float)
            diff = onsets[:n] - comp_onsets
            out.update(_summarise(diff, "days_gained_vs_comparator"))
        else:
            out["days_gained_note"] = (
                "Comparator draws have a different number of target days; days gained "
                "was not computed."
            )

    return out
