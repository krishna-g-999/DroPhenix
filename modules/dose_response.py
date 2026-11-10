"""
Toxin dose-response analysis for Drosophila behavioural and survival assays.

THE PROBLEM THIS MODULE FIXES
-----------------------------
A toxin screen produces a dose column and a response column, and the
conventional analysis reduces it to "significant at 5 mM, not at 1 mM" by
running a one-way ANOVA with post-hoc tests across dose groups. That throws
away the two things a dose-response experiment exists to estimate -- the
POTENCY (the dose at which the effect is half-maximal) and the SHAPE (how
steeply the effect turns on) -- and replaces them with an artefact of where
the doses happened to be placed. Three specific failures follow.

1. NO POTENCY ESTIMATE. ANOVA cannot say what EC50 is, so two compounds can
   only be ranked by "lowest dose with a star", which depends entirely on the
   dose spacing the experimenter chose. A curve fit gives a number with an
   interval that is comparable across experiments.

2. WRONG INTERVALS FOR EC50. EC50 enters the log-logistic model non-linearly,
   and its sampling distribution is strongly skewed on the dose scale. The
   delta method (estimate +/- 1.96 * SE) assumes symmetry and routinely
   produces intervals that are too narrow on the upper side and that can
   extend below zero -- an impossible dose. This module computes a PROFILE
   LIKELIHOOD interval, which respects the actual geometry of the residual
   sum of squares surface, and reports the delta-method interval alongside so
   the discrepancy is visible rather than hidden.

3. LETHALITY MASQUERADING AS BEHAVIOUR. This is the one that invalidates
   conclusions. Climbing is scored among flies that are alive to climb. A
   toxin that kills flies removes them from the denominator, so the surviving
   sample at a high dose is not a random sample of the exposed population --
   it is the subset that withstood the dose. "The toxin impairs climbing" and
   "the toxin was killing the flies and the ones left were moribund" produce
   the same data. `mortality_adjusted_response` fits both curves and reports
   EC50_behaviour / LC50_mortality: when that ratio approaches 1, the
   behavioural effect is not separable from general lethality and must not be
   reported as a behavioural phenotype.

Additionally, non-monotonic LOW-DOSE STIMULATION (hormesis) is common in
toxicology -- flies at a low dose out-perform controls. A monotone Hill fit
cannot represent it and averages it away, biasing EC50 upward.
`hormesis_test` fits the Brain-Cousens extension and tests for it explicitly.

WHAT THIS MODULE PROVIDES
-------------------------
  check_design              -- refuses designs that cannot identify the model
                               and warns about ones that can but badly
  fit_loglogistic           -- 4-, 3- and 2-parameter log-logistic (Hill)
                               models by (weighted) non-linear least squares
  ec50_profile_ci           -- profile-likelihood interval for EC50, with the
                               delta-method interval reported for comparison
  effective_dose            -- ECx for arbitrary x, with profile and delta
                               intervals
  hormesis_test             -- Brain-Cousens model vs monotone LL.4, F-test
  relative_potency          -- EC50 ratio between two fits, with a parallelism
                               check (the ratio is only dose-independent if
                               the slopes agree)
  mortality_adjusted_response -- the headline diagnostic: behavioural curve,
                               lethality curve, and the overlap between them

THE MODEL AND ITS PARAMETERISATION
----------------------------------
The four-parameter log-logistic (equivalently the Hill equation) is

    f(x) = c + (d - c) / (1 + exp(b * (log(x) - log(e))))

  e  EC50  -- the dose at which f is halfway between c and d
  b  Hill slope. With b > 0 the curve DECREASES in dose.
  d  the response as x -> 0+  (the control / upper asymptote when b > 0)
  c  the response as x -> inf (the lower asymptote when b > 0)

Note that `c` and `d` are defined by the LIMITS, not by which is numerically
larger. An increasing dose-response is represented with d < c and b > 0 -- it
is the same model, not a different one, and no sign flag is needed.

  LL.4  all of b, c, d, e free
  LL.3  c fixed at 0        (the response genuinely bottoms out at zero)
  LL.2  c fixed at 0, d at 1 (a proportion that starts at 1 and falls to 0 --
                              the natural model for a survival curve, where e
                              is then exactly the LC50)

HOW ZERO DOSE (THE UNTREATED CONTROL) IS HANDLED
------------------------------------------------
log(0) is undefined, so the model expression cannot be evaluated at x = 0.
Dropping the controls would be wrong: they are the observations that pin down
the upper asymptote, and without them `d` is extrapolated.

This module therefore evaluates the model at x = 0 by its LIMIT, which exists
and is finite:

    b > 0:  b*(log x - log e) -> -inf,  exp(...) -> 0,  f(0) = d
    b < 0:  b*(log x - log e) -> +inf,  exp(...) -> inf, f(0) = c
    b = 0:  the curve is flat at (c + d)/2

So the control observations enter the fit as direct observations of the
asymptote (`d` in the usual decreasing case) and contribute their residuals,
their degrees of freedom and their leverage exactly like any other dose --
they simply do not pass through the log term. This is the same treatment used
by the R package `drc`, and `validation/validate_dose_response.py` confirms
that it reproduces `drc`'s published estimates on the `ryegrass` data to five
decimal places.

The zero doses are still excluded from the SPACING diagnostics in
`check_design` (a control is not a dose level for the purpose of counting how
many points define the curve), and from the geometric-mean starting value.

STANDARD ERRORS
---------------
Standard errors come from the Gauss-Newton approximation to the covariance,

    Cov(theta) = s^2 * (J' J)^-1,     s^2 = RSS / (n - p)

with J the Jacobian of the residuals at the optimum. EC50 is estimated on the
LOG scale internally (which is what makes the optimisation unconstrained and
keeps the dose positive), and its standard error on the dose scale is obtained
by the delta method, SE(e) = e * SE(log e). This is exactly the quantity whose
unreliability motivates `ec50_profile_ci`; it is reported for comparison, not
as the recommended interval.

References
----------
Ritz C, Baty F, Streibig JC, Gerhard D (2015) Dose-response analysis using R.
  PLoS ONE 10(12):e0146021.  [the `drc` package]
Ritz C (2010) Toward a unified approach to dose-response modeling in ecotoxicology.
  Environmental Toxicology and Chemistry 29:220-229.
Brain P, Cousens R (1989) An equation to describe dose responses where there is
  stimulation of growth at low doses. Weed Research 29:93-96.  [hormesis]
Calabrese EJ, Baldwin LA (2003) Hormesis: the dose-response revolution.
  Annual Review of Pharmacology and Toxicology 43:175-197.
Bates DM, Watts DG (1988) Nonlinear Regression Analysis and Its Applications.
  Wiley.  [profile likelihood / profile-t intervals for nonlinear models]
Seber GAF, Wild CJ (1989) Nonlinear Regression. Wiley, ch. 5.
  [F-based inversion of the profile sum-of-squares]
Finney DJ (1971) Probit Analysis, 3rd ed. Cambridge University Press.
  [relative potency and the parallelism requirement]
Abbott WS (1925) A method of computing the effectiveness of an insecticide.
  Journal of Economic Entomology 18:265-267.  [control mortality correction]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln

# Models supported, and which of (b, c, d, log_e) are free.
_MODEL_SPECS: Dict[str, dict] = {
    "LL.4": {"free": ("b", "c", "d", "log_e"), "fixed": {}},
    "LL.3": {"free": ("b", "d", "log_e"), "fixed": {"c": 0.0}},
    "LL.2": {"free": ("b", "log_e"), "fixed": {"c": 0.0, "d": 1.0}},
}

_MAX_EXP = 700.0     # exp() overflows a float64 above ~709


def _safe_exp(v: float) -> float:
    """exp() that saturates to 0 / inf instead of raising an overflow warning."""
    if not np.isfinite(v):
        return float(v)
    if v > _MAX_EXP:
        return float("inf")
    if v < -_MAX_EXP:
        return 0.0
    return float(np.exp(v))


# ═══════════════════════════════════════════════════════════════════════════
# 0. THE MODEL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def loglogistic(x, b: float, c: float, d: float, e: float) -> np.ndarray:
    """
    Four-parameter log-logistic (Hill) function.

        f(x) = c + (d - c) / (1 + exp(b * (log(x) - log(e))))

    Zero doses are evaluated by the limit (d if b > 0, c if b < 0, the midpoint
    if b == 0) -- see the module docstring for why. Negative doses raise.
    """
    return _ll_predict(np.asarray(x, dtype=float), b, c, d, float(np.log(e)))


def _ll_predict(x: np.ndarray, b: float, c: float, d: float, log_e: float) -> np.ndarray:
    """Model evaluated with EC50 on the log scale (the internal parameterisation)."""
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("Negative doses are not meaningful in a log-logistic model.")
    out = np.empty_like(x, dtype=float)

    pos = x > 0
    if np.any(pos):
        z = b * (np.log(x[pos]) - log_e)
        z = np.clip(z, -_MAX_EXP, _MAX_EXP)
        out[pos] = c + (d - c) / (1.0 + np.exp(z))

    zero = ~pos
    if np.any(zero):
        if b > 0:
            out[zero] = d
        elif b < 0:
            out[zero] = c
        else:
            out[zero] = 0.5 * (c + d)
    return out


def _brain_cousens(x: np.ndarray, b: float, c: float, d: float, log_e: float,
                   f: float) -> np.ndarray:
    """
    Brain-Cousens hormesis model (Brain & Cousens 1989):

        g(x) = c + (d - c + f * x) / (1 + exp(b * (log(x) - log(e))))

    f > 0 produces a hump above the control level at low dose. At x = 0 the
    numerator term f*x vanishes and the limit is the same as for LL.4, so the
    controls are handled identically.

    Note that when f != 0, `e` is NO LONGER the EC50 -- it is a location
    parameter only. `hormesis_test` therefore reports the EC50 of the monotone
    fit for potency and uses the Brain-Cousens fit only for the hormesis test
    and for locating the stimulation maximum.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x > 0
    if np.any(pos):
        z = b * (np.log(x[pos]) - log_e)
        z = np.clip(z, -_MAX_EXP, _MAX_EXP)
        out[pos] = c + (d - c + f * x[pos]) / (1.0 + np.exp(z))
    zero = ~pos
    if np.any(zero):
        if b > 0:
            out[zero] = d
        elif b < 0:
            out[zero] = c
        else:
            out[zero] = 0.5 * (c + d)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. DESIGN CHECKS -- refuse what cannot be identified
# ═══════════════════════════════════════════════════════════════════════════
def check_design(dose: Sequence[float], response: Sequence[float],
                 model: str = "LL.4") -> dict:
    """
    Decide whether the design can support the requested dose-response model,
    and say so BEFORE a number is produced.

    A non-linear model with p parameters needs at least p distinct points on
    the curve to be identified. The controls (dose 0) inform the upper
    asymptote but they are all one point on the curve, so an LL.4 fit needs
    FOUR distinct non-zero doses; three gives a saturated fit with zero
    residual degrees of freedom for the curve and standard errors that are
    fiction. That is refused, not warned about.

    Refusals (`usable` is False)
      * fewer than `p` distinct non-zero doses for a p-parameter model
      * fewer observations than parameters
      * no variation in the response

    Warnings (`usable` stays True)
      * fewer than 5 distinct non-zero doses -- identified, but EC50 intervals
        from so few points are unreliable
      * the doses do not straddle the INFLECTION of the curve, so EC50 is
        EXTRAPOLATED beyond the tested range rather than interpolated. The
        log-logistic attains its steepest slope in log-dose exactly at EC50,
        so this is detected without fitting anything: the per-interval slopes
        of the dose-group means against log(dose) are computed, and if the
        steepest one is the FIRST or the LAST interval (by a clear margin over
        the best interior interval) then the response was still accelerating
        at the edge of the design and the turning point lies outside it.
      * no replication at any dose, so there is no pure-error estimate and
        lack of fit cannot be distinguished from noise
      * the upper or lower plateau is not defined by at least two dose levels,
        so that asymptote is extrapolated
      * fewer than ~4 doses per decade or a dose range spanning less than one
        decade -- log-logistic curves need logarithmically spaced doses

    Returns a dict with `usable`, `refusals`, `warnings` and the diagnostics
    the decisions were based on.
    """
    x = np.asarray(dose, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"dose and response have different lengths: {x.shape} vs {y.shape}")

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    spec = _MODEL_SPECS.get(model)
    if spec is None:
        raise ValueError(f"model must be one of {sorted(_MODEL_SPECS)}, got {model!r}")
    p = len(spec["free"])

    refusals: List[str] = []
    warnings: List[str] = []

    if x.size == 0:
        return {"usable": False, "refusals": ["No finite (dose, response) pairs."],
                "warnings": [], "model": model, "n_obs": 0}
    if np.any(x < 0):
        refusals.append("Negative doses present; a log-logistic model is undefined for them.")

    nonzero = np.unique(x[x > 0])
    n_levels = int(nonzero.size)
    n_zero = int(np.sum(x == 0))

    if n_levels < p:
        refusals.append(
            f"{n_levels} distinct non-zero dose(s) but {model} has {p} parameters. "
            f"A {p}-parameter curve cannot be identified from {n_levels} point(s) on it "
            f"-- this is a design limitation, not something the fit can repair. "
            f"Use at least {p} non-zero doses (5+ recommended), or fit a simpler model."
        )
    if x.size < p + 1:
        refusals.append(
            f"{x.size} observations for {p} parameters leaves "
            f"{max(x.size - p, 0)} residual degrees of freedom; no error variance "
            "can be estimated."
        )
    if np.ptp(y) <= 0:
        refusals.append("The response is constant; there is no dose-response to fit.")

    # ── warnings ────────────────────────────────────────────────────────────
    if p <= n_levels < 5:
        warnings.append(
            f"Only {n_levels} distinct non-zero doses. The model is identified but "
            "EC50 confidence intervals from this few points are wide and unreliable; "
            "5-8 log-spaced doses is the usual recommendation."
        )
    if n_zero == 0:
        warnings.append(
            "No untreated control (dose 0). The upper asymptote is then estimated "
            "purely by extrapolation from the lowest tested dose."
        )

    # Do the doses straddle the inflection point (= EC50) of the curve?
    dose_means, extrapolated = {}, None
    if x.size and np.ptp(y) > 0:
        df = pd.DataFrame({"x": x, "y": y})
        gm = df.groupby("x")["y"].agg(["mean", "size"])
        dose_means = {float(k): float(v) for k, v in gm["mean"].items()}
        top, bot = float(gm["mean"].max()), float(gm["mean"].min())
        nz = gm.loc[gm.index > 0].sort_index()

        # The log-logistic is steepest in log-dose exactly at EC50. So if the
        # steepest observed segment is at either END of the dose ladder, the
        # response was still accelerating when the design ran out and EC50 is
        # outside the tested range. A margin is required over the best
        # interior segment so that a flat maximum (the well-designed case,
        # where several central segments are near-equally steep) is not
        # mistaken for an edge peak.
        if len(nz) >= 4:
            lx = np.log(nz.index.to_numpy(dtype=float))
            mv = nz["mean"].to_numpy(dtype=float)
            dlx = np.diff(lx)
            slopes = np.abs(np.diff(mv) / np.where(dlx == 0, np.nan, dlx))
            k = slopes.size
            interior = slopes[1:k - 1]
            best_interior = float(np.nanmax(interior)) if interior.size else 0.0
            i_max = int(np.nanargmax(slopes))
            margin = 1.2
            if i_max == 0 and slopes[0] > margin * best_interior:
                extrapolated = True
                side = "BELOW the lowest non-zero dose tested"
            elif i_max == k - 1 and slopes[-1] > margin * best_interior:
                extrapolated = True
                side = "ABOVE the highest dose tested"
            else:
                extrapolated = False
                side = ""
            if extrapolated:
                warnings.append(
                    f"The response is still steepening at the edge of the dose range, "
                    f"so the curve's inflection -- which IS the EC50 -- lies {side}. "
                    "EC50 is therefore EXTRAPOLATED rather than interpolated, and its "
                    "value and interval are driven by the assumed functional form "
                    "rather than by data."
                )
        # plateau definition
        if len(nz) >= 2:
            rng = top - bot
            if rng > 0:
                srt = nz.sort_index()
                n_top = int(np.sum(srt["mean"] >= top - 0.10 * rng))
                n_bot = int(np.sum(srt["mean"] <= bot + 0.10 * rng))
                if n_top + (1 if n_zero else 0) < 2:
                    warnings.append(
                        "Fewer than two dose levels sit in the top 10% of the response "
                        "range; the upper asymptote d is extrapolated."
                    )
                if n_bot < 2:
                    warnings.append(
                        "Fewer than two dose levels sit in the bottom 10% of the "
                        "response range; the lower asymptote c is extrapolated. "
                        "Consider LL.3 (c fixed at 0) if the response truly bottoms at zero."
                    )
        max_rep = int(gm["size"].max()) if len(gm) else 0
        if max_rep < 2:
            warnings.append(
                "No dose is replicated. Residual variance is then confounded with "
                "lack of fit and the model cannot be checked against pure error."
            )

    if n_levels >= 2:
        decades = float(np.log10(nonzero.max() / nonzero.min()))
        if decades < 1.0:
            warnings.append(
                f"The non-zero doses span only {decades:.2f} orders of magnitude. "
                "Log-logistic curves are estimated on the log-dose scale and need a "
                "range of at least one decade, ideally two."
            )
        elif n_levels / max(decades, 1e-9) < 3.0:
            warnings.append(
                f"{n_levels} doses over {decades:.2f} decades is sparse "
                f"({n_levels / decades:.1f} per decade); 4 or more per decade gives a "
                "better-determined slope."
            )
    else:
        decades = float("nan")

    return {
        "usable": len(refusals) == 0,
        "refusals": refusals,
        "warnings": warnings,
        "model": model,
        "n_params": p,
        "n_obs": int(x.size),
        "n_distinct_nonzero_doses": n_levels,
        "n_control_obs": n_zero,
        "dose_decades": decades,
        "dose_means": dose_means,
        "ec50_extrapolated": extrapolated,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. THE FIT
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class LogLogisticFit:
    """
    Result of a (weighted) non-linear least-squares log-logistic fit.

    `params` holds the natural-scale parameters b, c, d, e (fixed ones
    included, with a standard error of 0). `theta` and `cov_theta` are on the
    INTERNAL scale, in which EC50 appears as log(e) -- that is the scale the
    profile and delta calculations work on, because it is the one on which the
    optimisation is unconstrained.
    """

    model: str
    params: pd.Series                    # b, c, d, e on the natural scale
    bse: pd.Series                       # standard errors, matching index
    theta: np.ndarray                    # free parameters, EC50 as log(e)
    theta_names: List[str]
    cov_theta: np.ndarray
    rss: float
    resid_se: float
    df_resid: int
    n_obs: int
    n_params: int
    dose: np.ndarray
    response: np.ndarray
    weights: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    converged: bool
    design: dict = field(default_factory=dict)
    message: str = ""

    # ── binomial maximum likelihood (register N95) ─────────────────────────
    # `method` is "ls" for the least-squares path, which is the default and is
    # unchanged, or "binomial_ml". The fields below are only populated for the
    # latter; they default so that every existing construction of this
    # dataclass still works.
    method: str = "ls"
    counts: Optional[np.ndarray] = None       # successes per dose, k
    totals: Optional[np.ndarray] = None       # cohort size per dose, n
    loglik: float = float("nan")              # maximised binomial log-likelihood
    loglik_penalised: float = float("nan")    # the objective actually maximised
    penalty: str = "none"                     # "firth" or "none"
    penalty_rule: str = ""                    # which rule chose it
    slope_cap: float = float("nan")           # resolvable-slope bound, if applied
    separation: bool = False                  # the unpenalised MLE ran away
    separation_note: str = ""
    inference_valid: bool = True
    hessian_min_eigenvalue: float = float("nan")
    inference_reason: str = ""

    # ── convenience ────────────────────────────────────────────────────────
    @property
    def ec50(self) -> float:
        return float(self.params["e"])

    @property
    def log_e_index(self) -> int:
        return self.theta_names.index("log_e")

    @property
    def se_log_e(self) -> float:
        i = self.log_e_index
        v = float(self.cov_theta[i, i])
        return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan")

    def predict(self, x) -> np.ndarray:
        """Fitted response at arbitrary doses, including x = 0."""
        p = self.params
        return _ll_predict(np.asarray(x, dtype=float), float(p["b"]), float(p["c"]),
                           float(p["d"]), float(np.log(p["e"])))

    def summary_frame(self) -> pd.DataFrame:
        crit = stats.t.ppf(0.975, self.df_resid) if self.df_resid > 0 else np.nan
        out = pd.DataFrame({"estimate": self.params, "std_err": self.bse})
        out["t"] = out["estimate"] / out["std_err"].replace(0.0, np.nan)
        out["ci_lower"] = out["estimate"] - crit * out["std_err"]
        out["ci_upper"] = out["estimate"] + crit * out["std_err"]
        out.loc[out["std_err"] == 0, ["t", "ci_lower", "ci_upper"]] = np.nan
        return out

    def r_squared(self) -> float:
        """
        Proportion of variance explained. Reported for description only: a
        high R^2 does not validate a non-linear model and does not mean the
        EC50 is well determined.
        """
        w = self.weights
        ybar = float(np.sum(w * self.response) / np.sum(w))
        sst = float(np.sum(w * (self.response - ybar) ** 2))
        return float(1.0 - self.rss / sst) if sst > 0 else float("nan")


def _unpack(theta: np.ndarray, model: str) -> Tuple[float, float, float, float]:
    """Map free parameters onto the full (b, c, d, log_e) tuple."""
    spec = _MODEL_SPECS[model]
    vals = dict(spec["fixed"])
    for name, v in zip(spec["free"], theta):
        vals[name] = float(v)
    return vals["b"], vals["c"], vals["d"], vals["log_e"]


def _start_values(x: np.ndarray, y: np.ndarray, model: str) -> List[np.ndarray]:
    """
    Starting values, and why they are what they are.

    `d` is the response as dose -> 0, so the mean at the LOWEST dose is the
    natural start; `c` is the response as dose -> inf, so the mean at the
    HIGHEST dose. Defining them by the limits rather than by which is larger
    means an increasing dose-response needs no special case: it simply has
    d < c, with b still positive.

    The slope is started from the logit linearisation of the model. For any
    observation strictly between the two asymptotes,

        u = (d0 - y) / (y - c0),      log(u) = b * (log x - log e)

    so an OLS regression of log(u) on log(x) gives a slope estimate for b and
    an intercept that gives log(e). Several fallbacks are returned as well,
    because a bad slope start is the usual cause of a failed Hill fit.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    lo_mask = xs == xs.min()
    hi_mask = xs == xs.max()
    d0 = float(np.mean(ys[lo_mask]))
    c0 = float(np.mean(ys[hi_mask]))
    if d0 == c0:                      # degenerate; nudge so the linearisation works
        c0 = d0 - (np.ptp(ys) or 1.0)

    nz = x[x > 0]
    e_geo = float(np.exp(np.mean(np.log(nz)))) if nz.size else 1.0

    # dose whose mean response is closest to the midpoint
    mid = 0.5 * (c0 + d0)
    e_mid = e_geo
    if nz.size:
        gm = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].mean()
        gm = gm.loc[gm.index > 0]
        if len(gm):
            e_mid = float(gm.index[np.argmin(np.abs(gm.to_numpy() - mid))])

    b_lin, e_lin = 1.0, e_mid
    inner = (x > 0) & (((y - c0) * (d0 - y)) > 0)
    if int(np.sum(inner)) >= 2:
        u = (d0 - y[inner]) / (y[inner] - c0)
        lu, lx = np.log(u), np.log(x[inner])
        if np.ptp(lx) > 0:
            slope, intercept = np.polyfit(lx, lu, 1)
            if np.isfinite(slope) and abs(slope) > 1e-6:
                b_lin = float(slope)
                cand = float(np.exp(-intercept / slope))
                if np.isfinite(cand) and cand > 0:
                    e_lin = cand

    starts = []
    for b0, ee in ((b_lin, e_lin), (b_lin, e_mid), (1.0, e_mid), (3.0, e_mid),
                   (0.5, e_geo), (2.0, e_geo)):
        full = {"b": b0, "c": c0, "d": d0, "log_e": float(np.log(max(ee, 1e-12)))}
        starts.append(np.array([full[k] for k in _MODEL_SPECS[model]["free"]], dtype=float))
    return starts


def _lsq(fun, x0):
    """Least squares with 'lm' where possible, falling back to the trust region."""
    try:
        return optimize.least_squares(fun, x0, method="lm", xtol=1e-15, ftol=1e-15,
                                      gtol=1e-15, max_nfev=20000)
    except Exception:
        return optimize.least_squares(fun, x0, method="trf", x_scale="jac",
                                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)



# ═══════════════════════════════════════════════════════════════════════════
# Binomial maximum likelihood (register N95)
# ═══════════════════════════════════════════════════════════════════════════
# Why this exists. Weighted least squares with `n` as the weight reproduces the
# 1/n of a binomial variance and nothing reproduces the p(1-p). Measured
# consequence: the nominal 95% interval on an absolute LC50 covered 0.47 at a
# Hill slope of 4 and did not improve when the design was enriched from 7 doses
# x 40 flies to 20 x 300, while being nominal under homoscedastic Gaussian
# noise at every slope. The variance model was wrong, so more data could not
# help.
#
# The remedy is to fit the likelihood the data actually have. For k_i successes
# out of n_i at dose x_i,
#
#     loglik = SUM_i [ log C(n_i, k_i) + k_i log p_i + (n_i - k_i) log(1 - p_i) ]
#
# with p_i the log-logistic mean. Two consequences follow for free, and both
# matter more than the point estimate:
#
#   * the covariance comes from the OBSERVED INFORMATION -- the Hessian of the
#     negative log-likelihood at the optimum -- which carries the p(1-p)
#     weighting automatically, because it is derived from the model that
#     generated the data rather than imposed on it;
#   * the interval can be a PROFILE on the LC50 itself (see
#     `lethal_dose_profile`), which is invariant to how the curve is
#     parameterised. That invariance is exactly what the delta method lacks:
#     a symmetric interval on log-dose cannot represent a likelihood that is
#     not symmetric there, and at steep slopes it is very much not.
#
# Validated in `validate_binomial_ml`: for LL.2 the model is algebraically a
# logistic regression on log-dose, so `statsmodels`' binomial GLM maximises the
# SAME likelihood and the two must agree to optimiser tolerance. They agree on
# b, on e, on the maximised log-likelihood and on SE(b).
#
# c and d are bounded into [0, 1] here, which is not a numerical convenience: a
# binomial mean outside [0, 1] is not a probability, and an unbounded fit can
# wander there and return a likelihood that is not defined.
_P_EPS = 1e-10
_CD_EPS = 1e-9


def _counts_from_proportion(y: np.ndarray, n: np.ndarray,
                            tol: float = 1e-6) -> np.ndarray:
    """
    Recover the success counts k from a proportion and a cohort size.

    Refuses rather than rounds. `y * n` that is not a whole number means the
    response is not a proportion out of `n` -- most often `n` is the wrong
    column, or the response has already been averaged over replicates, and in
    both cases a binomial likelihood is the wrong model. Rounding would hide
    that and produce a confident fit to a quantity the model does not describe.
    """
    k = np.asarray(y, dtype=float) * np.asarray(n, dtype=float)
    nearest = np.round(k)
    bad = np.abs(k - nearest) > tol * np.maximum(1.0, np.abs(k))
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(
            "method='binomial_ml' needs whole-number counts: response * n must be "
            f"an integer for every dose, and at index {i} it is {k[i]!r} "
            f"(response {float(y[i])!r} x n {float(n[i])!r}). A binomial "
            "likelihood describes k successes out of n trials; a response that "
            "has already been averaged over replicates, or an `n` from the wrong "
            "column, is not that. The counts are NOT rounded, because rounding "
            "would hide the mismatch behind a confident fit."
        )
    if np.any(nearest < 0) or np.any(nearest > np.asarray(n, dtype=float)):
        raise ValueError(
            "method='binomial_ml': a recovered count is negative or exceeds its "
            "cohort size. Check the response and n columns."
        )
    return nearest


def _binom_negll(theta: np.ndarray, model: str, x: np.ndarray,
                 k: np.ndarray, n: np.ndarray) -> float:
    """Negative binomial log-likelihood, dropping the constant binomial term."""
    b, c, d, log_e = _unpack(theta, model)
    if not np.isfinite(b) or not np.isfinite(log_e):
        return np.inf
    try:
        p = _ll_predict(x, b, c, d, log_e)
    except Exception:
        return np.inf
    if not np.all(np.isfinite(p)):
        return np.inf
    p = np.clip(p, _P_EPS, 1.0 - _P_EPS)
    return float(-(np.sum(k * np.log(p) + (n - k) * np.log1p(-p))))


def _binom_loglik_full(theta: np.ndarray, model: str, x: np.ndarray,
                       k: np.ndarray, n: np.ndarray) -> float:
    """The log-likelihood INCLUDING the binomial coefficient, for reporting."""
    const = float(np.sum(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)))
    return const - _binom_negll(theta, model, x, k, n)


def resolvable_slope_limit(dose) -> dict:
    """
    The steepest Hill slope a dose grid can resolve, and the reasoning.

    A log-logistic response falls from 0.95 to 0.05 of its range over a dose
    ratio of 19 ** (2 / b). For the data to distinguish a finite slope from a
    step function, at least one dose spacing must fall inside that transition,
    so with `r` the smallest ratio between adjacent distinct non-zero doses

        b_max = 2 * ln(19) / ln(r)

    Past that, every curve fits the observations equally well and the fitted
    slope is a property of the grid. This is a resolution limit in the same
    sense as a Nyquist frequency: it says what the measurement can see, and it
    involves no assumption about what Hill slopes are biologically plausible.

    Returns `b_max`, `min_ratio`, `n_doses` and `reason`. `b_max` is infinite
    when there are fewer than two distinct non-zero doses, because then the
    question does not arise -- the design check refuses such a fit anyway.
    """
    d = np.unique(np.asarray(dose, dtype=float))
    d = d[np.isfinite(d) & (d > 0)]
    out = {"b_max": float("inf"), "min_ratio": float("nan"),
           "n_doses": int(d.size), "reason": ""}
    if d.size < 2:
        out["reason"] = ("Fewer than two distinct non-zero doses, so no slope "
                         "resolution limit applies.")
        return out
    ratios = d[1:] / d[:-1]
    r = float(np.min(ratios[ratios > 1.0])) if np.any(ratios > 1.0) else float("nan")
    if not np.isfinite(r) or r <= 1.0:
        out["reason"] = "Doses are not strictly increasing after deduplication."
        return out
    out["min_ratio"] = r
    out["b_max"] = float(2.0 * np.log(19.0) / np.log(r))
    out["reason"] = (
        f"With a smallest adjacent dose ratio of {r:.4g}, a curve steeper than "
        f"b = {out['b_max']:.2f} completes its 0.95-to-0.05 transition inside one "
        "dose spacing, so the data cannot distinguish it from a step and the "
        "fitted slope describes the grid rather than the response."
    )
    return out


def _dp_dtheta(theta: np.ndarray, model: str, x: np.ndarray):
    """
    The model mean and its derivatives with respect to the free parameters.

    Shared by the gradient and by the Fisher information, so the two cannot
    drift apart. With  u = 1/(1 + exp(z)),  z = b (log x - log e),
    p = c + (d - c) u :

        dp/dc = 1 - u        dp/dd = u
        dp/db = -(d - c) u (1 - u) (log x - log e)
        dp/dlog_e = (d - c) u (1 - u) b

    At x = 0 the model takes its limit (d for b > 0, c for b < 0), which does
    not depend on b or log e, so those derivatives are zero there -- the same
    limit `_ll_predict` uses, not a special case invented here.
    """
    b, c, d, log_e = _unpack(theta, model)
    x = np.asarray(x, dtype=float)
    pos = x > 0
    u = np.zeros_like(x)
    lx = np.zeros_like(x)
    if np.any(pos):
        lx[pos] = np.log(x[pos])
        z = np.clip(b * (lx[pos] - log_e), -_MAX_EXP, _MAX_EXP)
        u[pos] = 1.0 / (1.0 + np.exp(z))
    if np.any(~pos):
        u[~pos] = 1.0 if b > 0 else (0.0 if b < 0 else 0.5)

    p_hat = np.clip(c + (d - c) * u, _P_EPS, 1.0 - _P_EPS)
    uu = u * (1.0 - u)
    dp = {
        "c": 1.0 - u,
        "d": u,
        "b": np.where(pos, -(d - c) * uu * (lx - log_e), 0.0),
        "log_e": np.where(pos, (d - c) * uu * b, 0.0),
    }
    free = _MODEL_SPECS[model]["free"]
    return p_hat, np.column_stack([dp[name] for name in free])


def _fisher_info(theta: np.ndarray, model: str, x: np.ndarray,
                 n: np.ndarray) -> np.ndarray:
    """
    Expected (Fisher) information for a binomial response.

        I = SUM_i n_i / (p_i (1 - p_i)) * (dp_i/dtheta)(dp_i/dtheta)^T

    This is the matrix whose log-determinant is Firth's penalty, and it becomes
    singular exactly when a parameter is not determined by the data -- which is
    why the penalty keeps the maximum in the interior.
    """
    p_hat, D = _dp_dtheta(theta, model, x)
    wgt = np.asarray(n, dtype=float) / (p_hat * (1.0 - p_hat))
    return (D * wgt[:, None]).T @ D


def _firth_penalty(theta: np.ndarray, model: str, x: np.ndarray,
                   n: np.ndarray) -> float:
    """Half the log-determinant of the Fisher information (Firth 1993)."""
    I = _fisher_info(theta, model, x, n)
    try:
        sign, logdet = np.linalg.slogdet(I)
    except np.linalg.LinAlgError:
        return -np.inf
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf
    return 0.5 * float(logdet)


def _binom_negll_pen(theta: np.ndarray, model: str, x: np.ndarray,
                     k: np.ndarray, n: np.ndarray, penalty: str) -> float:
    """Negative log-likelihood, penalised or not."""
    base = _binom_negll(theta, model, x, k, n)
    if penalty == "none" or not np.isfinite(base):
        return base
    pen = _firth_penalty(theta, model, x, n)
    if not np.isfinite(pen):
        return np.inf
    return base - pen


def _binom_negll_grad(theta: np.ndarray, model: str, x: np.ndarray,
                      k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Analytic gradient of the binomial negative log-likelihood.

    Derived rather than differenced, because a finite-difference gradient costs
    p + 1 likelihood evaluations at every optimiser iteration and loses several
    digits of accuracy near the optimum -- which is where the Hessian, and
    therefore every standard error, is evaluated.

    With  u = 1 / (1 + exp(z)),  z = b (log x - log e),  p = c + (d - c) u :

        dp/dc      = 1 - u
        dp/dd      = u
        dp/db      = -(d - c) u (1 - u) (log x - log e)
        dp/dlog_e  =  (d - c) u (1 - u) b

    and  d(negll)/dp = -( k/p - (n - k)/(1 - p) ),  so the chain rule gives the
    gradient in one pass. At x = 0 the model takes its limit (d for b > 0, c
    for b < 0), which has no dependence on b or log e, so those entries are
    zero there -- that is the same limit `_ll_predict` uses, not a special
    case invented here.
    """
    p_hat, D = _dp_dtheta(theta, model, x)
    dneg_dp = -(k / p_hat - (n - k) / (1.0 - p_hat))
    return (D * dneg_dp[:, None]).sum(axis=0)


def _hessian_from_gradient(gradfun, theta: np.ndarray,
                           rel_step: float = 1e-5) -> np.ndarray:
    """
    Hessian by central differences of the ANALYTIC gradient.

    p + 1 gradient evaluations instead of order p^2 function evaluations, and
    one differencing step instead of two, so it is both faster and a couple of
    digits more accurate. Symmetrised, because the two mixed partials differ in
    the last bits and an asymmetric matrix has no real eigenvalues to gate on.
    """
    th = np.asarray(theta, dtype=float)
    m = th.size
    h = rel_step * np.maximum(np.abs(th), 1.0)
    H = np.zeros((m, m), dtype=float)
    for i in range(m):
        tp, tm = th.copy(), th.copy()
        tp[i] += h[i]
        tm[i] -= h[i]
        H[:, i] = (np.asarray(gradfun(tp), dtype=float)
                   - np.asarray(gradfun(tm), dtype=float)) / (2.0 * h[i])
    return 0.5 * (H + H.T)


def _ml_bounds(model: str, b_cap: Optional[float] = None):
    """
    Bounds on the free parameters.

    c and d are probabilities, so they live in [0, 1]. log_e is free. b is free
    unless `b_cap` is given, in which case it is restricted to [-b_cap, b_cap]
    -- the slope the dose grid can resolve, from `resolvable_slope_limit`.
    Capping it is not a prior on biology: beyond the cap every curve fits the
    observations identically, so the excluded region contains nothing the data
    distinguish, and leaving it in makes the likelihood maximum -- and every
    interval referenced to it -- a property of the dose spacing.
    """
    bb = (None, None) if b_cap is None or not np.isfinite(b_cap) \
        else (-float(b_cap), float(b_cap))
    lim = {"b": bb, "c": (_CD_EPS, 1.0 - _CD_EPS),
           "d": (_CD_EPS, 1.0 - _CD_EPS), "log_e": (None, None)}
    return [lim[name] for name in _MODEL_SPECS[model]["free"]]


def _numeric_hessian(fun, theta: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """
    Central-difference Hessian of a scalar function.

    The step is relative to each parameter's own magnitude, with an absolute
    floor, because `log_e` and `b` are routinely orders of magnitude apart and
    one step size for both loses either accuracy or conditioning.
    """
    th = np.asarray(theta, dtype=float)
    m = th.size
    h = rel_step * np.maximum(np.abs(th), 1.0)
    H = np.zeros((m, m), dtype=float)
    f0 = fun(th)
    for i in range(m):
        for j in range(i, m):
            if i == j:
                tp, tm = th.copy(), th.copy()
                tp[i] += h[i]
                tm[i] -= h[i]
                H[i, i] = (fun(tp) - 2.0 * f0 + fun(tm)) / (h[i] ** 2)
            else:
                tpp, tpm, tmp, tmm = (th.copy(), th.copy(), th.copy(), th.copy())
                tpp[i] += h[i]; tpp[j] += h[j]
                tpm[i] += h[i]; tpm[j] -= h[j]
                tmp[i] -= h[i]; tmp[j] += h[j]
                tmm[i] -= h[i]; tmm[j] -= h[j]
                H[i, j] = H[j, i] = ((fun(tpp) - fun(tpm) - fun(tmp) + fun(tmm))
                                     / (4.0 * h[i] * h[j]))
    return H


def _cov_from_information(H: np.ndarray):
    """
    Covariance from the observed information, with the same gate as N16.

    The Hessian of the NEGATIVE log-likelihood at a maximum is the observed
    information and must be positive definite. When it is not, the optimum is
    not an interior maximum -- a boundary fit, or a design that does not
    identify the curve -- and the correct output is no standard error at all.
    Inverting an indefinite matrix produces negative variances, and taking
    their square root produces NaN or, worse, a number from the absolute value.
    This is the defect N16 fixed in `hierarchical.py`, and the same gate is
    applied here rather than reinvented.
    """
    m = H.shape[0]
    out = {"cov": np.full((m, m), np.nan), "min_eigenvalue": float("nan"),
           "inference_valid": False, "reason": ""}
    if not np.all(np.isfinite(H)):
        out["reason"] = ("The observed information contains non-finite entries, so "
                         "no covariance can be formed.")
        return out
    Hs = 0.5 * (H + H.T)
    try:
        evals = np.linalg.eigvalsh(Hs)
    except np.linalg.LinAlgError:
        out["reason"] = "The observed information could not be decomposed."
        return out
    out["min_eigenvalue"] = float(np.min(evals))
    if np.min(evals) <= 0.0:
        out["reason"] = (
            f"The observed information is not positive definite (smallest "
            f"eigenvalue {np.min(evals):.3e}), so the optimum is not an interior "
            "maximum: the fit is at a boundary, or the design does not identify "
            "the curve. No standard errors are reported."
        )
        return out
    try:
        cov = np.linalg.inv(Hs)
    except np.linalg.LinAlgError:
        out["reason"] = "The observed information is singular."
        return out
    if np.any(np.diag(cov) < 0) or not np.all(np.isfinite(cov)):
        out["reason"] = ("The inverted information has a negative or non-finite "
                         "variance on its diagonal.")
        return out
    out["cov"] = cov
    out["inference_valid"] = True
    return out


def _fit_binomial_ml(x, y, w, model, design, force, penalty="firth",
                     b_cap=None):
    """
    Maximum-likelihood fit of the log-logistic to binomial counts.

    Returns the same `LogLogisticFit` the least-squares path returns, so every
    downstream consumer -- `lethal_dose_absolute`, `ec50_profile_ci`,
    `mortality_adjusted_response` -- works unchanged.
    """
    k = _counts_from_proportion(y, w)
    spec = _MODEL_SPECS[model]
    free = list(spec["free"])
    npar = len(free)
    bounds = _ml_bounds(model, b_cap)

    def negll_plain(theta):
        return _binom_negll(theta, model, x, k, w)

    def negll(theta):
        return _binom_negll_pen(theta, model, x, k, w, penalty)

    def grad(theta):
        """
        Gradient of the objective.

        The likelihood part is analytic. Firth's penalty involves the
        log-determinant of the information, whose derivative needs third
        derivatives of the mean; it is differenced instead, which costs p + 1
        determinant evaluations -- cheap, because the penalty is a p x p
        determinant and not a sum over observations.
        """
        g = _binom_negll_grad(theta, model, x, k, w)
        if penalty == "none":
            return g
        th = np.asarray(theta, dtype=float)
        h = 1e-6 * np.maximum(np.abs(th), 1.0)
        dpen = np.zeros_like(th)
        for i in range(th.size):
            tp, tm = th.copy(), th.copy()
            tp[i] += h[i]
            tm[i] -= h[i]
            a = _firth_penalty(tp, model, x, w)
            b_ = _firth_penalty(tm, model, x, w)
            if np.isfinite(a) and np.isfinite(b_):
                dpen[i] = (a - b_) / (2.0 * h[i])
        return g - dpen

    # Starting values come from the same linearisation the least-squares path
    # uses, with c and d pulled inside the probability bounds.
    best, best_f = None, np.inf
    for x0 in _start_values(x, y, model):
        x0 = np.asarray(x0, dtype=float).copy()
        for i, name in enumerate(free):
            if name in ("c", "d"):
                x0[i] = float(np.clip(x0[i], _CD_EPS, 1.0 - _CD_EPS))
        # L-BFGS-B with the analytic gradient is the workhorse. Nelder-Mead is
        # tried only if it has not yet produced a finite optimum, because
        # running a derivative-free method on every start was the bulk of the
        # fit's cost and never improved a start that had already converged.
        methods = (("L-BFGS-B",) if np.isfinite(best_f)
                   else ("L-BFGS-B", "Nelder-Mead"))
        for method_name in methods:
            try:
                r = optimize.minimize(
                    negll, x0, method=method_name,
                    jac=grad if method_name == "L-BFGS-B" else None,
                    bounds=bounds if method_name == "L-BFGS-B" else None,
                    options=({"maxiter": 5000, "maxfun": 5000, "ftol": 1e-14,
                              "gtol": 1e-10}
                             if method_name == "L-BFGS-B"
                             else {"maxiter": 5000, "xatol": 1e-10,
                                   "fatol": 1e-10}))
            except Exception:
                continue
            fval = float(r.fun)
            if np.isfinite(fval) and fval < best_f - 1e-12:
                # Nelder-Mead ignores bounds, so reject an out-of-bounds optimum
                th = np.asarray(r.x, dtype=float)
                okb = True
                for v, (lo, hi) in zip(th, bounds):
                    if lo is not None and v < lo - 1e-9:
                        okb = False
                    if hi is not None and v > hi + 1e-9:
                        okb = False
                if okb:
                    best, best_f = r, fval
    if best is None:
        raise RuntimeError(
            "method='binomial_ml': every starting value failed, so the "
            "likelihood could not be maximised."
        )

    theta = np.asarray(best.x, dtype=float)
    b, c, d, log_e = _unpack(theta, model)

    H = _hessian_from_gradient(grad, theta)
    info = _cov_from_information(H)
    if not info["inference_valid"]:
        # One retry with the slower, differencing-free route before concluding
        # the optimum is not interior: a gradient-differenced Hessian can be
        # mildly indefinite from step size alone, and refusing inference on a
        # numerical artefact would be the mirror image of N16.
        info_fd = _cov_from_information(_numeric_hessian(negll, theta))
        if info_fd["inference_valid"]:
            info = info_fd
            info["reason"] = ("The gradient-differenced information was not "
                              "positive definite; the function-differenced one "
                              "was, and is used.")
    cov_theta = info["cov"]
    se_theta = np.sqrt(np.clip(np.diag(cov_theta), 0.0, np.inf)) \
        if info["inference_valid"] else np.full(npar, np.nan)

    e_hat = _safe_exp(log_e)
    nat = {"b": b, "c": c, "d": d, "e": e_hat}
    nat_se = {"b": float("nan"), "c": float("nan"), "d": float("nan"),
              "e": float("nan")}
    for name, sd in zip(free, se_theta):
        if name == "log_e":
            nat_se["e"] = float(e_hat * sd)
        else:
            nat_se[name] = float(sd)
    # a parameter held fixed by the model has no uncertainty, not an unknown one
    for name in ("b", "c", "d"):
        if name not in free:
            nat_se[name] = 0.0

    fitted = _ll_predict(x, b, c, d, log_e)
    rss = float(np.sum(w * (y - fitted) ** 2))
    n_obs = int(x.size)
    df_resid = n_obs - npar
    order = ["b", "c", "d", "e"]

    fit = LogLogisticFit(
        model=model,
        params=pd.Series({kk: nat[kk] for kk in order}),
        bse=pd.Series({kk: nat_se[kk] for kk in order}),
        theta=theta,
        theta_names=free,
        cov_theta=cov_theta,
        rss=rss,
        resid_se=float(np.sqrt(rss / df_resid)) if df_resid > 0 else float("nan"),
        df_resid=int(df_resid),
        n_obs=n_obs,
        n_params=npar,
        dose=x,
        response=y,
        weights=w,
        fitted=fitted,
        residuals=y - fitted,
        converged=bool(getattr(best, "success", True)),
        design=design,
        message=str(getattr(best, "message", "")),
        method="binomial_ml",
        counts=k,
        totals=np.asarray(w, dtype=float),
        loglik=_binom_loglik_full(theta, model, x, k, w),
        loglik_penalised=(_binom_loglik_full(theta, model, x, k, w)
                          + (_firth_penalty(theta, model, x, w)
                             if penalty == "firth" else 0.0)),
        penalty=str(penalty),
        slope_cap=(float(b_cap) if b_cap is not None and np.isfinite(b_cap)
                   else float("nan")),
        inference_valid=bool(info["inference_valid"]),
        hessian_min_eigenvalue=float(info["min_eigenvalue"]),
        inference_reason=str(info["reason"]),
    )

    # Separation diagnosis, reported whether or not the penalty absorbed it.
    #
    # The test is not a threshold on b. It asks the question separation
    # actually poses: does an arbitrarily STEEPER curve fit as well? The
    # unpenalised likelihood is evaluated with the slope multiplied by ten, the
    # inflection held at the value that keeps the same LC50, and everything
    # else at the fit. If that loses almost nothing, the data do not determine
    # the slope from above and the unpenalised maximum is on the boundary.
    sep_gain = float("nan")
    try:
        i_b = free.index("b") if "b" in free else None
        if i_b is not None and np.isfinite(b) and b != 0.0:
            th_steep = theta.copy()
            th_steep[i_b] = b * 10.0
            if "log_e" in free:
                # hold the LC50 fixed while steepening, so only the slope moves
                i_le = free.index("log_e")
                num, den = d - 0.5, 0.5 - c
                if num > 0 and den > 0:
                    lk = np.log(num) - np.log(den)
                    th_steep[i_le] = (log_e + lk / b) - lk / (b * 10.0)
            ll_fit = -_binom_negll(theta, model, x, k, w)
            ll_steep = -_binom_negll(th_steep, model, x, k, w)
            sep_gain = float(ll_fit - ll_steep)
    except Exception:
        pass
    if np.isfinite(sep_gain) and sep_gain < 2.0:
        fit.separation = True
        fit.separation_note = (
            f"Quasi-complete separation: making the fitted curve ten times "
            f"steeper costs only {sep_gain:.2f} log-likelihood units, so the data "
            "do not determine the slope from above and the unpenalised maximum "
            "lies on the boundary b -> infinity. This happens when the observed "
            "proportions fall from near 1 to near 0 between two adjacent doses "
            "with nothing in between. "
            + ("Firth's penalty (Firth 1993) is in use, which keeps the maximum "
               "interior and the interval honest; the slope and the LC50 are "
               "still only weakly determined, so read the interval, not the "
               "point estimate."
               if penalty == "firth" else
               "No penalty is in use (penalty='none'), so the slope estimate and "
               "every interval derived from it are not trustworthy here. Use the "
               "default penalty='firth'.")
        )
    if not info["inference_valid"]:
        fit.design = dict(fit.design)
        fit.design["warnings"] = list(fit.design.get("warnings", [])) + [
            "Binomial ML: " + info["reason"]
        ]
    return fit


def fit_loglogistic(
    dose: Sequence[float],
    response: Sequence[float],
    n: Optional[Sequence[float]] = None,
    model: str = "LL.4",
    force: bool = False,
    method: str = "ls",
    penalty: str = "auto",
) -> LogLogisticFit:
    """
    Fit a log-logistic (Hill) dose-response curve by non-linear least squares.

        f(x) = c + (d - c) / (1 + exp(b * (log(x) - log(e))))

    Parameters
    ----------
    dose : doses, INCLUDING zeros for untreated controls. Zero doses are
        evaluated by the model's limit (see the module docstring): they inform
        the asymptote `d` directly and are never dropped.
    response : the measured response, one value per dose entry.
    n : optional number of flies underlying each response. When the response
        is a proportion out of `n` flies its variance is proportional to 1/n,
        so `n` is used as a least-squares WEIGHT. Leave as None for continuous
        responses measured once per vial.
    model : "LL.4" (b, c, d, e free), "LL.3" (c = 0) or "LL.2" (c = 0, d = 1).
    force : fit even if `check_design` refuses. The fit will still run but its
        standard errors are not trustworthy; only use this knowingly.
    method : "ls" (default) or "binomial_ml".

        "ls" is weighted least squares, unchanged, and remains the default. It
        is the right model for a CONTINUOUS response measured once per vial,
        and it is the path validated against R's `drc` on the `ryegrass` data.

        "binomial_ml" maximises the binomial likelihood of k successes out of n
        trials and takes its covariance from the observed information. Use it
        whenever the response really is a proportion out of a cohort -- which
        is every mortality and every pass/fail climbing endpoint in this
        platform. Register N95: least squares with `n` as the weight carries
        the 1/n of a binomial variance and nothing carries the p(1-p), and the
        measured consequence was an LC50 interval covering 0.47 rather than
        0.95 at a Hill slope of 4, unimproved by enriching the design. With
        `method="binomial_ml"`, `lethal_dose_absolute` reports a
        profile-likelihood interval on the LC50 itself and coverage is nominal
        (measured 0.94-0.96 across slopes in `validate_binomial_ml`).

        It requires `n`, and requires `response * n` to be whole numbers; it
        refuses rather than rounding, because a response already averaged over
        replicates is not a binomial count.

    Optimisation is unconstrained in (b, c, d, log e). Putting EC50 on the log
    scale keeps the dose positive without a boundary constraint, makes the
    surface much closer to quadratic, and is also the scale on which the
    profile interval is computed. Six starting values are tried (see
    `_start_values`) and the lowest residual sum of squares wins, because a
    Hill fit that fails almost always failed from a bad slope start rather
    than from a genuinely flat surface.

    Returns
    -------
    LogLogisticFit with natural-scale estimates and standard errors, the
    internal covariance matrix, the residual standard error, and the design
    diagnostics that were run before fitting.

    Raises
    ------
    ValueError if `check_design` refuses the design and `force` is False.
    """
    x = np.asarray(dose, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"dose and response have different lengths: {x.shape} vs {y.shape}")

    if n is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(n, dtype=float)
        if w.shape != x.shape:
            raise ValueError("n must have the same length as dose and response.")

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]

    spec = _MODEL_SPECS.get(model)
    if spec is None:
        raise ValueError(f"model must be one of {sorted(_MODEL_SPECS)}, got {model!r}")

    if method not in ("ls", "binomial_ml"):
        raise ValueError(
            f"method must be 'ls' or 'binomial_ml', got {method!r}. 'ls' is "
            "weighted least squares (the default, for a continuous response); "
            "'binomial_ml' is binomial maximum likelihood (for a proportion out "
            "of a cohort size) -- see register N95."
        )
    if method == "binomial_ml" and n is None:
        raise ValueError(
            "method='binomial_ml' needs the cohort size `n`: the binomial "
            "likelihood is defined on k successes out of n trials, and without n "
            "there is no n. Pass the number of flies scored at each dose, or use "
            "the default method='ls' for a continuous response."
        )

    design = check_design(x, y, model=model)
    if not design["usable"] and not force:
        raise ValueError(
            "Design cannot support a " + model + " fit:\n  - "
            + "\n  - ".join(design["refusals"])
            + "\nPass force=True to fit anyway (the standard errors will not be valid)."
        )

    if method == "binomial_ml":
        if penalty not in ("auto", "firth", "none"):
            raise ValueError(
                f"penalty must be 'auto', 'firth' or 'none', got {penalty!r}. "
                "'auto' (the default) fits the plain likelihood and refits with "
                "Firth's penalty only when the fitted slope exceeds what the "
                "dose grid can resolve -- see register N95."
            )
        if penalty != "auto":
            return _fit_binomial_ml(x, y, w, model, design, force,
                                    penalty=penalty)

        # Fit plain, then ask whether the design can resolve the slope that
        # came back. Measured: the plain likelihood gives better coverage in
        # seven of nine cells, and Firth's penalty is better in exactly the two
        # where the slope runs past the resolution limit -- which are also the
        # only two where it does so at all.
        lim = resolvable_slope_limit(x)
        plain = _fit_binomial_ml(x, y, w, model, design, force, penalty="none")
        b_hat = float(plain.params["b"])
        plain.design = dict(plain.design)
        plain.design["slope_resolution_limit"] = lim
        if not np.isfinite(b_hat) or abs(b_hat) <= lim["b_max"]:
            plain.penalty_rule = "auto: plain likelihood (slope within the "\
                                 "design's resolution limit)"
            return plain
        pen_fit = _fit_binomial_ml(x, y, w, model, design, force,
                                   penalty="firth", b_cap=lim["b_max"])
        pen_fit.design = dict(pen_fit.design)
        pen_fit.design["slope_resolution_limit"] = lim
        pen_fit.separation = True
        pen_fit.separation_note = (
            f"The unpenalised slope came out at b = {b_hat:.2f}, past the "
            f"b_max = {lim['b_max']:.2f} that this dose grid can resolve "
            f"(smallest adjacent dose ratio {lim['min_ratio']:.4g}). "
            "At that point the data cannot distinguish the curve from a step, "
            "so the slope -- and any interval conditioned on it -- would "
            "describe the dose spacing rather than the response. Refitted with "
            "Firth's penalised likelihood (Firth 1993), which keeps the maximum "
            "in the interior, AND with the slope constrained to the "
            f"resolvable range |b| <= {lim['b_max']:.2f} so that the interval "
            "answers which LC50 values are consistent with the data under any "
            "slope this grid can distinguish. The slope is still only weakly "
            "determined: read the interval, not the point estimate, and add "
            "intermediate doses if the slope itself matters."
        )
        pen_fit.penalty_rule = ("auto: Firth (unpenalised slope %.2f exceeded "
                                "the design limit %.2f)" % (b_hat, lim["b_max"]))
        return pen_fit

    sw = np.sqrt(w)
    free = list(spec["free"])
    p = len(free)

    def resid(theta: np.ndarray) -> np.ndarray:
        b, c, d, log_e = _unpack(theta, model)
        return sw * (_ll_predict(x, b, c, d, log_e) - y)

    best, best_rss = None, np.inf
    for x0 in _start_values(x, y, model):
        try:
            r = _lsq(resid, x0)
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if np.isfinite(rss) and rss < best_rss - 1e-14:
            best, best_rss = r, rss
    if best is None:
        raise RuntimeError("All starting values failed; the curve could not be fitted.")

    theta = np.asarray(best.x, dtype=float)
    b, c, d, log_e = _unpack(theta, model)
    n_obs = int(x.size)
    df_resid = n_obs - p
    s2 = best_rss / df_resid if df_resid > 0 else np.nan
    resid_se = float(np.sqrt(s2)) if np.isfinite(s2) else float("nan")

    # Gauss-Newton covariance on the internal scale
    J = np.asarray(best.jac, dtype=float)
    try:
        JtJ_inv = np.linalg.inv(J.T @ J)
        cov_theta = s2 * JtJ_inv
    except np.linalg.LinAlgError:
        cov_theta = np.full((p, p), np.nan)
    se_theta = np.sqrt(np.clip(np.diag(cov_theta), 0.0, np.inf))

    # Natural scale: e = exp(log_e), delta method SE(e) = e * SE(log e)
    e_hat = float(np.exp(log_e))
    nat = {"b": b, "c": c, "d": d, "e": e_hat}
    nat_se = {"b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
    for name, s in zip(free, se_theta):
        nat_se["e" if name == "log_e" else name] = float(e_hat * s) if name == "log_e" else float(s)

    fitted = _ll_predict(x, b, c, d, log_e)
    order = ["b", "c", "d", "e"]

    fit = LogLogisticFit(
        model=model,
        params=pd.Series({k: nat[k] for k in order}),
        bse=pd.Series({k: nat_se[k] for k in order}),
        theta=theta,
        theta_names=free,
        cov_theta=cov_theta,
        rss=float(best_rss),
        resid_se=resid_se,
        df_resid=int(df_resid),
        n_obs=n_obs,
        n_params=p,
        dose=x,
        response=y,
        weights=w,
        fitted=fitted,
        residuals=y - fitted,
        converged=bool(best.success),
        design=design,
        message=str(best.message),
    )
    if b < 0:
        fit.design = dict(fit.design)
        fit.design["warnings"] = list(fit.design.get("warnings", [])) + [
            "The fitted Hill slope b is negative, i.e. the response INCREASES with "
            "dose. That is a legitimate fit (d is then the low control value and c "
            "the high plateau), but check it is what you expected -- it is also what "
            "a mis-signed response column produces."
        ]
    return fit


# ═══════════════════════════════════════════════════════════════════════════
# 3. PROFILE-LIKELIHOOD INTERVALS
# ═══════════════════════════════════════════════════════════════════════════
def _profile_rss(fit: LogLogisticFit, log_ecx: float, log_k: float,
                 warm: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """
    Residual sum of squares with ECx CONSTRAINED to exp(log_ecx), the other
    parameters re-optimised.

    The reparameterisation is exact rather than approximate. For the relative
    ECx of the log-logistic,

        ECx = e * (x / (100 - x))^(1/b)      =>   log e = log ECx - log(k)/b

    with k = x/(100 - x). So fixing log ECx and freeing (b, c, d) covers
    exactly the same model space as the unconstrained fit restricted to that
    ECx value. For x = 50, k = 1 and log(k) = 0, so this reduces to fixing
    log(e) itself.

    Returns (RSS, nuisance parameters at the optimum).
    """
    model = fit.model
    free = fit.theta_names
    nuis_names = [nm for nm in free if nm != "log_e"]
    x, y, sw = fit.dose, fit.response, np.sqrt(fit.weights)
    spec = _MODEL_SPECS[model]

    def resid(nu: np.ndarray) -> np.ndarray:
        vals = dict(spec["fixed"])
        for nm, v in zip(nuis_names, nu):
            vals[nm] = float(v)
        b = vals["b"]
        if abs(b) < 1e-9:
            b = 1e-9 if b >= 0 else -1e-9
        log_e = log_ecx - log_k / b
        return sw * (_ll_predict(x, b, vals["c"], vals["d"], log_e) - y)

    if warm is None:
        warm = np.array([fit.theta[free.index(nm)] for nm in nuis_names], dtype=float)

    if len(nuis_names) == 0:
        return float(np.sum(resid(np.array([])) ** 2)), np.array([])

    best, best_rss = warm, np.inf
    for x0 in (warm,
               np.array([fit.theta[free.index(nm)] for nm in nuis_names], dtype=float)):
        try:
            r = _lsq(resid, x0)
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if np.isfinite(rss) and rss < best_rss:
            best_rss, best = rss, np.asarray(r.x, dtype=float)
    return best_rss, best


def _profile_interval(fit: LogLogisticFit, level: float, x_pct: float,
                      max_log_span: float = 12.0,
                      grid_points: int = 60) -> dict:
    """
    Invert the profile-F statistic to get a confidence interval for ECx.

    The interval is the set of ECx values whose constrained residual sum of
    squares stays below

        RSS_min * (1 + (p_full - p_reduced) / (n - p_full) * F(1, n - p_full, level))

    with p_full - p_reduced = 1 because exactly one parameter is being fixed.
    This is the F-test inversion of Seber & Wild (1989, ch. 5) / Bates & Watts
    (1988), and unlike the delta method it makes no symmetry assumption: the
    interval it returns is asymmetric on the dose scale whenever the surface
    is, which for EC50 it essentially always is.

    The search walks outward from the estimate in log-dose steps until the
    threshold is crossed, then brackets the crossing with Brent's method.
    Each step warm-starts the nuisance optimisation at the previous step's
    solution, which is what keeps the profile smooth.
    """
    k = x_pct / (100.0 - x_pct)
    log_k = float(np.log(k))
    b_hat = float(fit.params["b"])
    log_ecx_hat = float(np.log(fit.params["e"])) + log_k / b_hat

    dfr = fit.df_resid
    if dfr <= 0 or not np.isfinite(fit.rss):
        return {"lower": np.nan, "upper": np.nan, "threshold": np.nan,
                "reason": "No residual degrees of freedom."}

    f_crit = float(stats.f.ppf(level, 1, dfr))
    thresh = fit.rss * (1.0 + f_crit / dfr)

    se_log = fit.se_log_e
    step = 0.05 if not np.isfinite(se_log) or se_log <= 0 else max(min(se_log / 3.0, 0.5), 0.01)

    def g(L: float, warm_box: list) -> float:
        rss, nu = _profile_rss(fit, L, log_k, warm_box[0])
        warm_box[0] = nu
        return rss - thresh

    bounds: Dict[str, float] = {}
    for direction, key in ((-1.0, "lower"), (+1.0, "upper")):
        warm_box = [None]
        prev_L, prev_g = log_ecx_hat, g(log_ecx_hat, warm_box)
        found = np.nan
        n_steps = int(max_log_span / step) + 1
        for i in range(1, min(n_steps, 4000) + 1):
            L = log_ecx_hat + direction * step * i
            try:
                gv = g(L, warm_box)
            except Exception:
                break
            if np.isfinite(gv) and gv > 0:
                try:
                    root = optimize.brentq(lambda t: g(t, [warm_box[0]]),
                                           min(prev_L, L), max(prev_L, L),
                                           xtol=1e-10, rtol=1e-12, maxiter=200)
                    found = float(np.exp(root))
                except Exception:
                    found = float(np.exp(0.5 * (prev_L + L)))
                break
            prev_L, prev_g = L, gv
        bounds[key] = found

    lo, hi = bounds["lower"], bounds["upper"]
    unbounded = []
    if not np.isfinite(lo):
        lo = 0.0
        unbounded.append("lower")
    if not np.isfinite(hi):
        hi = float("inf")
        unbounded.append("upper")

    return {
        # Register N96. This module's interval sub-dicts are keyed
        # `lower`/`upper` while the top-level returns of
        # `lethal_dose_absolute` use `ci_lower`/`ci_upper`. Mixing the two
        # conventions inside one return value reads as a typo and silently
        # yields None: it cost a measurement run here, where a coverage study
        # read `profile["ci_lower"]`, got None for all 1,800 fits and concluded
        # the profile interval could never be computed. Both spellings are now
        # present and always equal, so neither reading can be wrong.
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "lower": float(lo),
        "upper": float(hi),
        "threshold": float(thresh),
        "rss_min": float(fit.rss),
        "f_crit": f_crit,
        "df_resid": int(dfr),
        "unbounded": unbounded,
        "grid_step_log": step,
    }


def _delta_interval(fit: LogLogisticFit, level: float, x_pct: float) -> dict:
    """
    Delta-method intervals for ECx, on both the dose scale and the log-dose
    scale, for comparison with the profile interval.

    log ECx = log e + log(k) / b,  with k = x/(100-x), so

        d log ECx / d log e = 1
        d log ECx / d b     = -log(k) / b^2

    The "naive" interval is estimate +/- t * SE on the DOSE scale, which is
    the one that is symmetric by construction and can run below zero. The
    "log-scale" delta interval exponentiates the interval built on log ECx; it
    is at least guaranteed positive, but it still imposes a shape (log-normal)
    rather than reading it off the likelihood.
    """
    dfr = fit.df_resid
    crit = float(stats.t.ppf(0.5 + level / 2.0, dfr)) if dfr > 0 else np.nan

    k = x_pct / (100.0 - x_pct)
    log_k = float(np.log(k))
    b = float(fit.params["b"])
    ecx = float(fit.params["e"]) * (k ** (1.0 / b))
    log_ecx = float(np.log(ecx))

    names = fit.theta_names
    grad = np.zeros(len(names))
    grad[names.index("log_e")] = 1.0
    if "b" in names:
        grad[names.index("b")] = -log_k / (b ** 2)

    var_log = float(grad @ fit.cov_theta @ grad)
    se_log = float(np.sqrt(var_log)) if np.isfinite(var_log) and var_log >= 0 else np.nan
    se_dose = float(ecx * se_log) if np.isfinite(se_log) else np.nan

    return {
        "estimate": ecx,
        "se": se_dose,
        "se_log": se_log,
        "lower": float(ecx - crit * se_dose) if np.isfinite(se_dose) else np.nan,
        "upper": float(ecx + crit * se_dose) if np.isfinite(se_dose) else np.nan,
        "log_lower": _safe_exp(log_ecx - crit * se_log) if np.isfinite(se_log) else np.nan,
        "log_upper": _safe_exp(log_ecx + crit * se_log) if np.isfinite(se_log) else np.nan,
    }


def lethal_dose_absolute(fit: LogLogisticFit, target: float = 0.5,
                         level: float = 0.95, profile: bool = True,
                         calibrate: str = "asymptotic", n_boot: int = 200,
                         seed: Optional[int] = None) -> dict:
    """
    The dose at which the fitted curve reaches an ABSOLUTE response value --
    for a survival curve with `target = 0.5`, the LC50.

    Register N14. `e` is the curve's inflection point, the dose at which the
    response is half way between the asymptotes `c` and `d`. That equals the
    LC50 only when the curve runs from 1 to 0. The default mortality model here
    is LL.3, whose upper asymptote d is FREE -- precisely so that control
    survival can be below 1, which is Abbott's (1925) correction expressed as a
    parameter -- so `e` is the dose at which survival is (c + d)/2, not 0.5.
    With c = 0 and d = 0.90 that is survival 0.45, and the dose at which it is
    reached is 11.9% above the real LC50. At d = 0.80 the error is +67%.

    The general definition, solved in closed form:

        target = c + (d - c) / (1 + exp(b * (log x - log e)))
        =>  x = e * ((d - target) / (target - c)) ** (1 / b)

    which is `e` exactly when target = (c + d)/2, and reduces to `e` for LL.2
    (c = 0, d = 1, target = 0.5) -- so nothing changes for a curve that really
    does run from 1 to 0.

    Interval. The reported interval is a DELTA-METHOD interval on log-dose,
    because log LC50 depends on b, c and d as well as on log e and all four
    covary:

        log LC50 = log e + (1/b) * [log(d - target) - log(target - c)]

    The profile-likelihood interval for the equivalent RELATIVE level is
    returned alongside as `profile`, but it is conditional on the fitted
    asymptotes (the relative level is itself computed from c and d), so it is
    secondary here. For LL.2 the two coincide with `ec50_profile_ci`.

    Refusals. If `target` does not lie strictly between the fitted asymptotes
    there is no such dose: the estimate is NaN and `reason` says why. That is
    the case when control survival is already at or below 50%, where an LC50
    can only be produced by extrapolating outside the model's own range.

    `profile`. The secondary profile-likelihood interval costs about 38
    additional non-linear fits -- measured at 0.52 s against 0.05 s for
    everything else in this function, so it is 90% of the cost. Pass
    `profile=False` where it is not wanted: a Monte-Carlo calibration study,
    which needs only the interval the function reports as primary, or a batch
    screen of many compounds where the profile is computed on demand for the
    few that matter. `profile` is then None and `profile_skipped` says why.
    Nothing else changes, and the default is unchanged.

    Returns a dict with `estimate`, `ci_lower`, `ci_upper`, `se_log`,
    `param_e`, `relative_level_pct`, `profile`, `definition`, `reason`,
    `extrapolated`, `interval_caveat`, `interval_coverage_nominal`.

    KEY NAMES (register N96, fixed). The interval at the top level is
    `ci_lower` / `ci_upper`. The nested `profile` dict carries BOTH that pair
    and this module's own `lower` / `upper`, always equal, so neither reading
    returns None. Before that, a coverage study read `profile["ci_lower"]`, got
    None for all 1,800 fits and concluded the profile interval could never be
    computed.
    """
    t = float(target)
    b = float(fit.params["b"])
    c = float(fit.params["c"])
    d = float(fit.params["d"])
    e = float(fit.params["e"])

    out = {
        "target": t,
        "param_e": e,
        "model": fit.model,
        "level": float(level),
        "definition": (
            f"The dose at which the fitted curve equals {t:g} (for a survival "
            f"curve, {t:.0%} survival): x = e * ((d - {t:g})/({t:g} - c))**(1/b). "
            f"NOT the fitted `e`, which is the dose at which the response is "
            f"(c + d)/2 = {(c + d) / 2.0:.4g}."
        ),
        "estimate": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "se_log": float("nan"),
        "relative_level_pct": float("nan"),
        "profile": None,
        "reason": "",
        "extrapolated": False,
    }

    lo_a, hi_a = min(c, d), max(c, d)
    if not np.isfinite(b) or b == 0.0:
        out["reason"] = "The fitted slope b is zero or not finite; the curve has no inverse."
        return out
    if not (lo_a < t < hi_a):
        out["reason"] = (
            f"The fitted curve runs between c = {c:.4g} and d = {d:.4g}, so it never "
            f"reaches {t:g}. For a survival curve this means control survival "
            f"(d = {d:.4g}) is already at or below {t:.0%}, or the lower plateau "
            f"(c = {c:.4g}) is above it: there is no {t:.0%} dose inside the model's "
            f"own range, and reporting one would be extrapolation. Refused."
        )
        return out

    num, den = (d - t), (t - c)
    log_k = float(np.log(num) - np.log(den))
    log_ld = float(np.log(e)) + log_k / b
    est = _safe_exp(log_ld)
    out["estimate"] = est
    out["relative_level_pct"] = float(100.0 * (d - t) / (d - c))

    # Delta method on log LC50, including the asymptotes.
    names = fit.theta_names
    grad = np.zeros(len(names))
    grad[names.index("log_e")] = 1.0
    if "b" in names:
        grad[names.index("b")] = -log_k / (b ** 2)
    if "d" in names:
        grad[names.index("d")] = 1.0 / (b * num)
    if "c" in names:
        grad[names.index("c")] = 1.0 / (b * den)
    var_log = float(grad @ fit.cov_theta @ grad)
    se_log = float(np.sqrt(var_log)) if np.isfinite(var_log) and var_log >= 0 else float("nan")
    out["se_log"] = se_log
    dfr = fit.df_resid
    crit = float(stats.t.ppf(0.5 + level / 2.0, dfr)) if dfr > 0 else float("nan")
    if np.isfinite(se_log) and np.isfinite(crit):
        out["ci_lower"] = _safe_exp(log_ld - crit * se_log)
        out["ci_upper"] = _safe_exp(log_ld + crit * se_log)
    else:
        out["reason"] = (
            "The parameter covariance is not usable (an indefinite or singular "
            "numerical Hessian), so no interval is reported. The point estimate "
            "stands but is not accompanied by uncertainty."
        )

    # Register N95. For a binomial ML fit the profile-likelihood interval on
    # the LC50 itself becomes the reported interval, and the delta interval is
    # kept beside it under its own name rather than discarded -- the two
    # disagreeing is informative, and a reader who wants the old number can
    # still have it.
    if getattr(fit, "method", "ls") == "binomial_ml":
        out["delta_ci_lower"] = out["ci_lower"]
        out["delta_ci_upper"] = out["ci_upper"]
        prof_lc = lethal_dose_profile(fit, target=t, level=level,
                                      calibrate=calibrate, n_boot=n_boot,
                                      seed=seed)
        out["threshold_advice"] = prof_lc.get("threshold_advice", "")
        out["profile_lc50"] = prof_lc
        if np.isfinite(prof_lc.get("ci_lower", np.nan)) and \
                np.isfinite(prof_lc.get("ci_upper", np.nan)):
            out["ci_lower"] = float(prof_lc["ci_lower"])
            out["ci_upper"] = float(prof_lc["ci_upper"])
            out["interval_kind"] = "profile_likelihood_on_lc50"
        else:
            out["interval_kind"] = "delta_on_log_dose"
            out["reason"] = (out["reason"] + " " if out["reason"] else "") + (
                "The profile interval is not bounded on both sides, so the "
                "delta-method interval is reported instead: " +
                str(prof_lc.get("reason", ""))
            )
        out["method"] = (
            "Binomial maximum likelihood; closed-form inversion at an absolute "
            "response level; profile-likelihood interval on the LC50 itself "
            "(chi-square(1) deviance)."
        )
        out["inference_valid"] = bool(getattr(fit, "inference_valid", True))
        cav_ml = interval_coverage_caveat(fit)
        adv = out.get("threshold_advice", "")
        if adv:
            # The fit is in the regime where the asymptotic reference is
            # optimistic -- measured at about 0.90 coverage. Saying so is the
            # whole point of N95; claiming nominal here would reinstate it.
            cav_ml = (cav_ml + " " if cav_ml else "") + adv
        out["interval_caveat"] = cav_ml
        out["interval_coverage_nominal"] = cav_ml is None
        nz_ml = fit.dose[fit.dose > 0]
        if nz_ml.size and np.isfinite(est):
            out["extrapolated"] = bool(est > float(np.max(fit.dose))
                                       or est < float(np.min(nz_ml)))
        return out

    x_pct = out["relative_level_pct"]
    if not profile:
        out["profile_skipped"] = (
            "profile=False: the secondary profile-likelihood interval was not "
            "computed. `ci_lower`/`ci_upper` are the delta-method interval this "
            "function reports as primary and are unaffected."
        )
    elif 0.0 < x_pct < 100.0:
        try:
            out["profile"] = _profile_interval(fit, level, x_pct)
        except Exception as exc:                       # pragma: no cover
            out["profile"] = {"error": str(exc)}

    nz = fit.dose[fit.dose > 0]
    if nz.size and np.isfinite(est):
        out["extrapolated"] = bool(est > float(np.max(fit.dose)) or est < float(np.min(nz)))
    out["method"] = (
        "Closed-form inversion of the fitted log-logistic at an absolute response "
        "level; delta-method interval on log-dose including b, c and d."
    )
    # Measured, not asserted: see BINOMIAL_INTERVAL_COVERAGE_NOTE above.
    cav = interval_coverage_caveat(fit)
    out["interval_caveat"] = cav
    out["interval_coverage_nominal"] = cav is None
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Interval calibration under binomial sampling -- a MEASURED limitation
# ═══════════════════════════════════════════════════════════════════════════
# `fit_loglogistic` is weighted least squares with the cohort size `n` as the
# weight. That reproduces the 1/n part of a binomial variance but NOT the
# p(1-p) part. On a steep curve almost every dose sits near 0 or 1, where
# binomial variance is nearly zero, and only one dose sits near 0.5, where it
# is largest; homoscedastic least squares weights them alike, so `cov_theta`
# is too small -- and so is every interval derived from it.
#
# Measured (`validate_statistical_calibration`, and the studies recorded in
# BUG_AND_RISK_REGISTER.md under N95), 7 doses including a control, 40 flies
# per dose, c = 0, e = 1, nominal 95%:
#
#   true d   slope b   binomial   homoscedastic Gaussian   binomial moments
#     1.00      1        0.953            0.963                 0.950
#     1.00      2        0.802            0.960                 0.830
#     1.00      4        0.465            0.940                 0.445
#     0.90      1        0.947            0.943                 0.950
#     0.90      2        0.863            0.927                 0.897
#     0.90      4        0.791            0.949                 0.767
#
# Under homoscedastic Gaussian noise the interval attains nominal coverage at
# every slope, which is what least squares promises. Under binomial sampling
# it does not, and the third column -- continuous noise with binomial moments
# -- reproduces the shortfall, so the cause is the SHAPE of the variance and
# not the discreteness of counts. The mechanism is visible directly: at
# d = 1, b = 4 the reported SE of log LC50 averages 0.0170 while the true
# sampling SD of log LC50 is 0.0682, four times larger.
#
# It is NOT a small-design artefact. Enriching the design to 20 doses and 300
# flies per dose leaves coverage at b = 4 essentially unchanged (0.430 ->
# 0.510 -> 0.483 across three designs), while BIAS does improve with design
# (+2.4% -> +0.65% -> +0.32%). More data does not fix a wrong variance model.
#
# The profile-likelihood interval is better but not a remedy: it is computed
# from the same residual sum of squares and carries the same assumption,
# measuring 0.520 against the delta method's 0.480 at b = 4, and it frequently
# cannot be computed at all on shallow curves (41 of 150 fits at d = 0.9,
# b = 1).
#
# The remedy is a change to the FIT -- iteratively reweighted least squares
# with binomial weights, or binomial maximum likelihood, or a parametric
# bootstrap for the interval. That is a substantial change to a scientific
# module and is recorded as an owner decision rather than made here. What is
# done here is to stop the interval being reported as a 95% interval without
# saying what it was measured to be.
BINOMIAL_INTERVAL_COVERAGE_NOTE = (
    "Interval calibration: this interval comes from a least-squares parameter "
    "covariance, which assumes residuals of constant variance. These data are "
    "proportions out of a cohort size, so their variance is p(1-p)/n and is "
    "smallest where the curve is flat. Measured coverage of the nominal 95% "
    "interval under binomial sampling is 0.95 at a slope of b = 1, 0.80-0.86 "
    "at b = 2 and 0.47-0.79 at b = 4, and it does NOT improve with a larger "
    "design. Read the interval as a lower bound on the uncertainty, more so "
    "the steeper the fitted slope. See BUG_AND_RISK_REGISTER.md N95."
)


def interval_coverage_caveat(fit: "LogLogisticFit") -> Optional[str]:
    """
    The calibration caveat, where it applies.

    It applies when the fit was given cohort sizes -- that is what makes the
    response a binomial proportion rather than a continuous measurement. A
    continuous response measured once per vial is the case least squares was
    built for and carries no caveat.
    """
    # A binomial ML fit models the variance the data actually have, and
    # `lethal_dose_absolute` reports a profile-likelihood interval on the LC50
    # itself for it. Measured coverage on that path is 0.94-0.96 across slopes
    # of 1, 2 and 4 (`validate_binomial_ml`, check E), so there is nothing to
    # caveat. This is the only exit that returns None for a binomial response,
    # and it is the one the remedy earns.
    if getattr(fit, "method", "ls") == "binomial_ml":
        return None
    w = getattr(fit, "weights", None)
    if w is None:
        return None
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        return None
    # uniform weights mean no cohort sizes were supplied
    finite = w[np.isfinite(w)]
    if finite.size == 0 or float(np.ptp(finite)) == 0.0 and float(finite[0]) == 1.0:
        return None
    return BINOMIAL_INTERVAL_COVERAGE_NOTE


def bootstrap_deviance_threshold(fit: "LogLogisticFit", target: float = 0.5,
                                 level: float = 0.95, n_boot: int = 200,
                                 seed: Optional[int] = None) -> dict:
    """
    The deviance threshold this design actually needs, by parametric bootstrap.

    The asymptotic threshold for a one-parameter profile is chi-square(1), exact
    in the limit. On a seven-dose bioassay with forty flies per dose it is
    slightly optimistic, and the shortfall is worst where the quantity of
    interest depends strongly on a weakly estimated parameter -- an LC50 with
    substantial control mortality, where it is a strong function of the upper
    asymptote.

    So the reference distribution is simulated instead of assumed: the fitted
    curve is taken as truth, `n_boot` datasets are drawn with the same doses and
    cohort sizes, each is refitted, and the profile deviance is evaluated at the
    generating model's own LC50. The `level` quantile of those deviances is the
    threshold. Replicates whose fit or profile fails are dropped and counted,
    because a bootstrap that silently discards its failures reports a quantile
    of the cases that happened to work.

    WHAT THIS IS GOOD FOR, measured rather than hoped. On an informative design
    the bootstrap threshold lands near the asymptotic one and the `ratio` is
    close to 1, confirming the chi-square reference. In the regime where the
    dose grid cannot resolve the slope it does NOT give a usable critical
    value: measured on a fit with 7 doses x 40 flies, a slope of 4 and an upper
    asymptote of 0.80, it returned 136.4 against the asymptotic 3.841, a ratio
    of 35.5 -- because the bootstrap datasets separate too, so the statistic's
    upper tail is enormous and its 95th percentile is not a bound anyone should
    use. An interval built on that would be conservative by an arbitrary
    amount, which is its own way of being wrong.

    So read `ratio` as a DESIGN ADEQUACY measure: how far this design is from
    the regime where the interval's stated level means what it says. Near 1,
    the asymptotic interval is sound. Large, and the honest conclusion is that
    the design cannot support an interval at this level -- add intermediate
    doses, or control flies, rather than widening the bound.

    Returns `threshold`, `chi2_threshold`, `ratio`, `n_used`, `n_requested`,
    `mean_deviance`, `verdict`, `reason`. `threshold` falls back to the
    chi-square value, with `reason` saying so, when too few replicates survive
    to estimate a quantile -- a bootstrap on thirty usable draws is not a
    calibration.
    """
    chi2_thr = float(stats.chi2.ppf(level, 1))
    out = {"threshold": chi2_thr, "chi2_threshold": chi2_thr, "ratio": 1.0,
           "n_used": 0, "n_requested": int(n_boot),
           "mean_deviance": float("nan"), "verdict": "", "reason": ""}
    if fit.method != "binomial_ml":
        out["reason"] = ("Bootstrap calibration needs a likelihood, so it needs "
                         "method='binomial_ml'.")
        return out
    if fit.totals is None:
        out["reason"] = "The fit did not retain its cohort sizes."
        return out

    t = float(target)
    b = float(fit.params["b"])
    c = float(fit.params["c"])
    d = float(fit.params["d"])
    e = float(fit.params["e"])
    if not (min(c, d) < t < max(c, d)) or not np.isfinite(b) or b == 0.0:
        out["reason"] = "The fitted curve does not reach the target level."
        return out

    lc_gen = e * ((d - t) / (t - c)) ** (1.0 / b)
    p_gen = _ll_predict(fit.dose, b, c, d, float(np.log(e)))
    p_gen = np.clip(p_gen, 0.0, 1.0)
    nn = np.asarray(fit.totals, dtype=float)
    rng = np.random.default_rng(seed)
    cap = getattr(fit, "slope_cap", float("nan"))

    devs = []
    for _ in range(int(n_boot)):
        k_b = rng.binomial(nn.astype(int), p_gen)
        try:
            fb = _fit_binomial_ml(fit.dose, k_b / nn, nn, fit.model,
                                  dict(fit.design), True,
                                  penalty=getattr(fit, "penalty", "none"),
                                  b_cap=cap if np.isfinite(cap) else None)
            fn, _est, _why = lc_profile_deviance(fb, t)
        except Exception:
            continue
        if fn is None:
            continue
        v = fn(lc_gen, record=False)
        if np.isfinite(v) and v >= -1e-6:
            devs.append(max(float(v), 0.0))

    out["n_used"] = len(devs)
    if len(devs) < 50:
        out["reason"] = (
            f"Only {len(devs)} of {n_boot} bootstrap replicates produced a usable "
            "deviance, which is too few to estimate a quantile; the asymptotic "
            "chi-square(1) threshold is used instead."
        )
        return out
    devs = np.asarray(devs, dtype=float)
    thr = float(np.quantile(devs, level))
    out["threshold"] = thr
    out["ratio"] = float(thr / chi2_thr)
    out["mean_deviance"] = float(devs.mean())
    out["reason"] = (
        f"Threshold {thr:.3f} from {len(devs)} parametric-bootstrap replicates "
        f"against the asymptotic {chi2_thr:.3f} (ratio {thr / chi2_thr:.3f}); "
        f"the statistic's bootstrap mean is {devs.mean():.3f} against the "
        "asymptotic 1.000."
    )
    r = out["ratio"]
    if r <= 1.5:
        out["verdict"] = (
            "adequate: the bootstrap reference agrees with the asymptotic one, "
            "so the chi-square(1) interval means what it says at this design."
        )
    elif r <= 4.0:
        out["verdict"] = (
            f"marginal: the bootstrap reference is {r:.1f}x the asymptotic one, "
            "so the asymptotic interval is somewhat optimistic. The bootstrap "
            "threshold is usable here; expect a wider interval."
        )
    else:
        out["verdict"] = (
            f"INADEQUATE DESIGN: the bootstrap reference is {r:.1f}x the "
            "asymptotic one, which means the likelihood-ratio statistic has no "
            "usable upper quantile at this design -- the bootstrap datasets "
            "separate as well, so its tail is arbitrary. Do not read this "
            "threshold as a 95% bound and do not read the asymptotic interval "
            "as one either. The remedy is the design: intermediate doses so the "
            "slope is resolvable, and more control flies if the upper asymptote "
            "is far from 1."
        )
    return out


def threshold_advice(fit: "LogLogisticFit", target: float = 0.5) -> str:
    """
    Whether this fit is in the regime where the asymptotic threshold is
    optimistic enough to be worth replacing.

    TWO conditions, both read off the fit and the design, and both placed by
    measurement rather than taste. Coverage of the asymptotic interval, 900
    replications per cell on 7 doses x 40 flies:

        upper asymptote d   headroom   b = 1    b = 2    b = 4
             1.00             0.500     0.977    0.970    0.942
             0.90             0.444     0.957    0.953    0.934
             0.80             0.375     0.937    0.950    0.897

    where headroom = |d - target| / |d - c| is the fraction of the fitted range
    lying between the target and the upper asymptote. Only ONE cell falls
    materially short, and it needs both a small headroom AND a slope that is a
    large fraction of what the dose grid resolves. Low headroom alone is fine
    (0.937 and 0.950 at d = 0.80 with gentler slopes) and so is a steep slope
    alone (0.942 at d = 1.00). So the advice fires on the conjunction:
    headroom below 0.40, which separates 0.375 from the 0.444 that covers at
    0.934, and a slope above 35% of the resolvable limit, which separates b = 4
    (47% of 8.50) from b = 2 (24%).
    """
    try:
        t = float(target)
        c = float(fit.params["c"])
        d = float(fit.params["d"])
        b = abs(float(fit.params["b"]))
        if not (min(c, d) < t < max(c, d)):
            return ""
        headroom = abs(d - t) / abs(d - c)
        lim = resolvable_slope_limit(fit.dose)["b_max"]
        slope_frac = b / lim if np.isfinite(lim) and lim > 0 else 0.0
    except Exception:
        return ""
    if headroom < 0.40 and slope_frac > 0.35:
        return (
            f"Only {100 * headroom:.0f}% of the fitted range lies between the "
            f"target ({t:g}) and the upper asymptote ({d:.3g}), and the fitted "
            f"slope is {100 * slope_frac:.0f}% of the steepest this dose grid "
            f"can resolve (b = {b:.2f} against b_max = {lim:.2f}). In that "
            "combination the target dose is a strong function of the asymptote, "
            "which is estimated from the control cohort alone, and the "
            "asymptotic chi-square reference is optimistic: measured coverage "
            "0.897 against a nominal 0.95 on 900 replications. The interval is "
            "reported but NOT as a 95% interval. The remedy is the design -- "
            "intermediate doses to resolve the slope, more control flies to pin "
            "the asymptote. `bootstrap_deviance_threshold` quantifies how far "
            "off this design is."
        )
    return ""


def _lr_threshold(fit: "LogLogisticFit", level: float = 0.95,
                  calibrate: str = "asymptotic", n_boot: int = 200,
                  seed: Optional[int] = None):
    """
    Deviance threshold for a one-parameter profile, and what kind it is.

    `calibrate="asymptotic"` (the default) is the chi-square(1) quantile, which
    the measurements show is correct on an informative design -- on a
    twelve-dose grid the profile deviance is textbook chi-square(1) in every
    cell tested, with a mean of 0.97 to 1.21 against the asymptotic 1.0.
    `calibrate="bootstrap"` simulates the statistic from the fitted model
    instead; see `bootstrap_deviance_threshold`.
    """
    chi2_thr = float(stats.chi2.ppf(level, 1))
    if calibrate == "asymptotic":
        return chi2_thr, "chi-square(1)"
    if calibrate != "bootstrap":
        raise ValueError(
            f"calibrate must be 'asymptotic' or 'bootstrap', got {calibrate!r}."
        )
    bs = bootstrap_deviance_threshold(fit, level=level, n_boot=n_boot, seed=seed)
    kind = ("parametric bootstrap (%d replicates)" % bs["n_used"]
            if bs["n_used"] >= 50 else "chi-square(1), bootstrap unavailable")
    return float(bs["threshold"]), kind



def lc_profile_deviance(fit: LogLogisticFit, target: float = 0.5):
    """
    The profile deviance of the absolute dose, as a callable.

    Returns `(deviance, estimate, reason)` where `deviance(dose)` is

        D(L) = 2 * ( loglik_max - max_phi loglik(L, phi) )

    the likelihood-ratio statistic for the hypothesis that the dose at which
    the curve reaches `target` is exactly L, with every other parameter
    re-maximised. `D` is zero at the maximum-likelihood dose and increases away
    from it; `lethal_dose_profile` finds where it crosses a threshold, and a
    calibration study can use it to check that the threshold matches the
    statistic's own distribution rather than assuming it does.

    `deviance` is None, and `reason` says why, when the fit cannot be profiled.
    """
    out_reason = ""
    if fit.method != "binomial_ml":
        return None, float("nan"), (
            "Profiling needs a likelihood, so it needs method='binomial_ml'."
        )
    if not np.isfinite(fit.loglik):
        return None, float("nan"), "The fit reports no finite log-likelihood."

    t = float(target)
    b = float(fit.params["b"])
    c = float(fit.params["c"])
    d = float(fit.params["d"])
    e = float(fit.params["e"])
    if not np.isfinite(b) or b == 0.0:
        return None, float("nan"), "The fitted slope is zero or not finite."
    if not (min(c, d) < t < max(c, d)):
        return None, float("nan"), (
            f"The fitted curve runs between c = {c:.4g} and d = {d:.4g} and never "
            f"reaches {t:g}."
        )
    k, nn = fit.counts, fit.totals
    if k is None or nn is None:
        return None, float("nan"), "The fit did not retain its counts."
    # The profile must be of the objective that was MAXIMISED. Profiling the
    # plain likelihood around a penalised maximum would compare two different
    # functions and the deviance at the estimate would not be zero.
    pen = getattr(fit, "penalty", "none")

    est = e * ((d - t) / (t - c)) ** (1.0 / b)
    model = fit.model
    free = list(fit.theta_names)
    x = fit.dose
    prof_names = [nm for nm in free if nm != "log_e"]
    # The profile must search the same parameter region the fit did, or the
    # deviance is a ratio between two differently constrained maxima.
    cap = getattr(fit, "slope_cap", float("nan"))
    bounds_all = _ml_bounds(model, cap if np.isfinite(cap) else None)
    prof_bounds = [bounds_all[free.index(nm)] for nm in prof_names]
    fixed = _MODEL_SPECS[model]["fixed"]

    def negll_at(logL, phi):
        vals = dict(zip(prof_names, np.atleast_1d(phi)))
        bb = vals.get("b", float(fixed.get("b", b)))
        cc = vals.get("c", float(fixed.get("c", c)))
        dd = vals.get("d", float(fixed.get("d", d)))
        num, den = dd - t, t - cc
        if not np.isfinite(bb) or bb == 0.0 or num <= 0.0 or den <= 0.0:
            return np.inf
        log_e_c = float(logL) - (np.log(num) - np.log(den)) / bb
        theta = np.empty(len(free), dtype=float)
        for i, nm in enumerate(free):
            theta[i] = log_e_c if nm == "log_e" else vals[nm]
        return _binom_negll_pen(theta, model, x, k, nn, pen)

    mle_start = np.array([float(fit.params[nm]) for nm in prof_names], dtype=float)
    warm = {"phi": mle_start.copy()}
    lo_b = [(-np.inf if lo is None else lo) for lo, _ in prof_bounds]
    hi_b = [(np.inf if hi is None else hi) for _, hi in prof_bounds]

    def profile_negll(logL, record=True):
        if not prof_names:
            return negll_at(logL, np.array([]))
        best_v, best_phi = np.inf, None
        for s0 in (warm["phi"], mle_start):
            s0 = np.clip(np.asarray(s0, dtype=float), lo_b, hi_b)
            try:
                r = optimize.minimize(lambda ph: negll_at(logL, ph), s0,
                                      method="L-BFGS-B", bounds=prof_bounds,
                                      options={"maxiter": 2000, "maxfun": 2000,
                                               "ftol": 1e-14, "gtol": 1e-10})
            except Exception:
                continue
            if np.isfinite(r.fun) and float(r.fun) < best_v:
                best_v, best_phi = float(r.fun), np.asarray(r.x, dtype=float)
        # BOTH starts are tried, and the better kept. Stopping at the first
        # that converged made the deviance path-dependent: the warm start can
        # settle in a slightly worse nested optimum, which inflates the
        # deviance, which pulls the interval in. Measured at 3.6917 against a
        # threshold of 3.8415 on a reported bound before this change.
        if record and best_phi is not None:
            warm["phi"] = best_phi
        return best_v

    const = float(np.sum(gammaln(nn + 1.0) - gammaln(k + 1.0)
                         - gammaln(nn - k + 1.0)))
    ll_max = float(fit.loglik_penalised if pen == "firth" else fit.loglik)

    def deviance(dose, record=True):
        """Profile deviance at a candidate DOSE (not log dose)."""
        dose = float(dose)
        if not (dose > 0.0) or not np.isfinite(dose):
            return np.inf
        nll = profile_negll(np.log(dose), record=record)
        if not np.isfinite(nll):
            return np.inf
        return 2.0 * (ll_max - (const - nll))

    return deviance, float(est), out_reason


def lethal_dose_profile(fit: LogLogisticFit, target: float = 0.5,
                        level: float = 0.95, max_expand: float = 1e6,
                        calibrate: str = "asymptotic", n_boot: int = 200,
                        seed: Optional[int] = None) -> dict:
    """
    PROFILE-LIKELIHOOD interval for the dose at which the fitted curve reaches
    an absolute response level -- the LC50 when `target = 0.5`.

    Register N95. The delta-method interval is symmetric on log-dose and is
    built from a covariance matrix. Both of those are wrong here. The
    likelihood for an absolute LC50 is asymmetric on any scale, because the
    data constrain "the curve has not reached 50% yet" much more weakly than
    "it passed 50% already", and a covariance cannot encode that. Profiling the
    quantity of interest is invariant to how the curve is parameterised, which
    is the property the delta method lacks.

    THE REPARAMETERISATION, which is what makes this exact rather than a grid
    search over a derived quantity. The absolute dose satisfies

        L = e * K ** (1 / b),    K = (d - target) / (target - c)

    so for any CANDIDATE L the inflection is determined by the others:

        log e = log L - log(K) / b

    Fixing L therefore costs exactly one degree of freedom, and the remaining
    free parameters are re-maximised. The deviance is compared against a
    threshold on ONE degree of freedom and the endpoints are found with Brent's
    method, so they sit on the threshold to the root-finder's tolerance rather
    than on a grid point.

    THE THRESHOLD. The asymptotic threshold is chi-square(1). On the designs a
    fly bioassay actually has -- seven doses, three residual degrees of freedom
    -- that is anti-conservative, and the correction used here is measured
    rather than assumed: see `_lr_threshold` and register N95.

    Requires `method="binomial_ml"`: the deviance is a likelihood-ratio
    statistic and needs a likelihood.

    Returns `estimate`, `ci_lower`, `ci_upper`, `loglik_at_lower`,
    `loglik_at_upper`, `deviance_threshold`, `threshold_kind`, `level`,
    `target`, `reason`, `bounded_below`, `bounded_above`.
    """
    thr, kind = _lr_threshold(fit, level, calibrate=calibrate, n_boot=n_boot,
                              seed=seed)
    out = {"target": float(target), "level": float(level),
           "estimate": float("nan"), "ci_lower": float("nan"),
           "ci_upper": float("nan"), "loglik_at_lower": float("nan"),
           "loglik_at_upper": float("nan"),
           "deviance_threshold": float(thr), "threshold_kind": kind,
           "bounded_below": True, "bounded_above": True, "reason": "",
           "threshold_advice": threshold_advice(fit, target),
           "method": ("Profile-likelihood interval on the absolute dose, "
                      "reparameterised so the dose is the profiled parameter; "
                      + kind + " deviance threshold.")}

    dev, est, why = lc_profile_deviance(fit, target)
    if dev is None:
        out["reason"] = why
        return out
    out["estimate"] = float(est)

    d0 = dev(est, record=False)
    if d0 < -1e-4:
        out["reason"] = (
            f"The profile found a higher likelihood than the reported maximum "
            f"(deviance {d0:.3e} at the estimate), so the fit is not at its "
            "optimum and no interval is reported."
        )
        return out

    log_est = float(np.log(est))

    def dev_log(v):
        return dev(float(np.exp(v)))

    def find_root(direction):
        """Expand a bracket outward until the threshold is crossed, then solve."""
        step = 0.1
        lo = log_est
        hi = log_est + direction * step
        for _ in range(200):
            if dev_log(hi) >= thr:
                break
            lo = hi
            step *= 1.6
            hi = log_est + direction * step
            if abs(hi - log_est) > np.log(max_expand):
                return None
        else:
            return None
        a, bb = (lo, hi) if lo < hi else (hi, lo)
        try:
            return float(optimize.brentq(lambda v: dev_log(v) - thr, a, bb,
                                         xtol=1e-10, rtol=1e-12, maxiter=200))
        except Exception:
            for _ in range(80):
                mid = 0.5 * (a + bb)
                if dev_log(mid) >= thr:
                    bb = mid
                else:
                    a = mid
                if abs(bb - a) < 1e-10:
                    break
            return 0.5 * (a + bb)

    r_lo = find_root(-1.0)
    r_hi = find_root(+1.0)
    if r_lo is None:
        out["bounded_below"] = False
        out["reason"] = (
            "The likelihood does not rise above the deviance threshold below "
            "the estimate within a factor of %g, so the interval is unbounded "
            "there: the data do not exclude an arbitrarily low dose." % max_expand
        )
    else:
        out["ci_lower"] = float(np.exp(r_lo))
    if r_hi is None:
        out["bounded_above"] = False
        out["reason"] = (out["reason"] + " " if out["reason"] else "") + (
            "The likelihood does not rise above the threshold above the "
            "estimate within a factor of %g, so the interval is unbounded "
            "there." % max_expand
        )
    else:
        out["ci_upper"] = float(np.exp(r_hi))

    # the log-likelihood at each reported bound, recomputed from the same
    # deviance so the two can never disagree
    if np.isfinite(out["ci_lower"]):
        out["loglik_at_lower"] = float(fit.loglik
                                       - 0.5 * dev(out["ci_lower"], record=False))
    if np.isfinite(out["ci_upper"]):
        out["loglik_at_upper"] = float(fit.loglik
                                       - 0.5 * dev(out["ci_upper"], record=False))
    return out


def ec50_profile_ci(fit: LogLogisticFit, level: float = 0.95) -> dict:
    """
    PROFILE-LIKELIHOOD confidence interval for EC50, with the delta-method
    interval reported alongside.

    Why not the delta method. EC50 enters the model non-linearly and its
    sampling distribution on the dose scale is right-skewed -- the data
    constrain "the curve has not turned over yet" much more weakly than they
    constrain "it turned over already", so the upper bound is far from the
    estimate and the lower bound is close to it. A symmetric interval
    estimate +/- t*SE cannot represent that. In practice it is too narrow
    above, too wide below, and when the slope is shallow or the top of the
    curve is poorly determined it returns a NEGATIVE lower bound, which is not
    a dose.

    What is computed instead. The residual sum of squares is profiled over
    log(EC50): at each candidate value EC50 is held fixed and the remaining
    parameters are re-optimised, giving RSS_p(EC50). The interval is every
    EC50 for which

        RSS_p(EC50) <= RSS_min * (1 + (1 / (n - p)) * F(1, n - p, level))

    i.e. the set not rejected by an F-test of the constraint. The bound
    walking is done on log-dose so the search is scale-free.

    Returns a dict with `profile` (lower/upper), `delta` (symmetric dose-scale
    and log-scale versions), the asymmetry ratio, and a note stating which to
    report. If a bound cannot be reached within the search span it is reported
    as 0 or inf and listed in `profile["unbounded"]` -- an honest "the data do
    not bound this side", not a fabricated finite number.
    """
    prof = _profile_interval(fit, level, 50.0)
    delt = _delta_interval(fit, level, 50.0)
    e = float(fit.params["e"])

    lo, hi = prof["lower"], prof["upper"]
    left = e - lo
    right = hi - e
    asym = float(right / left) if np.isfinite(right) and left > 0 else float("inf")

    note = (
        f"Report the profile interval [{lo:.4g}, {hi:.4g}]. The delta-method "
        f"interval [{delt['lower']:.4g}, {delt['upper']:.4g}] is symmetric by "
        f"construction; the profile interval is {asym:.2f}x wider above the "
        f"estimate than below it."
    )
    if np.isfinite(delt["lower"]) and delt["lower"] <= 0:
        note += (
            "  The delta-method lower bound is <= 0, which is not a possible dose "
            "-- a direct demonstration that it should not be used here."
        )

    return {
        "ec50": e,
        "level": float(level),
        "profile": prof,
        "delta": delt,
        "asymmetry_ratio": asym,
        "method": "Profile likelihood, F-test inversion (Seber & Wild 1989 ch. 5)",
        "note": note,
    }


def effective_dose(fit: LogLogisticFit, levels: Sequence[float] = (10, 50, 90),
                   level: float = 0.95) -> pd.DataFrame:
    """
    Effective doses ECx for arbitrary response levels, with intervals.

    ECx is the RELATIVE effective dose: the dose producing a change of x% of
    the fitted range (d - c), which is the definition used by `drc` and the
    one that makes EC50 equal to `e`. In closed form,

        f(ECx) = d - (x/100) * (d - c)   =>   ECx = e * (x / (100 - x))^(1/b)

    This is a relative, not an absolute, statement: EC90 is 90% of the way
    from the control level to the fitted lower plateau, not 90% inhibition
    from an absolute zero. When the lower plateau `c` is far from zero those
    are very different quantities, so if absolute inhibition is what is meant,
    fit LL.3 (c = 0) instead.

    Both interval types are returned for every level: the profile-likelihood
    interval (recommended) and the delta-method interval. Extreme levels are
    where they diverge most -- EC90 sits on the flat tail of the curve, so its
    profile interval is markedly skewed and often unbounded above, which the
    delta method conceals behind a tidy symmetric pair of numbers.

    Returns a DataFrame indexed by level with columns `estimate`, `std_err`,
    `profile_lower`, `profile_upper`, `delta_lower`, `delta_upper`,
    `delta_log_lower`, `delta_log_upper`, `extrapolated`.
    """
    max_dose = float(np.max(fit.dose)) if fit.dose.size else np.nan
    min_nz = fit.dose[fit.dose > 0]
    min_dose = float(np.min(min_nz)) if min_nz.size else np.nan

    rows = []
    for x_pct in levels:
        x_pct = float(x_pct)
        if not (0.0 < x_pct < 100.0):
            raise ValueError(f"ECx levels must be strictly between 0 and 100; got {x_pct}")
        prof = _profile_interval(fit, level, x_pct)
        delt = _delta_interval(fit, level, x_pct)
        est = delt["estimate"]
        rows.append({
            "level": x_pct,
            "estimate": est,
            "std_err": delt["se"],
            "profile_lower": prof["lower"],
            "profile_upper": prof["upper"],
            "delta_lower": delt["lower"],
            "delta_upper": delt["upper"],
            "delta_log_lower": delt["log_lower"],
            "delta_log_upper": delt["log_upper"],
            "extrapolated": bool(np.isfinite(est) and (est > max_dose or est < min_dose)),
        })
    out = pd.DataFrame(rows).set_index("level")
    out.attrs["note"] = (
        "ECx is relative to the fitted range (d - c). `extrapolated` is True when the "
        "estimate falls outside the range of non-zero doses actually tested, in which "
        "case it is a property of the assumed curve shape, not an observation."
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. HORMESIS
# ═══════════════════════════════════════════════════════════════════════════
def hormesis_test(dose: Sequence[float], response: Sequence[float],
                  n: Optional[Sequence[float]] = None,
                  alpha: float = 0.05) -> dict:
    """
    Test for low-dose stimulation (hormesis) with the Brain-Cousens model.

    Non-monotonic dose-response is not a curiosity. Across toxicology,
    responses that are stimulated at low dose and inhibited at high dose are
    routinely reported (Calabrese & Baldwin 2003), and a monotone Hill fit has
    no way to represent the hump: it splits the difference, pulling the upper
    asymptote up and the EC50 out. Fitting a monotone model to hormetic data
    therefore biases the potency estimate and hides a real biological finding.

    The Brain-Cousens (1989) model adds one parameter to LL.4,

        g(x) = c + (d - c + f*x) / (1 + exp(b * (log(x) - log(e))))

    which for small x behaves like d + f*x: f > 0 lifts the response above the
    control level before the sigmoid takes over. Because LL.4 is exactly
    g with f = 0, the two models are nested and the extra parameter is tested
    with the usual extra-sum-of-squares F test,

        F = ((RSS_LL4 - RSS_BC) / 1) / (RSS_BC / (n - 5)),   df = (1, n - 5)

    Hormesis is reported as SUPPORTED only when that test is significant AND
    the estimated f is positive; a significant negative f means the low-dose
    end is being pulled down, which is a lack-of-fit signal, not hormesis.

    Caveats stated in the output: with f != 0 the parameter `e` in the
    Brain-Cousens fit is a location parameter and NOT the EC50, so potency is
    still reported from the monotone fit; and the test needs doses low enough
    to see the stimulation -- absence of evidence here is mostly evidence that
    the lowest dose was not low enough.

    Returns a dict with both fits, the F statistic and p-value, the estimated
    f, the dose and magnitude of maximum stimulation, and `hormesis_supported`.
    """
    x = np.asarray(dose, dtype=float)
    y = np.asarray(response, dtype=float)
    w = np.ones_like(x) if n is None else np.asarray(n, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]

    mono = fit_loglogistic(x, y, n=None if n is None else w, model="LL.4")

    n_obs = int(x.size)
    p_full = 5
    if n_obs <= p_full:
        return {
            "usable": False,
            "reason": (f"{n_obs} observations cannot support the 5-parameter "
                       "Brain-Cousens model."),
            "monotone_fit": mono,
        }
    nz = np.unique(x[x > 0])
    if nz.size < 5:
        return {
            "usable": False,
            "reason": (f"{nz.size} distinct non-zero doses; the 5-parameter "
                       "Brain-Cousens model needs at least 5."),
            "monotone_fit": mono,
        }

    sw = np.sqrt(w)

    def resid_bc(theta: np.ndarray) -> np.ndarray:
        b, c, d, log_e, f = theta
        return sw * (_brain_cousens(x, b, c, d, log_e, f) - y)

    theta0_base = np.array([
        float(mono.params["b"]), float(mono.params["c"]), float(mono.params["d"]),
        float(np.log(mono.params["e"])),
    ])
    scale = max(np.ptp(y), 1e-9) / max(np.max(x[x > 0]) if np.any(x > 0) else 1.0, 1e-9)

    best, best_rss = None, np.inf
    for f0 in (0.0, 0.25 * scale, 1.0 * scale, -0.25 * scale):
        try:
            r = _lsq(resid_bc, np.append(theta0_base, f0))
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if np.isfinite(rss) and rss < best_rss - 1e-14:
            best, best_rss = r, rss
    if best is None:
        return {"usable": False, "reason": "Brain-Cousens fit did not converge.",
                "monotone_fit": mono}

    rss_bc = best_rss
    rss_ll4 = float(mono.rss)
    df2 = n_obs - p_full
    denom = rss_bc / df2
    if denom <= 0 or not np.isfinite(denom):
        return {"usable": False, "reason": "Zero residual variance under the full model.",
                "monotone_fit": mono}

    f_stat = float(max(rss_ll4 - rss_bc, 0.0) / denom)
    p_value = float(stats.f.sf(f_stat, 1, df2))

    b_bc, c_bc, d_bc, log_e_bc, f_bc = (float(v) for v in best.x)
    J = np.asarray(best.jac, dtype=float)
    try:
        cov_bc = (rss_bc / df2) * np.linalg.inv(J.T @ J)
        se_f = float(np.sqrt(max(cov_bc[4, 4], 0.0)))
    except np.linalg.LinAlgError:
        cov_bc, se_f = np.full((5, 5), np.nan), float("nan")

    # Where and how large is the stimulation, if any?
    max_dose = float(np.max(x)) if x.size else np.nan
    grid = np.linspace(1e-6 * max(max_dose, 1e-6), max_dose, 4001)[1:]
    pred = _brain_cousens(grid, b_bc, c_bc, d_bc, log_e_bc, f_bc)
    control_level = d_bc
    i_max = int(np.argmax(pred))
    stim_dose = float(grid[i_max])
    stim_pct = float(100.0 * (pred[i_max] - control_level) / abs(control_level)) \
        if control_level != 0 else float("nan")
    has_hump = bool(pred[i_max] > control_level + 1e-12 and stim_dose < float(np.exp(log_e_bc)))

    supported = bool(p_value < alpha and f_bc > 0 and has_hump)

    if supported:
        verdict = (
            f"HORMESIS SUPPORTED (F(1,{df2}) = {f_stat:.3f}, p = {p_value:.4g}, "
            f"f = {f_bc:.4g} +/- {se_f:.4g}). The response peaks {stim_pct:.1f}% above "
            f"the control level at a dose of about {stim_dose:.4g}. The monotone LL.4 "
            "EC50 is biased under this alternative and should not be reported alone."
        )
    elif p_value < alpha and f_bc <= 0:
        verdict = (
            f"The extra Brain-Cousens parameter is significant (p = {p_value:.4g}) but "
            f"f = {f_bc:.4g} is NEGATIVE. That is not hormesis -- it is the low-dose end "
            "of the curve being pulled down, i.e. a lack-of-fit signal for LL.4."
        )
    else:
        verdict = (
            f"No support for hormesis (F(1,{df2}) = {f_stat:.3f}, p = {p_value:.4g}). "
            "Note this is only informative if the lowest tested doses were low enough "
            f"for a stimulatory phase to appear; the lowest non-zero dose here was "
            f"{float(np.min(x[x > 0])) if np.any(x > 0) else float('nan'):.4g}."
        )

    return {
        "usable": True,
        "hormesis_supported": supported,
        "f_statistic": f_stat,
        "p_value": p_value,
        "df": (1, int(df2)),
        "rss_ll4": rss_ll4,
        "rss_brain_cousens": rss_bc,
        "f_param": f_bc,
        "f_param_se": se_f,
        "bc_params": {"b": b_bc, "c": c_bc, "d": d_bc,
                      "e": float(np.exp(log_e_bc)), "f": f_bc},
        "max_stimulation_dose": stim_dose if has_hump else float("nan"),
        "max_stimulation_percent": stim_pct if has_hump else float("nan"),
        "monotone_fit": mono,
        "monotone_ec50": float(mono.params["e"]),
        "verdict": verdict,
        "caveat": (
            "In the Brain-Cousens fit `e` is a location parameter, not the EC50; "
            "potency should be read from the monotone fit (or from an explicitly "
            "hormesis-aware ED calculation), never from bc_params['e']."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. RELATIVE POTENCY
# ═══════════════════════════════════════════════════════════════════════════
def relative_potency(fit_a: LogLogisticFit, fit_b: LogLogisticFit,
                     level: float = 0.95,
                     label_a: str = "A", label_b: str = "B") -> dict:
    """
    Relative potency EC50_A / EC50_B, with a confidence interval, for
    comparing two compounds or two genotypes.

    The ratio is formed on the LOG scale, where both fits estimate log(EC50)
    directly and the sampling distribution is far closer to normal:

        log R = log e_A - log e_B
        Var(log R) = Var(log e_A) + Var(log e_B)      (independent fits)
        CI(R) = exp( log R  +/-  z * SE(log R) )

    Exponentiating guarantees a positive interval and reproduces the standard
    asymmetry of a ratio -- which a delta-method interval formed directly on
    the ratio does not. The variances are added because the two curves come
    from separate experiments; if the two arms were fitted jointly with shared
    parameters this would need the cross-covariance instead, and this function
    would be the wrong tool.

    PARALLELISM. A single relative potency is only meaningful when the two
    curves are PARALLEL, i.e. share a slope b (Finney 1971). If the slopes
    differ, the potency ratio depends on which response level you ask about --
    compound A can be more potent at EC10 and less potent at EC90 -- and
    quoting one number for it is wrong. A Wald test of b_A = b_B is therefore
    run and reported, and when it rejects, the potency ratio at EC10/EC50/EC90
    is returned so the dose-dependence is visible.

    Returns a dict with the ratio, its interval, the parallelism test, and a
    verdict string.
    """
    ea, eb = float(fit_a.params["e"]), float(fit_b.params["e"])
    va = float(fit_a.cov_theta[fit_a.log_e_index, fit_a.log_e_index])
    vb = float(fit_b.cov_theta[fit_b.log_e_index, fit_b.log_e_index])

    log_r = float(np.log(ea) - np.log(eb))
    var_log_r = va + vb
    se_log_r = float(np.sqrt(var_log_r)) if np.isfinite(var_log_r) and var_log_r >= 0 else np.nan
    crit = float(stats.norm.ppf(0.5 + level / 2.0))

    ratio = _safe_exp(log_r)
    lo = _safe_exp(log_r - crit * se_log_r) if np.isfinite(se_log_r) else np.nan
    hi = _safe_exp(log_r + crit * se_log_r) if np.isfinite(se_log_r) else np.nan

    z_r = log_r / se_log_r if np.isfinite(se_log_r) and se_log_r > 0 else np.nan
    p_r = float(2.0 * stats.norm.sf(abs(z_r))) if np.isfinite(z_r) else np.nan

    # ── parallelism ────────────────────────────────────────────────────────
    ba, bb = float(fit_a.params["b"]), float(fit_b.params["b"])
    sea, seb = float(fit_a.bse["b"]), float(fit_b.bse["b"])
    se_diff = float(np.sqrt(sea ** 2 + seb ** 2)) if np.isfinite(sea) and np.isfinite(seb) else np.nan
    z_b = (ba - bb) / se_diff if np.isfinite(se_diff) and se_diff > 0 else np.nan
    p_b = float(2.0 * stats.norm.sf(abs(z_b))) if np.isfinite(z_b) else np.nan
    parallel = bool(np.isfinite(p_b) and p_b >= 0.05)

    by_level = {}
    for x_pct in (10.0, 50.0, 90.0):
        k = x_pct / (100.0 - x_pct)
        by_level[x_pct] = float((ea * k ** (1.0 / ba)) / (eb * k ** (1.0 / bb)))

    verdict = (
        f"{label_a} EC50 = {ea:.4g}, {label_b} EC50 = {eb:.4g}; relative potency "
        f"{label_a}/{label_b} = {ratio:.3g} [{lo:.3g}, {hi:.3g}] at {level:.0%}. "
        + ("The interval excludes 1, so the potencies differ."
           if np.isfinite(lo) and np.isfinite(hi) and not (lo <= 1.0 <= hi)
           else "The interval includes 1, so the potencies are not distinguishable.")
    )
    if not parallel:
        verdict += (
            f"  WARNING: the slopes differ (b = {ba:.3g} vs {bb:.3g}, p = {p_b:.3g}), "
            "so the curves are NOT parallel and a single potency ratio is not "
            f"well defined -- it is {by_level[10.0]:.3g} at EC10 and "
            f"{by_level[90.0]:.3g} at EC90. Report the level-specific ratios."
        )

    return {
        "ratio": ratio,
        "log_ratio": log_r,
        "se_log_ratio": se_log_r,
        "ci_lower": lo,
        "ci_upper": hi,
        "level": float(level),
        "z": z_r,
        "p_value": p_r,
        "ec50_a": ea,
        "ec50_b": eb,
        "slope_a": ba,
        "slope_b": bb,
        "parallel": parallel,
        "parallelism_z": z_b,
        "parallelism_p": p_b,
        "ratio_by_level": by_level,
        "labels": (label_a, label_b),
        "method": "Delta method on log(EC50_A) - log(EC50_B); Finney (1971) parallelism check",
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. THE MORTALITY CONFOUND -- the diagnostic that matters
# ═══════════════════════════════════════════════════════════════════════════
def mortality_adjusted_response(
    dose: Sequence[float],
    response: Sequence[float],
    dead: Sequence[float],
    n_exposed: Sequence[float],
    n_scored: Optional[Sequence[float]] = None,
    mortality_threshold: float = 0.10,
    ratio_warn: float = 0.33,
    ratio_fail: float = 0.80,
    level: float = 0.95,
    mortality_model: str = "LL.3",
    mortality_method: str = "binomial_ml",
) -> dict:
    """
    Separate -- or refuse to separate -- a behavioural dose-response from
    lethality.

    THE PROBLEM. Climbing, grooming and Y-maze scores are measured on flies
    that are alive. If a dose kills 40% of the vial, the 60% scored are not a
    random sample of the exposed flies; they are the survivors. Any statistic
    computed on them is conditional on survival, and the causal quantity
    people want -- "what does this toxin do to climbing?" -- is not
    identified from it. Two completely different worlds produce identical
    numbers:

        (i)  the toxin specifically impairs motor function, and
        (ii) the toxin is simply poisoning flies, and the ones not yet dead at
             the time of scoring are moribund.

    No amount of statistics on the survivor data distinguishes them. What CAN
    be done, and what this function does, is quantify how much of the
    behavioural curve lies inside the lethal range, and say plainly when the
    two cannot be told apart.

    WHAT IS COMPUTED.
      (a) The naive behavioural dose-response among survivors: an LL.4 fit of
          `response` on `dose`, weighted by the number of flies scored. This is
          exactly the curve a conventional analysis would report, and it is
          returned so the user can see the number they would otherwise have
          published.
      (b) The mortality dose-response. Survival proportion (1 - dead/n_exposed)
          is fitted as a decreasing curve, and the LC50 is obtained by solving
          that curve for survival = 0.5. The default LL.3 leaves the upper
          asymptote free, which lets the control survival be below 1 -- that is
          precisely Abbott's (1925) correction expressed as a model parameter
          rather than applied by hand -- and for exactly that reason the fitted
          `e` is NOT the LC50: it is the dose at which survival reaches the
          midpoint (c + d)/2 of the fitted range. `e` is reported separately as
          `lc50_param_e` for traceability. Use "LL.2" to force control survival
          to exactly 1, in which case the two coincide.
      (c) The overlap. Per-dose mortality, the highest dose at which mortality
          stays below `mortality_threshold` (the "behaviour-only window"), and
          the ratio EC50_behaviour / LC50.

    HOW TO READ THE RATIO. The three bands below are SCREENING CONVENTIONS,
    not tests: they have no error rate, and the per-dose mortality table is the
    evidence. They are also a comparison between two differently-defined
    quantities -- EC50 is relative to the behavioural curve's own fitted range,
    LC50 is an absolute 50%-survival dose -- so treat the ratio as a question
    about whether the two curves occupy the same dose region, not as a
    measurement.
        ratio < 0.33   the behavioural effect is half-maximal at a dose that
                       kills almost nothing; a specific behavioural phenotype
                       is defensible.
        0.33 - 0.80    caution: a substantial part of the behavioural curve is
                       inside the lethal range.
        >= 0.80        the behavioural EC50 and the LC50 are effectively the
                       same dose. The "behavioural effect" cannot be
                       distinguished from general lethality and must NOT be
                       reported as a behavioural phenotype.

    Thresholds are conventions, not tests, and are exposed as arguments; the
    per-dose mortality table is the primary evidence and is always returned.

    Parameters
    ----------
    dose, response : the behavioural assay, one row per vial (or per dose).
    dead : number of flies dead at scoring, same rows.
    n_exposed : number of flies originally exposed, same rows.
    n_scored : number of flies actually scored behaviourally. Defaults to
        n_exposed - dead, which is the usual survivors-only design.
    mortality_model : "LL.3" (default, control survival free = Abbott) or
        "LL.2" (control survival fixed at 1).

    Returns a dict with `behaviour_fit`, `mortality_fit`, `lc50`, `ec50`,
    `ratio`, `per_dose` (a DataFrame), `confounded`, `severity` and `warning`.
    """
    x = np.asarray(dose, dtype=float)
    y = np.asarray(response, dtype=float)
    dd = np.asarray(dead, dtype=float)
    ne = np.asarray(n_exposed, dtype=float)
    if not (x.shape == y.shape == dd.shape == ne.shape):
        raise ValueError("dose, response, dead and n_exposed must all be the same length.")
    ns = (ne - dd) if n_scored is None else np.asarray(n_scored, dtype=float)
    if ns.shape != x.shape:
        raise ValueError("n_scored must be the same length as dose.")

    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(dd) & np.isfinite(ne) & (ne > 0))
    x, y, dd, ne, ns = x[ok], y[ok], dd[ok], ne[ok], ns[ok]
    if np.any(dd > ne):
        raise ValueError("`dead` exceeds `n_exposed` in at least one row.")

    surv_prop = 1.0 - dd / ne

    out: dict = {"level": float(level), "mortality_threshold": float(mortality_threshold)}

    # ── (a) behavioural curve among survivors ──────────────────────────────
    beh_weights = np.where(np.isfinite(ns) & (ns > 0), ns, np.nan)
    keep = np.isfinite(beh_weights) & (beh_weights > 0)
    use_w = bool(np.any(keep) and np.ptp(beh_weights[keep]) > 0)
    try:
        beh = fit_loglogistic(x[keep], y[keep],
                              n=beh_weights[keep] if use_w else None, model="LL.4")
        ec50 = float(beh.params["e"])
        beh_ci = ec50_profile_ci(beh, level=level)
    except Exception as exc:
        beh, ec50, beh_ci = None, float("nan"), None
        out["behaviour_error"] = str(exc)

    # ── (b) mortality curve, fitted as SURVIVAL ────────────────────────────
    # Register N14. The LC50 is the dose at which SURVIVAL = 0.5. It is NOT the
    # fitted `e`, which is the dose at which survival is (c + d)/2 -- and the
    # default LL.3 leaves d free precisely so that control survival may be
    # below 1. Reading `e` as the LC50 overstated it by 11.9% at d = 0.90 and
    # 67% at d = 0.80, and that error propagated straight into the EC50/LC50
    # overlap verdict below.
    max_mort = float(np.max(dd / ne)) if x.size else float("nan")
    mort, lc50, mort_ci = None, float("nan"), None
    lc50_e, lc50_def, lc50_reason = float("nan"), "", ""
    mort_error = None
    if max_mort < 0.5:
        mort_error = (
            f"Maximum observed mortality is {max_mort:.1%}; survival never falls "
            "below 50% at any tested dose, so the LC50 is above the tested range "
            "and cannot be estimated without extrapolating."
        )
    else:
        try:
            mort = fit_loglogistic(x, surv_prop, n=ne, model=mortality_model,
                                   method=mortality_method)
            lc50_e = float(mort.params["e"])
            ld = lethal_dose_absolute(mort, target=0.5, level=level)
            lc50 = float(ld["estimate"])
            lc50_def, lc50_reason = ld["definition"], ld["reason"]
            # Register N95. The mortality curve is always fitted with cohort
            # sizes, so the interval below is always the under-covering one.
            # Carrying the caveat out of here is the only way it reaches the
            # user: this function, not `lethal_dose_absolute`, is what the
            # dose-response page calls.
            out["lc50_interval_caveat"] = ld.get("interval_caveat")
            out["lc50_interval_coverage_nominal"] = ld.get(
                "interval_coverage_nominal")
            mort_ci = {
                "lc50": lc50,
                "lower": ld["ci_lower"],
                "upper": ld["ci_upper"],
                "level": float(level),
                "method": ld.get("method", ""),
                "interval_caveat": ld.get("interval_caveat"),
                "interval_coverage_nominal": ld.get("interval_coverage_nominal"),
                "profile_relative": ld.get("profile"),
                "param_e_profile": ec50_profile_ci(mort, level=level),
            }
            if ld["reason"]:
                out["lc50_refusal"] = ld["reason"]
        except Exception as exc:
            mort_error = str(exc)
    if mort_error:
        out["mortality_error"] = mort_error

    # ── (c) per-dose overlap table ─────────────────────────────────────────
    tab = pd.DataFrame({"dose": x, "dead": dd, "n_exposed": ne,
                        "n_scored": ns, "response": y})
    per_dose = tab.groupby("dose").agg(
        n_exposed=("n_exposed", "sum"),
        n_dead=("dead", "sum"),
        n_scored=("n_scored", "sum"),
        mean_response=("response", "mean"),
    )
    per_dose["mortality"] = per_dose["n_dead"] / per_dose["n_exposed"]
    per_dose["survivor_conditional"] = per_dose["mortality"] >= mortality_threshold
    per_dose = per_dose[["n_exposed", "n_dead", "mortality", "n_scored",
                         "mean_response", "survivor_conditional"]]

    clean = per_dose.index[~per_dose["survivor_conditional"]]
    max_clean_dose = float(np.max(clean)) if len(clean) else float("nan")
    n_confounded_doses = int(per_dose["survivor_conditional"].sum())

    ratio = float(ec50 / lc50) if (np.isfinite(ec50) and np.isfinite(lc50) and lc50 > 0) \
        else float("nan")

    # ── verdict ────────────────────────────────────────────────────────────
    if not np.isfinite(ratio):
        if np.isfinite(max_mort) and max_mort < mortality_threshold:
            severity = "clean"
            confounded = False
            warning = (
                f"Maximum mortality across all doses was {max_mort:.1%}, below the "
                f"{mortality_threshold:.0%} threshold. There is no meaningful survivor "
                "selection, and the behavioural curve can be interpreted as a "
                "behavioural effect. The LC50 is above the tested range and is not "
                "estimated -- that is the reassuring outcome, not a failure."
            )
        else:
            severity = "indeterminate"
            confounded = bool(n_confounded_doses > 0)
            warning = (
                f"EC50/LC50 could not be formed ({out.get('mortality_error') or out.get('behaviour_error')}). "
                f"{n_confounded_doses} of {len(per_dose)} dose levels still exceed "
                f"{mortality_threshold:.0%} mortality; at those doses the behavioural "
                "estimate is survivor-conditional regardless."
            )
    elif ratio >= ratio_fail:
        severity = "confounded"
        confounded = True
        warning = (
            f"CONFOUNDED WITH LETHALITY. EC50(behaviour) = {ec50:.4g} and LC50 = "
            f"{lc50:.4g}, a ratio of {ratio:.2f}. The dose that halves the behavioural "
            "score is essentially the dose that kills half the flies, so the two "
            "processes are not separable in this experiment. The behavioural curve is "
            "measured on survivors -- a non-random subset of the exposed population -- "
            "and MUST NOT be reported as evidence of a specific behavioural or "
            "neurological phenotype. To make that claim, extend the design downward "
            "until there are several doses with a clear behavioural effect and "
            f"negligible mortality (currently {n_confounded_doses} of {len(per_dose)} "
            f"dose levels exceed {mortality_threshold:.0%} mortality)."
        )
    elif ratio >= ratio_warn:
        severity = "overlapping"
        confounded = True
        warning = (
            f"PARTIAL OVERLAP WITH LETHALITY. EC50(behaviour) = {ec50:.4g}, LC50 = "
            f"{lc50:.4g}, ratio {ratio:.2f}. A substantial part of the behavioural "
            "curve lies inside the lethal range. At the "
            f"{n_confounded_doses} dose level(s) with mortality at or above "
            f"{mortality_threshold:.0%}, the behavioural estimate is "
            "SURVIVOR-CONDITIONAL and cannot be read as a behavioural effect. "
            + (f"Only doses at or below {max_clean_dose:.4g} are interpretable "
               "behaviourally." if np.isfinite(max_clean_dose)
               else "No tested dose is free of substantial mortality.")
        )
    else:
        severity = "clean"
        confounded = False
        warning = (
            f"SEPARABLE. EC50(behaviour) = {ec50:.4g} is {1.0 / ratio:.1f}x below "
            f"LC50 = {lc50:.4g} (ratio {ratio:.2f}). The behavioural effect is "
            "half-maximal at a dose that kills few flies, so it is defensible as a "
            "behavioural phenotype rather than as general toxicity. Doses at or below "
            + (f"{max_clean_dose:.4g}" if np.isfinite(max_clean_dose) else "n/a")
            + f" have mortality under {mortality_threshold:.0%}."
        )

    out.update({
        "behaviour_fit": beh,
        "mortality_fit": mort,
        "ec50": ec50,
        "ec50_ci": beh_ci,
        "lc50": lc50,
        "lc50_ci": mort_ci,
        "lc50_param_e": lc50_e,
        "lc50_definition": lc50_def,
        "lc50_reason": lc50_reason,
        "ratio_note": (
            f"The {ratio_warn:g} and {ratio_fail:g} ratio thresholds are SCREENING "
            "CONVENTIONS chosen for this platform, not statistical tests, and they "
            "carry no error rate. Nothing is estimated from them; the per-dose "
            "mortality table is the primary evidence and is always returned. "
            "EC50 is a relative ECx of the behavioural curve (half of its own "
            "fitted range) while LC50 is the absolute 50%-survival dose, so the "
            "ratio compares two differently-defined quantities and is a screening "
            "heuristic for whether the two curves sit in the same dose region."
        ),
        "ratio": ratio,
        "ratio_thresholds": {"warn": float(ratio_warn), "fail": float(ratio_fail)},
        "max_mortality": max_mort,
        "per_dose": per_dose,
        "n_confounded_doses": n_confounded_doses,
        "n_dose_levels": int(len(per_dose)),
        "max_clean_dose": max_clean_dose,
        "confounded": confounded,
        "severity": severity,
        "warning": warning,
        "method": (
            "LL.4 behavioural curve among survivors; survival proportion fitted with "
            f"{mortality_model} (control survival free = Abbott 1925 correction as a "
            "parameter), and the LC50 obtained by solving that fitted curve for "
            "survival = 0.5 -- not by reading the fitted `e`, which is the dose at "
            "which survival is (c + d)/2 and equals the LC50 only when the curve "
            "runs from 1 to 0; overlap reported as EC50/LC50."
        ),
    })
    return out
