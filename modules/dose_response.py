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


def fit_loglogistic(
    dose: Sequence[float],
    response: Sequence[float],
    n: Optional[Sequence[float]] = None,
    model: str = "LL.4",
    force: bool = False,
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

    design = check_design(x, y, model=model)
    if not design["usable"] and not force:
        raise ValueError(
            "Design cannot support a " + model + " fit:\n  - "
            + "\n  - ".join(design["refusals"])
            + "\nPass force=True to fit anyway (the standard errors will not be valid)."
        )

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
          is fitted as a decreasing curve, so `e` is by definition the LC50.
          The default LL.3 leaves the upper asymptote free, which lets the
          control survival be below 1 -- that is precisely Abbott's (1925)
          correction expressed as a model parameter rather than applied by
          hand. Use "LL.2" to force control survival to exactly 1.
      (c) The overlap. Per-dose mortality, the highest dose at which mortality
          stays below `mortality_threshold` (the "behaviour-only window"), and
          the ratio EC50_behaviour / LC50.

    HOW TO READ THE RATIO.
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

    # ── (b) mortality curve (fitted as SURVIVAL, so e is the LC50) ─────────
    max_mort = float(np.max(dd / ne)) if x.size else float("nan")
    mort, lc50, mort_ci = None, float("nan"), None
    mort_error = None
    if max_mort < 0.5:
        mort_error = (
            f"Maximum observed mortality is {max_mort:.1%}; survival never falls "
            "below 50% at any tested dose, so the LC50 is above the tested range "
            "and cannot be estimated without extrapolating."
        )
    else:
        try:
            mort = fit_loglogistic(x, surv_prop, n=ne, model=mortality_model)
            lc50 = float(mort.params["e"])
            mort_ci = ec50_profile_ci(mort, level=level)
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
            f"{mortality_model} so e is the LC50 (control survival free = Abbott 1925 "
            "correction as a parameter); overlap reported as EC50/LC50."
        ),
    })
    return out
