"""
Time-of-day (circadian) modulation of the CLIMBING assay.

WHAT THIS MODULE IS FOR, AND WHAT IT IS NOT
-------------------------------------------
The lab tests the SAME flies at morning / afternoon / evening, sometimes
repeatedly across ages. Drosophila is crepuscular -- spontaneous locomotor
activity is bimodal, with dawn and dusk peaks and a midday siesta -- so the
question naturally arises whether *startle-induced climbing* also varies with
circadian phase, and whether the AMPLITUDE of that variation degrades early in
disease genotypes.

This module supplies the machinery to ASK that question properly. It does not
assert the answer, and the published evidence it is built against is mixed:

  * Gargano JW, Martin I, Bhandari P, Grotewiel MS (2005) Exp Gerontol
    40(5):386-395 explicitly report that performance in the RING (rapid
    iterative negative geotaxis) assay "is not influenced by the density of
    flies being tested, THE TIME OF DAY, or REPEATED TESTING". That is a
    published NULL for both effects this module models. It was obtained under
    one laboratory's conditions, genotypes and inter-trial intervals, so it
    does not prove the effects are absent everywhere -- but it does mean that
    a null result here is the *expected* result, not a failure, and that any
    positive finding needs a correspondingly strong design behind it.

    NOTE FOR MAINTAINERS: `modules/design_schema.py` currently cites Gargano
    et al. (2005) as evidence that repeated negative geotaxis *causes*
    within-session fatigue and across-session habituation. That is the
    opposite of what the paper reports. The fatigue/habituation confound is a
    real DESIGN RISK that must be modelled (it is not measured away by one
    study), but it should not be attributed to Gargano et al.

  * Vaccaro A, Issa AR, Seugnet L, Birman S, Klarsfeld A (2017) PLoS Genet
    13(1):e1006507 show that disrupting Clock in PDF pacemaker neurons
    accelerates age-related decline of startle-induced negative geotaxis,
    independently of behavioural rhythmicity. So the clock is linked to
    climbing senescence -- but that paper measured SING at a fixed time and
    did not report time-of-day variation in SING itself.

The honest summary encoded throughout this module: TIME-OF-DAY MODULATION OF
CLIMBING IS A HYPOTHESIS. Every function reports effect sizes with intervals
and states what would have to be true for the estimate to mean what it appears
to mean.

RELATIONSHIP TO modules/dams.py
-------------------------------
`dams.py` estimates the FREE-RUNNING PERIOD from continuous DAM beam-crossing
data with the Sokolove-Bushell chi-square periodogram. The period is unknown
there and must be searched for. Here the period is KNOWN (24 h, the entrained
LD cycle) and the sampling is sparse (3 assay sessions per day), so the
question is not "what is the period" but "given a 24 h period, is there a
detectable amplitude, and where is the peak". Those are different estimation
problems and nothing is duplicated between the two modules.

WHY COSINOR AND NOT ANOVA ACROSS THREE LABELS
---------------------------------------------
A one-way ANOVA over {morning, afternoon, evening} spends 2 degrees of freedom
and answers "do the three means differ". Cosinor spends the same 2 degrees of
freedom but reparameterises them into quantities that are biologically
interpretable and comparable across studies with different sampling times:
MESOR (rhythm-adjusted mean), AMPLITUDE (half the peak-to-trough excursion) and
ACROPHASE (clock time of the peak). It also extends immediately to unbalanced
and non-equidistant sampling, which categorical ANOVA does not.

THE LINEAR REPARAMETERISATION
-----------------------------
The model
        y(t) = M + A * cos(2*pi*(t - phi)/period) + e
is nonlinear in (A, phi), but expanding the cosine gives
        y(t) = M + beta*cos(w t) + gamma*sin(w t) + e,    w = 2*pi/period
with
        beta = A cos(w phi),   gamma = A sin(w phi)
which is LINEAR in (M, beta, gamma). Ordinary/weighted least squares therefore
solves it in closed form -- no starting values, no convergence failures, no
local minima -- and
        A   = sqrt(beta^2 + gamma^2)
        phi = atan2(gamma, beta) / w        (hours, the time of the peak)
recovers the interpretable parameters exactly. `validation/` asserts that this
agrees with a direct nonlinear least-squares fit to machine precision.

UNCERTAINTY IN AMPLITUDE AND ACROPHASE
--------------------------------------
A and phi are the polar coordinates of the vector (beta, gamma). Their
uncertainty must come from the JOINT covariance of (beta, gamma), never from
the two marginal variances separately. Three constructions are provided and
all three are reported:

  1. DELTA METHOD -- first-order propagation of Cov(beta, gamma). Fast, but the
     polar map is singular at the origin, so the approximation degrades badly
     when A is small relative to its standard error. Flagged automatically.
  2. PARAMETRIC BOOTSTRAP -- resimulate y* = fitted + N(0, sigma_hat) and refit.
     Because the model is linear the refit is a single matrix multiply, so this
     is cheap and, unlike the delta method, it also propagates the uncertainty
     in sigma_hat. Percentile intervals; the acrophase interval is taken on the
     circle relative to the point estimate.
  3. ELLIPTICAL JOINT CONFIDENCE REGION (Bingham et al. 1982) -- the exact
     (1-alpha) region for (beta, gamma) is the ellipse
        (v - v_hat)' Cov^{-1} (v - v_hat) <= 2 * F_{2, df}(1 - alpha)
     and the amplitude/acrophase intervals are the ranges of the radius and the
     polar angle over that ellipse.

     The JOINT region has exactly (1-alpha) coverage for the pair -- this is an
     exact finite-sample result, and it is verified at 0.9505 over 4000
     simulations in `validation/` test C. The amplitude and acrophase intervals
     read off it are PROJECTIONS of that joint region, so they are
     CONSERVATIVE: each covers its parameter with probability at least
     (1-alpha), in practice around 0.99 rather than 0.95. That is the price of
     simultaneity and it is deliberate. Use the delta-method or bootstrap
     interval when an exactly-nominal marginal interval is what is wanted; use
     the elliptical one when the amplitude and the phase are going to be
     interpreted together, which is the usual case.

     The elliptical construction also has a decisive property the other two
     lack: IF THE ELLIPSE CONTAINS THE ORIGIN, THE ACROPHASE IS NOT ESTIMABLE
     AT ALL and the amplitude interval includes zero. The module then refuses to
     report a phase rather than printing a meaningless number. Membership of the
     origin in the ellipse is algebraically the same statement as the
     zero-amplitude F-test's failure to reject, and `validation/` test D asserts
     the two never disagree on 600 datasets. This is the recommended interval and
     is what `preferred_interval` points at.

POPULATION AMPLITUDE IS NOT THE AVERAGE INDIVIDUAL AMPLITUDE
------------------------------------------------------------
If flies differ in the CLOCK TIME of their peak, the population rhythm is
damped even when every individual fly has a large amplitude. Writing each
subject's rhythm vector as v_i = A_i * u_i with u_i the unit vector at that
subject's acrophase, and taking A_i independent of phase,

        |E[v_i]| = E[A_i] * |E[u_i]| = (mean individual amplitude) * R

where R = |E[u_i]| in [0, 1] is the mean resultant length of the phases -- the
standard circular synchronisation index. For normally scattered acrophases with
standard deviation sigma_phi hours, R = exp(-(w sigma_phi)^2 / 2); 1.5 h of
scatter under a 24 h period already costs 7% of the amplitude.

This matters directly for the biomarker question: A SMALLER POPULATION
AMPLITUDE IN A DISEASE GENOTYPE HAS TWO DISTINCT EXPLANATIONS -- every fly's
rhythm is damped, or the flies' rhythms are intact but desynchronised from each
other. These are different biology and the population amplitude alone cannot
tell them apart. `population_cosinor` therefore reports the synchronisation
index and the implied phase dispersion alongside the amplitude, and
`amplitude_biomarker` attaches the ambiguity to its verdict.

WHY A SINGLE FLY MEASURED AT THREE TIMES CANNOT SUPPORT A COSINOR
-----------------------------------------------------------------
The cosinor has THREE parameters. Three observations therefore give ZERO
residual degrees of freedom: the cosine passes exactly through all three points
whatever they are, sigma^2 is not estimable, and every standard error, every
confidence interval and the F-test are undefined. The fit is interpolation, not
estimation. `cosinor_fit` returns the exact point estimates together with
`resid_df = 0` and NaN intervals, and says so.

The resolution is the POPULATION-MEAN COSINOR (Bingham et al. 1982): fit one
cosinor per subject, then treat each subject's (beta_i, gamma_i) as one
bivariate observation and do inference ACROSS subjects. Replication then comes
from the number of flies, not from the number of timepoints per fly, and three
timepoints per fly is sufficient. `population_cosinor` implements this.

THE CONFOUND THAT MATTERS MOST
------------------------------
If every fly is tested morning, then afternoon, then evening, TIME OF DAY IS
PERFECTLY COLLINEAR WITH CUMULATIVE TESTING ORDER. Any monotone drift caused by
handling, fatigue, habituation or CO2 recovery is then numerically
indistinguishable from a circadian trend, because the two regressors span the
same column space. No estimator, robust method or amount of data can separate
them; it is a rank deficiency in the design matrix, not a noise problem.
`separate_fatigue_from_rhythm` detects this by rank and by variance inflation
factor and REFUSES to attribute the effect to circadian phase when it holds.
`design_advisor` shows how to avoid it before the experiment is run.

References
----------
Cornelissen G (2014) Cosinor-based rhythmometry. Theoretical Biology and
  Medical Modelling 11:16. doi:10.1186/1742-4682-11-16.
  [single-component cosinor, zero-amplitude test, population-mean cosinor]
Bingham C, Arbogast B, Cornelissen G, Lee JK, Halberg F (1982) Inferential
  statistical methods for estimating and comparing cosinor parameters.
  Chronobiologia 9(4):397-439.
  [population-mean cosinor; elliptical joint confidence region for
   (amplitude, acrophase); tests comparing MESOR, amplitude and acrophase
   between populations]
Gargano JW, Martin I, Bhandari P, Grotewiel MS (2005) Rapid iterative negative
  geotaxis (RING): a new method for assessing age-related locomotor decline in
  Drosophila. Exp Gerontol 40(5):386-395.
  [reports NO effect of time of day and NO effect of repeated testing on RING
   performance under the conditions tested]
Vaccaro A, Issa AR, Seugnet L, Birman S, Klarsfeld A (2017) Drosophila Clock is
  required in brain pacemaker neurons to prevent premature locomotor aging
  independently of its circadian function. PLoS Genet 13(1):e1006507.
  [clock disruption accelerates senescence of startle-induced negative geotaxis]
Hotelling H (1931) The generalization of Student's ratio. Ann Math Stat
  2(3):360-378.  [the T^2 statistic used for the bivariate (beta, gamma) tests]
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from modules import design_schema as _ds

__all__ = [
    "CosinorFit",
    "cosinor_fit",
    "population_cosinor",
    "separate_fatigue_from_rhythm",
    "amplitude_biomarker",
    "design_advisor",
    "prepare_climbing_response",
    "zt_from_frame",
]

# Default number of parametric-bootstrap replicates. Deterministic given `seed`.
N_BOOT_DEFAULT = 2000
SEED_DEFAULT = 20260803


# ═══════════════════════════════════════════════════════════════════════════
# 0. GETTING ZT AND A RESPONSE OUT OF A DroPhenix FRAME
# ═══════════════════════════════════════════════════════════════════════════
def zt_from_frame(
    df: pd.DataFrame,
    zt_col: Optional[str] = None,
    lights_on_hour: float = 8.0,
) -> Tuple[pd.Series, bool, List[str]]:
    """
    Obtain Zeitgeber time for every row, delegating to `design_schema.derive_zt`.

    This module deliberately does NOT reimplement time-of-day parsing. All the
    alias handling, clock-time parsing and the categorical -> ZT midpoint
    convention (`design_schema.CATEGORICAL_TOD_ZT`) live in one place, so that a
    change to the laboratory's lights-on hour or to the assumed midpoint of the
    "afternoon" window propagates everywhere at once.

    Parameters
    ----------
    zt_col : if given, this column is used verbatim as ZT in hours (mod 24) and
        `derive_zt` is bypassed. Use it only when the column really is ZT.
    lights_on_hour : passed straight through to `design_schema.derive_zt`.

    Returns
    -------
    (zt, is_categorical, warnings)
        `is_categorical` is True when ZT was reconstructed from labels such as
        "Morning", in which case the acrophase inherits the width of the testing
        window and cannot be more precise than that window.
    """
    if zt_col is not None:
        if zt_col not in df.columns:
            raise ValueError(f"zt_col '{zt_col}' not found in the dataframe.")
        zt = pd.to_numeric(df[zt_col], errors="coerce").mod(24.0)
        return zt, False, []
    return _ds.derive_zt(df, lights_on_hour=lights_on_hour)


def prepare_climbing_response(
    df: pd.DataFrame,
    count_col: str = "Count",
    total_col: str = "n",
    scale: str = "empirical_logit",
) -> Tuple[pd.Series, pd.Series, str]:
    """
    Build the response and its analysis weights from climbing counts.

    The cosinor is a linear-Gaussian model, so it should not be fitted to a raw
    proportion: `Count/n` is binomial, its variance n*p*(1-p) depends on its
    mean, and climbing data routinely sit near the ceiling where that variance
    collapses. For exactly the reasons set out in `modules/hierarchical.py`,
    the analysis scale is the EMPIRICAL LOGIT

        z_i   = log((y_i + 0.5) / (n_i - y_i + 0.5))
        var_i = 1/(y_i + 0.5) + 1/(n_i - y_i + 0.5)

    which unbounds the ceiling and removes the mean-variance coupling, and the
    weights are 1/var_i so that vials with more flies count for more.

    A MESOR and an amplitude estimated on this scale are in logit units. An
    amplitude of A logits means the odds of climbing swing by a factor of
    exp(2A) from trough to peak; `expit` converts a MESOR back to a proportion.

    Returns
    -------
    (response, weights, scale_note)

    Set `scale="proportion"` to analyse Count/n directly (unweighted). That is
    offered only for comparison; it is not the recommended scale.
    """
    if count_col not in df.columns or total_col not in df.columns:
        raise ValueError(
            f"Need '{count_col}' and '{total_col}' to build a climbing response."
        )
    y = pd.to_numeric(df[count_col], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(df[total_col], errors="coerce").to_numpy(dtype=float)

    if scale == "proportion":
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, y / n, np.nan)
        return (
            pd.Series(p, index=df.index),
            pd.Series(np.ones_like(p), index=df.index),
            "Raw binomial proportion. Variance depends on the mean; intervals "
            "from a Gaussian cosinor on this scale are approximate.",
        )
    if scale != "empirical_logit":
        raise ValueError("scale must be 'empirical_logit' or 'proportion'.")

    from modules.hierarchical import empirical_logit

    z, v = empirical_logit(y, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.isfinite(v) & (v > 0), 1.0 / v, np.nan)
    return (
        pd.Series(z, index=df.index),
        pd.Series(w, index=df.index),
        "Empirical logit with a 0.5 continuity correction; weights = 1/var. "
        "MESOR and amplitude are in logit units (amplitude A => trough-to-peak "
        "odds ratio exp(2A)).",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. SINGLE-COMPONENT COSINOR
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CosinorFit:
    """
    Result of a single-component cosinor fit.

    All three parameterisations are carried: the linear (M, beta, gamma) that
    was actually estimated, the polar (MESOR, amplitude, acrophase) that is
    interpreted, and the covariance that links them.
    """

    # --- interpretable parameters ------------------------------------------
    mesor: float
    amplitude: float
    acrophase_h: float                     # hours; time of the PEAK, in [0, period)

    # --- the linear parameters that were estimated -------------------------
    beta: float                            # coefficient on cos(wt)
    gamma: float                           # coefficient on sin(wt)
    cov_full: np.ndarray                   # 3x3 Cov of (M, beta, gamma)
    cov_bg: np.ndarray                     # 2x2 Cov of (beta, gamma)

    # --- intervals ---------------------------------------------------------
    mesor_se: float
    mesor_ci: Tuple[float, float]
    amplitude_ci_delta: Tuple[float, float]
    amplitude_ci_boot: Tuple[float, float]
    amplitude_ci_ellipse: Tuple[float, float]
    acrophase_ci_delta: Tuple[float, float]
    acrophase_ci_boot: Tuple[float, float]
    acrophase_ci_ellipse: Tuple[float, float]
    amplitude_se_delta: float
    acrophase_se_delta_h: float

    # --- rhythm detection --------------------------------------------------
    f_statistic: float
    f_pvalue: float
    rhythm_detected: bool
    origin_in_confidence_ellipse: bool

    # --- fit diagnostics ---------------------------------------------------
    period: float
    n_obs: int
    resid_df: int
    sigma2: float
    r_squared: float
    n_distinct_zt: int
    alpha: float
    weighted: bool
    converged: bool = True

    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    method: str = ""

    @property
    def preferred_interval(self) -> Dict[str, Tuple[float, float]]:
        """
        The intervals that should be quoted: the elliptical ones (Bingham et
        al. 1982), which project an EXACT joint (1-alpha) region for
        (beta, gamma) and are therefore conservative marginally (coverage >=
        1-alpha, typically ~0.99). The delta-method and bootstrap intervals are
        retained for comparison, for an exactly-nominal marginal interval, and
        for diagnosing when the polar approximation has broken down.
        """
        return {
            "mesor": self.mesor_ci,
            "amplitude": self.amplitude_ci_ellipse,
            "acrophase_h": self.acrophase_ci_ellipse,
        }

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {
                "parameter": "MESOR", "estimate": self.mesor,
                "ci_lower": self.mesor_ci[0], "ci_upper": self.mesor_ci[1],
                "interval_method": f"t, df={self.resid_df}",
            },
            {
                "parameter": "Amplitude", "estimate": self.amplitude,
                "ci_lower": self.amplitude_ci_ellipse[0],
                "ci_upper": self.amplitude_ci_ellipse[1],
                "interval_method": "joint elliptical region (Bingham 1982)",
            },
            {
                "parameter": "Acrophase (h)", "estimate": self.acrophase_h,
                "ci_lower": self.acrophase_ci_ellipse[0],
                "ci_upper": self.acrophase_ci_ellipse[1],
                "interval_method": "joint elliptical region (Bingham 1982)",
            },
        ]
        return pd.DataFrame(rows).set_index("parameter")

    def verdict(self) -> str:
        if self.resid_df <= 0:
            return (
                f"Exact interpolation: {self.n_obs} observations for 3 parameters leaves "
                "0 residual degrees of freedom. Point estimates are exact but NO standard "
                "error, interval or test exists. Pool across subjects "
                "(population_cosinor) to obtain inference."
            )
        if self.origin_in_confidence_ellipse:
            return (
                f"NO RHYTHM DETECTED (F({2},{self.resid_df}) = {self.f_statistic:.3f}, "
                f"p = {self.f_pvalue:.4g}). The joint confidence ellipse for "
                "(beta, gamma) contains the origin, so a zero amplitude is compatible "
                "with these data and the acrophase is not estimable."
            )
        lo, hi = self.acrophase_ci_ellipse
        return (
            f"Rhythm detected (F({2},{self.resid_df}) = {self.f_statistic:.3f}, "
            f"p = {self.f_pvalue:.4g}). Amplitude = {self.amplitude:.4g} "
            f"[{self.amplitude_ci_ellipse[0]:.4g}, {self.amplitude_ci_ellipse[1]:.4g}], "
            f"peak at ZT {self.acrophase_h:.2f} h [{lo:.2f}, {hi:.2f}], "
            f"MESOR = {self.mesor:.4g}."
        )


def _design_matrix(zt: np.ndarray, period: float) -> np.ndarray:
    """Cosinor design matrix [1, cos(wt), sin(wt)] with w = 2*pi/period."""
    w = 2.0 * np.pi / float(period)
    return np.column_stack([np.ones_like(zt), np.cos(w * zt), np.sin(w * zt)])


def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def _polar_from_bg(beta: float, gamma: float, period: float) -> Tuple[float, float]:
    """(beta, gamma) -> (amplitude, acrophase in hours of the peak, in [0, period))."""
    amp = float(np.hypot(beta, gamma))
    ang = float(np.arctan2(gamma, beta))
    phi = (ang * period / (2.0 * np.pi)) % period
    return amp, phi


def _ellipse_polar_bounds(
    v_hat: np.ndarray,
    cov: np.ndarray,
    c: float,
    period: float,
    n_grid: int = 4001,
) -> Tuple[Tuple[float, float], Tuple[float, float], bool]:
    """
    Range of the radius and of the polar angle over the confidence ellipse

        {v : (v - v_hat)' cov^{-1} (v - v_hat) <= c}

    This is the Bingham et al. (1982) construction. Two facts make it exact and
    make the degenerate case unambiguous:

      * the region is convex, so if it does NOT contain the origin the radius
        |v| attains both its minimum and its maximum on the boundary and the
        set of polar angles is a single arc of width < pi;
      * if it DOES contain the origin then a zero amplitude is inside the
        confidence region, the amplitude interval must start at 0, and every
        angle is attained -- the acrophase is simply not estimable.

    The boundary is traced as v = v_hat + sqrt(c) * L u(theta) with cov = L L'
    obtained by eigendecomposition (robust to a singular covariance).

    Returns ((amp_lo, amp_hi), (phi_lo, phi_hi) in hours, origin_inside).
    `phi_hi` may exceed `period` when the arc wraps past 0; callers should read
    the interval as [phi_lo, phi_hi] modulo `period`.
    """
    v_hat = np.asarray(v_hat, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)) or not np.isfinite(c) or c <= 0:
        nan2 = (float("nan"), float("nan"))
        return nan2, nan2, False

    # Robust square root of a possibly near-singular PSD covariance
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    L = evecs @ np.diag(np.sqrt(evals))

    # Is the origin inside? Use the pseudo-inverse so a singular cov still works.
    cov_pinv = np.linalg.pinv(cov)
    d0 = float(v_hat @ cov_pinv @ v_hat)
    origin_inside = bool(d0 <= c)

    th = np.linspace(0.0, 2.0 * np.pi, n_grid)
    u = np.column_stack([np.cos(th), np.sin(th)])
    pts = v_hat[None, :] + np.sqrt(c) * (u @ L.T)

    r = np.hypot(pts[:, 0], pts[:, 1])
    amp_hi = float(r.max())

    if origin_inside:
        return (0.0, amp_hi), (0.0, float(period)), True

    amp_lo = float(r.min())

    ang = np.arctan2(pts[:, 1], pts[:, 0])
    ang_hat = float(np.arctan2(v_hat[1], v_hat[0]))
    d = _wrap_angle(ang - ang_hat)
    to_h = period / (2.0 * np.pi)
    phi_hat = (ang_hat * to_h) % period
    phi_lo = phi_hat + float(d.min()) * to_h
    phi_hi = phi_hat + float(d.max()) * to_h
    return (amp_lo, amp_hi), (phi_lo % period, (phi_lo % period) + (phi_hi - phi_lo)), False


def cosinor_fit(
    zt: Sequence[float],
    value: Sequence[float],
    period: float = 24.0,
    weights: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
    n_ellipse_grid: int = 4001,
) -> CosinorFit:
    """
    Single-component cosinor regression (Cornelissen 2014).

        y(t) = M + A * cos(2*pi*(t - phi)/period) + e

    fitted through the EXACT linear reparameterisation

        y(t) = M + beta*cos(w t) + gamma*sin(w t) + e,      w = 2*pi/period

    by (weighted) least squares. This needs no starting values and has no
    convergence behaviour to check: the estimate is the closed-form normal-
    equation solution and is identical to a nonlinear least-squares fit of the
    original parameterisation to machine precision (asserted in
    `validation/validate_circadian_climbing.py`, test B).

    Parameters
    ----------
    zt : Zeitgeber time in hours (hours since lights-on). Use
        `zt_from_frame` / `design_schema.derive_zt` to obtain it.
    value : response. For climbing counts use `prepare_climbing_response`
        rather than a raw proportion.
    period : the period held FIXED, in the same units as `zt`. 24 h under a
        12:12 LD cycle. The period is not estimated here -- with three
        timepoints per subject it cannot be. Use `modules.dams` for free-running
        period estimation from continuous activity data.
    weights : optional analysis weights (e.g. 1/variance of the empirical
        logit). None means ordinary least squares.
    alpha : 1 - alpha is the coverage of every interval returned.
    n_boot, seed : parametric-bootstrap replicates and the seed. The bootstrap
        is fully deterministic given `seed`.

    Returns
    -------
    CosinorFit. Read `.verdict()` first; quote `.preferred_interval`.

    Degenerate cases, all handled explicitly rather than by raising
    --------------------------------------------------------------
    fewer than 3 distinct ZT   -> the design matrix is rank deficient, amplitude
                                  and acrophase are NOT identifiable; a warning
                                  is recorded and the polar parameters are NaN.
    exactly 3 observations     -> exact interpolation, resid_df = 0, sigma^2 and
                                  every interval undefined. Point estimates are
                                  returned; inference requires pooling across
                                  subjects (`population_cosinor`).
    """
    t = np.asarray(zt, dtype=float).ravel()
    y = np.asarray(value, dtype=float).ravel()
    if t.shape != y.shape:
        raise ValueError(f"zt and value must have the same length ({t.size} vs {y.size}).")
    if period <= 0:
        raise ValueError("period must be positive.")

    if weights is None:
        w = np.ones_like(y)
        weighted = False
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != y.shape:
            raise ValueError("weights must have the same length as value.")
        weighted = True

    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    t, y, w = t[ok], y[ok], w[ok]
    n = int(t.size)

    warns: List[str] = []
    notes: List[str] = []

    if n < 3:
        raise ValueError(
            f"A cosinor has 3 parameters; only {n} usable observation(s) supplied."
        )

    n_distinct = int(np.unique(np.round(t % period, 9)).size)

    X = _design_matrix(t % period, period)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    rank = int(np.linalg.matrix_rank(Xw, tol=1e-10))
    identifiable = rank == 3
    if not identifiable:
        warns.append(
            f"Only {n_distinct} distinct time(s) of day (design-matrix rank {rank} < 3). "
            "MESOR, amplitude and acrophase are NOT jointly identifiable from this "
            "sampling schedule: infinitely many (amplitude, acrophase) pairs fit these "
            "data equally well. At least three well-separated times of day are required."
        )

    # ── closed-form (weighted) least squares ────────────────────────────────
    XtX = Xw.T @ Xw
    Xty = Xw.T @ yw
    if identifiable:
        XtX_inv = np.linalg.inv(XtX)
        theta = XtX_inv @ Xty
    else:
        XtX_inv = np.linalg.pinv(XtX)
        theta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    M, beta, gamma = (float(theta[0]), float(theta[1]), float(theta[2]))

    fitted = X @ theta
    resid = y - fitted
    resid_df = int(n - rank)

    ssr = float(np.sum(w * resid ** 2))
    ybar_w = float(np.sum(w * y) / np.sum(w))
    sst = float(np.sum(w * (y - ybar_w) ** 2))
    ssm = sst - ssr
    r2 = float(ssm / sst) if sst > 0 else float("nan")

    if resid_df > 0:
        sigma2 = ssr / resid_df
        cov_full = sigma2 * XtX_inv
    else:
        sigma2 = float("nan")
        cov_full = np.full((3, 3), np.nan)
        notes.append(
            f"{n} observations for 3 parameters leaves 0 residual degrees of freedom. "
            "The cosine passes exactly through the data whatever they are, so the "
            "residual variance -- and therefore every standard error, confidence "
            "interval and p-value -- is undefined. The point estimates below are exact "
            "interpolation, not estimation. Use population_cosinor() to obtain "
            "inference by pooling across subjects."
        )

    cov_bg = cov_full[1:, 1:]

    if identifiable:
        amp, phi = _polar_from_bg(beta, gamma, period)
    else:
        amp, phi = float("nan"), float("nan")

    nan2 = (float("nan"), float("nan"))

    # ── zero-amplitude F-test: H0: beta = gamma = 0 ─────────────────────────
    # The reduced model is the intercept alone, so SS_model carries exactly the
    # 2 df spent on the rhythm (Cornelissen 2014).
    if resid_df > 0 and identifiable and np.isfinite(sigma2) and sigma2 > 0:
        f_stat = float((ssm / 2.0) / sigma2)
        f_p = float(stats.f.sf(f_stat, 2, resid_df))
    else:
        f_stat, f_p = float("nan"), float("nan")

    # ── MESOR interval ──────────────────────────────────────────────────────
    if resid_df > 0:
        mesor_se = float(np.sqrt(max(cov_full[0, 0], 0.0)))
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, resid_df))
        mesor_ci = (M - tcrit * mesor_se, M + tcrit * mesor_se)
    else:
        mesor_se, mesor_ci, tcrit = float("nan"), nan2, float("nan")

    # ── 1. DELTA METHOD on the joint covariance of (beta, gamma) ────────────
    amp_ci_delta, phi_ci_delta = nan2, nan2
    amp_se_delta, phi_se_delta = float("nan"), float("nan")
    if resid_df > 0 and identifiable and amp > 0:
        g_amp = np.array([beta / amp, gamma / amp])
        var_amp = float(g_amp @ cov_bg @ g_amp)
        amp_se_delta = float(np.sqrt(max(var_amp, 0.0)))
        amp_ci_delta = (
            max(amp - tcrit * amp_se_delta, 0.0),
            amp + tcrit * amp_se_delta,
        )
        g_ang = np.array([-gamma / amp ** 2, beta / amp ** 2])
        var_ang = float(g_ang @ cov_bg @ g_ang)
        to_h = period / (2.0 * np.pi)
        phi_se_delta = float(np.sqrt(max(var_ang, 0.0)) * to_h)
        phi_ci_delta = (phi - tcrit * phi_se_delta, phi + tcrit * phi_se_delta)

        if np.isfinite(amp_se_delta) and amp_se_delta > 0 and amp / amp_se_delta < 3.0:
            warns.append(
                f"Amplitude/SE = {amp / amp_se_delta:.2f} < 3. The polar transformation "
                "is singular at the origin, so the DELTA-METHOD amplitude and acrophase "
                "intervals are unreliable at this signal-to-noise ratio. Use the "
                "elliptical intervals (preferred_interval), which remain exact."
            )
        if np.isfinite(phi_se_delta) and 2.0 * tcrit * phi_se_delta >= period:
            warns.append(
                "The delta-method acrophase interval spans a full cycle; the phase is "
                "effectively unconstrained by these data."
            )

    # ── 3. ELLIPTICAL JOINT REGION (Bingham et al. 1982) -- the exact one ──
    amp_ci_ell, phi_ci_ell, origin_inside = nan2, nan2, False
    if resid_df > 0 and identifiable:
        c = 2.0 * float(stats.f.ppf(1.0 - alpha, 2, resid_df))
        amp_ci_ell, phi_ci_ell, origin_inside = _ellipse_polar_bounds(
            np.array([beta, gamma]), cov_bg, c, period, n_grid=n_ellipse_grid
        )
        if origin_inside:
            notes.append(
                f"The {100 * (1 - alpha):.0f}% joint confidence ellipse for (beta, gamma) "
                "contains the origin. A zero amplitude is therefore compatible with these "
                "data and the acrophase is NOT estimable -- the reported acrophase "
                "interval covers the whole cycle deliberately, rather than reporting a "
                "phase that the data do not support."
            )

    # ── 2. PARAMETRIC BOOTSTRAP ─────────────────────────────────────────────
    # For a linear model the refit is a single matrix multiply, so this is a
    # genuine parametric bootstrap (resimulate, refit, recompute sigma_hat),
    # not merely a draw from the asymptotic normal of (beta, gamma). It
    # therefore also propagates the uncertainty in sigma_hat, which the delta
    # method conditions on.
    amp_ci_boot, phi_ci_boot = nan2, nan2
    if resid_df > 0 and identifiable and n_boot and n_boot > 0 and np.isfinite(sigma2):
        rng = np.random.default_rng(seed)
        P = XtX_inv @ Xw.T                      # theta_hat = P @ y_weighted
        sd_i = np.sqrt(sigma2) / sw             # residual sd on the raw scale
        eps = rng.normal(0.0, 1.0, size=(int(n_boot), n)) * sd_i[None, :]
        Y = (fitted[None, :] + eps) * sw[None, :]
        TH = Y @ P.T                            # (n_boot, 3)
        b_b, g_b = TH[:, 1], TH[:, 2]
        amp_b = np.hypot(b_b, g_b)
        amp_ci_boot = (
            float(np.quantile(amp_b, alpha / 2.0)),
            float(np.quantile(amp_b, 1.0 - alpha / 2.0)),
        )
        # Acrophase percentiles must be taken ON THE CIRCLE, relative to the
        # point estimate, or an interval straddling ZT0 would be reported
        # inside-out.
        ang_b = np.arctan2(g_b, b_b)
        d = _wrap_angle(ang_b - np.arctan2(gamma, beta)) * (period / (2.0 * np.pi))
        phi_ci_boot = (
            (phi + float(np.quantile(d, alpha / 2.0))) % period,
            (phi + float(np.quantile(d, alpha / 2.0))) % period
            + float(np.quantile(d, 1.0 - alpha / 2.0) - np.quantile(d, alpha / 2.0)),
        )

    if n_distinct == 3 and n <= 3:
        notes.append(
            "Three times of day exactly identify the three cosinor parameters. This is "
            "the minimum sampling schedule; it leaves no residual degrees of freedom and "
            "no ability to check the cosine shape against the data."
        )

    return CosinorFit(
        mesor=M,
        amplitude=amp,
        acrophase_h=phi,
        beta=beta,
        gamma=gamma,
        cov_full=cov_full,
        cov_bg=cov_bg,
        mesor_se=mesor_se,
        mesor_ci=mesor_ci,
        amplitude_ci_delta=amp_ci_delta,
        amplitude_ci_boot=amp_ci_boot,
        amplitude_ci_ellipse=amp_ci_ell,
        acrophase_ci_delta=phi_ci_delta,
        acrophase_ci_boot=phi_ci_boot,
        acrophase_ci_ellipse=phi_ci_ell,
        amplitude_se_delta=amp_se_delta,
        acrophase_se_delta_h=phi_se_delta,
        f_statistic=f_stat,
        f_pvalue=f_p,
        rhythm_detected=bool(np.isfinite(f_p) and f_p < alpha),
        origin_in_confidence_ellipse=origin_inside,
        period=float(period),
        n_obs=n,
        resid_df=resid_df,
        sigma2=sigma2,
        r_squared=r2,
        n_distinct_zt=n_distinct,
        alpha=float(alpha),
        weighted=weighted,
        warnings=warns,
        notes=notes,
        method=(
            "Single-component cosinor via the exact linear reparameterisation "
            "(Cornelissen 2014); amplitude/acrophase intervals from the joint "
            "(beta, gamma) covariance by delta method, parametric bootstrap and the "
            "elliptical joint confidence region (Bingham et al. 1982)."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. POPULATION-MEAN COSINOR
# ═══════════════════════════════════════════════════════════════════════════
def _hotelling_one_sample(V: np.ndarray, alpha: float = 0.05) -> dict:
    """
    One-sample Hotelling T^2 test of H0: E[(beta, gamma)] = (0, 0).

    V is (k, 2), one row per subject. This is the population zero-amplitude
    test of Bingham et al. (1982): it asks whether the MEAN rhythm vector
    across subjects differs from the origin, using the between-subject
    covariance as the error term. The unit of replication is the SUBJECT, which
    is what makes 3 timepoints per subject sufficient.

        T^2 = k * vbar' S^{-1} vbar
        F   = (k - 2) / (2 (k - 1)) * T^2   ~   F(2, k - 2)
    """
    V = np.asarray(V, dtype=float)
    k = int(V.shape[0])
    out = {"k": k, "T2": float("nan"), "F": float("nan"), "df1": 2,
           "df2": max(k - 2, 0), "p_value": float("nan"),
           "mean": np.full(2, np.nan), "cov_mean": np.full((2, 2), np.nan)}
    if k < 3:
        out["reason"] = (
            f"Only {k} subject(s) with an estimable rhythm vector. The population "
            "cosinor needs at least 3 to form F(2, k-2)."
        )
        return out

    vbar = V.mean(axis=0)
    S = np.cov(V, rowvar=False, ddof=1)
    out["mean"] = vbar
    out["cov_mean"] = S / k
    try:
        T2 = float(k * (vbar @ np.linalg.inv(S) @ vbar))
    except np.linalg.LinAlgError:
        out["reason"] = "Between-subject covariance of (beta, gamma) is singular."
        return out
    F = (k - 2) / (2.0 * (k - 1)) * T2
    out.update({
        "T2": T2, "F": float(F),
        "p_value": float(stats.f.sf(F, 2, k - 2)),
    })
    return out


def _phase_dispersion(V: np.ndarray, period: float) -> dict:
    """
    How synchronised are the subjects' acrophases, and how much amplitude does
    their scatter cost?

    Writing subject i's rhythm vector as v_i = A_i u_i with u_i the unit vector
    at that subject's acrophase, and taking amplitude independent of phase,

        population amplitude |mean v_i| = (mean A_i) * R,
        R = |mean u_i| in [0, 1]  (the circular mean resultant length)

    so R is exactly the factor by which between-fly phase scatter damps the
    population rhythm. R = 1 is perfect synchrony; R near 0 means the phases are
    spread around the clock and the population looks arrhythmic however strongly
    each individual fly oscillates.

    The circular standard deviation follows from R as sqrt(-2 ln R) radians.

    BOTH QUANTITIES ARE CONTAMINATED BY ESTIMATION NOISE, in a known direction:
    a noisy per-subject acrophase adds apparent scatter, so R is biased DOWN and
    the reported phase dispersion is biased UP. Read `phase_dispersion_sd_h` as
    an upper bound on the true between-fly scatter. With three timepoints per
    subject that noise is substantial, so treat these as descriptive.
    """
    V = np.asarray(V, dtype=float)
    V = V[np.all(np.isfinite(V), axis=1)]
    out: dict = {
        "synchronisation_index_R": float("nan"),
        "phase_dispersion_sd_h": float("nan"),
        "amplitude_attenuation_note": "",
    }
    if V.shape[0] < 2:
        return out
    amps = np.hypot(V[:, 0], V[:, 1])
    good = amps > 0
    if good.sum() < 2:
        return out
    U = V[good] / amps[good][:, None]
    R = float(np.hypot(*U.mean(axis=0)))
    R = min(max(R, 1e-12), 1.0)
    sd_rad = float(np.sqrt(max(-2.0 * np.log(R), 0.0)))
    out["synchronisation_index_R"] = R
    out["phase_dispersion_sd_h"] = float(sd_rad * period / (2.0 * np.pi))
    out["amplitude_attenuation_note"] = (
        f"Between-subject phase scatter (circular SD ~ {out['phase_dispersion_sd_h']:.2f} h, "
        f"synchronisation index R = {R:.3f}) damps the population amplitude to about "
        f"{100 * R:.0f}% of the mean individual amplitude. A REDUCED POPULATION AMPLITUDE "
        "IS THEREFORE AMBIGUOUS: it can mean that each fly's rhythm is damped, or that "
        "the flies' rhythms are intact but desynchronised from one another. Those are "
        "different biology and the population amplitude alone cannot separate them. "
        "R is biased downwards by per-subject estimation noise, so the true synchrony "
        "is at least this good."
    )
    return out


def population_cosinor(
    df: pd.DataFrame,
    zt_col: str,
    value_col: str,
    subject_col: str,
    group_col: Optional[str] = None,
    period: float = 24.0,
    weight_col: Optional[str] = None,
    alpha: float = 0.05,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """
    Population-mean cosinor (Bingham C, Arbogast B, Cornelissen G, Lee JK,
    Halberg F (1982) Chronobiologia 9:397-439).

    WHY THIS IS NOT OPTIONAL FOR THIS DESIGN
    ----------------------------------------
    With THREE times of day per subject, a single-subject cosinor has exactly
    ZERO residual degrees of freedom: three observations, three parameters. The
    fitted cosine passes through all three points no matter what they are, the
    residual variance is undefined, and amplitude and acrophase have no standard
    error, no confidence interval and no test. A per-subject cosinor is
    interpolation, not estimation.

    The population-mean cosinor fixes this by moving the replication from the
    timepoints to the SUBJECTS. Each subject contributes one bivariate
    observation (beta_i, gamma_i) -- an exact summary of that subject's rhythm --
    and inference is then a bivariate one-sample problem across k subjects:

        population (beta, gamma) = mean of the subject vectors
        Cov of that mean         = S / k, S the between-subject covariance
        zero-amplitude test      = one-sample Hotelling T^2 -> F(2, k - 2)
        amplitude / acrophase CI = the SAME elliptical construction as in
                                   cosinor_fit, applied to S / k with
                                   c = 2 (k - 1)/(k - 2) * F_{2, k-2}(1 - alpha)

    The population amplitude is `|mean of the vectors|`, NOT the `mean of the
    subjects' amplitudes`. Those differ, and the second is biased UPWARDS:
    |v_i| is a positive quantity contaminated by noise in both components, so
    E[|v_i|] > |E[v_i]| whenever there is any measurement error (the same
    Rayleigh-type bias that inflates naive vector-strength estimates). The
    vector mean is reported as the population amplitude for exactly this
    reason; the mean of individual amplitudes is also returned, clearly
    labelled, so the discrepancy is visible rather than hidden.

    Parameters
    ----------
    zt_col : column of Zeitgeber hours. Obtain it via `zt_from_frame`.
    value_col : response column (see `prepare_climbing_response`).
    subject_col : the unit of replication -- FlyID if flies are individually
        tracked and independently housed, otherwise VialID. Getting this wrong
        is the pseudoreplication error documented in `modules/hierarchical.py`;
        `design_schema.describe_design().cluster_unit` names the right column.
    group_col : optional grouping (e.g. Genotype). When supplied, a separate
        population cosinor is fitted per group and every pair of groups is
        compared on MESOR, on the joint rhythm vector, and on amplitude and
        acrophase separately.

    Returns
    -------
    dict with
      'per_subject'  : DataFrame of (beta_i, gamma_i, mesor_i, amplitude_i,
                       acrophase_i, n_obs_i, resid_df_i) -- the raw material
      'overall'      : population cosinor over all subjects
      'by_group'     : {group: population cosinor}
      'comparisons'  : list of pairwise group comparisons
      'warnings', 'notes'
    """
    for col in (zt_col, value_col, subject_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")
    if group_col is not None and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in the dataframe.")

    keep = [zt_col, value_col, subject_col] + (
        [group_col] if group_col else []
    ) + ([weight_col] if weight_col else [])
    work = df[keep].copy()
    work[zt_col] = pd.to_numeric(work[zt_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if weight_col:
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work.dropna(subset=[zt_col, value_col, subject_col])

    warns: List[str] = []
    notes: List[str] = []

    # ── per-subject cosinors ────────────────────────────────────────────────
    rows = []
    n_skipped = 0
    zero_df_subjects = 0
    for sid, sub in work.groupby(subject_col, sort=True):
        t = sub[zt_col].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        wts = sub[weight_col].to_numpy(dtype=float) if weight_col else None
        n_distinct = int(np.unique(np.round(t % period, 9)).size)
        if n_distinct < 3:
            n_skipped += 1
            continue
        fit = cosinor_fit(t, y, period=period, weights=wts, alpha=alpha,
                          n_boot=0, seed=seed)
        if fit.resid_df == 0:
            zero_df_subjects += 1
        rec = {
            subject_col: sid,
            "mesor": fit.mesor, "beta": fit.beta, "gamma": fit.gamma,
            "amplitude": fit.amplitude, "acrophase_h": fit.acrophase_h,
            "n_obs": fit.n_obs, "n_distinct_zt": fit.n_distinct_zt,
            "resid_df": fit.resid_df,
        }
        if group_col:
            g = sub[group_col].dropna().unique()
            if len(g) != 1:
                warns.append(
                    f"Subject '{sid}' appears under {len(g)} levels of '{group_col}' "
                    f"({list(g)[:4]}); it was assigned to '{g[0] if len(g) else 'NA'}'. "
                    "A subject that changes group is a data-integrity problem."
                )
            rec[group_col] = g[0] if len(g) else np.nan
        rows.append(rec)

    per_subject = pd.DataFrame(rows)

    if n_skipped:
        warns.append(
            f"{n_skipped} subject(s) had fewer than 3 distinct times of day and were "
            "excluded: a cosinor is not identifiable from fewer than 3 timepoints. "
            "Excluding subjects on a design criterion is safe here (it does not depend "
            "on the response), but it does reduce k."
        )
    if zero_df_subjects:
        notes.append(
            f"{zero_df_subjects} of {len(per_subject)} subject(s) were measured at exactly "
            "3 times of day, so their individual cosinors have ZERO residual degrees of "
            "freedom (resid_df = 0) and no within-subject standard error exists. This is "
            "expected and is precisely why the population-mean cosinor is used: the "
            "replication that supports inference comes from the k subjects, not from the "
            "timepoints within a subject."
        )

    if per_subject.empty:
        return {
            "per_subject": per_subject, "overall": None, "by_group": {},
            "comparisons": [], "warnings": warns + ["No subject had 3+ distinct times of day."],
            "notes": notes, "period": period, "alpha": alpha,
        }

    def _population(sub_tab: pd.DataFrame, label: str) -> dict:
        V = sub_tab[["beta", "gamma"]].to_numpy(dtype=float)
        V = V[np.all(np.isfinite(V), axis=1)]
        k = int(V.shape[0])
        ht = _hotelling_one_sample(V, alpha=alpha)
        Ms = sub_tab["mesor"].to_numpy(dtype=float)
        Ms = Ms[np.isfinite(Ms)]

        res: dict = {
            "label": label, "k_subjects": k,
            "mesor": float(np.mean(Ms)) if Ms.size else float("nan"),
            "mesor_sd": float(np.std(Ms, ddof=1)) if Ms.size > 1 else float("nan"),
            "mesor_se": float(np.std(Ms, ddof=1) / np.sqrt(Ms.size)) if Ms.size > 1 else float("nan"),
            "hotelling": ht,
            "period": period, "alpha": alpha,
            "mean_of_individual_amplitudes": (
                float(np.mean(sub_tab["amplitude"].dropna())) if k else float("nan")
            ),
            "amplitude_bias_note": (
                "mean_of_individual_amplitudes is biased UPWARDS relative to the "
                "population amplitude: |v_i| is non-negative and inflated by noise in "
                "both components (E|v_i| > |E v_i|). Quote 'amplitude' (the length of "
                "the mean vector), not this."
            ),
        }
        res.update(_phase_dispersion(V, period))
        if Ms.size > 1:
            tc = float(stats.t.ppf(1.0 - alpha / 2.0, Ms.size - 1))
            res["mesor_ci"] = (res["mesor"] - tc * res["mesor_se"],
                               res["mesor"] + tc * res["mesor_se"])
        else:
            res["mesor_ci"] = (float("nan"), float("nan"))

        if k < 3:
            res.update({
                "amplitude": float("nan"), "acrophase_h": float("nan"),
                "amplitude_ci": (float("nan"), float("nan")),
                "acrophase_ci": (float("nan"), float("nan")),
                "rhythm_detected": False,
                "origin_in_confidence_ellipse": False,
                "reason": ht.get("reason", "Fewer than 3 subjects."),
            })
            return res

        vbar = ht["mean"]
        amp, phi = _polar_from_bg(float(vbar[0]), float(vbar[1]), period)
        # Exact (1-alpha) region for the MEAN vector under multivariate normality:
        #   k (vbar - mu)' S^{-1} (vbar - mu) <= 2(k-1)/(k-2) F_{2,k-2}(1-alpha)
        # Written with Cov(vbar) = S/k this is the same ellipse form used above.
        c = 2.0 * (k - 1) / (k - 2) * float(stats.f.ppf(1.0 - alpha, 2, k - 2))
        amp_ci, phi_ci, origin_inside = _ellipse_polar_bounds(
            vbar, ht["cov_mean"], c, period
        )
        res.update({
            "amplitude": amp, "acrophase_h": phi,
            "amplitude_ci": amp_ci, "acrophase_ci": phi_ci,
            "origin_in_confidence_ellipse": origin_inside,
            "rhythm_detected": bool(np.isfinite(ht["p_value"]) and ht["p_value"] < alpha),
            "zero_amplitude_p": ht["p_value"],
            "method": (
                "Population-mean cosinor (Bingham et al. 1982): subject-level "
                "(beta, gamma) as the unit of replication; one-sample Hotelling T^2 "
                "for zero amplitude; elliptical joint region for amplitude/acrophase."
            ),
        })
        if origin_inside:
            res["verdict"] = (
                f"[{label}] NO POPULATION RHYTHM DETECTED (Hotelling T^2 = {ht['T2']:.3f}, "
                f"F({ht['df1']},{ht['df2']}) = {ht['F']:.3f}, p = {ht['p_value']:.4g}, "
                f"k = {k}). The joint confidence ellipse contains the origin, so the "
                "acrophase is not estimable."
            )
        else:
            res["verdict"] = (
                f"[{label}] Population rhythm detected (F({ht['df1']},{ht['df2']}) = "
                f"{ht['F']:.3f}, p = {ht['p_value']:.4g}, k = {k}). Amplitude "
                f"{amp:.4g} [{amp_ci[0]:.4g}, {amp_ci[1]:.4g}], peak at ZT {phi:.2f} h "
                f"[{phi_ci[0]:.2f}, {phi_ci[1]:.2f}], MESOR {res['mesor']:.4g}."
            )
        return res

    overall = _population(per_subject, "all subjects")

    by_group: Dict[str, dict] = {}
    comparisons: List[dict] = []
    if group_col:
        for g, tab in per_subject.groupby(group_col, sort=True):
            by_group[str(g)] = _population(tab, str(g))

        levels = sorted(by_group.keys())
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                comparisons.append(
                    _compare_populations(
                        per_subject[per_subject[group_col].astype(str) == levels[i]],
                        per_subject[per_subject[group_col].astype(str) == levels[j]],
                        levels[i], levels[j], period, alpha, n_boot, seed,
                    )
                )
        if len(levels) > 2:
            notes.append(
                f"{len(levels)} groups produced {len(comparisons)} pairwise comparisons. "
                "P-values are UNADJUSTED for multiplicity; a Holm correction across the "
                "family you actually care about is the caller's responsibility."
            )

    return {
        "per_subject": per_subject,
        "overall": overall,
        "by_group": by_group,
        "comparisons": comparisons,
        "warnings": warns,
        "notes": notes,
        "period": period,
        "alpha": alpha,
        "subject_col": subject_col,
        "group_col": group_col,
    }


def _compare_populations(
    tab_a: pd.DataFrame,
    tab_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    period: float,
    alpha: float,
    n_boot: int,
    seed: int,
) -> dict:
    """
    Compare two populations' cosinor parameters (Bingham et al. 1982).

    Three questions, kept separate because they answer different things:

      MESOR      -- Welch two-sample t on the subject MESORs. Welch rather than
                    pooled because genotypes routinely differ in variance as
                    well as in mean, and the pooled test is not robust to that.
      RHYTHM     -- two-sample Hotelling T^2 on the bivariate (beta, gamma).
                    This is the omnibus test: it is significant if the
                    amplitude differs, or the acrophase differs, or both. It is
                    the correct FIRST test, because testing amplitude and
                    acrophase separately without it inflates the error rate and
                    because amplitude and acrophase are not independent.
      AMPLITUDE  -- differences with nonparametric bootstrap intervals
      / ACROPHASE   (subjects resampled with replacement within group). The
                    bootstrap is used rather than a delta-method contrast
                    because the polar transform is singular near zero
                    amplitude, which is exactly the regime a disease genotype
                    with a damped rhythm is expected to occupy.

    An acrophase difference is meaningless if either group has no detectable
    rhythm, so that case is flagged and the acrophase contrast is suppressed.
    """
    out: dict = {"group_a": name_a, "group_b": name_b, "alpha": alpha}

    Ma = tab_a["mesor"].to_numpy(dtype=float)
    Mb = tab_b["mesor"].to_numpy(dtype=float)
    Ma, Mb = Ma[np.isfinite(Ma)], Mb[np.isfinite(Mb)]
    Va = tab_a[["beta", "gamma"]].to_numpy(dtype=float)
    Vb = tab_b[["beta", "gamma"]].to_numpy(dtype=float)
    Va = Va[np.all(np.isfinite(Va), axis=1)]
    Vb = Vb[np.all(np.isfinite(Vb), axis=1)]
    ka, kb = int(Va.shape[0]), int(Vb.shape[0])
    out["k_a"], out["k_b"] = ka, kb

    # ── MESOR ───────────────────────────────────────────────────────────────
    if Ma.size >= 2 and Mb.size >= 2:
        tt = stats.ttest_ind(Ma, Mb, equal_var=False)
        diff = float(Ma.mean() - Mb.mean())
        se = float(np.sqrt(Ma.var(ddof=1) / Ma.size + Mb.var(ddof=1) / Mb.size))
        dfw = (Ma.var(ddof=1) / Ma.size + Mb.var(ddof=1) / Mb.size) ** 2 / (
            (Ma.var(ddof=1) / Ma.size) ** 2 / (Ma.size - 1)
            + (Mb.var(ddof=1) / Mb.size) ** 2 / (Mb.size - 1)
        )
        tc = float(stats.t.ppf(1.0 - alpha / 2.0, dfw))
        sp = np.sqrt(((Ma.size - 1) * Ma.var(ddof=1) + (Mb.size - 1) * Mb.var(ddof=1))
                     / (Ma.size + Mb.size - 2))
        J = 1.0 - 3.0 / (4.0 * (Ma.size + Mb.size) - 9.0)   # Hedges' small-sample factor
        out["mesor"] = {
            "diff": diff, "se": se,
            "ci": (diff - tc * se, diff + tc * se),
            "t": float(tt.statistic), "df": float(dfw), "p_value": float(tt.pvalue),
            "hedges_g": float(J * diff / sp) if sp > 0 else float("nan"),
            "test": "Welch two-sample t on subject MESORs",
        }
    else:
        out["mesor"] = {"p_value": float("nan"),
                        "reason": "Need >=2 subjects per group."}

    # ── joint rhythm vector: two-sample Hotelling T^2 ───────────────────────
    if ka >= 2 and kb >= 2 and (ka + kb - 3) > 0:
        d = Va.mean(axis=0) - Vb.mean(axis=0)
        Sa = np.cov(Va, rowvar=False, ddof=1)
        Sb = np.cov(Vb, rowvar=False, ddof=1)
        Sp = ((ka - 1) * Sa + (kb - 1) * Sb) / (ka + kb - 2)
        try:
            T2 = float(d @ np.linalg.inv(Sp * (1.0 / ka + 1.0 / kb)) @ d)
            F = (ka + kb - 3) / (2.0 * (ka + kb - 2)) * T2
            out["rhythm_vector"] = {
                "T2": T2, "F": float(F), "df1": 2, "df2": ka + kb - 3,
                "p_value": float(stats.f.sf(F, 2, ka + kb - 3)),
                "test": ("Two-sample Hotelling T^2 on (beta, gamma) -- the omnibus "
                         "test that the rhythm differs in amplitude, phase, or both "
                         "(Bingham et al. 1982)"),
            }
        except np.linalg.LinAlgError:
            out["rhythm_vector"] = {"p_value": float("nan"),
                                    "reason": "Pooled covariance is singular."}
    else:
        out["rhythm_vector"] = {"p_value": float("nan"),
                                "reason": "Need >=2 subjects per group."}

    # ── amplitude and acrophase contrasts by nonparametric bootstrap ────────
    amp_a, phi_a = _polar_from_bg(*Va.mean(axis=0), period) if ka else (np.nan, np.nan)
    amp_b, phi_b = _polar_from_bg(*Vb.mean(axis=0), period) if kb else (np.nan, np.nan)
    d_amp = amp_a - amp_b
    d_phi = float(_wrap_angle((phi_a - phi_b) * 2 * np.pi / period)) * period / (2 * np.pi)

    if ka >= 3 and kb >= 3 and n_boot and n_boot > 0:
        rng = np.random.default_rng(seed + 991)
        ia = rng.integers(0, ka, size=(int(n_boot), ka))
        ib = rng.integers(0, kb, size=(int(n_boot), kb))
        va = Va[ia].mean(axis=1)
        vb = Vb[ib].mean(axis=1)
        Aa, Ab = np.hypot(va[:, 0], va[:, 1]), np.hypot(vb[:, 0], vb[:, 1])
        da = Aa - Ab
        ang_a = np.arctan2(va[:, 1], va[:, 0])
        ang_b = np.arctan2(vb[:, 1], vb[:, 0])
        dp = _wrap_angle(ang_a - ang_b) * period / (2.0 * np.pi)
        out["amplitude"] = {
            "amplitude_a": amp_a, "amplitude_b": amp_b, "diff": d_amp,
            "ci": (float(np.quantile(da, alpha / 2)),
                   float(np.quantile(da, 1 - alpha / 2))),
            "p_value_boot": float(2 * min((da <= 0).mean(), (da >= 0).mean())),
            "ratio": float(amp_a / amp_b) if amp_b > 0 else float("nan"),
            "test": "subject-level nonparametric bootstrap of the vector-mean amplitude",
        }
        out["acrophase"] = {
            "acrophase_a_h": phi_a, "acrophase_b_h": phi_b, "diff_h": d_phi,
            "ci_h": (float(np.quantile(dp, alpha / 2)),
                     float(np.quantile(dp, 1 - alpha / 2))),
            "p_value_boot": float(2 * min((dp <= 0).mean(), (dp >= 0).mean())),
            "test": ("subject-level nonparametric bootstrap of the circular "
                     "acrophase difference, wrapped to (-period/2, period/2]"),
        }
    else:
        out["amplitude"] = {"diff": d_amp, "reason": "Need >=3 subjects per group to bootstrap."}
        out["acrophase"] = {"diff_h": d_phi, "reason": "Need >=3 subjects per group to bootstrap."}

    # ── suppress a meaningless phase contrast ───────────────────────────────
    ha = _hotelling_one_sample(Va, alpha) if ka >= 3 else {"p_value": np.nan}
    hb = _hotelling_one_sample(Vb, alpha) if kb >= 3 else {"p_value": np.nan}
    rhythmic_a = bool(np.isfinite(ha.get("p_value", np.nan)) and ha["p_value"] < alpha)
    rhythmic_b = bool(np.isfinite(hb.get("p_value", np.nan)) and hb["p_value"] < alpha)
    out["both_groups_rhythmic"] = rhythmic_a and rhythmic_b
    if not (rhythmic_a and rhythmic_b):
        out["acrophase"]["interpretable"] = False
        out["acrophase"]["caveat"] = (
            f"Acrophase difference is NOT interpretable: rhythm detected in "
            f"{name_a}={rhythmic_a}, {name_b}={rhythmic_b}. The phase of a rhythm whose "
            "amplitude is not distinguishable from zero is an arbitrary angle, so a "
            "difference between such phases carries no information."
        )
    else:
        out["acrophase"]["interpretable"] = True

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. SEPARATING FATIGUE / TESTING ORDER FROM CIRCADIAN PHASE
# ═══════════════════════════════════════════════════════════════════════════
def _vif_table(X: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """
    Variance inflation factor for every non-intercept column of X.

        VIF_j = 1 / (1 - R^2_j)

    where R^2_j is from regressing column j on all the other columns (intercept
    included). VIF_j is the factor by which Var(beta_j_hat) is multiplied by
    collinearity with the rest of the design. VIF = inf means column j is an
    exact linear combination of the others: the parameters are not separately
    identified and this is a property of the DESIGN, not of the sample size.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    rows = []
    for j in range(p):
        nm = names[j]
        if nm.lower() in ("intercept", "const", "1"):
            continue
        xj = X[:, j]
        others = np.delete(X, j, axis=1)
        if not np.any(np.all(np.isclose(others, others[0, :], atol=0), axis=0)):
            others = np.column_stack([np.ones(X.shape[0]), others])
        sst = float(np.sum((xj - xj.mean()) ** 2))
        if sst <= 0:
            rows.append({"term": nm, "vif": float("inf"), "r2_on_others": 1.0})
            continue
        coef, *_ = np.linalg.lstsq(others, xj, rcond=None)
        ssr = float(np.sum((xj - others @ coef) ** 2))
        r2 = 1.0 - ssr / sst
        r2 = min(max(r2, 0.0), 1.0)
        vif = float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"term": nm, "vif": vif, "r2_on_others": r2})
    return pd.DataFrame(rows)


def separate_fatigue_from_rhythm(
    df: pd.DataFrame,
    zt_col: str,
    value_col: str,
    order_col: str = "SessionIndex",
    subject_col: Optional[str] = None,
    period: float = 24.0,
    weight_col: Optional[str] = None,
    alpha: float = 0.05,
    vif_warn: float = 10.0,
) -> dict:
    """
    Fit a model containing BOTH a circadian term and a testing-order term, and
    decide whether the data can tell them apart.

        y = M + beta*cos(w*ZT) + gamma*sin(w*ZT) + delta*order + e

    THE PROBLEM
    -----------
    If every fly is tested morning, then afternoon, then evening, the testing
    order is a deterministic function of the time of day. The columns
    [1, cos(w*ZT), sin(w*ZT), order] then span a 3-dimensional space, not a
    4-dimensional one: the design matrix is RANK DEFICIENT. Infinitely many
    (beta, gamma, delta) triples fit the data equally well and no estimator can
    choose between them. Collecting more flies does not help, because every fly
    adds the same three rows. This is a property of the schedule, so it must be
    detected structurally -- which is what the rank check below does -- and not
    by looking at a p-value.

    A monotone decline across the day is then EXACTLY as consistent with
    "flies climb worse in the evening" as with "flies climb worse the third
    time you tap them". When that holds this function sets
    `separable = False` and REFUSES to attribute the effect to circadian phase.

    WHAT THE LITERATURE SAYS ABOUT BOTH TERMS
    -----------------------------------------
    Gargano et al. (2005, Exp Gerontol 40:386-395) report that RING performance
    was influenced by NEITHER time of day NOR repeated testing under the
    conditions they tested. That is genuine evidence that both effects can be
    small; it is not a guarantee that they are zero in another laboratory, with
    other genotypes, other inter-trial intervals or an older cohort. The right
    response is to keep both terms in the model and to counterbalance the
    schedule so that they remain separable -- which is what `design_advisor`
    recommends.

    Parameters
    ----------
    order_col : cumulative testing-order covariate. `SessionIndex` (how many
        sessions this subject has been through) and `TrialIndex` (which tap
        within a session) are both recognised by `design_schema`; pass
        whichever is the relevant nuisance axis.
    subject_col : if given, standard errors are cluster-robust on this column,
        because repeated measurements of one fly are not independent.

    Returns
    -------
    dict. Read `verdict` and `separable` first. `attribution_allowed` is the
    flag downstream code should gate on before saying the word "circadian".
    """
    for col in (zt_col, value_col, order_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe.")

    keep = [zt_col, value_col, order_col]
    if subject_col:
        keep.append(subject_col)
    if weight_col:
        keep.append(weight_col)
    work = df[[c for c in dict.fromkeys(keep)]].copy()
    for c in (zt_col, value_col, order_col):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if weight_col:
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work.dropna(subset=[zt_col, value_col, order_col])
    if work.empty:
        return {"available": False, "separable": False, "attribution_allowed": False,
                "reason": "No rows with ZT, response and testing order all present."}

    t = work[zt_col].to_numpy(dtype=float) % period
    y = work[value_col].to_numpy(dtype=float)
    o = work[order_col].to_numpy(dtype=float)
    n = int(t.size)

    w_ang = 2.0 * np.pi / period
    X = np.column_stack([np.ones(n), np.cos(w_ang * t), np.sin(w_ang * t), o])
    names = ["Intercept", "cos_ZT", "sin_ZT", order_col]

    rank = int(np.linalg.matrix_rank(X, tol=1e-9))
    vif = _vif_table(X, names)
    max_vif = float(vif["vif"].max()) if len(vif) else float("nan")

    # ── the categorical confounding check from design_schema ────────────────
    tmp = work.copy()
    tmp["_ZT_label"] = np.round(t, 3).astype(str)
    tmp["_order_label"] = np.round(o, 3).astype(str)
    conf = _ds.check_confounding(tmp, "_ZT_label", "_order_label")

    separable = (rank == X.shape[1]) and np.isfinite(max_vif)
    complete_confounding = bool(conf.get("available") and conf.get("severity", 0.0) >= 1.0)

    out: dict = {
        "available": True,
        "n_obs": n,
        "n_distinct_zt": int(np.unique(np.round(t, 9)).size),
        "n_distinct_order": int(np.unique(np.round(o, 9)).size),
        "design_rank": rank,
        "design_ncol": int(X.shape[1]),
        "vif": vif,
        "max_vif": max_vif,
        "confounding_check": {k: v for k, v in conf.items() if k != "crosstab"},
        "crosstab": conf.get("crosstab"),
        "separable": bool(separable),
        "attribution_allowed": bool(separable and not complete_confounding),
        "period": period,
        "order_col": order_col,
    }

    if not separable or complete_confounding:
        out["verdict"] = (
            f"REFUSING TO ATTRIBUTE THIS EFFECT TO CIRCADIAN PHASE. Time of day and "
            f"'{order_col}' are completely collinear in this design: the four model "
            f"columns [1, cos(ZT), sin(ZT), {order_col}] have rank {rank}, so the "
            "circadian coefficients and the testing-order coefficient are not separately "
            "identified. Any apparent time-of-day effect is equally well explained by "
            "cumulative handling, fatigue or habituation across the three sessions, and "
            "no statistical method can distinguish them from these data. "
            "FIX THE DESIGN, not the analysis: counterbalance the order in which subjects "
            "visit the three times of day (design_advisor() will generate a schedule). "
            + (conf.get("verdict", "") if conf.get("available") else "")
        )
        out["recommendation"] = (
            "Randomise or counterbalance testing order across time of day (e.g. a Latin "
            "square over {morning, afternoon, evening} assigned cyclically across "
            "subjects), or test independent cohorts at each time of day so that no "
            "subject contributes more than one session per day."
        )
        # Report what a rhythm-only model would say, clearly labelled as
        # uninterpretable, so the caller can see the size of the ambiguity.
        try:
            fit_r = cosinor_fit(t, y, period=period,
                                weights=work[weight_col].to_numpy() if weight_col else None,
                                alpha=alpha, n_boot=0)
            out["rhythm_only_fit_UNINTERPRETABLE"] = {
                "amplitude": fit_r.amplitude, "acrophase_h": fit_r.acrophase_h,
                "f_pvalue": fit_r.f_pvalue,
                "caveat": ("This is the fit you would get if you ignored testing order. "
                           "It is reported ONLY to show the magnitude of what is "
                           "ambiguous. It must not be described as a circadian effect."),
            }
        except Exception:
            pass
        return out

    # ── both terms identified: fit them jointly ─────────────────────────────
    import statsmodels.api as sm

    Xdf = pd.DataFrame(X, columns=names)
    wts = work[weight_col].to_numpy(dtype=float) if weight_col else None

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        if wts is not None:
            model = sm.WLS(y, Xdf, weights=wts)
        else:
            model = sm.OLS(y, Xdf)
        if subject_col and work[subject_col].nunique() >= 2:
            groups = work[subject_col].to_numpy()
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            se_note = (
                f"Standard errors are cluster-robust on '{subject_col}' "
                f"({work[subject_col].nunique()} clusters); repeated measurements of one "
                "subject are not independent. With few clusters this correction is "
                "known to be anticonservative."
            )
        else:
            res = model.fit()
            se_note = ("Standard errors assume independent rows. If the same subject "
                       "contributes several rows this is anticonservative -- pass "
                       "subject_col.")

    params = pd.Series(res.params.to_numpy(), index=names)
    bse = pd.Series(res.bse.to_numpy(), index=names)
    pvals = pd.Series(res.pvalues.to_numpy(), index=names)
    ci = res.conf_int(alpha=alpha)
    ci = np.asarray(ci)

    # Joint test that the rhythm is present AFTER adjusting for testing order
    R = np.zeros((2, 4)); R[0, 1] = 1.0; R[1, 2] = 1.0
    wald = res.f_test(R)
    f_rhythm = float(np.ravel(wald.fvalue)[0])
    p_rhythm = float(np.ravel(wald.pvalue)[0])

    beta, gamma = float(params["cos_ZT"]), float(params["sin_ZT"])
    amp_adj, phi_adj = _polar_from_bg(beta, gamma, period)
    cov_bg = np.asarray(res.cov_params())[1:3, 1:3]
    df_resid = int(getattr(res, "df_resid", n - 4))
    c = 2.0 * float(stats.f.ppf(1.0 - alpha, 2, max(df_resid, 1)))
    amp_ci, phi_ci, origin_inside = _ellipse_polar_bounds(
        np.array([beta, gamma]), cov_bg, c, period
    )

    delta = float(params[order_col])
    p_order = float(pvals[order_col])

    out.update({
        "coefficients": pd.DataFrame({
            "coef": params, "std_err": bse, "p_value": pvals,
            "ci_lower": ci[:, 0], "ci_upper": ci[:, 1],
        }),
        "adjusted_amplitude": amp_adj,
        "adjusted_amplitude_ci": amp_ci,
        "adjusted_acrophase_h": phi_adj,
        "adjusted_acrophase_ci": phi_ci,
        "adjusted_rhythm_origin_in_ellipse": origin_inside,
        "rhythm_F": f_rhythm,
        "rhythm_p": p_rhythm,
        "rhythm_detected_adjusted": bool(p_rhythm < alpha),
        "order_effect": delta,
        "order_effect_ci": (float(ci[3, 0]), float(ci[3, 1])),
        "order_p": p_order,
        "order_effect_detected": bool(p_order < alpha),
        "se_note": se_note,
        "r_squared": float(res.rsquared),
    })

    if max_vif > vif_warn:
        out.setdefault("warnings", []).append(
            f"Maximum VIF = {max_vif:.1f} (> {vif_warn:g}). Time of day and "
            f"'{order_col}' are separable but heavily correlated: the standard errors on "
            f"the circadian coefficients are inflated by a factor of about "
            f"{np.sqrt(max_vif):.1f} relative to an orthogonal design, so a real rhythm "
            "may well be missed. Counterbalance more aggressively."
        )

    bits = [
        f"Time of day and '{order_col}' ARE separable in this design "
        f"(design rank {rank} = {X.shape[1]} columns, max VIF = {max_vif:.2f})."
    ]
    if out["rhythm_detected_adjusted"]:
        bits.append(
            f"After adjusting for testing order, a 24 h rhythm remains "
            f"(F(2,{df_resid}) = {f_rhythm:.3f}, p = {p_rhythm:.4g}); adjusted amplitude "
            f"{amp_adj:.4g} [{amp_ci[0]:.4g}, {amp_ci[1]:.4g}], peak ZT {phi_adj:.2f} h."
        )
    else:
        bits.append(
            f"After adjusting for testing order, NO 24 h rhythm is detected "
            f"(F(2,{df_resid}) = {f_rhythm:.3f}, p = {p_rhythm:.4g}). This is consistent "
            "with Gargano et al. (2005), who found no effect of time of day on RING "
            "performance."
        )
    if out["order_effect_detected"]:
        bits.append(
            f"A testing-order effect IS present ({order_col} coefficient "
            f"{delta:+.4g} per session, p = {p_order:.4g}): performance changes with "
            "cumulative testing independently of the clock."
        )
    else:
        bits.append(
            f"No testing-order effect is detected ({order_col} coefficient "
            f"{delta:+.4g}, p = {p_order:.4g})."
        )
    out["verdict"] = " ".join(bits)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. AMPLITUDE AS AN EARLY BIOMARKER -- A HYPOTHESIS, NOT A FINDING
# ═══════════════════════════════════════════════════════════════════════════
def amplitude_biomarker(
    df: pd.DataFrame,
    zt_col: str,
    value_col: str,
    subject_col: str,
    group_col: str,
    control_group: Optional[str] = None,
    period: float = 24.0,
    weight_col: Optional[str] = None,
    alpha: float = 0.05,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """
    Test the hypothesis that the circadian AMPLITUDE of climbing degrades
    EARLIER, or more, than the MESOR in a disease genotype.

    THIS IS A HYPOTHESIS THE MODULE CAN TEST. IT IS NOT AN ESTABLISHED FACT.
    -----------------------------------------------------------------------
    The reasoning behind it is that a rhythmic output requires an intact clock,
    intact output pathways AND enough motor reserve to express the swing, so it
    might collapse while the daily average is still within normal limits. That
    is a plausible mechanism, not a demonstrated one, and the strongest directly
    relevant published result runs the other way: Gargano et al. (2005) found NO
    effect of time of day on RING climbing performance at all, which -- if it
    generalises -- means there may be no amplitude to lose. Vaccaro et al.
    (2017, PLoS Genet 13:e1006507) do link clock disruption to accelerated
    senescence of startle-induced negative geotaxis, but they measured climbing
    at a fixed time and did not report a time-of-day rhythm in it.

    So the correct reading of a positive result here is "this dataset is
    consistent with the amplitude-first hypothesis", and the correct reading of
    a null is "no evidence for it", which is also the prior expectation.

    WHAT IS COMPUTED
    ----------------
    A population cosinor per group (Bingham et al. 1982), then, for each
    disease group against the control:

      * absolute and RELATIVE change in MESOR and in amplitude, each with a
        subject-level bootstrap interval. The relative scale matters because
        MESOR and amplitude are not on comparable absolute scales -- an
        amplitude is a half-swing, a MESOR is a level -- so "amplitude degrades
        first" can only be assessed as a proportional loss;
      * the DIFFERENTIAL statistic that actually encodes the hypothesis,

            D = (relative amplitude loss) - (relative MESOR loss)

        with a bootstrap CI. D > 0 with an interval excluding zero is the
        evidence the hypothesis predicts. D is estimated by resampling
        SUBJECTS, so control and disease uncertainty both propagate;
      * Holm-adjusted p-values across the {MESOR, amplitude} family, because
        the claim "amplitude but not MESOR" is a two-test claim and reporting
        two unadjusted p-values would overstate it.

    Returns
    -------
    dict with 'hypothesis', 'by_group', 'contrasts' and, per contrast, an
    explicit `evidence` string that states what the numbers do and do not
    support.
    """
    pop = population_cosinor(
        df, zt_col=zt_col, value_col=value_col, subject_col=subject_col,
        group_col=group_col, period=period, weight_col=weight_col,
        alpha=alpha, n_boot=n_boot, seed=seed,
    )
    per_subject = pop["per_subject"]
    if per_subject.empty or group_col not in per_subject.columns:
        return {
            "available": False,
            "reason": "No subject had 3+ distinct times of day, so no rhythm vectors exist.",
            "population": pop,
        }

    groups = sorted(per_subject[group_col].astype(str).unique())
    if control_group is None:
        # Prefer an explicitly control-like label; otherwise refuse to guess.
        cands = [g for g in groups
                 if any(k in g.lower() for k in
                        ("control", "wt", "wild", "oregon", "canton", "vehicle", "ctrl"))]
        if len(cands) == 1:
            control_group = cands[0]
        else:
            return {
                "available": False,
                "reason": (
                    f"control_group was not supplied and could not be inferred "
                    f"unambiguously from the levels {groups}. Name the control group "
                    "explicitly -- guessing which genotype is the reference would "
                    "silently determine the sign of every effect reported."
                ),
                "population": pop,
            }
    if str(control_group) not in groups:
        raise ValueError(
            f"control_group '{control_group}' not among the levels of "
            f"'{group_col}': {groups}"
        )
    control_group = str(control_group)

    def _vecs(g: str) -> Tuple[np.ndarray, np.ndarray]:
        tab = per_subject[per_subject[group_col].astype(str) == g]
        V = tab[["beta", "gamma"]].to_numpy(dtype=float)
        M = tab["mesor"].to_numpy(dtype=float)
        ok = np.all(np.isfinite(V), axis=1) & np.isfinite(M)
        return V[ok], M[ok]

    Vc, Mc = _vecs(control_group)
    amp_c = float(np.hypot(*Vc.mean(axis=0))) if Vc.size else float("nan")
    mesor_c = float(Mc.mean()) if Mc.size else float("nan")

    rng = np.random.default_rng(seed + 4242)
    contrasts = []
    for g in groups:
        if g == control_group:
            continue
        Vd, Md = _vecs(g)
        kc, kd = int(Vc.shape[0]), int(Vd.shape[0])
        if kc < 3 or kd < 3:
            contrasts.append({
                "group": g, "control": control_group, "available": False,
                "reason": f"Need >=3 subjects per group (control k={kc}, {g} k={kd}).",
            })
            continue

        amp_d = float(np.hypot(*Vd.mean(axis=0)))
        mesor_d = float(Md.mean())

        d_amp = amp_d - amp_c
        d_mesor = mesor_d - mesor_c
        rel_amp = -d_amp / abs(amp_c) if amp_c != 0 else float("nan")      # loss, positive = degraded
        rel_mesor = -d_mesor / abs(mesor_c) if mesor_c != 0 else float("nan")
        D = rel_amp - rel_mesor

        B = int(n_boot)
        ic = rng.integers(0, kc, size=(B, kc))
        idd = rng.integers(0, kd, size=(B, kd))
        vc = Vc[ic].mean(axis=1); vd = Vd[idd].mean(axis=1)
        Ac = np.hypot(vc[:, 0], vc[:, 1]); Ad = np.hypot(vd[:, 0], vd[:, 1])
        mc = Mc[ic].mean(axis=1); md = Md[idd].mean(axis=1)

        b_d_amp = Ad - Ac
        b_d_mesor = md - mc
        with np.errstate(divide="ignore", invalid="ignore"):
            b_rel_amp = np.where(Ac != 0, -b_d_amp / np.abs(Ac), np.nan)
            b_rel_mesor = np.where(mc != 0, -b_d_mesor / np.abs(mc), np.nan)
        b_D = b_rel_amp - b_rel_mesor
        b_D = b_D[np.isfinite(b_D)]

        def _ci(x):
            x = np.asarray(x); x = x[np.isfinite(x)]
            if x.size == 0:
                return (float("nan"), float("nan"))
            return (float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2)))

        def _p2(x):
            x = np.asarray(x); x = x[np.isfinite(x)]
            if x.size == 0:
                return float("nan")
            return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))

        p_amp = _p2(b_d_amp)
        p_mesor = _p2(b_d_mesor)
        # Holm across the two-member family {MESOR, amplitude}
        order = sorted([("MESOR", p_mesor), ("amplitude", p_amp)], key=lambda kv: kv[1])
        holm = {}
        prev = 0.0
        for rank_i, (nm, p) in enumerate(order):
            adj = min(1.0, max(prev, (2 - rank_i) * p))
            holm[nm] = adj
            prev = adj

        ci_D = _ci(b_D)
        p_D = _p2(b_D)

        # A smaller population amplitude can arise from damped individual
        # rhythms OR from desynchronised phases. Quantify which is in play
        # before the verdict is written.
        disp_c = _phase_dispersion(Vc, period)
        disp_d = _phase_dispersion(Vd, period)
        Rc, Rd = disp_c["synchronisation_index_R"], disp_d["synchronisation_index_R"]
        mean_ind_c = float(np.mean(np.hypot(Vc[:, 0], Vc[:, 1])))
        mean_ind_d = float(np.mean(np.hypot(Vd[:, 0], Vd[:, 1])))
        desync_caveat = ""
        if np.isfinite(Rc) and np.isfinite(Rd) and Rc > 0 and Rd < Rc * 0.85:
            desync_caveat = (
                f" MECHANISM AMBIGUOUS: the {g} flies are also less synchronised with "
                f"each other (R = {Rd:.3f} vs {Rc:.3f} in {control_group}; mean "
                f"INDIVIDUAL amplitude {mean_ind_d:.3f} vs {mean_ind_c:.3f}). Part or all "
                "of the population-amplitude loss may be phase DESYNCHRONISATION between "
                "flies rather than damping within each fly. Those are different biology; "
                "distinguishing them needs more timepoints per fly, not more flies."
            )
        elif np.isfinite(Rd) and np.isfinite(Rc):
            desync_caveat = (
                f" Synchronisation is comparable between groups (R = {Rd:.3f} vs "
                f"{Rc:.3f}), so the amplitude change is not explained by phase "
                "desynchronisation between flies."
            )

        supports = bool(np.isfinite(ci_D[0]) and ci_D[0] > 0)
        if supports:
            evidence = (
                f"CONSISTENT WITH the amplitude-first hypothesis in this dataset: the "
                f"relative amplitude loss ({100 * rel_amp:.1f}%) exceeds the relative "
                f"MESOR loss ({100 * rel_mesor:.1f}%), differential D = {D:+.3f} "
                f"[{ci_D[0]:+.3f}, {ci_D[1]:+.3f}], bootstrap p = {p_D:.4g}. This is "
                "evidence from one dataset, not a demonstration that amplitude is an "
                "early biomarker; it needs replication in an independent cohort with a "
                "counterbalanced schedule, and it must be checked against "
                "separate_fatigue_from_rhythm() before the word 'circadian' is used."
            )
        elif np.isfinite(ci_D[1]) and ci_D[1] < 0:
            evidence = (
                f"CONTRARY to the amplitude-first hypothesis: the MESOR degrades "
                f"proportionally MORE than the amplitude (D = {D:+.3f} "
                f"[{ci_D[0]:+.3f}, {ci_D[1]:+.3f}], p = {p_D:.4g})."
            )
        else:
            evidence = (
                f"NO EVIDENCE either way: the differential D = {D:+.3f} has interval "
                f"[{ci_D[0]:+.3f}, {ci_D[1]:+.3f}] which includes zero (p = {p_D:.4g}). "
                f"With k = {kc} control and {kd} {g} subjects this study may simply lack "
                "the power to separate the two; design_advisor() will say how many "
                "subjects would be needed. A null here is also the outcome expected from "
                "Gargano et al. (2005), who reported no time-of-day effect on climbing."
            )

        contrasts.append({
            "group": g, "control": control_group, "available": True,
            "k_control": kc, "k_group": kd,
            "mesor_control": mesor_c, "mesor_group": mesor_d,
            "mesor_diff": d_mesor, "mesor_diff_ci": _ci(b_d_mesor),
            "mesor_relative_loss": rel_mesor,
            "mesor_p_boot": p_mesor, "mesor_p_holm": holm["MESOR"],
            "amplitude_control": amp_c, "amplitude_group": amp_d,
            "amplitude_diff": d_amp, "amplitude_diff_ci": _ci(b_d_amp),
            "amplitude_relative_loss": rel_amp,
            "amplitude_p_boot": p_amp, "amplitude_p_holm": holm["amplitude"],
            "differential_D": D, "differential_D_ci": ci_D, "differential_D_p": p_D,
            "supports_amplitude_first": supports,
            "synchronisation_control": Rc, "synchronisation_group": Rd,
            "mean_individual_amplitude_control": mean_ind_c,
            "mean_individual_amplitude_group": mean_ind_d,
            "phase_dispersion_control_h": disp_c["phase_dispersion_sd_h"],
            "phase_dispersion_group_h": disp_d["phase_dispersion_sd_h"],
            "evidence": evidence + desync_caveat,
        })

    return {
        "available": True,
        "hypothesis": (
            "HYPOTHESIS (not an established finding): in a disease genotype the "
            "circadian AMPLITUDE of climbing performance degrades earlier, or "
            "proportionally more, than the MESOR (rhythm-adjusted mean). Formally, "
            "D = (relative amplitude loss) - (relative MESOR loss) > 0. The published "
            "evidence most directly relevant to the premise is a NULL: Gargano et al. "
            "(2005, Exp Gerontol 40:386-395) found no effect of time of day on RING "
            "climbing performance under the conditions they tested."
        ),
        "control_group": control_group,
        "by_group": pop["by_group"],
        "population": pop,
        "contrasts": contrasts,
        "prerequisite": (
            "This analysis is only meaningful if time of day is separable from "
            "cumulative testing order. Run separate_fatigue_from_rhythm() first; if it "
            "reports attribution_allowed=False, nothing here may be described as "
            "circadian."
        ),
        "alpha": alpha,
        "n_boot": n_boot,
        "seed": seed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. DESIGN ADVISOR -- ANSWER BEFORE THE EXPERIMENT, NOT AFTER
# ═══════════════════════════════════════════════════════════════════════════
def _latin_square(m: int) -> np.ndarray:
    """Cyclic Latin square of order m; row i is the visiting order for subject i."""
    return np.array([[(i + j) % m for j in range(m)] for i in range(m)], dtype=int)


def design_advisor(
    zt_values: Sequence[float],
    n_subjects: int,
    period: float = 24.0,
    sigma: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    order_is_zt_order: bool = True,
) -> dict:
    """
    Evaluate a proposed time-of-day sampling schedule BEFORE the flies are run.

    Parameters
    ----------
    zt_values : the Zeitgeber hours at which each subject will be assayed.
        Either a 1-D sequence (the same schedule for every subject) or a 2-D
        array of shape (n_subjects, m) giving a per-subject schedule.
    n_subjects : number of independent subjects (flies, or vials -- whichever is
        the true unit of replication; see `design_schema.describe_design`).
    sigma : assumed residual standard deviation on the ANALYSIS scale. If the
        response is the empirical logit, sigma is in logit units; a value of 1.0
        makes every precision figure read as "in units of sigma".
    power : target power for the minimum detectable amplitude.
    order_is_zt_order : whether subjects will visit the times of day in
        ascending ZT order (the default laboratory habit). If True with a shared
        schedule, testing order is perfectly collinear with time of day and the
        advisor says so.

    Returns
    -------
    dict with `identifiable`, precision figures, the minimum detectable
    amplitude at the requested power, a counterbalancing schedule, and
    `recommendations`.

    A NOTE ON WHAT THE PRECISION FIGURES ASSUME
    -------------------------------------------
    The standard errors below come from sigma^2 (X'X)^{-1} on the STACKED design
    over all subjects. That treats every observation as contributing independent
    information about one common rhythm -- i.e. it assumes NO between-subject
    variation in (beta, gamma). Real flies differ, and the population-mean
    cosinor uses the between-subject spread as its error term, so the achievable
    standard errors are LARGER than these. Read them as a best case / lower
    bound, not as a promise.
    """
    zt = np.asarray(zt_values, dtype=float)
    n_subjects = int(n_subjects)
    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1.")
    if zt.ndim == 1:
        shared = True
        schedule = np.tile(zt % period, (n_subjects, 1))
        m = int(zt.size)
    elif zt.ndim == 2:
        shared = False
        schedule = zt % period
        if schedule.shape[0] != n_subjects:
            raise ValueError(
                f"zt_values has {schedule.shape[0]} rows but n_subjects = {n_subjects}."
            )
        m = int(schedule.shape[1])
    else:
        raise ValueError("zt_values must be 1-D or 2-D.")

    flat = schedule.ravel()
    N = int(flat.size)
    distinct = np.unique(np.round(flat, 6))
    rec: List[str] = []
    warns: List[str] = []

    X = _design_matrix(flat, period)
    rank = int(np.linalg.matrix_rank(X, tol=1e-10))
    identifiable = rank == 3

    XtX = X.T @ X
    try:
        cond = float(np.linalg.cond(XtX))
    except Exception:
        cond = float("inf")

    out: dict = {
        "n_subjects": n_subjects,
        "n_timepoints_per_subject": m,
        "n_observations": N,
        "distinct_zt": distinct.tolist(),
        "n_distinct_zt": int(distinct.size),
        "period": period,
        "sigma": float(sigma),
        "identifiable": bool(identifiable),
        "design_rank": rank,
        "condition_number": cond,
        "shared_schedule": bool(shared),
    }

    if not identifiable:
        out["amplitude_identifiable"] = False
        out["acrophase_identifiable"] = False
        out["verdict"] = (
            f"NOT IDENTIFIABLE: {int(distinct.size)} distinct time(s) of day give a "
            f"cosinor design matrix of rank {rank} < 3. Amplitude and acrophase cannot be "
            "estimated at all from this schedule, at any sample size -- more flies will "
            "not help. Use at least three well-separated times of day."
        )
        out["recommendations"] = [
            "Sample at >=3 distinct times of day spread across the cycle; "
            "ZT2 / ZT8 / ZT14 (or 4 points at ZT1/ZT7/ZT13/ZT19) are workable choices "
            "under 12:12 LD.",
        ]
        return out

    out["amplitude_identifiable"] = True
    out["acrophase_identifiable"] = True

    # ── cycle coverage: how much of the 24 h circle the samples span ────────
    ang = np.sort((distinct % period) * 2 * np.pi / period)
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    largest_gap_h = float(gaps.max() * period / (2 * np.pi))
    out["largest_sampling_gap_h"] = largest_gap_h
    if largest_gap_h > period / 2:
        warns.append(
            f"The largest unsampled stretch is {largest_gap_h:.1f} h, more than half the "
            f"{period:g} h cycle. All samples fall within one arc, so the fitted cosine is "
            "extrapolating across the unobserved half. The acrophase estimate will be "
            "strongly leveraged by the endpoints and the amplitude is only weakly "
            "constrained."
        )

    # ── precision of (beta, gamma) and hence of amplitude / acrophase ───────
    Cov = float(sigma) ** 2 * np.linalg.inv(XtX)
    se_beta = float(np.sqrt(Cov[1, 1]))
    se_gamma = float(np.sqrt(Cov[2, 2]))
    se_mesor = float(np.sqrt(Cov[0, 0]))
    out["se_mesor"] = se_mesor
    out["se_beta"] = se_beta
    out["se_gamma"] = se_gamma
    out["cov_beta_gamma"] = Cov[1:, 1:].tolist()

    # Information about (beta, gamma) after projecting out the intercept:
    # the Schur complement. Its eigenvalues give the best and worst directions.
    Z = X[:, 1:]
    one = X[:, :1]
    Minf = Z.T @ Z - Z.T @ one @ np.linalg.inv(one.T @ one) @ one.T @ Z
    ev = np.linalg.eigvalsh(Minf)
    lam_min, lam_max = float(ev.min()), float(ev.max())
    out["rhythm_information_eigenvalues"] = [lam_min, lam_max]
    out["phase_isotropy"] = float(lam_min / lam_max) if lam_max > 0 else 0.0
    if out["phase_isotropy"] < 0.5:
        warns.append(
            f"The schedule is anisotropic in phase (eigenvalue ratio "
            f"{out['phase_isotropy']:.2f}): rhythms peaking at some clock times will be "
            "detected far more easily than rhythms peaking at others. Spread the sampling "
            "times more evenly around the cycle to make sensitivity phase-independent."
        )

    # ── minimum detectable amplitude at the requested power ─────────────────
    # Zero-amplitude F-test: F ~ F(2, N-3) under H0 and noncentral F with
    # noncentrality lambda = theta' M theta / sigma^2 under the alternative.
    # For a fixed amplitude A the noncentrality depends on the (unknown) phase
    # and lies in [A^2 lam_min, A^2 lam_max]/sigma^2, so the WORST CASE
    # (lam_min) is quoted -- the amplitude you are guaranteed to detect
    # whatever the phase turns out to be.
    df2 = N - 3
    out["residual_df"] = int(df2)
    if df2 <= 0:
        warns.append(
            f"{N} observations for 3 parameters leaves {df2} residual degrees of freedom; "
            "no test is possible. More observations or more subjects are required."
        )
        out["minimum_detectable_amplitude"] = float("nan")
    else:
        fcrit = float(stats.f.ppf(1.0 - alpha, 2, df2))

        def _power_at(A: float, lam_coef: float) -> float:
            lam = (A ** 2) * lam_coef / (float(sigma) ** 2)
            return float(stats.ncf.sf(fcrit, 2, df2, lam))

        def _solve(lam_coef: float) -> float:
            if lam_coef <= 0:
                return float("nan")
            lo, hi = 1e-9, 1.0
            for _ in range(200):
                if _power_at(hi, lam_coef) >= power:
                    break
                hi *= 2.0
                if hi > 1e9:
                    return float("nan")
            try:
                return float(optimize.brentq(
                    lambda A: _power_at(A, lam_coef) - power, lo, hi, xtol=1e-10
                ))
            except Exception:
                return float("nan")

        mda_worst = _solve(lam_min)
        mda_best = _solve(lam_max)
        out["minimum_detectable_amplitude"] = mda_worst
        out["minimum_detectable_amplitude_best_phase"] = mda_best
        out["mda_note"] = (
            f"Smallest amplitude detectable with {100 * power:.0f}% power at alpha = "
            f"{alpha:g} by the zero-amplitude F(2,{df2}) test, in the WORST-CASE "
            "acrophase. In the most favourable phase it is "
            f"{mda_best:.4g}. Both assume no between-subject rhythm heterogeneity, so "
            "they are optimistic; see the note in the docstring."
        )
        if np.isfinite(mda_worst) and float(sigma) > 0:
            out["mda_in_sigma_units"] = float(mda_worst / float(sigma))

        # Acrophase precision at the detection boundary and at 2x it.
        # For an isotropic design SE(acrophase in radians) ~ SE(amplitude)/A.
        se_amp_iso = float(sigma) * np.sqrt(2.0 / N)
        out["se_amplitude_orthogonal_reference"] = se_amp_iso
        prec = {}
        for mult, lab in ((1.0, "at_MDA"), (2.0, "at_2x_MDA"), (4.0, "at_4x_MDA")):
            A = mda_worst * mult
            if np.isfinite(A) and A > 0:
                prec[lab] = {
                    "amplitude": float(A),
                    "acrophase_se_h": float(se_amp_iso / A * period / (2 * np.pi)),
                }
        out["acrophase_precision"] = prec
        out["acrophase_precision_note"] = (
            "Acrophase SE scales as 1/amplitude: a weak rhythm has a badly determined "
            "phase no matter how many flies are run. If phase is the quantity of "
            "interest, the amplitude must be comfortably above the detection threshold."
        )

    # ── the order/ZT confound ───────────────────────────────────────────────
    order_confounded = bool(shared and order_is_zt_order)
    out["testing_order_confounded_with_zt"] = order_confounded
    ls = _latin_square(m)
    out["counterbalancing_schedule"] = {
        "description": (
            f"Cyclic Latin square of order {m}. Row r gives the ORDER in which subject "
            "r visits the sampled times of day (entries are indices into "
            "sorted(distinct_zt)). Assign subject i to row i mod "
            f"{m} so that every time of day occurs equally often in every session "
            "position."
        ),
        "sorted_zt": np.sort(distinct).tolist(),
        "square": ls.tolist(),
        "caveat": (
            "A cyclic Latin square balances each time of day across session POSITIONS "
            "but not across immediately preceding conditions. If carry-over from the "
            "previous session is a concern, use a Williams design (a Latin square "
            "balanced for first-order carry-over), which exists for every even m and "
            "requires two squares when m is odd."
        ),
    }

    if order_confounded:
        rec.append(
            "CRITICAL: with a single shared schedule visited in ascending ZT order, "
            "cumulative testing order is a deterministic function of time of day. The "
            "circadian and fatigue/habituation effects will be mathematically "
            "inseparable and separate_fatigue_from_rhythm() will refuse to attribute "
            "anything to the clock. Counterbalance using the Latin square returned "
            "above, or run independent cohorts at each time of day."
        )
    if m < 4:
        rec.append(
            f"{m} times of day exactly identifies the 3 cosinor parameters per subject "
            "and leaves 0 residual degrees of freedom within a subject, so the cosine "
            "shape can never be checked against the data and single-subject inference is "
            "impossible. A 4th time point costs one extra session and buys a residual "
            "degree of freedom, a testable lack-of-fit, and robustness to one missing "
            "session."
        )
    if n_subjects < 8:
        rec.append(
            f"k = {n_subjects} subjects gives the population cosinor F(2, {max(n_subjects - 2, 0)}) "
            "for the zero-amplitude test -- very few denominator degrees of freedom, so "
            "power is poor and the confidence ellipse is large. Aim for k >= 12 per group."
        )
    rec.append(
        "Record the ACTUAL CLOCK TIME of each session, not a category. "
        "design_schema.CATEGORICAL_TOD_ZT maps 'morning'/'afternoon'/'evening' to "
        "conventional midpoints (ZT2/ZT8/ZT11); the acrophase then inherits the full "
        "width of each testing window as unmodelled error."
    )
    rec.append(
        "Record lights-on time for the incubator and pass it as lights_on_hour. Getting "
        "it wrong rigidly rotates every acrophase estimate and will not show up as a "
        "worse fit."
    )
    rec.append(
        "Randomise the ORDER of genotypes within each session as well. If the disease "
        "genotype is always run last, genotype is confounded with within-session "
        "handling time exactly as time of day is confounded with session order."
    )

    out["warnings"] = warns
    out["recommendations"] = rec
    out["verdict"] = (
        f"Schedule is identifiable ({int(distinct.size)} distinct times of day, rank 3, "
        f"condition number {cond:.1f}). With {n_subjects} subjects x {m} sessions = {N} "
        f"observations, the smallest amplitude detectable with {100 * power:.0f}% power "
        f"(worst-case phase, alpha = {alpha:g}) is "
        f"{out.get('minimum_detectable_amplitude', float('nan')):.4g} "
        f"(sigma = {sigma:g}). "
        + ("Testing order is CONFOUNDED with time of day -- counterbalance before running."
           if order_confounded else
           "Testing order is not automatically confounded with time of day in this schedule.")
    )
    return out
