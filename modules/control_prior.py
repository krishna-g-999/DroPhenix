"""
Historical control arms for Drosophila behavioural experiments:
robust meta-analytic-predictive (MAP) priors.

THE MOTIVATION
--------------
A laboratory that has run the same wild-type control cohort twenty times
already knows a great deal about how that control behaves. Repeating a full
concurrent control arm in every new experiment spends flies on a question that
has already been answered many times. Borrowing that information lets the same
number of flies test MORE compounds, or lets the same experiment run with FEWER
flies -- a direct reduction in animal use, and a direct increase in throughput.

THE DANGER, WHICH IS THE WHOLE PROBLEM
--------------------------------------
Historical controls are not exchangeable with current ones by fiat. Fly stocks
drift, food batches change, incubators are replaced, seasons turn, and
experimenters differ. If the historical controls sit even a few percentage
points away from what the current control would have been, the treatment effect
absorbs the difference and the false-positive rate rises -- invisibly, because
nothing in the analysis output announces it.

Naive borrowing (pooling history with current data as if it were one sample) is
therefore among the fastest ways to manufacture a false discovery.

THE METHOD
----------
Robust meta-analytic-predictive priors (Schmidli H, Gsteiger S, Roychoudhury S,
O'Hagan A, Spiegelhalter D, Neuenschwander B (2014) Robust meta-analytic-
predictive priors in clinical trials with historical control information.
Biometrics 70:1023-1032; building on Neuenschwander B, Capkun-Niggli G, Branson
M, Spiegelhalter DJ (2010) Summarizing historical information on controls in
clinical trials. Clin Trials 7:5-18).

  1. Fit a random-effects meta-analysis to the historical control arms, so that
     BETWEEN-EXPERIMENT heterogeneity tau is estimated from the data rather than
     assumed to be zero. Two experiments that disagreed in the past produce a
     wide prior; twenty that agreed produce a narrow one.

  2. Form the PREDICTIVE distribution for a NEW experiment's control parameter.
     This is the MAP prior. Crucially it is wider than the posterior for the
     historical mean, because a new experiment is a new draw from the
     between-experiment distribution -- not another observation of the same one.

  3. ROBUSTIFY: mix the MAP prior with a weakly informative component,

         robust prior = w * MAP + (1 - w) * vague

     If the current control agrees with history, the MAP component dominates and
     the borrowing is efficient. If it conflicts, the posterior mass moves to the
     vague component and the prior is effectively discarded. This is what makes
     the procedure self-limiting rather than merely optimistic.

  4. Report the prior's EFFECTIVE SAMPLE SIZE in flies. That is the honest
     statement of what is being borrowed, and it is what an ethics committee or
     a reviewer should be shown.

  5. Verify OPERATING CHARACTERISTICS by simulation over a grid of drift between
     historical and current controls. Whether borrowing is acceptable is not a
     matter of opinion; it is a matter of the type I error it produces at a
     realistic amount of drift, and that is computable.

COMPUTATION
-----------
Everything here is deterministic -- quadrature and closed-form conjugate
updates, no MCMC. Analyses on the logit scale with normal approximations to the
arm-level likelihood are conjugate to a normal-mixture prior, so the posterior
is itself a normal mixture available in closed form, and
P(treatment > control) is a weighted sum of normal CDFs. This makes results
exactly reproducible and fast enough to simulate operating characteristics with
tens of thousands of replicates, which an MCMC implementation would not be.

SCOPE AND HONESTY
-----------------
This module borrows information about the CONTROL arm only. It never borrows
about a treatment arm, and it never replaces the concurrent control entirely --
a concurrent control is what detects drift in the first place, and
`recommended_design` will not return a design without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit


# ═══════════════════════════════════════════════════════════════════════════
# 1. NORMAL-MIXTURE REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class NormalMixture:
    """
    A prior or posterior represented as sum_k w_k * N(mean_k, sd_k^2).

    Everything downstream (robustification, conjugate update, effective sample
    size, exceedance probability) has a closed form for this representation,
    which is why it is used throughout.
    """

    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    scale: str = "logit"          # "logit" for proportions, "identity" otherwise
    notes: List[str] = field(default_factory=list)
    # Index of the weakly-informative component added by `robustify`, if any.
    # Diagnostics that ask "does the current data conflict with HISTORY?" must
    # exclude it -- otherwise the vague component absorbs any observation and
    # the test can never detect a conflict.
    vague_index: Optional[int] = None

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.sds = np.asarray(self.sds, dtype=float)
        s = self.weights.sum()
        if s <= 0:
            raise ValueError("Mixture weights must sum to a positive number.")
        self.weights = self.weights / s

    # --- moments -----------------------------------------------------------
    @property
    def mean(self) -> float:
        return float(np.sum(self.weights * self.means))

    @property
    def var(self) -> float:
        m = self.mean
        return float(np.sum(self.weights * (self.sds ** 2 + (self.means - m) ** 2)))

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.var))

    def pdf(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        for w, m, s in zip(self.weights, self.means, self.sds):
            out += w * stats.norm.pdf(x, m, s)
        return out

    def cdf(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        for w, m, s in zip(self.weights, self.means, self.sds):
            out += w * stats.norm.cdf(x, m, s)
        return out

    def quantile(self, q: float, lo: float = -20.0, hi: float = 20.0) -> float:
        from scipy.optimize import brentq
        f = lambda x: float(self.cdf(x)[0]) - q
        try:
            return float(brentq(f, lo, hi, xtol=1e-10))
        except ValueError:
            return float("nan")

    def interval(self, level: float = 0.95) -> Tuple[float, float]:
        a = (1.0 - level) / 2.0
        return self.quantile(a), self.quantile(1.0 - a)

    def on_probability_scale(self, level: float = 0.95) -> dict:
        """Summarise on the natural (proportion) scale when scale == 'logit'."""
        if self.scale != "logit":
            lo, hi = self.interval(level)
            return {"mean": self.mean, "ci_lower": lo, "ci_upper": hi}
        lo, hi = self.interval(level)
        return {
            "mean_proportion": float(expit(self.mean)),
            "ci_lower": float(expit(lo)),
            "ci_upper": float(expit(hi)),
            "logit_mean": self.mean,
            "logit_sd": self.sd,
        }

    def compress(self, n_components: int = 3, seed: int = 0,
                 n_samples: int = 40000) -> "NormalMixture":
        """
        Approximate this mixture by one with fewer components, via EM on samples
        drawn from it. Used to turn the fine quadrature mixture into a compact,
        reportable 2-3 component prior as in Schmidli et al. (2014).

        The compression is checked: if it shifts the mean by more than 0.02 or
        the SD by more than 5%, a note is recorded rather than the difference
        being passed over silently.
        """
        if len(self.weights) <= n_components:
            return self

        rng = np.random.default_rng(seed)
        comp = rng.choice(len(self.weights), size=n_samples, p=self.weights)
        x = rng.normal(self.means[comp], self.sds[comp])

        # EM for a k-component univariate Gaussian mixture
        k = n_components
        qs = np.linspace(0.1, 0.9, k)
        mu = np.quantile(x, qs)
        sd = np.full(k, float(np.std(x)) / np.sqrt(k) + 1e-6)
        w = np.full(k, 1.0 / k)

        for _ in range(500):
            dens = np.stack([w[j] * stats.norm.pdf(x, mu[j], sd[j]) for j in range(k)])
            tot = dens.sum(axis=0) + 1e-300
            r = dens / tot
            nk = r.sum(axis=1) + 1e-300
            w_new = nk / nk.sum()
            mu_new = (r * x).sum(axis=1) / nk
            sd_new = np.sqrt(np.clip((r * (x - mu_new[:, None]) ** 2).sum(axis=1) / nk,
                                     1e-8, None))
            if (np.max(np.abs(mu_new - mu)) < 1e-9 and np.max(np.abs(sd_new - sd)) < 1e-9):
                mu, sd, w = mu_new, sd_new, w_new
                break
            mu, sd, w = mu_new, sd_new, w_new

        out = NormalMixture(w, mu, sd, scale=self.scale, notes=list(self.notes))
        if abs(out.mean - self.mean) > 0.02 or abs(out.sd / max(self.sd, 1e-12) - 1) > 0.05:
            out.notes.append(
                f"Compression to {n_components} components shifted the summary from "
                f"mean {self.mean:.4f}/sd {self.sd:.4f} to mean {out.mean:.4f}/sd {out.sd:.4f}. "
                "Use the uncompressed prior for inference."
            )
        return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILDING THE MAP PRIOR
# ═══════════════════════════════════════════════════════════════════════════
def _arm_logit_summaries(
    successes: Sequence[float], totals: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical-logit point estimate and its variance for each historical arm.

    y_i   = log((s + 0.5) / (n - s + 0.5))
    var_i = 1/(s + 0.5) + 1/(n - s + 0.5)

    The 0.5 continuity correction keeps arms with 0% or 100% climbing finite,
    which matters because control arms often sit near the ceiling.
    """
    s = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    y = np.log((s + 0.5) / (n - s + 0.5))
    v = 1.0 / (s + 0.5) + 1.0 / (n - s + 0.5)
    return y, v


def build_map_prior(
    successes: Sequence[float],
    totals: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    tau_prior_scale: float = 0.5,
    n_tau_grid: int = 400,
    tau_max: Optional[float] = None,
) -> dict:
    """
    Meta-analytic-predictive prior for the control parameter of a NEW experiment,
    built from historical control arms.

    Hierarchical model on the logit scale:
        y_i | theta_i ~ N(theta_i, v_i)          (arm-level, v_i known)
        theta_i | mu, tau ~ N(mu, tau^2)         (between-experiment)
        mu ~ flat
        tau ~ HalfNormal(tau_prior_scale)

    With a flat prior on mu, mu integrates out analytically, leaving a
    one-dimensional posterior for tau that is evaluated exactly on a grid. The
    MAP prior is then the tau-mixture of normals

        theta_new | data = INT N(mu_hat(tau), se_mu(tau)^2 + tau^2) p(tau | data) dtau

    which is returned as a `NormalMixture`. Note the `+ tau^2`: a new experiment
    is a NEW DRAW from the between-experiment distribution, not another
    observation of the historical mean. Omitting that term is the single most
    common way to build a MAP prior that is far too confident.

    `tau_prior_scale` = 0.5 on the logit scale is the weakly-informative default
    recommended for binary endpoints in Schmidli et al. (2014); it permits
    substantial heterogeneity while excluding absurd values. It is reported in
    the output so the choice is never hidden.
    """
    s = np.asarray(successes, dtype=float)
    n = np.asarray(totals, dtype=float)
    if s.shape != n.shape or s.ndim != 1:
        raise ValueError("successes and totals must be 1-D arrays of equal length.")
    k = len(s)
    if k < 2:
        raise ValueError(
            f"Only {k} historical control arm(s). A between-experiment variance cannot "
            "be estimated from fewer than 2, and a MAP prior built from one arm is just "
            "that arm asserted as truth. Supply at least 2 (preferably 4 or more)."
        )

    y, v = _arm_logit_summaries(s, n)

    if tau_max is None:
        tau_max = max(4.0 * tau_prior_scale, 2.0)
    tau_grid = np.linspace(1e-6, float(tau_max), int(n_tau_grid))

    # log marginal likelihood of tau with mu integrated out under a flat prior
    log_post = np.empty_like(tau_grid)
    mu_hat = np.empty_like(tau_grid)
    se_mu = np.empty_like(tau_grid)
    for i, tau in enumerate(tau_grid):
        w = 1.0 / (v + tau ** 2)
        sw = w.sum()
        m = float(np.sum(w * y) / sw)
        mu_hat[i] = m
        se_mu[i] = float(np.sqrt(1.0 / sw))
        ll = (
            -0.5 * np.sum(np.log(2 * np.pi * (v + tau ** 2)))
            - 0.5 * float(np.sum(w * (y - m) ** 2))
            - 0.5 * np.log(sw)                      # from integrating out mu
            + 0.5 * np.log(2 * np.pi)
        )
        # HalfNormal(tau_prior_scale) density
        lp = stats.halfnorm.logpdf(tau, scale=tau_prior_scale)
        log_post[i] = ll + lp

    log_post -= log_post.max()
    post = np.exp(log_post)
    # trapezoidal weights so the grid integrates properly
    dtau = np.gradient(tau_grid)
    wts = post * dtau
    wts = wts / wts.sum()

    keep = wts > (1e-8 * wts.max())
    prior = NormalMixture(
        weights=wts[keep],
        means=mu_hat[keep],
        sds=np.sqrt(se_mu[keep] ** 2 + tau_grid[keep] ** 2),
        scale="logit",
    )

    tau_mean = float(np.sum(wts * tau_grid))
    tau_med = float(tau_grid[np.searchsorted(np.cumsum(wts), 0.5)])
    tau_q = [float(tau_grid[np.searchsorted(np.cumsum(wts), q)]) for q in (0.025, 0.975)]

    # I-squared style summary of how much of the variability is between-experiment
    typical_v = float(np.median(v))
    i2 = float(tau_mean ** 2 / (tau_mean ** 2 + typical_v)) if (tau_mean ** 2 + typical_v) > 0 else np.nan

    notes: List[str] = []
    if k < 4:
        notes.append(
            f"Only {k} historical arms. The between-experiment SD tau is poorly determined "
            "with so few, and the resulting prior is correspondingly fragile. Four or more "
            "arms is a reasonable working minimum."
        )
    if tau_mean > 0.5:
        notes.append(
            f"Substantial between-experiment heterogeneity (tau = {tau_mean:.2f} on the logit "
            "scale). The historical arms disagree with each other enough that little "
            "information can be borrowed -- which the prior correctly reflects by being wide."
        )

    summary = prior.on_probability_scale()
    return {
        "prior": prior,
        "n_historical_arms": int(k),
        "arm_logits": y,
        "arm_variances": v,
        "arm_proportions": (s / n).tolist(),
        "labels": list(labels) if labels is not None else [f"arm{i+1}" for i in range(k)],
        "tau_mean": tau_mean,
        "tau_median": tau_med,
        "tau_ci": tau_q,
        "tau_prior": f"HalfNormal(scale={tau_prior_scale})",
        "heterogeneity_fraction": i2,
        "prior_summary": summary,
        "notes": notes,
        "method": (
            "Meta-analytic-predictive prior; normal-normal hierarchical model on the "
            "logit scale with mu integrated analytically and tau by grid quadrature "
            "(Neuenschwander et al. 2010; Schmidli et al. 2014)."
        ),
    }


def robustify(
    prior: NormalMixture,
    weight: float = 0.8,
    vague_sd: float = 2.0,
    vague_mean: Optional[float] = None,
) -> NormalMixture:
    """
    Robust MAP prior: `weight` on the informative MAP component, the remainder
    on a weakly informative component centred at the same location.

    A vague SD of 2.0 on the logit scale corresponds to roughly a 95% interval
    of 2%-98% on the proportion scale -- deliberately uninformative. If the
    current control conflicts with history, the posterior mass shifts to this
    component and the borrowing switches itself off.

    Schmidli et al. (2014) study weights of 0.5 and 0.9; 0.8 is a middle default.
    The weight is the analyst's statement of how much they believe the historical
    arms are exchangeable with the new one, and it should be pre-specified.
    """
    if not (0.0 < weight <= 1.0):
        raise ValueError("weight must be in (0, 1].")
    if weight == 1.0:
        return prior
    m0 = prior.mean if vague_mean is None else float(vague_mean)
    w = np.concatenate([prior.weights * weight, [1.0 - weight]])
    mu = np.concatenate([prior.means, [m0]])
    sd = np.concatenate([prior.sds, [float(vague_sd)]])
    out = NormalMixture(w, mu, sd, scale=prior.scale, notes=list(prior.notes),
                        vague_index=len(w) - 1)
    out.notes.append(
        f"Robustified: {weight:.0%} weight on the MAP component, {1 - weight:.0%} on a "
        f"weakly informative N({m0:.3f}, {vague_sd}^2) component."
    )
    return out


def informative_component(prior: NormalMixture) -> NormalMixture:
    """
    Strip the weakly-informative component from a robustified prior, returning
    the MAP part alone with weights renormalised.

    Needed because two different questions are being asked of the same object:
      * "What should I believe about the control?"  -> use the FULL robust prior;
        the vague component is the safety valve.
      * "Does this experiment look like the historical ones?" -> use the
        INFORMATIVE part only; a vague component absorbs any observation, so
        including it makes the conflict test blind by construction.
    """
    if prior.vague_index is None:
        return prior
    keep = np.ones(len(prior.weights), dtype=bool)
    keep[prior.vague_index] = False
    return NormalMixture(prior.weights[keep], prior.means[keep], prior.sds[keep],
                         scale=prior.scale, notes=list(prior.notes))


# ═══════════════════════════════════════════════════════════════════════════
# 3. HOW MUCH IS BEING BORROWED?
# ═══════════════════════════════════════════════════════════════════════════
def prior_effective_sample_size(
    prior: NormalMixture,
    reference_proportion: Optional[float] = None,
) -> dict:
    """
    Effective sample size of the prior, in flies.

    Moment-matching approach: a single fly observed at control rate p carries
    Fisher information p(1-p) on the logit scale, so its variance contribution is
    1/(p(1-p)). Matching the prior's variance to that of n_eff independent flies,

        ESS = [1 / (p (1 - p))] / Var(prior)

    This is the quantity to report to a reviewer or an ethics committee: "this
    prior is worth about 45 control flies" is a statement anyone can evaluate,
    whereas "we used a MAP prior" is not.

    The moment-matching ESS is known to be somewhat optimistic for heavy-tailed
    robust mixtures (Neuenschwander et al. 2020 introduce the ELIR alternative);
    the number here should be read as an upper bound on what is being borrowed,
    and it is labelled as such in the output.
    """
    p = float(expit(prior.mean)) if reference_proportion is None else float(reference_proportion)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    unit_var = 1.0 / (p * (1.0 - p))

    ess_full = unit_var / prior.var if prior.var > 0 else float("inf")

    info = informative_component(prior)
    ess_info = unit_var / info.var if info.var > 0 else float("inf")
    robust = prior.vague_index is not None

    return {
        # Headline: what is actually borrowed when history and the current
        # control agree, which is the case borrowing is designed for.
        "ess_flies": float(ess_info),
        "ess_flies_map_component": float(ess_info),
        "ess_flies_full_mixture": float(ess_full),
        "is_robustified": bool(robust),
        "reference_proportion": p,
        "prior_sd_logit": info.sd,
        "method": "moment matching on the logit scale",
        "caveat": (
            "Two numbers are reported because a robust mixture has no single ESS. "
            "`ess_flies_map_component` is what is borrowed when the concurrent control "
            "agrees with history -- the posterior reweights onto the informative "
            "component, so this is the operative figure. "
            "`ess_flies_full_mixture` moment-matches the whole mixture, whose variance is "
            "dominated by the deliberately vague tail, and so understates the borrowing "
            "under agreement while correctly indicating that under CONFLICT almost "
            "nothing is borrowed. Neither is a substitute for the simulated operating "
            "characteristics, which are what actually determine whether borrowing is safe."
            if robust else
            "Moment-matching ESS is an upper bound; it assumes the prior behaves like a "
            "normal with the same variance."
        ),
    }


def prior_data_conflict(
    prior: NormalMixture,
    successes: float,
    totals: float,
) -> dict:
    """
    Test whether the concurrent control conflicts with the historical prior.

    Computes the prior-predictive tail probability of the observed concurrent
    control (Box 1980): how surprising is this control arm, if the historical
    arms really were exchangeable with it? A small p means the current
    experiment is not like the past ones and borrowing would inject bias.

    Returns a hard `conflict` flag at p < 0.05 and a plain-language verdict. The
    point of this function is to be BLOCKING: a borrowed analysis that has not
    been checked for conflict is not a valid analysis.

    If a ROBUSTIFIED prior is passed, its weakly-informative component is
    stripped before the test. That component is designed to accommodate any
    observation, so leaving it in would make the diagnostic unable to detect a
    conflict -- exactly the failure it exists to prevent.
    """
    prior = informative_component(prior)
    y, v = _arm_logit_summaries([successes], [totals])
    y0, v0 = float(y[0]), float(v[0])

    # Prior predictive for a new arm = prior convolved with the sampling variance
    pred = NormalMixture(prior.weights, prior.means, np.sqrt(prior.sds ** 2 + v0),
                         scale=prior.scale)
    tail = float(pred.cdf(y0)[0])
    p_two_sided = 2.0 * min(tail, 1.0 - tail)

    conflict = p_two_sided < 0.05
    obs_p = float((successes + 0.5) / (totals + 1.0))
    prior_p = float(expit(prior.mean))
    lo, hi = pred.interval(0.95)

    if conflict:
        verdict = (
            f"PRIOR-DATA CONFLICT (p = {p_two_sided:.3g}). The concurrent control is "
            f"{obs_p:.1%} climbing, whereas the historical arms predict "
            f"{prior_p:.1%} (95% predictive interval {expit(lo):.1%}-{expit(hi):.1%}). "
            "The current experiment does not look like the historical ones. Do NOT borrow: "
            "analyse with the concurrent control only, and investigate what changed "
            "(stock, food batch, incubator, operator, season)."
        )
    else:
        verdict = (
            f"No prior-data conflict (p = {p_two_sided:.3g}). The concurrent control "
            f"({obs_p:.1%}) is consistent with the historical prediction "
            f"({prior_p:.1%}, 95% predictive interval {expit(lo):.1%}-{expit(hi):.1%}). "
            "Borrowing is supportable."
        )

    return {
        "conflict": bool(conflict),
        "p_value": p_two_sided,
        "observed_proportion": obs_p,
        "prior_predicted_proportion": prior_p,
        "predictive_interval": (float(expit(lo)), float(expit(hi))),
        "verdict": verdict,
        "method": "prior-predictive tail probability (Box 1980)",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONJUGATE UPDATE AND DECISION
# ═══════════════════════════════════════════════════════════════════════════
def update_posterior(
    prior: NormalMixture,
    successes: float,
    totals: float,
) -> NormalMixture:
    """
    Closed-form posterior for the control parameter, combining a normal-mixture
    prior with the normal approximation to the concurrent control likelihood.

    For prior sum_k w_k N(m_k, v_k) and likelihood N(y | theta, s^2):

        v'_k = 1 / (1/v_k + 1/s^2)
        m'_k = v'_k * (m_k/v_k + y/s^2)
        w'_k proportional to w_k * N(y | m_k, v_k + s^2)

    The reweighting of w'_k is what makes the robust prior self-limiting: a
    conflicting observation has low density under the informative component, so
    that component loses its weight automatically.
    """
    y, v = _arm_logit_summaries([successes], [totals])
    y0, s2 = float(y[0]), float(v[0])

    vk = prior.sds ** 2
    v_post = 1.0 / (1.0 / vk + 1.0 / s2)
    m_post = v_post * (prior.means / vk + y0 / s2)
    log_w = np.log(np.clip(prior.weights, 1e-300, None)) + stats.norm.logpdf(
        y0, prior.means, np.sqrt(vk + s2)
    )
    log_w -= log_w.max()
    w_post = np.exp(log_w)

    return NormalMixture(w_post, m_post, np.sqrt(v_post), scale=prior.scale,
                         notes=list(prior.notes))


def prob_treatment_better(
    control_posterior: NormalMixture,
    treat_successes: float,
    treat_totals: float,
) -> float:
    """
    P(theta_treatment > theta_control | data), in closed form.

    With a vague prior on the treatment arm its posterior is N(y_t, s_t^2), and
    for a control posterior that is a normal mixture,

        P = sum_k w_k * Phi( (y_t - m_k) / sqrt(s_t^2 + v_k) )

    Exact, deterministic, and fast enough to simulate operating characteristics
    with tens of thousands of replicates.
    """
    yt, vt = _arm_logit_summaries([treat_successes], [treat_totals])
    yt, st2 = float(yt[0]), float(vt[0])
    z = (yt - control_posterior.means) / np.sqrt(st2 + control_posterior.sds ** 2)
    return float(np.sum(control_posterior.weights * stats.norm.cdf(z)))


# ═══════════════════════════════════════════════════════════════════════════
# 5. OPERATING CHARACTERISTICS -- the part that decides whether to trust it
# ═══════════════════════════════════════════════════════════════════════════
# The three helpers below are vectorised over simulation replicates. They are
# algebraically identical to `update_posterior`, `prob_treatment_better` and
# `prior_data_conflict`, which remain the readable single-case reference
# implementations; `validate_control_prior.py` asserts the two agree to 1e-12.
# The vectorised forms exist because credible operating characteristics need
# tens of thousands of replicates, which a per-replicate Python loop cannot
# deliver.

def _vectorised_posterior(prior: NormalMixture, y: np.ndarray, v: np.ndarray):
    """Conjugate update of a K-component mixture for each of N observations.

    Returns (weights, means, variances), each of shape (N, K).
    """
    vk = prior.sds ** 2                      # (K,)
    s2 = v[:, None]                          # (N, 1)
    v_post = 1.0 / (1.0 / vk[None, :] + 1.0 / s2)
    m_post = v_post * (prior.means[None, :] / vk[None, :] + y[:, None] / s2)

    log_w = (np.log(np.clip(prior.weights, 1e-300, None))[None, :]
             + stats.norm.logpdf(y[:, None], prior.means[None, :],
                                 np.sqrt(vk[None, :] + s2)))
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)
    return w, m_post, v_post


def _vectorised_prob_better(w: np.ndarray, m: np.ndarray, v: np.ndarray,
                            y_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
    """P(theta_treat > theta_control) for each replicate: sum_k w_k Phi(.)."""
    z = (y_t[:, None] - m) / np.sqrt(v_t[:, None] + v)
    return np.sum(w * stats.norm.cdf(z), axis=1)


def _vectorised_conflict_p(prior: NormalMixture, y: np.ndarray,
                           v: np.ndarray) -> np.ndarray:
    """Two-sided prior-predictive tail probability for each replicate."""
    sd = np.sqrt(prior.sds[None, :] ** 2 + v[:, None])
    tail = np.sum(prior.weights[None, :]
                  * stats.norm.cdf(y[:, None], prior.means[None, :], sd), axis=1)
    return 2.0 * np.minimum(tail, 1.0 - tail)


def operating_characteristics(
    historical_successes: Sequence[float],
    historical_totals: Sequence[float],
    n_control: int,
    n_treat: int,
    p_control_true_grid: Sequence[float],
    true_effect_logit: float = 0.0,
    robust_weight: float = 0.8,
    decision_threshold: float = 0.975,
    n_sim: int = 4000,
    tau_prior_scale: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate the whole borrowing procedure across a grid of DRIFT between the
    historical controls and the true current control rate, and report the
    resulting type I error (when `true_effect_logit` is 0) or power (when it is
    not).

    This is the deliverable that determines whether borrowing is acceptable for
    a given laboratory. Borrowing is not justified by the elegance of the prior;
    it is justified by the false-positive rate it produces at the amount of
    drift that laboratory actually experiences.

    Three analyses are compared at every drift level:
      concurrent_only -- no borrowing (the honest baseline)
      robust_map      -- robust MAP prior, weight `robust_weight`
      full_map        -- MAP prior with no robust component (weight 1.0), included
                         to show what the robustification is buying

    Returns one row per (p_control_true, method).
    """
    rng = np.random.default_rng(seed)

    built = build_map_prior(historical_successes, historical_totals,
                            tau_prior_scale=tau_prior_scale)
    # Compress the fine tau-quadrature mixture (hundreds of components) to a
    # compact one before simulating. This is the standard reportable form
    # (Schmidli et al. 2014) and makes the simulation tractable; `compress`
    # records a note if the approximation shifts the summary materially.
    map_prior = built["prior"].compress(n_components=3, seed=seed)
    rob_prior = robustify(map_prior, weight=robust_weight)
    map_info = informative_component(rob_prior)

    methods = {
        "concurrent_only": None,
        "robust_map": rob_prior,
        "full_map": map_prior,
    }

    rows = []
    for p_c in p_control_true_grid:
        p_c = float(p_c)
        p_t = float(expit(logit(p_c) + true_effect_logit))

        yc = rng.binomial(n_control, p_c, size=n_sim)
        yt = rng.binomial(n_treat, p_t, size=n_sim)

        # arm-level normal approximations, vectorised over simulations
        y_c, v_c = _arm_logit_summaries(yc, np.full(n_sim, n_control))
        y_t, v_t = _arm_logit_summaries(yt, np.full(n_sim, n_treat))

        for name, prior in methods.items():
            if prior is None:
                w_post = np.ones((n_sim, 1))
                m_post = y_c[:, None]
                v_post = v_c[:, None]
            else:
                w_post, m_post, v_post = _vectorised_posterior(prior, y_c, v_c)

            p_better = _vectorised_prob_better(w_post, m_post, v_post, y_t, v_t)
            declares = int(np.sum(p_better > decision_threshold))

            if name == "robust_map":
                conflict_rate = float(np.mean(
                    _vectorised_conflict_p(map_info, y_c, v_c) < 0.05
                ))
            else:
                conflict_rate = np.nan

            rows.append({
                "p_control_true": p_c,
                "drift_from_history": p_c - float(expit(map_prior.mean)),
                "method": name,
                "rejection_rate": declares / n_sim,
                "conflict_flag_rate": conflict_rate,
                "n_control": int(n_control),
                "n_treat": int(n_treat),
                "true_effect_logit": float(true_effect_logit),
                "n_sim": int(n_sim),
            })

    out = pd.DataFrame(rows)
    out.attrs["prior_mean_proportion"] = float(expit(map_prior.mean))
    out.attrs["prior_ess"] = prior_effective_sample_size(rob_prior)["ess_flies"]
    out.attrs["interpretation"] = (
        "With true_effect_logit = 0 the rejection rate IS the type I error. Compare each "
        "borrowing method against concurrent_only at every drift level: the excess is the "
        "price of borrowing, and it must be judged against the fly savings before the "
        "method is adopted."
    )
    return out


def recommended_design(
    historical_successes: Sequence[float],
    historical_totals: Sequence[float],
    n_control_standard: int,
    n_treat: int,
    max_type_I: float = 0.075,
    tolerable_drift: float = 0.05,
    robust_weight: float = 0.8,
    decision_threshold: float = 0.975,
    n_sim: int = 3000,
    seed: int = 0,
) -> dict:
    """
    Find the smallest concurrent control arm whose borrowed analysis keeps type
    I error below `max_type_I` across drift up to +/- `tolerable_drift`, and
    report the fly saving.

    A concurrent control is never eliminated: without one, drift is undetectable
    and the prior-data conflict check -- the safety mechanism of the whole
    procedure -- cannot run. The search therefore stops at a floor of 25% of the
    standard arm or 10 flies, whichever is larger.
    """
    built = build_map_prior(historical_successes, historical_totals)
    p_hist = float(expit(built["prior"].mean))
    grid = [p_hist - tolerable_drift, p_hist, p_hist + tolerable_drift]
    grid = [float(np.clip(g, 0.02, 0.98)) for g in grid]

    floor = max(10, int(round(0.25 * n_control_standard)))
    candidates = sorted({int(round(n_control_standard * f)) for f in
                         (1.0, 0.75, 0.5, 0.4, 0.3, 0.25)} | {floor})
    candidates = [c for c in candidates if c >= floor]

    results = []
    chosen = None
    for n_c in sorted(candidates):
        oc = operating_characteristics(
            historical_successes, historical_totals,
            n_control=n_c, n_treat=n_treat,
            p_control_true_grid=grid, true_effect_logit=0.0,
            robust_weight=robust_weight, decision_threshold=decision_threshold,
            n_sim=n_sim, seed=seed,
        )
        worst = float(oc[oc["method"] == "robust_map"]["rejection_rate"].max())
        # The simulated rate is an estimate, and the search picks the SMALLEST arm
        # whose estimate clears the cap -- selection on a noisy statistic. Judge
        # the upper 95% confidence limit instead, so Monte-Carlo error cannot buy
        # a smaller control arm than the evidence supports. The cost is a few
        # extra flies; the alternative is recommending a reduction that only
        # looked safe.
        se = float(np.sqrt(max(worst * (1.0 - worst), 1e-12) / max(n_sim, 1)))
        worst_ucl = worst + 1.959963985 * se
        results.append({"n_control": n_c, "worst_type_I": worst,
                        "worst_type_I_ucl": worst_ucl,
                        "acceptable": bool(worst_ucl <= max_type_I)})
        if worst_ucl <= max_type_I and chosen is None:
            chosen = n_c

    tab = pd.DataFrame(results).sort_values("n_control")
    saving = (n_control_standard - chosen) if chosen else 0

    if chosen is None:
        verdict = (
            f"No reduced control arm kept type I error at or below {max_type_I:.1%} across "
            f"drift of +/-{tolerable_drift:.0%}. Borrowing is NOT recommended for this "
            "historical set -- the historical arms are too heterogeneous, or the tolerated "
            "drift is too large, for the borrowed information to be safe. Run a full "
            "concurrent control."
        )
    else:
        ess = prior_effective_sample_size(robustify(built["prior"], robust_weight))
        verdict = (
            f"A concurrent control of {chosen} flies (instead of {n_control_standard}) keeps "
            f"type I error at or below {max_type_I:.1%} across drift of "
            f"+/-{tolerable_drift:.0%}, saving {saving} flies per experiment "
            f"({saving / max(n_control_standard, 1):.0%} of the control arm). The prior "
            f"contributes at most {ess['ess_flies']:.0f} flies of information. A concurrent "
            "control is retained deliberately: without it, drift cannot be detected and the "
            "prior-data conflict check cannot run."
        )

    return {
        "recommended_n_control": chosen,
        "standard_n_control": int(n_control_standard),
        "flies_saved_per_experiment": int(saving),
        "search_table": tab,
        "historical_mean_proportion": p_hist,
        "drift_grid_tested": grid,
        "max_type_I_allowed": float(max_type_I),
        "verdict": verdict,
        "prior_notes": built["notes"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONVENIENCE: BUILD FROM A DROPHENIX DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════
def historical_arms_from_df(
    df: pd.DataFrame,
    control_treatment: str = "Vehicle",
    experiment_col: str = "ExperimentID",
    treatment_col: str = "Treatment",
    genotype: Optional[str] = None,
    genotype_col: str = "Genotype",
    sex: Optional[str] = None,
    count_col: str = "Count",
    total_col: str = "n",
    endpoint_time_only: bool = True,
    time_col: str = "Time",
) -> pd.DataFrame:
    """
    Collapse a multi-experiment DroPhenix dataframe into one control arm per
    experiment, ready for `build_map_prior`.

    Filters to the control treatment (and optionally one genotype and sex),
    takes the endpoint timepoint of each vial by default, and sums counts within
    each experiment.
    """
    for c in (experiment_col, treatment_col, count_col, total_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found.")

    work = df[df[treatment_col].astype(str) == str(control_treatment)].copy()
    if genotype is not None and genotype_col in work.columns:
        work = work[work[genotype_col].astype(str) == str(genotype)]
    if sex is not None and "Sex" in work.columns:
        work = work[work["Sex"].astype(str) == str(sex)]
    if work.empty:
        raise ValueError("No rows match the requested control arm.")

    if endpoint_time_only and time_col in work.columns:
        keys = [c for c in (experiment_col, "VialID", "Replicate", genotype_col, "Sex")
                if c in work.columns]
        work = (work.sort_values(time_col)
                    .groupby(keys, dropna=False, as_index=False)
                    .tail(1))

    arms = (work.groupby(experiment_col, dropna=False)
                .agg(successes=(count_col, "sum"), totals=(total_col, "sum"))
                .reset_index())
    arms["proportion"] = arms["successes"] / arms["totals"]
    return arms
