"""
Hierarchical (clustered) analysis of Drosophila climbing counts.

THE PROBLEM THIS MODULE FIXES
-----------------------------
Climbing data are counts: `Count` flies out of `n` crossed the line. Two facts
follow, and the conventional analysis ignores both.

1. `Count/n` is a BINOMIAL PROPORTION. Its variance, n*p*(1-p), depends on its
   mean -- maximal at p = 0.5, zero at the boundaries. Tests that assume
   mean-independent variance (t-test, ANOVA) are therefore mis-specified on the
   Performance Index PI = (2*Count/n - 1)*100, which is only a linear rescaling
   of that proportion. Worse, a variance-homogeneity pre-test will flag the
   structural heteroscedasticity as a violation and silently reroute the
   analysis to a rank test.

2. FLIES IN A VIAL ARE NOT INDEPENDENT. They share food, the tap that startles
   them, temperature, humidity, and handling history. The experimental unit is
   the vial. Treating m flies per vial as m independent observations inflates
   the effective sample size by the design effect

       DEFF = 1 + (m - 1) * ICC

   With m = 20 flies and a modest ICC of 0.05, DEFF ~ 1.95: standard errors are
   too small by ~40% and the nominal 5% test rejects far more often than 5% of
   the time under the null.

WHAT THIS MODULE PROVIDES
-------------------------
  estimate_icc            -- ANOVA estimator of the intraclass correlation for
                             clustered binary data (Ridout, Demetrio & Firth
                             1999, Biometrics 55:137-148)
  design_effect           -- DEFF and effective sample size
  fit_betabinomial        -- maximum-likelihood beta-binomial regression with a
                             logit link. This is the correct marginal model for
                             overdispersed clustered counts: it estimates the
                             intra-vial correlation as a parameter rather than
                             assuming it away.
  fit_gee                 -- population-averaged GEE with cluster-robust
                             (sandwich) standard errors (Liang & Zeger 1986,
                             Biometrika 73:13-22), via statsmodels. An
                             independent cross-check on the beta-binomial fit
                             that makes no distributional assumption beyond the
                             mean model.
  fit_mixed_logit         -- linear mixed model on the empirical logit, which
                             supports random SLOPES and is therefore the engine
                             for longitudinal age trajectories.
  pseudoreplication_report-- the headline diagnostic: runs the naive
                             fly-as-independent analysis and the
                             cluster-correct analysis on the same data and
                             reports how much the naive p-value was inflated.

CLUSTERING DOES NOT ALWAYS WIDEN THE INTERVAL
---------------------------------------------
Which direction the correction goes depends on where the predictor varies, and
getting this backwards is a common error:

  BETWEEN-cluster predictor (a treatment applied to whole vials)
      The naive standard error is TOO SMALL, because it counts flies as
      independent replicates of the treatment when only the vials are. The
      cluster-correct interval is wider. This is the pseudoreplication case.

  WITHIN-cluster predictor (age, when every vial is measured at every age)
      The vial random intercept absorbs between-vial variation that would
      otherwise sit in the residual, so the cluster-correct interval is
      NARROWER. Respecting the design buys precision here.

Both directions are asserted in `validation/validate_hierarchical.py` (test F).

WHAT IT DOES NOT DO
-------------------
It does not rescue a design in which treatment is confounded with vial (one
vial per treatment). In that case the treatment effect and the vial effect are
the same parameter and no model can separate them; `pseudoreplication_report`
detects this and refuses rather than returning a number.

References
----------
Ridout MS, Demetrio CGB, Firth D (1999) Estimating intraclass correlation for
  binary data. Biometrics 55:137-148.
Liang KY, Zeger SL (1986) Longitudinal data analysis using generalized linear
  models. Biometrika 73:13-22.
Williams DA (1975) The analysis of binary responses from toxicological
  experiments involving reproduction and teratogenicity. Biometrics 31:949-952.
  [beta-binomial for clustered binary data]
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import betaln, expit, logit


# ═══════════════════════════════════════════════════════════════════════════
# 1. INTRACLASS CORRELATION AND DESIGN EFFECT
# ═══════════════════════════════════════════════════════════════════════════
def estimate_icc(counts: Sequence[float], totals: Sequence[float]) -> dict:
    """
    ANOVA estimator of the intraclass correlation for clustered binary data.

    Each element is one cluster (vial): `counts[i]` successes out of
    `totals[i]` trials. Follows Ridout, Demetrio & Firth (1999) Biometrics
    55:137-148, estimator rho_A:

        m0   = (N - sum(n_i^2)/N) / (k - 1)
        MSB  = sum n_i (p_i - pbar)^2 / (k - 1)
        MSW  = sum n_i p_i (1 - p_i) / (N - k)
        rho  = (MSB - MSW) / (MSB + (m0 - 1) * MSW)

    where pbar is the pooled proportion, k the number of clusters and
    N = sum(n_i).

    The estimator can be negative (clusters more homogeneous than binomial);
    the raw value is returned as `icc_raw` and the truncated-at-zero value as
    `icc`, because a negative ICC is not usable in a design effect.
    """
    y = np.asarray(counts, dtype=float)
    n = np.asarray(totals, dtype=float)
    ok = np.isfinite(y) & np.isfinite(n) & (n > 0)
    y, n = y[ok], n[ok]

    k = len(y)
    if k < 2:
        return {"icc": np.nan, "icc_raw": np.nan, "n_clusters": k,
                "reason": "Need at least 2 clusters to estimate an ICC."}

    N = float(n.sum())
    if N <= k:
        return {"icc": np.nan, "icc_raw": np.nan, "n_clusters": k,
                "reason": "Total trials must exceed the number of clusters."}

    p_i = y / n
    p_bar = float(y.sum() / N)

    m0 = (N - float((n ** 2).sum()) / N) / (k - 1)
    msb = float((n * (p_i - p_bar) ** 2).sum()) / (k - 1)
    msw = float((n * p_i * (1.0 - p_i)).sum()) / (N - k)

    denom = msb + (m0 - 1.0) * msw
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        icc_raw = np.nan
    else:
        icc_raw = (msb - msw) / denom

    icc = float(np.clip(icc_raw, 0.0, 1.0)) if np.isfinite(icc_raw) else np.nan

    return {
        "icc": icc,
        "icc_raw": float(icc_raw) if np.isfinite(icc_raw) else np.nan,
        "n_clusters": int(k),
        "mean_cluster_size": float(n.mean()),
        "m0": float(m0),
        "MSB": msb,
        "MSW": msw,
        "pooled_p": p_bar,
        "method": "ANOVA estimator (Ridout, Demetrio & Firth 1999)",
    }


def design_effect(mean_cluster_size: float, icc: float) -> dict:
    """
    Design effect DEFF = 1 + (m - 1) * ICC and the resulting inflation of the
    standard error (sqrt(DEFF)).

    A DEFF of 2 means the clustered sample carries the information of half as
    many independent observations, and a naive standard error is too small by
    a factor of sqrt(2) ~ 1.41.
    """
    m = float(mean_cluster_size)
    r = float(icc)
    if not np.isfinite(m) or not np.isfinite(r) or m <= 0:
        return {"deff": np.nan, "se_inflation": np.nan}
    deff = 1.0 + (m - 1.0) * r
    return {
        "deff": float(deff),
        "se_inflation": float(np.sqrt(deff)),
        "mean_cluster_size": m,
        "icc": r,
    }


def effective_sample_size(total_flies: float, mean_cluster_size: float, icc: float) -> float:
    """Effective (independent-equivalent) number of flies: N / DEFF."""
    d = design_effect(mean_cluster_size, icc)["deff"]
    if not np.isfinite(d) or d <= 0:
        return float("nan")
    return float(total_flies) / d


# ═══════════════════════════════════════════════════════════════════════════
# 2. BETA-BINOMIAL REGRESSION (the correct marginal model)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BetaBinomialFit:
    """Result of a maximum-likelihood beta-binomial regression."""

    params: pd.Series                    # coefficients on the logit scale
    bse: pd.Series                       # standard errors
    zvalues: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    rho: float                           # intra-cluster correlation (dispersion)
    rho_se: float
    loglike: float
    aic: float
    bic: float
    n_obs: int
    design_info: List[str]
    converged: bool
    message: str = ""
    _design_columns: List[str] = field(default_factory=list)

    def summary_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({
            "coef": self.params,
            "std_err": self.bse,
            "z": self.zvalues,
            "p_value": self.pvalues,
            "ci_lower": self.conf_int["lower"],
            "ci_upper": self.conf_int["upper"],
        })
        out["odds_ratio"] = np.exp(out["coef"])
        out["OR_lower"] = np.exp(out["ci_lower"])
        out["OR_upper"] = np.exp(out["ci_upper"])
        return out


def _betabinom_negloglike(theta: np.ndarray, X: np.ndarray, y: np.ndarray,
                          n: np.ndarray) -> float:
    """
    Negative log-likelihood of the beta-binomial with logit-linked mean.

    Parameterisation: mu_i = expit(X_i . beta); rho in (0, 1) is the
    intra-cluster correlation, transformed as rho = expit(gamma) so the
    optimiser is unconstrained.

        alpha_i = mu_i * (1 - rho) / rho
        beta_i  = (1 - mu_i) * (1 - rho) / rho

    Var(y_i) = n_i mu_i (1 - mu_i) [1 + (n_i - 1) rho]  -- i.e. rho IS the
    intraclass correlation, which is why it is directly interpretable.
    """
    beta = theta[:-1]
    gamma = theta[-1]
    rho = expit(gamma)
    rho = min(max(rho, 1e-9), 1.0 - 1e-9)

    eta = X @ beta
    # guard against overflow in expit for extreme linear predictors
    eta = np.clip(eta, -30.0, 30.0)
    mu = expit(eta)
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

    s = (1.0 - rho) / rho
    a = mu * s
    b = (1.0 - mu) * s

    # log C(n, y) is constant in the parameters but kept for a comparable AIC
    ll = (
        betaln(y + a, n - y + b)
        - betaln(a, b)
    )
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(np.sum(ll))


def fit_betabinomial(
    df: pd.DataFrame,
    formula_rhs: str = "Genotype + Treatment",
    count_col: str = "Count",
    total_col: str = "n",
    alpha: float = 0.05,
    maxiter: int = 2000,
) -> BetaBinomialFit:
    """
    Maximum-likelihood beta-binomial regression of `count_col`/`total_col` on
    the right-hand side `formula_rhs` (patsy syntax, e.g.
    "Genotype * Treatment + Age").

    The beta-binomial is the natural model for counts of flies clustered in
    vials: it has the binomial mean structure but an extra parameter `rho` that
    is exactly the intraclass correlation, so overdispersion is estimated
    rather than assumed to be absent.

    Returns a `BetaBinomialFit` with coefficients on the logit scale, odds
    ratios, Wald intervals, and the estimated rho with its standard error.
    """
    import patsy

    needed = {count_col, total_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.dropna(subset=[count_col, total_col]).copy()
    work = work[work[total_col] > 0]
    if work.empty:
        raise ValueError("No usable rows (all missing, or n <= 0).")

    y_design = patsy.dmatrix(f"{formula_rhs}", work, return_type="dataframe")
    X = np.asarray(y_design, dtype=float)
    colnames = list(y_design.columns)

    y = work[count_col].to_numpy(dtype=float)
    n = work[total_col].to_numpy(dtype=float)
    y = np.minimum(y, n)

    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"Model has {X.shape[1]} parameters but only {X.shape[0]} rows; "
            "not identifiable. Simplify the formula or supply more replicates."
        )

    # Start from the binomial GLM solution (rho -> 0) for a stable optimisation
    p_start = np.clip(y.sum() / n.sum(), 1e-3, 1 - 1e-3)
    beta0 = np.zeros(X.shape[1])
    if colnames and colnames[0] == "Intercept":
        beta0[0] = logit(p_start)
    theta0 = np.concatenate([beta0, [logit(0.05)]])   # rho starts at 0.05

    res = optimize.minimize(
        _betabinom_negloglike,
        theta0,
        args=(X, y, n),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
    )

    theta = res.x
    loglike = -float(res.fun)

    # Standard errors from the numerical Hessian of the negative log-likelihood
    try:
        H = _numerical_hessian(
            lambda t: _betabinom_negloglike(t, X, y, n), theta
        )
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except Exception:
        se = np.full(theta.shape, np.nan)

    k = X.shape[1]
    beta_hat = theta[:k]
    beta_se = se[:k]
    gamma_hat, gamma_se = theta[-1], se[-1]
    rho_hat = float(expit(gamma_hat))
    # delta method: d rho/d gamma = rho (1 - rho)
    rho_se = float(rho_hat * (1.0 - rho_hat) * gamma_se) if np.isfinite(gamma_se) else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta_hat / beta_se
        p = 2.0 * stats.norm.sf(np.abs(z))
    crit = stats.norm.ppf(1.0 - alpha / 2.0)

    params = pd.Series(beta_hat, index=colnames)
    bse = pd.Series(beta_se, index=colnames)
    ci = pd.DataFrame(
        {"lower": beta_hat - crit * beta_se, "upper": beta_hat + crit * beta_se},
        index=colnames,
    )

    n_par = k + 1
    n_obs = int(X.shape[0])
    aic = -2.0 * loglike + 2.0 * n_par
    bic = -2.0 * loglike + np.log(n_obs) * n_par

    return BetaBinomialFit(
        params=params,
        bse=bse,
        zvalues=pd.Series(z, index=colnames),
        pvalues=pd.Series(p, index=colnames),
        conf_int=ci,
        rho=rho_hat,
        rho_se=rho_se,
        loglike=loglike,
        aic=float(aic),
        bic=float(bic),
        n_obs=n_obs,
        design_info=colnames,
        converged=bool(res.success),
        message=str(res.message),
        _design_columns=colnames,
    )


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference Hessian. Step scaled to each parameter's magnitude."""
    x = np.asarray(x, dtype=float)
    p = len(x)
    h = eps * np.maximum(np.abs(x), 1.0)
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


# ═══════════════════════════════════════════════════════════════════════════
# 3. GEE WITH CLUSTER-ROBUST STANDARD ERRORS
# ═══════════════════════════════════════════════════════════════════════════
def expand_to_bernoulli(
    df: pd.DataFrame,
    count_col: str = "Count",
    total_col: str = "n",
    max_rows: int = 500_000,
) -> pd.DataFrame:
    """
    Expand aggregated counts into one row per fly with a 0/1 outcome.

    Every other column is carried through unchanged. This makes the fly-level
    model explicit and is required by GEE, which clusters individual records.

    Raises if the expansion would exceed `max_rows`, rather than silently
    producing a dataframe too large to fit.
    """
    work = df.dropna(subset=[count_col, total_col]).copy()
    work = work[work[total_col] > 0]
    total = int(work[total_col].sum())
    if total > max_rows:
        raise ValueError(
            f"Expansion would create {total:,} rows (limit {max_rows:,}). "
            "Use the beta-binomial fit on aggregated counts instead."
        )

    reps = work[total_col].to_numpy(dtype=int)
    expanded = work.loc[work.index.repeat(reps)].copy()

    successes = work[count_col].to_numpy(dtype=int)
    outcome = np.concatenate([
        np.concatenate([np.ones(int(s), dtype=int), np.zeros(int(t) - int(s), dtype=int)])
        for s, t in zip(successes, reps)
    ])
    expanded["Climbed"] = outcome
    expanded = expanded.reset_index(drop=True)
    return expanded


def fit_gee(
    df: pd.DataFrame,
    formula_rhs: str = "Genotype + Treatment",
    cluster_col: str = "VialID",
    count_col: str = "Count",
    total_col: str = "n",
    cov_struct: str = "exchangeable",
    alpha: float = 0.05,
) -> dict:
    """
    Population-averaged logistic GEE with cluster-robust (sandwich) standard
    errors, clustering on `cluster_col` (the vial).

    GEE makes no distributional assumption beyond the mean model and yields
    valid standard errors even if the working correlation is wrong, which makes
    it the right independent cross-check on the beta-binomial fit.

    Returns a dict with a coefficient table, the estimated working correlation,
    and the fitted statsmodels result object under "model".
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if cluster_col not in df.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found. GEE requires the "
            "identifier of the experimental unit (vial)."
        )

    long = expand_to_bernoulli(df, count_col=count_col, total_col=total_col)
    long = long.dropna(subset=[cluster_col])
    if long.empty:
        raise ValueError("No rows with a non-missing cluster identifier.")

    cs = {
        "exchangeable": sm.cov_struct.Exchangeable(),
        "independence": sm.cov_struct.Independence(),
    }.get(cov_struct.lower())
    if cs is None:
        raise ValueError("cov_struct must be 'exchangeable' or 'independence'.")

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        model = smf.gee(
            f"Climbed ~ {formula_rhs}",
            groups=cluster_col,
            data=long,
            family=sm.families.Binomial(),
            cov_struct=cs,
        )
        res = model.fit()

    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    tab = pd.DataFrame({
        "coef": res.params,
        "std_err": res.bse,
        "z": res.tvalues,
        "p_value": res.pvalues,
        "ci_lower": res.params - crit * res.bse,
        "ci_upper": res.params + crit * res.bse,
    })
    tab["odds_ratio"] = np.exp(tab["coef"])
    tab["OR_lower"] = np.exp(tab["ci_lower"])
    tab["OR_upper"] = np.exp(tab["ci_upper"])

    try:
        working_corr = float(np.asarray(cs.dep_params).ravel()[0])
    except Exception:
        working_corr = float("nan")

    return {
        "table": tab,
        "working_correlation": working_corr,
        "n_clusters": int(long[cluster_col].nunique()),
        "n_obs": int(len(long)),
        "cov_struct": cov_struct,
        "model": res,
        "method": "GEE, logit link, cluster-robust sandwich SE (Liang & Zeger 1986)",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. MIXED MODEL ON THE EMPIRICAL LOGIT (supports random slopes)
# ═══════════════════════════════════════════════════════════════════════════
def empirical_logit(counts, totals, adjust: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical logit and its approximate sampling variance.

        z_i   = log((y_i + c) / (n_i - y_i + c))
        var_i = 1/(y_i + c) + 1/(n_i - y_i + c)

    The continuity correction `c` (default 0.5) keeps the transform finite at
    y = 0 and y = n -- the ceiling and floor that climbing data routinely hit.
    Working on this scale removes the mean-variance coupling of the raw
    proportion and unbounds the ceiling, which is what makes a linear mixed
    model defensible here.
    """
    y = np.asarray(counts, dtype=float)
    n = np.asarray(totals, dtype=float)
    c = float(adjust)
    z = np.log((y + c) / (n - y + c))
    v = 1.0 / (y + c) + 1.0 / (n - y + c)
    return z, v


def fit_mixed_logit(
    df: pd.DataFrame,
    fixed_rhs: str = "Age",
    group_col: str = "VialID",
    random_slope: Optional[str] = None,
    count_col: str = "Count",
    total_col: str = "n",
    alpha: float = 0.05,
) -> dict:
    """
    Linear mixed model on the empirical logit, with a random intercept per
    `group_col` and optionally a random slope on `random_slope`.

    This is the engine for LONGITUDINAL trajectories: a random slope on Age
    lets each vial (or fly) decline at its own rate, which is the whole point
    of a trajectory model, and is not expressible in GEE or in the marginal
    beta-binomial.

    Returns a dict with the fixed-effect table, the random-effect variances,
    the ICC implied by those variances, and the fitted statsmodels result.
    """
    import statsmodels.formula.api as smf

    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found.")

    work = df.dropna(subset=[count_col, total_col, group_col]).copy()
    work = work[work[total_col] > 0]
    if work.empty:
        raise ValueError("No usable rows.")

    z, v = empirical_logit(work[count_col], work[total_col])
    work["_elogit"] = z
    work["_elogit_var"] = v

    n_groups = work[group_col].nunique()
    if n_groups < 3:
        raise ValueError(
            f"Only {n_groups} level(s) of '{group_col}'. A random effect cannot be "
            "estimated from fewer than 3 groups; use a fixed effect instead."
        )

    re_formula = "1"
    if random_slope:
        if random_slope not in work.columns:
            raise ValueError(f"random_slope column '{random_slope}' not found.")
        re_formula = f"1 + {random_slope}"

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        model = smf.mixedlm(
            f"_elogit ~ {fixed_rhs}",
            data=work,
            groups=work[group_col],
            re_formula=re_formula,
        )
        try:
            res = model.fit(reml=True, method="lbfgs")
        except Exception:
            res = model.fit(reml=True)

    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    fe = pd.DataFrame({
        "coef": res.fe_params,
        "std_err": res.bse[: len(res.fe_params)],
        "p_value": res.pvalues[: len(res.fe_params)],
    })
    fe["ci_lower"] = fe["coef"] - crit * fe["std_err"]
    fe["ci_upper"] = fe["coef"] + crit * fe["std_err"]

    # ICC implied by the random-intercept variance vs residual variance
    try:
        var_re = float(np.asarray(res.cov_re)[0, 0])
        var_resid = float(res.scale)
        icc_model = var_re / (var_re + var_resid) if (var_re + var_resid) > 0 else np.nan
    except Exception:
        var_re, var_resid, icc_model = np.nan, np.nan, np.nan

    return {
        "fixed_effects": fe,
        "random_intercept_var": var_re,
        "residual_var": var_resid,
        "icc_model": icc_model,
        "n_groups": int(n_groups),
        "n_obs": int(len(work)),
        "converged": bool(getattr(res, "converged", True)),
        "random_slope": random_slope,
        "model": res,
        "scale_note": (
            "Coefficients are on the empirical-logit scale. Exponentiate for an "
            "approximate odds ratio; back-transform with expit() for a predicted "
            "proportion."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. THE HEADLINE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════
def pseudoreplication_report(
    df: pd.DataFrame,
    group_col: str = "Treatment",
    cluster_col: str = "VialID",
    count_col: str = "Count",
    total_col: str = "n",
) -> dict:
    """
    Quantify the cost of ignoring clustering, on the user's own data.

    Runs two analyses of the same two-group comparison:

      NAIVE      -- every fly treated as an independent Bernoulli observation
                    (a chi-square / logistic regression with no clustering).
                    This is what a t-test or chi-square on pooled flies does.
      CLUSTERED  -- GEE with cluster-robust standard errors on `cluster_col`,
                    plus the beta-binomial estimate of the intra-vial
                    correlation.

    Reports the ICC, the design effect, the effective sample size, and the
    ratio of the two p-values -- i.e. exactly how much the naive analysis
    overstated the evidence.

    Refuses (returns `usable=False`) when treatment is completely confounded
    with cluster, because then the two effects are one parameter.
    """
    out: dict = {"usable": False}

    for col in (group_col, cluster_col, count_col, total_col):
        if col not in df.columns:
            out["reason"] = f"Column '{col}' not found."
            return out

    work = df.dropna(subset=[group_col, cluster_col, count_col, total_col]).copy()
    work = work[work[total_col] > 0]
    if work.empty:
        out["reason"] = "No usable rows."
        return out

    levels = list(pd.unique(work[group_col]))
    if len(levels) != 2:
        out["reason"] = (
            f"'{group_col}' has {len(levels)} levels; this diagnostic compares exactly two. "
            f"Levels found: {levels[:6]}"
        )
        return out

    # Confounding check: does each cluster belong to exactly one group?
    per_cluster_groups = work.groupby(cluster_col)[group_col].nunique()
    clusters_per_group = work.groupby(group_col)[cluster_col].nunique()
    if int(clusters_per_group.min()) < 2:
        out["reason"] = (
            f"At least one level of '{group_col}' has only one {cluster_col}. "
            "Treatment and vial effects are then the same parameter and cannot be "
            "separated -- this is a design problem, not an analysis problem. "
            f"Clusters per group: {clusters_per_group.to_dict()}"
        )
        return out

    a, b = levels
    grp_a = work[work[group_col] == a]
    grp_b = work[work[group_col] == b]

    # ── NAIVE: pooled flies, no clustering (2x2 chi-square on total flies) ──
    ya, na = float(grp_a[count_col].sum()), float(grp_a[total_col].sum())
    yb, nb = float(grp_b[count_col].sum()), float(grp_b[total_col].sum())
    table = np.array([[ya, na - ya], [yb, nb - yb]])
    try:
        chi2, p_naive, _, _ = stats.chi2_contingency(table, correction=False)
    except Exception:
        chi2, p_naive = np.nan, np.nan

    # ── ICC per group and pooled ────────────────────────────────────────────
    icc_a = estimate_icc(
        grp_a.groupby(cluster_col)[count_col].sum(),
        grp_a.groupby(cluster_col)[total_col].sum(),
    )
    icc_b = estimate_icc(
        grp_b.groupby(cluster_col)[count_col].sum(),
        grp_b.groupby(cluster_col)[total_col].sum(),
    )
    iccs = [v["icc"] for v in (icc_a, icc_b) if np.isfinite(v.get("icc", np.nan))]
    icc_pooled = float(np.mean(iccs)) if iccs else np.nan

    mean_m = float(work.groupby(cluster_col)[total_col].sum().mean())
    deff = design_effect(mean_m, icc_pooled)
    n_total = float(work[total_col].sum())
    n_eff = effective_sample_size(n_total, mean_m, icc_pooled)

    # ── CLUSTERED: GEE with robust SEs ──────────────────────────────────────
    p_clustered, gee_note = np.nan, ""
    try:
        gee = fit_gee(work, formula_rhs=f"C({group_col})", cluster_col=cluster_col,
                      count_col=count_col, total_col=total_col)
        tab = gee["table"]
        eff_rows = [i for i in tab.index if i != "Intercept"]
        if eff_rows:
            p_clustered = float(tab.loc[eff_rows[0], "p_value"])
        gee_note = gee["method"]
    except Exception as exc:
        gee_note = f"GEE unavailable: {exc}"

    # ── beta-binomial rho as an independent estimate of the same quantity ───
    rho_bb = np.nan
    try:
        bb = fit_betabinomial(work, formula_rhs=f"C({group_col})",
                              count_col=count_col, total_col=total_col)
        rho_bb = bb.rho
    except Exception:
        pass

    inflation = (
        float(p_clustered / p_naive)
        if np.isfinite(p_clustered) and np.isfinite(p_naive) and p_naive > 0
        else np.nan
    )

    verdict_bits = []
    if np.isfinite(icc_pooled):
        verdict_bits.append(f"Intra-vial correlation ICC = {icc_pooled:.3f}")
    if np.isfinite(deff["deff"]):
        verdict_bits.append(
            f"design effect {deff['deff']:.2f} (standard errors {(deff['se_inflation'] - 1) * 100:.0f}% larger than naive)"
        )
    if np.isfinite(n_eff):
        verdict_bits.append(
            f"{n_total:.0f} flies carry the information of {n_eff:.0f} independent flies"
        )
    if np.isfinite(p_naive) and np.isfinite(p_clustered):
        crossed = (p_naive < 0.05) and (p_clustered >= 0.05)
        verdict_bits.append(
            f"p = {p_naive:.2g} treating flies as independent vs p = {p_clustered:.2g} "
            f"clustering on {cluster_col}"
            + ("  -- THE NAIVE ANALYSIS CROSSES 0.05 AND THE CORRECT ONE DOES NOT" if crossed else "")
        )

    out.update({
        "usable": True,
        "levels": [str(a), str(b)],
        "p_naive": float(p_naive) if np.isfinite(p_naive) else np.nan,
        "p_clustered": float(p_clustered) if np.isfinite(p_clustered) else np.nan,
        "p_ratio": inflation,
        "chi2_naive": float(chi2) if np.isfinite(chi2) else np.nan,
        "icc": icc_pooled,
        "icc_by_group": {str(a): icc_a.get("icc"), str(b): icc_b.get("icc")},
        "icc_betabinomial": rho_bb,
        "design_effect": deff["deff"],
        "se_inflation": deff["se_inflation"],
        "n_flies": n_total,
        "n_effective": n_eff,
        "n_clusters": int(work[cluster_col].nunique()),
        "mean_flies_per_cluster": mean_m,
        "conclusion_changes": bool(
            np.isfinite(p_naive) and np.isfinite(p_clustered)
            and (p_naive < 0.05) != (p_clustered < 0.05)
        ),
        "verdict": "; ".join(verdict_bits),
        "gee_note": gee_note,
    })
    return out
