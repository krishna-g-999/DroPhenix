"""
DroPhenix — drug_ranking.py
Cross-strain, cross-assay compound ranking with rigorous unequal-N handling.

The problem (from the project objective)
----------------------------------------
A compound is tested in several Drosophila strains across several assays, and
the strains are NOT equally represented (e.g. TDP43 has 3 vials, HTT-Q93 has 7).
We must quantify the drug's effect per strain, then combine across strains and
assays into a single, defensible ranking — without letting small, noisy strains
dominate or be ignored.

Statistical / actuarial approach
--------------------------------
1. Per strain x sex, the drug-vs-vehicle effect is a standardized mean
   difference corrected for small and unequal samples: Hedges' g
   (Hedges & Olkin, 1985). Its sampling variance explicitly depends on the two
   group sizes, so unequal N is handled at the source.
       d  = (mean_drug - mean_vehicle) / s_pooled
       J  = 1 - 3/(4*(n1+n2) - 9)          (small-sample correction)
       g  = J * d
       Var(g) = J^2 * [ (n1+n2)/(n1*n2) + d^2 / (2*(n1+n2-2)) ]
   For survival, the effect is the Cox log-hazard-ratio (drug vs vehicle),
   converted to the same standardized scale via d = logHR * sqrt(3)/pi
   (Chinn, 2000) so it can be combined with the other assays.

2. Across strains, effects are combined by INVERSE-VARIANCE weighting — the
   actuarially correct way to pool estimates of differing precision: a strain
   with larger N (smaller variance) receives proportionally more weight. We
   report both the fixed-effect estimate and the DerSimonian-Laird (1986)
   random-effects estimate, plus Cochran's Q, tau^2 and I^2 (heterogeneity =
   does the drug act consistently across strains?).

3. Each metric is oriented so that POSITIVE = therapeutic benefit (e.g. higher
   climbing PI, higher alternation, longer lifespan = lower hazard, circadian
   period closer to 24 h). Per-assay pooled effects are then combined across
   assays by inverse-variance weighting into a single Therapeutic Score, and
   compounds are ranked by it.

All pooling math is validated against statsmodels.stats.meta_analysis in
validation/validate_drug_ranking.py.

KNOWN LIMITATION — small-k confidence intervals
-----------------------------------------------
The DerSimonian-Laird interval uses a normal 1.96 multiplier on the
random-effects SE. That interval is known to UNDERCOVER when the number of
pooled units k is small (IntHout, Ioannidis & Borm 2014, BMC Med Res Methodol
14:25), which is the usual case here: a compound is typically tested in 2-4
strains. The Hartung-Knapp-Sidik-Jonkman adjustment (a t multiplier on df = k-1
with a variance-corrected SE) is the recommended interval in that regime and is
what `metafor` would apply. It is NOT implemented here; the reported CIs should
be read as somewhat too narrow when n_strata is below about 5, and that is
stated rather than hidden. See docs/MODULE_AUDIT_2026-09.md finding 3.8.

Note also that `drug_ranking.hedges_g(control, treated)` and
`stats_engine.hedges_g(a, b)` take their arguments in OPPOSITE orders. Both are
internally consistent and correctly called; do not mix them up.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

HR_TO_D = np.sqrt(3) / np.pi   # log-hazard-ratio -> standardized mean diff (Chinn 2000)


# ──────────────────────────────────────────────────────────────────────────────
# Standardized effect size (Hedges' g) with unequal-N variance
# ──────────────────────────────────────────────────────────────────────────────
def hedges_g(control: Sequence[float], treated: Sequence[float]) -> dict:
    """
    Hedges' g for treated-vs-control with its unequal-N sampling variance.
    Returns g, se, var, ci_lo, ci_hi, z, p, n_control, n_treated.
    Positive g = treated mean higher than control.
    """
    c = np.asarray(control, dtype=float); c = c[~np.isnan(c)]
    t = np.asarray(treated, dtype=float); t = t[~np.isnan(t)]
    n1, n2 = len(t), len(c)
    if n1 < 2 or n2 < 2:
        return {"g": np.nan, "se": np.nan, "var": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "z": np.nan, "p": np.nan, "n_control": n2, "n_treated": n1}
    s1, s2 = np.var(t, ddof=1), np.var(c, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp == 0 or not np.isfinite(sp):
        return {"g": np.nan, "se": np.nan, "var": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "z": np.nan, "p": np.nan, "n_control": n2, "n_treated": n1}
    d = (np.mean(t) - np.mean(c)) / sp
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = J * d
    var_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2))
    var_g = J ** 2 * var_d
    se = np.sqrt(var_g)
    z = g / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return {"g": float(g), "se": float(se), "var": float(var_g),
            "ci_lo": float(g - 1.96 * se), "ci_hi": float(g + 1.96 * se),
            "z": float(z) if np.isfinite(z) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
            "n_control": n2, "n_treated": n1}


# ──────────────────────────────────────────────────────────────────────────────
# Inverse-variance pooling (fixed + DerSimonian-Laird random effects)
# ──────────────────────────────────────────────────────────────────────────────
def inverse_variance_pool(effects: Sequence[float], variances: Sequence[float],
                          level: float = 0.95) -> dict:
    """
    Combine independent effect estimates of differing precision.

    Returns the fixed-effect and DerSimonian-Laird random-effects pooled
    estimates, Cochran's Q, df, tau^2 and I^2, and -- as the PRIMARY interval
    -- the Hartung-Knapp-Sidik-Jonkman interval.

    Why HKSJ (register N35). The DerSimonian-Laird interval treats tau^2 as if
    it were known, and uses a normal quantile. With the handful of strata or
    assays a fly experiment actually has, tau^2 is estimated very imprecisely
    and the interval is too narrow: measured coverage is 88.2% at k = 3,
    tau^2 = 0.30, against a nominal 95%. HKSJ replaces the variance with a
    quantity estimated from the weighted residual spread,

        var_HK = (1 / (k - 1)) * sum_i w_i (y_i - theta_RE)^2 / sum_i w_i,
        w_i = 1 / (v_i + tau^2),

    and uses t(k-1) instead of 1.96, giving 94.6% in the same simulation.
    Following IntHout et al. (2014) and Roever et al. (2015) the ad-hoc
    variance correction is applied: when var_HK falls BELOW the DL variance the
    DL variance is used, because HKSJ can otherwise produce an interval
    narrower than the fixed-effect one when the studies happen to agree
    closely, which is an artefact of the small sample of studies and not
    evidence of precision.

    k = 1 has no between-study information at all: there is no tau^2 to
    estimate and no t(k-1), so the interval reported is that single estimate's
    own normal-theory interval and `method` says so.

    References
    ----------
    DerSimonian R, Laird N (1986) Meta-analysis in clinical trials.
      Controlled Clinical Trials 7:177-188.
    Hartung J, Knapp G (2001) A refined method for the meta-analysis of
      controlled clinical trials with a binary outcome. Stat Med 20:3875-3889.
    Sidik K, Jonkman JN (2002) A simple confidence interval for meta-analysis.
      Stat Med 21:3153-3159.
    IntHout J, Ioannidis JPA, Borm GF (2014) The Hartung-Knapp-Sidik-Jonkman
      method for random effects meta-analysis is straightforward and
      considerably outperforms the standard DerSimonian-Laird method.
      BMC Medical Research Methodology 14:25.
    Roever C, Knapp G, Friede T (2015) Hartung-Knapp-Sidik-Jonkman approach and
      its modification for random-effects meta-analysis with few studies.
      BMC Medical Research Methodology 15:99.
    """
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[mask], v[mask]
    k = len(y)
    if k == 0:
        return {"k": 0}
    alpha = 1.0 - float(level)
    w = 1.0 / v
    fe = float(np.sum(w * y) / np.sum(w))
    var_fe = 1.0 / np.sum(w)
    se_fe = float(np.sqrt(var_fe))
    p_fixed = float(2.0 * stats.norm.sf(abs(fe / se_fe))) if se_fe > 0 else np.nan

    if k == 1:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return {"k": 1, "fixed": fe, "se_fixed": se_fe,
                "random": fe, "se_random": se_fe, "se_hksj": se_fe,
                "t_crit": z, "df_hksj": 0,
                "ci_lo": float(fe - z * se_fe), "ci_hi": float(fe + z * se_fe),
                "ci_lo_dl": float(fe - z * se_fe), "ci_hi_dl": float(fe + z * se_fe),
                "p_fixed": p_fixed, "p_random": p_fixed, "p_random_dl": p_fixed,
                "Q": 0.0, "df": 0, "tau2": 0.0, "I2": 0.0,
                "hksj_floored": False, "level": float(level),
                "interval_warning": (
                    "k = 1: there is no between-study variance to estimate and "
                    "no Hartung-Knapp interval; this is one estimate's own "
                    "interval and carries none of the uncertainty about whether "
                    "other strata or assays would agree with it."),
                "method": ("Single estimate (k = 1): no between-study variance is "
                           "estimable and no Hartung-Knapp interval exists; the "
                           "interval is that estimate's own normal-theory interval.")}

    Q = float(np.sum(w * (y - fe) ** 2))
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = 1.0 / (v + tau2)
    re = float(np.sum(wr * y) / np.sum(wr))
    var_re = 1.0 / np.sum(wr)
    se_re = float(np.sqrt(var_re))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0

    # Hartung-Knapp-Sidik-Jonkman variance, with the ad-hoc floor at the DL
    # variance (IntHout 2014; Roever 2015).
    var_hk = float(np.sum(wr * (y - re) ** 2) / (df * np.sum(wr)))
    floored = bool(var_hk < var_re)
    var_hksj = max(var_hk, var_re)
    se_hksj = float(np.sqrt(var_hksj))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    # Two random-effects p-values are reported because they differ materially
    # at the k this platform works with. `p_random` is the HKSJ one -- t(k-1)
    # on the HKSJ standard error -- and is the one to report. `p_random_dl` is
    # the DerSimonian-Laird figure (normal quantile on the DL standard error),
    # kept so the gap is visible: on the register's four-assay case the fixed
    # effect gives 3.0e-7, DL gives 0.107, and HKSJ gives 0.205.
    p_random = float(2.0 * stats.t.sf(abs(re / se_hksj), df)) if se_hksj > 0 else np.nan
    p_random_dl = float(2.0 * stats.norm.sf(abs(re / se_re))) if se_re > 0 else np.nan

    # The cost of HKSJ at small k, measured on 2000 replicates per cell
    # (…\scratchpad\coverage_hksj.py). Coverage of the nominal 95% interval,
    # and the HKSJ-to-DL width ratio:
    #
    #   k    tau2=0           tau2=0.30        width ratio
    #   2    DL 95.8 HK 100   DL 80.4 HK 100   6.48
    #   3    DL 96.8 HK 100   DL 85.1 HK 98.7  2.2
    #   4    DL 96.0 HK 99.9  DL 86.9 HK 97.0  1.65
    #   6    DL 96.5 HK 99.3  DL 90.0 HK 96.2  1.33
    #  10    DL 96.7 HK 98.6  DL 92.2 HK 95.3  1.17
    #
    # HKSJ never UNDER-covers in that range; DL under-covers whenever tau2 > 0,
    # which is exactly the case this platform meets (four different assays, or
    # four strains). Under-coverage is the dangerous direction for a therapeutic
    # claim, so HKSJ is the default. But at k = 2 the t(1) quantile is 12.7 and
    # the interval is 6.5x the DL one: technically honest and practically a
    # refusal, because two studies carry almost no information about the
    # between-study variance. That is said rather than left for the reader to
    # infer from a very wide interval.
    warn = ""
    if k <= 3:
        warn = (
            f"k = {k}: the interval is dominated by the t({df}) quantile "
            f"({t_crit:.2f} against 1.96) because {k} estimates barely determine "
            f"the between-study variance tau^2. It is conservative by "
            f"construction -- measured coverage at k = {k} is 97-100% against a "
            f"nominal 95% -- so a wide interval here is weak information, not a "
            f"wide true effect. The fixed-effect and DerSimonian-Laird figures "
            f"are reported alongside; neither is a substitute, since DL "
            f"under-covers (80-85% at tau^2 = 0.30) in the same cells."
        )
    return {"k": k, "fixed": fe, "se_fixed": se_fe,
            "random": re, "se_random": se_re,
            "se_hksj": se_hksj, "t_crit": t_crit, "df_hksj": df,
            "interval_warning": warn,
            # Primary interval: HKSJ.
            "ci_lo": float(re - t_crit * se_hksj),
            "ci_hi": float(re + t_crit * se_hksj),
            # Secondary, kept so the difference is visible rather than implied.
            "ci_lo_dl": float(re - z * se_re),
            "ci_hi_dl": float(re + z * se_re),
            "p_fixed": p_fixed, "p_random": p_random, "p_random_dl": p_random_dl,
            "Q": Q, "df": df, "tau2": float(tau2), "I2": float(I2),
            "hksj_floored": floored, "level": float(level),
            "method": ("Random effects, DerSimonian-Laird tau^2 with the "
                       "Hartung-Knapp-Sidik-Jonkman variance and a t(k-1) "
                       "quantile (HKSJ); fixed effect reported as secondary.")}


HETEROGENEITY_I2_P_LIMIT = 50.0


def composite_score(per_assay: pd.DataFrame,
                    assay_weights: Optional[Dict[str, float]] = None,
                    level: float = 0.95) -> pd.DataFrame:
    """
    Per-drug Therapeutic Score across assays.

    Register N34. The composite used to be reported as a FIXED-EFFECT pooled
    estimate -- point estimate, SE, 1.96 interval, z and p -- in the same row
    as `I2_assay_pct` and `assays_consistent`. A fixed-effect pool across
    climbing, lifespan, Y-maze and circadian period assumes those four
    measurements share one underlying effect, which is the assumption in this
    whole module most likely to fail, and the row reported the evidence that it
    had failed next to a p-value computed as though it had not. On the
    register's four per-assay effects (1.20, -0.10, 0.90, 0.05, each var 0.04)
    that produced p = 3.0e-7 where random effects on the same four numbers give
    p = 0.107: four orders of magnitude, and the difference between
    "overwhelming cross-assay evidence" and none.

    What is reported now.
      * `Therapeutic_Score` is the RANDOM-EFFECTS estimate, with an HKSJ
        interval (register N35) -- small k is the normal case here.
      * `Score_fixed` and `se_fixed` are kept as secondary, so a reader can see
        what the precision-weighted pool would have said.
      * `Score_equal_assay` is unchanged: the unweighted mean across assays, so
        no single low-noise assay dominates.
      * When I2 across assays reaches `HETEROGENEITY_I2_P_LIMIT` the p-value is
        NaN, not small, and `p_value_note` states that a single cross-assay
        p-value is undefined when the assays disagree. There is no pooled
        effect for it to test: the per-assay rows are the result.

    `assay_weights` still folds a prior preference between assays into the
    variances, and that is a judgement, not a measurement -- it is recorded in
    the returned frame's `attrs`.
    """
    rows = []
    if per_assay is None or per_assay.empty:
        out = pd.DataFrame()
        out.attrs["assay_weights"] = dict(assay_weights or {})
        return out

    for drug, sub in per_assay.groupby("drug"):
        g = sub["g_pooled"].to_numpy(dtype=float)
        var = sub["var"].to_numpy(dtype=float)
        if assay_weights:
            wmul = np.array([assay_weights.get(a, 1.0) for a in sub["assay"]], dtype=float)
            # Fold weights in by inflating variance inversely (lower weight ->
            # higher variance). This is a stated preference, not evidence.
            var = var / np.clip(wmul, 1e-9, None)
        pooled = inverse_variance_pool(g, var, level=level)
        if pooled.get("k", 0) == 0:
            continue

        i2 = float(pooled.get("I2", np.nan))
        consistent = (i2 < HETEROGENEITY_I2_P_LIMIT) if np.isfinite(i2) else None
        se_h = pooled["se_hksj"]
        t_stat = (pooled["random"] / se_h) if se_h > 0 else np.nan

        if np.isfinite(i2) and i2 >= HETEROGENEITY_I2_P_LIMIT:
            p_val = np.nan
            note = (
                f"A single cross-assay p-value is UNDEFINED here: I2 = {i2:.1f}% of "
                f"the variation between the {int(pooled['k'])} assays is not "
                f"attributable to sampling error, so there is no common effect for "
                f"a p-value to test. The assays disagree, and that disagreement is "
                f"the finding. Read the per-assay rows and Score_equal_assay; the "
                f"interval reported here still describes the spread of assay "
                f"effects, but it is not a test of a shared one."
            )
        else:
            p_val = pooled["p_random"]
            note = ""

        rows.append({
            "drug": drug, "n_assays": int(pooled["k"]),
            "assays": ", ".join(sorted(sub["assay"])),
            # Primary: random effects with an HKSJ interval.
            "Therapeutic_Score": round(pooled["random"], 4),
            "se": round(se_h, 4),
            "ci_lo": round(pooled["ci_lo"], 4),
            "ci_hi": round(pooled["ci_hi"], 4),
            "t": round(float(t_stat), 3) if np.isfinite(t_stat) else np.nan,
            "df": int(pooled["df_hksj"]),
            "p_value": p_val,
            "p_value_note": note,
            # Secondary: what the fixed-effect pool would have said.
            "Score_fixed": round(pooled["fixed"], 4),
            "se_fixed": round(pooled["se_fixed"], 4),
            "p_fixed": pooled["p_fixed"],
            "Q_assay": round(float(pooled.get("Q", np.nan)), 3),
            "I2_assay_pct": round(i2, 1) if np.isfinite(i2) else np.nan,
            "tau2_assay": round(float(pooled.get("tau2", np.nan)), 4),
            "assays_consistent": consistent,
            "ci_method": pooled.get("method", ""),
            "interval_warning": pooled.get("interval_warning", ""),
            # Each assay counts equally, so no single low-noise assay dominates.
            "Score_equal_assay": round(float(np.mean(g)), 4),
        })

    composite = pd.DataFrame(rows)
    if not composite.empty:
        composite = composite.sort_values("Therapeutic_Score",
                                          ascending=False).reset_index(drop=True)
        composite.insert(0, "Rank", np.arange(1, len(composite) + 1))
    composite.attrs["assay_weights"] = dict(assay_weights or {})
    composite.attrs["i2_p_limit"] = HETEROGENEITY_I2_P_LIMIT
    return composite


# ──────────────────────────────────────────────────────────────────────────────
# Per-assay drug-vs-vehicle effects (oriented so positive = benefit)
# ──────────────────────────────────────────────────────────────────────────────
def _per_fly_value_effects(per_fly: pd.DataFrame, metric: str, higher_is_better: bool,
                           vehicle: str, drug: str, assay: str) -> List[dict]:
    """Hedges' g (drug vs vehicle) per Genotype x Sex stratum, oriented to benefit."""
    out = []
    sign = 1.0 if higher_is_better else -1.0
    for (geno, sex), sub in per_fly.groupby(["Genotype", "Sex"]):
        ctrl = sub.loc[sub["Treatment"] == vehicle, metric]
        trt = sub.loc[sub["Treatment"] == drug, metric]
        if len(ctrl) < 2 or len(trt) < 2:
            continue
        eff = hedges_g(ctrl.to_numpy(), trt.to_numpy())
        if not np.isfinite(eff["g"]):
            continue
        out.append({"assay": assay, "Genotype": geno, "Sex": sex, "drug": drug,
                    "g": sign * eff["g"], "se": eff["se"], "var": eff["var"],
                    "n_vehicle": eff["n_control"], "n_drug": eff["n_treated"],
                    "p": eff["p"]})
    return out


def climbing_effects(climbing_df: pd.DataFrame, vehicle: str, drugs: List[str]) -> pd.DataFrame:
    """Endpoint-PI Hedges' g (drug vs vehicle) per strain x sex. Higher PI = better."""
    from modules.data_parser import load_data
    from modules.normalizer import compute_all_metrics
    df, _, _ = load_data(climbing_df)
    pi_raw = compute_all_metrics(df)["pi_raw"]   # replicate-level PI
    rows = []
    for drug in drugs:
        rows += _per_fly_value_effects(pi_raw.rename(columns={"PI": "PI"}),
                                       "PI", True, vehicle, drug, "climbing")
    return pd.DataFrame(rows)


def ymaze_effects(ymaze_df: pd.DataFrame, vehicle: str, drugs: List[str],
                  metric: str = "alternation_pct") -> pd.DataFrame:
    """Y-maze Hedges' g (drug vs vehicle) per strain x sex. Higher alternation = better."""
    from modules.ymaze import load_ymaze, per_fly_metrics
    pf = per_fly_metrics(load_ymaze(ymaze_df))
    rows = []
    for drug in drugs:
        rows += _per_fly_value_effects(pf, metric, True, vehicle, drug, "ymaze")
    return pd.DataFrame(rows)


def dams_effects(dams_df: pd.DataFrame, vehicle: str, drugs: List[str],
                 target_period: float = 24.0) -> pd.DataFrame:
    """
    DAMS circadian Hedges' g per strain x sex. Benefit metric = -|period - 24 h|
    (closer to 24 h = better), so a drug that restores the period scores positive.
    """
    from modules.dams import load_dams, period_per_fly
    ppf = period_per_fly(load_dams(dams_df))
    ppf = ppf.copy()
    ppf["period_dev"] = -(ppf["Period_h_chisq"] - target_period).abs()  # higher = better
    rows = []
    for drug in drugs:
        rows += _per_fly_value_effects(ppf, "period_dev", True, vehicle, drug, "dams")
    return pd.DataFrame(rows)


def survival_effects(lifespan_df: pd.DataFrame, vehicle: str, drugs: List[str]) -> pd.DataFrame:
    """
    Survival effect per strain x sex from a Cox model (drug vs vehicle within
    stratum). Benefit = -logHR (drug lowering death hazard scores positive),
    converted to the standardized (g) scale via sqrt(3)/pi for cross-assay
    combination. SE scales by the same factor.
    """
    from modules.survival import to_event_table, cox_ph
    ev = to_event_table(lifespan_df)
    rows = []
    for drug in drugs:
        for (geno, sex), sub in ev.groupby(["Genotype", "Sex"]):
            pair = sub[sub["Treatment"].isin([vehicle, drug])]
            if pair["Treatment"].nunique() < 2:
                continue
            n_v = int((pair["Treatment"] == vehicle).sum())
            n_d = int((pair["Treatment"] == drug).sum())
            if min(n_v, n_d) < 3:
                continue
            try:
                cox = cox_ph(pair, covariates=["Treatment"], interaction=False,
                             ref={"Treatment": vehicle})
                r = cox["summary"].iloc[0]
                log_hr = float(r["log_HR"])
                # SE(logHR) from the 95% CI on the HR scale
                se_loghr = (np.log(float(r["HR_CI_upper"])) - np.log(float(r["HR_CI_lower"]))) / (2 * 1.96)
            except Exception:
                continue
            g = -log_hr * HR_TO_D          # benefit: lower hazard -> positive
            se = se_loghr * HR_TO_D
            if not (np.isfinite(g) and np.isfinite(se) and se > 0):
                continue
            rows.append({"assay": "survival", "Genotype": geno, "Sex": sex, "drug": drug,
                         "g": g, "se": se, "var": se ** 2,
                         "n_vehicle": n_v, "n_drug": n_d, "p": float(r["p_value"])})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level ranking
# ──────────────────────────────────────────────────────────────────────────────
def rank_compounds(data: Dict[str, pd.DataFrame], vehicle: str = "Vehicle",
                   drugs: Optional[List[str]] = None,
                   assay_weights: Optional[Dict[str, float]] = None) -> dict:
    """
    Full cross-assay, cross-strain ranking.

    data : {'climbing','lifespan','ymaze','dams'} DataFrames (any subset).
    Returns dict with:
      per_stratum   : every strain x sex x assay drug effect (Hedges' g, unequal-N)
      per_assay     : per drug x assay pooled effect across strains (DL random eff.)
      composite     : per drug Therapeutic Score across assays + rank
      unequal_n     : per group sample sizes and imbalance flags
    """
    # discover drugs (any treatment that is not the vehicle), from any assay
    if drugs is None:
        found = set()
        for k, df in data.items():
            if df is not None and not df.empty and "Treatment" in df.columns:
                found |= set(df["Treatment"].unique())
        drugs = sorted(found - {vehicle})

    # 1. per-stratum effects
    eff_frames = []
    if "climbing" in data and data["climbing"] is not None and not data["climbing"].empty:
        eff_frames.append(climbing_effects(data["climbing"], vehicle, drugs))
    if "ymaze" in data and data["ymaze"] is not None and not data["ymaze"].empty:
        eff_frames.append(ymaze_effects(data["ymaze"], vehicle, drugs))
    if "dams" in data and data["dams"] is not None and not data["dams"].empty:
        eff_frames.append(dams_effects(data["dams"], vehicle, drugs))
    if "lifespan" in data and data["lifespan"] is not None and not data["lifespan"].empty:
        eff_frames.append(survival_effects(data["lifespan"], vehicle, drugs))
    per_stratum = pd.concat([f for f in eff_frames if not f.empty], ignore_index=True) \
        if any((not f.empty) for f in eff_frames) else pd.DataFrame()

    # 2. pool across strains (x sex) within each drug x assay (DL random effects)
    per_assay_rows = []
    if not per_stratum.empty:
        for (drug, assay), sub in per_stratum.groupby(["drug", "assay"]):
            pooled = inverse_variance_pool(sub["g"].to_numpy(), sub["var"].to_numpy())
            if pooled.get("k", 0) == 0:
                continue
            per_assay_rows.append({
                "drug": drug, "assay": assay, "n_strata": pooled["k"],
                "g_pooled": round(pooled["random"], 4),
                "se": round(pooled["se_random"], 4),
                "ci_lo": round(pooled["ci_lo"], 4), "ci_hi": round(pooled["ci_hi"], 4),
                "I2_pct": round(pooled["I2"], 1), "tau2": round(pooled["tau2"], 4),
                "var": pooled["se_random"] ** 2,
            })
    per_assay = pd.DataFrame(per_assay_rows)

    # 3. composite Therapeutic Score across assays (random effects + HKSJ)
    composite = composite_score(per_assay, assay_weights=assay_weights)

    # 4. unequal-N report
    un_rows = []
    for assay, df in data.items():
        if df is None or df.empty or "Treatment" not in df.columns:
            continue
        unit = "FlyID" if "FlyID" in df.columns else None
        for (geno, treat, sex), sub in df.groupby(["Genotype", "Treatment", "Sex"]):
            n = sub[unit].nunique() if unit else (
                sub["Replicate"].nunique() if "Replicate" in sub.columns else len(sub))
            un_rows.append({"assay": assay, "Genotype": geno, "Treatment": treat,
                            "Sex": sex, "N": int(n)})
    unequal_n = pd.DataFrame(un_rows)
    if not unequal_n.empty:
        imb = (unequal_n.groupby("assay")["N"]
               .agg(N_min="min", N_max="max", N_mean="mean").reset_index())
        imb["imbalance_ratio"] = (imb["N_max"] / imb["N_min"]).round(2)
        imb["flag"] = np.where(imb["imbalance_ratio"] >= 1.5,
                               "unequal N (inverse-variance weighting applied)", "balanced")
    else:
        imb = pd.DataFrame()

    return {"per_stratum": per_stratum, "per_assay": per_assay,
            "composite": composite, "unequal_n": unequal_n, "imbalance": imb,
            "drugs": drugs, "vehicle": vehicle}
