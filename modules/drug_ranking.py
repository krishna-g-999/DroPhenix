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
def inverse_variance_pool(effects: Sequence[float], variances: Sequence[float]) -> dict:
    """
    Combine independent effect estimates of differing precision.

    Returns fixed-effect and DerSimonian-Laird random-effects pooled estimates
    with SE and 95% CI, plus Cochran's Q, df, tau^2, and I^2.
    """
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[mask], v[mask]
    k = len(y)
    if k == 0:
        return {"k": 0}
    w = 1.0 / v
    fe = float(np.sum(w * y) / np.sum(w))
    var_fe = 1.0 / np.sum(w)
    if k == 1:
        return {"k": 1, "fixed": fe, "se_fixed": float(np.sqrt(var_fe)),
                "random": fe, "se_random": float(np.sqrt(var_fe)),
                "ci_lo": float(fe - 1.96 * np.sqrt(var_fe)),
                "ci_hi": float(fe + 1.96 * np.sqrt(var_fe)),
                "Q": 0.0, "df": 0, "tau2": 0.0, "I2": 0.0}
    Q = float(np.sum(w * (y - fe) ** 2))
    df = k - 1
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = 1.0 / (v + tau2)
    re = float(np.sum(wr * y) / np.sum(wr))
    var_re = 1.0 / np.sum(wr)
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    return {"k": k, "fixed": fe, "se_fixed": float(np.sqrt(var_fe)),
            "random": re, "se_random": float(np.sqrt(var_re)),
            "ci_lo": float(re - 1.96 * np.sqrt(var_re)),
            "ci_hi": float(re + 1.96 * np.sqrt(var_re)),
            "Q": Q, "df": df, "tau2": float(tau2), "I2": float(I2)}


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

    # 3. composite Therapeutic Score across assays (inverse-variance, optional weights)
    comp_rows = []
    if not per_assay.empty:
        for drug, sub in per_assay.groupby("drug"):
            g = sub["g_pooled"].to_numpy()
            var = sub["var"].to_numpy()
            if assay_weights:
                wmul = np.array([assay_weights.get(a, 1.0) for a in sub["assay"]])
                # fold weights in by inflating variance inversely (lower weight -> higher var)
                var = var / np.clip(wmul, 1e-9, None)
            pooled = inverse_variance_pool(g, var)
            comp_rows.append({
                "drug": drug, "n_assays": int(len(sub)),
                "assays": ", ".join(sorted(sub["assay"])),
                # Primary: inverse-variance (precision-weighted) composite.
                "Therapeutic_Score": round(pooled["fixed"], 4),
                "se": round(pooled["se_fixed"], 4),
                "ci_lo": round(pooled["fixed"] - 1.96 * pooled["se_fixed"], 4),
                "ci_hi": round(pooled["fixed"] + 1.96 * pooled["se_fixed"], 4),
                "z": round(pooled["fixed"] / pooled["se_fixed"], 3) if pooled["se_fixed"] > 0 else np.nan,
                # Secondary: each assay counts equally (mean of per-assay g),
                # so no single low-noise assay dominates the score.
                "Score_equal_assay": round(float(np.mean(g)), 4),
            })
    composite = pd.DataFrame(comp_rows)
    if not composite.empty:
        composite["p_value"] = composite["z"].apply(
            lambda z: round(2 * (1 - stats.norm.cdf(abs(z))), 6) if pd.notna(z) else np.nan)
        composite = composite.sort_values("Therapeutic_Score", ascending=False).reset_index(drop=True)
        composite.insert(0, "Rank", np.arange(1, len(composite) + 1))

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
