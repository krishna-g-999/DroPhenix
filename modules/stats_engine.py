"""
DroPhenix Analytics — stats_engine.py
Publication-safer statistical testing engine for Drosophila climbing assay data.

Design goals
------------
- Keep statistical logic transparent and reproducible.
- Prefer conservative decisions when sample size is small.
- Always accompany p-values with effect sizes.
- Support both direct dict-based testing and DataFrame-based workflows.
- Preserve backward compatibility with existing DroPhenix app imports.

Implemented
-----------
1. Shapiro-Wilk normality testing
2. Levene variance testing
3. Welch t-test / Mann-Whitney U for 2-group comparisons
4. One-way ANOVA / Welch-style fallback / Kruskal-Wallis for >2 groups
5. Pairwise post hoc testing with BH-FDR correction
6. Bootstrap confidence intervals
7. SDQ permutation testing
8. auto_test() wrapper for dict[str, array-like]
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


MIN_GROUP_SIZE = 3
MIN_GROUP_SIZE_NORM = 7
DEFAULT_BOOT = 2000
DEFAULT_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Utility labels
# ──────────────────────────────────────────────────────────────────────────────
def sig_label(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _cohen_d_label(d: float) -> str:
    if pd.isna(d):
        return "n/a"
    ad = abs(d)
    if ad >= 0.8:
        return "large"
    if ad >= 0.5:
        return "medium"
    if ad >= 0.2:
        return "small"
    return "negligible"


def _rbr_label(r: float) -> str:
    if pd.isna(r):
        return "n/a"
    ar = abs(r)
    if ar >= 0.5:
        return "large"
    if ar >= 0.3:
        return "medium"
    if ar >= 0.1:
        return "small"
    return "negligible"


# ──────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ──────────────────────────────────────────────────────────────────────────────
def _clean_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[~np.isnan(arr)]


def _safe_mean(a: np.ndarray) -> float:
    return float(np.mean(a)) if len(a) else np.nan


def _safe_median(a: np.ndarray) -> float:
    return float(np.median(a)) if len(a) else np.nan


def _safe_std(a: np.ndarray) -> float:
    return float(np.std(a, ddof=1)) if len(a) > 1 else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Effect sizes
# ──────────────────────────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for two independent groups using pooled SD.
    """
    a = _clean_array(a)
    b = _clean_array(b)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    n1, n2 = len(a), len(b)
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """
    Small-sample corrected standardized mean difference.
    """
    d = cohens_d(a, b)
    a = _clean_array(a)
    b = _clean_array(b)
    n = len(a) + len(b)
    if np.isnan(d) or n <= 3:
        return np.nan
    correction = 1 - (3 / (4 * n - 9))
    return float(d * correction)


def rank_biserial_r(a: np.ndarray, b: np.ndarray) -> float:
    """
    Rank-biserial correlation for Mann-Whitney U.
    """
    a = _clean_array(a)
    b = _clean_array(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    try:
        u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        return np.nan
    return float(1 - (2 * u_stat) / (n1 * n2))


# ──────────────────────────────────────────────────────────────────────────────
# Multiple testing
# ──────────────────────────────────────────────────────────────────────────────
def bh_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Handles NaN safely by leaving them NaN.
    """
    pv = np.asarray(p_values, dtype=float)
    out = np.full_like(pv, np.nan, dtype=float)

    mask = ~np.isnan(pv)
    if mask.sum() == 0:
        return out

    vals = pv[mask]
    n = len(vals)
    order = np.argsort(vals)
    ranks = np.arange(1, n + 1)

    adj = np.empty(n, dtype=float)
    adj[order] = np.minimum(1.0, vals[order] * n / ranks)

    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])

    out[mask] = np.clip(adj, 0.0, 1.0)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Assumption checks
# ──────────────────────────────────────────────────────────────────────────────
def test_normality(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Shapiro-Wilk normality test per group.
    n < MIN_GROUP_SIZE: insufficient for testing
    n < MIN_GROUP_SIZE_NORM: test reported, but flagged as low-confidence
    """
    rows = []
    for group, values in data.items():
        v = _clean_array(values)
        n = len(v)

        if n < MIN_GROUP_SIZE:
            rows.append(
                {
                    "Group": group,
                    "n": n,
                    "W": np.nan,
                    "p_value": np.nan,
                    "Normal": False,
                    "Decision": f"Insufficient data (n={n})",
                }
            )
            continue

        # Guard: identical replicates -> range=0 -> Shapiro undefined
        if np.ptp(v) == 0:
            rows.append({
                "Group": group, "n": n,
                "W": float("nan"), "p_value": float("nan"),
                "Normal": False,
                "Decision": f"Constant data (range=0) non-normal by definition",
            })
            continue
        try:
            w, p = stats.shapiro(v)
        except Exception:
            w, p = np.nan, np.nan

        normal = bool(p > 0.05) if pd.notna(p) else False
        note = ""
        if n < MIN_GROUP_SIZE_NORM:
            note = f" [caution: n={n}, Shapiro unreliable < {MIN_GROUP_SIZE_NORM}]"

        rows.append(
            {
                "Group": group,
                "n": n,
                "W": round(float(w), 4) if pd.notna(w) else np.nan,
                "p_value": round(float(p), 6) if pd.notna(p) else np.nan,
                "Normal": normal,
                "Decision": ("Normal" if normal else "Non-normal") + note,
            }
        )
    return pd.DataFrame(rows)


def test_levene(data: Dict[str, np.ndarray]) -> dict:
    """
    Levene test for homogeneity of variance.
    """
    arrays = [_clean_array(v) for v in data.values()]
    valid = [a for a in arrays if len(a) >= MIN_GROUP_SIZE]

    if len(valid) < 2:
        return {
            "Levene_statistic": np.nan,
            "p_value": np.nan,
            "Equal_variance": "Insufficient data",
        }

    try:
        stat, p = stats.levene(*valid, center="median")
        return {
            "Levene_statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "Equal_variance": "Yes" if p > 0.05 else "No",
        }
    except Exception as e:
        return {
            "Levene_statistic": np.nan,
            "p_value": np.nan,
            "Equal_variance": f"Error: {e}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Two-group tests
# ──────────────────────────────────────────────────────────────────────────────
def _two_group_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, prefer_parametric: bool) -> dict:
    a = _clean_array(a)
    b = _clean_array(b)

    if len(a) < MIN_GROUP_SIZE or len(b) < MIN_GROUP_SIZE:
        return {
            "method": "Insufficient data",
            "main_result": {
                "Statistic": np.nan,
                "p_value": np.nan,
                "Significant": "n/a",
            },
            "pairwise": pd.DataFrame(),
        }

    if prefer_parametric:
        try:
            stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        except Exception:
            stat, p = np.nan, np.nan

        d = cohens_d(a, b)
        g = hedges_g(a, b)

        pairwise = pd.DataFrame(
            [
                {
                    "Group A": label_a,
                    "Group B": label_b,
                    "Mean A": round(_safe_mean(a), 3),
                    "Mean B": round(_safe_mean(b), 3),
                    "Mean_diff": round(_safe_mean(a) - _safe_mean(b), 3),
                    "Statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
                    "p_raw": round(float(p), 6) if pd.notna(p) else np.nan,
                    "p_adj_BH": round(float(p), 6) if pd.notna(p) else np.nan,
                    "Sig": sig_label(p),
                    "Cohens_d": round(d, 3) if pd.notna(d) else np.nan,
                    "Hedges_g": round(g, 3) if pd.notna(g) else np.nan,
                    "Effect_size": _cohen_d_label(d),
                    "Method": "Welch t-test",
                }
            ]
        )

        return {
            "method": "Welch t-test",
            "main_result": {
                "Statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
                "p_value": round(float(p), 6) if pd.notna(p) else np.nan,
                "Significant": sig_label(p),
            },
            "pairwise": pairwise,
        }

    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        stat, p = np.nan, np.nan

    r = rank_biserial_r(a, b)

    pairwise = pd.DataFrame(
        [
            {
                "Group A": label_a,
                "Group B": label_b,
                "Median A": round(_safe_median(a), 3),
                "Median B": round(_safe_median(b), 3),
                "Statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
                "p_raw": round(float(p), 6) if pd.notna(p) else np.nan,
                "p_adj_BH": round(float(p), 6) if pd.notna(p) else np.nan,
                "Sig": sig_label(p),
                "Rank_biserial_r": round(r, 3) if pd.notna(r) else np.nan,
                "Effect_size": _rbr_label(r),
                "Method": "Mann-Whitney U",
            }
        ]
    )

    return {
        "method": "Mann-Whitney U",
        "main_result": {
            "Statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
            "p_value": round(float(p), 6) if pd.notna(p) else np.nan,
            "Significant": sig_label(p),
        },
        "pairwise": pairwise,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-group tests
# ──────────────────────────────────────────────────────────────────────────────
def anova_tukey(data: Dict[str, np.ndarray]) -> Tuple[dict, pd.DataFrame]:
    """
    One-way ANOVA + pairwise Welch t-tests with BH-FDR correction.
    Uses pairwise Welch tests as a robust practical post hoc layer.
    """
    groups = list(data.keys())
    arrays = [_clean_array(v) for v in data.values()]
    valid = [(g, a) for g, a in zip(groups, arrays) if len(a) >= MIN_GROUP_SIZE]

    if len(valid) < 2:
        return {
            "F_statistic": np.nan,
            "p_value": np.nan,
            "Significant": "Insufficient data",
        }, pd.DataFrame()

    v_groups, v_arrays = zip(*valid)

    try:
        f_stat, p_anova = stats.f_oneway(*v_arrays)
    except Exception:
        f_stat, p_anova = np.nan, np.nan

    pair_meta = []
    raw_p = []

    for i, j in combinations(range(len(v_groups)), 2):
        ga, gb = v_groups[i], v_groups[j]
        a, b = v_arrays[i], v_arrays[j]
        try:
            stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        except Exception:
            stat, p = np.nan, np.nan
        raw_p.append(p if pd.notna(p) else np.nan)
        pair_meta.append((ga, gb, a, b, stat, p))

    adj_p = bh_correction(np.array(raw_p, dtype=float))

    rows = []
    for (ga, gb, a, b, stat, p_raw), p_adj in zip(pair_meta, adj_p):
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        rows.append(
            {
                "Group A": ga,
                "Group B": gb,
                "Mean A": round(_safe_mean(a), 3),
                "Mean B": round(_safe_mean(b), 3),
                "Mean_diff": round(_safe_mean(a) - _safe_mean(b), 3),
                "Statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
                "p_raw": round(float(p_raw), 6) if pd.notna(p_raw) else np.nan,
                "p_adj_BH": round(float(p_adj), 6) if pd.notna(p_adj) else np.nan,
                "Sig": sig_label(p_adj),
                "Cohens_d": round(d, 3) if pd.notna(d) else np.nan,
                "Hedges_g": round(g, 3) if pd.notna(g) else np.nan,
                "Effect_size": _cohen_d_label(d),
                "Method": "Welch t-test + BH-FDR",
            }
        )

    return (
        {
            "F_statistic": round(float(f_stat), 4) if pd.notna(f_stat) else np.nan,
            "p_value": round(float(p_anova), 6) if pd.notna(p_anova) else np.nan,
            "Significant": sig_label(p_anova),
        },
        pd.DataFrame(rows),
    )


def kruskal_dunn(data: Dict[str, np.ndarray]) -> Tuple[dict, pd.DataFrame]:
    """
    Kruskal-Wallis + pairwise Mann-Whitney U with BH-FDR correction.
    """
    groups = list(data.keys())
    arrays = [_clean_array(v) for v in data.values()]
    valid = [(g, a) for g, a in zip(groups, arrays) if len(a) >= MIN_GROUP_SIZE]

    if len(valid) < 2:
        return {
            "H_statistic": np.nan,
            "p_value": np.nan,
            "Significant": "Insufficient data",
        }, pd.DataFrame()

    v_groups, v_arrays = zip(*valid)

    try:
        h_stat, p_kw = stats.kruskal(*v_arrays)
    except Exception:
        h_stat, p_kw = np.nan, np.nan

    pair_meta = []
    raw_p = []

    for i, j in combinations(range(len(v_groups)), 2):
        ga, gb = v_groups[i], v_groups[j]
        a, b = v_arrays[i], v_arrays[j]
        try:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        raw_p.append(p if pd.notna(p) else np.nan)
        pair_meta.append((ga, gb, a, b, stat, p))

    adj_p = bh_correction(np.array(raw_p, dtype=float))

    rows = []
    for (ga, gb, a, b, stat, p_raw), p_adj in zip(pair_meta, adj_p):
        r = rank_biserial_r(a, b)
        rows.append(
            {
                "Group A": ga,
                "Group B": gb,
                "Median A": round(_safe_median(a), 3),
                "Median B": round(_safe_median(b), 3),
                "Statistic": round(float(stat), 4) if pd.notna(stat) else np.nan,
                "p_raw": round(float(p_raw), 6) if pd.notna(p_raw) else np.nan,
                "p_adj_BH": round(float(p_adj), 6) if pd.notna(p_adj) else np.nan,
                "Sig": sig_label(p_adj),
                "Rank_biserial_r": round(r, 3) if pd.notna(r) else np.nan,
                "Effect_size": _rbr_label(r),
                "Method": "Mann-Whitney U + BH-FDR",
            }
        )

    return (
        {
            "H_statistic": round(float(h_stat), 4) if pd.notna(h_stat) else np.nan,
            "p_value": round(float(p_kw), 6) if pd.notna(p_kw) else np.nan,
            "Significant": sig_label(p_kw),
        },
        pd.DataFrame(rows),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Auto-test router
# ──────────────────────────────────────────────────────────────────────────────
def auto_test(data: Dict[str, np.ndarray]) -> dict:
    """
    Automated statistical decision engine for dict[str, array-like].

    Decision policy
    ---------------
    - 2 groups:
        parametric only if both groups have n >= 5 and both pass normality;
        otherwise Mann-Whitney U.
    - >2 groups:
        parametric only if all groups have n >= 5, all pass normality,
        and Levene indicates equal variance; otherwise Kruskal-Wallis.
    """
    if len(data) < 2:
        return {
            "method": "N/A",
            "normality": pd.DataFrame(),
            "levene": {},
            "main_result": {},
            "pairwise": pd.DataFrame(),
            "all_normal": False,
            "equal_variance": False,
        }

    cleaned = {k: _clean_array(v) for k, v in data.items()}
    normality_df = test_normality(cleaned)
    levene_res = test_levene(cleaned)

    valid_ns = normality_df.set_index("Group")["n"].to_dict() if not normality_df.empty else {}
    all_normal = bool(normality_df["Normal"].all()) if len(normality_df) else False
    all_n_ge_5 = all(n >= 5 for n in valid_ns.values()) if valid_ns else False
    equal_var = levene_res.get("Equal_variance", "No") == "Yes"

    group_names = list(cleaned.keys())
    arrays = [cleaned[g] for g in group_names]

    if len(cleaned) == 2:
        prefer_parametric = all_normal and all_n_ge_5
        res = _two_group_test(arrays[0], arrays[1], group_names[0], group_names[1], prefer_parametric)
        return {
            "method": res["method"],
            "normality": normality_df,
            "levene": levene_res,
            "main_result": res["main_result"],
            "pairwise": res["pairwise"],
            "all_normal": all_normal,
            "equal_variance": equal_var,
        }

    use_parametric = all_normal and all_n_ge_5 and equal_var
    if use_parametric:
        main_result, pairwise_df = anova_tukey(cleaned)
        method = "One-way ANOVA + pairwise Welch t-tests (BH-FDR)"
    else:
        main_result, pairwise_df = kruskal_dunn(cleaned)
        method = "Kruskal-Wallis + pairwise Mann-Whitney U (BH-FDR)"

    return {
        "method": method,
        "normality": normality_df,
        "levene": levene_res,
        "main_result": main_result,
        "pairwise": pairwise_df,
        "all_normal": all_normal,
        "equal_variance": equal_var,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = DEFAULT_BOOT,
    ci: float = 0.95,
    stat_fn: Callable = np.mean,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI.
    Returns (lower, upper, point_estimate).
    """
    rng = np.random.default_rng(seed)
    v = _clean_array(values)

    if len(v) == 0:
        return np.nan, np.nan, np.nan

    boot = np.array(
        [stat_fn(rng.choice(v, size=len(v), replace=True)) for _ in range(int(n_boot))],
        dtype=float,
    )

    alpha = 1 - ci
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    est = float(stat_fn(v))
    return round(lo, 3), round(hi, 3), round(est, 3)


def bootstrap_ci_table(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Bootstrap 95% CI for a metric column across groups.
    """
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(grp[value_col], errors="coerce").dropna().values
        if len(vals) < 2:
            continue

        lo, hi, est = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row[value_col] = est
        row[f"{value_col}_CI_lower"] = lo
        row[f"{value_col}_CI_upper"] = hi
        row[f"{value_col}_CI_95"] = f"[{lo}, {hi}]"
        row["n_obs"] = len(vals)
        row["bootstrap_n"] = int(n_boot)
        row["seed"] = int(seed)
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# SDQ permutation
# ──────────────────────────────────────────────────────────────────────────────
def sdq_permutation_test(
    pi_df: pd.DataFrame,
    n_perm: int = 10000,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Permutation test for sex dimorphism in PI.

    Accepts either:
    - replicate-level PI table with columns: Genotype, Treatment, Sex, PI
    - summary-level PI table with columns:   Genotype, Treatment, Sex, PI_mean

    For publication, replicate-level input is preferable.
    """
    if pi_df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)

    if "PI" in pi_df.columns:
        value_col = "PI"
    elif "PI_mean" in pi_df.columns:
        value_col = "PI_mean"
    else:
        raise ValueError("sdq_permutation_test requires either 'PI' or 'PI_mean' column.")

    rows = []
    for (geno, treat), grp in pi_df.groupby(["Genotype", "Treatment"], dropna=False):
        grp = grp.copy()
        grp[value_col] = pd.to_numeric(grp[value_col], errors="coerce")
        grp = grp.dropna(subset=["Sex", value_col])

        male_vals = grp.loc[grp["Sex"] == "Male", value_col].values.astype(float)
        female_vals = grp.loc[grp["Sex"] == "Female", value_col].values.astype(float)

        if len(male_vals) == 0 or len(female_vals) == 0:
            continue

        pi_m = float(np.mean(male_vals))
        pi_f = float(np.mean(female_vals))
        denom = max(abs(pi_m), abs(pi_f), 1e-6)
        obs_sdq = abs(pi_m - pi_f) / denom

        all_vals = np.concatenate([male_vals, female_vals])
        n_m = len(male_vals)

        null_sdq = np.empty(int(n_perm), dtype=float)
        for i in range(int(n_perm)):
            perm = rng.permutation(all_vals)
            pm = float(np.mean(perm[:n_m]))
            pf = float(np.mean(perm[n_m:]))
            d = max(abs(pm), abs(pf), 1e-6)
            null_sdq[i] = abs(pm - pf) / d

        emp_p = float(np.mean(null_sdq >= obs_sdq))

        rows.append(
            {
                "Genotype": geno,
                "Treatment": treat,
                "PI_Male_mean": round(pi_m, 3),
                "PI_Female_mean": round(pi_f, 3),
                "n_Male": int(len(male_vals)),
                "n_Female": int(len(female_vals)),
                "SDQ": round(obs_sdq, 4),
                "p_permutation": round(emp_p, 6),
                "Sig": sig_label(emp_p),
                "n_perm": int(n_perm),
                "seed": int(seed),
                "Interpretation": (
                    "Strong significant dimorphism"
                    if obs_sdq > 0.5 and emp_p < 0.05
                    else "Moderate significant dimorphism"
                    if obs_sdq > 0.25 and emp_p < 0.05
                    else "No significant dimorphism"
                ),
            }
        )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility alias
# ──────────────────────────────────────────────────────────────────────────────
def run_statistical_tests(*args, **kwargs):
    """
    Backward-compatible alias for older DroPhenix code.
    """
    return auto_test(*args, **kwargs)