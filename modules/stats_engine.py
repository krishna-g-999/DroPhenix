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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
    Rank-biserial correlation for Mann-Whitney U (Kerby 2014; Wendt 1972).

        r = 2 * U1 / (n1 * n2) - 1

    where U1 is scipy's statistic for the FIRST sample, i.e. the number of
    (a, b) pairs with a > b (ties counting a half). r = +1 when every value in
    `a` exceeds every value in `b`, -1 in the opposite case, 0 at no effect.

    SIGN FIX (audit S3 finding 3.3): the previous formula was
    `1 - 2*U/(n1*n2)`, which returned -1 when group A dominated group B. In the
    same pairwise table Cohen's d for that comparison is +6.97, so the two
    effect-size columns pointed in opposite directions and any figure taking
    its direction from this column was drawn backwards.
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
    return float((2 * u_stat) / (n1 * n2) - 1)


def min_attainable_p(n1: int, n2: int) -> float:
    """
    The smallest two-sided Mann-Whitney U p-value a design with group sizes
    n1, n2 can produce, i.e. 2 / C(n1+n2, n1) for the fully separated case.

    For the standard Drosophila triplicate (3 vs 3) this is 0.100: no effect,
    however large, can reach the 0.05 level. Reporting a test whose rejection
    region is empty as "ns" is misleading, so `auto_test` surfaces this value
    (audit S1 finding 1.2).
    """
    from math import comb
    if n1 < 1 or n2 < 1:
        return np.nan
    return float(min(1.0, 2.0 / comb(n1 + n2, n1)))


def van_elteren_test(strata: Sequence[Tuple[Iterable[float], Iterable[float]]],
                     n_perm: int = 100000, seed: int = DEFAULT_SEED,
                     exact_limit: int = 400000) -> dict:
    """
    Stratified Wilcoxon rank-sum test (van Elteren 1960, Bull. Inst. Internat.
    Statist. 37:351; see also Lehmann 1975, Nonparametrics, sec. 3.3).

    Each stratum contributes its own Wilcoxon statistic, weighted by
    1/(N_i + 1) -- van Elteren's design-free weighting, which is optimal
    against a common location shift and gives small strata proportionately
    less say. The strata are never pooled into one ranking, so a difference in
    LEVEL between strata cannot masquerade as a treatment effect. This is the
    correct answer when a factor is known to matter but is not the factor under
    test -- sex in a Drosophila sleep experiment being the standard case: males
    and females differ several hundred minutes per day in total sleep, so
    ranking them together dilutes the contrast, while testing them separately
    throws away half the flies at an n where none can be spared.

    P-VALUE. Exact by complete enumeration of the within-stratum label
    assignments whenever the number of assignments is at most `exact_limit`;
    otherwise a within-stratum random permutation with the Phipson & Smyth
    (2010, Stat Appl Genet Mol Biol 9:39) (1 + c) / (1 + B) estimator, which is
    never zero. Two-sided as 2 * min(tail), matching the convention used by
    `permutation_test_means` and `scipy.stats.mannwhitneyu` in this module.

    strata : sequence of (control, treated) pairs, one per stratum.
    Returns the statistic, its standardised value, the p-value, and how it was
    obtained. Strata with fewer than one observation on either side are dropped
    and counted in `n_strata_dropped`.
    """
    import itertools
    from math import comb

    use = []
    dropped = 0
    for a, b in strata:
        a = _clean_array(a)
        b = _clean_array(b)
        if len(a) < 1 or len(b) < 1:
            dropped += 1
            continue
        use.append((a, b))
    if not use:
        return {"statistic": np.nan, "z": np.nan, "p": np.nan,
                "method": "no usable stratum", "n_strata": 0,
                "n_strata_dropped": dropped}

    # Ranks are a property of the DATA, not of the label assignment, so they are
    # computed once per stratum. Permuting labels only changes WHICH ranks are
    # summed. Re-ranking inside the enumeration loop is what made a 4000-run
    # simulation of this test intractable.
    ranks, weights, centres, sizes = [], [], [], []
    var = 0.0
    for a, b in use:
        r = stats.rankdata(np.concatenate([a, b]))
        n1, n2 = len(a), len(b)
        N = n1 + n2
        ranks.append(r)
        weights.append(1.0 / (N + 1.0))
        centres.append(n2 * (N + 1) / 2.0)
        sizes.append((n1, n2, N))
        if N > 1:
            var += (n1 * n2 / (N * (N - 1.0))
                    * (float((r ** 2).sum()) - N * (N + 1.0) ** 2 / 4.0)) / (N + 1.0) ** 2

    T_obs = float(sum(w * (float(r[n1:].sum()) - c)
                      for r, w, c, (n1, _, _) in zip(ranks, weights, centres, sizes)))
    z = float(T_obs / np.sqrt(var)) if var > 0 else np.nan

    n_assign = 1
    for (n1, n2, N) in sizes:
        n_assign *= comb(N, n2)
        if n_assign > exact_limit:
            break

    if n_assign <= exact_limit:
        # Per stratum, the complete set of weighted centred rank sums. The null
        # distribution of the total is their successive outer sum -- the same
        # enumeration, without materialising the cross product of index tuples.
        null = np.zeros(1)
        for r, w, c, (n1, n2, N) in zip(ranks, weights, centres, sizes):
            idx = np.fromiter(itertools.chain.from_iterable(
                itertools.combinations(range(N), n2)), dtype=int)
            vals = w * (r[idx].reshape(-1, n2).sum(axis=1) - c)
            null = (null[:, None] + vals[None, :]).ravel()
        # Ties in the data make exact equality unreliable in floating point.
        tol = 1e-9 * max(1.0, float(np.abs(null).max()))
        lo = float(np.mean(null <= T_obs + tol))
        hi = float(np.mean(null >= T_obs - tol))
        p = float(min(1.0, 2.0 * min(lo, hi)))
        method = f"exact ({len(null)} stratified assignments)"
    else:
        rng = np.random.default_rng(seed)
        null = np.zeros(n_perm)
        for r, w, c, (n1, n2, N) in zip(ranks, weights, centres, sizes):
            draw = np.argsort(rng.random((n_perm, N)), axis=1)[:, :n2]
            null += w * (r[draw].sum(axis=1) - c)
        tol = 1e-9 * max(1.0, float(np.abs(null).max()))
        c_lo = int(np.count_nonzero(null <= T_obs + tol))
        c_hi = int(np.count_nonzero(null >= T_obs - tol))
        p = float(min(1.0, 2.0 * min((1 + c_lo) / (1 + n_perm), (1 + c_hi) / (1 + n_perm))))
        method = f"permutation ({n_perm} draws, Phipson-Smyth)"

    return {"statistic": float(T_obs), "z": z, "p": p, "method": method,
            "n_strata": len(use), "n_strata_dropped": dropped,
            "n_per_stratum": [(len(a), len(b)) for a, b in use],
            "p_min_attainable": float(min(1.0, 2.0 / max(n_assign, 1)))}


def permutation_test_means(a: np.ndarray, b: np.ndarray,
                           n_perm: int = 20000, seed: int = DEFAULT_SEED) -> dict:
    """
    Two-sided permutation test on the difference in means (Fisher 1935;
    Ernst 2004, Statist. Sci. 19:676; Good 2005).

    TWO-SIDED CONVENTION. The p-value is 2 * min(P(T <= t_obs), P(T >= t_obs)),
    capped at 1, computed on the SIGNED statistic. This is the convention used
    by `scipy.stats.permutation_test(..., alternative='two-sided')` and by
    `scipy.stats.mannwhitneyu`, so the permutation p and the Mann-Whitney p this
    engine reports beside it are on the same footing, and the implementation is
    validated to machine precision against scipy's exact enumeration in
    validation/validate_stats.py.

    (The other published convention, P(|T| >= |t_obs|), is equally valid but can
    differ by up to a factor of two when the null is asymmetric — which it is
    whenever n1 != n2, because complementing a partition then changes the group
    sizes and the partition set is not closed under negation. Mixing the two
    conventions inside one results table would be misleading, so this module
    uses one throughout.)

    All distinct assignments are enumerated exactly when there are few enough,
    giving an exact p. Otherwise a seeded Monte-Carlo permutation is used with
    the Phipson & Smyth (2010, Stat. Appl. Genet. Mol. Biol. 9:39)
    (1 + count) / (1 + B) estimator per tail, which never returns an
    unattainable p of exactly 0.

    THIS DOES NOT RESCUE A SMALL DESIGN. At 3 vs 3 the smallest attainable
    two-sided p is 2/C(6,3) = 0.100, identical to Mann-Whitney. The floor is a
    property of the DESIGN, not of the test. Use `min_attainable_p` to report it
    rather than implying a test choice fixes it.
    """
    from itertools import combinations as _combs
    from math import comb

    a = _clean_array(a)
    b = _clean_array(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"diff": np.nan, "p_value": np.nan, "n_perm": 0, "exact": False}

    pool = np.concatenate([a, b])
    obs = float(np.mean(a) - np.mean(b))       # SIGNED
    total = comb(n1 + n2, n1)
    tol = 1e-12 * max(1.0, abs(obs))

    if total <= n_perm:                        # enumerate exactly
        n_le = n_ge = 0
        for sel in _combs(range(n1 + n2), n1):
            m = np.zeros(n1 + n2, dtype=bool)
            m[list(sel)] = True
            d = float(pool[m].mean() - pool[~m].mean())
            if d <= obs + tol:
                n_le += 1
            if d >= obs - tol:
                n_ge += 1
        p = min(1.0, 2.0 * min(n_le, n_ge) / total)
        return {"diff": obs, "p_value": float(p),
                "n_perm": int(total), "exact": True}

    rng = np.random.default_rng(seed)
    n_le = n_ge = 0
    B = int(n_perm)
    for _ in range(B):
        q = rng.permutation(pool)
        d = float(q[:n1].mean() - q[n1:].mean())
        if d <= obs + tol:
            n_le += 1
        if d >= obs - tol:
            n_ge += 1
    p = min(1.0, 2.0 * min((1 + n_le) / (1 + B), (1 + n_ge) / (1 + B)))
    return {"diff": obs, "p_value": float(p), "n_perm": B, "exact": False}


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
                # Not applicable: the t statistic is continuous, so there is no
                # combinatorial floor on its p-value. Keys are present so the
                # returned dict has a stable shape across both branches.
                "p_min_attainable": np.nan,
                "underpowered_by_design": False,
                "note": "",
            },
            "pairwise": pairwise,
        }

    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        stat, p = np.nan, np.nan

    r = rank_biserial_r(a, b)

    # ── Underpowered-by-construction guard (audit S1 finding 1.2) ────────────
    # The two-sided Mann-Whitney p cannot go below 2 / C(n1+n2, n1). At the
    # standard Drosophila triplicate (3 vs 3) that floor is 0.100, so the test
    # has an EMPTY rejection region at alpha = 0.05: perfectly separated data
    # returns "ns". Reporting that as a non-significant result is misleading.
    # When the rank test cannot reach alpha, a permutation test on the
    # difference in means is reported as the primary result instead, and the
    # substitution is stated in `Method`.
    n1, n2 = len(a), len(b)
    p_min = min_attainable_p(n1, n2)
    underpowered = pd.notna(p_min) and p_min > 0.05

    method = "Mann-Whitney U"
    p_primary, stat_primary = p, stat
    note = ""
    perm_p = np.nan
    if underpowered:
        # A permutation test on the difference in means is ALSO reported, but it
        # is not a rescue: its two-sided floor is the same 2/C(n1+n2,n1),
        # because |mean(a)-mean(b)| is label-exchange invariant. Substituting it
        # and calling the result significant would be a fabricated finding.
        perm = permutation_test_means(a, b)
        perm_p = perm["p_value"]
        note = (f"DESIGN IS UNDERPOWERED BY CONSTRUCTION at n1={n1}, n2={n2}: the "
                f"smallest two-sided p ANY exact test can attain is {p_min:.3f}, "
                f"so no effect however large can reach alpha = 0.05. This is a "
                f"property of the sample size, not of the test — a permutation "
                f"test on the means gives p = {perm_p:.3f} on the same data. "
                f"Report the effect size and its interval; to test at 0.05 you "
                f"need at least 4 vs 4 (floor 0.029) or 5 vs 5 (floor 0.008).")

    pairwise = pd.DataFrame(
        [
            {
                "Group A": label_a,
                "Group B": label_b,
                "Median A": round(_safe_median(a), 3),
                "Median B": round(_safe_median(b), 3),
                "Statistic": round(float(stat_primary), 4) if pd.notna(stat_primary) else np.nan,
                "p_raw": round(float(p_primary), 6) if pd.notna(p_primary) else np.nan,
                "p_adj_BH": round(float(p_primary), 6) if pd.notna(p_primary) else np.nan,
                "Sig": sig_label(p_primary),
                "Rank_biserial_r": round(r, 3) if pd.notna(r) else np.nan,
                "Effect_size": _rbr_label(r),
                "p_permutation_means": round(float(perm_p), 6) if pd.notna(perm_p) else np.nan,
                "p_min_attainable": round(float(p_min), 4) if pd.notna(p_min) else np.nan,
                "Underpowered_by_design": bool(underpowered),
                "Method": method,
            }
        ]
    )

    return {
        "method": method,
        "main_result": {
            "Statistic": round(float(stat_primary), 4) if pd.notna(stat_primary) else np.nan,
            "p_value": round(float(p_primary), 6) if pd.notna(p_primary) else np.nan,
            # "ns" is not an honest verdict when the rejection region is empty.
            "Significant": ("underpowered by design" if underpowered
                            else sig_label(p_primary)),
            "p_min_attainable": round(float(p_min), 4) if pd.notna(p_min) else np.nan,
            "underpowered_by_design": bool(underpowered),
            "note": note,
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
def auto_test(data: Dict[str, np.ndarray], method: str = "rank") -> dict:
    """
    Statistical test engine for dict[str, array-like].

    THE TEST IS CHOSEN A PRIORI, NOT FROM THE DATA (audit S3 finding 3.4)
    ---------------------------------------------------------------------
    Default is the RANK test throughout:
      - 2 groups  : Mann-Whitney U
      - >2 groups : Kruskal-Wallis + pairwise Mann-Whitney U with BH-FDR

    WHAT WAS ACTUALLY MEASURED, on PI generated the way real vials generate it
    (Count ~ Binomial(10 flies, p), PI = 2*Count/n*100 - 100):

    1. THE DECISION IS IRREPRODUCIBLE. Run the SAME design on data from the SAME
       distribution and the old router picks the parametric branch:

           p = 0.50, 6 vials/group  -> 72.8% of the time
           p = 0.80, 6 vials/group  -> 60.8%
           p = 0.80, 10 vials/group -> 46.6%   <- a coin flip
           p = 0.95, 10 vials/group ->  0.1%

       Two labs running the same experiment, or the same lab running it twice,
       get different TESTS. That is the concrete problem, and it is a
       reproducibility problem before it is a type-I problem.

    2. Theory says the two-stage procedure does not preserve the nominal alpha
       of the test it selects, because that test's null is conditioned on the
       pre-test having passed (Rochon, Gondan & Kieser 2012, BMC Med Res
       Methodol 12:81; Zimmerman 2004, Br J Math Stat Psychol 57:173). BE
       HONEST ABOUT THE SIZE OF THIS: in the designs above the measured
       inflation is small (true-null rejection 0.0537 for 'auto' vs 0.0480 for
       'rank' at p=0.5, 10 vials; all three policies are at or below 0.05 under
       a 10%-bad-vial contamination). The inflation is not what justifies the
       change here — point 1 is.

    3. THE A PRIORI RANK TEST COSTS ALMOST NO POWER, so there is no trade-off
       to weigh against reproducibility. Power at a real effect:

           p 0.80 vs 0.50,  6 vials : rank 0.852   parametric 0.888
           p 0.80 vs 0.50, 10 vials : rank 0.989   parametric 0.993
           p 0.90 vs 0.60,  6 vials : rank 0.923   parametric 0.936
           p 0.95 vs 0.70, 10 vials : rank 0.996   parametric 0.998

    4. The primary metric is a rescaled BINOMIAL PROPORTION: bounded on
       [-100, +100], discrete in steps of 200/n, with variance tied to its mean.
       Gaussian assumptions are wrong by construction, most visibly near the
       ceiling where a healthy climbing cohort actually sits. A rank test makes
       no such assumption, which is why it is the a priori choice rather than
       merely the fallback.

    Shapiro-Wilk and Levene are still COMPUTED AND REPORTED, because they are
    informative diagnostics. They no longer decide anything.

    method
    ------
    "rank"       (default) rank tests, chosen a priori. Recommended.
    "parametric" Welch t-test / one-way ANOVA + pairwise Welch, chosen a priori
                 by the analyst. Legitimate if declared in advance.
    "auto"       DEPRECATED legacy behaviour: pre-test normality and branch on
                 it. Emits a warning; retained only to reproduce pre-2026-09-07
                 outputs. Do not use for new analyses.
    """
    import warnings

    if method not in ("rank", "parametric", "auto"):
        raise ValueError(f"method must be 'rank', 'parametric' or 'auto'; got {method!r}")

    if len(data) < 2:
        return {
            "method": "N/A",
            "normality": pd.DataFrame(),
            "levene": {},
            "main_result": {},
            "pairwise": pd.DataFrame(),
            "all_normal": False,
            "equal_variance": False,
            "test_choice": "n/a",
        }

    cleaned = {k: _clean_array(v) for k, v in data.items()}
    normality_df = test_normality(cleaned)          # reported, not routed on
    levene_res = test_levene(cleaned)

    valid_ns = normality_df.set_index("Group")["n"].to_dict() if not normality_df.empty else {}
    all_normal = bool(normality_df["Normal"].all()) if len(normality_df) else False
    all_n_ge_5 = all(n >= 5 for n in valid_ns.values()) if valid_ns else False
    equal_var = levene_res.get("Equal_variance", "No") == "Yes"

    if method == "auto":
        warnings.warn(
            "auto_test(method='auto') pre-tests normality and branches on the "
            "result. That two-stage procedure does not preserve the nominal "
            "alpha of the test it selects, and at n = 3-6 the pre-test is "
            "decided by noise. Use method='rank' (default) or declare "
            "method='parametric' in advance.",
            RuntimeWarning, stacklevel=2,
        )
        parametric = all_normal and all_n_ge_5 and (equal_var or len(cleaned) == 2)
        choice = "data-dependent (deprecated)"
    else:
        parametric = (method == "parametric")
        choice = f"a priori ({method})"

    group_names = list(cleaned.keys())
    arrays = [cleaned[g] for g in group_names]

    if len(cleaned) == 2:
        res = _two_group_test(arrays[0], arrays[1], group_names[0], group_names[1], parametric)
        return {
            "method": res["method"],
            "normality": normality_df,
            "levene": levene_res,
            "main_result": res["main_result"],
            "pairwise": res["pairwise"],
            "all_normal": all_normal,
            "equal_variance": equal_var,
            "test_choice": choice,
        }

    if parametric:
        main_result, pairwise_df = anova_tukey(cleaned)
        name = "One-way ANOVA + pairwise Welch t-tests (BH-FDR)"
    else:
        main_result, pairwise_df = kruskal_dunn(cleaned)
        name = "Kruskal-Wallis + pairwise Mann-Whitney U (BH-FDR)"

    return {
        "method": name,
        "normality": normality_df,
        "levene": levene_res,
        "main_result": main_result,
        "pairwise": pairwise_df,
        "all_normal": all_normal,
        "equal_variance": equal_var,
        "test_choice": choice,
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
    import warnings

    rng = np.random.default_rng(seed)
    v = _clean_array(values)

    if len(v) == 0:
        return np.nan, np.nan, np.nan

    # Audit S3 finding 3.10: a percentile bootstrap resamples only the observed
    # values, so with n = 3 replicates the interval cannot be wider than the
    # observed range and undercovers badly — the resampling distribution has at
    # most C(2n-1, n) = 10 distinct means. It is still computed (the point
    # estimate is fine) but the caller is told not to read the interval as a
    # calibrated 95% CI.
    if len(v) < 5:
        warnings.warn(
            f"bootstrap_ci: n = {len(v)}. A percentile bootstrap at this sample "
            f"size cannot exceed the observed range and does not achieve nominal "
            f"{ci:.0%} coverage. Report the interval as descriptive, or use an "
            f"exact/analytic interval instead.",
            RuntimeWarning, stacklevel=2,
        )

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
        n_m, n_f = len(male_vals), len(female_vals)

        # SDQ is reported as a descriptive effect size, on the [0, 1] scale
        # normalisation (see metrics.compute_sdq): |PI_M - PI_F| / 200.
        obs_sdq = abs(pi_m - pi_f) / 200.0

        # ── The test statistic is the DIFFERENCE IN MEANS, not SDQ ───────────
        # (audit S1 finding 1.3). The old ratio statistic
        # |PI_M - PI_F| / max(|PI_M|, |PI_F|) saturates at its ceiling of 2
        # whenever the two means have opposite signs, which under permutation of
        # a bimodal +/- pool happens almost every time. Six males at PI +92 vs
        # six females at -92 — total dimorphism — therefore returned p = 0.95,
        # because the observed value merely tied a null that was pinned at the
        # ceiling. On the same data the difference in means gives p = 0.0024.
        if n_m < 2 or n_f < 2:
            rows.append({
                "Genotype": geno, "Treatment": treat,
                "PI_Male_mean": round(pi_m, 3), "PI_Female_mean": round(pi_f, 3),
                "n_Male": n_m, "n_Female": n_f,
                "SDQ": round(obs_sdq, 4),
                "p_permutation": np.nan, "Sig": "n/a",
                "n_perm": 0, "seed": int(seed),
                "test_statistic": "difference in means",
                "Interpretation": (
                    f"NOT TESTABLE: {n_m} male and {n_f} female value(s). A "
                    f"permutation test needs replicate-level PI (>=2 per sex); "
                    f"a summary table with one row per sex gives a 2-element "
                    f"pool and no usable null. Supply the replicate-level "
                    f"'PI' column rather than 'PI_mean'."),
            })
            continue

        res = permutation_test_means(male_vals, female_vals,
                                     n_perm=int(n_perm), seed=int(seed))
        emp_p = res["p_value"]

        rows.append(
            {
                "Genotype": geno,
                "Treatment": treat,
                "PI_Male_mean": round(pi_m, 3),
                "PI_Female_mean": round(pi_f, 3),
                "n_Male": n_m,
                "n_Female": n_f,
                "SDQ": round(obs_sdq, 4),
                "Delta_PI": round(pi_m - pi_f, 3),
                # Never 0: exact enumeration bottoms out at 1/C(n,k), and the
                # Monte-Carlo estimator at 1/(B+1) (Phipson & Smyth 2010).
                "p_permutation": float(f"{emp_p:.3g}") if pd.notna(emp_p) else np.nan,
                "Sig": sig_label(emp_p),
                "n_perm": int(res["n_perm"]),
                "p_exact": bool(res["exact"]),
                "test_statistic": "difference in means",
                "seed": int(seed),
                "Interpretation": (
                    "Strong significant dimorphism"
                    if obs_sdq > 0.25 and pd.notna(emp_p) and emp_p < 0.05
                    else "Moderate significant dimorphism"
                    if obs_sdq > 0.10 and pd.notna(emp_p) and emp_p < 0.05
                    else "Significant but small dimorphism"
                    if pd.notna(emp_p) and emp_p < 0.05
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