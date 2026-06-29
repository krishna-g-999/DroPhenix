"""
DroPhenix — survival.py
Actuarial survival-analysis engine for Drosophila lifespan assays.

Scientific basis
----------------
Lifespan (longevity) assays produce right-censored time-to-death data: a fly
either dies on a known day (event) or is alive when scoring ends / is lost
(censored). Reporting only the median is statistically inadequate — the field
standard for lifespan papers (e.g. fly neurodegeneration models of ALS, HD, PD)
is:

1. Kaplan-Meier (1958) product-limit survival estimate  S(t)
2. Greenwood (1926) variance -> pointwise confidence intervals
   (complementary log-log transform keeps CIs within [0, 1]).
3. Log-rank test (Mantel-Cox) for pairwise group comparison, with
   Benjamini-Hochberg FDR correction across the family of pairwise tests.
4. Cox (1972) proportional-hazards regression for drug x genotype effects,
   reporting hazard ratios with 95% CIs.
5. Restricted Mean Survival Time (RMST) — a model-free summary that remains
   valid when hazards are non-proportional.

All heavy lifting uses statsmodels (SurvfuncRight, survdiff, PHReg) so the
numerical results match an established, peer-reviewed implementation. This
module adds the Drosophila-specific data model, pairwise orchestration,
FDR correction, and reporting tables.

Input data model
-----------------
`to_event_table()` accepts either of two layouts and returns a canonical
event-level table (one row per fly):

  (A) Event-level   : Genotype, Treatment, Sex, Time, Event
                      Event = 1 (death) or 0 (censored/alive at end).
  (B) Daily cohort  : Genotype, Treatment, Sex, Day, Dead_count,
                      [Censored_count], Total_start
                      -> expanded so each death contributes a duration = Day,
                         each censored fly a censored duration = Day,
                         and any flies still alive after the last scored day
                         are censored at that last day.

Canonical output columns: Group, Genotype, Treatment, Sex, Time, Event.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels provides the peer-reviewed numerical core.
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.duration.hazard_regression import PHReg


GROUP_KEYS = ["Genotype", "Treatment", "Sex"]
MIN_GROUP_EVENTS = 3  # below this, group tests are flagged low-confidence


# ──────────────────────────────────────────────────────────────────────────────
# Significance label (kept consistent with stats_engine.sig_label)
# ──────────────────────────────────────────────────────────────────────────────
def _sig(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _bh_correction(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. NaNs left as NaN. (Mirrors stats_engine.)"""
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
# Data normalisation
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map flexible user column names onto canonical names (case-insensitive)."""
    lut = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "Genotype": ["genotype", "strain", "line"],
        "Treatment": ["treatment", "drug", "compound", "condition"],
        "Sex": ["sex"],
        "Time": ["time", "day", "days", "age", "time_days", "lifespan", "duration"],
        "Event": ["event", "status", "dead", "death", "died"],
        "Day": ["day", "days", "time"],
        "Dead_count": ["dead_count", "dead", "deaths", "new_deaths", "n_dead"],
        "Censored_count": ["censored_count", "censored", "n_censored", "lost"],
        "Total_start": ["total_start", "total", "n_start", "n_total", "cohort", "starting_n"],
    }
    found = {}
    for canon, opts in aliases.items():
        for o in opts:
            if o in lut:
                found[canon] = lut[o]
                break
    return found


def to_event_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise either supported layout to a canonical event-level table.

    Returns columns: Group, Genotype, Treatment, Sex, Time, Event
      Time  : duration in days (float)
      Event : 1 = death observed, 0 = right-censored
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Group", *GROUP_KEYS, "Time", "Event"])

    col = _resolve_columns(df)
    for k in ("Genotype", "Treatment", "Sex"):
        if k not in col:
            # allow missing Sex/Treatment by filling a constant
            if k == "Sex":
                df = df.copy(); df["Sex"] = "Pooled"; col["Sex"] = "Sex"
            elif k == "Treatment":
                df = df.copy(); df["Treatment"] = "All"; col["Treatment"] = "Treatment"
            else:
                raise ValueError(f"Required column '{k}' not found in lifespan data.")

    g, t, s = col["Genotype"], col["Treatment"], col["Sex"]

    # ── Layout A: event-level ────────────────────────────────────────────────
    if "Time" in col and "Event" in col:
        out = pd.DataFrame({
            "Genotype": df[g].astype(str),
            "Treatment": df[t].astype(str),
            "Sex": df[s].astype(str),
            "Time": pd.to_numeric(df[col["Time"]], errors="coerce"),
            "Event": pd.to_numeric(df[col["Event"]], errors="coerce"),
        })
        # Coerce event to {0,1}; treat anything >=1 as event.
        out["Event"] = (out["Event"] >= 1).astype(int)
        out = out.dropna(subset=["Time"])

    # ── Layout B: daily cohort counts ────────────────────────────────────────
    elif "Day" in col and "Dead_count" in col and "Total_start" in col:
        rows = []
        day, dead, total = col["Day"], col["Dead_count"], col["Total_start"]
        cens = col.get("Censored_count")
        for (gg, tt, ss), grp in df.groupby([g, t, s]):
            grp = grp.sort_values(day)
            start_n = int(pd.to_numeric(grp[total], errors="coerce").iloc[0])
            last_day = float(pd.to_numeric(grp[day], errors="coerce").max())
            n_dead_total = 0
            for _, r in grp.iterrows():
                d = float(r[day])
                nd = int(pd.to_numeric(r[dead], errors="coerce") or 0)
                n_dead_total += nd
                rows += [(gg, tt, ss, d, 1)] * nd
                if cens is not None:
                    nc = int(pd.to_numeric(r[cens], errors="coerce") or 0)
                    rows += [(gg, tt, ss, d, 0)] * nc
            # Any flies never recorded as dead -> censored at last scored day.
            alive = start_n - n_dead_total
            if alive > 0:
                rows += [(gg, tt, ss, last_day, 0)] * alive
        out = pd.DataFrame(rows, columns=["Genotype", "Treatment", "Sex", "Time", "Event"])
    else:
        raise ValueError(
            "Unrecognised lifespan format. Provide either event-level columns "
            "(Genotype, Treatment, Sex, Time, Event) or daily-cohort columns "
            "(Genotype, Treatment, Sex, Day, Dead_count, Total_start)."
        )

    out = out[out["Time"] >= 0].reset_index(drop=True)
    out["Group"] = out["Genotype"] + " | " + out["Treatment"] + " | " + out["Sex"]
    return out[["Group", *GROUP_KEYS, "Time", "Event"]]


# ──────────────────────────────────────────────────────────────────────────────
# Kaplan-Meier with Greenwood confidence intervals
# ──────────────────────────────────────────────────────────────────────────────
def kaplan_meier(event_df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, pd.DataFrame]:
    """
    Per-group Kaplan-Meier survival curve with Greenwood standard errors and
    complementary log-log (cloglog) confidence intervals.

    Returns {group_label: DataFrame[Time, n_risk, n_event, Survival,
                                     Survival_SE, CI_lower, CI_upper]}.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    curves: Dict[str, pd.DataFrame] = {}

    for grp, sub in event_df.groupby("Group"):
        time = sub["Time"].to_numpy(dtype=float)
        status = sub["Event"].to_numpy(dtype=int)
        if len(time) == 0:
            continue
        sf = SurvfuncRight(time, status)
        S = np.asarray(sf.surv_prob, dtype=float)
        se = np.asarray(sf.surv_prob_se, dtype=float)
        times = np.asarray(sf.surv_times, dtype=float)

        # cloglog transform CI (Kalbfleisch-Prentice): keeps bounds in [0,1].
        # CI = S(t) ** exp(±z * se / (S * |log S|)).  Using |log S| keeps theta
        # positive so that lower = S**exp(+theta) < S < S**exp(-theta) = upper.
        with np.errstate(divide="ignore", invalid="ignore"):
            logS = np.log(S)
            theta = z * se / (S * np.abs(logS))  # SE on log(-log S) scale (delta method)
            lower = S ** np.exp(theta)
            upper = S ** np.exp(-theta)
        # boundary handling (S=1 -> CI [1,1]; S=0 -> [0,0])
        lower = np.where(S >= 1.0, 1.0, lower)
        upper = np.where(S >= 1.0, 1.0, upper)
        lower = np.clip(np.nan_to_num(lower, nan=0.0), 0.0, 1.0)
        upper = np.clip(np.nan_to_num(upper, nan=0.0), 0.0, 1.0)

        n_risk = np.asarray(sf.n_risk, dtype=float)
        n_event = np.asarray(sf.n_events, dtype=float) if hasattr(sf, "n_events") else np.full_like(times, np.nan)

        curves[grp] = pd.DataFrame({
            "Time": times,
            "n_risk": n_risk,
            "n_event": n_event,
            "Survival": np.round(S, 4),
            "Survival_SE": np.round(se, 4),
            "CI_lower": np.round(lower, 4),
            "CI_upper": np.round(upper, 4),
        })
    return curves


def _km_median(sf: SurvfuncRight) -> float:
    """
    Median survival = smallest event time t with S(t) <= 0.5.

    This is the convention used by R's survival::survfit, GraphPad Prism, and
    OASIS — the tools Drosophila labs cross-check against. (statsmodels'
    SurvfuncRight.quantile uses a strict S(t) < 0.5, which can differ by one
    step when the curve lands exactly on 0.5; we deliberately do not use it.)
    Returns NaN if the curve never reaches 0.5 (median not reached).
    """
    t = np.asarray(sf.surv_times, dtype=float)
    S = np.asarray(sf.surv_prob, dtype=float)
    idx = np.where(S <= 0.5 + 1e-12)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def median_survival_table(event_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Per-group median survival with Brookmeyer-Crowley confidence interval
    (from the KM curve), plus n, events, and censored counts.

    Median uses the R/Prism convention (smallest t with S(t) <= 0.5); see
    _km_median. The CI uses statsmodels' Brookmeyer-Crowley quantile_ci.
    """
    rows = []
    for grp, sub in event_df.groupby("Group"):
        time = sub["Time"].to_numpy(dtype=float)
        status = sub["Event"].to_numpy(dtype=int)
        n = len(time)
        n_event = int(status.sum())
        n_cens = int(n - n_event)
        meta = sub.iloc[0]
        try:
            sf = SurvfuncRight(time, status)
            med = _km_median(sf)
            try:
                # statsmodels' quantile_ci does a cloglog root-find that emits
                # benign divide-by-zero warnings at S=0/1; silence them.
                with np.errstate(divide="ignore", invalid="ignore"):
                    lo, hi = sf.quantile_ci(0.5, alpha=alpha)
                lo = float(lo); hi = float(hi)
                # A non-finite bound means the CI limit is "not reached" within
                # follow-up (e.g. upper bound when the curve never drops far
                # enough); represent it as NaN rather than inf.
                if not np.isfinite(lo): lo = np.nan
                if not np.isfinite(hi): hi = np.nan
            except Exception:
                lo, hi = np.nan, np.nan
        except Exception:
            med, lo, hi = np.nan, np.nan, np.nan
        rows.append({
            "Group": grp,
            "Genotype": meta["Genotype"],
            "Treatment": meta["Treatment"],
            "Sex": meta["Sex"],
            "n": n,
            "Events": n_event,
            "Censored": n_cens,
            "Median_Survival": round(med, 2) if pd.notna(med) else np.nan,
            "Median_CI_lower": round(lo, 2) if pd.notna(lo) else np.nan,
            "Median_CI_upper": round(hi, 2) if pd.notna(hi) else np.nan,
            "Max_observed": round(float(np.max(time)), 2) if n else np.nan,
            "low_confidence": n_event < MIN_GROUP_EVENTS,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Restricted Mean Survival Time (RMST)
# ──────────────────────────────────────────────────────────────────────────────
def rmst_table(event_df: pd.DataFrame, tau: Optional[float] = None) -> pd.DataFrame:
    """
    Restricted Mean Survival Time = area under the KM curve up to time tau.
    Model-free; robust to non-proportional hazards. If tau is None it is set to
    the minimum across groups of each group's maximum observed time (so every
    group's curve is defined over [0, tau]).
    """
    groups = list(event_df.groupby("Group"))
    if not groups:
        return pd.DataFrame()
    if tau is None:
        tau = min(float(sub["Time"].max()) for _, sub in groups)

    rows = []
    for grp, sub in groups:
        time = sub["Time"].to_numpy(dtype=float)
        status = sub["Event"].to_numpy(dtype=int)
        sf = SurvfuncRight(time, status)
        t = np.asarray(sf.surv_times, dtype=float)
        S = np.asarray(sf.surv_prob, dtype=float)
        # Step-function integral of S(t) on [0, tau]; S=1 before first event time.
        grid_t = np.concatenate(([0.0], t[t <= tau], [tau]))
        grid_S = np.concatenate(([1.0], S[t <= tau], [S[t <= tau][-1] if np.any(t <= tau) else 1.0]))
        dt = np.diff(grid_t)
        rmst = float(np.sum(grid_S[:-1] * dt))
        meta = sub.iloc[0]
        rows.append({
            "Group": grp,
            "Genotype": meta["Genotype"], "Treatment": meta["Treatment"], "Sex": meta["Sex"],
            "tau": round(float(tau), 2),
            "RMST": round(rmst, 3),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Log-rank test (pairwise, BH-FDR corrected) + overall
# ──────────────────────────────────────────────────────────────────────────────
def logrank_overall(event_df: pd.DataFrame, weight_type: Optional[str] = None) -> dict:
    """
    Omnibus log-rank (Mantel-Cox) across all groups. weight_type=None gives the
    standard log-rank; 'gb' = Gehan-Breslow (Wilcoxon), 'tw' = Tarone-Ware.
    """
    grp = event_df["Group"].to_numpy()
    if len(np.unique(grp)) < 2:
        return {"chisq": np.nan, "p_value": np.nan, "df": 0, "Sig": "n/a"}
    chisq, p = survdiff(event_df["Time"].to_numpy(float),
                        event_df["Event"].to_numpy(int),
                        grp, weight_type=weight_type)
    return {
        "chisq": round(float(chisq), 4),
        "p_value": round(float(p), 6),
        "df": int(len(np.unique(grp)) - 1),
        "Sig": _sig(p),
        "test": "Log-rank" if weight_type is None else f"Weighted log-rank ({weight_type})",
    }


def logrank_pairwise(event_df: pd.DataFrame, weight_type: Optional[str] = None) -> pd.DataFrame:
    """
    All pairwise log-rank tests with Benjamini-Hochberg FDR correction.
    """
    groups = sorted(event_df["Group"].unique())
    if len(groups) < 2:
        return pd.DataFrame()

    meta, raw_p = [], []
    for ga, gb in combinations(groups, 2):
        sub = event_df[event_df["Group"].isin([ga, gb])]
        try:
            chisq, p = survdiff(sub["Time"].to_numpy(float),
                                sub["Event"].to_numpy(int),
                                sub["Group"].to_numpy(), weight_type=weight_type)
        except Exception:
            chisq, p = np.nan, np.nan
        na = int(event_df[event_df["Group"] == ga]["Event"].sum())
        nb = int(event_df[event_df["Group"] == gb]["Event"].sum())
        meta.append((ga, gb, chisq, p, na, nb))
        raw_p.append(p if pd.notna(p) else np.nan)

    adj = _bh_correction(np.array(raw_p, dtype=float))
    rows = []
    for (ga, gb, chisq, p, na, nb), padj in zip(meta, adj):
        rows.append({
            "Group A": ga, "Group B": gb,
            "Chisq": round(float(chisq), 4) if pd.notna(chisq) else np.nan,
            "p_raw": round(float(p), 6) if pd.notna(p) else np.nan,
            "p_adj_BH": round(float(padj), 6) if pd.notna(padj) else np.nan,
            "Sig": _sig(padj),
            "Events A": na, "Events B": nb,
            "Method": "Log-rank + BH-FDR" if weight_type is None
                      else f"Weighted log-rank ({weight_type}) + BH-FDR",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Cox proportional-hazards regression
# ──────────────────────────────────────────────────────────────────────────────
def cox_ph(event_df: pd.DataFrame,
           covariates: Optional[List[str]] = None,
           interaction: bool = True,
           ref: Optional[Dict[str, str]] = None) -> dict:
    """
    Cox proportional-hazards model for drug x genotype effects.

    covariates : subset of ['Genotype','Treatment','Sex'] to model
                 (default: those with >1 level).
    interaction: include Genotype:Treatment interaction when both are present.
    ref        : optional reference level per covariate (the baseline category).

    Returns dict with:
      summary  : DataFrame[covariate, coef(log HR), HR, CI_lower, CI_upper, p, Sig]
      n, n_events, loglik, concordance (Harrell's c), formula
    """
    df = event_df.copy()
    if covariates is None:
        covariates = [c for c in GROUP_KEYS if df[c].nunique() > 1]
    if not covariates:
        return {"error": "No covariate has more than one level; Cox model not identifiable."}

    # Set reference categories. We reorder each covariate's categories so the
    # chosen baseline is first and use a plain C(col) term — patsy then uses the
    # first level as the reference. (Avoids the C(col, Treatment(reference=...))
    # form, whose 'Treatment' contrast name collides with a data column named
    # 'Treatment' and triggers a patsy "Series object is not callable" error.)
    ref = ref or {}
    terms = []
    for c in covariates:
        levels = sorted(df[c].dropna().astype(str).unique())
        df[c] = df[c].astype(str)
        base = ref.get(c)
        if base is not None and base in levels:
            levels = [base] + [x for x in levels if x != base]
        df[c] = pd.Categorical(df[c], categories=levels)
        terms.append(f"C({c})")
    rhs = " + ".join(terms)
    if interaction and "Genotype" in covariates and "Treatment" in covariates:
        gi = terms[covariates.index("Genotype")]
        ti = terms[covariates.index("Treatment")]
        rhs += f" + {gi}:{ti}"
    formula = f"Time ~ {rhs}"

    model = PHReg.from_formula(formula, data=df, status=df["Event"].to_numpy(int))
    res = model.fit()

    params = res.params
    ci = res.conf_int()
    pvals = res.pvalues
    names = res.model.exog_names

    rows = []
    for i, name in enumerate(names):
        coef = float(params[i])
        lo, hi = float(ci[i, 0]), float(ci[i, 1])
        p = float(pvals[i])
        rows.append({
            "Covariate": _pretty_term(name),
            "log_HR": round(coef, 4),
            "HR": round(float(np.exp(coef)), 4),
            "HR_CI_lower": round(float(np.exp(lo)), 4),
            "HR_CI_upper": round(float(np.exp(hi)), 4),
            "p_value": round(p, 6),
            "Sig": _sig(p),
        })

    # Harrell's concordance (c-index)
    try:
        c_index = float(_concordance(df["Time"].to_numpy(float),
                                     df["Event"].to_numpy(int),
                                     res.predict().predicted_values))
    except Exception:
        c_index = np.nan

    return {
        "summary": pd.DataFrame(rows),
        "formula": formula,
        "n": int(len(df)),
        "n_events": int(df["Event"].sum()),
        "loglik": round(float(res.llf), 3) if hasattr(res, "llf") else np.nan,
        "concordance": round(c_index, 4) if pd.notna(c_index) else np.nan,
    }


def _pretty_term(name: str) -> str:
    """
    Turn patsy/statsmodels term names into readable labels, e.g.
      "C(Treatment)[T.Drug]"                              -> "Treatment: Drug"
      "C(Genotype)[T.TDP43]:C(Treatment)[T.Drug]"         -> "Genotype: TDP43 x Treatment: Drug"
      "C(Treatment, Treatment(reference='Veh'))[T.Drug]"  -> "Treatment: Drug"
    """
    import re

    def _one(term: str) -> str:
        m = re.search(r"C\(\s*(\w+).*?\)\[T\.(.+?)\]", term)
        if m:
            return f"{m.group(1)}: {m.group(2)}"
        return term.strip()

    parts = name.split(":")
    return " x ".join(_one(p) for p in parts)


def _concordance(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """
    Harrell's c-index. Higher risk score should accompany shorter survival.
    Concordant if, for a comparable pair, the one that died first had higher risk.
    O(n^2) — fine for fly-cohort sizes.
    """
    n = len(time)
    conc = disc = tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Determine which (if any) clearly died first and is comparable.
            if time[i] == time[j]:
                continue
            if time[i] < time[j]:
                if event[i] != 1:
                    continue  # i censored before j -> not comparable
                first, second = i, j
            else:
                if event[j] != 1:
                    continue
                first, second = j, i
            if risk[first] > risk[second]:
                conc += 1
            elif risk[first] < risk[second]:
                disc += 1
            else:
                tied += 1
    denom = conc + disc + tied
    return (conc + 0.5 * tied) / denom if denom else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# One-call orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_survival_analysis(df: pd.DataFrame,
                          weight_type: Optional[str] = None,
                          do_cox: bool = True) -> dict:
    """
    Full actuarial pipeline from raw lifespan input to report-ready tables.
    """
    events = to_event_table(df)
    result = {
        "event_table": events,
        "km_curves": kaplan_meier(events),
        "median_table": median_survival_table(events),
        "rmst_table": rmst_table(events),
        "logrank_overall": logrank_overall(events, weight_type=weight_type),
        "logrank_pairwise": logrank_pairwise(events, weight_type=weight_type),
    }
    if do_cox and events["Group"].nunique() >= 2:
        try:
            result["cox"] = cox_ph(events)
        except Exception as e:
            result["cox"] = {"error": str(e)}
    return result
