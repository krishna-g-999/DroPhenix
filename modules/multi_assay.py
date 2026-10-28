"""
DroPhenix — multi_assay.py
Multi-Assay Composite Motor Score (MCMS) integration module.

Integrates:
  1. Climbing assay PI (primary, required)
  2. Survival / lifespan median (optional)
  3. Flight assay index (optional)
  4. DAMS locomotor activity (optional)

MCMS is a weighted mean of the per-assay normalised scores, computed over the
assays that are ACTUALLY PRESENT for each group, with the weights renormalised
to sum to 1 over those assays. It is therefore on [0, 1] whatever subset of
assays a lab ran.

Behavioural changes vs. the 2026-08 version (audit docs/MODULE_AUDIT_2026-09.md)
--------------------------------------------------------------------------------
1. MEDIAN SURVIVAL is now the Kaplan-Meier median from `survival.py`, and is
   NaN when the curve never reaches S(t) = 0.5 ("median not reached").
   Previously `process_lifespan` fell back to `Day.max()`, so a cohort with
   40/100 dead by day 50 was assigned a median of 50. Cross-checked against
   lifelines, which returns inf for that cohort; survival.py returns NaN.
   Justification: reporting the stop day as the median is a factual error that
   propagates into Survival_norm and the MCMS tier. Audit S1 finding 1.4.

2. WEIGHTS ARE RENORMALISED over the assays present for each group. Previously
   an absent assay was mean-imputed to a normalised value of 0 while still
   being charged its weight, so a group that was best-possible on every axis
   scored 1.000 "Strong rescue" with 4 assays and 0.400 "Weak" with climbing
   alone. Audit S1 finding 1.5.

3. MISSING VALUES ARE NO LONGER MEAN-IMPUTED. Imputing a group's survival with
   the column mean invents a measurement and feeds it into a published score.
   A group missing an assay now simply has that assay excluded from its own
   weighted mean, and `Assays_used` records per-row which assays contributed.
   Justification: the project's standing rule forbids emitting numbers the data
   does not support.

4. PI and Flight are normalised on their OWN ABSOLUTE SCALES
   (PI_norm = (PI + 100)/200 on [-100, 100]; Flight_norm = Flight_Index/100 on
   [0, 100]) rather than min-max within the submitted dataset. Previously two
   groups both climbing at PI 95 both normalised to 0.0 and were both labelled
   "No rescue", because min-max maps a tie to zero. Audit S1 finding 1.6.

5. Survival and Activity have no universal ceiling, so they are normalised
   against caller-supplied anchors when given (`refs=`), and otherwise min-max
   within the dataset. When any dataset-relative component contributes, the row
   is flagged `Dataset_relative = True` and the tier label is suffixed
   " (dataset-relative)" — the score is then a within-experiment ranking, not a
   value comparable to another paper's MCMS.

6. "Activity_Fragmentation_Idx" is renamed "Activity_CV". The quantity computed
   is the coefficient of variation of pooled activity counts, not a bout-based
   fragmentation index (which requires the sleep/wake bout structure that
   `dams.py` computes). No consumer referenced the old name.

Validated by validation/validate_multi_assay.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_MCMS_WEIGHTS = {
    "PI_norm":       0.40,
    "Survival_norm": 0.25,
    "Flight_norm":   0.20,
    "Activity_norm": 0.15,
}

# Assays whose normalisation has a natural absolute scale.
_ABSOLUTE_COMPONENTS = ("PI_norm", "Flight_norm")
# Assays with no universal ceiling: dataset-relative unless the caller anchors them.
_RELATIVE_COMPONENTS = ("Survival_norm", "Activity_norm")

_COMPONENT_SOURCE = {
    "PI_norm":       "PI_mean",
    "Survival_norm": "Median_Survival_days",
    "Flight_norm":   "Flight_Index",
    "Activity_norm": "Daily_Activity_mean",
}
_COMPONENT_LABEL = {
    "PI_norm":       "Climbing (PI)",
    "Survival_norm": "Survival",
    "Flight_norm":   "Flight",
    "Activity_norm": "Activity (DAMS)",
}


def _norm_minmax(series: pd.Series) -> pd.Series:
    """
    Min-max normalise to [0, 1], preserving NaN.

    A zero-range column maps to 0.5 (neutral), NOT 0.0. Mapping a tie to the
    worst possible score made two groups both at PI 95 read "No rescue".
    """
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)
    lo, hi = float(valid.min()), float(valid.max())
    if hi == lo:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _norm_absolute(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Normalise onto [0, 1] against fixed anchors, preserving NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if hi == lo:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0)


def process_lifespan(lifespan_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Kaplan-Meier median survival per group.

    lifespan_raw columns:
        Genotype, Treatment, Sex, Day, Dead_count (new deaths on this day),
        [Censored_count], Total_start (total flies at start of experiment)

    Returns: Genotype, Treatment, Sex, Median_Survival_days, Max_Survival_days,
             Median_Reached, n, Events, Censored

    Median_Survival_days is NaN when the survival curve never reaches 0.5
    within follow-up ("median not reached"). It is NOT filled with the last
    scored day — that was the pre-2026-09 behaviour and it was wrong.
    """
    from modules.survival import to_event_table, median_survival_table

    if lifespan_raw is None or lifespan_raw.empty:
        return pd.DataFrame(columns=["Genotype", "Treatment", "Sex",
                                     "Median_Survival_days", "Max_Survival_days",
                                     "Median_Reached", "n", "Events", "Censored"])

    ev = to_event_table(lifespan_raw)
    med = median_survival_table(ev)
    if med.empty:
        return pd.DataFrame(columns=["Genotype", "Treatment", "Sex",
                                     "Median_Survival_days", "Max_Survival_days",
                                     "Median_Reached", "n", "Events", "Censored"])

    out = med.rename(columns={"Median_Survival": "Median_Survival_days",
                              "Max_observed": "Max_Survival_days"})
    out["Median_Reached"] = out["Median_Survival_days"].notna()
    return out[["Genotype", "Treatment", "Sex",
                "Median_Survival_days", "Max_Survival_days",
                "Median_Reached", "n", "Events", "Censored"]].reset_index(drop=True)


def process_flight(flight_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute flight index per group.
    flight_raw columns: Genotype, Treatment, Sex, Trial, Flew, NotFlew, Total
    Returns: Genotype, Treatment, Sex, Flight_Index (% flew)
    """
    agg = (
        flight_raw.groupby(["Genotype", "Treatment", "Sex"])
        .agg(Flew=("Flew", "sum"), Total=("Total", "sum"))
        .reset_index()
    )
    agg["Flight_Index"] = (agg["Flew"] / agg["Total"].replace(0, np.nan) * 100).round(2)
    return agg[["Genotype", "Treatment", "Sex", "Flight_Index"]]


def process_dams(dams_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Mean daily locomotor activity and its coefficient of variation from DAMS data.
    dams_raw columns: Genotype, Treatment, Sex, Day, Monitor_ID, Activity_counts

    Activity_CV = SD / Mean of the pooled activity counts.

    NOTE ON NAMING: this is a coefficient of variation, not a fragmentation
    index. A sleep/activity fragmentation index is defined on the bout
    structure (number and length of consolidated bouts) — see dams.py, which
    computes bouts under the Shaw/Hendricks >=5 min inactivity rule. The old
    column name "Activity_Fragmentation_Idx" overstated what this number is.
    """
    rows = []
    for (g, t, s), grp in dams_raw.groupby(["Genotype", "Treatment", "Sex"]):
        mean_act = float(grp["Activity_counts"].mean())
        sd_act   = float(grp["Activity_counts"].std())
        cv       = sd_act / mean_act if mean_act > 0 else np.nan
        rows.append({
            "Genotype": g, "Treatment": t, "Sex": s,
            "Daily_Activity_mean": round(mean_act, 2),
            "Activity_CV": round(cv, 3) if pd.notna(cv) else np.nan,
        })
    return pd.DataFrame(rows)


def compute_mcms(
    pi_df:       pd.DataFrame,
    survival_df: pd.DataFrame = None,
    flight_df:   pd.DataFrame = None,
    activity_df: pd.DataFrame = None,
    weights:     dict = None,
    refs:        Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Multi-Assay Composite Motor Score (MCMS).

    Required: pi_df with Genotype, Treatment, Sex, PI_mean.
    Optional: survival_df, flight_df, activity_df.

    refs : optional absolute upper anchors for the assays that have no natural
           ceiling, e.g. {"Median_Survival_days": 80.0,
                          "Daily_Activity_mean": 1200.0}.
           When supplied, that component is normalised as value/anchor (clipped
           to [0, 1]) and the score becomes comparable across experiments.
           When omitted, that component is min-max normalised within this
           dataset and the row is flagged Dataset_relative = True.

    For each group the score is the weighted mean over the components that
    group actually has, with weights renormalised to sum to 1:

        MCMS = sum(w_i * x_i) / sum(w_i)   over present components i

    A group with no computable component gets MCMS = NaN, not 0.

    Returns DataFrame sorted by MCMS descending.
    """
    w    = {**DEFAULT_MCMS_WEIGHTS, **(weights or {})}
    refs = dict(refs or {})

    if pi_df is None or pi_df.empty:
        return pd.DataFrame()

    df = pi_df[["Genotype", "Treatment", "Sex", "PI_mean"]].copy()

    def _merge(src, col):
        nonlocal df
        if src is not None and not src.empty and col in src.columns:
            df = df.merge(src[["Genotype", "Treatment", "Sex", col]],
                          on=["Genotype", "Treatment", "Sex"], how="left")
        else:
            df[col] = np.nan

    _merge(survival_df, "Median_Survival_days")
    _merge(flight_df,   "Flight_Index")
    _merge(activity_df, "Daily_Activity_mean")

    # ── Normalise each component. NaN in, NaN out — never imputed. ────────────
    # PI is defined on [-100, +100] and Flight_Index on [0, 100]: both have a
    # natural absolute scale, so a score means the same thing in every dataset.
    df["PI_norm"]     = _norm_absolute(df["PI_mean"],      -100.0, 100.0)
    df["Flight_norm"] = _norm_absolute(df["Flight_Index"],    0.0, 100.0)

    relative_used = []
    for comp, src in (("Survival_norm", "Median_Survival_days"),
                      ("Activity_norm", "Daily_Activity_mean")):
        anchor = refs.get(src)
        if anchor is not None and float(anchor) > 0:
            df[comp] = _norm_absolute(df[src], 0.0, float(anchor))
        else:
            df[comp] = _norm_minmax(df[src])
            relative_used.append(comp)

    # ── Weighted mean over the components each group actually has ────────────
    comps = list(_COMPONENT_SOURCE.keys())
    vals = df[comps].to_numpy(dtype=float)                    # (n_rows, n_comp)
    wvec = np.array([float(w.get(c, DEFAULT_MCMS_WEIGHTS[c])) for c in comps])
    present = ~np.isnan(vals)
    wmat = np.where(present, wvec[None, :], 0.0)
    wsum = wmat.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mcms = np.where(wsum > 0,
                        np.nansum(np.where(present, vals, 0.0) * wmat, axis=1) / wsum,
                        np.nan)

    df["MCMS"] = np.round(mcms, 4)
    df["MCMS_n_assays"] = present.sum(axis=1)
    df["Assays_used"] = [
        "; ".join(_COMPONENT_LABEL[c] for c, ok in zip(comps, row) if ok) or "none"
        for row in present
    ]
    # A row is dataset-relative only if a relatively-scaled component actually
    # contributed to that row's score.
    rel_idx = [comps.index(c) for c in relative_used]
    df["Dataset_relative"] = (present[:, rel_idx].any(axis=1)
                              if rel_idx else np.zeros(len(df), dtype=bool))

    def _tier(v, rel):
        if pd.isna(v):
            return "Not computable (no assay data)"
        base = ("Strong rescue" if v >= 0.75
                else "Moderate" if v >= 0.50
                else "Weak" if v >= 0.25
                else "No rescue")
        return base + (" (dataset-relative)" if rel else "")

    df["MCMS_Tier"] = [_tier(v, r) for v, r in zip(df["MCMS"], df["Dataset_relative"])]

    return df.sort_values("MCMS", ascending=False, na_position="last").reset_index(drop=True)


def assay_correlation_matrix(mcms_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix across assay components and MCMS."""
    cols = ["PI_norm", "Survival_norm", "Flight_norm", "Activity_norm", "MCMS"]
    avail = [c for c in cols if c in mcms_df.columns and mcms_df[c].notna().sum() > 1]
    if len(avail) < 2:
        return pd.DataFrame()
    return mcms_df[avail].corr(method="spearman").round(3)
