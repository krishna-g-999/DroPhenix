"""
DroPhenix — multi_assay.py
Multi-Assay Composite Motor Score (MCMS) integration module.

Integrates:
  1. Climbing assay PI (primary, required)
  2. Survival / lifespan median (optional)
  3. Flight assay index (optional)
  4. DAMS locomotor activity (optional)

MCMS = w1*PI_norm + w2*Survival_norm + w3*Flight_norm + w4*Activity_norm

Missing assay data is mean-imputed and flagged in Assays_used column.
"""

import numpy as np
import pandas as pd

DEFAULT_MCMS_WEIGHTS = {
    "PI_norm":       0.40,
    "Survival_norm": 0.25,
    "Flight_norm":   0.20,
    "Activity_norm": 0.15,
}


def _norm_col(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns 0.0 series if no variation."""
    lo, hi = float(series.min()), float(series.max())
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def process_lifespan(lifespan_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median survival per group from daily survival data.

    lifespan_raw columns:
        Genotype, Treatment, Sex, Day, Dead_count (new deaths on this day),
        Total_start (total flies at start of experiment)

    Returns: Genotype, Treatment, Sex, Median_Survival_days, Max_Survival_days
    """
    rows = []
    for (g, t, s), grp in lifespan_raw.groupby(["Genotype", "Treatment", "Sex"]):
        grp_s = grp.sort_values("Day")
        total = int(grp_s["Total_start"].iloc[0])
        cum_dead = grp_s["Dead_count"].cumsum()
        # Median survival: first day cumulative deaths >= 50% of cohort
        med_rows = grp_s.loc[cum_dead >= total * 0.5, "Day"]
        median = int(med_rows.iloc[0]) if len(med_rows) > 0 else int(grp_s["Day"].max())
        rows.append({
            "Genotype": g, "Treatment": t, "Sex": s,
            "Median_Survival_days": median,
            "Max_Survival_days": int(grp_s["Day"].max()),
        })
    return pd.DataFrame(rows)


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
    Compute mean daily locomotor activity and fragmentation index from DAMS data.
    dams_raw columns: Genotype, Treatment, Sex, Day, Monitor_ID, Activity_counts
    Fragmentation Index = SD / Mean (coefficient of variation of activity)
    """
    rows = []
    for (g, t, s), grp in dams_raw.groupby(["Genotype", "Treatment", "Sex"]):
        mean_act = float(grp["Activity_counts"].mean())
        sd_act   = float(grp["Activity_counts"].std())
        frag     = sd_act / mean_act if mean_act > 0 else np.nan
        rows.append({
            "Genotype": g, "Treatment": t, "Sex": s,
            "Daily_Activity_mean": round(mean_act, 2),
            "Activity_Fragmentation_Idx": round(frag, 3),
        })
    return pd.DataFrame(rows)


def compute_mcms(
    pi_df:       pd.DataFrame,
    survival_df: pd.DataFrame = None,
    flight_df:   pd.DataFrame = None,
    activity_df: pd.DataFrame = None,
    weights:     dict = None,
) -> pd.DataFrame:
    """
    Compute Multi-Assay Composite Motor Score (MCMS).

    Required: pi_df with Genotype, Treatment, Sex, PI_mean.
    Optional: survival_df, flight_df, activity_df.
    Missing assays are mean-imputed and logged in Assays_used column.

    Returns DataFrame sorted by MCMS descending.
    """
    w  = {**DEFAULT_MCMS_WEIGHTS, **(weights or {})}
    df = pi_df[["Genotype", "Treatment", "Sex", "PI_mean"]].copy()
    assays_present = ["Climbing (PI)"]

    if survival_df is not None and not survival_df.empty:
        df = df.merge(survival_df[["Genotype","Treatment","Sex","Median_Survival_days"]],
                      on=["Genotype","Treatment","Sex"], how="left")
        assays_present.append("Survival")
    else:
        df["Median_Survival_days"] = np.nan

    if flight_df is not None and not flight_df.empty:
        df = df.merge(flight_df[["Genotype","Treatment","Sex","Flight_Index"]],
                      on=["Genotype","Treatment","Sex"], how="left")
        assays_present.append("Flight")
    else:
        df["Flight_Index"] = np.nan

    if activity_df is not None and not activity_df.empty:
        df = df.merge(activity_df[["Genotype","Treatment","Sex","Daily_Activity_mean"]],
                      on=["Genotype","Treatment","Sex"], how="left")
        assays_present.append("Activity (DAMS)")
    else:
        df["Daily_Activity_mean"] = np.nan

    # Mean-impute missing assay columns
    for col in ["PI_mean", "Median_Survival_days", "Flight_Index", "Daily_Activity_mean"]:
        col_mean = df[col].mean()
        if pd.isna(col_mean):
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(col_mean)

    df["PI_norm"]       = _norm_col(df["PI_mean"])
    df["Survival_norm"] = _norm_col(df["Median_Survival_days"])
    df["Flight_norm"]   = _norm_col(df["Flight_Index"])
    df["Activity_norm"] = _norm_col(df["Daily_Activity_mean"])

    df["MCMS"] = (
        w.get("PI_norm",       0.40) * df["PI_norm"]       +
        w.get("Survival_norm", 0.25) * df["Survival_norm"] +
        w.get("Flight_norm",   0.20) * df["Flight_norm"]   +
        w.get("Activity_norm", 0.15) * df["Activity_norm"]
    ).round(4)

    df["MCMS_Tier"] = df["MCMS"].apply(
        lambda v: "Strong rescue" if v >= 0.75
                  else "Moderate" if v >= 0.50
                  else "Weak"     if v >= 0.25
                  else "No rescue"
    )
    df["Assays_used"] = "; ".join(assays_present)

    return df.sort_values("MCMS", ascending=False).reset_index(drop=True)


def assay_correlation_matrix(mcms_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation matrix across assay components and MCMS."""
    cols = ["PI_norm", "Survival_norm", "Flight_norm", "Activity_norm", "MCMS"]
    avail = [c for c in cols if c in mcms_df.columns and mcms_df[c].notna().sum() > 1]
    if len(avail) < 2:
        return pd.DataFrame()
    return mcms_df[avail].corr(method="spearman").round(3)
