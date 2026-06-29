"""
DroPhenix Analytics — normalizer.py
Core normalization and primary climbing assay metrics
for Drosophila negative geotaxis assays.

Primary outputs
---------------
- normalised: canonical long-format data with Pct and PI added
- aggregated: group x time summaries for Pct and PI
- pi_raw: replicate-level endpoint PI
- pi: group-level endpoint PI summary
- pi_timecourse: group x time PI summary
- auc_raw: replicate-level AUC
- auc: group-level AUC summary
- t50_raw: replicate-level t50
- t50: group-level t50 summary
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import integrate


COMMON_TMIN = 10.0


def _sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
    - Pct = 100 * Count / n
    - PI  = 2 * Pct - 100
    """
    out = df.copy()

    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    out["Count"] = pd.to_numeric(out["Count"], errors="coerce")
    out["Time"] = pd.to_numeric(out["Time"], errors="coerce")

    out["Pct"] = np.where(
        out["n"] > 0,
        (out["Count"] / out["n"]) * 100.0,
        np.nan,
    )
    out["PI"] = np.where(
        out["n"] > 0,
        ((2.0 * out["Count"] - out["n"]) / out["n"]) * 100.0,
        np.nan,
    )

    out["Pct"] = out["Pct"].round(4)
    out["PI"] = out["PI"].round(4)
    return out


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate biological replicates by Genotype x Sex x Treatment x Time.
    Returns both Pct and PI summaries.
    """
    if "Pct" not in df.columns or "PI" not in df.columns:
        df = normalise(df)

    agg = (
        df.groupby(["Genotype", "Sex", "Treatment", "Time"], dropna=False)
        .agg(
            n_mean=("n", "mean"),
            n_rep=("Replicate", "nunique"),
            Pct_mean=("Pct", "mean"),
            Pct_std=("Pct", "std"),
            Pct_sem=("Pct", _sem),
            PI_mean=("PI", "mean"),
            PI_std=("PI", "std"),
            PI_sem=("PI", _sem),
        )
        .reset_index()
    )

    for col in ["n_mean", "Pct_mean", "Pct_std", "Pct_sem", "PI_mean", "PI_std", "PI_sem"]:
        agg[col] = agg[col].round(4)

    return agg


def _pi_label(pi: float) -> str:
    if pd.isna(pi):
        return "Unknown"
    if pi >= 80:
        return "Excellent climb"
    if pi >= 50:
        return "Good climb"
    if pi >= 20:
        return "Moderate climb"
    if pi >= -20:
        return "Poor climb"
    if pi >= -60:
        return "Severely impaired"
    return "Complete failure"


def calculate_pi(df: pd.DataFrame, timepoint: Optional[float] = None) -> pd.DataFrame:
    """
    Compute endpoint PI per replicate.
    If timepoint is None, use each replicate's final timepoint.
    """
    if "PI" not in df.columns:
        df = normalise(df)

    rows = []
    for (geno, sex, trt, rep), grp in df.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        grp = grp.sort_values("Time").dropna(subset=["PI", "Time"])
        if grp.empty:
            continue

        if timepoint is None:
            row = grp.iloc[-1]
        else:
            exact = grp.loc[np.isclose(grp["Time"].astype(float), float(timepoint))]
            if exact.empty:
                continue
            row = exact.iloc[0]

        rows.append(
            {
                "Genotype": geno,
                "Sex": sex,
                "Treatment": trt,
                "Replicate": rep,
                "n": row["n"],
                "Time_used": row["Time"],
                "PI": row["PI"],
            }
        )

    pi_raw = pd.DataFrame(rows)
    if pi_raw.empty:
        return pd.DataFrame(
            columns=["Genotype", "Sex", "Treatment", "n", "n_rep", "PI_mean", "PI_sem", "Time_used", "Time", "PI_label"]
        )

    pi_summary = (
        pi_raw.groupby(["Genotype", "Sex", "Treatment"], dropna=False)
        .agg(
            n=("n", "mean"),
            n_rep=("Replicate", "nunique"),
            PI_mean=("PI", "mean"),
            PI_std=("PI", "std"),
            PI_sem=("PI", _sem),
            Time_used=("Time_used", "median"),
        )
        .reset_index()
    )

    pi_summary["PI_mean"] = pi_summary["PI_mean"].round(2)
    pi_summary["PI_std"] = pi_summary["PI_std"].round(2)
    pi_summary["PI_sem"] = pi_summary["PI_sem"].round(2)
    pi_summary["Time_used"] = pi_summary["Time_used"].round(2)
    pi_summary["Time"] = pi_summary["Time_used"]
    pi_summary["PI_label"] = pi_summary["PI_mean"].apply(_pi_label)

    return pi_summary


def calculate_pi_timecourse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group-level PI summary at each timepoint.
    This is the preferred input for trajectory plots and motor-decline modeling.
    """
    if "PI" not in df.columns:
        df = normalise(df)

    out = (
        df.groupby(["Genotype", "Sex", "Treatment", "Time"], dropna=False)
        .agg(
            n_mean=("n", "mean"),
            n_rep=("Replicate", "nunique"),
            PI_mean=("PI", "mean"),
            PI_std=("PI", "std"),
            PI_sem=("PI", _sem),
        )
        .reset_index()
    )

    for col in ["n_mean", "PI_mean", "PI_std", "PI_sem"]:
        out[col] = out[col].round(4)

    return out


def calculate_auc(df: pd.DataFrame, min_time: float = COMMON_TMIN) -> pd.DataFrame:
    """
    Trapezoidal AUC of percent climbed vs time per replicate.
    Uses timepoints >= min_time.
    Also returns AUC_norm = observed AUC / maximal possible AUC over the same time range.
    """
    if "Pct" not in df.columns:
        df = normalise(df)

    rows = []
    for (geno, sex, trt, rep), grp in df.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        grp = grp.sort_values("Time").dropna(subset=["Pct", "Time"])
        grp = grp.loc[grp["Time"] >= float(min_time)]
        if len(grp) < 2:
            continue

        times = grp["Time"].astype(float).values
        pcts = grp["Pct"].astype(float).values

        auc = float(integrate.trapezoid(pcts, times))
        t_range = float(times[-1] - times[0])
        auc_max = 100.0 * t_range if t_range > 0 else np.nan
        auc_norm = auc / auc_max if pd.notna(auc_max) and auc_max > 0 else np.nan

        rows.append(
            {
                "Genotype": geno,
                "Sex": sex,
                "Treatment": trt,
                "Replicate": rep,
                "n": grp["n"].iloc[0],
                "AUC": auc,
                "AUC_norm": auc_norm,
                "Time_min": times[0],
                "Time_max": times[-1],
            }
        )

    auc_raw = pd.DataFrame(rows)
    if auc_raw.empty:
        return pd.DataFrame(columns=["Genotype", "Sex", "Treatment", "n", "n_rep", "AUC_mean", "AUC_norm", "AUC_sem"])

    auc_summary = (
        auc_raw.groupby(["Genotype", "Sex", "Treatment"], dropna=False)
        .agg(
            n=("n", "mean"),
            n_rep=("Replicate", "nunique"),
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
            AUC_sem=("AUC", _sem),
            AUC_norm=("AUC_norm", "mean"),
        )
        .reset_index()
    )

    auc_summary["AUC_mean"] = auc_summary["AUC_mean"].round(2)
    auc_summary["AUC_std"] = auc_summary["AUC_std"].round(2)
    auc_summary["AUC_sem"] = auc_summary["AUC_sem"].round(2)
    auc_summary["AUC_norm"] = auc_summary["AUC_norm"].round(4)
    return auc_summary


def _interpolate_t50(times: np.ndarray, pcts: np.ndarray) -> float:
    """
    First interpolated time at which percent climbed reaches 50%.
    Returns NaN if the threshold is never reached.
    """
    if len(times) == 0 or len(pcts) == 0:
        return np.nan

    for i, p in enumerate(pcts):
        if p >= 50.0:
            if i == 0:
                return float(times[0])
            t0, t1 = float(times[i - 1]), float(times[i])
            p0, p1 = float(pcts[i - 1]), float(pcts[i])
            if p1 == p0:
                return float(t0)
            return float(t0 + (50.0 - p0) * (t1 - t0) / (p1 - p0))
    return np.nan


def calculate_t50(df: pd.DataFrame, min_time: float = COMMON_TMIN) -> pd.DataFrame:
    """
    Interpolated time to 50% climbed per replicate, then summarized by group.
    """
    if "Pct" not in df.columns:
        df = normalise(df)

    rows = []
    for (geno, sex, trt, rep), grp in df.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        grp = grp.sort_values("Time").dropna(subset=["Pct", "Time"])
        grp = grp.loc[grp["Time"] >= float(min_time)]
        if grp.empty:
            continue

        t50 = _interpolate_t50(
            grp["Time"].astype(float).values,
            grp["Pct"].astype(float).values,
        )

        rows.append(
            {
                "Genotype": geno,
                "Sex": sex,
                "Treatment": trt,
                "Replicate": rep,
                "n": grp["n"].iloc[0],
                "t50": t50,
            }
        )

    t50_raw = pd.DataFrame(rows)
    if t50_raw.empty:
        return pd.DataFrame(columns=["Genotype", "Sex", "Treatment", "n", "n_rep", "t50_mean", "t50_sem"])

    t50_summary = (
        t50_raw.groupby(["Genotype", "Sex", "Treatment"], dropna=False)
        .agg(
            n=("n", "mean"),
            n_rep=("Replicate", "nunique"),
            t50_mean=("t50", "mean"),
            t50_std=("t50", "std"),
            t50_sem=("t50", _sem),
        )
        .reset_index()
    )

    t50_summary["t50_mean"] = t50_summary["t50_mean"].round(2)
    t50_summary["t50_std"] = t50_summary["t50_std"].round(2)
    t50_summary["t50_sem"] = t50_summary["t50_sem"].round(2)
    return t50_summary


def compute_all_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Full core-metrics pipeline.
    """
    df_norm = normalise(df)

    pi_raw_rows = []
    for (geno, sex, trt, rep), grp in df_norm.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        grp = grp.sort_values("Time").dropna(subset=["PI", "Time"])
        if grp.empty:
            continue
        row = grp.iloc[-1]
        pi_raw_rows.append(
            {
                "Genotype": geno,
                "Sex": sex,
                "Treatment": trt,
                "Replicate": rep,
                "n": row["n"],
                "Time_used": row["Time"],
                "PI": row["PI"],
            }
        )
    pi_raw = pd.DataFrame(pi_raw_rows)

    auc_raw_rows = []
    for (geno, sex, trt, rep), grp in df_norm.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        grp = grp.sort_values("Time").dropna(subset=["Pct", "Time"])
        grp = grp.loc[grp["Time"] >= float(COMMON_TMIN)]
        if len(grp) < 2:
            continue
        times = grp["Time"].astype(float).values
        pcts = grp["Pct"].astype(float).values
        auc = float(integrate.trapezoid(pcts, times))
        t_range = float(times[-1] - times[0])
        auc_max = 100.0 * t_range if t_range > 0 else np.nan
        auc_norm = auc / auc_max if pd.notna(auc_max) and auc_max > 0 else np.nan
        auc_raw_rows.append(
            {
                "Genotype": geno,
                "Sex": sex,
                "Treatment": trt,
                "Replicate": rep,
                "n": grp["n"].iloc[0],
                "AUC": auc,
                "AUC_norm": auc_norm,
                "Time_min": times[0],
                "Time_max": times[-1],
            }
        )
    auc_raw = pd.DataFrame(auc_raw_rows)

    t50_raw_rows = []
    for (geno, sex, trt, rep), grp in df_norm.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        grp = grp.sort_values("Time").dropna(subset=["Pct", "Time"])
        grp = grp.loc[grp["Time"] >= float(COMMON_TMIN)]
        if grp.empty:
            continue
        t50_val = _interpolate_t50(grp["Time"].astype(float).values, grp["Pct"].astype(float).values)
        t50_raw_rows.append(
            {
                "Genotype": geno,
                "Sex": sex,
                "Treatment": trt,
                "Replicate": rep,
                "n": grp["n"].iloc[0],
                "t50": t50_val,
            }
        )
    t50_raw = pd.DataFrame(t50_raw_rows)

    return {
        "normalised": df_norm,
        "aggregated": aggregate_replicates(df_norm),
        "pi_raw": pi_raw,
        "pi": calculate_pi(df_norm),
        "pi_timecourse": calculate_pi_timecourse(df_norm),
        "auc_raw": auc_raw,
        "auc": calculate_auc(df_norm),
        "t50_raw": t50_raw,
        "t50": calculate_t50(df_norm),
    }