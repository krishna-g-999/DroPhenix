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


def _warn_time_filter(df: pd.DataFrame, min_time: float, what: str) -> None:
    """
    Warn when the `Time >= min_time` filter removes everything.

    COMMON_TMIN is 10.0 and assumes Time is in SECONDS. The same experiment
    recorded in MINUTES has every timepoint below 10, so AUC and t50 silently
    returned nothing at all — no error, no empty-result message, the metrics
    simply vanished from the results (audit S3 finding 3.6).
    """
    import warnings
    t = pd.to_numeric(df.get("Time"), errors="coerce").dropna()
    if len(t) and float(t.max()) < float(min_time):
        warnings.warn(
            f"{what}: every timepoint is below min_time={min_time} "
            f"(observed range {t.min():g}-{t.max():g}), so no {what} could be "
            f"computed. COMMON_TMIN assumes Time is in SECONDS — if this "
            f"experiment records Time in minutes, convert it or pass "
            f"min_time explicitly.",
            RuntimeWarning, stacklevel=3,
        )


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
    - Pct = 100 * Count / n          percent of the vial above the scoring line
    - PI  = 2 * Pct - 100            performance index on [-100, +100]

    RELATION TO THE PUBLISHED PERFORMANCE INDEX (audit S3 finding 3.5)
    -------------------------------------------------------------------
    The negative-geotaxis literature defines

        PI_published = 0.5 * (n_total + n_top - n_bottom) / n_total     on [0, 1]

    Under this protocol's TWO-ZONE geometry — one scoring line, so every fly is
    either above it or below it and n_bottom = n_total - n_top — that reduces to
    n_top / n_total, and therefore

        PI_DroPhenix = 100 * (2 * PI_published - 1)

    verified to 1.4e-14 over every count for n = 10 and n = 20. The transform is
    affine and monotone, so every rank, correlation and statistical test is
    identical either way — this is a reporting-scale difference, NOT a different
    measurement. But the NUMBERS ARE NOT COMPARABLE: a paper reporting
    "PI = 60" from DroPhenix and one reporting "PI = 0.8" describe the same
    climb. State the scale in the methods.

    Note also that the published formula assumes a THREE-zone apparatus (top /
    middle / bottom, with flies between the lines counted in neither). DroPhenix
    scores a single operator-confirmed line, which is the two-zone case; that
    choice must be stated too, because with a genuine middle zone
    PI_published != n_top / n_total.
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
    _warn_time_filter(df, min_time, "AUC")

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
    _warn_time_filter(df, min_time, "t50")

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
        return pd.DataFrame(columns=["Genotype", "Sex", "Treatment", "n", "n_rep",
                                     "n_reached", "n_censored", "t50_mean", "t50_sem",
                                     "t50_censored", "t50_note"])

    t50_summary = (
        t50_raw.groupby(["Genotype", "Sex", "Treatment"], dropna=False)
        .agg(
            n=("n", "mean"),
            n_rep=("Replicate", "nunique"),
            n_reached=("t50", "count"),          # count() skips NaN
            t50_mean=("t50", "mean"),
            t50_std=("t50", "std"),
            t50_sem=("t50", _sem),
        )
        .reset_index()
    )

    # ── Right-censoring (audit S2 finding 2.1) ───────────────────────────────
    # A replicate whose climbing curve never reaches 50% has t50 = NaN, and
    # pandas' mean() silently skips it. The replicates that get dropped are
    # exactly the WORST ones, so the bias is systematic: a group where 2 of 3
    # replicates never reached 50% reported the identical t50_mean (27.5 s) and
    # the identical n_rep (3) as a group where all three did.
    #
    # t50 is right-censored time-to-event data. The mean over uncensored
    # replicates only is not an estimate of anything, so it is suppressed
    # whenever any replicate is censored, and the caller is told how many.
    t50_summary["n_censored"] = t50_summary["n_rep"] - t50_summary["n_reached"]
    t50_summary["t50_censored"] = t50_summary["n_censored"] > 0

    censored = t50_summary["t50_censored"]
    t50_summary["t50_mean_uncensored_only"] = t50_summary["t50_mean"].round(2)
    t50_summary.loc[censored, ["t50_mean", "t50_std", "t50_sem"]] = np.nan
    t50_summary["t50_note"] = np.where(
        censored,
        (t50_summary["n_censored"].astype("Int64").astype(str) + " of "
         + t50_summary["n_rep"].astype("Int64").astype(str)
         + " replicates never reached 50%; the mean over the remainder would be "
           "biased low because the censored replicates are the slowest. Use the "
           "Kaplan-Meier median (see modules/survival.py) or report "
           "n_reached/n_rep as the endpoint."),
        "all replicates reached 50%",
    )

    for c in ("t50_mean", "t50_std", "t50_sem"):
        t50_summary[c] = t50_summary[c].round(2)
    return t50_summary


def compute_all_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Full core-metrics pipeline.
    """
    df_norm = normalise(df)
    _warn_time_filter(df_norm, COMMON_TMIN, "AUC/t50")

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