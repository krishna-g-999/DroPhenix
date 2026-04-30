"""
DroPhenix Analytics — motor_decline.py
Motor Decline Rate (MDR) and trajectory phenotyping module.

Computes per-group motor function dynamics from PI-vs-time curves:

1. Exponential decay fit:  PI(t) = PI0 * exp(-k * t)
   - k    : Motor Decline Rate constant (higher = faster decline)
   - PI0  : Initial motor performance intercept
   - t½   : Motor function half-life = ln(2) / k

2. Goodness of fit: R² and residual standard error (RSE)

3. Delta-k (Motor Rescue Delta):
   k_vehicle − k_treated
   Positive = treatment slows decline (rescue)
   Negative = treatment accelerates decline (impairment)

4. Trajectory phenotype classification (rule-based):
   Sustained Rescue | Transient Rescue | Progressive Decline |
   Plateau | Irregular

5. Motor-Survival Coupling Index (MSCI):
   Spearman ρ between per-group MDR (k) and median lifespan.

Key bug fixes vs previous version
----------------------------------
Root cause of "operands could not be broadcast together with shapes (11,) (11,2)":

  aggregate_replicates() produces agg["Pct_mean"].
  app.py calls compute_mdr_table(agg.rename({"Pct_mean": "PI_mean"})).
  If agg ALREADY contained a "PI_mean" column (from any prior merge or
  session-state bleed), pandas.rename() creates TWO columns both named
  "PI_mean".  grp["PI_mean"].values on a duplicate-column DataFrame returns
  shape (n_timepoints, 2) instead of (n_timepoints,).
  Then np.isnan(t) | np.isnan(p) tries to broadcast (11,) with (11,2)
  and raises ValueError.

Fixes applied
  1. _normalize_pi_table: deduplicates columns (keep first occurrence),
     adds "Pct_mean" -> "PI_mean" as an explicit rename path so the module
     accepts both naming conventions without relying on an external rename.
     Aggregates to ONE mean PI value per (Genotype, Treatment, Sex, Time)
     to handle both replicate-level and summary-level input transparently.
  2. fit_decay: .ravel() on t and p guarantees strictly 1-D arrays even if
     a 2-D array somehow slips through earlier guards.
  3. _r_squared / _residual_se: .ravel() defensive guards.
  4. compute_mdr_table: per-group values extracted via .to_numpy().ravel().

Scientific rationale
--------------------
Exponential decay is the appropriate model for neuromotor decline in
Drosophila ALS/PD models because PI decreases monotonically with age/time
in affected genotypes, decay rate k is biologically interpretable as disease
progression rate, and the model is parsimonious (2 parameters).

References
----------
Feany & Bender (2000) Nature 404:394
Gargano et al. (2005) J Neurobiol 65:11
Ritson et al. (2010) J Neurosci 30:543
Elden et al. (2010) Nature 466:1069
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

# ─── Decay and linear models ──────────────────────────────────────────────────
def _exp_decay(t: np.ndarray, PI0: float, k: float) -> np.ndarray:
    """Exponential decay: PI(t) = PI0 * exp(-k * t)"""
    return PI0 * np.exp(-k * t)


def _linear_decline(t: np.ndarray, m: float, b: float) -> np.ndarray:
    """Linear fallback: PI(t) = b + m * t"""
    return b + m * t


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Defensive: guarantee 1-D
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return round(1.0 - ss_res / ss_tot, 4)


def _residual_se(y_true: np.ndarray, y_pred: np.ndarray, n_params: int = 2) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = len(y_true)
    if n <= n_params:
        return float("nan")
    return round(float(np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - n_params))), 4)


# ─── Single-group decay fit ───────────────────────────────────────────────────
def fit_decay(
    time: np.ndarray,
    pi: np.ndarray,
    try_linear_fallback: bool = True,
) -> dict:
    """
    Fit exponential decay to one group's PI-vs-time series.

    Bug fix: .ravel() on t and p guarantees strictly 1-D float arrays,
    preventing the (11,) vs (11,2) broadcast crash that occurred when
    duplicate column names caused grp["PI_mean"].values to be 2-D.

    Returns
    -------
    dict: PI0, k, t_half, R2, RSE, fit_model, fit_success,
          linear_slope, linear_R2
    """
    # CRITICAL: guarantee 1-D regardless of input shape
    t = np.asarray(time, dtype=float).ravel()
    p = np.asarray(pi,   dtype=float).ravel()

    # Remove NaN pairs
    mask = ~(np.isnan(t) | np.isnan(p))
    t, p = t[mask], p[mask]

    base = {
        "PI0": float("nan"), "k": float("nan"), "t_half": float("nan"),
        "R2": float("nan"),  "RSE": float("nan"),
        "fit_model": "none", "fit_success": False,
        "linear_slope": float("nan"), "linear_R2": float("nan"),
    }

    if len(t) < 3:
        return base

    # ── Exponential fit ───────────────────────────────────────────────────────
    exp_result = base.copy()
    try:
        PI0_guess = max(float(np.nanmax(p)), 1.0)
        k_guess   = max(1e-4, float(
            -np.polyfit(t, np.log(np.clip(p, 1e-6, None)), 1)[0]
        ))
        popt, _ = curve_fit(
            _exp_decay, t, p,
            p0=[PI0_guess, k_guess],
            bounds=([0.0, 1e-6], [200.0, 10.0]),
            maxfev=20000,
            method="trf",
        )
        PI0, k   = float(popt[0]), float(popt[1])
        pred     = _exp_decay(t, PI0, k)
        r2       = _r_squared(p, pred)
        rse      = _residual_se(p, pred, n_params=2)
        t_half   = float(np.log(2) / k) if k > 1e-9 else float("inf")
        exp_result.update({
            "PI0":         round(PI0, 3),
            "k":           round(k, 5),
            "t_half":      round(t_half, 2) if np.isfinite(t_half) else 9999.0,
            "R2":          r2,
            "RSE":         rse,
            "fit_model":   "exponential",
            "fit_success": True,
        })
    except Exception:
        pass

    # ── Linear fallback ───────────────────────────────────────────────────────
    lin_result: dict = {}
    if try_linear_fallback:
        try:
            coeffs     = np.polyfit(t, p, 1)
            m, b       = float(coeffs[0]), float(coeffs[1])
            pred_lin   = _linear_decline(t, m, b)
            r2_lin     = _r_squared(p, pred_lin)
            lin_result = {
                "linear_slope":     round(m,     5),
                "linear_intercept": round(b,     3),
                "linear_R2":        round(r2_lin, 4),
            }
        except Exception:
            lin_result = {"linear_slope": float("nan"), "linear_R2": float("nan")}

    # Choose best model
    exp_r2 = exp_result.get("R2", float("nan"))
    lin_r2 = lin_result.get("linear_R2", float("nan"))

    if pd.notna(exp_r2) and exp_r2 >= 0.5:
        result = exp_result
        result.update(lin_result)
    elif pd.notna(lin_r2):
        result = base.copy()
        result.update(lin_result)
        result["fit_model"]   = "linear_fallback"
        result["fit_success"] = bool(pd.notna(lin_r2))
    else:
        result = base.copy()

    return result


# ─── PI table normalisation helper ───────────────────────────────────────────
def _normalize_pi_table(pi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a PI/Pct table for MDR computation.

    Key fixes applied here
    ----------------------
    1. Rename "Pct_mean" -> "PI_mean" directly so this module is independent
       of whether app.py pre-renamed the column (avoids the duplicate-column
       trap that caused the (11,) vs (11,2) broadcast crash).
    2. Deduplicate columns: if a "PI_mean" column existed AND "Pct_mean" was
       renamed to create a second one, only the first is kept.
    3. Aggregate to one mean per (Genotype, Treatment, Sex, Time): handles
       both replicate-level and summary-level inputs.
    """
    df = pi_df.copy()

    # ── Rename variants (order matters: most specific first) ─────────────────
    rename_map = {}
    if "PI_mean" not in df.columns:
        if "Pct_mean" in df.columns:
            rename_map["Pct_mean"] = "PI_mean"
        elif "PI" in df.columns:
            rename_map["PI"] = "PI_mean"
    if "Time" not in df.columns and "Time_used" in df.columns:
        rename_map["Time_used"] = "Time"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── Remove duplicate columns (take first occurrence) ─────────────────────
    # This is the primary defence against the (11,2) broadcast crash:
    # pandas.rename({col: same_col}) on a df that already has that col creates
    # two identically named columns, making df[col].values return 2-D.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    required = {"Genotype", "Treatment", "Sex", "Time", "PI_mean"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"MDR module: PI table missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    df["PI_mean"] = pd.to_numeric(df["PI_mean"], errors="coerce")
    df["Time"]    = pd.to_numeric(df["Time"],    errors="coerce")

    # ── Aggregate to one mean per timepoint ───────────────────────────────────
    # Handles replicate-level data (multiple rows per timepoint) and
    # prevents duplicate timepoints from distorting the exponential fit.
    key_cols   = ["Genotype", "Treatment", "Sex", "Time"]
    other_cols = [c for c in df.columns if c not in key_cols + ["PI_mean"]]

    try:
        agg_spec = {"PI_mean": ("PI_mean", "mean")}
        for c in other_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                agg_spec[c] = (c, "mean")
            else:
                agg_spec[c] = (c, "first")
        df = df.groupby(key_cols, as_index=False).agg(**agg_spec)
    except Exception:
        # Fallback: minimal aggregation
        df = df.groupby(key_cols, as_index=False)["PI_mean"].mean()

    return df


# ─── MDR table for all groups ─────────────────────────────────────────────────
def compute_mdr_table(
    pi_df: pd.DataFrame,
    vehicle_treatments: Optional[List[str]] = None,
    min_timepoints: int = 4,
) -> pd.DataFrame:
    """
    Compute Motor Decline Rate (MDR) for every Genotype × Treatment × Sex group.

    Accepts either:
    - aggregated table with "Pct_mean" (raw normalizer output)
    - aggregated table with "PI_mean" (already renamed)
    Both are handled transparently by _normalize_pi_table().

    Returns
    -------
    DataFrame: Genotype, Treatment, Sex, PI0, k, t_half, R2, RSE,
               fit_model, fit_success, linear_slope, linear_R2,
               delta_k, Rescue_pct, Trajectory, MDR_label
    """
    if vehicle_treatments is None:
        vehicle_treatments = ["untreated", "vehicle", "dmso", "control"]
    veh_lower = [v.lower() for v in vehicle_treatments]

    if pi_df is None or (isinstance(pi_df, pd.DataFrame) and pi_df.empty):
        return pd.DataFrame()

    df = _normalize_pi_table(pi_df)

    rows = []
    for (geno, treat, sex), grp in df.groupby(
        ["Genotype", "Treatment", "Sex"], dropna=False
    ):
        grp = grp.sort_values("Time").dropna(subset=["PI_mean"])

        if len(grp) < min_timepoints:
            rows.append({
                "Genotype": geno, "Treatment": treat, "Sex": sex,
                "PI0": float("nan"), "k": float("nan"), "t_half": float("nan"),
                "R2": float("nan"),  "RSE": float("nan"),
                "fit_model":   "insufficient_data",
                "fit_success": False,
                "linear_slope": float("nan"), "linear_R2": float("nan"),
                "n_timepoints": len(grp),
            })
            continue

        # .to_numpy().ravel() — the final safety net against 2-D slices
        t_arr = grp["Time"].to_numpy(dtype=float).ravel()
        p_arr = grp["PI_mean"].to_numpy(dtype=float).ravel()

        res = fit_decay(t_arr, p_arr)
        res.update({
            "Genotype":     geno,
            "Treatment":    treat,
            "Sex":          sex,
            "n_timepoints": len(grp),
        })
        rows.append(res)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # ── Delta-k: rescue vs. vehicle ──────────────────────────────────────────
    veh_mask = out["Treatment"].astype(str).str.lower().isin(veh_lower)
    if veh_mask.any():
        k_veh = (
            out.loc[veh_mask]
               .groupby(["Genotype", "Sex"])["k"]
               .mean()
               .rename("k_vehicle")
        )
        out = out.merge(k_veh, on=["Genotype", "Sex"], how="left")
        out["delta_k"]    = (out["k_vehicle"] - out["k"]).round(5)
        out["Rescue_pct"] = (
            (out["delta_k"] / out["k_vehicle"].replace(0, float("nan"))) * 100
        ).round(1)
        out = out.drop(columns=["k_vehicle"])
    else:
        out["delta_k"]    = float("nan")
        out["Rescue_pct"] = float("nan")

    # ── Trajectory labels ─────────────────────────────────────────────────────
    out["Trajectory"] = out.apply(
        lambda r: classify_trajectory(
            k=r.get("k",       float("nan")),
            delta_k=r.get("delta_k",  float("nan")),
            r2=r.get("R2",      float("nan")),
            pi0=r.get("PI0",     float("nan")),
        ),
        axis=1,
    )
    out["MDR_label"] = out.apply(_mdr_label, axis=1)

    col_order = [
        "Genotype", "Treatment", "Sex", "n_timepoints",
        "PI0", "k", "t_half", "R2", "RSE",
        "fit_model", "fit_success",
        "linear_slope", "linear_R2",
        "delta_k", "Rescue_pct",
        "Trajectory", "MDR_label",
    ]
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order].reset_index(drop=True)


# ─── Trajectory classification ────────────────────────────────────────────────
def classify_trajectory(
    k: float,
    delta_k: float,
    r2: float,
    pi0: float,
) -> str:
    """
    Rule-based trajectory phenotype labeler.

    Priority order
    --------------
    1. Poor fit (R² < 0.5)              → Irregular / poor fit
    2. Strong rescue (Δk > 0.005, PI0 ≥ 30) → Sustained Rescue
    3. Moderate rescue (Δk > 0.002)     → Transient Rescue
    4. Rapid decline (k > 0.025)        → Progressive Decline
    5. Very slow decline (k < 0.005)    → Plateau
    6. Default                          → Moderate Decline
    """
    if pd.isna(r2) or r2 < 0.5:
        return "Irregular / poor fit"
    if pd.notna(delta_k) and delta_k > 0.005 and pd.notna(pi0) and pi0 >= 30:
        return "Sustained Rescue"
    if pd.notna(delta_k) and delta_k > 0.002:
        return "Transient Rescue"
    if pd.notna(k) and k > 0.025:
        return "Progressive Decline"
    if pd.notna(k) and k < 0.005:
        return "Plateau"
    return "Moderate Decline"


def _mdr_label(row: pd.Series) -> str:
    k = row.get("k", float("nan"))
    if pd.isna(k):
        return "No fit"
    if k > 0.03:
        return "Rapid decline (k > 0.03)"
    if k > 0.015:
        return "Moderate decline (k 0.015-0.03)"
    if k > 0.005:
        return "Slow decline (k 0.005-0.015)"
    return "Minimal decline (k < 0.005)"


# ─── Motor-Survival Coupling Index ───────────────────────────────────────────
def compute_msci(
    mdr_df: pd.DataFrame,
    lifespan_df: pd.DataFrame,
    min_groups: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Motor-Survival Coupling Index (MSCI).

    Spearman ρ between per-group MDR (k) and median lifespan.
    Positive ρ = faster motor decline → shorter lifespan.
    """
    if mdr_df is None or lifespan_df is None:
        return pd.DataFrame(), pd.DataFrame()
    if mdr_df.empty or lifespan_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = mdr_df.merge(
        lifespan_df[["Genotype", "Treatment", "Sex", "Median_Survival_days"]],
        on=["Genotype", "Treatment", "Sex"],
        how="inner",
    )

    rows = []
    for (geno, sex), grp in merged.groupby(["Genotype", "Sex"], dropna=False):
        valid = grp[["k", "Median_Survival_days"]].dropna()
        if len(valid) < min_groups:
            continue
        try:
            rho, p = spearmanr(valid["k"], valid["Median_Survival_days"])
        except Exception:
            rho, p = float("nan"), float("nan")
        try:
            pr, pp = pearsonr(valid["k"], valid["Median_Survival_days"])
        except Exception:
            pr, pp = float("nan"), float("nan")

        rows.append({
            "Genotype":            geno,
            "Sex":                 sex,
            "n_groups":            len(valid),
            "MSCI_Spearman_rho":   round(float(rho), 4) if pd.notna(rho) else float("nan"),
            "p_Spearman":          round(float(p),   4) if pd.notna(p)   else float("nan"),
            "MSCI_Pearson_r":      round(float(pr),  4) if pd.notna(pr)  else float("nan"),
            "p_Pearson":           round(float(pp),  4) if pd.notna(pp)  else float("nan"),
            "Significant":         "Yes" if pd.notna(p) and p < 0.05 else "No",
            "Interpretation": (
                "Motor decline strongly predicts lifespan"
                if pd.notna(rho) and abs(rho) >= 0.7 and pd.notna(p) and p < 0.05
                else "Moderate motor-survival coupling"
                if pd.notna(rho) and abs(rho) >= 0.4 and pd.notna(p) and p < 0.05
                else "No significant motor-survival coupling"
            ),
        })

    return pd.DataFrame(rows), merged


# ─── Population-level MDR summary ────────────────────────────────────────────
def mdr_population_summary(mdr_df: pd.DataFrame) -> pd.DataFrame:
    """Population-level summary of MDR across genotypes and treatments."""
    if mdr_df is None or mdr_df.empty:
        return pd.DataFrame()

    rows = []
    for (geno, sex), grp in mdr_df.groupby(["Genotype", "Sex"], dropna=False):
        for treat, tgrp in grp.groupby("Treatment", dropna=False):
            valid_k  = tgrp["k"].dropna()
            valid_r2 = tgrp["R2"].dropna()
            rows.append({
                "Genotype":   geno,
                "Sex":        sex,
                "Treatment":  treat,
                "k_mean":     round(float(valid_k.mean()),  5) if len(valid_k)  else float("nan"),
                "k_std":      round(float(valid_k.std()),   5) if len(valid_k) > 1 else float("nan"),
                "R2_mean":    round(float(valid_r2.mean()), 4) if len(valid_r2) else float("nan"),
                "n_fits":     int(tgrp["fit_success"].sum()) if "fit_success" in tgrp.columns else 0,
                "dominant_trajectory": (
                    tgrp["Trajectory"].mode()[0]
                    if "Trajectory" in tgrp.columns and len(tgrp) > 0
                    else "Unknown"
                ),
            })

    return (
        pd.DataFrame(rows)
          .sort_values(["Genotype", "Sex", "k_mean"], ascending=[True, True, False])
          .reset_index(drop=True)
    )


# ── Compatibility alias ────────────────────────────────────────────────────────
compute_motor_decline = compute_msci
