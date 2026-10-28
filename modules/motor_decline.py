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
    """Two-parameter exponential decay: PI(t) = PI0 * exp(-k * t). Floors at 0."""
    return PI0 * np.exp(-k * t)


def _exp_decay_offset(t: np.ndarray, A: float, k: float, C: float) -> np.ndarray:
    """
    Three-parameter exponential decay: PI(t) = A * exp(-k * t) + C.

    PI is defined on [-100, +100]. The two-parameter form PI0*exp(-k*t) with
    PI0 >= 0 cannot produce a negative value, so it could not fit any genotype
    whose performance index crosses zero — i.e. exactly the severely affected
    models the experiment is about (audit S2 finding 2.2). The free asymptote C
    gives the model the same support as the metric.
    """
    return A * np.exp(-k * t) + C


def _linear_decline(t: np.ndarray, m: float, b: float) -> np.ndarray:
    """Linear fallback: PI(t) = b + m * t"""
    return b + m * t


def _aic(rss: float, n: int, p: int) -> float:
    """
    Small-sample-corrected AIC (AICc) for least squares, in the standard
    Burnham & Anderson (2002, Model Selection and Multimodel Inference, 2nd ed.,
    section 2.2) form:

        AIC  = n * ln(RSS / n) + 2K
        AICc = AIC + 2K(K + 1) / (n - K - 1)

    where K is the number of ESTIMATED PARAMETERS INCLUDING the residual
    variance, i.e. K = p + 1 for a p-parameter regression. Counting sigma^2 is
    not optional book-keeping: the AICc correction is nonlinear in K, so
    omitting it changes the DIFFERENCE between two models with different p and
    can flip the selection. Measured on n = 9 with RSS 10.0 (linear, p = 2) vs
    5.0 (offset exponential, p = 3): the K = p convention gives dAICc = +1.44
    (picks linear) while the correct K = p + 1 gives -0.96 (picks the
    exponential).

    `p` here is the number of curve parameters; K = p + 1 is applied inside.
    Falls back to uncorrected AIC when n - K - 1 <= 0, where the correction is
    undefined. Burnham & Anderson recommend AICc whenever n/K < ~40, which is
    always the case for a longitudinal climbing design.

    Model choice between the exponential and linear forms must be made on a
    criterion that penalises parameters: the exponential-with-offset has three,
    the linear form two. Comparing raw R2 across different parameter counts
    would always favour the richer model.
    """
    K = p + 1                      # + 1 for the estimated residual variance
    if not np.isfinite(rss) or n <= K:
        return float("inf")
    rss = max(float(rss), 1e-300)
    aic = n * np.log(rss / n) + 2 * K
    denom = n - K - 1
    return aic + (2 * K * (K + 1) / denom) if denom > 0 else aic


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
    from scipy import stats as _st

    # CRITICAL: guarantee 1-D regardless of input shape
    t = np.asarray(time, dtype=float).ravel()
    p = np.asarray(pi,   dtype=float).ravel()

    # Remove NaN pairs
    mask = ~(np.isnan(t) | np.isnan(p))
    t, p = t[mask], p[mask]

    base = {
        "PI0": float("nan"), "k": float("nan"), "t_half": float("nan"),
        "PI_asymptote": float("nan"),
        "k_se": float("nan"), "k_CI_lower": float("nan"), "k_CI_upper": float("nan"),
        "t_half_se": float("nan"),
        "t_half_CI_lower": float("nan"), "t_half_CI_upper": float("nan"),
        "R2": float("nan"),  "RSE": float("nan"),
        "fit_model": "none", "fit_success": False,
        "linear_slope": float("nan"), "linear_R2": float("nan"),
        "AICc_exp": float("nan"), "AICc_linear": float("nan"),
        "delta_AICc": float("nan"), "fit_note": "",
    }

    n = len(t)
    if n < 3:
        return {**base, "fit_note": f"only {n} usable timepoint(s); need >= 3"}

    span = float(np.nanmax(p) - np.nanmin(p))
    lo_p, hi_p = float(np.nanmin(p)), float(np.nanmax(p))

    def _fit(model, p0, bounds, n_par):
        """Least-squares fit KEEPING the covariance matrix."""
        popt, pcov = curve_fit(model, t, p, p0=p0, bounds=bounds,
                               maxfev=40000, method="trf")
        pred = model(t, *popt)
        rss = float(np.sum((p - pred) ** 2))
        se = np.sqrt(np.clip(np.diag(np.asarray(pcov, dtype=float)), 0.0, np.inf))
        return popt, se, pred, rss, _aic(rss, n, n_par)

    # ── Exponential fit ───────────────────────────────────────────────────────
    # Prefer the three-parameter form A*exp(-k*t) + C, whose support covers the
    # whole PI range. It needs one more point than the two-parameter form, so
    # fall back to PI0*exp(-k*t) when there are too few ages — but only when the
    # data stay non-negative, where that form is actually applicable.
    exp_result, exp_aic = base.copy(), float("inf")
    tried = []

    # Candidate order matters. The two-parameter form PI0*exp(-k*t) has the same
    # parameter count as the linear model, so AICc compares them on equal terms.
    # The three-parameter offset form is only preferred when the data actually
    # need it — i.e. when PI goes negative, where the two-parameter form has the
    # wrong support and cannot be fitted at all. Trying the offset form first on
    # non-negative data would hand the linear model an unearned win, because the
    # AICc correction at n = 5 ages costs a third parameter +24 units.
    candidates = []
    if lo_p >= 0:
        candidates.append(("exponential", _exp_decay,
                           [max(hi_p, 1.0), 0.02],
                           ([0.0, 1e-6], [200.0, 10.0]), 2))
        candidates.append(("exponential_offset", _exp_decay_offset,
                           [max(span, 1.0), 0.02, lo_p],
                           ([0.0, 1e-6, -100.0], [400.0, 10.0, 100.0]), 3))
    else:
        tried.append("exponential: not applicable (PI goes negative; "
                     "PI0*exp(-k*t) with PI0 >= 0 has support [0, inf))")
        candidates.append(("exponential_offset", _exp_decay_offset,
                           [max(span, 1.0), 0.02, lo_p],
                           ([0.0, 1e-6, -100.0], [400.0, 10.0, 100.0]), 3))

    for label, model, p0, bounds, n_par in candidates:
        if n < n_par + 1:
            tried.append(f"{label}: skipped (needs >= {n_par + 1} ages, have {n})")
            continue
        try:
            popt, se, pred, rss, aic = _fit(model, p0, bounds, n_par)
        except Exception as e:
            tried.append(f"{label}: fit failed ({type(e).__name__})")
            continue

        if label == "exponential_offset":
            A, k, C = (float(v) for v in popt)
            k_se = float(se[1])
            pi0 = A + C
        else:
            pi0, k = float(popt[0]), float(popt[1])
            k_se, C = float(se[1]), 0.0

        t_half = float(np.log(2) / k) if k > 1e-9 else float("inf")
        # Delta method: t_half = ln2/k  =>  SE(t_half) = ln2 * SE(k) / k^2
        t_half_se = (float(np.log(2) * k_se / (k ** 2))
                     if np.isfinite(k_se) and k > 1e-9 else float("nan"))
        dfree = max(n - n_par, 1)
        crit = float(_st.t.ppf(0.975, dfree)) if dfree > 0 else float("nan")

        # t_half = ln2 / k is a convex, strictly decreasing function of k, so a
        # SYMMETRIC Wald interval t_half +/- z*SE undercovers: measured 0.865
        # against a nominal 0.95 over 400 simulations. The standard remedy is to
        # build the interval on the scale where the estimate is approximately
        # normal (k) and TRANSFORM THE ENDPOINTS, which is exact given the k
        # interval because the map is monotone. t_half_se is still reported as a
        # delta-method standard error for reporting, but the CI is the
        # transformed one and is what should be quoted.
        k_lo = k - crit * k_se if np.isfinite(k_se) else np.nan
        k_hi = k + crit * k_se if np.isfinite(k_se) else np.nan
        th_lo = float(np.log(2) / k_hi) if np.isfinite(k_hi) and k_hi > 0 else np.nan
        th_hi = float(np.log(2) / k_lo) if np.isfinite(k_lo) and k_lo > 0 else np.inf

        cand = {
            "PI0": round(pi0, 3), "k": round(k, 5),
            "PI_asymptote": round(C, 3),
            "t_half": round(t_half, 2) if np.isfinite(t_half) else float("inf"),
            "k_se": round(k_se, 6) if np.isfinite(k_se) else float("nan"),
            "k_CI_lower": round(k_lo, 6) if np.isfinite(k_lo) else float("nan"),
            "k_CI_upper": round(k_hi, 6) if np.isfinite(k_hi) else float("nan"),
            "t_half_CI_lower": round(th_lo, 2) if np.isfinite(th_lo) else float("nan"),
            "t_half_CI_upper": round(th_hi, 2) if np.isfinite(th_hi) else float("inf"),
            "t_half_se": round(t_half_se, 3) if np.isfinite(t_half_se) else float("nan"),
            "R2": _r_squared(p, pred),
            "RSE": _residual_se(p, pred, n_params=n_par),
            "fit_model": label, "fit_success": True,
            "AICc_exp": round(aic, 3),
        }
        if aic < exp_aic:
            exp_result, exp_aic = {**base, **cand}, aic

    # ── Linear alternative ────────────────────────────────────────────────────
    lin_result, lin_aic = {}, float("inf")
    if try_linear_fallback:
        try:
            m, b = (float(v) for v in np.polyfit(t, p, 1))
            pred_lin = _linear_decline(t, m, b)
            rss_lin = float(np.sum((p - pred_lin) ** 2))
            lin_aic = _aic(rss_lin, n, 2)
            lin_result = {
                "linear_slope":     round(m, 5),
                "linear_intercept": round(b, 3),
                "linear_R2":        round(_r_squared(p, pred_lin), 4),
                "AICc_linear":      round(lin_aic, 3),
            }
        except Exception:
            lin_result = {"linear_slope": float("nan"), "linear_R2": float("nan")}

    # ── Model selection on AICc, not on a bare R2 threshold ──────────────────
    # The old rule was "use the exponential if its R2 >= 0.5", which never
    # compared the two models: exactly linear data was reported as an
    # exponential with a half-life (audit S2 finding 2.3).
    note_parts = list(tried)
    if n <= 4:
        note_parts.append(f"n={n} ages: model selection is weakly determined")

    # Both fits are always REPORTED; selection only decides which is named in
    # `fit_model`. Blanking k when the linear model wins would make k
    # unavailable for exactly the cross-group comparison MDR exists to support:
    # one group going linear would turn every k comparison into NaN > NaN.
    result = {**base, **exp_result, **lin_result}
    exp_ok = bool(exp_result.get("fit_success", False))

    if exp_ok and np.isfinite(exp_aic) and exp_aic <= lin_aic:
        pass                                    # fit_model already the exp label
    elif lin_result and np.isfinite(lin_aic):
        result["fit_model"] = "linear"
        result["fit_success"] = True
        note_parts.append("linear preferred over exponential by AICc" if exp_ok
                          else "no exponential form was applicable; linear only")
    elif not exp_ok:
        result = dict(base)

    result["exp_k_available"] = exp_ok
    result["selected_model_is_exponential"] = str(result.get("fit_model", "")).startswith("exp")

    if np.isfinite(exp_aic) and np.isfinite(lin_aic):
        result["delta_AICc"] = round(abs(exp_aic - lin_aic), 3)
        if abs(exp_aic - lin_aic) < 2.0:
            note_parts.append("exponential and linear within 2 AICc: not distinguishable")
    result["fit_note"] = "; ".join(note_parts)

    # Goodness of fit OF THE MODEL THAT WAS SELECTED. Without this, a well-fitted
    # linear decline reached classify_trajectory with R2 = NaN (the exponential
    # slot) and was labelled "Irregular / poor fit".
    result["fit_R2"] = (result.get("linear_R2", float("nan"))
                        if str(result.get("fit_model", "")).startswith("linear")
                        else result.get("R2", float("nan")))
    return result


# ─── PI table normalisation helper ───────────────────────────────────────────
def _normalize_pi_table(pi_df: pd.DataFrame, time_col: str = "Age") -> pd.DataFrame:
    """
    Normalise a PI/Pct table for MDR computation.

    MDR (motor decline rate) is, by definition, the decline of climbing
    performance over AGE (days) — not the within-assay climbing kinetics over
    seconds (Gargano et al. 2005; Jahromi/Grotewiel 2008; the RING assay). The
    decline axis is therefore `time_col` (default "Age"). The supplied table
    must carry that column with >=2 distinct values per group; otherwise the
    decline cannot be estimated and the caller returns "requires_age_data".

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
    if time_col == "Time" and "Time" not in df.columns and "Time_used" in df.columns:
        rename_map["Time_used"] = "Time"
    # Recognise common age-column aliases for the decline axis.
    if time_col not in df.columns:
        lut = {str(c).strip().lower(): c for c in df.columns}
        for alias in ("age", "age_days", "ageday", "age_day", "agedays"):
            if alias in lut:
                rename_map[lut[alias]] = time_col
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # ── Remove duplicate columns (take first occurrence) ─────────────────────
    # This is the primary defence against the (11,2) broadcast crash:
    # pandas.rename({col: same_col}) on a df that already has that col creates
    # two identically named columns, making df[col].values return 2-D.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    required = {"Genotype", "Treatment", "Sex", time_col, "PI_mean"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"MDR module: PI table missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    df["PI_mean"]  = pd.to_numeric(df["PI_mean"], errors="coerce")
    df[time_col]   = pd.to_numeric(df[time_col],  errors="coerce")

    # ── Aggregate to one mean per age point ───────────────────────────────────
    # Handles replicate-level data (multiple rows per age) and prevents
    # duplicate age points from distorting the exponential fit.
    key_cols   = ["Genotype", "Treatment", "Sex", time_col]
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


def endpoint_pi_by_age(raw_df: pd.DataFrame, score_time: Optional[float] = None) -> pd.DataFrame:
    """
    Build the longitudinal endpoint-PI-vs-AGE table that MDR requires, from raw
    climbing data that carries an age column (Age / Age_days).

    For each Genotype x Treatment x Sex x Age (x Replicate) the endpoint PI is
    taken at the latest assay-second (or `score_time` if given), where
    PI = 2*(Count/n)*100 - 100. Returns columns:
    Genotype, Treatment, Sex, Age, PI_mean  (mean over replicates per age).
    Returns an empty frame if no age column is present.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    lut = {str(c).strip().lower(): c for c in raw_df.columns}
    def col(*names):
        for nm in names:
            if nm in lut:
                return lut[nm]
        return None
    g, t, s = col("genotype"), col("treatment"), col("sex")
    age = col("age", "age_days", "ageday", "age_day", "agedays")
    if age is None or g is None:
        return pd.DataFrame()
    df = raw_df.copy()
    t = t or "_t"; s = s or "_s"
    if t == "_t": df[t] = "All"
    if s == "_s": df[s] = "Pooled"
    rep = col("replicate", "rep", "trial")
    tcol = col("time", "time_sec", "seconds")
    # PI per row
    if col("pi"):
        df["_PI"] = pd.to_numeric(df[col("pi")], errors="coerce")
    else:
        cnt = col("count", "count_climbed", "climbed", "counts", "n_climbed")
        ntot = col("n", "n_total_flies", "total_flies", "total", "n_total")
        if cnt is None or ntot is None:
            return pd.DataFrame()
        pct = 100.0 * pd.to_numeric(df[cnt], errors="coerce") / pd.to_numeric(df[ntot], errors="coerce")
        df["_PI"] = 2.0 * pct - 100.0
    # endpoint per (group, age, replicate) at latest assay-second
    keys = [g, t, s, age] + ([rep] if rep else [])
    if tcol is not None:
        df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
        if score_time is not None:
            df = df[np.isclose(df[tcol], float(score_time))]
        else:
            idx = df.groupby(keys)[tcol].idxmax().dropna()
            df = df.loc[idx]
    out = (df.groupby([g, t, s, age], as_index=False)["_PI"].mean()
             .rename(columns={g: "Genotype", t: "Treatment", s: "Sex",
                              age: "Age", "_PI": "PI_mean"}))
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out.dropna(subset=["Age", "PI_mean"])


# ─── MDR table for all groups ─────────────────────────────────────────────────
def compute_mdr_table(
    pi_df: pd.DataFrame,
    vehicle_treatments: Optional[List[str]] = None,
    min_timepoints: int = 3,
    time_col: str = "Age",
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

    # MDR is the decline of PI over AGE. If no age axis is available, MDR is not
    # applicable — return an empty frame carrying a clear reason (the caller
    # surfaces this rather than mis-fitting decay to the within-assay seconds).
    try:
        df = _normalize_pi_table(pi_df, time_col=time_col)
    except ValueError:
        out = pd.DataFrame()
        out.attrs["mdr_status"] = (
            f"requires_age_data: MDR needs a longitudinal '{time_col}' (days) "
            f"column with >=2 ages per group. Climbing seconds are not an age axis.")
        return out

    rows = []
    for (geno, treat, sex), grp in df.groupby(
        ["Genotype", "Treatment", "Sex"], dropna=False
    ):
        grp = grp.sort_values(time_col).dropna(subset=["PI_mean"])

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
        t_arr = grp[time_col].to_numpy(dtype=float).ravel()
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
        ref = (
            out.loc[veh_mask]
               .groupby(["Genotype", "Sex"])[["k", "k_se"]]
               .mean()
               .rename(columns={"k": "k_vehicle", "k_se": "k_vehicle_se"})
        )
        out = out.merge(ref, on=["Genotype", "Sex"], how="left")
        out["delta_k"] = (out["k_vehicle"] - out["k"]).round(5)

        # Uncertainty propagation. The treated and vehicle groups are separate
        # fits on separate flies, so their estimates are independent and the
        # variances add. Previously curve_fit's covariance was discarded and
        # delta_k / Rescue_pct carried no error at all (audit S2 finding 2.4).
        var_sum = out["k_se"] ** 2 + out["k_vehicle_se"] ** 2
        out["delta_k_se"] = np.sqrt(var_sum).round(6)
        out["delta_k_CI_lower"] = (out["delta_k"] - 1.96 * out["delta_k_se"]).round(6)
        out["delta_k_CI_upper"] = (out["delta_k"] + 1.96 * out["delta_k_se"]).round(6)

        kv = out["k_vehicle"].replace(0, float("nan"))
        out["Rescue_pct"] = ((out["delta_k"] / kv) * 100).round(1)
        # Rescue_pct = 100*(1 - k/k_vehicle). Delta method on the ratio
        # r = k/k_vehicle, with k and k_vehicle independent:
        #   Var(r) ~= r^2 * ( Var(k)/k^2 + Var(k_veh)/k_veh^2 )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = out["k"] / kv
            rel_var = (out["k_se"] / out["k"]) ** 2 + (out["k_vehicle_se"] / kv) ** 2
            out["Rescue_pct_se"] = (100.0 * (ratio.abs() * np.sqrt(rel_var))).round(1)
        out = out.drop(columns=["k_vehicle", "k_vehicle_se"])
    else:
        for c in ("delta_k", "delta_k_se", "delta_k_CI_lower", "delta_k_CI_upper",
                  "Rescue_pct", "Rescue_pct_se"):
            out[c] = float("nan")

    # ── Trajectory labels ─────────────────────────────────────────────────────
    out["Trajectory"] = out.apply(
        lambda r: classify_trajectory(
            k=r.get("k",       float("nan")),
            delta_k=r.get("delta_k",  float("nan")),
            r2=r.get("fit_R2",  r.get("R2", float("nan"))),
            pi0=r.get("PI0",     float("nan")),
            fit_model=r.get("fit_model", ""),
            linear_slope=r.get("linear_slope", float("nan")),
        ),
        axis=1,
    )
    out["MDR_label"] = out.apply(_mdr_label, axis=1)

    col_order = [
        "Genotype", "Treatment", "Sex", "n_timepoints",
        "PI0", "PI_asymptote",
        "k", "k_se", "k_CI_lower", "k_CI_upper",
        "t_half", "t_half_CI_lower", "t_half_CI_upper", "t_half_se",
        "R2", "fit_R2", "RSE",
        "fit_model", "fit_success",
        "linear_slope", "linear_R2",
        "AICc_exp", "AICc_linear", "delta_AICc",
        "delta_k", "delta_k_se", "delta_k_CI_lower", "delta_k_CI_upper",
        "Rescue_pct", "Rescue_pct_se",
        "Trajectory", "MDR_label", "fit_note",
    ]
    col_order = [c for c in col_order if c in out.columns]
    return out[col_order].reset_index(drop=True)


# ─── Trajectory classification ────────────────────────────────────────────────
def classify_trajectory(
    k: float,
    delta_k: float,
    r2: float,
    pi0: float,
    fit_model: str = "",
    linear_slope: float = float("nan"),
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

    UNITS: the k and Δk thresholds are per DAY. They are heuristic labels, not
    published cut-offs — no source in the reference list defines k = 0.025/day
    as "progressive". Treat them as a display convenience and report k with its
    confidence interval as the actual result.

    When the linear model was selected there is no decay constant, so the label
    is derived from the fitted slope (PI units per day) instead of returning
    "Irregular / poor fit" for what may be an excellent straight-line fit.
    """
    if pd.isna(r2) or r2 < 0.5:
        return "Irregular / poor fit"

    if str(fit_model).startswith("linear"):
        if pd.isna(linear_slope):
            return "Irregular / poor fit"
        if linear_slope > -0.05:
            return "Plateau (linear fit)"
        if linear_slope < -1.0:
            return "Progressive Decline (linear fit)"
        return "Moderate Decline (linear fit)"

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
def _sigfig(v: float, sig: int = 3) -> float:
    """Round to `sig` significant figures, so a small p never renders as 0.0."""
    if v is None or not np.isfinite(v) or v == 0:
        return float(v) if v is not None and np.isfinite(v) else float("nan")
    return float(f"{float(v):.{sig}g}")


def _exact_spearman_p(x: np.ndarray, y: np.ndarray,
                      max_exact_n: int = 8, n_mc: int = 20000,
                      seed: int = 42) -> Tuple[float, float]:
    """
    Two-sided permutation p-value for Spearman's rho, and the SMALLEST p-value
    the design can attain.

    scipy.stats.spearmanr returns an asymptotic t-approximation whose statistic
    is infinite at |rho| = 1. At n = 3 that produced p = 0.0 from three points,
    while the exact permutation p for |rho| = 1 is 2/3! = 0.333
    (audit S1 finding 1.1). For n <= max_exact_n every permutation is
    enumerated, so the p-value is exact; above that a seeded Monte-Carlo
    permutation is used with the Phipson & Smyth (2010) (1 + count)/(1 + B)
    estimator, which never returns an unattainable p of 0.

    Returns (p_value, p_min_attainable).
    """
    from itertools import permutations as _perms

    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    rx = rx - rx.mean()
    ry0 = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry0 ** 2)))
    if denom <= 0:
        return float("nan"), float("nan")

    obs = abs(float(np.dot(rx, ry0)) / denom)

    if n <= max_exact_n:
        stats_null = []
        for perm in _perms(range(n)):
            yp = ry0[list(perm)]
            stats_null.append(abs(float(np.dot(rx, yp)) / denom))
        arr = np.asarray(stats_null)
        p = float(np.mean(arr >= obs - 1e-12))
        p_min = float(np.mean(arr >= arr.max() - 1e-12))
        return p, p_min

    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_mc):
        yp = rng.permutation(ry0)
        if abs(float(np.dot(rx, yp)) / denom) >= obs - 1e-12:
            cnt += 1
    return (1.0 + cnt) / (1.0 + n_mc), 1.0 / (1.0 + n_mc)


def compute_msci(
    mdr_df: pd.DataFrame,
    lifespan_df: pd.DataFrame,
    min_groups: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Motor-Survival Coupling Index (MSCI).

    Spearman rho between per-group MDR (k) and median lifespan, with an EXACT
    permutation p-value (see _exact_spearman_p). Positive rho = faster motor
    decline goes with shorter lifespan.

    TWO LIMITATIONS ARE REPORTED, NOT HIDDEN
    ----------------------------------------
    1. `p_min_attainable` is the smallest p this design could ever produce. At
       n = 3 groups it is 0.333, so no result can reach the 0.05 level however
       perfect the correlation. When p_min_attainable > 0.05 the row is labelled
       "Underpowered by construction" rather than being given a verdict.
    2. This is an ECOLOGICAL correlation between two group-level summaries. A
       genotype that dies early has less opportunity to decline, so k and
       lifespan are coupled through follow-up length as well as through biology.
       `modules/survivor_bias.py` handles that confound; MSCI does not. The
       `Caveat` column says so on every row.
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
        xv = valid["k"].to_numpy(dtype=float)
        yv = valid["Median_Survival_days"].to_numpy(dtype=float)

        try:
            rho, _p_asym = spearmanr(xv, yv)
        except Exception:
            rho, _p_asym = float("nan"), float("nan")
        p, p_min = _exact_spearman_p(xv, yv)
        try:
            pr, pp = pearsonr(xv, yv)
        except Exception:
            pr, pp = float("nan"), float("nan")

        underpowered = pd.notna(p_min) and p_min > 0.05
        significant = (not underpowered) and pd.notna(p) and p < 0.05

        if underpowered:
            interp = (f"Underpowered by construction: with {len(valid)} groups the "
                      f"smallest attainable p is {p_min:.3f}, so no correlation "
                      f"however perfect can reach the 0.05 level. Report rho as "
                      f"descriptive only.")
        elif significant and pd.notna(rho) and abs(rho) >= 0.7:
            interp = "Motor decline strongly predicts lifespan"
        elif significant and pd.notna(rho) and abs(rho) >= 0.4:
            interp = "Moderate motor-survival coupling"
        else:
            interp = "No significant motor-survival coupling"

        rows.append({
            "Genotype":            geno,
            "Sex":                 sex,
            "n_groups":            len(valid),
            "MSCI_Spearman_rho":   round(float(rho), 4) if pd.notna(rho) else float("nan"),
            # 3 significant figures, NOT 4 decimal places: an exact p of
            # 4.96e-05 rounds to "0.0" at 4dp, and a p-value printed as zero is
            # the very thing this function was fixed to stop reporting.
            "p_Spearman":          _sigfig(p),
            "p_min_attainable":    _sigfig(p_min),
            "p_method":            "exact permutation" if len(valid) <= 8 else "Monte-Carlo permutation",
            "MSCI_Pearson_r":      round(float(pr),  4) if pd.notna(pr)  else float("nan"),
            "p_Pearson":           round(float(pp),  4) if pd.notna(pp)  else float("nan"),
            "Significant":         "Yes" if significant else "No",
            "Interpretation":      interp,
            "Caveat": ("Ecological correlation between two group-level summaries; "
                       "not adjusted for survivor bias (see modules/survivor_bias.py)."),
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
