"""
DroPhenix — dams.py
Drosophila Activity Monitor System (DAMS / DAM2) analysis engine.

Two biologically meaningful read-outs from per-minute beam-crossing data:

1. Sleep / rest analysis
   ---------------------
   Drosophila sleep is operationally defined as >= 5 minutes of continuous
   inactivity (zero beam crossings) recorded by the DAM system
   (Hendricks et al., 2000, Neuron 25:129-138; Shaw et al., 2000, Science
   287:1834-1837). From this rule we derive total sleep, day/night sleep,
   bout number and length, sleep latency, and waking activity. Wild-type
   flies show a bimodal crepuscular profile (active at dawn/dusk, midday
   siesta, consolidated night sleep) and males sleep more than females,
   especially during the day.

2. Circadian period
   ----------------
   The free-running locomotor period is estimated with the chi-square (Qp)
   periodogram (Sokolove & Bushell, 1978, J. Theor. Biol. 72:131-160), the
   field standard for DAM data. The dominant period is the test period that
   maximises Qp; rhythmicity is called when peak Qp exceeds the chi-square
   significance threshold (df = p-1). A Lomb-Scargle periodogram is provided
   as an independent cross-check. Reference periods (Konopka & Benzer, 1971,
   PNAS 68:2112-2116): wild-type ~24 h, per-short ~19 h, per-long ~28 h,
   per-0 arrhythmic.

Canonical input (long format), one row per fly per time bin:
    Genotype, Treatment, Sex, FlyID, Time_min, Activity
  Time_min = minutes elapsed from experiment start; Activity = beam crossings
  in that bin. A wide DAM layout (one column per channel) is auto-melted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GROUP_KEYS = ["Genotype", "Treatment", "Sex"]
SLEEP_MIN_DEFAULT = 5          # minutes of inactivity defining sleep
DAY_MINUTES = 1440             # minutes per 24 h day


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


# ──────────────────────────────────────────────────────────────────────────────
# Input parsing
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    lut = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "Genotype": ["genotype", "strain", "line"],
        "Treatment": ["treatment", "drug", "compound", "condition"],
        "Sex": ["sex"],
        "FlyID": ["flyid", "fly_id", "fly", "channel", "monitor_id", "id", "subject"],
        "Time_min": ["time_min", "time", "minute", "min", "t", "datetime", "timestamp"],
        "Activity": ["activity", "activity_counts", "counts", "beam", "beam_crossings", "value"],
    }
    found = {}
    for canon, opts in aliases.items():
        for o in opts:
            if o in lut:
                found[canon] = lut[o]
                break
    return found


def load_dams(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise input to canonical long: Genotype,Treatment,Sex,FlyID,Time_min,Activity."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Genotype", "Treatment", "Sex", "FlyID", "Time_min", "Activity"])
    col = _resolve_columns(df)
    df = df.copy()
    for k, const in (("Treatment", "All"), ("Sex", "Pooled")):
        if k not in col:
            df[k] = const
            col[k] = k
    if "Genotype" not in col:
        df["Genotype"] = "Genotype1"; col["Genotype"] = "Genotype"

    if "Activity" in col and "Time_min" in col and "FlyID" in col:
        t = df[col["Time_min"]]
        if np.issubdtype(pd.Series(t).dtype, np.datetime64) or _looks_datetime(t):
            t = pd.to_datetime(t, errors="coerce")
            t = (t - t.min()).dt.total_seconds() / 60.0
        out = pd.DataFrame({
            "Genotype": df[col["Genotype"]].astype(str),
            "Treatment": df[col["Treatment"]].astype(str),
            "Sex": df[col["Sex"]].astype(str),
            "FlyID": df[col["FlyID"]].astype(str),
            "Time_min": pd.to_numeric(t, errors="coerce"),
            "Activity": pd.to_numeric(df[col["Activity"]], errors="coerce").fillna(0.0),
        })
    else:
        raise ValueError(
            "Unrecognised DAMS format. Provide long-format columns "
            "Genotype, Treatment, Sex, FlyID, Time_min, Activity.")
    out = out.dropna(subset=["Time_min"]).sort_values(
        ["Genotype", "Treatment", "Sex", "FlyID", "Time_min"]).reset_index(drop=True)
    return out


def _looks_datetime(s) -> bool:
    try:
        return pd.Series(s).astype(str).str.contains(r"[-:/]").mean() > 0.5
    except Exception:
        return False


def _bin_minutes(times: np.ndarray) -> float:
    """Infer the sampling interval (minutes) from consecutive time stamps."""
    d = np.diff(np.unique(times))
    if len(d) == 0:
        return 1.0
    return float(np.median(d))


# ──────────────────────────────────────────────────────────────────────────────
# Sleep analysis (5-min inactivity rule)
# ──────────────────────────────────────────────────────────────────────────────
def _zero_run_bouts(activity: np.ndarray, bin_min: float, sleep_min: float) -> List[Tuple[int, int]]:
    """Return (start_idx, length_in_bins) of every inactivity run >= sleep_min minutes."""
    is_zero = activity == 0
    bouts = []
    n = len(is_zero)
    i = 0
    min_bins = int(np.ceil(sleep_min / bin_min))
    while i < n:
        if is_zero[i]:
            j = i
            while j < n and is_zero[j]:
                j += 1
            length = j - i
            if length >= min_bins:
                bouts.append((i, length))
            i = j
        else:
            i += 1
    return bouts


def sleep_per_fly(long_df: pd.DataFrame,
                  sleep_min: float = SLEEP_MIN_DEFAULT,
                  day_start_min: float = 0.0,
                  photoperiod_min: float = 720.0,
                  day_length_min: float = DAY_MINUTES) -> pd.DataFrame:
    """
    Per-fly sleep metrics using the >=5-min inactivity rule.

    day_start_min   : minutes from record start at which lights-on (ZT0) occurs.
    photoperiod_min : light-phase duration (default 12 h = 720 min).
    day_length_min  : full cycle length (default 24 h = 1440 min).
    Sleep totals are normalised to minutes per 24 h.
    """
    rows = []
    keys = GROUP_KEYS + ["FlyID"]
    for kv, sub in long_df.groupby(keys, sort=False):
        sub = sub.sort_values("Time_min")
        t = sub["Time_min"].to_numpy(float)
        act = sub["Activity"].to_numpy(float)
        if len(act) < 2:
            continue
        bin_min = _bin_minutes(t)
        total_min = len(act) * bin_min
        n_days = max(total_min / day_length_min, 1e-9)

        bouts = _zero_run_bouts(act, bin_min, sleep_min)
        # Per-bin sleep mask (1 if the bin is part of a sleep bout)
        sleep_mask = np.zeros(len(act), dtype=bool)
        for s, l in bouts:
            sleep_mask[s:s + l] = True

        # Day/night classification by zeitgeber time
        zt = ((t - day_start_min) % day_length_min)
        is_day = zt < photoperiod_min

        total_sleep = sleep_mask.sum() * bin_min
        day_sleep = sleep_mask[is_day].sum() * bin_min
        night_sleep = sleep_mask[~is_day].sum() * bin_min

        bout_lengths = np.array([l * bin_min for _, l in bouts], dtype=float)
        waking = act[~sleep_mask]
        rec = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "n_bins": len(act),
            "bin_min": round(bin_min, 3),
            "Total_sleep_min_per_day": round(total_sleep / n_days, 1),
            "Day_sleep_min_per_day": round(day_sleep / n_days, 1),
            "Night_sleep_min_per_day": round(night_sleep / n_days, 1),
            "N_bouts_per_day": round(len(bouts) / n_days, 2),
            "Mean_bout_min": round(float(np.mean(bout_lengths)), 1) if len(bout_lengths) else 0.0,
            "Sleep_pct": round(100.0 * sleep_mask.mean(), 1),
            "Waking_activity": round(float(np.mean(waking)), 3) if len(waking) else 0.0,
            "Total_activity_per_day": round(float(act.sum()) / n_days, 1),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _empty_per_fly(per_fly: pd.DataFrame) -> bool:
    return per_fly is None or per_fly.empty or "Genotype" not in per_fly.columns


def sleep_summary(per_fly: pd.DataFrame) -> pd.DataFrame:
    """Per-group mean +/- SEM of the key sleep metrics."""
    if _empty_per_fly(per_fly):
        return pd.DataFrame()
    metrics = ["Total_sleep_min_per_day", "Day_sleep_min_per_day",
               "Night_sleep_min_per_day", "N_bouts_per_day", "Mean_bout_min",
               "Total_activity_per_day"]
    rows = []
    for kv, sub in per_fly.groupby(GROUP_KEYS, sort=False):
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec["n_flies"] = int(len(sub))
        for m in metrics:
            v = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy()
            rec[m] = round(float(np.mean(v)), 1) if len(v) else np.nan
            rec[m + "_SEM"] = round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 1) if len(v) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def sleep_profile(long_df: pd.DataFrame, epoch_min: float = 30.0,
                  day_start_min: float = 0.0,
                  day_length_min: float = DAY_MINUTES,
                  sleep_min: float = SLEEP_MIN_DEFAULT) -> pd.DataFrame:
    """
    Average sleep (minutes per epoch) as a function of zeitgeber time, per group
    — the data behind the bimodal sleep/activity profile plot.
    """
    keys = GROUP_KEYS + ["FlyID"]
    per_fly_zt = []
    for kv, sub in long_df.groupby(keys, sort=False):
        sub = sub.sort_values("Time_min")
        t = sub["Time_min"].to_numpy(float)
        act = sub["Activity"].to_numpy(float)
        if len(act) < 2:
            continue
        bin_min = _bin_minutes(t)
        bouts = _zero_run_bouts(act, bin_min, sleep_min)
        sleep_mask = np.zeros(len(act), dtype=bool)
        for s, l in bouts:
            sleep_mask[s:s + l] = True
        zt = ((t - day_start_min) % day_length_min)
        epoch = (zt // epoch_min).astype(int)
        g = dict(zip(GROUP_KEYS, kv[:3] if isinstance(kv, tuple) else (kv,)))
        # For each ZT epoch: mean fraction of bins spent asleep (across days),
        # converted to sleep-minutes per epoch.
        dff = pd.DataFrame({"epoch": epoch, "asleep": sleep_mask.astype(float)})
        frac = dff.groupby("epoch")["asleep"].mean()
        for ep, fr in frac.items():
            per_fly_zt.append({**g, "ZT_hour": round(ep * epoch_min / 60.0, 2),
                               "sleep_min": float(fr) * epoch_min})
    if not per_fly_zt:
        return pd.DataFrame()
    prof = pd.DataFrame(per_fly_zt)
    return (prof.groupby(GROUP_KEYS + ["ZT_hour"])["sleep_min"].mean().reset_index())


# ──────────────────────────────────────────────────────────────────────────────
# Circadian period — chi-square (Sokolove-Bushell) periodogram
# ──────────────────────────────────────────────────────────────────────────────
def chi_square_periodogram(activity: np.ndarray, bin_min: float,
                           period_lo_h: float = 16.0, period_hi_h: float = 32.0,
                           alpha: float = 0.05) -> dict:
    """
    Sokolove-Bushell chi-square periodogram.

    For each trial period (in bins) p, fold the series into p phase bins using
    only complete cycles (K cycles, N' = K*p points):
        Qp = N' * sum_h (Xbar_h - Xbar)^2 / sum_i (X_i - Xbar)^2
    Qp is compared to the chi-square (df = p-1) threshold at level alpha.
    Returns periods (h), Qp, threshold, and the dominant period (argmax Qp).
    """
    from scipy.stats import chi2
    x = np.asarray(activity, dtype=float)
    N = len(x)
    if N < 10:
        return {"periods_h": np.array([]), "Qp": np.array([]),
                "threshold": np.array([]), "peak_period_h": np.nan,
                "peak_Qp": np.nan, "rhythmic": False}

    p_lo = max(2, int(round(period_lo_h * 60.0 / bin_min)))
    p_hi = int(round(period_hi_h * 60.0 / bin_min))
    p_hi = min(p_hi, N // 2)          # need >=2 complete cycles
    periods_bins = np.arange(p_lo, p_hi + 1)
    # Bonferroni correction across the number of trial periods tested — without
    # it, testing ~tens of periods and flagging if ANY exceeds the per-test
    # threshold inflates the false-positive rate to ~1-(1-alpha)^n_periods.
    alpha_corr = alpha / max(len(periods_bins), 1)

    Qp = np.empty(len(periods_bins))
    thr = np.empty(len(periods_bins))
    for k, p in enumerate(periods_bins):
        K = N // p
        if K < 2:
            Qp[k] = np.nan; thr[k] = np.nan
            continue
        seg = x[:K * p].reshape(K, p)
        col_mean = seg.mean(axis=0)             # Xbar_h (mean over K cycles), length p
        grand = seg.mean()
        Np = K * p
        sstot = np.sum((seg - grand) ** 2)
        if sstot <= 0:
            Qp[k] = 0.0
        else:
            # Sokolove-Bushell Qp = between-phase SS / (total variance), which is
            # ~chi-square_(p-1) under the no-rhythm null. between-phase SS uses K
            # points per phase; total variance = SStot/Np. The leading factor is
            # therefore K*Np = K^2 * p (E[Qp] = p-1 under the null).
            Qp[k] = K * Np * np.sum((col_mean - grand) ** 2) / sstot
        thr[k] = chi2.ppf(1 - alpha_corr, p - 1)

    periods_h = periods_bins * bin_min / 60.0
    valid = ~np.isnan(Qp)
    if not valid.any():
        return {"periods_h": periods_h, "Qp": Qp, "threshold": thr,
                "peak_period_h": np.nan, "peak_Qp": np.nan, "rhythmic": False}
    # Dominant period = test period with largest Qp - threshold margin among
    # those above threshold; else the global Qp max (reported, flagged arrhythmic).
    margin = Qp - thr
    above = valid & (margin > 0)
    if above.any():
        idx = np.where(above)[0][np.argmax(margin[above])]
        rhythmic = True
    else:
        idx = np.where(valid)[0][np.argmax(Qp[valid])]
        rhythmic = False
    return {
        "periods_h": periods_h, "Qp": Qp, "threshold": thr,
        "peak_period_h": round(float(periods_h[idx]), 3),
        "peak_Qp": round(float(Qp[idx]), 2),
        "threshold_at_peak": round(float(thr[idx]), 2),
        "rhythmic": bool(rhythmic),
    }


def lomb_scargle_period(activity: np.ndarray, bin_min: float,
                        period_lo_h: float = 16.0, period_hi_h: float = 32.0) -> float:
    """Independent period estimate via the Lomb-Scargle periodogram (hours)."""
    from scipy.signal import lombscargle
    x = np.asarray(activity, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 10 or np.allclose(x, 0):
        return np.nan
    t = np.arange(n) * bin_min               # minutes
    periods_h = np.linspace(period_lo_h, period_hi_h, 600)
    ang = 2 * np.pi / (periods_h * 60.0)     # angular freq (per minute)
    pgram = lombscargle(t, x, ang, normalize=True)
    return round(float(periods_h[np.argmax(pgram)]), 3)


def period_per_fly(long_df: pd.DataFrame,
                   period_lo_h: float = 16.0, period_hi_h: float = 32.0,
                   periodogram_bin_min: float = 30.0,
                   alpha: float = 0.05) -> pd.DataFrame:
    """Per-fly circadian period (chi-square periodogram + Lomb-Scargle cross-check)."""
    rows = []
    keys = GROUP_KEYS + ["FlyID"]
    for kv, sub in long_df.groupby(keys, sort=False):
        sub = sub.sort_values("Time_min")
        t = sub["Time_min"].to_numpy(float)
        act = sub["Activity"].to_numpy(float)
        if len(act) < 10:
            continue
        raw_bin = _bin_minutes(t)
        # Resample to the periodogram bin (sum activity) for speed/standard practice.
        factor = max(1, int(round(periodogram_bin_min / raw_bin)))
        if factor > 1:
            K = len(act) // factor
            act_rs = act[:K * factor].reshape(K, factor).sum(axis=1)
            bin_used = raw_bin * factor
        else:
            act_rs = act
            bin_used = raw_bin
        cp = chi_square_periodogram(act_rs, bin_used, period_lo_h, period_hi_h, alpha)
        ls = lomb_scargle_period(act_rs, bin_used, period_lo_h, period_hi_h)
        rec = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "Period_h_chisq": cp["peak_period_h"],
            "Qp_peak": cp["peak_Qp"],
            "Qp_threshold": cp.get("threshold_at_peak", np.nan),
            "Rhythmic": cp["rhythmic"],
            "Period_h_lombscargle": ls,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def period_summary(per_fly: pd.DataFrame) -> pd.DataFrame:
    """Per-group mean period (rhythmic flies only) and percent rhythmic."""
    if _empty_per_fly(per_fly) or "Rhythmic" not in per_fly.columns:
        return pd.DataFrame()
    rows = []
    for kv, sub in per_fly.groupby(GROUP_KEYS, sort=False):
        rhythmic = sub[sub["Rhythmic"] == True]
        per = pd.to_numeric(rhythmic["Period_h_chisq"], errors="coerce").dropna().to_numpy()
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "n_flies": int(len(sub)),
            "Pct_rhythmic": round(100.0 * len(rhythmic) / len(sub), 1) if len(sub) else np.nan,
            "Mean_period_h": round(float(np.mean(per)), 2) if len(per) else np.nan,
            "SD_period_h": round(float(np.std(per, ddof=1)), 2) if len(per) > 1 else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_groups(per_fly: pd.DataFrame, metric: str, group_col: str = "Treatment") -> dict:
    """Compare a per-fly DAMS metric across groups via the shared stats engine."""
    from modules.stats_engine import auto_test
    if metric not in per_fly.columns or group_col not in per_fly.columns:
        return {}
    data = {str(k): pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()
            for k, g in per_fly.groupby(group_col)}
    data = {k: v for k, v in data.items() if len(v) >= 2}
    if len(data) < 2:
        return {}
    return auto_test(data)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo data (literature-structured)
# ──────────────────────────────────────────────────────────────────────────────
def demo_dams(seed: int = 7, n_days: int = 8, bin_min: int = 1) -> pd.DataFrame:
    """
    Synthetic free-running (constant-darkness, DD) DAM data. In DD the endogenous
    circadian period is expressed, so genotype period differences become visible
    (they would not in light-dark, where flies entrain to exactly 24 h). Each
    cycle has a bimodal active phase (subjective morning + evening peaks) and a
    consolidated rest phase that yields >=5-min inactivity (sleep).

    Periods (cf. Konopka & Benzer, 1971): Oregon-WT ~24 h; TDP43 vehicle disrupted
    to ~22.5 h; the drug partially restores it to ~23.6 h. Males have a longer
    rest fraction and therefore sleep more than females (published sexual
    dimorphism). N is deliberately unequal across groups.
    """
    rng = np.random.default_rng(seed)
    minutes = np.arange(n_days * DAY_MINUTES, dtype=float)

    def make_counts(period_h, rest_frac):
        # position within the endogenous cycle, 0..1
        pos = (minutes % (period_h * 60.0)) / (period_h * 60.0)
        active = pos < (1.0 - rest_frac)            # subjective day
        # bimodal activity within the active phase
        u = np.clip(pos / (1.0 - rest_frac), 0, 1)
        morning = np.exp(-((u - 0.10) ** 2) / (2 * 0.10 ** 2))
        evening = np.exp(-((u - 0.90) ** 2) / (2 * 0.12 ** 2))
        rate = (1.3 * morning + 1.6 * evening + 0.25) * active
        counts = rng.poisson(np.maximum(rate, 0) * 6.0)
        counts[~active] = 0                          # rest phase -> sleep
        return counts.astype(float)

    groups = {
        ("Oregon-WT", "Vehicle"): 24.0,
        ("Oregon-WT", "Drug"):    24.0,
        ("TDP43", "Vehicle"):     22.5,    # disrupted (short) period
        ("TDP43", "Drug"):        23.6,    # partial rescue
    }
    rows = []
    fid = 0
    for (geno, treat), base_period in groups.items():
        for sex in ("Male", "Female"):
            n_flies = int(rng.integers(8, 13))
            rest_frac = 0.42 if sex == "Male" else 0.30   # males rest/sleep more
            for _ in range(n_flies):
                per = base_period + rng.normal(0, 0.15)    # individual variation
                counts = make_counts(per, rest_frac + rng.normal(0, 0.02))
                rows.append(pd.DataFrame({
                    "Genotype": geno, "Treatment": treat, "Sex": sex,
                    "FlyID": f"fly{fid}", "Time_min": minutes, "Activity": counts}))
                fid += 1
    return pd.concat(rows, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# One-call orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_dams_analysis(df: pd.DataFrame, sleep_min: float = SLEEP_MIN_DEFAULT,
                      do_period: bool = True) -> dict:
    long_df = load_dams(df)
    spf = sleep_per_fly(long_df, sleep_min=sleep_min)
    out = {
        "long": long_df,
        "sleep_per_fly": spf,
        "sleep_summary": sleep_summary(spf),
        "sleep_profile": sleep_profile(long_df, sleep_min=sleep_min),
    }
    if do_period:
        ppf = period_per_fly(long_df)
        out["period_per_fly"] = ppf
        out["period_summary"] = period_summary(ppf)
    return out
