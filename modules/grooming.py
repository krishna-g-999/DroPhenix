"""
DroPhenix — grooming.py
Drosophila grooming-behaviour assay engine.

Grooming is quantified from a behavioural ethogram: the intervals during which
each fly is grooming (from manual blind scoring or an automated classifier such
as the kNN method of Qiao et al., 2018, eLife 7:e34497). From those intervals
this module computes the standard grooming read-outs and group statistics.

Per-fly metrics:
  - total_grooming_s, observation_s
  - pct_time_grooming   (the primary read-out; wild-type ~13% of waking time,
                         Qiao et al. 2018)
  - n_bouts, mean_bout_s, grooming_rate_per_min

Accepted input (either):
  (A) Bout-level: Genotype, Treatment, Sex, FlyID, Bout_start_s, Bout_end_s
      (one row per grooming bout) plus an Observation_s per fly (column or arg).
  (B) Per-fly summary: Genotype, Treatment, Sex, FlyID, Total_grooming_s,
      Observation_s [, N_bouts].

NOTE: automated grooming *detection* from raw video is a supervised-ML problem
(pose/behaviour classification) and is out of scope here; this module analyses
the scored/classified grooming intervals, which is the standard quantification
output of both manual and automated pipelines.

CLUSTERING CAVEAT (audit S3 finding 3.9)
-----------------------------------------
`compare_groups` passes one row per ANIMAL straight to stats_engine.auto_test,
which treats every animal as an independent observation. Animals from the same
vial share a food batch, a rearing density and a handling event, so they are
correlated. The 2026-08 simulation measured what that costs: a naive
fly-as-independent test rejects 25.9% of the time under a TRUE NULL at
ICC = 0.10 — a 5.2x inflation of the nominal 5% error rate.

If the input carries a vial identifier, route the comparison through
`modules/hierarchical.py` instead, or at minimum report the design effect
DEFF = 1 + (m - 1) * ICC alongside the p-value. The p-values from this helper
are only valid when each animal really is an independent experimental unit.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

GROUP_KEYS = ["Genotype", "Treatment", "Sex"]


def _resolve(df: pd.DataFrame) -> Dict[str, str]:
    lut = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "Genotype": ["genotype", "strain", "line"],
        "Treatment": ["treatment", "drug", "compound", "condition"],
        "Sex": ["sex"],
        "FlyID": ["flyid", "fly_id", "fly", "id", "animal"],
        "Bout_start_s": ["bout_start_s", "start_s", "start", "onset_s", "onset"],
        "Bout_end_s": ["bout_end_s", "end_s", "end", "offset_s", "offset"],
        "Observation_s": ["observation_s", "obs_s", "duration_s", "total_time_s", "recording_s"],
        "Total_grooming_s": ["total_grooming_s", "grooming_s", "groom_s", "total_groom_s"],
        "N_bouts": ["n_bouts", "bouts", "num_bouts"],
    }
    found = {}
    for canon, opts in aliases.items():
        for o in opts:
            if o in lut:
                found[canon] = lut[o]
                break
    return found


def per_fly_metrics(df: pd.DataFrame, observation_s: Optional[float] = None) -> pd.DataFrame:
    """Compute per-fly grooming metrics from bout-level OR summary input."""
    if df is None or df.empty:
        return pd.DataFrame()
    col = _resolve(df)
    df = df.copy()
    for k, const in (("Treatment", "All"), ("Sex", "Pooled")):
        if k not in col:
            df[k] = const; col[k] = k
    if "Genotype" not in col:
        df["Genotype"] = "Genotype1"; col["Genotype"] = "Genotype"
    if "FlyID" not in col:
        df["FlyID"] = np.arange(len(df)); col["FlyID"] = "FlyID"

    rows = []
    # ── Summary input ─────────────────────────────────────────────────────────
    if "Total_grooming_s" in col:
        for _, r in df.iterrows():
            obs = float(r[col["Observation_s"]]) if "Observation_s" in col else (observation_s or np.nan)
            tot = float(r[col["Total_grooming_s"]])
            nb = int(r[col["N_bouts"]]) if "N_bouts" in col and pd.notna(r[col["N_bouts"]]) else np.nan
            rows.append(_rec(r[col["Genotype"]], r[col["Treatment"]], r[col["Sex"]],
                             r[col["FlyID"]], tot, obs, nb))
    # ── Bout-level input ──────────────────────────────────────────────────────
    elif "Bout_start_s" in col and "Bout_end_s" in col:
        keys = [col["Genotype"], col["Treatment"], col["Sex"], col["FlyID"]]
        for kv, sub in df.groupby(keys):
            starts = pd.to_numeric(sub[col["Bout_start_s"]], errors="coerce")
            ends = pd.to_numeric(sub[col["Bout_end_s"]], errors="coerce")
            iv = pd.DataFrame({"s": starts, "e": ends}).dropna()
            iv = iv[iv["e"] > iv["s"]]
            merged = _merge_intervals(iv["s"].tolist(), iv["e"].tolist())
            dur = np.array([e - s for s, e in merged], dtype=float)
            tot = float(dur.sum())
            nb = int(len(dur))
            obs = float(sub[col["Observation_s"]].iloc[0]) if "Observation_s" in col else (observation_s or np.nan)
            g, t, s, fid = kv
            rows.append(_rec(g, t, s, fid, tot, obs, nb,
                             mean_bout=float(dur.mean()) if nb else np.nan,
                             n_raw_bouts=int(len(iv))))
    else:
        raise ValueError("Grooming input needs either Total_grooming_s (summary) "
                         "or Bout_start_s/Bout_end_s (bout-level) columns.")
    return pd.DataFrame(rows)


def _merge_intervals(starts, ends):
    """
    Merge overlapping/touching [start, end) intervals into disjoint ones.

    Audit S2 finding 2.9: bout durations were summed without merging, so three
    overlapping scored bouts on one fly over a 600 s observation reported
    1100 s of grooming = 183.33% of the time, unflagged. The true union of those
    intervals is 600 s = 100%. Overlapping intervals are routine in manual
    ethogram scoring and in the output of the kNN classifier the module cites.
    """
    iv = sorted(zip(starts, ends))
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _rec(g, t, s, fid, tot, obs, nb, mean_bout=None, n_raw_bouts=None):
    pct = 100.0 * tot / obs if obs and obs > 0 else np.nan
    if mean_bout is None:
        mean_bout = tot / nb if (isinstance(nb, (int, float)) and nb and nb > 0) else np.nan
    rate = nb / (obs / 60.0) if (obs and obs > 0 and isinstance(nb, (int, float)) and np.isfinite(nb)) else np.nan

    # Grooming time cannot exceed observation time. After interval merging this
    # can only happen if the scored bouts run past the stated observation
    # window, which is a data-entry error the caller must see.
    flag = ""
    if np.isfinite(pct) and pct > 100.0 + 1e-9:
        flag = (f"IMPLAUSIBLE: {tot:.1f} s of grooming in a {obs:.1f} s "
                f"observation ({pct:.1f}%). Bouts extend past the stated "
                f"Observation_s — check the scoring window.")
    if n_raw_bouts is not None and nb is not None and n_raw_bouts > nb:
        flag = (flag + " " if flag else "") + (
            f"{n_raw_bouts - nb} of {n_raw_bouts} scored bouts overlapped and "
            f"were merged into {nb} disjoint bouts.")

    return {"Genotype": str(g), "Treatment": str(t), "Sex": str(s), "FlyID": str(fid),
            "Observation_s": round(obs, 1) if obs and np.isfinite(obs) else np.nan,
            "Total_grooming_s": round(tot, 2),
            "Pct_time_grooming": round(pct, 2) if np.isfinite(pct) else np.nan,
            "N_bouts": int(nb) if isinstance(nb, (int, float)) and np.isfinite(nb) else np.nan,
            "N_bouts_scored": int(n_raw_bouts) if n_raw_bouts is not None else np.nan,
            "Mean_bout_s": round(mean_bout, 2) if np.isfinite(mean_bout) else np.nan,
            "Grooming_rate_per_min": round(rate, 2) if np.isfinite(rate) else np.nan,
            "QC_flag": flag}


def group_summary(per_fly: pd.DataFrame) -> pd.DataFrame:
    """Per-group mean +/- SEM of grooming metrics."""
    if per_fly is None or per_fly.empty or "Genotype" not in per_fly.columns:
        return pd.DataFrame()
    metrics = ["Pct_time_grooming", "Total_grooming_s", "N_bouts",
               "Mean_bout_s", "Grooming_rate_per_min"]
    rows = []
    for kv, sub in per_fly.groupby(GROUP_KEYS, sort=False):
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec["n_flies"] = int(len(sub))
        for m in metrics:
            v = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy()
            rec[m] = round(float(np.mean(v)), 2) if len(v) else np.nan
            rec[m + "_SEM"] = round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_groups(per_fly: pd.DataFrame, metric: str = "Pct_time_grooming",
                   group_col: str = "Treatment") -> dict:
    from modules.stats_engine import auto_test
    if per_fly is None or metric not in per_fly.columns or group_col not in per_fly.columns:
        return {}
    data = {str(k): pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()
            for k, g in per_fly.groupby(group_col)}
    data = {k: v for k, v in data.items() if len(v) >= 2}
    if len(data) < 2:
        return {}
    return auto_test(data)


def demo_grooming(seed: int = 7, observation_s: float = 600.0) -> pd.DataFrame:
    """
    Synthetic bout-level grooming data calibrated so wild-type spends ~13% of
    time grooming (Qiao et al., 2018). The TDP43 model grooms less (motor
    impairment); the drug partially restores it. Bouts ~3-6 s. Unequal N.
    Returns bout-level rows (Genotype, Treatment, Sex, FlyID, Bout_start_s,
    Bout_end_s, Observation_s).
    """
    rng = np.random.default_rng(seed)
    pct = {("Oregon-WT", "Vehicle"): 0.13, ("Oregon-WT", "Drug"): 0.135,
           ("TDP43", "Vehicle"): 0.06, ("TDP43", "Drug"): 0.10}
    rows = []
    fid = 0
    for (geno, treat), frac in pct.items():
        for sex in ("Male", "Female"):
            n = int(rng.integers(12, 18))
            for _ in range(n):
                target = max(0.0, rng.normal(frac, 0.02)) * observation_s
                tcur = 0.0
                acc = 0.0
                while acc < target and tcur < observation_s:
                    gap = rng.exponential(25.0)               # inter-bout interval
                    bout = max(0.5, rng.normal(4.0, 1.2))     # bout ~3-6 s
                    tcur += gap
                    if tcur + bout > observation_s:
                        break
                    rows.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                                 "FlyID": f"fly{fid}", "Bout_start_s": round(tcur, 1),
                                 "Bout_end_s": round(tcur + bout, 1),
                                 "Observation_s": observation_s})
                    tcur += bout; acc += bout
                fid += 1
    return pd.DataFrame(rows)


def run_grooming_analysis(df: pd.DataFrame, observation_s: Optional[float] = None) -> dict:
    pf = per_fly_metrics(df, observation_s=observation_s)
    return {"per_fly": pf, "summary": group_summary(pf)}
