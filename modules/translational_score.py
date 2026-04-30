"""
DroPhenix Analytics — translational_score.py
Translational Candidate Score (TCS) module.

TCS = w1*CRI_norm + w2*TSW_norm + w3*SDQ_consistency + w4*MDR_norm + w5*Traj_score

Default weights (evidence-based, all configurable via run_tcs_pipeline weights=):
  CRI : 0.35  — primary rescue magnitude
  TSW : 0.25  — selectivity / translational safety
  SDQ : 0.15  — sex consistency / generalizability
  MDR : 0.15  — kinetic rescue / disease modification
  Traj: 0.10  — phenotype stability / reproducibility
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Default weights ──────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "CRI":  0.35,
    "TSW":  0.25,
    "SDQ":  0.15,
    "MDR":  0.15,
    "Traj": 0.10,
}

TRAJECTORY_RANK: Dict[str, float] = {
    "Sustained Rescue":        1.0,
    "Motor Responder":         0.9,
    "High Performer":          0.85,
    "Progressive Rescue":      0.75,
    "Transient Rescue":        0.6,
    "Early Rescue Late Decline": 0.55,
    "Plateau Phenotype":       0.5,
    "Plateau":                 0.5,
    "Moderate Decline":        0.35,
    "Declining Performer":     0.3,
    "Mixed Phenotype":         0.3,
    "Progressive Decline":     0.15,
    "Severely Impaired":       0.05,
    "Irregular / poor fit":    0.0,
    "Undetermined":            0.2,
    "Unclassified":            0.2,
    "No fit":                  0.0,
}


# ─── Normalization helpers ────────────────────────────────────────────────────
def _minmax_norm(
    series: pd.Series,
    clip_low: float = -50.0,
    clip_high: float = 150.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").clip(clip_low, clip_high)
    s_min, s_max = s.min(), s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(
            [np.nan if pd.isna(v) else (target_min + target_max) / 2 for v in s],
            index=series.index,
        )
    return target_min + (s - s_min) / (s_max - s_min) * (target_max - target_min)


def _norm_tsw(tsw_series: pd.Series) -> pd.Series:
    s = pd.to_numeric(tsw_series, errors="coerce")
    return s.clip(0.0, 2.0) / 2.0


def _norm_sdq_consistency(sdq_series: pd.Series) -> pd.Series:
    s = pd.to_numeric(sdq_series, errors="coerce").clip(0.0, 1.0)
    return 1.0 - s


def _traj_score(traj_series: pd.Series) -> pd.Series:
    return traj_series.astype(str).map(lambda x: TRAJECTORY_RANK.get(x, 0.2))


# ─── Build TCS input table ────────────────────────────────────────────────────
def _build_tcs_input(
    cri_df: pd.DataFrame,
    tsw_df: pd.DataFrame,
    sdq_df: pd.DataFrame,
    mdr_df: pd.DataFrame,
    cluster_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if cri_df is None or cri_df.empty:
        return pd.DataFrame()

    base = cri_df[["Genotype", "Treatment", "Sex", "CRI"]].copy()

    if tsw_df is not None and not tsw_df.empty and "TSW" in tsw_df.columns:
        tsw_cols = [c for c in ["Genotype", "Treatment", "Sex", "TSW"] if c in tsw_df.columns]
        base = base.merge(tsw_df[tsw_cols], on=["Genotype", "Treatment", "Sex"], how="left")
    else:
        base["TSW"] = np.nan

    if sdq_df is not None and not sdq_df.empty and "SDQ" in sdq_df.columns:
        sdq_merge = sdq_df[["Genotype", "Treatment", "SDQ"]].copy()
        base = base.merge(sdq_merge, on=["Genotype", "Treatment"], how="left")
    else:
        base["SDQ"] = np.nan

    if mdr_df is not None and not mdr_df.empty:
        mdr_needed = [c for c in ["Genotype", "Treatment", "Sex", "delta_k", "Trajectory", "Rescue_pct"] if c in mdr_df.columns]
        base = base.merge(mdr_df[mdr_needed].copy(), on=["Genotype", "Treatment", "Sex"], how="left")
    else:
        base["delta_k"]   = np.nan
        base["Trajectory"] = np.nan
        base["Rescue_pct"] = np.nan

    if cluster_df is not None and not cluster_df.empty:
        cluster_needed = [c for c in ["Genotype", "Treatment", "Sex", "Archetype", "Cluster"] if c in cluster_df.columns]
        cluster_sub = cluster_df[cluster_needed].drop_duplicates()
        base = base.merge(cluster_sub, on=["Genotype", "Treatment", "Sex"], how="left")

    if "Archetype" in base.columns:
        base["Trajectory"] = base["Trajectory"].fillna(base["Archetype"])

    return base.reset_index(drop=True)


# ─── TCS label ────────────────────────────────────────────────────────────────
def _tcs_label(tcs: float) -> str:
    if pd.isna(tcs):
        return "Insufficient data"
    if tcs >= 80: return "Tier 1 — Strong translational candidate"
    if tcs >= 60: return "Tier 2 — Promising candidate"
    if tcs >= 40: return "Tier 3 — Moderate evidence"
    if tcs >= 20: return "Tier 4 — Weak evidence"
    return "Tier 5 — Not recommended"


# ─── TCS computation ──────────────────────────────────────────────────────────
def compute_tcs(
    cri_df: pd.DataFrame,
    tsw_df: pd.DataFrame,
    sdq_df: pd.DataFrame,
    mdr_df: pd.DataFrame,
    cluster_df: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    min_data_sources: int = 2,
) -> pd.DataFrame:
    """
    Compute TCS per Treatment × Genotype × Sex.
    weights: override DEFAULT_WEIGHTS (FIX: accepted via run_tcs_pipeline too)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 0.01:
        weights = {k: v / wsum for k, v in weights.items()}

    w_cri  = weights.get("CRI",  0.35)
    w_tsw  = weights.get("TSW",  0.25)
    w_sdq  = weights.get("SDQ",  0.15)
    w_mdr  = weights.get("MDR",  0.15)
    w_traj = weights.get("Traj", 0.10)

    inp = _build_tcs_input(cri_df, tsw_df, sdq_df, mdr_df, cluster_df)
    if inp.empty:
        return pd.DataFrame()

    inp["CRI_norm"]  = _minmax_norm(inp["CRI"], clip_low=-50, clip_high=150)
    inp["TSW_norm"]  = _norm_tsw(inp["TSW"])
    inp["SDQ_cons"]  = _norm_sdq_consistency(inp["SDQ"])

    if "delta_k" in inp.columns:
        dk = pd.to_numeric(inp["delta_k"], errors="coerce")
        max_dk = dk.max()
        inp["MDR_norm"] = (dk / max_dk).clip(0.0, 1.0) if (pd.notna(max_dk) and max_dk > 0) else 0.0
    else:
        inp["MDR_norm"] = np.nan

    if "Trajectory" in inp.columns:
        inp["Traj_score"] = _traj_score(inp["Trajectory"].fillna("Undetermined"))
    else:
        inp["Traj_score"] = np.nan

    tcs_list, ci_lo, ci_hi, n_comp_list = [], [], [], []

    for _, row in inp.iterrows():
        components = {
            "CRI":  (w_cri,  row.get("CRI_norm",  np.nan)),
            "TSW":  (w_tsw,  row.get("TSW_norm",  np.nan)),
            "SDQ":  (w_sdq,  row.get("SDQ_cons",  np.nan)),
            "MDR":  (w_mdr,  row.get("MDR_norm",  np.nan)),
            "Traj": (w_traj, row.get("Traj_score", np.nan)),
        }
        valid = {k: (w, v) for k, (w, v) in components.items() if pd.notna(v)}
        n_comp = len(valid)

        if n_comp < min_data_sources:
            tcs_list.append(np.nan); ci_lo.append(np.nan)
            ci_hi.append(np.nan); n_comp_list.append(n_comp)
            continue

        w_total = sum(w for w, _ in valid.values())
        tcs_raw = sum(w * v for w, v in valid.values()) / w_total * 100.0

        vals   = np.array([v for _, v in valid.values()])
        spread = float(np.std(vals)) if len(vals) > 1 else 0.0

        ci_lo.append(round(max(0.0,   tcs_raw - spread * 10), 2))
        ci_hi.append(round(min(100.0, tcs_raw + spread * 10), 2))
        tcs_list.append(round(tcs_raw, 2))
        n_comp_list.append(n_comp)

    inp["TCS"]           = tcs_list
    inp["TCS_CI_lower"]  = ci_lo
    inp["TCS_CI_upper"]  = ci_hi
    inp["n_components"]  = n_comp_list

    inp["TCS_CRI"]  = (inp["CRI_norm"].fillna(0.0)   * w_cri  * 100).round(2)
    inp["TCS_TSW"]  = (inp["TSW_norm"].fillna(0.0)   * w_tsw  * 100).round(2)
    inp["TCS_SDQ"]  = (inp["SDQ_cons"].fillna(0.0)   * w_sdq  * 100).round(2)
    inp["TCS_MDR"]  = (inp["MDR_norm"].fillna(0.0)   * w_mdr  * 100).round(2)
    inp["TCS_Traj"] = (inp["Traj_score"].fillna(0.0) * w_traj * 100).round(2)

    inp["TCS_label"] = inp["TCS"].apply(_tcs_label)
    inp["TCS_rank"]  = inp["TCS"].rank(ascending=False, method="min", na_option="bottom").astype(int)

    out_cols = [
        "Genotype", "Treatment", "Sex",
        "TCS", "TCS_CI_lower", "TCS_CI_upper",
        "TCS_CRI", "TCS_TSW", "TCS_SDQ", "TCS_MDR", "TCS_Traj",
        "n_components", "TCS_label", "TCS_rank",
        "CRI", "TSW", "SDQ",
    ]
    for extra in ["delta_k", "Rescue_pct", "Trajectory", "Archetype", "Cluster"]:
        if extra in inp.columns:
            out_cols.append(extra)

    out_cols = [c for c in out_cols if c in inp.columns]
    return inp[out_cols].sort_values("TCS_rank").reset_index(drop=True)


# ─── TCS summary table ────────────────────────────────────────────────────────
def tcs_summary_table(tcs_df: pd.DataFrame) -> pd.DataFrame:
    if tcs_df is None or tcs_df.empty:
        return pd.DataFrame()

    rows = []
    for (geno, trt), grp in tcs_df.groupby(["Genotype", "Treatment"], dropna=False):
        male_row   = grp.loc[grp["Sex"] == "Male",   "TCS"]
        female_row = grp.loc[grp["Sex"] == "Female", "TCS"]
        tcs_m    = float(male_row.mean())   if len(male_row)   > 0 else np.nan
        tcs_f    = float(female_row.mean()) if len(female_row) > 0 else np.nan
        tcs_mean = float(np.nanmean([v for v in [tcs_m, tcs_f] if pd.notna(v)]))
        sex_delta = abs(tcs_m - tcs_f) if pd.notna(tcs_m) and pd.notna(tcs_f) else np.nan
        concordant = bool(sex_delta < 15.0) if pd.notna(sex_delta) else False
        rows.append({
            "Genotype": geno, "Treatment": trt,
            "TCS_Male":    round(tcs_m, 2)         if pd.notna(tcs_m)    else np.nan,
            "TCS_Female":  round(tcs_f, 2)         if pd.notna(tcs_f)    else np.nan,
            "TCS_mean":    round(tcs_mean, 2)       if pd.notna(tcs_mean) else np.nan,
            "TCS_sex_delta": round(float(sex_delta), 2) if pd.notna(sex_delta) else np.nan,
            "Sex_concordant": concordant,
            "TCS_label": _tcs_label(tcs_mean),
        })

    return pd.DataFrame(rows).sort_values("TCS_mean", ascending=False).reset_index(drop=True)


# ─── Radar chart data ─────────────────────────────────────────────────────────
def tcs_radar_data(
    tcs_df: pd.DataFrame,
    top_n: int = 5,
) -> Tuple[List[str], Dict[str, List[float]]]:
    axes = ["CRI", "TSW", "SDQ Consistency", "MDR Rescue", "Trajectory"]
    top  = tcs_df.nsmallest(top_n, "TCS_rank")
    data: Dict[str, List[float]] = {}
    for _, row in top.iterrows():
        label = f"{row['Treatment']} / {row['Sex']}"
        data[label] = [
            float(row.get("TCS_CRI",  0) / (0.35 * 100) * 100),
            float(row.get("TCS_TSW",  0) / (0.25 * 100) * 100),
            float(row.get("TCS_SDQ",  0) / (0.15 * 100) * 100),
            float(row.get("TCS_MDR",  0) / (0.15 * 100) * 100),
            float(row.get("TCS_Traj", 0) / (0.10 * 100) * 100),
        ]
    return axes, data


# ─── Full TCS pipeline ────────────────────────────────────────────────────────
def run_tcs_pipeline(
    base_metrics: dict,
    novel_metrics: dict,
    mdr_df: Optional[pd.DataFrame] = None,
    cluster_result: Optional[dict] = None,     # FIX: accepted kwarg
    weights: Optional[Dict[str, float]] = None, # FIX: accepted kwarg for Recompute TCS
    wt_genotype: Optional[str] = None,
    vehicle_treatment: Optional[str] = None,
) -> Dict[str, object]:
    """
    Full TCS pipeline.
    FIX: accepts both cluster_result and weights keyword arguments
         so app.py run_tcs_pipeline(..., cluster_result=cr) and
         Recompute TCS button run_tcs_pipeline(..., weights=w) both work.
    """
    cri_df = novel_metrics.get("cri", pd.DataFrame())
    tsw_df = novel_metrics.get("tsw", pd.DataFrame())
    sdq_df = novel_metrics.get("sdq", pd.DataFrame())

    cluster_df = None
    if cluster_result is not None:
        cluster_df = cluster_result.get("cluster_table", None)

    if mdr_df is None:
        mdr_df = pd.DataFrame()

    tcs_full = compute_tcs(
        cri_df=cri_df,
        tsw_df=tsw_df,
        sdq_df=sdq_df,
        mdr_df=mdr_df,
        cluster_df=cluster_df,
        weights=weights,   # FIX: pass weights through
    )

    tcs_sum = tcs_summary_table(tcs_full)
    radar_axes, radar_data = tcs_radar_data(tcs_full) if not tcs_full.empty else ([], {})

    return {
        "tcs_full":    tcs_full,
        "tcs_summary": tcs_sum,
        "radar_axes":  radar_axes,
        "radar_data":  radar_data,
    }


# Compatibility alias
compute_translational_score = compute_tcs
