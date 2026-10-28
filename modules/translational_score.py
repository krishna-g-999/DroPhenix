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


# CRI is a percent-rescue relative to the disease-to-wildtype gap and is
# clipped to [-50, +150] upstream (metrics._rescue_fraction). Those are the
# FIXED anchors of the scale, so a CRI of 60 normalises to the same value in
# every dataset.
CRI_SCALE_LOW, CRI_SCALE_HIGH = -50.0, 150.0


# ─── Normalization helpers ────────────────────────────────────────────────────
def _norm_cri(series: pd.Series) -> pd.Series:
    """
    Normalise CRI onto [0, 1] against its own FIXED scale, not against the other
    compounds in the submitted table.

    Audit S1 finding 1.7. The previous `_minmax_norm` rescaled CRI within the
    dataset while `_tcs_label` applied ABSOLUTE tier thresholds, so the same
    compound at CRI = 60 scored TCS 84.50 "Tier 1" or 49.50 "Tier 3" purely
    according to which other compounds shared the file. A tier that moves when
    an unrelated compound is added or removed is not a property of the drug.
    """
    s = pd.to_numeric(series, errors="coerce").clip(CRI_SCALE_LOW, CRI_SCALE_HIGH)
    return (s - CRI_SCALE_LOW) / (CRI_SCALE_HIGH - CRI_SCALE_LOW)


def _norm_tsw(tsw_series: pd.Series) -> pd.Series:
    """TSW >= 2 is 'excellent selectivity'; that is the fixed ceiling."""
    s = pd.to_numeric(tsw_series, errors="coerce")
    return s.clip(0.0, 2.0) / 2.0


def _norm_sdq_consistency(sdq_series: pd.Series) -> pd.Series:
    """
    Sex consistency = 1 - SDQ. SDQ is on [0, 1] since the 2026-09 scale fix in
    metrics.compute_sdq (|dPI| / 200); before that its range was [0, 2] and this
    could go negative.
    """
    s = pd.to_numeric(sdq_series, errors="coerce").clip(0.0, 1.0)
    return 1.0 - s


def _traj_score(traj_series: pd.Series) -> pd.Series:
    """
    Map a trajectory label to [0, 1]. An UNRECOGNISED label returns NaN, not the
    0.2 "Undetermined" score it used to return silently — scoring a label the
    module does not understand as if it were a measured middling result is a
    fabricated component (audit S1 finding 1.7). NaN drops the component from
    the weighted mean instead, and `unknown_trajectory_labels` in the pipeline
    result names what was not recognised.
    """
    return traj_series.astype(str).map(lambda x: TRAJECTORY_RANK.get(x, np.nan))


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

    # Audit S1 finding 1.7: all-zero weights raised ZeroDivisionError.
    wsum = sum(weights.values())
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError(
            f"TCS weights must sum to a positive number; got {weights} "
            f"(sum = {wsum}). A zero total gives every component zero influence, "
            f"so there is no score to compute."
        )
    if any(v < 0 for v in weights.values()):
        raise ValueError(f"TCS weights must be non-negative; got {weights}.")
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

    inp["CRI_norm"]  = _norm_cri(inp["CRI"])
    inp["TSW_norm"]  = _norm_tsw(inp["TSW"])
    inp["SDQ_cons"]  = _norm_sdq_consistency(inp["SDQ"])

    # MDR: prefer Rescue_pct, which is already a percentage of the vehicle
    # decline rate and so has a fixed 0-100 scale. delta_k has no natural
    # ceiling, so normalising it by the dataset maximum would reintroduce the
    # dataset-relative scoring that finding 1.7 is about; that path is kept only
    # as a fallback and is flagged.
    mdr_relative = False
    if "Rescue_pct" in inp.columns and pd.to_numeric(inp["Rescue_pct"], errors="coerce").notna().any():
        inp["MDR_norm"] = (pd.to_numeric(inp["Rescue_pct"], errors="coerce") / 100.0).clip(0.0, 1.0)
    elif "delta_k" in inp.columns:
        dk = pd.to_numeric(inp["delta_k"], errors="coerce")
        max_dk = dk.max()
        if pd.notna(max_dk) and max_dk > 0:
            inp["MDR_norm"] = (dk / max_dk).clip(0.0, 1.0)
            mdr_relative = True
        else:
            inp["MDR_norm"] = np.nan
    else:
        inp["MDR_norm"] = np.nan

    unknown_labels = []
    if "Trajectory" in inp.columns:
        labels = inp["Trajectory"].fillna("Undetermined")
        inp["Traj_score"] = _traj_score(labels)
        unknown_labels = sorted({str(l) for l, v in zip(labels, inp["Traj_score"])
                                 if pd.isna(v)})
    else:
        inp["Traj_score"] = np.nan

    comp_cols = {"CRI": ("CRI_norm", w_cri), "TSW": ("TSW_norm", w_tsw),
                 "SDQ": ("SDQ_cons", w_sdq), "MDR": ("MDR_norm", w_mdr),
                 "Traj": ("Traj_score", w_traj)}

    tcs_list, spread_list, n_comp_list, used_list = [], [], [], []
    bar_rows = {k: [] for k in comp_cols}

    for _, row in inp.iterrows():
        valid = {k: (w, row.get(col, np.nan))
                 for k, (col, w) in comp_cols.items() if pd.notna(row.get(col, np.nan))}
        n_comp = len(valid)
        w_total = sum(w for w, _ in valid.values())

        if n_comp < min_data_sources or w_total <= 0:
            tcs_list.append(np.nan); spread_list.append(np.nan)
            n_comp_list.append(n_comp); used_list.append("; ".join(sorted(valid)) or "none")
            for k in comp_cols:
                bar_rows[k].append(np.nan)
            continue

        tcs_raw = sum(w * v for w, v in valid.values()) / w_total * 100.0
        vals = np.array([v for _, v in valid.values()])

        # Stacked component bars must use the SAME renormalised weights as the
        # score, or they do not add up to it. Previously they used the nominal
        # weights and filled missing components with 0, so a row with TCS 37.65
        # had bars summing to 32.00 (audit S1 finding 1.7). A component that is
        # missing is NaN here — an absent bar, not a zero-height one.
        for k, (col, w) in comp_cols.items():
            bar_rows[k].append(round(w / w_total * row.get(col, np.nan) * 100.0, 2)
                               if k in valid else np.nan)

        tcs_list.append(round(tcs_raw, 2))
        spread_list.append(round(float(np.std(vals)) if len(vals) > 1 else 0.0, 4))
        n_comp_list.append(n_comp)
        used_list.append("; ".join(sorted(valid)))

    inp["TCS"]          = tcs_list
    inp["n_components"] = n_comp_list
    inp["components_used"] = used_list

    # NOT a confidence interval (audit S1 finding 1.7). The old TCS_CI_lower /
    # TCS_CI_upper were `score +/- 10 * std(component values)`: a spread across
    # five quantities measured in different units, with the sample size never
    # entering. It could not narrow with more replicates and had no coverage
    # property. It is renamed to what it actually is — a disagreement measure
    # between the components — and the CI column names are gone so nothing
    # downstream can plot it as an interval.
    inp["TCS_component_disagreement"] = spread_list

    for k in comp_cols:
        inp[f"TCS_{k}"] = bar_rows[k]

    inp["TCS_label"] = inp["TCS"].apply(_tcs_label)
    inp["TCS_rank"]  = inp["TCS"].rank(ascending=False, method="min", na_option="bottom").astype(int)
    inp.attrs["mdr_dataset_relative"] = mdr_relative
    inp.attrs["unknown_trajectory_labels"] = unknown_labels
    inp.attrs["weights_used"] = {"CRI": w_cri, "TSW": w_tsw, "SDQ": w_sdq,
                                 "MDR": w_mdr, "Traj": w_traj}

    out_cols = [
        "Genotype", "Treatment", "Sex",
        "TCS", "TCS_component_disagreement",
        "TCS_CRI", "TCS_TSW", "TCS_SDQ", "TCS_MDR", "TCS_Traj",
        "n_components", "components_used", "TCS_label", "TCS_rank",
        "CRI", "TSW", "SDQ",
    ]
    for extra in ["delta_k", "Rescue_pct", "Trajectory", "Archetype", "Cluster"]:
        if extra in inp.columns:
            out_cols.append(extra)

    out_cols = [c for c in out_cols if c in inp.columns]
    out = inp[out_cols].sort_values("TCS_rank").reset_index(drop=True)
    out.attrs.update(inp.attrs)      # slicing drops attrs; carry them forward
    return out


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
    """
    Per-axis percentage of each component's own maximum, for a radar plot.

    Audit S1 finding 1.7: the divisors used to be the hard-coded DEFAULT_WEIGHTS
    (0.35, 0.25, 0.15, 0.15, 0.10). Running the pipeline with any other weights
    therefore mis-scaled every axis — with CRI weight 0.9 the CRI axis rendered
    at 257.1%. The weights the score was actually computed with are recorded in
    tcs_df.attrs["weights_used"], and are used here. Because the stacked
    components are now renormalised over the components present in each row, the
    divisor is that row's renormalised weight.
    """
    axes = ["CRI", "TSW", "SDQ Consistency", "MDR Rescue", "Trajectory"]
    keys = ["CRI", "TSW", "SDQ", "MDR", "Traj"]
    if tcs_df is None or tcs_df.empty:
        return axes, {}

    nominal = {**DEFAULT_WEIGHTS, **(tcs_df.attrs.get("weights_used") or {})}
    top = tcs_df.nsmallest(top_n, "TCS_rank")
    data: Dict[str, List[float]] = {}
    for _, row in top.iterrows():
        label = f"{row['Treatment']} / {row['Sex']}"
        # Recover this row's renormalisation factor from the score itself:
        # sum(TCS_*) == TCS, and each TCS_k == (w_k / w_total) * v_k * 100.
        present = [k for k in keys if pd.notna(row.get(f"TCS_{k}", np.nan))]
        w_total = sum(nominal.get(k, 0.0) for k in present)
        vals = []
        for k in keys:
            bar = row.get(f"TCS_{k}", np.nan)
            w = nominal.get(k, 0.0)
            if pd.isna(bar) or w <= 0 or w_total <= 0:
                vals.append(np.nan)          # component absent: no axis point
            else:
                vals.append(float(np.clip(bar / (w / w_total * 100.0) * 100.0, 0.0, 100.0)))
        data[label] = vals
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
        "weights_used": tcs_full.attrs.get("weights_used", {}) if not tcs_full.empty else {},
        # Labels the trajectory table used that TRAJECTORY_RANK does not know.
        # They are scored as NaN (component dropped), not silently as 0.2.
        "unknown_trajectory_labels": (tcs_full.attrs.get("unknown_trajectory_labels", [])
                                      if not tcs_full.empty else []),
        "mdr_dataset_relative": (tcs_full.attrs.get("mdr_dataset_relative", False)
                                 if not tcs_full.empty else False),
    }


# Compatibility alias
compute_translational_score = compute_tcs
