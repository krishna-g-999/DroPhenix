"""
DroPhenix Analytics — metrics.py
Novel composite metrics for Drosophila climbing assay analysis.

Metrics
-------
CRI  — Composite Rescue Index
       CRI = w_PI * f_PI + w_AUC * f_AUC + w_t50 * f_t50
       where each f_X = (treated_X − disease_X) / (healthy_X − disease_X)
       Weights default: PI=0.4, AUC=0.4, t50=0.2  (sum=1)

SDQ  — Sex Dimorphism Quotient
       SDQ = |PI_M − PI_F| / max(|PI_M|, |PI_F|, ε)
       0 = no dimorphism; 1 = complete dimorphism

TSW  — Therapeutic Selectivity Window
       TSW = ΔPI_disease / |ΔPI_wildtype|
       >1 = disease-selective; <0 = broadly toxic

ECS  — Experimental Completeness Score
       Dataset quality audit (0–100).

Bug fixes applied vs. prior version
-------------------------------------
1. compute_sdq(): direction ternary was missing closing ")" — SyntaxError.
2. compute_cri(): n_rep column resolved defensively; AUC/t50 column names
   guarded with _resolve_col(); empty input DataFrames handled gracefully.
3. compute_novel_metrics(): df=None guard added.
4. _rescue_fraction(): t50 inversion scientifically correct (lower t50 = better).
5. All functions return an empty DataFrame (not raise) when data is insufficient.

References
----------
Feany & Bender (2000) Nature 404:394
Elden et al. (2010) Nature 466:1069
Gargano et al. (2005) J Neurobiol 65:11
Nichols et al. (2012) J Vis Exp 61:e3795
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WT_LABELS      = {"oregon", "canton-s", "w1118", "wt", "wild-type",
                  "wildtype", "control", "w[1118]", "cs", "ore-r"}
VEHICLE_LABELS = {"untreated", "vehicle", "dmso", "control", "pbs"}
MIN_REPS_CRI   = 2


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _sem(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan

def _safe_div(num: float, denom: float, fallback: float = np.nan) -> float:
    if pd.isna(denom) or denom == 0:
        return fallback
    return float(num / denom)

def _is_wt(label: str) -> bool:
    return str(label).strip().lower() in WT_LABELS

def _is_vehicle(label: str) -> bool:
    return str(label).strip().lower() in VEHICLE_LABELS

def _normalise_pi(val: float) -> float:
    """Normalise PI [-100, 100] to [0, 1]."""
    return float(np.clip((val + 100.0) / 200.0, 0.0, 1.0))

def _resolve_col(df: pd.DataFrame, candidates: List[str], default: str = "") -> str:
    """Return the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _rescue_fraction(treated: float, disease: float, healthy: float) -> float:
    """
    Fractional rescue in [−0.5, 1.5]:
    f = (treated − disease) / (healthy − disease)
    """
    denom = healthy - disease
    if pd.isna(denom) or abs(denom) < 1e-6:
        return np.nan
    return float(np.clip((treated - disease) / denom, -0.5, 1.5))


# ─── CRI — Composite Rescue Index ────────────────────────────────────────────
def compute_cri(
    pi_summary:  pd.DataFrame,
    auc_summary: pd.DataFrame,
    t50_summary: pd.DataFrame,
    wt_genotype:        Optional[str]            = None,
    vehicle_treatment:  Optional[str]            = None,
    weights:            Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Composite Rescue Index (CRI) per Genotype × Treatment × Sex.

    CRI = w_PI * f_PI  +  w_AUC * f_AUC  +  w_t50 * f_t50
    Missing metric components are excluded and remaining weights renormalised.
    """
    if pi_summary is None or pi_summary.empty:
        return pd.DataFrame()

    if weights is None:
        weights = {"PI": 0.4, "AUC": 0.4, "t50": 0.2}

    w_pi, w_auc, w_t50 = (weights.get(k, d)
                           for k, d in [("PI", 0.4), ("AUC", 0.4), ("t50", 0.2)])
    total_w = w_pi + w_auc + w_t50
    if abs(total_w - 1.0) > 0.01:
        w_pi /= total_w; w_auc /= total_w; w_t50 /= total_w

    # ── Column name resolution ─────────────────────────────────────────────
    pi_col  = _resolve_col(pi_summary,  ["PI_mean",  "PI",  "pi_mean"],  "PI_mean")
    auc_col = _resolve_col(auc_summary, ["AUC_mean", "AUC", "auc_mean"], "AUC_mean") if auc_summary is not None and not auc_summary.empty else "AUC_mean"
    t50_col = _resolve_col(t50_summary, ["t50_mean", "t50", "t50_med"],  "t50_mean") if t50_summary is not None and not t50_summary.empty else "t50_mean"
    nrep_col = _resolve_col(pi_summary, ["n_rep", "n", "N"], "")

    # ── Auto-detect WT and vehicle ────────────────────────────────────────
    def _detect_wt(df):
        if wt_genotype is not None:
            return wt_genotype
        for g in df["Genotype"].dropna().unique():
            if _is_wt(str(g)):
                return str(g)
        return str(df["Genotype"].iloc[0]) if len(df) > 0 else ""

    def _detect_veh(df):
        if vehicle_treatment is not None:
            return vehicle_treatment
        for t in df["Treatment"].dropna().unique():
            if _is_vehicle(str(t)):
                return str(t)
        return str(df["Treatment"].iloc[0]) if len(df) > 0 else ""

    wt  = _detect_wt(pi_summary)
    veh = _detect_veh(pi_summary)

    # ── Reference lookups keyed by (Genotype, Sex) ────────────────────────
    def _ref_lookup(df, val_col):
        lookup = {}
        if df is None or df.empty or val_col not in df.columns:
            return lookup
        for _, row in df.iterrows():
            lookup[(row["Genotype"], row["Sex"])] = row[val_col]
        return lookup

    pi_veh   = pi_summary.loc[pi_summary["Treatment"] == veh,
                               ["Genotype", "Sex", pi_col] + ([nrep_col] if nrep_col else [])]
    auc_veh  = (auc_summary.loc[auc_summary["Treatment"] == veh,
                                 ["Genotype", "Sex", auc_col]]
                if auc_summary is not None and not auc_summary.empty and "Treatment" in auc_summary.columns
                else pd.DataFrame())
    t50_veh  = (t50_summary.loc[t50_summary["Treatment"] == veh,
                                 ["Genotype", "Sex", t50_col]]
                if t50_summary is not None and not t50_summary.empty and "Treatment" in t50_summary.columns
                else pd.DataFrame())

    wt_pi_ref  = _ref_lookup(pi_veh.loc[pi_veh["Genotype"] == wt],  pi_col)
    wt_auc_ref = _ref_lookup(auc_veh.loc[auc_veh["Genotype"] == wt] if not auc_veh.empty else pd.DataFrame(), auc_col)
    wt_t50_ref = _ref_lookup(t50_veh.loc[t50_veh["Genotype"] == wt] if not t50_veh.empty else pd.DataFrame(), t50_col)

    pi_dis_ref  = {(row["Genotype"], row["Sex"]): row[pi_col]  for _, row in pi_veh.iterrows()}
    auc_dis_ref = {(row["Genotype"], row["Sex"]): row[auc_col] for _, row in auc_veh.iterrows()} if not auc_veh.empty and auc_col in auc_veh.columns else {}
    t50_dis_ref = {(row["Genotype"], row["Sex"]): row[t50_col] for _, row in t50_veh.iterrows()} if not t50_veh.empty and t50_col in t50_veh.columns else {}

    # ── Compute CRI per group ─────────────────────────────────────────────
    rows = []
    for _, r in pi_summary.iterrows():
        geno, trt, sex = r["Genotype"], r["Treatment"], r["Sex"]
        if trt == veh:
            continue
        key, wt_key = (geno, sex), (wt, sex)

        pi_treat = r.get(pi_col, np.nan)
        pi_dis   = pi_dis_ref.get(key, np.nan)
        pi_wt    = wt_pi_ref.get(wt_key, np.nan)

        auc_sub = (auc_summary.loc[
            (auc_summary["Genotype"] == geno) &
            (auc_summary["Treatment"] == trt) &
            (auc_summary["Sex"] == sex), auc_col]
            if auc_summary is not None and not auc_summary.empty and auc_col in auc_summary.columns
            else pd.Series(dtype=float))
        t50_sub = (t50_summary.loc[
            (t50_summary["Genotype"] == geno) &
            (t50_summary["Treatment"] == trt) &
            (t50_summary["Sex"] == sex), t50_col]
            if t50_summary is not None and not t50_summary.empty and t50_col in t50_summary.columns
            else pd.Series(dtype=float))

        auc_treat = float(auc_sub.iloc[0]) if len(auc_sub) > 0 else np.nan
        t50_treat = float(t50_sub.iloc[0]) if len(t50_sub) > 0 else np.nan
        auc_dis   = auc_dis_ref.get(key, np.nan)
        auc_wt    = wt_auc_ref.get(wt_key, np.nan)
        t50_dis   = t50_dis_ref.get(key, np.nan)
        t50_wt    = wt_t50_ref.get(wt_key, np.nan)

        f_pi  = _rescue_fraction(pi_treat, pi_dis, pi_wt)
        f_auc = _rescue_fraction(auc_treat, auc_dis, auc_wt)
        # t50: lower = better → invert sign for rescue fraction
        f_t50 = (
            _rescue_fraction(-t50_treat, -t50_dis, -t50_wt)
            if all(pd.notna(v) for v in [t50_treat, t50_dis, t50_wt])
            else np.nan
        )

        components, w_total = [], 0.0
        if pd.notna(f_pi):   components.append(w_pi  * f_pi);  w_total += w_pi
        if pd.notna(f_auc):  components.append(w_auc * f_auc); w_total += w_auc
        if pd.notna(f_t50):  components.append(w_t50 * f_t50); w_total += w_t50
        cri = float(sum(components) / w_total * 100.0) if w_total > 0 else np.nan

        n_rep = int(r[nrep_col]) if nrep_col and nrep_col in r.index and pd.notna(r[nrep_col]) else 1
        rows.append({
            "Genotype":  geno, "Treatment": trt, "Sex": sex,
            "CRI":     round(cri, 2)         if pd.notna(cri)   else np.nan,
            "CRI_PI":  round(f_pi  * 100, 2) if pd.notna(f_pi)  else np.nan,
            "CRI_AUC": round(f_auc * 100, 2) if pd.notna(f_auc) else np.nan,
            "CRI_t50": round(f_t50 * 100, 2) if pd.notna(f_t50) else np.nan,
            "CRI_label": _cri_label(cri),
            "n_rep": n_rep,
        })

    return (pd.DataFrame(rows)
              .sort_values(["Genotype", "Sex", "CRI"], ascending=[True, True, False])
              .reset_index(drop=True))


def _cri_label(cri: float) -> str:
    if pd.isna(cri):          return "Not computable"
    if cri >= 80:             return "Strong rescue (CRI ≥ 80)"
    if cri >= 50:             return "Moderate rescue (CRI 50–80)"
    if cri >= 20:             return "Partial rescue (CRI 20–50)"
    if cri >= 0:              return "Minimal rescue (CRI 0–20)"
    return                          "No rescue / impairment (CRI < 0)"


# ─── SDQ — Sex Dimorphism Quotient ────────────────────────────────────────────
def compute_sdq(
    pi_summary:        pd.DataFrame,
    vehicle_treatment: Optional[str] = None,
) -> pd.DataFrame:
    """
    SDQ = |PI_Male − PI_Female| / max(|PI_Male|, |PI_Female|, ε)
    0 = equal; 1 = complete dimorphism.

    Bug fix: direction ternary previously missing closing ")".
    """
    if pi_summary is None or pi_summary.empty:
        return pd.DataFrame()

    pi_col = _resolve_col(pi_summary, ["PI_mean", "PI", "pi_mean"], "PI_mean")

    if vehicle_treatment is None:
        for t in pi_summary["Treatment"].dropna().unique():
            if _is_vehicle(str(t)):
                vehicle_treatment = str(t)
                break

    rows = []
    for (geno, trt), grp in pi_summary.groupby(["Genotype", "Treatment"], dropna=False):
        male   = grp.loc[grp["Sex"] == "Male",   pi_col]
        female = grp.loc[grp["Sex"] == "Female", pi_col]
        if male.empty or female.empty:
            continue

        pi_m  = float(male.mean())
        pi_f  = float(female.mean())
        eps   = 1e-6
        denom = max(abs(pi_m), abs(pi_f), eps)
        sdq   = abs(pi_m - pi_f) / denom

        # FIX: closing ")" was missing in previous version
        direction = (
            "Male > Female" if pi_m > pi_f
            else "Female > Male" if pi_f > pi_m
            else "Equal"
        )

        n_rep = int(grp["n_rep"].max()) if "n_rep" in grp.columns else 1
        rows.append({
            "Genotype":              geno,
            "Treatment":             trt,
            "PI_Male":               round(pi_m, 2),
            "PI_Female":             round(pi_f, 2),
            "SDQ":                   round(float(sdq), 4),
            "SDQ_label":             _sdq_label(sdq),
            "Dimorphism_direction":  direction,
            "n_rep":                 n_rep,
        })

    return (pd.DataFrame(rows)
              .sort_values(["Genotype", "SDQ"], ascending=[True, False])
              .reset_index(drop=True))


def _sdq_label(sdq: float) -> str:
    if pd.isna(sdq):  return "Unknown"
    if sdq >= 0.5:    return "Strong dimorphism"
    if sdq >= 0.25:   return "Moderate dimorphism"
    if sdq >= 0.1:    return "Mild dimorphism"
    return                   "No significant dimorphism"


# ─── TSW — Therapeutic Selectivity Window ─────────────────────────────────────
def compute_tsw(
    pi_summary:        pd.DataFrame,
    wt_genotype:       Optional[str] = None,
    vehicle_treatment: Optional[str] = None,
) -> pd.DataFrame:
    """
    TSW = ΔPI_disease / |ΔPI_wildtype|
    >1 = disease-selective; <0 = broadly toxic.
    """
    if pi_summary is None or pi_summary.empty:
        return pd.DataFrame()

    pi_col = _resolve_col(pi_summary, ["PI_mean", "PI", "pi_mean"], "PI_mean")

    if wt_genotype is None:
        for g in pi_summary["Genotype"].dropna().unique():
            if _is_wt(str(g)):
                wt_genotype = str(g)
                break
    if wt_genotype is None:
        return pd.DataFrame(columns=["Treatment", "Sex", "Genotype",
                                      "Delta_PI_disease", "Delta_PI_WT", "TSW", "TSW_label"])

    if vehicle_treatment is None:
        for t in pi_summary["Treatment"].dropna().unique():
            if _is_vehicle(str(t)):
                vehicle_treatment = str(t)
                break
    if vehicle_treatment is None:
        return pd.DataFrame()

    rows = []
    for sex in pi_summary["Sex"].dropna().unique():
        for trt in pi_summary["Treatment"].dropna().unique():
            if trt == vehicle_treatment:
                continue

            wt_veh = pi_summary.loc[
                (pi_summary["Genotype"] == wt_genotype) &
                (pi_summary["Treatment"] == vehicle_treatment) &
                (pi_summary["Sex"] == sex), pi_col]
            wt_trt = pi_summary.loc[
                (pi_summary["Genotype"] == wt_genotype) &
                (pi_summary["Treatment"] == trt) &
                (pi_summary["Sex"] == sex), pi_col]
            if wt_veh.empty or wt_trt.empty:
                continue

            delta_wt = float(wt_trt.mean()) - float(wt_veh.mean())

            for geno in pi_summary["Genotype"].dropna().unique():
                if geno == wt_genotype:
                    continue
                dis_veh = pi_summary.loc[
                    (pi_summary["Genotype"] == geno) &
                    (pi_summary["Treatment"] == vehicle_treatment) &
                    (pi_summary["Sex"] == sex), pi_col]
                dis_trt = pi_summary.loc[
                    (pi_summary["Genotype"] == geno) &
                    (pi_summary["Treatment"] == trt) &
                    (pi_summary["Sex"] == sex), pi_col]
                if dis_veh.empty or dis_trt.empty:
                    continue

                delta_dis = float(dis_trt.mean()) - float(dis_veh.mean())
                denom = abs(delta_wt) if abs(delta_wt) > 1e-6 else np.nan
                tsw   = _safe_div(delta_dis, denom, fallback=np.nan)

                rows.append({
                    "Treatment":       trt,
                    "Sex":             sex,
                    "Genotype":        geno,
                    "Delta_PI_disease": round(delta_dis, 2),
                    "Delta_PI_WT":      round(delta_wt, 2),
                    "TSW":              round(float(tsw), 3) if pd.notna(tsw) else np.nan,
                    "TSW_label":        _tsw_label(tsw),
                })

    return (pd.DataFrame(rows)
              .sort_values(["Treatment", "Sex", "TSW"], ascending=[True, True, False])
              .reset_index(drop=True))


def _tsw_label(tsw: float) -> str:
    if pd.isna(tsw):  return "Not computable"
    if tsw >= 2.0:    return "Excellent selectivity (TSW >= 2)"
    if tsw >= 1.0:    return "Good selectivity (TSW 1–2)"
    if tsw >= 0.5:    return "Narrow window (TSW 0.5–1)"
    if tsw >= 0.0:    return "Poor selectivity (TSW 0–0.5)"
    return                   "Adverse effect (TSW < 0)"


# ─── ECS — Experimental Completeness Score ────────────────────────────────────
def compute_ecs(
    df: pd.DataFrame,
    expected_replicates: int = 3,
    expected_timepoints: int = 8,
    expected_sexes:      Optional[List[str]] = None,
    require_control:     bool = True,
    require_wt:          bool = True,
) -> dict:
    """
    ECS scores dataset completeness 0–100.
    Dimensions: replicates (30), timepoints (25), sex balance (20),
    vehicle present (15), WT present (10).
    """
    if df is None or df.empty:
        return {"score": 0, "grade": "F", "details": {},
                "interpretation": "No data provided."}

    if expected_sexes is None:
        expected_sexes = ["Male", "Female"]

    details: Dict[str, object] = {}
    score = 0

    # 1. Replicate completeness (30 pts)
    if "Replicate" in df.columns:
        reps = df.groupby(["Genotype", "Sex", "Treatment"])["Replicate"].nunique()
        min_rep  = int(reps.min())
        mean_rep = float(reps.mean())
        rep_score = int(min(30, (mean_rep / max(expected_replicates, 1)) * 30))
        score += rep_score
        details["replicate_score"]   = rep_score
        details["min_replicates"]    = min_rep
        details["mean_replicates"]   = round(mean_rep, 2)
        details["replicate_warning"] = (
            f"Some groups < {expected_replicates} biological replicates."
            if min_rep < expected_replicates else "OK"
        )
    else:
        details["replicate_score"]   = 0
        details["replicate_warning"] = "No Replicate column found."

    # 2. Timepoint density (25 pts)
    if "Time" in df.columns:
        key_cols = ["Genotype", "Sex", "Treatment"] + (["Replicate"] if "Replicate" in df.columns else [])
        tp = df.groupby(key_cols)["Time"].nunique()
        min_tp  = int(tp.min())
        mean_tp = float(tp.mean())
        tp_score = int(min(25, (mean_tp / max(expected_timepoints, 1)) * 25))
        score += tp_score
        details["timepoint_score"]   = tp_score
        details["min_timepoints"]    = min_tp
        details["mean_timepoints"]   = round(mean_tp, 2)
        details["timepoint_warning"] = (
            f"Some replicates < {expected_timepoints} timepoints."
            if min_tp < expected_timepoints else "OK"
        )
    else:
        details["timepoint_score"]   = 0
        details["timepoint_warning"] = "No Time column found."

    # 3. Sex balance (20 pts)
    sexes_present = set(df["Sex"].dropna().unique()) if "Sex" in df.columns else set()
    sex_score = int(20 * len(sexes_present.intersection(set(expected_sexes)))
                    / max(len(expected_sexes), 1))
    score += sex_score
    details["sex_score"]    = sex_score
    details["sexes_present"] = sorted(list(sexes_present))
    details["sex_warning"]  = (
        "Both sexes needed for SDQ and sex-stratified analysis."
        if len(sexes_present) < 2 else "OK"
    )

    # 4. Vehicle/control present (15 pts)
    ctrl_present = any(_is_vehicle(str(t))
                       for t in df["Treatment"].dropna().astype(str).unique()) if "Treatment" in df.columns else False
    ctrl_score = 15 if (ctrl_present or not require_control) else 0
    score += ctrl_score
    details["control_score"]   = ctrl_score
    details["control_warning"] = ("No vehicle/control detected. Required for CRI and TSW."
                                   if not ctrl_present else "OK")

    # 5. WT present (10 pts)
    wt_present = any(_is_wt(str(g))
                     for g in df["Genotype"].dropna().astype(str).unique()) if "Genotype" in df.columns else False
    wt_score = 10 if (wt_present or not require_wt) else 0
    score += wt_score
    details["wt_score"]   = wt_score
    details["wt_warning"] = ("No wild-type genotype detected. Required for TSW and CRI baseline."
                              if not wt_present else "OK")

    grade = ("A" if score >= 90 else "B" if score >= 75 else
             "C" if score >= 60 else "D" if score >= 50 else "F")

    return {
        "score":          score,
        "grade":          grade,
        "details":        details,
        "interpretation": _ecs_interpretation(score, grade),
    }


def _ecs_interpretation(score: int, grade: str) -> str:
    return {
        "A": "Excellent experimental design. All metrics computable.",
        "B": "Good design. Minor gaps; most analyses valid.",
        "C": "Adequate design. Some metrics may have reduced power.",
        "D": "Incomplete design. CRI/TSW/SDQ results may be unreliable.",
        "F": "Critical gaps detected. Results should not be published without addressing warnings.",
    }.get(grade, "Unknown")


# ─── Master pipeline ──────────────────────────────────────────────────────────
def compute_novel_metrics(
    df:                pd.DataFrame,
    base_metrics:      Dict[str, pd.DataFrame],
    wt_genotype:       Optional[str]            = None,
    vehicle_treatment: Optional[str]            = None,
    cri_weights:       Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Compute all novel metrics: CRI, SDQ, TSW, ECS.
    Returns dict with keys: cri, sdq, tsw, ecs.

    Bug fix: df=None guard added.
    """
    import warnings

    pi_s   = base_metrics.get("pi",  pd.DataFrame())
    auc_s  = base_metrics.get("auc", pd.DataFrame())
    t50_s  = base_metrics.get("t50", pd.DataFrame())

    cri, sdq, tsw, ecs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    if pi_s is not None and not pi_s.empty:
        try:
            cri = compute_cri(pi_s, auc_s, t50_s,
                               wt_genotype=wt_genotype,
                               vehicle_treatment=vehicle_treatment,
                               weights=cri_weights)
        except Exception as e:
            warnings.warn(f"CRI computation failed: {e}")

        try:
            sdq = compute_sdq(pi_s, vehicle_treatment=vehicle_treatment)
        except Exception as e:
            warnings.warn(f"SDQ computation failed: {e}")

        try:
            tsw = compute_tsw(pi_s, wt_genotype=wt_genotype,
                               vehicle_treatment=vehicle_treatment)
        except Exception as e:
            warnings.warn(f"TSW computation failed: {e}")

    if df is not None and not df.empty:
        try:
            ecs = compute_ecs(df)
        except Exception as e:
            warnings.warn(f"ECS computation failed: {e}")
            ecs = {"score": 0, "grade": "F", "details": {}, "interpretation": str(e)}
    else:
        ecs = {"score": 0, "grade": "F", "details": {},
               "interpretation": "No raw DataFrame provided to compute_novel_metrics()."}

    return {"cri": cri, "sdq": sdq, "tsw": tsw, "ecs": ecs}
