"""
DroPhenix — larval.py
Drosophila larval crawling (locomotion) assay engine.

Two entry points:
  * track_larva(video)  — centroid-track a single larva (dark on a light agar
    plate) frame-by-frame and derive locomotion metrics.
  * crawl metrics from an already-tracked trajectory or summary table.

Metrics (per larva):
  - path_length_mm     : total distance crawled
  - displacement_mm    : straight-line start->end distance
  - duration_s, mean_speed_mm_s, max_speed_mm_s
  - tortuosity         : path_length / displacement (1 = straight; higher = winding)
  - n_pauses           : low-speed episodes (proxy for stops/turns)

Scientific notes:
  - Wild-type third-instar larvae crawl on the order of ~1 mm/s (commonly
    reported 0.5-2 mm/s depending on substrate/temperature; e.g. the JoVE
    "Muscle Function Assessment Using a Drosophila Larvae Crawling Assay"
    protocol scores distance over a fixed 10-s window). Speeds therefore
    require a pixels-per-mm spatial calibration (px_per_mm).
  - The larva is detected as the largest dark object per frame (Otsu inverse
    threshold), the standard approach for a backlit/contrast plate.

Calibration of speed REQUIRES px_per_mm (pixels per millimetre). Without it,
metrics are reported in pixel units (mm columns are NaN).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

GROUP_KEYS = ["Genotype", "Treatment", "Sex"]


# ──────────────────────────────────────────────────────────────────────────────
# Video centroid tracking
# ──────────────────────────────────────────────────────────────────────────────
def track_larva(video_path: str, px_per_mm: Optional[float] = None,
                min_area: int = 30, max_frames: Optional[int] = None) -> dict:
    """
    Track a single larva's centroid through a video. Returns
      {'trajectory': DataFrame[frame, t_s, x_px, y_px], 'summary': {...}, 'fps': ...}
    The larva is the largest dark blob (Otsu inverse threshold) per frame.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))   # default backend for files
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 20.0
    rows = []
    fidx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and fidx >= max_frames):
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    rows.append({"frame": fidx, "t_s": fidx / fps, "x_px": cx, "y_px": cy})
            fidx += 1
    finally:
        cap.release()

    traj = pd.DataFrame(rows)
    summary = _trajectory_metrics(traj, fps, px_per_mm)
    return {"trajectory": traj, "summary": summary, "fps": fps}


def _trajectory_metrics(traj: pd.DataFrame, fps: float,
                        px_per_mm: Optional[float]) -> dict:
    """Locomotion metrics from a centroid trajectory."""
    if traj is None or len(traj) < 2:
        return {"n_points": 0}
    x = traj["x_px"].to_numpy(float); y = traj["y_px"].to_numpy(float)
    t = traj["t_s"].to_numpy(float)
    dx = np.diff(x); dy = np.diff(y); dt = np.diff(t)
    step_px = np.sqrt(dx ** 2 + dy ** 2)
    path_px = float(np.sum(step_px))
    disp_px = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    dur = float(t[-1] - t[0]) if len(t) > 1 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        inst_speed_px = step_px / dt
    inst_speed_px = inst_speed_px[np.isfinite(inst_speed_px)]
    scale = (1.0 / px_per_mm) if px_per_mm else np.nan   # mm per px
    mean_speed_px = path_px / dur if dur and dur > 0 else np.nan
    out = {
        "n_points": int(len(traj)),
        "duration_s": round(dur, 2) if np.isfinite(dur) else np.nan,
        "path_length_px": round(path_px, 1),
        "displacement_px": round(disp_px, 1),
        "mean_speed_px_s": round(mean_speed_px, 2) if np.isfinite(mean_speed_px) else np.nan,
        "tortuosity": round(path_px / disp_px, 3) if disp_px > 0 else np.nan,
        "n_pauses": int(np.sum(inst_speed_px < 0.1 * (np.median(inst_speed_px) + 1e-9))),
    }
    if px_per_mm:
        out["path_length_mm"] = round(path_px * scale, 3)
        out["displacement_mm"] = round(disp_px * scale, 3)
        out["mean_speed_mm_s"] = round(mean_speed_px * scale, 3) if np.isfinite(mean_speed_px) else np.nan
        out["max_speed_mm_s"] = round(float(np.max(inst_speed_px)) * scale, 3) if len(inst_speed_px) else np.nan
    else:
        for k in ("path_length_mm", "displacement_mm", "mean_speed_mm_s", "max_speed_mm_s"):
            out[k] = np.nan
    return out


def analyze_crawl_videos(specs: List[dict], px_per_mm: Optional[float] = None) -> pd.DataFrame:
    """
    Batch-track several larva videos.
    specs : list of {'path':..., 'Genotype':..., 'Treatment':..., 'Sex':..., 'FlyID':...}
    Returns one summary row per larva with group metadata.
    """
    rows = []
    for s in specs:
        try:
            res = track_larva(s["path"], px_per_mm=px_per_mm)
            rec = {k: s.get(k, "") for k in GROUP_KEYS + ["FlyID"]}
            rec.update(res["summary"])
            rows.append(rec)
        except Exception as e:
            rows.append({**{k: s.get(k, "") for k in GROUP_KEYS + ["FlyID"]}, "error": str(e)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Summary-table analysis (when crawling metrics already exist)
# ──────────────────────────────────────────────────────────────────────────────
def crawling_group_summary(per_larva: pd.DataFrame,
                           metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Per-group mean +/- SEM of crawling metrics."""
    if per_larva is None or per_larva.empty or "Genotype" not in per_larva.columns:
        return pd.DataFrame()
    if metrics is None:
        metrics = [m for m in ["mean_speed_mm_s", "path_length_mm", "displacement_mm",
                               "tortuosity", "mean_speed_px_s", "path_length_px"]
                   if m in per_larva.columns]
    rows = []
    for kv, sub in per_larva.groupby(GROUP_KEYS, sort=False):
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec["n_larvae"] = int(len(sub))
        for m in metrics:
            v = pd.to_numeric(sub[m], errors="coerce").dropna().to_numpy()
            rec[m] = round(float(np.mean(v)), 3) if len(v) else np.nan
            rec[m + "_SEM"] = round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 3) if len(v) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_groups(per_larva: pd.DataFrame, metric: str, group_col: str = "Treatment") -> dict:
    """Compare a crawling metric across groups via the shared stats engine."""
    from modules.stats_engine import auto_test
    if per_larva is None or metric not in per_larva.columns or group_col not in per_larva.columns:
        return {}
    data = {str(k): pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()
            for k, g in per_larva.groupby(group_col)}
    data = {k: v for k, v in data.items() if len(v) >= 2}
    if len(data) < 2:
        return {}
    return auto_test(data)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo (summary-level, literature-calibrated)
# ──────────────────────────────────────────────────────────────────────────────
def demo_larval(seed: int = 7) -> pd.DataFrame:
    """
    Synthetic per-larva crawling summaries. Wild-type ~1.0 mm/s; the TDP43 motor
    model is slower (~0.5 mm/s) with a more tortuous path; the drug partially
    rescues speed. Unequal N. (Speeds in the published ~0.5-2 mm/s range.)
    """
    rng = np.random.default_rng(seed)
    groups = {("Oregon-WT", "Vehicle"): 1.00, ("Oregon-WT", "Drug"): 1.02,
              ("TDP43", "Vehicle"): 0.52, ("TDP43", "Drug"): 0.78}
    rows = []
    fid = 0
    for (geno, treat), spd in groups.items():
        for sex in ("Male", "Female"):
            n = int(rng.integers(10, 16))
            for _ in range(n):
                v = max(0.05, rng.normal(spd, 0.12))
                dur = 60.0
                tort = rng.normal(1.15 if geno == "Oregon-WT" else 1.45, 0.1)
                path = v * dur
                rows.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                             "FlyID": f"larva{fid}", "duration_s": dur,
                             "mean_speed_mm_s": round(v, 3),
                             "path_length_mm": round(path, 2),
                             "displacement_mm": round(path / max(tort, 1.0), 2),
                             "tortuosity": round(max(tort, 1.0), 3)})
                fid += 1
    return pd.DataFrame(rows)


def run_larval_analysis(per_larva: pd.DataFrame) -> dict:
    return {"per_larva": per_larva, "summary": crawling_group_summary(per_larva)}
