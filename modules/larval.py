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
    # Audit S2 finding 2.8: this used to fall back to `or 20.0`, so a container
    # reporting no frame rate produced speeds wrong by an unknown factor with
    # nothing in the output saying so. A speed with an invented time base is not
    # a measurement — refuse it and make the caller supply the true rate.
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not np.isfinite(fps_raw) or fps_raw <= 0:
        cap.release()
        raise ValueError(
            f"{video_path}: the container reports no usable frame rate "
            f"(CAP_PROP_FPS = {fps_raw}). Every speed and duration derived from "
            f"this video would be scaled by an unknown factor. Pass the true "
            f"frame rate explicitly rather than assuming one."
        )
    fps = fps_raw
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


PAUSE_SPEED_MM_S = 0.1   # absolute immobility threshold; see _trajectory_metrics
PAUSE_MIN_S      = 0.5   # a pause must last at least this long to be one pause

# Speed-curvature power law (Zago et al. 2016) — see speed_curvature_exponent.
POWER_LAW_CUTOFF_HZ = 0.07   # the cut-off used in the source paper


def _tls_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Total-least-squares (orthogonal) regression slope of y on x.

    Zago et al. estimate the power-law exponent by "least-squares
    orthogonal-regression of log10 A(t) versus log10 C(t)", NOT by ordinary
    least squares. The distinction matters: both axes are estimated from the
    same noisy trajectory, so OLS is attenuated toward zero by regression
    dilution. This is the first principal component's slope.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    sxx = float((x * x).mean()); syy = float((y * y).mean()); sxy = float((x * y).mean())
    if abs(sxy) < 1e-300:
        return float("nan")
    return float((syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy))


def speed_curvature_exponent(x_mm, y_mm, fps: float,
                             cutoff_hz: float = POWER_LAW_CUTOFF_HZ) -> dict:
    """
    Estimate the speed-curvature power-law exponent of a larval trajectory.

    Crawling and reaching movements across many species obey a power law
    relating angular speed A to path curvature C:

        A(t) = k * C(t) ** beta

    In human drawing beta is close to 2/3 (the "two-thirds power law"). Zago,
    Lacquaniti & Gomez-Marin (2016) "The speed-curvature power law in
    Drosophila larval locomotion", Biol Lett 12:20160597
    (doi:10.1098/rsbl.2016.0597) showed larvae obey it with beta nearer 3/4,
    which they attribute to dynamic effects on top of the kinematic constraint.

    This is the one larval read-out in DroPhenix with a QUANTITATIVE published
    reference value rather than only a direction, which is why it is worth
    computing: it is an external check that the trajectory differentiation is
    right, not just self-consistent.

    Method follows the paper exactly:
      1. positions in mm, sampled at `fps`
      2. second-order zero-phase Butterworth low-pass at `cutoff_hz`
         (filtfilt, so no phase lag)
      3. cubic-spline interpolation; first and second derivatives taken
         analytically from the spline
      4. standard differential geometry for a planar curve:
             v = hypot(x', y')
             C = |x' y'' - y' x''| / v**3        (curvature)
             A = |x' y'' - y' x''| / v**2        (angular speed, = C * v)
      5. total-least-squares regression of log10 A on log10 C

    Returns dict: beta, r2, k, n_points, cutoff_hz, fps.
    beta is NaN when the trajectory is too short or degenerate.

    VALIDATED against the authors' own raw data in
    validation/validate_larval_powerlaw.py: 123 real larvae across three
    behavioural conditions reproduce their published medians and IQRs.
    """
    from scipy.signal import butter, filtfilt
    from scipy.interpolate import CubicSpline

    base = {"beta": float("nan"), "r2": float("nan"), "k": float("nan"),
            "n_points": 0, "cutoff_hz": float(cutoff_hz), "fps": float(fps)}

    x = np.asarray(x_mm, float).ravel()
    y = np.asarray(y_mm, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 60 or not np.isfinite(fps) or fps <= 0:
        return base
    nyq = fps / 2.0
    if not (0 < cutoff_hz < nyq):
        return base

    b, a = butter(2, cutoff_hz / nyq, btype="low")
    # filtfilt needs more samples than its padlen; guard rather than raise.
    if len(x) <= 3 * max(len(a), len(b)):
        return base
    x = filtfilt(b, a, x)
    y = filtfilt(b, a, y)

    t = np.arange(len(x)) / float(fps)
    cx, cy = CubicSpline(t, x), CubicSpline(t, y)
    dx, dy = cx(t, 1), cy(t, 1)
    ddx, ddy = cx(t, 2), cy(t, 2)
    v = np.hypot(dx, dy)
    cross = np.abs(dx * ddy - dy * ddx)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = cross / v ** 3
        A = cross / v ** 2

    m = np.isfinite(C) & np.isfinite(A) & (C > 0) & (A > 0)
    if int(m.sum()) < 50:
        return {**base, "n_points": int(m.sum())}

    lc, la = np.log10(C[m]), np.log10(A[m])
    beta = _tls_slope(lc, la)
    r = float(np.corrcoef(lc, la)[0, 1])
    k = float(10.0 ** (la.mean() - beta * lc.mean())) if np.isfinite(beta) else float("nan")
    return {"beta": beta, "r2": r ** 2, "k": k, "n_points": int(m.sum()),
            "cutoff_hz": float(cutoff_hz), "fps": float(fps)}


def _count_pause_episodes(inst_speed: np.ndarray, dt: np.ndarray,
                          threshold: float, min_s: float) -> tuple:
    """
    Count RUNS of consecutive below-threshold frames lasting >= min_s, and the
    total time paused. Returns (n_episodes, total_paused_s).

    Audit S2 finding 2.7: the previous implementation returned
    `np.sum(inst_speed < 0.1 * median(inst_speed))`, which
      (a) counted FRAMES, not episodes — one continuous 5-second pause at 20 fps
          scored 99 — making the metric frame-rate dependent, and
      (b) used a threshold relative to that same animal's median speed, so a
          larva crawling twenty times slower reported the identical pause count
          as a fast one. A self-referential threshold cannot express absolute
          immobility, which is the whole point of the measure.
    """
    below = inst_speed < threshold
    n_ep, total = 0, 0.0
    i = 0
    while i < len(below):
        if not below[i]:
            i += 1
            continue
        j = i
        run = 0.0
        while j < len(below) and below[j]:
            run += float(dt[j]) if j < len(dt) else 0.0
            j += 1
        if run >= min_s:
            n_ep += 1
            total += run
        i = j
    return n_ep, total


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
    finite = np.isfinite(inst_speed_px)
    inst_speed_px = inst_speed_px[finite]
    dt_ok = dt[finite]
    scale = (1.0 / px_per_mm) if px_per_mm else np.nan   # mm per px
    mean_speed_px = path_px / dur if dur and dur > 0 else np.nan
    out = {
        "n_points": int(len(traj)),
        "duration_s": round(dur, 2) if np.isfinite(dur) else np.nan,
        "path_length_px": round(path_px, 1),
        "displacement_px": round(disp_px, 1),
        "mean_speed_px_s": round(mean_speed_px, 2) if np.isfinite(mean_speed_px) else np.nan,
        "tortuosity": round(path_px / disp_px, 3) if disp_px > 0 else np.nan,
    }
    if px_per_mm:
        out["path_length_mm"] = round(path_px * scale, 3)
        out["displacement_mm"] = round(disp_px * scale, 3)
        out["mean_speed_mm_s"] = round(mean_speed_px * scale, 3) if np.isfinite(mean_speed_px) else np.nan
        out["max_speed_mm_s"] = round(float(np.max(inst_speed_px)) * scale, 3) if len(inst_speed_px) else np.nan
        # Pauses need an ABSOLUTE speed threshold, which needs a spatial
        # calibration. Without px_per_mm the metric is not defined, and is
        # reported as NaN rather than computed against a self-referential
        # threshold that cannot be compared across animals.
        n_ep, paused_s = _count_pause_episodes(inst_speed_px * scale, dt_ok,
                                               PAUSE_SPEED_MM_S, PAUSE_MIN_S)
        out["n_pauses"] = int(n_ep)
        out["total_paused_s"] = round(float(paused_s), 2)
        out["pause_threshold_mm_s"] = PAUSE_SPEED_MM_S
        out["pause_min_s"] = PAUSE_MIN_S
    else:
        for k in ("path_length_mm", "displacement_mm", "mean_speed_mm_s", "max_speed_mm_s"):
            out[k] = np.nan
        out["n_pauses"] = np.nan
        out["total_paused_s"] = np.nan
        out["pause_threshold_mm_s"] = np.nan
        out["pause_min_s"] = np.nan
        out["pause_note"] = ("n_pauses requires px_per_mm: a pause is defined by an "
                             "absolute speed threshold in mm/s, not relative to this "
                             "animal's own median speed.")
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
