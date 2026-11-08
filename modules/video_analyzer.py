"""
DroPhenix Analytics — video_analyzer.py
Frame-level video analysis of Drosophila climbing assay recordings.

Capabilities:
- Extracts fly counts at specified timepoints from any video file.
- Supports both offline (recorded) video and batch processing.
- Background frame captured at a configurable warmup period.
- Outputs long-format DataFrame aligned with DroPhenix schema.
- QC metrics per timepoint: confidence, detection flags, background quality.
- Optional annotated frame export for visual validation.
- Batch processing with per-video error isolation.

Scientific notes:
- Standard DroPhenix assay timepoints: 10,20,30,40,50,60,70,80,90,100,120 s
- Background should be captured AFTER fly-tap when all flies are at bottom.
- BG_WARMUP_FRAMES controls how many frames are discarded before the
  background reference is locked. Longer warmup = more stable background.
- For accurate per-vial counts, ROIs must be calibrated to actual vial
  positions in the frame before analysis.

References:
  Feany & Bender (2000) Nature 404:394
  Nichols et al. (2012) J Vis Exp 61:e3795
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from modules.fly_detector import (
        DetectorConfig,
        FlyDetector,
        VialROI,
        VialCount,
        counts_to_dataframe,
    )
except ImportError:
    from modules.fly_detector import (
        DetectorConfig,
        FlyDetector,
        VialROI,
        VialCount,
        counts_to_dataframe,
    )

from modules import missingness as _MS

DEFAULT_TIMEPOINTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
BG_WARMUP_FRAMES = 30           # frames discarded before background is locked
BG_STABLE_FRAMES = 10           # frames averaged for stable background
MIN_USABLE_FRAMES = 5           # minimum valid frames per timepoint sampling


def get_video_info(video_path: str) -> dict:
    """
    Return metadata for a video file.
    Returns dict with: fps, frame_count, width, height, duration_sec, error.
    """
    # NOTE: use the default (FFMPEG) backend for FILES. CAP_DSHOW is a Windows
    # camera/DirectShow backend and cannot open video files (returns not-opened).
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return {"error": f"Cannot open: {video_path}"}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        fps = max(fps, 1.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": w,
            "height": h,
            "duration_sec": round(frame_count / fps, 2),
            "error": None,
        }
    finally:
        cap.release()


def build_sample_frame_map(
    timepoints_sec: List[int],
    fps: float,
) -> Dict[int, int]:
    """
    Map video frame indices to biological timepoints in seconds.
    Rounds to nearest frame for each timepoint.
    """
    return {int(round(tp * fps)): tp for tp in sorted(timepoints_sec)}


def _capture_background(
    cap: cv2.VideoCapture,
    warmup_frames: int = BG_WARMUP_FRAMES,
    stable_frames: int = BG_STABLE_FRAMES,
) -> Optional[np.ndarray]:
    """
    Discard warmup frames, then average stable_frames to build
    a stable background reference.
    Returns averaged background frame or None on failure.
    """
    for _ in range(warmup_frames):
        ret, _ = cap.read()
        if not ret:
            return None

    stack = []
    for _ in range(stable_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            stack.append(frame.astype(np.float32))

    if not stack:
        return None

    avg = np.mean(stack, axis=0).astype(np.uint8)
    return avg


def _median_background(cap: cv2.VideoCapture, n_samples: int = 15) -> Optional[np.ndarray]:
    """
    Robust background = pixelwise temporal MEDIAN of frames sampled across the
    whole clip. Because flies move (and climb), the per-pixel median converges
    to the empty vial — unlike averaging the first frames, which bakes in the
    flies sitting at the bottom right after the tap. This is the standard
    background-estimation approach for locomotor video.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        return None
    idxs = np.unique(np.linspace(0, total - 1, max(3, n_samples)).astype(int))
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, fr = cap.read()
        if ret and fr is not None:
            frames.append(fr.astype(np.float32))
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


class VideoAnalyzer:
    """
    Offline video analysis engine for DroPhenix climbing assay videos.

    Usage
    -----
    analyzer = VideoAnalyzer()
    df = analyzer.analyze(
        video_path="session_001.mp4",
        vials=vial_list,
        sample_timepoints=[10, 20, 30, 60, 120],
        replicate=1,
    )
    """

    def __init__(self, detector_config: Optional[DetectorConfig] = None):
        self.config = detector_config or DetectorConfig()
        self.detector = FlyDetector(self.config)

    def analyze(
        self,
        video_path: str,
        vials: List[VialROI],
        sample_timepoints: Optional[List[int]] = None,
        replicate: int = 1,
        session_id: str = "",
        label_map: Optional[Dict[int, str]] = None,
        export_annotated_frames: bool = False,
        annotated_dir: Optional[str] = None,
        bg_warmup_frames: int = BG_WARMUP_FRAMES,
        bg_stable_frames: int = BG_STABLE_FRAMES,
    ) -> pd.DataFrame:
        """
        Analyse a single video file and return fly counts per vial per timepoint.

        Parameters
        ----------
        video_path             : path to video (.mp4, .avi, .mov, .mkv)
        vials                  : list of VialROI objects
        sample_timepoints      : timepoints in seconds to sample (default: standard assay)
        replicate              : biological replicate number to tag in output
        session_id             : session identifier string
        label_map              : {vial_id: label} override dict
        export_annotated_frames: save annotated frame images for QC
        annotated_dir          : directory for annotated frame exports
        bg_warmup_frames       : frames to discard before locking background
        bg_stable_frames       : frames averaged for background reference

        Returns
        -------
        pd.DataFrame with DroPhenix long-format columns:
            Video, SessionID, Replicate, Time, VialID, Label,
            Genotype, Sex, Treatment, n_above, n_below, Total, N_detected,
            N_start, Pct, PI, QC_flag, Confidence, Missingness, unusable_reason

        `N_start` is the loading-record count carried from `VialROI.expected_n`;
        `N_detected` is what the detector found. Pct and PI are computed on
        `N_start` only (register N5), and are NaN — never 0.0 — wherever no
        usable measurement exists (register N8).
        """
        if sample_timepoints is None:
            sample_timepoints = DEFAULT_TIMEPOINTS

        video_path = str(video_path)
        info = get_video_info(video_path)
        if info.get("error"):
            raise RuntimeError(f"VideoAnalyzer: {info['error']}")

        fps = info["fps"]
        sample_frame_map = build_sample_frame_map(sample_timepoints, fps)
        max_needed_frame = max(sample_frame_map.keys()) if sample_frame_map else 0

        cap = cv2.VideoCapture(video_path)   # default FFMPEG backend for files
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            rows: List[dict] = []
            skipped_tps: List[int] = []

            # Background is only needed for mode='bg'. The default 'dark' mode
            # detects flies by intensity and ignores the background frame.
            bg_frame = None
            if getattr(self.config, "mode", "dark") == "bg":
                bg_frame = _median_background(cap)
                if bg_frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    bg_frame = _capture_background(cap, bg_warmup_frames, bg_stable_frames)
            if getattr(self.config, "mode", "dark") == "bg" and bg_frame is None:
                warnings.warn(
                    f"VideoAnalyzer: Background capture failed for {Path(video_path).name}. "
                    f"Falling back to Otsu thresholding.",
                    UserWarning,
                )

            # Reset to beginning for actual sampling
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            sampled_frame_indices = set(sample_frame_map.keys())

            while frame_idx <= max_needed_frame + int(fps * 2):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in sampled_frame_indices:
                    tp = sample_frame_map[frame_idx]

                    if bg_frame is None and frame_idx < bg_warmup_frames:
                        skipped_tps.append(tp)
                        frame_idx += 1
                        continue

                    counts = self.detector.count_in_vials(frame, vials, bg_frame)

                    # Apply label_map overrides if provided
                    if label_map:
                        for vc in counts:
                            if vc.vial_id in label_map:
                                vc.label = label_map[vc.vial_id]

                    # Export annotated frame if requested
                    if export_annotated_frames and annotated_dir:
                        ann_frame = self.detector.annotate_frame(frame, vials, counts)
                        ann_path = Path(annotated_dir) / f"{Path(video_path).stem}_t{tp:04d}s_rep{replicate}.jpg"
                        ann_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(ann_path), ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    for vc in counts:
                        row = vc.to_dict()
                        row["Video"] = Path(video_path).name
                        row["SessionID"] = session_id
                        row["Replicate"] = replicate
                        row["Time"] = tp
                        rows.append(row)

                frame_idx += 1

        finally:
            cap.release()

        if skipped_tps:
            warnings.warn(
                f"VideoAnalyzer: {len(skipped_tps)} timepoints skipped (background not ready): "
                f"{skipped_tps}. Reduce bg_warmup_frames or increase video pre-roll.",
                UserWarning,
            )

        # ── N7: a requested timepoint that produced nothing must still appear ──
        #
        # The sampling loop above breaks at end-of-stream, so a timepoint beyond
        # the end of the clip simply vanished: no row, no warning, nothing in the
        # output to say it had been asked for. Late timepoints are where a slow
        # climber's signal lives, so the set of timepoints actually analysed
        # ended up differing BY GENOTYPE, which is a confound rather than a gap.
        # `climbing_video.py` has emitted an explicit unusable row all along;
        # this is that behaviour, back-ported.
        #
        # The row carries NaN counts and a QC_flag, never a zero and never a
        # percentage: a timepoint that was not observed is not 0% climbed.
        observed_tps = {r["Time"] for r in rows}
        duration_s = info.get("duration_sec")
        if not duration_s:
            fc = info.get("frame_count", 0)
            duration_s = (fc / fps) if (fps and fc) else None
        for tp in sample_timepoints:
            if tp in observed_tps:
                continue
            if tp in skipped_tps:
                flag = "unusable:bg_warmup_skip"
                why = (f"background not ready at t={tp}s "
                       f"(first {bg_warmup_frames} frames reserved)")
            elif duration_s is not None and tp > duration_s:
                flag = "unusable:video_end_before_timepoint"
                why = f"requested t={tp}s but the video ends at {duration_s:.1f}s"
            else:
                flag = "unusable:frame_not_decoded"
                why = f"frame for t={tp}s could not be decoded"
            for v in vials:
                rows.append({
                    "Video": Path(video_path).name,
                    "SessionID": session_id,
                    "Replicate": replicate,
                    "Time": tp,
                    "VialID": v.vial_id,
                    "Label": (label_map or {}).get(v.vial_id, v.label),
                    "Genotype": v.genotype,
                    "Sex": v.sex,
                    "Treatment": v.treatment,
                    "n_above": np.nan,
                    "n_below": np.nan,
                    "Total": np.nan,
                    "N_detected": np.nan,
                    "N_start": int(getattr(v, "expected_n", 0) or 0),
                    "Pct": np.nan,
                    "PI": np.nan,
                    "QC_flag": flag,
                    "Confidence": 0.0,
                    # The timepoint was requested and never observed: missing by
                    # design of what the clip contains, not a failed attempt to
                    # score a frame that existed.
                    _MS.COLUMN: _MS.NOT_MEASURED,
                    "unusable_reason": why,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Reorder to DroPhenix standard
        preferred = [
            "Video", "SessionID", "Replicate", "Time",
            "VialID", "Label", "Genotype", "Sex", "Treatment",
            "n_above", "n_below", "Total", "N_detected", "N_start", "Pct", "PI",
            "QC_flag", "Confidence", _MS.COLUMN, "unusable_reason",
        ]
        extra = [c for c in df.columns if c not in preferred]
        df = df[[c for c in preferred if c in df.columns] + extra]
        # Unusable rows are appended after the loop, so re-sort: a reader
        # scanning the time course must see the gap in place, not at the end.
        if "Time" in df.columns and "VialID" in df.columns:
            df = df.sort_values(["Time", "VialID"], kind="stable")
        return df.reset_index(drop=True)

    def analyze_with_n(
        self,
        video_path: str,
        vials: List[VialROI],
        sample_timepoints: Optional[List[int]] = None,
        replicate: int = 1,
        session_id: str = "",
    ) -> pd.DataFrame:
        """
        Analyze video and convert to DroPhenix count schema.

        `n` is `VialROI.expected_n` — N_start from the loading record — and
        nothing else. It was `max(expected_n, detected_total)`, which raised the
        denominator to the detected count whenever the detector over-segmented,
        so an over-counted vial read as a tidy 100% instead of the >100% that
        would have shown the operator the ROI was wrong (register N5).
        """
        df_raw = self.analyze(
            video_path=video_path,
            vials=vials,
            sample_timepoints=sample_timepoints,
            replicate=replicate,
            session_id=session_id,
        )

        if df_raw.empty:
            return df_raw

        # Build N_start lookup from the loading record.
        n_map = {v.vial_id: int(v.expected_n or 0) for v in vials}
        no_record = sorted(vid for vid, n in n_map.items() if n <= 0)
        if no_record:
            raise ValueError(
                f"VideoAnalyzer.analyze_with_n: vials {no_record} have no "
                f"N_start (VialROI.expected_n). The number of flies loaded is "
                f"the only legitimate denominator for a climbing rate, so this "
                f"conversion is refused rather than computed against the "
                f"detected total. Use .analyze() if you want raw counts only."
            )

        df_out = df_raw.copy()
        # Count keeps its NaN where no measurement was made: `.astype(int)` here
        # would either raise or, worse, turn a failed vial into a zero.
        df_out["Count"] = pd.to_numeric(df_out["n_above"], errors="coerce")
        df_out["n"] = df_out["VialID"].map(n_map).astype("Int64")

        return df_out


def _resolve_replicate_map(video_paths, replicate_map) -> Dict[str, int]:
    """Match each video path to its declared replicate number, or refuse.

    A mapping may be keyed by full path or by file name — both are convenient
    and neither is ambiguous unless two paths share a basename, which is
    reported rather than resolved by guessing.
    """
    by_name: Dict[str, List[str]] = {}
    for vp in video_paths:
        by_name.setdefault(Path(str(vp)).name, []).append(str(vp))

    out: Dict[str, int] = {}
    unmapped = []
    for vp in video_paths:
        key_full, key_name = str(vp), Path(str(vp)).name
        if key_full in replicate_map:
            val = replicate_map[key_full]
        elif key_name in replicate_map:
            if len(by_name[key_name]) > 1:
                raise ValueError(
                    f"_resolve_replicate_map: {key_name!r} is the file name of "
                    f"{len(by_name[key_name])} different paths in this batch, "
                    f"so a name-keyed replicate_map is ambiguous. Key the "
                    f"mapping by full path instead."
                )
            val = replicate_map[key_name]
        else:
            unmapped.append(key_name)
            continue
        try:
            out[key_full] = int(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"_resolve_replicate_map: replicate for {key_name!r} is "
                f"{val!r}, which is not an integer replicate number."
            )
    if unmapped:
        raise ValueError(
            f"_resolve_replicate_map: no replicate declared for {unmapped}. "
            f"Every recording must be assigned to a biological replicate "
            f"explicitly; a recording with no declared replicate is refused "
            f"rather than given the next free number."
        )
    return out


class BatchVideoAnalyzer:
    """
    Process multiple video files as biological replicates **declared by the
    experimenter**, not numbered by file order (register N10).
    Errors on individual files are isolated and reported in output.
    """

    def __init__(self, detector_config: Optional[DetectorConfig] = None):
        self.analyzer = VideoAnalyzer(detector_config=detector_config)

    def analyze_batch(
        self,
        video_paths: List[str],
        vials: List[VialROI],
        sample_timepoints: Optional[List[int]] = None,
        session_prefix: str = "batch",
        replicate_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """
        Analyze a list of video files as biological replicates.

        Parameters
        ----------
        video_paths      : list of video file paths
        vials            : VialROI list (same for all videos)
        sample_timepoints: timepoints in seconds
        session_prefix   : prefix for session IDs
        replicate_map    : **required.** {file name or path: replicate number}

        Why `replicate_map` is required (register N10)
        ----------------------------------------------
        This used to number the files 1, 2, 3... in the order they arrived, so
        the replicate index was a property of the file list rather than of the
        experiment. Two camera angles of the same vial, two clips of one cohort
        split by a dropped frame, or a re-recording after a failed tap all
        became independent biological replicates — pseudoreplication by
        construction, which narrows every confidence interval and inflates
        every test. Only the experimenter knows which recordings share a
        cohort, so the mapping is supplied, not inferred. Giving two files the
        same number is the correct way to declare them one replicate.

        Returns
        -------
        combined_df : concatenated results with Replicate column
        error_log   : list of {video, replicate, error} for failed files
        """
        if not replicate_map:
            raise ValueError(
                "BatchVideoAnalyzer.analyze_batch: replicate_map is required. "
                "Replicate numbers are no longer assigned from file order — "
                "two recordings of one cohort are not two biological "
                "replicates, and auto-numbering made them so. Pass "
                "{file name: replicate number}; repeat a number to declare two "
                "files the same replicate."
            )
        resolved = _resolve_replicate_map(video_paths, replicate_map)

        frames = []
        error_log = []

        for vp in video_paths:
            i = resolved[str(vp)]
            sid = f"{session_prefix}_R{i:02d}"
            try:
                df = self.analyzer.analyze_with_n(
                    video_path=vp,
                    vials=vials,
                    sample_timepoints=sample_timepoints,
                    replicate=i,
                    session_id=sid,
                )
                df["SourceFile"] = Path(vp).name
                frames.append(df)
            except Exception as exc:
                error_log.append({
                    "video": Path(vp).name,
                    "replicate": i,
                    "error": str(exc),
                })

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return combined, error_log

    def summary_report(
        self,
        combined_df: pd.DataFrame,
        error_log: List[dict],
    ) -> dict:
        """
        Generate a QC summary report from batch analysis results.
        """
        if combined_df.empty:
            return {"status": "no_data", "errors": error_log}

        total_timepoints = len(combined_df)
        low_confidence = combined_df[combined_df.get("Confidence", pd.Series(dtype=float)) < 0.5] if "Confidence" in combined_df.columns else pd.DataFrame()
        flagged = combined_df[combined_df.get("QC_flag", pd.Series(dtype=str)) != "ok"] if "QC_flag" in combined_df.columns else pd.DataFrame()

        return {
            "status": "ok" if not error_log else "partial",
            "total_rows": total_timepoints,
            "videos_processed": combined_df["Replicate"].nunique() if "Replicate" in combined_df.columns else 0,
            "low_confidence_rows": len(low_confidence),
            "flagged_rows": len(flagged),
            "errors": error_log,
            "timepoints_covered": sorted(combined_df["Time"].dropna().astype(int).unique().tolist()) if "Time" in combined_df.columns else [],
        }


def video_counts_to_drophenix(
    df_raw: pd.DataFrame,
    vial_registry: Optional[dict] = None,
    replicate: int = 1,
) -> pd.DataFrame:
    """
    Convert VideoAnalyzer output to DroPhenix canonical long-format.

    Maps `n_above` → `Count` and **`N_start` → `n`**. It previously mapped
    `max(Total, Count)` → `n`, i.e. the detected total (register N5): a detector
    that missed five non-climbing flies in a vial of 30 reported 48.0% against a
    truth of 40.0%, and PI −4.0 against −20.0.

    The QC fields travel with the data or the conversion is refused (register
    N8). `QC_flag` and `Confidence` were emitted by the detector on every failed
    vial and then dropped here, one function downstream of the code that
    produced them — so the platform computed on `PI = 0.0` rows it had itself
    marked unusable.

    Returns DroPhenix-schema DataFrame with:
        Genotype, Sex, Treatment, Replicate, Time, Count, n,
        N_start, N_detected, QC_flag, Confidence, Missingness
    """
    if df_raw.empty:
        return pd.DataFrame()

    missing_qc = [c for c in ("QC_flag", "Confidence") if c not in df_raw.columns]
    if missing_qc:
        raise ValueError(
            f"video_counts_to_drophenix: {missing_qc} absent from the input. "
            f"The detector emits QC_flag and Confidence on every row, including "
            f"the rows it could not score; a frame that has lost them cannot be "
            f"converted safely, because a failed vial is indistinguishable from "
            f"a vial with no flies above the line. Refused."
        )

    df = df_raw.copy()

    if vial_registry is not None:
        for col in ["Genotype", "Sex", "Treatment"]:
            if col not in df.columns or df[col].eq("").all():
                df[col] = df["VialID"].map(
                    {vid: meta.get(col, "") for vid, meta in vial_registry.items()}
                )

    # NaN survives: a vial the detector could not score has no Count, and
    # `.astype(int)` here used to raise on it (or silently floor it to 0 once
    # the value had already been defaulted upstream).
    df["Count"] = pd.to_numeric(df["n_above"], errors="coerce")

    # Belt and braces at the boundary: whatever a producer put in `n_above`,
    # a row typed as a failed or absent measurement cannot carry a count. A
    # zero here against n = N_start computes PI = -100, a maximally impaired
    # vial, out of a vial that was never scored.
    if _MS.COLUMN in df.columns:
        not_obs = ~df[_MS.COLUMN].map(_MS.is_observed).fillna(False)
        # REFUSED keeps its counts: only the rate was withheld, so a row that
        # has since acquired an N_start is still usable.
        not_obs &= df[_MS.COLUMN].astype(str) != _MS.REFUSED
        df.loc[not_obs, "Count"] = np.nan

    # ── the denominator ──────────────────────────────────────────────────────
    n_start = None
    if "N_start" in df.columns:
        n_start = pd.to_numeric(df["N_start"], errors="coerce")
    if (n_start is None or n_start.isna().all() or (n_start.fillna(0) <= 0).all()) \
            and vial_registry is not None:
        reg_n = {}
        for vid, meta in vial_registry.items():
            for k in ("N_start", "n_start", "expected_n", "N", "n"):
                if isinstance(meta, dict) and k in meta:
                    try:
                        reg_n[vid] = int(meta[k])
                    except (TypeError, ValueError):
                        continue
                    break
        if reg_n:
            n_start = pd.to_numeric(df["VialID"].map(reg_n), errors="coerce")
    if n_start is None or n_start.isna().all() or (n_start.fillna(0) <= 0).all():
        raise ValueError(
            "video_counts_to_drophenix: no N_start (flies loaded) is available, "
            "either as an N_start column or from vial_registry. The detected "
            "total is not a substitute — using it shrinks the denominator by "
            "exactly the flies the detector missed, and a missed fly in a "
            "negative-geotaxis assay is usually one that did not climb, so "
            "every miss moves the vial toward the healthy phenotype. Refused."
        )

    df["N_start"] = n_start
    df["n"] = n_start
    if "N_detected" not in df.columns:
        df["N_detected"] = pd.to_numeric(df.get("Total"), errors="coerce")
    if _MS.COLUMN not in df.columns:
        # Derive the code rather than assume the row is a measurement.
        qc = df["QC_flag"].astype(str)
        df[_MS.COLUMN] = np.where(
            df["Count"].notna(), _MS.OBSERVED,
            np.where(qc.str.startswith("unusable") | qc.eq("not_measured"),
                     _MS.NOT_MEASURED, _MS.MEASUREMENT_FAILED))
    df["Replicate"] = df.get("Replicate", replicate)

    required = ["Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n",
                "N_start", "N_detected", "QC_flag", "Confidence", _MS.COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"video_counts_to_drophenix: missing columns after conversion: {missing}")

    extra = [c for c in df.columns if c not in required]
    return df[required + extra].reset_index(drop=True)