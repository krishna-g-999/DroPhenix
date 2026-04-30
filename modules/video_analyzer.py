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

DEFAULT_TIMEPOINTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
BG_WARMUP_FRAMES = 30           # frames discarded before background is locked
BG_STABLE_FRAMES = 10           # frames averaged for stable background
MIN_USABLE_FRAMES = 5           # minimum valid frames per timepoint sampling


def get_video_info(video_path: str) -> dict:
    """
    Return metadata for a video file.
    Returns dict with: fps, frame_count, width, height, duration_sec, error.
    """
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_DSHOW)
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
            Genotype, Sex, Treatment, n_above, n_below, Total,
            Pct, PI, QC_flag, Confidence
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

        cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            rows: List[dict] = []
            skipped_tps: List[int] = []

            # Capture stable background
            bg_frame = _capture_background(cap, bg_warmup_frames, bg_stable_frames)
            if bg_frame is None:
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

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Reorder to DroPhenix standard
        preferred = [
            "Video", "SessionID", "Replicate", "Time",
            "VialID", "Label", "Genotype", "Sex", "Treatment",
            "n_above", "n_below", "Total", "Pct", "PI",
            "QC_flag", "Confidence",
        ]
        extra = [c for c in df.columns if c not in preferred]
        df = df[[c for c in preferred if c in df.columns] + extra]
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
        Uses VialROI.expected_n as 'n' for normalization.
        Returns DroPhenix long-format DataFrame with Count and n columns.
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

        # Build expected_n lookup from vials
        n_map = {v.vial_id: v.expected_n for v in vials}

        df_out = df_raw.copy()
        df_out["Count"] = df_out["n_above"]
        df_out["n"] = df_out["VialID"].map(n_map).fillna(df_out["Total"])
        df_out["n"] = df_out[["n", "Total"]].max(axis=1).astype(int)

        return df_out


class BatchVideoAnalyzer:
    """
    Process multiple video files as sequential biological replicates.
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
        start_replicate: int = 1,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """
        Analyze a list of video files.

        Parameters
        ----------
        video_paths      : list of video file paths
        vials            : VialROI list (same for all videos)
        sample_timepoints: timepoints in seconds
        session_prefix   : prefix for session IDs
        start_replicate  : starting replicate number

        Returns
        -------
        combined_df : concatenated results with Replicate column
        error_log   : list of {video, replicate, error} for failed files
        """
        frames = []
        error_log = []

        for i, vp in enumerate(video_paths, start=start_replicate):
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
    Maps n_above → Count, Total → n.
    Merges genotype/sex/treatment from vial_registry if provided.

    Returns DroPhenix-schema DataFrame with:
        Genotype, Sex, Treatment, Replicate, Time, Count, n
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    if vial_registry is not None:
        for col in ["Genotype", "Sex", "Treatment"]:
            if col not in df.columns or df[col].eq("").all():
                df[col] = df["VialID"].map(
                    {vid: meta.get(col, "") for vid, meta in vial_registry.items()}
                )

    df["Count"] = df["n_above"].astype(int)
    df["n"] = df[["Total", "Count"]].max(axis=1).astype(int)
    df["Replicate"] = df.get("Replicate", replicate)

    required = ["Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"video_counts_to_drophenix: missing columns after conversion: {missing}")

    extra = [c for c in df.columns if c not in required]
    return df[required + extra].reset_index(drop=True)