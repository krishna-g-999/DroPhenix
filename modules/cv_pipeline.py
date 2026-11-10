"""
DroPhenix — cv_pipeline.py
High-level CV pipeline wrapping VideoAnalyzer for single and batch analysis.
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd
from modules.video_analyzer import VideoAnalyzer, _resolve_replicate_map
from modules.fly_detector import VialROI, DetectorConfig


class CVPipeline:
    def __init__(self, detector_config: Optional[DetectorConfig] = None):
        self.analyzer = VideoAnalyzer(detector_config=detector_config)

    def analyze_video(
        self,
        video_path: str,
        vial_rois: List[VialROI],
        sample_timepoints: List[int],
        replicate: int = 1,
    ) -> pd.DataFrame:
        """Analyse a single video file. Returns long-format DataFrame."""
        return self.analyzer.analyze(
            video_path, vial_rois, sample_timepoints, replicate=replicate
        )

    def batch_analyze(
        self,
        video_paths: List[str],
        vial_rois: List[VialROI],
        sample_timepoints: List[int],
        replicate_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Analyse multiple video files as biological replicates.

        `replicate_map` ({file name or path: replicate number}) is required.
        Replicates were previously numbered from file order, which made two
        camera angles of one vial — or a cohort recorded in two clips — two
        independent biological replicates. That is pseudoreplication by
        construction; see register N10 and
        `video_analyzer._resolve_replicate_map`.

        Errors on individual videos are caught and flagged in the output
        (Error column) rather than crashing the whole batch.
        """
        if not replicate_map:
            raise ValueError(
                "CVPipeline.batch_analyze: replicate_map is required. "
                "Replicate numbers are no longer taken from file order; pass "
                "{file name: replicate number}, repeating a number where two "
                "recordings belong to the same biological replicate."
            )
        resolved = _resolve_replicate_map(video_paths, replicate_map)
        frames = []
        for vp in video_paths:
            i = resolved[str(vp)]
            try:
                df = self.analyzer.analyze(
                    vp, vial_rois, sample_timepoints, replicate=i
                )
                df["SourceFile"] = Path(vp).name
                df["Error"] = ""
                frames.append(df)
            except Exception as exc:
                err_df = pd.DataFrame([{
                    "Video": Path(vp).name,
                    "Replicate": i,
                    "SourceFile": Path(vp).name,
                    "Error": str(exc),
                }])
                frames.append(err_df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()