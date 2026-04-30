"""
DroPhenix — cv_pipeline.py
High-level CV pipeline wrapping VideoAnalyzer for single and batch analysis.
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd
from modules.video_analyzer import VideoAnalyzer
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
    ) -> pd.DataFrame:
        """
        Analyse multiple video files as sequential replicates.
        Errors on individual videos are caught and flagged in the output
        (Error column) rather than crashing the whole batch.
        """
        frames = []
        for i, vp in enumerate(video_paths, start=1):
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