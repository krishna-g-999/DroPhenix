"""
DroPhenix — recorder.py
Camera management and timed-capture engine for automated climbing assays.

Responsibilities:
  - Open/close any camera (USB webcam, built-in, IP stream)
  - Capture background reference frame (post-tap, flies at bottom)
  - Execute the DroPhenix timing protocol automatically:
      tap → wait settle_s → capture at each timepoint
  - Count flies using FlyDetector at each timepoint
  - Save raw_counts CSV and optional MP4 video
  - Provide live_frame for Streamlit preview

Scientific note:
  settle_time allows vortexed flies to settle before assay starts.
  The background frame is captured after vortexing so background
  subtraction removes static vial features, not fly positions.
  Standard DroPhenix timepoints: 10,20,30,40,50,60,70,80,90,100,120 s.
"""

import cv2
import time
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Union, List

try:
    from modules.fly_detector import FlyDetector, VialROI, VialCount, DetectorConfig
except ImportError:
    from fly_detector import FlyDetector, VialROI, VialCount, DetectorConfig

DEFAULT_TIMEPOINTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
DEFAULT_SETTLE_S   = 3.0
DEFAULT_WIDTH      = 1280
DEFAULT_HEIGHT     = 720
DEFAULT_FPS        = 24


# ── Simple camera check ───────────────────────────────────────────────────────
def camera_available(index: int = 0) -> tuple:
    """
    Test whether a camera is readable.

    Returns
    -------
    (ok: bool, message: str)
    """
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    ok = cap.isOpened()
    if ok:
        ret, _ = cap.read()
        ok = bool(ret)
    cap.release()
    if ok:
        return True, f"Camera index {index} detected and readable."
    return False, (
        f"Camera index {index} is not readable. "
        "Check USB connection, try a different index, or confirm camera permissions."
    )


# ── Simple one-shot recording ─────────────────────────────────────────────────
def record_session(
    camera_index: int = 0,
    duration_sec: float = 120.0,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    settle_time: float = DEFAULT_SETTLE_S,
    out_dir: str = "recordings",
) -> tuple:
    """
    One-shot recording function (no vial detection, pure MP4 capture).
    Use RecordingSession for full fly-counting pipeline.

    Returns
    -------
    (video_path: str, metadata_path: str, n_frames: int)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path  = out / f"drophenix_{ts}.mp4"
    meta_path = out / f"drophenix_{ts}_meta.csv"

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Check connection or try a different index."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS,          int(fps))

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or int(width)
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, float(fps), (aw, ah))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "VideoWriter failed to open. "
            "Ensure the output path is writable and codec is supported."
        )

    t0 = time.time()
    while time.time() - t0 < settle_time:
        cap.read()

    start = time.time()
    n = 0
    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            continue
        elapsed = time.time() - start
        cv2.putText(
            frame,
            f"DroPhenix  |  {elapsed:06.2f}s / {duration_sec:.0f}s",
            (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
        )
        writer.write(frame)
        n += 1

    writer.release()
    cap.release()

    pd.DataFrame([{
        "path": str(vid_path),
        "frames": n,
        "resolution": f"{aw}x{ah}",
        "fps": fps,
        "duration_sec": duration_sec,
        "timestamp": ts,
    }]).to_csv(meta_path, index=False)

    return str(vid_path), str(meta_path), n


# ── Camera wrapper ────────────────────────────────────────────────────────────
class DroPhenixCamera:
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int  = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int    = DEFAULT_FPS,
    ):
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
        try:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# ── Full recording session with fly counting ──────────────────────────────────
class RecordingSession:
    """
    Full automated climbing assay session.

    Workflow:
      session = RecordingSession(camera_source=0, vials=[...])
      session.configure_vials(vials)
      session.start()            # launches background thread
      # poll session.live_frame and session.results_df in Streamlit
      session.stop()
    """

    def __init__(
        self,
        camera_source: Union[int, str] = 0,
        timepoints: Optional[List[int]] = None,
        settle_s: float = DEFAULT_SETTLE_S,
        replicate: int = 1,
        output_dir: str = "data/auto",
        on_timepoint: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        detector_config: Optional["DetectorConfig"] = None,
        width: int  = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int    = DEFAULT_FPS,
        save_video: bool = True,
    ):
        self.session_id   = datetime.now().strftime("S%Y%m%d_%H%M%S")
        self.camera       = DroPhenixCamera(camera_source, width, height, fps)
        self.timepoints   = sorted(timepoints or DEFAULT_TIMEPOINTS)
        self.settle_s     = settle_s
        self.replicate    = replicate
        self.output_dir   = Path(output_dir)
        self.on_timepoint = on_timepoint
        self.on_complete  = on_complete
        self.detector     = FlyDetector(detector_config)
        self.fps          = fps
        self.width        = width
        self.height       = height
        self.save_video   = save_video

        self.vials: List[VialROI] = []
        self.background: Optional[np.ndarray] = None
        self.live_frame: Optional[np.ndarray] = None
        self._rows: List[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.state      = "idle"
        self.current_tp: Optional[int] = None
        self.elapsed_s  = 0.0
        self.error_msg  = ""
        self.video_path: Optional[Path] = None

    def configure_vials(self, vials: List[VialROI]):
        self.vials = vials

    def set_background(self, frame: np.ndarray):
        self.background = frame.copy()

    @property
    def results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def save_results(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.session_id}_raw_counts.csv"
        self.results_df.to_csv(path, index=False)
        return path

    def start(self):
        if not self.vials:
            raise ValueError("Configure vials before starting.")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.camera.release()

    def _run(self):
        writer = None
        try:
            if not self.camera.open():
                self.state     = "error"
                self.error_msg = "Cannot open camera. Check source index or permissions."
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)

            if self.save_video:
                self.video_path = self.output_dir / f"{self.session_id}_session.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self.video_path), fourcc,
                    float(self.fps), (self.width, self.height)
                )

            # Wait settle time, capture background
            self.state = "waiting_bg"
            time.sleep(self.settle_s)
            bg = self.camera.read()
            if bg is None:
                self.state     = "error"
                self.error_msg = "Failed to capture background frame."
                return
            self.set_background(bg)

            # Recording loop
            self.state  = "running"
            t_start     = time.monotonic()
            tp_index    = 0

            while not self._stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    continue
                self.live_frame = frame.copy()
                self.elapsed_s  = time.monotonic() - t_start

                if writer is not None:
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        f"{self.session_id}  |  {self.elapsed_s:06.2f}s",
                        (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                    )
                    writer.write(overlay)

                if (tp_index < len(self.timepoints) and
                        self.elapsed_s >= self.timepoints[tp_index]):
                    self.current_tp = self.timepoints[tp_index]
                    counts = self.detector.count_in_vials(
                        frame, self.vials, self.background
                    )
                    for vc in counts:
                        self._rows.append({
                            "SessionID":   self.session_id,
                            "Replicate":   self.replicate,
                            "Time":        self.current_tp,
                            "VialID":      vc.vial_id,
                            "Label":       vc.label,
                            "Climbed":     vc.n_above,
                            "NotClimbed":  vc.n_below,
                            "Total":       vc.total,
                            "Pct":         vc.pct_above,
                            "PI":          vc.pi,
                        })
                    if self.on_timepoint:
                        self.on_timepoint(self.current_tp, counts)
                    tp_index += 1

                if tp_index >= len(self.timepoints):
                    break
                time.sleep(0.01)

            self.state = "complete"
            self.save_results()
            if self.on_complete:
                self.on_complete(self.results_df)

        except Exception as e:
            self.state     = "error"
            self.error_msg = str(e)
        finally:
            if writer is not None:
                writer.release()
            self.camera.release()
