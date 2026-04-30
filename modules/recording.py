# -*- coding: utf-8 -*-
"""
DroPhenix -- modules/recording.py
Thread-safe camera recording for the DroPhenix Live Recording page.

Thread safety contract
  Background thread (_record_worker) never touches st.* or session_state.
  Shared state lives in a plain dict + threading.Lock.
  Preview JPEG is written to a temp file; st.image() reads by path.
  Pause is implemented via threading.Event; stop via a shared flag.

Exports required by app.py
  list_cameras, capture_test_frame, capture_background_frame,
  RecordingSettings, RecordingSession,
  run_pre_assay_qc, analyze_recorded_video_qc, CV2_AVAILABLE
"""
from __future__ import annotations

import os
import platform
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress verbose DSHOW/FFMPEG probing on Windows
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"]  = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _backend() -> int:
    if not CV2_AVAILABLE:
        return 0
    s = platform.system()
    if s == "Windows":
        return cv2.CAP_MSMF
    if s == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_V4L2


def _open_cap(index: int):
    """Open camera with preferred backend; fall back to CAP_ANY."""
    if not CV2_AVAILABLE:
        return None
    for bk in [_backend(), cv2.CAP_ANY]:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        except Exception:
            continue
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        if ret:
            return cap
        cap.release()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_cameras(max_probe: int = 5) -> List[Tuple[int, str]]:
    """Return [(index, label), ...] for connected cameras."""
    if not CV2_AVAILABLE:
        return []
    out: List[Tuple[int, str]] = []
    for idx in range(max_probe):
        cap = _open_cap(idx)
        if cap is None:
            continue
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        tag = "Built-in" if idx == 0 else f"USB #{idx}"
        out.append((idx, f"Camera {idx} -- {tag} ({w}x{h} @ {fps:.0f} fps)"))
    return out


def capture_test_frame(
    index: int,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> Optional[bytes]:
    """Return JPEG bytes of a single test frame, or None on failure."""
    if not CV2_AVAILABLE:
        return None
    cap = _open_cap(index)
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(10):  # warm-up frames
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return bytes(buf) if ok else None


def capture_background_frame(
    index: int,
    width: int = 1280,
    height: int = 720,
) -> Optional[bytes]:
    """Capture a still frame for background subtraction reference."""
    return capture_test_frame(index, width, height)


@dataclass
class RecordingSettings:
    camera_index: int   = 0
    width:        int   = 1280
    height:       int   = 720
    fps:          int   = 24
    duration:     int   = 120
    settle:       float = 3.0
    output_dir:   str   = "recordings"
    label:        str   = ""


@dataclass
class RecordingSession:
    path:    str            = ""
    frames:  int            = 0
    elapsed: float          = 0.0
    status:  str            = "idle"
    error:   Optional[str]  = None


def run_pre_assay_qc(settings: RecordingSettings) -> Dict[str, Any]:
    """Quick QC check before recording starts."""
    qc: Dict[str, Any] = {"ok": False, "errors": [], "warnings": []}
    if not CV2_AVAILABLE:
        qc["errors"].append("OpenCV not installed. pip install opencv-python")
        return qc
    cap = _open_cap(settings.camera_index)
    if cap is None:
        qc["errors"].append(
            f"Camera {settings.camera_index} not accessible. "
            "Check connection and close Teams/Zoom/OBS."
        )
        return qc
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        qc["errors"].append("Camera opened but could not read a frame.")
        return qc
    mean_brightness = float(frame.mean())
    if mean_brightness < 20:
        qc["warnings"].append("Frame is very dark -- check lighting.")
    if mean_brightness > 230:
        qc["warnings"].append("Frame is very bright -- reduce exposure.")
    qc["ok"] = True
    qc["brightness"] = round(mean_brightness, 1)
    return qc


def analyze_recorded_video_qc(
    video_path: str,
    threshold_fraction: float = 0.5,
    sample_seconds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Lightweight post-recording QC: fly counts at sampled timepoints."""
    if not CV2_AVAILABLE:
        return {"error": "OpenCV not installed"}
    if sample_seconds is None:
        sample_seconds = [5, 15, 30, 60, 90, 120]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    liney  = int(height * threshold_fraction)
    bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False
    )
    for _ in range(20):
        ret, f = cap.read()
        if not ret:
            break
        bg.apply(cv2.GaussianBlur(f, (5, 5), 0))
    results = []
    for sec in sample_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            results.append({"time_s": sec, "above": None, "total": None, "pct_above": None})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fgm  = bg.apply(blur)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgm  = cv2.morphologyEx(fgm, cv2.MORPH_OPEN,  k)
        fgm  = cv2.morphologyEx(fgm, cv2.MORPH_CLOSE, k)
        _, th = cv2.threshold(fgm, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = above = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 20 or area > 800:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cy = int(M["m01"] / M["m00"])
            total += 1
            if cy < liney:
                above += 1
        pct = round(above / total * 100.0, 1) if total else 0.0
        results.append({"time_s": sec, "above": above, "total": total, "pct_above": pct})
    cap.release()
    valid  = [r for r in results if r["total"] is not None and r["total"] > 0]
    counts = [r["pct_above"] for r in valid]
    return {
        "results": results,
        "summary": {
            "mean_pct_above": round(sum(counts) / len(counts), 1) if counts else None,
            "max_pct_above":  round(max(counts), 1) if counts else None,
            "n_timepoints":   len(valid),
        },
        "threshold_fraction": threshold_fraction,
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# Background recording worker (runs in daemon thread -- never touches st.*)
# ---------------------------------------------------------------------------

def _record_worker(
    index: int,
    out_path: str,
    fps: int,
    width: int,
    height: int,
    duration: float,
    settle: float,
    shared: dict,
    lock: threading.Lock,
    preview_path: str,
    pause_event: threading.Event,
) -> None:
    """Run entirely in a background thread."""

    def _set(**kw):
        with lock:
            shared.update(kw)

    _set(status="init", frames=0, elapsed=0.0, error=None, done=False)

    if not CV2_AVAILABLE:
        _set(error="OpenCV not installed. pip install opencv-python", done=True)
        return

    cap = _open_cap(index)
    if cap is None:
        _set(error=(
            f"Cannot open camera {index}. "
            "Ensure camera is connected and not used by another app."
        ), done=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or width
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (aw, ah))
    if not writer.isOpened():
        cap.release()
        _set(error="VideoWriter failed. Check disk space and write permissions.", done=True)
        return

    # Settle phase
    _set(status="settle")
    t0 = time.time()
    while time.time() - t0 < settle:
        with lock:
            if shared.get("stop_requested"):
                break
        cap.read()

    # Recording phase
    _set(status="recording")
    preview_every = max(1, fps // 2)
    t_start = time.time()
    n = 0

    while True:
        with lock:
            stop = shared.get("stop_requested", False)
        if stop:
            break
        # Pause: block until resumed or stopped
        while not pause_event.is_set():
            with lock:
                if shared.get("stop_requested"):
                    break
            shared["status"] = "paused"
            time.sleep(0.2)
        _set(status="recording")
        elapsed = time.time() - t_start
        if elapsed >= duration:
            break
        ret, frame = cap.read()
        if not ret:
            continue
        # Timestamp overlay
        cv2.putText(
            frame,
            f"DroPhenix {elapsed:05.1f}s / {duration:.0f}s",
            (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            (0, 165, 255), 2, cv2.LINE_AA,
        )
        writer.write(frame)
        n += 1
        with lock:
            shared["frames"]  = n
            shared["elapsed"] = elapsed
        if n % preview_every == 0:
            try:
                cv2.imwrite(preview_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            except Exception:
                pass

    writer.release()
    cap.release()
    _set(frames=n, done=True, status="done")