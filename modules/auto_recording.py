"""
modules/auto_recording.py
DroPhenix Analytics v2.0  —  Adaptive Recording Module

Behaviour
---------
* Streamlit Cloud  : no physical camera available → renders an Offline
  Video Analysis panel (upload MP4/AVI → CV2 fly-count extraction) plus
  a clearly labelled "Desktop-only" notice for the live-capture workflow.
* Local / desktop  : delegates entirely to modules/recording.py which
  provides the full threaded MSMF/V4L2 recording interface.

Detection heuristic
-------------------
Streamlit Community Cloud mounts the repository under /mount/src.
We also check for an attached video device via cv2 as a secondary guard.
"""

from __future__ import annotations

import io
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

# ── Cloud-environment detection ──────────────────────────────────────────────
_IS_CLOUD: bool = (
    os.path.exists("/mount/src")                      # Streamlit Community Cloud
    or os.environ.get("STREAMLIT_SHARING_MODE") == "1"
    or os.environ.get("IS_STREAMLIT_CLOUD", "0") == "1"
)

# ── Optional OpenCV import ────────────────────────────────────────────────────
try:
    import cv2
    _CV2 = True
except ImportError:
    cv2 = None          # type: ignore
    _CV2 = False

# ── Camera-availability probe (skipped on cloud to avoid hang) ───────────────
def _camera_available() -> bool:
    if _IS_CLOUD or not _CV2:
        return False
    try:
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# Offline video analysis  (works on Cloud AND desktop)
# ═════════════════════════════════════════════════════════════════════════════

def _render_offline_analysis() -> None:
    """Upload a pre-recorded video and extract fly counts via background
    subtraction + morphological filtering.  Identical to the Offline Analysis
    tab in recording.py but self-contained so it loads on Cloud."""

    st.markdown("#### 📁 Offline Video Analysis")
    st.caption(
        "Upload an MP4 / AVI / MOV recorded with any camera.  "
        "DroPhenix extracts frame-level fly counts using background subtraction "
        "and morphological clean-up (identical algorithm to the desktop recorder)."
    )

    if not _CV2:
        st.error(
            "OpenCV is not installed.  Add `opencv-python-headless` to "
            "`requirements.txt` and redeploy."
        )
        return

    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        vf = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov"],
            key="ar_cloud_upload",
        )
    with col_cfg:
        thr = st.slider(
            "Threshold line (fraction of frame height)",
            0.2, 0.8, 0.5, 0.05,
            key="ar_cloud_thr",
            help="Flies *above* this line are counted as 'climbed'.",
        )
        sample_pts_str = st.text_input(
            "Sample timepoints (s, comma-separated)",
            value="5,15,30,60,90,120",
            key="ar_cloud_pts",
        )
        try:
            sample_pts = [float(x) for x in sample_pts_str.split(",") if x.strip()]
        except ValueError:
            sample_pts = [5, 15, 30, 60, 90, 120]

    if vf is None:
        st.info("Upload a video file to begin offline analysis.")
        return

    # ── Write upload to a temp file ──────────────────────────────────────────
    suffix = Path(vf.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(vf.read())
        tmp_path = tmp.name

    # ── Video metadata ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(tmp_path)
    fps_vid  = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur_s    = n_frames / fps_vid if fps_vid > 0 else 0
    cap.release()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration (s)",   f"{dur_s:.1f}")
    m2.metric("FPS",            f"{fps_vid:.1f}")
    m3.metric("Resolution",     f"{w_vid}×{h_vid}")
    m4.metric("Total frames",   str(n_frames))

    if st.button("🔬 Analyse Video", key="ar_cloud_run", type="primary"):
        with st.spinner("Extracting fly counts — this may take 30–60 s for long videos…"):
            results = _analyse_video(tmp_path, thr, sample_pts)

        if "error" in results:
            st.error(results["error"])
            return

        smry = results["summary"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean climbed (%)",  f"{smry.get('mean_pct_above', 'N/A')}")
        s2.metric("Peak climbed (%)",  f"{smry.get('max_pct_above',  'N/A')}")
        s3.metric("Timepoints analysed", str(smry.get("n_timepoints", 0)))

        df_qc = pd.DataFrame(results["results"])
        df_qc.columns = ["Time (s)", "Above line", "Total detected", "% climbed"]
        st.dataframe(df_qc, use_container_width=True)

        csv_bytes = df_qc.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download QC Table (.csv)",
            csv_bytes,
            "recording_qc.csv",
            "text/csv",
            key="ar_cloud_dl",
        )

        st.info(
            "To load these counts into the DroPhenix analysis pipeline, "
            "download the CSV above then re-upload it via **CSV Upload and Analysis**."
        )


def _analyse_video(
    video_path: str,
    threshold_fraction: float = 0.5,
    sample_seconds: list[float] | None = None,
) -> dict[str, Any]:
    """Background-subtraction fly counter (cloud-safe, no GUI windows)."""
    if sample_seconds is None:
        sample_seconds = [5, 15, 30, 60, 90, 120]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    line_y = int(h * threshold_fraction)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False
    )
    for _ in range(20):                           # warm-up background model
        ret, f = cap.read()
        if not ret:
            break
        bg.apply(cv2.GaussianBlur(f, (5, 5), 0))

    results = []
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for sec in sample_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            results.append({"time_s": sec, "above": None, "total": None, "pct_above": None})
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        fgm   = bg.apply(blur)
        fgm   = cv2.morphologyEx(fgm, cv2.MORPH_OPEN,  kernel)
        fgm   = cv2.morphologyEx(fgm, cv2.MORPH_CLOSE, kernel)
        _, th = cv2.threshold(fgm, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total = above = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if not (20 < area < 800):
                continue
            M  = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cy = int(M["m01"] / M["m00"])
            total += 1
            if cy < line_y:
                above += 1

        pct = round(above / total * 100.0, 1) if total > 0 else 0.0
        results.append({"time_s": sec, "above": above, "total": total, "pct_above": pct})

    cap.release()

    valid  = [r for r in results if r["total"] is not None and r["total"] > 0]
    counts = [r["pct_above"] for r in valid]
    return {
        "results": [
            (r["time_s"], r["above"], r["total"], r["pct_above"]) for r in results
        ],
        "summary": {
            "mean_pct_above": round(sum(counts) / len(counts), 1) if counts else None,
            "max_pct_above":  round(max(counts), 1)               if counts else None,
            "n_timepoints":   len(valid),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Desktop live-recording UI
# ═════════════════════════════════════════════════════════════════════════════

def _render_desktop_recording() -> None:
    """Delegates to the full threaded recording UI in modules/recording.py."""
    try:
        from modules.recording import render as _rec_render  # type: ignore
        _rec_render(st.subheader, st.info, st.warning, st.success)
    except ImportError as exc:
        st.error(
            f"modules/recording.py not found: {exc}.  "
            "Ensure recording.py is present in the modules/ folder."
        )
    except Exception as exc:
        st.error(f"Recording module error: {exc}")
        import traceback
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point  —  called by app.py
# ═════════════════════════════════════════════════════════════════════════════

def render(*_args: Any, **_kwargs: Any) -> None:
    """
    Main entry point for the Live Recording page.

    On Streamlit Cloud  → Offline Video Analysis only.
    On local desktop    → Full live recording + offline analysis.
    """

    # ── Page header ──────────────────────────────────────────────────────────
    try:
        from modules.styles import render_header          # type: ignore
        render_header(
            "Live Recording",
            "Camera capture · assay QC · offline video analysis",
        )
    except ImportError:
        st.markdown("## 🎥 Live Recording")
        st.caption("Camera capture · assay QC · offline video analysis")

    # ── Environment banner ───────────────────────────────────────────────────
    if _IS_CLOUD:
        st.info(
            "**Cloud deployment detected.**  "
            "Streamlit Community Cloud is a headless Linux server — "
            "no physical camera is attached.  

"
            "✅ **Offline Video Analysis** (below) is fully functional: upload "
            "any MP4 / AVI recorded in your lab and DroPhenix will extract "
            "frame-by-frame fly counts.  

"
            "🖥 **Live recording** requires the desktop version of DroPhenix.  "
            "Clone the repository and run `streamlit run app.py` locally to "
            "access the full MSMF / V4L2 camera interface.",
            icon="ℹ️",
        )
        st.divider()
        _render_offline_analysis()

    else:
        # Local desktop — show live recording first, offline analysis in expander
        cam_ok = _camera_available()
        if not cam_ok:
            st.warning(
                "No camera detected on this machine.  "
                "Connect a USB webcam or built-in camera, then refresh.  "
                "Offline Video Analysis is still available below.",
                icon="⚠️",
            )
            with st.expander("📁 Offline Video Analysis", expanded=True):
                _render_offline_analysis()
        else:
            tab_live, tab_offline = st.tabs(["📷 Live Recording", "📁 Offline Analysis"])
            with tab_live:
                _render_desktop_recording()
            with tab_offline:
                _render_offline_analysis()
