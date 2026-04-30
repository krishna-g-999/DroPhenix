"""modules/auto_recording.py - DroPhenix v2.0 adaptive recording module."""
from __future__ import annotations
import os, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

_IS_CLOUD = (
    os.path.exists("/mount/src")
    or os.environ.get("STREAMLIT_SHARING_MODE") == "1"
    or os.environ.get("IS_STREAMLIT_CLOUD", "0") == "1"
)

try:
    import cv2
    _CV2 = True
except ImportError:
    cv2 = None
    _CV2 = False


def _camera_available():
    if _IS_CLOUD or not _CV2:
        return False
    try:
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


def _analyse_video(video_path, thr=0.5, sample_pts=None):
    if sample_pts is None:
        sample_pts = [5, 15, 30, 60, 90, 120]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video: " + video_path}
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    line_y = int(h * thr)
    bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False)
    for _ in range(20):
        ret, f = cap.read()
        if not ret:
            break
        bg.apply(cv2.GaussianBlur(f, (5, 5), 0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    results = []
    for sec in sample_pts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            results.append((sec, None, None, None))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fgm  = bg.apply(blur)
        fgm  = cv2.morphologyEx(fgm, cv2.MORPH_OPEN,  kernel)
        fgm  = cv2.morphologyEx(fgm, cv2.MORPH_CLOSE, kernel)
        _, th = cv2.threshold(fgm, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        total = above = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if not (20 < area < 800):
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            total += 1
            if int(M["m01"] / M["m00"]) < line_y:
                above += 1
        pct = round(above / total * 100.0, 1) if total > 0 else 0.0
        results.append((sec, above, total, pct))
    cap.release()
    valid  = [r for r in results if r[2] is not None and r[2] > 0]
    counts = [r[3] for r in valid]
    return {
        "results": results,
        "summary": {
            "mean_pct_above": round(sum(counts)/len(counts), 1) if counts else None,
            "max_pct_above":  round(max(counts), 1) if counts else None,
            "n_timepoints":   len(valid),
        },
    }


def _render_offline():
    st.markdown("#### Offline Video Analysis")
    st.caption(
        "Upload an MP4 / AVI / MOV recorded in your lab. "
        "DroPhenix extracts fly counts using background subtraction "
        "and morphological filtering."
    )
    if not _CV2:
        st.error("OpenCV not installed. Add opencv-python-headless to requirements.txt.")
        return
    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        vf = st.file_uploader(
            "Upload video", type=["mp4", "avi", "mov"], key="ar_cloud_upload")
    with col_cfg:
        thr = st.slider(
            "Threshold line (fraction of height)",
            0.2, 0.8, 0.5, 0.05, key="ar_cloud_thr")
        pts_str = st.text_input(
            "Sample timepoints (s, comma-separated)",
            value="5,15,30,60,90,120", key="ar_cloud_pts")
        try:
            pts = [float(x) for x in pts_str.split(",") if x.strip()]
        except ValueError:
            pts = [5, 15, 30, 60, 90, 120]
    if vf is None:
        st.info("Upload a video file to begin.")
        return
    suffix = Path(vf.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(vf.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 24.0
    nf    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wv    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hv    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration (s)", f"{nf/fps_v:.1f}")
    m2.metric("FPS",          f"{fps_v:.1f}")
    m3.metric("Resolution",   f"{wv}x{hv}")
    m4.metric("Frames",       str(nf))
    if st.button("Analyse Video", key="ar_cloud_run", type="primary"):
        with st.spinner("Extracting fly counts..."):
            res = _analyse_video(tmp_path, thr, pts)
        if "error" in res:
            st.error(res["error"])
            return
        smry = res["summary"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean climbed (%)", str(smry.get("mean_pct_above", "N/A")))
        s2.metric("Peak climbed (%)", str(smry.get("max_pct_above",  "N/A")))
        s3.metric("Timepoints",       str(smry.get("n_timepoints",   0)))
        df = pd.DataFrame(
            res["results"],
            columns=["Time (s)", "Above line", "Total", "% climbed"])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download QC CSV", df.to_csv(index=False).encode(),
            "recording_qc.csv", "text/csv", key="ar_cloud_dl")
        st.info(
            "Download this CSV and upload via CSV Upload and Analysis "
            "to run the full DroPhenix pipeline."
        )


def _render_desktop():
    try:
        from modules.recording import render as _rec
        _rec(st.subheader, st.info, st.warning, st.success)
    except ImportError as exc:
        st.error("modules/recording.py not found: " + str(exc))
    except Exception as exc:
        import traceback
        st.error("Recording error: " + str(exc))
        st.code(traceback.format_exc())


def render(*_args, **_kwargs):
    try:
        from modules.styles import render_header
        render_header(
            "Live Recording",
            "Camera capture | assay QC | offline video analysis")
    except ImportError:
        st.markdown("## Live Recording")

    if _IS_CLOUD:
        st.info(
            "Cloud deployment detected. "
            "Streamlit Community Cloud is a headless server with no physical camera. "
            "Offline Video Analysis below is fully functional: upload any MP4/AVI "
            "recorded in your lab and DroPhenix extracts frame-by-frame fly counts. "
            "For live recording, run DroPhenix locally with: streamlit run app.py"
        )
        st.divider()
        _render_offline()
    else:
        if not _camera_available():
            st.warning("No camera detected. Offline Video Analysis is available below.")
            _render_offline()
        else:
            t1, t2 = st.tabs(["Live Recording", "Offline Analysis"])
            with t1:
                _render_desktop()
            with t2:
                _render_offline()
