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


def _analyse_video(video_path, thr=0.5, sample_pts=None, roi=None):
    """Offline climbing counts from an uploaded clip.

    THREE CORRECTIONS, 2026-11 (register findings N6, N9, N7)
    ---------------------------------------------------------
    N6. The background subtractor was trained on 3-channel colour frames and
        then applied to 1-channel grayscale ones. OpenCV 4.13 raises nothing for
        this: it reports the ENTIRE FRAME as foreground. Measured on a synthetic
        clip of ten flies climbing to the top by 60 s, the old code returned
        `0.0 % climbed` at t = 90 and t = 120 where the truth is 100 %, because
        the early timepoint was destroyed and stationary flies at the top never
        re-registered -- a phenotype inversion, not a crash. Training and
        application now use the same grayscale representation, and the
        foreground fraction is reported per timepoint so a recurrence is visible
        rather than silent.

    N9. The scoring line was `int(h * thr)` -- a fixed fraction of the FULL
        FRAME height, with no region of interest at all, so any bench object or
        hand of fly-like area was counted and the line moved whenever the camera
        or framing did. `roi` is now the operator-confirmed vertical extent of
        the tube, `(y_top, y_bottom)`, and the line is placed within it. Without
        an ROI the counts are still returned for diagnostics but every row is
        marked `usable: False`: an uncalibrated reading is not a biological
        measurement. This is the behaviour `climbing_video.py` has had all along.

    N7. A requested timepoint past the end of the clip simply vanished -- no
        row, no warning -- and a frame in which nothing was detected was
        recorded as `0.0 % climbed` and then dropped from the headline mean.
        Late timepoints are where a slow climber's signal lives, so the set of
        timepoints actually analysed differed by genotype. Every requested
        timepoint now yields a row, with `usable` and a `reason`, and no
        percentage is invented for a failed measurement.

    Parameters
    ----------
    thr        : scoring line as a fraction of the tube height (not of the frame)
    sample_pts : timepoints in seconds
    roi        : (y_top, y_bottom) pixel extent of the tube, operator-confirmed

    Returns
    -------
    {"results": [ {t_s, above, total, pct, usable, reason, fgmask_fraction} ],
     "summary": {mean_pct_above, max_pct_above, n_timepoints, n_unusable, ...}}
    """
    if sample_pts is None:
        sample_pts = [5, 15, 30, 60, 90, 120]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video: " + video_path}

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (n_frames / fps) if (fps > 0 and n_frames > 0) else None

    # N9: the line belongs to the tube, not to the image.
    if roi is not None:
        y_top, y_bot = int(min(roi)), int(max(roi))
        y_top = max(0, min(y_top, h - 1))
        y_bot = max(y_top + 1, min(y_bot, h))
        line_y = int(y_top + thr * (y_bot - y_top))
        roi_note = ""
    else:
        y_top, y_bot = 0, h
        line_y = int(h * thr)
        roi_note = ("no tube ROI was supplied, so the scoring line is a fraction "
                    "of the whole frame and objects outside the tube are counted; "
                    "this is a diagnostic reading, not a biological measurement")

    bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False)

    # N6: train on grayscale, because that is what we apply it to.
    for _ in range(20):
        ret, f = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        bg.apply(cv2.GaussianBlur(g, (5, 5), 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    results = []
    for sec in sample_pts:
        rec = {"t_s": sec, "above": None, "total": None, "pct": None,
               "usable": False, "reason": "", "fgmask_fraction": None,
               "line_y": line_y, "roi": (y_top, y_bot) if roi is not None else None}

        # N7: refuse before reading, so the reason names the clip, not the codec.
        if duration_s is not None and sec > duration_s:
            rec["reason"] = f"requested t={sec:g}s but the video ends at {duration_s:.1f}s"
            results.append(rec)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if not ret:
            rec["reason"] = (f"frame for t={sec:g}s could not be decoded"
                             + (f"; the video ends at {duration_s:.1f}s" if duration_s else ""))
            results.append(rec)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fgm  = bg.apply(blur)
        fgm  = cv2.morphologyEx(fgm, cv2.MORPH_OPEN,  kernel)
        fgm  = cv2.morphologyEx(fgm, cv2.MORPH_CLOSE, kernel)
        _, th = cv2.threshold(fgm, 200, 255, cv2.THRESH_BINARY)

        # Restrict to the tube when we have one, and record how much of the
        # analysed area came back as foreground.
        band = th[y_top:y_bot, :]
        frac = float((band > 0).mean()) if band.size else 0.0
        rec["fgmask_fraction"] = round(frac, 4)

        cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL,
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
            if int(M["m01"] / M["m00"]) + y_top < line_y:
                above += 1
        rec["above"], rec["total"] = above, total

        # A whole-frame foreground mask means the subtractor is not separating
        # anything; counts from it are meaningless whatever their value.
        if frac > 0.5:
            rec["reason"] = (f"{frac:.0%} of the analysed area is foreground: the "
                             "background model is not separating flies from the scene")
        elif total == 0:
            # Not 0 % climbed. No measurement.
            rec["reason"] = "no flies detected: measurement failed, not zero climbing"
        elif roi is None:
            rec["reason"] = roi_note
        else:
            rec["usable"] = True
            rec["pct"] = round(above / total * 100.0, 1)
        # Both facts can be true at once -- a detection failure AND an
        # uncalibrated geometry -- and dropping either would mislead about why
        # the row cannot be reported.
        if roi is None and not rec["usable"] and roi_note not in rec["reason"]:
            rec["reason"] = (rec["reason"] + "; " if rec["reason"] else "") + roi_note
        results.append(rec)

    cap.release()

    usable = [r for r in results if r["usable"]]
    counts = [r["pct"] for r in usable]
    return {
        "results": results,
        "summary": {
            "mean_pct_above": round(sum(counts) / len(counts), 1) if counts else None,
            "max_pct_above":  round(max(counts), 1) if counts else None,
            "n_timepoints":   len(usable),
            "n_requested":    len(results),
            "n_unusable":     len(results) - len(usable),
            "duration_s":     round(duration_s, 1) if duration_s else None,
            "roi_supplied":   roi is not None,
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
            "Scoring line (fraction of TUBE height)",
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
    # The scoring line belongs to the TUBE, not to the frame. Without the
    # operator confirming where the tube is, a reading here is a diagnostic
    # number and the results table says so on every row.
    st.markdown("**Tube extent (required for a reportable measurement)**")
    st.caption(
        "Give the vertical pixel range the tube occupies in this framing. The "
        "scoring line is placed inside that range, and anything outside it is "
        "ignored. Without it the line is a fraction of the whole frame and a "
        "hand or a bench object of fly-like size is counted as a fly."
    )
    r1, r2, r3 = st.columns([1, 1, 2])
    use_roi = r3.checkbox("I have confirmed the tube extent",
                          value=False, key="ar_cloud_useroi")
    y_top = r1.number_input("Tube TOP (px)", min_value=0, max_value=max(hv - 1, 1),
                            value=int(hv * 0.08), step=5, key="ar_cloud_ytop")
    y_bot = r2.number_input("Tube BOTTOM (px)", min_value=1, max_value=max(hv, 2),
                            value=int(hv * 0.92), step=5, key="ar_cloud_ybot")
    roi = (int(y_top), int(y_bot)) if use_roi else None
    if roi is None:
        st.warning(
            "No confirmed tube extent: results will be returned for diagnosis "
            "but marked **not usable as a biological measurement**."
        )

    if st.button("Analyse Video", key="ar_cloud_run", type="primary"):
        with st.spinner("Extracting fly counts..."):
            res = _analyse_video(tmp_path, thr, pts, roi=roi)
        if "error" in res:
            st.error(res["error"])
            return
        smry = res["summary"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean climbed (%)", str(smry.get("mean_pct_above", "N/A")))
        s2.metric("Peak climbed (%)", str(smry.get("max_pct_above",  "N/A")))
        s3.metric("Usable timepoints",
                  f"{smry.get('n_timepoints', 0)} / {smry.get('n_requested', 0)}")
        s4.metric("Clip length (s)", str(smry.get("duration_s", "?")))

        rows = res["results"]
        df = pd.DataFrame([{
            "Time (s)":        r["t_s"],
            "Above line":      r["above"],
            "Total detected":  r["total"],
            "% climbed":       r["pct"],
            "Usable":          r["usable"],
            "Foreground frac": r["fgmask_fraction"],
            "Reason":          r["reason"],
        } for r in rows])
        st.dataframe(df, width="stretch")

        n_bad = int(smry.get("n_unusable", 0))
        if n_bad:
            st.error(
                f"{n_bad} of {smry.get('n_requested', 0)} requested timepoints did "
                "not yield a measurement. They are listed above with a reason and "
                "carry **no percentage** — a failed detection is not 0% climbed, and "
                "a timepoint past the end of the clip is not a measurement of "
                "anything. Earlier versions dropped these rows from the mean "
                "without saying so."
            )
        st.download_button(
            "Download QC CSV", df.to_csv(index=False).encode(),
            "recording_qc.csv", "text/csv", key="ar_cloud_dl")
        st.info(
            "Download this CSV and upload via CSV Upload and Analysis "
            "to run the full DroPhenix pipeline. Drop the unusable rows there — "
            "do not fill them with zeros."
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
