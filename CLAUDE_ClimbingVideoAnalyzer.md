# CLAUDE.md — DroPhenix Climbing Video Analyser
# Extends DroPhenix with a production-quality video climbing assay analysis module
# Completely replaces the existing webcam-only video_analyzer.py and fly_detector.py
# Outputs DroPhenix-compatible CSV that feeds directly into the Climbing Assay page

## WHAT WE ARE BUILDING
A Streamlit-based video analysis tool that:
1. Accepts uploaded climbing assay videos (MP4, AVI, MOV, any format)
2. Lets the user define regions of interest (ROI) per vial and set the reference line
3. Counts flies above the reference line at each user-specified timepoint
4. Computes Performance Index (PI) = flies_above / n_flies_expected
5. Outputs a DroPhenix-compatible CSV
6. Works regardless of lab setup (vials, cylinders, any size, any background)

## HOW THIS DIFFERS FROM FREECLIMBER AND FLYCLIMBER
- FreeClimber: measures velocity, requires custom hardware, not setup-agnostic
- FlyClimber: requires white background, no multi-vial ROI selection
- THIS TOOL: user-defined ROI (any setup), counts above a line (traditional PI metric),
  multi-vial in one video, integrates with DroPhenix, flags low-confidence frames

## VERIFIED TECHNICAL BASIS
- Computer vision (OpenCV) for Drosophila detection: proven (Yarwais et al. 2020)
- Background subtraction + particle detection: FreeClimber (Spierer et al. 2021)
- Deep learning vial segmentation + CV tracking: Melkani et al. 2026 (IoU > 0.95)
- This tool uses ADAPTIVE THRESHOLDING (no DL needed because user defines ROI)

## CRITICAL RULES
1. ALWAYS validate detected count against expected N flies — flag if > 30% discrepancy.
2. In image coordinates: Y=0 is TOP of image. "Above line" means centroid_y < line_y.
3. Never hardcode pixel sizes for flies — calculate from user-provided vial dimensions.
4. The reference line Y-coordinate is stored in VIAL coordinate space (relative to ROI).
5. Every output CSV must have the DroPhenix required columns (listed below).
6. Show the user a preview with detected flies overlaid BEFORE running full analysis.
7. If analysis fails for a frame, mark that frame's confidence = 0 — never skip silently.

---

## DEPENDENCIES TO INSTALL

```bash
pip install opencv-python streamlit Pillow numpy pandas openpyxl scipy
pip install streamlit-drawable-canvas  # for interactive ROI and line drawing
pip install ffmpeg-python              # for video frame extraction
```

Add to requirements.txt:
```
opencv-python>=4.8.0
streamlit-drawable-canvas>=0.9.0
ffmpeg-python>=0.2.0
```

---

## PHASE 1 — Core detection module

### File: modules/climbing_video_analyzer.py

```python
"""
Core climbing assay video analysis module.

Algorithm summary:
1. Extract frame at specified time (seconds since tap-down)
2. Crop to vial ROI
3. Convert to grayscale
4. Gaussian blur to reduce noise
5. Adaptive thresholding (dark flies on light, or light flies on dark)
6. Morphological operations (open: remove noise, close: fill gaps in fly bodies)
7. Find contours, filter by area (fly size range in pixels)
8. Count contours with centroid ABOVE the reference line Y
9. Report count, PI, and confidence (detected_total / expected_N)

Key design choices:
- Adaptive thresholding beats global threshold (handles lighting gradient in vials)
- Contour area filtering handles glass reflections and background artifacts
- Confidence score enables the user to flag and discard unreliable frames
- All pixel coordinates are stored relative to the ROI origin for portability
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VialConfig:
    """Configuration for one vial in the video."""
    vial_id: int
    genotype: str
    replicate: int
    n_flies_expected: int
    roi_x: int        # ROI top-left X in full frame
    roi_y: int        # ROI top-left Y in full frame
    roi_w: int        # ROI width in pixels
    roi_h: int        # ROI height in pixels
    line_y_frac: float  # Reference line Y as fraction of ROI height (0.0=top, 1.0=bottom)
    fly_color: str = 'dark'  # 'dark' or 'light' flies
    sex: str = 'Mixed'
    treatment: str = 'vehicle'
    notes: str = ''

    @property
    def line_y_pixels(self) -> int:
        """Reference line Y coordinate in ROI pixel space."""
        return int(self.line_y_frac * self.roi_h)


@dataclass
class FrameResult:
    """Result of analysis for one vial at one timepoint."""
    vial_id: int
    time_sec: float
    frame_number: int
    flies_above: int
    flies_below: int
    flies_detected: int
    n_flies_expected: int
    pi: float          # Performance Index = flies_above / n_flies_expected
    confidence: float  # flies_detected / n_flies_expected (capped at 1.0)
    low_confidence: bool  # True if confidence < 0.70


@dataclass
class VideoAnalysisSession:
    """Holds all configuration and results for one video analysis session."""
    video_path: str
    fps: float = 0.0
    total_frames: int = 0
    duration_sec: float = 0.0
    t_zero_frame: int = 0        # Frame number of tap-down (T=0)
    timepoints_sec: list[float] = field(default_factory=lambda: [10, 20, 30, 60])
    vials: list[VialConfig] = field(default_factory=list)
    results: list[FrameResult] = field(default_factory=list)

    # Detection parameters (adjustable by user during calibration)
    detection_params: dict = field(default_factory=lambda: {
        'blur_kernel': 5,           # Gaussian blur kernel size (odd number)
        'adaptive_block': 21,       # Adaptive threshold block size (odd, > 3)
        'adaptive_c': 4,            # Adaptive threshold constant
        'min_area_frac': 0.0003,    # Min fly area as fraction of ROI area
        'max_area_frac': 0.015,     # Max fly area as fraction of ROI area
        'morph_open': 2,            # Morphological open iterations
        'morph_close': 2,           # Morphological close iterations
        'confidence_threshold': 0.70,  # Below this → flagged as low confidence
    })


def load_video_metadata(video_path: str) -> dict:
    """
    Load basic video metadata: fps, total frames, duration.
    Returns dict with keys: fps, total_frames, duration_sec, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration_sec': total_frames / fps if fps > 0 else 0,
        'width': width,
        'height': height,
    }


def get_frame_at_time(video_path: str, time_sec: float,
                       t_zero_frame: int = 0, fps: float = 30.0) -> Optional[np.ndarray]:
    """
    Extract a specific frame from the video at time_sec after t_zero_frame.
    Returns BGR numpy array or None if frame cannot be read.
    """
    target_frame = t_zero_frame + int(time_sec * fps)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_frame_at_number(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Get a specific frame by frame number."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def detect_flies_in_roi(frame: np.ndarray, vial: VialConfig,
                          params: dict) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Detect flies in a vial's ROI and return their centroid coordinates + debug image.

    Args:
        frame: Full BGR video frame
        vial: VialConfig with ROI coordinates and fly color
        params: Detection parameter dict from VideoAnalysisSession

    Returns:
        (centroids, debug_image)
        centroids: list of (x, y) tuples in ROI coordinate space
        debug_image: BGR image of ROI with detected flies overlaid
    """
    # Crop to ROI
    x, y, w, h = vial.roi_x, vial.roi_y, vial.roi_w, vial.roi_h
    roi = frame[y:y+h, x:x+w].copy()
    if roi.size == 0:
        return [], np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce sensor noise
    blur_k = params['blur_kernel']
    blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1  # must be odd
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Adaptive thresholding
    block = params['adaptive_block']
    block = block if block % 2 == 1 and block > 3 else 21
    c_val = params['adaptive_c']

    if vial.fly_color == 'dark':
        # Dark flies on light background → threshold inverted
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block, c_val
        )
    else:
        # Light flies on dark background → threshold normal
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, c_val
        )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                                iterations=params['morph_open'])
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel,
                                iterations=params['morph_close'])

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area (expressed as fraction of ROI area for scale-independence)
    roi_area = w * h
    min_area = params['min_area_frac'] * roi_area
    max_area = params['max_area_frac'] * roi_area

    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))

    # Build debug image (ROI with overlays)
    debug = roi.copy()
    line_y = vial.line_y_pixels

    # Draw reference line (green)
    cv2.line(debug, (0, line_y), (w - 1, line_y), (0, 255, 0), 2)

    # Draw detected fly centroids (red dots = above, blue dots = below)
    for (cx, cy) in centroids:
        color = (0, 0, 255) if cy < line_y else (255, 0, 0)  # red=above, blue=below
        cv2.circle(debug, (cx, cy), 5, color, -1)

    # Count labels
    above = sum(1 for (cx, cy) in centroids if cy < line_y)
    below = sum(1 for (cx, cy) in centroids if cy >= line_y)
    cv2.putText(debug, f"Above: {above}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(debug, f"Below: {below}", (5, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return centroids, debug


def analyze_frame_for_vial(video_path: str, vial: VialConfig,
                             time_sec: float, t_zero_frame: int,
                             fps: float, params: dict) -> FrameResult:
    """
    Analyze one vial at one timepoint.
    Returns FrameResult with count, PI, and confidence.
    """
    target_frame = t_zero_frame + int(time_sec * fps)
    frame = get_frame_at_number(video_path, target_frame)

    if frame is None:
        return FrameResult(
            vial_id=vial.vial_id, time_sec=time_sec, frame_number=target_frame,
            flies_above=0, flies_below=0, flies_detected=0,
            n_flies_expected=vial.n_flies_expected,
            pi=0.0, confidence=0.0, low_confidence=True
        )

    centroids, _ = detect_flies_in_roi(frame, vial, params)
    line_y = vial.line_y_pixels
    above = sum(1 for (cx, cy) in centroids if cy < line_y)
    below = sum(1 for (cx, cy) in centroids if cy >= line_y)
    total_detected = len(centroids)
    n_expected = vial.n_flies_expected

    pi = above / max(n_expected, 1)
    confidence = min(total_detected / max(n_expected, 1), 1.0)
    low_conf = confidence < params.get('confidence_threshold', 0.70)

    return FrameResult(
        vial_id=vial.vial_id,
        time_sec=time_sec,
        frame_number=target_frame,
        flies_above=above,
        flies_below=below,
        flies_detected=total_detected,
        n_flies_expected=n_expected,
        pi=round(pi, 4),
        confidence=round(confidence, 3),
        low_confidence=low_conf,
    )


def run_full_analysis(session: VideoAnalysisSession) -> pd.DataFrame:
    """
    Run analysis for all vials at all timepoints.
    Returns results DataFrame in DroPhenix-compatible format.
    """
    all_results = []
    total_jobs = len(session.vials) * len(session.timepoints_sec)
    job_num = 0

    for vial in session.vials:
        for t in session.timepoints_sec:
            result = analyze_frame_for_vial(
                session.video_path, vial, t,
                session.t_zero_frame, session.fps,
                session.detection_params
            )
            all_results.append(result)
            job_num += 1
            # Yield progress for Streamlit progress bar (0-1)
            yield result, job_num / total_jobs

    session.results = all_results


def try_detect_reference_line(frame: np.ndarray, roi_x: int, roi_y: int,
                               roi_w: int, roi_h: int) -> Optional[float]:
    """
    Attempt to auto-detect a horizontal black reference line in the vial.
    Returns line_y as fraction of ROI height, or None if not detected confidently.

    Method: horizontal Sobel gradient → find strongest horizontal edge
    in the middle 50% of the vial height (lines are typically at 1/4 to 3/4 height)
    """
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Horizontal gradient (detects horizontal edges)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = np.abs(sobel_y)

    # Sum across each row to find rows with strong horizontal edges
    row_strength = sobel_y.sum(axis=1)

    # Smooth the row strength signal
    kernel_size = max(3, roi_h // 20)
    smoothed = np.convolve(row_strength, np.ones(kernel_size) / kernel_size, mode='same')

    # Search in middle 50% of vial height (avoid top/bottom artifacts)
    search_start = roi_h // 4
    search_end = 3 * roi_h // 4
    search_region = smoothed[search_start:search_end]

    if len(search_region) == 0:
        return None

    peak_idx = np.argmax(search_region)
    peak_y = search_start + peak_idx
    peak_strength = smoothed[peak_y]
    mean_strength = smoothed.mean()

    # Only accept if the peak is at least 3x stronger than average (clear line)
    if peak_strength > mean_strength * 3.0:
        return round(peak_y / roi_h, 3)  # Return as fraction
    return None


def results_to_drophenix_csv(session: VideoAnalysisSession) -> pd.DataFrame:
    """
    Convert analysis results to DroPhenix climbing assay format.

    DroPhenix Climbing CSV columns:
    Genotype, Treatment, Sex, Replicate, Vial_ID, Time_sec,
    Count_climbed, n_total_flies, PI, Confidence, Low_confidence,
    Video_file, Analysis_notes
    """
    if not session.results:
        return pd.DataFrame()

    vial_map = {v.vial_id: v for v in session.vials}
    rows = []
    for r in session.results:
        vial = vial_map.get(r.vial_id)
        if not vial:
            continue
        rows.append({
            'Genotype': vial.genotype,
            'Treatment': vial.treatment,
            'Sex': vial.sex,
            'Replicate': vial.replicate,
            'Vial_ID': r.vial_id,
            'Time_sec': r.time_sec,
            'Count_climbed': r.flies_above,
            'Count_below': r.flies_below,
            'Detected_total': r.flies_detected,
            'n_total_flies': r.n_flies_expected,
            'PI': r.pi,
            'Confidence': r.confidence,
            'Low_confidence': r.low_confidence,
            'Frame_number': r.frame_number,
            'Video_file': Path(session.video_path).name,
            'Analysis_notes': vial.notes,
        })

    return pd.DataFrame(rows)
```

---

## PHASE 2 — Streamlit app page

### Add to app.py PAGES:
```python
"Climbing Video",   # NEW page — after Video Analysis
```

### New page function: page_climbing_video() in app.py

```python
def page_climbing_video():
    """
    Full Streamlit page for climbing assay video analysis.

    Flow:
    1. Video upload
    2. Session metadata form
    3. T=0 frame selection (tap-down moment)
    4. ROI + reference line definition (per vial)
    5. Calibration preview
    6. Full analysis run
    7. Results display + export
    """
    import streamlit as st
    import numpy as np
    from PIL import Image
    import tempfile
    import os
    import cv2
    from modules.climbing_video_analyzer import (
        VideoAnalysisSession, VialConfig, load_video_metadata,
        get_frame_at_number, detect_flies_in_roi, analyze_frame_for_vial,
        results_to_drophenix_csv, try_detect_reference_line, run_full_analysis
    )

    render_header("Climbing Video Analyser",
                  "Count flies above the reference line from recorded assay videos")

    # ── STEP 1: VIDEO UPLOAD ──────────────────────────────────────────────────
    st.subheader("Step 1 — Upload video")
    video_file = st.file_uploader(
        "Upload climbing assay video",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Any format is supported. Videos from phones or lab cameras all work."
    )

    if video_file is None:
        st.info("Upload a video to begin. The video can be recorded with any camera — "
                "phone, lab webcam, or DSLR.")
        return

    # Save uploaded video to temp file (required for OpenCV)
    if 'temp_video_path' not in st.session_state:
        suffix = Path(video_file.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(video_file.read())
        tmp.flush()
        st.session_state['temp_video_path'] = tmp.name

    video_path = st.session_state['temp_video_path']
    meta = load_video_metadata(video_path)
    fps = meta['fps']

    st.success(f"Video loaded: {video_file.name} — "
               f"{meta['total_frames']} frames, {meta['duration_sec']:.1f}s, "
               f"{meta['fps']:.1f} fps, {meta['width']}×{meta['height']}px")

    # ── STEP 2: SESSION METADATA ─────────────────────────────────────────────
    st.subheader("Step 2 — Experiment details")

    col_left, col_right = st.columns(2)
    with col_left:
        n_vials_in_video = st.number_input(
            "Number of vials/cylinders visible in this video",
            min_value=1, max_value=12, value=1, step=1
        )
        n_flies_per_vial = st.number_input(
            "Number of flies per vial (expected total)",
            min_value=1, max_value=50, value=20, step=1
        )
        fly_color = st.selectbox(
            "Fly appearance against background",
            ["Dark flies, light background (typical)",
             "Light flies, dark background"]
        )
        fly_color_key = 'dark' if 'Dark' in fly_color else 'light'

    with col_right:
        genotype_input = st.text_area(
            "Genotype / strain name per vial (one per line — must match number of vials)",
            value="\n".join([f"Genotype_{i+1}" for i in range(n_vials_in_video)]),
            height=120
        )
        replicate_start = st.number_input(
            "Starting replicate number",
            min_value=1, max_value=100, value=1, step=1
        )
        treatment = st.text_input("Treatment", value="vehicle")
        sex = st.selectbox("Sex of flies", ["Mixed", "Male", "Female"])

    genotype_list = [g.strip() for g in genotype_input.strip().split('\n') if g.strip()]
    if len(genotype_list) != n_vials_in_video:
        st.warning(f"You entered {len(genotype_list)} genotype(s) but specified "
                   f"{n_vials_in_video} vials. Adjust to match.")
        genotype_list = (genotype_list + [f"Genotype_{i+1}" for i in range(n_vials_in_video)])[:n_vials_in_video]

    timepoints_str = st.text_input(
        "Analysis timepoints (seconds after tap-down, comma-separated)",
        value="10, 20, 30, 60"
    )
    try:
        timepoints = [float(t.strip()) for t in timepoints_str.split(',') if t.strip()]
    except ValueError:
        st.error("Invalid timepoints — enter numbers separated by commas")
        timepoints = [10, 20, 30, 60]

    # ── STEP 3: TAP-DOWN FRAME SELECTION ─────────────────────────────────────
    st.subheader("Step 3 — Mark the tap-down moment (T=0)")
    st.markdown("Use the slider to find the frame right when the flies are tapped to the bottom.")

    t_zero_sec = st.slider(
        "T=0 position in video (seconds)",
        min_value=0.0,
        max_value=float(meta['duration_sec']),
        value=0.0,
        step=0.5,
        format="%.1f sec"
    )
    t_zero_frame = int(t_zero_sec * fps)
    preview_frame = get_frame_at_number(video_path, t_zero_frame)

    if preview_frame is not None:
        st.image(
            cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB),
            caption=f"Frame at T=0 (frame {t_zero_frame})",
            use_container_width=True
        )
    else:
        st.error("Could not read this frame — try adjusting the slider")

    # ── STEP 4: ROI AND LINE DEFINITION ──────────────────────────────────────
    st.subheader("Step 4 — Define vial regions and reference line")
    st.markdown("""
    **For each vial**, enter the coordinates of its region in the video frame:
    - X_start, Y_start: top-left corner (pixels from top-left of video)
    - Width, Height: size of the vial region in pixels

    **Tip**: Use a photo editor (Paint, Preview, ImageJ) to find pixel coordinates,
    or use the trial-and-error approach with the preview below.

    **Reference line**: The horizontal line flies must cross. Enter as fraction of
    vial height (0.0 = top, 1.0 = bottom). For a line at 1/3 from bottom, use 0.67.
    """)

    vial_configs = []
    for i in range(n_vials_in_video):
        with st.expander(f"Vial {i+1} — {genotype_list[i]}", expanded=(i == 0)):
            col1, col2, col3, col4 = st.columns(4)
            rx = col1.number_input(f"X start (px)", key=f"rx_{i}", value=0, min_value=0)
            ry = col2.number_input(f"Y start (px)", key=f"ry_{i}", value=0, min_value=0)
            rw = col3.number_input(f"Width (px)", key=f"rw_{i}",
                                    value=meta['width'] // n_vials_in_video, min_value=10)
            rh = col4.number_input(f"Height (px)", key=f"rh_{i}",
                                    value=meta['height'], min_value=10)

            # Try auto-detect reference line
            auto_line = None
            if preview_frame is not None and rx + rw <= meta['width'] and ry + rh <= meta['height']:
                auto_line = try_detect_reference_line(preview_frame, rx, ry, rw, rh)
            line_y_frac = st.slider(
                f"Reference line position (0=top, 1=bottom)",
                key=f"line_{i}",
                min_value=0.0, max_value=1.0,
                value=auto_line if auto_line else 0.5,
                step=0.01,
                format="%.2f"
            )
            if auto_line:
                st.caption(f"🔍 Auto-detected a possible line at {auto_line:.2f} — "
                           f"adjust if needed")

            vial_cfg = VialConfig(
                vial_id=i + 1,
                genotype=genotype_list[i],
                replicate=replicate_start + i,
                n_flies_expected=n_flies_per_vial,
                roi_x=rx, roi_y=ry, roi_w=rw, roi_h=rh,
                line_y_frac=line_y_frac,
                fly_color=fly_color_key,
                sex=sex,
                treatment=treatment,
            )
            vial_configs.append(vial_cfg)

    # ── STEP 5: CALIBRATION PREVIEW ───────────────────────────────────────────
    st.subheader("Step 5 — Calibration")
    st.markdown("Check that flies are detected correctly before running the full analysis. "
                "Adjust detection parameters if needed.")

    with st.expander("Advanced detection parameters"):
        col_a, col_b = st.columns(2)
        blur_k = col_a.slider("Blur strength (odd number)", 1, 15, 5, 2,
                               help="Higher = less noise but less detail")
        adapt_c = col_b.slider("Threshold sensitivity", 1, 15, 4, 1,
                                help="Higher = more flies detected (may increase false positives)")
        min_area = col_a.slider("Min fly size (% of ROI)", 0.01, 0.5, 0.03, 0.01)
        max_area = col_b.slider("Max fly size (% of ROI)", 0.5, 5.0, 1.5, 0.1)

    detect_params = {
        'blur_kernel': int(blur_k),
        'adaptive_block': 21,
        'adaptive_c': int(adapt_c),
        'min_area_frac': min_area / 100.0,
        'max_area_frac': max_area / 100.0,
        'morph_open': 2,
        'morph_close': 2,
        'confidence_threshold': 0.70,
    }

    if preview_frame is not None and vial_configs:
        calibration_cols = st.columns(min(n_vials_in_video, 4))
        for i, vial_cfg in enumerate(vial_configs):
            centroids, debug_img = detect_flies_in_roi(preview_frame, vial_cfg, detect_params)
            line_y = vial_cfg.line_y_pixels
            above = sum(1 for (cx, cy) in centroids if cy < line_y)
            below = sum(1 for (cx, cy) in centroids if cy >= line_y)
            total = len(centroids)
            confidence = min(total / max(vial_cfg.n_flies_expected, 1), 1.0)

            col_idx = i % 4
            with calibration_cols[col_idx]:
                st.image(
                    cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                    caption=f"Vial {vial_cfg.vial_id}: {total} detected "
                            f"(expected {vial_cfg.n_flies_expected})\n"
                            f"Above: {above} | Below: {below} | Conf: {confidence:.0%}",
                    use_container_width=True
                )
                if confidence < 0.60:
                    st.warning(f"⚠️ Low detection ({confidence:.0%}) — adjust parameters")
                elif confidence < 0.80:
                    st.info(f"Moderate detection ({confidence:.0%})")
                else:
                    st.success(f"Good detection ({confidence:.0%})")

    # ── STEP 6: RUN FULL ANALYSIS ─────────────────────────────────────────────
    st.subheader("Step 6 — Run analysis")

    if st.button("▶ Run Climbing Assay Analysis", type="primary", use_container_width=True):
        session = VideoAnalysisSession(
            video_path=video_path,
            fps=fps,
            total_frames=meta['total_frames'],
            duration_sec=meta['duration_sec'],
            t_zero_frame=t_zero_frame,
            timepoints_sec=timepoints,
            vials=vial_configs,
            detection_params=detect_params,
        )

        progress_bar = st.progress(0.0, text="Analysing frames...")
        all_results = []
        for result, progress in run_full_analysis(session):
            all_results.append(result)
            progress_bar.progress(progress, text=f"Analysing... {int(progress * 100)}%")
        session.results = all_results
        progress_bar.empty()

        results_df = results_to_drophenix_csv(session)
        st.session_state['climbing_video_results'] = results_df
        st.success(f"✓ Analysis complete — {len(results_df)} data points across "
                   f"{len(vial_configs)} vials × {len(timepoints)} timepoints")

    # ── STEP 7: RESULTS AND EXPORT ────────────────────────────────────────────
    results_df = st.session_state.get('climbing_video_results')
    if results_df is not None and not results_df.empty:
        st.subheader("Step 7 — Results")

        # Summary table: PI by genotype × timepoint
        pivot = results_df.pivot_table(
            index=['Genotype', 'Replicate'],
            columns='Time_sec',
            values='PI',
            aggfunc='mean'
        ).reset_index()
        st.dataframe(pivot, use_container_width=True)

        # Flag low-confidence rows
        low_conf = results_df[results_df['Low_confidence'] == True]
        if not low_conf.empty:
            st.warning(f"⚠️ {len(low_conf)} timepoints had low detection confidence (<70%). "
                       "These are included in the CSV but flagged. "
                       "Consider excluding or re-analyzing these timepoints.")
            with st.expander("Show flagged timepoints"):
                st.dataframe(low_conf[['Genotype', 'Replicate', 'Time_sec',
                                        'Detected_total', 'n_total_flies', 'Confidence']])

        # Plot
        try:
            import plotly.express as px
            fig = px.line(
                results_df[results_df['Low_confidence'] == False],
                x='Time_sec', y='PI',
                color='Genotype',
                line_dash='Replicate',
                error_y=None,
                title='Performance Index over Time',
                labels={'Time_sec': 'Time (seconds)', 'PI': 'Performance Index'},
                markers=True
            )
            fig.add_hline(y=0.5, line_dash='dot', line_color='gray',
                          annotation_text='PI = 0.5')
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # Export buttons
        col_exp1, col_exp2 = st.columns(2)
        csv_bytes = results_df.to_csv(index=False).encode()
        col_exp1.download_button(
            "⬇ Download DroPhenix CSV",
            data=csv_bytes,
            file_name=f"climbing_video_{Path(video_path).stem}.csv",
            mime='text/csv',
            use_container_width=True
        )

        col_exp2.info(
            "To load into DroPhenix Climbing Assay:\n"
            "Go to **Data Input** → **CSV Upload** → select this file"
        )
```

---

## PHASE 3 — Testing protocol

```bash
# Install and run
pip install opencv-python streamlit streamlit-drawable-canvas ffmpeg-python Pillow
streamlit run app.py

# Test with a short synthetic test video:
python - << 'EOF'
import cv2, numpy as np

# Generate a test video: white background, dark circles as "flies"
# 10 seconds at 30fps, one vial visible, circles start low then move up

out = cv2.VideoWriter('test_climbing.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (200, 600))
for frame_num in range(300):
    t = frame_num / 30.0  # seconds
    frame = np.ones((600, 200, 3), dtype=np.uint8) * 230  # light gray
    # Draw reference line in red at y=200 (1/3 from top)
    cv2.line(frame, (0, 200), (199, 200), (0, 0, 0), 3)
    # Draw 20 "flies" (dark circles), rising over time
    n_flies = 20
    for i in range(n_flies):
        y_base = 550 - int(i * 5)  # starting positions at bottom
        y_rise = max(50, y_base - int(t * 30))  # flies rise at 30px/sec
        x_pos = 20 + (i % 9) * 18
        cv2.circle(frame, (x_pos, y_rise), 5, (30, 30, 30), -1)
    out.write(frame)
out.release()
print("Test video written: test_climbing.mp4")
print("Expected: at t=5s, flies above y=200 should be rising")
EOF

# Unit tests
python -c "
from modules.climbing_video_analyzer import *
import numpy as np, cv2

# Test frame reading
meta = load_video_metadata('test_climbing.mp4')
assert meta['total_frames'] > 0, 'frame reading failed'
assert meta['fps'] > 0, 'fps reading failed'

# Test detection
frame = get_frame_at_number('test_climbing.mp4', 150)
assert frame is not None, 'frame extraction failed'

vial = VialConfig(
    vial_id=1, genotype='Test', replicate=1, n_flies_expected=20,
    roi_x=0, roi_y=0, roi_w=200, roi_h=600,
    line_y_frac=0.333,  # y=200 in 600px vial
    fly_color='dark'
)
params = VideoAnalysisSession().detection_params
centroids, debug = detect_flies_in_roi(frame, vial, params)
print(f'Detected {len(centroids)} flies at t=5s')
assert len(centroids) >= 10, f'expected ~20 flies, got {len(centroids)}'
print('All unit tests passed')
"
```

---

## KNOWN LIMITATIONS — be honest with users in the UI

Display this text in an expandable info box on the app page:

```
Known limitations:
• Clustered flies: when all flies are piled at the bottom right after tapping,
  touching flies appear as one blob. Detection improves as flies disperse
  (typically after 5-10 seconds).
• Dark backgrounds with dark flies: adjust threshold in advanced settings.
• Reflective glass: if glare creates false positives, increase minimum fly size.
• Very fast movement + low frame rate (<15fps): some fast-moving flies may be
  missed. Confidence score flags these timepoints.
• Flies outside the ROI: make sure your ROI rectangle captures the full vial height
  including the area above the reference line.
```

---

## OUTPUT FORMAT (exact DroPhenix column names)

```
Genotype       — strain/genotype name entered by user
Treatment      — vehicle, drug name, etc.
Sex            — Male, Female, Mixed
Replicate      — replicate number (integer)
Vial_ID        — vial position in video (1, 2, 3...)
Time_sec       — seconds since tap-down
Count_climbed  — flies detected above reference line
Count_below    — flies detected below reference line
Detected_total — total flies detected (above + below)
n_total_flies  — flies expected (user-entered)
PI             — Count_climbed / n_total_flies
Confidence     — Detected_total / n_total_flies (capped at 1.0)
Low_confidence — True if Confidence < 0.70
Frame_number   — exact video frame number analyzed
Video_file     — source video filename
Analysis_notes — user notes from setup
```

Load this CSV directly into DroPhenix → Data Input → CSV Upload.
DroPhenix automatically computes AUC, t50, CRI, SDQ from this format.

---

## COMMIT MESSAGES

```
git commit -m "Add modules/climbing_video_analyzer.py — core detection engine"
git commit -m "Add page_climbing_video() to app.py — full Streamlit UI"
git commit -m "Add test video generator and unit tests"
git commit -m "Add auto-detection of reference line from frame"
git commit -m "Add low-confidence flagging and Plotly PI time-series chart"
git commit -m "v1.0 complete: Climbing Video Analyser — multi-vial, any lab setup"
```
