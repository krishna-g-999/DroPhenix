# CLAUDE.md — Climbing Video Analyser: Three Priority Fixes
# Continue from existing DroPhenix climbing_video_analyzer.py
# Address in this exact order: denominator → tap detection → band discriminator

## CONTEXT FROM STATUS REPORT
# Current state: MAE 2.69 flies/vial on 22 held-out vials (biological noise floor = 3)
# Three structural problems blocking scientific validity:
# 1. Denominator wrong: 8.7/10 = 87% vs 8.7/20 = 43.5% — different number, same detections
# 2. Tap time unverified: wrong T=0 shifts all timepoint readings
# 3. Band vs label unresolved: two geometry roll-backs failed in opposite directions

## CRITICAL RULES FOR THIS SESSION
# 1. Read existing climbing_video_analyzer.py FULLY before editing any function
# 2. Do NOT touch the band discriminator until fixes 1 and 2 are validated
# 3. The two failed geometry roll-backs are behind `if False` — leave them there with notes
# 4. Every new function needs a unit test before integration
# 5. Run on the existing SP dataset after each fix and compare MAE to 2.69 baseline
# 6. Results go to D:\DroPhenix\DroPhenix-main\results\climbing_video\ as before

## =============================================================================
## FIX 1: DENOMINATOR VERIFICATION (implement first, validate, then proceed)
## =============================================================================

# PROBLEM: The software was accepting user-entered N flies without verification.
# Same detector output = 87% recall vs true 10 flies vs 32% vs assumed 20 flies.
# The number the detector produces did not change. Only the assumed denominator did.

# SOLUTION: Force a verification step between ROI setup and full analysis.
# Detect total flies in the T=0 frame, show count, require user confirmation.

# Add to modules/climbing_video_analyzer.py:

def estimate_total_flies_at_tzero(video_path: str, vial: 'VialConfig',
                                   t_zero_frame: int, params: dict) -> dict:
    """
    Estimate total fly count at T=0 (immediately after tap-down, flies all at bottom).
    This is the moment when ALL flies should be visible and at rest — best count frame.

    At T=0, flies are clustered at bottom. Detection may undercount due to occlusion,
    but this gives the user a floor estimate to compare against their entered N.

    Returns:
        {
            'detected': int,           # flies detected in T=0 frame
            'entered_n': int,          # what user entered
            'discrepancy_pct': float,  # abs difference as % of entered_n
            'flag': bool,              # True if discrepancy > 30%
            'message': str,            # human-readable recommendation
        }
    """
    frame = get_frame_at_number(video_path, t_zero_frame)
    if frame is None:
        return {'detected': -1, 'entered_n': vial.n_flies_expected,
                'discrepancy_pct': 0.0, 'flag': False,
                'message': 'Could not read T=0 frame — verify manually'}

    centroids, _ = detect_flies_in_roi(frame, vial, params)
    detected = len(centroids)
    entered = vial.n_flies_expected

    # At T=0 flies are clustered — detection undercounts by ~15-30% due to occlusion
    # So detected * 1.25 is a rough correction for cluster occlusion
    corrected_estimate = int(detected * 1.25)

    discrepancy = abs(corrected_estimate - entered) / max(entered, 1)
    flag = discrepancy > 0.30  # Flag if corrected estimate is >30% off from entered

    if detected < entered * 0.5:
        msg = (f"WARNING: Detected only {detected} flies at T=0 "
               f"(you entered {entered}). "
               f"Corrected for clustering: ~{corrected_estimate}. "
               f"Please verify actual fly count in your notebook — "
               f"wrong N is the largest error source in this analysis.")
    elif detected > entered * 1.1:
        msg = (f"Note: Detected {detected} flies but you entered {entered}. "
               f"If you entered fewer flies than are actually in the vial, "
               f"PI will be over-estimated. Verify your count.")
    else:
        msg = f"Good: Detected {detected} flies at T=0 vs entered {entered}. Proceeding."

    return {
        'detected': detected,
        'corrected_estimate': corrected_estimate,
        'entered_n': entered,
        'discrepancy_pct': round(discrepancy, 3),
        'flag': flag,
        'message': msg,
    }


# Add to the Streamlit page BETWEEN Step 4 (ROI definition) and Step 6 (run analysis):
# This is a new Step 5B — Denominator Verification

# In the page_climbing_video() function, after vial_configs are defined, add:

"""
# ── STEP 5B: DENOMINATOR VERIFICATION (new — before full analysis) ─────────
st.subheader("Step 5B — Verify fly count (most important step)")
st.warning(
    "**Why this matters**: detecting 8.7 flies = 87% recall if you have 10 flies, "
    "but 43% recall if you assumed 20 flies. The detector does not change. "
    "The denominator must be correct."
)

if preview_frame is not None and vial_configs:
    for vial_cfg in vial_configs:
        check = estimate_total_flies_at_tzero(
            video_path, vial_cfg, t_zero_frame, detect_params
        )
        col_v1, col_v2 = st.columns([2, 1])
        col_v1.write(f"**Vial {vial_cfg.vial_id} — {vial_cfg.genotype}**")
        col_v1.write(check['message'])

        if check['flag']:
            col_v2.error(f"⚠️ Discrepancy: {check['discrepancy_pct']*100:.0f}%")
            confirmed_n = col_v2.number_input(
                "Correct N flies for this vial",
                key=f"confirmed_n_{vial_cfg.vial_id}",
                value=check['corrected_estimate'],
                min_value=1, max_value=50
            )
            vial_cfg.n_flies_expected = confirmed_n
        else:
            col_v2.success(f"✓ Count consistent")
"""


## =============================================================================
## FIX 2: AUTOMATIC TAP-DOWN DETECTION (implement second)
## =============================================================================

# PROBLEM: Wrong T=0 shifts all timepoint measurements.
# T=0 wrong by 2s means "T=10s" is actually T=8s or T=12s — large error in climbing window.

# SOLUTION: Detect the tap event as a spike in frame-difference energy within the vial ROI.
# The tap causes a sudden large motion as all flies drop simultaneously.
# This is detectable as a sharp peak in mean absolute frame difference.

# Add to modules/climbing_video_analyzer.py:

def detect_tap_events(video_path: str, vial: 'VialConfig', fps: float,
                       search_start_sec: float = 0.0,
                       search_duration_sec: float = 30.0) -> dict:
    """
    Detect tap-down event(s) as spikes in frame-to-frame pixel difference
    within the vial ROI.

    The tap causes all flies to suddenly drop to the bottom simultaneously,
    creating a large, brief spike in pixel difference energy.

    Args:
        video_path: path to video
        vial: VialConfig with ROI coordinates
        fps: video fps
        search_start_sec: where to start searching (seconds)
        search_duration_sec: how long to search (seconds)

    Returns:
        {
            'tap_candidates': list of (time_sec, motion_score) sorted by score desc,
            'best_tap_sec': float or None,
            'best_tap_frame': int or None,
            'motion_trace': list of (time_sec, score),  # for plotting
            'confidence': float,  # ratio of peak score to baseline
        }
    """
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    start_frame = int(search_start_sec * fps)
    end_frame = start_frame + int(search_duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    motion_trace = []

    x, y, w, h = vial.roi_x, vial.roi_y, vial.roi_w, vial.roi_h

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Restrict to vial ROI to avoid external movement
        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
            score = float(diff.mean())
            motion_trace.append((frame_num / fps - search_start_sec, score))

        prev_gray = gray

    cap.release()

    if len(motion_trace) < 10:
        return {'tap_candidates': [], 'best_tap_sec': None, 'best_tap_frame': None,
                'motion_trace': motion_trace, 'confidence': 0.0}

    times = np.array([t for t, s in motion_trace])
    scores = np.array([s for t, s in motion_trace])

    # Smooth to reduce per-frame jitter
    kernel = np.ones(int(fps * 0.1) + 1) / (int(fps * 0.1) + 1)
    smoothed = np.convolve(scores, kernel, mode='same')

    # Threshold: mean + 3 SD of baseline (first 5 seconds = no tap yet)
    baseline_n = min(int(fps * 5), len(smoothed) // 3)
    baseline = smoothed[:baseline_n]
    threshold = baseline.mean() + 3.0 * baseline.std()

    # Find peaks above threshold (minimum 2 seconds apart)
    from scipy.signal import find_peaks
    min_distance = int(fps * 2.0)
    peaks, props = find_peaks(smoothed, height=threshold, distance=min_distance)

    tap_candidates = sorted(
        [(float(times[p] + search_start_sec), float(smoothed[p])) for p in peaks],
        key=lambda x: -x[1]
    )

    best_tap_sec = tap_candidates[0][0] if tap_candidates else None
    best_tap_frame = int(best_tap_sec * fps) if best_tap_sec is not None else None
    confidence = (smoothed[peaks[0]] / threshold) if peaks.size > 0 else 0.0

    return {
        'tap_candidates': tap_candidates,
        'best_tap_sec': round(best_tap_sec, 2) if best_tap_sec else None,
        'best_tap_frame': best_tap_frame,
        'motion_trace': list(zip(times.tolist(), scores.tolist())),
        'confidence': round(float(confidence), 2),
    }


# Add to Streamlit page in Step 3 (T=0 selection):
"""
# After the manual T=0 slider, add auto-detect button:
if st.button("🔍 Auto-detect tap event from video motion"):
    if vial_configs:
        with st.spinner("Scanning video for motion spike..."):
            tap_result = detect_tap_events(
                video_path, vial_configs[0], fps,
                search_start_sec=0.0,
                search_duration_sec=min(60.0, meta['duration_sec'])
            )
        if tap_result['best_tap_sec'] is not None:
            st.success(
                f"Detected tap at T={tap_result['best_tap_sec']:.1f}s "
                f"(confidence: {tap_result['confidence']:.1f}x baseline). "
                f"Adjust the slider to confirm."
            )
            # Update slider value via session_state
            st.session_state['suggested_t_zero'] = tap_result['best_tap_sec']
            # Plot motion trace
            import plotly.graph_objects as go
            times_trace = [t for t, s in tap_result['motion_trace']]
            scores_trace = [s for t, s in tap_result['motion_trace']]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times_trace, y=scores_trace,
                                      name='Motion energy', line=dict(color='blue')))
            for tc_sec, tc_score in tap_result['tap_candidates']:
                fig.add_vline(x=tc_sec, line_color='red', line_dash='dash')
            fig.update_layout(title='Motion energy — red lines are tap candidates',
                              xaxis_title='Time (sec)', yaxis_title='Frame difference')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not auto-detect tap — set T=0 manually with the slider")
"""


## =============================================================================
## FIX 3: BAND VS LABEL DISCRIMINATOR (implement third, after 1 and 2 validated)
## =============================================================================

# PROBLEM: Two geometry roll-backs failed in opposite directions.
# Under-refused: Day 7 readout 70% → 34% (accepted label as band)
# Over-refused: Day 7 readout 70% → 91% (rejected actual band)
# Both behind `if False` in code — DO NOT remove them, just add new approach alongside.

# ROOT CAUSE OF BOTH FAILURES: geometry alone is insufficient.
# Both the drawn band and the printed label are horizontal dark features.
# The discriminating features are: coverage, singleness, and texture — not position.

# DISCRIMINATOR LOGIC:
#
# Drawn band properties:
#   1. Coverage:   spans > 80% of vial width with continuous darkness
#   2. Singleness: ONE thick dark row, not a cluster of thin rows
#   3. Thickness:  typically 3-8 pixels thick (marker on glass)
#   4. Texture:    low column variance (uniform, no character patterns)
#
# Printed label properties:
#   1. Coverage:   text characters = discontinuous dark regions, <60% width coverage
#   2. Multiple:   several thin dark rows (text baseline, cap height, descenders)
#   3. Thickness:  text strokes ~1-2 pixels, graduation marks even thinner
#   4. Texture:    HIGH column variance (alternating character/space pattern)

def score_horizontal_candidate_as_band(gray_roi: 'np.ndarray',
                                        candidate_y: int,
                                        dark_threshold: int = 80) -> dict:
    """
    Score a candidate horizontal row as drawn band vs printed label/graduation mark.
    Uses coverage, singleness, and texture — NOT geometry/position.

    Args:
        gray_roi: grayscale crop of the vial ROI
        candidate_y: y-coordinate in ROI pixel space
        dark_threshold: pixel value below which a pixel is considered 'dark'

    Returns:
        {
            'band_score': float 0-1,  # 1.0 = definitely drawn band
            'coverage': float,        # fraction of row width that is dark
            'singleness': float,      # 1.0 = single line, 0.0 = many lines nearby
            'texture_uniformity': float,  # 1.0 = uniform (band-like), 0.0 = textured (label-like)
            'is_band': bool,          # final classification
            'confidence': str,        # 'high', 'medium', 'low'
        }
    """
    import numpy as np
    h, w = gray_roi.shape

    # ── FEATURE 1: Coverage ──────────────────────────────────────────────────
    # Drawn band: dark across >80% of width continuously
    # Label text: dark in patches, <60% coverage
    row = gray_roi[candidate_y, :].astype(float)
    is_dark = row < dark_threshold
    coverage = float(is_dark.mean())

    # Also check for largest CONTINUOUS dark run (band = one long run)
    # Convert to run-length encoding
    transitions = np.diff(is_dark.astype(int))
    run_starts = np.where(transitions == 1)[0] + 1
    run_ends = np.where(transitions == -1)[0] + 1
    if is_dark[0]:
        run_starts = np.concatenate([[0], run_starts])
    if is_dark[-1]:
        run_ends = np.concatenate([run_ends, [w]])
    run_lengths = run_ends - run_starts if len(run_ends) == len(run_starts) else np.array([0])
    max_continuous_run = float(run_lengths.max() / w) if len(run_lengths) > 0 else 0.0

    # ── FEATURE 2: Singleness ────────────────────────────────────────────────
    # Examine ±15 rows around the candidate
    context_half = 15
    y_start = max(0, candidate_y - context_half)
    y_end = min(h, candidate_y + context_half + 1)
    context = gray_roi[y_start:y_end, :]
    row_darkness = (context < dark_threshold).mean(axis=1)

    # Count rows with substantial darkness (>30% of pixels dark)
    substantial_rows = int(np.sum(row_darkness > 0.30))

    # Band: 2-6 dark rows (thickness of marker stroke)
    # Label: many dark rows (full label block) or very few (single graduation mark)
    singleness = 1.0 - min(max(substantial_rows - 4, 0) * 0.08, 1.0)
    # perfect band: 4 substantial rows → singleness = 1.0
    # 15 substantial rows (full label) → singleness = max(0, 1-0.88) = 0.12

    # ── FEATURE 3: Texture uniformity (column variance) ──────────────────────
    # Drawn band: uniform dark stripe, low column variance
    # Label text: characters create high column variance (alternating dark/light)
    if y_end > y_start:
        col_variance = context.var(axis=0)   # variance per column
        # Normalize by the mean column variance in the whole ROI for comparison
        roi_baseline_variance = gray_roi.var(axis=0).mean() + 1e-6
        relative_variance = col_variance.mean() / roi_baseline_variance
        # Low relative variance = uniform = band-like
        texture_uniformity = max(0.0, 1.0 - relative_variance * 0.5)
    else:
        texture_uniformity = 0.5  # unknown

    # ── COMBINED SCORE ────────────────────────────────────────────────────────
    # Weights tuned against the two known failure modes:
    # Weight coverage highest (this is what failed both previous geometry attempts)
    band_score = (
        0.40 * max_continuous_run +   # Most discriminating feature
        0.25 * coverage +              # Overall darkness fraction
        0.20 * singleness +            # One line vs many
        0.15 * texture_uniformity      # Uniform vs textured
    )

    is_band = band_score > 0.55
    confidence = 'high' if abs(band_score - 0.55) > 0.15 else \
                 'medium' if abs(band_score - 0.55) > 0.07 else 'low'

    return {
        'band_score': round(band_score, 3),
        'coverage': round(coverage, 3),
        'max_continuous_run': round(max_continuous_run, 3),
        'singleness': round(singleness, 3),
        'texture_uniformity': round(texture_uniformity, 3),
        'is_band': is_band,
        'confidence': confidence,
    }


def find_and_validate_reference_band(gray_roi: 'np.ndarray',
                                      dark_threshold: int = 80,
                                      search_fraction: tuple = (0.15, 0.85)) -> dict:
    """
    Find the reference band in a vial ROI by:
    1. Finding all candidate horizontal features (strong horizontal edges)
    2. Scoring each with score_horizontal_candidate_as_band()
    3. Returning the best band candidate with its confidence

    Args:
        gray_roi: grayscale vial ROI
        dark_threshold: darkness threshold
        search_fraction: search between these Y fractions (avoid vial caps)

    Returns:
        {
            'found': bool,
            'line_y': int,            # y-coordinate in ROI space
            'line_y_frac': float,     # as fraction of ROI height
            'band_score': float,
            'confidence': str,
            'all_candidates': list,   # all scored candidates for debugging
        }
    """
    import numpy as np
    h, w = gray_roi.shape

    # Find candidate rows using horizontal Sobel
    sobel_y = cv2.Sobel(gray_roi.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
    row_strength = np.abs(sobel_y).sum(axis=1)

    # Smooth
    smooth_k = max(3, h // 30)
    smoothed = np.convolve(row_strength, np.ones(smooth_k) / smooth_k, mode='same')

    # Search range
    y_min = int(search_fraction[0] * h)
    y_max = int(search_fraction[1] * h)
    search_region = smoothed[y_min:y_max]

    if len(search_region) == 0:
        return {'found': False, 'line_y': h // 2, 'line_y_frac': 0.5,
                'band_score': 0.0, 'confidence': 'none', 'all_candidates': []}

    # Find top-5 candidate y positions
    threshold = search_region.mean() + 1.5 * search_region.std()
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(search_region, height=threshold, distance=h // 15)
    top_candidates = sorted(peaks + y_min, key=lambda y: -smoothed[y])[:5]

    # Score each candidate
    scored = []
    for cand_y in top_candidates:
        score_result = score_horizontal_candidate_as_band(gray_roi, cand_y, dark_threshold)
        scored.append({'y': cand_y, **score_result})

    # Sort by band_score
    scored.sort(key=lambda x: -x['band_score'])

    if not scored:
        return {'found': False, 'line_y': h // 2, 'line_y_frac': 0.5,
                'band_score': 0.0, 'confidence': 'none', 'all_candidates': []}

    best = scored[0]
    found = best['is_band']

    return {
        'found': found,
        'line_y': best['y'],
        'line_y_frac': round(best['y'] / h, 3),
        'band_score': best['band_score'],
        'confidence': best['confidence'],
        'all_candidates': scored,
    }


## =============================================================================
## VALIDATION AGAINST KNOWN FAILURE CASES
## =============================================================================

# After implementing each fix, validate against the two known failure cases
# before merging into the main analysis pipeline.

def validate_band_discriminator_on_failures(video_path_day7: str,
                                             vial_cfg_day7: 'VialConfig') -> dict:
    """
    Validate the new band discriminator against the Day 7 failure case.
    Day 7 baseline (before roll-backs): 70% PI
    Under-refused version: 34% PI (accepted label as band)
    Over-refused version: 91% PI (rejected actual band)
    Target: close to 70%

    Run this after implement fix 3 and compare.
    """
    frame = get_frame_at_number(video_path_day7, 0)
    if frame is None:
        return {'error': 'Could not read Day 7 frame'}

    x, y, w, h = vial_cfg_day7.roi_x, vial_cfg_day7.roi_y, vial_cfg_day7.roi_w, vial_cfg_day7.roi_h
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    result = find_and_validate_reference_band(gray)
    print(f"Day 7 validation:")
    print(f"  Found band: {result['found']}")
    print(f"  Line at y={result['line_y']} (frac={result['line_y_frac']:.2f})")
    print(f"  Band score: {result['band_score']:.3f}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  All candidates:")
    for c in result['all_candidates']:
        print(f"    y={c['y']}: score={c['band_score']:.3f}, "
              f"coverage={c['coverage']:.2f}, "
              f"continuous_run={c['max_continuous_run']:.2f}, "
              f"is_band={c['is_band']}")

    return result


## =============================================================================
## RECORDING PROTOCOL RECOMMENDATIONS (for future experiments)
## =============================================================================

# The largest remaining error is OCCLUSION — flies behind the printed label block
# and on the far side of curved glass. On the SP set: detector finds ~8.7/10,
# manual counting at 5× magnification finds ~the same.
# This is a recording issue, not an algorithm issue.

RECORDING_PROTOCOL_RECOMMENDATIONS = """
To maximise fly visibility in climbing assay videos:

1. Backlight the vial from behind with a diffuse white light source (LED panel or
   lightbox). Camera faces the light source through the vial. This creates a
   silhouette effect that makes flies visible regardless of the far-side glass curve.

2. Use plain/unmarked vials for imaging when possible. If a physical reference line
   is needed, use a thin rubber band or washi tape instead of a pen mark — these are
   thicker and easier to discriminate from printed text.

3. If using printed cylinders, position the label to face AWAY from the camera.
   Rotate the cylinder 180° before filming so the printed block is on the far side.

4. Record at minimum 30fps. The climbing window is 0-60 seconds; most informative
   timepoints are 5-30 seconds when flies are actively dispersing.

5. Record a 5-second baseline BEFORE the tap (T=-5s to T=0s). This establishes the
   background model and allows automatic tap detection from the motion spike.
"""


## =============================================================================
## RUNNING ORDER FOR THIS SESSION
## =============================================================================

# Step 1: Read existing climbing_video_analyzer.py fully
# Step 2: Add estimate_total_flies_at_tzero() and the UI step
# Step 3: Run on SP dataset — report new MAE, compare to 2.69 baseline
# Step 4: Add detect_tap_events() and auto-detect UI
# Step 5: Test on a video where tap time is known, verify detection is within 0.5s
# Step 6: ONLY THEN add the band discriminator (score_horizontal_candidate_as_band)
# Step 7: Test discriminator on Day 7 failure case specifically
#         Target: result close to 70% baseline (not 34% or 91%)
# Step 8: If discriminator passes Day 7 test, enable it in the main pipeline
# Step 9: Re-run full SP dataset, report final MAE

## COMMIT MESSAGES
# git commit -m "Fix 1: denominator verification — force user to confirm actual N flies"
# git commit -m "Fix 1: validate on SP dataset, compare MAE to 2.69 baseline"
# git commit -m "Fix 2: detect_tap_events() — auto-detect T=0 from motion spike"
# git commit -m "Fix 2: UI integration — show motion trace, suggest T=0"
# git commit -m "Fix 3: score_horizontal_candidate_as_band() — coverage+singleness+texture"
# git commit -m "Fix 3: validate on Day 7 failure case — target ~70% vs previous 34/91%"
# git commit -m "Fix 3: enable in main pipeline if Day 7 validation passes"
# git commit -m "v1.1: all three structural fixes integrated — final SP MAE reported"
