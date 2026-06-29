"""
validate_video.py
Validation of the DroPhenix climbing-assay VIDEO pipeline
(modules/video_analyzer.py + modules/fly_detector.py).

A computer-vision counter is validated the rigorous way: against synthetic
videos with KNOWN ground-truth fly positions and counts. We render dark flies
(as Drosophila photograph against a lit vial) at controlled positions above/
below the mid-line, then check the analyzer recovers the exact counts and the
exact Performance Index PI = (above - below)/total * 100.

This avoids any dependence on copyrighted/paywalled assay footage (e.g. JoVE),
while testing exactly what matters: detection count accuracy and PI accuracy.

Run:  python validation/validate_video.py   (exit 0 = all passed)
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_climbing_video(path, above_pos, below_pos, W=300, H=400, fps=20, sec=6,
                        bottom_preroll=True):
    """Render a synthetic climbing clip; flies are dark discs on a light vial."""
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = len(above_pos) + len(below_pos)
    for f in range(fps * sec):
        t = f / fps
        frame = np.full((H, W, 3), 235, np.uint8)
        if bottom_preroll and t < 2.0:
            xs = np.linspace(30, W - 30, n).astype(int)
            pts = [(int(x), 330 + (i % 3) * 18) for i, x in enumerate(xs)]
        else:
            pts = list(above_pos) + list(below_pos)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 5, (25, 25, 25), -1)
        vw.write(frame)
    vw.release()


def main() -> int:
    try:
        import cv2  # noqa: F401
    except Exception as e:
        print("OpenCV not available:", e)
        return 1
    from modules.video_analyzer import VideoAnalyzer, get_video_info
    from modules.fly_detector import VialROI, DetectorConfig

    tmp = tempfile.gettempdir()
    cfg = DetectorConfig(min_area=20, max_area=400, threshold_frac=0.5)   # mode='dark'
    az = VideoAnalyzer(cfg)
    results = []

    def chk(name, ok, detail):
        results.append((name, str(detail), bool(ok)))

    # Scenario 1: 8 above, 4 below -> Pct 66.67, PI 33.33
    above = [(40, 60), (110, 60), (180, 60), (250, 60),
             (40, 130), (110, 130), (180, 130), (250, 130)]
    below = [(40, 300), (110, 300), (180, 300), (250, 300)]
    p1 = os.path.join(tmp, "dp_val_climb1.mp4")
    make_climbing_video(p1, above, below)
    info = get_video_info(p1)
    chk("Video opens (CAP bug fix)", info.get("error") is None and info.get("frame_count", 0) > 0,
        f"{info.get('frame_count')} frames")
    df1 = az.analyze(p1, [VialROI(x=0, y=0, w=300, h=400, vial_id=1, expected_n=12)],
                     sample_timepoints=[3, 4, 5])
    ok1 = ((df1.n_above == 8) & (df1.n_below == 4) & (df1.Total == 12)).all()
    chk("Scenario A counts exact (8/4/12)", ok1,
        f"above={list(df1.n_above)} below={list(df1.n_below)}")
    chk("Scenario A PI exact (33.33)", bool((abs(df1.PI - 33.33) < 0.01).all()),
        f"PI={list(df1.PI.round(2))}")

    # Scenario 2: 3 above, 9 below -> Pct 25, PI -50
    above2 = above[:3]
    below2 = below + [(40, 250), (110, 250), (180, 250), (250, 250), (40, 360)]
    p2 = os.path.join(tmp, "dp_val_climb2.mp4")
    make_climbing_video(p2, above2, below2)
    df2 = az.analyze(p2, [VialROI(x=0, y=0, w=300, h=400, vial_id=1, expected_n=12)],
                     sample_timepoints=[3, 5])
    ok2 = ((df2.n_above == 3) & (df2.n_below == 9) & (df2.Total == 12)).all()
    chk("Scenario B counts exact (3/9/12)", ok2,
        f"above={list(df2.n_above)} below={list(df2.n_below)}")
    chk("Scenario B PI exact (-50)", bool((abs(df2.PI + 50) < 0.01).all()),
        f"PI={list(df2.PI.round(2))}")

    # Scenario 3: end-to-end into DroPhenix schema + PI sign sanity
    from modules.video_analyzer import video_counts_to_drophenix
    dd = video_counts_to_drophenix(df1.assign(Genotype="WT", Sex="Male", Treatment="Veh"))
    chk("Converts to DroPhenix schema", set(["Genotype", "Treatment", "Time", "Count", "n"]).issubset(dd.columns),
        list(dd.columns)[:6])

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("DroPhenix climbing-video pipeline validation (synthetic ground truth)")
    print("=" * 72)
    print(f"{'Check':<40}{'Detail':>24}{'Pass':>7}")
    print("-" * 72)
    all_ok = True
    for name, detail, ok in results:
        all_ok &= ok
        print(f"{name:<40}{detail[:24]:>24}{'OK' if ok else 'FAIL':>7}")
    print("-" * 72)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
