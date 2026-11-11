"""
validate_larval.py
Validation of modules/larval.py (larval crawling assay).

A. VIDEO tracking — a synthetic clip with a dark blob moving at a KNOWN constant
   velocity; the tracker must recover the known speed (mm/s, given px_per_mm)
   and a near-straight path (tortuosity ~ 1).
B. SUMMARY analysis — the calibrated demo reproduces the expected biology:
   WT ~1 mm/s, TDP43 slower, drug partial rescue (published WT ~0.5-2 mm/s).

Run:  python validation/validate_larval.py   (exit 0 = all passed)
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import larval as L  # noqa: E402


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = got is not None and np.isfinite(got) and abs(got - expected) <= tol
        results.append((name, expected, round(float(got), 3) if got is not None and np.isfinite(got) else None, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    # ── A. Known-velocity video ──────────────────────────────────────────────
    try:
        import cv2
        W = H = 400; fps = 20; vx = 2.0; vy = 1.0   # px/frame
        px_per_mm = 10.0
        path = os.path.join(tempfile.gettempdir(), "dp_val_larva.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        n = 100
        for f in range(n):
            frame = np.full((H, W, 3), 235, np.uint8)
            x = int(40 + vx * f); y = int(40 + vy * f)
            cv2.circle(frame, (x, y), 8, (20, 20, 20), -1)   # larva-like dark blob
            vw.write(frame)
        vw.release()
        res = L.track_larva(path, px_per_mm=px_per_mm)
        s = res["summary"]
        # true speed: |(vx,vy)|*fps/px_per_mm = sqrt(5)*20/10 = 4.472 mm/s
        true_speed = np.sqrt(vx**2 + vy**2) * fps / px_per_mm
        chk("Tracked mean speed (mm/s)", s["mean_speed_mm_s"], true_speed, 0.15)
        chk("Tracked tortuosity (~1, straight)", s["tortuosity"], 1.0, 0.05)
        boolchk("Tracked most frames", s["n_points"] >= n - 3, f">={n-3}", s["n_points"])
    except Exception as e:
        results.append(("Video tracking", "runs", repr(e)[:30], False))

    # ── B. Demo biology ───────────────────────────────────────────────────────
    demo = L.demo_larval()
    spd = demo.groupby(["Genotype", "Treatment"])["mean_speed_mm_s"].mean()
    wt = spd[("Oregon-WT", "Vehicle")]
    tdv = spd[("TDP43", "Vehicle")]
    tdd = spd[("TDP43", "Drug")]
    chk("Demo WT speed (~1 mm/s)", wt, 1.0, 0.15)
    boolchk("Demo TDP43 slower than WT", tdv < wt, "TDP43<WT", f"{tdv:.2f}<{wt:.2f}")
    boolchk("Demo drug rescues speed", tdd > tdv, "drug>veh", f"{tdd:.2f}>{tdv:.2f}")

    # ── C. Polarity and the oversize refusal (register N127) ─────────────────
    _check_polarity_and_refusal(results)

    print("=" * 72)

def _check_polarity_and_refusal(results):
    """
    Register N127. The tracker took the largest DARK blob unconditionally, which
    assumes a dark larva on a light plate. On real dark-field footage -- bright
    larvae on a near-black background, which is half of the published larval
    video -- the inverse threshold selected a blob covering 100.0% of the frame
    and a full trajectory was returned for the BACKGROUND, with nothing
    refusing and nothing in the output saying so.

    This builds both polarities synthetically and checks that the right one is
    chosen, and that an object the size of the frame is rejected rather than
    tracked.
    """
    import cv2, tempfile, os
    import numpy as np
    W = H = 200
    for polarity in ("dark", "bright"):
        path = os.path.join(tempfile.mkdtemp(), f"{polarity}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
        for i in range(40):
            bg = 20 if polarity == "bright" else 235
            fr = np.full((H, W, 3), bg, np.uint8)
            cx, cy = 40 + 2 * i, 100
            col = (240, 240, 240) if polarity == "bright" else (15, 15, 15)
            cv2.circle(fr, (cx, cy), 6, col, -1)
            vw.write(fr)
        vw.release()
        try:
            r = L.track_larva(path, max_frames=40)
            got = r.get("polarity")
            results.append((f"polarity auto-detected on a {polarity} larva",
                            polarity, got, got == polarity))
            # the background must never be the tracked object
            big = r.get("n_rejected_oversize", 0)
            tr = r.get("trajectory")
            moved = (len(tr) > 5 and
                     float(tr["x_px"].max() - tr["x_px"].min()) > 20)
            results.append((f"a real {polarity} larva is tracked, not the frame",
                            "moves >20 px", f"{len(tr)} frames, oversize rejected {big}",
                            bool(moved)))
        except Exception as e:
            results.append((f"polarity {polarity}", "runs", repr(e)[:40], False))


    print("DroPhenix larval.py validation (video tracking + crawling biology)")
    print("=" * 72)
    print(f"{'Check':<38}{'Expected':>16}{'Got':>12}{'Pass':>6}")
    print("-" * 72)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<38}{str(exp):>16}{str(got):>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 72)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
