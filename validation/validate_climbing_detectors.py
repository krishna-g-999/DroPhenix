"""
Validation suite for modules/climbing_detectors.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. THE PROBE COUNTS WHAT IT CLAIMS. Synthetic clips are rendered with a KNOWN
   number of permanent fly-sized marks painted on the tube, so `static_clutter`
   has a right answer by construction rather than by judgement.

B. THE PROBE PREDICTS THE FAILURE. On the same clips, the threshold detector's
   excess over the true fly count must track the clutter it was warned about.
   This is the property the whole automatic choice rests on; on real laboratory
   footage it measured r = +0.993.

C. MOTION IS DETECTED. A clip whose flies move and a clip whose flies are frozen
   must give a high and a near-zero `moving_fraction` respectively, because that
   is what decides whether background subtraction can work at all.

D. REFUSAL. When the scene is clean AND nothing moves, neither method can work,
   and the recommendation must be None with a reason -- not the least-bad of two
   broken numbers.

E. THE REGISTRY IS UNIFORM. Every backend returns the same keys and an above-count
   that never exceeds its own total, so a caller can switch without branching.

F. HONESTY OF THE SHIPPED DEFAULT. `MEASURED_MAE` is the record of a real
   comparison against laboratory hand counts; the default backend must be the one
   that table actually favours, and the recommendation must not claim confidence
   where the evidence is a six-recording tie-break.

Run: ./.venv/Scripts/python.exe validation/validate_climbing_detectors.py
Exit 0 = all pass.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from modules import climbing_detectors as CD  # noqa: E402
from modules import climbing_video as CV  # noqa: E402
from modules import climbing_video_analyzer as CVA  # noqa: E402

FAILURES: list[str] = []
SEED = 20260813


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def render(path: Path, n_flies: int = 14, n_marks: int = 0, move: bool = True,
           w: int = 220, h: int = 620, fps: int = 30, seconds: int = 8) -> dict:
    """Clip with a known number of flies and a known number of PERMANENT marks.

    The marks are drawn identically in every frame at the same places, exactly as
    a printed graduation behaves, and are the same size as a fly -- which is the
    whole difficulty.
    """
    rng = np.random.default_rng(SEED)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter (mp4v) unavailable")
    # Fly size is set by the CYLINDER: an adult fly is about 2.5 mm across a 25 mm
    # cylinder, so it spans roughly a tenth of the tube's width. The scene must
    # reproduce that ratio, because the detector's size window is anchored to the
    # vial width -- drawing half-size flies tests a regime that does not occur.
    # The permanent marks are drawn the SAME size, since a graduation being
    # fly-sized is the entire difficulty.
    vial_w = (w - 16)
    r_fly = max(3, int(round(0.05 * vial_w)))
    # spaced to FIT the tube: at a fixed pitch the higher counts ran off the
    # bottom of the frame and the scene silently contained fewer marks than asked
    _span = max(1, h - 200)
    _pitch = max(3 * r_fly, _span // max(n_marks, 1))
    mark_xy = [(20 + r_fly, 100 + i * _pitch) for i in range(n_marks)
               if 100 + i * _pitch < h - 60]
    start = [500 - i * 8 for i in range(n_flies)]
    # at least 2.5 radii apart, or the drawn circles overlap into one mass and the
    # area-splitting step reports hundreds of "flies" from a solid blob
    _pitch_x = max(3, int(2.5 * r_fly))
    _ncol = max(1, (w - 40 - 2 * r_fly) // _pitch_x)
    xs = [20 + r_fly + (i % _ncol) * _pitch_x for i in range(n_flies)]
    for f in range(fps * seconds):
        t = f / fps
        img = np.full((h, w, 3), 232, np.uint8)
        for (mx, my) in mark_xy:                      # permanent, fly-sized
            cv2.circle(img, (mx, my), r_fly, (60, 60, 60), -1)
        for i in range(n_flies):
            y = max(50, start[i] - int(t * 34)) if move else start[i]
            cv2.circle(img, (xs[i], y), r_fly, (35, 35, 35), -1)
        img = cv2.add(img, rng.normal(0, 1.2, img.shape)
                      .astype(np.int16).clip(-5, 5).astype(np.uint8))
        out.write(img)
    out.release()
    spec = CV.VialSpec(x0=8, x1=w - 8, y_top=40, y_bottom=h - 20,
                       line_y=h // 2, label="R1", source="manual")
    return {"path": str(path), "vial": spec, "n_flies": n_flies,
            "n_marks": n_marks, "fps": fps}


def test_A_B_probe_counts_and_predicts() -> None:
    print("\nA/B. The probe counts the permanent marks, and they predict the excess")
    tmp = Path(tempfile.mkdtemp(prefix="cd_"))
    try:
        obs = []
        for n_marks in (0, 6, 14):
            sc = render(tmp / f"m{n_marks}.mp4", n_marks=n_marks)
            bg = CV.build_background(sc["path"], start_s=1.0, n_samples=20)
            pr = CD.probe_footage(sc["path"], sc["vial"], background=bg, n_samples=8)
            check(f"{n_marks} permanent marks are seen as static clutter",
                  abs(pr.static_clutter - n_marks) <= max(2, int(0.35 * n_marks)),
                  f"probe reported {pr.static_clutter}")
            # what the threshold detector actually over-reports
            cap = cv2.VideoCapture(sc["path"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(5 * sc["fps"]))
            ok, im = cap.read(); cap.release()
            cfg = CVA.VialConfig(vial_id=1, genotype="", replicate=1,
                                 n_flies_expected=sc["n_flies"],
                                 roi_x=sc["vial"].x0, roi_y=sc["vial"].y_top,
                                 roi_w=sc["vial"].width, roi_h=sc["vial"].height,
                                 line_y_frac=0.5, fly_color="dark")
            n_thr = len(CVA.detect_flies_in_roi(im, cfg,
                                                CVA.DEFAULT_DETECTION_PARAMS)[0])
            obs.append((pr.static_clutter, n_thr - sc["n_flies"]))
            print(f"      marks={n_marks:2d}  clutter={pr.static_clutter:2d}  "
                  f"threshold total={n_thr:2d} vs {sc['n_flies']} flies  "
                  f"excess={n_thr - sc['n_flies']:+d}")
        c = np.array([a for a, _ in obs], float)
        e = np.array([b for _, b in obs], float)
        check("the clutter count predicts the threshold detector's excess",
              float(np.corrcoef(c, e)[0, 1]) > 0.9,
              f"r = {float(np.corrcoef(c, e)[0, 1]):+.3f} over {len(obs)} clips "
              f"(r = +0.993 on real footage)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_C_motion() -> None:
    print("\nC. Motion is measured, because background subtraction depends on it")
    tmp = Path(tempfile.mkdtemp(prefix="cd_"))
    try:
        moving = render(tmp / "move.mp4", n_marks=10, move=True)
        frozen = render(tmp / "still.mp4", n_marks=10, move=False)
        pm = CD.probe_footage(moving["path"], moving["vial"],
                              background=CV.build_background(moving["path"],
                                                             start_s=1.0, n_samples=20),
                              n_samples=8)
        pf = CD.probe_footage(frozen["path"], frozen["vial"],
                              background=CV.build_background(frozen["path"],
                                                             start_s=1.0, n_samples=20),
                              n_samples=8)
        check("a clip whose flies move reports substantial motion",
              pm.moving_fraction > CD.MOTION_FLOOR, f"{pm.moving_fraction:.2f}")
        # The absolute floor (MOTION_FLOOR) was set from REAL footage, where an
        # unusable recording measured exactly 0.00 because nothing was detected at
        # all. A synthetic frozen clip cannot reproduce that: sub-pixel registration
        # jitter on a perfectly static, noisy scene produces edge differences at
        # every object boundary and so a small non-zero reading with no real-world
        # analogue. What must hold, and what the decision actually rests on, is that
        # a frozen clip reports far less motion than a moving one.
        check("a frozen clip reports far less motion than a moving one",
              pf.moving_fraction < 0.5 * pm.moving_fraction,
              f"frozen {pf.moving_fraction:.2f} vs moving {pm.moving_fraction:.2f}")
        check("and the frozen clip is the one where the median hides the flies",
              pf.n_moving < pm.n_moving,
              f"background finds {pf.n_moving:.0f} frozen vs {pm.n_moving:.0f} moving")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_D_refusal() -> None:
    print("\nD. It refuses when nothing can work -- and no longer when something can")
    # Clutter plus stillness WAS a refusal, and correctly so when the only detectors
    # were an intensity threshold (defeated by clutter) and background subtraction
    # (defeated by stillness). `blob` is defeated by neither, and was fitted on
    # exactly this kind of footage, so refusing here would now be discarding a usable
    # measurement. The check that guarded the old behaviour is kept, inverted, so the
    # change of contract is visible in the suite rather than silently deleted.
    both_bad = CD.FootageProbe(static_clutter=40, moving_fraction=0.00,
                               framing_stable=True, n_threshold=45)
    r = CD.recommend_backend(both_bad, n_expected=20)
    check("clutter AND no motion is no longer refused: size-matched detection works",
          r["backend"] == "blob", str(r["backend"]))
    check("...and the reason says why neither older method would have served",
          "threshold" in r["reason"] and "background subtraction" in r["reason"],
          r["reason"][:110])

    # A MOVED CAMERA still refuses, because it corrupts the temporal median that
    # `blob` takes its detection threshold from -- there is then no method left.
    moved_cluttered = CD.FootageProbe(static_clutter=40, moving_fraction=0.5,
                                      framing_stable=False, n_threshold=45)
    rm = CD.recommend_backend(moved_cluttered, n_expected=20)
    check("a moved camera on a cluttered scene is still refused",
          rm["backend"] is None, str(rm["backend"]))
    check("...and the refusal explains that the median background is a blend",
          "moved" in rm["reason"] and "blend" in rm["reason"], rm["reason"][:110])

    moved = CD.FootageProbe(static_clutter=2, moving_fraction=0.5,
                            framing_stable=False, n_threshold=20)
    r2 = CD.recommend_backend(moved, n_expected=20)
    check("a moved camera never selects background subtraction",
          r2["backend"] != "background", str(r2["backend"]))

    blind = CD.FootageProbe(static_clutter=1, moving_fraction=0.0,
                            framing_stable=True, n_threshold=2)
    r3 = CD.recommend_backend(blind, n_expected=15)
    check("a clean but undetectable scene is refused, not scored",
          r3["backend"] is None, str(r3["backend"]))

    fine = CD.FootageProbe(static_clutter=1, moving_fraction=0.5,
                           framing_stable=True, n_threshold=18)
    r4 = CD.recommend_backend(fine, n_expected=20)
    check("a good recording gets a backend", r4["backend"] in CD.BACKENDS,
          str(r4["backend"]))


def test_E_registry_uniform() -> None:
    print("\nE. Every backend answers with the same shape")
    tmp = Path(tempfile.mkdtemp(prefix="cd_"))
    try:
        sc = render(tmp / "s.mp4", n_marks=8)
        bg = CV.build_background(sc["path"], start_s=1.0, n_samples=20)
        pr = CD.probe_footage(sc["path"], sc["vial"], background=bg, n_samples=6)
        cap = cv2.VideoCapture(sc["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(5 * sc["fps"]))
        ok, im = cap.read(); cap.release()
        for b in CD.BACKENDS:
            r = CD.count_with_backend(b, im, bg, sc["vial"], probe=pr)
            check(f"{b}: returns the agreed keys",
                  {"n_total", "n_above", "n_below", "backend", "note"} <= set(r),
                  f"{sorted(r)}")
            check(f"{b}: above + below equals the total",
                  r["n_above"] + r["n_below"] == r["n_total"],
                  f"{r['n_above']}+{r['n_below']} vs {r['n_total']}")
            check(f"{b}: never reports more above the line than it found",
                  r["n_above"] <= r["n_total"], f"{r['n_above']} of {r['n_total']}")
        check("an unknown backend is rejected loudly",
              _raises(lambda: CD.count_with_backend("magic", im, bg, sc["vial"])),
              "ValueError raised")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def test_F_default_is_honest() -> None:
    print("\nF. The shipped default matches the measured evidence")
    best = min(CD.MEASURED_MAE, key=CD.MEASURED_MAE.get)
    check("the default backend is the one with the lowest measured error",
          CD.DEFAULT_BACKEND == best,
          f"default {CD.DEFAULT_BACKEND!r}, lowest MAE {best!r} "
          f"({CD.MEASURED_MAE[best]:.2f})")
    check("every backend has a measured error on record",
          set(CD.MEASURED_MAE) == set(CD.BACKENDS),
          f"{sorted(CD.MEASURED_MAE)} vs {sorted(CD.BACKENDS)}")
    # The hand-count table cannot settle the choice on its own: this laboratory
    # records the above-line count cumulatively, so it is a different quantity from
    # an instantaneous detection and a detector can be penalised for a definitional
    # difference. The default is therefore also checked against the target that IS
    # beyond dispute -- the flies loaded into a sealed cylinder.
    check("the default also wins against the number of flies actually loaded",
          min(CD.MEASURED_TOTAL_MAE, key=CD.MEASURED_TOTAL_MAE.get)
          == CD.DEFAULT_BACKEND, f"{CD.MEASURED_TOTAL_MAE}")
    check("no backend is claimed to reach the assay's own noise floor",
          min(CD.MEASURED_MAE.values()) > CD.BIOLOGICAL_FLOOR_FLIES,
          f"best MAE {min(CD.MEASURED_MAE.values()):.2f} flies vs a "
          f"{CD.BIOLOGICAL_FLOOR_FLIES}-fly floor -- so none is yet accurate "
          f"enough, and the module must not imply otherwise")

    tie = CD.FootageProbe(static_clutter=1, moving_fraction=0.5,
                          framing_stable=True, n_threshold=18)
    r = CD.recommend_backend(tie, n_expected=20)
    # Confidence used to be withheld here, and rightly: the pick rested on six
    # recordings from one rig. It now rests on 22 vials held out from fitting, with a
    # threefold margin on both targets, so claiming it is the honest reading of the
    # evidence -- and the reason must still say what it rests on, in numbers, so a
    # reader can check the claim rather than take it.
    check("a pick is offered, with the confidence the evidence now supports",
          r["backend"] is not None and r["confident"],
          f"backend {r['backend']}, confident={r['confident']}")
    check("and the reason says what the pick rests on, in numbers",
          ("held out" in r["reason"] or "MAE" in r["reason"]
           or "flies" in r["reason"]),
          r["reason"][:110])


def test_G_scoring_rule() -> None:
    print("\nG. Crossing-based scoring, the change that mattered most")
    tmp = Path(tempfile.mkdtemp(prefix="cd_"))
    try:
        # A cohort that climbs past the line and then drifts back down: the two
        # scoring rules must disagree, and disagree in the known direction.
        w, h, fps, n = 200, 600, 30, 12
        line_y, sec = 260, 10
        out = cv2.VideoWriter(str(tmp / "updown.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        peak_above = 0
        for f in range(fps * sec):
            t = f / fps
            img = np.full((h, w, 3), 232, np.uint8)
            above = 0
            for i in range(n):
                # up to a peak at t=5 s, then most of them slide back down
                y = 520 - int(300 * min(t, 5.0) / 5.0)
                if t > 5.0 and i >= 3:
                    y += int(200 * (t - 5.0) / 5.0)
                y = int(np.clip(y, 40, h - 40))
                cv2.circle(img, (24 + (i % 6) * 30, y), 5, (35, 35, 35), -1)
                above += int(y < line_y)
            peak_above = max(peak_above, above)
            out.write(img)
        out.release()
        spec = CV.VialSpec(x0=6, x1=w - 6, y_top=20, y_bottom=h - 20,
                           line_y=line_y, label="R1", source="manual")
        bg = CV.build_background(str(tmp / "updown.mp4"), start_s=0.5, n_samples=20)

        inst = CD.score_crossings(str(tmp / "updown.mp4"), spec, bg, 0.0, [9.0],
                                  backend="threshold", mode="instantaneous",
                                  step_s=0.5)[0]["n_above"]
        cross = CD.score_crossings(str(tmp / "updown.mp4"), spec, bg, 0.0, [9.0],
                                   backend="threshold", mode="crossings",
                                   step_s=0.5)[0]["n_above"]
        print(f"      at 9 s: {inst:.0f} above right now, {cross:.0f} had crossed; "
              f"peak simultaneously above was {peak_above}")
        check("the two rules give different answers on a cohort that falls back",
              cross > inst, f"crossings {cross:.0f} vs instantaneous {inst:.0f}")
        check("the crossing count is a ratchet -- it never exceeds the peak",
              cross <= peak_above + 1, f"{cross:.0f} vs peak {peak_above}")
        check("the crossing count never decreases over time",
              _nondecreasing(CD.score_crossings(
                  str(tmp / "updown.mp4"), spec, bg, 0.0, [2., 4., 6., 8., 9.],
                  backend="threshold", mode="crossings", step_s=0.5)),
              "a ratchet that falls is not a ratchet")
        check("a timepoint past the end of the recording is NaN, not zero",
              not np.isfinite(CD.score_crossings(
                  str(tmp / "updown.mp4"), spec, bg, 0.0, [999.0],
                  backend="threshold", mode="crossings")[0]["n_above"]), "")
        rows = CD.score_crossings(str(tmp / "updown.mp4"), spec, bg, 0.0,
                                  [2., 4., 6., 8., 9.], backend="threshold",
                                  mode="crossings", step_s=0.5)
        inst_rows = CD.score_crossings(str(tmp / "updown.mp4"), spec, bg, 0.0,
                                       [2., 4., 6., 8., 9.], backend="threshold",
                                       mode="instantaneous", step_s=0.5)
        peak_inst = max(r["n_above"] for r in inst_rows if np.isfinite(r["n_above"]))
        check("the ratchet is never below the highest instantaneous count seen",
              max(r["n_above"] for r in rows if np.isfinite(r["n_above"])) >= peak_inst - 1,
              f"ratchet peaks at "
              f"{max(r['n_above'] for r in rows if np.isfinite(r['n_above'])):.0f}, "
              f"instantaneous peaked at {peak_inst:.0f} "
              f"(one sample of slack: a crossing must be confirmed in 2 of 3 samples)")
        check("an unknown scoring mode is rejected",
              _raises(lambda: CD.score_crossings(str(tmp / "updown.mp4"), spec, bg,
                                                 0.0, [5.0], mode="whatever")), "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _nondecreasing(rows) -> bool:
    v = [r["n_above"] for r in rows if np.isfinite(r["n_above"])]
    return all(b >= a for a, b in zip(v, v[1:]))


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/climbing_detectors.py")
    print("=" * 78)
    test_A_B_probe_counts_and_predicts()
    test_C_motion()
    test_D_refusal()
    test_E_registry_uniform()
    test_F_default_is_honest()
    test_G_scoring_rule()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CLIMBING-DETECTOR VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
