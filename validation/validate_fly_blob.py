"""
Validation suite for modules/fly_blob.py -- size-matched fly detection.

WHAT IS VALIDATED, AND WHY THESE THINGS
---------------------------------------
A. THE PROPERTIES THAT MADE THIS METHOD WORTH ADOPTING, on synthetic scenes where the
   answer is known exactly. Each is the direct counterpart of a way the intensity
   threshold failed:
     - a drawn line is NOT a fly, however dark it is (threshold counted it);
     - an illumination gradient across the tube changes nothing (a global threshold
       cannot survive one);
     - an empty tube yields nothing;
     - and, stated rather than claimed away, the RESOLUTION LIMIT: flies closer than
       the fitted separation of 1.2 fly-widths still merge into one detection. Local
       maxima can split them in principle, but on real footage a smaller separation
       cost more in false positives than it recovered, so the fit chose not to.

B. SIZE SELECTIVITY, and its limits. The premise is that the detector responds to
   objects about one fly wide, and the band-pass response must peak at the fly's
   actual size. But a single-scale matched filter DOES respond to a point source --
   that is what a matched filter is -- so a speck is separated by MASS (a twentieth
   of a fly's), not by being invisible; and an object several times a fly's size
   fragments into detections around its rim rather than being ignored. Both are
   pinned here as limits rather than left to be discovered later.

C. REAL FOOTAGE AGAINST GROUND TRUTH THAT CANNOT BE ARGUED WITH. A sealed cylinder
   contains exactly the flies that were put into it, at every timepoint. The
   laboratory's above-the-line hand count is NOT used: that laboratory records it
   cumulatively, so it is a different quantity from an instantaneous detection and
   cannot separate a detector's error from a definitional difference.

   The published figures -- MAE 2.69 for this module against 8.38 for the shipped
   threshold, on 110 cells from 22 held-out vials -- were produced by fitting on 19
   vials and testing on 22 from SEPARATE recordings. This suite re-measures on the
   cached held-out vials when that cache is present, and SKIPS, loudly, when it is
   not. It never asserts the published number without re-deriving it.

D. REFUSAL. Malformed input is refused rather than silently returning zero.

Run: ./.venv/Scripts/python.exe validation/validate_fly_blob.py
Exit 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from modules import fly_blob as FB  # noqa: E402

FAILURES: list[str] = []

CACHE = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/D--DroPhenix/"
             r"ab90d08c-0de9-4002-af04-af9e0767544d/scratchpad")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def scene(w=200, h=700, flies=(), line_y=None, gradient=0.0, marks=(),
          radius=7, noise=1.5, seed=7):
    """A tube: light background, dark flies, optionally a drawn line and marks."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 225, np.float32)
    if gradient:
        img += np.linspace(-gradient, gradient, w)[None, :]
    for (mx, my, mw, mh) in marks:
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), 60, -1)
    if line_y is not None:
        cv2.line(img, (0, line_y), (w - 1, line_y), 20, 3)
    for (fx, fy) in flies:
        cv2.circle(img, (int(fx), int(fy)), radius, 35, -1)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def background_of(w=200, h=700, line_y=None, gradient=0.0, marks=(), seed=11):
    return scene(w, h, flies=(), line_y=line_y, gradient=gradient, marks=marks,
                 noise=1.5, seed=seed)


# ═════════════════════════════════════════════════════════════════════ tests ══
def test_A_the_reasons_it_was_adopted() -> None:
    print("\nA. The failures of the intensity threshold, one by one")

    ys = list(range(90, 640, 55))
    flies = [(40 + 40 * (i % 4), y) for i, y in enumerate(ys)]

    # 1. a drawn line is not a fly
    bg = background_of(line_y=300)
    fr = scene(flies=flies, line_y=300)
    n = len(FB.detect(fr, bg))
    check(f"a drawn scoring line is not counted as a fly ({len(flies)} drawn)",
          n == len(flies), f"detected {n}")

    # 2. THE RESOLUTION LIMIT, asserted rather than assumed. Local maxima can in
    #    principle split touching flies where connected components cannot, but the
    #    minimum separation is a fitted parameter and the fit chose 1.2 fly widths --
    #    on real footage, splitting closer pairs cost more in false positives than it
    #    recovered. So the honest claim is a measured resolution limit, not "touching
    #    flies are resolved". This pins where that limit actually falls.
    bg2 = background_of()
    d = FB.fly_diameter(200)
    resolved = {}
    for gap_frac in (0.9, 1.1, 1.3, 1.6, 2.0):
        gap = int(round(gap_frac * d))
        n = len(FB.detect(scene(flies=[(100, 300), (100, 300 + gap)]), bg2))
        resolved[gap_frac] = n
    check("flies 1.6 fly-widths apart are two detections",
          resolved[1.6] == 2, f"{resolved[1.6]} detected")
    check("flies closer than the fitted separation MERGE (a stated limit, not a bug)",
          resolved[0.9] == 1,
          f"0.9d->{resolved[0.9]}  1.1d->{resolved[1.1]}  1.3d->{resolved[1.3]}  "
          f"1.6d->{resolved[1.6]}  2.0d->{resolved[2.0]} detections")

    # 3. an illumination gradient does not move the count
    bgg = background_of(gradient=35.0)
    frg = scene(flies=flies, gradient=35.0)
    ng = len(FB.detect(frg, bgg))
    check(f"a strong illumination gradient changes nothing ({len(flies)} drawn)",
          ng == len(flies), f"detected {ng}")

    # 4. an empty tube is empty
    ne = len(FB.detect(background_of(line_y=300), background_of(line_y=300, seed=12)))
    check("an empty tube yields no detections", ne == 0, f"detected {ne}")

    # 5. printed graduations are rejected on shape
    ticks = [(6, 120 + 40 * i, 26, 2) for i in range(10)]
    bgm = background_of(marks=ticks)
    frm = scene(flies=flies, marks=ticks)
    nm = len(FB.detect(frm, bgm))
    # Ten ticks leak two detections: 80% are rejected on shape, not all. That is the
    # measured property and it is what the real-footage figure in section C already
    # includes, since those cylinders carry real graduations.
    leak = nm - len(flies)
    check(f"printed graduation ticks are mostly rejected on shape "
          f"({len(flies)} flies + 10 ticks drawn)",
          leak <= 2, f"detected {nm}, i.e. {leak} of 10 ticks leaked through")


def test_B_size_selectivity() -> None:
    print("\nB. It responds to things the size of a fly, and not to other sizes")
    bg = background_of()
    d = FB.fly_diameter(200)
    check("the fly diameter follows the vial width",
          abs(FB.fly_diameter(200) - 14.0) < 0.01
          and abs(FB.fly_diameter(400) - 28.0) < 0.01,
          f"200 px vial -> {FB.fly_diameter(200):.1f} px, "
          f"400 px vial -> {FB.fly_diameter(400):.1f} px")

    ys = list(range(90, 640, 60))
    counts, masses = {}, {}
    for r in (1, int(d / 2), int(3.5 * d)):
        fr = scene(flies=[(100, y) for y in ys], radius=max(r, 1))
        pts = FB.detect(fr, bg)
        counts[r] = len(pts)
        masses[r] = float(np.median([m for (_, _, m) in pts])) if pts else 0.0
    fly_r = int(d / 2)
    check(f"a fly-sized disc (r={fly_r}) is found, all of them",
          counts[fly_r] == len(ys), f"{counts[fly_r]} of {len(ys)}")
    # A single-scale matched filter DOES respond to a point source -- that is what a
    # matched filter is. It is separated by MASS, not by being invisible, and the
    # fitted mass floor deliberately sits below it because raising it cost real flies
    # (minmass 1.0 -> MAE 15.07 on real footage). The limit is stated here.
    check("a one-pixel speck carries far less mass than a fly (how they differ)",
          masses[1] < 0.25 * masses[fly_r],
          f"speck mass {masses[1]:.0f} vs fly {masses[fly_r]:.0f} "
          f"({masses[1]/max(masses[fly_r],1):.2f}x)")
    big = int(3.5 * d)
    check(f"an object several times a fly (r={big}) fragments rather than being "
          f"ignored -- a stated limit",
          counts[big] > len(ys),
          f"{counts[big]} detections from {len(ys)} drawn objects")

    # the band-pass response must peak at the true size
    true_r = 7
    fr = scene(flies=[(100, y) for y in ys], radius=true_r)
    resp = {dd: float(FB.bandpass(fr, dd).max()) for dd in (4, 8, 14, 20, 30, 44)}
    peak = max(resp, key=resp.get)
    check("the band-pass response peaks near the true fly size",
          8 <= peak <= 20, f"peak at d={peak} for a true diameter of {2*true_r}; "
                           + " ".join(f"{k}:{v:.0f}" for k, v in resp.items()))


def test_C_real_footage() -> None:
    print("\nC. Real footage: detected total vs flies actually loaded (sealed vial)")
    npz, meta_csv = CACHE / "vial_cache_big.npz", CACHE / "vial_cache_big_meta.csv"
    if not (npz.exists() and meta_csv.exists()):
        print("  [SKIP] the cached held-out vials are not on this machine; "
              "section C established nothing")
        return
    import pandas as pd
    from modules import climbing_video_analyzer as CVA

    cache = np.load(npz)
    meta = pd.read_csv(meta_csv)
    meta["aspect"] = meta.box_h / meta.box_w.clip(lower=1)
    dropped = meta[meta.aspect < 2.5].key.nunique()
    meta = meta[meta.aspect >= 2.5]
    vids = sorted(meta.video.unique())
    rng = np.random.default_rng(20260821)
    tune = set(rng.choice(vids, size=max(1, len(vids) // 2), replace=False))
    test = meta[~meta.video.isin(tune)]
    print(f"      {dropped} vials excluded: the automatic finder put a box across "
          f"most of the frame (aspect < 2.5), which is a geometry failure, not a "
          f"detection one")

    blob, thr = [], []
    for key, grp in test.groupby("key"):
        bg = cache[key + "|bg"]
        fr = cache[key + "|fr"]
        h, w = bg.shape[:2]
        cfg = CVA.VialConfig(vial_id=1, genotype="x", replicate=1,
                             n_flies_expected=20, roi_x=0, roi_y=0, roi_w=w,
                             roi_h=h, line_y_frac=0.5, fly_color="dark")
        for _, r in grp.iterrows():
            f = fr[int(r.t_index)]
            blob.append(len(FB.detect(f, bg)) - r.n_flies)
            thr.append(len(CVA.detect_flies_in_roi(
                cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), cfg,
                CVA.DEFAULT_DETECTION_PARAMS)[0]) - r.n_flies)
    b, t = np.array(blob, float), np.array(thr, float)
    print(f"      {len(b)} held-out cells from {test.key.nunique()} vials")
    print(f"      blob      MAE {np.abs(b).mean():5.2f}  bias {b.mean():+6.2f}  "
          f"within 3 {(np.abs(b) <= 3).mean():5.1%}")
    print(f"      threshold MAE {np.abs(t).mean():5.2f}  bias {t.mean():+6.2f}  "
          f"within 3 {(np.abs(t) <= 3).mean():5.1%}")
    check("size-matched detection beats the intensity threshold on held-out vials",
          np.abs(b).mean() < np.abs(t).mean(),
          f"MAE {np.abs(b).mean():.2f} vs {np.abs(t).mean():.2f}")
    check("its bias is smaller than the assay's own 3-fly noise floor",
          abs(b.mean()) < 3.0, f"bias {b.mean():+.2f}")
    check(f"the published MAE of {FB.MEASURED['mae_total']} is reproduced within 0.5",
          abs(np.abs(b).mean() - FB.MEASURED["mae_total"]) < 0.5,
          f"measured {np.abs(b).mean():.2f}, published {FB.MEASURED['mae_total']}")


def test_D_refusal() -> None:
    print("\nD. Bad input is refused, not silently answered")
    bg = background_of()
    bad = False
    try:
        FB.detect(bg, bg[:100])
    except ValueError:
        bad = True
    check("a background of a different shape is refused", bad, "ValueError raised")
    bad2 = False
    try:
        FB.detect(np.zeros((50, 50, 3), np.uint8), np.zeros((50, 50, 3), np.uint8))
    except ValueError:
        bad2 = True
    check("a colour image is refused (this works on grey regions)", bad2,
          "ValueError raised")
    flat = np.full((300, 200), 200, np.uint8)
    check("a featureless region yields nothing rather than noise",
          len(FB.detect(flat, flat)) == 0, "")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/fly_blob.py -- size-matched fly detection")
    print("=" * 78)
    test_A_the_reasons_it_was_adopted()
    test_B_size_selectivity()
    test_C_real_footage()
    test_D_refusal()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL FLY-BLOB VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
