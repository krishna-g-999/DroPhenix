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

# Register N112. This pointed at a Claude-session temp directory:
#   C:/Users/gkp11/AppData/Local/Temp/claude/D--DroPhenix/ab90d08c-.../scratchpad
# The directory survives; `vial_cache_big.npz` and its meta CSV do not. So the
# re-derivation that `run_all.py` advertises for this suite -- "re-derives its
# held-out accuracy against flies actually loaded rather than asserting it" --
# could no longer run for ANYONE, including the authors, while
# `fly_blob.MEASURED` continued to assert MAE 2.69 as a module-level fact and
# the suite continued to report PASS.
#
# The path is now repository-relative so a rebuilt cache is findable, and
# overridable for a cache kept outside the tree. The cache itself is NOT in the
# repository (it is tens of MB of laboratory video crops).
CACHE = Path(
    __import__("os").environ.get(
        "DROPHENIX_FLY_BLOB_CACHE",
        str(Path(__file__).resolve().parent.parent / ".cache"))
)


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
    """
    The detector's hold-out accuracy, RE-DERIVED FROM PIXELS (register N128).

    This used to read `vial_cache_big.npz` from a Claude-session temp
    directory. That file is gone and no committed script ever built it, so the
    re-derivation `run_all.py` advertises for this suite was lost for the
    authors as well as for a stranger -- and the suite still reported PASS
    (N112).

    It now re-derives from `.cache/climbing_cells.npz`, which exists and is
    rebuildable by `scripts/rebuild_fly_blob_holdout.py`. Both accuracies in
    `MEASURED` are recomputed here and compared with the literal, so the
    literal can no longer drift away from the code.

    On a machine without the cache this SKIPS and says it established nothing,
    rather than passing.
    """
    print("\nC. Hold-out accuracy, re-derived from the cached cells")
    cells = Path(__file__).resolve().parents[1] / ".cache" / "climbing_cells.npz"
    record = (Path(__file__).resolve().parents[1] / "data" / "validation"
              / "fly_blob_holdout_rebuilt.csv")
    if not cells.exists():
        print(f"  [SKIP] {cells.name} is not on this machine, so section C "
              f"established NOTHING about detector accuracy.")
        if record.exists():
            print(f"         The committed record at {record.name} states what "
                  f"was measured where the cache is present; it is a RECORD, "
                  f"not a re-derivation.")
        return

    import json
    z = np.load(cells, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    tot, loaded, above, hand, vials = [], [], [], [], set()
    for m in meta:
        pts = FB.detect(z[m["key"] + "_gray"], z[m["key"] + "_bg"], None)
        line = int(m["line_in_crop"])
        tot.append(len(pts))
        above.append(sum(1 for (_x, cy, _mm) in pts if cy < line))
        loaded.append(float(m.get("n_flies", 0)))
        hand.append(float(m["hand"]))
        vials.add((m["video"], m["vial"]))
    tot = np.asarray(tot, float); loaded = np.asarray(loaded, float)
    above = np.asarray(above, float); hand = np.asarray(hand, float)

    M = FB.MEASURED
    check(f"the cell count matches MEASURED ({M['n_cells']})",
          len(meta) == M["n_cells"], f"{len(meta)} cells")
    check(f"the vial count matches MEASURED ({M['n_vials']})",
          len(vials) == M["n_vials"], f"{len(vials)} vials")

    et, ea = tot - loaded, above - hand
    for label, err, keys in (
            ("n_total vs flies loaded", et,
             ("mae_total", "bias_total", "within3_total")),
            ("n_above vs the hand count", ea,
             ("mae_above", "bias_above", "within3_above"))):
        mae = float(np.mean(np.abs(err)))
        bias = float(np.mean(err))
        w3 = float(np.mean(np.abs(err) <= 3))
        check(f"{label}: MAE re-derives to the stated {M[keys[0]]}",
              abs(mae - M[keys[0]]) < 0.05, f"re-derived {mae:.3f}")
        check(f"{label}: bias re-derives to the stated {M[keys[1]]}",
              abs(bias - M[keys[1]]) < 0.05, f"re-derived {bias:.3f}")
        check(f"{label}: within-3 re-derives to the stated {M[keys[2]]}",
              abs(w3 - M[keys[2]]) < 0.02, f"re-derived {w3:.3f}")

    # The old literal's MAE is NOT reproduced and must not be reinstated.
    check("MEASURED no longer claims the unreproducible 2.69",
          abs(M["mae_total"] - 2.69) > 0.1,
          f"mae_total is {M['mae_total']}, re-derived; the 2.69 covered a "
          f"22-vial subset that is identified nowhere in the repository")
    check("MEASURED declares its provenance and is marked verified",
          bool(M.get("verified")) and "re-derived" in str(M.get("provenance", "")),
          str(M.get("provenance", ""))[:70])


def test_E_ratio_estimator() -> None:
    """
    `count_above_by_ratio`, RE-DERIVED rather than asserted (register N126).

    This is deliberately the opposite of `fly_blob.MEASURED`, whose hold-out
    cache was lost so that no suite can check it (N112). The cells this reads
    are in `.cache/climbing_cells.npz`, which is present, so every number in
    `RATIO_MEASURED` is recomputed here from the pixels. If the estimator
    changes, these fail.

    What is checked is what the estimator CLAIMS: that it beats the published
    bar on MAE and RMSE, and -- the reason it exists -- that it preserves all
    six arm orderings, which peak counting does not. It is NOT claimed to clear
    the spread criterion, and the check below asserts that it does not, so the
    docstring cannot drift into overstating it.
    """
    print("\nE. the ratio estimator, re-derived from the cached cells")
    cells = Path(__file__).resolve().parents[1] / ".cache" / "climbing_cells.npz"
    if not cells.exists():
        print(f"  [SKIP] {cells} is not on this machine; section E "
              f"established nothing")
        return
    import json
    z = np.load(cells, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    hand, est, peak, arms = [], [], [], []
    for m in meta:
        g, b = z[m["key"] + "_gray"], z[m["key"] + "_bg"]
        line, n = int(m["line_in_crop"]), int(m.get("n_flies", 20))
        r = FB.count_above_by_ratio(g, b, line, n)
        if not r["usable"]:
            continue
        pts = FB.detect(g, b, None)
        hand.append(float(m["hand"]))
        est.append(float(r["n_above"]))
        peak.append(float(sum(1 for (_x, cy, _mm) in pts if cy < line)))
        arms.append(f"{m['Genotype']} {m['Treatment']}")
    hand = np.asarray(hand); est = np.asarray(est); peak = np.asarray(peak)
    check(f"every cached cell yields a usable ratio ({len(hand)} of {len(meta)})",
          len(hand) == len(meta), f"{len(hand)} usable")

    err = est - hand
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    claim = FB.RATIO_MEASURED
    check(f"MAE re-derives to the claimed {claim['mae']}",
          abs(mae - claim["mae"]) < 0.05, f"re-derived {mae:.3f}")
    check(f"RMSE re-derives to the claimed {claim['rmse']}",
          abs(rmse - claim["rmse"]) < 0.05, f"re-derived {rmse:.3f}")
    check("it beats the published bar on MAE (3.589)", mae < 3.589,
          f"{mae:.3f}")
    check("it beats the published bar on RMSE (4.436)", rmse < 4.436,
          f"{rmse:.3f}")

    def arm_means(v):
        out = {}
        for a in sorted(set(arms)):
            sel = np.array([x == a for x in arms])
            out[a] = float(v[sel].mean())
        return out

    mh, me, mp = arm_means(hand), arm_means(est), arm_means(peak)
    order = sorted(mh, key=lambda a: mh[a])
    seq_e = [me[a] for a in order]
    seq_p = [mp[a] for a in order]
    mono_e = all(seq_e[i] <= seq_e[i + 1] for i in range(len(seq_e) - 1))
    mono_p = all(seq_p[i] <= seq_p[i + 1] for i in range(len(seq_p) - 1))
    check("the arm means are MONOTONE in the true ordering", mono_e,
          "  ".join(f"{v:.2f}" for v in seq_e))
    # the incumbent is NOT, and that is the whole reason this exists. If peak
    # counting ever becomes monotone here, this claim must be revisited.
    check("peak counting is NOT monotone, which is why this exists",
          not mono_p, "  ".join(f"{v:.2f}" for v in seq_p))

    sh = max(mh.values()) - min(mh.values())
    se = max(me.values()) - min(me.values())
    spread = 100.0 * se / sh
    check("the spread criterion is NOT met, and is not claimed to be",
          spread < 69.0,
          f"recovers {spread:.1f}% of the between-arm range, against the "
          f"bar's 69%. Stated so the docstring cannot drift into claiming it.")


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
    test_E_ratio_estimator()
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
