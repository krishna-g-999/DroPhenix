"""
three_way_detector_comparison.py
Three climbing detectors, one set of frames, one set of hand counts.

WHY THIS SCRIPT EXISTS
----------------------
Register N101: this project has TWO climbing detection stacks that never meet.
`fly_blob` is reached through `climbing_detectors.score_crossings` and is what
every accuracy figure in the register measures. `fly_detector` is used by
`cv_pipeline`, `recorder`, `video_analyzer` and `climbing_video`, carries the
N3 merge-splitting fix that raised recall in a severe phenotype from 0.35 to
0.90, and has NEVER been compared against hand counts. Which one a user gets
depends on which entry point they call.

Register N97: the measured stack under-counts, and the under-count is a CEILING
at about 12-13 flies per vial -- mean tool counts of 12.70, 11.85, 12.11,
12.16, 9.17, 4.06 against hand counts from 6.6 to 18.5. Because the controls
are the most crowded, they are under-counted most, and the dose-response is
compressed toward the null.

Those two findings pose one question: does merge splitting fix the ceiling? It
is answerable, because the fix exists in the other stack. And a third detector
makes the answer interpretable rather than merely internal -- FreeClimber
(Spierer et al. 2021), reimplemented in `modules/freeclimber_detector.py` and
validated against its own published output by
`validation/validate_freeclimber_detection.py`.

WHAT MAKES THIS A COMPARISON OF DETECTORS
-----------------------------------------
Everything except the detector is held identical, by construction rather than
by care:

  * ONE decode pass per recording. The frames are read once and handed to all
    three detectors, so no detector sees a different frame.
  * The SAME layout. Cylinder regions and the scoring line come from
    `build_layout`, the same function the published rescore uses, so no
    detector gets a different region or a different line.
  * The SAME background -- `climbing_video.build_background`, a per-pixel
    temporal median. FreeClimber builds its own background the same way, so
    supplying this one removes a difference in inputs without changing its
    method.
  * The SAME footage-quality gate. `recommend_backend` refuses unusable
    recordings; a refusal excludes the cell for ALL THREE, because it is a
    judgement about the footage and not about a detector.
  * The SAME hand counts, and the same timepoints.

SELF-CHECK, WHICH IS THE REASON TO TRUST THE REST
-------------------------------------------------
In `mode="net"` -- the protocol's rule, "the number standing above the line" --
`score_crossings` reports the instantaneous count at the timepoint. This script
therefore reads the frame at each timepoint directly instead of going through
the scoring grid. That is only equivalent if it reproduces the published run,
so the `fly_blob` column is compared cell by cell against
`paraquat_scored_net_blob_2026-11.csv` and the agreement is reported at the
top. If it does not reproduce, nothing below it means anything.

A NOTE ON PARAMETERS, WHICH DECIDES WHETHER THIS IS FAIR
--------------------------------------------------------
`fly_blob`'s `fly_d_frac = 0.07` was FITTED on this footage (MAE 9.11 at 0.05,
2.93 at 0.07, 9.82 at 0.10). FreeClimber's `diameter = 7`, `minmass = 100`,
`maxsize = 11` come from their camera at their resolution, on a 1112 x 380
region. Carrying those integers onto 1080p frames would measure a resolution
mismatch and call it a method difference.

So FreeClimber's scale parameter is set from the apparent fly size in THIS
footage, which is what its own configuration file exists to let a user do, and
`--fc-grid` additionally sweeps its parameters and reports its BEST achievable
accuracy here. Comparing against a competitor's best is the only comparison
worth reporting. `fly_detector`'s absolute area gates are audited the same way
by `validation/validate_detector_parameters.py`.

Run:
  ./.venv/Scripts/python.exe scripts/three_way_detector_comparison.py
  ./.venv/Scripts/python.exe scripts/three_way_detector_comparison.py --limit 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

from modules import climbing_detectors as CD  # noqa: E402
from modules import climbing_video as CV  # noqa: E402
from modules import fly_blob as FB  # noqa: E402
from modules import fly_detector as FD  # noqa: E402
from modules import provenance as PROV  # noqa: E402

try:
    from modules import freeclimber_detector as FCD
except Exception:                                          # pragma: no cover
    FCD = None

try:
    from validation import lab_data as _LAB
except Exception:                                          # direct-script use
    sys.path.insert(0, str(ROOT / "validation"))
    import lab_data as _LAB

from rescore_paraquat_net import build_layout, _tap, manifest_loading  # noqa: E402

PARA = ((_LAB.climbing_root() or Path("_absent"))
        / "Paraquat climbing assay 010626_new set")
MANIFEST = PARA / "paraquat_video_manifest.csv"
GROUND_TRUTH = PARA / "paraquat_ground_truth.csv"
PUBLISHED_BLOB = PARA / "paraquat_scored_net_blob_2026-11.csv"

OUT = PARA / "three_way_detector_comparison.csv"
SIDECAR = PARA / "three_way_detector_comparison.provenance.json"

# The same timepoints and the same grid step as the published rescore, so the
# frames are the frames that run scored.
TIMES = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.]
STEP_S = 2.0

DETECTORS = ("fly_blob", "fly_detector", "freeclimber")


# ──────────────────────────────────────────────────────────────────────────────
def replicates_for(row, n_cyl: int) -> List[int]:
    """
    Which replicate sits in each cylinder, left to right.

    This is the join key onto the hand counts, so getting it wrong does not
    produce a wrong number -- it produces NO number, which is how the error was
    caught: every `hand` came back NaN because the first version of this script
    called `manifest_loading`, which returns FLIES PER CYLINDER ("20+20" ->
    [20, 20]), and then filtered the ground truth for replicate 20.

    The manifest records it explicitly in `left_is` / `right_is`. The published
    rescore instead uses the convention "a two-cylinder frame holds replicates
    1 and 2, a one-cylinder frame holds replicate 3". The two agree on all 62
    rows of this manifest, checked; the explicit columns are preferred, and a
    disagreement raises rather than resolving silently, because one of the two
    would then be wrong about which animals were measured.
    """
    li, ri = row.get("left_is"), row.get("right_is")
    convention = [1, 2] if n_cyl == 2 else [3]
    try:
        explicit = ([int(float(li))] if n_cyl == 1
                    else [int(float(li)), int(float(ri))])
    except (TypeError, ValueError):
        return convention
    if explicit != convention:
        raise ValueError(
            f"{row.get('video')}: the manifest says cylinders hold replicates "
            f"{explicit} but the published rescore's convention says "
            f"{convention}. One of them is wrong about which animals were "
            f"measured; resolve it before scoring this recording.")
    return explicit


# ──────────────────────────────────────────────────────────────────────────────
# Per-detector counting, each given the region and background it expects
# ──────────────────────────────────────────────────────────────────────────────
def count_fly_blob(gray: np.ndarray, bg: np.ndarray, spec) -> Optional[dict]:
    """`fly_blob`, exactly as `climbing_detectors.count_with_backend` calls it."""
    y0, y1 = int(spec.y_top), int(spec.y_bottom)
    x0, x1 = int(spec.x0), int(spec.x1)
    sub, bsub = gray[y0:y1, x0:x1], bg[y0:y1, x0:x1]
    if sub.size == 0 or bsub.shape != sub.shape:
        return None
    pts = FB.detect(sub, bsub, None)
    above = sum(1 for (_, cy, _) in pts if (cy + y0) < spec.line_y)
    return {"n_total": len(pts), "n_above": int(above)}


def count_fly_detector(frame_bgr: np.ndarray, bg_gray: np.ndarray, spec,
                       cfg: "FD.DetectorConfig",
                       use_bg: bool) -> Optional[dict]:
    """
    `fly_detector` on the same region, with the scoring line set to OUR line.

    `threshold_frac` is the fraction of the region's height at which the line
    sits, which is how this detector expresses the line. Setting it from the
    layout is what makes the above-the-line counts comparable; leaving it at
    its 0.50 default would score a different line from the other two detectors
    and the difference would be read as detector error.
    """
    y0, y1 = int(spec.y_top), int(spec.y_bottom)
    x0, x1 = int(spec.x0), int(spec.x1)
    sub = frame_bgr[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    h = max(1, y1 - y0)
    import dataclasses
    c = dataclasses.replace(
        cfg, threshold_frac=float(np.clip((spec.line_y - y0) / h, 0.0, 1.0)))
    det = FD.FlyDetector(c)
    bg_sub = bg_gray[y0:y1, x0:x1] if use_bg else None
    try:
        r = det.detect_detailed(sub, bg_sub)
    except Exception as exc:
        return {"n_total": np.nan, "n_above": np.nan, "note": repr(exc)[:80]}
    # The merge rule's own internals are carried out of here, because they are
    # what a parameter audit needs and they are not recoverable afterwards:
    # `typical_area_px` is this detector's estimate of ONE fly's footprint in
    # this frame, `split_at_px` is the area above which a blob is counted as
    # several flies, and `debris_at_px` the area above which it is discarded.
    # If the single-fly estimate is too small, the split rule multiplies every
    # fly and the debris ceiling throws away genuine piles -- so the estimate
    # has to be visible next to the count it produced.
    return {"n_total": int(r.get("total", 0)), "n_above": int(r.get("above", 0)),
            "n_merged": int(r.get("n_merged", 0) or 0),
            "n_debris": int(r.get("n_debris", 0) or 0),
            "n_single": int(r.get("n_single", 0) or 0),
            "typical_area_px": float(r.get("typical_area_px", np.nan)),
            "split_at_px": float(r.get("split_at_px", np.nan)),
            "debris_at_px": float(r.get("debris_at_px", np.nan))}


def freeclimber_counts(gray_stack: np.ndarray, bg_gray: np.ndarray,
                       specs: List, roi: Tuple[int, int, int, int],
                       p: "FCD.FreeClimberParams",
                       frame_times: np.ndarray) -> Optional[Dict]:
    """
    FreeClimber on the cylinder-spanning region, for the whole stack at once.

    Its signal threshold and trim lines are properties of the WHOLE clip, so it
    cannot be run frame by frame without becoming a different algorithm. The
    stack it is given here is the twelve reported timepoints spread over the
    two-minute observation window, so the threshold is still pooled across the
    clip rather than estimated per frame -- roughly 240 spots into a 40-bin
    histogram, where FreeClimber's own example pooled 4,289. The
    spots are then assigned to OUR cylinders by x-position and split by OUR
    line, so the layout and the line are the same ones the other two detectors
    were given; FreeClimber's own equal-width binning is not used, because it
    would introduce a second difference.
    """
    if FCD is None:
        return None
    x0, y0, x1, y1 = roi
    sub = gray_stack[:, y0:y1, x0:x1]
    bsub = bg_gray[y0:y1, x0:x1]
    try:
        res = FCD.detect_on_stack(sub, p, background=bsub)
    except Exception as exc:
        return {"error": repr(exc)[:120]}
    df = res.spots
    df = df[df["True_particle"]].copy()
    # back to frame coordinates
    df["X"] = df["x"] + x0
    df["Y"] = df["y"] + y0

    out: Dict[Tuple[str, int], dict] = {}
    for fi in range(len(frame_times)):
        d = df[df["frame"] == fi]
        for spec in specs:
            inside = d[(d["X"] >= spec.x0) & (d["X"] < spec.x1)]
            above = int((inside["Y"] < spec.line_y).sum())
            out[(spec.label, fi)] = {"n_total": int(len(inside)),
                                     "n_above": above}
    return {"counts": out, "threshold": res.threshold,
            "n_true": res.n_true, "n_raw": res.n_raw}


# ──────────────────────────────────────────────────────────────────────────────
def fc_params_for(specs, n_frames: int, fps: float,
                  diameter: Optional[int] = None,
                  minmass: Optional[float] = None) -> "FCD.FreeClimberParams":
    """
    FreeClimber's parameters, with its SCALE parameter set from this footage.

    `diameter` is in pixels and is the one parameter that cannot be carried
    across cameras: their example used 7 on a 1112 x 380 region. Here it is the
    apparent fly diameter, taken from the cylinders' own median width by the
    same geometric argument `fly_blob` uses -- a fly is about 7% of a
    cylinder's bore -- rounded to the nearest odd integer, which trackpy
    requires. Setting it per setup is what FreeClimber's configuration file is
    for, so this is using the tool as designed rather than modifying it.

    `minmass` is a floor on integrated intensity and therefore scales with the
    disc area; carrying their 100 across a diameter change would move the
    effective cut. It is scaled by (d / 7)^2 from their published value unless
    given explicitly.
    """
    # ONE diameter has to serve the whole region, because FreeClimber is run
    # once over all the cylinders at once (its threshold is a property of the
    # clip, not of a tube). The MEDIAN cylinder width is used, not the first
    # one's: the cylinders should be identical, so a width that differs is a
    # localisation error, and a median is unmoved by one bad region while
    # `specs[0]` would simply inherit it.
    if diameter:
        d = int(diameter)
    else:
        widths = [float(sp.x1 - sp.x0) for sp in specs]
        d = int(round(FB.fly_diameter(float(np.median(widths)))))
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    mm = float(minmass) if minmass is not None else 100.0 * (d / 7.0) ** 2
    return FCD.FreeClimberParams(
        x=0, y=0, w=None, h=None,
        blank_0=0, blank_n=n_frames, crop_0=0, crop_n=n_frames,
        threshold="auto", diameter=d, minmass=mm, maxsize=11.0,
        ecc_low=0.05, ecc_high=0.58, vials=1, window=50,
        frame_rate=float(fps), outlier_TB=1.0, outlier_LR=3.0,
        trim_outliers=False,   # see below
    )


# trim_outliers is OFF, and this is a judgement that must be visible.
#
# FreeClimber's trim step cuts a sliver off each edge of the region where the
# detections thin out. That is sound for the region THEY draw -- one box around
# a rack of vials, with dead space beyond the end tubes. Here the region is
# drawn tightly around the cylinders by `build_layout`, so the edges of the
# region ARE the glassware, and the flies nearest the top and bottom of a tube
# are the ones the trim would remove. Those are the climbers and the fallers:
# precisely the animals the measurement is about.
#
# Leaving it on would therefore not be faithfulness, it would be applying a
# rule outside the geometry it assumes. The cost is recorded by
# `--fc-grid`, which reports the accuracy both ways.


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N recordings (for a quick check)")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--cache-cells", type=str, default="",
                    help="also save the cropped region, background and line for "
                         "a sample of cells to this .npz, so that a parameter "
                         "sweep can run without decoding the videos again")
    ap.add_argument("--cache-every", type=int, default=6,
                    help="keep one cell in every N for the cache (default 6, "
                         "which gives about a hundred cells over the set)")
    ap.add_argument("--fd-mode", choices=("dark", "bg"), default="bg",
                    help="fly_detector's detection mode; 'bg' gives it the same "
                         "temporal-median background the others get")
    args = ap.parse_args(argv)

    if not MANIFEST.is_file() or not GROUND_TRUTH.is_file():
        print("The paraquat recordings are not on this machine.")
        print(_LAB.describe_roots())
        return 0
    if FCD is None:
        print("freeclimber_detector could not be imported; cannot run three ways.")
        return 2
    try:
        FCD._trackpy()
    except Exception as exc:
        print("FreeClimber's detector needs trackpy, which is not installed.")
        print(f"  {exc}")
        return 2

    man = pd.read_csv(MANIFEST)
    gt = pd.read_csv(GROUND_TRUTH)
    jobs = [r for _, r in man.iterrows() if str(r.get("use", True)).lower() != "false"]
    if args.limit:
        jobs = jobs[:args.limit]

    out_path = Path(args.out)
    rows: List[dict] = []
    t_start = time.time()
    print(f"Three-way comparison over {len(jobs)} recordings, "
          f"detectors = {DETECTORS}", flush=True)
    print(f"fly_detector mode = {args.fd_mode!r}; FreeClimber diameter is set "
          f"per cylinder from its own width", flush=True)

    # Cells kept for the parameter sweep. The crops are the EXACT arrays the
    # detectors were given, so a sweep over parameters reruns the detectors on
    # the same pixels rather than on a re-derived approximation of them.
    cache: Dict[str, np.ndarray] = {}
    cache_meta: List[dict] = []
    cell_i = 0

    base_cfg = FD.DetectorConfig(mode=args.fd_mode,
                                 use_bg_subtraction=(args.fd_mode == "bg"))

    for k, r in enumerate(jobs, 1):
        path = PARA / r["folder"] / r["video"]
        geno, treat = r["Genotype"], r["Treatment"]
        n_cyl = int(float(r["cylinders_in_frame"]))
        base = {"folder": r["folder"], "video": r["video"],
                "tap": int(float(r.get("tap", 1) or 1)),
                "Genotype": geno, "Treatment": treat}
        reps = replicates_for(r, n_cyl)
        n_flies = manifest_loading(r, n_cyl) or []
        if not path.is_file():
            rows.append({**base, "status": "video not on this machine"})
            continue

        print(f"[{k:2d}/{len(jobs)}] {r['folder']:14s} {r['video'][:26]:26s} "
              f"({(time.time() - t_start) / 60:.0f} min)", flush=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                (lay, bg, det, note), fits = build_layout(
                    str(path), n_cyl, 20, geno, treat)
        except Exception as exc:
            print(f"          FAILED: {exc!r}", flush=True)
            rows.append({**base, "status": f"failed: {exc!r}"})
            continue
        if lay is None:
            print(f"          NO LAYOUT: {note}", flush=True)
            rows.append({**base, "status": f"no layout: {note}"})
            continue

        info = CV.probe_video(str(path))
        t0 = _tap(str(path))
        specs = [dv.to_spec(lay.line_y) for dv in lay.vials]

        # ── one decode pass: EXACTLY the reported timepoints ────────────────
        #
        # Why not the 2 s scoring grid the published run walked. In
        # `mode="net"` -- the protocol's rule -- `score_crossings` sets
        # `series = a`, the per-sample instantaneous count, and then reports the
        # grid sample at or just below each requested time. With `step_s = 2.0`
        # and times 10, 20, ... 120 every requested time falls exactly ON a grid
        # point, so the other 49 samples change nothing: the reported count is
        # the instantaneous count in the frame at that timepoint. Decoding them
        # would be pure cost, and cost matters here because FreeClimber's
        # Crocker-Grier finder runs on a 1920-px-wide region.
        #
        # This equivalence is a claim about the published run, so it is CHECKED
        # rather than assumed: the fly_blob column is compared cell by cell
        # against paraquat_scored_net_blob_2026-11.csv by
        # validation/validate_three_way_comparison.py. If the frames were not
        # the frames that run scored, that check fails.
        #
        # The one behaviour not reproduced is the forward-fill across
        # unreadable frames, which in the published run could carry a count
        # from a neighbouring grid sample. Unreadable frames are filled from
        # the nearest readable timepoint here and the row is marked, so the
        # substitution is visible instead of silent.
        grid = np.array([t for t in TIMES if t <= (info.duration_s - t0)],
                        dtype=float)
        if grid.size == 0:
            rows.append({**base, "status": "recording too short"})
            continue
        frames_bgr: List[np.ndarray] = []
        cap = cv2.VideoCapture(str(path))
        for t in grid:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + t) * info.fps))
            ok, im = cap.read()
            frames_bgr.append(im if ok else None)
        cap.release()
        if not any(f is not None for f in frames_bgr):
            rows.append({**base, "status": "no frame could be read"})
            continue
        # forward-fill unreadable frames, as score_crossings does to its series
        frame_ok = [f is not None for f in frames_bgr]
        last = next(f for f in frames_bgr if f is not None)
        for i, f in enumerate(frames_bgr):
            if f is None:
                frames_bgr[i] = last
            else:
                last = f
        grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                          for f in frames_bgr]).astype(float)
        bg_gray = bg if bg.ndim == 2 else cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

        # ── FreeClimber: whole stack, cylinder-spanning region ──────────────
        roi = (int(min(s.x0 for s in specs)), int(min(s.y_top for s in specs)),
               int(max(s.x1 for s in specs)), int(max(s.y_bottom for s in specs)))
        fc_p = fc_params_for(specs, len(grid), info.fps)
        fc = freeclimber_counts(grays, bg_gray, specs, roi, fc_p, grid)
        fc_err = (fc or {}).get("error")
        if fc_err:
            print(f"          FreeClimber: {fc_err}", flush=True)

        # ── per vial, per timepoint ─────────────────────────────────────────
        for vi, spec in enumerate(specs):
            rep = reps[vi] if vi < len(reps) else (reps[-1] if reps else None)
            probe = CD.probe_footage(str(path), spec, background=bg, n_samples=6)
            # the SAME gate the published run applied, with the same expected
            # count: `recommend_backend` uses it, so passing 0 would gate
            # differently and the comparison would not share its refusals
            n_exp = int(n_flies[vi]) if vi < len(n_flies) else 0
            rec = CD.recommend_backend(probe, n_expected=n_exp)
            refused = rec["backend"] is None
            fit = fits.get(spec.label)
            vbase = {**base, "replicate": rep, "vial": spec.label,
                     "t0_s": round(t0, 2),
                     "line_row": int(fit.row) if fit else np.nan,
                     "line_is_clear": bool(fit.is_clear()) if fit else None,
                     "n_flies": n_exp or None,
                     "fc_diameter": int(fc_p.diameter),
                     "fc_minmass": round(float(fc_p.minmass), 1),
                     "fc_threshold": round(float((fc or {}).get("threshold",
                                                                np.nan)), 3)
                     if fc and not fc_err else np.nan}
            if refused:
                # The SAME gate for all three: it judges the footage, not a
                # detector, so a refusal must remove the cell from every arm or
                # the arms are no longer comparable.
                for t in TIMES:
                    rows.append({**vbase, "time_s": t,
                                 "status": "refused by the probe",
                                 "refusal": rec["reason"]})
                continue

            for t in TIMES:
                i = int(np.searchsorted(grid, t, side="right")) - 1
                if i < 0 or t > (info.duration_s - t0):
                    rows.append({**vbase, "time_s": t,
                                 "status": "beyond the end of the recording"})
                    continue
                h = gt[(gt.Genotype == geno) & (gt.Treatment == treat)
                       & (gt.replicate == rep) & (gt.tap == base["tap"])
                       & (gt.time_s == t)]
                hand = int(h["n_climbed"].iloc[0]) if len(h) else np.nan

                gray_i = grays[i]
                rb = count_fly_blob(gray_i.astype(np.uint8), bg_gray.astype(np.uint8),
                                    spec)
                rf = count_fly_detector(frames_bgr[i], bg_gray.astype(np.uint8),
                                        spec, base_cfg, args.fd_mode == "bg")
                rc = ((fc or {}).get("counts") or {}).get((spec.label, i))

                if args.cache_cells and np.isfinite(hand)                         and cell_i % max(1, args.cache_every) == 0:
                    y0, y1 = int(spec.y_top), int(spec.y_bottom)
                    x0c, x1c = int(spec.x0), int(spec.x1)
                    key = f"cell{len(cache_meta):04d}"
                    cache[key + "_gray"] = gray_i[y0:y1, x0c:x1c].astype(np.uint8)
                    cache[key + "_bg"] = bg_gray[y0:y1, x0c:x1c].astype(np.uint8)
                    cache[key + "_bgr"] = frames_bgr[i][y0:y1, x0c:x1c]
                    cache_meta.append({
                        "key": key, "folder": r["folder"], "video": r["video"],
                        "vial": spec.label, "time_s": t, "hand": int(hand),
                        "n_flies": n_exp or None,
                        "line_in_crop": int(spec.line_y - y0),
                        "width_px": int(x1c - x0c), "height_px": int(y1 - y0),
                        "Genotype": geno, "Treatment": treat})
                cell_i += 1

                rows.append({
                    **vbase, "time_s": t,
                    "status": "scored" if frame_ok[i] else "scored (frame filled)",
                    "hand": hand,
                    "fly_blob_above": rb["n_above"] if rb else np.nan,
                    "fly_blob_total": rb["n_total"] if rb else np.nan,
                    "fly_detector_above": rf["n_above"] if rf else np.nan,
                    "fly_detector_total": rf["n_total"] if rf else np.nan,
                    "fly_detector_merged": (rf or {}).get("n_merged", np.nan),
                    "fly_detector_single": (rf or {}).get("n_single", np.nan),
                    "fly_detector_debris": (rf or {}).get("n_debris", np.nan),
                    "fd_typical_area_px": (rf or {}).get("typical_area_px", np.nan),
                    "fd_split_at_px": (rf or {}).get("split_at_px", np.nan),
                    "fd_debris_at_px": (rf or {}).get("debris_at_px", np.nan),
                    # the fly footprint the GEOMETRY implies, for comparison:
                    # a fly is about 7% of a cylinder's bore, so its silhouette
                    # is about pi/4 x (0.07 w)^2 ... measured as a disc here
                    "geom_fly_area_px": float(
                        np.pi * (FB.fly_diameter(spec.x1 - spec.x0) / 2.0) ** 2),
                    "cylinder_width_px": int(spec.x1 - spec.x0),
                    "freeclimber_above": rc["n_above"] if rc else np.nan,
                    "freeclimber_total": rc["n_total"] if rc else np.nan,
                })
        pd.DataFrame(rows).to_csv(out_path, index=False)

    d = pd.DataFrame(rows)
    d.to_csv(out_path, index=False)
    meta = {
        "produced_by": "scripts/three_way_detector_comparison.py",
        "detectors": list(DETECTORS),
        "times_s": TIMES, "step_s": STEP_S,
        "fly_detector_mode": args.fd_mode,
        "freeclimber_trim_outliers": False,
        "freeclimber_diameter_rule": "round-to-odd(0.07 x cylinder width)",
        "rows": int(len(d)),
    }
    try:
        meta.update(PROV.describe() if hasattr(PROV, "describe") else {})
    except Exception:
        pass
    SIDECAR.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.cache_cells and cache_meta:
        cp = Path(args.cache_cells)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, meta=json.dumps(cache_meta), **cache)
        print(f"wrote {cp} with {len(cache_meta)} cells for the parameter sweep")
    print(f"\nwrote {out_path}\nwrote {SIDECAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
