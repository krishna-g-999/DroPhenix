"""
study_detector_parameters.py
Every parameter of every climbing detector: what it is, where it came from,
whether it does anything, and whether it is set correctly for this footage.

WHY THIS SCRIPT EXISTS
----------------------
A detector comparison is only a comparison of methods if each method's
parameters are right for the data. Otherwise it measures a misconfiguration and
reports it as a property of the algorithm. Three specific hazards here, each of
which is checked below rather than argued about:

1. `fly_blob`'s `fly_d_frac = 0.07` was FITTED ON THIS FOOTAGE. Its own module
   records the curve: MAE 9.11 at 0.05, 5.26 at 0.06, 2.93 at 0.07, 5.92 at
   0.085, 9.82 at 0.10. A detector tuned on the data it is then scored on has
   an advantage that has to be stated, and the sweep below re-measures that
   curve so the size of the advantage is visible.

2. `fly_detector`'s area bounds are ABSOLUTE PIXEL AREAS -- `min_area = 20`,
   `max_area = 800` px^2. An absolute area is correct at one zoom and one
   resolution and nowhere else. Worse, its merge rule divides blob area by an
   ESTIMATE of one fly's area, so if that estimate is wrong every count is
   wrong by the same factor: too small an estimate and each fly is counted
   several times; too large and a pile is counted as one.

3. FreeClimber's `diameter = 7`, `minmass = 100`, `maxsize = 11` come from
   their camera on a 1112 x 380 region. Carried onto 1920 x 1080 frames they
   would measure a resolution mismatch. The comparison scales `diameter` from
   the cylinder's own width; this script checks that the scaling helps, and by
   how much, rather than assuming it.

WHAT IS MEASURED
----------------
A. PROVENANCE. Every parameter of all three detectors, with where its value
   came from -- geometry, a fit on this footage, a published configuration, or
   a bare default nobody chose. A parameter whose provenance is "default" is
   not necessarily wrong, but it is unexamined, and the table says so.

B. DOES IT BIND? Each parameter is varied and the counts recomputed on the
   cached cells. A parameter that changes nothing over a wide range is inert
   here -- which is worth knowing, because it is then not the knob to turn. One
   that changes everything is the dominant one and deserves the scrutiny.

C. IS IT SET WELL? Where hand counts exist the sweep reports MAE against them,
   so the shipped value can be compared with the best value on this footage.
   A shipped value far from the optimum is a finding; a shipped value AT the
   optimum on the same data it was fitted to is a warning, not a result.

D. THE SINGLE-FLY AREA, which is the one number `fly_detector`'s merge rule
   cannot survive being wrong about. Its own per-frame estimate is compared
   against the footprint the geometry implies.

INPUT
    The cell cache written by
      scripts/three_way_detector_comparison.py --cache-cells <path>
    Each cell is the EXACT cropped region, background and scoring line the
    detectors were given, plus the hand count, so a sweep reruns the detectors
    on the same pixels instead of on a re-derivation of them.

Run:
  ./.venv/Scripts/python.exe validation/study_detector_parameters.py \\
      --cells .cache/climbing_cells.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import fly_blob as FB  # noqa: E402
from modules import fly_detector as FD  # noqa: E402

try:
    from modules import freeclimber_detector as FCD
except Exception:                                          # pragma: no cover
    FCD = None

DEFAULT_CELLS = ROOT / ".cache" / "climbing_cells.npz"


# ──────────────────────────────────────────────────────────────────────────────
# A. Provenance. Written out by hand because provenance cannot be introspected:
#    a float in a dataclass does not record who chose it or why.
# ──────────────────────────────────────────────────────────────────────────────
PROVENANCE: Dict[str, Dict[str, str]] = {
    "fly_blob": {
        "fly_d_frac": "FITTED on this footage (9.11/5.26/2.93/5.92/9.82 at "
                      "0.05/0.06/0.07/0.085/0.10) -- scale-free, a fraction of "
                      "the cylinder's own width",
        "null_q": "FITTED on this footage (10.59/2.93/6.80 at 99.0/99.5/99.8)",
        "thr_mult": "default 1.0, never varied",
        "sep_frac": "GEOMETRIC: a fly is longer than it is wide, so two maxima "
                    "must be more than one fly-width apart",
        "noise_mult": "chosen as a guard, not fitted; only binds when the empty "
                      "tube is featureless",
        "minmass_mult": "default, not fitted",
        "ecc_window_frac": "FITTED compromise (1.0); the measured trade is "
                           "held-out MAE 2.69 with the shape filter against "
                           "2.43 without it",
        "max_ecc": "MEASURED from the thing it must reject: drawn line 3.40, "
                   "fly 1.00, fly standing on the line 2.55 -- hence 2.8",
        "mask_marks": "OFF (0.0) by decision, see the module docstring",
        "mark_q": "inert while mask_marks is 0",
        "mark_r_frac": "inert while mask_marks is 0",
    },
    "fly_detector": {
        "threshold_frac": "SET BY THE CALLER from the layout's scoring line; "
                          "its 0.50 default would score a different line from "
                          "the other detectors",
        "min_area": "ABSOLUTE px^2 default (20). Scale-dependent: correct at "
                    "one zoom only",
        "max_area": "ABSOLUTE px^2 default (800). Scale-dependent, and since "
                    "N3 it is a debris ceiling rather than a fly filter",
        "split_merged": "N3 fix, on by default: a blob larger than one fly is "
                        "counted as the number of flies its area implies",
        "merge_split_factor": "1.6, taken from climbing_video for consistency",
        "debris_area_mult": "8.0, chosen; it is the BINDING ceiling whenever "
                            "8 x the single-fly estimate is below max_area",
        "max_merge_multiplicity": "kept equal to debris_area_mult by "
                                  "construction so the two rules agree",
        "mode": "'dark' default (Otsu inverse, needs no background) or 'bg' "
                "(absdiff against a supplied background, thresholded at a "
                "FIXED 25 grey levels -- itself an unexamined constant)",
        "use_bg_subtraction": "default True, but INERT whenever `mode` is "
                              "'dark' or a background frame is supplied: the "
                              "mode precedence in detect_detailed consults it "
                              "only as the third branch, so it selects MOG2 "
                              "and nothing else",
        "use_morphology": "default True",
        "morph_kernel_size": "default 3 px, absolute and therefore "
                             "scale-dependent",
    },
    "freeclimber": {
        "diameter": "PUBLISHED as 7 for a 1112 x 380 region; in pixels, so it "
                    "CANNOT be carried across cameras. Scaled here from the "
                    "cylinder's own width, which is what their config file is "
                    "for",
        "minmass": "PUBLISHED as 100 at diameter 7; an integrated intensity, "
                   "so it scales with disc area -- rescaled as 100 (d/7)^2",
        "maxsize": "PUBLISHED 11",
        "ecc_low": "PUBLISHED 0.05",
        "ecc_high": "PUBLISHED 0.58",
        "threshold": "'auto' -- and the published auto rule returns a HISTOGRAM "
                     "BIN INDEX applied to photometric values (register N100), "
                     "reproduced deliberately",
        "trim_outliers": "PUBLISHED True, but OFF here: their trim assumes a "
                         "region with dead space beyond the end tubes, while "
                         "these regions are drawn tightly around the glass, so "
                         "it would remove the climbers and the fallers",
        "outlier_TB": "PUBLISHED 1.0, inert while trim_outliers is off",
        "outlier_LR": "PUBLISHED 3.0, inert while trim_outliers is off",
    },
}


def load_cells(path: Path) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    if not path.is_file():
        return [], {}
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arrays = {k: z[k] for k in z.files if k != "meta"}
    return meta, arrays


# ──────────────────────────────────────────────────────────────────────────────
# Counting one cached cell with each detector, under a parameter override
# ──────────────────────────────────────────────────────────────────────────────
def count_blob(cell: dict, arrays: Dict[str, np.ndarray],
               params: Optional[dict]) -> float:
    g = arrays[cell["key"] + "_gray"]
    b = arrays[cell["key"] + "_bg"]
    pts = FB.detect(g, b, params)
    line = int(cell["line_in_crop"])
    return float(sum(1 for (_, cy, _) in pts if cy < line))


def count_fd(cell: dict, arrays: Dict[str, np.ndarray],
             overrides: Optional[dict] = None) -> float:
    bgr = arrays[cell["key"] + "_bgr"]
    bg = arrays[cell["key"] + "_bg"]
    h = max(1, int(cell["height_px"]))
    kw = dict(mode="bg", use_bg_subtraction=True,
              threshold_frac=float(np.clip(cell["line_in_crop"] / h, 0.0, 1.0)))
    kw.update(overrides or {})
    cfg = FD.DetectorConfig(**kw)
    det = FD.FlyDetector(cfg)
    r = det.detect_detailed(bgr, bg)
    return float(r.get("above", np.nan))


def count_fc(cell: dict, arrays: Dict[str, np.ndarray],
             overrides: Optional[dict] = None,
             threshold: float = 25.0) -> float:
    """
    FreeClimber on one cached cell.

    Its automatic threshold is a property of a whole clip, so in a sweep it is
    pinned to a fixed value: a sweep must vary ONE thing, and letting the
    threshold float would confound every row. The pinned value is the median
    the full comparison run recorded, and `--fc-threshold` overrides it.
    """
    if FCD is None:
        return float("nan")
    g = arrays[cell["key"] + "_gray"].astype(float)[None, :, :]
    bg = arrays[cell["key"] + "_bg"].astype(float)
    kw = dict(x=0, y=0, w=None, h=None, blank_0=0, blank_n=1, crop_0=0, crop_n=1,
              threshold=float(threshold), maxsize=11.0, ecc_low=0.05,
              ecc_high=0.58, vials=1, frame_rate=30.0, trim_outliers=False)
    d = int(round(FB.fly_diameter(float(cell["width_px"]))))
    kw["diameter"] = d + (1 - d % 2)
    kw["minmass"] = 100.0 * (kw["diameter"] / 7.0) ** 2
    kw.update(overrides or {})
    try:
        res = FCD.detect_on_stack(g, FCD.FreeClimberParams(**kw), background=bg)
    except Exception:
        return float("nan")
    df = res.spots
    df = df[df["True_particle"]]
    return float((df["y"] < int(cell["line_in_crop"])).sum())


def sweep(name: str, counter: Callable, cells: List[dict],
          arrays: Dict[str, np.ndarray], param: str,
          values: Sequence, baseline) -> None:
    """Vary one parameter, recount every cell, report MAE against the hand counts."""
    hand = np.array([c["hand"] for c in cells], dtype=float)
    base_counts = None
    print("    %-22s %10s %8s %9s %9s %s"
          % (param, "value", "MAE", "bias", "changed", ""))
    for v in values:
        try:
            counts = np.array([counter(c, arrays, {param: v}) for c in cells],
                              dtype=float)
        except Exception as exc:
            print("    %-22s %10s  raised %r" % ("", v, exc))
            continue
        ok = np.isfinite(counts) & np.isfinite(hand)
        if ok.sum() == 0:
            print("    %-22s %10s  no usable cell" % ("", v))
            continue
        e = counts[ok] - hand[ok]
        if base_counts is None and v == baseline:
            base_counts = counts
        changed = ("" if base_counts is None
                   else "%d/%d cells" % (int(np.sum(counts != base_counts)),
                                         counts.size))
        mark = "  <- shipped" if v == baseline else ""
        print("    %-22s %10s %8.3f %+9.3f %9s%s"
              % ("", str(v), float(np.mean(np.abs(e))), float(np.mean(e)),
                 changed, mark))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=str, default=str(DEFAULT_CELLS))
    ap.add_argument("--fc-threshold", type=float, default=25.0)
    ap.add_argument("--skip-freeclimber", action="store_true")
    args = ap.parse_args(argv)

    # ── A. provenance, which needs no data ──────────────────────────────────
    print("=" * 92)
    print("A. EVERY PARAMETER, AND WHERE ITS VALUE CAME FROM")
    print("=" * 92)
    live = {
        "fly_blob": dict(FB.PARAMS),
        "fly_detector": dataclasses.asdict(FD.DetectorConfig()),
        "freeclimber": (dataclasses.asdict(FCD.FreeClimberParams())
                        if FCD else {}),
    }
    missing: List[str] = []
    for det, params in live.items():
        print(f"\n{det}:")
        for k, v in sorted(params.items()):
            if k in ("vial_rois", "vial_preset", "sample_interval_sec",
                     "brightness", "contrast", "bg_history", "bg_var_threshold",
                     "x", "y", "w", "h", "blank_0", "blank_n", "crop_0",
                     "crop_n", "vials", "window", "frame_rate"):
                continue           # layout/plumbing, not detection parameters
            prov = PROVENANCE.get(det, {}).get(k)
            if prov is None:
                missing.append(f"{det}.{k}")
                prov = "** NO PROVENANCE RECORDED **"
            print(f"  {k:24s} = {str(v):>8s}   {prov}")
    if missing:
        print("\n  PARAMETERS WITH NO RECORDED PROVENANCE: %s" % ", ".join(missing))
        print("  Each of these is a number nobody in this project has explained.")
    else:
        print("\n  Every detection parameter of all three detectors has a "
              "recorded provenance.")

    cells, arrays = load_cells(Path(args.cells))
    if not cells:
        print()
        print("=" * 92)
        print("No cell cache, so B-D establish nothing. Build one with:")
        print("  ./.venv/Scripts/python.exe scripts/three_way_detector_comparison.py \\")
        print("      --cache-cells .cache/climbing_cells.npz")
        return 0

    hand = np.array([c["hand"] for c in cells], dtype=float)
    print()
    print("=" * 92)
    print(f"{len(cells)} cached cells, hand counts {hand.min():.0f}-{hand.max():.0f} "
          f"(mean {hand.mean():.2f})")
    widths = np.array([c["width_px"] for c in cells], dtype=float)
    print(f"cylinder widths {widths.min():.0f}-{widths.max():.0f} px "
          f"(median {np.median(widths):.0f})")

    # ── D. the single-fly area, first, because the merge rule rests on it ───
    print()
    print("=" * 92)
    print("D. THE SINGLE-FLY AREA: fly_detector's estimate vs the geometry")
    print("=" * 92)
    print("fly_detector divides blob area by its own estimate of ONE fly's")
    print("footprint. If that estimate is too small, every fly is counted several")
    print("times; too large, and a pile is counted as one. The geometry says a")
    print("fly is about 7% of a cylinder's bore across.")
    print()
    print("  %-26s %8s %9s %9s %9s %8s %8s"
          % ("cell", "width", "geom a", "fd est", "ratio", "split@", "debris@"))
    ratios = []
    for c in cells[:12]:
        bgr = arrays[c["key"] + "_bgr"]
        bg = arrays[c["key"] + "_bg"]
        h = max(1, int(c["height_px"]))
        cfg = FD.DetectorConfig(
            mode="bg", use_bg_subtraction=True,
            threshold_frac=float(np.clip(c["line_in_crop"] / h, 0.0, 1.0)))
        r = FD.FlyDetector(cfg).detect_detailed(bgr, bg)
        d = FB.fly_diameter(float(c["width_px"]))
        geom = np.pi * (d / 2.0) ** 2
        est = float(r.get("typical_area_px", np.nan))
        ratios.append(geom / est if est and np.isfinite(est) else np.nan)
        print("  %-26s %8d %9.0f %9.0f %9.2f %8.0f %8.0f"
              % ((c["video"][:14] + " " + c["vial"]), c["width_px"], geom, est,
                 geom / est if est else np.nan,
                 float(r.get("split_at_px", np.nan)),
                 float(r.get("debris_at_px", np.nan))))
    rr = np.array([x for x in ratios if np.isfinite(x)])
    if rr.size:
        print()
        print("  geometry / fly_detector's estimate: median %.2f (range %.2f-%.2f)"
              % (np.median(rr), rr.min(), rr.max()))
        if np.median(rr) > 1.5:
            print("  -> fly_detector thinks a fly is MUCH SMALLER than the geometry")
            print("     implies. Its split rule will then treat a single fly as a")
            print("     merge and count it as round(area/estimate) animals, which")
            print("     OVER-counts by about that ratio.")
        elif np.median(rr) < 0.67:
            print("  -> fly_detector thinks a fly is much LARGER than the geometry")
            print("     implies, so merged blobs will be counted as one animal and")
            print("     the count will saturate.")
        else:
            print("  -> the two agree to within a factor of 1.5, so the split rule")
            print("     is operating on a sound estimate of one fly.")

    # ── B/C. the sweeps ─────────────────────────────────────────────────────
    print()
    print("=" * 92)
    print("B/C. DOES EACH PARAMETER BIND, AND IS THE SHIPPED VALUE GOOD HERE?")
    print("=" * 92)
    print("`changed` counts cells whose count differs from the shipped setting.")
    print("A parameter that changes no cell over its whole range is inert here.")

    print("\n  fly_blob  (NOTE: fly_d_frac and null_q were FITTED on this footage,")
    print("             so a minimum at the shipped value is not independent")
    print("             evidence that the value is right)")
    for param, values, base in [
        ("fly_d_frac", [0.05, 0.06, 0.07, 0.085, 0.10], 0.07),
        ("null_q", [99.0, 99.5, 99.8], 99.5),
        ("sep_frac", [1.0, 1.2, 1.5], 1.2),
        ("max_ecc", [1.8, 2.8, 3.5, 99.0], 2.8),
        ("minmass_mult", [0.25, 0.5, 1.0], 0.5),
        ("noise_mult", [3.0, 6.0, 12.0], 6.0),
        ("thr_mult", [0.8, 1.0, 1.3], 1.0),
    ]:
        sweep("fly_blob", count_blob, cells, arrays, param, values, base)

    print("\n  fly_detector")
    for param, values, base in [
        ("min_area", [5, 20, 50, 100], 20),
        ("max_area", [200, 800, 3000], 800),
        ("split_merged", [False, True], True),
        ("merge_split_factor", [1.2, 1.6, 2.5], 1.6),
        ("debris_area_mult", [4.0, 8.0, 20.0], 8.0),
        ("mode", ["dark", "bg"], "bg"),
        ("use_morphology", [False, True], True),
        ("morph_kernel_size", [3, 5], 3),
    ]:
        sweep("fly_detector", count_fd, cells, arrays, param, values, base)

    if FCD is not None and not args.skip_freeclimber:
        print("\n  freeclimber  (threshold pinned at %.1f so each row varies ONE "
              "thing)" % args.fc_threshold)

        def _fc(c, a, o):
            return count_fc(c, a, o, threshold=args.fc_threshold)

        for param, values, base in [
            ("diameter", [7, 11, 15, 21], None),
            ("minmass", [25.0, 100.0, 400.0, 1000.0], None),
            ("maxsize", [5.0, 11.0, 30.0], 11.0),
            ("ecc_high", [0.3, 0.58, 1.0], 0.58),
            ("threshold", [10.0, 25.0, 45.0], args.fc_threshold),
            ("trim_outliers", [False, True], False),
        ]:
            sweep("freeclimber", _fc, cells, arrays, param, values, base)
        print("    diameter/minmass have no single shipped value here: they are")
        print("    derived per cylinder from its width, so the row marked as")
        print("    shipped is absent by design.")

    print()
    print("=" * 92)
    print("WHAT THIS DOES NOT ESTABLISH")
    print("=" * 92)
    print("A minimum in a sweep is not a validated setting. These cells are a")
    print("sample of one laboratory's footage, two of fly_blob's parameters were")
    print("fitted on exactly this data, and the hand counts are one scorer's")
    print("reading with no second scorer, so the floor on measurable accuracy is")
    print("unknown. A parameter should be changed on this evidence only if the")
    print("change is also argued from geometry or from the published method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
