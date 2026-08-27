"""Score every paraquat recording under the PROTOCOL'S rule, and record that it did.

WHY THIS SCRIPT EXISTS
----------------------
Two things went wrong that this fixes together.

1. THE RULE. The published negative-geotaxis protocols say: "if a fly climbs back
   down or falls, record that fly as -1 and count the next fly to cross the line as
   the same number." The tally therefore steps BACK when a fly falls, so the net of
   up- and down-crossings is simply how many are STANDING above the line. That is
   `mode="net"`. For a period this project used a ratchet (`max_reached`) instead --
   a count that never decreases and is therefore always at least as large. Anything
   scored under the ratchet is a different quantity from the hand counts it was
   compared against.

2. THE PROVENANCE. None of the scored tables recorded WHICH rule produced them, so
   the only way to tell a ratchet table from a protocol table was to find the script
   that wrote it -- and for two of them that script no longer exists. Every row this
   script writes carries `scoring_mode`, `backend` and `step_s`, and a sidecar JSON
   records the library versions, because an OpenCV minor version changes contour and
   threshold behaviour.

WHAT "DRAWN" MEANS HERE, AND WHAT IT DOES NOT
---------------------------------------------
The layouts are CONSTRUCTED, not literally drawn by a person: the cylinder is
located automatically, then the box is extended to cover the glassware the way an
operator would. That is a fair stand-in for a hand-drawn box and it is honest about
being one. Where the cylinder is not found at all, nothing is scored and the row
says so, rather than a region being invented.

THE SCORING LINE
----------------
Taken from `climbing_simple.find_line`, which requires the dark feature to STOP at
the glass. The rule this replaces took the globally darkest horizontal feature and
put the line on the bench mat at the 10 mL graduation -- about 400 px low, on 12 of
12 vials -- while reporting margins of 1.02-1.06. Each vial's line is reported with
its depth and margin, and `line_is_clear` says whether the software would have asked
an operator to confirm it. Nothing is discarded on that basis here; it is recorded so
the accuracy can be read separately for the vials the software was sure about.

Run: ./.venv/Scripts/python.exe scripts/rescore_paraquat_net.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from modules import climbing_detectors as CD  # noqa: E402
from modules import climbing_layout as LAY  # noqa: E402
from modules import climbing_simple as CS  # noqa: E402
from modules import climbing_video as CV  # noqa: E402
from modules import provenance as PROV  # noqa: E402

B = Path(r"D:/DroPhenix/Testing data/Climbing assay/"
         r"Paraquat climbing assay 010626_new set")
OUT = B / "paraquat_scored_net_blob.csv"
SIDECAR = B / "paraquat_scored_net_blob.provenance.json"

TIMES = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.]
SCORING_MODE = "net"       # the protocol: up minus down = the standing count
# Size-matched blob detection. On 22 vials held out from its own fitting its vial
# total was out by 2.69 flies against the threshold's 8.38, and its above-the-line
# count by 3.53 against 5.42 -- see modules/fly_blob.py.
BACKEND = "blob"
STEP_S = 2.0


def build_layout(path: str, n_cyl: int, n_flies: int, geno: str, treat: str):
    """Locate the cylinders and lay a generous box over each, as an operator would."""
    info = CV.probe_video(path)
    bg = CV.build_background(path)
    det = CV.detect_vials(path, n_vials=n_cyl, background=bg)
    vials = det.get("vials") or []
    if len(vials) != n_cyl:
        # RETURN THE SAME SHAPE AS THE SUCCESS PATH. This used to return a bare
        # 4-tuple while the success path returns `(4-tuple, fits)`, and the caller
        # unpacks `(lay, bg, det, note), fits = build_layout(...)`. A miscounted
        # cylinder therefore raised `ValueError: too many values to unpack
        # (expected 2)` inside the caller's `except Exception`, so the run recorded
        # `failed: ValueError(...)` and the `if lay is None:` branch below it was
        # unreachable -- the informative reason ("located 2 cylinder(s), expected
        # 3") was discarded on every recording where the locator miscounted, which
        # is precisely the case the message exists for.
        return (None, bg, det,
                f"located {len(vials)} cylinder(s), expected {n_cyl}"), {}

    drawn, fits = [], {}
    for i, v in enumerate(vials, 1):
        padx = int(0.10 * v.width)
        dv = LAY.DrawnVial(
            x0=max(0, v.x0 - padx), x1=min(info.width, v.x1 + padx),
            y0=max(0, v.y_top - 60), y1=min(info.height, v.y_bottom + 20),
            label=f"R{i}", genotype=geno, treatment=treat, n_flies=n_flies)
        drawn.append(dv)
        fits[dv.label] = CS.find_line(
            bg, CS.Box(x0=dv.x0, y0=dv.y0, x1=dv.x1, y1=dv.y1,
                       label=dv.label, n_flies=max(n_flies, 1)))

    lay = LAY.Layout(vials=drawn, line_y=0, frame_w=info.width,
                     frame_h=info.height, reference_video=os.path.basename(path))
    LAY.set_vial_lines(lay, {k: int(f.row) for k, f in fits.items()})
    return (lay, bg, det,
            "; ".join(lay.problems()) if lay.problems() else ""), fits


def _tap(path: str) -> float:
    tp = CV.detect_taps(path)
    return float(tp["taps"][0]["t0_s"]) if tp.get("taps") else 0.0


def manifest_loading(row, n_cyl: int) -> "list[int] | None":
    """Flies per cylinder as the OPERATOR recorded them in the manifest.

    The column is written left-to-right, one entry per cylinder in frame: "20+20"
    for a two-cylinder recording, "20" for one. Returns None when it is absent or
    does not describe `n_cyl` cylinders, so the caller can refuse rather than guess.
    """
    raw = row.get("n_flies_per_cylinder")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    parts = [q.strip() for q in str(raw).replace(",", "+").split("+") if q.strip()]
    try:
        vals = [int(float(q)) for q in parts]
    except ValueError:
        return None
    if len(vals) == 1 and n_cyl > 1:
        vals = vals * n_cyl              # one figure quoted for every cylinder
    return vals if len(vals) == n_cyl and all(v > 0 for v in vals) else None


def main() -> int:
    man = pd.read_csv(B / "paraquat_video_manifest.csv")
    gt = pd.read_csv(B / "paraquat_ground_truth.csv")
    n_map = (gt[gt.time_s == 10].set_index(
        ["Genotype", "Treatment", "replicate"])["n_total"].to_dict())

    jobs = man[man["use"]].copy()
    print(f"Scoring {len(jobs)} recordings under mode={SCORING_MODE!r}, "
          f"backend={BACKEND!r}\n", flush=True)

    rows, t_start = [], time.time()
    for k, (_, r) in enumerate(jobs.iterrows(), 1):
        path = str(B / r["folder"] / r["video"])
        geno, treat = r["Genotype"], r["Treatment"]
        n_cyl = int(r["cylinders_in_frame"])
        reps = [1, 2] if n_cyl == 2 else [3]
        # THE DENOMINATOR IS NEVER GUESSED.
        # This used to be `n_map.get((geno, treat, reps[0]), 20)`: a missing
        # ground-truth key silently produced a cohort of 20, and every PI derived
        # from that recording then divided by a number nobody had checked. A wrong
        # denominator is the one error a perfect detector cannot survive -- it
        # yields a confidently wrong proportion -- so a miss now falls back to the
        # OPERATOR'S manifest entry, and if that is absent too the recording is not
        # scored at all. `n_flies_source` records which of the three happened.
        man_n = manifest_loading(r, n_cyl)
        base = {"folder": r["folder"], "video": r["video"], "tap": int(r["tap"]),
                "Genotype": geno, "Treatment": treat,
                "scoring_mode": SCORING_MODE, "backend": BACKEND, "step_s": STEP_S}

        def loading(rep_no: int, pos: int):
            """(flies, where it came from) for one cylinder -- never a guess."""
            if (geno, treat, rep_no) in n_map:
                return int(n_map[(geno, treat, rep_no)]), "ground truth"
            if man_n is not None and pos < len(man_n):
                return int(man_n[pos]), "manifest"
            return None, "unknown"

        n_flies, n_src = loading(reps[0], 0)
        if n_flies is None:
            print(f"          NOT SCORED: no fly count for {geno}/{treat} "
                  f"replicate {reps[0]} in either the ground truth or the "
                  f"manifest", flush=True)
            rows.append({**base, "status": "not scored: fly count unknown",
                         "n_flies_source": "unknown"})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            continue

        print(f"[{k:2d}/{len(jobs)}] {r['folder']:14s} {r['video'][:26]:26s} "
              f"tap {int(r['tap'])}  ({(time.time()-t_start)/60:.0f} min)", flush=True)
        try:
            (lay, bg, det, note), fits = build_layout(path, n_cyl, n_flies,
                                                      geno, treat)
        except Exception as exc:
            print(f"          FAILED: {exc!r}", flush=True)
            rows.append({**base, "status": f"failed: {exc!r}"})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            continue
        if lay is None:
            print(f"          NO LAYOUT: {note} -- not scored", flush=True)
            rows.append({**base, "status": f"no layout: {note}"})
            pd.DataFrame(rows).to_csv(OUT, index=False)
            continue

        t0 = _tap(path)
        for vi, dv in enumerate(lay.vials):
            rep = reps[vi] if vi < len(reps) else reps[-1]
            # per cylinder, and again never guessed -- the old form fell back to
            # this recording's FIRST replicate count, which is a different vial
            n_v, src_v = loading(rep, vi)
            if n_v is None:
                print(f"          {dv.label} (rep {rep}): NOT SCORED -- "
                      f"fly count unknown", flush=True)
                rows.append({**base, "replicate": rep, "vial": dv.label,
                             "status": "not scored: fly count unknown",
                             "n_flies_source": "unknown"})
                continue
            dv.n_flies = n_v
            fit = fits[dv.label]
            spec = dv.to_spec(lay.line_y)
            vbase = {**base, "replicate": rep, "vial": dv.label,
                     "n_flies": dv.n_flies, "n_flies_source": src_v,
                     "t0_s": round(t0, 2),
                     "line_row": int(fit.row), "line_depth": round(fit.depth, 1),
                     "line_margin": round(fit.margin, 3),
                     "line_is_clear": bool(fit.is_clear())}
            probe = CD.probe_footage(path, spec, background=bg, n_samples=6)
            rec = CD.recommend_backend(probe, n_expected=dv.n_flies)
            if rec["backend"] is None:
                print(f"          {dv.label} (rep {rep}): REFUSED -- "
                      f"{rec['reason'][:70]}", flush=True)
                rows.append({**vbase, "status": "refused by the probe",
                             "refusal": rec["reason"]})
                continue
            for s in CD.score_crossings(path, spec, bg, t0, TIMES, backend=BACKEND,
                                        probe=probe, mode=SCORING_MODE,
                                        step_s=STEP_S):
                h = gt[(gt.Genotype == geno) & (gt.Treatment == treat)
                       & (gt.replicate == rep) & (gt.tap == int(r["tap"]))
                       & (gt.time_s == s["time_s"])]
                rows.append({**vbase, "time_s": s["time_s"],
                             "video_count": s["n_above"],
                             "hand": int(h["n_climbed"].iloc[0]) if len(h) else np.nan,
                             "status": "scored"})
        pd.DataFrame(rows).to_csv(OUT, index=False)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    SIDECAR.write_text(json.dumps({
        "produced_by": "scripts/rescore_paraquat_net.py",
        "scoring_mode": SCORING_MODE, "backend": BACKEND, "step_s": STEP_S,
        "timepoints_s": TIMES, "recordings": int(len(jobs)),
        "rule": ("net = up-minus-down = the number STANDING above the line; "
                 "the published protocol. NOT the ratchet (max_reached)."),
        "versions": PROV.versions()}, indent=2), encoding="utf-8")

    ok = d[d["status"] == "scored"].dropna(subset=["hand", "video_count"])
    print("\n" + "=" * 74)
    print(f"{len(ok)} paired cells; {int((d['status'] != 'scored').sum())} "
          f"rows refused or unscoreable")
    print("=" * 74)
    if len(ok):
        e = ok["video_count"] - ok["hand"]
        print(f"  bias {e.mean():+.2f}   MAE {e.abs().mean():.2f}   "
              f"within 3 flies {(e.abs() <= 3).mean():.1%}")
        clear = ok[ok["line_is_clear"]]
        if len(clear):
            ec = clear["video_count"] - clear["hand"]
            print(f"  vials the software was SURE of the line on ({len(clear)} cells): "
                  f"bias {ec.mean():+.2f}  MAE {ec.abs().mean():.2f}  "
                  f"within 3 {(ec.abs() <= 3).mean():.1%}")
        print("\n  by condition (MAE):")
        g = ok.assign(err=(ok.video_count - ok.hand),
                      abserr=(ok.video_count - ok.hand).abs()).groupby(
            ["Genotype", "Treatment"]).agg(cells=("abserr", "size"),
                                           MAE=("abserr", "mean"),
                                           bias=("err", "mean"))
        print(g.round(2).to_string())
        print("\n  mean count by time (video vs hand):")
        print(ok.groupby("time_s")[["video_count", "hand"]].mean()
              .round(1).to_string())
    print(f"\nwrote {OUT}\nwrote {SIDECAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
