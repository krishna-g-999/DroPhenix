"""
Measure the tripwire (line-crossing) counter against the published bar.

WHAT THIS DECIDES
-----------------
`modules/fly_tripwire.py` counts flies by LINE CROSSINGS over time instead of
by blobs in one frame, on the argument that the counting ceiling is a property
of simultaneous spatial occupancy and not of event counting. That argument is
not evidence. This script is the measurement.

The bar is the one already published in the register for any replacement of
`fly_blob`, and it is not negotiable after the fact:

  * beat MAE 3.589 and RMSE 4.436 against the hand counts;
  * recover more than 69% of the between-arm spread;
  * and preserve the ordering of all six arms.

The third is the one that killed the accepted-peak-mass candidate, which beat
the ceiling on aggregate and then reported WT 10 mM above TDP Untreated when
the truth is the reverse. An instrument whose output is a comparison between
arms cannot invert one.

HOW IT DIFFERS FROM `three_way_detector_comparison.py`
-------------------------------------------------------
That script decodes TWELVE frames per recording, one at each scored timepoint,
because a spatial counter only needs the frame it scores. A crossing counter
needs the frames BETWEEN the timepoints -- that is the whole idea -- so this
decodes continuously from the tap to the last timepoint at `--fps` and reads
every timepoint off one pass of the resulting crossing series.

Everything else is deliberately identical: the same manifest, the same tap
detection, the same cylinder layout and scoring lines, the same hand counts and
the same join. Only the counting rule changes, so a difference in the result is
a difference between counting rules.

USAGE
    python scripts/tripwire_detector_study.py --limit 3     # pilot
    python scripts/tripwire_detector_study.py               # full run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
from modules import fly_blob as FB  # noqa: E402
from modules import fly_tripwire as TW  # noqa: E402
import lab_data as _LAB  # noqa: E402
from rescore_paraquat_net import build_layout, _tap  # noqa: E402

PARA = ((_LAB.climbing_root() or Path("_absent"))
        / "Paraquat climbing assay 010626_new set")
MANIFEST = PARA / "paraquat_video_manifest.csv"
GROUND_TRUTH = PARA / "paraquat_ground_truth.csv"

#: The same timepoints the published rescore and the three-way comparison score.
TIMES = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.]

#: The published bar, from register N103.
BAR = {"mae": 3.589, "rmse": 4.436, "spread_pct": 69.0}


def replicates_for(row, n_cyl: int) -> List[int]:
    """Which replicate sits in each cylinder, left to right (manifest columns)."""
    li, ri = row.get("left_is"), row.get("right_is")
    out: List[int] = []
    for v in (li, ri):
        if pd.notna(v):
            out.append(int(float(v)))
    if len(out) < n_cyl:
        out = [3] if n_cyl == 1 else [1, 2][:n_cyl]
    return out[:n_cyl]


def centroids_for_specs(gray: np.ndarray, bg: np.ndarray, specs
                        ) -> List[np.ndarray]:
    """Fly centroids inside each cylinder, in FULL-FRAME pixel coordinates."""
    out: List[np.ndarray] = []
    for s in specs:
        y0, y1 = int(s.y_top), int(s.y_bottom)
        x0, x1 = int(s.x0), int(s.x1)
        sub, bsub = gray[y0:y1, x0:x1], bg[y0:y1, x0:x1]
        if sub.size == 0 or bsub.shape != sub.shape:
            out.append(np.empty((0, 2), dtype=float))
            continue
        try:
            pts = FB.detect(sub.astype(np.uint8), bsub.astype(np.uint8), None)
        except Exception:
            pts = []
        out.append(np.array([[float(px) + x0, float(py) + y0]
                             for (px, py, *_rest) in pts], dtype=float)
                   .reshape(-1, 2))
    return out


def run(limit: Optional[int], fps_sample: float, out_csv: Path) -> int:
    if not MANIFEST.exists():
        print(f"[SKIP] manifest not found at {MANIFEST}")
        print(_LAB.describe_roots())
        return 0

    man = pd.read_csv(MANIFEST)
    man = man[man.get("use", True).astype(str).str.lower().isin(["true", "1"])]
    gt = pd.read_csv(GROUND_TRUTH)

    if limit:
        man = man.head(limit)

    rows: List[dict] = []
    t_start = time.time()
    for mi, (_, row) in enumerate(man.iterrows(), 1):
        folder, video = str(row["folder"]), str(row["video"])
        path = PARA / folder / video
        if not path.exists():
            print(f"  [{mi}/{len(man)}] {video}: not on disk")
            continue
        n_cyl = int(row["cylinders_in_frame"])
        geno, treat = str(row["Genotype"]), str(row["Treatment"])
        try:
            n_flies = int(str(row["n_flies_per_cylinder"]).split("+")[0])
        except Exception:
            n_flies = 20

        try:
            (lay, bg, _det, _note), _fits = build_layout(
                str(path), n_cyl, n_flies, geno, treat)
        except Exception as exc:
            print(f"  [{mi}/{len(man)}] {video}: layout failed ({exc})")
            continue
        if lay is None:
            print(f"  [{mi}/{len(man)}] {video}: {_note}")
            continue

        specs = lay.specs()
        bg_gray = bg if bg.ndim == 2 else cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        t0 = _tap(str(path))

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps / fps_sample)), 1)
        last_t = max(TIMES)
        start_f = int(t0 * fps)
        end_f = int((t0 + last_t) * fps)

        per_cyl: List[List[np.ndarray]] = [[] for _ in specs]
        times_s: List[float] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        fidx = start_f
        n_read = 0
        while fidx <= end_f:
            ok, im = cap.read()
            if not ok:
                break
            if (fidx - start_f) % step == 0:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                cs = centroids_for_specs(gray, bg_gray, specs)
                for k in range(len(specs)):
                    per_cyl[k].append(cs[k])
                times_s.append((fidx - start_f) / fps)
                n_read += 1
            fidx += 1
        cap.release()

        reps = replicates_for(row, n_cyl)
        tap_no = int(row.get("tap", 1) or 1)
        for k, s in enumerate(specs):
            rep = reps[k] if k < len(reps) else (k + 1)
            sub_gt = gt[(gt["Genotype"] == geno) & (gt["Treatment"] == treat)
                        & (gt["replicate"] == rep) & (gt["tap"] == tap_no)]
            n_loaded = (int(sub_gt["n_total"].iloc[0])
                        if len(sub_gt) else n_flies)
            # The seed: a SPATIAL count of the flies above the line at the
            # first sampled frame. Crossing counting gives the change, not the
            # absolute, and on this footage the tap does not put every fly
            # below the line -- measured, 8 of 20 were already above it at the
            # detected tap and never crossed. See the note in
            # `fly_tripwire.count_crossings`. The seed is taken at the frame
            # where the region ABOVE the line is emptiest, which is where the
            # spatial counter's crowding ceiling bites least.
            above_series = [int(np.sum(f[:, 1] < float(s.line_y)))
                            if len(f) else 0 for f in per_cyl[k]]
            seed_idx = int(np.argmin(above_series)) if above_series else 0
            seed = int(above_series[seed_idx]) if above_series else 0
            res = TW.count_crossings(
                per_cyl[k][seed_idx:], float(s.line_y), FB.fly_diameter(
                    int(s.x1) - int(s.x0), dict(FB.PARAMS)),
                TW.TripwireParams(fps_sample=fps_sample),
                n_loaded=n_loaded, times_s=times_s[seed_idx:],
                seed_above=seed)
            at = TW.counts_at_times(res, TIMES)
            for t in TIMES:
                g = sub_gt[sub_gt["time_s"] == t]
                rows.append({
                    "video": video, "folder": folder, "Genotype": geno,
                    "Treatment": treat, "replicate": rep, "tap": tap_no,
                    "cylinder": s.label, "time_s": t,
                    "tripwire": at.get(t, np.nan),
                    "hand": int(g["n_climbed"].iloc[0]) if len(g) else np.nan,
                    "n_loaded": n_loaded,
                    "usable": res.usable, "reason": res.reason,
                    "up": res.up, "down": res.down, "n_frames": res.n_frames,
                    "seed": seed, "seed_t": times_s[seed_idx] if times_s else 0.0,
                })
        el = time.time() - t_start
        print(f"  [{mi}/{len(man)}] {video}: {n_read} frames, "
              f"{len(specs)} cylinder(s), {el:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows)")
    report(df)
    return 0


def report(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["hand", "tripwire"])
    if d.empty:
        print("no joined rows -- nothing to report")
        return
    err = d["tripwire"].to_numpy(float) - d["hand"].to_numpy(float)
    hand = d["hand"].to_numpy(float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r = float(np.corrcoef(err, hand)[0, 1]) if len(d) > 2 else np.nan
    print("\n" + "=" * 74)
    print(f"cells joined: {len(d)}   unusable: {int((~d['usable']).sum())}")
    print(f"  MAE  {mae:6.3f}   (bar: beat {BAR['mae']})")
    print(f"  RMSE {rmse:6.3f}   (bar: beat {BAR['rmse']})")
    print(f"  bias {np.mean(err):+6.3f}   r(err, hand) {r:+.3f}")

    arms = d.groupby(["Genotype", "Treatment"]).agg(
        hand=("hand", "mean"), tool=("tripwire", "mean")).reset_index()
    arms = arms.sort_values("hand")
    sh = arms["hand"].max() - arms["hand"].min()
    st = arms["tool"].max() - arms["tool"].min()
    print(f"\n  between-arm spread: {st:.2f} = {100*st/sh:.0f}% of hand "
          f"({sh:.2f})   (bar: beat {BAR['spread_pct']}%)")
    print(f"  {'arm':24s} {'hand':>8s} {'tripwire':>10s}")
    for _, a in arms.iterrows():
        print(f"  {a.Genotype + ' ' + a.Treatment:24s} "
              f"{a.hand:8.2f} {a.tool:10.2f}")
    order_ok = list(arms.sort_values("hand")["Genotype"] + " " +
                    arms.sort_values("hand")["Treatment"]) == \
        list(arms.sort_values("tool")["Genotype"] + " " +
             arms.sort_values("tool")["Treatment"])
    print(f"\n  arm ordering preserved: {order_ok}   (bar: MUST be True)")
    passed = (mae < BAR["mae"] and rmse < BAR["rmse"]
              and 100 * st / sh > BAR["spread_pct"] and order_ok)
    print("\n  VERDICT: " + ("CLEARS THE BAR" if passed
                             else "DOES NOT CLEAR THE BAR"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default=str(PARA / "tripwire_detector_study.csv"))
    a = ap.parse_args(argv)
    return run(a.limit, a.fps, Path(a.out))


if __name__ == "__main__":
    raise SystemExit(main())
