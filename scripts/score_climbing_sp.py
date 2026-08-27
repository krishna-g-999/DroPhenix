"""Score every recording in `Climbing assay_SP`, and say plainly what is measured.

WHAT THIS SET IS
----------------
An ageing time course: Day 3, Day 7 and Day 14, three graduated cylinders per frame,
each carrying a band drawn round it in marker that serves as the scoring line.

WHAT CAN AND CANNOT BE CLAIMED HERE
-----------------------------------
There are NO hand counts for this set, so ACCURACY CANNOT BE MEASURED on it. Nothing
below is an accuracy claim. What can be established without any ground truth is:

  * whether the tool finds the geometry at all, and refuses where it cannot;
  * whether the fly count it reports for a SEALED cylinder holds still over two
    minutes -- it must, because the number of flies in a sealed tube cannot change,
    so every wobble is detector error and needs no ground truth to interpret;
  * the climbing curves themselves, and how they differ between days.

THE DENOMINATOR IS THE OPERATOR'S, NOT THE DETECTOR'S: ten flies per vial, as
loaded by the laboratory and confirmed by it. Every proportion here divides by that
(`N_FLIES` below, and `n_flies` in the output). Flies the detector cannot account
for at an instant could be on either side of the line, so `pct_above_lo/hi` bracket
the proportion rather than hiding the gap.

An earlier version of this script ESTIMATED the denominator as the median detected
total across the clip, and this paragraph used to describe that. It was wrong twice
over: it made the denominator depend on the detector's own error, and it absorbed
the drawn band's false positives into "flies loaded". Told the true value, the same
detections read completely differently -- recall looked like 32% against an assumed
20 and is 87% against the true 10.

That estimate survives in the output as `n_est_loaded` / `n_est_iqr`. IT IS NOT THE
DENOMINATOR AND IS NOT USED AS ONE. It is a detector-coverage statistic -- how many
objects were typically found in the tube -- kept because it is the quality signal
for everything else. Read as a loading estimate it is nonsense: it exceeds the true
10 on some vials and, being a median over frames that include the clustered period
just after the tap, falls below the concurrent above-the-line count on others.

Genotype and treatment are written on cards in the frame in handwriting
("HTT + TLA Male", "HTT Control (Male)", "W/H type + TLA"). Those are not read here.
Vials are identified by DAY, VIDEO and POSITION (P1..P3, left to right). Note that
the tubes are also marked R1/R2/R3 in pen and their physical order CHANGES between
recordings, so position is not identity -- which is why position is what is reported.

GEOMETRY
--------
Two independent locators are tried and the better is used:

  glass  `climbing_video.detect_vials`  -- works on Day 7 and Day 14, fails on Day 3
  band   `cylinder_finder.locate_cylinders` -- uses the operator's drawn band

A second pass then uses the fact that the rig does not move within a day: the day's
consensus band height and tube spacing resolve recordings whose own fit is poor.
Anything still inconsistent is REFUSED rather than scored.

Run: ./.venv/Scripts/python.exe scripts/score_climbing_sp.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import cv2  # noqa: E402

from modules import climbing_video as CV  # noqa: E402
from modules import climbing_simple as CS  # noqa: E402
from modules import cylinder_finder as CF  # noqa: E402
from modules import fly_blob as FB  # noqa: E402
from modules import provenance as PROV  # noqa: E402

D = Path(r"D:/DroPhenix/Testing data/Climbing assay/Climbing assay_SP")
#: WHERE THE OUTPUTS GO. This was hardcoded to a Claude session's scratchpad --
#: `.../Temp/claude/D--DroPhenix/ab90d08c-.../scratchpad` -- which is temporary by
#: construction. It still existed when this was found, but the day it is cleaned up
#: the script dies at `mkdir` and `report_climbing_sp.py`, which pointed at the same
#: literal, dies on its first read. Worse, the shipped results in `results/` had no
#: traceable relationship to a path that was about to disappear.
#: Now: the repository's own results directory, overridable for a scratch run.
OUT = Path(os.environ.get(
    "DROPHENIX_SP_OUT",
    Path(__file__).resolve().parents[1] / "results" / "climbing_video"))

#: The temporal-median backgrounds are a CACHE, not a result -- one float array per
#: recording, 68 of them on this set. They are kept out of the repository on
#: purpose; delete the directory and the next run rebuilds them.
BG_CACHE = Path(os.environ.get(
    "DROPHENIX_SP_BGCACHE", Path(tempfile.gettempdir()) / "drophenix_sp_bg"))
DAYS = ("Day 3", "Day7", "Day 14")
TIMES = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.]
GRID_S = 2.0          # detection grid; `max_reached` needs it finer than the report
MIN_DURATION_S = 30.0

#: FLIES PER VIAL. This is an operator input, not something to be inferred, and the
#: whole analysis rests on it: it is the denominator of every proportion the assay
#: reports. Estimating it from the detector instead -- which an earlier version of
#: this script did -- was wrong twice over. It makes the denominator depend on the
#: detector's own error, and it silently absorbed the drawn band's false positives
#: into "flies loaded". Told the true value, the same detections read completely
#: differently: measured against an assumed 20 the recall looked like 32%; against
#: the true 10 it is 87%.
N_FLIES = 10


def band_mask(bg_sub, line_local, frac=0.35):
    """The rows the DRAWN BAND occupies, to be excluded from counting.

    This is the single largest source of false positives on this rig, and it is not
    a tuning failure: the band here is a thick marker ring with mottled texture, and
    a size-matched detector answers "yes, that is a fly-sized dark blob" to each
    mottle. Rendered on a real frame, the detections sat in a ROW along the band --
    which also pinned them at the scoring line, the worst possible place. Masking
    the band removed 12 of 12 such detections on one Day-14 tube and took another
    from 3 detections to 0.

    A GENERIC static-mark mask was tried first and is wrong here. On the paraquat
    rig it destroyed the count (MAE 3.28 -> 8.95), because there the flies settle and
    end up IN the temporal median, so masking "marks" masks flies. Masking only the
    band's own rows, whose position is already known from the band locator, removes
    the artefact without touching anything else.

    Cost, stated: a fly resting ON the band is lost. That is bounded -- the band is
    7 to 29 rows of an 800-row tube -- and it is a known loss rather than an unknown
    error.
    """
    prof = bg_sub.mean(axis=1).astype(np.float32)
    dark = np.clip(float(np.percentile(prof, 75)) - prof, 0, None)
    if not (0 <= line_local < len(dark)):
        return None
    peak = float(dark[max(0, line_local - 3):line_local + 4].max())
    if peak <= 1:
        return None
    thr = frac * peak
    a = b = int(line_local)
    while a > 0 and dark[a - 1] > thr:
        a -= 1
    while b < len(dark) - 1 and dark[b + 1] > thr:
        b += 1
    m = np.zeros(bg_sub.shape, bool)
    m[a:b + 1, :] = True
    return m, a, b


def tube_box(bg, frames, v, frame_h):
    """The box for one tube: the band gives x, and y is the whole frame column.

    An earlier version took the vertical extent from where flies were DETECTED, and
    that was circular: a fly missed near the top or bottom shrank the box, and the
    smaller box then missed more. Measured against the true loading of 10 flies per
    tube, four definitions were compared on 160 cells:

        full frame column        recall 86.8%   bias -1.33   MAE 3.68   <- used
        band -3.5w .. +5.0w      recall 75.3%   bias -2.48   MAE 4.54
        fly-derived (circular)   recall 72.2%   bias -2.78   MAE 4.80
        band -3.0w .. +4.5w      recall 63.8%   bias -3.62   MAE 5.12

    The full column wins because the band's x-range already isolates the tube, so
    there is little to gain by cropping vertically and a fly to lose every time the
    crop is slightly wrong. `frames` is no longer needed and is accepted only so the
    caller does not change.
    """
    return (0, int(frame_h) - 1), 0


def main() -> int:
    files = [(day, f) for day in DAYS
             for f in sorted(glob.glob(str(D / day / "*.MP4")))]
    print(f"{len(files)} recordings\n", flush=True)

    # ---- pass 1: geometry ---------------------------------------------------
    geo, bgs, t0 = [], {}, time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    # `parents=True` matters: the old form assumed OUT already existed, so a fresh
    # checkout failed here with FileNotFoundError rather than creating its output
    # directory.
    _bgdir = BG_CACHE
    _bgdir.mkdir(parents=True, exist_ok=True)
    for k, (day, f) in enumerate(files, 1):
        nm = os.path.basename(f)
        print(f"[geom {k:2d}/{len(files)}] {day:7s} {nm:11s} "
              f"({(time.time()-t0)/60:5.1f} min)", flush=True)
        rec = {"day": day, "video": nm, "path": f}
        try:
            info = CV.probe_video(f)
        except Exception as exc:
            geo.append({**rec, "status": f"probe failed: {exc}"[:70]}); continue
        rec.update({"fps": round(info.fps, 2), "dur_s": round(info.duration_s, 1),
                    "frame_w": info.width, "frame_h": info.height})
        if info.duration_s < MIN_DURATION_S:
            geo.append({**rec, "status": f"only {info.duration_s:.1f} s long -- not "
                                         f"an assay recording"})
            print("          EXCLUDED: too short", flush=True)
            continue
        _cached = _bgdir / (day.replace(" ", "_") + "__" + nm + ".npy")
        try:
            bg = (np.load(str(_cached)) if _cached.exists()
                  else CV.build_background(f))
        except Exception as exc:
            geo.append({**rec, "status": f"background failed: {exc}"[:70]}); continue
        bgs[f] = bg
        np.save(str(_bgdir / (day.replace(" ", "_") + "__" + nm + ".npy")), bg)
        band = CF.locate_cylinders(bg, 3)
        try:
            gl = CV.detect_vials(f, n_vials=3, background=bg)
            gv = gl.get("vials") or []
            gconf = float(gl.get("confidence", float("nan")))
        except Exception:
            gv, gconf = [], float("nan")
        rec.update({"glass_n": len(gv), "glass_conf": round(gconf, 2),
                    "glass_x0": [v.x0 for v in gv], "glass_x1": [v.x1 for v in gv],
                    "glass_top": [v.y_top for v in gv],
                    "glass_bot": [v.y_bottom for v in gv]})
        geo.append({**rec, "status": "geometry",
                    "fit": round(band.fit, 3), "n_cands": band.n_candidates,
                    "x0": [v.x0 for v in band.vials],
                    "x1": [v.x1 for v in band.vials],
                    "rows": [v.band_row for v in band.vials],
                    "self_ok": bool(band.ok())})
    g = pd.DataFrame(geo)
    g.to_csv(OUT / "sp_geometry.csv", index=False)

    # ---- pass 2: the rig does not move within a day -------------------------
    # A recording whose own fit is poor is not necessarily unscoreable: the same
    # three tubes stand in the same places all session. The day's consensus band
    # height and tube centres are taken from the recordings that DID fit well, and
    # the others are re-selected against it. Only what remains inconsistent is
    # refused.
    consensus = {}
    ok_geo = g[(g.status == "geometry") & (g.self_ok)]
    for day in DAYS:
        d_ok = ok_geo[ok_geo.day == day]
        if len(d_ok) < 3:
            continue
        rows = np.concatenate([np.asarray(r) for r in d_ok["rows"]])
        cx = np.array([[0.5 * (a + b) for a, b in zip(x0, x1)]
                       for x0, x1 in zip(d_ok["x0"], d_ok["x1"])], float)
        wid = np.concatenate([np.asarray(x1) - np.asarray(x0)
                              for x0, x1 in zip(d_ok["x0"], d_ok["x1"])])
        consensus[day] = {"row": float(np.median(rows)),
                          "cx": np.median(cx, axis=0),
                          "w": float(np.median(wid)), "n": int(len(d_ok))}
        print(f"\n{day}: consensus from {len(d_ok)} well-fitted recordings -- "
              f"band row {consensus[day]['row']:.0f}, tube centres "
              f"{np.round(consensus[day]['cx']).astype(int).tolist()}, "
              f"width {consensus[day]['w']:.0f}", flush=True)

    # ---- pass 3: score ------------------------------------------------------
    rows_out, t1 = [], time.time()
    todo = g[g.status == "geometry"]
    for k, (_, r) in enumerate(todo.iterrows(), 1):
        f, day, nm = r["path"], r["day"], r["video"]
        print(f"[score {k:2d}/{len(todo)}] {day:7s} {nm:11s} "
              f"({(time.time()-t1)/60:5.1f} min)", flush=True)
        bg = bgs.get(f)
        if bg is None:
            continue
        info = CV.probe_video(f)
        con = consensus.get(day)
        used = "own fit"
        cands = CF.find_band_candidates(bg)
        sel = CF.locate_cylinders(bg, 3, candidates=cands)
        if not sel.ok() and con is not None:
            # re-pick against the day's consensus rather than refusing outright
            from itertools import combinations
            best, cost = None, float("inf")
            for combo in combinations(sorted(cands, key=lambda v: -v.darkness)[:16], 3):
                s = sorted(combo, key=lambda v: v.x0)
                c = (abs(np.median([v.band_row for v in s]) - con["row"]) / con["w"]
                     + float(np.mean(np.abs(np.array([v.cx for v in s])
                                            - con["cx"]))) / con["w"]
                     + float(np.std([v.width for v in s]) / con["w"]))
                if c < cost:
                    best, cost = s, c
            if best is not None and cost <= 1.0:
                sel = CF.LocateResult(best, float(cost), len(cands), "")
                used = f"day consensus (cost {cost:.2f})"
        if not sel.vials or (not sel.ok() and used == "own fit"):
            # The band locator is not the only one, and on the Day-7 rig it is the
            # WRONG one: those bands are faint, while `detect_vials` finds the glass
            # there at confidence 0.86-0.96. Falling back to it recovers a whole
            # day's recordings that would otherwise have been refused wholesale --
            # which is the point of having two independent locators rather than one.
            gx0 = r.get("glass_x0") or []
            if len(gx0) == 3:
                gv = []
                for gi in range(3):
                    x0g, x1g = int(r["glass_x0"][gi]), int(r["glass_x1"][gi])
                    ytg, ybg = int(r["glass_top"][gi]), int(r["glass_bot"][gi])
                    pad = int(0.10 * (x1g - x0g))
                    bx0 = max(0, x0g - pad)
                    bx1 = min(info.width, x1g + pad)
                    box = CS.Box(x0=bx0, y0=max(0, ytg - 40),
                                 x1=bx1, y1=min(info.height, ybg + 20),
                                 label=f"P{gi+1}", n_flies=1)
                    fit_l = CS.find_line(bg, box)
                    gv.append(CF.BandVial(x0=bx0, x1=bx1, band_row=int(fit_l.row),
                                          n_rows=0, darkness=float(fit_l.depth)))
                gv.sort(key=lambda v: v.x0)
                sel = CF.LocateResult(gv, float("nan"), 0, "")
                used = f"glass (conf {r.get('glass_conf')})"
            else:
                rows_out.append({"day": day, "video": nm,
                                 "status": "refused: geometry",
                                 "note": sel.reason[:90]})
                print(f"          REFUSED: {sel.reason[:70]}", flush=True)
                continue

        # The box no longer needs a survey pass: its vertical extent is the whole
        # frame column (see `tube_box`), so the 14 extra frames that pass used to
        # read are no longer decoded.
        ext_frames = []
        tp = CV.detect_taps(f)
        taps = tp.get("taps") or []
        t_zero = float(taps[0]["t0_s"]) if taps else 0.0

        grid = np.arange(0.0, min(max(TIMES) + GRID_S,
                                  max(info.duration_s - t_zero, 0.0)), GRID_S)
        gidx = [int((t_zero + t) * info.fps) for t in grid]
        gframes = CV._read_frames(f, gidx)
        if len(gframes) < len(grid) * 0.6:
            rows_out.append({"day": day, "video": nm,
                             "status": "refused: too few frames after the tap"})
            continue
        grid = grid[:len(gframes)]

        # A `find_line` row-agreement gate was tried here and REVERTED. It does fix
        # two recordings whose band locator read the printed label instead of the
        # drawn band -- but applied across the set it refused 15 recordings instead
        # of 4, took Day 7 from 21 usable to 10, and the survivors then reported
        # 91-99% of flies above the line against 70% before, with the detected total
        # running 20% OVER the known 10. That is selection on the outcome, not a
        # better measurement, and a biased survivor set is worse than a few refusals.
        # The two mis-read recordings are better handled by drawing their boxes.
        for pi, v in enumerate(sel.vials, 1):
            ext, n_ext = tube_box(bg, ext_frames, v, info.height)
            if ext is None:
                rows_out.append({"day": day, "video": nm, "position": f"P{pi}",
                                 "status": "refused: no flies detected in the tube"})
                continue
            y0, y1 = ext
            sub_bg = bg[y0:y1, v.x0:v.x1]
            line_local = int(v.band_row - y0)
            bm = band_mask(sub_bg, line_local)
            mask, band_a, band_b = (bm if bm is not None else (None, -1, -1))
            above, totals = [], []
            for fr in gframes:
                pts = FB.detect(fr[y0:y1, v.x0:v.x1], sub_bg, marks=mask)
                totals.append(len(pts))
                above.append(sum(1 for (_, cy, _) in pts if cy < line_local))
            above = np.array(above, float)
            totals = np.array(totals, float)
            # the denominator is NOT recorded for this set; estimate it and carry
            # the spread of that estimate with every proportion derived from it
            n_est = float(np.median(totals))
            n_iqr = float(np.percentile(totals, 75) - np.percentile(totals, 25))
            n_true = float(N_FLIES)
            cvpct = float(totals.std() / max(totals.mean(), 1e-9) * 100)
            run_max = np.maximum.accumulate(above)
            for t in TIMES:
                j = int(np.argmin(np.abs(grid - t)))
                if abs(grid[j] - t) > GRID_S:
                    continue
                rows_out.append({
                    "day": day, "video": nm, "position": f"P{pi}",
                    "vial_id": f"{day}|{nm}|P{pi}", "time_s": t,
                    "n_above": float(above[j]),
                    "n_above_max_reached": float(run_max[j]),
                    "n_detected_total": float(totals[j]),
                    "n_est_loaded": n_est, "n_est_iqr": n_iqr,
                    "n_flies": n_true,
                    # the assay's readout: what fraction of the KNOWN cohort is above
                    "pct_above": round(100 * above[j] / n_true, 1),
                    # and the uncertainty on it: flies the detector cannot account
                    # for at this instant could be on either side of the line, so
                    # this is the numerator's error bar, not a nuisance statistic
                    "n_unaccounted": max(0.0, n_true - float(totals[j])),
                    "pct_above_lo": round(100 * above[j] / n_true, 1),
                    "pct_above_hi": round(100 * min(n_true, above[j]
                                                    + max(0.0, n_true - totals[j]))
                                          / n_true, 1),
                    "pct_above_est": (round(100 * above[j] / n_est, 1)
                                      if n_est > 0 else np.nan),
                    "total_cv_pct": round(cvpct, 1),
                    "line_row": int(v.band_row), "box_y0": y0, "box_y1": y1,
                    "band_rows_masked": int(band_b - band_a + 1) if band_a >= 0 else 0,
                    "tube_x0": v.x0, "tube_x1": v.x1, "tube_w": v.width,
                    "band_darkness": round(v.darkness, 1),
                    "t_zero_s": round(t_zero, 2), "n_taps": len(taps),
                    "framing_stable": bool(tp.get("framing_stable", True)),
                    "geometry": used, "geom_fit": round(sel.fit, 3),
                    "fps": info.fps, "backend": "blob", "scoring_grid_s": GRID_S,
                    "status": "scored"})
        pd.DataFrame(rows_out).to_csv(OUT / "sp_results.csv", index=False)

    res = pd.DataFrame(rows_out)
    res.to_csv(OUT / "sp_results.csv", index=False)
    (OUT / "sp_results.provenance.json").write_text(json.dumps({
        "produced_by": "scripts/score_climbing_sp.py",
        "backend": "blob", "scoring_grid_s": GRID_S, "timepoints_s": TIMES,
        "labels": "day / video / position (P1..P3 left to right)",
        "n_flies_per_vial": N_FLIES,
        "denominator": ("the OPERATOR-SUPPLIED loading of 10 flies per vial, "
                        "confirmed by the laboratory. Every proportion uses it. "
                        "pct_above_lo/hi bracket the proportion by assuming the "
                        "flies the detector cannot account for are all below, or "
                        "all above, the line."),
        "accuracy": ("NOT measurable on this set -- there are no hand counts. "
                     "Detector accuracy was measured elsewhere: see "
                     "modules/fly_blob.py"),
        "versions": PROV.versions()}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'sp_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
