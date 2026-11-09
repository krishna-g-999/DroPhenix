"""
study_three_way_detectors.py
Three climbing detectors against one set of hand counts.

Reads `three_way_detector_comparison.csv`, produced by
`scripts/three_way_detector_comparison.py`, which runs all three detectors on
ONE decode pass per recording with the same layout, the same scoring line, the
same background and the same footage-quality gate. Reporting is separated from
measuring so that the analysis can be redone without decoding 62 videos again.

THE QUESTIONS, IN THE ORDER THEY HAVE TO BE ANSWERED
----------------------------------------------------
0. IS THE HARNESS FAITHFUL? The `fly_blob` column must reproduce the published
   run, `paraquat_scored_net_blob_2026-11.csv`, cell by cell. The harness reads
   the frame at each timepoint directly rather than walking the 2-second
   scoring grid, which is equivalent in `mode="net"` -- but equivalence is a
   claim, and if it fails nothing below means anything. This is reported first
   and loudly.

0b. WHAT DID THE HAND SCORER COUNT? Every accuracy number in this project
   compares a detector's INSTANTANEOUS count -- the number standing above the
   line, which is what the protocol's wording implies and what
   `climbing_detectors` decided on 2026-08-17 -- against the laboratory's hand
   counts. If those hand counts are instead CUMULATIVE crossings, the two are
   different quantities and the difference is charged to the detector. The
   test is monotonicity: a cumulative count can never fall, an instantaneous
   one must fall whenever a fly does, and in the impaired arms flies
   demonstrably fall. This is reported before any accuracy figure because it
   decides what the figures mean.

1. DOES MERGE SPLITTING FIX THE CEILING? Register N97 measured the `fly_blob`
   path under-counting with a ceiling near 12-13 flies per vial, worst in the
   most crowded arm. Register N3's fix -- count a blob larger than one fly as
   the number of flies its area implies -- lives in `fly_detector`, the other
   stack, and has never been scored against hand counts. The ceiling is visible
   only when stratified, so the stratified table is the answer and the pooled
   number is not.

2. HOW DOES EITHER COMPARE WITH THE FIELD'S TOOL? FreeClimber
   (Spierer et al. 2021) is an independent, peer-reviewed detector using
   Crocker-Grier centroid finding. Its reimplementation is validated against
   its own published output by `validation/validate_freeclimber_detection.py`.

3. WHERE DO THEY DISAGREE, AND WITH WHAT CONSEQUENCE? Two detectors can share
   a mean absolute error and differ completely per cell. Agreement is therefore
   reported per cell and the bias is reported against the true count, because a
   bias that grows with the count compresses a dose-response while a constant
   one does not.

Run: ./.venv/Scripts/python.exe validation/study_three_way_detectors.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from validation import lab_data as _LAB
except Exception:                                      # direct-script use
    import lab_data as _LAB

PARA = ((_LAB.climbing_root() or Path("_absent"))
        / "Paraquat climbing assay 010626_new set")
COMPARISON = PARA / "three_way_detector_comparison.csv"
PUBLISHED = PARA / "paraquat_scored_net_blob_2026-11.csv"

DETECTORS = ["fly_blob", "fly_detector", "freeclimber"]
KEYS = ["folder", "video", "tap", "replicate", "vial", "time_s"]


def summarise(err: np.ndarray) -> Dict[str, float]:
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, "MAE": np.nan, "bias": np.nan, "RMSE": np.nan,
                "within2": np.nan}
    return {"n": int(e.size), "MAE": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "within2": 100.0 * float(np.mean(np.abs(e) <= 2))}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="three-way detector analysis")
    ap.add_argument("--comparison", type=str, default=str(COMPARISON),
                    help="the comparison CSV to analyse")
    args = ap.parse_args(argv)
    comparison = Path(args.comparison)
    if not comparison.is_file():
        print(f"{COMPARISON.name} is not present. Run\n"
              f"  ./.venv/Scripts/python.exe scripts/"
              f"three_way_detector_comparison.py\nfirst.")
        print(_LAB.describe_roots())
        return 0

    d = pd.read_csv(comparison)
    scored = d[d["status"].astype(str).str.startswith("scored")].copy()
    print("=" * 90)
    print("WHAT THE RUN DID")
    print("=" * 90)
    print(f"{len(d)} rows; {len(scored)} scored")
    for s, n in d["status"].value_counts().items():
        print(f"   {n:5d}  {str(s)[:78]}")

    # ──────────────────────────────────────────────────────── question 0 ────
    print()
    print("=" * 90)
    print("0. IS THE HARNESS FAITHFUL? fly_blob vs the published run")
    print("=" * 90)
    if not PUBLISHED.is_file():
        print(f"   [cannot check] {PUBLISHED.name} is absent, so the harness is")
        print("   UNVERIFIED and everything below should be read as provisional.")
    else:
        pub = pd.read_csv(PUBLISHED)
        pub = pub[pub["status"] == "scored"][KEYS + ["video_count", "hand"]]
        m = scored.merge(pub, on=KEYS, suffixes=("", "_pub"))
        print(f"   {len(m)} cells matched against the published run")
        if len(m):
            same_count = int((m["fly_blob_above"] == m["video_count"]).sum())
            same_hand = int((m["hand"] == m["hand_pub"]).sum())
            dif = (m["fly_blob_above"] - m["video_count"]).abs()
            print(f"   fly_blob identical to the published count : "
                  f"{same_count}/{len(m)} = {100.0 * same_count / len(m):.2f}%")
            print(f"   hand counts identical                    : "
                  f"{same_hand}/{len(m)} = {100.0 * same_hand / len(m):.2f}%")
            print(f"   max |difference| where they differ       : "
                  f"{float(dif.max()):.0f}")
            if same_count == len(m):
                print("   -> the harness reproduces the published run EXACTLY, so the")
                print("      three detectors are being compared on the frames that run")
                print("      scored.")
            else:
                print("   -> THE HARNESS DOES NOT REPRODUCE THE PUBLISHED RUN. Everything")
                print("      below is a comparison of something else, and the cause must")
                print("      be found before any of it is reported.")

    # ─────────────────────────────────────────────────────── question 0b ────
    print()
    print("=" * 90)
    print("0b. WHAT DID THE HAND SCORER COUNT? standing count, or crossings?")
    print("=" * 90)
    print("A cumulative count can never decrease. An instantaneous count must")
    print("decrease whenever a fly falls back, and in the impaired arms flies")
    print("demonstrably fall back. So monotonicity in time separates the two.")
    print()
    mono_rows = []
    for (vid, vial), g in scored.sort_values("time_s").groupby(["video", "vial"]):
        hh = g["hand"].to_numpy(float)
        hh = hh[np.isfinite(hh)]
        if hh.size < 3:
            continue
        nf = g["n_flies"].dropna()
        mono_rows.append({
            "Genotype": g["Genotype"].iloc[0], "Treatment": g["Treatment"].iloc[0],
            "video": vid, "vial": vial,
            "monotone": bool(np.all(np.diff(hh) >= 0)),
            "n_drops": int(np.sum(np.diff(hh) < 0)),
            "max_drop": float(-np.min(np.diff(hh))) if hh.size > 1 else 0.0,
            "saturated": bool(nf.size and hh.max() >= float(nf.iloc[0])),
            "flat_at_max": bool(np.all(hh == hh.max())),
        })
    mono = pd.DataFrame(mono_rows)
    if len(mono):
        print("  %-10s %-11s %6s %10s %10s %12s %12s"
              % ("Genotype", "Treatment", "vials", "monotone", "mean drops",
                 "reached N", "flat"))
        for (g, t), sub in mono.groupby(["Genotype", "Treatment"], sort=True):
            print("  %-10s %-11s %6d %9.0f%% %10.2f %11.0f%% %11.0f%%"
                  % (str(g)[:10], str(t)[:11], len(sub),
                     100.0 * sub["monotone"].mean(), sub["n_drops"].mean(),
                     100.0 * sub["saturated"].mean(),
                     100.0 * sub["flat_at_max"].mean()))
        print("  %-10s %-11s %6d %9.0f%% %10.2f %11.0f%% %11.0f%%"
              % ("ALL", "", len(mono), 100.0 * mono["monotone"].mean(),
                 mono["n_drops"].mean(), 100.0 * mono["saturated"].mean(),
                 100.0 * mono["flat_at_max"].mean()))
        print()
        print("  `monotone`   = the hand count never fell over the 12 timepoints")
        print("  `mean drops` = how many times it fell, per vial")
        print("  `reached N`  = the count reached the number of flies loaded")
        print("  `flat`       = the count never changed at all")

    # Both readings, scored against the same hand counts.
    print()
    print("  The same detectors, scored both ways:")
    print("    instantaneous = the count in the frame at that timepoint")
    print("    ratchet       = the running maximum over the timepoints so far,")
    print("                    which is what a cumulative scorer produces")
    print()
    print("  %-14s %10s %9s %10s %9s %10s"
          % ("detector", "inst MAE", "inst bias", "ratch MAE", "ratch bias",
             "MAE change"))
    ratchet_cols = {}
    for k in [d_ for d_ in DETECTORS if f"{d_}_above" in scored.columns]:
        c = f"{k}_above"
        inst_e, rat_e = [], []
        rat_vals = np.full(len(scored), np.nan)
        for (vid, vial), g in scored.sort_values("time_s").groupby(["video", "vial"]):
            x = g[c].to_numpy(float)
            hh = g["hand"].to_numpy(float)
            if not np.isfinite(x).any():
                continue
            xr = np.maximum.accumulate(pd.Series(x).ffill().bfill().to_numpy())
            rat_vals[[scored.index.get_loc(i) for i in g.index]] = xr
            inst_e.append(x - hh)
            rat_e.append(xr - hh)
        if not inst_e:
            continue
        ie = np.concatenate(inst_e)
        re_ = np.concatenate(rat_e)
        ratchet_cols[k] = rat_vals
        f = lambda e: (float(np.nanmean(np.abs(e))), float(np.nanmean(e)))
        im, ib = f(ie)
        rm, rb = f(re_)
        print("  %-14s %10.3f %+9.3f %10.3f %+10.3f %+10.3f"
              % (k, im, ib, rm, rb, rm - im))
    for k, v in ratchet_cols.items():
        scored[f"{k}_ratchet"] = v

    print()
    print("  HOW TO READ THIS. If every detector improves by more under the")
    print("  ratchet than they differ from each other, then the dominant error in")
    print("  this comparison is the SCORING DEFINITION and not the detector, and")
    print("  N97's ceiling is substantially a definitional mismatch rather than a")
    print("  detection failure. That would not make the detectors good; it would")
    print("  mean they have been charged for answering a different question.")
    print()
    print("  THIS IS NOT SETTLED HERE. modules/climbing_detectors.py argues the")
    print("  instantaneous rule from the protocol's own words -- a fallen fly is")
    print("  recorded as -1 and the next crosser retakes its number, so the tally")
    print("  is the number currently above the line -- and that reading looks")
    print("  right. What is in question is whether THIS scorer followed it, which")
    print("  is a question about people and is answered by asking them, as the")
    print("  scoring-line ambiguity was.")

    cols = {k: f"{k}_above" for k in DETECTORS}
    have = [k for k in DETECTORS if cols[k] in scored.columns]
    for k in have:
        scored[f"err_{k}"] = scored[cols[k]] - scored["hand"]
    usable = scored[scored["hand"].notna()].copy()

    # ──────────────────────────────────────────────────────── question 1 ────
    print()
    print("=" * 90)
    print("1. STRATIFIED BY DOSE -- does any detector escape the ceiling?")
    print("=" * 90)
    print("The ceiling is invisible in a pooled mean. Read `mean tool` against")
    print("`mean hand`: a detector that saturates reports the same number however")
    print("many flies are actually above the line.")
    print()
    hdr = "  %-10s %-11s %5s" % ("Genotype", "Treatment", "n")
    hdr += " %10s" % "mean hand"
    for k in have:
        hdr += " %12s %8s" % (k[:12], "bias")
    print(hdr)
    for (g, t), sub in usable.groupby(["Genotype", "Treatment"], sort=True):
        line = "  %-10s %-11s %5d %10.2f" % (str(g)[:10], str(t)[:11], len(sub),
                                             sub["hand"].mean())
        for k in have:
            line += " %12.2f %+8.2f" % (sub[cols[k]].mean(),
                                        sub[f"err_{k}"].mean())
        print(line)

    print()
    print("  Saturation test: the spread of each detector's mean count across")
    print("  arms, against the spread of the hand counts. A detector that tracks")
    print("  the truth reproduces its spread; one that saturates compresses it.")
    arm_means = usable.groupby(["Genotype", "Treatment"])
    h = arm_means["hand"].mean()
    print("    %-14s range %5.2f - %5.2f  (spread %5.2f)"
          % ("hand", h.min(), h.max(), h.max() - h.min()))
    for k in have:
        v = arm_means[cols[k]].mean()
        print("    %-14s range %5.2f - %5.2f  (spread %5.2f, %.0f%% of the hand "
              "spread)"
              % (k, v.min(), v.max(), v.max() - v.min(),
                 100.0 * (v.max() - v.min()) / max(h.max() - h.min(), 1e-9)))

    # ──────────────────────────────────────────────────────── question 2 ────
    print()
    print("=" * 90)
    print("2. ACCURACY AGAINST THE HAND COUNTS")
    print("=" * 90)
    print("%-16s %6s %8s %9s %8s %9s %12s %10s"
          % ("detector", "n", "MAE", "bias", "RMSE", "within2", "r(err,hand)",
             "slope"))
    for k in have:
        e = usable[f"err_{k}"].to_numpy(float)
        hh = usable["hand"].to_numpy(float)
        ok = np.isfinite(e) & np.isfinite(hh)
        s = summarise(e)
        if ok.sum() >= 3:
            r = float(np.corrcoef(e[ok], hh[ok])[0, 1])
            sl = float(np.polyfit(hh[ok], e[ok], 1)[0])
        else:
            r = sl = np.nan
        print("%-16s %6d %8.3f %+9.3f %8.3f %8.1f%% %12.4f %10.4f"
              % (k, s["n"], s["MAE"], s["bias"], s["RMSE"], s["within2"], r, sl))
    print()
    print("r(err, hand) is the ceiling expressed as a correlation: a detector")
    print("whose error grows more negative as the true count rises is saturating,")
    print("and it compresses every treatment contrast toward the null.")

    # ──────────────────────────────────────────────────────── question 3 ────
    print()
    print("=" * 90)
    print("3. DO THEY AGREE CELL BY CELL?")
    print("=" * 90)
    both = usable.dropna(subset=[cols[k] for k in have])
    print(f"{len(both)} cells where all {len(have)} detectors returned a count")
    if len(both) >= 3:
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                diff = (both[cols[a]] - both[cols[b]]).to_numpy(float)
                r = float(np.corrcoef(both[cols[a]], both[cols[b]])[0, 1])
                print("  %-14s vs %-14s mean|diff| %6.3f   identical %5.1f%%   "
                      "r %6.4f"
                      % (a, b, float(np.mean(np.abs(diff))),
                         100.0 * float(np.mean(diff == 0)), r))
        print()
        print("  Closest to the hand count, per cell (ties shared):")
        errs = np.vstack([np.abs(both[f"err_{k}"].to_numpy(float)) for k in have])
        best = errs.min(axis=0)
        for j, k in enumerate(have):
            wins = float(np.mean(errs[j] == best))
            print("    %-16s best or tied on %5.1f%% of cells" % (k, 100 * wins))

        # A paired test on the two DroPhenix stacks, which is the decision the
        # project actually has to make. Cells are not independent -- several
        # share a video and a vial -- so this is descriptive.
        if "fly_blob" in have and "fly_detector" in have:
            from scipy import stats
            da = np.abs(both["err_fly_blob"].to_numpy(float))
            db = np.abs(both["err_fly_detector"].to_numpy(float))
            nz = (db - da)[(db - da) != 0]
            print()
            print("  fly_blob vs fly_detector, paired on the same cells:")
            print("    fly_detector closer on %d cells, further on %d, tied on %d"
                  % (int(np.sum(db < da)), int(np.sum(db > da)),
                     int(np.sum(db == da))))
            if nz.size >= 6:
                w = stats.wilcoxon(nz)
                print("    Wilcoxon on |error| change: statistic %.1f, p = %.3g "
                      "(n = %d non-tied)" % (w.statistic, w.pvalue, nz.size))
                print("    Cells sharing a video and a vial are not independent, so")
                print("    this is a descriptive statistic and not a test.")

    # ─────────────────────────────────────────────── merge diagnostics ─────
    if "fly_detector_merged" in usable.columns:
        print()
        print("=" * 90)
        print("THE MERGE CORRECTION ITSELF (register N3)")
        print("=" * 90)
        mg = usable["fly_detector_merged"].to_numpy(float)
        mg = mg[np.isfinite(mg)]
        if mg.size:
            print("blobs judged merged per cell: mean %.2f, max %.0f, "
                  "zero in %.1f%% of cells"
                  % (mg.mean(), mg.max(), 100.0 * float(np.mean(mg == 0))))
            print("If that is near zero the correction is not firing, and")
            print("fly_detector's count is not being raised by it at all -- in")
            print("which case any difference from fly_blob is some other effect.")

    print()
    print("=" * 90)
    print("WHAT THIS DOES NOT ESTABLISH")
    print("=" * 90)
    print("One laboratory, one camera, one compound, two genotypes. The hand")
    print("counts are themselves one scorer's reading, with no second scorer, so")
    print("the floor on any detector's measurable accuracy is unknown. And a")
    print("detector that wins here may be the one whose parameters were fitted")
    print("here: fly_blob's fly_d_frac was fitted on this footage, which is")
    print("stated in modules/fly_blob.py and must be stated in any report of")
    print("these numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
