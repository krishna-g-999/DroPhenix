"""
study_climbing_accuracy_2026_11.py
The climbing detector's accuracy against hand counts -- for the CURRENT build,
and for what the repairs bought.

WHY THIS SCRIPT EXISTS
----------------------
The register carried an accuracy figure of MAE 2.69 flies, then briefly MAE
3.70, and both described a build that no longer exists. The CSVs those numbers
came from are dated 2026-08-20 and 2026-08-22; four climbing-video defects were
repaired on 2026-08-27, and P0 Blocks A-D landed on 2026-11-08 including N3's
merge-splitting fix, which raised measured recall in a severe phenotype from
0.35 to 0.90. So the numbers measured a detector that predated both rounds of
repair, and N3 is precisely the fix that bears on the defect they described.

A data file carries no version. This script therefore compares TWO runs over
the SAME recordings and the SAME hand counts:

    paraquat_scored_net_blob.csv           2026-08-22, the pre-fix baseline
    paraquat_scored_net_blob_2026-11.csv   the current build

The difference between them is the measure of what the repairs bought, and
keeping the August file rather than overwriting it is what makes that
measurable at all.

GROUND TRUTH
    Hand counts made by this laboratory on its own paraquat recordings: one
    count per (video, tap, vial, timepoint) cell, 1,080 cells in the full
    design. They are independent of the software and were made before any of
    this.

THE QUESTION THAT MATTERS MOST IS NOT THE POOLED ERROR
    It is whether accuracy DEGRADES WITH IMPAIRMENT. A detector that misses
    flies in proportion to how sick they are confounds every treatment effect
    with its own detection failure, because the arm with the largest effect is
    also the worst measured. So everything here is stratified by paraquat dose,
    and the pooled figure is reported last and labelled as the least
    informative one.

Run: ./.venv/Scripts/python.exe validation/study_climbing_accuracy_2026_11.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from validation import lab_data as _LAB
except Exception:                                      # direct-script use
    import lab_data as _LAB

PARA = ((_LAB.climbing_root() or Path("_absent"))
        / "Paraquat climbing assay 010626_new set")

RUNS = {
    "Aug 2026 (pre-fix)": "paraquat_scored_net_blob.csv",
    "Nov 2026 (current)": "paraquat_scored_net_blob_2026-11.csv",
}


def load(name: str) -> Optional[pd.DataFrame]:
    p = PARA / name
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    d["err"] = d["video_count"] - d["hand"]
    return d


def usable(d: pd.DataFrame) -> pd.DataFrame:
    """Only cells where BOTH a tool count and a hand count exist are comparable."""
    return d[(d["status"] == "scored") & d["video_count"].notna()
             & d["hand"].notna()].copy()


def summarise(e: np.ndarray) -> Dict[str, float]:
    return {
        "n": float(e.size),
        "MAE": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "within2": 100.0 * float(np.mean(np.abs(e) <= 2)),
    }


def main() -> int:
    if not PARA.is_dir():
        print("The paraquat recordings are not on this machine.")
        print(_LAB.describe_roots())
        return 0

    loaded = {k: load(v) for k, v in RUNS.items()}
    missing = [k for k, v in loaded.items() if v is None]
    for k in missing:
        print(f"[absent] {k}: {RUNS[k]}")
        loaded.pop(k)
    if not loaded:
        print("Neither run is present; nothing to measure.")
        return 0

    print("=" * 88)
    print("WHAT EACH RUN DID WITH THE 705 CELLS")
    print("=" * 88)
    for k, d in loaded.items():
        print(f"{k:22s} {len(d):4d} cells   {dict(d['status'].value_counts())}")
        refs = d.loc[d["refusal"].notna(), "refusal"]
        if len(refs):
            for reason, n in refs.value_counts().head(4).items():
                print(f"{'':22s}   {n:3d} x {str(reason)[:70]}")

    U = {k: usable(d) for k, d in loaded.items()}

    print()
    print("=" * 88)
    print("STRATIFIED BY PARAQUAT DOSE -- does accuracy degrade with impairment?")
    print("=" * 88)
    print("This is the reading that matters. A detector whose error grows with")
    print("the dose confounds the treatment effect with its own failure.")
    print()
    for k, d in U.items():
        if not len(d):
            print(f"{k}: no comparable cells")
            continue
        print(f"{k}:")
        print("  %-10s %-12s %5s %8s %8s %10s %10s"
              % ("Genotype", "Treatment", "n", "MAE", "bias", "mean hand", "mean tool"))
        for (g, t), sub in d.groupby(["Genotype", "Treatment"], sort=True):
            e = sub["err"].to_numpy(float)
            print("  %-10s %-12s %5d %8.3f %+8.3f %10.2f %10.2f"
                  % (str(g)[:10], str(t)[:12], len(sub), np.mean(np.abs(e)),
                     np.mean(e), sub["hand"].mean(), sub["video_count"].mean()))
        print()

    print("=" * 88)
    print("THE CONFOUND TEST: is the error correlated with the true count?")
    print("=" * 88)
    print("%-22s %12s %12s %10s %s"
          % ("run", "r(err,hand)", "slope", "n", "reading"))
    for k, d in U.items():
        if len(d) < 3:
            continue
        h = d["hand"].to_numpy(float)
        e = d["err"].to_numpy(float)
        r = float(np.corrcoef(e, h)[0, 1])
        slope = float(np.polyfit(h, e, 1)[0])
        reading = ("error grows with the count -- CONFOUNDED"
                   if abs(r) > 0.3 else "little dependence on the count")
        print("%-22s %12.4f %12.4f %10d %s" % (k, r, slope, len(d), reading))

    print()
    print("=" * 88)
    print("CELL-BY-CELL, ON THE CELLS BOTH RUNS SCORED")
    print("=" * 88)
    keys = ["folder", "video", "tap", "replicate", "vial", "time_s"]
    names = list(U)
    if len(names) == 2 and all(len(U[n]) for n in names):
        a, b = U[names[0]], U[names[1]]
        m = a.merge(b, on=keys, suffixes=("_a", "_b"))
        print(f"{len(m)} cells scored by BOTH runs "
              f"(of {len(a)} and {len(b)} respectively)")
        if len(m):
            ea = (m["video_count_a"] - m["hand_a"]).to_numpy(float)
            eb = (m["video_count_b"] - m["hand_b"]).to_numpy(float)
            sa, sb = summarise(ea), summarise(eb)
            print()
            print("%-14s %10s %10s %10s"
                  % ("", names[0][:10], names[1][:10], "change"))
            for field in ("MAE", "bias", "RMSE", "within2"):
                print("%-14s %10.3f %10.3f %+10.3f"
                      % (field, sa[field], sb[field], sb[field] - sa[field]))
            better = int(np.sum(np.abs(eb) < np.abs(ea)))
            worse = int(np.sum(np.abs(eb) > np.abs(ea)))
            same = int(np.sum(np.abs(eb) == np.abs(ea)))
            print()
            print("per cell: %d closer to the hand count, %d further, %d unchanged"
                  % (better, worse, same))
            # A paired test, because the same cells are measured twice.
            from scipy import stats
            d_abs = np.abs(eb) - np.abs(ea)
            nz = d_abs[d_abs != 0]
            if nz.size >= 6:
                w = stats.wilcoxon(nz)
                print("Wilcoxon signed-rank on |error| change: "
                      "statistic %.1f, p = %.3g  (n = %d non-tied cells)"
                      % (w.statistic, w.pvalue, nz.size))
                print("NOTE: these cells are not independent -- several share a")
                print("video and a vial -- so this p-value is anticonservative and")
                print("is reported as a descriptive statistic, not a test.")

    print()
    print("=" * 88)
    print("POOLED, WHICH IS THE LEAST INFORMATIVE FIGURE HERE")
    print("=" * 88)
    print("%-22s %6s %8s %8s %8s %9s"
          % ("run", "n", "MAE", "bias", "RMSE", "within±2"))
    for k, d in U.items():
        if not len(d):
            continue
        s = summarise(d["err"].to_numpy(float))
        print("%-22s %6d %8.3f %+8.3f %8.3f %8.1f%%"
              % (k, int(s["n"]), s["MAE"], s["bias"], s["RMSE"], s["within2"]))
    print()
    print("Pooled over doses, an error that grows with impairment and one that")
    print("is uniform produce the same number. Read the stratified table above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
