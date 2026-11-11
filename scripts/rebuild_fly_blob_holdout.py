"""
Rebuild the detector's hold-out accuracy measurement so that it can be CHECKED.

WHY THIS EXISTS (register N112 -> N128)
----------------------------------------
`fly_blob.MEASURED` asserted the shipped detector's accuracy as a module-level
literal: MAE 2.69, bias -1.60, within-3 0.673, over 110 cells and 22 vials.
`run_all.py` advertised `validate_fly_blob` as the suite that
"re-derives its held-out accuracy rather than asserting it". The data it
re-derived from lived in a Claude-session temp directory. That directory
survives; `vial_cache_big.npz` does not, and no committed script ever built it,
so the re-derivation was lost for the authors as well as for a stranger -- and
the suite still reported PASS.

This script rebuilds the measurement from data that EXISTS and commits a record
of it, so the claim becomes auditable. It does NOT recover the original 22-vial
subset, which cannot be identified from anything in the repository; see
"what this does and does not reproduce" below.

WHAT IT MEASURES
----------------
Two different accuracies, which the old literal conflated under one heading:

  n_total vs flies LOADED      -- how many animals the detector finds in the
                                  whole cylinder, against the number put in.
                                  This is what `mae_total` and `bias_total`
                                  were.
  n_above vs the HAND COUNT    -- how many it places above the scoring line,
                                  against a person scoring the same frame.
                                  This is the number the assay actually uses,
                                  and it is the better-grounded of the two:
                                  "flies loaded" assumes none are dead at the
                                  bottom, occluded, or outside the crop, and
                                  that assumption is not checkable from a frame.

WHAT THIS DOES AND DOES NOT REPRODUCE
--------------------------------------
Re-derived here on 112 cells from 56 vials:

    MAE(total)    3.286   against the old literal's 2.69
    bias(total)  -1.554   against the old literal's -1.60
    within3       0.625   against the old literal's 0.673

The BIAS agrees to 0.05, which is strong evidence this is the same measurement
on overlapping footage. The MAE does not, and the reason is visible in the
counts: the old figure covered 110 cells from 22 vials, and the cells available
here are 112 from 56. The original subset is not identified anywhere in the
repository, so it cannot be reconstructed, and claiming the old MAE as
reproduced would be false.

`MEASURED` is therefore replaced with these figures, which a suite re-derives
from pixels on any machine holding `.cache/climbing_cells.npz`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import fly_blob as FB  # noqa: E402

CELLS = ROOT / ".cache" / "climbing_cells.npz"
OUT_CSV = ROOT / "data" / "validation" / "fly_blob_holdout_rebuilt.csv"


def main() -> int:
    if not CELLS.exists():
        print(f"[SKIP] {CELLS} is not on this machine. The hold-out cannot be "
              f"rebuilt here; it needs the cached climbing cells.")
        return 0
    z = np.load(CELLS, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    rows = []
    for m in meta:
        g, b = z[m["key"] + "_gray"], z[m["key"] + "_bg"]
        line = int(m["line_in_crop"])
        pts = FB.detect(g, b, None)
        n_total = len(pts)
        n_above = sum(1 for (_x, cy, _mm) in pts if cy < line)
        rows.append({
            "key": m["key"], "video": m["video"], "vial": m["vial"],
            "time_s": m["time_s"], "Genotype": m["Genotype"],
            "Treatment": m["Treatment"],
            "n_flies_loaded": int(m.get("n_flies", 0)),
            "hand_above": int(m["hand"]),
            "detected_total": int(n_total), "detected_above": int(n_above),
        })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    et = df["detected_total"] - df["n_flies_loaded"]
    ea = df["detected_above"] - df["hand_above"]
    summary = {
        "n_cells": int(len(df)),
        "n_vials": int(df.groupby(["video", "vial"]).ngroups),
        "mae_total": round(float(np.mean(np.abs(et))), 3),
        "bias_total": round(float(np.mean(et)), 3),
        "within3_total": round(float(np.mean(np.abs(et) <= 3)), 3),
        "mae_above": round(float(np.mean(np.abs(ea))), 3),
        "bias_above": round(float(np.mean(ea)), 3),
        "within3_above": round(float(np.mean(np.abs(ea) <= 3)), 3),
    }
    print(f"wrote {OUT_CSV}  ({len(df)} cells, {summary['n_vials']} vials)")
    print()
    print("  n_total vs flies LOADED : MAE %(mae_total)s  bias %(bias_total)s  "
          "within3 %(within3_total)s" % summary)
    print("  n_above vs HAND COUNT   : MAE %(mae_above)s  bias %(bias_above)s  "
          "within3 %(within3_above)s" % summary)
    print()
    print("  paste into modules/fly_blob.py as MEASURED:")
    print("  " + json.dumps(summary, indent=2).replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
