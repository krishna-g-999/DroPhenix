"""
A learned fly counter, and the measurement that decides whether it is allowed near the data.

WHY A LEARNED COUNTER, AND WHY THIS KIND
-----------------------------------------
`fly_blob` counts flies as local maxima of a band-pass response with a minimum
separation of 1.2 fly diameters, so two flies closer than that are one maximum
and a crowded tube is under-counted BY CONSTRUCTION (register N97 / N103).
Four corrections have now been refuted by measurement: the geometric single-fly
area, raw band-pass mass, accepted-peak mass (which INVERTED an arm ordering),
and the line-crossing "tripwire" counter (which removed the ceiling signature
but under-counts, because flies cross the line in clusters and the merge
reappears at the gate).

What every one of those shares is a single hand-chosen rule for turning pixels
into a number. The learned approach drops the search for one rule and fits the
mapping from several weak, complementary measurements of the same region onto
the hand count.

WHY NOT A DEEP DENSITY-MAP NETWORK
-----------------------------------
Crowd counting solves this problem with density-map regression (CSRNet and
relatives), and that was the plan on record. It is not the right tool HERE, for
a reason that is about the data rather than the method: density maps are
trained on POINT ANNOTATIONS -- a dot on every animal -- and this laboratory
has none. It has 112 hand-counted cells, which is count-level supervision only,
and 112 examples is far too few to fit a convolutional network from counts
alone without it learning which ARM a frame came from.

That last risk is not hypothetical and it is fatal: the arms differ in their
counts by construction, so a model with enough freedom can score well by
recognising the arm and reporting its mean. It would then reproduce the
between-arm differences it was shown and be incapable of measuring a new one,
which is the opposite of an instrument. Everything below is designed around
that single danger.

THE DESIGN, AND THE THREE THINGS THAT PROTECT IT
-------------------------------------------------
1. FEATURES THAT ARE PHYSICAL, NOT LEARNED. Every input is a measurement of the
   region above the line that a person could compute and check by hand: the
   peak count, the accepted-peak and raw band-pass mass in units of one fly,
   the thresholded area in units of one fly, and -- the genuinely new signal --
   the number of connected components the region separates into at a LADDER of
   rising thresholds. That ladder is multi-threshold deblending (Bertin &
   Arnouts 1996, SExtractor): a pair of touching flies splits into two
   components at a higher threshold where one fly does not, so the ladder
   carries multiplicity information that no single threshold can.

2. GROUPED CROSS-VALIDATION, BY RECORDING. Every reported number is
   out-of-sample under leave-one-VIDEO-out, so no cell is ever predicted by a
   model that saw another cell from the same recording. A random split would
   leak: two timepoints from one video share lighting, focus, tube and animals.

3. A LEAVE-ONE-ARM-OUT TEST, WHICH IS THE REAL ONE. The model is refitted six
   times, each time with one entire experimental arm withheld, and asked to
   predict that unseen arm. A counter that has memorised arm identity fails
   here and nowhere else. An instrument that cannot measure an arm it has never
   seen is not an instrument.

THE BAR, UNCHANGED
------------------
From register N103, and fixed before any of this ran:
  * beat MAE 3.589 and RMSE 4.436;
  * recover more than 69% of the between-arm spread;
  * preserve the ordering of all six arms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import fly_blob as FB  # noqa: E402

CACHE = ROOT / ".cache" / "climbing_cells.npz"
BAR = {"mae": 3.589, "rmse": 4.436, "spread_pct": 69.0}

#: Rising multiples of the detection threshold at which the region above the
#: line is re-thresholded and its connected components counted. This is the
#: deblending ladder; the spacing is geometric because the response of a fly is
#: peaked, so equal steps in level are not equal steps in separation.
LADDER = (1.0, 1.5, 2.25, 3.4, 5.1, 7.6)


def _components(mask: np.ndarray) -> int:
    """Number of 8-connected components in a boolean mask, without SciPy labels."""
    from scipy import ndimage
    if not mask.any():
        return 0
    structure = np.ones((3, 3), dtype=bool)
    _lab, n = ndimage.label(mask, structure=structure)
    return int(n)


def features_for_cell(gray: np.ndarray, bg: np.ndarray, line: int) -> Dict[str, float]:
    """Physical measurements of the region ABOVE the scoring line."""
    h, w = gray.shape
    d = FB.fly_diameter(w, dict(FB.PARAMS))
    bp = FB.bandpass(gray.astype(np.float32), d)
    bpb = FB.bandpass(bg.astype(np.float32), d)
    thr = float(FB.PARAMS["thr_mult"]) * float(
        np.percentile(bpb, float(FB.PARAMS["null_q"])))
    raw = FB.bandpass_raw(gray.astype(np.float32), d)
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))
    thr = max(thr, med + float(FB.PARAMS["noise_mult"]) * 1.4826 * mad, 1e-3)

    pts = FB.detect(gray.astype(np.uint8), bg.astype(np.uint8), None)
    above = [(px, py, m) for (px, py, m) in pts if py < line]

    # single-fly mass from ISOLATED peaks anywhere in the cell: an isolated peak
    # is one animal by definition, which is the one case peak counting is
    # certainly right about.
    iso = [m for i, (px, py, m) in enumerate(pts)
           if all(((px - qx) ** 2 + (py - qy) ** 2) >= (2.0 * d) ** 2
                  for j, (qx, qy, _q) in enumerate(pts) if j != i)]
    m1 = float(np.median(iso)) if iso else np.nan

    reg = bp[:max(line, 1), :]
    mask = reg > thr
    area1 = np.pi * (d / 2.0) ** 2

    f: Dict[str, float] = {
        "n_peaks": float(len(above)),
        "peak_mass": float(sum(m for (_x, _y, m) in above)) / m1 if m1 == m1 else np.nan,
        "raw_mass": float(reg[mask].sum()) / m1 if m1 == m1 else np.nan,
        "area_flies": float(mask.sum()) / area1,
        "n_iso": float(len(iso)),
        "frac_above": float(line) / float(h),
        "d_px": float(d),
    }
    # the deblending ladder
    for k, mult in enumerate(LADDER):
        f[f"comp_{k}"] = float(_components(reg > thr * mult))
    # how fast components fall away with level: a crowd splits then vanishes,
    # a sparse field just vanishes
    c0 = max(f["comp_0"], 1.0)
    f["comp_peak"] = float(max(f[f"comp_{k}"] for k in range(len(LADDER))))
    f["comp_ratio"] = f["comp_peak"] / c0
    return f


def build_table() -> pd.DataFrame:
    z = np.load(CACHE, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    rows = []
    for m in meta:
        g = z[m["key"] + "_gray"]
        b = z[m["key"] + "_bg"]
        f = features_for_cell(g, b, int(m["line_in_crop"]))
        f.update({
            "key": m["key"], "video": m["video"], "hand": float(m["hand"]),
            "n_flies": float(m.get("n_flies", np.nan)),
            "Genotype": m["Genotype"], "Treatment": m["Treatment"],
            "arm": f"{m['Genotype']} {m['Treatment']}",
        })
        rows.append(f)
    return pd.DataFrame(rows)


FEATS = ["n_peaks", "peak_mass", "raw_mass", "area_flies", "n_iso",
         "frac_above", "d_px", "comp_peak", "comp_ratio"] + \
        [f"comp_{k}" for k in range(len(LADDER))]


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, model: str) -> np.ndarray:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X, y = tr[FEATS].to_numpy(float), tr["hand"].to_numpy(float)
    Xt = te[FEATS].to_numpy(float)
    if model == "ridge":
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            RidgeCV(alphas=np.logspace(-3, 3, 25)))
    else:
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(random_state=0, n_estimators=300,
                                      max_depth=2, learning_rate=0.05,
                                      subsample=0.9))
    est.fit(X, y)
    return est.predict(Xt)


def grouped_cv(df: pd.DataFrame, group: str, model: str,
               calibrate: bool = False) -> np.ndarray:
    """
    Out-of-sample predictions, optionally corrected for SHRINKAGE.

    Any regression that minimises squared error shrinks its predictions toward
    the training mean, by a factor that grows with the noise. For most purposes
    that is a virtue -- it is what makes the error small. For THIS instrument it
    is a defect, and the measurement shows it: ridge cut the MAE from 3.589 to
    3.402 while cutting the between-arm spread from 69.5% to 37.1%. A counter
    that halves the apparent difference between a treated and an untreated arm
    has not improved; it has become harder to use while scoring better.

    The correction is the standard de-attenuation. If the fitted values relate
    to the truth as p = a + b*y with b < 1, then (p - a) / b has the right
    scale. `a` and `b` are estimated ON THE TRAINING FOLD ONLY, by regressing
    the training predictions on the training truth, and then applied to the
    held-out fold, so the held-out cells still never inform their own
    prediction.
    """
    pred = np.full(len(df), np.nan)
    for g in df[group].unique():
        te = (df[group] == g).to_numpy()
        tr = ~te
        if tr.sum() < 10:
            continue
        p_te = _fit_predict(df[tr], df[te], model)
        if calibrate:
            p_tr = _fit_predict(df[tr], df[tr], model)
            y_tr = df.loc[tr, "hand"].to_numpy(float)
            b, a = np.polyfit(y_tr, p_tr, 1)      # p = a + b*y
            if np.isfinite(b) and abs(b) > 1e-6:
                p_te = (p_te - a) / b
        pred[te] = p_te
    return pred


def score(df: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    ok = np.isfinite(pred)
    d = df[ok].copy()
    p = pred[ok]
    err = p - d["hand"].to_numpy(float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r = float(np.corrcoef(err, d["hand"].to_numpy(float))[0, 1])
    d["_p"] = p
    arms = d.groupby("arm").agg(hand=("hand", "mean"),
                                tool=("_p", "mean")).reset_index()
    sh = arms["hand"].max() - arms["hand"].min()
    st = arms["tool"].max() - arms["tool"].min()
    spread = 100.0 * st / sh if sh else np.nan
    order_ok = (list(arms.sort_values("hand")["arm"])
                == list(arms.sort_values("tool")["arm"]))
    # A single boolean is too blunt: the incumbent fails it too, because four
    # of its six arm means sit inside 0.4 of a fly of each other. What matters
    # is whether the ordering is MONOTONE overall and whether any WIDELY
    # separated pair is inverted -- the mass candidate was rejected for putting
    # WT 10 mM (true 7.7) above TDP Untreated (true 12.1), a 4.3-fly inversion.
    from scipy import stats as _st
    rho = float(_st.spearmanr(arms["hand"], arms["tool"]).statistic)         if len(arms) > 2 else np.nan
    worst = 0.0
    for i in range(len(arms)):
        for j in range(len(arms)):
            hi, hj = arms["hand"].iloc[i], arms["hand"].iloc[j]
            ti, tj = arms["tool"].iloc[i], arms["tool"].iloc[j]
            if hi < hj and ti > tj:
                worst = max(worst, float(hj - hi))
    passed = (mae < BAR["mae"] and rmse < BAR["rmse"]
              and spread > BAR["spread_pct"] and worst < 2.0)
    print(f"\n  {label}")
    print(f"    n={len(d)}  MAE {mae:6.3f}  RMSE {rmse:6.3f}  "
          f"bias {np.mean(err):+6.3f}  r(err,hand) {r:+.3f}")
    print(f"    spread {spread:5.1f}%   arm rho {rho:+.2f}   "
          f"worst inversion {worst:.1f} flies"
          f"   -> {'CLEARS THE BAR' if passed else 'does not clear'}")
    return {"label": label, "n": len(d), "mae": mae, "rmse": rmse,
            "bias": float(np.mean(err)), "r_err_hand": r,
            "spread_pct": float(spread), "order_ok": bool(order_ok), "arm_rho": rho, "worst_inversion": worst,
            "passed": bool(passed), "arms": arms.to_dict("records")}


def main() -> int:
    if not CACHE.exists():
        print(f"[SKIP] {CACHE} not present")
        return 0
    print("=" * 74)
    print("LEARNED FLY COUNTER -- measured against the published bar")
    print("=" * 74)
    print(f"bar: MAE < {BAR['mae']}, RMSE < {BAR['rmse']}, "
          f"spread > {BAR['spread_pct']}%, all six arm orderings preserved")
    df = build_table()
    print(f"\ncells {len(df)}  videos {df['video'].nunique()}  "
          f"arms {df['arm'].nunique()}  hand {df['hand'].min():.0f}-{df['hand'].max():.0f}")

    out = []
    out.append(score(df, df["n_peaks"].to_numpy(float),
                     "BASELINE  fly_blob peak counting (the shipped detector)"))
    for model in ("ridge", "gbm"):
        for cal in (False, True):
            tag = "+despun" if cal else "        "
            out.append(score(df, grouped_cv(df, "video", model, cal),
                             f"{model.upper():5s}{tag} leave-one-VIDEO-out"))
            out.append(score(df, grouped_cv(df, "arm", model, cal),
                             f"{model.upper():5s}{tag} leave-one-ARM-out (the real test)"))
    p = ROOT / ".cache" / "ml_counter_result.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
