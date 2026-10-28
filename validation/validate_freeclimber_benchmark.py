"""
validate_freeclimber_benchmark.py
DroPhenix vs FreeClimber on REAL published experimental data.

WHY THIS SUITE EXISTS
---------------------
Every other climbing check in this repository is either simulation with a known
ground truth, or the laboratory's own footage with no independent reference.
Neither can establish that the climbing read-out agrees with the method the
field already accepts. This suite closes that gap: it reproduces the published
outputs of an independent, peer-reviewed tool from that tool's own raw data.

DATA (real, published, redistributed with the tool under its own licence)
    Spierer, Yoon, Zhu & Rand (2021) "FreeClimber: automated quantification of
    climbing performance in Drosophila", J Exp Biol 224:jeb229377.
    Repository: https://github.com/adamspierer/FreeClimber  (example/)
    Local copy:  reference_datasets/freeclimber/
      w1118_m_2_1.filtered.csv  -- 4,093 particle detections, 3 vials,
                                   145 frames of a real w1118 male climbing assay
      w1118_m_2_1.slopes.csv    -- FreeClimber's OWN published output for that file
      example.cfg               -- the analysis parameters they used (window = 50)

METHOD BEING CHECKED
    FreeClimber's velocity read-out: take the mean vertical position of the
    detected particles in each frame, then slide a `window`-frame linear
    regression across the series and report the best-fitting window. This is the
    "local linear regression" of the paper's title.

WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT
    ESTABLISHES: on real experimental data, an independent reimplementation of
    the published method recovers FreeClimber's published slope, frame window,
    correlation and standard error for every vial. The velocity read-out is
    therefore not a DroPhenix invention that happens to agree with itself.

    DOES NOT ESTABLISH: anything about DroPhenix's own particle DETECTION (this
    starts from FreeClimber's detections, not from the video), nor anything
    about the age-trajectory / MDR model, which needs longitudinal
    climbing-vs-age data. See docs/METHODS_PROVENANCE.md.

Run:  python validation/validate_freeclimber_benchmark.py   (exit 0 = passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path(__file__).resolve().parent.parent / "reference_datasets" / "freeclimber"
WINDOW = 50            # example.cfg: window=50
ROI_H = 380            # example.cfg: h=380


def best_window_slope(series: pd.Series, window: int = WINDOW):
    """
    FreeClimber's local linear regression: slide a `window`-frame regression
    across the per-frame mean position and keep the best-fitting window.
    Returns (first_frame, last_frame, slope, r_value, std_err).
    """
    f = series.index.to_numpy(dtype=float)
    y = series.to_numpy(dtype=float)
    best = None
    for i in range(0, len(f) - window):
        r = stats.linregress(f[i:i + window + 1], y[i:i + window + 1])
        if best is None or r.rvalue ** 2 > best[0]:
            best = (r.rvalue ** 2, f[i], f[i + window], r.slope, r.rvalue, r.stderr)
    if best is None:
        return None
    _, f0, f1, sl, rv, se = best
    return int(f0), int(f1), float(sl), float(rv), float(se)


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 5) if got is not None and np.isfinite(got) else got, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    det_path = DATA / "w1118_m_2_1.filtered.csv"
    pub_path = DATA / "w1118_m_2_1.slopes.csv"

    if not det_path.exists() or not pub_path.exists():
        print("=" * 78)
        print("DroPhenix vs FreeClimber benchmark")
        print("=" * 78)
        print(f"[SKIP] reference data not present at {DATA}")
        print("       Fetch with:")
        print("       curl -sLO https://raw.githubusercontent.com/adamspierer/"
              "FreeClimber/master/example/w1118_m_2_1.filtered.csv")
        print("       curl -sLO https://raw.githubusercontent.com/adamspierer/"
              "FreeClimber/master/example/w1118_m_2_1.slopes.csv")
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    det = pd.read_csv(det_path)
    pub = pd.read_csv(pub_path).set_index("vial_ID")

    # ── Sanity: the data really is what the paper describes ──────────────────
    chk("real dataset: particle detections", float(len(det)), 4093, 0)
    chk("real dataset: vials", float(det["vial"].nunique()), 3, 0)
    chk("real dataset: frames", float(det["frame"].max() - det["frame"].min() + 1), 145, 0)
    boolchk("real dataset: genotype is w1118",
            set(det["geno"].unique()) == {"w1118"}, "w1118", set(det["geno"].unique()))

    # ── Reproduce every published per-vial value ─────────────────────────────
    max_dslope = max_dr = max_dse = 0.0
    windows_match = True
    for vial in sorted(det["vial"].unique()):
        f0, f1, sl, rv, se = best_window_slope(
            det[det["vial"] == vial].groupby("frame")["y"].mean())
        p = pub.loc[f"w1118_m_{vial}"]
        windows_match &= (f0 == int(p["first_frame"]) and f1 == int(p["last_frame"]))
        max_dslope = max(max_dslope, abs(sl - float(p["slope"])))
        max_dr = max(max_dr, abs(rv - float(p["r_value"])))
        max_dse = max(max_dse, abs(se - float(p["std_err"])))
        results.append((f"vial {vial}: window {f0}-{f1} == published "
                        f"{int(p['first_frame'])}-{int(p['last_frame'])}",
                        f"{int(p['first_frame'])}-{int(p['last_frame'])}",
                        f"{f0}-{f1}",
                        f0 == int(p["first_frame"]) and f1 == int(p["last_frame"])))
        chk(f"vial {vial}: slope == published {p['slope']}", sl, float(p["slope"]), 1e-3)

    # ── The pooled "all vials" value too ─────────────────────────────────────
    f0, f1, sl, rv, se = best_window_slope(det.groupby("frame")["y"].mean())
    p = pub.loc["w1118_m_all"]
    max_dslope = max(max_dslope, abs(sl - float(p["slope"])))
    max_dr = max(max_dr, abs(rv - float(p["r_value"])))
    max_dse = max(max_dse, abs(se - float(p["std_err"])))
    chk(f"pooled: slope == published {p['slope']}", sl, float(p["slope"]), 1e-3)
    results.append((f"pooled: window {f0}-{f1} == published "
                    f"{int(p['first_frame'])}-{int(p['last_frame'])}",
                    f"{int(p['first_frame'])}-{int(p['last_frame'])}", f"{f0}-{f1}",
                    f0 == int(p["first_frame"]) and f1 == int(p["last_frame"])))

    # ── Headline agreement across all four published values ──────────────────
    boolchk("ALL frame windows identical to FreeClimber", windows_match, "identical",
            "identical" if windows_match else "differ")
    chk("max |slope - published| over 4 values", max_dslope, 0.0, 1e-3)
    chk("max |r - published| over 4 values", max_dr, 0.0, 1e-3)
    chk("max |std_err - published| over 4 values", max_dse, 0.0, 1e-3)

    # ── The cm/s unit conversion FreeClimber documents ───────────────────────
    # example.cfg: frame_rate=29, convert_to_cm_sec=True. cm/s = px/frame *
    # frame_rate / pixel_to_cm. The shipped results.csv is the same analysis in
    # px/frame; the ratio between the two files pins the conversion exactly.
    res = pd.read_csv(DATA / "results.csv")
    if not res.empty:
        # results.csv row w1118_m_1 shares the pooled window (68-118) and its
        # r/std_err match the pooled row, so it is the unconverted pooled value.
        raw = float(res.loc[res["vial_ID"] == "w1118_m_1", "slope"].iloc[0])
        conv = float(pub.loc["w1118_m_all", "slope"])
        chk("documented cm/s conversion factor = frame_rate / pixel_to_cm",
            conv / raw, 29.0 / 45.0, 1e-4)

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 78)
    print("DroPhenix vs FreeClimber (Spierer et al. 2021, J Exp Biol 224:jeb229377)")
    print("REAL published data: 4,093 detections, 3 vials, w1118 males")
    print("=" * 78)
    print(f"{'Check':<54}{'Expected':>12}{'Got':>8}{'Pass':>4}")
    print("-" * 78)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<54}{str(exp):>12}{str(got):>8}{'OK' if ok else 'FAIL':>4}")
    print("-" * 78)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
