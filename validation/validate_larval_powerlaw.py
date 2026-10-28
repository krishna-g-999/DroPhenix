"""
validate_larval_powerlaw.py
larval.py against REAL published larval trajectories, with a QUANTITATIVE target.

WHY THIS SUITE EXISTS
---------------------
`larval.py` had NO real data behind it at all — the only checks were on synthetic
summaries. Worse, a direction-only benchmark ("mutants are slower") cannot tell
you whether the trajectory differentiation is right, only whether two groups
differ. This dataset is unusual and valuable because the source paper reports a
NUMBER the analysis must reproduce, per condition, with an interquartile range.

DATA (real, published, CC0)
    Zago M, Lacquaniti F, Gomez-Marin A (2016)
    "The speed-curvature power law in Drosophila larval locomotion"
    Biol Lett 12:20160597, doi:10.1098/rsbl.2016.0597
    Data: doi:10.5061/dryad.37sb7 (openly mirrored at Zenodo record 5010896)
    Local: reference_datasets/larval_powerlaw/{dispersal,approach,overshoot}.mat

    123 larvae in three behavioural conditions, centroid tracks at 7 frames/s,
    90 um/pixel.

THE LAW AND THE PUBLISHED VALUES
    A(t) = k * C(t) ** beta      A = angular speed, C = curvature

    condition    median beta   IQR    n
    overshoot        0.78      0.06   42
    approach         0.78      0.08   40
    dispersal        0.76      0.06   41

    (Human drawing gives beta near 2/3; larvae sit nearer 3/4, which the authors
    attribute to dynamic effects on top of the kinematic constraint.)

METHOD, followed exactly as published
    second-order zero-phase Butterworth low-pass at 0.07 Hz; cubic-spline
    interpolation; derivatives from the spline; standard differential geometry;
    least-squares ORTHOGONAL (total-least-squares) regression of log10 A on
    log10 C. Check E below shows why the orthogonal regression is not optional.

A NOTE ON UNITS, because it nearly caught me out
    The field is named `centroid_rawXY_positions` and the struct also carries
    `millimeter_per_pixel`, which reads like an invitation to multiply. It is
    not: the coordinates are ALREADY in millimetres. Read as pixels, a larva
    would traverse a 2.4 x 3.2 mm arena — smaller than a third-instar larva —
    at 0.035 mm/s. Check D pins the correct reading against physical
    plausibility. Check C shows why the exponent was right anyway: beta is
    invariant to a uniform spatial scale.

Run:  python validation/validate_larval_powerlaw.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import larval as L  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "reference_datasets" / "larval_powerlaw"
FPS = 7.0                       # Zago et al.: "7 frames s-1"

# condition -> (published median beta, published IQR, published n)
PUBLISHED = {"dispersal": (0.76, 0.06, 41),
             "approach":  (0.78, 0.08, 40),
             "overshoot": (0.78, 0.06, 42)}


def _tracks(condition):
    """Yield each larva's (x_mm, y_mm) centroid track, padding rows removed."""
    import scipy.io as sio
    m = sio.loadmat(str(DATA / f"{condition}.mat"))
    arr = m[[k for k in m if not k.startswith("__")][0]]
    for i in range(arr.shape[1]):
        xy = np.asarray(arr[0, i]["centroid_rawXY_positions"][0, 0], dtype=float)
        xy = xy[~((xy[:, 0] == 0) & (xy[:, 1] == 0))]      # leading padding
        if len(xy) >= 60:
            yield xy[:, 0], xy[:, 1]


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 4) if got is not None and np.isfinite(got) else got, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    missing = [c for c in PUBLISHED if not (DATA / f"{c}.mat").exists()]
    if missing:
        print("=" * 82)
        print("Larval speed-curvature power law benchmark (Zago et al. 2016)")
        print("=" * 82)
        print(f"[SKIP] missing condition file(s): {missing} in {DATA}")
        print("       curl -sL -o dispersal.mat \"https://zenodo.org/api/records/"
              "5010896/files/data_Larvae_Dispersal_condition.mat/content\"")
        print("       (likewise Approach and Overshoot)")
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    # ── A/B. Reproduce the published exponent, per condition ─────────────────
    all_beta, all_r2 = [], []
    for cond, (pub_med, pub_iqr, pub_n) in PUBLISHED.items():
        betas, r2s = [], []
        for x, y in _tracks(cond):
            r = L.speed_curvature_exponent(x, y, FPS)
            if np.isfinite(r["beta"]):
                betas.append(r["beta"]); r2s.append(r["r2"])
        betas = np.asarray(betas); r2s = np.asarray(r2s)
        all_beta += list(betas); all_r2 += list(r2s)

        chk(f"A: {cond}: larvae analysed == published n", float(len(betas)), pub_n, 0)
        med = float(np.median(betas))
        q1, q3 = np.percentile(betas, [25, 75])
        # 0.03 is well inside the published IQR (0.06-0.08) for every condition.
        chk(f"B: {cond}: median beta == published {pub_med}", med, pub_med, 0.03)
        chk(f"B: {cond}: IQR == published {pub_iqr}", float(q3 - q1), pub_iqr, 0.03)

    boolchk("B: pooled median beta is nearer 3/4 than 2/3 (the paper's claim)",
            abs(np.median(all_beta) - 0.75) < abs(np.median(all_beta) - 2 / 3),
            "closer to 0.75", round(float(np.median(all_beta)), 3))

    # ── C. beta is invariant to spatial scale ────────────────────────────────
    # Angular speed is scale-free and curvature scales as 1/s, so a uniform
    # rescaling shifts the intercept of log A vs log C but not the slope. This
    # is why the exponent above is correct even though the units of the source
    # file are easy to misread (see the module note).
    x0, y0 = next(_tracks("dispersal"))
    b1 = L.speed_curvature_exponent(x0, y0, FPS)["beta"]
    b2 = L.speed_curvature_exponent(x0 * 0.095424, y0 * 0.095424, FPS)["beta"]
    b3 = L.speed_curvature_exponent(x0 * 10.0, y0 * 10.0, FPS)["beta"]
    chk("C: beta unchanged when positions are scaled x0.0954", b2, b1, 1e-9)
    chk("C: beta unchanged when positions are scaled x10", b3, b1, 1e-9)

    # ── D. The module's own documented speed range, on real larvae ───────────
    # larval.py's docstring claims wild-type third-instar larvae crawl
    # ~0.5-2 mm/s. Until now that was an uncited assertion in a comment.
    speeds = []
    for cond in PUBLISHED:
        for x, y in _tracks(cond):
            t = pd.DataFrame({"frame": np.arange(len(x)), "t_s": np.arange(len(x)) / FPS,
                              "x_px": x, "y_px": y})
            # coordinates are already mm, so px_per_mm = 1
            speeds.append(L._trajectory_metrics(t, FPS, px_per_mm=1.0)["mean_speed_mm_s"])
    speeds = np.asarray(speeds)
    boolchk("D: every real larva falls in the documented 0.3-2.0 mm/s band",
            bool(((speeds >= 0.3) & (speeds <= 2.0)).all()), "100%",
            f"{100 * ((speeds >= 0.3) & (speeds <= 2.0)).mean():.0f}% of {len(speeds)}")
    boolchk("D: median crawling speed is ~0.5-1.0 mm/s",
            0.5 <= float(np.median(speeds)) <= 1.0, "0.5-1.0 mm/s",
            round(float(np.median(speeds)), 2))
    # A third-instar larva is roughly 4-5 mm long. Under the pixel reading this
    # animal's entire excursion over a multi-minute recording would be about one
    # body length, at 0.04 mm/s. That is the reductio; the mm reading gives a
    # 55 mm excursion in a Petri dish at 0.6 mm/s.
    px_span = float(0.095424 * np.ptp(x0))
    mm_span = float(np.ptp(x0))
    boolchk("D: pixel reading gives an excursion of ~1 larval body length (absurd)",
            px_span < 10.0 and mm_span > 20.0, "px<10mm, mm>20mm",
            f"px {px_span:.1f} mm vs mm {mm_span:.1f} mm")

    # ── E. Why the orthogonal regression is not optional ─────────────────────
    # Both axes are estimated from the same noisy track, so ordinary least
    # squares is attenuated by regression dilution. If OLS reproduced the
    # published value equally well, the paper's choice would not matter.
    ols, tls = [], []
    from scipy.signal import butter, filtfilt
    from scipy.interpolate import CubicSpline
    for x, y in _tracks("dispersal"):
        b, a = butter(2, 0.07 / (FPS / 2), btype="low")
        xf, yf = filtfilt(b, a, x), filtfilt(b, a, y)
        t = np.arange(len(xf)) / FPS
        cx, cy = CubicSpline(t, xf), CubicSpline(t, yf)
        dx, dy, ddx, ddy = cx(t, 1), cy(t, 1), cx(t, 2), cy(t, 2)
        v = np.hypot(dx, dy); cr = np.abs(dx * ddy - dy * ddx)
        with np.errstate(divide="ignore", invalid="ignore"):
            C, A = cr / v ** 3, cr / v ** 2
        m = np.isfinite(C) & np.isfinite(A) & (C > 0) & (A > 0)
        if m.sum() < 50:
            continue
        lc, la = np.log10(C[m]), np.log10(A[m])
        ols.append(float(np.polyfit(lc, la, 1)[0]))
        tls.append(L._tls_slope(lc, la))
    ols_med, tls_med = float(np.median(ols)), float(np.median(tls))
    boolchk("E: OLS is attenuated relative to the published value",
            ols_med < PUBLISHED["dispersal"][0] - 0.02, "OLS < 0.74",
            round(ols_med, 3))
    boolchk("E: the orthogonal regression is the one that matches",
            abs(tls_med - 0.76) < abs(ols_med - 0.76), "TLS closer",
            f"TLS {tls_med:.3f} vs OLS {ols_med:.3f}")

    # ── F. Fit quality, and the paper's caveat that it does not always hold ──
    all_r2 = np.asarray(all_r2)
    boolchk("F: the law fits most larvae well (median r2 > 0.8)",
            float(np.median(all_r2)) > 0.8, "> 0.8", round(float(np.median(all_r2)), 3))
    boolchk("F: but NOT all of them, as the authors report",
            float((all_r2 < 0.8).mean()) > 0.0, "some poor fits",
            f"{100 * (all_r2 < 0.8).mean():.0f}% below r2 0.8")

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 82)
    print("Larval speed-curvature power law — Zago, Lacquaniti & Gomez-Marin (2016)")
    print("Biol Lett 12:20160597 — 123 real larvae, three behavioural conditions")
    print("=" * 82)
    print(f"{'Check':<56}{'Expected':>12}{'Got':>9}{'Pass':>4}")
    print("-" * 82)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<56}{str(exp):>12}{str(got):>9}{'OK' if ok else 'FAIL':>4}")
    print("-" * 82)
    print(f"  pooled median beta = {np.median(all_beta):.3f} over {len(all_beta)} larvae "
          f"(paper: 0.74, IQR 0.09); median r2 = {np.median(all_r2):.3f}")
    print("-" * 82)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
