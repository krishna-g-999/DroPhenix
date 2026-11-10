"""
validate_dams.py
Validation of DroPhenix modules/dams.py.

Layers:

A. ALGORITHMIC exactness of the 5-minute sleep rule on hand-built minute-level
   activity, including the exact 4-min (not sleep) vs 5-min (sleep) boundary
   (Hendricks et al., 2000; Shaw et al., 2000 — sleep = >=5 min inactivity).

B. PUBLISHED period anchors — the chi-square periodogram (Sokolove & Bushell,
   1978) must recover the foundational Konopka & Benzer (1971, PNAS) periods:
   wild-type ~24 h, per-short ~19 h, per-long ~28 h, and must NOT call a
   noise (per-0-like) series rhythmic. A Lomb-Scargle periodogram must agree.

C. CALIBRATION/structure — the DD demo reproduces the expected biology:
   WT ~24 h, disease period shortened, drug rescue; males sleep more than
   females (published sexual dimorphism).

Run:  python validation/validate_dams.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import dams as D  # noqa: E402


def _check(name, got, expected, tol, results):
    ok = (got is not None) and np.isfinite(got) and abs(got - expected) <= tol
    results.append((name, expected, round(float(got), 3) if got is not None and np.isfinite(got) else None, ok))
    return ok


def _bool_check(name, ok, expectation, got_str, results):
    results.append((name, expectation, got_str, bool(ok)))
    return bool(ok)


def _periodic(period_h, n_days=8, bin_min=30, seed=0, noise=False):
    rng = np.random.default_rng(seed)
    n = int(n_days * 1440 / bin_min)
    t = np.arange(n) * bin_min
    if noise:
        return rng.poisson(5, n).astype(float), bin_min
    rate = 3 * (1 + np.sin(2 * np.pi * t / (period_h * 60))) ** 2
    return rng.poisson(rate).astype(float), bin_min


def main() -> int:
    results = []

    # ── A. 5-min sleep rule — exact ───────────────────────────────────────────
    # 1-min bins. zeros[0:10] (10-min sleep), active at 10, zeros[11:30] (19-min,
    # contiguous) -> total 29 min sleep, 2 bouts, mean 14.5 min.
    act = np.ones(60); act[0:10] = 0; act[11:30] = 0
    df = pd.DataFrame({"Genotype": ["G"] * 60, "Treatment": ["T"] * 60, "Sex": ["M"] * 60,
                       "FlyID": ["f1"] * 60, "Time_min": np.arange(60), "Activity": act})
    spf = D.sleep_per_fly(D.load_dams(df), day_length_min=60, photoperiod_min=30).iloc[0]
    _check("Sleep total min (=29)", spf["Total_sleep_min_per_day"], 29.0, 0.0, results)
    _check("Sleep bouts (=2)", spf["N_bouts_per_day"], 2.0, 0.0, results)
    _check("Mean bout min (=14.5)", spf["Mean_bout_min"], 14.5, 0.0, results)

    # ── A2. Waking_activity is bin-width invariant (register N105) ──────────
    # `Waking_activity` is DECLARED as "counts/waking min" (dam_pipeline.py:63)
    # but was computed as the mean count per BIN, so the same fly read about b
    # times higher on a b-minute monitor -- and this laboratory records at more
    # than one bin width. Every other check in this file uses 1-minute bins,
    # where counts/bin and counts/min coincide, which is exactly why the defect
    # survived. This re-bins the SAME underlying trace to 1, 2 and 5 minutes and
    # requires the reported rate to be identical, because a fly's waking
    # activity is a property of the fly and not of the recording interval.
    #
    # This check is here, in the PUBLISHED suite, rather than in the DAM suites
    # that stay local: it is built from a synthetic trace and quotes no
    # laboratory recording, so the guard must ship with the fix.
    _n = 1440 * 2
    _zt = np.arange(_n, dtype=float) % 1440
    _act1 = np.ones(_n)
    _act1[_zt >= 13 * 60] = 0.0                 # awake ZT0-13, asleep ZT13-24

    def _waking_at(bin_min):
        k = _n // bin_min
        a = _act1[:k * bin_min].reshape(k, bin_min).sum(axis=1)
        t = np.arange(k, dtype=float) * bin_min
        d = pd.DataFrame({"Genotype": ["G"] * k, "Treatment": ["T"] * k,
                          "Sex": ["M"] * k, "FlyID": ["f1"] * k,
                          "Time_min": t, "Activity": a.astype(float)})
        return float(D.sleep_per_fly(D.load_dams(d)).iloc[0]["Waking_activity"])

    _w1 = _waking_at(1)
    _check("Waking activity = 1.000 count per waking MINUTE (1-min bins)",
           _w1, 1.0, 1e-6, results)
    _check("Waking activity is unchanged on 2-min bins (N105)",
           _waking_at(2), _w1, 1e-6, results)
    _check("Waking activity is unchanged on 5-min bins (N105)",
           _waking_at(5), _w1, 1e-6, results)

    # Boundary: 4-min zero run is NOT sleep; 5-min zero run IS sleep.
    a4 = np.ones(20); a4[5:9] = 0                      # 4 zeros
    df4 = pd.DataFrame({"Genotype": ["G"] * 20, "Treatment": ["T"] * 20, "Sex": ["M"] * 20,
                        "FlyID": ["f"] * 20, "Time_min": np.arange(20), "Activity": a4})
    s4 = D.sleep_per_fly(D.load_dams(df4), day_length_min=20, photoperiod_min=10).iloc[0]
    _check("4-min run -> 0 sleep", s4["Total_sleep_min_per_day"], 0.0, 0.0, results)
    a5 = np.ones(20); a5[5:10] = 0                     # 5 zeros
    df5 = df4.copy(); df5["Activity"] = a5
    s5 = D.sleep_per_fly(D.load_dams(df5), day_length_min=20, photoperiod_min=10).iloc[0]
    _check("5-min run -> 5 sleep", s5["Total_sleep_min_per_day"], 5.0, 0.0, results)

    # ── B. Period recovery vs Konopka & Benzer ───────────────────────────────
    for truep in (24.0, 19.0, 28.0):
        x, b = _periodic(truep)
        cp = D.chi_square_periodogram(x, b, 14, 34)
        ls = D.lomb_scargle_period(x, b, 14, 34)
        _check(f"Chi-sq period (={truep}h)", cp["peak_period_h"], truep, 0.5, results)
        _check(f"Lomb-Scargle period (={truep}h)", ls, truep, 0.5, results)
        _bool_check(f"Rhythmic flag @ {truep}h", cp["rhythmic"], "True", str(cp["rhythmic"]), results)

    # per-0-like noise: sensitivity/false-positive control over 40 series
    detected = sum(D.chi_square_periodogram(*_periodic(24, seed=s), 14, 34)["rhythmic"] for s in range(40))
    fp = sum(D.chi_square_periodogram(*_periodic(24, seed=s, noise=True), 14, 34)["rhythmic"] for s in range(40))
    _bool_check("True 24h detected >=38/40", detected >= 38, ">=38/40", f"{detected}/40", results)
    _bool_check("Noise false-positive <=2/40 (<=5%)", fp <= 2, "<=2/40", f"{fp}/40", results)

    # ── C. Demo structure ─────────────────────────────────────────────────────
    res = D.run_dams_analysis(D.demo_dams())
    ps = res["period_summary"]
    wt = ps[ps.Genotype == "Oregon-WT"]["Mean_period_h"].mean()
    tdv = ps[(ps.Genotype == "TDP43") & (ps.Treatment == "Vehicle")]["Mean_period_h"].mean()
    tdd = ps[(ps.Genotype == "TDP43") & (ps.Treatment == "Drug")]["Mean_period_h"].mean()
    _check("Demo WT period (~24)", wt, 24.0, 0.4, results)
    _bool_check("Demo disease period shortened (<WT)", tdv < wt - 0.5, "TDP43veh<WT", f"{tdv:.2f}<{wt:.2f}", results)
    _bool_check("Demo drug rescues period (veh<drug<=WT)", tdv < tdd <= wt + 0.2, "veh<drug<=WT", f"{tdv:.2f}<{tdd:.2f}", results)
    ss = res["sleep_summary"]
    male = ss[ss.Sex == "Male"]["Total_sleep_min_per_day"].mean()
    female = ss[ss.Sex == "Female"]["Total_sleep_min_per_day"].mean()
    _bool_check("Demo males sleep > females (dimorphism)", male > female, "M>F", f"{male:.0f}>{female:.0f}", results)

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 84)
    print("DroPhenix dams.py validation (5-min rule + Sokolove-Bushell + Konopka-Benzer)")
    print("=" * 84)
    print(f"{'Check':<42}{'Expected':>16}{'Engine':>16}{'Pass':>7}")
    print("-" * 84)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<42}{str(exp):>16}{str(got):>16}{'OK' if ok else 'FAIL':>7}")
    print("-" * 84)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
