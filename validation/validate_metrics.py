"""
validate_metrics.py
Validation of the DroPhenix novel/kinetic metrics that drive drug-effect
interpretation: MDR (motor decline rate), SDQ (sex dimorphism), TSW
(therapeutic selectivity window) and CRI (composite rescue index).

These are validated against ANALYTIC ground truth (closed-form values that the
metric must recover) and, for CRI, against the realistic synthetic cohort with
a known disease-selective drug effect. Multiple independent sample sets (seeds)
are used for the stochastic MDR check. This regenerates — with real,
reproducible numbers — the validation table the README used to claim.

Run:  python validation/validate_metrics.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.motor_decline import fit_decay          # noqa: E402
from modules.metrics import compute_sdq, compute_tsw  # noqa: E402


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = got is not None and np.isfinite(got) and abs(got - expected) <= tol
        results.append((name, expected, round(float(got), 4) if got is not None and np.isfinite(got) else None, ok))

    def ge(name, got, thresh):
        ok = got is not None and np.isfinite(got) and got >= thresh
        results.append((name, f">={thresh}", round(float(got), 4) if got is not None and np.isfinite(got) else None, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    # ── MDR: recover the known decay constant k across many groups/seeds ──────
    times = np.array([0, 5, 10, 15, 20, 25, 30], float)   # e.g. age (days)
    r_vals, r2_meds = [], []
    for seed in range(5):                                  # 5 independent sample sets
        rng = np.random.default_rng(seed)
        true_k, fit_k, r2s = [], [], []
        for _ in range(40):
            PI0 = rng.uniform(60, 95)
            k = rng.uniform(0.01, 0.15)
            p = PI0 * np.exp(-k * times) + rng.normal(0, 3.0, len(times))
            res = fit_decay(times, p)
            if res["fit_success"] and np.isfinite(res["k"]):
                true_k.append(k); fit_k.append(res["k"]); r2s.append(res["R2"])
        r = np.corrcoef(true_k, fit_k)[0, 1]
        r_vals.append(r); r2_meds.append(np.median(r2s))
    ge("MDR k recovery Pearson r (mean of 5 sets)", float(np.mean(r_vals)), 0.95)
    ge("MDR fit median R^2 (mean of 5 sets)", float(np.mean(r2_meds)), 0.85)

    # ── MDR over AGE (the scientifically correct decline axis) ────────────────
    # Climbing PI declines with AGE (days), not within-assay seconds. The
    # pipeline must fit PI vs Age and recover: disease declines faster than WT,
    # and the drug slows the disease decline; seconds-only data must be refused.
    from modules import synthetic as SYN
    from modules.data_parser import load_data as _ld
    from modules.normalizer import compute_all_metrics as _cam
    from modules.motor_decline import endpoint_pi_by_age, compute_mdr_table
    dfa, _, _ = _ld(SYN.generate_climbing_aging())
    age_pi = endpoint_pi_by_age(dfa)
    boolchk("MDR endpoint-PI table spans >=4 ages",
            age_pi["Age"].nunique() >= 4, ">=4 ages", int(age_pi["Age"].nunique()))
    mdr = compute_mdr_table(age_pi, time_col="Age")
    ok_fit = mdr[mdr.fit_success]
    km = ok_fit.groupby(["Genotype", "Treatment"])["k"].mean()
    kw = km.get(("Oregon-WT", "Vehicle"), np.nan)
    kv = km.get(("TDP43", "Vehicle"), np.nan)
    kd = km.get(("TDP43", "Drug"), np.nan)
    boolchk("MDR: TDP43 declines faster than WT (k)", kv > kw, "TDP43>WT", f"{kv:.3f}>{kw:.3f}")
    boolchk("MDR: drug slows TDP43 decline (k drug<veh)", kd < kv, "drug<veh", f"{kd:.3f}<{kv:.3f}")
    ge("MDR-age median R^2", float(ok_fit["R2"].median()), 0.80)
    # seconds-only input must be refused (not mis-fit)
    aggsec = _cam(_ld(SYN.generate_climbing())[0])["aggregated"].rename(columns={"Pct_mean": "PI_mean"})
    refused = compute_mdr_table(aggsec)   # default time_col='Age' -> no Age present
    boolchk("MDR refuses seconds-only data (requires age)",
            refused.empty and "requires_age_data" in str(refused.attrs.get("mdr_status", "")),
            "refused", "empty+flag" if refused.empty else "fit anyway")

    # ── SDQ: exact analytic value ─────────────────────────────────────────────
    # PI_Male=80, PI_Female=40 -> SDQ = |80-40|/max(80,40) = 0.5
    pi_sdq = pd.DataFrame({"Genotype": ["G", "G"], "Treatment": ["Vehicle", "Vehicle"],
                           "Sex": ["Male", "Female"], "PI_mean": [80.0, 40.0]})
    sdq = compute_sdq(pi_sdq)
    chk("SDQ exact (=0.5)", float(sdq["SDQ"].iloc[0]), 0.5, 1e-9)

    # ── TSW: exact analytic value ─────────────────────────────────────────────
    # WT: Veh 70 -> Drug 72 (dWT=+2). Disease: Veh -30 -> Drug 10 (dDis=+40).
    # TSW = dDis / |dWT| = 40 / 2 = 20
    pi_tsw = pd.DataFrame({
        "Genotype": ["Oregon-WT", "Oregon-WT", "TDP43", "TDP43"],
        "Treatment": ["Vehicle", "Drug", "Vehicle", "Drug"],
        "Sex": ["Male"] * 4, "PI_mean": [70.0, 72.0, -30.0, 10.0]})
    tsw = compute_tsw(pi_tsw, wt_genotype="Oregon-WT", vehicle_treatment="Vehicle")
    trow = tsw[tsw["Treatment"] == "Drug"].iloc[0]
    chk("TSW exact (=20)", float(trow["TSW"]), 20.0, 1e-6)

    # ── CRI: ordinal correctness on the known-efficacy screen (A > C) ─────────
    # CRI is computed for treated groups vs the vehicle baseline (rescue index);
    # a stronger compound must yield a higher CRI in the disease model.
    from modules import synthetic as SYN
    from modules.data_parser import load_data
    from modules.normalizer import compute_all_metrics
    from modules.metrics import compute_novel_metrics
    screen = SYN.generate_screen()
    df, _, _ = load_data(screen["climbing"])
    base = compute_all_metrics(df)
    nov = compute_novel_metrics(df, base, wt_genotype="Oregon-WT", vehicle_treatment="Vehicle")
    cri = nov["cri"]
    cri_g = cri.groupby(["Genotype", "Treatment"])["CRI"].mean()
    cri_A = cri_g.get(("TDP43", "Drug_A"), np.nan)
    cri_C = cri_g.get(("TDP43", "Drug_C"), np.nan)
    boolchk("CRI ordinal: strong Drug_A > weak Drug_C (TDP43)",
            np.isfinite(cri_A) and np.isfinite(cri_C) and cri_A > cri_C,
            "A>C", f"{cri_A:.1f}>{cri_C:.1f}")
    boolchk("CRI: strong drug shows substantial rescue (>20)",
            np.isfinite(cri_A) and cri_A > 20, ">20", round(float(cri_A), 1))

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 76)
    print("DroPhenix novel-metric validation (MDR / SDQ / TSW / CRI)")
    print("=" * 76)
    print(f"{'Check':<44}{'Expected':>14}{'Got':>12}{'Pass':>6}")
    print("-" * 76)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<44}{str(exp):>14}{str(got):>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 76)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
