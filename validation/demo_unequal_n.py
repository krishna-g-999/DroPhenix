"""
demo_unequal_n.py
Demonstration and statistical justification of DroPhenix's CORE INNOVATION:
comparing a compound's effect across genetic backgrounds (strains) that have
UNEQUAL numbers of flies/vials, using effect-size + actuarial (inverse-variance)
meta-analytic pooling.

Why this is correct (the science)
---------------------------------
When strain i contributes n_i flies, the drug-vs-vehicle effect is estimated as
Hedges' g_i with sampling variance v_i that DEPENDS on n_i (small strains ->
large v_i). Naively averaging the g_i (equal weights) ignores this and is a
high-variance, statistically inefficient estimator. The minimum-variance
unbiased linear combination weights each strain by its precision w_i = 1/v_i
(the Gauss-Markov / actuarial credibility principle):
        g_pooled = Σ w_i g_i / Σ w_i ,   Var = 1/Σ w_i .
Between-strain heterogeneity is added via DerSimonian-Laird random effects
(τ²), and I² reports the fraction of variance due to true strain differences.

This script (a) shows a worked unequal-N example with the per-strain weights,
and (b) proves by simulation that inverse-variance pooling has lower error and
correct 95% CI coverage versus naive averaging when N is unequal.

Run:  python validation/demo_unequal_n.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.drug_ranking import hedges_g, inverse_variance_pool  # noqa: E402


# ── A worked example: one compound, 3 disease strains, deliberately unequal N ──
# (mirrors the classic screen problem: e.g. TDP43 = 3 vials, HTT-Q93 = 7 vials)
def worked_example(seed=1):
    rng = np.random.default_rng(seed)
    # true standardized drug benefit per strain (SD units); strains differ a bit
    strains = {
        "TDP43":   dict(n=3, true=0.9),   # small N
        "HTT-Q93": dict(n=7, true=0.7),   # large N
        "FUS":     dict(n=5, true=0.8),
        "SOD1":    dict(n=4, true=1.0),
    }
    rows = []
    for name, sp in strains.items():
        n = sp["n"]
        veh = rng.normal(0.0, 1.0, n)
        drug = rng.normal(sp["true"], 1.0, n)     # mean shift = true g (SD=1)
        e = hedges_g(veh, drug)
        rows.append({"Strain": name, "n_vehicle": n, "n_drug": n,
                     "g": round(e["g"], 3), "var": round(e["var"], 4),
                     "se": round(e["se"], 3), "weight_pct": np.nan})
    df = pd.DataFrame(rows)
    pooled = inverse_variance_pool(df["g"].values, df["var"].values)
    w = 1.0 / df["var"].values
    df["weight_pct"] = (100 * w / w.sum()).round(1)
    naive = float(np.mean(df["g"]))
    return df, pooled, naive


# ── Simulation: efficiency & coverage, naive mean vs inverse-variance pool ─────
def simulation(true_g=0.7, ns=(3, 7, 5, 4, 9), reps=2000, seed=7):
    rng = np.random.default_rng(seed)
    naive_est, iv_est = [], []
    iv_cov = 0
    dl_cov = 0
    w_hksj = w_dl = 0.0
    for _ in range(reps):
        gs, vs = [], []
        for n in ns:
            veh = rng.normal(0, 1, n)
            drug = rng.normal(true_g, 1, n)
            e = hedges_g(veh, drug)
            if np.isfinite(e["g"]):
                gs.append(e["g"]); vs.append(e["var"])
        if len(gs) < 2:
            continue
        gs = np.array(gs); vs = np.array(vs)
        naive_est.append(gs.mean())
        p = inverse_variance_pool(gs, vs)
        iv_est.append(p["fixed"])
        # Register N35. `ci_lo`/`ci_hi` are now the Hartung-Knapp-Sidik-Jonkman
        # interval, not the 1.96-based DerSimonian-Laird one, so both are
        # tracked: this demo's own purpose is to show that the pooling is
        # well calibrated, and after the change that claim depends on which
        # interval is meant. The DL figures stay available as `ci_lo_dl` /
        # `ci_hi_dl`.
        if p["ci_lo"] <= true_g <= p["ci_hi"]:
            iv_cov += 1
        if p["ci_lo_dl"] <= true_g <= p["ci_hi_dl"]:
            dl_cov += 1
        w_hksj += p["ci_hi"] - p["ci_lo"]
        w_dl += p["ci_hi_dl"] - p["ci_lo_dl"]
    naive_est = np.array(naive_est); iv_est = np.array(iv_est)
    rmse_naive = float(np.sqrt(np.mean((naive_est - true_g) ** 2)))
    rmse_iv = float(np.sqrt(np.mean((iv_est - true_g) ** 2)))
    n = len(iv_est)
    return {"rmse_naive": rmse_naive, "rmse_iv": rmse_iv,
            "var_naive": float(np.var(naive_est)), "var_iv": float(np.var(iv_est)),
            "iv_coverage": iv_cov / n, "dl_coverage": dl_cov / n,
            "width_hksj": w_hksj / n, "width_dl": w_dl / n,
            "reps": n,
            "efficiency_gain_pct": round(100 * (1 - rmse_iv / rmse_naive), 1)}


def main():
    import warnings; warnings.filterwarnings("ignore")
    df, pooled, naive = worked_example()
    print("=" * 72)
    print("UNEQUAL-N CROSS-STRAIN COMPOUND COMPARISON  (worked example)")
    print("=" * 72)
    print("One compound vs vehicle in 4 disease strains of UNEQUAL size:")
    print(df.to_string(index=False))
    print("-" * 72)
    print(f"Naive (equal-weight) mean of g          : {naive:.3f}")
    print(f"Inverse-variance pooled (fixed)         : {pooled['fixed']:.3f}  "
          f"(95% CI {pooled['ci_lo']:.3f}, {pooled['ci_hi']:.3f})")
    print(f"DerSimonian-Laird random-effects pooled : {pooled['random']:.3f}")
    print(f"Heterogeneity: Q={pooled['Q']:.3f}, tau^2={pooled['tau2']:.4f}, "
          f"I^2={pooled['I2']:.1f}%")
    print("  -> larger strains carry more weight (weight_pct column); the small "
          "n=3 strain is down-weighted, not allowed to dominate.")

    print()
    sim = simulation()
    print("=" * 72)
    print(f"SIMULATION ({sim['reps']} runs; true g=0.7; strain sizes 3,7,5,4,9)")
    print("=" * 72)
    print(f"RMSE  naive equal-weight mean : {sim['rmse_naive']:.4f}")
    print(f"RMSE  inverse-variance pool   : {sim['rmse_iv']:.4f}  "
          f"({sim['efficiency_gain_pct']}% lower error)")
    print(f"Sampling variance  naive/IV   : {sim['var_naive']:.4f} / {sim['var_iv']:.4f}")
    print(f"95% CI coverage, DerSimonian-Laird : {100*sim['dl_coverage']:.1f}%  "
          f"(mean width {sim['width_dl']:.3f})")
    print(f"95% CI coverage, HKSJ (reported)   : {100*sim['iv_coverage']:.1f}%  "
          f"(mean width {sim['width_hksj']:.3f}, "
          f"{sim['width_hksj']/sim['width_dl']:.2f}x)")
    print("-" * 72)
    print("  Both numbers are from the same 2000 runs. This simulation has "
          "tau^2 = 0 by")
    print("  construction -- one true effect in every strain -- which is the "
          "case where")
    print("  HKSJ is at its most conservative: the t(k-1) quantile replaces "
          "1.96 while the")
    print("  variance is barely larger, so the interval widens and coverage "
          "rises above")
    print("  nominal. It is the price of the other direction. Measured over "
          "2000 runs per")
    print("  cell (register N35, Block C): DerSimonian-Laird covers 80.4% at "
          "k=2 and 85.1%")
    print("  at k=3 once tau^2 = 0.30, while HKSJ stays at or above nominal "
          "everywhere in")
    print("  the range tested. A therapeutic claim is damaged more by an "
          "interval that is")
    print("  too narrow than by one that is too wide, so HKSJ is what is "
          "reported and the")
    print("  DL figures are kept beside it.")
    # The claim this demo makes is about EFFICIENCY of the point estimate and
    # about the interval not UNDER-covering. Over-coverage at tau^2 = 0 is the
    # documented cost of HKSJ at small k, not a calibration failure, so the
    # gate is one-sided for HKSJ and two-sided for DL.
    ok = (sim["rmse_iv"] < sim["rmse_naive"]
          and sim["iv_coverage"] >= 0.95
          and 0.90 <= sim["dl_coverage"] <= 0.99)
    print("-" * 72)
    print("RESULT:", "INNOVATION VALIDATED — inverse-variance pooling is more "
          "efficient under unequal N, and the reported HKSJ interval does not "
          "under-cover" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
