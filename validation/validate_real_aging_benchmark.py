"""
validate_real_aging_benchmark.py
The trajectory / MDR / survival stack tested on REAL published experimental data.

WHY THIS SUITE EXISTS
---------------------
Until now every check on the age-decline model was SIMULATION from a known
truth. A simulation establishes that an estimator works under its own
assumptions; it cannot show the model is right for real flies. This suite runs
the shipped pipeline on a published longitudinal experiment and asks whether it
recovers the finding the authors reported.

DATA (real, published, openly licensed)
    Shahrestani P, Wilson JB, Mueller LD, Rose MR (2016)
    "Patterns of physiological decline due to age and selection in
     Drosophila melanogaster". Evolution 70:2550-2561. doi:10.1111/evo.13065
    Data: doi:10.5061/dryad.qb509 (mirrored, openly downloadable, at
          Zenodo record 4954719)
    Local copy: reference_datasets/physio_decline/physio_decline.xlsx

    Ten outbred populations: ACO1-5 selected for rapid development, CO1-5
    controls. Sheets used here:
      `geotaxis`  405 rows -- proportion of flies passing the line, per
                  population x sex x age, ages 1-70 d
      `mortality` 586 rows -- daily survivors and deaths per population x sex,
                  ages 2-90 d, cohorts of ~850-5000 flies

PUBLISHED BIOLOGICAL GROUND TRUTH (what the paper reports)
    Populations selected for rapid development (ACO) age faster: they show
    steeper physiological decline and shorter lifespan than the CO controls.
    That direction is the ground truth these checks are scored against.

MAPPING ONTO DroPhenix
    The geotaxis score is the proportion of flies above the line -- the same
    quantity as Pct -- so PI = 2*p*100 - 100 puts it on DroPhenix's scale.
    No other transformation is applied.

A. MDR recovers the published direction from real geotaxis-vs-age.
B. The result survives the unequal-follow-up confound (ACO is only observed to
   age 45, CO to 70), tested on the common window.
C. Kaplan-Meier on the real mortality data reproduces the lifespan difference
   and agrees with lifelines fly-for-fly.
D. MSCI on 10 real populations recovers the negative decline-lifespan coupling,
   and at n = 10 is NOT underpowered, so both branches of the 2026-09-06
   permutation fix are exercised on real data.

HONEST LIMITS
    - D is an ECOLOGICAL correlation across populations. It cannot separate a
      within-population motor-survival coupling from the selection regime that
      created both. The module reports that caveat on every row; this suite
      checks that it does.
    - This validates the DECLINE MODEL, not the video pipeline: the geotaxis
      scores are the authors' own, not produced by DroPhenix's detector.

Run:  python validation/validate_real_aging_benchmark.py   (exit 0 = passed)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import motor_decline as MD   # noqa: E402
from modules import stats_engine as SE    # noqa: E402
from modules import survival as SV        # noqa: E402

XLSX = (Path(__file__).resolve().parent.parent / "reference_datasets"
        / "physio_decline" / "physio_decline.xlsx")


def _geotaxis(cap_age=None) -> pd.DataFrame:
    g = pd.read_excel(XLSX, "geotaxis")
    for c in ("pop", "sex"):
        g[c] = g[c].astype(str).str.strip()      # the file has stray whitespace
    g["PI_mean"] = 2.0 * g["geotaxis"] * 100.0 - 100.0
    g = g.rename(columns={"pop": "Genotype", "age": "Age", "sex": "Sex"})
    g["Treatment"] = "None"
    if cap_age is not None:
        g = g[g["Age"] <= cap_age]
    return g[["Genotype", "Treatment", "Sex", "Age", "PI_mean"]]


def _mortality_events() -> pd.DataFrame:
    m = pd.read_excel(XLSX, "mortality")
    for c in ("pop", "sex"):
        m[c] = m[c].astype(str).str.strip()
    m["Genotype"] = m["pop"] + m["rep"].astype(str)
    m["Treatment"] = "None"
    m["Sex"] = m["sex"].str.upper()
    lf = m.rename(columns={"age": "Day", "deaths": "Dead_count"})
    lf["Total_start"] = lf.groupby(["Genotype", "Sex"])["surv"].transform("max")
    return SV.to_event_table(
        lf[["Genotype", "Treatment", "Sex", "Day", "Dead_count", "Total_start"]])


def main() -> int:
    warnings.filterwarnings("ignore")
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 5) if got is not None and np.isfinite(got) else got, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    if not XLSX.exists():
        print("=" * 80)
        print("Real-data aging benchmark (Shahrestani et al. 2016)")
        print("=" * 80)
        print(f"[SKIP] reference data not present at {XLSX}")
        print("       Fetch with:")
        print('       curl -sL -o physio_decline.xlsx "https://zenodo.org/api/'
              'records/4954719/files/physio%20decline%20data.xlsx/content"')
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    # ── Provenance: the file really is the published dataset ─────────────────
    g_all = _geotaxis()
    ev = _mortality_events()
    chk("real data: geotaxis rows", float(len(g_all)), 405, 0)
    chk("real data: populations", float(g_all["Genotype"].nunique()), 10, 0)
    boolchk("real data: 5 ACO + 5 CO populations",
            sorted(g_all["Genotype"].unique()) ==
            [f"ACO{i}" for i in range(1, 6)] + [f"CO{i}" for i in range(1, 6)],
            "ACO1-5, CO1-5", sorted(g_all["Genotype"].unique())[:3] + ["..."])
    boolchk("real data: mortality cohorts are large (>800 flies)",
            ev.groupby("Group").size().min() > 800, "> 800",
            int(ev.groupby("Group").size().min()))

    # ── A. MDR recovers the published direction ──────────────────────────────
    def k_by_regime(cap):
        t = MD.compute_mdr_table(_geotaxis(cap))
        t["reg"] = t["Genotype"].str[:-1]
        return (t.loc[t["reg"] == "ACO", "k"].dropna(),
                t.loc[t["reg"] == "CO", "k"].dropna(), t)

    k_aco, k_co, tab = k_by_regime(None)
    chk("A: MDR fitted every population x sex (20 fits)", float(len(tab)), 20, 0)
    boolchk("A: every fit converged", bool(tab["fit_success"].all()), True,
            bool(tab["fit_success"].all()))
    boolchk("A: k is reported even though AICc selected the LINEAR model",
            bool(tab["k"].notna().all()) and (tab["fit_model"] == "linear").all(),
            "k present, model=linear",
            f"k n/a={int(tab['k'].isna().sum())}, models={set(tab['fit_model'])}")
    boolchk("A: ACO declines FASTER than CO (published direction)",
            k_aco.mean() > k_co.mean(), "k_ACO > k_CO",
            f"{k_aco.mean():.5f} > {k_co.mean():.5f}")
    r = SE.auto_test({"ACO": k_aco.to_numpy(), "CO": k_co.to_numpy()})
    boolchk("A: a priori rank test finds it significant",
            r["main_result"]["p_value"] < 0.01, "p < 0.01",
            r["main_result"]["p_value"])
    boolchk("A: the test used was the rank test, chosen a priori",
            r["method"] == "Mann-Whitney U" and r["test_choice"] == "a priori (rank)",
            "MWU a priori", f"{r['method']} / {r['test_choice']}")
    boolchk("A: n = 10 per regime is NOT underpowered by design",
            not r["main_result"]["underpowered_by_design"], False,
            r["main_result"]["underpowered_by_design"])

    # ── B. The unequal-follow-up confound ────────────────────────────────────
    # ACO is only observed to age 45, CO to 70. Refit on the common window.
    k_aco45, k_co45, _ = k_by_regime(45)
    boolchk("B: direction holds on the COMMON age window (<= 45 d)",
            k_aco45.mean() > k_co45.mean(), "k_ACO > k_CO",
            f"{k_aco45.mean():.5f} > {k_co45.mean():.5f}")
    ratio_full = k_aco.mean() / k_co.mean()
    ratio_cap = k_aco45.mean() / k_co45.mean()
    boolchk("B: restricting follow-up does not create the effect (ratio grows)",
            ratio_cap >= ratio_full, "ratio_capped >= ratio_full",
            f"{ratio_cap:.2f}x vs {ratio_full:.2f}x")

    # ── C. Survival on the real mortality data ───────────────────────────────
    med = SV.median_survival_table(ev)
    med["reg"] = med["Genotype"].str[:-1]
    m_aco = med.loc[med["reg"] == "ACO", "Median_Survival"]
    m_co = med.loc[med["reg"] == "CO", "Median_Survival"]
    boolchk("C: ACO is shorter-lived than CO (published direction)",
            m_aco.mean() < m_co.mean(), "ACO < CO",
            f"{m_aco.mean():.1f} d < {m_co.mean():.1f} d")
    rs = SE.auto_test({"ACO": m_aco.to_numpy(), "CO": m_co.to_numpy()})
    boolchk("C: rank test finds the lifespan difference",
            rs["main_result"]["p_value"] < 0.01, "p < 0.01",
            rs["main_result"]["p_value"])

    try:
        from lifelines import KaplanMeierFitter
        worst = 0.0
        for grp in list(ev["Group"].unique())[:6]:
            sub = ev[ev["Group"] == grp]
            ll = KaplanMeierFitter().fit(sub["Time"], sub["Event"]).median_survival_time_
            dp = float(med.loc[med["Group"] == grp, "Median_Survival"].iloc[0])
            if np.isfinite(ll):
                worst = max(worst, abs(ll - dp))
        chk("C: KM median == lifelines on 6 real cohorts", worst, 0.0, 1e-9)
    except ImportError:
        boolchk("C: lifelines cross-check (SKIPPED, not installed)", True, "skip", "skip")

    # ── D. MSCI on 10 real populations ───────────────────────────────────────
    mdr2 = tab.copy()
    mdr2["Treatment"] = mdr2["Genotype"]; mdr2["Genotype"] = "all_populations"
    med2 = med.rename(columns={"Median_Survival": "Median_Survival_days"}).copy()
    med2["Treatment"] = med2["Genotype"]; med2["Genotype"] = "all_populations"
    msci, merged = MD.compute_msci(mdr2, med2, min_groups=5)

    boolchk("D: MSCI computed for both sexes", len(msci) == 2, 2, len(msci))
    for _, row in msci.iterrows():
        boolchk(f"D: {row['Sex']} coupling is NEGATIVE (faster decline, shorter life)",
                row["MSCI_Spearman_rho"] < 0, "rho < 0", row["MSCI_Spearman_rho"])
        boolchk(f"D: {row['Sex']} rho is strong (|rho| > 0.7)",
                abs(row["MSCI_Spearman_rho"]) > 0.7, "|rho| > 0.7",
                abs(row["MSCI_Spearman_rho"]))
        boolchk(f"D: {row['Sex']} permutation p < 0.05",
                row["p_Spearman"] < 0.05, "p < 0.05", row["p_Spearman"])
        boolchk(f"D: {row['Sex']} n=10 is NOT underpowered by construction",
                "Underpowered" not in row["Interpretation"], "real verdict",
                row["Interpretation"][:26])
        boolchk(f"D: {row['Sex']} carries the ecological-correlation caveat",
                "survivor_bias" in str(row["Caveat"]), "caveat present",
                "present" if "survivor_bias" in str(row["Caveat"]) else "MISSING")
    chk("D: paired (k, median) points used", float(len(merged.dropna(
        subset=["k", "Median_Survival_days"]))), 20, 0)

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 84)
    print("Real-data benchmark: Shahrestani, Wilson, Mueller & Rose (2016) Evolution 70:2550")
    print("10 Drosophila populations, geotaxis + mortality across age, cohorts of ~850-5000")
    print("=" * 84)
    print(f"{'Check':<58}{'Expected':>14}{'Got':>8}{'Pass':>4}")
    print("-" * 84)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<58}{str(exp):>14}{str(got):>8}{'OK' if ok else 'FAIL':>4}")
    print("-" * 84)
    print(f"  k_ACO = {k_aco.mean():.5f}/day   k_CO = {k_co.mean():.5f}/day   "
          f"({ratio_full:.2f}x, {ratio_cap:.2f}x on the common window)")
    print(f"  median lifespan  ACO = {m_aco.mean():.1f} d   CO = {m_co.mean():.1f} d")
    print("-" * 84)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
