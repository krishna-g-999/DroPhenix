"""
validate_real_lifespan_benchmark.py
survival.py on a REAL published Drosophila lifespan experiment.

WHY THIS SUITE EXISTS
---------------------
`validate_survival.py` benchmarks the engine against Freireich et al. (1963) —
real, published, and the standard textbook survival dataset, but a HUMAN
leukaemia trial with 42 patients. Nothing tested survival.py on an actual
Drosophila lifespan assay at the scale a fly lab works at, nor on both of the
input layouts `to_event_table` accepts.

DATA (real, published, CC0)
    Brengdahl, Kimber, Maguire-Baxter & Friberg, via
    "Heritability of lifespan is largely sex-limited in Drosophila"
    Am Nat (2013), doi:10.1086/673296
    Data: doi:10.5061/dryad.q412q (openly mirrored at Zenodo record 4996358)
    Local copy: reference_datasets/lifespan_heritability/lifespan.xlsx

    50 hemiclonal lines x 2 sexes x 2 social environments x replicate vials.
      sheet "Lifespan of individual flies"      39,581 individual death ages
      sheet "# dead flies per vial and day"     34,400 daily death counts
    The SAME experiment recorded both ways, which is what makes check B possible.

A. Scale and parsing: the event table is built from the individual-fly layout.
B. LAYOUT EQUIVALENCE: the daily-count layout of the same experiment must give
   the same survival curves. This is the check that `to_event_table`'s two
   branches agree, and it cannot be done with synthetic data alone.
C. Kaplan-Meier medians against `lifelines` across all 200 real groups, with
   the ONE documented convention difference isolated and its mechanism pinned.
D. Log-rank on 39,581 flies, including the regression test for a p-value that
   used to print as exactly 0.0.

NOT ESTABLISHED HERE
    The paper's headline result is that the GENETIC correlation between male and
    female lifespan is low, estimated from a quantitative-genetic model of the
    hemiclonal design. This suite does NOT reproduce that: a raw correlation of
    hemiclone means is a different quantity (it measures 0.44, which is neither
    the paper's estimate nor evidence about it). Only the survival machinery is
    validated here.

Run:  python validation/validate_real_lifespan_benchmark.py   (exit 0 = passed)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import survival as SV   # noqa: E402

XLSX = (Path(__file__).resolve().parent.parent / "reference_datasets"
        / "lifespan_heritability" / "lifespan.xlsx")

_REN = {"Hemiclone": "Genotype",
        "Sex (1=female; 2=male)": "Sex",
        "Environment (1=same sex; 2=opposite sex)": "Treatment"}


def _label(d: pd.DataFrame) -> pd.DataFrame:
    d = d.rename(columns=_REN).copy()
    d["Sex"] = d["Sex"].map({1: "Female", 2: "Male"})
    d["Treatment"] = d["Treatment"].map({1: "SameSex", 2: "OppositeSex"})
    d["Genotype"] = "HC" + d["Genotype"].astype(str)
    return d


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
        print("Real Drosophila lifespan benchmark")
        print("=" * 80)
        print(f"[SKIP] reference data not present at {XLSX}")
        print('       curl -sL -o lifespan.xlsx "https://zenodo.org/api/records/'
              '4996358/files/Lifespan%20data.xlsx/content"')
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    # ── A. Individual-fly layout ─────────────────────────────────────────────
    ind = _label(pd.read_excel(XLSX, "Lifespan of individual flies"))
    ind = ind.rename(columns={"Life span": "Time"})
    ind["Event"] = 1
    evA = SV.to_event_table(ind[["Genotype", "Treatment", "Sex", "Time", "Event"]])

    chk("A: real flies in the experiment", float(len(evA)), 39581, 0)
    chk("A: hemiclonal lines", float(ind["Genotype"].nunique()), 50, 0)
    chk("A: groups (line x sex x environment)", float(evA["Group"].nunique()), 200, 0)
    boolchk("A: every fly is an observed death (no censoring in this design)",
            int(evA["Event"].sum()) == len(evA), "all events",
            f"{int(evA['Event'].sum())}/{len(evA)}")

    medA = SV.median_survival_table(evA)
    boolchk("A: median survival is in a plausible fly range (20-90 d)",
            20 <= medA["Median_Survival"].min() and medA["Median_Survival"].max() <= 90,
            "20-90 d",
            f"{medA['Median_Survival'].min():.0f}-{medA['Median_Survival'].max():.0f}")

    # ── B. The same experiment, daily-count layout ───────────────────────────
    cnt = _label(pd.read_excel(XLSX, "# dead flies per vial and day"))
    cnt = cnt.rename(columns={"# dead": "Dead_count"})
    cnt["Total_start"] = cnt.groupby(["Genotype", "Treatment", "Sex"])["Dead_count"].transform("sum")
    evB = SV.to_event_table(cnt[["Genotype", "Treatment", "Sex", "Day", "Dead_count", "Total_start"]])

    boolchk("B: daily-count layout recovers the same cohort size (+-0.5%)",
            abs(len(evB) - len(evA)) / len(evA) < 0.005,
            "within 0.5%", f"{len(evB)} vs {len(evA)}")
    boolchk("B: both layouts produce the same 200 groups",
            set(evB["Group"]) == set(evA["Group"]), "identical",
            "identical" if set(evB["Group"]) == set(evA["Group"]) else "differ")

    medB = SV.median_survival_table(evB).set_index("Group")["Median_Survival"]
    join = pd.concat([medA.set_index("Group")["Median_Survival"].rename("A"),
                      medB.rename("B")], axis=1).dropna()
    dmax = float((join["A"] - join["B"]).abs().max())
    dmean = float((join["A"] - join["B"]).abs().mean())
    # Deaths are scored every 2 days, so the count layout can only place a
    # death on a scoring day; one interval is the finest possible disagreement.
    boolchk("B: medians agree within one 2-day scoring interval, all 200 groups",
            dmax <= 2.0, "<= 2 d", f"max {dmax:.1f} d, mean {dmean:.2f} d")
    boolchk("B: most groups agree exactly",
            float((join["A"] == join["B"]).mean()) > 0.9, "> 90% exact",
            f"{100 * (join['A'] == join['B']).mean():.1f}%")

    # ── C. Kaplan-Meier against lifelines, on all 200 real groups ────────────
    try:
        from lifelines import KaplanMeierFitter
        diffs = []
        for g in medA["Group"]:
            s = evA[evA["Group"] == g]
            ll = KaplanMeierFitter().fit(s["Time"], s["Event"]).median_survival_time_
            dp = float(medA.loc[medA["Group"] == g, "Median_Survival"].iloc[0])
            if np.isfinite(ll) and abs(ll - dp) > 1e-9:
                diffs.append((g, dp, ll))
        boolchk("C: KM median matches lifelines on >=97% of 200 real groups",
                len(diffs) <= 6, "<= 6 of 200", f"{len(diffs)} of 200")

        # Every disagreement must be the ONE documented convention difference:
        # DroPhenix takes the smallest t with S(t) <= 0.5, lifelines uses < 0.5,
        # so they can only differ where the curve lands EXACTLY on 0.5.
        from statsmodels.duration.survfunc import SurvfuncRight
        all_exact_tie = True
        for g, dp, _ll in diffs:
            s = evA[evA["Group"] == g]
            sf = SurvfuncRight(s["Time"].to_numpy(float), s["Event"].to_numpy(int))
            S = np.asarray(sf.surv_prob)
            t = np.asarray(sf.surv_times)
            i = np.where(S <= 0.5 + 1e-12)[0]
            all_exact_tie &= (len(i) > 0 and abs(S[i[0]] - 0.5) < 1e-9
                              and abs(t[i[0]] - dp) < 1e-9)
        boolchk("C: every disagreement is an EXACT S(t)=0.5 tie (the documented "
                "convention)", all_exact_tie, "all ties",
                "all ties" if all_exact_tie else "NOT all ties")
    except ImportError:
        boolchk("C: lifelines cross-check (SKIPPED, not installed)", True, "skip", "skip")

    # ── D. Log-rank on 39,581 real flies ─────────────────────────────────────
    ev_sex = evA.copy()
    ev_sex["Group"] = ev_sex["Sex"]
    lr = SV.logrank_overall(ev_sex)
    f_med = medA.loc[medA["Sex"] == "Female", "Median_Survival"].mean()
    m_med = medA.loc[medA["Sex"] == "Male", "Median_Survival"].mean()
    boolchk("D: females outlive males (the standard Drosophila pattern)",
            f_med > m_med, "F > M", f"{f_med:.1f} d > {m_med:.1f} d")
    boolchk("D: log-rank detects it", lr["Sig"] == "***", "***", lr["Sig"])
    # Regression: this p used to be round(p, 6) and printed as exactly 0.0.
    boolchk("D: an underflowing p is NOT reported as 0.0",
            lr["p_value"] > 0, "> 0", lr["p_value"])
    boolchk("D: chi-square is finite and large", np.isfinite(lr["chisq"]),
            "finite", lr["chisq"])

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 84)
    print("Real Drosophila lifespan benchmark — Am Nat (2013) doi:10.1086/673296")
    print("39,581 individual flies, 50 hemiclonal lines, 200 groups, both input layouts")
    print("=" * 84)
    print(f"{'Check':<58}{'Expected':>13}{'Got':>9}{'Pass':>4}")
    print("-" * 84)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<58}{str(exp):>13}{str(got):>9}{'OK' if ok else 'FAIL':>4}")
    print("-" * 84)
    print(f"  median lifespan: Female {f_med:.1f} d, Male {m_med:.1f} d; "
          f"log-rank chisq {lr['chisq']:.0f}, p = {lr['p_value']:g}")
    print("-" * 84)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
