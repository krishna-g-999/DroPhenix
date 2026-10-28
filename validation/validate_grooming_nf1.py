"""
validate_grooming_nf1.py
grooming.py against a REAL published grooming experiment.

WHY THIS SUITE EXISTS
---------------------
`validate_grooming.py` checks exact arithmetic on hand-made bouts and calibrates
a synthetic demo to the ~13 % wild-type figure of Qiao et al. Neither uses real
scored data. In particular, the interval-merging fix of 2026-09-06 (audit S2
finding 2.9) was justified only by a constructed example: three deliberately
overlapping bouts producing 183 % of the observation window. This suite asks
whether that fix does anything on genuine ethograms — and it does.

DATA (real, published, CC0)
    Bonnay F et al. / King LB et al. (2020)
    "Developmental loss of neurofibromin across distributed neuronal circuits
     drives excessive grooming in Drosophila"
    PLOS Genetics 16:e1008920, doi:10.1371/journal.pgen.1008920
    Data: doi:10.5061/dryad.bvq83bk67 (openly mirrored at Zenodo 3964901)
    Local: reference_datasets/grooming_nf1/

      Fig_1C  head-intact grooming, PER-FLY SUMMARY layout, 108 flies
      Fig_1F  headless grooming, BOUT-LEVEL layout, 1,158 scored bouts,
              51 flies x 3 five-minute segments, body part annotated

    Both of grooming.py's input layouts are therefore exercised on real data.

PUBLISHED GROUND TRUTH USED HERE
    Fig 1C: Nf1 mutants groom significantly longer than wCS10 controls,
    Wilcoxon rank-sum p < 0.01. (Mann-Whitney U and Wilcoxon rank-sum are the
    same test, and it is what auto_test now selects a priori.)

A DISCREPANCY I AM REPORTING RATHER THAN RESOLVING
    In Fig_1F the two files use OPPOSITE genotype codings, each annotated in a
    spare column of its own sheet:
        Fig_1C: "genotype 1 = wCS10", "genotype 2 = Nf1P1"
        Fig_1F: "Genotype 1 = nf1P1", "Genotype 2 = wCS10"
    Under Fig_1F's own annotation, nf1P1 grooms LESS than wCS10 in every one of
    the three segments and in essentially every body part (total per fly:
    229/125/119 vs 350/462/445). That is the opposite of the direction the paper
    reports. The two genotype columns in that file (`genotype` and
    `genotype_wcs10First`) are exact relabellings of each other, so the
    ambiguity cannot be resolved from the file, and picking whichever mapping
    agrees with the paper would make the check circular and worthless.

    So Fig_1F is used ONLY for MECHANICS, where the genotype label is
    irrelevant. The directional claim rests on Fig_1C alone. This is flagged in
    docs/REFERENCE_DATASETS.md as a question for the data's authors.

Run:  python validation/validate_grooming_nf1.py   (exit 0 = all passed)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import grooming as G       # noqa: E402
from modules import stats_engine as SE  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "reference_datasets" / "grooming_nf1"
F1C = DATA / "fig1C_head_intact.xlsx"
F1F = DATA / "Fig_1F_-_Headless_grooming.xlsx"
SEGMENT_UNITS = 2250.0     # each 5-min segment spans 0-2250 in the file's own units


def main() -> int:
    warnings.filterwarnings("ignore")
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 4) if got is not None and np.isfinite(got) else got, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    if not F1C.exists() or not F1F.exists():
        print("=" * 82)
        print("Grooming benchmark (King et al. 2020, PLOS Genetics)")
        print("=" * 82)
        print(f"[SKIP] reference data not present in {DATA}")
        print("       Zenodo record 3964901, files 'Fig_1C_-_Head-intact_grooming.xlsx'")
        print("       and 'Fig_1F_-_Headless_grooming.xlsx'")
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    # ── A. Summary layout + the published direction and test ────────────────
    c = pd.read_excel(F1C)
    # Read the coding from the file's own annotation, never assume it.
    coding = {int(k.split()[-1]): v
              for k, v in (x.split(" = ") for x in c["Unnamed: 8"].dropna())}
    boolchk("A: genotype coding read from the file, not assumed",
            coding == {1: "wCS10", 2: "Nf1P1"}, "{1:wCS10, 2:Nf1P1}", coding)

    c["Genotype"] = c["genotype"].map(coding)
    c = c.rename(columns={"duration": "Total_grooming_s", "subject": "FlyID"})
    c["Treatment"] = "head-intact"; c["Sex"] = "Pooled"; c["Observation_s"] = np.nan
    pf_c = G.per_fly_metrics(c[["Genotype", "Treatment", "Sex", "FlyID",
                                "Total_grooming_s", "Observation_s"]])
    chk("A: flies parsed from the summary layout", float(len(pf_c)), 108, 0)
    boolchk("A: balanced design, 54 per genotype",
            set(pf_c.groupby("Genotype").size()) == {54}, "54 / 54",
            pf_c.groupby("Genotype").size().to_dict())

    nf1 = pf_c.loc[pf_c["Genotype"] == "Nf1P1", "Total_grooming_s"].to_numpy()
    wcs = pf_c.loc[pf_c["Genotype"] == "wCS10", "Total_grooming_s"].to_numpy()
    boolchk("A: PUBLISHED DIRECTION — Nf1 mutants groom MORE than wCS10",
            nf1.mean() > wcs.mean(), "Nf1 > wCS10",
            f"{nf1.mean():.0f} vs {wcs.mean():.0f}")

    res = SE.auto_test({"Nf1P1": nf1, "wCS10": wcs})
    boolchk("A: the test used is the rank test the paper used (Wilcoxon/MWU)",
            res["method"] == "Mann-Whitney U", "Mann-Whitney U", res["method"])
    boolchk("A: chosen a priori, not from the data",
            res["test_choice"] == "a priori (rank)", "a priori (rank)",
            res["test_choice"])
    boolchk("A: PUBLISHED STATISTIC — p < 0.01 as reported",
            res["main_result"]["p_value"] < 0.01, "p < 0.01",
            res["main_result"]["p_value"])
    boolchk("A: this design is not underpowered by construction",
            not res["main_result"]["underpowered_by_design"], False,
            res["main_result"]["underpowered_by_design"])
    rbr = float(res["pairwise"]["Rank_biserial_r"].iloc[0])
    boolchk("A: rank-biserial sign agrees with the direction (2026-09-06 fix)",
            rbr > 0, "positive", round(rbr, 3))

    # ── B. Bout-level layout on real scored ethograms ───────────────────────
    f = pd.read_excel(F1F)
    b = f.dropna(subset=["start", "stop"]).copy()
    b = b.rename(columns={"start": "Bout_start_s", "stop": "Bout_end_s"})
    # Genotype deliberately NOT used here — see the module docstring.
    b["Genotype"] = "pooled"; b["Treatment"] = "headless"; b["Sex"] = "Pooled"
    b["FlyID"] = b["subject"].astype(str) + "_" + b["video"].astype(str)
    b["Observation_s"] = SEGMENT_UNITS
    pf_f = G.per_fly_metrics(b[["Genotype", "Treatment", "Sex", "FlyID",
                                "Bout_start_s", "Bout_end_s", "Observation_s"]])

    chk("B: real scored bouts in the file", float(len(b)), 1096, 0)
    chk("B: fly-segments analysed", float(len(pf_f)), 91, 0)
    boolchk("B: the file's own `length` equals stop - start",
            bool(np.allclose(b["length"], b["Bout_end_s"] - b["Bout_start_s"])),
            "exact", "exact")

    # THE POINT OF THIS SUITE: the interval-merging fix fires on real data.
    merged = pf_f[pf_f["N_bouts_scored"] > pf_f["N_bouts"]]
    boolchk("B: overlapping scored bouts EXIST in real ethograms",
            len(merged) > 0, "> 0 fly-segments", len(merged))
    naive = b.groupby("FlyID")["length"].sum()
    got = pf_f.set_index("FlyID")["Total_grooming_s"]
    delta = (naive.reindex(got.index) - got).abs()
    boolchk("B: merging changes ONLY the overlapping fly-segments",
            int((delta > 1e-9).sum()) == len(merged),
            f"{len(merged)} changed", int((delta > 1e-9).sum()))
    boolchk("B: without merging, those totals would be inflated",
            float(delta.max()) > 0, "> 0 units", round(float(delta.max()), 1))
    boolchk("B: every merged fly-segment carries a QC_flag",
            bool((merged["QC_flag"].str.len() > 0).all()), "all flagged",
            f"{int((merged['QC_flag'].str.len() > 0).sum())}/{len(merged)}")

    # Physical plausibility: grooming cannot exceed the observation window.
    boolchk("B: no fly grooms for more than 100 % of the window",
            int((pf_f["Pct_time_grooming"] > 100.0).sum()) == 0, 0,
            int((pf_f["Pct_time_grooming"] > 100.0).sum()))
    boolchk("B: percent-time grooming is in a plausible range",
            0 < float(pf_f["Pct_time_grooming"].median()) < 60,
            "0-60 %", round(float(pf_f["Pct_time_grooming"].median()), 1))
    boolchk("B: mean bout duration is positive and finite",
            bool(pf_f["Mean_bout_s"].gt(0).all()), "all > 0",
            round(float(pf_f["Mean_bout_s"].median()), 1))

    # ── C. The reported discrepancy is real and stays visible ───────────────
    coding_f = {int(k.split()[-1]): v
                for k, v in (x.split(" = ") for x in f["Unnamed: 11"].dropna())}
    boolchk("C: Fig_1F uses the OPPOSITE genotype coding to Fig_1C",
            coding_f.get(1, "").lower() == "nf1p1" and coding.get(1) == "wCS10",
            "1F:1=nf1P1, 1C:1=wCS10", f"1F {coding_f}")
    # Under 1F's own annotation the direction is inverted; assert that we
    # measured it, so nobody later "fixes" it by silently flipping the label.
    f2 = f.copy(); f2["geno"] = f2["genotype"].map(coding_f)
    tot = (f.dropna(subset=["start", "stop"])
             .groupby(["subject", "video"])["length"].sum().reset_index())
    subj = f2[["subject", "geno"]].drop_duplicates()
    grid = subj.merge(pd.DataFrame({"video": f["video"].dropna().unique()}), how="cross")
    gg = grid.merge(tot, on=["subject", "video"], how="left").fillna({"length": 0})
    means = gg.groupby("geno")["length"].mean()
    boolchk("C: under 1F's own coding the direction is INVERTED (documented, "
            "not silently corrected)",
            means.get("nf1P1", 0) < means.get("wCS10", 0), "nf1P1 < wCS10",
            f"{means.get('nf1P1', float('nan')):.0f} vs {means.get('wCS10', float('nan')):.0f}")

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 82)
    print("Grooming benchmark — King et al. (2020) PLOS Genetics 16:e1008920")
    print("108 flies (summary layout) + 1,096 scored bouts (bout-level layout)")
    print("=" * 82)
    print(f"{'Check':<58}{'Expected':>12}{'Got':>8}{'Pass':>4}")
    print("-" * 82)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<58}{str(exp):>12}{str(got):>8}{'OK' if ok else 'FAIL':>4}")
    print("-" * 82)
    print(f"  Fig 1C: Nf1P1 {nf1.mean():.0f} vs wCS10 {wcs.mean():.0f}, "
          f"Mann-Whitney p = {res['main_result']['p_value']:g} (paper: p < 0.01)")
    print(f"  Fig 1F: {len(merged)} of {len(pf_f)} fly-segments had overlapping bouts; "
          f"merging removed up to {delta.max():.0f} units of double-counted time")
    print("-" * 82)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
