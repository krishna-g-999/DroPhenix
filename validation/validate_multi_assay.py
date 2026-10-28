"""
validate_multi_assay.py
Validation of modules/multi_assay.py — the Multi-Assay Composite Motor Score.

These checks assert SCIENTIFIC properties, not return shapes. Each of A1-A3 is
a regression test for a defect that was live in the 2026-08 module and that the
shape-only tests elsewhere in this suite could not have caught
(see docs/MODULE_AUDIT_2026-09.md, findings 1.4-1.6).

A. MEDIAN SURVIVAL
   A1. A cohort in which fewer than half the flies die has NO median survival.
       Cross-checked against lifelines (which returns inf) when installed, and
       against DroPhenix's own survival.py Kaplan-Meier median (NaN).
       The old code returned the last scored day (50) as if it were the median.
   A2. A cohort in which more than half die has the exact KM median.

B. WEIGHT RENORMALISATION
   B1. A group that is best-possible on every axis scores 1.000 whether the lab
       ran four assays or only climbing. Previously: 1.000 vs 0.400 "Weak".
   B2. Weights are renormalised PER ROW: two groups in one table with different
       assay coverage are each scored over their own present assays.
   B3. A group with no computable component scores NaN, not 0.

C. NORMALISATION
   C1. A zero-range column is neutral (0.5), not the worst score. Two groups
       both at PI 95 must not both be labelled "No rescue".
   C2. PI_norm is absolute: the same PI gives the same normalised value in two
       different datasets. This is what makes the tier labels meaningful.
   C3. Dataset-relative components are FLAGGED, and supplying an anchor via
       `refs=` removes the flag and fixes the scale.

D. NO IMPUTATION
   D1. A missing assay value is never replaced by the column mean. A group's
       score must not move when an UNRELATED group's value for a missing assay
       changes.

E. NAMING
   E1. process_dams reports Activity_CV (a coefficient of variation), and the
       value equals SD/mean of the pooled counts exactly.

Run:  python validation/validate_multi_assay.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import multi_assay as MA          # noqa: E402
from modules import survival as SV             # noqa: E402


def _grp(geno, pi):
    return {"Genotype": geno, "Treatment": "T", "Sex": "M", "PI_mean": pi}


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got) and np.isfinite(expected)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 4) if got is not None and np.isfinite(got) else got,
                        ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    # ── A. Median survival ───────────────────────────────────────────────────
    # 100 flies, 40 dead by the last scored day (50). The median is NOT reached.
    under = pd.DataFrame([
        {"Genotype": "WT", "Treatment": "V", "Sex": "M",
         "Day": d, "Dead_count": k, "Total_start": 100}
        for d, k in [(10, 5), (20, 5), (30, 10), (40, 10), (50, 10)]])
    ls_under = MA.process_lifespan(under)
    med_under = ls_under["Median_Survival_days"].iloc[0]
    boolchk("A1 <50% dead -> median NOT reached (NaN)",
            pd.isna(med_under), "NaN", str(med_under))
    boolchk("A1 Median_Reached flag is False",
            bool(ls_under["Median_Reached"].iloc[0]) is False, False,
            bool(ls_under["Median_Reached"].iloc[0]))
    boolchk("A1 regression: NOT the last scored day (50)",
            not (pd.notna(med_under) and float(med_under) == 50.0),
            "!= 50", str(med_under))
    boolchk("A1 cohort size preserved (n=100, 40 events)",
            int(ls_under["n"].iloc[0]) == 100 and int(ls_under["Events"].iloc[0]) == 40,
            "n=100, ev=40",
            f"n={int(ls_under['n'].iloc[0])}, ev={int(ls_under['Events'].iloc[0])}")

    # Independent cross-check against lifelines, when available.
    try:
        from lifelines import KaplanMeierFitter
        T, E = [], []
        for d, k in [(10, 5), (20, 5), (30, 10), (40, 10), (50, 10)]:
            T += [d] * k; E += [1] * k
        T += [50] * 60; E += [0] * 60
        ll_med = KaplanMeierFitter().fit(T, E).median_survival_time_
        boolchk("A1 cross-check vs lifelines (inf = not reached)",
                not np.isfinite(ll_med), "inf", str(ll_med))
    except ImportError:
        boolchk("A1 cross-check vs lifelines (SKIPPED, not installed)", True, "skip", "skip")

    # DroPhenix's own KM median must agree with what multi_assay now reports.
    km_med = SV.median_survival_table(SV.to_event_table(under))["Median_Survival"].iloc[0]
    boolchk("A1 agrees with survival.py KM median",
            pd.isna(km_med) and pd.isna(med_under), "both NaN",
            f"{km_med} / {med_under}")

    # More than half die: the median IS reached, at the exact KM step.
    over = pd.DataFrame([
        {"Genotype": "WT", "Treatment": "V", "Sex": "M",
         "Day": d, "Dead_count": k, "Total_start": 100}
        for d, k in [(10, 10), (20, 20), (30, 30), (40, 30)]])
    med_over = MA.process_lifespan(over)["Median_Survival_days"].iloc[0]
    chk("A2 >50% dead -> KM median = 30", med_over, 30.0, 1e-9)

    # ── B. Weight renormalisation ────────────────────────────────────────────
    best_pi  = pd.DataFrame([_grp("A", 100.0)])
    best_sur = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M",
                              "Median_Survival_days": 80.0}])
    best_fl  = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M",
                              "Flight_Index": 100.0}])
    best_ac  = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M",
                              "Daily_Activity_mean": 1000.0}])
    refs = {"Median_Survival_days": 80.0, "Daily_Activity_mean": 1000.0}

    all4 = MA.compute_mcms(best_pi, best_sur, best_fl, best_ac, refs=refs).iloc[0]
    only1 = MA.compute_mcms(best_pi).iloc[0]
    chk("B1 best-possible, 4 assays -> MCMS 1.0", all4["MCMS"], 1.0, 1e-9)
    chk("B1 best-possible, climbing only -> MCMS 1.0", only1["MCMS"], 1.0, 1e-9)
    boolchk("B1 regression: climbing-only is NOT 0.400 'Weak'",
            abs(float(only1["MCMS"]) - 0.4) > 1e-9, "!= 0.400",
            f"{only1['MCMS']} ({only1['MCMS_Tier']})")

    # Per-row renormalisation: A has survival, B does not. Both are perfect on
    # the assays they have, so both must score 1.0.
    pi2 = pd.DataFrame([_grp("A", 100.0), _grp("B", 100.0)])
    sur_a = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M",
                           "Median_Survival_days": 80.0}])
    mixed = MA.compute_mcms(pi2, sur_a, refs=refs).set_index("Genotype")
    chk("B2 row with 2 assays -> 1.0", mixed.loc["A", "MCMS"], 1.0, 1e-9)
    chk("B2 row with 1 assay  -> 1.0", mixed.loc["B", "MCMS"], 1.0, 1e-9)
    boolchk("B2 Assays_used is per row",
            mixed.loc["A", "Assays_used"] != mixed.loc["B", "Assays_used"],
            "differ", f"{mixed.loc['A','Assays_used']!r} vs {mixed.loc['B','Assays_used']!r}")
    boolchk("B2 MCMS_n_assays counts contributors",
            int(mixed.loc["A", "MCMS_n_assays"]) == 2 and int(mixed.loc["B", "MCMS_n_assays"]) == 1,
            "2 and 1",
            f"{int(mixed.loc['A','MCMS_n_assays'])} and {int(mixed.loc['B','MCMS_n_assays'])}")

    # A group with nothing computable scores NaN, not 0.
    none_df = MA.compute_mcms(pd.DataFrame([_grp("Z", np.nan)])).iloc[0]
    boolchk("B3 no computable component -> MCMS NaN (not 0)",
            pd.isna(none_df["MCMS"]), "NaN", str(none_df["MCMS"]))
    boolchk("B3 tier says not computable",
            "Not computable" in str(none_df["MCMS_Tier"]),
            "Not computable", none_df["MCMS_Tier"])

    # ── C. Normalisation ─────────────────────────────────────────────────────
    tie = MA.compute_mcms(pd.DataFrame([_grp("A", 95.0), _grp("B", 95.0)]))
    chk("C1 two groups both at PI 95 -> PI_norm 0.975",
        tie["PI_norm"].iloc[0], 0.975, 1e-9)
    boolchk("C1 regression: neither is labelled 'No rescue'",
            not any("No rescue" in t for t in tie["MCMS_Tier"]),
            "no 'No rescue'", "; ".join(tie["MCMS_Tier"]))

    # Absolute scale: the same PI normalises identically in a different dataset.
    ctx1 = MA.compute_mcms(pd.DataFrame([_grp("A", 50.0), _grp("B", 100.0)]))
    ctx2 = MA.compute_mcms(pd.DataFrame([_grp("A", 50.0), _grp("B", -100.0)]))
    a1 = float(ctx1.set_index("Genotype").loc["A", "PI_norm"])
    a2 = float(ctx2.set_index("Genotype").loc["A", "PI_norm"])
    chk("C2 PI_norm is dataset-independent", a2, a1, 1e-12)
    chk("C2 PI_norm(50) = 0.75 on the absolute [-100,100] scale", a1, 0.75, 1e-12)

    # Relative components are flagged; an anchor removes the flag.
    rel = MA.compute_mcms(pi2, pd.DataFrame([
        {"Genotype": "A", "Treatment": "T", "Sex": "M", "Median_Survival_days": 80.0},
        {"Genotype": "B", "Treatment": "T", "Sex": "M", "Median_Survival_days": 40.0}]))
    boolchk("C3 un-anchored survival flags Dataset_relative",
            bool(rel["Dataset_relative"].all()), True, bool(rel["Dataset_relative"].all()))
    boolchk("C3 tier label carries the caveat",
            all("dataset-relative" in t for t in rel["MCMS_Tier"]),
            "suffix present", rel["MCMS_Tier"].iloc[0])
    anch = MA.compute_mcms(pi2, pd.DataFrame([
        {"Genotype": "A", "Treatment": "T", "Sex": "M", "Median_Survival_days": 80.0},
        {"Genotype": "B", "Treatment": "T", "Sex": "M", "Median_Survival_days": 40.0}]),
        refs={"Median_Survival_days": 80.0})
    boolchk("C3 anchored survival is NOT dataset-relative",
            not bool(anch["Dataset_relative"].any()), False,
            bool(anch["Dataset_relative"].any()))
    chk("C3 anchored: 40 days of an 80-day anchor -> 0.5",
        float(anch.set_index("Genotype").loc["B", "Survival_norm"]), 0.5, 1e-12)

    # ── D. No mean imputation ────────────────────────────────────────────────
    # Group B has no flight measurement. Changing group A's flight value must
    # not move group B's score (under mean-imputation it would).
    fl_lo = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M", "Flight_Index": 10.0}])
    fl_hi = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M", "Flight_Index": 90.0}])
    b_lo = float(MA.compute_mcms(pi2, flight_df=fl_lo).set_index("Genotype").loc["B", "MCMS"])
    b_hi = float(MA.compute_mcms(pi2, flight_df=fl_hi).set_index("Genotype").loc["B", "MCMS"])
    chk("D1 unrelated group's value does not move this group's MCMS", b_hi, b_lo, 1e-12)
    boolchk("D1 missing component stays NaN (not imputed)",
            pd.isna(MA.compute_mcms(pi2, flight_df=fl_lo)
                    .set_index("Genotype").loc["B", "Flight_norm"]),
            "NaN", str(MA.compute_mcms(pi2, flight_df=fl_lo)
                       .set_index("Genotype").loc["B", "Flight_norm"]))

    # ── E. Naming and exactness of the DAMS read-out ─────────────────────────
    counts = [10.0, 20.0, 30.0, 40.0]
    dams = pd.DataFrame([{"Genotype": "A", "Treatment": "T", "Sex": "M",
                          "Day": i + 1, "Monitor_ID": 1, "Activity_counts": c}
                         for i, c in enumerate(counts)])
    dm = MA.process_dams(dams).iloc[0]
    expected_cv = float(np.std(counts, ddof=1) / np.mean(counts))
    boolchk("E1 column is named Activity_CV",
            "Activity_CV" in MA.process_dams(dams).columns, "Activity_CV",
            ", ".join(MA.process_dams(dams).columns))
    boolchk("E1 misnomer 'Activity_Fragmentation_Idx' is gone",
            "Activity_Fragmentation_Idx" not in MA.process_dams(dams).columns,
            "absent", "absent" if "Activity_Fragmentation_Idx" not in
            MA.process_dams(dams).columns else "still present")
    chk("E1 Activity_CV = SD/mean exactly", dm["Activity_CV"], round(expected_cv, 3), 1e-9)

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 78)
    print("DroPhenix multi_assay.py validation (MCMS: KM median, weight renormalisation,")
    print("absolute normalisation, no imputation) — regression tests for audit S1 1.4-1.6")
    print("=" * 78)
    print(f"{'Check':<52}{'Expected':>12}{'Got':>10}{'Pass':>4}")
    print("-" * 78)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<52}{str(exp):>12}{str(got):>10}{'OK' if ok else 'FAIL':>4}")
    print("-" * 78)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
