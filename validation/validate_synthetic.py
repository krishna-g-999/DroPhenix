"""
validate_synthetic.py
Sanity/coherence validation of the unified multi-assay synthetic cohort
(modules/synthetic.py). It does NOT re-test the analysis maths (the other
suites do that against published data); it confirms the generated cohort has:
  * consistent Genotype x Treatment x Sex labels across all four assays, and
  * the intended disease-selective biology on every assay:
      TDP43 (vehicle) impaired vs Oregon-WT, and the Drug partially rescues
      TDP43 toward wild type.

Run:  python validation/validate_synthetic.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import synthetic as SYN          # noqa: E402
from modules import survival as SV             # noqa: E402
from modules import ymaze as YM                # noqa: E402
from modules import dams as DM                 # noqa: E402
from modules.data_parser import load_data      # noqa: E402
from modules.normalizer import compute_all_metrics  # noqa: E402


def main() -> int:
    results = []

    def chk(name, ok, detail):
        results.append((name, str(detail), bool(ok)))
        return bool(ok)

    data = SYN.generate_all()

    # ── cohort label consistency ──────────────────────────────────────────────
    labelsets = {k: (tuple(sorted(v["Genotype"].unique())),
                     tuple(sorted(v["Treatment"].unique())))
                 for k, v in data.items()}
    ref = ("Oregon-WT", "TDP43"), ("Drug", "Vehicle")
    consistent = all(ls == ref for ls in labelsets.values())
    chk("Consistent cohort across assays", consistent, labelsets["climbing"])

    # ── climbing: WT PI > disease; drug rescues ──────────────────────────────
    df, _, e = load_data(data["climbing"])
    pi = compute_all_metrics(df)["pi"].set_index(["Genotype", "Treatment", "Sex"])["PI_mean"]
    wt = pi.xs("Oregon-WT", level=0).mean()
    tdv = pi.loc[("TDP43", "Vehicle")].mean()
    tdd = pi.loc[("TDP43", "Drug")].mean()
    chk("Climbing WT > TDP43-vehicle", wt > tdv, f"{wt:.1f} > {tdv:.1f}")
    chk("Climbing drug rescues TDP43", tdd > tdv, f"{tdd:.1f} > {tdv:.1f}")

    # ── lifespan: WT median > disease; drug extends ──────────────────────────
    med = SV.median_survival_table(SV.to_event_table(data["lifespan"]))
    med["g"] = med["Genotype"] + "|" + med["Treatment"]
    ms = med.groupby("g")["Median_Survival"].mean()
    chk("Lifespan WT > TDP43-vehicle",
        ms["Oregon-WT|Vehicle"] > ms["TDP43|Vehicle"],
        f"{ms['Oregon-WT|Vehicle']:.0f} > {ms['TDP43|Vehicle']:.0f}")
    chk("Lifespan drug extends TDP43",
        ms["TDP43|Drug"] > ms["TDP43|Vehicle"],
        f"{ms['TDP43|Drug']:.0f} > {ms['TDP43|Vehicle']:.0f}")

    # ── Y-maze: WT alternation > disease (working-memory deficit) ────────────
    pf = YM.per_fly_metrics(YM.load_ymaze(data["ymaze"]))
    pf["g"] = pf["Genotype"] + "|" + pf["Treatment"]
    alt = pf.groupby("g")["alternation_pct"].mean()
    chk("Y-maze WT alternation > TDP43-vehicle",
        alt["Oregon-WT|Vehicle"] > alt["TDP43|Vehicle"],
        f"{alt['Oregon-WT|Vehicle']:.1f} > {alt['TDP43|Vehicle']:.1f}")
    chk("Y-maze drug improves TDP43 alternation",
        alt["TDP43|Drug"] > alt["TDP43|Vehicle"],
        f"{alt['TDP43|Drug']:.1f} > {alt['TDP43|Vehicle']:.1f}")

    # ── DAMS: WT ~24 h; disease period shortened; drug rescue ────────────────
    ppf = DM.period_per_fly(DM.load_dams(data["dams"]))
    ppf["g"] = ppf["Genotype"] + "|" + ppf["Treatment"]
    per = ppf.groupby("g")["Period_h_chisq"].mean()
    chk("DAMS WT period ~24 h", abs(per["Oregon-WT|Vehicle"] - 24) < 0.5,
        f"{per['Oregon-WT|Vehicle']:.2f}")
    chk("DAMS TDP43-vehicle period shortened", per["TDP43|Vehicle"] < per["Oregon-WT|Vehicle"] - 0.5,
        f"{per['TDP43|Vehicle']:.2f} < {per['Oregon-WT|Vehicle']:.2f}")
    chk("DAMS drug rescues period", per["TDP43|Vehicle"] < per["TDP43|Drug"],
        f"{per['TDP43|Vehicle']:.2f} < {per['TDP43|Drug']:.2f}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 74)
    print("DroPhenix synthetic.py unified-cohort coherence validation")
    print("=" * 74)
    print(f"{'Check':<46}{'Detail':>20}{'Pass':>7}")
    print("-" * 74)
    all_ok = True
    for name, detail, ok in results:
        all_ok &= ok
        print(f"{name:<46}{detail:>20}{'OK' if ok else 'FAIL':>7}")
    print("-" * 74)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
