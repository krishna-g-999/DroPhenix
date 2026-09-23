"""Does the QC layer discriminate, or is it decoration?

The audit's complaint about the old `confidence` flag was that it fired
everywhere and so carried no information. The new codes must do better on the
only test that matters: a reading the layer flags should be MEASURABLY WORSE
than one it does not.

Measured on Day 15, which has a hand count beside every reading. The codes are
applied to the stored columns exactly as `assess_reading` would apply them.
Note the ESTIMATOR_NOT_VALIDATED code is excluded here -- it fires on every
reading by design, because it is a statement about the instrument and not about
the reading, and including it would make the test vacuous.
"""
import sys
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from modules import climbing_band as CB

OUT = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/_DroPhenix_analysis")
d = pd.read_csv(OUT / "phenotype_bias_readings.csv")
d["static_est"] = d.count_upper - d.count_lower
d["abs_err"] = (d.reported_count - d.hand_count).abs()

rows = []
for _i, r in d.iterrows():
    a = CB.assess_reading(
        r.reported_count, 20, n_source="MANUAL", n_quality="VERIFIED",
        bracket_width=r.bracket_width, static_removed=r.static_est,
        count_upper=r.count_upper, accounted_for=r.accounted_for,
        validated_domain="test")     # exclude the instrument-level code
    rows.append(";".join(a.reasons))
d["codes"] = rows

print(f"n = {len(d)} readings with a hand count\n")
print(f"{'code':<26}{'fires on':>10}{'MAE when flagged':>19}"
      f"{'MAE when clean':>17}{'Welch p':>10}")
any_code = False
for code in ("BRACKET_TOO_WIDE", "CONSERVATION_VIOLATED",
             "STATIC_ARTIFACT_EXCESS"):
    hit = d.codes.str.contains(code)
    if hit.sum() == 0 or (~hit).sum() == 0:
        print(f"{code:<26}{100 * hit.mean():>9.0f}%   never varies -- no test")
        continue
    a, b = d.abs_err[hit], d.abs_err[~hit]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"{code:<26}{100 * hit.mean():>9.0f}%{a.mean():>19.2f}"
          f"{b.mean():>17.2f}{p:>10.3f}")
    any_code = True

print()
clean = d[d.codes == ""]
flagged = d[d.codes != ""]
print(f"readings with NO code at all: {len(clean)} ({100*len(clean)/len(d):.0f}%)"
      f"   MAE {clean.abs_err.mean():.2f}" if len(clean) else
      "readings with NO code at all: 0 -- THE LAYER FIRES ON EVERYTHING")
if len(clean) and len(flagged):
    print(f"readings with at least one code: {len(flagged)} "
          f"({100*len(flagged)/len(d):.0f}%)   MAE {flagged.abs_err.mean():.2f}")
    t, p = stats.ttest_ind(flagged.abs_err, clean.abs_err, equal_var=False)
    print(f"\nWelch t-test, flagged vs clean: p = {p:.4f}")
    print(f"difference in MAE: {flagged.abs_err.mean() - clean.abs_err.mean():+.2f} flies")

print("\nfor comparison, the OLD flag this replaces:")
if "confidence" in d.columns:
    for c, q in d.groupby("confidence"):
        print(f"  confidence={c:<10}{len(q):>5} readings   MAE {q.abs_err.mean():.2f}")
else:
    dd = d.assign(conf=np.where(d.bracket_width <= 3, "HIGH",
                        np.where(d.bracket_width <= 7, "MODERATE", "LOW")))
    for c, q in dd.groupby("conf"):
        print(f"  confidence={c:<10}{len(q):>5} readings   MAE {q.abs_err.mean():.2f}")
