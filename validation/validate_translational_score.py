"""
validate_translational_score.py
Validation of modules/translational_score.py — the Translational Candidate Score.

Regression tests for docs/MODULE_AUDIT_2026-09.md finding S1 1.7. Every check
asserts a scientific property; none of them would have been caught by a
shape-only test, which is why the module previously shipped with no suite.

A. DATASET INDEPENDENCE
   A compound's tier is a claim about that compound. Components were min-max
   normalised WITHIN the submitted table while _tcs_label applied ABSOLUTE
   thresholds, so a drug at CRI = 60 scored TCS 84.50 "Tier 1" or 49.50
   "Tier 3" purely according to which other compounds shared the file.
   A1. The same compound scores identically in two different companies.
   A2. Adding a better and a worse compound does not move it either.

B. COMPONENT BARS SUM TO THE SCORE
   The stacked bars used the nominal weights and filled missing components with
   zero, so a row with TCS 37.65 had bars summing to 32.00. They must now use
   the same renormalised weights as the score, and an absent component must be
   NaN (no bar), not a zero-height bar.

C. RADAR AXES RESPECT THE WEIGHTS ACTUALLY USED
   The radar divided by the hard-coded DEFAULT_WEIGHTS, so running with CRI
   weight 0.9 rendered that axis at 257.1%. No axis may exceed 100%.

D. DEGENERATE AND UNKNOWN INPUTS ARE REFUSED, NOT GUESSED
   D1. All-zero weights raise, rather than ZeroDivisionError.
   D2. Negative weights raise.
   D3. An unrecognised trajectory label scores NaN and is NAMED in the result,
       rather than being silently given the 0.2 "Undetermined" score — which
       fabricated a middling measurement out of a label the module did not
       understand.

E. THE INTERVAL THAT WAS NOT AN INTERVAL
   TCS_CI_lower/upper were score +/- 10 * std(component values): a spread over
   five quantities in different units, with n never entering, so it could not
   narrow with more replicates and had no coverage property. Those column names
   must be gone so nothing downstream can plot them as a confidence interval.

Run:  python validation/validate_translational_score.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import translational_score as TS  # noqa: E402


def _inputs(cris, traj="Sustained Rescue", rescue=50.0):
    """Build the four upstream tables for a set of compounds with given CRIs."""
    n = len(cris)
    cri = pd.DataFrame([{"Genotype": "G", "Treatment": f"D{i}", "Sex": "M", "CRI": c}
                        for i, c in enumerate(cris)])
    tsw = pd.DataFrame([{"Genotype": "G", "Treatment": f"D{i}", "Sex": "M", "TSW": 1.5}
                        for i in range(n)])
    sdq = pd.DataFrame([{"Genotype": "G", "Treatment": f"D{i}", "SDQ": 0.10}
                        for i in range(n)])
    mdr = pd.DataFrame([{"Genotype": "G", "Treatment": f"D{i}", "Sex": "M",
                         "delta_k": 0.01, "Rescue_pct": rescue, "Trajectory": traj}
                        for i in range(n)])
    return cri, tsw, sdq, mdr


def main() -> int:
    results = []

    def chk(name, got, expected, tol):
        ok = (got is not None and np.isfinite(got)
              and abs(float(got) - float(expected)) <= tol)
        results.append((name, expected,
                        round(float(got), 4) if got is not None and np.isfinite(got) else got, ok))

    def boolchk(name, ok, exp, got):
        results.append((name, exp, got, bool(ok)))

    def score_of(cris, treatment="D0", **kw):
        df = TS.compute_tcs(*_inputs(cris), **kw)
        row = df[df["Treatment"] == treatment].iloc[0]
        return df, row

    # ── A. Dataset independence ──────────────────────────────────────────────
    _, alone = score_of([60.0])
    _, wide = score_of([60.0, 10.0, 100.0])     # a much worse and a much better peer
    _, narrow = score_of([60.0, 55.0, 65.0])    # near-identical peers
    chk("A1 CRI=60 scores the same beside CRI 10/100 as alone",
        wide["TCS"], alone["TCS"], 1e-9)
    chk("A2 ... and the same beside CRI 55/65",
        narrow["TCS"], alone["TCS"], 1e-9)
    boolchk("A2 the tier label is therefore stable too",
            alone["TCS_label"] == wide["TCS_label"] == narrow["TCS_label"],
            "identical",
            f"{alone['TCS_label'][:6]} / {wide['TCS_label'][:6]} / {narrow['TCS_label'][:6]}")
    boolchk("A2 regression: NOT the old 84.50 vs 49.50 split",
            abs(float(wide["TCS"]) - float(narrow["TCS"])) < 1e-9,
            "no spread", f"{wide['TCS']} vs {narrow['TCS']}")
    # CRI_norm is the fixed-scale value: (60 - (-50)) / (150 - (-50)) = 0.55
    chk("A2 CRI=60 normalises to 0.55 on the fixed [-50,150] scale",
        float(TS._norm_cri(pd.Series([60.0])).iloc[0]), 0.55, 1e-12)

    # ── B. Component bars sum to the score ───────────────────────────────────
    bar_cols = ["TCS_CRI", "TCS_TSW", "TCS_SDQ", "TCS_MDR", "TCS_Traj"]
    _, full = score_of([60.0])
    chk("B1 bars sum to TCS (all five components present)",
        float(np.nansum([full[c] for c in bar_cols])), float(full["TCS"]), 0.02)

    # Now with three of five components missing.
    cri = pd.DataFrame([{"Genotype": "G", "Treatment": "D", "Sex": "M", "CRI": 60.0}])
    mdr = pd.DataFrame([{"Genotype": "G", "Treatment": "D", "Sex": "M",
                         "delta_k": np.nan, "Rescue_pct": np.nan,
                         "Trajectory": "Sustained Rescue"}])
    part = TS.compute_tcs(cri, pd.DataFrame(), pd.DataFrame(), mdr).iloc[0]
    chk("B2 bars sum to TCS when 3 of 5 components are missing",
        float(np.nansum([part[c] for c in bar_cols])), float(part["TCS"]), 0.02)
    boolchk("B2 a missing component is NaN, not a zero-height bar",
            pd.isna(part["TCS_TSW"]) and pd.isna(part["TCS_SDQ"]),
            "NaN", f"{part['TCS_TSW']}, {part['TCS_SDQ']}")
    boolchk("B2 components_used names what contributed",
            part["components_used"] == "CRI; Traj", "CRI; Traj", part["components_used"])
    chk("B2 n_components = 2", float(part["n_components"]), 2, 0)

    # ── C. Radar respects the weights actually used ──────────────────────────
    heavy = {"CRI": 0.9, "TSW": 0.025, "SDQ": 0.025, "MDR": 0.025, "Traj": 0.025}
    df_heavy = TS.compute_tcs(*_inputs([60.0]), weights=heavy)
    axes, data = TS.tcs_radar_data(df_heavy)
    vals = list(data.values())[0]
    boolchk("C1 no radar axis exceeds 100% under CRI weight 0.9",
            all((not np.isfinite(v)) or v <= 100.0 + 1e-6 for v in vals),
            "all <= 100", [round(v, 1) for v in vals])
    boolchk("C1 regression: the CRI axis is not the old 257.1%",
            np.isfinite(vals[0]) and vals[0] < 150.0, "< 150", round(vals[0], 1))
    # Each axis is the component's own fraction of its maximum, independent of
    # weighting: CRI_norm = 0.55 -> 55%, TSW 1.5/2 -> 75%, SDQ 1-0.1 -> 90%.
    df_def = TS.compute_tcs(*_inputs([60.0]))
    _, d_def = TS.tcs_radar_data(df_def)
    v_def = list(d_def.values())[0]
    boolchk("C2 axis values do not change when only the weights change",
            all(abs(a - b) < 1.0 for a, b in zip(vals, v_def)
                if np.isfinite(a) and np.isfinite(b)),
            "same axes", f"{[round(v,1) for v in v_def]}")
    chk("C2 CRI axis = 55% (CRI 60 on the fixed scale)", v_def[0], 55.0, 1.0)

    # ── D. Degenerate and unknown inputs ─────────────────────────────────────
    for label, w in (("D1 all-zero weights raise", {k: 0.0 for k in TS.DEFAULT_WEIGHTS}),
                     ("D2 negative weight raises", {**TS.DEFAULT_WEIGHTS, "CRI": -0.5})):
        try:
            TS.compute_tcs(*_inputs([60.0]), weights=w)
            boolchk(label, False, "ValueError", "no error raised")
        except ValueError as e:
            boolchk(label, True, "ValueError", str(e)[:28])
        except ZeroDivisionError:
            boolchk(label, False, "ValueError", "ZeroDivisionError")

    unk = TS.compute_tcs(*_inputs([60.0], traj="Wobbly Thing"))
    boolchk("D3 unrecognised trajectory scores NaN, not 0.2",
            pd.isna(unk.iloc[0]["TCS_Traj"]), "NaN", unk.iloc[0]["TCS_Traj"])
    boolchk("D3 the unrecognised label is named in the result",
            "Wobbly Thing" in unk.attrs.get("unknown_trajectory_labels", []),
            "named", unk.attrs.get("unknown_trajectory_labels"))
    boolchk("D3 a KNOWN label is still scored",
            pd.notna(TS.compute_tcs(*_inputs([60.0])).iloc[0]["TCS_Traj"]),
            "finite", TS.compute_tcs(*_inputs([60.0])).iloc[0]["TCS_Traj"])

    # ── E. The fake confidence interval is gone ──────────────────────────────
    cols = set(TS.compute_tcs(*_inputs([60.0])).columns)
    boolchk("E1 TCS_CI_lower / TCS_CI_upper no longer exist",
            not ({"TCS_CI_lower", "TCS_CI_upper"} & cols), "absent",
            sorted({"TCS_CI_lower", "TCS_CI_upper"} & cols) or "absent")
    boolchk("E1 replaced by an honestly named disagreement measure",
            "TCS_component_disagreement" in cols, "present",
            "TCS_component_disagreement" in cols)

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 80)
    print("DroPhenix translational_score.py validation (dataset-independent TCS,")
    print("summing component bars, weight-aware radar, refusal of degenerate input)")
    print("=" * 80)
    print(f"{'Check':<56}{'Expected':>12}{'Got':>8}{'Pass':>4}")
    print("-" * 80)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<56}{str(exp):>12}{str(got):>8}{'OK' if ok else 'FAIL':>4}")
    print("-" * 80)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
