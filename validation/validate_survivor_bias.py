"""
Validation suite for modules/survivor_bias.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
Every test here uses a GENERATIVE MODEL WITH A KNOWN TRUE SLOPE, so each
estimator is scored against the quantity it is supposed to recover rather than
against another estimator.

A. NULL CASE. With frailty coupling switched off there is no survivor bias, and
   every estimator -- cohort mean, balanced panel, mixed model, pattern mixture
   -- must recover the true individual slope. An estimator that "corrects" a
   bias that is not there is as wrong as one that misses a bias that is.

B. THE CENTRAL CLAIM. With frailty coupling switched on (frail flies both
   decline faster and die sooner), the cohort-mean slope must be BIASED TOWARD
   ZERO, while the mixed model and the pattern-mixture estimate recover the
   truth. The magnitude of the bias is reported, because that magnitude is the
   scientific point: it is how wrong the conventional analysis is.

C. THE DIAGNOSTIC MUST NOT CRY WOLF. `frailty_association_test` must report no
   informative dropout in the null case and detect it in the coupled case. This
   test also pins the confound that a raw mean-value correlation would produce a
   large FALSE association driven purely by each fly's observation window --
   the reason the module uses the age-adjusted intercept instead.

D. Differential selection between genotypes: two genotypes with the SAME true
   decline but DIFFERENT lifespans must not appear to differ under the
   cohort-mean analysis... and if they do, the balanced-panel estimate must
   remove the difference. This is the false-drug-effect scenario.

E. TIME-TO-DEATH AXIS. When decline is generated as phase-locked to death,
   `compare_time_axes` must prefer the time-to-death axis; when it is generated
   as a function of chronological age, it must prefer age.

F. Graceful, explicit refusal when the death-age column is absent.

Run: ./.venv/Scripts/python.exe validation/validate_survivor_bias.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import survivor_bias as SB  # noqa: E402

FAILURES: list[str] = []
SEED = 20260803
TRUE_SLOPE = -0.012


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


# ─────────────────────────────────────────────────────────────────────────────
def test_A_null_case() -> None:
    print("\nA. No frailty coupling: every estimator must recover the true slope")
    df = SB.simulate_frailty_cohort(n_flies=500, mean_slope=TRUE_SLOPE,
                                    frailty_coupling=0.0, seed=SEED)
    rep = SB.survivor_bias_report(df)
    ests = {
        "cohort mean": rep["slope_survivor_mean"],
        "balanced panel": rep["slope_balanced_panel"],
        "mixed model": rep["slope_mixed_model"],
        "pattern mixture": rep["slope_pattern_mixture"],
    }
    for name, val in ests.items():
        check(f"{name} recovers the true slope ({TRUE_SLOPE:+.4f}/day)",
              np.isfinite(val) and abs(val - TRUE_SLOPE) < 0.0035,
              f"{val:+.5f}/day")
    check("bias ratio is close to 1 when there is nothing to correct",
          abs(rep["slope_bias_ratio"] - 1.0) < 0.15,
          f"ratio = {rep['slope_bias_ratio']:.3f}")
    check("verdict correctly reports no material survivor bias",
          "not materially distorting" in rep["verdict"], rep["verdict"][:80])


def test_B_central_claim() -> None:
    print("\nB. Strong frailty coupling: the cohort mean must understate decline")
    df = SB.simulate_frailty_cohort(n_flies=500, mean_slope=TRUE_SLOPE,
                                    frailty_coupling=1.0, seed=SEED + 1)
    rep = SB.survivor_bias_report(df)
    cm = rep["slope_survivor_mean"]
    mm = rep["slope_mixed_model"]
    pm = rep["slope_pattern_mixture"]
    bp = rep["slope_balanced_panel"]

    print(f"      true individual slope : {TRUE_SLOPE:+.5f}/day")
    print(f"      cohort mean           : {cm:+.5f}/day   "
          f"({(1 - cm / TRUE_SLOPE) * 100:+.0f}% error)")
    print(f"      balanced panel        : {bp:+.5f}/day")
    print(f"      mixed model           : {mm:+.5f}/day   "
          f"({(1 - mm / TRUE_SLOPE) * 100:+.0f}% error)")
    print(f"      pattern mixture       : {pm:+.5f}/day   "
          f"({(1 - pm / TRUE_SLOPE) * 100:+.0f}% error)")

    check("cohort mean is biased TOWARD ZERO (understates decline)",
          abs(cm) < abs(TRUE_SLOPE) * 0.90,
          f"|{cm:.5f}| vs true |{TRUE_SLOPE:.5f}| -- understated by "
          f"{(1 - abs(cm) / abs(TRUE_SLOPE)) * 100:.0f}%")
    check("mixed model recovers the true individual slope",
          abs(mm - TRUE_SLOPE) < 0.0025, f"{mm:+.5f} vs {TRUE_SLOPE:+.5f}")
    check("pattern-mixture estimate recovers the true slope",
          abs(pm - TRUE_SLOPE) < 0.0035, f"{pm:+.5f} vs {TRUE_SLOPE:+.5f}")
    check("mixed model is closer to the truth than the cohort mean",
          abs(mm - TRUE_SLOPE) < abs(cm - TRUE_SLOPE),
          f"|err| {abs(mm - TRUE_SLOPE):.5f} vs {abs(cm - TRUE_SLOPE):.5f}")
    check("the report flags survivor bias explicitly",
          "SURVIVOR BIAS DETECTED" in rep["verdict"], rep["verdict"][:90])

    # bias must grow monotonically with the strength of the coupling
    ratios = []
    for coup in (0.0, 0.5, 1.0, 1.5):
        d = SB.simulate_frailty_cohort(n_flies=400, mean_slope=TRUE_SLOPE,
                                       frailty_coupling=coup, seed=SEED + 2)
        r = SB.survivor_bias_report(d)
        ratios.append((coup, r["slope_bias_ratio"]))
    print("      bias ratio vs coupling: "
          + ", ".join(f"{c:.1f}->{v:.3f}" for c, v in ratios))
    check("bias ratio decreases monotonically as coupling strengthens",
          all(ratios[i][1] >= ratios[i + 1][1] - 0.03 for i in range(len(ratios) - 1)),
          ", ".join(f"{c:.1f}:{v:.3f}" for c, v in ratios))


def test_C_diagnostic_calibration() -> None:
    print("\nC. Informative-dropout diagnostic: no false alarm, real detection")
    df0 = SB.simulate_frailty_cohort(n_flies=400, frailty_coupling=0.0, seed=SEED + 3)
    df1 = SB.simulate_frailty_cohort(n_flies=400, frailty_coupling=1.0, seed=SEED + 3)

    f0 = SB.frailty_association_test(df0)
    f1 = SB.frailty_association_test(df1)

    check("no informative dropout reported when coupling is absent",
          not f0["informative_dropout"],
          f"slope rho = {f0['rho_slope_vs_death']:+.3f} (p = {f0['p_slope_vs_death']:.2g})")
    check("informative dropout detected when coupling is present",
          f1["informative_dropout"] and f1["p_slope_vs_death"] < 1e-5,
          f"slope rho = {f1['rho_slope_vs_death']:+.3f} (p = {f1['p_slope_vs_death']:.2g})")
    check("faster decliners are correctly identified as dying earlier",
          f1["rho_slope_vs_death"] > 0.2,
          f"rho = {f1['rho_slope_vs_death']:+.3f} (slope is negative for decline, so a "
          f"positive correlation with death age means steeper decliners die sooner)")

    # The confound the module exists to avoid: a RAW mean-value correlation is
    # large and significant even with zero coupling, purely because long-lived
    # flies are observed at later, lower-performing ages.
    print(f"      null case -- age-adjusted level vs death: rho = "
          f"{f0['rho_level_vs_death']:+.3f} (p = {f0['p_level_vs_death']:.2g})")
    print(f"      null case -- RAW mean value  vs death: rho = "
          f"{f0['rho_rawmean_vs_death']:+.3f} (p = {f0['p_rawmean_vs_death']:.2g})  "
          f"<- artefact of the observation window")
    check("the raw-mean correlation IS a large artefact in the null case",
          abs(f0["rho_rawmean_vs_death"]) > 0.25 and f0["p_rawmean_vs_death"] < 0.01,
          f"rho = {f0['rho_rawmean_vs_death']:+.3f} -- this is why the module uses the "
          f"age-adjusted intercept instead")
    check("the age-adjusted level shows no such artefact",
          f0["p_level_vs_death"] > 0.01,
          f"rho = {f0['rho_level_vs_death']:+.3f}, p = {f0['p_level_vs_death']:.3g}")


def test_D_false_genotype_difference() -> None:
    print("\nD. Two genotypes, same true decline, different lifespans")
    # Identical decline biology; genotype B simply dies earlier. A cohort-mean
    # comparison should not conclude that their decline rates differ.
    a = SB.simulate_frailty_cohort(n_flies=350, mean_slope=TRUE_SLOPE,
                                   frailty_coupling=1.0, median_lifespan=48.0,
                                   seed=SEED + 4)
    b = SB.simulate_frailty_cohort(n_flies=350, mean_slope=TRUE_SLOPE,
                                   frailty_coupling=1.0, median_lifespan=30.0,
                                   seed=SEED + 5)
    ra = SB.survivor_bias_report(a)
    rb = SB.survivor_bias_report(b)

    cm_gap = abs(ra["slope_survivor_mean"] - rb["slope_survivor_mean"])
    mm_gap = abs(ra["slope_mixed_model"] - rb["slope_mixed_model"])
    print(f"      cohort-mean slopes : {ra['slope_survivor_mean']:+.5f} (long-lived) vs "
          f"{rb['slope_survivor_mean']:+.5f} (short-lived)   gap = {cm_gap:.5f}")
    print(f"      mixed-model slopes : {ra['slope_mixed_model']:+.5f} vs "
          f"{rb['slope_mixed_model']:+.5f}   gap = {mm_gap:.5f}")

    check("mixed-model slopes agree between the two genotypes (as they should)",
          mm_gap < 0.0025, f"gap = {mm_gap:.5f} against a true gap of 0")
    check("the mixed model narrows the apparent genotype difference",
          mm_gap <= cm_gap + 1e-9,
          f"cohort-mean gap {cm_gap:.5f} -> mixed-model gap {mm_gap:.5f}")


def test_E_time_axis_selection() -> None:
    print("\nE. Axis selection: chronological age vs time-to-death")
    rng = np.random.default_rng(SEED + 6)

    # (i) decline generated on the AGE axis
    age_df = SB.simulate_frailty_cohort(n_flies=300, frailty_coupling=0.0, seed=SEED + 7)
    res_age = SB.compare_time_axes(age_df)
    check("age-generated decline: chronological age is preferred",
          res_age["better_axis"] == "chronological_age", res_age["verdict"][:110])

    # (ii) decline generated as TERMINAL -- phase-locked to death, flat before it
    rows = []
    for i in range(300):
        death = float(np.clip(rng.weibull(4.0) * 45 / (np.log(2) ** 0.25), 12, None))
        for a in np.arange(5, 60, 5.0):
            if a > death:
                continue
            ttd = death - a
            # high and flat until the final ~8 days, then a steep terminal drop
            val = 0.9 - 0.09 * max(0.0, 8.0 - ttd)
            rows.append({"FlyID": f"T{i}", "Age": a, "DeathAge": death,
                         "Value": float(np.clip(val + rng.normal(0, 0.03), 0, 1))})
    term_df = pd.DataFrame(rows)
    res_term = SB.compare_time_axes(term_df)
    check("terminal decline: time-to-death is preferred",
          res_term["better_axis"] == "time_to_death", res_term["verdict"][:130])

    ve_age = res_term["axes"]["chronological_age"]["variance_explained"]
    ve_ttd = res_term["axes"]["time_to_death"]["variance_explained"]
    check("time-to-death explains substantially more variance for terminal decline",
          ve_ttd > ve_age,
          f"variance explained {ve_ttd:.3f} (time-to-death) vs {ve_age:.3f} (age)")

    aligned = SB.align_to_death(term_df)
    check("align_to_death produces a correct backward axis",
          np.allclose(aligned["DaysBeforeDeath"],
                      aligned["DeathAge"] - aligned["Age"]) and
          bool((aligned["DaysBeforeDeath"] >= 0).all()),
          "DaysBeforeDeath = DeathAge - Age, non-negative for all observations")


def test_F_graceful_refusal() -> None:
    print("\nF. Explicit refusal when required information is absent")
    df = SB.simulate_frailty_cohort(n_flies=120, seed=SEED + 8).drop(columns=["DeathAge"])
    rep = SB.survivor_bias_report(df)
    check("report runs without a death-age column",
          rep["available"] and np.isfinite(rep["slope_survivor_mean"]),
          "survivor mean still computed")
    check("the missing death-age column is called out with its consequence",
          any("DeathAge" in w and "most valuable" in w for w in rep["warnings"]),
          rep["warnings"][0][:120] if rep["warnings"] else "no warning issued")
    check("mixed-model slope is still available without death ages",
          np.isfinite(rep["slope_mixed_model"]),
          f"{rep['slope_mixed_model']:+.5f}/day")

    no_id = df.drop(columns=["FlyID"])
    panel, meta = SB.balanced_panel_trajectory(no_id)
    check("balanced panel refuses without an animal identifier",
          not meta["available"] and "identifier" in meta["reason"],
          meta["reason"][:80])

    tiny = SB.simulate_frailty_cohort(n_flies=3, seed=SEED + 9)
    ft = SB.frailty_association_test(tiny)
    check("frailty test refuses on too few animals",
          not ft["available"], ft.get("reason", "")[:70])


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/survivor_bias.py")
    print("=" * 78)
    test_A_null_case()
    test_B_central_claim()
    test_C_diagnostic_calibration()
    test_D_false_genotype_difference()
    test_E_time_axis_selection()
    test_F_graceful_refusal()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL SURVIVOR-BIAS VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
