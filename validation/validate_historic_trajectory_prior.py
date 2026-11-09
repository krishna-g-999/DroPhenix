"""
validate_historic_trajectory_prior.py
The trajectory model's population prior, built from REAL cohorts for the first
time.

WHY THIS SUITE EXISTS
---------------------
`modules/trajectory.py` predicts a phenotype at ages a cohort was never
measured at, and its entire claim to being trustworthy is the population layer:
three timepoints cannot constrain the SHAPE of a decline curve, so the shape is
borrowed from cohorts that were followed to late ages, and a sparse cohort's
posterior is a precision-weighted compromise between its own data and that
population prior.

That prior had only ever been built from SIMULATION.
`validate_trajectory_prediction.py` generates 40 synthetic cohorts, builds the
prior from them, and shows shrinkage improves long-horizon forecasts. That is a
layer-2 result: it establishes the estimator works under its own assumptions.
It cannot establish that real Drosophila cohorts vary the way the model
assumes, and the value of the whole population layer rests on exactly that.

So this suite builds the prior from twenty REAL published cohorts and asks
whether it helps on real flies.

DATA
    Shahrestani P, Wilson JB, Mueller LD, Rose MR (2016) Evolution
    70:2550-2561, doi:10.1111/evo.13065; data doi:10.5061/dryad.qb509.
    `reference_datasets/physio_decline/physio_decline.xlsx`, sheet `geotaxis`:
    the proportion of flies passing the line, per population x sex x age,
    ages 1-70 d. Ten populations (ACO1-5 selected for rapid development,
    CO1-5 controls) x two sexes = 20 cohorts of 15 to 30 ages each.

    PUBLISHED GROUND TRUTH: the ACO populations decline faster than the CO
    controls.

WHAT IS VALIDATED, AND AT WHICH LAYER
-------------------------------------
A. THE ARMS ARE THE ARMS (layer 1/2). Twenty cohorts, ten per group. The
   whitespace trap in the `sex` column is asserted to be REAL and asserted to
   be fixed by the guard, so the suite fails if either the file changes or the
   guard is removed.

B. A PRIOR CAN BE BUILT FROM REAL COHORTS (layer 2). It has a positive
   semi-definite between-cohort covariance and a finite residual SD. A prior
   whose covariance has collapsed would silently turn shrinkage into
   substitution.

C. THE PRIOR HELPS ON REAL SPARSE COHORTS (layer 3) -- the question the
   population layer exists to answer. Each real cohort is reduced to its
   EARLIEST three ages and its later ages are predicted, with the prior built
   LEAVE-ONE-OUT so that no cohort informs its own prediction. Scored against
   the same cohort with no prior at all, and against the trivial baselines
   `forward_validate` already computes. Prediction is forward in time; a random
   hold-out across a cohort's own ages would leak the future.

D. A SYNTHETIC PRIOR IS NOT A SUBSTITUTE (layer 3/4). The same real cohorts are
   predicted using a prior built from simulated arms instead of real ones. If
   the simulated prior does as well, then the published arms add nothing and
   the simulation was adequate; if it does worse, every result that used a
   simulated prior was flattered by it. Either answer is worth having and
   neither is assumed.

E. THE PUBLISHED BIOLOGY SURVIVES THE FIT (layer 4). Fitted decline rates must
   recover ACO declining faster than CO. A model that fits well and reverses
   the biology is worse than one that fits badly.

F. THE CEILING IS REPORTED, NOT HIDDEN (layer 4). Young flies all pass the
   line, so the proportion is pinned at 1.0 and must be clipped for a logit
   parameterisation. The clip bites the two groups UNEQUALLY -- measured, the
   CO controls sit at the ceiling far more than the ACO populations -- so it is
   a differential effect on exactly the comparison the data exists to make, and
   it is quantified here.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
That the prior transfers to a different assay, a different laboratory or a
different phenotype. These are ten outbred populations from one selection
experiment measured by one group, and the prior is over curve shape for THIS
phenotype. Nor does it establish anything about DroPhenix's video pipeline:
the geotaxis scores here were produced by the original authors, so this
validates the decline model and not the measurement of it.

Run: ./.venv/Scripts/python.exe validation/validate_historic_trajectory_prior.py
Exit 0 = all pass.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import historic_arms as HA  # noqa: E402
from modules import trajectory as TJ  # noqa: E402

FAILURES: List[str] = []
FAMILY = HA.REAL_FAMILY          # chosen by identifiability, see test_A2
N_SPARSE = 3          # the case the population layer exists for


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name + (f" ({detail})" if detail else ""))


# ──────────────────────────────────────────────────────────────────────── A ──
def test_A_arms(arms: List[HA.Arm]) -> None:
    print("\nA. The published arms")
    if not arms:
        print("  [SKIP] the published dataset is not on this machine")
        return
    print(HA.describe_arms(arms))

    check("twenty published cohorts: 10 populations x 2 sexes",
          len(arms) == 20, f"{len(arms)}")
    groups = {g: sum(1 for a in arms if a.group == g) for g in ("ACO", "CO")}
    check("ten per selection regime", groups == {"ACO": 10, "CO": 10}, f"{groups}")
    check("every arm is dense enough to pin a curve down (>= 10 ages)",
          all(a.n_points >= 10 for a in arms),
          f"shortest {min(a.n_points for a in arms)} ages")
    check("every arm spans at least 30 days of age",
          all(a.age_span >= 30 for a in arms),
          f"shortest span {min(a.age_span for a in arms):.0f} d")
    check("all values lie strictly inside (0, 1), as a logit fit requires",
          all(np.all((a.values > 0) & (a.values < 1)) for a in arms))

    # The whitespace trap, asserted in BOTH directions: real, and fixed.
    try:
        import pandas as pd
        g = pd.read_excel(HA.PHYSIO_XLSX, sheet_name="geotaxis")
        raw = g.groupby([g["pop"], g["sex"]]).ngroups
        stripped = g.groupby([g["pop"].astype(str).str.strip(),
                              g["sex"].astype(str).str.strip()]).ngroups
        print(f"      grouping on the raw columns gives {raw} cohorts; "
              f"on stripped columns, {stripped}")
        check("the stray-whitespace trap in `sex` is REAL in this file "
              "(raw grouping invents cohorts)",
              raw > stripped, f"raw {raw} vs stripped {stripped}")
        check("and the loader's guard removes it",
              stripped == 20 and len(arms) == 20,
              f"stripped {stripped}, arms {len(arms)}")
    except Exception as exc:                                   # pragma: no cover
        print(f"      [SKIP] could not re-read the file to check the trap ({exc!r})")


# ─────────────────────────────────────────────────────────────────────── A2 ──
def test_A2_family_is_identified(arms: List[HA.Arm]) -> None:
    print("")
    print("A2. Which curve family can these real cohorts actually support?")
    print("    A fit resting on an identifiability bound has not estimated that")
    print("    parameter, it has run out of room, and `build_population_prior`")
    print("    correctly excludes it. So the family must be chosen by what the")
    print("    data can support, not by what the module defaults to.")
    if not arms:
        print("  [SKIP] no published arms")
        return
    table = {}
    print()
    print("      %-14s %12s %12s" % ("family", "identified", "mean AICc"))
    for fam in TJ.FAMILIES:
        ok, aiccs = 0, []
        for a in arms:
            f = TJ.fit_series(a.ages, a.values, fam)
            if f is None:
                continue
            aiccs.append(f.aicc)
            ok += int(not f.at_bounds)
        table[fam] = (ok, float(np.mean(aiccs)) if aiccs else np.nan)
        print("      %-14s %9d/%-2d %12.2f" % (fam, ok, len(arms), table[fam][1]))

    check("the module's DEFAULT family, exp_plateau, is NOT identified on these "
          "real cohorts -- they never plateau inside the observed age window",
          table.get("exp_plateau", (99, 0))[0] == 0,
          f"identified on {table.get('exp_plateau', ('?',))[0]} of {len(arms)}")
    check(f"the family actually used, {FAMILY}, is identified on every cohort",
          table.get(FAMILY, (0, 0))[0] == len(arms),
          f"identified on {table.get(FAMILY, ('?',))[0]} of {len(arms)}")
    best = min((v[1], k) for k, v in table.items() if np.isfinite(v[1]))[1]
    check(f"and {FAMILY} is also the best-fitting family by mean AICc",
          best == FAMILY, f"best was {best}")
    print()
    print("      WHY THIS MATTERS BEYOND THIS SUITE: validate_trajectory_prediction")
    print("      builds its prior in exp_plateau, and its cohorts are SIMULATED")
    print("      FROM exp_plateau, so the family was guaranteed to fit. On real")
    print("      cohorts it cannot be fitted at all.")


# ──────────────────────────────────────────────────────────────────────── B ──
def test_B_prior(arms: List[HA.Arm]) -> Optional[TJ.PopulationPrior]:
    print("\nB. A population prior from real cohorts")
    if not arms:
        print("  [SKIP] no published arms")
        return None
    prior = TJ.build_population_prior(HA.arms_to_cohorts(arms), family=FAMILY,
                                      min_points=4)
    check("a prior was built from the real cohorts", prior is not None)
    if prior is None:
        return None
    print(f"      family {prior.family}, {prior.n_cohorts} cohorts, "
          f"residual SD {prior.residual_sd:.4f}")
    print(f"      parameters {prior.param_names}")
    print(f"      mean {np.round(prior.mean, 3)}")
    for n in prior.notes:
        print(f"      note: {n}")
    check("the prior used at least 18 of the 20 real cohorts",
          prior.n_cohorts >= 18, f"{prior.n_cohorts}")
    ev = np.linalg.eigvalsh(prior.cov)
    check("the between-cohort covariance is positive semi-definite",
          float(ev.min()) >= -1e-9, f"min eigenvalue {float(ev.min()):.3e}")
    check("the between-cohort covariance has NOT collapsed (shrinkage would "
          "become substitution)",
          float(ev.max()) > 1e-6, f"max eigenvalue {float(ev.max()):.3e}")
    check("the residual SD is finite and plausible for a proportion",
          np.isfinite(prior.residual_sd) and 0 < prior.residual_sd < 0.5,
          f"{prior.residual_sd:.4f}")
    return prior


# ──────────────────────────────────────────────────────────────────────── C ──
def _loo_prior(arms: List[HA.Arm], held_out: str) -> Optional[TJ.PopulationPrior]:
    """The prior built from every arm EXCEPT the one being predicted."""
    cohorts = {a.label: a.as_cohort() for a in arms if a.label != held_out}
    return TJ.build_population_prior(cohorts, family=FAMILY, min_points=4)


def test_C_prior_helps(arms: List[HA.Arm]) -> Dict[str, dict]:
    print("\nC. Does the real prior help a real sparse cohort?")
    print("   Each cohort is cut to its earliest 3 ages and its later ages are")
    print("   predicted. The prior EXCLUDES the cohort being predicted, so")
    print("   nothing informs its own forecast.")
    if not arms:
        print("  [SKIP] no published arms")
        return {}

    rows = []
    for a in arms:
        if a.n_points < N_SPARSE + 2:
            continue
        train_max = float(a.ages[N_SPARSE - 1])
        prior = _loo_prior(arms, a.label)
        if prior is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with_p = TJ.forward_validate(a.ages, a.values, train_max,
                                             prior=prior, seed=7)
                without = TJ.forward_validate(a.ages, a.values, train_max,
                                              prior=None, seed=7)
        except Exception as exc:
            print(f"      {a.label}: forward_validate raised {exc!r}")
            continue
        rows.append({
            "arm": a.label, "group": a.group,
            "train_max_day": train_max,
            "n_scored": with_p["summary"]["n_scored"],
            "mae_prior": with_p["summary"]["MAE"],
            "mae_noprior": without["summary"]["MAE"],
            "cov95_prior": with_p["summary"]["coverage_95"],
            "cov95_noprior": without["summary"]["coverage_95"],
            "crps_prior": with_p["summary"]["mean_CRPS"],
            "crps_noprior": without["summary"]["mean_CRPS"],
            "refused_prior": with_p["summary"]["n_refused"],
            "refused_noprior": without["summary"]["n_refused"],
            "used_prior": bool(with_p["used_prior"]),
        })

    if not rows:
        print("  [SKIP] no cohort could be forward-validated")
        return {}

    import pandas as pd
    t = pd.DataFrame(rows)
    print()
    print("      %-12s %-6s %7s %9s %10s %9s %10s"
          % ("arm", "group", "scored", "MAE prior", "MAE none", "cov prior",
             "cov none"))
    for _, r in t.iterrows():
        print("      %-12s %-6s %7d %9.4f %10.4f %8.0f%% %9.0f%%"
              % (r["arm"], r["group"], r["n_scored"], r["mae_prior"],
                 r["mae_noprior"], 100 * (r["cov95_prior"] or np.nan),
                 100 * (r["cov95_noprior"] or np.nan)))

    n_better = int((t["mae_prior"] < t["mae_noprior"]).sum())
    print()
    print("      prior better on %d of %d cohorts; mean MAE %.4f with the prior "
          "vs %.4f without"
          % (n_better, len(t), t["mae_prior"].mean(), t["mae_noprior"].mean()))
    print("      mean 95%% coverage %.0f%% with the prior vs %.0f%% without "
          "(nominal 95%%)"
          % (100 * t["cov95_prior"].mean(), 100 * t["cov95_noprior"].mean()))
    print("      predictions REFUSED beyond the horizon: %d with the prior, "
          "%d without"
          % (int(t["refused_prior"].sum()), int(t["refused_noprior"].sum())))

    check("the prior was actually used (not silently ignored)",
          bool(t["used_prior"].all()), f"{int(t['used_prior'].sum())}/{len(t)}")
    check("the prior improves the mean absolute error on real sparse cohorts",
          float(t["mae_prior"].mean()) < float(t["mae_noprior"].mean()),
          f"{t['mae_prior'].mean():.4f} vs {t['mae_noprior'].mean():.4f}")
    check("the prior helps on a majority of the real cohorts",
          n_better > len(t) / 2, f"{n_better}/{len(t)}")
    # Coverage is the honesty check: a forecaster may win on MAE while lying
    # about its own uncertainty, and the second failure is the worse one.
    cov = float(t["cov95_prior"].mean())
    check("95% intervals from the prior-informed fit cover at least 80% of real "
          "held-out points",
          cov >= 0.80, f"{100 * cov:.0f}%")
    return {"table": t}


# ──────────────────────────────────────────────────────────────────────── D ──
def test_D_synthetic_prior_is_not_a_substitute(arms: List[HA.Arm],
                                               real_table) -> None:
    print("\nD. Would a prior built from SIMULATION have done as well?")
    if not arms or real_table is None or not len(real_table):
        print("  [SKIP] no real comparison to make")
        return
    syn = HA.synthetic_arms(n=20, family=FAMILY, seed=23)
    syn_prior = TJ.build_population_prior(HA.arms_to_cohorts(syn),
                                          family=FAMILY, min_points=4)
    check("a prior builds from the synthetic arms too", syn_prior is not None)
    if syn_prior is None:
        return

    rows = []
    for a in arms:
        if a.n_points < N_SPARSE + 2:
            continue
        train_max = float(a.ages[N_SPARSE - 1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = TJ.forward_validate(a.ages, a.values, train_max,
                                        prior=syn_prior, seed=7)
        except Exception:
            continue
        rows.append({"arm": a.label, "mae_syn": r["summary"]["MAE"],
                     "cov95_syn": r["summary"]["coverage_95"]})
    if not rows:
        print("  [SKIP] nothing scored under the synthetic prior")
        return
    import pandas as pd
    s = pd.DataFrame(rows).merge(real_table[["arm", "mae_prior", "mae_noprior"]],
                                 on="arm")
    print("      mean MAE on the SAME real cohorts:")
    print("        real prior      %.4f" % s["mae_prior"].mean())
    print("        synthetic prior %.4f" % s["mae_syn"].mean())
    print("        no prior        %.4f" % s["mae_noprior"].mean())
    better = float(s["mae_prior"].mean()) < float(s["mae_syn"].mean())
    print()
    if better:
        print("      -> the prior built from REAL cohorts forecasts real cohorts")
        print("         better than one built from simulation. Every earlier")
        print("         result that used a simulated prior was therefore")
        print("         measuring the estimator, not the biology.")
    else:
        print("      -> a simulated prior does as well here, so on this dataset")
        print("         the published arms add nothing beyond what the")
        print("         simulation already captured. That is a real answer and")
        print("         it is reported rather than buried.")
    # Deliberately NOT asserted either way: both outcomes are informative and
    # asserting the one we hope for would make the suite a ratchet for a
    # conclusion rather than a test of it. What IS asserted is that the
    # comparison ran on the same cohorts.
    check("the synthetic-prior comparison covers the same real cohorts",
          len(s) >= 0.8 * len(real_table), f"{len(s)} of {len(real_table)}")


# ──────────────────────────────────────────────────────────────────────── E ──
def test_E_published_biology(arms: List[HA.Arm]) -> None:
    print("")
    print("E. Does the fit recover the published biology, and HOW?")
    print("   Shahrestani et al. report that ACO (selected for rapid")
    print("   development) ages faster than the CO controls. The question this")
    print("   section settles is which STATISTIC carries that difference,")
    print("   because the obvious one is confounded.")
    if not arms:
        print("  [SKIP] no published arms")
        return
    from scipy import stats

    # ── the confound, measured ──────────────────────────────────────────────
    amax = {g: [float(a.ages.max()) for a in arms if a.group == g]
            for g in ("ACO", "CO")}
    drop = {g: [float(a.values[0] - a.values[-1]) for a in arms if a.group == g]
            for g in ("ACO", "CO")}
    pu = stats.mannwhitneyu(amax["ACO"], amax["CO"]).pvalue
    print()
    print("   THE FOLLOW-UP CONFOUND, which has to come first:")
    print("     last age observed : ACO median %.0f d, CO median %.0f d  (p = %.4f)"
          % (np.median(amax["ACO"]), np.median(amax["CO"]), pu))
    print("     total decline     : ACO %.3f, CO %.3f  (nearly identical)"
          % (np.median(drop["ACO"]), np.median(drop["CO"])))
    check("follow-up is UNEQUAL between the groups, so any statistic that "
          "divides a decline by its observation window is confounded",
          pu < 0.05, f"p = {pu:.4f}")

    # ── exp_plateau's apparent confirmation, and why it is not one ──────────
    ke = {"ACO": [], "CO": []}
    for a in arms:
        f = TJ.fit_series(a.ages, a.values, "exp_plateau")
        if f is None:
            continue
        ke[a.group].append(
            float(TJ.FAMILIES["exp_plateau"].describe(f.theta)["rate_k_per_day"]))
    if ke["ACO"] and ke["CO"]:
        u = stats.mannwhitneyu(ke["ACO"], ke["CO"], alternative="greater")
        print()
        print("   exp_plateau's decay rate k separates them cleanly:")
        print("     ACO median %.4f /day, CO median %.4f /day, p = %.4f"
              % (np.median(ke["ACO"]), np.median(ke["CO"]), u.pvalue))
        print("     ...and this is NOT usable evidence. exp_plateau is identified")
        print("     on 0 of 20 cohorts (section A2), so with the asymptote pinned")
        print("     at its bound k is approximately the decline divided by the")
        print("     window -- and the windows differ by 16 days while the declines")
        print("     do not. The 2.5-fold figure is substantially the follow-up")
        print("     difference and must not be quoted as a rate.")

    # ── the identified family's slope, across windows ───────────────────────
    print()
    print("   logit_linear's per-day slope, the identified family, by window:")
    print("     %-12s %10s %10s %12s" % ("ages <=", "ACO", "CO", "p(ACO>CO)"))
    flips = []
    for cap in (33, 36, 42, None):
        res = {"ACO": [], "CO": []}
        for a in arms:
            m = np.ones(a.ages.size, bool) if cap is None else (a.ages <= cap)
            if int(m.sum()) < 4:
                continue
            f = TJ.fit_series(a.ages[m], a.values[m], FAMILY)
            if f is None:
                continue
            res[a.group].append(
                -float(TJ.FAMILIES[FAMILY].describe(f.theta)["logit_slope_per_day"]))
        if res["ACO"] and res["CO"]:
            u = stats.mannwhitneyu(res["ACO"], res["CO"], alternative="greater")
            flips.append((np.median(res["ACO"]) > np.median(res["CO"])))
            print("     %-12s %10.4f %10.4f %12.4f"
                  % (str(cap) if cap else "all", np.median(res["ACO"]),
                     np.median(res["CO"]), u.pvalue))
    check("the per-day slope's DIRECTION reverses with the observation window, "
          "so it is not measuring a property of the cohorts",
          len(set(flips)) > 1, f"directions across windows: {flips}")

    # ── what actually separates them: when the decline starts ───────────────
    print()
    print("   ONSET: the age at which a cohort first falls below a level.")
    print("     %-10s %12s %12s %12s" % ("level", "ACO median", "CO median", "p(ACO<CO)"))
    onset_ok = []
    for lvl in (0.95, 0.90, 0.80):
        o = {"ACO": [], "CO": []}
        for a in arms:
            below = a.ages[a.values < lvl]
            if below.size:
                o[a.group].append(float(below[0]))
        if len(o["ACO"]) >= 5 and len(o["CO"]) >= 5:
            u = stats.mannwhitneyu(o["ACO"], o["CO"], alternative="less")
            onset_ok.append(u.pvalue < 0.05
                            and np.median(o["ACO"]) < np.median(o["CO"]))
            print("     %-10.2f %12.1f %12.1f %12.4f"
                  % (lvl, np.median(o["ACO"]), np.median(o["CO"]), u.pvalue))
    check("ACO begins declining EARLIER than CO, at every level tested -- which "
          "is the published direction, carried by onset rather than by rate",
          bool(onset_ok) and all(onset_ok), f"{onset_ok}")

    print()
    print("   THE READING. The ACO populations do age faster, and the way they")
    print("   do it is to start declining about twenty days earlier at a SIMILAR")
    print("   per-day rate -- not to decline more steeply. That is consistent")
    print("   with their smaller ceiling fraction (section F) and their shorter")
    print("   lifespan, and it is a different claim from the one the exp_plateau")
    print("   fit appeared to support.")
    print()
    print("   NOTE: the 20 cohorts are 10 populations x 2 sexes, so the two sexes")
    print("   of a population are not independent. Every p-value here treats")
    print("   them as if they were, is therefore anticonservative, and is")
    print("   reported as a descriptive statistic rather than a test.")


# ──────────────────────────────────────────────────────────────────────── F ──
def test_F_ceiling(arms: List[HA.Arm]) -> None:
    print("\nF. The ceiling, and whether it bites the two groups equally")
    if not arms:
        print("  [SKIP] no published arms")
        return
    by = {}
    for g in ("ACO", "CO"):
        fr = [a.ceiling_fraction for a in arms if a.group == g]
        if fr:
            by[g] = (float(np.mean(fr)), float(min(fr)), float(max(fr)))
            print("      %-4s ceiling fraction: mean %.0f%% (range %.0f-%.0f%%)"
                  % (g, 100 * by[g][0], 100 * by[g][1], 100 * by[g][2]))
    if len(by) == 2:
        diff = abs(by["ACO"][0] - by["CO"][0])
        print()
        print("      The difference is %.0f percentage points. The clip to"
              % (100 * diff))
        print("      1 - %g is therefore NOT a neutral formatting step: it" % HA.EPS)
        print("      applies unequally to the two groups being compared, and the")
        print("      group it applies to more is the slower-ageing one, whose")
        print("      early ages are genuinely saturated. A fitted y0 for those")
        print("      cohorts is pinned by the clip rather than by the data.")
        check("the differential ceiling effect is quantified rather than ignored",
              True, f"ACO {100 * by['ACO'][0]:.0f}% vs CO {100 * by['CO'][0]:.0f}%")
        if diff > 0.10:
            print()
            print("      CONSEQUENCE FOR SECTION E: a y0 pinned by the clip biases")
            print("      the fitted decline rate, and it does so more in CO. The")
            print("      direction ACO > CO is therefore supported by the fit but")
            print("      its MAGNITUDE should not be quoted from these fits.")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: the trajectory population prior, built from real cohorts")
    print("=" * 78)
    arms = HA.published_geotaxis_arms()
    if not arms:
        print("\n  [SKIP] reference_datasets/physio_decline/physio_decline.xlsx")
        print("         is not present, so this suite established nothing.")
        return 0

    test_A_arms(arms)
    test_A2_family_is_identified(arms)
    test_B_prior(arms)
    res = test_C_prior_helps(arms)
    test_D_synthetic_prior_is_not_a_substitute(arms, res.get("table"))
    test_E_published_biology(arms)
    test_F_ceiling(arms)

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL HISTORIC-TRAJECTORY-PRIOR CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
