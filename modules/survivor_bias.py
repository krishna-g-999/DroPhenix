"""
Survivor bias in Drosophila ageing and neurodegeneration trajectories.

THE PROBLEM
-----------
In an ageing cohort the weakest animals die first. Any phenotype measured at
age t is therefore measured in a PROGRESSIVELY SELECTED, progressively fitter
subpopulation. The cohort-mean trajectory -- the mean over whatever flies are
alive at each age, which is what almost every published climbing curve plots --
is not the trajectory of a fly. It is the trajectory of a survivor.

Two consequences, both of which push in the direction that manufactures false
findings:

1. The apparent rate of decline is BIASED TOWARD ZERO, because the flies that
   would have pulled the mean down have been removed from it. A cohort mean can
   be flat while every individual in it is declining.

2. The bias is LARGER IN THE SICKER GENOTYPE, because that genotype loses more
   of its frail animals. A neurotoxic manipulation that kills frail flies early
   will therefore appear to have a SHALLOWER motor decline among survivors --
   which is indistinguishable, in a cohort-mean plot, from a protective effect.

The published evidence that this matters in flies specifically: in longitudinal
single-fly tracking of wild-type Drosophila, ~64% of flies develop locomotor
impairment and impairments appear predominantly in the LAST 4-6 DAYS OF LIFE
(~8% of lifespan), with >80% impaired within 48 h of death (Gaitanidis et al.
2019, Aging 11:1850-1873). Motor collapse in flies is largely a TERMINAL event
phase-locked to death, not a smooth function of chronological age. Comparing
genotypes with different lifespans at the same chronological age therefore
compares animals at entirely different points in their terminal trajectory.

WHAT THIS MODULE PROVIDES
-------------------------
  survivor_mean_trajectory   -- the conventional (biased) estimate, computed
                                explicitly so it can be compared against
  balanced_panel_trajectory  -- restricted to animals surviving to a common
                                age; removes differential selection at the cost
                                of sample size
  mixed_slope                -- individual-level slope from a random-slope
                                mixed model, which uses each animal's own
                                pre-death observations and is valid under
                                dropout that depends on observed history
  pattern_mixture_trajectory -- Little (1993) pattern-mixture estimate: stratify
                                by death time, then reweight to the ORIGINAL
                                cohort composition. The identifying assumption
                                is stated explicitly rather than buried.
  frailty_association_test   -- does an animal's own trajectory predict its
                                death? If yes, dropout is informative and the
                                ignorability assumption is doing real work.
  align_to_death             -- re-express the time axis as time-to-death, the
                                axis on which terminal decline is actually
                                phase-locked
  survivor_bias_report       -- runs all of the above and quantifies the
                                disagreement between them

WHAT THIS MODULE DOES NOT CLAIM
-------------------------------
None of these estimators recovers "what a dead fly's climbing would have been",
because that quantity is not defined. What they do is make the ESTIMAND
explicit -- survivor-conditional, balanced-cohort, or individual-level -- and
show how far apart those three answers are in a given dataset. When they
disagree substantially, a cohort-mean decline constant should not be reported
as if it described an individual.

References
----------
Gaitanidis A, Dimitriadou A, Dowse H, Sanyal S, Duch C, Consoulas C (2019)
  Longitudinal assessment of health-span and pre-death morbidity in wild type
  Drosophila. Aging (Albany NY) 11:1850-1873.
Little RJA (1993) Pattern-mixture models for multivariate incomplete data.
  J Am Stat Assoc 88:125-134.
Rizopoulos D (2012) Joint Models for Longitudinal and Time-to-Event Data.
  Chapman & Hall/CRC.
Wulfsohn MS, Tsiatis AA (1997) A joint model for survival and longitudinal data
  measured with error. Biometrics 53:330-339.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# 1. THE THREE ESTIMANDS
# ═══════════════════════════════════════════════════════════════════════════
def survivor_mean_trajectory(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
) -> pd.DataFrame:
    """
    Mean phenotype among animals ALIVE AND MEASURED at each age.

    This is the conventional cohort-mean curve. It is computed here not because
    it is right but because it is the thing everything else must be compared
    against: it is what the field plots, and its estimand is
    "E[Y(t) | alive at t]", which changes meaning at every age.
    """
    work = df.dropna(subset=[age_col, value_col])
    out = (
        work.groupby(age_col)
        .agg(
            mean=(value_col, "mean"),
            sd=(value_col, "std"),
            n_alive=(id_col, "nunique") if id_col in work.columns else (value_col, "size"),
        )
        .reset_index()
        .sort_values(age_col)
    )
    out["sem"] = out["sd"] / np.sqrt(out["n_alive"].clip(lower=1))
    out["estimand"] = "E[Y(t) | alive at t]  (survivor-conditional)"
    return out


def balanced_panel_trajectory(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
    to_age: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Mean phenotype restricted to animals that survive to a COMMON age.

    Because the same animals contribute at every age, differential selection is
    removed and the curve describes a fixed cohort. The price is sample size,
    and the estimand becomes "E[Y(t) | alive at T]" for the fixed T -- which is
    itself a selected group, but a CONSISTENTLY selected one, so slopes are
    comparable across ages and across genotypes.

    `to_age` defaults to the latest age at which at least 3 animals remain.
    """
    work = df.dropna(subset=[age_col, value_col]).copy()
    if id_col not in work.columns:
        return pd.DataFrame(), {"available": False,
                                "reason": f"Balanced panel needs an animal identifier ('{id_col}')."}

    counts = work.groupby(age_col)[id_col].nunique().sort_index()
    if to_age is None:
        eligible = counts[counts >= 3]
        if eligible.empty:
            return pd.DataFrame(), {"available": False,
                                    "reason": "No age retains at least 3 animals."}
        to_age = float(eligible.index.max())

    survivors = set(work.loc[work[age_col] >= float(to_age), id_col].unique())
    if len(survivors) < 3:
        return pd.DataFrame(), {"available": False,
                                "reason": f"Only {len(survivors)} animals survive to age {to_age:g}."}

    panel = work[work[id_col].isin(survivors) & (work[age_col] <= float(to_age))]
    out = (
        panel.groupby(age_col)
        .agg(mean=(value_col, "mean"), sd=(value_col, "std"), n=(id_col, "nunique"))
        .reset_index()
        .sort_values(age_col)
    )
    out["sem"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["estimand"] = f"E[Y(t) | alive at age {to_age:g}]  (balanced panel)"

    meta = {
        "available": True,
        "to_age": float(to_age),
        "n_survivors": int(len(survivors)),
        "n_total_animals": int(work[id_col].nunique()),
        "retention": float(len(survivors) / max(work[id_col].nunique(), 1)),
    }
    return out, meta


def mixed_slope(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
    group_col: Optional[str] = None,
) -> dict:
    """
    Population-average slope from a random-intercept + random-slope mixed model
    fitted to each animal's own repeated observations.

    Because the model uses every animal's within-animal change rather than
    cross-sectional means, it is not driven by which animals are still present
    at late ages, and it is valid when dropout depends on the animal's OBSERVED
    history. It is not automatically valid when dropout depends on the animal's
    latent trajectory -- see `frailty_association_test`, which measures exactly
    that.
    """
    import statsmodels.formula.api as smf

    work = df.dropna(subset=[age_col, value_col, id_col]).copy()
    work = work.rename(columns={age_col: "_age", value_col: "_val", id_col: "_id"})

    n_ids = work["_id"].nunique()
    if n_ids < 3:
        return {"available": False, "reason": f"Only {n_ids} animals; need at least 3."}

    per = work.groupby("_id")["_age"].nunique()
    if int((per >= 2).sum()) < 3:
        return {"available": False,
                "reason": "Fewer than 3 animals have 2+ observations; a slope cannot be "
                          "estimated within animals."}

    fixed = "_val ~ _age"
    if group_col and group_col in df.columns:
        # `work` is `df` with rows dropped, so the index still aligns.
        work["_grp"] = df.loc[work.index, group_col]
        if work["_grp"].nunique() < 2:
            return {"available": False,
                    "reason": f"'{group_col}' has fewer than 2 levels in the usable rows."}
        fixed = "_val ~ _age * C(_grp)"

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        try:
            model = smf.mixedlm(fixed, work, groups=work["_id"], re_formula="1 + _age")
            res = model.fit(reml=True, method="lbfgs")
        except Exception:
            try:
                model = smf.mixedlm(fixed, work, groups=work["_id"], re_formula="1")
                res = model.fit(reml=True)
            except Exception as exc:
                return {"available": False, "reason": f"Mixed model failed: {exc}"}

    fe = res.fe_params
    slope_key = "_age"
    if slope_key not in fe.index:
        return {"available": False, "reason": "No age term in the fitted model."}

    se = float(res.bse[slope_key])
    est = float(fe[slope_key])
    return {
        "available": True,
        "slope": est,
        "se": se,
        "ci_lower": est - 1.96 * se,
        "ci_upper": est + 1.96 * se,
        "p_value": float(res.pvalues[slope_key]),
        "n_animals": int(n_ids),
        "n_obs": int(len(work)),
        "estimand": "individual-level mean slope dE[Y_i(t)]/dt (mixed model)",
        "model": res,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. PATTERN-MIXTURE REWEIGHTING
# ═══════════════════════════════════════════════════════════════════════════
def pattern_mixture_trajectory(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
    death_col: str = "DeathAge",
    n_strata: int = 3,
) -> Tuple[pd.DataFrame, dict]:
    """
    Little (1993) pattern-mixture estimate of the FULL-COHORT mean trajectory.

    Animals are stratified by when they died. Within each stratum the mean
    trajectory is estimated where observed, and each stratum's own within-animal
    linear trend is extrapolated past its death to the end of the window. The
    strata are then averaged with weights equal to their share of the ORIGINAL
    cohort, so the result is an estimate of

        E[Y(t)]  over the whole starting cohort, not over survivors.

    IDENTIFYING ASSUMPTION, STATED PLAINLY: the post-death portion of each
    stratum's curve is a linear extrapolation of that stratum's own observed
    trend. This is an assumption about unobservable quantities and it cannot be
    checked from the data. It is reported alongside every result. The point of
    the estimator is not that the assumption is true but that it is EXPLICIT --
    whereas the cohort-mean curve makes the far stronger implicit assumption
    that dead animals would have behaved exactly like survivors.
    """
    needed = {id_col, age_col, value_col, death_col}
    missing = needed - set(df.columns)
    if missing:
        return pd.DataFrame(), {"available": False,
                                "reason": f"Missing columns: {sorted(missing)}"}

    work = df.dropna(subset=[age_col, value_col, id_col, death_col]).copy()
    if work.empty:
        return pd.DataFrame(), {"available": False, "reason": "No usable rows."}

    deaths = work.groupby(id_col)[death_col].first()
    n_animals = int(len(deaths))
    if n_animals < max(3 * n_strata, 9):
        return pd.DataFrame(), {"available": False,
                                "reason": f"Only {n_animals} animals; need at least "
                                          f"{max(3 * n_strata, 9)} for {n_strata} strata."}

    qs = np.linspace(0, 1, n_strata + 1)[1:-1]
    cuts = np.unique(np.quantile(deaths.values, qs))
    stratum_of = np.digitize(deaths.values, cuts)
    strat_map = pd.Series(stratum_of, index=deaths.index, name="_stratum")
    work = work.join(strat_map, on=id_col)

    ages = np.sort(work[age_col].unique())
    rows, notes = [], []
    weights = strat_map.value_counts(normalize=True).sort_index()

    for s, w in weights.items():
        sub = work[work["_stratum"] == s]
        obs = sub.groupby(age_col)[value_col].mean()
        if len(obs) >= 2:
            slope, intercept = np.polyfit(obs.index.values.astype(float), obs.values, 1)
        else:
            slope, intercept = 0.0, float(obs.values[0]) if len(obs) else np.nan
        last_obs_age = float(obs.index.max()) if len(obs) else np.nan
        for a in ages:
            if a in obs.index:
                val, kind = float(obs.loc[a]), "observed"
            else:
                val, kind = float(intercept + slope * a), "extrapolated"
            rows.append({"_stratum": int(s), age_col: float(a), "value": val,
                         "weight": float(w), "kind": kind})
        notes.append(
            f"stratum {int(s)}: {int((strat_map == s).sum())} animals, "
            f"weight {w:.3f}, observed to age {last_obs_age:g}, "
            f"slope {slope:+.5f}/day"
        )

    long = pd.DataFrame(rows)
    est = (
        long.assign(wv=lambda d: d["value"] * d["weight"])
        .groupby(age_col)
        .agg(mean=("wv", "sum"),
             extrapolated_weight=("weight", lambda w: float(
                 long.loc[w.index].query("kind == 'extrapolated'")["weight"].sum())))
        .reset_index()
        .sort_values(age_col)
    )
    est["estimand"] = "E[Y(t)] over the full starting cohort (pattern-mixture)"

    meta = {
        "available": True,
        "n_strata": int(len(weights)),
        "n_animals": n_animals,
        "stratum_notes": notes,
        "identifying_assumption": (
            "Post-death values within each death-time stratum are LINEAR "
            "EXTRAPOLATIONS of that stratum's own observed trend. This is "
            "unverifiable from the data and is the price of estimating a "
            "full-cohort mean at ages after some animals have died."
        ),
    }
    return est, meta


# ═══════════════════════════════════════════════════════════════════════════
# 3. IS DROPOUT INFORMATIVE?
# ═══════════════════════════════════════════════════════════════════════════
def frailty_association_test(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
    death_col: str = "DeathAge",
) -> dict:
    """
    Test whether an animal's own trajectory predicts when it dies.

    Each animal's own slope and AGE-ADJUSTED LEVEL (the intercept of its own
    least-squares line) are correlated with its death age by Spearman
    correlation, which needs no linearity assumption.

    Why the age-adjusted level and not the animal's raw mean value: a long-lived
    animal is necessarily observed at later ages, where the phenotype is lower
    for reasons that have nothing to do with frailty. Correlating the raw mean
    with death age therefore produces a large spurious association driven purely
    by each animal's observation window. In simulation with the frailty coupling
    switched OFF, raw mean value against death age gives rho = -0.44
    (p = 3e-20) -- an artefact -- while the slope gives rho = -0.002 (p = 0.97),
    correctly detecting nothing. The intercept is used because it is the level
    at a common reference age and so is not window-dependent.

    If there is no association, dropout carries no information about the
    trajectory and the survivor-conditional curve is close to the full-cohort
    curve. If there IS an association -- which is the biologically expected
    case, since frail animals both decline faster and die sooner -- then the
    cohort-mean curve is selected on the very quantity being measured, and the
    size of the correlation indicates how strongly.

    This is the diagnostic that tells a user whether they need to care.
    """
    needed = {id_col, age_col, value_col, death_col}
    missing = needed - set(df.columns)
    if missing:
        return {"available": False, "reason": f"Missing columns: {sorted(missing)}"}

    work = df.dropna(subset=[id_col, age_col, value_col, death_col])
    recs = []
    for fid, g in work.groupby(id_col):
        g = g.sort_values(age_col)
        if g[age_col].nunique() < 2:
            continue
        t = g[age_col].to_numpy(dtype=float)
        v = g[value_col].to_numpy(dtype=float)
        try:
            slope, intercept = np.polyfit(t, v, 1)
        except Exception:
            continue
        recs.append({
            "id": fid,
            "slope": float(slope),
            "intercept": float(intercept),
            "mean_value": float(v.mean()),
            "death_age": float(g[death_col].iloc[0]),
        })

    if len(recs) < 5:
        return {"available": False,
                "reason": f"Only {len(recs)} animals have 2+ observations; need at least 5."}

    tab = pd.DataFrame(recs)
    rho_slope, p_slope = stats.spearmanr(tab["slope"], tab["death_age"])
    # age-adjusted level, NOT the raw mean -- see the docstring for why the raw
    # mean is confounded by each animal's observation window
    rho_level, p_level = stats.spearmanr(tab["intercept"], tab["death_age"])
    rho_rawmean, p_rawmean = stats.spearmanr(tab["mean_value"], tab["death_age"])

    informative = bool((p_slope < 0.05) or (p_level < 0.05))
    bits = []
    if p_slope < 0.05:
        direction = "faster-declining animals die earlier" if rho_slope > 0 else \
                    "faster-declining animals die later"
        bits.append(f"slope vs death age: rho = {rho_slope:+.3f}, p = {p_slope:.2g} ({direction})")
    if p_level < 0.05:
        direction = "lower-performing animals die earlier" if rho_level > 0 else \
                    "lower-performing animals die later"
        bits.append(f"age-adjusted level vs death age: rho = {rho_level:+.3f}, "
                    f"p = {p_level:.2g} ({direction})")

    if informative:
        verdict = (
            "DROPOUT IS INFORMATIVE. " + "; ".join(bits) + ". The cohort-mean trajectory is "
            "selected on the quantity being measured, so its decline constant does not "
            "describe an individual fly. Report the balanced-panel and mixed-model "
            "estimates alongside it, and prefer a time-to-death axis for terminal decline."
        )
    else:
        verdict = (
            f"No detectable association between trajectory and death age "
            f"(slope rho = {rho_slope:+.3f}, p = {p_slope:.2g}; level rho = {rho_level:+.3f}, "
            f"p = {p_level:.2g}). Survivor selection is unlikely to be distorting the "
            "cohort mean in this dataset -- though absence of evidence here is weak when "
            f"only {len(tab)} animals contribute."
        )

    return {
        "available": True,
        "informative_dropout": informative,
        "rho_slope_vs_death": float(rho_slope),
        "p_slope_vs_death": float(p_slope),
        "rho_level_vs_death": float(rho_level),
        "p_level_vs_death": float(p_level),
        "rho_rawmean_vs_death": float(rho_rawmean),
        "p_rawmean_vs_death": float(p_rawmean),
        "rawmean_note": (
            "The raw-mean correlation is reported for completeness only. It is "
            "confounded by each animal's observation window (long-lived animals are "
            "observed at later, lower-performing ages) and must not be used as "
            "evidence of informative dropout."
        ),
        "n_animals": int(len(tab)),
        "per_animal": tab,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. TIME-TO-DEATH ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════
def align_to_death(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    death_col: str = "DeathAge",
    out_col: str = "DaysBeforeDeath",
) -> pd.DataFrame:
    """
    Add a time-to-death axis: days remaining until the animal died (0 at death,
    positive before it).

    Motor decline in Drosophila is concentrated in the final days of life
    (Gaitanidis et al. 2019), so this is the axis on which terminal decline is
    actually phase-locked. Two genotypes with different lifespans are NOT
    comparable at the same chronological age -- they are at different points in
    their terminal trajectory -- but they are comparable at the same distance
    from death.
    """
    work = df.copy()
    if death_col not in work.columns or age_col not in work.columns:
        raise ValueError(f"Need both '{age_col}' and '{death_col}' columns.")
    work[out_col] = work[death_col].astype(float) - work[age_col].astype(float)
    return work


def compare_time_axes(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
    death_col: str = "DeathAge",
) -> dict:
    """
    Ask which time axis explains the phenotype better: chronological age, or
    time-to-death.

    Fits both as mixed models with an animal random intercept and compares
    marginal R-squared (variance explained by the fixed time term). If
    time-to-death wins substantially, then age-matched comparison of genotypes
    with different lifespans is confounded, because the same chronological age
    corresponds to different distances from death in each genotype.
    """
    import statsmodels.formula.api as smf

    needed = {id_col, age_col, value_col, death_col}
    missing = needed - set(df.columns)
    if missing:
        return {"available": False, "reason": f"Missing columns: {sorted(missing)}"}

    work = align_to_death(df, id_col=id_col, age_col=age_col, death_col=death_col)
    work = work.dropna(subset=[age_col, value_col, "DaysBeforeDeath", id_col])
    if work[id_col].nunique() < 5:
        return {"available": False, "reason": "Need at least 5 animals."}

    out = {}
    for label, col in (("chronological_age", age_col), ("time_to_death", "DaysBeforeDeath")):
        tmp = work.rename(columns={col: "_x", value_col: "_y", id_col: "_id"})
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                res = smf.mixedlm("_y ~ _x", tmp, groups=tmp["_id"]).fit(reml=False)
            except Exception as exc:
                out[label] = {"error": str(exc)}
                continue
        pred = res.fittedvalues
        resid_var = float(np.var(tmp["_y"] - pred, ddof=1))
        total_var = float(np.var(tmp["_y"], ddof=1))
        out[label] = {
            "slope": float(res.fe_params["_x"]),
            "se": float(res.bse["_x"]),
            "p_value": float(res.pvalues["_x"]),
            "aic": float(res.aic) if np.isfinite(res.aic) else np.nan,
            "variance_explained": float(1.0 - resid_var / total_var) if total_var > 0 else np.nan,
        }

    a = out.get("chronological_age", {})
    d = out.get("time_to_death", {})
    verdict = "Both axes could not be compared."
    better = None
    if "aic" in a and "aic" in d and np.isfinite(a["aic"]) and np.isfinite(d["aic"]):
        better = "time_to_death" if d["aic"] < a["aic"] else "chronological_age"
        delta = abs(d["aic"] - a["aic"])
        if better == "time_to_death":
            verdict = (
                f"Time-to-death fits better than chronological age (dAIC = {delta:.1f}; "
                f"variance explained {d.get('variance_explained', float('nan')):.3f} vs "
                f"{a.get('variance_explained', float('nan')):.3f}). Decline in this cohort is "
                "phase-locked to death rather than to age, so genotypes with different "
                "lifespans should not be compared at matched chronological ages."
            )
        else:
            verdict = (
                f"Chronological age fits better than time-to-death (dAIC = {delta:.1f}). "
                "Age-matched comparison is defensible for this cohort."
            )

    return {"available": True, "axes": out, "better_axis": better, "verdict": verdict}


# ═══════════════════════════════════════════════════════════════════════════
# 5. THE COMBINED REPORT
# ═══════════════════════════════════════════════════════════════════════════
def survivor_bias_report(
    df: pd.DataFrame,
    id_col: str = "FlyID",
    age_col: str = "Age",
    value_col: str = "Value",
    death_col: Optional[str] = "DeathAge",
) -> dict:
    """
    Run every estimator on the same data and quantify how far apart they are.

    The headline number is `slope_bias_ratio`: the survivor-conditional
    (cohort-mean) decline slope divided by the individual-level (mixed-model)
    slope. A ratio of 0.6 means the conventional analysis reports a decline
    only 60% as steep as the flies actually experienced.
    """
    out: dict = {"available": True, "warnings": []}

    surv = survivor_mean_trajectory(df, id_col, age_col, value_col)
    out["survivor_mean"] = surv

    if len(surv) >= 2:
        s_slope, _ = np.polyfit(surv[age_col].to_numpy(float), surv["mean"].to_numpy(float), 1)
        out["slope_survivor_mean"] = float(s_slope)
    else:
        out["slope_survivor_mean"] = np.nan

    panel, panel_meta = balanced_panel_trajectory(df, id_col, age_col, value_col)
    out["balanced_panel"] = panel
    out["balanced_panel_meta"] = panel_meta
    if panel_meta.get("available") and len(panel) >= 2:
        b_slope, _ = np.polyfit(panel[age_col].to_numpy(float), panel["mean"].to_numpy(float), 1)
        out["slope_balanced_panel"] = float(b_slope)
    else:
        out["slope_balanced_panel"] = np.nan

    mm = mixed_slope(df, id_col, age_col, value_col)
    out["mixed_model"] = mm
    out["slope_mixed_model"] = mm.get("slope", np.nan) if mm.get("available") else np.nan

    if death_col and death_col in df.columns:
        out["frailty_test"] = frailty_association_test(df, id_col, age_col, value_col, death_col)
        out["axis_comparison"] = compare_time_axes(df, id_col, age_col, value_col, death_col)
        pm, pm_meta = pattern_mixture_trajectory(df, id_col, age_col, value_col, death_col)
        out["pattern_mixture"] = pm
        out["pattern_mixture_meta"] = pm_meta
        if pm_meta.get("available") and len(pm) >= 2:
            p_slope, _ = np.polyfit(pm[age_col].to_numpy(float), pm["mean"].to_numpy(float), 1)
            out["slope_pattern_mixture"] = float(p_slope)
        else:
            out["slope_pattern_mixture"] = np.nan
    else:
        out["warnings"].append(
            f"No '{death_col}' column: informative-dropout testing, pattern-mixture "
            "reweighting and time-to-death alignment are unavailable. Record the age at "
            "death of every animal -- it is the single most valuable extra column for "
            "trajectory work."
        )
        out["slope_pattern_mixture"] = np.nan

    s_mm = out["slope_mixed_model"]
    s_sv = out["slope_survivor_mean"]
    if np.isfinite(s_mm) and np.isfinite(s_sv) and abs(s_mm) > 1e-12:
        ratio = float(s_sv / s_mm)
        out["slope_bias_ratio"] = ratio
        pct = (1.0 - ratio) * 100.0
        if ratio < 0.85:
            out["verdict"] = (
                f"SURVIVOR BIAS DETECTED. The cohort-mean slope ({s_sv:+.5f}/day) is only "
                f"{ratio * 100:.0f}% of the individual-level slope ({s_mm:+.5f}/day) -- the "
                f"conventional analysis understates decline by {pct:.0f}%. Selective loss of "
                "the worst-performing animals is flattening the observed curve."
            )
        elif ratio > 1.15:
            out["verdict"] = (
                f"The cohort-mean slope ({s_sv:+.5f}/day) is STEEPER than the individual-level "
                f"slope ({s_mm:+.5f}/day), ratio {ratio:.2f}. This is the opposite of classical "
                "survivor bias and usually indicates a cohort or measurement effect rather "
                "than selective mortality; inspect the per-age sample sizes."
            )
        else:
            out["verdict"] = (
                f"Cohort-mean and individual-level slopes agree closely (ratio {ratio:.2f}). "
                "Survivor selection is not materially distorting this trajectory."
            )
    else:
        out["slope_bias_ratio"] = np.nan
        out["verdict"] = "Could not compare slopes (a required estimate was unavailable)."

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 6. GENERATIVE MODEL FOR VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def simulate_frailty_cohort(
    n_flies: int = 300,
    ages: Sequence[float] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
    intercept: float = 0.95,
    mean_slope: float = -0.012,
    sd_slope: float = 0.006,
    frailty_coupling: float = 1.0,
    median_lifespan: float = 45.0,
    noise_sd: float = 0.04,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate a cohort in which frailty drives BOTH faster decline AND earlier
    death -- the structure that creates survivor bias.

    Each fly has its own decline slope s_i ~ N(mean_slope, sd_slope). Its death
    age is drawn so that steeper decliners die earlier, with the strength of
    that coupling set by `frailty_coupling` (0 = no coupling, so no survivor
    bias; 1 = strong coupling). Observations exist only at ages before death.

    Returns a long dataframe with FlyID, Age, Value, DeathAge, and the true
    per-fly slope in `TrueSlope` -- so an estimator can be scored against the
    quantity it is supposed to recover.
    """
    rng = np.random.default_rng(seed)
    ages = np.asarray(ages, dtype=float)

    slopes = rng.normal(mean_slope, sd_slope, size=n_flies)
    z = (slopes - slopes.mean()) / (slopes.std() + 1e-12)   # +z = shallower decline

    # Steeper decline (negative z relative to mean slope) -> shorter life.
    # Weibull baseline lifespan, shifted on the log scale by frailty.
    shape = 4.0
    scale = median_lifespan / (np.log(2.0) ** (1.0 / shape))
    base = rng.weibull(shape, size=n_flies) * scale
    death = base * np.exp(frailty_coupling * 0.25 * z)
    death = np.clip(death, 2.0, None)

    rows = []
    for i in range(n_flies):
        for a in ages:
            if a > death[i]:
                continue
            val = intercept + slopes[i] * a + rng.normal(0.0, noise_sd)
            rows.append({
                "FlyID": f"F{i:04d}",
                "Age": float(a),
                "Value": float(np.clip(val, 0.0, 1.0)),
                "DeathAge": float(death[i]),
                "TrueSlope": float(slopes[i]),
            })
    return pd.DataFrame(rows)
