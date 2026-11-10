"""
DroPhenix — synthetic.py
Unified, literature-calibrated synthetic data generator for ALL FOUR assays,
sharing ONE consistent cohort so that cross-assay / cross-strain drug ranking
(drug_ranking.py) can join them by Genotype x Treatment x Sex.

Cohort
------
Genotypes : Oregon-WT (healthy) and TDP43 (ALS motor-neuron-disease model)
Treatments: Vehicle and Drug
Sexes     : Male, Female
Group sizes are deliberately UNEQUAL to exercise the unequal-N statistics.

Biological design (consistent across assays): the disease model (TDP43) is
impaired relative to wild type on every assay, and the Drug partially rescues
the TDP43 phenotype while having little effect on wild type — the canonical
shape of a disease-selective therapeutic.

Calibration anchors (per assay generator docstring):
  climbing : negative-geotaxis pass fraction; WT ~85-90% (PI ~ +70-80).
  lifespan : Gompertz mortality; WT median ~ 60-70 d, TDP43 shorter.
  Y-maze   : turn bias Beta(4.6,4.6) -> 23.5% extreme (de Bivort 2015);
             alternation above 50% chance for intact memory (Kim 2017).
  DAMS     : 24 h free-running period (Konopka & Benzer 1971); 5-min sleep rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.ymaze import demo_ymaze as _demo_ymaze
from modules.dams import demo_dams as _demo_dams

GENOTYPES = ["Oregon-WT", "TDP43"]
TREATMENTS = ["Vehicle", "Drug"]
SEXES = ["Male", "Female"]


# ──────────────────────────────────────────────────────────────────────────────
# Climbing (negative geotaxis)
# ──────────────────────────────────────────────────────────────────────────────
def generate_climbing(seed: int = 11, n_reps: int = 8,
                       times=(10, 20, 30, 60)) -> pd.DataFrame:
    """
    Climbing assay long format: Genotype, Treatment, Sex, Replicate, Time,
    Count_climbed, n_total_flies.

    Pass fraction rises with time within a trial as flies climb the vial:
        p(t) = asymptote * (1 - exp(-rate * t))
    Wild type reaches ~0.88 (PI ~ +76); TDP43-vehicle plateaus low (~0.34,
    PI ~ -32); the drug rescues TDP43 toward ~0.60 (PI ~ +20). Counts are drawn
    Binomial(n, p). n per vial ~ 18-25 flies; replicate count is unequal.
    """
    rng = np.random.default_rng(seed)
    params = {  # (asymptote pass fraction, rate per second)
        ("Oregon-WT", "Vehicle"): (0.88, 0.12),
        ("Oregon-WT", "Drug"):    (0.89, 0.12),
        ("TDP43", "Vehicle"):     (0.34, 0.06),
        ("TDP43", "Drug"):        (0.60, 0.09),
    }
    rows = []
    for (geno, treat), (asym, rate) in params.items():
        for sex in SEXES:
            sex_mod = 1.0 if sex == "Male" else 0.96
            reps = n_reps + int(rng.integers(-2, 3))   # unequal replicates
            for rep in range(1, reps + 1):
                n = int(rng.integers(18, 26))
                a = np.clip(asym * sex_mod + rng.normal(0, 0.03), 0.02, 0.98)
                for t in times:
                    p = a * (1 - np.exp(-rate * t))
                    count = int(rng.binomial(n, np.clip(p, 0, 1)))
                    rows.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                                 "Replicate": rep, "Time": t,
                                 "Count_climbed": count, "n_total_flies": n})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan (Gompertz mortality) — consolidated here from the app
# ──────────────────────────────────────────────────────────────────────────────
def generate_lifespan(seed: int = 7, end_of_study: float = 90.0) -> pd.DataFrame:
    """
    Event-level lifespan data from a Gompertz hazard h(t)=a*exp(b*t). Disease
    raises baseline mortality; the drug partially rescues. Right-censoring at
    end of study. Columns: Genotype, Treatment, Sex, Time, Event.
    """
    rng = np.random.default_rng(seed)
    groups = {
        ("Oregon-WT", "Vehicle"): (0.0012, 0.085),
        ("Oregon-WT", "Drug"):    (0.0011, 0.083),
        ("TDP43", "Vehicle"):     (0.0060, 0.110),
        ("TDP43", "Drug"):        (0.0028, 0.095),
    }
    rows = []
    for (geno, treat), (a, b) in groups.items():
        for sex in SEXES:
            n = int(rng.integers(45, 60))
            U = rng.uniform(0, 1, n)
            t = (1.0 / b) * np.log(1.0 + (b / a) * (-np.log(1.0 - U)))
            for ti in t:
                if ti >= end_of_study:
                    rows.append((geno, treat, sex, end_of_study, 0))
                else:
                    rows.append((geno, treat, sex, round(float(ti), 1), 1))
    return pd.DataFrame(rows, columns=["Genotype", "Treatment", "Sex", "Time", "Event"])


# ──────────────────────────────────────────────────────────────────────────────
# Y-maze and DAMS — re-exported from their modules (same cohort labels)
# ──────────────────────────────────────────────────────────────────────────────
def generate_ymaze(seed: int = 7) -> pd.DataFrame:
    return _demo_ymaze(seed=seed)


def generate_dams(seed: int = 7) -> pd.DataFrame:
    return _demo_dams(seed=seed)


# ──────────────────────────────────────────────────────────────────────────────
# Unified multi-assay cohort
# ──────────────────────────────────────────────────────────────────────────────
def generate_climbing_aging(seed: int = 11, ages=(5, 15, 25, 35, 45),
                            n_reps: int = 10) -> pd.DataFrame:
    """
    Longitudinal climbing data across AGE for Motor Decline Rate (MDR) analysis.
    The climbing Performance Index declines with age (negative geotaxis is
    age-dependent; Gargano et al. 2005). Oregon-WT declines slowly (k~0.004/day);
    TDP43-Vehicle declines steeply (k~0.030); the Drug slows TDP43 decline
    (k~0.015). Within each age the assay still rises over seconds.

    Columns: Genotype, Treatment, Sex, Replicate, Age, Time, Count_climbed,
    n_total_flies  (Age in days; Time in seconds).
    """
    rng = np.random.default_rng(seed)
    PI0 = 82.0
    decay = {("Oregon-WT", "Vehicle"): 0.008, ("Oregon-WT", "Drug"): 0.008,
             ("TDP43", "Vehicle"): 0.030, ("TDP43", "Drug"): 0.015}
    rows = []
    for (geno, treat), k in decay.items():
        for sex in SEXES:
            sx = 1.0 if sex == "Male" else 0.97
            for age in ages:
                pi_end = PI0 * np.exp(-k * age)              # endpoint PI at this age
                p_end = np.clip((pi_end * sx + 100) / 200.0, 0.02, 0.98)
                for rep in range(1, n_reps + 1):
                    n = int(rng.integers(18, 26))
                    a = np.clip(p_end + rng.normal(0, 0.02), 0.02, 0.98)
                    for t in (10, 20, 30, 60):
                        p = a * (1 - np.exp(-0.12 * t))
                        rows.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                                     "Replicate": rep, "Age": age, "Time": t,
                                     "Count_climbed": int(rng.binomial(n, np.clip(p, 0, 1))),
                                     "n_total_flies": n})
    return pd.DataFrame(rows)


def generate_screen(seed: int = 5, efficacy: dict = None, n_days_dams: int = 7) -> dict:
    """
    Multi-compound screen across all four assays for cross-strain ranking.

    Treatments: Vehicle + several drugs of KNOWN, decreasing efficacy on the
    TDP43 disease model (efficacy rho in [0,1]; rho is the fraction by which the
    drug shifts each TDP43 phenotype from its disease value toward wild type).
    Drugs do not affect wild type. Defaults: Drug_A (0.80) > Drug_B (0.45) >
    Drug_C (0.05). Strains have deliberately UNEQUAL N. The known benefit order
    A > B > C is the ground truth used to validate the ranking engine.
    """
    rng = np.random.default_rng(seed)
    efficacy = efficacy or {"Drug_A": 0.80, "Drug_B": 0.45, "Drug_C": 0.05}
    treatments = ["Vehicle", *efficacy.keys()]

    # per-assay (wild-type value, disease value) of the driving parameter
    P = {
        "climb_asym": (0.88, 0.34),
        "life_a":     (0.0012, 0.0060),   # Gompertz baseline mortality (lower=better)
        "ymaze_rho":  (0.36, 0.00),
        "dams_period": (24.0, 22.5),
    }

    def val(key, geno, treat):
        wt, dis = P[key]
        if geno == "Oregon-WT":
            return wt
        rho = 0.0 if treat == "Vehicle" else efficacy[treat]
        return dis + rho * (wt - dis)

    # unequal N per (geno) — TDP43 deliberately smaller, like real screens
    n_clim = {"Oregon-WT": 8, "TDP43": 4}
    n_life = {"Oregon-WT": 55, "TDP43": 32}
    n_fly = {"Oregon-WT": 12, "TDP43": 6}

    clim, life, ym, dm = [], [], [], []
    for geno in GENOTYPES:
        for treat in treatments:
            for sex in SEXES:
                sx = 1.0 if sex == "Male" else 0.96
                # climbing
                for rep in range(1, n_clim[geno] + 1):
                    n = int(rng.integers(18, 26))
                    a = np.clip(val("climb_asym", geno, treat) * sx + rng.normal(0, 0.03), 0.02, 0.98)
                    for t in (10, 20, 30, 60):
                        p = a * (1 - np.exp(-0.10 * t))
                        clim.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                                     "Replicate": rep, "Time": t,
                                     "Count_climbed": int(rng.binomial(n, np.clip(p, 0, 1))),
                                     "n_total_flies": n})
                # lifespan (Gompertz, fixed b)
                a_mort = val("life_a", geno, treat); b = 0.095
                U = rng.uniform(0, 1, n_life[geno])
                tt = (1 / b) * np.log(1 + (b / a_mort) * (-np.log(1 - U)))
                for ti in tt:
                    life.append((geno, treat, sex, 90.0, 0) if ti >= 90 else
                                (geno, treat, sex, round(float(ti), 1), 1))
                # y-maze
                rho = val("ymaze_rho", geno, treat)
                for k in range(n_fly[geno]):
                    bbias = rng.beta(4.6, 4.6)
                    turns = ["R" if rng.random() < bbias else "L"]
                    for _ in range(199):
                        turns.append(turns[-1] if rng.random() < rho else
                                     ("R" if rng.random() < bbias else "L"))
                    arms = "".join(_arms_from_turns(turns))
                    ym.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                               "FlyID": f"{geno}_{treat}_{sex}_y{k}", "Sequence": arms})
                # DAMS (DD free-running)
                period = val("dams_period", geno, treat)
                rest = 0.42 if sex == "Male" else 0.30
                for k in range(max(4, n_fly[geno] // 2)):
                    per = period + rng.normal(0, 0.15)
                    dm.append(_dams_fly(rng, geno, treat, sex,
                                        f"{geno}_{treat}_{sex}_d{k}", per, rest, n_days_dams))

    return {
        "climbing": pd.DataFrame(clim),
        "lifespan": pd.DataFrame(life, columns=["Genotype", "Treatment", "Sex", "Time", "Event"]),
        "ymaze": pd.DataFrame(ym),
        "dams": pd.concat(dm, ignore_index=True),
    }


def _arms_from_turns(turns):
    from modules.ymaze import turns_to_arm_sequence
    return turns_to_arm_sequence(turns, start="A")


def _dams_fly(rng, geno, treat, sex, fid, period_h, rest_frac, n_days):
    minutes = np.arange(n_days * 1440, dtype=float)
    pos = (minutes % (period_h * 60.0)) / (period_h * 60.0)
    active = pos < (1.0 - rest_frac)
    u = np.clip(pos / (1.0 - rest_frac), 0, 1)
    rate = (1.3 * np.exp(-((u - 0.10) ** 2) / (2 * 0.10 ** 2)) +
            1.6 * np.exp(-((u - 0.90) ** 2) / (2 * 0.12 ** 2)) + 0.25) * active
    counts = rng.poisson(np.maximum(rate, 0) * 6.0).astype(float)
    counts[~active] = 0
    return pd.DataFrame({"Genotype": geno, "Treatment": treat, "Sex": sex,
                         "FlyID": fid, "Time_min": minutes, "Activity": counts})


def generate_all(seed: int = 7) -> dict:
    """
    Return one coherent cohort across all four assays as a dict of DataFrames:
        {'climbing', 'lifespan', 'ymaze', 'dams'}
    All share Genotype in {Oregon-WT, TDP43}, Treatment in {Vehicle, Drug},
    Sex in {Male, Female}. Feed straight into drug_ranking.rank_compounds().
    """
    return {
        "climbing": generate_climbing(seed=seed),
        "lifespan": generate_lifespan(seed=seed),
        "ymaze": generate_ymaze(seed=seed),
        "dams": generate_dams(seed=seed),
    }


# ── the demonstration study ──────────────────────────────────────────────────
#: Doses in mM. 0 is the untreated control and is never dropped: it pins down
#: the upper asymptote of a dose-response curve.
#: Seven doses per compound INCLUDING ZERO, spanning 0.1 to 30 mM.
#:
#: The design is chosen so that the Dose-Response page can show a user what a
#: GOOD design looks like, because that is what a demonstration is for. With
#: four non-zero doses and no control the page was right to object, and it
#: objected eight times: no dose-0 anchor, under two levels per decade, the
#: asymptotes extrapolated, the doses not straddling the inflection, and no
#: LC50 formable because mortality never reached half. Every one of those is a
#: real property of a weak design, and a demo that triggers all of them teaches
#: the warnings rather than the analysis.
#:
#: Zero is included per compound as that compound's OWN vehicle vials -- a real
#: screen runs a control alongside each compound, and generating separate vials
#: for each avoids the objection that answered this the first time round: a
#: single shared control arm placed in both compounds' dose series would count
#: the same vials twice in every categorical comparison.
DEMO_DOSES = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
DEMO_AGES = (5, 20, 35, 50)
#: Five timepoints rather than seven: the dose axis grew from four levels to
#: seven, and the frame has to stay small enough to load and analyse in a few
#: seconds. Five still resolves the climbing curve's rise and its plateau.
DEMO_TIMES = (10, 20, 30, 60, 120)
#: Zeitgeber time, one per replicate, SIX HOURS APART and the SAME at every age.
#:
#: Two constraints pull against each other here and the balance matters more
#: than the coverage. A 24-hour cosinor cannot be identified from testing times
#: spanning a third of the cycle, which is what ZT2 / ZT6 / ZT10 gave -- eight
#: hours of coverage, and the Time-of-Day page correctly produced nothing. But
#: spreading six times across the (replicate, age) combinations instead made ZT
#: a FUNCTION OF AGE, so the fitted rhythm would have absorbed part of the age
#: decline: a confound, and precisely the kind the Design Check page exists to
#: catch. Shipping a demo with a confounded design would teach the mistake.
#:
#: Four times six hours apart, assigned by replicate and held constant across
#: ages, satisfies both: the cycle is covered, the cosinor has a residual degree
#: of freedom, and ZT is orthogonal to age, genotype, dose and sex by
#: construction.
#: Each vial is tested at ALL FOUR times, within each age -- a repeated-measures
#: circadian design rather than one time per vial.
#:
#: The page needs it: with one time per vial no subject has three distinct times
#: of day and no population cosinor can be fitted, which is what it said. And it
#: is the right design independently: a rhythm is a within-animal quantity, and
#: testing different vials at different times confounds the rhythm with
#: between-vial variation.
#:
#: The ORDER of the four is shuffled per vial, so the time of day is orthogonal
#: to the trial number. Without that, ZT19 would always be the fourth trial of
#: the day and the circadian effect would be inseparable from fatigue -- which
#: is the confound `circadian_climbing` exists to disentangle, so handing it a
#: design where the two are aliased would make the page unable to do its job.
DEMO_ZTS = (1, 7, 13, 19)
DEMO_REPS = (1, 2)

#: Habituation: each successive trial within a day is slightly worse. Real, and
#: the reason the page checks for it before believing a rhythm.
DEMO_FATIGUE_PER_TRIAL = 0.012

#: Background mortality during the assay, and how steeply it rises with dose.
#:
#: A compound screened to 10 mM is screened past the point where it is
#: harmless. Flies dead at scoring are not flies that failed to climb, and a
#: proportion computed over the flies LOADED rather than the flies ALIVE
#: understates performance by exactly that fraction -- the confound Abbott's
#: correction exists for, and the reason `dose_response` carries a
#: mortality-adjusted path at all. Set so that mortality is negligible where
#: the compounds work and appreciable only at the top dose, which is what makes
#: a therapeutic window a real comparison rather than a decoration.
DEMO_BASE_MORTALITY = 0.01
#: Set so that survival actually falls below half at the top dose: with the
#: previous 14 mM and a 45% ceiling, mortality peaked at 33% and the page
#: correctly refused to report an LC50 at all.
DEMO_LC50_MM = 9.0
DEMO_MAX_MORTALITY = 0.92

#: Half-maximal rescue, in mM, and the fraction of the deficit each compound can
#: recover at saturation. Spermine is the effective one; Pantothenate is a
#: weak comparator, so that ranking two compounds is a real comparison.
DEMO_COMPOUNDS = {
    "Spermine":     {"ec50": 0.6, "emax": 0.75},
    "Pantothenate": {"ec50": 2.5, "emax": 0.45},
}



def _emit(rows, rng, geno, sex, treat, comp, dose, rep, vial, da, age, zt,
          trial, rescued, vial_effect):
    """One vial's climbing time course for one test, appended to `rows`."""
    import numpy as np

    pmax, k = rescued(geno, age, comp, dose)
    # A genuine 24-hour rhythm with its peak in the early morning, which is what
    # the cosinor on the Time-of-Day page is there to recover.
    pmax *= 1.0 + 0.06 * np.cos(2 * np.pi * (zt - 3) / 24.0)
    # Habituation: each successive trial of the day is a little worse. Shuffled
    # against the time of day, so the page can separate the two.
    pmax *= 1.0 - DEMO_FATIGUE_PER_TRIAL * (trial - 1)
    pmax = float(np.clip(pmax + vial_effect, 0.02, 0.99))
    if sex == "Female":
        pmax = float(np.clip(pmax * 1.04, 0.02, 0.99))
    n = 20
    # Flies dead at scoring. Rises with dose and with age; the climbing counts
    # below are over the flies ALIVE, so a caller that ignores n_dead and
    # divides by n is making exactly the error Abbott's correction removes.
    p_dead = DEMO_BASE_MORTALITY + 0.02 * (age / 50.0)
    if dose > 0:
        p_dead += DEMO_MAX_MORTALITY * dose / (DEMO_LC50_MM + dose)
    n_dead = int(rng.binomial(n, float(np.clip(p_dead, 0.0, 0.95))))
    n_alive = max(n - n_dead, 1)
    for t in DEMO_TIMES:
        p = float(np.clip(pmax * (1.0 - np.exp(-k * t)), 0.0, 1.0))
        # Drawn over the LIVING flies: a dead fly cannot climb, and counting it
        # as a fly that chose not to is the error the correction undoes.
        rows.append({
            "Genotype": geno, "Sex": sex, "Treatment": treat,
            "Replicate": rep, "Time": int(t),
            "Count": int(rng.binomial(n_alive, p)), "n": n,
            "n_dead": n_dead, "n_alive": n_alive,
            "Age": int(age), "Dose": float(dose), "Compound": comp or "Vehicle",
            "TimeOfDay": int(zt), "ZT": int(zt),
            "SessionIndex": int(DEMO_AGES.index(age) + 1),
            "TrialIndex": int(trial),
            "VialID": vial, "DeathAge": round(da, 1),
            "ExperimentID": "DEMO_TDP43_SCREEN",
        })


def generate_demo_study(seed: int = 19) -> "pd.DataFrame":
    """
    One coherent drug screen that reaches every page of the app.

    A TDP-43 model of ALS with two compounds across four doses, both sexes, four
    ages, three vials per cell and a Zeitgeber time per vial -- climbing counts
    out of n = 20, drawn from a binomial.

    See the module note above for the biology encoded and for what this is not:
    it is SIMULATED, nothing in it is evidence about flies, and no number from
    it belongs in a manuscript.

    Returns the canonical long format plus the optional design columns the app
    looks for -- `Age`, `Dose`, `Compound`, `TimeOfDay`, `VialID`, `DeathAge` --
    all of which are already in `data_parser.DESIGN_KEY_COLS`, so rows that
    differ only by dose or age are not mistaken for duplicates.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(int(seed))

    def baseline(geno: str, age: int):
        """(ceiling, rate) for an untreated fly of this genotype at this age."""
        if geno == "Oregon-R":
            # Wild type: a high ceiling that erodes gently with age.
            pmax = 0.95 - 0.0022 * age
            k = 0.055 - 0.00012 * age
        else:
            # TDP43: lower to begin with and falling about three times faster,
            # which is the difference the Motor Decline page exists to measure.
            pmax = 0.80 - 0.0068 * age
            k = 0.040 - 0.00026 * age
        return float(np.clip(pmax, 0.05, 0.99)), float(max(k, 0.004))

    def rescued(geno, age, compound, dose):
        """The ceiling after a compound has recovered part of the deficit."""
        pmax, k = baseline(geno, age)
        if geno == "Oregon-R" or compound in (None, "Vehicle") or dose <= 0:
            # A wild-type fly has little deficit to rescue, so a compound moves
            # it hardly at all -- which is what makes the disease model's
            # response interpretable rather than a general stimulant effect.
            return pmax, k
        wt_max, wt_k = baseline("Oregon-R", age)
        deficit = max(wt_max - pmax, 0.0)
        c = DEMO_COMPOUNDS[compound]
        # Hill function with slope 1: the standard single-site occupancy curve.
        frac = c["emax"] * dose / (c["ec50"] + dose)
        return float(min(pmax + deficit * frac, 0.99)), float(k + (wt_k - k) * frac)

    def death_age(geno, compound, dose, rng):
        """
        When this vial's cohort dies.

        Untreated TDP43 vials die earliest, so they drop out of the late ages
        and the survivors there are the healthiest -- the survivorship bias the
        demo is meant to contain rather than hide.
        """
        if geno == "Oregon-R":
            mu = 68.0
        else:
            mu = 42.0
            if compound not in (None, "Vehicle") and dose > 0:
                c = DEMO_COMPOUNDS[compound]
                mu += 16.0 * (dose / (c["ec50"] + dose))
        return float(max(12.0, rng.normal(mu, 7.0)))

    # Each compound carries its OWN vehicle vials at dose 0, which is what a
    # screen actually runs and what lets the dose-response fit anchor its upper
    # asymptote on measured data instead of extrapolating to it.
    cells = []
    for comp in DEMO_COMPOUNDS:
        for d in DEMO_DOSES:
            treat = "Untreated" if d == 0 else comp
            cells.append((treat, comp, float(d)))

    rows = []
    for geno in ("Oregon-R", "TDP43"):
        for sex in ("Male", "Female"):
            for treat, comp, dose in cells:
                for rep in DEMO_REPS:
                    vial = f"{geno}_{sex}_{treat}_{dose:g}_r{rep}"
                    da = death_age(geno, comp, dose, rng)
                    # A vial's own quality: some vials simply climb better than
                    # others, which is the between-vial variance the
                    # hierarchical page exists to separate from the treatment.
                    # Kept small enough that the WEAKER compound's real effect
                    # is still visible: at sd 0.035 it was swamped, and a demo
                    # in which the weak comparator looks like nothing teaches
                    # that a weak effect cannot be seen.
                    vial_effect = float(rng.normal(0.0, 0.022))
                    for age in DEMO_AGES:
                        if age >= da:
                            continue        # this vial is no longer alive
                        # The day's four tests, in a shuffled order so that the
                        # time of day is not aliased with the trial number.
                        order = list(DEMO_ZTS)
                        rng.shuffle(order)
                        for trial, zt in enumerate(order, start=1):
                            _emit(rows, rng, geno, sex, treat, comp, dose, rep,
                                  vial, da, age, zt, trial, rescued,
                                  vial_effect)
    return pd.DataFrame(rows)
