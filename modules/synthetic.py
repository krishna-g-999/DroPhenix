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
