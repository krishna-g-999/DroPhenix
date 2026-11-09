"""
historic_arms.py
Completed cohorts for the trajectory model's population prior -- real ones from
published experiments, and synthetic ones with known parameters.

WHY THIS EXISTS
---------------
`modules/trajectory.py` predicts a phenotype at ages a cohort was never
measured at, and its whole argument for being trustworthy is the population
layer: a cohort with three timepoints cannot constrain the SHAPE of its own
decline curve, so the shape is borrowed from cohorts that were followed to late
ages. `build_population_prior` estimates that prior by the two-stage
nonlinear-mixed-effects construction, and `predict_trajectory` shrinks a sparse
cohort toward it.

That prior has only ever been built from SIMULATION.
`validation/validate_trajectory_prediction.py` generates 40 synthetic cohorts,
builds the prior from them, and shows that shrinkage improves long-horizon
forecasts. That establishes the estimator works under its own assumptions -- it
cannot establish that real Drosophila cohorts vary the way the model says they
do, and the whole value of the population layer rests on exactly that.

So this module supplies the arms the prior should be built from:

  HISTORIC ARMS -- real, published, longitudinal climbing-vs-age cohorts.
  SYNTHETIC ARMS -- simulated from known parameters, for calibration and for
                    the cases the published data cannot cover.

The distinction is kept in the label of every arm, and
`validate_historic_trajectory_prior.py` never mixes them without saying so: a
prior built from simulation and tested on simulation proves nothing about
flies.

THE PUBLISHED SOURCE
--------------------
Shahrestani P, Wilson JB, Mueller LD, Rose MR (2016) "Patterns of physiological
decline due to age and selection in Drosophila melanogaster". Evolution
70:2550-2561. doi:10.1111/evo.13065. Data doi:10.5061/dryad.qb509, mirrored at
Zenodo record 4954719. Local copy
`reference_datasets/physio_decline/physio_decline.xlsx`, sheet `geotaxis`:
405 rows of the proportion of flies passing the line, per population x sex x
age, ages 1-70 days.

Ten outbred populations: ACO1-5 selected for rapid development, CO1-5 controls.
Crossed with sex that is 20 cohorts, each followed over 15 to 28 ages -- which
is what a completed cohort means here, and far denser than the three-point
cohorts the prior exists to rescue.

The published biological ground truth, which any fitted trajectory must
reproduce: the rapidly-developing ACO populations decline FASTER than the CO
controls.

WHY THERE IS ONLY ONE PUBLISHED SOURCE HERE
-------------------------------------------
`reference_datasets/vaccaro_sing/` holds startle-induced negative geotaxis
across age from Vaccaro et al., which would be a second source, but the files
are GraphPad Prism `.pzf` -- a proprietary binary format with no reader
available here, so the numbers cannot be extracted. That is a limitation of the
archive's format and is recorded rather than worked around. It is also why no
suite in this repository uses that dataset.

TWO TRAPS IN THE PUBLISHED FILE, BOTH HANDLED
---------------------------------------------
1. STRAY WHITESPACE. The `sex` column contains both `'F'` and `'F '`, and both
   `'M'` and `'M '`. Grouping on the raw column therefore splits CO1_F into a
   27-row cohort and a 3-row fragment, and CO1_M into 28 rows and 1. The
   fragments are not distinguishable in printed output, because the trailing
   space is invisible. Unhandled, that both invents cohorts and -- worse --
   inflates the BETWEEN-cohort variance the prior is made of, since a 1-point
   fragment of a cohort looks like a cohort that disagrees with its own twin.
   `validate_real_aging_benchmark.py` already strips these columns and notes
   the whitespace, so this is a known trap in this file and not a new finding;
   it is guarded here for the same reason.

2. ONE MISSING VALUE. `geotaxis` is NaN for CO1, age 68, male -- one cell of
   405. `np.clip` preserves NaN, so an unfiltered arm carries it into the fit
   and `build_population_prior` returns None for the whole cohort set, which
   looks like "a prior cannot be built from real data" rather than "one cell is
   blank". Non-finite points are dropped per arm and counted, and the count is
   printed, because a silently shortened cohort is a cohort whose late ages
   have quietly disappeared.

3. A CEILING AT 1.0. Young flies all pass the line, so the proportion is
   exactly 1.0 for the early ages of most cohorts. The curve families in
   `trajectory.py` are parameterised through a logit, so 1.0 is infinite on the
   fitted scale and must be clipped -- and the clip has to respect the model's
   OWN identifiability bound, `trajectory.LOGIT_BOUND = 6`, which caps the
   representable proportion at expit(6) = 0.99753. Clipping at 0.999 instead,
   as the existing synthetic suite does, places the data outside what the model
   can represent; every fit then rests on the bound, every cohort is excluded
   as unidentified, and the prior cannot be built at all. See `EPS`. The clip is a real modelling decision and
   not a formatting step: a cohort sitting at the ceiling for its first ten
   ages carries no information about how far ABOVE the ceiling its latent
   ability is, so its fitted y0 is pinned by the clip rather than by the data.
   `describe_arms` therefore reports the fraction of each arm that is at the
   ceiling, so a reader can see which fits are leaning on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHYSIO_XLSX = ROOT / "reference_datasets" / "physio_decline" / "physio_decline.xlsx"

#: Proportions are clipped into (EPS, 1 - EPS) before fitting, because the
#: curve families in `trajectory.py` are parameterised through a logit.
#:
#: THE VALUE IS DERIVED, NOT CHOSEN, and the first version of this module got it
#: wrong. `trajectory.LOGIT_BOUND = 6` is the identifiability bound on the
#: logit-scale parameters, so the largest proportion the model can REPRESENT is
#: expit(6) = 0.99753. Clipping at 0.999, as the existing synthetic trajectory
#: suite does, puts the data ABOVE that: the fitted asymptote is then pinned at
#: the bound, `fit_series` flags `at_bounds`, and `build_population_prior`
#: excludes the cohort -- so every one of the twenty real cohorts was excluded
#: and the prior silently could not be built.
#:
#: So the clip is taken from the bound itself, with a small margin so that a
#: fit sitting at the top of the data is not also sitting on the bound.
_LOGIT_MARGIN = 0.25

def _eps_from_bound() -> float:
    from scipy.special import expit
    try:
        from modules import trajectory as _TJ
        bound = float(_TJ.LOGIT_BOUND)
    except Exception:                                      # pragma: no cover
        bound = 6.0
    return float(1.0 - expit(bound - _LOGIT_MARGIN))


EPS = _eps_from_bound()

#: The curve family the real cohorts can actually support.
#:
#: NOT the module default. `exp_plateau` is identified on 0 of the 20 published
#: cohorts: they decline from near 1.0 to 0.25-0.6 and never reach a plateau
#: inside the observed age window, so the asymptote is unidentifiable and
#: `fit_series` correctly refuses it. Measured over all four families --
#: identified cohorts, then mean AICc:
#:
#:     exp_plateau    0/20   -87.7
#:     logistic      13/20  -107.7
#:     gompertz      12/20  -108.0
#:     logit_linear  20/20  -110.2
#:
#: `logit_linear` is both the best-fitting and the only fully identified family,
#: and with two parameters it is identifiable from the three points the sparse
#: case provides. The existing synthetic validation builds its prior in
#: `exp_plateau` -- and its cohorts are SIMULATED FROM `exp_plateau`, so the
#: family was guaranteed to fit there. That is the same pattern as validating an
#: estimator against its own assumptions.
REAL_FAMILY = "logit_linear"

#: The published direction the fitted trajectories must reproduce: populations
#: selected for rapid development age faster than their controls.
PUBLISHED_DIRECTION = ("ACO", "CO")


@dataclass
class Arm:
    """One completed cohort: a phenotype followed across age."""

    label: str
    ages: np.ndarray
    values: np.ndarray                      # proportion passing, in (0, 1)
    origin: str                             # "published" or "synthetic"
    group: str = ""                         # e.g. "ACO" / "CO"
    sex: str = ""
    source: str = ""
    truth: Optional[Dict[str, float]] = None   # synthetic only: the real answer
    n_at_ceiling: int = 0
    n_at_floor: int = 0
    n_missing: int = 0                      # points dropped as non-finite

    @property
    def n_points(self) -> int:
        return int(np.size(self.ages))

    @property
    def age_span(self) -> float:
        return float(np.nanmax(self.ages) - np.nanmin(self.ages))

    @property
    def ceiling_fraction(self) -> float:
        return self.n_at_ceiling / max(self.n_points, 1)

    def as_cohort(self) -> Tuple[np.ndarray, np.ndarray]:
        """The `(ages, values)` pair `build_population_prior` expects."""
        return self.ages, self.values


def _clip(y: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Clip into the open unit interval and report how often that bit."""
    y = np.asarray(y, dtype=float)
    hi = int(np.sum(y >= 1.0 - EPS))
    lo = int(np.sum(y <= EPS))
    return np.clip(y, EPS, 1.0 - EPS), hi, lo


def published_geotaxis_arms(min_points: int = 6,
                            path: Optional[Path] = None) -> List[Arm]:
    """
    The 20 real cohorts from Shahrestani et al. 2016, as completed arms.

    One arm per population x sex. `min_points` drops an arm too short to pin a
    curve down; with the whitespace handled, no real cohort is anywhere near
    that limit (the shortest has 15 ages), so a dropped arm is a signal that
    the file has changed or that the guard has stopped working.
    """
    import pandas as pd

    p = Path(path) if path else PHYSIO_XLSX
    if not p.is_file():
        return []
    g = pd.read_excel(p, sheet_name="geotaxis")

    # TRAP 1. See the module docstring: 'F' and 'F ' are different keys to
    # pandas and identical on screen, so grouping on the raw column splits real
    # cohorts into fragments and inflates the between-cohort variance.
    for c in ("pop", "sex"):
        g[c] = g[c].astype(str).str.strip()

    arms: List[Arm] = []
    for (pop, sex), sub in g.groupby(["pop", "sex"], sort=True):
        sub = sub.sort_values("age")
        ages_all = sub["age"].to_numpy(float)
        vals_all = sub["geotaxis"].to_numpy(float)
        # TRAP 2. One cell of 405 is blank (CO1, age 68, male). np.clip keeps a
        # NaN, which then propagates into the curve fit and makes
        # build_population_prior return None for the entire cohort set -- an
        # error that reads as "real data cannot support a prior" rather than
        # "one cell is empty". Dropped and COUNTED, never silently.
        finite = np.isfinite(ages_all) & np.isfinite(vals_all)
        n_missing = int(np.sum(~finite))
        ages = ages_all[finite]
        vals, n_hi, n_lo = _clip(vals_all[finite])
        if ages.size < int(min_points):
            continue
        arms.append(Arm(
            label=f"{pop}_{sex}", ages=ages, values=vals, origin="published",
            group="ACO" if str(pop).startswith("ACO") else "CO", sex=str(sex),
            source="Shahrestani et al. 2016, Evolution 70:2550-2561 "
                   "(doi:10.1111/evo.13065), sheet `geotaxis`",
            n_at_ceiling=n_hi, n_at_floor=n_lo, n_missing=n_missing))
    return arms


def synthetic_arms(n: int = 40, family: str = "exp_plateau",
                   seed: int = 17, ages: Optional[Sequence[float]] = None,
                   noise_sd: float = 0.04,
                   y0: Tuple[float, float] = (0.90, 0.99),
                   plateau: Tuple[float, float] = (0.15, 0.55),
                   rate: Tuple[float, float] = (0.02, 0.09)) -> List[Arm]:
    """
    Simulated completed arms whose parameters are known by construction.

    These exist for two purposes and no others. They calibrate -- a recovery
    test needs a truth to recover -- and they fill shapes the published data
    does not contain, such as a cohort that declines to a low plateau early.
    They are NOT evidence that real cohorts vary this way, which is the whole
    reason `published_geotaxis_arms` exists, and every arm is labelled
    `origin="synthetic"` so that no suite can pool the two without saying so.

    The default age grid matches the published one in span and density (ages 1
    to 70, every few days), so a prior built from synthetic arms is comparable
    with one built from the real ones rather than differing in design as well.
    """
    rng = np.random.default_rng(int(seed))
    t = (np.asarray(ages, dtype=float) if ages is not None
         else np.arange(1.0, 71.0, 3.0))
    out: List[Arm] = []
    for i in range(int(n)):
        a = float(rng.uniform(*y0))
        c = float(rng.uniform(*plateau))
        k = float(rng.uniform(*rate))
        if family == "exp_plateau":
            mu = c + (a - c) * np.exp(-k * t)
        elif family == "logistic":
            mid = float(rng.uniform(t.min() + 5, t.max() - 5))
            mu = c + (a - c) / (1.0 + np.exp(k * (t - mid)))
        elif family == "logit_linear":
            # A straight decline on the logit scale, which is the family the
            # real cohorts support. The ranges are taken from the prior fitted
            # to the published arms -- intercept about 4.7, slope about -0.10
            # logit units per day -- so a synthetic arm is comparable in SCALE
            # with a real one. Simulating on a different scale would make the
            # comparison in section D a comparison of scales.
            from scipy.special import expit
            b0 = float(rng.uniform(3.5, 6.0))
            b1 = -float(rng.uniform(0.04, 0.18))
            mu = expit(b0 + b1 * t)
            # the describe() keys of this family, so `truth` is comparable
            truth = {"logit_intercept": b0, "logit_slope_per_day": b1}
        else:
            raise ValueError(f"unsupported family for simulation: {family!r}")
        if family != "logit_linear":
            truth = {"y0": a, "plateau": c, "rate_k_per_day": k,
                     "half_life_days": (math.log(2.0) / k if k > 0
                                        else float("nan"))}
        y = mu + rng.normal(0.0, float(noise_sd), size=t.size)
        vals, n_hi, n_lo = _clip(y)
        out.append(Arm(
            label=f"SYN{i:02d}", ages=t.copy(), values=vals, origin="synthetic",
            group="synthetic", source=f"simulated, family={family}, seed={seed}",
            truth=truth, n_at_ceiling=n_hi, n_at_floor=n_lo))
    return out


def arms_to_cohorts(arms: Sequence[Arm]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The `{label: (ages, values)}` mapping `build_population_prior` takes."""
    return {a.label: a.as_cohort() for a in arms}


def sparsify(arm: Arm, n_keep: int = 3,
             max_age: Optional[float] = None) -> Arm:
    """
    The same arm as a SPARSE cohort: its earliest few ages only.

    This is how a completed arm becomes a test case for the thing the
    population layer exists to do. The points kept are the EARLIEST ones, never
    a random subset: a random hold-out across a cohort's own timepoints leaks
    the future into the training set and flatters the model, which
    `trajectory.forward_validate` already refuses to do. Prediction is forward
    in time or it is not prediction.
    """
    ages, vals = arm.ages, arm.values
    if max_age is not None:
        m = ages <= float(max_age)
        ages, vals = ages[m], vals[m]
    k = int(min(n_keep, ages.size))
    return Arm(label=arm.label + f"_sparse{k}", ages=ages[:k], values=vals[:k],
               origin=arm.origin, group=arm.group, sex=arm.sex,
               source=arm.source, truth=arm.truth)


def describe_arms(arms: Sequence[Arm]) -> str:
    """A table of what the arms are, including how much sits at the ceiling."""
    if not arms:
        return "       no arms"
    lines = ["       %-14s %-10s %6s %7s %7s %9s %9s %8s"
             % ("arm", "group", "pts", "age min", "age max", "ceiling", "floor",
                "dropped")]
    for a in arms:
        lines.append("       %-14s %-10s %6d %7.0f %7.0f %8.0f%% %8.0f%% %8d"
                     % (a.label, a.group, a.n_points, np.nanmin(a.ages),
                        np.nanmax(a.ages), 100.0 * a.ceiling_fraction,
                        100.0 * a.n_at_floor / max(a.n_points, 1), a.n_missing))
    tot_missing = sum(a.n_missing for a in arms)
    if tot_missing:
        lines.append("       %d point(s) dropped as non-finite across all arms"
                     % tot_missing)
    pub = sum(1 for a in arms if a.origin == "published")
    syn = len(arms) - pub
    lines.append("       %d arms: %d published, %d synthetic" % (len(arms), pub, syn))
    return "\n".join(lines)


if __name__ == "__main__":
    pub = published_geotaxis_arms()
    print("PUBLISHED ARMS (Shahrestani et al. 2016)")
    print(describe_arms(pub))
    if pub:
        import collections
        by = collections.Counter(a.group for a in pub)
        print("       groups: %s" % dict(by))
        print("       mean ceiling fraction: %.0f%%"
              % (100.0 * float(np.mean([a.ceiling_fraction for a in pub]))))
    print()
    syn = synthetic_arms(6)
    print("SYNTHETIC ARMS (for calibration only)")
    print(describe_arms(syn))
