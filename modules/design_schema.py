"""
Extended experimental-design schema for longitudinal, chronobiological and
dose-response Drosophila designs.

WHY THIS MODULE EXISTS
----------------------
The original DroPhenix long-format schema is
    Genotype, Sex, Treatment, Replicate, Time, Count, n  (+ optional Age)
which is sufficient for a single cross-sectional climbing assay. It cannot
express any of the following, all of which change which statistical model is
valid:

  * which physical VIAL a row came from  -> the vial is the experimental unit;
    flies inside it are not independent (design effect 1 + (m-1)*ICC)
  * which individual FLY a row came from -> enables true repeated measures
  * WHEN in the day the assay ran        -> Drosophila is crepuscular; climbing
    varies with circadian phase
  * how many times that fly/vial has     -> repeated negative geotaxis causes
    already been tested                     within-session fatigue and
                                            across-session habituation, both of
                                            which mimic ageing
  * what DOSE of what COMPOUND           -> dose-response needs a numeric dose
  * which EXPERIMENT/BATCH               -> required to borrow historical
                                            controls and to model batch effects

This module adds those columns as *optional* extensions. Nothing here is
required; a legacy file parses exactly as before. What the module guarantees is
that when the information IS present it is recognised, canonicalised, and
surfaced to the analysis layer as a machine-readable DesignSpec, so that the
statistics chosen downstream match the design that was actually run.

DESIGN PRINCIPLE
----------------
The module never silently repairs an ambiguous design. Where a design cannot
support an analysis (e.g. three categorical times of day cannot identify a
24 h cosine from one fly), it says so explicitly in `DesignSpec.warnings`
rather than producing a number that looks fine.

References for the design facts encoded here:
  Gargano JW, Martin I, Bhandari P, Grotewiel MS (2005) Rapid iterative negative
    geotaxis (RING): a new method for assessing age-related locomotor decline in
    Drosophila. Exp Gerontol 40:386-395.
  Killeen PR (2005) / standard cluster-sampling theory for the design effect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── CANONICAL EXTENDED COLUMNS ──────────────────────────────────────────────
# canonical name -> set of accepted lower-cased aliases
DESIGN_ALIASES: Dict[str, set] = {
    # --- identity / nesting -------------------------------------------------
    "FlyID": {
        "flyid", "fly_id", "fly", "animal", "animal_id", "individual",
        "individual_id", "subject", "subject_id",
    },
    "VialID": {
        "vialid", "vial_id", "vial", "tube", "tube_id", "chamber", "chamber_id",
        "container", "cohort_vial",
    },
    "ExperimentID": {
        "experimentid", "experiment_id", "experiment", "expt", "expt_id",
        "study", "study_id", "run", "run_id", "assay_id",
    },
    "Batch": {
        "batch", "batch_id", "food_batch", "cohort", "cohort_id", "block",
    },
    # --- time axes ----------------------------------------------------------
    "Age": {"age", "age_days", "ageday", "age_day", "agedays", "age_d"},
    "TimeOfDay": {
        "timeofday", "time_of_day", "tod", "clock_time", "clocktime",
        "assay_time", "time_of_assay", "daytime", "session_time",
    },
    "ZT": {"zt", "zeitgeber", "zeitgeber_time", "zt_hours", "zt_h"},
    "Date": {"date", "assay_date", "test_date", "day_of_experiment"},
    "SessionIndex": {
        "sessionindex", "session_index", "session", "session_no", "session_n",
        "testing_session", "n_prior_tests", "test_number",
    },
    "TrialIndex": {
        "trialindex", "trial_index", "trial_no", "trial_n", "tap",
        "tap_number", "iteration", "within_session_trial",
    },
    # --- pharmacology / toxicology -----------------------------------------
    "Compound": {
        "compound", "drug", "toxin", "chemical", "agent", "substance",
        "compound_name", "drug_name",
    },
    "Dose": {
        "dose", "concentration", "conc", "dose_mm", "dose_um", "dose_level",
        "exposure", "dose_value",
    },
    "DoseUnit": {"doseunit", "dose_unit", "conc_unit", "concentration_unit", "units"},
    # --- environment --------------------------------------------------------
    "Temperature": {"temperature", "temp", "temp_c", "temperature_c"},
    "RackPosition": {"rackposition", "rack_position", "position", "slot", "rack_slot"},
}

# Columns that, if present, must be treated as categorical labels (never numeric)
_LABEL_LIKE = {"FlyID", "VialID", "ExperimentID", "Batch", "Compound", "DoseUnit", "RackPosition"}

# Representative Zeitgeber hours for categorical times of day, used ONLY when no
# clock time is supplied. These are conventional midpoints of the common
# morning/afternoon/evening testing windows under a 12:12 LD cycle with
# lights-on at ZT0. They are stated explicitly because assuming them silently
# would fabricate precision the data does not have.
CATEGORICAL_TOD_ZT: Dict[str, float] = {
    "morning": 2.0,
    "am": 2.0,
    "early": 2.0,
    "dawn": 0.5,
    "midday": 6.0,
    "noon": 6.0,
    "afternoon": 8.0,
    "pm": 8.0,
    "evening": 11.0,
    "dusk": 11.5,
    "night": 18.0,
}


@dataclass
class DesignSpec:
    """Machine-readable description of the design a dataframe actually encodes."""

    # what identity information exists
    has_fly_id: bool = False
    has_vial_id: bool = False
    has_experiment_id: bool = False
    has_batch: bool = False
    # what time axes exist
    has_age: bool = False
    has_time_of_day: bool = False
    has_zt: bool = False
    time_of_day_is_categorical: bool = False
    has_session_index: bool = False
    has_trial_index: bool = False
    # pharmacology
    has_compound: bool = False
    has_dose: bool = False
    # derived design classification
    is_longitudinal: bool = False          # >=2 ages
    is_repeated_measures: bool = False     # same FlyID (or VialID) seen at >=2 ages/times
    repeated_unit: Optional[str] = None    # "FlyID" | "VialID" | None
    is_nested: bool = False                # flies within vials
    n_ages: int = 0
    n_times_of_day: int = 0
    n_doses: int = 0
    n_experiments: int = 0
    # the unit of independent replication that downstream models must cluster on
    cluster_unit: Optional[str] = None
    mean_flies_per_cluster: float = float("nan")

    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = []
        if self.is_longitudinal:
            bits.append(f"longitudinal ({self.n_ages} ages)")
        if self.is_repeated_measures:
            bits.append(f"repeated-measures on {self.repeated_unit}")
        if self.has_time_of_day:
            kind = "categorical" if self.time_of_day_is_categorical else "clock-time"
            bits.append(f"time-of-day ({self.n_times_of_day} levels, {kind})")
        if self.has_dose:
            bits.append(f"dose-response ({self.n_doses} doses)")
        if self.is_nested:
            bits.append("nested (flies within vials)")
        if self.n_experiments > 1:
            bits.append(f"{self.n_experiments} experiments")
        if not bits:
            return "cross-sectional, single timepoint"
        return "; ".join(bits)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["summary"] = self.summary()
        return d


# ─── TIME-OF-DAY PARSING ─────────────────────────────────────────────────────
_CLOCK_RE = re.compile(r"^\s*(\d{1,2})[:.h]\s*(\d{1,2})?\s*(am|pm)?\s*$", re.IGNORECASE)


def parse_clock_to_hours(value) -> Optional[float]:
    """
    Convert a time-of-day value to decimal hours on a 24 h clock.

    Accepts: 9, 9.5, "09:00", "9:30", "9.30", "9h30", "9:00 AM", "5:00 pm".
    Returns None when the value is not a clock time (e.g. "Morning") so the
    caller can fall back to the categorical map -- deliberately, so that a
    category is never mistaken for a measured time.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    # Already numeric: interpret as decimal hours if in range
    if isinstance(value, (int, float, np.integer, np.floating)):
        h = float(value)
        return h if 0.0 <= h < 24.0 else None

    s = str(value).strip()
    if not s:
        return None

    m = _CLOCK_RE.match(s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = (m.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour + minute / 60.0
        return None

    # bare number as string, e.g. "14"
    try:
        h = float(s)
        return h if 0.0 <= h < 24.0 else None
    except ValueError:
        return None


def categorical_to_zt(value) -> Optional[float]:
    """Map a categorical time-of-day label to its conventional ZT midpoint."""
    if value is None:
        return None
    key = str(value).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return CATEGORICAL_TOD_ZT.get(key)


def derive_zt(
    df: pd.DataFrame,
    lights_on_hour: float = 8.0,
) -> Tuple[pd.Series, bool, List[str]]:
    """
    Derive Zeitgeber time (hours since lights-on) from TimeOfDay or ZT.

    Parameters
    ----------
    df : dataframe containing 'TimeOfDay' and/or 'ZT'
    lights_on_hour : clock hour at which the incubator lights come on.
        Only used to convert clock times to ZT. The default of 08:00 is the
        common laboratory convention but MUST be set to the actual value for
        the facility -- getting it wrong rotates every phase estimate.

    Returns
    -------
    (zt_series, is_categorical, warnings)
        zt_series is float hours in [0, 24) with NaN where undeterminable.
    """
    warnings: List[str] = []

    if "ZT" in df.columns:
        zt = pd.to_numeric(df["ZT"], errors="coerce")
        if zt.notna().any():
            return zt.mod(24.0), False, warnings

    if "TimeOfDay" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float), False, warnings

    raw = df["TimeOfDay"]
    clock = raw.map(parse_clock_to_hours)
    n_clock = int(clock.notna().sum())

    if n_clock == len(raw) and n_clock > 0:
        zt = (clock.astype(float) - float(lights_on_hour)) % 24.0
        return zt, False, warnings

    # Fall back to categorical labels for the rows that are not clock times
    cat = raw.map(categorical_to_zt)
    n_cat = int(cat.notna().sum())

    if n_clock == 0 and n_cat > 0:
        warnings.append(
            "TimeOfDay is categorical (e.g. Morning/Afternoon/Evening). Conventional "
            f"Zeitgeber midpoints were assigned ({', '.join(f'{k}=ZT{v:g}' for k, v in sorted(set(CATEGORICAL_TOD_ZT.items()), key=lambda kv: kv[1]) if k in set(str(x).strip().lower() for x in raw.dropna().unique()))}). "
            "Circadian phase (acrophase) estimated from category midpoints inherits "
            "the width of each testing window; record actual clock times to remove "
            "this assumption."
        )
        return cat.astype(float), True, warnings

    if n_clock > 0 and n_cat > 0:
        merged = clock.astype(float)
        merged = merged.where(merged.notna(), np.nan)
        zt = (merged - float(lights_on_hour)) % 24.0
        zt = zt.where(zt.notna(), cat.astype(float))
        warnings.append(
            f"TimeOfDay mixes clock times ({n_clock} rows) and categories ({n_cat} rows). "
            "Categories were mapped to conventional Zeitgeber midpoints; the two are "
            "not equally precise and this is a heterogeneous predictor."
        )
        return zt, False, warnings

    warnings.append(
        "TimeOfDay values could not be interpreted as clock times or as recognised "
        f"categories: {sorted(set(str(v) for v in raw.dropna().unique()))[:8]}. "
        "Time-of-day analysis is unavailable."
    )
    return pd.Series(np.nan, index=df.index, dtype=float), False, warnings


# ─── STANDARDISATION ─────────────────────────────────────────────────────────
def standardize_design_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rename any recognised design-column aliases to their canonical names and
    coerce dtypes. Unrecognised columns are left untouched.

    Returns (df, warnings). Never raises on unexpected content.
    """
    warnings: List[str] = []
    df = df.copy()

    # Build alias -> canonical lookup, skipping columns already canonical
    existing_canonical = {c for c in df.columns if c in DESIGN_ALIASES}
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        if col in DESIGN_ALIASES:          # already canonical
            continue
        cl = str(col).strip().lower().replace(" ", "_")
        for canonical, aliases in DESIGN_ALIASES.items():
            if canonical in existing_canonical:
                continue                    # don't create a duplicate
            if cl in aliases:
                rename_map[col] = canonical
                existing_canonical.add(canonical)
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # dtype coercion
    for col in ("Age", "Dose", "SessionIndex", "TrialIndex", "Temperature", "ZT"):
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = int(coerced.isna().sum() - df[col].isna().sum())
            if n_bad > 0:
                warnings.append(
                    f"{n_bad} value(s) in '{col}' were not numeric and became NaN."
                )
            df[col] = coerced

    for col in _LABEL_LIKE:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    if "Date" in df.columns:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        if parsed.notna().any():
            df["Date"] = parsed

    return df, warnings


# ─── DESIGN INFERENCE ────────────────────────────────────────────────────────
def describe_design(df: pd.DataFrame, lights_on_hour: float = 8.0) -> DesignSpec:
    """
    Inspect a standardised dataframe and report what design it encodes, what
    the unit of independent replication is, and what analyses it can and cannot
    support.

    This is deliberately conservative: when identity information is missing it
    reports the coarser (safer) clustering unit rather than assuming
    independence.
    """
    spec = DesignSpec()
    cols = set(df.columns)

    spec.has_fly_id = "FlyID" in cols and df["FlyID"].notna().any()
    spec.has_vial_id = "VialID" in cols and df["VialID"].notna().any()
    spec.has_experiment_id = "ExperimentID" in cols and df["ExperimentID"].notna().any()
    spec.has_batch = "Batch" in cols and df["Batch"].notna().any()
    spec.has_age = "Age" in cols and df["Age"].notna().any()
    spec.has_session_index = "SessionIndex" in cols and df["SessionIndex"].notna().any()
    spec.has_trial_index = "TrialIndex" in cols and df["TrialIndex"].notna().any()
    spec.has_compound = "Compound" in cols and df["Compound"].notna().any()
    spec.has_dose = "Dose" in cols and df["Dose"].notna().any()

    if spec.has_age:
        spec.n_ages = int(df["Age"].dropna().nunique())
        spec.is_longitudinal = spec.n_ages >= 2

    if spec.has_dose:
        spec.n_doses = int(df["Dose"].dropna().nunique())

    if spec.has_experiment_id:
        spec.n_experiments = int(df["ExperimentID"].dropna().nunique())
    elif spec.has_batch:
        spec.n_experiments = int(df["Batch"].dropna().nunique())

    # time of day
    zt, is_cat, tod_warn = derive_zt(df, lights_on_hour=lights_on_hour)
    spec.warnings.extend(tod_warn)
    if zt.notna().any():
        spec.has_time_of_day = True
        spec.has_zt = "ZT" in cols and df["ZT"].notna().any()
        spec.time_of_day_is_categorical = is_cat
        spec.n_times_of_day = int(zt.dropna().round(3).nunique())

    # ── unit of independent replication ─────────────────────────────────────
    # The vial is the experimental unit whenever flies share one. FlyID alone
    # does NOT make flies independent if they were housed together.
    if spec.has_vial_id:
        spec.cluster_unit = "VialID"
    elif "Replicate" in cols:
        spec.cluster_unit = "Replicate"
        spec.notes.append(
            "No VialID column; 'Replicate' is being used as the clustering unit. "
            "This is correct only if one Replicate == one physical vial."
        )
    elif spec.has_fly_id:
        spec.cluster_unit = "FlyID"
        spec.warnings.append(
            "No vial or replicate identifier. Clustering falls back to FlyID, which "
            "assumes flies were housed and handled independently -- rarely true. "
            "Between-vial variance cannot be estimated and standard errors will be "
            "too small if flies shared vials."
        )
    else:
        spec.cluster_unit = None
        spec.warnings.append(
            "No vial, replicate or fly identifier: every row is treated as an "
            "independent observation. If rows are flies from shared vials, all "
            "p-values from this dataset are anticonservative."
        )

    spec.is_nested = bool(spec.has_fly_id and spec.has_vial_id)

    # mean cluster size (flies per cluster) where derivable
    if spec.cluster_unit and spec.cluster_unit in cols:
        if "n" in cols:
            per = df.groupby(spec.cluster_unit, dropna=True)["n"].mean()
            if len(per):
                spec.mean_flies_per_cluster = float(per.mean())
        elif spec.has_fly_id:
            per = df.groupby(spec.cluster_unit, dropna=True)["FlyID"].nunique()
            if len(per):
                spec.mean_flies_per_cluster = float(per.mean())

    # ── repeated measures ───────────────────────────────────────────────────
    # A unit is repeatedly measured if the same identifier appears at more than
    # one age OR more than one time of day.
    occasion_cols = [c for c in ("Age",) if c in cols and df[c].notna().any()]
    tmp = df.copy()
    if zt.notna().any():
        tmp["_ZT"] = zt
        occasion_cols.append("_ZT")
    if "Date" in cols and df["Date"].notna().any():
        occasion_cols.append("Date")

    for unit in ("FlyID", "VialID"):
        if unit in cols and df[unit].notna().any() and occasion_cols:
            occ = tmp.groupby(unit, dropna=True)[occasion_cols].apply(
                lambda g: g.drop_duplicates().shape[0]
            )
            if len(occ) and float(occ.max()) >= 2:
                spec.is_repeated_measures = True
                spec.repeated_unit = unit
                break

    # ── design warnings that change which analysis is valid ────────────────
    if spec.is_repeated_measures and not (spec.has_session_index or spec.has_trial_index):
        spec.warnings.append(
            f"The same {spec.repeated_unit} is measured on multiple occasions but no "
            "SessionIndex/TrialIndex is recorded. Without a testing-order covariate, any "
            "order effect is mathematically indistinguishable from an ageing or treatment "
            "effect. Note that Gargano et al. (2005, Exp Gerontol 40:386-395) reported the "
            "opposite of a problem -- in wild-type flies they found RING performance was "
            "'not influenced by the density of flies being tested, the time of day, or "
            "repeated testing'. That null was established in wild-type animals under one "
            "set of conditions; recording the order costs nothing and is what allows the "
            "null to be checked rather than assumed for a new genotype or protocol."
        )

    if spec.has_time_of_day and spec.n_times_of_day < 3:
        spec.warnings.append(
            f"Only {spec.n_times_of_day} distinct time(s) of day. A 24 h cosinor has three "
            "parameters (MESOR, amplitude, acrophase), so at least three well-separated "
            "times are required to identify a rhythm, and more to estimate it with any "
            "precision."
        )

    if spec.has_time_of_day and spec.n_times_of_day == 3:
        spec.notes.append(
            "Three times of day exactly identify a 24 h cosine with zero residual "
            "degrees of freedom per unit. Amplitude and acrophase are therefore only "
            "estimable by pooling across units (population-mean cosinor)."
        )

    if spec.has_dose and spec.n_doses < 4:
        spec.warnings.append(
            f"Only {spec.n_doses} dose level(s). A four-parameter log-logistic curve cannot "
            "be identified from fewer than four doses, and EC50 intervals from fewer than "
            "five doses spanning the response are unreliable."
        )

    if spec.n_experiments > 1 and not (spec.has_experiment_id or spec.has_batch):
        spec.warnings.append(
            "Multiple experiments appear to be pooled without an ExperimentID/Batch "
            "column. Between-experiment variance cannot be separated from treatment "
            "effects."
        )

    return spec


# ─── CONFOUNDING DIAGNOSTICS ─────────────────────────────────────────────────
def check_confounding(
    df: pd.DataFrame,
    factor: str,
    nuisance: str,
) -> dict:
    """
    Test whether a design factor is confounded with a nuisance variable by
    checking the cross-tabulation for structural imbalance.

    Complete confounding (each level of `factor` occurring at exactly one level
    of `nuisance`) makes the two effects mathematically inseparable -- no
    statistical method can rescue it, so it must be caught before analysis, not
    after.

    Returns a dict with `confounded` (bool), `severity` in [0, 1] where 1 is
    complete confounding, the crosstab, and a plain-language verdict.
    """
    if factor not in df.columns or nuisance not in df.columns:
        return {
            "available": False,
            "reason": f"Need both '{factor}' and '{nuisance}' in the data.",
        }

    sub = df[[factor, nuisance]].dropna()
    if sub.empty:
        return {"available": False, "reason": "No rows with both values present."}

    tab = pd.crosstab(sub[factor], sub[nuisance])
    n_factor, n_nuis = tab.shape
    if n_factor < 2 or n_nuis < 2:
        return {
            "available": True,
            "confounded": False,
            "severity": 0.0,
            "crosstab": tab,
            "verdict": f"'{nuisance}' has only one level within these data; nothing to confound.",
        }

    # Severity: fraction of factor levels that occur at only one nuisance level.
    levels_with_one = int((tab > 0).sum(axis=1).eq(1).sum())
    severity = levels_with_one / n_factor

    # Balance measure: Cramer's V on the contingency table (0 = independent)
    try:
        from scipy.stats import chi2_contingency

        chi2 = chi2_contingency(tab.values)[0]
        n_obs = tab.values.sum()
        cramers_v = float(np.sqrt(chi2 / (n_obs * (min(tab.shape) - 1))))
    except Exception:
        cramers_v = float("nan")

    if severity >= 1.0:
        verdict = (
            f"COMPLETE CONFOUNDING: every level of '{factor}' occurs at exactly one level "
            f"of '{nuisance}'. The two effects cannot be separated by any analysis. "
            f"This must be fixed in the design (counterbalance '{nuisance}' across '{factor}')."
        )
    elif severity > 0:
        verdict = (
            f"PARTIAL CONFOUNDING: {levels_with_one} of {n_factor} levels of '{factor}' occur at "
            f"only one level of '{nuisance}' (Cramer's V = {cramers_v:.2f}). Effects are "
            f"separable but with reduced precision and residual bias risk."
        )
    else:
        verdict = (
            f"'{factor}' is crossed with '{nuisance}' (Cramer's V = {cramers_v:.2f}); "
            f"effects are separable."
        )

    return {
        "available": True,
        "confounded": severity > 0,
        "severity": float(severity),
        "cramers_v": cramers_v,
        "crosstab": tab,
        "verdict": verdict,
    }


def design_report(df: pd.DataFrame, lights_on_hour: float = 8.0) -> dict:
    """
    Full pre-analysis design report: what the design is, what the replication
    unit is, and which factor pairs are confounded.

    Intended to be run and READ before any hypothesis test. Returns a dict with
    `spec` (DesignSpec), `confounds` (list of dicts) and `blocking` (list of
    strings that should stop the analysis).
    """
    spec = describe_design(df, lights_on_hour=lights_on_hour)

    confounds = []
    pairs = []
    factors = [c for c in ("Genotype", "Treatment", "Compound", "Sex") if c in df.columns]
    nuisances = [
        c for c in ("TimeOfDay", "ZT", "Batch", "ExperimentID", "Date", "SessionIndex", "RackPosition")
        if c in df.columns and df[c].notna().any()
    ]
    for f in factors:
        for nz in nuisances:
            pairs.append((f, nz))

    for f, nz in pairs:
        res = check_confounding(df, f, nz)
        if res.get("available") and res.get("confounded"):
            confounds.append({"factor": f, "nuisance": nz, **{k: v for k, v in res.items() if k != "crosstab"}})

    blocking = [c["verdict"] for c in confounds if c.get("severity", 0) >= 1.0]

    return {
        "spec": spec,
        "design_summary": spec.summary(),
        "cluster_unit": spec.cluster_unit,
        "warnings": spec.warnings,
        "notes": spec.notes,
        "confounds": confounds,
        "blocking": blocking,
    }
