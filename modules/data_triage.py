"""
Data triage: what questions can this dataset actually answer?

WHY THIS EXISTS
---------------
Every check in this module already existed somewhere in DroPhenix. They ran in
different places, at different times, and mostly AFTER the user had chosen an
analysis and seen a p-value. That is the wrong order. By the time a number is on
screen it is already believed, and a warning underneath it competes with a
result that looks like an answer.

Two real datasets from this laboratory make the case. In the activity-monitor
files, every old fly was in Monitor 1 and every young fly in Monitor 2, so the
age comparison was inseparable from any difference between the two machines --
a fact that took an explicit confounding test to surface, and that no amount of
downstream statistics could have repaired. In the same files sleep sat at 88% of
the day, pinned against its own ceiling, so real differences were compressed
into nothing. Both are properties of the DESIGN and the RECORDING. Both were
knowable before a single hypothesis was tested.

WHAT IT DOES DIFFERENTLY
------------------------
It does not return a list of warnings. It returns a VERDICT PER QUESTION:

    Genotype  ANSWERABLE
    Sex       ANSWERABLE
    Age       BLOCKED - completely confounded with Monitor

A warning invites the reader to weigh it against the result. A blocked question
tells them there is no result to weigh, which is the true state of affairs and
is far more useful.

WHAT IT CHECKS
--------------
  confounding    a factor that varies only alongside a nuisance variable cannot
                 be estimated at all -- this is arithmetic, not caution
  replication    at least two INDEPENDENT units per group; with one, the group
                 effect and the unit effect are one parameter
  saturation     a bounded measure pinned near its ceiling or floor has no room
                 left to show a difference
  clustering     flies sharing a vial are not independent; the design effect
                 says how much less information the sample carries
  plausibility   values outside published ranges usually mean the instrument,
                 not the biology
  coverage       enough timepoints, days or doses to identify what is being asked

Nothing here is a substitute for looking at the data. It is a way of being told
the things that are easy to miss and impossible to fix afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

BLOCKER, WARNING, NOTE, OK = "blocker", "warning", "note", "ok"

# Nuisance variables: things that should not explain a biological result, and
# that a factor must be crossed with rather than nested inside.
NUISANCE_COLS = ["Monitor", "Batch", "ExperimentID", "Date", "TimeOfDay",
                 "SessionIndex", "RackPosition", "SourceVideo"]

# Columns that name a biological question rather than a nuisance.
FACTOR_COLS = ["Genotype", "Treatment", "Sex", "Age", "Compound", "Dose"]

# Units of independent replication, best first.
UNIT_COLS = ["VialID", "FlyID", "Channel", "Replicate"]


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    factor: Optional[str] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class TriageReport:
    n_rows: int
    unit_col: Optional[str]             # readable label for the unit key
    factors: List[str]
    answerable: Dict[str, str]          # factor -> "answerable" | "blocked" | "weak"
    findings: List[Finding]
    # The unit key as COLUMNS (register N4). `unit_col` stays a label so that
    # existing callers keep working -- it is the column name when one column
    # suffices, and a joined description when the key is composite.
    unit_cols: List[str] = field(default_factory=list)

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == BLOCKER]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    def verdict(self) -> str:
        ok = [f for f, v in self.answerable.items() if v == "answerable"]
        weak = [f for f, v in self.answerable.items() if v == "weak"]
        bad = [f for f, v in self.answerable.items() if v == "blocked"]
        parts = []
        if ok:
            parts.append("Can answer: " + ", ".join(ok) + ".")
        if weak:
            parts.append("Weakened: " + ", ".join(weak) + ".")
        if bad:
            parts.append("CANNOT answer: " + ", ".join(bad) + ".")
        if not parts:
            parts.append("No biological factors found to assess.")
        return " ".join(parts)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"severity": f.severity, "code": f.code,
                              "factor": f.factor, "message": f.message}
                             for f in self.findings])

    def report(self, width: int = 100) -> str:
        lines = [self.verdict(), ""]
        for label, sev in (("BLOCKERS", BLOCKER), ("WARNINGS", WARNING), ("NOTES", NOTE)):
            group = [f for f in self.findings if f.severity == sev]
            if not group:
                continue
            lines.append(f"{label}:")
            for f in group:
                lines.append(f"  [{f.code}] {f.message}")
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Which column, or columns, identify one experimental unit (register N4)
# ═══════════════════════════════════════════════════════════════════════════
# This used to return the first present of VialID, FlyID, Channel, Replicate.
# `Replicate` is a bare integer reused in every group, so on DroPhenix's own
# demo data -- 36 real vials -- the unit was taken to be `Replicate` with three
# distinct values, and two numbers were then fabricated from it:
#
#   * a design effect: "ICC = 0.11 over 3 Replicates ... standard errors must
#     be 378% larger", computed by pooling twelve unrelated vials (different
#     genotypes, different treatments) into each of three "clusters", so the
#     between-cluster variance measured was genotype variance;
#   * a BLOCKER: Pantothenate, Spermine and Spermine+Pantothenate each have one
#     distinct `Replicate` value, so the Treatment effect was declared
#     unanalysable -- when each of those arms has SIX real vials. The platform
#     told the user its own demo data could not answer the question it is a
#     demo of.
#
# THE TEST THAT DECIDES IT. A physical unit cannot be measured twice at the
# same timepoint. So a candidate key is a valid unit if and only if it does not
# repeat within a timepoint. On the demo data:
#
#     Replicate alone                           315 repeats  -> not a unit
#     Genotype x Treatment x Sex x Replicate       0 repeats  -> a unit
#
# That is an objective property of the table rather than a naming convention,
# it is checkable on any dataset, and it rejects a column called `VialID` that
# behaves like `Replicate` just as readily as it rejects `Replicate` -- the
# name is not evidence.
def unit_key_is_valid(df: pd.DataFrame, cols: Sequence[str],
                      time_col: str = "Time"):
    """
    Is `cols` a usable experimental-unit key for this table?

    Returns `(ok, reason, n_repeats, n_units)`. A key fails when the same key
    value appears twice at one timepoint, because no physical unit can be
    measured twice at the same instant: that means the key identifies a GROUP
    of units, not a unit.

    With no time column the key cannot be tested this way, so it is accepted
    only if it is not a known pooled column -- a bare `Replicate` is rejected
    on its own because its values are reused across groups by construction.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return False, "none of those columns are present", 0, 0
    sub = df[cols].dropna(how="all")
    if sub.empty:
        return False, "the key is empty for every row", 0, 0
    n_units = int(df[cols].drop_duplicates().shape[0])
    if time_col in df.columns and df[time_col].notna().any():
        rep = int(df.duplicated(subset=cols + [time_col]).sum())
        if rep:
            return (False,
                    f"the key repeats {rep} times within a timepoint, so it "
                    f"identifies a group of units rather than a unit",
                    rep, n_units)
        return True, "", 0, n_units
    # No timepoint to test against.
    if list(cols) == ["Replicate"]:
        return (False,
                "`Replicate` alone is a within-group index, reused across "
                "groups, and there is no timepoint column to test it against",
                0, n_units)
    return True, "no timepoint column, so the repeat test could not be run", 0, n_units


def unit_key(df: pd.DataFrame):
    """
    The experimental-unit key, chosen by test.

    Tried in order: each dedicated identifier on its own, then the composite of
    the biological factors with `Replicate`. The first that passes
    `unit_key_is_valid` wins. Returns `(cols, label, reason)`; `cols` is empty
    when nothing qualifies, and `reason` then says why, which is what the
    caller turns into an `ambiguous_unit` finding.
    """
    tried = []
    for c in UNIT_COLS:
        if c == "Replicate":
            continue                      # never on its own; see below
        if c in df.columns and df[c].notna().any():
            ok, why, _rep, _n = unit_key_is_valid(df, [c])
            if ok:
                return [c], c, ""
            tried.append(f"{c} ({why})")

    if "Replicate" in df.columns and df["Replicate"].notna().any():
        facs = [c for c in FACTOR_COLS
                if c in df.columns and df[c].notna().any()
                and df[c].nunique(dropna=True) > 1]
        if facs:
            comp = facs + ["Replicate"]
            ok, why, _rep, _n = unit_key_is_valid(df, comp)
            if ok:
                return comp, " x ".join(comp), ""
            tried.append(f"{' x '.join(comp)} ({why})")
        ok, why, _rep, _n = unit_key_is_valid(df, ["Replicate"])
        if ok:
            return ["Replicate"], "Replicate", ""
        tried.append(f"Replicate ({why})")

    return [], None, ("no candidate identifies a single experimental unit: "
                      + "; ".join(tried) if tried else
                      "no unit column is present at all")


def _pick_unit(df: pd.DataFrame) -> Optional[str]:
    """Backwards-compatible single-column view of `unit_key`."""
    cols, _label, _why = unit_key(df)
    return cols[0] if len(cols) == 1 else None


def _present_factors(df: pd.DataFrame, factors: Optional[Sequence[str]]) -> List[str]:
    cand = list(factors) if factors else FACTOR_COLS
    return [c for c in cand
            if c in df.columns and df[c].notna().any() and df[c].nunique() > 1]


def _as_cols(unit_cols) -> List[str]:
    """
    Accept either a column name or a list of them.

    A bare string must not be iterated: `list("Channel")` is seven characters,
    which pandas then looks up as seven columns. `triage_dam` passes a string,
    so normalising here keeps both call styles valid instead of making the
    difference a trap.
    """
    if unit_cols is None:
        return []
    if isinstance(unit_cols, str):
        return [unit_cols]
    return [c for c in unit_cols]


def _check_confounding(df, factors, unit_cols) -> List[Finding]:
    """
    A factor whose every level sits inside a single level of a nuisance variable
    cannot be separated from it. No model, no correction, no amount of data
    fixes this -- the two effects are literally the same parameter.
    """
    out: List[Finding] = []
    nuis = [c for c in NUISANCE_COLS
            if c in df.columns and df[c].notna().any() and df[c].nunique() > 1]
    if not nuis:
        return out

    # Compare at the level of UNITS, not rows: repeated measurements of one fly
    # would otherwise look like extra evidence of crossing.
    _uc = _as_cols(unit_cols)
    key = _uc if _uc else None
    meta = df.drop_duplicates(subset=key)[list(set(factors + nuis))] if key else df[
        list(set(factors + nuis))].drop_duplicates()

    for f in factors:
        for nz in nuis:
            sub = meta[[f, nz]].dropna()
            if sub.empty or sub[f].nunique() < 2 or sub[nz].nunique() < 2:
                continue
            tab = pd.crosstab(sub[f], sub[nz])
            one_only = int((tab > 0).sum(axis=1).eq(1).sum())
            severity_frac = one_only / tab.shape[0]
            if severity_frac >= 1.0:
                out.append(Finding(
                    BLOCKER, "confounded", factor=f,
                    message=(f"'{f}' is completely confounded with '{nz}': every level of "
                             f"{f} appears at exactly one level of {nz}. Any difference "
                             f"between {nz} levels is indistinguishable from a {f} effect. "
                             f"This cannot be corrected in analysis -- it has to be fixed in "
                             f"the design, by splitting each {f} group across every {nz}."),
                    evidence={"crosstab": tab.to_dict()}))
            elif severity_frac > 0:
                out.append(Finding(
                    WARNING, "partial_confounding", factor=f,
                    message=(f"'{f}' is partly confounded with '{nz}': {one_only} of "
                             f"{tab.shape[0]} levels occur at only one {nz}. The effect is "
                             f"estimable but with reduced precision and some residual bias."),
                    evidence={"levels_nested": one_only}))
    return out


def _check_replication(df, factors, unit_cols) -> List[Finding]:
    """At least two independent units per group, or the group IS the unit."""
    out: List[Finding] = []
    unit_cols = _as_cols(unit_cols)
    if not unit_cols:
        out.append(Finding(
            WARNING, "no_unit_column",
            message=("No column identifies the unit of replication (VialID, FlyID, "
                     "Channel or Replicate). Every row is being treated as independent. "
                     "If rows are repeated measurements, or flies sharing a vial, all "
                     "p-values will be too small.")))
        return out

    label = " x ".join(unit_cols)
    # Count DISTINCT UNITS per level, from the unit key -- not distinct values
    # of one pooled column. Counting `Replicate` values here is what declared
    # three treatment arms of six vials each to be single-replicate (N4).
    units = df[list(unit_cols) + [c for c in factors if c not in unit_cols]] \
        .drop_duplicates()
    for f in factors:
        n_units = units.groupby(f).size() if f not in unit_cols else \
            df.groupby(f)[list(unit_cols)].apply(
                lambda g: g.drop_duplicates().shape[0])
        thin = n_units[n_units < 2]
        if len(thin):
            out.append(Finding(
                BLOCKER, "no_replication", factor=f,
                message=(f"'{f}' has level(s) {list(thin.index)} with only one {label}. "
                         f"With a single unit the {f} effect and that unit's own peculiarity "
                         f"are the same number, so nothing can be concluded."),
                evidence={"units_per_level": n_units.to_dict()}))
        elif n_units.min() < 3:
            out.append(Finding(
                WARNING, "low_replication", factor=f,
                message=(f"'{f}' has as few as {int(n_units.min())} {label} units in a level "
                         f"({n_units.to_dict()}). Estimates will be very imprecise and "
                         f"assumption checks essentially powerless."),
                evidence={"units_per_level": n_units.to_dict()}))
    return out


def _check_saturation(df, value_col=None, bounds=None) -> List[Finding]:
    """
    A bounded measure pressed against its ceiling or floor has no room to show a
    difference. This is not a statistical nicety: in the laboratory's own
    activity data, sleep sat at 88% of the day and the entire night phase was
    flat against the maximum, so no group difference could appear there however
    real it was.
    """
    out: List[Finding] = []
    checks = []
    if value_col and value_col in df.columns and bounds:
        checks.append((value_col, bounds[0], bounds[1]))
    else:
        if {"Count", "n"}.issubset(df.columns):
            p = (df["Count"] / df["n"]).dropna()
            if len(p):
                checks.append(("proportion climbing", 0.0, 1.0, p))
        if "Total_sleep_min_per_day" in df.columns:
            # The ceiling that matters is the BIOLOGICAL one, not the arithmetic
            # one. Sleep cannot exceed 1440 min in a day, but a fly does not
            # approach that: published totals top out near 1000-1150 min, so a
            # cohort at 1300 has already lost its dynamic range even though it
            # is nowhere near 1440. Testing against the arithmetic bound would
            # have passed the laboratory's own data, where the entire night
            # phase was flat against the maximum.
            checks.append(("sleep (min/day)", 0.0, 1150.0,
                           df["Total_sleep_min_per_day"].dropna().clip(upper=1150.0)))

    for item in checks:
        name, lo, hi = item[0], item[1], item[2]
        vals = item[3] if len(item) > 3 else df[name].dropna()
        if not len(vals):
            continue
        span = hi - lo
        at_top = float((vals >= hi - 0.05 * span).mean())
        at_bot = float((vals <= lo + 0.05 * span).mean())
        if at_top >= 0.25:
            out.append(Finding(
                WARNING, "ceiling", factor=None,
                message=(f"{at_top:.0%} of {name} values sit in the top 5% of the possible "
                         f"range. A measure at its ceiling cannot show a difference, so real "
                         f"effects are compressed and the assay is running out of dynamic "
                         f"range. Consider a harder criterion, an earlier timepoint, or "
                         f"checking whether the instrument is under-reading."),
                evidence={"fraction_at_ceiling": at_top}))
        if at_bot >= 0.25:
            out.append(Finding(
                WARNING, "floor", factor=None,
                message=(f"{at_bot:.0%} of {name} values sit in the bottom 5% of the range. "
                         f"At the floor, differences cannot appear."),
                evidence={"fraction_at_floor": at_bot}))
    return out


def _check_clustering(df, factors, unit_cols) -> List[Finding]:
    """How much less information does the sample carry than its row count suggests?"""
    out: List[Finding] = []
    unit_cols = _as_cols(unit_cols)
    if not unit_cols or not {"Count", "n"}.issubset(df.columns):
        return out
    try:
        from modules.hierarchical import estimate_icc, design_effect, effective_sample_size
    except Exception:
        return out

    sub = df
    if "Time" in df.columns and df["Time"].nunique() > 1:
        # One timepoint only: repeated times are repeated measures of a unit,
        # not extra animals in it, and pooling them inflates the cluster size.
        sub = df[df["Time"] == df["Time"].max()]

    label = " x ".join(unit_cols)
    counts = sub.groupby(list(unit_cols))["Count"].sum()
    totals = sub.groupby(list(unit_cols))["n"].sum()
    icc = estimate_icc(counts, totals)
    if not np.isfinite(icc.get("icc", np.nan)):
        return out
    de = design_effect(icc["mean_cluster_size"], icc["icc"])
    if de["deff"] > 1.5:
        n_tot = float(totals.sum())
        n_eff = effective_sample_size(n_tot, icc["mean_cluster_size"], icc["icc"])
        out.append(Finding(
            WARNING, "clustering",
            message=(f"Animals within a unit ({label}) are correlated "
                     f"(ICC = {icc['icc']:.2f} "
                     f"over {icc['n_clusters']} units). The design effect is "
                     f"{de['deff']:.1f}, so standard errors must be "
                     f"{(de['se_inflation'] - 1) * 100:.0f}% larger than a per-animal "
                     f"analysis would give: {n_tot:.0f} animals carry the information of "
                     f"about {n_eff:.0f} independent ones. Analyse at the unit level "
                     f"or use the hierarchical model."),
            evidence={"icc": icc["icc"], "deff": de["deff"]}))
    return out


def _check_coverage(df, factors) -> List[Finding]:
    """Enough timepoints, ages or doses to identify what is being asked."""
    out: List[Finding] = []
    if "Dose" in df.columns and df["Dose"].nunique() > 1 and df["Dose"].nunique() < 4:
        out.append(Finding(
            BLOCKER, "too_few_doses", factor="Dose",
            message=(f"Only {df['Dose'].nunique()} dose levels. A four-parameter "
                     f"dose-response curve has four parameters; it cannot be identified "
                     f"from fewer than four doses, and reliable EC50 intervals need five "
                     f"or more spanning the response.")))
    if "Age" in df.columns and 1 < df["Age"].nunique() < 3:
        out.append(Finding(
            WARNING, "few_ages", factor="Age",
            message=(f"Only {df['Age'].nunique()} ages. A decline curve fitted to two "
                     f"points is a straight line by construction; its shape carries no "
                     f"information.")))
    return out


# ═══════════════════════════════════════════════════════════════════════════
def triage(df: pd.DataFrame,
           factors: Optional[Sequence[str]] = None,
           unit_col: Optional[str] = None,
           value_col: Optional[str] = None,
           value_bounds: Optional[Sequence[float]] = None) -> TriageReport:
    """
    Assess a dataset BEFORE analysing it and report which questions it supports.

    `factors` defaults to whichever of Genotype/Treatment/Sex/Age/Compound/Dose
    are present and vary. `unit_col` defaults to the best available identifier
    of independent replication.
    """
    if df is None or df.empty:
        return TriageReport(0, None, [], {},
                            [Finding(BLOCKER, "empty", message="The dataset is empty.")],
                            unit_cols=[])

    if unit_col:
        ucols, ulabel, ureason = [unit_col], unit_col, ""
    else:
        ucols, ulabel, ureason = unit_key(df)
    facs = _present_factors(df, factors)

    findings: List[Finding] = []
    if not ucols:
        # Register N4's own acceptance criterion: where no candidate identifies
        # a unit, say so and compute no ICC -- rather than deriving a design
        # effect from a column that pools unrelated groups.
        findings.append(Finding(
            WARNING, "ambiguous_unit",
            message=("The unit of replication cannot be identified, so no "
                     "intraclass correlation or design effect is reported: "
                     + ureason + ". Add a VialID (or FlyID, or Channel) that is "
                     "unique per physical unit, or label the biological factors "
                     "so that a composite key can be formed with Replicate.")))

    # "No factors" is almost never what the experimenter intended; it usually
    # means the metadata was never filled in. Say which columns are present but
    # constant, so the cause is obvious rather than mysterious.
    if not facs:
        constant = [c for c in (factors or FACTOR_COLS)
                    if c in df.columns and df[c].nunique(dropna=True) == 1]
        missing = [c for c in (factors or FACTOR_COLS) if c not in df.columns]
        detail = []
        if constant:
            vals = {c: str(df[c].dropna().iloc[0])[:24] for c in constant}
            detail.append(f"present but with a single value ({vals})")
        if missing:
            detail.append(f"absent ({missing})")
        findings.append(Finding(
            BLOCKER, "no_factors",
            message=("No biological factor varies in this dataset, so there is nothing to "
                     "compare. Grouping columns are " + "; ".join(detail) + ". If this came "
                     "from a video or monitor batch, the metadata manifest was probably left "
                     "unfilled -- the measurements are fine, but they are unlabelled.")))

    findings += _check_confounding(df, facs, ucols)
    findings += _check_replication(df, facs, ucols)
    findings += _check_saturation(df, value_col, value_bounds)
    findings += _check_clustering(df, facs, ucols)
    findings += _check_coverage(df, facs)

    answerable: Dict[str, str] = {}
    for f in facs:
        sev = [x.severity for x in findings if x.factor == f]
        answerable[f] = ("blocked" if BLOCKER in sev
                         else "weak" if WARNING in sev else "answerable")

    if ucols:
        n_u = int(df[list(ucols)].drop_duplicates().shape[0])
        findings.append(Finding(
            NOTE, "unit",
            message=(f"Unit of replication taken to be '{ulabel}' "
                     f"({n_u} distinct). Chosen because it does not repeat "
                     f"within a timepoint, which a group of units would.")))
    return TriageReport(len(df), ulabel, facs, answerable, findings,
                        unit_cols=list(ucols))


def triage_dam(long: pd.DataFrame, channel_map: Optional[pd.DataFrame] = None,
               per_fly: Optional[pd.DataFrame] = None) -> TriageReport:
    """
    Triage for raw activity-monitor data, adding the checks that only apply
    there: light regime, recording length, restarts, dead tubes and death.
    """
    from modules import dam_reader as DR

    meta = long
    if channel_map is not None:
        meta = DR.apply_channel_map(long, channel_map)

    unit = "Channel" if "Channel" in meta.columns else None
    facs = _present_factors(meta, None)
    per_unit = meta.drop_duplicates(subset=["Monitor", "Channel"]) if unit else meta

    findings: List[Finding] = []
    findings += _check_confounding(per_unit, facs, None)
    findings += _check_replication(per_unit, facs, ["Channel"] if unit else None)

    a = DR.assess_recording(long)
    if not a.can_analyse_period:
        findings.append(Finding(BLOCKER, "period_unavailable", factor="circadian period",
                                message=a.period_refusal))
    if a.n_sessions > 1:
        findings.append(Finding(WARNING, "monitor_restart",
                                message=f"The monitor restarted {a.n_sessions - 1} time(s)."))
    n_dead = sum(len(v) for v in a.dead_channels.values())
    if n_dead:
        findings.append(Finding(WARNING, "dead_tubes",
                                message=f"{n_dead} tube(s) recorded almost no activity."))

    dr = DR.death_report(long)
    if dr["n_likely_dead"]:
        findings.append(Finding(WARNING, "deaths",
                                message=dr["flags"][0] if dr["flags"] else
                                f"{dr['n_likely_dead']} flies appear to have died."))

    if per_fly is not None and not per_fly.empty:
        pr = DR.plausibility_report(per_fly)
        for f in pr.get("flags", []):
            findings.append(Finding(WARNING, "implausible", message=f))
        # The report also returns `notes`: the per-group breakdown behind a
        # pooled verdict (N63), and the statement that the sleep check was
        # DECLINED because the values were scored under a criterion the
        # published range was not measured under. Reading only `flags` would
        # drop both, and a declined check that looks like a passed one is the
        # failure mode the gate exists to prevent.
        for nmsg in pr.get("notes", []):
            findings.append(Finding(NOTE, "plausibility_note", message=nmsg))
        findings += _check_saturation(per_fly)

    answerable: Dict[str, str] = {}
    for f in facs:
        sev = [x.severity for x in findings if x.factor == f]
        answerable[f] = ("blocked" if BLOCKER in sev
                         else "weak" if WARNING in sev else "answerable")
    answerable["circadian period"] = ("blocked" if not a.can_analyse_period
                                      else "answerable")

    return TriageReport(len(long), unit, facs, answerable, findings)
