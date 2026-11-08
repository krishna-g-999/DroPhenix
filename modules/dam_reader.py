"""
Read raw DAMSystem monitor files into the DroPhenix activity schema.

WHAT THIS IS FOR
----------------
`modules/dams.py` analyses activity and sleep, but it expects tidy data: one row
per fly per minute, already labelled with genotype and sex. A TriKinetics DAM
monitor does not produce that. It produces a 42-column tab-separated file with
32 anonymous channels, a light sensor, a status flag, and whatever the monitor
did when somebody restarted it. This module turns the second thing into the
first, and refuses when the file cannot support the analysis being asked of it.

FOUR THINGS THAT SILENTLY CORRUPT A DAM ANALYSIS
------------------------------------------------
Every one of these was present in the first real monitor files this module was
written against, and none of them announces itself.

1. STATUS ROWS READ AS ZERO ACTIVITY.
   Column 4 is the monitor status; 1 means a valid reading. When it is anything
   else the channel counts are written as zeros -- but those zeros mean "not
   measured", not "the fly did not move". Sleep is defined as five consecutive
   minutes of inactivity, so a handful of status rows read as genuine zeros
   manufactures sleep out of a monitor fault, and can splice two real bouts into
   one long one. These rows are returned as MISSING.

2. THE MONITOR RESTARTED MID-RECORDING.
   The index column resets to 1 and the clock jumps. Concatenating the two
   sessions without noticing gives a file whose row count and index disagree,
   and any analysis keyed on row number is then wrong. Sessions are detected and
   labelled.

3. DEAD OR EMPTY TUBES.
   A channel with no fly, or a fly that died, records near-zero counts forever.
   That is not a fly that sleeps 24 hours a day; it is an absence of data, and
   including it drags every group mean toward "maximally asleep".

4. ASKING FOR CIRCADIAN PERIOD FROM LIGHT-DARK DATA.
   This is the most consequential one because the number that comes back looks
   perfectly reasonable. Under a light-dark cycle the fly is ENTRAINED: its
   rhythm is driven by the lights, so a periodogram returns ~24 h whatever the
   endogenous clock is doing. A free-running period is a property of the animal
   and can only be measured in constant darkness. `assess_recording` detects the
   light regime and refuses period analysis in LD, and also refuses when the
   recording is too short to resolve a period at all.

FILE FORMAT (DAMSystem2/3, 42 columns, tab separated)
    1  index                     6  monitor number
    2  date  (DD MMM YY)         7  tube number / unused
    3  time  (HH:MM:SS)          8  data type ("Ct" = counts)
    4  monitor status (1 = OK)   9  unused
    5  extras                   10  light sensor (0/1)
                            11-42  32 activity channels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

N_CHANNELS = 32
_FIRST_CHANNEL_COL = 10          # zero-based index of channel 1
STATUS_OK = 1


# ═══════════════════════════════════════════════════════════════════════════
# 1. READING
# ═══════════════════════════════════════════════════════════════════════════
def read_monitor(
    path: str,
    monitor: Optional[str] = None,
    drop_bad_status: bool = True,
) -> pd.DataFrame:
    """
    Read one monitor file into a tidy long table.

    Returns one row per (channel, minute) with columns
    `Monitor, Channel, Timestamp, Activity, Light, Status, Session, Valid`.

    `Activity` is NaN wherever the monitor status was not OK, so that downstream
    sleep detection sees missing data rather than an hour of perfect stillness.
    Set `drop_bad_status=False` to keep the raw zeros for inspection.
    """
    p = Path(path)
    raw = pd.read_csv(p, sep="\t", header=None, engine="python")

    n_cols = raw.shape[1]
    expected = _FIRST_CHANNEL_COL + N_CHANNELS
    if n_cols < expected:
        raise ValueError(
            f"{p.name} has {n_cols} columns; a DAMSystem monitor file has at least "
            f"{expected} (10 header columns plus {N_CHANNELS} channels). This does not "
            "look like a raw monitor file."
        )

    try:
        ts = pd.to_datetime(raw[1].astype(str).str.strip() + " "
                            + raw[2].astype(str).str.strip(),
                            format="%d %b %y %H:%M:%S")
    except Exception:
        ts = pd.to_datetime(raw[1].astype(str).str.strip() + " "
                            + raw[2].astype(str).str.strip(),
                            dayfirst=True, errors="coerce")
    if ts.isna().all():
        raise ValueError(f"Could not parse dates/times from {p.name} columns 2 and 3.")

    status = pd.to_numeric(raw[3], errors="coerce").fillna(-1).astype(int)
    light = pd.to_numeric(raw[9], errors="coerce")

    # A restart shows up as the index column going backwards.
    #
    # Register N80. This was `np.diff(idx, prepend=idx[0]) <= 0`, which makes
    # the FIRST difference idx[0] - idx[0] = 0 -- and 0 satisfies `<= 0`, so the
    # first row was counted as a restart and the cumulative sum started at 1
    # before the `+ 1`. Every session label was therefore one too high and the
    # recording began at "Session 2": any report, filter or figure legend
    # looking for session 1 found nothing, on every file. Prepending a value
    # one SMALLER than the first index makes the first difference positive,
    # which is what it is -- the recording starting is not a restart.
    idx = pd.to_numeric(raw[0], errors="coerce").fillna(0).to_numpy()
    if idx.size == 0:
        session = np.zeros(0, dtype=int)
    else:
        session = (np.diff(idx, prepend=idx[0] - 1.0) <= 0).cumsum() + 1

    counts = raw.iloc[:, _FIRST_CHANNEL_COL:_FIRST_CHANNEL_COL + N_CHANNELS]
    counts = counts.apply(pd.to_numeric, errors="coerce").to_numpy(float)

    valid = (status == STATUS_OK).to_numpy()
    if drop_bad_status:
        counts[~valid, :] = np.nan

    mon = monitor or p.stem
    n_rows = len(raw)
    long = pd.DataFrame({
        "Monitor": np.repeat(mon, n_rows * N_CHANNELS),
        "Channel": np.tile(np.arange(1, N_CHANNELS + 1), n_rows),
        "Timestamp": np.repeat(ts.to_numpy(), N_CHANNELS),
        "Activity": counts.ravel(),
        "Light": np.repeat(light.to_numpy(), N_CHANNELS),
        "Status": np.repeat(status.to_numpy(), N_CHANNELS),
        "Session": np.repeat(session, N_CHANNELS),
        "Valid": np.repeat(valid, N_CHANNELS),
    })
    long.attrs["source_file"] = str(p)
    return long


def read_monitors(paths: Sequence[str],
                  monitors: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read several monitor files and stack them."""
    out = []
    for i, pth in enumerate(paths):
        name = monitors[i] if monitors and i < len(monitors) else Path(pth).stem
        out.append(read_monitor(pth, monitor=name))
    return pd.concat(out, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. QUALITY ASSESSMENT AND REFUSALS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class RecordingAssessment:
    duration_h: float
    n_days: float
    n_sessions: int
    session_breaks: List[str]
    time_gaps: List[str]
    n_invalid_rows: int
    light_regime: str                # "LD" | "DD" | "LL" | "unknown"
    n_light_transitions: int
    dead_channels: Dict[str, List[int]]
    can_analyse_sleep: bool
    can_analyse_period: bool
    period_refusal: str
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.duration_h:.1f} h ({self.n_days:.1f} days), {self.light_regime}, "
                f"{self.n_sessions} session(s), "
                f"{sum(len(v) for v in self.dead_channels.values())} dead channel(s)")


def assess_recording(long: pd.DataFrame,
                     dead_threshold: float = 50.0,
                     min_days_for_period: float = 3.0) -> RecordingAssessment:
    """
    Decide what this recording can and cannot support, and say why.

    The refusal that matters most is CIRCADIAN PERIOD. Under a light-dark cycle
    the animal is entrained, so a periodogram measures the LAMP, not the fly: it
    will return approximately 24 h for a wild-type and for an arrhythmic mutant
    alike. Free-running period is only meaningful in constant darkness. A tool
    that returns "period = 24.0 h" from LD data is not being helpful; it is
    producing a number with no biological referent.
    """
    warnings: List[str] = []
    ts = pd.to_datetime(long["Timestamp"])
    dur_h = float((ts.max() - ts.min()).total_seconds() / 3600.0)
    n_days = dur_h / 24.0

    per_time = long.drop_duplicates("Timestamp").sort_values("Timestamp")
    n_sessions = int(per_time["Session"].nunique())
    breaks = []
    if n_sessions > 1:
        for s in sorted(per_time["Session"].unique())[1:]:
            t = per_time.loc[per_time["Session"] == s, "Timestamp"].min()
            breaks.append(str(t))
        warnings.append(
            f"The monitor was restarted {n_sessions - 1} time(s) during this recording "
            f"(at {', '.join(breaks)}). The sessions are labelled; check that the tubes "
            "were not disturbed, because a restart usually means somebody opened the "
            "incubator."
        )

    d = per_time["Timestamp"].diff().dropna().dt.total_seconds()
    step = float(d.mode().iloc[0]) if len(d) else 60.0
    gaps = per_time["Timestamp"][1:][(d > step * 1.5).to_numpy()]
    gap_list = [str(g) for g in gaps]
    if gap_list:
        warnings.append(
            f"{len(gap_list)} gap(s) in the recording (first at {gap_list[0]}). Minutes "
            "that were never recorded are not minutes of stillness; they are excluded "
            "rather than counted as sleep."
        )

    n_invalid = int((~long["Valid"]).drop_duplicates().sum()) if "Valid" in long else 0
    n_invalid_rows = int(long.loc[~long["Valid"], "Timestamp"].nunique())
    if n_invalid_rows:
        warnings.append(
            f"{n_invalid_rows} minute(s) had a monitor status other than OK. The counts "
            "the monitor wrote for those minutes are zeros, but they mean 'not measured'. "
            "They are treated as missing -- read as real zeros they would fabricate sleep, "
            "since sleep is defined as five consecutive still minutes."
        )

    # light regime
    lt = per_time["Light"].dropna()
    n_trans = int((lt.diff().fillna(0) != 0).sum())
    frac_on = float((lt == 1).mean()) if len(lt) else np.nan
    if len(lt) == 0 or lt.nunique() < 2:
        regime = "DD" if (len(lt) and lt.iloc[0] == 0) else ("LL" if len(lt) else "unknown")
    elif n_trans >= 2:
        regime = "LD"
    else:
        regime = "unknown"

    # dead / empty channels
    dead: Dict[str, List[int]] = {}
    for mon, g in long.groupby("Monitor"):
        tot = g.groupby("Channel")["Activity"].sum(min_count=1)
        dead[mon] = sorted(int(c) for c in tot.index[tot.fillna(0) < dead_threshold])
    n_dead = sum(len(v) for v in dead.values())
    if n_dead:
        warnings.append(
            f"{n_dead} channel(s) recorded almost no activity for the whole run "
            f"({dead}). These are empty tubes or dead flies, not animals that slept "
            "continuously; including them would drag every group mean toward maximal "
            "sleep. Exclude them, or confirm the flies were alive."
        )

    can_sleep = dur_h >= 12.0
    if not can_sleep:
        warnings.append(f"Only {dur_h:.1f} h recorded; too short for meaningful sleep "
                        "architecture.")

    refusal = ""
    can_period = True
    if regime == "LD":
        can_period = False
        refusal = (
            "Free-running circadian period cannot be measured from this recording because "
            "the lights were cycling (a light-dark cycle was detected). Under LD the fly is "
            "entrained, so a periodogram reports the period of the LAMP, near 24 h, for a "
            "wild-type and an arrhythmic mutant alike. Period must be measured in constant "
            "darkness. What this recording DOES support is sleep amount and architecture, "
            "day/night distribution, and activity level -- all of which are valid in LD and "
            "are what an entrained recording is for."
        )
    elif n_days < min_days_for_period:
        can_period = False
        refusal = (
            f"The recording is {n_days:.1f} days long. Resolving a circadian period needs at "
            f"least {min_days_for_period:.0f} complete cycles, and 5-7 days is normal "
            "practice; a periodogram over fewer cycles cannot separate a 23 h rhythm from a "
            "25 h one."
        )
    if refusal:
        warnings.append(refusal)

    return RecordingAssessment(
        duration_h=dur_h, n_days=n_days, n_sessions=n_sessions,
        session_breaks=breaks, time_gaps=gap_list,
        n_invalid_rows=n_invalid_rows,
        light_regime=regime, n_light_transitions=n_trans,
        dead_channels=dead,
        can_analyse_sleep=can_sleep,
        can_analyse_period=can_period,
        period_refusal=refusal,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHANNEL MAP
# ═══════════════════════════════════════════════════════════════════════════
def channel_map_template(monitors: Sequence[str],
                         n_channels: int = N_CHANNELS) -> pd.DataFrame:
    """
    A blank channel map: which tube held which genotype, sex and age.

    This information exists only in the experimenter's notebook. It cannot be
    inferred from the counts, and guessing it is the one error that cannot be
    detected later -- every downstream number would be internally consistent and
    attached to the wrong group.
    """
    rows = []
    for m in monitors:
        for c in range(1, n_channels + 1):
            rows.append({"Monitor": m, "Channel": c, "Genotype": "", "Sex": "",
                         "Age": "", "Treatment": "", "Exclude": ""})
    return pd.DataFrame(rows)


def apply_channel_map(long: pd.DataFrame,
                      channel_map: "pd.DataFrame | str") -> pd.DataFrame:
    """
    Attach group labels to channels, and drop channels marked Exclude or left
    unlabelled.

    An unlabelled channel is dropped rather than silently pooled into an
    "unknown" group, because a mislabelled tube is indistinguishable from a real
    biological effect once the analysis has run.
    """
    cmap = pd.read_csv(channel_map) if isinstance(channel_map, str) else channel_map.copy()
    need = {"Monitor", "Channel"}
    if not need.issubset(cmap.columns):
        raise ValueError(f"The channel map needs columns {sorted(need)}.")

    cmap["Channel"] = pd.to_numeric(cmap["Channel"], errors="coerce").astype("Int64")

    # A join that matches nothing is the most dangerous kind of failure here: it
    # returns a valid dataframe with every label blank, so the analysis proceeds
    # and simply reports that no biological factor varies. The usual cause is
    # trivial -- the monitor is called "Monitor1" in the file and "M1" in the map
    # -- and the fix is trivial too, but only if somebody is told.
    file_mons = set(long["Monitor"].astype(str).unique())
    map_mons = set(cmap["Monitor"].astype(str).unique())
    if not (file_mons & map_mons):
        raise ValueError(
            f"The channel map does not refer to any monitor in these files. The data "
            f"contain monitor(s) {sorted(file_mons)} but the map names {sorted(map_mons)}. "
            f"Rename one to match the other -- an unmatched map would silently leave every "
            f"tube unlabelled, and the analysis would then report that nothing varies."
        )
    unmatched = map_mons - file_mons
    if unmatched:
        long.attrs["channel_map_unmatched_monitors"] = sorted(unmatched)

    merged = long.merge(cmap, on=["Monitor", "Channel"], how="left", suffixes=("", "_map"))

    if "Exclude" in merged.columns:
        excl = merged["Exclude"].astype(str).str.strip().str.lower()
        merged = merged[~excl.isin({"1", "y", "yes", "true", "x"})]

    if "Genotype" in merged.columns:
        blank = merged["Genotype"].isna() | (merged["Genotype"].astype(str).str.strip() == "")
        n_blank_ch = merged.loc[blank, ["Monitor", "Channel"]].drop_duplicates().shape[0]
        merged = merged[~blank]
        if n_blank_ch:
            merged.attrs["unlabelled_channels_dropped"] = int(n_blank_ch)
    return merged.reset_index(drop=True)


def detect_death(long: pd.DataFrame,
                 silence_h: float = 12.0) -> pd.DataFrame:
    """
    Find flies that stopped moving and never started again.

    A TOTAL-activity threshold cannot catch this. A fly that lived actively for
    a day and then died records a perfectly healthy total; only the SHAPE of its
    record gives it away. And the consequence is severe and one-directional:
    every minute after death is scored as sleep, because sleep is defined as the
    absence of beam breaks. A cohort in which a third of the flies died halfway
    through will appear to sleep enormously and to sleep more with age -- which
    is exactly the result an ageing experiment is looking for, arriving for
    entirely the wrong reason.

    A fly is called dead from its last recorded movement when nothing follows
    for `silence_h` hours. Twelve hours is deliberately conservative: flies do
    sleep for many consecutive hours, particularly at night, so a shorter window
    would call healthy sleepers dead.

    Returns one row per channel with the last movement, hours of terminal
    silence, and a `likely_dead` flag. Nothing is removed; truncating or
    excluding is the analyst's decision, and it should be a deliberate one.
    """
    out = []
    for (mon, ch), g in long.groupby(["Monitor", "Channel"], sort=True):
        g = g.sort_values("Timestamp")
        act = g["Activity"].to_numpy(float)
        ts = pd.to_datetime(g["Timestamp"]).to_numpy()
        moved = np.where(np.nan_to_num(act, nan=0.0) > 0)[0]
        end = pd.Timestamp(ts[-1])
        if moved.size == 0:
            out.append({"Monitor": mon, "Channel": int(ch), "last_movement": pd.NaT,
                        "silent_h": np.inf, "likely_dead": True,
                        "note": "never moved -- empty tube or dead before recording began"})
            continue
        last = pd.Timestamp(ts[moved[-1]])
        silent_h = float((end - last).total_seconds() / 3600.0)
        out.append({"Monitor": mon, "Channel": int(ch), "last_movement": last,
                    "silent_h": silent_h,
                    "likely_dead": bool(silent_h >= silence_h),
                    "note": (f"no movement for the final {silent_h:.1f} h"
                             if silent_h >= silence_h else "")})
    return pd.DataFrame(out)


def death_report(long: pd.DataFrame, silence_h: float = 12.0) -> dict:
    """Summarise `detect_death` with the consequence spelled out."""
    tab = detect_death(long, silence_h=silence_h)
    n_dead = int(tab["likely_dead"].sum())
    flags = []
    if n_dead:
        med = tab.loc[tab["likely_dead"], "silent_h"].replace(np.inf, np.nan).median()
        flags.append(
            f"{n_dead} of {len(tab)} tubes show no movement for the final "
            f"{silence_h:.0f} hours or more (median terminal silence "
            f"{med:.0f} h). Every silent minute is scored as sleep, so these flies inflate "
            "sleep and reduce activity in whatever group they belong to. Either exclude "
            "them, or truncate each fly's record at its last movement -- but do not leave "
            "them in as sleeping animals."
        )
    return {"table": tab, "n_likely_dead": n_dead, "n_tubes": len(tab), "flags": flags}


def truncate_at_death(long: pd.DataFrame, silence_h: float = 12.0) -> pd.DataFrame:
    """
    Drop each fly's record after its last movement, when it looks dead.

    This is the conservative repair: the fly's living hours are kept and
    analysed, and only the period it cannot have been asleep in -- because it
    was not alive -- is removed.
    """
    tab = detect_death(long, silence_h=silence_h).set_index(["Monitor", "Channel"])
    keep = []
    for (mon, ch), g in long.groupby(["Monitor", "Channel"], sort=False):
        row = tab.loc[(mon, ch)]
        if not row["likely_dead"] or pd.isna(row["last_movement"]):
            if row["likely_dead"]:
                continue                       # never moved: drop entirely
            keep.append(g)
        else:
            keep.append(g[pd.to_datetime(g["Timestamp"]) <= row["last_movement"]])
    return (pd.concat(keep, ignore_index=True) if keep
            else long.iloc[0:0].copy())


# ═══════════════════════════════════════════════════════════════════════════
# 3b. IS THIS BIOLOGICALLY PLAUSIBLE?
# ═══════════════════════════════════════════════════════════════════════════
# Published reference ranges for adult Drosophila melanogaster in a DAM under
# 12:12 LD at 25 C. These are ORDER-OF-MAGNITUDE sanity bounds, not norms to
# test against: strains, ages, temperature and food all move them.
#   Sleep: total daily sleep is conventionally around 500-1000 min, with males
#   sleeping more than females -- the sex difference in daytime sleep is one of
#   the most reproducible features of the assay (Hendricks et al. 2000, Neuron
#   25:129-138; Shaw et al. 2000, Science 287:1834-1837; Andretic & Shaw 2005).
#   Activity: healthy flies typically record several hundred to a few thousand
#   beam crossings per day.
#
#   MEASURED CAVEAT -- the upper bound is method-dependent, and 1150 is not an
#   upper bound on published values.
#   Two real published datasets were run through this pipeline:
#     * Selkrig et al. 2018, Sci Rep 8:10646 (raw single-beam DAM, 379 flies,
#       12:12 LD, 7 days): median 1,188 min/day = 19.8 h, with a healthy median
#       521 beam crossings/fly/day and almost nothing lost to death truncation.
#       69% of those flies fall outside (300, 1150).
#     * Jagannath et al. 2024, eLife 13:RP91355, ethoscope position tracking at
#       the same >=5-min criterion: median 730-847 min/day, and 128 of 128
#       undisturbed flies inside (300, 1150).
#   Same criterion, same species, ~400 min/day apart -- because a single-beam
#   monitor scores a fly that is awake but not crossing the beam as asleep,
#   while position tracking does not. So an excursion above 1150 on DAM data is
#   evidence about the monitor at least as much as about the flies, which is
#   what the flag text must say. Whether to widen the DAM bound is a
#   calibration decision for the owner and is recorded as an open decision in
#   PRIORITISED_DEVELOPMENT_PLAN.md; it is NOT changed here on the strength of
#   two datasets.
PLAUSIBLE_SLEEP_MIN_PER_DAY = (300.0, 1150.0)
PLAUSIBLE_ACTIVITY_PER_DAY = (150.0, 5000.0)

# The sleep range above is published for the CONVENTIONAL >=5-min inactivity
# rule (Hendricks 2000; Shaw 2000) and for nothing else. `dams.sleep_per_fly`
# takes `sleep_min` as a parameter, so a table of sleep scored under a
# different criterion can reach this function and be judged against a range
# that was never derived for it.
#
# Measured on real published data -- Jagannath et al. 2024, eLife 13:RP91355,
# the SAME undisturbed Canton-S flies scored under both criteria
# (`validate_sleep_homeostasis`, checks G1-G5): under the 25-min deep-sleep
# criterion that paper uses, the 58 flies have a median of 356 min/day and 22
# of the 58 fall outside (300, 1150). They are healthy, undisturbed flies. A
# range cannot be stretched to a criterion it was not measured under, and
# there is no published 25-min-criterion range to substitute, so the only
# honest behaviour is to decline to judge and say so.
PUBLISHED_SLEEP_DEFINITION_MIN = 5.0


def plausibility_report(per_fly: pd.DataFrame,
                        sleep_min: float = PUBLISHED_SLEEP_DEFINITION_MIN,
                        group_cols=("Genotype", "Sex")) -> dict:
    """
    Compare per-fly sleep and activity against published ranges and say plainly
    when they do not belong there.

    This exists because an implausible number is the hardest kind to catch: it
    is internally consistent, it survives every statistical test, and it looks
    like a finding. Sleep of 90% of the day is not a strain difference, it is a
    signal that the flies were barely moving or the monitor was barely reading
    -- and no amount of downstream analysis will reveal that.

    sleep_min  : the inactivity criterion, in minutes, that produced the sleep
                 column. The published range applies at 5 min only; at any
                 other value the sleep check is DECLINED rather than applied to
                 a criterion it was not measured under, and
                 `sleep_range_applicable` says so.
    group_cols : columns to stratify the sleep check by, where present.
                 Register N63 -- the check tested the pooled median only, so
                 a genotype x sex group sleeping close to 90% of the day passed
                 as long as the other groups pulled the pooled median back
                 inside the range. The range is a per-population range, so
                 comparing each group's median against it is the comparison the
                 citation supports; no new threshold is introduced.

    Three things this deliberately does NOT do.
      * It does not flag on the FRACTION of flies outside the range. Some
        fraction outside a population range is expected, and no published work
        states how large a fraction is too large -- a threshold here would be
        invented, and an invented threshold inside a plausibility gate is the
        very failure mode the gate exists to catch. The fraction is reported,
        per group, so it cannot be missed.
      * It does not rescale the range for a different criterion.
      * It does not treat an absent sleep column and a WITHHELD one alike.
    """
    out: dict = {"flags": [], "notes": [], "ok": True,
                 "sleep_definition_min": float(sleep_min),
                 "sleep_range_applicable":
                     float(sleep_min) == PUBLISHED_SLEEP_DEFINITION_MIN,
                 "sleep_checked": False}
    if per_fly is None or per_fly.empty:
        out["flags"].append("No per-fly metrics to check.")
        out["ok"] = False
        return out

    if "Total_sleep_min_per_day" in per_fly.columns:
        raw = pd.to_numeric(per_fly["Total_sleep_min_per_day"], errors="coerce")
        s = raw.dropna()
        lo, hi = PLAUSIBLE_SLEEP_MIN_PER_DAY
        out["median_sleep_min_per_day"] = float(s.median()) if len(s) else float("nan")
        out["n_sleep_values"] = int(len(s))
        out["n_sleep_withheld"] = int(len(raw) - len(s))
        out["fraction_outside_sleep_range"] = (
            float(((s < lo) | (s > hi)).mean()) if len(s) else float("nan"))

        if len(s) == 0:
            # Block D (N81) withholds sleep where the bin width cannot resolve
            # the criterion. Before this, an all-NaN sleep column made every
            # comparison below False, so the gate returned ok=True with no
            # flags: a table whose sleep had been refused passed the
            # plausibility check silently. That is the worst available outcome,
            # because the refusal was correct and then invisible.
            out["ok"] = False
            out["flags"].append(
                f"Sleep was not checked: all {len(raw)} sleep values are missing or "
                "withheld. `sleep_summary` withholds sleep when the recording's bin "
                "width cannot resolve the inactivity criterion, so this is expected "
                "on coarse-binned data -- and it means there is no sleep measurement "
                "here to judge. Do not read the absence of a sleep flag as a passing "
                "sleep check."
            )
        elif not out["sleep_range_applicable"]:
            out["notes"].append(
                f"Sleep check DECLINED: these values were scored with a "
                f"{sleep_min:g}-min inactivity criterion, and the {lo:.0f}-{hi:.0f} "
                f"min/day reference range is published for the conventional "
                f"{PUBLISHED_SLEEP_DEFINITION_MIN:g}-min rule. A longer criterion "
                "scores strictly less sleep, so the range does not transfer. Median "
                f"here is {out['median_sleep_min_per_day']:.0f} min/day over {len(s)} "
                f"flies ({100 * out['fraction_outside_sleep_range']:.0f}% outside the "
                "5-min-rule range), reported for information and not as a verdict."
            )
        else:
            out["sleep_checked"] = True
            strata = [c for c in (group_cols or ()) if c in per_fly.columns]
            out["sleep_strata"] = strata
            groups = ([("ALL", per_fly)] if not strata
                      else list(per_fly.groupby(strata, sort=False)))
            per_group = {}
            for key, sub in groups:
                g = pd.to_numeric(sub["Total_sleep_min_per_day"],
                                  errors="coerce").dropna()
                if len(g) == 0:
                    continue
                if isinstance(key, str):
                    label = key
                elif isinstance(key, tuple):
                    label = " / ".join(str(k) for k in key)
                else:
                    label = str(key)
                per_group[label] = {"n": int(len(g)), "median": float(g.median()),
                                    "fraction_outside":
                                        float(((g < lo) | (g > hi)).mean())}
            out["sleep_by_group"] = per_group
            for label, st in per_group.items():
                where = "" if label == "ALL" else f" in {label}"
                if st["median"] > hi:
                    out["ok"] = False
                    out["flags"].append(
                        f"Median sleep is {st['median']:.0f} min/day "
                        f"({st['median'] / 14.4:.0f}% of the day){where}, above the "
                        f"~{hi:.0f} min/day typically reported for adult Drosophila. "
                        "Sleep is scored as five or more consecutive minutes without "
                        "a beam break, so anything that stops beam breaks -- a fly "
                        "that has died, a misaligned or dirty beam, flies resting "
                        "away from the beam, or a tube loaded with food at the wrong "
                        "end -- is indistinguishable from sleep. Note that this is "
                        "not an upper bound on published values: a real published "
                        "single-beam DAM dataset (Selkrig 2018) has a median of 1,188 "
                        "min/day with healthy activity, while position tracking on "
                        "the same criterion gives 730-847. Check activity per day and "
                        "the survival of the flies, and consider the monitor geometry, "
                        "before interpreting this as biology."
                    )
                elif st["median"] < lo:
                    out["ok"] = False
                    out["flags"].append(
                        f"Median sleep is only {st['median']:.0f} min/day{where}, "
                        f"below the ~{lo:.0f} min/day lower end of published values. "
                        "Constant beam breaking can indicate a vibrating monitor or a "
                        "light leak."
                    )
            if len(per_group) > 1:
                out["notes"].append(
                    "Sleep checked per " + " x ".join(strata) + " group ("
                    + "; ".join(f"{k}: n={v['n']}, median {v['median']:.0f}"
                                for k, v in per_group.items())
                    + "). Register N63: a pooled median can sit inside the range "
                      "while one group is far outside it."
                )

    if "Total_activity_per_day" in per_fly.columns:
        a = per_fly["Total_activity_per_day"].dropna()
        lo, hi = PLAUSIBLE_ACTIVITY_PER_DAY
        out["median_activity_per_day"] = float(a.median())
        if a.median() < lo:
            out["ok"] = False
            out["flags"].append(
                f"Median activity is {a.median():.0f} beam crossings per fly per day, below "
                f"the ~{lo:.0f} typical of healthy flies. Low counts inflate every sleep "
                "measure mechanically, so the sleep numbers should not be interpreted until "
                "this is explained -- by the age of the flies, by their health, or by the "
                "monitor."
            )

    # The male/female daytime sleep difference is among the most reproducible
    # features of the assay; its absence is a useful check on the CHANNEL MAP,
    # which is the one input that cannot be verified from the data itself.
    if {"Sex", "Day_sleep_min_per_day"}.issubset(per_fly.columns):
        g = per_fly.groupby("Sex")["Day_sleep_min_per_day"].mean()
        if {"Male", "Female"}.issubset(set(g.index)):
            diff = float(g["Male"] - g["Female"])
            out["male_minus_female_day_sleep"] = diff
            if diff <= 0:
                out["flags"].append(
                    f"Males show {abs(diff):.0f} min/day LESS daytime sleep than females. "
                    "The usual finding is the reverse, and by a wide margin. This is worth "
                    "checking against the channel map: if two groups were swapped, every "
                    "downstream number would still be self-consistent and still be wrong."
                )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. HAND-OFF TO THE ANALYSIS MODULE
# ═══════════════════════════════════════════════════════════════════════════
def to_dams_long(labelled: pd.DataFrame,
                 drop_dead: bool = True,
                 dead_threshold: float = 50.0) -> pd.DataFrame:
    """
    Convert to the schema `modules/dams.py` expects: Genotype, Treatment, Sex,
    FlyID, Time_min, Activity.

    Each channel becomes one FlyID -- one tube is one animal, and it is the unit
    of replication for everything downstream.
    """
    df = labelled.copy()
    dropped_info = {"dead_channels_dropped": 0, "dead_channels": [],
                    "dead_threshold": float(dead_threshold),
                    "dead_channels_by_group": {},
                    "dead_filter_applied": bool(drop_dead)}
    if drop_dead:
        # Register N82. This filter removed channels and recorded nothing,
        # while `unlabelled_channels_dropped` two hundred lines earlier is
        # recorded. Exclusion by total activity is not a neutral filter: a sick
        # or heavily treated cohort moves less, so a threshold on movement
        # removes those cohorts preferentially, and the surviving comparison is
        # between the healthiest members of each group. The count, the
        # threshold, the channels and -- because that is the part that matters
        # -- the distribution of the loss across groups are all recorded.
        tot = df.groupby(["Monitor", "Channel"])["Activity"].sum(min_count=1)
        alive_mask = tot.fillna(0) >= dead_threshold
        alive = tot[alive_mask].index
        dead = tot[~alive_mask].index

        if len(dead):
            key = df.set_index(["Monitor", "Channel"]).index
            dead_rows = df[key.isin(dead)]
            by_group = {}
            if "Genotype" in dead_rows.columns:
                per_ch = dead_rows.drop_duplicates(["Monitor", "Channel"])
                by_group = {str(k): int(v) for k, v in
                            per_ch["Genotype"].astype(str).value_counts().items()}
            dropped_info.update({
                "dead_channels_dropped": int(len(dead)),
                "dead_channels": [f"{m}_C{c}" for m, c in dead],
                "dead_channels_by_group": by_group,
            })

        df = df.set_index(["Monitor", "Channel"]).loc[
            df.set_index(["Monitor", "Channel"]).index.isin(alive)].reset_index()

    df = df.sort_values(["Monitor", "Channel", "Timestamp"])
    t0 = df["Timestamp"].min()
    out = pd.DataFrame({
        "Genotype": df["Genotype"].astype(str),
        "Treatment": (df["Treatment"].astype(str)
                      if "Treatment" in df.columns else "Untreated"),
        "Sex": df["Sex"].astype(str) if "Sex" in df.columns else "Unknown",
        "FlyID": df["Monitor"].astype(str) + "_C" + df["Channel"].astype(str),
        "Time_min": ((df["Timestamp"] - t0).dt.total_seconds() // 60).astype(int),
        "Activity": df["Activity"],
    })
    for extra in ("Age", "Monitor", "Channel", "Light", "Session"):
        if extra in df.columns:
            out[extra] = df[extra].to_numpy()
    out.loc[out["Treatment"].isin(["nan", ""]), "Treatment"] = "Untreated"
    out = out.reset_index(drop=True)
    # Carry anything the source frame recorded, then this function's own
    # exclusion record. `attrs` is how every other exclusion in this module is
    # reported, so the run summary reads one place (N82).
    for k, v in (getattr(labelled, "attrs", None) or {}).items():
        out.attrs.setdefault(k, v)
    out.attrs.update(dropped_info)
    if dropped_info["dead_channels_dropped"]:
        out.attrs["dead_filter_note"] = (
            f"{dropped_info['dead_channels_dropped']} channel(s) were removed "
            f"for total activity below {dead_threshold:g}: "
            f"{dropped_info['dead_channels']}. By group: "
            f"{dropped_info['dead_channels_by_group']}. Removal by activity is "
            f"differential -- an impaired or treated cohort moves less -- so if "
            f"the loss is concentrated in one group the remaining comparison is "
            f"between the most active members of each, not between the groups "
            f"as loaded. Check these channels against the death report before "
            f"accepting the exclusion."
        )
    return out
