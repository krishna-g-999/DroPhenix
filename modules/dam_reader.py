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
    idx = pd.to_numeric(raw[0], errors="coerce").fillna(0).to_numpy()
    session = (np.diff(idx, prepend=idx[0]) <= 0).cumsum() + 1

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
PLAUSIBLE_SLEEP_MIN_PER_DAY = (300.0, 1150.0)
PLAUSIBLE_ACTIVITY_PER_DAY = (150.0, 5000.0)


def plausibility_report(per_fly: pd.DataFrame) -> dict:
    """
    Compare per-fly sleep and activity against published ranges and say plainly
    when they do not belong there.

    This exists because an implausible number is the hardest kind to catch: it
    is internally consistent, it survives every statistical test, and it looks
    like a finding. Sleep of 90% of the day is not a strain difference, it is a
    signal that the flies were barely moving or the monitor was barely reading
    -- and no amount of downstream analysis will reveal that.
    """
    out: dict = {"flags": [], "ok": True}
    if per_fly is None or per_fly.empty:
        return {"flags": ["No per-fly metrics to check."], "ok": False}

    if "Total_sleep_min_per_day" in per_fly.columns:
        s = per_fly["Total_sleep_min_per_day"].dropna()
        lo, hi = PLAUSIBLE_SLEEP_MIN_PER_DAY
        out["median_sleep_min_per_day"] = float(s.median())
        out["fraction_outside_sleep_range"] = float(((s < lo) | (s > hi)).mean())
        if s.median() > hi:
            out["ok"] = False
            out["flags"].append(
                f"Median sleep is {s.median():.0f} min/day ({s.median() / 14.4:.0f}% of the "
                f"day), above the ~{hi:.0f} min/day upper end of published values for adult "
                "Drosophila. Sleep is scored as five or more consecutive minutes without a "
                "beam break, so anything that stops beam breaks -- a fly that has died, a "
                "misaligned or dirty beam, flies resting away from the beam, or a tube "
                "loaded with food at the wrong end -- is indistinguishable from sleep. "
                "Check activity per day and the survival of the flies before interpreting "
                "this as biology."
            )
        elif s.median() < lo:
            out["ok"] = False
            out["flags"].append(
                f"Median sleep is only {s.median():.0f} min/day, below the ~{lo:.0f} min/day "
                "lower end of published values. Constant beam breaking can indicate a "
                "vibrating monitor or a light leak."
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
    if drop_dead:
        tot = df.groupby(["Monitor", "Channel"])["Activity"].sum(min_count=1)
        alive = tot[tot.fillna(0) >= dead_threshold].index
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
    return out.reset_index(drop=True)
