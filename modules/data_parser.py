"""
DroPhenix Analytics — data_parser.py
Scientific data ingestion, validation, and format normalization
for Drosophila negative geotaxis / climbing assay data.

Canonical long-format schema
----------------------------
Genotype   : str
Sex        : str
Treatment  : str
Replicate  : int
Time       : int or float (seconds)
Count      : int   (# climbed above threshold)
n          : int   (total flies in that replicate/group)

Design principles
-----------------
- Preserve raw information whenever possible.
- Never silently invent missing scientific structure.
- Emit warnings for low-confidence datasets.
- Return canonical long-format tables for downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import re

import numpy as np
import pandas as pd

try:  # package-relative import when loaded as modules.data_parser
    from .design_schema import DESIGN_ALIASES, standardize_design_columns
except ImportError:  # direct-script fallback
    from design_schema import DESIGN_ALIASES, standardize_design_columns


REQUIRED_LONG_COLS = {"Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n"}

# Optional design columns that must participate in the row-uniqueness key.
# Omitting them would silently average genuinely distinct observations -- e.g.
# the same vial measured morning and evening, or two different flies.
DESIGN_KEY_COLS = ["Age", "FlyID", "VialID", "TimeOfDay", "ZT", "SessionIndex",
                   "TrialIndex", "Dose", "Compound", "Batch", "ExperimentID", "Date"]
MIN_N_WARNING = 10
MIN_REPLICATES_WARNING = 3
MIN_TIMEPOINTS_WARNING = 5


@dataclass
class DataAudit:
    """
    The result of auditing a frame: what was wrong, and whether it is usable.

    Register N71. This is a dataclass, and the CSV Analysis page called
    `.get("errors", [])` on it, which raises `AttributeError` -- so the
    validation step that page exists to run never executed, on any file, and
    the page reported neither errors nor warnings while appearing to have
    checked. `to_dict()` exists for callers that genuinely want a mapping (a
    JSON manifest, an export); the fix for the page was to read `.errors` and
    `.warnings`, not to make a dataclass pretend to be a dict, because a silent
    `.get` default is how the step came to be skipped without anyone noticing.
    """

    warnings: List[str]
    errors: List[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors),
                "warnings": list(self.warnings),
                "n_errors": len(self.errors), "n_warnings": len(self.warnings)}


def load_data(
    source: Optional[Union[str, Path, pd.DataFrame, io.BytesIO, io.BufferedReader]] = None,
    duplicate_policy: str = "refuse",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load and standardize climbing assay data.

    Parameters
    ----------
    source
        None -> built-in demo data
        str/Path -> CSV/XLSX path
        DataFrame -> copied directly
        uploaded file-like object -> parsed by extension when possible

    Returns
    -------
    df : pd.DataFrame
        Canonical long-format DataFrame. Always carries `Count_raw`, `n_raw`
        and `QC_corrections`: the raw values exactly as the file gave them, and
        a per-row record of anything found invalid (register N11/N12). An
        invalid observation is NaN in `Count`, never a substituted 0 or n.
    warnings : list[str]
    errors : list[str]

    Other parameters
    ----------------
    duplicate_policy
        "refuse" (default) reports duplicated (Genotype, Sex, Treatment,
        Replicate, Time [, design keys]) rows as an error and collapses
        nothing. "mean" averages them, with half-to-even rounding, and is
        refused anyway when the duplicates disagree about n.
    """
    warnings_out: List[str] = []
    errors_out: List[str] = []

    if source is None:
        df = _get_demo_data()
        warnings_out.append(
            "Demo data loaded from built-in SSSIHL-style dataset "
            "(Oregon, TDP43, FUS; both sexes; multiple treatments)."
        )
        return df, warnings_out, errors_out

    try:
        df_raw = _read_source(source)
    except Exception as e:
        return pd.DataFrame(), warnings_out, [f"Could not read input: {e}"]

    fmt = _detect_format(df_raw)

    if fmt == "long":
        df = df_raw.copy()
    elif fmt == "wide":
        df, w = _convert_wide_to_long(df_raw)
        warnings_out.extend(w)
    else:
        return pd.DataFrame(), warnings_out, [
            "Unrecognized input format. Provide canonical long format or wide format "
            "with recognizable timepoint columns such as t10, t20, 30, 40."
        ]

    df, w, e = _standardize(df)
    warnings_out.extend(w)
    errors_out.extend(e)
    if errors_out:
        return pd.DataFrame(), warnings_out, errors_out

    df, w, e = _validate_science(df, duplicate_policy=duplicate_policy)
    warnings_out.extend(w)
    errors_out.extend(e)

    if errors_out:
        return pd.DataFrame(), warnings_out, errors_out

    df = _finalize_column_order(df)
    return df.reset_index(drop=True), warnings_out, errors_out


def _read_source(
    source: Union[str, Path, pd.DataFrame, io.BytesIO, io.BufferedReader]
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (str, Path)):
        src = str(source)
        if src.lower().endswith(".csv"):
            return pd.read_csv(src)
        return pd.read_excel(src)

    name = str(getattr(source, "name", "")).lower()
    if hasattr(source, "seek"):
        source.seek(0)

    if name.endswith(".csv"):
        return pd.read_csv(source)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(source)

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source)
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_excel(source)


def _detect_format(df: pd.DataFrame) -> str:
    cols = {str(c).strip().lower() for c in df.columns}

    long_core = {"genotype", "sex", "treatment"}
    # Note: must stay in sync with the alias sets in _standardize(). Includes the
    # names used by the shipped data template (count_climbed, n_total_flies).
    # `time` was previously required LITERALLY here while _standardize() also
    # accepts time_sec/seconds/sec, so a file using those aliases was rejected as
    # "unrecognized format" before the renaming that would have fixed it ever ran
    # -- which is exactly what the video analyser's export does.
    time_like = {"time", "time_sec", "seconds", "sec"}
    count_like = {"count", "climbed", "count_climbed", "flies_climbed",
                  "flies climbed", "counts", "n_climbed", "num_climbed"}
    n_like = {"n", "total", "total_flies", "n_total_flies", "total flies",
              "n_total", "ntotal", "total_n"}

    if (long_core.issubset(cols) and (cols & time_like)
            and (cols & count_like) and (cols & n_like)):
        return "long"

    time_cols = []
    for c in df.columns:
        s = str(c).strip().lower()
        s = re.sub(r"^(time_|t|sec_|s_)", "", s)
        s = s.replace("sec", "").replace("s", "") if s.isdigit() else s
        if re.fullmatch(r"\d+(\.\d+)?", s):
            time_cols.append(c)

    if len(time_cols) >= 3:
        return "wide"

    return "unknown"


def _convert_wide_to_long(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    meta_cols = []
    time_cols = []

    for col in df.columns:
        label = str(col).strip().lower()
        stripped = re.sub(r"^(time_|t|sec_|s_)", "", label)
        if re.fullmatch(r"\d+(\.\d+)?", stripped):
            time_cols.append(col)
        else:
            meta_cols.append(col)

    if not time_cols:
        return df, ["Could not identify timepoint columns in wide format."]

    meta_lower = {c.lower(): c for c in meta_cols}
    if "replicate" not in meta_lower:
        df["Replicate"] = 1
        meta_cols.append("Replicate")
        warnings.append("No Replicate column found in wide file; assigned Replicate=1.")

    # Register N72. The cohort-size column is renamed to `n` here, and
    # `meta_cols` -- which becomes `melt`'s `id_vars` -- was built BEFORE that
    # rename. So a wide file whose total column is literally named `Total`,
    # which is the first name this very loop looks for, left "Total" in
    # `id_vars` after the column had become "n", and `melt` raised an uncaught
    # KeyError that escaped `load_data` as a red traceback on upload. For an
    # advertised format. The rename is now reflected in `meta_cols`.
    if "n" not in meta_lower:
        for candidate in ["Total", "total", "total_flies", "N", "n_total"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "n"})
                meta_cols = ["n" if c == candidate else c for c in meta_cols]
                meta_lower = {c.lower(): c for c in meta_cols}
                warnings.append(
                    f"Wide file: column {candidate!r} read as the cohort size n."
                )
                break
        else:
            # No cohort size at all. `_standardize` would report this a step
            # later as a missing required column, but it is clearer to say it
            # here, where the wide-format assumption is being made.
            warnings.append(
                "Wide file has no column that looks like the number of flies "
                "per vial (looked for Total, total, total_flies, N, n_total). "
                "Every climbing rate needs that denominator, so the file will "
                "be refused as incomplete."
            )

    # Whatever is in `id_vars` must exist, or melt raises. Anything that was
    # renamed or dropped above is caught here rather than four frames up.
    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        return df, warnings + [
            f"Internal error converting the wide file: the columns "
            f"{missing_meta} were expected to carry metadata but are not in "
            f"the frame after renaming. Please report this file."
        ]

    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=time_cols,
        var_name="Time",
        value_name="Count",
    )

    df_long["Time"] = (
        df_long["Time"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"^(time_|t|sec_|s_)", "", regex=True)
        .astype(float)
    )

    warnings.append(f"Wide format converted to long format ({len(time_cols)} timepoints detected).")
    return df_long, warnings


def _standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    df = df.copy()

    # Optional extended design columns (FlyID, VialID, TimeOfDay, Dose, Batch, ...)
    # are canonicalised first so the core renaming below sees stable names.
    df, design_warnings = standardize_design_columns(df)
    warnings.extend(design_warnings)

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl == "genotype":
            rename_map[col] = "Genotype"
        elif cl == "sex":
            rename_map[col] = "Sex"
        elif cl == "treatment":
            rename_map[col] = "Treatment"
        elif cl in {"replicate", "rep", "trial"}:
            rename_map[col] = "Replicate"
        elif cl in {"time", "time_sec", "seconds", "sec"}:
            rename_map[col] = "Time"
        elif cl in {"count", "climbed", "count_climbed", "flies_climbed",
                    "flies climbed", "counts", "n_climbed", "num_climbed"}:
            rename_map[col] = "Count"
        elif cl in {"n", "total", "total_flies", "n_total_flies", "total flies",
                    "n_total", "ntotal", "total_n"}:
            rename_map[col] = "n"
        elif cl in {"age", "age_days", "ageday", "age_day", "agedays"}:
            rename_map[col] = "Age"   # optional longitudinal axis for MDR

    df = df.rename(columns=rename_map)

    missing = REQUIRED_LONG_COLS - set(df.columns)
    if missing:
        return df, warnings, [f"Missing required columns: {sorted(missing)}"]

    before = len(df)
    df = df.dropna(subset=["Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n"])
    dropped = before - len(df)
    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows with missing critical values.")

    try:
        df["Replicate"] = pd.to_numeric(df["Replicate"], errors="coerce")
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
        df["n"] = pd.to_numeric(df["n"], errors="coerce")
    except Exception as e:
        return df, warnings, [f"Numeric conversion failed: {e}"]

    if df[["Replicate", "Time", "Count", "n"]].isna().any().any():
        bad_rows = int(df[["Replicate", "Time", "Count", "n"]].isna().any(axis=1).sum())
        return df, warnings, [f"{bad_rows} rows contain non-numeric required values after parsing."]

    df["Replicate"] = df["Replicate"].astype(int)
    df["Count"] = df["Count"].astype(int)

    if np.allclose(df["Time"], np.round(df["Time"]), equal_nan=False):
        df["Time"] = df["Time"].astype(int)

    sex_clean = (
        df["Sex"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "m": "Male",
                "male": "Male",
                "f": "Female",
                "female": "Female",
            }
        )
    )
    df["Sex"] = sex_clean

    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df["Treatment"] = df["Treatment"].astype(str).str.strip()

    return df, warnings, errors


DUPLICATE_POLICIES = ("refuse", "mean")


def _note(df: pd.DataFrame, mask, text: str) -> None:
    """Append a per-row correction note, so the record travels with the data."""
    if not mask.any():
        return
    cur = df.loc[mask, "QC_corrections"].astype(str)
    df.loc[mask, "QC_corrections"] = np.where(cur.str.len() > 0,
                                              cur + "; " + text, text)


def _validate_science(df: pd.DataFrame,
                      duplicate_policy: str = "refuse"
                      ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Scientific validation of a standardised long-format table.

    Register N11/N12. This function used to WRITE values that were never
    observed: a negative count became 0, a count above n was capped at n,
    duplicate rows were averaged and rounded, and when duplicates disagreed
    about n the collapse kept the `first` one -- silently discarding a vial.
    Each of those is a measurement the instrument did not produce, and the
    frame carrying them was then shown to the user as the upload "Preview",
    with the only record of the change a transient warning on one page.

    What happens now.
      * `Count_raw` and `n_raw` preserve exactly what the file said, on every
        row, always.
      * An invalid observation -- a negative count, or a count above n -- is
        set to NaN and flagged. NaN means "this number is not usable"; 0 means
        "no fly climbed" and n means "every fly climbed", and both are real
        results that must not be manufactured from an error.
      * `QC_corrections` names, per row, what was found. It is a column, so it
        survives into every later page and every export.
      * Duplicates require an explicit policy. The default is "refuse": the
        duplicated key is reported and nothing is collapsed, because averaging
        two disagreeing records of the same vial at the same timepoint is a
        decision only the experimenter can make. "mean" reproduces the old
        behaviour, on request, and still refuses when the duplicates disagree
        about n.
    """
    warnings: List[str] = []
    errors: List[str] = []
    df = df.copy()
    if str(duplicate_policy) not in DUPLICATE_POLICIES:
        errors.append(f"duplicate_policy must be one of {DUPLICATE_POLICIES}; "
                      f"got {duplicate_policy!r}.")
        return df, warnings, errors

    # Raw values first, before anything can touch them.
    if "Count_raw" not in df.columns:
        df["Count_raw"] = pd.to_numeric(df["Count"], errors="coerce")
    if "n_raw" not in df.columns:
        df["n_raw"] = pd.to_numeric(df["n"], errors="coerce")
    if "QC_corrections" not in df.columns:
        df["QC_corrections"] = ""
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")

    neg_time = df["Time"] < 0
    if neg_time.any():
        errors.append(f"{int(neg_time.sum())} rows have negative Time values.")

    nonpositive_n = df["n"] <= 0
    if nonpositive_n.any():
        errors.append(f"{int(nonpositive_n.sum())} rows have n <= 0, which is invalid for normalization.")

    # A negative count is not a count of zero. Setting it to 0 asserted that no
    # fly climbed in that vial at that timepoint -- a real, publishable result
    # manufactured from a data-entry error.
    neg_count = df["Count"] < 0
    if neg_count.any():
        nneg = int(neg_count.sum())
        df.loc[neg_count, "Count"] = np.nan
        _note(df, neg_count, "INVALID: negative Count; raw value kept in "
                             "Count_raw, Count withheld as NaN")
        warnings.append(
            f"{nneg} rows had a negative Count. The raw value is preserved in "
            f"Count_raw and Count is NaN: a negative count is not an observation "
            f"of zero climbers, and these rows are excluded from every mean "
            f"rather than contributing a fabricated zero. See QC_corrections."
        )

    # Nor is a count above n an observation that every fly climbed. Capping at n
    # produced exactly 100%, which is the one number that looks like a clean
    # result while the loading record and the count disagree.
    too_high = df["Count"] > df["n"]
    if too_high.any():
        nbad = int(too_high.sum())
        df.loc[too_high, "Count"] = np.nan
        _note(df, too_high, "INVALID: Count exceeds n; raw value kept in "
                            "Count_raw, Count withheld as NaN")
        warnings.append(
            f"{nbad} rows had Count > n. The raw value is preserved in Count_raw "
            f"and Count is NaN: capping at n reported 100% climbing for a vial "
            f"whose count and loading record contradict each other. Check the "
            f"loading record and the scoring sheet for those rows."
        )

    invalid_sex = ~df["Sex"].isin(["Male", "Female"])
    if invalid_sex.any():
        vals = sorted(df.loc[invalid_sex, "Sex"].astype(str).unique().tolist())
        warnings.append(f"Unrecognized Sex labels found: {vals}")

    # Age (optional longitudinal axis for MDR) must be part of the uniqueness
    # key, otherwise the same (Genotype,Sex,Treatment,Replicate,Time) measured at
    # different ages would be wrongly collapsed into one row.
    key = ["Genotype", "Sex", "Treatment", "Replicate", "Time"]
    key += [c for c in DESIGN_KEY_COLS if c in df.columns and df[c].notna().any()]
    dup_mask = df.duplicated(subset=key, keep=False)
    if dup_mask.any():
        dup_n = int(dup_mask.sum())
        dup_keys = (df.loc[dup_mask, key].drop_duplicates()
                    .astype(str).agg(" | ".join, axis=1).tolist())
        if duplicate_policy == "refuse":
            errors.append(
                f"{dup_n} rows duplicate the key {'×'.join(key)}: "
                f"{dup_keys[:6]}{' ...' if len(dup_keys) > 6 else ''}. Two records "
                f"of the same vial at the same Time are either a transcription "
                f"error or two technical reads of one measurement, and averaging "
                f"them writes a value neither record contains -- duplicates of 4 "
                f"and 5 become 4, because np.round is half-to-even. Which it is, "
                f"and what to do, is an experimental judgement: fix the file, or "
                f"pass duplicate_policy='mean' to average them deliberately."
            )
            return df, warnings, errors

        # duplicate_policy == "mean": the old behaviour, now explicit, and it
        # still refuses when the duplicates disagree about the cohort size. The
        # previous `n=("n", "first")` silently discarded a vial in that case.
        n_disagree = (df[dup_mask].groupby(key, dropna=False)["n"].nunique() > 1)
        if bool(n_disagree.any()):
            errors.append(
                f"{int(n_disagree.sum())} duplicated keys disagree about n (the "
                f"number of flies in the vial). Collapsing them by taking the "
                f"first n discards one vial's loading record while averaging its "
                f"count into the other's denominator. Duplicates can only be "
                f"averaged when they describe the same vial of the same size."
            )
            return df, warnings, errors

        warnings.append(
            f"{dup_n} duplicated rows for {'×'.join(key)} were collapsed by mean "
            f"Count (half-to-even rounding) at the caller's explicit request; "
            f"Count_raw retains the first raw value and QC_corrections records "
            f"the collapse on each resulting row."
        )
        df = (
            df.groupby(key, dropna=False, as_index=False)
            .agg(
                Count=("Count", "mean"),
                n=("n", "first"),
                _n_collapsed=("Count", "size"),
                **{
                    c: (c, "first")
                    for c in df.columns
                    if c not in key + ["Count", "n"]
                },
            )
        )
        coll = df["_n_collapsed"] > 1
        _note(df, coll, "duplicates averaged at caller request")
        df = df.drop(columns=["_n_collapsed"])
        # Rounded, not truncated, and NaN-preserving: a row whose count was
        # withheld above must not be resurrected as an integer.
        df["Count"] = np.where(df["Count"].notna(),
                               np.round(df["Count"].astype(float)), np.nan)

    n_consistency = (
        df.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False)["n"]
        .nunique()
    )
    bad_n = n_consistency[n_consistency > 1]
    if not bad_n.empty:
        warnings.append(
            f"{len(bad_n)} replicate groups show inconsistent n across timepoints; "
            f"n was retained as provided, but this should be checked in raw records."
        )

    bad_time_order = 0
    for _, grp in df.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False):
        times = grp["Time"].astype(float).tolist()
        if len(times) > 1 and any(np.diff(sorted(times)) == 0):
            bad_time_order += 1
    if bad_time_order > 0:
        warnings.append(
            f"{bad_time_order} replicate groups contain repeated time values after standardization."
        )

    group_cols = ["Genotype", "Sex", "Treatment"]
    rep_counts = df.groupby(group_cols, dropna=False)["Replicate"].nunique()
    if not rep_counts.empty and int(rep_counts.min()) < MIN_REPLICATES_WARNING:
        warnings.append(
            f"Some groups have fewer than {MIN_REPLICATES_WARNING} biological replicates."
        )

    tp_counts = df.groupby(group_cols + ["Replicate"], dropna=False)["Time"].nunique()
    if not tp_counts.empty and int(tp_counts.min()) < MIN_TIMEPOINTS_WARNING:
        warnings.append(
            f"Some replicates have fewer than {MIN_TIMEPOINTS_WARNING} timepoints."
        )

    small_n = sorted(df.loc[df["n"] < MIN_N_WARNING, "n"].dropna().astype(int).unique().tolist())
    if small_n:
        warnings.append(
            f"Some rows have n < {MIN_N_WARNING}: {small_n}. "
            f"Recommended assay size is typically >= 15 flies per replicate."
        )

    df = df.sort_values(["Genotype", "Sex", "Treatment", "Replicate", "Time"]).reset_index(drop=True)
    return df, warnings, errors


def _finalize_column_order(df: pd.DataFrame) -> pd.DataFrame:
    preferred = ["Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n",
                 "Count_raw", "n_raw", "QC_corrections"]
    extra = [c for c in df.columns if c not in preferred]
    return df[preferred + extra]


def get_groups(df: pd.DataFrame) -> Dict[str, List[Any]]:
    return {
        "genotypes": sorted(df["Genotype"].dropna().astype(str).unique().tolist()),
        "sexes": sorted(df["Sex"].dropna().astype(str).unique().tolist()),
        "treatments": sorted(df["Treatment"].dropna().astype(str).unique().tolist()),
        "replicates": sorted(df["Replicate"].dropna().astype(int).unique().tolist()),
        "timepoints": sorted(pd.to_numeric(df["Time"], errors="coerce").dropna().tolist()),
    }


def get_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Genotype", "Sex", "Treatment", "Replicate"], dropna=False)
        .agg(
            n=("n", "first"),
            timepoints=("Time", "nunique"),
            min_time=("Time", "min"),
            max_time=("Time", "max"),
            min_count=("Count", "min"),
            max_count=("Count", "max"),
        )
        .reset_index()
    )


def validate_dataframe(df: pd.DataFrame) -> DataAudit:
    df_std, warnings_1, errors_1 = _standardize(df.copy())
    if errors_1:
        return DataAudit(warnings=warnings_1, errors=errors_1)
    _, warnings_2, errors_2 = _validate_science(df_std.copy(),
                                                duplicate_policy="refuse")
    return DataAudit(warnings=warnings_1 + warnings_2, errors=errors_2)


def parse_uploaded_file(*args, **kwargs):
    """
    Backward-compatible alias used by older DroPhenix code.
    """
    return load_data(*args, **kwargs)


def _get_demo_data() -> pd.DataFrame:
    rows = []
    untreated_data = {
        ("Oregon", "Male"): [[10, 12, 13, 13, 15, 15, 15, 15], [5, 8, 12, 15, 15, 15, 15, 15], [6, 10, 10, 12, 15, 15, 15, 15]],
        ("Oregon", "Female"): [[6, 8, 9, 11, 15, 15, 15, 15], [12, 14, 15, 15, 15, 15, 15, 15], [12, 15, 15, 14, 15, 15, 15, 15]],
        ("TDP43", "Male"): [[4, 6, 8, 10, 12, 13, 14, 14], [3, 5, 7, 9, 11, 12, 13, 13], [5, 7, 9, 11, 12, 13, 14, 14]],
        ("TDP43", "Female"): [[5, 7, 9, 11, 13, 13, 14, 14], [6, 8, 10, 12, 13, 14, 14, 14], [4, 7, 8, 10, 12, 13, 13, 14]],
        ("FUS", "Male"): [[2, 4, 6, 8, 9, 10, 11, 11], [3, 4, 7, 8, 9, 10, 11, 11], [2, 5, 6, 8, 9, 10, 10, 11]],
        ("FUS", "Female"): [[3, 5, 7, 9, 10, 11, 12, 12], [4, 6, 8, 10, 11, 12, 12, 12], [3, 5, 7, 9, 10, 11, 11, 12]],
    }
    tp_untreated = [10, 20, 30, 40, 60, 80, 100, 120]

    for (geno, sex), trials in untreated_data.items():
        for rep_idx, counts in enumerate(trials, start=1):
            for t, c in zip(tp_untreated, counts):
                rows.append(
                    {
                        "Genotype": geno,
                        "Sex": sex,
                        "Treatment": "Untreated",
                        "Replicate": rep_idx,
                        "Time": t,
                        "Count": c,
                        "n": 15,
                    }
                )

    tp_treated = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]
    treated_data = {
        "Spermine": {
            ("Oregon", "Male", 16): [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
            ("Oregon", "Female", 18): [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
            ("TDP43", "Male", 15): [8, 10, 11, 11, 11, 10, 9, 8, 7, 7, 7],
            ("TDP43", "Female", 15): [9, 11, 12, 12, 12, 11, 10, 9, 8, 7, 7],
            ("FUS", "Male", 20): [12, 14, 15, 15, 16, 16, 16, 16, 15, 15, 15],
            ("FUS", "Female", 20): [14, 16, 17, 18, 18, 18, 18, 17, 17, 16, 16],
        },
        "Pantothenate": {
            ("Oregon", "Male", 25): [22, 23, 23, 23, 22, 20, 18, 17, 16, 16, 16],
            ("Oregon", "Female", 15): [12, 12, 12, 12, 12, 12, 11, 9, 9, 9, 9],
            ("TDP43", "Male", 20): [10, 12, 13, 13, 13, 12, 11, 10, 10, 9, 9],
            ("TDP43", "Female", 25): [14, 16, 17, 18, 18, 18, 17, 15, 13, 12, 12],
            ("FUS", "Male", 15): [9, 11, 12, 12, 12, 12, 11, 10, 10, 10, 10],
            ("FUS", "Female", 25): [12, 14, 14, 14, 14, 13, 12, 10, 8, 7, 7],
        },
        "Spermine+Pantothenate": {
            ("Oregon", "Male", 20): [17, 18, 18, 18, 17, 16, 14, 13, 12, 12, 12],
            ("Oregon", "Female", 20): [18, 19, 19, 18, 17, 16, 16, 16, 14, 11, 11],
            ("TDP43", "Male", 30): [18, 20, 22, 23, 23, 22, 20, 19, 18, 17, 16],
            ("TDP43", "Female", 18): [11, 13, 13, 14, 14, 13, 13, 13, 13, 13, 13],
            ("FUS", "Male", 11): [6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            ("FUS", "Female", 20): [11, 12, 12, 12, 12, 11, 10, 10, 10, 10, 10],
        },
    }

    for treatment, groups in treated_data.items():
        for (geno, sex, n_flies), counts in groups.items():
            for t, c in zip(tp_treated, counts):
                rows.append(
                    {
                        "Genotype": geno,
                        "Sex": sex,
                        "Treatment": treatment,
                        "Replicate": 1,
                        "Time": t,
                        "Count": c,
                        "n": n_flies,
                    }
                )

    df = pd.DataFrame(rows)
    return df.astype({"Replicate": int, "Count": int, "n": int})