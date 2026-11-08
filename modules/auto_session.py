"""
DroPhenix — auto_session.py
Bridges RecordingSession raw counts -> DroPhenix normalizer/metrics pipeline.

After recording completes:
  1. Reshapes raw_counts into DroPhenix standard long-format DataFrame
  2. Calls compute_all_metrics() and compute_novel_metrics()
  3. Returns base_metrics + novel_metrics ready for visualizer
  4. Optionally exports full Excel results workbook
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from modules import missingness as MS


def build_vial_registry(vial_configs: List[dict]) -> dict:
    """
    Build a vial_id -> metadata lookup dict.
    """
    return {cfg["vial_id"]: cfg for cfg in vial_configs}


_N_START_KEYS = ("N_start", "n_start", "expected_n", "N", "n")


def _registry_n_start(meta: dict, vid) -> int:
    """N_start for one vial, from the loading record. Refuse if absent.

    Register N5: the denominator is the number of flies LOADED. There is no
    default, because a default denominator produces a plausible percentage for a
    vial whose loading was never recorded.
    """
    for k in _N_START_KEYS:
        if k in meta and pd.notna(meta[k]):
            try:
                n = int(meta[k])
            except (TypeError, ValueError):
                continue
            if n > 0:
                return n
    raise ValueError(
        f"auto_session: vial {vid} has no N_start in the vial registry "
        f"(looked for {', '.join(_N_START_KEYS)}). The number of flies loaded "
        f"is the only legitimate climbing denominator, so this conversion is "
        f"refused rather than computed against the detected total."
    )


def _round_count(v):
    """Round a possibly-fractional count to the nearest integer, or keep NaN.

    Register N79: this was `int()`, which truncates. Counts here are genuinely
    fractional — the modern detectors take a median over a frame window and
    `climbing_band` applies a merge correction — so truncation was a systematic
    DOWNWARD bias on every vial at every timepoint, i.e. a bias toward the
    impaired phenotype in the climbing assay. `np.round` is half-to-even
    (6.5 -> 6, 7.5 -> 8), which is unbiased over many vials; the unrounded value
    is kept in `Count_raw` so nothing is lost.
    """
    f = pd.to_numeric(v, errors="coerce")
    if pd.isna(f):
        return float("nan")
    return float(np.round(float(f)))


def raw_counts_to_drophenix(
    raw_df: pd.DataFrame,
    vial_registry: dict,
    replicate: int = 1,
) -> pd.DataFrame:
    """
    Convert RecordingSession.results_df to DroPhenix standard long format.

    Three Block B behaviours, each replacing a silent failure:

    * **The denominator is N_start** from `vial_registry`, not the row's
      detected `Total` (register N5). A detected total differing from N_start is
      carried as `N_detected` and flagged, never used as the denominator.
    * **A failed measurement stays a failed measurement** (register N8): a NaN
      `n_above` yields `Count = NaN` and `Missingness = MEASUREMENT_FAILED`.
      The row is kept — dropping it would hide the failure, and failures are not
      independent of genotype — but it carries no number that could be averaged.
    * **Nothing is cast with `int()`** (register N79). `int()` raised on NaN,
      which coupled this function to the N8 fix, and truncated fractional counts
      downward. Both are gone.

    A row whose `VialID` is not in the registry is an error, not something to
    skip: silently dropping it removes an observation with no record.
    """
    for col in ("VialID", "Time", "n_above"):
        if col not in raw_df.columns:
            raise ValueError(
                f"auto_session.raw_counts_to_drophenix: required column "
                f"{col!r} is absent (columns present: "
                f"{list(raw_df.columns)}). Refused rather than guessed."
            )

    has_total = "Total" in raw_df.columns or "N_detected" in raw_df.columns
    det_col = "N_detected" if "N_detected" in raw_df.columns else "Total"

    rows = []
    unknown = []
    for _, row in raw_df.iterrows():
        vid = row["VialID"]
        try:
            vid_key = int(vid)
        except (TypeError, ValueError):
            vid_key = vid
        meta = vial_registry.get(vid_key, vial_registry.get(vid))
        if meta is None:
            unknown.append(vid)
            continue

        n_start = _registry_n_start(meta, vid_key)
        count = _round_count(row["n_above"])
        det = pd.to_numeric(row[det_col], errors="coerce") if has_total else float("nan")

        qc = str(row.get("QC_flag", "") or "")
        miss = str(row.get(MS.COLUMN, "") or "")
        if not miss:
            # The recorder may predate the Missingness column. Derive the code
            # from what is observable rather than assuming the row is good.
            if pd.isna(count):
                miss = MS.MEASUREMENT_FAILED
            elif qc.startswith("unusable") or qc == "not_measured":
                miss = MS.NOT_MEASURED
            else:
                miss = MS.OBSERVED
        if pd.isna(count) and miss == MS.OBSERVED:
            miss = MS.MEASUREMENT_FAILED
        # And the converse: a row typed as failed or never measured cannot
        # carry a count, whatever the recorder wrote there. A 0 against
        # n = N_start computes PI = -100 for a vial that was not scored.
        if miss in (MS.MEASUREMENT_FAILED, MS.NOT_MEASURED):
            count = float("nan")

        rows.append({
            "Genotype": meta.get("Genotype", ""),
            "Sex": meta.get("Sex", ""),
            "Treatment": meta.get("Treatment", ""),
            "VialID": vid_key,
            "Replicate": pd.to_numeric(row.get("Replicate", replicate),
                                       errors="coerce"),
            # Time is NOT cast to int: a 10.5 s scoring time is a real
            # timepoint, and truncating it merged distinct timepoints.
            "Time": pd.to_numeric(row["Time"], errors="coerce"),
            "Count": count,
            "Count_raw": pd.to_numeric(row["n_above"], errors="coerce"),
            "n": n_start,
            "N_start": n_start,
            "N_detected": det,
            "QC_flag": qc or ("ok" if miss == MS.OBSERVED else miss),
            "Confidence": pd.to_numeric(row.get("Confidence", np.nan),
                                        errors="coerce"),
            MS.COLUMN: miss,
        })

    if unknown:
        raise ValueError(
            f"auto_session.raw_counts_to_drophenix: VialID "
            f"{sorted({str(u) for u in unknown})} not in the vial registry "
            f"(keys: {sorted(str(k) for k in vial_registry)}). These rows were "
            f"previously dropped without a record; they are refused instead."
        )

    return pd.DataFrame(rows)


def run_auto_pipeline(
    raw_df: pd.DataFrame,
    vial_configs: List[dict],
    replicate: int = 1,
    export_path: Optional[str] = None,
) -> Tuple[dict, dict]:
    """
    End-to-end: raw counts -> base metrics -> novel metrics.
    """
    from modules.normalizer import compute_all_metrics
    from modules.metrics import compute_novel_metrics

    registry = build_vial_registry(vial_configs)
    df_std = raw_counts_to_drophenix(raw_df, registry, replicate)

    if df_std.empty:
        raise ValueError(
            "No valid data after vial registry matching. "
            "Verify that VialID values in raw_df match vial_registry keys."
        )

    base = compute_all_metrics(df_std)
    novel = compute_novel_metrics(df_std, base)

    if export_path:
        _export_excel(base, novel, export_path)

    return base, novel


def _export_excel(base: dict, novel: dict, path: str) -> None:
    """Write all metric tables to a multi-sheet Excel workbook."""
    ecs_raw = novel.get("ecs", {})
    ecs_rows = []
    for k, v in ecs_raw.get("details", {}).items():
        ecs_rows.append({"Criterion": k, "Value": str(v)})
    ecs_df = pd.DataFrame(ecs_rows) if ecs_rows else pd.DataFrame(columns=["Criterion", "Value"])
    if ecs_raw:
        header_rows = pd.DataFrame([
            {"Criterion": "Score", "Value": str(ecs_raw.get("score", ""))},
            {"Criterion": "Grade", "Value": str(ecs_raw.get("grade", ""))},
        ])
        ecs_df = pd.concat([header_rows, ecs_df], ignore_index=True)

    sheets = {
        "PI": base.get("pi", pd.DataFrame()),
        "AUC": base.get("auc", pd.DataFrame()),
        "t50": base.get("t50", pd.DataFrame()),
        "Aggregated": base.get("aggregated", pd.DataFrame()),
        "CRI": novel.get("cri", pd.DataFrame()),
        "SDQ": novel.get("sdq", pd.DataFrame()),
        "TSW": novel.get("tsw", pd.DataFrame()),
        "ECS": ecs_df,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def partial_update(
    partial_raw_df: pd.DataFrame,
    vial_configs: List[dict],
    replicate: int = 1,
) -> pd.DataFrame:
    """
    Called at each timepoint during live recording.
    Returns minimal aggregated table for real-time plotting.
    Does NOT run full CRI/SDQ (needs all timepoints first).
    """
    if partial_raw_df.empty:
        return pd.DataFrame()

    registry = build_vial_registry(vial_configs)
    df_std = raw_counts_to_drophenix(partial_raw_df, registry, replicate)
    if df_std.empty:
        return pd.DataFrame()

    df_std["Pct"] = np.where(
        (df_std["n"] > 0) & df_std["Count"].notna(),
        (df_std["Count"] / df_std["n"]) * 100.0,
        np.nan,
    )
    # N8/N5: a vial that was not measured must not contribute its loading
    # number to the denominator sum. Before this, `n=("n","sum")` added the
    # N_start of every failed vial while `Pct_mean` excluded it, so the live
    # plot reported a cohort larger than the one it had averaged.
    observed = df_std[MS.COLUMN].map(MS.is_observed) & df_std["Pct"].notna()
    df_std["_n_obs"] = np.where(observed, df_std["n"], 0)
    df_std["_failed"] = (~observed).astype(int)
    summary = (
        df_std.groupby(["Genotype", "Sex", "Treatment", "Time"])
        .agg(Pct_mean=("Pct", "mean"),
             n=("_n_obs", "sum"),
             n_vials_observed=("_failed", lambda s: int((s == 0).sum())),
             n_failed=("_failed", "sum"))
        .reset_index()
    )
    return summary