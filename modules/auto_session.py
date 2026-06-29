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


def build_vial_registry(vial_configs: List[dict]) -> dict:
    """
    Build a vial_id -> metadata lookup dict.
    """
    return {cfg["vial_id"]: cfg for cfg in vial_configs}


def raw_counts_to_drophenix(
    raw_df: pd.DataFrame,
    vial_registry: dict,
    replicate: int = 1,
) -> pd.DataFrame:
    """
    Convert RecordingSession.results_df to DroPhenix standard long format.
    """
    rows = []
    for _, row in raw_df.iterrows():
        vid = int(row["VialID"])
        meta = vial_registry.get(vid)
        if meta is None:
            continue
        rows.append({
            "Genotype": meta["Genotype"],
            "Sex": meta["Sex"],
            "Treatment": meta["Treatment"],
            "Replicate": int(row.get("Replicate", replicate)),
            "Time": int(row["Time"]),
            "Count": int(row["n_above"]),
            "n": int(row["Total"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Count"] = df["Count"].astype(int)
        df["n"] = df["n"].astype(int)
        df["Time"] = df["Time"].astype(int)
        df["Replicate"] = df["Replicate"].astype(int)
    return df


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
        df_std["n"] > 0,
        (df_std["Count"] / df_std["n"]) * 100.0,
        np.nan,
    )
    summary = (
        df_std.groupby(["Genotype", "Sex", "Treatment", "Time"])
        .agg(Pct_mean=("Pct", "mean"), n=("n", "sum"))
        .reset_index()
    )
    return summary