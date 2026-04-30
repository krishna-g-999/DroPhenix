"""
DroPhenix Analytics — trajectory_cluster.py
DTW-based trajectory clustering for Drosophila climbing assay time-series.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ─── DTW distance ─────────────────────────────────────────────────────────────
def dtw_distance(
    s: np.ndarray,
    t: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """
    Compute DTW distance between two 1-D time series s and t.
    FIX: enforces scalar dtype so dtw_mat[i,j] never receives a sequence.
    """
    s = np.asarray(s, dtype=float).ravel()   # FIX: force 1-D float
    t = np.asarray(t, dtype=float).ravel()   # FIX: force 1-D float
    n, m = len(s), len(t)

    w = max(abs(n - m), window) if window is not None else max(n, m)

    dtw_mat = np.full((n + 1, m + 1), np.inf, dtype=float)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end   = min(m + 1, i + w + 1)
        for j in range(j_start, j_end):
            cost = float((s[i - 1] - t[j - 1]) ** 2)   # FIX: scalar float
            prev = float(min(
                dtw_mat[i - 1, j],
                dtw_mat[i,     j - 1],
                dtw_mat[i - 1, j - 1],
            ))
            dtw_mat[i, j] = cost + prev                 # FIX: always scalar

    return float(np.sqrt(dtw_mat[n, m]))


def dtw_distance_matrix(
    trajectories: Dict[str, np.ndarray],
    window: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute full pairwise DTW distance matrix."""
    labels = list(trajectories.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(
                trajectories[labels[i]],
                trajectories[labels[j]],
                window=window,
            )
            mat[i, j] = d
            mat[j, i] = d

    return mat, labels


# ─── Trajectory extraction ────────────────────────────────────────────────────
def build_trajectory_dict(
    pi_df: pd.DataFrame,
    common_timepoints: Optional[List[int]] = None,
    interpolate: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """
    Build {label: PI_array} dict from PI summary table.
    FIX: aggregates replicates to mean per timepoint before building arrays.
    """
    if pi_df is None or pi_df.empty:
        return {}, {}

    df = pi_df.copy()

    if "PI_mean" not in df.columns and "PI" in df.columns:
        df = df.rename(columns={"PI": "PI_mean"})
    if "Time" not in df.columns and "Time_used" in df.columns:
        df = df.rename(columns={"Time_used": "Time"})

    required = {"Genotype", "Treatment", "Sex", "Time", "PI_mean"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"trajectory_cluster: PI table missing: {required - set(df.columns)}"
        )

    trajectories: Dict[str, np.ndarray] = {}
    metadata: Dict[str, dict] = {}

    for (geno, treat, sex), grp in df.groupby(
        ["Genotype", "Treatment", "Sex"], dropna=False
    ):
        # FIX: collapse replicates → single mean PI per timepoint
        grp_mean = (
            grp.groupby("Time", as_index=False)["PI_mean"]
               .mean()
               .sort_values("Time")
               .dropna(subset=["PI_mean"])
        )
        if len(grp_mean) < 2:
            continue

        times = grp_mean["Time"].astype(float).values.ravel()  # FIX: 1-D
        pis   = grp_mean["PI_mean"].astype(float).values.ravel()  # FIX: 1-D

        if interpolate and common_timepoints is not None:
            common_t = np.array(sorted(set(common_timepoints)), dtype=float)
            t_min, t_max = times[0], times[-1]
            valid_common = common_t[(common_t >= t_min) & (common_t <= t_max)]
            pis_aligned = np.interp(valid_common, times, pis) if len(valid_common) >= 2 else pis
        else:
            pis_aligned = pis

        label = f"{geno} | {treat} | {sex}"
        trajectories[label] = pis_aligned.ravel()  # FIX: guarantee 1-D
        metadata[label] = {
            "Genotype": geno,
            "Treatment": treat,
            "Sex": sex,
            "n_timepoints": len(grp_mean),
        }

    return trajectories, metadata


# ─── Optimal cluster number ───────────────────────────────────────────────────
def _optimal_k(
    dist_matrix: np.ndarray,
    k_range: Tuple[int, int] = (2, 8),
) -> Tuple[int, pd.DataFrame]:
    n = dist_matrix.shape[0]
    k_max = min(k_range[1], n - 1)
    k_min = min(k_range[0], k_max)

    if k_max < 2:
        return 2, pd.DataFrame()

    cond = squareform(dist_matrix, checks=False)
    Z = linkage(cond, method="ward")

    sil_rows = []
    best_k, best_sil = k_min, -1.0

    for k in range(k_min, k_max + 1):
        lbls = fcluster(Z, k, criterion="maxclust")
        if len(set(lbls)) < 2:
            continue
        try:
            sil = silhouette_score(dist_matrix, lbls, metric="precomputed")
        except Exception:
            sil = np.nan
        sil_rows.append({"k": k, "silhouette": round(float(sil), 4) if pd.notna(sil) else np.nan})
        if pd.notna(sil) and sil > best_sil:
            best_sil = sil
            best_k = k

    return best_k, pd.DataFrame(sil_rows)


# ─── Barycenter (archetype) computation ──────────────────────────────────────
def _dtw_barycenter(sequences: List[np.ndarray], n_iter: int = 10) -> np.ndarray:
    if not sequences:
        return np.array([])

    max_len = max(len(s) for s in sequences)
    padded = [np.pad(s, (0, max_len - len(s)), mode="edge") for s in sequences]
    center = np.mean(padded, axis=0)

    for _ in range(n_iter):
        assignments = [[] for _ in range(max_len)]
        for s in sequences:
            for i in range(max_len):
                best_j = int(np.argmin(np.abs(s - center[i]))) if len(s) > 0 else 0
                best_j = min(best_j, len(s) - 1)
                assignments[i].append(float(s[best_j]))
        center = np.array([
            float(np.mean(a)) if a else center[k]
            for k, a in enumerate(assignments)
        ])

    return center


# ─── Bootstrap stability ──────────────────────────────────────────────────────
def _bootstrap_stability(
    trajectories: Dict[str, np.ndarray],
    labels: List[str],
    original_labels: np.ndarray,
    n_clusters: int,
    n_boot: int = 100,
    rng: Optional[np.random.Generator] = None,
    dtw_window: Optional[int] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(labels)
    if n < 3 or n_clusters < 2:
        return np.nan

    agreement_scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_labels = [labels[i] for i in idx]
        boot_trajs = {
            f"{boot_labels[j]}_{j}": trajectories[boot_labels[j]]
            for j in range(n)
        }
        try:
            mat, blabels = dtw_distance_matrix(boot_trajs, window=dtw_window)
            cond = squareform(mat, checks=False)
            Z_boot = linkage(cond, method="ward")
            boot_cl = fcluster(Z_boot, n_clusters, criterion="maxclust")
        except Exception:
            continue

        label_map: Dict[str, int] = {}
        for j, orig_lbl in enumerate(boot_labels):
            label_map[orig_lbl] = boot_cl[j]

        agree, total = 0, 0
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] not in label_map or labels[j] not in label_map:
                    continue
                orig_same = int(original_labels[i] == original_labels[j])
                boot_same = int(label_map[labels[i]] == label_map[labels[j]])
                agree += int(orig_same == boot_same)
                total += 1

        if total > 0:
            agreement_scores.append(agree / total)

    return round(float(np.mean(agreement_scores)), 4) if agreement_scores else np.nan


# ─── Main clustering pipeline ─────────────────────────────────────────────────
def cluster_trajectories(
    pi_df: pd.DataFrame,
    common_timepoints: Optional[List[int]] = None,
    k_range: Tuple[int, int] = (2, 8),
    dtw_window: Optional[int] = None,
    bootstrap_reps: int = 100,
    seed: int = 42,
) -> dict:
    """Full DTW-based trajectory clustering pipeline."""
    rng = np.random.default_rng(seed)

    trajectories, metadata = build_trajectory_dict(
        pi_df, common_timepoints=common_timepoints, interpolate=True
    )

    n = len(trajectories)
    if n < 3:
        return {
            "trajectories": trajectories,
            "metadata": metadata,
            "dist_matrix": np.array([]),
            "labels": list(trajectories.keys()),
            "cluster_labels": np.array([]),
            "n_clusters": 0,
            "silhouette_table": pd.DataFrame(),
            "archetypes": {},
            "cluster_table": pd.DataFrame(),
            "pca_embedding": pd.DataFrame(),
            "umap_embedding": pd.DataFrame(),
            "stability": np.nan,
            "linkage_matrix": np.array([]),
            "error": "Fewer than 3 trajectories — clustering not possible.",
        }

    dist_matrix, labels = dtw_distance_matrix(trajectories, window=dtw_window)
    cond = squareform(dist_matrix, checks=False)
    Z = linkage(cond, method="ward")

    best_k, sil_table = _optimal_k(dist_matrix, k_range=k_range)
    cluster_labels = fcluster(Z, best_k, criterion="maxclust")

    # Archetype curves per cluster
    archetypes: Dict[int, np.ndarray] = {}
    for cid in sorted(set(cluster_labels)):
        members = [
            trajectories[labels[i]]
            for i, cl in enumerate(cluster_labels) if cl == cid
        ]
        archetypes[int(cid)] = _dtw_barycenter(members)

    # Cluster table — FIX: always returns pd.DataFrame
    ct_rows = []
    for i, lbl in enumerate(labels):
        meta = metadata.get(lbl, {})
        ct_rows.append({
            "Label": lbl,
            "Genotype": meta.get("Genotype", ""),
            "Treatment": meta.get("Treatment", ""),
            "Sex": meta.get("Sex", ""),
            "Cluster": int(cluster_labels[i]),
            "n_timepoints": meta.get("n_timepoints", 0),
        })
    cluster_table = pd.DataFrame(ct_rows)  # FIX: always DataFrame, never list

    # Assign archetypes immediately so cluster_table has Archetype column
    cluster_table = interpret_clusters(cluster_table, archetypes)

    # PCA 2-D embedding
    pca_embedding = pd.DataFrame()
    try:
        max_len = max(len(v) for v in trajectories.values())
        feat_mat = np.array([
            np.pad(trajectories[lbl], (0, max_len - len(trajectories[lbl])), mode="edge")
            for lbl in labels
        ])
        if feat_mat.shape[0] >= 2 and feat_mat.shape[1] >= 2:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(feat_mat)
            n_comp = min(2, feat_scaled.shape[1], feat_scaled.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            coords = pca.fit_transform(feat_scaled)
            pca_df = pd.DataFrame({
                "Label": labels,
                "PC1": coords[:, 0] if coords.shape[1] > 0 else np.nan,
                "PC2": coords[:, 1] if coords.shape[1] > 1 else np.nan,
                "Cluster": cluster_labels,
            })
            for col in ["Genotype", "Treatment", "Sex"]:
                pca_df[col] = [metadata.get(l, {}).get(col, "") for l in labels]
            pca_embedding = pca_df
    except Exception as e:
        warnings.warn(f"trajectory_cluster PCA failed: {e}")

    # UMAP (optional)
    umap_embedding = pd.DataFrame()
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(5, n - 1),
            min_dist=0.1,
            metric="precomputed",
            random_state=seed,
        )
        umap_coords = reducer.fit_transform(dist_matrix)
        umap_embedding = pd.DataFrame({
            "Label": labels,
            "UMAP1": umap_coords[:, 0],
            "UMAP2": umap_coords[:, 1],
            "Cluster": cluster_labels,
        })
        for col in ["Genotype", "Treatment", "Sex"]:
            umap_embedding[col] = [metadata.get(l, {}).get(col, "") for l in labels]
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"trajectory_cluster UMAP failed: {e}")

    stability = _bootstrap_stability(
        trajectories, labels, cluster_labels, best_k,
        n_boot=bootstrap_reps, rng=rng, dtw_window=dtw_window,
    )

    return {
        "trajectories": trajectories,
        "metadata": metadata,
        "dist_matrix": dist_matrix,
        "labels": labels,
        "cluster_labels": cluster_labels,
        "n_clusters": best_k,
        "silhouette_table": sil_table,
        "archetypes": archetypes,
        "cluster_table": cluster_table,
        "pca_embedding": pca_embedding,
        "umap_embedding": umap_embedding,
        "stability": stability,
        "linkage_matrix": Z,
        "error": None,
    }


# ─── Cluster interpretation ───────────────────────────────────────────────────
def interpret_clusters(cluster_table: pd.DataFrame, archetypes: dict) -> pd.DataFrame:
    """
    Assign biological archetype names to each cluster.

    FIX: takes 2 positional arguments (cluster_table, archetypes)
         and ALWAYS returns pd.DataFrame (never a list).

    Archetype naming (biologically grounded):
    PI mean >= 60, flat slope        → High Performer
    PI mean >= 60, declining slope   → Early Rescue Late Decline
    PI mean >= 40, declining         → Declining Performer
    PI mean >= 40, improving         → Progressive Rescue
    PI mean < 20                     → Severely Impaired
    slope > 0.5, mean >= 30         → Motor Responder
    |slope| < 0.3                    → Plateau Phenotype
    else                             → Mixed Phenotype
    """
    if cluster_table is None or cluster_table.empty:
        return pd.DataFrame()

    if not archetypes:
        ct = cluster_table.copy()
        ct["Archetype"] = "Unclassified"
        return ct

    archetype_names: Dict[int, str] = {}
    for cid, curve in archetypes.items():
        curve = np.asarray(curve, dtype=float).ravel()
        if len(curve) < 2:
            archetype_names[cid] = "Undetermined"
            continue

        mean_pi = float(np.mean(curve))
        slope   = float(np.polyfit(np.arange(len(curve)), curve, 1)[0])

        if mean_pi >= 60 and slope < -0.5:
            name = "Early Rescue Late Decline"
        elif mean_pi >= 60 and abs(slope) < 0.5:
            name = "High Performer"
        elif mean_pi >= 40 and slope < -0.5:
            name = "Declining Performer"
        elif mean_pi >= 40 and slope > 0.3:
            name = "Progressive Rescue"
        elif mean_pi < 20:
            name = "Severely Impaired"
        elif slope > 0.5 and mean_pi >= 30:
            name = "Motor Responder"
        elif abs(slope) < 0.3:
            name = "Plateau Phenotype"
        else:
            name = "Mixed Phenotype"

        archetype_names[cid] = name

    ct = cluster_table.copy()
    ct["Archetype"] = ct["Cluster"].map(archetype_names).fillna("Undetermined")
    return ct   # FIX: always returns pd.DataFrame


# ─── Cluster summary ──────────────────────────────────────────────────────────
def cluster_summary_table(cluster_result: dict) -> pd.DataFrame:
    """Concise per-cluster summary."""
    ct = cluster_result.get("cluster_table", pd.DataFrame())
    archetypes = cluster_result.get("archetypes", {})
    stability  = cluster_result.get("stability", np.nan)

    if ct is None or (isinstance(ct, pd.DataFrame) and ct.empty):
        return pd.DataFrame()

    if "Archetype" not in ct.columns:
        ct = interpret_clusters(ct, archetypes)

    rows = []
    for cid, grp in ct.groupby("Cluster"):
        arc = archetypes.get(int(cid), np.array([]))
        rows.append({
            "Cluster": int(cid),
            "Archetype": grp["Archetype"].mode()[0] if "Archetype" in grp.columns else "Unknown",
            "n_members": len(grp),
            "Genotypes":  ", ".join(sorted(grp["Genotype"].dropna().unique().tolist())),
            "Treatments": ", ".join(sorted(grp["Treatment"].dropna().unique().tolist())),
            "Mean_archetype_PI": round(float(np.mean(arc)), 2) if len(arc) > 0 else np.nan,
            "Bootstrap_stability": stability,
        })

    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)
