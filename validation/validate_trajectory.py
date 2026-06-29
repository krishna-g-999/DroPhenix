"""
validate_trajectory.py
Validation of modules/trajectory_cluster.py (DTW-based climbing-trajectory
clustering used for drug/genotype phenotype grouping).

Ground-truth test: build groups whose PI-vs-time trajectories belong to TWO
clearly distinct shape families — (i) "stable/high" (healthy-like, flat near
PI~80) and (ii) "steep decline" (disease-like, decaying toward low PI). The
clustering must (a) choose 2 clusters and (b) separate the two families with
high purity. Run across multiple seeds for robustness.

Run:  python validation/validate_trajectory.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.trajectory_cluster import cluster_trajectories  # noqa: E402

TIMES = [0, 5, 10, 15, 20, 25, 30]


def _make_traj_df(seed):
    rng = np.random.default_rng(seed)
    rows = []
    # 5 stable-high groups, 5 steep-decline groups
    for i in range(5):
        for t in TIMES:
            pi = 80 + rng.normal(0, 2)                       # flat high
            rows.append({"Genotype": f"stable{i}", "Sex": "Male",
                         "Treatment": "x", "Time": t, "PI_mean": pi})
    for i in range(5):
        for t in TIMES:
            pi = 80 * np.exp(-0.10 * t) + rng.normal(0, 2)    # steep decline
            rows.append({"Genotype": f"declin{i}", "Sex": "Male",
                         "Treatment": "x", "Time": t, "PI_mean": pi})
    return pd.DataFrame(rows)


def _purity(labels_map):
    """labels_map: dict group_label -> cluster_id. Compute cluster purity vs the
    true family encoded in the group label prefix ('stable' / 'declin')."""
    # true family encoded in the group label ('stable...' vs 'declin...')
    df = pd.DataFrame([("stable" if "stable" in k else "declin", v)
                       for k, v in labels_map.items()], columns=["truth", "cluster"])
    # purity = sum over clusters of max class count / N
    tot = 0
    for _, sub in df.groupby("cluster"):
        tot += sub["truth"].value_counts().iloc[0]
    return tot / len(df)


def main() -> int:
    results = []
    purities, kvals = [], []
    for seed in range(4):
        res = cluster_trajectories(_make_traj_df(seed), k_range=(2, 6), seed=seed)
        labels = res["labels"]
        clabels = res["cluster_labels"]
        if len(clabels) == 0:
            results.append((f"seed{seed} clustering ran", "ok", "empty", False))
            continue
        lm = {labels[i]: int(clabels[i]) for i in range(len(labels))}
        purities.append(_purity(lm))
        kvals.append(res["n_clusters"])

    mean_purity = float(np.mean(purities)) if purities else 0.0
    ok_k = all(k == 2 for k in kvals) if kvals else False
    results.append(("Clusters chosen = 2 (all seeds)", "2", str(kvals), ok_k))
    results.append(("Mean cluster purity (>=0.9)", ">=0.9",
                    round(mean_purity, 3), mean_purity >= 0.9))

    print("=" * 70)
    print("DroPhenix trajectory-clustering validation (DTW, known shape families)")
    print("=" * 70)
    print(f"{'Check':<38}{'Expected':>12}{'Got':>12}{'Pass':>6}")
    print("-" * 70)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<38}{str(exp):>12}{str(got):>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 70)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
