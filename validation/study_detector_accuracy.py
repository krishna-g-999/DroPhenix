"""
Accuracy of DroPhenix's two climbing detectors against hand counts, on the
laboratory's own paraquat footage.

705 paired cells per backend: each row is one (video, tap, vial, timepoint)
with the tool's count and a human's count of the same cell. Both backends were
run on identical frames, so the comparison is like for like.

The question that matters most is NOT the pooled error. It is whether accuracy
DEGRADES WITH IMPAIRMENT -- register N3 measured detector recall falling from
0.88 in healthy flies to 0.35 in a severe phenotype, and if that reproduces
here then every treatment effect this tool reports is confounded with its own
detection failure, because the sickest animals are the worst measured.

So everything is stratified by paraquat dose, never pooled.
"""
import os
import sys

import numpy as np
import pandas as pd

PARA = r"C:/Climbing assay/Paraquat climbing assay 010626_new set"


def load(fn):
    df = pd.read_csv(os.path.join(PARA, fn))
    df["err"] = df["video_count"] - df["hand"]
    return df


thr = load("paraquat_scored_net.csv")
blob = load("paraquat_scored_net_blob.csv")

print("=" * 78)
print("PAIRED CELLS AND WHAT HAPPENED TO THEM")
print("=" * 78)
for name, d in (("threshold", thr), ("blob", blob)):
    tot = len(d)
    scored = d["status"].eq("scored").sum()
    print("%-10s %4d cells | status: %s" % (name, tot,
          dict(d["status"].value_counts())))
    if d["refusal"].notna().any():
        print("           refusal reasons: %s"
              % dict(d.loc[d["refusal"].notna(), "refusal"].value_counts().head(4)))

# only cells where BOTH a tool count and a hand count exist can be compared
def usable(d):
    return d[(d["status"] == "scored") & d["video_count"].notna()
             & d["hand"].notna()].copy()


U = {"threshold": usable(thr), "blob": usable(blob)}

print()
print("=" * 78)
print("OVERALL ACCURACY vs HAND COUNTS")
print("=" * 78)
print("%-10s %5s %8s %8s %8s %8s %8s"
      % ("backend", "n", "MAE", "bias", "RMSE", "r", "within±2"))
for name, d in U.items():
    e = d["err"].to_numpy(float)
    r = np.corrcoef(d["video_count"], d["hand"])[0, 1]
    print("%-10s %5d %8.3f %+8.3f %8.3f %8.4f %7.1f%%"
          % (name, len(d), np.mean(np.abs(e)), np.mean(e),
             np.sqrt(np.mean(e ** 2)), r, 100 * np.mean(np.abs(e) <= 2)))

print()
print("=" * 78)
print("STRATIFIED BY PARAQUAT DOSE -- does accuracy degrade with impairment?")
print("=" * 78)
for name, d in U.items():
    print("\n%s backend:" % name)
    print("  %-12s %-6s %5s %8s %8s %8s %8s"
          % ("Genotype", "Dose", "n", "MAE", "bias", "mean hand", "mean tool"))
    for (g, t), sub in d.groupby(["Genotype", "Treatment"], sort=False):
        e = sub["err"].to_numpy(float)
        print("  %-12s %-6s %5d %8.3f %+8.3f %8.2f %8.2f"
              % (g, t, len(sub), np.mean(np.abs(e)), np.mean(e),
                 sub["hand"].mean(), sub["video_count"].mean()))

print()
print("=" * 78)
print("THE CONFOUND TEST: is the error correlated with the true count?")
print("=" * 78)
print("A detector that misses flies when few have climbed produces an error")
print("that tracks the hand count, which biases every treatment contrast.")
print("%-10s %12s %12s %s" % ("backend", "r(err,hand)", "slope", "reading"))
for name, d in U.items():
    h = d["hand"].to_numpy(float)
    e = d["err"].to_numpy(float)
    r = np.corrcoef(e, h)[0, 1]
    slope = np.polyfit(h, e, 1)[0]
    verdict = ("error grows with the count" if abs(r) > 0.3 else
               "little dependence on the count")
    print("%-10s %12.4f %12.4f %s" % (name, r, slope, verdict))

print()
print("=" * 78)
print("REPRODUCIBILITY: agreement between the two backends on identical frames")
print("=" * 78)
key = ["folder", "video", "tap", "replicate", "vial", "time_s"]
m = U["threshold"].merge(U["blob"], on=key, suffixes=("_thr", "_blob"))
if len(m):
    d = (m["video_count_thr"] - m["video_count_blob"]).to_numpy(float)
    print("paired cells scored by both : %d" % len(m))
    print("mean |threshold - blob|     : %.3f flies" % np.mean(np.abs(d)))
    print("identical counts            : %.1f%%" % (100 * np.mean(d == 0)))
    print("r between the two backends  : %.4f"
          % np.corrcoef(m["video_count_thr"], m["video_count_blob"])[0, 1])
    print()
    print("and against the same hand counts:")
    for tag in ("thr", "blob"):
        e = (m["video_count_" + tag] - m["hand_" + tag]).to_numpy(float)
        print("  %-10s MAE %.3f  bias %+.3f" % (tag, np.mean(np.abs(e)),
                                                np.mean(e)))
