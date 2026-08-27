"""Turn the Climbing assay_SP scores into figures and a summary, honestly labelled.

WHAT THE FIGURES MAY AND MAY NOT SAY
------------------------------------
This set has no hand counts, so nothing here is an accuracy claim. Two things ARE
measurable without any ground truth and both are plotted:

  * the climbing curves themselves, and how they differ between days;
  * STABILITY -- the number of flies in a sealed cylinder cannot change over two
    minutes, so every wobble in the detected total is detector error. That panel
    needs no ground truth to interpret and is the quality control for everything
    else on the page.

THE DENOMINATOR IS KNOWN: 10 flies per vial, as loaded by the laboratory. An earlier
version of this analysis estimated it from the detector instead, which was wrong twice
over -- it made the denominator depend on the detector's own error, and it absorbed the
drawn band's false positives into "flies loaded". Told the true value the same
detections read completely differently: recall looked like 32% against an assumed 20
and is 87% against the true 10.

THE READOUT IS THE ONE THE FIELD USES: how many of the cohort are above the line at
each 10 s, as a percentage of the flies loaded. No plateau at 0 or 100 is expected --
what fraction is up at two minutes is a property of the flies, not of the protocol.
Flies the detector cannot account for at an instant could be on either side of the
line, so `pct_above_lo/hi` bracket the proportion. That band is drawn, not hidden.

THE EXPERIMENTAL UNIT
---------------------
Three cylinders share one recording, so their counts are not independent -- the same
tap, the same operator, the same instant. Averaging over vials as though they were
independent would understate the uncertainty. Each recording is therefore reduced to
its own mean first, and the intervals are computed ACROSS RECORDINGS. The
intraclass correlation between vials of a recording is reported so the size of that
correction is visible rather than assumed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Reads what `score_climbing_sp.py` wrote, so it MUST resolve to the same place.
#: Both scripts used to carry the same hardcoded Claude-session scratchpad literal,
#: which is temporary by construction -- two copies of a path that was going to stop
#: existing, with nothing tying them together but the fact that someone had pasted
#: the same string twice. They now read the same environment variable and fall back
#: to the same repository directory, so they cannot drift apart.
SP = Path(os.environ.get(
    "DROPHENIX_SP_OUT",
    Path(__file__).resolve().parents[1] / "results" / "climbing_video"))

# Day is ORDINAL (3 -> 7 -> 14), so a single-hue ramp, not categorical hues.
# Validated: light surface, ordinal mode -- monotone lightness, adjacent dL >= 0.06,
# light end 2.06:1 against the surface, hue spread 3 degrees.
DAY_COLOR = {"Day 3": "#86b6ef", "Day7": "#2a78d6", "Day 14": "#104281"}
DAY_LABEL = {"Day 3": "Day 3", "Day7": "Day 7", "Day 14": "Day 14"}
ORDER = ["Day 3", "Day7", "Day 14"]
SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": AXIS, "grid.linewidth": 0.6, "grid.alpha": 0.5,
})


def icc(df, value):
    """Intraclass correlation of vials within a recording, at a fixed time."""
    g = df.groupby("video")[value]
    k = g.size()
    if len(k) < 2 or k.min() < 2:
        return np.nan
    grand = df[value].mean()
    ms_b = (k * (g.mean() - grand) ** 2).sum() / (len(k) - 1)
    ms_w = sum(((v - v.mean()) ** 2).sum() for _, v in g) / max(
        (k.sum() - len(k)), 1)
    k0 = k.mean()
    denom = ms_b + (k0 - 1) * ms_w
    return float((ms_b - ms_w) / denom) if denom > 0 else np.nan


def curve(df, value):
    """Mean and 95% CI across RECORDINGS (each recording reduced to its own mean)."""
    # observed=True matters: `day` is categorical, and with the pandas default this
    # produces a row for every day x video combination, including the ones that do
    # not exist. Video names are unique to a day, so each recording then appeared
    # under all three days and the group size came out as 68 -- the total across the
    # set -- for every day. The means survived it (the empty cells are NaN) but the
    # counts did not, and the confidence intervals are computed from those counts.
    per_vid = (df.groupby(["day", "video", "time_s"], observed=True)[value]
               .mean().reset_index())
    out = []
    for (day, t), g in per_vid.groupby(["day", "time_s"], observed=True):
        n = len(g)
        m, sd = g[value].mean(), g[value].std(ddof=1) if n > 1 else 0.0
        # t, NOT 1.96. The standard deviation is ESTIMATED from the same handful of
        # recordings the mean comes from, so the normal quantile is too small: with
        # this set's counts (Day 3 n=23, Day 7 n=21, Day 14 n=24 recordings) the
        # 97.5% t values are 2.074, 2.086 and 2.069, so the ribbons drawn with 1.96
        # were 5.5%, 6.0% and 5.3% too NARROW. Small, but in the anti-conservative
        # direction on a figure whose caption advertises correct uncertainty
        # handling, and the error grows as n falls -- which is exactly what happens
        # when recordings are refused.
        crit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        half = crit * sd / np.sqrt(n) if n > 1 else 0.0
        out.append({"day": day, "time_s": t, "mean": m, "lo": m - half,
                    "hi": m + half, "n_videos": n})
    return pd.DataFrame(out)


def main() -> int:
    d = pd.read_csv(SP / "sp_results.csv")
    sc = d[d.status == "scored"].copy()
    if sc.empty:
        print("nothing scored"); return 1
    sc["day"] = pd.Categorical(sc["day"], ORDER, ordered=True)

    print("=" * 78)
    print("CLIMBING ASSAY_SP -- what was scored")
    print("=" * 78)
    per_vial = sc.drop_duplicates("vial_id")
    summ = sc.groupby("day", observed=True).agg(
        recordings=("video", "nunique"), vials=("vial_id", "nunique"),
        cells=("time_s", "size")).reset_index()
    # `n_est_loaded` is a legacy NAME for a detector-coverage statistic: the median
    # number of objects found in the tube across the clip. It is NOT the denominator
    # -- that is the laboratory's loading of 10 -- and it must not be displayed as
    # though it were. Under the old name this column read as "flies loaded", which
    # invites the reader to compare it with the counts beside it; do that and 17 of
    # 204 vials appear to have MORE flies above the line than were in the tube,
    # because the median runs over frames including the clustered period just after
    # the tap and is biased low. Renamed here at the point of display.
    summ["median_detected_n"] = [per_vial[per_vial.day == dd].n_est_loaded.median()
                                 for dd in summ.day]
    summ["median_total_CV_pct"] = [per_vial[per_vial.day == dd].total_cv_pct.median()
                                   for dd in summ.day]
    print(summ.to_string(index=False))
    refused = d[d.status.astype(str).str.startswith("refused")]
    print(f"\nrefused: {len(refused.drop_duplicates(['day','video']))} recording(s)")
    if len(refused):
        print(refused.drop_duplicates(["day", "video"])[["day", "video", "status"]]
              .to_string(index=False))

    at60 = sc[sc.time_s == 60]
    print("\nclustering of vials within a recording (ICC at 60 s):")
    for dd in ORDER:
        sub = at60[at60.day == dd]
        if len(sub) > 4:
            r = icc(sub, "n_above")
            m = sub.groupby("video").size().mean()
            print(f"  {DAY_LABEL[dd]:7s} ICC {r:+.3f}   design effect "
                  f"{1 + (m - 1) * (r if np.isfinite(r) else 0):.2f} "
                  f"({sub.video.nunique()} recordings)")

    # ---------------- figure 1: the climbing curves ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, val, ylab, ttl in (
            (axes[0], "n_above", "Flies above the line (of 10)",
             "Climbing curve — flies above the line at each 10 s"),
            (axes[1], "pct_above", "% of the 10 flies loaded",
             "The assay readout — % of the cohort above the line")):
        c = curve(sc, val)
        for dd in ORDER:
            g = c[c.day == dd].sort_values("time_s")
            if g.empty:
                continue
            ax.fill_between(g.time_s, g.lo, g.hi, color=DAY_COLOR[dd], alpha=0.15,
                            linewidth=0)
            ax.plot(g.time_s, g["mean"], color=DAY_COLOR[dd], linewidth=2,
                    label=f"{DAY_LABEL[dd]}  (n={int(g.n_videos.iloc[0])} recordings)")
            ax.annotate(DAY_LABEL[dd], (g.time_s.iloc[-1], g["mean"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=INK2, fontsize=9, va="center")
        ax.set_xlabel("Seconds after the tap"); ax.set_ylabel(ylab)
        ax.set_title(ttl, color=INK, fontsize=11, loc="left")
        ax.grid(axis="y"); ax.set_xlim(0, 132); ax.legend(frameon=False, fontsize=9)
    axes[1].set_ylim(0, 100)
    for dd in ORDER:
        hb = curve(sc, "pct_above_hi")
        gh = hb[hb.day == dd].sort_values("time_s")
        if not gh.empty:
            axes[1].plot(gh.time_s, gh["mean"], color=DAY_COLOR[dd], linewidth=1,
                         linestyle=(0, (2, 2)), alpha=0.9)
    axes[1].plot([], [], color=INK2, linewidth=1, linestyle=(0, (2, 2)),
                 label="upper bracket: unaccounted flies counted as above")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Climbing assay_SP — ribbons are 95% CI across recordings, "
                 "not across vials (three vials share one tap)",
                 color=INK2, fontsize=9.5, y=1.02, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(SP / "sp_fig1_curves.png", dpi=170,
                                    bbox_inches="tight")
    print(f"\nwrote {SP/'sp_fig1_curves.png'}")

    # ---------------- figure 2: quality control ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax = axes[0]
    for dd in ORDER:
        sub = sc[sc.day == dd]
        for vid, g in sub.groupby("vial_id"):
            g = g.sort_values("time_s")
            ax.plot(g.time_s, g.n_detected_total, color=DAY_COLOR[dd],
                    alpha=0.35, linewidth=1)
    for dd in ORDER:
        ax.plot([], [], color=DAY_COLOR[dd], linewidth=2, label=DAY_LABEL[dd])
    ax.set_xlabel("Seconds after the tap")
    ax.axhline(10, color=INK, linewidth=2, linestyle=(0, (5, 3)))
    ax.set_ylim(0, 22)
    ax.set_ylabel("Total flies detected in the tube")
    ax.set_title("Detector QC against the known cohort of 10\n"
                 "(dashed line = truth; the gap is detector error)", color=INK,
                 fontsize=11, loc="left")
    ax.grid(axis="y"); ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    cv = per_vial.groupby("day", observed=True)["total_cv_pct"].apply(list)
    ax.boxplot([cv.get(dd, []) for dd in ORDER], tick_labels=[DAY_LABEL[d] for d in ORDER],
               patch_artist=True,
               boxprops=dict(facecolor="#e8eef7", edgecolor=AXIS),
               medianprops=dict(color=INK, linewidth=2),
               whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS),
               flierprops=dict(marker="o", markersize=4, markerfacecolor=MUTED,
                               markeredgecolor="none", alpha=0.6))
    ax.set_ylabel("Coefficient of variation of the total (%)")
    ax.set_title("Detector error per vial\n(lower is better; 0 would be perfect)",
                 color=INK, fontsize=11, loc="left")
    ax.grid(axis="y")

    ax = axes[2]
    src = per_vial.groupby(
        [per_vial.day, per_vial.geometry.astype(str).str.split(" (", regex=False).str[0]],
        observed=True).size().unstack(fill_value=0)
    bottom = np.zeros(len(src))
    shades = {"own fit": "#2a78d6", "day consensus": "#86b6ef", "glass": "#104281"}
    for col in src.columns:
        ax.bar([DAY_LABEL[i] for i in src.index], src[col], bottom=bottom,
               label=col, color=shades.get(col, MUTED), edgecolor=SURFACE,
               linewidth=2)
        bottom += src[col].values
    ax.set_ylabel("Vials")
    ax.set_title("How the geometry was found\n(no vial was scored on a poor fit)",
                 color=INK, fontsize=11, loc="left")
    ax.grid(axis="y"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(SP / "sp_fig2_quality.png", dpi=170,
                                    bbox_inches="tight")
    print(f"wrote {SP/'sp_fig2_quality.png'}")

    # ---------------- figure 3: endpoint and the two rules ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    rng = np.random.default_rng(3)
    for i, dd in enumerate(ORDER):
        sub = at60[at60.day == dd]
        if sub.empty:
            continue
        y = sub.n_above.values
        x = i + rng.uniform(-0.16, 0.16, len(y))
        ax.scatter(x, y, s=26, color=DAY_COLOR[dd], alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8)
        ax.plot([i - 0.28, i + 0.28], [np.median(y)] * 2, color=INK, linewidth=2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([DAY_LABEL[d] for d in ORDER])
    ax.set_ylabel("Flies above the line at 60 s (of 10)")
    ax.set_title("Every vial at 60 s\n(bar is the median)", color=INK,
                 fontsize=11, loc="left")
    ax.grid(axis="y")

    ax = axes[1]
    for dd in ORDER:
        a = curve(sc, "n_above"); b = curve(sc, "n_above_max_reached")
        ga, gb = a[a.day == dd].sort_values("time_s"), b[b.day == dd].sort_values("time_s")
        if ga.empty:
            continue
        ax.plot(ga.time_s, ga["mean"], color=DAY_COLOR[dd], linewidth=2)
        ax.plot(gb.time_s, gb["mean"], color=DAY_COLOR[dd], linewidth=2,
                linestyle=(0, (4, 2)))
    ax.plot([], [], color=INK2, linewidth=2, label="standing above (protocol)")
    ax.plot([], [], color=INK2, linewidth=2, linestyle=(0, (4, 2)),
            label="most ever above (ratchet)")
    ax.set_xlabel("Seconds after the tap"); ax.set_ylabel("Flies")
    ax.set_title("The two scoring rules, same detections\n"
                 "(the gap is flies that climbed and came back down)",
                 color=INK, fontsize=11, loc="left")
    ax.grid(axis="y"); ax.legend(frameon=False, fontsize=9); ax.set_xlim(0, 132)
    fig.tight_layout(); fig.savefig(SP / "sp_fig3_endpoints.png", dpi=170,
                                    bbox_inches="tight")
    print(f"wrote {SP/'sp_fig3_endpoints.png'}")

    # ---------------- the tables ----------------
    tab = curve(sc, "n_above").pivot_table(index="time_s", columns="day",
                                           values="mean", observed=True)
    tab.columns = [DAY_LABEL[c] for c in tab.columns]
    print("\nmean flies above the line, by day:")
    print(tab.round(1).to_string())
    # TIMEPOINT COLUMNS ARE SELECTED BY LABEL, NOT BY POSITION.
    # This block used to read `s.iloc[0]`, `s.iloc[min(5, len(s) - 1)]` and
    # `s.iloc[-1]`, which is correct only when all twelve timepoints are present.
    # `score_climbing_sp` appends nothing for a timepoint that falls beyond the end
    # of the recording, so vials are routinely short: measured on the shipped
    # sp_results.csv, 177 of 204 vials carry 12 timepoints, 21 carry 11, three
    # carry 10 and three carry 6.
    #
    # The gaps are all at the TAIL, so `above_10s` and `above_60s` happened to
    # survive positional indexing -- 0 of 204 were wrong. `above_120s` did not:
    # it held the last AVAILABLE timepoint in 27 of 204 vials -- 110 s in 21 of
    # them, 100 s in Day-3 C0092 P1/P2/P3, and 60 s in Day-7 C0148 P1/P2/P3. A
    # column read as the two-minute endpoint was carrying a one-minute value, and
    # nothing in the file said so. `at60 = sc[sc.time_s == 60]` earlier in this
    # same script already selects by label; this now does the same.
    #
    # An absent timepoint is NaN. It is not filled from a neighbour: a vial whose
    # recording stopped at 60 s has no two-minute reading, and inventing one is
    # the defect this replaces. `n_timepoints` is carried so the truncation is
    # visible in the file rather than inferable only by counting rows elsewhere.
    rows_sum = []
    for (day_, vid_, pos_), g in sc.groupby(["day", "video", "position"],
                                            observed=True):
        by_t = dict(zip(g["time_s"], g["n_above"]))
        rows_sum.append({
            "day": day_, "video": vid_, "position": pos_,
            # same rename as above: what the detector typically found, not a loading
            "median_detected_total": g["n_est_loaded"].iloc[0],
            "total_cv_pct": g["total_cv_pct"].iloc[0],
            "above_10s": by_t.get(10.0, np.nan),
            "above_60s": by_t.get(60.0, np.nan),
            "above_120s": by_t.get(120.0, np.nan),
            "n_timepoints": int(len(g))})
    out = pd.DataFrame(rows_sum)
    _short = out[out.above_120s.isna()]
    if len(_short):
        last = (sc.groupby(["day", "video", "position"], observed=True)["time_s"]
                  .max().rename("last_timepoint_s").reset_index())
        why = _short.merge(last, on=["day", "video", "position"], how="left")
        print(f"{len(_short)} vial(s) have no 120 s reading -- the recording "
              "ended first. above_120s is NaN for them, NOT back-filled from "
              "the last available timepoint:")
        print(why.groupby("last_timepoint_s", observed=True).size()
                 .rename("vials").to_frame().to_string())
    out.to_csv(SP / "sp_per_vial_summary.csv", index=False)
    print(f"\nwrote {SP/'sp_per_vial_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
