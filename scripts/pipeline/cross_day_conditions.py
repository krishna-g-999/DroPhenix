# NOTE: preserved from the analysis session. Paths to the video folders and
# the background cache are still absolute; see docs/CLIMBING_BAND_STATUS_2026-09.md
# section 2.2. These are the batch/figure scripts, not yet refactored into the
# module. d15_figures.py and cross_day*.py import d15_run.py from the same dir.
"""Day 3 / Day 7 / Day 15, per condition, on one instrument.

Rows 1-3: the tool on each day. Row 4: Day 15 hand counts -- the calibration
row that shows how much the tool compresses a real curve on this rig.

Experimental unit: on Days 3/7 the three cylinders in a video are replicates of
one condition and share a tap, so each VIDEO is reduced to its mean first and
SEM runs across videos. On Day 15 each cylinder is its own genotype, so the
unit is the cylinder. n is stated per line. Days 3/7 use only card readings of
medium or high confidence; low/unreadable videos are left out, not guessed.
"""
import sys, warnings
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/_DroPhenix_analysis")
MAP = Path(r"D:\DroPhenix\DroPhenix-main\data\validation\condition_map_day3_day7.csv")
GENO = ["Wildtype", "Htt", "CyO"]
COL = {"Wildtype": "#2a78d6", "Htt": "#eb6834", "CyO": "#1baf7a"}
MRK = {"Wildtype": "o", "Htt": "s", "CyO": "^"}
LS = {"Untreated": "-", "Withaferin": (0, (4, 2))}
SURF, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": AXIS, "grid.linewidth": 0.6, "grid.alpha": 0.5})
N_NOMINAL = 20


def sem(x): return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def load():
    cm = pd.read_csv(MAP); cm = cm[cm.confidence.isin(["high", "medium"])]
    a = pd.read_csv(OUT / "hd_all_vials.csv"); a = a[a.status == "scored"]
    a = a.merge(cm[["video", "genotype", "sex", "treatment"]], on="video", how="inner")
    a["PI"] = a.reported_count / N_NOMINAL
    # unit = video: reduce the three replicate cylinders first
    a = a.groupby(["day", "video", "genotype", "sex", "treatment", "time_sec"], as_index=False).PI.mean()
    a["unit"] = a.video
    d = pd.read_csv(OUT / "d15_analysis.csv")
    d["genotype"] = d.genotype.replace({"CYO": "CyO"})
    d["unit"] = d.video + "|" + d.position
    d15_tool = d.assign(day="Day 15 (tool)", PI=d.PI_tool)[["day", "unit", "genotype", "sex", "treatment", "time_sec", "PI"]]
    d15_hand = d.assign(day="Day 15 (hand count)", PI=d.PI_hand)[["day", "unit", "genotype", "sex", "treatment", "time_sec", "PI"]]
    a = a.assign(day=a.day + " (tool)")[["day", "unit", "genotype", "sex", "treatment", "time_sec", "PI"]]
    return pd.concat([a, d15_tool, d15_hand], ignore_index=True)


def main():
    x = load()
    rows = ["Day 3 (tool)", "Day 7 (tool)", "Day 15 (tool)", "Day 15 (hand count)"]
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13), sharex=True, sharey=True)
    for i, day in enumerate(rows):
        for j, sx in enumerate(("Male", "Female")):
            ax = axes[i, j]; s = x[(x.day == day) & (x.sex == sx)]
            for g in GENO:
                for tr in ("Untreated", "Withaferin"):
                    q = s[(s.genotype == g) & (s.treatment == tr)]
                    if q.empty: continue
                    grp = q.groupby("time_sec").PI; mu, se = grp.mean(), grp.apply(sem); n = q.unit.nunique()
                    ax.fill_between(mu.index, mu - se, mu + se, color=COL[g], alpha=0.10, linewidth=0)
                    ax.plot(mu.index, mu, color=COL[g], linestyle=LS[tr], linewidth=1.9, marker=MRK[g],
                            markersize=3.5, markeredgecolor=SURF, markerfacecolor=(COL[g] if tr == "Untreated" else SURF),
                            label=f"{g} {'+With' if tr == 'Withaferin' else ''} (n={n})")
            ax.set_title(f"{day} · {sx}", loc="left", fontsize=10.5, color=INK)
            ax.grid(axis="y"); ax.set_ylim(0, 1.0); ax.set_xlim(0, 125)
            if not s.empty: ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
            else: ax.text(0.5, 0.5, "no labelled videos", transform=ax.transAxes, ha="center", color=MUTED)
    for ax in axes[-1]: ax.set_xlabel("Seconds after the start of the recording")
    fig.supylabel("Performance index  (N = 20 nominal; Day 15 uses the video-pinned N where higher)", fontsize=9.5, color=INK2)
    fig.suptitle("Negative geotaxis across days, per condition — solid = untreated, dashed/hollow = withaferin",
                 x=0.01, ha="left", fontsize=12.5, y=0.995)
    fig.text(0.01, 0.975, "Rows 1–3: the tool on each day. It compresses toward ~0.4 (floor from label print, ceiling from crowding).",
             fontsize=8.5, color=INK2)
    fig.text(0.01, 0.962, "Row 4: the hand count, the true scale. Read Days 3/7 for ordering and direction only, not level.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0.02, 0, 1, 0.955)); fig.savefig(OUT / "fig6_cross_day_conditions.png", dpi=160); plt.close(fig)

    # the numbers behind it: 10 s and 60 s per condition per day
    tab = (x[x.time_sec.isin([10, 60])].groupby(["day", "sex", "treatment", "genotype", "time_sec"])
             .agg(PI=("PI", "mean"), n=("unit", "nunique")).reset_index())
    tab.to_csv(OUT / "cross_day_table.csv", index=False)
    print(tab.pivot_table(index=["day", "sex", "treatment"], columns=["genotype", "time_sec"], values="PI").round(2).to_string())


if __name__ == "__main__":
    main()
