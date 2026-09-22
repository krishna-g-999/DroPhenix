# NOTE: preserved from the analysis session. Paths to the video folders and
# the background cache are still absolute; see docs/CLIMBING_BAND_STATUS_2026-09.md
# section 2.2. These are the batch/figure scripts, not yet refactored into the
# module. d15_figures.py and cross_day*.py import d15_run.py from the same dir.
"""Cross-day pattern (Day 3, 7, 15) from what exists, and the card sheet needed to
do it properly.

Days 3 and 7 have no hand counts, so their curves are the tool's -- which has a
measured floor (~5, label print) and ceiling (~12-15, crowding/plug). Day 15 has
hand counts. The figure therefore shows all three days on the SAME instrument
(the tool), with Day 15's hand-count curve alongside as the scale reference.
Days are ordinal, so one hue, light to dark.
"""
import sys, warnings
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/"
          r"D--DroPhenix-DroPhenix-main/f06626e3-0892-4286-ba94-5be2b7280e21/scratchpad")
OUT = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/_DroPhenix_analysis")
B = Path(r"C:/Users/gkp11/Videos/HD Climbing assay")
SURF, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
DAYCOL = {"Day 3": "#86b6ef", "Day 7": "#2a78d6", "Day 15": "#104281"}   # ordinal ramp, validated
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": AXIS, "grid.linewidth": 0.6, "grid.alpha": 0.5})
N_NOMINAL = 20


def sem(x): return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def load_days():
    a = pd.read_csv(OUT / "hd_all_vials.csv"); a = a[a.status == "scored"].copy()
    a["PI_tool"] = a.reported_count / N_NOMINAL
    a["vial"] = a.video + "|" + a.position
    d15 = pd.read_csv(OUT / "d15_analysis.csv"); d15["day"] = "Day 15"; d15["vial"] = d15.video + "|" + d15.position
    return a[["day", "vial", "time_sec", "PI_tool"]], d15[["day", "vial", "time_sec", "PI_tool", "PI_hand", "genotype"]]


def fig5(a, d15):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    # left: all three days on the tool, pooled over whatever conditions each day holds
    ax = axes[0]
    for day, s in list(a.groupby("day")) + [("Day 15", d15)]:
        q = s.groupby("time_sec").PI_tool; mu, se = q.mean(), q.apply(sem)
        n = s.vial.nunique()
        ax.fill_between(mu.index, mu - se, mu + se, color=DAYCOL[day], alpha=0.15, linewidth=0)
        ax.plot(mu.index, mu, color=DAYCOL[day], linewidth=2.2, marker="o", markersize=4,
                markeredgecolor=SURF, label=f"{day} — tool, {n} vials")
        ax.annotate(day, (mu.index[-1], mu.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")
    ax.set_title("Same instrument on every day: the tool, all cylinders pooled", loc="left", fontsize=11)
    ax.set_xlabel("Seconds after the start of the recording"); ax.set_ylabel("Performance index (N = 20 nominal)")
    ax.set_ylim(0, 1.0); ax.set_xlim(0, 135); ax.grid(axis="y"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    # right: Day 15, tool vs hand, to show what the tool's compression does to a real curve
    ax = axes[1]
    q = d15.groupby("time_sec")
    h, hs = q.PI_hand.mean(), q.PI_hand.apply(sem); t, ts = q.PI_tool.mean(), q.PI_tool.apply(sem)
    ax.fill_between(h.index, h - hs, h + hs, color=DAYCOL["Day 15"], alpha=0.15, linewidth=0)
    ax.plot(h.index, h, color=DAYCOL["Day 15"], linewidth=2.2, marker="o", markersize=4, markeredgecolor=SURF, label="Day 15 — hand count")
    ax.plot(t.index, t, color=INK2, linewidth=1.8, linestyle=(0, (4, 2)), marker="o", markersize=4, markerfacecolor=SURF, label="Day 15 — tool")
    ax.set_title("Day 15 only: what the tool does to a hand-counted curve", loc="left", fontsize=11)
    ax.set_xlabel("Seconds"); ax.set_ylim(0, 1.0); ax.set_xlim(0, 135); ax.grid(axis="y"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Across days — read the LEFT panel as ordering only: the tool compresses toward ~0.4 (right panel), "
                 "and Days 3/7 pool unlabelled conditions", x=0.01, ha="left", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / "fig5_cross_day_tool.png", dpi=170); plt.close(fig)


def card_sheets():
    """Every handwritten card, cropped from the background, for labelling."""
    g = pd.read_csv(OUT / "hd_geometry.csv"); g = g[g.status == "ok"]
    tmpl = []
    for day in ("Day 3", "Day 7"):
        tiles = []
        for _, r in g[g.day == day].sort_values("video").iterrows():
            bgp = SP / "hd_bg" / (r.video[:-4] + ".npy")
            if not bgp.exists():
                continue
            bg = np.load(str(bgp))
            x3 = int(r.get("P3_x1", np.nan)) if pd.notna(r.get("P3_x1")) else None
            x0 = (x3 + 20) if x3 else int(bg.shape[1] * 0.5)
            crop = bg[120:900, x0:min(bg.shape[1], x0 + 520)]
            crop = cv2.resize(crop, (260, 390)); crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(crop, (0, 0), (259, 22), (255, 255, 255), -1)
            cv2.putText(crop, r.video.replace(".MOV", ""), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            tiles.append(crop)
            sub = str(r.subfolder) if pd.notna(r.subfolder) and str(r.subfolder) != "nan" else ""
            tmpl.append({"day": day, "video": r.video, "folder_label": sub,
                         "left_genotype": "", "middle_genotype": "", "right_genotype": "",
                         "sex": "", "treatment": "", "notes": ""})
        cols = 8
        rows_ = [np.hstack(tiles[i:i + cols] + [np.full((390, 260, 3), 255, np.uint8)] * (cols - len(tiles[i:i + cols])))
                 for i in range(0, len(tiles), cols)]
        sheet = np.vstack(rows_)
        cv2.imwrite(str(OUT / f"cards_{day.replace(' ', '')}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"cards_{day.replace(' ', '')}.jpg: {len(tiles)} cards")
    pd.DataFrame(tmpl).to_csv(Path(r"D:\DroPhenix\DroPhenix-main\data\validation\condition_map_day3_day7.csv"), index=False)
    print("template: data/validation/condition_map_day3_day7.csv  (%d videos)" % len(tmpl))


if __name__ == "__main__":
    a, d15 = load_days()
    fig5(a, d15)
    print("fig5_cross_day_tool.png written")
    card_sheets()
