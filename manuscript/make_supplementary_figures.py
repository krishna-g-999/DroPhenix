"""
make_supplementary_figures.py — per-module Supplementary Figures (S1-S6).

Each figure shows the DroPhenix analysis result for one assay on its
literature-referenced reference datasets (reference_datasets/), produced live by
the real modules. 300 dpi, Okabe-Ito colour-blind-safe palette, single Arial
font, consistent title style with the main figures. No emoji.

Run:  python manuscript/make_supplementary_figures.py -> manuscript/figures/FigureS*.png|pdf
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
REF = HERE.parent / "reference_datasets"

OI = {"blue": "#0072B2", "sky": "#56B4E9", "orange": "#E69F00", "vermilion": "#D55E00",
      "green": "#009E73", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "figure.facecolor": "white", "axes.facecolor": "white"})
SUP = dict(fontsize=13, fontweight="bold", ha="left", x=0.02)
GC = {"Oregon-WT": OI["blue"], "WT": OI["blue"]}


def _dcol(g, i):
    if g.startswith("Oregon-WT") or g == "wild-type":
        return OI["blue"]
    if g.endswith("Drug") or "+Drug" in g or g == "Drug":
        return OI["green"]
    return [OI["vermilion"], OI["orange"], OI["purple"]][i % 3]


def s1_climbing():
    from modules.data_parser import load_data
    from modules.normalizer import compute_all_metrics
    files = sorted((REF / "climbing").glob("sample*.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4)); ax = np.atleast_1d(ax)
    for k, f in enumerate(files):
        a = ax[k]; df, _, _ = load_data(pd.read_csv(f))
        pi = compute_all_metrics(df)["pi"]
        pg = pi.groupby(["Genotype", "Treatment"])["PI_mean"].mean().reset_index()
        pg["lab"] = pg["Genotype"] + "\n" + pg["Treatment"]
        a.bar(range(len(pg)), pg["PI_mean"],
              color=[_dcol(g + t, i) for i, (g, t) in enumerate(zip(pg.Genotype, pg.Treatment))],
              edgecolor=OI["black"], linewidth=0.5)
        a.axhline(0, color=OI["black"], lw=0.8)
        a.set_xticks(range(len(pg))); a.set_xticklabels(pg["lab"], fontsize=7.5)
        a.set_ylabel("Endpoint Performance Index" if k == 0 else "")
        a.set_title(f.stem.split("_", 1)[1])
    fig.suptitle("Figure S1.  Climbing (negative geotaxis): endpoint PI by group across three disease-model datasets", y=1.02, **SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); _save(fig, "FigureS1_climbing")


def s2_survival():
    from modules import survival as SV
    files = sorted((REF / "survival").glob("sample*.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4)); ax = np.atleast_1d(ax)
    for k, f in enumerate(files):
        a = ax[k]; ev = SV.to_event_table(pd.read_csv(f)); km = SV.kaplan_meier(ev)
        for i, (grp, cur) in enumerate(km.items()):
            t = np.concatenate(([0], cur["Time"])); s = np.concatenate(([1], cur["Survival"]))
            a.step(t, s, where="post", lw=2, color=_dcol(grp.split("|")[0].strip() + grp.split("|")[1].strip() if "|" in grp else grp, i),
                   label=" ".join(grp.split("|")[:2]).strip()[:22])
        a.set_ylim(0, 1.03); a.set_xlabel("Time"); a.set_ylabel("Survival S(t)" if k == 0 else "")
        a.set_title(f.stem.split("_", 1)[1][:26]); a.legend(frameon=False, fontsize=7)
    fig.suptitle("Figure S2.  Lifespan: Kaplan-Meier survival curves (sample 1 = real Freireich 1963 data)", y=1.02, **SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); _save(fig, "FigureS2_survival")


def s3_ymaze():
    from modules import ymaze as YM
    files = sorted((REF / "ymaze").glob("sample*.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4)); ax = np.atleast_1d(ax)
    for k, f in enumerate(files):
        a = ax[k]; pf = YM.per_fly_metrics(YM.load_ymaze(pd.read_csv(f)))
        g = pf.groupby("Genotype")["alternation_pct"].agg(["mean", "sem"]).reset_index()
        a.bar(range(len(g)), g["mean"], yerr=g["sem"], capsize=4,
              color=[_dcol(x, i) for i, x in enumerate(g.Genotype)], edgecolor=OI["black"], linewidth=0.5)
        a.axhline(50, color=OI["vermilion"], ls="--", lw=1.2, label="chance (50%)")
        a.set_xticks(range(len(g))); a.set_xticklabels(g["Genotype"], fontsize=8)
        a.set_ylabel("Spontaneous alternation (%)" if k == 0 else ""); a.set_ylim(0, 100)
        a.set_title(f.stem.split("_", 1)[1]); a.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure S3.  Y-maze: spontaneous alternation vs 50% chance (Lewis 2017) across cohorts", y=1.02, **SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); _save(fig, "FigureS3_ymaze")


def s4_dams():
    from modules import dams as DM
    f = REF / "dams" / "sample1_KonopkaBenzer_periods.csv"
    df = DM.load_dams(pd.read_csv(f))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4)); ax = np.atleast_1d(ax)
    for k, (geno, exp) in enumerate([("wild-type", 24), ("per-short", 19), ("per-long", 28)]):
        a = ax[k]; sub = df[df.Genotype == geno]
        fly = sub[sub.FlyID == sub.FlyID.iloc[0]].sort_values("Time_min")
        act = fly["Activity"].to_numpy(float); binm = DM._bin_minutes(fly["Time_min"].to_numpy(float))
        fac = max(1, int(round(30 / binm))); K = len(act) // fac
        rs = act[:K * fac].reshape(K, fac).sum(1) if fac > 1 else act
        cp = DM.chi_square_periodogram(rs, binm * fac, 14, 34)
        a.plot(cp["periods_h"], cp["Qp"], color=OI["blue"], lw=1.8, label="Qp")
        a.plot(cp["periods_h"], cp["threshold"], color=OI["vermilion"], ls="--", lw=1.2, label="p<0.05")
        a.axvline(cp["peak_period_h"], color=OI["green"], ls=":", lw=1.4, label=f"peak {cp['peak_period_h']:.1f} h")
        a.set_xlabel("Period (h)"); a.set_ylabel("Qp" if k == 0 else "")
        a.set_title(f"{geno} (published {exp} h)"); a.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure S4.  DAMS circadian period: chi-square periodograms recover the Konopka-Benzer (1971) periods", y=1.02, **SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); _save(fig, "FigureS4_dams")


def s5_larval():
    files = sorted((REF / "larval").glob("sample*.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4)); ax = np.atleast_1d(ax)
    for k, f in enumerate(files):
        a = ax[k]; df = pd.read_csv(f)
        g = df.groupby("Genotype")["mean_speed_mm_s"].agg(["mean", "sem"]).reset_index()
        a.bar(range(len(g)), g["mean"], yerr=g["sem"], capsize=4,
              color=[_dcol(x, i) for i, x in enumerate(g.Genotype)], edgecolor=OI["black"], linewidth=0.5)
        a.axhline(1.0, color=OI["grey"], ls=":", lw=1, label="WT ~1 mm/s")
        a.set_xticks(range(len(g))); a.set_xticklabels(g["Genotype"], fontsize=7.5)
        a.set_ylabel("Crawling speed (mm/s)" if k == 0 else ""); a.set_title(f.stem.split("_", 1)[1])
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure S5.  Larval crawling: mean speed by genotype (wild-type ~1 mm/s) across three datasets", y=1.02, **SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); _save(fig, "FigureS5_larval")


def s6_grooming():
    from modules import grooming as GR
    files = sorted((REF / "grooming").glob("sample*.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4)); ax = np.atleast_1d(ax)
    for k, f in enumerate(files):
        a = ax[k]; pf = GR.per_fly_metrics(pd.read_csv(f))
        g = pf.groupby("Genotype")["Pct_time_grooming"].agg(["mean", "sem"]).reset_index()
        a.bar(range(len(g)), g["mean"], yerr=g["sem"], capsize=4,
              color=[_dcol(x, i) for i, x in enumerate(g.Genotype)], edgecolor=OI["black"], linewidth=0.5)
        a.axhline(13, color=OI["grey"], ls=":", lw=1, label="WT ~13% (Qiao 2018)")
        a.set_xticks(range(len(g))); a.set_xticklabels(g["Genotype"], fontsize=7.5)
        a.set_ylabel("Time grooming (%)" if k == 0 else ""); a.set_title(f.stem.split("_", 1)[1])
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure S6.  Grooming: percentage of time grooming by genotype (wild-type ~13%) across cohorts", y=1.02, **SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.99]); _save(fig, "FigureS6_grooming")


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print("wrote", name)


if __name__ == "__main__":
    for fn in (s1_climbing, s2_survival, s3_ymaze, s4_dams, s5_larval, s6_grooming):
        try:
            fn()
        except Exception as e:
            print("FAILED", fn.__name__, repr(e)[:120])
