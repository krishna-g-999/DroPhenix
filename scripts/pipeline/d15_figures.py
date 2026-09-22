# NOTE: preserved from the analysis session. Paths to the video folders and
# the background cache are still absolute; see docs/CLIMBING_BAND_STATUS_2026-09.md
# section 2.2. These are the batch/figure scripts, not yet refactored into the
# module. d15_figures.py and cross_day*.py import d15_run.py from the same dir.
"""15th-Day figures: the biology from the operator's hand counts, and the tool's
agreement with those counts, presented the way a methods paper would.

Categorical order is fixed (Wildtype, Htt, CYO) and never cycled; each series
also carries its own marker so identity never rests on hue alone. One y-axis per
panel. Individual vials are shown wherever n is small enough to show them.
"""
import sys, warnings, importlib.util
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/"
          r"D--DroPhenix-DroPhenix-main/f06626e3-0892-4286-ba94-5be2b7280e21/scratchpad")
OUT = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/_DroPhenix_analysis")
_r = importlib.util.spec_from_file_location("run", str(Path(__file__).parent / "d15_run.py"))
run = importlib.util.module_from_spec(_r); _r.loader.exec_module(run)

GENO = ["Wildtype", "Htt", "CYO"]
COL = {"Wildtype": "#2a78d6", "Htt": "#eb6834", "CYO": "#1baf7a"}      # validated
MRK = {"Wildtype": "o", "Htt": "s", "CYO": "^"}
SURF, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": AXIS, "grid.linewidth": 0.6, "grid.alpha": 0.5})
HUMAN_FLOOR = 3.0     # measured: two operator counts of the same instant, MAE 2.92 (n=12)


def load():
    hand = run.parse_hand(run.D / "Manual counting.xlsx")
    present = {f.name for f in run.D.glob("**/*.MP4")}
    hand = hand[hand.video.isin(present)].copy()
    cond = {f.name: f.parent.name for f in run.D.glob("**/*.MP4")}
    hand["condition"] = hand.video.map(cond)
    hand["sex"] = hand.condition.str.split().str[0]
    hand["treatment"] = np.where(hand.condition.str.contains("withaferin"), "Withaferin", "Untreated")
    tool = pd.read_csv(OUT / "d15_tool_fullzone.csv"); tool = tool[tool.status == "scored"]
    m = hand.merge(tool.drop(columns=["genotype", "condition"]), on=["video", "position", "time_sec"], how="left")
    # N: operator prior of 20, overridden upward only where the video proves more
    floor = pd.concat([m.groupby(["video", "genotype"]).hand_count.max(),
                       m.groupby(["video", "genotype"]).accounted_for.apply(lambda s: np.percentile(s.dropna(), 75) if s.notna().any() else 0)],
                      axis=1).max(axis=1).rename("N_floor").reset_index()
    m = m.merge(floor, on=["video", "genotype"])
    m["N"] = np.maximum(20, np.round(m.N_floor))
    m["PI_hand"] = m.hand_count / m.N
    m["PI_tool"] = m.reported_count / m.N
    m["err"] = m.reported_count - m.hand_count
    return m


def sem(x): return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _end_labels(ax, ends, min_gap=0.055):
    """Direct labels at the right edge, nudged apart when two series end close."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (x, _y, g), y in zip(ends, ys):
        ax.annotate(g, (x, y), xytext=(5, 0), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")


def fig1_curves(m):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=True, sharey=True)
    for ax, (sx, tr) in zip(axes.flat, [("Female", "Untreated"), ("Female", "Withaferin"),
                                        ("Male", "Untreated"), ("Male", "Withaferin")]):
        s = m[(m.sex == sx) & (m.treatment == tr)]
        ends = []
        for g in GENO:
            q = s[s.genotype == g].groupby("time_sec").PI_hand
            mu, se = q.mean(), q.apply(sem)
            ax.fill_between(mu.index, mu - se, mu + se, color=COL[g], alpha=0.15, linewidth=0)
            ax.plot(mu.index, mu, color=COL[g], marker=MRK[g], markersize=5, linewidth=2,
                    markeredgecolor=SURF, markeredgewidth=1)
            ends.append((mu.index[-1], mu.iloc[-1], g))
        _end_labels(ax, ends)
        n = s.groupby("genotype").video.nunique().min()
        ax.set_title(f"{sx} · {tr}   (n = {n} vials per genotype)", loc="left", fontsize=11, color=INK)
        ax.grid(axis="y"); ax.set_ylim(0, 1.05); ax.set_xlim(0, 140)
    for ax in axes[1]: ax.set_xlabel("Seconds after the start of the recording")
    fig.supylabel("Performance index  (flies above the 40 ml band / N)", fontsize=10, color=INK2)
    handles = [plt.Line2D([], [], color=COL[g], marker=MRK[g], linewidth=2, markersize=5,
                          markeredgecolor=SURF, label=g) for g in GENO]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.99, 0.965))
    fig.suptitle("15th-day negative geotaxis — operator hand counts, mean ± SEM across vials",
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
    fig.text(0.01, 0.955, "N = 20 per cylinder unless the video showed more (then that count). "
             "Ribbons are SEM across vials.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94)); fig.savefig(OUT / "fig1_climbing_curves.png", dpi=170)
    plt.close(fig)


def fig2_endpoints(m):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    rng = np.random.default_rng(1)
    for ax, t in zip(axes, (10, 60)):
        s = m[m.time_sec == t]
        xt, xl = [], []
        x = 0
        for sx in ("Female", "Male"):
            for tr in ("Untreated", "Withaferin"):
                for g in GENO:
                    q = s[(s.sex == sx) & (s.treatment == tr) & (s.genotype == g)].PI_hand.values
                    jitter = rng.uniform(-0.12, 0.12, len(q))
                    ax.scatter(x + jitter, q, s=34, color=COL[g], marker=MRK[g], alpha=0.85,
                               edgecolor=SURF, linewidth=0.8, zorder=3)
                    ax.plot([x - 0.28, x + 0.28], [q.mean()] * 2, color=INK, linewidth=2, zorder=4)
                    x += 1
                xt.append(x - 2); xl.append(f"{sx}\n{tr}")
                x += 0.8
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=9)
        ax.set_title(f"t = {t} s" + ("  (protocol endpoint)" if t == 10 else ""), loc="left", fontsize=11)
        ax.grid(axis="y"); ax.set_ylim(-0.02, 1.08)
    axes[0].set_ylabel("Performance index")
    handles = [plt.Line2D([], [], linestyle="none", color=COL[g], marker=MRK[g], markersize=7,
                          markeredgecolor=SURF, label=g) for g in GENO]
    handles.append(plt.Line2D([], [], color=INK, linewidth=2, label="mean of vials"))
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.99, 0.97))
    fig.suptitle("Every vial at the two endpoints — the Htt response to withaferin is clearest at 60 s in males",
                 x=0.01, ha="left", fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "fig2_endpoints.png", dpi=170)
    plt.close(fig)


def fig3_agreement(m):
    p = m.dropna(subset=["reported_count"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    for g in GENO:
        q = p[p.genotype == g]
        ax.scatter(q.hand_count + np.random.default_rng(2).uniform(-0.2, 0.2, len(q)), q.reported_count,
                   s=22, color=COL[g], marker=MRK[g], alpha=0.7, edgecolor=SURF, linewidth=0.5, label=g)
    lim = max(p.hand_count.max(), p.reported_count.max()) + 1
    ax.plot([0, lim], [0, lim], color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.fill_between([0, lim], [-HUMAN_FLOOR, lim - HUMAN_FLOOR], [HUMAN_FLOOR, lim + HUMAN_FLOOR],
                    color=AXIS, alpha=0.25, linewidth=0, label=f"±{HUMAN_FLOOR:.0f} flies (human repeat floor)")
    ax.set_xlabel("Operator hand count (flies above band)"); ax.set_ylabel("Tool count")
    ax.set_title("Tool against hand count, every reading", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5); ax.grid(axis="both"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax = axes[1]
    mean_ = (p.reported_count + p.hand_count) / 2; diff = p.err
    bias, sd = diff.mean(), diff.std(ddof=1)
    for g in GENO:
        k = p.genotype == g
        ax.scatter(mean_[k], diff[k], s=22, color=COL[g], marker=MRK[g], alpha=0.7, edgecolor=SURF, linewidth=0.5)
    ax.axhspan(-HUMAN_FLOOR, HUMAN_FLOOR, color=AXIS, alpha=0.25, linewidth=0)
    ax.axhline(bias, color=INK, linewidth=1.5)
    for y, lab in ((bias + 1.96 * sd, "+1.96 SD"), (bias - 1.96 * sd, "−1.96 SD")):
        ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"{lab} = {y:+.1f}", (mean_.max(), y), xytext=(-4, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
    ax.annotate(f"bias = {bias:+.2f}", (mean_.max(), bias), xytext=(-4, 3), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK)
    # proportional bias: the disagreement depends on the level, which is the
    # signature of a floor (label print on near-empty tubes) and a ceiling
    # (crowding and the plug on full ones). Quantified, not eyeballed.
    slope, icpt, r, pval, _ = stats.linregress(mean_, diff)
    xs = np.array([mean_.min(), mean_.max()])
    ax.plot(xs, icpt + slope * xs, color=INK, linewidth=1.2, linestyle=(0, (1, 2)))
    line1 = f"proportional bias: slope {slope:+.2f} flies per fly  (r = {r:.2f}, p = {pval:.1e})"
    line2 = f"tool reads {icpt + slope*3:+.1f} at 3 flies and {icpt + slope*15:+.1f} at 15"
    ax.text(0.98, 0.74, line1 + chr(10) + line2,
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="top", ha="right")
    ax.set_xlabel("Mean of tool and hand count"); ax.set_ylabel("Tool − hand (flies)")
    ax.set_title("Bland–Altman: the error depends on the count", loc="left", fontsize=11); ax.grid(axis="y")
    fig.suptitle(f"Tool validation on {len(p)} hand-counted readings, {p.groupby(['video','position']).ngroups} cylinders "
                 f"— MAE {diff.abs().mean():.2f}, bias {bias:+.2f}, within 3 flies {100*(diff.abs()<=3).mean():.0f}%",
                 x=0.01, ha="left", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig3_tool_agreement.png", dpi=170)
    plt.close(fig)


def fig4_tool_vs_hand_curves(m):
    p = m.dropna(subset=["reported_count"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, g in zip(axes, GENO):
        q = p[p.genotype == g].groupby("time_sec")
        h, t_ = q.PI_hand.mean(), q.PI_tool.mean()
        hs, ts = q.PI_hand.apply(sem), q.PI_tool.apply(sem)
        ax.fill_between(h.index, h - hs, h + hs, color=COL[g], alpha=0.15, linewidth=0)
        ax.plot(h.index, h, color=COL[g], linewidth=2.2, marker=MRK[g], markersize=5, markeredgecolor=SURF, label="hand count")
        ax.plot(t_.index, t_, color=INK2, linewidth=1.6, linestyle=(0, (4, 2)), marker=MRK[g], markersize=4,
                markerfacecolor=SURF, label="tool")
        ax.set_title(f"{g}  (n = {q.size().iloc[0]} vials, all conditions)", loc="left", fontsize=10.5)
        ax.grid(axis="y"); ax.set_ylim(0, 1.0); ax.set_xlim(0, 125); ax.set_xlabel("Seconds")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0].set_ylabel("Performance index (mean ± SEM)")
    fig.suptitle("Does the tool track the curve? Hand count (solid) against tool (dashed), pooled across conditions",
                 x=0.01, ha="left", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "fig4_tool_vs_hand_curves.png", dpi=170)
    plt.close(fig)


def hedges_g(a, b):
    na, nb = len(a), len(b); sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    return g * (1 - 3 / (4 * (na + nb) - 9))


def stats_table(m):
    print("\nCOMPARISONS (hand-count PI; Welch t, Hedges g). n = 3 vials per cell: read effect sizes, treat p as indicative only.")
    rows = []
    for t in (10, 60):
        s = m[m.time_sec == t]
        for sx in ("Female", "Male"):
            for tr in ("Untreated", "Withaferin"):
                w = s[(s.sex == sx) & (s.treatment == tr) & (s.genotype == "Wildtype")].PI_hand
                h = s[(s.sex == sx) & (s.treatment == tr) & (s.genotype == "Htt")].PI_hand
                if len(w) > 1 and len(h) > 1:
                    rows.append((t, f"{sx} {tr}", "Wildtype vs Htt", w.mean() - h.mean(), hedges_g(w, h), stats.ttest_ind(w, h, equal_var=False).pvalue, len(w), len(h)))
            u = s[(s.sex == sx) & (s.treatment == "Untreated") & (s.genotype == "Htt")].PI_hand
            v = s[(s.sex == sx) & (s.treatment == "Withaferin") & (s.genotype == "Htt")].PI_hand
            if len(u) > 1 and len(v) > 1:
                rows.append((t, f"{sx} Htt", "Withaferin vs untreated", v.mean() - u.mean(), hedges_g(v, u), stats.ttest_ind(v, u, equal_var=False).pvalue, len(v), len(u)))
    d = pd.DataFrame(rows, columns=["t_s", "cell", "contrast", "delta_PI", "hedges_g", "welch_p", "n1", "n2"])
    print(d.round(3).to_string(index=False)); d.to_csv(OUT / "d15_stats.csv", index=False)


def main():
    m = load(); m.to_csv(OUT / "d15_analysis.csv", index=False)
    p = m.dropna(subset=["reported_count"])
    print(f"hand counts: {len(m)} readings, {m.groupby(['video','position']).ngroups} cylinders, {m.video.nunique()} videos")
    print(f"paired with tool: {len(p)}   MAE {p.err.abs().mean():.2f}  bias {p.err.mean():+.2f}  within3 {100*(p.err.abs()<=3).mean():.0f}%")
    t10 = p[p.time_sec == 10]; print(f"   at t=10 s: MAE {t10.err.abs().mean():.2f}  bias {t10.err.mean():+.2f}")
    print("N used:", m.groupby(['video','genotype']).N.first().value_counts().to_dict())
    fig1_curves(m); fig2_endpoints(m); fig3_agreement(m); fig4_tool_vs_hand_curves(m)
    stats_table(m)
    print("\nwrote fig1..fig4 to", OUT)


if __name__ == "__main__":
    main()
