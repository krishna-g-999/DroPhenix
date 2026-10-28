"""
DroPhenix — dam_figures.py
Figures for a DAM run in the format of the Drosophila sleep literature, drawn
from a `dam_pipeline.DamRun` and knowing nothing about genotype names.

  headline(run, ...)      the answers: forest of primary effects with FDR q
  profiles(run, ...)      sleep per 30-min epoch vs ZT, mean +/- SEM, L/D bar;
                          rows = stratum levels, columns = genotype or run
  bars(run, ...)          mean +/- SEM with every fly; x = genotype, paired
                          bars per treatment; per-stratum panels; brackets
  age_bars(run_a, run_b)  the same for two runs, phase-normalised

Every statistic drawn comes from the run's contrast tables; nothing is
computed here. Two rules keep the figures honest:
  - brackets inside a per-sex panel carry the per-sex exact Mann-Whitney p
    (uncorrected, the convention of the literature); the pre-specified
    stratified FDR result is printed as the row header so the two tiers are
    never confused;
  - comparisons confounded with monitor are drawn grey with "(conf.)" and
    never given asterisks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from modules import dam_pipeline as P

WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
INK, MUTED, AXIS = "#0b0b0b", "#6b6b6b", "#c9c9c9"
GOOD = "#0E6B45"
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
         "axes.edgecolor": INK, "axes.linewidth": 0.8,
         "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
         "axes.spines.top": False, "axes.spines.right": False,
         "legend.frameon": False, "legend.fontsize": 7.5,
         "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
         "figure.facecolor": "white", "savefig.facecolor": "white"}


def colours(run: P.DamRun) -> Dict[str, str]:
    d = run.design
    # Order of first appearance down the monitors -- the experimenter's own
    # ordering -- with the reference genotype first.
    pf = run.per_fly.sort_values(["Monitor", "Channel"])
    levels = list(dict.fromkeys(pf[d.genotype_col].dropna().astype(str)))
    if d.ref_genotype in levels:
        levels = [d.ref_genotype] + [l for l in levels if l != d.ref_genotype]
    return {g: WONG[i % len(WONG)] for i, g in enumerate(levels)}


def stars(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _sec_p(df: pd.DataFrame, metric, stratum, ref, cmp_):
    if df is None or df.empty:
        return np.nan
    d = df[(df.analysis == "secondary") & (df.metric == metric) & (df.stratum == stratum)
           & (df.ref == ref) & (df.cmp == cmp_)]
    return float(d.p.iloc[0]) if len(d) else np.nan


def _prim(df: pd.DataFrame, metric, **eq):
    if df is None or df.empty:
        return None
    d = df[(df.analysis == "primary") & (df.metric == metric)]
    for k, v in eq.items():
        d = d[d[k] == v]
    return d.iloc[0] if len(d) else None


# ═════════════════════════════════════════════════════════════════════════════
def headline(run: P.DamRun, run_b: Optional[P.DamRun] = None, cmp: Optional[dict] = None,
             metrics: Sequence[str] = ("Total_sleep_min_per_day", "Night_sleep_min_per_day",
                                       "Longest_bout_min", "Sleep_latency_night_min",
                                       "Total_activity_per_day")):
    """The answers on one page: primary effects only, with FDR q."""
    d = run.design; col = colours(run)
    panels = []
    # A: each non-reference genotype vs reference
    rows = []
    gen = run.contrasts.get("genotype")
    for g in [x for x in sorted(run.per_fly[d.genotype_col].unique()) if x != d.ref_genotype]:
        for m in metrics:
            r = _prim(gen, m, ref=d.ref_genotype, cmp=g)
            if r is not None:
                rows.append((f"{P.LABEL[m]}: {g}" + ("" if r.monitor_clean else " (conf.)"),
                             r.hedges_g, r.g_lo, r.g_hi, r.q_BH, col[g] if r.monitor_clean else MUTED))
    panels.append((f"{d.ref_genotype} vs other genotypes ({d.ref_treatment})", rows,
                   f"Hedges' g   ← lower than {d.ref_genotype}    higher →"))
    # B: treatment pooled, clean and all
    rows = []
    for key, tag, c in (("treatment_pooled_clean", "clean blocks", WONG[0]),
                        ("treatment_pooled_all", "all blocks", WONG[5])):
        t = run.contrasts.get(key)
        if t is None or t.empty:
            continue
        for trt in sorted(t.cmp.unique()):
            for m in metrics:
                r = _prim(t, m, cmp=trt)
                if r is not None:
                    rows.append((f"{P.LABEL[m]}: {trt} ({tag})", r.hedges_g, r.g_lo, r.g_hi, r.q_BH, c))
    panels.append((f"treatment vs {d.ref_treatment}, pooled over genotype × {d.stratify_col or 'all'}",
                   rows, "Hedges' g   ← lower with treatment    higher →"))
    # C: timepoint
    if run_b is not None and cmp is not None:
        rows = []
        tp = cmp["timepoint"]
        for g in sorted(run.per_fly[d.genotype_col].unique()):
            for trt in sorted(run.per_fly[d.treatment_col].unique()):
                for m in ("Total_sleep_min_per_day", "Day_sleep_per_light_h"):
                    r = _prim(tp, m, stratum=f"{g} {trt} stratified")
                    if r is not None:
                        rows.append((f"{g} {trt}: {P.LABEL[m]}", r.hedges_g, r.g_lo, r.g_hi, r.q_BH, col[g]))
        panels.append((f"{run.name} → {run_b.name}, within each group", rows,
                       f"Hedges' g   ← less at {run_b.name}    more →"))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 0.28 * max(len(r) for _, r, _ in panels) + 2.2))
        axes = np.atleast_1d(axes)
        for ax, (title, rows, xlab) in zip(axes, panels):
            y = np.arange(len(rows))[::-1]
            for yy, (lab, g, lo, hi, q, c) in zip(y, rows):
                sig = np.isfinite(q) and q < 0.05
                ax.plot([lo, hi], [yy, yy], color=c, lw=2.2 if sig else 1.4, alpha=1 if sig else 0.55)
                ax.plot(g, yy, "s" if sig else "o", color=c, ms=8 if sig else 6.5, markeredgecolor="white", markeredgewidth=0.8)
                ax.annotate(f"q = {q:.3f}" + ("  †" if sig else "") if np.isfinite(q) else "",
                            (3.55, yy), fontsize=7.5, color=GOOD if sig else MUTED, va="center",
                            fontweight="bold" if sig else "normal")
            ax.axvline(0, color=AXIS, lw=1, ls="--"); ax.axvspan(-4.2, 0, color="#2b2b2b", alpha=0.03, lw=0)
            ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
            ax.set_xlim(-4.2, 5.2); ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax.set_xlabel(xlab)
            ax.set_title(title, loc="left", fontweight="bold")
        fig.legend(handles=[Line2D([], [], marker="s", color=INK, ls="", ms=8, label="survives BH-FDR at 5% (†)"),
                            Line2D([], [], marker="o", color=MUTED, ls="", ms=6.5, label="does not"),
                            Line2D([], [], color=INK, lw=2, label="95% CI")],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"Primary analysis — {d.stratify_col or 'unstratified'}-stratified tests, FDR within each family",
                     y=0.995, fontsize=10)
        fig.tight_layout(w_pad=2.5, rect=(0, 0.06, 1, 0.95))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
def _ld_bar(ax, L_h, y=-3.6, h=1.6, second=None):
    ax.add_patch(Rectangle((0, y), L_h, h, facecolor="white", edgecolor=INK, lw=0.8, clip_on=False))
    ax.add_patch(Rectangle((L_h, y), 24 - L_h, h, facecolor=INK, edgecolor=INK, lw=0.8, clip_on=False))
    if second is not None:
        ax.add_patch(Rectangle((0, y + h + 0.25), second, h * 0.5, facecolor="white", edgecolor=MUTED, lw=0.7, clip_on=False))
        ax.add_patch(Rectangle((second, y + h + 0.25), 24 - second, h * 0.5, facecolor=MUTED, edgecolor=MUTED, lw=0.7, clip_on=False))


def _ribbon(ax, prof, sel, color, ls="-", value="sleep"):
    s = prof[sel]
    if s.empty:
        return 0
    m = s.groupby("zt")[value].agg(["mean", "sem", "count"])
    ax.plot(m.index, m["mean"], color=color, ls=ls, lw=1.6, zorder=3)
    ax.fill_between(m.index, m["mean"] - m["sem"].fillna(0), m["mean"] + m["sem"].fillna(0),
                    color=color, alpha=0.18 if ls == "-" else 0.10, lw=0, zorder=2)
    return int(m["count"].max())


def _style_profile(ax, ylab, value):
    ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
    if value == "sleep":
        ax.set_ylim(0, 30); ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(axis="x", pad=13); ax.spines["bottom"].set_visible(False)
    if ylab:
        ax.set_ylabel("sleep (min / 30 min)" if value == "sleep" else "activity (counts / 30 min)")
    ax.set_xlabel("Zeitgeber time (h)")


def profiles(run: P.DamRun, treatment: Optional[str] = None, by: str = "genotype",
             run_b: Optional[P.DamRun] = None, value: str = "sleep"):
    """
    by="genotype":  columns = one panel per stratum level; genotypes overlaid;
                    `treatment` selects which arm (default: reference).
    by="treatment": columns = genotypes; reference treatment solid, others dashed.
    by="run":       columns = genotypes; `run` dashed vs `run_b` solid (reference treatment).
    Rows = stratum levels (e.g. males / females).
    """
    d = run.design; col = colours(run); prof = run.profiles
    L = run.photoperiod_min / 60.0
    strata = sorted(prof["Sex"].unique()) if d.stratify_col == "Sex" else [None]
    genos = list(col)
    trt = treatment or d.ref_treatment
    trts = sorted(prof["Treatment"].unique())
    if by == "genotype":
        cols_ = [None]
    else:
        cols_ = genos
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(strata), len(cols_), figsize=(3.4 * len(cols_) + 0.6, 2.6 * len(strata) + 0.5),
                                 sharey=True, squeeze=False)
        for i, sx in enumerate(strata):
            for j, c in enumerate(cols_):
                ax = axes[i, j]; handles = []
                base = (prof["Sex"] == sx) if sx is not None else pd.Series(True, index=prof.index)
                if by == "genotype":
                    for g in genos:
                        n = _ribbon(ax, prof, base & (prof.Genotype == g) & (prof.Treatment == trt), col[g], "-", value)
                        handles.append(Line2D([], [], color=col[g], lw=1.6, label=f"{g} (n={n})"))
                    _ld_bar(ax, L); title = f"{trt}" + (f"  ·  {sx}" if sx else "")
                elif by == "treatment":
                    for k, t in enumerate(trts):
                        ls = "-" if t == d.ref_treatment else ["--", ":", "-."][k % 3]
                        n = _ribbon(ax, prof, base & (prof.Genotype == c) & (prof.Treatment == t), col[c], ls, value)
                        handles.append(Line2D([], [], color=col[c], lw=1.6, ls=ls, label=f"{t} (n={n})"))
                    _ld_bar(ax, L); title = f"{c}" + (f"  ·  {sx}" if sx else "")
                else:
                    pb = run_b.profiles; Lb = run_b.photoperiod_min / 60.0
                    baseb = (pb["Sex"] == sx) if sx is not None else pd.Series(True, index=pb.index)
                    n1 = _ribbon(ax, prof, base & (prof.Genotype == c) & (prof.Treatment == trt), col[c], "--", value)
                    n2 = _ribbon(ax, pb, baseb & (pb.Genotype == c) & (pb.Treatment == trt), col[c], "-", value)
                    handles = [Line2D([], [], color=col[c], lw=1.6, ls="--", label=f"{run.name} (n={n1})"),
                               Line2D([], [], color=col[c], lw=1.6, ls="-", label=f"{run_b.name} (n={n2})")]
                    _ld_bar(ax, Lb, second=L); title = f"{c}" + (f"  ·  {sx}" if sx else "")
                _style_profile(ax, j == 0, value)
                ax.set_title(title, loc="left", fontweight="bold")
                ax.legend(handles=handles, loc="lower right" if value == "sleep" else "upper right")
        head = {"genotype": f"{value.capitalize()} profile by genotype, {trt}",
                "treatment": f"{value.capitalize()} profile: effect of treatment",
                "run": f"{value.capitalize()} profile: {run.name} (dashed) vs "
                       f"{run_b.name if run_b else '?'} (solid), {trt}"}[by]
        fig.suptitle(head
                     + f"  —  mean ± SEM per 30-min epoch; bar = light : dark ({L:.0f}L:{24-L:.0f}D"
                     + (f"; thin bar {run.photoperiod_min/60:.0f}L" if by == "run" else "") + ")",
                     y=1.01, fontsize=9.5)
        fig.tight_layout(h_pad=2.2)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
def _bracket(ax, x1, x2, y, text, color=INK, fs=7.5):
    yl = ax.get_ylim(); h = 0.018 * (yl[1] - yl[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.15, text, ha="center", va="bottom", fontsize=fs, color=color,
            fontweight="bold" if text not in ("ns", "(conf.)", "") else "normal")


def bars(run: P.DamRun, metrics: Sequence[str], title: str = ""):
    """Paper-style bar panels: x = genotype, paired bars per treatment, one
    column per stratum level; brackets from the run's contrast tables."""
    d = run.design; col = colours(run); pf = run.per_fly
    genos = list(col); trts = sorted(pf[d.treatment_col].unique())
    strata = sorted(pf[d.stratify_col].unique()) if d.stratify_col else [None]
    gen, trt_f, tpc = run.contrasts.get("genotype"), run.contrasts.get("treatment"), run.contrasts.get("treatment_pooled_clean")
    rng = np.random.default_rng(7); w = 0.8 / len(trts)
    HATCH = {t: ["////", "....", "xxxx"][i % 3] for i, t in enumerate([x for x in trts if x != d.ref_treatment])}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(metrics), len(strata), figsize=(3.0 * len(strata) + 0.8, 2.55 * len(metrics) + 0.6),
                                 sharey="row", squeeze=False)
        for r_, m in enumerate(metrics):
            top = float(np.nanmax(pf[m])) if pf[m].notna().any() else 1.0
            for c_, sx in enumerate(strata):
                ax = axes[r_, c_]; xs = {}
                sub = pf[pf[d.stratify_col] == sx] if sx is not None else pf
                for i, g in enumerate(genos):
                    for k, t in enumerate(trts):
                        x = i + (k - (len(trts) - 1) / 2) * (w + 0.04); xs[(g, t)] = x
                        v = sub[(sub[d.genotype_col] == g) & (sub[d.treatment_col] == t)][m].dropna().to_numpy(float)
                        if not len(v):
                            continue
                        mu, se = v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                        ax.bar(x, mu, w, color=col[g] if t == d.ref_treatment else "white", edgecolor=col[g], lw=1.0,
                               hatch=HATCH.get(t), zorder=2)
                        ax.errorbar(x, mu, yerr=se, color=INK, lw=0.9, capsize=2.5, zorder=4)
                        ax.scatter(x + rng.uniform(-w * 0.32, w * 0.32, len(v)), v, s=9, color=INK, alpha=0.65, zorder=5, linewidths=0)
                ax.set_ylim(0, top * 1.45); ax.set_xticks(range(len(genos))); ax.set_xticklabels(genos)
                ax.set_xlim(-0.6, len(genos) - 0.4); ax.tick_params(axis="x", length=0)
                if c_ == 0:
                    ax.set_ylabel(f"{P.LABEL[m]}\n({P.UNIT[m]})")
                # brackets: treatment within genotype (row 1), genotype vs reference (row 2+)
                y0 = top * 1.06; step = 0.085 * top * 1.45
                for i, g in enumerate(genos):
                    for t in trts:
                        if t == d.ref_treatment:
                            continue
                        p = _sec_p(trt_f, m, f"{g} {sx}" if sx is not None else f"{g} stratified", d.ref_treatment, t)
                        rr = _prim(trt_f, m, stratum=f"{g} stratified", cmp=t)
                        clean = bool(rr.monitor_clean) if rr is not None else True
                        _bracket(ax, xs[(g, d.ref_treatment)], xs[(g, t)], y0, stars(p) if clean else "(conf.)", INK if clean else MUTED)
                for k, g in enumerate([x for x in genos if x != d.ref_genotype]):
                    p = _sec_p(gen, m, str(sx) if sx is not None else "stratified", d.ref_genotype, g)
                    rr = _prim(gen, m, ref=d.ref_genotype, cmp=g)
                    clean = bool(rr.monitor_clean) if rr is not None else True
                    _bracket(ax, xs[(d.ref_genotype, d.ref_treatment)], xs[(g, d.ref_treatment)], y0 + (k + 1) * step,
                             stars(p) if clean else "(conf.)", INK if clean else MUTED)
            # row header: the primary result
            parts = []
            for g in [x for x in genos if x != d.ref_genotype]:
                rr = _prim(gen, m, ref=d.ref_genotype, cmp=g)
                if rr is not None:
                    parts.append(f"{d.ref_genotype} vs {g} q={rr.q_BH:.3f}{'†' if rr.significant else ''}")
            if tpc is not None and not tpc.empty:
                for t in sorted(tpc.cmp.unique()):
                    rr = _prim(tpc, m, cmp=t)
                    if rr is not None:
                        parts.append(f"{t} (clean) q={rr.q_BH:.3f}{'†' if rr.significant else ''}")
            axes[r_, 0].set_title(f"{P.LABEL[m]}  —  stratified FDR:  " + ";  ".join(parts),
                                  loc="left", fontsize=7.2, color="#39424F", pad=6)
        handles = [Patch(facecolor=MUTED, edgecolor=MUTED, label=d.ref_treatment)] + \
                  [Patch(facecolor="white", edgecolor=MUTED, hatch=HATCH[t], label=t)
                   for t in trts if t != d.ref_treatment] + \
                  [Line2D([], [], color=INK, marker="o", ls="", ms=3, label="one fly"),
                   Line2D([], [], color=INK, lw=0.8, label="bracket: per-stratum exact Mann–Whitney, uncorrected"),
                   Line2D([], [], color=MUTED, lw=0.8, label="(conf.) = confounded with monitor; not tested")]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7.2)
        fig.suptitle((title or f"{run.name}") + "\nbars = mean ± SEM; * p<0.05, ** p<0.01, *** p<0.001;\n"
                     "† = survives BH-FDR at 5% in the primary stratified test", y=0.995, fontsize=9)
        fig.tight_layout(h_pad=2.6, rect=(0, 0.045, 1, 0.94))
        for c_, sx in enumerate(strata):
            bb = axes[0, c_].get_position()
            fig.text(bb.x0, bb.y1 + 0.032, f"{sx}" if sx is not None else "all", fontsize=9, fontweight="bold", ha="left", va="bottom")
    return fig


def age_bars(run_a: P.DamRun, run_b: P.DamRun, cmp: dict,
             metrics: Sequence[str] = ("Total_sleep_min_per_day", "Day_sleep_per_light_h",
                                       "Night_sleep_per_dark_h", "Total_activity_per_day")):
    """Two runs side by side within each genotype x treatment x stratum."""
    d = run_a.design; col = colours(run_a)
    genos = list(col); trts = sorted(run_a.per_fly[d.treatment_col].unique())
    strata = sorted(run_a.per_fly[d.stratify_col].unique()) if d.stratify_col else [None]
    cols_ = [(t, s) for t in trts for s in strata]
    tp = cmp["timepoint"]; rng = np.random.default_rng(3); w = 0.36
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(metrics), len(cols_), figsize=(2.8 * len(cols_) + 0.8, 2.55 * len(metrics) + 0.6),
                                 sharey="row", squeeze=False)
        for r_, m in enumerate(metrics):
            top = float(np.nanmax(pd.concat([run_a.per_fly[m], run_b.per_fly[m]])))
            for c_, (t, sx) in enumerate(cols_):
                ax = axes[r_, c_]; xs = {}
                for i, g in enumerate(genos):
                    for k, (run, tag) in enumerate(((run_a, run_a.name), (run_b, run_b.name))):
                        pf = run.per_fly
                        sel = (pf[d.genotype_col] == g) & (pf[d.treatment_col] == t)
                        if sx is not None:
                            sel &= pf[d.stratify_col] == sx
                        v = pf.loc[sel, m].dropna().to_numpy(float)
                        x = i + (k - 0.5) * (w + 0.04); xs[(g, tag)] = x
                        if not len(v):
                            continue
                        mu, se = v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                        ax.bar(x, mu, w, color=col[g], alpha=0.45 if k == 0 else 1.0, edgecolor=col[g], lw=1.0, zorder=2)
                        ax.errorbar(x, mu, yerr=se, color=INK, lw=0.9, capsize=2.5, zorder=4)
                        ax.scatter(x + rng.uniform(-w * 0.32, w * 0.32, len(v)), v, s=9, color=INK, alpha=0.65, zorder=5, linewidths=0)
                ax.set_ylim(0, top * 1.45); ax.set_xticks(range(len(genos))); ax.set_xticklabels(genos)
                ax.set_xlim(-0.6, len(genos) - 0.4); ax.tick_params(axis="x", length=0)
                if c_ == 0:
                    ax.set_ylabel(f"{P.LABEL[m]}\n({P.UNIT[m]})")
                for i, g in enumerate(genos):
                    p = _sec_p(tp, m, f"{g} {t} {sx}" if sx is not None else f"{g} {t} stratified", run_a.name, run_b.name)
                    _bracket(ax, xs[(g, run_a.name)], xs[(g, run_b.name)], top * 1.06, stars(p))
            qs = []
            for g in genos:
                for t in trts:
                    rr = _prim(tp, m, stratum=f"{g} {t} stratified")
                    if rr is not None:
                        qs.append(f"{g} {t} q={rr.q_BH:.3f}{'†' if rr.significant else ''}")
            axes[r_, 0].set_title(f"{P.LABEL[m]}  —  stratified FDR:  " + ";  ".join(qs),
                                  loc="left", fontsize=7.2, color="#39424F", pad=6)
        fig.legend(handles=[Patch(facecolor=MUTED, alpha=0.45, label=f"{run_a.name} ({run_a.photoperiod_min/60:.0f}L)"),
                            Patch(facecolor=MUTED, label=f"{run_b.name} ({run_b.photoperiod_min/60:.0f}L)"),
                            Line2D([], [], color=INK, marker="o", ls="", ms=3, label="one fly"),
                            Line2D([], [], color=INK, lw=0.8, label="bracket: per-stratum exact Mann–Whitney, uncorrected")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=7.2)
        fig.suptitle(f"{run_a.name} vs {run_b.name} within each group\nday and night sleep per hour of the phase recorded;\n"
                     "tubes are not paired across runs", y=0.995, fontsize=9)
        fig.tight_layout(h_pad=2.6, rect=(0, 0.045, 1, 0.94))
        for c_, (t, sx) in enumerate(cols_):
            bb = axes[0, c_].get_position()
            fig.text(bb.x0, bb.y1 + 0.032, f"{t}" + (f"  ·  {sx}" if sx is not None else ""),
                     fontsize=9, fontweight="bold", ha="left", va="bottom")
    return fig


def outlier_map(run: P.DamRun):
    d = run.design; col = colours(run); pf = run.per_fly
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for g in col:
            s = pf[pf[d.genotype_col] == g]; out = s["Outlier"].to_numpy()
            ax.scatter(s.Total_activity_per_day[~out], s.Total_sleep_min_per_day[~out], s=24, color=col[g], alpha=0.75,
                       edgecolor="white", linewidth=0.5, label=g)
            for k_, (_, r) in enumerate(s[out].iterrows()):
                ax.scatter(r.Total_activity_per_day, r.Total_sleep_min_per_day, s=90, facecolor="none", edgecolor=col[g], linewidth=1.6)
                ax.annotate(f"{r.Monitor.split('_')[-1] if '_' in str(r.Monitor) else r.Monitor} C{r.Channel}: {r.Outlier_reason}",
                            (r.Total_activity_per_day, r.Total_sleep_min_per_day),
                            xytext=(7, 4) if k_ % 2 == 0 else (-7, -14), textcoords="offset points", fontsize=6.4,
                            ha="left" if k_ % 2 == 0 else "right")
        ax.set_xlabel("total activity (counts / 24 h)"); ax.set_ylabel("total sleep (min / 24 h)"); ax.set_ylim(0, 1440)
        ax.legend(loc="lower right")
        ax.set_title(f"{run.name}: flagged outliers (open markers) — |modified z| > {P.OUTLIER_Z} vs own group median",
                     loc="left", fontsize=9)
        fig.tight_layout()
    return fig
