"""
DroPhenix Analytics -- modules/visualizer.py
Publication-quality figures. Wong (2011) palette, 300 DPI, Arial, white background.
All axes labelled with units. SEM error bars. Weissgerber (2015) raw-data overlay.
"""
from __future__ import annotations
import io, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# --- Wong (2011) Nat Methods 8:441 -- colorblind-safe palette -----------------
WONG = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000"]
TREATMENT_COLOURS = {
    "Untreated":"#666666","Vehicle":"#666666","DMSO":"#666666","Control":"#666666",
    "Spermine":"#0072B2","Pantothenate":"#009E73","Spermine+Pantothenate":"#D55E00",
}
GENOTYPE_MARKERS = {"Oregon":"o","TDP43":"s","FUS":"^","Default":"D"}
SEX_LINESTYLE    = {"Male":"-","Female":"--"}
SEX_HATCH        = {"Male":"","Female":"//"}

DPI=300; FS_SINGLE=(7.0,4.5); FS_DOUBLE=(12.0,4.5); FS_TALL=(7.0,6.5); FS_SQ=(5.5,5.5)
FT_TITLE=11; FT_LABEL=10; FT_TICK=8; FT_ANNOT=7; FT_LEGEND=8

def _style():
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
        "font.size":FT_TICK,"axes.labelsize":FT_LABEL,"axes.titlesize":FT_TITLE,
        "axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
        "xtick.major.width":0.8,"ytick.major.width":0.8,
        "xtick.labelsize":FT_TICK,"ytick.labelsize":FT_TICK,
        "legend.fontsize":FT_LEGEND,"legend.frameon":False,
        "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight",
        "savefig.facecolor":"white","lines.linewidth":1.8,
        "lines.markersize":5,"errorbar.capsize":3,
    })
_style()

# --- Helpers ------------------------------------------------------------------
def _col(label:str, idx:int)->str:
    return TREATMENT_COLOURS.get(label, WONG[idx % len(WONG)])

def _to_bytes(fig, fmt="png")->bytes:
    buf=io.BytesIO()
    fig.savefig(buf,format=fmt,dpi=DPI,bbox_inches="tight",facecolor="white")
    buf.seek(0); return buf.read()

def _exp_decay_local(t, PI0:float, k:float)->np.ndarray:
    """PI(t) = PI0 * exp(-k*t). Inline -- avoids private cross-module import."""
    return PI0 * np.exp(-k * np.asarray(t, dtype=float))

def _get_key(d:dict, *keys):
    """Try key variants; return first non-empty match."""
    for k in keys:
        v=d.get(k)
        if v is not None and not (hasattr(v,"empty") and v.empty): return v
    return None

def _rl(row, col:str, n:int=12)->str:
    """Safe row label: row.get(col)[:n] with no f-string quoting issues."""
    return str(row.get(col,""))[:n]

def _pval_stars(p)->str:
    if p is None: return ""
    try: p=float(p)
    except: return ""
    if p<0.001: return "***"
    if p<0.01:  return "**"
    if p<0.05:  return "*"
    return "ns"

# --- 0. Significance bracket -------------------------------------------------

# ── Gaps, not zeros (register N22) ───────────────────────────────────────────
# A missing measurement used to be plotted as 0 and a missing SEM as a
# zero-length error bar. In this platform's metrics 0 is never "no data": a
# Performance Index of 0 is a vial with exactly half its flies above the
# scoring line, an SDQ of 0 is "no sex difference at all", a CRI of 0 is "no
# rescue", and a zero-width interval is a claim of perfect precision. Each of
# those is a result, and substituting it for an absent measurement publishes a
# finding nobody made -- in a figure, which is the form most likely to be
# reproduced without the table beside it.
#
# Every figure function therefore: draws nothing where there is no number,
# marks the gap so it is visible as a gap, counts what it could not draw in
# `fig.drophenix_gaps`, and reports the number of INDEPENDENT units behind each
# bar when the table carries one. The audit found the figures reported no n at
# all, which is worse than reporting the wrong one.
N_UNIT_COLS = ("n_rep", "n_replicates", "n_units", "n_vials_observed", "n")

GAP_HATCH = "xxx"
GAP_COLOUR = "#bbbbbb"


def _num(row, col):
    """A finite number from a row, or NaN. Never a substituted default."""
    try:
        v = row.get(col, np.nan) if hasattr(row, "get") else np.nan
    except Exception:
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


def _n_units(df):
    """The first independent-unit column the table actually carries."""
    for c in N_UNIT_COLS:
        if c in getattr(df, "columns", []):
            return c
    return None


def _mark_gap(ax, pos, horizontal=False, label="not measured"):
    """Draw the absence: a hatched zero-extent marker plus a text label."""
    if horizontal:
        ax.barh(pos, 0.0, height=0.65, color=GAP_COLOUR, hatch=GAP_HATCH,
                edgecolor="#777", linewidth=0.5, zorder=2)
        ax.annotate(label, (0.0, pos), textcoords="offset points", xytext=(4, 0),
                    fontsize=FT_ANNOT, color="#777", va="center")
    else:
        ax.bar(pos, 0.0, width=0.72, color=GAP_COLOUR, hatch=GAP_HATCH,
               edgecolor="#777", linewidth=0.5, zorder=2)
        ax.annotate(label, (pos, 0.0), textcoords="offset points", xytext=(0, 4),
                    fontsize=FT_ANNOT, color="#777", ha="center", rotation=90)


def _attach_gaps(fig, **counts):
    """Record on the figure what could not be drawn, and why."""
    g = dict(counts)
    g.setdefault("missing_metric", 0)
    g.setdefault("missing_sem", 0)
    g.setdefault("missing_ci", 0)
    g.setdefault("drawn", 0)
    g.setdefault("errorbars_drawn", 0)
    g.setdefault("n_units_reported", False)
    g["note"] = (
        "Counts of values this figure could NOT draw. A gap is drawn as a "
        "hatched zero-extent marker labelled 'not measured', never as a zero "
        "bar, and a missing SEM or interval is drawn as no error bar rather "
        "than as a zero-width one."
    )
    fig.drophenix_gaps = g
    return fig


def _nested_xaxis(ax, x_pos, tick_labels, groups, tick_fs=None,
                  group_fs=None, y_rule=-0.46, y_text=-0.50):
    """
    Draw a TWO-LEVEL categorical x axis: treatments on the ticks, genotype as a
    labelled band under the group of bars it spans.

    Both bar charts in this module got this wrong, in opposite directions.
    `t50_plot` packed both levels into one tick label, truncated and rotated,
    and on twelve groups they overlapped into unreadable text. `pi_barplot`
    labelled only the treatment, so the genotype appeared nowhere on the figure
    and the three blocks of bars were unidentifiable.

    The band sits below the rotated tick labels; `y_rule` is set low enough
    that a long label such as "Spermine+Pantothenate" rotated 38 degrees clears
    the rule rather than crossing it.

    `groups` is a sequence of `(name, [x positions])`. The band is drawn in
    axes-fraction coordinates below the tick labels with `clip_on=False`, so the
    caller must leave room for it -- `fig.subplots_adjust(bottom=...)` after
    `tight_layout`, which does not account for artists outside the axes.
    """
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(list(tick_labels), rotation=38, ha="right",
                       fontsize=tick_fs or (FT_ANNOT + 1))
    trans = ax.get_xaxis_transform()          # x in data, y in axes fraction
    for name, pos in groups:
        pos = [p for p in pos if p is not None]
        if not pos:
            continue
        a, b = min(pos), max(pos)
        ax.plot([a - 0.45, b + 0.45], [y_rule, y_rule], transform=trans,
                color="#999999", linewidth=0.8, clip_on=False, zorder=1)
        ax.text((a + b) / 2.0, y_text, str(name), transform=trans,
                ha="center", va="top", fontsize=group_fs or (FT_ANNOT + 1),
                color="#222222", clip_on=False)


def stats_annotation(ax,x1,x2,y,label,dy=1.5,lw=0.8):
    yt=y+dy
    ax.plot([x1,x1,x2,x2],[y,yt,yt,y],lw=lw,color="black")
    ax.text((x1+x2)/2,yt+dy*0.1,label,ha="center",va="bottom",fontsize=FT_ANNOT)

# --- 1. Climbing curves ------------------------------------------------------
def climbing_curves(agg_df,metric="Pct_mean",sem_col="Pct_sem",
    genotypes=None,treatments=None,sexes=None,
    title="Climbing Performance Over Time",ylabel="% Climbed (Mean +/- SEM)",
    show_raw=False,raw_df=None):
    """Line chart of climbing performance vs time. SEM shading. Optional raw overlay."""
    df=agg_df.copy()
    if genotypes:  df=df[df["Genotype"].isin(genotypes)]
    if treatments: df=df[df["Treatment"].isin(treatments)]
    if sexes:      df=df[df["Sex"].isin(sexes)]
    if df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    groups=df.groupby(["Genotype","Treatment","Sex"],sort=False)
    fig,ax=plt.subplots(figsize=FS_SINGLE)
    handles=[]
    for idx,((geno,trt,sex),grp) in enumerate(groups):
        grp=grp.sort_values("Time")
        color=_col(trt,idx)
        ls=SEX_LINESTYLE.get(sex,"-")
        mk=GENOTYPE_MARKERS.get(geno,GENOTYPE_MARKERS["Default"])
        y=grp[metric].values if metric in grp.columns else np.zeros(len(grp))
        x=grp["Time"].values
        sem=grp[sem_col].values if sem_col in grp.columns else np.zeros_like(y)
        ax.plot(x,y,color=color,linestyle=ls,marker=mk,
                markerfacecolor="white",markeredgewidth=1.4,zorder=3)
        ax.fill_between(x,y-sem,y+sem,color=color,alpha=0.13,zorder=2)
        if show_raw and raw_df is not None:
            rg=raw_df.loc[(raw_df["Genotype"]==geno)&(raw_df["Treatment"]==trt)&(raw_df["Sex"]==sex)]
            if not rg.empty and "Pct" in rg.columns:
                for _,rep in rg.groupby("Replicate"):
                    rep=rep.sort_values("Time")
                    ax.plot(rep["Time"],rep["Pct"],color=color,linestyle=ls,
                            alpha=0.22,linewidth=0.6,marker=mk,markersize=2.5,zorder=1)
        handles.append(Line2D([0],[0],color=color,linestyle=ls,marker=mk,
            markerfacecolor="white",markeredgewidth=1.4,label=geno+" | "+trt+" | "+sex))
    ax.set_xlabel("Time (s)",fontsize=FT_LABEL)
    ax.set_ylabel(ylabel,fontsize=FT_LABEL)
    ax.set_title(title,fontsize=FT_TITLE,pad=8)
    ax.set_ylim(-5,105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(length=3)
    ncol=max(1,groups.ngroups//6)
    if groups.ngroups<=12:
        ax.legend(handles=handles,loc="lower right",fontsize=FT_LEGEND,ncol=ncol,frameon=False)
    else:
        ax.legend(handles=handles,fontsize=FT_LEGEND-1,ncol=2,
                  bbox_to_anchor=(1.01,1),loc="upper left",frameon=False)
    fig.tight_layout()
    return fig,_to_bytes(fig)

# --- 2. PI bar chart ---------------------------------------------------------
def pi_barplot(pi_df,genotypes=None,treatments=None,sexes=None,
    title="Performance Index by Group",pairwise_df=None):
    """
    Grouped bar chart PI_mean +/- SEM. Individual replicate dots (Weissgerber 2015).

    Register N22. Three things changed here. A missing `PI_sem` is no longer
    drawn as `yerr=0.0` -- a zero-length error bar is a claim that the mean is
    known exactly, which is the opposite of what a missing SEM means. A missing
    `PI_mean` is drawn as a labelled gap, not as a bar at 0, because PI = 0 is a
    vial with half its flies above the line. And several rows matching one
    (Genotype, Treatment, Sex) cell is refused instead of silently plotting
    `.iloc[0]`: that happens when the grouping keys do not identify a cell --
    an extra Age or Batch, say -- and quietly showing the first of them puts an
    arbitrary subset of the experiment on the axis.

    The number of independent units behind each bar is annotated when the table
    carries one of `N_UNIT_COLS`.
    """
    df=pi_df.copy()
    if genotypes:  df=df[df["Genotype"].isin(genotypes)]
    if treatments: df=df[df["Treatment"].isin(treatments)]
    if sexes:      df=df[df["Sex"].isin(sexes)]
    if df.empty:
        fig,ax=plt.subplots(figsize=FS_DOUBLE)
        ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
        return _attach_gaps(fig),_to_bytes(fig)
    _cell=df.groupby(["Genotype","Treatment","Sex"],dropna=False).size()
    _multi=_cell[_cell>1]
    if len(_multi):
        raise ValueError(
            f"{len(_multi)} (Genotype, Treatment, Sex) cell(s) match more than one "
            f"row of the PI table: {[' | '.join(map(str,k)) for k in _multi.index[:5]]}. "
            f"The figure used to plot the first matching row and discard the rest "
            f"without a mark, so the bar shown was an arbitrary subset. Either the "
            f"table has not been aggregated, or the design has a key these three "
            f"columns do not cover (Age, Batch, Dose): aggregate it, or pass the "
            f"extra key as a filter."
        )
    _nc=_n_units(df)
    _gap_metric=_gap_sem=_drawn=_ebars=0
    trts=df["Treatment"].unique(); genos=df["Genotype"].unique()
    sxs=df["Sex"].unique()
    n_sx=len(sxs)
    fig,axes=plt.subplots(1,n_sx,figsize=(4.2*n_sx,4.5),sharey=True)
    if n_sx==1: axes=[axes]
    for ai,sex in enumerate(sxs):
        ax=axes[ai]; sdf=df[df["Sex"]==sex]
        x=0; x_pos=[]; x_lbl=[]; _groups=[]
        for geno in genos:
            _gpos=[]
            for ti,trt in enumerate(trts):
                row=sdf[(sdf["Genotype"]==geno)&(sdf["Treatment"]==trt)]
                if row.empty: x+=1; continue
                r0=row.iloc[0]
                pi_m=_num(r0,"PI_mean")
                pi_s=_num(r0,"PI_sem")
                if not np.isfinite(pi_m):
                    # PI = 0 is "half the flies climbed", so an absent
                    # measurement cannot be drawn there.
                    _mark_gap(ax,x); _gap_metric+=1
                else:
                    _drawn+=1
                    if np.isfinite(pi_s):
                        ax.bar(x,pi_m,yerr=pi_s,color=_col(trt,ti),width=0.72,
                               capsize=3,error_kw={"linewidth":0.8,"ecolor":"#333",
                                 # `ecolor` colours the error bar's LINE only;
                                 # its CAPS fall back to the default colour
                                 # cycle, which put 60 artists at #1f77b4 on
                                 # these figures -- the one colour on them that
                                 # is neither the house style nor
                                 # colour-blind-safe. Measured: `ecolor` alone
                                 # leaves the caps at #1f77b4 and
                                 # `markeredgecolor` does nothing -- `color` is
                                 # the key matplotlib uses for the cap style.
                                 "capthick":0.8,"color":"#333"},
                               edgecolor="black",linewidth=0.5,
                               hatch=SEX_HATCH.get(sex,""),zorder=3)
                        _ebars+=1
                    else:
                        # No SEM: no error bar. A zero-length one would assert
                        # that this mean is known exactly.
                        #
                        # MARKED BY ITS EDGE, NOT BY A LABEL ON EVERY BAR. This
                        # used to annotate "SEM n/a" above each such bar: drawn
                        # horizontally the labels collided with each other and
                        # with the n= row, and drawn vertically they ran past
                        # the axes and over the panel title. On this dataset
                        # that is twelve overlapping texts carrying one fact.
                        # A dotted edge says the same thing once per bar at no
                        # cost in ink, and the figure note below says what it
                        # means -- which is the ordinary convention for marking
                        # a subset of bars.
                        ax.bar(x,pi_m,color=_col(trt,ti),width=0.72,
                               edgecolor="black",linewidth=0.9,linestyle=":",
                               hatch=SEX_HATCH.get(sex,""),zorder=3)
                        _gap_sem+=1
                    if "PI_replicates" in row.columns:
                        reps=r0["PI_replicates"]
                        if hasattr(reps,"__iter__"):
                            for rv in reps: ax.scatter(x,rv,color="black",s=14,zorder=5,linewidths=0)
                    if _nc is not None and np.isfinite(_num(r0,_nc)):
                        ax.annotate(f"n={int(_num(r0,_nc))}",(x,0.0),
                                    textcoords="offset points",xytext=(0,-12),
                                    fontsize=FT_ANNOT,color="#333",ha="center")
                x_pos.append(x); x_lbl.append(str(trt)); _gpos.append(x); x+=1
            if _gpos: _groups.append((geno,_gpos))
            x+=0.7
        ax.axhline(0,color="black",linewidth=0.6,linestyle="--",alpha=0.45)
        # The genotype was previously NOWHERE on this figure: the ticks carried
        # the treatment alone, so the blocks of bars could not be told apart.
        _nested_xaxis(ax,x_pos,x_lbl,_groups)
        ax.set_xlabel("Treatment, grouped by genotype",fontsize=FT_LABEL,
                      labelpad=46)
        ax.set_ylabel("Performance Index" if ai==0 else "",fontsize=FT_LABEL)
        ax.set_title(sex,fontsize=FT_LABEL,pad=4)
        ax.set_ylim(-110,115)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.tick_params(length=3)
    patches=[mpatches.Patch(color=_col(t,i),label=t) for i,t in enumerate(trts)]
    fig.legend(handles=patches,loc="lower center",ncol=len(trts),
               fontsize=FT_LEGEND,bbox_to_anchor=(0.5,-0.04),frameon=False)
    if _nc is not None:
        fig.text(0.01,0.01,f"n = independent units ({_nc})",fontsize=FT_ANNOT,
                 color="#555")
    else:
        fig.text(0.01,0.01,"n per bar not available in this table",
                 fontsize=FT_ANNOT,color="#a00")
    if _gap_sem:
        # Says once what used to be written on every affected bar.
        fig.text(0.01,0.035,
                 f"dotted outline: no SEM available for that bar "
                 f"({_gap_sem} of {_drawn}); no error bar is drawn rather than "
                 f"a zero-width one",
                 fontsize=FT_ANNOT,color="#555")
    fig.suptitle(title,fontsize=FT_TITLE,y=1.01)
    fig.tight_layout()
    # tight_layout does not account for artists drawn outside the axes, and the
    # genotype band is one, so the room for it is reserved explicitly.
    fig.subplots_adjust(bottom=0.42)
    _attach_gaps(fig,missing_metric=_gap_metric,missing_sem=_gap_sem,
                 drawn=_drawn,errorbars_drawn=_ebars,
                 n_units_reported=bool(_nc is not None),n_unit_column=_nc)
    return fig,_to_bytes(fig)

# --- 3. AUC heatmap ----------------------------------------------------------
def auc_heatmap(auc_df,sex="Male",metric="AUC_norm",title=None):
    """
    Heatmap Genotype x Treatment, annotated values, 300 DPI.

    COLORMAP: `cividis`, not the `RdYlGn` this used to use. A red-green
    diverging map is the worst available choice for a colour-vision-deficient
    reader -- red and green are precisely the pair deuteranopes and protanopes
    cannot separate -- so on a figure whose entire message is carried by colour
    it makes that message unavailable to about one man in twelve, in a module
    whose first line claims a colour-blind-safe palette. `cividis` was designed
    for that readership (Nunez, Anderton & Renslow 2018, PLoS ONE 13:e0199239)
    and is perceptually uniform, so an equal step in AUC is an equal step in
    apparent lightness.
    """
    df=auc_df[auc_df["Sex"]==sex].copy() if "Sex" in auc_df.columns else auc_df.copy()
    if df.empty or metric not in df.columns:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No AUC data ("+sex+")",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    piv=df.pivot_table(index="Genotype",columns="Treatment",values=metric,aggfunc="mean")
    nr,nc=len(piv.index),len(piv.columns)
    fig,ax=plt.subplots(figsize=(max(4.5,nc*1.5),max(3.0,nr*0.9)))
    vmin=float(piv.min().min()); vmax=float(piv.max().max())
    im=ax.imshow(piv.values,cmap=plt.cm.cividis,aspect="auto",vmin=vmin,vmax=vmax)
    ax.set_xticks(range(nc)); ax.set_xticklabels(piv.columns,rotation=38,ha="right",fontsize=FT_TICK)
    ax.set_yticks(range(nr)); ax.set_yticklabels(piv.index,fontsize=FT_TICK)
    rng=vmax-vmin+1e-9
    for i in range(nr):
        for j in range(nc):
            v=piv.values[i,j]
            if not np.isnan(v):
                # cividis runs dark blue (low) to bright yellow (high), so
                # readable annotation ink is white on the lower half and black
                # on the upper -- the opposite of what a light-middle diverging
                # map needs, and wrong text contrast is itself a legibility bug.
                tc="white" if (v-vmin)/rng<0.55 else "black"
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=FT_ANNOT,color=tc)
    cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.04)
    cb.set_label("Normalised AUC",fontsize=FT_TICK); cb.ax.tick_params(labelsize=FT_TICK)
    ax.set_title(title or "AUC Heatmap -- "+sex,fontsize=FT_TITLE,pad=8)
    ax.set_xlabel("Treatment",fontsize=FT_LABEL); ax.set_ylabel("Genotype",fontsize=FT_LABEL)
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- 4. t50 bar chart --------------------------------------------------------
def t50_plot(t50_df,sexes=None,title="Time to 50% Climbing (t50)"):
    """Bar chart of t50 per group. t50 = seconds until PI >= 50%%."""
    df=t50_df.copy()
    if sexes: df=df[df["Sex"].isin(sexes)]
    if df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No t50 data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    trts=df["Treatment"].unique(); genos=df["Genotype"].unique(); sxs=df["Sex"].unique()
    n_sx=len(sxs)
    fig,axes=plt.subplots(1,n_sx,figsize=(4.2*n_sx,4.5),sharey=True)
    if n_sx==1: axes=[axes]
    for ai,sex in enumerate(sxs):
        ax=axes[ai]; sdf=df[df["Sex"]==sex]
        x=0; x_pos=[]; x_lbl=[]; _groups=[]
        for geno in genos:
            _gpos=[]
            for ti,trt in enumerate(trts):
                row=sdf[(sdf["Genotype"]==geno)&(sdf["Treatment"]==trt)]
                if row.empty: x+=1; continue
                t50_m=float(row["t50_mean"].iloc[0])
                t50_s=float(row["t50_sem"].iloc[0]) if "t50_sem" in row.columns else 0.0
                ax.bar(x,t50_m,yerr=t50_s,color=_col(trt,ti),width=0.72,capsize=3,
                       error_kw={"linewidth":0.8,"ecolor":"#333",
                                 # `ecolor` colours the error bar's LINE only;
                                 # its CAPS fall back to the default colour
                                 # cycle, which put 60 artists at #1f77b4 on
                                 # these figures -- the one colour on them that
                                 # is neither the house style nor
                                 # colour-blind-safe. Measured: `ecolor` alone
                                 # leaves the caps at #1f77b4 and
                                 # `markeredgecolor` does nothing -- `color` is
                                 # the key matplotlib uses for the cap style.
                                 "capthick":0.8,"color":"#333"},
                       edgecolor="black",linewidth=0.5,hatch=SEX_HATCH.get(sex,""))
                x_pos.append(x); x_lbl.append(str(trt)); _gpos.append(x); x+=1
            if _gpos: _groups.append((geno,_gpos))
            x+=0.6
        # Was `geno[:6] + "\n" + trt[:8]` rotated 30 degrees: twelve truncated
        # two-line labels that overlapped into unreadable text on the demo data.
        _nested_xaxis(ax,x_pos,x_lbl,_groups)
        ax.set_xlabel("Treatment, grouped by genotype",fontsize=FT_LABEL,
                      labelpad=46)
        ax.set_ylabel("t50 (s)" if ai==0 else "",fontsize=FT_LABEL)
        ax.set_title(sex,fontsize=FT_LABEL,pad=4); ax.tick_params(length=3)
    fig.suptitle(title,fontsize=FT_TITLE); fig.tight_layout()
    fig.subplots_adjust(bottom=0.42)      # room for the genotype band
    return fig,_to_bytes(fig)

# --- 5. CRI bar chart --------------------------------------------------------
def cri_barplot(cri_df,title="Composite Rescue Index (CRI)"):
    """Vertical bars with PI/AUC/t50 component tick marks. Reference lines at 50 and 80."""
    if cri_df is None or cri_df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No CRI data",ha="center",va="center",transform=ax.transAxes)
        return _attach_gaps(fig),_to_bytes(fig)
    df=cri_df.copy().sort_values("CRI",ascending=False,na_position="last").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=FS_DOUBLE)
    _gap_metric=_drawn=0
    _nc=_n_units(df)
    for i,row in df.iterrows():
        trt=str(row.get("Treatment","")); sex=str(row.get("Sex",""))
        cri_v=_num(row,"CRI")
        if not np.isfinite(cri_v):
            # CRI = 0 means "no rescue", a real result; an absent CRI is not it.
            _mark_gap(ax,i); _gap_metric+=1
            continue
        _drawn+=1
        ax.bar(i,cri_v,color=_col(trt,i),width=0.72,
               edgecolor="black",linewidth=0.5,hatch=SEX_HATCH.get(sex,""),zorder=3)
        if _nc is not None and np.isfinite(_num(row,_nc)):
            ax.annotate(f"n={int(_num(row,_nc))}",(i,0.0),textcoords="offset points",
                        xytext=(0,-12),fontsize=FT_ANNOT,color="#333",ha="center")
        for cc,ms,al in [("CRI_PI",14,0.55),("CRI_AUC",12,0.40),("CRI_t50",10,0.30)]:
            if cc in row and pd.notna(row[cc]):
                ax.plot(i,row[cc],marker="_",markersize=ms,color="#222",alpha=al,zorder=4)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.axhline(50,color="#888",  linewidth=0.5, linestyle=":",  alpha=0.55)
    ax.axhline(80,color="#009E73",linewidth=0.5,linestyle=":",  alpha=0.65)
    ax.text(len(df)-0.5,81,"Strong rescue", fontsize=6,color="#009E73",va="bottom")
    ax.text(len(df)-0.5,51,"Moderate rescue",fontsize=6,color="#888",  va="bottom")
    lbls=[str(row.get("Treatment",""))[:10]+"\n"+str(row.get("Genotype",""))[:6]+"\n"+str(row.get("Sex",""))[:1]
          for _,row in df.iterrows()]
    ax.set_xticks(range(len(df))); ax.set_xticklabels(lbls,fontsize=FT_ANNOT,rotation=35,ha="right")
    ax.set_ylabel("CRI (%)",fontsize=FT_LABEL); ax.set_ylim(-65,120)
    ax.set_title(title,fontsize=FT_TITLE); ax.tick_params(length=3)
    fig.tight_layout()
    _attach_gaps(fig,missing_metric=_gap_metric,drawn=_drawn,
                 n_units_reported=bool(_nc is not None))
    return fig,_to_bytes(fig)
# --- 6. SDQ lollipop plot ----------------------------------------------------
def sdq_plot(sdq_df,title="Sex Dimorphism Quotient (SDQ)"):
    """Lollipop. SDQ=0 identical response, SDQ=1 complete dimorphism."""
    if sdq_df is None or sdq_df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No SDQ data",ha="center",va="center",transform=ax.transAxes)
        return _attach_gaps(fig),_to_bytes(fig)
    df=sdq_df.copy().sort_values("SDQ",ascending=False,na_position="last").reset_index(drop=True)
    fig,ax=plt.subplots(figsize=FS_SINGLE)
    _gap_metric=_drawn=0
    for i,row in df.iterrows():
        color=_col(str(row.get("Treatment","")),i)
        sdq_v=_num(row,"SDQ")
        if not np.isfinite(sdq_v):
            # SDQ = 0.0 is the strongest possible positive statement this plot
            # can make -- "the sexes respond identically". Substituting it for a
            # missing SDQ published that claim for every unmeasured group.
            ax.annotate("not measured",(0.0,i),textcoords="offset points",
                        xytext=(4,0),fontsize=FT_ANNOT,color="#777",va="center")
            ax.scatter(0.0,i,marker="x",color="#999",s=45,zorder=4)
            _gap_metric+=1
            continue
        _drawn+=1
        ax.hlines(i,0,sdq_v,colors=color,linewidth=1.4,alpha=0.55)
        ax.scatter(sdq_v,i,color=color,s=70,zorder=4,edgecolors="black",linewidth=0.5)
    ax.axvline(0.5, color="#D55E00",lw=0.9,ls="--",label="Strong dimorphism (0.5)")
    ax.axvline(0.25,color="#E69F00",lw=0.8,ls=":", label="Moderate (0.25)")
    ylbls=[str(row.get("Genotype",""))[:8]+" | "+str(row.get("Treatment",""))[:12] for _,row in df.iterrows()]
    ax.set_yticks(range(len(df))); ax.set_yticklabels(ylbls,fontsize=FT_ANNOT)
    ax.set_xlabel("SDQ  (0 = no sex difference,  1 = complete dimorphism)",fontsize=FT_LABEL)
    ax.set_title(title,fontsize=FT_TITLE); ax.set_xlim(-0.05,1.1)
    ax.legend(fontsize=FT_LEGEND,loc="lower right",frameon=False)
    fig.tight_layout()
    _attach_gaps(fig,missing_metric=_gap_metric,drawn=_drawn)
    return fig,_to_bytes(fig)

# --- 7. TSW horizontal bar ---------------------------------------------------
def tsw_plot(tsw_df,title="Therapeutic Selectivity Window (TSW)"):
    """Horizontal bars. TSW = CRI_diseased / CRI_wildtype."""
    if tsw_df is None or tsw_df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No TSW data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    df=tsw_df.copy().sort_values("TSW",ascending=False).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.5,max(3.5,len(df)*0.45)))
    for i,row in df.iterrows():
        tsw_v=row.get("TSW",np.nan)
        if pd.isna(tsw_v): continue
        ax.barh(i,float(tsw_v),color=_col(str(row.get("Treatment","")),i),
                height=0.65,edgecolor="black",linewidth=0.5,
                hatch=SEX_HATCH.get(str(row.get("Sex","")),""))
    ax.axvline(0,  color="black",   lw=0.7)
    ax.axvline(1.0,color="#009E73", lw=0.9,ls="--",label="Good selectivity (TSW=1.0)")
    ax.axvline(2.0,color="#005826", lw=0.8,ls=":", label="Excellent (TSW=2.0)")
    ylbls=[str(row.get("Genotype",""))[:7]+" | "+str(row.get("Treatment",""))[:12]+" | "+str(row.get("Sex",""))[:1]
           for _,row in df.iterrows()]
    ax.set_yticks(range(len(df))); ax.set_yticklabels(ylbls,fontsize=FT_ANNOT)
    ax.set_xlabel("TSW  (treatment rescue / wild-type perturbation)",fontsize=FT_LABEL)
    ax.set_title(title,fontsize=FT_TITLE); ax.legend(fontsize=FT_LEGEND,frameon=False)
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- 8. TCS radar chart ------------------------------------------------------
def tcs_radar(axes_labels,radar_data,title="Translational Candidate Score -- Radar"):
    """Polar radar of TCS component contributions per candidate."""
    if not axes_labels or not radar_data:
        fig,ax=plt.subplots(figsize=FS_SQ)
        ax.text(0.5,0.5,"No TCS radar data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    N=len(axes_labels)
    ang=np.linspace(0,2*np.pi,N,endpoint=False).tolist()
    ang_c=ang+ang[:1]
    fig,ax=plt.subplots(figsize=FS_SQ,subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(ang),axes_labels,fontsize=FT_TICK)
    ax.set_ylim(0,100); ax.set_yticks([20,40,60,80,100])
    ax.yaxis.set_ticklabels(["20","40","60","80","100"],fontsize=FT_ANNOT-1)
    ax.yaxis.set_tick_params(labelsize=FT_ANNOT)
    for i,(lbl,vals) in enumerate(radar_data.items()):
        v=list(vals)+[vals[0]]
        ax.plot(ang_c,v,color=WONG[i%len(WONG)],linewidth=1.8,label=lbl)
        ax.fill(ang_c,v,color=WONG[i%len(WONG)],alpha=0.10)
    ax.set_title(title,fontsize=FT_TITLE,pad=18)
    ax.legend(loc="upper right",bbox_to_anchor=(1.38,1.15),fontsize=FT_LEGEND,frameon=False)
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- 9. TCS ranked horizontal bar --------------------------------------------
def tcs_ranked_bar(tcs_df,top_n=15,title="Translational Candidate Score -- Ranked"):
    """Horizontal bar, CI error bars, tier reference lines."""
    if tcs_df is None or tcs_df.empty:
        fig,ax=plt.subplots(figsize=FS_TALL)
        ax.text(0.5,0.5,"No TCS data",ha="center",va="center",transform=ax.transAxes)
        return _attach_gaps(fig),_to_bytes(fig)
    df=tcs_df.nsmallest(top_n,"TCS_rank").sort_values("TCS",ascending=True,
                                                      na_position="first")
    fig,ax=plt.subplots(figsize=(8.5,max(4.0,len(df)*0.46)))
    _gap_metric=_gap_ci=_drawn=_ebars=0
    for i,(_,row) in enumerate(df.iterrows()):
        tcs_v=_num(row,"TCS")
        if not np.isfinite(tcs_v):
            # `row.get("TCS", 0)` scored an absent candidate zero, which on this
            # 0-100 scale is the worst possible candidate rather than an unknown.
            _mark_gap(ax,i,horizontal=True); _gap_metric+=1
            continue
        _drawn+=1
        ci_lo=_num(row,"TCS_CI_lower"); ci_hi=_num(row,"TCS_CI_upper")
        ax.barh(i,tcs_v,color=_col(str(row.get("Treatment","")),i),height=0.65,
                edgecolor="black",linewidth=0.4,hatch=SEX_HATCH.get(str(row.get("Sex","")),""))
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax.errorbar(tcs_v,i,xerr=[[max(tcs_v-ci_lo,0.0)],[max(ci_hi-tcs_v,0.0)]],
                        fmt="none",color="#333",capsize=3,linewidth=0.8,zorder=5)
            _ebars+=1
        else:
            # The interval columns were deliberately removed upstream (the old
            # TCS_CI was score +/- 10*std of five quantities in different units,
            # with no coverage property). Defaulting the bounds to the estimate
            # drew a zero-width interval on every bar: a precision claim where
            # there is no interval at all.
            _gap_ci+=1
        if np.isfinite(_num(row,"n_components")):
            ax.annotate(f"{int(_num(row,'n_components'))}/5 components",
                        (tcs_v,i),textcoords="offset points",xytext=(4,0),
                        fontsize=FT_ANNOT,color="#555",va="center")
    for thresh,lbl,col in [(80,"Tier 1","#005826"),(60,"Tier 2","#6b8e23"),(40,"Tier 3","#b8860b")][::-1]:
        ax.axvline(thresh,lw=0.7,ls="--",color=col,alpha=0.65)
        ax.text(thresh+0.5,-0.9,lbl,fontsize=6,color=col,va="top")
    ylbls=[str(row.get("Treatment",""))[:12]+" / "+str(row.get("Genotype",""))[:8]+" / "+str(row.get("Sex",""))[:1]
           for _,row in df.iterrows()]
    ax.set_yticks(range(len(df))); ax.set_yticklabels(ylbls,fontsize=FT_ANNOT)
    ax.set_xlabel("TCS (0-100)",fontsize=FT_LABEL); ax.set_xlim(0,105)
    ax.set_title(title,fontsize=FT_TITLE); ax.tick_params(length=3)
    if _gap_ci:
        fig.text(0.01,0.01,f"{_gap_ci} bar(s) have no interval: none is computed "
                           f"for the TCS, so none is drawn",fontsize=FT_ANNOT,color="#a00")
    fig.tight_layout()
    _attach_gaps(fig,missing_metric=_gap_metric,missing_ci=_gap_ci,
                 drawn=_drawn,errorbars_drawn=_ebars)
    return fig,_to_bytes(fig)

# --- 10. MDR decay curves ----------------------------------------------------
def mdr_decay_curves(pi_df,mdr_df,sexes=None,title="Motor Decline Rate -- Exponential Decay Fits",
                     time_col="Age",xlabel="Age (days)"):
    """Scatter + fitted PI(t)=PI0*exp(-k*t) overlay over the decline axis
    (age, by default). R-squared annotated."""
    df=pi_df.copy()
    # the decline axis column ('Age'); fall back to 'Time' for legacy callers
    if time_col not in df.columns and "Time" in df.columns:
        time_col="Time"; xlabel="Time (s)"
    if sexes:
        # accept a single sex string or a list; .isin() requires list-like
        sex_list=[sexes] if isinstance(sexes,str) else list(sexes)
        df=df[df["Sex"].isin(sex_list)]
    if "PI_mean" not in df.columns and "PI" in df.columns: df=df.rename(columns={"PI":"PI_mean"})
    sxs=df["Sex"].unique() if "Sex" in df.columns else ["All"]
    n_sx=len(sxs)
    fig,axes=plt.subplots(1,n_sx,figsize=(5.5*n_sx,4.5),sharey=True)
    if n_sx==1: axes=[axes]
    for ai,sex in enumerate(sxs):
        ax=axes[ai]
        sdf=df[df["Sex"]==sex] if "Sex" in df.columns else df
        for idx,((geno,trt),grp) in enumerate(sdf.groupby(["Genotype","Treatment"],sort=False)):
            grp=grp.sort_values(time_col)
            x=grp[time_col].values.astype(float)
            y=grp["PI_mean"].values.astype(float)
            color=_col(trt,idx)
            mk=GENOTYPE_MARKERS.get(geno,"o")
            ax.scatter(x,y,color=color,marker=mk,s=28,
                       edgecolors="black",linewidth=0.4,label=geno+" | "+trt,zorder=3)
            if mdr_df is not None and not mdr_df.empty:
                mrow=mdr_df.loc[(mdr_df["Genotype"]==geno)&(mdr_df["Treatment"]==trt)&(mdr_df["Sex"]==sex)]
                if not mrow.empty:
                    PI0=float(mrow["PI0"].iloc[0]); k=float(mrow["k"].iloc[0])
                    r2=float(mrow["R2"].iloc[0]) if "R2" in mrow.columns else float("nan")
                    if pd.notna(PI0) and pd.notna(k) and k>0:
                        t_fit=np.linspace(x.min(),x.max(),200)
                        y_fit=_exp_decay_local(t_fit,PI0,k)
                        r2_s=(" R2="+str(round(r2,2))) if pd.notna(r2) else ""
                        ax.plot(t_fit,y_fit,color=color,linewidth=1.6,linestyle="--",alpha=0.82,
                                label="Fit k="+str(round(k,4))+r2_s)
        ax.set_xlabel(xlabel,fontsize=FT_LABEL)
        ax.set_ylabel("PI (mean)" if ai==0 else "",fontsize=FT_LABEL)
        ax.set_title(sex,fontsize=FT_LABEL,pad=4)
        # The legend carries one entry per group PLUS one per fit, so on a
        # six-group panel it is twelve lines deep and inside the axes it sits
        # on top of the data it describes. Below the panel it covers nothing.
        ax.legend(fontsize=FT_LEGEND-1,frameon=False,ncol=2,
                  loc="upper center",bbox_to_anchor=(0.5,-0.16))
        ax.tick_params(length=3)
    fig.suptitle(title,fontsize=FT_TITLE); fig.tight_layout()
    fig.subplots_adjust(bottom=0.38)      # room for the legend below the axes
    return fig,_to_bytes(fig)

# --- 11. Trajectory dendrogram -----------------------------------------------
def trajectory_dendrogram(cluster_result,title="DTW Trajectory Clustering -- Dendrogram"):
    """Hierarchical dendrogram of DTW distances."""
    Z=cluster_result.get("linkage_matrix",np.array([]))
    labels=cluster_result.get("labels",[])
    if Z is None or len(Z)==0:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No clustering data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*0.7),5))
    short=[str(l).replace(" | ","\n")[:28] for l in labels]
    dendrogram(Z,labels=short,ax=ax,leaf_rotation=45,leaf_font_size=FT_ANNOT,
               color_threshold=0.7*float(Z[:,2].max()) if len(Z)>0 else 0)
    ax.set_ylabel("DTW Distance",fontsize=FT_LABEL)
    ax.set_xlabel("Group  (Genotype | Treatment | Sex)",fontsize=FT_LABEL)
    ax.set_title(title,fontsize=FT_TITLE); ax.tick_params(length=3)
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- 12. Trajectory PCA / UMAP -----------------------------------------------
def trajectory_pca(cluster_result,use_umap=False,title=None):
    """2-D scatter of PCA (or UMAP) trajectory embedding, coloured by cluster."""
    key="umap_embedding" if use_umap else "pca_embedding"
    emb=cluster_result.get(key,pd.DataFrame())
    # Register N117. When UMAP was requested but unavailable -- `umap-learn` is
    # in neither requirements file and `trajectory_cluster.py` swallows the
    # ImportError -- this silently substituted the PCA embedding and relabelled
    # the axes PC1/PC2. A reader who asked for UMAP got PCA with no indication
    # anywhere on the figure. The substitution still happens, because a figure
    # is better than no figure, but it now says so.
    _substituted = False
    if emb is None or (hasattr(emb,"empty") and emb.empty):
        if use_umap:
            emb=cluster_result.get("pca_embedding",pd.DataFrame())
            use_umap=False
            _substituted=True
    if emb is None or (hasattr(emb,"empty") and emb.empty):
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No embedding data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    pfx="UMAP" if use_umap else "PC"
    xcol="UMAP1" if use_umap else "PC1"
    ycol="UMAP2" if use_umap else "PC2"
    fig,ax=plt.subplots(figsize=FS_SINGLE)
    for cid in sorted(emb["Cluster"].unique()):
        cg=emb[emb["Cluster"]==cid]
        color=WONG[(int(cid)-1)%len(WONG)]
        ax.scatter(cg[xcol],cg[ycol],color=color,s=62,
                   edgecolors="black",linewidth=0.5,label="Cluster "+str(cid),zorder=3)
        for _,row in cg.iterrows():
            lbl=str(row.get("Genotype",""))[:5]+"|"+str(row.get("Treatment",""))[:5]+"|"+str(row.get("Sex",""))[:1]
            ax.annotate(lbl,(row[xcol],row[ycol]),fontsize=5.5,alpha=0.72,
                        xytext=(3,3),textcoords="offset points")
    ax.set_xlabel(pfx+"1",fontsize=FT_LABEL); ax.set_ylabel(pfx+"2",fontsize=FT_LABEL)
    _ttl = title or "Trajectory Embedding ("+pfx+")"
    if _substituted:
        _ttl += "  [UMAP requested; umap-learn not installed, PCA shown]"
    ax.set_title(_ttl,fontsize=FT_TITLE)
    ax.legend(fontsize=FT_LEGEND,frameon=False); ax.tick_params(length=3)
    if _substituted:
        # Register N117: the substitution must be legible on the figure itself,
        # not merely in a checkbox caption the reader may never see.
        fig.text(0.01,0.01,
                 "UMAP was requested but `umap-learn` is not installed; the PCA "
                 "embedding is shown and the axes are PCA axes.",
                 fontsize=5.5,style="italic",color="#555555",ha="left")
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- 13. Sex comparison connected dot plot -----------------------------------
def sex_comparison_plot(pi_df,treatments=None,title="Male vs Female Performance Index"):
    """Connected dots: circle=Male, square=Female."""
    df=pi_df.copy()
    if treatments: df=df[df["Treatment"].isin(treatments)]
    if df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    males  =df[df["Sex"]=="Male"  ].set_index(["Genotype","Treatment"])["PI_mean"].rename("Male")
    females=df[df["Sex"]=="Female"].set_index(["Genotype","Treatment"])["PI_mean"].rename("Female")
    comp=pd.concat([males,females],axis=1).dropna()
    if comp.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"Insufficient data for both sexes",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    fig,ax=plt.subplots(figsize=FS_SINGLE)
    for i,(idx,row) in enumerate(comp.iterrows()):
        geno,trt=idx; color=_col(trt,i)
        ax.plot([row["Male"],row["Female"]],[i,i],color=color,linewidth=1.5,alpha=0.7)
        ax.scatter(row["Male"],  i,color=color,marker="o",s=52,edgecolors="black",linewidth=0.5,zorder=4)
        ax.scatter(row["Female"],i,color=color,marker="s",s=52,edgecolors="black",linewidth=0.5,zorder=4)
    ax.axvline(0,color="black",lw=0.6,ls="--",alpha=0.45)
    ylbls=[str(g)[:7]+" | "+str(t)[:10] for g,t in comp.index]
    ax.set_yticks(range(len(comp))); ax.set_yticklabels(ylbls,fontsize=FT_ANNOT)
    ax.set_xlabel("Performance Index",fontsize=FT_LABEL); ax.set_title(title,fontsize=FT_TITLE)
    mh=Line2D([0],[0],marker="o",color="#555",ls="",markersize=6,label="Male")
    fh=Line2D([0],[0],marker="s",color="#555",ls="",markersize=6,label="Female")
    ax.legend(handles=[mh,fh],fontsize=FT_LEGEND,frameon=False); ax.tick_params(length=3)
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- 14. Bootstrap CI forest plot -------------------------------------------
def bootstrap_ci_plot(ci_df,metric="Mean",title="Bootstrap 95% Confidence Intervals"):
    """Forest plot of bootstrap CIs. Used on Statistical Analysis page."""
    if ci_df is None or ci_df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No CI data",ha="center",va="center",transform=ax.transAxes)
        return _attach_gaps(fig),_to_bytes(fig)
    df=ci_df.copy().reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7.0,max(3.5,len(df)*0.5)))
    _gap_metric=_gap_ci=_drawn=_ebars=0
    for i,row in df.iterrows():
        grp=str(row.get("Treatment",row.get("Group","Group "+str(i))))
        mean=_num(row,metric)
        if not np.isfinite(mean):
            mean=_num(row,"Mean")
        if not np.isfinite(mean):
            # `row.get(metric, row.get("Mean", 0))` put an unmeasured group at
            # zero -- on a forest plot of differences, exactly the position that
            # reads as "no effect".
            ax.annotate("not measured",(0.0,i),textcoords="offset points",
                        xytext=(4,0),fontsize=FT_ANNOT,color="#777",va="center")
            ax.scatter(0.0,i,marker="x",color="#999",s=45,zorder=4)
            _gap_metric+=1
            continue
        _drawn+=1
        lo=_num(row,"CI95_lo"); hi=_num(row,"CI95_hi")
        color=_col(grp,i)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo,hi],[i,i],color=color,linewidth=2.2,solid_capstyle="round")
            ax.scatter([lo,hi],[i,i],color=color,marker="|",s=40,linewidths=1.5,zorder=4)
            _ebars+=1
        else:
            # Defaulting the bounds to the estimate drew a point interval, i.e.
            # a bootstrap CI of zero width.
            ax.annotate("CI n/a",(mean,i),textcoords="offset points",
                        xytext=(6,0),fontsize=FT_ANNOT,color="#a00",va="center")
            _gap_ci+=1
        ax.scatter(mean,i,color=color,s=60,zorder=5,edgecolors="black",linewidth=0.5)
    ax.axvline(0,color="black",lw=0.6,ls="--",alpha=0.35)
    ylbls=[str(row.get("Treatment",row.get("Group","G"+str(i)))) for i,row in df.iterrows()]
    ax.set_yticks(range(len(df))); ax.set_yticklabels(ylbls,fontsize=FT_TICK)
    ax.set_xlabel(metric+" (95% bootstrap CI)",fontsize=FT_LABEL)
    ax.set_title(title,fontsize=FT_TITLE); ax.tick_params(length=3)
    fig.tight_layout()
    _attach_gaps(fig,missing_metric=_gap_metric,missing_ci=_gap_ci,
                 drawn=_drawn,errorbars_drawn=_ebars)
    return fig,_to_bytes(fig)

# --- 15. Pairwise p-value dot matrix ----------------------------------------
def pairwise_dot_plot(pairwise_df,title="Pairwise Comparisons (BH-FDR corrected)"):
    """Dot matrix of adjusted p-values. Stars in cells. Replaces raw JSON display."""
    if pairwise_df is None or pairwise_df.empty:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"No pairwise data",ha="center",va="center",transform=ax.transAxes)
        return fig,_to_bytes(fig)
    df=pairwise_df.copy()
    if "group1" in df.columns and "group2" in df.columns and "p-adj" in df.columns:
        groups=sorted(set(df["group1"].tolist()+df["group2"].tolist()))
        mat=pd.DataFrame(np.nan,index=groups,columns=groups)
        for _,row in df.iterrows():
            p=float(row["p-adj"])
            mat.loc[row["group1"],row["group2"]]=p
            mat.loc[row["group2"],row["group1"]]=p
        fig,ax=plt.subplots(figsize=(max(4.5,len(groups)*1.1),max(3.5,len(groups)*0.9)))
        im=ax.imshow(mat.values.astype(float),cmap=plt.cm.cividis_r,vmin=0,vmax=0.05,aspect="auto")
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups,rotation=40,ha="right",fontsize=FT_TICK)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups,fontsize=FT_TICK)
        for i in range(len(groups)):
            for j in range(len(groups)):
                v=mat.values[i,j]
                if not np.isnan(v): ax.text(j,i,_pval_stars(v),ha="center",va="center",fontsize=FT_ANNOT+1)
        fig.colorbar(im,ax=ax,fraction=0.03,pad=0.04).set_label("p-adj (BH-FDR)",fontsize=FT_TICK)
        ax.set_title(title,fontsize=FT_TITLE)
    else:
        fig,ax=plt.subplots(figsize=FS_SINGLE)
        ax.text(0.5,0.5,"Unexpected column format",ha="center",va="center",transform=ax.transAxes)
    fig.tight_layout(); return fig,_to_bytes(fig)

# --- export_all_figures -------------------------------------------------------
def export_all_figures(results,out_dir="drophenix_figures",fmt="png"):
    """Batch export. Flexible key lookup -- accepts pi_df, PI_df, or pi."""
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); saved=[]
    def _save(fig,name):
        p=out/(name+"."+fmt)
        fig.savefig(str(p),dpi=DPI,bbox_inches="tight",facecolor="white")
        plt.close(fig); saved.append(str(p))
    agg  =_get_key(results,"agg_df","norm_df","agg")
    pi   =_get_key(results,"pi_df","PI_df","pi")
    auc  =_get_key(results,"auc_df","AUC_df","auc")
    t50  =_get_key(results,"t50_df","T50_df","t50")
    cri  =_get_key(results,"cri_df","CRI_df","cri")
    sdq  =_get_key(results,"sdq_df","SDQ_df","sdq")
    tsw  =_get_key(results,"tsw_df","TSW_df","tsw")
    tcs_r=_get_key(results,"tcs_result","tcs")
    mdr  =_get_key(results,"mdr_df","MDR_df","mdr")
    clust=_get_key(results,"cluster_result","cluster")
    pairs=[(agg,lambda:climbing_curves(agg),"01_climbing_curves"),
           (pi, lambda:pi_barplot(pi),       "02_pi_barplot"),
           (auc,lambda:auc_heatmap(auc,sex="Male"),  "03_auc_heatmap_male"),
           (auc,lambda:auc_heatmap(auc,sex="Female"),"03_auc_heatmap_female"),
           (t50,lambda:t50_plot(t50),        "04_t50_plot"),
           (cri,lambda:cri_barplot(cri),     "05_cri_barplot"),
           (sdq,lambda:sdq_plot(sdq),        "06_sdq_plot"),
           (tsw,lambda:tsw_plot(tsw),        "07_tsw_plot"),
           (pi, lambda:sex_comparison_plot(pi),"13_sex_comparison")]
    for data,fn,name in pairs:
        if data is not None:
            try: fig,_=fn(); _save(fig,name)
            except Exception as exc: warnings.warn(name+" export failed: "+str(exc))
    if tcs_r is not None:
        try:
            full=tcs_r.get("tcs_full") if isinstance(tcs_r,dict) else tcs_r
            if full is not None and not full.empty: fig,_=tcs_ranked_bar(full); _save(fig,"08_tcs_ranked_bar")
            ra=tcs_r.get("radar_axes") if isinstance(tcs_r,dict) else None
            rd=tcs_r.get("radar_data") if isinstance(tcs_r,dict) else None
            if ra and rd: fig,_=tcs_radar(ra,rd); _save(fig,"09_tcs_radar")
        except Exception as exc: warnings.warn("TCS export failed: "+str(exc))
    if mdr is not None and pi is not None:
        try: fig,_=mdr_decay_curves(pi,mdr); _save(fig,"10_mdr_decay_curves")
        except Exception as exc: warnings.warn("MDR export failed: "+str(exc))
    if clust is not None:
        try:
            Z=clust.get("linkage_matrix") if isinstance(clust,dict) else None
            if Z is not None and len(Z)>0: fig,_=trajectory_dendrogram(clust); _save(fig,"11_dendrogram")
            pe=clust.get("pca_embedding") if isinstance(clust,dict) else None
            if pe is not None and not pe.empty: fig,_=trajectory_pca(clust); _save(fig,"12_trajectory_pca")
        except Exception as exc: warnings.warn("Cluster export failed: "+str(exc))
    return saved

# --- Compatibility aliases (FIXED: both were = tsw_plot in old version) -------
plot_climbing_index  = climbing_curves    # was tsw_plot -- WRONG
plot_pi_curves       = climbing_curves    # was tsw_plot -- WRONG
plot_pi_barplot      = pi_barplot
plot_auc_heatmap     = auc_heatmap
plot_t50             = t50_plot
plot_cri             = cri_barplot
plot_sdq             = sdq_plot
plot_tsw             = tsw_plot
plot_tcs_radar       = tcs_radar
plot_tcs_bar         = tcs_ranked_bar
plot_mdr             = mdr_decay_curves
plot_dendrogram      = trajectory_dendrogram
plot_pca             = trajectory_pca
plot_sex_comparison  = sex_comparison_plot
plot_bootstrap_ci    = bootstrap_ci_plot
plot_pairwise        = pairwise_dot_plot
