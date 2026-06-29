"""
make_figures.py — publication figures for the DroPhenix manuscript (NAR).

Professional, colour-blind-safe (Okabe-Ito palette), 300 dpi (PNG + PDF), with
bold panel letters, axis labels + units, uncluttered legends. All panels are
generated live from the validated pipeline on benchmark / literature-calibrated
data.

Run:  python manuscript/make_figures.py  ->  manuscript/figures/*.png|pdf
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)

# Okabe-Ito colour-blind-safe palette
OI = {"blue": "#0072B2", "sky": "#56B4E9", "orange": "#E69F00", "vermilion": "#D55E00",
      "green": "#009E73", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
      "grey": "#999999"}
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white",
})

def _panel(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left")

# Single, consistent figure-title style for ALL figures (same font/size/weight/align)
SUP_KW = dict(fontsize=13, fontweight="bold", ha="left", x=0.02)

def _subtitle(fig, text, y):
    fig.text(0.02, y, text, ha="left", fontsize=9.5, color="#5A5A5A")

# Freireich 1963 6-MP data
MP_T=[10,7,32,23,22,6,16,34,32,25,11,20,19,6,17,35,6,13,9,6,10]
MP_E=[1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0]
PL_T=[1,22,3,12,8,17,2,11,8,12,2,5,4,15,8,23,5,11,4,1,8]; PL_E=[1]*21
GMAP={"Oregon-WT|Vehicle":OI["blue"],"Oregon-WT|Drug":OI["sky"],
      "TDP43|Vehicle":OI["vermilion"],"TDP43|Drug":OI["green"]}


def fig_validation():
    from modules import survival as SV, dams as DM, ymaze as YM, synthetic as SYN
    from modules.data_parser import load_data
    from modules.normalizer import compute_all_metrics
    from modules.motor_decline import endpoint_pi_by_age, compute_mdr_table
    from modules import drug_ranking as DR

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.8)); ax = ax.ravel()

    # (A) Kaplan-Meier vs lifelines
    a = ax[0]
    ev = SV.to_event_table(pd.DataFrame({"Genotype":["Lk"]*42,
        "Treatment":["6-MP"]*21+["Placebo"]*21,"Sex":["P"]*42,"Time":MP_T+PL_T,"Event":MP_E+PL_E}))
    km = SV.kaplan_meier(ev)
    for grp,col,lab in [("Lk | 6-MP | P",OI["green"],"6-MP (treated)"),
                        ("Lk | Placebo | P",OI["vermilion"],"Placebo")]:
        c=km[grp]; t=np.concatenate(([0],c["Time"])); s=np.concatenate(([1],c["Survival"]))
        a.step(t,s,where="post",color=col,lw=2.4,label=lab)
    try:
        from lifelines import KaplanMeierFitter
        kmf=KaplanMeierFitter()
        for tt,ee in [(MP_T,MP_E),(PL_T,PL_E)]:
            kmf.fit(tt,ee); sf=kmf.survival_function_
            a.step(sf.index,sf.iloc[:,0],where="post",color=OI["black"],lw=1.0,ls=(0,(1,1)),alpha=0.85)
        a.plot([],[],color=OI["black"],ls=(0,(1,1)),label="lifelines (overlay)")
    except Exception: pass
    a.set_xlabel("Time (weeks)"); a.set_ylabel("Survival probability  S(t)"); a.set_ylim(0,1.03)
    a.set_title("Kaplan–Meier survival\n(Freireich 1963; vs lifelines)"); a.legend(frameon=False,loc="upper right")
    _panel(a,"A")

    # (B) Climbing kinetics
    b=ax[1]
    df,_,_=load_data(SYN.generate_climbing()); agg=compute_all_metrics(df)["aggregated"]
    agg["g"]=agg["Genotype"]+"|"+agg["Treatment"]
    for g,sub in agg[agg["Sex"]=="Male"].groupby("g"):
        sub=sub.sort_values("Time")
        b.plot(sub["Time"],2*sub["Pct_mean"]-100,"-o",ms=5,lw=2,color=GMAP.get(g,OI["grey"]),
               label=g.replace("|"," · "))
    b.axhline(0,color=OI["grey"],lw=0.8,ls=":")
    b.set_xlabel("Time after tap (s)"); b.set_ylabel("Performance Index  (2·%climbed−100)")
    b.set_title("Climbing kinetics\n(within-assay, seconds)"); b.legend(frameon=False,loc="lower right")
    _panel(b,"B")

    # (C) MDR over age
    c=ax[2]
    dfa,_,_=load_data(SYN.generate_climbing_aging()); ap=endpoint_pi_by_age(dfa)
    mdr=compute_mdr_table(ap,time_col="Age")
    for (g,t),sub in ap[ap.Sex=="Male"].groupby(["Genotype","Treatment"]):
        sub=sub.sort_values("Age"); col=GMAP.get(f"{g}|{t}",OI["grey"])
        c.plot(sub["Age"],sub["PI_mean"],"o",ms=6,color=col)
        mr=mdr[(mdr.Genotype==g)&(mdr.Treatment==t)&(mdr.Sex=="Male")&(mdr.fit_success)]
        if not mr.empty and pd.notna(mr["k"].iloc[0]):
            PI0=mr["PI0"].iloc[0]; k=mr["k"].iloc[0]; xx=np.linspace(5,45,100)
            c.plot(xx,PI0*np.exp(-k*xx),"--",color=col,lw=2,
                   label=f"{g}·{t}  k={k:.3f} d⁻¹")
    c.set_xlabel("Age (days)"); c.set_ylabel("Performance Index")
    c.set_title("Motor Decline Rate\nPI(age)=PI₀·e^(−k·age)"); c.legend(frameon=False,fontsize=8,loc="upper right")
    _panel(c,"C")

    # (D) periodogram
    d=ax[3]
    rng=np.random.default_rng(0); binm=30; n=int(8*1440/binm); tt=np.arange(n)*binm
    y=rng.poisson(3*(1+np.sin(2*np.pi*tt/(24*60)))**2).astype(float)
    cp=DM.chi_square_periodogram(y,binm,16,32)
    d.plot(cp["periods_h"],cp["Qp"],color=OI["blue"],lw=2,label="Q_p statistic")
    d.plot(cp["periods_h"],cp["threshold"],color=OI["vermilion"],lw=1.4,ls="--",label="p<0.05 (Bonferroni)")
    d.axvline(cp["peak_period_h"],color=OI["green"],lw=1.6,ls=":",label=f"peak = {cp['peak_period_h']:.2f} h")
    d.set_xlabel("Test period (h)"); d.set_ylabel("Q_p  (χ² periodogram)")
    d.set_title("Circadian period\n(Sokolove–Bushell 1978)"); d.legend(frameon=False,loc="upper right")
    _panel(d,"D")

    # (E) Y-maze handedness
    e=ax[4]
    pf=YM.per_fly_metrics(YM.load_ymaze(SYN.generate_ymaze()))
    e.hist(pf["turn_bias"].dropna(),bins=20,range=(0,1),color=OI["sky"],edgecolor="white",zorder=2)
    e.axvspan(0.3,0.7,color=OI["grey"],alpha=0.15,zorder=1,label="central [0.3–0.7]")
    e.axvline(pf["turn_bias"].mean(),color=OI["vermilion"],lw=2,ls="--",
              label=f"mean = {pf['turn_bias'].mean():.2f}")
    e.set_xlabel("Turn bias,  P(right)"); e.set_ylabel("Number of flies")
    e.set_title("Y-maze handedness\n(Buchanan/de Bivort 2015)"); e.legend(frameon=False,loc="upper right")
    _panel(e,"E")

    # (F) drug ranking
    f=ax[5]
    res=DR.rank_compounds(SYN.generate_screen()); comp=res["composite"]
    cols=[OI["green"] if p<0.05 else OI["grey"] for p in comp["p_value"]]
    f.bar(comp["drug"],comp["Therapeutic_Score"],color=cols,
          yerr=comp["ci_hi"]-comp["Therapeutic_Score"],capsize=4,edgecolor=OI["black"],linewidth=0.6)
    f.axhline(0,color=OI["black"],lw=0.8)
    f.set_ylabel("Therapeutic Score  (SMD, benefit)"); f.set_xlabel("Compound")
    f.set_title("Cross-assay drug ranking\n(green: p<0.05)")
    for i,r in comp.iterrows():
        f.text(i,r["Therapeutic_Score"]+0.03,f"rank {r['Rank']}",ha="center",va="bottom",fontsize=9)
    _panel(f,"F")

    fig.suptitle("Figure 1.  DroPhenix multi-assay analysis and validation", y=1.005, **SUP_KW)
    fig.tight_layout(rect=[0,0,1,0.99])
    for ext in ("png","pdf"): fig.savefig(OUT/f"Figure1_validation.{ext}",bbox_inches="tight")
    plt.close(fig); print("wrote Figure1_validation.png/pdf")


def fig_innovation():
    """The core innovation: unequal-N cross-strain compound comparison."""
    import importlib
    dm = importlib.import_module("validation.demo_unequal_n")
    from modules.drug_ranking import hedges_g, inverse_variance_pool
    df, pooled, naive = dm.worked_example()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    # (A) Forest plot
    a = ax[0]
    df2 = df.sort_values("n_drug")
    y = np.arange(len(df2))[::-1]
    w = 1.0/df2["var"].values; sizes = 60 + 600*w/w.sum()
    for yi,(_,r),sz in zip(y, df2.iterrows(), sizes):
        lo,hi = r["g"]-1.96*r["se"], r["g"]+1.96*r["se"]
        a.plot([lo,hi],[yi,yi],color=OI["blue"],lw=2)
        a.scatter([r["g"]],[yi],s=sz,color=OI["blue"],edgecolors=OI["black"],linewidth=0.6,zorder=3)
        a.text(hi+0.12, yi, f"n={int(r['n_drug'])} · w={r['weight_pct']:.0f}%",
               va="center", fontsize=9)
    # pooled diamond
    yp=-1.2; lo,hi=pooled["ci_lo"],pooled["ci_hi"]; ctr=pooled["random"]
    a.add_patch(Polygon([[lo,yp],[ctr,yp+0.28],[hi,yp],[ctr,yp-0.28]],
                        closed=True,facecolor=OI["green"],edgecolor=OI["black"],zorder=3))
    a.text(hi+0.12, yp, "pooled (DL)", va="center", fontsize=9, fontweight="bold")
    a.axvline(0,color=OI["grey"],lw=1,ls=":")
    a.axvline(naive,color=OI["vermilion"],lw=1.4,ls="--",label=f"naïve mean = {naive:.2f}")
    a.axvline(pooled["random"],color=OI["green"],lw=1.2,ls="-",alpha=0.5)
    a.set_yticks(list(y)+[yp]); a.set_yticklabels(list(df2["Strain"])+["Pooled"])
    a.set_ylim(yp-0.8, y.max()+0.8)
    a.set_xlabel("Drug effect, Hedges' g  (SD units; →benefit)")
    a.set_title("Inverse-variance forest plot\n(unequal-N strains; marker area ∝ weight)")
    a.legend(frameon=False, loc="lower right")
    _panel(a,"A")

    # (B) Simulation: naive vs IV estimator distributions
    b = ax[1]
    rng=np.random.default_rng(7); true_g=0.7; ns=(3,7,5,4,9)
    naive_e, iv_e = [], []
    for _ in range(3000):
        gs,vs=[],[]
        for n in ns:
            e=hedges_g(rng.normal(0,1,n), rng.normal(true_g,1,n))
            if np.isfinite(e["g"]): gs.append(e["g"]); vs.append(e["var"])
        if len(gs)<2: continue
        naive_e.append(np.mean(gs)); iv_e.append(inverse_variance_pool(np.array(gs),np.array(vs))["fixed"])
    naive_e=np.array(naive_e); iv_e=np.array(iv_e)
    bins=np.linspace(-0.6,2.0,46)
    b.hist(naive_e,bins=bins,color=OI["vermilion"],alpha=0.55,label=f"naïve mean (RMSE {np.sqrt(np.mean((naive_e-true_g)**2)):.3f})")
    b.hist(iv_e,bins=bins,color=OI["green"],alpha=0.6,label=f"inverse-variance (RMSE {np.sqrt(np.mean((iv_e-true_g)**2)):.3f})")
    b.axvline(true_g,color=OI["black"],lw=2,label=f"true g = {true_g}")
    b.set_xlabel("Estimated pooled drug effect (Hedges' g)"); b.set_ylabel("Count (of 3000 simulations)")
    b.set_title("Estimator efficiency under unequal N\n(strain sizes 3,7,5,4,9)")
    b.legend(frameon=False, loc="upper right")
    _panel(b,"B")

    fig.suptitle("Figure 2.  Core innovation: actuarial cross-strain compound comparison with unequal N",
                 y=1.02, **SUP_KW)
    fig.tight_layout(rect=[0,0,1,0.99])
    for ext in ("png","pdf"): fig.savefig(OUT/f"Figure2_innovation.{ext}",bbox_inches="tight")
    plt.close(fig); print("wrote Figure2_innovation.png/pdf")


def fig_agreement():
    import importlib
    cs = importlib.import_module("validation.comparison_study")
    cs.ROWS.clear()
    cs.compare_survival(); cs.compare_stats(); cs.compare_meta(); cs.compare_circadian()
    cs.compare_R(); cs.compare_pyboat()
    rows=[r for r in cs.ROWS if r[6] in ("yes","NO")]
    dp=np.array([float(r[2]) for r in rows]); ref=np.array([float(r[4]) for r in rows])
    fig,a=plt.subplots(figsize=(6.6,5.6))
    lim=[min(dp.min(),ref.min())-0.5,max(dp.max(),ref.max())+0.5]
    a.plot(lim,lim,color=OI["grey"],ls="--",lw=1,label="identity (perfect agreement)")
    a.scatter(ref,dp,s=55,color=OI["blue"],edgecolors=OI["black"],linewidth=0.5,zorder=3)
    r=np.corrcoef(ref,dp)[0,1]
    a.set_xlabel("Reference-tool value"); a.set_ylabel("DroPhenix value")
    a.set_title("")  # figure title carried by the suptitle, like Figs 1–2
    a.legend(frameon=False,loc="upper left")
    fig.suptitle("Figure 3.  Agreement with established tools", y=0.99, **SUP_KW)
    _subtitle(fig, f"n = {len(rows)} metrics; Pearson r = {r:.4f}; vs R, lifelines, "
                   f"statsmodels, scipy, pingouin, astropy, pyBOAT", y=0.945)
    fig.tight_layout(rect=[0,0,1,0.93])
    for ext in ("png","pdf"): fig.savefig(OUT/f"Figure3_tool_agreement.{ext}",bbox_inches="tight")
    plt.close(fig); print(f"wrote Figure3_tool_agreement.png/pdf (n={len(rows)}, r={r:.4f})")


if __name__ == "__main__":
    fig_validation(); fig_innovation(); fig_agreement()
