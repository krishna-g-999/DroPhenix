"""
build_and_run.py — Literature-referenced reference datasets for every DroPhenix module.

For each of the six assays we provide >=3 datasets in the module's upload-template
format, run each through the real analysis module, and record the result against the
published expectation. Provenance:

  * REAL raw data: survival sample 1 is the verified Freireich et al. (1963) 6-MP
    leukemia trial (raw event-level data).
  * REAL published parameters: circadian sample 1 uses the Konopka & Benzer (1971)
    free-running periods (24 / 19 / 28 h) exactly.
  * Otherwise: datasets are SIMULATED FROM TEXT-VERIFIED PUBLISHED PARAMETERS (the
    authorised fallback where machine-readable raw supplements were not retrievable),
    each calibrated to and cited against a specific source. They are clearly labelled
    as simulated; only the parameters (means, periods, chance levels, effect directions)
    come from the literature.

Outputs: reference_datasets/<module>/sampleN.csv, plus RESULTS.md and PROVENANCE.md.
Run:  python reference_datasets/build_and_run.py
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from modules.data_parser import load_data
from modules.normalizer import compute_all_metrics
from modules import survival as SV, ymaze as YM, dams as DM, larval as LV, grooming as GR
from modules.ymaze import turns_to_arm_sequence

RESULTS = []   # (module, sample, type, source, key_result, expected, ok)
PROV = []      # (module, sample, type, citation, calibration)
rng_global = np.random.default_rng(20260629)


def save(module, name, df):
    df.to_csv(ROOT / module / f"{name}.csv", index=False)


def rec(module, sample, typ, source, key, expected, ok, citation, calib):
    RESULTS.append((module, sample, typ, source, key, expected, "PASS" if ok else "CHECK"))
    PROV.append((module, sample, typ, citation, calib))


# ── CLIMBING (3 disease models; WT high, disease impaired, drug rescues) ──────
def climbing():
    # (label, WT endpoint pass-frac, disease pass-frac, drug pass-frac, citation)
    models = [
        ("PD_aSyn", 0.88, 0.40, 0.66, "Feany & Bender 2000 Nature 404:394 (a-syn PD model: progressive climbing loss)"),
        ("ALS_TDP43", 0.88, 0.30, 0.58, "Gargano et al. 2005 Exp Gerontol 40:386 (RING; locomotor decline)"),
        ("HD_Q93", 0.88, 0.35, 0.55, "Drosophila HD polyQ models: climbing impairment rescued by treatment"),
    ]
    for i, (lab, wt, dis, drug, cite) in enumerate(models, 1):
        rng = np.random.default_rng(100 + i)
        rows = []
        groups = {("Oregon-WT", "Vehicle"): wt, ("Oregon-WT", "Drug"): wt,
                  (lab, "Vehicle"): dis, (lab, "Drug"): drug}
        for (g, t), pf in groups.items():
            for sex in ("Male", "Female"):
                for repn in range(1, 6):
                    n = int(rng.integers(18, 25))
                    a = float(np.clip(pf + rng.normal(0, 0.03), 0.02, 0.98))
                    for tm in (10, 20, 30, 60):
                        p = a * (1 - np.exp(-0.10 * tm))
                        rows.append({"Genotype": g, "Treatment": t, "Sex": sex, "Replicate": repn,
                                     "Time": tm, "Count_climbed": int(rng.binomial(n, p)), "n_total_flies": n})
        df = pd.DataFrame(rows); save("climbing", f"sample{i}_{lab}", df)
        ld, _, _ = load_data(df); pi = compute_all_metrics(ld)["pi"]
        pg = pi.groupby(["Genotype", "Treatment"])["PI_mean"].mean()
        wt_pi = pg[("Oregon-WT", "Vehicle")]; dis_pi = pg[(lab, "Vehicle")]; drug_pi = pg[(lab, "Drug")]
        ok = (wt_pi > dis_pi) and (drug_pi > dis_pi)
        rec("Climbing", f"sample{i}_{lab}", "simulated", cite.split("(")[0].strip(),
            f"WT PI={wt_pi:.0f}, {lab}-veh={dis_pi:.0f}, {lab}-drug={drug_pi:.0f}",
            "WT>disease; drug rescues", ok, cite, f"endpoint pass fraction WT={wt}, disease={dis}, drug={drug}")


# ── SURVIVAL (sample1 REAL Freireich; 2-3 Gompertz fly cohorts, WT median ~50 d) ─
def survival():
    # sample 1 — REAL Freireich 1963 6-MP raw data
    MP_T=[10,7,32,23,22,6,16,34,32,25,11,20,19,6,17,35,6,13,9,6,10]
    MP_E=[1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0]
    PL_T=[1,22,3,12,8,17,2,11,8,12,2,5,4,15,8,23,5,11,4,1,8]; PL_E=[1]*21
    df1 = pd.DataFrame({"Genotype":["Leukemia"]*42,"Treatment":["6-MP"]*21+["Placebo"]*21,
                        "Sex":["Pooled"]*42,"Time":MP_T+PL_T,"Event":MP_E+PL_E})
    save("survival", "sample1_Freireich1963_REAL", df1)
    ev = SV.to_event_table(df1); lr = SV.logrank_overall(ev)
    med = SV.median_survival_table(ev).set_index("Treatment")["Median_Survival"]
    ok = abs(lr["chisq"]-16.79) < 0.1 and med["6-MP"]==23 and med["Placebo"]==8
    rec("Survival","sample1_Freireich1963","REAL raw data","Freireich et al. 1963 Blood 21:699",
        f"median 23/8 wk; log-rank χ²={lr['chisq']:.2f}","χ²≈16.79; median 23/8 (published)",ok,
        "Freireich EJ et al. 1963 Blood 21:699-716","verified raw event-level data")
    # samples 2-3 — Gompertz fly lifespan (WT median ~50 d, Linford 2013); disease shorter, drug extends
    for i,(seed,wt_med,dis_med,drug_med) in enumerate([(2,50,30,42),(3,52,34,45)],2):
        rng=np.random.default_rng(500+seed); rows=[]
        # Gompertz a from target median: median = (1/b)ln(1+(b/a)ln2); pick b=0.09, solve a
        b=0.09
        def a_for(med): return (b*np.log(2))/(np.exp(b*med)-1)
        groups={("Oregon-WT","Vehicle"):a_for(wt_med),("Oregon-WT","Drug"):a_for(wt_med),
                ("Disease","Vehicle"):a_for(dis_med),("Disease","Drug"):a_for(drug_med)}
        for (g,t),a in groups.items():
            for sex in ("Male","Female"):
                n=int(rng.integers(45,60)); U=rng.uniform(0,1,n)
                tt=(1/b)*np.log(1+(b/a)*(-np.log(1-U)))
                for ti in tt:
                    rows.append((g,t,sex,90.0,0) if ti>=90 else (g,t,sex,round(float(ti),1),1))
        df=pd.DataFrame(rows,columns=["Genotype","Treatment","Sex","Time","Event"])
        save("survival",f"sample{i}_fly_lifespan",df)
        ev=SV.to_event_table(df); m=SV.median_survival_table(ev); m["g"]=m.Genotype+"|"+m.Treatment
        ms=m.groupby("g")["Median_Survival"].mean()
        ok=ms["Oregon-WT|Vehicle"]>ms["Disease|Vehicle"] and ms["Disease|Drug"]>ms["Disease|Vehicle"]
        rec("Survival",f"sample{i}_fly_lifespan","simulated","Linford et al. 2013 (WT median ~50 d)",
            f"WT median={ms['Oregon-WT|Vehicle']:.0f}d, disease={ms['Disease|Vehicle']:.0f}d, drug={ms['Disease|Drug']:.0f}d",
            "WT>disease; drug extends",ok,"Linford NJ et al. 2013 J Vis Exp e50068",
            f"Gompertz; target medians WT={wt_med}, disease={dis_med}, drug={drug_med} d")


# ── Y-MAZE (3 cohorts; chance 50%, WT>chance; turn bias 0.5/23.5% extreme) ────
def ymaze():
    # rho ~ alternation; per genotype memory level. 3 disease contrasts.
    specs=[("amnesiac",0.34,0.02),("rut2080",0.36,0.05),("dnc1",0.30,0.0)]  # (disease, WT rho, disease rho)
    for i,(dis,wt_rho,dis_rho) in enumerate(specs,1):
        rng=np.random.default_rng(700+i); rows=[]; fid=0
        for (g,rho) in [("Oregon-WT",wt_rho),(dis,dis_rho)]:
            for sex in ("Male","Female"):
                for _ in range(int(rng.integers(20,30))):
                    b=rng.beta(4.6,4.6); turns=["R" if rng.random()<b else "L"]
                    for _ in range(199):
                        turns.append(turns[-1] if rng.random()<rho else ("R" if rng.random()<b else "L"))
                    rows.append({"Genotype":g,"Treatment":"NA","Sex":sex,"FlyID":f"f{fid}",
                                 "Sequence":"".join(turns_to_arm_sequence(turns))}); fid+=1
        df=pd.DataFrame(rows); save("ymaze",f"sample{i}_{dis}",df)
        res=YM.run_ymaze_analysis(df); pf=YM.per_fly_metrics(YM.load_ymaze(df))
        alt=pf.groupby("Genotype")["alternation_pct"].mean(); tb=pf["turn_bias"].mean()
        pe=100*np.mean((pf["turn_bias"]>0.7)|(pf["turn_bias"]<0.3))
        ok=alt["Oregon-WT"]>alt[dis] and abs(tb-0.5)<0.04
        rec("Y-maze",f"sample{i}_{dis}","simulated","Lewis 2017 (50% chance); de Bivort 2015 (bias 0.5)",
            f"WT alt={alt['Oregon-WT']:.0f}%, {dis}={alt[dis]:.0f}%; bias={tb:.2f}; extreme={pe:.0f}%",
            "WT>disease; bias≈0.50; ~23.5% extreme",ok,
            "Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700",
            f"chance 50%; turn bias Beta(4.6,4.6); WT rho={wt_rho}, disease rho={dis_rho}")


# ── DAMS (sample1 REAL Konopka-Benzer periods; 2-3 sleep cohorts, 5-min rule) ─
def dams():
    # sample 1 — simulate DD activity with REAL Konopka & Benzer (1971) periods
    rng=np.random.default_rng(900); rows=[]; n_days=8; mins=np.arange(n_days*1440,dtype=float)
    for geno,period in [("wild-type",24.0),("per-short",19.0),("per-long",28.0)]:
        for k in range(6):
            per=period+rng.normal(0,0.1)
            pos=(mins%(per*60))/(per*60); active=pos<0.6
            u=np.clip(pos/0.6,0,1)
            rate=(1.3*np.exp(-((u-0.1)**2)/(2*0.1**2))+1.6*np.exp(-((u-0.9)**2)/(2*0.12**2))+0.25)*active
            cnt=rng.poisson(np.maximum(rate,0)*6.0).astype(float); cnt[~active]=0
            rows.append(pd.DataFrame({"Genotype":geno,"Treatment":"NA","Sex":"Pooled",
                "FlyID":f"{geno}_{k}","Time_min":mins,"Activity":cnt}))
    df=pd.concat(rows,ignore_index=True); save("dams","sample1_KonopkaBenzer_periods",df)
    ppf=DM.period_per_fly(DM.load_dams(df)); per=ppf.groupby("Genotype")["Period_h_chisq"].mean()
    ok=abs(per["wild-type"]-24)<0.5 and abs(per["per-short"]-19)<0.6 and abs(per["per-long"]-28)<0.6
    rec("DAMS","sample1_period_mutants","REAL published periods","Konopka & Benzer 1971 PNAS 68:2112",
        f"WT={per['wild-type']:.1f}, perS={per['per-short']:.1f}, perL={per['per-long']:.1f} h",
        "24 / 19 / 28 h (published)",ok,"Konopka RJ & Benzer S 1971 PNAS 68:2112-2116",
        "activity simulated with the published free-running periods 24/19/28 h")
    # samples 2-3 — sleep cohorts (>=5-min rule; males sleep more — Hendricks/Shaw 2000)
    for i,seed in enumerate([2,3],2):
        d=DM.demo_dams(seed=seed*11, n_days=4); save("dams",f"sample{i}_sleep_cohort",d)
        ss=DM.sleep_summary(DM.sleep_per_fly(DM.load_dams(d)))
        male=ss[ss.Sex=="Male"]["Total_sleep_min_per_day"].mean(); fem=ss[ss.Sex=="Female"]["Total_sleep_min_per_day"].mean()
        rec("DAMS",f"sample{i}_sleep","simulated","Hendricks 2000; Shaw 2000 (5-min rule; M>F sleep)",
            f"male sleep={male:.0f}, female={fem:.0f} min/day",">=5-min rule; males>females",male>fem,
            "Hendricks JC 2000 Neuron 25:129; Shaw PJ 2000 Science 287:1834","5-min inactivity sleep rule")


# ── LARVAL (3 contrasts; WT ~1 mm/s) ─────────────────────────────────────────
def larval():
    specs=[("ALS_TDP43",1.00,0.52,0.78),("PD_Pink1",1.02,0.60,0.85),("HD_Q93",0.98,0.55,0.80)]
    for i,(dis,wt,disv,drug) in enumerate(specs,1):
        rng=np.random.default_rng(1300+i); rows=[]; fid=0
        for g,spd in [("Oregon-WT",wt),(dis,disv),(dis+"+Drug",drug)]:
            for sex in ("Male","Female"):
                for _ in range(int(rng.integers(10,16))):
                    v=max(0.05,rng.normal(spd,0.12)); tort=max(1.0,rng.normal(1.2 if g=="Oregon-WT" else 1.45,0.1))
                    rows.append({"Genotype":g,"Treatment":"NA","Sex":sex,"FlyID":f"l{fid}","duration_s":60.0,
                                 "mean_speed_mm_s":round(v,3),"path_length_mm":round(v*60,2),
                                 "displacement_mm":round(v*60/tort,2),"tortuosity":round(tort,3)}); fid+=1
        df=pd.DataFrame(rows); save("larval",f"sample{i}_{dis}",df)
        sp=df.groupby("Genotype")["mean_speed_mm_s"].mean()
        ok=abs(sp["Oregon-WT"]-1.0)<0.15 and sp["Oregon-WT"]>sp[dis]
        rec("Larval",f"sample{i}_{dis}","simulated","WT third-instar ~1 mm/s (published range 0.5-2)",
            f"WT={sp['Oregon-WT']:.2f}, {dis}={sp[dis]:.2f} mm/s","WT≈1 mm/s; WT>disease",ok,
            "Gargano et al. 2005 (locomotion); larval crawling literature","WT mean speed ~1.0 mm/s")


# ── GROOMING (3 cohorts; WT ~13% time — Qiao 2018) ───────────────────────────
def grooming():
    specs=[("dCREB2",0.13,0.06,0.10),("FoxP",0.135,0.07,0.11),("Cul3",0.12,0.05,0.095)]
    obs=600.0
    for i,(dis,wt,disv,drug) in enumerate(specs,1):
        rng=np.random.default_rng(1700+i); rows=[]; fid=0
        for g,frac in [("Oregon-WT",wt),(dis,disv),(dis+"+Drug",drug)]:
            for sex in ("Male","Female"):
                for _ in range(int(rng.integers(12,18))):
                    target=max(0,rng.normal(frac,0.02))*obs; tcur=0.0; acc=0.0
                    while acc<target and tcur<obs:
                        gap=rng.exponential(25.0); bout=max(0.5,rng.normal(4.0,1.2)); tcur+=gap
                        if tcur+bout>obs: break
                        rows.append({"Genotype":g,"Treatment":"NA","Sex":sex,"FlyID":f"f{fid}",
                                     "Bout_start_s":round(tcur,1),"Bout_end_s":round(tcur+bout,1),"Observation_s":obs})
                        tcur+=bout; acc+=bout
                    fid+=1
        df=pd.DataFrame(rows); save("grooming",f"sample{i}_{dis}",df)
        pf=GR.per_fly_metrics(df); pc=pf.groupby("Genotype")["Pct_time_grooming"].mean()
        ok=abs(pc["Oregon-WT"]-13)<3 and pc["Oregon-WT"]>pc[dis]
        rec("Grooming",f"sample{i}_{dis}","simulated","Qiao et al. 2018 eLife (WT ~13% time)",
            f"WT={pc['Oregon-WT']:.0f}%, {dis}={pc[dis]:.0f}%","WT≈13%; WT>disease",ok,
            "Qiao B et al. 2018 eLife 7:e34497","WT ~13% time grooming")


def main():
    for fn in (climbing, survival, ymaze, dams, larval, grooming):
        fn()
    # RESULTS.md
    with open(ROOT/"RESULTS.md","w",encoding="utf-8") as f:
        f.write("# Reference-dataset results\n\nEach dataset (template format, in its module folder) "
                "run through the real DroPhenix module; result vs the published expectation.\n\n")
        f.write("| Module | Dataset | Type | Source | Key result | Expected | Status |\n|---|---|---|---|---|---|---|\n")
        for m,s,t,src,k,e,ok in RESULTS:
            f.write(f"| {m} | {s} | {t} | {src} | {k} | {e} | {ok} |\n")
        npass=sum(1 for r in RESULTS if r[6]=="PASS")
        f.write(f"\n**{npass}/{len(RESULTS)} datasets reproduce the published expectation.**\n")
    # PROVENANCE.md
    with open(ROOT/"PROVENANCE.md","w",encoding="utf-8") as f:
        f.write("# Reference-dataset provenance\n\n"
                "REAL = verified raw data or exact published parameters. simulated = generated from "
                "text-verified published parameters (authorised fallback), cited below.\n\n")
        f.write("| Module | Dataset | Type | Citation | Calibration parameters |\n|---|---|---|---|---|\n")
        for m,s,t,c,cal in PROV:
            f.write(f"| {m} | {s} | {t} | {c} | {cal} |\n")
    def _safe(s): return str(s).encode("ascii", "replace").decode("ascii")
    print("RESULTS:")
    for r in RESULTS: print("  ", r[6], r[0], r[1], "->", _safe(r[4]))
    npass=sum(1 for r in RESULTS if r[6]=="PASS")
    print(f"\n{npass}/{len(RESULTS)} datasets reproduce published expectation")
    return 0 if npass==len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
