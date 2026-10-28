"""
DroPhenix — dam_pipeline.py
From raw TriKinetics monitor files to a design-aware, verified DAM analysis,
without a person having to know which of the pieces to call or in what order.

This is the workflow that produced docs/DAM_RESULTS_REPORT.docx, generalised so
that it does not know the names Wt, Htt or CyO. It reuses the existing modules
rather than re-deriving anything:

    dam_reader     read raw files, channel map, death detection, plausibility
    dams           sleep parameters, light channel, photoperiod, ZT
    data_triage    what the design can answer
    stats_engine   van Elteren stratified test, BH-FDR, effect sizes
    drug_ranking   Hedges' g with unequal-n variance, inverse-variance pooling

WHAT IT ADDS
------------
1. A recording QUALITY pass that proposes the analysis window: complete days on
   which the light schedule was the same as every other day, with no gap and
   no flicker. On the laboratory's two runs this recovers exactly the windows
   chosen by hand (23-30 Aug 2026; 12-14 Aug 2026) and excludes the two days on
   which the light cycle failed and the set-up day on which it flickered.
2. DESIGN-AWARE contrasts. Every contrast is labelled monitor-clean or
   confounded by asking whether the two arms sit in the same set of monitors --
   generically, from the channel map -- and the primary treatment analysis
   pools only the clean strata.
3. The pre-specified OUTLIER rule, with every family run twice: all flies, and
   without the flagged flies.
4. Every parameter the published DAM packages report, sex-stratified primary
   tests with FDR within each family, per-sex secondary tests, phase-normalised
   sleep and activity, and a two-run (timepoint) comparison.

Nothing here is a new statistical method. It is the analysis of
scripts/report_dam_full.py made reusable, and validation/validate_dam_pipeline.py
checks that it reproduces that report's numbers on the same files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from modules import dam_reader as DR
from modules import dams as DM
from modules import drug_ranking as DRK
from modules import stats_engine as SE

# The published parameter set, in the order the literature reports it.
PARAMS: List[Tuple[str, str, str]] = [
    ("Total_sleep_min_per_day", "Total sleep", "min/24 h"),
    ("Day_sleep_min_per_day", "Day sleep", "min"),
    ("Night_sleep_min_per_day", "Night sleep", "min"),
    ("N_bouts_per_day", "Sleep bouts", "n/24 h"),
    ("Mean_bout_min", "Mean bout length", "min"),
    ("Day_mean_bout_min", "Day bout length", "min"),
    ("Night_mean_bout_min", "Night bout length", "min"),
    ("Longest_bout_min", "Longest bout", "min"),
    ("Sleep_latency_night_min", "Night sleep latency", "min"),
    ("Waking_activity", "Waking activity", "counts/waking min"),
    ("Total_activity_per_day", "Total activity", "counts/24 h"),
]
PHASE_PARAMS: List[Tuple[str, str, str]] = [
    ("Day_sleep_per_light_h", "Day sleep per light-hour", "min/h"),
    ("Night_sleep_per_dark_h", "Night sleep per dark-hour", "min/h"),
    ("Day_activity_per_light_h", "Day activity per light-hour", "counts/h"),
    ("Night_activity_per_dark_h", "Night activity per dark-hour", "counts/h"),
]
METRICS = [m for m, _, _ in PARAMS]
LABEL = {m: l for m, l, _ in PARAMS + PHASE_PARAMS}
UNIT = {m: u for m, _, u in PARAMS + PHASE_PARAMS}

OUTLIER_Z = 3.5
OUTLIER_ON = ["Total_sleep_min_per_day", "Total_activity_per_day"]

# Names that usually mean "this is the reference level".
_REF_GENO = ("wt", "wild", "control", "ctrl", "w1118", "oregon", "canton", "cs", "+")
_REF_TRT = ("untreated", "vehicle", "control", "ctrl", "none", "dmso", "water", "0")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Loading and recording quality
# ═════════════════════════════════════════════════════════════════════════════
def load_run(monitor_paths: Sequence[str], channel_map: "pd.DataFrame | str",
             monitor_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Raw monitor files + channel map -> labelled long frame (one row per
    channel per minute, with Genotype / Sex / Treatment / Light / Status)."""
    long = DR.read_monitors([str(p) for p in monitor_paths], monitors=monitor_names)
    cmap = pd.read_csv(channel_map) if isinstance(channel_map, str) else channel_map.copy()
    # Tolerate the commonest mismatch: the map names monitors with a suffix the
    # files lack, or vice versa ("Monitor_1_25th day" vs "Monitor_1").
    file_mons = set(long["Monitor"].astype(str).unique())
    if not (file_mons & set(cmap["Monitor"].astype(str).unique())):
        def _stem(s):
            """'Monitor_1_25th day', 'Monitor1', 'M1', 'CtM001' -> '1': the FIRST
            number in the name identifies the monitor; later numbers are dates."""
            import re
            m = re.search(r"\d+", str(s))
            return str(int(m.group(0))) if m else str(s).lower()
        fm = {_stem(m): m for m in file_mons}
        cmap["Monitor"] = cmap["Monitor"].astype(str).map(lambda m: fm.get(_stem(m), m))
    lab = DR.apply_channel_map(long, cmap)
    lab.attrs["channel_map"] = cmap
    return lab


def _cycling_monitor(lab: pd.DataFrame) -> Optional[str]:
    """The monitor whose light channel changes state most often -- the one to
    believe when they disagree (a stuck sensor changes state once or never)."""
    best, n_best = None, -1
    for m, g in lab.groupby("Monitor"):
        d = g[["Timestamp", "Light"]].dropna().drop_duplicates("Timestamp").sort_values("Timestamp")
        n = int((d["Light"].diff().abs() > 0).sum())
        if n > n_best:
            best, n_best = m, n
    return best


def quality_report(lab: pd.DataFrame, min_light_span_min: float = DM.MIN_LIGHT_SPAN_MIN) -> dict:
    """
    What the recording physically contains, day by day, and which days can be
    analysed. Returns a dict with:

      monitors     per-monitor light-channel table (light h/24 h, transitions)
      cycling      the monitor whose light channel is trusted for ZT
      agreement    fraction of timestamps on which the monitors' light agree
      bin_min      recording bin
      days         one row per calendar day: minutes present, light hours,
                   transitions, largest gap, and whether the day is usable
      photoperiod_min  the schedule, measured from the trusted monitor
      window       (start, end) proposed: the longest run of usable days
      faults       plain-language list of what was found
    """
    faults: List[str] = []
    mons = DM.light_channel_report(
        DM.load_dams(DR.to_dams_long(lab, drop_dead=False)))
    cyc = _cycling_monitor(lab)
    piv = lab.pivot_table(index="Timestamp", columns="Monitor", values="Light", aggfunc="max").dropna()
    agreement = float((piv.nunique(axis=1) == 1).mean()) if piv.shape[1] > 1 else 1.0
    if piv.shape[1] > 1 and agreement < 0.95:
        stuck = [m for m in piv.columns if m != cyc]
        faults.append(f"The monitors' light channels agree on only {100*agreement:.1f}% of "
                      f"timestamps; {', '.join(stuck)} looks stuck. Zeitgeber time is taken "
                      f"from {cyc}.")

    d = lab[lab["Monitor"] == cyc][["Timestamp", "Light", "Status"]].drop_duplicates("Timestamp") \
        .sort_values("Timestamp")
    bin_min = float(np.median(np.diff(d["Timestamp"].to_numpy()).astype("timedelta64[s]").astype(float)) / 60.0)
    d = d.assign(date=d["Timestamp"].dt.date,
                 gap=d["Timestamp"].diff().dt.total_seconds().div(60.0))
    rows = []
    for day, g in d.groupby("date"):
        L = g["Light"].to_numpy(float)
        rows.append({"date": str(day), "minutes": int(len(g) * bin_min),
                     "light_h": round(float(np.nansum(L) * bin_min / 60.0), 2),
                     "transitions": int(np.nansum(np.abs(np.diff(L)) > 0)),
                     "largest_gap_min": float(np.nanmax(g["gap"].to_numpy())) if len(g) > 1 else 0.0,
                     "bad_status_min": int((g["Status"] != 1).sum() * bin_min)})
    days = pd.DataFrame(rows)
    complete = days[days["minutes"] >= 1440 - 5]
    med_light = float(complete["light_h"].median()) if len(complete) else np.nan
    reasons = []
    for _, r in days.iterrows():
        why = []
        if r["minutes"] < 1440 - 5:
            why.append("partial day")
        if np.isfinite(med_light) and abs(r["light_h"] - med_light) > 0.5:
            why.append(f"light {r['light_h']:.1f} h vs usual {med_light:.1f} h")
        if r["transitions"] > 3:
            why.append(f"light channel toggled {int(r['transitions'])} times")
        if r["largest_gap_min"] > 30:
            why.append(f"gap of {r['largest_gap_min']:.0f} min")
        reasons.append("; ".join(why))
    days["exclude_reason"] = reasons
    days["usable"] = days["exclude_reason"] == ""

    # Longest run of consecutive usable days.
    best, cur = [], []
    for _, r in days.iterrows():
        if r["usable"]:
            cur.append(r["date"])
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    window = None
    if best:
        window = (pd.Timestamp(best[0]), pd.Timestamp(best[-1]) + pd.Timedelta(1, "D"))
    for _, r in days[~days["usable"]].iterrows():
        faults.append(f"{r['date']} excluded: {r['exclude_reason']}.")
    n_use = int(days["usable"].sum())
    if 0 < len(best) < 3:
        faults.append(f"Only {len(best)} consecutive usable day(s). Published DAM sleep analyses average "
                      f"3 or more; per-fly estimates from this window will be noisy.")

    dl = DM.load_dams(DR.to_dams_long(
        lab[(lab.Timestamp >= window[0]) & (lab.Timestamp < window[1])] if window else lab,
        drop_dead=False))
    L = DM.photoperiod_minutes(dl)
    return {"monitors": mons, "cycling": cyc, "agreement": agreement, "bin_min": bin_min,
            "days": days, "photoperiod_min": L, "window": window, "faults": faults,
            "n_usable_days": int(days["usable"].sum()),
            "sleep_scoreable": bool(bin_min <= DM.SLEEP_MIN_DEFAULT)}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Design: factors, references, what is monitor-clean
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Design:
    genotype_col: str = "Genotype"
    treatment_col: str = "Treatment"
    stratify_col: Optional[str] = "Sex"
    ref_genotype: Optional[str] = None
    ref_treatment: Optional[str] = None

    def resolve(self, pf: pd.DataFrame) -> "Design":
        """Fill the reference levels from the data if not given."""
        def pick(col, hints):
            levels = sorted(pf[col].dropna().astype(str).unique())
            for lv in levels:
                if any(lv.lower().startswith(h) or lv.lower() == h for h in hints):
                    return lv
            return levels[0] if levels else None
        d = Design(self.genotype_col, self.treatment_col, self.stratify_col,
                   self.ref_genotype, self.ref_treatment)
        if d.ref_genotype is None and d.genotype_col in pf:
            d.ref_genotype = pick(d.genotype_col, _REF_GENO)
        if d.ref_treatment is None and d.treatment_col in pf:
            d.ref_treatment = pick(d.treatment_col, _REF_TRT)
        if d.stratify_col and (d.stratify_col not in pf or pf[d.stratify_col].nunique() < 2):
            d.stratify_col = None
        return d


def monitor_clean(pf: pd.DataFrame, mask_a, mask_b) -> bool:
    """True when both arms of a contrast sit in the same set of monitors."""
    return set(pf.loc[mask_a, "Monitor"]) == set(pf.loc[mask_b, "Monitor"])


def design_table(pf: pd.DataFrame, d: Design) -> pd.DataFrame:
    """Which monitors each genotype x treatment group occupies -- the table a
    reader needs to see why a contrast is or is not clean."""
    return (pf.groupby([d.genotype_col, d.treatment_col])["Monitor"]
              .agg(lambda s: ", ".join(sorted(set(s)))).reset_index()
              .rename(columns={"Monitor": "monitors"}))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Per-fly parameters and outliers
# ═════════════════════════════════════════════════════════════════════════════
def per_fly(lab: pd.DataFrame, window: Optional[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    df = lab if window is None else lab[(lab.Timestamp >= window[0]) & (lab.Timestamp < window[1])]
    dl = DM.load_dams(DR.to_dams_long(df, drop_dead=True))
    L = float(DM.photoperiod_minutes(dl) or 720.0)
    pf = DM.sleep_per_fly(dl, photoperiod_min=L)
    pf["Monitor"] = pf["FlyID"].str.extract(r"^(.*)_C\d+$")[0]
    pf["Channel"] = pf["FlyID"].str.extract(r"_C(\d+)$")[0].astype(int)
    pf["Photoperiod_h"] = round(L / 60.0, 2)
    pf["Day_activity_per_light_h"] = (pf.Day_activity_per_day / (L / 60.0)).round(2)
    pf["Night_activity_per_dark_h"] = (pf.Night_activity_per_day / ((1440 - L) / 60.0)).round(2)
    pf["Day_sleep_per_light_h"] = (pf.Day_sleep_min_per_day / (L / 60.0)).round(2)
    pf["Night_sleep_per_dark_h"] = (pf.Night_sleep_min_per_day / ((1440 - L) / 60.0)).round(2)
    return pf


def flag_outliers(pf: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """
    Pre-specified rule: modified z (Iglewicz & Hoaglin 1993) of the deviation
    from the fly's own group median, MAD pooled over all groups at this
    timepoint, |z| > 3.5 on total sleep or total activity. The within-genotype
    alternative is recorded beside it; it does not drive the flag. Outlier
    removal is a SENSITIVITY analysis; the all-flies analysis is primary.
    """
    pf = pf.copy()
    keys = [k for k in keys if k in pf.columns]
    reasons = pd.Series("", index=pf.index)
    for metric in OUTLIER_ON:
        med = pf.groupby(keys)[metric].transform("median")
        resid = pf[metric] - med
        mad = float(np.median(np.abs(resid - np.median(resid))))
        if mad <= 0:
            pf[f"z_{metric}"] = 0.0
            continue
        z = 0.6745 * resid / mad
        pf[f"z_{metric}"] = z.round(2)
        gcol = keys[0] if keys else None
        zg = pd.Series(np.nan, index=pf.index)
        if gcol:
            for _, idx in pf.groupby(gcol).groups.items():
                r = resid.loc[idx]; m_ = float(np.median(np.abs(r - np.median(r))))
                zg.loc[idx] = 0.6745 * r / m_ if m_ > 0 else 0.0
        pf[f"z_{metric}_within_genotype"] = zg.round(2)
        hit = z.abs() > OUTLIER_Z
        short = "sleep" if "sleep" in metric else "activity"
        reasons[hit] = reasons[hit].apply(lambda s, m=short: (s + "; " if s else "") + m)
    pf["Outlier"] = reasons != ""
    pf["Outlier_reason"] = reasons
    return pf


def per_fly_profile(lab: pd.DataFrame, window, epoch_min: float = 30.0) -> Tuple[pd.DataFrame, float]:
    """Per-fly sleep and activity per epoch of zeitgeber time -- the material
    for a mean +/- SEM profile."""
    df = lab if window is None else lab[(lab.Timestamp >= window[0]) & (lab.Timestamp < window[1])]
    dl = DM.load_dams(DR.to_dams_long(df, drop_dead=True))
    L = float(DM.photoperiod_minutes(dl) or 720.0)
    start = float(DM.lights_on_offset(dl) or 0.0)
    rows = []
    for (g, tr, sx, fid), s in dl.groupby(["Genotype", "Treatment", "Sex", "FlyID"], sort=False):
        s = s.sort_values("Time_min")
        t = s.Time_min.to_numpy(float); a = s.Activity.to_numpy(float)
        bouts = DM._zero_run_bouts(a, 1.0, DM.SLEEP_MIN_DEFAULT)
        mask = np.zeros(len(a), bool)
        for st, ln in bouts:
            mask[st:st + ln] = True
        zt = (t - start) % 1440.0
        ep = (zt // epoch_min).astype(int)
        act = pd.Series(a).groupby(ep).mean() * epoch_min
        slp = pd.Series(mask.astype(float)).groupby(ep).mean() * epoch_min
        for e in slp.index:
            rows.append({"Genotype": g, "Treatment": tr, "Sex": sx, "FlyID": fid,
                         "zt": e * epoch_min / 60.0, "sleep": float(slp[e]), "activity": float(act[e])})
    return pd.DataFrame(rows), L


# ═════════════════════════════════════════════════════════════════════════════
# 4. Contrasts
# ═════════════════════════════════════════════════════════════════════════════
def contrast(a, b, label_a: str, label_b: str, metric: str, stratum: str,
             clean: bool, note: str = "") -> dict:
    """One unstratified contrast: exact Mann-Whitney (rank test chosen a
    priori), Hedges' g with CI, rank-biserial r, smallest attainable p."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    rec = {"metric": metric, "stratum": stratum, "ref": label_a, "cmp": label_b,
           "n_ref": len(a), "n_cmp": len(b), "monitor_clean": clean, "note": note,
           "analysis": "secondary", "test": "exact Mann-Whitney"}
    if len(a) < 2 or len(b) < 2:
        rec.update({"p": np.nan}); return rec
    method = "exact" if (len(a) + len(b)) <= 30 else "asymptotic"
    try:
        U, p = stats.mannwhitneyu(b, a, alternative="two-sided", method=method)
    except ValueError:
        method = "asymptotic"; U, p = stats.mannwhitneyu(b, a, alternative="two-sided", method=method)
    g = DRK.hedges_g(a, b)
    rec.update({
        "mean_ref": round(float(a.mean()), 3), "mean_cmp": round(float(b.mean()), 3),
        "median_ref": round(float(np.median(a)), 3), "median_cmp": round(float(np.median(b)), 3),
        "sem_ref": round(float(a.std(ddof=1) / np.sqrt(len(a))), 3),
        "sem_cmp": round(float(b.std(ddof=1) / np.sqrt(len(b))), 3),
        "diff": round(float(b.mean() - a.mean()), 3),
        "pct_change": round(100.0 * (b.mean() - a.mean()) / a.mean(), 2) if a.mean() != 0 else np.nan,
        "hedges_g": round(g["g"], 3), "g_lo": round(g["ci_lo"], 3), "g_hi": round(g["ci_hi"], 3),
        "g_var": g["var"], "rank_biserial_r": round(SE.rank_biserial_r(b, a), 3),
        "U": float(U), "p": float(p), "p_method": method,
        "p_min_attainable": round(SE.min_attainable_p(len(a), len(b)), 5),
    })
    return rec


def stratified(sub: pd.DataFrame, mask_ref, mask_cmp, label_a: str, label_b: str,
               metric: str, stratum: str, clean: bool, note: str = "",
               strat_col: Optional[str] = "Sex") -> dict:
    """The PRIMARY contrast: van Elteren stratified Wilcoxon (exact), with
    fixed-effect pooled Hedges' g; random-effects g and I^2 beside it."""
    pairs, gs, vs, ns = [], [], [], []
    levels = sorted(sub[strat_col].dropna().unique()) if strat_col else [None]
    for lev in levels:
        sel = (sub[strat_col] == lev) if strat_col else pd.Series(True, index=sub.index)
        a = sub.loc[mask_ref & sel, metric].to_numpy(float); a = a[np.isfinite(a)]
        b = sub.loc[mask_cmp & sel, metric].to_numpy(float); b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        pairs.append((a, b)); e = DRK.hedges_g(a, b)
        gs.append(e["g"]); vs.append(e["var"]); ns.append((len(a), len(b)))
    rec = {"metric": metric, "stratum": stratum, "ref": label_a, "cmp": label_b,
           "monitor_clean": clean, "note": note, "analysis": "primary",
           "test": f"van Elteren stratified Wilcoxon ({strat_col} as stratum)" if strat_col
                   else "exact Mann-Whitney",
           "n_strata": len(pairs), "n_ref": int(sum(n[0] for n in ns)),
           "n_cmp": int(sum(n[1] for n in ns)),
           "n_per_stratum": "; ".join(f"{x}v{y}" for x, y in ns)}
    if not pairs:
        rec["p"] = np.nan; return rec
    ve = SE.van_elteren_test(pairs)
    pool = DRK.inverse_variance_pool(np.array(gs), np.array(vs))
    if pool.get("k", 0) == 0:
        # Every stratum had zero pooled variance (e.g. a parameter that is 0 in
        # every fly): no effect size is defined. Report the test, not a number.
        pool = {"fixed": np.nan, "se_fixed": np.nan, "random": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "I2": np.nan}
    A = np.concatenate([a for a, _ in pairs]); B = np.concatenate([b for _, b in pairs])
    rec.update({
        "mean_ref": round(float(A.mean()), 3), "mean_cmp": round(float(B.mean()), 3),
        "median_ref": round(float(np.median(A)), 3), "median_cmp": round(float(np.median(B)), 3),
        "sem_ref": round(float(A.std(ddof=1) / np.sqrt(len(A))), 3),
        "sem_cmp": round(float(B.std(ddof=1) / np.sqrt(len(B))), 3),
        "diff": round(float(B.mean() - A.mean()), 3),
        "pct_change": round(100.0 * (B.mean() - A.mean()) / A.mean(), 2) if A.mean() != 0 else np.nan,
        "hedges_g": round(pool["fixed"], 3),
        "g_lo": round(pool["fixed"] - 1.96 * pool["se_fixed"], 3),
        "g_hi": round(pool["fixed"] + 1.96 * pool["se_fixed"], 3),
        "g_random": round(pool["random"], 3), "g_random_lo": round(pool["ci_lo"], 3),
        "g_random_hi": round(pool["ci_hi"], 3), "I2_pct": round(pool["I2"], 1),
        "g_per_stratum": "; ".join(f"{x:+.2f}" for x in gs),
        "p": float(ve["p"]), "p_method": ve["method"],
        "p_min_attainable": round(ve["p_min_attainable"], 5),
    })
    return rec


def add_fdr(rows: List[dict], family: str) -> pd.DataFrame:
    """Benjamini-Hochberg across the PRIMARY tests of one family only."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ok = (df["analysis"] == "primary") & df["p"].notna()
    q = np.full(len(df), np.nan)
    if ok.any():
        q[ok.to_numpy()] = SE.bh_correction(df.loc[ok, "p"].to_numpy())
    df["q_BH"] = np.round(q, 4); df["family"] = family
    df["significant"] = df["q_BH"] < 0.05
    return df


def _pairs(levels: Sequence[str], ref: Optional[str]) -> List[Tuple[str, str]]:
    """Reference first against each other level, then the remaining pairs."""
    levels = list(levels)
    out = []
    if ref in levels:
        out += [(ref, l) for l in levels if l != ref]
        rest = [l for l in levels if l != ref]
        out += [(a, b) for i, a in enumerate(rest) for b in rest[i + 1:]]
    else:
        out += [(a, b) for i, a in enumerate(levels) for b in levels[i + 1:]]
    return out


def family_genotype(pf: pd.DataFrame, d: Design, tag: str = "") -> pd.DataFrame:
    """Genotypes against each other among REFERENCE-treatment flies."""
    u = pf[pf[d.treatment_col] == d.ref_treatment] if d.ref_treatment else pf
    levels = sorted(u[d.genotype_col].dropna().unique())
    rows = []
    for m in METRICS:
        for ref, cmp_ in _pairs(levels, d.ref_genotype):
            ma, mb = (u[d.genotype_col] == ref), (u[d.genotype_col] == cmp_)
            if ma.sum() < 2 or mb.sum() < 2:
                continue
            clean = monitor_clean(u, ma, mb)
            note = "" if clean else "genotype is confounded with monitor here"
            rows.append(stratified(u, ma, mb, ref, cmp_, m, "stratified", clean, note, d.stratify_col))
            if d.stratify_col:
                for lev in sorted(u[d.stratify_col].dropna().unique()):
                    s = u[u[d.stratify_col] == lev]
                    m1, m2 = (s[d.genotype_col] == ref), (s[d.genotype_col] == cmp_)
                    if m1.sum() >= 2 and m2.sum() >= 2:
                        rows.append(contrast(s.loc[m1, m], s.loc[m2, m], ref, cmp_, m, str(lev), clean, note))
    return add_fdr(rows, f"genotype ({d.ref_treatment or 'all'}) {tag}".strip())


def family_genotype_within_monitor(pf: pd.DataFrame, d: Design, tag: str = "") -> pd.DataFrame:
    """
    Genotype contrasts INSIDE a single monitor, at a single treatment level:
    the check that separates genotype from the instrument and from channel
    position, wherever the layout allows it. Empty when no monitor holds >=2
    flies of two genotypes at the same treatment.
    """
    rows = []
    for mon in sorted(pf["Monitor"].unique()):
        for trt in sorted(pf[d.treatment_col].dropna().unique()):
            s = pf[(pf["Monitor"] == mon) & (pf[d.treatment_col] == trt)]
            levels = [g for g in sorted(s[d.genotype_col].unique()) if (s[d.genotype_col] == g).sum() >= 2]
            if len(levels) < 2:
                continue
            for m in METRICS:
                for ref, cmp_ in _pairs(levels, d.ref_genotype):
                    rows.append(stratified(s, s[d.genotype_col] == ref, s[d.genotype_col] == cmp_,
                                           ref, cmp_, m, f"{mon}, {trt}", True,
                                           "both genotypes inside one monitor", d.stratify_col))
    return add_fdr(rows, f"genotype within monitor {tag}".strip())


def family_treatment(pf: pd.DataFrame, d: Design, tag: str = "", params=None) -> pd.DataFrame:
    """Each treatment against the reference, within each genotype."""
    params = params or METRICS
    rows = []
    trts = sorted(pf[d.treatment_col].dropna().unique())
    for m in params:
        for g in sorted(pf[d.genotype_col].dropna().unique()):
            s = pf[pf[d.genotype_col] == g]
            for trt in trts:
                if trt == d.ref_treatment:
                    continue
                ma, mb = (s[d.treatment_col] == d.ref_treatment), (s[d.treatment_col] == trt)
                if ma.sum() < 2 or mb.sum() < 2:
                    continue
                clean = monitor_clean(s, ma, mb)
                note = "" if clean else "treatment is confounded with monitor here"
                rows.append(stratified(s, ma, mb, d.ref_treatment, trt, m, f"{g} stratified", clean, note, d.stratify_col))
                if d.stratify_col:
                    for lev in sorted(s[d.stratify_col].dropna().unique()):
                        ss = s[s[d.stratify_col] == lev]
                        m1, m2 = (ss[d.treatment_col] == d.ref_treatment), (ss[d.treatment_col] == trt)
                        if m1.sum() >= 2 and m2.sum() >= 2:
                            rows.append(contrast(ss.loc[m1, m], ss.loc[m2, m], d.ref_treatment, trt, m, f"{g} {lev}", clean, note))
    return add_fdr(rows, f"treatment within genotype {tag}".strip())


def family_treatment_pooled(pf: pd.DataFrame, d: Design, clean_only: bool, tag: str = "",
                            params=None) -> pd.DataFrame:
    """
    The highest-power treatment test: every fly, with genotype AND the stratum
    factor as strata. `clean_only` keeps only the genotype x stratum blocks in
    which the two treatment arms share a monitor -- the defensible primary.
    """
    params = params or METRICS
    key = pf[d.genotype_col].astype(str) + (" " + pf[d.stratify_col].astype(str) if d.stratify_col else "")
    pf = pf.assign(_block=key)
    if clean_only:
        keep = []
        for blk, s in pf.groupby("_block"):
            arms = s.groupby(d.treatment_col)["Monitor"].agg(lambda x: frozenset(x))
            if len(arms) >= 2 and len(set(arms)) == 1:
                keep.append(blk)
        pf = pf[pf["_block"].isin(keep)]
    rows = []
    for m in params:
        for trt in sorted(pf[d.treatment_col].dropna().unique()):
            if trt == d.ref_treatment:
                continue
            ma, mb = (pf[d.treatment_col] == d.ref_treatment), (pf[d.treatment_col] == trt)
            rows.append(stratified(pf, ma, mb, d.ref_treatment, trt, m,
                                   "monitor-clean blocks" if clean_only else "all blocks",
                                   clean_only, "" if clean_only else "includes monitor-confounded blocks",
                                   "_block"))
    return add_fdr(rows, f"treatment pooled ({'clean' if clean_only else 'all'}) {tag}".strip())


def family_timepoint(pf_a: pd.DataFrame, pf_b: pd.DataFrame, d: Design,
                     label_a: str, label_b: str, tag: str = "") -> pd.DataFrame:
    """Two runs, within each genotype x treatment group, stratified. Uses the
    phase-normalised parameters as well, because the photoperiods may differ."""
    both = pd.concat([pf_a.assign(_run=label_a), pf_b.assign(_run=label_b)], ignore_index=True)
    La, Lb = pf_a["Photoperiod_h"].median(), pf_b["Photoperiod_h"].median()
    note = ("tubes are not the same individuals across runs (unpaired)"
            + (f"; photoperiod differs ({La:.1f} h vs {Lb:.1f} h light), so day/night are per hour of phase"
               if abs(La - Lb) > 0.1 else ""))
    rows = []
    for m in METRICS + [p for p, _, _ in PHASE_PARAMS]:
        for g in sorted(both[d.genotype_col].dropna().unique()):
            for trt in sorted(both[d.treatment_col].dropna().unique()):
                s = both[(both[d.genotype_col] == g) & (both[d.treatment_col] == trt)]
                ma, mb = (s["_run"] == label_a), (s["_run"] == label_b)
                if ma.sum() < 2 or mb.sum() < 2:
                    continue
                rows.append(stratified(s, ma, mb, label_a, label_b, m, f"{g} {trt} stratified", True, note, d.stratify_col))
                if d.stratify_col:
                    for lev in sorted(s[d.stratify_col].dropna().unique()):
                        ss = s[s[d.stratify_col] == lev]
                        m1, m2 = (ss["_run"] == label_a), (ss["_run"] == label_b)
                        if m1.sum() >= 2 and m2.sum() >= 2:
                            rows.append(contrast(ss.loc[m1, m], ss.loc[m2, m], label_a, label_b, m, f"{g} {trt} {lev}", True, note))
    return add_fdr(rows, f"timepoint {tag}".strip())


def pairing_test(pf_a: pd.DataFrame, pf_b: pd.DataFrame, d: Design) -> pd.DataFrame:
    """Are the same individuals in the same tubes in both runs? Within-group
    Spearman correlation of each tube's value across runs, per parameter."""
    keys = [d.genotype_col, d.stratify_col, d.treatment_col] if d.stratify_col else [d.genotype_col, d.treatment_col]
    a = pf_a.assign(Tube=pf_a["Monitor"].astype(str).str.extract(r"(\d+)")[0] + "_C" + pf_a["Channel"].astype(str))
    b = pf_b.assign(Tube=pf_b["Monitor"].astype(str).str.extract(r"(\d+)")[0] + "_C" + pf_b["Channel"].astype(str))
    m = a.merge(b, on="Tube", suffixes=("_a", "_b"))
    rows = []
    for metric in ["Total_sleep_min_per_day", "Total_activity_per_day", "Mean_bout_min", "Waking_activity"]:
        ka = [k + "_a" for k in keys]
        ra = m[metric + "_a"] - m.groupby(ka)[metric + "_a"].transform("mean")
        rb = m[metric + "_b"] - m.groupby(ka)[metric + "_b"].transform("mean")
        rho, p = stats.spearmanr(ra, rb)
        rows.append({"metric": metric, "n_tubes": len(m), "within_group_rho": round(float(rho), 3), "p": round(float(p), 3)})
    out = pd.DataFrame(rows)
    out.attrs["paired"] = bool((out["within_group_rho"] > 0.5).all() and (out["p"] < 0.05).all())
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5. The run object and the orchestration
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class DamRun:
    name: str
    lab: pd.DataFrame
    quality: dict
    window: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
    design: Design
    per_fly: pd.DataFrame
    triage: object
    contrasts: Dict[str, pd.DataFrame] = field(default_factory=dict)
    contrasts_noout: Dict[str, pd.DataFrame] = field(default_factory=dict)
    profiles: Optional[pd.DataFrame] = None
    photoperiod_min: float = 720.0

    def group_table(self, outliers: bool = True) -> pd.DataFrame:
        d = self.design
        pf = self.per_fly if outliers else self.per_fly[~self.per_fly["Outlier"]]
        keys = [d.genotype_col] + ([d.stratify_col] if d.stratify_col else []) + [d.treatment_col]
        agg = {m: ["count", "mean", "sem"] for m in METRICS}
        return pf.groupby(keys).agg(agg).round(3)

    def summary(self) -> dict:
        """The counts a person asks for first."""
        out = {"name": self.name, "n_flies": int(len(self.per_fly)),
               "n_outliers": int(self.per_fly["Outlier"].sum()),
               "photoperiod": f"{self.photoperiod_min/60:.1f}L:{24 - self.photoperiod_min/60:.1f}D",
               "window": (str(self.window[0].date()), str((self.window[1] - pd.Timedelta(1, "D")).date()))
                         if self.window else None,
               "families": {}}
        for k, df in self.contrasts.items():
            p = df[df["analysis"] == "primary"] if not df.empty else df
            out["families"][k] = {"n_tests": int(len(p)), "n_significant": int(p["significant"].sum()) if len(p) else 0}
        return out


def analyse_run(lab: pd.DataFrame, name: str = "run", design: Optional[Design] = None,
                window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                quality: Optional[dict] = None) -> DamRun:
    """Labelled long frame -> everything. `window=None` uses the proposed one."""
    from modules import data_triage as T
    q = quality or quality_report(lab)
    win = window or q["window"]
    d = (design or Design()).resolve(DR.to_dams_long(lab, drop_dead=False))
    pf = per_fly(lab, win)
    keys = [d.genotype_col] + ([d.stratify_col] if d.stratify_col else []) + [d.treatment_col]
    pf = flag_outliers(pf, keys)
    cmap = lab.attrs.get("channel_map")
    tri = T.triage_dam(lab[(lab.Timestamp >= win[0]) & (lab.Timestamp < win[1])] if win else lab,
                       channel_map=None, per_fly=pf)
    run = DamRun(name, lab, q, win, d, pf, tri, photoperiod_min=float(q.get("photoperiod_min") or 720.0))
    for variant, sub in (("all", pf), ("noout", pf[~pf["Outlier"]])):
        fams = {
            "genotype": family_genotype(sub, d),
            "genotype_within_monitor": family_genotype_within_monitor(sub, d),
            "treatment": family_treatment(sub, d),
            "treatment_pooled_clean": family_treatment_pooled(sub, d, True),
            "treatment_pooled_all": family_treatment_pooled(sub, d, False),
        }
        if variant == "all":
            run.contrasts = fams
        else:
            run.contrasts_noout = fams
    run.profiles, _ = per_fly_profile(lab, win)
    return run


def compare_runs(a: DamRun, b: DamRun) -> dict:
    """Two-timepoint comparison: unpaired, stratified, phase-normalised, with
    the pairing test reported so the reader knows why it is unpaired."""
    d = a.design
    return {
        "timepoint": family_timepoint(a.per_fly, b.per_fly, d, a.name, b.name),
        "timepoint_noout": family_timepoint(a.per_fly[~a.per_fly["Outlier"]], b.per_fly[~b.per_fly["Outlier"]],
                                            d, a.name, b.name, "(no outliers)"),
        "pairing": pairing_test(a.per_fly, b.per_fly, d),
    }


def run_from_files(monitor_paths: Sequence[str], channel_map, name: str = "run",
                   design: Optional[Design] = None, window=None) -> DamRun:
    """One call: files + map -> analysed run."""
    lab = load_run(monitor_paths, channel_map)
    return analyse_run(lab, name, design, window)
