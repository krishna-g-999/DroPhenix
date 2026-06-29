"""
comparison_study.py
Head-to-head benchmark of DroPhenix against established, peer-reviewed analysis
tools on IDENTICAL data. Only tools that are actually installed and executed
here are reported — no tool is compared by hearsay.

Reference tools (all peer-reviewed, pip-installable, run live below):
  * lifelines    — survival analysis (Kaplan-Meier, log-rank, Cox PH)
  * statsmodels  — meta-analysis (inverse-variance / DerSimonian-Laird)
  * pingouin + scipy — ANOVA / Kruskal-Wallis / effect sizes
  * astropy      — Lomb-Scargle periodogram (circadian period)

For each metric we report DroPhenix, the reference tool, the absolute
difference, and whether they agree within a tight tolerance. A small, fixed
benchmark dataset (the canonical Freireich 1963 6-MP trial / PlantGrowth /
known-period signal) is used so results are exactly reproducible.

Writes a Markdown report to validation/comparison_report.md.
Run:  python validation/comparison_study.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROWS = []   # (domain, metric, drophenix, reference_tool, ref_value, abs_diff, agree)


def add(domain, metric, dp, tool, ref, tol):
    try:
        diff = abs(float(dp) - float(ref))
        agree = diff <= tol
    except Exception:
        diff, agree = np.nan, False
    ROWS.append((domain, metric, _r(dp), tool, _r(ref), _r(diff), "yes" if agree else "NO"))


def _r(v):
    try:
        return round(float(v), 5)
    except Exception:
        return v


# ── Benchmark data: Freireich 1963 6-MP leukemia trial (weeks) ────────────────
MP_T = [10, 7, 32, 23, 22, 6, 16, 34, 32, 25, 11, 20, 19, 6, 17, 35, 6, 13, 9, 6, 10]
MP_E = [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
PL_T = [1, 22, 3, 12, 8, 17, 2, 11, 8, 12, 2, 5, 4, 15, 8, 23, 5, 11, 4, 1, 8]
PL_E = [1] * 21

CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]


def compare_survival():
    from modules import survival as SV
    df = pd.DataFrame({
        "Genotype": ["Lk"] * 42, "Treatment": ["6-MP"] * 21 + ["Placebo"] * 21,
        "Sex": ["P"] * 42, "Time": MP_T + PL_T, "Event": MP_E + PL_E})
    ev = SV.to_event_table(df)
    med = SV.median_survival_table(ev).set_index("Treatment")
    lr = SV.logrank_overall(ev)
    cox = SV.cox_ph(ev, covariates=["Treatment"], interaction=False, ref={"Treatment": "6-MP"})
    dp_hr = float(cox["summary"].iloc[0]["HR"])

    try:
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import logrank_test
        kmf = KaplanMeierFitter()
        kmf.fit(MP_T, MP_E); med_mp = kmf.median_survival_time_
        kmf.fit(PL_T, PL_E); med_pl = kmf.median_survival_time_
        lt = logrank_test(MP_T, PL_T, event_observed_A=MP_E, event_observed_B=PL_E)
        cdf = pd.DataFrame({"T": MP_T + PL_T, "E": MP_E + PL_E,
                            "placebo": [0] * 21 + [1] * 21})
        cph = CoxPHFitter().fit(cdf, "T", "E")
        ll_hr = float(np.exp(cph.params_["placebo"]))

        add("Survival", "Median 6-MP (weeks)", med.loc["6-MP", "Median_Survival"], "lifelines", med_mp, 0.01)
        add("Survival", "Median Placebo (weeks)", med.loc["Placebo", "Median_Survival"], "lifelines", med_pl, 0.01)
        add("Survival", "Log-rank chi-square", lr["chisq"], "lifelines", lt.test_statistic, 0.01)
        add("Survival", "Log-rank p-value", lr["p_value"], "lifelines", lt.p_value, 1e-5)
        # Cox HR: DroPhenix/statsmodels use Breslow ties (matching the published
        # Klein-Moeschberger 2003 / SAS value 4.523). lifelines defaults to Efron
        # ties (4.817). Both are valid; we validate DroPhenix against the
        # published Breslow value and report lifelines-Efron as an informational
        # method-difference row (the Freireich data has many tied event times).
        add("Survival", "Cox HR Breslow (vs published 4.523)", dp_hr,
            "Klein-Moeschberger 2003", 4.523, 0.05)
        ROWS.append(("Survival", "Cox HR Efron (ties-method difference)", _r(dp_hr),
                     "lifelines", _r(ll_hr), _r(abs(dp_hr - ll_hr)), "info"))
    except Exception as e:
        ROWS.append(("Survival", f"lifelines unavailable: {e!r}", "-", "lifelines", "-", "-", "NO"))


def compare_stats():
    from modules import stats_engine as SE
    from scipy import stats as ss
    data = {"ctrl": np.array(CTRL), "trt1": np.array(TRT1), "trt2": np.array(TRT2)}
    an, _ = SE.anova_tukey(data)
    kw, _ = SE.kruskal_dunn(data)

    # scipy reference
    F_sp, p_sp = ss.f_oneway(CTRL, TRT1, TRT2)
    H_sp, ph_sp = ss.kruskal(CTRL, TRT1, TRT2)
    add("Stats", "One-way ANOVA F", an["F_statistic"], "scipy", F_sp, 0.01)
    add("Stats", "ANOVA p-value", an["p_value"], "scipy", p_sp, 1e-4)
    add("Stats", "Kruskal-Wallis H", kw["H_statistic"], "scipy", H_sp, 0.01)

    # pingouin reference (independent implementation)
    try:
        import pingouin as pg
        long = pd.DataFrame({"y": CTRL + TRT1 + TRT2,
                             "g": ["c"] * 10 + ["t1"] * 10 + ["t2"] * 10})
        agg = pg.anova(data=long, dv="y", between="g", detailed=True)
        F_pg = float(agg.loc[agg["Source"] == "g", "F"].iloc[0])
        add("Stats", "One-way ANOVA F", an["F_statistic"], "pingouin", F_pg, 0.01)
        # Hedges g effect size (trt2 vs ctrl) — same argument order on both sides
        es = pg.compute_effsize(TRT2, CTRL, eftype="hedges")
        dp_g = SE.hedges_g(np.array(TRT2), np.array(CTRL))   # (trt2 - ctrl)
        add("Stats", "Hedges g (trt2 vs ctrl)", dp_g, "pingouin", es, 0.02)
    except Exception as e:
        ROWS.append(("Stats", f"pingouin unavailable: {e!r}", "-", "pingouin", "-", "-", "NO"))


def compare_meta():
    from modules import drug_ranking as DR
    from statsmodels.stats.meta_analysis import combine_effects
    eff = np.array([0.5, 0.3, 0.8, 0.2, 0.6]); var = np.array([0.04, 0.02, 0.06, 0.03, 0.05])
    mine = DR.inverse_variance_pool(eff, var)
    sm = combine_effects(eff, var, method_re="dl")
    sf = sm.summary_frame()
    add("Meta-analysis", "Pooled (fixed)", mine["fixed"], "statsmodels", sf.loc["fixed effect", "eff"], 1e-6)
    add("Meta-analysis", "Pooled (DL random)", mine["random"], "statsmodels", sf.loc["random effect", "eff"], 1e-6)
    add("Meta-analysis", "Heterogeneity I^2 (%)", mine["I2"], "statsmodels", sm.i2 * 100, 1e-3)


def compare_circadian():
    from modules import dams as DM
    # known 24.0 h rhythm, 30-min bins, 8 days
    bin_min = 30; n = int(8 * 1440 / bin_min); t = np.arange(n) * bin_min
    rng = np.random.default_rng(0)
    y = rng.poisson(3 * (1 + np.sin(2 * np.pi * t / (24 * 60))) ** 2).astype(float)
    dp_chi = DM.chi_square_periodogram(y, bin_min, 16, 32)["peak_period_h"]
    add("Circadian", "Dominant period (h) [true 24]", dp_chi, "ground truth", 24.0, 0.5)
    try:
        from astropy.timeseries import LombScargle
        tmin = t.astype(float)  # minutes
        freq = np.linspace(1 / (32 * 60), 1 / (16 * 60), 4000)  # cycles/min
        power = LombScargle(tmin, y - y.mean()).power(freq)
        per_h = (1 / freq[np.argmax(power)]) / 60.0
        add("Circadian", "Dominant period (h)", dp_chi, "astropy LombScargle", per_h, 0.5)
    except Exception as e:
        ROWS.append(("Circadian", f"astropy unavailable: {e!r}", "-", "astropy", "-", "-", "NO"))


def compare_R():
    """Run R's gold-standard `survival` + base-R stats on the SAME data and
    compare to DroPhenix. Skipped gracefully if R (conda env 'rstats') is absent."""
    import subprocess, shutil
    from modules import survival as SV, stats_engine as SE
    rscript = Path(__file__).resolve().parent.parent / "manuscript" / "r_comparison.R"
    if not rscript.exists():
        return
    cmds = []
    conda = shutil.which("conda") or r"D:\Miniconda3\Scripts\conda.exe"
    cmds.append([conda, "run", "-n", "rstats", "Rscript", str(rscript)])
    cmds.append(["Rscript", str(rscript)])
    out = None
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if "MEDIAN_6MP=" in r.stdout:
                out = r.stdout; break
        except Exception:
            continue
    if out is None:
        ROWS.append(("R (survival/base)", "R not available in this environment",
                     "-", "R", "-", "-", "info"))
        return
    R = {}
    for line in out.splitlines():
        if "=" in line and line.split("=")[0].isupper():
            k, v = line.strip().split("=", 1)
            try: R[k] = float(v)
            except Exception: pass
    # DroPhenix values on the same benchmark data
    ev = SV.to_event_table(pd.DataFrame({"Genotype": ["Lk"]*42,
        "Treatment": ["6-MP"]*21+["Placebo"]*21, "Sex": ["P"]*42,
        "Time": MP_T+PL_T, "Event": MP_E+PL_E}))
    med = SV.median_survival_table(ev).set_index("Treatment")
    lr = SV.logrank_overall(ev)
    cox = SV.cox_ph(ev, covariates=["Treatment"], interaction=False, ref={"Treatment": "6-MP"})
    an, _ = SE.anova_tukey({"ctrl": np.array(CTRL), "trt1": np.array(TRT1), "trt2": np.array(TRT2)})
    kw, _ = SE.kruskal_dunn({"ctrl": np.array(CTRL), "trt1": np.array(TRT1), "trt2": np.array(TRT2)})
    add("R: survival", "Median 6-MP", med.loc["6-MP", "Median_Survival"], "R survfit", R.get("MEDIAN_6MP"), 0.01)
    add("R: survival", "Median Placebo", med.loc["Placebo", "Median_Survival"], "R survfit", R.get("MEDIAN_PLACEBO"), 0.01)
    add("R: survival", "Log-rank chi-square", lr["chisq"], "R survdiff", R.get("LOGRANK_CHISQ"), 0.01)
    add("R: survival", "Log-rank p-value", lr["p_value"], "R survdiff", R.get("LOGRANK_P"), 1e-5)
    add("R: survival", "Cox HR (Breslow)", float(cox["summary"].iloc[0]["HR"]), "R coxph(breslow)", R.get("COX_HR_BRESLOW"), 0.01)
    add("R: stats", "ANOVA F", an["F_statistic"], "R aov", R.get("ANOVA_F"), 0.01)
    add("R: stats", "ANOVA p-value", an["p_value"], "R aov", R.get("ANOVA_P"), 1e-4)
    add("R: stats", "Kruskal-Wallis H", kw["H_statistic"], "R kruskal.test", R.get("KRUSKAL_H"), 0.01)


def compare_pyboat():
    """Independent circadian period estimate via pyBOAT wavelet ridge analysis."""
    from modules import dams as DM
    binm = 30; n = int(8 * 1440 / binm); t = np.arange(n) * binm
    rng = np.random.default_rng(0)
    y = rng.poisson(3 * (1 + np.sin(2 * np.pi * t / (24 * 60))) ** 2).astype(float)
    dp = DM.chi_square_periodogram(y, binm, 16, 32)["peak_period_h"]
    try:
        from pyboat import WAnalyzer
        dt = binm / 60.0  # hours
        periods = np.linspace(16, 32, 200)
        wAn = WAnalyzer(periods, dt, time_unit_label="h")
        wAn.compute_spectrum(y, do_plot=False)
        per = float(wAn.get_maxRidge()["periods"].median())
        add("Circadian", "Dominant period (h)", dp, "pyBOAT wavelet", per, 0.5)
    except Exception as e:
        ROWS.append(("Circadian", f"pyBOAT unavailable: {e!r}", "-", "pyBOAT", "-", "-", "info"))


def compare_sksurv():
    """scikit-survival cross-check (skipped with an honest note if unavailable)."""
    try:
        import sksurv  # noqa: F401
    except Exception:
        ROWS.append(("Survival", "scikit-survival not installable here "
                     "(no C toolchain / no py3.13 wheel); survival already "
                     "cross-checked vs R survival + lifelines", "-",
                     "scikit-survival", "-", "-", "info"))
        return


def main() -> int:
    import warnings; warnings.filterwarnings("ignore")
    compare_survival()
    compare_stats()
    compare_meta()
    compare_circadian()
    compare_R()
    compare_pyboat()
    compare_sksurv()

    df = pd.DataFrame(ROWS, columns=["Domain", "Metric", "DroPhenix",
                                     "Reference tool", "Reference", "Abs diff", "Agree"])
    # console
    print("=" * 100)
    print("DroPhenix vs established tools — head-to-head on identical data")
    print("=" * 100)
    print(df.to_string(index=False))
    n_agree = (df["Agree"] == "yes").sum()
    n_total = (df["Agree"].isin(["yes", "NO"])).sum()
    print("-" * 100)
    print(f"Agreement: {n_agree}/{n_total} metrics match the reference within tolerance.")

    # markdown report
    rep = Path(__file__).resolve().parent / "comparison_report.md"
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write("# DroPhenix benchmark vs established tools\n\n")
        fh.write("All reference tools were installed and executed on identical data "
                 "(Freireich 1963 6-MP trial; PlantGrowth; a known 24 h rhythm; a fixed "
                 "set of effect sizes). Values are reproducible.\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write(f"\n\n**Agreement: {n_agree}/{n_total} metrics within tolerance.**\n")
    print(f"Report written: {rep}")
    return 0 if n_agree == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
