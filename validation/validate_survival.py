"""
validate_survival.py
Validation of DroPhenix modules/survival.py against PUBLISHED raw data with
known, peer-reviewed results — not synthetic data.

Primary benchmark: the Freireich et al. (1963) 6-mercaptopurine (6-MP) vs
placebo acute-leukemia remission trial. This is the canonical survival-analysis
dataset; raw data and reference statistics are tabulated in:

  - Freireich EJ et al. (1963) Blood 21:699-716 (original trial).
  - Gehan EA (1965) Biometrika 52:203-223 (generalized Wilcoxon).
  - Klein JP & Moeschberger ML (2003) "Survival Analysis", 2nd ed.,
    Table 1.1 / Section 1.2 (raw data reproduced below, in weeks).
  - Kleinbaum DG & Klein M, "Survival Analysis: A Self-Learning Text"
    (log-rank chi-square = 16.79; Cox treatment HR = 4.52).

Published reference values (treatment = 6-MP vs Placebo):
  median survival : 6-MP = 23 weeks, Placebo = 8 weeks
  log-rank        : chi-square = 16.79, p < 0.0001
  Gehan-Wilcoxon  : chi-square = 13.46 (Breslow weighting)
  Cox PH (Breslow): beta(Placebo) = 1.509, HR = 4.523, Wald p ~ 2e-4

Run:  python validation/validate_survival.py
Exit code 0 = all checks passed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the repo root or the validation/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import survival as S  # noqa: E402


# ── Published raw data (Klein & Moeschberger, weeks) ──────────────────────────
# 6-MP group: time, event (1 = relapse observed, 0 = censored)
MP_TIME = [10, 7, 32, 23, 22, 6, 16, 34, 32, 25, 11, 20, 19, 6, 17, 35, 6, 13, 9, 6, 10]
MP_EVENT = [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
# Placebo group: all 21 relapsed
PL_TIME = [1, 22, 3, 12, 8, 17, 2, 11, 8, 12, 2, 5, 4, 15, 8, 23, 5, 11, 4, 1, 8]
PL_EVENT = [1] * 21


def _freireich_event_df() -> pd.DataFrame:
    df = pd.DataFrame({
        "Genotype": ["Leukemia"] * 42,
        "Treatment": ["6-MP"] * 21 + ["Placebo"] * 21,
        "Sex": ["Pooled"] * 42,
        "Time": MP_TIME + PL_TIME,
        "Event": MP_EVENT + PL_EVENT,
    })
    return S.to_event_table(df)


def _check(name, got, expected, tol, results):
    ok = (got is not None) and np.isfinite(got) and abs(got - expected) <= tol
    results.append((name, expected, round(float(got), 4) if got is not None else None, tol, ok))
    return ok


def main() -> int:
    ev = _freireich_event_df()
    results = []

    # 1. Raw-data integrity: event/censoring counts must match the published trial.
    n_mp_event = int(ev[(ev.Treatment == "6-MP") & (ev.Event == 1)].shape[0])
    n_mp_cens = int(ev[(ev.Treatment == "6-MP") & (ev.Event == 0)].shape[0])
    _check("6-MP events (exp 9)", n_mp_event, 9, 0, results)
    _check("6-MP censored (exp 12)", n_mp_cens, 12, 0, results)
    _check("Placebo events (exp 21)", int(ev[ev.Treatment == "Placebo"].Event.sum()), 21, 0, results)

    # 2. Median survival
    med = S.median_survival_table(ev).set_index("Treatment")
    _check("Median 6-MP (exp 23)", med.loc["6-MP", "Median_Survival"], 23.0, 0.0, results)
    _check("Median Placebo (exp 8)", med.loc["Placebo", "Median_Survival"], 8.0, 0.0, results)

    # 3. Log-rank (Mantel-Cox)
    lr = S.logrank_overall(ev)
    _check("Log-rank chi-square (exp 16.79)", lr["chisq"], 16.79, 0.02, results)
    _check("Log-rank p-value (exp <1e-4)", lr["p_value"], 4.2e-5, 5e-6, results)

    # 4. Gehan-Breslow generalized Wilcoxon
    gb = S.logrank_overall(ev, weight_type="gb")
    _check("Gehan-Breslow chi-square (exp 13.46)", gb["chisq"], 13.46, 0.05, results)

    # 5. Cox PH (treatment only; Breslow ties — statsmodels default)
    cox = S.cox_ph(ev, covariates=["Treatment"], interaction=False)
    row = cox["summary"].iloc[0]
    _check("Cox beta Placebo (exp 1.509)", row["log_HR"], 1.509, 0.01, results)
    _check("Cox HR Placebo (exp 4.523)", row["HR"], 4.523, 0.02, results)

    # ── Report ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("DroPhenix survival.py validation vs PUBLISHED Freireich (1963) 6-MP data")
    print("=" * 78)
    print(f"{'Check':<40}{'Published':>12}{'Engine':>12}{'Pass':>7}")
    print("-" * 78)
    all_ok = True
    for name, exp, got, tol, ok in results:
        all_ok &= ok
        print(f"{name:<40}{exp:>12}{str(got):>12}{'OK' if ok else 'FAIL':>7}")
    print("-" * 78)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
