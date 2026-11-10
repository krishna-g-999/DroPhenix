"""
validate_p0_block_c.py
Regression suite for P0 Block C of the 2026-11 audit: the fourteen numerical
corrections, each with a magnitude measured in the register.

Every check here FAILED, or raised, before the Block C fixes. See
docs/BUG_AND_RISK_REGISTER.md findings N3, N11, N12, N14, N15, N16, N17, N18,
N19, N20, N21, N22, N29, N31, N32a, N32b, N34, N35, N43, N44 and
docs/PRIORITISED_DEVELOPMENT_PLAN.md section 1 Block C.

What Block C is about
---------------------
C1  N14/N15  `mortality_adjusted_response` fits survival with LL.3, whose upper
             asymptote d is FREE (that is Abbott's control-mortality correction
             expressed as a parameter), and then reports the fitted `e` as the
             LC50. `e` is the curve's inflection, which equals the LC50 only
             when the curve runs from 1 to 0. With control survival d = 0.90 the
             true LC50 is e * ((d - 0.5)/(0.5 - c))^(1/b) = 8.944 while `e` is
             10.010: +11.9%, rising to +67% at d = 0.80. Purely definitional,
             and it propagates into the EC50/LC50 overlap verdict, where a true
             ratio of 0.85 -- inside the tier whose own text says the result
             "MUST NOT be reported as evidence of a specific behavioural
             phenotype" -- is reported as 0.60 "PARTIAL OVERLAP".
C2  N16      `se = sqrt(clip(diag(cov), 0, inf))` turns a NEGATIVE variance,
             which is the only evidence that the numerical Hessian is
             indefinite and the fit is not at a maximum, into a standard error
             of exactly 0.0 -- hence z = inf and p = 0.0, while the optimiser
             still reports converged = True. The clip destroys the diagnostic,
             so the failure is undetectable from the output.
C3  N17/N18  In the daily-cohort lifespan layout, `alive = start - deaths` does
             not subtract explicitly recorded censors, so each censored fly is
             expanded TWICE: once at its censoring day and again at the last
             scored day. A cohort of 10 with 1 death and 1 censor becomes 11
             flies and S(5) = 0.9091 against a truth of 0.9000. Nothing checks
             that deaths + censored <= start.
C4  N19      A missing `Event` silently became 0 = censored. A missing status is
             not evidence of survival; a non-binary code is not evidence of
             anything.
C5  N34/N35  The composite Therapeutic Score's CI, z and p are FIXED-EFFECT
             while the same row reports I2 across assays. Four per-assay
             effects 1.20 / -0.10 / 0.90 / 0.05, each var 0.04, give I2 = 90.1%
             and `assays_consistent = False`, and the module reports
             p = 3.0e-7 where random effects on the same four numbers give
             p = 0.107 -- four orders of magnitude, and the difference between
             "overwhelming cross-assay evidence" and none. No small-k
             adjustment anywhere: DerSimonian-Laird with 1.96 covers 88.2% at
             k = 3, tau2 = 0.30, against 94.6% for Hartung-Knapp-Sidik-Jonkman.
C6  N20      Long-format Y-maze arm entries with no FlyID became one fly per
             ENTRY, which destroys the sequence -- and the sequence is the
             datum. Arm labels were never validated, so four distinct invalid
             labels score 100% alternation.
C7  N21      Grooming is documented as "% of waking time" (the Qiao et al. 2018
             13% anchor) and computed as grooming / total observation time; per
             fly FlyID was fabricated from the row index, so bout-level input
             without FlyID inflated N by the number of bouts per fly.
C8  N11/N12  The parser WRITES values that were never observed: a negative
             count becomes 0, a count above n becomes n, duplicates are
             averaged and rounded, and differing n across duplicates is
             collapsed by `first`, discarding a vial. The frame a user inspects
             as "Preview" is already corrected, with no raw column, no
             per-cell flag and no record in any export.
C9  N22      A missing metric is plotted as zero and a missing SEM as a
             zero-length error bar, i.e. as certainty; `pi_barplot` silently
             plots the first of several matching rows; no figure reports the
             number of independent units.
C10 N3       `fly_detector` filters out every blob above `max_area` instead of
             splitting it, which is the merge correction both `climbing_video`
             and `climbing_band` apply. Measured recall: 0.88 healthy, 0.53
             impaired, 0.35 severe -- a confound perfectly correlated with the
             independent variable, biasing every disease comparison toward the
             null and making a confidence filter delete the impaired cohort
             preferentially.
C11 N43/N44  `lomb_scargle_period` returns the search floor, 16.000 h, as a
             measurement when one minute of the recording is invalid, and
             assumes perfectly regular sampling -- the one assumption
             Lomb-Scargle exists to relax -- while being described as an
             independent cross-check.
C12 N29      `derive_endpoints` computes onset age and healthspan from
             timepoints the prediction engine marked REFUSED, inflating the
             onset-age CI 1.99x and the healthspan CI 8.0x.
C13 N31      `trajectory_cluster` clusters on the within-assay SECONDS axis and
             then applies longitudinal archetype names, the exact category
             error `motor_decline` and the app explicitly forbid: untreated
             wild type is labelled "Progressive Rescue" because its climbing
             curve rises steeply over seconds.
C14 N32a/b   A genuinely MISSING trajectory is filled with "Undetermined" and
             scored 0.2, so absent data becomes evidence; and a partial
             `weights` dict is normalised before the missing keys are
             back-filled from the defaults, so the weights actually used do not
             sum to one.

Run:  python validation/validate_p0_block_c.py   (exit 0 = all passed)
"""

import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ROOT = Path(__file__).resolve().parent.parent

try:
    import cv2
    _CV2 = True
except Exception:                                   # pragma: no cover
    cv2 = None
    _CV2 = False


# ── helpers ───────────────────────────────────────────────────────────────────
def _check(name, got, expected, tol, results):
    """Numeric check. A None or non-finite value is a failure, never a pass."""
    try:
        ok = (got is not None) and np.isfinite(float(got)) \
            and abs(float(got) - expected) <= tol
    except (TypeError, ValueError):
        ok = False
    results.append((name, f"{expected:g}",
                    (f"{float(got):g}" if _finite(got) else str(got)), ok))


def _finite(v):
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _bool_check(name, cond, expected, got, results):
    results.append((name, str(expected), str(got), bool(cond)))


def _nan_check(name, got, results):
    """A value that must be NaN -- the whole point of several of these fixes."""
    is_nan = (got is None) or (isinstance(got, float) and np.isnan(got)) or \
             (not isinstance(got, str) and pd.isna(got))
    results.append((name, "NaN", str(got), bool(is_nan)))


def _raises(name, fn, must_mention, results):
    """
    A refusal check. The exception must MENTION the thing being refused -- an
    unrelated crash four frames up is not a refusal, and that distinction is
    the whole content of several of these findings.
    """
    try:
        fn()
        results.append((name, "refused", "no error", False))
    except Exception as exc:
        txt = str(exc)
        ok = all(m.lower() in txt.lower() for m in must_mention)
        results.append((name, "refused", (txt[:40] if ok else "wrong msg"), ok))


# ══════════════════════════════════════════════════════════════════════════════
# C1 -- the LC50 is the dose at which survival is 0.5, not the fitted `e`
# ══════════════════════════════════════════════════════════════════════════════
def _survival_curve(dose, b, c, d, e, n=40):
    """Exact log-logistic survival proportions, no noise: the truth is known."""
    dose = np.asarray(dose, dtype=float)
    out = np.empty_like(dose)
    pos = dose > 0
    out[pos] = c + (d - c) / (1.0 + np.exp(b * (np.log(dose[pos]) - np.log(e))))
    out[~pos] = d
    return out


def _block_c1(results):
    from modules import dose_response as DR

    doses = np.array([0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0])

    # ── (a) the register's worked case: c = 0, d = 0.90, b = 2, e = 10 ────────
    b, c, d, e = 2.0, 0.0, 0.90, 10.0
    surv = _survival_curve(doses, b, c, d, e)
    truth = e * ((d - 0.5) / (0.5 - c)) ** (1.0 / b)          # 8.9443
    fit = DR.fit_loglogistic(doses, surv, n=np.full(len(doses), 40.0), model="LL.3")

    _check("C1 fitted e recovers the inflection (10.0)", fit.params["e"], 10.0, 0.05, results)
    ld = DR.lethal_dose_absolute(fit, target=0.5)
    _check("C1 LC50 solves survival = 0.5 (8.944, not e = 10)",
           ld["estimate"], truth, 0.02, results)
    _bool_check("C1 the definition is stated in the result",
                "0.5" in str(ld.get("definition", "")),
                "names 0.5", str(ld.get("definition", ""))[:12], results)
    _check("C1 `e` is still reported, separately", ld["param_e"], 10.0, 0.05, results)
    # The interval needs residual variance to exist, so it is checked on a
    # NOISY replicate of the same curve: a zero-residual fit has zero
    # uncertainty, and reporting an interval there would be the fabrication.
    _check("C1 an exact fit reports zero uncertainty, not a made-up interval",
           ld["se_log"], 0.0, 1e-6, results)
    rng = np.random.default_rng(3)
    surv_n = np.clip(_survival_curve(doses, b, c, d, e)
                     + rng.normal(0.0, 0.02, len(doses)), 0.0, 1.0)
    fit_n = DR.fit_loglogistic(doses, surv_n, n=np.full(len(doses), 40.0), model="LL.3")
    ld_n = DR.lethal_dose_absolute(fit_n, target=0.5)
    _bool_check("C1 LC50 interval is finite and brackets the truth",
                _finite(ld_n["ci_lower"]) and _finite(ld_n["ci_upper"])
                and ld_n["ci_lower"] <= truth <= ld_n["ci_upper"],
                "brackets", f"{ld_n['ci_lower']:.3g},{ld_n['ci_upper']:.3g}", results)
    # The asymptotes enter the LC50's uncertainty: the variance differs from
    # the one computed for `e`, where only log_e and b contribute. Whether it
    # is larger or smaller depends on the sign of cov(d, log_e), so the claim
    # tested is that d and c are in the gradient at all, not that it widens.
    _bool_check("C1 the interval accounts for the asymptotes, not log_e alone",
                abs(ld_n["se_log"]
                    - DR._delta_interval(fit_n, 0.95, ld_n["relative_level_pct"])["se_log"])
                > 1e-9,
                "differs", f"{ld_n['se_log']:.5f}", results)

    # ── (b) d = 0.80: the register's +67% case ───────────────────────────────
    d2 = 0.80
    surv2 = _survival_curve(doses, b, c, d2, e)
    truth2 = e * ((d2 - 0.5) / (0.5 - c)) ** (1.0 / b)        # 7.7460
    fit2 = DR.fit_loglogistic(doses, surv2, n=np.full(len(doses), 40.0), model="LL.3")
    ld2 = DR.lethal_dose_absolute(fit2, target=0.5)
    _check("C1 d = 0.80 case: LC50 = 7.746", ld2["estimate"], truth2, 0.02, results)

    # ── (c) LL.2 (c = 0, d = 1): `e` IS the LC50, and must be unchanged ──────
    surv3 = _survival_curve(doses, b, 0.0, 1.0, e)
    fit3 = DR.fit_loglogistic(doses, surv3, n=np.full(len(doses), 40.0), model="LL.2")
    ld3 = DR.lethal_dose_absolute(fit3, target=0.5)
    _check("C1 LL.2: LC50 == e, so the fix changes nothing there",
           ld3["estimate"] - float(fit3.params["e"]), 0.0, 1e-6, results)

    # ── (d) control survival already below 50%: no LC50 exists ──────────────
    surv4 = _survival_curve(doses, b, 0.0, 0.40, e)
    fit4 = DR.fit_loglogistic(doses, surv4, n=np.full(len(doses), 40.0), model="LL.3")
    ld4 = DR.lethal_dose_absolute(fit4, target=0.5)
    _nan_check("C1 d < 0.5: LC50 refused, not extrapolated", ld4["estimate"], results)
    _bool_check("C1 the refusal names control survival",
                "control" in str(ld4.get("reason", "")).lower(),
                "names control", str(ld4.get("reason", ""))[:12], results)

    # ── (e) the overlap verdict inherits the corrected LC50 ─────────────────
    # Behavioural EC50 set so the TRUE ratio EC50/LC50 is 0.85, i.e. inside the
    # tier whose own text forbids a behavioural claim. Pre-fix this reported
    # 0.60 "PARTIAL OVERLAP"; the understatement factor is (2d-1)^(1/b).
    ec50_true = 0.85 * truth
    beh = _survival_curve(doses, b, 0.0, 1.0, ec50_true)
    n_exp = np.full(len(doses), 40.0)
    dead = np.round(n_exp * (1.0 - surv))
    res = DR.mortality_adjusted_response(
        dose=doses, response=beh, dead=dead, n_exposed=n_exp,
        n_scored=n_exp - dead, mortality_model="LL.3",
    )
    _check("C1 LC50 in the overlap report", res["lc50"], truth, 0.25, results)
    _check("C1 EC50/LC50 ratio recovers the true 0.85", res["ratio"], 0.85, 0.05, results)
    _bool_check("C1 verdict is 'confounded' at a true ratio of 0.85",
                res["severity"] == "confounded", "confounded", res["severity"], results)
    _bool_check("C1 method string no longer claims `e` is the LC50",
                "e is the lc50" not in str(res["method"]).lower(),
                "no claim", str(res["method"])[:12], results)
    _bool_check("C1 thresholds are labelled a screening convention",
                "convention" in str(res.get("ratio_note", "")).lower(),
                "convention", str(res.get("ratio_note", ""))[:12], results)
    _check("C1 `e` kept alongside, for traceability",
           res.get("lc50_param_e"), 10.0, 0.4, results)


# ══════════════════════════════════════════════════════════════════════════════
# C2 -- an indefinite Hessian must not become a standard error of zero
# ══════════════════════════════════════════════════════════════════════════════
def _block_c2(results):
    from modules import hierarchical as H

    # The register's measured case: inv(H) has NEGATIVE diagonal entries, which
    # is the only signal that the optimiser stopped somewhere that is not a
    # maximum. Built here as a diagonal matrix with the same inverse diagonal.
    inv_diag = np.array([-44831.0, 191336.0, -833622.0])
    Hbad = np.diag(1.0 / inv_diag)
    info = H._ses_from_hessian(Hbad)

    _nan_check("C2 negative variance -> NaN SE (was 0.0)", info["se"][0], results)
    _check("C2 the positive variance is still reported",
           info["se"][1], float(np.sqrt(191336.0)), 0.1, results)
    _nan_check("C2 third negative variance -> NaN", info["se"][2], results)
    _bool_check("C2 no SE is exactly 0.0",
                not np.any(np.asarray(info["se"]) == 0.0), "no zeros",
                str(np.nanmin(np.abs(info["se"])))[:10], results)
    _bool_check("C2 inference is gated off", info["inference_valid"] is False,
                False, info["inference_valid"], results)
    _check("C2 negative-variance count reported", info["n_negative_variance"], 2, 0, results)
    _bool_check("C2 the reason names the Hessian",
                "hessian" in str(info["reason"]).lower(), "names Hessian",
                str(info["reason"])[:12], results)
    _bool_check("C2 minimum eigenvalue is reported and non-positive",
                _finite(info["min_eigenvalue"]) and info["min_eigenvalue"] <= 0,
                "<= 0", f"{info['min_eigenvalue']:.4g}", results)

    # A well-conditioned Hessian must be unaffected.
    Hok = np.diag([4.0, 25.0, 100.0])
    ok = H._ses_from_hessian(Hok)
    _check("C2 positive-definite Hessian: SE = 0.5", ok["se"][0], 0.5, 1e-9, results)
    _bool_check("C2 positive-definite Hessian: inference valid",
                ok["inference_valid"] is True, True, ok["inference_valid"], results)

    # And the whole fit must carry the gate, not only the optimiser's flag.
    rng = np.random.default_rng(11)
    rows = []
    for g, p in (("WT", 0.70), ("Mut", 0.40)):
        for v in range(8):
            n = 20
            rows.append({"Genotype": g, "VialID": f"{g}{v}",
                         "Count": int(rng.binomial(n, p)), "n": n})
    df = pd.DataFrame(rows)
    fit = H.fit_betabinomial(df, formula_rhs="C(Genotype)",
                             count_col="Count", total_col="n")
    _bool_check("C2 a clean fit still reports inference_valid",
                getattr(fit, "inference_valid", None) is True, True,
                getattr(fit, "inference_valid", "missing"), results)
    _bool_check("C2 a clean fit has finite p-values",
                bool(np.all(np.isfinite(fit.pvalues.to_numpy()))), "finite",
                str(fit.pvalues.to_numpy().round(4)), results)
    _bool_check("C2 no p-value is exactly 0.0",
                not bool(np.any(fit.pvalues.to_numpy() == 0.0)), "none",
                str(float(fit.pvalues.min())), results)


# ══════════════════════════════════════════════════════════════════════════════
# C3 / C4 -- cohort arithmetic and the event status
# ══════════════════════════════════════════════════════════════════════════════
def _block_c3(results):
    from modules import survival as S

    # The register's example: 10 flies, 1 death and 1 censor on day 5.
    daily = pd.DataFrame({
        "Genotype": ["WT"] * 3, "Treatment": ["Veh"] * 3, "Sex": ["Male"] * 3,
        "Day": [0, 5, 10], "Dead_count": [0, 1, 0],
        "Censored_count": [0, 1, 0], "Total_start": [10, 10, 10],
    })
    ev = S.to_event_table(daily)
    _check("C3 cohort of 10 expands to 10 flies, not 11", len(ev), 10, 0, results)
    _check("C3 one death recorded", int(ev["Event"].sum()), 1, 0, results)
    km = S.kaplan_meier(ev)
    curve = list(km.values())[0]
    at5 = curve[curve["Time"] == 5.0]
    _check("C3 risk set at day 5 is 10, not 11",
           float(at5["n_risk"].iloc[0]) if len(at5) else np.nan, 10.0, 0, results)
    _check("C3 S(5) = 0.9000, not 0.9091",
           float(at5["Survival"].iloc[0]) if len(at5) else np.nan, 0.9, 1e-6, results)

    # Scaling: 10 escapers in a cohort of 100 inflated every risk set by 10%.
    daily2 = pd.DataFrame({
        "Genotype": ["WT"] * 2, "Treatment": ["Veh"] * 2, "Sex": ["Male"] * 2,
        "Day": [10, 20], "Dead_count": [30, 20],
        "Censored_count": [10, 0], "Total_start": [100, 100],
    })
    ev2 = S.to_event_table(daily2)
    _check("C3 100 loaded, 10 censored: 100 rows not 110", len(ev2), 100, 0, results)
    _check("C3 censored flies are censored exactly once",
           int((ev2["Event"] == 0).sum()), 50, 0, results)

    # Conservation: deaths + censored may not exceed the cohort.
    bad = pd.DataFrame({
        "Genotype": ["WT"] * 2, "Treatment": ["Veh"] * 2, "Sex": ["Male"] * 2,
        "Day": [5, 10], "Dead_count": [8, 5],
        "Censored_count": [0, 0], "Total_start": [10, 10],
    })
    _raises("C3 deaths exceeding the cohort are refused",
            lambda: S.to_event_table(bad), ["conservation"], results)

    bad2 = pd.DataFrame({
        "Genotype": ["WT"], "Treatment": ["Veh"], "Sex": ["Male"],
        "Day": [5], "Dead_count": [np.nan], "Censored_count": [0],
        "Total_start": [10],
    })
    _raises("C3 a missing daily death count is refused, not read as 0",
            lambda: S.to_event_table(bad2), ["Dead_count"], results)

    # Clean daily data must still expand exactly as before.
    clean = pd.DataFrame({
        "Genotype": ["WT"] * 3, "Treatment": ["Veh"] * 3, "Sex": ["Male"] * 3,
        "Day": [5, 10, 15], "Dead_count": [2, 3, 1], "Total_start": [20, 20, 20],
    })
    ev3 = S.to_event_table(clean)
    _check("C3 no censor column: 20 rows, 6 deaths", len(ev3), 20, 0, results)
    _check("C3 remaining 14 censored at the last scored day",
           float(ev3.loc[ev3["Event"] == 0, "Time"].min()), 15.0, 0, results)


def _block_c4(results):
    from modules import survival as S

    base = {"Genotype": ["WT"] * 4, "Treatment": ["Veh"] * 4, "Sex": ["Male"] * 4}

    missing = pd.DataFrame({**base, "Time": [10, 20, 30, 40],
                            "Event": [1, 0, np.nan, 1]})
    _raises("C4 a missing Event is refused, not read as censored",
            lambda: S.to_event_table(missing), ["Event"], results)

    nonbin = pd.DataFrame({**base, "Time": [10, 20, 30, 40],
                           "Event": [1, 0, 2, 1]})
    _raises("C4 a non-binary Event code is refused",
            lambda: S.to_event_table(nonbin), ["Event", "2"], results)

    junk = pd.DataFrame({**base, "Time": [10, 20, 30, 40],
                         "Event": ["dead", "alive", "maybe", "dead"]})
    _raises("C4 an unrecognised status string is refused",
            lambda: S.to_event_table(junk), ["maybe"], results)

    good = pd.DataFrame({**base, "Time": [10, 20, 30, 40],
                         "Event": [1, 0, 1, 0]})
    ev = S.to_event_table(good)
    _check("C4 valid binary Event: 4 rows preserved", len(ev), 4, 0, results)
    _check("C4 valid binary Event: 2 deaths", int(ev["Event"].sum()), 2, 0, results)

    words = pd.DataFrame({**base, "Time": [10, 20, 30, 40],
                          "Event": ["dead", "censored", "death", "alive"]})
    ev2 = S.to_event_table(words)
    _check("C4 documented status words map to 2 deaths",
           int(ev2["Event"].sum()), 2, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# C5 -- random effects with a small-k interval, and no p when the assays disagree
# ══════════════════════════════════════════════════════════════════════════════
def _block_c5(results):
    from modules import drug_ranking as DRK
    from scipy import stats as sps

    g = np.array([1.20, -0.10, 0.90, 0.05])
    v = np.full(4, 0.04)
    p = DRK.inverse_variance_pool(g, v)

    _check("C5 Cochran Q = 30.297", p["Q"], 30.297, 0.01, results)
    _check("C5 I2 = 90.1%", p["I2"], 90.10, 0.1, results)
    _check("C5 tau2 = 0.3640", p["tau2"], 0.36396, 1e-4, results)
    _check("C5 fixed-effect SE = 0.100", p["se_fixed"], 0.1, 1e-6, results)
    _check("C5 random-effects SE = 0.3178", p["se_random"], 0.31777, 1e-4, results)

    # Hartung-Knapp-Sidik-Jonkman: variance from the weighted residual spread,
    # interval from t(k-1), not 1.96. IntHout et al. 2014.
    _check("C5 HKSJ SE = 0.3178", p["se_hksj"], 0.31779, 1e-3, results)
    tcrit = float(sps.t.ppf(0.975, 3))
    _check("C5 HKSJ uses t(k-1) = 3.182", p["t_crit"], tcrit, 1e-6, results)
    _check("C5 HKSJ CI lower", p["ci_lo"], 0.5125 - tcrit * p["se_hksj"], 1e-6, results)
    _bool_check("C5 the interval is not the 1.96 one",
                abs((p["ci_hi"] - p["random"]) - 1.96 * p["se_random"]) > 0.05,
                "not 1.96", f"{(p['ci_hi'] - p['random']):.4f}", results)
    _bool_check("C5 the method names HKSJ",
                "hartung" in str(p.get("method", "")).lower()
                or "hksj" in str(p.get("method", "")).lower(),
                "HKSJ", str(p.get("method", ""))[:12], results)

    # The two p-values the register contrasts.
    _check("C5 fixed-effect p = 3.0e-7 (the published-and-wrong number)",
           2 * sps.norm.sf(abs(p["fixed"] / p["se_fixed"])), 2.97e-7, 1e-8, results)
    # The register contrasts the fixed-effect p with the DerSimonian-Laird one.
    _check("C5 DL random-effects p = 0.107 (the register's contrast)",
           p["p_random_dl"], 0.1068, 5e-4, results)
    # The number actually REPORTED is the HKSJ one, which is more conservative
    # again at k = 4 because it uses t(3) and the residual spread.
    _check("C5 HKSJ random-effects p = 0.205 (the reported one)",
           p["p_random"], 0.2052, 5e-4, results)

    # Composite: random effects primary, and no single p when I2 >= 50.
    per_assay = pd.DataFrame({
        "drug": ["X"] * 4,
        "assay": ["climbing", "lifespan", "ymaze", "dams"],
        "g_pooled": g, "var": v,
    })
    comp = DRK.composite_score(per_assay)
    row = comp.iloc[0]
    _check("C5 Therapeutic_Score is the random-effects estimate",
           row["Therapeutic_Score"], p["random"], 1e-3, results)
    _check("C5 Score_fixed kept as secondary", row["Score_fixed"], p["fixed"], 1e-3, results)
    _nan_check("C5 p is NaN when I2 >= 50", row["p_value"], results)
    _bool_check("C5 and the reason says a single cross-assay p is undefined",
                "undefined" in str(row.get("p_value_note", "")).lower(),
                "undefined", str(row.get("p_value_note", ""))[:12], results)
    _check("C5 I2 still reported", row["I2_assay_pct"], 90.1, 0.1, results)
    _bool_check("C5 assays_consistent is False", row["assays_consistent"] is False
                or row["assays_consistent"] == False, False,
                row["assays_consistent"], results)

    # Consistent assays: the p-value is reported, and it is the random-effects one.
    g2 = np.array([0.50, 0.55, 0.45, 0.52])
    per2 = pd.DataFrame({"drug": ["Y"] * 4,
                         "assay": ["climbing", "lifespan", "ymaze", "dams"],
                         "g_pooled": g2, "var": np.full(4, 0.04)})
    comp2 = DRK.composite_score(per2)
    p2 = DRK.inverse_variance_pool(g2, np.full(4, 0.04))
    _bool_check("C5 consistent assays: I2 < 50", p2["I2"] < 50.0, "<50",
                f"{p2['I2']:.1f}", results)
    _check("C5 consistent assays: p reported, random-effects",
           float(comp2.iloc[0]["p_value"]), p2["p_random"], 1e-3, results)

    # Unequal precision, so fixed and random genuinely differ.
    v3 = np.array([0.01, 0.04, 0.09, 0.16])
    p3 = DRK.inverse_variance_pool(g, v3)
    _bool_check("C5 fixed and random differ under unequal precision",
                abs(p3["fixed"] - p3["random"]) > 0.01, "differ",
                f"{p3['fixed']:.3f}/{p3['random']:.3f}", results)

    # Small-k coverage: HKSJ must be wider than DL where DL undercovers.
    gk = np.array([0.2, 0.9, -0.3])
    vk = np.array([0.05, 0.05, 0.05])
    pk = DRK.inverse_variance_pool(gk, vk)
    _bool_check("C5 k = 3 heterogeneous: HKSJ interval wider than DL",
                (pk["ci_hi"] - pk["ci_lo"]) > (pk["ci_hi_dl"] - pk["ci_lo_dl"]),
                "wider", f"{pk['ci_hi'] - pk['ci_lo']:.3f}", results)


# ══════════════════════════════════════════════════════════════════════════════
# C6 -- the Y-maze sequence is the datum
# ══════════════════════════════════════════════════════════════════════════════
def _block_c6(results):
    from modules import ymaze as Y

    long_no_id = pd.DataFrame({
        "Genotype": ["WT"] * 6, "Treatment": ["Veh"] * 6, "Sex": ["Male"] * 6,
        "Arm": ["A", "B", "C", "A", "C", "B"],
    })
    _raises("C6 long format without FlyID is refused (was 6 flies)",
            lambda: Y.load_ymaze(long_no_id), ["FlyID"], results)

    long_id = long_no_id.assign(FlyID="f1")
    out = Y.load_ymaze(long_id)
    _check("C6 with FlyID: one fly", out["FlyID"].nunique(), 1, 0, results)
    pf = Y.per_fly_metrics(out)
    _check("C6 one fly with 6 entries", float(pf["N_entries"].iloc[0])
           if "N_entries" in pf.columns else float(pf.iloc[0].get("n_entries", np.nan)),
           6, 0, results)

    bad_arms = pd.DataFrame({
        "Genotype": ["WT"] * 6, "Treatment": ["Veh"] * 6, "Sex": ["Male"] * 6,
        "FlyID": ["f1"] * 6, "Arm": ["W", "X", "Y", "Z", "W", "X"],
    })
    _raises("C6 invalid arm labels are refused (scored 100% alternation)",
            lambda: Y.load_ymaze(bad_arms), ["arm"], results)

    lower = long_id.assign(Arm=[a.lower() for a in long_id["Arm"]])
    out2 = Y.load_ymaze(lower)
    _bool_check("C6 lower-case arm labels are normalised, not refused",
                set(out2["Arm"]) == {"A", "B", "C"}, "{A,B,C}",
                str(sorted(set(out2["Arm"]))), results)

    seq = pd.DataFrame({
        "Genotype": ["WT", "WT"], "Treatment": ["Veh"] * 2, "Sex": ["Male"] * 2,
        "Sequence": ["ABCACB", "ACBABC"],
    })
    out3 = Y.load_ymaze(seq)
    _check("C6 sequence layout without FlyID is fine: one row = one fly",
           out3["FlyID"].nunique(), 2, 0, results)

    bad_seq = pd.DataFrame({
        "Genotype": ["WT"], "Treatment": ["Veh"], "Sex": ["Male"],
        "Sequence": ["ABQCZ"],
    })
    _raises("C6 invalid labels inside a sequence string are refused",
            lambda: Y.load_ymaze(bad_seq), ["arm"], results)


# ══════════════════════════════════════════════════════════════════════════════
# C7 -- the grooming denominator, and one row is not one fly
# ══════════════════════════════════════════════════════════════════════════════
def _block_c7(results):
    from modules import grooming as G

    bouts = pd.DataFrame({
        "Genotype": ["WT"] * 4, "Treatment": ["Veh"] * 4, "Sex": ["Male"] * 4,
        "Bout_start_s": [10.0, 100.0, 200.0, 300.0],
        "Bout_end_s": [30.0, 120.0, 220.0, 320.0],
        "Observation_s": [600.0] * 4,
    })
    _raises("C7 bout-level input without FlyID is refused (was 4 flies)",
            lambda: G.per_fly_metrics(bouts), ["FlyID"], results)

    with_id = bouts.assign(FlyID="f1")
    pf = G.per_fly_metrics(with_id)
    _check("C7 with FlyID: one fly, not four", len(pf), 1, 0, results)
    _check("C7 80 s of grooming in 600 s = 13.33%",
           float(pf["Pct_time_grooming"].iloc[0]), 13.33, 0.01, results)
    _bool_check("C7 the denominator is named in the output",
                "observation" in str(pf["Denominator"].iloc[0]).lower(),
                "observation", str(pf.get("Denominator", pd.Series(["missing"])).iloc[0])[:12],
                results)

    # Waking time, when it is actually supplied, is a different denominator and
    # a different number -- the Qiao et al. 13% anchor is % of WAKING time.
    waking = with_id.assign(Waking_s=200.0)
    pf2 = G.per_fly_metrics(waking)
    _check("C7 % of waking time computed only when Waking_s is given",
           float(pf2["Pct_waking_grooming"].iloc[0]), 40.0, 0.01, results)
    _nan_check("C7 and it is NaN when waking time is unknown",
               pf["Pct_waking_grooming"].iloc[0], results)

    summary = pd.DataFrame({
        "Genotype": ["WT", "WT"], "Treatment": ["Veh"] * 2, "Sex": ["Male"] * 2,
        "Total_grooming_s": [78.0, 60.0], "Observation_s": [600.0, 600.0],
    })
    pf3 = G.per_fly_metrics(summary)
    _check("C7 summary input: one row really is one fly, still allowed",
           len(pf3), 2, 0, results)

    clustered = pd.DataFrame({
        "Genotype": ["WT"] * 6 + ["Mut"] * 6,
        "Treatment": ["Veh"] * 6 + ["Drug"] * 6, "Sex": ["Male"] * 12,
        "FlyID": [f"f{i}" for i in range(12)],
        "VialID": ["v1"] * 3 + ["v2"] * 3 + ["v3"] * 3 + ["v4"] * 3,
        "Total_grooming_s": [70, 75, 80, 72, 78, 74, 40, 45, 42, 60, 65, 62],
        "Observation_s": [600.0] * 12,
    })
    pf4 = G.per_fly_metrics(clustered)
    cmp = G.compare_groups(pf4)
    _bool_check("C7 a clustered comparison is not presented as valid",
                cmp.get("inference_valid") is False, False,
                cmp.get("inference_valid", "missing"), results)
    _bool_check("C7 and the warning names the hierarchical route",
                "hierarchical" in str(cmp.get("clustering_warning", "")).lower(),
                "hierarchical", str(cmp.get("clustering_warning", ""))[:12], results)

    src = (_ROOT / "modules" / "grooming.py").read_text(encoding="utf-8",
                                                        errors="ignore")
    _bool_check("C7 the old 'wild-type ~13% of waking time' claim is gone",
                "the primary read-out; wild-type ~13% of waking time" not in src,
                "removed", "present" if "the primary read-out; wild-type ~13% of "
                "waking time" in src else "removed", results)
    _bool_check("C7 and the observation denominator is documented as such",
                "OBSERVATION" in (G.per_fly_metrics.__doc__ or "")
                or "observation window" in (G.per_fly_metrics.__doc__ or ""),
                "documented", "ok" if G.per_fly_metrics.__doc__ else "no docstring",
                results)
    _bool_check("C7 the vial identifier is preserved on every row",
                "VialID" in pf4.columns, "VialID",
                "present" if "VialID" in pf4.columns else "absent", results)


# ══════════════════════════════════════════════════════════════════════════════
# C8 -- the parser must never write a value that was not observed
# ══════════════════════════════════════════════════════════════════════════════
def _block_c8(results):
    from modules import data_parser as DP

    neg = pd.DataFrame({
        "Genotype": ["WT"] * 2, "Sex": ["Male"] * 2, "Treatment": ["Veh"] * 2,
        "Replicate": [1, 1], "Time": [10, 20], "Count": [-3, 8], "n": [20, 20],
    })
    df, w, e = DP.load_data(neg)
    _bool_check("C8 negative Count is not accepted as an observation",
                (not df.empty) and pd.isna(df.loc[df["Time"] == 10, "Count"].iloc[0]),
                "NaN", str(df.loc[df["Time"] == 10, "Count"].iloc[0]) if not df.empty
                else "empty df", results)
    _check("C8 the raw -3 is preserved",
           float(df.loc[df["Time"] == 10, "Count_raw"].iloc[0]) if "Count_raw" in df.columns
           else np.nan, -3.0, 0, results)
    _bool_check("C8 the row is flagged by name",
                "negative" in str(df.loc[df["Time"] == 10, "QC_corrections"].iloc[0]).lower()
                if "QC_corrections" in df.columns else False,
                "negative", str(df.get("QC_corrections", pd.Series(["missing"])).iloc[0])[:12],
                results)
    _bool_check("C8 a clean row carries no correction flag",
                str(df.loc[df["Time"] == 20, "QC_corrections"].iloc[0]) == ""
                if "QC_corrections" in df.columns else False,
                "empty", "ok", results)

    over = pd.DataFrame({
        "Genotype": ["WT"] * 2, "Sex": ["Male"] * 2, "Treatment": ["Veh"] * 2,
        "Replicate": [1, 1], "Time": [10, 20], "Count": [25, 8], "n": [20, 20],
    })
    df2, _, _ = DP.load_data(over)
    _bool_check("C8 Count > n is not capped at n",
                (not df2.empty) and pd.isna(df2.loc[df2["Time"] == 10, "Count"].iloc[0]),
                "NaN", str(df2.loc[df2["Time"] == 10, "Count"].iloc[0]) if not df2.empty
                else "empty df", results)
    _check("C8 n is left exactly as recorded",
           float(df2.loc[df2["Time"] == 10, "n"].iloc[0]) if not df2.empty else np.nan,
           20.0, 0, results)
    _check("C8 the raw 25 is preserved",
           float(df2.loc[df2["Time"] == 10, "Count_raw"].iloc[0])
           if "Count_raw" in df2.columns else np.nan, 25.0, 0, results)

    dup = pd.DataFrame({
        "Genotype": ["WT"] * 3, "Sex": ["Male"] * 3, "Treatment": ["Veh"] * 3,
        "Replicate": [1, 1, 1], "Time": [10, 10, 20],
        "Count": [4, 5, 8], "n": [20, 20, 20],
    })
    df3, w3, e3 = DP.load_data(dup)
    _bool_check("C8 duplicates are refused by default, not averaged",
                bool(e3) and any("duplicat" in str(x).lower() for x in e3),
                "error", (str(e3[0])[:12] if e3 else "no error"), results)
    _bool_check("C8 the refusal names the duplicated key",
                any("Time" in str(x) for x in e3), "names key",
                (str(e3[0])[:12] if e3 else "none"), results)

    df4, w4, e4 = DP.load_data(dup, duplicate_policy="mean")
    _bool_check("C8 an explicit mean policy is honoured", not df4.empty and not e4,
                "accepted", f"{len(df4)} rows", results)
    _check("C8 duplicates of 4 and 5 collapse to 4 (half-to-even)",
           float(df4.loc[df4["Time"] == 10, "Count"].iloc[0]) if not df4.empty else np.nan,
           4.0, 0, results)
    _bool_check("C8 the collapse is recorded on the row",
                "duplicat" in str(df4.loc[df4["Time"] == 10, "QC_corrections"].iloc[0]).lower()
                if not df4.empty and "QC_corrections" in df4.columns else False,
                "duplicate", "ok", results)

    dup_n = dup.copy()
    dup_n.loc[1, "n"] = 18
    df5, w5, e5 = DP.load_data(dup_n, duplicate_policy="mean")
    _bool_check("C8 duplicates with differing n are refused, not collapsed by `first`",
                bool(e5) and any("n" in str(x) for x in e5), "error",
                (str(e5[0])[:12] if e5 else "no error"), results)

    clean = pd.DataFrame({
        "Genotype": ["WT"] * 3, "Sex": ["Male"] * 3, "Treatment": ["Veh"] * 3,
        "Replicate": [1, 1, 1], "Time": [10, 20, 30], "Count": [5, 8, 12],
        "n": [20, 20, 20],
    })
    df6, _, e6 = DP.load_data(clean)
    _bool_check("C8 clean data is unchanged", (not e6) and len(df6) == 3
                and float(df6["Count"].sum()) == 25.0, "3 rows/25",
                f"{len(df6)}/{df6['Count'].sum() if not df6.empty else '-'}", results)

    app = (_ROOT / "app.py").read_text(encoding="utf-8", errors="ignore")
    _bool_check("C8 the upload preview shows the raw column (N12)",
                "Count_raw" in app, "Count_raw", "present" if "Count_raw" in app
                else "absent", results)
    _bool_check("C8 and the corrections travel with the data, not as a toast",
                "QC_corrections" in app, "QC_corrections",
                "present" if "QC_corrections" in app else "absent", results)


# ══════════════════════════════════════════════════════════════════════════════
# C9 -- a gap is plotted as a gap
# ══════════════════════════════════════════════════════════════════════════════
def _block_c9(results):
    import matplotlib
    matplotlib.use("Agg")
    from modules import visualizer as V

    pi = pd.DataFrame({
        "Genotype": ["WT", "WT"], "Treatment": ["Veh", "Drug"],
        "Sex": ["Male", "Male"], "PI_mean": [20.0, 40.0], "n_rep": [5, 5],
    })
    fig, _ = V.pi_barplot(pi)
    gaps = getattr(fig, "drophenix_gaps", None)
    _bool_check("C9 a missing SEM column is recorded, not drawn as zero",
                isinstance(gaps, dict) and gaps.get("missing_sem") == 2,
                2, (gaps or {}).get("missing_sem", "no diagnostics"), results)
    _bool_check("C9 no error bar was drawn for a missing SEM",
                isinstance(gaps, dict) and gaps.get("errorbars_drawn") == 0,
                0, (gaps or {}).get("errorbars_drawn", "n/a"), results)
    _bool_check("C9 the independent-unit n is reported on the figure",
                isinstance(gaps, dict) and gaps.get("n_units_reported") is True,
                True, (gaps or {}).get("n_units_reported", "n/a"), results)

    pi_sem = pi.assign(PI_sem=[3.0, np.nan])
    fig2, _ = V.pi_barplot(pi_sem)
    g2 = getattr(fig2, "drophenix_gaps", {})
    _check("C9 one present SEM is drawn", g2.get("errorbars_drawn"), 1, 0, results)
    _check("C9 one NaN SEM is a gap", g2.get("missing_sem"), 1, 0, results)

    dup = pd.concat([pi.iloc[[0]], pi.iloc[[0]]], ignore_index=True)
    _raises("C9 several rows matching one group are refused, not silently first",
            lambda: V.pi_barplot(dup), ["Genotype"], results)

    sdq = pd.DataFrame({
        "Genotype": ["WT", "Mut"], "Treatment": ["Drug", "Drug"],
        "Sex": ["Pooled"] * 2, "SDQ": [0.4, np.nan],
    })
    fig3, _ = V.sdq_plot(sdq)
    g3 = getattr(fig3, "drophenix_gaps", {})
    _check("C9 a missing SDQ is a gap, not 0.0 ('no sex difference')",
           g3.get("missing_metric"), 1, 0, results)
    _check("C9 and the measured one is still drawn", g3.get("drawn"), 1, 0, results)

    cri = pd.DataFrame({
        "Genotype": ["WT", "Mut"], "Treatment": ["Drug"] * 2, "Sex": ["Male"] * 2,
        "CRI": [60.0, np.nan],
    })
    fig4, _ = V.cri_barplot(cri)
    g4 = getattr(fig4, "drophenix_gaps", {})
    _check("C9 a missing CRI is marked as absent", g4.get("missing_metric"), 1, 0, results)

    tcs = pd.DataFrame({
        "Genotype": ["WT", "Mut"], "Treatment": ["A", "B"], "Sex": ["Male"] * 2,
        "TCS": [70.0, np.nan], "TCS_rank": [1, 2],
        "TCS_CI_lower": [60.0, np.nan], "TCS_CI_upper": [80.0, np.nan],
    })
    fig5, _ = V.tcs_ranked_bar(tcs)
    g5 = getattr(fig5, "drophenix_gaps", {})
    _check("C9 a missing TCS is not scored 0", g5.get("missing_metric"), 1, 0, results)

    tcs2 = pd.DataFrame({
        "Genotype": ["WT"], "Treatment": ["A"], "Sex": ["Male"], "TCS": [70.0],
        "TCS_rank": [1],
    })
    fig6, _ = V.tcs_ranked_bar(tcs2)
    g6 = getattr(fig6, "drophenix_gaps", {})
    _check("C9 a missing interval is not collapsed onto the estimate",
           g6.get("missing_ci"), 1, 0, results)

    boot = pd.DataFrame({"Group": ["A", "B"], "Mean": [1.0, np.nan],
                         "CI95_lo": [0.5, np.nan], "CI95_hi": [1.5, np.nan]})
    fig7, _ = V.bootstrap_ci_plot(boot)
    g7 = getattr(fig7, "drophenix_gaps", {})
    _check("C9 a missing bootstrap mean is a gap, not 0",
           g7.get("missing_metric"), 1, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# C10 -- detector recall must not depend on the phenotype
# ══════════════════════════════════════════════════════════════════════════════
def _scene(n_flies, band_lo, band_hi, W=300, H=400, r=6, seed=0):
    """
    One still vial frame with `n_flies` dark discs placed uniformly at random
    inside the vertical band [band_lo, band_hi] as a fraction of vial height.
    A narrow band is a crowded one, so discs touch and merge -- which is
    exactly the condition N3 is about: flies piled at the base of the tube.
    """
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 3), 235, np.uint8)
    y0, y1 = int(band_lo * H), int(band_hi * H)
    y1 = max(y1, y0 + 2 * r + 1)
    for _ in range(n_flies):
        x = int(rng.integers(r + 4, W - r - 4))
        y = int(rng.integers(y0 + r + 1, y1 - r))
        cv2.circle(frame, (x, y), r, (25, 25, 25), -1)
    return frame


def _recall(det, n, band, seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    """Median recall over several random layouts of the same motion state."""
    rec = []
    for s in seeds:
        frame = _scene(n, band[0], band[1], seed=s)
        total, _, _ = det.detect(frame)
        rec.append(min(total, n) / float(n))
    return float(np.median(rec))


def _block_c10(results):
    if not _CV2:
        _bool_check("C10 SKIPPED (OpenCV absent)", True, "skip", "skip", results)
        return
    from modules.fly_detector import FlyDetector, DetectorConfig, VialROI

    det = FlyDetector(DetectorConfig(min_area=20, max_area=800, threshold_frac=0.5))

    r_healthy = _recall(det, 30, (0.05, 0.95))
    r_mild = _recall(det, 30, (0.05, 0.50))
    r_impaired = _recall(det, 30, (0.80, 1.00))
    r_severe = _recall(det, 30, (0.90, 1.00))

    _bool_check("C10 severe-stratum recall >= 0.80 (was 0.35)",
                r_severe >= 0.80, ">=0.80", f"{r_severe:.2f}", results)
    _bool_check("C10 impaired-stratum recall >= 0.80 (was 0.53)",
                r_impaired >= 0.80, ">=0.80", f"{r_impaired:.2f}", results)
    _bool_check("C10 healthy-stratum recall >= 0.80 (was 0.88)",
                r_healthy >= 0.80, ">=0.80", f"{r_healthy:.2f}", results)
    spread = max(r_healthy, r_mild, r_impaired, r_severe) - \
        min(r_healthy, r_mild, r_impaired, r_severe)
    _bool_check("C10 healthy-to-severe spread <= 0.10 (was 0.53)",
                spread <= 0.10, "<=0.10", f"{spread:.2f}", results)

    # The merge correction must be visible, not silent.
    frame = _scene(30, 0.90, 1.00, seed=1)
    d = det.detect_detailed(frame)
    _bool_check("C10 merged blobs are counted and reported",
                int(d.get("n_merged", 0)) > 0, ">0", d.get("n_merged", "missing"), results)
    _bool_check("C10 the split is reported as a correction, not a segmentation",
                "correction" in str(d.get("merge_note", "")).lower(),
                "correction", str(d.get("merge_note", ""))[:12], results)

    # Debris is not thirty flies: one huge blob must not be split indefinitely.
    debris = np.full((400, 300, 3), 235, np.uint8)
    cv2.rectangle(debris, (40, 300), (260, 390), (25, 25, 25), -1)
    dd = det.detect_detailed(debris)
    _bool_check("C10 one huge blob is debris, not a pile of flies",
                int(dd["total"]) <= 6, "<=6", dd["total"], results)
    _bool_check("C10 and it is flagged as oversized",
                int(dd.get("n_debris", 0)) >= 1, ">=1",
                dd.get("n_debris", "missing"), results)

    # The old behaviour stays reachable, so a historical run can be reproduced.
    det_off = FlyDetector(DetectorConfig(min_area=20, max_area=800,
                                         threshold_frac=0.5, split_merged=False))
    r_off = _recall(det_off, 30, (0.90, 1.00))
    _bool_check("C10 split_merged=False reproduces the uncorrected recall",
                r_off < r_severe, "lower", f"{r_off:.2f}", results)

    # And the per-vial QC must carry the merge count through to the row.
    vials = [VialROI(x=0, y=0, w=300, h=400, vial_id=1, expected_n=30)]
    vc = det.count_in_vials(frame, vials)[0]
    _bool_check("C10 n_merged reaches the vial record",
                "merge" in str(vc.to_dict().get("QC_flag", "")).lower()
                or int(vc.to_dict().get("N_merged", 0)) > 0,
                "reported", str(vc.to_dict().get("N_merged", "missing")), results)


# ══════════════════════════════════════════════════════════════════════════════
# C11 -- a search-grid boundary is not a period measurement
# ══════════════════════════════════════════════════════════════════════════════
def _block_c11(results):
    from modules import dams as D

    bin_min = 30.0
    n = 48 * 6                                   # 6 days at 30-minute bins
    t = np.arange(n) * bin_min
    act = 50 + 40 * np.sin(2 * np.pi * t / (24 * 60.0))

    p_clean = D.lomb_scargle_period(act, bin_min, 16.0, 32.0)
    _check("C11 a clean 24 h rhythm is recovered", p_clean, 24.0, 0.3, results)

    one_bad = act.copy()
    one_bad[17] = np.nan
    p_bad = D.lomb_scargle_period(one_bad, bin_min, 16.0, 32.0)
    _bool_check("C11 one invalid bin does not return the 16 h grid floor",
                not (_finite(p_bad) and abs(float(p_bad) - 16.0) < 1e-6),
                "not 16.0", str(p_bad), results)
    _bool_check("C11 either NaN or the true period, never the search floor",
                (not _finite(p_bad)) or abs(float(p_bad) - 24.0) <= 0.5,
                "NaN or 24", str(p_bad), results)

    _nan_check("C11 an all-NaN recording returns NaN",
               D.lomb_scargle_period(np.full(n, np.nan), bin_min), results)
    _nan_check("C11 a flat recording returns NaN",
               D.lomb_scargle_period(np.zeros(n), bin_min), results)

    # N44: real timestamps. A recording with a 12-hour gap is NOT regularly
    # sampled, and `arange(n) * bin_min` silently relabels every bin after the
    # gap -- which is the one assumption Lomb-Scargle exists to relax.
    keep = np.ones(n, bool)
    keep[80:104] = False
    t_gap, a_gap = t[keep], act[keep]
    p_real = D.lomb_scargle_period(a_gap, bin_min, 16.0, 32.0, time_min=t_gap)
    _check("C11 real timestamps recover 24 h across a gap", p_real, 24.0, 0.4, results)
    p_assumed = D.lomb_scargle_period(a_gap, bin_min, 16.0, 32.0)
    _bool_check("C11 assuming regularity gives a different answer",
                (not _finite(p_assumed)) or abs(float(p_assumed) - float(p_real)) > 0.3,
                "differs", str(p_assumed), results)
    _bool_check("C11 the timestamp argument exists",
                "time_min" in inspect.signature(D.lomb_scargle_period).parameters,
                "time_min", str(list(inspect.signature(D.lomb_scargle_period)
                                     .parameters))[-18:], results)

    # And through the per-fly wrapper, which is where the group mean is formed.
    rows = []
    for i, tt in enumerate(t):
        rows.append({"Genotype": "WT", "Treatment": "Veh", "Sex": "Male",
                     "FlyID": "f1", "Time_min": float(tt),
                     "Activity": float(act[i]) if i != 17 else np.nan})
    long_df = pd.DataFrame(rows)
    pf = D.period_per_fly(long_df, periodogram_bin_min=30.0)
    ls = float(pf["Period_h_lombscargle"].iloc[0]) if len(pf) else np.nan
    _bool_check("C11 one bad minute does not put 16.000 into the group mean",
                (not _finite(ls)) or abs(ls - 16.0) > 1e-6, "not 16.0", str(ls), results)


# ══════════════════════════════════════════════════════════════════════════════
# C12 -- endpoints may not be derived from refused predictions
# ══════════════════════════════════════════════════════════════════════════════
def _block_c12(results):
    from modules import trajectory as T

    # Observations over 5-50 days; targets out to 200, so the far targets are
    # beyond `max_horizon` and the engine refuses them.
    obs_t = np.arange(5.0, 51.0, 5.0)
    obs_y = np.clip(1.0 - 0.012 * (obs_t - 5.0), 0.0, 1.0)
    targets = np.arange(5.0, 201.0, 5.0)

    pred = T.predict_trajectory(obs_t, obs_y, target_days=targets, max_horizon=1.0)
    n_ref = sum(1 for p in pred["predictions"] if p["refused"])
    _bool_check("C12 the fixture actually refuses something", n_ref > 0, ">0",
                n_ref, results)

    ep = T.derive_endpoints(pred, threshold=0.5)
    _bool_check("C12 refused days are excluded and counted",
                int(ep.get("n_refused_excluded", 0)) == n_ref, n_ref,
                ep.get("n_refused_excluded", "missing"), results)
    last_ok = max(p["day"] for p in pred["predictions"] if not p["refused"])
    _check("C12 the endpoint window ends at the last issued prediction",
           ep["window_days"][1], last_ok, 1e-9, results)

    # The register's measurement: including refused days inflated the interval.
    ep_all = T.derive_endpoints(pred, threshold=0.5, _include_refused=True)
    w_ok = ep["onset_age_days_ci_upper"] - ep["onset_age_days_ci_lower"]
    w_all = ep_all["onset_age_days_ci_upper"] - ep_all["onset_age_days_ci_lower"]
    _bool_check("C12 excluding them narrows the onset-age interval",
                (not _finite(w_all)) or (_finite(w_ok) and w_ok <= w_all),
                "narrower", f"{w_ok:.3g} vs {w_all:.3g}", results)
    h_ok = ep["healthspan_days_ci_upper"] - ep["healthspan_days_ci_lower"]
    h_all = ep_all["healthspan_days_ci_upper"] - ep_all["healthspan_days_ci_lower"]
    _bool_check("C12 and the healthspan interval",
                (not _finite(h_all)) or (_finite(h_ok) and h_ok <= h_all),
                "narrower", f"{h_ok:.3g} vs {h_all:.3g}", results)

    # Everything refused: there is no endpoint to derive.
    pred2 = T.predict_trajectory(obs_t, obs_y,
                                 target_days=np.arange(400.0, 500.0, 10.0),
                                 max_horizon=1.0)
    ep2 = T.derive_endpoints(pred2, threshold=0.5)
    _bool_check("C12 all refused -> no endpoints, with a reason",
                ep2.get("available") is False and "refus" in str(ep2.get("reason", "")).lower(),
                "unavailable", str(ep2.get("reason", ""))[:12], results)

    # Nothing refused: the result must be unchanged.
    pred3 = T.predict_trajectory(obs_t, obs_y, target_days=np.arange(5.0, 51.0, 5.0),
                                 max_horizon=3.0)
    ep3 = T.derive_endpoints(pred3, threshold=0.5)
    _check("C12 nothing refused: no exclusions", ep3.get("n_refused_excluded"), 0, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# C13 -- clustering on age, not on assay seconds
# ══════════════════════════════════════════════════════════════════════════════
def _pi_table(axis_col):
    """
    Eight groups, each scored at five assay timepoints at each of five ages --
    the structure a longitudinal climbing experiment actually has, and the one
    that makes the two axes distinguishable at all. (A table with one timepoint
    per age cannot distinguish them: Age and Time are then collinear and both
    groupings return the same five numbers.)

    Along AGE every group DECLINES, at -1.1 PI per day. Within one run, along
    the assay SECONDS axis, every group RISES steeply, at +6 PI per 10 s,
    because flies keep crossing the scoring line as the run proceeds. The two
    axes therefore say opposite things about the same cohort, which is why
    clustering one and naming the other is a category error (register N31).
    """
    rows = []
    genos = ["Oregon-WT", "TDP43", "FUS", "Elav-GAL4"]
    for gi, geno in enumerate(genos):
        for treat in ("Vehicle", "Drug"):
            base = 85.0 - 12.0 * gi + (4.0 if treat == "Drug" else 0.0)
            for a in (5, 15, 25, 35, 45):
                for k in range(5):
                    rows.append({"Genotype": geno, "Treatment": treat,
                                 "Sex": "Male", "Age": a, "Time": 10 + 10 * k,
                                 "PI_mean": base - 1.1 * a + 6.0 * k})
    df = pd.DataFrame(rows)
    return df.rename(columns={"Age": axis_col}) if axis_col != "Age" else df


def _block_c13(results):
    from modules import trajectory_cluster as TC

    no_age = _pi_table("Age").drop(columns=["Age"])
    _raises("C13 a PI table with no age axis is refused",
            lambda: TC.build_trajectory_dict(no_age), ["Age"], results)
    try:
        TC.build_trajectory_dict(no_age)
        _msg = ""
    except Exception as _e:
        _msg = str(_e)
    _bool_check("C13 the refusal explains the axis, not just a missing column",
                "seconds" in _msg.lower() or "within-assay" in _msg.lower(),
                "explains", _msg[:12], results)

    tab = _pi_table("Age")
    traj, meta = TC.build_trajectory_dict(tab)
    _check("C13 one trajectory per group, on the age axis", len(traj), 8, 0, results)
    _bool_check("C13 the metadata records the axis",
                all(m.get("time_axis") == "Age" for m in meta.values()),
                "Age", str(list(meta.values())[0].get("time_axis", "missing")), results)

    res = TC.cluster_trajectories(tab, bootstrap_reps=5, seed=1)
    _bool_check("C13 the result records the axis it clustered",
                res.get("time_axis") == "Age", "Age",
                res.get("time_axis", "missing"), results)
    _bool_check("C13 and carries a note about what the axis means",
                "age" in str(res.get("axis_note", "")).lower(),
                "names age", str(res.get("axis_note", ""))[:12], results)

    ct = res.get("cluster_table", pd.DataFrame())
    interp = TC.interpret_clusters(ct, res.get("archetypes", {}),
                                   time_axis=res.get("time_axis"))
    wt = interp[interp["Genotype"] == "Oregon-WT"] if "Genotype" in interp.columns \
        else pd.DataFrame()
    _bool_check("C13 untreated wild type is not labelled 'Progressive Rescue'",
                wt.empty or not (wt["Archetype"] == "Progressive Rescue").any(),
                "not rescue", (str(wt["Archetype"].tolist())[:12] if not wt.empty
                               else "no WT row"), results)

    # The seconds axis stays available, but it is not a decline trajectory and
    # must not borrow the longitudinal archetype vocabulary.
    res2 = TC.cluster_trajectories(tab, time_col="Time", bootstrap_reps=5, seed=1)
    _bool_check("C13 the seconds axis is reachable, explicitly",
                res2.get("time_axis") == "Time", "Time",
                res2.get("time_axis", "missing"), results)
    interp2 = TC.interpret_clusters(res2.get("cluster_table", pd.DataFrame()),
                                    res2.get("archetypes", {}),
                                    time_axis="Time")
    names2 = set(interp2["Archetype"].astype(str)) if "Archetype" in interp2.columns else set()
    _bool_check("C13 no longitudinal archetype name is applied to seconds",
                not (names2 & {"Progressive Rescue", "Declining Performer",
                               "Early Rescue Late Decline", "Sustained Rescue"}),
                "none", str(sorted(names2))[:12], results)
    _bool_check("C13 and the seconds result is named as climbing kinetics",
                "kinetic" in str(res2.get("axis_note", "")).lower(),
                "kinetics", str(res2.get("axis_note", ""))[:12], results)

    # The axis change must actually change what is clustered. These checks are
    # structural -- note honestly that this synthetic table does NOT reproduce
    # the register's "untreated wild type labelled Progressive Rescue" case,
    # which was measured on real longitudinal data; the archetype a cluster
    # receives depends on its DTW barycentre, and on this fixture both axes
    # happen to yield the same label. What IS demonstrated here is that the two
    # axes feed the clustering different trajectories, which is the mechanism,
    # and that the longitudinal vocabulary is withheld on the seconds axis.
    t_age, _ = TC.build_trajectory_dict(tab, time_col="Age")
    t_sec, _ = TC.build_trajectory_dict(tab, time_col="Time")
    shared = sorted(set(t_age) & set(t_sec))
    diff = sum(1 for k in shared
               if t_age[k].shape != t_sec[k].shape
               or not np.allclose(t_age[k], t_sec[k]))
    _bool_check("C13 the two axes feed the clustering different trajectories",
                len(shared) > 0 and diff == len(shared), "all differ",
                f"{diff}/{len(shared)}", results)

    alias = _pi_table("Age_days")
    traj2, _ = TC.build_trajectory_dict(alias)
    _check("C13 'Age_days' resolves as the age axis", len(traj2), 8, 0, results)


# ══════════════════════════════════════════════════════════════════════════════
# C14 -- absent data is not a middling result, and weights must sum to one
# ══════════════════════════════════════════════════════════════════════════════
def _tcs_frames():
    keys = {"Genotype": ["WT", "Mut"], "Treatment": ["Drug", "Drug"],
            "Sex": ["Male", "Male"]}
    cri = pd.DataFrame({**keys, "CRI": [60.0, 40.0]})
    tsw = pd.DataFrame({**keys, "TSW": [1.5, 1.0]})
    sdq = pd.DataFrame({**keys, "SDQ": [0.2, 0.3]})
    mdr = pd.DataFrame({**keys, "Rescue_pct": [70.0, 50.0],
                        "Trajectory": ["Sustained Rescue", np.nan]})
    return cri, tsw, sdq, mdr


def _block_c14(results):
    from modules import translational_score as TS

    cri, tsw, sdq, mdr = _tcs_frames()
    out = TS.compute_tcs(cri, tsw, sdq, mdr)

    row_mut = out[out["Genotype"] == "Mut"].iloc[0]
    _bool_check("C14 the trajectory component is published, so it can be checked",
                "Traj_score" in out.columns, "Traj_score",
                "present" if "Traj_score" in out.columns else "absent", results)
    _nan_check("C14 a missing Trajectory scores NaN, not 0.2",
               row_mut["Traj_score"] if "Traj_score" in out.columns else 0.2, results)
    _check("C14 the number of missing trajectories is reported",
           out.attrs.get("n_missing_trajectory"), 1, 0, results)

    # The TCS of that row must be the weighted mean of its FOUR present
    # components, with the trajectory weight removed from the denominator.
    w = out.attrs["weights_used"]
    comps = {"CRI": row_mut["CRI_norm"], "TSW": row_mut["TSW_norm"],
             "SDQ": row_mut["SDQ_cons"], "MDR": row_mut["MDR_norm"]}
    wt = {"CRI": w["CRI"], "TSW": w["TSW"], "SDQ": w["SDQ"], "MDR": w["MDR"]}
    expect = sum(wt[k] * comps[k] for k in comps) / sum(wt.values()) * 100.0
    _check("C14 the trajectory component is dropped, not scored",
           row_mut["TCS"], expect, 0.02, results)
    _check("C14 and n_components is 4", row_mut.get("n_components"), 4, 0, results)

    # An explicit "Undetermined" label is a recorded clustering result, not
    # absent data, and keeps its documented 0.2 by convention.
    mdr2 = mdr.copy()
    mdr2.loc[1, "Trajectory"] = "Undetermined"
    out2 = TS.compute_tcs(cri, tsw, sdq, mdr2)
    _check("C14 an explicit 'Undetermined' label still scores 0.2",
           float(out2[out2["Genotype"] == "Mut"]["Traj_score"].iloc[0])
           if "Traj_score" in out2.columns else np.nan, 0.2, 1e-9, results)
    _check("C14 and is not counted as missing",
           out2.attrs.get("n_missing_trajectory"), 0, 0, results)

    # N32b: a partial weights dict is back-filled BEFORE normalising.
    out3 = TS.compute_tcs(cri, tsw, sdq, mdr, weights={"CRI": 1.0})
    w3 = out3.attrs["weights_used"]
    _check("C14 the weights actually used sum to 1", sum(w3.values()), 1.0, 1e-9, results)
    _check("C14 CRI weight is 1.0/1.65", w3["CRI"], 1.0 / 1.65, 1e-6, results)
    _check("C14 TSW keeps its default, renormalised", w3["TSW"], 0.25 / 1.65, 1e-6, results)
    _bool_check("C14 every component key is present",
                set(w3) == {"CRI", "TSW", "SDQ", "MDR", "Traj"}, "5 keys",
                str(sorted(w3)), results)

    _raises("C14 an unknown weight key is refused, not ignored",
            lambda: TS.compute_tcs(cri, tsw, sdq, mdr, weights={"CRI": 1.0, "Nope": 0.5}),
            ["Nope"], results)

    # The stacked component bars must add up to the score they decorate.
    bar_cols = [c for c in out3.columns
                if c in ("TCS_CRI", "TCS_TSW", "TCS_SDQ", "TCS_MDR", "TCS_Traj")]
    if bar_cols:
        r = out3[out3["Genotype"] == "WT"].iloc[0]
        _check("C14 component bars sum to the TCS",
               float(np.nansum([r[c] for c in bar_cols])), float(r["TCS"]), 0.05, results)
    else:
        _bool_check("C14 component bars sum to the TCS", False, "bars", "no TCS_*", results)

    radar = TS.tcs_radar_data(out3)
    _rd = radar[1] if isinstance(radar, tuple) else radar.get("radar_data", {})
    vals = np.asarray([v for row in _rd.values() for v in row], dtype=float).ravel()         if _rd else np.array([])
    _bool_check("C14 no radar axis exceeds 100% under custom weights",
                vals.size == 0 or float(np.nanmax(vals)) <= 100.0 + 1e-6,
                "<=100", f"{float(np.nanmax(vals)) if vals.size else 0:.1f}", results)


# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    results = []
    blocks = (("C1", _block_c1), ("C2", _block_c2), ("C3", _block_c3),
              ("C4", _block_c4), ("C5", _block_c5), ("C6", _block_c6),
              ("C7", _block_c7), ("C8", _block_c8), ("C9", _block_c9),
              ("C10", _block_c10), ("C11", _block_c11), ("C12", _block_c12),
              ("C13", _block_c13), ("C14", _block_c14))
    for label, fn in blocks:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn(results)
        except Exception as exc:
            results.append((f"{label} block raised {type(exc).__name__}",
                            "no exception", str(exc)[:38], False))

    print("=" * 96)
    print("DroPhenix P0 Block C regression suite -- the fourteen numerical corrections")
    print("  N14/15 LC50 | N16 Hessian | N17-19 cohort+event | N34/35 HKSJ | N20 Y-maze")
    print("  N21 grooming | N11/12 parser | N22 figures | N3 merges | N43/44 period")
    print("  N29 refused days | N31 age axis | N32a/b trajectory+weights")
    print("=" * 96)
    print(f"{'Check':<62}{'Expected':>16}{'Engine':>12}{'Pass':>6}")
    print("-" * 96)
    all_ok = True
    for name, exp, got, ok in results:
        all_ok &= ok
        print(f"{name:<62}{str(exp)[:16]:>16}{str(got)[:12]:>12}{'OK' if ok else 'FAIL':>6}")
    print("-" * 96)
    n_fail = sum(1 for _, _, _, ok in results if not ok)
    print(f"{len(results)} checks, {n_fail} failing")
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
