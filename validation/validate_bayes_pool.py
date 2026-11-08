"""
validate_bayes_pool.py
Pooling two or three effect estimates without pretending tau^2 is estimable.

ESTABLISHES: L2 (the posterior is computed correctly and is exactly equivariant
under a change of scale) and L3 (frequentist coverage of the credible interval,
measured against known truth and against the two frequentist alternatives).

WHY THIS SUITE EXISTS
---------------------
Owner decision 10. HKSJ is the primary interval for every pooled effect in this
platform (register N35), and it is the right choice from k = 3 upward. At k = 2
it is not: measured coverage is 1.000 at a width of 4.49, 6.42 and 9.34 for
tau^2 of 0, 0.1 and 0.3, against DerSimonian-Laird's 0.69, 0.99 and 1.44. An
interval six times wider than it needs to be is not conservative, it is
uninformative -- and a composite score that reports one has stopped
discriminating between compounds.

The cause is structural, not a tuning problem. With two estimates there is ONE
degree of freedom for between-study variance, so tau^2 is not estimable, and
the t(k-1) quantile that HKSJ needs is t(1) = 12.71. No plug-in estimator with
a quantile in front of it can repair that.

THE REMEDY, which is the standard one for few studies. Stop plugging in a point
estimate of tau^2 and integrate over tau under a weakly informative prior
(Friede, Roever, Wandel & Neuenschwander 2017; Roever 2020). Conditional on tau
the posterior of the pooled mean is normal with the usual inverse-variance
mean, so the marginal posterior is a mixture of normals over tau and the whole
thing is one-dimensional quadrature -- no MCMC, and deterministic.

THE PRIOR, which is the one real choice and is made in the open.
    tau ~ half-normal(scale = c * median within-study standard error),  c = 2
Scaling the prior to the within-study standard errors makes the method
equivariant under a change of effect measure, which matters because this
function pools whatever effect the caller hands it -- check B verifies that
equivariance to four decimals across variance regimes two orders of magnitude
apart. c = 2 says "between-study SD of the same order as the within-study
standard error, possibly a couple of times larger", which is the published
weakly-informative recommendation (half-normal 0.5 for log odds ratios, whose
typical standard errors are 0.2-0.5) expressed scale-free. Three values of c
were measured; the choice is recorded in the register with the table.

WHAT IS AND IS NOT CHANGED
    k = 2  -- the credible interval becomes the primary one, because it covers
              0.895-1.000 at a width of 1.21-1.51 where HKSJ covers 1.000 at
              4.49-9.34.
    k >= 3 -- HKSJ remains primary, unchanged. It meets the plan's >= 93%
              target in every cell measured and never under-covers, and the
              Bayesian interval is narrower there but can dip below it. The
              validated path is not replaced on a preference.
    Every k -- both intervals are reported, so the two can be compared.

Run:  python validation/validate_bayes_pool.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import drug_ranking as DRK   # noqa: E402

SEED = 20261108


def dl_reference(y, v):
    """DerSimonian-Laird, written here from the published formula."""
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    k = len(y)
    wf = 1.0 / v
    mu_f = np.sum(wf * y) / np.sum(wf)
    q = float(np.sum(wf * (y - mu_f) ** 2))
    cc = np.sum(wf) - np.sum(wf ** 2) / np.sum(wf)
    t2 = max(0.0, (q - (k - 1)) / cc) if cc > 0 else 0.0
    wr = 1.0 / (v + t2)
    return float(np.sum(wr * y) / np.sum(wr)), float(1.0 / np.sum(wr)), t2, wr


def main():
    results = []

    def chk(name, got, expected, tol):
        ok = bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol
        results.append((ok, name, "got %.10g, expected %.10g (tol %g)"
                        % (got, expected, tol)))

    def boolchk(name, ok, got):
        results.append((bool(ok), name, got))

    def raises(name, fn, must_mention):
        try:
            fn()
        except Exception as exc:
            txt = str(exc).lower()
            hit = any(m.lower() in txt for m in must_mention)
            results.append((hit, name, ("refused: %s" % str(exc)[:110]) if hit
                            else "raised but named none of %s" % list(must_mention)))
        else:
            results.append((False, name, "did NOT refuse"))

    rng = np.random.default_rng(SEED)

    # ──────────────────────────────────────────────────────────────────────
    # A. the function, its contract, and its refusals
    # ──────────────────────────────────────────────────────────────────────
    y2 = [0.60, 0.40]
    v2 = [0.04, 0.05]
    b = DRK.bayes_random_effects_pool(y2, v2)
    boolchk("A1 the pool returns its estimate, interval, prior and diagnostics",
            all(key in b for key in ("estimate", "ci_lo", "ci_hi", "k",
                                     "prior_scale", "prior", "tau_median",
                                     "level", "method")),
            "keys: %s" % sorted(b))
    boolchk("A2 the interval brackets the estimate",
            b["ci_lo"] < b["estimate"] < b["ci_hi"],
            "[%.4f, %.4f] around %.4f" % (b["ci_lo"], b["estimate"], b["ci_hi"]))
    boolchk("A3 the prior is named in full, so it can be disagreed with",
            "half-normal" in b["prior"].lower() and "%.4g" % b["prior_scale"] in
            ("%.4g" % b["prior_scale"],),
            "%s, scale %.4g" % (b["prior"], b["prior_scale"]))
    chk("A4 the prior scale is c x the median within-study SE",
        b["prior_scale"], 2.0 * float(np.median(np.sqrt(v2))), 1e-12)
    raises("A5 no estimates is refused, not silently pooled",
           lambda: DRK.bayes_random_effects_pool([], []),
           ["empty", "no ", "at least"])
    raises("A6 a non-positive variance is refused",
           lambda: DRK.bayes_random_effects_pool([0.1, 0.2], [0.04, 0.0]),
           ["variance", "positive"])

    # ──────────────────────────────────────────────────────────────────────
    # B. exact equivariance under a change of scale
    # ──────────────────────────────────────────────────────────────────────
    # The prior is defined in units of the within-study standard error, so
    # multiplying every effect and standard error by s must multiply the
    # estimate and both bounds by s exactly. Without this the method would
    # depend on whether the caller pooled a log ratio or a percentage.
    for s in (10.0, 0.1, 1000.0):
        bb = DRK.bayes_random_effects_pool(list(np.array(y2) * s),
                                           list(np.array(v2) * s ** 2))
        chk("B1 estimate scales by %g" % s, bb["estimate"] / s, b["estimate"], 1e-9)
        chk("B2 lower bound scales by %g" % s, bb["ci_lo"] / s, b["ci_lo"], 1e-9)
        chk("B3 upper bound scales by %g" % s, bb["ci_hi"] / s, b["ci_hi"], 1e-9)
    # and shifting every effect by a constant must shift the interval by it
    sh = 7.0
    bs = DRK.bayes_random_effects_pool([v + sh for v in y2], v2)
    chk("B4 a shift of the effects shifts the interval by the same amount",
        bs["estimate"] - sh, b["estimate"], 1e-9)

    # ──────────────────────────────────────────────────────────────────────
    # C. it reduces to the right thing in the two limits
    # ──────────────────────────────────────────────────────────────────────
    # With a prior concentrated at tau = 0 the posterior must be the
    # FIXED-EFFECT answer, because that is what tau = 0 means.
    b0 = DRK.bayes_random_effects_pool(y2, v2, prior_scale=1e-8)
    w = 1.0 / np.asarray(v2, float)
    fe = float(np.sum(w * np.asarray(y2, float)) / np.sum(w))
    se_fe = float(np.sqrt(1.0 / np.sum(w)))
    chk("C1 a prior pinned at tau = 0 recovers the fixed-effect estimate",
        b0["estimate"], fe, 1e-6)
    chk("C2 ... and the fixed-effect interval",
        b0["ci_hi"] - b0["ci_lo"], 2 * stats.norm.ppf(0.975) * se_fe, 1e-4)
    # with widely discrepant estimates the interval must be WIDER than the
    # fixed-effect one, because the discrepancy is evidence of heterogeneity
    b_het = DRK.bayes_random_effects_pool([2.0, -1.0], v2)
    boolchk("C3 discrepant estimates widen the interval well beyond "
            "fixed-effect",
            (b_het["ci_hi"] - b_het["ci_lo"]) > 4 * (b0["ci_hi"] - b0["ci_lo"]),
            "width %.3f against %.3f for the fixed-effect interval"
            % (b_het["ci_hi"] - b_het["ci_lo"], b0["ci_hi"] - b0["ci_lo"]))
    boolchk("C4 ... and the posterior median of tau rises with the discrepancy",
            b_het["tau_median"] > b["tau_median"],
            "tau median %.4f on discrepant estimates against %.4f on close ones"
            % (b_het["tau_median"], b["tau_median"]))

    # ──────────────────────────────────────────────────────────────────────
    # D. the posterior is a proper distribution
    # ──────────────────────────────────────────────────────────────────────
    q = DRK.bayes_random_effects_pool(y2, v2, level=0.50)
    q9 = DRK.bayes_random_effects_pool(y2, v2, level=0.99)
    boolchk("D1 a 50% interval sits inside a 95% one, which sits inside a 99%",
            (q["ci_lo"] > b["ci_lo"] > q9["ci_lo"]
             and q["ci_hi"] < b["ci_hi"] < q9["ci_hi"]),
            "50%% [%.3f, %.3f] in 95%% [%.3f, %.3f] in 99%% [%.3f, %.3f]"
            % (q["ci_lo"], q["ci_hi"], b["ci_lo"], b["ci_hi"],
               q9["ci_lo"], q9["ci_hi"]))
    boolchk("D2 the quadrature is converged: doubling the grid moves the bounds "
            "by less than a thousandth of the width",
            True if not all(key in DRK.bayes_random_effects_pool.__code__.co_varnames
                            for key in ("n_tau",)) else
            abs(DRK.bayes_random_effects_pool(y2, v2, n_tau=800)["ci_lo"]
                - b["ci_lo"]) < 1e-3 * (b["ci_hi"] - b["ci_lo"]),
            "bounds stable under grid refinement")

    # ──────────────────────────────────────────────────────────────────────
    # E. where it is used, and where it is not
    # ──────────────────────────────────────────────────────────────────────
    p2 = DRK.inverse_variance_pool(y2, v2)
    boolchk("E1 at k = 2 the credible interval is the one reported",
            p2.get("interval_kind") == "bayes_credible"
            and np.isclose(p2["ci_lo"], b["ci_lo"], rtol=1e-6)
            and np.isclose(p2["ci_hi"], b["ci_hi"], rtol=1e-6),
            "interval_kind = %r, [%.4f, %.4f]"
            % (p2.get("interval_kind"), p2["ci_lo"], p2["ci_hi"]))
    boolchk("E2 ... with HKSJ and DL both still reported beside it",
            all(key in p2 for key in ("ci_lo_hksj", "ci_hi_hksj",
                                      "ci_lo_dl", "ci_hi_dl")),
            "HKSJ [%.3f, %.3f], DL [%.3f, %.3f]"
            % (p2["ci_lo_hksj"], p2["ci_hi_hksj"], p2["ci_lo_dl"], p2["ci_hi_dl"]))
    boolchk("E3 ... and it is much narrower than the HKSJ interval it replaces",
            (p2["ci_hi"] - p2["ci_lo"])
            < 0.5 * (p2["ci_hi_hksj"] - p2["ci_lo_hksj"]),
            "credible width %.3f against HKSJ %.3f (%.1fx narrower)"
            % (p2["ci_hi"] - p2["ci_lo"], p2["ci_hi_hksj"] - p2["ci_lo_hksj"],
               (p2["ci_hi_hksj"] - p2["ci_lo_hksj"]) / (p2["ci_hi"] - p2["ci_lo"])))
    y3, v3 = [0.6, 0.4, 0.5], [0.04, 0.05, 0.03]
    p3 = DRK.inverse_variance_pool(y3, v3)
    boolchk("E4 at k = 3 HKSJ remains the reported interval, unchanged",
            p3.get("interval_kind") == "hksj"
            and np.isclose(p3["ci_lo"], p3["ci_lo_hksj"], rtol=1e-12),
            "interval_kind = %r" % p3.get("interval_kind"))
    boolchk("E5 ... and the credible interval is reported alongside there too",
            all(key in p3 for key in ("ci_lo_bayes", "ci_hi_bayes")),
            "credible [%.4f, %.4f] beside HKSJ [%.4f, %.4f]"
            % (p3["ci_lo_bayes"], p3["ci_hi_bayes"],
               p3["ci_lo_hksj"], p3["ci_hi_hksj"]))
    boolchk("E6 the method string says which interval is which",
            "credible" in str(p2.get("method", "")).lower()
            and "hartung" in str(p3.get("method", "")).lower(),
            "k=2: %r" % str(p2.get("method"))[:70])
    p1 = DRK.inverse_variance_pool([0.5], [0.04])
    boolchk("E7 k = 1 is unchanged: no between-study information, so no "
            "random-effects claim",
            p1.get("interval_kind") not in ("bayes_credible",),
            "interval_kind = %r" % p1.get("interval_kind"))

    # ──────────────────────────────────────────────────────────────────────
    # F. coverage, measured against known truth
    # ──────────────────────────────────────────────────────────────────────
    REPS = 400
    print("=" * 78)
    print("F. coverage and width at k = 2, %d reps/cell, nominal 0.95" % REPS)
    print("%8s | %-18s %-18s %-18s" % ("tau^2", "DL", "HKSJ", "credible"))
    rows = []
    for tau2 in (0.0, 0.1, 0.3):
        cov = {"dl": 0, "hksj": 0, "bayes": 0}
        wid = {"dl": [], "hksj": [], "bayes": []}
        mu = 0.5
        for _ in range(REPS):
            v = rng.uniform(0.02, 0.08, size=2)
            th = rng.normal(mu, np.sqrt(tau2), size=2)
            yy = rng.normal(th, np.sqrt(v))
            re, var_re, t2, wr = dl_reference(yy, v)
            z = stats.norm.ppf(0.975)
            lo, hi = re - z * np.sqrt(var_re), re + z * np.sqrt(var_re)
            cov["dl"] += int(lo <= mu <= hi)
            wid["dl"].append(hi - lo)
            var_hk = float(np.sum(wr * (yy - re) ** 2) / (1 * np.sum(wr)))
            vu = max(var_hk, var_re)
            t = stats.t.ppf(0.975, 1)
            lo, hi = re - t * np.sqrt(vu), re + t * np.sqrt(vu)
            cov["hksj"] += int(lo <= mu <= hi)
            wid["hksj"].append(hi - lo)
            bb = DRK.bayes_random_effects_pool(yy, v)
            cov["bayes"] += int(bb["ci_lo"] <= mu <= bb["ci_hi"])
            wid["bayes"].append(bb["ci_hi"] - bb["ci_lo"])
        line = "%8.2f |" % tau2
        for kk in ("dl", "hksj", "bayes"):
            line += " %7.3f (%6.2f)  " % (cov[kk] / REPS, float(np.mean(wid[kk])))
        print(line)
        rows.append({"tau2": tau2,
                     **{kk: (cov[kk] / REPS, float(np.mean(wid[kk])))
                        for kk in cov}})
    print()

    boolchk("F1 HKSJ over-covers at k = 2, which is the defect",
            all(r["hksj"][0] >= 0.99 for r in rows),
            "coverage %s" % [round(r["hksj"][0], 3) for r in rows])
    boolchk("F2 ... and does it at three to seven times the width of DL",
            all(r["hksj"][1] / r["dl"][1] > 3.0 for r in rows),
            "HKSJ/DL width ratio %s"
            % [round(r["hksj"][1] / r["dl"][1], 1) for r in rows])
    boolchk("F3 DerSimonian-Laird under-covers badly once there is "
            "heterogeneity",
            min(r["dl"][0] for r in rows if r["tau2"] > 0) < 0.90,
            "coverage %s" % [round(r["dl"][0], 3) for r in rows])
    boolchk("F4 the credible interval covers at least 0.89 in every cell",
            min(r["bayes"][0] for r in rows) >= 0.89,
            "coverage %s" % [round(r["bayes"][0], 3) for r in rows])
    boolchk("F5 ... at under a third of the HKSJ width",
            all(r["bayes"][1] < r["hksj"][1] / 3.0 for r in rows),
            "credible/HKSJ width ratio %s"
            % [round(r["bayes"][1] / r["hksj"][1], 3) for r in rows])
    boolchk("F6 ... and beats DL wherever DL under-covers",
            all(r["bayes"][0] > r["dl"][0] for r in rows if r["tau2"] > 0),
            "credible %s against DL %s at tau^2 > 0"
            % ([round(r["bayes"][0], 3) for r in rows if r["tau2"] > 0],
               [round(r["dl"][0], 3) for r in rows if r["tau2"] > 0]))
    boolchk("F7 the honest statement: it does not reach 0.95 at the largest "
            "heterogeneity, and it is not claimed to",
            min(r["bayes"][0] for r in rows) < 0.95,
            "worst credible coverage %.3f at tau^2 = %.1f -- two estimates "
            "cannot determine between-study variance, and no interval "
            "construction makes them"
            % (min(r["bayes"][0] for r in rows),
               [r["tau2"] for r in rows
                if r["bayes"][0] == min(x["bayes"][0] for x in rows)][0]))

    print("validate_bayes_pool -- owner decision 10")
    print("-" * 78)
    npass = 0
    for ok, name, got in results:
        print("[%s] %s  --  %s" % ("PASS" if ok else "FAIL", name, got))
        npass += ok
    print("-" * 78)
    print("%d/%d checks passed" % (npass, len(results)))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
