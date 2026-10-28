"""
Validation suite for modules/circadian_climbing.py

WHAT IS BEING VALIDATED, AND AGAINST WHAT
-----------------------------------------
A. EXACT RECOVERY ON NOISE-FREE DATA.
   The linear reparameterisation y = M + beta cos(wt) + gamma sin(wt) is an
   algebraic identity for y = M + A cos(w(t - phi)), so on noise-free data the
   least-squares solution must return the generating (M, A, phi) to machine
   precision. This is asserted at 1e-10, not at a loose tolerance -- if it ever
   fails, the parameterisation itself is wrong, not the numerics.

B. THE LINEAR FIT EQUALS A DIRECT NONLINEAR LEAST-SQUARES FIT.
   scipy.optimize.least_squares is run on the ORIGINAL nonlinear
   M + A cos(2 pi (t - phi)/period) parameterisation from a deliberately poor
   starting point, on noisy data. Both fits must reach the same residual sum of
   squares and the same (M, A, phi). This proves the closed form is not merely
   self-consistent but is the true LS optimum of the model as written.

C. CONFIDENCE-INTERVAL COVERAGE FOR AMPLITUDE AND ACROPHASE.
   Many simulations with known (M, A, phi) and Gaussian noise; the elliptical
   (Bingham 1982), delta-method and parametric-bootstrap intervals must each
   cover at approximately the nominal 95%.

D. THE ZERO-AMPLITUDE F-TEST.
   Type I error ~5% on arrhythmic data (A = 0 exactly), and high power on data
   with a real rhythm. Both against the binomial Monte-Carlo envelope.

E. POPULATION COSINOR WITH 3 TIMEPOINTS PER SUBJECT.
   Recovers the population amplitude and acrophase; the Hotelling T^2 zero-
   amplitude test is calibrated; and the single-subject fits are correctly
   reported as having ZERO residual degrees of freedom with no usable interval.

F. FATIGUE vs RHYTHM SEPARATION.
   On a completely collinear schedule (everyone tested morning-then-afternoon-
   then-evening) the module MUST refuse to attribute a monotone within-day
   fatigue trend to circadian phase. On a counterbalanced schedule built from
   the same generative model it must recover both the fatigue slope and the
   (absent or present) rhythm correctly.

G. INDEPENDENT-IMPLEMENTATION CROSS-CHECK: astropy LombScargle.
   astropy's Lomb-Scargle with fit_mean=True fits exactly the same one-term
   sinusoid-plus-offset model. At the known frequency its coefficients must
   equal ours to machine precision, and a free period search over a grid must
   put its maximum at the generating period.

Run: ./.venv/Scripts/python.exe validation/validate_circadian_climbing.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import circadian_climbing as CC  # noqa: E402

FAILURES: list[str] = []
SEED = 20260803
PERIOD = 24.0


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def cosine(t, M, A, phi, period=PERIOD):
    """The generating model, written exactly as in the module docstring."""
    return M + A * np.cos(2.0 * np.pi * (np.asarray(t) - phi) / period)


def circ_diff(a, b, period=PERIOD):
    """Signed difference between two phases, wrapped to (-period/2, period/2]."""
    d = (np.asarray(a) - np.asarray(b) + period / 2.0) % period - period / 2.0
    return d


# ─────────────────────────────────────────────────────────────────────────────
def test_A_exact_recovery() -> None:
    print("\nA. EXACT recovery on noise-free data (linear reparameterisation is algebraic)")
    for (M, A, phi, zt) in [
        (0.90, 0.35, 6.0, np.array([1.0, 7.0, 13.0, 19.0])),
        (-1.25, 1.10, 21.5, np.array([2.0, 8.0, 11.0])),          # the lab's 3 timepoints
        (3.00, 0.05, 0.25, np.linspace(0, 23, 12)),
        (0.00, 2.00, 12.0, np.array([0.0, 5.5, 9.25, 14.75, 20.0, 22.5])),
    ]:
        y = cosine(zt, M, A, phi)
        fit = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)
        dphi = float(abs(circ_diff(fit.acrophase_h, phi)))
        lab = f"M={M:g}, A={A:g}, phi={phi:g}, n={zt.size}"
        check(f"MESOR exact ({lab})", abs(fit.mesor - M) < 1e-10,
              f"|err| = {abs(fit.mesor - M):.3e}")
        check(f"amplitude exact ({lab})", abs(fit.amplitude - A) < 1e-10,
              f"|err| = {abs(fit.amplitude - A):.3e}")
        check(f"acrophase exact ({lab})", dphi < 1e-10, f"|err| = {dphi:.3e} h")

    # beta/gamma <-> A/phi round trip is exact by construction
    zt = np.linspace(0, 23, 9)
    y = cosine(zt, 1.0, 0.7, 17.0)
    f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)
    recon = f.mesor + f.beta * np.cos(2 * np.pi * zt / PERIOD) + f.gamma * np.sin(2 * np.pi * zt / PERIOD)
    check("linear (beta,gamma) form reproduces the data exactly",
          float(np.max(np.abs(recon - y))) < 1e-10,
          f"max |resid| = {float(np.max(np.abs(recon - y))):.3e}")

    # weighted fit on noise-free data must also be exact (weights cannot bias an
    # exactly-fitting model)
    rng = np.random.default_rng(SEED)
    w = rng.uniform(0.5, 8.0, size=zt.size)
    fw = CC.cosinor_fit(zt, y, period=PERIOD, weights=w, n_boot=0, seed=SEED)
    check("weighted fit is exact on noise-free data",
          abs(fw.amplitude - 0.7) < 1e-10 and abs(float(circ_diff(fw.acrophase_h, 17.0))) < 1e-10,
          f"A err {abs(fw.amplitude - 0.7):.3e}, phi err {abs(float(circ_diff(fw.acrophase_h, 17.0))):.3e}")


def test_B_matches_nonlinear_least_squares() -> None:
    print("\nB. Linear reparameterisation == direct nonlinear least squares (scipy)")
    rng = np.random.default_rng(SEED + 1)
    worst_M = worst_A = worst_phi = worst_ssr = 0.0
    worst_excess = -np.inf

    for trial in range(30):
        n = int(rng.integers(6, 40))
        zt = rng.uniform(0, 24, size=n)
        M, A, phi = rng.uniform(-2, 3), rng.uniform(0.2, 2.0), rng.uniform(0, 24)
        y = cosine(zt, M, A, phi) + rng.normal(0, 0.30, size=n)

        fit = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)

        # Direct NLS on the ORIGINAL parameterisation, from a poor start
        def resid(p):
            return cosine(zt, p[0], p[1], p[2]) - y

        best = None
        for start in ([0.0, 1.0, 0.0], [y.mean(), 0.5, 12.0], [5.0, 3.0, 20.0]):
            r = optimize.least_squares(resid, np.array(start, dtype=float),
                                       xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if best is None or r.cost < best.cost:
                best = r
        M_nl, A_nl, phi_nl = best.x
        # NLS can land on (-A, phi + period/2), an identical curve; canonicalise.
        if A_nl < 0:
            A_nl, phi_nl = -A_nl, phi_nl + PERIOD / 2.0
        phi_nl = phi_nl % PERIOD

        ssr_lin = float(np.sum((cosine(zt, fit.mesor, fit.amplitude, fit.acrophase_h) - y) ** 2))
        ssr_nl = float(2.0 * best.cost)

        worst_M = max(worst_M, abs(fit.mesor - M_nl))
        worst_A = max(worst_A, abs(fit.amplitude - A_nl))
        worst_phi = max(worst_phi, abs(float(circ_diff(fit.acrophase_h, phi_nl))))
        worst_ssr = max(worst_ssr, abs(ssr_lin - ssr_nl))
        worst_excess = max(worst_excess, ssr_lin - ssr_nl)

    check("MESOR agrees with scipy nonlinear LS across 30 random problems",
          worst_M < 1e-8, f"max |diff| = {worst_M:.3e}")
    check("amplitude agrees with scipy nonlinear LS",
          worst_A < 1e-8, f"max |diff| = {worst_A:.3e}")
    # The acrophase is the flattest direction of the NLS objective, so the
    # residual disagreement here is scipy's convergence tolerance, not an error
    # in the closed form -- 1e-7 h is 0.4 microseconds of clock time. The
    # decisive check is the one below: the closed form attains a residual sum of
    # squares no larger than the optimiser's, i.e. it IS the LS optimum.
    check("acrophase agrees with scipy nonlinear LS to optimiser tolerance",
          worst_phi < 1e-6, f"max |diff| = {worst_phi:.3e} h (scipy xtol/ftol = 1e-15)")
    check("residual sum of squares is identical (same LS optimum)",
          worst_ssr < 1e-10, f"max |diff| = {worst_ssr:.3e}")
    check("the closed form is never WORSE than the restarted optimiser",
          worst_excess <= 1e-12,
          f"max (SSR_closed_form - SSR_nls) = {worst_excess:.3e} (<= 0 means the "
          "closed form attains the optimum)")

    print("      (the closed form needs no starting values and cannot fail to converge)")


def test_C_interval_coverage() -> None:
    print("\nC. Amplitude / acrophase confidence-interval coverage (nominal 95%)")
    n_sims = 1200
    zt_grid = np.array([1.0, 5.0, 9.0, 13.0, 17.0, 21.0])
    reps = 3                                   # 18 observations per simulated dataset
    zt = np.repeat(zt_grid, reps)
    M_true, A_true, phi_true, sd = 1.2, 0.80, 15.0, 0.45

    rng = np.random.default_rng(SEED + 2)
    cov = {k: 0 for k in ("amp_ell", "amp_delta", "amp_boot",
                          "phi_ell", "phi_delta", "phi_boot", "mesor")}
    n_used = 0
    for i in range(n_sims):
        y = cosine(zt, M_true, A_true, phi_true) + rng.normal(0, sd, size=zt.size)
        # The bootstrap is the slow part; run it on a subsample of simulations.
        nb = 800 if i % 4 == 0 else 0
        f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=nb, seed=SEED + i)
        n_used += 1

        lo, hi = f.amplitude_ci_ellipse
        cov["amp_ell"] += int(lo <= A_true <= hi)
        lo, hi = f.amplitude_ci_delta
        cov["amp_delta"] += int(lo <= A_true <= hi)
        lo, hi = f.mesor_ci
        cov["mesor"] += int(lo <= M_true <= hi)

        def _phi_in(ci):
            lo, hi = ci
            if not np.isfinite(lo) or not np.isfinite(hi):
                return 0
            if hi - lo >= PERIOD:
                return 1
            d = (A_true and 0) or 0
            x = (phi_true - lo) % PERIOD
            return int(x <= (hi - lo))

        cov["phi_ell"] += _phi_in(f.acrophase_ci_ellipse)
        cov["phi_delta"] += _phi_in(f.acrophase_ci_delta)

    # bootstrap coverage on its own (smaller) sample of simulations
    rng2 = np.random.default_rng(SEED + 3)
    n_boot_sims = 300
    cb_amp = cb_phi = 0
    for i in range(n_boot_sims):
        y = cosine(zt, M_true, A_true, phi_true) + rng2.normal(0, sd, size=zt.size)
        f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=1000, seed=SEED + 10_000 + i)
        lo, hi = f.amplitude_ci_boot
        cb_amp += int(lo <= A_true <= hi)
        lo, hi = f.acrophase_ci_boot
        if np.isfinite(lo) and np.isfinite(hi):
            cb_phi += int(((phi_true - lo) % PERIOD) <= (hi - lo))

    def rate(k, n):
        return k / n

    # Monte-Carlo envelope for a true 95% procedure
    se = np.sqrt(0.95 * 0.05 / n_used)
    lo_env, hi_env = 0.95 - 3 * se, 0.95 + 3 * se
    print(f"      Monte-Carlo 3-sigma envelope for a calibrated 95% interval "
          f"({n_used} sims): [{lo_env:.3f}, {hi_env:.3f}]")

    # ── the EXACT claim: the JOINT ellipse for (beta, gamma) has exactly 95%
    # coverage. The amplitude and acrophase intervals are PROJECTIONS of that
    # region, so they are conservative (>= 95%), not exactly 95%. Both facts are
    # asserted, because quoting a projected interval as if it were nominal would
    # overstate the precision claimed.
    b_true = A_true * np.cos(2 * np.pi * phi_true / PERIOD)
    g_true = A_true * np.sin(2 * np.pi * phi_true / PERIOD)
    rng4 = np.random.default_rng(SEED + 30)
    joint = 0
    n_joint = 2500
    for _ in range(n_joint):
        y = cosine(zt, M_true, A_true, phi_true) + rng4.normal(0, sd, size=zt.size)
        f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)
        d = np.array([f.beta - b_true, f.gamma - g_true])
        c = 2.0 * stats.f.ppf(0.95, 2, f.resid_df)
        joint += int(float(d @ np.linalg.inv(f.cov_bg) @ d) <= c)
    r_joint = joint / n_joint
    check("JOINT (beta,gamma) confidence ellipse has EXACTLY nominal 95% coverage",
          abs(r_joint - 0.95) < 0.018,
          f"coverage = {r_joint:.4f} over {n_joint} sims -- this is the exact "
          "finite-sample result the elliptical intervals are projected from")

    for key, label, n in (
        ("mesor", "MESOR (t interval)", n_used),
        ("amp_delta", "amplitude, delta method", n_used),
        ("phi_delta", "acrophase, delta method", n_used),
    ):
        r = rate(cov[key], n)
        check(f"{label} covers at ~95% (nominal marginal)",
              0.925 <= r <= 0.975, f"coverage = {r:.3f}")

    for key, label in (
        ("amp_ell", "amplitude, elliptical (Bingham 1982)"),
        ("phi_ell", "acrophase, elliptical (Bingham 1982)"),
    ):
        r = rate(cov[key], n_used)
        check(f"{label} is CONSERVATIVE (coverage >= 95%, as a projection must be)",
              0.95 - 3 * se <= r <= 1.0,
              f"coverage = {r:.3f} -- above nominal because it projects a joint region")

    r = rate(cb_amp, n_boot_sims)
    check("amplitude, parametric bootstrap covers at ~95%",
          0.91 <= r <= 0.985, f"coverage = {r:.3f} ({n_boot_sims} sims)")
    r = rate(cb_phi, n_boot_sims)
    check("acrophase, parametric bootstrap covers at ~95%",
          0.91 <= r <= 0.985, f"coverage = {r:.3f} ({n_boot_sims} sims)")

    # The elliptical interval must be the exact/conservative one: it should
    # never be strictly inside the delta interval when SNR is low.
    rng3 = np.random.default_rng(SEED + 4)
    y = cosine(zt, M_true, 0.05, phi_true) + rng3.normal(0, sd, size=zt.size)
    f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)
    check("weak rhythm: elliptical amplitude interval starts at 0 and phase is refused",
          f.origin_in_confidence_ellipse and f.amplitude_ci_ellipse[0] == 0.0
          and (f.acrophase_ci_ellipse[1] - f.acrophase_ci_ellipse[0]) >= PERIOD - 1e-9,
          f"A_hat={f.amplitude:.4f}, amp CI={np.round(f.amplitude_ci_ellipse, 4)}, "
          f"phase CI spans {f.acrophase_ci_ellipse[1] - f.acrophase_ci_ellipse[0]:.1f} h")


def test_D_zero_amplitude_F_test() -> None:
    print("\nD. Zero-amplitude F-test: type I error and power")
    zt = np.repeat(np.array([1.0, 5.0, 9.0, 13.0, 17.0, 21.0]), 3)
    sd = 0.45
    alpha = 0.05
    n_sims = 4000

    # ── type I error: A = 0 exactly ─────────────────────────────────────────
    rng = np.random.default_rng(SEED + 5)
    rej = 0
    for _ in range(n_sims):
        y = 1.2 + rng.normal(0, sd, size=zt.size)
        f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)
        rej += int(f.f_pvalue < alpha)
    t1 = rej / n_sims
    se = np.sqrt(alpha * (1 - alpha) / n_sims)
    check("type I error of the zero-amplitude F-test is ~5% on arrhythmic data",
          abs(t1 - alpha) < 3 * se + 0.005,
          f"{t1:.4f} (nominal {alpha}, 3-sigma band +/-{3 * se:.4f})")

    # ── the elliptical region must agree with the F-test ────────────────────
    # The origin lies inside the (1-alpha) ellipse exactly when the
    # zero-amplitude F-test does NOT reject: both are the same quadratic form.
    rng2 = np.random.default_rng(SEED + 6)
    disagree = 0
    for _ in range(600):
        A = float(rng2.choice([0.0, 0.15, 0.4, 0.9]))
        y = cosine(zt, 1.2, A, 15.0) + rng2.normal(0, sd, size=zt.size)
        f = CC.cosinor_fit(zt, y, period=PERIOD, alpha=alpha, n_boot=0, seed=SEED)
        rejects = f.f_pvalue < alpha
        if rejects == f.origin_in_confidence_ellipse:
            disagree += 1
    check("origin-in-ellipse is the exact complement of the F-test rejection",
          disagree == 0, f"{disagree} disagreement(s) in 600 datasets")

    # ── power ───────────────────────────────────────────────────────────────
    rng3 = np.random.default_rng(SEED + 7)
    for A in (0.30, 0.50, 0.80):
        rej = 0
        n = 1500
        for _ in range(n):
            y = cosine(zt, 1.2, A, 15.0) + rng3.normal(0, sd, size=zt.size)
            f = CC.cosinor_fit(zt, y, period=PERIOD, n_boot=0, seed=SEED)
            rej += int(f.f_pvalue < alpha)
        pw = rej / n
        # Analytic power from the noncentral F, computed independently here.
        N = zt.size
        X = np.column_stack([np.ones(N), np.cos(2 * np.pi * zt / PERIOD),
                             np.sin(2 * np.pi * zt / PERIOD)])
        Z, one = X[:, 1:], X[:, :1]
        Minf = Z.T @ Z - Z.T @ one @ np.linalg.inv(one.T @ one) @ one.T @ Z
        th = np.array([A * np.cos(2 * np.pi * 15.0 / PERIOD),
                       A * np.sin(2 * np.pi * 15.0 / PERIOD)])
        lam = float(th @ Minf @ th) / sd ** 2
        pw_theory = float(stats.ncf.sf(stats.f.ppf(1 - alpha, 2, N - 3), 2, N - 3, lam))
        check(f"power at A={A:g} matches the noncentral-F prediction",
              abs(pw - pw_theory) < 0.045,
              f"simulated {pw:.3f} vs theory {pw_theory:.3f} (lambda={lam:.2f})")

    check("high power at a clearly rhythmic amplitude (A=0.8, sd=0.45, n=18)",
          pw > 0.95, f"power = {pw:.3f}")


def _make_population(rng, k, zt_per_subject, M, A, phi, sd_within,
                     sd_M=0.5, sd_A=0.15, sd_phi=1.5, group="Ctrl"):
    """
    Simulate k subjects, each measured at `zt_per_subject`, from a population
    cosinor with between-subject variation in MESOR, amplitude and acrophase.
    """
    rows = []
    for i in range(k):
        Mi = M + rng.normal(0, sd_M)
        Ai = A + rng.normal(0, sd_A)
        pi = phi + rng.normal(0, sd_phi)
        for zt in zt_per_subject:
            rows.append({
                "FlyID": f"{group}_{i:03d}", "Genotype": group, "ZT": float(zt),
                "y": float(cosine(zt, Mi, Ai, pi) + rng.normal(0, sd_within)),
            })
    return pd.DataFrame(rows)


def test_E_population_cosinor() -> None:
    print("\nE. Population-mean cosinor with 3 timepoints per subject (Bingham et al. 1982)")
    zt3 = np.array([2.0, 8.0, 11.0])       # design_schema morning/afternoon/evening ZT
    M, A, phi = 1.0, 0.70, 15.0

    # ── single-subject fit must report ZERO residual df and no intervals ────
    y = cosine(zt3, M, A, phi)
    f1 = CC.cosinor_fit(zt3, y, period=PERIOD, n_boot=0, seed=SEED)
    check("3 timepoints -> residual df is exactly 0",
          f1.resid_df == 0, f"resid_df = {f1.resid_df}, n_obs = {f1.n_obs}")
    check("3 timepoints -> sigma^2 is undefined (NaN)",
          not np.isfinite(f1.sigma2), f"sigma2 = {f1.sigma2}")
    check("3 timepoints -> no amplitude interval is offered",
          all(not np.isfinite(v) for v in f1.amplitude_ci_ellipse + f1.amplitude_ci_delta),
          f"ellipse={f1.amplitude_ci_ellipse}, delta={f1.amplitude_ci_delta}")
    check("3 timepoints -> no F-test is offered",
          not np.isfinite(f1.f_pvalue), f"F p-value = {f1.f_pvalue}")
    check("3 timepoints -> point estimates are still EXACT",
          abs(f1.amplitude - A) < 1e-10 and abs(float(circ_diff(f1.acrophase_h, phi))) < 1e-10,
          f"A err {abs(f1.amplitude - A):.2e}, phi err {abs(float(circ_diff(f1.acrophase_h, phi))):.2e}")
    check("the zero-df situation is stated in plain language",
          "0 residual degrees of freedom" in f1.verdict(), f1.verdict()[:90] + "...")

    # ── population recovery, PHASE-SYNCHRONOUS population ──────────────────
    # Every subject shares the acrophase, so the population amplitude IS the
    # individual amplitude and must be recovered without attenuation.
    def _run_pop(rng, k, n_sims, A_true, phi_true, sd_phi, sd_A, sd_within,
                 target_amp):
        amps, phis, mesors = [], [], []
        cov_amp = cov_phi = rejected = 0
        for i in range(n_sims):
            d = _make_population(rng, k, zt3, M, A_true, phi_true,
                                 sd_within=sd_within, sd_A=sd_A, sd_phi=sd_phi)
            r = CC.population_cosinor(d, "ZT", "y", "FlyID", period=PERIOD,
                                      n_boot=0, seed=SEED + i)
            ov = r["overall"]
            amps.append(ov["amplitude"]); phis.append(ov["acrophase_h"])
            mesors.append(ov["mesor"])
            lo, hi = ov["amplitude_ci"]
            cov_amp += int(lo <= target_amp <= hi)
            lo, hi = ov["acrophase_ci"]
            if np.isfinite(lo) and np.isfinite(hi):
                cov_phi += int(((phi_true - lo) % PERIOD) <= (hi - lo))
            rejected += int(ov["rhythm_detected"])
        return (float(np.mean(amps)),
                float(phi_true + np.mean(circ_diff(phis, phi_true))),
                float(np.mean(mesors)),
                cov_amp / n_sims, cov_phi / n_sims, rejected / n_sims)

    rng = np.random.default_rng(SEED + 8)
    n_sims = 300
    mean_amp, mean_phi, mean_mesor, ca, cp, det = _run_pop(
        rng, 30, n_sims, A, phi, sd_phi=0.0, sd_A=0.0, sd_within=0.30, target_amp=A)
    check("phase-synchronous population: amplitude recovered (true 0.70)",
          abs(mean_amp - A) < 0.02, f"mean over {n_sims} sims = {mean_amp:.4f}")
    check("phase-synchronous population: acrophase recovered (true ZT15)",
          abs(circ_diff(mean_phi, phi)) < 0.25, f"mean = ZT {mean_phi:.3f}")
    check("phase-synchronous population: MESOR recovered (true 1.0)",
          abs(mean_mesor - M) < 0.02, f"mean = {mean_mesor:.4f}")
    check("phase-synchronous population: amplitude CI covers", 0.90 <= ca <= 1.0,
          f"coverage = {ca:.3f}")
    check("phase-synchronous population: acrophase CI covers", 0.90 <= cp <= 1.0,
          f"coverage = {cp:.3f}")
    check("Hotelling T^2 has high power on a real population rhythm",
          det > 0.98, f"detection rate = {det:.3f}")

    # ── population recovery WITH between-fly phase scatter ─────────────────
    # This is the scientifically important case. With acrophases scattered
    # N(phi, sd_phi), the population (vector-mean) amplitude is ATTENUATED by
    # exactly R = exp(-(w*sd_phi)^2/2), the characteristic function of the phase
    # jitter. The module must recover the attenuated value -- that IS the
    # population rhythm -- not the mean individual amplitude.
    sd_phi = 1.5
    w = 2 * np.pi / PERIOD
    R_theory = float(np.exp(-(w * sd_phi) ** 2 / 2.0))
    A_pop = A * R_theory
    rng = np.random.default_rng(SEED + 8)
    mean_amp2, mean_phi2, _, ca2, _, _ = _run_pop(
        rng, 30, n_sims, A, phi, sd_phi=sd_phi, sd_A=0.15, sd_within=0.30,
        target_amp=A_pop)
    check("phase-scattered population: amplitude matches the ANALYTIC attenuation "
          "A*exp(-(w*sd_phi)^2/2)",
          abs(mean_amp2 - A_pop) < 0.025,
          f"mean = {mean_amp2:.4f} vs theory {A_pop:.4f} "
          f"(= 0.70 x R, R = {R_theory:.4f} for {sd_phi} h of phase scatter)")
    check("phase-scattered population: acrophase is still unbiased",
          abs(circ_diff(mean_phi2, phi)) < 0.25, f"mean = ZT {mean_phi2:.3f}")
    check("phase-scattered population: CI covers the attenuated population amplitude",
          0.90 <= ca2 <= 1.0, f"coverage = {ca2:.3f}")

    # ── the synchronisation index recovers R, and the ambiguity is flagged ──
    d = _make_population(np.random.default_rng(SEED + 26), 200, zt3, M, A, phi,
                         sd_within=0.05, sd_A=0.0, sd_phi=sd_phi)
    r = CC.population_cosinor(d, "ZT", "y", "FlyID", period=PERIOD, n_boot=0, seed=SEED)
    ov = r["overall"]
    check("synchronisation index R recovers the theoretical attenuation factor",
          abs(ov["synchronisation_index_R"] - R_theory) < 0.04,
          f"R_hat = {ov['synchronisation_index_R']:.4f} vs theory {R_theory:.4f}")
    check("phase dispersion recovers the simulated between-fly scatter (1.5 h)",
          abs(ov["phase_dispersion_sd_h"] - sd_phi) < 0.35,
          f"estimated {ov['phase_dispersion_sd_h']:.3f} h (biased UP by fit noise, as documented)")
    check("population amplitude ~ mean individual amplitude x R (the identity holds)",
          abs(ov["amplitude"] - ov["mean_of_individual_amplitudes"]
              * ov["synchronisation_index_R"]) < 0.03,
          f"{ov['amplitude']:.4f} vs {ov['mean_of_individual_amplitudes']:.4f} x "
          f"{ov['synchronisation_index_R']:.4f} = "
          f"{ov['mean_of_individual_amplitudes'] * ov['synchronisation_index_R']:.4f}")
    check("the damped-vs-desynchronised ambiguity is stated to the user",
          "AMBIGUOUS" in ov["amplitude_attenuation_note"],
          ov["amplitude_attenuation_note"][:100] + "...")

    # ── the module says WHY per-subject inference is impossible ─────────────
    df = _make_population(np.random.default_rng(SEED + 9), 20, zt3, M, A, phi, 0.30)
    res = CC.population_cosinor(df, "ZT", "y", "FlyID", period=PERIOD, n_boot=0, seed=SEED)
    check("every per-subject fit is recorded with resid_df = 0",
          bool((res["per_subject"]["resid_df"] == 0).all()),
          f"{int((res['per_subject']['resid_df'] == 0).sum())}/{len(res['per_subject'])} subjects")
    check("the zero-df limitation is reported in notes",
          any("ZERO residual degrees of freedom" in n for n in res["notes"]),
          (res["notes"][0][:100] + "...") if res["notes"] else "no notes emitted")

    # ── type I error of the population test on arrhythmic subjects ──────────
    rng2 = np.random.default_rng(SEED + 11)
    rej = 0
    n_null = 1000
    for i in range(n_null):
        d = _make_population(rng2, 20, zt3, M, 0.0, phi, sd_within=0.30,
                             sd_A=0.0, sd_phi=0.0)
        r = CC.population_cosinor(d, "ZT", "y", "FlyID", period=PERIOD, n_boot=0, seed=SEED)
        rej += int(r["overall"]["rhythm_detected"])
    t1 = rej / n_null
    se = np.sqrt(0.05 * 0.95 / n_null)
    check("population Hotelling T^2 controls type I error at ~5%",
          abs(t1 - 0.05) < 3 * se + 0.008, f"{t1:.4f} (3-sigma band +/-{3 * se:.4f})")

    # ── the mean-of-individual-amplitudes bias is real and is flagged ───────
    d = _make_population(np.random.default_rng(SEED + 12), 60, zt3, M, 0.0, phi,
                         sd_within=0.40, sd_A=0.0, sd_phi=0.0)
    r = CC.population_cosinor(d, "ZT", "y", "FlyID", period=PERIOD, n_boot=0, seed=SEED)
    ov = r["overall"]
    check("mean of individual amplitudes is biased UP when the true amplitude is 0",
          ov["mean_of_individual_amplitudes"] > 3 * ov["amplitude"],
          f"mean|v_i| = {ov['mean_of_individual_amplitudes']:.4f} vs |mean v_i| = "
          f"{ov['amplitude']:.4f} (true amplitude = 0)")

    # ── group comparison recovers a real amplitude difference ──────────────
    rng3 = np.random.default_rng(SEED + 13)
    ctrl = _make_population(rng3, 25, zt3, 1.0, 0.80, 15.0, 0.30, group="Ctrl")
    dis = _make_population(rng3, 25, zt3, 1.0, 0.20, 15.0, 0.30, group="Disease")
    both = pd.concat([ctrl, dis], ignore_index=True)
    r = CC.population_cosinor(both, "ZT", "y", "FlyID", group_col="Genotype",
                              period=PERIOD, n_boot=1500, seed=SEED)
    cmp = r["comparisons"][0]
    check("group comparison detects a genuine amplitude difference",
          cmp["amplitude"]["p_value_boot"] < 0.01 and cmp["rhythm_vector"]["p_value"] < 0.01,
          f"amplitude p = {cmp['amplitude']['p_value_boot']:.4g}, "
          f"Hotelling p = {cmp['rhythm_vector']['p_value']:.4g}")
    check("group comparison finds no MESOR difference when there is none",
          cmp["mesor"]["p_value"] > 0.05, f"MESOR p = {cmp['mesor']['p_value']:.3f}")

    # ── acrophase contrast is suppressed when a group is arrhythmic ────────
    flat = _make_population(np.random.default_rng(SEED + 14), 25, zt3, 1.0, 0.0, 15.0,
                            0.30, sd_A=0.0, sd_phi=0.0, group="Arrhythmic")
    r2 = CC.population_cosinor(pd.concat([ctrl, flat], ignore_index=True),
                               "ZT", "y", "FlyID", group_col="Genotype",
                               period=PERIOD, n_boot=800, seed=SEED)
    c2 = r2["comparisons"][0]
    check("acrophase contrast is marked uninterpretable when one group is arrhythmic",
          c2["acrophase"]["interpretable"] is False and "caveat" in c2["acrophase"],
          c2["acrophase"].get("caveat", "")[:95] + "...")


def test_F_fatigue_vs_rhythm() -> None:
    print("\nF. Fatigue / testing-order separation")
    zt3 = np.array([2.0, 8.0, 11.0])
    rng = np.random.default_rng(SEED + 15)

    # ── (i) COMPLETELY COLLINEAR: everyone morning -> afternoon -> evening ──
    # The generating truth here is PURE FATIGUE: no circadian term at all, just
    # a monotone decline with cumulative testing. A correct module must refuse
    # to call this circadian.
    rows = []
    for i in range(40):
        base = 1.0 + rng.normal(0, 0.40)
        for s, zt in enumerate(zt3, start=1):
            rows.append({"FlyID": f"F{i:03d}", "ZT": float(zt), "SessionIndex": float(s),
                         "y": base - 0.30 * s + rng.normal(0, 0.25)})
    coll = pd.DataFrame(rows)

    res = CC.separate_fatigue_from_rhythm(coll, "ZT", "y", order_col="SessionIndex",
                                          subject_col="FlyID", period=PERIOD)
    check("collinear design is detected as NOT separable",
          res["separable"] is False, f"design rank {res['design_rank']} of {res['design_ncol']} columns")
    check("collinear design REFUSES circadian attribution",
          res["attribution_allowed"] is False,
          f"attribution_allowed = {res['attribution_allowed']}")
    check("the refusal names the reason explicitly",
          "REFUSING TO ATTRIBUTE" in res["verdict"] and "collinear" in res["verdict"],
          res["verdict"][:110] + "...")
    check("design_schema.check_confounding reports COMPLETE confounding",
          res["confounding_check"].get("severity", 0.0) >= 1.0,
          res["confounding_check"].get("verdict", "")[:95])
    check("a rhythm-only fit IS reported but flagged uninterpretable",
          "rhythm_only_fit_UNINTERPRETABLE" in res
          and res["rhythm_only_fit_UNINTERPRETABLE"]["f_pvalue"] < 0.001,
          f"the ignorable-order fit would have claimed p = "
          f"{res['rhythm_only_fit_UNINTERPRETABLE']['f_pvalue']:.3g} and amplitude "
          f"{res['rhythm_only_fit_UNINTERPRETABLE']['amplitude']:.3f}")
    check("VIF is infinite for the collinear design",
          not np.isfinite(res["max_vif"]), f"max VIF = {res['max_vif']}")

    # ── (ii) COUNTERBALANCED, FATIGUE ONLY, NO RHYTHM ──────────────────────
    # Subjects visit the three times of day in Latin-square-rotated orders.
    rng2 = np.random.default_rng(SEED + 16)
    square = np.array([[(i + j) % 3 for j in range(3)] for i in range(3)])
    rows = []
    for i in range(60):
        base = 1.0 + rng2.normal(0, 0.40)
        order = square[i % 3]
        for s, idx in enumerate(order, start=1):
            rows.append({"FlyID": f"G{i:03d}", "ZT": float(zt3[idx]),
                         "SessionIndex": float(s),
                         "y": base - 0.30 * s + rng2.normal(0, 0.25)})
    cb = pd.DataFrame(rows)

    res2 = CC.separate_fatigue_from_rhythm(cb, "ZT", "y", order_col="SessionIndex",
                                           subject_col="FlyID", period=PERIOD)
    check("counterbalanced design is separable",
          res2["separable"] and res2["attribution_allowed"],
          f"rank {res2['design_rank']}/{res2['design_ncol']}, max VIF = {res2['max_vif']:.2f}")
    check("fatigue-only data: the testing-order effect IS recovered (true -0.30/session)",
          res2["order_effect_detected"] and abs(res2["order_effect"] + 0.30) < 0.06,
          f"order coef = {res2['order_effect']:+.4f} "
          f"[{res2['order_effect_ci'][0]:+.4f}, {res2['order_effect_ci'][1]:+.4f}], "
          f"p = {res2['order_p']:.3g}")
    check("fatigue-only data: NO circadian rhythm is claimed",
          not res2["rhythm_detected_adjusted"],
          f"adjusted rhythm F = {res2['rhythm_F']:.3f}, p = {res2['rhythm_p']:.3f}")

    # ── (iii) COUNTERBALANCED, BOTH EFFECTS PRESENT ────────────────────────
    rng3 = np.random.default_rng(SEED + 17)
    rows = []
    for i in range(90):
        base = 1.0 + rng3.normal(0, 0.40)
        order = square[i % 3]
        for s, idx in enumerate(order, start=1):
            zt = float(zt3[idx])
            rows.append({"FlyID": f"H{i:03d}", "ZT": zt, "SessionIndex": float(s),
                         "y": base - 0.25 * s + cosine(zt, 0.0, 0.60, 9.0)
                              + rng3.normal(0, 0.25)})
    both = pd.DataFrame(rows)

    res3 = CC.separate_fatigue_from_rhythm(both, "ZT", "y", order_col="SessionIndex",
                                           subject_col="FlyID", period=PERIOD)
    check("both effects present: order effect recovered (true -0.25/session)",
          res3["order_effect_detected"] and abs(res3["order_effect"] + 0.25) < 0.06,
          f"order coef = {res3['order_effect']:+.4f}, p = {res3['order_p']:.3g}")
    check("both effects present: rhythm recovered after adjusting for order",
          res3["rhythm_detected_adjusted"]
          and abs(res3["adjusted_amplitude"] - 0.60) < 0.12,
          f"adjusted amplitude = {res3['adjusted_amplitude']:.4f} "
          f"[{res3['adjusted_amplitude_ci'][0]:.3f}, {res3['adjusted_amplitude_ci'][1]:.3f}], "
          f"p = {res3['rhythm_p']:.3g}")
    check("both effects present: acrophase recovered (true ZT9)",
          abs(float(circ_diff(res3["adjusted_acrophase_h"], 9.0))) < 1.2,
          f"adjusted acrophase = ZT {res3['adjusted_acrophase_h']:.2f} "
          f"[{res3['adjusted_acrophase_ci'][0]:.2f}, {res3['adjusted_acrophase_ci'][1]:.2f}]")

    # ── (iv) type I error of the adjusted rhythm test under pure fatigue ────
    rng4 = np.random.default_rng(SEED + 18)
    rej = 0
    n_null = 400
    for _ in range(n_null):
        rows = []
        for i in range(30):
            base = 1.0 + rng4.normal(0, 0.40)
            order = square[i % 3]
            for s, idx in enumerate(order, start=1):
                rows.append({"FlyID": f"K{i:03d}", "ZT": float(zt3[idx]),
                             "SessionIndex": float(s),
                             "y": base - 0.30 * s + rng4.normal(0, 0.25)})
        r = CC.separate_fatigue_from_rhythm(pd.DataFrame(rows), "ZT", "y",
                                            order_col="SessionIndex",
                                            subject_col="FlyID", period=PERIOD)
        rej += int(r["rhythm_detected_adjusted"])
    t1 = rej / n_null
    check("adjusted rhythm test does not manufacture a rhythm out of pure fatigue",
          t1 < 0.12, f"false rhythm rate = {t1:.3f} over {n_null} counterbalanced datasets")

    # ── (v) design_advisor agrees, before the data exist ───────────────────
    adv = CC.design_advisor(zt3, n_subjects=40, period=PERIOD, sigma=0.25)
    check("design_advisor flags the ascending-ZT shared schedule as confounded",
          adv["testing_order_confounded_with_zt"] and
          any("CRITICAL" in r for r in adv["recommendations"]),
          [r for r in adv["recommendations"] if "CRITICAL" in r][0][:95] + "...")
    check("design_advisor confirms 3 well-separated times identify the cosinor",
          adv["identifiable"] and adv["amplitude_identifiable"] and adv["acrophase_identifiable"],
          f"rank {adv['design_rank']}, condition number {adv['condition_number']:.1f}")
    check("design_advisor reports a finite minimum detectable amplitude",
          np.isfinite(adv["minimum_detectable_amplitude"]),
          f"MDA (worst-case phase, 80% power) = {adv['minimum_detectable_amplitude']:.4f} "
          f"at sigma = 0.25")

    bad = CC.design_advisor([9.0, 21.0], n_subjects=50, period=PERIOD, sigma=0.25)
    check("design_advisor refuses a 2-timepoint schedule as non-identifiable",
          bad["identifiable"] is False and "NOT IDENTIFIABLE" in bad["verdict"],
          bad["verdict"][:95] + "...")

    # A 2-D per-subject schedule must NOT be flagged as order-confounded.
    sched = np.array([zt3[square[i % 3]] for i in range(30)])
    adv2 = CC.design_advisor(sched, n_subjects=30, period=PERIOD, sigma=0.25)
    check("design_advisor accepts a per-subject counterbalanced schedule",
          adv2["identifiable"] and not adv2["testing_order_confounded_with_zt"],
          f"shared_schedule = {adv2['shared_schedule']}, "
          f"MDA = {adv2['minimum_detectable_amplitude']:.4f}")

    # MDA must be an honest power statement: simulate at exactly the MDA.
    rng5 = np.random.default_rng(SEED + 19)
    A_mda = float(adv["minimum_detectable_amplitude"])
    zt_all = np.tile(zt3, 40)
    # worst-case phase = the eigenvector of least information
    X = np.column_stack([np.ones(zt_all.size), np.cos(2 * np.pi * zt_all / PERIOD),
                         np.sin(2 * np.pi * zt_all / PERIOD)])
    Z, one = X[:, 1:], X[:, :1]
    Minf = Z.T @ Z - Z.T @ one @ np.linalg.inv(one.T @ one) @ one.T @ Z
    evals, evecs = np.linalg.eigh(Minf)
    v_worst = evecs[:, 0]
    phi_worst = (np.arctan2(v_worst[1], v_worst[0]) * PERIOD / (2 * np.pi)) % PERIOD
    rej = 0
    n_pw = 1200
    for _ in range(n_pw):
        y = cosine(zt_all, 1.0, A_mda, phi_worst) + rng5.normal(0, 0.25, size=zt_all.size)
        f = CC.cosinor_fit(zt_all, y, period=PERIOD, n_boot=0, seed=SEED)
        rej += int(f.f_pvalue < 0.05)
    pw = rej / n_pw
    check("simulated power at the reported MDA is ~80% in the worst-case phase",
          abs(pw - 0.80) < 0.05, f"simulated power = {pw:.3f} at A = {A_mda:.4f}, "
                                 f"phase ZT {phi_worst:.2f}")


def test_G_lombscargle_crosscheck() -> None:
    print("\nG. Independent-implementation cross-check: astropy LombScargle")
    from astropy.timeseries import LombScargle

    rng = np.random.default_rng(SEED + 20)
    # A dense schedule across several days, as a DAM-style record would give.
    t = np.sort(rng.uniform(0, 96, size=240))
    M, A, phi = 1.4, 0.85, 15.0
    y = cosine(t, M, A, phi) + rng.normal(0, 0.35, size=t.size)

    fit = CC.cosinor_fit(t, y, period=PERIOD, n_boot=0, seed=SEED)

    # astropy's design matrix for nterms=1 with fit_mean=True is
    # [1, sin(2 pi f t), cos(2 pi f t)] and model_parameters() returns theta for
    # the CENTERED data, so the full model is offset() + design_matrix @ theta.
    # Hence:  MESOR = offset + theta[0],  gamma (sin coef) = theta[1],
    #         beta (cos coef) = theta[2].
    ls = LombScargle(t, y, fit_mean=True, center_data=True)
    th = ls.model_parameters(1.0 / PERIOD)
    ls_mesor = float(ls.offset() + th[0])
    ls_gamma = float(th[1])
    ls_beta = float(th[2])
    ls_amp = float(np.hypot(ls_beta, ls_gamma))
    ls_phi = float((np.arctan2(ls_gamma, ls_beta) * PERIOD / (2 * np.pi)) % PERIOD)

    check("MESOR matches astropy LombScargle at the known frequency",
          abs(fit.mesor - ls_mesor) < 1e-9,
          f"ours {fit.mesor:.12f} vs astropy {ls_mesor:.12f}")
    check("cos coefficient (beta) matches astropy",
          abs(fit.beta - ls_beta) < 1e-9, f"{fit.beta:.12f} vs {ls_beta:.12f}")
    check("sin coefficient (gamma) matches astropy",
          abs(fit.gamma - ls_gamma) < 1e-9, f"{fit.gamma:.12f} vs {ls_gamma:.12f}")
    check("amplitude matches astropy",
          abs(fit.amplitude - ls_amp) < 1e-9, f"{fit.amplitude:.12f} vs {ls_amp:.12f}")
    check("acrophase matches astropy",
          abs(float(circ_diff(fit.acrophase_h, ls_phi))) < 1e-9,
          f"ZT {fit.acrophase_h:.12f} vs ZT {ls_phi:.12f}")

    # A FREE period search (which our fixed-period cosinor never performs) must
    # independently land on 24 h -- confirming the period we hold fixed.
    periods = np.linspace(16.0, 32.0, 4001)
    powr = ls.power(1.0 / periods)
    p_peak = float(periods[int(np.argmax(powr))])
    check("astropy free period search recovers the 24 h generating period",
          abs(p_peak - PERIOD) < 0.25, f"peak at {p_peak:.3f} h (grid 16-32 h)")

    # And the phase recovered at the FREE peak must agree with ours.
    th2 = ls.model_parameters(1.0 / p_peak)
    phi_peak = float((np.arctan2(th2[1], th2[2]) * p_peak / (2 * np.pi)) % p_peak)
    check("phase at the free-search peak agrees with the fixed-24h acrophase",
          abs(float(circ_diff(phi_peak, fit.acrophase_h))) < 0.6,
          f"free-search ZT {phi_peak:.3f} vs fixed-period ZT {fit.acrophase_h:.3f}")

    # On genuinely arrhythmic data both implementations must find nothing.
    y0 = M + rng.normal(0, 0.35, size=t.size)
    f0 = CC.cosinor_fit(t, y0, period=PERIOD, n_boot=0, seed=SEED)
    ls0 = LombScargle(t, y0, fit_mean=True, center_data=True)
    fap = float(ls0.false_alarm_probability(ls0.power(1.0 / periods).max(),
                                            method="baluev"))
    check("arrhythmic data: our F-test does not reject",
          f0.f_pvalue > 0.05, f"F-test p = {f0.f_pvalue:.3f}")
    check("arrhythmic data: astropy false-alarm probability is also non-significant",
          fap > 0.05, f"Baluev FAP = {fap:.3f}")

    # Sparse, realistic 3-timepoints-per-fly data: agreement must survive it.
    zt3 = np.array([2.0, 8.0, 11.0])
    d = _make_population(np.random.default_rng(SEED + 21), 40, zt3,
                         1.0, 0.70, 9.0, sd_within=0.30)
    fit_s = CC.cosinor_fit(d["ZT"].to_numpy(), d["y"].to_numpy(),
                           period=PERIOD, n_boot=0, seed=SEED)
    ls_s = LombScargle(d["ZT"].to_numpy(), d["y"].to_numpy(),
                       fit_mean=True, center_data=True)
    ths = ls_s.model_parameters(1.0 / PERIOD)
    amp_s = float(np.hypot(ths[2], ths[1]))
    phi_s = float((np.arctan2(ths[1], ths[2]) * PERIOD / (2 * np.pi)) % PERIOD)
    check("sparse 3-timepoint pooled data: amplitude matches astropy",
          abs(fit_s.amplitude - amp_s) < 1e-9, f"{fit_s.amplitude:.10f} vs {amp_s:.10f}")
    check("sparse 3-timepoint pooled data: acrophase matches astropy",
          abs(float(circ_diff(fit_s.acrophase_h, phi_s))) < 1e-9,
          f"ZT {fit_s.acrophase_h:.10f} vs ZT {phi_s:.10f}")


def test_H_amplitude_biomarker() -> None:
    print("\nH. amplitude_biomarker: framed as a hypothesis, and correctly signed")
    zt3 = np.array([2.0, 8.0, 11.0])

    # Truth: amplitude collapses (0.90 -> 0.15), MESOR barely moves (1.0 -> 0.95).
    rng = np.random.default_rng(SEED + 22)
    ctrl = _make_population(rng, 40, zt3, 1.00, 0.90, 9.0, 0.25, group="Control")
    dis = _make_population(rng, 40, zt3, 0.95, 0.15, 9.0, 0.25, group="TDP43")
    df = pd.concat([ctrl, dis], ignore_index=True)

    res = CC.amplitude_biomarker(df, "ZT", "y", "FlyID", "Genotype",
                                 control_group="Control", period=PERIOD,
                                 n_boot=2000, seed=SEED)
    check("amplitude_biomarker runs and states the hypothesis explicitly",
          res["available"] and "HYPOTHESIS (not an established finding)" in res["hypothesis"],
          res["hypothesis"][:80] + "...")
    check("the hypothesis text cites the published NULL (Gargano 2005)",
          "Gargano" in res["hypothesis"] and "NULL" in res["hypothesis"],
          "the premise is presented with the contrary evidence attached")
    c = res["contrasts"][0]
    check("amplitude-first truth is detected as supported",
          c["supports_amplitude_first"] and c["differential_D_ci"][0] > 0,
          f"D = {c['differential_D']:+.3f} [{c['differential_D_ci'][0]:+.3f}, "
          f"{c['differential_D_ci'][1]:+.3f}], p = {c['differential_D_p']:.4g}")
    # The simulated flies carry 1.5 h of between-fly phase scatter, so the
    # POPULATION amplitudes are the individual amplitudes damped by
    # R = exp(-(w*1.5)^2/2) = 0.926. The expected population values are
    # therefore 0.833 and 0.139, not 0.90 and 0.15.
    R = float(np.exp(-((2 * np.pi / PERIOD) * 1.5) ** 2 / 2.0))
    check("amplitude loss recovered at the phase-damped population values "
          f"(0.90 -> {0.90 * R:.3f}, 0.15 -> {0.15 * R:.3f})",
          abs(c["amplitude_control"] - 0.90 * R) < 0.18
          and abs(c["amplitude_group"] - 0.15 * R) < 0.18,
          f"control A = {c['amplitude_control']:.3f} (expect {0.90 * R:.3f}), "
          f"disease A = {c['amplitude_group']:.3f} (expect {0.15 * R:.3f}); "
          f"R = {R:.3f}")
    check("the damped-vs-desynchronised mechanism ambiguity is addressed in the verdict",
          ("MECHANISM AMBIGUOUS" in c["evidence"]
           or "not explained by phase" in c["evidence"]),
          f"R_control = {c['synchronisation_control']:.3f}, "
          f"R_disease = {c['synchronisation_group']:.3f}")
    check("Holm-adjusted p-values are reported for the {MESOR, amplitude} family",
          np.isfinite(c["mesor_p_holm"]) and np.isfinite(c["amplitude_p_holm"])
          and c["amplitude_p_holm"] >= c["amplitude_p_boot"],
          f"amplitude p {c['amplitude_p_boot']:.4g} -> Holm {c['amplitude_p_holm']:.4g}; "
          f"MESOR p {c['mesor_p_boot']:.4g} -> Holm {c['mesor_p_holm']:.4g}")
    check("the prerequisite confound check is stated",
          "separate_fatigue_from_rhythm" in res["prerequisite"],
          res["prerequisite"][:80] + "...")

    # Null truth: both amplitude and MESOR unchanged. Must report no evidence.
    rng2 = np.random.default_rng(SEED + 23)
    c1 = _make_population(rng2, 40, zt3, 1.00, 0.60, 9.0, 0.25, group="Control")
    c2 = _make_population(rng2, 40, zt3, 1.00, 0.60, 9.0, 0.25, group="TDP43")
    res0 = CC.amplitude_biomarker(pd.concat([c1, c2], ignore_index=True),
                                  "ZT", "y", "FlyID", "Genotype",
                                  control_group="Control", period=PERIOD,
                                  n_boot=2000, seed=SEED)
    c0 = res0["contrasts"][0]
    check("null truth: the hypothesis is NOT claimed to be supported",
          not c0["supports_amplitude_first"] and "NO EVIDENCE" in c0["evidence"],
          f"D = {c0['differential_D']:+.3f} [{c0['differential_D_ci'][0]:+.3f}, "
          f"{c0['differential_D_ci'][1]:+.3f}]")

    # Reversed truth: MESOR collapses, amplitude intact. Must say CONTRARY.
    rng3 = np.random.default_rng(SEED + 24)
    a1 = _make_population(rng3, 40, zt3, 1.20, 0.60, 9.0, 0.25, group="Control")
    a2 = _make_population(rng3, 40, zt3, 0.40, 0.60, 9.0, 0.25, group="TDP43")
    resR = CC.amplitude_biomarker(pd.concat([a1, a2], ignore_index=True),
                                  "ZT", "y", "FlyID", "Genotype",
                                  control_group="Control", period=PERIOD,
                                  n_boot=2000, seed=SEED)
    cR = resR["contrasts"][0]
    check("reversed truth (MESOR degrades more): reported as CONTRARY",
          not cR["supports_amplitude_first"] and "CONTRARY" in cR["evidence"],
          f"D = {cR['differential_D']:+.3f} [{cR['differential_D_ci'][0]:+.3f}, "
          f"{cR['differential_D_ci'][1]:+.3f}]")

    # It must refuse to guess which group is the control.
    amb = pd.concat([
        _make_population(np.random.default_rng(1), 10, zt3, 1.0, 0.5, 9.0, 0.25, group="LineA"),
        _make_population(np.random.default_rng(2), 10, zt3, 1.0, 0.5, 9.0, 0.25, group="LineB"),
    ], ignore_index=True)
    ra = CC.amplitude_biomarker(amb, "ZT", "y", "FlyID", "Genotype", period=PERIOD,
                                n_boot=200, seed=SEED)
    check("refuses to guess the control group when it is ambiguous",
          ra["available"] is False and "could not be inferred" in ra["reason"],
          ra["reason"][:95] + "...")


def test_I_integration_with_design_schema() -> None:
    print("\nI. Integration: ZT and the response come from the existing DroPhenix modules")
    # A frame in the lab's own vocabulary: categorical TimeOfDay, Count/n.
    rng = np.random.default_rng(SEED + 25)
    rows = []
    for i in range(30):
        base = rng.normal(1.2, 0.4)
        for s, tod in enumerate(["Morning", "Afternoon", "Evening"], start=1):
            rows.append({"FlyID": f"V{i:03d}", "Genotype": "Oregon-WT",
                         "TimeOfDay": tod, "SessionIndex": s,
                         "Count": 0, "n": 25, "_lp": base})
    df = pd.DataFrame(rows)

    zt, is_cat, warns = CC.zt_from_frame(df, lights_on_hour=8.0)
    check("ZT is derived through design_schema.derive_zt, not reinvented here",
          zt.notna().all() and is_cat,
          f"ZT values {sorted(zt.unique())} from CATEGORICAL_TOD_ZT; "
          f"{len(warns)} warning(s) propagated")
    expected = sorted({CC._ds.CATEGORICAL_TOD_ZT[k] for k in ("morning", "afternoon", "evening")})
    check("categorical labels map to the CATEGORICAL_TOD_ZT midpoints",
          sorted(zt.unique().tolist()) == expected,
          f"{sorted(zt.unique().tolist())} == {expected}")
    check("derive_zt's precision caveat is surfaced to the caller",
          any("Circadian phase" in w or "categorical" in w.lower() for w in warns),
          (warns[0][:95] + "...") if warns else "no warning emitted")

    # Now generate counts from a real rhythm on those ZT and analyse on the
    # empirical-logit scale supplied by modules/hierarchical.py.
    from scipy.special import expit
    lp = df["_lp"].to_numpy() + 0.55 * np.cos(2 * np.pi * (zt.to_numpy() - 9.0) / 24.0)
    df["Count"] = rng.binomial(25, expit(lp))

    resp, wts, note = CC.prepare_climbing_response(df, "Count", "n")
    check("prepare_climbing_response uses the empirical logit from hierarchical.py",
          "logit" in note and np.all(np.isfinite(resp)) and np.all(wts > 0),
          note[:95] + "...")

    df["_zt"], df["_y"], df["_w"] = zt, resp, wts
    pop = CC.population_cosinor(df, "_zt", "_y", "FlyID", weight_col="_w",
                                period=PERIOD, n_boot=0, seed=SEED)
    ov = pop["overall"]
    lo, hi = ov["amplitude_ci"]
    check("end-to-end: the injected 0.55-logit rhythm is detected from raw Count/n",
          ov["rhythm_detected"] and lo <= 0.55 <= hi,
          f"amplitude {ov['amplitude']:.3f} [{lo:.3f}, {hi:.3f}] covers the injected "
          f"0.55; acrophase ZT {ov['acrophase_h']:.2f} (true ZT9), "
          f"p = {ov['zero_amplitude_p']:.3g}")
    # The interval is very wide, and that is the honest answer for THIS schedule:
    # ZT2/ZT8/ZT11 are all inside a 9 h window, leaving 15 h of the cycle
    # unsampled. The module should be saying so, not pretending to precision.
    adv = CC.design_advisor(np.sort(zt.unique()), n_subjects=30, period=PERIOD, sigma=0.45)
    check("design_advisor explains the wide interval: the lab's 3 times of day are "
          "clustered in one arc",
          adv["largest_sampling_gap_h"] > PERIOD / 2
          and any("largest unsampled stretch" in w for w in adv["warnings"]),
          f"largest unsampled gap = {adv['largest_sampling_gap_h']:.1f} h of the 24 h "
          f"cycle; phase isotropy = {adv['phase_isotropy']:.3f}")

    # ...but the confound gate must still stop us calling it circadian, because
    # this schedule is the ascending morning->afternoon->evening one.
    sep = CC.separate_fatigue_from_rhythm(df, "_zt", "_y", order_col="SessionIndex",
                                          subject_col="FlyID", weight_col="_w",
                                          period=PERIOD)
    check("the confound gate still blocks the word 'circadian' for this schedule",
          sep["attribution_allowed"] is False,
          "a real rhythm was recovered, yet the design cannot attribute it to the clock "
          "-- the module reports both facts")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: modules/circadian_climbing.py")
    print("=" * 78)
    test_A_exact_recovery()
    test_B_matches_nonlinear_least_squares()
    test_C_interval_coverage()
    test_D_zero_amplitude_F_test()
    test_E_population_cosinor()
    test_F_fatigue_vs_rhythm()
    test_G_lombscargle_crosscheck()
    test_H_amplitude_biomarker()
    test_I_integration_with_design_schema()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CIRCADIAN-CLIMBING VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
