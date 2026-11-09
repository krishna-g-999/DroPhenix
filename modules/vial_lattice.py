"""
vial_lattice.py
How many cylinders are in the frame, answered as a model-order question.

WHY THIS EXISTS
---------------
`climbing_video.auto_n_vials` counts blobs. It finds bands of fly activity,
merges fragments whose gap is under 40% of the median gap, keeps those carrying
at least 25% of the strongest band's activity, and reports how many are left.
Every step of that is reasonable and it is still the wrong shape of answer,
because a count of blobs is not a count of cylinders:

  * one cylinder whose flies happen to leave a vertical slice unvisited during
    the sampled frames breaks into two bands, and is reported as two tubes;
  * a reflection, a label, or the operator's sleeve makes a band of its own;
  * the merge threshold is a fraction of the MEDIAN gap, so it works only while
    within-tube cracks and between-tube spaces are well separated in size. On
    this laboratory's own footage they are not: measured on real recordings,
    the estimator returned 2 cylinders for a 1-cylinder paraquat recording
    (`Tap 1 replicate 3.mp4`) and FIVE for the three-cylinder Day 7 rig
    (`C0142.MP4`), where four of the five regions then produced scoring-line
    proposals marked "clear" at rows 872, 581, 558 and 591 -- three real tubes
    at about 580, and a confident, entirely spurious line 290 px away.

An over-count is the worst of the two errors. An under-count drops a tube and
the operator notices a missing column; an over-count INVENTS a tube, fills it
with whatever is in that part of the frame, and reports a climbing index for it
alongside the real ones.

THE RESTATEMENT
---------------
The apparatus is known. Negative-geotaxis rigs hold identical cylinders
standing in a row: same bore, same spacing, vertical, side by side. That is a
one-dimensional LATTICE, and "how many cylinders" is therefore not a
segmentation problem but a MODEL-ORDER problem -- the same question as how many
sinusoids are in a spectrum, or how many components in a mixture.

So we write the generative model the geometry dictates and let the data choose
its order. For a column position x, the activity profile y(x) -- the number of
(pixel, frame) events in which something moved in column x -- is modelled as

    y(x) ~ Poisson( A  if x is inside a tube,  B  otherwise )

where the tubes are n rectangles of COMMON width w on a COMMON pitch d,
starting at offset o:

    inside(x)  <=>  exists i in 0..n-1 with  o + i*d  <=  x  <  o + i*d + w

Four geometric parameters (n, o, d, w) and two rates (A, B). The rigidity is
the whole point: a comb cannot grow a tooth to cover a reflection without also
placing teeth at that spacing everywhere else, and it cannot split one tube in
two without halving the pitch and so inventing teeth in the empty space beside
it. The known geometry does the work that a threshold was doing.

Given a mask, A and B are the inside and outside means, so the profile
likelihood collapses to a closed form (see `_fit_mask`) and the search over
(n, o, d, w) is a cumulative-sum lookup. Nothing is iterative and nothing is
tuned.

TWO CUES, ONE GEOMETRY
----------------------
Fly activity alone is not enough, and the way it fails matters. Measured on
synthetic profiles with known truth, an activity-only lattice reads one wide
tube with a dead vertical slice through it as two narrower tubes -- nothing in
an occupancy profile distinguishes an unvisited slice from the space between
tubes -- and reads a tube whose flies are alive but barely moving, at 8% of its
neighbours' activity, as absent. The second error is the serious one: the vial
it drops is the most impaired one, so the instrument's own sensitivity would
depend on the phenotype being measured.

The glassware does not move. Tube walls appear as strong vertical edges in the
temporal-median background whether the animals are active, sluggish or dead, so
the column-wise horizontal-gradient profile is a second, independent cue that
fails under completely different conditions (glare and clutter, not stillness).
The lattice predicts where the walls must be -- at `o + i*d` and `o + i*d + w`
for each i -- so the same four geometric parameters predict both signals.

The two cues are different KINDS of measurement and are modelled accordingly.
Activity is a count, so it gets the quasi-Poisson treatment above. Wall-edge
energy is a continuous quantity whose numerical scale is set by the frame
height and the bit depth, so it gets the two-group Gaussian likelihood on
log1p -- equivalently a one-way analysis of variance between the predicted wall
bins and the rest:

    logL_structure = -(N/2) * log( RSS(mask) / N )

That form is scale-free, which matters: modelling edge energy as Poisson counts
instead put its evidence on an arbitrary scale, its estimated dispersion came
out near 1e5, and the cue was crushed to about 19 BIC units against an
activity-term preference of 21 for the wrong order -- the cue was right and its
weight was wrong. Both terms now sit on a per-bin scale of N observations, so
each cue is weighted by how much it actually distinguishes the orders rather
than by the magnitude of its units.

    logL_total = logL_activity / phi_activity  +  logL_structure

Because the geometry is SHARED, the parameter count does not grow. That is the
reason to fit jointly rather than run two estimators and write a tie-break
rule: the second cue buys evidence at no cost in model complexity, so BIC can
compare orders on the strength of both at once.

CHOOSING n
----------
Larger n does not buy likelihood for free -- the comb is rigid, so an extra
tooth must land on real activity or it costs. But the profile is strongly
OVER-DISPERSED relative to Poisson: flies cluster, so the variance of y(x)
greatly exceeds its mean, and an uncorrected likelihood ratio would be inflated
by exactly that factor and would chase noise. We therefore use the standard
quasi-likelihood correction (Wedderburn 1974; McCullagh & Nelder 1989, ch. 9):
the dispersion phi is estimated from the Pearson residuals of the richest model
considered, the log-likelihood is scaled by 1/phi, and the order is chosen by

    BIC(n) = -2 * logL(n) / phi_hat  +  k(n) * log(W)

with k = 5 for n >= 2 and k = 4 for n = 1, where the pitch is unidentifiable.
No constant in that expression was chosen to make an answer come out.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
It returns an estimate, a per-order score table, and a `confident` flag, set
only when the chosen order beats the runner-up by the published very-strong-
evidence margin of 10 BIC units (Kass & Raftery 1995). It is NOT a licence to
score: `detect_vials` still refuses when the geometry checks fail, and every
caller must still offer manual regions.

AND IT IS NOT YET THE DEFAULT, because it is not yet calibrated on real
footage. On eight manifest-labelled recordings it agreed 4 of 8, and every
error was on a frame holding ONE cylinder -- empty frame leaves room for
spurious teeth, and a real background is full of vertical edges a comb can land
on that a synthetic one is not: a bench edge, a label, a printed scale, the
rack frame. The 12/12 synthetic recovery was a layer-2 result and that is all
it was. `climbing_video.auto_n_vials` therefore still defaults to the blob
counter, this is opt-in, and the 62-recording calibration in
`validation/study_vial_lattice_accuracy.py` -- which measures BOTH against the
same ground truth -- is what decides which becomes the default.

References
----------
Wedderburn RWM (1974) Biometrika 61:439-447 -- quasi-likelihood.
McCullagh P, Nelder JA (1989) Generalized Linear Models, 2nd ed., ch. 9.
Schwarz G (1978) Ann Stat 6:461-464 -- the BIC penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The coarse grid the search runs on. The profile is summed into this many
# bins, which costs nothing in resolution that matters -- a cylinder is a large
# fraction of the frame width -- and makes the exhaustive search instant.
N_BINS = 192

# A tube narrower than this many bins cannot be resolved at all, and one wider
# than this fraction of the frame cannot be one of several. Both are limits of
# the REPRESENTATION, not assumptions about biology or about any apparatus.
MIN_W_BINS = 3
MAX_W_FRAC = 0.98

# How much better the chosen order must be than the runner-up before the
# estimate is reported as confident, in BIC units.
#
# This was 1.0, chosen as "better than a tie", which is close to meaningless:
# BIC differences are on a deviance scale and one unit is noise. It is now the
# published threshold for VERY STRONG evidence in model selection -- Kass &
# Raftery (1995), table 6 -- which is a convention from the literature rather
# than a number fitted to the recordings that exposed the problem.
#
# Measured on eight manifest-labelled recordings of this laboratory's own
# apparatus: among the calls the estimator called confident under the old cut,
# every CORRECT one scored 33 BIC units or more and every ERROR scored 8 or
# less. The evidence measure separates them; the cut was in the wrong place.
# Under this threshold the same sample gives three confident calls, all three
# correct, and five routed to the manual region editor.
CONFIDENT_DELTA_BIC = 10.0

# Adjacent tubes cannot be closer together than one outside diameter, and the
# activity band inside a tube is at most its BORE. So the pitch is at least the
# band width divided by the bore-to-outside-diameter ratio of the glassware.
# Standard Drosophila climbing vials are borosilicate tubes of about 25 mm
# outside diameter with a wall near 1.2 mm, giving a bore ratio of about 0.90;
# 0.93 is used here so that an unusually thin-walled tube is not excluded. This
# is a dimension of the apparatus, not a tuning constant: it is what stops the
# fit from explaining one tube with a gap in it as two tubes standing in
# contact, which is physically impossible because each has glass on both sides.
BORE_TO_OD = 0.93


@dataclass
class LatticeFit:
    """The chosen lattice and the evidence for it."""

    n_vials: int
    offset_px: int
    pitch_px: float
    width_px: float
    confident: bool
    bic: Dict[int, float]
    delta_bic: float                      # runner-up minus chosen; larger = safer
    dispersion: float
    bands: List[Tuple[int, int]] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        d = {
            "n_vials": self.n_vials,
            "offset_px": self.offset_px,
            "pitch_px": self.pitch_px,
            "width_px": self.width_px,
            "confident": self.confident,
            "bic": dict(self.bic),
            "delta_bic": self.delta_bic,
            "dispersion": self.dispersion,
            "bands": list(self.bands),
            "reason": self.reason,
        }
        return d


def _rebin(profile: np.ndarray, n_bins: int) -> Tuple[np.ndarray, float]:
    """
    Sum the profile into `n_bins` equal bins.

    Summing, not averaging or interpolating: the profile is a COUNT, and the
    sum of Poisson counts is Poisson, so the likelihood below stays exact under
    rebinning. An interpolated profile would not be counts any more.
    """
    y = np.asarray(profile, dtype=np.float64).ravel()
    W = y.size
    if W == 0:
        return np.zeros(0), 1.0
    n_bins = int(min(n_bins, W))
    edges = np.linspace(0, W, n_bins + 1).astype(int)
    out = np.add.reduceat(y, edges[:-1])
    # `reduceat` keeps empty trailing segments; guard against zero-length ones.
    widths = np.diff(edges)
    out = np.where(widths > 0, out, 0.0)
    return out, W / float(n_bins)


def _mass_lookup(y: np.ndarray) -> np.ndarray:
    """Prefix sums, so the mass of any window is one subtraction."""
    return np.concatenate([[0.0], np.cumsum(y)])


def _anova_ll(S_in: np.ndarray, n_in: np.ndarray,
              total: float, total_sq: float, n_tot: int) -> np.ndarray:
    """
    Two-group Gaussian log-likelihood, up to an additive constant, for many
    masks at once.

    With the mask splitting the N bins into two groups whose means are free,
    the maximised log-likelihood is -(N/2) * log(RSS/N), and the within-group
    sum of squares is a prefix-sum lookup:

        RSS = sum(z^2) - S_in^2/n_in - S_out^2/n_out

    Used for the structural cue, which is a continuous signal rather than a
    count; see the module docstring for why that distinction decides the
    weighting of the two cues.
    """
    S_out = total - S_in
    n_out = float(n_tot) - n_in
    rss = total_sq - np.divide(S_in ** 2, n_in, out=np.zeros_like(S_in),
                               where=n_in > 0)
    rss = rss - np.divide(S_out ** 2, n_out, out=np.zeros_like(S_out),
                          where=n_out > 0)
    rss = np.maximum(rss, 1e-12)
    return -0.5 * n_tot * np.log(rss / n_tot)


def _fit_mask(S_in: np.ndarray, n_in: np.ndarray,
              total: float, n_tot: int) -> np.ndarray:
    """
    Profile log-likelihood of the two-rate Poisson model, for many masks at once.

    With lambda = A inside and B outside, the maximum-likelihood rates are the
    inside and outside means, and the log-likelihood reduces to

        S_in  * log(S_in  / n_in )
      + S_out * log(S_out / n_out)
      - total                              (constant in the mask)
      - sum(log y!)                        (constant in the mask)

    so only the first two terms are kept. The limit s*log(s/n) -> 0 as s -> 0 is
    taken explicitly, which is the correct value and not a fudge: a mask region
    with no activity contributes nothing.
    """
    S_out = total - S_in
    n_out = n_tot - n_in

    def term(s, n):
        s = np.asarray(s, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        ok = (s > 0) & (n > 0)
        out = np.zeros(np.broadcast(s, n).shape, dtype=np.float64)
        np.divide(s, n, out=out, where=ok)
        np.log(out, out=out, where=ok)
        return np.where(ok, s * out, 0.0)

    return term(S_in, n_in) + term(S_out, n_out)


def _best_for_order(C: np.ndarray, total: float, n_tot: int, n: int,
                    w_grid: Sequence[int],
                    Cs: Optional[np.ndarray] = None,
                    Cs2: Optional[np.ndarray] = None,
                    phi_a: float = 1.0) -> Tuple[float, int, int, int]:
    """
    The best (offset, pitch, width) for a given number of cylinders.

    Returns (scaled logL, offset_bins, pitch_bins, width_bins). Exhaustive over
    the coarse grid: for each (width, pitch) the mask mass for EVERY offset is a
    vectorised sum of window masses, so the whole search is a handful of array
    additions per (width, pitch) pair.

    When `Cs`/`Cs2` are given they are the prefix sums of log1p(structure) and
    of its square, and the objective becomes the quasi-corrected activity
    log-likelihood plus the two-group Gaussian log-likelihood of the structure.
    The structural mask is the one-bin band at each predicted wall -- 2n of
    them, at the left and right edge of every tube -- so the same four
    parameters predict both signals.
    """
    best = (-np.inf, 0, 0, 0)
    for w in w_grid:
        if n * w > n_tot:
            continue
        if n == 1:
            pitches = [w]                       # unidentifiable; fix it at w
        else:
            # Tubes have glass on both sides, so their activity bands cannot be
            # contiguous: the pitch is at least one outside diameter, which is
            # the band width divided by the bore ratio.
            d_min = max(w + 1, int(np.ceil(w / BORE_TO_OD)))
            d_max = (n_tot - w) // (n - 1)
            if d_max < d_min:
                continue
            pitches = range(d_min, d_max + 1)
        for d in pitches:
            span = (n - 1) * d + w
            n_off = n_tot - span + 1
            if n_off <= 0:
                continue
            offs = np.arange(n_off)
            S_in = np.zeros(n_off, dtype=np.float64)
            for i in range(n):
                a = offs + i * d
                S_in += C[a + w] - C[a]
            n_in = np.full(n_off, float(n * w))
            ll = _fit_mask(S_in, n_in, total, n_tot) / phi_a

            if Cs is not None and Cs2 is not None:
                # The walls: 2n one-bin bands at the tube edges. Clipped to the
                # frame, and distinct by construction, since d > w.
                S_e = np.zeros(n_off, dtype=np.float64)
                for i in range(n):
                    for off in (0, w):
                        a = np.clip(offs + i * d + off, 0, n_tot - 1)
                        S_e += Cs[a + 1] - Cs[a]
                n_e = np.full(n_off, float(2 * n))
                ll = ll + _anova_ll(S_e, n_e, float(Cs[-1]), float(Cs2[-1]),
                                    n_tot)

            j = int(np.argmax(ll))
            if ll[j] > best[0]:
                best = (float(ll[j]), int(offs[j]), int(d), int(w))
    return best


def structure_profile(background: np.ndarray) -> np.ndarray:
    """
    Column-wise wall-edge energy of the static background.

    The tube walls are long vertical features, so their signature is the
    horizontal gradient summed down the frame. Taking the absolute gradient
    makes it blind to whether the glass is darker or lighter than what is behind
    it, which varies with the backdrop and the lighting and is not something to
    depend on.

    This cue is available whether the animals move or not, which is the whole
    reason it is here: see the module docstring.
    """
    g = np.asarray(background, dtype=np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    gx = np.abs(np.diff(g, axis=1))
    prof = gx.sum(axis=0)
    # one column shorter than the frame; pad so the two cues share an x-axis
    return np.concatenate([prof, prof[-1:]]) if prof.size else prof


def fit_lattice(profile: np.ndarray, max_vials: int = 12,
                n_bins: int = N_BINS,
                background: Optional[np.ndarray] = None,
                structure: Optional[np.ndarray] = None) -> LatticeFit:
    """
    Estimate how many identical, equally spaced cylinders explain this profile.

    `profile` is the column-wise activity profile -- a count per column, as
    produced by summing `climbing_video._activity_map` down its rows. Counts,
    not a normalised or smoothed signal: the likelihood below is Poisson and
    rescaling the profile would rescale the evidence.

    `background` (or a precomputed `structure` profile) adds the second,
    phenotype-independent cue described in the module docstring. Supply it
    whenever it is available. Without it the fit still works, but it inherits
    the two failure modes of an activity-only profile, which are documented
    there and measured in `validation/validate_vial_lattice.py`.
    """
    y, px_per_bin = _rebin(profile, n_bins)
    n_tot = int(y.size)
    total = float(y.sum())
    if n_tot < 8 or total <= 0:
        return LatticeFit(0, 0, 0.0, 0.0, False, {}, 0.0, 1.0, [],
                          "no activity in the profile")

    if structure is None and background is not None:
        structure = structure_profile(background)
    ys = Cs = Cs2 = None
    if structure is not None and np.size(structure):
        raw, _ = _rebin(structure, n_bins)
        if raw.size == n_tot and float(raw.sum()) > 0:
            # log1p, because edge energy is positive and multiplicatively
            # dispersed: glare is a factor on the gradient, not an increment.
            ys = np.log1p(np.maximum(raw, 0.0))
            Cs = _mass_lookup(ys)
            Cs2 = _mass_lookup(ys ** 2)
        else:
            ys = None

    C = _mass_lookup(y)
    w_max = max(MIN_W_BINS, int(MAX_W_FRAC * n_tot))
    w_grid = list(range(MIN_W_BINS, w_max + 1))
    orders = [n for n in range(1, int(max_vials) + 1) if n * MIN_W_BINS <= n_tot]

    # --- dispersions, from the richest order ---------------------------------
    # The profile is far more variable than Poisson because flies cluster, so an
    # uncorrected likelihood ratio is inflated by that factor and would chase
    # clustering noise. phi is the Pearson statistic per degree of freedom at
    # the richest model (McCullagh & Nelder 1989, ch. 9) -- the conventional
    # choice, since the richest model is the least biased. It is estimated in an
    # unscaled first pass, then held fixed while the orders are compared, so
    # that every order is judged on the same scale.
    n_rich = max(orders)
    f_rich = _best_for_order(C, total, n_tot, n_rich, w_grid)
    phi_a = 1.0
    if np.isfinite(f_rich[0]):
        _, o_r, d_r, w_r = f_rich
        phi_a = _dispersion(y, _mask_of(n_tot, n_rich, o_r, d_r, w_r), k=5)

    fits: Dict[int, Tuple[float, int, int, int]] = {}
    for n in orders:
        f = _best_for_order(C, total, n_tot, n, w_grid, Cs=Cs, Cs2=Cs2,
                            phi_a=phi_a)
        if np.isfinite(f[0]):
            fits[n] = f
    if not fits:
        return LatticeFit(0, 0, 0.0, 0.0, False, {}, 0.0, 1.0, [],
                          "no lattice could be fitted")

    # --- order selection ----------------------------------------------------
    bic: Dict[int, float] = {}
    for n, (ll, _o, _d, _w) in fits.items():
        k = 4 if n == 1 else 5
        bic[n] = -2.0 * ll + k * np.log(n_tot)

    order = sorted(bic, key=lambda n: bic[n])
    n_hat = order[0]
    delta = (bic[order[1]] - bic[n_hat]) if len(order) > 1 else np.inf
    ll, o, d, w = fits[n_hat]

    bands = [(int(round((o + i * d) * px_per_bin)),
              int(round((o + i * d + w) * px_per_bin)))
             for i in range(n_hat)]

    return LatticeFit(
        n_vials=int(n_hat),
        offset_px=int(round(o * px_per_bin)),
        pitch_px=float(d * px_per_bin),
        width_px=float(w * px_per_bin),
        confident=bool(delta > CONFIDENT_DELTA_BIC),
        bic=bic,
        delta_bic=float(delta),
        dispersion=float(phi_a),
        bands=bands,
        reason=("" if delta > CONFIDENT_DELTA_BIC else
                f"orders {order[0]} and {order[1]} fit this recording almost "
                f"equally well (BIC differ by {delta:.2f}); the number of "
                f"cylinders cannot be read off it. Set the regions manually."),
    )


def _wall_mask(n_tot: int, n: int, o: int, d: int, w: int) -> np.ndarray:
    """The one-bin bands at the 2n predicted tube walls."""
    m = np.zeros(n_tot, dtype=bool)
    for i in range(n):
        for off in (0, w):
            a = int(np.clip(o + i * d + off, 0, n_tot - 1))
            m[a] = True
    return m


def _mask_of(n_tot: int, n: int, o: int, d: int, w: int) -> np.ndarray:
    m = np.zeros(n_tot, dtype=bool)
    for i in range(n):
        a = o + i * d
        m[max(0, a):min(n_tot, a + w)] = True
    return m


def _dispersion(y: np.ndarray, mask: np.ndarray, k: int) -> float:
    """
    Pearson dispersion of the two-rate Poisson fit, floored at 1.

    Floored because under-dispersion relative to Poisson cannot happen in this
    measurement -- the counts are sums over clustered, correlated events -- so
    a phi below 1 is an artefact of a small bin count, and using it would
    SHARPEN the evidence rather than temper it. Erring toward 1 keeps the
    correction conservative in the only direction that matters here.
    """
    lam = np.where(mask, y[mask].mean() if mask.any() else 0.0,
                   y[~mask].mean() if (~mask).any() else 0.0)
    lam = np.maximum(lam, 1e-9)
    df = max(1, y.size - k)
    chi2 = float(np.sum((y - lam) ** 2 / lam))
    return max(1.0, chi2 / df)


def profile_from_activity(activity: np.ndarray) -> np.ndarray:
    """The column-wise activity profile: a count per column."""
    return np.asarray(activity, dtype=np.float64).sum(axis=0)
