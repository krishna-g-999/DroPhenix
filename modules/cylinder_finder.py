"""Locate the cylinders using the operator's own drawn band as the fiducial.

WHY A SECOND WAY OF FINDING THE TUBES
-------------------------------------
`climbing_video.detect_vials` looks for the GLASS. On the rigs in this project that
works some of the time and fails silently the rest: on the Day-3 recordings of the
`Climbing assay_SP` set it returned 0 of 3 cylinders on most clips, because the tubes
stand further from the camera, are thinner, and are set against a bright window.

But every tube in these experiments carries a band drawn round it in marker, and that
band has two properties the glass does not:

  * it is by a wide margin the darkest horizontal thing on the tube;
  * it spans exactly the tube's width and STOPS at the glass.

So the band gives the cylinder's horizontal extent AND the scoring row in a single
measurement. Using the operator's mark as the fiducial is what the mark is for, and it
is the same discriminator `climbing_simple.find_line` already relies on -- applied to
locate the tube, not merely the line inside it.

WHY A CONSISTENCY MODEL RATHER THAN A THRESHOLD
-----------------------------------------------
Dark horizontal runs are not rare: the foam board, the bench edge, the frame border
and a label card all produce them, and on these recordings the raw search returns four
to six candidates where three tubes exist. No threshold separates them reliably,
because some clutter is darker than some bands.

What DOES separate them is that the three tubes are a SET. They stand side by side, so
their bands sit at nearly the same height, their widths are nearly equal, and their
spacing is regular. Clutter satisfies none of these jointly. So every combination of
`n` candidates is scored on how well it fits "n parallel tubes of equal width with a
band ruled at one height", and the best-fitting set wins -- with the fit reported, so a
caller can refuse a poor one instead of accepting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import numpy as np

__all__ = ["BandVial", "find_band_candidates", "locate_cylinders", "LocateResult"]


@dataclass
class BandVial:
    """One cylinder, located by the band drawn round it."""

    x0: int
    x1: int
    band_row: int
    n_rows: int          # how many image rows voted for this cluster
    darkness: float      # how much darker than that row's own background

    @property
    def width(self) -> int:
        return int(self.x1 - self.x0)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x0 + self.x1)


@dataclass
class LocateResult:
    vials: List[BandVial]
    fit: float                  # 0 is a perfect set; smaller is better
    n_candidates: int
    reason: str = ""

    def ok(self, max_fit: float = 0.35) -> bool:
        return bool(self.vials) and self.fit <= max_fit


def find_band_candidates(background: np.ndarray, min_w_frac: float = 0.035,
                         max_w_frac: float = 0.12, min_depth: float = 15.0,
                         min_rows: int = 2, edge_margin_frac: float = 0.01,
                         row_gap_px: int = 15) -> List[BandVial]:
    """Every dark horizontal run that could be a band round a tube.

    Darkness is measured against each row's OWN local background, so a bright window
    on one side of the frame cannot swamp a dark band on the other.
    """
    g = background.astype(np.float32)
    h, w = g.shape[:2]
    k = int(2 * round(0.25 * w) + 1)
    dark = np.clip(cv2.blur(g, (k, 1)) - g, 0, None)

    lo, hi = int(min_w_frac * w), int(max_w_frac * w)
    runs = []
    for y in range(h):
        d = dark[y]
        thr = max(min_depth, float(np.percentile(d, 99.0)) * 0.5)
        on = d > thr
        if not on.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], on.view(np.int8), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            if lo <= (b - a) <= hi:
                runs.append((y, int(a), int(b), float(d[a:b].mean())))
    if not runs:
        return []

    # Cluster in BOTH axes. Clustering by x alone was wrong and quietly so: a tube
    # carries several dark horizontal features at different heights -- the printed
    # label near the top, the graduations, and the drawn band -- and merging them by
    # column produced one cluster whose median row fell between them, pointing at
    # nothing. Measured on Day-3 C0077 that put the "band" at row 123 when the real
    # band is at 490. Splitting each column group by vertical gaps gives one
    # candidate per feature, and the consistency model below then picks the height
    # that is common to all three tubes -- which is what a ruled line is.
    runs.sort(key=lambda r: 0.5 * (r[1] + r[2]))
    col_groups, cur = [], [runs[0]]
    for r in runs[1:]:
        prev_c = 0.5 * (cur[-1][1] + cur[-1][2])
        if abs(0.5 * (r[1] + r[2]) - prev_c) <= 0.5 * (r[2] - r[1]):
            cur.append(r)
        else:
            col_groups.append(cur)
            cur = [r]
    col_groups.append(cur)

    clusters = []
    for grp in col_groups:
        grp.sort(key=lambda r: r[0])
        sub = [grp[0]]
        for r in grp[1:]:
            if r[0] - sub[-1][0] <= row_gap_px:
                sub.append(r)
            else:
                clusters.append(sub)
                sub = [r]
        clusters.append(sub)

    out = [BandVial(x0=int(np.median([r[1] for r in c])),
                    x1=int(np.median([r[2] for r in c])),
                    band_row=int(np.median([r[0] for r in c])),
                    n_rows=len(c),
                    darkness=float(np.median([r[3] for r in c])))
           for c in clusters if len(c) >= min_rows]
    # A tube stands clear of the frame. Anything touching the border is the foam
    # board, the bench edge or the frame itself -- measured on the Day-3 recordings,
    # those edge runs were darker than some real bands, so they outranked the tubes
    # and were selected in 13 of 21 clips. They are excluded by position, which is a
    # property of the rig rather than a threshold that has to be tuned.
    m = max(int(edge_margin_frac * w), 2)
    out = [v for v in out if v.x0 > m and v.x1 < (w - m)]
    out.sort(key=lambda v: v.x0)
    return out


def _fit_cost(sel: List[BandVial], frame_w: int) -> float:
    """How well a set of candidates fits 'n parallel tubes with one ruled height'.

    Three dimensionless penalties, each scaled by the tubes' own width so the cost
    means the same thing on any rig:

      width disagreement   -- tubes side by side are the same width
      band height spread   -- one line ruled across all of them
      spacing irregularity -- they stand in a row, evenly placed
    """
    wds = np.array([v.width for v in sel], float)
    rows = np.array([v.band_row for v in sel], float)
    cxs = np.array(sorted(v.cx for v in sel), float)
    mw = float(wds.mean())
    if mw <= 0:
        return float("inf")
    cost = 2.0 * float(wds.std() / mw) + float(rows.std() / mw)
    if len(cxs) > 2:
        gaps = np.diff(cxs)
        cost += float(gaps.std() / max(gaps.mean(), 1e-6))
    # a band should be genuinely dark; weak candidates are usually clutter
    cost += float(np.clip((40.0 - min(v.darkness for v in sel)) / 80.0, 0, 1))
    return cost


def _band_darkness_at(background: np.ndarray, cx: float, width: float,
                      row: int, half_rows: int = 6) -> float:
    """How dark a band would be if a tube stood centred at `cx` with this width.

    Measured the same way the candidate finder measures it -- against the row's own
    local background -- so the number is comparable with a real candidate's.
    """
    g = background.astype(np.float32)
    h, w = g.shape[:2]
    x0, x1 = int(round(cx - width / 2)), int(round(cx + width / 2))
    if x0 < 0 or x1 > w or x1 - x0 < 4:
        return 0.0
    r0, r1 = max(0, row - half_rows), min(h, row + half_rows + 1)
    if r1 <= r0:
        return 0.0
    k = int(2 * round(0.25 * w) + 1)
    dark = np.clip(cv2.blur(g, (k, 1)) - g, 0, None)
    return float(dark[r0:r1, x0:x1].mean(axis=1).max())


def _complete_row(background: np.ndarray, pool: List[BandVial],
                  max_fit: float) -> Optional[Tuple[List[BandVial], float]]:
    """Find two consistent tubes and predict the third's position from them.

    Returns `(vials, cost)` where the predicted tube carries `n_rows = 0` to mark
    that it was inferred rather than detected, or None if no pair supports a
    prediction the image agrees with.
    """
    h, w = background.shape[:2]
    best = None
    for a, b in combinations(pool, 2):
        if a.cx == b.cx:
            continue
        wa, wb = a.width, b.width
        mw = 0.5 * (wa + wb)
        if mw <= 0:
            continue
        # the pair itself must look like two tubes of one rack: same width, one
        # ruled height, and standing near each other rather than across the bench
        if abs(wa - wb) / mw > 0.25:
            continue
        if abs(a.band_row - b.band_row) / mw > 0.6:
            continue
        lo, hi = (a, b) if a.cx < b.cx else (b, a)
        sep = hi.cx - lo.cx
        row = int(round(0.5 * (a.band_row + b.band_row)))
        # the pair may be adjacent (predict outside) or the outer two (predict the
        # middle); both are tried and the image decides
        for label, cx_new, spacing in (("left", lo.cx - sep, sep),
                                       ("right", hi.cx + sep, sep),
                                       ("middle", 0.5 * (lo.cx + hi.cx), sep / 2)):
            if not (1.0 <= spacing / mw <= 5.0):
                continue
            if not (mw / 2 <= cx_new <= w - mw / 2):
                continue
            dark = _band_darkness_at(background, cx_new, mw, row)
            # the prediction must be supported by real darkness there, judged
            # against the pair's own bands rather than an absolute threshold
            ref = min(a.darkness, b.darkness)
            if dark < 0.45 * ref:
                continue
            pred = BandVial(x0=int(round(cx_new - mw / 2)),
                            x1=int(round(cx_new + mw / 2)),
                            band_row=row, n_rows=0, darkness=dark)
            trio = sorted([a, b, pred], key=lambda v: v.x0)
            cost = _fit_cost(trio, w)
            if cost <= max_fit and (best is None or cost < best[1]):
                best = (trio, cost)
    return best


def locate_cylinders(background: np.ndarray, n_expected: int = 3,
                     candidates: Optional[List[BandVial]] = None,
                     max_fit: float = 0.35) -> LocateResult:
    """Pick the `n_expected` candidates that best form a row of identical tubes."""
    cands = (find_band_candidates(background) if candidates is None else candidates)
    h, w = background.shape[:2]
    if len(cands) < n_expected:
        return LocateResult([], float("inf"), len(cands),
                            f"only {len(cands)} band candidate(s) found, "
                            f"{n_expected} needed")
    # keep the search bounded on cluttered frames, strongest evidence first
    # more candidates now that each dark feature is its own cluster, so a wider pool
    pool = sorted(cands, key=lambda v: (-v.darkness, -v.n_rows))[:16]
    best, best_cost = None, float("inf")
    for sel in combinations(pool, n_expected):
        c = _fit_cost(list(sel), w)
        if c < best_cost:
            best, best_cost = list(sel), c
    if best is None:
        return LocateResult([], float("inf"), len(cands), "no combination scored")

    # COMPLETE THE ROW when a tube's band was never detected at all.
    #
    # The four recordings this project refused were not cases of the locator
    # choosing badly among good options -- the right answer was ABSENT from the
    # candidate list. On Day-3 C0078 no candidate existed anywhere near the middle
    # tube; on Day-7 C0145 and C0153 the only candidates on those tubes were the
    # PRINTED LABEL rows near the top, because the drawn band on that rig is faint.
    # The locator was picking the best triple from a list that did not contain the
    # tubes, and the fit score correctly said so.
    #
    # But three cylinders stand in a row, evenly spaced, with one line ruled across
    # them. That means two confidently-found tubes PREDICT the third: same row, same
    # width, one spacing away. So rather than demand three independent detections,
    # find the best-fitting PAIR, extrapolate the missing position, and verify by
    # measuring the darkness there. A prediction that the image does not support is
    # rejected; one it does support is used, and flagged as predicted.
    # ROW COMPLETION WAS TRIED HERE AND IS DISABLED. Predicting a missing tube from
    # the other two is sound geometry and it does place C0078's middle tube
    # correctly. But the fit cost cannot tell WHICH dark feature on a tube it has
    # found, so completing a row from two label-height candidates makes a recording
    # PASS with the scoring line on the printed specification label instead of the
    # drawn band. Measured: Day-7 C0145 was then scored with rows 306/297/302 where
    # the band is near 580, and that day's readout fell from 70% of flies above the
    # line to 34% -- a confident wrong answer where there had been an honest refusal.
    # Re-enable only together with a check that the completed row IS the drawn band.
    if False and n_expected == 3 and best_cost > max_fit:
        filled = _complete_row(background, pool, max_fit)
        if filled is not None and filled[1] < best_cost:
            best, best_cost = filled[0], filled[1]

    best.sort(key=lambda v: v.x0)
    reason = ("" if best_cost <= max_fit else
              f"the best set of {n_expected} does not look like a row of identical "
              f"tubes (fit {best_cost:.2f} against a limit of {max_fit:.2f}); "
              f"widths {[v.width for v in best]}, band rows "
              f"{[v.band_row for v in best]}")
    return LocateResult(best, float(best_cost), len(cands), reason)
