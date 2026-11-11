"""
Counting flies by LINE CROSSINGS over time, instead of by blobs in one frame.

THE PROBLEM THIS EXISTS TO SOLVE (register N97 / N103)
-------------------------------------------------------
`fly_blob` finds flies as local maxima of a band-pass response with a minimum
separation of `sep_frac = 1.2` fly diameters. Two flies closer together than
that produce ONE maximum, so a crowded tube is under-counted BY CONSTRUCTION.
Measured on 672 hand-counted cells: 73% of the hand counts' dynamic range,
r(error, hand) = -0.57, counts saturating near 12-13 flies whatever the truth.

Three corrections were tried and all three were refuted, because each tried to
undo the merge AFTER it happened: the geometric single-fly area (gave the
ceiling straight back), raw band-pass mass (over-counts sparse cells fivefold
by bypassing the minmass and eccentricity filters), and accepted-peak mass
(bought dynamic range but INVERTED an arm ordering, which is the one failure a
comparative instrument cannot have).

THE RE-FORMULATION
------------------
All three failed in the same place: they are spatial. They ask "how many
objects are in this region of this frame", which is exactly the question that
overlapping bodies make unanswerable.

A crossing counter asks a different question: "how many times did something
pass this line". That question is TEMPORAL, and it is the question an
instrument can actually answer, because two flies that are inseparable in any
single frame still usually cross the line at different MOMENTS. The ceiling is
a property of simultaneous spatial occupancy; it is not a property of event
counting over time.

This is not a novel invention -- it is how the laboratory's own DAM monitors
work. A DAM channel has one infrared beam across the middle of the tube and
counts interruptions; it never tries to resolve how many flies are in the tube,
and it is trusted for exactly that reason. It is also how induction loops at
traffic lights, turnstiles and optical particle counters work: put a gate
somewhere, count what goes through it, and refuse to care what the crowd looks
like on either side.

WHAT THIS BUYS, AND WHAT IT COSTS
----------------------------------
Buys: immunity to spatial crowding. Ten flies stacked against the tube wall
above the line are TEN counted events if they crossed at ten different moments,
where `fly_blob` sees one or two maxima.

Costs, stated plainly because they are real:

  1. It needs DENSE frames. A crossing seen at 10 fps is a crossing; at one
     frame per ten seconds there are no crossings to see, only positions. This
     is strictly more decoding than a spatial counter.
  2. Flies that cross SIMULTANEOUSLY, in the same place, still merge. The
     method reduces the problem from "resolve bodies in a crowd" to "resolve
     bodies at one instant at one line", which is far easier but not free.
  3. It measures a different quantity. The assay's hand count is "how many are
     above the line at time T"; this is "how many net crossings happened by
     time T". They agree only if every fly starts below the line, which is what
     the tap enforces, and if nothing crosses unobserved between frames.
  4. It needs the number of flies loaded. That is a feature rather than a
     burden: `n_flies` is already recorded in this laboratory's manifests, and
     knowing it lets the counter REFUSE an impossible answer instead of
     reporting it.

THE SIGN CONVENTION, STATED ONCE
---------------------------------
Image rows increase DOWNWARD. A fly climbing moves to SMALLER y. So an upward
(climbing) crossing is y going from above `line_y` to below it in value, and
`net = up - down` is the number of flies currently above the line.

WHAT IS NOT ESTABLISHED HERE
-----------------------------
This module is the algorithm. Whether it beats `fly_blob` on this laboratory's
footage is a MEASUREMENT, made by `validation/validate_fly_tripwire.py` against
the same 672 hand-counted cells and the same published bar: it must beat
MAE 3.589 and RMSE 4.436, recover more than 69% of the between-arm spread, and
preserve the ordering of all six arms. Until that measurement exists, nothing
here is evidence about anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TripwireParams:
    """Geometry and association limits, all in units of the fly diameter `d`."""

    #: Half-height of the band around the line in which flies are tracked, in
    #: fly diameters. Tracking is deliberately LOCAL: association only has to
    #: work near the gate, where flies are sparse and moving, rather than
    #: everywhere, where they are crowded and still. Too small and a fast fly
    #: steps over the band between frames; too large and the crowding the
    #: method exists to avoid comes back.
    band_frac: float = 3.0

    #: Largest per-frame displacement that may still be the same fly, in fly
    #: diameters. A link longer than this is treated as two different animals.
    gate_frac: float = 2.0

    #: A crossing is only counted once the fly is this far past the line, in
    #: fly diameters. Without it a fly hovering exactly on the line produces a
    #: burst of up/down pairs; `net` cancels them, but `up` and `down`
    #: separately become meaningless and the diagnostics mislead.
    hysteresis_frac: float = 0.35

    #: Frames per second at which the video is sampled. The association gate
    #: above is per FRAME, so this and `gate_frac` together state the fastest
    #: fly the counter can follow.
    fps_sample: float = 10.0


@dataclass
class TripwireResult:
    """Counts at one timepoint, with the diagnostics that justify trusting them."""

    n_above: float
    up: int
    down: int
    net: int
    n_loaded: Optional[int]
    usable: bool
    reason: str = ""
    n_frames: int = 0
    n_links: int = 0
    series: List[Tuple[float, int]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_above": self.n_above, "up": self.up, "down": self.down,
            "net": self.net, "n_loaded": self.n_loaded, "usable": self.usable,
            "reason": self.reason, "n_frames": self.n_frames,
            "n_links": self.n_links,
        }


def _greedy_link(prev: np.ndarray, cur: np.ndarray, gate: float
                 ) -> List[Tuple[int, int]]:
    """
    Nearest-neighbour association between two small point sets, within `gate`.

    Greedy on the sorted pair distances. The optimal assignment (Hungarian)
    would be better in a crowd, but this runs only on points INSIDE the band,
    where there are a handful of fast-moving animals rather than a cluster, and
    a greedy pass with a tight gate makes the same choices there. Using the
    cheap method where it is sufficient is deliberate; where it is not
    sufficient -- a dense crowd straddling the line -- no association method
    recovers the truth and the counter should say so instead.
    """
    if prev.size == 0 or cur.size == 0:
        return []
    d = np.hypot(prev[:, None, 0] - cur[None, :, 0],
                 prev[:, None, 1] - cur[None, :, 1])
    pairs: List[Tuple[int, int]] = []
    used_p, used_c = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] > gate:
            break
        if i in used_p or j in used_c:
            continue
        used_p.add(int(i)); used_c.add(int(j))
        pairs.append((int(i), int(j)))
    return pairs


def count_crossings(points_per_frame: Sequence[np.ndarray],
                    line_y: float,
                    d: float,
                    params: Optional[TripwireParams] = None,
                    n_loaded: Optional[int] = None,
                    times_s: Optional[Sequence[float]] = None,
                    seed_above: int = 0,
                    ) -> TripwireResult:
    """
    Count net upward crossings of `line_y` across a sequence of detections.

    `points_per_frame[i]` is an (N, 2) array of (x, y) fly centroids in frame i,
    in the SAME pixel frame as `line_y`. `d` is the fly diameter in pixels.

    Returns a `TripwireResult` whose `n_above` is the estimated number of flies
    above the line at the last frame. When `n_loaded` is supplied, a net count
    outside [0, n_loaded] marks the result UNUSABLE rather than clamping it:
    an impossible count is evidence that the association failed, and reporting
    a clamped value would hide exactly the failure the caller needs to see.

    `seed_above` IS NOT OPTIONAL IN PRACTICE, AND HERE IS WHY
    ---------------------------------------------------------
    Counting crossings gives the CHANGE in the number above the line. It equals
    the absolute number only if every fly began below the line, which is what
    the tap is supposed to guarantee. Measured on this laboratory's own
    footage, it does not: at the detected tap of a WT untreated recording,
    8 of the ~20 flies were ALREADY above the line and stayed there, and a scan
    of the whole 127-second recording at 2 Hz found no moment at which the
    number above the line approached zero. Those 8 animals never cross, so no
    crossing counter can ever see them.

    So the absolute count is

        n_above(T) = n_above(t_seed) + net_crossings(t_seed -> T)

    and `seed_above` is `n_above(t_seed)`, supplied by a spatial count at the
    seed frame. This is a genuine hybrid and it is not a defeat: the spatial
    counter is asked for the number above the line at the ONE moment when that
    region is emptiest, which is where its crowding ceiling bites least, and
    the crossing counter carries the number forward through exactly the crowded
    period where the spatial counter fails.
    """
    p = params or TripwireParams()
    band = p.band_frac * d
    gate = p.gate_frac * d
    hyst = p.hysteresis_frac * d

    up = down = 0
    n_links = 0
    series: List[Tuple[float, int]] = []

    #: side of the line for each tracked point in the previous frame:
    #: -1 above (smaller y), +1 below, 0 inside the hysteresis deadband.
    def side(y: float) -> int:
        if y < line_y - hyst:
            return -1
        if y > line_y + hyst:
            return +1
        return 0

    prev_pts = np.empty((0, 2), dtype=float)
    prev_side: List[int] = []

    for fi, pts in enumerate(points_per_frame):
        cur = np.asarray(pts, dtype=float).reshape(-1, 2) if pts is not None \
            else np.empty((0, 2))
        if cur.size:
            cur = cur[np.abs(cur[:, 1] - line_y) <= band]

        cur_side = [side(float(y)) for y in cur[:, 1]] if cur.size else []

        for i, j in _greedy_link(prev_pts, cur, gate):
            n_links += 1
            s_prev, s_cur = prev_side[i], cur_side[j]
            # A crossing is a change of SIDE with the deadband passed. A point
            # that enters the deadband and comes back out the way it went in
            # contributes nothing, which is the whole point of the deadband.
            if s_prev == +1 and s_cur == -1:
                up += 1
            elif s_prev == -1 and s_cur == +1:
                down += 1

        # Carry the last decided side through the deadband, so a fly that
        # pauses on the line is still compared against where it came FROM.
        carried = []
        for k, s in enumerate(cur_side):
            if s != 0:
                carried.append(s)
            else:
                prior = 0
                for i, j in _greedy_link(prev_pts, cur, gate):
                    if j == k:
                        prior = prev_side[i]
                        break
                carried.append(prior)
        prev_pts, prev_side = cur, carried

        t = float(times_s[fi]) if (times_s is not None and fi < len(times_s)) \
            else float(fi) / max(p.fps_sample, 1e-9)
        series.append((t, up - down + int(seed_above)))

    net = up - down + int(seed_above)
    usable, reason = True, ""
    if n_loaded is not None:
        if net < 0:
            usable, reason = False, (
                f"net crossings {net} is negative: more flies left the region "
                f"above the line than ever entered it, so the association "
                f"failed or the tap did not clear the line")
        elif net > n_loaded:
            usable, reason = False, (
                f"net crossings {net} exceeds the {n_loaded} flies loaded, so "
                f"at least one animal was counted twice; the gate or the "
                f"sampling rate does not match this footage")
    if not points_per_frame:
        usable, reason = False, "no frames were supplied"

    return TripwireResult(
        n_above=float(net), up=up, down=down, net=net, n_loaded=n_loaded,
        usable=usable, reason=reason, n_frames=len(points_per_frame),
        n_links=n_links, series=series)


def counts_at_times(result: TripwireResult,
                    query_times_s: Sequence[float]) -> Dict[float, int]:
    """
    Read the running net count off a finished run at the assay's timepoints.

    One decode of a recording yields the whole crossing series, so every
    timepoint the assay scores is a lookup rather than another pass over the
    video. The value at `t` is the net count at the last sampled frame at or
    before `t` -- a step function, which is what a count of discrete events is.
    """
    out: Dict[float, int] = {}
    if not result.series:
        return {float(t): 0 for t in query_times_s}
    ts = np.array([s[0] for s in result.series], dtype=float)
    vs = np.array([s[1] for s in result.series], dtype=int)
    for q in query_times_s:
        idx = np.searchsorted(ts, float(q), side="right") - 1
        out[float(q)] = int(vs[max(idx, 0)])
    return out
