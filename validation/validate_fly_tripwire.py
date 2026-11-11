"""
Validation suite for the line-crossing ("tripwire") counter, modules/fly_tripwire.py.

WHY THIS SUITE EXISTS FOR A REFUTED CANDIDATE
----------------------------------------------
`fly_tripwire` is NOT the live counting path and did not clear the bar on this
laboratory's footage (register N120). It is kept, and tested, because the
register's claim about it is precise and needs to stay true: the ALGORITHM is
correct and the FOOTAGE defeats it. That is a different statement from "it does
not work", it is the statement that justifies trying the method again on
recordings with a clean post-tap state, and it is only defensible while the
algorithm's correctness is still checked.

So this suite establishes layer 2 (the maths does what it says) and nothing
else. It uses synthetic motion where the truth is known by construction. It
says NOTHING about accuracy on real video; that measurement lives in
`scripts/tripwire_detector_study.py` and its result is a refusal.

WHAT IS CHECKED
---------------
A. No ceiling. The defect that motivated the whole exercise is that peak
   counting saturates near 12-13 flies. A crossing counter must not, so 1, 5,
   10 and 20 flies crossing must return exactly 1, 5, 10 and 20.
B. Direction. A fly that climbs and falls back must contribute ZERO net, with
   one up and one down recorded separately -- the operator's own requirement
   ("if any comes down we detect").
C. The deadband. A fly hovering ON the line must not emit a burst of
   up/down pairs; `net` would cancel them but `up` and `down` would become
   meaningless diagnostics.
D. Refusal. A count outside [0, n_loaded] must mark the result UNUSABLE rather
   than silently clamping, because an impossible count is evidence that the
   association failed.
E. Seeding. `n_above = seed + net`, because on real footage the tap does not
   clear the line and crossings alone give the change rather than the absolute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.fly_tripwire import (  # noqa: E402
    TripwireParams, count_crossings, counts_at_times)

FAILURES: list[str] = []
D = 10.0
LINE = 100.0


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def _climbers(n, n_frames=60, speed=3.0, start_y=160.0, seed=1, stagger=True):
    rng = np.random.default_rng(seed)
    xs = np.linspace(20, 160, n)
    t0 = rng.integers(0, 20, n) if stagger else np.zeros(n, int)
    out = []
    for f in range(n_frames):
        out.append(np.array([[xs[i], start_y - speed * max(f - t0[i], 0)]
                             for i in range(n)], dtype=float))
    return out


def test_A_no_ceiling() -> None:
    print("\nA. no counting ceiling (the defect that motivated the method)")
    for n in (1, 5, 10, 20):
        r = count_crossings(_climbers(n), LINE, D, TripwireParams(), n_loaded=25)
        check(f"{n:2d} flies crossing once are counted exactly",
              r.net == n, f"net {r.net}, expected {n}")
    r = count_crossings(_climbers(20, stagger=False), LINE, D,
                        TripwireParams(), n_loaded=25)
    check("20 SIMULTANEOUS crossings, separated in x, are all counted",
          r.net == 20, f"net {r.net}, expected 20")


def test_B_direction() -> None:
    print("\nB. a fly that comes back down is detected and cancels")
    ys = list(np.arange(160, 40, -3)) + list(np.arange(40, 160, 3))
    r = count_crossings([np.array([[50.0, y]]) for y in ys], LINE, D,
                        TripwireParams(), n_loaded=20)
    check("climb then fall gives up=1, down=1, net=0",
          (r.up, r.down, r.net) == (1, 1, 0),
          f"up {r.up}, down {r.down}, net {r.net}")

    frames = []
    for f in range(60):
        pts = [(30 + 25 * i, 160 - 3 * f) for i in range(3)]
        pts.append((120, 40 + 3 * f if f > 20 else 160 - 3 * f))
        frames.append(np.array(pts, dtype=float))
    r = count_crossings(frames, LINE, D, TripwireParams(), n_loaded=20)
    check("3 climb and 1 climbs-then-falls gives net 3",
          r.net == 3, f"net {r.net}, up {r.up}, down {r.down}")


def test_C_deadband() -> None:
    print("\nC. a fly hovering on the line does not emit spurious pairs")
    ys = [LINE + 1.5 * np.sin(k) for k in range(80)]
    r = count_crossings([np.array([[50.0, y]]) for y in ys], LINE, D,
                        TripwireParams(), n_loaded=20)
    check("hovering gives up=0, down=0, net=0",
          (r.up, r.down, r.net) == (0, 0, 0),
          f"up {r.up}, down {r.down}, net {r.net}")


def test_D_refusal() -> None:
    print("\nD. an impossible count is refused, not clamped")
    frames = [np.array([[50.0, 160.0 - 6.0 * (f % 40)]]) for f in range(120)]
    r = count_crossings(frames, LINE, D, TripwireParams(), n_loaded=2)
    check("net above the flies loaded marks the result UNUSABLE",
          (not r.usable) and r.net > 2,
          f"net {r.net}, n_loaded 2, usable {r.usable}")
    check("the refusal says why", "exceeds" in r.reason, r.reason[:70])
    r2 = count_crossings([], LINE, D, TripwireParams(), n_loaded=20)
    check("no frames is refused", not r2.usable, r2.reason[:60])


def test_E_seed_and_series() -> None:
    print("\nE. seeding, because the tap does not clear the line on real footage")
    frames = [np.array([[50.0, y]]) for y in np.arange(160, 40, -3)]
    r = count_crossings(frames, LINE, D, TripwireParams(), n_loaded=20,
                        seed_above=7)
    check("n_above = seed + net", r.n_above == 8.0,
          f"seed 7 + 1 crossing gave {r.n_above}, expected 8")
    r0 = count_crossings(frames, LINE, D, TripwireParams(), n_loaded=20)
    check("seed defaults to 0 and leaves the count unchanged",
          r0.n_above == 1.0, f"got {r0.n_above}, expected 1")

    ts = [i * 0.1 for i in range(len(frames))]
    r3 = count_crossings(frames, LINE, D, TripwireParams(fps_sample=10.0),
                         n_loaded=20, times_s=ts, seed_above=2)
    at = counts_at_times(r3, [0.0, ts[-1]])
    check("the running series is readable at arbitrary timepoints",
          at[0.0] == 2 and at[ts[-1]] == 3,
          f"t=0 -> {at[0.0]} (expected 2), t=end -> {at[ts[-1]]} (expected 3)")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: the line-crossing (tripwire) counter")
    print("=" * 78)
    print("Layer 2 ONLY. This establishes that the algorithm is correct on")
    print("synthetic motion. It establishes NOTHING about accuracy on real")
    print("video, where the method was REFUTED -- see register N120.")
    test_A_no_ceiling()
    test_B_direction()
    test_C_deadband()
    test_D_refusal()
    test_E_seed_and_series()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL TRIPWIRE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
