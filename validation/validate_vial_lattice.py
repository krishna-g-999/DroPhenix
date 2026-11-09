"""
Validation suite for the lattice cylinder-count estimator (modules/vial_lattice.py).

WHY THIS SUITE EXISTS
---------------------
The number of cylinders decides how the frame is carved up, and a wrong answer
does not fail loudly. An under-count drops a tube, which an operator notices
because a column is missing. An OVER-count invents a tube, fills it with
whatever happens to be in that part of the frame, and reports a climbing index
for it alongside the real ones -- a confidently wrong number, which this
project holds to be worse than no number.

The estimator being replaced counted blobs of fly activity. Measured on this
laboratory's own recordings, it returned 2 cylinders for a one-cylinder
paraquat recording and 5 for the three-cylinder Day 7 rig, where four of the
five invented regions then produced scoring-line proposals marked "clear".

WHAT IS VALIDATED HERE, AND AT WHICH LAYER
------------------------------------------
A. THE LIKELIHOOD IS THE LIKELIHOOD IT CLAIMS TO BE (layer 2, maths).
   The closed forms in `_fit_mask` and `_anova_ll` are checked against direct,
   naive computation of the same quantities -- a Poisson log-likelihood summed
   term by term, and a two-group residual sum of squares computed by brute
   force. A closed form that is wrong is the one error this module could make
   that would be invisible everywhere else.

B. THE ORDER IS RECOVERED ON KNOWN GEOMETRY (layer 2).
   Profiles are generated from a lattice that is known by construction, with
   negative-binomial over-dispersion standing in for fly clustering, and the
   estimate is compared with the truth.

C. THE TWO FAILURE MODES OF AN ACTIVITY-ONLY FIT (layer 3, inference).
   Two cases defeat any estimator that reads fly motion alone: a single tube
   with a dead vertical slice through it, which looks like two tubes, and a
   tube whose flies are alive but barely moving, which looks absent. The second
   is recorded here as the serious one, because the vial it drops is the most
   impaired one -- so the instrument's sensitivity would depend on the
   phenotype it is measuring. Both are asserted to be fixed by the structural
   cue, and the activity-only failure is asserted to still happen, so that the
   suite fails if the cue is ever quietly removed.

D. THE PHYSICAL CONSTRAINT IS ENFORCED (layer 2/4).
   Tubes have glass on both sides, so adjacent activity bands cannot touch. The
   fitted pitch must exceed the fitted width by the bore ratio of real
   glassware. Without this the fit can explain one tube with a gap in it as two
   tubes in contact, which cannot exist.

E. IT AGREES WITH THE EXPERIMENT'S OWN RECORD (layer 4, measurement).
   Where this machine holds the laboratory's recordings, the estimate is
   compared against `cylinders_in_frame` in the paraquat manifest -- ground
   truth written when the videos were taken, independent of any software. A
   bounded sample is checked here; the full 62-recording calibration is
   `validation/study_vial_lattice_accuracy.py`, and the accuracy figure comes
   from there, not from this suite.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
That the regions are right. A correct COUNT with wrong boundaries still scores
the wrong pixels, and the boundaries are judged by the confidence components in
`detect_vials` and by the operator-confirmable editor, not here. Nor does it
establish anything about rigs with other geometries -- a ring rack, or tubes at
an angle -- which this lattice does not model and which it should be expected
to refuse rather than mis-fit.

Run: ./.venv/Scripts/python.exe validation/validate_vial_lattice.py
Exit 0 = all pass.
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import vial_lattice as VL  # noqa: E402

try:
    from validation import lab_data as _LAB
except Exception:                                      # direct-script use
    import lab_data as _LAB

FAILURES: List[str] = []
W, H = 1920, 1080


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name + (f" ({detail})" if detail else ""))


def synth(n: int, width: int, pitch: int, offset: int, rate: float = 60.0,
          bg: float = 1.0, overdisp: float = 6.0, frag=None, clutter=None,
          sick=None, glare=None, wall_contrast: float = 55.0, seed: int = 7):
    """
    An activity profile and a background frame from one shared geometry.

    The two cues are corrupted INDEPENDENTLY -- activity by fly clustering and
    by the phenotype, structure by glare -- which is the situation the joint
    fit exists for. Over-dispersion is negative-binomial, i.e. Poisson with a
    gamma-distributed rate, which is the standard model for clustered counts
    and gives variance = mu*(1 + mu/r).
    """
    rng = np.random.default_rng(seed)
    lam = np.full(W, bg)
    rates = [rate] * n
    if sick is not None:
        i, f = sick
        rates[i] = rate * f
    for i in range(n):
        a = offset + i * pitch
        lam[a:a + width] = rates[i]
    if frag is not None:
        lam[frag[0]:frag[1]] = bg
    if clutter is not None:
        lam[clutter[0]:clutter[1]] = clutter[2]
    r = max(lam.mean() / max(overdisp - 1.0, 1e-6), 1e-6)
    act = rng.poisson(lam * rng.gamma(r, 1.0 / r, size=W)).astype(float)

    frame = np.full((H, W), 150.0)
    for i in range(n):
        a = offset + i * pitch
        frame[:, a:a + width] = 185.0
        for e in (a, a + width):
            frame[:, max(0, e - 3):e + 3] = 185.0 - wall_contrast
    if glare is not None:
        frame[:, glare[0]:glare[1]] = 245.0
    frame += rng.normal(0, 6.0, size=(H, W))
    return act, np.clip(frame, 0.0, 255.0)


# --------------------------------------------------------------------------
def test_A_closed_forms_are_right() -> None:
    print("\nA. The closed forms equal the quantities they claim to compute")

    rng = np.random.default_rng(3)
    y = rng.poisson(12.0, size=64).astype(float)
    total, n_tot = float(y.sum()), y.size

    for trial in range(5):
        mask = rng.random(n_tot) < 0.35
        if not mask.any() or mask.all():
            continue
        S_in = float(y[mask].sum())
        n_in = float(mask.sum())

        # --- Poisson, by hand, term by term ---------------------------------
        A = y[mask].mean()
        B = y[~mask].mean()
        lam = np.where(mask, A, B)
        naive = float(np.sum(y * np.log(np.maximum(lam, 1e-300)) - lam))
        # the closed form drops the constant -total; add it back to compare
        closed = float(VL._fit_mask(np.array([S_in]), np.array([n_in]),
                                    total, n_tot)[0]) - total
        check(f"Poisson profile log-likelihood matches naive sum (trial {trial})",
              abs(naive - closed) < 1e-6 * max(1.0, abs(naive)),
              f"naive {naive:.6f} vs closed {closed:.6f}")

        # --- two-group Gaussian, by hand ------------------------------------
        z = np.log1p(y)
        Sz_in = float(z[mask].sum())
        rss_naive = float(np.sum((z[mask] - z[mask].mean()) ** 2)
                          + np.sum((z[~mask] - z[~mask].mean()) ** 2))
        ll_naive = -0.5 * n_tot * np.log(rss_naive / n_tot)
        ll_closed = float(VL._anova_ll(np.array([Sz_in]), np.array([n_in]),
                                       float(z.sum()), float((z ** 2).sum()),
                                       n_tot)[0])
        check(f"two-group Gaussian log-likelihood matches naive RSS (trial {trial})",
              abs(ll_naive - ll_closed) < 1e-9 * max(1.0, abs(ll_naive)),
              f"naive {ll_naive:.9f} vs closed {ll_closed:.9f}")

    # A mask region with no activity must contribute nothing, not -inf.
    v = VL._fit_mask(np.array([0.0]), np.array([10.0]), 100.0, 64)
    check("an empty mask region contributes zero, not a divergence",
          np.isfinite(v[0]), f"{v[0]}")

    # Scale-freedom of the structural term: the gradient profile's units are
    # set by the frame height and bit depth, so the term must not care.
    z = np.log1p(rng.poisson(30.0, size=64).astype(float))
    mask = np.zeros(64, bool)
    mask[[3, 9, 20, 41]] = True
    a = VL._anova_ll(np.array([float(z[mask].sum())]), np.array([4.0]),
                     float(z.sum()), float((z ** 2).sum()), 64)[0]
    z2 = z + 5.0                         # log-scale shift = multiplicative factor
    b = VL._anova_ll(np.array([float(z2[mask].sum())]), np.array([4.0]),
                     float(z2.sum()), float((z2 ** 2).sum()), 64)[0]
    check("the structural term is invariant to a constant factor on the profile",
          abs(a - b) < 1e-9, f"{a:.9f} vs {b:.9f}")


# --------------------------------------------------------------------------
def test_B_order_recovery() -> None:
    print("\nB. The order is recovered on geometry that is known by construction")

    cases = [
        ("3 clean tubes", 3, dict(n=3, width=300, pitch=560, offset=180)),
        ("1 tube", 1, dict(n=1, width=420, pitch=0, offset=700)),
        ("2 tubes", 2, dict(n=2, width=330, pitch=700, offset=280)),
        ("6 tubes", 6, dict(n=6, width=130, pitch=300, offset=90)),
        ("3 tubes, one fragmented", 3,
         dict(n=3, width=300, pitch=560, offset=180, frag=(290, 360))),
        ("3 tubes + bright clutter", 3,
         dict(n=3, width=300, pitch=560, offset=180, clutter=(1750, 1850, 45.0))),
        ("2 tubes, heavily over-dispersed", 2,
         dict(n=2, width=330, pitch=700, offset=280, overdisp=25.0)),
        ("3 tubes + glare on a wall", 3,
         dict(n=3, width=300, pitch=560, offset=180, glare=(1500, 1530))),
        ("3 tubes, faint walls", 3,
         dict(n=3, width=300, pitch=560, offset=180, wall_contrast=12.0)),
    ]
    for label, truth, kw in cases:
        act, frame = synth(**kw)
        f = VL.fit_lattice(act, background=frame)
        check(f"{label}: order {truth} recovered",
              f.n_vials == truth,
              f"got {f.n_vials}, dBIC {f.delta_bic:.1f}")
        if f.n_vials == truth and truth > 1:
            # the geometry, not just the count
            check(f"{label}: pitch recovered to within 5%",
                  abs(f.pitch_px - kw["pitch"]) <= 0.05 * kw["pitch"],
                  f"{f.pitch_px:.0f} px vs {kw['pitch']} px")

    # determinism: the same profile must give the same answer
    act, frame = synth(n=3, width=300, pitch=560, offset=180)
    a = VL.fit_lattice(act, background=frame)
    b = VL.fit_lattice(act, background=frame)
    check("the estimate is deterministic",
          (a.n_vials, a.offset_px, a.pitch_px, a.width_px)
          == (b.n_vials, b.offset_px, b.pitch_px, b.width_px))

    # an empty profile is refused, not answered
    f = VL.fit_lattice(np.zeros(W))
    check("a profile with no activity returns no cylinders and says why",
          f.n_vials == 0 and bool(f.reason), f"{f.n_vials}, {f.reason!r}")


# --------------------------------------------------------------------------
def test_C_the_two_activity_only_failures() -> None:
    print("\nC. The failure modes of an activity-only fit, and the cue that fixes them")

    # (i) one wide tube with a dead vertical slice through it
    act, frame = synth(n=1, width=500, pitch=0, offset=600, frag=(800, 880))
    f_act = VL.fit_lattice(act)
    f_joint = VL.fit_lattice(act, background=frame)
    check("activity alone reads a fragmented tube as two tubes "
          "(the failure being fixed; if this ever passes, the test data no "
          "longer reproduces the problem)",
          f_act.n_vials != 1, f"activity-only gave {f_act.n_vials}")
    check("the wall structure resolves it to one tube",
          f_joint.n_vials == 1, f"joint gave {f_joint.n_vials}")

    # (ii) the serious one: a tube whose flies are alive but barely moving
    for frac in (0.08, 0.02):
        act, frame = synth(n=3, width=300, pitch=560, offset=180, sick=(2, frac))
        f_act = VL.fit_lattice(act)
        f_joint = VL.fit_lattice(act, background=frame)
        check(f"activity alone drops a vial at {frac:.0%} of normal activity "
              f"(the phenotype-dependent failure being fixed)",
              f_act.n_vials != 3, f"activity-only gave {f_act.n_vials}")
        check(f"the wall structure keeps that vial at {frac:.0%} activity",
              f_joint.n_vials == 3,
              f"joint gave {f_joint.n_vials}, dBIC {f_joint.delta_bic:.1f}")

    print("      WHY THIS MATTERS: the vial an activity-only estimator drops is")
    print("      the least active one, which is the most impaired one. That makes")
    print("      the instrument's sensitivity depend on the phenotype being")
    print("      measured, and biases every treatment contrast in the direction")
    print("      of the treatment's own effect.")


# --------------------------------------------------------------------------
def test_D_physical_constraint() -> None:
    print("\nD. Tubes have glass on both sides, so their bands cannot touch")

    check("the bore-to-outside-diameter ratio is a dimension of real glassware",
          0.85 <= VL.BORE_TO_OD <= 0.96, f"{VL.BORE_TO_OD}")

    for kw in (dict(n=3, width=300, pitch=560, offset=180),
               dict(n=6, width=130, pitch=300, offset=90),
               dict(n=2, width=330, pitch=700, offset=280)):
        act, frame = synth(**kw)
        f = VL.fit_lattice(act, background=frame)
        if f.n_vials > 1:
            check(f"{f.n_vials} tubes: fitted pitch exceeds width / bore ratio",
                  f.pitch_px >= f.width_px / VL.BORE_TO_OD - 1e-6,
                  f"pitch {f.pitch_px:.1f} vs width/{VL.BORE_TO_OD} = "
                  f"{f.width_px / VL.BORE_TO_OD:.1f}")
            check(f"{f.n_vials} tubes: the fitted bands do not overlap",
                  all(f.bands[i][1] <= f.bands[i + 1][0]
                      for i in range(len(f.bands) - 1)),
                  f"{f.bands}")


# --------------------------------------------------------------------------
def test_E_against_the_manifest() -> None:
    print("\nE. Against the experiment's own record of its own apparatus")

    root = _LAB.climbing_root()
    para = (root / "Paraquat climbing assay 010626_new set") if root else None
    man = (para / "paraquat_video_manifest.csv") if para else None
    if man is None or not man.is_file():
        print("  [SKIP] the paraquat manifest is not on this machine; "
              "section E established nothing")
        print(_LAB.describe_roots())
        return

    try:
        from modules import climbing_video as CV
    except Exception as exc:                                   # pragma: no cover
        print(f"  [SKIP] climbing_video unavailable ({exc})")
        return

    with man.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    # A bounded, STRATIFIED sample: one recording per (arm, true count), so
    # both cylinder counts and every treatment arm are represented. Decoding a
    # video is tens of seconds and this suite must stay runnable; the full
    # 62-recording calibration is study_vial_lattice_accuracy.py, and the
    # accuracy figure comes from there.
    seen = set()
    sample = []
    for r in rows:
        try:
            key = (r.get("folder", ""), int(float(r["cylinders_in_frame"])))
        except (KeyError, TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        sample.append(r)
    sample = sample[:8]

    n_ok = n_tested = 0
    results = []
    for r in sample:
        path = para / r.get("folder", "") / r.get("video", "")
        if not path.is_file():
            print(f"  [SKIP] {r.get('video', '')}: not on this machine")
            continue
        n_true = int(float(r["cylinders_in_frame"]))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bg = CV.build_background(str(path), start_s=4.0)
                act = CV._activity_map(str(path), bg, start_s=4.0)
        except Exception as exc:
            print(f"  [SKIP] {path.name}: decode failed ({exc!r})")
            continue
        ap = act.sum(axis=0).astype(np.float64)
        f = VL.fit_lattice(ap, background=bg)
        n_tested += 1
        n_ok += int(f.n_vials == n_true)
        results.append((path.name, n_true, f.n_vials, f.confident, f.delta_bic))
        print("        %-14s %-26s manifest %d, lattice %d, dBIC %6.1f  %s"
              % (r.get("folder", "")[:14], path.name[:26], n_true, f.n_vials,
                 f.delta_bic, "agrees" if f.n_vials == n_true else "DISAGREES"))

    if not n_tested:
        print("  [SKIP] no recordings could be read; section E established nothing")
        return

    print("")
    print(f"      {n_ok}/{n_tested} agree with the manifest on this sample."
          f" The RATE is reported, not asserted:")
    print("      a sample of 8 cannot establish an accuracy, and the full "
          "62-recording")
    print("      calibration is validation/study_vial_lattice_accuracy.py.")

    # What IS asserted is the DIRECTION of any error. An over-count invents a
    # cylinder, fills it with whatever is in that part of the frame, and
    # reports a climbing index for it next to the real ones -- a confidently
    # wrong number. An under-count leaves a visibly missing column, which an
    # operator catches. Only the first is silent, so only the first is a
    # failure of this suite; and an over-count is tolerated only when the
    # estimator itself says it is not confident, because that routes the
    # recording to the manual region editor.
    silent_over = [t for t in results if t[2] > t[1] and t[3]]
    check("no recording is silently OVER-counted (an invented cylinder reported "
          "with confidence)",
          not silent_over,
          "; ".join(f"{n}: manifest {a}, lattice {b}, dBIC {d:.1f}"
                    for n, a, b, _c, d in silent_over))

    unconf = [t for t in results if not t[3]]
    if unconf:
        print(f"      {len(unconf)} recording(s) flagged not confident, which "
              f"routes them to the manual editor:")
        for n, a, b, _c, d in unconf:
            print(f"        {n}: manifest {a}, lattice {b}, dBIC {d:.2f}")


def main() -> int:
    print("=" * 78)
    print("VALIDATION: the lattice cylinder-count estimator")
    print("=" * 78)
    test_A_closed_forms_are_right()
    test_B_order_recovery()
    test_C_the_two_activity_only_failures()
    test_D_physical_constraint()
    test_E_against_the_manifest()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL VIAL-LATTICE VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
