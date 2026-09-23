"""validate_climbing_band — the band-and-zone scorer's arithmetic and its QC gate.

THIS MODULE HAD NO VALIDATION SUITE AT ALL until now, which was worse than it
looked: `modules/climbing_band.py` is the scorer that produced every Day-3,
Day-7 and Day-15 number, and earlier reports of "28 suites passed" meant the
OTHER modules still passed. Nothing was checking this one.

What is testable here is the part that does not need a video: the Performance
Index arithmetic, the three-layer split, and the reason codes that decide
whether a reading may be reported as a biological result. The detector itself
is measured against hand counts, not asserted against fixtures, and those
numbers live in the module's VALIDATION_EVIDENCE.

Each check states what it is protecting, because a test whose purpose is not
recorded gets deleted the first time it is inconvenient.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import climbing_band as CB  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, PASS if ok else FAIL, detail))


# ─────────────────────────────────────────────────────────── the arithmetic ──
def test_pi_bounds():
    """A count is a cardinality and a cylinder is sealed.

    Protects F3. `pi(-32, 20)` returned -1.60 before it was fixed: a negative
    proportion reported as a score rather than as the estimator failure it was.
    Reachable from inside the module -- `detect_crossings` produced net counts
    to -32, because nothing in a difference of two running totals forbids it.
    """
    check("pi refuses a negative count", CB.pi(-32, 20) is None,
          "was -1.60 before F3")
    check("pi refuses a small negative count", CB.pi(-5, 20) is None)
    check("pi refuses a count above N", CB.pi(21, 20) is None)
    check("pi refuses a zero denominator", CB.pi(7, 0) is None)
    check("pi refuses a negative denominator", CB.pi(7, -20) is None)
    check("pi accepts zero", CB.pi(0, 20) == 0.0)
    check("pi accepts the full cylinder", CB.pi(20, 20) == 1.0)
    check("pi is a plain proportion", abs(CB.pi(7, 20) - 0.35) < 1e-12)


# ──────────────────────────────────────────────────────── the three layers ──
def test_layers_are_separate():
    """Layer 2 always computes; Layer 3 only when entitled to.

    The point of the split: a refusal must not also hide the diagnosis. If
    `pi_estimate` went None whenever `pi_reportable` did, a user could not see
    what the instrument thought it saw, and would have no way to work out why
    it refused.
    """
    a = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          validated_domain=None)
    check("an unvalidated estimator still yields an estimate",
          a.pi_estimate is not None and abs(a.pi_estimate - 0.35) < 1e-12)
    check("an unvalidated estimator yields NO reportable PI",
          a.pi_reportable is None)
    check("and says so by name", a.verdict == CB.NOT_REPORTABLE)
    check("naming the reason", "ESTIMATOR_NOT_VALIDATED" in a.reasons)


def test_gate_opens_when_validated():
    """The gate must be a gate, not a wall.

    If nothing can ever pass, the refusal carries no information and the first
    person to find it inconvenient will delete it. So the same reading passes
    the moment the manifest says the estimator is validated.
    """
    a = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          validated_domain="day7-male-10-60s")
    check("a clean reading passes once the estimator is validated",
          a.verdict == CB.VALID, str(a.reasons))
    check("and only then is a PI reportable",
          a.pi_reportable is not None and abs(a.pi_reportable - 0.35) < 1e-12)


def test_impossible_count_is_never_reportable():
    """An impossible count must refuse even with everything else perfect."""
    a = CB.assess_reading(-32, 20, n_source="MANUAL", n_quality="VERIFIED",
                          validated_domain="anything")
    check("a negative count is NOT_REPORTABLE", a.verdict == CB.NOT_REPORTABLE)
    check("by INVALID_COUNT_RANGE", "INVALID_COUNT_RANGE" in a.reasons)
    check("and no number escapes", a.pi_reportable is None)
    b = CB.assess_reading(25, 20, n_source="MANUAL", n_quality="VERIFIED",
                          validated_domain="anything")
    check("a count above N is NOT_REPORTABLE", b.verdict == CB.NOT_REPORTABLE)


# ──────────────────────────────────────────────────────────── the denominator ──
def test_denominator_provenance():
    """PI is a proportion. Without a denominator it is not a weak measurement,
    it is not a measurement.

    NOMINAL is not treated as good enough, and that is measured rather than
    fastidious: on this dataset the loading stated as 20 per cylinder was 14-25
    in the operator's own notes, which moves the PI by up to 30%.
    """
    u = CB.assess_reading(7, None, n_source="UNKNOWN", n_quality="UNCERTAIN",
                          validated_domain="anything")
    check("an unknown N is NOT_REPORTABLE", u.verdict == CB.NOT_REPORTABLE)
    check("by DENOMINATOR_UNKNOWN", "DENOMINATOR_UNKNOWN" in u.reasons)

    n = CB.assess_reading(7, 20, n_source="NOMINAL", n_quality="ESTIMATED",
                          validated_domain="anything")
    check("a nominal N needs review", n.verdict == CB.REVIEW_REQUIRED)
    check("by DENOMINATOR_UNCERTAIN", "DENOMINATOR_UNCERTAIN" in n.reasons)
    check("and a reviewable reading reports no PI either",
          n.pi_reportable is None)

    z = CB.assess_reading(0, 0, n_source="MANUAL", n_quality="VERIFIED",
                          validated_domain="anything")
    check("a zero N is NOT_REPORTABLE", z.verdict == CB.NOT_REPORTABLE)
    check("by INVALID_DENOMINATOR", "INVALID_DENOMINATOR" in z.reasons)
    check("and does not divide by one instead",
          z.pi_estimate is None,
          "the old inline `mid / max(n_flies, 1)` returned the raw count here")

    for bad in ("nominal", "GUESS", ""):
        try:
            CB.assess_reading(7, 20, n_source=bad)
            check("an unknown n_source is refused (%r)" % bad, False)
        except ValueError:
            check("an unknown n_source is refused (%r)" % bad, True)


# ───────────────────────────────────────────────────────────────── geometry ──
def test_geometry_codes():
    """Measured: widening the box on a 100 px cylinder holding 20 flies gave 19
    found at 100-200 px and 0 of 20 at 300 px, silently, because the assumed
    fly grew from 4.0 to 12.0 px. 1.6x is where that is refused.
    """
    a = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          box_over_tube=1.9, validated_domain="anything")
    check("a box far wider than the tube is NOT_REPORTABLE",
          a.verdict == CB.NOT_REPORTABLE)
    check("by TUBE_WIDTH_MISMATCH", "TUBE_WIDTH_MISMATCH" in a.reasons)

    b = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          box_over_tube=1.35, validated_domain="anything")
    check("a slightly wide box needs review", b.verdict == CB.REVIEW_REQUIRED)
    check("by ROI_OVERSIZED", "ROI_OVERSIZED" in b.reasons)

    c = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          box_over_tube=1.02, validated_domain="anything")
    check("a well-drawn box raises nothing", c.verdict == CB.VALID, str(c.reasons))


# ──────────────────────────────────────────────────── the estimator's doubts ──
def test_quality_codes():
    """The estimator already reports its own uncertainty; these turn it into a
    verdict instead of leaving it in a column nobody reads.
    """
    w = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          bracket_width=9, validated_domain="anything")
    check("a wide bracket needs review", "BRACKET_TOO_WIDE" in w.reasons)

    g = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          accounted_for=28, validated_domain="anything")
    check("a broken conservation sum needs review",
          "CONSERVATION_VIOLATED" in g.reasons,
          "above + below must equal N in a sealed cylinder")

    s = CB.assess_reading(4.5, 20, n_source="MANUAL", n_quality="VERIFIED",
                          static_removed=7.2, count_upper=8.0,
                          validated_domain="anything")
    check("a reading carried by static objects needs review",
          "STATIC_ARTIFACT_EXCESS" in s.reasons,
          "the empty-tube case: 7.2 of 8.0 detections never moved")

    ok = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                           bracket_width=2, static_removed=1.0, count_upper=8.0,
                           accounted_for=20, validated_domain="anything")
    check("a good reading raises none of them", ok.verdict == CB.VALID,
          str(ok.reasons))


def test_all_reasons_are_collected():
    """Every applicable code is reported, not just the first one noticed.

    A reading that fails three ways should say so three times: an operator who
    fixes only the first failure and re-runs has wasted a bench session.
    """
    a = CB.assess_reading(-5, None, n_source="UNKNOWN", n_quality="UNCERTAIN",
                          bracket_width=9, box_over_tube=1.9,
                          accounted_for=40, validated_domain=None)
    for code in ("DENOMINATOR_UNKNOWN", "TUBE_WIDTH_MISMATCH",
                 "BRACKET_TOO_WIDE", "ESTIMATOR_NOT_VALIDATED"):
        check("multiple failures all reported: %s" % code, code in a.reasons)
    check("worst severity wins", a.verdict == CB.NOT_REPORTABLE)
    check("codes are ordered worst first",
          CB.REASONS[a.reasons[0]][0] == CB.NOT_REPORTABLE)


def test_every_code_is_documented():
    """A reason code the software can emit but cannot explain is worse than no
    code at all: it looks like a diagnosis and is not one.
    """
    for code, (sev, what, why) in CB.REASONS.items():
        check("%s has a severity" % code,
              sev in (CB.VALID, CB.REVIEW_REQUIRED, CB.NOT_REPORTABLE))
        check("%s says what it means" % code, bool(what.strip()))
        check("%s says where its threshold comes from" % code, bool(why.strip()))


def test_manifest_is_honest():
    """The manifest is read by `assess_reading`, so it is code, not prose.

    It must not claim a validated domain while the evidence in the same file
    says the slope against hand counts is 0.003 with a CI spanning zero.
    """
    d37 = CB.VALIDATION_EVIDENCE["day37"]
    lo, hi = d37["slope_ci95"]
    unfaithful = not (lo <= 1.0 <= hi)
    check("the confirmation set excludes a faithful instrument", unfaithful,
          "slope %.3f, CI [%.3f, %.3f], n=%d" % (d37["slope"], lo, hi, d37["n"]))
    check("so VALIDATED_DOMAIN is None", CB.VALIDATED_DOMAIN is None,
          "opening this gate is a deliberate edit, made when evidence allows")
    check("and the estimator is frozen", CB.ESTIMATOR_SELECTION == "FROZEN")
    check("with the sets it may not be re-chosen on recorded",
          len(CB.DEVELOPMENT_SETS) >= 3 and len(CB.CONFIRMATION_SETS) >= 1)


def test_review_codes_are_not_a_quality_filter():
    """The REVIEW_REQUIRED codes describe a state; they do not predict error.

    Measured on Day 15 (432 readings with hand counts): a reading raising ANY
    code has MAE 4.42, one raising none has 5.97, Welch p = 0.012. The codes
    run BACKWARDS as a quality filter, because the error on this instrument is
    set by where a reading sits on a compressed calibration (reported = 6.08 +
    0.199 x true) rather than by any per-reading condition.

    This test exists so that the distinction cannot be quietly lost: the codes
    that GATE must be the logical ones, and the descriptive ones must never be
    promoted to NOT_REPORTABLE on the theory that they mark bad readings.
    """
    descriptive = ("BRACKET_TOO_WIDE", "CONSERVATION_VIOLATED",
                   "STATIC_ARTIFACT_EXCESS")
    for code in descriptive:
        check("%s only asks for review, it does not gate" % code,
              CB.REASONS[code][0] == CB.REVIEW_REQUIRED,
              "measured to correlate with error in the WRONG direction; "
              "gating on it would select the worse readings")

    logical = ("INVALID_COUNT_RANGE", "INVALID_DENOMINATOR",
               "DENOMINATOR_UNKNOWN", "TUBE_WIDTH_MISMATCH",
               "ESTIMATOR_NOT_VALIDATED")
    for code in logical:
        check("%s gates, and does so on logic not on a correlation" % code,
              CB.REASONS[code][0] == CB.NOT_REPORTABLE)

    # a reading flagged only descriptively must still keep its estimate
    a = CB.assess_reading(7, 20, n_source="MANUAL", n_quality="VERIFIED",
                          bracket_width=9, static_removed=7.0, count_upper=8.0,
                          validated_domain="anything")
    check("a descriptively flagged reading keeps its estimate",
          a.pi_estimate is not None)
    check("and is REVIEW_REQUIRED, not refused", a.verdict == CB.REVIEW_REQUIRED)


def test_explain_is_readable():
    a = CB.assess_reading(7, 20, n_source="NOMINAL", n_quality="ESTIMATED",
                          validated_domain=None)
    t = a.explain()
    check("explain names the verdict", "NOT REPORTABLE" in t)
    check("explain names each code", "ESTIMATOR_NOT_VALIDATED" in t
          and "DENOMINATOR_UNCERTAIN" in t)
    check("explain does not print a PI it refused", "0.35" not in t)


def main() -> int:
    for fn in (test_pi_bounds, test_layers_are_separate,
               test_gate_opens_when_validated,
               test_impossible_count_is_never_reportable,
               test_denominator_provenance, test_geometry_codes,
               test_quality_codes, test_all_reasons_are_collected,
               test_every_code_is_documented, test_manifest_is_honest,
               test_review_codes_are_not_a_quality_filter,
               test_explain_is_readable):
        fn()

    width = max(len(n) for n, _s, _d in _results)
    fails = 0
    for name, status, detail in _results:
        if status == FAIL:
            fails += 1
        mark = "ok  " if status == PASS else "FAIL"
        line = "  %s  %-*s" % (mark, width, name)
        if detail and status == FAIL:
            line += "   <- %s" % detail
        print(line)
    print("\n%d checks, %d failed" % (len(_results), fails))
    if not fails:
        print("\nNOTE: these check the arithmetic and the QC gate. They do NOT")
        print("establish that the detector counts flies correctly -- that is")
        print("measured against hand counts, and the current numbers are in")
        print("climbing_band.VALIDATION_EVIDENCE. The slope against hand counts")
        print("on the only independent set is %.3f, CI [%.3f, %.3f], n=%d."
              % (CB.VALIDATION_EVIDENCE["day37"]["slope"],
                 *CB.VALIDATION_EVIDENCE["day37"]["slope_ci95"],
                 CB.VALIDATION_EVIDENCE["day37"]["n"]))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
