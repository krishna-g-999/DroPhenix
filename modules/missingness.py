"""
DroPhenix — missingness.py
The typed vocabulary for "there is no value here", and why one is needed.

A NaN records that a value is absent. It does not record WHY, and the reason is
what determines what an analysis is allowed to do next:

  OBSERVED           a real measurement. The only code under which a value may
                     enter an estimate.
  NOT_MEASURED       the measurement was never attempted at this unit/timepoint
                     (a timepoint beyond the end of a recording, a vial outside
                     the field of view, a day the monitor was not running).
                     Missing by design, independent of the phenotype — so it may
                     be excluded, provided the exclusion is reported and is not
                     correlated with genotype or treatment.
  MEASUREMENT_FAILED the measurement was attempted and did not produce a usable
                     value (zero detections in a loaded vial, an undecodable
                     frame, a monitor status other than OK). NOT a measurement
                     of zero. Potentially informative missingness: a detector
                     fails more often on the vials that are hardest to score,
                     which in a climbing assay are the impaired ones, so
                     dropping these rows can bias the comparison toward the
                     null. Must be reported per group before exclusion.
  BELOW_THRESHOLD    the quantity was measured and lies below the limit of
                     detection or the analysable minimum. Left-censored, not
                     missing: the value is known to be in [0, threshold), which
                     is information a drop discards.
  CENSORED_RIGHT     the event had not occurred when observation ended
  CENSORED_LEFT      the event had already occurred when observation began
  CENSORED_INTERVAL  the event occurred between two observations
                     — survival codes. These rows carry information and must
                     reach the estimator, never be filtered out.
  REFUSED            the input was present but a precondition for the derived
                     quantity was not met, so this module declines to compute
                     it (no N_start, so no rate; a non-finite series, so no
                     period). A refusal is a result: it is recorded, reported
                     and never substituted with a default.

The distinction that motivates the whole module: in a negative-geotaxis assay a
Performance Index of 0.0 is the PI of a vial with exactly half its flies above
the scoring line, and -100.0 is the PI of a vial where none climbed. So writing
`PI = 0.0` for a vial the detector could not score moves that vial +100 points
toward the healthy phenotype — and detector failures are not independent of
genotype. See docs/BUG_AND_RISK_REGISTER.md finding N8.

Usage: carry `Missingness` as a column alongside the value. The value is NaN
whenever the code is not OBSERVED; the code says what the NaN means, and
`QC_flag` / `unusable_reason` say which specific condition fired.
"""
from typing import Iterable, Sequence

OBSERVED = "OBSERVED"
NOT_MEASURED = "NOT_MEASURED"
MEASUREMENT_FAILED = "MEASUREMENT_FAILED"
BELOW_THRESHOLD = "BELOW_THRESHOLD"
CENSORED_RIGHT = "CENSORED_RIGHT"
CENSORED_LEFT = "CENSORED_LEFT"
CENSORED_INTERVAL = "CENSORED_INTERVAL"
REFUSED = "REFUSED"

#: Every code this vocabulary recognises.
CODES: Sequence[str] = (
    OBSERVED, NOT_MEASURED, MEASUREMENT_FAILED, BELOW_THRESHOLD,
    CENSORED_RIGHT, CENSORED_LEFT, CENSORED_INTERVAL, REFUSED,
)

#: Codes whose rows carry information and must reach the estimator rather than
#: being filtered out upstream of it.
INFORMATIVE: Sequence[str] = (
    BELOW_THRESHOLD, CENSORED_RIGHT, CENSORED_LEFT, CENSORED_INTERVAL,
)

#: The column name used throughout DroPhenix.
COLUMN = "Missingness"


def is_observed(code) -> bool:
    """True only for a real measurement. An empty or absent code is not one."""
    return str(code) == OBSERVED


def check_code(code) -> str:
    """Return `code` if it is part of the vocabulary; raise otherwise.

    A typo in a missingness code is worse than no code at all: it reads as
    "not OBSERVED" to every consumer while naming a reason that no consumer
    recognises, so the row is silently excluded for an unstated cause.
    """
    s = str(code)
    if s not in CODES:
        raise ValueError(
            f"missingness.check_code: {s!r} is not a recognised code. "
            f"Use one of: {', '.join(CODES)}."
        )
    return s


def summarise(codes: Iterable) -> dict:
    """Count each code present, for the exclusion report an analysis must give.

    The point of returning this rather than a single "n excluded" is that
    NOT_MEASURED and MEASUREMENT_FAILED need different handling: the first is
    missing by design, the second can be correlated with the phenotype.
    """
    out: dict = {}
    for c in codes:
        s = str(c) if c is not None and str(c) != "nan" else OBSERVED
        out[s] = out.get(s, 0) + 1
    return out
