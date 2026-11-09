"""
validate_experimental_unit.py
N4 — what counts as a replicate, decided by a test rather than by a column name.

ESTABLISHES: L1 and L2. The unit key is identified by an objective property of
the data, and the consequences of getting it wrong are measured on DroPhenix's
own demo data.

WHY THIS SUITE EXISTS
---------------------
Register N4. `data_triage._pick_unit` returned the first present of `VialID`,
`FlyID`, `Channel`, `Replicate`. `Replicate` is a bare integer reused in every
group, so on the platform's own built-in demo data -- which contains **36 real
vials** -- the unit was taken to be `Replicate`, with three distinct values.

Two measured consequences, both on that demo data:

  1. A FABRICATED DESIGN EFFECT. `_check_clustering` pooled twelve unrelated
     vials -- different genotypes, different treatments -- into each of three
     "clusters", so the between-cluster variance it measured was genotype
     variance. It reported "ICC = 0.11 over 3 Replicates. The design effect is
     22.9, so standard errors must be 378% larger", on the basis of a number
     with no meaning.

  2. A FABRICATED BLOCKER, which the register did not record and which is
     worse. `_check_replication` counted distinct `Replicate` values per
     treatment level and found one each for Pantothenate, Spermine and
     Spermine+Pantothenate, so it declared the Treatment effect unanalysable:
     "has level(s) [...] with only one Replicate. With a single unit the
     Treatment effect and that unit's own peculiarity are the same number."
     Each of those arms in fact has **6 real vials**. The platform told the
     user, on its own demo data, that the treatment comparison could not be
     made, when it has six replicates per arm.

THE TEST THAT DECIDES IT, which needs no convention and no belief
    A physical unit cannot be measured twice at the same timepoint.

    So a candidate key is a valid experimental unit if and only if it does not
    repeat within a timepoint. Measured on the demo data:

        Replicate alone                              315 repeats  -> NOT a unit
        Genotype x Treatment x Sex x Replicate          0 repeats  -> a unit

    That is an objective property of the table, checkable on any dataset, and
    it is what the fix uses. It also means the fix does not have to fall back
    on refusal here: the composite key is a unit, and it recovers all 36.

    Where no candidate passes -- `Replicate` with nothing to compose it with --
    the ICC is refused with `ambiguous_unit` rather than computed, which is the
    register's own acceptance criterion.

Run:  python validation/validate_experimental_unit.py   (exit 0 = all passed)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import data_triage as T        # noqa: E402
from modules.data_parser import load_data   # noqa: E402


def demo_frame():
    out = load_data()
    return out[0] if isinstance(out, tuple) else out


def repeats_within_timepoint(df, cols, time_col="Time"):
    """How often a candidate key repeats inside one timepoint."""
    key = list(cols) + ([time_col] if time_col in df.columns else [])
    return int(df.duplicated(subset=key).sum())


def main():
    results = []

    def chk(name, got, expected, tol=0):
        ok = bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol
        results.append((ok, name, "got %g, expected %g" % (got, expected)))

    def boolchk(name, ok, got):
        results.append((bool(ok), name, got))

    df = demo_frame()
    FACT = ["Genotype", "Treatment", "Sex"]
    COMPOSITE = FACT + ["Replicate"]

    # ──────────────────────────────────────────────────────────────────────
    # A. the criterion itself
    # ──────────────────────────────────────────────────────────────────────
    chk("A1 the demo data holds 36 real vials",
        df[COMPOSITE].drop_duplicates().shape[0], 36)
    chk("A2 Replicate alone has only three values",
        df["Replicate"].nunique(), 3)
    r_rep = repeats_within_timepoint(df, ["Replicate"])
    boolchk("A3 Replicate alone repeats within a timepoint, so it cannot be a "
            "unit", r_rep > 0,
            "%d duplicate (Replicate x Time) rows -- a vial cannot be measured "
            "twice at one timepoint" % r_rep)
    r_comp = repeats_within_timepoint(df, COMPOSITE)
    boolchk("A4 the composite key never repeats within a timepoint, so it is "
            "a unit", r_comp == 0,
            "%d duplicate (composite x Time) rows" % r_comp)
    boolchk("A5 the module exposes the criterion rather than hiding it",
            hasattr(T, "unit_key_is_valid"),
            "unit_key_is_valid present: %s" % hasattr(T, "unit_key_is_valid"))
    if hasattr(T, "unit_key_is_valid"):
        boolchk("A6 ... and it agrees with the measurement, both ways",
                T.unit_key_is_valid(df, ["Replicate"])[0] is False
                and T.unit_key_is_valid(df, COMPOSITE)[0] is True,
                "Replicate rejected, composite accepted")

    # ──────────────────────────────────────────────────────────────────────
    # B. the register's acceptance test, on the demo data
    # ──────────────────────────────────────────────────────────────────────
    rep = T.triage(df)
    n_units = (df[list(getattr(rep, "unit_cols", []) or COMPOSITE)]
               .drop_duplicates().shape[0])
    boolchk("B1 triage identifies 36 units on the demo data, not 3",
            n_units == 36,
            "unit_cols = %s -> %d units" % (getattr(rep, "unit_cols", None),
                                            n_units))
    boolchk("B2 the chosen key is the composite, and is reported as such",
            list(getattr(rep, "unit_cols", []) or []) == COMPOSITE,
            "unit_cols = %s, unit_col label = %r"
            % (getattr(rep, "unit_cols", None), rep.unit_col))
    clus = [f for f in rep.findings if f.code == "clustering"]
    boolchk("B3 no design effect is reported from Replicate alone",
            all("over 3 " not in f.message and "3 Replicates" not in f.message
                for f in clus),
            "clustering findings: %d; none cites 3 clusters" % len(clus))
    if clus:
        boolchk("B4 ... and any ICC reported is over the real units",
                any(("36" in f.message) or ("ambiguous" in f.code)
                    for f in clus),
                clus[0].message[:150])
    else:
        boolchk("B4 ... or the ICC is not reported at all", True,
                "no clustering finding emitted")

    # ──────────────────────────────────────────────────────────────────────
    # C. the fabricated blocker must be gone
    # ──────────────────────────────────────────────────────────────────────
    boolchk("C1 Treatment is no longer blocked on the platform's own demo data",
            rep.answerable.get("Treatment") != "blocked",
            "Treatment verdict = %r (was 'blocked')"
            % rep.answerable.get("Treatment"))
    bad = [f for f in rep.findings
           if f.code == "no_replication" and "Replicate" in f.message
           and f.factor == "Treatment"]
    boolchk("C2 ... and the 'only one Replicate' finding is gone",
            not bad, "%d such findings" % len(bad))
    # the counts the replication check should now be using
    per_treat = (df[COMPOSITE].drop_duplicates()
                 .groupby("Treatment").size().to_dict())
    boolchk("C3 the real unit count per treatment arm is six, not one",
            min(per_treat.values()) == 6,
            "vials per arm: %s" % per_treat)

    # ──────────────────────────────────────────────────────────────────────
    # D. refusal where the unit genuinely is ambiguous
    # ──────────────────────────────────────────────────────────────────────
    # The refusal path is when NO candidate passes the repeat test. Here
    # `Replicate` repeats within a timepoint -- two rows share (Replicate 1,
    # Time 10) -- so it pools units, and there is no varying factor to compose
    # it with. Nothing qualifies, so nothing may be claimed.
    #
    # Note what is NOT this case: `Replicate` with no factors but no repeats
    # either. There the index is consistent with being the unit and nothing in
    # the data contradicts it, so it is accepted. An earlier version of this
    # check demanded a refusal there, which would have been refusing on the
    # strength of a column's name rather than its behaviour -- the same mistake
    # as trusting it.
    thin = pd.DataFrame({
        "Replicate": [1, 1, 2, 2, 3, 3],
        "Time": [10, 10, 10, 10, 20, 20],
        "Count": [5, 7, 6, 8, 4, 6],
        "n": [20] * 6,
    })
    ok_rep, why_rep, n_rep_thin, _ = T.unit_key_is_valid(thin, ["Replicate"])
    boolchk("D0 the fixture really does pool: Replicate repeats within a "
            "timepoint",
            ok_rep is False and n_rep_thin > 0,
            "%d repeats -- %s" % (n_rep_thin, why_rep))
    rt = T.triage(thin)
    amb = [f for f in rt.findings if f.code == "ambiguous_unit"]
    boolchk("D1 with no candidate passing the test, the unit is declared "
            "ambiguous",
            len(amb) >= 1, "findings: %s" % [f.code for f in rt.findings])
    boolchk("D2 ... and no ICC or design effect is reported there",
            not any(f.code == "clustering" for f in rt.findings),
            "clustering findings: %d"
            % len([f for f in rt.findings if f.code == "clustering"]))
    if amb:
        boolchk("D3 ... and the refusal says what would resolve it",
                any(k in amb[0].message.lower()
                    for k in ("vialid", "identifier", "unit")),
                amb[0].message[:150])

    # ──────────────────────────────────────────────────────────────────────
    # E. a real identifier is still used alone, and still checked
    # ──────────────────────────────────────────────────────────────────────
    with_vial = df.copy()
    with_vial["VialID"] = (with_vial["Genotype"].astype(str) + "_"
                           + with_vial["Treatment"].astype(str) + "_"
                           + with_vial["Sex"].astype(str) + "_"
                           + with_vial["Replicate"].astype(str))
    rv = T.triage(with_vial)
    boolchk("E1 a valid VialID is used on its own",
            list(getattr(rv, "unit_cols", []) or []) == ["VialID"]
            and rv.unit_col == "VialID",
            "unit_cols = %s" % (getattr(rv, "unit_cols", None),))
    chk("E2 ... and recovers the same 36 units",
        with_vial["VialID"].nunique(), 36)
    # an identifier that is reused across groups must be REJECTED, not trusted
    bad_vial = df.copy()
    bad_vial["VialID"] = bad_vial["Replicate"]        # reused in every group
    rb = T.triage(bad_vial)
    boolchk("E3 an identifier that repeats within a timepoint is rejected even "
            "when it is called VialID",
            list(getattr(rb, "unit_cols", []) or []) != ["VialID"],
            "unit_cols = %s -- the name is not evidence; the repeat test is"
            % (getattr(rb, "unit_cols", None),))
    boolchk("E4 ... and the composite is used instead, recovering 36",
            (bad_vial[list(getattr(rb, "unit_cols", []) or ["VialID"])]
             .drop_duplicates().shape[0]) == 36,
            "units found: %d"
            % bad_vial[list(getattr(rb, "unit_cols", []) or ["VialID"])]
            .drop_duplicates().shape[0])

    print("validate_experimental_unit -- N4")
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
