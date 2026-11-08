"""
validate_climbing_thiem.py
Climbing-versus-age against REAL published per-fly escape-climbing records,
with the paper's own summary table as the target.

ESTABLISHES
    L2 -- the published Figure 1B summary (eight values, mean and SEM, two
          stimulus sets x four ages) is reproduced exactly from the per-trial
          cells, and the per-trial cells are reproduced from the per-fly cells.
    L3 -- the repeated-measures structure is measured, and the consequence of
          ignoring it is quantified.
    It does NOT establish anything about DroPhenix's own fly DETECTION. This is
    hand-scored data; no image was analysed.

WHY THIS SUITE EXISTS
---------------------
`docs/REFERENCE_DATASETS.md` section 3 listed this dataset as "found but not
usable -- the sheets are unlabelled figure-panel exports (`Unnamed: 0...`),
would need the Data guide.docx read alongside". The data guide has now been
downloaded from the same Zenodo record, the panels are labelled, and the
dataset is usable. `SCIENTIFIC_VALIDATION_PLAN.md` section 2.5 wanted a second
independent climbing-versus-age benchmark; this is it.

What makes it worth the trouble is the FAILURE CODING. Every trial cell is
either a time in seconds or a typed failure: the region the fly reached before
the 10 s cut-off (I, II, III, IV), a refusal to initiate (FAILED, F), a jump or
flight instead of a climb (JUMP, A), or a reason the fly could not be scored at
all (DEAD, LEG DEFECT). Register N8 is that DroPhenix scored a failed
measurement as `PI = 0.0`; this is a real dataset in which the distinction
between "did not climb", "climbed partway", "jumped instead" and "could not be
tested" is recorded, and it is the right place to check that those are not
collapsed.

DATA (real, published, CC-BY 4.0)
    Thiem J, Viskadourou M, Gaitanidis A, Stravopodis DJ, Strauss R, Duch C,
    Consoulas C (2024)
    "Biological aging of two innate behaviors of Drosophila melanogaster:
     escape climbing versus courtship learning and memory"
    Data: doi:10.5281/zenodo.10629747 (Zenodo record 10629747)
    Local: reference_datasets/climbing_thiem/thiem2024.xlsx
           reference_datasets/climbing_thiem/Data_guide.docx

    Oregon-R males, singly isolated. Each fly receives, in two rounds, three
    Light Stimuli (LS, a tap or flip) and six Moderate Stimuli (MS, 1 s
    vortexing). The measurement is the time to climb 6 cm up the vial wall
    within 10 s; the 6 cm is divided into four 1.5 cm regions and a fly that
    does not finish is coded by the region it reached. Ages 4, 6, 8 and 10
    weeks.

THE PUBLISHED TARGET (the sheet's own Figure 1B table)
    age (weeks)      LS mean    SEM     N        MS mean    SEM     N
         4            85.50    2.318    6          70.64   3.187   12
         6            80.74    5.226    6          65.48   3.840   12
         8            73.15    3.095    6          77.65   1.717   12
        10            22.09    4.086    6          39.77   1.411   12

    N is the number of TRIALS (2 rounds x 3 LS = 6; 2 x 6 = 12), not the
    number of flies -- which is the point of check E.

ONE TRAP IN THE FILE, worth knowing before trusting any parse of it
    The region codes mix Latin and GREEK letters: Latin "I" and Greek capital
    Iota (U+0399) both appear, as do Greek Alpha for "A" and Greek Epsilon for
    "E". Unnormalised, region I splits into two different failure codes and
    both counts are wrong. Check F pins this.

Run:  python validation/validate_climbing_thiem.py   (exit 0 = all passed)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "reference_datasets" / "climbing_thiem" / "thiem2024.xlsx"
GUIDE = ROOT / "reference_datasets" / "climbing_thiem" / "Data_guide.docx"

SHEET = "Figure  1"          # two spaces, as the file has it
LS_COLS = [5, 6, 7, 8, 9, 10]
MS_COLS = list(range(13, 25))
AGES = [4, 6, 8, 10]

# The sheet's own Figure 1B table, read out of rows 193-196.
PUBLISHED = {(4, "LS"): (85.50, 2.318, 6), (4, "MS"): (70.64, 3.187, 12),
             (6, "LS"): (80.74, 5.226, 6), (6, "MS"): (65.48, 3.840, 12),
             (8, "LS"): (73.15, 3.095, 6), (8, "MS"): (77.65, 1.717, 12),
             (10, "LS"): (22.09, 4.086, 6), (10, "MS"): (39.77, 1.411, 12)}

# Greek look-alikes used in the failure codes.
GREEK = {"Ι": "I", "Ν": "N", "Υ": "V", "Α": "A", "Ε": "E"}
FRACTION = re.compile(r"^(\d+)\s*/\s*(\d+)$")

# Typed failures, grouped by what they mean for the measurement.
PARTIAL = {"I", "II", "III", "IV"}          # climbed, did not finish in 10 s
NO_CLIMB = {"FAILED", "F"}                  # did not initiate a climb
NOT_CLIMBING = {"JUMP", "A"}                # jumped or flew instead
UNTESTABLE = {"DEAD", "LEG DEFECT"}         # could not be scored


def normalise(v):
    s = str(v).strip()
    for g, latin in GREEK.items():
        s = s.replace(g, latin)
    return s


def classify(v):
    """
    One trial cell: a time, a typed failure, a trial-summary fraction, or empty.

    Nothing is coerced. A region code is not a missing value and it is not a
    zero time; collapsing it to either is the error this dataset is here to
    catch.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "empty", np.nan
    if isinstance(v, (int, float)):
        return "time", float(v)
    s = normalise(v)
    if FRACTION.match(s):
        return "fraction", np.nan
    try:
        return "time", float(s)
    except ValueError:
        return "fail:" + s.upper(), np.nan


def main():
    results = []

    def chk(name, got, expected, tol):
        ok = bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol
        results.append((ok, name, "got %.4f, published %.4f (tol %g)"
                        % (got, expected, tol)))

    def boolchk(name, ok, got):
        results.append((bool(ok), name, got))

    if not XLSX.exists():
        print("[SKIP] dataset not present: %s" % XLSX)
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    d = pd.ExcelFile(XLSX).parse(SHEET, header=None)

    boolchk("A1 the data guide is in the repository, so the panels are labelled",
            GUIDE.exists(),
            "%s" % ("present" if GUIDE.exists() else "MISSING -- the sheets are "
                    "unlabelled figure exports without it"))

    # ---- A. block boundaries from the sheet's own AGE: markers -------------
    # Figure 1B's own header row also reads "AGE (WEEKS)", so the search stops
    # at the FIGURE 1B marker rather than matching on text alone.
    fig1b = next((i for i in range(d.shape[0])
                  if any(isinstance(v, str) and "FIGURE 1B" in v.upper()
                         for v in d.iloc[i].dropna())), d.shape[0])
    marks = []
    for i in range(fig1b):
        vals = [v for v in d.iloc[i].dropna() if isinstance(v, str)]
        joined = " ".join(vals).upper()
        if joined.startswith("AGE") and "WEEK" in joined:
            marks.append(i)
    boolchk("A2 four age blocks, located from the sheet not from a guess",
            len(marks) == 4,
            "AGE: markers at rows %s, Figure 1B begins at row %d" % (marks, fig1b))
    bounds = [(marks[k], marks[k + 1] if k + 1 < len(marks) else fig1b)
              for k in range(len(marks))]

    # ---- B. the published Figure 1B, reproduced from the n/N cells --------
    recon = {}
    for age, (r0, r1) in zip(AGES, bounds):
        for label, cols in (("LS", LS_COLS), ("MS", MS_COLS)):
            pcts, fracs = [], []
            for c in cols:
                for i in range(r0, r1):
                    v = d.iat[i, c]
                    if isinstance(v, str):
                        m = FRACTION.match(normalise(v))
                        if m:
                            k, n = int(m.group(1)), int(m.group(2))
                            pcts.append(100.0 * k / n)
                            fracs.append((i, c, k, n))
            if pcts:
                recon[(age, label)] = (float(np.mean(pcts)),
                                       float(np.std(pcts, ddof=1) / np.sqrt(len(pcts))),
                                       len(pcts), fracs)

    boolchk("B1 all eight (age, stimulus) cells reconstructed",
            len(recon) == 8, "%d of 8" % len(recon))
    for (age, label), (pm, ps, pn) in PUBLISHED.items():
        m, se, n, _ = recon.get((age, label), (np.nan, np.nan, 0, []))
        chk("B2 %2dw %s mean reproduces the published value" % (age, label), m, pm, 0.005)
        chk("B3 %2dw %s SEM reproduces the published value" % (age, label), se, ps, 0.0006)
        chk("B4 %2dw %s N is the number of trials" % (age, label), n, pn, 0)

    # ---- C. the per-fly cells do NOT reconcile with the per-trial row ----
    # This was written as an assertion and it fails, so it is reported as the
    # measurement it is. The recorded numerator equals the count of numeric
    # times in 63 of the 72 trial columns, and no single denominator rule fits
    # more than 32: at 4 weeks the denominator excludes flies that jumped or
    # failed to initiate (23 successes + 6 partial climbs = 29, and 29 is what
    # the sheet records), while at 10 weeks it is the full cohort of 36
    # regardless of how many flies produced no climb at all. The convention
    # changes within one file.
    #
    # The consequence is a limit on the benchmark, not a correction to it: this
    # dataset is usable at the PER-TRIAL level, where checks B, D, F and G
    # operate, and the per-fly cells cannot serve as an independent check of
    # the published percentages. Resolving it needs the authors; it is NOT
    # resolved here by selecting whichever rule happens to agree, which would
    # make the check circular -- the same discipline the grooming benchmark
    # applied to that dataset's genotype-coding contradiction.
    PARTIAL_ = PARTIAL
    EXCLUDED_ = NO_CLIMB | NOT_CLIMBING | UNTESTABLE
    # The block runs from the AGE: marker, so it contains the column headers
    # too. They are structure, not observations.
    HEADERS = ({"ROUND 1", "ROUND 2", "BEST SCORE LS", "BEST SCORE MS", "NO"}
               | {"LS%d" % i for i in range(1, 4)}
               | {"MS%d" % i for i in range(1, 7)}
               | {"SS%d" % i for i in range(1, 25)})
    num_ok = den_ok = total = 0
    unclassified = {}
    for age, (r0, r1) in zip(AGES, bounds):
        for label, cols in (("LS", LS_COLS), ("MS", MS_COLS)):
            _, _, _, fracs = recon[(age, label)]
            for (frow, c, k, n) in fracs:
                times = part = 0
                for i in range(r0, frow):
                    kind, _ = classify(d.iat[i, c])
                    if kind == "time":
                        times += 1
                    elif kind.startswith("fail:"):
                        code = kind[5:]
                        if code in PARTIAL_:
                            part += 1
                        elif code not in EXCLUDED_ and code not in HEADERS:
                            unclassified[code] = unclassified.get(code, 0) + 1
                total += 1
                num_ok += int(times == k)
                den_ok += int(times + part == n)
    boolchk("C1 the recorded numerator usually equals the count of timed "
            "climbs, but NOT in every column",
            total == 72 and 0.90 <= num_ok / total < 1.0,
            "%d of %d trial columns agree on the numerator (%.1f%%)"
            % (num_ok, total, 100.0 * num_ok / total))
    boolchk("C2 the denominator agrees less often still, so the per-fly layer "
            "cannot check the published percentages",
            den_ok < num_ok and den_ok / total < 0.80,
            "%d of %d agree under 'timed climbs + partial climbs' (%.1f%%); at "
            "4 weeks the sheet records 23/29 where 30 flies were tested and 6 "
            "climbed partway, while at 10 weeks it records /36 and /34 "
            "regardless -- the convention changes within one file"
            % (den_ok, total, 100.0 * den_ok / total))
    boolchk("C3 and a handful of cells carry annotations outside the documented "
            "code set, which must not be silently read as either outcome",
            len(unclassified) > 0,
            "undocumented cell contents: %s" % dict(sorted(unclassified.items())))

    # ---- D. typed failures, and what collapsing them would cost -----------
    codes = {}
    for age, (r0, r1) in zip(AGES, bounds):
        for c in LS_COLS + MS_COLS:
            for i in range(r0, r1):
                kind, _ = classify(d.iat[i, c])
                if kind.startswith("fail:"):
                    codes[kind[5:]] = codes.get(kind[5:], 0) + 1
    partial = sum(v for k, v in codes.items() if k in PARTIAL)
    noclimb = sum(v for k, v in codes.items() if k in NO_CLIMB)
    notclimb = sum(v for k, v in codes.items() if k in NOT_CLIMBING)
    untest = sum(v for k, v in codes.items() if k in UNTESTABLE)
    boolchk("D1 failures are TYPED in this dataset, not a single zero",
            partial > 0 and noclimb > 0 and notclimb > 0 and untest > 0,
            "partial-climb %d, no-climb %d, jumped/flew %d, untestable %d"
            % (partial, noclimb, notclimb, untest))
    boolchk("D2 the four region codes are all present, so partial climbs carry "
            "a distance",
            all(k in codes for k in ("I", "II", "III", "IV")),
            "I=%d II=%d III=%d IV=%d"
            % tuple(codes.get(k, 0) for k in ("I", "II", "III", "IV")))
    boolchk("D3 a fly that jumped instead of climbing is a DIFFERENT "
            "observation from one that failed to climb, and both exist",
            notclimb > 0 and noclimb > 0,
            "%d jumped or flew, %d did not initiate -- collapsing either to a "
            "time of 0 s or to a missing value changes the endpoint"
            % (notclimb, noclimb))
    boolchk("D4 a fly that could not be tested is neither: it is censored",
            untest > 0,
            "%d cells coded DEAD or LEG DEFECT; scoring these as failures would "
            "attribute a husbandry loss to the phenotype" % untest)

    # ---- E. the published SEM is over TRIALS, not over flies -------------
    # Legitimate, and stated in the sheet itself -- the unit is the stimulus
    # trial. But an SEM over trials describes how reproducible the stimulus is,
    # not how variable flies are, and the two are not interchangeable. This
    # project treats conflating them as a defect (principle 5), so the
    # distinction is pinned here rather than inferred by a later reader of the
    # table. The fly count comes from the sheet's own numbering column.
    fly_counts = {}
    for age, (r0, r1) in zip(AGES, bounds):
        nums = [d.iat[i, 3] for i in range(r0, r1)]
        nums = [v for v in nums if isinstance(v, (int, float)) and not pd.isna(v)]
        fly_counts[age] = int(max(nums)) if nums else 0
    boolchk("E1 the number of FLIES per age is read from the sheet and is not "
            "the published N",
            all(v >= 25 for v in fly_counts.values())
            and all(v != PUBLISHED[(a, "LS")][2] for a, v in fly_counts.items()),
            "flies per age %s, against published N = 6 (LS) and 12 (MS) trials"
            % fly_counts)
    boolchk("E2 so each published mean rests on 6 or 12 repeated measures of "
            "the same 26-36 flies",
            min(fly_counts.values()) >= 26,
            "4w n=%d, 6w n=%d, 8w n=%d, 10w n=%d flies"
            % tuple(fly_counts[a] for a in AGES))

    # ---- F. the Greek-letter trap -----------------------------------------
    raw_codes = {}
    for age, (r0, r1) in zip(AGES, bounds):
        for c in LS_COLS + MS_COLS:
            for i in range(r0, r1):
                v = d.iat[i, c]
                if isinstance(v, str) and not FRACTION.match(normalise(v)):
                    try:
                        float(v)
                    except ValueError:
                        raw_codes[v.strip().upper()] = raw_codes.get(v.strip().upper(), 0) + 1
    greek_used = [k for k in raw_codes if any(g in k for g in GREEK)]
    boolchk("F1 the file really does mix Greek and Latin letters in the codes",
            len(greek_used) > 0,
            "%d distinct codes contain a Greek look-alike, e.g. %s"
            % (len(greek_used), sorted(greek_used)[:4]))
    boolchk("F2 unnormalised, region I would be counted as two different codes",
            len(raw_codes) > len(codes),
            "%d raw codes collapse to %d after normalising"
            % (len(raw_codes), len(codes)))

    # ---- G. the biology: stable in mid life, collapsing in late life ------
    ls = [recon[(a, "LS")][0] for a in AGES]
    ms = [recon[(a, "MS")][0] for a in AGES]
    boolchk("G1 climbing success declines with age in both stimulus sets",
            ls[0] > ls[-1] and ms[0] > ms[-1],
            "LS %s, MS %s" % ([round(x, 1) for x in ls], [round(x, 1) for x in ms]))
    # the paper's own claim: stable in MID life, decreasing in LATE life
    mid_drop = ls[0] - ls[2]          # 4 -> 8 weeks
    late_drop = ls[2] - ls[3]         # 8 -> 10 weeks
    boolchk("G2 the decline is NOT linear in age: it collapses between 8 and "
            "10 weeks, as the paper states",
            late_drop > 3.0 * mid_drop,
            "LS falls %.1f points over 4->8 weeks and %.1f over 8->10"
            % (mid_drop, late_drop))
    boolchk("G3 so a linear-in-age motor-decline rate would misdescribe this "
            "cohort, and a model must be able to say so",
            late_drop / max(mid_drop, 1e-9) > 4.0,
            "late:mid slope ratio %.1f -- fitting one slope reports neither"
            % (late_drop / max(mid_drop, 1e-9)))
    boolchk("G4 MS is not monotone, so the two stimulus sets are not "
            "interchangeable read-outs",
            not (ms[0] > ms[1] > ms[2] > ms[3]),
            "MS %s -- 8 weeks is above 6 weeks" % [round(x, 1) for x in ms])

    # ---- H. what this does NOT establish ----------------------------------
    boolchk("H1 this is hand-scored data: nothing here validates DroPhenix's "
            "own fly detection", True,
            "L2 and L3 only; the detector benchmark in plan 2.1 remains open")

    print("validate_climbing_thiem -- real published escape-climbing records")
    print("-" * 78)
    npass = 0
    for ok, name, got in results:
        # The file's failure codes contain Greek letters, and a Windows console
        # is cp1252, which cannot encode them. Escaping rather than crashing
        # keeps the whole table readable -- and the escape itself shows the
        # reader that the character is not the Latin letter it resembles.
        line = "[%s] %s  --  %s" % ("PASS" if ok else "FAIL", name, got)
        print(line.encode("ascii", "backslashreplace").decode("ascii"))
        npass += ok
    print("-" * 78)
    print("%d/%d checks passed" % (npass, len(results)))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
