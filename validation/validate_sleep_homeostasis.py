"""
validate_sleep_homeostasis.py
The DAM/sleep path against REAL published sleep recordings.

ESTABLISHES: L2 (arithmetic on real data) and L3 (direction of a known
biological effect). It does NOT establish L4 for the >=5-min bout rule, and
check I states why in the suite's own output.

WHY THIS SUITE EXISTS
---------------------
`docs/SCIENTIFIC_VALIDATION_PLAN.md` section 2.4 names this dataset as
unblocked and ready to wire in, and `docs/REFERENCE_DATASETS.md` section 2a
records a characterisation of it. Neither had a suite. Until one exists the
characterisation is a note in a document, which is exactly the class of claim
this project's governance rules say may not stand on its own.

Everything the suite asserts about the files is re-derived here from the files.
Nothing is read across from the document, so the document can be checked rather
than trusted.

DATA (real, published, CC0)
    Jagannath A, Sengupta T, et al. (2024)
    "Homeostatic control of deep sleep and molecular correlates of sleep
     pressure in Drosophila"
    eLife 13:RP91355, doi:10.7554/eLife.91355
    Data: doi:10.5061/dryad.qnk98sfpp, mirrored at Zenodo record 10069769
    Local: reference_datasets/sleep_homeostasis/dataset.zip  (35 .txt files)

WHAT THE DATASET IS, AND THE TWO KINDS OF FILE IN IT
    Column 1 is Zeitgeber time; every other column is one fly; cells are
    MINUTES OF SLEEP PER BIN -- already-scored sleep, not raw beam counts.

    Figure1/*            DAM-style, mechanical deprivation, 3 days.
                         6 h and 12 h protocols: 60-min bins, 24 bins/day.
                         24 h protocol:          30-min bins, 48 bins/day.
    Figures 2 to 4/*,    Ethoscope, 3 or 5 cycles, and -- the part that makes
    Figure5/*            them valuable -- the SAME flies scored under BOTH a
                         5-min and a 25-min sleep definition.

    Bin width is read per file from two independent signals that must agree:
    the increment of the ZT column, and the maximum cell value (minutes of
    sleep in a bin cannot exceed the bin width). Day index comes from the ZT
    WRAP, never the row index.

THE GROUND TRUTH, AND WHY IT IS NOT CIRCULAR
    Day 1 = baseline, day 2 = deprivation, day 3 = recovery. This is not
    assumed from the README: it is visible in the data, because sleep collapses
    on day 2 in every deprived group and in no control group, and the fly
    counts per file (44, 62, 32, 64, 32, 32) match the n reported in the
    paper's Figure 1. Homeostatic rebound -- more sleep on recovery than at
    that group's own baseline, and more than contemporaneous controls gain --
    is then a DIRECTIONAL test that the summary and comparison path must pass.

TWO THINGS THIS DATASET CANNOT DO, STATED SO THEY ARE NOT CLAIMED
    1. It cannot validate the >=5-min bout rule. The sleep is already scored;
       the rule needs raw beam counts. Check I pins this as a stated gap.
    2. The Figure1 files are NOT an absolute sleep-amount anchor: their
       undisturbed controls sit at 1,061-1,278 min/day median, above every
       published DAM total-sleep value. Check F measures that, and measures
       that the ETHOSCOPE 5-min-definition files ARE such an anchor -- which
       corrects `docs/REFERENCE_DATASETS.md`, where the caution is written as
       if it applied to the whole dataset.

Run:  python validation/validate_sleep_homeostasis.py   (exit 0 = all passed)
"""

import os
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import dams as D            # noqa: E402
from modules import dam_reader as DR     # noqa: E402

ZIP = (Path(__file__).resolve().parent.parent / "reference_datasets"
       / "sleep_homeostasis" / "dataset.zip")

# Published n per file, from the paper's Figure 1. Used only as a corroboration
# of the group mapping -- the mapping itself is established from the data.
PUBLISHED_N = {
    "Figure1/6h deprivation/Control_20sec_SleepData.txt": 44,
    "Figure1/6h deprivation/Dep_20sec_SleepData.txt": 62,
    "Figure1/12h deprivation/Control_20sec_SleepData.txt": 32,
    "Figure1/12h deprivation/Dep_20sec_SleepData.txt": 64,
    "Figure1/24h deprivation/Control_20sec_SleepData.txt": 32,
    "Figure1/24h deprivation/Dep_20sec_SleepData.txt": 32,
}


# ---------------------------------------------------------------- reading ----
def extract(dest):
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            if n.startswith("__MACOSX") or n.endswith(".DS_Store") or n.endswith("/"):
                continue
            t = dest / n
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(z.read(n))


def read_positional(path):
    """
    Read a file keying flies by POSITION.

    Column labels REPEAT within a file. `pandas` does not collide them -- it
    silently invents `I1.1`, `I2.1` -- which is worse than a collision, because
    the frame then looks well-formed and any join on the fly label matches the
    wrong cohort. Check B pins this.
    """
    raw = pd.read_csv(path, sep="\t", header=0)
    zt = raw.iloc[:, 0].to_numpy(float)
    vals = raw.iloc[:, 1:].to_numpy(float)
    return zt, vals


def raw_header(path):
    line = open(path, encoding="utf-8", errors="replace").readline()
    return line.rstrip("\n").split("\t")[1:]


def bin_width(zt, vals):
    """Two independent readings of the bin width, which must agree."""
    d = np.diff(zt)
    from_increment = float(np.median(d[d > 0]) * 60.0)
    from_max_cell = float(np.nanmax(vals))   # a bin cannot hold more sleep than its width
    return from_increment, from_max_cell


def day_index(zt):
    """Day from the ZT wrap. The row index is not the day: bins per day differ."""
    return (np.diff(zt, prepend=zt[0]) < 0).cumsum() + 1


def per_fly_per_day(zt, vals):
    """Minutes of sleep per fly per day. Flies are columns, keyed by position."""
    day = day_index(zt)
    out = {}
    for d in range(1, int(day.max()) + 1):
        out[d] = np.nansum(vals[day == d, :], axis=0)
    return out


# ------------------------------------------------------------------ runner ----
def main():
    results = []

    def chk(name, got, expected, tol):
        ok = np.isfinite(got) and abs(float(got) - float(expected)) <= tol
        results.append((ok, name, "got %.4g, expected %.4g +/- %g" % (got, expected, tol)))

    def boolchk(name, ok, got):
        results.append((bool(ok), name, got))

    def report(frame, **kw):
        """
        Call `plausibility_report`, turning a missing parameter into a FAILING
        check rather than a crash. Checks G3-G5 probe for an interface that did
        not exist when this suite was written; a traceback there would hide
        every check after it, which is the failure mode the register's N71 is.
        """
        try:
            return DR.plausibility_report(frame, **kw)
        except TypeError as exc:
            return {"ok": None, "flags": ["plausibility_report refused the call: %s" % exc]}

    if not ZIP.exists():
        print("[SKIP] dataset not present: %s" % ZIP)
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        extract(root)
        files = sorted(p for p in root.rglob("*SleepData*.txt"))
        rel = {str(p.relative_to(root)).replace(os.sep, "/"): p for p in files}
        boolchk("A0 all 33 sleep files extracted", len(files) == 33, "%d files" % len(files))

        # ---- A. bin width is readable per file, from two signals that agree --
        widths = {}
        agree = 0
        for k, p in rel.items():
            zt, vals = read_positional(p)
            a, b = bin_width(zt, vals)
            widths[k] = a
            if abs(a - b) < 1e-9:
                agree += 1
        boolchk("A1 bin width agrees between ZT increment and max cell, every file",
                agree == len(rel), "%d/%d files agree" % (agree, len(rel)))
        boolchk("A2 the dataset does NOT share one bin width",
                len(set(widths.values())) > 1,
                "widths present: %s" % sorted(set(widths.values())))
        chk("A3 6 h protocol is 60-min bins",
            widths["Figure1/6h deprivation/Control_20sec_SleepData.txt"], 60.0, 0)
        chk("A4 12 h protocol is 60-min bins",
            widths["Figure1/12h deprivation/Control_20sec_SleepData.txt"], 60.0, 0)
        chk("A5 24 h protocol is 30-min bins",
            widths["Figure1/24h deprivation/Control_20sec_SleepData.txt"], 30.0, 0)
        # bins/day follows from the width, and is NOT the row count
        zt, vals = read_positional(rel["Figure1/24h deprivation/Control_20sec_SleepData.txt"])
        chk("A6 24 h file has 48 bins/day, not 144 rows of one day",
            int((day_index(zt) == 1).sum()), 48, 0)
        chk("A7 ... and exactly 3 days", int(day_index(zt).max()), 3, 0)

        # ---- B. identifier reuse: pandas invents labels, it does not collide --
        dup_files, invented = 0, 0
        for k, p in rel.items():
            hdr = raw_header(p)
            if len(hdr) != len(set(hdr)):
                dup_files += 1
                cols = list(pd.read_csv(p, sep="\t", nrows=0).columns[1:])
                invented += sum("." in c for c in cols)
        boolchk("B1 column labels repeat within files (the identifier-reuse trap)",
                dup_files == 7,
                "%d of %d files have duplicate fly labels" % (dup_files, len(rel)))
        boolchk("B2 pandas SILENTLY RENAMES rather than colliding, so a name-keyed "
                "read looks well-formed", invented == 196,
                "%d labels invented across those files (e.g. I1.1)" % invented)
        p = rel["Figure1/6h deprivation/Control_20sec_SleepData.txt"]
        hdr = raw_header(p)
        boolchk("B3 the first repeated label denotes two different flies",
                hdr.count("I1") == 2, "label I1 appears %d times" % hdr.count("I1"))

        # ---- C. group mapping corroborated by the paper's own n --------------
        nmatch = 0
        for k, want in PUBLISHED_N.items():
            v = read_positional(rel[k])[1]
            nmatch += int(v.shape[1] == want)
        boolchk("C1 fly counts match the n reported in the paper's Figure 1",
                nmatch == len(PUBLISHED_N),
                "%d/%d files match" % (nmatch, len(PUBLISHED_N)))

        # ---- D. the biology: deprivation then rebound, direction only --------
        dep_drop, ctl_flat = [], []
        for k in rel:
            if "Figure1" not in k:
                continue
            zt, vals = read_positional(rel[k])
            pfd = per_fly_per_day(zt, vals)
            base, dep, rec = (float(np.nanmean(pfd[1])), float(np.nanmean(pfd[2])),
                              float(np.nanmean(pfd[3])))
            (dep_drop if "/Dep" in k else ctl_flat).append((k, base, dep, rec))
        boolchk("D1 every deprived file loses sleep on day 2 vs its own baseline",
                all(d < b for _, b, d, _ in dep_drop),
                "%d/%d files; worst retention %.2f of baseline"
                % (sum(d < b for _, b, d, _ in dep_drop), len(dep_drop),
                   min(d / b for _, b, d, _ in dep_drop)))
        boolchk("D2 no control file loses sleep on day 2",
                all(d >= b for _, b, d, _ in ctl_flat),
                "%d/%d controls flat or up"
                % (sum(d >= b for _, b, d, _ in ctl_flat), len(ctl_flat)))
        boolchk("D3 every deprived file rebounds above its own baseline on day 3",
                all(r > b for _, b, _, r in dep_drop),
                "%d/%d files rebound"
                % (sum(r > b for _, b, _, r in dep_drop), len(dep_drop)))
        dep_reb = float(np.mean([r - b for _, b, _, r in dep_drop]))
        ctl_reb = float(np.mean([r - b for _, b, _, r in ctl_flat]))
        boolchk("D4 rebound is larger in deprived than in control flies",
                dep_reb > ctl_reb,
                "deprived +%.1f vs control +%.1f min on day 3" % (dep_reb, ctl_reb))
        # and DroPhenix's own comparison path must find it, on real per-fly data
        zt, v_dep = read_positional(rel["Figure1/24h deprivation/Dep_20sec_SleepData.txt"])
        pdep = per_fly_per_day(zt, v_dep)
        zt, v_ctl = read_positional(rel["Figure1/24h deprivation/Control_20sec_SleepData.txt"])
        pctl = per_fly_per_day(zt, v_ctl)
        frame = pd.concat([
            pd.DataFrame({"Genotype": "CS", "Sex": "Mixed",
                          "FlyID": ["D%d" % i for i in range(len(pdep[2]))],
                          "Treatment": "Deprived",
                          "Total_sleep_min_per_day": pdep[2]}),
            pd.DataFrame({"Genotype": "CS", "Sex": "Mixed",
                          "FlyID": ["C%d" % i for i in range(len(pctl[2]))],
                          "Treatment": "Control",
                          "Total_sleep_min_per_day": pctl[2]}),
        ], ignore_index=True)
        cmp_dep = D.compare_groups(frame, "Total_sleep_min_per_day", group_col="Treatment")
        main = cmp_dep.get("main_result", {})
        boolchk("D5 dams.compare_groups detects the day-2 deprivation on real flies",
                float(main.get("p_value", 1.0)) < 1e-6,
                "%s, p = %s, 32 deprived vs 32 control flies"
                % (cmp_dep.get("method"), main.get("p_value")))

        # ---- E. paired 5-min / 25-min definitions, the same flies ------------
        pairs = [k for k in rel if "25minDef" in k]
        cellviol = 0
        ratios = []
        for k in pairs:
            a = read_positional(rel[k])[1]
            b = read_positional(rel[k.replace("25minDef", "5minDef")])[1]
            cellviol += int(np.nansum(a > b + 1e-9))
            ratios.append(float(np.nansum(a) / np.nansum(b)))
        boolchk("E1 six paired files present (same flies, two sleep definitions)",
                len(pairs) == 6, "%d pairs" % len(pairs))
        boolchk("E2 a LONGER bout criterion never scores MORE sleep, in any cell",
                cellviol == 0, "%d violating cells across all pairs" % cellviol)
        boolchk("E3 the 25-min definition recovers 36-49% of 5-min-rule sleep",
                0.30 < min(ratios) and max(ratios) < 0.55,
                "ratios %.3f-%.3f" % (min(ratios), max(ratios)))

        # ---- F. which files are an absolute anchor, and which are not --------
        def per_fly_day_rates(k):
            """Minutes of sleep per fly per OBSERVED day, one value per fly."""
            zt, v = read_positional(rel[k])
            nd = int(day_index(zt).max())
            return np.nansum(v, axis=0) / nd

        def median_per_day(k):
            return float(np.median(per_fly_day_rates(k)))

        lo, hi = DR.PLAUSIBLE_SLEEP_MIN_PER_DAY
        # "5minDef" is a substring of "25minDef". Matching without the
        # separator silently pools the two definitions, which is how the first
        # version of this check reported 275 min/day as a 5-min-rule value.
        eth = [k for k in rel if "UNDIST" in k and "_5minDef" in k]
        eth_med = [median_per_day(k) for k in eth]
        boolchk("F1 ethoscope 5-min-rule undisturbed flies ARE a usable absolute anchor",
                all(lo <= m <= hi for m in eth_med),
                "medians %s inside %.0f-%.0f"
                % (["%.0f" % m for m in eth_med], lo, hi))
        fig1_ctl = [k for k in rel if "Figure1" in k and "/Control" in k]
        f1_pool = np.concatenate([per_fly_day_rates(k) for k in fig1_ctl])
        eth_pool = np.concatenate([per_fly_day_rates(k) for k in eth])
        f1_out = float(np.mean(f1_pool > hi))
        boolchk("F2 Figure1 undisturbed flies are NOT such an anchor",
                np.median(f1_pool) > 1100.0 and f1_out > 0.40,
                "pooled median %.1f min/day (%.1f h) over %d flies, %.1f%% above "
                "the %.0f ceiling" % (np.median(f1_pool), np.median(f1_pool) / 60.0,
                                      len(f1_pool), 100 * f1_out, hi))
        boolchk("F3 ... while every ethoscope 5-min-rule fly is inside it",
                float(np.mean((eth_pool >= lo) & (eth_pool <= hi))) == 1.0,
                "%d/%d flies inside, pooled median %.1f min/day"
                % (int(np.sum((eth_pool >= lo) & (eth_pool <= hi))), len(eth_pool),
                   np.median(eth_pool)))

        # ---- G. the published range is DEFINITION-SPECIFIC -------------------
        # The same healthy, undisturbed Canton-S flies, scored under the 25-min
        # deep-sleep definition this paper uses, sit below the lower bound of a
        # range published for the >=5-min rule. `sleep_per_fly` takes
        # `sleep_min` as a parameter, so the platform invites exactly this.
        k25 = "Figures 2 to 4/CS_UNDIST_SleepData_25minDef_220secTrigger.txt"
        zt, v = read_positional(rel[k25])
        nd = int(day_index(zt).max())
        per = np.nansum(v, axis=0) / nd
        pf25 = pd.DataFrame({"Total_sleep_min_per_day": per})
        n_out = int(np.sum((per < lo) | (per > hi)))
        boolchk("G1 a 25-min-definition table of HEALTHY flies falls outside the "
                "5-min-rule range", n_out > 0,
                "%d/%d undisturbed CS flies outside %.0f-%.0f" % (n_out, len(per), lo, hi))
        # What the undeclared call actually does is subtler than a false alarm,
        # and worse. The median of this table (356 min/day) sits just inside the
        # 5-min-rule range, so the gate raises nothing at all -- while 38% of
        # these flies are outside it and the range does not apply to them in the
        # first place. The criterion is NOT recoverable from the data, so no
        # amount of checking can detect it; the only fix available is that the
        # report must state the criterion it assumed, so that an undeclared call
        # is visible in the output instead of invisible.
        rep_naive = report(pf25)
        boolchk("G2 an undeclared call assumes the 5-min rule, and now SAYS so "
                "rather than judging silently",
                float(rep_naive.get("sleep_definition_min", -1)) == 5.0
                and bool(rep_naive.get("sleep_checked")) is True,
                "sleep_definition_min = %s, sleep_checked = %s, flags = %s "
                "(median %.0f sits inside the range, so nothing was raised)"
                % (rep_naive.get("sleep_definition_min"), rep_naive.get("sleep_checked"),
                   rep_naive.get("flags"), rep_naive.get("median_sleep_min_per_day")))
        rep25 = report(pf25, sleep_min=25.0)
        boolchk("G3 told the definition, it must NOT call healthy flies implausible",
                bool(rep25.get("ok")) is True,
                "ok = %s; flags = %s" % (rep25.get("ok"), rep25.get("flags")))
        boolchk("G4 ... and must report which definition the range applies to",
                float(rep25.get("sleep_definition_min", -1)) == 25.0,
                "sleep_definition_min = %s" % rep25.get("sleep_definition_min"))
        boolchk("G5 the default is still the published 5-min rule, unchanged",
                float(report(pd.DataFrame(
                    {"Total_sleep_min_per_day": [700.0, 720.0, 690.0]})
                ).get("sleep_definition_min", -1)) == 5.0,
                "default sleep_definition_min = %s"
                % report(pd.DataFrame({"Total_sleep_min_per_day": [700.0]})
                         ).get("sleep_definition_min"))

        # ---- H. a table whose sleep was WITHHELD must not pass silently ------
        # Block D (N81) makes `sleep_summary` return NaN sleep where the bin
        # width cannot resolve the criterion. Both Figure1 bin widths (60 and
        # 30 min) exceed the 5-min rule, so every Figure1 file is such a case.
        withheld = pd.DataFrame({"Total_sleep_min_per_day": [np.nan, np.nan, np.nan],
                                 "Total_activity_per_day": [900.0, 870.0, 910.0]})
        repw = report(withheld)
        boolchk("H1 plausibility_report does not report OK on withheld sleep",
                bool(repw.get("ok")) is False,
                "ok = %s; flags = %s" % (repw.get("ok"), repw.get("flags")))
        boolchk("H2 ... and says so rather than omitting the sleep line",
                any(("withheld" in f.lower() or "no sleep" in f.lower()
                     or "not scoreable" in f.lower()) for f in repw.get("flags", [])),
                "flags = %s" % repw.get("flags"))
        boolchk("H3 both Figure1 bin widths exceed the 5-min rule, so DroPhenix "
                "would withhold sleep on every Figure1 file",
                all(w > D.SLEEP_MIN_DEFAULT for k, w in widths.items() if "Figure1" in k),
                "Figure1 widths %s vs rule %s min"
                % (sorted({w for k, w in widths.items() if "Figure1" in k}),
                   D.SLEEP_MIN_DEFAULT))

        # ---- I. stated gap ---------------------------------------------------
        boolchk("I1 the dataset is already-scored sleep, so the >=5-min rule is "
                "NOT validated here", True,
                "stated gap: needs raw beam counts (plan 2.4 / ref doc 2b)")

    print("validate_sleep_homeostasis -- real published sleep-deprivation data")
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
