"""
validate_dam_raw_microbiome.py
The >=5-min sleep rule, the light channel and the clustering, against REAL RAW
Trikinetics beam counts.

ESTABLISHES
    L2 -- the >=5-min inactivity rule reproduces an independent implementation
          of the published definition on 379 real flies, exactly.
    L3 -- respecting the experiment's own clustering reproduces the paper's
          NEGATIVE conclusion, where a per-fly test contradicts it.
    It does NOT establish L4: no annotated ground truth for sleep exists here,
    and check F measures a method difference this dataset cannot adjudicate.

WHY THIS SUITE EXISTS
---------------------
`docs/SCIENTIFIC_VALIDATION_PLAN.md` section 2.4 and `docs/REFERENCE_DATASETS.md`
section 2b both name the same single remaining gap: every sleep dataset in the
repository was ALREADY-SCORED sleep, so the >=5-min bout rule -- the definition
on which every sleep number in this platform rests -- had no validation at all
beyond its own unit tests. Closing that needs raw beam counts. This is them.

DATA (real, published, CC-BY 4.0)
    Selkrig J, Mohammad F, Ng SH, et al. (2018)
    "The Drosophila microbiome has a limited influence on sleep, activity, and
     courtship behaviors"
    Sci Rep 8:10646, doi:10.1038/s41598-018-28764-5
    Data: doi:10.5281/zenodo.3403481 (Zenodo record 3403481)
    Local: reference_datasets/dam_microbiome/RawData.zip

    `RawData/Figure5/Monitor{26,27,36,37,46,47,81,82,83,84,85,86}.txt` are
    twelve raw DAMSystem monitor files: 10,030 one-minute rows each, 32
    channels, 14-21 September 2017, under 12:12 LD. 384 channels in total, of
    which the dataset's own channel map (`Joel Selkrig Monitor List.xlsx`)
    assigns 379 to one of three microbiome states -- germ-free, germ-free
    conventionalised, and conventionally reared.

WHY THIS PAPER IS AN UNUSUALLY GOOD BENCHMARK
    Its result is NEGATIVE: the microbiome has a *limited* influence on sleep.
    Almost every benchmark asks whether a tool can find a difference, which
    rewards a tool that finds differences whether or not they are there. This
    one asks the harder question. And the design makes it sharp, because flies
    share monitors and monitors are nested in replicate cohorts, so a per-fly
    test is pseudoreplicated -- which is this project's own principle 5, "never
    treat technical replicates as biological replicates", on real data where
    the published conclusion says which answer is right.

Run:  python validation/validate_dam_raw_microbiome.py   (exit 0 = all passed)
"""

import re
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import dam_reader as DR      # noqa: E402
from modules import dams as D             # noqa: E402
from modules import design_schema as DSC   # noqa: E402
from modules import hierarchical as H      # noqa: E402

ZIP = (Path(__file__).resolve().parent.parent / "reference_datasets"
       / "dam_microbiome" / "RawData.zip")
SUBDIR = "RawData/Figure5"
XLSX = "Joel Selkrig Monitor List.xlsx"


# --------------------------------------------------------------- the rule ----
def published_sleep_minutes(activity, sleep_min=5.0, bin_min=1.0):
    """
    Minutes of sleep by the published definition, written from the definition.

    Hendricks et al. 2000 (Neuron 25:129) and Shaw et al. 2000 (Science
    287:1834): a fly is asleep during any run of at least five consecutive
    minutes with no beam crossing, and every minute of such a run is sleep.

    This function exists so that the module is compared against the DEFINITION
    rather than against itself. It deliberately shares no code with
    `modules.dams`. NaN breaks a run, because a minute the monitor did not read
    is not a minute of stillness -- scoring it as sleep is the single most
    consequential way to be wrong here, and it is the error `load_dams` used to
    make by zero-filling.
    """
    a = np.asarray(activity, dtype=float)
    need = int(np.ceil(sleep_min / bin_min))
    is_zero = (a == 0.0)              # NaN == 0.0 is False, so NaN ends a run
    total = 0
    i = 0
    n = len(a)
    while i < n:
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < n and is_zero[j]:
            j += 1
        if (j - i) >= need:
            total += (j - i)
        i = j
    return float(total) * bin_min


def icc_oneway(values, clusters):
    """
    One-way ANOVA intraclass correlation for a CONTINUOUS outcome.

    `hierarchical.estimate_icc` is the Ridout/Demetrio/Firth estimator for
    clustered BINARY data and does not apply to minutes of sleep, so the
    standard one-way random-effects estimator is used here and handed to the
    module's own `design_effect`:

        ICC = (MSB - MSW) / (MSB + (m0 - 1) * MSW)

    with m0 the mean cluster size corrected for unequal sizes.
    """
    df = pd.DataFrame({"y": np.asarray(values, float), "g": np.asarray(clusters)}).dropna()
    k = df["g"].nunique()
    n = len(df)
    if k < 2 or n <= k:
        return np.nan, np.nan
    sizes = df.groupby("g").size().to_numpy(float)
    m0 = (n - np.sum(sizes ** 2) / n) / (k - 1)
    gm = df["y"].mean()
    means = df.groupby("g")["y"].mean()
    msb = float(np.sum(sizes * (means.to_numpy(float) - gm) ** 2) / (k - 1))
    msw = float(np.sum([((df.loc[df.g == g, "y"] - means[g]) ** 2).sum()
                        for g in means.index]) / (n - k))
    denom = msb + (m0 - 1.0) * msw
    icc = (msb - msw) / denom if denom > 0 else np.nan
    return float(icc), float(np.mean(sizes))


# --------------------------------------------------------------- loading -----
def parse_channel_map(xlsx_path):
    """
    The dataset's own channel map.

    Three group blocks are laid out side by side in one sheet with hand-written
    channel ranges -- 'Ch1 - Ch25', 'Ch 1 - Ch15', 'Ch29 - 32', 'Ch1 -Ch32'.
    Rows 4-12 are the per-replicate listing; rows 17 onward repeat the same
    assignments grouped by genotype, so only the first listing is read and no
    channel can be claimed twice.
    """
    x = pd.read_excel(xlsx_path, header=None)
    out = []
    for r in range(4, 13):
        for c0, label in ((1, "GF"), (7, "GF_conventionalised"), (13, "SPF")):
            fly, mon, ch = x.iat[r, c0], x.iat[r, c0 + 1], x.iat[r, c0 + 2]
            if not isinstance(mon, str) or not isinstance(ch, str):
                continue
            nums = [int(v) for v in re.findall(r"(\d+)", ch)]
            if len(nums) != 2:
                continue
            for k in range(min(nums), max(nums) + 1):
                out.append({"Monitor": mon.strip(), "Channel": k,
                            "Group": label, "Replicate": str(fly).strip()})
    return pd.DataFrame(out).drop_duplicates(["Monitor", "Channel"])


def to_long(frame):
    f = frame.copy()
    f["Genotype"] = f["Group"]
    f["Treatment"] = "None"
    f["Sex"] = "Unknown"
    f["FlyID"] = f["Monitor"] + "_Ch" + f["Channel"].astype(str)
    t0 = f["Timestamp"].min()
    f["Time_min"] = (f["Timestamp"] - t0).dt.total_seconds() / 60.0
    return f


# ------------------------------------------------------------------ runner ---
def main():
    results = []

    def chk(name, got, expected, tol):
        ok = bool(np.isfinite(got)) and abs(float(got) - float(expected)) <= tol
        results.append((ok, name, "got %.6g, expected %.6g +/- %g" % (got, expected, tol)))

    def boolchk(name, ok, got):
        results.append((bool(ok), name, got))

    if not ZIP.exists():
        print("[SKIP] dataset not present: %s" % ZIP)
        print("       Fetch with: curl -sL -o %s \\" % ZIP)
        print("         https://zenodo.org/api/records/3403481/files/RawData.zip/content")
        print("       This suite SKIPPED and therefore established nothing.")
        return 0

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with zipfile.ZipFile(ZIP) as z:
            for n in z.namelist():
                if n.startswith("__MACOSX") or n.endswith("/"):
                    continue
                t = root / n
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(z.read(n))
        base = root / SUBDIR
        mon_files = sorted(base.glob("Monitor*.txt"))

        # ---- A. the raw files parse, and the format is what it claims -------
        boolchk("A1 twelve raw monitor files present", len(mon_files) == 12,
                "%d files" % len(mon_files))
        frames = []
        for p in mon_files:
            mon = "M" + re.findall(r"(\d+)", p.name)[0]
            frames.append(DR.read_monitor(str(p), monitor=mon))
        raw = pd.concat(frames, ignore_index=True)
        n_ch = raw.groupby(["Monitor", "Channel"]).ngroups
        chk("A2 12 monitors x 32 channels", n_ch, 384, 0)
        one = frames[0]
        ts = one.drop_duplicates("Timestamp")["Timestamp"].sort_values()
        step = ts.diff().dropna().mode().iloc[0]
        boolchk("A3 one-minute bins", step == pd.Timedelta("1min"), str(step))
        chk("A4 10,030 minutes per monitor", one["Timestamp"].nunique(), 10030, 0)
        # Status is NOT uniformly OK across the twelve files, and that is the
        # more valuable check: a bad-status row must reach the analysis as
        # MISSING, never as an hour of perfect stillness. Zero-filling those
        # rows is the defect `load_dams` used to have, and on this data it
        # would have been scored as sleep.
        bad = raw[raw["Status"] != 1]
        boolchk("A5 rows with a bad monitor status exist in this real recording",
                len(bad) > 0,
                "%d of %d rows, status values %s"
                % (len(bad), len(raw), sorted(set(pd.unique(raw["Status"])))))
        boolchk("A6 ... and every one of them is NaN activity, never zero",
                bool(bad["Activity"].isna().all()),
                "%d bad-status rows, %d NaN, %d zero"
                % (len(bad), int(bad["Activity"].isna().sum()),
                   int((bad["Activity"] == 0).sum())))
        boolchk("A7 good-status rows are never silently NaN",
                float(raw.loc[raw["Status"] == 1, "Activity"].isna().mean()) == 0.0,
                "NaN fraction among OK rows = %.6f"
                % raw.loc[raw["Status"] == 1, "Activity"].isna().mean())
        boolchk("A8 N80 is fixed on real data: a file with no restart is session 1",
                sorted(set(pd.unique(one["Session"]))) == [1],
                "sessions %s" % sorted(set(pd.unique(one["Session"]))))

        # ---- B. the light channel, on a real 12:12 LD record ---------------
        # `lights_on_offset` reads the Light channel off a long frame, so it
        # needs the schema columns the analysis module works in.
        one_long = to_long(one.assign(Group="GF", Replicate="R1"))
        off = D.lights_on_offset(one_long)
        ph = D.photoperiod_minutes(one_long)
        lit = one.drop_duplicates("Timestamp").set_index("Timestamp")["Light"]
        v = lit.to_numpy(float)
        tr = np.where((v[1:] > 0.5) & (v[:-1] <= 0.5))[0]
        first_on = lit.index[tr[0] + 1]
        # the derived offset must land on the record's own first lights-on
        pred = (lit.index[0] + pd.Timedelta(minutes=float(off))
                if off is not None else pd.NaT)
        boolchk("B1 lights_on_offset lands on the record's real lights-on",
                off is not None and pred == first_on,
                "offset %.0f min from %s -> %s; first 0->1 at %s"
                % (off if off is not None else float("nan"),
                   lit.index[0], pred, first_on))
        chk("B2 photoperiod is 12 h to within one bin", ph, 720.0, 1.0)
        boolchk("B3 lights-on is 06:00, as a 12:12 LD protocol implies",
                first_on.hour == 6, str(first_on.time()))

        # ---- C. the channel map, and what it excludes ----------------------
        cmap = parse_channel_map(base / XLSX)
        chk("C1 the map assigns 379 channels", len(cmap), 379, 0)
        boolchk("C2 no channel is claimed by two groups",
                not cmap.duplicated(["Monitor", "Channel"]).any(), "none")
        lab = raw.merge(cmap, on=["Monitor", "Channel"], how="inner")
        n_lab = lab.groupby(["Monitor", "Channel"]).ngroups
        chk("C3 five channels are unassigned and must not be analysed",
            n_ch - n_lab, 5, 0)
        boolchk("C4 three microbiome states, sizes from the map",
                cmap.groupby("Group").size().to_dict()
                == {"GF": 121, "GF_conventionalised": 93, "SPF": 165},
                str(cmap.groupby("Group").size().to_dict()))

        long = to_long(lab)
        pf = D.sleep_per_fly(long, sleep_min=5.0)
        chk("C5 sleep_per_fly returns one row per mapped channel", len(pf), 379, 0)
        boolchk("C6 one-minute bins resolve the 5-min criterion, so sleep is "
                "scoreable (N81)",
                bool(pf["Sleep_scoreable"].all()),
                "scoreable %d/%d, bin_min %s"
                % (int(pf["Sleep_scoreable"].sum()), len(pf),
                   sorted(set(pf["bin_min"]))))

        # ---- D. THE GAP THIS SUITE CLOSES: the >=5-min rule itself ---------
        ind = {}
        for fid, sub in long.groupby("FlyID", sort=False):
            sub = sub.sort_values("Time_min")
            a = sub["Activity"].to_numpy(float)
            obs = int(np.count_nonzero(np.isfinite(a)))
            ind[fid] = published_sleep_minutes(a) / (obs / 1440.0)
        mod = pf.set_index("FlyID")["Total_sleep_min_per_day"].to_dict()
        keys = sorted(set(ind) & set(mod))
        diff = np.array([mod[k] - ind[k] for k in keys])
        chk("D1 every mapped fly is compared", len(keys), 379, 0)
        # The module publishes this column rounded to one decimal, so exact
        # agreement means agreement within half of that last place.
        boolchk("D2 the >=5-min rule matches the published definition on every "
                "fly, to the module's own rounding",
                float(np.nanmax(np.abs(diff))) <= 0.05 + 1e-9,
                "max |difference| %.6f min/day over %d real flies"
                % (np.nanmax(np.abs(diff)), len(keys)))
        boolchk("D3 the agreement is not a tie at zero sleep",
                float(np.nanmin([mod[k] for k in keys])) > 600.0,
                "lowest fly sleeps %.1f min/day" % np.nanmin([mod[k] for k in keys]))
        # and the criterion must actually bite: a different threshold must give
        # a different answer, or the comparison above proves nothing
        med = {}
        for sm in (1.0, 5.0, 25.0):
            tot = []
            for fid, sub in long.groupby("FlyID", sort=False):
                a = sub.sort_values("Time_min")["Activity"].to_numpy(float)
                obs = int(np.count_nonzero(np.isfinite(a)))
                tot.append(published_sleep_minutes(a, sleep_min=sm) / (obs / 1440.0))
            med[sm] = float(np.median(tot))
        boolchk("D4 the criterion bites: 1 / 5 / 25 min give different sleep",
                med[1.0] > med[5.0] > med[25.0],
                "median %.1f / %.1f / %.1f min/day"
                % (med[1.0], med[5.0], med[25.0]))

        # ---- E. the high sleep is not post-mortem silence ------------------
        rep_death = DR.death_report(lab, silence_h=12.0)
        trunc = DR.truncate_at_death(lab, silence_h=12.0)
        pf_t = D.sleep_per_fly(to_long(trunc), sleep_min=5.0)
        kept = len(trunc) / len(lab)
        boolchk("E1 death truncation keeps almost the whole record",
                kept > 0.97, "%.1f%% of minutes kept, %s tubes likely dead"
                % (100 * kept, rep_death.get("n_likely_dead")))
        s0 = float(pf["Total_sleep_min_per_day"].median())
        s1 = float(pf_t["Total_sleep_min_per_day"].median())
        boolchk("E2 and barely moves the median, so the high sleep is real, "
                "not silence after death",
                abs(s0 - s1) < 10.0,
                "median %.1f -> %.1f min/day after truncation" % (s0, s1))
        boolchk("E3 no fly reads a full 1440 min/day after truncation",
                float(pf_t["Total_sleep_min_per_day"].max()) < 1440.0,
                "max %.1f min/day (was %.1f)"
                % (pf_t["Total_sleep_min_per_day"].max(),
                   pf["Total_sleep_min_per_day"].max()))
        boolchk("E4 activity is in the healthy published range, so these are "
                "occupied tubes",
                150.0 <= float(pf_t["Total_activity_per_day"].median()) <= 5000.0,
                "median %.0f beam crossings/fly/day"
                % pf_t["Total_activity_per_day"].median())

        # ---- F. single-beam DAM scores much more sleep than published -------
        # This is a measurement, not a verdict. It is here because the
        # repository's plausibility ceiling judges data like this, and a real
        # published DAM dataset of healthy flies sits above it.
        lo, hi = DR.PLAUSIBLE_SLEEP_MIN_PER_DAY
        frac_out = float(((pf_t["Total_sleep_min_per_day"] < lo)
                          | (pf_t["Total_sleep_min_per_day"] > hi)).mean())
        boolchk("F1 most flies in this real published DAM dataset fall OUTSIDE "
                "the repository's plausible sleep range",
                frac_out > 0.5,
                "%.1f%% of %d flies outside %.0f-%.0f (median %.1f = %.1f h)"
                % (100 * frac_out, len(pf_t), lo, hi, s1, s1 / 60.0))
        rep = DR.plausibility_report(pf_t)
        boolchk("F2 the plausibility gate therefore fires on it, per group (N63)",
                bool(rep.get("ok")) is False and len(rep.get("sleep_by_group", {})) == 3,
                "ok=%s, groups flagged: %s"
                % (rep.get("ok"), sorted(rep.get("sleep_by_group", {}))))
        boolchk("F3 and the flag must not claim the value exceeds EVERY "
                "published figure, because this dataset is one",
                not any("above every" in f.lower() for f in rep.get("flags", [])),
                "flag wording checked on %d flags" % len(rep.get("flags", [])))
        boolchk("F4 the flag names a monitor-geometry explanation, not only a "
                "biological one",
                any("resting away from the beam" in f for f in rep.get("flags", [])),
                "single-beam DAM scores a fly away from the beam as asleep")

        # ---- G. the paper's NEGATIVE result, and pseudoreplication ----------
        pf_t = pf_t.copy()
        pf_t["Monitor"] = pf_t["FlyID"].str.split("_Ch").str[0]
        key = cmap.assign(FlyID=cmap["Monitor"] + "_Ch" + cmap["Channel"].astype(str))
        pf_t = pf_t.merge(key[["FlyID", "Replicate"]], on="FlyID", how="left")

        naive = D.compare_groups(pf_t, "Total_sleep_min_per_day", group_col="Genotype")
        p_naive = float(naive.get("main_result", {}).get("p_value", np.nan))
        boolchk("G1 a PER-FLY test declares a microbiome effect on sleep",
                p_naive < 0.05,
                "%s p = %.4g on n = %d flies" % (naive.get("method"), p_naive, len(pf_t)))

        rep_lvl = (pf_t.groupby(["Genotype", "Replicate"])["Total_sleep_min_per_day"]
                        .mean().reset_index())
        r_rep = D.compare_groups(rep_lvl, "Total_sleep_min_per_day", group_col="Genotype")
        p_rep = float(r_rep.get("main_result", {}).get("p_value", np.nan))
        boolchk("G2 aggregating to the REPLICATE COHORT -- the unit the "
                "experiment randomised -- does not",
                p_rep > 0.05,
                "p = %.4g on n = %d cohorts (%s)"
                % (p_rep, len(rep_lvl),
                   rep_lvl.groupby("Genotype").size().to_dict()))

        mon_lvl = (pf_t.groupby(["Genotype", "Monitor"])["Total_sleep_min_per_day"]
                        .mean().reset_index())
        r_mon = D.compare_groups(mon_lvl, "Total_sleep_min_per_day", group_col="Genotype")
        p_mon = float(r_mon.get("main_result", {}).get("p_value", np.nan))
        boolchk("G3 nor does aggregating to the MONITOR, the physical cluster",
                p_mon > 0.05,
                "p = %.4g on n = %d monitors" % (p_mon, len(mon_lvl)))
        boolchk("G4 so the published conclusion -- a LIMITED influence on sleep "
                "-- is recovered only when clustering is respected",
                p_naive < 0.05 and p_rep > 0.05 and p_mon > 0.05,
                "per-fly p = %.4g (n=379) vs cohort p = %.4g (n=19) and "
                "monitor p = %.4g (n=16)" % (p_naive, p_rep, p_mon))

        # how much the naive analysis overstated the evidence, from the data
        icc, mbar = icc_oneway(pf_t["Total_sleep_min_per_day"], pf_t["Monitor"])
        deff = H.design_effect(mbar, icc)
        boolchk("G5 the clustering is real and the module quantifies its cost",
                np.isfinite(icc) and icc > 0.02 and deff["deff"] > 1.3,
                "ICC = %.4f over %d monitors, mean cluster %.1f, DEFF = %.2f, "
                "SE inflated %.2fx" % (icc, pf_t["Monitor"].nunique(), mbar,
                                       deff["deff"], deff["se_inflation"])),
        boolchk("G6 ... and the effective sample size is far below 379 flies",
                379.0 / deff["deff"] < 200.0,
                "effective n = %.0f, not 379" % (379.0 / deff["deff"]))

        conf = DSC.check_confounding(
            pf_t[["Genotype", "Monitor"]], "Genotype", "Monitor")
        # Genotype is not COMPLETELY confounded with monitor -- every group
        # spans four to six monitors -- so `severity` is 0 and `confounded` is
        # False, both correctly. What was wrong was the conclusion drawn from
        # that: the verdict read "effects are separable" while Cramer's V, in
        # the same return value, was 0.91. Nine of the twelve monitors hold a
        # single group, which is what puts between-monitor variance into the
        # group contrast and makes G1 disagree with G2.
        boolchk("G7 strict confounding is correctly absent",
                bool(conf.get("confounded")) is False
                and float(conf.get("severity", -1)) == 0.0,
                "severity %.3f, Cramer's V %.2f"
                % (conf.get("severity", float("nan")),
                   conf.get("cramers_v", float("nan"))))
        boolchk("G8 ... but the checker reports that most monitors hold a "
                "single group",
                bool(conf.get("cluster_confounded")) is True
                and int(conf.get("n_nuisance_levels_single_factor", 0)) == 9,
                "%s of %s monitors hold one group (purity %.3f)"
                % (conf.get("n_nuisance_levels_single_factor"),
                   conf.get("n_nuisance_levels"),
                   float(conf.get("nuisance_purity", float("nan")))))
        boolchk("G9 ... and no longer tells the user the effects are separable "
                "without saying what that costs",
                "separable." not in str(conf.get("verdict", "")).split("--")[0]
                and "unit" in str(conf.get("verdict", "")),
                str(conf.get("verdict", ""))[:150])

        # ---- H. N82 on real data: a dropped channel is recorded -------------
        labelled = lab.rename(columns={"Group": "Genotype"}).assign(
            Treatment="None", Sex="Unknown")
        out = DR.to_dams_long(labelled, drop_dead=True, dead_threshold=50.0)
        at = out.attrs
        boolchk("H1 to_dams_long records the dead-channel filter on real data",
                "dead_channels_dropped" in at and "dead_channels_by_group" in at,
                "dropped %s at threshold %s, by group %s"
                % (at.get("dead_channels_dropped"), at.get("dead_threshold"),
                   at.get("dead_channels_by_group")))
        boolchk("H2 ... and the exclusion is small on a healthy recording",
                int(at.get("dead_channels_dropped", -1)) <= 10,
                "%s of 379 channels" % at.get("dead_channels_dropped"))

        # ---- I. what this suite does NOT establish -------------------------
        boolchk("I1 no annotated ground truth for sleep exists here, so the "
                "RULE is validated and the MEASUREMENT is not", True,
                "L2 and L3 established; L4 for sleep remains open (see F1-F4)")

    print("validate_dam_raw_microbiome -- real raw Trikinetics beam counts")
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
