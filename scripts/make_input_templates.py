"""
Generate the input templates and ready-to-run demo files for every DroPhenix assay.

WHY THIS EXISTS
---------------
`app.py` has 27 `st.file_uploader` call sites and the repository shipped exactly
three template files, all of them for one climbing hand-count validation. A user
arriving with their own data had no way to learn what columns any page expected
except by uploading something and reading the error -- and several pages fail
with a bare `KeyError` rather than naming the missing column.

This writes two things per assay into `data/templates/`:

  <assay>_template.csv   the column header plus three EXAMPLE rows, so the shape
                         and the units are visible. Delete the examples, keep the
                         header, paste your own data.
  <assay>_demo.csv       a complete, analysable dataset produced by the module's
                         own demo generator. Upload it to see the page work
                         before trusting it with real data.

It is deliberately repository-relative and takes no laboratory data, so it runs
anywhere: `python scripts/make_input_templates.py`.

WHAT THE TEMPLATES ARE AND ARE NOT
-----------------------------------
They are a statement of the INPUT CONTRACT -- which columns are required, which
are optional, and what the units are. They are not a validation of the analysis
that consumes them, and a file that loads is not a file that is correct: the
units, the experimental unit and the denominator are still the user's
responsibility, and the pages that can check those will say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "templates"

#: assay -> (required columns, optional columns, one-line note per column)
CONTRACTS: dict[str, dict] = {
    "climbing": {
        "required": ["Genotype", "Sex", "Treatment", "Replicate", "Time", "Count", "n"],
        "optional": ["Age", "Dose", "Compound", "VialID", "ZT", "n_alive"],
        "units": {
            "Time": "seconds since the tap",
            "Count": "flies ABOVE the scoring line at that time",
            "n": "flies in the vial -- see the note on the denominator below",
            "Replicate": "the VIAL, not the fly: one row per vial per timepoint",
            "Age": "days", "Dose": "mM", "ZT": "Zeitgeber hour, 0-23",
            "n_alive": "flies ALIVE at scoring; give this when mortality is "
                       "non-trivial, or the index is diluted by dead flies",
        },
        "rows": [
            ("Oregon-R", "Male", "Untreated", 1, 10, 10, 20),
            ("Oregon-R", "Male", "Untreated", 1, 20, 14, 20),
            ("TDP43", "Male", "Untreated", 1, 10, 4, 20),
        ],
    },
    "lifespan_per_fly": {
        "required": ["Genotype", "Treatment", "Sex", "Time", "Event"],
        "optional": [],
        "units": {
            "Time": "age in days at death or at censoring",
            "Event": "1 = died, 0 = censored (escaped, lost, still alive at the "
                     "end). Words work too: dead/alive/censored/escaped",
        },
        "rows": [
            ("Oregon-R", "Vehicle", "Male", 62, 1),
            ("Oregon-R", "Vehicle", "Male", 70, 0),
            ("TDP43", "Vehicle", "Male", 31, 1),
        ],
    },
    "lifespan_counts": {
        "required": ["Genotype", "Treatment", "Sex", "Day", "Dead_count"],
        "optional": ["Censored_count", "Total_start"],
        "units": {
            "Day": "day of the census",
            "Dead_count": "NEW deaths counted that day, not cumulative",
            "Censored_count": "flies lost or escaped that day",
            "Total_start": "cohort size at day 0; give it once per group",
        },
        "rows": [
            ("Oregon-R", "Vehicle", "Male", 0, 0, 0, 100),
            ("Oregon-R", "Vehicle", "Male", 7, 2, 1, 100),
            ("TDP43", "Vehicle", "Male", 7, 9, 0, 100),
        ],
    },
    "ymaze": {
        "required": ["Genotype", "Treatment", "Sex", "FlyID", "Sequence"],
        "optional": [],
        "units": {
            "FlyID": "unique per fly -- the fly IS the experimental unit here",
            "Sequence": "the arms visited in order, as one string of A/B/C "
                        "(e.g. ABCACB). Turn bias and alternation are both "
                        "derived from it",
        },
        "rows": [
            ("Oregon-R", "Vehicle", "Male", "f001", "ABCABCACBA"),
            ("Oregon-R", "Vehicle", "Female", "f002", "ACBACBABCA"),
            ("TDP43", "Vehicle", "Male", "f003", "ABABABACAB"),
        ],
    },
    "dams_per_fly": {
        "required": ["Genotype", "Treatment", "Sex", "FlyID", "Time_min", "Activity"],
        "optional": ["Light"],
        "units": {
            "Time_min": "minutes since the start of the recording, one row per bin",
            "Activity": "beam crossings in that bin. Leave BLANK for a missing "
                        "bin -- do not write 0, which reads as sleep",
            "Light": "1 = lights on, 0 = lights off; supply it when the schedule "
                     "is not a plain 12:12 starting at Time_min = 0",
        },
        "rows": [
            ("Oregon-R", "Vehicle", "Male", "M1C01", 0, 3),
            ("Oregon-R", "Vehicle", "Male", "M1C01", 1, 0),
            ("Oregon-R", "Vehicle", "Male", "M1C01", 2, 7),
        ],
    },
    "dams_channel_map": {
        "required": ["Monitor", "Channel", "Genotype", "Sex", "Treatment"],
        "optional": [],
        "units": {
            "Monitor": "monitor number, matching the raw DAMSystem .txt file",
            "Channel": "channel 1-32 within that monitor",
            "Genotype": "SPREAD GENOTYPES ACROSS MONITORS. Putting one genotype "
                        "in one monitor confounds it with the device; the "
                        "design check will refuse the comparison",
        },
        "rows": [
            (1, 1, "Oregon-R", "Male", "Vehicle"),
            (1, 2, "TDP43", "Male", "Vehicle"),
            (1, 3, "Oregon-R", "Male", "Drug"),
        ],
    },
    "grooming": {
        "required": ["Genotype", "Treatment", "Sex", "FlyID",
                     "Bout_start_s", "Bout_end_s", "Observation_s"],
        "optional": [],
        "units": {
            "Bout_start_s": "seconds from the start of observation",
            "Bout_end_s": "seconds; must be greater than Bout_start_s",
            "Observation_s": "total length of the observation for that fly -- "
                             "the denominator. Constant per fly, repeated on "
                             "every bout row",
        },
        "rows": [
            ("Oregon-R", "Vehicle", "Male", "f001", 12.4, 18.9, 600.0),
            ("Oregon-R", "Vehicle", "Male", "f001", 45.0, 52.2, 600.0),
            ("TDP43", "Vehicle", "Male", "f002", 8.1, 30.7, 600.0),
        ],
    },
    "larval": {
        "required": ["Genotype", "Treatment", "Sex", "FlyID", "duration_s",
                     "mean_speed_mm_s", "path_length_mm", "displacement_mm"],
        "optional": ["tortuosity"],
        "units": {
            "duration_s": "seconds tracked",
            "mean_speed_mm_s": "mm per second",
            "path_length_mm": "total distance travelled, mm",
            "displacement_mm": "straight-line start-to-end distance, mm",
            "tortuosity": "path_length / displacement; computed if omitted",
        },
        "rows": [
            ("Oregon-R", "Vehicle", "Male", "L001", 60.0, 0.72, 43.2, 18.1, 2.39),
            ("Oregon-R", "Vehicle", "Female", "L002", 60.0, 0.68, 40.8, 21.0, 1.94),
            ("TDP43", "Vehicle", "Male", "L003", 60.0, 0.31, 18.6, 6.2, 3.00),
        ],
    },
}


def write_templates() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for assay, spec in CONTRACTS.items():
        cols = spec["required"] + [c for c in spec["optional"]
                                   if c not in spec["required"]]
        rows = [list(r) + [""] * (len(cols) - len(r)) for r in spec["rows"]]
        df = pd.DataFrame(rows, columns=cols)
        p = OUT / f"{assay}_template.csv"
        df.to_csv(p, index=False)
        written.append(p.name)
    return written


def write_demos() -> list[str]:
    """Complete, analysable datasets from each module's own demo generator."""
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    from modules import synthetic as S
    from modules import ymaze as Y
    from modules import grooming as G
    from modules import larval as L
    from modules import dams as D

    climb = S.generate_demo_study()
    climb.to_csv(OUT / "climbing_demo.csv", index=False)
    written.append("climbing_demo.csv")

    # A lifespan table derived from the SAME demo study, so the two can be used
    # together on the cross-assay pages rather than describing different flies.
    if "DeathAge" in climb.columns:
        per_fly = (climb.drop_duplicates(subset=["VialID"])
                        .loc[:, ["Genotype", "Treatment", "Sex", "DeathAge"]]
                        .rename(columns={"DeathAge": "Time"}))
        per_fly["Time"] = per_fly["Time"].round(1)
        per_fly["Event"] = 1
        per_fly.to_csv(OUT / "lifespan_demo.csv", index=False)
        written.append("lifespan_demo.csv")

    Y.demo_ymaze().to_csv(OUT / "ymaze_demo.csv", index=False)
    written.append("ymaze_demo.csv")
    G.demo_grooming().to_csv(OUT / "grooming_demo.csv", index=False)
    written.append("grooming_demo.csv")
    L.demo_larval().to_csv(OUT / "larval_demo.csv", index=False)
    written.append("larval_demo.csv")

    # The DAM demo is one row per fly per MINUTE -- 8 days x 86 flies is about a
    # million rows and ~30 MB, which is not a useful download. Ship two days.
    # One day and 16 flies, not eight days and 86. The per-minute table is
    # ~4 MB per day at full width, which is not a useful thing to ship or to
    # download; 16 flies still gives four flies in each of the four
    # genotype x treatment cells, which is enough for every group comparison on
    # the page to run.
    dam = D.demo_dams(n_days=1)
    keep = (dam.drop_duplicates(subset=["Genotype", "Treatment", "Sex", "FlyID"])
               .groupby(["Genotype", "Treatment"], as_index=False)
               .head(4)["FlyID"].tolist())
    dam = dam[dam["FlyID"].isin(keep)]
    dam.to_csv(OUT / "dams_demo.csv", index=False)
    written.append("dams_demo.csv")

    return written


def main() -> int:
    # The repository ALREADY ships input templates in `templates/`, and they all
    # parse (verified through `data_parser.load_data`). What was missing is a
    # ready-to-run DEMO DATASET per assay -- a complete file a user can upload to
    # watch a page work before trusting it with their own data. Only those are
    # written. `CONTRACTS` above is retained because it is the written-down input
    # contract (required vs optional columns, units, and the traps), which lives
    # nowhere else; `write_templates()` can regenerate a template if one is lost.
    d = write_demos()
    print("demo datasets:")
    for n in d:
        sz = (OUT / n).stat().st_size / 1e6
        print(f"  {n}  ({sz:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
