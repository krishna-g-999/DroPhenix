# `scripts/` — the author's own analysis runs, not part of the product

**Read this before trying to run anything here.** Register N117.

These are the one-off scripts that produced particular measurements for this
project: pipeline runs over the laboratory's recordings, parameter studies,
detector comparisons. They are tracked because the analyses they performed are
cited in `docs/` and in the manuscript, and a cited analysis whose script is not
published is not reproducible. They are **not** part of the application, nothing
in `app.py` or `drophenix_cli.py` imports them, and `validation/run_all.py` does
not run them.

## They are machine-specific, and deliberately left that way

**10 of the 14 tracked scripts here hardcode an absolute path** — `D:/DroPhenix/...`
or `C:/Users/gkp11/...` — so they will not run unchanged on another computer.
That is a real limitation and it is stated here rather than discovered.

They have been left as they are on purpose. Each one is a record of an analysis
that was actually run, against data at a particular place on a particular
machine; rewriting ten untested research scripts to be portable would risk
silently changing what they compute, and the thing worth preserving about them is
exactly what they computed. The product's own code is a different matter: the
same defect inside `app.py` (`_climbing_simple_validate`, which silently dropped
the Day 3 / Day 7 stratification off the author's machine) **was** fixed, because
that one affects users.

## To run one of these elsewhere

1. Open the script and read the path constants at the top — they are all in the
   first ~30 lines, by convention.
2. Point them at your own copy of the data.
3. Expect to supply laboratory recordings or monitor files that are **not** in
   this repository. See `validation/lab_data.py` for the discovery logic the
   validation suite uses, and these environment variables, which the product and
   the suite both honour:

   - `DROPHENIX_CLIMBING_ROOT` — the folder holding the climbing recordings
   - `DROPHENIX_DAM_ROOT` — the folder holding the activity-monitor files
   - `DROPHENIX_FLY_BLOB_CACHE` — a pre-built detector hold-out cache
   - `DROPHENIX_NO_LAB_DATA` — set to any value to force every lab-data path to
     resolve to "absent", which is how CI runs

## What this directory does *not* establish

Nothing. A script here having run successfully on one machine is not validation
of anything; the validation suite is `validation/run_all.py`, and the separation
between "it ran" and "it is correct" is the whole point of the L1/L2/L3/L4
layering described in `docs/`.
