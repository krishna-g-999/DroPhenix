"""Accuracy of the Climbing (Simple) tool against hand counts.

Takes the filled-in bench sheet and the CSV the page exports, and prints the
only numbers that decide whether the tool is usable: MAE overall, MAE by day,
MAE by timepoint, the vials that disagree badly, and whether the result sits
inside the assay's own 3-fly noise floor.

The report itself lives in `modules.climbing_band.validation_report`, which the
Validate tab also calls, so the printed and on-screen versions cannot diverge.

Run:
    ./.venv/Scripts/python.exe scripts/validate_climbing_band.py \
        data/validation/hand_count_template.csv  tool_output.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.climbing_band import validation_report  # noqa: E402

#: Where the recordings live, used only to say whether a video is Day 3 or Day 7.
#: The day is not in either CSV, and guessing it from a filename would be a guess.
DEFAULT_DATASET = Path(r"C:/Users/gkp11/Videos/HD Climbing assay")


def build_day_lookup(root: Path) -> dict:
    """{filename: 'Day 3'|'Day 7'} from the folder each recording sits in."""
    if not root or not root.exists():
        return {}
    out = {}
    for f in root.glob("**/*.MOV"):
        s = str(f)
        if "3 rd day" in s:
            out[f.name] = "Day 3"
        elif "7 th day" in s:
            out[f.name] = "Day 7"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hand_counts", help="the filled-in bench sheet")
    ap.add_argument("tool_output", help="CSV exported by the Climbing (Simple) page")
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET),
                    help="folder holding the recordings, for the Day 3 / Day 7 split")
    a = ap.parse_args()

    hand = pd.read_csv(a.hand_counts)
    tool = pd.read_csv(a.tool_output)
    lookup = build_day_lookup(Path(a.dataset_root))
    print(validation_report(hand, tool, day_lookup=lookup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
