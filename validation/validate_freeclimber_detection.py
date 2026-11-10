"""
validate_freeclimber_detection.py
Is the FreeClimber reimplementation faithful enough to serve as a reference?

WHY THIS SUITE EXISTS
---------------------
`validate_freeclimber_benchmark.py` reproduces FreeClimber's published SLOPES
from FreeClimber's published detections, and says plainly what it cannot
establish: "anything about DroPhenix's own particle DETECTION (this starts from
FreeClimber's detections, not from the video)". Comparing two detectors needs
both run on the same video, so `modules/freeclimber_detector.py` reimplements
FreeClimber's detection method.

A reimplementation used as a reference must be shown faithful before it is used
to judge anything. It cannot be checked by running the published program: that
program pins `trackpy==0.4.2`, which does not build on Python 3.13, and with a
current trackpy it cannot run at all, because `detector.find_spots` passes
`invert=True` through to `trackpy.batch` and that argument was removed after
0.4.x.

What makes the check possible anyway is that FreeClimber ships its example
video together with the raw detections, the filtered detections and the slopes
it produced from them. So the reimplementation is run on their video, with
their configuration, and compared against their own expected answer.

DATA (redistributed with the tool under its own licence, Copyright 2020 Adam
Spierer)
    reference_datasets/freeclimber/
      w1118_m_2_1.h264          -- the example recording as the H.264 stream
                                   FreeClimber actually processed
                                   (`file_suffix="h264"` in their config)
      w1118_m_2_1.mp4           -- the same recording re-encoded; kept as a
                                   fallback, and the difference between the two
                                   is itself measured in section A
      example.cfg               -- the parameters they used
      w1118_m_2_1.raw.csv       -- their 4,289 located spots, with the
                                   True_particle column their filters produced
      w1118_m_2_1.filtered.csv  -- their 4,093 surviving spots

WHAT IS VALIDATED, AND AT WHICH LAYER
-------------------------------------
A. THE DETECTION CORE (layer 2). The number of located spots, and the mass,
   eccentricity and signal distributions of the survivors, against theirs.

B. THE FILTER RULE IS THE RULE THEY APPLIED (layer 2). Their published
   `True_particle` column records the outcome of their filters, so the rule is
   recoverable rather than guessable. It is reconstructed and the
   reconstruction is checked against the column.

C. PER-FRAME AND PER-VIAL AGREEMENT (layer 2/4). Totals can agree while every
   frame disagrees, so the comparison is made frame by frame and vial by vial.

D. THE PUBLISHED AUTO-THRESHOLD IS A BIN INDEX (layer 2, a defect in the
   reference tool). `detector.find_threshold` returns the INDEX of the most
   prominent bin of a 40-bin histogram of `signal`, and `step_4b` compares that
   integer against raw signal values. This is asserted, not remarked, because
   the reimplementation deliberately reproduces it and a future reader will
   otherwise take it for a bug here.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
Anything about which detector is more accurate. This suite establishes only
that the reference is a faithful stand-in for the published tool on the one
recording for which a published answer exists -- one video, one parameter set,
one laboratory's camera. The accuracy comparison against hand counts is a
separate measurement and must not be read out of this file.

Run: ./.venv/Scripts/python.exe validation/validate_freeclimber_detection.py
Exit 0 = all pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(__file__).resolve().parents[1] / "reference_datasets" / "freeclimber"
# The H.264 ELEMENTARY STREAM, not the .mp4 beside it.
#
# FreeClimber's config says `file_suffix="h264"`, so the published detections
# came from this stream. The .mp4 in the same directory is a re-encode of it,
# and re-encoding is lossy: running the identical code on the .mp4 gives 4,262
# located spots and 4,070 survivors, against 4,273 and 4,091 from the stream,
# where the published answer is 4,289 and 4,093. Vial 2's total is exact from
# the stream and 19 spots short from the .mp4.
#
# So the residual disagreement between this reimplementation and the published
# output is dominated by WHICH ENCODING IS DECODED, not by the method. The
# comparison therefore uses the input the published output was computed from,
# and falls back to the .mp4 only if the stream is absent -- in which case the
# tolerances below are the ones that matter and the difference is attributable.
VIDEO = DATA / "w1118_m_2_1.h264"
VIDEO_FALLBACK = DATA / "w1118_m_2_1.mp4"
CFG = DATA / "example.cfg"
RAW = DATA / "w1118_m_2_1.raw.csv"
FILT = DATA / "w1118_m_2_1.filtered.csv"

FAILURES: List[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name + (f" ({detail})" if detail else ""))


def read_frames(path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])         # cv2 decodes BGR; ffmpeg gives RGB
    cap.release()
    return frames


def main() -> int:
    print("=" * 78)
    print("VALIDATION: FreeClimber's detection method, reimplemented")
    print("=" * 78)

    video = VIDEO if VIDEO.is_file() else VIDEO_FALLBACK
    for p in (video, CFG, RAW, FILT):
        if not p.is_file():
            print(f"  [SKIP] {p.name} is not present; this suite established nothing")
            return 0

    try:
        import pandas as pd
        from modules import freeclimber_detector as FCD
    except Exception as exc:                                   # pragma: no cover
        print(f"  [SKIP] cannot import the reference implementation ({exc!r})")
        return 0

    try:
        FCD._trackpy()
    except FCD.TrackpyMissing as exc:
        print("  [SKIP] trackpy is not installed, so FreeClimber's detector "
              "cannot be run here.")
        print(f"         {exc}")
        return 0

    ref_raw = pd.read_csv(RAW)
    ref_filt = pd.read_csv(FILT)
    p = FCD.FreeClimberParams.from_cfg(str(CFG))

    frames = read_frames(VIDEO)
    print(f"\ndecoded {len(frames)} frames; FreeClimber's config covers frames "
          f"{p.crop_0}-{p.crop_n}")
    if not frames:
        print("  [SKIP] the video could not be decoded here")
        return 0

    res = FCD.detect(frames, p)
    got = res.spots
    got_true = got[got["True_particle"]]

    # ---------------------------------------------------------------- A -----
    print("\nA. The detection core against their published detections")
    print(f"      located spots: {res.n_raw} here, {len(ref_raw)} published")
    print(f"      survivors    : {res.n_true} here, {len(ref_filt)} published")
    check("located-spot count within 1% of the published count",
          abs(res.n_raw - len(ref_raw)) <= 0.01 * len(ref_raw),
          f"{res.n_raw} vs {len(ref_raw)}")
    check("surviving-spot count within 0.5% of the published count",
          abs(res.n_true - len(ref_filt)) <= 0.005 * len(ref_filt),
          f"{res.n_true} vs {len(ref_filt)}")

    ref_true = ref_raw[ref_raw["True_particle"]]
    for col, tol in (("mass", 0.01), ("ecc", 0.02), ("signal", 0.01)):
        a = float(got_true[col].mean())
        b = float(ref_true[col].mean())
        check(f"mean {col} of the survivors agrees to {tol:.0%}",
              abs(a - b) <= tol * abs(b), f"{a:.3f} vs {b:.3f}")

    # Their `size` column is a constant, which is worth asserting so that a
    # reader does not mistake the disagreement for an error here.
    check("their published `size` column is a single constant, because "
          "`df['size'] = df.size` reads the DataFrame's element count",
          ref_raw["size"].nunique() == 1,
          f"unique values: {ref_raw['size'].unique()[:3]}")

    # ---------------------------------------------------------------- B -----
    print("\nB. The filter rule is the rule they applied")
    thr = FCD.auto_signal_threshold(ref_raw["signal"].to_numpy())
    trim = {e: FCD.trim_line(
        ref_raw["x" if e in ("left", "right") else "y"].to_numpy(), e,
        p.outlier_LR if e in ("left", "right") else p.outlier_TB)
        for e in ("left", "right", "bottom", "top")}
    rule = ((ref_raw["signal"] >= thr)
            & (ref_raw["ecc"] >= p.ecc_low) & (ref_raw["ecc"] <= p.ecc_high)
            & (ref_raw["x"] >= trim["left"]) & (ref_raw["x"] <= trim["right"])
            & (ref_raw["y"] >= trim["bottom"]) & (ref_raw["y"] <= trim["top"]))
    agree = float((rule.to_numpy() == ref_raw["True_particle"].to_numpy()).mean())
    print(f"      threshold {thr:.3f}; trim "
          f"{ {k: round(v, 1) for k, v in trim.items()} }")
    check("the reconstructed filter reproduces their True_particle column "
          "for at least 99.5% of their own spots",
          agree >= 0.995, f"{100 * agree:.2f}% of {len(ref_raw)} rows")

    # ---------------------------------------------------------------- C -----
    print("\nC. Frame by frame and vial by vial, not just in total")
    mine = got_true.groupby("frame").size()
    theirs = ref_filt.groupby("frame").size()
    common = sorted(set(mine.index) & set(theirs.index))
    d = np.array([int(mine[f]) - int(theirs[f]) for f in common], dtype=float)
    print(f"      {len(common)} frames in common; per-frame difference "
          f"mean {d.mean():+.3f}, max |diff| {np.abs(d).max():.0f}")
    check("every frame's spot count is within 2 of the published count",
          float(np.abs(d).max()) <= 2.0, f"max |diff| {np.abs(d).max():.0f}")
    check("the mean per-frame difference is under half a spot",
          abs(float(d.mean())) < 0.5, f"{d.mean():+.4f}")

    mv = got_true["vial"].value_counts().sort_index()
    tv = ref_filt["vial"].value_counts().sort_index()
    print(f"      per vial: {dict(mv)} here, {dict(tv)} published")
    for v in sorted(set(mv.index) & set(tv.index)):
        check(f"vial {v} total within 0.5% of the published total",
              abs(int(mv[v]) - int(tv[v])) <= max(2, 0.005 * int(tv[v])),
              f"{int(mv[v])} vs {int(tv[v])}")

    # ---------------------------------------------------------------- D -----
    print("\nD. The published auto-threshold is a histogram BIN INDEX")
    sig = ref_raw["signal"].to_numpy(float)
    counts, edges = np.histogram(sig, bins=40)
    published = FCD.auto_signal_threshold(sig, mode="published")
    edge = FCD.auto_signal_threshold(sig, mode="bin_edge")
    print(f"      signal spans {sig.min():.2f} to {sig.max():.2f}; "
          f"bin width {edges[1] - edges[0]:.3f}")
    print(f"      published rule returns {published:.3f}; the lower edge of "
          f"that bin is {edge:.3f}")
    check("the published rule returns a whole number, i.e. an index rather "
          "than a measured signal value",
          float(published).is_integer(), f"{published!r}")
    check("that index lies outside the bin it indexes, so it cannot be a "
          "signal value from that bin",
          not (edge <= published <= edge + (edges[1] - edges[0])),
          f"{published:.2f} vs bin [{edge:.2f}, {edge + (edges[1] - edges[0]):.2f}]")
    n_pub = int((sig >= published).sum())
    n_edge = int((sig >= edge).sum())
    print(f"      as published it keeps {n_pub} of {len(sig)} spots; using the "
          f"bin edge would keep {n_edge}")
    check("the two readings of that number give materially different "
          "analyses, so the unit error is not harmless",
          abs(n_pub - n_edge) > 0.2 * len(sig),
          f"{n_pub} vs {n_edge} of {len(sig)}")
    check("and it is the PUBLISHED reading that matches their output, which is "
          "what establishes the mechanism",
          abs(n_pub - len(ref_filt)) < abs(n_edge - len(ref_filt)),
          f"published {n_pub}, bin edge {n_edge}, their survivors {len(ref_filt)}")

    print("\n" + "=" * 78)
    print("WHAT THIS SUITE DOES NOT ESTABLISH: anything about which detector is")
    print("more accurate. One video, one parameter set, one camera. The")
    print("accuracy comparison against hand counts is a separate measurement.")
    print("=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL FREECLIMBER-DETECTION VALIDATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
