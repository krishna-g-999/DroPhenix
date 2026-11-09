"""
freeclimber_detector.py
FreeClimber's published detection method, reimplemented, so that DroPhenix's
own detector can be compared against the tool the field already uses -- on the
same frames, against the same hand counts.

WHY A REIMPLEMENTATION RATHER THAN THE TOOL ITSELF
--------------------------------------------------
`validation/validate_freeclimber_benchmark.py` already reproduces FreeClimber's
published SLOPES from FreeClimber's own published detections, and says plainly
what it does not establish: "anything about DroPhenix's own particle
DETECTION (this starts from FreeClimber's detections, not from the video)".
That is the gap this module exists to close. Comparing detectors needs both
detectors run on the same video.

Running the published program on this machine is not possible:

  * it pins `trackpy==0.4.2`, which does not build on Python 3.13;
  * and with a current trackpy it cannot run at all, independently of any
    build problem: `detector.find_spots` passes `invert=True` through to
    `trackpy.batch`, and that argument was removed after 0.4.x. The call is a
    TypeError on any modern trackpy.

So the method is reimplemented from the published source and the published
configuration, exactly as the slope benchmark reimplements the regression. The
reimplementation is not assumed faithful: it is checked against FreeClimber's
OWN published `raw.csv` and `filtered.csv` for its OWN example video, by
`validation/validate_freeclimber_detection.py`. If those do not reproduce, this
module must not be used as a reference and the suite says so.

THE METHOD, from Spierer et al. 2021 and scripts/detector.py
------------------------------------------------------------
1. Crop each frame to the ROI (x, y, w, h) and convert to greyscale with the
   ITU-R BT.601 luma weights 0.2989 / 0.5870 / 0.1140.
2. Background = the per-pixel MEDIAN over frames `blank_0` to `blank_n`.
3. Subtract: the published code computes `frame - background` and then asks
   trackpy to invert, which finds dark features. `background - frame` is the
   same thing without the deprecated argument, and is what is done here --
   flies are darker than the lit backdrop, so this is positive where a fly is.
4. `trackpy.locate` per frame with `diameter`, `minmass`, `maxsize`. This is
   the Crocker-Grier centroid finder (Crocker & Grier 1996), which is what
   FreeClimber uses and is a genuinely different algorithm from either of
   DroPhenix's detectors.
5. Threshold the `signal` metric. With `threshold="auto"` FreeClimber
   histograms signal into 40 bins and returns the INDEX of the most prominent
   peak, which `step_4b` then compares against raw signal values. That is a
   unit error in the published tool -- a dimensionless bin index used as a
   photometric threshold -- and it is reproduced here deliberately, because a
   comparison has to compare against the tool as the field runs it. See
   `auto_signal_threshold`, which establishes the behaviour from FreeClimber's
   own published output rather than inferring it, and which offers the
   corrected variant only for measuring how much the defect changes the
   answer.
6. Keep spots whose eccentricity lies in [`ecc_low`, `ecc_high`].
7. Trim spots outside the bulk of the detections at each edge, by the
   median-absolute-deviation rule in `detector.get_trim_lines`.
8. Assign the survivors to `vials` equal-width bins across the x-range of the
   surviving spots.

WHAT THIS MODULE IS NOT
-----------------------
It is not part of the DroPhenix pipeline and nothing in the pipeline imports
it. It is a REFERENCE IMPLEMENTATION used only for comparison, and it lives in
`modules/` only because the validation suites import from there. It also
requires `trackpy`, which is not a DroPhenix dependency: every entry point
degrades to an explicit refusal when trackpy is absent, rather than failing
somewhere deeper.

References
----------
Spierer AN, Yoon D, Zhu C-T, Rand DM (2021) FreeClimber: automated
  quantification of climbing performance in Drosophila. J Exp Biol
  224:jeb229377.
Crocker JC, Grier DG (1996) Methods of digital video microscopy for colloidal
  studies. J Colloid Interface Sci 179:298-310.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ITU-R BT.601 luma weights, as in detector.crop_and_grayscale.
_LUMA = (0.2989, 0.5870, 0.1140)


class TrackpyMissing(RuntimeError):
    """Raised when trackpy is not installed, instead of failing later."""


def _trackpy():
    try:
        import trackpy as tp
    except Exception as exc:                                    # pragma: no cover
        raise TrackpyMissing(
            "FreeClimber's detector is the Crocker-Grier centroid finder, which "
            "is provided by trackpy. trackpy is not installed, and it is NOT a "
            "DroPhenix dependency -- it is needed only to run the reference "
            "implementation this module provides. Install it with "
            "`pip install trackpy looseversion` to enable the comparison, or "
            "skip the comparison; nothing in the DroPhenix pipeline needs it."
        ) from exc
    try:
        tp.quiet()
    except Exception:
        pass
    return tp


@dataclass
class FreeClimberParams:
    """
    The analysis parameters, under the names FreeClimber's own config uses.

    The defaults are the values in the published `example/example.cfg`, which
    is the only parameter set for which a published expected output exists, and
    therefore the only one against which this reimplementation can be checked.
    """

    x: int = 0
    y: int = 0
    w: Optional[int] = None
    h: Optional[int] = None
    blank_0: int = 0
    blank_n: int = 145
    crop_0: int = 0
    crop_n: int = 145
    threshold: object = "auto"
    diameter: int = 7
    minmass: float = 100.0
    maxsize: float = 11.0
    ecc_low: float = 0.05
    ecc_high: float = 0.58
    vials: int = 3
    window: int = 50
    frame_rate: float = 29.0
    outlier_TB: float = 1.0
    outlier_LR: float = 3.0
    trim_outliers: bool = True

    @classmethod
    def from_cfg(cls, path: str) -> "FreeClimberParams":
        """
        Read a FreeClimber `.cfg`.

        The format is `key=value` with `#` comments and optional quotes, which
        is what FreeClimber writes; values are coerced to the field types
        declared above, so a malformed file raises here rather than producing a
        silently different analysis.
        """
        import dataclasses
        types = {f.name: f.type for f in dataclasses.fields(cls)}
        kw: Dict[str, object] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k not in types:
                    continue
                if k == "threshold":
                    kw[k] = v if v == "auto" else float(v)
                elif k == "trim_outliers":
                    kw[k] = v.lower() in ("true", "1", "yes")
                elif k in ("diameter", "vials", "window", "x", "y", "w", "h",
                           "blank_0", "blank_n", "crop_0", "crop_n"):
                    kw[k] = int(float(v))
                else:
                    try:
                        kw[k] = float(v)
                    except ValueError:
                        continue
        return cls(**kw)                                 # type: ignore[arg-type]


def to_greyscale_stack(frames: Sequence[np.ndarray],
                       p: FreeClimberParams) -> np.ndarray:
    """
    Crop to the ROI and convert to greyscale, as `crop_and_grayscale` does.

    Frames may be colour (H, W, 3) or already greyscale (H, W); a greyscale
    frame is used as-is rather than being re-weighted, since the luma weights
    sum to one and applying them to a single channel would be a no-op anyway.
    """
    y0, x0 = int(p.y), int(p.x)
    y1 = y0 + int(p.h) if p.h else None
    x1 = x0 + int(p.w) if p.w else None
    out = []
    for f in frames:
        a = np.asarray(f)
        if a.ndim == 3 and a.shape[2] >= 3:
            sub = a[y0:y1, x0:x1, :3].astype(float)
            g = (_LUMA[0] * sub[..., 0] + _LUMA[1] * sub[..., 1]
                 + _LUMA[2] * sub[..., 2])
        else:
            g = a[y0:y1, x0:x1].astype(float)
        out.append(g)
    return np.stack(out) if out else np.zeros((0, 0, 0))


def subtract_background(stack: np.ndarray,
                        p: FreeClimberParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel median background over the blank frames, subtracted.

    FreeClimber computes `frame - background` and asks trackpy to invert, which
    finds dark features; the sign is flipped here instead, which is the same
    operation without the argument trackpy removed. Flies are darker than the
    lit backdrop, so the result is positive where a fly is.

    The median is taken as `int`, as the published code does, because that
    rounding changes the detections by a fraction of a grey level and the point
    of this module is to reproduce the published numbers rather than to improve
    on them.
    """
    lo, hi = int(p.blank_0), int(p.blank_n)
    hi = min(hi, stack.shape[0]) if hi else stack.shape[0]
    bg = np.median(stack[lo:hi].astype(float), axis=0).astype(int)
    return (bg.astype(float) - stack.astype(float)), bg


def auto_signal_threshold(signal: np.ndarray, bins: int = 40,
                          mode: str = "published") -> float:
    """
    FreeClimber's automatic signal threshold, as the published code behaves.

    WHAT THE PUBLISHED CODE DOES, which is not what its docstring says. The
    docstring promises "the local minimum between two local maxima, or the
    average between 0 and the global maximum". The body histograms the signal
    into `bins`, finds peaks, and returns

        find_peaks(counts, prominence=np.max(peak_prominences(counts, peaks)))[0][0]

    `find_peaks(...)[0]` is an array of INDICES into the histogram. The value
    returned is therefore a BIN INDEX in 0..bins-1, and `step_4b` compares that
    integer against raw `signal` values. It is not a threshold on the data: it
    is a dimensionless index used as though it were a photometric value, and it
    only falls inside the signal range by coincidence.

    Measured on FreeClimber's own example recording: it returns 25; bin 25
    spans signal 45.28 to 46.78; and 25 used as a signal threshold keeps 4,098
    of 4,289 spots, which is the published 4,093 after eccentricity and trim
    remove five more. Using the bin's actual lower edge instead would keep
    2,595. So the behaviour is established from the published output rather
    than deduced from the source.

    (A second, smaller slip is kept because it changes the result: `np.max` is
    taken over the whole 3-tuple `peak_prominences` returns, so it maximises
    over the left and right BASES -- which are also indices -- as well as the
    prominences. On this example the prominences happen to dominate, so it has
    no effect here, but it is reproduced rather than silently repaired.)

    `mode="published"` reproduces the tool. `mode="bin_edge"` returns the lower
    edge of that bin, which is a real signal value; it exists only to measure
    how much the unit error changes the answer, and must not be used when the
    point is to compare against FreeClimber as the field runs it.
    """
    from scipy.signal import find_peaks, peak_prominences

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    counts, edges = np.histogram(x, bins=int(bins))
    peaks = find_peaks(counts)[0]
    if peaks.size == 0:
        # The published code would raise IndexError here. Returning the
        # midpoint between zero and the maximum is what its docstring promises
        # as the fallback, and a refusal is not available to a caller that has
        # already decoded a video, so the documented fallback is used and the
        # divergence is recorded here rather than hidden.
        return float(0.5 * np.max(x))
    prom = peak_prominences(counts, peaks)
    idx = find_peaks(counts, prominence=np.max(prom))[0]
    if idx.size == 0:
        return float(0.5 * np.max(x))
    bin_index = int(idx[0])
    if str(mode) == "bin_edge":
        return float(edges[min(bin_index, len(edges) - 1)])
    # The published behaviour: the bin INDEX, used as a signal threshold.
    return float(bin_index)


def trim_line(values: np.ndarray, edge: str, sensitivity: float) -> float:
    """
    One edge of the region the bulk of the detections occupy.

    This is `detector.get_trim_lines`, reimplemented branch for branch. The
    rule is a QUANTILE-SPACING one, not a robust-spread one: five quantiles are
    taken from the extreme 4% of the distribution -- 0.96, 0.97, 0.98, 0.99,
    1.00 at the top and right, and 0.00 to 0.04 at the bottom and left -- the
    median of three of the four gaps between them is taken as a cutoff, and the
    crop sits that distance in from the most extreme value.

    So it trims a sliver where the detections thin out at an edge, which is why
    FreeClimber keeps about 95% of its spots. An earlier version of this
    function guessed at a median-plus-k-MAD rule and kept 64%, because a median
    plus one MAD discards half a distribution by construction; the error was
    found by comparing against FreeClimber's published output, not by reading.

    The two conditionals are the published ones and are kept even where they
    look odd -- at the bottom edge, `cutoff > threshold` is false whenever the
    cutoff is positive and the boundary is non-negative, so the bottom is
    ordinarily left uncropped. Reproducing the published method means
    reproducing that too; this module is a reference, not an improvement.
    """
    import pandas as pd

    v = pd.Series(np.asarray(values, dtype=float)).dropna()
    if v.empty:
        return 0.0
    quant = 0.0 if edge in ("left", "bottom") else 0.96
    vals = [float(v.quantile(min(1.0, quant + i * 0.01))) for i in range(5)]
    diffs = [abs(vals[i] - vals[i + 1]) for i in range(4)]
    k = float(sensitivity)

    if edge in ("right", "top"):
        boundary = vals[-1]                                   # the maximum
        cutoff = float(np.median(diffs[:-1])) * k
        threshold = boundary - cutoff
        return boundary - cutoff if cutoff < threshold else cutoff

    boundary = vals[0]                                        # the minimum
    cutoff = float(np.median(diffs[1:])) * k
    threshold = boundary + cutoff
    return boundary + cutoff if cutoff > threshold else boundary


def invert_y(y: np.ndarray) -> np.ndarray:
    """
    y measured from the BOTTOM of the region of interest, as FreeClimber
    reports it: `abs(y - y.max())`.

    An image y-axis points down and climbing is upward, so without this every
    position is mirrored and every slope has the wrong sign. It does not show
    up in a comparison of counts, which is how it came to be missing from the
    first version of this module.
    """
    a = np.asarray(y, dtype=float)
    return np.abs(a - np.nanmax(a)) if a.size else a


def bin_vials(x: np.ndarray, n_vials: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign spots to `n_vials` equal-width bins across the x-range of the spots.

    Returns the bin edges and the 1-indexed vial of each spot. FreeClimber
    bins on the range of the SURVIVING (true) spots, not of the frame, so the
    bins follow the apparatus rather than the crop.
    """
    xs = np.asarray(x, dtype=float)
    if xs.size == 0:
        return np.zeros(int(n_vials) + 1), np.zeros(0, dtype=int)
    edges = np.linspace(float(xs.min()), float(xs.max()), int(n_vials) + 1)
    idx = np.clip(np.digitize(xs, edges[1:-1], right=False) + 1, 1, int(n_vials))
    return edges, idx.astype(int)


@dataclass
class FreeClimberResult:
    """Detections and the decisions that produced them."""

    spots: object                      # DataFrame: every located spot
    threshold: float
    n_raw: int
    n_true: int
    bin_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trim: Dict[str, float] = field(default_factory=dict)
    params: Optional[FreeClimberParams] = None

    def counts_per_vial(self, frame: Optional[int] = None) -> Dict[int, int]:
        """How many true spots are in each vial, for one frame or for all."""
        df = self.spots
        sel = df[df["True_particle"]]
        if frame is not None:
            sel = sel[sel["frame"] == int(frame)]
        return {int(v): int(c) for v, c in sel["vial"].value_counts().items()}


def detect(frames: Sequence[np.ndarray],
           p: Optional[FreeClimberParams] = None) -> FreeClimberResult:
    """
    Run the reimplemented FreeClimber detection on a stack of frames.

    `frames` are raw frames in capture order, cropped here rather than by the
    caller, so that the ROI is applied exactly where the published method
    applies it.
    """
    tp = _trackpy()
    import pandas as pd

    p = p or FreeClimberParams()
    grey = to_greyscale_stack(frames, p)
    if grey.size == 0:
        raise ValueError("no frames were supplied")
    lo, hi = int(p.crop_0), int(p.crop_n or grey.shape[0])
    grey = grey[lo:min(hi, grey.shape[0])]
    stack, _bg = subtract_background(grey, p)

    diameter = int(p.diameter)
    if diameter % 2 == 0:
        diameter += 1

    rows = []
    for i in range(stack.shape[0]):
        f = tp.locate(stack[i], diameter=diameter, minmass=float(p.minmass),
                      maxsize=float(p.maxsize))
        if f is None or not len(f):
            continue
        f = f.copy()
        f["frame"] = i
        rows.append(f)
    if not rows:
        raise ValueError(
            "no spots were detected. FreeClimber raises SystemExit here; this "
            "raises so that a caller can report the refusal rather than die.")
    df = pd.concat(rows, ignore_index=True)
    df = df[df["raw_mass"] > 0].sort_values(by="frame").reset_index(drop=True)
    df["t"] = (df["frame"] / float(p.frame_rate)).round(3)

    thr = (auto_signal_threshold(df["signal"].to_numpy())
           if p.threshold == "auto" else float(p.threshold))
    df["True_particle"] = df["signal"] >= thr
    ecc_ok = (df["ecc"] >= float(p.ecc_low)) & (df["ecc"] <= float(p.ecc_high))
    df.loc[df["True_particle"] & ~ecc_ok, "True_particle"] = False

    trim: Dict[str, float] = {}
    if p.trim_outliers and len(df):
        # The published code computes the trim lines from ALL spots and then
        # DROPS the rows outside them, rather than marking them false. Both
        # details matter for reproducing the published row counts.
        trim = {
            "left": trim_line(df["x"].to_numpy(), "left", p.outlier_LR),
            "right": trim_line(df["x"].to_numpy(), "right", p.outlier_LR),
            "bottom": trim_line(df["y"].to_numpy(), "bottom", p.outlier_TB),
            "top": trim_line(df["y"].to_numpy(), "top", p.outlier_TB),
        }
        inside = ((df["x"] >= trim["left"]) & (df["x"] <= trim["right"])
                  & (df["y"] <= trim["top"]) & (df["y"] >= trim["bottom"]))
        df = df[inside].reset_index(drop=True)

    df["vial"] = 0
    if df["True_particle"].any():
        edges, idx = bin_vials(df.loc[df["True_particle"], "x"].to_numpy(),
                               int(p.vials))
        df.loc[df["True_particle"], "vial"] = idx
    else:
        edges = np.zeros(int(p.vials) + 1)

    # y from the bottom of the ROI, which is how FreeClimber reports position
    # and the only orientation in which a climbing slope has the right sign.
    df["y_from_bottom"] = invert_y(df["y"].to_numpy())

    return FreeClimberResult(spots=df, threshold=float(thr), n_raw=int(len(df)),
                             n_true=int(df["True_particle"].sum()),
                             bin_edges=edges, trim=trim, params=p)
