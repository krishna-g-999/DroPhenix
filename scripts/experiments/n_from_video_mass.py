"""EXPERIMENT (2026-09-15): estimate N per cylinder from the video, by mass.

RESULT: does not beat assuming the stated loading. Kept as a record.

    N-hat from video      MAE 9.10   bias +6.69   (8 cylinders, true N 14-25)
    "assume 20"           MAE 2.62

What was learned, each step measured against the operator's per-cylinder
loadings from data/validation/hand_count_holdout.csv (never used for tuning):

    raw percentile background              MAE 206.8   ink erased, frame 0 flood
    + skip t<2s, mass from comps >= 12 px  MAE  70.0
    + static-ink mask (dilated)            MAE  63.8   small: ink was not the bulk
    + PER-FRAME REGISTRATION               MAE  11.8   every Day-7 video moves 3-7 px
    + pooled fly-size across the 3 tubes   MAE   8.95  m1 stable at 25-31 px
    + conservation outlier rule            MAE   9.10  no help: floods are epochs

Remaining, unfixed:
  * motion blur: fast wild-type flies smear ~40% larger than the resting flies
    the calibration comes from, so wild-type tubes over-count by about that
  * sustained external disturbances: IMG_6182 left reads 30-74 in a 14-fly tube
    for 35 s (t=60-94) then returns to ~15. Not flies. Needs epoch detection.
  * the registration reference at t=2 s is itself the disturbed frame
  * severe crowding never disperses enough for the mass to reach N

The transferable finding is REGISTRATION: it was the dominant error here and
the production counter does not register either.
"""

import sys, warnings, importlib.util, re
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from pathlib import Path
from scipy import ndimage
from modules import climbing_video as CV, climbing_band as CB

SP = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/"
          r"D--DroPhenix-DroPhenix-main/f06626e3-0892-4286-ba94-5be2b7280e21/scratchpad")
from modules import cylinder_finder as CF


def _sane(vials, w):
    if len(vials) != 3:
        return False
    xs = sorted(vials, key=lambda v: v.x0)
    wd = np.array([v.x1 - v.x0 for v in xs], float)
    if min(v.x0 for v in xs) < 5 or max(v.x1 for v in xs) > w - 5:
        return False
    if wd.std() / max(wd.mean(), 1e-9) > 0.25:
        return False
    g = np.diff([0.5 * (v.x0 + v.x1) for v in xs])
    return abs(g[0] - g[1]) / max(g.mean(), 1e-9) <= 0.35


def geometry_for(path, bg, width):
    """Per-tube x-extents. cylinder_finder first: its widths are per tube, where
    the glass locator returns one constant width per video and over-measures
    1.31x (which scaled the fly diameter 31% too large)."""
    r = CF.locate_cylinders(bg, 3)
    if _sane(r.vials, width):
        return sorted(r.vials, key=lambda v: v.x0)
    g = CV.detect_vials(str(path), n_vials=3, background=bg)
    gv = [CF.BandVial(x0=int(v.x0), x1=int(v.x1), band_row=0, n_rows=0, darkness=0.0)
          for v in (g.get("vials") or [])]
    if _sane(gv, width):
        return sorted(gv, key=lambda v: v.x0)
    return []


class _HD:  # keep the call site unchanged
    geometry_for = staticmethod(lambda f, bg, w: (geometry_for(f, bg, w), "", ""))
hd = _HD()
B = Path(r"C:/Users/gkp11/Videos/HD Climbing assay")
POS = {"left": 0, "middle": 1, "right": 2}

STEP_S = 2.0          # sampling grid over the recording
T_START = 2.0         # frame 0 carried 31,512 px of foreground against ~2,000
                      # elsewhere: the tap disturbance. Skip the first seconds.
BG_PCT = 90           # brightness percentile = "no fly here"
FG_TAU = 30           # grey levels darker than background to count as fly
MIN_CC = 6            # px; smaller components are noise (histogram only)
MASS_MIN_CC = 12      # px; a component must be ~half a fly to contribute MASS.
                      # Edge flicker on graduations and tube walls produced 166
                      # fragments of ~5 px per frame -- 3x the real fly mass.
N_QUANT = 90          # robust high quantile for N-hat
INK_FRAC = 0.60       # dark in >= this fraction of frames -> static ink, not a fly
INK_DILATE = 2        # px; jitter amplitude. A tick wobbles +/-1 px between frames.


def ink_mask(strip, bg):
    """Pixels that are dark in most frames are graduations, the band, labels.

    The 90th-percentile background ERASED thin static ink: a tick under sub-pixel
    jitter is absent from any one pixel in some frames, the high percentile takes
    the bright value, and the tick returns as foreground in every frame. Seen
    directly on IMG_6179: a solid red column down every graduation, the band
    edge, and the R-label. Masking what is dark in >= 60% of frames, dilated by
    the jitter, removes ink while leaving a fly -- which does not hold one pixel
    for 60% of two minutes -- alone. That claim is measured below, not assumed.
    """
    b = bg.astype(np.float32)
    dark = np.zeros(bg.shape, np.int32)
    for fr in strip:
        f = fr.astype(np.float32); gain = np.median(b) / max(np.median(f), 1.0)
        dark += (np.clip(b - f * gain, 0, None) > FG_TAU)
    m = (dark >= INK_FRAC * len(strip))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INK_DILATE + 1,) * 2)
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def register(strips):
    """Rigidly align every frame to the first, by phase correlation.

    Measured on IMG_6179: the frame holds within 0.2 px for 50 s, then steps
    +2.8 px and stays -- the bench or camera was nudged. A background built
    from both halves puts every tube wall 3 px off in half the frames, and a
    displaced wall is a continuous 800-row foreground stripe that no size or
    ink filter can reject. Translation is the whole of the problem, so a
    translation is the whole of the fix. Returns aligned strips and the shifts.
    """
    ref = strips[0].astype(np.float32)
    out, shifts = [], []
    h, w = ref.shape
    for fr in strips:
        cur = fr.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ref, cur)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        out.append(cv2.warpAffine(fr, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
        shifts.append((dx, dy))
    return out, np.array(shifts)


def percentile_background(frames):
    """Per-pixel high brightness percentile. Flies are dark; ink is static."""
    return np.percentile(np.stack(frames), BG_PCT, axis=0).astype(np.uint8)


def foreground(frame, bg, ink=None):
    """Darkening below background, gain-corrected, ink removed, fragments removed."""
    f = frame.astype(np.float32); b = bg.astype(np.float32)
    gain = np.median(b) / max(np.median(f), 1.0)     # flies are few pixels; median is glass
    m = (np.clip(b - f * gain, 0, None) > FG_TAU)
    if ink is not None:
        m &= ~ink
    lab, n = ndimage.label(m)
    if n:
        a = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = a >= MASS_MIN_CC
        m = keep[lab]
    return m


def one_fly_area(masks):
    """Mode of connected-component area across all sampled frames.

    Isolated flies are the most frequent component during dispersion, so the
    histogram peak is one fly. Log-binned, because areas are right-skewed by
    clumps and we want the peak, not the mean.
    """
    areas = []
    for m in masks:
        lab, n = ndimage.label(m)
        if n:
            a = ndimage.sum(m, lab, index=np.arange(1, n + 1))
            areas.extend(float(x) for x in a if x >= MIN_CC)
    areas = np.array(areas)
    if areas.size < 10:
        return np.nan, areas
    la = np.log(areas)
    hist, edges = np.histogram(la, bins=40)
    hist = np.convolve(hist, np.ones(3) / 3, mode="same")   # light smoothing
    k = int(np.argmax(hist))
    return float(np.exp(0.5 * (edges[k] + edges[k + 1]))), areas


def tube_masks(video, x0, x1):
    """Registered, background-subtracted, ink-masked masks for one tube."""
    info = CV.probe_video(str(video))
    grid = np.arange(T_START, info.duration_s - 0.5, STEP_S)
    frames = CV._read_frames(str(video), [int(t * info.fps) for t in grid])
    strip = [f[:, x0:x1] for f in frames]
    strip, shifts = register(strip)
    bg = percentile_background(strip)
    band = CB.find_black_band(bg, 0, x1 - x0)
    ink = ink_mask(strip, bg)
    masks = [foreground(s, bg, ink) for s in strip]
    return dict(masks=masks, band=band, grid=grid, shifts=shifts, ink=ink)


def estimate(video, x0, x1, m1=None, pre=None):
    """N-hat for one tube. `m1` may be supplied from a per-video pool: the three
    tubes in one recording share a camera distance, so they share a fly size,
    and the pooled histogram cannot be dragged to a half-fly by one tube's
    fragments the way IMG_6179 right was (mode 16 px, N-hat doubled to 62)."""
    d = pre or tube_masks(video, x0, x1)
    masks, band, grid, shifts = d["masks"], d["band"], d["grid"], d["shifts"]
    if m1 is None:
        m1, _ = one_fly_area(masks)
    if not np.isfinite(m1):
        return dict(N_hat=np.nan, m1=np.nan, band=band, note="no components")
    total = np.array([m.sum() / m1 for m in masks])
    above = np.array([m[:band].sum() / m1 for m in masks]) if band else np.full_like(total, np.nan)
    # CONSERVATION AS AN OUTLIER RULE. The mass in a sealed tube cannot change
    # by a large factor between samples 2 s apart -- flies do not teleport and
    # occlusion changes gradually. A sample far above its neighbours is a
    # registration miss, not animals: IMG_6182 left read median 15.1 against a
    # true 14 with single frames at 74. Those frames are dropped before N-hat.
    roll = pd.Series(total).rolling(5, center=True, min_periods=2).median().to_numpy()
    keep = total <= 1.5 * roll
    clean = total[keep] if keep.sum() >= 5 else total
    return dict(N_hat=float(np.percentile(clean, N_QUANT)),
                n_dropped=int((~keep).sum()),
                N_med=float(np.median(total)), N_max=float(total.max()),
                m1=m1, band=band, n_frames=len(masks), ink_px=int(d["ink"].sum()),
                max_shift=float(np.abs(shifts).max()),
                total_series=total, above_series=above, t=grid[:len(total)])


def _estimate_old(video, x0, x1):
    info = CV.probe_video(str(video))
    grid = np.arange(T_START, info.duration_s - 0.5, STEP_S)
    frames = CV._read_frames(str(video), [int(t * info.fps) for t in grid])
    strip = [f[:, x0:x1] for f in frames]
    strip, shifts = register(strip)
    bg = percentile_background(strip)
    band = CB.find_black_band(bg, 0, x1 - x0)
    ink = ink_mask(strip, bg)
    masks = [foreground(s, bg, ink) for s in strip]
    m1, areas = one_fly_area(masks)
    if not np.isfinite(m1):
        return dict(N_hat=np.nan, m1=np.nan, band=band, note="no components")
    total = np.array([m.sum() / m1 for m in masks])
    above = np.array([m[:band].sum() / m1 for m in masks]) if band else np.full_like(total, np.nan)
    # CONSERVATION AS AN OUTLIER RULE. The mass in a sealed tube cannot change
    # by a large factor between samples 2 s apart -- flies do not teleport and
    # occlusion changes gradually. A sample far above its neighbours is a
    # registration miss, not animals: IMG_6182 left read median 15.1 against a
    # true 14 with single frames at 74. Those frames are dropped before N-hat.
    roll = pd.Series(total).rolling(5, center=True, min_periods=2).median().to_numpy()
    keep = total <= 1.5 * roll
    clean = total[keep] if keep.sum() >= 5 else total
    return dict(N_hat=float(np.percentile(clean, N_QUANT)),
                n_dropped=int((~keep).sum()),
                N_med=float(np.median(total)), N_max=float(total.max()),
                m1=m1, band=band, n_frames=len(masks), ink_px=int(ink.sum()),
                max_shift=float(np.abs(shifts).max()),
                total_series=total, above_series=above,
                t=grid[:len(total)])


def main():
    h = pd.read_csv(r"D:\DroPhenix\DroPhenix-main\data\validation\hand_count_holdout.csv")
    truths = {}
    for _, r in h.iterrows():
        m = re.search(r"(\d+)", str(r.notes))
        if m and "total" in str(r.notes).lower():
            truths[(r.video_filename, r.cylinder_position)] = int(m.group(1))
    bound = [(r.video_filename, r.cylinder_position) for _, r in h.iterrows()
             if "more than" in str(r.notes).lower()]

    rows = []
    pooled, cache = {}, {}
    for (vid, pos), N in list(truths.items()) + [(k, None) for k in bound]:
        f = next(B.glob("**/" + vid), None)
        if f is None: continue
        if vid not in pooled:
            bgp = SP / ("hd_bg/" + f.stem + ".npy")
            bg0 = np.load(str(bgp)) if bgp.exists() else CV.build_background(str(f))
            info = CV.probe_video(str(f))
            vials, _, _ = hd.geometry_for(f, bg0, info.width)
            if len(vials) < 3:
                pooled[vid] = None
            else:
                cache[vid] = [tube_masks(f, int(v.x0), int(v.x1)) for v in vials]
                allm = [m for d in cache[vid] for m in d["masks"]]
                pooled[vid], _ = one_fly_area(allm)
        bgp = SP / ("hd_bg/" + f.stem + ".npy")
        if bgp.exists():
            bg0 = np.load(str(bgp))
        else:
            bg0 = CV.build_background(str(f)); bgp.parent.mkdir(exist_ok=True)
            np.save(str(bgp), bg0)
        info = CV.probe_video(str(f))
        vials, _, _ = hd.geometry_for(f, bg0, info.width)
        if len(vials) < 3:
            print("  %-13s %-7s geometry refused" % (vid, pos)); continue
        v = vials[POS[pos]]
        e = estimate(f, int(v.x0), int(v.x1), m1=pooled.get(vid), pre=cache[vid][POS[pos]])
        rows.append({"video": vid, "pos": pos, "N_true": N, "N_hat": e["N_hat"],
                     "N_med": e.get("N_med"), "N_max": e.get("N_max"),
                     "m1_px": e["m1"], "band": e["band"], "w": int(v.x1 - v.x0),
                     "max_shift": e.get("max_shift")})
        print("  %-13s %-7s N_true %-4s  N_hat %5.1f  (median %5.1f, max %5.1f)  "
              "one-fly %4.0f px  shift %.1f px  dropped %d" % (
                  vid, pos, N if N else ">20", e["N_hat"], e.get("N_med", np.nan),
                  e.get("N_max", np.nan), e["m1"], e.get("max_shift", 0),
                  e.get("n_dropped", 0)), flush=True)
    d = pd.DataFrame(rows); d.to_csv(SP / "proto_N.csv", index=False)
    ok = d.dropna(subset=["N_true"])
    err = ok.N_hat - ok.N_true
    print("\n  N-hat vs true loading, %d cylinders" % len(ok))
    print("  MAE %.2f   bias %+.2f   within 2 flies: %d/%d   within 3: %d/%d"
          % (err.abs().mean(), err.mean(), (err.abs() <= 2).sum(), len(ok),
             (err.abs() <= 3).sum(), len(ok)))
    print("  correlation r = %.3f" % np.corrcoef(ok.N_hat, ok.N_true)[0, 1])
    bd = d[d.N_true.isna()]
    if len(bd):
        print("  bound check (true >20): N_hat = %s" % bd.N_hat.round(1).tolist())
    print("\n  reference: the stated N was 20 for every cylinder -> MAE %.2f"
          % (ok.N_true - 20).abs().mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
