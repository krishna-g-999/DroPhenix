# NOTE: preserved from the analysis session. Paths to the video folders and
# the background cache are still absolute; see docs/CLIMBING_BAND_STATUS_2026-09.md
# section 2.2. These are the batch/figure scripts, not yet refactored into the
# module. d15_figures.py and cross_day*.py import d15_run.py from the same dir.
"""Score every cylinder in the HD set with the committed band-and-zone estimator.

ACCURACY OF WHAT THIS PRODUCES: hold-out MAE 6.53 flies, bias -3.33, against a
3-fly assay noise floor (data/validation/hand_count_holdout.csv). Roughly
twice as noisy as a human count. Nothing here is publication-grade.

DENOMINATOR: the operator stated 20 per cylinder; hand counts on 9 cylinders
gave 14-25. `reported_count` is denominator-free. `pct_nominal` divides by 20
and is labelled nominal. `pct_known` divides by the operator's count where one
exists and is blank otherwise.
"""
import sys, warnings, re, time, json
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from modules import climbing_video as CV, cylinder_finder as CF, climbing_band as CB

SP = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/"
          r"D--DroPhenix-DroPhenix-main/f06626e3-0892-4286-ba94-5be2b7280e21/scratchpad")
OUT = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/_DroPhenix_analysis")
OUT.mkdir(exist_ok=True)
CACHE = SP / "hd_bg"; CACHE.mkdir(exist_ok=True)
B = Path(r"C:/Users/gkp11/Videos/HD Climbing assay")
N_NOMINAL = 20
TIMES = [float(t) for t in range(10, 130, 10)]
POSNAME = {1: "left", 2: "middle", 3: "right"}


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
    r = CF.locate_cylinders(bg, 3)
    if _sane(r.vials, width):
        return sorted(r.vials, key=lambda v: v.x0), "band-locator"
    g = CV.detect_vials(str(path), n_vials=3, background=bg)
    gv = [CF.BandVial(x0=int(v.x0), x1=int(v.x1), band_row=0, n_rows=0, darkness=0.0)
          for v in (g.get("vials") or [])]
    if _sane(gv, width):
        return sorted(gv, key=lambda v: v.x0), "glass-locator"
    return [], ""


def known_n():
    out = {}
    for fn in ("hand_count_holdout.csv", "hand_count_template.csv"):
        p = Path(r"D:\DroPhenix\DroPhenix-main\data\validation") / fn
        if not p.exists():
            continue
        h = pd.read_csv(p)
        for _, r in h.iterrows():
            m = re.search(r"(\d+)", str(r.get("notes", "")))
            if m and "total" in str(r.get("notes", "")).lower():
                pos = str(r.cylinder_position).lower()
                pos = {"p1": "left", "p2": "middle", "p3": "right"}.get(pos, pos)
                out[(r.video_filename, pos)] = int(m.group(1))
    return out


def main():
    nk = known_n()
    vids = sorted(B.glob("**/*.MOV"))
    rows, t0 = [], time.time()
    for k, f in enumerate(vids, 1):
        day = "Day 3" if "3 rd day" in str(f) else "Day 7"
        sub = f.parent.name if f.parent.name not in ("3 rd day", "7 th day") else ""
        base = {"day": day, "subfolder": sub, "video": f.name}
        tag = "[%3d/%d] %s %-14s" % (k, len(vids), day, f.name)
        try:
            info = CV.probe_video(str(f))
            c = CACHE / (f.stem + ".npy")
            bg = np.load(str(c)) if c.exists() else CV.build_background(str(f))
            if not c.exists():
                np.save(str(c), bg)
        except Exception as exc:
            rows.append({**base, "status": "refused: unreadable (%r)" % exc})
            print(tag + " REFUSED unreadable", flush=True)
            continue
        vials, how = geometry_for(f, bg, info.width)
        if not vials:
            rows.append({**base, "status": "refused: cylinders not located"})
            print(tag + " REFUSED geometry", flush=True)
            continue
        line = []
        for i, v in enumerate(vials, 1):
            pos = POSNAME[i]
            N_known = nk.get((f.name, pos))
            try:
                df = CB.analyse_video(str(f), int(v.x0), int(v.x1), N_NOMINAL, TIMES)
            except Exception as exc:
                rows.append({**base, "position": pos, "status": "refused: %r" % exc})
                line.append("err")
                continue
            if str(df.status.iloc[0]).startswith("refused"):
                rows.append({**base, "position": pos, "status": df.status.iloc[0]})
                line.append("no-band")
                continue
            for _, r in df.iterrows():
                cnt = float(r.reported_count)
                rows.append({
                    **base, "position": pos, "time_sec": r.time_sec,
                    "band_y": r.band_y, "tube_w_px": int(v.x1 - v.x0),
                    "reported_count": cnt,
                    "count_lower": r.count_lower, "count_upper": r.count_upper,
                    "bracket_width": r.bracket_width,
                    "count_below": r.count_below, "accounted_for": r.accounted_for,
                    "n_nominal": N_NOMINAL,
                    "pct_nominal": round(100 * cnt / N_NOMINAL, 1),
                    "n_known": N_known,
                    "pct_known": (round(100 * cnt / N_known, 1) if N_known else np.nan),
                    "geometry": how, "status": "scored"})
            line.append("%d" % df.reported_count.iloc[0])
        print(tag + " t10 counts %-18s (%.1f min)" % (line, (time.time() - t0) / 60),
              flush=True)
        pd.DataFrame(rows).to_csv(OUT / "hd_all_vials.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "hd_all_vials.csv", index=False)
    (OUT / "hd_all_vials.provenance.json").write_text(json.dumps({
        "produced_by": "hd_full_run.py", "estimator": "climbing_band bracket midpoint",
        "holdout_accuracy": "MAE 6.53 flies, bias -3.33, n=30, vs 3-fly noise floor",
        "usable_for_science": False,
        "denominator": "nominal 20; true N is 14-25 where counted (9 cylinders)",
        "timepoints_s": TIMES, "t_zero": "first frame",
        "known_registration_defect": "camera moves 3-7 px per take; not corrected here"},
        indent=2), encoding="utf-8")
    print("\nwrote", OUT / "hd_all_vials.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
