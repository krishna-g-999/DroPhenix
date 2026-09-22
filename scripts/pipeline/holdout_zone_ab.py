# NOTE: preserved from the analysis session. Paths to the video folders and
# the background cache are still absolute; see docs/CLIMBING_BAND_STATUS_2026-09.md
# section 2.2. These are the batch/figure scripts, not yet refactored into the
# module. d15_figures.py and cross_day*.py import d15_run.py from the same dir.
import sys, warnings, re; sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from modules import climbing_video as CV, cylinder_finder as CF, climbing_band as CB
SP = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/D--DroPhenix-DroPhenix-main/f06626e3-0892-4286-ba94-5be2b7280e21/scratchpad")
B = Path(r"C:/Users/gkp11/Videos/HD Climbing assay"); CACHE = SP/"hd_bg"; CACHE.mkdir(exist_ok=True)
POS = {"left":0,"middle":1,"right":2}
h = pd.read_csv(r"D:/DroPhenix/DroPhenix-main/data/validation/hand_count_holdout.csv")
h["hand_count"] = pd.to_numeric(h.hand_count, errors="coerce"); h = h[h.hand_count.notna()]
rows = []
for (vid,pos), grp in h.groupby(["video_filename","cylinder_position"]):
    f = next(B.glob("**/"+vid), None)
    if f is None: continue
    c = CACHE/(f.stem+".npy"); bg = np.load(str(c)) if c.exists() else CV.build_background(str(f))
    if not c.exists(): np.save(str(c), bg)
    v = sorted(CF.locate_cylinders(bg,3).vials, key=lambda a:a.x0)
    if len(v) != 3: continue
    a = v[POS[pos]]
    for zone in (250, None):
        df = CB.analyse_video(str(f), int(a.x0), int(a.x1), 20, [10.,30.,60.], zone_height_px=zone)
        if str(df.status.iloc[0]).startswith("refused"): continue
        for _, r in grp.iterrows():
            m = df[df.time_sec==r.timepoint_sec]
            if m.empty: continue
            rows.append({"video":vid,"pos":pos,"t":r.timepoint_sec,"hand":r.hand_count,
                         "zone":"250px" if zone else "full","rep":m.reported_count.iloc[0]})
    print(vid, pos, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv(SP/"holdout_zone_ab.csv", index=False)
print("Day-3/7 HOLD-OUT, same geometry, zone 250 px vs full tube:")
for z, s in d.groupby("zone"):
    e = s.rep - s.hand
    print("  zone %-5s n=%d  MAE %.2f  bias %+.2f  within3 %.0f%%" % (z, len(s), e.abs().mean(), e.mean(), 100*(e.abs()<=3).mean()))
