# NOTE: preserved from the analysis session. Paths to the video folders and
# the background cache are still absolute; see docs/CLIMBING_BAND_STATUS_2026-09.md
# section 2.2. These are the batch/figure scripts, not yet refactored into the
# module. d15_figures.py and cross_day*.py import d15_run.py from the same dir.
"""15th-Day set: parse the operator's manual counts, score every cylinder, compare.

A pure hold-out. Nothing in climbing_band has seen these recordings or counts.

Spreadsheet layout ('15 Day' sheet): blocks of three videos side by side, each
block a header row naming the video, then 'SET n | Wildtype | Htt | CYO', then
twelve time-window rows. Windows are labelled inconsistently ('100-120' and
'110-120' both appear) so time is assigned by ROW ORDER within a block:
row k -> t = 10*(k+1) s, i.e. the reading at the END of each 10 s window.
Position: left = Wildtype, middle = Htt, right = CYO, read from the stickers.
"""
import sys, re, warnings, time, json
sys.path.insert(0, r"D:\DroPhenix\DroPhenix-main")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, openpyxl
from pathlib import Path
from modules import climbing_video as CV, cylinder_finder as CF, climbing_band as CB

SP = Path(r"C:/Users/gkp11/AppData/Local/Temp/claude/"
          r"D--DroPhenix-DroPhenix-main/f06626e3-0892-4286-ba94-5be2b7280e21/scratchpad")
D = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/15th Day")
OUT = Path(r"C:/Users/gkp11/Videos/HD Climbing assay/_DroPhenix_analysis")
CACHE = SP / "d15_bg"
GENO_POS = {"Wildtype": "left", "Htt": "middle", "CYO": "right"}
TIMES = [float(t) for t in range(10, 130, 10)]


def parse_hand(xlsx):
    ws = openpyxl.load_workbook(xlsx, data_only=True)["15 Day"]
    grid = [list(r) for r in ws.iter_rows(values_only=True)]
    vid_re = re.compile(r"C[O0](\d{3})")
    rows, header, video_at, k_at = [], None, {}, {}
    for r in grid:
        cells = [(ci, c) for ci, c in enumerate(r) if isinstance(c, str)]
        if any(vid_re.search(c.replace(" ", "")) and "SET" not in c for _, c in cells):
            header = r; video_at, k_at = {}, {}
            continue
        if any(c.strip().upper().startswith("SET") for _, c in cells):
            continue                       # decorative; the header row is the anchor
        for ci, c in cells:
            if re.match(r"^\s*\d+\s*-{1,2}\s*\d+\s*$", c) and header is not None:
                if ci not in video_at:
                    # the video name sits in the header within a column or two of
                    # this time-label column (C0665 is one to the right of its
                    # labels; C0666 is directly above; CO698 has no SET row at all)
                    for dj in (1, 0, 2, -1):
                        cj = ci + dj
                        if 0 <= cj < len(header) and isinstance(header[cj], str):
                            m = vid_re.search(header[cj].replace(" ", ""))
                            if m:
                                video_at[ci] = "C0" + m.group(1) + ".MP4"; k_at[ci] = 0
                                break
                if ci not in video_at:
                    continue
                k = k_at[ci]; t = 10.0 * (k + 1); k_at[ci] = k + 1
                for jdx, g in enumerate(("Wildtype", "Htt", "CYO")):
                    val = r[ci + 1 + jdx] if ci + 1 + jdx < len(r) else None
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                        rows.append({"video": video_at[ci], "genotype": g,
                                     "position": GENO_POS[g], "time_sec": t,
                                     "hand_count": int(val)})
    return pd.DataFrame(rows).drop_duplicates(["video", "genotype", "time_sec"])


def main():
    hand = parse_hand(D / "Manual counting.xlsx")
    present = {f.name for f in D.glob("**/*.MP4")}
    hand = hand[hand.video.isin(present)]
    hand.to_csv(OUT / "d15_hand_counts.csv", index=False)
    print("hand counts parsed: %d rows, %d videos, %d cylinders"
          % (len(hand), hand.video.nunique(), hand.groupby(["video", "position"]).ngroups))
    print("  per video:", hand.groupby("video").size().to_dict(), flush=True)

    rows, t0 = [], time.time()
    for f in sorted(D.glob("**/*.MP4")):
        cond = f.parent.name
        bg = np.load(str(CACHE / (f.stem + ".npy")))
        info = CV.probe_video(str(f))
        r = CF.locate_cylinders(bg, 3)
        vials = sorted(r.vials, key=lambda v: v.x0)
        if len(vials) != 3:
            print(f.name, "geometry refused"); continue
        tag = "%-10s %s" % (f.name, cond)
        outs = []
        for pos, v in zip(("left", "middle", "right"), vials):
            geno = {"left": "Wildtype", "middle": "Htt", "right": "CYO"}[pos]
            try:
                df = CB.analyse_video(str(f), int(v.x0), int(v.x1), 20, TIMES)
            except Exception as exc:
                rows.append({"video": f.name, "condition": cond, "position": pos,
                             "genotype": geno, "status": "refused: %r" % exc}); outs.append("err"); continue
            if str(df.status.iloc[0]).startswith("refused"):
                rows.append({"video": f.name, "condition": cond, "position": pos,
                             "genotype": geno, "status": df.status.iloc[0]}); outs.append("no-band"); continue
            for _, q in df.iterrows():
                rows.append({"video": f.name, "condition": cond, "position": pos,
                             "genotype": geno, "time_sec": q.time_sec, "band_y": q.band_y,
                             "tube_w_px": int(v.x1 - v.x0),
                             "reported_count": q.reported_count,
                             "count_lower": q.count_lower, "count_upper": q.count_upper,
                             "bracket_width": q.bracket_width, "count_below": q.count_below,
                             "accounted_for": q.accounted_for, "status": "scored"})
            outs.append("%.0f" % df.reported_count.iloc[0])
        print("%s  t10 %-16s (%.1f min)" % (tag, outs, (time.time() - t0) / 60), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "d15_tool_fullzone.csv", index=False)

    tool = pd.DataFrame(rows); tool.to_csv(OUT / "d15_tool_fullzone.csv", index=False)
    sc = tool[tool.status == "scored"]
    m = hand.merge(sc.drop(columns=["genotype"]), on=["video", "position", "time_sec"], how="inner")
    m["err"] = m.reported_count - m.hand_count
    m.to_csv(OUT / "d15_paired_fullzone.csv", index=False)
    print("\n" + "=" * 70)
    print("15th DAY HOLD-OUT: tool vs %d hand counts" % len(m))
    print("=" * 70)
    print("  MAE %.2f   bias %+.2f   within 3 flies %.0f%%" %
          (m.err.abs().mean(), m.err.mean(), 100 * (m.err.abs() <= 3).mean()))
    print("  bracket contains hand count: %.0f%%" %
          (100 * ((m.hand_count >= m.count_lower) & (m.hand_count <= m.count_upper)).mean()))
    print("\n  by genotype (position):")
    for g, s in m.groupby(["genotype", "position"]):
        print("    %-9s %-7s n=%3d  MAE %5.2f  bias %+.2f" %
              (g[0], g[1], len(s), s.err.abs().mean(), s.err.mean()))
    print("\n  by condition:")
    for c, s in m.groupby("condition"):
        print("    %-18s n=%3d  MAE %5.2f  bias %+.2f" % (c, len(s), s.err.abs().mean(), s.err.mean()))
    print("\n  by time:")
    for t, s in m.groupby("time_sec"):
        print("    t=%3.0f  MAE %5.2f  bias %+.2f" % (t, s.err.abs().mean(), s.err.mean()))
    print("\n  hand-count max per cylinder = LOWER BOUND on N (N not recorded):")
    lb = hand.groupby(["video", "genotype"]).hand_count.max().unstack()
    print(lb.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
