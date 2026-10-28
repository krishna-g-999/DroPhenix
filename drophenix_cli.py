#!/usr/bin/env python
"""
DroPhenix command-line interface — score climbing assays without a browser.

WHY A COMMAND LINE, WHEN THERE IS ALREADY A WEB APP
---------------------------------------------------
The web app is the right place to explore a dataset. It is the wrong place to
process video, for four reasons that are properties of the problem rather than
of Streamlit:

  SIZE      One recording of this assay is 120 MB and a session is nearly two
            gigabytes. Uploading that to a hosted app is slow, often blocked by
            upload limits, and pointless when the file is already on the machine
            doing the work.
  DURATION  Scoring fifteen recordings takes about forty-five minutes. Any
            browser-based framework re-runs its script on interaction and hosted
            instances time out; long jobs do not belong behind a web request.
  PRIVACY   Unpublished data leaves the laboratory the moment it is uploaded.
            Here the video never moves.
  METHODS   A command is quotable. "We scored the recordings with
            `drophenix_cli.py batch --manifest ... --height-mm 80`" is a
            reproducible methods sentence; "we clicked through a web form" is
            not.

THE RECOMMENDED SPLIT
---------------------
Do the heavy, private, slow part locally with this tool, which emits a small
CSV; then use the app -- or R, or a notebook, or anything -- for the statistics
on that CSV. The video is gigabytes; the scored table is kilobytes.

    drophenix_cli.py manifest --folder ./videos > manifest.csv
    # fill in Genotype / Treatment / Sex / n_per_vial
    drophenix_cli.py batch --folder ./videos --manifest manifest.csv \
        --vials 3 --diameter-mm 25 --height-mm 80 --out results/
    # results/climbing_table.csv now feeds the statistics

COMMANDS
--------
  plan-dam   plan an activity-monitor run BEFORE loading the flies
  dam-run    raw monitor files + channel map -> quality, tests, figures, verified report
  card       print a calibration card, so scale is measured not remembered
  setup-check  check a TEST recording against the protocol, before the real run
  triage     WHAT QUESTIONS CAN THIS DATA ANSWER? run this before anything else
  manifest   write a starter manifest for a folder of recordings
  inspect    examine ONE recording: regions, scoring lines, tap, framing, plus
             annotated frames to look at before trusting anything
  batch      score a folder against a manifest into one analysis-ready table
  check      run the design and clustering diagnostics on a scored CSV

Run any command with --help for its options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

VIDEO_EXT = (".mp4", ".mov", ".avi", ".m4v", ".mkv")


def _eprint(*a):
    print(*a, file=sys.stderr, flush=True)


def _find_videos(folder: str):
    p = Path(folder)
    if not p.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    return sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)


def _calibration(args):
    from modules import climbing_video as CV
    if getattr(args, "px_per_mm", None):
        return CV.Calibration(px_per_mm=args.px_per_mm, source="stated directly")
    if getattr(args, "diameter_mm", None):
        return CV.Calibration(tube_width_mm=args.diameter_mm,
                              source="cylinder outside diameter")
    return None


def _rule(args, calibrated: bool):
    from modules import climbing_video as CV
    if getattr(args, "height_mm", None):
        if not calibrated:
            raise SystemExit(
                "--height-mm needs a calibration. Add --diameter-mm (25 for a 50 ml "
                "graduated cylinder, 23.5 for a standard Drosophila vial) or --px-per-mm."
            )
        return CV.ScoringRule("height_mm", height_mm=args.height_mm)
    if getattr(args, "fraction", None):
        return CV.ScoringRule("height_fraction", fraction=args.fraction)
    return CV.ScoringRule("drawn_line")


# ─────────────────────────────────────────────────────────────────────────────
def cmd_manifest(args) -> int:
    vids = _find_videos(args.folder)
    if not vids:
        raise SystemExit(f"No video files found in {args.folder}")
    man = pd.DataFrame({
        "video": [v.name for v in vids],
        "Genotype": "", "Treatment": "", "Sex": "",
        "Age": "", "TimeOfDay": "", "ExperimentID": "", "n_per_vial": "",
    })
    out = args.out
    if out:
        man.to_csv(out, index=False)
        _eprint(f"Wrote {out} with {len(man)} rows. Fill in Genotype, Treatment, Sex "
                f"and n_per_vial before running `batch`.")
    else:
        man.to_csv(sys.stdout, index=False)
    return 0


def cmd_inspect(args) -> int:
    from modules import climbing_video as CV
    import cv2

    path = args.video
    info = CV.probe_video(path)
    print(f"file      : {Path(path).name}")
    print(f"format    : {info.width}x{info.height}, {info.fps:.2f} fps, "
          f"{info.duration_s:.1f} s, codec {info.codec}")

    taps = CV.detect_taps(path)
    print(f"taps      : {taps.get('n_taps', 0)}  "
          f"({', '.join('%.2fs' % t['t0_s'] for t in taps.get('taps', [])) or 'none'})")
    print(f"protocol  : {taps.get('protocol', '')}")
    if not taps.get("framing_stable", True):
        print("WARNING   : the camera moved during this recording; trim it to one framing.")

    bg = CV.build_background(path)
    det = CV.detect_vials(path, n_vials=args.vials, background=bg)
    print(f"cylinders : {len(det['vials'])}  confidence {det['confidence']:.2f}"
          f"{'  NEEDS REVIEW' if det['needs_review'] else ''}")
    if det.get("reason"):
        print(f"            {det['reason']}")

    cal = _calibration(args)
    for v in det["vials"]:
        ppm = cal.resolve(v) if cal else None
        extra = f", {v.height / ppm:.0f} mm tall, {ppm:.2f} px/mm" if ppm else ""
        print(f"  {v.label}: x {v.x0}-{v.x1}, y {v.y_top}-{v.y_bottom}, "
              f"line at row {v.line_y} ({v.line_fraction():.0%} up){extra}")

    if det["vials"] and args.annotate:
        outdir = Path(args.annotate)
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = taps["taps"][0]["t0_s"] if taps.get("taps") else 0.0
        for t in args.times:
            try:
                img = CV.annotate_frame(path, det["vials"], bg, at_time_s=t, t0_s=t0)
                f = outdir / f"{Path(path).stem}_t{int(t):03d}.jpg"
                cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"  wrote {f}")
            except Exception as exc:
                _eprint(f"  could not annotate t={t}: {exc}")
    return 0


def cmd_batch(args) -> int:
    from modules import climbing_video as CV

    cal = _calibration(args)
    rule = _rule(args, cal is not None)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        man = pd.read_csv(args.manifest)
    else:
        vids = _find_videos(args.folder)
        man = pd.DataFrame({"video": [v.name for v in vids],
                            "Genotype": "UNSPECIFIED", "Treatment": "UNSPECIFIED",
                            "Sex": "UNSPECIFIED"})
        _eprint("No --manifest given: scoring with placeholder metadata. The counts will "
                "be right but the labels will not; generate one with `manifest` and fill "
                "it in before analysing.")

    n = len(man)
    _eprint(f"Scoring {n} recording(s). Roughly {n * 90 / 60:.0f} minutes at ~90 s each.")
    _eprint(f"Scoring rule: {rule.describe()}"
            + (f"; calibrated via {cal.source}" if cal else "; NOT calibrated"))

    def progress(frac, label):
        _eprint(f"[{frac * 100:5.1f}%] {label}")

    res = CV.analyze_batch(
        man, video_dir=args.folder, n_vials=args.vials,
        scoring_times_s=args.times, calibration=cal, scoring_rule=rule,
        min_confidence=args.min_confidence, progress=progress,
    )

    tab = res["table"]
    if not tab.empty:
        f = outdir / "climbing_table.csv"
        tab.to_csv(f, index=False)
        print(f"\nWrote {f}: {len(tab)} rows, {tab['VialID'].nunique()} vials, "
              f"{tab['SourceVideo'].nunique()} recordings")
    else:
        print("\nNo recording produced usable counts.")

    for name in ("qc", "excluded", "failed"):
        df = res[name]
        if isinstance(df, pd.DataFrame) and not df.empty:
            f = outdir / f"{name}.csv"
            df.to_csv(f, index=False)
            print(f"Wrote {f}: {len(df)} rows")

    print()
    for note in res["notes"]:
        print("*", note)

    if not res["qc"].empty and "reliability" in res["qc"].columns:
        print("\nPer-vial detector reliability across every recording examined:")
        for grade, count in res["qc"]["reliability"].value_counts().items():
            print(f"  {count:3d}  {grade}")
        usable = res["qc"]["reliability"].isin(
            ["good", "acceptable - supply the true total"]).sum()
        print(f"\n{usable} of {len(res['qc'])} cylinders are good or acceptable. If that "
              "fraction is low the limit is usually the recording, not the software -- see "
              "docs/VIDEO_SCORING.md section 6.")
    return 0 if not tab.empty else 1


def cmd_dam_run(args) -> int:
    """Raw monitor files + channel map -> quality, design-aware tests, figures, verified report."""
    import io
    from pathlib import Path as _P
    from modules import dam_pipeline as P, dam_figures as F, dam_report as R

    out = _P(args.out); out.mkdir(parents=True, exist_ok=True)
    design = P.Design(ref_genotype=args.ref_genotype, ref_treatment=args.ref_treatment,
                      stratify_col=None if args.no_stratify else "Sex")

    def one(files, cmap, name):
        lab = P.load_run(files, cmap, monitor_names=[_P(f).stem for f in files])
        q = P.quality_report(lab)
        print(f"[{name}] bin {q['bin_min']:.0f} min; photoperiod "
              f"{(q['photoperiod_min'] or 0)/60:.1f} h light; usable days {q['n_usable_days']}; "
              f"light channels agree {100*q['agreement']:.1f}%")
        for f in q["faults"]:
            print(f"[{name}] {f}")
        if not q["sleep_scoreable"]:
            print(f"[{name}] REFUSED: bin wider than the 5-min sleep criterion; re-export at 1 min.")
            return None
        if q["window"] is None:
            print(f"[{name}] REFUSED: no usable complete day."); return None
        win = q["window"]
        if args.window:
            a, b = args.window.split(":"); win = (pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(1, "D"))
        run = P.analyse_run(lab, name, design, win, q)
        print(f"[{name}] window {run.window[0].date()}..{(run.window[1]-pd.Timedelta(1, "D")).date()}; "
              f"{len(run.per_fly)} flies; {int(run.per_fly.Outlier.sum())} outliers; "
              f"reference {run.design.ref_genotype} / {run.design.ref_treatment}")
        for k, v in run.summary()["families"].items():
            print(f"[{name}]   {k:26s} {v['n_significant']:2d} of {v['n_tests']:2d} pass FDR 5%")
        return run

    run = one(args.monitors, args.map, args.name)
    if run is None:
        return 1
    run_b = cmp = None
    if args.second:
        run_b = one(args.second, args.second_map or args.map, args.second_name)
        if run_b is None:
            return 1
        cmp = P.compare_runs(run, run_b)
        tp = cmp["timepoint"]; tp = tp[tp.analysis == "primary"]
        print(f"[timepoint] {int(tp.significant.sum())} of {len(tp)} pass FDR 5%; "
              f"tubes paired: {cmp['pairing'].attrs.get('paired')}")

    # tables
    for r in [run] + ([run_b] if run_b else []):
        tag = r.name.replace(" ", "_")
        (out / tag).mkdir(exist_ok=True)
        r.per_fly.to_csv(out / tag / "per_fly.csv", index=False)
        r.quality["days"].to_csv(out / tag / "quality_days.csv", index=False)
        r.profiles.to_csv(out / tag / "profiles_per_fly.csv", index=False)
        for k, df in r.contrasts.items():
            df.to_csv(out / tag / f"contrasts_{k}.csv", index=False)
        for k, df in r.contrasts_noout.items():
            df.to_csv(out / tag / f"contrasts_{k}_nooutliers.csv", index=False)
    if cmp is not None:
        (out / "timepoint").mkdir(exist_ok=True)
        cmp["timepoint"].to_csv(out / "timepoint" / "contrasts_timepoint.csv", index=False)
        cmp["timepoint_noout"].to_csv(out / "timepoint" / "contrasts_timepoint_nooutliers.csv", index=False)
        cmp["pairing"].to_csv(out / "timepoint" / "pairing_test.csv", index=False)

    # figures
    figs = {}
    def png(fig, fn):
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=200); fig.savefig(out / "figures" / fn, dpi=300)
        return b.getvalue()
    (out / "figures").mkdir(exist_ok=True)
    d = run.design
    figs["Primary analysis: every headline effect with its FDR q."] = png(F.headline(run, run_b, cmp), "F1_headline.png")
    figs[f"Sleep profile by genotype, {d.ref_treatment}, {run.name}."] = png(F.profiles(run, by="genotype"), "F2_profiles_genotype.png")
    if run.per_fly[d.treatment_col].nunique() > 1:
        figs[f"Effect of treatment on the sleep profile, {run.name}."] = png(F.profiles(run, by="treatment"), "F3_profiles_treatment.png")
    figs[f"Sleep amount, {run.name}."] = png(F.bars(run, ["Total_sleep_min_per_day", "Day_sleep_min_per_day", "Night_sleep_min_per_day", "Sleep_latency_night_min"], run.name), "F4_bars_sleep.png")
    figs[f"Sleep architecture, {run.name}."] = png(F.bars(run, ["N_bouts_per_day", "Mean_bout_min", "Night_mean_bout_min", "Longest_bout_min"], run.name), "F5_bars_architecture.png")
    figs[f"Activity, {run.name}."] = png(F.bars(run, ["Total_activity_per_day", "Waking_activity", "Day_activity_per_light_h", "Night_activity_per_dark_h"], run.name), "F6_bars_activity.png")
    figs[f"Outlier map, {run.name}."] = png(F.outlier_map(run), "F7_outliers.png")
    if run_b is not None:
        figs[f"Sleep profile by genotype, {run_b.name}."] = png(F.profiles(run_b, by="genotype"), "F8_profiles_genotype_B.png")
        figs[f"{run.name} (dashed) vs {run_b.name} (solid)."] = png(F.profiles(run, by="run", run_b=run_b), "F9_profiles_runs.png")
        figs[f"{run.name} vs {run_b.name} within each group."] = png(F.age_bars(run, run_b, cmp), "F10_bars_runs.png")

    docx = out / "DAM_report.docx"
    R.build(run, str(docx), run_b=run_b, cmp=cmp, figures=figs)
    ok, rep = R.verify(str(docx), run, run_b, cmp)
    print(f"[report] {docx}  --  verification: {rep}")
    return 0 if ok else 1


def cmd_plan_dam(args) -> int:
    """Plan an activity-monitor run so no factor is confounded with the instrument."""
    from modules import dam_planner as DP

    if args.print_protocol:
        print(DP.protocol_text())
        return 0

    if args.check:
        v = DP.check_plan(args.check)
        print("Layout acceptable:", v["ok"])
        for f, st in v["per_factor"].items():
            print(f"  {f}: {st} with Monitor")
        for b in v["blocking"]:
            print("BLOCKING:", b)
        for w in v["warnings"]:
            print("WARNING :", w)
        return 1 if v["blocking"] else 0

    groups = []
    for g in (args.genotype or ["WT"]):
        for sx in (args.sex or [""]):
            for ag in (args.age or [""]):
                for tr in (args.treatment or [""]):
                    groups.append({"Genotype": g, "Sex": sx, "Age": ag, "Treatment": tr})
    plan = DP.plan_run(groups, n_per_group=args.n_per_group,
                       n_monitors=args.monitors, n_channels=args.channels,
                       goal=args.goal, seed=args.seed)
    if not plan.get("ok"):
        print("PLAN REFUSED:", plan.get("reason"))
        return 1
    print(DP.plan_report(plan))
    plan["channel_map"].to_csv(args.out, index=False)
    print()
    print(f"Channel map written to {args.out} -- load the tubes to match it, then pass the")
    print("same file to the analysis so nothing has to be remembered.")
    return 0


def cmd_card(args) -> int:
    """Generate the printable calibration card."""
    from modules import calibration_card as CC
    out = CC.make_card(args.out, page=args.page)
    lines = [
        f"Wrote {out} (and .png alongside).",
        "",
        "PRINT AT 100% / 'actual size' -- NOT 'fit to page'. Then measure the 100 mm",
        "rule on the printout with a ruler. If it is not exactly 100 mm the page was",
        "scaled, and every distance the software reports will be wrong by the same",
        "fraction, silently.",
        "",
        "Stand it BESIDE the cylinders, upright, in the same plane as the front of the",
        "tubes. A card set back behind them reads a smaller scale and makes every",
        "climbing height come out too large.",
    ]
    for line in lines:
        print(line)
    return 0


def cmd_setup_check(args) -> int:
    """Score a TEST recording against the protocol, before the real experiment."""
    from modules import protocol as P

    if args.print_protocol:
        print(P.protocol_text())
        return 0
    if not args.video:
        raise SystemExit("Give a test recording, or use --print-protocol.")
    rep = P.check_setup(args.video, n_vials=args.vials)
    print(rep.report())
    return 1 if rep.failures else 0


def cmd_triage(args) -> int:
    """What questions can this dataset answer? Run this BEFORE analysing."""
    from modules import data_triage as T
    from modules.data_parser import load_data

    if args.dam:
        from modules import dam_reader as DR, dams as DM
        paths = args.csv.split(",")
        cmap = pd.read_csv(args.channel_map) if args.channel_map else None
        # Name the monitors to match the channel map when we can, so a trivial
        # naming mismatch ("Monitor1" vs "M1") does not silently unlabel every
        # tube. Order is preserved: the first file is the first monitor named.
        names = [Path(p).stem for p in paths]
        if cmap is not None:
            map_names = list(pd.unique(cmap["Monitor"].astype(str)))
            if not (set(names) & set(map_names)) and len(map_names) == len(names):
                _eprint(f"Monitor names {names} do not appear in the channel map "
                        f"{map_names}; matching them in file order.")
                names = map_names
        long = DR.read_monitors(paths, monitors=names)
        per_fly = None
        if cmap is not None:
            tidy = DR.to_dams_long(DR.apply_channel_map(
                DR.truncate_at_death(long, 12.0), cmap), drop_dead=True)
            per_fly = DM.sleep_per_fly(tidy)
        rep = T.triage_dam(long, channel_map=cmap, per_fly=per_fly)
    else:
        df, warn, err = load_data(args.csv)
        if err:
            raise SystemExit(f"Could not read {args.csv}: {err}")
        rep = T.triage(df)

    print(rep.report())
    return 1 if rep.blockers else 0


def cmd_check(args) -> int:
    """Design and clustering diagnostics on an already-scored table."""
    from modules.data_parser import load_data
    from modules import design_schema as ds
    from modules import hierarchical as H

    df, warn, err = load_data(args.csv)
    if err:
        raise SystemExit(f"Could not read {args.csv}: {err}")
    print(f"{len(df)} rows loaded from {args.csv}")
    for w in warn:
        print("  warning:", w)

    rep = ds.design_report(df)
    print(f"\ndesign        : {rep['design_summary']}")
    print(f"replication unit: {rep['cluster_unit']}")
    for w in rep["warnings"]:
        print("  WARNING:", w)
    for b in rep["blocking"]:
        print("  BLOCKING:", b)

    cluster = rep["cluster_unit"]
    if not cluster:
        return 0

    # Diagnostics must be run at ONE timepoint. Summing counts over a vial's
    # timepoints counts the same flies repeatedly: with six scoring times and
    # twenty flies the apparent cluster size becomes 120, and the design effect
    # is inflated several-fold. The timepoints are repeated measures OF a vial,
    # not additional flies IN it.
    times = sorted(df["Time"].dropna().unique()) if "Time" in df.columns else []
    at = args.at_time if getattr(args, "at_time", None) is not None else (
        times[-1] if times else None)
    sub = df[df["Time"] == at] if at is not None else df
    if at is not None:
        print(f"\ndiagnostics computed at Time = {at:g} "
              f"(of {len(times)} timepoints; use --at-time to change)")

    if "Treatment" in sub.columns and sub["Treatment"].nunique() == 2:
        pr = H.pseudoreplication_report(sub, "Treatment", cluster)
        print("clustering:")
        print(" ", pr.get("verdict") or pr.get("reason"))
    else:
        icc = H.estimate_icc(sub.groupby(cluster)["Count"].sum(),
                             sub.groupby(cluster)["n"].sum())
        if np.isfinite(icc.get("icc", np.nan)):
            de = H.design_effect(icc["mean_cluster_size"], icc["icc"])
            n_eff = H.effective_sample_size(sub["n"].sum(),
                                            icc["mean_cluster_size"], icc["icc"])
            print(f"intraclass correlation: {icc['icc']:.3f} over {icc['n_clusters']} "
                  f"{cluster}s of ~{icc['mean_cluster_size']:.0f} flies")
            print(f"design effect {de['deff']:.2f}: standard errors are "
                  f"{(de['se_inflation'] - 1) * 100:.0f}% larger than a fly-level analysis "
                  f"would give, and {sub['n'].sum():.0f} flies carry the information of "
                  f"{n_eff:.0f} independent ones.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="drophenix_cli.py",
        description="Score Drosophila climbing assays from video, locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Videos never leave this machine. See docs/VIDEO_SCORING.md.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="write a starter manifest for a folder")
    m.add_argument("--folder", required=True)
    m.add_argument("--out", help="write here instead of stdout")
    m.set_defaults(func=cmd_manifest)

    i = sub.add_parser("inspect", help="examine one recording before trusting it")
    i.add_argument("video")
    i.add_argument("--vials", type=int, default=None,
                   help="number of cylinders (default: detect automatically)")
    i.add_argument("--diameter-mm", type=float, dest="diameter_mm",
                   help="cylinder outside diameter, e.g. 25")
    i.add_argument("--px-per-mm", type=float, dest="px_per_mm")
    i.add_argument("--annotate", help="folder for annotated frames")
    i.add_argument("--times", type=float, nargs="+", default=[10, 30],
                   help="seconds after the tap to annotate")
    i.set_defaults(func=cmd_inspect)

    b = sub.add_parser("batch", help="score a folder into one table")
    b.add_argument("--folder", required=True)
    b.add_argument("--manifest", help="CSV of metadata; see the `manifest` command")
    b.add_argument("--out", default="drophenix_out")
    b.add_argument("--vials", type=int, default=None,
                   help="number of cylinders. STATE IT when every recording is of the "
                        "same rig; automatic detection can miscount an individual video")
    b.add_argument("--times", type=float, nargs="+",
                   default=[5, 10, 15, 20, 30, 60],
                   help="seconds after the tap to score")
    b.add_argument("--diameter-mm", type=float, dest="diameter_mm")
    b.add_argument("--px-per-mm", type=float, dest="px_per_mm")
    b.add_argument("--height-mm", type=float, dest="height_mm",
                   help="score flies above this height in mm (needs a calibration). "
                        "This is the only criterion comparable between laboratories")
    b.add_argument("--fraction", type=float,
                   help="score flies above this fraction of tube height, e.g. 0.5")
    b.add_argument("--min-confidence", type=float, default=0.70,
                   dest="min_confidence")
    b.set_defaults(func=cmd_batch)

    dr = sub.add_parser("dam-run",
                        help="raw DAMSystem monitor files + channel map -> recording quality, "
                             "design-aware stratified tests with FDR, figures, verified Word report")
    dr.add_argument("--monitors", nargs="+", required=True, help="raw MonitorN.txt files, one per monitor")
    dr.add_argument("--map", required=True, help="channel map CSV (Monitor, Channel, Genotype, Sex, Treatment)")
    dr.add_argument("--name", default="run A")
    dr.add_argument("--second", nargs="+", help="monitor files of a second run (timepoint comparison)")
    dr.add_argument("--second-map", dest="second_map", help="channel map for the second run (default: --map)")
    dr.add_argument("--second-name", dest="second_name", default="run B")
    dr.add_argument("--ref-genotype", dest="ref_genotype", help="reference genotype (default: auto)")
    dr.add_argument("--ref-treatment", dest="ref_treatment", help="reference treatment (default: auto)")
    dr.add_argument("--no-stratify", action="store_true", dest="no_stratify", help="do not stratify by Sex")
    dr.add_argument("--window", help="override the proposed window as YYYY-MM-DD:YYYY-MM-DD (first:last day)")
    dr.add_argument("--out", default="dam_run_output")
    dr.set_defaults(func=cmd_dam_run)

    pd_ = sub.add_parser("plan-dam",
                         help="plan an activity-monitor run: allocate channels so no "
                              "factor is confounded with the monitor")
    pd_.add_argument("--genotype", nargs="+")
    pd_.add_argument("--sex", nargs="+")
    pd_.add_argument("--age", nargs="+")
    pd_.add_argument("--treatment", nargs="+")
    pd_.add_argument("--n-per-group", type=int, default=8, dest="n_per_group")
    pd_.add_argument("--monitors", type=int, default=2)
    pd_.add_argument("--channels", type=int, default=32)
    pd_.add_argument("--goal", default="sleep", choices=["sleep", "period"])
    pd_.add_argument("--seed", type=int, default=0)
    pd_.add_argument("--out", default="dam_channel_map.csv")
    pd_.add_argument("--check", help="verify an EXISTING channel map instead of planning")
    pd_.add_argument("--print-protocol", action="store_true", dest="print_protocol")
    pd_.set_defaults(func=cmd_plan_dam)

    cd = sub.add_parser("card", help="generate the printable calibration card")
    cd.add_argument("--out", default="DroPhenix_calibration_card")
    cd.add_argument("--page", default="A4", choices=["A4", "Letter"])
    cd.set_defaults(func=cmd_card)

    sc = sub.add_parser("setup-check",
                        help="check a TEST recording against the protocol, before "
                             "running the real experiment")
    sc.add_argument("video", nargs="?", help="30 s of test footage")
    sc.add_argument("--vials", type=int, default=None)
    sc.add_argument("--print-protocol", action="store_true", dest="print_protocol",
                    help="print the written protocol and exit")
    sc.set_defaults(func=cmd_setup_check)

    tr = sub.add_parser("triage",
                        help="what questions can this data answer? run this FIRST")
    tr.add_argument("csv", help="a scored CSV, or comma-separated monitor files with --dam")
    tr.add_argument("--dam", action="store_true",
                    help="inputs are raw DAMSystem monitor files")
    tr.add_argument("--channel-map", dest="channel_map",
                    help="channel map CSV (with --dam)")
    tr.set_defaults(func=cmd_triage)

    c = sub.add_parser("check", help="design diagnostics on a scored CSV")
    c.add_argument("csv")
    c.add_argument("--at-time", type=float, dest="at_time",
                   help="timepoint at which to compute clustering diagnostics "
                        "(default: the last one). Repeated timepoints are repeated "
                        "measures of a vial, not extra flies in it")
    c.set_defaults(func=cmd_check)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
