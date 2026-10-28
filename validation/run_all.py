"""
run_all.py — DroPhenix master validation runner.

Executes every per-module validation suite (each benchmarked against published
raw data / hand-checked analytics) and prints a consolidated pass/fail report.
This is the real, runnable validation suite for DroPhenix.

Suites:
  validate_stats     — stats_engine vs PlantGrowth ANOVA/Kruskal (Dobson 1983)
  validate_survival  — survival vs Freireich 1963 6-MP leukemia data
  validate_ymaze     — Y-maze vs de Bivort 2015 / Kim 2017 anchors
  validate_dams      — DAMS vs Konopka & Benzer 1971 periods + 5-min sleep rule
  validate_synthetic — unified synthetic cohort sanity (disease-selective shape)
  validate_dose_response — dose_response vs the drc `ryegrass` ferulic acid data
                           (Inderjit, Streibig & Olofsdotter 2002)
  validate_hierarchical  — clustered/binomial climbing analysis: ICC against the
                           exact one-way random-effects identity, beta-binomial
                           against statsmodels GLM, and the type I error of the
                           naive fly-as-independent test under clustering
  validate_survivor_bias — survivor bias in ageing trajectories, against a
                           generative model with a known true slope
  validate_trajectory_prediction — hierarchical forecasting: population prior,
                           forward-in-time validation, conformal calibration
  validate_control_prior — robust MAP historical-control priors: closed forms
                           against numerical integration, ESS against its exact
                           analytic limit, and simulated operating characteristics
  validate_circadian_climbing — time-of-day cosinor analysis of climbing
  validate_data_triage — upfront go/no-go: which questions the data can answer
                           (confounding, replication, ceilings, clustering,
                           plausibility, coverage), tested as matched pairs so a
                           checker cannot pass by firing on everything
  validate_dam_reader — reading raw DAMSystem monitor files: exact parsing,
                           monitor-fault minutes treated as missing rather than
                           as sleep, restart and death detection, and refusal of
                           circadian period under LD or too-short recordings
  validate_climbing_band — the band-and-zone scorer: the Performance Index
                           arithmetic, the observation/estimate/result split,
                           and the reason codes that gate whether a reading may
                           be reported as a biological measurement at all. Does
                           NOT establish that the detector counts correctly --
                           that is measured against hand counts and lives in
                           climbing_band.VALIDATION_EVIDENCE
  validate_climbing_video — scoring a climbing assay from video: exact counts on
                           synthetic footage with known ground truth, and the
                           frame-registration ablation on real laboratory
                           recordings (real-video tests SKIP, not fail, when the
                           recordings are absent)
  validate_climbing_video_analyzer — the ROI-based analyser built from
                           CLAUDE_ClimbingVideoAnalyzer.md: exact counts on
                           synthetic footage checked in BOTH directions, the
                           drawn-scoring-line regression, and a measured read of
                           what adaptive thresholding does on a graduated cylinder
  validate_fly_blob — size-matched fly detection (Crocker-Grier band-pass,
                      local maxima, mass and shape filters). Pins the properties
                      an intensity threshold lacks, the resolution and size limits
                      it does NOT have, and re-derives its held-out accuracy
                      against flies actually loaded rather than asserting it
  validate_climbing_detectors — offering every detection method and choosing one
                           on evidence: the probe counts permanent marks and they
                           predict the threshold detector's false positives, motion
                           is measured because background subtraction needs it, and
                           a scene neither method can handle is REFUSED
  validate_climbing_layout — hand-drawn vial layouts: canvas-to-native coordinate
                           scaling (where a drawing interface fails invisibly),
                           left-to-right labelling, the invariants that made the
                           page necessary (line outside the box, overlaps, missing
                           fly count), and carrying a layout to another recording
                           with a refusal when the camera moved too far
  validate_confirmable_line — the scoring line the operator confirms: a
                             confirmed row survives the shared-height-plus-offset
                             round trip with zero drift, the confidence gate
                             refuses the margin-1.04 case that put the line 400 px
                             wrong on 12 of 12 paraquat vials, and confirmation
                             does not license a line outside its own tube
  validate_dam_planner — allocating flies to monitor channels BEFORE the run, so
                           that no factor is confounded with the instrument;
                           includes the regression test that the laboratory's
                           own first layout (all old in Monitor 1, all young in
                           Monitor 2) is rejected on Age while Genotype and Sex
                           still pass
  validate_freeclimber_benchmark — DroPhenix vs an independent published tool on
                           REAL data: an independent reimplementation of
                           FreeClimber's local-linear-regression velocity
                           read-out reproduces its published slope, frame
                           window, r and standard error for every vial of
                           Spierer et al. (2021)'s own example recording
  validate_real_aging_benchmark — the decline model on REAL published
                           longitudinal data (Shahrestani et al. 2016,
                           Evolution 70:2550; 10 populations, geotaxis and
                           mortality across age). Recovers the published
                           direction: rapid-development populations decline
                           2.1x faster and live 27.7 vs 50.1 days, and the
                           decline-lifespan coupling is rho = -0.82/-0.84
  validate_real_lifespan_benchmark — survival.py on a REAL Drosophila lifespan
                           experiment (Am Nat 2013, doi:10.1086/673296): 39,581
                           individual flies, 50 hemiclonal lines, 200 groups.
                           Checks that BOTH input layouts of the same experiment
                           give the same curves, that KM medians match lifelines
                           on 195/200 groups with every disagreement an exact
                           S(t)=0.5 tie, and that an underflowing log-rank p is
                           no longer printed as 0.0
  validate_larval_powerlaw — larval.py against 123 REAL larval trajectories with
                           a QUANTITATIVE published target (Zago, Lacquaniti &
                           Gomez-Marin 2016, Biol Lett 12:20160597): the
                           speed-curvature power-law exponent must reproduce
                           their per-condition medians AND interquartile ranges,
                           the exponent must be scale-invariant, ordinary least
                           squares must be visibly attenuated relative to the
                           orthogonal regression they used, and every real larva
                           must fall inside the crawling speed range larval.py's
                           own docstring claims
  validate_grooming_nf1 — grooming.py on REAL scored ethograms (King et al. 2020,
                           PLOS Genet 16:e1008920). Exercises BOTH input layouts:
                           the per-fly summary reproduces the published direction
                           and statistic (Nf1 groom more, Wilcoxon p < 0.01), and
                           1,096 real bouts show that the 2026-09-06 interval-
                           merging fix fires on genuine data — 3 of 91
                           fly-segments contain overlapping scored bouts. Also
                           pins a genotype-coding contradiction in the source
                           file rather than silently resolving it
  validate_dam_25thday — the Shaw/Hendricks >=5-min sleep rule on the LAB'S OWN
                           raw DAM beam counts (10.2 days, 60 tubes): the rule
                           matches an independent reimplementation to 1.1
                           min/day, faulty minutes stay NaN instead of
                           fabricating sleep, plausibility PASSES on healthy
                           data (the 2026-08 recording failed it, so the pair
                           shows the checker discriminates), and the design
                           checker flags this layout's Genotype x Monitor
                           confound while correctly clearing Sex
  validate_dam_15v25 — the 15th-day vs 25th-day comparison, and the limits on it:
                           the 15th-day file is 1-HOUR binned so sleep cannot be
                           scored from it at all; the two ran on DIFFERENT
                           photoperiods (15L:9D vs 12L:12D) so age is completely
                           confounded with light schedule; and the tubes are NOT
                           the same flies (raw rho +0.35 collapses to -0.02
                           within group), so the timepoints are unpaired
  validate_dam_parameters — the published DAM parameter set (Pfeiffenberger 2010
                           Counting Macro; Chiu 2010; ShinyR-DAM, Cichewicz &
                           Hirsh 2018) against closed forms, plus the two things
                           that have to be measured rather than assumed: the
                           photoperiod, and WHICH monitor's light channel to
                           believe when they disagree (on the lab's 11-15 Aug
                           run one sensor was stuck on and the module used to
                           take whichever fly sorted first). Also validates van
                           Elteren's stratified Wilcoxon: exact agreement with
                           Mann-Whitney on one stratum, refusal to read a pure
                           between-stratum difference as a treatment effect,
                           and nominal type I error over 3,000 null experiments
  verify_dam_docx — the DELIVERABLE itself: reads docs/DAM_RESULTS_REPORT.docx
                           back and checks every number printed in it against the
                           CSV the analysis wrote, at the precision the document
                           displays. A report is where numbers get retyped, and
                           this caught six wrong values on 2026-09-09. Also
                           checks that the Markdown and shareable HTML versions
                           print only effect sizes the analysis actually produced
  validate_dam_pipeline — the AUTONOMOUS run analysis (app page "DAM Run
                           Analysis", CLI `dam-run`): the window it proposes on
                           synthetic light traces excludes failed, flickering,
                           partial and gapped days for the stated reason; the
                           design is read from the channel map without knowing
                           any genotype name; and on the laboratory's two raw
                           recordings it REPRODUCES docs/DAM_RESULTS_REPORT.docx
                           -- same windows, same 55/58 flies, same 4/6 outliers,
                           g/p/q identical to the report's CSVs, unpaired
                           timepoints -- with no hand-holding. Refuses a 60-min
                           export and a recording with no usable day
  validate_multi_assay — the Multi-Assay Composite Motor Score: median survival
                           is the Kaplan-Meier median and is NaN when fewer than
                           half the cohort dies (cross-checked against lifelines),
                           weights renormalise over the assays each group
                           actually has, normalisation is on absolute scales so a
                           tie is not scored as the worst case, and a missing
                           assay is never mean-imputed. Regression tests for the
                           three S1 defects in docs/MODULE_AUDIT_2026-09.md

A NOTE ON WHAT THESE SUITES DO AND DO NOT ESTABLISH
---------------------------------------------------
Suites divide into three kinds, and they are not equivalent evidence:

  ANALYTIC       — the estimator is checked against a closed form or an exact
                   identity (ICC, CRPS, cosinor, prior ESS, curve recovery).
                   These establish that the implementation is correct.
  PUBLISHED DATA — real raw data with published reference statistics
                   (Freireich 1963, PlantGrowth, drc ryegrass). These establish
                   agreement with the literature.
  SIMULATION     — data generated from a known truth, then recovered. These
                   establish that the estimator works UNDER ITS OWN
                   ASSUMPTIONS, which is necessary but not sufficient: a
                   simulation cannot detect a model that is mis-specified for
                   real flies.

No amount of simulation substitutes for real laboratory data. See
docs/INVERSION_ANALYSIS.md, entries E1 and F3.

Run:  python validation/run_all.py        (exit 0 = every suite passed)
"""

import importlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")   # keep the consolidated report clean
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUITES = ["validate_stats", "validate_survival", "validate_ymaze",
          "validate_dams", "validate_synthetic", "validate_drug_ranking",
          "validate_video", "validate_larval", "validate_grooming",
          "validate_metrics", "validate_trajectory", "validate_dose_response",
          "validate_hierarchical", "validate_survivor_bias",
          "validate_trajectory_prediction", "validate_control_prior",
          "validate_circadian_climbing", "validate_climbing_video",
          "validate_dam_reader", "validate_data_triage", "validate_calibration_card",
          "validate_dam_planner", "validate_climbing_video_analyzer",
          "validate_fly_blob", "validate_climbing_detectors", "validate_climbing_layout",
          "validate_climbing_band",
          "validate_confirmable_line", "validate_multi_assay",
          "validate_motor_decline", "validate_translational_score",
          "validate_freeclimber_benchmark", "validate_real_aging_benchmark",
          "validate_real_lifespan_benchmark", "validate_larval_powerlaw",
          "validate_grooming_nf1", "validate_dam_25thday", "validate_dam_15v25",
          "validate_dam_parameters", "verify_dam_docx", "validate_dam_pipeline",
          "demo_unequal_n"]


def main() -> int:
    results = {}
    skipped = {}
    for name in SUITES:
        # A missing OPTIONAL dependency must not abort the whole run. Before
        # this guard, importing a suite under an interpreter without cv2 raised
        # ModuleNotFoundError straight out of the loop and stopped every
        # remaining suite mid-run, with no summary printed at all. That is only
        # ever an environment problem, so it is reported as SKIPPED and the run
        # continues; a suite that imports and then fails is still a FAILURE.
        try:
            mod = importlib.import_module(f"validation.{name}")
        except ImportError as e:
            missing = getattr(e, "name", None) or str(e)
            if missing == f"validation.{name}":
                # The suite itself is not in this checkout. Suites written
                # around unpublished laboratory recordings are kept out of the
                # public repository; their absence is not an error.
                print(f"[{name}] SKIPPED: this suite is not present in this checkout.")
                skipped[name] = "suite not in this checkout"
            else:
                print(f"[{name}] SKIPPED: missing dependency '{missing}'. "
                      f"Install it (see requirements) to run this suite.")
                skipped[name] = str(missing)
            print()
            continue
        try:
            rc = mod.main()
        except Exception as e:  # a suite that crashes is a failure
            print(f"[{name}] CRASHED: {e!r}")
            rc = 1
        results[name] = (rc == 0)
        print()

    print("#" * 70)
    print("# DroPhenix MASTER VALIDATION SUMMARY")
    print("#" * 70)
    for name in SUITES:
        if name in skipped:
            print(f"  {name:<32} SKIP  (missing {skipped[name]})")
        elif name in results:
            print(f"  {name:<32} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values()) if results else False
    print("#" * 70)
    if skipped:
        print(f"NOTE: {len(skipped)} suite(s) were SKIPPED for missing dependencies "
              f"and therefore establish NOTHING. Run in the project venv "
              f"(.venv/Scripts/python.exe) for the complete suite.")
    print("OVERALL:", "ALL SUITES PASSED" if all_ok else "FAILURES PRESENT")
    # Skipped suites are not failures, but they are not passes either; the exit
    # code reflects only the suites that actually ran.
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
