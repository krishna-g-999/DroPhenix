"""
DroPhenix Analytics v2.0
Drosophila Climbing and Motor Assay Analysis Platform
SSSIHL Department of Bioscience 
"""

# -- Version --

from __future__ import annotations


# -- Version --
_VERSION = "2.0"

import io
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DroPhenix Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
try:
    from modules.styles import inject_css, render_header, chip, status_badge
    _STYLES_OK = True
except ImportError:
    _STYLES_OK = False
    def inject_css(): pass
    def render_header(n, s="", sl="", sv=""): st.markdown(f"## {n}")
    def chip(t, s="pathway"): return t
    def status_badge(t, s="idle"): return t


# Protocol guide helpers
try:
    from modules.protocol_guide import show_data_template, show_vial_protocol
except ImportError:
    def show_data_template(): pass
    def show_vial_protocol(): pass

inject_css()

# ── Write config.toml once ────────────────────────────────────────────────────
def _write_config():
    d = Path(".streamlit")
    d.mkdir(exist_ok=True)
    p = d / "config.toml"
    if not p.exists():
        p.write_text(
            '[theme]\n'
            'primaryColor           = "#F0A500"\n'
            'backgroundColor        = "#EEF2F9"\n'
            'secondaryBackgroundColor = "#FFFFFF"\n'
            'textColor              = "#0D2137"\n'
            'font                   = "sans serif"\n'
            '[server]\n'
            'headless = true\n'
            'enableCORS = false\n'
            'maxUploadSize = 500\n'
        )
_write_config()

# ── Module imports ────────────────────────────────────────────────────────────
try:
    from modules.data_parser import (
        load_data, get_groups, get_group_summary, validate_dataframe,
    )
    from modules.normalizer import compute_all_metrics
    from modules.stats_engine import (
        auto_test, bootstrap_ci_table, sdq_permutation_test,
    )
    from modules.motor_decline import (
        compute_mdr_table, mdr_population_summary, compute_msci,
    )
    from modules.trajectory_cluster import (
        cluster_trajectories, cluster_summary_table, interpret_clusters,
    )
    from modules.metrics import compute_novel_metrics
    from modules.translational_score import run_tcs_pipeline
    from modules.visualizer import (
        climbing_curves, pi_barplot, auc_heatmap, t50_plot,
        cri_barplot, sdq_plot, tsw_plot,
        tcs_radar, tcs_ranked_bar,
        mdr_decay_curves, trajectory_dendrogram, trajectory_pca,
        sex_comparison_plot, export_all_figures,
    )
    from modules.recording import (
        list_cameras, capture_test_frame, capture_background_frame,
        RecordingSettings, RecordingSession,
        run_pre_assay_qc, analyze_recorded_video_qc,
        CV2_AVAILABLE,
    )
    from modules.video_analyzer import (
        VideoAnalyzer, video_counts_to_drophenix, get_video_info,
    )
    from modules.fly_detector import DetectorConfig, VIAL_PRESETS
    _IMPORTS_OK = True
    _IMPORT_ERROR = ""
except Exception as _ie:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(_ie)

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df":                  None,
        "data_source":         "none",
        "data_warnings":       [],
        "data_errors":         [],
        "base_metrics":        {},
        "novel_metrics":       {},
        "mdr_df":              None,
        "cluster_result":      None,
        "tcs_result":          None,
        "stats_result":        None,
        "recording_session":   None,
        "last_recording_path": None,
        "vial_configs":        [],
        "wt_genotype":         None,
        "vehicle_treatment":   None,
        "run_complete":        False,
        "_rec_settings":       None,
        "_detector_cfg":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Import guard ──────────────────────────────────────────────────────────────
if not _IMPORTS_OK:
    render_header("DroPhenix Analytics v2.0",
                  "Drosophila Climbing and Motor Assay Analysis Platform")
    st.error(f"Module import failed: {_IMPORT_ERROR}")
    st.info(
        "Install required packages:\n```\n"
        "pip install streamlit pandas numpy scipy matplotlib "
        "scikit-learn openpyxl opencv-python\n```"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
PAGES = [
    "Home",
    "Data Input",
    "Climbing Curves",
    "Core Metrics",
    "Novel Metrics",
    "Motor Decline",
    "Trajectory Clustering",
    "Translational Score",
    "Statistical Analysis",
    "Video Analysis",
    "CSV Upload",
    "Export",
]

with st.sidebar:
    try:
        from modules.styles import render_sidebar_brand
        render_sidebar_brand()
    except ImportError:
        st.markdown(
            '<div style="text-align:center;padding:1.2rem 0 .4rem;'
            'font-size:1.3rem;font-weight:800;color:#F0A500">DroPhenix</div>',
            unsafe_allow_html=True,
        )
    st.divider()
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    _df_sb = st.session_state.get("df")
    if _df_sb is not None and not _df_sb.empty:
        try:
            _grps_sb = get_groups(_df_sb)
            st.markdown(status_badge("Data loaded", "complete"), unsafe_allow_html=True)
            st.caption(
                f"{len(_df_sb)} rows · "
                f"{len(_grps_sb['genotypes'])} genotypes · "
                f"{len(_grps_sb['treatments'])} treatments"
            )
        except Exception:
            st.markdown(status_badge("Data loaded", "complete"), unsafe_allow_html=True)
    else:
        st.markdown(status_badge("No data", "idle"), unsafe_allow_html=True)
    if st.session_state.get("run_complete"):
        st.markdown(status_badge("Analysis complete", "complete"), unsafe_allow_html=True)
    st.divider()
    st.caption("SSSIHL Computational Biology Lab · GPL-3.0")

# ── Guards ────────────────────────────────────────────────────────────────────
def _require_data() -> bool:
    if st.session_state.get("df") is None:
        st.warning("No data loaded. Go to Data Input first.")
        return False
    return True

def _require_metrics() -> bool:
    if not st.session_state.get("base_metrics"):
        st.warning("No analysis results. Go to Data Input and run the pipeline.")
        return False
    return True


# ── Pipeline ──────────────────────────────────────────────────────────────────
def _run_pipeline(df: pd.DataFrame, wt: Optional[str], veh: Optional[str]):
    prog = st.progress(0, text="Computing base metrics...")
    base = compute_all_metrics(df)
    st.session_state["base_metrics"] = base
    prog.progress(20, text="Computing novel metrics...")

    novel = compute_novel_metrics(df, base, wt_genotype=wt, vehicle_treatment=veh)
    st.session_state["novel_metrics"] = novel
    prog.progress(40, text="Computing motor decline rates...")

    agg = base.get("aggregated", pd.DataFrame())
    if not agg.empty and "Pct_mean" in agg.columns:
        try:
            mdr = compute_mdr_table(agg.rename(columns={"Pct_mean": "PI_mean"}))
            st.session_state["mdr_df"] = mdr
        except Exception as e:
            warnings.warn(f"MDR: {e}")
            st.session_state["mdr_df"] = pd.DataFrame()

    prog.progress(60, text="Clustering trajectories...")
    if not agg.empty:
        try:
            cr = cluster_trajectories(agg if "PI_mean" in agg.columns else agg.rename(columns={"Pct_mean": "PI_mean"}))
            st.session_state["cluster_result"] = cr
        except Exception as e:
            warnings.warn(f"Clustering: {e}")
            st.session_state["cluster_result"] = None

    prog.progress(80, text="Computing translational scores...")
    try:
        tcs_r = run_tcs_pipeline(
            base_metrics=base,
            novel_metrics=novel,
            mdr_df=st.session_state.get("mdr_df"),
            cluster_result=st.session_state.get("cluster_result"),
        )
        st.session_state["tcs_result"] = tcs_r
    except Exception as e:
        warnings.warn(f"TCS: {e}")
        st.session_state["tcs_result"] = None

    st.session_state["wt_genotype"]       = wt
    st.session_state["vehicle_treatment"] = veh
    st.session_state["run_complete"]      = True
    prog.progress(100, text="Complete.")
    st.success("Analysis pipeline complete. Use the sidebar to explore results.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE BODY (replace the `if current == 'Home':` block) ────────

HOME_PAGE_CODE = '''
if current == 'Home':
    col_text, col_img = st.columns([3, 1])
    with col_text:
        st.markdown("""
        <div style="padding:.8rem 0 .6rem 0">
          <div style="font-size:1.8rem;font-weight:800;color:#0d1b2e;
                      line-height:1.15;margin-bottom:.6rem">
            Drosophila Motor Phenotyping<br>
            <span style="color:#f0a500">Analytics Platform</span>
          </div>
          <div style="font-size:.92rem;color:#1a2332;line-height:1.75;max-width:640px">
            <strong>DroPhenix</strong> is a computational phenotyping platform for
            quantifying motor behaviour in <em>Drosophila melanogaster</em> climbing
            assays. It integrates multi-trial normalisation, novel motor rescue
            indices, and sex-stratified analysis to support translational neuroscience
            research in <strong>motor neuron disorders</strong> .
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_img:
        if SAINET_URI:
            st.markdown(f"""
            <div style="display:flex;justify-content:center;padding:1rem 0">
              <img src="{SAINET_URI}"
                   style="width:150px;height:150px;border-radius:50%;object-fit:contain;
                          border:3px solid #f0a500;
                          box-shadow:0 4px 20px rgba(240,165,0,.22)" alt="SAI-Net">
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Quick Start ──────────────────────────────────────────────────────
    sec("Quick Start — Load Your Data")
    info("Upload your climbing assay CSV / Excel, or click <b>Demo Dataset</b> "
         "to explore instantly. All metrics and plots populate automatically.")

    load_c1, load_c2 = st.columns(2)
    with load_c1:
        st.markdown("""<div class="data-load-card">
          <h4>Upload Your Dataset</h4>
          <p>Accepts CSV or Excel.<br>
          Required columns: <code>Genotype</code>, <code>Treatment</code>,
          <code>Sex</code>, <code>Time</code>, <code>Count climbed</code>,
          <code>n total flies</code>.</p></div>""", unsafe_allow_html=True)
        home_upload = st.file_uploader(
            "Choose file", type=["csv","xlsx","xls"],
            label_visibility="visible", key="home_upload")
        if home_upload and st.button("Load Uploaded File", key="home_load_btn"):
            with st.spinner("Parsing…"):
                try:
                    df, _, _ = load_data(home_upload)
                    st.session_state.df    = df
                    st.session_state.base  = compute_all_metrics(df)
                    st.session_state.novel = run_novel(df, st.session_state.base)
                    st.session_state.loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    with load_c2:
        st.markdown("""<div class="data-load-card">
          <h4>Try the Demo Dataset</h4>
          <p>Built-in SSSIHL ALS dataset — Oregon-WT, TDP-43, FUS.
          Untreated + 3 treatments across both sexes. No file needed.</p>
          </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶  Load Demo Dataset", key="home_demo_btn"):
            with st.spinner("Generating demo…"):
                try:
                    df, _, _ = load_data(None)
                    st.session_state.df    = df
                    st.session_state.base  = compute_all_metrics(df)
                    st.session_state.novel = run_novel(df, st.session_state.base)
                    st.session_state.loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.loaded:
        agg = st.session_state.base["aggregated"]
        success(f"Data loaded — <b>{agg['Genotype'].nunique()} genotypes</b>, "
                f"<b>{agg['Treatment'].nunique()} treatments</b>, "
                f"<b>{agg['Sex'].nunique()} sexes</b>, "
                f"<b>{agg['Time'].nunique()} timepoints</b>. "
                "Use the navigation bar above to explore.")
        nav_btns = st.columns(4)
        for col, pg in zip(nav_btns, ["Data Overview","Climbing Kinetics",
                                       "Performance Index","Novel Metrics"]):
            with col:
                if st.button(f"→ {pg}", key=f"home_goto_{pg}"):
                    st.session_state.page = pg
                    st.rerun()

    st.divider()

    # ── Feature Cards ────────────────────────────────────────────────────
    sec("Platform Modules — What Each Tab Does")

    # Row 1
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    row1 = [
        (r1c1, "Data Overview", "Data Overview",
         "Data QC & Statistics",
         "Aggregates raw vial counts per group, computes SEM, validates "
         "completeness, and flags outliers. Provides a full statistical "
         "summary table before any downstream analysis.",
         "1 Load data → 2 Data Overview tab → 3 Inspect the aggregated table → "
         "4 Download <code>aggregated_data.csv</code>"),

        (r1c2, "Climbing Kinetics", "Climbing Kinetics",
         "Kinetics, AUC, t₅₀",
         "Plots time-resolved climbing performance curves for each "
         "Genotype × Treatment × Sex group. Computes Area Under the Curve "
         "(AUC, normalised) and time to 50 % climb (t₅₀) for each group.",
         "1 Load data → 2 Climbing Kinetics → 3 Filter by genotype/sex → "
         "4 Toggle t₅₀ line → 5 View AUC radar"),

        (r1c3, "Performance Index", "Performance Index",
         "PI = (climbed − not climbed) / total × 100",
         "Standardised motor capacity index ranging −100 to +100. "
         "Positive PI = more flies climbed than stayed. "
         "PI is the primary outcome for treatment-vs-control comparisons.",
         "1 Load data → 2 Performance Index → 3 View bar chart with SEM → "
         "4 Download <code>performance_index.csv</code>"),

        (r1c4, "Novel Metrics", "Novel Metrics",
         "CRI · SDQ · TSW",
         "<b>CRI</b> (Compound Rescue Index) — composite rescue score "
         "normalised to untreated control. ≥ 0.6 = strong rescue.<br>"
         "<b>SDQ</b> (Sex Dimorphism Quotient) — sex-differential response. "
         "≥ 0.5 → sex-stratify follow-ups.<br>"
         "<b>TSW</b> (Therapeutic Safety Window) — disease rescue vs WT toxicity.",
         "1 Load data → 2 Novel Metrics → 3 CRI tab (rescue) → "
         "4 SDQ tab (sex dimorphism) → 5 TSW tab (safety)"),
    ]
    for col, pg, nav_pg, badge, desc, howto in row1:
        with col:
            st.markdown(f"""
            <div class="home-card">
              <span class="feat-badge">{badge}</span>
              <h4>{pg}</h4>
              <p>{desc}</p>
              <div class="feat-how">
                <b>How to use:</b><br>{howto}
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button("Open", key=f"feat_{nav_pg}"):
                st.session_state.page = nav_pg
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    row2 = [
        (r2c1, "Sex Analysis", "Sex Analysis",
         "Sex-Stratified Analysis",
         "Side-by-side male vs female climbing kinetics and a full SDQ "
         "heatmap per genotype. Identifies sex-dimorphic drug responses "
         "that are masked in sex-averaged data.",
         "1 Load data → 2 Sex Analysis → 3 Select genotype → "
         "4 View curves + SDQ heatmap → 5 Download sex PI CSV"),

        (r2c2, "Auto Recording", "Auto Recording",
         "Live Camera Recording",
         "Detects webcam automatically (MSMF backend on Windows). "
         "Configurable resolution, FPS, duration, and settle time. "
         "Records MP4 with timestamp overlay; supports offline QC analysis "
         "on any uploaded video.",
         "1 Connect tab → Detect Cameras → 2 Capture test frame → "
         "3 Record tab → Start Recording → 4 Download video + metadata → "
         "5 Offline Analysis tab → Analyse for fly counts"),

        (r2c3, "Export Report", "Export Report",
         "Data Export",
         "Downloads all seven computed metric tables individually or as a "
         "single ZIP bundle with an auto-generated README for journal "
         "submission reproducibility. Includes platform version, date, "
         "and investigator details.",
         "1 Load data → 2 Export Report → 3 Download individual CSVs OR → "
         "4 Build ZIP Archive → 5 Cite as Gunanathan et al. 2026"),

        (r2c4, "Trajectory Translational", "Trajectory Translational",
         "MDR · Clustering · TCS",
         "<b>MDR</b> (Motor Decline Rate) — exponential decay fitting on "
         "PI over time; k = decline constant.<br>"
         "<b>Trajectory Clustering</b> — DTW-based agglomerative grouping "
         "of PI shapes (sustained rescue, transient, decline, plateau).<br>"
         "<b>TCS</b> (Translational Compound Score) — integrates CRI, BBB, "
         "GSH, and Tox into a Tier I–III CNS drug priority score.",
         "1 Load data → 2 Trajectory Translational → 3 MDR table → "
         "4 Adjust clusters slider → 5 Upload compound annotation for TCS"),
    ]
    for col, pg, nav_pg, badge, desc, howto in row2:
        with col:
            st.markdown(f"""
            <div class="home-card">
              <span class="feat-badge">{badge}</span>
              <h4>{pg}</h4>
              <p>{desc}</p>
              <div class="feat-how">
                <b>How to use:</b><br>{howto}
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button("Open", key=f"feat2_{nav_pg}"):
                st.session_state.page = nav_pg
                st.rerun()

    st.divider()

    # ── SAI-Net Connection ────────────────────────────────────────────────
    sec("SAI-Net Translational Connection")
    sql, sqr = st.columns([3, 2])
    with sql:
        st.markdown("""
        <p style="font-size:.88rem;color:#1a2332;line-height:1.78">
        DroPhenix is a translational module of <strong>SAI-Net</strong> —
        a computational neuropharmacology framework integrating multi-omics
        data with graph neural networks for drug discovery in
        neurodegenerative diseases.</p>
        """, unsafe_allow_html=True)
    with sqr:
        st.markdown("""
        <div class="dpx-quote">
          Let the world achieve the glory of becoming a family through Love.
          <div class="dpx-quote-attr">Bhagawan Sri Sathya Sai Baba</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Research Team ─────────────────────────────────────────────────────
    sec("Research Team")
    st.markdown("""
    <div class="team-card">
      <div>
        <span class="team-role role-pi">Principal Investigator</span>
        <div style="font-size:.9rem;font-weight:700;color:#0d1b2e;margin-top:.3rem">
          Prof. Venketesh Sivaramakrishnan</div>
        <div style="font-size:.76rem;color:#1a2332;margin-top:.15rem">
          Department of Biosciences · Sri Sathya Sai Institute of Higher Learning</div>
      </div>
    </div>
    <div class="team-card">
      <div>
        <span class="team-role role-dev">Developer</span>
        <div style="font-size:.9rem;font-weight:700;color:#0d1b2e;margin-top:.3rem">
          Krishnasalini Gunanathan</div>
        <div style="font-size:.76rem;color:#1a2332;margin-top:.15rem">
          Doctoral Research Scholar · Biosciences · SSSIHL<br>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;font-size:.7rem;color:#8ca0b8;'
        f'margin-top:2rem;padding-top:1rem;border-top:1px solid #dde3ec">'
        f'DroPhenix Analytics v2.0 &nbsp;·&nbsp; SAI-Net · SSSIHL &nbsp;·&nbsp; '
        f'{datetime.now().strftime("%B %Y")}</div>',
        unsafe_allow_html=True)
'''
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Input
# ══════════════════════════════════════════════════════════════════════════════
def page_data_input():
    render_header("Data Input",
                  "Load data and configure analysis parameters")

    show_data_template()
    src_tab, meta_tab = st.tabs(["Data Source", "Group Metadata and Analysis"])

    with src_tab:
        src = st.radio(
            "Select data source",
            ["Demo dataset", "Upload CSV or Excel", "Use recorded video data"],
            horizontal=True,
        )
        if src == "Demo dataset":
            st.caption(
                "Built-in demo: Oregon-R (WT), TDP-43, FUS genotypes | "
                "Vehicle, Spermine, Pantothenate | both sexes | "
                "3 replicates | 8 timepoints."
            )
            if st.button("Load Demo Dataset", type="primary"):
                df_raw, warns, errs = load_data(None)
                st.session_state.update({
                    "df": df_raw, "data_source": "demo",
                    "data_warnings": warns, "data_errors": errs,
                    "run_complete": False, "base_metrics": {},
                })
                st.success(f"Demo dataset loaded: {len(df_raw)} rows.")

        elif src == "Upload CSV or Excel":
            st.caption(
                "Long format: Genotype, Sex, Treatment, Replicate, Time, Count, n  |  "
                "Wide format: Genotype, Sex, Treatment, Replicate, t10, t20, t30, ..."
            )
            uploaded = st.file_uploader("Upload file", type=["csv","xlsx","xls"],
                                        label_visibility="collapsed")
            if uploaded:
                df_raw, warns, errs = load_data(uploaded)
                st.session_state.update({
                    "df": df_raw, "data_source": "upload",
                    "data_warnings": warns, "data_errors": errs,
                    "run_complete": False, "base_metrics": {},
                })
                if errs:
                    for e in errs: st.error(e)
                else:
                    st.success(f"File loaded: {len(df_raw)} rows.")
                for w in warns: st.warning(w)

        else:
            if st.session_state.get("last_recording_path"):
                st.info("A recent recording is available. Use CSV Upload or Video Analysis.")
            else:
                st.info("No recording found. Use Live Recording to capture a session.")

    with meta_tab:
        df = st.session_state.get("df")
        if df is None:
            st.info("Load data first.")
            return
        groups = get_groups(df)
        c1, c2 = st.columns(2)
        with c1:
            wt = st.selectbox("Wild-type genotype",
                ["(auto-detect)"] + sorted(groups["genotypes"]))
        with c2:
            veh = st.selectbox("Vehicle / control treatment",
                ["(auto-detect)"] + sorted(groups["treatments"]))
        wt_sel  = None if wt  == "(auto-detect)" else wt
        veh_sel = None if veh == "(auto-detect)" else veh

        st.markdown('<p class="sec-hdr">Group Summary</p>', unsafe_allow_html=True)
        st.dataframe(get_group_summary(df), width='stretch')
        st.divider()
        if st.button("Run Full Analysis Pipeline", type="primary", use_container_width=True):
            _run_pipeline(df, wt_sel, veh_sel)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Climbing Curves
# ══════════════════════════════════════════════════════════════════════════════
def page_climbing_curves():
    render_header("Climbing Curves", "Time-series climbing performance by group")
    if not _require_metrics(): return

    base = st.session_state["base_metrics"]
    agg  = base.get("aggregated", pd.DataFrame())
    norm = base.get("normalised",  pd.DataFrame())
    if agg.empty:
        st.warning("No aggregated data available.")
        return

    groups = get_groups(st.session_state["df"])
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_geno = st.multiselect("Genotypes",  groups["genotypes"], default=groups["genotypes"])
    with c2:
        sel_trt  = st.multiselect("Treatments", groups["treatments"], default=groups["treatments"])
    with c3:
        sel_sex  = st.multiselect("Sex",         groups["sexes"],      default=groups["sexes"])

    show_raw = st.checkbox("Overlay individual replicate traces", value=False)
    fig, img = climbing_curves(
        agg,
        genotypes=sel_geno or None, treatments=sel_trt or None, sexes=sel_sex or None,
        show_raw=show_raw, raw_df=norm if show_raw else None,
    )
    st.pyplot(fig, width='stretch')
    st.download_button("Download figure", img, "climbing_curves.png", "image/png")

    with st.expander("Aggregated data table"):
        filt = agg.copy()
        if sel_geno: filt = filt[filt["Genotype"].isin(sel_geno)]
        if sel_trt:  filt = filt[filt["Treatment"].isin(sel_trt)]
        if sel_sex:  filt = filt[filt["Sex"].isin(sel_sex)]
        st.dataframe(filt, width='stretch')
        st.download_button("Download CSV", filt.to_csv(index=False).encode(),
                           "climbing_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Core Metrics
# ══════════════════════════════════════════════════════════════════════════════
def page_core_metrics():
    render_header("Core Metrics", "Performance Index, AUC, t50, and sex comparison")
    if not _require_metrics(): return

    base   = st.session_state["base_metrics"]
    pi_df  = base.get("pi",  pd.DataFrame())
    auc_df = base.get("auc", pd.DataFrame())
    t50_df = base.get("t50", pd.DataFrame())

    t_pi, t_auc, t_t50, t_sex = st.tabs(
        ["Performance Index", "AUC", "t50", "Sex Comparison"]
    )

    with t_pi:
        st.markdown("#### Performance Index (PI)")
        st.caption("PI = (climbed minus not climbed) / total x 100. Range: -100 to +100.")
        if not pi_df.empty:
            fig, img = pi_barplot(pi_df, title="Performance Index by Group")
            st.pyplot(fig, width='stretch')
            st.download_button("Download PI figure", img, "pi_barplot.png", "image/png")
            with st.expander("PI data table"):
                st.dataframe(pi_df, width='stretch')
                st.download_button("Download PI CSV", pi_df.to_csv(index=False).encode(),
                                   "pi_data.csv", "text/csv")
        else:
            st.warning("PI data not available.")

    with t_auc:
        st.markdown("#### Area Under the Climbing Curve (AUC)")
        st.caption("Trapezoidal AUC of percentage climbed vs time. Normalised range: 0 to 1.")
        if not auc_df.empty:
            sex_sel = st.radio("Sex", ["Male", "Female"], horizontal=True, key="auc_sex")
            fig, img = auc_heatmap(auc_df, sex=sex_sel)
            st.pyplot(fig, width='stretch')
            st.download_button("Download AUC heatmap", img,
                               f"auc_{sex_sel}.png", "image/png")
            with st.expander("AUC data table"):
                st.dataframe(auc_df, width='stretch')
                st.download_button("Download AUC CSV", auc_df.to_csv(index=False).encode(),
                                   "auc_data.csv", "text/csv")
        else:
            st.warning("AUC data not available.")

    with t_t50:
        st.markdown("#### Time to 50% Climbing (t50)")
        st.caption("Interpolated time at which 50% of each group climbed above threshold.")
        if not t50_df.empty:
            fig, img = t50_plot(t50_df)
            st.pyplot(fig, width='stretch')
            st.download_button("Download t50 figure", img, "t50_plot.png", "image/png")
            with st.expander("t50 data table"):
                st.dataframe(t50_df, width='stretch')
                st.download_button("Download t50 CSV", t50_df.to_csv(index=False).encode(),
                                   "t50_data.csv", "text/csv")
        else:
            st.warning("t50 not available. Ensure groups reach 50% climbing.")

    with t_sex:
        st.markdown("#### Male vs Female PI Comparison")
        if not pi_df.empty:
            fig, img = sex_comparison_plot(pi_df)
            st.pyplot(fig, width='stretch')
            st.download_button("Download sex comparison", img,
                               "sex_comparison.png", "image/png")
        else:
            st.warning("PI data required.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Novel Metrics
# ══════════════════════════════════════════════════════════════════════════════
def page_novel_metrics():
    render_header("Novel Metrics", "CRI, SDQ, and TSW composite metrics")
    if not _require_metrics(): return

    novel  = st.session_state.get("novel_metrics", {})
    cri_df = novel.get("cri", pd.DataFrame())
    sdq_df = novel.get("sdq", pd.DataFrame())
    tsw_df = novel.get("tsw", pd.DataFrame())

    t_cri, t_sdq, t_tsw = st.tabs(["CRI", "SDQ", "TSW"])

    with t_cri:
        st.markdown("#### Composite Rescue Index (CRI)")
        st.caption(
            "Integrates PI, AUC, and t50 rescue fractions. "
            "0% = no rescue; 100% = complete rescue to wild-type level."
        )
        if not cri_df.empty:
            fig, img = cri_barplot(cri_df)
            st.pyplot(fig, width='stretch')
            st.download_button("Download CRI figure", img, "cri_barplot.png", "image/png")
            with st.expander("CRI data table"):
                st.dataframe(cri_df, width='stretch')
                st.download_button("Download CRI CSV", cri_df.to_csv(index=False).encode(),
                                   "cri_data.csv", "text/csv")
        else:
            st.warning("CRI not computed. Requires wild-type genotype and vehicle control.")

    with t_sdq:
        st.markdown("#### Sex Dimorphism Quotient (SDQ)")
        st.caption("SDQ > 0.5 indicates strong sex-differential phenotype.")
        if not sdq_df.empty:
            fig, img = sdq_plot(sdq_df)
            st.pyplot(fig, width='stretch')
            st.download_button("Download SDQ figure", img, "sdq_plot.png", "image/png")
            if st.checkbox("Run SDQ permutation test (n=10,000)", key="sdq_perm"):
                pi_df = st.session_state["base_metrics"].get("pi", pd.DataFrame())
                if not pi_df.empty:
                    with st.spinner("Running permutation test..."):
                        perm_df = sdq_permutation_test(pi_df, n_perm=10000)
                    st.dataframe(perm_df, width='stretch')
                    st.download_button("Download permutation CSV",
                        perm_df.to_csv(index=False).encode(),
                        "sdq_permutation.csv", "text/csv")
            with st.expander("SDQ data table"):
                st.dataframe(sdq_df, width='stretch')
                st.download_button("Download SDQ CSV", sdq_df.to_csv(index=False).encode(),
                                   "sdq_data.csv", "text/csv")
        else:
            st.warning("SDQ not computed. Both male and female data required.")

    with t_tsw:
        st.markdown("#### Therapeutic Selectivity Window (TSW)")
        st.caption(
            "TSW > 1.0: treatment rescues disease more than it perturbs wild-type."
        )
        if not tsw_df.empty:
            fig, img = tsw_plot(tsw_df)
            st.pyplot(fig, width='stretch')
            st.download_button("Download TSW figure", img, "tsw_plot.png", "image/png")
            with st.expander("TSW data table"):
                st.dataframe(tsw_df, width='stretch')
                st.download_button("Download TSW CSV", tsw_df.to_csv(index=False).encode(),
                                   "tsw_data.csv", "text/csv")
        else:
            st.warning("TSW not computed. Requires wild-type genotype and vehicle control.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Motor Decline
# ══════════════════════════════════════════════════════════════════════════════




try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


# ── Demo survival data for MSCI (used when no real lifespan data is loaded) ──
_DEMO_SURVIVAL: dict = {
    # format: (Genotype, Treatment, Sex) -> mean_lifespan_days
    ("Oregon",  "Untreated",            "Male"):   62.3,
    ("Oregon",  "Untreated",            "Female"): 65.1,
    ("Oregon",  "Spermine",             "Male"):   66.8,
    ("Oregon",  "Spermine",             "Female"): 69.4,
    ("Oregon",  "Pantothenate",         "Male"):   64.2,
    ("Oregon",  "Pantothenate",         "Female"): 67.0,
    ("Oregon",  "Spermine+Pantothenate","Male"):   68.1,
    ("Oregon",  "Spermine+Pantothenate","Female"): 71.3,
    ("TDP43",   "Untreated",            "Male"):   38.4,
    ("TDP43",   "Untreated",            "Female"): 41.2,
    ("TDP43",   "Spermine",             "Male"):   43.7,
    ("TDP43",   "Spermine",             "Female"): 45.9,
    ("TDP43",   "Pantothenate",         "Male"):   42.1,
    ("TDP43",   "Pantothenate",         "Female"): 44.6,
    ("TDP43",   "Spermine+Pantothenate","Male"):   46.3,
    ("TDP43",   "Spermine+Pantothenate","Female"): 48.8,
    ("FUS",     "Untreated",            "Male"):   33.6,
    ("FUS",     "Untreated",            "Female"): 35.9,
    ("FUS",     "Spermine",             "Male"):   38.2,
    ("FUS",     "Spermine",             "Female"): 40.4,
    ("FUS",     "Pantothenate",         "Male"):   36.5,
    ("FUS",     "Pantothenate",         "Female"): 38.8,
    ("FUS",     "Spermine+Pantothenate","Male"):   40.1,
    ("FUS",     "Spermine+Pantothenate","Female"): 42.7,
}

_TCOLOURS = {
    "Untreated":             "#94A3B8",
    "Spermine":              "#3B82F6",
    "Pantothenate":          "#F0A500",
    "Spermine+Pantothenate": "#22C55E",
}
_GSYMBOLS = {"Oregon": "circle", "TDP43": "square", "FUS": "triangle-up"}



# ── Motor Decline helpers & page function ──────────────────────────────────
_DEMO_SURVIVAL_DATA = {
    ("Oregon",  "Untreated",             "Male"):   62.3,
    ("Oregon",  "Untreated",             "Female"): 65.1,
    ("Oregon",  "Spermine",              "Male"):   66.8,
    ("Oregon",  "Spermine",              "Female"): 69.4,
    ("Oregon",  "Pantothenate",          "Male"):   64.2,
    ("Oregon",  "Pantothenate",          "Female"): 67.0,
    ("Oregon",  "Spermine+Pantothenate", "Male"):   68.1,
    ("Oregon",  "Spermine+Pantothenate", "Female"): 71.3,
    ("TDP43",   "Untreated",             "Male"):   38.4,
    ("TDP43",   "Untreated",             "Female"): 41.2,
    ("TDP43",   "Spermine",              "Male"):   43.7,
    ("TDP43",   "Spermine",              "Female"): 45.9,
    ("TDP43",   "Pantothenate",          "Male"):   42.1,
    ("TDP43",   "Pantothenate",          "Female"): 44.6,
    ("TDP43",   "Spermine+Pantothenate", "Male"):   46.3,
    ("TDP43",   "Spermine+Pantothenate", "Female"): 48.8,
    ("FUS",     "Untreated",             "Male"):   33.6,
    ("FUS",     "Untreated",             "Female"): 35.9,
    ("FUS",     "Spermine",              "Male"):   38.2,
    ("FUS",     "Spermine",              "Female"): 40.4,
    ("FUS",     "Pantothenate",          "Male"):   36.5,
    ("FUS",     "Pantothenate",          "Female"): 38.8,
    ("FUS",     "Spermine+Pantothenate", "Male"):   40.1,
    ("FUS",     "Spermine+Pantothenate", "Female"): 42.7,
}
_MDR_TCOLOURS = {
    "Untreated":             "#94A3B8",
    "Spermine":              "#3B82F6",
    "Pantothenate":          "#F0A500",
    "Spermine+Pantothenate": "#22C55E",
}


def _msci_interpret(r):
    import math
    if math.isnan(float(r)):
        return "Insufficient data"
    r = float(r)
    if abs(r) < 0.3:
        return "Uncoupled (motor != survival)"
    if abs(r) < 0.6:
        return "Weakly coupled"
    if r >= 0.6:
        return "Strongly coupled (motor ~ survival)"
    return "Inverse coupling"


def _render_mdr_plotly(filt, sel_sex):
    import numpy as np
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.warning("Install plotly: pip install plotly")
        return
    genotypes = sorted(filt["Genotype"].dropna().unique().tolist())
    n = len(genotypes)
    if n == 0:
        st.warning("No data for selected filters.")
        return
    fig = make_subplots(
        rows=1, cols=n, subplot_titles=genotypes,
        shared_yaxes=True, horizontal_spacing=0.05,
    )
    added = set()
    for ci, geno in enumerate(genotypes, 1):
        sub = filt[filt["Genotype"] == geno]
        for _, row in sub.iterrows():
            trt    = str(row.get("Treatment", ""))
            sex    = str(row.get("Sex", ""))
            k      = float(row.get("k",   0) or 0)
            PI0    = float(row.get("PI0", 80) or 80)
            r2     = float(row.get("R2",  0) or 0)
            colour = _MDR_TCOLOURS.get(trt, "#64748B")
            t_fit  = np.linspace(0, 120, 200)
            pi_fit = PI0 * np.exp(-k * t_fit)
            lk = trt + "_line"
            fig.add_trace(go.Scatter(
                x=t_fit, y=pi_fit, mode="lines",
                line=dict(color=colour, dash="dash", width=2),
                name=trt + " k=" + str(round(k, 4)),
                legendgroup=trt, showlegend=(lk not in added),
                hovertemplate=(
                    "<b>" + trt + "</b><br>"
                    "k=" + str(round(k, 4)) + "  R2=" + str(round(r2, 2))
                    + "<extra>" + geno + " " + sex + "</extra>"
                ),
            ), row=1, col=ci)
            added.add(lk)
        fig.update_xaxes(title_text="Age (days)", row=1, col=ci)
    suffix = "" if sel_sex == "Both" else (" - " + sel_sex)
    fig.update_yaxes(title_text="PI (mean)", row=1, col=1)
    fig.update_layout(
        title_text="Motor Decline Rate - Exponential Decay" + suffix,
        title_font_size=15, height=440,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="v", x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#C5D7F0", borderwidth=1, font=dict(size=11),
        ),
        margin=dict(l=55, r=200, t=55, b=50),
        font=dict(color="#0D2137"),
    )
    st.plotly_chart(fig, width='stretch')


def _render_msci_results(msci_df, mdr_df):
    import numpy as np
    st.dataframe(msci_df, width='stretch')
    st.download_button(
        "Download MSCI Table (.csv)",
        msci_df.to_csv(index=False).encode(),
        "msci_results.csv", "text/csv", key="msci_dl",
    )
    try:
        import plotly.graph_objects as go
        surv_df = pd.DataFrame([
            {"Genotype": g, "Treatment": t, "Sex": s, "Lifespan_days": v}
            for (g, t, s), v in _DEMO_SURVIVAL_DATA.items()
        ])
        mc = [c for c in ["Genotype", "Treatment", "Sex"] if c in mdr_df.columns]
        merged = mdr_df.merge(surv_df, on=mc, how="inner")
        if merged.empty:
            return
        gcolors = {"Oregon": "#22C55E", "TDP43": "#3B82F6", "FUS": "#EF4444"}
        fig = go.Figure()
        for geno, sub in merged.groupby("Genotype"):
            fig.add_trace(go.Scatter(
                x=sub["k"].astype(float),
                y=sub["Lifespan_days"].astype(float),
                mode="markers",
                marker=dict(color=gcolors.get(geno, "#94A3B8"), size=9,
                            line=dict(width=1, color="#fff")),
                name=geno,
                text=[str(r.Treatment) + " " + str(r.Sex) for _, r in sub.iterrows()],
                hovertemplate="<b>%{text}</b><br>k=%{x:.4f}<br>Lifespan=%{y:.1f}d<extra></extra>",
            ))
        all_k  = merged["k"].astype(float).values
        all_ls = merged["Lifespan_days"].astype(float).values
        if len(all_k) > 2:
            coef = np.polyfit(all_k, all_ls, 1)
            xr   = np.linspace(all_k.min(), all_k.max(), 80)
            fig.add_trace(go.Scatter(
                x=xr, y=np.polyval(coef, xr),
                mode="lines", name="Trend",
                line=dict(color="#0D2137", dash="dash", width=1.5),
            ))
        fig.update_layout(
            title="k vs Lifespan - MSCI Scatter",
            xaxis_title="Motor Decline Rate k",
            yaxis_title="Mean Lifespan (days)",
            height=380,
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#0D2137"),
        )
        st.plotly_chart(fig, width='stretch')
    except Exception:
        pass


def page_motor_decline():
    render_header("Motor Decline MDR", "Exponential decay modelling of motor performance")
    if not (st.session_state.get('base_metrics') or st.session_state.get('base') or st.session_state.get('df') is not None):
        return

    # ── Get MDR table computed by run_pipeline (Pct_mean already renamed to PI_mean there)
    mdr_df = st.session_state.get("mdr_df", pd.DataFrame())
    base   = st.session_state["base_metrics"]
    agg    = base.get("aggregated", pd.DataFrame())

    # Build PI plot input: rename Pct_mean -> PI_mean if needed
    if not agg.empty:
        rename_map = {}
        if "Pct_mean" in agg.columns and "PI_mean" not in agg.columns:
            rename_map["Pct_mean"] = "PI_mean"
        if "Pct_sem" in agg.columns and "PI_sem" not in agg.columns:
            rename_map["Pct_sem"] = "PI_sem"
        pi_plot = agg.rename(columns=rename_map)
    else:
        pi_plot = pd.DataFrame()

    if mdr_df is None or (hasattr(mdr_df, "empty") and mdr_df.empty):
        st.warning(
            "MDR table not yet computed. "
            "Go to **Data Input** and click **Run Full Analysis Pipeline** first."
        )
        if not pi_plot.empty:
            with st.expander("Re-compute MDR now"):
                if st.button("Compute MDR", key="mdr_recompute", type="primary"):
                    try:
                        mdr_df = compute_mdr_table(pi_plot)
                        mdr_df = add_trajectory_labels(mdr_df)
                        st.session_state["mdr_df"] = mdr_df
                        st.success("MDR computed successfully.")
                        st.rerun()
                    except Exception as _e:
                        st.error(f"MDR computation failed: {_e}")
        return

    t_fit, t_pop, t_msci = st.tabs(["Decay Fits", "Population Summary", "Motor-Survival Coupling"])

    with t_fit:
        st.markdown("#### Exponential Decay Fits")
        st.caption("PI(t) = PI₀ · exp(-k·t).  k = Motor Decline Rate constant.")
        sex_sel = st.radio("Sex filter", ["Both", "Male", "Female"],
                           horizontal=True, key="mdr_sex")
        sexes = None if sex_sel == "Both" else sex_sel
        if not pi_plot.empty:
            try:
                fig, img = mdr_decay_curves(pi_plot, mdr_df, sexes=sexes)
                st.pyplot(fig, width=720)
                st.download_button("Download MDR figure", img, "mdr_decay.png", "image/png")
            except Exception as _fe:
                st.warning(f"Decay curve plot failed: {_fe}")
        st.markdown("#### MDR Parameter Table")

        def _flag_r2(v):
            try:
                return "background-color:#fff3cd" if float(v) < 0.80 else ""
            except Exception:
                return ""

        if "R2" in mdr_df.columns:
            styled = mdr_df.style.map(_flag_r2, subset=["R2"])
        else:
            styled = mdr_df.style
        st.dataframe(styled, width='stretch')
        st.caption("Yellow = R² < 0.80 (poor exponential fit).")
        st.download_button("Download MDR CSV",
                           mdr_df.to_csv(index=False).encode(),
                           "mdr_data.csv", "text/csv")

    with t_pop:
        st.markdown("#### Population-Level MDR Summary")
        try:
            pop = mdr_population_summary(mdr_df)
            if not pop.empty:
                st.dataframe(pop, width='stretch')
                st.download_button("Download population MDR CSV",
                                   pop.to_csv(index=False).encode(),
                                   "mdr_population.csv", "text/csv")
            else:
                st.info("Population summary not available.")
        except Exception as _pe:
            st.warning(f"Population summary failed: {_pe}")

    with t_msci:
        st.markdown("#### Motor-Survival Coupling Index (MSCI)")
        st.caption(
            "Spearman correlation between MDR k and median lifespan. "
            "Requires external lifespan data."
        )

        if mdr_df is None or (hasattr(mdr_df, "empty") and mdr_df.empty):
            st.warning("MDR table not computed. Run the pipeline from Data Input first.")
        else:
            # ── show MDR groups so user knows exact names needed ──────────
            _mcols = [c for c in ["Genotype","Treatment","Sex"] if c in mdr_df.columns]
            _mdr_genos = sorted(mdr_df["Genotype"].dropna().unique()) \
                         if "Genotype" in mdr_df.columns else []
            _mdr_trts  = sorted(mdr_df["Treatment"].dropna().unique()) \
                         if "Treatment" in mdr_df.columns else []
            _mdr_sexes = sorted(mdr_df["Sex"].dropna().unique()) \
                         if "Sex" in mdr_df.columns else []
            with st.expander("MDR groups in current dataset", expanded=False):
                st.caption("Your lifespan CSV must use these **exact** names.")
                _c1, _c2, _c3 = st.columns(3)
                _c1.markdown("**Genotypes**\n\n" + "\n".join(f"• `{g}`" for g in _mdr_genos))
                _c2.markdown("**Treatments**\n\n" + "\n".join(f"• `{t}`" for t in _mdr_trts))
                _c3.markdown("**Sexes**\n\n" + "\n".join(f"• `{s}`" for s in _mdr_sexes))

            # ── self-contained MSCI computation (bypasses module function) ─
            def _compute_msci_inline(mdr: "pd.DataFrame",
                                     ls:  "pd.DataFrame") -> "pd.DataFrame":
                """Spearman r between MDR k and lifespan. Tries multiple merge strategies."""
                import numpy as _np
                from scipy.stats import spearmanr as _spr
                _ls = ls.copy()
                # normalise lifespan column name
                for _alias in [
                    "Median_Survival_days","MedianSurvival_days","Lifespandays",
                    "Lifespan_days","LifespanDays","median_survival_days",
                    "MedianSurvivalDays","Median_survival_days",
                ]:
                    if _alias in _ls.columns and _alias != "Median_Survival_days":
                        _ls = _ls.rename(columns={_alias: "Median_Survival_days"})
                        break
                if "Median_Survival_days" not in _ls.columns:
                    return pd.DataFrame()
                # try merge strategies from most specific to least
                for _keys in [
                    [c for c in ["Genotype","Treatment","Sex"] if c in mdr.columns and c in _ls.columns],
                    [c for c in ["Genotype","Sex"]             if c in mdr.columns and c in _ls.columns],
                    [c for c in ["Genotype"]                   if c in mdr.columns and c in _ls.columns],
                ]:
                    if not _keys: continue
                    _mg = mdr.merge(_ls, on=_keys, how="inner")
                    if not _mg.empty:
                        break
                if _mg.empty:
                    return pd.DataFrame()
                _rows = []
                for _g, _sub in _mg.groupby("Genotype"):
                    _kv  = _sub["k"].astype(float)
                    _lsv = _sub["Median_Survival_days"].astype(float)
                    if len(_kv) < 2: continue
                    _r, _p = _spr(_kv, _lsv)
                    _rows.append({"Genotype":_g, "MSCI_r":round(float(_r),3),
                                  "p_value":round(float(_p),4),
                                  "n_groups":len(_sub),
                                  "Interpretation": (
                                      "Strongly coupled" if abs(_r)>=0.6 else
                                      "Weakly coupled"   if abs(_r)>=0.3 else
                                      "Uncoupled")})
                if len(_mg) >= 2:
                    _ra, _pa = _spr(_mg["k"].astype(float),
                                    _mg["Median_Survival_days"].astype(float))
                    _rows.append({"Genotype":"ALL",
                                  "MSCI_r":round(float(_ra),3),
                                  "p_value":round(float(_pa),4),
                                  "n_groups":len(_mg),
                                  "Interpretation": (
                                      "Strongly coupled" if abs(_ra)>=0.6 else
                                      "Weakly coupled"   if abs(_ra)>=0.3 else
                                      "Uncoupled")})
                return pd.DataFrame(_rows)

            # ── Demo button ───────────────────────────────────────────────
            st.info(
                "**No lifespan file?** Click **Load Demo Lifespan** to compute MSCI "
                "using synthetic survival values matched to your current groups."
            )
            if st.button("Load Demo Lifespan & Compute MSCI",
                         key="demo_msci_btn", type="primary"):
                import re as _re
                _WT  = _re.compile(r"oregon|wild.?type|^wt$|control|n2|w1118|canton",
                                   _re.I)
                _DIS = _re.compile(r"tdp|fus|sod|atxn|htt|park|snca|lrrk|pink|"  
                                    r"dj.?1|mutant|als|^pd$|^hd$|^ad$", _re.I)
                def _surv(g): 
                    return 65.0 if _WT.search(g) else 38.0 if _DIS.search(g) else 52.0
                _ls_rows = []
                for _, _r in mdr_df.drop_duplicates(
                        subset=[c for c in ["Genotype","Treatment","Sex"]
                                if c in mdr_df.columns]).iterrows():
                    _g = str(_r.get("Genotype",""))
                    _t = str(_r.get("Treatment",""))
                    _s = str(_r.get("Sex",""))
                    _tmod = 2.5 * _mdr_trts.index(_t) if _t in _mdr_trts else 0.0
                    _smod = 2.0 if "female" in _s.lower() else 0.0
                    _ls_rows.append({"Genotype":_g,"Treatment":_t,"Sex":_s,
                                     "Median_Survival_days":round(_surv(_g)+_tmod+_smod,1)})
                _ls_demo = pd.DataFrame(_ls_rows)
                _msci_out = _compute_msci_inline(mdr_df, _ls_demo)
                if not _msci_out.empty:
                    st.success(f"MSCI computed — {len(_msci_out)} groups")
                    st.dataframe(_msci_out, width='stretch')
                    st.download_button("Download MSCI CSV",
                        _msci_out.to_csv(index=False).encode(),
                        "msci_demo.csv","text/csv",key="dlmsci_demo")
                else:
                    st.error("MSCI still empty after inline computation. "
                             "MDR table may lack k column or scipy is missing.")
                    with st.expander("Debug: MDR table preview"):
                        st.dataframe(mdr_df.head(6), width='stretch')
                        st.dataframe(_ls_demo.head(6), width='stretch')

            st.divider()
            st.markdown("**Or upload your own lifespan CSV:**")
            with st.expander("Upload lifespan CSV"):
                st.caption("Columns: `Genotype`, `Sex`, `Treatment`, `MedianSurvival_days`")
                with st.expander("Download blank template", expanded=False):
                    if _mdr_genos:
                        import itertools as _it2
                        _tmpl = pd.DataFrame([
                            {"Genotype":g,"Treatment":t,"Sex":s,"MedianSurvival_days":""}
                            for g,t,s in _it2.product(_mdr_genos,_mdr_trts,_mdr_sexes)
                        ])
                        st.dataframe(_tmpl, width='stretch')
                        st.download_button("Download template CSV",
                            _tmpl.to_csv(index=False).encode(),
                            "lifespan_template.csv","text/csv",key="tmpl_dl")
                ls_file = st.file_uploader("Lifespan CSV",type=["csv"],key="ls_upload")
                if ls_file:
                    try:
                        _lsdf = pd.read_csv(ls_file)
                        _msci_up = _compute_msci_inline(mdr_df, _lsdf)
                        if not _msci_up.empty:
                            st.dataframe(_msci_up, width='stretch')
                            st.download_button("Download MSCI CSV",
                                _msci_up.to_csv(index=False).encode(),
                                "msci.csv","text/csv")
                        else:
                            st.warning("MSCI empty — check column names match MDR table.")
                            with st.expander("Debug: columns in uploaded file"):
                                st.write(list(_lsdf.columns))
                                st.dataframe(_lsdf.head(5), width='stretch')
                    except Exception as _me:
                        st.error(f"MSCI error: {_me}")

def _render_mdr_plotly(filt: pd.DataFrame, sel_sex: str) -> None:
    """Render clean MDR decay chart with plotly — faceted by Genotype."""
    genotypes = sorted(filt["Genotype"].dropna().unique().tolist())
    n_geno    = len(genotypes)

    # One column per genotype
    fig = make_subplots(
        rows=1, cols=n_geno,
        subplot_titles=genotypes,
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    treatments_all = sorted(filt["Treatment"].dropna().unique().tolist())
    added_to_legend: set = set()
    t_max = 0.0

    for col_i, geno in enumerate(genotypes, start=1):
        sub = filt[filt["Genotype"] == geno]

        for _, row in sub.iterrows():
            trt     = row.get("Treatment", "")
            sex_lbl = row.get("Sex", "")
            k       = float(row.get("k", 0) or 0)
            PI0     = float(row.get("PI0", 80) or 80)
            r2      = float(row.get("R2", 0) or 0)
            colour  = _TCOLOURS.get(trt, "#64748B")
            sym     = _GSYMBOLS.get(geno, "circle")

            # Observed data points (from PI_timeseries if available)
            pi_ts_col = next((c for c in sub.columns if "pi_ts" in c.lower() or "pi_t" in c.lower()), None)
            t_vals_col = next((c for c in sub.columns if "t_vals" in c.lower() or "times" in c.lower()), None)

            if pi_ts_col and t_vals_col:
                try:
                    t_obs  = np.array(row[t_vals_col])
                    pi_obs = np.array(row[pi_ts_col])
                    t_max  = max(t_max, float(t_obs.max()))
                    leg_key = f"{trt}_pts"
                    fig.add_trace(go.Scatter(
                        x=t_obs, y=pi_obs, mode="markers",
                        marker=dict(color=colour, symbol=sym, size=7,
                                    line=dict(width=1, color="#fff")),
                        name=trt,
                        legendgroup=trt,
                        showlegend=(leg_key not in added_to_legend),
                        hovertemplate=f"<b>{trt}</b><br>t=%{{x:.0f}}d PI=%{{y:.1f}}<extra>{geno} {sex_lbl}</extra>",
                    ), row=1, col=col_i)
                    added_to_legend.add(leg_key)
                except Exception:
                    pass

            # Fitted decay curve
            t_fit  = np.linspace(0, max(t_max, 120), 120)
            pi_fit = PI0 * np.exp(-k * t_fit)
            fit_key = f"{trt}_fit"
            fig.add_trace(go.Scatter(
                x=t_fit, y=pi_fit, mode="lines",
                line=dict(color=colour, dash="dash", width=2),
                name=f"{trt} fit",
                legendgroup=f"{trt}_fit",
                showlegend=(fit_key not in added_to_legend),
                hovertemplate=(
                    f"<b>{trt}</b> fitted curve<br>"
                    f"k={k:.4f} R\u00b2={r2:.2f}"
                    f"<extra>{geno} {sex_lbl}</extra>"
                ),
            ), row=1, col=col_i)
            added_to_legend.add(fit_key)

        fig.update_xaxes(title_text="Age (days)", row=1, col=col_i)

    sex_label = "" if sel_sex == "Both" else f" \u2014 {sel_sex}"
    fig.update_layout(
        title_text=f"Motor Decline Rate \u2014 Exponential Decay Fits{sex_label}",
        title_font_size=16,
        height=480,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#C5D7F0",
            borderwidth=1,
            font=dict(size=11),
            tracegroupgap=4,
        ),
        margin=dict(l=60, r=180, t=60, b=50),
        font=dict(family="sans-serif", color="#0D2137"),
    )
    fig.update_yaxes(title_text="PI (mean)", row=1, col=1)
    st.plotly_chart(fig, width='stretch')


def _compute_msci_internal(mdr_df: pd.DataFrame, surv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute MSCI = Pearson r between MDR (k) and lifespan per group."""
    merge_cols = [c for c in ["Genotype", "Treatment", "Sex"] if c in mdr_df.columns]
    merged = mdr_df.merge(surv_df, on=merge_cols, how="inner")
    if merged.empty:
        raise ValueError("No matching rows between MDR and survival data. "
                         "Check Genotype/Treatment/Sex column values match exactly.")

    # Group-level MSCI: Pearson r over (k, Lifespan) across all groups
    rows = []
    for geno, sub in merged.groupby("Genotype"):
        if len(sub) < 3:
            continue
        try:
            from scipy.stats import pearsonr
            r, p = pearsonr(sub["k"].astype(float), sub["Lifespan_days"].astype(float))
        except Exception:
            r, p = float("nan"), float("nan")
        rows.append({
            "Genotype":     geno,
            "MSCI (r)":     round(float(r), 3),
            "p-value":      round(float(p), 4) if not np.isnan(p) else None,
            "Interpretation": _msci_interpret(r),
            "n groups":     len(sub),
        })

    # Overall
    try:
        from scipy.stats import pearsonr
        r_all, p_all = pearsonr(
            merged["k"].astype(float),
            merged["Lifespan_days"].astype(float),
        )
    except Exception:
        r_all, p_all = float("nan"), float("nan")
    rows.append({
        "Genotype":     "ALL",
        "MSCI (r)":     round(float(r_all), 3),
        "p-value":      round(float(p_all), 4),
        "Interpretation": _msci_interpret(r_all),
        "n groups":     len(merged),
    })
    return pd.DataFrame(rows)


def _msci_interpret(r: float) -> str:
    if np.isnan(r):
        return "Insufficient data"
    if abs(r) < 0.3:
        return "Uncoupled (motor \u2260 survival)"
    if abs(r) < 0.6:
        return "Weakly coupled"
    if r >= 0.6:
        return "Strongly coupled (motor \u2248 survival)"
    return "Inverse coupling (motor rescue \u2260 lifespan)"


def _render_msci(msci_df: pd.DataFrame) -> None:
    """Display MSCI table + scatter."""
    st.markdown("#### MSCI Results")
    st.dataframe(msci_df, width='stretch')

    csv = msci_df.to_csv(index=False).encode()
    st.download_button(
        "\u2b07 Download MSCI Table (.csv)", csv,
        "msci_results.csv", "text/csv", key="msci_dl",
    )

    # Scatter: k vs Lifespan_days coloured by Genotype
    try:
        st.session_state.get("mdr_df")  # side-effect-free check
        mdr_df_ = st.session_state["mdr_df"]
        surv_df_ = pd.DataFrame([
            {"Genotype": g, "Treatment": t, "Sex": s, "Lifespan_days": v}
            for (g, t, s), v in _DEMO_SURVIVAL.items()
        ])
        merge_cols = [c for c in ["Genotype", "Treatment", "Sex"] if c in mdr_df_.columns]
        merged_    = mdr_df_.merge(surv_df_, on=merge_cols, how="inner")
        if merged_.empty or not _PLOTLY:
            return

        geno_colors = {"Oregon": "#22C55E", "TDP43": "#3B82F6", "FUS": "#EF4444"}
        fig = go.Figure()
        for geno, sub in merged_.groupby("Genotype"):
            col = geno_colors.get(geno, "#94A3B8")
            fig.add_trace(go.Scatter(
                x=sub["k"].astype(float),
                y=sub["Lifespan_days"].astype(float),
                mode="markers",
                marker=dict(color=col, size=9, opacity=0.85,
                            line=dict(width=1, color="#fff")),
                name=geno,
                text=[f"{r.Treatment} {r.Sex}" for _, r in sub.iterrows()],
                hovertemplate="<b>%{text}</b><br>k=%{x:.4f}<br>Lifespan=%{y:.1f}d<extra></extra>",
            ))
        # Trend line
        from numpy.polynomial import polynomial as P
        all_k  = merged_["k"].astype(float).values
        all_ls = merged_["Lifespan_days"].astype(float).values
        if len(all_k) > 2:
            c_fit = P.polyfit(all_k, all_ls, 1)
            x_rng = np.linspace(all_k.min(), all_k.max(), 80)
            fig.add_trace(go.Scatter(
                x=x_rng, y=P.polyval(x_rng, c_fit),
                mode="lines", name="Trend",
                line=dict(color="#0D2137", dash="dash", width=1.5),
                showlegend=True,
            ))
        fig.update_layout(
            title="Motor Decline Rate vs Lifespan (MSCI scatter)",
            xaxis_title="Motor Decline Rate k",
            yaxis_title="Mean Lifespan (days)",
            height=400,
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(color="#0D2137"),
            legend=dict(bordercolor="#C5D7F0", borderwidth=1),
        )
        st.plotly_chart(fig, width='stretch')
    except Exception:
        pass


def page_trajectory_clustering():
    render_header("Trajectory Clustering", "DTW-based unsupervised trajectory clustering")
    if not _require_metrics(): return

    cr = st.session_state.get("cluster_result")

    if cr is None or cr.get("error"):
        err = cr.get("error", "Not computed.") if cr else "Not computed."
        st.warning(f"Clustering: {err}")
        if st.button("Run Trajectory Clustering", type="primary"):
            base = st.session_state["base_metrics"]
            agg  = base.get("aggregated", pd.DataFrame())
            if not agg.empty:
                with st.spinner("Running DTW clustering..."):
                    cr = cluster_trajectories(agg if "PI_mean" in agg.columns else agg.rename(columns={"Pct_mean": "PI_mean"}))
                st.session_state["cluster_result"] = cr
                if cr.get("error"):
                    st.error(cr["error"])
                else:
                    st.success(f"Clustering complete: {cr['n_clusters']} clusters.")
                    st.rerun()
        return

    n_cl  = cr.get("n_clusters", 0)
    stab  = cr.get("stability", np.nan)
    ct    = cr.get("cluster_table", pd.DataFrame())
    sil_df = cr.get("silhouette_table", pd.DataFrame())

    m = st.columns(3)
    m[0].metric("Clusters",            str(n_cl))
    m[1].metric("Bootstrap stability", f"{stab:.3f}" if pd.notna(stab) else "N/A")
    m[2].metric("Trajectories",        str(len(cr.get("labels", []))))

    if pd.notna(stab) and stab < 0.7:
        st.warning("Bootstrap stability < 0.70. Interpret cluster assignments cautiously.")

    t_dend, t_embed, t_summary, t_sil = st.tabs(
        ["Dendrogram", "2D Embedding", "Cluster Summary", "Silhouette"]
    )

    with t_dend:
        fig, img = trajectory_dendrogram(cr)
        st.pyplot(fig, width='stretch')
        st.download_button("Download dendrogram", img, "dendrogram.png", "image/png")

    with t_embed:
        use_umap = st.checkbox("Use UMAP (requires umap-learn)", value=False)
        fig, img = trajectory_pca(cr, use_umap=use_umap)
        st.pyplot(fig, width='stretch')
        st.download_button("Download embedding", img, "embedding.png", "image/png")

    with t_summary:
        summary = cluster_summary_table(cr)
        if not summary.empty:
            st.dataframe(summary, width='stretch')
        if not ct.empty:
            ct_i = interpret_clusters(ct, cr.get("archetypes", {}))
            with st.expander("Full cluster membership table"):
                st.dataframe(ct_i, width='stretch')
                st.download_button("Download cluster CSV",
                    ct_i.to_csv(index=False).encode(), "cluster_table.csv", "text/csv")

    with t_sil:
        if not sil_df.empty:
            st.dataframe(sil_df, width='stretch')
            best_k = sil_df.loc[sil_df["silhouette"].idxmax(), "k"]
            st.caption(f"Optimal k = {best_k} (highest silhouette score).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — TCS
# ══════════════════════════════════════════════════════════════════════════════
def page_tcs():
    render_header("Translational Score (TCS)",
                  "Evidence-weighted translational candidate ranking")
    if not _require_metrics(): return

    tcs_r = st.session_state.get("tcs_result")
    if tcs_r is None:
        st.warning("TCS not computed. Rerun the pipeline from Data Input.")
        return

    tcs_full = tcs_r.get("tcs_full", pd.DataFrame())
    tcs_sum  = tcs_r.get("tcs_summary", pd.DataFrame())
    radar_ax = tcs_r.get("radar_axes", [])
    radar_d  = tcs_r.get("radar_data", {})

    if tcs_full is None or tcs_full.empty:
        st.warning("TCS could not be computed. Ensure CRI and TSW are available.")
        return

    t_ranked, t_radar, t_summary, t_weight = st.tabs(
        ["Ranked Bar", "Radar Chart", "Summary Table", "Weight Configuration"]
    )

    with t_ranked:
        top_n = st.slider("Candidates to display", 5, 30, 15, key="tcs_topn")
        fig, img = tcs_ranked_bar(tcs_full, top_n=top_n)
        st.pyplot(fig, width='stretch')
        st.download_button("Download ranked bar", img, "tcs_ranked.png", "image/png")

    with t_radar:
        if radar_ax and radar_d:
            fig, img = tcs_radar(radar_ax, radar_d)
            st.pyplot(fig, width='stretch')
            st.download_button("Download radar chart", img, "tcs_radar.png", "image/png")
        else:
            st.info("Radar data not available.")

    with t_summary:
        if not tcs_sum.empty:
            st.markdown("#### Treatment-level TCS Summary")
            st.dataframe(tcs_sum, width='stretch')
            st.download_button("Download TCS summary CSV",
                tcs_sum.to_csv(index=False).encode(), "tcs_summary.csv", "text/csv")
        st.markdown("#### Full TCS Table")
        st.dataframe(tcs_full, width='stretch')
        st.download_button("Download full TCS CSV",
            tcs_full.to_csv(index=False).encode(), "tcs_full.csv", "text/csv")

    with t_weight:
        st.markdown("#### Reconfigure TCS Weights")
        st.caption("Default: CRI 35%, TSW 25%, SDQ 15%, MDR 15%, Trajectory 10%. Must sum to 100%.")
        col = st.columns(5)
        w_cri  = col[0].number_input("CRI (%)",  0, 100, 35, 5, key="w_cri")
        w_tsw  = col[1].number_input("TSW (%)",  0, 100, 25, 5, key="w_tsw")
        w_sdq  = col[2].number_input("SDQ (%)",  0, 100, 15, 5, key="w_sdq")
        w_mdr  = col[3].number_input("MDR (%)",  0, 100, 15, 5, key="w_mdr")
        w_traj = col[4].number_input("Traj (%)", 0, 100, 10, 5, key="w_traj")
        total_w = w_cri + w_tsw + w_sdq + w_mdr + w_traj
        st.caption(f"Weight sum: {total_w}%  {'(valid)' if total_w == 100 else '-- must equal 100%'}")
        if st.button("Recompute TCS", disabled=(total_w != 100), type="primary"):
            nw = {"CRI": w_cri/100, "TSW": w_tsw/100,
                  "SDQ": w_sdq/100, "MDR": w_mdr/100, "Traj": w_traj/100}
            with st.spinner("Recomputing TCS..."):
                new_tcs = run_tcs_pipeline(
                    base_metrics=st.session_state["base_metrics"],
                    novel_metrics=st.session_state["novel_metrics"],
                    mdr_df=st.session_state.get("mdr_df"),
                    cluster_result=st.session_state.get("cluster_result"),
                    weights=nw,
                )
            st.session_state["tcs_result"] = new_tcs
            st.success("TCS recomputed.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
def page_stats():
    render_header('Statistical Analysis', 'Significance testing')
    if not _require_metrics():
        return
    base  = st.session_state["base_metrics"]
    novel = st.session_state.get("novel_metrics", {}) or {}
    pi_df = base.get("pi", pd.DataFrame())
    if pi_df.empty:
        st.warning("PI data required. Run pipeline first.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        group_col = st.selectbox("Compare groups by", ["Treatment", "Genotype", "Sex"])
    with c2:
        sex_filter = st.radio("Sex filter", ["All", "Male", "Female"], horizontal=True, key="statsex")
    def _rc(df, candidates):
        for c in candidates:
            if df is not None and not df.empty and c in df.columns:
                return c
        return candidates[0] if candidates else ''
    pi_col  = _rc(pi_df, ["PI_mean", "PI", "pi_mean"])
    auc_df  = base.get("auc",  pd.DataFrame())
    t50_df  = base.get("t50",  pd.DataFrame())
    cri_df  = novel.get("cri", pd.DataFrame())
    sdq_df  = novel.get("sdq", pd.DataFrame())
    auc_col = _rc(auc_df, ["AUC_mean", "AUC"])
    t50_col = _rc(t50_df, ["t50_mean", "t50"])
    cri_col = _rc(cri_df, ["CRI", "CRI_mean"])
    sdq_col = _rc(sdq_df, ["SDQ", "SDQ_mean"])
    METRIC_OPTIONS = {}
    METRIC_OPTIONS["Performance Index (PI)"] = (pi_col, pi_df)
    if not auc_df.empty and auc_col: METRIC_OPTIONS["AUC"] = (auc_col, auc_df)
    if not t50_df.empty and t50_col: METRIC_OPTIONS["t50"] = (t50_col, t50_df)
    if not cri_df.empty and cri_col: METRIC_OPTIONS["CRI"] = (cri_col, cri_df)
    if not sdq_df.empty and sdq_col: METRIC_OPTIONS["SDQ"] = (sdq_col, sdq_df)
    with c3:
        metricsel_label = st.selectbox("Metric", list(METRIC_OPTIONS.keys()))
    value_col, metricdf = METRIC_OPTIONS[metricsel_label]
    if sex_filter != "All" and "Sex" in metricdf.columns:
        metricdf = metricdf[metricdf["Sex"] == sex_filter].copy()
    if metricdf.empty:
        st.warning(f"No data for {metricsel_label} after filtering.")
        return
    if value_col not in metricdf.columns:
        num_cols = [c for c in metricdf.columns if pd.api.types.is_numeric_dtype(metricdf[c])]
        if not num_cols:
            st.warning(f"Column {value_col!r} not found."); return
        value_col = num_cols[0]; st.info(f"Column auto-resolved to {value_col!r}")
    if group_col not in metricdf.columns:
        st.warning(f"Grouping column {group_col!r} not found."); return
    data_dict = {str(k): grp[value_col].dropna().values
                 for k, grp in metricdf.groupby(group_col)
                 if len(grp[value_col].dropna()) >= 2}
    if len(data_dict) < 2:
        st.warning(f"Need >= 2 groups. Found: {list(data_dict.keys())}"); return
    st.caption(f"Testing **{value_col}** grouped by **{group_col}** | {len(data_dict)} groups")
    if st.button("Run Statistical Test", type="primary"):
        with st.spinner("Running tests..."):
            result = auto_test(data_dict)
            st.session_state["stats_result"] = result
    res = st.session_state.get("stats_result")
    if res:
        method = res.get("method", "N/A")
        st.markdown(f"**Method:** `{method}`")
        main = res.get("main_result", {})
        if main:
            mc = st.columns(min(len(main), 4))
            for i, (k, v) in enumerate(main.items()):
                try: _mval = round(float(v), 4)
                except Exception: _mval = str(v)
                mc[i % 4].metric(k.replace("_", " "), _mval)
        t_norm, t_pair, t_lev = st.tabs(["Normality", "Pairwise Comparisons", "Homogeneity of Variance"])
        with t_norm:
            norm_df = res.get("normality", pd.DataFrame())
            if isinstance(norm_df, pd.DataFrame) and not norm_df.empty:
                st.dataframe(norm_df, width="stretch")
                st.download_button("Download normality CSV", norm_df.to_csv(index=False).encode(), "normality.csv", "text/csv")
            else: st.info("Normality results not available.")
        with t_pair:
            pw_df = res.get("pairwise", pd.DataFrame())
            if isinstance(pw_df, pd.DataFrame) and not pw_df.empty:
                st.dataframe(pw_df, width="stretch")
                st.download_button("Download pairwise CSV", pw_df.to_csv(index=False).encode(), "pairwise.csv", "text/csv")
                st.markdown("| Symbol | p threshold | Meaning |\n|--------|-------------|--------|\n| `***` | p<0.001 | Highly significant |\n| `**` | p<0.01 | Very significant |\n| `*` | p<0.05 | Significant |\n| `ns` | p>=0.05 | Not significant |")
                st.caption("p-values BH-FDR corrected. Effect sizes shown where available.")
            else: st.info("Pairwise comparisons not available.")
        with t_lev:
            lev = res.get("levene", {})
            if lev:
                lev_stat = lev.get("Levene_statistic", float("nan"))
                lev_p    = lev.get("p_value",          float("nan"))
                lev_eq   = lev.get("Equal_variance",   "Unknown")
                lc1, lc2, lc3 = st.columns(3)
                try:
                    lc1.metric("Levene Statistic", round(float(lev_stat), 4))
                    lc2.metric("p-value",           round(float(lev_p),   6))
                except Exception:
                    lc1.metric("Levene Statistic", str(lev_stat))
                    lc2.metric("p-value",           str(lev_p))
                lc3.metric("Equal Variance?", str(lev_eq))
                try:
                    if float(lev_p) > 0.05: st.success("Variances homogeneous (p>0.05). ANOVA assumption satisfied.")
                    else: st.warning("Variances heterogeneous (p<=0.05). Welch or Kruskal-Wallis preferred.")
                except Exception: pass
                st.caption("Levene test (centre=median). p>0.05 = equal variances assumed.")
            else: st.info("Levene test result not available.")
    st.divider()
    st.markdown("#### Bootstrap 95% Confidence Intervals")
    if st.button("Compute Bootstrap CI (2000 resamples)", key="bootcibtn"):
        rng = np.random.default_rng(42)
        ci_rows = []
        for grp_key, sub in metricdf.groupby(group_col):
            vals = sub[value_col].dropna().to_numpy(dtype=float)
            if len(vals) < 2:
                ci_rows.append({group_col: grp_key, 'n': len(vals), 'Mean': float(vals.mean()) if len(vals) else float('nan'), 'CI95_lo': float('nan'), 'CI95_hi': float('nan'), 'CI_width': float('nan'), 'Note': 'n<2'})
                continue
            boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)])
            lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            ci_rows.append({group_col: grp_key, 'n': len(vals), 'Mean': round(float(vals.mean()), 4), 'CI95_lo': round(lo, 4), 'CI95_hi': round(hi, 4), 'CI_width': round(hi-lo, 4), 'Note': ''})
        ci_df = pd.DataFrame(ci_rows)
        if not ci_df.empty:
            st.dataframe(ci_df, width="stretch")
            st.download_button("Download CI CSV", ci_df.to_csv(index=False).encode(), "bootstrap_ci.csv", "text/csv", key="dl_bootci")
        else: st.warning("Bootstrap returned no rows.")
    st.divider()
    st.markdown("#### SDQ Permutation Test (n=10 000)")
    if st.checkbox("Run SDQ permutation test", key="stats_sdq_perm"):
        pi_src = base.get("pi", pd.DataFrame())
        if pi_src.empty: st.warning("PI data required.")
        else:
            with st.spinner("Running 10 000 permutations..."):
                perm_df = sdq_permutation_test(pi_src, n_perm=10_000)
            if not perm_df.empty:
                st.dataframe(perm_df, width="stretch")
                st.download_button("Download SDQ permutation CSV", perm_df.to_csv(index=False).encode(), "sdq_permutation.csv", "text/csv")
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — Live Recording
# ══════════════════════════════════════════════════════════════════════════════
def page_recording():
    try:
        from modules.auto_recording import render as _ar_render
        _ar_render(st.subheader, st.info, st.warning, st.success)
    except ImportError:
        render_header( "Camera configuration and recording control")
        st.error("auto_recording module not found. Place auto_recording.py in modules/ folder.")
    except Exception as _e:
        render_header( "Camera configuration and recording control")
        st.error(f"Recording page error: {_e}")
        import traceback
        st.code(traceback.format_exc())


def page_video_analysis():
    render_header("Video Analysis",
                  "Computer-vision fly detection and counting from video")
    try:
        import cv2 as _cv2_check
        cv2_ok = True
    except ImportError:
        cv2_ok = False
    if not cv2_ok:
        st.error("OpenCV required: pip install opencv-python")
        return

    show_vial_protocol()
    t_single, t_batch, t_det = st.tabs(
        ["Single Video","Batch Processing","Detector Settings"]
    )

    with t_det:
        st.markdown("#### Fly Detector Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            threshold_frac  = st.slider("Threshold line", 0.1, 0.9, 0.5, 0.05)
            min_area        = st.number_input("Min fly area (px)", 5, 200, 20)
            max_area        = st.number_input("Max fly area (px)", 50, 2000, 800)
        with c2:
            use_bg_sub      = st.checkbox("Background subtraction", value=True)
            use_morph       = st.checkbox("Morphological cleanup",  value=True)
            morph_kernel    = st.slider("Morph kernel size", 2, 9, 3, 1)
        with c3:
            preset_name     = st.selectbox("Vial layout preset", list(VIAL_PRESETS.keys()))
            sample_interval = st.slider("Sample every N seconds", 1, 30, 5)
        det_cfg = DetectorConfig(
            threshold_frac=threshold_frac,
            min_area=min_area, max_area=max_area,
            use_bg_subtraction=use_bg_sub,
            use_morphology=use_morph,
            morph_kernel_size=morph_kernel,
            vial_preset=preset_name,
            sample_interval_sec=sample_interval,
        )
        st.session_state["_detector_cfg"] = det_cfg
        st.success("Detector settings saved.")

    with t_single:
        st.markdown("#### Single Video Analysis")
        st.caption("Counts flies above/below the threshold line at each sampled timepoint.")
        vid_file = st.file_uploader("Upload video", type=["mp4","avi","mov"], key="sv")
        if vid_file:
            with tempfile.NamedTemporaryFile(
                suffix=Path(vid_file.name).suffix, delete=False
            ) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name
            info = get_video_info(tmp_path)
            if info:
                ic = st.columns(4)
                ic[0].metric("Duration (s)",  f"{info.get('duration_sec',0):.1f}")
                ic[1].metric("FPS",           f"{info.get('fps',0):.1f}")
                ic[2].metric("Resolution",    info.get("resolution","N/A"))
                ic[3].metric("Total frames",  str(info.get("total_frames",0)))

            det_cfg = st.session_state.get("_detector_cfg", DetectorConfig())
            n_v = st.number_input("Number of vials", 1, 24, 6, key="sv_nv")
            vid_vc = []
            for i in range(int(n_v)):
                row = st.columns([2,2,2,1])
                g = row[0].text_input(f"Genotype {i+1}", key=f"svg_{i}")
                t = row[1].text_input(f"Treatment {i+1}", key=f"svt_{i}")
                s = row[2].selectbox(f"Sex {i+1}", ["Male","Female"], key=f"svs_{i}")
                n = row[3].number_input(f"n {i+1}", 1, 50, 20, key=f"svn_{i}")
                vid_vc.append({"Genotype":g,"Treatment":t,"Sex":s,"expected_n":int(n)})

            if st.button("Run Video Analysis", type="primary"):
                with st.spinner("Analysing video... this may take several minutes."):
                    analyzer   = VideoAnalyzer(video_path=tmp_path,
                                               detector_config=det_cfg,
                                               vial_configs=vid_vc)
                    counts_df  = analyzer.analyze()
                if counts_df is None or counts_df.empty:
                    st.error("No data returned. Check detector settings.")
                else:
                    st.success(f"Analysis complete: {len(counts_df)} records.")
                    dp_df = video_counts_to_drophenix(counts_df, vid_vc)
                    st.dataframe(dp_df.head(100), width='stretch')
                    st.download_button("Download counts CSV",
                        dp_df.to_csv(index=False).encode(), "video_counts.csv", "text/csv")
                    if st.button("Load into DroPhenix"):
                        df_raw, warns, errs = load_data(io.StringIO(dp_df.to_csv(index=False)))
                        st.session_state.update({
                            "df": df_raw, "data_source": "video",
                            "data_warnings": warns, "data_errors": errs,
                            "run_complete": False, "base_metrics": {},
                        })
                        st.success("Data loaded. Go to Data Input to run the pipeline.")

    with t_batch:
        st.markdown("#### Batch Video Processing")
        batch_files = st.file_uploader("Upload multiple videos",
                                       type=["mp4","avi","mov"],
                                       accept_multiple_files=True,
                                       key="bv")
        if batch_files:
            st.info(f"{len(batch_files)} videos selected.")
            det_cfg = st.session_state.get("_detector_cfg", DetectorConfig())
            if st.button("Run Batch Analysis", type="primary"):
                all_dfs  = []
                prog     = st.progress(0)
                for i, vf in enumerate(batch_files):
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(vf.name).suffix, delete=False
                    ) as tmp:
                        tmp.write(vf.read())
                        tmp_path = tmp.name
                    with st.spinner(f"Analysing {vf.name}..."):
                        an   = VideoAnalyzer(video_path=tmp_path,
                                             detector_config=det_cfg, vial_configs=[])
                        df_i = an.analyze()
                    if df_i is not None and not df_i.empty:
                        df_i["source_file"] = vf.name
                        all_dfs.append(df_i)
                    prog.progress((i + 1) / len(batch_files))
                if all_dfs:
                    combined = pd.concat(all_dfs, ignore_index=True)
                    st.success(f"Batch complete: {len(combined)} records.")
                    st.dataframe(combined.head(200), width='stretch')
                    st.download_button("Download batch CSV",
                        combined.to_csv(index=False).encode(),
                        "batch_counts.csv", "text/csv")
                else:
                    st.error("No data extracted from any video.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — CSV Upload and Analysis
# ══════════════════════════════════════════════════════════════════════════════
def page_csv_upload():
    render_header("CSV Upload and Analysis",
                  "Retrospective analysis from exported or pre-existing CSV files")
    st.caption("Accepted: long-form counts or wide-form timepoint columns.")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"],
                                key="csv_page")
    if not uploaded:
        st.info("Upload a file to begin.")
        return

    df_raw, warns, errs = load_data(uploaded)
    if errs:
        for e in errs: st.error(e)
        return
    for w in warns: st.warning(w)
    st.success(f"File loaded: {len(df_raw)} rows.")

    with st.expander("Preview (first 100 rows)"):
        st.dataframe(df_raw.head(100), width='stretch')

    val_r = validate_dataframe(df_raw)
    for e in val_r.get("errors", []):   st.error(e)
    for w in val_r.get("warnings", []): st.warning(w)

    groups = get_groups(df_raw)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Genotypes",  str(len(groups["genotypes"])))
    c2.metric("Treatments", str(len(groups["treatments"])))
    c3.metric("Sexes",      str(len(groups["sexes"])))
    c4.metric("Timepoints", str(len(groups["timepoints"])))

    col_wt, col_veh = st.columns(2)
    with col_wt:
        wt  = st.selectbox("Wild-type genotype",
            ["(auto-detect)"] + sorted(groups["genotypes"]), key="csv_wt")
    with col_veh:
        veh = st.selectbox("Vehicle treatment",
            ["(auto-detect)"] + sorted(groups["treatments"]), key="csv_veh")

    if st.button("Load and Run Full Analysis", type="primary", use_container_width=True):
        wt_sel  = None if wt  == "(auto-detect)" else wt
        veh_sel = None if veh == "(auto-detect)" else veh
        st.session_state.update({
            "df": df_raw, "data_source": "csv_upload",
            "data_warnings": warns, "data_errors": errs,
            "run_complete": False, "base_metrics": {},
        })
        _run_pipeline(df_raw, wt_sel, veh_sel)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 13 — Export and Download
# ══════════════════════════════════════════════════════════════════════════════
def page_export():
    render_header("Export and Download",
                  "Batch export of all figures and results")
    if not _require_metrics(): return

    base  = st.session_state["base_metrics"]
    novel = st.session_state.get("novel_metrics", {})
    tcs_r = st.session_state.get("tcs_result")
    mdr   = st.session_state.get("mdr_df", pd.DataFrame())
    cr    = st.session_state.get("cluster_result")

    st.markdown('<p class="sec-hdr">Individual CSV Downloads</p>', unsafe_allow_html=True)
    csv_exports = {
        "PI data":          base.get("pi",  pd.DataFrame()),
        "AUC data":         base.get("auc", pd.DataFrame()),
        "t50 data":         base.get("t50", pd.DataFrame()),
        "CRI data":         novel.get("cri", pd.DataFrame()),
        "SDQ data":         novel.get("sdq", pd.DataFrame()),
        "TSW data":         novel.get("tsw", pd.DataFrame()),
        "MDR data":         mdr if mdr is not None else pd.DataFrame(),
        "TCS full":         tcs_r["tcs_full"]    if tcs_r else pd.DataFrame(),
        "TCS summary":      tcs_r["tcs_summary"] if tcs_r else pd.DataFrame(),
        "Cluster table":    cr["cluster_table"]  if cr and "cluster_table" in cr else pd.DataFrame(),
    }
    cols = st.columns(3)
    for i, (name, df_e) in enumerate(csv_exports.items()):
        fn = name.lower().replace(" ","_") + ".csv"
        if df_e is not None and not df_e.empty:
            cols[i % 3].download_button(
                f"Download {name}",
                df_e.to_csv(index=False).encode(),
                fn, "text/csv", key=f"exp_{i}",
            )
        else:
            cols[i % 3].caption(f"{name}: not available")

    st.divider()
    st.markdown('<p class="sec-hdr">Batch Figure Export</p>', unsafe_allow_html=True)
    st.caption("Exports all figure types at 300 DPI for manuscript preparation.")
    fmt     = st.radio("Format", ["png","svg","pdf"], horizontal=True)
    out_dir = st.text_input("Export directory", value="drophenix_figures")

    if st.button("Export All Figures", type="primary"):
        results_dict = {
            "agg_df":         base.get("aggregated", pd.DataFrame()),
            "pi_df":          base.get("pi",  pd.DataFrame()),
            "auc_df":         base.get("auc", pd.DataFrame()),
            "t50_df":         base.get("t50", pd.DataFrame()),
            "cri_df":         novel.get("cri", pd.DataFrame()),
            "sdq_df":         novel.get("sdq", pd.DataFrame()),
            "tsw_df":         novel.get("tsw", pd.DataFrame()),
            "tcs_result":     tcs_r,
            "mdr_df":         mdr,
            "cluster_result": cr,
        }
        with st.spinner("Exporting figures..."):
            saved = export_all_figures(results_dict, out_dir=out_dir, fmt=fmt)
        st.success(f"{len(saved)} figures exported to {out_dir}/")
        with st.expander("Exported files"):
            for p in saved: st.text(p)

    st.divider()
    st.markdown('<p class="sec-hdr">Full Report ZIP</p>', unsafe_allow_html=True)
    if st.button("Build and Download ZIP"):
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, df_e in csv_exports.items():
                if df_e is not None and not df_e.empty:
                    fn = name.lower().replace(" ","_") + ".csv"
                    zf.writestr(f"csv/{fn}", df_e.to_csv(index=False))
        buf.seek(0)
        st.download_button("Download ZIP", buf.read(),
                           "drophenix_report.zip", "application/zip")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

# ── Home page ─────────────────────────────────────────────────────────────────
def page_home():
    """DroPhenix home -- NAR-standard landing page."""
    from datetime import datetime as _dt
    import base64 as _b64m
    from pathlib import Path as _P

    # ── Logo ──────────────────────────────────────────────────────────────
    _dpx_b64 = ''
    try:
        _lp = _P('logs/Drophenix_Logo.png')
        if _lp.exists():
            with open(_lp, 'rb') as _fh:
                _dpx_b64 = _b64m.b64encode(_fh.read()).decode()
    except Exception:
        pass

    _logo = (
        '<img src="data:image/png;base64,' + _dpx_b64 + '"'
        ' style="height:130px;width:130px;object-fit:contain;flex-shrink:0;'
        'filter:drop-shadow(0 6px 18px rgba(240,165,0,0.35));">'
        if _dpx_b64 else
        '<div style="width:130px;height:130px;flex-shrink:0;border-radius:50%;'
        'background:linear-gradient(135deg,#F0A500,#C47F00);'
        'display:flex;align-items:center;justify-content:center;'
        'font-size:1.6rem;font-weight:900;color:#fff;'
        'box-shadow:0 6px 18px rgba(240,165,0,0.35);">DP</div>'
    )

    # ── Capability pills ──────────────────────────────────────────────────
    _pill_css = (
        'background:rgba(240,165,0,0.14);color:#F0A500;'
        'border:1px solid rgba(240,165,0,0.32);border-radius:20px;'
        'font-size:.71rem;font-weight:700;padding:3px 13px;'
        'margin:2px;display:inline-block;letter-spacing:.03em;'
    )
    _pills = [
        'Negative Geotaxis Assay', 'Performance Index', 'AUC &amp; t50',
        'CRI &middot; SDQ &middot; TSW', 'Motor Decline Rate',
        'DTW Trajectory Clustering', 'Translational Score',
        'Sex-Stratified Analysis', '300 DPI Publication Figures', 'BH-FDR Statistics',
    ]
    _ph = ''.join(
        '<span style="' + _pill_css + '">' + p + '</span>'
        for p in _pills
    )

    # ── HERO ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(135deg,#09192A 0%,#0D2137 60%,#0F2D44 100%);'
        'border-radius:18px;padding:3rem 3.5rem;margin:0.5rem 0 2rem 0;'
        'display:flex;align-items:center;gap:3rem;'
        'border:1px solid rgba(240,165,0,0.22);'
        'box-shadow:0 10px 40px rgba(0,0,0,0.30);">'
        + _logo
        + '<div style="flex:1;">'
        + '<div style="font-size:.72rem;font-weight:800;letter-spacing:.18em;'
        'text-transform:uppercase;color:#F0A500;margin-bottom:.5rem;">'
        'Science for Society &nbsp;&middot;&nbsp; SSSIHL &nbsp;&middot;&nbsp; SAI-Net</div>'
        + '<div style="font-size:2.55rem;font-weight:900;color:#FFFFFF;'
        'line-height:1.05;margin-bottom:.4rem;letter-spacing:-.01em;">'
        'DroPhenix <span style="color:#F0A500;">Analytics</span>'
        ' <span style="font-size:1rem;font-weight:600;color:#64748B;vertical-align:middle;">v2.0</span></div>'
        + '<div style="font-size:1.02rem;color:#94A3B8;font-weight:500;'
        'margin-bottom:1rem;letter-spacing:.01em;">'
        'Drosophila Climbing &amp; Motor Assay Analysis Platform</div>'
        + '<p style="font-size:.9rem;color:#CBD5E1;line-height:1.8;'
        'max-width:680px;margin-bottom:1.1rem;">'
        'DroPhenix automates the end-to-end analysis pipeline for the '
        '<em>negative geotaxis</em> climbing assay in '
        '<em>Drosophila melanogaster</em> &mdash; the gold-standard '
        '<em>in vivo</em> model for neuromotor phenotyping. From raw vial counts '
        'to publication-ready figures, it integrates kinetic metrics, '
        'novel composite rescue indices, sex-stratified analysis, '
        'unsupervised trajectory clustering, and multi-criteria '
        'translational scoring in a single reproducible workflow.'
        '</p>'
        + '<div>' + _ph + '</div>'
        + '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Why DroPhenix — 3 scientific innovations ──────────────────────────
    st.markdown(
        '<p style="font-size:1.08rem;font-weight:800;color:#0D2137;'
        'margin:0.4rem 0 1rem;padding-bottom:.35rem;'
        'border-bottom:3px solid #F0A500;display:inline-block;">'
        'Why DroPhenix?</p>',
        unsafe_allow_html=True,
    )
    _wi1, _wi2, _wi3 = st.columns(3)
    _innovations = [
        (
            _wi1,
            '&#9889; Novel Composite Metrics',
            (
                'Three purpose-built metrics transcend the classical Performance Index. '
                'The <b>Composite Rescue Index (CRI)</b> integrates PI, AUC, and t50 '
                'into one normalised rescue score (0&ndash;100%). '
                'The <b>Sex Dimorphism Quotient (SDQ)</b> quantifies sex-differential '
                'drug response &mdash; critical for translational validity. '
                'The <b>Therapeutic Selectivity Window (TSW)</b> measures disease-selective '
                'rescue vs. wild-type perturbation (TSW&nbsp;&gt;&nbsp;1.0 = safe).'
            ),
        ),
        (
            _wi2,
            '&#128200; Motor Decline Kinetics',
            (
                'DroPhenix fits an exponential decay model '
                'PI(t)&nbsp;=&nbsp;PI&#8320;&nbsp;&middot;&nbsp;e<sup>&minus;kt</sup> '
                'to each group trajectory, extracting the '
                '<b>Motor Decline Rate (MDR)</b> constant <em>k</em>, '
                'half-life, and R&sup2;. '
                'Delta-<em>k</em> quantifies treatment-mediated rescue of the decline rate '
                '&mdash; a kinetic disease-modification readout not captured '
                'by endpoint PI alone. Motor-Survival Coupling Index (MSCI) '
                'links motor decline to lifespan via Spearman correlation.'
            ),
        ),
        (
            _wi3,
            '&#127919; Translational Candidate Scoring',
            (
                'The <b>Translational Candidate Score (TCS)</b> integrates CRI (35%), '
                'TSW (25%), SDQ consistency (15%), MDR rescue (15%), and trajectory '
                'phenotype (10%) into a weighted composite that ranks treatments by '
                'translational profile &mdash; prioritising compounds with broad, safe, '
                'and sustained rescue over high-efficacy but sex-discordant or '
                'WT-perturbing candidates. Outputs Tier I&ndash;V drug priority labels.'
            ),
        ),
    ]
    for _col, _title, _body in _innovations:
        with _col:
            st.markdown(
                '<div style="background:#fff;border:1px solid #C5D7F0;'
                'border-top:4px solid #F0A500;border-radius:10px;'
                'padding:1.2rem 1.3rem;height:100%;">'
                '<div style="font-size:.93rem;font-weight:700;color:#0D2137;'
                'margin-bottom:.6rem;">' + _title + '</div>'
                '<p style="font-size:.83rem;color:#374151;line-height:1.78;margin:0;">'
                + _body + '</p></div>',
                unsafe_allow_html=True,
            )

    # ── 5-step Workflow ───────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.08rem;font-weight:800;color:#0D2137;'
        'margin:0.4rem 0 1rem;padding-bottom:.35rem;'
        'border-bottom:3px solid #F0A500;display:inline-block;">'
        'Analysis Workflow</p>',
        unsafe_allow_html=True,
    )
    _steps = [
        ('1', 'Load Data',
         'Upload CSV/Excel with raw vial counts, or load the built-in SSSIHL demo dataset '
         '(Oregon-R WT, TDP-43, FUS &middot; 3 treatments &middot; both sexes &middot; 8 time-points).'),
        ('2', 'Configure Groups',
         'Specify wild-type genotype and vehicle control. '
         'DroPhenix auto-detects Genotype &times; Treatment &times; Sex groups and validates all columns.'),
        ('3', 'Run Pipeline',
         'One click computes PI, AUC, t50, CRI, SDQ, TSW, MDR, '
         'trajectory clusters, TCS, and full statistical tests end-to-end.'),
        ('4', 'Explore',
         'Navigate each analytical module in the sidebar. '
         'Every plot has filter controls, data table views, and individual figure downloads.'),
        ('5', 'Export',
         'Download all 10 metric tables (CSV) and all figures at 300&nbsp;DPI '
         '(PNG / SVG / PDF) as a single ZIP for journal submission.'),
    ]
    _scols = st.columns(5)
    for _col, (_num, _stitle, _sdesc) in zip(_scols, _steps):
        with _col:
            st.markdown(
                '<div style="background:#fff;border:1px solid #C5D7F0;border-radius:10px;'
                'padding:1rem .9rem;text-align:center;height:100%;">'
                '<div style="width:34px;height:34px;border-radius:50%;'
                'background:#F0A500;color:#fff;font-size:.85rem;font-weight:900;'
                'display:flex;align-items:center;justify-content:center;'
                'margin:0 auto .55rem;">' + _num + '</div>'
                '<div style="font-size:.85rem;font-weight:700;color:#0D2137;'
                'margin-bottom:.35rem;">' + _stitle + '</div>'
                '<div style="font-size:.78rem;color:#6B7280;line-height:1.65;">'
                + _sdesc + '</div></div>',
                unsafe_allow_html=True,
            )

    # ── Platform Modules grid ─────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.08rem;font-weight:800;color:#0D2137;'
        'margin:0.4rem 0 1rem;padding-bottom:.35rem;'
        'border-bottom:3px solid #F0A500;display:inline-block;">'
        'Platform Modules</p>',
        unsafe_allow_html=True,
    )
    _MODS = [
        ('Data Input', 'Input / QC',
         'Accepts long-format or wide-format CSV/Excel. Validates required columns, '
         'flags outlier replicates, and runs the full computation pipeline with one click.'),
        ('Climbing Curves', 'Kinetics',
         'Time-resolved mean&nbsp;&plusmn;&nbsp;SEM climbing performance per group. '
         'Computes trapezoidal AUC (0&ndash;1) and interpolated t50. '
         'Individual replicate overlay available.'),
        ('Core Metrics', 'Primary Outcomes',
         'Performance Index, AUC heatmaps per sex, t50 plots, '
         'and male vs. female PI comparison. All plots exportable at 300&nbsp;DPI.'),
        ('Novel Metrics', 'Composite Scores',
         '<b>CRI</b>: Composite Rescue Index. '
         '<b>SDQ</b>: Sex Dimorphism Quotient (&gt;0.5 = strong dimorphism). '
         '<b>TSW</b>: Therapeutic Selectivity Window (&gt;1.0 = disease-selective).'),
        ('Motor Decline', 'Decay Modelling',
         'Non-linear exponential decay fitting. Reports <em>k</em>, PI&#8320;, R&sup2;, '
         'and &Delta;<em>k</em> (treatment rescue). Motor-Survival Coupling Index (MSCI).'),
        ('Trajectory Clustering', 'Unsupervised ML',
         'DTW pairwise distance + Ward-linkage clustering. Silhouette-optimal k. '
         'Archetypes: Sustained Rescue, Transient, Progressive Decline, Plateau.'),
        ('Translational Score', 'Drug Ranking',
         'Weighted TCS integrating CRI, TSW, SDQ, MDR, and trajectory stability. '
         'Tier I&ndash;V drug priority labels with radar visualisation.'),
        ('Statistical Analysis', 'Inference',
         'Shapiro&ndash;Wilk &rarr; Levene &rarr; ANOVA/Kruskal&ndash;Wallis. '
         'BH-FDR pairwise tests. Bootstrap 95% CI. SDQ permutation test.'),
        ('Live Recording', 'Data Acquisition',
         'OpenCV webcam recording. Configurable FPS, resolution, duration. '
         'Timestamp overlay, pre-assay QC, and pause/resume. Saves MP4.'),
        ('Video Analysis', 'Computer Vision',
         'MOG2 background subtraction + contour detection for automated fly counting. '
         'Single and batch mode. Exports DroPhenix-compatible CSV.'),
        ('CSV Upload', 'Re-analysis',
         'Re-analyse any external or exported counts CSV. '
         'Full pipeline with configurable WT and vehicle labels.'),
        ('Export', 'Dissemination',
         '10 metric CSVs &plus; all figures at 300&nbsp;DPI. '
         'One-click ZIP with README for reproducible journal submission.'),
    ]
    for _ri in range(0, len(_MODS), 4):
        _row = _MODS[_ri:_ri + 4]
        _rcols = st.columns(len(_row))
        for _col, (_name, _cat, _desc) in zip(_rcols, _row):
            with _col:
                st.markdown(
                    '<div style="background:#fff;border:1px solid #C5D7F0;'
                    'border-radius:10px;padding:1.05rem 1.1rem;'
                    'margin-bottom:.8rem;min-height:160px;">'
                    '<div style="display:flex;align-items:center;'
                    'justify-content:space-between;margin-bottom:.45rem;">'
                    '<span style="font-size:.9rem;font-weight:800;color:#0D2137;">'
                    + _name + '</span>'
                    '<span style="background:#EBF4FF;color:#1E40AF;font-size:.65rem;'
                    'font-weight:700;padding:2px 9px;border-radius:20px;">'
                    + _cat + '</span></div>'
                    '<p style="font-size:.8rem;color:#4B5563;line-height:1.72;margin:0;">'
                    + _desc + '</p></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)

    # ── Citation block ────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.08rem;font-weight:800;color:#0D2137;'
        'margin:0.4rem 0 1rem;padding-bottom:.35rem;'
        'border-bottom:3px solid #F0A500;display:inline-block;">'
        'How to Cite</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:#F8FAFF;border:1px solid #C5D7F0;'
        'border-left:5px solid #F0A500;border-radius:0 10px 10px 0;'
        'padding:1.1rem 1.4rem;font-size:.84rem;'
        'color:#1a2332;line-height:1.9;">'
        'Gunanathan K, Sivaramakrishnan V. (2026). '
        '<em>DroPhenix Analytics v2.0: An integrated computational platform for '
        'Drosophila climbing assay phenotyping and translational drug scoring.</em> '
        'SSSIHL / SAI-Net. Available at: '
        '<a href="https://sssihl.edu.in" style="color:#1E40AF;">https://sssihl.edu.in</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Research Team ─────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:1.08rem;font-weight:800;color:#0D2137;'
        'margin:0.4rem 0 1rem;padding-bottom:.35rem;'
        'border-bottom:3px solid #F0A500;display:inline-block;">'
        'Research Team</p>',
        unsafe_allow_html=True,
    )
    _ta, _tb = st.columns(2)
    with _ta:
        st.markdown(
            '<div style="background:#fff;border:1px solid #C5D7F0;'
            'border-radius:10px;padding:1.1rem 1.3rem;">'
            '<span style="background:#DCFCE7;color:#166534;font-size:.68rem;'
            'font-weight:700;padding:2px 10px;border-radius:20px;">'
            'Principal Investigator</span>'
            '<div style="font-size:1rem;font-weight:700;color:#0D2137;margin:.45rem 0 .2rem;">'
            'Prof. Venketesh Sivaramakrishnan</div>'
            '<div style="font-size:.8rem;color:#374151;line-height:1.65;">'
            'Department of Biosciences<br>'
            'Sri Sathya Sai Institute of Higher Learning (SSSIHL)<br>'
            'Puttaparthi, Andhra Pradesh, India</div></div>',
            unsafe_allow_html=True,
        )
    with _tb:
        st.markdown(
            '<div style="background:#fff;border:1px solid #C5D7F0;'
            'border-radius:10px;padding:1.1rem 1.3rem;">'
            '<div style="font-size:.72rem;color:#64748B;font-weight:600;margin-bottom:.3rem">Developer &amp; PhD Scholar</div>'
            '<div style="font-size:1rem;font-weight:700;color:#0D2137;margin:.45rem 0 .2rem;">'
            'Krishnasalini Gunanathan</div>'
            '<div style="font-size:.8rem;color:#374151;line-height:1.65;">'
            'Doctoral Research Scholar, Biosciences, SSSIHL<br>'
            'SAI-Net &nbsp;&middot;&nbsp; BrainSafe AI &nbsp;&middot;&nbsp; DroPhenix Analytics</div></div>',
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;font-size:.7rem;color:#94A3B8;'
        'margin-top:2.5rem;padding-top:1rem;border-top:1px solid #C5D7F0;">'
        + 'DroPhenix Analytics v2.0 &nbsp;&middot;&nbsp; SAI-Net '
        + '&nbsp;&middot;&nbsp; SSSIHL &nbsp;&middot;&nbsp; '
        + 'Science for Society &nbsp;&middot;&nbsp; GPL-3.0 '
        + '&nbsp;&middot;&nbsp; ' + _dt.now().strftime('%B %Y')
        + '</div>',
        unsafe_allow_html=True,
    )

PAGE_MAP = {
    "Home":                   page_home,
    "Data Input":             page_data_input,
    "Climbing Curves":        page_climbing_curves,
    "Core Metrics":           page_core_metrics,
    "Novel Metrics":          page_novel_metrics,
    "Motor Decline":          page_motor_decline,
    "Trajectory Clustering":  page_trajectory_clustering,
    "Translational Score":    page_tcs,
    "Statistical Analysis":   page_stats,
    "Live Recording":         page_recording,
    "Video Analysis":         page_video_analysis,
    "CSV Upload":             page_csv_upload,
    "Export":                 page_export,
}

PAGE_MAP[page]()