"""
DroPhenix Analytics v2.0
Drosophila Climbing and Motor Assay Analysis Platform
SSSIHL Department of Bioscience 
"""

# -- Version --

from __future__ import annotations


# -- Version --
_VERSION = "3.0"

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
        endpoint_pi_by_age,
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
    "Plan a Run",
    "Data Input",
    "Design Check",
    "Climbing Curves",
    "Core Metrics",
    "Novel Metrics",
    "Motor Decline",
    "Survival Analysis",
    "Y-maze",
    "Activity Monitoring (DAMS)",
    "DAM Run Analysis",
    "Larval Crawling",
    "Grooming",
    "Dose-Response",
    "Time-of-Day",
    "Drug Ranking",
    "Trajectory Clustering",
    "Trajectory Forecast",
    "Translational Score",
    "Statistical Analysis",
    "Hierarchical Statistics",
    "Survivor Bias",
    "Historical Controls",
    "Live Recording",
    "Video Analysis",
    "Climbing Video",
    "Climbing (Simple)",
    "Draw Vials",
    "Video Scoring",
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


def _template_button(filename: str, label: str = "Download data template",
                     key: Optional[str] = None):
    """Offer the matching CSV template (from templates/) as a download."""
    p = Path("templates") / filename
    if p.exists():
        st.download_button(label, p.read_bytes(), filename, "text/csv",
                           key=key or f"tpl_{filename}")
    else:
        st.caption(f"(template {filename} not found)")


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
    # MDR = motor decline over AGE (days), NOT within-assay seconds. Only run it
    # when the data carries an Age column; otherwise leave it empty with a status.
    try:
        if "Age" in df.columns and df["Age"].notna().any():
            age_pi = endpoint_pi_by_age(df)
            st.session_state["mdr_age_pi"] = age_pi
            st.session_state["mdr_df"] = compute_mdr_table(age_pi, time_col="Age")
        else:
            empty = pd.DataFrame()
            empty.attrs["mdr_status"] = ("requires_age_data: MDR is the decline of "
                "climbing PI over AGE. Add an 'Age' (days) column with >=2 ages per "
                "group. Within-assay seconds are not an age axis.")
            st.session_state["mdr_df"] = empty
            st.session_state["mdr_age_pi"] = pd.DataFrame()
    except Exception as e:
        warnings.warn(f"MDR: {e}")
        st.session_state["mdr_df"] = pd.DataFrame()
        st.session_state["mdr_age_pi"] = pd.DataFrame()

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
            ["Demo dataset", "Demo: aging series (for MDR)",
             "Upload CSV or Excel", "Use recorded video data"],
            horizontal=True,
        )
        if src == "Demo: aging series (for MDR)":
            st.caption(
                "Longitudinal climbing across ages 5–45 days (Oregon-WT vs TDP43, "
                "Vehicle vs Drug). Climbing PI declines with age; this enables "
                "Motor Decline Rate (MDR). Includes an `Age` column."
            )
            if st.button("Load Aging Demo", type="primary", key="load_aging"):
                from modules import synthetic as _syn
                df_raw, warns, errs = load_data(_syn.generate_climbing_aging())
                st.session_state.update({
                    "df": df_raw, "data_source": "demo_aging",
                    "data_warnings": warns, "data_errors": errs,
                    "run_complete": False, "base_metrics": {},
                })
                st.success(f"Aging demo loaded: {len(df_raw)} rows, "
                           f"{df_raw['Age'].nunique()} ages.")
        elif src == "Demo dataset":
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
            _template_button("climbing_template.csv", "Download climbing template", "tpl_clim")
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
        _status = mdr_df.attrs.get("mdr_status", "") if hasattr(mdr_df, "attrs") else ""
        if "requires_age_data" in str(_status):
            st.info(
                "**Motor Decline Rate is the decline of climbing performance over "
                "AGE (days)** — negative geotaxis is age-dependent (Gargano et al. "
                "2005). This dataset has no age axis, so MDR is not applicable.\n\n"
                "To use MDR, upload longitudinal climbing data with an **`Age`** "
                "(days) column (≥2 ages per group); the assay's per-second "
                "timecourse is used for the climbing curves, AUC and t₅₀, not for "
                "decline. A ready template is on the Data Input page."
            )
            return
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
        st.markdown("#### Exponential Decay Fits over Age")
        st.caption("PI(age) = PI₀ · exp(−k·age).  k = Motor Decline Rate "
                   "(per day). Climbing PI declines with age; faster k = faster "
                   "neurodegeneration, lower k under drug = rescue.")
        sex_sel = st.radio("Sex filter", ["Both", "Male", "Female"],
                           horizontal=True, key="mdr_sex")
        sexes = None if sex_sel == "Both" else sex_sel
        age_pi = st.session_state.get("mdr_age_pi", pd.DataFrame())
        if age_pi is not None and not age_pi.empty:
            try:
                fig, img = mdr_decay_curves(age_pi, mdr_df, sexes=sexes,
                                            time_col="Age", xlabel="Age (days)")
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
# PAGE — Video Scoring (climbing_video pipeline)
# ══════════════════════════════════════════════════════════════════════════════
_VS_DEFAULT_TIMES = [5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
_VS_SPEC_FIELDS = ("x0", "x1", "y_top", "y_bottom", "line_y")


def _vs_stamp(path: str) -> float:
    """File identity for the caches: a re-encoded file at the same path is new."""
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return 0.0


def _vs_specs_to_vials(specs):
    """Rebuild VialSpec objects from the plain tuples the caches are keyed on."""
    from modules.climbing_video import VialSpec
    return [VialSpec(x0=int(a), x1=int(b), y_top=int(c), y_bottom=int(d),
                     line_y=int(e), label=str(lab), source=str(src))
            for (lab, a, b, c, d, e, src) in specs]


def _vs_spec_tuple(s: dict) -> tuple:
    return (str(s["label"]), int(s["x0"]), int(s["x1"]), int(s["y_top"]),
            int(s["y_bottom"]), int(s["line_y"]), str(s.get("source", "manual")))


_VS_VIDEO_EXT = (".mp4", ".mov", ".avi", ".m4v", ".mkv")

# Calibration and scoring rule are carried around the page as plain tuples for
# the same reason the regions are: `st.cache_data` keys on its arguments, and a
# dataclass instance is a new object on every rerun.
_VS_CAL_MODES = {
    "not_calibrated": "Not calibrated",
    "tube_width_mm": "Cylinder outside diameter (mm)",
    "tube_height_mm": "Tube height in shot (mm)",
    "px_per_mm": "Pixels per mm, given directly",
}
_VS_RULE_MODES = {
    "drawn_line": "Marked line on the cylinder",
    "height_fraction": "Fraction of tube height",
    "height_mm": "Height above the base in mm",
}


def _vs_calibration(calib: tuple):
    """Rebuild a Calibration from the plain tuple the caches are keyed on."""
    if not calib or calib[0] == "not_calibrated":
        return None
    from modules.climbing_video import Calibration
    kind, value = str(calib[0]), float(calib[1])
    return Calibration(**{kind: value}, source=_VS_CAL_MODES.get(kind, kind))


def _vs_rule(rule: tuple):
    """Rebuild a ScoringRule from the plain tuple the caches are keyed on."""
    from modules.climbing_video import ScoringRule
    if not rule:
        return ScoringRule(mode="drawn_line")
    mode = str(rule[0])
    if mode == "height_fraction":
        return ScoringRule(mode="height_fraction", fraction=float(rule[1]))
    if mode == "height_mm":
        return ScoringRule(mode="height_mm", height_mm=float(rule[1]))
    return ScoringRule(mode="drawn_line")


def _vs_calibration_controls(prefix: str) -> tuple:
    """
    The calibration control group, shared by the single-video and batch flows.

    Returns the plain tuple the caches are keyed on, or () when the recording is
    left uncalibrated.
    """
    st.caption(
        "Give **one** known physical dimension and the pixels-to-millimetres "
        "scale follows. This matters more than it looks: a count above a "
        "hand-drawn line is not comparable between laboratories, or even "
        "between two recordings in the same laboratory, because it depends on "
        "where somebody drew the line and how far away the camera was. Height "
        "climbed **in millimetres** is a physical quantity, comparable to "
        "anyone, and it is what the RING assay was originally defined on.")
    c1, c2 = st.columns([2, 1])
    mode = c1.radio("Known dimension", list(_VS_CAL_MODES),
                    format_func=lambda k: _VS_CAL_MODES[k], index=1,
                    key=f"{prefix}_cal_mode")
    if mode == "not_calibrated":
        c2.caption("Heights will be reported only as a fraction of tube "
                   "height, which is comparable within this recording but not "
                   "between recordings at different zoom levels.")
        return ()
    if mode == "tube_width_mm":
        val = c2.number_input(
            "Outside diameter (mm)", 1.0, 500.0, 25.0, 0.5,
            key=f"{prefix}_cal_width",
            help="A 50 ml graduated cylinder is about 25 mm across; a standard "
                 "Drosophila vial is about 23.5 mm. Diameter is usually the "
                 "easier dimension to state accurately, and it is measured "
                 "across the region the detector already knows.")
    elif mode == "tube_height_mm":
        val = c2.number_input(
            "Tube height in shot (mm)", 1.0, 2000.0, 150.0, 1.0,
            key=f"{prefix}_cal_height",
            help="The height of the graduated portion actually visible in the "
                 "frame, not the catalogue height of the cylinder.")
    else:
        val = c2.number_input("Pixels per mm", 0.1, 500.0, 5.0, 0.1,
                              key=f"{prefix}_cal_ppm")
    return (str(mode), float(val))


def _vs_rule_controls(prefix: str, calibrated: bool) -> tuple:
    """The scoring-rule selector, shared by the single-video and batch flows."""
    st.caption(
        "The rule is applied to stored fly **heights**, not baked into the "
        "detection, so the same recording can be re-scored under a "
        "collaborator's criterion without touching the video again.")
    c1, c2 = st.columns([2, 1])
    mode = c1.radio("How a fly counts as climbed", list(_VS_RULE_MODES),
                    format_func=lambda k: _VS_RULE_MODES[k], index=0,
                    key=f"{prefix}_rule_mode")
    if mode == "height_fraction":
        frac = c2.slider("Fraction of tube height", 0.05, 0.95, 0.5, 0.05,
                         key=f"{prefix}_rule_frac",
                         help="Needs no mark on the glass at all.")
        return ("height_fraction", float(frac))
    if mode == "height_mm":
        hmm = c2.number_input(
            "Height above the base (mm)", 1.0, 1000.0, 80.0, 1.0,
            key=f"{prefix}_rule_mm", disabled=not calibrated,
            help="The standard RING criterion, and the only rule that is "
                 "directly comparable between laboratories.")
        if not calibrated:
            st.warning(
                "A height in millimetres needs a calibration. Give the "
                "cylinder diameter or the tube height above; until then the "
                "marked line is used instead.")
        return ("height_mm", float(hmm))
    return ("drawn_line", 0.0)


def _vsb_new_manifest():
    """A different manifest is now in play: drop the old batch result and give
    the editor a fresh key."""
    st.session_state["_vsb_manifest_n"] = (
        int(st.session_state.get("_vsb_manifest_n", 0)) + 1)
    st.session_state.pop("_vsb_result", None)
    st.session_state.pop("_vsb_sig", None)


def _vs_default_spec(i: int, n: int, width: int, height: int) -> dict:
    """Evenly spaced regions, as a starting point for manual placement."""
    pad  = int(0.05 * width)
    span = max(1, (width - 2 * pad) // max(n, 1))
    x0   = pad + i * span
    return {"label": f"R{i + 1}",
            "x0": int(x0 + 0.10 * span), "x1": int(x0 + 0.90 * span),
            "y_top": int(0.10 * height), "y_bottom": int(0.90 * height),
            "line_y": int(0.45 * height)}


@st.cache_data(show_spinner=False)
def _vs_probe(path: str, stamp: float) -> dict:
    from modules.climbing_video import probe_video
    return probe_video(path).to_dict()


@st.cache_resource(show_spinner=False)
def _vs_background(path: str, stamp: float):
    """The temporal-median background. Cached as a resource: it is a large array,
    every later step needs it, and it depends only on the file."""
    from modules.climbing_video import build_background
    return build_background(path)


@st.cache_data(show_spinner=False)
def _vs_detect(path: str, stamp: float, n_vials: int) -> dict:
    from modules.climbing_video import detect_vials
    det = detect_vials(path, n_vials=int(n_vials),
                       background=_vs_background(path, stamp))
    # The activity map and background are megabyte-scale arrays the page never
    # draws, and the background is already cached in its own entry.
    return {k: v for k, v in det.items() if k not in ("activity", "background")}


@st.cache_data(show_spinner=False)
def _vs_tap(path: str, stamp: float) -> dict:
    from modules.climbing_video import detect_tap_time
    tap = detect_tap_time(path)
    return {k: v for k, v in tap.items() if k not in ("motion", "times")}


@st.cache_data(show_spinner=False)
def _vs_annotate(path: str, stamp: float, specs: tuple,
                 at_time_s: float, t0_s: float):
    """One annotated frame, as RGB. Keyed on the regions, so moving a region
    re-renders it but changing anything else does not."""
    import cv2 as _cv2
    from modules.climbing_video import annotate_frame
    vis = annotate_frame(path, _vs_specs_to_vials(specs),
                         _vs_background(path, stamp),
                         at_time_s=float(at_time_s), t0_s=float(t0_s))
    return _cv2.cvtColor(vis, _cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def _vs_analyze(path: str, stamp: float, specs: tuple, times: tuple,
                totals: tuple, t0_s: float,
                calib: tuple = (), rule: tuple = ()) -> dict:
    """The 60-90 second step. Keyed on every input that can change the numbers,
    so no widget interaction on the page can trigger it a second time."""
    from modules.climbing_video import analyze_climbing_video
    res = analyze_climbing_video(
        path, vials=_vs_specs_to_vials(specs),
        scoring_times_s=[float(t) for t in times],
        t0_s=float(t0_s),
        n_total_flies={str(k): int(v) for k, v in totals} or None,
        calibration=_vs_calibration(calib),
        scoring_rule=_vs_rule(rule),
    )
    res.pop("background", None)
    return res


def _vs_reset_downstream():
    """Forget everything derived from the previous recording."""
    for k in ("_vs_det", "_vs_auto", "_vs_specs", "_vs_result", "_vs_table",
              "_vs_preview_ok"):
        st.session_state.pop(k, None)
    for k in [k for k in list(st.session_state.keys())
              if isinstance(k, str) and k.startswith("vs_spec_")]:
        st.session_state.pop(k, None)
    st.session_state.pop("vs_t0", None)


def page_draw_vials():
    """
    Draw the cylinders, mark the line, score -- with a zoom for checking the counts.

    Automatic geometry is what failed hardest on real footage: the scoring line was
    found outside the tube in 4 of 15 recordings, once BELOW the base; boxes landed on
    the frame edges while the cylinders stood in the middle; three cylinders were
    found where two exist; and half the exported rows of that dataset came from
    recordings in that state. Drawing the geometry removes that whole class of error,
    which is the reason this page exists.
    """
    import tempfile

    import cv2
    from PIL import Image

    from modules import climbing_detectors as CD
    from modules import climbing_layout as LAY
    from modules import climbing_simple as CS
    from modules import climbing_video as CVm

    render_header("Draw Vials",
                  "Draw the cylinders, mark the line, and score -- with a zoom to "
                  "check every count")

    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("`streamlit-drawable-canvas` is not installed. Run "
                 "`pip install streamlit-drawable-canvas` and reload.")
        return

    # 1 - the recording
    st.subheader("1 - Recording")
    up = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv", "wmv"], key="dv_up")
    local = st.text_input("...or a path already on this machine", key="dv_path")
    path = None
    if up is not None:
        k = f"dv_tmp_{up.name}_{up.size}"
        if k not in st.session_state:
            t = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            t.write(up.read()); t.flush(); t.close()
            st.session_state[k] = t.name
        path = st.session_state[k]
    elif local.strip():
        path = local.strip() if Path(local.strip()).is_file() else None
        if path is None:
            st.error(f"No file at {local}")
    if not path:
        st.info("Upload a recording, or give the path to one.")
        return

    try:
        info = CVm.probe_video(path)
    except Exception as exc:
        st.error(f"Could not open that video: {exc}")
        return
    st.caption(f"{Path(path).name} - {info.width}x{info.height}, {info.fps:.2f} fps, "
               f"{info.duration_s:.1f} s")

    # 2 - the tap
    st.subheader("2 - When were they tapped down?")
    st.caption("One second of error here moves the 10-second score by more than its "
               "own value. The suggestion below is the moment the flies are "
               "physically lowest in the tube, which is measurable rather than eyeballed.")
    if st.button("Find the tap for me", key="dv_findtap"):
        with st.spinner("Looking for the moment the flies are lowest..."):
            st.session_state["dv_t0"] = _suggest_tap(path, CVm)
    t0 = st.slider("Tap (seconds into the file)", 0.0,
                   float(max(info.duration_s - 1, 0.1)),
                   float(st.session_state.get("dv_t0", 0.0)), 0.1, key="dv_t0slider")

    # 3 - draw
    st.subheader("3 - Draw a box round each cylinder")
    at_s = st.slider("Show the frame this many seconds after the tap",
                     0.0, 60.0, 10.0, 1.0, key="dv_at")
    frames = CVm._read_frames(path, [int((t0 + at_s) * info.fps)])
    if not frames:
        st.error("That frame could not be read.")
        return
    rgb = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2RGB)
    disp_w = 700
    disp_h = int(round(info.height * disp_w / max(info.width, 1)))
    bg_img = Image.fromarray(cv2.resize(rgb, (disp_w, disp_h)))

    st.caption("Drag a rectangle over each cylinder, covering its full height. They "
               "are numbered left to right as R1, R2 ... the same names the rest of "
               "DroPhenix uses.")
    canvas = st_canvas(fill_color="rgba(255,165,0,0.15)", stroke_width=2,
                       stroke_color="#FF6A00", background_image=bg_img,
                       update_streamlit=True, height=disp_h, width=disp_w,
                       drawing_mode="rect", key="dv_canvas")

    objs = (canvas.json_data or {}).get("objects", []) if canvas.json_data else []
    layout = LAY.layout_from_canvas(objs, disp_w, disp_h, info.width, info.height,
                                    int(disp_h * 0.45), Path(path).name, float(t0))
    if not layout.vials:
        st.info("Draw at least one box to continue.")
        return

    # 3b - the scoring line, PROPOSED per cylinder and CONFIRMED by the operator
    st.subheader("3b - The scoring line, one per cylinder")
    st.caption(
        "The software proposes a line and says how sure it is; you decide. This is "
        "what the published method does -- Cao et al. 2017 have the operator mark the "
        "vial baselines with a cursor, and nothing automatic here is better than that."
        "\n\nThe proposal is the darkest horizontal feature in the middle of the box "
        "that does **not** continue past the glass. That last condition is what "
        "matters: a mark drawn on the tube stops at the glass, while a bench mat, a "
        "shelf or a skirting board runs across the whole frame. Without it the rule "
        "put the line on the mat at the 10 mL mark, 400 px low, on 12 of 12 vials."
        "\n\n**Margin** is how much darker the winner is than the next candidate. "
        "Near 1.0 means a coin toss: those lines were wrong while looking confident, "
        "so anything under 1.3 must be confirmed by eye before it will score.")

    if st.button("Propose a line in each cylinder", key="dv_findlines"):
        with st.spinner("Building the background and measuring each cylinder..."):
            bgm = CVm.build_background(path)
            props = {}
            for v in layout.vials:
                box = CS.Box(x0=v.x0, y0=v.y0, x1=v.x1, y1=v.y1,
                             label=v.label, n_flies=max(v.n_flies, 1))
                fit = CS.find_line(bgm, box)
                props[v.label] = {"row": int(fit.row), "depth": float(fit.depth),
                                  "margin": float(fit.margin),
                                  "clear": bool(fit.is_clear()),
                                  "verdict": fit.verdict()}
            st.session_state["dv_lineprops"] = props
            st.session_state["dv_bg_key"] = Path(path).name

    props = st.session_state.get("dv_lineprops") or {}
    # a proposal belongs to one recording and one set of boxes; anything else is stale
    if props and (st.session_state.get("dv_bg_key") != Path(path).name
                  or set(props) != {v.label for v in layout.vials}):
        props = {}
        st.session_state.pop("dv_lineprops", None)
        st.info("The boxes or the recording changed, so the previous proposals were "
                "discarded. Press the button again.")
    scale = info.height / max(disp_h, 1)          # native rows per display row
    scale_x = info.width / max(disp_w, 1)         # native cols per display col
    rows_native, unconfirmed = {}, []
    if props:
        for v in layout.vials:
            pr = props.get(v.label)
            c1, c2 = st.columns([3, 2])
            with c1:
                default_disp = int(round((pr["row"] if pr else info.height * 0.45) / scale))
                default_disp = int(np.clip(default_disp, 0, disp_h))
                d = st.slider(f"{v.label} - line (pixels from the top of the picture)",
                              0, disp_h, default_disp, 1, key=f"dv_line_{v.label}")
                # The picture is shown at 700 px wide, so one slider step is
                # several native rows. Routing an untouched proposal through the
                # slider would round it away, so an unmoved slider keeps the measured
                # row exactly; moving it hands the choice to the operator.
                rows_native[v.label] = (int(pr["row"]) if (pr and d == default_disp)
                                        else int(round(d * scale)))
                st.caption(f"row {rows_native[v.label]} of {info.height} in the "
                           f"full-size frame"
                           + ("  (the measured proposal)" if pr and d == default_disp
                              else "  (moved by you)"))
            with c2:
                if pr is None:
                    st.warning("no proposal - set it yourself")
                    unconfirmed.append(v.label)
                elif pr["clear"]:
                    st.success(f"clear (depth {pr['depth']:.0f}, "
                               f"margin {pr['margin']:.2f})")
                else:
                    st.error(pr["verdict"])
                    if not st.checkbox(f"I have checked {v.label}'s line",
                                       key=f"dv_ok_{v.label}"):
                        unconfirmed.append(v.label)
    else:
        st.info("Press the button above, or set each line by hand below.")
        for v in layout.vials:
            d = st.slider(f"{v.label} - line (pixels from the top of the picture)",
                          0, disp_h, int(disp_h * 0.45), 1, key=f"dv_line_{v.label}")
            rows_native[v.label] = int(round(d * scale))
            st.caption(f"row {rows_native[v.label]} of {info.height} in the "
                       f"full-size frame")
        # Nothing has been measured and nothing has been confirmed: these sliders
        # start at 45% of frame height, which is a guess, not a line. Scoring on an
        # untouched guess is the exact failure this section exists to prevent, so it
        # takes one explicit statement that a person actually looked.
        if not st.checkbox("I have placed every line on the mark myself",
                           key="dv_manual_ok"):
            unconfirmed.append("all cylinders (no proposal, none confirmed)")

    # every line drawn where it will actually be used
    prev = np.array(bg_img).copy()
    for v in layout.vials:
        r_disp = int(round(rows_native.get(v.label, info.height * 0.45) / scale))
        x0d, x1d = int(v.x0 / scale_x), int(v.x1 / scale_x)
        pr = props.get(v.label)
        col = (0, 170, 0) if (pr and pr["clear"]) else (220, 0, 0)
        cv2.rectangle(prev, (x0d, int(v.y0 / scale)), (x1d, int(v.y1 / scale)),
                      (255, 140, 0), 1)  # y by scale, x by scale_x
        cv2.line(prev, (max(0, x0d - 12), r_disp), (min(disp_w - 1, x1d + 12), r_disp),
                 col, 2)
        cv2.putText(prev, v.label, (x0d, max(12, int(v.y0 / scale) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 1)
    st.image(prev, caption="Green = the software is confident; red = you must confirm it",
             width="stretch")

    # carry the confirmed rows into the layout (shared height + per-vial offset)
    LAY.set_vial_lines(layout, rows_native)

    if unconfirmed:
        why = ("nothing has been measured and nothing confirmed - those sliders are "
               "still at 45% of frame height, which is a starting position, not a line"
               if not props else
               "the proposal there is not distinguishable from another dark feature")
        st.warning(
            f"Not scoring yet: {', '.join(unconfirmed)} - {why}. Check the picture "
            f"above, then tick the box or move the line. A line placed 400 px wrong "
            f"still produces a confident-looking number, which is the failure this "
            f"stops.")

    # 4 - label
    st.subheader("4 - Label the cylinders")
    st.caption("The number of flies loaded is required. Without it the denominator is "
               "taken from the detector, and that changed across timepoints in 24 of "
               "24 vials the last time it was relied on - for sealed cylinders whose "
               "count cannot change.\n\n"
               "**The vial column is editable, and you should check it.** Boxes are "
               "numbered left to right as a starting point, but that is only a "
               "convenience: in the Day7 recordings the tubes are marked R1/R2/R3 in "
               "pen and their physical order changes between videos (R3,R2,R1 in one, "
               "R1,R2,R3 in the next). Type what is written on the glass.")
    ed = st.data_editor(
        pd.DataFrame([{"vial": v.label, "Genotype": "", "Treatment": "Untreated",
                       "Sex": "Mixed", "flies_loaded": 20}
                      for v in layout.vials]),
        hide_index=True, width="stretch", key="dv_labels")
    for v, (_, r) in zip(layout.vials, ed.iterrows()):
        # the label is what is written on the tube, not where it happens to stand
        lab = str(r.get("vial") or "").strip()
        if lab:
            v.label = lab
        v.genotype = str(r.get("Genotype") or v.label)
        v.treatment = str(r.get("Treatment") or "Untreated")
        v.sex = str(r.get("Sex") or "Mixed")
        v.n_flies = int(r.get("flies_loaded") or 0)
        # line_nudge_px is NOT edited here -- it carries the line confirmed in 3b

    probs = layout.problems()
    if probs:
        for msg in probs:
            st.error(msg)
        return
    st.success(f"{len(layout.vials)} cylinder(s) drawn and labelled.")
    st.download_button("Save this layout", layout.to_json(),
                       file_name=f"layout_{Path(path).stem}.json",
                       mime="application/json", key="dv_savelayout",
                       help="Re-use it on the other recordings of this session. The "
                            "software realigns it to each one and refuses if the "
                            "camera moved too far for the boxes to still fit.")

    # 5 - score
    st.subheader("5 - Score")
    tp_txt = st.text_input("Timepoints after the tap (seconds)",
                           "10, 20, 30, 40, 50, 60", key="dv_times")
    try:
        times = [float(x) for x in tp_txt.split(",") if x.strip()]
    except ValueError:
        st.error("Timepoints must be numbers separated by commas.")
        return
    backend = st.selectbox("Detection method", list(CD.BACKENDS),
                           format_func=lambda b: CD.BACKEND_LABELS[b],
                           index=list(CD.BACKENDS).index(CD.DEFAULT_BACKEND),
                           key="dv_backend")
    # The published rule is the NET of up- and down-crossings, which is simply how
    # many are standing above the line: "if a fly climbs back down or falls, record
    # that fly as -1 and count the next fly to cross as the same number". The
    # ratchet is a different question and is always at least as large. These labels
    # previously had it the other way round -- the ratchet was offered first, and
    # captioned as the published protocol, so it was what the page scored by
    # default while `analyze_climbing_video` defaulted to the protocol's rule.
    mode = st.radio("Scoring rule",
                    ["Flies STANDING above the line - up minus down (the published protocol)",
         "The MOST that were ever above the line at one moment (a ratchet)"], key="dv_mode")
    scoring = "net" if "STANDING" in mode else "max_reached"

    blocked = bool(unconfirmed)
    if blocked:
        st.error(f"Scoring is disabled until the line is confirmed in: "
                 f"{', '.join(unconfirmed)} (section 3b).")
    if st.button("Score these cylinders", type="primary", key="dv_go",
                 disabled=blocked):
        with st.spinner("Building the background model..."):
            bg = CVm.build_background(path)
        st.session_state["dv_bg_path"] = path
        out, diags = [], []
        bar = st.progress(0.0)
        for i, dv in enumerate(layout.vials, 1):
            bar.progress(i / len(layout.vials), text=f"Vial {i} of {len(layout.vials)}")
            spec = dv.to_spec(layout.line_y)
            probe = CD.probe_footage(path, spec, background=bg, n_samples=8)
            scored = CD.score_crossings(path, spec, bg, t0, times, backend=backend,
                                        probe=probe, mode=scoring)
            diag = CD.timepoint_diagnostics(path, spec, bg, t0, times,
                                            n_flies_loaded=dv.n_flies,
                                            backend=backend, probe=probe)
            diag.insert(0, "vial", dv.label)
            diags.append(diag)
            for row in scored:
                n = row["n_above"]
                out.append({"Genotype": dv.genotype, "Treatment": dv.treatment,
                            "Sex": dv.sex, "Vial_ID": dv.label, "Replicate": i,
                            "Time_sec": row["time_s"],
                            "Count_climbed": int(round(n)) if np.isfinite(n) else None,
                            "n_total_flies": dv.n_flies,
                            "PI": (round(n / max(dv.n_flies, 1), 4)
                                   if np.isfinite(n) else None),
                            "scoring_mode": scoring, "backend": backend,
                            "Video_file": Path(path).name})
        bar.empty()
        st.session_state["dv_results"] = pd.DataFrame(out)
        st.session_state["dv_diag"] = pd.concat(diags, ignore_index=True)
        st.session_state["dv_layout_json"] = layout.to_json()

    res = st.session_state.get("dv_results")
    diag = st.session_state.get("dv_diag")
    if res is None or res.empty:
        return

    st.subheader("6 - Results")
    st.dataframe(res, width="stretch")
    st.download_button("Download DroPhenix CSV", res.to_csv(index=False).encode(),
                       file_name=f"climbing_{Path(path).stem}.csv",
                       mime="text/csv", key="dv_dl")

    st.subheader("7 - What limits the count at each timepoint")
    st.markdown(
        "A single number per timepoint hides whether it should be believed. Every "
        "quantity below was measured at the moment of the count and needs no ground "
        "truth. **uncertain_flies** adds up the animals whose side of the line is "
        "genuinely in doubt - compare it with the **3-fly** spread you get from "
        "tapping the same vial five times, which is the floor a perfect counter hits.")
    cols = [c for c in ["vial", "time_s", "n_above", "n_total_detected",
                        "flies_loaded", "not_visible", "near_line", "frame_spread",
                        "merged_blobs", "on_static_mark", "uncertain_flies",
                        "confidence", "limiting_factor"] if c in diag.columns]
    st.dataframe(diag[cols], width="stretch")
    with st.expander("What each column means"):
        st.markdown("""
- **near_line** - detections within about two fly body-lengths of the line. These are
  the animals whose classification is decided by where the line sits to within a few
  pixels, so this is the count's sensitivity to the geometry.
- **frame_spread** - how much the count moved across the consecutive frames averaged
  at that instant. Flies barely move in 0.17 s, so this is the detector disagreeing
  with itself.
- **merged_blobs** - touching animals counted as one. Worst early, when the cohort is
  still piled at the base - which is exactly when the 10-second score is taken.
- **on_static_mark** - detections rejected for sitting on a printed graduation. A high
  number means the mask is doing its job, not that the count is fragile; it is
  reported but deliberately excluded from the limiting factor.
- **not_visible** - flies loaded minus objects found. Positive means animals the
  camera cannot see; negative means something that is not a fly is being counted.
""")

    st.subheader("8 - Zoom in and check the counting")
    st.caption("The counts are worth exactly what the pictures show. Pick a cylinder "
               "and a timepoint and look at it.")
    z1, z2, z3 = st.columns(3)
    zv = z1.selectbox("Cylinder", [v.label for v in layout.vials], key="dv_zv")
    zt = z2.selectbox("Timepoint (s)", times, key="dv_zt")
    zoom = z3.slider("Magnification", 1, 6, 3, 1, key="dv_zoom")
    above_only = st.checkbox("Show only the part above the line", key="dv_half")
    dv = next(v for v in layout.vials if v.label == zv)
    with st.spinner("Rendering..."):
        img = _zoom_panel(path, dv.to_spec(layout.line_y), t0, float(zt),
                          backend, int(zoom), bool(above_only), CVm, CD)
    if img is None:
        st.error("That frame could not be read.")
    else:
        st.image(img, width="stretch",
                 caption=f"{zv} at {zt:.0f} s after the tap - red = counted above the "
                         f"line, blue = below, green = the line")


def _suggest_tap(path, CVm) -> float:
    """The tap is the moment the flies are physically lowest, which is measurable."""
    import cv2
    info = CVm.probe_video(path)
    bg = CVm.build_background(path)
    det = CVm.detect_vials(path, n_vials=None, background=bg)
    vials = det.get("vials") or []
    if not vials:
        return 0.0
    p = CVm.DetectorParams(register=False)
    cap = cv2.VideoCapture(str(path))
    best_t, best_h = 0.0, float("inf")
    for t in np.arange(0, min(info.duration_s - 0.5, 40.0), 1.0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info.fps))
        ok, im = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g, _, _ = CVm.align_to_background(g, bg)
        hs = []
        for v in vials:
            r = CVm.count_in_vial(g, bg, v, p)
            hs += [(v.y_bottom - b["cy"]) / max(v.height, 1) for b in r["blobs"]]
        if hs and float(np.mean(hs)) < best_h:
            best_h, best_t = float(np.mean(hs)), float(t)
    cap.release()
    return best_t


def _zoom_panel(path, spec, t0, t, backend, zoom, above_only, CVm, CD):
    """One vial at one instant, magnified, with every detection marked."""
    import cv2

    from modules import climbing_video_analyzer as CVA
    info = CVm.probe_video(path)
    fr = CVm._read_frames(path, [int((t0 + t) * info.fps)])
    if not fr:
        return None
    bgr = cv2.cvtColor(fr[0], cv2.COLOR_GRAY2BGR)
    bg = CVm.build_background(path)
    cfg = CD._vialconfig_for(spec)
    cfg.line_y_frac = float(np.clip((spec.line_y - spec.y_top)
                                    / max(spec.height, 1), 0.0, 1.0))
    cents, _ = CVA.detect_flies_in_roi(bgr, cfg, CVA.DEFAULT_DETECTION_PARAMS)
    if backend == "threshold_masked":
        m = CD.static_mask(bg, spec)
        cents = [(x, y) for (x, y) in cents
                 if not m[int(np.clip(y, 0, m.shape[0] - 1)),
                          int(np.clip(x, 0, m.shape[1] - 1))]]
    bot = spec.line_y if above_only else spec.y_bottom
    crop = bgr[spec.y_top:bot, spec.x0:spec.x1].copy()
    if crop.size == 0:
        return None
    ly = spec.line_y - spec.y_top
    if 0 <= ly < crop.shape[0]:
        cv2.line(crop, (0, ly), (crop.shape[1] - 1, ly), (0, 200, 0), 1)
    for (x, y) in cents:
        if 0 <= int(y) < crop.shape[0]:
            colour = (0, 0, 255) if int(y) < ly else (255, 0, 0)
            cv2.circle(crop, (int(x), int(y)), 6, colour, 2)
    z = max(1, int(zoom))
    crop = cv2.resize(crop, (crop.shape[1] * z, crop.shape[0] * z),
                      interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def page_climbing_video():
    """
    Climbing-assay video analysis with user-defined regions.

    Flow: upload -> experiment details -> mark T=0 -> define ROI and reference
    line per vial -> calibration preview -> full run -> results and export.

    This page states the geometry rather than inferring it, which is the point:
    it is the path for footage the automatic detector in "Video Scoring" cannot
    handle. Nothing is guessed on the user's behalf, and the preview in step 5
    exists so that no analysis is run before the detections have been looked at.
    """
    import tempfile

    import cv2

    from modules.climbing_video import VialSpec
    from modules.climbing_video_analyzer import (
        VideoAnalysisSession, VialConfig, load_video_metadata,
        get_frame_at_number, detect_flies_in_roi, results_to_drophenix_csv,
        try_detect_reference_line, run_full_analysis, DISCREPANCY_LIMIT,
    )

    def _spec(cfg):
        """The same region, expressed the way the detector registry wants it."""
        return VialSpec(x0=int(cfg.roi_x), x1=int(cfg.roi_x + cfg.roi_w),
                        y_top=int(cfg.roi_y), y_bottom=int(cfg.roi_y + cfg.roi_h),
                        line_y=int(cfg.roi_y + cfg.line_y_frac * cfg.roi_h),
                        label=f"R{cfg.vial_id}", source="manual")

    render_header("Climbing Video Analyser",
                  "Count flies above a reference line from a recorded assay video")

    with st.expander("Known limitations — read before trusting a number"):
        st.markdown(f"""
- **Clustered flies.** Immediately after the tap the flies are piled at the base and
  touching animals appear as one blob. Detection improves as they disperse, typically
  after 5–10 s.
- **Printed graduations.** On a graduated cylinder the printed ticks and numerals are
  dark and roughly fly-sized. Adaptive thresholding removes a lighting gradient but not
  a locally dark mark, so this method can **over**-count on graduated glassware. Any
  frame where detected and expected differ by more than {DISCREPANCY_LIMIT:.0%} is
  flagged in both directions — check the preview before trusting the run.
- **Dark background with dark flies.** Switch the fly-appearance setting.
- **Reflective glass.** If glare produces false positives, raise the minimum fly size.
- **Low frame rate (< 15 fps).** Fast animals can be missed; the confidence score
  flags those timepoints.
- **ROI coverage.** The rectangle must include the full vial height, including the
  region above the reference line — flies outside it are simply not counted.
""")

    # ── STEP 1: video ────────────────────────────────────────────────────────
    st.subheader("Step 1 — Upload video")
    video_file = st.file_uploader(
        "Climbing assay video", type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Any common format. Phone, webcam or DSLR footage all work.")
    local_path = st.text_input(
        "…or the path to a video already on this machine",
        value="", placeholder=r"D:\DroPhenix\Testing data\...\video.mp4",
        help="Avoids re-uploading gigabytes when the file is already local.")

    video_path = None
    if video_file is not None:
        key = f"cva_tmp_{video_file.name}_{video_file.size}"
        if key not in st.session_state:
            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=Path(video_file.name).suffix)
            tmp.write(video_file.read())
            tmp.flush()
            tmp.close()
            st.session_state[key] = tmp.name
        video_path = st.session_state[key]
    elif local_path.strip():
        if Path(local_path.strip()).is_file():
            video_path = local_path.strip()
        else:
            st.error(f"No file at {local_path}")

    if not video_path:
        st.info("Upload a video, or give the path to one already on this machine.")
        return

    try:
        meta = load_video_metadata(video_path)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    fps = meta["fps"]
    if not fps or fps <= 0:
        st.error("This file reports no frame rate, so timepoints cannot be located. "
                 "Re-encode it (for example with ffmpeg) and try again.")
        return

    st.success(f"{Path(video_path).name} — {meta['total_frames']} frames, "
               f"{meta['duration_sec']:.1f} s, {fps:.2f} fps, "
               f"{meta['width']}x{meta['height']} px")

    # ── STEP 2: experiment details ───────────────────────────────────────────
    st.subheader("Step 2 — Experiment details")
    c1, c2 = st.columns(2)
    with c1:
        n_vials = st.number_input("Vials/cylinders visible in this video",
                                  1, 12, 1, 1, key="cva_nv")
        n_flies = st.number_input("Flies per vial (expected total)",
                                  1, 200, 20, 1, key="cva_nf")
        n_taps = int(st.number_input(
            "Taps in this recording", 1, 20, 1, 1, key="cva_ntaps",
            help="Most negative-geotaxis protocols tap the same vial three to "
                 "five times with a minute's recovery between them and score "
                 "after each. Each tap is a technical replicate of the SAME "
                 "flies and has its own T=0."))
        fly_col = st.selectbox("Fly appearance",
                               ["Dark flies, light background (typical)",
                                "Light flies, dark background"], key="cva_col")
        fly_color_key = "dark" if "Dark" in fly_col else "light"
    with c2:
        geno_txt = st.text_area(
            "Genotype per vial (one per line)",
            value="\n".join(f"Genotype_{i+1}" for i in range(int(n_vials))),
            height=120, key="cva_geno")
        rep_start = st.number_input("Starting replicate number", 1, 100, 1, 1,
                                    key="cva_rep")
        treatment = st.text_input("Treatment", value="vehicle", key="cva_treat")
        sex = st.selectbox("Sex", ["Mixed", "Male", "Female"], key="cva_sex")

    genos = [g.strip() for g in geno_txt.strip().split("\n") if g.strip()]
    if len(genos) != int(n_vials):
        st.warning(f"{len(genos)} genotype(s) given for {int(n_vials)} vial(s); "
                   "the list is padded so the run can proceed — fix it before "
                   "exporting.")
        genos = (genos + [f"Genotype_{i+1}" for i in range(int(n_vials))])[:int(n_vials)]

    tp_txt = st.text_input("Timepoints, seconds after tap-down (comma-separated)",
                           value="10, 20, 30, 60", key="cva_tp")
    try:
        timepoints = [float(t.strip()) for t in tp_txt.split(",") if t.strip()]
    except ValueError:
        st.error("Timepoints must be numbers separated by commas.")
        return
    if not timepoints:
        st.error("Give at least one timepoint.")
        return

    # ── STEP 3: T=0 ──────────────────────────────────────────────────────────
    st.subheader("Step 3 — Mark the tap-down moment (T=0) for each tap")
    # THIS USED TO BE A SINGLE SLIDER DEFAULTING TO 0.0, AND THAT WAS TWO FAULTS.
    # First, `climbing_video.detect_taps` already finds EVERY tap in a recording
    # and returns the moment each disturbance ENDS -- which is when the flies are
    # down and free to climb, and therefore that trial's zero -- but this page
    # never called it, so the operator was left to eyeball a 0.5 s slider for a
    # quantity the software could measure. Second, only ONE T=0 existed, so a
    # five-tap recording was scored as a single trial and the other four taps were
    # read as part of one long observation after tap 1.
    if st.button("Find the taps in this recording", key="cva_findtaps"):
        with st.spinner("Looking for whole-frame motion bursts…"):
            import modules.climbing_video as _CVt
            st.session_state["cva_taps_" + Path(video_path).name] = (
                _CVt.detect_taps(video_path))
    tapinfo = st.session_state.get("cva_taps_" + Path(video_path).name)

    found: list = []
    if tapinfo:
        found = [float(t["t0_s"]) for t in (tapinfo.get("taps") or [])]
        if found:
            st.success(f"Found {len(found)} tap(s) at "
                       + ", ".join(f"{v:.1f} s" for v in found) + ".")
        else:
            st.warning("No tap was detected. That is not proof there is none — a "
                       "gentle tap, or one made before recording started, leaves "
                       "no motion burst. Enter the times yourself below.")
        for w in ([] if tapinfo.get("framing_stable", True) else
                  ["The camera appears to have MOVED during this recording. The "
                   "background model assumes fixed framing, so counts either side "
                   "of the move are not comparable — trim the file first."]):
            st.error(w)
        if found and len(found) != n_taps:
            st.warning(
                f"You declared **{n_taps}** tap(s); the detector found "
                f"**{len(found)}**. Neither is automatically right — a rack "
                f"knocked in passing looks like a tap, and a gentle one does not. "
                f"Reconcile them below before scoring; the values you leave in the "
                f"boxes are what gets used.")
    else:
        st.caption("Press the button to measure the taps, or enter the times "
                   "yourself below.")

    tap_zeros_sec: list = []
    tcols = st.columns(min(n_taps, 4))
    for ti in range(n_taps):
        # detected value as the default where there is one, otherwise spaced by a
        # nominal minute -- never silently 0.0 for every tap, which would score the
        # same instant n times and look like perfect agreement between replicates
        default = (found[ti] if ti < len(found)
                   else min(float(ti) * 60.0, float(meta["duration_sec"])))
        with tcols[ti % len(tcols)]:
            tap_zeros_sec.append(float(st.number_input(
                f"Tap {ti+1} — T=0 (s)", 0.0, float(max(meta["duration_sec"], 0.1)),
                float(default), 0.1, format="%.2f", key=f"cva_t0_{ti}")))
    if any(b <= a for a, b in zip(tap_zeros_sec, tap_zeros_sec[1:])):
        st.error("Tap times must increase. Two taps at the same instant would score "
                 "the same frames twice and report them as agreeing replicates.")
        return

    tap_zero_frames = [int(v * fps) for v in tap_zeros_sec]
    t_zero_sec, t_zero_frame = tap_zeros_sec[0], tap_zero_frames[0]
    preview = get_frame_at_number(video_path, t_zero_frame)
    if preview is None:
        st.error("That frame could not be read — adjust the tap times.")
        return
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
             caption=f"Tap 1, T=0 (frame {t_zero_frame})", width="stretch")
    st.caption("Scoring times are measured from each tap's own T=0. A one-second "
               "error moves an early score substantially, so check each one. The "
               "frame above is tap 1, and it is also the frame the regions and the "
               "reference line below are set against.")
    last_end = tap_zeros_sec[-1] + (max(timepoints) if timepoints else 0.0)
    if last_end > meta["duration_sec"]:
        st.warning(
            f"The last tap plus your longest timepoint reaches "
            f"{last_end:.0f} s, but the recording is only "
            f"{meta['duration_sec']:.0f} s long. Those readings will be reported "
            f"as not scoreable rather than guessed.")

    # ── STEP 4: ROI and reference line ───────────────────────────────────────
    st.subheader("Step 4 — Vial regions and reference line")
    st.markdown(
        "Give each vial's rectangle in **pixels of the full frame** (X and Y of the "
        "top-left corner, then width and height). The reference line is set as a "
        "fraction of the vial's height, 0.0 at the top and 1.0 at the bottom.")

    vials: list = []
    for i in range(int(n_vials)):
        with st.expander(f"Vial {i+1} — {genos[i]}", expanded=(i == 0)):
            a, b, c, d = st.columns(4)
            rx = a.number_input("X start", 0, int(meta["width"]),
                                int(i * meta["width"] / int(n_vials)), key=f"cva_x{i}")
            ry = b.number_input("Y start", 0, int(meta["height"]), 0, key=f"cva_y{i}")
            rw = c.number_input("Width", 10, int(meta["width"]),
                                int(meta["width"] / int(n_vials)), key=f"cva_w{i}")
            rh = d.number_input("Height", 10, int(meta["height"]),
                                int(meta["height"]), key=f"cva_h{i}")
            auto = try_detect_reference_line(preview, int(rx), int(ry), int(rw), int(rh))
            line_frac = st.slider("Reference line (0 = top, 1 = bottom)", 0.0, 1.0,
                                  float(auto) if auto else 0.5, 0.01,
                                  format="%.2f", key=f"cva_line{i}")
            if auto:
                st.caption(f"A strong horizontal edge was found at {auto:.2f} — "
                           "check it is the drawn line and not a graduation or a "
                           "shelf edge.")
            else:
                st.caption("No single strong horizontal edge stood out, so nothing was "
                           "guessed. Set the line yourself.")
            vials.append(VialConfig(
                vial_id=i + 1, genotype=genos[i], replicate=int(rep_start) + i,
                n_flies_expected=int(n_flies), roi_x=int(rx), roi_y=int(ry),
                roi_w=int(rw), roi_h=int(rh), line_y_frac=float(line_frac),
                fly_color=fly_color_key, sex=sex, treatment=treatment))

    # ── STEP 5: calibration ──────────────────────────────────────────────────
    st.subheader("Step 5 — Calibration preview")
    from modules import fly_blob as _FB
    with st.expander("Detection parameters"):
        pa, pb = st.columns(2)
        blur_k = pa.slider("Blur strength (odd)", 1, 15, 5, 2, key="cva_blur")
        adapt_c = pb.slider("Threshold sensitivity", 1, 15, 4, 1, key="cva_c")
        # THESE TWO SLIDERS WERE DEAD UNTIL 2026-08-26, AND THE BUG WAS THE UNITS.
        # They were labelled "% of ROI" and written into `min_area_frac` /
        # `max_area_frac`, which `detect_flies_in_roi` reads ONLY when
        # `size_basis == "roi_area"`. The shipped default is "width", so both
        # values were silently discarded. Measured on synthetic footage with 20
        # flies drawn: an impossible window (min = max = 0.5%) returned 19
        # detections -- identical to a wide-open window -- while the same window
        # with `size_basis="roi_area"` correctly returned 0. Meanwhile the app
        # told users whose count was off by >30% to "adjust the size limits",
        # advice that could not work.
        #
        # The units are now the ones the detector actually uses: blob area as a
        # fraction of the VIAL WIDTH squared, which does not change when the user
        # draws a taller box. The "% of ROI area" basis is deliberately NOT
        # restored -- `climbing_video_analyzer` line 372 records why it was
        # abandoned (extending a box upward moved the area floor from 34 to
        # 44 px^2 and the count from 29 detections to 22, against a hand count
        # of 19).
        #
        # The defaults below reproduce the module constants EXACTLY: 0.16% and
        # 8.1% of w^2 are min_area_w2 = 0.0016 and max_area_w2 = 0.081, so making
        # the controls live does not move any existing result.
        min_area_pct = pa.slider("Min fly area (% of vial width squared)",
                                 0.02, 2.0, 0.16, 0.02, key="cva_min_w2")
        max_area_pct = pb.slider("Max fly area (% of vial width squared)",
                                 2.0, 25.0, 8.1, 0.1, key="cva_max_w2")
        _d_lo = 100.0 * (4.0 * (min_area_pct / 100.0) / np.pi) ** 0.5
        _d_hi = 100.0 * (4.0 * (max_area_pct / 100.0) / np.pi) ** 0.5
        st.caption(
            f"A fly spans roughly a tenth of a cylinder's width, so its area is "
            f"about 1% of the width squared; the window brackets that. As a "
            f"diameter, this window accepts a blob between {_d_lo:.1f}% and "
            f"{_d_hi:.1f}% of the vial's width across. "
            f"These limits govern the threshold-based methods below. The default "
            f"`blob` method is size-matched and uses a FITTED fly width of "
            f"{_FB.PARAMS['fly_d_frac']:.2f} of the vial width, which is not "
            f"adjustable here: that value is sharply peaked (0.05 -> MAE 9.11, "
            f"0.07 -> 2.93, 0.10 -> 9.82 flies), so a slider on it would be a "
            f"slider on the error.")
    params = {"blur_kernel": int(blur_k), "adaptive_block": 21,
              "adaptive_c": int(adapt_c),
              # name the basis explicitly rather than leaning on the default, so
              # that a later change of default cannot silently orphan these again
              "size_basis": "width",
              "min_area_w2": min_area_pct / 100.0,
              "max_area_w2": max_area_pct / 100.0,
              "morph_open": 2, "morph_close": 2,
              "confidence_threshold": 0.70}

    # ── detection method: every option exposed, with the evidence for each ───
    from modules import climbing_detectors as CD

    st.markdown("**Detection method**")
    choice = st.radio(
        "How should flies be found inside each region?",
        ["Automatic — measure the footage and choose"] +
        [CD.BACKEND_LABELS[b] for b in CD.BACKENDS],
        key="cva_backend", label_visibility="collapsed")
    manual_backend = next((b for b in CD.BACKENDS
                           if CD.BACKEND_LABELS[b] == choice), None)

    probe_vial = None
    if vials:
        v0 = vials[0]
        probe_vial = _spec(vials[0])

    backend = manual_backend
    if manual_backend is None and probe_vial is not None:
        if st.button("Measure this recording and choose", key="cva_probe"):
            with st.spinner("Measuring static clutter and motion…"):
                import modules.climbing_video as _CV
                bgm = _CV.build_background(video_path)
                pr = CD.probe_footage(video_path, probe_vial, background=bgm)
                st.session_state["cva_probe"] = (pr, CD.recommend_backend(
                    pr, n_expected=int(n_flies)))
        got = st.session_state.get("cva_probe")
        if got:
            pr, rec = got
            m1, m2, m3 = st.columns(3)
            m1.metric("Static marks in the empty scene", pr.static_clutter,
                      help="Fly-sized dark features present with no flies in shot — "
                           "graduations, numerals, label edges. An intensity "
                           "threshold counts these as flies, roughly one for one.")
            m2.metric("Objects that move", f"{pr.moving_fraction:.0%}",
                      help="Background subtraction can only see what moves. Near "
                           "zero means it has nothing to subtract.")
            m3.metric("Camera held still", "yes" if pr.framing_stable else "NO")
            if rec["backend"] is None:
                st.error(f"**Refused.** {rec['reason']}")
            else:
                backend = rec["backend"]
                (st.success if rec["confident"] else st.info)(
                    f"**Using `{backend}`.** {rec['reason']}")
            for w in rec["warnings"]:
                st.warning(w)
        else:
            st.caption("Press the button to measure the recording, or pick a method "
                       "above yourself.")
    if backend is None:
        backend = CD.DEFAULT_BACKEND
    st.caption(
        f"Measured against laboratory hand counts on 24 paired measurements: "
        f"adaptive threshold MAE {CD.MEASURED_MAE['threshold']:.1f} flies, median "
        f"background {CD.MEASURED_MAE['background']:.1f}, declutter "
        f"{CD.MEASURED_MAE['threshold_declutter']:.1f}. For scale, five repeat taps "
        f"of the SAME vial by a human differ by about "
        f"{CD.BIOLOGICAL_FLOOR_FLIES} flies — so none of these methods is yet at the "
        f"accuracy of the assay itself, on that laboratory's glassware.")

    # ── scoring rule: what "flies above the line at t" is taken to mean ──────
    st.markdown("**Scoring rule**")
    score_mode = st.radio(
        "How is the count at each timepoint defined?",
        ["Flies STANDING above the line - up minus down (the published protocol)",
         "The MOST that were ever above the line at one moment (a ratchet)"],
        key="cva_scoremode", label_visibility="collapsed")
    scoring_mode = "net" if "STANDING" in score_mode else "max_reached"
    st.caption(
        "The standard negative-geotaxis protocols say: *if a fly climbs back down "
        "or falls, record that fly as -1 and count the next fly to cross the line "
        "as the same number.* The tally therefore steps back when a fly falls, so "
        "the net of up- and down-crossings is simply **how many are standing above "
        "the line** — the first option, and the default."
        "\n\nThe second option is a ratchet: the most that were ever above at one "
        "moment. It never decreases, so it is always at least as large. That is a "
        "legitimate question but it is **not** what the protocol scores, and it was "
        "previously offered here first and captioned as though it were.")

    prev_t = st.slider("Preview at this many seconds after T=0", 0.0,
                       float(max(meta["duration_sec"] - t_zero_sec, 0.1)),
                       float(min(10.0, max(meta["duration_sec"] - t_zero_sec, 0.1))),
                       0.5, key="cva_prevt")
    prev_frame = get_frame_at_number(video_path, t_zero_frame + int(prev_t * fps))
    if prev_frame is None:
        prev_frame = preview

    _bg_cache = None
    if backend != "threshold":
        import modules.climbing_video as _CV
        _bg_cache = st.session_state.get("cva_bg_" + Path(video_path).name)
        if _bg_cache is None:
            with st.spinner("Building the background model…"):
                _bg_cache = _CV.build_background(video_path)
            st.session_state["cva_bg_" + Path(video_path).name] = _bg_cache

    cols = st.columns(min(int(n_vials), 4))
    for i, vc in enumerate(vials):
        cents, dbg = detect_flies_in_roi(prev_frame, vc, params)
        ly = vc.line_y_for_height(dbg.shape[0])
        if backend == "threshold":
            above = sum(1 for (_, cy) in cents if cy < ly)
            total = len(cents)
        else:
            pr_obj = (st.session_state.get("cva_probe") or (None, None))[0]
            r = CD.count_with_backend(backend, prev_frame, _bg_cache, _spec(vc),
                                      probe=pr_obj, params=params)
            above, total = r["n_above"], r["n_total"]
        conf = min(total / max(vc.n_flies_expected, 1), 1.0)
        disc = (total - vc.n_flies_expected) / max(vc.n_flies_expected, 1)
        with cols[i % len(cols)]:
            st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
                     caption=f"Vial {vc.vial_id}: {total} detected "
                             f"(expected {vc.n_flies_expected}) · above {above}",
                     width="stretch")
            if abs(disc) > DISCREPANCY_LIMIT:
                st.error(f"{disc:+.0%} vs expected — "
                         f"{'over' if disc > 0 else 'under'}-detecting. "
                         "Adjust the size limits or the ROI.")
            elif conf < 0.80:
                st.info(f"Detection {conf:.0%} of expected.")
            else:
                st.success(f"Detection {conf:.0%} of expected.")

    # ── STEP 6: run ──────────────────────────────────────────────────────────
    st.subheader("Step 6 — Run")
    if st.button("Run climbing assay analysis", type="primary", key="cva_run"):
        session = VideoAnalysisSession(
            video_path=video_path, fps=fps, total_frames=meta["total_frames"],
            duration_sec=meta["duration_sec"], t_zero_frame=t_zero_frame,
            tap_zero_frames=tap_zero_frames,
            timepoints_sec=timepoints, vials=vials, detection_params=params)
        bar = st.progress(0.0, text="Analysing…")
        collected = []
        from modules.climbing_video_analyzer import FrameResult
        import modules.climbing_video as _CVm
        if _bg_cache is None:
            with st.spinner("Building the background model…"):
                _bg_cache = _CVm.build_background(video_path)
        pr_obj = (st.session_state.get("cva_probe") or (None, None))[0]
        # EVERY TAP IS SCORED, EACH FROM ITS OWN T=0.
        # Previously this loop ran once, from a single T=0, so a five-tap
        # recording yielded one trial and the remaining four taps were treated as
        # part of one long observation after tap 1 -- which also means the flies
        # were rising, falling, being re-tapped and rising again inside what was
        # reported as a single climbing curve.
        jobs_total = max(len(tap_zeros_sec) * len(vials), 1)
        job_i = 0
        for tap_i, tz_sec in enumerate(tap_zeros_sec, 1):
          for vi, vc in enumerate(vials, 1):
            job_i += 1
            bar.progress(job_i / jobs_total,
                         text=f"Tap {tap_i} of {len(tap_zeros_sec)}, "
                              f"vial {vi} of {len(vials)}…")
            spec = _spec(vc)
            try:
                scored = CD.score_crossings(
                    video_path, spec, _bg_cache, float(tz_sec),
                    timepoints, backend=backend, probe=pr_obj,
                    mode=scoring_mode, params=params)
            except Exception as exc:
                st.error(f"Vial {vc.vial_id}, tap {tap_i} could not be scored: "
                         f"{exc}")
                continue
            n_exp = max(vc.n_flies_expected, 1)
            for row in scored:
                above = row["n_above"]
                # this tap's own zero, not tap 1's
                fno = int(tz_sec * fps) + int(row["time_s"] * fps)
                if not np.isfinite(above):
                    collected.append(FrameResult(
                        vial_id=vc.vial_id, time_sec=float(row["time_s"]),
                        frame_number=fno, flies_above=0, flies_below=0,
                        flies_detected=0, n_flies_expected=vc.n_flies_expected,
                        pi=0.0, confidence=0.0, low_confidence=True,
                        count_discrepancy=-1.0, flagged=True, tap=tap_i,
                        reason=row.get("note") or "not scoreable"))
                    continue
                above = int(round(above))
                # THE DETECTED TOTAL IS A DIFFERENT QUANTITY FROM THE COUNT ABOVE
                # THE LINE, AND THIS BLOCK USED TO CONFLATE THEM.
                # `flies_detected` was set to `above`, which had three consequences,
                # all of them scientific rather than cosmetic:
                #   1. the exported `Detected_total` equalled `Count_climbed`, so
                #      coverage could not be audited after the fact at all;
                #   2. `Confidence` became PI capped at 1, so a TDP-43 vial with a
                #      genuine PI of 0.5 exported as `Low_confidence = True` -- a
                #      real biological effect reported as a data-quality fault, and
                #      the direction that matters most, since a sick cohort is
                #      exactly the one whose low PI is the result;
                #   3. flagging reduced to `above > n_expected`, so a vial finding
                #      59 objects with 15 above the line and N = 20 passed silently.
                #      Over-detection on graduated glassware is THE failure mode of
                #      the threshold backends (measured: 2.9x on real footage), and
                #      it was the one direction the flag could not see.
                # `score_crossings` now returns `n_total`, so all three are honest.
                total_raw = row.get("n_total", float("nan"))
                if not np.isfinite(total_raw):
                    # never silently substitute `above` here -- that is precisely
                    # the defect above. PI is still valid; coverage is not known.
                    collected.append(FrameResult(
                        vial_id=vc.vial_id, time_sec=float(row["time_s"]),
                        frame_number=fno, flies_above=above, flies_below=0,
                        flies_detected=0, n_flies_expected=vc.n_flies_expected,
                        pi=round(above / n_exp, 4), confidence=0.0,
                        low_confidence=True, count_discrepancy=-1.0, flagged=True,
                        tap=tap_i,
                        reason=(f"backend={backend}, scoring={scoring_mode}; "
                                "detected total unavailable, so coverage was not "
                                "checked")))
                    continue
                total = int(round(total_raw))
                disc = (total - vc.n_flies_expected) / n_exp
                note = f"backend={backend}, scoring={scoring_mode}"
                if above > total:
                    # only reachable under the ratchet -- see `score_crossings`
                    note += ("; count above the line exceeds the concurrent total "
                             "because the ratchet retains flies that crossed and "
                             "then fell back")
                collected.append(FrameResult(
                    vial_id=vc.vial_id, time_sec=float(row["time_s"]),
                    frame_number=fno, flies_above=above,
                    # what was SEEN below the line, not N minus what was seen above:
                    # the old form asserted that every fly not counted above was
                    # below, which assumes the detection is complete
                    flies_below=max(0, total - above),
                    flies_detected=total,
                    n_flies_expected=vc.n_flies_expected,
                    pi=round(above / n_exp, 4),
                    # coverage: how much of the loaded cohort the detector accounts
                    # for. Independent of how many of them climbed.
                    confidence=round(min(total / n_exp, 1.0), 3),
                    low_confidence=bool(total / n_exp < 0.70),
                    count_discrepancy=round(disc, 3),
                    flagged=bool(abs(disc) > DISCREPANCY_LIMIT),
                    tap=tap_i,
                    reason=note))
        bar.empty()
        session.results = collected
        bar.empty()
        st.session_state["cva_results"] = results_to_drophenix_csv(session)
        st.success(f"{len(collected)} measurements: {len(tap_zeros_sec)} tap(s) x "
                   f"{len(vials)} vial(s) x {len(timepoints)} timepoint(s).")

    # ── STEP 7: results ──────────────────────────────────────────────────────
    df = st.session_state.get("cva_results")
    if df is None or df.empty:
        return
    st.subheader("Step 7 — Results")

    # TAP IS PART OF THE INDEX, NOT SOMETHING TO AVERAGE OVER SILENTLY.
    # With `index=["Genotype", "Replicate"]` alone, a five-tap recording collapsed
    # into one row per vial and `aggfunc="mean"` quietly averaged the taps -- which
    # is a defensible summary but not one anybody asked for, and it hid the
    # tap-to-tap spread that IS the assay's own noise floor (about 3 flies).
    _idx = ["Genotype", "Replicate"] + (["Tap"] if df["Tap"].nunique() > 1 else [])
    st.dataframe(df.pivot_table(index=_idx, columns="Time_sec",
                                values="PI", aggfunc="mean").reset_index(),
                 width="stretch")
    if df["Tap"].nunique() > 1:
        st.caption(
            "One row per tap: repeat taps of the same vial are technical "
            "replicates of the SAME flies, so they are not independent samples "
            "and must not be pooled as though they were. Their spread is the "
            "assay's own noise floor — about three flies per vial on this "
            "protocol — and it is the yardstick any detector error should be "
            "judged against.")

    flagged = df[df["Flagged"] == True]        # noqa: E712
    if not flagged.empty:
        st.warning(
            f"{len(flagged)} of {len(df)} measurements differ from the expected fly "
            f"count by more than {DISCREPANCY_LIMIT:.0%}. They are exported and "
            "labelled, not dropped — but they should not be analysed without "
            "looking at them.")
        with st.expander("Show flagged measurements"):
            st.dataframe(flagged[["Genotype", "Replicate", "Time_sec",
                                  "Detected_total", "n_total_flies",
                                  "Count_discrepancy", "Confidence",
                                  "Analysis_notes"]], width="stretch")

    clean = df[df["Flagged"] == False]         # noqa: E712
    if not clean.empty:
        try:
            import plotly.express as px
            fig = px.line(clean, x="Time_sec", y="PI", color="Genotype",
                          line_dash=clean["Replicate"].astype(str), markers=True,
                          title="Performance index over time (unflagged only)",
                          labels={"Time_sec": "Time (s)", "PI": "PI = climbed / total"})
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.line_chart(clean.pivot_table(index="Time_sec", columns="Genotype",
                                            values="PI", aggfunc="mean"))
    else:
        st.info("Every measurement was flagged, so no curve is drawn.")

    st.download_button("Download DroPhenix CSV",
                       df.to_csv(index=False).encode(),
                       file_name=f"climbing_video_{Path(video_path).stem}.csv",
                       mime="text/csv", key="cva_dl")
    st.caption("Column names are aliases DroPhenix already recognises "
               "(`Time_sec`→Time, `Count_climbed`→Count, `n_total_flies`→n), so this "
               "file loads through **CSV Upload** without editing.")


def page_video_scoring():
    render_header("Video Scoring",
                  "Score a climbing assay directly from a phone recording")
    try:
        from modules import climbing_video as cvid
    except Exception as _e:
        st.error(f"Climbing-video module import failed: {_e}")
        return

    st.markdown(
        "Turns a recording of graduated cylinders into the canonical DroPhenix "
        "table — flies **above the line** and flies **present**, per cylinder, at "
        "each scoring time. The scale markings on a cylinder are static and the "
        "flies move, so a temporal-median background reconstructs each tube "
        "complete with its printed scale and *without* its animals; subtracting "
        "that leaves only the flies. Counting flies above the line is the easy "
        "half, so the **denominator is taken from you** and the detector's own "
        "total is reported beside it as a cross-check rather than a replacement.")

    t_in, t_reg, t_score, t_batch, t_exp = st.tabs(
        ["Video", "Regions and Lines", "Scoring", "Batch", "Export"])

    # ── 1. input ──────────────────────────────────────────────────────────────
    with t_in:
        st.markdown("#### Source recording")
        c_up, c_path = st.columns(2)
        with c_up:
            up = st.file_uploader("Upload a recording",
                                  type=["mp4", "mov", "avi", "m4v"], key="vs_upload")
        with c_path:
            path_in = st.text_input(
                "...or the path to the file on this machine", key="vs_path",
                help="Laboratory clips run to 100-150 MB each. Pointing at the "
                     "file on disk avoids copying it through the browser.")

        resolved, src_note = "", ""
        p_txt = (path_in or "").strip().strip('"').strip("'")
        if p_txt:
            if Path(p_txt).is_file():
                resolved, src_note = str(Path(p_txt)), "path on disk"
            else:
                st.error(f"No file at '{p_txt}'.")
        elif up is not None:
            key = (up.name, int(up.size))
            if st.session_state.get("_vs_upload_key") != key:
                with st.spinner(f"Copying {up.name} "
                                f"({up.size / 1e6:.0f} MB) to a temporary file..."):
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(up.name).suffix, delete=False
                    ) as tmp:
                        tmp.write(up.getbuffer())
                        st.session_state["_vs_upload_path"] = tmp.name
                st.session_state["_vs_upload_key"] = key
            resolved = st.session_state.get("_vs_upload_path", "")
            src_note = f"uploaded file {up.name}"

        if resolved and resolved != st.session_state.get("_vs_video"):
            _vs_reset_downstream()
            st.session_state["_vs_video"] = resolved
        elif not resolved and p_txt == "" and up is None:
            st.session_state.pop("_vs_video", None)

        video = st.session_state.get("_vs_video") or ""
        if not video:
            st.info(
                "Load a recording to begin — upload one above, or give the path "
                "to a file on this machine.\n\n"
                "The pipeline expects a clip that looks like this:\n\n"
                "- the graduated cylinders standing side by side against a pale, "
                "plain background\n"
                "- a visible line drawn on each cylinder at the scoring height\n"
                "- the flies tapped to the bottom at the start, with the "
                "operator's hands then out of shot\n"
                "- the camera held still for the whole observation (a phone "
                "propped against something solid is enough — the analysis "
                "corrects a pixel or two of settling, not a moving camera)\n"
                "- the full height of every cylinder in frame, filmed square-on "
                "rather than from above")
        else:
            try:
                info = _vs_probe(video, _vs_stamp(video))
            except Exception as _e:
                st.error(f"Could not read this file: {_e}")
                st.session_state.pop("_vs_video", None)
                video = ""
            else:
                st.success(f"Loaded from {src_note}: {Path(video).name}")
                m = st.columns(5)
                m[0].metric("Resolution", f"{info['width']} x {info['height']}")
                m[1].metric("Frame rate", f"{info['fps']:.2f} fps")
                m[2].metric("Duration", f"{info['duration_s']:.1f} s")
                m[3].metric("Frames", str(info["n_frames"]))
                m[4].metric("Codec", info["codec"] or "unknown")
                st.caption(
                    "If the duration or the frame count reads as zero, OpenCV "
                    "decoded the container but not the stream — iPhone HEVC "
                    "(H.265) is the usual cause. Re-export as H.264 MP4.")

    video = st.session_state.get("_vs_video") or ""
    stamp = _vs_stamp(video) if video else 0.0
    info = {}
    if video:
        try:
            info = _vs_probe(video, stamp)
        except Exception:
            video, info = "", {}

    def _vs_needs_video():
        st.info("Load a recording on the **Video** tab first.")

    W = int(info.get("width", 1920) or 1920)
    H = int(info.get("height", 1080) or 1080)
    DUR = float(info.get("duration_s", 0.0) or 0.0)

    # ── 2. regions and lines ──────────────────────────────────────────────────
    with t_reg:
        if not video:
            _vs_needs_video()
        else:
            st.markdown("#### Automatic detection")
            st.caption(
                "The cylinders are located from where flies have been seen to "
                "move, and the scoring line from the darkest sustained row "
                "inside each tube. On a set of 15 laboratory recordings this "
                "found all three cylinders in 11. Correcting the other 4 by hand "
                "below is the documented fallback, not an edge case.")

            cd1, cd2 = st.columns([1, 3])
            n_vials = cd1.number_input("Cylinders in frame", 1, 12, 3, 1,
                                       key="vs_nvials")
            run_det = cd2.button("Detect cylinders and scoring lines",
                                 type="primary", key="vs_det_run")

            if run_det:
                with st.spinner("Building the temporal-median background and "
                                "locating the cylinders. This reads the whole "
                                "clip and takes a minute or so the first time; "
                                "it is cached afterwards."):
                    try:
                        det = _vs_detect(video, stamp, int(n_vials))
                        tap = _vs_tap(video, stamp)
                    except Exception as _e:
                        det, tap = None, None
                        st.error(f"Detection failed: {_e}")
                if det is not None:
                    st.session_state["_vs_det"] = det
                    auto = [v.to_dict() for v in det.get("vials", [])]
                    st.session_state["_vs_auto"] = auto
                    for i, a in enumerate(auto):
                        st.session_state[f"vs_spec_label_{i}"] = str(a["label"])
                        for f in _VS_SPEC_FIELDS:
                            st.session_state[f"vs_spec_{f}_{i}"] = int(a[f])
                    if tap and tap.get("available"):
                        st.session_state["vs_t0"] = float(
                            np.clip(tap.get("suggested_t0_s", 0.0), 0.0,
                                    max(DUR - 1.0, 0.0)))
                    st.session_state.pop("_vs_result", None)

            det = st.session_state.get("_vs_det")
            if det is not None:
                conf = float(det.get("confidence", 0.0))
                dm = st.columns(4)
                dm[0].metric("Detection confidence", f"{conf:.2f}")
                _wcv = det.get("width_cv", np.nan)
                _gcv = det.get("gap_cv", np.nan)
                dm[1].metric("Width consistency (CV)",
                             f"{_wcv:.3f}" if pd.notna(_wcv) else "N/A")
                dm[2].metric("Spacing regularity (CV)",
                             f"{_gcv:.3f}" if pd.notna(_gcv) else "N/A")
                dm[3].metric("Line strength", f"{float(det.get('line_score', 0.0)):.2f}")
                if det.get("needs_review"):
                    st.warning(
                        "This detection needs review before the counts can be "
                        "used. " + (det.get("reason") or "Confidence is below "
                        "the 0.70 threshold.") + " A wrong region produces a "
                        "confidently wrong count, which is worse than no count — "
                        "check the annotated frame and correct the numbers below.")
                else:
                    st.success("All cylinders and scoring lines located with "
                               "high confidence. Still confirm on the annotated "
                               "frame before scoring.")
                if not det.get("vials"):
                    st.info("No regions were returned, so the fields below start "
                            "from an evenly spaced guess. Set them by hand.")

            st.divider()
            st.markdown("#### Regions and scoring lines")
            st.caption(
                "`line_y` is an absolute image row and image y increases "
                "downwards, so rows **above** it (a smaller number) count as "
                "climbed. Editing any field marks that cylinder as manually "
                "specified and redraws the frame below.")

            auto = st.session_state.get("_vs_auto", [])
            if st.button("Reset all regions to the auto-detected values",
                         key="vs_reset_auto", disabled=not auto):
                for i, a in enumerate(auto):
                    st.session_state[f"vs_spec_label_{i}"] = str(a["label"])
                    for f in _VS_SPEC_FIELDS:
                        st.session_state[f"vs_spec_{f}_{i}"] = int(a[f])
                st.session_state.pop("_vs_result", None)

            specs = []
            for i in range(int(n_vials)):
                base = dict(auto[i]) if i < len(auto) else _vs_default_spec(
                    i, int(n_vials), W, H)
                st.session_state.setdefault(f"vs_spec_label_{i}", str(base["label"]))
                for f in _VS_SPEC_FIELDS:
                    st.session_state.setdefault(f"vs_spec_{f}_{i}",
                                                int(np.clip(base[f], 0, max(W, H))))

                st.markdown(f"**Cylinder {i + 1}**")
                cc = st.columns([1.2, 1, 1, 1, 1, 1])
                lab = cc[0].text_input("Label", key=f"vs_spec_label_{i}")
                x0v = cc[1].number_input("x0", 0, W, step=1, key=f"vs_spec_x0_{i}")
                x1v = cc[2].number_input("x1", 0, W, step=1, key=f"vs_spec_x1_{i}")
                ytv = cc[3].number_input("y_top", 0, H, step=1,
                                         key=f"vs_spec_y_top_{i}")
                ybv = cc[4].number_input("y_bottom", 0, H, step=1,
                                         key=f"vs_spec_y_bottom_{i}")
                lyv = cc[5].number_input("line_y", 0, H, step=1,
                                         key=f"vs_spec_line_y_{i}")

                cur = {"label": lab or f"R{i + 1}", "x0": int(x0v), "x1": int(x1v),
                       "y_top": int(ytv), "y_bottom": int(ybv), "line_y": int(lyv)}
                same_as_auto = (
                    i < len(auto) and
                    all(int(auto[i][f]) == cur[f] for f in _VS_SPEC_FIELDS) and
                    str(auto[i]["label"]) == cur["label"])
                cur["source"] = "auto" if same_as_auto else "manual"
                specs.append(cur)

                if cur["x1"] <= cur["x0"] or cur["y_bottom"] <= cur["y_top"]:
                    st.error(f"Cylinder {i + 1}: the region is empty "
                             "(x1 must exceed x0 and y_bottom must exceed y_top).")
                elif not (cur["y_top"] < cur["line_y"] < cur["y_bottom"]):
                    st.warning(f"Cylinder {i + 1}: the scoring line at row "
                               f"{cur['line_y']} lies outside the tube "
                               f"[{cur['y_top']}, {cur['y_bottom']}]. Every fly "
                               "would fall on one side of it.")

            st.session_state["_vs_specs"] = specs
            if any(s["source"] == "manual" for s in specs):
                st.caption("Cylinders marked manually: " + ", ".join(
                    s["label"] for s in specs if s["source"] == "manual"))

            st.divider()
            st.markdown("#### Annotated frame")
            st.caption(
                "Blue rectangle = the region, orange line = the scoring line, "
                "green circle = a detection above the line, red = below. This "
                "one picture confirms or rejects the whole analysis in a way no "
                "summary statistic can.")
            valid = [s for s in specs
                     if s["x1"] > s["x0"] and s["y_bottom"] > s["y_top"]]
            t0_prev = float(st.session_state.get("vs_t0", 0.0) or 0.0)
            prev_max = max(float(DUR - t0_prev - 1.0), 1.0)
            prev_t = st.slider("Preview at t (seconds after the tap)",
                               0.0, float(round(prev_max, 1)),
                               float(min(10.0, round(prev_max, 1))), 1.0,
                               key="vs_prev_t")
            # The first frame drawn has to build the temporal-median background,
            # which reads the whole clip. That is not something to trigger merely
            # because a video was loaded, so it is asked for explicitly and every
            # later redraw comes from the cache.
            if st.session_state.get("_vs_det") is not None:
                st.session_state["_vs_preview_ok"] = True
            if st.button("Draw the annotated frame", key="vs_draw"):
                st.session_state["_vs_preview_ok"] = True
            if not valid:
                st.info("Define at least one valid region to draw the frame.")
            elif not st.session_state.get("_vs_preview_ok"):
                st.info("Press **Draw the annotated frame** to render the "
                        "regions on the video. The first render builds the "
                        "temporal-median background by reading the whole clip "
                        "(about a minute on a two-minute 1080p recording) and is "
                        "cached from then on, so moving a region afterwards "
                        "redraws immediately.")
            else:
                with st.spinner("Rendering the annotated frame..."):
                    try:
                        rgb = _vs_annotate(video, stamp,
                                           tuple(_vs_spec_tuple(s) for s in valid),
                                           float(prev_t), t0_prev)
                    except Exception as _e:
                        rgb = None
                        st.error(f"Could not draw the frame: {_e}")
                if rgb is not None:
                    st.image(rgb, width="stretch",
                             caption=f"{Path(video).name} at t0 + {prev_t:g} s")

    # ── 3. scoring ────────────────────────────────────────────────────────────
    with t_score:
        if not video:
            _vs_needs_video()
        else:
            specs = st.session_state.get("_vs_specs", [])
            valid = [s for s in specs
                     if s["x1"] > s["x0"] and s["y_bottom"] > s["y_top"]]
            if not valid:
                st.info("Set the cylinder regions on the **Regions and Lines** "
                        "tab first.")
            else:
                st.markdown("#### Flies loaded per cylinder")
                st.markdown(
                    "The performance index needs a numerator and a denominator, "
                    "and they are not equally hard to measure. Flies above the "
                    "line are few and spread out in the clean upper tube; **all** "
                    "the flies include a pile at the base, overlapping, against a "
                    "bright reflective foot. You loaded the vial and you know how "
                    "many went in, so that number is used as the denominator and "
                    "the detector's own total is reported beside it as a check.")
                tc = st.columns(max(len(valid), 1))
                totals_pairs = []
                blank_labels = []
                for i, s in enumerate(valid):
                    n_in = tc[i].number_input(
                        f"{s['label']} — flies loaded", min_value=1, max_value=500,
                        value=None, step=1, key=f"vs_ntot_{i}",
                        placeholder="leave blank")
                    if n_in is None:
                        blank_labels.append(s["label"])
                    else:
                        totals_pairs.append((s["label"], int(n_in)))
                if blank_labels:
                    st.caption(
                        "Left blank for " + ", ".join(blank_labels) +
                        ". The detected total will be used as the denominator "
                        "for those cylinders, so the proportion climbed inherits "
                        "every counting error in the denominator as well as the "
                        "numerator — read the stability check below before "
                        "trusting it.")

                st.divider()
                st.markdown("#### Physical calibration")
                calib = _vs_calibration_controls("vs")

                st.divider()
                st.markdown("#### Scoring rule")
                rule = _vs_rule_controls("vs", calibrated=bool(calib))
                st.caption("Scoring will use: **"
                           + _vs_rule(rule).describe() + "**.")

                st.divider()
                st.markdown("#### Scoring times")
                extra_txt = st.text_input(
                    "Additional scoring times (seconds after the tap, comma "
                    "separated)", key="vs_extra_t",
                    help="Entered times join the list below; select them there "
                         "to include them.")
                extra = []
                for tok in (extra_txt or "").replace(";", ",").split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        extra.append(float(tok))
                    except ValueError:
                        st.warning(f"Ignoring '{tok}': not a number of seconds.")
                opts = sorted(set(_VS_DEFAULT_TIMES) | set(extra))
                times = st.multiselect(
                    "Score at", opts,
                    default=[t for t in _VS_DEFAULT_TIMES if t in opts],
                    format_func=lambda t: f"{t:g} s", key="vs_times")
                st.caption("A single video yields the whole climbing time-course, "
                           "so there is no reason to keep only one endpoint.")

                st.session_state.setdefault("vs_t0", 0.0)
                ct1, ct2 = st.columns([1, 2])
                with ct1:
                    t0 = st.number_input(
                        "Tap time t0 (seconds from the start of the clip)",
                        min_value=0.0, max_value=float(max(DUR - 1.0, 0.1)),
                        step=0.5, key="vs_t0",
                        help="Scoring times are measured from the END of the tap, "
                             "when the flies are at the bottom and free to climb.")
                with ct2:
                    if st.button("Estimate the tap time from the clip",
                                 key="vs_tap_run"):
                        with st.spinner("Looking for the burst of whole-frame "
                                        "motion that marks the tap..."):
                            try:
                                tap = _vs_tap(video, stamp)
                            except Exception as _e:
                                tap = None
                                st.error(f"Tap detection failed: {_e}")
                        if tap and tap.get("available"):
                            st.session_state["vs_t0"] = float(np.clip(
                                tap["suggested_t0_s"], 0.0, max(DUR - 1.0, 0.1)))
                            st.success(
                                f"Motion burst from {tap['burst_start_s']:.1f} s "
                                f"to {tap['burst_end_s']:.1f} s; t0 set to "
                                f"{st.session_state['vs_t0']:.1f} s.")
                        elif tap is not None:
                            st.warning(
                                "No tap could be located ("
                                f"{tap.get('reason', 'no motion burst')}). Set t0 "
                                "by hand from the annotated frame.")

                st.divider()
                run_score = st.button("Score the recording", type="primary",
                                      key="vs_score_run", disabled=not times)
                if not times:
                    st.info("Choose at least one scoring time.")
                if run_score:
                    with st.spinner(
                        "Scoring. Every scoring time is counted over several "
                        "consecutive frames, and the detected total is then "
                        "re-measured across the whole clip for the stability "
                        "check. Typically 60-90 seconds for a two-minute 1080p "
                        "recording; the result is cached, so changing anything "
                        "on the Export tab will not repeat it."
                    ):
                        try:
                            res = _vs_analyze(
                                video, stamp,
                                tuple(_vs_spec_tuple(s) for s in valid),
                                tuple(float(t) for t in times),
                                tuple(totals_pairs), float(t0),
                                tuple(calib), tuple(rule))
                        except Exception as _e:
                            res = None
                            st.error(f"Scoring failed: {_e}")
                    if res is not None:
                        st.session_state["_vs_result"] = res
                        st.session_state.pop("_vs_table", None)

                res = st.session_state.get("_vs_result")
                if res is None:
                    st.info("No scoring run yet for this recording.")
                elif not res.get("ok"):
                    st.error(res.get("reason", "The recording could not be scored."))
                else:
                    for w in res.get("warnings", []) or []:
                        st.warning(w)

                    counts = res.get("counts", pd.DataFrame())
                    qc = res.get("qc", pd.DataFrame())

                    _rule = res.get("scoring_rule")
                    _cal = res.get("calibration")
                    _ppm = []
                    if _cal is not None:
                        _ppm = [p for p in (_cal.resolve(v)
                                            for v in res.get("vials", []))
                                if p]
                    rc = st.columns(3)
                    rc[0].metric("Scoring rule",
                                 _rule.describe() if _rule else
                                 "above the marked line")
                    rc[1].metric("Calibration",
                                 f"{float(np.mean(_ppm)):.2f} px/mm" if _ppm
                                 else "none")
                    rc[2].metric("Height reported in",
                                 "mm and fraction" if _ppm else "fraction only")

                    st.markdown("#### Counts")
                    if counts.empty:
                        st.warning("No countable scoring times fell inside the clip.")
                    else:
                        st.dataframe(counts, width="stretch")
                        st.download_button(
                            "Download counts CSV",
                            counts.to_csv(index=False).encode(),
                            "video_scoring_counts.csv", "text/csv",
                            key="dl_vs_counts")
                        st.caption(
                            "`n_total_used` is the denominator actually applied: "
                            "your supplied total where you gave one, the detected "
                            "total otherwise. `n_merged_blobs` counts detections "
                            "that were split by area because two or more flies "
                            "were touching.")
                        if _ppm:
                            st.caption(
                                "`mean_height_mm` is the mean climbing height "
                                "of every detected fly, in millimetres above "
                                "the base of the tube. It is the "
                                "lab-independent number in this table: it does "
                                "not depend on where anyone drew a line, and it "
                                "uses every animal rather than collapsing the "
                                "cohort to a proportion.")
                        else:
                            st.caption(
                                "`mean_height_mm` is empty because no "
                                "calibration was given. Supply the cylinder "
                                "diameter above and climbing height is also "
                                "reported in millimetres, which is comparable "
                                "between laboratories and zoom levels in a way "
                                "that a count above a hand-drawn line is not.")

                        usable = counts[counts.get("usable", True) == True]  # noqa: E712
                        if not usable.empty:
                            st.markdown("#### Climbing curve")
                            import plotly.graph_objects as go
                            palette = ["#0D2137", "#F0A500", "#2A7DE1", "#C0392B",
                                       "#27AE60", "#8E44AD", "#16A085", "#D35400"]
                            fig = go.Figure()
                            for i, (lab, sub) in enumerate(
                                    usable.groupby("vial", sort=True)):
                                sub = sub.sort_values("time_s")
                                fig.add_trace(go.Scatter(
                                    x=sub["time_s"], y=sub["proportion_climbed"],
                                    mode="lines+markers", name=str(lab),
                                    line=dict(color=palette[i % len(palette)],
                                              width=2),
                                    marker=dict(size=8)))
                            has_mm = bool(
                                _ppm and "mean_height_mm" in usable.columns
                                and usable["mean_height_mm"].notna().any())
                            if has_mm:
                                for i, (lab, sub) in enumerate(
                                        usable.groupby("vial", sort=True)):
                                    sub = sub.sort_values("time_s")
                                    fig.add_trace(go.Scatter(
                                        x=sub["time_s"],
                                        y=sub["mean_height_mm"],
                                        mode="lines+markers",
                                        name=f"{lab} (mm)", yaxis="y2",
                                        line=dict(
                                            color=palette[i % len(palette)],
                                            width=2, dash="dot"),
                                        marker=dict(size=6, symbol="diamond")))
                            fig.update_layout(
                                xaxis_title="Seconds after the tap",
                                yaxis_title=(
                                    "Proportion "
                                    + (_rule.describe() if _rule
                                       else "above the marked line")),
                                yaxis=dict(range=[0, 1.05]), height=420,
                                legend=dict(orientation="h", yanchor="bottom",
                                            y=1.02),
                                margin=dict(l=50, r=(70 if has_mm else 20),
                                            t=30, b=45))
                            if has_mm:
                                fig.update_layout(yaxis2=dict(
                                    title="Mean climbing height (mm)",
                                    overlaying="y", side="right",
                                    showgrid=False, rangemode="tozero"))
                            st.plotly_chart(fig, width="stretch")
                            st.caption("Each cylinder is one experimental unit, so "
                                       "these are replicates rather than repeats "
                                       "of one measurement.")
                            if has_mm:
                                st.caption(
                                    "Solid lines (left axis) are the proportion "
                                    "that met the scoring rule; dotted lines "
                                    "(right axis) are the mean climbing height "
                                    "in millimetres. The proportion depends on "
                                    "where the criterion was set and saturates "
                                    "once every fly has crossed it; the height "
                                    "in mm does neither, and is the quantity "
                                    "another laboratory can compare against.")

                    st.markdown("#### Stability check")
                    if qc.empty:
                        st.info("The clip was too short for the stability check.")
                    else:
                        def _vs_qc_row(row):
                            bad = not bool(row.get("stable", True))
                            return ["background-color: #FCE4E2" if bad else ""
                                    for _ in row]
                        st.dataframe(qc.style.apply(_vs_qc_row, axis=1),
                                     width="stretch")
                        n_bad = int((~qc["stable"].astype(bool)).sum())
                        if n_bad:
                            st.error(
                                f"{n_bad} of {len(qc)} cylinders are flagged "
                                "unstable (highlighted). The number of flies in a "
                                "sealed cylinder is physically constant, so all "
                                "variation in the detected total across the clip "
                                "is detector error. Supply the true totals and "
                                "check the regions before using these counts.")
                        else:
                            st.success("Every cylinder's detected total held "
                                       "steady across the clip (CV at or below "
                                       "15%), which is what a reliable detector "
                                       "looks like on a sealed tube.")
                        st.download_button("Download QC CSV",
                                           qc.to_csv(index=False).encode(),
                                           "video_scoring_qc.csv", "text/csv",
                                           key="dl_vs_qc")

                    if not counts.empty:
                        st.markdown("#### Annotated frames at the scoring times")
                        st.caption("Confirm by eye that the green detections are "
                                   "the flies you would have counted yourself.")
                        ok_times = sorted(
                            float(t) for t in
                            counts[counts.get("usable", True) == True]  # noqa: E712
                            ["time_s"].unique())
                        if ok_times:
                            pick = [ok_times[0]]
                            if len(ok_times) > 2:
                                pick.append(ok_times[len(ok_times) // 2])
                            if len(ok_times) > 1:
                                pick.append(ok_times[-1])
                            cols = st.columns(len(pick))
                            for c, t in zip(cols, pick):
                                try:
                                    rgb = _vs_annotate(
                                        video, stamp,
                                        tuple(_vs_spec_tuple(s) for s in valid),
                                        float(t), float(res.get("t0_s", 0.0)))
                                except Exception as _e:
                                    c.error(f"t = {t:g} s: {_e}")
                                else:
                                    c.image(rgb, width="stretch",
                                            caption=f"t0 + {t:g} s")

    # ── 4. batch ──────────────────────────────────────────────────────────────
    with t_batch:
        st.markdown("#### A folder of recordings, scored in one run")
        st.markdown(
            "Instead of opening fifteen videos one at a time, re-entering the "
            "genotype and treatment for each and merging fifteen CSVs by hand, "
            "you write the metadata **once** in a manifest and the pipeline "
            "does the rest. That matters beyond convenience — hand-merging is "
            "where mislabelling happens, and a mislabelled treatment column is "
            "not recoverable from the data afterwards. Every video is scored "
            "independently, because background, regions, tap time and zoom all "
            "change between recordings; what is shared is the metadata and the "
            "scoring rule.")

        b_dir = st.text_input(
            "Folder containing the recordings", key="vsb_dir",
            help="Every video named in the manifest is looked up inside this "
                 "folder unless the manifest gives an absolute path.")
        b_folder = (b_dir or "").strip().strip('"').strip("'")
        found = []
        if b_folder:
            p_dir = Path(b_folder)
            if not p_dir.is_dir():
                st.error(f"No folder at '{b_folder}'.")
                b_folder = ""
            else:
                found = sorted(f.name for f in p_dir.iterdir()
                               if f.is_file()
                               and f.suffix.lower() in _VS_VIDEO_EXT)
                if found:
                    st.success(f"{len(found)} recording(s) in this folder.")
                else:
                    st.warning("No video files found in that folder.")

        st.divider()
        st.markdown("#### Manifest")
        st.caption(
            "Required columns: `video`, `Genotype`, `Treatment`, `Sex`. "
            "Optional: `Age`, `TimeOfDay`, `ExperimentID`, `Batch`, "
            "`Compound`, `Dose`, and either `n_per_vial` (the same number of "
            "flies in every cylinder) or `n_R1`, `n_R2`, ... for a named "
            "cylinder.")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            man_up = st.file_uploader("Upload a manifest CSV", type=["csv"],
                                      key="vsb_manifest_up")
        with mc2:
            if st.button("Build a starter manifest from this folder",
                         key="vsb_manifest_build", disabled=not found):
                # Age and n_per_vial are float columns of NaN rather than
                # empty strings: `analyze_batch` calls float() on whatever is
                # not null, and an empty string would abort the whole run.
                st.session_state["_vsb_manifest"] = pd.DataFrame({
                    "video": found,
                    "Genotype": "", "Treatment": "", "Sex": "",
                    "Age": np.full(len(found), np.nan),
                    "TimeOfDay": "", "ExperimentID": "",
                    "n_per_vial": np.full(len(found), np.nan),
                })
                _vsb_new_manifest()
            st.caption("One row per video, with Genotype, Treatment and Sex "
                       "left blank for you to fill in.")
        with mc3:
            _template_button("video_manifest_template.csv",
                             "Download the manifest template",
                             key="vsb_tpl")

        if man_up is not None:
            key = (man_up.name, int(man_up.size))
            if st.session_state.get("_vsb_manifest_key") != key:
                try:
                    st.session_state["_vsb_manifest"] = pd.read_csv(man_up)
                except Exception as _e:
                    st.error(f"Could not read that manifest: {_e}")
                else:
                    st.session_state["_vsb_manifest_key"] = key
                    _vsb_new_manifest()

        man = st.session_state.get("_vsb_manifest")
        if man is None:
            st.info("Upload a manifest, or build a starter one from the folder "
                    "above and fill it in.")
            man_run = pd.DataFrame()
        else:
            st.caption("Edit the manifest here before running. Rows can be "
                       "added and removed.")
            # The editor is re-keyed whenever a different manifest is loaded:
            # a keyed data_editor stores the user's edits as a delta against
            # the frame it was given, and replaying that delta onto a new
            # manifest with different rows silently corrupts it.
            man_run = st.data_editor(
                man, width="stretch", num_rows="dynamic",
                key=f"vsb_editor_{st.session_state.get('_vsb_manifest_n', 0)}")
            st.download_button(
                "Download this manifest as CSV",
                man_run.to_csv(index=False).encode(),
                "video_manifest.csv", "text/csv", key="vsb_dl_manifest")

        st.divider()
        st.markdown("#### Batch settings")
        bs1, bs2 = st.columns(2)
        with bs1:
            b_auto = st.checkbox(
                "Detect the number of cylinders automatically", value=True,
                key="vsb_auto_n")
            b_n = st.number_input("Cylinders in frame", 1, 12, 3, 1,
                                  key="vsb_nvials", disabled=b_auto)
        with bs2:
            b_minconf = st.slider(
                "Minimum region confidence to keep a recording", 0.0, 1.0,
                0.70, 0.05, key="vsb_minconf",
                help="A recording below this is reported in the excluded "
                     "table with its reason rather than quietly contributing "
                     "suspect counts.")
        if b_auto:
            st.warning(
                "Automatic cylinder counting is on. It is the right default "
                "when you do not know the rig, but it decides **per video**, "
                "and a miscount is not a small error: an invented cylinder "
                "contributes fabricated replicates to a clustered analysis, "
                "where extra replicates directly inflate confidence, and a "
                "missed one throws a real replicate away. If every recording "
                "in this folder is of the same rig, set the number "
                "explicitly. Either way the run reports a note whenever the "
                "videos disagree with each other — read it before using the "
                "merged table.")

        b_times = st.multiselect(
            "Score at", _VS_DEFAULT_TIMES,
            default=list(_VS_DEFAULT_TIMES),
            format_func=lambda t: f"{t:g} s", key="vsb_times")

        st.markdown("**Physical calibration**")
        b_calib = _vs_calibration_controls("vsb")
        st.markdown("**Scoring rule**")
        b_rule = _vs_rule_controls("vsb", calibrated=bool(b_calib))
        st.caption("Scoring will use: **" + _vs_rule(b_rule).describe() + "**.")

        st.divider()
        n_req = int(len(man_run)) if man_run is not None else 0
        # Roughly 80-110 seconds per two-minute 1080p recording, depending on
        # how many scoring times are asked for and what else the machine is
        # doing: a batch of 15 clips takes something like 20-27 minutes. The
        # figure is deliberately quoted before the run rather than discovered
        # during it, because a user who does not know it will assume the page
        # has hung and reload, losing the work.
        est_min = max(1, int(round(n_req * 90.0 / 60.0)))
        if n_req:
            st.warning(
                f"This will read every frame of {n_req} recording(s) and takes "
                f"about **{est_min} minute(s)**. Nothing on this page can be "
                "changed while it runs. The result is cached against the "
                "folder, the manifest and these settings, so browsing the "
                "tables afterwards never repeats it — but editing any of them "
                "means running it again.")

        ready = bool(b_folder) and n_req > 0 and bool(b_times)
        sig = (b_folder, (man_run.to_csv(index=False) if n_req else ""),
               tuple(float(t) for t in b_times), tuple(b_calib), tuple(b_rule),
               (None if b_auto else int(b_n)), float(b_minconf))
        run_batch = st.button("Score the whole folder", type="primary",
                              key="vsb_run", disabled=not ready)
        if not b_folder:
            st.info("Give the folder holding the recordings.")
        elif not n_req:
            st.info("Load a manifest with at least one row.")
        elif not b_times:
            st.info("Choose at least one scoring time.")

        if run_batch:
            prog = st.progress(0.0, text="Starting the batch...")

            def _vsb_progress(frac, label):
                try:
                    prog.progress(float(np.clip(frac, 0.0, 1.0)),
                                  text=f"Scoring {label}")
                except Exception:
                    pass

            with st.spinner(f"Scoring {n_req} recording(s). About "
                            f"{est_min} minute(s)."):
                try:
                    bres = cvid.analyze_batch(
                        man_run, video_dir=b_folder,
                        scoring_times_s=[float(t) for t in b_times],
                        calibration=_vs_calibration(b_calib),
                        scoring_rule=_vs_rule(b_rule),
                        n_vials=(None if b_auto else int(b_n)),
                        min_confidence=float(b_minconf),
                        progress=_vsb_progress)
                except Exception as _e:
                    bres = None
                    st.error(f"Batch scoring failed: {_e}")
            prog.empty()
            if bres is not None:
                st.session_state["_vsb_result"] = bres
                st.session_state["_vsb_sig"] = sig

        bres = st.session_state.get("_vsb_result")
        if bres is None:
            st.info("No batch has been run yet.")
        else:
            if st.session_state.get("_vsb_sig") != sig:
                st.warning(
                    "The folder, the manifest or the settings have changed "
                    "since this result was produced. The tables below are the "
                    "previous run; press **Score the whole folder** to redo it "
                    "with the current settings.")

            for note in bres.get("notes", []) or []:
                low = note.lower()
                if ("excluded" in low or "could not be processed" in low
                        or "not consistent" in low):
                    st.warning(note)
                else:
                    st.info(note)

            bm = st.columns(4)
            bm[0].metric("Recordings requested", str(bres.get("n_requested", 0)))
            bm[1].metric("Scored", str(bres.get("n_scored", 0)))
            bm[2].metric("Excluded", str(len(bres.get("excluded",
                                                      pd.DataFrame()))))
            bm[3].metric("Failed", str(len(bres.get("failed", pd.DataFrame()))))

            merged = bres.get("table", pd.DataFrame())
            bqc = bres.get("qc", pd.DataFrame())
            bexc = bres.get("excluded", pd.DataFrame())
            bfail = bres.get("failed", pd.DataFrame())

            st.markdown("#### Merged table")
            if merged is None or merged.empty:
                st.warning("No recording produced usable rows.")
            else:
                st.dataframe(merged, width="stretch")
                st.caption(
                    "Each cylinder of each video is its own `VialID`, so a "
                    "folder of five three-cylinder videos gives fifteen "
                    "independent replicates for the clustered statistics — not "
                    "five.")
                st.download_button(
                    "Download merged CSV",
                    merged.to_csv(index=False).encode(),
                    "video_batch_climbing.csv", "text/csv",
                    key="dl_vsb_merged")

            st.markdown("#### Per-video quality control")
            if bqc is None or bqc.empty:
                st.info("No quality-control rows were produced.")
            else:
                st.dataframe(bqc, width="stretch")
                st.caption(
                    "The number of flies in a sealed cylinder is physically "
                    "constant, so all variation in `detected_total_cv_pct` is "
                    "detector error. This table is per cylinder per video and "
                    "includes recordings that were excluded, so nothing is "
                    "hidden.")
                st.download_button("Download batch QC CSV",
                                   bqc.to_csv(index=False).encode(),
                                   "video_batch_qc.csv", "text/csv",
                                   key="dl_vsb_qc")

            st.markdown("#### Excluded on quality grounds")
            if bexc is None or bexc.empty:
                st.success("No recording was excluded.")
            else:
                st.dataframe(bexc, width="stretch")
                st.caption("These are reported rather than silently dropped. "
                           "Scoring one by hand on the other tabs, after "
                           "correcting its regions, is a deliberate act.")

            st.markdown("#### Could not be processed")
            if bfail is None or bfail.empty:
                st.success("Every recording named in the manifest was read.")
            else:
                st.dataframe(bfail, width="stretch")

            if merged is not None and not merged.empty:
                st.divider()
                if st.button("Load merged table into DroPhenix", type="primary",
                             key="vsb_load_run"):
                    df_raw, warns, errs = load_data(
                        io.StringIO(merged.to_csv(index=False)))
                    if errs:
                        for e in errs:
                            st.error(e)
                    else:
                        for w in warns:
                            st.warning(w)
                        st.session_state.update({
                            "df": df_raw, "data_source": "video_batch",
                            "data_warnings": warns, "data_errors": errs,
                            "run_complete": False, "base_metrics": {},
                        })
                        _run_pipeline(df_raw, None, None)
                st.caption(
                    "Loading replaces whatever dataset is currently open and "
                    "runs the full pipeline. A batch is what the hierarchical "
                    "and ranking pages need: one video is three vials, which "
                    "is not enough to separate a genotype effect from a vial "
                    "effect.")

    # ── 5. export ─────────────────────────────────────────────────────────────
    with t_exp:
        res = st.session_state.get("_vs_result")
        if not video:
            _vs_needs_video()
        elif res is None or not res.get("ok"):
            st.info("Score the recording on the **Scoring** tab first.")
        else:
            st.markdown("#### Experimental metadata")
            st.caption(
                "Each cylinder becomes its own `VialID`, because the vial is the "
                "experimental unit: a video of three cylinders is three "
                "replicates, not one. Every scoring time becomes a row, giving "
                "the full climbing curve.")
            e1, e2, e3 = st.columns(3)
            geno = e1.text_input("Genotype", value="Oregon-R", key="vs_geno")
            trt  = e2.text_input("Treatment", value="Vehicle", key="vs_trt")
            sex  = e3.selectbox("Sex", ["Male", "Female", "Mixed"], key="vs_sex")
            e4, e5, e6 = st.columns(3)
            exp_id = e4.text_input("ExperimentID", value="", key="vs_expid")
            age    = e5.number_input("Age (days)", min_value=0.0, max_value=200.0,
                                     value=None, step=1.0, key="vs_age",
                                     placeholder="optional")
            tod    = e6.text_input("TimeOfDay (HH:MM)", value="", key="vs_tod")

            try:
                tab = cvid.to_drophenix_table(
                    res, genotype=geno or "Unknown",
                    treatment=trt or "Unknown", sex=sex,
                    experiment_id=exp_id or None,
                    age_days=(None if age is None else float(age)),
                    time_of_day=tod or None)
            except Exception as _e:
                tab = pd.DataFrame()
                st.error(f"Export failed: {_e}")

            if tab.empty:
                st.warning("No usable rows to export. Every scoring time fell "
                           "outside the clip.")
            else:
                st.session_state["_vs_table"] = tab
                st.dataframe(tab, width="stretch")
                st.download_button(
                    "Download DroPhenix CSV", tab.to_csv(index=False).encode(),
                    f"{Path(video).stem}_climbing.csv", "text/csv",
                    key="dl_vs_table")

                st.divider()
                if st.button("Load into DroPhenix", type="primary",
                             key="vs_load_run"):
                    df_raw, warns, errs = load_data(
                        io.StringIO(tab.to_csv(index=False)))
                    if errs:
                        for e in errs:
                            st.error(e)
                    else:
                        for w in warns:
                            st.warning(w)
                        st.session_state.update({
                            "df": df_raw, "data_source": "video_scoring",
                            "data_warnings": warns, "data_errors": errs,
                            "run_complete": False, "base_metrics": {},
                        })
                        _run_pipeline(df_raw, None, None)
                st.caption(
                    "Loading replaces whatever dataset is currently open and runs "
                    "the full pipeline. One video gives three vials at a handful "
                    "of times — enough for the climbing curves, but the "
                    "hierarchical and ranking pages need several videos combined.")


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
        'Negative Geotaxis Assay', 'Lifespan / Survival (KM &middot; log-rank &middot; Cox)',
        'Y-maze Cognition', 'Sleep &amp; Circadian (DAMS)', 'Larval Crawling', 'Grooming',
        'Performance Index', 'AUC &amp; t50', 'CRI &middot; SDQ &middot; TSW',
        'Motor Decline Rate (over age)', 'DTW Trajectory Clustering',
        'Cross-Strain Drug Ranking (meta-analysis)', 'Translational Score',
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
        ' <span style="font-size:1rem;font-weight:600;color:#64748B;vertical-align:middle;">v3.0</span></div>'
        + '<div style="font-size:1.02rem;color:#94A3B8;font-weight:500;'
        'margin-bottom:1rem;letter-spacing:.01em;">'
        'Multi-Assay Drosophila Phenotyping &amp; Cross-Strain Compound Ranking</div>'
        + '<p style="font-size:.9rem;color:#CBD5E1;line-height:1.8;'
        'max-width:680px;margin-bottom:1.1rem;">'
        'DroPhenix provides one reproducible, statistically rigorous workflow for '
        '<em>Drosophila melanogaster</em> neuromotor and behavioural phenotyping across '
        '<b>six assays</b> &mdash; climbing (negative geotaxis), lifespan, Y-maze, '
        'activity/sleep (DAMS), larval crawling, and grooming. It pairs assumption-aware '
        'statistics (normality/variance-gated tests, effect sizes, BH-FDR, bootstrap) with '
        'actuarial survival analysis (Kaplan&ndash;Meier, log-rank, Cox), circadian '
        'periodograms, age-resolved motor-decline kinetics, and a meta-analytic engine that '
        'ranks compounds <b>across strains of unequal size</b> by inverse-variance / '
        'DerSimonian&ndash;Laird pooling. Every result is benchmarked against published data '
        'and established tools.'
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
            'Novel Composite Metrics',
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
            'Motor Decline Kinetics',
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
            'Translational Candidate Scoring',
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
        ('Motor Decline', 'Decay Modelling (over age)',
         'Exponential decay PI(age)&nbsp;=&nbsp;PI&#8320;&middot;e<sup>&minus;k&middot;age</sup> '
         'fit over <b>age (days)</b>. Reports <em>k</em>, half-life, R&sup2;, and '
         '&Delta;<em>k</em> (treatment rescue). Motor-Survival Coupling Index (MSCI).'),
        ('Survival Analysis', 'Lifespan / Actuarial',
         'Kaplan&ndash;Meier curves with Greenwood CIs, median survival, '
         'Mantel&ndash;Cox <b>log-rank</b> (BH-FDR pairwise), and <b>Cox proportional '
         'hazards</b> (HR, 95% CI). Accepts event-level or daily-cohort data.'),
        ('Y-maze', 'Cognition / Individuality',
         'Spontaneous <b>alternation</b> (% vs chance, working memory), '
         '<b>turn bias</b>/handedness with a Monte-Carlo individuality test, '
         'and exploration. Input: per-fly arm-entry sequences.'),
        ('Activity Monitoring (DAMS)', 'Sleep &amp; Circadian',
         '<b>Sleep</b> by the &ge;5-min inactivity rule (total/day/night, bouts), '
         'and free-running <b>circadian period</b> via the chi-square '
         '(Sokolove&ndash;Bushell) periodogram with Lomb&ndash;Scargle cross-check.'),
        ('Larval Crawling', 'Locomotion',
         'Video centroid tracking of a single larva &rarr; crawling speed (mm&nbsp;s&#8315;&sup1;), '
         'path length, displacement, and tortuosity; plus summary-table analysis '
         'and group comparison.'),
        ('Grooming', 'Behaviour',
         'Quantifies scored grooming bouts &rarr; %&nbsp;time grooming, bout count and '
         'duration, grooming rate, with group statistics. Wild-type ~13% of time.'),
        ('Drug Ranking', 'Cross-Strain Meta-analysis',
         'Ranks compounds across strains of <b>unequal N</b>: per-strain Hedges&rsquo;&nbsp;<em>g</em> '
         'pooled by inverse-variance / DerSimonian&ndash;Laird random effects, '
         'composited across all assays into a Therapeutic Score with heterogeneity (I&sup2;).'),
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Survival Analysis (actuarial lifespan)
# ══════════════════════════════════════════════════════════════════════════════
def _demo_lifespan_gompertz(seed: int = 7) -> "pd.DataFrame":
    """
    Synthetic Drosophila lifespan data from a Gompertz mortality model
    (hazard h(t) = a*exp(b*t)), the standard demographic model for fly ageing.
    Disease genotypes have higher baseline mortality; the drug partially
    rescues. End-of-study censoring at day 90. Returned as event-level data.
    """
    rng = np.random.default_rng(seed)
    # (a = baseline mortality, b = ageing rate) per genotype x treatment
    groups = {
        ("Oregon-WT", "Vehicle"):   (0.0012, 0.085),
        ("Oregon-WT", "Drug"):      (0.0011, 0.083),
        ("TDP43", "Vehicle"):       (0.0060, 0.110),
        ("TDP43", "Drug"):          (0.0028, 0.095),
    }
    end_of_study = 90.0
    rows = []
    for (geno, treat), (a, b) in groups.items():
        for sex in ("Male", "Female"):
            n = int(rng.integers(45, 60))            # deliberately unequal N
            U = rng.uniform(0, 1, n)
            # inverse-CDF of Gompertz: t = (1/b) ln(1 + (b/a)(-ln(1-U)))
            t = (1.0 / b) * np.log(1.0 + (b / a) * (-np.log(1.0 - U)))
            for ti in t:
                if ti >= end_of_study:
                    rows.append((geno, treat, sex, end_of_study, 0))   # censored
                else:
                    rows.append((geno, treat, sex, round(float(ti), 1), 1))
    return pd.DataFrame(rows, columns=["Genotype", "Treatment", "Sex", "Time", "Event"])


def page_survival():
    render_header("Survival Analysis",
                  "Actuarial lifespan analysis — Kaplan-Meier, log-rank, Cox PH")
    try:
        from modules import survival as surv
    except Exception as _e:
        st.error(f"Survival module import failed: {_e}")
        return

    st.markdown(
        "Upload lifespan data in **either** layout, or load synthetic demo data:\n"
        "- **Event-level**: `Genotype, Treatment, Sex, Time, Event` "
        "(Event = 1 death, 0 censored)\n"
        "- **Daily cohort**: `Genotype, Treatment, Sex, Day, Dead_count, "
        "[Censored_count], Total_start`"
    )
    _tc1, _tc2, _ = st.columns([1, 1, 2])
    with _tc1:
        _template_button("lifespan_event_template.csv", "Event-level template", "tpl_ls_ev")
    with _tc2:
        _template_button("lifespan_cohort_template.csv", "Daily-cohort template", "tpl_ls_co")

    src = st.radio("Data source", ["Synthetic demo (Gompertz)", "Upload CSV / Excel"],
                   horizontal=True, key="surv_src")
    raw = None
    if src == "Synthetic demo (Gompertz)":
        raw = _demo_lifespan_gompertz()
        st.caption("Synthetic Gompertz mortality data: Oregon-WT vs TDP43, "
                   "Vehicle vs Drug, both sexes, unequal N, end-of-study "
                   "censoring at day 90.")
    else:
        up = st.file_uploader("Lifespan data", type=["csv", "xlsx", "xls"], key="surv_up")
        if up is not None:
            try:
                raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            except Exception as _e:
                st.error(f"Could not read file: {_e}")
                return

    if raw is None or raw.empty:
        st.info("Load or upload lifespan data to begin.")
        return

    with st.expander("Preview input data", expanded=False):
        st.dataframe(raw.head(20), width="stretch")

    # ── Parse to canonical event table ────────────────────────────────────────
    try:
        events = surv.to_event_table(raw)
    except Exception as _e:
        st.error(f"Could not parse lifespan data: {_e}")
        return
    if events.empty or events["Group"].nunique() < 1:
        st.warning("No usable survival records found after parsing.")
        return

    n_groups = events["Group"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Flies", len(events))
    c2.metric("Groups", n_groups)
    c3.metric("Total events (deaths)", int(events["Event"].sum()))

    wt_label = st.selectbox(
        "Log-rank weighting",
        ["Standard log-rank (Mantel-Cox)",
         "Gehan-Breslow (Wilcoxon, weights early deaths)",
         "Tarone-Ware"],
        key="surv_wt",
    )
    wt_map = {"Standard log-rank (Mantel-Cox)": None,
              "Gehan-Breslow (Wilcoxon, weights early deaths)": "gb",
              "Tarone-Ware": "tw"}
    weight_type = wt_map[wt_label]

    # ── Kaplan-Meier curves ───────────────────────────────────────────────────
    st.markdown("#### Kaplan-Meier survival curves")
    km = surv.kaplan_meier(events)
    all_groups = list(km.keys())
    show = st.multiselect("Groups to plot", all_groups, default=all_groups, key="surv_groups")
    show_ci = st.checkbox("Show 95% confidence bands (Greenwood / cloglog)", value=True)

    import plotly.graph_objects as go
    palette = ["#0D2137", "#F0A500", "#2A7DE1", "#C0392B",
               "#27AE60", "#8E44AD", "#16A085", "#D35400"]
    fig = go.Figure()
    for i, g in enumerate(show):
        cur = km[g]
        color = palette[i % len(palette)]
        # step from S=1 at t=0
        t = np.concatenate(([0.0], cur["Time"].to_numpy()))
        s = np.concatenate(([1.0], cur["Survival"].to_numpy()))
        fig.add_trace(go.Scatter(x=t, y=s, mode="lines", name=g,
                                 line=dict(color=color, shape="hv", width=2)))
        if show_ci:
            lo = np.concatenate(([1.0], cur["CI_lower"].to_numpy()))
            hi = np.concatenate(([1.0], cur["CI_upper"].to_numpy()))
            rgba = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself", fillcolor=rgba, line=dict(width=0),
                hoverinfo="skip", showlegend=False, name=g + " CI"))
    fig.update_layout(
        xaxis_title="Time (days)", yaxis_title="Survival probability S(t)",
        yaxis=dict(range=[0, 1.02]), height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig, width="stretch")

    t_med, t_lr, t_cox, t_rmst = st.tabs(
        ["Median Survival", "Log-rank Tests", "Cox PH Model", "RMST"])

    with t_med:
        med = surv.median_survival_table(events)
        st.dataframe(med, width="stretch")
        st.caption("Median = smallest time with S(t) <= 0.5 (R/Prism convention). "
                   "CI = Brookmeyer-Crowley. `low_confidence` flags groups with "
                   "< 3 deaths. NaN median = not reached within follow-up.")
        st.download_button("Download median CSV", med.to_csv(index=False).encode(),
                           "median_survival.csv", "text/csv", key="dl_med")

    with t_lr:
        overall = surv.logrank_overall(events, weight_type=weight_type)
        oc = st.columns(4)
        oc[0].metric("Test", overall.get("test", "Log-rank"))
        oc[1].metric("Chi-square", overall.get("chisq"))
        oc[2].metric("df", overall.get("df"))
        oc[3].metric("p-value", overall.get("p_value"))
        if overall.get("p_value") is not None and not pd.isna(overall.get("p_value")):
            if overall["p_value"] < 0.05:
                st.success(f"Overall difference between groups is significant "
                           f"({overall['Sig']}).")
            else:
                st.info("No significant overall difference between survival curves.")
        st.markdown("**Pairwise comparisons (BH-FDR corrected)**")
        pw = surv.logrank_pairwise(events, weight_type=weight_type)
        if not pw.empty:
            st.dataframe(pw, width="stretch")
            st.download_button("Download pairwise log-rank CSV",
                               pw.to_csv(index=False).encode(),
                               "logrank_pairwise.csv", "text/csv", key="dl_lr")
        else:
            st.info("Need >= 2 groups for pairwise comparison.")

    with t_cox:
        st.caption("Cox proportional-hazards model. HR > 1 = higher death "
                   "hazard vs the reference level; HR < 1 = protective.")
        if n_groups < 2:
            st.info("Cox model needs >= 2 groups.")
        else:
            try:
                cox = surv.cox_ph(events)
                if "error" in cox:
                    st.warning(cox["error"])
                else:
                    cc = st.columns(3)
                    cc[0].metric("N", cox["n"])
                    cc[1].metric("Events", cox["n_events"])
                    cc[2].metric("Concordance (c-index)", cox["concordance"])
                    st.dataframe(cox["summary"], width="stretch")
                    st.code(cox["formula"], language="text")
                    st.download_button("Download Cox summary CSV",
                                       cox["summary"].to_csv(index=False).encode(),
                                       "cox_ph.csv", "text/csv", key="dl_cox")
            except Exception as _e:
                st.warning(f"Cox model could not be fit: {_e}")

    with t_rmst:
        st.caption("Restricted Mean Survival Time = area under the KM curve up "
                   "to tau (common follow-up horizon). Model-free; valid even "
                   "when hazards are non-proportional.")
        rmst = surv.rmst_table(events)
        st.dataframe(rmst, width="stretch")
        st.download_button("Download RMST CSV", rmst.to_csv(index=False).encode(),
                           "rmst.csv", "text/csv", key="dl_rmst")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Y-maze (spontaneous alternation, turn bias, exploration)
# ══════════════════════════════════════════════════════════════════════════════
def page_ymaze():
    render_header("Y-maze",
                  "Spontaneous alternation, turn bias / handedness, exploration")
    try:
        from modules import ymaze as ym
    except Exception as _e:
        st.error(f"Y-maze module import failed: {_e}")
        return

    st.markdown(
        "Upload per-fly Y-maze data, or load synthetic demo data:\n"
        "- **Per-fly sequence**: `Genotype, Treatment, Sex, FlyID, Sequence` "
        "where Sequence is the ordered arm-entry string, e.g. `ABCACB...`\n"
        "- **Long per-entry**: one row per arm entry with `Genotype, Treatment, "
        "Sex, FlyID, Order, Arm` (and/or a `Turn` column of L/R)"
    )
    _template_button("ymaze_template.csv", key="tpl_ym")

    src = st.radio("Data source", ["Synthetic demo (de Bivort/Kim-calibrated)",
                                    "Upload CSV / Excel"], horizontal=True, key="ym_src")
    raw = None
    if src.startswith("Synthetic"):
        raw = ym.demo_ymaze()
        st.caption("Synthetic data: turn bias ~ Beta(4.6,4.6) (population mean "
                   "~0.50, ~23.5% of flies outside [0.3,0.7], per de Bivort 2015); "
                   "Oregon-WT alternates above chance, TDP43 at chance (memory "
                   "deficit), drug partially rescues TDP43. Unequal N per group.")
    else:
        up = st.file_uploader("Y-maze data", type=["csv", "xlsx", "xls"], key="ym_up")
        if up is not None:
            try:
                raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            except Exception as _e:
                st.error(f"Could not read file: {_e}")
                return
    if raw is None or raw.empty:
        st.info("Load or upload Y-maze data to begin.")
        return

    with st.expander("Preview input data", expanded=False):
        st.dataframe(raw.head(20), width="stretch")

    chance = st.slider("Spontaneous-alternation chance level (%)", 20.0, 60.0, 50.0, 0.1,
                       help="Field-standard 50% for sequential 3-arm alternation "
                            "(Kim et al. 2017). Some authors use 33.3% (pure-random "
                            "arm choice) or 22.2% (rodent SAP).") / 100.0

    try:
        res = ym.run_ymaze_analysis(raw, chance=chance)
    except Exception as _e:
        st.error(f"Could not analyse Y-maze data: {_e}")
        return
    per_fly = res["per_fly"]
    if per_fly.empty:
        st.warning("No usable Y-maze records found after parsing.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Flies", len(per_fly))
    c2.metric("Groups", per_fly.groupby(["Genotype", "Treatment", "Sex"]).ngroups)
    c3.metric("Mean entries / fly", round(float(per_fly["n_entries"].mean()), 1))

    import plotly.graph_objects as go
    palette = ["#0D2137", "#F0A500", "#2A7DE1", "#C0392B",
               "#27AE60", "#8E44AD", "#16A085", "#D35400"]

    t_alt, t_tb, t_exp, t_cmp = st.tabs(
        ["Spontaneous Alternation", "Turn Bias / Handedness", "Exploration",
         "Group Comparison"])

    with t_alt:
        alt = res["alternation"]
        alt = alt.copy()
        alt["Group"] = alt["Genotype"] + " | " + alt["Treatment"] + " | " + alt["Sex"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=alt["Group"], y=alt["Mean_alternation_pct"],
            error_y=dict(type="data", array=alt["SEM"]),
            marker_color="#2A7DE1", name="Mean alternation %"))
        fig.add_hline(y=chance * 100, line_dash="dash", line_color="#C0392B",
                      annotation_text=f"chance = {chance*100:.0f}%")
        fig.update_layout(yaxis_title="Spontaneous alternation (%)", height=420,
                          margin=dict(l=40, r=20, t=30, b=80))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(alt.drop(columns="Group"), width="stretch")
        st.caption("One-sample test of each group's alternation vs the chance "
                   "level (t-test if n>=8 and ~normal, else Wilcoxon).")
        st.download_button("Download alternation CSV", alt.to_csv(index=False).encode(),
                           "ymaze_alternation.csv", "text/csv", key="dl_ymalt")

    with t_tb:
        tb = res["turn_bias"]
        fig = go.Figure()
        groups = per_fly.assign(
            G=per_fly["Genotype"] + " | " + per_fly["Treatment"] + " | " + per_fly["Sex"])
        for i, (g, sub) in enumerate(groups.groupby("G")):
            fig.add_trace(go.Histogram(
                x=sub["turn_bias"], name=g, opacity=0.6,
                marker_color=palette[i % len(palette)], xbins=dict(start=0, end=1, size=0.05)))
        fig.add_vrect(x0=0.30, x1=0.70, fillcolor="grey", opacity=0.10, line_width=0,
                      annotation_text="central [0.3-0.7]")
        fig.update_layout(barmode="overlay", xaxis_title="Turn bias  P(right)",
                          yaxis_title="# flies", height=420,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(tb, width="stretch")
        st.caption("Mean turn bias ~0.5 is expected. `Pct_extreme_30_70` and the "
                   "Monte-Carlo `Individuality_p` (observed vs binomial-null MAD) "
                   "quantify stable individual handedness (de Bivort method).")
        st.download_button("Download turn-bias CSV", tb.to_csv(index=False).encode(),
                           "ymaze_turnbias.csv", "text/csv", key="dl_ymtb")

    with t_exp:
        exp = res["exploration"]
        st.dataframe(exp, width="stretch")
        st.caption("Total arm entries (choices) per fly — general activity / "
                   "exploration read-out.")
        st.download_button("Download exploration CSV", exp.to_csv(index=False).encode(),
                           "ymaze_exploration.csv", "text/csv", key="dl_ymexp")

    with t_cmp:
        st.markdown("Compare a per-fly metric across groups "
                    "(normality-gated ANOVA/Kruskal, BH-FDR post-hoc).")
        mc1, mc2 = st.columns(2)
        metric = mc1.selectbox("Metric", ["alternation_pct", "turn_bias",
                                          "n_entries", "switch_rate"], key="ym_metric")
        gcol = mc2.selectbox("Group by", ["Treatment", "Genotype", "Sex"], key="ym_gcol")
        if st.button("Run comparison", key="ym_cmp_btn"):
            out = ym.compare_groups(per_fly, metric, gcol)
            if not out:
                st.warning("Need >= 2 groups with >= 2 flies each.")
            else:
                st.markdown(f"**Method:** `{out.get('method','')}`")
                main = out.get("main_result", {})
                if main:
                    cols = st.columns(min(len(main), 4))
                    for i, (k, v) in enumerate(main.items()):
                        try: _v = round(float(v), 4)
                        except Exception: _v = str(v)
                        cols[i % 4].metric(k.replace("_", " "), _v)
                pw = out.get("pairwise", pd.DataFrame())
                if isinstance(pw, pd.DataFrame) and not pw.empty:
                    st.dataframe(pw, width="stretch")
                    st.download_button("Download comparison CSV",
                                       pw.to_csv(index=False).encode(),
                                       "ymaze_comparison.csv", "text/csv", key="dl_ymcmp")

    st.divider()
    with st.expander("Per-fly metric table", expanded=False):
        st.dataframe(per_fly, width="stretch")
        st.download_button("Download per-fly CSV", per_fly.to_csv(index=False).encode(),
                           "ymaze_per_fly.csv", "text/csv", key="dl_ympf")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DAMS (activity monitoring: sleep + circadian period)
# ══════════════════════════════════════════════════════════════════════════════
def page_dams():
    render_header("Activity Monitoring (DAMS)",
                  "Sleep / rest analysis and circadian period from DAM data")
    try:
        from modules import dams as dm
    except Exception as _e:
        st.error(f"DAMS module import failed: {_e}")
        return

    st.markdown(
        "Upload per-minute DAM activity, or load synthetic demo data. "
        "Canonical long format: `Genotype, Treatment, Sex, FlyID, Time_min, "
        "Activity` (one row per fly per time bin; Activity = beam crossings).")
    _template_button("dams_template.csv", key="tpl_dm")

    src = st.radio("Data source", ["Synthetic demo (DD free-running)",
                                   "Upload CSV / Excel"], horizontal=True, key="dm_src")
    raw = None
    if src.startswith("Synthetic"):
        with st.spinner("Generating synthetic DAM data..."):
            raw = dm.demo_dams()
        st.caption("Synthetic constant-darkness (DD) data: Oregon-WT ~24 h, "
                   "TDP43-vehicle disrupted (~22.5 h), drug partially rescues "
                   "(~23.6 h) — cf. Konopka & Benzer (1971). Males sleep more "
                   "than females (sexual dimorphism). Unequal N per group.")
    else:
        up = st.file_uploader("DAM data", type=["csv", "xlsx", "xls"], key="dm_up")
        if up is not None:
            try:
                raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            except Exception as _e:
                st.error(f"Could not read file: {_e}")
                return
    if raw is None or raw.empty:
        st.info("Load or upload DAM data to begin.")
        return

    c0a, c0b = st.columns(2)
    sleep_min = c0a.number_input("Sleep threshold (min of inactivity)", 1, 30, 5, 1,
                                 help="Drosophila sleep = >=5 min zero activity "
                                      "(Hendricks/Shaw 2000).")
    do_period = c0b.checkbox("Compute circadian period (chi-square periodogram)",
                             value=True)

    sig = (len(raw), tuple(raw.columns), int(sleep_min), bool(do_period))
    if st.session_state.get("_dams_sig") != sig:
        with st.spinner("Analysing DAM data (sleep + periodogram)..."):
            try:
                st.session_state["_dams_res"] = dm.run_dams_analysis(
                    raw, sleep_min=sleep_min, do_period=do_period)
                st.session_state["_dams_sig"] = sig
            except Exception as _e:
                st.error(f"DAMS analysis failed: {_e}")
                return
    res = st.session_state.get("_dams_res", {})
    spf = res.get("sleep_per_fly", pd.DataFrame())
    if spf.empty:
        st.warning("No usable DAM records found after parsing.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Flies", len(spf))
    c2.metric("Groups", spf.groupby(["Genotype", "Treatment", "Sex"]).ngroups)
    c3.metric("Recording (bins/fly)", int(spf["n_bins"].median()))

    import plotly.graph_objects as go
    palette = ["#0D2137", "#F0A500", "#2A7DE1", "#C0392B",
               "#27AE60", "#8E44AD", "#16A085", "#D35400"]

    t_sleep, t_prof, t_per, t_cmp = st.tabs(
        ["Sleep", "Sleep Profile", "Circadian Period", "Group Comparison"])

    with t_sleep:
        ss = res["sleep_summary"].copy()
        ss["Group"] = ss["Genotype"] + " | " + ss["Treatment"] + " | " + ss["Sex"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ss["Group"], y=ss["Day_sleep_min_per_day"],
                             name="Day sleep", marker_color="#F0A500"))
        fig.add_trace(go.Bar(x=ss["Group"], y=ss["Night_sleep_min_per_day"],
                             name="Night sleep", marker_color="#0D2137"))
        fig.update_layout(barmode="stack", yaxis_title="Sleep (min / 24 h)",
                          height=420, legend=dict(orientation="h", y=1.02),
                          margin=dict(l=40, r=20, t=30, b=90))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(res["sleep_summary"], width="stretch")
        st.caption("Sleep = >=%d-min zero-activity bouts (Hendricks/Shaw 2000). "
                   "Totals normalised to minutes per 24 h." % int(sleep_min))
        st.download_button("Download sleep summary CSV",
                           res["sleep_summary"].to_csv(index=False).encode(),
                           "dams_sleep_summary.csv", "text/csv", key="dl_dmsleep")

    with t_prof:
        prof = res.get("sleep_profile", pd.DataFrame())
        if prof.empty:
            st.info("Sleep profile not available.")
        else:
            prof = prof.copy()
            prof["Group"] = prof["Genotype"] + " | " + prof["Treatment"] + " | " + prof["Sex"]
            fig = go.Figure()
            for i, (g, sub) in enumerate(prof.groupby("Group")):
                sub = sub.sort_values("ZT_hour")
                fig.add_trace(go.Scatter(x=sub["ZT_hour"], y=sub["sleep_min"],
                                         mode="lines+markers", name=g,
                                         line=dict(color=palette[i % len(palette)])))
            fig.update_layout(xaxis_title="Zeitgeber / circadian time (h)",
                              yaxis_title="Sleep (min per epoch)", height=420,
                              legend=dict(orientation="h", y=1.02),
                              margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig, width="stretch")
            st.caption("Sleep across the circadian cycle. Wild-type flies show a "
                       "bimodal profile (rest in subjective night, midday siesta).")

    with t_per:
        if not do_period or "period_summary" not in res:
            st.info("Enable 'Compute circadian period' above to run the periodogram.")
        else:
            st.dataframe(res["period_summary"], width="stretch")
            st.caption("Chi-square (Sokolove-Bushell 1978) periodogram, "
                       "Bonferroni-corrected across tested periods; Lomb-Scargle "
                       "cross-check per fly. Reference (Konopka & Benzer 1971): "
                       "WT ~24 h.")
            st.download_button("Download period summary CSV",
                               res["period_summary"].to_csv(index=False).encode(),
                               "dams_period_summary.csv", "text/csv", key="dl_dmper")
            # periodogram of one representative fly
            ppf = res["period_per_fly"]
            long_df = res["long"]
            fly_ids = ppf["FlyID"].tolist()
            sel = st.selectbox("Show periodogram for fly", fly_ids, key="dm_flysel")
            sub = long_df[long_df["FlyID"] == sel].sort_values("Time_min")
            if not sub.empty:
                t = sub["Time_min"].to_numpy(float); act = sub["Activity"].to_numpy(float)
                raw_bin = dm._bin_minutes(t)
                factor = max(1, int(round(30.0 / raw_bin)))
                K = len(act) // factor
                act_rs = act[:K * factor].reshape(K, factor).sum(axis=1) if factor > 1 else act
                cp = dm.chi_square_periodogram(act_rs, raw_bin * factor, 16, 32)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cp["periods_h"], y=cp["Qp"], mode="lines",
                                         name="Qp", line=dict(color="#0D2137")))
                fig.add_trace(go.Scatter(x=cp["periods_h"], y=cp["threshold"], mode="lines",
                                         name="significance", line=dict(color="#C0392B", dash="dash")))
                fig.add_vline(x=cp["peak_period_h"], line_dash="dot", line_color="#F0A500",
                              annotation_text=f"{cp['peak_period_h']} h")
                fig.update_layout(xaxis_title="Test period (h)", yaxis_title="Qp",
                                  height=380, legend=dict(orientation="h", y=1.02),
                                  margin=dict(l=40, r=20, t=30, b=40))
                st.plotly_chart(fig, width="stretch")

    with t_cmp:
        st.markdown("Compare a per-fly metric across groups "
                    "(normality-gated ANOVA/Kruskal, BH-FDR post-hoc).")
        opts = ["Total_sleep_min_per_day", "Night_sleep_min_per_day",
                "Day_sleep_min_per_day", "N_bouts_per_day", "Mean_bout_min",
                "Total_activity_per_day"]
        if "period_per_fly" in res:
            opts.append("Period_h_chisq")
        mc1, mc2 = st.columns(2)
        metric = mc1.selectbox("Metric", opts, key="dm_metric")
        gcol = mc2.selectbox("Group by", ["Treatment", "Genotype", "Sex"], key="dm_gcol")
        if st.button("Run comparison", key="dm_cmp_btn"):
            src_df = res["period_per_fly"] if metric == "Period_h_chisq" else spf
            out = dm.compare_groups(src_df, metric, gcol)
            if not out:
                st.warning("Need >= 2 groups with >= 2 flies each.")
            else:
                st.markdown(f"**Method:** `{out.get('method','')}`")
                main = out.get("main_result", {})
                if main:
                    cols = st.columns(min(len(main), 4))
                    for i, (k, v) in enumerate(main.items()):
                        try: _v = round(float(v), 4)
                        except Exception: _v = str(v)
                        cols[i % 4].metric(k.replace("_", " "), _v)
                pw = out.get("pairwise", pd.DataFrame())
                if isinstance(pw, pd.DataFrame) and not pw.empty:
                    st.dataframe(pw, width="stretch")
                    st.download_button("Download comparison CSV",
                                       pw.to_csv(index=False).encode(),
                                       "dams_comparison.csv", "text/csv", key="dl_dmcmp")

    st.divider()
    with st.expander("Per-fly sleep table", expanded=False):
        st.dataframe(spf, width="stretch")
        st.download_button("Download per-fly sleep CSV", spf.to_csv(index=False).encode(),
                           "dams_per_fly.csv", "text/csv", key="dl_dmpf")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DAM Run Analysis (raw monitor files -> design-aware, verified report)
# ══════════════════════════════════════════════════════════════════════════════
def page_dam_pipeline():
    render_header("DAM Run Analysis",
                  "Raw TriKinetics monitor files to a design-aware, verified report")
    try:
        from modules import dam_pipeline as DP_, dam_figures as DF_, dam_report as DRp_
        from modules import dam_reader as DRd_
    except Exception as _e:
        st.error(f"Pipeline import failed: {_e}")
        return
    import io as _io
    import os
    import tempfile as _tmp
    import zipfile as _zip
    import matplotlib.pyplot as plt

    st.markdown(
        "Upload the **raw `MonitorN.txt` files** DAMSystem wrote (one row per minute) and the "
        "**channel map** saying which tube held which genotype, sex and treatment. The tool then: "
        "reads the light channel of every monitor and picks the one that cycles; measures the "
        "photoperiod; proposes the analysis window (complete days with a consistent light schedule, "
        "no gaps, no flicker); drops dead tubes; scores every published sleep parameter; flags "
        "outliers by a pre-specified rule; labels every contrast monitor-clean or confounded from "
        "the map; runs sex-stratified primary tests with FDR; and writes a Word report whose every "
        "number is checked against the tables. The quick-look **Activity Monitoring (DAMS)** page "
        "still takes a pre-formatted CSV.")

    def _save_uploads(ups, tag):
        paths = []
        tdir = st.session_state.setdefault("_dp_tmpdir", _tmp.mkdtemp(prefix="drophenix_dam_"))
        for u in ups:
            p = os.path.join(tdir, f"{tag}_{u.name}")
            with open(p, "wb") as f:
                f.write(u.getbuffer())
            paths.append(p)
        return paths

    # ── inputs ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    ups = c1.file_uploader("Monitor files (raw DAMSystem .txt, one per monitor)", type=["txt"],
                           accept_multiple_files=True, key="dp_mon")
    cm_up = c2.file_uploader("Channel map (CSV: Monitor, Channel, Genotype, Sex, Treatment)",
                             type=["csv"], key="dp_map")
    with c2.expander("Need a channel map?"):
        st.caption("Make one on the **Plan a Run** page, or download this template and fill it in. "
                   "Monitor names may be `Monitor_1`, `Monitor1`, `M1` — the first number in the "
                   "name identifies the monitor.")
        tpl = DRd_.channel_map_template(["Monitor_1", "Monitor_2"])
        tpl["Treatment"] = ""
        st.download_button("Channel map template", tpl.to_csv(index=False).encode(),
                           "channel_map_template.csv", "text/csv", key="dp_tpl")
    run_name = c1.text_input("Name for this run (e.g. 'day 15')", value="run A", key="dp_name")

    with st.expander("Second run (optional): compare two timepoints", expanded=False):
        ups_b = st.file_uploader("Monitor files, second run", type=["txt"], accept_multiple_files=True, key="dp_mon_b")
        cm_up_b = st.file_uploader("Channel map, second run (leave empty to reuse the first)", type=["csv"], key="dp_map_b")
        run_name_b = st.text_input("Name for the second run", value="run B", key="dp_name_b")

    if not ups or cm_up is None:
        st.info("Upload the monitor files and a channel map to begin.")
        return

    # ── load + quality ────────────────────────────────────────────────────────
    sig = (tuple(sorted(u.name for u in ups)), cm_up.name, run_name,
           tuple(sorted(u.name for u in (ups_b or []))), (cm_up_b.name if cm_up_b else None), run_name_b)
    if st.session_state.get("_dp_sig") != sig:
        for k in ("_dp_lab", "_dp_q", "_dp_lab_b", "_dp_q_b", "_dp_run", "_dp_run_b", "_dp_cmp", "_dp_figs"):
            st.session_state.pop(k, None)
        st.session_state["_dp_sig"] = sig
        try:
            cmap = pd.read_csv(cm_up)
            with st.spinner("Reading monitor files and checking the recording..."):
                lab = DP_.load_run(_save_uploads(ups, "A"), cmap,
                                   monitor_names=[Path(u.name).stem for u in ups])
                st.session_state["_dp_lab"] = lab
                st.session_state["_dp_q"] = DP_.quality_report(lab)
                if ups_b:
                    cmap_b = pd.read_csv(cm_up_b) if cm_up_b is not None else cmap
                    lab_b = DP_.load_run(_save_uploads(ups_b, "B"), cmap_b,
                                         monitor_names=[Path(u.name).stem for u in ups_b])
                    st.session_state["_dp_lab_b"] = lab_b
                    st.session_state["_dp_q_b"] = DP_.quality_report(lab_b)
        except Exception as _e:
            st.error(f"Could not read the files: {_e}")
            return
    lab, q = st.session_state["_dp_lab"], st.session_state["_dp_q"]
    lab_b, q_b = st.session_state.get("_dp_lab_b"), st.session_state.get("_dp_q_b")

    st.subheader("1 · What the recording contains")
    runs_q = [(run_name, q)] + ([(run_name_b, q_b)] if q_b else [])
    cols = st.columns(len(runs_q))
    windows = {}
    for col, (nm, qq) in zip(cols, runs_q):
        with col:
            st.markdown(f"**{nm}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Bin (min)", f"{qq['bin_min']:.0f}")
            m2.metric("Photoperiod", f"{(qq['photoperiod_min'] or 0)/60:.1f} h light")
            m3.metric("Usable days", qq["n_usable_days"])
            if not qq["sleep_scoreable"]:
                st.error("This file is binned wider than the 5-min sleep criterion; sleep cannot be "
                         "scored from it. Re-export at 1-minute binning from DAMSystem.")
            for f in qq["faults"]:
                st.warning(f)
            st.dataframe(qq["monitors"], width="stretch", hide_index=True)
            st.dataframe(qq["days"][["date", "minutes", "light_h", "transitions", "largest_gap_min", "usable", "exclude_reason"]],
                         width="stretch", hide_index=True)
            if qq["window"] is None:
                st.error("No usable day found."); return
            d0, d1 = qq["window"][0].date(), (qq["window"][1] - pd.Timedelta(1, "D")).date()
            st.caption(f"Proposed window: **{d0} to {d1}** ({qq['n_usable_days']} complete days). Adjust if needed:")
            w0 = st.date_input("first day", d0, key=f"dp_w0_{nm}")
            w1 = st.date_input("last day", d1, key=f"dp_w1_{nm}")
            windows[nm] = (pd.Timestamp(w0), pd.Timestamp(w1) + pd.Timedelta(1, "D"))

    # ── design ────────────────────────────────────────────────────────────────
    st.subheader("2 · Design")
    tidy = DRd_.to_dams_long(lab, drop_dead=False)
    d_auto = DP_.Design().resolve(tidy)
    dc = st.columns(4)
    genos = sorted(tidy["Genotype"].astype(str).unique()); trts = sorted(tidy["Treatment"].astype(str).unique())
    ref_g = dc[0].selectbox("Reference genotype", genos, index=genos.index(d_auto.ref_genotype) if d_auto.ref_genotype in genos else 0, key="dp_refg")
    ref_t = dc[1].selectbox("Reference treatment", trts, index=trts.index(d_auto.ref_treatment) if d_auto.ref_treatment in trts else 0, key="dp_reft")
    strat_opts = ["Sex", "none"] if tidy["Sex"].nunique() > 1 else ["none"]
    strat = dc[2].selectbox("Stratify by", strat_opts, key="dp_strat")
    dc[3].metric("Tubes labelled", int(tidy.groupby(["Monitor", "Channel"]).ngroups))
    design = DP_.Design(ref_genotype=ref_g, ref_treatment=ref_t, stratify_col=None if strat == "none" else strat)

    if st.button("Run the analysis", type="primary", key="dp_run_btn"):
        with st.spinner("Scoring sleep, testing every contrast, drawing figures..."):
            try:
                run = DP_.analyse_run(lab, run_name, design, windows[run_name], q)
                st.session_state["_dp_run"] = run
                if lab_b is not None:
                    run_b = DP_.analyse_run(lab_b, run_name_b, design, windows[run_name_b], q_b)
                    st.session_state["_dp_run_b"] = run_b
                    st.session_state["_dp_cmp"] = DP_.compare_runs(run, run_b)
                else:
                    st.session_state.pop("_dp_run_b", None); st.session_state.pop("_dp_cmp", None)
                st.session_state.pop("_dp_figs", None)
            except Exception as _e:
                st.error(f"Analysis failed: {_e}")
                return
    run = st.session_state.get("_dp_run")
    if run is None:
        return
    run_b, cmp = st.session_state.get("_dp_run_b"), st.session_state.get("_dp_cmp")
    d = run.design

    # ── results ───────────────────────────────────────────────────────────────
    st.subheader("3 · Results")
    s = run.summary()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flies analysed", s["n_flies"]); k2.metric("Outliers flagged", s["n_outliers"])
    k3.metric("Photoperiod", s["photoperiod"]); k4.metric("Window", f"{s['window'][0]} → {s['window'][1]}")
    tri = run.triage
    for f in tri.blockers:
        st.error(f"Triage: {f.message}")
    for f in tri.warnings:
        st.warning(f"Triage: {f.message}")

    if "_dp_figs" not in st.session_state:
        with st.spinner("Drawing figures..."):
            figs = {}
            def _png(fig):
                b = _io.BytesIO(); fig.savefig(b, format="png", dpi=170); plt.close(fig); return b.getvalue()
            figs["Primary analysis: every headline effect with its FDR q."] = _png(DF_.headline(run, run_b, cmp))
            figs[f"Sleep profile by genotype, {d.ref_treatment} flies, {run.name}."] = _png(DF_.profiles(run, by="genotype"))
            if len(trts) > 1:
                figs[f"Effect of treatment on the sleep profile, {run.name}."] = _png(DF_.profiles(run, by="treatment"))
            figs[f"Sleep amount, {run.name}."] = _png(DF_.bars(run, ["Total_sleep_min_per_day", "Day_sleep_min_per_day", "Night_sleep_min_per_day", "Sleep_latency_night_min"], run.name))
            figs[f"Sleep architecture, {run.name}."] = _png(DF_.bars(run, ["N_bouts_per_day", "Mean_bout_min", "Night_mean_bout_min", "Longest_bout_min"], run.name))
            figs[f"Activity, {run.name}."] = _png(DF_.bars(run, ["Total_activity_per_day", "Waking_activity", "Day_activity_per_light_h", "Night_activity_per_dark_h"], run.name))
            figs[f"Outlier map, {run.name}."] = _png(DF_.outlier_map(run))
            if run_b is not None:
                figs[f"Sleep profile by genotype, {run_b.name}."] = _png(DF_.profiles(run_b, by="genotype"))
                figs[f"{run.name} (dashed) vs {run_b.name} (solid), {d.ref_treatment} flies."] = _png(DF_.profiles(run, by="run", run_b=run_b))
                figs[f"{run.name} vs {run_b.name} within each group."] = _png(DF_.age_bars(run, run_b, cmp))
            st.session_state["_dp_figs"] = figs
    figs = st.session_state["_dp_figs"]

    t_ans, t_fig, t_tab, t_out, t_dl = st.tabs(["Answers", "Figures", "Tables", "Outliers", "Report & downloads"])
    with t_ans:
        st.image(list(figs.values())[0], width="stretch")
        st.caption("Filled squares survive Benjamini–Hochberg FDR at 5% in the pre-specified stratified test. "
                   "Grey rows are confounded with monitor and are not evidence on their own.")
        rows = []
        for fam, v in s["families"].items():
            rows.append({"family": fam.replace("_", " "), "primary tests": v["n_tests"], "significant (FDR 5%)": v["n_significant"]})
        if cmp is not None:
            p = cmp["timepoint"]; p = p[p.analysis == "primary"]
            rows.append({"family": f"{run.name} → {run_b.name}", "primary tests": len(p), "significant (FDR 5%)": int(p.significant.sum())})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with t_fig:
        for cap, png in list(figs.items())[1:]:
            st.image(png, width="stretch"); st.caption(cap)
    with t_tab:
        fam = st.selectbox("Family", list(run.contrasts.keys()) + (["timepoint"] if cmp else []), key="dp_fam")
        variant = st.radio("Flies", ["all flies (primary)", "outliers removed (sensitivity)"], horizontal=True, key="dp_var")
        if fam == "timepoint":
            df = cmp["timepoint"] if variant.startswith("all") else cmp["timepoint_noout"]
        else:
            df = (run.contrasts if variant.startswith("all") else run.contrasts_noout)[fam]
        show = ["metric", "stratum", "ref", "cmp", "n_per_stratum", "mean_ref", "mean_cmp", "pct_change",
                "hedges_g", "g_lo", "g_hi", "I2_pct", "p", "q_BH", "significant", "monitor_clean"]
        st.dataframe(df[df.analysis == "primary"][[c for c in show if c in df.columns]], width="stretch", hide_index=True)
        with st.expander("Secondary (per-stratum, uncorrected)"):
            st.dataframe(df[df.analysis == "secondary"][[c for c in show if c in df.columns and c not in ("q_BH", "significant", "n_per_stratum", "I2_pct")]],
                         width="stretch", hide_index=True)
        with st.expander("Group means"):
            st.dataframe(run.group_table(), width="stretch")
    with t_out:
        o = run.per_fly[run.per_fly.Outlier]
        st.dataframe(o[["Monitor", "Channel", d.genotype_col] + ([d.stratify_col] if d.stratify_col else []) + [d.treatment_col, "Outlier_reason",
                        "Total_sleep_min_per_day", "z_Total_sleep_min_per_day", "Total_activity_per_day", "z_Total_activity_per_day"]],
                     width="stretch", hide_index=True)
        st.caption(f"Rule: |modified z| > {DP_.OUTLIER_Z} versus the fly's own group median, MAD pooled per run, on total "
                   "sleep or total activity. The all-flies analysis is primary; every table is also given without these flies.")
    with t_dl:
        if st.button("Build the Word report", key="dp_docx"):
            with st.spinner("Writing and verifying the report..."):
                tdir = st.session_state["_dp_tmpdir"]
                out = os.path.join(tdir, "DAM_report.docx")
                DRp_.build(run, out, run_b=run_b, cmp=cmp, figures=figs)
                ok, rep = DRp_.verify(out, run, run_b, cmp)
                st.session_state["_dp_docx"] = (out, ok, rep)
        if "_dp_docx" in st.session_state:
            out, ok, rep = st.session_state["_dp_docx"]
            (st.success if ok else st.error)(f"Verification: {rep}")
            with open(out, "rb") as f:
                st.download_button("Download Word report", f.read(), "DAM_report.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dp_dl_docx")
        buf = _io.BytesIO()
        with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as z:
            for r in [run] + ([run_b] if run_b else []):
                tag = r.name.replace(" ", "_")
                z.writestr(f"{tag}/per_fly.csv", r.per_fly.to_csv(index=False))
                z.writestr(f"{tag}/quality_days.csv", r.quality["days"].to_csv(index=False))
                z.writestr(f"{tag}/profiles_per_fly.csv", r.profiles.to_csv(index=False))
                for k, df in r.contrasts.items():
                    z.writestr(f"{tag}/contrasts_{k}.csv", df.to_csv(index=False))
                for k, df in r.contrasts_noout.items():
                    z.writestr(f"{tag}/contrasts_{k}_nooutliers.csv", df.to_csv(index=False))
            if cmp is not None:
                z.writestr("timepoint/contrasts_timepoint.csv", cmp["timepoint"].to_csv(index=False))
                z.writestr("timepoint/contrasts_timepoint_nooutliers.csv", cmp["timepoint_noout"].to_csv(index=False))
                z.writestr("timepoint/pairing_test.csv", cmp["pairing"].to_csv(index=False))
            for i, (cap, png) in enumerate(figs.items(), 1):
                z.writestr(f"figures/fig{i:02d}.png", png)
        st.download_button("Download all tables and figures (zip)", buf.getvalue(), "DAM_analysis.zip", "application/zip", key="dp_dl_zip")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Larval Crawling
# ══════════════════════════════════════════════════════════════════════════════
def page_larval():
    render_header("Larval Crawling",
                  "Larval locomotion — video centroid tracking + crawling metrics")
    try:
        from modules import larval as lv
    except Exception as _e:
        st.error(f"Larval module import failed: {_e}")
        return

    _template_button("larval_template.csv", "Summary CSV template", key="tpl_lv")
    mode = st.radio("Mode", ["Synthetic demo (summary)", "Track a video",
                             "Upload summary CSV"], horizontal=True, key="lv_mode")

    per_larva = None
    if mode == "Synthetic demo (summary)":
        per_larva = lv.demo_larval()
        st.caption("Synthetic per-larva crawling summaries: Oregon-WT ~1.0 mm/s, "
                   "TDP43 slower (~0.5 mm/s, more tortuous path), drug partial "
                   "rescue. Published WT third-instar speed ~0.5-2 mm/s.")
    elif mode == "Track a video":
        up = st.file_uploader("Larva video (single larva, dark on light plate)",
                              type=["mp4", "avi", "mov", "mkv"], key="lv_vid")
        c1, c2, c3 = st.columns(3)
        px_per_mm = c1.number_input("Pixels per mm (spatial calibration)", 0.0, 1000.0, 10.0, 0.5,
                                    help="Measure a known distance in the frame to set this; "
                                         "required for mm/s speeds.")
        geno = c2.text_input("Genotype", "Oregon-WT", key="lv_g")
        treat = c3.text_input("Treatment", "Vehicle", key="lv_t")
        if up is not None and st.button("Track larva", key="lv_track"):
            import tempfile, os
            tmp = os.path.join(tempfile.gettempdir(), up.name)
            with open(tmp, "wb") as fh:
                fh.write(up.getbuffer())
            with st.spinner("Tracking larva centroid..."):
                try:
                    res = lv.track_larva(tmp, px_per_mm=px_per_mm or None)
                except Exception as _e:
                    st.error(f"Tracking failed: {_e}")
                    return
            st.session_state["_lv_track"] = res
            st.session_state["_lv_meta"] = (geno, treat)
        res = st.session_state.get("_lv_track")
        if res:
            s = res["summary"]
            cc = st.columns(4)
            cc[0].metric("Frames tracked", s.get("n_points", 0))
            cc[1].metric("Mean speed (mm/s)", s.get("mean_speed_mm_s"))
            cc[2].metric("Path length (mm)", s.get("path_length_mm"))
            cc[3].metric("Tortuosity", s.get("tortuosity"))
            traj = res["trajectory"]
            if not traj.empty:
                import plotly.graph_objects as go
                fig = go.Figure(go.Scatter(x=traj["x_px"], y=traj["y_px"], mode="lines+markers",
                                           marker=dict(size=3), line=dict(color="#2A7DE1")))
                fig.update_layout(title="Crawling path", xaxis_title="x (px)",
                                  yaxis_title="y (px)", height=420,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(pd.DataFrame([s]), width="stretch")
        return
    else:
        up = st.file_uploader("Per-larva summary CSV (Genotype, Treatment, Sex, "
                              "FlyID, mean_speed_mm_s, ...)", type=["csv", "xlsx"], key="lv_csv")
        if up is not None:
            per_larva = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        if per_larva is None:
            st.info("Upload a per-larva summary to analyse.")
            return

    out = lv.run_larval_analysis(per_larva)
    summ = out["summary"]
    c1, c2 = st.columns(2)
    c1.metric("Larvae", len(per_larva))
    c2.metric("Groups", per_larva.groupby(["Genotype", "Treatment", "Sex"]).ngroups
              if "Genotype" in per_larva.columns else 0)

    t_sum, t_cmp = st.tabs(["Group Summary", "Group Comparison"])
    with t_sum:
        if not summ.empty:
            import plotly.graph_objects as go
            s2 = summ.copy(); s2["Group"] = s2["Genotype"] + " | " + s2["Treatment"] + " | " + s2["Sex"]
            if "mean_speed_mm_s" in s2.columns:
                fig = go.Figure(go.Bar(x=s2["Group"], y=s2["mean_speed_mm_s"],
                                       error_y=dict(type="data", array=s2.get("mean_speed_mm_s_SEM")),
                                       marker_color="#27AE60"))
                fig.update_layout(yaxis_title="Mean crawling speed (mm/s)", height=400,
                                  margin=dict(l=40, r=20, t=20, b=90))
                st.plotly_chart(fig, width="stretch")
            st.dataframe(summ, width="stretch")
            st.download_button("Download summary CSV", summ.to_csv(index=False).encode(),
                               "larval_summary.csv", "text/csv", key="dl_lv")
    with t_cmp:
        opts = [c for c in ["mean_speed_mm_s", "path_length_mm", "tortuosity"]
                if c in per_larva.columns]
        if opts:
            m1, m2 = st.columns(2)
            metric = m1.selectbox("Metric", opts, key="lv_metric")
            gcol = m2.selectbox("Group by", ["Treatment", "Genotype", "Sex"], key="lv_gcol")
            if st.button("Run comparison", key="lv_cmp"):
                r = lv.compare_groups(per_larva, metric, gcol)
                if not r:
                    st.warning("Need >= 2 groups with >= 2 larvae.")
                else:
                    st.markdown(f"**Method:** `{r.get('method','')}`")
                    pw = r.get("pairwise", pd.DataFrame())
                    if isinstance(pw, pd.DataFrame) and not pw.empty:
                        st.dataframe(pw, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Grooming
# ══════════════════════════════════════════════════════════════════════════════
def page_grooming():
    render_header("Grooming",
                  "Grooming behaviour — % time grooming, bouts, group statistics")
    try:
        from modules import grooming as gr
    except Exception as _e:
        st.error(f"Grooming module import failed: {_e}")
        return

    st.markdown(
        "Quantifies scored grooming intervals (from manual scoring or an "
        "automated classifier). Input: **bout-level** "
        "(`Genotype, Treatment, Sex, FlyID, Bout_start_s, Bout_end_s, "
        "Observation_s`) **or per-fly summary** (`..., Total_grooming_s, "
        "Observation_s`).")
    _template_button("grooming_template.csv", key="tpl_gr")

    src = st.radio("Data source", ["Synthetic demo (bout-level)",
                                   "Upload CSV / Excel"], horizontal=True, key="gr_src")
    raw = None
    if src.startswith("Synthetic"):
        raw = gr.demo_grooming()
        st.caption("Synthetic bout-level data calibrated to wild-type ~13% time "
                   "grooming (Qiao et al. 2018, eLife); TDP43 grooms less, drug "
                   "partially restores. Bouts ~3-6 s.")
    else:
        up = st.file_uploader("Grooming data", type=["csv", "xlsx", "xls"], key="gr_up")
        if up is not None:
            raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    if raw is None or raw.empty:
        st.info("Load or upload grooming data to begin.")
        return

    with st.expander("Preview input", expanded=False):
        st.dataframe(raw.head(20), width="stretch")
    try:
        res = gr.run_grooming_analysis(raw)
    except Exception as _e:
        st.error(f"Could not analyse grooming data: {_e}")
        return
    pf = res["per_fly"]
    if pf.empty:
        st.warning("No usable grooming records after parsing.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Flies", len(pf))
    c2.metric("Mean % time grooming", round(float(pf["Pct_time_grooming"].mean()), 1))

    import plotly.graph_objects as go
    t_sum, t_cmp = st.tabs(["Group Summary", "Group Comparison"])
    with t_sum:
        summ = res["summary"]
        s2 = summ.copy(); s2["Group"] = s2["Genotype"] + " | " + s2["Treatment"] + " | " + s2["Sex"]
        fig = go.Figure(go.Bar(x=s2["Group"], y=s2["Pct_time_grooming"],
                               error_y=dict(type="data", array=s2.get("Pct_time_grooming_SEM")),
                               marker_color="#8E44AD"))
        fig.update_layout(yaxis_title="% time grooming", height=400,
                          margin=dict(l=40, r=20, t=20, b=90))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summ, width="stretch")
        st.download_button("Download grooming summary CSV", summ.to_csv(index=False).encode(),
                           "grooming_summary.csv", "text/csv", key="dl_gr")
    with t_cmp:
        m1, m2 = st.columns(2)
        metric = m1.selectbox("Metric", ["Pct_time_grooming", "Total_grooming_s",
                                         "N_bouts", "Mean_bout_s"], key="gr_metric")
        gcol = m2.selectbox("Group by", ["Treatment", "Genotype", "Sex"], key="gr_gcol")
        if st.button("Run comparison", key="gr_cmp"):
            r = gr.compare_groups(pf, metric, gcol)
            if not r:
                st.warning("Need >= 2 groups with >= 2 flies.")
            else:
                st.markdown(f"**Method:** `{r.get('method','')}`")
                pw = r.get("pairwise", pd.DataFrame())
                if isinstance(pw, pd.DataFrame) and not pw.empty:
                    st.dataframe(pw, width="stretch")
    st.divider()
    with st.expander("Per-fly grooming table", expanded=False):
        st.dataframe(pf, width="stretch")
        st.download_button("Download per-fly CSV", pf.to_csv(index=False).encode(),
                           "grooming_per_fly.csv", "text/csv", key="dl_grpf")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Drug Ranking (cross-strain, cross-assay compound ranking)
# ══════════════════════════════════════════════════════════════════════════════
def page_drug_ranking():
    render_header("Drug Ranking",
                  "Cross-strain, cross-assay compound ranking with unequal-N handling")
    try:
        from modules import drug_ranking as dr
        from modules import synthetic as syn
    except Exception as _e:
        st.error(f"Drug-ranking module import failed: {_e}")
        return

    st.markdown(
        "Quantifies each compound's effect **per strain** (Hedges' *g*, "
        "drug vs vehicle — corrected for unequal N), pools across strains by "
        "**inverse-variance / DerSimonian-Laird random-effects** meta-analysis, "
        "then combines assays into one **Therapeutic Score** and ranks compounds. "
        "Positive = beneficial (higher climbing/alternation, longer life, "
        "circadian period restored toward 24 h).")

    src = st.radio("Data source", ["Synthetic multi-compound screen",
                                   "Upload assay files"], horizontal=True, key="dr_src")
    data = {}
    vehicle = "Vehicle"
    if src.startswith("Synthetic"):
        if st.session_state.get("_screen") is None:
            with st.spinner("Generating multi-compound screen across 4 assays..."):
                st.session_state["_screen"] = syn.generate_screen()
        data = st.session_state["_screen"]
        st.caption("Synthetic screen: Vehicle + Drug_A/B/C of decreasing efficacy "
                   "on TDP43 (rho 0.80 / 0.45 / 0.05), Oregon-WT + TDP43, both "
                   "sexes, deliberately unequal N (TDP43 smaller).")
    else:
        st.caption("Upload one file per assay (any subset). Each needs Genotype, "
                   "Treatment, Sex columns plus that assay's data columns.")
        cc = st.columns(4)
        up_c = cc[0].file_uploader("Climbing", type=["csv", "xlsx"], key="dr_c")
        up_l = cc[1].file_uploader("Lifespan", type=["csv", "xlsx"], key="dr_l")
        up_y = cc[2].file_uploader("Y-maze", type=["csv", "xlsx"], key="dr_y")
        up_d = cc[3].file_uploader("DAMS", type=["csv", "xlsx"], key="dr_d")

        def _rd(u):
            if u is None:
                return None
            return pd.read_csv(u) if u.name.lower().endswith(".csv") else pd.read_excel(u)
        for key, u in [("climbing", up_c), ("lifespan", up_l), ("ymaze", up_y), ("dams", up_d)]:
            try:
                df = _rd(u)
                if df is not None and not df.empty:
                    data[key] = df
            except Exception as _e:
                st.warning(f"{key}: {_e}")
        if not data:
            st.info("Upload at least one assay file to begin.")
            return
        all_treats = set()
        for df in data.values():
            if "Treatment" in df.columns:
                all_treats |= set(df["Treatment"].unique())
        vehicle = st.selectbox("Vehicle / control treatment", sorted(all_treats))

    sig = (src, vehicle, tuple(sorted(data.keys())),
           tuple(int(len(v)) for v in data.values()))
    if st.session_state.get("_dr_sig") != sig:
        with st.spinner("Ranking compounds (effect sizes + meta-analysis)..."):
            try:
                st.session_state["_dr_res"] = dr.rank_compounds(data, vehicle=vehicle)
                st.session_state["_dr_sig"] = sig
            except Exception as _e:
                st.error(f"Ranking failed: {_e}")
                return
    res = st.session_state.get("_dr_res", {})
    comp = res.get("composite", pd.DataFrame())
    if comp is None or comp.empty:
        st.warning("No rankable compounds found (need a vehicle plus >=1 drug "
                   "with >=2 strata).")
        return

    import plotly.graph_objects as go

    t_rank, t_assay, t_strat, t_n = st.tabs(
        ["Ranking", "Per-Assay Effects", "Per-Stratum Effects", "Unequal-N Report"])

    with t_rank:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp["drug"], y=comp["Therapeutic_Score"],
            error_y=dict(type="data",
                         array=comp["ci_hi"] - comp["Therapeutic_Score"],
                         arrayminus=comp["Therapeutic_Score"] - comp["ci_lo"]),
            marker_color=["#27AE60" if p < 0.05 else "#95A5A6"
                          for p in comp["p_value"]],
            text=[f"#{r}" for r in comp["Rank"]], textposition="outside"))
        fig.add_hline(y=0, line_color="#0D2137")
        fig.update_layout(yaxis_title="Therapeutic Score (SMD, benefit)",
                          height=420, margin=dict(l=40, r=20, t=30, b=60))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(comp, width="stretch")
        st.caption("Therapeutic Score = inverse-variance composite of per-assay "
                   "pooled effects (95% CI). Green = significant (p<0.05). "
                   "`Score_equal_assay` weights each assay equally. p-values "
                   "two-sided from the composite z.")
        st.download_button("Download ranking CSV", comp.to_csv(index=False).encode(),
                           "drug_ranking.csv", "text/csv", key="dl_rank")

    with t_assay:
        pa = res.get("per_assay", pd.DataFrame())
        if pa.empty:
            st.info("No per-assay effects.")
        else:
            # forest-style: pooled g per drug x assay with CI
            pa2 = pa.copy()
            pa2["label"] = pa2["drug"] + " — " + pa2["assay"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pa2["g_pooled"], y=pa2["label"], mode="markers",
                error_x=dict(type="data", array=pa2["ci_hi"] - pa2["g_pooled"],
                             arrayminus=pa2["g_pooled"] - pa2["ci_lo"]),
                marker=dict(size=9, color="#2A7DE1")))
            fig.add_vline(x=0, line_color="#C0392B", line_dash="dash")
            fig.update_layout(xaxis_title="Pooled Hedges' g (benefit)", height=460,
                              margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pa, width="stretch")
            st.caption("Per-drug, per-assay effect pooled across strains x sex "
                       "(DerSimonian-Laird random effects). I2 = heterogeneity "
                       "across strains (does the drug act consistently?).")
            st.download_button("Download per-assay CSV", pa.to_csv(index=False).encode(),
                               "drug_per_assay.csv", "text/csv", key="dl_pa")

    with t_strat:
        ps = res.get("per_stratum", pd.DataFrame())
        if ps.empty:
            st.info("No per-stratum effects.")
        else:
            st.dataframe(ps, width="stretch")
            st.caption("Every Genotype x Sex x assay drug-vs-vehicle effect "
                       "(Hedges' g with unequal-N variance, n per group shown).")
            st.download_button("Download per-stratum CSV", ps.to_csv(index=False).encode(),
                               "drug_per_stratum.csv", "text/csv", key="dl_ps")

    with t_n:
        imb = res.get("imbalance", pd.DataFrame())
        un = res.get("unequal_n", pd.DataFrame())
        if not imb.empty:
            st.markdown("**Group-size balance per assay**")
            st.dataframe(imb, width="stretch")
            if (imb["flag"].str.contains("unequal")).any():
                st.warning("Unequal group sizes detected. Effects use Hedges' g "
                           "(unequal-N-corrected) and strains are combined by "
                           "inverse-variance weighting, so larger strains "
                           "contribute proportionally more — the statistically "
                           "correct handling of unbalanced designs.")
            else:
                st.success("Group sizes are approximately balanced.")
        if not un.empty:
            st.dataframe(un, width="stretch")
            st.download_button("Download sample-size CSV", un.to_csv(index=False).encode(),
                               "drug_sample_sizes.csv", "text/csv", key="dl_un")


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers — design-aware pages
# ══════════════════════════════════════════════════════════════════════════════
def _needs_columns(df: pd.DataFrame, needed: dict, why: str) -> bool:
    """
    True when every column in `needed` is present.

    Otherwise renders a single st.info naming exactly which columns are missing
    and what each one is for, so a page degrades into an instruction rather than
    a traceback.
    """
    absent = [c for c in needed if c not in df.columns]
    if not absent:
        return True
    lines = "\n".join(f"- `{c}` — {needed[c]}" for c in absent)
    st.info(f"{why}\n\nMissing column(s):\n{lines}")
    return False


def _factor_columns(df: pd.DataFrame, max_levels: int = 12) -> list:
    """Categorical columns with >= 2 levels, canonical design factors first."""
    pref = ["Treatment", "Genotype", "Sex", "Compound", "Batch", "ExperimentID"]
    out = [c for c in pref
           if c in df.columns and 2 <= int(df[c].nunique(dropna=True)) <= 200]
    for c in df.columns:
        if c in out or c in ("Count", "n"):
            continue
        try:
            nu = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= nu <= max_levels:
            out.append(c)
    return out


def _cluster_columns(df: pd.DataFrame) -> list:
    """Candidate replication units, most specific first (VialID, then Replicate)."""
    return [c for c in ("VialID", "FlyID", "Replicate", "Batch", "ExperimentID")
            if c in df.columns and df[c].notna().any()]


def _climb_proportion(df: pd.DataFrame) -> Optional[pd.Series]:
    """Count/n as a 0-1 proportion, or None when the counts are absent."""
    if "Count" not in df.columns or "n" not in df.columns:
        return None
    c = pd.to_numeric(df["Count"], errors="coerce")
    n = pd.to_numeric(df["n"], errors="coerce").replace(0, np.nan)
    return (c / n).clip(0.0, 1.0)


_PROP_OPTION = "Climbing proportion (Count / n)"

def _value_options(df: pd.DataFrame) -> list:
    """Selectable response columns, with the derived climbing proportion first."""
    opts = []
    if "Count" in df.columns and "n" in df.columns:
        opts.append(_PROP_OPTION)
    skip = {"Count", "n", "Replicate", "SessionIndex", "TrialIndex"}
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            opts.append(c)
    return opts


def _value_series(df: pd.DataFrame, option: str) -> pd.Series:
    if option == _PROP_OPTION:
        return _climb_proportion(df)
    return pd.to_numeric(df[option], errors="coerce")


def _parse_day_list(text: str) -> list:
    """Parse a comma/space separated list of ages into floats, silently dropping junk."""
    out = []
    for tok in str(text).replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return sorted(set(out))


def _spec_frame(spec_dict: dict) -> pd.DataFrame:
    rows = []
    for k, v in spec_dict.items():
        if isinstance(v, (list, tuple)):
            v = f"{len(v)} item(s)" if v else "none"
        rows.append({"property": k, "value": str(v)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Design Check
# ══════════════════════════════════════════════════════════════════════════════
def page_plan_run():
    render_header("Plan a Run",
                  "Allocate flies to activity-monitor channels BEFORE loading "
                  "them — the only moment a confound is still fixable")
    try:
        from modules import dam_planner as DPL
    except Exception as _e:
        st.error(f"Run-planner module import failed: {_e}")
        return

    st.markdown(
        "Every other page in this tool tells you what your data can answer. "
        "This one decides it. If an entire age group sits in one monitor, then "
        "monitor and age are the same column of numbers, and no model can "
        "separate a beam-sensitivity difference from an ageing effect — that is "
        "arithmetic, not a modelling choice. The layout below splits every "
        "factor across every monitor, interleaves the groups across channel "
        "positions rather than blocking them, and rotates the arrangement so a "
        "position effect cannot repeat identically in the next experiment.")

    tab_plan, tab_check, tab_proto = st.tabs(
        ["Plan a new run", "Check a layout I already have", "Protocol"])

    with tab_plan:
        c1, c2 = st.columns(2)
        with c1:
            genos = st.text_input("Genotypes (comma-separated)",
                                  "Oregon-R, TDP43", key="pl_g")
            sexes = st.text_input("Sexes", "Female, Male", key="pl_s")
            ages = st.text_input("Ages", "young, old", key="pl_a")
            treats = st.text_input("Treatments", "Untreated", key="pl_t")
        with c2:
            n_per = st.number_input("Flies per group", 1, 64, 8, key="pl_n",
                                    help="Eight tubes per group is the practical "
                                         "minimum; fewer and a single dead fly "
                                         "moves the group mean.")
            n_mon = st.number_input("Monitors available", 1, 16, 2, key="pl_m")
            n_ch = st.number_input("Channels per monitor", 4, 64, 32, key="pl_c")
            goal = st.selectbox("What is being measured", ["sleep", "period"],
                                key="pl_goal",
                                help="Free-running period requires constant "
                                     "darkness; sleep and activity do not.")
            seed = st.number_input("Randomisation seed", 0, 10_000, 0, key="pl_seed",
                                   help="Change it between experiments so the "
                                        "same tube never sits in the same "
                                        "position twice.")

        def _split(s):
            return [x.strip() for x in str(s).split(",") if x.strip()] or [""]

        groups = [{"Genotype": g, "Sex": sx, "Age": ag, "Treatment": tr}
                  for g in _split(genos) for sx in _split(sexes)
                  for ag in _split(ages) for tr in _split(treats)]
        st.caption(f"{len(groups)} groups x {int(n_per)} flies = "
                   f"{len(groups) * int(n_per)} tubes; capacity is "
                   f"{int(n_mon) * int(n_ch)}.")

        if st.button("Plan the layout", type="primary", key="pl_go"):
            plan = DPL.plan_run(groups, n_per_group=int(n_per),
                                n_monitors=int(n_mon), n_channels=int(n_ch),
                                goal=goal, seed=int(seed))
            if not plan.get("ok"):
                st.error(plan.get("reason", "The run could not be planned."))
            else:
                st.session_state["dam_plan"] = plan

        plan = st.session_state.get("dam_plan")
        if plan:
            v = plan["verification"]
            for f, status in v["per_factor"].items():
                (st.success if status == "crossed" else st.error)(
                    f"**{f}** is {status} with Monitor"
                    + ("" if status == "crossed" else
                       " — this layout cannot answer that question"))
            for w in plan.get("warnings", []):
                st.warning(w)
            st.markdown("**Run advice**")
            for a in plan.get("advice", []):
                st.markdown(f"- {a}")

            cmap = plan["channel_map"]
            st.dataframe(cmap, width="stretch", height=320)
            st.download_button(
                "Download channel map (CSV)", cmap.to_csv(index=False),
                "dam_channel_map.csv", "text/csv", key="pl_dl",
                help="Load the tubes to match this file, then upload the same "
                     "file with the monitor data so nothing has to be remembered.")

    with tab_check:
        st.markdown(
            "Already loaded the monitors? Upload the channel map and this will "
            "say which questions the run can answer — before you spend days "
            "recording.")
        up = st.file_uploader("Channel map CSV", type=["csv"], key="pl_up")
        if up is not None:
            try:
                cm = pd.read_csv(up)
                v = DPL.check_plan(cm)
            except Exception as _e:
                st.error(f"Could not check that layout: {_e}")
            else:
                for f, status in v["per_factor"].items():
                    (st.success if status == "crossed" else st.error)(
                        f"**{f}** is {status} with Monitor")
                for b in v.get("blocking", []):
                    st.error(b)
                for w in v.get("warnings", []):
                    st.warning(w)
                if v["ok"]:
                    st.success("This layout is usable.")

    with tab_proto:
        st.code(DPL.protocol_text(), language=None)


def page_design_check():
    render_header("Design Check",
                  "Pre-analysis design audit — replication unit, confounding, "
                  "blocking defects")
    try:
        from modules import design_schema as dsch
    except Exception as _e:
        st.error(f"Design-schema module import failed: {_e}")
        return
    if not _require_data():
        return
    df = st.session_state["df"]

    st.markdown(
        "Reads the design the data actually encode — which physical **vial** or "
        "**fly** each row came from, which time axes exist, which compound and "
        "dose — and names the **unit of independent replication** that every "
        "downstream model must cluster on. Factor pairs that cannot be "
        "separated are reported as **blocking**: no model can estimate them, so "
        "the analysis stops rather than returning a number that looks fine.")
    _template_button("climbing_longitudinal_template.csv",
                     "Extended-design template", "tpl_dc")

    lights_on = st.number_input(
        "Lights-on hour (ZT0) on the local clock", min_value=0.0, max_value=23.5,
        value=8.0, step=0.5, key="dc_lights",
        help="Used only to convert clock times, or morning/afternoon/evening "
             "labels, into Zeitgeber hours.")

    try:
        rep = dsch.design_report(df, lights_on_hour=float(lights_on))
    except Exception as _e:
        st.error(f"Design report failed: {_e}")
        return
    spec = rep["spec"]

    blocking = rep.get("blocking", [])
    if blocking:
        st.error("BLOCKING DESIGN DEFECT. The effects below are mathematically "
                 "inseparable — no amount of data or modelling can estimate "
                 "them separately. The design must change; the analysis must "
                 "not proceed as if it could.")
        for b in blocking:
            st.error(b)

    st.markdown(f"**Design:** {rep.get('design_summary', 'n/a')}")

    m = st.columns(4)
    m[0].metric("Replication unit", str(rep.get("cluster_unit") or "not identified"))
    _mfc = spec.mean_flies_per_cluster
    m[1].metric("Mean flies per cluster",
                f"{_mfc:.1f}" if pd.notna(_mfc) else "N/A")
    m[2].metric("Ages", str(spec.n_ages))
    m[3].metric("Experiments", str(spec.n_experiments))

    if rep.get("cluster_unit") is None:
        st.warning(
            "No replication unit could be identified. Add a `VialID` column "
            "(one label per physical vial) — without it, flies in the same vial "
            "are indistinguishable from independent replicates and every "
            "standard error on this dataset is too small.")

    t_warn, t_conf, t_spec = st.tabs(
        ["Warnings and Notes", "Confounding", "Design Spec"])

    with t_warn:
        warns = rep.get("warnings", [])
        notes = rep.get("notes", [])
        if warns:
            st.markdown("**Warnings**")
            for w in warns:
                st.warning(w)
        else:
            st.success("No design warnings.")
        if notes:
            st.markdown("**Notes**")
            for nt in notes:
                st.info(nt)

    with t_conf:
        confounds = rep.get("confounds", [])
        if not confounds:
            st.success("No design factor is confounded with a nuisance "
                       "variable (time of day, batch, experiment, date, "
                       "session index, rack position).")
        else:
            for c in confounds:
                sev = float(c.get("severity", 0.0) or 0.0)
                txt = f"{c.get('factor')} x {c.get('nuisance')}: {c.get('verdict', '')}"
                if sev >= 1.0:
                    st.error(txt)
                else:
                    st.warning(txt)
            ctab = pd.DataFrame([
                {"factor": c.get("factor"), "nuisance": c.get("nuisance"),
                 "severity": c.get("severity"), "cramers_v": c.get("cramers_v")}
                for c in confounds
            ])
            st.dataframe(ctab, width="stretch")
            st.download_button("Download confounding CSV",
                               ctab.to_csv(index=False).encode(),
                               "design_confounds.csv", "text/csv", key="dl_dc_conf")
            st.caption("severity = fraction of factor levels occurring at only "
                       "one level of the nuisance variable; 1.0 is complete "
                       "confounding and is blocking.")

    with t_spec:
        sf = _spec_frame(spec.to_dict())
        st.dataframe(sf, width="stretch")
        st.download_button("Download design spec CSV", sf.to_csv(index=False).encode(),
                           "design_spec.csv", "text/csv", key="dl_dc_spec")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Hierarchical Statistics
# ══════════════════════════════════════════════════════════════════════════════
def page_hierarchical():
    render_header("Hierarchical Statistics",
                  "Clustered counts — ICC, design effect, beta-binomial and GEE")
    try:
        from modules import hierarchical as hier
    except Exception as _e:
        st.error(f"Hierarchical module import failed: {_e}")
        return
    if not _require_data():
        return
    df = st.session_state["df"]

    st.markdown(
        "Flies in a vial share food, the tap that startles them, temperature and "
        "handling history, so the **vial** is the experimental unit. Treating "
        "*m* flies per vial as *m* independent observations inflates the "
        "effective sample size by the design effect `1 + (m - 1) * ICC`; with "
        "*m* = 20 and ICC = 0.05 the standard errors are about 40% too small. "
        "This page runs the naive fly-as-independent analysis and the "
        "cluster-correct analysis side by side on your own data.")
    _template_button("climbing_longitudinal_template.csv",
                     "Extended-design template", "tpl_hier")

    if not _needs_columns(df, {
        "Count": "flies that crossed the line — the binomial successes",
        "n": "flies tested in that vial — the binomial trials",
    }, "This page models climbing as clustered binomial counts, so it needs "
       "the raw counts rather than a pre-computed performance index."):
        return

    cats = _factor_columns(df)
    clus = _cluster_columns(df)
    if not cats:
        st.info("No categorical column with at least two levels was found. Add "
                "`Treatment`, `Genotype` or `Sex` so there is something to compare.")
        return
    if not clus:
        st.info(
            "No cluster identifier was found. Add a `VialID` column (one label "
            "per physical vial) so the analysis can cluster on the experimental "
            "unit. `Replicate` is accepted as a fallback when it identifies vials.")
        return

    c1, c2 = st.columns(2)
    group_col = c1.selectbox(
        "Grouping column (the diagnostic compares exactly two levels)",
        cats, key="hier_grp")
    cluster_col = c2.selectbox("Cluster (replication) unit", clus, index=0,
                               key="hier_clu")

    try:
        rep = hier.pseudoreplication_report(df, group_col=group_col,
                                            cluster_col=cluster_col)
    except Exception as _e:
        st.error(f"Pseudoreplication diagnostic failed: {_e}")
        return

    if not rep.get("usable"):
        st.warning(rep.get("reason", "The diagnostic could not be run."))
    else:
        if rep.get("conclusion_changes"):
            st.error(
                "THE CONCLUSION CHANGES WHEN THE DESIGN IS RESPECTED. "
                f"Treating flies as independent gives p = {rep['p_naive']:.3g}; "
                f"clustering on {cluster_col} gives p = {rep['p_clustered']:.3g}. "
                "These fall on opposite sides of 0.05. The clustered result is "
                "the valid one — the naive p-value is an artefact of counting "
                "flies as replicates of a treatment that was applied to vials.")

        m = st.columns(4)
        _icc = rep.get("icc", np.nan)
        _deff = rep.get("design_effect", np.nan)
        m[0].metric("ICC (intra-vial)", f"{_icc:.3f}" if pd.notna(_icc) else "N/A")
        m[1].metric("Design effect", f"{_deff:.2f}" if pd.notna(_deff) else "N/A")
        m[2].metric("Flies", f"{rep.get('n_flies', float('nan')):.0f}")
        _neff = rep.get("n_effective", np.nan)
        m[3].metric("Effective sample size",
                    f"{_neff:.0f}" if pd.notna(_neff) else "N/A")

        p1, p2 = st.columns(2)
        _pn, _pc = rep.get("p_naive", np.nan), rep.get("p_clustered", np.nan)
        p1.metric("p — flies treated as independent (naive)",
                  f"{_pn:.4g}" if pd.notna(_pn) else "N/A")
        p2.metric(f"p — clustered on {cluster_col}",
                  f"{_pc:.4g}" if pd.notna(_pc) else "N/A")

        st.markdown(f"**Verdict:** {rep.get('verdict', '')}")

        detail = pd.DataFrame([{
            "levels_compared": " vs ".join(rep.get("levels", [])),
            "n_clusters": rep.get("n_clusters"),
            "mean_flies_per_cluster": rep.get("mean_flies_per_cluster"),
            "icc_anova": rep.get("icc"),
            "icc_betabinomial": rep.get("icc_betabinomial"),
            "se_inflation": rep.get("se_inflation"),
            "p_naive": rep.get("p_naive"),
            "p_clustered": rep.get("p_clustered"),
            "p_ratio": rep.get("p_ratio"),
        }])
        with st.expander("Full diagnostic table", expanded=False):
            st.dataframe(detail, width="stretch")
            st.caption(rep.get("gee_note", ""))
            st.download_button("Download diagnostic CSV",
                               detail.to_csv(index=False).encode(),
                               "pseudoreplication.csv", "text/csv", key="dl_hier_diag")

    st.divider()
    st.markdown("#### Model fits")
    rhs_default = group_col if group_col not in ("Genotype",) else "Genotype"
    rhs = st.text_input(
        "Model right-hand side (patsy syntax)", value=rhs_default, key="hier_rhs",
        help="For example `Treatment`, `Genotype + Treatment`, or "
             "`Genotype * Treatment + Age`.")

    t_bb, t_gee = st.tabs(["Beta-binomial", "GEE (cluster-robust)"])

    with t_bb:
        st.caption("Maximum-likelihood beta-binomial regression with a logit "
                   "link. The binomial mean structure plus one extra parameter "
                   "rho, which is exactly the intraclass correlation — "
                   "overdispersion is estimated rather than assumed away.")
        if st.button("Fit beta-binomial", key="hier_bb_run"):
            try:
                bb = hier.fit_betabinomial(df, formula_rhs=rhs)
            except Exception as _e:
                st.error(f"Beta-binomial fit failed: {_e}")
            else:
                b1, b2, b3 = st.columns(3)
                b1.metric("rho (intra-cluster)", f"{bb.rho:.4f}")
                b2.metric("AIC", f"{bb.aic:.1f}")
                b3.metric("Converged", "yes" if bb.converged else "no")
                if not bb.converged:
                    st.warning(f"Optimiser did not converge: {bb.message}")
                sf = bb.summary_frame()
                st.dataframe(sf, width="stretch")
                st.download_button("Download beta-binomial CSV",
                                   sf.to_csv().encode(),
                                   "betabinomial_coefficients.csv", "text/csv",
                                   key="dl_hier_bb")
                st.caption("Coefficients are on the logit scale; odds ratios and "
                           "their Wald intervals are exponentiated alongside.")

    with t_gee:
        st.caption("Population-averaged logistic GEE with cluster-robust "
                   "(sandwich) standard errors. No distributional assumption "
                   "beyond the mean model, so it is an independent cross-check "
                   "on the beta-binomial fit.")
        if st.button("Fit GEE", key="hier_gee_run"):
            try:
                gee = hier.fit_gee(df, formula_rhs=rhs, cluster_col=cluster_col)
            except Exception as _e:
                st.error(f"GEE fit failed: {_e}")
            else:
                g1, g2, g3 = st.columns(3)
                g1.metric("Clusters", str(gee.get("n_clusters", "N/A")))
                g2.metric("Bernoulli rows", str(gee.get("n_obs", "N/A")))
                _wc = gee.get("working_correlation", np.nan)
                g3.metric("Working correlation",
                          f"{_wc:.4f}" if pd.notna(_wc) else "N/A")
                tab = gee["table"]
                st.dataframe(tab, width="stretch")
                st.download_button("Download GEE CSV", tab.to_csv().encode(),
                                   "gee_coefficients.csv", "text/csv",
                                   key="dl_hier_gee")
                st.caption(gee.get("method", ""))

    st.caption(
        "Clustering does not always widen the interval. For a BETWEEN-cluster "
        "predictor (a treatment applied to whole vials) the naive standard "
        "error is too small and the corrected interval is wider. For a "
        "WITHIN-cluster predictor (age, when every vial is measured at every "
        "age) the vial random intercept absorbs between-vial variation and the "
        "corrected interval is narrower — respecting the design buys precision.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Survivor Bias
# ══════════════════════════════════════════════════════════════════════════════
def page_survivor_bias():
    render_header("Survivor Bias",
                  "Trajectory estimands — survivor mean, balanced panel, "
                  "pattern mixture")
    try:
        from modules import survivor_bias as sb
    except Exception as _e:
        st.error(f"Survivor-bias module import failed: {_e}")
        return

    st.markdown(
        "In an ageing cohort the weakest animals die first, so a phenotype "
        "measured at age *t* is measured in a progressively fitter "
        "subpopulation. The cohort-mean curve — what almost every published "
        "climbing figure plots — is the trajectory of a **survivor**, not of a "
        "fly, and its decline is biased toward zero. The bias is larger in the "
        "sicker genotype, so a manipulation that kills frail flies early can "
        "look protective. This page computes three estimands on the same data "
        "and shows how far apart they are.")
    _template_button("climbing_longitudinal_template.csv",
                     "Longitudinal template (has DeathAge)", "tpl_sb")

    src = st.radio("Data source",
                   ["Loaded data", "Frailty simulation demo"],
                   horizontal=True, key="sb_src")

    id_col, age_col, val_col, death_col = "FlyID", "Age", "Value", "DeathAge"

    if src == "Frailty simulation demo":
        st.caption(
            "Each simulated fly has its own decline slope; steeper decliners "
            "die earlier, with the strength of that coupling set below. At "
            "coupling 0 there is no survivor bias and the three estimands "
            "agree. Raise it and watch the cohort mean flatten while the "
            "individual-level slope does not.")
        d1, d2, d3 = st.columns(3)
        coupling = d1.slider("Frailty coupling", 0.0, 2.0, 1.0, 0.1, key="sb_coupling",
                             help="0 = decline rate and death age are "
                                  "independent; 1 = strong coupling.")
        n_flies = d2.slider("Flies", 60, 600, 300, 20, key="sb_nflies")
        seed = d3.number_input("Seed", 0, 9999, 0, 1, key="sb_seed")
        try:
            work = sb.simulate_frailty_cohort(
                n_flies=int(n_flies), frailty_coupling=float(coupling),
                seed=int(seed))
        except Exception as _e:
            st.error(f"Simulation failed: {_e}")
            return
        st.caption(f"Simulated {work['FlyID'].nunique()} flies, "
                   f"{len(work)} observations, true mean slope "
                   f"{work['TrueSlope'].mean():+.5f}/day.")
    else:
        if not _require_data():
            return
        df = st.session_state["df"]
        if not _needs_columns(df, {
            "Age": "age in days, the time axis of the trajectory",
        }, "This page compares trajectory estimands across ages within animals."):
            return
        id_opts = [c for c in ("FlyID", "VialID") if c in df.columns
                   and df[c].notna().any()]
        if not id_opts:
            st.info(
                "This page follows the same animal across ages, so it needs an "
                "identifier.\n\nMissing column(s):\n"
                "- `FlyID` — the individual animal; without it there is no "
                "individual trajectory, only a cohort mean\n"
                "- `VialID` — accepted as a coarser unit when flies are not "
                "tracked individually, in which case the 'individual' slope is "
                "a per-vial slope and terminal decline within a vial is averaged "
                "away")
            return
        vopts = _value_options(df)
        if not vopts:
            st.info("No numeric response column was found. Add `Count` and `n` "
                    "(climbing counts), or any numeric phenotype column.")
            return
        v1, v2 = st.columns(2)
        val_opt = v1.selectbox("Value column (the phenotype)", vopts, key="sb_val")
        id_col = v2.selectbox("Animal identifier", id_opts, key="sb_id")
        if id_col != "FlyID":
            st.warning(
                f"Using `{id_col}` as the unit. Trajectories are then per-vial "
                "means, so the individual-level slope is a vial-level slope and "
                "within-vial terminal decline is averaged away. Record `FlyID` "
                "to recover the individual estimand.")
        work = df.copy()
        work["Value"] = _value_series(df, val_opt)
        if "DeathAge" not in work.columns:
            st.warning(
                "No `DeathAge` column. Informative-dropout testing, "
                "pattern-mixture reweighting and time-to-death alignment are "
                "unavailable. Recording the age at death of every animal is the "
                "single most valuable extra column for trajectory work.")

    try:
        rep = sb.survivor_bias_report(work, id_col=id_col, age_col=age_col,
                                      value_col=val_col, death_col=death_col)
    except Exception as _e:
        st.error(f"Survivor-bias report failed: {_e}")
        return

    ratio = rep.get("slope_bias_ratio", np.nan)
    if pd.notna(ratio) and ratio < 0.85:
        st.error(rep.get("verdict", ""))
    elif pd.notna(ratio):
        st.info(rep.get("verdict", ""))
    else:
        st.warning(rep.get("verdict", ""))

    m = st.columns(4)
    m[0].metric("Slope bias ratio", f"{ratio:.2f}" if pd.notna(ratio) else "N/A",
                help="Cohort-mean slope divided by the individual-level "
                     "(mixed-model) slope. 0.60 means the conventional analysis "
                     "reports a decline only 60% as steep as the flies experienced.")
    for lbl, key, col in (("Survivor mean", "slope_survivor_mean", m[1]),
                          ("Balanced panel", "slope_balanced_panel", m[2]),
                          ("Mixed model", "slope_mixed_model", m[3])):
        v = rep.get(key, np.nan)
        col.metric(f"{lbl} slope /day", f"{v:+.5f}" if pd.notna(v) else "N/A")

    for w in rep.get("warnings", []):
        st.warning(w)

    t_traj, t_slopes, t_frail, t_axis = st.tabs(
        ["Trajectories", "Slope Table", "Informative Dropout", "Time Axis"])

    with t_traj:
        import plotly.graph_objects as go
        fig = go.Figure()
        series = [
            ("Survivor mean (conventional)", rep.get("survivor_mean"), "#C0392B"),
            ("Balanced panel", rep.get("balanced_panel"), "#0D2137"),
            ("Pattern mixture", rep.get("pattern_mixture"), "#F0A500"),
        ]
        plotted = 0
        for name, tab, colour in series:
            if tab is None or not isinstance(tab, pd.DataFrame) or tab.empty:
                continue
            if age_col not in tab.columns or "mean" not in tab.columns:
                continue
            fig.add_trace(go.Scatter(
                x=tab[age_col], y=tab["mean"], mode="lines+markers", name=name,
                line=dict(color=colour, width=2)))
            plotted += 1
        if plotted == 0:
            st.info("No estimand could be computed from these data.")
        else:
            fig.update_layout(xaxis_title="Age (days)", yaxis_title="Phenotype",
                              height=440,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02),
                              margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Survivor mean = E[Y(t) | alive at t], the conventional curve. "
                "Balanced panel = a fixed cohort surviving to a common age, so "
                "selection is constant across ages. Pattern mixture = the "
                "full-cohort mean, reweighting death-time strata back to the "
                "original composition (Little 1993). Where they diverge, the "
                "cohort-mean decline constant does not describe an individual.")
        pm_meta = rep.get("pattern_mixture_meta", {})
        if pm_meta.get("available"):
            st.warning("Pattern-mixture identifying assumption: "
                       + pm_meta.get("identifying_assumption", ""))
        bp_meta = rep.get("balanced_panel_meta", {})
        if bp_meta.get("available"):
            st.caption(
                f"Balanced panel restricted to age {bp_meta.get('to_age'):g}: "
                f"{bp_meta.get('n_survivors')} of {bp_meta.get('n_total_animals')} "
                f"animals retained ({bp_meta.get('retention', 0):.0%}).")
        elif bp_meta:
            st.caption(f"Balanced panel unavailable: {bp_meta.get('reason', '')}")

    with t_slopes:
        rows = [
            {"estimand": "E[Y(t) | alive at t] (survivor-conditional)",
             "slope_per_day": rep.get("slope_survivor_mean")},
            {"estimand": "E[Y(t) | alive at common age] (balanced panel)",
             "slope_per_day": rep.get("slope_balanced_panel")},
            {"estimand": "E[Y(t)] over the full starting cohort (pattern mixture)",
             "slope_per_day": rep.get("slope_pattern_mixture")},
            {"estimand": "individual-level mean slope (mixed model)",
             "slope_per_day": rep.get("slope_mixed_model")},
        ]
        stab = pd.DataFrame(rows)
        st.dataframe(stab, width="stretch")
        st.download_button("Download slope CSV", stab.to_csv(index=False).encode(),
                           "survivor_bias_slopes.csv", "text/csv", key="dl_sb_slopes")
        mm = rep.get("mixed_model", {})
        if mm.get("available"):
            st.caption(
                f"Mixed model: slope {mm['slope']:+.5f} "
                f"[{mm['ci_lower']:+.5f}, {mm['ci_upper']:+.5f}], "
                f"p = {mm['p_value']:.3g}, {mm['n_animals']} animals, "
                f"{mm['n_obs']} observations.")
        elif mm:
            st.caption(f"Mixed model unavailable: {mm.get('reason', '')}")

    with t_frail:
        ft = rep.get("frailty_test")
        if not ft:
            st.info("Requires a `DeathAge` column: the test asks whether an "
                    "animal's own trajectory predicts when it dies.")
        elif not ft.get("available"):
            st.warning(ft.get("reason", "Unavailable."))
        else:
            if ft.get("informative_dropout"):
                st.error(ft.get("verdict", ""))
            else:
                st.success(ft.get("verdict", ""))
            fr = pd.DataFrame([{
                "rho_slope_vs_death": ft.get("rho_slope_vs_death"),
                "p_slope_vs_death": ft.get("p_slope_vs_death"),
                "rho_level_vs_death": ft.get("rho_level_vs_death"),
                "p_level_vs_death": ft.get("p_level_vs_death"),
                "n_animals": ft.get("n_animals"),
            }])
            st.dataframe(fr, width="stretch")
            st.caption(ft.get("rawmean_note", ""))

    with t_axis:
        ax = rep.get("axis_comparison")
        if not ax:
            st.info("Requires a `DeathAge` column: the comparison asks whether "
                    "chronological age or time-to-death explains the phenotype "
                    "better.")
        elif not ax.get("available"):
            st.warning(ax.get("reason", "Unavailable."))
        else:
            if ax.get("better_axis") == "time_to_death":
                st.error(ax.get("verdict", ""))
            else:
                st.info(ax.get("verdict", ""))
            atab = pd.DataFrame(ax.get("axes", {})).T
            st.dataframe(atab, width="stretch")
            st.caption(
                "Motor collapse in flies is largely a terminal event "
                "phase-locked to death (Gaitanidis et al. 2019, Aging "
                "11:1850-1873). When time-to-death wins, genotypes with "
                "different lifespans must not be compared at matched "
                "chronological ages.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Trajectory Forecast
# ══════════════════════════════════════════════════════════════════════════════
def _traj_cohorts(df: pd.DataFrame, group_cols: list, value: pd.Series,
                  age_col: str = "Age") -> dict:
    """Collapse a frame into {cohort label: (ages, mean values)}."""
    work = df.copy()
    work["_val"] = value
    work = work.dropna(subset=[age_col, "_val"])
    if work.empty:
        return {}
    cohorts = {}
    if group_cols:
        grouped = work.groupby(group_cols, dropna=False)
    else:
        work["_all"] = "all"
        grouped = work.groupby("_all", dropna=False)
    for key, g in grouped:
        label = " | ".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        agg = g.groupby(age_col)["_val"].mean().sort_index()
        if len(agg) >= 2:
            cohorts[label] = (agg.index.to_numpy(float), agg.to_numpy(float))
    return cohorts


def page_trajectory_forecast():
    render_header("Trajectory Forecast",
                  "Hierarchical disease-trajectory prediction with honest intervals")
    try:
        from modules import trajectory as traj
    except Exception as _e:
        st.error(f"Trajectory module import failed: {_e}")
        return
    if not _require_data():
        return
    df = st.session_state["df"]

    st.markdown(
        "Fitting a curve to three points and extrapolating to day 40 is an "
        "exercise in guessing the curve's **shape**. The fix is to borrow that "
        "shape from cohorts that have been followed to late ages: completed "
        "cohorts define a population prior over the curve parameters, and a "
        "sparse new cohort's posterior is a precision-weighted compromise "
        "between its own data and that prior. Predictions beyond a configurable "
        "horizon are **refused**, not merely flagged.")
    _template_button("climbing_aging_template.csv", "Ageing-series template", "tpl_tf")

    if not _needs_columns(df, {
        "Age": "age in days — the axis the trajectory is predicted along",
    }, "This page forecasts a phenotype forward in age, so it needs an age axis "
       "with at least two ages per cohort."):
        return

    vopts = _value_options(df)
    if not vopts:
        st.info("No numeric response column was found. Add `Count` and `n` "
                "(climbing counts), or any numeric phenotype column.")
        return

    gopts = [c for c in ("Genotype", "Treatment", "Sex", "Compound")
             if c in df.columns]
    c1, c2 = st.columns([2, 3])
    val_opt = c1.selectbox("Response column", vopts, key="tf_val")
    group_cols = c2.multiselect(
        "Cohorts are defined by", gopts,
        default=[c for c in ("Genotype", "Treatment") if c in gopts],
        key="tf_grp")

    value = _value_series(df, val_opt)
    if value is None or not value.notna().any():
        st.info("The selected response column has no usable values.")
        return
    lo_v, hi_v = float(np.nanmin(value)), float(np.nanmax(value))
    if not (0.0 <= lo_v and hi_v <= 1.0):
        st.warning(
            f"The response ranges {lo_v:.3g} to {hi_v:.3g}. The curve families "
            "are defined on a 0-1 scale; values outside it will be clipped. "
            "Use the climbing proportion, or rescale the column, for a "
            "meaningful fit.")

    cohorts = _traj_cohorts(df, group_cols, value)
    if len(cohorts) < 1:
        st.info("No cohort has at least two distinct ages. A trajectory cannot "
                "be fitted to a single timepoint.")
        return

    st.caption(f"{len(cohorts)} cohort(s) found: "
               + ", ".join(f"{k} ({len(v[0])} ages)" for k, v in list(cohorts.items())[:8])
               + (" ..." if len(cohorts) > 8 else ""))

    f1, f2, f3 = st.columns(3)
    family = f1.selectbox("Curve family for the population prior",
                          list(traj.FAMILIES.keys()), key="tf_fam")
    min_points = f2.slider("Minimum ages for a cohort to enter the prior",
                           2, 8, 4, 1, key="tf_minpts")
    max_horizon = f3.slider("Refuse beyond this multiple of the observed span",
                            0.5, 4.0, 2.0, 0.5, key="tf_hor")

    use_conformal = st.checkbox(
        "Calibrate intervals conformally on held-out cohorts", value=False,
        key="tf_conf",
        help="Splits the cohorts: half build the prior, half supply the "
             "out-of-sample residuals used to correct the interval width. "
             "Needs at least four cohorts.")

    labels = sorted(cohorts.keys())
    target = st.selectbox("Cohort to forecast", labels, key="tf_target")

    obs_t, obs_y = cohorts[target]
    prior_pool = {k: v for k, v in cohorts.items() if k != target}

    calib_pool = {}
    if use_conformal and len(prior_pool) >= 4:
        pk = sorted(prior_pool.keys())
        calib_pool = {k: prior_pool[k] for k in pk[1::2]}
        prior_pool = {k: prior_pool[k] for k in pk[0::2]}
    elif use_conformal:
        st.warning(
            f"Conformal calibration needs at least four cohorts other than the "
            f"target; {len(prior_pool)} available. Intervals will be reported "
            "uncalibrated.")

    prior = None
    if prior_pool:
        try:
            prior = traj.build_population_prior(prior_pool, family=family,
                                                min_points=int(min_points))
        except Exception as _e:
            st.warning(f"Population prior could not be built: {_e}")
    if prior is None:
        st.warning(
            "No population prior. Fewer than two completed cohorts qualified "
            f"(each needs at least {int(min_points)} ages and a stable "
            f"{family} fit). The forecast falls back to AICc-weighted model "
            "averaging across curve families — the spread across families is "
            "the model-form uncertainty, and it is wide, as it should be."
            + (f" Conformal calibration held {len(calib_pool)} cohort(s) out of "
               f"the prior, leaving only {len(prior_pool)}; turn it off, or add "
               "more completed cohorts." if calib_pool else ""))
    else:
        if calib_pool:
            st.caption(
                f"Split-conformal: {len(prior_pool)} cohort(s) build the prior, "
                f"{len(calib_pool)} supply out-of-sample residuals. The two sets "
                "must be disjoint — overlapping ones give in-sample residuals "
                "and an under-correction.")
        pm = st.columns(3)
        pm[0].metric("Prior cohorts", str(prior.n_cohorts))
        pm[1].metric("Prior family", prior.family)
        pm[2].metric("Residual SD", f"{prior.residual_sd:.4f}")
        for nt in prior.notes:
            st.caption(nt)

    calibrator = None
    if calib_pool and prior is not None:
        tmax_cal = float(np.median([np.median(v[0]) for v in calib_pool.values()]))
        try:
            calibrator = traj.build_conformal_calibrator(
                prior, calib_pool, train_max_day=tmax_cal)
        except Exception as _e:
            st.warning(f"Conformal calibration failed: {_e}")
        if calibrator is None:
            st.warning("Conformal calibration produced no usable residuals; "
                       "intervals are uncalibrated.")

    default_targets = ", ".join(
        str(int(round(d))) for d in np.unique(np.concatenate([
            obs_t, obs_t.max() + np.array([5.0, 10.0, 20.0])])))
    tgt_text = st.text_input("Target ages (days)", value=default_targets,
                             key="tf_targets")
    targets = _parse_day_list(tgt_text)
    if not targets:
        st.info("Enter at least one target age, for example `10, 20, 30, 40`.")
        return

    try:
        pred = traj.predict_trajectory(
            obs_t, obs_y, targets, prior=prior, max_horizon=float(max_horizon),
            calibrator=calibrator)
    except Exception as _e:
        st.error(f"Prediction failed: {_e}")
        return

    ptab = pd.DataFrame(pred["predictions"])
    refused = ptab[ptab["refused"]]
    if not refused.empty:
        st.error(
            f"{len(refused)} target age(s) REFUSED: "
            + "; ".join(f"day {int(r['day'])}" for _, r in refused.iterrows())
            + ". " + str(refused.iloc[0]["refusal_reason"]))

    ok = ptab[~ptab["refused"]]
    n_cal = int(ok["calibrated"].sum()) if not ok.empty else 0
    if calibrator is not None and n_cal:
        st.success(f"{n_cal} of {len(ok)} issued intervals are conformally "
                   "calibrated against held-out cohorts.")
    else:
        st.info(
            "Intervals are NOT conformally calibrated. Empirical-Bayes "
            "intervals treat the estimated prior as if it were known and are "
            "systematically too narrow at long horizons; read them as "
            "optimistic.")

    st.markdown(f"**Method:** {pred.get('method', '')}")

    t_plot, t_pred, t_model, t_end, t_val = st.tabs(
        ["Forecast", "Predictions", "Models", "Endpoints", "Forward Validation"])

    with t_plot:
        import plotly.graph_objects as go
        fig = go.Figure()
        if not ok.empty:
            xs = ok["day"].tolist()
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1],
                y=ok["ci_upper"].tolist() + ok["ci_lower"].tolist()[::-1],
                fill="toself", fillcolor="rgba(240,165,0,0.18)",
                line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=xs, y=ok["predicted"], mode="lines+markers", name="Predicted",
                line=dict(color="#F0A500", width=2)))
        fig.add_trace(go.Scatter(
            x=obs_t, y=obs_y, mode="markers", name="Observed",
            marker=dict(color="#0D2137", size=9)))
        fig.update_layout(
            xaxis_title="Age (days)", yaxis_title="Phenotype", height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Cohort `{target}`; observed range "
                   f"{pred['observed_range_days'][0]:g}-"
                   f"{pred['observed_range_days'][1]:g} days from "
                   f"{pred['n_observed']} points. Band = 95% predictive interval.")

    with t_pred:
        st.dataframe(ptab, width="stretch")
        st.download_button("Download predictions CSV",
                           ptab.to_csv(index=False).encode(),
                           "trajectory_predictions.csv", "text/csv", key="dl_tf_pred")
        st.caption("`regime` says whether a target is interpolated, just beyond "
                   "the data, or extrapolated; `horizon_x_span` is how far "
                   "beyond the last observation it sits, in multiples of the "
                   "observed span.")

    with t_model:
        mtab = pd.DataFrame(pred.get("models", []))
        if mtab.empty:
            st.info("No model table available.")
        else:
            st.dataframe(mtab, width="stretch")
            st.download_button("Download model CSV", mtab.to_csv(index=False).encode(),
                               "trajectory_models.csv", "text/csv", key="dl_tf_mod")
            st.caption("Without a prior, `weight` is the AICc model-averaging "
                       "weight; the spread across families is the model-form "
                       "uncertainty and it is not hidden.")

    with t_end:
        thr = st.slider("Impairment threshold", 0.0, 1.0, 0.5, 0.05, key="tf_thr")
        try:
            ep = traj.derive_endpoints(pred, threshold=float(thr))
        except Exception as _e:
            st.error(f"Endpoint derivation failed: {_e}")
            ep = {"available": False}
        if not ep.get("available"):
            st.info(ep.get("reason", "Endpoints unavailable."))
        else:
            e1, e2 = st.columns(2)
            _oa = ep.get("onset_age_days", np.nan)
            _hs = ep.get("healthspan_days", np.nan)
            e1.metric("Onset age (days)",
                      f"{_oa:.1f}" if pd.notna(_oa) else "not reached",
                      help="First predicted age at which the phenotype falls "
                           "below the threshold.")
            e2.metric("Healthspan (days)",
                      f"{_hs:.1f}" if pd.notna(_hs) else "N/A",
                      help="Predicted days at or above the threshold within the "
                           "prediction window.")
            etab = pd.DataFrame([{k: v for k, v in ep.items()
                                  if not isinstance(v, (list, dict))}])
            st.dataframe(etab, width="stretch")
            for k in ("onset_age_days_note", "healthspan_days_note"):
                if ep.get(k):
                    st.warning(ep[k])
            st.caption("Every quantity is computed within each posterior draw "
                       "and then summarised, so the interval propagates "
                       "trajectory uncertainty rather than being read off the "
                       "median curve.")

    with t_val:
        st.caption(
            "Validation is FORWARD IN TIME only. A random hold-out across a "
            "cohort's own timepoints leaks the future into training and "
            "flatters the model several fold.")
        if len(obs_t) < 3:
            st.info("Forward validation needs at least three observed ages "
                    "(two to train on and one later one to score).")
        else:
            tmax = st.select_slider(
                "Train on ages up to", options=[float(v) for v in obs_t[:-1]],
                value=float(obs_t[len(obs_t) // 2]), key="tf_tmax")
            if st.button("Run forward validation", key="tf_fv"):
                try:
                    fv = traj.forward_validate(obs_t, obs_y, train_max_day=float(tmax),
                                               prior=prior, calibrator=calibrator)
                except Exception as _e:
                    st.error(f"Forward validation failed: {_e}")
                else:
                    ftab = pd.DataFrame([{k: v for k, v in fv.items()
                                          if isinstance(v, (int, float, str, bool))}])
                    st.dataframe(ftab, width="stretch")
                    for k, v in fv.items():
                        if isinstance(v, pd.DataFrame) and not v.empty:
                            st.markdown(f"**{k}**")
                            st.dataframe(v, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Historical Controls
# ══════════════════════════════════════════════════════════════════════════════
def page_historical_controls():
    render_header("Historical Controls",
                  "Robust meta-analytic-predictive priors — borrow control "
                  "information, safely")
    try:
        from modules import control_prior as cp
    except Exception as _e:
        st.error(f"Control-prior module import failed: {_e}")
        return

    st.markdown(
        "A laboratory that has run the same wild-type control twenty times "
        "already knows how that control behaves. Borrowing it lets the same "
        "flies test more compounds, or the same experiment run with fewer "
        "flies. The danger is that historical controls are **not** exchangeable "
        "by fiat — stocks drift, food batches change, incubators are replaced — "
        "and naive pooling is among the fastest ways to manufacture a false "
        "discovery. The robust MAP prior switches its own borrowing off when "
        "the concurrent control disagrees with history.")
    _template_button("historical_controls_template.csv",
                     "Historical-controls template", "tpl_hc")

    df = st.session_state.get("df")
    has_expt = (df is not None and not df.empty
                and "ExperimentID" in df.columns
                and int(df["ExperimentID"].nunique(dropna=True)) >= 3)

    modes = ["Manual entry"]
    if has_expt:
        modes.insert(0, "From loaded data (ExperimentID)")
    else:
        st.info(
            "To build the prior from your own data, load a dataset with an "
            "`ExperimentID` column carrying at least three experiments "
            "(two or more historical control arms plus the current one), a "
            "`Treatment` column naming the control, and `Count` / `n` climbing "
            "counts. Manual entry is available below in the meantime.")
    mode = st.radio("Historical arms from", modes, horizontal=True, key="hc_mode")

    arms = None
    if mode.startswith("From loaded"):
        a1, a2, a3 = st.columns(3)
        ctrl = a1.selectbox("Control treatment",
                            sorted(df["Treatment"].dropna().astype(str).unique()),
                            key="hc_ctrl")
        geno_opts = ["(all)"] + (sorted(df["Genotype"].dropna().astype(str).unique())
                                 if "Genotype" in df.columns else [])
        geno = a2.selectbox("Genotype", geno_opts, key="hc_geno")
        sex_opts = ["(all)"] + (sorted(df["Sex"].dropna().astype(str).unique())
                                if "Sex" in df.columns else [])
        sex = a3.selectbox("Sex", sex_opts, key="hc_sex")
        try:
            arms = cp.historical_arms_from_df(
                df, control_treatment=ctrl,
                genotype=None if geno == "(all)" else geno,
                sex=None if sex == "(all)" else sex)
        except Exception as _e:
            st.warning(f"Could not extract control arms: {_e}")
            arms = None
        if arms is not None and len(arms) < 3:
            st.warning(
                f"Only {len(arms)} control arm(s) matched. At least three are "
                "needed: two or more historical arms to estimate the "
                "between-experiment variance, plus the current one.")
            arms = None

    if arms is None:
        st.markdown("**Historical control arms** — one row per past experiment.")
        seed_tab = pd.DataFrame({
            "label": [f"EXP{i}" for i in range(1, 6)],
            "successes": [17, 18, 16, 19, 17],
            "totals": [20, 20, 20, 20, 20],
        })
        edited = st.data_editor(seed_tab, num_rows="dynamic", width="stretch",
                                key="hc_editor")
        arms = edited.rename(columns={"label": "ExperimentID"}).copy()
        arms = arms.dropna(subset=["successes", "totals"])
        arms["successes"] = pd.to_numeric(arms["successes"], errors="coerce")
        arms["totals"] = pd.to_numeric(arms["totals"], errors="coerce")
        arms = arms.dropna(subset=["successes", "totals"])
        arms = arms[arms["totals"] > 0]
        arms["proportion"] = arms["successes"] / arms["totals"]
        cur1, cur2 = st.columns(2)
        cur_s = cur1.number_input("Concurrent control — flies that climbed",
                                  min_value=0, value=15, step=1, key="hc_cur_s")
        cur_n = cur2.number_input("Concurrent control — flies tested",
                                  min_value=1, value=20, step=1, key="hc_cur_n")
        hist = arms
        cur_label = "concurrent"
    else:
        id_col = arms.columns[0]
        st.dataframe(arms, width="stretch")
        cur_label = st.selectbox(
            "Which experiment is the CURRENT (concurrent) control?",
            [str(v) for v in arms[id_col]], index=len(arms) - 1, key="hc_cur")
        cur_row = arms[arms[id_col].astype(str) == cur_label].iloc[0]
        cur_s, cur_n = float(cur_row["successes"]), float(cur_row["totals"])
        hist = arms[arms[id_col].astype(str) != cur_label]

    if hist is None or len(hist) < 2:
        st.info("At least two historical control arms are required. A "
                "between-experiment variance cannot be estimated from fewer, "
                "and a prior built from one arm is that arm asserted as truth.")
        return

    w = st.slider("Robustification weight on the MAP component", 0.1, 1.0, 0.8, 0.05,
                  key="hc_w",
                  help="The analyst's pre-specified statement of how much the "
                       "historical arms are believed exchangeable with the new "
                       "one. The remainder goes on a deliberately vague "
                       "component that absorbs a conflicting control.")

    try:
        built = cp.build_map_prior(hist["successes"].to_numpy(float),
                                   hist["totals"].to_numpy(float),
                                   labels=[str(v) for v in hist.iloc[:, 0]])
    except Exception as _e:
        st.error(f"MAP prior could not be built: {_e}")
        return

    prior = built["prior"]
    robust = cp.robustify(prior, weight=float(w))
    ess = cp.prior_effective_sample_size(robust)
    summ = built["prior_summary"]

    conflict = cp.prior_data_conflict(robust, float(cur_s), float(cur_n))
    if conflict.get("conflict"):
        st.error("PRIOR-DATA CONFLICT — DO NOT BORROW. " + conflict.get("verdict", ""))
    else:
        st.success(conflict.get("verdict", ""))

    m = st.columns(4)
    m[0].metric("Historical arms", str(built["n_historical_arms"]))
    m[1].metric("Prior control rate", f"{summ['mean_proportion']:.1%}")
    m[2].metric("Prior worth (flies)", f"{ess['ess_flies']:.0f}",
                help="Effective sample size of the informative component, in "
                     "control flies. This is the number to show a reviewer or "
                     "an ethics committee.")
    m[3].metric("Concurrent control", f"{cur_s / max(cur_n, 1):.1%}")

    for nt in built.get("notes", []):
        st.warning(nt)

    t_prior, t_post, t_design, t_oc = st.tabs(
        ["Prior", "Posterior", "Recommended Design", "Operating Characteristics"])

    with t_prior:
        ptab = pd.DataFrame([{
            "n_historical_arms": built["n_historical_arms"],
            "prior_mean_proportion": summ["mean_proportion"],
            "prior_ci_lower": summ["ci_lower"],
            "prior_ci_upper": summ["ci_upper"],
            "tau_mean_logit": built["tau_mean"],
            "heterogeneity_fraction": built["heterogeneity_fraction"],
            "tau_prior": built["tau_prior"],
            "ess_flies_map_component": ess["ess_flies_map_component"],
            "ess_flies_full_mixture": ess["ess_flies_full_mixture"],
        }])
        st.dataframe(ptab, width="stretch")
        st.download_button("Download prior summary CSV",
                           ptab.to_csv(index=False).encode(),
                           "map_prior_summary.csv", "text/csv", key="dl_hc_prior")
        atab = hist.copy()
        st.markdown("**Contributing historical arms**")
        st.dataframe(atab, width="stretch")
        st.caption(built.get("method", ""))
        st.caption(ess.get("caveat", ""))

    with t_post:
        post = cp.update_posterior(robust, float(cur_s), float(cur_n))
        psum = post.on_probability_scale()
        q = st.columns(3)
        q[0].metric("Posterior control rate", f"{psum['mean_proportion']:.1%}")
        q[1].metric("95% lower", f"{psum['ci_lower']:.1%}")
        q[2].metric("95% upper", f"{psum['ci_upper']:.1%}")
        ctab = pd.DataFrame([{
            "conflict": conflict.get("conflict"),
            "conflict_p_value": conflict.get("p_value"),
            "observed_proportion": conflict.get("observed_proportion"),
            "prior_predicted_proportion": conflict.get("prior_predicted_proportion"),
            "predictive_lower": conflict.get("predictive_interval", (np.nan, np.nan))[0],
            "predictive_upper": conflict.get("predictive_interval", (np.nan, np.nan))[1],
        }])
        st.dataframe(ctab, width="stretch")
        st.caption(conflict.get("method", "") +
                   ". The weakly-informative component is stripped before the "
                   "test — it is designed to accommodate any observation, so "
                   "leaving it in would make the diagnostic unable to detect a "
                   "conflict.")

    with t_design:
        st.caption(
            "Finds the smallest concurrent control arm whose borrowed analysis "
            "keeps type I error inside the limit across the stated drift. A "
            "concurrent control is never eliminated: without one, drift is "
            "undetectable and the conflict check cannot run.")
        d1, d2, d3 = st.columns(3)
        n_std = d1.number_input("Standard control arm (flies)", 10, 500, 60, 5,
                                key="hc_nstd")
        n_trt = d2.number_input("Treatment arm (flies)", 10, 500, 60, 5, key="hc_ntrt")
        drift = d3.slider("Tolerable drift in control rate", 0.01, 0.20, 0.05, 0.01,
                          key="hc_drift")
        max_t1 = st.slider("Maximum acceptable type I error", 0.05, 0.20, 0.075, 0.005,
                           key="hc_t1")
        if st.button("Search for a reduced control arm", key="hc_design_run"):
            with st.spinner("Simulating the borrowed analysis across drift..."):
                try:
                    rd = cp.recommended_design(
                        hist["successes"].to_numpy(float),
                        hist["totals"].to_numpy(float),
                        n_control_standard=int(n_std), n_treat=int(n_trt),
                        max_type_I=float(max_t1), tolerable_drift=float(drift),
                        robust_weight=float(w), n_sim=800)
                except Exception as _e:
                    st.error(f"Design search failed: {_e}")
                else:
                    if rd.get("recommended_n_control"):
                        st.success(rd.get("verdict", ""))
                    else:
                        st.error(rd.get("verdict", ""))
                    r1, r2 = st.columns(2)
                    r1.metric("Recommended control arm",
                              str(rd.get("recommended_n_control") or "none"))
                    r2.metric("Flies saved per experiment",
                              str(rd.get("flies_saved_per_experiment", 0)))
                    stab = rd.get("search_table", pd.DataFrame())
                    if not stab.empty:
                        st.dataframe(stab, width="stretch")
                        st.download_button("Download search CSV",
                                           stab.to_csv(index=False).encode(),
                                           "control_design_search.csv", "text/csv",
                                           key="dl_hc_design")

    with t_oc:
        st.caption(
            "Whether borrowing is acceptable is not a matter of opinion; it is "
            "the false-positive rate it produces at the drift a laboratory "
            "actually experiences. Concurrent-only, robust MAP and full MAP are "
            "compared at every drift level.")
        o1, o2 = st.columns(2)
        n_c = o1.number_input("Concurrent control arm (flies)", 10, 500, 30, 5,
                              key="hc_oc_nc")
        n_t = o2.number_input("Treatment arm (flies)", 10, 500, 60, 5, key="hc_oc_nt")
        if st.button("Simulate operating characteristics", key="hc_oc_run"):
            p_hist = float(summ["mean_proportion"])
            grid = [float(np.clip(p_hist + d, 0.02, 0.98))
                    for d in (-0.10, -0.05, 0.0, 0.05, 0.10)]
            with st.spinner("Simulating..."):
                try:
                    oc = cp.operating_characteristics(
                        hist["successes"].to_numpy(float),
                        hist["totals"].to_numpy(float),
                        n_control=int(n_c), n_treat=int(n_t),
                        p_control_true_grid=grid, true_effect_logit=0.0,
                        robust_weight=float(w), n_sim=800)
                except Exception as _e:
                    st.error(f"Simulation failed: {_e}")
                else:
                    st.dataframe(oc, width="stretch")
                    st.download_button("Download operating-characteristics CSV",
                                       oc.to_csv(index=False).encode(),
                                       "control_operating_characteristics.csv",
                                       "text/csv", key="dl_hc_oc")
                    st.caption("With no true effect, `rejection_rate` IS the "
                               "type I error at that amount of drift.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Dose-Response
# ══════════════════════════════════════════════════════════════════════════════
def page_dose_response():
    render_header("Dose-Response",
                  "Log-logistic potency, profile-likelihood EC50, and lethality "
                  "separation")
    try:
        from modules import dose_response as dr
    except Exception as _e:
        st.error(f"Dose-response module import failed: {_e}")
        return
    if not _require_data():
        return
    df = st.session_state["df"]

    st.markdown(
        "A one-way ANOVA across dose groups answers \"significant at 5 mM, not "
        "at 1 mM\", which depends entirely on where the doses happened to be "
        "placed. A curve fit estimates **potency** (EC50) and **shape** (the "
        "Hill slope) with intervals that are comparable across experiments. "
        "EC50 intervals are computed by **profile likelihood**, because the "
        "delta method assumes a symmetry the estimate does not have and can "
        "return a negative lower bound — an impossible dose.")
    _template_button("toxin_dose_response_template.csv", "Toxin template", "tpl_dr2")

    if not _needs_columns(df, {
        "Dose": "numeric dose or concentration, including 0 for untreated "
                "controls (they pin down the upper asymptote and are never dropped)",
    }, "This page fits a dose-response curve, so it needs a numeric dose axis "
       "with several distinct levels."):
        return

    if "Compound" in df.columns and df["Compound"].nunique(dropna=True) > 1:
        comp = st.selectbox("Compound",
                            sorted(df["Compound"].dropna().astype(str).unique()),
                            key="dr2_comp")
        work = df[df["Compound"].astype(str) == comp].copy()
    else:
        work = df.copy()

    vopts = _value_options(work)
    if not vopts:
        st.info("No numeric response column was found. Add `Count` and `n` "
                "(climbing counts), or any numeric response column.")
        return

    c1, c2 = st.columns(2)
    val_opt = c1.selectbox("Response column", vopts, key="dr2_val")
    model = c2.selectbox("Model", ["LL.4", "LL.3", "LL.2"], key="dr2_model",
                         help="LL.4 = b, c, d, e free. LL.3 fixes the lower "
                              "asymptote at 0. LL.2 also fixes the upper at 1 — "
                              "the natural model for a survival curve, where e "
                              "is exactly the LC50.")

    dose = pd.to_numeric(work["Dose"], errors="coerce")
    resp = _value_series(work, val_opt)
    wt = pd.to_numeric(work["n"], errors="coerce") if (
        val_opt == _PROP_OPTION and "n" in work.columns) else None

    keep = dose.notna() & resp.notna()
    if wt is not None:
        keep &= wt.notna() & (wt > 0)
    dose, resp = dose[keep].to_numpy(float), resp[keep].to_numpy(float)
    wt = wt[keep].to_numpy(float) if wt is not None else None

    if dose.size == 0:
        st.info("No rows have both a dose and a response.")
        return

    try:
        chk = dr.check_design(dose, resp, model=model)
    except Exception as _e:
        st.error(f"Design check failed: {_e}")
        return

    for r in chk.get("refusals", []):
        st.error(r)
    for wmsg in chk.get("warnings", []):
        st.warning(wmsg)

    force = False
    if not chk.get("usable"):
        st.info(
            f"The design cannot identify {model}: a {chk['n_params']}-parameter "
            f"curve needs at least that many distinct non-zero doses, and there "
            f"are {chk['n_distinct_nonzero_doses']}. Choose a simpler model, or "
            "add dose levels.")
        force = st.checkbox(
            "Fit anyway (standard errors will not be trustworthy)", value=False,
            key="dr2_force")
        if not force:
            return

    try:
        fit = dr.fit_loglogistic(dose, resp, n=wt, model=model, force=bool(force))
    except Exception as _e:
        st.error(f"Curve fit failed: {_e}")
        return

    try:
        ci = dr.ec50_profile_ci(fit)
    except Exception as _e:
        st.warning(f"Profile interval unavailable: {_e}")
        ci = None

    m = st.columns(4)
    m[0].metric("EC50", f"{fit.ec50:.4g}")
    if ci:
        m[1].metric("Profile 95% lower", f"{ci['profile']['lower']:.4g}")
        m[2].metric("Profile 95% upper", f"{ci['profile']['upper']:.4g}")
    m[3].metric("Hill slope b", f"{float(fit.params['b']):.3f}")
    if ci:
        st.markdown(f"**Interval:** {ci.get('note', '')}")
    if chk.get("ec50_extrapolated"):
        st.warning("The doses do not straddle the curve's inflection, so EC50 is "
                   "EXTRAPOLATED beyond the tested range rather than "
                   "interpolated. It is a property of the assumed shape, not an "
                   "observation.")

    t_curve, t_par, t_ecx, t_horm, t_mort = st.tabs(
        ["Curve", "Parameters", "Effective Doses", "Hormesis", "Lethality"])

    with t_curve:
        import plotly.graph_objects as go
        nz = dose[dose > 0]
        if nz.size:
            grid = np.geomspace(max(nz.min() * 0.5, 1e-9), nz.max() * 1.5, 200)
        else:
            grid = np.linspace(0, max(dose.max(), 1.0), 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dose, y=resp, mode="markers", name="Observed",
                                 marker=dict(color="#0D2137", size=8)))
        fig.add_trace(go.Scatter(x=grid, y=fit.predict(grid), mode="lines",
                                 name=f"{model} fit",
                                 line=dict(color="#F0A500", width=2)))
        if np.isfinite(fit.ec50) and fit.ec50 > 0:
            fig.add_vline(x=float(fit.ec50), line=dict(color="#C0392B", dash="dash"))
        fig.update_layout(
            xaxis_title="Dose", yaxis_title="Response", height=440,
            xaxis_type="log" if nz.size else "linear",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Zero doses are evaluated by the model's limit, so untreated "
            "controls enter the fit as direct observations of the asymptote "
            "rather than being dropped. They are shown at the left edge of the "
            "log axis only if non-zero doses exist. Dashed line = EC50.")

    with t_par:
        sf = fit.summary_frame()
        st.dataframe(sf, width="stretch")
        st.download_button("Download parameter CSV", sf.to_csv().encode(),
                           "dose_response_parameters.csv", "text/csv", key="dl_dr2_par")
        q = st.columns(3)
        q[0].metric("Residual SE", f"{fit.resid_se:.4g}")
        q[1].metric("Residual df", str(fit.df_resid))
        q[2].metric("R-squared", f"{fit.r_squared():.3f}")
        st.caption("R-squared is descriptive only: a high value does not "
                   "validate a non-linear model and does not mean EC50 is well "
                   "determined.")

    with t_ecx:
        try:
            ed = dr.effective_dose(fit, levels=(10, 50, 90))
        except Exception as _e:
            st.warning(f"Effective doses unavailable: {_e}")
        else:
            st.dataframe(ed, width="stretch")
            st.download_button("Download ECx CSV", ed.to_csv().encode(),
                               "effective_doses.csv", "text/csv", key="dl_dr2_ecx")
            st.caption(ed.attrs.get("note", ""))

    with t_horm:
        st.caption(
            "Low-dose stimulation is common in toxicology and a monotone Hill "
            "fit cannot represent it: it splits the difference, pulling the "
            "upper asymptote up and the EC50 out. The Brain-Cousens model adds "
            "one parameter and is tested against LL.4 by an extra-sum-of-squares "
            "F test.")
        if st.button("Test for hormesis", key="dr2_horm_run"):
            try:
                hm = dr.hormesis_test(dose, resp, n=wt)
            except Exception as _e:
                st.error(f"Hormesis test failed: {_e}")
            else:
                if not hm.get("usable"):
                    st.info(hm.get("reason", "Not testable."))
                else:
                    if hm.get("hormesis_supported"):
                        st.warning(hm.get("verdict", ""))
                    else:
                        st.success(hm.get("verdict", ""))
                    htab = pd.DataFrame([{
                        "F": hm.get("f_statistic"), "p_value": hm.get("p_value"),
                        "f_param": hm.get("f_param"), "f_param_se": hm.get("f_param_se"),
                        "max_stimulation_dose": hm.get("max_stimulation_dose"),
                        "max_stimulation_percent": hm.get("max_stimulation_percent"),
                        "monotone_ec50": hm.get("monotone_ec50"),
                    }])
                    st.dataframe(htab, width="stretch")
                    st.caption(hm.get("caveat", ""))

    with t_mort:
        if "n_dead" not in work.columns:
            st.info(
                "Requires an `n_dead` column (flies dead at scoring) alongside "
                "`n` (flies exposed). Climbing is scored among flies that are "
                "alive to climb, so a toxin that kills flies removes them from "
                "the denominator: \"the toxin impairs climbing\" and \"the "
                "toxin was poisoning the flies and the survivors were "
                "moribund\" produce identical data. Without mortality counts "
                "the two cannot be told apart.")
        elif "n" not in work.columns:
            st.info("Requires an `n` column alongside `n_dead` so the exposed "
                    "and the scored denominators can both be formed.")
        else:
            dead = pd.to_numeric(work["n_dead"], errors="coerce")[keep].to_numpy(float)
            n_col = pd.to_numeric(work["n"], errors="coerce")[keep].to_numpy(float)
            meaning = st.selectbox(
                "The `n` column counts",
                ["flies scored after mortality — exposed = n + n_dead",
                 "flies originally exposed — scored = n - n_dead"],
                key="dr2_nmeaning",
                help="Which denominator `n` refers to changes the mortality "
                     "curve and therefore the LC50. It cannot be inferred from "
                     "the numbers, so it is asked rather than assumed.")
            if meaning.startswith("flies scored"):
                expo, scored = n_col + dead, n_col
            else:
                expo, scored = n_col, n_col - dead
            if np.any(dead > expo) or np.any(scored < 0):
                st.warning(
                    "Under this reading the number dead exceeds the number "
                    "exposed in at least one row, which is impossible. Switch "
                    "the setting above, or check that `n` and `n_dead` mean "
                    "what they are assumed to mean.")
            else:
                try:
                    ma = dr.mortality_adjusted_response(dose, resp, dead, expo,
                                                        n_scored=scored)
                except Exception as _e:
                    st.error(f"Mortality adjustment failed: {_e}")
                else:
                    sev = ma.get("severity", "")
                    if sev == "confounded":
                        st.error(ma.get("warning", ""))
                    elif sev in ("overlapping", "indeterminate"):
                        st.warning(ma.get("warning", ""))
                    else:
                        st.success(ma.get("warning", ""))
                    cols = st.columns(4)
                    for col, (lbl, key_) in zip(cols, (
                            ("EC50 (behaviour)", "ec50"),
                            ("LC50 (mortality)", "lc50"),
                            ("EC50 / LC50", "ratio"),
                            ("Max mortality", "max_mortality"))):
                        v = ma.get(key_, np.nan)
                        col.metric(lbl, f"{v:.4g}" if pd.notna(v) else "N/A")
                    pdz = ma.get("per_dose", pd.DataFrame())
                    if isinstance(pdz, pd.DataFrame) and not pdz.empty:
                        st.dataframe(pdz, width="stretch")
                        st.download_button("Download per-dose CSV",
                                           pdz.to_csv().encode(),
                                           "dose_mortality.csv", "text/csv",
                                           key="dl_dr2_mort")
                        st.caption("`survivor_conditional` marks doses at which "
                                   "mortality exceeds the threshold; at those "
                                   "doses the behavioural estimate is "
                                   "conditional on survival and cannot be read "
                                   "as a behavioural effect.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — Time-of-Day
# ══════════════════════════════════════════════════════════════════════════════
def page_time_of_day():
    render_header("Time-of-Day",
                  "Circadian modulation of climbing — cosinor, and the fatigue "
                  "confound")
    try:
        from modules import circadian_climbing as cc
    except Exception as _e:
        st.error(f"Circadian module import failed: {_e}")
        return
    if not _require_data():
        return
    df = st.session_state["df"]

    st.markdown(
        "Time-of-day modulation of climbing is a **hypothesis**, not an "
        "established fact: Gargano et al. (2005, Exp Gerontol 40:386-395) "
        "report that RING performance is not influenced by the time of day or "
        "by repeated testing, so a null here is the expected result. Cosinor "
        "spends the same two degrees of freedom as a three-level ANOVA but "
        "returns MESOR, amplitude and acrophase, which are comparable across "
        "studies. The decisive check comes first: if every fly is tested "
        "morning, then afternoon, then evening, time of day is perfectly "
        "collinear with testing order and no method can separate them.")
    _template_button("climbing_longitudinal_template.csv",
                     "Extended-design template", "tpl_tod")

    if not (("TimeOfDay" in df.columns and df["TimeOfDay"].notna().any())
            or ("ZT" in df.columns and df["ZT"].notna().any())):
        st.info(
            "This page needs a time-of-day axis.\n\nMissing column(s):\n"
            "- `TimeOfDay` — the clock time of the assay session (`09:10`, "
            "`14:05`) or a label (`Morning`, `Afternoon`, `Evening`)\n"
            "- `ZT` — Zeitgeber hours since lights-on, if you already have them\n\n"
            "Either one is sufficient; ZT is used directly, TimeOfDay is "
            "converted using the lights-on hour.")
        return
    if not _needs_columns(df, {
        "Count": "flies that crossed the line",
        "n": "flies tested in that vial",
    }, "The cosinor is fitted on the empirical logit of the climbing counts, "
       "which unbounds the ceiling and removes the mean-variance coupling of a "
       "raw proportion."):
        return

    subj_opts = _cluster_columns(df)
    if not subj_opts:
        st.info(
            "No unit of replication was found. Add a `VialID` column (or "
            "`FlyID` if flies are individually tracked and independently "
            "housed). A single subject measured at three times of day supports "
            "no cosinor at all — three observations, three parameters, zero "
            "residual degrees of freedom — so replication must come from the "
            "number of subjects.")
        return

    c1, c2, c3 = st.columns(3)
    subject_col = c1.selectbox("Subject (unit of replication)", subj_opts,
                               key="tod_subj")
    lights_on = c2.number_input("Lights-on hour (ZT0)", 0.0, 23.5, 8.0, 0.5,
                                key="tod_lights")
    gopts = ["(none)"] + [c for c in ("Genotype", "Treatment", "Sex")
                          if c in df.columns and df[c].nunique(dropna=True) >= 2]
    group_col = c3.selectbox("Group by", gopts, key="tod_group")

    work = df.copy()
    try:
        zt, is_cat, zt_warn = cc.zt_from_frame(work, lights_on_hour=float(lights_on))
    except Exception as _e:
        st.error(f"Zeitgeber time could not be derived: {_e}")
        return
    work["_ZT"] = zt
    for wmsg in zt_warn:
        st.warning(wmsg)
    if is_cat:
        st.warning(
            "Zeitgeber time was reconstructed from categorical labels using the "
            "conventional midpoints of the morning/afternoon/evening testing "
            "windows. The acrophase inherits the width of those windows and "
            "cannot be more precise than that. Record clock times to fix this.")

    try:
        resp, wts, scale_note = cc.prepare_climbing_response(work)
    except Exception as _e:
        st.error(f"Response could not be built: {_e}")
        return
    work["_Y"] = resp
    work["_W"] = wts
    st.caption(scale_note)

    n_zt = int(work["_ZT"].dropna().round(6).nunique())
    if n_zt < 3:
        st.info(
            f"Only {n_zt} distinct Zeitgeber hour(s) are present. A cosinor has "
            "three parameters, so at least three distinct times of day are "
            "required before amplitude and acrophase mean anything.")
        return

    order_opts = [c for c in ("SessionIndex", "TrialIndex") if c in work.columns
                  and work[c].notna().any()]

    attribution_allowed = True
    if order_opts:
        order_col = st.selectbox("Cumulative testing-order covariate", order_opts,
                                 key="tod_order")
        try:
            sep = cc.separate_fatigue_from_rhythm(
                work, zt_col="_ZT", value_col="_Y", order_col=order_col,
                subject_col=subject_col, weight_col="_W")
        except Exception as _e:
            st.warning(f"Fatigue/rhythm separation failed: {_e}")
            sep = {"available": False, "attribution_allowed": True}
        attribution_allowed = bool(sep.get("attribution_allowed", True))
        if not attribution_allowed:
            st.error(sep.get("verdict", "Time of day and testing order are not "
                                        "separable in this design."))
            if sep.get("recommendation"):
                st.error(sep["recommendation"])
        elif sep.get("available"):
            st.success(sep.get("verdict", ""))
        for wmsg in sep.get("warnings", []) or []:
            st.warning(wmsg)
    else:
        st.info(
            "No `SessionIndex` or `TrialIndex` column, so the fatigue/"
            "habituation confound cannot be checked. If every subject visits "
            "the times of day in the same order, a monotone drift caused by "
            "handling is numerically indistinguishable from a circadian trend. "
            "Record the cumulative testing order.")
        order_col = None

    t_pop, t_sub, t_amp, t_adv = st.tabs(
        ["Population Cosinor", "Per-subject Vectors", "Amplitude Biomarker",
         "Design Advisor"])

    pop = None
    with t_pop:
        try:
            pop = cc.population_cosinor(
                work, zt_col="_ZT", value_col="_Y", subject_col=subject_col,
                group_col=None if group_col == "(none)" else group_col,
                weight_col="_W", n_boot=400)
        except Exception as _e:
            st.error(f"Population cosinor failed: {_e}")
            pop = None
        if pop is not None and not pop.get("overall"):
            st.info("No subject had three or more distinct times of day, so no "
                    "population cosinor could be fitted.")
            for wmsg in pop.get("warnings", []):
                st.warning(wmsg)
        elif pop is not None:
            ov = pop.get("overall") or {}
            if ov.get("rhythm_detected") and attribution_allowed:
                st.success(ov.get("verdict", ""))
            elif ov.get("rhythm_detected"):
                st.warning(
                    "A rhythm-shaped effect was detected, but time of day is "
                    "not separable from testing order in this design, so it "
                    "must NOT be described as circadian. "
                    + ov.get("verdict", ""))
            else:
                st.info(ov.get("verdict", ov.get("reason", "No verdict.")))
            k = st.columns(4)
            for lbl, key_, col in (("MESOR (logit)", "mesor", k[0]),
                                   ("Amplitude (logit)", "amplitude", k[1]),
                                   ("Acrophase (ZT h)", "acrophase_h", k[2])):
                v = ov.get(key_, np.nan)
                col.metric(lbl, f"{v:.4g}" if pd.notna(v) else "N/A")
            k[3].metric("Subjects", str(ov.get("k_subjects", "N/A")))
            for wmsg in pop.get("warnings", []):
                st.warning(wmsg)
            for nt in pop.get("notes", []):
                st.caption(nt)
            byg = pop.get("by_group", {})
            if byg:
                gtab = pd.DataFrame([
                    {"group": g,
                     "mesor": v.get("mesor"), "amplitude": v.get("amplitude"),
                     "acrophase_h": v.get("acrophase_h"),
                     "rhythm_detected": v.get("rhythm_detected"),
                     "zero_amplitude_p": v.get("zero_amplitude_p")}
                    for g, v in byg.items()])
                st.dataframe(gtab, width="stretch")
                st.download_button("Download group cosinor CSV",
                                   gtab.to_csv(index=False).encode(),
                                   "cosinor_by_group.csv", "text/csv",
                                   key="dl_tod_grp")
            comps = pop.get("comparisons", [])
            if comps:
                ctab = pd.DataFrame([
                    {kk: vv for kk, vv in c.items()
                     if isinstance(vv, (int, float, str, bool))} for c in comps])
                with st.expander("Pairwise group comparisons", expanded=False):
                    st.dataframe(ctab, width="stretch")

    with t_sub:
        if pop is None:
            st.info("Run the population cosinor first.")
        else:
            ps = pop.get("per_subject", pd.DataFrame())
            if ps.empty:
                st.info("No subject had three or more distinct times of day, so "
                        "no rhythm vectors exist.")
            else:
                st.dataframe(ps, width="stretch")
                st.download_button("Download per-subject CSV",
                                   ps.to_csv(index=False).encode(),
                                   "cosinor_per_subject.csv", "text/csv",
                                   key="dl_tod_sub")
                st.caption(
                    "Each subject contributes one bivariate observation "
                    "(beta, gamma) — an exact summary of that subject's rhythm. "
                    "Inference is then across subjects, so three timepoints per "
                    "subject is sufficient. The population amplitude is the "
                    "length of the MEAN VECTOR, not the mean of the individual "
                    "amplitudes, which is biased upwards.")

    with t_amp:
        st.caption(
            "HYPOTHESIS, not an established finding: in a disease genotype the "
            "circadian amplitude degrades earlier, or proportionally more, than "
            "the MESOR. A smaller population amplitude has two explanations — "
            "every fly's rhythm is damped, or the flies' rhythms are intact but "
            "desynchronised — and the population amplitude alone cannot tell "
            "them apart.")
        if group_col == "(none)":
            st.info("Choose a grouping column above (Genotype, Treatment or "
                    "Sex) to contrast a disease group against a control.")
        else:
            levels = sorted(work[group_col].dropna().astype(str).unique())
            ctrl = st.selectbox("Control group", levels, key="tod_ctrl")
            if st.button("Run amplitude-biomarker contrast", key="tod_amp_run"):
                with st.spinner("Bootstrapping subject-level contrasts..."):
                    try:
                        ab = cc.amplitude_biomarker(
                            work, zt_col="_ZT", value_col="_Y",
                            subject_col=subject_col, group_col=group_col,
                            control_group=ctrl, weight_col="_W", n_boot=400)
                    except Exception as _e:
                        st.error(f"Amplitude-biomarker analysis failed: {_e}")
                        ab = None
                if ab is not None:
                    if not ab.get("available"):
                        st.info(ab.get("reason", "Unavailable."))
                    else:
                        st.caption(ab.get("hypothesis", ""))
                        if not attribution_allowed:
                            st.error(ab.get("prerequisite", ""))
                        for con in ab.get("contrasts", []):
                            if not con.get("available"):
                                st.warning(f"{con.get('group')}: "
                                           f"{con.get('reason', 'unavailable')}")
                                continue
                            if con.get("supports_amplitude_first"):
                                st.warning(f"{con['group']}: {con['evidence']}")
                            else:
                                st.info(f"{con['group']}: {con['evidence']}")
                        atab = pd.DataFrame([
                            {kk: vv for kk, vv in c.items()
                             if isinstance(vv, (int, float, str, bool))}
                            for c in ab.get("contrasts", [])])
                        if not atab.empty:
                            st.dataframe(atab, width="stretch")
                            st.download_button(
                                "Download contrast CSV",
                                atab.to_csv(index=False).encode(),
                                "amplitude_biomarker.csv", "text/csv",
                                key="dl_tod_amp")

    with t_adv:
        st.caption("Evaluate a proposed sampling schedule BEFORE the flies are "
                   "run. Precision figures assume no between-subject variation "
                   "in the rhythm vector, so read them as a best case.")
        a1, a2, a3 = st.columns(3)
        zt_text = a1.text_input("Planned ZT hours", value="2, 8, 14",
                                key="tod_adv_zt")
        n_sub = a2.number_input("Subjects", 3, 2000, 24, 1, key="tod_adv_n")
        ascending = a3.checkbox("Subjects visit times in ascending ZT order",
                                value=True, key="tod_adv_order")
        zts = _parse_day_list(zt_text)
        if len(zts) < 3:
            st.info("Enter at least three distinct Zeitgeber hours.")
        elif st.button("Evaluate schedule", key="tod_adv_run"):
            try:
                adv = cc.design_advisor(zts, n_subjects=int(n_sub),
                                        order_is_zt_order=bool(ascending))
            except Exception as _e:
                st.error(f"Design advisor failed: {_e}")
            else:
                if adv.get("testing_order_confounded_with_zt"):
                    st.error("Testing order is confounded with time of day in "
                             "this schedule.")
                st.markdown(f"**Verdict:** {adv.get('verdict', '')}")
                for wmsg in adv.get("warnings", []) or []:
                    st.warning(wmsg)
                for rec in adv.get("recommendations", []) or []:
                    st.info(rec)
                flat = pd.DataFrame([{k: v for k, v in adv.items()
                                      if isinstance(v, (int, float, str, bool))}])
                st.dataframe(flat, width="stretch")
                sched = adv.get("counterbalancing_schedule", {})
                if isinstance(sched, dict) and sched:
                    with st.expander("Counterbalancing schedule", expanded=False):
                        st.json({k: (v if not isinstance(v, np.ndarray)
                                     else np.asarray(v).tolist())
                                 for k, v in sched.items()})


def _climbing_simple_score():
    """Band-and-zone scoring: find the drawn band, count the strip above it.

    Deliberately has no cylinder model, no glass detection and no background
    subtraction for counting. The geometry the user states is two column numbers
    per cylinder; everything else is one horizontal threshold and one window.
    """
    import tempfile as _tf
    from modules import climbing_band as CB
    from modules import fly_blob as _FB

    up = st.file_uploader("Climbing video", type=["mp4", "mov", "avi", "mkv"],
                          key="cbs_up")
    if up is None:
        st.info("Upload a recording to begin.")
        return
    path = st.session_state.get("cbs_path")
    if st.session_state.get("cbs_name") != up.name:
        tmp = Path(_tf.mkdtemp(prefix="cbs_")) / up.name
        tmp.write_bytes(up.getbuffer())
        path = str(tmp)
        st.session_state["cbs_path"], st.session_state["cbs_name"] = path, up.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        st.error("That file could not be opened.")
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first = cap.read()
    cap.release()
    if not ok or fps <= 0:
        st.error("That file reports no frame rate or no readable frames.")
        return
    dur = n_fr / fps
    fh, fw = first.shape[:2]
    st.success(f"{up.name} — {fw}x{fh}, {fps:.2f} fps, {dur:.0f} s")

    c1, c2, c3 = st.columns(3)
    n_flies = c1.number_input("Flies per cylinder", 1, 200, 20, 1, key="cbs_n")
    n_cyl = c2.number_input("Cylinders in frame", 1, 3, 3, 1, key="cbs_ncyl")
    # THE ZONE IS THE WHOLE TUBE ABOVE THE BAND BY DEFAULT. A fixed 250 px was
    # tuned on one rig and on a closer camera it stopped two-thirds of the way up
    # the glass: measured on the 15th-Day set, 13 of 15 wild-type flies above the
    # band sat beyond it and MAE against 361 hand counts was 7.38; with the whole
    # tube it is 4.53, and the older rig's hold-out is unchanged (6.53 -> 6.52).
    # A limit is still offered for footage where something above the glass must
    # be excluded, but it is opt-in.
    limit_zone = c3.checkbox("Limit the counting zone height", value=False,
                             key="cbs_limit")
    zone_h = (c3.slider("Zone height above the band (px)", 100, 800, 400, 10,
                        key="cbs_zone") if limit_zone else None)

    tp_txt = st.text_input("Timepoints, seconds after the start of the recording",
                           "10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120",
                           key="cbs_tp")
    try:
        times = [float(t) for t in tp_txt.split(",") if t.strip()]
    except ValueError:
        st.error("Timepoints must be numbers separated by commas.")
        return

    with st.expander("Detection"):
        band_dark = st.slider("Band darkness (the drawn mark)", 20, 150,
                              CB.BAND_DARK, 5, key="cbs_bd")
        st.caption(
            "This finds the BAND only. The flies are counted by a size-matched "
            "blob detector — no threshold adjustment needed, and none is offered: "
            "its fly diameter is a fitted value (0.07 of the cylinder width) and "
            "is sharply peaked (0.05 → MAE 9.11, 0.07 → 2.93, 0.10 → 9.82 flies), "
            "so a slider on it would be a slider on the error.")
        st.caption(
            "The Fly-darkness and Min/Max blob-area sliders that used to sit here "
            "have been removed rather than left in place: once counting moved to "
            "the blob detector they no longer affected any number, and a control "
            "that does nothing is worse than no control — this page previously "
            "shipped exactly that defect and told users to adjust it when a count "
            "looked wrong.")
        drop_static = st.checkbox(
            "Discard detections that never move", value=False, key="cbs_static")
        st.caption(
            "Applies to the snapshot columns only. It removes printed label text "
            "and also removes flies that cling motionless to the glass — measured "
            "on the wild-type control it cut a correct count of 15-18 to 6-9, so "
            "it is off by default. The crossing gate does not need it.")
    fly_dark, min_a, max_a = CB.FLY_DARK, 20, 400

    st.subheader("Cylinder columns")
    st.caption("Give the left and right pixel column of each cylinder. "
               "Everything outside these columns is ignored.")
    cols_xy, step = [], fw // max(int(n_cyl), 1)
    for i in range(int(n_cyl)):
        d1, d2 = st.columns(2)
        c0 = d1.number_input(f"Cylinder {i+1} — left column", 0, fw - 2,
                             min(i * step + step // 4, fw - 2), 1, key=f"cbs_c0{i}")
        c1_ = d2.number_input(f"Cylinder {i+1} — right column", 1, fw - 1,
                              min(i * step + 3 * step // 4, fw - 1), 1,
                              key=f"cbs_c1{i}")
        if c1_ > c0:
            cols_xy.append((int(c0), int(c1_)))

    # ── the check that matters: the band it found and the strip it will read ──
    st.subheader("What it found")
    st.caption("Red line = the band. Orange box = the strip that will be counted. "
               "If the red line is not on the drawn band, nothing below is valid.")
    probe = []
    cap = cv2.VideoCapture(path)
    for idx in np.linspace(0, max(n_fr - 1, 0), 15).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            probe.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not probe:
        st.error("No frames could be read from this file.")
        return
    median_img = np.median(np.stack(probe), axis=0).astype(np.uint8)
    st.caption(
        "The band is located on a 15-frame median, not on the first frame. At "
        "the tap the flies are heaped in the bottom of the cylinder and a raft "
        "of flies is dark right across the tube — measured, reading frame 0 put "
        "the band 322 px too low on one recording.")

    vis = first.copy()
    bands, tubes = [], []
    for i, (c0, c1_) in enumerate(cols_xy, 1):
        by = CB.find_black_band(median_img, c0, c1_, band_dark)
        bands.append(by)
        cv2.rectangle(vis, (c0, 0), (c1_, fh - 1), (255, 190, 0), 1)
        if by is None:
            tubes.append(None)
            cv2.putText(vis, f"C{i} NO BAND", (max(c0 - 20, 2), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue
        top = 0 if zone_h is None else max(0, by - int(zone_h))
        cv2.rectangle(vis, (c0, top), (c1_, by), (0, 190, 255), 2)
        cv2.line(vis, (max(c0 - 25, 0), by), (min(c1_ + 25, fw - 1), by),
                 (0, 0, 255), 3)
        cv2.putText(vis, f"C{i} band={by}", (max(c0 - 20, 2), max(top - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        # THE FLY SIZE IS SET BY THE TUBE WIDTH, SO SHOW IT. The detector looks
        # for a fly of 0.07 x width; measured on a 100 px cylinder holding 20
        # flies, a box drawn 3x too wide assumed a 12 px fly and reported ZERO,
        # silently. The tube is now measured from the band's own dark run and
        # drawn in green, so a mis-drawn box is visible before the run, not
        # after it.
        _tube = CB.tube_width_from_band(median_img, by, c0, c1_, band_dark)
        tubes.append(_tube)
        if _tube is not None:
            tx0, tx1 = _tube
            cv2.line(vis, (tx0, by + 26), (tx1, by + 26), (0, 200, 0), 3)
            cv2.putText(vis, f"tube {tx1 - tx0}px  fly {_FB.fly_diameter(tx1 - tx0):.1f}px",
                        (max(tx0 - 20, 2), by + 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 140, 0), 2)
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")

    for i, (by, _tube) in enumerate(zip(bands, tubes), 1):
        if by is None or _tube is None:
            continue
        tw = _tube[1] - _tube[0]
        bw = cols_xy[i - 1][1] - cols_xy[i - 1][0]
        if bw / max(tw, 1) > 1.6:
            st.error(
                f"Cylinder {i}: your box is {bw} px wide but the glass measures "
                f"{tw} px ({bw / max(tw, 1):.1f}x). The detector sizes a fly from "
                f"the width it is given, so this box looks for a "
                f"{_FB.fly_diameter(bw):.1f} px fly instead of {_FB.fly_diameter(tw):.1f} px "
                "and will undercount. Draw the box to the glass — the green bar "
                "shows where it is.")
        elif bw / max(tw, 1) > 1.25:
            st.info(
                f"Cylinder {i}: box {bw} px, glass {tw} px. Scoring uses the "
                f"measured {tw} px, so the count is unaffected — but tightening "
                "the box makes the preview easier to read.")

    for i, by in enumerate(bands, 1):
        if by is None:
            st.error(
                f"Cylinder {i}: no row reaches {100*0.60:.0f}% dark coverage, so "
                "the band was not found and this cylinder will not be scored. "
                "Lower the band-darkness threshold only if you can see on the "
                "image above that the real band is being missed.")

    if not st.button("Run", type="primary", key="cbs_run"):
        return
    if not any(b is not None for b in bands):
        st.error("No band was found on any cylinder — nothing to score.")
        return

    frames = []
    all_rows = []
    bar = st.progress(0.0, text="Scoring…")
    for i, (c0, c1_) in enumerate(cols_xy, 1):
        bar.progress(i / max(len(cols_xy), 1), text=f"Cylinder {i}…")
        df = CB.analyse_video(path, c0, c1_, int(n_flies), times,
                              zone_height_px=(None if zone_h is None else int(zone_h)),
                              band_dark=int(band_dark),
                              fly_dark=int(fly_dark), min_blob_area=int(min_a),
                              max_blob_area=int(max_a), drop_static=drop_static)
        df.insert(0, "cylinder", f"C{i}")
        df.insert(1, "video", up.name)
        all_rows.append(df)
    bar.empty()
    res = pd.concat(all_rows, ignore_index=True)
    st.session_state["cbs_res"] = res

    scored = res[res.status == "scored"]
    if scored.empty:
        st.error("Nothing was scored. Every cylinder refused — see the messages above.")
        return

    st.subheader("Flies above the band")
    wide = scored.pivot_table(index="time_sec", columns="cylinder",
                              values="reported_count", aggfunc="mean")
    st.line_chart(wide)
    st.caption(
        "The reported count is the midpoint of a bracket. Its upper bound counts "
        "every detection above the band and is necessarily too high (the printed "
        "label is dark and fly-sized); its lower bound counts only detections "
        "that move and is necessarily too low (a fly clinging still is discarded "
        "with the print). Measured against 15 hand counts: upper MAE 3.47 "
        "(bias +2.53), lower MAE 2.80 (bias −2.67), **midpoint MAE 2.20 "
        "(bias −0.07)** — inside the assay's own 3-fly noise floor.")

    low = scored[scored.confidence == "LOW"]
    if not low.empty:
        st.warning(
            f"{len(low)} of {len(scored)} readings have a bracket wider than 7 "
            "flies. The two bounds disagree, so the midpoint carries real "
            "uncertainty — read `count_lower` and `count_upper` rather than the "
            "single number.")

    gap = scored[scored.accounted_gap.abs() > 5]
    if not gap.empty:
        st.error(
            f"{len(gap)} of {len(scored)} readings do not account for the "
            f"{int(n_flies)} flies loaded: the two zones together found "
            f"{gap.accounted_for.min()}–{gap.accounted_for.max()}. A sealed "
            "cylinder holds exactly what was put in it, so a shortfall means "
            "flies are merging in the crowd below the band and a surplus means "
            "the print is being counted above it. This is a diagnostic, not a "
            "correction — forcing the sum to match was tested and made the "
            "answer worse (MAE 4.66 against 2.20).")
        st.dataframe(gap[["cylinder", "time_sec", "count_above", "count_below",
                          "accounted_for", "accounted_gap"]], width="stretch")

    st.dataframe(scored[["cylinder", "time_sec", "band_y", "reported_count",
                         "reported_pi", "count_lower", "count_upper",
                         "bracket_width", "confidence", "count_below",
                         "accounted_for"]], width="stretch")
    st.download_button("Download CSV", res.to_csv(index=False).encode(),
                       file_name=f"climbing_simple_{Path(up.name).stem}.csv",
                       mime="text/csv", key="cbs_dl")


def _climbing_simple_validate():
    """Hand counts in, MAE out. Same report the command-line script prints."""
    from modules.climbing_band import validation_report, NOISE_FLOOR_FLIES

    st.caption(
        "Upload the bench sheet with your hand counts filled in, and the CSV the "
        "Score tab exported. Nothing here changes the scoring — it only measures "
        "it.")

    tmpl = Path("data/validation/hand_count_template.csv")
    if tmpl.exists():
        st.download_button("Download the blank bench sheet",
                           tmpl.read_bytes(),
                           file_name="hand_count_template.csv",
                           mime="text/csv", key="cbv_tmpl")
        st.caption(
            "Take this to the bench. Pause each recording at 10 s, 30 s and 60 s "
            "and write the number of flies you can see above the black band. "
            "Leave a row blank if you cannot count it confidently — a guessed "
            "hand count is worse than a missing one, because it becomes the "
            "yardstick everything else is judged against.")

    c1, c2 = st.columns(2)
    hf = c1.file_uploader("Hand counts (filled in)", type=["csv"], key="cbv_hand")
    tf = c2.file_uploader("Tool output CSV", type=["csv"], key="cbv_tool")
    if hf is None or tf is None:
        st.info("Upload both files to compute the accuracy.")
        return

    try:
        hand = pd.read_csv(hf)
        tool = pd.read_csv(tf)
    except Exception as exc:
        st.error(f"One of those files could not be read: {exc}")
        return

    if not st.button("Compute MAE", type="primary", key="cbv_go"):
        return

    root = Path(r"C:/Users/gkp11/Videos/HD Climbing assay")
    lookup = {}
    if root.exists():
        for f in root.glob("**/*.MOV"):
            s_ = str(f)
            if "3 rd day" in s_:
                lookup[f.name] = "Day 3"
            elif "7 th day" in s_:
                lookup[f.name] = "Day 7"

    report = validation_report(hand, tool, day_lookup=lookup)
    st.code(report, language="text")
    st.download_button("Download the report (.txt)", report.encode(),
                       file_name="climbing_band_validation.txt",
                       mime="text/plain", key="cbv_dl")
    if "IS within" in report:
        st.success(f"MAE is at or below the {NOISE_FLOOR_FLIES}-fly noise floor of "
                   "the assay itself — the tool is as accurate as the measurement "
                   "it replaces.")
    elif "IS NOT within" in report:
        st.warning(f"MAE exceeds the {NOISE_FLOOR_FLIES}-fly noise floor. The "
                   "vials listed as differing by more than 5 flies are where to "
                   "look first.")


def page_climbing_simple():
    """Score a recording, or measure the scoring against hand counts."""
    st.header("Climbing (Simple)")
    st.caption(
        "Find the drawn band, count dark blobs in a strip above it, divide by "
        "the flies you loaded. No cylinder fitting, no background subtraction.")
    tab_score, tab_val = st.tabs(["Score", "Validate"])
    with tab_score:
        _climbing_simple_score()
    with tab_val:
        _climbing_simple_validate()


PAGE_MAP = {
    "Home":                   page_home,
    "Plan a Run":             page_plan_run,
    "Data Input":             page_data_input,
    "Design Check":           page_design_check,
    "Climbing Curves":        page_climbing_curves,
    "Core Metrics":           page_core_metrics,
    "Novel Metrics":          page_novel_metrics,
    "Motor Decline":          page_motor_decline,
    "Survival Analysis":      page_survival,
    "Y-maze":                 page_ymaze,
    "Activity Monitoring (DAMS)": page_dams,
    "DAM Run Analysis":       page_dam_pipeline,
    "Larval Crawling":        page_larval,
    "Grooming":               page_grooming,
    "Dose-Response":          page_dose_response,
    "Time-of-Day":            page_time_of_day,
    "Drug Ranking":           page_drug_ranking,
    "Trajectory Clustering":  page_trajectory_clustering,
    "Trajectory Forecast":    page_trajectory_forecast,
    "Translational Score":    page_tcs,
    "Statistical Analysis":   page_stats,
    "Hierarchical Statistics": page_hierarchical,
    "Survivor Bias":          page_survivor_bias,
    "Historical Controls":    page_historical_controls,
    "Live Recording":         page_recording,
    "Video Analysis":         page_video_analysis,
    "Climbing Video":         page_climbing_video,
    "Climbing (Simple)":      page_climbing_simple,
    "Draw Vials":             page_draw_vials,
    "Video Scoring":          page_video_scoring,
    "CSV Upload":             page_csv_upload,
    "Export":                 page_export,
}

PAGE_MAP[page]()