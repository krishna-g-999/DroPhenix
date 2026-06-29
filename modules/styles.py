"""DroPhenix styles v6 — all issues fixed."""
from __future__ import annotations
import base64
import streamlit as st
from pathlib import Path


def _logo_b64() -> str:
    try:
        p = Path("logs/Drophenix_Logo.png")
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    except Exception:
        pass
    return ""


def inject_css() -> None:
    b64 = _logo_b64()
    logo_src = f"data:image/png;base64,{b64}" if b64 else ""

    # ── Inject sidebar brand with logo ────────────────────────────────────────
    if logo_src:
        logo_html = (
            f'<img src="{logo_src}" style="height:62px;width:62px;'
            f'object-fit:contain;display:block;margin:0 auto 6px;'
            f'filter:drop-shadow(0 3px 8px rgba(240,165,0,0.35));">'
        )
    else:
        logo_html = (
            '<div style="width:62px;height:62px;border-radius:50%;margin:0 auto 6px;'
            'background:linear-gradient(135deg,#F0A500,#C47F00);'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:1.1rem;font-weight:900;color:#fff;">DP</div>'
        )

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1.1rem 0 0.6rem;">'
            + logo_html
            + '<div style="font-size:1rem;font-weight:800;color:#FFFFFF;'
            'line-height:1.2;letter-spacing:-.01em;">DroPhenix</div>'
            '<div style="font-size:.68rem;color:#94A3B8;letter-spacing:.05em;'
            'text-transform:uppercase;margin-top:2px;">Analytics v2.0</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Main CSS injection ────────────────────────────────────────────────────
    st.markdown("""
<style>
/* ── Reset ───────────────────────────────────────────────────────────────── */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

/* ── Sidebar shell ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09192A 0%, #0D2137 100%) !important;
    border-right: 1px solid rgba(240,165,0,0.18) !important;
    min-width: 200px !important;
}

/* ── Sidebar text — SCOPED SPECIFICALLY (no wildcard bleed) ─────────────── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] label {
    color: #CBD5E1 !important;
}

/* ── Sidebar nav radio ───────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stRadio > div > label {
    display: block !important;
    padding: 0.42rem 0.85rem !important;
    border-radius: 7px !important;
    border-left: 3px solid transparent !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    margin: 1px 0 !important;
    white-space: nowrap !important;
    transition: background 0.15s !important;
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(240,165,0,0.12) !important;
    border-left-color: rgba(240,165,0,0.55) !important;
    color: #F0A500 !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(240,165,0,0.18) !important;
    border-left-color: #F0A500 !important;
    color: #F0A500 !important;
    font-weight: 700 !important;
}
/* Hide the radio dot */
section[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}
/* Hide the radio group header */
section[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
    display: none !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(240,165,0,0.18) !important;
    margin: 0.5rem 0 !important;
}

/* ── MAIN AREA TEXT — always dark on white/grey backgrounds ─────────────── */
/* This is the root fix for "white numbers on white background"             */
[data-testid="stMain"] input[type="number"],
[data-testid="stMain"] input[type="text"],
[data-testid="stMain"] .stNumberInput input,
[data-testid="stMain"] .stTextInput input {
    color: #0D2137 !important;
    background: #FFFFFF !important;
    border: 1px solid #C5D7F0 !important;
}
/* Metric widget values */
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #0D2137 !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #C5D7F0 !important;
    border-radius: 10px !important;
    padding: 0.7rem 0.9rem !important;
}
/* Dataframe cell text */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    color: #0D2137 !important;
}
/* Slider value label */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: #374151 !important;
}
/* Select boxes, multiselect */
[data-testid="stMain"] .stSelectbox [data-baseweb="select"] div,
[data-testid="stMain"] .stMultiSelect [data-baseweb="select"] div {
    color: #0D2137 !important;
}
/* Caption and small text */
[data-testid="stMain"] .stCaption,
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: #6B7280 !important;
}

/* ── Persistent page header ───────────────────────────────────────────────── */
.dpx-page-header {
    background: #FFFFFF;
    border-bottom: 3px solid #F0A500;
    border-radius: 12px 12px 0 0;
    padding: 0.8rem 1.4rem;
    margin: -0.5rem -0.5rem 1.6rem -0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.dpx-page-header-logo {
    height: 44px; width: 44px; object-fit: contain; flex-shrink: 0;
    filter: drop-shadow(0 2px 6px rgba(240,165,0,0.28));
}
.dpx-page-header-logo-fb {
    width: 44px; height: 44px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg,#F0A500,#C47F00);
    display: flex; align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 900; color: #fff;
}
.dpx-page-header-brand {
    font-size: .63rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #F0A500; margin-bottom: 2px;
}
.dpx-page-header-title {
    font-size: 1.22rem; font-weight: 800; color: #0D2137; line-height: 1.15;
}
.dpx-page-header-sub {
    font-size: .78rem; color: #6B7280; margin-top: 2px;
}

/* ── Tabs — always visible ───────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #EEF2F9 !important;
    border-bottom: 2px solid #C5D7F0 !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.3rem 0.4rem 0 !important;
    gap: 0.2rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: #F1F5FB !important;
    border: 1px solid #C5D7F0 !important;
    border-bottom: none !important;
    border-radius: 7px 7px 0 0 !important;
    color: #374151 !important;
    font-size: .84rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1.1rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background: #FFFFFF !important;
    color: #0D2137 !important;
}
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"] {
    background: #FFFFFF !important;
    border-bottom: 3px solid #F0A500 !important;
    color: #0D2137 !important;
    font-weight: 800 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#F0A500,#D4920A) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 3px 10px rgba(240,165,0,0.28) !important;
}
.stButton > button[kind="secondary"],
.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #0D2137 !important;
    border: 1px solid #C5D7F0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
    background: #F1F5FB !important;
    border-color: #F0A500 !important;
    color: #0D2137 !important;
}

/* ── Status badges ───────────────────────────────────────────────────────── */
.dpx-badge-complete {
    background:#DCFCE7;color:#166534;border:1px solid #86EFAC;
    border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:700;
}
.dpx-badge-idle {
    background:#F1F5FB;color:#64748B;border:1px solid #CBD5E1;
    border-radius:20px;padding:2px 10px;font-size:.72rem;font-weight:700;
}
</style>""", unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "", extra: str = "") -> None:
    """Persistent branded top header on every page."""
    b64 = _logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{b64}" class="dpx-page-header-logo">'
        if b64 else
        '<div class="dpx-page-header-logo-fb">DP</div>'
    )
    sub_html = (
        f'<div class="dpx-page-header-sub">{subtitle}</div>'
        if subtitle else ''
    )
    st.markdown(
        '<div class="dpx-page-header">'
        + logo_html
        + '<div style="flex:1;">'
        + '<div class="dpx-page-header-brand">'
        + 'DroPhenix Analytics v2.0 \u00b7 Science for Society</div>'
        + f'<div class="dpx-page-header-title">{title}</div>'
        + sub_html
        + '</div></div>',
        unsafe_allow_html=True,
    )


def chip(text: str, style: str = "pathway") -> str:
    colours = {
        "pathway": ("#EBF4FF", "#1E40AF"),
        "rescue":  ("#DCFCE7", "#166534"),
        "warning": ("#FEF9C3", "#854D0E"),
        "danger":  ("#FEE2E2", "#991B1B"),
    }
    bg, col = colours.get(style, ("#EBF4FF", "#1E40AF"))
    return (
        f'<span style="background:{bg};color:{col};border-radius:20px;'
        f'font-size:.7rem;font-weight:700;padding:2px 9px;">{text}</span>'
    )


def status_badge(text: str, state: str = "idle") -> str:
    cls = {
        "complete": "dpx-badge-complete",
        "idle":     "dpx-badge-idle",
    }.get(state, "dpx-badge-idle")
    return f'<span class="{cls}">{text}</span>'
