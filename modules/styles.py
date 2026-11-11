"""
DroPhenix interface styling: one token system, no decoration.

WHY THIS WAS REWRITTEN (register N122)
---------------------------------------
The owner's report was "it looks like an AI generated tool" and "the logo and
the colour of the page doesnt match and it looks off". Both turned out to be
measurable rather than matters of taste.

**Three oranges at two hues.** Measured off the files themselves:

    the logo's dominant saturated hue   26.3 deg   (logs/Drophenix_Logo.png)
    the UI accent  #F0A500              41.2 deg   (this file, 14 places)
    Wong orange    #E69F00              41.5 deg   (visualizer.py, every figure)

So the chrome matched the FIGURES and clashed with the LOGO, by 15 degrees of
hue -- far enough to read as a mistake, close enough that nobody can name why.
It also sat 0.3 degrees from the Wong orange used for data series, which is a
near-miss rather than a system.

The resolution uses a measured coincidence: **Wong vermilion #D55E00 sits at
26.5 degrees, 0.2 degrees from the logo's own 26.3**. So one value is
simultaneously an exact member of the colourblind-safe figure palette and an
exact match for the logo. It is used ONLY for non-text brand marks -- the active
navigation rail, the active tab underline, the mark's dot -- because #D55E00 on
white is 3.87:1, which is fine for a 2px rule and not fine for a word.

Interactive elements that carry TEXT use a deep desaturated navy #1B3A5C
(hue 209 deg, saturation 54%). Against Wong blue #0072B2 (201 deg, saturation
100%) it reads as chrome rather than as a data series, which is the point:
in a tool whose output is figures, saturated colour should mean "this is data".

**A live accessibility failure.** Caption text was #6B7280. On a white card that
is 4.83:1 and passes; on the real page background #EEF2F9 it is **4.31:1 and
fails WCAG AA**. Captions carry the caveats in this tool, so they are the last
text that should be hard to read. Measured, not estimated; now 4.72:1.

**A dead import.** `app.py:215` has imported `render_sidebar_brand` from this
module for as long as it has existed, inside a `try/except ImportError`. The
function was never defined, so the fallback -- bare amber text -- always fired.
It is defined here now.

WHAT WAS REMOVED, AND WHY EACH ONE READS AS MACHINE-MADE
---------------------------------------------------------
* Coloured glow shadows (`rgba(240,165,0,.28)` on buttons, a drop-shadow on the
  logo). Instruments do not glow.
* Five gradients. The sidebar's spanned about 3% lightness -- invisible, and
  pure template tax.
* Six corner radii with no scale (7, 8, 10, 12, 20px, 50%), now a 2/3/4 scale.
  Small radii read as precision.
* Font weights 800 and 900, now capped at 600. Scientific interfaces separate
  hierarchy with size and space, not with fat.
* Pill-shaped badges, now rectangular tags.
* Stock Tailwind semantic colours, which were never contrast-checked against
  their own tints; the replacements are all at least 5.2:1 on their own
  backgrounds.
* "Analytics v3.0 - Science for Society" on all 37 page headers. A tagline
  inside an instrument reads as promotional to a reviewer.

The raster logo is kept and still available (`render_header(..., raster=True)`
and `USE_RASTER_MARK`), but the default mark is now the inline SVG below: the
PNG is 1,025,182 bytes, base64-inlined into every one of 37 headers.

WHAT THIS FILE DOES NOT CLAIM
------------------------------
Nothing here is validated by running the app. `validate_app_pages` checks that
pages render and produce output, not that they look right. Contrast ratios and
hues above were computed from the actual values and are checkable; everything
else is a design decision.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

#: Set True to go back to the raster logo everywhere.
USE_RASTER_MARK = True

#: The mark: a graticule (the reticle of a microscope eyepiece) with a
#: climbing-assay step trace across it. Two ideas that are both true of this
#: product -- an optical field of view, and the negative-geotaxis curve that is
#: its core assay. Drawn in `currentColor` so one definition works white on the
#: navy sidebar and navy on the white page. The single dot is Wong vermilion,
#: an exact member of the figure palette and a 0.2-degree match to the logo.
DPX_MARK = """
<svg viewBox="0 0 24 24" width="{w}" height="{w}" role="img" aria-label="DroPhenix"
     fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.4"/>
  <g stroke="currentColor" stroke-width="1.4" opacity=".5">
    <path d="M12 3v2.4M12 18.6V21M3 12h2.4M18.6 12H21"/>
  </g>
  <path d="M5.8 15.8h3.3v-3.1h3.3V9.4h2.3" stroke="currentColor"
        stroke-width="1.9" stroke-linejoin="miter" stroke-linecap="butt"/>
  <circle cx="15.1" cy="9.4" r="2" fill="#D55E00"/>
</svg>
"""


def mark(size: int = 28, colour: str = "currentColor") -> str:
    """
    The mark, as an <img> carrying a data-URI SVG.

    Streamlit's markdown sanitiser strips a raw `<svg>` element even under
    `unsafe_allow_html=True` -- verified in the running app, where the inline
    form rendered zero marks. A data-URI `<img>` survives. The cost is that
    `currentColor` no longer inherits, so the stroke colour is baked in and the
    caller states it: white for the navy sidebar, navy for the white page.
    """
    import base64 as _b64
    svg = DPX_MARK.format(w=int(size)).replace("currentColor", colour)
    enc = _b64.b64encode(svg.encode("utf-8")).decode()
    return (f'<img src="data:image/svg+xml;base64,{enc}" '
            f'width="{int(size)}" height="{int(size)}" alt="DroPhenix" '
            f'style="display:block;flex:0 0 auto;">')


@st.cache_data(show_spinner=False)
def _glyph_b64(retina: bool = True) -> str:
    """
    The logo CROPPED to its ring-fly-flame detail, which is the part that
    survives small sizes.

    Measured on the original 1254x1254 file: the full mark carries SEVEN motifs
    (ring, fly, flame, phoenix, helix, bar chart, line chart) plus a wordmark
    and a tagline. At the 28-30px it is shown at on 37 page headers and in the
    sidebar, the emblem gets about 13px of height for all of that, and nothing
    resolves. The crop keeps the ring, the fly and the flame and discards the
    rest; the fly is the one element a Drosophila reviewer recognises instantly.

    Two further measured facts drive this. The file is opaque RGB with an
    off-white ground (mean 254.2, only 13.7% exactly white), so it paints a
    white box on the navy sidebar; and keying the white out destroys the mark,
    because the ring and helix are #112133 against a #0F2133 sidebar -- a
    contrast ratio of 1.0. The glyph therefore sits on a white CHIP on navy.

    Crop box on the original: (178, 264, 666, 752).
    """
    name = "drophenix_glyph@2x.png" if retina else "drophenix_glyph.png"
    try:
        q = Path(__file__).resolve().parent.parent / "logs" / name
        if q.exists():
            return base64.b64encode(q.read_bytes()).decode()
    except Exception:
        pass
    return ""


def glyph_img(size: int = 28, chip: bool = False) -> str:
    """The glyph as an <img>, optionally on the white chip a dark ground needs."""
    b64 = _glyph_b64()
    inner = (f'<img src="data:image/png;base64,{b64}" '
             f'style="width:{int(size)}px;height:{int(size)}px;display:block;" '
             f'alt="DroPhenix">') if b64 else mark(size, "#FFFFFF")
    if not chip:
        return inner
    pad = max(int(round(size * 0.14)), 3)
    return (f'<span style="flex:0 0 auto;display:flex;align-items:center;'
            f'justify-content:center;width:{size+2*pad}px;height:{size+2*pad}px;'
            f'border-radius:{max(int(size*0.19),4)}px;background:#FFFFFF;'
            f'box-shadow:inset 0 0 0 1px rgba(15,33,51,.10);">{inner}</span>')


@st.cache_data(show_spinner=False)
def _logo_b64() -> str:
    """The full logo, base64. Cached: it is a 1,025,182-byte read and encode,
    and `render_header` runs on every rerun of every page."""
    try:
        p = Path(__file__).resolve().parent.parent / "logs" / "Drophenix_Logo.png"
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    except Exception:
        pass
    return ""


_CSS = """
<style>
:root{
  /* Surfaces. Neutral greys: the previous blue-tinted #EEF2F9 / #C5D7F0 set up
     a simultaneous-contrast fight with a warm logo. */
  --dpx-canvas:#F2F4F6; --dpx-surface:#FFFFFF; --dpx-surface-sunken:#E9ECF0;

  /* Borders. 1.33:1 is correct for a divider (decorative under WCAG 1.4.11);
     controls get the stronger value. */
  --dpx-border:#DCE0E6; --dpx-border-strong:#848F9C; --dpx-border-focus:#8A3D00;

  /* Text. Ratios against --dpx-canvas: 14.81, 7.34, 4.72. All pass AA. */
  /* #0F2133 is a 1.5% match to the logo's own ink #112133 (211.8 deg), so the
     chrome is logo-led in BOTH hues, not just the orange. */
  --dpx-text-primary:#0F2133; --dpx-text-secondary:#45515F;
  /* The logo's own chart-bar grey, 211.4 deg: figure furniture and rules. */
  --dpx-grey-figure:#6F7984;
  --dpx-text-muted:#626E7B;   --dpx-text-invert:#FFFFFF;

  /* Interactive, text-bearing. LOGO-LED: #A84A00 sits at hue 26.4 deg, within
     0.1 deg of the logo's own 26.3, and passes AA both ways (5.76:1 on white,
     and 5.76:1 for white text on it). The earlier navy was accessible but made
     the page a different colour from the mark at the top of it, which is the
     thing that looked wrong. */
  --dpx-accent:#A84A00; --dpx-accent-hover:#8A3D00; --dpx-accent-weak:#FBEEE6;

  /* Brand mark only, never behind or as text: 3.87:1 on white. Wong vermilion,
     0.2 degrees from the logo's own hue. */
  --dpx-brand:#D55E00;

  /* Semantic, each at least 5.2:1 on its OWN tint. */
  --dpx-success:#1B6E3F; --dpx-success-bg:#EAF4EE;
  --dpx-warning:#8A5A00; --dpx-warning-bg:#FBF1DF;
  --dpx-danger:#A32015;  --dpx-danger-bg:#FAEAE8;

  /* Sidebar: flat, not a 3%-lightness gradient. */
  --dpx-nav-bg:#0F2133; --dpx-nav-text:#C8D2DC;
  --dpx-nav-muted:#94A3B1; --dpx-nav-active:#FFFFFF;

  --dpx-r-sm:2px; --dpx-r-md:3px; --dpx-r-lg:4px;
  --dpx-w-normal:400; --dpx-w-medium:500; --dpx-w-semibold:600;
  --dpx-s1:4px; --dpx-s2:8px; --dpx-s3:12px; --dpx-s4:16px;
  --dpx-s6:24px; --dpx-s8:32px;

  /* Streamlit self-hosts these as variable fonts under the EXACT family name
     "Source Sans" -- not "Source Sans 3". The wrong string falls through to
     the system stack silently. */
  --dpx-font:"Source Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --dpx-font-mono:"Source Code Pro",ui-monospace,Consolas,monospace;
}

/* Guard: if a host forces dark, keep the light instrument. Every figure this
   app renders has a white facecolor (visualizer.py), so a dark chrome would
   frame a hundred glaring white panels. */
@media (prefers-color-scheme: dark){
  :root{ color-scheme: light; }
  [data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background:var(--dpx-canvas) !important; color:var(--dpx-text-primary) !important; }
}

[data-testid="stMainBlockContainer"],[data-testid="stAppViewBlockContainer"]{
  /* 72px clears Streamlit's own 60px fixed header with one 12px gap. Its
     default is 96px, which left the page title floating a line and a half
     below the header -- visible as "the header is not at the top". */
  padding:72px var(--dpx-s6) var(--dpx-s8) !important;
  max-width:1480px !important;
}
[data-testid="stVerticalBlock"]{ gap:.6rem !important; }
[data-testid="stHorizontalBlock"]{ gap:var(--dpx-s4) !important; }
[data-testid="stCaptionContainer"]{
  margin-top:-.35rem !important; margin-bottom:var(--dpx-s3) !important; }

/* Type scale: six steps, weight capped at 600. */
[data-testid="stMain"] h1,.dpx-page-header-title{
  font-size:1.5rem !important; font-weight:var(--dpx-w-semibold) !important;
  line-height:1.25 !important; letter-spacing:-.012em !important;
  color:var(--dpx-text-primary) !important; }
[data-testid="stMain"] h2{
  font-size:1.125rem !important; font-weight:var(--dpx-w-semibold) !important;
  line-height:1.3 !important; color:var(--dpx-text-primary) !important;
  margin:var(--dpx-s6) 0 var(--dpx-s2) 0 !important; }
[data-testid="stMain"] h3{
  font-size:.9375rem !important; font-weight:var(--dpx-w-semibold) !important;
  color:var(--dpx-text-secondary) !important;
  margin:var(--dpx-s4) 0 var(--dpx-s1) 0 !important; }
/* The colour here is deliberately NOT !important, while the metrics are.
   An author !important declaration beats an INLINE one, so a blanket
   `color: ... !important` on every paragraph painted the Home hero's own lede
   #0F2133 on a #0D2137 card -- measured at 1.00:1, invisible -- and no inline
   style could rescue it. The first attempt at a fix was worse: forcing
   `color:inherit !important` on the panel's descendants overrode their inline
   colours too, so the title, eyebrow and strapline disappeared as well. The
   rule simply must not be !important for colour; its specificity already beats
   Streamlit's own base styles, and an author who sets an inline colour on a
   dark card then gets what they asked for. */
[data-testid="stMain"],[data-testid="stMain"] p,[data-testid="stMain"] li{
  font-size:.9375rem !important; line-height:1.55 !important;
  color:var(--dpx-text-primary); }
/* A dark panel states its own text colour, so anything inheriting lands light. */
[data-testid="stMain"] .dpx-on-dark{ color:#CBD5E1; }
[data-testid="stMain"] .stCaption,
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p{
  font-size:.8125rem !important; line-height:1.45 !important;
  color:var(--dpx-text-muted) !important; }

/* Tabular figures. Without this, digits are proportional and columns of numbers
   do not line up -- the clearest signal that a tool was not built by people who
   read tables. Also stops metric values jittering between reruns. */
[data-testid="stDataFrame"],[data-testid="stTable"],
[data-testid="stMetricValue"],[data-testid="stMetricDelta"],
[data-testid="stMain"] code,.dpx-num{
  font-variant-numeric:tabular-nums lining-nums !important;
  font-feature-settings:"tnum" 1,"lnum" 1 !important; }
[data-testid="stDataFrame"]{ font-size:.8125rem !important; }
[data-testid="stDataFrame"] [role="gridcell"]{
  padding-top:2px !important; padding-bottom:2px !important; }

/* Sidebar: flat navy; active state is white + a rail, not a hue change. */
section[data-testid="stSidebar"]{
  background:var(--dpx-nav-bg) !important;
  border-right:1px solid rgba(255,255,255,.08) !important;
  min-width:224px !important; }
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] label{ color:var(--dpx-nav-text) !important; }
section[data-testid="stSidebar"] .stRadio > div > label{
  display:block !important; padding:7px 12px !important;
  border-radius:var(--dpx-r-md) !important;
  border-left:3px solid transparent !important;
  /* The options are navigation, not prose: medium by default so they read as
     controls, semibold when active. */
  font-size:.8125rem !important; font-weight:var(--dpx-w-medium) !important;
  line-height:1.4 !important; margin:0 !important; cursor:pointer !important;
  color:var(--dpx-nav-text) !important;
  transition:background .12s ease,color .12s ease !important; }
section[data-testid="stSidebar"] .stRadio > div > label:hover{
  background:rgba(255,255,255,.06) !important; color:var(--dpx-nav-active) !important; }
/* The active option is a filled box in the logo's own colour, not a tint of
   white. A 3px rail alone was too quiet to find at a glance in a 32-item list. */
/* Measured, as a SURFACE against the #0F2133 sidebar and as a text background:
     #D55E00  4.22:1 surface / 3.87:1 white text  -- text fails AA
     #A84A00  2.83:1 surface / 5.76:1 white text  -- surface fails 1.4.11's 3:1
     #FFF4EC 15.09:1 surface / 7.06:1 #8A3D00     -- both pass comfortably
   A pale warm plate is the brightest thing available in a dark sidebar, so it
   is also the strongest highlight, and it still carries the logo's hue (25.3
   deg). Darkening the saturated fill trades a text failure for a surface one
   and makes the box sink into the navy. */
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked){
  background:#FFF4EC !important;
  border-left-color:var(--dpx-brand) !important;
  color:#8A3D00 !important;
  font-weight:var(--dpx-w-semibold) !important;
  box-shadow:none !important; }
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked):hover{
  background:#FFE8D6 !important; }
/* Streamlit renders each option's text in a <p> inside the label, and that <p>
   carries its own weight and size. Setting them on the label alone changes
   nothing visible -- measured: label 500, inner <p> still 400. */
section[data-testid="stSidebar"] .stRadio > div > label p,
section[data-testid="stSidebar"] .stRadio > div > label [data-testid="stMarkdownContainer"]{
  font-weight:var(--dpx-w-medium) !important;
  font-size:.8125rem !important;
  color:inherit !important;
  margin:0 !important; }
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) p,
section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) [data-testid="stMarkdownContainer"]{
  font-weight:var(--dpx-w-semibold) !important;
  color:#8A3D00 !important; }
section[data-testid="stSidebar"] .stRadio > div > label > div:first-child{ display:none !important; }
section[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"]{ display:none !important; }
section[data-testid="stSidebar"] hr{
  border-color:rgba(255,255,255,.1) !important; margin:10px 0 !important; }

/* Page header: a rule, not a floating card. */
.dpx-page-header{
  background:transparent; border-bottom:1px solid var(--dpx-border);
  border-radius:0; box-shadow:none;
  padding:0 0 var(--dpx-s3) 0; margin:0 0 var(--dpx-s6) 0;
  display:flex; align-items:center; gap:var(--dpx-s3); }
/* A replaced element's baseline is its BOTTOM margin edge, so under
   align-items:baseline the mark's bottom landed on the title's baseline and it
   floated above the cap height, making the header taller than its own text.
   The title and subtitle keep their baseline relationship in their own row. */
.dpx-page-header-text{ display:flex; align-items:baseline; gap:var(--dpx-s3);
  min-width:0; flex-wrap:wrap; }
.dpx-page-header-title{
  font-size:1.5rem; font-weight:var(--dpx-w-semibold);
  color:var(--dpx-text-primary); line-height:1.2; letter-spacing:-.012em; }
.dpx-page-header-sub{ font-size:.8125rem; color:var(--dpx-text-muted); }
.dpx-eyebrow{
  font-size:.6875rem; font-weight:var(--dpx-w-semibold); letter-spacing:.08em;
  text-transform:uppercase; color:var(--dpx-text-muted); }

/* Tabs: flat baseline with a 2px underline, not browser chrome. */
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  background:transparent !important; border-radius:0 !important;
  border-bottom:1px solid var(--dpx-border) !important;
  padding:0 !important; gap:var(--dpx-s4) !important; }
[data-testid="stTabs"] [data-baseweb="tab"]{
  background:transparent !important; border:none !important;
  border-bottom:2px solid transparent !important; border-radius:0 !important;
  color:var(--dpx-text-muted) !important;
  font-size:.875rem !important; font-weight:var(--dpx-w-medium) !important;
  padding:var(--dpx-s2) 0 !important; }
[data-testid="stTabs"] [data-baseweb="tab"]:hover{ color:var(--dpx-text-primary) !important; }
[data-testid="stTabs"] [aria-selected="true"][data-baseweb="tab"]{
  color:var(--dpx-text-primary) !important;
  border-bottom-color:var(--dpx-brand) !important;
  font-weight:var(--dpx-w-semibold) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{ display:none !important; }

/* Buttons: flat, no gradient, no glow. */
.stButton > button[kind="primary"]{
  background:var(--dpx-accent) !important; color:var(--dpx-text-invert) !important;
  font-weight:var(--dpx-w-semibold) !important; font-size:.875rem !important;
  border:1px solid var(--dpx-accent) !important;
  border-radius:var(--dpx-r-md) !important; box-shadow:none !important;
  padding:6px 14px !important; }
.stButton > button[kind="primary"]:hover{
  background:var(--dpx-accent-hover) !important; border-color:var(--dpx-accent-hover) !important; }
.stButton > button[kind="secondary"],.stDownloadButton > button{
  background:var(--dpx-surface) !important; color:var(--dpx-text-primary) !important;
  border:1px solid var(--dpx-border-strong) !important;
  border-radius:var(--dpx-r-md) !important;
  font-weight:var(--dpx-w-medium) !important; font-size:.875rem !important;
  box-shadow:none !important; padding:6px 14px !important; }
.stButton > button[kind="secondary"]:hover,.stDownloadButton > button:hover{
  background:var(--dpx-surface-sunken) !important;
  border-color:var(--dpx-text-secondary) !important; }
.stButton > button:focus-visible,.stDownloadButton > button:focus-visible{
  outline:2px solid var(--dpx-border-focus) !important; outline-offset:2px !important; }

/* Inputs and controls: desktop density, not touch targets. */
[data-testid="stMain"] input[type="number"],[data-testid="stMain"] input[type="text"],
[data-testid="stMain"] .stNumberInput input,[data-testid="stMain"] .stTextInput input{
  color:var(--dpx-text-primary) !important; background:var(--dpx-surface) !important;
  border:1px solid var(--dpx-border-strong) !important;
  border-radius:var(--dpx-r-md) !important; font-size:.875rem !important;
  min-height:34px !important; }
[data-testid="stMain"] .stSelectbox div[data-baseweb="select"]{ min-height:34px !important; }
[data-testid="stMain"] .stSelectbox [data-baseweb="select"] div,
[data-testid="stMain"] .stMultiSelect [data-baseweb="select"] div{
  color:var(--dpx-text-primary) !important; }

/* Metrics: a readout. */
[data-testid="stMetric"]{
  background:var(--dpx-surface) !important; border:1px solid var(--dpx-border) !important;
  border-radius:var(--dpx-r-md) !important; padding:var(--dpx-s3) var(--dpx-s4) !important; }
[data-testid="stMetricLabel"]{
  color:var(--dpx-text-muted) !important; font-size:.75rem !important;
  font-weight:var(--dpx-w-medium) !important;
  letter-spacing:.04em !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"]{
  color:var(--dpx-text-primary) !important; font-size:1.5rem !important;
  font-weight:var(--dpx-w-semibold) !important; }

/* Expanders and figures. Figures sit flush on white with a hairline, matching
   matplotlib's savefig.facecolor. */
[data-testid="stExpander"]{
  border:1px solid var(--dpx-border) !important; border-radius:var(--dpx-r-md) !important;
  box-shadow:none !important; background:var(--dpx-surface) !important; }
[data-testid="stImage"] img,[data-testid="stPlotlyChart"]{
  background:var(--dpx-surface) !important; border:1px solid var(--dpx-border) !important;
  border-radius:var(--dpx-r-md) !important; }

/* Tags: rectangular. */
.dpx-tag{
  display:inline-block; border-radius:var(--dpx-r-sm); padding:1px 7px;
  font-size:.75rem; font-weight:var(--dpx-w-medium); line-height:1.5;
  border:1px solid transparent; }
.dpx-tag--complete{ background:var(--dpx-success-bg); color:var(--dpx-success); border-color:#C5DFD0; }
.dpx-tag--warning { background:var(--dpx-warning-bg); color:var(--dpx-warning); border-color:#E8D5AC; }
.dpx-tag--danger  { background:var(--dpx-danger-bg);  color:var(--dpx-danger);  border-color:#E8C6C2; }
.dpx-tag--idle    { background:var(--dpx-surface-sunken); color:var(--dpx-text-muted); border-color:var(--dpx-border); }
.dpx-tag--neutral { background:var(--dpx-accent-weak); color:var(--dpx-accent); border-color:#CDD7E3; }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """
    The sidebar lockup.

    `app.py` has imported this for as long as it has existed, inside a
    `try/except ImportError` whose fallback is bare amber text. The function was
    never defined, so the fallback always fired. Defining it is the fix.
    """
    # The laboratory's own logo when it is available, the drawn mark otherwise.
    # The raster is scaled to 30px here rather than shown at its native 1254px:
    # the earlier 130px treatment with a coloured drop-shadow was most of what
    # made the page look unserious.
    # The file's own wordmark is NOT used. Measured: the letters "Dro" have a
    # median value of RGB(221,221,221) against a 254 ground, so the logo reads
    # as "Phenix" with a ghost in front of it on any light surface, and at 30px
    # its cap height is under 4px. The name is set in live type instead.
    glyph = glyph_img(28, chip=True)
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;'
        'padding:18px 14px 14px;border-bottom:1px solid rgba(255,255,255,.1);'
        'margin-bottom:10px;color:#FFFFFF;">'
        + glyph
        + '<div style="line-height:1.15;">'
          '<div style="font-size:.95rem;font-weight:600;color:#FFFFFF;'
          'letter-spacing:-.005em;">DroPhenix</div>'
          '<div style="font-size:.6875rem;color:#94A3B1;letter-spacing:.06em;">'
          'Phenomics workbench</div>'
          '</div></div>',
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str = "", extra: str = "",
                  raster: bool = False) -> None:
    """
    The page header: a title, an optional subtitle, and a rule.

    The tagline that used to sit above every title on all 37 pages is gone.
    Promotional copy inside an instrument reads badly to a reviewer, and the
    page title alone is what the reader needs.
    """
    bits = [f'<div class="dpx-page-header-title">{title}</div>']
    if subtitle:
        bits.append(f'<div class="dpx-page-header-sub">{subtitle}</div>')
    if extra:
        bits.append(f'<div class="dpx-page-header-sub">{extra}</div>')
    lead = ""
    if raster or USE_RASTER_MARK:
        # The CROPPED glyph, not the full three-band logo: at 28px the full mark
        # gives its emblem about 13px of height for seven motifs and resolves
        # into a smudge. The crop keeps the ring, the fly and the flame.
        lead = glyph_img(26, chip=False)
    st.markdown(
        '<div class="dpx-page-header">' + lead
        + '<div class="dpx-page-header-text">' + "".join(bits) + '</div></div>',
        unsafe_allow_html=True,
    )


def chip(text: str, style: str = "pathway") -> str:
    cls = {"pathway": "neutral", "rescue": "complete",
           "warning": "warning", "danger": "danger"}.get(style, "neutral")
    return f'<span class="dpx-tag dpx-tag--{cls}">{text}</span>'


def status_badge(text: str, state: str = "idle") -> str:
    cls = state if state in {"complete", "warning", "danger", "idle"} else "idle"
    return f'<span class="dpx-tag dpx-tag--{cls}">{text}</span>'
