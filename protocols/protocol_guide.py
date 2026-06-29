"""
DroPhenix — Protocol Guide module
Provides two public functions:
  show_data_template()   — call at top of page_data_input()
  show_vial_protocol()   — call at top of page_recording()
Both functions are wrapped in st.expander so they don't disrupt
any existing UI below them.
"""
from __future__ import annotations
import base64
import io
from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).parent.parent / "assets"
_TEMPLATE_CSV = _ASSETS / "drophenix_data_template.csv"
_PROTOCOL_IMG = _ASSETS / "vial_setup_protocol.png"


# ── helper ────────────────────────────────────────────────────────────────────
def _img_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except Exception:
        return None


def _csv_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────
def show_data_template() -> None:
    """
    Adds a collapsible 'Data Template & Format Guide' section at the
    top of the Data Input page.  Zero impact on existing code below.
    """
    with st.expander("📋  Data Template & Format Guide", expanded=False):
        # ── Download buttons row ─────────────────────────────────────────────
        dl1, dl2, dl3 = st.columns(3)

        csv_b = _csv_bytes(_TEMPLATE_CSV)
        if csv_b:
            dl1.download_button(
                "⬇ Download CSV Template",
                csv_b,
                "drophenix_data_template.csv",
                "text/csv",
                key="tmpl_csv_dl",
                help="Pre-filled with 3 genotypes × 2 treatments × 2 sexes × 3 reps × 4 time-points",
            )
        else:
            dl1.warning("Template CSV not found in assets/")

        dl2.markdown(
            '<a href="#" style="pointer-events:none;opacity:.6;font-size:.8rem;">'
            'Demo data: use Data Input → Demo Dataset button</a>',
            unsafe_allow_html=True,
        )

        img_b = _img_bytes(_PROTOCOL_IMG)
        if img_b:
            dl3.download_button(
                "⬇ Download Protocol PDF",
                img_b,
                "vial_setup_protocol.png",
                "image/png",
                key="tmpl_img_dl",
            )

        st.divider()

        # ── Column specification ─────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.95rem;font-weight:800;color:#0D2137;'
            'margin-bottom:.5rem;">Required CSV Columns</p>',
            unsafe_allow_html=True,
        )
        col_data = {
            "Column": [
                "Genotype", "Treatment", "Sex", "Replicate",
                "Time", "Count_climbed", "n_total_flies",
            ],
            "Type": [
                "text", "text", "text (Male/Female)", "integer",
                "integer (seconds)", "integer", "integer",
            ],
            "Example": [
                "Oregon-R_WT", "Spermine_1mM", "Male", "1",
                "30", "17", "22",
            ],
            "Notes": [
                "Consistent spelling across all rows",
                "Include Untreated / Vehicle control",
                "Case-sensitive: Male or Female",
                "Minimum 3 replicates per group",
                "e.g. 10, 20, 30, 60 — min 3 time-points",
                "Flies ABOVE threshold line",
                "Total flies in vial (climbed + not climbed)",
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

        st.divider()

        # ── Quick tips ───────────────────────────────────────────────────────
        st.markdown(
            '<p style="font-size:.9rem;font-weight:700;color:#0D2137;'
            'margin-bottom:.4rem;">Common errors to avoid</p>',
            unsafe_allow_html=True,
        )
        tips = [
            ("❌", "Inconsistent genotype names",
             '"Oregon-R_WT" and "OregonWT" are treated as two different genotypes'),
            ("❌", "Count_climbed > n_total_flies",
             "Impossible — DroPhenix will flag and reject this row"),
            ("❌", "M / F instead of Male / Female",
             "Sex column must use exactly 'Male' or 'Female'"),
            ("❌", "Only one sex group",
             "SDQ and sex comparison require both Male and Female"),
            ("❌", "Only 1 time-point per group",
             "AUC and t50 require ≥ 3 time-points"),
        ]
        for icon, title, detail in tips:
            st.markdown(
                f'<div style="background:#FFF5F5;border-left:3px solid #EF4444;'
                f'border-radius:0 6px 6px 0;padding:.4rem .8rem;margin:.3rem 0;'
                f'font-size:.82rem;color:#374151;">'
                f'<strong style="color:#991B1B;">{icon} {title}:</strong> {detail}</div>',
                unsafe_allow_html=True,
            )


def show_vial_protocol() -> None:
    """
    Adds a collapsible 'Vial Setup & Recording Protocol' section at the
    top of the Live Recording page.  Zero impact on existing code below.
    """
    with st.expander("🔬  Vial Setup & Recording Protocol", expanded=False):
        img_b = _img_bytes(_PROTOCOL_IMG)
        if img_b:
            st.image(img_b, use_container_width=True,
                     caption="DroPhenix Negative Geotaxis Vial Setup Protocol (SSSIHL)")
        else:
            st.warning("Protocol image not found. Place vial_setup_protocol.png in assets/ folder.")

        st.divider()

        # ── Protocol quick-reference cards ───────────────────────────────────
        st.markdown(
            '<p style="font-size:.95rem;font-weight:800;color:#0D2137;'
            'margin-bottom:.6rem;">Quick Reference</p>',
            unsafe_allow_html=True,
        )
        cards = [
            ("🧪", "Vial Preparation",
             "Clear glass vials 25mm × 95mm. 15–25 flies per vial. "
             "Mark threshold line at exactly 8 cm from bottom. "
             "Age-match all flies to 3–7 days post-eclosion."),
            ("🏷️", "Labelling",
             "Label format: Genotype_Treatment_Sex_Replicate\n"
             "Example: TDP43_Spermine_Male_Rep1\n"
             "Arrange in rack: Row = Genotype, Column = Treatment×Rep"),
            ("🎬", "Recording",
             "Settle 60 s → Tap 3× → Start timer immediately → "
             "Count above threshold at 10 s, 20 s, 30 s, 60 s → "
             "Rest 5 min → Repeat ≥ 3 times (= 3 Replicates)"),
            ("💊", "Drug Treatment",
             "Mix compound in fly food. Chronic exposure 7–14 days. "
             "Vehicle: 1% DMSO in food (if DMSO-soluble). "
             "Label Treatment as compound name + concentration."),
        ]
        r1, r2 = st.columns(2)
        pairs = [(r1, cards[0]), (r2, cards[1]), (r1, cards[2]), (r2, cards[3])]
        for col, (icon, title, body) in pairs:
            with col:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #C5D7F0;'
                    f'border-top:3px solid #F0A500;border-radius:8px;'
                    f'padding:.85rem 1rem;margin-bottom:.7rem;">'
                    f'<div style="font-size:.9rem;font-weight:700;color:#0D2137;'
                    f'margin-bottom:.35rem;">{icon} {title}</div>'
                    f'<div style="font-size:.79rem;color:#374151;line-height:1.65;">'
                    + body.replace("\n", "<br>") + "</div></div>",
                    unsafe_allow_html=True,
                )

        # ── Download ─────────────────────────────────────────────────────────
        if img_b:
            st.download_button(
                "⬇ Download Protocol Diagram (PNG)",
                img_b,
                "vial_setup_protocol.png",
                "image/png",
                key="vial_proto_dl",
            )
        csv_b = _csv_bytes(_TEMPLATE_CSV)
        if csv_b:
            st.download_button(
                "⬇ Download CSV Template",
                csv_b,
                "drophenix_data_template.csv",
                "text/csv",
                key="vial_csv_dl",
            )
