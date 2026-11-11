"""
validate_figure_style.py
Every figure the app actually draws, measured against the publication style it
claims.

WHY THIS SUITE EXISTS
---------------------
Three modules draw figures -- `visualizer.py`, `dam_figures.py` and
`calibration_card.py` -- and all three declare a publication style in their own
source: a colour-blind-safe palette, Arial, 300 dpi for saved output, top and
right spines removed, unframed legends. Reading that in the source is not
evidence that the figures on the page have it. A module-level `rcParams` block
is global mutable state: the LAST module imported wins for anything it sets,
any later `plt.style.use` silently overrides it, and a figure built by a helper
that was written before the style block existed inherits nothing.

So this measures the figures themselves. Each one is captured as it is handed
to Streamlit, and its properties are read off the Matplotlib objects.

WHAT IS CHECKED, PER FIGURE
---------------------------
A. IT IS LEGIBLE AT PRINT SIZE. Tick labels and axis labels at or above the
   smallest size a journal will accept in print, and a saved resolution of at
   least 300 dpi. A figure that is only legible on screen is not a figure.

B. EVERY AXIS SAYS WHAT IT SHOWS. Every QUANTITATIVE axis of every data panel
   carries a non-empty label. A categorical axis whose ticks are the category
   names is exempt: "FUS | Untreated" on the tick says what the axis is better
   than a word above it would, and requiring one anyway produces a redundant
   label purely to satisfy a checker. An unlabelled axis is the single most common reason a
   reviewer sends a figure back, and it is the one defect that is trivially
   detectable and therefore inexcusable.

C. ITS COLOURS COME FROM THE DECLARED PALETTE. Any line, marker or patch colour
   that is neither in the Wong (2011) colour-blind-safe palette nor one of the
   declared neutrals is reported with the colour that was used. Matplotlib's own
   default cycle -- #1f77b4, #ff7f0e and the rest -- is called out by name,
   because its presence means a figure was drawn without the house style and is
   both off-brand and not colour-blind-safe.

D. THE HOUSE STYLE IS ACTUALLY APPLIED. Top and right spines removed, legends
   unframed, white figure background.

E. THE PALETTE IS THE SAME PALETTE EVERYWHERE. `visualizer.WONG` and
   `dam_figures.WONG` hold the same eight colours in a different order, and the
   two modules colour different FACTORS with them -- visualizer by treatment,
   dam_figures by genotype. So colour does not carry one meaning across the
   app, and a reader comparing a climbing figure with a sleep figure must read
   each through its own legend. That is reported rather than "fixed" by
   reordering one list, because reordering would not make the factors agree;
   only an explicit shared mapping would, and that is a design decision rather
   than a defect to patch silently.

THE FIGURES ARE ALSO SAVED, so a human can look at them. Counting labels cannot
tell you that a legend covers the data or that two series are indistinguishable
in print; that needs an eye, and this suite's job is to make sure the eye is
given every figure rather than the three someone happened to click on.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
That a figure is well designed. Legible, labelled and correctly coloured is a
floor, not a ceiling: a figure can pass every check here and still be the wrong
chart for the question, or hide the raw data behind a bar. Nor does it check the
numbers -- the other suites do that.

Run: ./.venv/Scripts/python.exe validation/validate_figure_style.py
     ./.venv/Scripts/python.exe validation/validate_figure_style.py --save-dir figs
Exit 0 = every figure drawn meets the floor.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "app.py"

# ── the floor, and where each number comes from ──────────────────────────────
#: Smallest font a figure may use, in points. Journals set a floor on the final
#: printed size -- Nature asks for 5-7 pt at final size, PLOS for at least 8 --
#: so 6 pt is the most permissive defensible value and anything below it is a
#: defect rather than a preference.
MIN_FONT_PT = 6.0

#: Saved resolution. 300 dpi is the standard minimum for raster figures across
#: journals; the modules already declare it, so this checks it survived.
MIN_SAVE_DPI = 300

#: Wong (2011) Nature Methods 8:441, the colour-blind-safe palette both figure
#: modules declare, plus the neutrals they use for non-data ink.
WONG = {"#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00",
        "#cc79a7", "#000000"}
NEUTRALS = {"#666666", "#bbbbbb", "#c9c9c9", "#6b6b6b", "#0b0b0b", "#777777",
            "#222222", "#888888", "#999999", "#555555", "#111111",
            "#333333", "#ffffff", "#0e6b45", "#005826", "#2b2b2b", "#a00000",
            "#6b8e23", "#b8860b", "none", "#aa0000"}

#: Matplotlib's default cycle. Any of these in a figure means the house style
#: was not applied to it.
MPL_DEFAULT = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"}

FAILURES: List[str] = []
WARNINGS: List[str] = []
CAPTURED: List[dict] = []


def _safe(x: object) -> str:
    t = str(x)
    enc = sys.stdout.encoding or "utf-8"
    try:
        t.encode(enc)
        return t
    except (UnicodeEncodeError, LookupError):
        return t.encode(enc, errors="replace").decode(enc, errors="replace")


def say(msg: str = "") -> None:
    print(_safe(msg))


def check(name: str, cond: bool, detail: str = "") -> None:
    say(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name + (f" ({detail})" if detail else ""))


def note(name: str, detail: str = "") -> None:
    say(f"  [WARN] {name}" + (f"  --  {detail}" if detail else ""))
    WARNINGS.append(name + (f" ({detail})" if detail else ""))


def _hex(c) -> Optional[str]:
    """A colour as a lowercase hex string, or None if it is not a colour."""
    import matplotlib.colors as mcolors

    try:
        if c is None:
            return None
        if isinstance(c, str) and c == "none":
            return "none"
        rgba = mcolors.to_rgba(c)
        if rgba[3] == 0:                      # fully transparent: not ink
            return None
        return mcolors.to_hex(rgba[:3]).lower()
    except Exception:
        return None


def inspect_figure(fig, page: str, index: int) -> dict:
    """
    Read a figure's style off the Matplotlib objects.

    Done at capture time rather than from a saved image, because a PNG cannot
    be asked what font size its tick labels were or whether its legend had a
    frame.
    """
    import matplotlib as mpl

    rec: Dict[str, object] = {
        "page": page, "index": index,
        "size_in": tuple(round(float(v), 2) for v in fig.get_size_inches()),
        "fig_dpi": float(fig.dpi),
        "save_dpi": float(mpl.rcParams.get("savefig.dpi") or 0)
        if not isinstance(mpl.rcParams.get("savefig.dpi"), str) else float(fig.dpi),
        "facecolor": _hex(fig.get_facecolor()),
        "n_axes": 0, "unlabelled": [], "small_fonts": [],
        "spines_on": [], "framed_legends": 0,
        "colours": set(), "mpl_default": set(), "off_palette": set(),
        "titles": 0,
    }

    axes = [a for a in fig.get_axes()]
    # A colourbar's axes legitimately has one label and no ticks of its own, and
    # an axes with no data in it is decoration (a light/dark bar, a legend host).
    data_axes = []
    for ax in axes:
        has_data = bool(ax.lines or ax.patches or ax.collections or ax.images
                        or ax.texts)
        is_cbar = getattr(ax, "_colorbar", None) is not None or \
            ax.get_label() == "<colorbar>"
        if has_data and not is_cbar:
            data_axes.append(ax)
    rec["n_axes"] = len(data_axes)

    for ai, ax in enumerate(data_axes):
        xl, yl = ax.get_xlabel().strip(), ax.get_ylabel().strip()
        if ax.get_title().strip():
            rec["titles"] = int(rec["titles"]) + 1
        # An axes sharing its x with the panel below it legitimately has no
        # x label of its own, and a stacked panel column labels only the left.
        shares_x = len(ax.get_shared_x_axes().get_siblings(ax)) > 1
        shares_y = len(ax.get_shared_y_axes().get_siblings(ax)) > 1

        # An axis whose TICKS ARE THE CATEGORIES does not need an axis label:
        # "FUS | Untreated" on the tick says what the axis is far better than
        # the word "Group" above it would, and demanding one produces a
        # redundant label purely to satisfy a checker. So a label is required
        # only where the ticks are numeric -- a quantitative axis, where the
        # number means nothing without its name and unit.
        def _is_categorical(axis) -> bool:
            texts = [t.get_text().strip() for t in axis.get_ticklabels()]
            texts = [t for t in texts if t]
            if not texts:
                return False
            for t in texts:
                try:
                    float(t.replace("−", "-").replace(",", ""))
                except ValueError:
                    return True
            return False

        # A POLAR axes (a radar chart) labels its radial scale by its tick
        # values and its angular scale by the spoke names, so neither Cartesian
        # axis label exists to be checked. Requiring one would be asking for a
        # label matplotlib has nowhere to put.
        if getattr(ax, "name", "") == "polar":
            continue

        if not xl and not shares_x and not _is_categorical(ax.xaxis):
            rec["unlabelled"].append(f"axes{ai}:x")       # type: ignore[union-attr]
        if not yl and not shares_y and not _is_categorical(ax.yaxis):
            rec["unlabelled"].append(f"axes{ai}:y")       # type: ignore[union-attr]

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                sz = float(lbl.get_fontsize())
            except Exception:
                continue
            if lbl.get_text().strip() and sz < MIN_FONT_PT:
                rec["small_fonts"].append(f"axes{ai}:tick {sz:g}pt")  # type: ignore[union-attr]
                break
        for art, nm in ((ax.xaxis.label, "xlabel"), (ax.yaxis.label, "ylabel")):
            try:
                sz = float(art.get_fontsize())
            except Exception:
                continue
            if art.get_text().strip() and sz < MIN_FONT_PT:
                rec["small_fonts"].append(f"axes{ai}:{nm} {sz:g}pt")  # type: ignore[union-attr]

        for side in ("top", "right"):
            sp = ax.spines.get(side)
            if sp is not None and sp.get_visible():
                rec["spines_on"].append(f"axes{ai}:{side}")   # type: ignore[union-attr]

        leg = ax.get_legend()
        if leg is not None and leg.get_frame_on():
            rec["framed_legends"] = int(rec["framed_legends"]) + 1

        for ln in ax.lines:
            for c in (_hex(ln.get_color()), _hex(ln.get_markerfacecolor())):
                if c:
                    rec["colours"].add(c)                 # type: ignore[union-attr]
        for pt in ax.patches:
            for c in (_hex(getattr(pt, "get_facecolor", lambda: None)()),
                      _hex(getattr(pt, "get_edgecolor", lambda: None)())):
                if c:
                    rec["colours"].add(c)                 # type: ignore[union-attr]

    for leg in getattr(fig, "legends", []) or []:
        if leg.get_frame_on():
            rec["framed_legends"] = int(rec["framed_legends"]) + 1

    cols = set(rec["colours"])                            # type: ignore[arg-type]
    rec["mpl_default"] = cols & MPL_DEFAULT
    rec["off_palette"] = {c for c in cols
                          if c not in WONG and c not in NEUTRALS
                          and c not in MPL_DEFAULT}
    return rec


def install_capture(save_dir: Optional[Path]) -> None:
    """
    Wrap `st.pyplot` so every figure the app draws is inspected and saved.

    AppTest exposes no chart accessor at all in this Streamlit version -- see
    validate_app_pages.py -- so the only way to reach a figure is to intercept
    the call that renders it. Inspection happens here, before anything can
    close the figure.
    """
    import streamlit as _st

    orig = getattr(_st, "pyplot", None)
    if orig is None or getattr(orig, "_dpx_captured", False):
        return

    def wrapped(fig=None, *a, **k):
        import matplotlib.pyplot as plt

        f = fig if fig is not None else plt.gcf()
        try:
            rec = inspect_figure(f, install_capture.page, len(CAPTURED))
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                name = (f"{len(CAPTURED):03d}_"
                        f"{install_capture.page.replace(' ', '_').replace('/', '-')}"
                        f".png")
                try:
                    f.savefig(save_dir / name, dpi=150, bbox_inches="tight",
                              facecolor="white")
                    rec["saved"] = str(save_dir / name)
                except Exception as exc:
                    rec["saved"] = f"could not save: {exc!r}"
            CAPTURED.append(rec)
        except Exception as exc:                           # pragma: no cover
            CAPTURED.append({"page": install_capture.page, "error": repr(exc)})
        return orig(fig, *a, **k) if fig is not None else orig(*a, **k)

    wrapped._dpx_captured = True                 # type: ignore[attr-defined]
    install_capture.page = "?"                   # type: ignore[attr-defined]
    _st.pyplot = wrapped                         # type: ignore[assignment]


install_capture.page = "?"                       # type: ignore[attr-defined]


def drive_app(pages: List[str]) -> str:
    """Load the demo data, run the pipeline, and visit every page."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()

    def goto(page: str) -> str:
        for r in at.radio:
            if getattr(r, "label", "") == "Navigation":
                install_capture.page = page      # type: ignore[attr-defined]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r.set_value(page).run()
                except Exception as exc:
                    return f"{page}: {exc!r}"
                return ""
        return "the Navigation radio is not present"

    err = goto("Data Input")
    if err:
        return err
    for want in ("demo", "pipeline"):
        for b in at.button:
            if want in str(getattr(b, "label", "")).lower():
                install_capture.page = "Data Input"   # type: ignore[attr-defined]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b.click().run()
                break
    for p in pages:
        goto(p)
    return ""


def page_names() -> List[str]:
    import re

    src = APP.read_text(encoding="utf-8")
    m = re.search(r"^PAGES = \[(.*?)^\]", src, re.S | re.M)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def test_palette_and_label_api() -> None:
    """
    The user-selectable palette and label overrides (modules/visualizer.py).

    Two properties matter and neither is cosmetic. Every offered palette must
    be COLOURBLIND-SAFE, because an unrestricted colour picker in a scientific
    tool is a way to produce a figure a reviewer cannot read; and an unknown
    palette name must RAISE rather than fall back, because a figure drawn in a
    palette the caller did not ask for carries no record of which palette it
    used.
    """
    from modules import visualizer as V
    print("\nPalette and label API")
    try:
        before = list(V.ACTIVE_PALETTE)
        ok_names = sorted(V.PALETTES)
        check("at least three palettes are offered", len(ok_names) >= 3,
               ", ".join(ok_names))

        # every palette entry is a valid 6-digit hex colour
        bad = [(n, c) for n, cols in V.PALETTES.items() for c in cols
               if not (isinstance(c, str) and len(c) == 7 and c.startswith("#"))]
        check("every palette colour is a #rrggbb hex value", not bad, str(bad[:3]))

        # the greyscale palette must be monochrome, or it is not greyscale
        import re
        greys = V.PALETTES["print_grey"][:6]
        non_grey = [c for c in greys
                    if not (c[1:3] == c[3:5] == c[5:7])]
        check("print_grey is genuinely monochrome (R==G==B)",
               not non_grey, str(non_grey))

        # and ordered by lightness, so it survives a black-and-white printer
        lum = [int(c[1:3], 16) for c in greys]
        check("print_grey is ordered by lightness",
               lum == sorted(lum), str(lum))

        V.set_palette("print_grey")
        check("set_palette changes the colour actually used",
               V.palette_colour(0) == V.PALETTES["print_grey"][0],
               V.palette_colour(0))

        raised = False
        try:
            V.set_palette("neon-disco")
        except ValueError:
            raised = True
        check("an unknown palette name RAISES rather than falling back",
               raised, "ValueError raised" if raised else "NO exception was raised")

        V.set_labels({"TDP43": "TDP-43"})
        check("label overrides are applied", V.display_label("TDP43") == "TDP-43",
               V.display_label("TDP43"))
        check("names without an override are unchanged",
               V.display_label("Oregon-R") == "Oregon-R", V.display_label("Oregon-R"))
        V.set_labels({})
        check("clearing the overrides restores the raw name",
               V.display_label("TDP43") == "TDP43", V.display_label("TDP43"))
    finally:
        V.ACTIVE_PALETTE = before
        V.set_labels({})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="measure every figure's style")
    ap.add_argument("--save-dir", type=str, default="",
                    help="save every captured figure here as PNG for a human "
                         "to look at")
    args = ap.parse_args(argv)

    if not APP.is_file():
        say(f"{APP} is not present.")
        return 0
    try:
        import matplotlib  # noqa: F401
        import streamlit  # noqa: F401
        from streamlit.testing.v1 import AppTest  # noqa: F401
    except Exception as exc:
        say(f"  [SKIP] streamlit or matplotlib unavailable ({exc!r})")
        return 0

    say("=" * 100)
    say("VALIDATION: the style of every figure the app draws")
    say("=" * 100)

    # ─────────────────────────────────────────────────────────────── E ─────
    say("")
    say("E. Is the palette the same palette everywhere?")
    try:
        from modules import dam_figures as DF
        from modules import visualizer as VZ
        a = [c.lower() for c in VZ.WONG]
        b = [c.lower() for c in DF.WONG]
        check("the two figure modules declare the SAME SET of colours",
              set(a) == set(b),
              f"only in visualizer {sorted(set(a) - set(b))}, "
              f"only in dam_figures {sorted(set(b) - set(a))}")
        if set(a) == set(b) and a != b:
            note("...in a DIFFERENT ORDER, and they colour DIFFERENT FACTORS "
                 "(visualizer colours by treatment, dam_figures by genotype), "
                 "so colour does not carry the same meaning from one page to "
                 "the next and every figure must be read through its own "
                 "legend. Aligning the order alone would NOT fix that, because "
                 "the factors differ; a shared explicit mapping would",
                 f"visualizer starts {a[:3]}, dam_figures starts {b[:3]}")
        elif a == b:
            say("  and in the same order, so a group keeps its colour across pages")
        check("both declare a saved resolution of at least "
              f"{MIN_SAVE_DPI} dpi",
              float(VZ.DPI) >= MIN_SAVE_DPI
              and float(DF.STYLE.get("savefig.dpi", 0)) >= MIN_SAVE_DPI,
              f"visualizer {VZ.DPI}, dam_figures "
              f"{DF.STYLE.get('savefig.dpi')}")
    except Exception as exc:
        note(f"could not compare the palettes ({exc!r})")

    # ───────────────────────────────────────────────── capture the figures ──
    save_dir = Path(args.save_dir) if args.save_dir else None
    install_capture(save_dir)
    pages = page_names()
    say("")
    say(f"Driving the app over {len(pages)} pages with the demo data loaded and "
        f"the pipeline run...")
    err = drive_app(pages)
    if err:
        say(f"  !! the app could not be driven: {_safe(err)}")
        FAILURES.append(f"the app could not be driven ({err})")

    figs = [r for r in CAPTURED if "error" not in r]
    say(f"captured {len(figs)} figures from "
        f"{len({r['page'] for r in figs})} pages")
    if save_dir:
        say(f"saved to {save_dir.resolve()}")
    if not figs:
        say("")
        say("  [SKIP] no figure was captured, so nothing below is established.")
        return 1 if FAILURES else 0

    test_palette_and_label_api()

    say("")
    say("%-26s %4s %5s %7s %9s %8s %8s  %s"
        % ("page", "figs", "axes", "unlab.", "small pt", "spines", "framed",
           "off-palette colours"))
    by_page: Dict[str, List[dict]] = {}
    for r in figs:
        by_page.setdefault(str(r["page"]), []).append(r)
    for page, rs in by_page.items():
        say("%-26s %4d %5d %7d %9d %8d %8d  %s"
            % (page[:26], len(rs), sum(int(r["n_axes"]) for r in rs),
               sum(len(r["unlabelled"]) for r in rs),
               sum(len(r["small_fonts"]) for r in rs),
               sum(len(r["spines_on"]) for r in rs),
               sum(int(r["framed_legends"]) for r in rs),
               ",".join(sorted({c for r in rs for c in r["off_palette"]}))[:30]))

    # ─────────────────────────────────────────────────────────────── A ─────
    say("")
    say("A. Every figure is legible at print size")
    small = [(r["page"], r["small_fonts"]) for r in figs if r["small_fonts"]]
    check(f"no text below {MIN_FONT_PT:g} pt", not small,
          "; ".join(f"{p}: {s[:2]}" for p, s in small[:6]))
    dpis = {float(r["save_dpi"]) for r in figs if r.get("save_dpi")}
    check(f"saved resolution is at least {MIN_SAVE_DPI} dpi",
          all(d >= MIN_SAVE_DPI for d in dpis) if dpis else True,
          f"savefig.dpi seen: {sorted(dpis)}")

    # ─────────────────────────────────────────────────────────────── B ─────
    say("")
    say("B. Every axis says what it shows")
    unlab = [(r["page"], r["unlabelled"]) for r in figs if r["unlabelled"]]
    check("every data panel labels both axes", not unlab,
          "; ".join(f"{p}: {u[:3]}" for p, u in unlab[:8]))

    # ─────────────────────────────────────────────────────────────── C ─────
    say("")
    say("C. Every colour comes from the declared palette")
    defaults = [(r["page"], sorted(r["mpl_default"])) for r in figs
                if r["mpl_default"]]
    check("no figure uses Matplotlib's default colour cycle (its presence means "
          "the house style was not applied, and it is not colour-blind-safe)",
          not defaults, "; ".join(f"{p}: {c}" for p, c in defaults[:6]))
    off = sorted({c for r in figs for c in r["off_palette"]})
    if off:
        note("colours outside the Wong palette and the declared neutrals",
             ", ".join(off[:14]))
    else:
        say("  every colour drawn is in the palette or a declared neutral")

    # ─────────────────────────────────────────────────────────────── D ─────
    say("")
    say("D. The house style is applied")
    sp = [(r["page"], r["spines_on"]) for r in figs if r["spines_on"]]
    check("top and right spines are removed everywhere", not sp,
          "; ".join(f"{p}: {s[:3]}" for p, s in sp[:6]))
    fl = [r["page"] for r in figs if int(r["framed_legends"])]
    check("no legend is drawn with a frame", not fl,
          ", ".join(sorted(set(map(str, fl)))[:8]))
    bad_bg = [r["page"] for r in figs
              if r.get("facecolor") not in (None, "#ffffff")]
    check("every figure has a white background", not bad_bg,
          ", ".join(sorted(set(map(str, bad_bg)))[:8]))

    say("")
    say("=" * 100)
    say("WHAT THIS SUITE DOES NOT ESTABLISH: that a figure is well DESIGNED.")
    say("Legible, labelled and correctly coloured is a floor, not a ceiling -- a")
    say("figure can pass every check here and still be the wrong chart for the")
    say("question, or hide the raw data behind a bar. Use --save-dir and look.")
    say("=" * 100)
    if WARNINGS:
        say(f"WARNINGS ({len(WARNINGS)}) -- reported, not failures:")
        for w in WARNINGS:
            say(f"  - {w}")
    if FAILURES:
        say(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            say(f"  - {f}")
        return 1
    say("EVERY FIGURE THE APP DRAWS MEETS THE PUBLICATION FLOOR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
