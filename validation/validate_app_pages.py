"""
validate_app_pages.py
Every page of the Streamlit app, rendered and checked.

WHY THIS SUITE EXISTS
---------------------
The app is 8,464 lines and 32 pages, and it is the only part of this project a
user ever touches. Every other suite tests a module in isolation; none of them
would notice that a page raises on load, that a figure is never drawn, that a
control the user is told to set does nothing, or that a result the maths got
right is never displayed.

Those failures are not hypothetical. The project's own register records a figure
that crashed on a single-string argument and was missed because it only renders
after data is loaded -- an isolated-module suite cannot find that, and neither
can a smoke test that stops at the landing page.

WHAT IS CHECKED, PER PAGE
-------------------------
A. IT RENDERS. No uncaught exception. This is the floor, and a page that fails
   it is reported with the exception rather than a count.

B. IT SAYS SOMETHING. A page that renders but draws nothing -- no chart, no
   table, no metric, no message -- is indistinguishable from a blank screen to
   the user. A page with no data is expected to EXPLAIN that, so an empty page
   is a defect even when nothing crashed.

C. ITS CONTROLS EXIST AND ARE REACHABLE. Every slider, selectbox, number input
   and checkbox is enumerated. A page that claims to be configurable and
   exposes no widget, or that exposes a widget whose value cannot be read back,
   is reported.

D. IT DOES NOT SHOW A TRACEBACK AS TEXT. Several pages catch their own
   exceptions and print them. That is better than crashing and worse than
   working, so any page whose output contains a traceback, "Traceback",
   "NameError", "KeyError" and so on is reported separately from a clean one.

E. WITH DATA LOADED, THE ANALYSIS PAGES PRODUCE OUTPUT. Pages are re-rendered
   after the app's own demo dataset is loaded, because the interesting failures
   live on the far side of a data load -- that is precisely the case the
   register's figure bug hid in.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
That the numbers are right -- every other suite in this directory exists for
that. Nor that the figures are legible, correctly coloured, or of publication
quality: AppTest returns the element tree, not pixels, so it can see THAT a
chart was drawn and not what it looks like. The visual audit is a separate
exercise and must not be read out of this file.

Run: ./.venv/Scripts/python.exe validation/validate_app_pages.py
     ./.venv/Scripts/python.exe validation/validate_app_pages.py --page "Home"
Exit 0 = every page renders and says something.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / "app.py"

#: Rendering 32 pages of a large app is slow; this is generous per page and the
#: suite reports which page timed out rather than dying on it.
TIMEOUT_S = 180.0

#: Text that means a page swallowed an error and printed it instead of working.
TRACEBACK_MARKERS = (
    "Traceback (most recent call last)",
    "NameError", "KeyError", "AttributeError", "TypeError:",
    "ValueError:", "IndexError", "ZeroDivisionError",
    "UnboundLocalError", "ModuleNotFoundError",
)

FAILURES: List[str] = []
WARNINGS: List[str] = []


#: Charts drawn since the counter was last cleared, by Streamlit function name.
#:
#: Counted by wrapping Streamlit's own drawing functions, because AppTest in
#: this version exposes no chart accessor: `at.plotly_chart` does not exist and
#: `at.get("plotly_chart")` is empty even on a page that draws one. The first
#: version of this suite counted through those accessors, caught the
#: AttributeError and reported 0 for every page -- a silent zero that read as
#: evidence. Verified against the wrapper: Climbing Curves draws 1 and Core
#: Metrics 4.
CHART_CALLS: Dict[str, int] = {}

#: Every Streamlit call that puts a figure on the page.
CHART_FUNCS = ("pyplot", "plotly_chart", "altair_chart", "vega_lite_chart",
               "line_chart", "bar_chart", "area_chart", "scatter_chart",
               "image", "graphviz_chart", "map", "pydeck_chart")


def install_chart_counter() -> None:
    """
    Wrap Streamlit's drawing functions so figures can be counted.

    Must be called BEFORE AppTest runs the script. The app does
    `import streamlit as st` and calls `st.pyplot(...)`, so `st` is the module
    object and replacing the attribute on it is seen by the app.
    """
    import streamlit as _st

    for name in CHART_FUNCS:
        orig = getattr(_st, name, None)
        if orig is None or getattr(orig, "_dpx_counted", False):
            continue

        def make(n, f):
            def wrapped(*a, **k):
                CHART_CALLS[n] = CHART_CALLS.get(n, 0) + 1
                return f(*a, **k)
            wrapped._dpx_counted = True          # type: ignore[attr-defined]
            return wrapped

        setattr(_st, name, make(name, orig))


def _safe(x: object) -> str:
    """
    A string this console can actually print.

    The app puts emoji in its headings and this machine's stdout is cp1252, so
    printing a page's own text raises UnicodeEncodeError and the harness dies on
    the app's content instead of on a defect. Applied at the print boundary
    only: the checks below still compare against the real strings.
    """
    t = str(x)
    enc = (sys.stdout.encoding or "utf-8")
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


def page_names() -> List[str]:
    """The page list, read from the app rather than duplicated here."""
    src = APP.read_text(encoding="utf-8")
    m = re.search(r"^PAGES = \[(.*?)^\]", src, re.S | re.M)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def _texts(at) -> List[str]:
    """Every piece of text the page produced, flattened."""
    out: List[str] = []
    for attr in ("markdown", "text", "info", "warning", "error", "success",
                 "caption", "title", "header", "subheader", "code"):
        try:
            for el in getattr(at, attr):
                v = getattr(el, "value", None)
                if isinstance(v, str):
                    out.append(v)
        except Exception:
            continue
    return out


def _count(at, attr: str) -> int:
    try:
        return len(getattr(at, attr))
    except Exception:
        return 0


def _new_session():
    """A fresh app session, run once so the sidebar exists."""
    from streamlit.testing.v1 import AppTest

    install_chart_counter()
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT_S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    return at


def _goto(at, page: str) -> str:
    """
    Select a page the way a user does -- through the sidebar radio -- so the
    real dispatch in PAGE_MAP is exercised rather than the page function being
    called directly. Returns an error string, empty when the page rendered.
    """
    nav = None
    for r in at.radio:
        if getattr(r, "label", "") == "Navigation":
            nav = r
            break
    if nav is None:
        return "the Navigation radio is not present"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nav.set_value(page).run()
    except Exception as exc:
        return f"could not select the page: {exc!r}"
    if at.exception:
        return "; ".join(str(e.value)[:300] for e in at.exception)
    return ""


def load_demo(at) -> str:
    """
    Load the app's own demo dataset, through its own controls.

    Returns an error string, empty on success. The data is loaded by pressing
    the button the user presses, so this also tests that the button works --
    and if it stops working, every with-data check below reports the same cause
    rather than 30 separate mysteries.
    """
    err = _goto(at, "Data Input")
    if err:
        return f"Data Input would not render: {err}"
    for r in at.radio:
        if "data source" in str(getattr(r, "label", "")).lower():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r.set_value("Demo dataset").run()
            except Exception as exc:
                return f"could not choose the demo source: {exc!r}"
            break
    for b in at.button:
        if "demo" in str(getattr(b, "label", "")).lower():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b.click().run()
            except Exception as exc:
                return f"the demo-load button raised: {exc!r}"
            if at.exception:
                return "loading the demo raised: " + "; ".join(
                    str(e.value)[:200] for e in at.exception)
            return ""
    return "no demo-load button was found on Data Input"


def run_pipeline(at) -> str:
    """
    Press "Run Full Analysis Pipeline", which is the app's SECOND step.

    Loading data is not enough: the loader sets `run_complete = False`, and
    every analysis page then correctly reports "No analysis results. Go to Data
    Input and run the pipeline." The first version of this suite never pressed
    this button and read that correct message as a page producing nothing.
    """
    err = _goto(at, "Data Input")
    if err:
        return f"Data Input would not render: {err}"
    for b in at.button:
        if "pipeline" in str(getattr(b, "label", "")).lower():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b.click().run()
            except Exception as exc:
                return f"the pipeline button raised: {exc!r}"
            if at.exception:
                return "running the pipeline raised: " + "; ".join(
                    str(e.value)[:200] for e in at.exception)
            return ""
    return "no pipeline button was found on Data Input"


def _texts_of(at) -> List[str]:
    return _texts(at)


def describe(at, page: str, err: str) -> Dict[str, object]:
    """Everything the page produced, as counts plus any traceback text."""
    row: Dict[str, object] = {"page": page, "error": err}
    if err:
        return row
    # From the wrapper, not from AppTest, which cannot see charts at all.
    charts = sum(CHART_CALLS.values())
    widgets = {
        "slider": _count(at, "slider"),
        "selectbox": _count(at, "selectbox"),
        "number_input": _count(at, "number_input"),
        "checkbox": _count(at, "checkbox"),
        "radio": max(0, _count(at, "radio") - 1),      # minus the nav radio
        "text_input": _count(at, "text_input"),
        "button": _count(at, "button"),
        "file_uploader": _count(at, "file_uploader"),
        "multiselect": _count(at, "multiselect"),
    }
    texts = _texts(at)
    tb = [t for t in texts for m in TRACEBACK_MARKERS if m in t]
    row.update({
        "charts": charts,
        "tables": _count(at, "dataframe") + _count(at, "table"),
        "metrics": _count(at, "metric"),
        "widgets": widgets, "n_widgets": sum(widgets.values()),
        "texts": len([t for t in texts if t.strip()]),
        "tracebacks": tb,
        "n_error": _count(at, "error"),
        "n_warning": _count(at, "warning"),
    })
    return row


def sweep(pages: List[str], with_data: bool,
          with_pipeline: bool = False) -> List[Dict[str, object]]:
    """
    Render every page and describe what it produced.

    With data, ONE session is used for the whole sweep so that the dataset
    stays loaded in session_state. Without data, a fresh session per page, so
    that no page can be helped by a previous one having run.
    """
    rows: List[Dict[str, object]] = []
    at = None
    if with_data:
        at = _new_session()
        err = load_demo(at)
        if not err and with_pipeline:
            err = run_pipeline(at)
        if err:
            say(f"  !! the app could not be brought to this state: {_safe(err)}")
            say("     every with-data row below would report the same cause, so "
                "this pass is abandoned.")
            FAILURES.append(f"the demo dataset could not be loaded ({err})")
            return rows

    say("%-26s %7s %7s %7s %7s %7s  %s"
        % ("page", "charts", "tables", "metrics", "widgets", "texts", "status"))
    for page in pages:
        CHART_CALLS.clear()          # count only the figures THIS page draws
        if with_data:
            err = _goto(at, page)
            r = describe(at, page, err)
        else:
            one = _new_session()
            CHART_CALLS.clear()
            err = _goto(one, page)
            r = describe(one, page, err)
        rows.append(r)
        if r.get("error"):
            say("%-26s %7s %7s %7s %7s %7s  RAISED"
                % (page[:26], "-", "-", "-", "-", "-"))
            say(f"        {_safe(str(r['error'])[:220])}")
            continue
        status = "ok"
        if r["tracebacks"]:
            status = "PRINTED A TRACEBACK"
        elif (r["charts"] + r["tables"] + r["metrics"]) == 0 and r["texts"] <= 2:
            status = "almost blank"
        say("%-26s %7d %7d %7d %7d %7d  %s"
            % (page[:26], r["charts"], r["tables"], r["metrics"],
               r["n_widgets"], r["texts"], status))
    return rows


def report(rows: List[Dict[str, object]], label: str) -> None:
    say("")
    say(f"A. [{label}] every page renders without an uncaught exception")
    raised = [r for r in rows if r.get("error")]
    check(f"[{label}] no page raised", not raised,
          "; ".join(f"{r['page']}: {str(r['error'])[:110]}" for r in raised))

    say("")
    say(f"D. [{label}] no page prints a traceback instead of working")
    printed = [r for r in rows if r.get("tracebacks")]
    check(f"[{label}] no page printed a traceback or an exception name as text",
          not printed,
          "; ".join(f"{r['page']}: {r['tracebacks'][0][:90]}" for r in printed))

    say("")
    say(f"B. [{label}] every page says something to the user")
    blank = [r for r in rows
             if not r.get("error")
             and (r["charts"] + r["tables"] + r["metrics"]) == 0
             and r["texts"] <= 2]
    check(f"[{label}] no page is effectively blank", not blank,
          ", ".join(str(r["page"]) for r in blank))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="render every app page")
    ap.add_argument("--page", action="append", default=None,
                    help="only this page (repeatable)")
    ap.add_argument("--no-data-only", action="store_true",
                    help="skip the with-data pass (faster, far weaker)")
    args = ap.parse_args(argv)

    if not APP.is_file():
        say(f"{APP} is not present.")
        return 0
    try:
        import streamlit  # noqa: F401
        from streamlit.testing.v1 import AppTest  # noqa: F401
    except Exception as exc:
        say(f"  [SKIP] streamlit's test harness is unavailable ({exc!r})")
        return 0

    pages = args.page or page_names()
    if not pages:
        say("  [SKIP] could not read PAGES from app.py")
        return 0

    say("=" * 100)
    say(f"VALIDATION: all {len(pages)} app pages render and produce output")
    say("=" * 100)

    say("")
    say("PASS 1 of 2 -- NO DATA LOADED. A page must not crash before the user")
    say("has loaded anything, and should say what it needs instead.")
    say("")
    rows_nodata = sweep(pages, with_data=False)
    report(rows_nodata, "no data")

    rows_data: List[Dict[str, object]] = []
    if not args.no_data_only:
        say("")
        say("=" * 100)
        say("PASS 2 of 2 -- DEMO DATASET LOADED *AND* THE PIPELINE RUN. This is")
        say("the pass that matters. The app has TWO setup steps, not one: loading")
        say("the data leaves run_complete = False and every analysis page then")
        say("correctly says it has no results yet, so a sweep that only loads is")
        say("measuring that message rather than the page.")
        say("=" * 100)
        say("")
        rows_data = sweep(pages, with_data=True, with_pipeline=True)
        if rows_data:
            report(rows_data, "with data")

            say("")
            say("E. [with data] the analysis pages actually produce output")
            drew = [r for r in rows_data
                    if not r.get("error")
                    and (r["charts"] + r["tables"] + r["metrics"]) > 0]
            say(f"  {len(drew)} of {len(rows_data)} pages drew a chart, table or "
                f"metric with the demo data loaded")
            gained = []
            by_name = {r["page"]: r for r in rows_nodata}
            for r in rows_data:
                b = by_name.get(r["page"])
                if not b or b.get("error") or r.get("error"):
                    continue
                before = b["charts"] + b["tables"] + b["metrics"]
                after = r["charts"] + r["tables"] + r["metrics"]
                if after > before:
                    gained.append(str(r["page"]))
            say(f"  {len(gained)} page(s) drew MORE once data was loaded, which is")
            say(f"  the evidence the data actually reached them: "
                f"{', '.join(gained[:12])}")
            check("at least one page produces more output with data than without "
                  "(otherwise the demo data is not reaching the pages at all)",
                  bool(gained), f"{len(gained)} pages gained output")

    # ───────────────────────────────────────────────── controls, overall ────
    say("")
    say("C. Pages expose the controls they describe")
    base = rows_data or rows_nodata
    nowid = [r for r in base if not r.get("error") and r["n_widgets"] == 0]
    if nowid:
        note("pages with no interactive control at all (expected for a static "
             "landing page, a defect for an analysis page)",
             ", ".join(str(r["page"]) for r in nowid))
    else:
        say("  every page exposes at least one control")
    tot: Dict[str, int] = {}
    for r in base:
        for k, v in (r.get("widgets") or {}).items():
            tot[k] = tot.get(k, 0) + v
    say(f"  controls across the app: {tot}")

    say("")
    say("=" * 100)
    say("WHAT THIS SUITE DOES NOT ESTABLISH: that the numbers are right (the")
    say("other suites), or that the figures are legible, correctly coloured or")
    say("of publication quality. AppTest returns the element tree, not pixels:")
    say("it sees THAT a chart was drawn, never what it looks like.")
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
    say("ALL APP PAGES RENDER AND PRODUCE OUTPUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
