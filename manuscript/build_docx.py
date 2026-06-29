"""
build_docx.py — render DroPhenix_manuscript.md to a SUBMISSION-STANDARD Word
document (NAR style): single consistent font (Arial), continuous line numbers,
double-spaced body, justified text, embedded figures with legends, two tables,
numbered hanging references.

Run:  python manuscript/build_docx.py  ->  manuscript/DroPhenix_manuscript.docx
"""
import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = Path(__file__).resolve().parent
MD = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "DroPhenix_manuscript.md"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "DroPhenix_manuscript.docx"
FIGDIR = HERE / "figures"
FIG_IMG = {"Figure 1.": "Figure1_validation.png",
           "Figure 2.": "Figure2_innovation.png",
           "Figure 3.": "Figure3_tool_agreement.png"}

doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Arial"; normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
for h, sz in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11), ("Title", 17)):
    st = doc.styles[h]; st.font.name = "Arial"; st.font.color.rgb = RGBColor(0, 0, 0); st.font.size = Pt(sz)

# Continuous line numbers + US-Letter page + 1-inch margins (NAR review standard)
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, m, Inches(1))
sectPr = sec._sectPr
ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1")
ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "360")
# OOXML CT_SectPr child order: ... pgMar, paperSrc, pgBorders, lnNumType, pgNumType, cols ...
# Insert lnNumType immediately after pgMar (else schema-invalid).
pgmar = sectPr.find(qn("w:pgMar"))
if pgmar is not None:
    pgmar.addnext(ln)
else:
    sectPr.append(ln)

REF_MODE = [False]  # toggled when REFERENCES reached


def style_body(p, justify=True, double=True):
    pf = p.paragraph_format
    if double:
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_after = Pt(6)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_runs(p, text):
    for tok in re.split(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)", text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            p.add_run(tok[2:-2]).bold = True
        elif tok.startswith("*") and tok.endswith("*"):
            p.add_run(tok[1:-1]).italic = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        else:
            p.add_run(tok)


def add_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        run = t.rows[0].cells[i].paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(8.5)
    for brow in body:
        rc = t.add_row().cells
        for i, val in enumerate(brow[:len(header)]):
            add_runs(rc[i].paragraphs[0], val)
    for row in t.rows:
        for c in row.cells:
            for pa in c.paragraphs:
                pa.paragraph_format.space_after = Pt(1)
                pa.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                for run in pa.runs:
                    run.font.name = "Arial"; run.font.size = Pt(8.5)
    doc.add_paragraph()


lines = MD.read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    ln_txt = lines[i].rstrip()
    if not ln_txt.strip():
        i += 1; continue
    if ln_txt.lstrip().startswith("|"):
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i]); i += 1
        if len(block) >= 2:
            add_table(block)
        continue
    if ln_txt.startswith("# "):
        p = doc.add_paragraph(ln_txt[2:], style="Title"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif ln_txt.startswith("## "):
        if "REFERENCES" in ln_txt:
            REF_MODE[0] = True
        doc.add_paragraph(ln_txt[3:], style="Heading 1")
    elif ln_txt.startswith("### "):
        doc.add_paragraph(ln_txt[4:], style="Heading 2")
    elif ln_txt.startswith("> "):
        p = doc.add_paragraph(); r = p.add_run(ln_txt[2:].strip("*")); r.italic = True
        r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(90, 90, 90)
        p.paragraph_format.space_after = Pt(8)
    elif ln_txt.strip() == "---":
        pass
    else:
        m = re.match(r"^(\d+)\.\s+(.*)", ln_txt)
        if REF_MODE[0] and m:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            style_body(p, justify=False, double=True)
            add_runs(p, f"{m.group(1)}. {m.group(2)}")
        else:
            is_legend = ln_txt.startswith("**Figure")
            p = doc.add_paragraph(); add_runs(p, ln_txt)
            style_body(p, justify=not is_legend, double=not is_legend)
            for key, img in FIG_IMG.items():
                if ln_txt.startswith("**" + key) and (FIGDIR / img).exists():
                    fp = doc.add_paragraph(); fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    fp.add_run().add_picture(str(FIGDIR / img), width=Inches(6.3))
    i += 1

doc.save(str(OUT))
print("wrote", OUT)
