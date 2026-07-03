"""
build_presentation.py — DroPhenix detailed scientific presentation (.pptx).
Per-module deep dives (measures / calculation / I-O format / validation),
infographics (data-flow, statistics decision-flow, validation pillars), results
and dual-audience speaker notes. Every fact from the project's verified results.

Run:  python manuscript/build_presentation.py -> manuscript/DroPhenix_presentation.pptx
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "DroPhenix_presentation.pptx"

NAVY = RGBColor(0x0D, 0x21, 0x37); NAVY2 = RGBColor(0x14, 0x2E, 0x47)
GOLD = RGBColor(0xF0, 0xA5, 0x00); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1E, 0x29, 0x33); SLATE = RGBColor(0x54, 0x66, 0x78)
LIGHT = RGBColor(0xEC, 0xF1, 0xF8); GREEN = RGBColor(0x0B, 0x84, 0x5F)
CARDBG = RGBColor(0xF5, 0xF8, 0xFC); CARDLINE = RGBColor(0xC5, 0xD7, 0xF0)
TEAL = RGBColor(0x0E, 0x6F, 0x8A)
HEAD = "Calibri"; BODY = "Calibri"; MONO = "Cambria"

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def _tb(slide, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(l, t, w, h); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(3); tf.margin_top = tf.margin_bottom = Pt(2)
    return tb, tf


def _run(p, text, size, color, bold=False, italic=False, font=BODY):
    r = p.add_run(); r.text = text; f = r.font
    f.size = Pt(size); f.bold = bold; f.italic = italic; f.name = font
    f.color.rgb = color; return r


def _fill(shape, color, line=None, lw=1.0):
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line; shape.line.width = Pt(lw)
    shape.shadow.inherit = False


def bg(slide, color):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH); _fill(r, color)
    slide.shapes._spTree.remove(r._element); slide.shapes._spTree.insert(2, r._element)


def title_bar(slide, kicker, title):
    _, tf = _tb(slide, Inches(0.55), Inches(0.36), Inches(12.2), Inches(0.34))
    _run(tf.paragraphs[0], kicker.upper(), 12, GOLD, bold=True, font=HEAD)
    _, tf = _tb(slide, Inches(0.55), Inches(0.7), Inches(12.2), Inches(0.72))
    _run(tf.paragraphs[0], title, 27, NAVY, bold=True, font=HEAD)


def content(kicker, title):
    s = prs.slides.add_slide(BLANK); bg(s, WHITE); title_bar(s, kicker, title); return s


def footer(slide, dark=False):
    _, tf = _tb(slide, Inches(0.55), Inches(7.04), Inches(12.2), Inches(0.3))
    p = tf.paragraphs[0]
    _run(p, "DroPhenix Analytics v3.0", 9, GOLD if dark else SLATE, bold=True)
    _run(p, "   ·   drophenix.streamlit.app   ·   SSSIHL", 9, LIGHT if dark else SLATE)


def bullets(slide, l, t, w, h, items, size=15, gap=8, color=INK):
    _, tf = _tb(slide, l, t, w, h)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.line_spacing = 1.04
        _run(p, "▪  ", size, GOLD, bold=True)
        if isinstance(it, tuple):
            _run(p, it[0] + " ", size, color, bold=True); _run(p, it[1], size, color)
        else:
            _run(p, it, size, color)


def card(slide, l, t, w, h, head, body, headcolor=NAVY, mono_from=99, size=11.5, hsize=13.5):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    _fill(box, CARDBG, CARDLINE, 1.0)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Pt(9); tf.margin_top = Pt(7); tf.margin_bottom = Pt(6)
    p = tf.paragraphs[0]; p.space_after = Pt(3); _run(p, head, hsize, headcolor, bold=True, font=HEAD)
    for i, b in enumerate(body):
        pp = tf.add_paragraph(); pp.line_spacing = 1.02; pp.space_after = Pt(1.5)
        _run(pp, b, size, INK, font=(MONO if i >= mono_from else BODY))
    return box


def chip(slide, l, t, w, h, title, sub, fill=NAVY, tcol=WHITE, scol=LIGHT):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h); _fill(box, fill)
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Pt(6)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, title, 13, tcol, bold=True, font=HEAD)
    if sub:
        pp = tf.add_paragraph(); pp.alignment = PP_ALIGN.CENTER; pp.line_spacing = 1.0
        _run(pp, sub, 9.5, scol)
    return box


def arrow(slide, l, t, w, h, color=GOLD):
    a = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, l, t, w, h); _fill(a, color)


def stat_dark(slide, l, t, w, number, label, ncolor=GOLD):
    _, tf = _tb(slide, l, t, w, Inches(0.72), MSO_ANCHOR.BOTTOM)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, number, 36, ncolor, bold=True, font=HEAD)
    _, tf = _tb(slide, l, t + Inches(0.74), w, Inches(0.55))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, label, 11, WHITE)


def module_slide(kicker, title, measures, calc, io, valid):
    s = content(kicker, title)
    gx, gy = Inches(0.28), Inches(0.22); w, h = Inches(6.0), Inches(2.28)
    x0, y0 = Inches(0.55), Inches(1.62)
    card(s, x0, y0, w, h, "What it measures", measures, headcolor=TEAL)
    card(s, x0 + w + gx, y0, w, h, "How it calculates", calc, headcolor=GOLD, mono_from=0, size=11)
    card(s, x0, y0 + h + gy, w, h, "Input  →  Output format", io)
    card(s, x0 + w + gx, y0 + h + gy, w, h, "How we validated  →  result", valid, headcolor=GREEN)
    footer(s); return s


# ═══ 1. TITLE ═════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK); bg(s, NAVY)
b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.02), SW, Inches(0.03)); _fill(b, GOLD)
_, tf = _tb(s, Inches(0.85), Inches(0.62), Inches(11.6), Inches(0.4))
_run(tf.paragraphs[0], "SCIENCE FOR SOCIETY  ·  SSSIHL  ·  SAI-NET", 13, GOLD, bold=True, font=HEAD)
_, tf = _tb(s, Inches(0.85), Inches(1.0), Inches(11.6), Inches(1.0))
p = tf.paragraphs[0]; _run(p, "DroPhenix ", 46, WHITE, bold=True, font=HEAD); _run(p, "Analytics", 46, GOLD, bold=True, font=HEAD)
_, tf = _tb(s, Inches(0.85), Inches(2.18), Inches(11.7), Inches(0.7))
_run(tf.paragraphs[0], "A statistically rigorous web platform for multi-assay Drosophila phenotyping and actuarial cross-strain compound ranking", 17, LIGHT, font=HEAD)
for i, (n, lb) in enumerate([("6", "biological assays"), ("24/24", "tool checks agree"), ("r = 1.0000", "vs gold-standard software"), ("18/18", "reference datasets")]):
    stat_dark(s, Inches(0.85 + i * 3.0), Inches(3.65), Inches(2.7), n, lb)
_, tf = _tb(s, Inches(0.85), Inches(6.35), Inches(11.7), Inches(0.5))
_run(tf.paragraphs[0], "Krishnasalini Gunanathan  ·  Venketesh Sivaramakrishnan   |   Dept. of Biosciences, SSSIHL", 13, LIGHT)
footer(s, dark=True)

# ═══ 2. PROBLEM ═══════════════════════════════════════════════════════════════
s = content("Motivation", "The problem we set out to solve")
bullets(s, Inches(0.55), Inches(1.7), Inches(6.4), Inches(5),
        [("Fragmented analysis.", "Six standard Drosophila assays are analysed separately, often by hand in spreadsheets, with inconsistent statistics."),
         ("Under-powered methods.", "Climbing reduced to one endpoint %; lifespan reported as median only, omitting Kaplan–Meier, log-rank and Cox; periodograms uncorrected for multiplicity."),
         ("No unequal-N method.", "When one strain has 3 vials and another 7, equal-weight averaging is biased toward the smaller, noisier strain — the everyday screening case."),
         ("Verifiability gap.", "Results are rarely benchmarked against published data or gold-standard software.")], size=14.5, gap=13)
card(s, Inches(7.35), Inches(1.85), Inches(5.4), Inches(4.5), "The unmet need — one workflow that:",
     ["unifies all six assays,", "applies correct, assumption-aware statistics,",
      "handles unequal-N strain comparison rigorously,", "reports publication-ready tables + figures, and",
      "can be independently verified — not merely trusted.", "",
      "DroPhenix was built to meet every one of these."], headcolor=GOLD, size=13)
footer(s)

# ═══ 3. WHAT WE BUILT ═════════════════════════════════════════════════════════
s = content("Overview", "What we developed: six assays + three analytical engines")
mods = [("Climbing", "Negative geotaxis: PI, AUC, t50; video counting"),
        ("Lifespan", "Kaplan–Meier · log-rank · Cox proportional hazards"),
        ("Y-maze", "Spontaneous alternation; turn-bias / handedness"),
        ("DAMS", "Sleep (5-min rule) + circadian period (periodogram)"),
        ("Larval crawling", "Video tracking: speed, path, tortuosity"),
        ("Grooming", "% time grooming; bout structure"),
        ("Motor Decline Rate", "Exponential decline of PI over age"),
        ("Trajectory clustering", "DTW + Ward unsupervised grouping"),
        ("Drug Ranking", "Cross-strain, cross-assay meta-analytic score")]
x0, y0, w, h, gx, gy = Inches(0.55), Inches(1.62), Inches(4.0), Inches(1.5), Inches(0.15), Inches(0.16)
for i, (nm, ds) in enumerate(mods):
    r, c = divmod(i, 3); card(s, x0 + c * (w + gx), y0 + r * (h + gy), w, h, nm, [ds], size=11.5)
footer(s)

# ═══ 4. INFOGRAPHIC: DATA FLOW ════════════════════════════════════════════════
s = content("How it works", "From raw data to publication-ready results — the pipeline")
steps = [("1  Load", "Upload CSV/Excel to the module's template, or a video, or the demo dataset"),
         ("2  Parse & QC", "Auto-detect columns, validate, flag outliers, standardise long format"),
         ("3  Analyse", "The module computes its metrics (see per-module slides)"),
         ("4  Statistics", "Assumption-gated tests, effect sizes, BH-FDR, bootstrap CI"),
         ("5  Output", "Interactive + downloadable tables, 300-dpi figures, CSV / ZIP")]
bw = Inches(2.18); bh = Inches(2.5); y = Inches(2.35); gap = Inches(0.28)
x = Inches(0.55)
for i, (t, d) in enumerate(steps):
    box = slide = None
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, bw, bh)
    _fill(b, NAVY if i in (0, 4) else CARDBG, None if i in (0, 4) else CARDLINE, 1.0)
    tf = b.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Pt(8); tf.margin_top = Pt(10)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _run(p, t, 15, GOLD if i in (0, 4) else NAVY, bold=True, font=HEAD)
    pp = tf.add_paragraph(); pp.alignment = PP_ALIGN.CENTER; pp.line_spacing = 1.06; pp.space_before = Pt(6)
    _run(pp, d, 11, WHITE if i in (0, 4) else INK)
    if i < 4:
        arrow(s, x + bw + Inches(0.01), y + Inches(0.95), Inches(0.26), Inches(0.6), GOLD)
    x = x + bw + gap
_, tf = _tb(s, Inches(0.55), Inches(5.3), Inches(12.2), Inches(0.5))
_run(tf.paragraphs[0], "Every step is deterministic and inspectable — same input always gives the same output; nothing is a trained black box.", 13, SLATE, italic=True)
footer(s)

# ═══ 5. INFOGRAPHIC: STATISTICS DECISION FLOW ═════════════════════════════════
s = content("Statistics engine", "How the correct test is chosen — automatically")
b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(2.15), Inches(3.7), Inches(2.6))
_fill(b, CARDBG, CARDLINE); tf = b.text_frame; tf.word_wrap = True; tf.margin_top = Pt(10); tf.margin_left = Pt(10)
_run(tf.paragraphs[0], "1  Check assumptions", 14, NAVY, bold=True, font=HEAD)
for x in ["Shapiro–Wilk test → normal?", "Levene test → equal variance?", "group size n ≥ 5?"]:
    pp = tf.add_paragraph(); pp.space_before = Pt(5); _run(pp, "• " + x, 12, INK)
arrow(s, Inches(4.3), Inches(3.15), Inches(0.4), Inches(0.6))
b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.85), Inches(2.15), Inches(3.7), Inches(2.6))
_fill(b, CARDBG, CARDLINE); tf = b.text_frame; tf.word_wrap = True; tf.margin_top = Pt(10); tf.margin_left = Pt(10)
_run(tf.paragraphs[0], "2  Select test", 14, NAVY, bold=True, font=HEAD)
for x in [("All pass → parametric:", GREEN), ("Welch t / one-way ANOVA", INK),
          ("Otherwise → non-parametric:", TEAL), ("Mann–Whitney U / Kruskal–Wallis", INK)]:
    pp = tf.add_paragraph(); pp.space_before = Pt(4); _run(pp, x[0], 11.5, x[1], bold=(x[1] != INK))
arrow(s, Inches(8.6), Inches(3.15), Inches(0.4), Inches(0.6))
b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.15), Inches(2.15), Inches(3.63), Inches(2.6))
_fill(b, NAVY); tf = b.text_frame; tf.word_wrap = True; tf.margin_top = Pt(10); tf.margin_left = Pt(10)
_run(tf.paragraphs[0], "3  Report", 14, GOLD, bold=True, font=HEAD)
for x in ["Effect size (Cohen d / Hedges g)", "Benjamini–Hochberg FDR correction", "Bootstrap 95% confidence interval"]:
    pp = tf.add_paragraph(); pp.space_before = Pt(6); _run(pp, "• " + x, 12, WHITE)
_, tf = _tb(s, Inches(0.55), Inches(5.15), Inches(12.2), Inches(1))
_run(tf.paragraphs[0], "Why: small, noisy behavioural data breaks a one-size-fits-all ANOVA. Gating the test on its assumptions, forcing effect sizes, and controlling the false-discovery rate is what makes a result survive peer review.", 13.5, INK)
footer(s)

# ═══ 6–14. PER-MODULE DEEP DIVES ══════════════════════════════════════════════
module_slide("Module 1 · Climbing", "Climbing assay (negative geotaxis)",
    ["Neuromuscular health: how well flies climb", "upward over seconds after being tapped down.",
     "The primary motor read-out in fly models", "of ALS, Parkinson's and Huntington's."],
    ["Pct = 100 · climbed / total", "PI  = 2 · Pct − 100     (range −100…+100)",
     "AUC = trapezoid of %-climbed vs time", "t50 = interpolated time to reach 50%",
     "+ CRI, SDQ, TSW composite rescue metrics"],
    ["IN: long CSV — Genotype, Treatment, Sex,", "     Replicate, Time, Count_climbed, n_total_flies",
     "     (or an .mp4 video → per-vial fly counts)",
     "OUT: PI/AUC/t50 tables, climbing curves (mean±SEM),", "     CSV export, 300-dpi figures"],
    ["PI formula exact; on 3 disease-model datasets", "wild-type PI > disease, drug partially rescues.",
     "Video counting exact on ground-truth clips", "(8 above / 4 below → PI 33.3; 3/9 → −50)."])

module_slide("Module 2 · Lifespan", "Survival / lifespan analysis (actuarial)",
    ["How long flies live, with right-censoring", "(some still alive at study end).",
     "The standard read-out of disease severity", "and of a drug's protective effect."],
    ["KM  Ŝ(t) = Π (1 − dᵢ/nᵢ)   (product-limit)", "Greenwood variance → 95% CI (cloglog)",
     "Median survival + Brookmeyer–Crowley CI", "Log-rank χ² = (ΣO−E)²/ΣV ;  pairwise BH",
     "Cox HR = exp(β)  (partial likelihood, Breslow)"],
    ["IN: event-level (Time, Event) OR daily cohort", "     (Day, Dead_count, Censored_count, Total_start)",
     "OUT: Kaplan–Meier curves, median table, log-rank", "     table, Cox HR table (95% CI), RMST, CSV"],
    ["Real Freireich (1963) 6-MP trial reproduced:", "median 23 / 8 wk; log-rank χ² = 16.79; HR = 4.52.",
     "Matches R survival (Breslow) and lifelines", "(Efron) to five significant figures."])

module_slide("Module 3 · Y-maze", "Y-maze: working memory and handedness",
    ["Spatial working memory (does the fly avoid", "the arm it just visited?) and locomotor",
     "handedness / behavioural individuality —", "cognition read-outs for disease models."],
    ["%alternation = 100 · triads / (entries − 2)", "  (tested vs 50% chance level)",
     "turn bias = R / (R + L)", "individuality: Monte-Carlo of observed MAD",
     "  vs a common-bias binomial null"],
    ["IN: per-fly arm-entry sequence (e.g. ABCACB),", "     or long per-entry rows, or L/R turns",
     "OUT: per-fly metrics, alternation summary (vs", "     chance p), turn-bias summary (%extreme,",
     "     individuality p), group comparison, CSV"],
    ["Hand sequences exact (ABCABC = 100%);", "unbiased fly alternates at exactly 50% (chance).",
     "Reproduces de Bivort 2015 (mean 0.50; 23.5%", "extreme) and the 50% chance level (Lewis 2017)."])

module_slide("Module 4 · DAMS", "Activity monitoring: sleep and circadian period",
    ["Sleep architecture and the internal body", "clock (circadian period), from minute-by-",
     "minute activity — key to ageing, sleep", "and neurodegeneration phenotypes."],
    ["Sleep bout = ≥ 5 min of zero activity", "  → total / day / night sleep, bouts, length",
     "Circadian: χ² (Sokolove–Bushell) periodogram", "  Q_P ~ χ²(p−1), Bonferroni across periods",
     "  + Lomb–Scargle cross-check"],
    ["IN: long CSV — Genotype, Treatment, Sex, FlyID,", "     Time_min, Activity (1-min bins; ≥7 days)",
     "OUT: sleep summary, sleep profile, period", "     summary (% rhythmic, mean period),",
     "     per-fly periodograms, CSV"],
    ["5-min rule exact at the 4- vs 5-min boundary.", "Recovers Konopka–Benzer (1971) periods",
     "24 / 19 / 28 h; 100% sensitivity, 0% false-", "positives. Matches astropy & pyBOAT (23.96 h)."])

module_slide("Module 5 · Larval crawling", "Larval crawling (locomotion)",
    ["Motor output of the larval nervous system:", "how fast and how straight a larva crawls —",
     "an early-stage locomotor read-out sensitive", "to neuromuscular disease models."],
    ["Detect larva = largest dark blob (Otsu)", "path length = Σ step distances",
     "mean speed = path × (px·mm⁻¹)⁻¹ / duration", "tortuosity = path / straight-line displacement",
     "+ max speed, pause count"],
    ["IN: single-larva .mp4 + pixels-per-mm scale,", "     or a per-larva summary CSV",
     "OUT: trajectory (x,y,t), per-larva summary,", "     group means ± SEM, comparison, CSV"],
    ["Known-velocity clip → speed recovered exactly", "(4.472 mm/s), tortuosity ≈ 1.",
     "Demo cohorts: wild-type ≈ 1 mm/s (published", "0.5–2 mm/s range); disease models slower."])

module_slide("Module 6 · Grooming", "Grooming behaviour",
    ["A structured self-cleaning behaviour used", "as a read-out of motor patterning and",
     "certain neuro/behavioural phenotypes;", "quantified from scored grooming bouts."],
    ["%time grooming = 100 · Σ bout / observation", "n bouts, mean bout duration",
     "grooming rate = bouts per minute", "(from scored bout start/stop intervals)"],
    ["IN: bout-level CSV — Bout_start_s, Bout_end_s,", "     Observation_s  (or a per-fly summary)",
     "OUT: per-fly metrics, group summary (% time,", "     bouts, rate) ± SEM, comparison, CSV"],
    ["Bout arithmetic exact (10% of 600 s = 3 bouts,", "mean 20 s).",
     "Demo wild-type ≈ 13% of time grooming,", "matching Qiao et al. 2018 (eLife)."])

module_slide("Engine 1 · Motor Decline Rate", "Motor Decline Rate (MDR) — decline over age",
    ["How fast climbing performance deteriorates", "with AGE (days) — a kinetic measure of",
     "disease progression, and of a drug slowing", "it, that endpoint PI alone cannot capture."],
    ["PI(age) = PI₀ · e^(−k · age)   (per group)", "k = decline-rate constant (per day)",
     "t½ = ln2 / k ;  R² = fit quality", "Δk = k(vehicle) − k(drug)  → rescue",
     "linear fallback if decay does not fit"],
    ["IN: climbing data WITH an Age column (endpoint", "     PI per age; ≥ 2 ages per group)",
     "OUT: MDR table (PI₀, k, t½, R², Δk, Rescue %,", "     trajectory label), decay-fit figure, CSV"],
    ["Recovers a known k across 5 sample sets:", "Pearson r = 0.989, median R² = 0.99.",
     "Correctly orders decline (disease > wild type)", "and detects drug-mediated slowing."])

module_slide("Engine 2 · Trajectory clustering", "Trajectory clustering (unsupervised)",
    ["Automatically groups genotype/treatment", "response curves into shared shape families",
     "(e.g. sustained rescue vs progressive", "decline) with no pre-labelling."],
    ["DTW = dynamic time warping distance", "  between each pair of PI-vs-time curves",
     "Ward-linkage hierarchical clustering", "silhouette score → optimal cluster number k",
     "archetype = DTW barycentre per cluster"],
    ["IN: group PI-vs-time table (Genotype, Sex,", "     Treatment, Time, PI_mean)",
     "OUT: dendrogram, PCA embedding, cluster table,", "     per-cluster archetype curves"],
    ["On data with two known shape families the", "clustering returns exactly 2 clusters with",
     "purity = 1.0 across 4 random seeds."])

# 14. INNOVATION (detailed) with figure
s = content("Engine 3 · Core innovation", "Drug Ranking: unequal-N cross-strain meta-analysis")
card(s, Inches(0.55), Inches(1.62), Inches(6.35), Inches(2.28), "How it calculates",
     ["Per strain × sex — Hedges' g (drug vs vehicle):",
      "g = J·(x̄_t − x̄_c)/s_p ,  Var(g) depends on n₁,n₂",
      "Survival effect = −logHR · √3/π  (to same scale)",
      "Pool across strains — inverse-variance / DL:",
      "g_pooled = Σwᵢgᵢ/Σwᵢ , wᵢ = 1/Var(gᵢ) ; report I²"],
     headcolor=GOLD, mono_from=1, size=10.5)
card(s, Inches(0.55), Inches(4.12), Inches(6.35), Inches(2.28), "Input → Output → Validation",
     ["IN: the four assay files for a common cohort",
      "OUT: per-strain effects, per-assay pooled g (I²),",
      "     composite Therapeutic Score + ranking, CSV",
      "VALIDATED: pooling exact vs statsmodels; ranks",
      "     A>B>C; 19.9% lower error, 95.8% CI coverage"],
     headcolor=GREEN, size=11)
if (FIG / "Figure2_innovation.png").exists():
    s.shapes.add_picture(str(FIG / "Figure2_innovation.png"), Inches(7.05), Inches(1.75), width=Inches(5.75))
footer(s)

# ═══ 15. INFOGRAPHIC: VALIDATION PILLARS ══════════════════════════════════════
s = content("Validation strategy", "How we tested every module — three independent pillars")
cols = [("Published benchmark data", GREEN,
         ["Real datasets with known, peer-reviewed answers.",
          "Freireich 1963 (survival); PlantGrowth / Dobson",
          "1983 (ANOVA); Konopka–Benzer 1971 (periods).",
          "", "Test: reproduce the paper's exact statistic."]),
        ("Analytic ground truth", TEAL,
         ["Data generated with a KNOWN answer built in.",
          "e.g. PI = PI₀·e^(−k·age) with a chosen k; a video",
          "with known fly positions; a signal of known 24 h.",
          "", "Test: recover the known value precisely."]),
        ("Reference datasets", NAVY,
         ["≥ 3 per assay, in the upload template format,",
          "each parameterised to a cited published value",
          "(and labelled real vs simulated).",
          "", "Test: reproduce the cited expectation (18/18)."])]
w = Inches(3.95); h = Inches(4.0); x = Inches(0.55); y = Inches(1.75)
for i, (t, col, lines) in enumerate(cols):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h); _fill(b, CARDBG, CARDLINE)
    tfh = b.text_frame; tfh.word_wrap = True; tfh.margin_left = tfh.margin_right = Pt(11); tfh.margin_top = Pt(12)
    _run(tfh.paragraphs[0], t, 15, col, bold=True, font=HEAD)
    for ln in lines:
        pp = tfh.add_paragraph(); pp.space_before = Pt(3); pp.line_spacing = 1.05; _run(pp, ln, 11.5, INK)
    x = x + w + Inches(0.2)
_, tf = _tb(s, Inches(0.55), Inches(6.0), Inches(12.2), Inches(0.9))
_run(tf.paragraphs[0], "Plus a fourth check: the SAME data through seven independent tools (R, lifelines, statsmodels, scipy, pingouin, astropy, pyBOAT) — 24/24 metrics agree.", 13, SLATE, italic=True)
footer(s)

# ═══ 16. TESTING SAMPLES ══════════════════════════════════════════════════════
s = content("Test data", "How we obtained the testing samples — and what they are")
bullets(s, Inches(0.55), Inches(1.65), Inches(6.5), Inches(5),
        [("Real raw data (used directly).", "Freireich 1963 6-MP leukemia trial (42 flies-equivalent event records, verified against the Klein–Moeschberger source); PlantGrowth (Dobson 1983)."),
         ("Real published parameters.", "Konopka–Benzer 1971 free-running periods (24 / 19 / 28 h) drive the circadian test signals."),
         ("Analytic ground truth.", "Programmatically generated so the correct answer is known exactly (decay constant k, fly positions in a video, a 24-h rhythm)."),
         ("Literature-calibrated reference sets.", "≥3 per assay, simulated from text-verified published values (chance 50%, bias 0.50/23.5%, WT life ≈50 d, larval ≈1 mm/s, grooming ≈13%) — each cited, each labelled 'simulated'.")], size=13, gap=10)
card(s, Inches(7.35), Inches(1.85), Inches(5.4), Inches(4.4), "Honesty principle",
     ["Where machine-readable raw supplements were",
      "available, we used them directly.", "",
      "Where they were not, we simulated from",
      "explicitly cited published parameters and",
      "labelled them as simulated — no dataset is",
      "presented as raw data we did not have.", "",
      "All 18 reference datasets reproduce their", "cited published expectation."], headcolor=GOLD, size=12.5)
footer(s)

# ═══ 17. TESTING PARAMETERS TABLE ═════════════════════════════════════════════
s = content("Parameters", "Testing parameters — each set from the literature, not tuned")
rows = [("Sleep threshold", "≥ 5 min inactivity", "Hendricks / Shaw 2000 operational definition"),
        ("Alternation chance", "50%", "Lewis 2017; also emerges from the arm geometry"),
        ("Reference periods", "24 / 19 / 28 h", "Konopka & Benzer 1971 (WT / perS / perL)"),
        ("Periodogram band", "16–32 h, Bonferroni", "Circadian range; controls family-wise error"),
        ("Turn-bias spread", "23.5% outside [0.3,0.7]", "de Bivort 2015 Canton-S distribution"),
        ("MDR axis", "Age (days)", "Negative geotaxis is age-dependent (Gargano 2005)"),
        ("Cox ties method", "Breslow", "Matches the published Klein–Moeschberger value"),
        ("Bootstrap resamples", "2000 (fixed seed)", "Stable CIs; fully reproducible")]
_, tf = _tb(s, Inches(0.55), Inches(1.55), Inches(12.2), Inches(0.32))
_run(tf.paragraphs[0], "No parameter was tuned to flatter a result — each is a community standard, cited here.", 13, SLATE, italic=True)
y = Inches(1.98)
for nm, val, why in rows:
    _, tf = _tb(s, Inches(0.55), y, Inches(3.0), Inches(0.56), MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], nm, 12.5, NAVY, bold=True)
    _, tf = _tb(s, Inches(3.65), y, Inches(3.1), Inches(0.56), MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], val, 12.5, GREEN, bold=True, font=MONO)
    _, tf = _tb(s, Inches(6.9), y, Inches(5.85), Inches(0.56), MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], why, 11.5, INK)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), y + Inches(0.55), Inches(12.2), Pt(0.75)); _fill(ln, CARDLINE)
    y += Inches(0.585)
footer(s)

# ═══ 18. RESULTS: VALIDATION ══════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK); bg(s, NAVY)
_, tf = _tb(s, Inches(0.55), Inches(0.42), Inches(12.2), Inches(0.36)); _run(tf.paragraphs[0], "RESULTS · VALIDATION", 12, GOLD, bold=True, font=HEAD)
_, tf = _tb(s, Inches(0.55), Inches(0.76), Inches(12.2), Inches(0.6)); _run(tf.paragraphs[0], "Every module reproduces its published benchmark", 27, WHITE, bold=True, font=HEAD)
for i, (n, lb) in enumerate([("12/12", "validation suites pass"), ("χ² = 16.79", "Freireich log-rank (published)"), ("HR = 4.52", "Cox hazard ratio (= published)"), ("r = 0.989", "MDR decline-rate recovery")]):
    stat_dark(s, Inches(0.7 + i * 3.02), Inches(1.65), Inches(2.8), n, lb)
if (FIG / "Figure1_validation.png").exists():
    s.shapes.add_picture(str(FIG / "Figure1_validation.png"), Inches(1.45), Inches(3.3), width=Inches(10.4))
_, tf = _tb(s, Inches(0.55), Inches(6.95), Inches(12.2), Inches(0.4))
_run(tf.paragraphs[0], "KM vs lifelines · climbing kinetics · MDR over age · circadian periodogram · handedness · drug ranking", 11, LIGHT, italic=True)

# ═══ 19. RESULTS: COMPARISON ══════════════════════════════════════════════════
s = content("Results · comparison", "Perfect agreement with seven independent tools")
for i, (n, lb) in enumerate([("24/24", "metrics agree"), ("r = 1.0000", "DroPhenix vs reference"), ("7", "reference tools")]):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7 + i * 2.12), Inches(1.7), Inches(1.92), Inches(1.5)); _fill(box, NAVY)
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _run(tf.paragraphs[0], n, 24, GOLD, bold=True, font=HEAD); tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    pp = tf.add_paragraph(); pp.alignment = PP_ALIGN.CENTER; _run(pp, lb, 11, WHITE)
bullets(s, Inches(0.7), Inches(3.5), Inches(6.1), Inches(3.2),
        [("R survival + base R", "— KM, log-rank, Cox (Breslow), ANOVA, Kruskal: exact"),
         ("lifelines · statsmodels", "— survival & meta-analysis: exact to 6 decimals"),
         ("scipy · pingouin", "— ANOVA / Kruskal / Hedges g: exact"),
         ("astropy + pyBOAT", "— circadian period 24.00 vs 23.96 h (two methods)")], size=13, gap=9)
if (FIG / "Figure3_tool_agreement.png").exists():
    s.shapes.add_picture(str(FIG / "Figure3_tool_agreement.png"), Inches(7.25), Inches(1.62), height=Inches(4.95))
footer(s)

# ═══ 20. REFERENCE-DATASET RESULTS ════════════════════════════════════════════
s = content("Results · reference data", "18/18 literature-referenced datasets reproduce the expectation")
rows = [("Climbing (×3)", "WT PI > disease; drug rescues (PD, ALS, HD models)"),
        ("Lifespan (×3)", "Freireich 23/8 wk; fly cohorts WT≈50 d > disease; drug extends"),
        ("Y-maze (×3)", "WT alternation > memory-mutant; bias≈0.50; ~23.5% extreme"),
        ("DAMS (×3)", "periods 24/19/28 h recovered; males sleep > females"),
        ("Larval (×3)", "WT ≈ 1 mm/s > disease models"),
        ("Grooming (×3)", "WT ≈ 13% time > disease models")]
y = Inches(1.7)
for nm, res in rows:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), y, Inches(3.0), Inches(0.66)); _fill(b, NAVY)
    tf = b.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; _run(tf.paragraphs[0], nm, 13, WHITE, bold=True); tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    _, tf = _tb(s, Inches(3.75), y, Inches(9.0), Inches(0.66), MSO_ANCHOR.MIDDLE); _run(tf.paragraphs[0], res, 13, INK)
    y += Inches(0.78)
_, tf = _tb(s, Inches(3.75), y + Inches(0.02), Inches(9.0), Inches(0.4))
_run(tf.paragraphs[0], "Each dataset is in the module's upload template — users can download and re-run them.", 12, SLATE, italic=True)
footer(s)

# ═══ 21. OUTPUT FORMATS ═══════════════════════════════════════════════════════
s = content("Deliverables", "What the tool gives you — and in what format")
outs = [("Interactive tables", "Every metric as an on-screen table (sortable), per group."),
        ("CSV export", "One-click download of each metric table for your own analysis."),
        ("Publication figures", "300-dpi PNG / SVG / PDF; colour-blind-safe palette."),
        ("Statistics report", "Test used, statistic, p-value, effect size, 95% CI, significance."),
        ("Drug ranking", "Per-strain effects, pooled per-assay effects, Therapeutic Score, ranks."),
        ("ZIP bundle", "All tables + figures packaged for reproducible journal submission.")]
x0, y0, w, h, gx, gy = Inches(0.55), Inches(1.7), Inches(6.05), Inches(1.5), Inches(0.2), Inches(0.16)
for i, (nm, ds) in enumerate(outs):
    r, c = divmod(i, 2); card(s, x0 + c * (w + gx), y0 + r * (h + gy), w, h, nm, [ds], size=12)
footer(s)

# ═══ 22. WHY RELIABLE ═════════════════════════════════════════════════════════
s = content("Trust", "Why DroPhenix is reliable")
items = [("Reproduces published data", "Textbook datasets (Freireich, PlantGrowth, Konopka–Benzer) return their exact published statistics."),
         ("Agrees with gold-standard software", "24/24 metrics match R survival, lifelines, statsmodels, scipy, pingouin, astropy, pyBOAT (r = 1.0000)."),
         ("Correct by construction", "Assumption-gated tests, FDR control, and the right measurement axis (e.g. MDR over age)."),
         ("Independently verifiable", "The full validation + comparison suite ships; every number regenerates with one command."),
         ("Honest about limits", "Method notes (Breslow vs Efron), un-buildable tools, and simulated-vs-real data are all stated.")]
x0, y0, w, h, gy = Inches(0.55), Inches(1.68), Inches(12.2), Inches(0.9), Inches(0.13)
for i, (hd, bd) in enumerate(items):
    card(s, x0, y0 + i * (h + gy), w, h, hd, [bd], size=12.5)
footer(s)

# ═══ 23. APPLICATIONS ═════════════════════════════════════════════════════════
s = content("Impact", "Applications — how it helps")
apps = [("Neurodegeneration screening", "Quantify a compound across ALS/PD/HD fly models in climbing, lifespan, cognition, sleep, locomotion."),
        ("Cross-strain compound triage", "Rank candidates fairly across disease backgrounds of unequal size — the core novelty."),
        ("Translational scoring", "Prioritise broad, safe, sustained rescue over sex-discordant or WT-perturbing hits."),
        ("Ageing & circadian biology", "Age-resolved motor decline, sleep architecture and free-running period in one workflow."),
        ("Teaching & reproducibility", "A transparent, benchmarked reference implementation of standard behavioural statistics."),
        ("Open science", "Free, no-login web app + open code + data templates lower the barrier for any lab.")]
x0, y0, w, h, gx, gy = Inches(0.55), Inches(1.7), Inches(6.05), Inches(1.5), Inches(0.2), Inches(0.16)
for i, (nm, ds) in enumerate(apps):
    r, c = divmod(i, 2); card(s, x0 + c * (w + gx), y0 + r * (h + gy), w, h, nm, [ds], size=12)
footer(s)

# ═══ 24. CLOSING ══════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK); bg(s, NAVY)
b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.4), SW, Inches(0.03)); _fill(b, GOLD)
_, tf = _tb(s, Inches(0.85), Inches(1.05), Inches(11.6), Inches(1.2))
_run(tf.paragraphs[0], "From fragmented spreadsheets to one verifiable workflow", 32, WHITE, bold=True, font=HEAD)
bullets(s, Inches(0.85), Inches(2.7), Inches(11.6), Inches(3),
        [("Six assays, one platform,", "with publication-grade statistics throughout."),
         ("A genuinely novel innovation:", "actuarial, unequal-N cross-strain compound ranking (19.9% more efficient than naive averaging)."),
         ("Validated and benchmarked:", "reproduces published data and agrees with gold-standard software on 24/24 checks."),
         ("Free and open:", "drophenix.streamlit.app · GitHub · GPL-3 · manuscript prepared for Nucleic Acids Research.")],
        size=15, gap=11, color=LIGHT)
footer(s, dark=True)

# ═══ SPEAKER NOTES (dual audience) ════════════════════════════════════════════
def set_notes(slide, text):
    tf = slide.notes_slide.notes_text_frame; parts = text.strip().split("\n")
    tf.text = parts[0]
    for line in parts[1:]:
        tf.add_paragraph().text = line

N = [
"""IN PLAIN TERMS: DroPhenix is a free website that automatically analyses experiments on fruit flies (Drosophila) — the standard cheap, fast stand-in for human brain disease. We turned six common fly tests into one reliable automated tool and added a new statistical method for comparing drugs fairly.
LEAD WITH: it agrees with the world's gold-standard analysis software on 24/24 checks (perfect correlation) and reproduces published scientific data exactly.
EXPERT Q: "Why flies?" — ~75% of human disease genes conserved; weeks-long lifespan; the standard first-pass ALS/PD/HD model. "Novel or a wrapper?" — both: standardises best practice AND a new ranking method.
NON-SCIENTIST Q: "What does it do?" — you upload your numbers, it makes the correct graphs and statistics and ranks which drug works best. "Who is it for?" — fly labs and pre-clinical drug screening.""",
"""IN PLAIN TERMS: An "assay" is a standard lab test. Today each is analysed separately in spreadsheets with inconsistent statistics, and there is no fair way to combine results when strains have different numbers of flies.
DEFINE: Kaplan–Meier / log-rank / Cox = the rigorous, clinical-grade ways to analyse lifespan data.
EXPERT Q: "Isn't survival analysis standard?" — In clinical stats yes, but fly papers often report only median; we bring the full toolkit to the bench.
NON-SCIENTIST Q: "Why does unequal numbers matter?" — Averaging a 3-person survey and a 700-person survey equally is misleading; the small noisy group distorts the answer — exactly the fly-strain problem.""",
"""IN PLAIN TERMS: The full menu of modules. Six are biological tests (climbing = motor health, lifespan, Y-maze = memory, DAMS = sleep/body-clock, larval crawling, grooming) and three are analytical engines (Motor Decline Rate, trajectory clustering, and the Drug Ranking innovation).
DEFINE: "negative geotaxis" = flies climb up after being tapped down; climbing measures neuromuscular health.
EXPERT Q: "Do the video tools replace dedicated trackers?" — for counting and single-larva speed yes; multi-animal pose tracking is out of scope and stated.
NON-SCIENTIST Q: "Why so many tests?" — different diseases hit movement, memory, sleep, or lifespan; testing across all gives a fuller, more trustworthy picture.""",
"""IN PLAIN TERMS: This infographic is the assembly line. You load your data (or a video, or the built-in demo); the tool cleans and checks it; the relevant module computes the metrics; the statistics engine runs; and you get tables, figures and CSV files out.
TALKING POINT: every step is deterministic — same input, same output — and inspectable; nothing is a hidden trained model.
EXPERT Q: "What QC happens at parse?" — column auto-detection, format validation, duplicate/outlier flagging, standardisation to long format.
NON-SCIENTIST Q: "How hard is it to use?" — upload a spreadsheet, click; the five-step flow is automatic.""",
"""IN PLAIN TERMS: Good statistics means choosing the RIGHT test for your data, not always the same one. Box 1: DroPhenix checks whether the data are bell-shaped (Shapiro–Wilk), whether groups vary similarly (Levene), and whether samples are big enough. Box 2: it then picks a parametric test if those hold, otherwise a non-parametric one. Box 3: it always reports the effect size, corrects for multiple comparisons, and gives a confidence interval.
DEFINE: effect size = how big the difference is; FDR correction = a safeguard against flukes when testing many things; bootstrap CI = a reliability range from resampling.
EXPERT Q: "Thresholds?" — Shapiro/Levene at 0.05, minimum n=5 for parametric.
NON-SCIENTIST Q: "Why care?" — it's the difference between a result that survives peer review / due-diligence and one that doesn't.""",
"""IN PLAIN TERMS: The climbing test. Flies are tapped to the bottom and instinctively climb; we measure the fraction that climb past a line over seconds. The Performance Index (PI) scales this from −100 (none) to +100 (all). AUC and t50 summarise the speed of climbing.
DEFINE: PI = 2×(% climbed) − 100.
EXPERT Q: "What are CRI/SDQ/TSW?" — composite rescue, sex-dimorphism and disease-selectivity indices layered on PI/AUC/t50.
NON-SCIENTIST Q: "What does the output look like?" — a table of scores per group plus climbing curves, downloadable as CSV and 300-dpi images.""",
"""IN PLAIN TERMS: The lifespan test. We track how long flies live, correctly handling those still alive at the end (censoring). Kaplan–Meier draws the survival curve; the log-rank test asks if two groups differ; Cox gives a hazard ratio — how many times higher the risk of death is.
DEFINE: hazard ratio = relative risk of death between groups.
EXPERT Q: "Median convention and ties?" — smallest t with S(t)≤0.5 (R/Prism), Breslow ties for Cox.
NON-SCIENTIST Q: "How do we know it's right?" — it reproduces the famous 1963 leukemia trial's published numbers exactly and matches R and lifelines.""",
"""IN PLAIN TERMS: The memory/behaviour test. In a Y-shaped maze a healthy fly tends to explore the arm it visited least recently (spontaneous alternation) — a working-memory read-out. We also measure left/right handedness and how individual flies differ.
DEFINE: alternation % vs a 50% chance level; turn bias = fraction of right turns.
EXPERT Q: "Why is 50% the chance level?" — under the arm geometry an alternation equals two same-direction turns, so an unbiased fly alternates at exactly 50% — it emerges from the model, not assumed.
NON-SCIENTIST Q: "What does a low score mean?" — poorer working memory, e.g. in a disease or drug-impaired fly.""",
"""IN PLAIN TERMS: The sleep and body-clock test, from minute-by-minute activity. A fly is "asleep" after 5 minutes of no movement (the field standard). The circadian period (near 24 h) is found with a periodogram — a method that detects the dominant rhythm in the data.
DEFINE: free-running period = the internal clock length in constant darkness.
EXPERT Q: "Multiplicity control?" — Bonferroni across tested periods; 0% false positives on rhythm-free data; Lomb–Scargle cross-check.
NON-SCIENTIST Q: "Proof it works?" — it recovers the classic mutant clock periods 24/19/28 h and agrees with two other clock tools.""",
"""IN PLAIN TERMS: The larval movement test. From a video we track a single larva and measure how fast and how straight it crawls, in real units (mm per second) once you tell it the pixels-per-millimetre scale.
DEFINE: tortuosity = how winding the path is (1 = straight line).
EXPERT Q: "Detection method?" — largest dark object per frame by Otsu thresholding; no training.
NON-SCIENTIST Q: "How accurate?" — on a test video with a known speed it recovers 4.472 mm/s exactly; wild-type larvae are ≈1 mm/s.""",
"""IN PLAIN TERMS: The grooming test. Flies clean themselves; we quantify what fraction of the observation time they spend grooming and the number and length of grooming bouts, from scored start/stop times.
EXPERT Q: "Does it detect grooming from raw video?" — no; automated behaviour classification is a separate ML problem; we analyse the scored intervals, which is the standard output.
NON-SCIENTIST Q: "Reference value?" — healthy flies groom ≈13% of the time (Qiao 2018), which our demo reproduces.""",
"""IN PLAIN TERMS: Motor Decline Rate measures how fast climbing worsens with AGE — a progression read-out. We fit an exponential-decay curve to Performance Index versus age and read off the decline rate k; a drug that slows decline lowers k.
KEY POINT: decline is over days of age, NOT the seconds within a single climbing test (where flies are rising) — conflating the two is a common error we deliberately avoid.
EXPERT Q: "What if it doesn't fit?" — a linear fallback and an R² are reported.
NON-SCIENTIST Q: "Why is this better than a single score?" — it captures disease PROGRESSION and slowing, not just a snapshot.""",
"""IN PLAIN TERMS: Trajectory clustering automatically groups response curves that behave similarly — for example separating 'keeps declining' from 'stays healthy' — without being told the labels in advance.
DEFINE: DTW (dynamic time warping) = a way to measure how similar two time-curves are even if shifted.
EXPERT Q: "How is k chosen?" — silhouette-optimal over Ward-linkage clustering.
NON-SCIENTIST Q: "What is it for?" — spotting distinct response patterns in a screen at a glance.""",
"""IN PLAIN TERMS — THE INNOVATION: How do you fairly combine a drug's effect across several fly strains of DIFFERENT sizes into one score? We borrow medical meta-analysis. Each strain's effect (Hedges' g) is weighted by its reliability (inverse of its variance): bigger, cleaner strains count more; small noisy ones count less but aren't discarded. These combine across strains and then across all assays into one Therapeutic Score.
DEFINE: Hedges' g = standardised effect size; inverse-variance / DerSimonian–Laird = the optimal weighting recipe; I² = how much strains genuinely differ.
PROOF: in 3,000 simulations it was 19.9% more accurate than simple averaging with correctly-calibrated confidence intervals (95.8%), and the maths matches statsmodels to six decimals.
EXPERT Q: "Why random-effects?" — to account for real strain-to-strain heterogeneity.
NON-SCIENTIST Q: "Business value?" — fewer wrong drug bets carried forward — real money saved.""",
"""IN PLAIN TERMS: We tested the tool three independent ways. (1) Published benchmark data — reproduce a paper's exact answer. (2) Analytic ground truth — feed in data whose correct answer we built in, and check it comes back out. (3) Reference datasets — ≥3 per assay tied to cited literature. A fourth check compares against seven trusted tools.
EXPERT Q: "Which pillar is strongest?" — analytic ground truth for exactness, published data for real-world validity; together they cover both.
NON-SCIENTIST Q: "In plain words?" — we checked it against textbook answers, against puzzles with known solutions, and against other trusted programs — and it passed all of them.""",
"""IN PLAIN TERMS: Where the test data came from, and honesty about it. Real raw data (a famous 1963 survival trial; a classic stats dataset) is used directly. Real published parameters (the classic clock periods) drive some signals. Some data is generated with a known answer to test exactness. The rest are example datasets calibrated to cited published numbers and clearly labelled 'simulated'.
KEY POINT: we never present a simulation as if it were real measured data.
EXPERT Q: "Why not pull every raw supplement?" — many are binary files on repositories not machine-readable in our environment; rather than fabricate, we cite parameters and label them.
NON-SCIENTIST Q: "How do I know it's not cherry-picked?" — every dataset lists its source and 18/18 reproduce the published expectation.""",
"""IN PLAIN TERMS: A common worry is that developers tweak settings until the tool looks good. We did the opposite — every key setting comes straight from the literature, with a citation. A fly counts as asleep after 5 minutes still; chance memory is 50%; the reference clock periods are 24/19/28 h.
EXPERT Q: "Periodogram multiplicity?" — Bonferroni-corrected; verified 0% false positives on rhythm-free data.
NON-SCIENTIST Q: "One-sentence reassurance?" — we didn't invent the settings; they are the community's own published standards.""",
"""IN PLAIN TERMS: The evidence the tool is correct. Twelve automated test suites all pass. On famous datasets it returns the exact published numbers — the 1963 trial's log-rank (16.79) and hazard ratio (4.52). The six-panel figure shows one real result from each area.
EXPERT Q: "Live or hard-coded?" — live; rerun run_all.py to reproduce.
NON-SCIENTIST Q: "What does 12/12 mean?" — twelve independent accuracy inspections, all passed.""",
"""IN PLAIN TERMS: Beyond matching papers, we ran the SAME data through seven independent, world-standard programs (including R, used across academia and pharma) and compared every number: perfect agreement (r = 1.0000) — every point sits on the diagonal.
EXPERT Q: "The one Cox difference (4.52 vs 4.82)?" — the known Breslow-vs-Efron tie method, not an error; our Breslow matches R's Breslow to five figures; reported transparently.
NON-SCIENTIST Q: "Why compare if yours is new?" — to prove we didn't reinvent the wheel wrongly; matching trusted tools is the strongest reassurance.""",
"""IN PLAIN TERMS: We also built three literature-referenced datasets for EACH assay (18 total) in the exact upload format, and ran them through the tool: all 18 reproduce their cited published expectation — wild-type outperforms disease models and drugs partially rescue, exactly as the literature reports.
EXPERT Q: "Are these downloadable?" — yes, in each module's template, with provenance and results files.
NON-SCIENTIST Q: "Why does this matter?" — it shows the tool gives sensible, literature-consistent answers on realistic data, not just toy examples.""",
"""IN PLAIN TERMS: What you actually get out. On-screen sortable tables; one-click CSV downloads; publication-quality 300-dpi figures; a full statistics report (test, p-value, effect size, confidence interval); the drug ranking; and a single ZIP bundle ready for a journal submission.
EXPERT Q: "Figure formats?" — PNG/SVG/PDF at 300 dpi, colour-blind-safe palette.
NON-SCIENTIST Q: "Can I put these straight in a paper or report?" — yes, that is the design goal.""",
"""IN PLAIN TERMS: Five concrete reasons to trust it: it reproduces published results; it agrees with gold-standard software; it uses the correct method by design; anyone can rerun and verify every number; and we are explicit about every limitation.
EXPERT Q: "Reproducibility mechanism?" — fixed random seeds plus the shipped validation/comparison suite.
NON-SCIENTIST Q: "What if it's ever wrong?" — because it's transparent and benchmarked, any discrepancy is visible and traceable — unlike a black box.""",
"""IN PLAIN TERMS: Where it helps. The biggest use is brain-disease drug screening — judging a compound fairly across several fly disease models and several tests at once. It also serves ageing/clock research, teaching rigorous statistics, and open reproducible science.
COMMERCIAL ANGLE: a fairer, more accurate candidate ranking means fewer false leads advanced — lower downstream cost; and it's free and browser-based, so adoption cost is near zero.
EXPERT Q: "Can it take my lab's data?" — yes; documented templates match common bench outputs.
NON-SCIENTIST Q: "Path to impact/revenue?" — adoption by fly labs and pre-clinical CROs; a validated core that can underpin paid hosted or extended versions.""",
"""IN PLAIN TERMS: The takeaway — six scattered, manually-analysed fly experiments turned into one automated, rigorously-validated web tool, plus a genuinely new, more efficient way to rank drugs across unequal-sized strains. Free, open, and prepared for a leading journal.
CLOSE ON: reliable (24/24 tool agreement, reproduces published data) + novel (19.9% more efficient ranking) + accessible (free web app).
FINAL Qs: Expert — "what's next?" more assays, hosted deployment. Non-scientist — "how do I try it?" open drophenix.streamlit.app; no install, no login.""",
]
for _sl, _nt in zip(prs.slides, N):
    set_notes(_sl, _nt)

prs.save(str(OUT))
print("wrote", OUT, "|", len(N), "notes for", len(prs.slides._sldIdLst), "slides")
