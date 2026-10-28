"""
Draw the climbing-assay recording setup: a printable protocol diagram.

Two panels, because the two mistakes are different in kind. The side view fixes
the PHYSICAL arrangement -- where the camera sits, what is behind the tubes,
where the light comes from. The frame view fixes what the operator should see
on screen before pressing record, which is where framing and label placement go
wrong.

Every annotation states the reason, not just the rule. A protocol that says
"space the tubes apart" is obeyed until someone is in a hurry; one that says
"tubes closer than this merge into one region and a whole replicate is invented
or lost" survives contact with a busy afternoon.

Run: ./.venv/Scripts/python.exe manuscript/make_protocol_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK, MUTED, GOOD, BAD, ACCENT = "#1a1a1a", "#5c5c5c", "#0072B2", "#D55E00", "#009E73"


def note(ax, x, y, text, color=MUTED, size=7.4, weight="normal", ha="left"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center",
            weight=weight, zorder=6)


def arrow(ax, xy, xytext, color=MUTED, lw=1.0):
    ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, zorder=5,
                                 shrinkA=0, shrinkB=2))


# ==========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 8.2))

# ── PANEL A: side view ──────────────────────────────────────────────────────
axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
axL.set_title("A   Bench arrangement (side view)", loc="left",
              fontsize=12.5, weight="bold", color=INK)

# Layout zones are kept strictly separate so no two labels can collide:
#   x 0.2-2.9  camera and its note   x 3.6-3.9  backing board
#   x 4.4-5.0  cylinder              x 5.8-10   all right-hand annotation
axL.add_patch(Rectangle((0.2, 1.15), 9.6, 0.32, color="#c9a227", alpha=0.5))
note(axL, 0.25, 0.92, "bench", MUTED, 7)

axL.add_patch(Rectangle((3.62, 1.47), 0.20, 5.9, color="#e8e8e8", ec=INK, lw=1.1))
axL.add_patch(Rectangle((3.95, 1.47), 1.55, 0.10, color="white", ec=INK, lw=0.7))

cx, cw, cbase, ctop = 4.45, 0.52, 1.57, 5.35
axL.add_patch(Rectangle((cx, cbase), cw, ctop - cbase, fc="#f2f8fc", ec=INK, lw=1.3))
axL.add_patch(Ellipse((cx + cw / 2, cbase), cw * 2.1, 0.20, fc="#f2f8fc", ec=INK, lw=1.0))
axL.add_patch(Ellipse((cx + cw / 2, ctop), cw, 0.15, fc="white", ec=INK, lw=1.0))
axL.add_patch(Ellipse((cx + cw / 2, ctop + 0.13), cw * 0.95, 0.28, fc="white",
                      ec=MUTED, lw=0.9))
ly = cbase + (ctop - cbase) * 0.55
axL.plot([cx, cx + cw], [ly, ly], color="black", lw=3.2, solid_capstyle="butt")
for fx, fy in [(0.22, 0.30), (0.62, 0.20), (0.36, 0.62), (0.75, 0.44), (0.5, 0.10)]:
    axL.plot(cx + fx * cw, cbase + fy * (ctop - cbase), marker="o", ms=2.2, color=INK)

camx, camy = 0.95, 3.35
axL.add_patch(FancyBboxPatch((camx, camy), 0.92, 0.58, boxstyle="round,pad=0.04",
                             fc="#dbe9f5", ec=INK, lw=1.2))
axL.add_patch(Rectangle((camx + 0.92, camy + 0.16), 0.20, 0.25, fc=INK))
note(axL, camx + 0.46, camy + 0.29, "CAMERA", INK, 7, "bold", ha="center")
axL.plot([camx + 0.46, camx + 0.18], [camy, 1.47], color=INK, lw=1.4)
axL.plot([camx + 0.46, camx + 0.74], [camy, 1.47], color=INK, lw=1.4)
note(axL, 0.25, 2.68, "PROP IT — do not hold it.", BAD, 8, "bold")
note(axL, 0.25, 2.22,
     "A hand-held phone drifts. One pixel of drift\n"
     "makes every printed scale mark look like a\n"
     "fly: counts read 109, 117, 97, 151 in the\n"
     "first seconds, then settled at 15.", MUTED, 7)

axL.plot([camx + 1.15, cx], [camy + 0.29, ly], ls=":", color=MUTED, lw=1.0)
note(axL, 1.30, 4.35, "lens at mid-cylinder height,\nsquare on to the tubes", MUTED, 7)

axL.annotate("", xy=(cx, 1.92), xytext=(camx + 1.15, 1.92),
             arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.1))
note(axL, 2.80, 1.72, "close enough that the tubes fill the frame",
     ACCENT, 7, ha="center")

RX = 5.85
arrow(axL, (3.86, 6.60), (RX - 0.12, 6.80), MUTED)
note(axL, RX, 7.00, "Matte pale backing board", INK, 8.5, "bold")
note(axL, RX, 6.52,
     "White or pale grey, NOT glossy. Flies are dark\n"
     "specks; anything dark or textured behind the\n"
     "glass competes with them.", MUTED, 7)

arrow(axL, (cx + cw + 0.05, ly), (RX - 0.12, 5.20), INK)
note(axL, RX, 5.40, "Scoring line", INK, 8.5, "bold")
note(axL, RX, 4.88,
     "Thick, fresh marker, right around the tube.\n"
     "Below ~40 grey levels of contrast it cannot be\n"
     "located, and a mislocated line silently shifts\nevery count.", MUTED, 7)

axL.add_patch(Ellipse((RX + 0.42, 3.55), 0.72, 0.72, fc="#ffe9a8",
                      ec="#c9a227", lw=1.1))
note(axL, RX + 0.42, 3.55, "light", "#8a6d1f", 7, "bold", ha="center")
arrow(axL, (cx + cw + 0.10, 4.10), (RX + 0.10, 3.75), "#c9a227", 1.2)
note(axL, RX + 1.00, 3.78, "Light from the FRONT", INK, 8.5, "bold")
note(axL, RX + 1.00, 3.35,
     "A window behind the tubes silhouettes\nthe glass and washes the flies out.", MUTED, 7)

note(axL, RX, 2.42, "White paper under the bases", INK, 8, "bold")
note(axL, RX, 2.05,
     "Stops the reflective feet swallowing the\nflies that sit lowest.", MUTED, 7)

# ── PANEL B: the frame ──────────────────────────────────────────────────────
axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis("off")
axR.set_title("B   What you should see before pressing record", loc="left",
              fontsize=12.5, weight="bold", color=INK)

fx0, fy0, fw, fh = 0.35, 2.95, 5.70, 5.80
axR.add_patch(Rectangle((fx0, fy0), fw, fh, fc="#fafafa", ec=INK, lw=1.6))
note(axR, fx0 + 0.10, fy0 + fh - 0.22, "camera frame", MUTED, 6.8)

n, cwid, gap = 3, 0.82, 0.62
total = n * cwid + (n - 1) * gap
sx = fx0 + 0.80
by, ty = fy0 + 0.78, fy0 + fh - 0.72
for i in range(n):
    x = sx + i * (cwid + gap)
    axR.add_patch(Rectangle((x, by), cwid, ty - by, fc="#f2f8fc", ec=INK, lw=1.2))
    axR.add_patch(Ellipse((x + cwid / 2, by), cwid * 1.5, 0.14,
                          fc="#f2f8fc", ec=INK, lw=0.9))
    axR.add_patch(Ellipse((x + cwid / 2, ty + 0.09), cwid * 0.9, 0.20,
                          fc="white", ec=MUTED, lw=0.8))
    lyy = by + (ty - by) * 0.55
    axR.plot([x, x + cwid], [lyy, lyy], color="black", lw=2.6)
    axR.text(x + cwid / 2, by - 0.32, f"R{i + 1}", ha="center", fontsize=8,
             color=INK, weight="bold")
    for fxx, fyy in [(0.25, 0.20), (0.62, 0.36), (0.40, 0.72), (0.72, 0.14), (0.50, 0.52)]:
        axR.plot(x + fxx * cwid, by + fyy * (ty - by), marker="o", ms=2.0, color=INK)

g0 = sx + cwid
axR.annotate("", xy=(g0 + gap, by + 0.55), xytext=(g0, by + 0.55),
             arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.2))
note(axR, sx + total / 2, by - 0.72,
     "gap ≥ ⅓ tube width — closer and two tubes merge into\n"
     "one region, inventing or losing a whole replicate",
     GOOD, 7, "bold", ha="center")

axR.annotate("", xy=(fx0 + 0.28, ty), xytext=(fx0 + 0.28, by),
             arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.2))
note(axR, fx0 + 0.40, (by + ty) / 2 + 0.60, "tubes fill\n≥45% of\nframe height",
     ACCENT, 7, "bold")

axR.add_patch(Rectangle((fx0 + fw - 1.28, fy0 + 0.55), 1.02, 0.98,
                        fc="white", ec=MUTED, lw=1.0))
for k in range(4):
    axR.plot([fx0 + fw - 1.18, fx0 + fw - 0.36],
             [fy0 + 1.36 - k * 0.21] * 2, color=MUTED, lw=0.8)
note(axR, fx0 + fw - 0.77, fy0 + 0.38, "label card", MUTED, 6.6, ha="center")
note(axR, fx0 + fw - 0.77, fy0 + 1.74, "well clear of\nthe tubes",
     GOOD, 6.8, "bold", ha="center")

CX = 6.40
note(axR, CX, 8.55, "BEFORE YOU PRESS RECORD", INK, 9, "bold")
for i, line in enumerate([
        "Phone propped, not held",
        "Recording started BEFORE the tap",
        "Hands out after the tap",
        "Nothing dark behind the tubes",
        "Line dark, unbroken, on every tube",
        "Flies-per-tube written down",
        "Tube diameter noted (25 mm)",
        "Each group in ≥2 tubes, ≥2 sessions"]):
    y = 8.12 - i * 0.38
    axR.text(CX, y, "☐", fontsize=9, color=GOOD, va="center")
    note(axR, CX + 0.30, y, line, INK, 7.2)

note(axR, CX, 4.60, "THEN CHECK IT", INK, 9, "bold")
axR.add_patch(FancyBboxPatch((CX, 3.95), 3.42, 0.40, boxstyle="round,pad=0.05",
                             fc="#f0f0f0", ec=MUTED, lw=0.8))
note(axR, CX + 0.10, 4.15, "drophenix_cli.py setup-check test.mov", INK, 7)
note(axR, CX, 3.28,
     "Record 30 s of test footage and run that.\n"
     "It measures camera drift, tube spacing, line\n"
     "contrast and framing, and names what to\n"
     "change. Five minutes here saves a session\n"
     "that would otherwise be discarded.", MUTED, 7)

fig.suptitle("Climbing assay: recording setup", x=0.010, ha="left",
             fontsize=15.5, weight="bold", color=INK)
fig.text(0.010, 0.945,
         "None of this is enforced by the software — it analyses whatever it is given and "
         "reports how reliable the result is. Following it raises the fraction of recordings "
         "that can be scored at all: in the first\nlaboratory set, about half could not be. "
         "A protocol improves the INPUT; it does not validate the OUTPUT, which needs counts "
         "from a person to compare against.",
         fontsize=8.2, color=MUTED, va="top")

fig.tight_layout(rect=[0, 0, 1, 0.900])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Protocol_climbing_setup.{ext}", dpi=200,
                bbox_inches="tight", facecolor="white")
print("wrote", OUT / "Protocol_climbing_setup.png")
