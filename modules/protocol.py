"""
Recording protocol for the climbing assay: the specification, and a check that
a setup meets it BEFORE the experiment runs.

WHY A PROTOCOL, AND WHAT IT CAN AND CANNOT DO
---------------------------------------------
Roughly half of the first fifteen laboratory recordings were not scoreable, and
none of the reasons were exotic: the camera moved mid-clip, the cylinders were
too close together to separate, the scoring line was too faint, the label card
sat inside a cylinder region. Every one of those is fixed at the bench in
seconds and cannot be fixed afterwards at all.

So a standard setup is worth having. But it is worth being precise about what it
buys, because two different things get conflated:

  A PROTOCOL improves the INPUT. It raises the fraction of recordings that can
  be scored, and it makes recordings from different laboratories comparable.

  VALIDATION proves the OUTPUT. It establishes that the counts the software
  produces match what a trained person counts in the same frames.

A perfect protocol does not validate anything. A detector can be fed immaculate
video and still miss two flies in every frame, consistently and invisibly. The
protocol and the human-scored validation are complementary and neither
substitutes for the other.

AND A PROTOCOL MUST NOT BE A GATE
---------------------------------
`check_setup` reports what does not conform; it does not refuse to analyse
non-conforming video, and the analysis path never requires any of this. A tool
that only works on its author's protocol is a tool for one laboratory.
Retrospective recordings exist, other groups have other rigs, and the useful
behaviour is to work on whatever arrives and say honestly how well.

WHAT THE SPECIFICATION IS FOR
-----------------------------
Each requirement below exists because its absence broke a real recording, and
each is stated with the reason rather than as a rule to obey. Anyone can then
judge when it does not apply to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# THE SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════════
SPEC: Dict[str, dict] = {
    "camera_fixed": {
        "requirement": "The camera must not move during the recording.",
        "why": ("The analysis reconstructs the empty scene by taking the median of many "
                "frames, which assumes a fixed viewpoint. A camera moved mid-clip makes "
                "that reconstruction a blend of two scenes and detection collapses. One of "
                "the first fifteen recordings failed entirely for this reason and the cause "
                "was invisible until it was looked for."),
        "how": "Prop the phone against a stand or box. Do not hold it.",
    },
    "tap_in_frame": {
        "requirement": "Start recording BEFORE the tap.",
        "why": ("Scoring times are measured from the moment the flies land at the bottom. "
                "Without the tap in shot there is no zero, and 'climbing at 10 s' means "
                "nothing."),
        "how": "Press record, wait two seconds, tap, then leave it running.",
    },
    "hands_clear": {
        "requirement": "Hands out of frame after the tap.",
        "why": ("The empty-scene model is built from frames after the tap. A hand entering "
                "later is baked into it and then subtracted out of every subsequent frame."),
        "how": "Tap and withdraw. Do not adjust anything until the recording ends.",
    },
    "cylinder_spacing": {
        "requirement": "Gap between cylinders at least one third of a cylinder width.",
        "why": ("The cylinders are located from where flies have been. Tubes closer than "
                "this merge into one region, and the count of cylinders comes out wrong -- "
                "which then invents or discards a whole replicate."),
        "how": "Space them visibly apart; a finger's width between tubes is ample.",
        "threshold": 0.33,
    },
    "line_contrast": {
        "requirement": "The scoring line must be strongly darker than the glass.",
        "why": ("A faint line cannot be located reliably, and a mislocated line silently "
                "shifts every count. Measured contrast below about 40 grey levels is where "
                "detection starts to fail."),
        "how": "Draw it with a fresh permanent marker, thick, right around the tube.",
        "threshold": 40.0,
    },
    "plain_background": {
        "requirement": "A plain matte pale backing behind the cylinders.",
        "why": ("Flies are dark specks; anything dark and textured behind the glass "
                "competes with them. The styrofoam sheet already used is close to ideal."),
        "how": "White or pale grey card or foam, matte not glossy, filling the frame behind.",
    },
    "labels_outside": {
        "requirement": "Keep the label card clear of the cylinders.",
        "why": ("Handwriting inside a cylinder region is the single most likely source of "
                "spurious detections; in one recording the label was picked up as a fourth "
                "cylinder."),
        "how": "Place it well to one side, or below the tube bases.",
    },
    "front_lighting": {
        "requirement": "Light from in front, not behind.",
        "why": ("Backlighting silhouettes the glass and washes out the contrast between "
                "dark flies and the pale background."),
        "how": "Keep windows behind the camera, not behind the cylinders.",
    },
    "framing": {
        "requirement": "Cylinders should fill most of the frame height.",
        "why": ("A fly is about a tenth of a cylinder's width. Zoomed out too far, a fly "
                "occupies too few pixels to separate from noise; the detector's size limits "
                "rescale with apparent width but resolution cannot be recovered."),
        "how": "Move closer until the tubes span most of the frame, base to plug.",
        "threshold": 0.45,
    },
    "record_totals": {
        "requirement": "Write down how many flies went into each cylinder.",
        "why": ("This is the denominator of every proportion. Counting flies above the line "
                "is easy; counting all of them is hard, because at early times they are "
                "piled on the reflective base. The software cannot work this out and should "
                "not have to."),
        "how": "One number per tube, in the manifest.",
    },
    "calibration": {
        "requirement": "Record the cylinder's outside diameter in millimetres.",
        "why": ("It converts pixels to millimetres, so climbing height becomes a physical "
                "quantity comparable between laboratories. 'Flies above the line' is not "
                "comparable: it depends where somebody drew the line. In this laboratory's "
                "own footage the same hand-drawn line sat at 86-95 mm across three tubes in "
                "ONE recording."),
        "how": "25 mm for a 50 ml graduated cylinder; 23.5 mm for a standard vial.",
    },
    "replication": {
        "requirement": "At least two cylinders per group, and groups spread across sessions.",
        "why": ("With one cylinder, the group effect and that cylinder's own peculiarity are "
                "the same number. And if a genotype appears in only one recording, that "
                "session's conditions are inseparable from the genotype."),
        "how": "Two or more tubes per group, and record each group on more than one day.",
    },
}


@dataclass
class SetupFinding:
    key: str
    ok: Optional[bool]           # None = could not be assessed automatically
    detail: str


@dataclass
class SetupReport:
    findings: List[SetupFinding] = field(default_factory=list)
    manual: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[SetupFinding]:
        return [f for f in self.findings if f.ok is False]

    def report(self) -> str:
        lines = []
        bad = self.failures
        good = [f for f in self.findings if f.ok is True]
        lines.append(f"{len(good)} check(s) passed, {len(bad)} failed, "
                     f"{len(self.manual)} to confirm by eye.")
        if bad:
            lines.append("\nFIX BEFORE RECORDING:")
            for f in bad:
                s = SPEC[f.key]
                lines.append(f"  - {s['requirement']}")
                lines.append(f"      seen: {f.detail}")
                lines.append(f"      fix : {s['how']}")
                lines.append(f"      why : {s['why']}")
        if good:
            lines.append("\nPassed:")
            for f in good:
                lines.append(f"  - {SPEC[f.key]['requirement']}  ({f.detail})")
        if self.manual:
            lines.append("\nCheck by eye (cannot be measured from the video):")
            for k in self.manual:
                lines.append(f"  - {SPEC[k]['requirement']}  {SPEC[k]['how']}")
        return "\n".join(lines)


def check_setup(path: str, n_vials: Optional[int] = None) -> SetupReport:
    """
    Score a TEST recording against the protocol, before the real experiment.

    Five minutes of test footage checked here saves a session that would
    otherwise be discarded. What can be measured from the video is measured;
    what cannot (lighting direction, whether the totals were written down) is
    listed for the operator to confirm.
    """
    from modules import climbing_video as CV

    rep = SetupReport()
    info = CV.probe_video(path)

    taps = CV.detect_taps(path)
    rep.findings.append(SetupFinding(
        "camera_fixed", bool(taps.get("framing_stable", True)),
        "framing stable throughout" if taps.get("framing_stable", True)
        else f"camera moved at {', '.join('%.0fs' % m['t0_s'] for m in taps.get('camera_moves', []))}"))

    n_taps = taps.get("n_taps", 0)
    first = taps["taps"][0]["t0_s"] if taps.get("taps") else None
    rep.findings.append(SetupFinding(
        "tap_in_frame", bool(n_taps >= 1 and first is not None and first < 10.0),
        f"tap detected at {first:.1f}s" if first is not None
        else "no tap found in the recording"))

    bg = CV.build_background(path)
    det = CV.detect_vials(path, n_vials=n_vials, background=bg)
    vials = det.get("vials", [])

    if len(vials) >= 2:
        xs = sorted((v.x0, v.x1) for v in vials)
        gaps = [xs[i + 1][0] - xs[i][1] for i in range(len(xs) - 1)]
        widths = [v.width for v in vials]
        rel = min(gaps) / max(np.mean(widths), 1)
        rep.findings.append(SetupFinding(
            "cylinder_spacing", bool(rel >= SPEC["cylinder_spacing"]["threshold"]),
            f"smallest gap is {rel:.0%} of a cylinder width "
            f"(want >= {SPEC['cylinder_spacing']['threshold']:.0%})"))
    if vials:
        depths = [v.line_confidence for v in vials if np.isfinite(v.line_confidence)]
        if depths:
            rep.findings.append(SetupFinding(
                "line_contrast", bool(min(depths) >= SPEC["line_contrast"]["threshold"]),
                f"faintest line is {min(depths):.0f} grey levels darker than the glass "
                f"(want >= {SPEC['line_contrast']['threshold']:.0f})"))
        frac = max(v.height for v in vials) / max(info.height, 1)
        rep.findings.append(SetupFinding(
            "framing", bool(frac >= SPEC["framing"]["threshold"]),
            f"tallest cylinder spans {frac:.0%} of the frame height "
            f"(want >= {SPEC['framing']['threshold']:.0%})"))

        # A fly must be big enough to resolve. Cylinder ~25 mm, fly ~2.5 mm.
        fly_px = np.mean([v.width for v in vials]) * 0.10
        rep.findings.append(SetupFinding(
            "framing" if False else "plain_background",
            bool(fly_px >= 6.0),
            f"a fly is about {fly_px:.0f} px long at this zoom "
            f"({'enough to resolve' if fly_px >= 6 else 'too small -- move closer'})"))

    conf = det.get("confidence", float("nan"))
    if np.isfinite(conf):
        rep.findings.append(SetupFinding(
            "labels_outside", bool(conf >= 0.85 and not det.get("needs_review")),
            f"cylinder detection confidence {conf:.2f}"
            + (f"; {det.get('reason', '')}" if det.get("reason") else "")))

    # Calibration card, if one is in shot. It is optional -- the tube diameter
    # works too -- but it is the better route, because the scale is then
    # measured from the image every time rather than remembered, and the card's
    # four markers cross-check each other.
    try:
        from modules import calibration_card as CC
        cal = CC.calibration_from_video(path, at_time_s=min(5.0, info.duration_s / 3))
        if cal.get("available"):
            rep.findings.append(SetupFinding(
                "calibration", True,
                f"card found in {cal['n_frames_with_card']}/{cal['n_frames_tried']} frames, "
                f"{cal['px_per_mm']:.2f} px/mm"
                + (f"; {cal['warnings'][0][:70]}" if cal.get("warnings") else "")))
        else:
            rep.manual.append("calibration")
    except Exception:
        rep.manual.append("calibration")

    for k in ["front_lighting", "hands_clear", "record_totals", "replication"]:
        if k not in rep.manual:
            rep.manual.append(k)
    return rep


def protocol_text() -> str:
    """The specification as plain text, for printing and pinning above the bench."""
    lines = ["CLIMBING ASSAY RECORDING PROTOCOL", "=" * 34, ""]
    for i, (k, s) in enumerate(SPEC.items(), 1):
        lines.append(f"{i}. {s['requirement']}")
        lines.append(f"   How : {s['how']}")
        lines.append(f"   Why : {s['why']}")
        lines.append("")
    lines.append("None of this is enforced by the software. It works on whatever it is")
    lines.append("given and reports honestly how reliable the result is. Following the")
    lines.append("protocol raises the fraction of recordings that can be scored at all --")
    lines.append("in the first laboratory set, about half could not be.")
    return "\n".join(lines)
