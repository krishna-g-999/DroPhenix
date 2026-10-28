"""
Plan an activity-monitor run before it happens.

WHY THIS EXISTS
---------------
For the climbing assay the thing that goes wrong is the RECORDING, and
`modules/protocol.py` addresses that. For activity monitors the recording is
automatic; what goes wrong is the LAYOUT — which fly goes in which tube of which
monitor.

That is not a hypothetical. In the first laboratory dataset every old fly sat in
Monitor 1 and every young fly in Monitor 2. Age was therefore completely
confounded with the instrument: any difference between the two machines — a
dusty beam, a different shelf position, a slightly different gain — was
arithmetically indistinguishable from an effect of ageing. The primary
comparison of the experiment could not be made at all, no analysis could repair
it, and by the time anyone could look the flies were dead.

The fix costs nothing and takes a minute at the bench: split every group across
every monitor. `plan_run` does the allocation; `check_plan` verifies any layout,
including one somebody has already loaded, so a mistake is caught while it is
still cheap.

WHAT THE ALLOCATION DOES, AND IN WHAT ORDER OF IMPORTANCE
---------------------------------------------------------
1. CROSS EVERY GROUP WITH EVERY MONITOR. This decides whether a question can be
   answered at all, so it comes first and is never traded away.
2. INTERLEAVE GROUPS ACROSS CHANNEL POSITIONS. Tubes at the ends of a rack are
   not equivalent to tubes in the middle — temperature and light vary across an
   incubator shelf — so a group loaded as a contiguous block inherits whatever
   is different about that end. This affects precision rather than
   answerability, so it comes second.
3. ROTATE THE PATTERN BETWEEN RUNS. If every experiment uses the identical
   layout, a position effect stops being noise and becomes a reproducible
   artefact that looks like biology.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SPEC: Dict[str, dict] = {
    "cross_monitors": {
        "requirement": "Split every experimental group across every monitor.",
        "why": ("If a group sits in only one monitor, that monitor's peculiarities and the "
                "group's biology are the same number. This is the error that made the first "
                "laboratory dataset's age comparison unanswerable."),
        "how": "Half of each group in monitor 1, half in monitor 2. The planner does this.",
    },
    "spread_positions": {
        "requirement": "Spread each group across channel positions within a monitor.",
        "why": ("Temperature and light vary across an incubator shelf, so tubes at the ends "
                "of a rack are not equivalent to tubes in the middle. A group loaded into "
                "channels 1-8 as a block inherits whatever is different about that end."),
        "how": "Interleave groups rather than blocking them. The planner does this too.",
    },
    "duration": {
        "requirement": "Record at least 3 full days; 5-7 if period matters.",
        "why": ("Sleep architecture needs several complete cycles to average over. A "
                "free-running period cannot be resolved from fewer than about 3 cycles at "
                "all, and separating a 23 h rhythm from a 25 h one needs more."),
        "how": "Decide the run length before loading, not afterwards.",
    },
    "light_regime": {
        "requirement": "Constant darkness for the clock; light-dark is fine for sleep.",
        "why": ("Under a light-dark cycle the fly is entrained, so a periodogram returns the "
                "period of the LAMP — near 24 h for a wild-type and for an arrhythmic mutant "
                "alike. Free-running period is a property of the animal and only appears once "
                "the lights stop telling it what to do."),
        "how": "Entrain in 12:12 LD for 3 days, then release into DD for the measurement.",
    },
    "expect_deaths": {
        "requirement": "Expect deaths, especially in aged cohorts.",
        "why": ("A dead fly breaks no beams, so it scores as sleeping perfectly. In the first "
                "dataset 10 of 64 flies died and they were concentrated in the old groups, "
                "which would have made 'old flies sleep more' partly 'old flies were dead'."),
        "how": "Start aged cohorts with a margin. The reader detects deaths automatically.",
    },
    "replication": {
        "requirement": "At least 8 tubes per group; 12-16 for aged cohorts.",
        "why": ("One tube is one fly, and the tube is the unit of replication. Aged cohorts "
                "lose animals during the run, so the number that finishes is what counts."),
        "how": "Plan for attrition rather than discovering it.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
def plan_run(groups: Sequence[dict],
             n_per_group: int = 8,
             n_monitors: int = 2,
             n_channels: int = 32,
             goal: str = "sleep",
             seed: int = 0) -> dict:
    """
    Allocate flies to monitors and channels so that no factor is confounded with
    the instrument.

    `groups` describes the biological conditions, e.g.
        [{"Genotype": "Oregon-R", "Sex": "Female", "Age": "old"}, ...]
    `goal` is "sleep" or "period"; it changes only the duration and light advice.

    Returns the channel map (ready for `dam_reader.apply_channel_map`), a
    verification of that map, and the run advice.
    """
    rng = np.random.default_rng(seed)
    capacity = n_monitors * n_channels
    needed = len(groups) * n_per_group
    warnings: List[str] = []

    if not groups:
        return {"ok": False, "reason": "No groups given.", "channel_map": pd.DataFrame()}
    if needed > capacity:
        return {
            "ok": False,
            "reason": (f"{len(groups)} groups x {n_per_group} flies = {needed} tubes, but "
                       f"{n_monitors} monitors x {n_channels} channels = {capacity}. Reduce "
                       f"the groups, reduce the flies per group, or add a monitor."),
            "channel_map": pd.DataFrame()}

    if n_monitors < 2:
        warnings.append(
            "With a single monitor nothing can be crossed with the instrument: any monitor "
            "fault is inseparable from every biological effect. A second monitor is the "
            "cheapest insurance available.")

    per_mon, remainder = divmod(n_per_group, n_monitors)
    if remainder:
        warnings.append(
            f"{n_per_group} flies per group does not divide evenly across {n_monitors} "
            f"monitors. The remainder is rotated between groups so no monitor is "
            f"systematically favoured, but a multiple of {n_monitors} would be tidier.")

    # 1. cross every group with every monitor
    slots: Dict[int, List[dict]] = {m: [] for m in range(1, n_monitors + 1)}
    for gi, g in enumerate(groups):
        counts = [per_mon] * n_monitors
        for r in range(remainder):
            counts[(gi + r) % n_monitors] += 1        # rotate the surplus
        for m, c in zip(range(1, n_monitors + 1), counts):
            slots[m] += [dict(g) for _ in range(c)]

    # 2. interleave within each monitor, 3. rotate the starting position
    rows: List[dict] = []
    for m, entries in slots.items():
        pools: Dict[str, List[dict]] = {}
        for e in entries:
            pools.setdefault(_key(e), []).append(e)
        order: List[dict] = []
        pool_list = list(pools.values())
        while any(pool_list):
            for pool in pool_list:
                if pool:
                    order.append(pool.pop())
        channels = list(range(1, len(order) + 1))
        shift = int(rng.integers(0, max(len(channels), 1)))
        channels = channels[shift:] + channels[:shift]
        for ch, e in zip(channels, order):
            rows.append({
                "Monitor": f"M{m}", "Channel": ch,
                "Genotype": e.get("Genotype", ""),
                "Sex": e.get("Sex", ""),
                "Age": e.get("Age", ""),
                "Treatment": e.get("Treatment", "") or "Untreated",
                "Exclude": "",
            })

    cmap = (pd.DataFrame(rows)
            .sort_values(["Monitor", "Channel"])
            .reset_index(drop=True))

    verdict = check_plan(cmap)
    unused = capacity - len(cmap)
    if unused:
        warnings.append(
            f"{unused} channel(s) left empty. Leave them empty rather than padding a group: "
            f"an unlabelled tube contributes nothing, and an unbalanced one reintroduces the "
            f"very confound this layout avoids.")

    return {"ok": True, "channel_map": cmap, "verification": verdict,
            "advice": run_advice(goal),
            "warnings": warnings + verdict.get("warnings", []),
            "n_tubes": len(cmap), "capacity": capacity}


def _key(e: dict) -> str:
    return "|".join(f"{k}={e.get(k, '')}" for k in ("Genotype", "Sex", "Age", "Treatment"))


def check_plan(channel_map) -> dict:
    """
    Verify a channel layout BEFORE the flies go in — or an existing one, to learn
    what a completed run will and will not be able to answer.

    The decisive test is whether each biological factor is CROSSED with Monitor.
    Everything else affects precision; this one decides whether the question can
    be answered at all, so it is reported separately as `blocking`.
    """
    from modules import design_schema as ds

    cmap = pd.read_csv(channel_map) if isinstance(channel_map, str) else channel_map.copy()
    if "Monitor" not in cmap.columns or "Channel" not in cmap.columns:
        return {"ok": False, "blocking": ["The layout needs Monitor and Channel columns."],
                "warnings": [], "per_factor": {}, "answerable": [], "blocked": []}

    factors = [c for c in ("Genotype", "Sex", "Age", "Treatment")
               if c in cmap.columns
               and cmap[c].astype(str).str.strip().ne("").any()
               and cmap[c].nunique(dropna=True) > 1]

    blocking: List[str] = []
    warnings: List[str] = []
    per_factor: Dict[str, str] = {}

    n_mon = cmap["Monitor"].nunique()
    for f in factors:
        if n_mon < 2:
            per_factor[f] = "unverifiable (one monitor)"
            continue
        res = ds.check_confounding(cmap, f, "Monitor")
        sev = res.get("severity", 0.0) if res.get("available") else 0.0
        if sev >= 1.0:
            per_factor[f] = "confounded"
            blocking.append(
                f"'{f}' is completely confounded with Monitor: every level sits in one "
                f"monitor only. This comparison cannot be made from the resulting data, and "
                f"no analysis will repair it. Split each {f} group across both monitors "
                f"before loading.")
        elif sev > 0:
            per_factor[f] = "partly confounded"
            warnings.append(
                f"'{f}' is only partly crossed with Monitor (Cramer's V = "
                f"{res.get('cramers_v', float('nan')):.2f}). Estimable, but with reduced "
                f"precision and some residual bias.")
        else:
            per_factor[f] = "crossed"

    # position balance within a monitor
    for f in factors:
        blocked_here = False
        for (lvl, mon), sub in cmap.groupby([f, "Monitor"]):
            ch = np.sort(sub["Channel"].to_numpy())
            if len(ch) >= 3 and (ch.max() - ch.min() + 1) == len(ch):
                blocked_here = True
                break
        if blocked_here:
            warnings.append(
                f"Some '{f}' groups occupy a contiguous block of channels. Temperature and "
                f"light vary across a rack, so a blocked group inherits whatever is different "
                f"about that end of it. Interleave the groups.")
            break

    if factors:
        n_per = cmap.groupby(factors).size()
        if len(n_per) and int(n_per.min()) < 8:
            warnings.append(
                f"The smallest group has {int(n_per.min())} tubes. Eight is a usual minimum, "
                f"and aged cohorts need more because animals die during the run.")

    return {"ok": not blocking, "per_factor": per_factor,
            "blocking": blocking, "warnings": warnings,
            "answerable": [f for f, v in per_factor.items() if v == "crossed"],
            "blocked": [f for f, v in per_factor.items() if v == "confounded"]}


def run_advice(goal: str = "sleep") -> List[str]:
    g = (goal or "sleep").lower()
    if g.startswith("period") or "circadian" in g or "clock" in g:
        return [
            "GOAL: free-running circadian period.",
            "Entrain in 12:12 LD for at least 3 days, then release into CONSTANT DARKNESS "
            "and measure in DD.",
            "Record at least 5 days in DD; 7 is better. Fewer than 3 cycles cannot resolve a "
            "period at all.",
            "Period CANNOT be measured in LD. Under a light cycle the fly is entrained and a "
            "periodogram returns the period of the lamp — about 24 h for a wild-type and for "
            "an arrhythmic mutant alike.",
        ]
    return [
        "GOAL: sleep and activity.",
        "A 12:12 light-dark cycle is correct: sleep amount, architecture and the day-night "
        "distribution are all valid under LD.",
        "Record at least 3 full days so there are several cycles to average over, and discard "
        "the first day as acclimation to the tube.",
        "Do not report a circadian period from these data — that needs constant darkness.",
    ]


def plan_report(plan: dict) -> str:
    """Render a plan as text for the bench."""
    if not plan.get("ok"):
        return f"PLAN REFUSED: {plan.get('reason', 'unknown reason')}"

    v = plan["verification"]
    lines = [f"DAM RUN PLAN — {plan['n_tubes']} tubes of {plan['capacity']} channels", ""]
    lines.append("Factor status:")
    for f, s in v["per_factor"].items():
        mark = "OK " if s == "crossed" else "!! "
        lines.append(f"  {mark}{f}: {s} with Monitor")
    lines.append("")
    if v["blocking"]:
        lines.append("BLOCKING — fix before loading:")
        for b in v["blocking"]:
            lines.append(f"  - {b}")
        lines.append("")
    lines.append("Run advice:")
    for a in plan["advice"]:
        lines.append(f"  {a}")
    if plan["warnings"]:
        lines.append("")
        lines.append("Notes:")
        for w in plan["warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def protocol_text() -> str:
    """The activity-monitor specification, for printing."""
    lines = ["ACTIVITY MONITOR (DAM) RUN PROTOCOL", "=" * 35, ""]
    for i, (_, s) in enumerate(SPEC.items(), 1):
        lines.append(f"{i}. {s['requirement']}")
        lines.append(f"   How : {s['how']}")
        lines.append(f"   Why : {s['why']}")
        lines.append("")
    return "\n".join(lines)
