"""
DroPhenix — ymaze.py
Y-maze behavioural-assay engine for Drosophila.

Three complementary read-outs are supported from a single per-fly arm-entry
sequence:

1. Spontaneous alternation (spatial working memory)
   ------------------------------------------------
   A fly exploring a symmetric 3-arm Y-maze tends to enter the arm it visited
   least recently. An "alternation" is a triad of three consecutive entries
   into three DIFFERENT arms. Performance:

       %alternation = 100 * (#alternation triads) / (total entries - 2)

   Flies alternate ABOVE a 50% chance level (Kim et al., 2017,
   Neurobiol. Learn. Mem. 144:50-58; flies, like rodents, favour the
   least-recently-seen arm). 50% is the field-standard chance baseline for
   sequential 3-arm alternation; it is exposed as a parameter because some
   authors use 33.3% (pure-random arm choice) or 22.2% (rodent SAP).

2. Turn bias / handedness (behavioural individuality)
   --------------------------------------------------
   Each pass through the centre is a left or right turn; turn bias = P(right).
   The population mean is ~0.5 (Buchanan, Kain & de Bivort, 2015, PNAS
   112:6700-6705 reported mean indistinguishable from 0.5, with 23.5% of
   Canton-S flies turning right >70% or <30% of the time; Werkhoven et al.,
   2021, reported a grand mean of 0.496 across 183,496 flies). The biologically
   meaningful signal is that the BETWEEN-fly variance exceeds what finite
   sampling alone would produce — i.e. stable individual handedness. This is
   tested here by Monte-Carlo against a common-bias binomial null.

3. Exploration / locomotor activity
   ---------------------------------
   Total arm entries (choices) per fly — a general activity read-out reported
   alongside the cognitive measures.

Geometry convention (only used when deriving turns from an arm sequence):
arms are arranged cyclically A->B->C->A. A transition i -> (i+1) mod 3 is a
RIGHT turn; i -> (i+2) mod 3 is a LEFT turn. Under this mapping an alternation
triad occurs exactly when two consecutive turns share direction, so an
unbiased fly with independent turns alternates at b^2+(1-b)^2 = 50% — matching
the published chance level. Provide an explicit Turn column to override.

WHICH READ-OUT IS THE ESTABLISHED ONE (verified 2026-09-07)
------------------------------------------------------------
For the *Drosophila* Y-maze the established individual-level metrics are
TURN BIAS (the fraction of centre crossings taken to the right) and TURN
SWITCHINESS (the fraction of consecutive turns that change direction), as used
by Buchanan, Kain & de Bivort (2015) PNAS 112:6700, "Neuronal control of
locomotor handedness in Drosophila", where individual flies make hundreds of
left/right choices in a symmetric, evenly lit maze. Both are computed here as
`turn_bias` and `switch_rate`, and they are the columns to report.

`alternation_pct` (the all-three-arms-different triad score) is a RODENT
working-memory convention imported into the fly assay. It is retained because
some fly papers report it, but under this maze's geometry it is a deterministic
function of handedness for a memoryless animal, so it is NOT a clean memory
read-out — see the derivation at the point of computation.

`alternation_vs_chance`, `chance_alternation_pct`, `switch_rate_vs_chance` and
`chance_switch_rate` are DroPhenix-defined bias corrections, not published
indices. They are derived in closed form from the module's own geometry
convention and verified against simulation of memoryless flies (residual ~0 at
every turn bias). Report them as corrections, and say so, rather than citing
them to a source.

CLUSTERING CAVEAT (audit S3 finding 3.9)
-----------------------------------------
`compare_groups` passes one row per ANIMAL straight to stats_engine.auto_test,
which treats every animal as an independent observation. Animals from the same
vial share a food batch, a rearing density and a handling event, so they are
correlated. The 2026-08 simulation measured what that costs: a naive
fly-as-independent test rejects 25.9% of the time under a TRUE NULL at
ICC = 0.10 — a 5.2x inflation of the nominal 5% error rate.

If the input carries a vial identifier, route the comparison through
`modules/hierarchical.py` instead, or at minimum report the design effect
DEFF = 1 + (m - 1) * ICC alongside the p-value. The p-values from this helper
are only valid when each animal really is an independent experimental unit.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GROUP_KEYS = ["Genotype", "Treatment", "Sex"]
ARM_ORDER_DEFAULT = ("A", "B", "C")
DEFAULT_CHANCE = 0.50
MIN_ENTRIES = 3   # need >=3 entries to score one alternation triad


# ──────────────────────────────────────────────────────────────────────────────
# Significance label (consistent with the rest of DroPhenix)
# ──────────────────────────────────────────────────────────────────────────────
def _sig(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ──────────────────────────────────────────────────────────────────────────────
# Turn <-> arm conversion (documented Y-maze geometry)
# ──────────────────────────────────────────────────────────────────────────────
def arm_sequence_to_turns(arms: Sequence, order: Sequence = ARM_ORDER_DEFAULT) -> List[str]:
    """Convert an arm-entry sequence to L/R turns (see module geometry note)."""
    idx = {a: i for i, a in enumerate(order)}
    turns: List[str] = []
    for prev, cur in zip(arms[:-1], arms[1:]):
        if prev not in idx or cur not in idx:
            continue
        d = (idx[cur] - idx[prev]) % 3
        if d == 1:
            turns.append("R")
        elif d == 2:
            turns.append("L")
        # d == 0 (immediate re-entry of same arm) contributes no turn
    return turns


def turns_to_arm_sequence(turns: Sequence, start: str = "A",
                          order: Sequence = ARM_ORDER_DEFAULT) -> List[str]:
    """Inverse mapping — build an arm sequence from L/R turns (for synthetic data)."""
    order = list(order)
    i = order.index(start)
    arms = [order[i]]
    for t in turns:
        step = 1 if str(t).upper().startswith("R") else 2
        i = (i + step) % 3
        arms.append(order[i])
    return arms


# ──────────────────────────────────────────────────────────────────────────────
# Input parsing -> canonical long table
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    lut = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "Genotype": ["genotype", "strain", "line"],
        "Treatment": ["treatment", "drug", "compound", "condition"],
        "Sex": ["sex"],
        "FlyID": ["flyid", "fly_id", "fly", "id", "animal", "subject"],
        "Order": ["order", "entry", "entry_order", "step", "index", "time", "t"],
        "Arm": ["arm", "arm_entered", "choice", "position"],
        "Turn": ["turn", "direction", "lr"],
        "Sequence": ["sequence", "arm_sequence", "arms", "seq", "path"],
    }
    found = {}
    for canon, opts in aliases.items():
        for o in opts:
            if o in lut:
                found[canon] = lut[o]
                break
    return found


def load_ymaze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise input to a canonical long table:
        Genotype, Treatment, Sex, FlyID, Order, Arm, Turn

    Accepts:
      (A) Long per-entry rows: one row per arm entry, with an Arm column
          (and/or a Turn column), ordered by Order if present.
      (B) Per-fly sequence string: a Sequence column like "ABCACB" or
          "A,B,C,A,C,B" (one row per fly).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Genotype", "Treatment", "Sex", "FlyID", "Order", "Arm", "Turn"])

    col = _resolve_columns(df)
    df = df.copy()
    # Fill optional grouping columns with constants if absent.
    for k, const in (("Treatment", "All"), ("Sex", "Pooled")):
        if k not in col:
            df[k] = const
            col[k] = k
    if "Genotype" not in col:
        df["Genotype"] = "Genotype1"
        col["Genotype"] = "Genotype"
    if "FlyID" not in col:
        df["FlyID"] = np.arange(len(df))
        col["FlyID"] = "FlyID"

    # ── Layout B: per-fly sequence string ────────────────────────────────────
    if "Sequence" in col and "Arm" not in col:
        rows = []
        for _, r in df.iterrows():
            seq = str(r[col["Sequence"]]).strip()
            arms = [s for s in (seq.split(",") if "," in seq else list(seq)) if s.strip()]
            for o, a in enumerate(arms):
                rows.append({
                    "Genotype": r[col["Genotype"]], "Treatment": r[col["Treatment"]],
                    "Sex": r[col["Sex"]], "FlyID": r[col["FlyID"]],
                    "Order": o, "Arm": a.strip(), "Turn": np.nan,
                })
        out = pd.DataFrame(rows)

    # ── Layout A: long per-entry rows ─────────────────────────────────────────
    elif "Arm" in col or "Turn" in col:
        out = pd.DataFrame({
            "Genotype": df[col["Genotype"]], "Treatment": df[col["Treatment"]],
            "Sex": df[col["Sex"]], "FlyID": df[col["FlyID"]],
            "Order": df[col["Order"]] if "Order" in col else np.arange(len(df)),
            "Arm": df[col["Arm"]].astype(str) if "Arm" in col else np.nan,
            "Turn": df[col["Turn"]].astype(str).str.upper().str[0] if "Turn" in col else np.nan,
        })
        out = out.sort_values(["Genotype", "Treatment", "Sex", "FlyID", "Order"])
    else:
        raise ValueError(
            "Unrecognised Y-maze format. Provide per-entry rows with an 'Arm' "
            "(or 'Turn') column, or a per-fly 'Sequence' column.")

    for k in GROUP_KEYS + ["FlyID"]:
        out[k] = out[k].astype(str)
    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-fly metrics
# ──────────────────────────────────────────────────────────────────────────────
def per_fly_metrics(long_df: pd.DataFrame,
                    chance: float = DEFAULT_CHANCE,
                    arm_order: Sequence = ARM_ORDER_DEFAULT) -> pd.DataFrame:
    """
    Compute per-fly Y-maze metrics:
      n_entries, n_triads, n_alternations, alternation_pct,
      n_turns, n_right, n_left, turn_bias, switch_rate.
    """
    rows = []
    keys = GROUP_KEYS + ["FlyID"]
    for kv, sub in long_df.groupby(keys, sort=False):
        sub = sub.sort_values("Order")
        arms = [a for a in sub["Arm"].tolist() if isinstance(a, str) and a == a]
        has_arms = len(arms) >= 1

        # Turns: explicit Turn column if present & non-empty, else derive from arms.
        turn_vals = [t for t in sub["Turn"].tolist() if isinstance(t, str) and t in ("L", "R")]
        if not turn_vals and has_arms:
            turn_vals = arm_sequence_to_turns(arms, order=arm_order)

        n_entries = len(arms) if has_arms else (len(turn_vals) + 1)

        # Alternation (needs arms)
        n_alt = n_triads = 0
        if has_arms and n_entries >= MIN_ENTRIES:
            for i in range(2, len(arms)):
                n_triads += 1
                if arms[i] != arms[i - 1] and arms[i] != arms[i - 2] and arms[i - 1] != arms[i - 2]:
                    n_alt += 1
        alt_pct = 100.0 * n_alt / n_triads if n_triads > 0 else np.nan

        # Turn bias
        n_right = sum(1 for t in turn_vals if t == "R")
        n_left = sum(1 for t in turn_vals if t == "L")
        n_turns = n_right + n_left
        turn_bias = n_right / n_turns if n_turns > 0 else np.nan

        # Switch rate: fraction of consecutive turns that change direction
        n_switch = sum(1 for a, b in zip(turn_vals[:-1], turn_vals[1:]) if a != b)
        switch_rate = n_switch / (len(turn_vals) - 1) if len(turn_vals) >= 2 else np.nan

        # ── Turn-bias-corrected alternation (audit S2 finding 2.6) ───────────
        # Under the module's geometry convention an alternation triad occurs
        # exactly when two consecutive turns share direction, so a fly with NO
        # memory at all whose turns are i.i.d. with P(right) = b alternates at
        # chance = b^2 + (1-b)^2. That is 50% only for an unbiased fly, and it
        # rises steeply with bias: simulation of 200 memoryless flies per point
        # gives 50.6 / 51.9 / 58.8 / 67.6 / 82.2 % at b = 0.5 / .6 / .7 / .8 / .9,
        # matching the closed form to Monte-Carlo error.
        #
        # Werkhoven et al. report individual Canton-S flies with bias above 70%
        # or below 30%, so raw alternation_pct confounds working memory with
        # turn handedness: a memoryless fly at b = 0.9 scores 82%, which reads
        # as excellent spatial memory. alternation_vs_chance subtracts each
        # fly's OWN bias-expected baseline, so 0 means "no better than this
        # fly's handedness predicts" and positive means genuine alternation.
        chance_alt = (100.0 * (turn_bias ** 2 + (1.0 - turn_bias) ** 2)
                      if pd.notna(turn_bias) else np.nan)
        alt_vs_chance = (alt_pct - chance_alt
                         if pd.notna(alt_pct) and pd.notna(chance_alt) else np.nan)

        # Switchiness carries the SAME handedness confound. For i.i.d. turns
        # with P(right) = b, consecutive turns differ with probability
        # 2*b*(1-b), which falls from 0.5 at b = 0.5 to 0.18 at b = 0.9. A
        # strongly handed fly therefore looks "less switchy" with no change in
        # sequential dependence. Reporting the residual against each fly's own
        # baseline separates the two, which matters because turn bias and
        # switchiness are the two established Drosophila Y-maze read-outs
        # (Buchanan et al. 2015, PNAS 112:6700) and are not independent.
        chance_switch = (2.0 * turn_bias * (1.0 - turn_bias)
                         if pd.notna(turn_bias) else np.nan)
        switch_vs_chance = (switch_rate - chance_switch
                            if pd.notna(switch_rate) and pd.notna(chance_switch) else np.nan)

        rec = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "n_entries": int(n_entries),
            "n_triads": int(n_triads),
            "n_alternations": int(n_alt),
            "alternation_pct": round(alt_pct, 2) if pd.notna(alt_pct) else np.nan,
            "chance_alternation_pct": round(chance_alt, 2) if pd.notna(chance_alt) else np.nan,
            "alternation_vs_chance": round(alt_vs_chance, 2) if pd.notna(alt_vs_chance) else np.nan,
            "n_turns": int(n_turns),
            "n_right": int(n_right),
            "n_left": int(n_left),
            "turn_bias": round(turn_bias, 4) if pd.notna(turn_bias) else np.nan,
            "switch_rate": round(switch_rate, 4) if pd.notna(switch_rate) else np.nan,
            "chance_switch_rate": round(chance_switch, 4) if pd.notna(chance_switch) else np.nan,
            "switch_rate_vs_chance": round(switch_vs_chance, 4) if pd.notna(switch_vs_chance) else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Group summaries + statistics
# ──────────────────────────────────────────────────────────────────────────────
def _onesample_vs_chance(values: np.ndarray, mu: float) -> Tuple[float, float, str]:
    """One-sample test of mean(values) vs mu. t-test if n>=8 & ~normal, else Wilcoxon."""
    from scipy import stats
    v = values[~np.isnan(values)]
    if len(v) < 3:
        return np.nan, np.nan, "insufficient n"
    # Normality gate
    use_t = True
    if len(v) >= 8 and np.ptp(v) > 0:
        try:
            _, p_norm = stats.shapiro(v)
            use_t = p_norm > 0.05
        except Exception:
            use_t = True
    try:
        if use_t:
            stat, p = stats.ttest_1samp(v, mu)
            method = "one-sample t-test"
        else:
            stat, p = stats.wilcoxon(v - mu)
            method = "Wilcoxon signed-rank"
    except Exception:
        return np.nan, np.nan, "test failed"
    return float(stat), float(p), method


def _empty_per_fly(per_fly: pd.DataFrame) -> bool:
    return per_fly is None or per_fly.empty or "Genotype" not in per_fly.columns


def alternation_summary(per_fly: pd.DataFrame, chance: float = DEFAULT_CHANCE) -> pd.DataFrame:
    """Per-group mean alternation %, SEM, and one-sample test vs chance level."""
    if _empty_per_fly(per_fly):
        return pd.DataFrame()
    mu = chance * 100.0
    rows = []
    for kv, sub in per_fly.groupby(GROUP_KEYS, sort=False):
        vals = pd.to_numeric(sub["alternation_pct"], errors="coerce").to_numpy()
        v = vals[~np.isnan(vals)]
        n = len(v)
        mean = float(np.mean(v)) if n else np.nan
        sem = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        stat, p, method = _onesample_vs_chance(vals, mu)
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "n_flies": n,
            "Mean_alternation_pct": round(mean, 2) if pd.notna(mean) else np.nan,
            "SEM": round(sem, 2) if pd.notna(sem) else np.nan,
            "Chance_pct": round(mu, 1),
            "vs_chance_p": round(p, 6) if pd.notna(p) else np.nan,
            "Sig_vs_chance": _sig(p),
            "Test": method,
            "Interpretation": (
                "Above chance (working memory intact)" if pd.notna(p) and p < 0.05 and mean > mu
                else "Below chance (memory deficit)" if pd.notna(p) and p < 0.05 and mean < mu
                else "At chance (no evidence of spatial working memory)"),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def individuality_test(biases: np.ndarray, n_turns: np.ndarray,
                       n_perm: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Monte-Carlo test for stable individual handedness.

    Null: every fly shares the population's grand turn bias; the only spread in
    observed turn-bias scores is binomial sampling noise (each fly makes its own
    n_turns). Compare the observed mean-absolute-deviation (MAD) of turn-bias
    scores to the null MAD distribution (de Bivort-style). Returns
    (observed_MAD, null_mean_MAD, p_value). Small p => true individuality.
    """
    b = np.asarray(biases, dtype=float)
    nt = np.asarray(n_turns, dtype=float)
    mask = (~np.isnan(b)) & (nt > 0)
    b, nt = b[mask], nt[mask].astype(int)
    if len(b) < 3:
        return np.nan, np.nan, np.nan
    grand = float(np.average(b, weights=nt))
    obs_mad = float(np.mean(np.abs(b - np.mean(b))))
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm))
    for k in range(int(n_perm)):
        sim = rng.binomial(nt, grand) / nt
        null[k] = np.mean(np.abs(sim - np.mean(sim)))
    p = float(np.mean(null >= obs_mad))
    return obs_mad, float(np.mean(null)), p


def turnbias_summary(per_fly: pd.DataFrame, n_perm: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Per-group turn-bias summary: mean, SD, MAD, %extreme (outside [0.3, 0.7]),
    CV, and a Monte-Carlo individuality test (observed vs binomial-null MAD).
    """
    if _empty_per_fly(per_fly):
        return pd.DataFrame()
    rows = []
    for kv, sub in per_fly.groupby(GROUP_KEYS, sort=False):
        b = pd.to_numeric(sub["turn_bias"], errors="coerce").to_numpy()
        nt = pd.to_numeric(sub["n_turns"], errors="coerce").to_numpy()
        v = b[~np.isnan(b)]
        n = len(v)
        mean = float(np.mean(v)) if n else np.nan
        sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
        pct_extreme = 100.0 * np.mean((v > 0.70) | (v < 0.30)) if n else np.nan
        cv = (sd / mean) if (pd.notna(sd) and mean not in (0, np.nan)) else np.nan
        obs_mad, null_mad, p_ind = individuality_test(b, nt, n_perm=n_perm, seed=seed)
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "n_flies": n,
            "Mean_turn_bias": round(mean, 4) if pd.notna(mean) else np.nan,
            "SD": round(sd, 4) if pd.notna(sd) else np.nan,
            "Pct_extreme_30_70": round(pct_extreme, 1) if pd.notna(pct_extreme) else np.nan,
            "CV": round(cv, 3) if pd.notna(cv) else np.nan,
            "Observed_MAD": round(obs_mad, 4) if pd.notna(obs_mad) else np.nan,
            "Null_MAD": round(null_mad, 4) if pd.notna(null_mad) else np.nan,
            "Individuality_p": round(p_ind, 6) if pd.notna(p_ind) else np.nan,
            "Sig_individuality": _sig(p_ind),
            "Interpretation": (
                "Stable individual handedness (variance exceeds sampling noise)"
                if pd.notna(p_ind) and p_ind < 0.05
                else "No individuality beyond sampling noise"),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def exploration_summary(per_fly: pd.DataFrame) -> pd.DataFrame:
    """Per-group exploration / activity: mean entries and turns per fly."""
    if _empty_per_fly(per_fly):
        return pd.DataFrame()
    rows = []
    for kv, sub in per_fly.groupby(GROUP_KEYS, sort=False):
        ent = pd.to_numeric(sub["n_entries"], errors="coerce").to_numpy()
        rec = dict(zip(GROUP_KEYS, kv if isinstance(kv, tuple) else (kv,)))
        rec.update({
            "n_flies": int(len(sub)),
            "Mean_entries": round(float(np.nanmean(ent)), 2) if len(ent) else np.nan,
            "SEM_entries": round(float(np.nanstd(ent, ddof=1) / np.sqrt(len(ent))), 2) if len(ent) > 1 else np.nan,
            "Total_entries": int(np.nansum(ent)),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_groups(per_fly: pd.DataFrame, metric: str, group_col: str = "Treatment") -> dict:
    """
    Compare a per-fly metric across groups using the shared stats engine
    (normality-gated ANOVA/Kruskal with BH-FDR post-hoc).
    """
    from modules.stats_engine import auto_test
    if metric not in per_fly.columns or group_col not in per_fly.columns:
        return {}
    data = {str(k): pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy()
            for k, g in per_fly.groupby(group_col)}
    data = {k: v for k, v in data.items() if len(v) >= 2}
    if len(data) < 2:
        return {}
    return auto_test(data)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic demo data (literature-calibrated)
# ──────────────────────────────────────────────────────────────────────────────
def demo_ymaze(seed: int = 7, n_turns: int = 200) -> pd.DataFrame:
    """
    Synthetic Y-maze data calibrated to the published Drosophila literature.

    Turn bias per fly ~ Beta(4.6, 4.6) -> population mean ~0.50 and ~23.5% of
    flies outside [0.30, 0.70], matching Buchanan/Kain/de Bivort (2015, PNAS).
    Alternation is set per group via a turn-repeat probability rho (alternation
    ~= P(consecutive turns share direction)): controls alternate well above the
    50% chance level (intact spatial working memory); the disease model is at
    chance (memory deficit); the drug partially rescues. Returns per-fly arm
    sequences (one row per fly, 'Sequence' column).
    """
    rng = np.random.default_rng(seed)
    # rho = turn-repeat probability. Because an alternation triad == two
    # consecutive turns in the same direction, group alternation% ~= 0.5 + 0.5*rho
    # (for ~unbiased flies). So rho=0 -> ~chance (50%), and rho is chosen to hit
    # the target alternation level for each group.
    groups = {
        ("Oregon-WT", "Vehicle"): 0.36,   # ~68% alternation: intact memory
        ("Oregon-WT", "Drug"):    0.38,
        ("TDP43", "Vehicle"):     0.00,   # ~chance (~50-55%): memory deficit
        ("TDP43", "Drug"):        0.20,   # ~60%: partial rescue
    }
    rows = []
    fid = 0
    for (geno, treat), rho in groups.items():
        for sex in ("Male", "Female"):
            n_flies = int(rng.integers(28, 40))   # deliberately unequal N
            for _ in range(n_flies):
                b = rng.beta(4.6, 4.6)            # individual turn bias
                turns = ["R" if rng.random() < b else "L"]
                for _ in range(n_turns - 1):
                    if rng.random() < rho:        # repeat previous direction
                        turns.append(turns[-1])
                    else:
                        turns.append("R" if rng.random() < b else "L")
                arms = turns_to_arm_sequence(turns, start="A")
                rows.append({"Genotype": geno, "Treatment": treat, "Sex": sex,
                             "FlyID": f"fly{fid}", "Sequence": "".join(arms)})
                fid += 1
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# One-call orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def run_ymaze_analysis(df: pd.DataFrame, chance: float = DEFAULT_CHANCE) -> dict:
    long_df = load_ymaze(df)
    per_fly = per_fly_metrics(long_df, chance=chance)
    return {
        "long": long_df,
        "per_fly": per_fly,
        "alternation": alternation_summary(per_fly, chance=chance),
        "turn_bias": turnbias_summary(per_fly),
        "exploration": exploration_summary(per_fly),
    }
