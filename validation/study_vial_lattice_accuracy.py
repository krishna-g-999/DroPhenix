"""
study_vial_lattice_accuracy.py
How often is the cylinder count right, on this laboratory's own recordings?

WHY THIS SCRIPT EXISTS
----------------------
`climbing_video.auto_n_vials` has never been measured. It is used to decide how
many regions to carve out of a frame, and a wrong answer does not fail loudly:
an over-count invents a cylinder, fills it with whatever is in that part of the
frame, and reports a climbing index for it next to the real ones. On this
laboratory's own footage the blob-counting estimator returned 2 for a
one-cylinder paraquat recording and 5 for the three-cylinder Day 7 rig.

Ground truth exists and is independent of the software: the paraquat
experiment's own manifest records `cylinders_in_frame` for all 62 recordings,
written when the videos were taken. That is a labelled set, so the question
"how accurate is the count" has an answer rather than an opinion.

WHAT IS COMPARED
----------------
  legacy  -- `climbing_video.auto_n_vials`, the blob counter
  lattice -- `vial_lattice.fit_lattice` on the activity profile alone
  joint   -- `vial_lattice.fit_lattice` on activity AND the wall structure

All three read the SAME activity map and the SAME background, computed once per
recording and cached, so the comparison isolates the estimator and nothing
else. Decoding a video is the whole cost here (tens of seconds); the three
estimators together cost a few seconds.

Results are stratified by the true count and by paraquat dose. The stratified
reading is the one that matters: an estimator that is accurate overall but
fails on the impaired arm makes the instrument's sensitivity depend on the
phenotype, which is worse than being uniformly mediocre.

Run: ./.venv/Scripts/python.exe validation/study_vial_lattice_accuracy.py
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import climbing_video as CV  # noqa: E402
from modules import vial_lattice as VL  # noqa: E402

try:
    from validation import lab_data as _LAB
except Exception:                                      # direct-script use
    import lab_data as _LAB

PARA = (_LAB.climbing_root() or Path("_absent")) / "Paraquat climbing assay 010626_new set"
MANIFEST = PARA / "paraquat_video_manifest.csv"

# Profiles are cached beside this script's output, not in the repository: they
# are derived from unpublished recordings. One pass over a video costs tens of
# seconds, and this study is meant to be re-run as the estimator changes.
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "vial_profiles.npz"


def manifest_rows() -> List[dict]:
    if not MANIFEST.is_file():
        return []
    with MANIFEST.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_cache() -> Dict[str, np.ndarray]:
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return {}


def save_cache(d: Dict[str, np.ndarray]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **d)


def profiles_for(path: Path, cache: Dict[str, np.ndarray]) -> Optional[dict]:
    """
    The activity profile and the structure profile for one recording.

    Both are derived from one decode pass and cached, because the decode is the
    entire cost of this study and the estimators under comparison are cheap.
    """
    key = path.name
    ka, ks = key + "::act", key + "::str"
    if ka in cache and ks in cache:
        return {"activity_profile": cache[ka], "structure": cache[ks],
                "cached": True}
    if not path.is_file():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bg = CV.build_background(str(path), start_s=4.0)
            act = CV._activity_map(str(path), bg, start_s=4.0)
    except Exception as exc:
        print(f"    {key}: decode failed: {exc!r}")
        return None
    ap = act.sum(axis=0).astype(np.float64)
    sp = VL.structure_profile(bg)
    cache[ka], cache[ks] = ap, sp
    return {"activity_profile": ap, "structure": sp, "cached": False}


def legacy_count(ap: np.ndarray) -> int:
    """
    `auto_n_vials`'s decision rule, applied to the cached profile.

    Reimplemented here rather than calling `auto_n_vials`, which would decode
    the video a second time. The rule is copied from it exactly -- bands at
    `max_vials`, then keep those carrying at least 25% of the strongest band --
    and the equality of the two is asserted on real recordings by
    `validation/validate_vial_lattice.py` section E, so this shortcut is
    checked rather than assumed.
    """
    import cv2
    bands = CV._bands_from_profile(ap, n_vials=12)
    if not bands:
        return 0
    smooth = cv2.GaussianBlur(ap.reshape(1, -1).astype(np.float32), (31, 1), 0).ravel()
    strength = np.array([float(smooth[a:b].sum()) for a, b in bands])
    return int((strength >= 0.25 * strength.max()).sum())


def main() -> int:
    if not MANIFEST.is_file():
        print("No paraquat manifest on this machine; nothing to measure.")
        print(_LAB.describe_roots())
        return 0

    rows = manifest_rows()
    cache = load_cache()
    n_cached_at_start = len(cache)

    print("=" * 86)
    print("CYLINDER COUNT vs THE EXPERIMENT'S OWN MANIFEST")
    print("=" * 86)
    print(f"{len(rows)} recordings in the manifest; profiles cached: "
          f"{n_cached_at_start // 2}")
    print()
    print("%-14s %-28s %5s %7s %8s %7s %9s"
          % ("arm", "video", "true", "legacy", "lattice", "joint", "dBIC"))

    recs = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        folder, video = r.get("folder", ""), r.get("video", "")
        try:
            n_true = int(float(r["cylinders_in_frame"]))
        except (KeyError, TypeError, ValueError):
            continue
        path = PARA / folder / video
        prof = profiles_for(path, cache)
        if prof is None:
            print("%-14s %-28s %5d   (no profile)" % (folder[:14], video[:28], n_true))
            continue
        ap, sp = prof["activity_profile"], prof["structure"]
        n_leg = legacy_count(ap)
        f_act = VL.fit_lattice(ap)
        f_joint = VL.fit_lattice(ap, structure=sp)
        recs.append({"folder": folder, "video": video, "n_true": n_true,
                     "legacy": n_leg, "lattice": f_act.n_vials,
                     "joint": f_joint.n_vials, "dbic": f_joint.delta_bic,
                     "confident": f_joint.confident,
                     "treatment": r.get("Treatment", ""),
                     "genotype": r.get("Genotype", "")})
        print("%-14s %-28s %5d %7d %8d %7d %9.1f%s"
              % (folder[:14], video[:28], n_true, n_leg, f_act.n_vials,
                 f_joint.n_vials, f_joint.delta_bic,
                 "" if f_joint.confident else "  (not confident)"))
        if i % 10 == 0 and not prof["cached"]:
            save_cache(cache)                      # survive an interruption

    save_cache(cache)
    if not recs:
        print("\nNo recordings could be read.")
        return 0

    print("\n" + "=" * 86)
    print("AGREEMENT WITH THE MANIFEST")
    print("=" * 86)
    n = len(recs)
    for tag in ("legacy", "lattice", "joint"):
        hit = sum(r[tag] == r["n_true"] for r in recs)
        over = sum(r[tag] > r["n_true"] for r in recs)
        under = sum(r[tag] < r["n_true"] for r in recs)
        print("%-8s exact %3d/%3d = %5.1f%%   over-count %3d   under-count %3d"
              % (tag, hit, n, 100.0 * hit / n, over, under))

    print("\nBy the true number of cylinders:")
    print("  %-6s %5s %10s %10s %10s" % ("true", "n", "legacy", "lattice", "joint"))
    for t in sorted({r["n_true"] for r in recs}):
        sub = [r for r in recs if r["n_true"] == t]
        print("  %-6d %5d %9.1f%% %9.1f%% %9.1f%%"
              % (t, len(sub),
                 100.0 * sum(r["legacy"] == t for r in sub) / len(sub),
                 100.0 * sum(r["lattice"] == t for r in sub) / len(sub),
                 100.0 * sum(r["joint"] == t for r in sub) / len(sub)))

    print("\nBy paraquat dose -- does the estimator fail on the impaired arms?")
    print("  %-10s %-10s %5s %10s %10s %10s"
          % ("genotype", "dose", "n", "legacy", "lattice", "joint"))
    keys = sorted({(r["genotype"], r["treatment"]) for r in recs})
    for g, t in keys:
        sub = [r for r in recs if r["genotype"] == g and r["treatment"] == t]
        if not sub:
            continue
        print("  %-10s %-10s %5d %9.1f%% %9.1f%% %9.1f%%"
              % (g[:10], t[:10], len(sub),
                 100.0 * sum(r["legacy"] == r["n_true"] for r in sub) / len(sub),
                 100.0 * sum(r["lattice"] == r["n_true"] for r in sub) / len(sub),
                 100.0 * sum(r["joint"] == r["n_true"] for r in sub) / len(sub)))

    conf = [r for r in recs if r["confident"]]
    unconf = [r for r in recs if not r["confident"]]
    print("\nDoes the confidence flag mean anything?")
    for label, sub in (("confident", conf), ("not confident", unconf)):
        if sub:
            print("  %-14s %3d recordings, joint exact %5.1f%%"
                  % (label, len(sub),
                     100.0 * sum(r["joint"] == r["n_true"] for r in sub) / len(sub)))
        else:
            print("  %-14s none" % label)

    wrong = [r for r in recs if r["joint"] != r["n_true"]]
    if wrong:
        print("\nEvery recording the joint fit got wrong:")
        for r in wrong:
            print("  %-14s %-28s manifest %d, joint %d, dBIC %.1f%s"
                  % (r["folder"][:14], r["video"][:28], r["n_true"], r["joint"],
                     r["dbic"], "" if r["confident"] else " (not confident)"))

    print("\n%.1f s; profiles cached at %s" % (time.time() - t0, CACHE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
