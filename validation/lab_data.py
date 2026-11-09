"""
lab_data.py
Where this laboratory's own recordings are, resolved rather than hardcoded.

WHY THIS EXISTS
---------------
Nine validation suites hardcoded `D:/DroPhenix/Testing data/...`. The data is
on a different drive -- `C:/Testing data/DAM` and `C:/Climbing assay` -- so for
months every one of those suites reported SKIPPED, and `validation/run_all.py`
counted 26 skips as normal. Nothing was broken and nothing was missing: a drive
letter was wrong, and the suites that exercise the laboratory's own real data
silently did not run.

That is the worst failure mode this project has a name for. A suite that skips
reads as coverage in the summary table, and the one place a skip is visible is
a line that scrolls past. `run_all.py` prints SKIPPED, but no one reads 26 of
them.

So the paths are resolved at run time, from, in order:

  1. the `DROPHENIX_LAB_DATA` environment variable, if set -- one directory
     that contains `DAM/` and/or `Climbing assay/`;
  2. a list of known locations, searched in order;
  3. nothing, in which case the caller skips as before -- but
     `describe_roots()` says exactly which candidates were tried, so a skip is
     diagnosable instead of mysterious.

Adding a machine means adding one line to `_CANDIDATES`, not editing nine
suites.

NOTE ON WHAT IS AND IS NOT IN THE REPOSITORY. None of this data is in the
repository and none of it may be: it is the laboratory's own unpublished
recordings, and this repository is public. These are paths to local files, and
the suites that use them SKIP cleanly on any machine that does not have them,
which is what lets a public clone validate without them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Known locations, searched in order. The first entry that exists and contains
# the requested subtree wins.
_CANDIDATES: List[Path] = [
    Path(r"C:/Testing data"),
    Path(r"D:/DroPhenix/Testing data"),
    Path(r"D:/Testing data"),
    Path.home() / "Testing data",
]

# The climbing recordings sit in their own tree on this machine rather than
# under the DAM root, so they are resolved separately.
_CLIMBING_CANDIDATES: List[Path] = [
    Path(r"C:/Climbing assay"),
    Path(r"C:/Testing data/Climbing assay"),
    Path(r"D:/DroPhenix/Testing data/Climbing assay"),
    Path.home() / "Climbing assay",
]

_ENV = "DROPHENIX_LAB_DATA"

#: Set this to 1 to make both roots report ABSENT even on a machine that has
#: the recordings.
#:
#: The suite is meant to be portable: a clone with no laboratory data should
#: SKIP the suites built around that data and pass everything else. That claim
#: had never been tested, because the suite had only ever run on the machine
#: holding the data -- and the first CI run on a clean runner failed, so the
#: claim was false. Testing it needed either a clean machine or this switch;
#: this switch is cheaper and can be run before every push.
#:
#:     DROPHENIX_NO_LAB_DATA=1 ./.venv/Scripts/python.exe validation/run_all.py
_ENV_NONE = "DROPHENIX_NO_LAB_DATA"


def _suppressed() -> bool:
    """True when the caller has asked us to behave as a machine with no data."""
    return str(os.environ.get(_ENV_NONE, "")).strip().lower() in (
        "1", "true", "yes", "on")


def _env_root() -> Optional[Path]:
    v = os.environ.get(_ENV)
    if not v:
        return None
    p = Path(v)
    return p if p.is_dir() else None


def dam_root() -> Optional[Path]:
    """
    The directory holding the DAM recordings, or None.

    Expected to contain `25th day/`, `15th day/` and the monitor files. The
    environment variable may point either at the parent (containing `DAM/`) or
    at the DAM directory itself, because both are natural readings of it.
    """
    if _suppressed():
        return None
    env = _env_root()
    if env is not None:
        for cand in (env / "DAM", env):
            if (cand / "25th day").is_dir() or any(cand.glob("Monitor*.txt")):
                return cand
    for base in _CANDIDATES:
        cand = base / "DAM"
        if cand.is_dir() and ((cand / "25th day").is_dir()
                              or any(cand.glob("Monitor*.txt"))):
            return cand
    return None


def climbing_root() -> Optional[Path]:
    """The directory holding the climbing recordings, or None."""
    if _suppressed():
        return None
    env = _env_root()
    if env is not None:
        for cand in (env / "Climbing assay", env):
            if cand.is_dir() and _looks_like_climbing(cand):
                return cand
    for cand in _CLIMBING_CANDIDATES:
        if cand.is_dir() and _looks_like_climbing(cand):
            return cand
    return None


def _looks_like_climbing(p: Path) -> bool:
    """A climbing root holds video files somewhere in its first two levels."""
    for pat in ("*.MP4", "*.mp4", "*.MOV", "*.mov"):
        if any(p.glob(pat)) or any(p.glob("*/" + pat)):
            return True
    return False


def describe_roots() -> str:
    """
    A one-paragraph account of what was found and what was tried.

    Printed by a suite when it skips, so that a skip states its cause instead
    of merely occurring.
    """
    lines = []
    d, c = dam_root(), climbing_root()
    lines.append("DAM root      : %s" % (d if d else "NOT FOUND"))
    lines.append("climbing root : %s" % (c if c else "NOT FOUND"))
    if _suppressed():
        lines.append("%s is set, so both roots are reported ABSENT on purpose "
                     "(portability check)." % _ENV_NONE)
    if os.environ.get(_ENV):
        lines.append("%s = %r" % (_ENV, os.environ[_ENV]))
    else:
        lines.append("%s is not set" % _ENV)
    if d is None:
        lines.append("DAM candidates tried: "
                     + ", ".join(str(b / "DAM") for b in _CANDIDATES))
    if c is None:
        lines.append("climbing candidates tried: "
                     + ", ".join(str(b) for b in _CLIMBING_CANDIDATES))
    return "\n".join("       " + ln for ln in lines)


def require(path: Optional[Path], what: str) -> Path:
    """Raise with the diagnosis attached, for callers that cannot skip."""
    if path is None:
        raise FileNotFoundError(
            "%s could not be located.\n%s" % (what, describe_roots()))
    return path


if __name__ == "__main__":
    print("DroPhenix laboratory data roots")
    print(describe_roots())
    d = dam_root()
    if d:
        print("\nDAM contents:")
        for p in sorted(d.rglob("Monitor*.txt")):
            print("   %10d  %s" % (p.stat().st_size, p.relative_to(d)))
        for p in sorted(d.rglob("channel_map*.csv")):
            print("   %10d  %s" % (p.stat().st_size, p.relative_to(d)))
    c = climbing_root()
    if c:
        vids = [p for pat in ("*.MP4", "*.mp4", "*.MOV", "*.mov")
                for p in c.rglob(pat)]
        print("\nclimbing: %d video files" % len(vids))
        import collections
        by = collections.Counter(str(p.parent.relative_to(c)) for p in vids)
        for k, v in sorted(by.items()):
            print("   %3d  %s" % (v, k))
