"""
What produced this number: the software and library versions behind every output.

WHY THIS EXISTS
---------------
A result that cannot be attributed to a specific build is not reproducible, and
none of DroPhenix's outputs recorded one. Two laboratories running "DroPhenix"
could differ in OpenCV minor version -- which changes contour and threshold
behaviour -- and nothing in the exported table would show it.

This is deliberately tiny and dependency-free so it can be called from anywhere,
including inside an exception handler.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = ["versions", "stamp", "provenance_line",
           "manifest", "write_manifest", "git_commit", "checksum"]

#: Kept in step with app.py's _VERSION. A manifest that cannot name the build
#: it came from is not a provenance record.
DROPHENIX_VERSION = "3.1"

#: Libraries whose version can change a NUMBER, not merely a plot.
_NUMERICAL = ("numpy", "scipy", "pandas", "cv2", "statsmodels", "sklearn")


def versions() -> Dict[str, str]:
    """Version of Python and of every library that can move a result."""
    out = {"python": platform.python_version(),
           "platform": f"{platform.system()} {platform.release()}"}
    for name in _NUMERICAL:
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "absent"
    return out


def provenance_line() -> str:
    """One line, suitable for a CSV comment or a figure caption."""
    v = versions()
    return ("DroPhenix | " +
            " ".join(f"{k}={v[k]}" for k in ("python", "numpy", "scipy", "pandas",
                                             "cv2", "statsmodels") if k in v))


def git_commit(repo_root: Optional[Path] = None) -> str:
    """The commit the analysis ran from, or "unknown".

    Tried without a subprocess first, because an export must not depend on git
    being installed: `.git/HEAD` plus the ref file is enough in the normal case.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    head = root / ".git" / "HEAD"
    try:
        if head.exists():
            txt = head.read_text(encoding="utf-8").strip()
            if txt.startswith("ref:"):
                ref = (root / ".git" / txt.split(" ", 1)[1].strip())
                if ref.exists():
                    return ref.read_text(encoding="utf-8").strip()[:40]
                packed = root / ".git" / "packed-refs"
                if packed.exists():
                    want = txt.split(" ", 1)[1].strip()
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(" " + want):
                            return line.split(" ", 1)[0][:40]
            elif txt:
                return txt[:40]
    except Exception:
        pass
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:40]
    except Exception:
        pass
    return "unknown"


def checksum(obj: Any) -> str:
    """SHA-256 of a file path, raw bytes, or a DataFrame's canonical CSV form.

    A DataFrame is hashed via its CSV rendering so that the same table hashes
    the same whatever produced it, and so an exported CSV can be checked against
    the manifest that travels with it.
    """
    h = hashlib.sha256()
    try:
        if hasattr(obj, "to_csv"):
            h.update(obj.to_csv(index=False).encode("utf-8"))
        elif isinstance(obj, (bytes, bytearray)):
            h.update(bytes(obj))
        else:
            p = Path(str(obj))
            if p.exists() and p.is_file():
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            else:
                h.update(str(obj).encode("utf-8"))
    except Exception:
        return "unhashable"
    return "sha256:" + h.hexdigest()


def manifest(inputs: Optional[Mapping[str, Any]] = None,
             parameters: Optional[Mapping[str, Any]] = None,
             seeds: Optional[Mapping[str, Any]] = None,
             analysis: str = "",
             extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The full reproducibility record for one analysis output.

    `stamp()` writes the versions into `DataFrame.attrs`, which does not survive
    `to_csv` -- so every export this project ever produced was unattributed.
    This returns a plain dict intended to be written beside the export as a
    sidecar file, where it does survive.

    Parameters
    ----------
    inputs     : {name: path | bytes | DataFrame} -- each is checksummed, so a
                 result can be tied to the exact input that produced it.
    parameters : the analysis settings that change the numbers.
    seeds      : random seeds, without which a stochastic result is not
                 reproducible even given the same code and input.
    """
    v = versions()
    man: Dict[str, Any] = {
        "drophenix_version": DROPHENIX_VERSION,
        "git_commit": git_commit(),
        "analysis": str(analysis),
        "python": v.get("python", "unknown"),
        "platform": v.get("platform", "unknown"),
        "dependencies": {k: val for k, val in v.items()
                         if k not in ("python", "platform")},
        "parameters": dict(parameters or {}),
        "seeds": dict(seeds or {}),
        "input_checksums": {str(k): checksum(val) for k, val in (inputs or {}).items()},
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "working_directory": os.getcwd(),
    }
    if extra:
        man["extra"] = dict(extra)
    return man


def manifest_json(man: Optional[Mapping[str, Any]] = None, **kwargs) -> str:
    """The manifest as indented JSON, ready to drop into a ZIP or a download."""
    return json.dumps(dict(man) if man is not None else manifest(**kwargs),
                      indent=2, sort_keys=True, default=str)


def write_manifest(export_path, man: Optional[Mapping[str, Any]] = None,
                   **kwargs) -> str:
    """Write the manifest beside `export_path` as `<name>.manifest.json`.

    Returns the path written. The export itself is not touched: a provenance
    comment inside a CSV would break strict parsers, so the record travels
    alongside instead of inside.
    """
    target = Path(str(export_path))
    out = target.with_name(target.name + ".manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest_json(man, **kwargs), encoding="utf-8")
    return str(out)


def stamp(df, inplace: bool = False):
    """Attach the versions to a DataFrame's `.attrs`, where they survive a copy.

    Deliberately NOT added as columns: a provenance column repeated down every row
    is noise in an analysis table, and `.attrs` is where pandas keeps metadata.
    """
    target = df if inplace else df.copy()
    target.attrs["drophenix_provenance"] = versions()
    return target
