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

import platform
import sys
from typing import Dict

__all__ = ["versions", "stamp", "provenance_line"]

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


def stamp(df, inplace: bool = False):
    """Attach the versions to a DataFrame's `.attrs`, where they survive a copy.

    Deliberately NOT added as columns: a provenance column repeated down every row
    is noise in an analysis table, and `.attrs` is where pandas keeps metadata.
    """
    target = df if inplace else df.copy()
    target.attrs["drophenix_provenance"] = versions()
    return target
