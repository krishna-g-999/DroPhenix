"""
Validation suite: the countable claims in README.md are the counts the code produces.

WHY THIS SUITE EXISTS
---------------------
Register N116. `README.md` advertised "41 suites, 1,067 checks". The runner
listed 57 suites on the working branch and 42 on the public default branch, so
the number was wrong on BOTH branches, and no counting convention anyone could
reconstruct yielded 1,067 checks. The suite count is the first number a
reviewer checks, and a number typed by hand into prose drifts away from the
code the moment a suite is added.

This is therefore not a test of the science. It is a test of the one class of
claim that costs nothing to verify mechanically and costs a submission to get
wrong: a number in the documentation that the repository itself can count.

WHAT IS VALIDATED HERE, AND AT WHICH LAYER
------------------------------------------
A. THE SUITE COUNT IN README.md EQUALS THE NUMBER run_all.py ACTUALLY LISTS
   (layer 1, bookkeeping). Both figures are recomputed from the files on disk
   and compared. There is no tolerance: an added suite must be reflected in the
   README or this fails.

B. THE README DOES NOT CARRY A CHECK COUNT IT CANNOT SUBSTANTIATE (layer 1).
   The "1,067 checks" claim is asserted ABSENT. If someone reinstates a
   hand-typed check count, this fails and points them at the runner's own
   printed total instead.

C. EVERY DEPENDENCY THE PRODUCTION CODE IMPORTS UNGUARDED IS DECLARED
   (layer 1). Register N113: `dam_report.build()` imported `docx` unguarded
   while `python-docx` sat only in requirements-validation.txt, so the
   documented CLI died at its deliverable on a clean install. This walks the
   AST of the production modules, collects every top-level third-party import,
   and asserts each is named in requirements.txt. Imports inside a
   try/except ImportError are exempt, because those degrade by design.

WHAT THIS SUITE DOES NOT ESTABLISH
----------------------------------
Nothing about whether the suites PASS, whether their science is right, or
whether a declared version floor is high enough -- only that the counts and the
dependency list are honest. A green result here is layer-1 bookkeeping and must
never be cited as evidence of anything else.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    if ok:
        print(f"  [PASS] {label}")
    else:
        FAILURES.append(label if not detail else f"{label} -- {detail}")
        print(f"  [FAIL] {label}" + (f"\n         {detail}" if detail else ""))


# ── A. the suite count ───────────────────────────────────────────────────────
def suites_listed_by_run_all() -> set[str]:
    """Every suite name run_all.py dispatches, read from its SUITES literal."""
    src = (ROOT / "validation" / "run_all.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "SUITES":
                    # EVERY string in the SUITES literal, with no prefix
                    # filter. The filter used to require `validate_` or
                    # `verify_`, which silently excluded `demo_unequal_n` --
                    # a suite `run_all` genuinely runs. So the runner ran 60
                    # suites, the README claimed 59, and THIS CHECK PASSED,
                    # because it undercounted by exactly the same one it was
                    # supposed to catch. A guard whose blind spot matches the
                    # drift it polices is worse than no guard: it certifies
                    # the error.
                    return {
                        e.value for e in ast.walk(node.value)
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                        and e.value.strip()
                    }
    return set()


def test_A_suite_count() -> None:
    print("\nA. the README's suite count equals what run_all.py lists")
    listed = suites_listed_by_run_all()
    check(len(listed) > 0, "run_all.py's SUITES literal was parsed",
          "could not find a SUITES assignment")
    if not listed:
        return

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claimed = {int(m) for m in re.findall(r"(\d+) suites", readme)}
    check(len(claimed) == 1,
          "README.md states ONE suite count, consistently",
          f"found {sorted(claimed) or 'none'}; every mention must agree")
    if len(claimed) != 1:
        return
    n_claimed = claimed.pop()
    check(n_claimed == len(listed),
          f"README's suite count ({n_claimed}) == run_all.py's ({len(listed)})",
          f"README says {n_claimed}, run_all.py lists {len(listed)}. "
          f"Update README.md -- do not change this threshold.")

    # every listed suite is a file that exists, or is one of the deliberately
    # excluded lab-data suites. A name in SUITES with no file is a silent skip.
    missing = sorted(
        s for s in listed
        if not (ROOT / "validation" / f"{s}.py").exists()
    )
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    unexplained = [s for s in missing if s not in gitignore]
    check(not unexplained,
          "every suite in SUITES either exists or is named in .gitignore",
          f"absent and unexplained: {unexplained}")
    if missing:
        print(f"         ({len(missing)} excluded by .gitignore, as designed: "
              f"{missing})")


# ── B. no unsubstantiated check count ────────────────────────────────────────
def test_B_no_hand_typed_check_count() -> None:
    print("\nB. the README carries no check count it cannot substantiate")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    bad = re.findall(r"[\d,]+\s+checks", readme)
    check(not bad,
          "no hand-typed 'N checks' claim in README.md",
          f"found {bad}. The runner prints its own total; cite that instead "
          f"of retyping a number that drifts (register N116).")


# ── C. unguarded production imports are declared ─────────────────────────────
#: import name -> the distribution that provides it, where they differ.
DIST_FOR = {
    "docx": "python-docx", "cv2": "opencv-python", "sklearn": "scikit-learn",
    "PIL": "Pillow", "skimage": "scikit-image", "dateutil": "python-dateutil",
    "yaml": "PyYAML", "serial": "pyserial", "umap": "umap-learn",
    "statsmodels": "statsmodels", "streamlit_webrtc": "streamlit-webrtc",
    "streamlit_drawable_canvas": "streamlit-drawable-canvas",
    "ffmpeg": "ffmpeg-python",
}
STDLIB = set(sys.stdlib_module_names)


def _guarded_imports(tree: ast.AST) -> set[int]:
    """Line numbers of every import inside a try/except that catches ImportError.

    Those degrade by design -- `trajectory_cluster` falls back to PCA when
    `umap` is absent -- so they are not required to be declared.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import = any(
            h.type is not None and (
                (isinstance(h.type, ast.Name)
                 and h.type.id in ("ImportError", "ModuleNotFoundError", "Exception"))
                or (isinstance(h.type, ast.Tuple) and any(
                    isinstance(e, ast.Name)
                    and e.id in ("ImportError", "ModuleNotFoundError", "Exception")
                    for e in h.type.elts))
            )
            for h in node.handlers
        )
        if catches_import:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    guarded.add(sub.lineno)
    return guarded


def test_C_declared_dependencies() -> None:
    print("\nC. every unguarded third-party import in production code is declared")
    # Strip comments and blank lines BEFORE matching. Matching the raw file
    # text is how the first version of this check passed vacuously: the
    # requirements file explains N113 in a comment that names `python-docx`,
    # so deleting the actual requirement left the word behind and the check
    # still found it. Only declared requirements count, never prose about them.
    req_lines = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            req_lines.append(line)
    req = "\n".join(req_lines)

    files = [ROOT / "app.py", ROOT / "drophenix_cli.py"]
    files += sorted((ROOT / "modules").glob("*.py"))
    files = [f for f in files if f.exists()]

    local = {p.stem for p in (ROOT / "modules").glob("*.py")} | {
        "modules", "validation", "scripts", "app", "drophenix_cli"}

    undeclared: dict[str, list[str]] = {}
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            check(False, f"{f.name} parses", str(exc))
            continue
        guarded = _guarded_imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if node.lineno in guarded:
                continue
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            else:
                if node.level:          # relative import -- local by definition
                    continue
                names = [(node.module or "").split(".")[0]]
            for n in names:
                if not n or n in STDLIB or n in local:
                    continue
                dist = DIST_FOR.get(n, n).lower()
                if dist not in req and n.lower() not in req:
                    undeclared.setdefault(n, []).append(
                        f"{f.name}:{node.lineno}")

    check(not undeclared,
          "no unguarded production import is missing from requirements.txt",
          "; ".join(f"{k} ({', '.join(v[:3])})" for k, v in undeclared.items()))


def main() -> int:
    print("=" * 78)
    print("VALIDATION: README.md's countable claims and the declared dependencies")
    print("=" * 78)
    print("Layer 1 ONLY: this establishes that the counts and the dependency")
    print("list are honest. It says NOTHING about whether any suite passes.")
    test_A_suite_count()
    test_B_no_hand_typed_check_count()
    test_C_declared_dependencies()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL README-CLAIM AND DEPENDENCY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
