# Reproducibility and reliability audit

**Date:** 2026-08-17 · **Scope:** the whole tool, not only the video module ·
**Standard applied:** what a methods reviewer at a journal like NAR checks before accepting
that a result can be reproduced by someone else.

Every finding below is a command that was run, not an opinion.

---

## What holds up

**Determinism.** Two suites were run twice and the output byte-compared:

```
validate_hierarchical  : IDENTICAL across runs
validate_control_prior : IDENTICAL across runs
```

**No unseeded randomness anywhere.** A grep for `np.random.rand/randn/randint/choice/
normal/shuffle/permutation` and bare `random.*` across `modules/` and `validation/`
returns nothing: every stochastic path goes through an explicitly seeded
`np.random.default_rng`. Twelve modules and eight suites seed generators.

**The validation suite is real and it runs.** 26 suites, all passing, from one command
(`validation/run_all.py`). Several are checked against published raw data (Freireich 1963,
PlantGrowth, drc `ryegrass`, Konopka & Benzer 1971) rather than against themselves.

**Optional heavy dependencies are correctly guarded.** `umap` sits inside a `try/except`
in `trajectory_cluster.py`; `sksurv` appears only in the optional cross-package comparison
study. Neither is installed in this environment and nothing breaks — which is the
behaviour you want, but see the caveat below.

---

## Defects, in order of how badly they bite

### 1. A missing dependency broke a clean install — FIXED

`app.py` imported `PIL.Image` unguarded for the Draw Vials canvas while `Pillow` was
absent from `requirements.txt`. It is present in this virtual environment, so every test
passed locally and a fresh install would have crashed on that page. **This was mine, and
it is the exact failure mode a reviewer reproducing from `requirements.txt` would hit
first.** Added, with the reason recorded in the file.

### 2. No output recorded what produced it — FIXED

No exported table, figure or CSV carried the software or library versions behind it. Two
laboratories can differ in an OpenCV minor version — which changes contour and threshold
behaviour, and therefore changes counts — and nothing in the output would reveal it.

`modules/provenance.py` now supplies `versions()`, `provenance_line()` and `stamp(df)`.
Current environment:

```
DroPhenix | python=3.13.13 numpy=2.5.0 scipy=1.18.0 pandas=2.3.3 cv2=4.13.0 statsmodels=0.14.6
```

*Remaining work:* call `stamp()` from the export paths so it actually reaches the files.
Written but not yet wired — flagged rather than claimed.

### 3. Numbers in the documentation are not all regenerable — OPEN, and the largest gap

The project's own rule is that no number in the docs may exist without a script that
regenerates it. Measured against that:

| document | measured-looking numbers | not traceable to any validation script |
|---|---|---|
| VIDEO_PHASE1_DESIGN.md | 25 | **15** |
| VIDEO_AUDIT_2026-08.md | 16 | **8** |
| VIDEO_SCORING.md | 11 | 5 |
| VIDEO_DRAW_ROI_DESIGN.md | 5 | **5** |
| UPLIFT_PLAN.md | 7 | 3 |
| INVERSION_ANALYSIS.md | 5 | 2 |

Most of the untraceable ones are mine, from this month's video work: the r = +0.993 clutter
correlation, the ±78 px camera shift, the backend MAE table, the 216-cell re-measurement,
the box-size sensitivity sweep. They were all genuinely measured — the scripts exist — but
the scripts live in a scratch directory outside the repository, so nobody else can re-run
them. **That is not reproducible, and the honest description is "measured once, by me, on
this machine".**

*Fix:* move those scripts into `validation/` (or an `analysis/` directory) so each number
has a committed regenerating command. This is bookkeeping, not new science, but it is the
difference between a claim and a result.

### 4. `requirements.txt` is loosely pinned and internally inconsistent

The header states the file covers "the Streamlit APPLICATION only", with validation
dependencies in `requirements-validation.txt` (which does exist). But `matplotlib`,
`seaborn`, `pingouin`, `streamlit-drawable-canvas` and `ffmpeg-python` were appended below
the closing note with no upper bounds and no comment. Lower bounds only means a future
NumPy 3 or OpenCV 5 can change results silently.

*Recommendation for submission:* ship a `requirements.lock` produced by `pip freeze` from
the environment that generated the published numbers, alongside the permissive
`requirements.txt` for ordinary use.

### 5. Superseded modules are still shipped and still imported

`video_analyzer.py` (3 importers), `fly_detector.py` (5) and `recording.py` (2) remain in
the tree. `climbing_video.py`'s own docstring records that the first two "fail badly" on
real recordings — three flies found where thirty were present — and `recording.py` was the
module whose FFMPEG-priority setting broke all video reading app-wide earlier in this
project. They are reachable from the app.

*Recommendation:* either delete them or mark them clearly as deprecated in the UI, so a
user cannot reach a detector the project already knows does not work.

---

## Summary

| property | status |
|---|---|
| deterministic re-runs | **verified** |
| seeded randomness throughout | **verified** |
| one-command validation, 26 suites | **verified** |
| validated against external published data | **verified** for stats/survival/dose-response |
| installs cleanly from requirements | **fixed today** |
| outputs record their provenance | **written today, not yet wired** |
| every documented number regenerable | **NO — the main gap** |
| dependency versions pinned for publication | **NO — needs a lockfile** |
| superseded code removed | **NO — still reachable** |

The statistical core is in good shape: seeded, deterministic, externally validated, and
tested against closed forms. The reproducibility gaps are all in the *packaging and
provenance* layer, which is exactly where a methods reviewer looks and exactly what is
cheapest to fix.
