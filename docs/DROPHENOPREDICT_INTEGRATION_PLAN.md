# Bringing DroPhenoPredict into DroPhenix

**Status: PLAN, awaiting approval.** Nothing implemented.

---

## 1. The licence question, corrected

Earlier in this project I argued that merging the two tools would force a GPL-3.0 adoption.
**That reasoning was about the opposite direction** — importing DroPhenix (GPL-3.0) *into*
DroPhenoPredict, which would indeed propagate copyleft to DroPhenoPredict and to everyone
downstream of it.

The direction now proposed is DroPhenoPredict **into** DroPhenix, and that concern does not
apply: you are the sole author of both, so you may license your own code into your own
project on whatever terms you choose. There is no third-party obligation in the way.

Two housekeeping items that a reviewer *will* check, and that are currently wrong:

- **Neither repository contains a LICENSE file.** DroPhenix's GPL-3.0 is asserted on its
  GitHub page but there is no `LICENSE` in the tree; DroPhenoPredict has none at all. Add
  `LICENSE` to DroPhenix before submission, and state the licence of the contributed
  DroPhenoPredict code explicitly in its module headers.
- DroPhenoPredict currently depends on `lifelines`, `sksurv` and other packages absent from
  DroPhenix's `requirements.txt`. Only the parts actually imported should be carried over.

---

## 2. The scientific boundary that must survive the merge

This is the part worth getting right, and it is not a packaging question.

| | DroPhenix | DroPhenoPredict |
|---|---|---|
| answers | *what did I measure* | *what would this line probably do* |
| genotypes | any, free text | 205 fixed DGRP lines |
| basis | an experiment the user ran | a model fitted to public panel data |
| accuracy | the assay's own noise | r ≈ 0.13–0.34 out of sample |

Merging the **code** is fine. Merging the two into one undifferentiated results table is the
overclaiming risk both projects have spent months eliminating. A predicted value at
r ≈ 0.2 and a measured climbing count must never appear in the same column, in the same
plot series, or in the same exported CSV field.

The honest numbers to carry across, from the audit of DroPhenoPredict:

- 8 of 8 trait-sex models are significant under a **matched** permutation test (the shipped
  test was under-powered and called 6 of 8);
- the 90% prediction intervals are **wider than the panel's own marginal spread** in 7 of 8
  models, so they are calibrated but nearly uninformative;
- `adaptive_nested_r_sd` is a Monte-Carlo SD across CV repeats, **not** the SE of r.

All three must be visible wherever a prediction is shown.

---

## 3. Proposed shape: one page, clearly fenced

A single new page, **"Genomic Prediction (DGRP)"**, placed after Historical Controls and
before the video pages — i.e. among the modelling tools, not among the assays.

**What it does**

1. Pick a DGRP line and a trait (climbing, startle, starvation, sleep; each × sex).
2. Show the predicted value with its 90% interval, the modality actually used (genotype
   GRM or RNA-seq expression), and the permutation *p*.
3. Show, beside it and impossible to miss, **the marginal spread of the panel** — because
   for 7 of 8 models the interval is wider than that, and a user is entitled to know the
   prediction is barely narrowing anything.
4. Offer the DroPhenix-schema CSV template already built in `drophenix_export.py`, so a
   real assay can be run and compared.

**What it must not do**

- write a predicted value into any column named `Count`, `Count_climbed` or `n`;
- appear in Data Input, CSV Upload or Export as though it were measured data;
- feed the hierarchical, trajectory or dose-response modules, all of which assume
  observations.

**Visual separation:** predictions on a distinct background with a persistent
"predicted, not measured" badge — the convention DroPhenoPredict's own app already uses.

---

## 4. What to copy, and what to leave

| bring across | why |
|---|---|
| `gblup.py` | REML, BLUP, CV, conformal intervals — audited, correct |
| `multimodal.py` | nested modality selection |
| `predictor.py` | the inference API |
| `drophenix_export.py` | already speaks DroPhenix's schema |
| `models/trained/*.json`, `*.npz` | the fitted artefacts (a few MB) |

| leave behind | why |
|---|---|
| the v2 disease-convergence resource | separate question, separate data, large |
| `scripts/01`–`44` | data-fetch and training; not needed at runtime |
| `data/raw/**` | gigabytes; the fitted model is the deliverable |
| Yang2023 / proteomics / metabolome findings | not part of the predictor |

**Correct the permutation test before copying.** `scripts/13_finalize_multimodal.py`
compares a statistic from the nested adaptive procedure against a null from a single fixed
kernel with δ borrowed from the unpermuted phenotypes. The matched test gives 8/8
predictable, not 6/8. Copying the artefact without re-running it would import the wrong
verdict into DroPhenix.

---

## 5. Order of work

1. Add `LICENSE` to DroPhenix; state licence in the copied module headers.
2. Re-run `scripts/13` with the matched permutation test; regenerate `v1.1.json`.
3. Copy the five modules and the two artefacts into `modules/genomic/`.
4. Add `validate_genomic_prediction.py` to `run_all.py` — the 41 DroPhenoPredict tests do
   not run here and would otherwise be lost.
5. Build the page with the fencing in §3.
6. Update `requirements.txt` with only what the copied code imports.

Steps 1–4 are the ones that carry scientific risk; step 5 is interface work.

---

## 6. What this does NOT give you

It does not make DroPhenix able to predict a TDP-43 or FUS line. The DGRP contains no
transgenic disease models, so those remain an explicitly-labelled extrapolation, exactly as
DroPhenoPredict's own charter states. Anyone hoping the merge yields disease-model
prediction should be told plainly that it does not.
