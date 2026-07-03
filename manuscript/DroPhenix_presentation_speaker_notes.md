# DroPhenix presentation — speaker notes

Dual-audience notes (plain-language explanation, expert Q&A, non-scientific/commercial Q&A) for each slide. All content is grounded in the project's verified results.

## Slide 1. Title

IN PLAIN TERMS: DroPhenix is a free website that automatically analyses experiments on fruit flies (Drosophila) — the standard cheap, fast stand-in for human brain disease. We turned six common fly tests into one reliable automated tool and added a new statistical method for comparing drugs fairly.

LEAD WITH: it agrees with the world's gold-standard analysis software on 24/24 checks (perfect correlation) and reproduces published scientific data exactly.

EXPERT Q: "Why flies?" — ~75% of human disease genes conserved; weeks-long lifespan; the standard first-pass ALS/PD/HD model. "Novel or a wrapper?" — both: standardises best practice AND a new ranking method.

NON-SCIENTIST Q: "What does it do?" — you upload your numbers, it makes the correct graphs and statistics and ranks which drug works best. "Who is it for?" — fly labs and pre-clinical drug screening.

## Slide 2. The problem

IN PLAIN TERMS: An "assay" is a standard lab test. Today each is analysed separately in spreadsheets with inconsistent statistics, and there is no fair way to combine results when strains have different numbers of flies.

DEFINE: Kaplan–Meier / log-rank / Cox = the rigorous, clinical-grade ways to analyse lifespan data.

EXPERT Q: "Isn't survival analysis standard?" — In clinical stats yes, but fly papers often report only median; we bring the full toolkit to the bench.

NON-SCIENTIST Q: "Why does unequal numbers matter?" — Averaging a 3-person survey and a 700-person survey equally is misleading; the small noisy group distorts the answer — exactly the fly-strain problem.

## Slide 3. What we developed

IN PLAIN TERMS: The full menu of modules. Six are biological tests (climbing = motor health, lifespan, Y-maze = memory, DAMS = sleep/body-clock, larval crawling, grooming) and three are analytical engines (Motor Decline Rate, trajectory clustering, and the Drug Ranking innovation).

DEFINE: "negative geotaxis" = flies climb up after being tapped down; climbing measures neuromuscular health.

EXPERT Q: "Do the video tools replace dedicated trackers?" — for counting and single-larva speed yes; multi-animal pose tracking is out of scope and stated.

NON-SCIENTIST Q: "Why so many tests?" — different diseases hit movement, memory, sleep, or lifespan; testing across all gives a fuller, more trustworthy picture.

## Slide 4. How it works (pipeline)

IN PLAIN TERMS: This infographic is the assembly line. You load your data (or a video, or the built-in demo); the tool cleans and checks it; the relevant module computes the metrics; the statistics engine runs; and you get tables, figures and CSV files out.

TALKING POINT: every step is deterministic — same input, same output — and inspectable; nothing is a hidden trained model.

EXPERT Q: "What QC happens at parse?" — column auto-detection, format validation, duplicate/outlier flagging, standardisation to long format.

NON-SCIENTIST Q: "How hard is it to use?" — upload a spreadsheet, click; the five-step flow is automatic.

## Slide 5. Statistics decision flow

IN PLAIN TERMS: Good statistics means choosing the RIGHT test for your data, not always the same one. Box 1: DroPhenix checks whether the data are bell-shaped (Shapiro–Wilk), whether groups vary similarly (Levene), and whether samples are big enough. Box 2: it then picks a parametric test if those hold, otherwise a non-parametric one. Box 3: it always reports the effect size, corrects for multiple comparisons, and gives a confidence interval.

DEFINE: effect size = how big the difference is; FDR correction = a safeguard against flukes when testing many things; bootstrap CI = a reliability range from resampling.

EXPERT Q: "Thresholds?" — Shapiro/Levene at 0.05, minimum n=5 for parametric.

NON-SCIENTIST Q: "Why care?" — it's the difference between a result that survives peer review / due-diligence and one that doesn't.

## Slide 6. Module: Climbing

IN PLAIN TERMS: The climbing test. Flies are tapped to the bottom and instinctively climb; we measure the fraction that climb past a line over seconds. The Performance Index (PI) scales this from −100 (none) to +100 (all). AUC and t50 summarise the speed of climbing.

DEFINE: PI = 2×(% climbed) − 100.

EXPERT Q: "What are CRI/SDQ/TSW?" — composite rescue, sex-dimorphism and disease-selectivity indices layered on PI/AUC/t50.

NON-SCIENTIST Q: "What does the output look like?" — a table of scores per group plus climbing curves, downloadable as CSV and 300-dpi images.

## Slide 7. Module: Lifespan

IN PLAIN TERMS: The lifespan test. We track how long flies live, correctly handling those still alive at the end (censoring). Kaplan–Meier draws the survival curve; the log-rank test asks if two groups differ; Cox gives a hazard ratio — how many times higher the risk of death is.

DEFINE: hazard ratio = relative risk of death between groups.

EXPERT Q: "Median convention and ties?" — smallest t with S(t)≤0.5 (R/Prism), Breslow ties for Cox.

NON-SCIENTIST Q: "How do we know it's right?" — it reproduces the famous 1963 leukemia trial's published numbers exactly and matches R and lifelines.

## Slide 8. Module: Y-maze

IN PLAIN TERMS: The memory/behaviour test. In a Y-shaped maze a healthy fly tends to explore the arm it visited least recently (spontaneous alternation) — a working-memory read-out. We also measure left/right handedness and how individual flies differ.

DEFINE: alternation % vs a 50% chance level; turn bias = fraction of right turns.

EXPERT Q: "Why is 50% the chance level?" — under the arm geometry an alternation equals two same-direction turns, so an unbiased fly alternates at exactly 50% — it emerges from the model, not assumed.

NON-SCIENTIST Q: "What does a low score mean?" — poorer working memory, e.g. in a disease or drug-impaired fly.

## Slide 9. Module: DAMS (sleep & circadian)

IN PLAIN TERMS: The sleep and body-clock test, from minute-by-minute activity. A fly is "asleep" after 5 minutes of no movement (the field standard). The circadian period (near 24 h) is found with a periodogram — a method that detects the dominant rhythm in the data.

DEFINE: free-running period = the internal clock length in constant darkness.

EXPERT Q: "Multiplicity control?" — Bonferroni across tested periods; 0% false positives on rhythm-free data; Lomb–Scargle cross-check.

NON-SCIENTIST Q: "Proof it works?" — it recovers the classic mutant clock periods 24/19/28 h and agrees with two other clock tools.

## Slide 10. Module: Larval crawling

IN PLAIN TERMS: The larval movement test. From a video we track a single larva and measure how fast and how straight it crawls, in real units (mm per second) once you tell it the pixels-per-millimetre scale.

DEFINE: tortuosity = how winding the path is (1 = straight line).

EXPERT Q: "Detection method?" — largest dark object per frame by Otsu thresholding; no training.

NON-SCIENTIST Q: "How accurate?" — on a test video with a known speed it recovers 4.472 mm/s exactly; wild-type larvae are ≈1 mm/s.

## Slide 11. Module: Grooming

IN PLAIN TERMS: The grooming test. Flies clean themselves; we quantify what fraction of the observation time they spend grooming and the number and length of grooming bouts, from scored start/stop times.

EXPERT Q: "Does it detect grooming from raw video?" — no; automated behaviour classification is a separate ML problem; we analyse the scored intervals, which is the standard output.

NON-SCIENTIST Q: "Reference value?" — healthy flies groom ≈13% of the time (Qiao 2018), which our demo reproduces.

## Slide 12. Engine: Motor Decline Rate

IN PLAIN TERMS: Motor Decline Rate measures how fast climbing worsens with AGE — a progression read-out. We fit an exponential-decay curve to Performance Index versus age and read off the decline rate k; a drug that slows decline lowers k.

KEY POINT: decline is over days of age, NOT the seconds within a single climbing test (where flies are rising) — conflating the two is a common error we deliberately avoid.

EXPERT Q: "What if it doesn't fit?" — a linear fallback and an R² are reported.

NON-SCIENTIST Q: "Why is this better than a single score?" — it captures disease PROGRESSION and slowing, not just a snapshot.

## Slide 13. Engine: Trajectory clustering

IN PLAIN TERMS: Trajectory clustering automatically groups response curves that behave similarly — for example separating 'keeps declining' from 'stays healthy' — without being told the labels in advance.

DEFINE: DTW (dynamic time warping) = a way to measure how similar two time-curves are even if shifted.

EXPERT Q: "How is k chosen?" — silhouette-optimal over Ward-linkage clustering.

NON-SCIENTIST Q: "What is it for?" — spotting distinct response patterns in a screen at a glance.

## Slide 14. Innovation: Drug Ranking

IN PLAIN TERMS — THE INNOVATION: How do you fairly combine a drug's effect across several fly strains of DIFFERENT sizes into one score? We borrow medical meta-analysis. Each strain's effect (Hedges' g) is weighted by its reliability (inverse of its variance): bigger, cleaner strains count more; small noisy ones count less but aren't discarded. These combine across strains and then across all assays into one Therapeutic Score.

DEFINE: Hedges' g = standardised effect size; inverse-variance / DerSimonian–Laird = the optimal weighting recipe; I² = how much strains genuinely differ.

PROOF: in 3,000 simulations it was 19.9% more accurate than simple averaging with correctly-calibrated confidence intervals (95.8%), and the maths matches statsmodels to six decimals.

EXPERT Q: "Why random-effects?" — to account for real strain-to-strain heterogeneity.

NON-SCIENTIST Q: "Business value?" — fewer wrong drug bets carried forward — real money saved.

## Slide 15. Validation strategy (3 pillars)

IN PLAIN TERMS: We tested the tool three independent ways. (1) Published benchmark data — reproduce a paper's exact answer. (2) Analytic ground truth — feed in data whose correct answer we built in, and check it comes back out. (3) Reference datasets — ≥3 per assay tied to cited literature. A fourth check compares against seven trusted tools.

EXPERT Q: "Which pillar is strongest?" — analytic ground truth for exactness, published data for real-world validity; together they cover both.

NON-SCIENTIST Q: "In plain words?" — we checked it against textbook answers, against puzzles with known solutions, and against other trusted programs — and it passed all of them.

## Slide 16. Test data / testing samples

IN PLAIN TERMS: Where the test data came from, and honesty about it. Real raw data (a famous 1963 survival trial; a classic stats dataset) is used directly. Real published parameters (the classic clock periods) drive some signals. Some data is generated with a known answer to test exactness. The rest are example datasets calibrated to cited published numbers and clearly labelled 'simulated'.

KEY POINT: we never present a simulation as if it were real measured data.

EXPERT Q: "Why not pull every raw supplement?" — many are binary files on repositories not machine-readable in our environment; rather than fabricate, we cite parameters and label them.

NON-SCIENTIST Q: "How do I know it's not cherry-picked?" — every dataset lists its source and 18/18 reproduce the published expectation.

## Slide 17. Testing parameters

IN PLAIN TERMS: A common worry is that developers tweak settings until the tool looks good. We did the opposite — every key setting comes straight from the literature, with a citation. A fly counts as asleep after 5 minutes still; chance memory is 50%; the reference clock periods are 24/19/28 h.

EXPERT Q: "Periodogram multiplicity?" — Bonferroni-corrected; verified 0% false positives on rhythm-free data.

NON-SCIENTIST Q: "One-sentence reassurance?" — we didn't invent the settings; they are the community's own published standards.

## Slide 18. Results: validation

IN PLAIN TERMS: The evidence the tool is correct. Twelve automated test suites all pass. On famous datasets it returns the exact published numbers — the 1963 trial's log-rank (16.79) and hazard ratio (4.52). The six-panel figure shows one real result from each area.

EXPERT Q: "Live or hard-coded?" — live; rerun run_all.py to reproduce.

NON-SCIENTIST Q: "What does 12/12 mean?" — twelve independent accuracy inspections, all passed.

## Slide 19. Results: tool comparison

IN PLAIN TERMS: Beyond matching papers, we ran the SAME data through seven independent, world-standard programs (including R, used across academia and pharma) and compared every number: perfect agreement (r = 1.0000) — every point sits on the diagonal.

EXPERT Q: "The one Cox difference (4.52 vs 4.82)?" — the known Breslow-vs-Efron tie method, not an error; our Breslow matches R's Breslow to five figures; reported transparently.

NON-SCIENTIST Q: "Why compare if yours is new?" — to prove we didn't reinvent the wheel wrongly; matching trusted tools is the strongest reassurance.

## Slide 20. Results: reference datasets

IN PLAIN TERMS: We also built three literature-referenced datasets for EACH assay (18 total) in the exact upload format, and ran them through the tool: all 18 reproduce their cited published expectation — wild-type outperforms disease models and drugs partially rescue, exactly as the literature reports.

EXPERT Q: "Are these downloadable?" — yes, in each module's template, with provenance and results files.

NON-SCIENTIST Q: "Why does this matter?" — it shows the tool gives sensible, literature-consistent answers on realistic data, not just toy examples.

## Slide 21. Output formats

IN PLAIN TERMS: What you actually get out. On-screen sortable tables; one-click CSV downloads; publication-quality 300-dpi figures; a full statistics report (test, p-value, effect size, confidence interval); the drug ranking; and a single ZIP bundle ready for a journal submission.

EXPERT Q: "Figure formats?" — PNG/SVG/PDF at 300 dpi, colour-blind-safe palette.

NON-SCIENTIST Q: "Can I put these straight in a paper or report?" — yes, that is the design goal.

## Slide 22. Why it is reliable

IN PLAIN TERMS: Five concrete reasons to trust it: it reproduces published results; it agrees with gold-standard software; it uses the correct method by design; anyone can rerun and verify every number; and we are explicit about every limitation.

EXPERT Q: "Reproducibility mechanism?" — fixed random seeds plus the shipped validation/comparison suite.

NON-SCIENTIST Q: "What if it's ever wrong?" — because it's transparent and benchmarked, any discrepancy is visible and traceable — unlike a black box.

## Slide 23. Applications

IN PLAIN TERMS: Where it helps. The biggest use is brain-disease drug screening — judging a compound fairly across several fly disease models and several tests at once. It also serves ageing/clock research, teaching rigorous statistics, and open reproducible science.

COMMERCIAL ANGLE: a fairer, more accurate candidate ranking means fewer false leads advanced — lower downstream cost; and it's free and browser-based, so adoption cost is near zero.

EXPERT Q: "Can it take my lab's data?" — yes; documented templates match common bench outputs.

NON-SCIENTIST Q: "Path to impact/revenue?" — adoption by fly labs and pre-clinical CROs; a validated core that can underpin paid hosted or extended versions.

## Slide 24. Closing

IN PLAIN TERMS: The takeaway — six scattered, manually-analysed fly experiments turned into one automated, rigorously-validated web tool, plus a genuinely new, more efficient way to rank drugs across unequal-sized strains. Free, open, and prepared for a leading journal.

CLOSE ON: reliable (24/24 tool agreement, reproduces published data) + novel (19.9% more efficient ranking) + accessible (free web app).

FINAL Qs: Expert — "what's next?" more assays, hosted deployment. Non-scientist — "how do I try it?" open drophenix.streamlit.app; no install, no login.
