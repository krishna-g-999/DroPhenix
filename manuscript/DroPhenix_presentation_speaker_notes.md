# DroPhenix presentation — speaker notes

Dual-audience notes (expert + non-scientific / commercial) for each slide of DroPhenix_presentation.pptx. All content is grounded in the project's verified results.

## Slide 1. Title

IN PLAIN TERMS: DroPhenix is a free website that automatically analyses experiments done on fruit flies (Drosophila melanogaster). Fruit flies are the workhorse "model organism" for brain-disease research — they share roughly three-quarters of human disease genes, are cheap, and live only weeks, so a whole disease-and-drug experiment finishes fast. We turned six separate, commonly-used fly tests into one reliable, automated analysis tool, and added a brand-new statistical method for comparing drugs fairly.

TALKING POINTS: Built at SSSIHL. Free, no login. Headline proof up front: it agrees with the world's gold-standard analysis software on 24 out of 24 checks (perfect correlation), and every one of its results was checked against published scientific data.

EXPERT MIGHT ASK: "Why Drosophila rather than mice?" — Faster, far cheaper, genetically tractable, and the standard first-pass model for ALS, Parkinson's and Huntington's. "Is this novel or just a wrapper?" — Both: it standardises best-practice statistics AND introduces a genuinely new cross-strain ranking method (slide 4).

NON-SCIENTIST / INVESTOR MIGHT ASK: "What does it actually do for a user?" — You upload your experiment's raw numbers; it produces the correct graphs and statistics automatically and tells you which drug candidate performs best. "Who is it for?" — Academic fly labs and pre-clinical drug-screening groups.

## Slide 2. The problem

IN PLAIN TERMS: An "assay" is just a standard laboratory test. Today, labs analyse each of these six tests separately, often by hand in Excel, using inconsistent statistics. Worse, when different genetic strains of fly are tested with different numbers of animals, there is no accepted, fair way to combine the results.

DEFINE: "Under-powered" = too little data / wrong method, so real effects are missed or false ones appear. "Kaplan-Meier / log-rank / Cox" = the standard, rigorous ways to analyse survival (lifespan) data used in clinical trials.

EXPERT MIGHT ASK: "Isn't survival analysis already standard?" — In clinical statistics yes, but fly-lifespan papers frequently report only the median and skip the survival curve, the log-rank test and the hazard model; we bring those to the bench by default.

NON-SCIENTIST MIGHT ASK: "Why is unequal numbers such a big deal?" — Analogy: if you average a survey of 3 people and a survey of 700 people with equal weight, the tiny, noisy survey distorts the answer. The same happens when you compare drug effects across fly strains of very different size — and that is exactly the everyday screening situation.

## Slide 3. What we developed

IN PLAIN TERMS: This is the full menu. Each card is one analysis module. Six are biological tests: climbing (do the flies climb up — a motor/health readout), lifespan (how long they live), Y-maze (a simple memory/behaviour test), DAMS (sleep and body-clock), larval crawling (baby-fly movement), and grooming (self-cleaning behaviour). The other three are analytical: Motor Decline Rate (how fast climbing worsens with age), trajectory clustering (grouping similar response patterns automatically), and Drug Ranking (our headline innovation).

DEFINE: "Negative geotaxis" = flies instinctively climb upward after being tapped down; how well they climb measures neuromuscular health.

EXPERT MIGHT ASK: "Do the video modules replace dedicated trackers?" — For counting and single-larva speed, yes; full multi-animal pose tracking is out of scope and stated as such.

NON-SCIENTIST MIGHT ASK: "Why so many tests?" — Different diseases hit different systems (movement, memory, sleep, lifespan). Testing across all of them gives a fuller, more trustworthy picture of whether a drug truly helps.

## Slide 4. Core innovation (unequal-N ranking)

IN PLAIN TERMS — THE KEY SLIDE: This is what makes DroPhenix new. When you test a drug in several fly disease-strains that have different numbers of animals, how do you fairly combine the results into one score? We borrow a method from medical "meta-analysis" (the science of combining many studies). Each strain's result is weighted by how reliable it is: bigger, less-noisy strains count more; small noisy ones count less but are not thrown away.

DEFINE: "Hedges' g / effect size" = a standardised measure of how big a drug's effect is, comparable across experiments. "Inverse-variance / DerSimonian-Laird pooling" = the mathematically optimal recipe for weighting and combining those results. The forest plot (right) shows each strain's effect and the combined answer (green diamond).

PROOF: In 3,000 simulated experiments where we knew the true answer, our method was 19.9% more accurate than simple averaging and its confidence intervals were correctly calibrated (95.8% vs a 95% target).

EXPERT MIGHT ASK: "Why DerSimonian-Laird over a fixed effect?" — It accounts for genuine strain-to-strain differences (heterogeneity, reported as I-squared). "Is the pooling implemented correctly?" — It matches statsmodels to six decimals.

NON-SCIENTIST MIGHT ASK: "So what's the business value?" — A more accurate, defensible ranking of drug candidates means fewer wrong bets carried forward — real money saved in a screening pipeline.

## Slide 5. Modules & calculations

IN PLAIN TERMS: For transparency, here is the exact formula behind each module and, crucially, WHY we use it. There is no hidden "black box"; every number can be traced to a published, standard equation.

DEFINE a few: "PI (Performance Index)" scales climbing from -100 (none climb) to +100 (all climb). "Hazard ratio" = how many times higher the risk of death is in one group vs another. "Periodogram" = a method that finds the body-clock period (near 24 h) hidden in activity data.

EXPERT MIGHT ASK: "Why fit motor decline over age, not over the assay seconds?" — Because within one climbing test the flies are rising (PI increases over seconds); decline is a property of ageing across days. Fitting decay to the seconds axis is a category error we deliberately avoid.

NON-SCIENTIST MIGHT ASK: "Why show formulas at all?" — Trust. Reviewers and regulators want to see that we use the accepted, textbook methods — not something invented and unverifiable.

## Slide 6. Statistics engine

IN PLAIN TERMS: Good statistics is about choosing the right test for your data, not always using the same one. DroPhenix first checks the shape of your data (is it bell-curved? do the groups vary similarly?) and then automatically picks the correct test. It always reports not just "is it significant" but "how big is the effect", and it corrects for the fact that running many comparisons produces false positives by chance.

DEFINE: "Effect size" = the practical size of a difference. "False-discovery-rate (Benjamini-Hochberg) correction" = a safeguard so that testing many things at once doesn't throw up flukes. "Bootstrap confidence interval" = a range showing how sure we are, obtained by resampling the data thousands of times.

EXPERT MIGHT ASK: "What are the gating thresholds?" — Shapiro-Wilk and Levene at 0.05, with a minimum n of 5 before a parametric test is allowed; otherwise non-parametric.

NON-SCIENTIST MIGHT ASK: "Why does this matter commercially?" — It is the difference between a result that survives peer review / due-diligence and one that quietly falls apart.

## Slide 7. Tools & techniques

IN PLAIN TERMS: The software is built entirely from well-established, open-source scientific tools — nothing exotic or proprietary. Streamlit gives the browser interface; NumPy/pandas/SciPy do the maths; statsmodels provides the peer-reviewed survival and meta-analysis engine; OpenCV reads the videos; Matplotlib/Plotly draw the figures.

IMPORTANT DISTINCTION: DroPhenix is NOT an "AI model" that was "trained" on data with hidden weights. It is deterministic — the fly detection uses a classic image-brightness threshold (Otsu), the clustering is unsupervised, and every statistic is a closed-form formula. Same input always gives the same output, and every step is inspectable.

EXPERT MIGHT ASK: "Why not scikit-survival?" — It would not build in our environment (needs a C compiler / no Python-3.13 wheel); we instead validate survival against R's survival package and lifelines, covering both tie-handling methods.

NON-SCIENTIST MIGHT ASK: "Is it a machine-learning / AI product?" — No, and that is a strength here: results are exact, reproducible and explainable, not a probabilistic guess.

## Slide 8. Data sources

IN PLAIN TERMS: Any tool is only as trustworthy as the data used to test it. We were scrupulously honest about our data. Where real published raw data existed and was retrievable, we used it directly (the famous 1963 leukemia survival dataset; a classic statistics dataset). Where only summary numbers were published, we generated example datasets calibrated to those published numbers and clearly LABELLED them as simulated — we never present a simulation as if it were real data.

DEFINE: "Free-running period" = the fly's internal body-clock length when kept in constant darkness; wild-type is ~24 h.

EXPERT MIGHT ASK: "Why not pull raw supplements for every assay?" — Many are deposited as binary Excel/zip on repositories that were not machine-readable in our build environment; rather than fabricate, we used cited parameters and labelled them.

NON-SCIENTIST MIGHT ASK: "How do I know the numbers aren't cherry-picked?" — Every dataset lists its source; 18 of 18 reproduce the published expectation; and the code to regenerate them ships with the tool.

## Slide 9. Testing parameters

IN PLAIN TERMS: A common worry is that developers "tune" settings until the tool gives a flattering answer. We did the opposite: every key setting comes straight from the published literature, with the citation. For example, a fly is scored as "asleep" after 5 minutes of no movement — that is the field's accepted definition, not our choice.

WHY EACH: the 5-minute sleep rule, the 50% chance level for the memory test, the 24/19/28-hour reference clock periods, and the Breslow method for the survival model are all standard, published values. Using them means our results are directly comparable to the literature.

EXPERT MIGHT ASK: "Is the periodogram multiplicity-corrected?" — Yes; without a Bonferroni correction across tested periods the false-positive rate would balloon; we verified 0% false positives on rhythm-free data.

NON-SCIENTIST MIGHT ASK: "In one sentence, why trust the settings?" — Because we did not invent them — they are the community's own standards, cited on this slide.

## Slide 10. Results: validation

IN PLAIN TERMS: This is the evidence that the tool is correct. We ran 12 automated test suites; all 12 pass. On famous published datasets, DroPhenix returns the exact published numbers — for example the 1963 leukemia trial's log-rank statistic (16.79) and hazard ratio (4.52). The six-panel figure shows one real result per module.

DEFINE: "Log-rank / hazard ratio" = standard survival statistics; matching the published value to the decimal means our maths is right.

EXPERT MIGHT ASK: "Are these live or hard-coded?" — Live: generated by the shipped scripts; anyone can rerun run_all.py and reproduce them.

NON-SCIENTIST MIGHT ASK: "What does '12/12 pass' mean?" — Think of it as 12 independent MOT/inspection tests of the tool's accuracy; it passed every one.

## Slide 11. Results: tool comparison

IN PLAIN TERMS: Beyond matching published papers, we ran the SAME data through seven independent, world-standard analysis programs (including R, the statistics language used across academia and pharma) and compared every number. Across 24 comparisons the agreement is perfect (correlation r = 1.0000) — the scatter plot shows every point sitting exactly on the diagonal "perfect-agreement" line.

DEFINE: "R survival / lifelines / statsmodels / astropy / pyBOAT" = the reference tools domain experts already trust; if we match them, we inherit that trust.

EXPERT MIGHT ASK: "You noted one Cox value differs (4.52 vs 4.82)?" — That is the known Breslow-vs-Efron tie-handling difference, not an error; our Breslow value matches R's Breslow to five figures and the published value. We report it transparently as a method note.

NON-SCIENTIST MIGHT ASK: "Why compare to other tools if yours is new?" — To prove we did not reinvent the wheel incorrectly; matching the trusted tools is the strongest possible reassurance.

## Slide 12. Why it is reliable

IN PLAIN TERMS: Pulling it together — five concrete reasons the tool can be trusted. It reproduces published results, it agrees with gold-standard software, it uses the correct method by design, anyone can independently rerun and verify every number with a single command, and we are explicit about every limitation rather than hiding them.

TALKING POINT: "Correct by construction" means correctness is built into the method choice, not patched afterwards.

EXPERT MIGHT ASK: "How is reproducibility guaranteed?" — Fixed random seeds and the full validation + comparison code ship with the software; results are deterministic.

NON-SCIENTIST MIGHT ASK: "What if it gives a wrong answer?" — Because it is transparent and benchmarked, any discrepancy is visible and traceable — unlike a black-box tool where you simply have to hope.

## Slide 13. Applications

IN PLAIN TERMS: Where this helps. The biggest use is drug screening for brain diseases: quickly and fairly judging whether a compound helps across several fly disease models and several types of test. Other uses include ageing and body-clock research, teaching rigorous statistics, and open, reproducible science.

COMMERCIAL ANGLE: In a screening pipeline, a fairer, more accurate ranking of candidates means fewer false leads advanced — directly reducing wasted downstream cost. It is free and browser-based, so adoption cost is essentially zero.

EXPERT MIGHT ASK: "Can it ingest my lab's existing data?" — Yes; each module has a documented upload template, and the formats match common bench outputs.

NON-SCIENTIST MIGHT ASK: "What is the path to impact / revenue?" — Adoption by fly labs and pre-clinical CROs; the transparent, validated core is also a foundation for paid, hosted or extended commercial versions.

## Slide 14. Closing

IN PLAIN TERMS: The takeaway. We turned six scattered, manually-analysed fly experiments into a single, automated, rigorously-validated web tool — and added a genuinely new, more efficient way to rank drugs across unequal-sized strains. It is free, open, and prepared for publication in a leading journal (Nucleic Acids Research).

CLOSE ON: reliability (24/24 tool agreement, reproduces published data) + novelty (19.9% more efficient ranking) + accessibility (free web app).

LIKELY FINAL QUESTIONS: Expert — "What's next?" More assays and community-contributed modules; hosted deployment. Non-scientist — "How do I try it?" Just open drophenix.streamlit.app in any browser; no install, no login.
