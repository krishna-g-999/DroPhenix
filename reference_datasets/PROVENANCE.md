# Reference-dataset provenance

REAL = verified raw data or exact published parameters. simulated = generated from text-verified published parameters (authorised fallback), cited below.

| Module | Dataset | Type | Citation | Calibration parameters |
|---|---|---|---|---|
| Climbing | sample1_PD_aSyn | simulated | Feany & Bender 2000 Nature 404:394 (a-syn PD model: progressive climbing loss) | endpoint pass fraction WT=0.88, disease=0.4, drug=0.66 |
| Climbing | sample2_ALS_TDP43 | simulated | Gargano et al. 2005 Exp Gerontol 40:386 (RING; locomotor decline) | endpoint pass fraction WT=0.88, disease=0.3, drug=0.58 |
| Climbing | sample3_HD_Q93 | simulated | Drosophila HD polyQ models: climbing impairment rescued by treatment | endpoint pass fraction WT=0.88, disease=0.35, drug=0.55 |
| Survival | sample1_Freireich1963 | REAL raw data | Freireich EJ et al. 1963 Blood 21:699-716 | verified raw event-level data |
| Survival | sample2_fly_lifespan | simulated | Linford NJ et al. 2013 J Vis Exp e50068 | Gompertz; target medians WT=50, disease=30, drug=42 d |
| Survival | sample3_fly_lifespan | simulated | Linford NJ et al. 2013 J Vis Exp e50068 | Gompertz; target medians WT=52, disease=34, drug=45 d |
| Y-maze | sample1_amnesiac | simulated | Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700 | chance 50%; turn bias Beta(4.6,4.6); WT rho=0.34, disease rho=0.02 |
| Y-maze | sample2_rut2080 | simulated | Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700 | chance 50%; turn bias Beta(4.6,4.6); WT rho=0.36, disease rho=0.05 |
| Y-maze | sample3_dnc1 | simulated | Lewis SA et al. 2017 NLM 142:230; Buchanan/de Bivort 2015 PNAS 112:6700 | chance 50%; turn bias Beta(4.6,4.6); WT rho=0.3, disease rho=0.0 |
| DAMS | sample1_period_mutants | REAL published periods | Konopka RJ & Benzer S 1971 PNAS 68:2112-2116 | activity simulated with the published free-running periods 24/19/28 h |
| DAMS | sample2_sleep | simulated | Hendricks JC 2000 Neuron 25:129; Shaw PJ 2000 Science 287:1834 | 5-min inactivity sleep rule |
| DAMS | sample3_sleep | simulated | Hendricks JC 2000 Neuron 25:129; Shaw PJ 2000 Science 287:1834 | 5-min inactivity sleep rule |
| Larval | sample1_ALS_TDP43 | simulated | Gargano et al. 2005 (locomotion); larval crawling literature | WT mean speed ~1.0 mm/s |
| Larval | sample2_PD_Pink1 | simulated | Gargano et al. 2005 (locomotion); larval crawling literature | WT mean speed ~1.0 mm/s |
| Larval | sample3_HD_Q93 | simulated | Gargano et al. 2005 (locomotion); larval crawling literature | WT mean speed ~1.0 mm/s |
| Grooming | sample1_dCREB2 | simulated | Qiao B et al. 2018 eLife 7:e34497 | WT ~13% time grooming |
| Grooming | sample2_FoxP | simulated | Qiao B et al. 2018 eLife 7:e34497 | WT ~13% time grooming |
| Grooming | sample3_Cul3 | simulated | Qiao B et al. 2018 eLife 7:e34497 | WT ~13% time grooming |
