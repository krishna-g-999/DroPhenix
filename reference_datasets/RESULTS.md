# Reference-dataset results

Each dataset (template format, in its module folder) run through the real DroPhenix module; result vs the published expectation.

| Module | Dataset | Type | Source | Key result | Expected | Status |
|---|---|---|---|---|---|---|
| Climbing | sample1_PD_aSyn | simulated | Feany & Bender 2000 Nature 404:394 | WT PI=73, PD_aSyn-veh=-21, PD_aSyn-drug=26 | WT>disease; drug rescues | PASS |
| Climbing | sample2_ALS_TDP43 | simulated | Gargano et al. 2005 Exp Gerontol 40:386 | WT PI=71, ALS_TDP43-veh=-41, ALS_TDP43-drug=3 | WT>disease; drug rescues | PASS |
| Climbing | sample3_HD_Q93 | simulated | Drosophila HD polyQ models: climbing impairment rescued by treatment | WT PI=88, HD_Q93-veh=-21, HD_Q93-drug=12 | WT>disease; drug rescues | PASS |
| Survival | sample1_Freireich1963 | REAL raw data | Freireich et al. 1963 Blood 21:699 | median 23/8 wk; log-rank χ²=16.79 | χ²≈16.79; median 23/8 (published) | PASS |
| Survival | sample2_fly_lifespan | simulated | Linford et al. 2013 (WT median ~50 d) | WT median=53d, disease=30d, drug=40d | WT>disease; drug extends | PASS |
| Survival | sample3_fly_lifespan | simulated | Linford et al. 2013 (WT median ~50 d) | WT median=54d, disease=33d, drug=45d | WT>disease; drug extends | PASS |
| Y-maze | sample1_amnesiac | simulated | Lewis 2017 (50% chance); de Bivort 2015 (bias 0.5) | WT alt=70%, amnesiac=57%; bias=0.51; extreme=28% | WT>disease; bias≈0.50; ~23.5% extreme | PASS |
| Y-maze | sample2_rut2080 | simulated | Lewis 2017 (50% chance); de Bivort 2015 (bias 0.5) | WT alt=72%, rut2080=58%; bias=0.50; extreme=28% | WT>disease; bias≈0.50; ~23.5% extreme | PASS |
| Y-maze | sample3_dnc1 | simulated | Lewis 2017 (50% chance); de Bivort 2015 (bias 0.5) | WT alt=70%, dnc1=55%; bias=0.51; extreme=19% | WT>disease; bias≈0.50; ~23.5% extreme | PASS |
| DAMS | sample1_period_mutants | REAL published periods | Konopka & Benzer 1971 PNAS 68:2112 | WT=24.0, perS=19.0, perL=28.0 h | 24 / 19 / 28 h (published) | PASS |
| DAMS | sample2_sleep | simulated | Hendricks 2000; Shaw 2000 (5-min rule; M>F sleep) | male sleep=592, female=427 min/day | >=5-min rule; males>females | PASS |
| DAMS | sample3_sleep | simulated | Hendricks 2000; Shaw 2000 (5-min rule; M>F sleep) | male sleep=593, female=418 min/day | >=5-min rule; males>females | PASS |
| Larval | sample1_ALS_TDP43 | simulated | WT third-instar ~1 mm/s (published range 0.5-2) | WT=1.05, ALS_TDP43=0.51 mm/s | WT≈1 mm/s; WT>disease | PASS |
| Larval | sample2_PD_Pink1 | simulated | WT third-instar ~1 mm/s (published range 0.5-2) | WT=1.02, PD_Pink1=0.59 mm/s | WT≈1 mm/s; WT>disease | PASS |
| Larval | sample3_HD_Q93 | simulated | WT third-instar ~1 mm/s (published range 0.5-2) | WT=0.97, HD_Q93=0.57 mm/s | WT≈1 mm/s; WT>disease | PASS |
| Grooming | sample1_dCREB2 | simulated | Qiao et al. 2018 eLife (WT ~13% time) | WT=12%, dCREB2=7% | WT≈13%; WT>disease | PASS |
| Grooming | sample2_FoxP | simulated | Qiao et al. 2018 eLife (WT ~13% time) | WT=13%, FoxP=7% | WT≈13%; WT>disease | PASS |
| Grooming | sample3_Cul3 | simulated | Qiao et al. 2018 eLife (WT ~13% time) | WT=12%, Cul3=6% | WT≈13%; WT>disease | PASS |

**18/18 datasets reproduce the published expectation.**
