# DroPhenix Analytics v2.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drophenix.streamlit.app)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

> **A web-based analytics platform for quantitative Drosophila melanogaster motor phenotyping in neurodegeneration research**

Developed at the **Department of Bioinformatics, SSSIHL, Puttaparthi, India**  
PI: Prof. Venketesh Sivaramakrishnan · Scholar: Krishnasalini Gunanathan

---

## Live Server

🔗 **https://drophenix.streamlit.app**  
No installation required · No login · Works in any browser

---

## Statement of Need

Drosophila climbing (negative geotaxis) assays are the primary motor readout in fly models of ALS, Parkinson's, and Huntington's disease. Existing analysis workflows rely on endpoint Performance Index (PI) calculated manually in spreadsheets — a method that:

- Discards longitudinal trajectory information
- Cannot quantify sex-differential drug responses
- Provides no measure of WT-selective therapeutic safety
- Offers no standardised kinetic disease-progression metric

DroPhenix addresses all four gaps through a suite of purpose-built composite metrics.

---

## Novel Metrics

| Metric | Formula | Biological meaning |
|--------|---------|-------------------|
| **CRI** — Composite Rescue Index | `CRI = 0.4·f(PI) + 0.4·f(AUC) + 0.2·f(t50)` × 100 | Integrates endpoint, kinetic, and latency rescue into one score (0–100%) |
| **SDQ** — Sex Dimorphism Quotient | `SDQ = \|PI_M − PI_F\| / max(\|PI_M\|, \|PI_F\|)` | Quantifies sex-differential drug response; critical for translational validity |
| **TSW** — Therapeutic Selectivity Window | `TSW = ΔPI_disease / \|ΔPI_WT\|` | Measures disease-selective rescue; TSW > 1.0 = safe window |
| **MDR** — Motor Decline Rate | `PI(t) = PI₀ · e^{−kt}` | Exponential decay constant k; Δk = treatment-mediated rescue of progression rate |

---

## Features

- **Data Input** — Upload CSV/Excel or load built-in SSSIHL ALS demo dataset
- **Trajectory Analysis** — Longitudinal climbing curves with SEM ribbons
- **Statistical Engine** — ANOVA / Kruskal-Wallis, Tukey HSD, BH-FDR correction, bootstrap CI
- **Novel Metrics** — CRI, SDQ, TSW, MDR with 300 DPI export
- **Clustering** — DTW-based trajectory clustering, dendrogram, PCA/UMAP
- **Translational Scoring** — Tier I–III compound ranking (TCS)
- **Live Recording** — Webcam / MP4 upload with automated fly counting (OpenCV)
- **Export** — CSV, PNG, SVG, PDF, ZIP bundle

---

## Installation (local)

```bash
# Clone repository
git clone https://github.com/krishna-g-999/drophenix.git
cd drophenix

# Create environment (conda recommended)
conda create -n drophenix python=3.11
conda activate drophenix

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

---

## Repository Structure

```
drophenix/
├── app.py                  # Main Streamlit application
├── modules/
│   ├── data_parser.py      # Input validation and parsing
│   ├── normalizer.py       # PI, AUC, t50 computation
│   ├── metrics.py          # CRI, SDQ, TSW, ECS
│   ├── motor_decline.py    # MDR exponential decay, MSCI
│   ├── stats_engine.py     # Statistical tests, bootstrap CI
│   ├── trajectory_cluster.py # DTW clustering, PCA/UMAP
│   ├── translational_score.py # TCS compound ranking
│   ├── visualizer.py       # Plotly figure factory
│   ├── recording.py        # Webcam / video capture
│   └── video_analyzer.py   # OpenCV fly detection
├── protocols/
│   └── protocol_guide.py   # Vial setup & data template guide
├── assets/
│   └── Drophenix_Logo.png
├── requirements.txt
├── packages.txt
└── validate_drophenix_v2.py  # Independent validation suite
```

---

## Validation

An independent algorithmic validation suite (`validate_drophenix_v2.py`) tests all metric formulas against synthetic data with known analytical ground truth:

| Test | Result |
|------|--------|
| MDR k accuracy (Pearson r vs ground truth) | r = 0.987, p < 10⁻¹⁸ |
| MDR fit quality (median R²) | ≥ 0.90 |
| PI monotone decline (Spearman) | ≥ 80% disease groups |
| CRI ordinal ranking (Drug_A > Drug_B) | 4/4 groups correct |
| SDQ accuracy vs endpoint GT | Mean error < 30% |
| TSW selectivity ranking | Correct |
| PI replicate reproducibility (Bland–Altman) | CV < 15% |

---

## Citation

If you use DroPhenix in your research, please cite:

> Gunanathan K, Sivaramakrishnan V (2026).  
> **DroPhenix: A web-based analytics platform for quantitative Drosophila motor phenotyping.**  
> *Nucleic Acids Research*, Web Server Issue.  
> DOI: [pending]

---

## License

DroPhenix is released under the **GNU General Public License v3.0**.  
See [LICENSE](LICENSE) for details.

---

## Contact

- **Krishnasalini Gunanathan** — PhD Scholar, Dept. of Bioinformatics, SSSIHL  
- **Prof. Venketesh Sivaramakrishnan** — Principal Investigator  
- Sri Sathya Sai Institute of Higher Learning, Puttaparthi 515134, Andhra Pradesh, India