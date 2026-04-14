# 🔍 OpenXAI: NeurIPS 2022 Replication Project

[![Explainer Methods](https://img.shields.io/badge/Explainers-7_Methods-blueviolet)](https://github.com/AI4LIFE-GROUP/OpenXAI)
[![Metrics](https://img.shields.io/badge/Metrics-5_Quantitative-success)](#-metrics-evaluated)
[![Datasets](https://img.shields.io/badge/Datasets-Adult_&_COMPAS-orange)](#-datasets--models)

This repository contains a comprehensive replication of the core benchmarking experiments presented in the NeurIPS 2022 paper:  
> **"OpenXAI: Towards a Transparent Evaluation of Model Explanations"**  
> *Agarwal et al., NeurIPS 2022 Datasets & Benchmarks Track.*

Our project implements the OpenXAI pipeline to evaluate post-hoc explainability methods across multiple dimensions of faithfulness and stability.

---

## 👥 Team Members

| Name | ID | Phase 1 & 2 Role | Phase 3 Role |
| :--- | :--- | :--- | :--- |
| **P B Shreyas** | 23BDS041 | Visualizations, Results analysis | Noise Setup & Degradation Study |
| **Aditya Sahrawat** | 23BCS006 | Documentation, Experiment validation | SmoothSHAP / SmoothLIME & Before-After Analysis |
| **Alok Kumar** | 23BCS003 | Pipeline architecture, Metric integration | Visualization, Integration & Final Report |

---

## 🎯 Project Objectives

The primary goal of this replication was to validate the findings of the OpenXAI benchmark regarding the trade-offs between different explanation methods. Specifically, we focused on:
- **Reproducing Tables 1–5** of the original paper.
- Evaluating **7 post-hoc explainability methods** on pretrained Artificial Neural Networks (ANN) and Logistic Regression (LR) models.
- Measuring **5 quantitative metrics** covering Faithfulness and Stability.
- Generating standardized heatmaps and visual comparisons.

---

## 🛠️ Technical Stack

### 🧠 Explainers Evaluated
We evaluate 7 diverse explainability methods:
1. **LIME**: Local Interpretable Model-agnostic Explanations.
2. **SHAP**: SHapley Additive exPlanations.
3. **Vanilla Gradients**: Simple gradient-based attribution.
4. **Gradient × Input**: Gradient weighted by input values.
5. **SmoothGrad**: Averaged gradients over noisy perturbations.
6. **Integrated Gradients**: Path-integral based gradients.
7. **Random**: Baseline control (random attributions).

### 📈 Metrics Evaluated
| Category | Metric | Description | Goal |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **PGF (PGI)** | Prediction Gap on Frequent (Important) features | ↑ Maximize |
| **Faithfulness** | **PGU** | Prediction Gap on Unimportant features | ↓ Minimize |
| **Stability** | **RIS** | Relative Input Stability | ↓ Minimize |
| **Stability** | **RRS** | Relative Representation Stability | ↓ Minimize |
| **Stability** | **ROS** | Relative Output Stability | ↓ Minimize |

### 📊 Datasets & Models
- **Phase 1 & 2 Datasets**: Adult Income (Census), COMPAS (Recidivism).
- **Phase 3 Datasets**: Adult, COMPAS, German Credit, HELOC (4 datasets).
- **Models**: Pretrained 2-layer ANN and Logistic Regression (LR).

---

## 📂 Repository Structure

```text
XAI_project/
├── run_experiment.py        # 🚀 Phase 1 & 2: Main benchmark entry point
├── run_phase3.py            # 🔬 Phase 3: Noise-robustness study entry point
├── src/
│   ├── config.py            # Configuration, seeds, and constants
│   ├── train_model.py       # Data loading and model management
│   ├── run_explainers.py    # Generation of attribution maps
│   ├── compute_metrics.py   # Faithfulness and Stability calculations
│   ├── visualize_results.py # Figure and Table generation
│   ├── noise_utils.py       # 🆕 Gaussian noise injection across sigma levels
│   ├── phase3_metrics.py    # 🆕 Degradation study & before-after comparison
│   └── smooth_explainers.py # 🆕 SmoothSHAP and SmoothLIME implementations
├── data/                    # Dataset storage (auto-downloaded)
├── models/                  # Pretrained model weights
├── results/                 # 📈 Output directory for CSVs and PNGs
│   ├── tables/              # Raw metric CSVs (including Phase 3 per-sigma files)
│   └── *.png                # Heatmaps, bar charts, degradation curves
├── report/                  # Detailed replication summaries
│   ├── phase2_summary.md    # Phase 1 & 2 results vs paper
│   ├── replication_summary.md # 1-page replication summary
│   └── Phase3_Report.md     # 🆕 Full Phase 3 noise-robustness report
└── requirements.txt         # Project dependencies
```

---

## 🚀 Getting Started

### 1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Running Phase 1 & 2 Experiments
You can run the full benchmark or specific configurations using `run_experiment.py`.

**Quick Smoke Test (50 samples):**
```bash
python run_experiment.py --n_samples 50
```

**Full Replication — ANN (300 samples, both datasets):**
```bash
python run_experiment.py --dataset all --model ann --n_samples 300
```

**LR Model — Matches Paper Table 3 directly:**
```bash
python run_experiment.py --dataset adult --model lr --n_samples 300
```

---

### 3. Running Phase 3 — Noise Robustness Study

Phase 3 sweeps 4 noise levels (σ = 0.0, 0.1, 0.3, 0.5) and tests whether SmoothSHAP/SmoothLIME can recover explainer quality under noisy data.

**Full Phase 3 run (adult + compas):**
```bash
python run_phase3.py --dataset adult --n_samples 300
```

**Quick smoke-test (single sigma level):**
```bash
python run_phase3.py --dataset adult --n_samples 50 --sigma_only 0.3
```

**Resume a partially completed run:**
```bash
python run_phase3.py --dataset adult --n_samples 300 --resume
```

---

## 📊 Key Findings

### Phase 1 & 2 — Benchmark Replication
Our replication results (detailed in `report/phase2_summary.md`) align with the trends reported in the original paper:
- **Integrated Gradients** and **SmoothGrad** consistently show superior faithfulness (lowest PGU).
- **Stability** varies significantly across datasets, with gradient-based methods occasionally suffering from high variance.
- **LIME** remains a strong all-rounder but is computationally expensive.

### Phase 3 — Noise Robustness Extension
Our extension study (detailed in `report/Phase3_Report.md`) investigates how explainer quality degrades under real-world input noise:
- **Gradient-based methods** (VanillaGrad, GradxInput) show catastrophic instability (RIS explodes) even at σ=0.1.
- **LIME and SHAP** degrade more gracefully but still lose faithfulness at σ=0.5.
- **SmoothSHAP and SmoothLIME** (K=20 averaging copies) significantly recover stability and faithfulness, validating the smoothing approach as a practical noise-robustness fix.

### Phase 3 Team Assignments
| Member | Responsibility |
| :--- | :--- |
| **P B Shreyas** | Noise setup (`noise_utils.py`), Degradation study, Report Sections 1 & 2 |
| **Aditya Sahrawat** | SmoothSHAP/SmoothLIME (`smooth_explainers.py`), Before-after analysis, Report Sections 3 & 4 |
| **Alok Kumar** | All Phase 3 visualizations, pipeline integration (`run_phase3.py`), Report Sections 5 & 6 + final assembly |

For a full breakdown of the replication data, see the files generated in the `results/` folder after running the experiments.

---

## 💡 Theoretical Motivation — SmoothSHAP & SmoothLIME

Our Phase 3 extension introduces **SmoothSHAP** and **SmoothLIME** as noise-aware variants of the standard SHAP and LIME explainers.
This approach is directly inspired by **SmoothGrad** (Smilkov et al., 2017), which demonstrated that averaging gradients over noise-perturbed inputs produces more stable saliency maps for image classifiers.
We propose an analogous smoothing wrapper for tabular explainers:

> *Inspired by SmoothGrad (Smilkov et al., 2017), which demonstrated that averaging gradients over
> noise-perturbed inputs produces more stable saliency maps for image classifiers, we propose
> SmoothSHAP and SmoothLIME — analogous smoothing wrappers for tabular explainers. Formally:*
>
> **SmoothExplainer(x) = (1/K) &Sigma; E(x + &epsilon;_k)**&nbsp;&nbsp; where &nbsp;&nbsp;**&epsilon;_k ~ N(0, &sigma;&sup2;I)**

**SmoothGrad Citation:**
```bibtex
@article{smilkov2017smoothgrad,
  title={SmoothGrad: removing noise by adding noise},
  author={Smilkov, Daniel and Thorat, Nikhil and Kim, Been and Vi{\'e}gas, Fernanda and Wattenberg, Martin},
  journal={arXiv preprint arXiv:1706.03825},
  year={2017}
}
```

---

## 📜 Acknowledgments

This project uses the [OpenXAI library](https://github.com/AI4LIFE-GROUP/OpenXAI). We thank the authors of the original paper for providing the benchmark suite and pretrained models.

**Original Paper Citation:**
```bibtex
@inproceedings{agarwal2022openxai,
  title={OpenXAI: Towards a Transparent Evaluation of Post hoc Model Explanations},
  author={Agarwal, Chirag and Krishna, Satyapriya and Saxena, Eshika and Pawelczyk, Martin and Johnson, Nari and Puri, Isha and Zitnik, Marinka and Lakkaraju, Himabindu},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```
