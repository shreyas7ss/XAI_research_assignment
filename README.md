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

| Name | ID | Role |
| :--- | :--- | :--- |
| **ALok kumar** | 23BCS003 | Pipeline architecture, Metric integration |
| **P B Shreyas** | 23BDS041 | Visualizations, Results analysis |
| **Adtiya sahrawat** | 23BCS006 | Documentation, Experiment validation |

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
- **Datasets**: Adult Income (Census), COMPAS (Recidivism).
- **Models**: Pretrained 2-layer ANN and Logistic Regression (LR).

---

## 📂 Repository Structure

```text
XAI_project/
├── run_experiment.py       # 🚀 Main entry point for running benchmarks
├── src/
│   ├── config.py           # Configuration, seeds, and constants
│   ├── train_model.py      # Data loading and model management
│   ├── run_explainers.py   # Generation of attribution maps
│   ├── compute_metrics.py  # Faithfulness and Stability calculations
│   └── visualize_results.py # Figure and Table generation
├── data/                   # Dataset storage (auto-downloaded)
├── models/                 # Pretrained model weights
├── results/                # 📈 Output directory for CSVs and PNGs
│   ├── tables/             # Raw metric data in CSV format
│   └── *.png               # Generated visualization heatmaps and bar charts
├── report/                 # Detailed replication summaries
└── requirements.txt        # Project dependencies
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

### 2. Running Experiments
You can run the full benchmark or specific configurations using `run_experiment.py`.

**Quick Smoke Test (50 samples):**
```bash
python run_experiment.py --n_samples 50
```

**Full Replication (300 samples, all datasets):**
```bash
python run_experiment.py --dataset all --n_samples 300
```

**Specific Model/Dataset Configuration:**
```bash
python run_experiment.py --dataset adult --model lr --n_samples 300
```

---

## 📊 Key Findings

Our replication results (detailed in `report/phase2_summary.md`) align with the trends reported in the original paper:
- **Integrated Gradients** and **SmoothGrad** consistently show superior faithfulness (lowest PGU).
- **Stability** varies significantly across datasets, with gradient-based methods occasionally suffering from high variance.
- **LIME** remains a strong all-rounder but is computationally expensive.

For a full breakdown of the replication data, see the files generated in the `results/` folder after running the experiment.

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
