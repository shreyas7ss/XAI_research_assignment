# OpenXAI — NeurIPS 2022 Replication

> **Paper:** Agarwal et al., "OpenXAI: Towards a Transparent Evaluation of Model Explanations," *NeurIPS 2022 Datasets & Benchmarks Track.*

## What Is Being Replicated

This project replicates **Tables 1–2** (faithfulness and stability metric benchmarks) and the corresponding heatmap figures from the OpenXAI paper. Specifically, we:

- Evaluate **7 post-hoc explainability methods** (LIME, SHAP, Gradient, GradCAM, SmoothGrad, Integrated Gradients, Random baseline) on pretrained ANNs.
- Measure **5 quantitative metrics**: PGF, PGU (faithfulness) and RIS, RRS, ROS (stability).
- Run experiments on the **Adult Income** and **COMPAS Recidivism** datasets.

---

## Repository Structure

```
XAI_project/
├── requirements.txt
├── run_experiment.py          # ← main entry-point
├── src/
│   ├── config.py              # paths, seeds, constants
│   ├── train_model.py         # data / model loading
│   ├── run_explainers.py      # attribution generation
│   ├── compute_metrics.py     # metric computation
│   └── visualize_results.py   # figure generation
├── results/
│   ├── tables/                # CSVs saved here
│   ├── *.png                  # figures saved here
└── report/
    └── phase2_summary.md      # written report
```

---

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `openxai` will automatically download pretrained model weights and dataset files on first run.

---

## Running the Experiment

### Quick smoke-test (50 samples, ~2 minutes)

```bash
python run_experiment.py --n_samples 50
```

### Full replication run (300 samples, both datasets)

```bash
python run_experiment.py --dataset all --n_samples 300
```

### Single dataset

```bash
python run_experiment.py --dataset adult --n_samples 300
python run_experiment.py --dataset compas --n_samples 300
```

---

## Output Files

| File | Description |
|---|---|
| `results/tables/adult_metrics.csv` | Adult dataset metric table |
| `results/tables/compas_metrics.csv` | COMPAS dataset metric table |
| `results/adult_heatmap.png` | Heatmap (replicates paper Table style) |
| `results/adult_bar_charts.png` | Per-metric bar charts |
| `results/compas_heatmap.png` | Heatmap for COMPAS |
| `results/compas_bar_charts.png` | Bar charts for COMPAS |
| `results/multi_dataset_comparison.png` | Adult vs COMPAS grouped bars |

---

## Random Seed

All experiments use **seed = 42**, set globally via `src.config.set_seed()` which fixes `random`, `numpy`, and `torch`.

---

## Team Contributions

| Name | Contribution |
|---|---|
| [Team Member A] | Pipeline architecture, metric integration |
| [Team Member B] | Visualizations, report writing |
| [Team Member C] | Literature review, experiment validation |
| [Team Member D] | Documentation, README, AI usage log |
