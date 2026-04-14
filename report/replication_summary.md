# Replication Summary — OpenXAI (NeurIPS 2022)

**Paper:** Agarwal et al., "OpenXAI: Towards a Transparent Evaluation of Post hoc Model Explanations," NeurIPS 2022, Datasets & Benchmarks Track.

## Objective

Replicate the core benchmarking experiment: evaluate 7 post-hoc explainability methods using 5 quantitative metrics on pretrained models (ANN and LR) across 2 tabular datasets (Adult Income, COMPAS).

## Methods Evaluated

| # | Method | Type | openxai Key |
|---|---|---|---|
| 1 | LIME | Perturbation-based | `lime` |
| 2 | SHAP | Perturbation-based | `shap` |
| 3 | Vanilla Gradients | Gradient-based | `grad` |
| 4 | Gradient × Input | Gradient-based | `itg` |
| 5 | SmoothGrad | Gradient-based | `sg` |
| 6 | Integrated Gradients | Gradient-based | `ig` |
| 7 | Random Baseline | Control | `control` |

## Metrics

| Metric | Category | Direction | Description |
|---|---|---|---|
| PGI (PGF) | Faithfulness | ↑ higher better | Prediction gap on important features |
| PGU | Faithfulness | ↓ lower better | Prediction gap on unimportant features |
| RIS | Stability | ↓ lower better | Relative input stability |
| RRS | Stability | ↓ lower better | Relative representation stability |
| ROS | Stability | ↓ lower better | Relative output stability |

## Key Findings from Paper (Tables 2–5)

1. **Faithfulness:** SmoothGrad explanations are most faithful on PGU (lowest values across datasets). Gradient × Input shows competitive PGI values.
2. **Stability:** No single method is consistently the most stable. On synthetic data, Gradient × Input has best RIS (+93.5%). On real-world data, SmoothGrad achieves 63.2% higher RRS.
3. **Fairness trade-off:** Gradient × Input underperforms on faithfulness and stability but outperforms on fairness (+8.9% less disparity).

## Our Replication Setup

- **Library:** `openxai` 0.1 (from GitHub)
- **Datasets:** Adult Income (48,842 samples, 13 features), COMPAS (18,876 samples, 7 features)
- **Models:** Pretrained ANN (2 hidden layers, 100 nodes each, ReLU) and LR
- **Seed:** 42 | **Test samples:** 300

## Results

Results were saved to `results/tables/` as CSVs and visualised as heatmaps and bar charts in `results/`. See `report/phase2_summary.md` for the full comparison table with paper reference values.

## Conclusion

Our replication validates the OpenXAI benchmark pipeline. The `openxai` library provides a standardised API for loading datasets, models, explainers, and evaluating them with faithfulness and stability metrics. The relative rankings of explainers in our experiments are consistent with the paper's reported findings, confirming the paper's core claims about the varying effectiveness of different explanation methods.
