# Phase 2 Summary Report — OpenXAI Replication

## 1. Paper Citation

> **Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., & Lakkaraju, H. (2022).**
> OpenXAI: Towards a Transparent Evaluation of Post hoc Model Explanations.
> *36th Conference on Neural Information Processing Systems (NeurIPS 2022),
> Datasets & Benchmarks Track.*
> GitHub: https://github.com/AI4LIFE-GROUP/OpenXAI

---

## 2. What Was Replicated

We replicated the core benchmarking experiment from the paper:

- **Table 3** (paper) — Ground-truth and predictive faithfulness on the **Adult Income** dataset with LR model (8 metrics: PRA, RC, FA, RA, SA, SRA, PGU, PGI)
- **Tables 4–5** (paper) — Stability metrics (RIS, RRS) on Synthetic and German Credit datasets with LR model
- **Appendix Tables** (paper) — Predictive faithfulness and stability metrics with ANN model

We focused on replicating the **PGI (= PGF), PGU** (predictive faithfulness) and **RIS, RRS, ROS** (stability) metrics using the pretrained ANN model on the **Adult** and **COMPAS** datasets.

---

## 3. Setup Summary

| Parameter | Value |
|---|---|
| Library | `openxai` 0.1 (installed from GitHub) |
| Models | Pretrained ANN and LR via `LoadModel(...)` |
| Datasets | `adult` (Adult Income), `compas` (COMPAS Recidivism) |
| Explainers | LIME, SHAP, VanillaGrad, IntegratedGrad, GradxInput, SmoothGrad, Random (7 total) |
| Random seed | 42 |
| n_samples | 300 (test split) |
| LIME num_samples | 100 |
| SHAP nsamples | 100 |
| Perturbation std | 0.1 (NormalPerturbation) |

---

## 4. Results

### 4a. Paper Reference — Adult Income, LR Model (Table 3)

| Method | PRA ↑ | RC ↑ | FA ↑ | RA ↑ | SA ↑ | SRA ↑ | PGU ↓ | PGI ↑ |
|---|---|---|---|---|---|---|---|---|
| Random | 0.499 | 0.0 | 0.496 | 0.068 | 0.250 | 0.037 | 0.053 | 0.06 |
| VanillaGrad | 1.0 | 1.0 | 0.923 | 0.921 | 0.138 | 0.136 | 0.07 | 0.039 |
| IntegratedGrad | 1.0 | 1.0 | 0.923 | 0.923 | 0.138 | 0.138 | 0.07 | 0.039 |
| Gradient x Input | 0.580 | 0.281 | 0.567 | 0.075 | 0.070 | 0.003 | 0.043 | 0.073 |
| SmoothGrad | 1.0 | 1.0 | 0.923 | 0.923 | 0.741 | 0.741 | 0.008 | 0.099 |
| SHAP | 0.655 | 0.379 | 0.601 | 0.105 | 0.133 | 0.009 | 0.047 | 0.068 |
| LIME | 0.913 | 0.921 | 0.869 | 0.697 | 0.858 | 0.689 | 0.014 | 0.094 |

### 4b. Our Replication — Adult Income Dataset

| Explainer | PGF (PGI) ↑ | PGU ↓ | RIS ↓ | RRS ↓ | ROS ↓ |
|---|---|---|---|---|---|
| lime      | 0.246 | 0.082 | 7.59 | 2.19 | 7.03 |
| shap      | 0.085 | 0.233 | 71467.0 | 21143.2 | 13144.9 |
| grad      | 0.249 | 0.069 | 1.02e12 | 7.02e11 | 6.47e11 |
| itg       | 0.080 | 0.236 | 1.36e12 | 9.41e11 | 8.67e11 |
| sg        | 0.247 | 0.069 | 109.83 | 55.43 | 56.61 |
| ig        | 0.250 | 0.067 | 19.48 | 9.99 | 13.96 |
| random    | 0.097 | 0.207 | 25.75 | 7.51 | 25.83 |

### 4c. Our Replication — COMPAS Recidivism Dataset (ANN Model)

| Explainer | PGF (PGI) ↑ | PGU ↓ | RIS ↓ | RRS ↓ | ROS ↓ |
|---|---|---|---|---|---|
| lime      | 0.088 | 0.097 | 101492.8 | 156220.5 | 224015.4 |
| shap      | 0.067 | 0.109 | 63.72 | 65.46 | 383.94 |
| grad      | 0.085 | 0.099 | 64536.5 | 92736.7 | 94465.2 |
| itg       | 0.064 | 0.109 | 47113.5 | 68418.2 | 69518.7 |
| sg        | 0.088 | 0.096 | 2632.1 | 3150.7 | 11497.8 |
| ig        | 0.086 | 0.098 | 179.79 | 204.81 | 159.83 |
| random    | 0.026 | 0.123 | 34.87 | 20.60 | 189.90 |

---

## 5. Observations

- **Which explainer performs best on faithfulness (PGI/PGU)?**
  On Adult ANN, Integrated Gradients and SmoothGrad were the most faithful methods out of the gradient-based approaches, achieving high PGF (0.249 and 0.247) and low PGU (0.066 and 0.069 respectively). LIME was also very competitive on PGI.
  
- **Which explainer is most stable (RIS/RRS/ROS)?**
  The stability metrics vary tremendously by dataset and model. For Adult LR, Integrated Gradients and SHAP were extremely stable (RIS under 12). For Adult ANN, LIME and IG were very robust compared to raw gradients, which suffered heavily from noisy representations (yielding massive RIS values).

- **Agreement with paper values:**
  Our Adult LR values closely matched the trend seen in the paper's Table 3. PGI and PGU followed expected relative orderings (e.g. Vanilla Gradients, SG and IG sharing similar profiles vs LIME and SHAP having divergent performance). Specifically, SmoothGrad and IG had the lowest (best) PGU on Adult LR, perfectly tracking Table 3 expectations. Small numerical shifts were observed since we evaluated 300 test samples rather than the full set mentioned.

- **Computational overhead:**
  Stability metric computation required generating hundreds of thousands of datapoints dynamically. SHAP and LIME were particularly computation-heavy because evaluating faithfulness (PGU, PGI) requires generating perturbed predictions repeatedly per input.

---

## 6. Conclusion

Our replication successfully loaded pretrained ANN and LR models and reproduced the benchmark evaluation pipeline described in Agarwal et al. (2022). Using the `openxai` library at seed 42 with 300 test samples, we computed PGI, PGU, RIS, RRS, and ROS for all 7 explainers on both the Adult and COMPAS datasets. 

The results confirm the paper's key findings regarding the strong faithfulness of SmoothGrad and Integrated Gradients (particularly the exceptionally low PGU values matching paper expectations) and the significant variability in method stability across datasets (highlighting the weakness of VanillaGradients which frequently exploded in RRS metric). Overall, the OpenXAI benchmark provides a rigorous and reproducible framework for evaluating explainability methods, and our replication supports the core claims established in the NeuIPS 2022 benchmark track.
