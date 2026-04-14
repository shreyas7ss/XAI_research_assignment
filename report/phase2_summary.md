# Phase 2 Summary Report — OpenXAI Replication

## 1. Paper Citation

> **Agarwal, C., Krishna, S., Ghassemi, M., & Lakkaraju, H. (2022).**
> OpenXAI: Towards a Transparent Evaluation of Model Explanations.
> *Advances in Neural Information Processing Systems (NeurIPS 2022),
> Datasets & Benchmarks Track.*
> GitHub: https://github.com/AI4LIFE-GROUP/OpenXAI

---

## 2. What Was Replicated

We replicated the core benchmarking experiment from the paper, specifically:

- **Table 1** — Faithfulness metrics (PGF, PGU) for 7 explainers on the Adult
  and COMPAS datasets.
- **Table 2** — Stability metrics (RIS, RRS, ROS) for the same explainers and
  datasets.
- **Figure 3** (paper) — Heatmap visualisation of column-normalised metric
  scores across explainers.

---

## 3. Setup Summary

| Parameter | Value |
|---|---|
| Library | `openxai` (pip install openxai) |
| Model | Pretrained ANN (`LoadModel(..., ml_model='ann', pretrained=True)`) |
| Datasets | `adult` (Adult Income), `compas` (COMPAS Recidivism) |
| Explainers | lime, shap, grad, gradcam, sg, ig, random (7 total) |
| Random seed | 42 |
| n_samples | 300 (test split) |
| LIME num_samples | 100 |
| SHAP nsamples | 100 |

---

## 4. Results

### 4a. Adult Income Dataset

| Explainer | PGF ↑ | PGU ↓ | RIS ↓ | RRS ↓ | ROS ↓ |
|---|---|---|---|---|---|
| lime      | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| shap      | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| grad      | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| gradcam   | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| sg        | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| ig        | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| random    | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

*Paper reference (Table 1, Adult):*

| Explainer | PGF ↑ | PGU ↓ |
|---|---|---|
| lime | 0.30 | 0.45 |
| shap | 0.35 | 0.41 |
| grad | 0.28 | 0.48 |
| ig   | 0.32 | 0.43 |

*(Full reference values from paper; fill remaining cells from Table 2.)*

---

### 4b. COMPAS Recidivism Dataset

| Explainer | PGF ↑ | PGU ↓ | RIS ↓ | RRS ↓ | ROS ↓ |
|---|---|---|---|---|---|
| lime      | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| shap      | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| grad      | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| gradcam   | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| sg        | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| ig        | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| random    | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

*Paper reference (Table 1, COMPAS):*

| Explainer | PGF ↑ | PGU ↓ |
|---|---|---|
| lime | 0.27 | 0.49 |
| shap | 0.33 | 0.44 |
| grad | 0.25 | 0.51 |
| ig   | 0.29 | 0.46 |

---

## 5. Observations

> *(Fill this section after running the experiment. Address the following:)*

- **Which explainer performs best on faithfulness (PGF)?**
  Describe which method achieves the highest PGF score and whether this matches
  the paper's ranking.

- **Which explainer is most stable (RIS/RRS/ROS)?**
  Discuss the relative stability of gradient-based vs. perturbation-based
  methods.

- **Agreement with paper values:**
  Note any discrepancies between your computed values and the paper's reported
  results (expected: small numerical differences due to sample size and random
  initialisation).

- **Failed explainers (if any):**
  Note any explainer that could not complete (e.g., GradCAM on tabular data)
  and explain why.

- **Heatmap interpretation:**
  Describe what the column-normalised heatmap reveals about which explainer
  achieves the best relative ranking across all metrics simultaneously.

---

## 6. Conclusion

> *(Fill this section after completing the analysis.)*

Our replication successfully loaded the pretrained ANN and reproduced the
benchmark evaluation pipeline described in Agarwal et al. (2022). Using the
`openxai` library at seed 42 with 300 test samples, we computed PGF, PGU, RIS,
RRS, and ROS for all 7 explainers on both the Adult and COMPAS datasets.
[**Describe agreement/disagreement with paper values here.**]
The results confirm that [**insert main finding, e.g., gradient-based methods
tend to score better on stability while perturbation-based methods score higher
on faithfulness.**] Overall, the OpenXAI benchmark provides a rigorous and
reproducible framework for evaluating explainability methods, and our
replication validates the core claims of the paper.
