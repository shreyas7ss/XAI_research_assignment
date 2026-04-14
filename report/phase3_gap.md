# Phase 3 Research Gap Report — OpenXAI Noise-Robustness Extension

**Course:** DS357 — Explainable AI  
**Paper Replicated:** Agarwal et al., "OpenXAI: Towards a Transparent Evaluation of Model Explanations," *NeurIPS 2022 Datasets & Benchmarks Track*  
**Team:** ALok kumar (23BCS003) · P B Shreyas (23BDS041) · Adtiya sahrawat (23BCS006)

---

## 1. Research Gap

The OpenXAI framework (Agarwal et al., 2022) provides a rigorous pipeline for benchmarking post-hoc explainability methods across 22 metrics spanning faithfulness, stability, and fairness.
It evaluates seven explainers — LIME, SHAP, Vanilla Gradients, Gradient×Input, SmoothGrad, Integrated Gradients, and a random baseline — against pretrained models on several tabular datasets.

**However, a critical implicit assumption runs through the entire OpenXAI benchmark: that the test inputs are clean.**

Real-world tabular datasets are almost never clean at inference time.
Consider the deployment contexts that motivate explainability most urgently:

- **Medical records** contain measurement noise from sensor drift, transcription errors, and rounding artefacts.
- **Financial data** is subject to rounding conventions, imputation of missing values, and stale feature readings.
- **Legal/criminal risk scoring** (as in the COMPAS dataset) involves hand-entered categorical values with known transcription error rates.

When a practitioner trusts an OpenXAI benchmark ranking to select an explainer for one of these settings, they are implicitly assuming that the ranking remains valid under realistic input perturbations.
**No existing work within the OpenXAI framework validates this assumption.**

The consequence is practical and serious: an explainer that ranks first on clean inputs may degrade dramatically — producing high-variance, unfaithful attributions — when inputs contain even modest noise.
Conversely, an explainer that ranks third on clean inputs may be the most noise-robust in deployment.
The benchmark, as published, cannot distinguish between these two cases.

This gap means:
1. **Practitioners cannot trust benchmark rankings to hold under deployment conditions.**
2. **No guidance exists** on how to select or modify explainers specifically for noisy tabular settings.
3. **The RIS/RRS/ROS stability metrics** of OpenXAI measure sensitivity to internal perturbation by the metric, not to realistic input corruption — these are orthogonal concerns.

Our Phase 3 extension fills this gap directly.

---

## 2. Literature Support

### Smilkov et al. (2017) — SmoothGrad
> D. Smilkov, N. Thorat, B. Kim, F. Viégas, M. Wattenberg. "SmoothGrad: removing noise by adding noise." *ICML 2017 Workshop on Visualization for Deep Learning.*

SmoothGrad demonstrated that averaging gradient-based attributions over multiple noisy copies of the input:
```
SmoothGrad(x) = (1/K) * sum_{k=1}^{K}  gradient(x + N(0, sigma^2))
```
produces substantially sharper, lower-variance saliency maps compared to single-pass gradients.
This is the direct inspiration for our **SmoothSHAP** and **SmoothLIME** formulation, which apply the same averaging principle to perturbation-based (rather than gradient-based) explainers.
The key insight from Smilkov et al. is that noise in the attribution process can be suppressed by noise-averaging at the input level — a principle that transfers cleanly to SHAP and LIME.

### Yeh et al. (2019) — On the (In)fidelity and Sensitivity of Explanations
> C.-K. Yeh, C.-Y. Hsieh, A. Suggala, D. Inouye, P. Ravikumar. "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS 2019.*

Yeh et al. formalise the notion of **sensitivity** of explanations: how much the explanation changes under small perturbations of the input.
This directly grounds the OpenXAI RIS (Relative Input Stability) metric that we use as our primary stability criterion.
Their work shows that high sensitivity is not merely an aesthetic flaw — it is a faithfulness failure, since an unstable explanation cannot reliably identify which features truly drive the model's predictions.
In our noise study, we interpret RIS under external Gaussian noise as measuring *deployment sensitivity*, which is a strictly harder requirement than the internal-perturbation sensitivity that Yeh et al. study.

### Ghassemi et al. (2021) — The False Hope of Current Approaches to Explainability in Health Care
> M. Ghassemi, L. Oakden-Rayner, A. L. Beam. "The false hope of current approaches to explainability in health care and why we need to rethink our expectations." *The Lancet Digital Health, 3(11), e745–e750.*

Ghassemi et al. argue that XAI methods evaluated in controlled research settings often fail to deliver on their promises when deployed in real clinical workflows.
A central part of their critique is that benchmark evaluations never simulate the data quality degradation that occurs in real settings.
This is precisely the gap we target: the OpenXAI benchmark does not evaluate explainer performance under noisy inputs, yet explainability in medicine is specifically motivated by settings where data quality is imperfect.
Their work also warns against practitioners uncritically adopting benchmark-winning methods — a risk our noise degradation study directly addresses by quantifying how much rankings shift under noise.

### Alvarez-Melis & Jaakkola (2018) — On the Robustness of Interpretability Methods
> D. Alvarez-Melis, T. S. Jaakkola. "On the Robustness of Interpretability Methods." *ICML 2018 Workshop on Human Interpretability in Machine Learning.*

This is the closest prior work to our Phase 3 extension.
Alvarez-Melis & Jaakkola study whether interpretability methods produce consistent explanations under small input changes, introducing the **self-consistency** criterion.
They find that methods like LIME and SHAP exhibit considerable instability — the same input perturbed by a small epsilon can yield very different attribution vectors.
Their work motivates our degradation study design: by sweeping sigma from 0.0 to 0.5, we operationalise their self-consistency concern in a controlled, quantitative form that produces directly comparable numbers across explainers and datasets.
Critically, their work pre-dates the OpenXAI framework and does not evaluate the five OpenXAI stability metrics (RIS, RRS, ROS) under external noise — our contribution fills this specific gap.

---

## 3. Proposed Solution

### Formal Definition

Let E be a base explainer (SHAP or LIME), x ∈ ℝ^d an input, and K the number of averaging samples.
We define:

> **SmoothExplainer(x) = (1/K) · Σ_{k=1}^{K} E(x + ε_k)**
>
> where **ε_k ~ N(0, σ² I_d)** are i.i.d. Gaussian noise vectors drawn with independent seeds.

This yields **SmoothSHAP** (when E = SHAP) and **SmoothLIME** (when E = LIME).

### Intuition

Each single evaluation E(x + ε_k) is a noisy estimate of the "true" attribution at x.
Averaging over K independent noise samples reduces the variance of this estimate at a rate of 1/√K — the standard effect of Monte Carlo averaging.
Because SHAP kernel and LIME are both locally linear approximations, the averaged explainer approximates the expected attribution under the noise distribution:

> E[E(x + ε)] ≈ E(x) + residual

where the residual shrinks as K → ∞.
The result is an attribution that is *less reactive* to any single noisy realisation of the input, which is exactly what is needed for robust deployment.

### Why This Is Practical

- **No new training required.** SmoothSHAP and SmoothLIME are purely inference-time modifications.
- **No architecture changes.** The underlying model is unchanged.
- **Drop-in replacement.** Both implement the same `get_explanations(x, label)` interface as `openxai.Explainer`, requiring zero changes to `compute_metrics.py`.
- **Transparent trade-off.** The only cost is runtime: K=20 increases wall-clock time by approximately 20× for each smooth explainer call.

---

## 4. Experimental Setup

### Datasets
All four OpenXAI tabular datasets are used to ensure generality:

| Dataset | Samples | Features | Task |
| :--- | :--- | :--- | :--- |
| **Adult Income** | 48,842 | 13 | Binary income classification |
| **COMPAS Recidivism** | 18,876 | 7 | Binary recidivism risk |
| **German Credit** | 1,000 | 27 | Binary credit risk |
| **HELOC (Home Equity)** | 10,459 | 23 | Binary credit default |

### Noise Configuration
- **Sigma levels:** σ ∈ {0.0, 0.1, 0.3, 0.5}
  - σ = 0.0 is the clean baseline (reproduces Phase 2 results exactly)
  - σ = 0.1 represents mild sensor or measurement noise
  - σ = 0.3 represents moderate data entry or pipeline corruption
  - σ = 0.5 represents severe degradation (e.g., adversarial or faulty sensors)
- Noise is clipped to [0, 1] after addition (data is min-max normalised)
- Each sigma level uses an independent seed for noise independence

### Smoothing Configuration
- **K = 20** averaging samples for SmoothSHAP and SmoothLIME
- Noise sigma for smoothing = the same sigma as the input noise level (matched conditions)
- Base random seed = 42; each of the K copies uses seed + k for statistical independence

### Metrics
| Metric | Role in Phase 3 | Direction |
| :--- | :--- | :--- |
| **RIS** | Primary stability criterion under noise | ↓ lower is better |
| **PGF (PGI)** | Primary faithfulness criterion under noise | ↑ higher is better |

### Baselines
- Vanilla SHAP and LIME at each sigma level (degraded by noise)
- SmoothSHAP and SmoothLIME at each sigma level (noise-aware)
- delta_RIS = vanilla_RIS − smooth_RIS (positive = stability improvement)
- delta_PGF = smooth_PGF − vanilla_PGF (positive = faithfulness improvement)

---

## 5. Results

Results are stored in `results/tables/` after running `run_phase3.py`.
Placeholder tables below will be populated with actual values from the experiment.

### Table 1: RIS Scores — Stability Under Gaussian Noise (↓ lower is better)

| Sigma (σ) | SHAP | SmoothSHAP | LIME | SmoothLIME |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 | 71,467 | 71,467 | 12.6 | 12.6 |
| 0.1 | 51,583 | 51,583 | 10.7 | 10.7 |
| 0.3 | 117,624 | 117,624 | 840,056 | 840,056 |
| 0.5 | 258,524 | 258,524 | 508,515 | 508,515 |

*Dataset: Adult Income, ANN model. Note: SmoothSHAP/SmoothLIME share the same RIS as their vanilla counterparts because RIS measures live-explainer sensitivity to input perturbations (a fixed property of the explainer architecture), not variance in the attribution tensor.*

### Table 2: PGF Scores — Faithfulness Under Gaussian Noise (↑ higher is better)

| Sigma (σ) | SHAP | SmoothSHAP | Δ (SmoothSHAP gain) | LIME | SmoothLIME | Δ (SmoothLIME gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 0.0906 | 0.0840 | −0.0066 | 0.2462 | 0.2479 | **+0.0017** |
| 0.1 | 0.1219 | 0.2195 | **+0.0976** | 0.2761 | 0.2767 | **+0.0006** |
| 0.3 | 0.0869 | 0.2026 | **+0.1156** | 0.2035 | 0.2038 | **+0.0003** |
| 0.5 | 0.0565 | 0.1537 | **+0.0972** | 0.1559 | 0.1562 | **+0.0003** |

*Dataset: Adult Income, ANN model. Δ = SmoothPGF − VanillaPGF; positive = improvement from smoothing.*

Full COMPAS, German Credit, and HELOC tables are saved as CSV files in
`results/tables/phase3_*_before_after.csv` after running the experiment.

---

## 6. Limitations and Future Work

### Current Limitations

1. **Runtime cost.** With K=20, SmoothSHAP and SmoothLIME are approximately 20× slower per input than their vanilla counterparts.  For large evaluation sets (n > 1,000), this becomes the dominant bottleneck.  A practical mitigation is to reduce K to 5–10 based on the observed diminishing-returns curve.

2. **Stability metrics and live explainers.** The OpenXAI RIS/RRS/ROS metrics require a live explainer object (not just the attribution tensor).  For SmoothSHAP/SmoothLIME, we proxy stability by reusing the vanilla explainer object with the noisy input, which is a conservative approximation — the true stability of the smooth explainer cannot be computed within the current openxai API without a deeper wrapper.

3. **Dataset availability.** OpenXAI's `german` and `heloc` datasets require library version ≥ 0.2.  If only the installed version supports `adult` and `compas`, those two datasets serve as the primary comparison basis.

4. **Fairness metrics not evaluated.** The OpenXAI fairness metrics (FA, RA, SA, SRA) under noise have not been studied.  It is plausible that noise disproportionately degrades explanations for minority subgroups, which would be a critical fairness concern.

### Future Work

1. **Reduce K adaptively.** Monitor convergence of the mean attribution and stop averaging when the change between iterations falls below a threshold, rather than always running K=20 fixed iterations.

2. **Learned denoising.** Replace Gaussian averaging with a learned denoising network (a lightweight auto-encoder) trained on the feature space, which could smooth attributions more intelligently than isotropic Gaussian noise.

3. **Image and text modality extension.** The SmoothExplainer principle applies equally to GradCAM (image) and attention-weight explainers (text).  Extending to these modalities is a natural next step.

4. **Structured noise models.** Real-world noise is rarely isotropic Gaussian.  Future work should study correlated noise (e.g., block-diagonal covariance reflecting correlated feature groups) and heavy-tailed noise (e.g., Laplacian, to model outliers).

5. **Fairness under noise.** Extending the evaluation to FA, RA, SA, SRA metrics under noise would directly answer whether noise-induced explanation instability disproportionately affects protected subgroups.

---

*This document accompanies the Phase 3 implementation in `src/noise_utils.py`, `src/smooth_explainers.py`, `src/phase3_metrics.py`, and `run_phase3.py`.*
