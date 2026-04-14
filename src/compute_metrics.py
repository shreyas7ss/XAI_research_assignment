"""
compute_metrics.py — Evaluates explainer attributions using the five OpenXAI
benchmark metrics.

Real openxai 0.1 metric API
-----------------------------
Faithfulness (eval_pred_faithfulness):
  from openxai.metrics import eval_pred_faithfulness
  result, mean = eval_pred_faithfulness(
      explanations, inputs, model, k,
      perturb_method, feature_metadata, n_samples, invert
  )
  PGI (= PGF): invert=False
  PGU:         invert=True

Stability (eval_relative_stability):
  from openxai.metrics import eval_relative_stability
  result, mean = eval_relative_stability(
      explainer_obj, inputs, model, perturb_method,
      feature_metadata, metric='RIS'/'RRS'/'ROS'
  )
  Note: stability metrics require the live explainer object, not the tensor.

Note: PGF in the paper == PGI in the code (Prediction Gap on Important features).
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch

from openxai.metrics import eval_pred_faithfulness, eval_relative_stability
from openxai.explainers.perturbation_methods import get_perturb_method
from openxai import Explainer
import openxai.experiment_utils as utils

from src.config import RANDOM_SEED, set_seed

# ─── Monkey-patch OpenXAI bug ────────────────────────────────────────────────
# In openxai 0.1, convert_k_to_int fails to return the value if k is an int.
def fixed_convert_k_to_int(k, n_feat):
    if k == -1:
        return n_feat
    if isinstance(k, int):
        return k
    if isinstance(k, float):
        if 0 < k < 1:
            return int(np.ceil(k * n_feat))
    return k  # Fallback

utils.convert_k_to_int = fixed_convert_k_to_int
# ─────────────────────────────────────────────────────────────────────────────


# Top-k features to mask (as fraction of total features)
TOP_K_FRACTION: float = 0.25
# Perturbation standard deviation
PERTURB_STD: float = 0.1


def _safe_scalar(result, metric_name: str, explainer_name: str) -> float:
    """Extract a scalar from a metric result tuple, returning NaN on failure.

    Parameters
    ----------
    result : tuple or any
        Return value of an openxai metric function.
    metric_name : str
        Metric name for warning messages.
    explainer_name : str
        Explainer name for warning messages.

    Returns
    -------
    value : float
    """
    try:
        # openxai metric functions return (distribution_array, mean_float)
        if isinstance(result, (tuple, list)):
            val = result[1]
        else:
            val = result
        if isinstance(val, torch.Tensor):
            val = val.item()
        return float(val)
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARNING] Could not extract {metric_name} for '{explainer_name}': {exc}")
        return float("nan")


def compute_metrics_for_dataset(
    model: torch.nn.Module,
    X_test: torch.FloatTensor,
    y_test: torch.Tensor,
    X_train: torch.FloatTensor,
    explanations: Dict[str, torch.Tensor],
    data_name: str,
) -> pd.DataFrame:
    """Compute all five benchmark metrics for every explainer.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model in eval mode.
    X_test : torch.FloatTensor
        Feature tensor used for evaluation, shape ``(n, d)``.
    y_test : torch.Tensor
        Label tensor, shape ``(n,)``.
    X_train : torch.FloatTensor
        Training feature tensor (used for background in rebuilt explainers).
    explanations : dict[str, torch.Tensor]
        Mapping from explainer display name → attribution matrix ``(n, d)``.
    data_name : str
        Dataset name (for logging and feature metadata).

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame of shape ``(n_explainers, 5)`` indexed by explainer name.
    """
    set_seed(RANDOM_SEED)

    from openxai.dataloader import ReturnLoaders
    trainloader, _ = ReturnLoaders(data_name, download=False, batch_size=256)
    feature_metadata = trainloader.dataset.feature_metadata

    perturb_method = get_perturb_method(std=PERTURB_STD, data_name=data_name)
    n_features = X_test.shape[1]
    k = max(1, int(TOP_K_FRACTION * n_features))  # top-k as integer

    print(f"\n[compute_metrics] Dataset: '{data_name}' | k={k} of {n_features} features")

    # Map display name back to internal method key for rebuilding explainers (stability)
    from src.run_explainers import EXPLAINER_DISPLAY
    display_to_method = {v: k_key for k_key, v in EXPLAINER_DISPLAY.items()}

    records: Dict[str, Dict[str, float]] = {}

    for exp_name, attrs in explanations.items():
        print(f"  Evaluating '{exp_name}' …")
        row: Dict[str, float] = {}

        # ── Faithfulness: PGF (PGI, invert=False) ───────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = eval_pred_faithfulness(
                    explanations=attrs, inputs=X_test, model=model, k=k,
                    perturb_method=perturb_method, feature_metadata=feature_metadata,
                    n_samples=100, invert=False, seed=RANDOM_SEED,
                )
            row["PGF"] = _safe_scalar(result, "PGF", exp_name)
        except Exception as exc:
            print(f"    [WARNING] PGF failed for '{exp_name}': {exc}")
            row["PGF"] = float("nan")

        # ── Faithfulness: PGU (invert=True) ─────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = eval_pred_faithfulness(
                    explanations=attrs, inputs=X_test, model=model, k=k,
                    perturb_method=perturb_method, feature_metadata=feature_metadata,
                    n_samples=100, invert=True, seed=RANDOM_SEED,
                )
            row["PGU"] = _safe_scalar(result, "PGU", exp_name)
        except Exception as exc:
            print(f"    [WARNING] PGU failed for '{exp_name}': {exc}")
            row["PGU"] = float("nan")

        # ── Stability metrics (RIS, RRS, ROS) ───────────────────────────────
        # Stability requires the live explainer object, so we rebuild it.
        method_key = display_to_method.get(exp_name, exp_name)
        param_dict = utils.fill_param_dict(method_key, {}, X_train) \
            if method_key in ("lime", "ig") else \
            ({"n_samples": 100} if method_key == "shap" else {})

        explainer_obj = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            explainer_obj = Explainer(method=method_key, model=model, param_dict=param_dict)
        except Exception as exc:
            print(f"    [WARNING] Cannot rebuild explainer '{exp_name}' for stability: {exc}")

        for metric in ("RIS", "RRS", "ROS"):
            if explainer_obj is None:
                row[metric] = float("nan")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = eval_relative_stability(
                        explainer_obj, X_test, model, perturb_method,
                        feature_metadata, metric=metric,
                        n_samples=50, n_perturbations=10, seed=RANDOM_SEED,
                    )
                row[metric] = _safe_scalar(result, metric, exp_name)
            except Exception as exc:
                print(f"    [WARNING] {metric} failed for '{exp_name}': {exc}")
                row[metric] = float("nan")

        records[exp_name] = row

    df = pd.DataFrame.from_dict(records, orient="index",
                                 columns=["PGF", "PGU", "RIS", "RRS", "ROS"])
    df.index.name = "Explainer"
    print(f"[compute_metrics] Done for '{data_name}'.\n{df.to_string()}\n")
    return df
