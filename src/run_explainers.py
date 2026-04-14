"""
run_explainers.py — Generates feature-attribution explanations using the real
openxai 0.1 Explainer API.

Real API (openxai 0.1)
-----------------------
  from openxai import Explainer
  explainer = Explainer(method, model, param_dict={})

  Available methods: 'lime', 'shap', 'grad', 'sg', 'itg', 'ig', 'control'
  (Note: 'gradcam' is NOT in this version; 'random' baseline is called 'control')

  For LIME: param_dict must contain 'data' (FloatTensor of training data)
  For IG:   param_dict must contain 'baseline' (mean of training data, shape (1, d))
  Other methods: empty param_dict is fine

  Calling explanations:
    explainer.get_explanations(X, label=predicted_label)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import numpy as np

from openxai import Explainer
import openxai.experiment_utils as utils

from src.config import RANDOM_SEED, set_seed


# Correct method keys for openxai 0.1
EXPLAINER_METHODS: list[str] = ["lime", "shap", "grad", "sg", "itg", "ig", "control"]
# Human-readable display names
EXPLAINER_DISPLAY: dict[str, str] = {
    "lime": "lime", "shap": "shap", "grad": "grad",
    "sg": "sg", "itg": "itg", "ig": "ig", "control": "random",
}


def _build_param_dict(method: str, X_train: torch.FloatTensor) -> dict:
    """Build the correct ``param_dict`` for each explainer method.

    Parameters
    ----------
    method : str
        The explainer method key.
    X_train : torch.FloatTensor
        Full training data tensor, used as background/reference.

    Returns
    -------
    param_dict : dict
        Ready-to-pass parameter dictionary.
    """
    param_dict: dict = {}
    if method == "lime":
        param_dict = utils.fill_param_dict("lime", {"n_samples": 100}, X_train)
    elif method == "ig":
        param_dict = utils.fill_param_dict("ig", {}, X_train)
    elif method == "shap":
        param_dict = {"n_samples": 100}
    return param_dict


def _build_explainer(
    method: str,
    model: torch.nn.Module,
    X_train: torch.FloatTensor,
) -> Optional[object]:
    """Instantiate a single OpenXAI Explainer with graceful error handling.

    Parameters
    ----------
    method : str
        Explainer key (e.g. ``'lime'``, ``'shap'``).
    model : torch.nn.Module
        The model to explain.
    X_train : torch.FloatTensor
        Training data for methods that need a background dataset.

    Returns
    -------
    explainer : object or None
        Instantiated explainer, or ``None`` if construction failed.
    """
    param_dict = _build_param_dict(method, X_train)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = Explainer(method=method, model=model, param_dict=param_dict)
        return explainer
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARNING] Could not instantiate explainer '{method}': {exc}")
        return None


def _get_attributions(
    explainer,
    X: torch.FloatTensor,
    model: torch.nn.Module,
    method: str,
) -> Optional[torch.Tensor]:
    """Call the explainer and return a normalised 2-D attribution tensor.

    Parameters
    ----------
    explainer : openxai Explainer object
        Instantiated explainer.
    X : torch.FloatTensor
        Input batch of shape ``(n, d)``.
    model : torch.nn.Module
        The model (used to get predicted labels for the label argument).
    method : str
        Explainer name (for logging only).

    Returns
    -------
    attributions : torch.Tensor or None
        Attribution matrix of shape ``(n, d)``, or ``None`` on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                preds = torch.argmax(model(X.float()), dim=1)
            attrs = explainer.get_explanations(X.float(), label=preds)

        # Normalise to 2-D float tensor
        if not isinstance(attrs, torch.Tensor):
            attrs = torch.tensor(np.array(attrs), dtype=torch.float32)
        else:
            attrs = attrs.float()

        if attrs.ndim == 1:
            attrs = attrs.unsqueeze(0).expand(X.shape[0], -1)
        return attrs.detach()
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARNING] Explainer '{method}' failed during get_explanations: {exc}")
        return None


def run_all_explainers(
    model: torch.nn.Module,
    X_test: torch.FloatTensor,
    X_train: torch.FloatTensor,
    n_samples: int,
) -> Dict[str, torch.Tensor]:
    """Run every explainer on the first *n_samples* rows of ``X_test``.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model in eval mode.
    X_test : torch.FloatTensor
        Full test feature tensor of shape ``(N, d)``.
    X_train : torch.FloatTensor
        Full training feature tensor (background for LIME / IG).
    n_samples : int
        Number of samples to explain.

    Returns
    -------
    explanations : dict[str, torch.Tensor]
        Mapping from display explainer name → attribution matrix ``(n_samples, d)``.
        Explainers that fail are omitted from the dict.
    """
    set_seed(RANDOM_SEED)
    X = X_test[:n_samples]
    explanations: Dict[str, torch.Tensor] = {}

    for method in EXPLAINER_METHODS:
        display_name = EXPLAINER_DISPLAY[method]
        print(f"  → Running explainer: {display_name} (key='{method}') …")

        explainer = _build_explainer(method, model, X_train)
        if explainer is None:
            continue

        attrs = _get_attributions(explainer, X, model, method)
        if attrs is not None:
            explanations[display_name] = attrs
            print(f"    ✓ {display_name} — attribution shape: {tuple(attrs.shape)}")

    print(
        f"[run_all_explainers] Completed. "
        f"{len(explanations)}/{len(EXPLAINER_METHODS)} explainers succeeded."
    )
    return explanations
