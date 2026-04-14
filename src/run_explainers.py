"""
run_explainers.py — Generates feature-attribution explanations for a batch of
test samples using all 7 explainers supported by OpenXAI.

Responsibilities
----------------
* Iterate over every explainer name defined in config.EXPLAINERS.
* Wrap each call in try/except so a failing explainer is skipped gracefully.
* Return a dict mapping explainer name → attribution tensor.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import numpy as np

import openxai
from openxai import Explainer

from src.config import EXPLAINER_PARAM_DICTS, EXPLAINERS, set_seed, RANDOM_SEED


def _build_explainer(
    method: str,
    model: torch.nn.Module,
    dataset_tensor: torch.Tensor,
) -> Optional[Explainer]:
    """Instantiate a single OpenXAI Explainer.

    Parameters
    ----------
    method : str
        Explainer identifier (e.g. ``'lime'``, ``'shap'``, ``'grad'``).
    model : torch.nn.Module
        The model to explain.
    dataset_tensor : torch.Tensor
        Background / reference dataset used by some explainers (e.g. SHAP).

    Returns
    -------
    explainer : Explainer or None
        Instantiated explainer, or ``None`` if construction failed.
    """
    extra_kwargs = EXPLAINER_PARAM_DICTS.get(method, {})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = Explainer(
                method=method,
                model=model,
                dataset_tensor=dataset_tensor,
                **extra_kwargs,
            )
        return explainer
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARNING] Could not instantiate explainer '{method}': {exc}")
        return None


def _get_attributions(
    explainer: Explainer,
    X: torch.Tensor,
    method: str,
) -> Optional[torch.Tensor]:
    """Call the explainer and normalise the output shape.

    Parameters
    ----------
    explainer : Explainer
        Instantiated OpenXAI Explainer.
    X : torch.Tensor
        Input batch of shape ``(n, d)``.
    method : str
        Explainer name (for logging).

    Returns
    -------
    attributions : torch.Tensor or None
        Attribution matrix of shape ``(n, d)``, or ``None`` on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            attrs = explainer.get_explanations(X)

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
    X_test: torch.Tensor,
    n_samples: int,
) -> Dict[str, torch.Tensor]:
    """Run every explainer in ``config.EXPLAINERS`` on the first *n_samples*
    rows of ``X_test``.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model in eval mode.
    X_test : torch.Tensor
        Full test feature tensor of shape ``(N, d)``.
    n_samples : int
        Number of samples to explain.

    Returns
    -------
    explanations : dict[str, torch.Tensor]
        Mapping from explainer name to attribution matrix ``(n_samples, d)``.
        Explainers that fail are omitted from the dict.
    """
    set_seed(RANDOM_SEED)
    X = X_test[:n_samples]
    explanations: Dict[str, torch.Tensor] = {}

    for method in EXPLAINERS:
        print(f"  → Running explainer: {method} …")
        explainer = _build_explainer(method, model, X)
        if explainer is None:
            continue

        attrs = _get_attributions(explainer, X, method)
        if attrs is not None:
            explanations[method] = attrs
            print(f"    ✓ {method} — attribution shape: {tuple(attrs.shape)}")

    print(
        f"[run_all_explainers] Completed. "
        f"{len(explanations)}/{len(EXPLAINERS)} explainers succeeded."
    )
    return explanations
