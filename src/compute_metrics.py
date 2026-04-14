"""
compute_metrics.py — Evaluates explainer attributions using the five OpenXAI
benchmark metrics.

Metrics
-------
* Faithfulness: PGF (↑ higher is better), PGU (↓ lower is better)
* Stability:    RIS, RRS, ROS (↓ lower is better)

Each metric function is wrapped in try/except so that a failing metric for a
given explainer inserts NaN rather than crashing the run.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd
import torch

from openxai.metrics import PGF, PGU, RIS, RRS, ROS

from src.config import METRICS, set_seed, RANDOM_SEED


def _safe_metric(
    metric_fn,
    *args,
    metric_name: str,
    explainer_name: str,
    **kwargs,
) -> float:
    """Call *metric_fn* with *args*/*kwargs* and return the scalar result.

    Returns ``float('nan')`` and prints a warning on any exception.

    Parameters
    ----------
    metric_fn : callable
        An OpenXAI metric callable.
    *args
        Positional arguments forwarded to ``metric_fn``.
    metric_name : str
        Human-readable metric identifier (used in warning messages).
    explainer_name : str
        Human-readable explainer identifier (used in warning messages).
    **kwargs
        Keyword arguments forwarded to ``metric_fn``.

    Returns
    -------
    value : float
        Scalar metric value, or ``nan`` on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metric_fn(*args, **kwargs)

        # Unwrap tensors / arrays to a plain Python float
        if isinstance(result, torch.Tensor):
            result = result.mean().item()
        elif isinstance(result, np.ndarray):
            result = float(result.mean())
        else:
            result = float(result)
        return result
    except Exception as exc:  # noqa: BLE001
        print(
            f"    [WARNING] Metric '{metric_name}' failed for "
            f"explainer '{explainer_name}': {exc}"
        )
        return float("nan")


def compute_metrics_for_dataset(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    explanations: Dict[str, torch.Tensor],
    data_name: str,
) -> pd.DataFrame:
    """Compute all five benchmark metrics for every explainer.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model in eval mode.
    X_test : torch.Tensor
        Feature tensor used for evaluation, shape ``(n, d)``.
    y_test : torch.Tensor
        Label tensor, shape ``(n,)``.
    explanations : dict[str, torch.Tensor]
        Mapping from explainer name to attribution matrix ``(n, d)``.
    data_name : str
        Dataset name (used only for logging).

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame of shape ``(n_explainers, 5)`` indexed by explainer name,
        with one column per metric (PGF, PGU, RIS, RRS, ROS).
    """
    set_seed(RANDOM_SEED)
    records: Dict[str, Dict[str, float]] = {}

    print(f"\n[compute_metrics] Dataset: '{data_name}'")

    for exp_name, attrs in explanations.items():
        print(f"  Evaluating '{exp_name}' …")
        row: Dict[str, float] = {}

        # ── Faithfulness ────────────────────────────────────────────────────
        row["PGF"] = _safe_metric(
            PGF,
            model,
            X_test,
            attrs,
            metric_name="PGF",
            explainer_name=exp_name,
        )
        row["PGU"] = _safe_metric(
            PGU,
            model,
            X_test,
            attrs,
            metric_name="PGU",
            explainer_name=exp_name,
        )

        # ── Stability ───────────────────────────────────────────────────────
        row["RIS"] = _safe_metric(
            RIS,
            model,
            X_test,
            attrs,
            metric_name="RIS",
            explainer_name=exp_name,
        )
        row["RRS"] = _safe_metric(
            RRS,
            model,
            X_test,
            attrs,
            metric_name="RRS",
            explainer_name=exp_name,
        )
        row["ROS"] = _safe_metric(
            ROS,
            model,
            X_test,
            attrs,
            metric_name="ROS",
            explainer_name=exp_name,
        )

        records[exp_name] = row

    df = pd.DataFrame.from_dict(records, orient="index", columns=METRICS)
    df.index.name = "Explainer"
    print(f"[compute_metrics] Done for '{data_name}'.\n{df.to_string()}\n")
    return df
