"""
smooth_explainers.py — Noise-aware SmoothSHAP and SmoothLIME explainers for
the Phase 3 extension of the OpenXAI replication project.

Formal definition
-----------------
Given an explainer E and K noisy copies of input x:

    x_k = x + epsilon_k,  epsilon_k ~ N(0, sigma² I),  k = 1 … K

    SmoothExplainer(x) = (1/K') * sum_{k=1}^{K'} E(x_k)

where K' ≤ K is the number of *successful* runs (individual failures are
suppressed with a warning).  If K' < K/2 a warning is emitted but the
partial mean is still returned so the outer pipeline never crashes.

Both SmoothSHAP and SmoothLIME are exposed via the convenience function
``run_smooth_explainers`` which matches the return signature used by
``run_all_explainers`` in ``run_explainers.py``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import torch

from openxai import Explainer
import openxai.experiment_utils as utils

from src.config import RANDOM_SEED, set_seed
from src.noise_utils import add_gaussian_noise


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_param_dict_smooth(method: str, X_train: torch.FloatTensor) -> dict:
    """Build an openxai-compatible param_dict for SHAP or LIME.

    Args:
        method (str): Explainer method key: ``'shap'`` or ``'lime'``.
        X_train (torch.FloatTensor): Full training feature tensor used as the
            background / reference distribution.

    Returns:
        dict: Ready-to-use parameter dictionary for ``openxai.Explainer``.
    """
    if method == "lime":
        return utils.fill_param_dict("lime", {"n_samples": 100}, X_train)
    elif method == "shap":
        return {"n_samples": 100}
    return {}


# ─── SmoothExplainer class ────────────────────────────────────────────────────

class SmoothExplainer:
    """Noise-averaged wrapper around an ``openxai.Explainer`` (SHAP or LIME).

    For each input x the explainer is run K times on noisy copies
    ``x_k = x + N(0, sigma²)``.  The final attribution is the mean of all
    successful runs.

    The class intentionally mirrors the ``openxai.Explainer`` interface
    (specifically ``get_explanations(X, label=...)``), making it a drop-in
    replacement that requires *no changes* to ``compute_metrics.py``.

    Args:
        base_method (str): ``'shap'`` or ``'lime'`` — the underlying explainer.
        model (torch.nn.Module): Pretrained model in eval mode.
        dataset_tensor (torch.Tensor): Full training feature tensor,
            used as background for LIME and IG.
        K (int): Number of noisy copies to average over.  Default 20.
        sigma (float): Standard deviation of the added noise.  Default 0.1.
        seed (int): Base random seed; each of the K copies uses
            ``seed + k`` for statistical independence.  Default 42.

    Raises:
        ValueError: If ``base_method`` is not ``'shap'`` or ``'lime'``.
    """

    def __init__(
        self,
        base_method: str,
        model: torch.nn.Module,
        dataset_tensor: torch.Tensor,
        K: int = 20,
        sigma: float = 0.1,
        seed: int = RANDOM_SEED,
    ) -> None:
        if base_method not in ("shap", "lime"):
            raise ValueError(
                f"SmoothExplainer only supports 'shap' or 'lime', got '{base_method}'."
            )
        self.base_method = base_method
        self.model = model
        self.dataset_tensor = dataset_tensor
        self.K = K
        self.sigma = sigma
        self.seed = seed

        # Build the underlying openxai explainer once (background data fixed)
        set_seed(seed)
        param_dict = _build_param_dict_smooth(base_method, dataset_tensor)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._explainer = Explainer(
                    method=base_method,
                    model=model,
                    param_dict=param_dict,
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[SmoothExplainer] Failed to build base '{base_method}' explainer: {exc}"
            ) from exc

    # ── Public interface matching openxai.Explainer ───────────────────────────

    def get_explanations(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return noise-averaged feature attributions for input batch ``x``.

        Generates K perturbed copies of ``x``, calls the base explainer on
        each, and returns the element-wise mean over successful runs.

        Args:
            x (torch.Tensor): Input feature batch, shape (n_samples, n_features).
            label (torch.Tensor | None): Predicted-class labels of shape
                (n_samples,).  If ``None``, labels are inferred from the model.

        Returns:
            torch.Tensor: Averaged attribution matrix, shape
            (n_samples, n_features), dtype float32.

        Warns:
            RuntimeWarning: If fewer than K/2 of the K runs succeed.

        Notes:
            * Failed individual noisy runs are silently skipped (counted).
            * The function never raises an exception to the outer pipeline.
        """
        set_seed(self.seed)

        x_np = x.detach().cpu().numpy().astype(np.float32)

        # Infer labels once from the clean input
        if label is None:
            with torch.no_grad():
                label = torch.argmax(self.model(x.float()), dim=1)

        accumulated: Optional[np.ndarray] = None
        n_success = 0

        for k in range(self.K):
            copy_seed = self.seed + k + 1  # distinct per copy

            # Add noise (sigma=0.0 returns original array unchanged)
            x_k_np = add_gaussian_noise(x_np, sigma=self.sigma, seed=copy_seed, clip=True)
            x_k = torch.tensor(x_k_np, dtype=torch.float32)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    attrs_k = self._explainer.get_explanations(x_k, label=label)

                # Normalise to numpy float32
                if isinstance(attrs_k, torch.Tensor):
                    attrs_np = attrs_k.detach().cpu().numpy().astype(np.float32)
                else:
                    attrs_np = np.array(attrs_k, dtype=np.float32)

                if attrs_np.ndim == 1:
                    attrs_np = np.tile(attrs_np, (x_np.shape[0], 1))

                if accumulated is None:
                    accumulated = attrs_np
                else:
                    accumulated = accumulated + attrs_np
                n_success += 1

            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"[SmoothExplainer / {self.base_method}] run {k+1}/{self.K} failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # ── Guard: at least K/2 runs must succeed ────────────────────────────
        if n_success < self.K / 2:
            warnings.warn(
                f"[SmoothExplainer / {self.base_method}] Only {n_success}/{self.K} runs "
                "succeeded (< K/2). Result quality may be degraded.",
                RuntimeWarning,
                stacklevel=2,
            )

        if accumulated is None or n_success == 0:
            warnings.warn(
                f"[SmoothExplainer / {self.base_method}] ALL runs failed. "
                "Returning zero attributions.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.zeros(x.shape, dtype=torch.float32)

        mean_attrs = accumulated / n_success
        return torch.tensor(mean_attrs, dtype=torch.float32)


# ─── Convenience runner ───────────────────────────────────────────────────────

def run_smooth_explainers(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    X_train: np.ndarray,
    sigma: float,
    K: int = 20,
    seed: int = RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """Run SmoothSHAP and SmoothLIME on ``X_eval`` and return attribution arrays.

    This function mirrors the return format of ``run_all_explainers`` so that
    results can be fed directly into ``compute_metrics_for_dataset`` without
    any changes to that module.

    Args:
        model (torch.nn.Module): Pretrained model in eval mode.
        X_eval (np.ndarray): Evaluation feature matrix,
            shape (n_samples, n_features).  May already be a noisy version.
        X_train (np.ndarray): Training feature matrix used as background
            for LIME.
        sigma (float): Noise level for the smooth explainers.
        K (int): Number of noisy copies to average.  Default 20.
        seed (int): Base random seed.  Default ``RANDOM_SEED``.

    Returns:
        dict[str, np.ndarray]: ``{'smooth_shap': array, 'smooth_lime': array}``
        where each array has shape (n_samples, n_features).  If a method fails
        completely, it is omitted from the dict.
    """
    set_seed(seed)

    X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)

    results: Dict[str, np.ndarray] = {}

    for method, key in [("shap", "smooth_shap"), ("lime", "smooth_lime")]:
        print(f"  → Running {key} (K={K}, sigma={sigma}) …")
        try:
            explainer = SmoothExplainer(
                base_method=method,
                model=model,
                dataset_tensor=X_train_t,
                K=K,
                sigma=sigma,
                seed=seed,
            )
            with torch.no_grad():
                preds = torch.argmax(model(X_eval_t), dim=1)
            attrs_t = explainer.get_explanations(X_eval_t, label=preds)
            results[key] = attrs_t.cpu().numpy()
            print(f"    ✓ {key} — shape: {results[key].shape}")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[run_smooth_explainers] {key} failed entirely: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return results
