"""
smooth_explainers.py — Noise-aware SmoothSHAP and SmoothLIME explainers for
the Phase 3 extension of the OpenXAI replication project.

FIX LOG (Bug 3):
----------------
Original bug: run_smooth_explainers() was receiving X_noisy_np (already noisy)
and then SmoothExplainer.get_explanations() was adding sigma noise AGAIN on top,
effectively doubling the noise level (e.g. sigma=0.3 became ~sigma=0.6).

Fix: SmoothExplainer now accepts an explicit `internal_sigma` parameter that
controls the noise added during averaging. The outer pipeline always passes
the CLEAN X_eval to run_smooth_explainers(), and internal_sigma is set to
the desired sigma level. This gives correct, single-level noise averaging.

Formal definition
-----------------
Given an explainer E, clean input x, and target noise level sigma:

    x_k = x + epsilon_k,  epsilon_k ~ N(0, sigma² I),  k = 1 … K

    SmoothExplainer(x) = (1/K') * sum_{k=1}^{K'} E(x_k)

where K' <= K is the number of successful runs.
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
    if method == "lime":
        return utils.fill_param_dict("lime", {"n_samples": 100}, X_train)
    elif method == "shap":
        return {"n_samples": 100}
    return {}


# ─── SmoothExplainer class ────────────────────────────────────────────────────

class SmoothExplainer:
    """Noise-averaged wrapper around an openxai.Explainer (SHAP or LIME).

    IMPORTANT: This class expects CLEAN inputs x. It generates noisy copies
    internally using internal_sigma. Do NOT pre-noise the inputs before
    passing to get_explanations() — that was the original Bug 3.

    Args:
        base_method (str): 'shap' or 'lime'.
        model (torch.nn.Module): Pretrained model in eval mode.
        dataset_tensor (torch.Tensor): Full training feature tensor.
        K (int): Number of noisy copies to average over. Default 20.
        internal_sigma (float): Std dev of noise added to each copy. This
            should match the sigma level of the experiment. Default 0.1.
        seed (int): Base random seed. Default RANDOM_SEED.
    """

    def __init__(
        self,
        base_method: str,
        model: torch.nn.Module,
        dataset_tensor: torch.Tensor,
        K: int = 20,
        internal_sigma: float = 0.1,
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
        self.internal_sigma = internal_sigma  # FIX: renamed from sigma for clarity
        self.seed = seed

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
        except Exception as exc:
            raise RuntimeError(
                f"[SmoothExplainer] Failed to build base '{base_method}' explainer: {exc}"
            ) from exc

    def get_explanations(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return noise-averaged attributions for CLEAN input batch x.

        Generates K noisy copies x_k = x + N(0, internal_sigma²),
        runs the base explainer on each, and returns the element-wise mean.

        Args:
            x (torch.Tensor): CLEAN input features, shape (n_samples, n_features).
                Do NOT pass pre-noised inputs here.
            label (torch.Tensor | None): Predicted labels. Inferred if None.

        Returns:
            torch.Tensor: Averaged attributions, shape (n_samples, n_features).
        """
        set_seed(self.seed)
        x_np = x.detach().cpu().numpy().astype(np.float32)

        if label is None:
            with torch.no_grad():
                label = torch.argmax(self.model(x.float()), dim=1)

        accumulated: Optional[np.ndarray] = None
        n_success = 0

        for k in range(self.K):
            copy_seed = self.seed + k + 1

            # FIX: Always noise from the CLEAN x_np using internal_sigma
            x_k_np = add_gaussian_noise(x_np, sigma=self.internal_sigma, seed=copy_seed, clip=True)
            x_k = torch.tensor(x_k_np, dtype=torch.float32)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    attrs_k = self._explainer.get_explanations(x_k, label=label)

                if isinstance(attrs_k, torch.Tensor):
                    attrs_np = attrs_k.detach().cpu().numpy().astype(np.float32)
                else:
                    attrs_np = np.array(attrs_k, dtype=np.float32)

                if attrs_np.ndim == 1:
                    attrs_np = np.tile(attrs_np, (x_np.shape[0], 1))

                accumulated = attrs_np if accumulated is None else accumulated + attrs_np
                n_success += 1

            except Exception as exc:
                warnings.warn(
                    f"[SmoothExplainer / {self.base_method}] run {k+1}/{self.K} failed: {exc}",
                    RuntimeWarning, stacklevel=2,
                )

        if n_success < self.K / 2:
            warnings.warn(
                f"[SmoothExplainer / {self.base_method}] Only {n_success}/{self.K} runs "
                "succeeded (< K/2). Result quality may be degraded.",
                RuntimeWarning, stacklevel=2,
            )

        if accumulated is None or n_success == 0:
            warnings.warn(
                f"[SmoothExplainer / {self.base_method}] ALL runs failed. "
                "Returning zero attributions.",
                RuntimeWarning, stacklevel=2,
            )
            return torch.zeros(x.shape, dtype=torch.float32)

        mean_attrs = accumulated / n_success
        return torch.tensor(mean_attrs, dtype=torch.float32)


# ─── Convenience runner ───────────────────────────────────────────────────────

def run_smooth_explainers(
    model: torch.nn.Module,
    X_clean: np.ndarray,          # FIX: renamed from X_eval, must be CLEAN
    X_train: np.ndarray,
    sigma: float,                  # FIX: this now controls internal_sigma only
    K: int = 20,
    seed: int = RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """Run SmoothSHAP and SmoothLIME on CLEAN X_clean with internal noise sigma.

    FIX: Previously this function received X_noisy and added more noise inside,
    doubling the effective noise level. Now it always receives the clean inputs
    and controls noise level solely through internal_sigma.

    Args:
        model: Pretrained model in eval mode.
        X_clean (np.ndarray): CLEAN evaluation features, shape (n, d).
            This must be the original clean data, NOT pre-noised.
        X_train (np.ndarray): Training features for LIME background.
        sigma (float): Noise level for internal averaging (single source of noise).
        K (int): Number of noisy copies to average. Default 20.
        seed (int): Base random seed. Default RANDOM_SEED.

    Returns:
        dict[str, np.ndarray]: {'smooth_shap': array, 'smooth_lime': array}
    """
    set_seed(seed)

    X_clean_t = torch.tensor(X_clean, dtype=torch.float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)

    results: Dict[str, np.ndarray] = {}

    for method, key in [("shap", "smooth_shap"), ("lime", "smooth_lime")]:
        print(f"  → Running {key} (K={K}, internal_sigma={sigma}) on CLEAN inputs …")
        try:
            explainer = SmoothExplainer(
                base_method=method,
                model=model,
                dataset_tensor=X_train_t,
                K=K,
                internal_sigma=sigma,  # FIX: clean separation of concerns
                seed=seed,
            )
            with torch.no_grad():
                preds = torch.argmax(model(X_clean_t), dim=1)
            # FIX: pass CLEAN inputs — noise is handled internally
            attrs_t = explainer.get_explanations(X_clean_t, label=preds)
            results[key] = attrs_t.cpu().numpy()
            print(f"    ✓ {key} — shape: {results[key].shape}")
        except Exception as exc:
            warnings.warn(
                f"[run_smooth_explainers] {key} failed entirely: {exc}",
                RuntimeWarning, stacklevel=2,
            )

    return results