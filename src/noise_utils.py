"""
noise_utils.py — Gaussian noise injection utilities for the Phase 3
noise-robustness extension of the OpenXAI replication project.

This module provides functions to add controlled Gaussian noise to tabular
feature tensors and to produce the full dictionary of noisy inputs needed
for the sigma-sweep degradation study.
"""

from __future__ import annotations

import numpy as np

from src.config import RANDOM_SEED, set_seed

# ─── Sigma levels for the degradation study ───────────────────────────────────
SIGMA_LEVELS: list[float] = [0.0, 0.1, 0.3, 0.5]


def add_gaussian_noise(
    X: np.ndarray,
    sigma: float,
    seed: int = RANDOM_SEED,
    clip: bool = True,
) -> np.ndarray:
    """Add independent Gaussian noise N(0, sigma²) to every element of X.

    When ``sigma == 0.0`` the function is a strict no-op and returns the
    original array *unchanged* (not re-allocated), so sigma=0.0 always
    reproduces Phase 2 results exactly.

    Args:
        X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            Must be a 2-D float array.
        sigma (float): Standard deviation of the Gaussian noise.  Set to 0.0
            for a clean (no-noise) baseline.
        seed (int): Random seed for reproducibility.  Default is the global
            ``RANDOM_SEED`` from ``src.config``.
        clip (bool): If ``True``, clip the noisy output element-wise to
            ``[0, 1]``, which is appropriate for features that have been
            min-max normalised.  Default ``True``.

    Returns:
        np.ndarray: Noisy (or original) array of the same shape as ``X``.
    """
    # ── Fast path: no noise requested ────────────────────────────────────────
    if sigma == 0.0:
        return X

    set_seed(seed)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape).astype(np.float32)
    X_noisy = X.astype(np.float32) + noise
    if clip:
        X_noisy = np.clip(X_noisy, 0.0, 1.0)
    return X_noisy


def noise_experiment_inputs(
    X_eval: np.ndarray,
    sigma_levels: list[float] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[float, np.ndarray]:
    """Generate a dictionary of noisy input arrays for every sigma level.

    Each sigma level uses a *different* random seed (``seed + int(sigma*100)``)
    so the noise realisations are statistically independent across levels.
    sigma=0.0 always returns the original ``X_eval`` unchanged.

    Args:
        X_eval (np.ndarray): Clean evaluation feature matrix,
            shape (n_samples, n_features).
        sigma_levels (list[float] | None): List of sigma values to sweep.
            Defaults to ``SIGMA_LEVELS = [0.0, 0.1, 0.3, 0.5]``.
        seed (int): Base random seed.  The seed for sigma ``s`` is
            ``seed + int(s * 100)``.  Default ``RANDOM_SEED``.

    Returns:
        dict[float, np.ndarray]: Mapping from sigma value to the corresponding
        noisy (or clean) array of shape (n_samples, n_features).
    """
    if sigma_levels is None:
        sigma_levels = SIGMA_LEVELS

    result: dict[float, np.ndarray] = {}
    for sigma in sigma_levels:
        # Derive an independent seed for each noise level
        level_seed = seed + int(sigma * 100)
        result[sigma] = add_gaussian_noise(X_eval, sigma=sigma, seed=level_seed, clip=True)

    return result
