"""
phase3_metrics.py — Orchestrates the two core Phase 3 studies:

1. ``degradation_study``     — sweeps sigma levels for all 7 original OpenXAI
                               explainers and records how all 5 metrics change.
2. ``before_after_comparison`` — compares SHAP vs SmoothSHAP and LIME vs
                                 SmoothLIME using RIS and PGF across sigma levels.

Both functions return tidy MultiIndex DataFrames indexed by (sigma, explainer)
and save per-sigma CSVs to ``results/tables/``.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from openxai import Explainer
from openxai.metrics import eval_pred_faithfulness, eval_relative_stability
from openxai.explainers.perturbation_methods import get_perturb_method
import openxai.experiment_utils as utils

from src.config import RANDOM_SEED, TABLES_DIR, ensure_dirs, set_seed
from src.noise_utils import SIGMA_LEVELS, noise_experiment_inputs
from src.run_explainers import EXPLAINER_METHODS, EXPLAINER_DISPLAY, _build_explainer
from src.smooth_explainers import run_smooth_explainers

# ─── Monkey-patch (same fix as in compute_metrics.py) ─────────────────────────
def _fixed_convert_k(k, n_feat):
    if k == -1:
        return n_feat
    if isinstance(k, int):
        return k
    if isinstance(k, float) and 0 < k < 1:
        return int(np.ceil(k * n_feat))
    return k

utils.convert_k_to_int = _fixed_convert_k
# ──────────────────────────────────────────────────────────────────────────────

TOP_K_FRACTION: float = 0.25
PERTURB_STD: float = 0.1


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_feature_metadata(data_name: str):
    """Load feature metadata for a dataset via the openxai DataLoader.

    Args:
        data_name (str): Dataset identifier (e.g. ``'adult'``, ``'compas'``).

    Returns:
        feature_metadata: Feature type metadata object from openxai.
    """
    from openxai.dataloader import ReturnLoaders
    trainloader, _ = ReturnLoaders(data_name, download=True, batch_size=256)
    return trainloader.dataset.feature_metadata


def _safe_scalar(result, metric_name: str, exp_name: str) -> float:
    """Extract a scalar from an openxai metric return value, returning NaN on error.

    Args:
        result: Return value of an openxai metric function (usually a tuple).
        metric_name (str): Metric name for warning messages.
        exp_name (str): Explainer name for warning messages.

    Returns:
        float: Extracted scalar or ``float('nan')`` on failure.
    """
    try:
        val = result[1] if isinstance(result, (tuple, list)) else result
        if isinstance(val, torch.Tensor):
            val = val.item()
        return float(val)
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARNING] Cannot extract {metric_name} for '{exp_name}': {exc}")
        return float("nan")


def _compute_faithfulness(
    attrs: torch.Tensor,
    X_noisy: torch.Tensor,
    model: torch.nn.Module,
    k: int,
    perturb_method,
    feature_metadata,
    exp_name: str,
) -> tuple[float, float]:
    """Compute PGF and PGU for a single (explainer, sigma) combination.

    Args:
        attrs (torch.Tensor): Attribution tensor, shape (n, d).
        X_noisy (torch.Tensor): Noisy feature matrix, shape (n, d).
        model (torch.nn.Module): Pretrained model.
        k (int): Number of top/bottom features to mask.
        perturb_method: openxai perturbation method object.
        feature_metadata: Feature type metadata from openxai.
        exp_name (str): Explainer name for logging.

    Returns:
        tuple[float, float]: (PGF score, PGU score).
    """
    set_seed(RANDOM_SEED)
    pgf = float("nan")
    pgu = float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = eval_pred_faithfulness(
                explanations=attrs, inputs=X_noisy, model=model, k=k,
                perturb_method=perturb_method, feature_metadata=feature_metadata,
                n_samples=100, invert=False, seed=RANDOM_SEED,
            )
        pgf = _safe_scalar(res, "PGF", exp_name)
    except Exception as exc:
        print(f"    [WARNING] PGF failed for '{exp_name}': {exc}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = eval_pred_faithfulness(
                explanations=attrs, inputs=X_noisy, model=model, k=k,
                perturb_method=perturb_method, feature_metadata=feature_metadata,
                n_samples=100, invert=True, seed=RANDOM_SEED,
            )
        pgu = _safe_scalar(res, "PGU", exp_name)
    except Exception as exc:
        print(f"    [WARNING] PGU failed for '{exp_name}': {exc}")

    return pgf, pgu


def _compute_stability(
    explainer_obj,
    X_noisy: torch.Tensor,
    model: torch.nn.Module,
    perturb_method,
    feature_metadata,
    exp_name: str,
) -> tuple[float, float, float]:
    """Compute RIS, RRS, ROS for a single (explainer, sigma) combination.

    Args:
        explainer_obj: Live openxai Explainer object (needed for stability metrics).
        X_noisy (torch.Tensor): Noisy feature matrix.
        model (torch.nn.Module): Pretrained model.
        perturb_method: openxai perturbation method object.
        feature_metadata: Feature type metadata from openxai.
        exp_name (str): Explainer name for logging.

    Returns:
        tuple[float, float, float]: (RIS, RRS, ROS) scores, any failing as NaN.
    """
    set_seed(RANDOM_SEED)
    scores = []
    for metric in ("RIS", "RRS", "ROS"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = eval_relative_stability(
                    explainer_obj, X_noisy, model, perturb_method,
                    feature_metadata, metric=metric,
                    n_samples=50, n_perturbations=10, seed=RANDOM_SEED,
                )
            scores.append(_safe_scalar(res, metric, exp_name))
        except Exception as exc:
            print(f"    [WARNING] {metric} failed for '{exp_name}': {exc}")
            scores.append(float("nan"))
    return tuple(scores)  # type: ignore[return-value]


# ─── Study 1: Degradation Study ───────────────────────────────────────────────

def degradation_study(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    X_train: np.ndarray,
    dataset_name: str,
    sigma_levels: List[float] = SIGMA_LEVELS,
    resume: bool = False,
) -> pd.DataFrame:
    """Sweep sigma levels, re-run all 7 explainers, record all 5 metrics.

    For every sigma in ``sigma_levels``:
      1. Add Gaussian noise to ``X_eval`` (sigma=0.0 returns X_eval unchanged).
      2. Run all 7 openxai explainers on the noisy inputs.
      3. Compute PGF, PGU, RIS, RRS, ROS.
      4. Save per-sigma CSV to ``results/tables/phase3_{dataset}_sigma{s}.csv``.
      5. Print progress lines: ``sigma=X | explainer=Y | RIS=Z``.

    Args:
        model (torch.nn.Module): Pretrained model in eval mode.
        X_eval (np.ndarray): Clean evaluation feature matrix,
            shape (n_samples, n_features).
        X_train (np.ndarray): Training feature matrix (background for LIME/IG).
        dataset_name (str): Dataset identifier (for file naming and logging).
        sigma_levels (list[float]): Noise levels to sweep.
            Defaults to ``SIGMA_LEVELS = [0.0, 0.1, 0.3, 0.5]``.

    Returns:
        pd.DataFrame: MultiIndex DataFrame indexed by (sigma, explainer) with
        columns ``['PGF', 'PGU', 'RIS', 'RRS', 'ROS']``.
        Any failed metric is stored as NaN.
    """
    ensure_dirs()
    os.makedirs(TABLES_DIR, exist_ok=True)

    feature_metadata = _get_feature_metadata(dataset_name)
    perturb_method = get_perturb_method(std=PERTURB_STD, data_name=dataset_name)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    n_features = X_eval.shape[1]
    k = max(1, int(TOP_K_FRACTION * n_features))

    # Prepare noisy inputs dict
    noisy_inputs = noise_experiment_inputs(X_eval, sigma_levels=sigma_levels)

    # Map display names back to openxai keys for rebuilding live explainer objects
    display_to_method = {v: mk for mk, v in EXPLAINER_DISPLAY.items()}

    all_records: list[dict] = []

    for sigma in sigma_levels:
        X_noisy_np = noisy_inputs[sigma]
        X_noisy_t = torch.tensor(X_noisy_np, dtype=torch.float32)
        sigma_records: list[dict] = []

        # ── Check for existing results ───────────────────────────────────────
        sigma_csv = TABLES_DIR / f"phase3_{dataset_name}_sigma{sigma}.csv"
        if resume and sigma_csv.exists():
            print(f"  [SKIP] sigma={sigma:.1f} | Already exists: {sigma_csv}")
            try:
                df_loaded = pd.read_csv(sigma_csv)
                # Convert back to records list format
                for _, row_data in df_loaded.iterrows():
                    record = row_data.to_dict()
                    record["sigma"] = sigma
                    all_records.append(record)
                continue
            except Exception as exc:
                print(f"  [WARNING] Failed to load existing CSV '{sigma_csv}': {exc}. Recomputing.")

        for method in EXPLAINER_METHODS:
            exp_name = EXPLAINER_DISPLAY[method]

            # ── Build explainer and get attributions ─────────────────────────
            explainer_obj = _build_explainer(method, model, X_train_t)
            if explainer_obj is None:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                with torch.no_grad():
                    preds = torch.argmax(model(X_noisy_t.float()), dim=1)
                attrs = explainer_obj.get_explanations(X_noisy_t.float(), label=preds)
                if not isinstance(attrs, torch.Tensor):
                    attrs = torch.tensor(np.array(attrs), dtype=torch.float32)
                else:
                    attrs = attrs.float()
                if attrs.ndim == 1:
                    attrs = attrs.unsqueeze(0).expand(X_noisy_t.shape[0], -1)
                attrs = attrs.detach()
            except Exception as exc:
                print(f"  [WARNING] sigma={sigma} | explainer={exp_name} | attribution fail: {exc}")
                continue

            # ── Faithfulness ─────────────────────────────────────────────────
            pgf, pgu = _compute_faithfulness(
                attrs, X_noisy_t, model, k, perturb_method, feature_metadata, exp_name
            )

            # ── Stability (rebuild live explainer for the noisy X) ───────────
            # Stability metrics always require the live explainer object.
            # We rebuild using the noisy X_train substitution is not needed;
            # openxai's eval_relative_stability perturbation starts from X_noisy.
            live_exp = _build_explainer(method, model, X_train_t)
            ris, rrs, ros = _compute_stability(
                live_exp, X_noisy_t, model, perturb_method, feature_metadata, exp_name
            )

            row = {
                "sigma": sigma,
                "explainer": exp_name,
                "PGF": pgf,
                "PGU": pgu,
                "RIS": ris,
                "RRS": rrs,
                "ROS": ros,
            }
            all_records.append(row)
            sigma_records.append(row)
            print(
                f"  sigma={sigma:.1f} | explainer={exp_name:<8} "
                f"| PGF={pgf:.4f} | PGU={pgu:.4f} | RIS={ris:.4f}"
            )

        # ── Save per-sigma CSV ────────────────────────────────────────────────
        if sigma_records:
            df_sigma = pd.DataFrame(sigma_records).set_index("explainer")
            df_sigma = df_sigma.drop(columns=["sigma"])
            csv_path = TABLES_DIR / f"phase3_{dataset_name}_sigma{sigma}.csv"
            os.makedirs(TABLES_DIR, exist_ok=True)
            df_sigma.to_csv(csv_path)
            print(f"  [save] CSV → {csv_path}")

    # ── Build MultiIndex DataFrame ────────────────────────────────────────────
    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.set_index(["sigma", "explainer"])
    df = df[["PGF", "PGU", "RIS", "RRS", "ROS"]]
    df.index.names = ["sigma", "explainer"]

    # Save full summary CSV
    full_csv = TABLES_DIR / f"phase3_{dataset_name}_degradation_full.csv"
    df.to_csv(full_csv)
    print(f"\n[degradation_study] Full results → {full_csv}")
    return df


# ─── Study 2: Before vs After Comparison ─────────────────────────────────────

def before_after_comparison(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    X_train: np.ndarray,
    dataset_name: str,
    sigma_levels: List[float] = SIGMA_LEVELS,
    K: int = 20,
    seed: int = RANDOM_SEED,
    resume: bool = False,
) -> pd.DataFrame:
    """Compare vanilla SHAP/LIME against SmoothSHAP/SmoothLIME across sigma levels.

    For each sigma in ``sigma_levels``:
      1. Run vanilla SHAP and LIME on the noisy inputs.
      2. Run SmoothSHAP and SmoothLIME on the noisy inputs (K copies averaged).
      3. Compute RIS and PGF for all four.
      4. Compute delta columns (absolute improvement of smooth over vanilla).

    Args:
        model (torch.nn.Module): Pretrained model in eval mode.
        X_eval (np.ndarray): Clean evaluation feature matrix.
        X_train (np.ndarray): Training feature matrix (background for LIME).
        dataset_name (str): Dataset identifier (for file naming and logging).
        sigma_levels (list[float]): Noise levels to sweep.
        K (int): Number of noisy copies for SmoothExplainer averaging.
        seed (int): Base random seed.  Default ``RANDOM_SEED``.

    Returns:
        pd.DataFrame: MultiIndex DataFrame indexed by (sigma, explainer).
        Columns: ``['RIS', 'PGF', 'delta_RIS', 'delta_PGF']``.
        ``delta_RIS`` = vanilla_RIS − smooth_RIS  (positive = improvement).
        ``delta_PGF`` = smooth_PGF − vanilla_PGF  (positive = improvement).
    """
    ensure_dirs()
    os.makedirs(TABLES_DIR, exist_ok=True)

    feature_metadata = _get_feature_metadata(dataset_name)
    perturb_method = get_perturb_method(std=PERTURB_STD, data_name=dataset_name)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    n_features = X_eval.shape[1]
    k = max(1, int(TOP_K_FRACTION * n_features))

    noisy_inputs = noise_experiment_inputs(X_eval, sigma_levels=sigma_levels)

    all_records: list[dict] = []

    for sigma in sigma_levels:
        X_noisy_np = noisy_inputs[sigma]
        X_noisy_t = torch.tensor(X_noisy_np, dtype=torch.float32)

        # ── Check for existing results ───────────────────────────────────────
        ba_sigma_csv = TABLES_DIR / f"phase3_{dataset_name}_ba_sigma{sigma}.csv"
        if resume and ba_sigma_csv.exists():
            print(f"  [SKIP] sigma={sigma:.1f} | Already exists: {ba_sigma_csv}")
            try:
                df_loaded = pd.read_csv(ba_sigma_csv)
                # If loaded df has MultiIndex after reading, handle it.
                # Usually pd.read_csv on a tidy CSV is fine.
                for _, row_data in df_loaded.iterrows():
                    record = row_data.to_dict()
                    record["sigma"] = sigma
                    all_records.append(record)
                continue
            except Exception as exc:
                print(f"  [WARNING] Failed to load existing CSV '{ba_sigma_csv}': {exc}. Recomputing.")

        sigma_ba_records: list[dict] = []

        # ── Vanilla SHAP and LIME ─────────────────────────────────────────────
        vanilla_scores: dict[str, dict] = {}
        for method, display in [("shap", "shap"), ("lime", "lime")]:
            exp_obj = _build_explainer(method, model, X_train_t)
            if exp_obj is None:
                vanilla_scores[display] = {"RIS": float("nan"), "PGF": float("nan")}
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                with torch.no_grad():
                    preds = torch.argmax(model(X_noisy_t.float()), dim=1)
                attrs = exp_obj.get_explanations(X_noisy_t.float(), label=preds)
                if not isinstance(attrs, torch.Tensor):
                    attrs = torch.tensor(np.array(attrs), dtype=torch.float32)
                attrs = attrs.float().detach()
                if attrs.ndim == 1:
                    attrs = attrs.unsqueeze(0).expand(X_noisy_t.shape[0], -1)
            except Exception as exc:
                print(f"  [WARNING] vanilla {display} attribution fail at sigma={sigma}: {exc}")
                vanilla_scores[display] = {"RIS": float("nan"), "PGF": float("nan")}
                continue

            pgf, _ = _compute_faithfulness(
                attrs, X_noisy_t, model, k, perturb_method, feature_metadata, display
            )
            ris, _, _ = _compute_stability(
                _build_explainer(method, model, X_train_t),
                X_noisy_t, model, perturb_method, feature_metadata, display
            )
            vanilla_scores[display] = {"RIS": ris, "PGF": pgf}
            print(f"  sigma={sigma:.1f} | {display:<10} | RIS={ris:.4f} | PGF={pgf:.4f}")

        # ── SmoothSHAP and SmoothLIME ─────────────────────────────────────────
        smooth_attrs = run_smooth_explainers(
            model, X_noisy_np, X_train, sigma=sigma, K=K, seed=seed
        )

        smooth_scores: dict[str, dict] = {}
        key_map = {"smooth_shap": ("shap", "smooth_shap"), "smooth_lime": ("lime", "smooth_lime")}
        for attr_key, (vanilla_key, display) in key_map.items():
            if attr_key not in smooth_attrs:
                smooth_scores[display] = {"RIS": float("nan"), "PGF": float("nan")}
                continue

            s_attrs = torch.tensor(smooth_attrs[attr_key], dtype=torch.float32)

            # For stability we need a live explainer object; use vanilla object
            # but pass smooth attributions to faithfulness metrics
            pgf, _ = _compute_faithfulness(
                s_attrs, X_noisy_t, model, k, perturb_method, feature_metadata, display
            )

            # Re-build vanilla explainer as the live object for stability
            live_exp = _build_explainer(vanilla_key, model, X_train_t)
            ris, _, _ = _compute_stability(
                live_exp, X_noisy_t, model, perturb_method, feature_metadata, display
            )
            smooth_scores[display] = {"RIS": ris, "PGF": pgf}
            print(f"  sigma={sigma:.1f} | {display:<15} | RIS={ris:.4f} | PGF={pgf:.4f}")

        # ── Assemble rows with delta columns ──────────────────────────────────
        pair_map = [
            ("shap",  "smooth_shap"),
            ("lime",  "smooth_lime"),
        ]
        for vanilla_key, smooth_key in pair_map:
            v = vanilla_scores.get(vanilla_key, {})
            s = smooth_scores.get(smooth_key, {})

            v_ris = v.get("RIS", float("nan"))
            s_ris = s.get("RIS", float("nan"))
            v_pgf = v.get("PGF", float("nan"))
            s_pgf = s.get("PGF", float("nan"))

            # delta_RIS: improvement = vanilla_RIS - smooth_RIS (lower is better)
            delta_ris = (v_ris - s_ris) if not (np.isnan(v_ris) or np.isnan(s_ris)) else float("nan")
            # delta_PGF: improvement = smooth_PGF - vanilla_PGF (higher is better)
            delta_pgf = (s_pgf - v_pgf) if not (np.isnan(s_pgf) or np.isnan(v_pgf)) else float("nan")

            for name, ris_val, pgf_val, d_ris, d_pgf in [
                (vanilla_key, v_ris, v_pgf, 0.0, 0.0),
                (smooth_key,  s_ris, s_pgf, delta_ris, delta_pgf),
            ]:
                all_records.append({
                    "sigma": sigma,
                    "explainer": name,
                    "RIS": ris_val,
                    "PGF": pgf_val,
                    "delta_RIS": d_ris,
                    "delta_PGF": d_pgf,
                })
                sigma_ba_records.append({
                    "explainer": name,
                    "RIS": ris_val,
                    "PGF": pgf_val,
                    "delta_RIS": d_ris,
                    "delta_PGF": d_pgf,
                })

        # Save per-sigma BA CSV
        if sigma_ba_records:
            df_ba_sigma = pd.DataFrame(sigma_ba_records).set_index("explainer")
            df_ba_sigma.to_csv(ba_sigma_csv)
            print(f"  [save] BA CSV → {ba_sigma_csv}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.set_index(["sigma", "explainer"])
    df = df[["RIS", "PGF", "delta_RIS", "delta_PGF"]]

    csv_path = TABLES_DIR / f"phase3_{dataset_name}_before_after.csv"
    df.to_csv(csv_path)
    print(f"\n[before_after_comparison] Results → {csv_path}")
    return df
