"""
phase3_metrics.py — Orchestrates the two core Phase 3 studies.

FIX LOG:
--------
Bug 1 (Stability for smooth explainers used vanilla object):
    Before: live_exp = _build_explainer(vanilla_key, ...) → RIS identical for
            SHAP and SmoothSHAP because eval_relative_stability calls
            explainer_obj.get_explanations() internally.
    Fix:    Build a SmoothExplainer object and pass it as the live explainer
            for stability computation of smooth variants. SmoothExplainer
            implements get_explanations() so it's a drop-in.

Bug 2 (Faithfulness evaluated on noisy inputs):
    Before: _compute_faithfulness(attrs, X_noisy_t, ...) — PGF perturbs
            already-noisy inputs, producing unpredictable prediction gaps.
    Fix:    _compute_faithfulness always receives X_clean_t as the input
            tensor. The explanations (attrs) still come from noisy inputs,
            but the evaluation ground is always the clean data.

Bug 3 (Double noise in run_smooth_explainers):
    Before: X_noisy_np was passed to run_smooth_explainers, then noise was
            added again inside SmoothExplainer, doubling sigma.
    Fix:    X_clean (original data) is always passed to run_smooth_explainers.
            Noise is applied only once, internally, via internal_sigma=sigma.
"""

from __future__ import annotations

import os
import warnings
from typing import List

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
from src.smooth_explainers import run_smooth_explainers, SmoothExplainer

# ─── Monkey-patch ─────────────────────────────────────────────────────────────
def _fixed_convert_k(k, n_feat):
    if k == -1:
        return n_feat
    if isinstance(k, int):
        return k
    if isinstance(k, float) and 0 < k < 1:
        return int(np.ceil(k * n_feat))
    return k

utils.convert_k_to_int = _fixed_convert_k
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_FRACTION: float = 0.25
PERTURB_STD: float = 0.1


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_feature_metadata(data_name: str):
    from openxai.dataloader import ReturnLoaders
    trainloader, _ = ReturnLoaders(data_name, download=True, batch_size=256)
    return trainloader.dataset.feature_metadata


def _safe_scalar(result, metric_name: str, exp_name: str) -> float:
    try:
        val = result[1] if isinstance(result, (tuple, list)) else result
        if isinstance(val, torch.Tensor):
            val = val.item()
        return float(val)
    except Exception as exc:
        print(f"    [WARNING] Cannot extract {metric_name} for '{exp_name}': {exc}")
        return float("nan")


def _compute_faithfulness(
    attrs: torch.Tensor,
    X_clean: torch.Tensor,       # FIX Bug 2: always clean inputs
    model: torch.nn.Module,
    k: int,
    perturb_method,
    feature_metadata,
    exp_name: str,
) -> tuple[float, float]:
    """Compute PGF and PGU.

    FIX Bug 2: X_clean is always the original clean data regardless of
    which sigma level produced the attributions. This keeps the evaluation
    ground stable and comparable across sigma levels.
    """
    set_seed(RANDOM_SEED)
    pgf = float("nan")
    pgu = float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = eval_pred_faithfulness(
                explanations=attrs, inputs=X_clean, model=model, k=k,
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
                explanations=attrs, inputs=X_clean, model=model, k=k,
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
    """Compute RIS, RRS, ROS.

    Note: Stability is intentionally computed on X_noisy because RIS measures
    how stable the explainer is to small perturbations AROUND the noisy point.
    This is correct — we want to know if the explainer is stable at the
    operating point (noisy input), not at the clean input.
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
    return tuple(scores)


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

    FIX Bug 2: Faithfulness is always evaluated against clean X_eval,
    regardless of the sigma level used to generate attributions.
    """
    ensure_dirs()
    os.makedirs(TABLES_DIR, exist_ok=True)

    feature_metadata = _get_feature_metadata(dataset_name)
    perturb_method = get_perturb_method(std=PERTURB_STD, data_name=dataset_name)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    # FIX Bug 2: keep clean tensor permanently for faithfulness evaluation
    X_clean_t = torch.tensor(X_eval, dtype=torch.float32)

    n_features = X_eval.shape[1]
    k = max(1, int(TOP_K_FRACTION * n_features))

    noisy_inputs = noise_experiment_inputs(X_eval, sigma_levels=sigma_levels)

    all_records: list[dict] = []

    for sigma in sigma_levels:
        X_noisy_np = noisy_inputs[sigma]
        X_noisy_t = torch.tensor(X_noisy_np, dtype=torch.float32)

        sigma_csv = TABLES_DIR / f"phase3_{dataset_name}_sigma{sigma}.csv"
        if resume and sigma_csv.exists():
            print(f"  [SKIP] sigma={sigma:.1f} | Already exists: {sigma_csv}")
            try:
                df_loaded = pd.read_csv(sigma_csv)
                for _, row_data in df_loaded.iterrows():
                    record = row_data.to_dict()
                    record["sigma"] = sigma
                    all_records.append(record)
                continue
            except Exception as exc:
                print(f"  [WARNING] Failed to load '{sigma_csv}': {exc}. Recomputing.")

        sigma_records: list[dict] = []

        for method in EXPLAINER_METHODS:
            exp_name = EXPLAINER_DISPLAY[method]

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
                print(f"  [WARNING] sigma={sigma} | {exp_name} | attribution fail: {exc}")
                continue

            # FIX Bug 2: pass X_clean_t, not X_noisy_t
            pgf, pgu = _compute_faithfulness(
                attrs, X_clean_t, model, k, perturb_method, feature_metadata, exp_name
            )

            # Stability: use X_noisy_t (correct — measuring stability at operating point)
            live_exp = _build_explainer(method, model, X_train_t)
            ris, rrs, ros = _compute_stability(
                live_exp, X_noisy_t, model, perturb_method, feature_metadata, exp_name
            )

            row = {
                "sigma": sigma, "explainer": exp_name,
                "PGF": pgf, "PGU": pgu,
                "RIS": ris, "RRS": rrs, "ROS": ros,
            }
            all_records.append(row)
            sigma_records.append(row)
            print(
                f"  sigma={sigma:.1f} | {exp_name:<8} "
                f"| PGF={pgf:.4f} | PGU={pgu:.4f} | RIS={ris:.4f}"
            )

        if sigma_records:
            df_sigma = pd.DataFrame(sigma_records).set_index("explainer")
            df_sigma = df_sigma.drop(columns=["sigma"])
            csv_path = TABLES_DIR / f"phase3_{dataset_name}_sigma{sigma}.csv"
            df_sigma.to_csv(csv_path)
            print(f"  [save] CSV → {csv_path}")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.set_index(["sigma", "explainer"])
    df = df[["PGF", "PGU", "RIS", "RRS", "ROS"]]
    df.index.names = ["sigma", "explainer"]

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
    """Compare vanilla SHAP/LIME vs SmoothSHAP/SmoothLIME across sigma levels.

    FIX Bug 1: SmoothSHAP/SmoothLIME stability now uses a real SmoothExplainer
               object, so eval_relative_stability calls SmoothExplainer.get_explanations()
               internally — giving correct, distinct RIS values.

    FIX Bug 2: Faithfulness always evaluated on X_clean_t.

    FIX Bug 3: run_smooth_explainers() receives X_clean (not X_noisy).
               Noise is applied only once, internally.
    """
    ensure_dirs()
    os.makedirs(TABLES_DIR, exist_ok=True)

    feature_metadata = _get_feature_metadata(dataset_name)
    perturb_method = get_perturb_method(std=PERTURB_STD, data_name=dataset_name)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    # FIX Bug 2: permanent clean tensor for faithfulness
    X_clean_t = torch.tensor(X_eval, dtype=torch.float32)

    n_features = X_eval.shape[1]
    k = max(1, int(TOP_K_FRACTION * n_features))

    noisy_inputs = noise_experiment_inputs(X_eval, sigma_levels=sigma_levels)

    all_records: list[dict] = []

    for sigma in sigma_levels:
        X_noisy_np = noisy_inputs[sigma]
        X_noisy_t = torch.tensor(X_noisy_np, dtype=torch.float32)

        ba_sigma_csv = TABLES_DIR / f"phase3_{dataset_name}_ba_sigma{sigma}.csv"
        if resume and ba_sigma_csv.exists():
            print(f"  [SKIP] sigma={sigma:.1f} | Already exists: {ba_sigma_csv}")
            try:
                df_loaded = pd.read_csv(ba_sigma_csv)
                for _, row_data in df_loaded.iterrows():
                    record = row_data.to_dict()
                    record["sigma"] = sigma
                    all_records.append(record)
                continue
            except Exception as exc:
                print(f"  [WARNING] Failed to load '{ba_sigma_csv}': {exc}. Recomputing.")

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
                print(f"  [WARNING] vanilla {display} attr fail at sigma={sigma}: {exc}")
                vanilla_scores[display] = {"RIS": float("nan"), "PGF": float("nan")}
                continue

            # FIX Bug 2: evaluate faithfulness on clean inputs
            pgf, _ = _compute_faithfulness(
                attrs, X_clean_t, model, k, perturb_method, feature_metadata, display
            )
            # Stability on noisy inputs (correct operating point)
            ris, _, _ = _compute_stability(
                _build_explainer(method, model, X_train_t),
                X_noisy_t, model, perturb_method, feature_metadata, display
            )
            vanilla_scores[display] = {"RIS": ris, "PGF": pgf}
            print(f"  sigma={sigma:.1f} | {display:<10} | RIS={ris:.4f} | PGF={pgf:.4f}")

        # ── SmoothSHAP and SmoothLIME ─────────────────────────────────────────
        # FIX Bug 3: pass X_eval (clean), not X_noisy_np
        smooth_attrs = run_smooth_explainers(
            model, X_eval, X_train, sigma=sigma, K=K, seed=seed
        )

        smooth_scores: dict[str, dict] = {}
        key_map = {
            "smooth_shap": ("shap", "smooth_shap"),
            "smooth_lime": ("lime", "smooth_lime"),
        }
        for attr_key, (vanilla_method, display) in key_map.items():
            if attr_key not in smooth_attrs:
                smooth_scores[display] = {"RIS": float("nan"), "PGF": float("nan")}
                continue

            s_attrs = torch.tensor(smooth_attrs[attr_key], dtype=torch.float32)

            # FIX Bug 2: faithfulness on clean inputs
            pgf, _ = _compute_faithfulness(
                s_attrs, X_clean_t, model, k, perturb_method, feature_metadata, display
            )

            # FIX Bug 1: build a real SmoothExplainer as the live object for stability
            # so eval_relative_stability calls SmoothExplainer.get_explanations() internally
            try:
                smooth_live_exp = SmoothExplainer(
                    base_method=vanilla_method,
                    model=model,
                    dataset_tensor=X_train_t,
                    K=K,
                    internal_sigma=sigma,
                    seed=seed,
                )
            except Exception as exc:
                print(f"  [WARNING] Could not build SmoothExplainer for {display}: {exc}")
                smooth_scores[display] = {"RIS": float("nan"), "PGF": pgf}
                continue

            ris, _, _ = _compute_stability(
                smooth_live_exp, X_noisy_t, model, perturb_method, feature_metadata, display
            )
            smooth_scores[display] = {"RIS": ris, "PGF": pgf}
            print(f"  sigma={sigma:.1f} | {display:<15} | RIS={ris:.4f} | PGF={pgf:.4f}")

        # ── Assemble rows with delta columns ──────────────────────────────────
        pair_map = [("shap", "smooth_shap"), ("lime", "smooth_lime")]
        for vanilla_key, smooth_key in pair_map:
            v = vanilla_scores.get(vanilla_key, {})
            s = smooth_scores.get(smooth_key, {})

            v_ris = v.get("RIS", float("nan"))
            s_ris = s.get("RIS", float("nan"))
            v_pgf = v.get("PGF", float("nan"))
            s_pgf = s.get("PGF", float("nan"))

            delta_ris = (v_ris - s_ris) if not (np.isnan(v_ris) or np.isnan(s_ris)) else float("nan")
            delta_pgf = (s_pgf - v_pgf) if not (np.isnan(s_pgf) or np.isnan(v_pgf)) else float("nan")

            for name, ris_val, pgf_val, d_ris, d_pgf in [
                (vanilla_key, v_ris, v_pgf, 0.0, 0.0),
                (smooth_key,  s_ris, s_pgf, delta_ris, delta_pgf),
            ]:
                record = {
                    "sigma": sigma, "explainer": name,
                    "RIS": ris_val, "PGF": pgf_val,
                    "delta_RIS": d_ris, "delta_PGF": d_pgf,
                }
                all_records.append(record)
                sigma_ba_records.append({k: v for k, v in record.items() if k != "sigma"})

        if sigma_ba_records:
            df_ba = pd.DataFrame(sigma_ba_records).set_index("explainer")
            df_ba.to_csv(ba_sigma_csv)
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