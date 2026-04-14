"""
run_phase3.py — CLI entry-point for the Phase 3 noise-robustness extension
of the OpenXAI replication project.

Usage
-----
    # All 4 datasets, 300 samples, K=20 smoothing copies
    python run_phase3.py

    # Single dataset smoke-test
    python run_phase3.py --dataset adult --n_samples 50

    # Only sigma=0.3 (debug mode)
    python run_phase3.py --dataset compas --sigma_only 0.3

    # Custom K
    python run_phase3.py --dataset adult --n_samples 100 --K 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    RESULTS_DIR,
    TABLES_DIR,
    DEFAULT_N_SAMPLES,
    RANDOM_SEED,
    ensure_dirs,
    set_seed,
)
from src.noise_utils import SIGMA_LEVELS
from src.train_model import load_dataset, load_model, load_train_tensor
from src.phase3_metrics import degradation_study, before_after_comparison
from src.visualize_results import (
    degradation_curves,
    before_after_bars,
    smoothing_benefit_heatmap,
    multi_sigma_lines,
)

# OpenXAI supports these 4 datasets for Phase 3
PHASE3_DATASETS: list[str] = ["adult", "compas", "german", "heloc"]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 3 pipeline.

    Returns:
        argparse.Namespace: Parsed namespace with fields ``dataset``,
        ``n_samples``, ``K``, and ``sigma_only``.
    """
    parser = argparse.ArgumentParser(
        description="Phase 3 — Noise-Robustness Extension of OpenXAI Replication"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + PHASE3_DATASETS,
        help="Dataset to evaluate. 'all' runs adult, compas, german, heloc. Default: 'all'.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of test samples to evaluate (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=20,
        help="Number of noisy copies to average in SmoothSHAP/SmoothLIME (default: 20).",
    )
    parser.add_argument(
        "--sigma_only",
        type=float,
        default=None,
        help=(
            "Run only a single sigma level (for debugging). "
            "Must be one of 0.0, 0.1, 0.3, 0.5."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results to avoid recomputing sigma levels.",
    )
    return parser.parse_args()


# ─── Summary table helper ─────────────────────────────────────────────────────

def _print_summary_table(
    deg_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Print to console the best explainer per sigma level for RIS and PGF.

    Args:
        deg_df (pd.DataFrame): MultiIndex DataFrame from ``degradation_study``,
            indexed by (sigma, explainer).
        dataset_name (str): Dataset label for the heading.
    """
    width = 80
    print("\n" + "=" * width)
    print(f"  PHASE 3 SUMMARY — Dataset: {dataset_name.upper()}")
    print("  Best explainer at each sigma level")
    print("=" * width)
    print(f"  {'Sigma':<10} {'Best RIS (↓)':<25} {'Best PGF (↑)':<25}")
    print("  " + "-" * (width - 2))

    sigmas = deg_df.index.get_level_values("sigma").unique()
    for sigma in sorted(sigmas):
        try:
            sigma_df = deg_df.xs(sigma, level="sigma")
        except KeyError:
            continue
        best_ris_exp = sigma_df["RIS"].idxmin() if "RIS" in sigma_df else "N/A"
        best_pgf_exp = sigma_df["PGF"].idxmax() if "PGF" in sigma_df else "N/A"
        best_ris_val = sigma_df.loc[best_ris_exp, "RIS"] if best_ris_exp != "N/A" else float("nan")
        best_pgf_val = sigma_df.loc[best_pgf_exp, "PGF"] if best_pgf_exp != "N/A" else float("nan")
        print(
            f"  {sigma:<10.1f} "
            f"{best_ris_exp + f' ({best_ris_val:.4f})':<25} "
            f"{best_pgf_exp + f' ({best_pgf_val:.4f})':<25}"
        )
    print("=" * width + "\n")


# ─── Per-dataset pipeline ─────────────────────────────────────────────────────

def run_phase3_dataset(
    data_name: str,
    n_samples: int,
    K: int,
    sigma_levels: List[float],
    resume: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Execute the full Phase 3 pipeline for one dataset.

    Steps
    -----
    1. Load data and pretrained ANN.
    2. Run ``degradation_study`` for all 7 explainers across all sigma levels.
    3. Run ``before_after_comparison`` for SHAP/SmoothSHAP and LIME/SmoothLIME.
    4. Generate all 4 Phase 3 figures.
    5. Print summary table.

    Args:
        data_name (str): Dataset identifier (e.g. ``'adult'``, ``'german'``).
        n_samples (int): Number of evaluation samples.
        K (int): Number of smooth averaging copies.
        sigma_levels (list[float]): Sigma levels to sweep.

    Returns:
        dict[str, pd.DataFrame]: ``{'degradation': df, 'before_after': df}``.
    """
    print(f"\n{'═' * 60}")
    print(f"  PHASE 3 — Dataset: {data_name.upper()}  |  n_samples={n_samples}  |  K={K}")
    print(f"{'═' * 60}")

    # ── 1. Load data and model ────────────────────────────────────────────────
    try:
        X_test_t, y_test = load_dataset(data_name, n_samples=n_samples, split="test")
        X_train_t = load_train_tensor(data_name)
        model = load_model(data_name, model_type="ann")
    except Exception as exc:
        print(f"\n[ERROR] Could not load dataset '{data_name}': {exc}")
        print(f"  Skipping {data_name}. (German Credit and HELOC require openxai ≥ 0.2.)\n")
        return {}

    X_eval_np = X_test_t.numpy()
    X_train_np = X_train_t.numpy()

    # ── 2. Degradation study ──────────────────────────────────────────────────
    print(f"\n[Phase 3 / {data_name}] Running degradation study …")
    deg_df = degradation_study(
        model=model,
        X_eval=X_eval_np,
        X_train=X_train_np,
        dataset_name=data_name,
        sigma_levels=sigma_levels,
        resume=resume,
    )

    # ── 3. Before vs After comparison ─────────────────────────────────────────
    print(f"\n[Phase 3 / {data_name}] Running before-after comparison …")
    ba_df = before_after_comparison(
        model=model,
        X_eval=X_eval_np,
        X_train=X_train_np,
        dataset_name=data_name,
        sigma_levels=sigma_levels,
        K=K,
        seed=RANDOM_SEED,
        resume=resume,
    )

    # ── 4. Figures ────────────────────────────────────────────────────────────
    if not deg_df.empty:
        for metric in ("RIS", "PGF"):
            try:
                degradation_curves(deg_df, metric=metric, dataset_name=data_name)
            except Exception as exc:
                print(f"  [WARNING] degradation_curves({metric}) failed: {exc}")

    if not ba_df.empty:
        # Before-after bars at sigma=0.3 (fixed display sigma)
        for sigma_plot in (0.3, 0.5):
            if sigma_plot in ba_df.index.get_level_values("sigma"):
                try:
                    before_after_bars(ba_df, sigma=sigma_plot, dataset_name=data_name)
                except Exception as exc:
                    print(f"  [WARNING] before_after_bars(sigma={sigma_plot}) failed: {exc}")

        try:
            smoothing_benefit_heatmap(ba_df, dataset_name=data_name)
        except Exception as exc:
            print(f"  [WARNING] smoothing_benefit_heatmap failed: {exc}")

        try:
            multi_sigma_lines(ba_df, dataset_name=data_name)
        except Exception as exc:
            print(f"  [WARNING] multi_sigma_lines failed: {exc}")

    # ── 5. Summary table ──────────────────────────────────────────────────────
    if not deg_df.empty:
        _print_summary_table(deg_df, data_name)

    return {"degradation": deg_df, "before_after": ba_df}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the Phase 3 noise-robustness experiment."""
    args = parse_args()
    set_seed(RANDOM_SEED)
    ensure_dirs()

    # Resolve dataset list
    datasets_to_run: List[str] = (
        PHASE3_DATASETS if args.dataset == "all" else [args.dataset]
    )

    # Resolve sigma levels
    sigma_levels: List[float]
    if args.sigma_only is not None:
        if args.sigma_only not in SIGMA_LEVELS:
            print(
                f"[WARNING] sigma_only={args.sigma_only} not in {SIGMA_LEVELS}. "
                "Running anyway."
            )
        sigma_levels = [args.sigma_only]
    else:
        sigma_levels = SIGMA_LEVELS

    print(f"\n🔬 Phase 3 — Noise Robustness Study")
    print(f"   Datasets  : {datasets_to_run}")
    print(f"   Sigma lvls: {sigma_levels}")
    print(f"   n_samples : {args.n_samples}")
    print(f"   K (smooth): {args.K}")

    all_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds in datasets_to_run:
        # ── Dataset-level Resume Skip ─────────────────────────────────────────
        if args.resume:
            full_deg = TABLES_DIR / f"phase3_{ds}_degradation_full.csv"
            full_ba = TABLES_DIR / f"phase3_{ds}_before_after.csv"
            if full_deg.exists() and full_ba.exists():
                print(f"\n[SKIP] Dataset '{ds}' is already fully complete. Moving to next...")
                continue

        all_results[ds] = run_phase3_dataset(
            data_name=ds,
            n_samples=args.n_samples,
            K=args.K,
            sigma_levels=sigma_levels,
            resume=args.resume,
        )

    print("\n✅  Phase 3 complete.")
    print(f"   Tables  → {TABLES_DIR}")
    print(f"   Figures → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
