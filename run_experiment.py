"""
run_experiment.py — CLI entry-point for the OpenXAI replication pipeline.

Usage
-----
    # Quick smoke-test (50 samples, both datasets)
    python run_experiment.py --n_samples 50

    # Full run on both datasets
    python run_experiment.py

    # Single dataset
    python run_experiment.py --dataset adult --n_samples 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DATASETS,
    RESULTS_DIR,
    TABLES_DIR,
    DEFAULT_N_SAMPLES,
    ensure_dirs,
    set_seed,
)
from src.train_model import load_dataset, load_model
from src.run_explainers import run_all_explainers
from src.compute_metrics import compute_metrics_for_dataset
from src.visualize_results import plot_heatmap, plot_bar_charts, plot_multi_dataset


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed argument namespace with ``dataset`` and ``n_samples`` fields.
    """
    parser = argparse.ArgumentParser(
        description="OpenXAI NeurIPS 2022 — Replication Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + DATASETS,
        help="Dataset to evaluate (default: 'all' runs both datasets).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of test samples to evaluate (default: {DEFAULT_N_SAMPLES}).",
    )
    return parser.parse_args()


# ─── Pretty-print table ──────────────────────────────────────────────────────

def print_results_table(df: pd.DataFrame, data_name: str) -> None:
    """Print a formatted console table for *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Metric results indexed by explainer name.
    data_name : str
        Dataset label shown in the heading.
    """
    width = 80
    print("\n" + "=" * width)
    print(f"  RESULTS — Dataset: {data_name.upper()}")
    print("=" * width)
    print(df.round(4).to_string())
    print("-" * width + "\n")


# ─── Per-dataset pipeline ────────────────────────────────────────────────────

def run_dataset(data_name: str, n_samples: int) -> pd.DataFrame:
    """Execute the full pipeline for one dataset.

    Steps
    -----
    1. Load data and pretrained model.
    2. Generate explanations with all 7 explainers.
    3. Compute 5 benchmark metrics.
    4. Save CSV and figures.

    Parameters
    ----------
    data_name : str
        Dataset identifier (``'adult'`` or ``'compas'``).
    n_samples : int
        Number of test samples to process.

    Returns
    -------
    results_df : pd.DataFrame
        Metric DataFrame for downstream multi-dataset visualisation.
    """
    print(f"\n{'─' * 60}")
    print(f"  Dataset: {data_name.upper()}  |  n_samples={n_samples}")
    print(f"{'─' * 60}")

    # 1. Data & Model
    X_test, y_test = load_dataset(data_name, n_samples=n_samples, split="test")
    model = load_model(data_name)

    # 2. Explanations
    explanations = run_all_explainers(model, X_test, n_samples)

    # 3. Metrics
    results_df = compute_metrics_for_dataset(
        model, X_test, y_test, explanations, data_name
    )

    # 4a. Save CSV
    ensure_dirs()
    csv_path = TABLES_DIR / f"{data_name}_metrics.csv"
    results_df.to_csv(csv_path)
    print(f"  [save] CSV  → {csv_path}")

    # 4b. Console table
    print_results_table(results_df, data_name)

    # 4c. Figures
    plot_heatmap(results_df, data_name)
    plot_bar_charts(results_df, data_name)

    return results_df


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the replication experiment."""
    args = parse_args()
    set_seed()
    ensure_dirs()

    datasets_to_run = DATASETS if args.dataset == "all" else [args.dataset]

    all_results: Dict[str, pd.DataFrame] = {}
    for ds in datasets_to_run:
        all_results[ds] = run_dataset(ds, args.n_samples)

    # Multi-dataset comparison (only meaningful when ≥ 2 datasets are present)
    if len(all_results) >= 2:
        plot_multi_dataset(all_results, metric="PGF")

    print("\n✅  Experiment complete.")
    print(f"   Tables  → {TABLES_DIR}")
    print(f"   Figures → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
