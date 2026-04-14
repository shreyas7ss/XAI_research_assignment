"""
visualize_results.py — Produces the three publication-style figures used in the
OpenXAI replication report.

Figure types
------------
1. ``plot_heatmap``       — Metric × explainer heatmap (column-normalised).
2. ``plot_bar_charts``    — One subplot per metric with annotated bars.
3. ``plot_multi_dataset`` — Grouped bar chart comparing Adult vs COMPAS on a
                            single metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import METRIC_DIRECTION, RESULTS_DIR, ensure_dirs


# ─── Shared style helpers ─────────────────────────────────────────────────────

def _apply_style() -> None:
    """Apply a clean, publication-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    """Save *fig* to *path*, creating parent directories as needed.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    path : Path
        Destination file path (should end with ``.png``).
    """
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [save] Figure written → {path}")


# ─── Figure 1: Heatmap ────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, data_name: str) -> Path:
    """Render a column-normalised metric × explainer heatmap.

    Each column (metric) is min-max normalised independently so that colour
    intensity reflects relative rank rather than absolute scale.  Raw values
    are annotated in each cell.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame of shape ``(n_explainers, n_metrics)``.
    data_name : str
        Dataset name used in the figure title and output filename.

    Returns
    -------
    out_path : Path
        Absolute path to the saved PNG.
    """
    _apply_style()
    norm_df = df.copy().astype(float)
    for col in norm_df.columns:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        denom = col_max - col_min if (col_max - col_min) != 0 else 1.0
        norm_df[col] = (norm_df[col] - col_min) / denom

    annot = df.round(4).astype(str).replace("nan", "—")

    # Build column labels with direction arrows
    col_labels = [
        f"{m}\n({'↑' if METRIC_DIRECTION[m] == 'higher' else '↓'})"
        for m in df.columns
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.6), max(4, len(df) * 0.7)))
    sns.heatmap(
        norm_df,
        annot=annot,
        fmt="s",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Normalised score (column-wise)"},
    )
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(
        f"OpenXAI Benchmark Metrics — {data_name.upper()} dataset\n"
        "(column-normalised; raw values annotated)",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    out_path = RESULTS_DIR / f"{data_name}_heatmap.png"
    _save(fig, out_path)
    return out_path


# ─── Figure 2: Bar charts (one per metric) ───────────────────────────────────

def plot_bar_charts(df: pd.DataFrame, data_name: str) -> Path:
    """Bar-chart grid with one subplot per metric.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame of shape ``(n_explainers, n_metrics)``.
    data_name : str
        Dataset name used in the title and filename.

    Returns
    -------
    out_path : Path
        Absolute path to the saved PNG.
    """
    _apply_style()
    metrics = df.columns.tolist()
    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes = np.array(axes).flatten()

    colors = sns.color_palette("husl", len(df))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].fillna(0)
        bars = ax.bar(df.index, values, color=colors, edgecolor="white", linewidth=0.8)

        # Annotate bar tops
        for bar, val in zip(bars, df[metric]):
            if np.isnan(val):
                label = "N/A"
                ypos = 0
            else:
                label = f"{val:.3f}"
                ypos = val
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                label,
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

        direction = METRIC_DIRECTION.get(metric, "")
        arrow = "↑ higher better" if direction == "higher" else "↓ lower better"
        ax.set_title(f"{metric}  ({arrow})", fontsize=10)
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-Metric Bar Charts — {data_name.upper()} dataset",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    out_path = RESULTS_DIR / f"{data_name}_bar_charts.png"
    _save(fig, out_path)
    return out_path


# ─── Figure 3: Multi-dataset grouped bars ────────────────────────────────────

def plot_multi_dataset(
    results: Dict[str, pd.DataFrame],
    metric: str = "PGF",
) -> Path:
    """Grouped bar chart comparing explainer scores across two datasets.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Mapping from dataset name to metric DataFrame.
    metric : str
        The metric column to compare across datasets (default: ``'PGF'``).

    Returns
    -------
    out_path : Path
        Absolute path to the saved PNG.
    """
    _apply_style()
    dataset_names = list(results.keys())
    combined = pd.concat(
        [results[d][metric].rename(d) for d in dataset_names if metric in results[d].columns],
        axis=1,
    )

    explainers = combined.index.tolist()
    x = np.arange(len(explainers))
    width = 0.35
    palette = sns.color_palette("Set2", len(dataset_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (ds, color) in enumerate(zip(dataset_names, palette)):
        if ds not in combined.columns:
            continue
        vals = combined[ds].fillna(0).values
        bars = ax.bar(x + i * width, vals, width, label=ds.upper(), color=color, edgecolor="white")
        for bar, v in zip(bars, combined[ds]):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    direction = METRIC_DIRECTION.get(metric, "")
    arrow = "↑ higher better" if direction == "higher" else "↓ lower better"
    ax.set_title(f"Multi-Dataset Comparison — {metric}  ({arrow})", fontsize=12)
    ax.set_ylabel(metric)
    ax.set_xticks(x + width * (len(dataset_names) - 1) / 2)
    ax.set_xticklabels(explainers, rotation=20, ha="right")
    ax.legend(title="Dataset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = RESULTS_DIR / "multi_dataset_comparison.png"
    _save(fig, out_path)
    return out_path
