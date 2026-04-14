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
from typing import Dict, Optional

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
    """Render a column-normalised metric x explainer heatmap."""
    _apply_style()
    
    # Ensure consistent column order and handle missing metrics
    from src.config import METRICS
    cols = [m for m in METRICS if m in df.columns]
    df = df[cols]
    
    norm_df = df.copy().astype(float)
    for col in norm_df.columns:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        denom = col_max - col_min if (col_max - col_min) != 0 else 1.0
        norm_df[col] = (norm_df[col] - col_min) / denom

    # Format annotations: numeric for valid floats, "-" for NaN
    annot = df.map(lambda v: f"{v:.3f}" if pd.notnull(v) else "—")

    # Build column labels with direction arrows
    col_labels = [
        f"{m}\n({'↑' if METRIC_DIRECTION.get(m) == 'higher' else '↓'})"
        for m in df.columns
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 1.8), max(5, len(df) * 0.8)))
    sns.heatmap(
        norm_df,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"label": "Normalised score (column-wise)", "shrink": 0.8},
    )
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_title(
        f"OpenXAI Benchmark Metrics — {data_name.upper()} dataset\n"
        "(column-normalised; raw values annotated)",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Explainer")

    out_path = RESULTS_DIR / f"{data_name}_heatmap.png"
    _save(fig, out_path)
    return out_path


# ─── Figure 2: Bar charts (one per metric) ───────────────────────────────────

def plot_bar_charts(df: pd.DataFrame, data_name: str) -> Path:
    """Bar-chart grid with one subplot per metric."""
    _apply_style()
    from src.config import METRICS
    metrics = [m for m in METRICS if m in df.columns]
    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    colors = sns.color_palette("viridis", len(df))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].fillna(0)
        bars = ax.bar(df.index, values, color=colors, edgecolor="black", linewidth=0.5)

        # Annotate bar tops
        for bar, val in zip(bars, df[metric]):
            if pd.isnull(val):
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
                fontsize=8,
                fontweight="bold"
            )

        direction = METRIC_DIRECTION.get(metric, "")
        arrow = "↑ higher better" if direction == "higher" else "↓ lower better"
        ax.set_title(f"{metric}\n({arrow})", fontsize=11, pad=10)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xticks(range(len(df.index)))
        ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Hide unused subplots
    for ax in axes[n_metrics:]:
        ax.set_axis_off()

    fig.suptitle(
        f"Per-Metric Benchmark Results — {data_name.upper()}",
        fontsize=15,
        y=1.05,
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


# ─── Phase 3 Figure 1: Degradation Curves ────────────────────────────────────

def degradation_curves(
    df: pd.DataFrame,
    metric: str,
    dataset_name: str,
    save_path: Optional[Path] = None,
) -> Path:
    """Line plot of metric score vs sigma, one line per explainer.

    Highlights SmoothSHAP and SmoothLIME with a thicker linewidth to make
    the noise-robustness advantage visually prominent.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame indexed by (sigma, explainer)
            with metric columns.  Produced by ``degradation_study``.
        metric (str): Column name to plot, e.g. ``'RIS'`` or ``'PGF'``.
        dataset_name (str): Dataset label used in the plot title and filename.
        save_path (Path | None): Custom output path.  If ``None``, saves to
            ``results/phase3_{dataset}_degradation_{metric}.png``.

    Returns:
        Path: Absolute path to the saved PNG.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    sigmas = sorted(df.index.get_level_values("sigma").unique())
    explainers = df.index.get_level_values("explainer").unique().tolist()
    palette = sns.color_palette("tab10", len(explainers))
    smooth_keys = {"smooth_shap", "smooth_lime"}

    for exp, color in zip(explainers, palette):
        vals = []
        for s in sigmas:
            try:
                v = df.xs(s, level="sigma").loc[exp, metric]
            except KeyError:
                v = float("nan")
            vals.append(v)

        lw = 2.5 if exp in smooth_keys else 1.4
        ls = "-" if exp in smooth_keys else "--"
        ax.plot(
            sigmas, vals,
            marker="o", label=exp, color=color,
            linewidth=lw, linestyle=ls,
            markersize=5,
        )

    direction = METRIC_DIRECTION.get(metric, "")
    arrow = "↑ higher is better" if direction == "higher" else "↓ lower is better"
    ax.set_title(
        f"{metric} Degradation Under Gaussian Noise — {dataset_name.upper()}\n"
        f"({arrow}; thick solid = Smooth variants)",
        fontsize=12,
        pad=14,
    )
    ax.set_xlabel("Gaussian Noise σ", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xticks(sigmas)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        title="Explainer",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    fig.tight_layout()

    out_path = (
        save_path
        if save_path is not None
        else RESULTS_DIR / f"phase3_{dataset_name}_degradation_{metric}.png"
    )
    _save(fig, out_path)
    return out_path


# ─── Phase 3 Figure 2: Before vs After Bars ──────────────────────────────────

def before_after_bars(
    df: pd.DataFrame,
    sigma: float = 0.3,
    dataset_name: str = "dataset",
    save_path: Optional[Path] = None,
) -> Path:
    """Grouped bar chart comparing vanilla vs smooth explainers at a fixed sigma.

    Two side-by-side subplots: RIS (left) and PGF (right).  Four bars each:
    SHAP, SmoothSHAP, LIME, SmoothLIME.  Smooth bars are annotated with the
    percentage improvement over their vanilla counterpart.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame indexed by (sigma, explainer)
            with columns ``['RIS', 'PGF', 'delta_RIS', 'delta_PGF']``.
            Produced by ``before_after_comparison``.
        sigma (float): Sigma level to display.  Default 0.3.
        dataset_name (str): Dataset label for the title.
        save_path (Path | None): Custom output path.  If ``None``, saves to
            ``results/phase3_{dataset}_ba_sigma{sigma}.png``.

    Returns:
        Path: Absolute path to the saved PNG.
    """
    _apply_style()

    # Try to extract the requested sigma level
    available_sigmas = sorted(df.index.get_level_values("sigma").unique())
    if sigma not in available_sigmas:
        sigma = available_sigmas[-1]  # fall back to highest available

    try:
        sub = df.xs(sigma, level="sigma")
    except KeyError:
        sub = pd.DataFrame()

    explainers_order = ["shap", "smooth_shap", "lime", "smooth_lime"]
    display_labels = ["SHAP", "SmoothSHAP", "LIME", "SmoothLIME"]
    vanilla_grey = "#9E9E9E"
    smooth_blue = "#1565C0"
    smooth_green = "#2E7D32"
    bar_colors = [vanilla_grey, smooth_blue, vanilla_grey, smooth_green]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax_idx, (metric, better_word, delta_col) in enumerate([
        ("RIS", "↓ lower better", "delta_RIS"),
        ("PGF", "↑ higher better", "delta_PGF"),
    ]):
        ax = axes[ax_idx]
        values = []
        deltas = []
        for exp in explainers_order:
            if sub.empty or exp not in sub.index:
                values.append(0.0)
                deltas.append(float("nan"))
            else:
                values.append(float(sub.loc[exp, metric]) if metric in sub.columns else 0.0)
                deltas.append(
                    float(sub.loc[exp, delta_col])
                    if delta_col in sub.columns
                    else float("nan")
                )

        x = np.arange(len(explainers_order))
        bars = ax.bar(
            x, values,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.2,
            width=0.55,
        )

        # Annotate smooth bars with % improvement
        smooth_indices = [1, 3]  # SmoothSHAP, SmoothLIME
        for si in smooth_indices:
            delta = deltas[si]
            if not np.isnan(delta):
                vanilla_val = values[si - 1]
                pct = (delta / vanilla_val * 100) if vanilla_val != 0 else 0.0
                bar = bars[si]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"+{pct:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color=smooth_blue if si == 1 else smooth_green,
                )

        ax.set_title(f"{metric}  ({better_word})", fontsize=11, pad=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Vanilla vs Smooth Explainers at σ={sigma} — {dataset_name.upper()}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    sigma_str = str(sigma).replace(".", "")
    out_path = (
        save_path
        if save_path is not None
        else RESULTS_DIR / f"phase3_{dataset_name}_ba_sigma{sigma_str}.png"
    )
    _save(fig, out_path)
    return out_path


# ─── Phase 3 Figure 3: Smoothing Benefit Heatmap ─────────────────────────────

def smoothing_benefit_heatmap(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    save_path: Optional[Path] = None,
) -> Path:
    """RIS stability heatmap — rows = sigma levels, cols = 4 explainer variants.

    Uses the ``RdYlGn_r`` colormap: red = unstable (high RIS), green = stable
    (low RIS).  Raw RIS values are annotated in each cell.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame indexed by (sigma, explainer)
            with at least a ``'RIS'`` column.  From ``before_after_comparison``.
        dataset_name (str): Dataset label for the title.
        save_path (Path | None): Custom output path.  If ``None``, saves to
            ``results/phase3_{dataset}_smoothing_heatmap.png``.

    Returns:
        Path: Absolute path to the saved PNG.
    """
    _apply_style()

    explainers_order = ["shap", "smooth_shap", "lime", "smooth_lime"]
    col_labels = ["SHAP", "SmoothSHAP", "LIME", "SmoothLIME"]
    sigmas = sorted(df.index.get_level_values("sigma").unique())

    # Build a (n_sigma x 4) matrix for RIS
    data_matrix = np.full((len(sigmas), len(explainers_order)), np.nan)
    for i, sigma in enumerate(sigmas):
        try:
            sub = df.xs(sigma, level="sigma")
        except KeyError:
            continue
        for j, exp in enumerate(explainers_order):
            if exp in sub.index and "RIS" in sub.columns:
                data_matrix[i, j] = float(sub.loc[exp, "RIS"])

    heatmap_df = pd.DataFrame(
        data_matrix,
        index=[f"σ={s}" for s in sigmas],
        columns=col_labels,
    )

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(sigmas) * 1.1)))
    annot = heatmap_df.map(lambda v: f"{v:.3f}" if pd.notnull(v) else "—")

    sns.heatmap(
        heatmap_df,
        annot=annot,
        fmt="",
        cmap="RdYlGn_r",
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "RIS (↓ lower = more stable)", "shrink": 0.8},
    )
    ax.set_title(
        f"Stability (RIS) Heatmap — Vanilla vs Smooth — {dataset_name.upper()}",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("Explainer Variant", fontsize=10)
    ax.set_ylabel("Noise Level (σ)", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    out_path = (
        save_path
        if save_path is not None
        else RESULTS_DIR / f"phase3_{dataset_name}_smoothing_heatmap.png"
    )
    _save(fig, out_path)
    return out_path


# ─── Phase 3 Figure 4: Multi-Sigma Lines with Robustness Gap ─────────────────

def multi_sigma_lines(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    save_path: Optional[Path] = None,
) -> Path:
    """RIS vs sigma line plot for SHAP family and LIME family with gap shading.

    SHAP family: blue tones.  LIME family: orange tones.
    Vanilla lines are dashed; Smooth lines are solid.
    A shaded region between each vanilla/smooth pair visualises the
    robustness gap — how much SmoothSHAP/SmoothLIME resists noise.

    Args:
        df (pd.DataFrame): MultiIndex DataFrame indexed by (sigma, explainer)
            with at least a ``'RIS'`` column.  From ``before_after_comparison``.
        dataset_name (str): Dataset label for the title.
        save_path (Path | None): Custom output path.  If ``None``, saves to
            ``results/phase3_{dataset}_robustness_gap.png``.

    Returns:
        Path: Absolute path to the saved PNG.
    """
    _apply_style()

    sigmas = sorted(df.index.get_level_values("sigma").unique())

    def _get_ris(exp: str) -> list[float]:
        vals = []
        for s in sigmas:
            try:
                v = df.xs(s, level="sigma").loc[exp, "RIS"]
                vals.append(float(v))
            except KeyError:
                vals.append(float("nan"))
        return vals

    shap_ris         = _get_ris("shap")
    smooth_shap_ris  = _get_ris("smooth_shap")
    lime_ris         = _get_ris("lime")
    smooth_lime_ris  = _get_ris("smooth_lime")

    # Colour palette
    shap_dark   = "#1565C0"   # dark blue  — vanilla SHAP
    shap_light  = "#90CAF9"   # light blue — SmoothSHAP
    lime_dark   = "#E65100"   # dark orange — vanilla LIME
    lime_light  = "#FFCC80"   # light orange — SmoothLIME

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array(sigmas)

    # ── SHAP family ───────────────────────────────────────────────────────────
    ax.plot(x, shap_ris,        color=shap_dark,  linestyle="--", linewidth=1.8,
            marker="o", markersize=5, label="SHAP (vanilla)")
    ax.plot(x, smooth_shap_ris, color=shap_dark,  linestyle="-",  linewidth=2.5,
            marker="s", markersize=5, label="SmoothSHAP")
    ax.fill_between(
        x,
        [min(a, b) for a, b in zip(shap_ris, smooth_shap_ris)],
        [max(a, b) for a, b in zip(shap_ris, smooth_shap_ris)],
        alpha=0.15, color=shap_light, label="SHAP robustness gap",
    )

    # ── LIME family ───────────────────────────────────────────────────────────
    ax.plot(x, lime_ris,        color=lime_dark,  linestyle="--", linewidth=1.8,
            marker="o", markersize=5, label="LIME (vanilla)")
    ax.plot(x, smooth_lime_ris, color=lime_dark,  linestyle="-",  linewidth=2.5,
            marker="s", markersize=5, label="SmoothLIME")
    ax.fill_between(
        x,
        [min(a, b) for a, b in zip(lime_ris, smooth_lime_ris)],
        [max(a, b) for a, b in zip(lime_ris, smooth_lime_ris)],
        alpha=0.15, color=lime_light, label="LIME robustness gap",
    )

    ax.set_title(
        f"RIS Robustness Gap — {dataset_name.upper()}\n"
        "(dashed = vanilla | solid = smooth | shaded = improvement region)",
        fontsize=12, pad=14,
    )
    ax.set_xlabel("Gaussian Noise σ", fontsize=11)
    ax.set_ylabel("RIS  (↓ lower = more stable)", fontsize=11)
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8.5, title_fontsize=9, frameon=True)
    fig.tight_layout()

    out_path = (
        save_path
        if save_path is not None
        else RESULTS_DIR / f"phase3_{dataset_name}_robustness_gap.png"
    )
    _save(fig, out_path)
    return out_path
