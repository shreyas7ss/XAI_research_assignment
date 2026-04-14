"""
config.py — Centralised configuration for the OpenXAI replication project.

All paths and hyper-parameters are defined here so that no other file
contains hard-coded literals.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

# ─── Root directories ────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
TABLES_DIR: Path = RESULTS_DIR / "tables"
REPORT_DIR: Path = PROJECT_ROOT / "report"

# ─── Datasets ────────────────────────────────────────────────────────────────
DATASETS: list[str] = ["adult", "compas"]

# ─── Model ───────────────────────────────────────────────────────────────────
ML_MODEL: str = "ann"

# ─── Explainer identifiers ───────────────────────────────────────────────────
EXPLAINERS: list[str] = ["lime", "shap", "grad", "gradcam", "sg", "ig", "random"]

EXPLAINER_PARAM_DICTS: dict[str, dict] = {
    "lime": {"param_dict_lime": {"num_samples": 100}},
    "shap": {"param_dict_shap": {"nsamples": 100}},
}

# ─── Metrics ─────────────────────────────────────────────────────────────────
METRICS: list[str] = ["PGF", "PGU", "RIS", "RRS", "ROS"]
METRIC_DIRECTION: dict[str, str] = {
    "PGF": "higher",
    "PGU": "lower",
    "RIS": "lower",
    "RRS": "lower",
    "ROS": "lower",
}

# ─── Evaluation ──────────────────────────────────────────────────────────────
DEFAULT_N_SAMPLES: int = 300
RANDOM_SEED: int = 42


# ─── Seed utility ────────────────────────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        The random seed value to use (default: ``RANDOM_SEED`` from config).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """Create all required output directories (idempotent)."""
    for directory in (RESULTS_DIR, TABLES_DIR, REPORT_DIR):
        os.makedirs(directory, exist_ok=True)
