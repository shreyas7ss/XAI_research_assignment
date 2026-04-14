"""
train_model.py — Loads the pretrained ANN model and dataset from the OpenXAI
library.

This module has a single responsibility: provide helper functions to load model
and data objects that are reused across the pipeline.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import torch
from torch.utils.data import DataLoader as TorchDataLoader

import openxai
from openxai import DataLoader as OXDataLoader, LoadModel

from src.config import DEFAULT_N_SAMPLES, ML_MODEL, RANDOM_SEED, set_seed


def load_dataset(
    data_name: str,
    n_samples: int = DEFAULT_N_SAMPLES,
    split: str = "test",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a dataset from the OpenXAI library and return tensors.

    Parameters
    ----------
    data_name : str
        Name of the dataset (e.g. ``'adult'``, ``'compas'``).
    n_samples : int
        Maximum number of samples to return (taken from the front of the split).
    split : str
        Dataset split to load (``'test'`` or ``'train'``).

    Returns
    -------
    X_tensor : torch.Tensor
        Feature matrix of shape ``(n_samples, n_features)``.
    y_tensor : torch.Tensor
        Label vector of shape ``(n_samples,)``.
    """
    set_seed(RANDOM_SEED)
    loader: OXDataLoader = OXDataLoader(data_name=data_name)

    # The OpenXAI DataLoader exposes a torch DataLoader for each split.
    if split == "test":
        torch_loader: TorchDataLoader = loader.test_loader
    else:
        torch_loader = loader.train_loader

    X_list, y_list = [], []
    for batch_X, batch_y in torch_loader:
        X_list.append(batch_X)
        y_list.append(batch_y)
        if sum(t.shape[0] for t in X_list) >= n_samples:
            break

    X_tensor = torch.cat(X_list, dim=0)[:n_samples].float()
    y_tensor = torch.cat(y_list, dim=0)[:n_samples]

    print(
        f"[load_dataset] '{data_name}' — {X_tensor.shape[0]} samples, "
        f"{X_tensor.shape[1]} features."
    )
    return X_tensor, y_tensor


def load_model(data_name: str) -> torch.nn.Module:
    """Load the pretrained ANN for the given dataset from OpenXAI.

    Parameters
    ----------
    data_name : str
        Name of the dataset the model was trained on.

    Returns
    -------
    model : torch.nn.Module
        Pretrained ANN in evaluation mode.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LoadModel(data_name=data_name, ml_model=ML_MODEL, pretrained=True)
    model.eval()
    print(f"[load_model] Loaded pretrained ANN for dataset '{data_name}'.")
    return model
