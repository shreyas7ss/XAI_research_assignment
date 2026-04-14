"""
train_model.py — Loads the pretrained ANN model and dataset from the OpenXAI
library using the correct openxai 0.1 API.

Actual API (openxai 0.1)
------------------------
* ``openxai.dataloader.ReturnLoaders(data_name, download, batch_size)``
  → (trainloader, testloader) — torch DataLoader objects
* ``openxai.dataloader.ReturnTrainTestX(data_name, n_test, float_tensor)``
  → (X_train, X_test) as numpy arrays / FloatTensors
* ``openxai.LoadModel(data_name, ml_model, pretrained)``
  → pretrained ANN in eval mode
"""

from __future__ import annotations

import warnings
from typing import Tuple

import torch

from openxai import LoadModel
from openxai.dataloader import ReturnLoaders, ReturnTrainTestX

from src.config import DEFAULT_N_SAMPLES, ML_MODEL, RANDOM_SEED, set_seed


def load_dataset(
    data_name: str,
    n_samples: int = DEFAULT_N_SAMPLES,
    split: str = "test",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a dataset from the OpenXAI library and return float tensors.

    Uses ``ReturnLoaders`` to get the underlying ``TabularDataLoader`` objects,
    then slices ``n_samples`` rows from the requested split.

    Parameters
    ----------
    data_name : str
        Name of the dataset (e.g. ``'adult'``, ``'compas'``).
    n_samples : int
        Maximum number of samples to return.
    split : str
        Dataset split: ``'test'`` (default) or ``'train'``.

    Returns
    -------
    X_tensor : torch.FloatTensor
        Feature matrix of shape ``(n_samples, n_features)``.
    y_tensor : torch.Tensor
        Label vector of shape ``(n_samples,)``.
    """
    set_seed(RANDOM_SEED)
    trainloader, testloader = ReturnLoaders(data_name, download=True, batch_size=256)

    loader = testloader if split == "test" else trainloader
    dataset = loader.dataset  # TabularDataLoader (torch.utils.data.Dataset)

    X = torch.FloatTensor(dataset.data[:n_samples])
    y = torch.LongTensor(dataset.targets.values[:n_samples])

    print(
        f"[load_dataset] '{data_name}' — {X.shape[0]} samples, "
        f"{X.shape[1]} features."
    )
    return X, y


def load_train_tensor(data_name: str) -> torch.FloatTensor:
    """Return the full training feature tensor (used as background for LIME/IG).

    Parameters
    ----------
    data_name : str
        Name of the dataset.

    Returns
    -------
    X_train : torch.FloatTensor
        Full training feature matrix.
    """
    set_seed(RANDOM_SEED)
    trainloader, _ = ReturnLoaders(data_name, download=True, batch_size=256)
    X_train = torch.FloatTensor(trainloader.dataset.data)
    return X_train


def load_feature_metadata(data_name: str):
    """Return the feature metadata list (used by metric perturbation methods).

    Parameters
    ----------
    data_name : str
        Name of the dataset.

    Returns
    -------
    feature_metadata : list or dict
        Feature type metadata from the OpenXAI dataset object.
    """
    trainloader, _ = ReturnLoaders(data_name, download=True, batch_size=256)
    return trainloader.dataset.feature_metadata


def load_model(data_name: str, model_type: str = "ann") -> torch.nn.Module:
    """Load a pretrained model for the given dataset from OpenXAI.

    Parameters
    ----------
    data_name : str
        Name of the dataset the model was trained on.
    model_type : str
        Model type: ``'ann'`` (neural network) or ``'lr'`` (logistic regression).

    Returns
    -------
    model : torch.nn.Module
        Pretrained model in evaluation mode.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LoadModel(data_name=data_name, ml_model=model_type, pretrained=True)
    model.eval()
    print(f"[load_model] Loaded pretrained {model_type.upper()} for dataset '{data_name}'.")
    return model

