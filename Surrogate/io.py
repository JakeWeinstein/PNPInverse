"""Serialization utilities for BVSurrogateModel.

Uses pickle for full model persistence (including fitted RBF state).
"""

from __future__ import annotations

import os
import pickle
from typing import Any

from Surrogate.surrogate_model import BVSurrogateModel


def save_surrogate(model: BVSurrogateModel, path: str) -> None:
    """Save a fitted BVSurrogateModel to a pickle file.

    Parameters
    ----------
    model : BVSurrogateModel
        Must be fitted (``model.is_fitted`` is True).
    path : str
        Output file path (typically ends in ``.pkl``).
    """
    if not model.is_fitted:
        raise ValueError("Cannot save an unfitted model. Call fit() first.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Surrogate model saved to: {path}")


def load_surrogate(path: str) -> BVSurrogateModel:
    """Load a BVSurrogateModel from a pickle file.

    Parameters
    ----------
    path : str
        Path to ``.pkl`` file saved by :func:`save_surrogate`.

    Returns
    -------
    BVSurrogateModel
        The loaded, fitted model.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, BVSurrogateModel):
        raise TypeError(f"Expected BVSurrogateModel, got {type(model).__name__}")

    # Backward compatibility: older pickles may lack training_bounds
    if not hasattr(model, "training_bounds"):
        model.training_bounds = None

    # Backward compatibility: older configs may lack per-output smoothing
    if not hasattr(model.config, "smoothing_cd"):
        model.config.smoothing_cd = None
    if not hasattr(model.config, "smoothing_pc"):
        model.config.smoothing_pc = None

    print(f"Surrogate model loaded from: {path} (n_eta={model.n_eta})")
    if model.training_bounds is not None:
        print(f"  Training bounds: {model.training_bounds}")
    return model
