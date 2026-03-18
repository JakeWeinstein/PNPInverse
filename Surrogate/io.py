"""Serialization utilities for surrogate models.

Uses pickle for full model persistence (including fitted RBF state).
Supports BVSurrogateModel, PODRBFSurrogateModel, NNSurrogateModel,
GPSurrogateModel, and PCESurrogateModel.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Union

from Surrogate.surrogate_model import BVSurrogateModel
from Surrogate.pod_rbf_model import PODRBFSurrogateModel
from Surrogate.nn_model import NNSurrogateModel
from Surrogate.gp_model import GPSurrogateModel
from Surrogate.pce_model import PCESurrogateModel

#: Union of all recognised surrogate model types.
_SURROGATE_TYPES = (
    BVSurrogateModel,
    PODRBFSurrogateModel,
    NNSurrogateModel,
    GPSurrogateModel,
    PCESurrogateModel,
)


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


def load_surrogate(path: str) -> Any:
    """Load a surrogate model from a pickle file.

    Accepts any recognised surrogate type: BVSurrogateModel,
    PODRBFSurrogateModel, NNSurrogateModel, GPSurrogateModel,
    or PCESurrogateModel.

    Parameters
    ----------
    path : str
        Path to ``.pkl`` file saved by :func:`save_surrogate`.

    Returns
    -------
    BVSurrogateModel | PODRBFSurrogateModel | NNSurrogateModel | GPSurrogateModel | PCESurrogateModel
        The loaded, fitted model.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, _SURROGATE_TYPES):
        accepted = ", ".join(t.__name__ for t in _SURROGATE_TYPES)
        raise TypeError(
            f"Expected one of ({accepted}), got {type(model).__name__}"
        )

    # Backward compatibility: older pickles may lack training_bounds
    if not hasattr(model, "training_bounds"):
        model.training_bounds = None

    # Backward compatibility: older configs may lack per-output smoothing
    # Only BVSurrogateModel/PODRBFSurrogateModel have a .config attribute;
    # NN/GP/PCE models do not, so guard accordingly.
    if hasattr(model, "config"):
        if not hasattr(model.config, "smoothing_cd"):
            model.config.smoothing_cd = None
        if not hasattr(model.config, "smoothing_pc"):
            model.config.smoothing_pc = None

    print(f"Surrogate model loaded from: {path} (n_eta={model.n_eta})")
    if model.training_bounds is not None:
        print(f"  Training bounds: {model.training_bounds}")
    return model
