"""NN ensemble wrapper and loader for BV kinetics surrogate inference.

Provides ``EnsembleMeanWrapper`` -- a thin adapter that wraps N
``NNSurrogateModel`` instances behind the same predict()/predict_batch()
API used by ``BVSurrogateModel`` and ``NNSurrogateModel``.

Usage::

    from Surrogate.ensemble import load_nn_ensemble

    model = load_nn_ensemble("StudyResults/surrogate_v11/nn_ensemble/D3-deeper")
    pred = model.predict(k0_1, k0_2, alpha_1, alpha_2)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class EnsembleMeanWrapper:
    """Wraps N ``NNSurrogateModel`` members behind the standard surrogate API.

    The ensemble prediction is the *mean* across members.  An optional
    ``predict_with_uncertainty()`` method returns per-point std as well.

    Parameters
    ----------
    models : sequence of NNSurrogateModel
        Fitted ensemble members (must share the same ``phi_applied`` grid).
    """

    def __init__(self, models: Sequence[Any]) -> None:
        if not models:
            raise ValueError("EnsembleMeanWrapper requires at least one model.")
        self.models = list(models)
        self._phi_applied: np.ndarray = models[0].phi_applied
        self._n_eta: int = models[0].n_eta

        # Validate all members share the same phi_applied grid
        for i, m in enumerate(models[1:], 1):
            if not np.allclose(m.phi_applied, self._phi_applied):
                raise ValueError(
                    f"Ensemble member {i} has different phi_applied grid than member 0."
                )

        # Merge training_bounds: intersection (max of lows, min of highs)
        self.training_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        bounds_list = [m.training_bounds for m in models if m.training_bounds is not None]
        if bounds_list:
            keys = bounds_list[0].keys()
            merged: Dict[str, Tuple[float, float]] = {}
            for k in keys:
                lo = max(b[k][0] for b in bounds_list)
                hi = min(b[k][1] for b in bounds_list)
                merged[k] = (lo, hi)
            self.training_bounds = merged

    # -----------------------------------------------------------------
    # Public API (compatible with BVSurrogateModel / NNSurrogateModel)
    # -----------------------------------------------------------------

    def predict(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
    ) -> Dict[str, np.ndarray]:
        """Predict I-V curves (ensemble mean) for a single parameter set."""
        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]])
        batch = self.predict_batch(params)
        return {
            "current_density": batch["current_density"][0],
            "peroxide_current": batch["peroxide_current"][0],
            "phi_applied": batch["phi_applied"],
        }

    def _predict_ensemble_raw(
        self, parameters: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect predictions from all members and return stacked arrays.

        Returns (cd_mean, cd_std, pc_mean, pc_std) — each shape (M, n_eta).
        Uses only the standard predict_batch API, avoiding a direct
        ``Surrogate.nn_training`` import (which pulls in PyTorch).
        """
        cd_preds: list[np.ndarray] = []
        pc_preds: list[np.ndarray] = []
        for m in self.models:
            result = m.predict_batch(parameters)
            cd_preds.append(result["current_density"])
            pc_preds.append(result["peroxide_current"])
        cd_stack = np.stack(cd_preds, axis=0)  # (E, M, n_eta)
        pc_stack = np.stack(pc_preds, axis=0)
        return (
            cd_stack.mean(axis=0),
            cd_stack.std(axis=0, ddof=1),
            pc_stack.mean(axis=0),
            pc_stack.std(axis=0, ddof=1),
        )

    def predict_batch(self, parameters: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict I-V curves (ensemble mean) for multiple parameter sets.

        Parameters
        ----------
        parameters : np.ndarray (M, 4)
            Each row: [k0_1, k0_2, alpha_1, alpha_2].

        Returns
        -------
        dict with keys 'current_density', 'peroxide_current', 'phi_applied'.
        """
        cd_mean, _, pc_mean, _ = self._predict_ensemble_raw(parameters)
        return {
            "current_density": cd_mean,
            "peroxide_current": pc_mean,
            "phi_applied": self._phi_applied.copy(),
        }

    def predict_with_uncertainty(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
    ) -> Dict[str, np.ndarray]:
        """Predict ensemble mean + std for a single parameter set.

        Returns
        -------
        dict with keys:
            'current_density', 'peroxide_current', 'phi_applied',
            'current_density_std', 'peroxide_current_std'
        """
        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]])
        cd_mean, cd_std, pc_mean, pc_std = self._predict_ensemble_raw(params)
        return {
            "current_density": cd_mean[0],
            "peroxide_current": pc_mean[0],
            "phi_applied": self._phi_applied.copy(),
            "current_density_std": cd_std[0],
            "peroxide_current_std": pc_std[0],
        }

    @property
    def phi_applied(self) -> np.ndarray:
        """Voltage grid."""
        return self._phi_applied.copy()

    @property
    def n_eta(self) -> int:
        """Number of voltage points."""
        return self._n_eta

    @property
    def is_fitted(self) -> bool:
        """Always True (all members must be fitted at construction)."""
        return True


def load_nn_ensemble(
    ensemble_dir: str,
    n_members: int = 5,
    device: str = "cpu",
) -> EnsembleMeanWrapper:
    """Load an NN ensemble from disk.

    Expects directories ``member_0/saved_model/``, ..., ``member_{n-1}/saved_model/``
    inside *ensemble_dir*.

    Parameters
    ----------
    ensemble_dir : str
        Path to the ensemble design directory (e.g. ``.../nn_ensemble/D3-deeper``).
    n_members : int
        Number of ensemble members to load.
    device : str
        PyTorch device.

    Returns
    -------
    EnsembleMeanWrapper

    Raises
    ------
    FileNotFoundError
        If any member directory is missing.
    """
    from Surrogate.nn_model import NNSurrogateModel

    models: list[Any] = []
    for i in range(n_members):
        member_path = os.path.join(ensemble_dir, f"member_{i}", "saved_model")
        if not os.path.isdir(member_path):
            raise FileNotFoundError(
                f"Ensemble member not found: {member_path}"
            )
        m = NNSurrogateModel.load(member_path, device=device)
        models.append(m)

    return EnsembleMeanWrapper(models)
