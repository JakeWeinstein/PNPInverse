"""RBF interpolation surrogate model for BV kinetics I-V curves.

Uses scipy.interpolate.RBFInterpolator to build a smooth surrogate from
(parameters) -> (I-V curve) training data.  Two separate RBF models are
fitted: one for current_density and one for peroxide_current.

The model supports:
- Log-space k0 transform (input k0 -> log10(k0))
- Z-score normalization of inputs for numerical conditioning
- Vector-valued outputs (full I-V curve predicted at once)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class SurrogateConfig:
    """Configuration for the RBF surrogate model.

    Attributes
    ----------
    kernel : str
        RBF kernel type.  Passed to ``scipy.interpolate.RBFInterpolator``.
        Options: "linear", "thin_plate_spline", "cubic", "quintic",
        "multiquadric", "inverse_multiquadric", "inverse_quadratic",
        "gaussian".
    degree : int
        Degree of the polynomial added to the RBF.  -1 means no polynomial,
        0 means constant, 1 means linear, etc.  Must satisfy degree >= 0
        for "thin_plate_spline" and "multiquadric".
    smoothing : float
        Default smoothing parameter (regularization).  0.0 means exact
        interpolation.  Used for both CD and PC unless overridden by
        ``smoothing_cd`` or ``smoothing_pc``.
    smoothing_cd : float or None
        Per-output smoothing for the current_density RBF.  If None, falls
        back to ``smoothing``.
    smoothing_pc : float or None
        Per-output smoothing for the peroxide_current RBF.  If None, falls
        back to ``smoothing``.
    log_space_k0 : bool
        If True, transform k0_1 and k0_2 to log10 space before fitting.
    normalize_inputs : bool
        If True, z-score normalize the (transformed) inputs.
    """
    kernel: str = "thin_plate_spline"
    degree: int = 1
    smoothing: float = 0.0
    smoothing_cd: Optional[float] = None
    smoothing_pc: Optional[float] = None
    log_space_k0: bool = True
    normalize_inputs: bool = True


class BVSurrogateModel:
    """RBF surrogate model mapping BV parameters to I-V curves.

    After fitting, predicts current_density(eta) and peroxide_current(eta)
    as functions of (k0_1, k0_2, alpha_1, alpha_2).

    Parameters
    ----------
    config : SurrogateConfig
        Model configuration.
    """

    def __init__(self, config: Optional[SurrogateConfig] = None):
        self.config = config or SurrogateConfig()
        self._rbf_cd: Optional[RBFInterpolator] = None
        self._rbf_pc: Optional[RBFInterpolator] = None
        self._phi_applied: Optional[np.ndarray] = None
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._n_eta: int = 0
        self._is_fitted: bool = False
        self.training_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    def _transform_inputs(self, parameters: np.ndarray) -> np.ndarray:
        """Apply log-space and normalization transforms to raw parameters.

        Parameters
        ----------
        parameters : np.ndarray of shape (N, 4)
            Columns: [k0_1, k0_2, alpha_1, alpha_2] in physical space.

        Returns
        -------
        np.ndarray of shape (N, 4)
            Transformed parameters.
        """
        X = parameters.copy()
        if self.config.log_space_k0:
            X[:, 0] = np.log10(np.maximum(X[:, 0], 1e-30))
            X[:, 1] = np.log10(np.maximum(X[:, 1], 1e-30))
        if self.config.normalize_inputs and self._input_mean is not None:
            X = (X - self._input_mean) / np.maximum(self._input_std, 1e-15)
        return X

    def fit(
        self,
        parameters: np.ndarray,
        current_density: np.ndarray,
        peroxide_current: np.ndarray,
        phi_applied: np.ndarray,
    ) -> "BVSurrogateModel":
        """Fit two RBF interpolators from training data.

        Parameters
        ----------
        parameters : np.ndarray of shape (N, 4)
            Training parameter samples [k0_1, k0_2, alpha_1, alpha_2].
        current_density : np.ndarray of shape (N, n_eta)
            Current density I-V curves for each sample.
        peroxide_current : np.ndarray of shape (N, n_eta)
            Peroxide current I-V curves for each sample.
        phi_applied : np.ndarray of shape (n_eta,)
            Voltage grid (stored for reference).

        Returns
        -------
        self
        """
        N = parameters.shape[0]
        self._n_eta = phi_applied.shape[0]
        self._phi_applied = phi_applied.copy()

        assert parameters.shape == (N, 4), f"Expected (N,4), got {parameters.shape}"
        assert current_density.shape == (N, self._n_eta), (
            f"Expected ({N},{self._n_eta}), got {current_density.shape}"
        )
        assert peroxide_current.shape == (N, self._n_eta), (
            f"Expected ({N},{self._n_eta}), got {peroxide_current.shape}"
        )

        # Compute normalization stats on transformed inputs
        X_raw = parameters.copy()
        if self.config.log_space_k0:
            X_raw[:, 0] = np.log10(np.maximum(X_raw[:, 0], 1e-30))
            X_raw[:, 1] = np.log10(np.maximum(X_raw[:, 1], 1e-30))

        if self.config.normalize_inputs:
            self._input_mean = X_raw.mean(axis=0)
            self._input_std = X_raw.std(axis=0)
        else:
            self._input_mean = np.zeros(4)
            self._input_std = np.ones(4)

        X = self._transform_inputs(parameters)

        # Resolve per-output smoothing (fall back to global default)
        sm_cd = self.config.smoothing_cd if self.config.smoothing_cd is not None else self.config.smoothing
        sm_pc = self.config.smoothing_pc if self.config.smoothing_pc is not None else self.config.smoothing

        # Fit RBF for current density (vector-valued: N -> n_eta)
        self._rbf_cd = RBFInterpolator(
            X, current_density,
            kernel=self.config.kernel,
            degree=self.config.degree,
            smoothing=sm_cd,
        )

        # Fit RBF for peroxide current (vector-valued: N -> n_eta)
        self._rbf_pc = RBFInterpolator(
            X, peroxide_current,
            kernel=self.config.kernel,
            degree=self.config.degree,
            smoothing=sm_pc,
        )

        # Store training bounds (min/max of each parameter column)
        self.training_bounds = {
            "k0_1": (float(parameters[:, 0].min()), float(parameters[:, 0].max())),
            "k0_2": (float(parameters[:, 1].min()), float(parameters[:, 1].max())),
            "alpha_1": (float(parameters[:, 2].min()), float(parameters[:, 2].max())),
            "alpha_2": (float(parameters[:, 3].min()), float(parameters[:, 3].max())),
        }

        self._is_fitted = True
        return self

    def predict(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
    ) -> Dict[str, np.ndarray]:
        """Predict I-V curves for a single parameter set.

        Parameters
        ----------
        k0_1, k0_2, alpha_1, alpha_2 : float
            BV kinetics parameters.

        Returns
        -------
        dict with keys:
            'current_density' : np.ndarray of shape (n_eta,)
            'peroxide_current' : np.ndarray of shape (n_eta,)
            'phi_applied' : np.ndarray of shape (n_eta,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]], dtype=float)
        X = self._transform_inputs(params)

        cd = self._rbf_cd(X).ravel()
        pc = self._rbf_pc(X).ravel()

        return {
            "current_density": cd,
            "peroxide_current": pc,
            "phi_applied": self._phi_applied.copy(),
        }

    def predict_batch(self, parameters: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict I-V curves for multiple parameter sets.

        Parameters
        ----------
        parameters : np.ndarray of shape (M, 4)
            Each row: [k0_1, k0_2, alpha_1, alpha_2].

        Returns
        -------
        dict with keys:
            'current_density' : np.ndarray of shape (M, n_eta)
            'peroxide_current' : np.ndarray of shape (M, n_eta)
            'phi_applied' : np.ndarray of shape (n_eta,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = self._transform_inputs(parameters)

        cd = self._rbf_cd(X)
        pc = self._rbf_pc(X)

        return {
            "current_density": cd,
            "peroxide_current": pc,
            "phi_applied": self._phi_applied.copy(),
        }

    @property
    def n_eta(self) -> int:
        """Number of voltage points in the I-V curve."""
        return self._n_eta

    @property
    def phi_applied(self) -> Optional[np.ndarray]:
        """Voltage grid used for training."""
        return self._phi_applied.copy() if self._phi_applied is not None else None

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted
