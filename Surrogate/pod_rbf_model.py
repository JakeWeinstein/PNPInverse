"""POD-decomposed RBF surrogate model for BV kinetics I-V curves.

Uses Proper Orthogonal Decomposition (SVD) to reduce output dimensionality,
then fits per-mode RBF interpolators with optimised smoothing.

Architecture:
    1. SVD on training outputs Y, retain modes for 99.9% variance
    2. Per-mode RBF interpolator (thin_plate_spline) with smoothing
       optimised via a grid search over 30 log-spaced values
    3. Optional log-scale PC transformation

Implements the same public API as BVSurrogateModel:
    fit(), predict(), predict_batch(), n_eta, phi_applied, is_fitted,
    training_bounds
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PODRBFConfig:
    """Configuration for the POD-RBF surrogate model.

    Attributes
    ----------
    variance_threshold : float
        Fraction of total variance to retain (default 0.999 = 99.9%).
    kernel : str
        RBF kernel type for scipy.interpolate.RBFInterpolator.
    degree : int
        Polynomial degree for the RBF.
    log_space_k0 : bool
        If True, transform k0_1 and k0_2 to log10 space before fitting.
    normalize_inputs : bool
        If True, z-score normalize the (transformed) inputs.
    optimize_smoothing : bool
        If True, optimize per-mode smoothing via cross-validation.
    n_smoothing_candidates : int
        Number of log-spaced smoothing values to test.
    smoothing_range : tuple of float
        (min, max) range for smoothing parameter search.
    default_smoothing : float
        Fallback smoothing if optimization is disabled.
    log_transform_pc : bool
        If True, apply log1p transform to peroxide_current before SVD.
    max_modes : int or None
        Maximum number of POD modes to retain (None = no limit).
    """

    variance_threshold: float = 0.999
    kernel: str = "thin_plate_spline"
    degree: int = 1
    log_space_k0: bool = True
    normalize_inputs: bool = True
    optimize_smoothing: bool = True
    n_smoothing_candidates: int = 30
    smoothing_range: Tuple[float, float] = (1e-8, 1e0)
    default_smoothing: float = 0.0
    log_transform_pc: bool = False
    max_modes: int | None = None


# ---------------------------------------------------------------------------
# POD-RBF Surrogate Model
# ---------------------------------------------------------------------------

class PODRBFSurrogateModel:
    """POD-decomposed RBF surrogate model for BV kinetics I-V curves.

    Reduces the output dimensionality via SVD, then fits a separate RBF
    interpolator for each retained POD coefficient.  Smoothing can be
    optimised per mode via leave-one-out cross-validation on a grid.

    Parameters
    ----------
    config : PODRBFConfig or None
        Model configuration.
    """

    def __init__(self, config: PODRBFConfig | None = None) -> None:
        self.config = config or PODRBFConfig()
        self._rbf_models: List[RBFInterpolator] = []
        self._U: Optional[np.ndarray] = None       # Left singular vectors (truncated)
        self._S: Optional[np.ndarray] = None       # Singular values (truncated)
        self._Vt: Optional[np.ndarray] = None      # Right singular vectors (truncated)
        self._Y_mean: Optional[np.ndarray] = None  # Mean of training outputs
        self._n_modes: int = 0
        self._phi_applied: Optional[np.ndarray] = None
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._n_eta: int = 0
        self._is_fitted: bool = False
        self.training_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        self._smoothing_values: List[float] = []

    # -----------------------------------------------------------------
    # Input transform
    # -----------------------------------------------------------------

    def _transform_inputs(self, parameters: np.ndarray) -> np.ndarray:
        """Apply log-space and normalization transforms.

        Parameters
        ----------
        parameters : np.ndarray (N, 4)

        Returns
        -------
        np.ndarray (N, 4)
        """
        X = parameters.copy()
        if self.config.log_space_k0:
            X[:, 0] = np.log10(np.maximum(X[:, 0], 1e-30))
            X[:, 1] = np.log10(np.maximum(X[:, 1], 1e-30))
        if self.config.normalize_inputs and self._input_mean is not None:
            X = (X - self._input_mean) / np.maximum(self._input_std, 1e-15)
        return X

    # -----------------------------------------------------------------
    # Output transform (optional log for PC)
    # -----------------------------------------------------------------

    def _transform_outputs(
        self,
        current_density: np.ndarray,
        peroxide_current: np.ndarray,
    ) -> np.ndarray:
        """Concatenate CD + PC into a single output matrix.

        If log_transform_pc is True, applies log1p to |PC| with sign
        preservation.

        Parameters
        ----------
        current_density : np.ndarray (N, n_eta)
        peroxide_current : np.ndarray (N, n_eta)

        Returns
        -------
        np.ndarray (N, 2*n_eta)
        """
        pc = peroxide_current.copy()
        if self.config.log_transform_pc:
            pc = np.sign(pc) * np.log1p(np.abs(pc))
        return np.concatenate([current_density, pc], axis=1)

    def _inverse_transform_outputs(
        self, Y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split output matrix back into CD and PC.

        Parameters
        ----------
        Y : np.ndarray (N, 2*n_eta)

        Returns
        -------
        (current_density, peroxide_current), each (N, n_eta)
        """
        n_eta = self._n_eta
        cd = Y[:, :n_eta]
        pc = Y[:, n_eta:]
        if self.config.log_transform_pc:
            pc = np.sign(pc) * (np.expm1(np.abs(pc)))
        return cd, pc

    # -----------------------------------------------------------------
    # Smoothing optimization
    # -----------------------------------------------------------------

    def _optimize_mode_smoothing(
        self,
        X: np.ndarray,
        coeffs: np.ndarray,
    ) -> float:
        """Find optimal smoothing for a single POD mode via LOO-CV on a grid.

        Parameters
        ----------
        X : np.ndarray (N, 4)
            Transformed input data.
        coeffs : np.ndarray (N,)
            POD coefficients for this mode.

        Returns
        -------
        float
            Best smoothing value.
        """
        lo, hi = self.config.smoothing_range
        candidates = np.logspace(
            np.log10(max(lo, 1e-15)),
            np.log10(hi),
            self.config.n_smoothing_candidates,
        )

        N = X.shape[0]
        best_sm = candidates[0]
        best_err = float("inf")

        for sm in candidates:
            # Leave-one-out cross-validation using sub-sampling for speed
            # If N > 100, use 5-fold CV instead of full LOO
            if N > 100:
                err = self._kfold_cv_error(X, coeffs, sm, n_folds=5)
            else:
                err = self._loo_cv_error(X, coeffs, sm)

            if err < best_err:
                best_err = err
                best_sm = sm

        return float(best_sm)

    def _loo_cv_error(
        self,
        X: np.ndarray,
        coeffs: np.ndarray,
        smoothing: float,
    ) -> float:
        """Leave-one-out cross-validation error for a single RBF mode.

        Parameters
        ----------
        X : np.ndarray (N, 4)
        coeffs : np.ndarray (N,)
        smoothing : float

        Returns
        -------
        float
            Mean squared LOO error.
        """
        N = X.shape[0]
        errors = np.zeros(N)

        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            rbf = RBFInterpolator(
                X[mask], coeffs[mask, np.newaxis],
                kernel=self.config.kernel,
                degree=self.config.degree,
                smoothing=smoothing,
            )
            pred = rbf(X[i:i+1]).ravel()
            errors[i] = (pred[0] - coeffs[i]) ** 2

        return float(errors.mean())

    def _kfold_cv_error(
        self,
        X: np.ndarray,
        coeffs: np.ndarray,
        smoothing: float,
        n_folds: int = 5,
    ) -> float:
        """K-fold cross-validation error for a single RBF mode.

        Parameters
        ----------
        X : np.ndarray (N, 4)
        coeffs : np.ndarray (N,)
        smoothing : float
        n_folds : int

        Returns
        -------
        float
            Mean squared CV error.
        """
        N = X.shape[0]
        indices = np.arange(N)
        fold_size = N // n_folds
        errors = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else N
            val_mask = np.zeros(N, dtype=bool)
            val_mask[start:end] = True
            train_mask = ~val_mask

            if train_mask.sum() < 2:
                continue

            rbf = RBFInterpolator(
                X[train_mask], coeffs[train_mask, np.newaxis],
                kernel=self.config.kernel,
                degree=self.config.degree,
                smoothing=smoothing,
            )
            pred = rbf(X[val_mask]).ravel()
            fold_errors = (pred - coeffs[val_mask]) ** 2
            errors.extend(fold_errors.tolist())

        return float(np.mean(errors)) if errors else float("inf")

    # -----------------------------------------------------------------
    # fit()
    # -----------------------------------------------------------------

    def fit(
        self,
        parameters: np.ndarray,
        current_density: np.ndarray,
        peroxide_current: np.ndarray,
        phi_applied: np.ndarray,
        verbose: bool = True,
    ) -> "PODRBFSurrogateModel":
        """Fit the POD-RBF surrogate from training data.

        Parameters
        ----------
        parameters : np.ndarray (N, 4)
            Training parameter samples [k0_1, k0_2, alpha_1, alpha_2].
        current_density : np.ndarray (N, n_eta)
            CD I-V curves.
        peroxide_current : np.ndarray (N, n_eta)
            PC I-V curves.
        phi_applied : np.ndarray (n_eta,)
            Voltage grid.
        verbose : bool
            Print progress.

        Returns
        -------
        self
        """
        N = parameters.shape[0]
        self._n_eta = phi_applied.shape[0]
        self._phi_applied = phi_applied.copy()

        assert parameters.shape == (N, 4)
        assert current_density.shape == (N, self._n_eta)
        assert peroxide_current.shape == (N, self._n_eta)

        # Store training bounds
        self.training_bounds = {
            "k0_1": (float(parameters[:, 0].min()), float(parameters[:, 0].max())),
            "k0_2": (float(parameters[:, 1].min()), float(parameters[:, 1].max())),
            "alpha_1": (float(parameters[:, 2].min()), float(parameters[:, 2].max())),
            "alpha_2": (float(parameters[:, 3].min()), float(parameters[:, 3].max())),
        }

        # Transform inputs
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

        # Transform outputs
        Y = self._transform_outputs(current_density, peroxide_current)
        self._Y_mean = Y.mean(axis=0)
        Y_centered = Y - self._Y_mean

        # SVD
        U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)

        # Determine number of modes for variance threshold
        cumulative_var = np.cumsum(S ** 2) / np.sum(S ** 2)
        n_modes = int(np.searchsorted(cumulative_var, self.config.variance_threshold) + 1)
        n_modes = min(n_modes, len(S))
        if self.config.max_modes is not None:
            n_modes = min(n_modes, self.config.max_modes)

        self._n_modes = n_modes
        self._U = U[:, :n_modes]
        self._S = S[:n_modes]
        self._Vt = Vt[:n_modes, :]

        if verbose:
            explained = cumulative_var[n_modes - 1] * 100 if n_modes > 0 else 0.0
            print(f"  POD: {n_modes} modes retain {explained:.2f}% variance "
                  f"(out of {len(S)} total)", flush=True)

        # POD coefficients: alpha_i = U_truncated[:, i] * S_i
        # Equivalently: coefficients = Y_centered @ Vt_truncated.T
        coefficients = Y_centered @ self._Vt.T  # (N, n_modes)

        # Fit per-mode RBF interpolators
        self._rbf_models = []
        self._smoothing_values = []

        for mode in range(n_modes):
            mode_coeffs = coefficients[:, mode]

            if self.config.optimize_smoothing:
                sm = self._optimize_mode_smoothing(X, mode_coeffs)
            else:
                sm = self.config.default_smoothing

            self._smoothing_values.append(sm)

            rbf = RBFInterpolator(
                X, mode_coeffs[:, np.newaxis],
                kernel=self.config.kernel,
                degree=self.config.degree,
                smoothing=sm,
            )
            self._rbf_models.append(rbf)

            if verbose and (mode < 5 or mode == n_modes - 1):
                print(f"    Mode {mode}: S={S[mode]:.4e}, smoothing={sm:.2e}",
                      flush=True)

        self._is_fitted = True
        if verbose:
            print(f"  POD-RBF fit complete: {n_modes} modes, {N} samples",
                  flush=True)

        return self

    # -----------------------------------------------------------------
    # predict() and predict_batch()
    # -----------------------------------------------------------------

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
            'current_density' : np.ndarray (n_eta,)
            'peroxide_current' : np.ndarray (n_eta,)
            'phi_applied' : np.ndarray (n_eta,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]], dtype=float)
        result = self.predict_batch(params)
        return {
            "current_density": result["current_density"][0],
            "peroxide_current": result["peroxide_current"][0],
            "phi_applied": result["phi_applied"],
        }

    def predict_batch(self, parameters: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict I-V curves for multiple parameter sets.

        Parameters
        ----------
        parameters : np.ndarray (M, 4)
            Each row: [k0_1, k0_2, alpha_1, alpha_2].

        Returns
        -------
        dict with keys:
            'current_density' : np.ndarray (M, n_eta)
            'peroxide_current' : np.ndarray (M, n_eta)
            'phi_applied' : np.ndarray (n_eta,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = self._transform_inputs(parameters)
        M = parameters.shape[0]

        # Predict POD coefficients for each mode
        coefficients = np.zeros((M, self._n_modes), dtype=float)
        for mode in range(self._n_modes):
            coefficients[:, mode] = self._rbf_models[mode](X).ravel()

        # Reconstruct output
        Y_centered = coefficients @ self._Vt  # (M, 2*n_eta)
        Y = Y_centered + self._Y_mean

        cd, pc = self._inverse_transform_outputs(Y)

        return {
            "current_density": cd,
            "peroxide_current": pc,
            "phi_applied": self._phi_applied.copy(),
        }

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

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

    @property
    def n_modes(self) -> int:
        """Number of retained POD modes."""
        return self._n_modes

    @property
    def singular_values(self) -> Optional[np.ndarray]:
        """Truncated singular values."""
        return self._S.copy() if self._S is not None else None
