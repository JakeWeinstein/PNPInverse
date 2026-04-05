"""Gaussian Process surrogate model for BV kinetics I-V curves (GPyTorch).

Architecture:
    44 independent exact GPs (22 CD + 22 PC), one per voltage point.
    Each GP: 4D input (log10(k0_1), log10(k0_2), alpha_1, alpha_2) -> 1D output.
    Kernel: ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))

Implements the same public API as BVSurrogateModel / NNSurrogateModel:
    fit(), predict(), predict_batch(), n_eta, phi_applied, is_fitted, training_bounds

Additional GP-specific methods:
    predict_with_uncertainty(), predict_batch_with_uncertainty()
    predict_torch() -- differentiable path for autograd gradients
    gradient_at() -- convenience method for dJ/dx via autograd
"""

from __future__ import annotations

import logging
import os
from math import log10
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import gpytorch

    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False

from Surrogate.nn_model import ZScoreNormalizer

logger = logging.getLogger(__name__)


def _check_torch() -> None:
    """Raise ImportError with install instructions if PyTorch is missing."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GPSurrogateModel but is not installed.\n"
            "Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


def _check_gpytorch() -> None:
    """Raise ImportError with install instructions if GPyTorch is missing."""
    if not _GPYTORCH_AVAILABLE:
        raise ImportError(
            "GPyTorch is required for GPSurrogateModel but is not installed.\n"
            "Install it with:\n"
            "  pip install gpytorch"
        )


# ---------------------------------------------------------------------------
# GPyTorch model class (defined only when gpytorch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE and _GPYTORCH_AVAILABLE:

    class ExactGPModel(gpytorch.models.ExactGP):
        """Single-output exact GP with Matern 5/2 ARD kernel.

        Parameters
        ----------
        train_x : torch.Tensor of shape (N, 4)
            Training inputs (Z-score normalized).
        train_y : torch.Tensor of shape (N,)
            Training targets (Z-score normalized, single output dim).
        likelihood : gpytorch.likelihoods.GaussianLikelihood
            Observation noise likelihood.
        """

        def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
        ) -> None:
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=4)
            )

        def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# Single-GP fitting function (top-level for joblib pickling)
# ---------------------------------------------------------------------------


def _fit_single_gp(
    train_x_np: np.ndarray,
    train_y_np: np.ndarray,
    n_iters: int = 200,
    lr: float = 0.1,
    early_stop_tol: float = 1e-6,
    early_stop_patience: int = 20,
) -> Tuple[dict, dict, float]:
    """Fit a single exact GP and return serializable state dicts.

    This function is defined at module level so that ``joblib.Parallel``
    can pickle it.  Returns state dicts (not live model objects) to avoid
    issues with pickling GPyTorch models across processes.

    Parameters
    ----------
    train_x_np : np.ndarray of shape (N, 4)
        Z-score normalized training inputs.
    train_y_np : np.ndarray of shape (N,)
        Z-score normalized training targets (single output dim).
    n_iters : int
        Maximum number of Adam optimization iterations.
    lr : float
        Learning rate for Adam.
    early_stop_tol : float
        Minimum loss change to count as improvement.
    early_stop_patience : int
        Stop if no improvement for this many iterations.

    Returns
    -------
    (model_state, likelihood_state, final_loss)
        Serializable state dicts and final marginal log-likelihood loss.
    """
    _check_torch()
    _check_gpytorch()

    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float("inf")
    iters_no_improve = 0

    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        if loss_val < best_loss - early_stop_tol:
            best_loss = loss_val
            iters_no_improve = 0
        else:
            iters_no_improve += 1

        if iters_no_improve >= early_stop_patience:
            break

    model.eval()
    likelihood.eval()

    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    likelihood_state = {k: v.cpu().clone() for k, v in likelihood.state_dict().items()}

    return model_state, likelihood_state, best_loss


# ---------------------------------------------------------------------------
# GPSurrogateModel -- same API as BVSurrogateModel / NNSurrogateModel
# ---------------------------------------------------------------------------


class GPSurrogateModel:
    """Gaussian Process surrogate model mapping BV parameters to I-V curves.

    After fitting, predicts current_density(eta) and peroxide_current(eta)
    as functions of (k0_1, k0_2, alpha_1, alpha_2).

    Uses 44 independent exact GPs (22 CD + 22 PC), one per voltage point,
    each with a Matern 5/2 ARD kernel.

    Implements the same public API as ``BVSurrogateModel`` and
    ``NNSurrogateModel``, plus uncertainty quantification and autograd
    gradient support.

    Parameters
    ----------
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, device: str = "cpu") -> None:
        _check_torch()
        _check_gpytorch()
        self._device = device
        self._gp_models: Optional[List[Tuple[Any, Any]]] = None  # [(model, likelihood), ...]
        self._input_normalizer: Optional[ZScoreNormalizer] = None
        self._output_normalizer: Optional[ZScoreNormalizer] = None
        self._phi_applied: Optional[np.ndarray] = None
        self._n_eta: int = 0
        self._is_fitted: bool = False
        self._training_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        # Training data (required for exact GP prediction)
        self._train_x: Optional[torch.Tensor] = None
        self._train_y_norm: Optional[np.ndarray] = None
        # Torch tensors for normalizer stats (for autograd path)
        self._input_mean_t: Optional[torch.Tensor] = None
        self._input_std_t: Optional[torch.Tensor] = None
        self._output_mean_t: Optional[torch.Tensor] = None
        self._output_std_t: Optional[torch.Tensor] = None
        # GP predict_torch uses Z-score I/O, not the log-space-in /
        # physical-out convention that the NN autograd path expects.
        self.supports_autograd: bool = False

    # -----------------------------------------------------------------
    # Input transform: physical params -> log-space + z-score
    # -----------------------------------------------------------------

    def _transform_inputs(self, parameters: np.ndarray) -> np.ndarray:
        """Transform raw parameters [k0_1, k0_2, alpha_1, alpha_2] to
        model input space: log10(k0) + z-score normalization.

        Parameters
        ----------
        parameters : np.ndarray of shape (N, 4)

        Returns
        -------
        np.ndarray of shape (N, 4)
        """
        X = parameters.copy()
        X[:, 0] = np.log10(np.maximum(X[:, 0], 1e-30))
        X[:, 1] = np.log10(np.maximum(X[:, 1], 1e-30))
        if self._input_normalizer is not None:
            X = self._input_normalizer.transform(X)
        return X

    # -----------------------------------------------------------------
    # fit()
    # -----------------------------------------------------------------

    def fit(
        self,
        parameters: np.ndarray,
        current_density: np.ndarray,
        peroxide_current: np.ndarray,
        phi_applied: np.ndarray,
        *,
        n_iters: int = 200,
        lr: float = 0.1,
        early_stop_tol: float = 1e-6,
        early_stop_patience: int = 20,
        n_jobs: int = -1,
        verbose: bool = True,
    ) -> "GPSurrogateModel":
        """Fit 44 independent exact GPs from training data.

        Parameters
        ----------
        parameters : np.ndarray (N, 4)
            Training parameter samples [k0_1, k0_2, alpha_1, alpha_2].
        current_density : np.ndarray (N, n_eta)
            Training CD curves.
        peroxide_current : np.ndarray (N, n_eta)
            Training PC curves.
        phi_applied : np.ndarray (n_eta,)
            Voltage grid.
        n_iters : int
            Maximum Adam iterations per GP (default 200).
        lr : float
            Adam learning rate (default 0.1).
        early_stop_tol : float
            Minimum loss improvement to reset patience counter.
        early_stop_patience : int
            Stop if no improvement for this many iterations.
        n_jobs : int
            Number of parallel jobs for joblib (-1 = all cores).
        verbose : bool
            Print training progress.

        Returns
        -------
        self
        """
        _check_torch()
        _check_gpytorch()

        N = parameters.shape[0]
        self._n_eta = phi_applied.shape[0]
        self._phi_applied = phi_applied.copy()
        n_out = 2 * self._n_eta

        assert parameters.shape == (N, 4), f"Expected (N,4), got {parameters.shape}"
        assert current_density.shape == (N, self._n_eta), (
            f"Expected ({N},{self._n_eta}), got {current_density.shape}"
        )
        assert peroxide_current.shape == (N, self._n_eta), (
            f"Expected ({N},{self._n_eta}), got {peroxide_current.shape}"
        )

        # Store training bounds
        self._training_bounds = {
            "k0_1": (float(parameters[:, 0].min()), float(parameters[:, 0].max())),
            "k0_2": (float(parameters[:, 1].min()), float(parameters[:, 1].max())),
            "alpha_1": (float(parameters[:, 2].min()), float(parameters[:, 2].max())),
            "alpha_2": (float(parameters[:, 3].min()), float(parameters[:, 3].max())),
        }

        # Build log-space inputs
        X_log = parameters.copy()
        X_log[:, 0] = np.log10(np.maximum(X_log[:, 0], 1e-30))
        X_log[:, 1] = np.log10(np.maximum(X_log[:, 1], 1e-30))

        # Build output matrix: concat CD and PC
        Y = np.concatenate([current_density, peroxide_current], axis=1)

        # Compute normalizers
        self._input_normalizer = ZScoreNormalizer.from_data(X_log)
        self._output_normalizer = ZScoreNormalizer.from_data(Y)

        X_norm = self._input_normalizer.transform(X_log).astype(np.float32)
        Y_norm = self._output_normalizer.transform(Y).astype(np.float32)

        # Store training data (needed for exact GP at prediction time)
        self._train_x = torch.tensor(X_norm, dtype=torch.float32)
        self._train_y_norm = Y_norm

        # Store torch tensors for normalizer stats (autograd path)
        self._input_mean_t = torch.tensor(
            self._input_normalizer.mean, dtype=torch.float32
        )
        self._input_std_t = torch.tensor(
            self._input_normalizer.std, dtype=torch.float32
        )
        self._output_mean_t = torch.tensor(
            self._output_normalizer.mean, dtype=torch.float32
        )
        self._output_std_t = torch.tensor(
            self._output_normalizer.std, dtype=torch.float32
        )

        # Fit 44 independent GPs in parallel via joblib
        if verbose:
            print(f"  Fitting {n_out} independent GPs (N={N}, n_iters={n_iters}, lr={lr})...")

        try:
            from joblib import Parallel, delayed
        except ImportError:
            raise ImportError(
                "joblib is required for parallel GP fitting but is not installed.\n"
                "Install it with: pip install joblib"
            )

        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_fit_single_gp)(
                X_norm, Y_norm[:, d],
                n_iters=n_iters, lr=lr,
                early_stop_tol=early_stop_tol,
                early_stop_patience=early_stop_patience,
            )
            for d in range(n_out)
        )

        # Reconstruct live model objects from state dicts
        self._gp_models = []
        final_losses = []
        for d, (model_state, likelihood_state, final_loss) in enumerate(results):
            train_x_t = self._train_x.clone()
            train_y_t = torch.tensor(Y_norm[:, d], dtype=torch.float32)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.load_state_dict(likelihood_state)

            model = ExactGPModel(train_x_t, train_y_t, likelihood)
            model.load_state_dict(model_state)

            model.eval()
            likelihood.eval()

            self._gp_models.append((model, likelihood))
            final_losses.append(final_loss)

        if verbose:
            losses = np.array(final_losses)
            print(f"  GP fitting complete.")
            print(f"  Final neg-MLL: mean={losses.mean():.4f}, "
                  f"min={losses.min():.4f}, max={losses.max():.4f}")
            # Check for non-finite hyperparameters
            n_bad = 0
            for d, (m, _) in enumerate(self._gp_models):
                for name, param in m.named_parameters():
                    if not torch.isfinite(param).all():
                        n_bad += 1
                        logger.warning(f"  GP {d}: non-finite param '{name}'")
            if n_bad == 0:
                print(f"  All GP hyperparameters are finite.")
            else:
                print(f"  WARNING: {n_bad} non-finite hyperparameters detected!")

        self._is_fitted = True
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

        _check_torch()
        _check_gpytorch()

        X = self._transform_inputs(parameters)
        X_t = torch.tensor(X, dtype=torch.float32)

        means = []
        n_train = self._train_x.shape[0] if self._train_x is not None else 0
        cholesky_size = max(n_train + 1, 3000)
        for model, likelihood in self._gp_models:
            model.eval()
            likelihood.eval()
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.max_cholesky_size(cholesky_size),
            ):
                posterior = likelihood(model(X_t))
                means.append(posterior.mean.numpy())

        # Stack means -> shape (M, 44)
        Y_norm = np.column_stack(means)

        # Inverse Z-score transform
        Y = self._output_normalizer.inverse_transform(Y_norm)

        n_eta = self._n_eta
        cd = Y[:, :n_eta]
        pc = Y[:, n_eta:]

        return {
            "current_density": cd,
            "peroxide_current": pc,
            "phi_applied": self._phi_applied.copy(),
        }

    # -----------------------------------------------------------------
    # predict_with_uncertainty() and predict_batch_with_uncertainty()
    # -----------------------------------------------------------------

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
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]], dtype=float)
        result = self.predict_batch_with_uncertainty(params)
        return {
            "current_density": result["current_density"][0],
            "peroxide_current": result["peroxide_current"][0],
            "phi_applied": result["phi_applied"],
            "current_density_std": result["current_density_std"][0],
            "peroxide_current_std": result["peroxide_current_std"][0],
        }

    def predict_batch_with_uncertainty(
        self, parameters: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Predict mean + standard deviation from the GP posterior.

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
            'current_density_std' : np.ndarray (M, n_eta)
            'peroxide_current_std' : np.ndarray (M, n_eta)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        _check_torch()
        _check_gpytorch()

        X = self._transform_inputs(parameters)
        X_t = torch.tensor(X, dtype=torch.float32)

        means = []
        variances = []
        n_train = self._train_x.shape[0] if self._train_x is not None else 0
        cholesky_size = max(n_train + 1, 3000)
        for model, likelihood in self._gp_models:
            model.eval()
            likelihood.eval()
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.max_cholesky_size(cholesky_size),
            ):
                posterior = likelihood(model(X_t))
                means.append(posterior.mean.numpy())
                variances.append(posterior.variance.numpy())

        # Stack -> shape (M, 44)
        Y_norm_mean = np.column_stack(means)
        Y_norm_var = np.column_stack(variances)

        # Inverse-transform mean
        Y_mean = self._output_normalizer.inverse_transform(Y_norm_mean)

        # Transform variance back to original scale:
        # If y_norm = (y - mu) / sigma, then var(y) = sigma^2 * var(y_norm)
        # So std(y) = sigma * std(y_norm) = sigma * sqrt(var(y_norm))
        output_std = self._output_normalizer.std  # shape (44,)
        Y_std = np.sqrt(Y_norm_var) * output_std[np.newaxis, :]

        n_eta = self._n_eta
        cd_mean = Y_mean[:, :n_eta]
        pc_mean = Y_mean[:, n_eta:]
        cd_std = Y_std[:, :n_eta]
        pc_std = Y_std[:, n_eta:]

        return {
            "current_density": cd_mean,
            "peroxide_current": pc_mean,
            "phi_applied": self._phi_applied.copy(),
            "current_density_std": cd_std,
            "peroxide_current_std": pc_std,
        }

    # -----------------------------------------------------------------
    # Autograd gradient support
    # -----------------------------------------------------------------

    def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable prediction: input (M,4) tensor -> output (M,44) tensor.

        Input must be in TRANSFORMED space (log10 k0 + Z-score normalized).
        Output is in Z-score normalized space.
        Caller handles normalization and inverse transforms.

        Uses the latent GP mean (not the noisy likelihood prediction) to
        keep the gradient clean.

        Parameters
        ----------
        x : torch.Tensor of shape (M, 4)
            Transformed, normalized inputs with requires_grad=True.

        Returns
        -------
        torch.Tensor of shape (M, 44)
            Z-score normalized predictions (differentiable).
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        means = []
        n_train = self._train_x.shape[0] if self._train_x is not None else 0
        cholesky_size = max(n_train + 1, 3000)
        for model, likelihood in self._gp_models:
            model.eval()
            likelihood.eval()
            # Use model(x) NOT likelihood(model(x)) -- we want the latent GP
            # mean, not the noisy prediction, for clean gradients.
            with gpytorch.settings.max_cholesky_size(cholesky_size):
                posterior = model(x)
            means.append(posterior.mean)  # shape (M,)
        return torch.stack(means, dim=1)  # shape (M, 44)

    def gradient_at(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
        target_cd: np.ndarray,
        target_pc: np.ndarray,
        secondary_weight: float = 1.0,
    ) -> np.ndarray:
        """Compute dJ/dx via autograd where J is the surrogate objective.

        The objective is:
            J = 0.5 * sum((cd_pred - target_cd)^2)
              + secondary_weight * 0.5 * sum((pc_pred - target_pc)^2)

        Parameters
        ----------
        k0_1, k0_2, alpha_1, alpha_2 : float
            BV kinetics parameters (physical space).
        target_cd : np.ndarray of shape (n_eta,)
            Target current density curve.
        target_pc : np.ndarray of shape (n_eta,)
            Target peroxide current curve.
        secondary_weight : float
            Weight on the peroxide current term (default 1.0).

        Returns
        -------
        np.ndarray of shape (4,)
            Gradient w.r.t. [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        _check_torch()
        _check_gpytorch()

        # Build differentiable input tensor in log-space
        x = torch.tensor(
            [[log10(max(k0_1, 1e-30)), log10(max(k0_2, 1e-30)), alpha_1, alpha_2]],
            dtype=torch.float32,
            requires_grad=True,
        )

        # Z-score normalize (torch ops to stay in autograd graph)
        x_norm = (x - self._input_mean_t) / self._input_std_t

        # Differentiable GP prediction in normalized space
        y_norm = self.predict_torch(x_norm)  # (1, 44)

        # Inverse Z-score transform (torch ops)
        y = y_norm * self._output_std_t + self._output_mean_t

        n_eta = self._n_eta
        cd = y[0, :n_eta]
        pc = y[0, n_eta:]

        target_cd_t = torch.tensor(target_cd, dtype=torch.float32)
        target_pc_t = torch.tensor(target_pc, dtype=torch.float32)

        J = 0.5 * torch.sum((cd - target_cd_t) ** 2) + \
            secondary_weight * 0.5 * torch.sum((pc - target_pc_t) ** 2)

        J.backward()

        return x.grad.numpy()[0]  # shape (4,)

    # -----------------------------------------------------------------
    # Properties (API-compatible with BVSurrogateModel / NNSurrogateModel)
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
    def training_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Training parameter bounds."""
        return self._training_bounds

    # -----------------------------------------------------------------
    # Serialization: save() and load()
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save GP model state to a directory.

        Creates:
            path/metadata.npz       - phi_applied, n_eta, training_bounds
            path/normalizers.npz    - input/output normalizer stats
            path/gp_model_XX.pt     - torch state dict for each GP
            path/likelihood_XX.pt   - torch state dict for each likelihood
            path/train_x.pt        - training inputs (needed for exact GP)
            path/train_y.npz       - training targets (44 dims)

        Parameters
        ----------
        path : str
            Output directory.
        """
        if not self._is_fitted:
            raise ValueError("Cannot save an unfitted model. Call fit() first.")

        _check_torch()
        os.makedirs(path, exist_ok=True)

        n_out = 2 * self._n_eta

        # Metadata
        bounds_keys = list(self._training_bounds.keys()) if self._training_bounds else []
        bounds_vals = (
            np.array([list(self._training_bounds[k]) for k in bounds_keys])
            if self._training_bounds
            else np.array([])
        )
        np.savez(
            os.path.join(path, "metadata.npz"),
            phi_applied=self._phi_applied,
            n_eta=self._n_eta,
            bounds_keys=bounds_keys,
            bounds_vals=bounds_vals,
        )

        # Normalizers
        np.savez(
            os.path.join(path, "normalizers.npz"),
            input_mean=self._input_normalizer.mean,
            input_std=self._input_normalizer.std,
            output_mean=self._output_normalizer.mean,
            output_std=self._output_normalizer.std,
        )

        # Training data (required for exact GP prediction)
        torch.save(self._train_x, os.path.join(path, "train_x.pt"))
        np.savez(
            os.path.join(path, "train_y.npz"),
            train_y_norm=self._train_y_norm,
        )

        # GP models and likelihoods
        for d in range(n_out):
            model, likelihood = self._gp_models[d]
            torch.save(
                model.state_dict(),
                os.path.join(path, f"gp_model_{d:02d}.pt"),
            )
            torch.save(
                likelihood.state_dict(),
                os.path.join(path, f"likelihood_{d:02d}.pt"),
            )

        print(f"GPSurrogateModel saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GPSurrogateModel":
        """Load a saved GPSurrogateModel from a directory.

        Parameters
        ----------
        path : str
            Directory containing saved GP model artifacts.
        device : str
            PyTorch device.

        Returns
        -------
        GPSurrogateModel
            Loaded, fitted model.
        """
        _check_torch()
        _check_gpytorch()

        # Load metadata
        meta = np.load(os.path.join(path, "metadata.npz"), allow_pickle=True)
        n_eta = int(meta["n_eta"])
        phi_applied = meta["phi_applied"]

        # Reconstruct training bounds
        bounds_keys = list(meta["bounds_keys"])
        bounds_vals = meta["bounds_vals"]
        training_bounds = None
        if len(bounds_keys) > 0:
            training_bounds = {
                str(k): (float(bounds_vals[i, 0]), float(bounds_vals[i, 1]))
                for i, k in enumerate(bounds_keys)
            }

        # Load normalizers
        norm_data = np.load(os.path.join(path, "normalizers.npz"))
        input_normalizer = ZScoreNormalizer(
            mean=norm_data["input_mean"],
            std=norm_data["input_std"],
        )
        output_normalizer = ZScoreNormalizer(
            mean=norm_data["output_mean"],
            std=norm_data["output_std"],
        )

        # Load training data
        dev = torch.device(device)
        train_x = torch.load(
            os.path.join(path, "train_x.pt"),
            map_location=dev,
            weights_only=True,
        )
        train_y_data = np.load(os.path.join(path, "train_y.npz"))
        train_y_norm = train_y_data["train_y_norm"]

        n_out = 2 * n_eta

        # Create model instance
        model_obj = cls(device=device)
        model_obj._n_eta = n_eta
        model_obj._phi_applied = phi_applied.copy()
        model_obj._input_normalizer = input_normalizer
        model_obj._output_normalizer = output_normalizer
        model_obj._training_bounds = training_bounds
        model_obj._train_x = train_x
        model_obj._train_y_norm = train_y_norm

        # Store torch tensors for normalizer stats (autograd path)
        model_obj._input_mean_t = torch.tensor(
            input_normalizer.mean, dtype=torch.float32
        )
        model_obj._input_std_t = torch.tensor(
            input_normalizer.std, dtype=torch.float32
        )
        model_obj._output_mean_t = torch.tensor(
            output_normalizer.mean, dtype=torch.float32
        )
        model_obj._output_std_t = torch.tensor(
            output_normalizer.std, dtype=torch.float32
        )

        # Reconstruct GP models from state dicts
        model_obj._gp_models = []
        for d in range(n_out):
            train_y_d = torch.tensor(train_y_norm[:, d], dtype=torch.float32)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood_state = torch.load(
                os.path.join(path, f"likelihood_{d:02d}.pt"),
                map_location=dev,
                weights_only=True,
            )
            likelihood.load_state_dict(likelihood_state)

            gp_model = ExactGPModel(train_x, train_y_d, likelihood)
            model_state = torch.load(
                os.path.join(path, f"gp_model_{d:02d}.pt"),
                map_location=dev,
                weights_only=True,
            )
            gp_model.load_state_dict(model_state)

            gp_model.eval()
            likelihood.eval()

            model_obj._gp_models.append((gp_model, likelihood))

        model_obj._is_fitted = True
        print(f"GPSurrogateModel loaded from: {path} (n_eta={n_eta})")
        return model_obj


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_gp_surrogate(path: str, device: str = "cpu") -> GPSurrogateModel:
    """Load a saved GP surrogate model.

    Parameters
    ----------
    path : str
        Directory containing saved GP model artifacts.
    device : str
        PyTorch device.

    Returns
    -------
    GPSurrogateModel
        Loaded, fitted model.
    """
    return GPSurrogateModel.load(path, device=device)
