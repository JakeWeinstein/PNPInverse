"""PyTorch ResNet-MLP surrogate model for BV kinetics I-V curves.

Architecture:
    Input(4) -> Linear(4,128) -> LayerNorm -> SiLU
    -> 4x ResBlock(128)
    -> Linear(128,64) -> LayerNorm -> SiLU
    -> Linear(64,44)

Each ResBlock:
    x -> Linear(128,128) -> LayerNorm -> SiLU -> Linear(128,128) -> LayerNorm
    + residual -> SiLU

Input:  [log10(k0_1), log10(k0_2), alpha_1, alpha_2]
Output: 44 values (22 CD + 22 PC concatenated)

Implements the same public API as BVSurrogateModel:
    fit(), predict(), predict_batch(), n_eta, phi_applied, is_fitted, training_bounds
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _check_torch() -> None:
    """Raise ImportError with install instructions if PyTorch is missing."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for NNSurrogateModel but is not installed.\n"
            "Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


# ---------------------------------------------------------------------------
# Z-score normalizer (computed from training data, stored alongside model)
# ---------------------------------------------------------------------------

@dataclass
class ZScoreNormalizer:
    """Z-score normalizer: x_norm = (x - mean) / std.

    Attributes
    ----------
    mean : np.ndarray
        Per-feature mean computed from training data.
    std : np.ndarray
        Per-feature standard deviation (clamped to >= 1e-15).
    """

    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def from_data(data: np.ndarray) -> "ZScoreNormalizer":
        """Compute normalizer statistics from training data.

        Parameters
        ----------
        data : np.ndarray of shape (N, D)
            Training data matrix.

        Returns
        -------
        ZScoreNormalizer
        """
        mean = data.mean(axis=0).astype(np.float64)
        std = np.maximum(data.std(axis=0).astype(np.float64), 1e-15)
        return ZScoreNormalizer(mean=mean, std=std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert z-score normalization."""
        return data * self.std + self.mean

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Serialize to dict of numpy arrays."""
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def from_dict(d: Dict[str, np.ndarray]) -> "ZScoreNormalizer":
        """Deserialize from dict of numpy arrays."""
        return ZScoreNormalizer(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# ResBlock and ResNetMLP (defined only when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class ResBlock(nn.Module):
        """Residual block: two linear layers with LayerNorm and SiLU.

        x -> Linear -> LayerNorm -> SiLU -> Linear -> LayerNorm + x -> SiLU
        """

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.ln1 = nn.LayerNorm(dim)
            self.fc2 = nn.Linear(dim, dim)
            self.ln2 = nn.LayerNorm(dim)
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.act(self.ln1(self.fc1(x)))
            out = self.ln2(self.fc2(out))
            return self.act(out + residual)

    class ResNetMLP(nn.Module):
        """ResNet-style MLP surrogate network.

        Architecture:
            Linear(n_in, hidden) -> LayerNorm -> SiLU
            -> n_blocks x ResBlock(hidden)
            -> Linear(hidden, hidden//2) -> LayerNorm -> SiLU
            -> Linear(hidden//2, n_out)
        """

        def __init__(
            self,
            n_in: int = 4,
            n_out: int = 44,
            hidden: int = 128,
            n_blocks: int = 4,
        ) -> None:
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(n_in, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            )
            self.res_blocks = nn.Sequential(
                *[ResBlock(hidden) for _ in range(n_blocks)]
            )
            self.output_layer = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, n_out),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_layer(x)
            h = self.res_blocks(h)
            return self.output_layer(h)


# ---------------------------------------------------------------------------
# NNSurrogateModel — same API as BVSurrogateModel
# ---------------------------------------------------------------------------

class NNSurrogateModel:
    """PyTorch ResNet-MLP surrogate model mapping BV parameters to I-V curves.

    After fitting, predicts current_density(eta) and peroxide_current(eta)
    as functions of (k0_1, k0_2, alpha_1, alpha_2).

    Implements the same public API as ``BVSurrogateModel``:
    ``fit()``, ``predict()``, ``predict_batch()``, ``n_eta``,
    ``phi_applied``, ``is_fitted``, ``training_bounds``.

    Parameters
    ----------
    hidden : int
        Hidden layer width (default 128).
    n_blocks : int
        Number of residual blocks (default 4).
    seed : int
        Random seed for weight initialization.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        hidden: int = 128,
        n_blocks: int = 4,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        _check_torch()
        self._hidden = hidden
        self._n_blocks = n_blocks
        self._seed = seed
        self._device = device
        self._model: Optional[ResNetMLP] = None
        self._input_normalizer: Optional[ZScoreNormalizer] = None
        self._output_normalizer: Optional[ZScoreNormalizer] = None
        self._phi_applied: Optional[np.ndarray] = None
        self._n_eta: int = 0
        self._is_fitted: bool = False
        self.training_bounds: Optional[Dict[str, Tuple[float, float]]] = None

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
        epochs: int = 5000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 500,
        batch_size: int | None = None,
        val_parameters: np.ndarray | None = None,
        val_cd: np.ndarray | None = None,
        val_pc: np.ndarray | None = None,
        verbose: bool = True,
        warm_start_state_dict: dict | None = None,
    ) -> "NNSurrogateModel":
        """Fit the ResNet-MLP from training data.

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
        epochs : int
            Maximum number of training epochs.
        lr : float
            Learning rate for AdamW.
        weight_decay : float
            Weight decay for AdamW.
        patience : int
            Early stopping patience (epochs without validation improvement).
        batch_size : int or None
            Mini-batch size. None uses full batch.
        val_parameters, val_cd, val_pc : optional
            Validation data. If None, 15% of training data is split off.
        verbose : bool
            Print training progress.
        warm_start_state_dict : dict or None
            If provided, load this state dict into the model after
            construction, overwriting random initialization. Used for
            ISMO warm-start retraining with analytically corrected weights.

        Returns
        -------
        self
        """
        _check_torch()

        N = parameters.shape[0]
        self._n_eta = phi_applied.shape[0]
        self._phi_applied = phi_applied.copy()
        n_out = 2 * self._n_eta

        if parameters.shape[1] != 4:
            raise ValueError(f"Expected parameters with 4 columns, got shape {parameters.shape}")
        if current_density.shape != (parameters.shape[0], self._n_eta):
            raise ValueError(f"current_density shape mismatch: expected ({parameters.shape[0]}, {self._n_eta}), got {current_density.shape}")
        if peroxide_current.shape != (parameters.shape[0], self._n_eta):
            raise ValueError(f"peroxide_current shape mismatch: expected ({parameters.shape[0]}, {self._n_eta}), got {peroxide_current.shape}")

        # Store training bounds
        self.training_bounds = {
            "k0_1": (float(parameters[:, 0].min()), float(parameters[:, 0].max())),
            "k0_2": (float(parameters[:, 1].min()), float(parameters[:, 1].max())),
            "alpha_1": (float(parameters[:, 2].min()), float(parameters[:, 2].max())),
            "alpha_2": (float(parameters[:, 3].min()), float(parameters[:, 3].max())),
        }

        # Auto-split validation if not provided
        if val_parameters is None:
            rng = np.random.default_rng(self._seed + 7777)
            n_val = max(1, int(N * 0.15))
            perm = rng.permutation(N)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            val_parameters = parameters[val_idx]
            val_cd = current_density[val_idx]
            val_pc = peroxide_current[val_idx]
            parameters = parameters[train_idx]
            current_density = current_density[train_idx]
            peroxide_current = peroxide_current[train_idx]
            N = parameters.shape[0]

        # Build log-space inputs
        X_log = parameters.copy()
        X_log[:, 0] = np.log10(np.maximum(X_log[:, 0], 1e-30))
        X_log[:, 1] = np.log10(np.maximum(X_log[:, 1], 1e-30))

        # Build output matrix: concat CD and PC
        Y = np.concatenate([current_density, peroxide_current], axis=1)

        # Compute normalizers
        self._input_normalizer = ZScoreNormalizer.from_data(X_log)
        self._output_normalizer = ZScoreNormalizer.from_data(Y)

        X_norm = self._input_normalizer.transform(X_log)
        Y_norm = self._output_normalizer.transform(Y)

        # Validation data
        X_val_log = val_parameters.copy()
        X_val_log[:, 0] = np.log10(np.maximum(X_val_log[:, 0], 1e-30))
        X_val_log[:, 1] = np.log10(np.maximum(X_val_log[:, 1], 1e-30))
        X_val_norm = self._input_normalizer.transform(X_val_log)
        Y_val = np.concatenate([val_cd, val_pc], axis=1)
        Y_val_norm = self._output_normalizer.transform(Y_val)

        # Convert to tensors
        device = torch.device(self._device)
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y_norm, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
        Y_val_t = torch.tensor(Y_val_norm, dtype=torch.float32, device=device)

        # Build model
        torch.manual_seed(self._seed)
        self._model = ResNetMLP(
            n_in=4, n_out=n_out,
            hidden=self._hidden, n_blocks=self._n_blocks,
        ).to(device)

        if warm_start_state_dict is not None:
            self._model.load_state_dict(warm_start_state_dict)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=1e-6,
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        if batch_size is None or batch_size >= N:
            batch_size = N

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self._model.train()
            perm_t = torch.randperm(N, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm_t[start : start + batch_size]
                xb = X_t[idx]
                yb = Y_t[idx]

                pred = self._model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = epoch_loss / n_batches

            # --- Validate ---
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_val_t)
                val_loss = torch.nn.functional.mse_loss(val_pred, Y_val_t).item()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % 500 == 0 or epoch == 1):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:>5d}/{epochs}  "
                    f"train_loss={train_loss:.6e}  val_loss={val_loss:.6e}  "
                    f"best_val={best_val_loss:.6e}  lr={current_lr:.2e}",
                    flush=True,
                )

            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"  Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)",
                        flush=True,
                    )
                break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

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

        X = self._transform_inputs(parameters)
        device = torch.device(self._device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        self._model.eval()
        with torch.no_grad():
            Y_norm = self._model(X_t).cpu().numpy()

        # Inverse-transform output
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
    # predict_torch() — differentiable forward pass for autograd
    # -----------------------------------------------------------------

    def predict_torch(self, x_logspace: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning a differentiable torch tensor.

        Parameters
        ----------
        x_logspace : torch.Tensor of shape (4,), requires_grad=True
            [log10(k0_1), log10(k0_2), alpha_1, alpha_2] in log-space.
            Note: k0 values are ALREADY in log10 space (matching optimizer space).

        Returns
        -------
        torch.Tensor of shape (2*n_eta,)
            Concatenated [current_density(n_eta), peroxide_current(n_eta)]
            in physical units, with autograd graph intact.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        _check_torch()

        # Cache z-score normalizer tensors on first call.
        # Store both float32 (for normal inference) and float64 (for autograd)
        # to avoid repeated casting.
        if not hasattr(self, "_input_mean_t"):
            self._input_mean_t = torch.tensor(
                self._input_normalizer.mean, dtype=torch.float32,
            )
            self._input_std_t = torch.tensor(
                self._input_normalizer.std, dtype=torch.float32,
            )
            self._out_mean_t = torch.tensor(
                self._output_normalizer.mean, dtype=torch.float32,
            )
            self._out_std_t = torch.tensor(
                self._output_normalizer.std, dtype=torch.float32,
            )
            self._input_mean_t64 = torch.tensor(
                self._input_normalizer.mean, dtype=torch.float64,
            )
            self._input_std_t64 = torch.tensor(
                self._input_normalizer.std, dtype=torch.float64,
            )
            self._out_mean_t64 = torch.tensor(
                self._output_normalizer.mean, dtype=torch.float64,
            )
            self._out_std_t64 = torch.tensor(
                self._output_normalizer.std, dtype=torch.float64,
            )

        if x_logspace.requires_grad:
            # Stay in float64 throughout to preserve gradient accuracy
            self._model.double()
            try:
                z = (x_logspace - self._input_mean_t64) / self._input_std_t64
                y_norm = self._model(z.unsqueeze(0)).squeeze(0)
                y = y_norm * self._out_std_t64 + self._out_mean_t64
                return y
            finally:
                # Restore model back to float32 for normal inference
                self._model.float()
        else:
            # Normal inference path: use float32
            x_f32 = x_logspace.float()
            z = (x_f32 - self._input_mean_t) / self._input_std_t
            y_norm = self._model(z.unsqueeze(0)).squeeze(0)
            y = y_norm * self._out_std_t + self._out_mean_t
            return y.double()

    # -----------------------------------------------------------------
    # Properties (API-compatible with BVSurrogateModel)
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

    # -----------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state to a directory.

        Creates:
            path/model.pt         - PyTorch state dict
            path/normalizers.npz  - input/output normalizer stats
            path/metadata.npz     - phi_applied, training_bounds, config

        Parameters
        ----------
        path : str
            Output directory.
        """
        if not self._is_fitted:
            raise ValueError("Cannot save an unfitted model. Call fit() first.")

        _check_torch()
        os.makedirs(path, exist_ok=True)

        # Model weights
        torch.save(self._model.state_dict(), os.path.join(path, "model.pt"))

        # Normalizers
        np.savez(
            os.path.join(path, "normalizers.npz"),
            input_mean=self._input_normalizer.mean,
            input_std=self._input_normalizer.std,
            output_mean=self._output_normalizer.mean,
            output_std=self._output_normalizer.std,
        )

        # Metadata
        bounds_keys = list(self.training_bounds.keys()) if self.training_bounds else []
        bounds_vals = (
            np.array([list(self.training_bounds[k]) for k in bounds_keys])
            if self.training_bounds
            else np.array([])
        )
        np.savez(
            os.path.join(path, "metadata.npz"),
            phi_applied=self._phi_applied,
            n_eta=self._n_eta,
            hidden=self._hidden,
            n_blocks=self._n_blocks,
            seed=self._seed,
            bounds_keys=bounds_keys,
            bounds_vals=bounds_vals,
        )

        print(f"NNSurrogateModel saved to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NNSurrogateModel":
        """Load a saved NNSurrogateModel from a directory.

        Parameters
        ----------
        path : str
            Directory containing model.pt, normalizers.npz, metadata.npz.
        device : str
            PyTorch device.

        Returns
        -------
        NNSurrogateModel
            Loaded, fitted model.
        """
        _check_torch()

        # Load metadata
        meta = np.load(os.path.join(path, "metadata.npz"), allow_pickle=True)
        n_eta = int(meta["n_eta"])
        hidden = int(meta["hidden"])
        n_blocks = int(meta["n_blocks"])
        seed = int(meta["seed"])
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

        # Create model instance
        model = cls(hidden=hidden, n_blocks=n_blocks, seed=seed, device=device)
        model._n_eta = n_eta
        model._phi_applied = phi_applied.copy()
        model._input_normalizer = input_normalizer
        model._output_normalizer = output_normalizer
        model.training_bounds = training_bounds

        # Build and load network
        n_out = 2 * n_eta
        dev = torch.device(device)
        model._model = ResNetMLP(
            n_in=4, n_out=n_out, hidden=hidden, n_blocks=n_blocks,
        ).to(dev)
        state = torch.load(
            os.path.join(path, "model.pt"),
            map_location=dev,
            weights_only=True,
        )
        model._model.load_state_dict(state)
        model._model.eval()
        model._is_fitted = True

        print(f"NNSurrogateModel loaded from: {path} (n_eta={n_eta})")
        return model
