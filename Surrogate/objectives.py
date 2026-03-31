"""Surrogate-based objective functions for BV kinetics inference.

Provides drop-in replacements for PDE-based objectives that use the
RBF surrogate model instead.  Gradients are computed via central finite
differences (8 surrogate evaluations for 4 parameters) or, when the
surrogate supports ``predict_torch()``, via PyTorch autograd (single
forward + backward pass).

Classes
-------
SurrogateObjective
    Full 4-parameter joint objective.
AlphaOnlySurrogateObjective
    2-parameter objective with k0 fixed.
ReactionBlockSurrogateObjective
    2-parameter objective for a single reaction's (k0, alpha) pair,
    used by Block Coordinate Descent.
SubsetSurrogateObjective
    Full 4-parameter objective on a voltage subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Surrogate.surrogate_model import BVSurrogateModel


# ---------------------------------------------------------------------------
# Autograd detection helper
# ---------------------------------------------------------------------------

def _has_autograd(surrogate) -> bool:
    """Return True if the surrogate supports NN-style autograd gradients.

    The surrogate must have a callable ``predict_torch`` **and** must not
    explicitly opt out via ``supports_autograd = False`` (e.g. GP models
    whose ``predict_torch`` uses Z-score I/O rather than the log-space-in /
    physical-out convention the autograd objective path expects).
    """
    if not (hasattr(surrogate, "predict_torch") and callable(surrogate.predict_torch)):
        return False
    # Allow models to explicitly declare incompatibility
    return getattr(surrogate, "supports_autograd", True)


class SurrogateObjective:
    """Surrogate-based multi-observable objective with FD gradient.

    The objective is::

        J = 0.5 * sum((cd_sim - cd_target)^2)
          + secondary_weight * 0.5 * sum((pc_sim - pc_target)^2)

    where cd = current_density, pc = peroxide_current.

    The control vector x layout is (by default)::

        x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2]

    When the surrogate supports ``predict_torch()`` (NN/ensemble models),
    gradients are computed via PyTorch autograd instead of finite differences.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd : np.ndarray
        Target current density I-V curve.
    target_pc : np.ndarray
        Target peroxide current I-V curve.
    secondary_weight : float
        Weight on the peroxide current objective term.
    fd_step : float
        Finite difference step size for gradient computation.
    log_space_k0 : bool
        If True (default), x[0:2] are log10(k0).
    bounds : list of tuple or None
        (lower, upper) for each component of x.
    """

    def __init__(
        self,
        surrogate: BVSurrogateModel,
        target_cd: np.ndarray,
        target_pc: np.ndarray,
        secondary_weight: float = 1.0,
        fd_step: float = 1e-5,
        log_space_k0: bool = True,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        self.surrogate = surrogate
        self.target_cd = np.asarray(target_cd, dtype=float)
        self.target_pc = np.asarray(target_pc, dtype=float)
        # Mask for valid (non-NaN) target entries
        self._valid_cd = ~np.isnan(self.target_cd)
        self._valid_pc = ~np.isnan(self.target_pc)
        self.secondary_weight = secondary_weight
        self.fd_step = fd_step
        self.log_space_k0 = log_space_k0
        self.bounds = bounds
        self._n_evals = 0
        self._use_autograd = _has_autograd(surrogate)

    def _x_to_params(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert optimizer x-vector to (k0_1, k0_2, alpha_1, alpha_2)."""
        x = np.asarray(x, dtype=float)
        if self.log_space_k0:
            k0_1 = 10.0 ** x[0]
            k0_2 = 10.0 ** x[1]
        else:
            k0_1 = x[0]
            k0_2 = x[1]
        alpha_1 = x[2]
        alpha_2 = x[3]
        return k0_1, k0_2, alpha_1, alpha_2

    def _compute_objective_at_params(
        self, k0_1: float, k0_2: float, alpha_1: float, alpha_2: float,
    ) -> float:
        """Evaluate the combined objective at given physical parameters."""
        pred = self.surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_sim = pred["current_density"]
        pc_sim = pred["peroxide_current"]

        # Mask out NaN targets
        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]

        J_cd = 0.5 * np.sum(cd_diff ** 2)
        J_pc = 0.5 * np.sum(pc_diff ** 2)

        J = float(J_cd + self.secondary_weight * J_pc)
        if not np.isfinite(J):
            return np.inf
        return J

    # -----------------------------------------------------------------
    # Autograd gradient path
    # -----------------------------------------------------------------

    def _autograd_objective_and_gradient(
        self, x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute J and dJ/dx via PyTorch autograd (single forward + backward)."""
        import torch

        x_t = torch.tensor(
            np.asarray(x, dtype=np.float64), dtype=torch.float64, requires_grad=True,
        )

        # Forward through surrogate (x_t is already in log-space for k0)
        y = self.surrogate.predict_torch(x_t)  # shape (2*n_eta,)
        n_eta = len(y) // 2
        cd_sim = y[:n_eta]
        pc_sim = y[n_eta:]

        # Cache target tensors on first call
        if not hasattr(self, "_target_cd_t"):
            self._target_cd_t = torch.tensor(
                self.target_cd[self._valid_cd], dtype=torch.float64,
            )
            self._target_pc_t = torch.tensor(
                self.target_pc[self._valid_pc], dtype=torch.float64,
            )
            self._valid_cd_idx = torch.tensor(
                np.where(self._valid_cd)[0], dtype=torch.long,
            )
            self._valid_pc_idx = torch.tensor(
                np.where(self._valid_pc)[0], dtype=torch.long,
            )

        cd_diff = cd_sim[self._valid_cd_idx] - self._target_cd_t
        pc_diff = pc_sim[self._valid_pc_idx] - self._target_pc_t

        J = (
            0.5 * torch.sum(cd_diff ** 2)
            + self.secondary_weight * 0.5 * torch.sum(pc_diff ** 2)
        )

        J.backward()

        J_val = float(J.detach())
        grad_val = x_t.grad.numpy().copy()

        self._n_evals += 1
        return J_val, grad_val

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective J(x).

        Parameters
        ----------
        x : np.ndarray of shape (4,)
            Control vector [log10(k0_1), log10(k0_2), alpha_1, alpha_2].

        Returns
        -------
        float
            Objective value.
        """
        k0_1, k0_2, alpha_1, alpha_2 = self._x_to_params(x)
        self._n_evals += 1
        return self._compute_objective_at_params(k0_1, k0_2, alpha_1, alpha_2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient via autograd (if available) or central FD.

        Parameters
        ----------
        x : np.ndarray of shape (4,)
            Control vector.

        Returns
        -------
        np.ndarray of shape (4,)
            Gradient dJ/dx.
        """
        if self._use_autograd:
            _, g = self._autograd_objective_and_gradient(x)
            return g

        x = np.asarray(x, dtype=float)
        n = len(x)
        grad = np.zeros(n, dtype=float)
        h = self.fd_step

        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * h)

        return grad

    def objective_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute objective and gradient in one call.

        Parameters
        ----------
        x : np.ndarray of shape (4,)
            Control vector.

        Returns
        -------
        (float, np.ndarray of shape (4,))
            (J, dJ/dx).
        """
        if self._use_autograd:
            return self._autograd_objective_and_gradient(x)
        # Compute objective value without incrementing _n_evals, since
        # gradient() will call self.objective() internally for FD stencil.
        k0_1, k0_2, alpha_1, alpha_2 = self._x_to_params(x)
        J = self._compute_objective_at_params(k0_1, k0_2, alpha_1, alpha_2)
        g = self.gradient(x)
        return J, g

    @property
    def n_evals(self) -> int:
        """Number of surrogate evaluations performed."""
        return self._n_evals


class AlphaOnlySurrogateObjective:
    """Surrogate objective that optimizes only alpha, with k0 fixed.

    The control vector is x = [alpha_1, alpha_2].

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd : np.ndarray
        Target current density I-V curve.
    target_pc : np.ndarray
        Target peroxide current I-V curve.
    fixed_k0 : tuple or list of float
        Fixed (k0_1, k0_2) values.
    secondary_weight : float
        Weight on the peroxide current objective term.
    fd_step : float
        Finite difference step size for gradient.
    """

    def __init__(
        self,
        surrogate: BVSurrogateModel,
        target_cd: np.ndarray,
        target_pc: np.ndarray,
        fixed_k0: Sequence[float],
        secondary_weight: float = 1.0,
        fd_step: float = 1e-5,
    ):
        self.surrogate = surrogate
        self.target_cd = np.asarray(target_cd, dtype=float)
        self.target_pc = np.asarray(target_pc, dtype=float)
        self._valid_cd = ~np.isnan(self.target_cd)
        self._valid_pc = ~np.isnan(self.target_pc)
        self.fixed_k0 = (float(fixed_k0[0]), float(fixed_k0[1]))
        self.secondary_weight = secondary_weight
        self.fd_step = fd_step
        self._n_evals = 0
        self._use_autograd = _has_autograd(surrogate)

    # -----------------------------------------------------------------
    # Autograd gradient path
    # -----------------------------------------------------------------

    def _autograd_objective_and_gradient(
        self, x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute J and dJ/dx via PyTorch autograd for alpha-only optimization."""
        import torch

        x_t = torch.tensor(
            np.asarray(x, dtype=np.float64), dtype=torch.float64, requires_grad=True,
        )

        # Build full 4D input: [log10(k0_1), log10(k0_2), alpha_1, alpha_2]
        # k0 values are fixed constants (detached, no gradient)
        log_k0_1 = torch.tensor(
            np.log10(max(self.fixed_k0[0], 1e-30)), dtype=torch.float64,
        )
        log_k0_2 = torch.tensor(
            np.log10(max(self.fixed_k0[1], 1e-30)), dtype=torch.float64,
        )
        x_full = torch.cat([log_k0_1.unsqueeze(0), log_k0_2.unsqueeze(0), x_t])

        y = self.surrogate.predict_torch(x_full)
        n_eta = len(y) // 2
        cd_sim = y[:n_eta]
        pc_sim = y[n_eta:]

        # Cache target tensors on first call
        if not hasattr(self, "_target_cd_t"):
            self._target_cd_t = torch.tensor(
                self.target_cd[self._valid_cd], dtype=torch.float64,
            )
            self._target_pc_t = torch.tensor(
                self.target_pc[self._valid_pc], dtype=torch.float64,
            )
            self._valid_cd_idx = torch.tensor(
                np.where(self._valid_cd)[0], dtype=torch.long,
            )
            self._valid_pc_idx = torch.tensor(
                np.where(self._valid_pc)[0], dtype=torch.long,
            )

        cd_diff = cd_sim[self._valid_cd_idx] - self._target_cd_t
        pc_diff = pc_sim[self._valid_pc_idx] - self._target_pc_t

        J = (
            0.5 * torch.sum(cd_diff ** 2)
            + self.secondary_weight * 0.5 * torch.sum(pc_diff ** 2)
        )

        J.backward()

        J_val = float(J.detach())
        grad_val = x_t.grad.numpy().copy()

        self._n_evals += 1
        return J_val, grad_val

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective J(alpha_1, alpha_2) with k0 fixed."""
        x = np.asarray(x, dtype=float)
        alpha_1, alpha_2 = float(x[0]), float(x[1])
        k0_1, k0_2 = self.fixed_k0

        pred = self.surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_sim = pred["current_density"]
        pc_sim = pred["peroxide_current"]

        # Mask out NaN targets
        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]

        J_cd = 0.5 * np.sum(cd_diff ** 2)
        J_pc = 0.5 * np.sum(pc_diff ** 2)

        self._n_evals += 1
        return float(J_cd + self.secondary_weight * J_pc)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Central FD gradient w.r.t. [alpha_1, alpha_2], or autograd."""
        if self._use_autograd:
            _, g = self._autograd_objective_and_gradient(x)
            return g

        x = np.asarray(x, dtype=float)
        n = len(x)
        grad = np.zeros(n, dtype=float)
        h = self.fd_step

        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * h)

        return grad

    def objective_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute objective and gradient."""
        if self._use_autograd:
            return self._autograd_objective_and_gradient(x)
        # Compute objective value without incrementing _n_evals, since
        # gradient() will call self.objective() internally for FD stencil.
        x = np.asarray(x, dtype=float)
        alpha_1, alpha_2 = float(x[0]), float(x[1])
        k0_1, k0_2 = self.fixed_k0
        pred = self.surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_diff = pred["current_density"][self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pred["peroxide_current"][self._valid_pc] - self.target_pc[self._valid_pc]
        J = float(0.5 * np.sum(cd_diff ** 2) + self.secondary_weight * 0.5 * np.sum(pc_diff ** 2))
        g = self.gradient(x)
        return J, g

    @property
    def n_evals(self) -> int:
        return self._n_evals


class ReactionBlockSurrogateObjective:
    """Surrogate objective for a single reaction's parameters.

    Optimizes [log10(k0_r), alpha_r] for reaction *r* while holding the
    other reaction's (k0, alpha) fixed.  Used by the Block Coordinate
    Descent (BCD) algorithm.

    The control vector is ``x = [log10(k0_r), alpha_r]`` (2-dimensional).

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd : np.ndarray
        Target current density I-V curve.
    target_pc : np.ndarray
        Target peroxide current I-V curve.
    reaction_index : int
        0 = reaction 1 free (k0_1, alpha_1), 1 = reaction 2 free (k0_2, alpha_2).
    fixed_k0_other : float
        k0 value for the *other* (fixed) reaction.
    fixed_alpha_other : float
        alpha value for the *other* (fixed) reaction.
    secondary_weight : float
        Weight on the peroxide current objective term.
    fd_step : float
        Finite difference step size for gradient computation.
    """

    def __init__(
        self,
        surrogate: BVSurrogateModel,
        target_cd: np.ndarray,
        target_pc: np.ndarray,
        reaction_index: int,
        fixed_k0_other: float,
        fixed_alpha_other: float,
        secondary_weight: float = 1.0,
        fd_step: float = 1e-5,
    ):
        if reaction_index not in (0, 1):
            raise ValueError(
                f"reaction_index must be 0 or 1, got {reaction_index}"
            )
        self.surrogate = surrogate
        self.target_cd = np.asarray(target_cd, dtype=float)
        self.target_pc = np.asarray(target_pc, dtype=float)
        self._valid_cd = ~np.isnan(self.target_cd)
        self._valid_pc = ~np.isnan(self.target_pc)
        self.reaction_index = reaction_index
        self.fixed_k0_other = float(fixed_k0_other)
        self.fixed_alpha_other = float(fixed_alpha_other)
        self.secondary_weight = secondary_weight
        self.fd_step = fd_step
        self._n_evals = 0
        self._use_autograd = _has_autograd(surrogate)

    def _x_to_full_params(
        self, x: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Map 2D control [log10(k0_r), alpha_r] to full (k0_1, k0_2, alpha_1, alpha_2)."""
        x = np.asarray(x, dtype=float)
        k0_r = 10.0 ** x[0]
        alpha_r = float(x[1])
        if self.reaction_index == 0:
            return k0_r, self.fixed_k0_other, alpha_r, self.fixed_alpha_other
        else:
            return self.fixed_k0_other, k0_r, self.fixed_alpha_other, alpha_r

    # -----------------------------------------------------------------
    # Autograd gradient path
    # -----------------------------------------------------------------

    def _autograd_objective_and_gradient(
        self, x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute J and dJ/dx via PyTorch autograd for block optimization."""
        import torch

        x_t = torch.tensor(
            np.asarray(x, dtype=np.float64), dtype=torch.float64, requires_grad=True,
        )

        # Build full 4D input from 2D control vector + fixed constants
        log_k0_other = torch.tensor(
            np.log10(max(self.fixed_k0_other, 1e-30)), dtype=torch.float64,
        )
        alpha_other = torch.tensor(self.fixed_alpha_other, dtype=torch.float64)

        if self.reaction_index == 0:
            # x_t = [log10(k0_1), alpha_1], fixed = k0_2, alpha_2
            x_full = torch.stack([
                x_t[0], log_k0_other, x_t[1], alpha_other,
            ])
        else:
            # x_t = [log10(k0_2), alpha_2], fixed = k0_1, alpha_1
            x_full = torch.stack([
                log_k0_other, x_t[0], alpha_other, x_t[1],
            ])

        y = self.surrogate.predict_torch(x_full)
        n_eta = len(y) // 2
        cd_sim = y[:n_eta]
        pc_sim = y[n_eta:]

        # Cache target tensors on first call
        if not hasattr(self, "_target_cd_t"):
            self._target_cd_t = torch.tensor(
                self.target_cd[self._valid_cd], dtype=torch.float64,
            )
            self._target_pc_t = torch.tensor(
                self.target_pc[self._valid_pc], dtype=torch.float64,
            )
            self._valid_cd_idx = torch.tensor(
                np.where(self._valid_cd)[0], dtype=torch.long,
            )
            self._valid_pc_idx = torch.tensor(
                np.where(self._valid_pc)[0], dtype=torch.long,
            )

        cd_diff = cd_sim[self._valid_cd_idx] - self._target_cd_t
        pc_diff = pc_sim[self._valid_pc_idx] - self._target_pc_t

        J = (
            0.5 * torch.sum(cd_diff ** 2)
            + self.secondary_weight * 0.5 * torch.sum(pc_diff ** 2)
        )

        J.backward()

        J_val = float(J.detach())
        grad_val = x_t.grad.numpy().copy()

        self._n_evals += 1
        return J_val, grad_val

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective J(x) for the free reaction block.

        Parameters
        ----------
        x : np.ndarray of shape (2,)
            Control vector [log10(k0_r), alpha_r].

        Returns
        -------
        float
            Objective value.
        """
        k0_1, k0_2, alpha_1, alpha_2 = self._x_to_full_params(x)
        pred = self.surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_sim = pred["current_density"]
        pc_sim = pred["peroxide_current"]

        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]

        j_cd = 0.5 * np.sum(cd_diff ** 2)
        j_pc = 0.5 * np.sum(pc_diff ** 2)

        self._n_evals += 1
        return float(j_cd + self.secondary_weight * j_pc)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Central FD gradient w.r.t. [log10(k0_r), alpha_r], or autograd.

        Parameters
        ----------
        x : np.ndarray of shape (2,)
            Control vector.

        Returns
        -------
        np.ndarray of shape (2,)
            Gradient dJ/dx.
        """
        if self._use_autograd:
            _, g = self._autograd_objective_and_gradient(x)
            return g

        x = np.asarray(x, dtype=float)
        n = len(x)
        grad = np.zeros(n, dtype=float)
        h = self.fd_step

        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * h)

        return grad

    def objective_and_gradient(
        self, x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute objective and gradient in one call.

        Parameters
        ----------
        x : np.ndarray of shape (2,)
            Control vector.

        Returns
        -------
        (float, np.ndarray of shape (2,))
            (J, dJ/dx).
        """
        if self._use_autograd:
            return self._autograd_objective_and_gradient(x)
        g = self.gradient(x)
        # Compute objective without incrementing _n_evals (gradient already counted)
        k0_1, k0_2, alpha_1, alpha_2 = self._x_to_full_params(x)
        pred = self.surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_diff = pred["current_density"][self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pred["peroxide_current"][self._valid_pc] - self.target_pc[self._valid_pc]
        j = float(0.5 * np.sum(cd_diff ** 2) + self.secondary_weight * 0.5 * np.sum(pc_diff ** 2))
        return j, g

    @property
    def n_evals(self) -> int:
        """Number of surrogate evaluations performed."""
        return self._n_evals


class SubsetSurrogateObjective:
    """Surrogate objective on a subset of voltage points.

    Evaluates the full surrogate grid but computes the loss only on the
    voltage indices in *subset_idx*.  Useful for phase-2 shallow-range
    optimization where the surrogate spans a wider grid than the target.

    Parameters
    ----------
    surrogate
        Any surrogate model with a ``.predict(k0_1, k0_2, alpha_1, alpha_2)``
        method returning ``{'current_density': ..., 'peroxide_current': ...}``.
    target_cd : np.ndarray
        Target current density at the subset voltages.
    target_pc : np.ndarray
        Target peroxide current at the subset voltages.
    subset_idx : np.ndarray of int
        Indices into the full surrogate output to compare against targets.
    secondary_weight : float
        Weight on the peroxide current objective term.
    fd_step : float
        Finite difference step size for gradient computation.
    log_space_k0 : bool
        If True (default), x[0:2] are log10(k0).
    """

    def __init__(
        self,
        surrogate,
        target_cd: np.ndarray,
        target_pc: np.ndarray,
        subset_idx: np.ndarray,
        secondary_weight: float = 1.0,
        fd_step: float = 1e-5,
        log_space_k0: bool = True,
    ):
        self.surrogate = surrogate
        self.target_cd = np.asarray(target_cd, dtype=float)
        self.target_pc = np.asarray(target_pc, dtype=float)
        self.subset_idx = np.asarray(subset_idx, dtype=int)
        if len(self.target_cd) != len(self.subset_idx):
            raise ValueError(
                f"target_cd length ({len(self.target_cd)}) != subset_idx length ({len(self.subset_idx)})"
            )
        if len(self.target_pc) != len(self.subset_idx):
            raise ValueError(
                f"target_pc length ({len(self.target_pc)}) != subset_idx length ({len(self.subset_idx)})"
            )
        self._valid_cd = ~np.isnan(self.target_cd)
        self._valid_pc = ~np.isnan(self.target_pc)
        self.secondary_weight = secondary_weight
        # fd_step=1e-5 is applied uniformly to log-k0 and linear-alpha dimensions;
        # acceptable for typical parameter ranges but may need tuning for extreme scales.
        self.fd_step = fd_step
        self.log_space_k0 = log_space_k0
        self._n_evals = 0
        self._use_autograd = _has_autograd(surrogate)

    def _x_to_params(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert optimizer x-vector to (k0_1, k0_2, alpha_1, alpha_2)."""
        x = np.asarray(x, dtype=float)
        if self.log_space_k0:
            k0_1 = 10.0 ** x[0]
            k0_2 = 10.0 ** x[1]
        else:
            k0_1 = x[0]
            k0_2 = x[1]
        return k0_1, k0_2, float(x[2]), float(x[3])

    # -----------------------------------------------------------------
    # Autograd gradient path
    # -----------------------------------------------------------------

    def _autograd_objective_and_gradient(
        self, x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute J and dJ/dx via PyTorch autograd on the voltage subset."""
        import torch

        x_t = torch.tensor(
            np.asarray(x, dtype=np.float64), dtype=torch.float64, requires_grad=True,
        )

        y = self.surrogate.predict_torch(x_t)  # shape (2*n_eta,)
        n_eta = len(y) // 2

        # Index into full output at subset positions (torch indexing supports grad)
        subset_idx_t = torch.tensor(self.subset_idx, dtype=torch.long)
        cd_sim = y[subset_idx_t]
        pc_sim = y[n_eta + subset_idx_t]

        # Cache target tensors on first call
        if not hasattr(self, "_target_cd_t"):
            self._target_cd_t = torch.tensor(
                self.target_cd[self._valid_cd], dtype=torch.float64,
            )
            self._target_pc_t = torch.tensor(
                self.target_pc[self._valid_pc], dtype=torch.float64,
            )
            self._valid_cd_idx = torch.tensor(
                np.where(self._valid_cd)[0], dtype=torch.long,
            )
            self._valid_pc_idx = torch.tensor(
                np.where(self._valid_pc)[0], dtype=torch.long,
            )

        cd_diff = cd_sim[self._valid_cd_idx] - self._target_cd_t
        pc_diff = pc_sim[self._valid_pc_idx] - self._target_pc_t

        J = (
            0.5 * torch.sum(cd_diff ** 2)
            + self.secondary_weight * 0.5 * torch.sum(pc_diff ** 2)
        )

        J.backward()

        J_val = float(J.detach())
        grad_val = x_t.grad.numpy().copy()

        self._n_evals += 1
        return J_val, grad_val

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """Evaluate objective on the subset of voltage points.

        Parameters
        ----------
        x : np.ndarray of shape (4,)
            Control vector [log10(k0_1), log10(k0_2), alpha_1, alpha_2].

        Returns
        -------
        float
        """
        k0_1, k0_2, a1, a2 = self._x_to_params(x)
        pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
        cd_sim = pred["current_density"][self.subset_idx]
        pc_sim = pred["peroxide_current"][self.subset_idx]

        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]

        j_cd = 0.5 * np.sum(cd_diff ** 2)
        j_pc = 0.5 * np.sum(pc_diff ** 2)

        self._n_evals += 1
        return float(j_cd + self.secondary_weight * j_pc)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Central FD gradient, or autograd if available.

        Parameters
        ----------
        x : np.ndarray of shape (4,)

        Returns
        -------
        np.ndarray of shape (4,)
        """
        if self._use_autograd:
            _, g = self._autograd_objective_and_gradient(x)
            return g

        x = np.asarray(x, dtype=float)
        n = len(x)
        grad = np.zeros(n, dtype=float)
        h = self.fd_step
        for i in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (self.objective(xp) - self.objective(xm)) / (2 * h)
        return grad

    def objective_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute objective and gradient in one call."""
        if self._use_autograd:
            return self._autograd_objective_and_gradient(x)
        g = self.gradient(x)
        # Compute objective without incrementing _n_evals (gradient already counted)
        k0_1, k0_2, a1, a2 = self._x_to_params(x)
        pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
        cd_sim = pred["current_density"][self.subset_idx]
        pc_sim = pred["peroxide_current"][self.subset_idx]
        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]
        j = float(0.5 * np.sum(cd_diff ** 2) + self.secondary_weight * 0.5 * np.sum(pc_diff ** 2))
        return j, g

    @property
    def n_evals(self) -> int:
        """Number of surrogate evaluations performed."""
        return self._n_evals
