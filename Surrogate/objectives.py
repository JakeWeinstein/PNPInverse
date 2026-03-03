"""Surrogate-based objective functions for BV kinetics inference.

Provides drop-in replacements for PDE-based objectives that use the
RBF surrogate model instead.  Gradients are computed via central finite
differences (8 surrogate evaluations for 4 parameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Surrogate.surrogate_model import BVSurrogateModel


class SurrogateObjective:
    """Surrogate-based multi-observable objective with FD gradient.

    The objective is::

        J = 0.5 * sum((cd_sim - cd_target)^2)
          + secondary_weight * 0.5 * sum((pc_sim - pc_target)^2)

    where cd = current_density, pc = peroxide_current.

    The control vector x layout is (by default)::

        x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2]

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

        return float(J_cd + self.secondary_weight * J_pc)

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
        """Compute gradient via central finite differences.

        Parameters
        ----------
        x : np.ndarray of shape (4,)
            Control vector.

        Returns
        -------
        np.ndarray of shape (4,)
            Gradient dJ/dx.
        """
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
        J = self.objective(x)
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
        """Central FD gradient w.r.t. [alpha_1, alpha_2]."""
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
        J = self.objective(x)
        g = self.gradient(x)
        return J, g

    @property
    def n_evals(self) -> int:
        return self._n_evals
