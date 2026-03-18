"""Polynomial Chaos Expansion surrogate model for BV kinetics I-V curves.

Uses ChaosPy to build PCE surrogates with Sobol sensitivity analysis.
Fits 44 independent scalar PCEs (22 current density + 22 peroxide current,
one per voltage point) via least-squares regression with optional LOO
cross-validation for degree selection.

Architecture:
    1. Transform k0_1, k0_2 to log10 space (optional, default True)
    2. Build Uniform joint distribution in transformed space
    3. Generate Legendre polynomial basis with hyperbolic truncation
    4. Fit per-voltage-point PCE via cp.fit_regression
    5. Extract Sobol indices from PCE coefficients

Implements the same public API as BVSurrogateModel:
    fit(), predict(), predict_batch(), n_eta, phi_applied, is_fitted,
    training_bounds
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import chaospy as cp

    _CHAOSPY_AVAILABLE = True
except ImportError:
    _CHAOSPY_AVAILABLE = False


def _check_chaospy() -> None:
    """Raise ImportError with install instructions if ChaosPy is missing."""
    if not _CHAOSPY_AVAILABLE:
        raise ImportError(
            "ChaosPy is required for PCESurrogateModel but is not installed.\n"
            "Install it with:\n"
            "  pip install chaospy"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PCEConfig:
    """Configuration for the PCE surrogate model.

    Attributes
    ----------
    max_degree : int
        Maximum total polynomial degree.
    sparse_truncation : bool
        Use hyperbolic truncation (q < 1.0) to reduce basis size.
    q_norm : float
        Hyperbolic truncation parameter (0 < q <= 1).  Lower values
        produce sparser bases.
    fitting_method : str
        Regression method: "lstsq" for ordinary least squares.
    log_space_k0 : bool
        If True, transform k0_1 and k0_2 to log10 space before fitting.
    cross_validation : bool
        If True, use LOO error to select polynomial degree.
    degree_candidates : tuple of int
        Degrees to try when cross_validation is enabled.
    """

    max_degree: int = 6
    sparse_truncation: bool = True
    q_norm: float = 0.75
    fitting_method: str = "lstsq"
    log_space_k0: bool = True
    cross_validation: bool = True
    degree_candidates: tuple = (3, 4, 5, 6, 7, 8)


# ---------------------------------------------------------------------------
# PCE Surrogate Model
# ---------------------------------------------------------------------------


class PCESurrogateModel:
    """Polynomial Chaos Expansion surrogate for BV kinetics I-V curves.

    Builds per-voltage-point PCE expansions for current density and
    peroxide current.  Supports Sobol sensitivity analysis via the
    PCE coefficient structure.

    Parameters
    ----------
    config : PCEConfig or None
        Model configuration.
    """

    def __init__(self, config: Optional[PCEConfig] = None) -> None:
        _check_chaospy()
        self.config = config or PCEConfig()
        self._pce_cd: List = []  # Per-voltage PCE for current density
        self._pce_pc: List = []  # Per-voltage PCE for peroxide current
        self._joint_dist = None  # chaospy Joint distribution
        self._expansion = None  # chaospy polynomial expansion
        self._phi_applied: Optional[np.ndarray] = None
        self._n_eta: int = 0
        self._is_fitted: bool = False
        self._selected_degree: int = 0
        self._n_terms: int = 0
        self._sobol_indices: Optional[Dict] = None
        self.training_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        self._transformed_bounds: Optional[Dict[str, Tuple[float, float]]] = None

    # -----------------------------------------------------------------
    # Input transforms
    # -----------------------------------------------------------------

    def _transform_inputs(self, parameters: np.ndarray) -> np.ndarray:
        """Apply log-space transform to raw parameters.

        Parameters
        ----------
        parameters : np.ndarray of shape (N, 4)
            Columns: [k0_1, k0_2, alpha_1, alpha_2] in physical space.

        Returns
        -------
        np.ndarray of shape (N, 4)
            Transformed parameters.
        """
        X = parameters.copy().astype(float)
        if self.config.log_space_k0:
            X[:, 0] = np.log10(np.maximum(X[:, 0], 1e-30))
            X[:, 1] = np.log10(np.maximum(X[:, 1], 1e-30))
        return X

    def _transform_single(
        self, k0_1: float, k0_2: float, alpha_1: float, alpha_2: float
    ) -> np.ndarray:
        """Transform a single parameter set.  Returns shape (4, 1) for ChaosPy."""
        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]], dtype=float)
        X = self._transform_inputs(params)  # (1, 4)
        return X.T  # (4, 1) for chaospy

    # -----------------------------------------------------------------
    # Distribution and basis construction
    # -----------------------------------------------------------------

    def _build_distribution(self, bounds: Dict[str, Tuple[float, float]]):
        """Construct chaospy Joint distribution from transformed bounds.

        Parameters
        ----------
        bounds : dict
            Keys: 'k0_1', 'k0_2', 'alpha_1', 'alpha_2' with (min, max) values
            in physical space.

        Returns
        -------
        chaospy.Distribution
            Joint Uniform distribution in transformed space.
        """
        if self.config.log_space_k0:
            log_k0_1 = cp.Uniform(
                np.log10(max(bounds["k0_1"][0], 1e-30)),
                np.log10(max(bounds["k0_1"][1], 1e-30)),
            )
            log_k0_2 = cp.Uniform(
                np.log10(max(bounds["k0_2"][0], 1e-30)),
                np.log10(max(bounds["k0_2"][1], 1e-30)),
            )
        else:
            log_k0_1 = cp.Uniform(bounds["k0_1"][0], bounds["k0_1"][1])
            log_k0_2 = cp.Uniform(bounds["k0_2"][0], bounds["k0_2"][1])

        alpha_1 = cp.Uniform(bounds["alpha_1"][0], bounds["alpha_1"][1])
        alpha_2 = cp.Uniform(bounds["alpha_2"][0], bounds["alpha_2"][1])

        joint = cp.J(log_k0_1, log_k0_2, alpha_1, alpha_2)
        return joint

    def _build_basis(self, degree: int):
        """Generate polynomial expansion at the given degree.

        Parameters
        ----------
        degree : int
            Total polynomial degree.

        Returns
        -------
        chaospy polynomial expansion
        """
        if self.config.sparse_truncation:
            expansion = cp.generate_expansion(
                order=degree,
                dist=self._joint_dist,
                cross_truncation=self.config.q_norm,
            )
        else:
            expansion = cp.generate_expansion(
                order=degree,
                dist=self._joint_dist,
            )
        return expansion

    # -----------------------------------------------------------------
    # LOO cross-validation for degree selection
    # -----------------------------------------------------------------

    def _loo_error_hat_matrix(
        self, expansion, X_T: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute LOO error using the hat-matrix closed-form formula.

        For a linear regression y = Phi * c, the LOO prediction error is:
            e_loo_i = residual_i / (1 - h_ii)
        where H = Phi @ (Phi^T Phi)^{-1} @ Phi^T and h_ii = diag(H).

        Parameters
        ----------
        expansion : chaospy expansion
            Polynomial basis.
        X_T : np.ndarray of shape (n_dims, n_samples)
            Transformed input samples (ChaosPy convention).
        y : np.ndarray of shape (n_samples,)
            Output values for a single voltage point.

        Returns
        -------
        float
            Root mean squared LOO error.
        """
        # Evaluate basis polynomials at sample points -> design matrix
        Phi = np.array(expansion(*X_T)).T  # (n_samples, n_terms)

        n_samples, n_terms = Phi.shape

        if n_terms >= n_samples:
            # Under-determined system: cannot compute LOO reliably
            return float("inf")

        # Solve normal equations
        try:
            # Use pseudo-inverse for numerical stability
            PhiT_Phi = Phi.T @ Phi
            # Add small regularization for stability
            PhiT_Phi += 1e-12 * np.eye(n_terms)
            PhiT_Phi_inv = np.linalg.inv(PhiT_Phi)
        except np.linalg.LinAlgError:
            return float("inf")

        # Regression coefficients
        c = PhiT_Phi_inv @ (Phi.T @ y)

        # Residuals
        y_pred = Phi @ c
        residuals = y - y_pred

        # Hat matrix diagonal
        H = Phi @ PhiT_Phi_inv @ Phi.T
        h_diag = np.diag(H)

        # LOO errors
        denom = 1.0 - h_diag
        # Guard against division by zero (shouldn't happen if n_terms < n_samples)
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        loo_errors = residuals / denom

        return float(np.sqrt(np.mean(loo_errors ** 2)))

    def _select_degree(
        self, X_T: np.ndarray, y_representative: np.ndarray, verbose: bool
    ) -> int:
        """Select optimal polynomial degree via LOO cross-validation.

        Parameters
        ----------
        X_T : np.ndarray of shape (n_dims, n_samples)
            Transformed samples in ChaosPy convention.
        y_representative : np.ndarray of shape (n_samples,)
            Representative output (e.g., CD at median voltage index).
        verbose : bool
            Print progress.

        Returns
        -------
        int
            Selected polynomial degree.
        """
        best_degree = self.config.degree_candidates[0]
        best_loo = float("inf")

        if verbose:
            print("  Degree selection via LOO cross-validation:")

        for degree in self.config.degree_candidates:
            expansion = self._build_basis(degree)
            n_terms = len(expansion)
            loo_err = self._loo_error_hat_matrix(expansion, X_T, y_representative)

            if verbose:
                print(
                    f"    degree={degree}, terms={n_terms}, "
                    f"LOO RMSE={loo_err:.6e}"
                )

            if loo_err < best_loo:
                best_loo = loo_err
                best_degree = degree

        if verbose:
            print(f"  Selected degree: {best_degree}")

        return best_degree

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
    ) -> "PCESurrogateModel":
        """Fit PCE surrogates from training data.

        Fits 44 independent scalar PCEs: 22 for current density and
        22 for peroxide current (one per voltage point).

        Parameters
        ----------
        parameters : np.ndarray of shape (N, 4)
            Training parameter samples [k0_1, k0_2, alpha_1, alpha_2].
        current_density : np.ndarray of shape (N, n_eta)
            Current density I-V curves for each sample.
        peroxide_current : np.ndarray of shape (N, n_eta)
            Peroxide current I-V curves for each sample.
        phi_applied : np.ndarray of shape (n_eta,)
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

        if parameters.shape[1] != 4:
            raise ValueError(f"Expected parameters with 4 columns, got shape {parameters.shape}")
        if current_density.shape != (N, self._n_eta):
            raise ValueError(f"current_density shape mismatch: expected ({N}, {self._n_eta}), got {current_density.shape}")
        if peroxide_current.shape != (N, self._n_eta):
            raise ValueError(f"peroxide_current shape mismatch: expected ({N}, {self._n_eta}), got {peroxide_current.shape}")

        # Store training bounds (physical space)
        self.training_bounds = {
            "k0_1": (float(parameters[:, 0].min()), float(parameters[:, 0].max())),
            "k0_2": (float(parameters[:, 1].min()), float(parameters[:, 1].max())),
            "alpha_1": (float(parameters[:, 2].min()), float(parameters[:, 2].max())),
            "alpha_2": (float(parameters[:, 3].min()), float(parameters[:, 3].max())),
        }

        # Transform inputs
        X = self._transform_inputs(parameters)  # (N, 4)

        # Compute transformed bounds for distribution construction
        self._transformed_bounds = {
            "k0_1": (float(X[:, 0].min()), float(X[:, 0].max())),
            "k0_2": (float(X[:, 1].min()), float(X[:, 1].max())),
            "alpha_1": (float(X[:, 2].min()), float(X[:, 2].max())),
            "alpha_2": (float(X[:, 3].min()), float(X[:, 3].max())),
        }

        # Build joint distribution from training bounds
        self._joint_dist = self._build_distribution(self.training_bounds)

        # ChaosPy expects (n_dims, n_samples)
        X_T = X.T  # (4, N)

        # Degree selection
        if self.config.cross_validation and len(self.config.degree_candidates) > 1:
            # Use CD at the median voltage index as representative output
            median_idx = self._n_eta // 2
            y_rep = current_density[:, median_idx]
            self._selected_degree = self._select_degree(X_T, y_rep, verbose)
        else:
            self._selected_degree = self.config.max_degree

        # Build final basis
        self._expansion = self._build_basis(self._selected_degree)
        self._n_terms = len(self._expansion)

        if verbose:
            print(
                f"  PCE basis: degree={self._selected_degree}, "
                f"terms={self._n_terms}, samples={N}"
            )

        # Fit 22 CD PCEs and 22 PC PCEs
        self._pce_cd = []
        self._pce_pc = []

        for j in range(self._n_eta):
            pce_cd_j = cp.fit_regression(self._expansion, X_T, current_density[:, j])
            self._pce_cd.append(pce_cd_j)

            pce_pc_j = cp.fit_regression(self._expansion, X_T, peroxide_current[:, j])
            self._pce_pc.append(pce_pc_j)

        if verbose:
            print(f"  Fitted {2 * self._n_eta} scalar PCEs ({self._n_eta} CD + {self._n_eta} PC)")

        # Compute and cache Sobol indices
        self._sobol_indices = self.compute_sobol_indices(verbose=verbose)

        self._is_fitted = True

        if verbose:
            print(f"  PCE fit complete: degree={self._selected_degree}, "
                  f"terms={self._n_terms}, samples={N}")

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
            'current_density' : np.ndarray of shape (n_eta,)
            'peroxide_current' : np.ndarray of shape (n_eta,)
            'phi_applied' : np.ndarray of shape (n_eta,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        x = self._transform_single(k0_1, k0_2, alpha_1, alpha_2)  # (4, 1)

        cd = np.array([float(np.squeeze(self._pce_cd[j](*x))) for j in range(self._n_eta)])
        pc = np.array([float(np.squeeze(self._pce_pc[j](*x))) for j in range(self._n_eta)])

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

        X = self._transform_inputs(parameters)  # (M, 4)
        X_T = X.T  # (4, M)

        cd = np.column_stack(
            [self._pce_cd[j](*X_T) for j in range(self._n_eta)]
        )  # (M, n_eta)
        pc = np.column_stack(
            [self._pce_pc[j](*X_T) for j in range(self._n_eta)]
        )  # (M, n_eta)

        return {
            "current_density": cd,
            "peroxide_current": pc,
            "phi_applied": self._phi_applied.copy(),
        }

    # -----------------------------------------------------------------
    # Analytic gradient via polynomial differentiation
    # -----------------------------------------------------------------

    def predict_gradient(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
    ) -> Dict[str, np.ndarray]:
        """Compute analytic gradient via polynomial differentiation.

        For each output dimension, differentiates the PCE polynomial w.r.t.
        each of the 4 transformed inputs.  If log_space_k0 is True, applies
        the chain rule: d/d(k0) = d/d(log10_k0) * 1/(k0 * ln(10)).

        Parameters
        ----------
        k0_1, k0_2, alpha_1, alpha_2 : float
            BV kinetics parameters (physical space).

        Returns
        -------
        dict with keys:
            'grad_cd' : np.ndarray of shape (4, n_eta)
                Gradient of current density w.r.t. [k0_1, k0_2, alpha_1, alpha_2].
            'grad_pc' : np.ndarray of shape (4, n_eta)
                Gradient of peroxide current w.r.t. [k0_1, k0_2, alpha_1, alpha_2].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        x = self._transform_single(k0_1, k0_2, alpha_1, alpha_2)  # (4, 1)

        # Variable names used by chaospy for differentiation
        # We need to get the polynomial variables
        q0, q1, q2, q3 = cp.variable(4)
        variables = [q0, q1, q2, q3]

        grad_cd = np.zeros((4, self._n_eta))
        grad_pc = np.zeros((4, self._n_eta))

        for j in range(self._n_eta):
            for dim in range(4):
                # Differentiate PCE polynomial w.r.t. dimension dim
                dpce_cd = cp.derivative(self._pce_cd[j], variables[dim])
                dpce_pc = cp.derivative(self._pce_pc[j], variables[dim])

                grad_cd[dim, j] = float(np.squeeze(dpce_cd(*x)))
                grad_pc[dim, j] = float(np.squeeze(dpce_pc(*x)))

        # Apply chain rule for log-space k0 parameters
        if self.config.log_space_k0:
            # d/d(k0) = d/d(log10_k0) * d(log10_k0)/d(k0)
            #         = d/d(log10_k0) * 1/(k0 * ln(10))
            ln10 = np.log(10.0)
            grad_cd[0, :] *= 1.0 / (k0_1 * ln10)
            grad_cd[1, :] *= 1.0 / (k0_2 * ln10)
            grad_pc[0, :] *= 1.0 / (k0_1 * ln10)
            grad_pc[1, :] *= 1.0 / (k0_2 * ln10)

        return {
            "grad_cd": grad_cd,
            "grad_pc": grad_pc,
        }

    # -----------------------------------------------------------------
    # Sobol sensitivity analysis
    # -----------------------------------------------------------------

    def compute_sobol_indices(self, verbose: bool = False) -> Dict:
        """Compute Sobol sensitivity indices from PCE coefficients.

        Uses ChaosPy's built-in Sobol computation functions.  Computes
        first-order, total-order, and second-order indices for all
        voltage points and both outputs.

        Parameters
        ----------
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Nested dict with structure described in the plan.
        """
        if self._joint_dist is None:
            raise RuntimeError("Model must be fitted before computing Sobol indices.")

        param_names = (
            ["log10_k0_1", "log10_k0_2", "alpha_1", "alpha_2"]
            if self.config.log_space_k0
            else ["k0_1", "k0_2", "alpha_1", "alpha_2"]
        )

        n_params = 4
        n_eta = self._n_eta

        # Initialize arrays
        cd_first = np.zeros((n_params, n_eta))
        cd_total = np.zeros((n_params, n_eta))
        cd_second = np.zeros((n_params, n_params, n_eta))

        pc_first = np.zeros((n_params, n_eta))
        pc_total = np.zeros((n_params, n_eta))
        pc_second = np.zeros((n_params, n_params, n_eta))

        if verbose:
            print("  Computing Sobol indices...")

        for j in range(n_eta):
            # Current density
            try:
                cd_first[:, j] = cp.Sens_m(self._pce_cd[j], self._joint_dist)
                cd_total[:, j] = cp.Sens_t(self._pce_cd[j], self._joint_dist)
                cd_second[:, :, j] = cp.Sens_m2(self._pce_cd[j], self._joint_dist)
            except Exception:
                logger.warning(
                    "Sobol computation failed for CD at voltage index %d, "
                    "leaving as zeros",
                    j,
                    exc_info=True,
                )

            # Peroxide current
            try:
                pc_first[:, j] = cp.Sens_m(self._pce_pc[j], self._joint_dist)
                pc_total[:, j] = cp.Sens_t(self._pce_pc[j], self._joint_dist)
                pc_second[:, :, j] = cp.Sens_m2(self._pce_pc[j], self._joint_dist)
            except Exception:
                logger.warning(
                    "Sobol computation failed for PC at voltage index %d, "
                    "leaving as zeros",
                    j,
                    exc_info=True,
                )

        # Mean indices (averaged over voltages)
        cd_mean_first = cd_first.mean(axis=1)
        cd_mean_total = cd_total.mean(axis=1)
        pc_mean_first = pc_first.mean(axis=1)
        pc_mean_total = pc_total.mean(axis=1)

        sobol = {
            "parameter_names": param_names,
            "current_density": {
                "first_order": cd_first,
                "total_order": cd_total,
                "second_order": cd_second,
                "mean_first_order": cd_mean_first,
                "mean_total_order": cd_mean_total,
            },
            "peroxide_current": {
                "first_order": pc_first,
                "total_order": pc_total,
                "second_order": pc_second,
                "mean_first_order": pc_mean_first,
                "mean_total_order": pc_mean_total,
            },
        }

        self._sobol_indices = sobol

        if verbose:
            print("  Sobol indices computed.")

        return sobol

    def print_sensitivity_report(self) -> None:
        """Print a formatted sensitivity analysis report."""
        if self._sobol_indices is None:
            raise RuntimeError(
                "Sobol indices not computed. Call fit() or compute_sobol_indices() first."
            )

        sobol = self._sobol_indices
        param_names = sobol["parameter_names"]
        n_eta = self._n_eta

        print(f"\n{'='*64}")
        print(f"  PCE SOBOL SENSITIVITY ANALYSIS")
        print(f"  Basis terms: {self._n_terms}, Degree: {self._selected_degree}, "
              f"Samples: (fitted model)")
        print(f"{'='*64}")

        for output_key, output_label in [
            ("current_density", "CURRENT DENSITY"),
            ("peroxide_current", "PEROXIDE CURRENT"),
        ]:
            data = sobol[output_key]
            mean_first = data["mean_first_order"]
            mean_total = data["mean_total_order"]
            interaction = mean_total - mean_first

            print(f"  {output_label} -- Mean Sobol Indices (averaged over {n_eta} voltages)")
            print(f"  {'-'*60}")
            print(f"  {'Parameter':<16s}  {'S_i (1st)':>12s}  {'ST_i (total)':>14s}  "
                  f"{'Interaction':>12s}")

            for i, name in enumerate(param_names):
                print(f"  {name:<16s}  {mean_first[i]:>12.4f}  {mean_total[i]:>14.4f}  "
                      f"{interaction[i]:>12.4f}")

            # Top second-order interactions
            second = data["second_order"]  # (4, 4, n_eta)
            mean_second = second.mean(axis=2)  # (4, 4)

            # Collect upper-triangle interactions
            interactions_list = []
            for i in range(4):
                for j in range(i + 1, 4):
                    interactions_list.append((i, j, mean_second[i, j]))

            interactions_list.sort(key=lambda x: abs(x[2]), reverse=True)

            print(f"  {'-'*60}")
            print(f"  Top interactions:")
            for i, j, val in interactions_list[:3]:
                print(f"    {param_names[i]} x {param_names[j]}: "
                      f"S_{i+1}{j+1} = {val:.4f}")

            print()

        # Key finding: k0_2 variance fraction
        # k0_2 is index 1 in the parameter list
        cd_data = sobol["current_density"]
        k0_2_first_cd = cd_data["mean_first_order"][1]
        k0_2_total_cd = cd_data["mean_total_order"][1]

        pc_data = sobol["peroxide_current"]
        k0_2_first_pc = pc_data["mean_first_order"][1]
        k0_2_total_pc = pc_data["mean_total_order"][1]

        print(f"{'='*64}")
        print(f"  KEY FINDING: k0_2 accounts for {k0_2_first_cd*100:.1f}% of CD variance")
        print(f"  (first-order) and {k0_2_total_cd*100:.1f}% including interactions (total-order).")
        print(f"  For PC: {k0_2_first_pc*100:.1f}% first-order, "
              f"{k0_2_total_pc*100:.1f}% total-order.")
        print(f"{'='*64}")

        # Voltage-resolved: where is k0_2 most/least influential?
        cd_first_k0_2 = cd_data["first_order"][1, :]  # (n_eta,)
        if self._phi_applied is not None:
            max_idx = int(np.argmax(cd_first_k0_2))
            min_idx = int(np.argmin(cd_first_k0_2))
            print(f"  k0_2 most influential for CD at V={self._phi_applied[max_idx]:.3f} "
                  f"(S_1={cd_first_k0_2[max_idx]:.4f})")
            print(f"  k0_2 least influential for CD at V={self._phi_applied[min_idx]:.3f} "
                  f"(S_1={cd_first_k0_2[min_idx]:.4f})")
        print()

    def save_sobol_indices(self, path: str) -> None:
        """Export Sobol indices to JSON.

        Parameters
        ----------
        path : str
            Output file path (typically .json).
        """
        if self._sobol_indices is None:
            raise RuntimeError(
                "Sobol indices not computed. Call fit() or compute_sobol_indices() first."
            )

        # Convert numpy arrays to lists for JSON serialization
        sobol = self._sobol_indices

        out = {
            "parameter_names": sobol["parameter_names"],
            "degree": self._selected_degree,
            "n_terms": self._n_terms,
        }

        for output_key in ["current_density", "peroxide_current"]:
            data = sobol[output_key]
            out[output_key] = {
                "first_order": data["first_order"].tolist(),
                "total_order": data["total_order"].tolist(),
                "second_order": data["second_order"].tolist(),
                "mean_first_order": data["mean_first_order"].tolist(),
                "mean_total_order": data["mean_total_order"].tolist(),
            }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"  Sobol indices saved to: {path}")

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save fitted PCE model to a pickle file.

        Parameters
        ----------
        path : str
            Output file path (typically .pkl).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  PCE model saved to: {path}")

    @staticmethod
    def load(path: str) -> "PCESurrogateModel":
        """Load a fitted PCE model from a pickle file.

        Parameters
        ----------
        path : str
            Path to .pkl file.

        Returns
        -------
        PCESurrogateModel
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, PCESurrogateModel):
            raise TypeError(
                f"Expected PCESurrogateModel, got {type(model).__name__}"
            )
        print(f"  PCE model loaded from: {path} (n_eta={model.n_eta})")
        return model

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
    def n_terms(self) -> int:
        """Number of PCE basis terms."""
        return self._n_terms

    @property
    def selected_degree(self) -> int:
        """Polynomial degree selected (by CV or config)."""
        return self._selected_degree
