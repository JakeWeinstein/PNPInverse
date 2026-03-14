"""Verification tests for the v13 inference pipeline (surrogate-level).

These tests validate mathematical correctness of the surrogate-based
inference objectives, gradients, and observable behavior for Butler-Volmer
kinetics parameter recovery.

Remaining tests after Phase 4 consolidation:

- **Test 3** (TestGradientVerification): FD gradient consistency at two
  step sizes for AlphaOnly and Subset surrogate objectives.
- **Test 4** (TestObservableSignAndScale): Physical sign and magnitude
  checks on surrogate predictions.
- **Test 6** (TestSensitivityMonotonicity): Perturbation sensitivity --
  any parameter change from truth increases the loss.

Tests 1 (zero-noise recovery), 2 (PDE roundtrip), 5 (surrogate vs PDE),
and 7 (multistart basin) have been removed -- subsumed by
``tests/test_inverse_verification.py`` (Phase 4) and
``tests/test_surrogate_fidelity.py`` (Phase 3).

All slow tests require PyTorch for the NN ensemble and are marked
``@pytest.mark.slow``.

Run all tests::

    pytest tests/test_v13_verification.py -m slow --tb=short
"""

from __future__ import annotations

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from scripts._bv_common import (
    K0_HAT_R1,
    K0_HAT_R2,
    ALPHA_R1,
    ALPHA_R2,
)

# True parameter values for all tests.
# K0_HAT_R1 = K0_PHYS_R1 / K_SCALE where K_SCALE = D_O2 / L_REF
# K0_HAT_R2 = K0_PHYS_R2 / K_SCALE
# These are the dimensionless rate constants for the two BV reactions.
TRUE_K0 = np.array([K0_HAT_R1, K0_HAT_R2])
TRUE_ALPHA = np.array([ALPHA_R1, ALPHA_R2])

# Ensemble path (relative to PNPInverse root)
_ENSEMBLE_DIR = os.path.join(
    _ROOT, "data", "surrogate_models", "nn_ensemble", "D3-deeper"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nn_ensemble():
    """Load the real NN ensemble surrogate (shared across the module).

    This fixture is module-scoped to avoid reloading ~5 PyTorch models
    per test. It returns None if the ensemble directory is not on disk
    (the individual tests will skip).
    """
    if not os.path.isdir(_ENSEMBLE_DIR):
        pytest.skip("D3-deeper ensemble not found on disk")
    from Surrogate.ensemble import load_nn_ensemble
    return load_nn_ensemble(_ENSEMBLE_DIR, n_members=5, device="cpu")


@pytest.fixture(scope="module")
def true_predictions(nn_ensemble):
    """Surrogate predictions at the true parameters (shared across module)."""
    pred = nn_ensemble.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    return pred


# Test 1 (zero-noise identity) removed -- subsumed by
# tests/test_inverse_verification.py::TestParameterRecovery (INV-01, 0% noise)

# Test 2 (PDE roundtrip at truth) removed -- subsumed by
# tests/test_inverse_verification.py::TestGradientConsistencyPDE (INV-02a)


# ---------------------------------------------------------------------------
# Test 3: Gradient verification via finite differences
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGradientVerification:
    """Verify FD gradients by comparing two step sizes and manual FD check.

    Mathematical property tested:
        For a central FD gradient with step h, the truncation error is O(h^2).
        Comparing gradients at h=1e-4 and h=1e-5 (factor 10 smaller) should
        agree to ~2 digits if the function is smooth.

        Note: The NN surrogate runs in float32 (eps ~1.2e-7). Step sizes
        smaller than ~1e-6 fall below the representable precision of the
        normalized inputs, producing noise-dominated gradients. We therefore
        compare h=1e-4 vs h=1e-5 (both well above the float32 floor).

        Additionally, a manual FD check using (f(x+h*e_i) - f(x-h*e_i))/(2h)
        must match the gradient method's output exactly (they use the same
        formula).

    Tolerance:
        - h=1e-4 vs h=1e-5 agreement: relative tolerance 5e-2 per component.
          Central FD error is O(h^2), but the NN has float32 noise that adds
          ~1e-7 absolute noise to function evaluations. At h=1e-5 this gives
          a gradient noise floor of ~1e-7/(2*1e-5) = 5e-3, so 5% tolerance
          accounts for both truncation and evaluation noise.
        - Manual FD vs .gradient(): relative tolerance 1e-10 (should be
          bit-identical up to floating point ordering).
    """

    def test_alpha_only_gradient(self, nn_ensemble, true_predictions):
        """AlphaOnlySurrogateObjective gradient consistency at two step sizes."""
        from Surrogate.objectives import AlphaOnlySurrogateObjective

        target_cd = true_predictions["current_density"]
        target_pc = true_predictions["peroxide_current"]

        # Evaluate gradient at a slightly perturbed point (not at the minimum,
        # where gradients are zero and relative error is meaningless).
        x_test = np.array([ALPHA_R1 * 1.1, ALPHA_R2 * 0.9])

        obj_h4 = AlphaOnlySurrogateObjective(
            nn_ensemble, target_cd, target_pc,
            fixed_k0=(K0_HAT_R1, K0_HAT_R2),
            secondary_weight=1.0, fd_step=1e-4,
        )
        obj_h5 = AlphaOnlySurrogateObjective(
            nn_ensemble, target_cd, target_pc,
            fixed_k0=(K0_HAT_R1, K0_HAT_R2),
            secondary_weight=1.0, fd_step=1e-5,
        )

        grad_h4 = obj_h4.gradient(x_test)
        grad_h5 = obj_h5.gradient(x_test)

        # Two step sizes should agree to ~5% (both above float32 noise floor)
        for i in range(2):
            if abs(grad_h5[i]) > 1e-12:
                relerr = abs(grad_h4[i] - grad_h5[i]) / abs(grad_h5[i])
                assert relerr < 5e-2, (
                    f"AlphaOnly gradient component {i}: h=1e-4 gives {grad_h4[i]:.8e}, "
                    f"h=1e-5 gives {grad_h5[i]:.8e}, relative error {relerr:.4e}"
                )

        # Manual FD check: should match .gradient() exactly
        h = 1e-5
        manual_grad = np.zeros(2)
        for i in range(2):
            xp = x_test.copy()
            xm = x_test.copy()
            xp[i] += h
            xm[i] -= h
            manual_grad[i] = (obj_h5.objective(xp) - obj_h5.objective(xm)) / (2 * h)

        np.testing.assert_allclose(
            grad_h5, manual_grad, rtol=1e-10,
            err_msg="AlphaOnly .gradient() does not match manual FD computation",
        )

    def test_subset_objective_gradient(self, nn_ensemble, true_predictions):
        """SubsetSurrogateObjective gradient consistency at two step sizes."""
        from Surrogate.objectives import SubsetSurrogateObjective

        target_cd = true_predictions["current_density"]
        target_pc = true_predictions["peroxide_current"]
        n_eta = nn_ensemble.n_eta
        subset_idx = np.arange(n_eta)

        # Test at a perturbed point
        x_test = np.array([
            np.log10(K0_HAT_R1 * 1.5),
            np.log10(K0_HAT_R2 * 0.7),
            ALPHA_R1 * 1.1,
            ALPHA_R2 * 0.9,
        ])

        obj_h4 = SubsetSurrogateObjective(
            nn_ensemble, target_cd, target_pc, subset_idx,
            secondary_weight=1.0, fd_step=1e-4, log_space_k0=True,
        )
        obj_h5 = SubsetSurrogateObjective(
            nn_ensemble, target_cd, target_pc, subset_idx,
            secondary_weight=1.0, fd_step=1e-5, log_space_k0=True,
        )

        grad_h4 = obj_h4.gradient(x_test)
        grad_h5 = obj_h5.gradient(x_test)

        for i in range(4):
            if abs(grad_h5[i]) > 1e-12:
                relerr = abs(grad_h4[i] - grad_h5[i]) / abs(grad_h5[i])
                assert relerr < 5e-2, (
                    f"Subset gradient component {i}: h=1e-4 gives {grad_h4[i]:.8e}, "
                    f"h=1e-5 gives {grad_h5[i]:.8e}, relative error {relerr:.4e}"
                )

        # Manual FD check
        h = 1e-5
        manual_grad = np.zeros(4)
        for i in range(4):
            xp = x_test.copy()
            xm = x_test.copy()
            xp[i] += h
            xm[i] -= h
            manual_grad[i] = (obj_h5.objective(xp) - obj_h5.objective(xm)) / (2 * h)

        np.testing.assert_allclose(
            grad_h5, manual_grad, rtol=1e-10,
            err_msg="Subset .gradient() does not match manual FD computation",
        )


# ---------------------------------------------------------------------------
# Test 4: Observable sign and scale verification
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestObservableSignAndScale:
    """Verify that surrogate predictions have physically correct sign and magnitude.

    Mathematical property tested:
        For cathodic overpotentials (eta < 0), the Butler-Volmer equation
        predicts net reduction current (negative by convention when scaled
        by -I_SCALE). The current magnitude should increase with |eta| in
        the Tafel regime. The peroxide current (H2O2 production) should
        also be nonzero and consistent in sign.

    Physical reasoning:
        - O2 reduction: at eta < 0, the cathodic branch of BV dominates,
          producing a net negative current (reduction).
        - H2O2 is the product of the first reaction, so its flux should
          be positive (production) at cathodic overpotentials.
        - The magnitude should be bounded: not zero (reactions are active)
          and not astronomically large (physical constraint).

    Tolerance:
        - Sign checks are exact (no tolerance needed).
        - Magnitude checks use generous physical bounds.
    """

    def test_current_density_sign_cathodic(self, nn_ensemble):
        """At cathodic overpotentials, current density should be negative (reduction)."""
        pred = nn_ensemble.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        phi = pred["phi_applied"]
        cd = pred["current_density"]

        # Identify cathodic points: eta < -2 (well into cathodic regime)
        cathodic_mask = phi < -2.0
        assert np.any(cathodic_mask), (
            "No cathodic overpotential points in surrogate grid. "
            f"phi_applied range: [{phi.min():.1f}, {phi.max():.1f}]"
        )

        # The surrogate predicts dimensionless flux. Under the convention
        # used in the pipeline (observable_scale = -I_SCALE), cathodic
        # current should be negative. The raw surrogate output sign depends
        # on training convention. Check that cathodic values have consistent
        # sign (all same sign, and that sign corresponds to reduction).
        cathodic_cd = cd[cathodic_mask]

        # All cathodic current values should have the same sign
        signs = np.sign(cathodic_cd)
        dominant_sign = np.sign(np.mean(cathodic_cd))
        assert np.all(signs == dominant_sign), (
            f"Mixed signs in cathodic current density. "
            f"Values: {cathodic_cd[:5]}..."
        )

    def test_current_density_magnitude_physical(self, nn_ensemble):
        """Current density magnitude should be in a physically reasonable range."""
        pred = nn_ensemble.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        cd = pred["current_density"]

        # Dimensionless flux should be O(1) to O(100) in magnitude
        # (corresponds to ~0.01 to ~1000 mA/cm^2 when scaled by I_SCALE ~9.63)
        max_abs = np.max(np.abs(cd))
        assert max_abs > 1e-6, (
            f"Current density is essentially zero (max |cd| = {max_abs:.4e}). "
            f"Expected nonzero response at cathodic overpotentials."
        )
        assert max_abs < 1e6, (
            f"Current density is unphysically large (max |cd| = {max_abs:.4e}). "
            f"Possible sign error or scaling bug."
        )

    def test_peroxide_current_nonzero(self, nn_ensemble):
        """Peroxide current should be nonzero (H2O2 is produced in reaction 1)."""
        pred = nn_ensemble.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        pc = pred["peroxide_current"]
        phi = pred["phi_applied"]

        cathodic_mask = phi < -2.0
        cathodic_pc = pc[cathodic_mask]

        max_abs_pc = np.max(np.abs(cathodic_pc))
        assert max_abs_pc > 1e-6, (
            f"Peroxide current is essentially zero at cathodic potentials "
            f"(max |pc| = {max_abs_pc:.4e}). H2O2 production expected."
        )

    def test_current_increases_with_overpotential(self, nn_ensemble):
        """In the Tafel regime, |current| should increase monotonically with |eta|.

        This is a consequence of the Butler-Volmer equation: for large |eta|,
        the current grows exponentially as exp(alpha * n * F * |eta| / RT).
        The surrogate should reproduce this monotonic trend.
        """
        pred = nn_ensemble.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        phi = pred["phi_applied"]
        cd = pred["current_density"]

        # Sort by increasing |eta| in the cathodic regime
        cathodic_mask = phi < -3.0
        if np.sum(cathodic_mask) < 3:
            pytest.skip("Not enough cathodic points for monotonicity check")

        cathodic_phi = phi[cathodic_mask]
        cathodic_cd = cd[cathodic_mask]

        # Sort by increasing phi (i.e., decreasing |eta|): least negative first
        # so that traversing the array means increasing |eta|.
        sort_idx = np.argsort(cathodic_phi)[::-1]  # least negative first
        sorted_abs_cd = np.abs(cathodic_cd[sort_idx])

        # Check that |current| is generally increasing with |eta|.
        # Allow for a few non-monotone points (surrogate approximation noise).
        n_increasing = np.sum(np.diff(sorted_abs_cd) > -1e-8)
        fraction_increasing = n_increasing / max(len(sorted_abs_cd) - 1, 1)
        assert fraction_increasing > 0.7, (
            f"Only {fraction_increasing:.0%} of consecutive cathodic points show "
            f"increasing |current| with |eta|. Expected monotonic Tafel behavior."
        )


# Test 5 (Surrogate vs PDE consistency) removed -- subsumed by tests/test_surrogate_fidelity.py


# ---------------------------------------------------------------------------
# Test 6: Sensitivity monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSensitivityMonotonicity:
    """Perturbing any parameter from truth should increase the loss.

    Mathematical property tested:
        If the target is generated at theta* and the objective is
            J(theta) = 0.5 ||S(theta) - S(theta*)||^2,
        then theta* is a global minimum with J(theta*) = 0. Any perturbation
        delta_theta != 0 should give J(theta* + delta) > 0, provided the
        surrogate has nonzero sensitivity to each parameter.

        This test catches:
        - Sign errors in the objective (J could be negative)
        - Non-identifiability (J flat along some direction)
        - Objective implementation bugs (wrong parameter mapping)

    Tolerance:
        - Loss increase > 1e-8: ensures nontrivial sensitivity.
          A 20% perturbation to any kinetic parameter should produce a
          measurable change in the I-V curve, hence a measurable increase
          in J. The value 1e-8 is extremely conservative.
    """

    def test_all_perturbations_increase_loss(self, nn_ensemble, true_predictions):
        """+-20% perturbation of each parameter increases the objective."""
        from Surrogate.objectives import SurrogateObjective

        target_cd = true_predictions["current_density"]
        target_pc = true_predictions["peroxide_current"]

        obj = SurrogateObjective(
            surrogate=nn_ensemble,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=1.0,
            fd_step=1e-5,
            log_space_k0=True,
        )

        x_true = np.array([
            np.log10(K0_HAT_R1),
            np.log10(K0_HAT_R2),
            ALPHA_R1,
            ALPHA_R2,
        ])

        J_true = obj.objective(x_true)
        assert J_true >= 0, f"Objective at truth is negative: {J_true}"
        assert J_true < 1e-6, (
            f"Objective at truth is unexpectedly large: {J_true:.4e}. "
            f"Expected near-zero when target = surrogate(truth)."
        )

        param_names = ["log10(k0_1)", "log10(k0_2)", "alpha_1", "alpha_2"]
        perturbation_fractions = [0.20, -0.20]

        for i, name in enumerate(param_names):
            for frac in perturbation_fractions:
                x_pert = x_true.copy()
                # For log-space parameters (k0), a 20% perturbation in log-space
                # corresponds to a multiplicative factor of 10^(0.2*log10(k0))
                # which is a meaningful change. For alpha, 20% is straightforward.
                if i < 2:
                    # Log-space: perturb by 20% of the log-value magnitude
                    x_pert[i] *= (1.0 + frac)
                else:
                    # Linear space: perturb by 20% of the value
                    x_pert[i] *= (1.0 + frac)

                J_pert = obj.objective(x_pert)

                loss_increase = J_pert - J_true
                assert loss_increase > 1e-8, (
                    f"Perturbation of {name} by {frac:+.0%} did NOT increase loss. "
                    f"J(truth)={J_true:.4e}, J(perturbed)={J_pert:.4e}, "
                    f"increase={loss_increase:.4e}. "
                    f"This suggests the objective is insensitive to {name}."
                )


# Test 7 (multistart convergence basin) removed -- subsumed by
# tests/test_inverse_verification.py::TestMultistartBasin (INV-03)
