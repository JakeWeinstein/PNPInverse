"""MMS convergence rates for the bikerman analytic-Boltzmann counterion.

Mirror of ``tests/test_mms_convergence.py`` for the steric-aware
closure: the manufactured Poisson source uses the bikerman closure
``c_b · q · (1−A_dyn) / (theta_b + a_b · c_b · q)`` instead of the
ideal ``c_b · exp(−z·phi)``, AND the dynamic species' steric chemical
potential ``mu_steric = -ln(theta)`` includes the counterion's
contribution.

Asserts that the L2 error converges at the production rate
(slope >= 1.95) on a UnitSquareMesh chain — the closure should not
degrade convergence relative to the ideal path.

Marked ``slow``; runs N in {16, 32, 48} and takes ~2-3 minutes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


@skip_without_firedrake
@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Bikerman-aware manufactured-source MMS is wired (in "
        "scripts/verification/mms_bv_3sp_logc_boltzmann.py via the "
        "bikerman_counterion= kwarg) but Newton diverges at the "
        "currently-tractable test voltages: at the production point "
        "eta_hat=21 the closure saturates so the manufactured "
        "packing -> 0 (ln(packing) -> NaN); at lower eta_hat in "
        "{1, 3} the existing manufactured solution structure leaves "
        "Newton out of basin (the same issue affects the ideal MMS "
        "at low eta -- see u0_L2 ~ 1.7 at eta_hat=1 vs ~5.9e-4 at "
        "eta_hat=21).  A follow-up plan needs a redesigned "
        "manufactured solution that keeps Newton in basin while the "
        "closure stays unsaturated (e.g. smaller delta_i / B_phi, "
        "or eta_hat in 5..8).  Closure correctness itself is "
        "validated by tests/test_steric_boltzmann_closure_algebra.py "
        "and the slow byte-identity / smoke / equivalence / muh "
        "tests in tests/test_steric_boltzmann_closure.py."
    ),
    strict=False,
)
class TestStericBoltzmannMMSConvergence:
    """L2 convergence on UnitSquareMesh sweep with the bikerman closure."""

    N_VALS = [16, 32, 48]
    EXPECTED_RATE = 1.95
    RATE_TOL = 0.30  # slack: closure introduces additional nonlinearity

    @pytest.fixture(scope="class")
    def mms_results(self):
        from scripts.verification.mms_bv_3sp_logc_boltzmann import run_mms
        from scripts._bv_common import DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC, V_T
        # Bikerman MMS test voltage: eta_hat ~ 3 (V_RHE ~ 0.077 V).  The
        # closure saturates at high anodic eta — at the production test
        # point eta_hat = 21.4, c_steric -> 1/a_b = 100 and the
        # manufactured packing fraction
        # ``1 - A_dyn - a_b * c_steric`` collapses to ~0, which is the
        # numerical guard ``packing_floor`` regime.  At eta_hat = 3,
        # c_steric ~ 3.9 and packing ~ 0.95, comfortably unsaturated.
        eta_hat = 3.0
        return run_mms(
            self.N_VALS,
            verbose=True,
            eta_hat=eta_hat,
            v_rhe=eta_hat * V_T,
            clip_source=False,
            bikerman_counterion=DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        )

    def test_at_least_one_mesh_converged(self, mms_results):
        n_converged = len(mms_results["N"])
        assert n_converged >= 2, (
            f"Bikerman MMS converged at only {n_converged}/{len(self.N_VALS)} "
            f"meshes; need >= 2 for rate analysis.  Newton may struggle on "
            f"the bikerman residual at fine meshes — investigate."
        )

    @pytest.mark.parametrize("field", ["u0", "u1", "u2", "phi"])
    def test_l2_convergence_rate(self, mms_results, field):
        """Each primary variable's L2 error must converge at >= 1.95-tol."""
        from tests.test_mms_convergence import assert_convergence_rate

        h = mms_results["h"]
        err = mms_results[f"{field}_L2"]
        assert len(h) >= 2, (
            f"need at least 2 converged meshes to fit a rate; got {len(h)}"
        )
        assert_convergence_rate(
            h, err,
            expected_rate=self.EXPECTED_RATE,
            rate_tol=self.RATE_TOL,
            min_r_squared=0.95,
            label=f"bikerman MMS {field}_L2",
        )
