"""Shared pytest fixtures for the PNPInverse test suite.

Fixtures here are available to all test modules under ``tests/``.
"""

from __future__ import annotations

import os
import sys

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the PNPInverse package root is importable regardless of how pytest
# is invoked (editable install, plain ``python -m pytest``, etc.).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Firedrake availability flag — used to skip tests that need the FEM backend
# ---------------------------------------------------------------------------

def _firedrake_available() -> bool:
    """Return True if Firedrake can be imported."""
    try:
        import firedrake  # noqa: F401
        return True
    except Exception:
        return False


FIREDRAKE_AVAILABLE = _firedrake_available()

skip_without_firedrake = pytest.mark.skipif(
    not FIREDRAKE_AVAILABLE,
    reason="Firedrake is not installed or not importable in this environment",
)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_species_diffusivities() -> tuple[float, float]:
    """Return representative O2 / H2O2 diffusivities in m^2/s."""
    return (1.5e-9, 1.6e-9)


@pytest.fixture()
def default_nondim_kwargs(two_species_diffusivities) -> dict:
    """Keyword arguments for ``build_physical_scales`` with sensible defaults."""
    return dict(
        d_species_m2_s=two_species_diffusivities,
        c_bulk_m=0.1,
        c_inf_m=0.01,
        temperature_k=298.15,
        relative_permittivity=78.5,
        length_scale_m=1e-4,
    )


@pytest.fixture()
def sample_solver_params():
    """Return a ``SolverParams`` instance for a simple 2-species neutral problem."""
    from Forward.params import SolverParams

    return SolverParams(
        n_species=2,
        order=1,
        dt=0.1,
        t_end=10.0,
        z_vals=[0, 0],
        D_vals=[1.0, 1.1],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[1.0, 0.0],
        phi0=0.0,
        solver_options={
            "snes_type": "newtonls",
            "snes_max_it": 50,
        },
    )


@pytest.fixture()
def sample_steady_state_results():
    """Return a small list of ``SteadyStateResult`` instances for testing."""
    from Forward.steady_state.common import SteadyStateResult

    return [
        SteadyStateResult(
            phi_applied=0.01,
            converged=True,
            steps_taken=50,
            final_time=5.0,
            species_flux=[0.123, -0.045],
            observed_flux=0.078,
            final_relative_change=1e-5,
            final_absolute_change=1e-9,
        ),
        SteadyStateResult(
            phi_applied=0.02,
            converged=True,
            steps_taken=60,
            final_time=6.0,
            species_flux=[0.246, -0.090],
            observed_flux=0.156,
            final_relative_change=2e-5,
            final_absolute_change=2e-9,
        ),
        SteadyStateResult(
            phi_applied=0.03,
            converged=False,
            steps_taken=200,
            final_time=20.0,
            species_flux=[0.300, -0.100],
            observed_flux=0.200,
            final_relative_change=5e-2,
            final_absolute_change=1e-3,
            failure_reason="max_steps exceeded",
        ),
    ]
