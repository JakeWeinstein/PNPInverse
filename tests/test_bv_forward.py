"""Pytest wrapper around the existing BV forward solver convergence tests.

The original verification script lives at
``scripts/verification/test_bv_forward.py`` and is driven by ``argparse``.
This module re-uses that script's helper functions (``neutral_species_params``,
``run_single_solve``, ``strategy_A``, etc.) so the actual convergence logic
stays in one place.

Every test is marked ``@pytest.mark.slow`` because it requires a working
Firedrake installation and performs real PDE solves.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure PNPInverse root is on the path (same trick the original script uses).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Also add scripts/verification so we can import the original test module.
_VERIF_DIR = os.path.join(_ROOT, "scripts", "verification")
if _VERIF_DIR not in sys.path:
    sys.path.insert(0, _VERIF_DIR)


pytestmark = [pytest.mark.slow]


# ---------------------------------------------------------------------------
# Lazy import — only attempt Firedrake-dependent imports when actually running
# a slow test.  This lets ``pytest --collect-only`` and ``-m "not slow"``
# succeed even without Firedrake.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bv_test_module():
    """Import the verification test script module on demand."""
    # Use importlib to load from the explicit path, avoiding the name
    # collision where pytest resolves "test_bv_forward" to this file.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bv_forward_verification",
        os.path.join(_VERIF_DIR, "test_bv_forward.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Individual strategy tests
# ---------------------------------------------------------------------------

class TestBVStrategies:
    """Wrap each BV strategy as a separate pytest case."""

    def test_strategy_a_neutral_eta0(self, bv_test_module):
        """Neutral O2+H2O2, eta=0 (equilibrium) — should always converge."""
        assert bv_test_module.strategy_A(), "Strategy A (neutral, eta=0) failed to converge"

    def test_strategy_b_neutral_small_eta(self, bv_test_module):
        """Neutral O2+H2O2, small cathodic overpotential eta=-1 V_T."""
        assert bv_test_module.strategy_B(), "Strategy B (neutral, eta=-1 V_T) failed to converge"

    def test_strategy_c_neutral_moderate_eta(self, bv_test_module):
        """Neutral O2+H2O2, moderate overpotential eta=-5 V_T."""
        assert bv_test_module.strategy_C(), "Strategy C (neutral, eta=-5 V_T) failed to converge"

    def test_strategy_d_neutral_continuation(self, bv_test_module):
        """Neutral O2+H2O2, potential continuation to eta=-10 V_T."""
        assert bv_test_module.strategy_D(eta_target=-10.0), (
            "Strategy D (continuation to eta=-10 V_T) failed"
        )

    def test_strategy_e_charged_eta0(self, bv_test_module):
        """Charged H+/Cl-, eta=0 — tests Poisson coupling."""
        assert bv_test_module.strategy_E(), "Strategy E (charged, eta=0) failed to converge"

    def test_strategy_f_charged_continuation(self, bv_test_module):
        """Charged H+/Cl-, potential continuation to eta=-1.2 V_T."""
        assert bv_test_module.strategy_F(eta_target=-1.2), (
            "Strategy F (charged, continuation to eta=-1.2 V_T) failed"
        )

    def test_strategy_g_small_k0(self, bv_test_module):
        """Neutral species, reduced k0, eta=-10 V_T (less stiff)."""
        assert bv_test_module.strategy_G_small_k0(), (
            "Strategy G (neutral, small k0, eta=-10 V_T) failed"
        )
