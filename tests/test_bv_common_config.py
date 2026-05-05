"""Unit tests for ``scripts/_bv_common.py`` factory wiring.

These tests exercise pure-Python logic and do NOT require Firedrake.

They lock down the conditional-write behaviour for the optional
``stern_capacitance_f_m2`` ``bv_bc`` key introduced for the Stern layer
test path (see ``docs/stern_layer_physics_and_next_steps.md``).  When
``stern_capacitance_f_m2`` is omitted or ``None``, the resulting
``bv_bc`` dict must be byte-identical to the pre-Stern shape so the
wiring change cannot silently perturb the production no-Stern path.

For the snapshot regression on numerical CD/PC see
``tests/test_stern_no_stern_snapshot.py`` (slow).
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    THREE_SPECIES_LOGC_BOLTZMANN,
    make_bv_solver_params,
)


_OMITTED = object()  # sentinel: kwarg should not be passed at all


def _build(stern_capacitance_f_m2=_OMITTED):
    """Build production-stack solver params; pass Stern only when supplied."""
    kwargs = dict(
        eta_hat=0.0,
        dt=0.25,
        t_end=10.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
    )
    if stern_capacitance_f_m2 is not _OMITTED:
        kwargs["stern_capacitance_f_m2"] = stern_capacitance_f_m2
    return make_bv_solver_params(**kwargs)


def _bv_bc(sp):
    return sp[10]["bv_bc"]


# ===================================================================
# Forward-direction wiring — does the kwarg flow into bv_bc correctly?
# ===================================================================

class TestSternConfigWiring:
    """Lock down the Stern key plumbing in ``_make_bv_bc_cfg``."""

    def test_default_omits_key(self):
        bv_bc = _bv_bc(_build())
        assert "stern_capacitance_f_m2" not in bv_bc

    def test_explicit_none_omits_key(self):
        bv_bc = _bv_bc(_build(stern_capacitance_f_m2=None))
        assert "stern_capacitance_f_m2" not in bv_bc

    def test_finite_stern_writes_value(self):
        bv_bc = _bv_bc(_build(stern_capacitance_f_m2=0.20))
        assert bv_bc["stern_capacitance_f_m2"] == 0.20
        assert isinstance(bv_bc["stern_capacitance_f_m2"], float)

    def test_zero_writes_zero_runtime_inactive(self):
        # ``forms_logc.py:229`` requires ``> 0`` to activate Stern, so 0.0
        # is debuggable but inactive at runtime.  The factory still writes
        # it through so the cfg dict reflects the caller's exact intent.
        bv_bc = _bv_bc(_build(stern_capacitance_f_m2=0.0))
        assert bv_bc["stern_capacitance_f_m2"] == 0.0

    def test_int_coerced_to_float(self):
        bv_bc = _bv_bc(_build(stern_capacitance_f_m2=1))
        assert bv_bc["stern_capacitance_f_m2"] == 1.0
        assert isinstance(bv_bc["stern_capacitance_f_m2"], float)


# ===================================================================
# Regression gate — no-Stern shape is unchanged by the wiring landing
# ===================================================================

class TestSternRegressionGate:
    """Lock down that wiring did NOT perturb the no-Stern dict shape."""

    def test_dict_byte_identity_when_omitted_vs_none(self):
        bv_bc_omitted = _bv_bc(_build())
        bv_bc_none = _bv_bc(_build(stern_capacitance_f_m2=None))
        assert bv_bc_omitted == bv_bc_none

    def test_no_stern_bv_bc_keys_unchanged(self):
        """Snapshot of the no-Stern key set.  Update this list deliberately
        if a new optional ``bv_bc`` field is added."""
        bv_bc = _bv_bc(_build())
        expected_keys = {
            "reactions",
            "k0",
            "alpha",
            "stoichiometry",
            "c_ref",
            "E_eq_v",
            "electrode_marker",
            "concentration_marker",
            "ground_marker",
            "boltzmann_counterions",
        }
        assert set(bv_bc.keys()) == expected_keys


# ===================================================================
# Validator contract — negative C_S surfaces at the config layer
# ===================================================================

class TestSternConfigValidator:
    """The factory itself does not validate negative C_S; the rejection
    fires inside ``Forward.bv_solver.config._get_bv_cfg`` during solver
    setup.  Confirm the end-to-end contract: factory passes through,
    config layer rejects."""

    def test_factory_does_not_reject_negative(self):
        # Documents the layer split: a negative value is silently written.
        bv_bc = _bv_bc(_build(stern_capacitance_f_m2=-0.1))
        assert bv_bc["stern_capacitance_f_m2"] == -0.1

    def test_config_layer_rejects_negative(self):
        from Forward.bv_solver.config import _get_bv_cfg
        sp = _build(stern_capacitance_f_m2=-0.1)
        with pytest.raises(ValueError, match="must be non-negative"):
            _get_bv_cfg(sp[10], n_species=3)

    def test_config_layer_accepts_zero(self):
        from Forward.bv_solver.config import _get_bv_cfg
        sp = _build(stern_capacitance_f_m2=0.0)
        out = _get_bv_cfg(sp[10], n_species=3)
        assert out["stern_capacitance_f_m2"] == 0.0

    def test_config_layer_accepts_positive(self):
        from Forward.bv_solver.config import _get_bv_cfg
        sp = _build(stern_capacitance_f_m2=0.20)
        out = _get_bv_cfg(sp[10], n_species=3)
        assert out["stern_capacitance_f_m2"] == 0.20

    def test_config_layer_treats_omitted_as_none(self):
        """When the cfg key is absent, the parsed value is ``None``
        (matches default no-Stern behaviour)."""
        from Forward.bv_solver.config import _get_bv_cfg
        sp = _build()
        out = _get_bv_cfg(sp[10], n_species=3)
        assert out["stern_capacitance_f_m2"] is None
