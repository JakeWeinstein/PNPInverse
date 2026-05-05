"""Validation tests for the new ``steric_mode`` and ``a_nondim`` fields
on ``bv_bc.boltzmann_counterions`` entries.

The parser lives at
``Forward/bv_solver/config.py::_get_bv_boltzmann_counterions_cfg``.

Tests pin the validation contract:

- Default behaviour (no ``steric_mode``) returns the legacy three-key
  shape *plus* the new ``steric_mode='ideal'`` and ``a_nondim=None``
  defaults.
- ``steric_mode`` must be one of ``{'ideal', 'bikerman'}``.
- ``steric_mode='bikerman'`` requires ``a_nondim`` and rejects
  non-positive values.
- ``a_nondim`` set in ``ideal`` mode is accepted but not promoted into
  the closure (no behavioural effect).

Pure-Python; no Firedrake.
"""
from __future__ import annotations

import pytest

from Forward.bv_solver.config import _get_bv_boltzmann_counterions_cfg


def _params(entries):
    return {"bv_bc": {"boltzmann_counterions": entries}}


# ---------------------------------------------------------------------------
# Defaults preserve legacy behaviour
# ---------------------------------------------------------------------------

def test_default_entry_gets_ideal_mode_and_none_a_nondim():
    """An entry without steric_mode/a_nondim parses to the legacy
    ``ideal`` mode with ``a_nondim=None`` — no behavioural change for
    existing callers."""
    out = _get_bv_boltzmann_counterions_cfg(
        _params([{"z": -1, "c_bulk_nondim": 0.2}])
    )
    assert len(out) == 1
    entry = out[0]
    assert entry["z"] == -1
    assert entry["c_bulk_nondim"] == 0.2
    assert entry["phi_clamp"] == 50.0
    assert entry["steric_mode"] == "ideal"
    assert entry["a_nondim"] is None


def test_explicit_ideal_mode_matches_default():
    explicit = _get_bv_boltzmann_counterions_cfg(
        _params([{"z": -1, "c_bulk_nondim": 0.2, "steric_mode": "ideal"}])
    )
    default = _get_bv_boltzmann_counterions_cfg(
        _params([{"z": -1, "c_bulk_nondim": 0.2}])
    )
    assert explicit == default


def test_steric_mode_case_insensitive():
    out = _get_bv_boltzmann_counterions_cfg(
        _params([{"z": -1, "c_bulk_nondim": 0.2, "steric_mode": "  IDEAL  "}])
    )
    assert out[0]["steric_mode"] == "ideal"


# ---------------------------------------------------------------------------
# Bikerman mode requires a_nondim
# ---------------------------------------------------------------------------

def test_bikerman_mode_parses_a_nondim():
    out = _get_bv_boltzmann_counterions_cfg(
        _params([{
            "z": -1,
            "c_bulk_nondim": 0.2,
            "steric_mode": "bikerman",
            "a_nondim": 0.01,
        }])
    )
    assert out[0]["steric_mode"] == "bikerman"
    assert out[0]["a_nondim"] == 0.01


def test_bikerman_mode_without_a_nondim_raises():
    with pytest.raises(ValueError, match="requires 'a_nondim'"):
        _get_bv_boltzmann_counterions_cfg(
            _params([{
                "z": -1,
                "c_bulk_nondim": 0.2,
                "steric_mode": "bikerman",
            }])
        )


def test_bikerman_mode_with_zero_a_nondim_raises():
    with pytest.raises(ValueError, match="a_nondim must be positive"):
        _get_bv_boltzmann_counterions_cfg(
            _params([{
                "z": -1,
                "c_bulk_nondim": 0.2,
                "steric_mode": "bikerman",
                "a_nondim": 0.0,
            }])
        )


def test_bikerman_mode_with_negative_a_nondim_raises():
    with pytest.raises(ValueError, match="a_nondim must be positive"):
        _get_bv_boltzmann_counterions_cfg(
            _params([{
                "z": -1,
                "c_bulk_nondim": 0.2,
                "steric_mode": "bikerman",
                "a_nondim": -0.01,
            }])
        )


# ---------------------------------------------------------------------------
# Invalid steric_mode rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_mode", ["bikermn", "borukhov", "kornyshev", ""])
def test_invalid_steric_mode_raises(bad_mode):
    with pytest.raises(ValueError, match="steric_mode must be"):
        _get_bv_boltzmann_counterions_cfg(
            _params([{
                "z": -1,
                "c_bulk_nondim": 0.2,
                "steric_mode": bad_mode,
            }])
        )


# ---------------------------------------------------------------------------
# a_nondim accepted but ignored in ideal mode
# ---------------------------------------------------------------------------

def test_ideal_mode_with_a_nondim_accepted():
    """Stale ``a_nondim`` from a copy-pasted bikerman config should not
    error out in ideal mode (the closure ignores it).  Documents the
    laxness for backward compat."""
    out = _get_bv_boltzmann_counterions_cfg(
        _params([{
            "z": -1,
            "c_bulk_nondim": 0.2,
            "steric_mode": "ideal",
            "a_nondim": 0.01,
        }])
    )
    assert out[0]["steric_mode"] == "ideal"
    assert out[0]["a_nondim"] == 0.01  # parsed but not used by the ideal path


# ---------------------------------------------------------------------------
# Empty/no-config cases unchanged
# ---------------------------------------------------------------------------

def test_no_counterions_returns_empty():
    assert _get_bv_boltzmann_counterions_cfg({}) == []
    assert _get_bv_boltzmann_counterions_cfg(_params([])) == []
    assert _get_bv_boltzmann_counterions_cfg(_params(None)) == []


def test_multiple_counterions_with_mixed_modes():
    out = _get_bv_boltzmann_counterions_cfg(
        _params([
            {"z": -1, "c_bulk_nondim": 0.2},  # legacy default
            {"z": -1, "c_bulk_nondim": 0.2,
             "steric_mode": "bikerman", "a_nondim": 0.01},
        ])
    )
    assert len(out) == 2
    assert out[0]["steric_mode"] == "ideal"
    assert out[0]["a_nondim"] is None
    assert out[1]["steric_mode"] == "bikerman"
    assert out[1]["a_nondim"] == 0.01
