"""Tests for electron-weighted current-density observable (M3a.1, 2026-05-07).

Once the BV reaction list contains heterogeneous ``n_electrons`` (e.g.
parallel 2e + 4e ORR per Ruggiero 2022), the ``current_density``
observable must weight each reaction by its electron count.  These
tests pin down four invariants:

1. Pure-2e regression: a uniform-2e reaction list (the legacy
   sequential preset) gives identical assembled output to the
   pre-M3a.1 unweighted-sum form.
2. Pure-4e: a single-reaction list with ``n_electrons=4`` gives
   ``current_density = 2 · R_4e`` (factor 2 = n_e_4e / N_ELECTRONS_REF).
3. Mixed: a 2e + 4e list gives
   ``current_density = R_2e + 2 · R_4e``.
4. ``gross_h2o2_current`` mode returns a single-reaction rate (R_2e
   alone), not the legacy net difference.

Tests are pure-Python+UFL and skip if Firedrake is missing.  No solver
run; rate expressions are synthesised as Firedrake ``Constant`` UFL
nodes on a 1D unit-interval mesh with a single boundary marker.
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.conftest import skip_without_firedrake


# ----------------------------------------------------------------------
# Fixture: synthetic ctx with controllable rate expressions
# ----------------------------------------------------------------------

def _build_synthetic_ctx(rate_values, n_electrons_list):
    """Return a ctx-like dict that ``_build_bv_observable_form`` can consume.

    ``rate_values`` is a list of floats, one per reaction; each becomes
    a Firedrake ``Constant`` UFL expression.  ``n_electrons_list`` is
    the per-reaction electron count threaded through
    ``ctx['nondim']['bv_reactions']`` so ``current_density`` mode can
    pick it up.

    A 1D unit-interval mesh with a single boundary marker (1) is built;
    ``ds(1)`` integrates to 1 over either endpoint, so for K reactions
    each contributing ``c_k`` the assembled
    ``Σ R_j * ds(1) = Σ c_k`` (one boundary point only --
    ``ds`` over the right endpoint).

    On Firedrake's ``UnitIntervalMesh``, only the left endpoint carries
    boundary marker 1 (the right endpoint carries marker 2), so
    ``ds(1)`` of a ``Constant`` integrand evaluates to that constant
    once.  Tests therefore expect ``Σ c_k * weight_k`` exactly.
    """
    import firedrake as fd

    mesh = fd.UnitIntervalMesh(2)  # 2 cells -> 3 nodes
    bv_rate_exprs = [fd.Constant(float(v)) for v in rate_values]

    bv_settings = {"electrode_marker": 1}

    # The per-reaction n_electrons reaches the observable layer via
    # ctx['nondim']['bv_reactions'].
    nondim = {
        "bv_reactions": [{"n_electrons": float(n_e)} for n_e in n_electrons_list],
    }

    ctx = {
        "mesh": mesh,
        "bv_rate_exprs": bv_rate_exprs,
        "bv_settings": bv_settings,
        "nondim": nondim,
    }
    return ctx


def _assemble(form):
    import firedrake as fd
    return float(fd.assemble(form))


# ----------------------------------------------------------------------
# 1) Pure-2e regression: weighted sum equals unweighted sum
# ----------------------------------------------------------------------

@skip_without_firedrake
def test_pure_2e_current_density_matches_unweighted_legacy():
    """Two reactions both at n_e=2: weighted form == unweighted form.

    Pre-M3a.1 the form was ``Σ R_j``.  Post-M3a.1 it is
    ``Σ (n_e_j / 2) · R_j``.  When all ``n_e_j == 2`` the weights all
    equal 1, and the forms are byte-identical at assembly time.
    """
    from Forward.bv_solver.observables import _build_bv_observable_form

    ctx = _build_synthetic_ctx(
        rate_values=[0.30, 0.20], n_electrons_list=[2, 2]
    )
    form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )
    assembled = _assemble(form)

    # Legacy reference: Σ R_j (single boundary point on UnitIntervalMesh).
    expected = 0.30 + 0.20
    assert assembled == pytest.approx(expected, rel=1e-12)


@skip_without_firedrake
def test_pure_2e_current_density_legacy_fallback_when_no_reactions_in_ctx():
    """Missing reactions list triggers unweighted fallback (back-compat)."""
    from Forward.bv_solver.observables import _build_bv_observable_form

    ctx = _build_synthetic_ctx(
        rate_values=[0.30, 0.20], n_electrons_list=[2, 2]
    )
    # Strip out the reactions list to simulate a legacy ctx
    ctx["nondim"] = {}

    form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )
    assembled = _assemble(form)
    expected = 0.30 + 0.20
    assert assembled == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------
# 2) Pure-4e: factor of 2 boost from n_e_4e / N_ELECTRONS_REF
# ----------------------------------------------------------------------

@skip_without_firedrake
def test_pure_4e_current_density_doubles_unweighted_sum():
    """Single 4e reaction: current_density = 2 · R_4e (factor n_e/2)."""
    from Forward.bv_solver.observables import _build_bv_observable_form

    R_4e = 0.10
    ctx = _build_synthetic_ctx(
        rate_values=[R_4e], n_electrons_list=[4]
    )
    form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )
    assembled = _assemble(form)
    # Reaction weight = 4/2 = 2; ds over single boundary point: 2 * 0.1 = 0.2
    expected = 2.0 * R_4e
    assert assembled == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------
# 3) Mixed 2e + 4e: combined weighting works correctly
# ----------------------------------------------------------------------

@skip_without_firedrake
def test_mixed_2e_4e_current_density_weights_each_reaction_by_n_e():
    """Mixed list: current_density = R_2e + 2 · R_4e."""
    from Forward.bv_solver.observables import _build_bv_observable_form

    R_2e, R_4e = 0.30, 0.10
    ctx = _build_synthetic_ctx(
        rate_values=[R_2e, R_4e], n_electrons_list=[2, 4]
    )
    form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )
    assembled = _assemble(form)
    # Weighted sum: (2/2)*R_2e + (4/2)*R_4e = R_2e + 2*R_4e
    expected = R_2e + 2.0 * R_4e
    assert assembled == pytest.approx(expected, rel=1e-12)


@skip_without_firedrake
def test_mixed_2e_4e_apparent_n_e_recovery():
    """Total disk current divided by sum of rates recovers a sensible n_e_apparent.

    For 2e+4e mixed: apparent_n_e = (n_2e * R_2e + n_4e * R_4e) / (R_2e + R_4e)
    (in the limit where reaction rates absorb the same O₂ flux).
    """
    from Forward.bv_solver.observables import _build_bv_observable_form

    R_2e, R_4e = 0.10, 0.10  # equal-rate mixed
    ctx = _build_synthetic_ctx(
        rate_values=[R_2e, R_4e], n_electrons_list=[2, 4]
    )
    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )
    cd = _assemble(form_cd)

    total_rate = R_2e + R_4e
    # cd = (R_2e + 2*R_4e) = 0.1 + 0.2 = 0.3; total_rate = 0.2.
    # cd / total_rate = 1.5 = average n_e / N_ELECTRONS_REF (= 3/2).
    ratio = cd / total_rate
    assert ratio == pytest.approx(1.5, rel=1e-12)


# ----------------------------------------------------------------------
# 4) gross_h2o2_current: single-reaction (R_2e at index 0)
# ----------------------------------------------------------------------

@skip_without_firedrake
def test_gross_h2o2_current_returns_single_R_2e_rate():
    """gross_h2o2_current is a single-reaction observable."""
    from Forward.bv_solver.observables import _build_bv_observable_form

    R_2e, R_other = 0.30, 0.10
    ctx = _build_synthetic_ctx(
        rate_values=[R_2e, R_other], n_electrons_list=[2, 4]
    )
    form = _build_bv_observable_form(
        ctx, mode="gross_h2o2_current", reaction_index=None, scale=1.0,
    )
    assembled = _assemble(form)
    # Default reaction_index=0 -> picks R_2e
    expected = R_2e
    assert assembled == pytest.approx(expected, rel=1e-12)


@skip_without_firedrake
def test_gross_h2o2_current_explicit_reaction_index():
    """Override reaction_index for non-default layouts."""
    from Forward.bv_solver.observables import _build_bv_observable_form

    R_first, R_h2o2 = 0.30, 0.10
    ctx = _build_synthetic_ctx(
        rate_values=[R_first, R_h2o2], n_electrons_list=[2, 2]
    )
    form = _build_bv_observable_form(
        ctx, mode="gross_h2o2_current", reaction_index=1, scale=1.0,
    )
    assembled = _assemble(form)
    expected = R_h2o2
    assert assembled == pytest.approx(expected, rel=1e-12)


@skip_without_firedrake
def test_gross_h2o2_current_out_of_bounds_raises():
    from Forward.bv_solver.observables import _build_bv_observable_form

    ctx = _build_synthetic_ctx(rate_values=[0.1], n_electrons_list=[2])
    with pytest.raises(ValueError, match="out of bounds"):
        _build_bv_observable_form(
            ctx, mode="gross_h2o2_current", reaction_index=2, scale=1.0,
        )


# ----------------------------------------------------------------------
# 5) Legacy peroxide_current still works
# ----------------------------------------------------------------------

@skip_without_firedrake
def test_legacy_peroxide_current_still_works():
    """``mode='peroxide_current'`` still gives R_0 - R_1 (back-compat)."""
    from Forward.bv_solver.observables import _build_bv_observable_form

    R_0, R_1 = 0.30, 0.20
    ctx = _build_synthetic_ctx(
        rate_values=[R_0, R_1], n_electrons_list=[2, 2]
    )
    form = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=1.0,
    )
    assembled = _assemble(form)
    expected = R_0 - R_1
    assert assembled == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------
# 6) Unknown mode raises with the new mode listed in the error
# ----------------------------------------------------------------------

@skip_without_firedrake
def test_unknown_mode_lists_gross_h2o2_current_in_message():
    from Forward.bv_solver.observables import _build_bv_observable_form

    ctx = _build_synthetic_ctx(rate_values=[0.1], n_electrons_list=[2])
    with pytest.raises(ValueError, match="gross_h2o2_current"):
        _build_bv_observable_form(
            ctx, mode="bogus", reaction_index=None, scale=1.0,
        )
