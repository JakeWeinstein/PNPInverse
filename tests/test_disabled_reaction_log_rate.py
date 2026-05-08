"""Tests for the disabled-reaction guard in log-rate BV form builders.

Both ``forms_logc.py`` and ``forms_logc_muh.py`` build ``fd.ln(k0_j)``
inside the log-rate branch.  When a reaction is configured with
``k0 = 0`` (used as "disabled" in pure-channel diagnostic studies, e.g.
the M3a.2 parallel 2e/4e study), the unguarded form contains
``ln(0) = -inf`` and Newton fails at iter 0.

The guard short-circuits non-positive ``k0`` to ``R_j = fd.Constant(0.0)``,
preserving ``bv_k0_funcs`` / ``bv_alpha_funcs`` / ``bv_rate_exprs`` index
alignment so the Stage 4 k0-continuation pattern
``ctx["bv_k0_funcs"][j].assign(...)`` still works for the disabled slot.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from conftest import skip_without_firedrake


def _make_parallel_reactions(*, k0_R2e: float, k0_R4e: float):
    """Build a 2-reaction parallel 2e/4e list with explicit k0 values.

    Mirrors ``PARALLEL_2E_4E_REACTIONS`` in ``scripts/_bv_common.py`` but
    lets the test choose ``k0`` directly (including ``0.0`` to exercise
    the disabled-reaction guard).
    """
    from scripts._bv_common import (
        ALPHA_R2E,
        ALPHA_R4E,
        E_EQ_R2E_V,
        E_EQ_R4E_V,
        C_HP_HAT,
    )
    return [
        {
            "k0": float(k0_R2e),
            "alpha": float(ALPHA_R2E),
            "cathodic_species": 0,
            "anodic_species": 1,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": 2,
            "reversible": True,
            "E_eq_v": float(E_EQ_R2E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {
            "k0": float(k0_R4e),
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]


def _build_ctx(formulation: str, *, k0_R2e: float, k0_R4e: float, mesh_n: int = 8):
    """Build a parallel-2e/4e ctx with the requested k0 values."""
    from scripts._bv_common import (
        setup_firedrake_env,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        make_bv_solver_params,
    )

    setup_firedrake_env()

    import firedrake as fd
    from Forward.bv_solver import (
        build_context,
        build_forms,
        set_initial_conditions,
    )

    mesh = fd.UnitSquareMesh(mesh_n, mesh_n)
    reactions = _make_parallel_reactions(k0_R2e=k0_R2e, k0_R4e=k0_R4e)
    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation=formulation,
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        initializer="linear_phi",
        bv_reactions=reactions,
    )
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    return ctx, sp


def _residual_is_finite(ctx) -> bool:
    """Assemble the residual at the IC and verify all entries are finite.

    Without the guard, ``fd.ln(0)`` poisons one of the surface terms
    inside ``F_res`` and the assembled residual has non-finite entries.
    """
    import firedrake as fd
    import numpy as np

    F = fd.assemble(ctx["F_res"])
    for sub in F.dat:
        if not np.all(np.isfinite(sub.data_ro)):
            return False
    return True


@skip_without_firedrake
@pytest.mark.slow
@pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
class TestDisabledReactionGuard:
    """``k0 <= 0`` must not poison the log-rate form with ``fd.ln(0)``."""

    def test_residual_finite_when_R4e_disabled(self, formulation):
        from scripts._bv_common import K0_HAT_R2E
        ctx, _ = _build_ctx(
            formulation, k0_R2e=float(K0_HAT_R2E), k0_R4e=0.0,
        )
        assert _residual_is_finite(ctx), (
            "F_res must have finite entries when k0_R4e=0; non-finite "
            "indicates fd.ln(0) leaked into the form."
        )

    def test_residual_finite_when_R2e_disabled(self, formulation):
        from scripts._bv_common import K0_HAT_R4E
        ctx, _ = _build_ctx(
            formulation, k0_R2e=0.0, k0_R4e=float(K0_HAT_R4E),
        )
        assert _residual_is_finite(ctx), (
            "F_res must have finite entries when k0_R2e=0."
        )

    def test_disabled_rate_assembles_to_zero(self, formulation):
        """The disabled reaction's rate expression integrates to exactly 0
        on the electrode boundary."""
        import firedrake as fd
        from scripts._bv_common import K0_HAT_R2E

        ctx, _ = _build_ctx(
            formulation, k0_R2e=float(K0_HAT_R2E), k0_R4e=0.0,
        )
        assert len(ctx["bv_rate_exprs"]) == 2

        mesh = ctx["U"].function_space().mesh()
        electrode_marker = 3  # make_bv_solver_params default
        val = fd.assemble(
            ctx["bv_rate_exprs"][1] * fd.ds(electrode_marker, domain=mesh)
        )
        assert math.isfinite(val)
        assert abs(val) < 1e-30, (
            f"Disabled reaction rate must integrate to 0; got {val!r}."
        )

    def test_index_preserved_when_disabled(self, formulation):
        """Stage 4 k0-continuation needs ``ctx["bv_k0_funcs"][1]`` to be
        a mutable Firedrake Function even when R4e is initially disabled.
        """
        import firedrake as fd
        from scripts._bv_common import K0_HAT_R2E

        ctx, _ = _build_ctx(
            formulation, k0_R2e=float(K0_HAT_R2E), k0_R4e=0.0,
        )
        assert len(ctx["bv_k0_funcs"]) == 2
        assert len(ctx["bv_alpha_funcs"]) == 2
        assert isinstance(ctx["bv_k0_funcs"][1], fd.Function)
        assert isinstance(ctx["bv_alpha_funcs"][1], fd.Function)
        # Stage 4 will call .assign(...) on this Function to ramp k0_R4e.
        ctx["bv_k0_funcs"][1].assign(1e-30)
        assert float(ctx["bv_k0_funcs"][1].dat.data_ro[0]) == pytest.approx(1e-30)

    def test_both_active_form_builds(self, formulation):
        """Regression: when both k0 > 0, the guard must NOT fire and the
        form must build identically to its pre-guard behavior."""
        from scripts._bv_common import K0_HAT_R2E, K0_HAT_R4E

        ctx, _ = _build_ctx(
            formulation,
            k0_R2e=float(K0_HAT_R2E),
            k0_R4e=float(K0_HAT_R4E),
        )
        assert _residual_is_finite(ctx)
        assert len(ctx["bv_rate_exprs"]) == 2
