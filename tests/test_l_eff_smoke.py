"""Smoke tests for the L_eff transport-domain knob.

The L_eff sweep (``.claude/plans/l-eff-transport-sweep.md``) needs the
mesh y-extent to decouple from L_REF so the H+ Levich ceiling
``F * D_H * c_H_bulk / L_REF * 0.1`` can be tested as a function of a
swept transport-domain height ``L_eff_m`` without rescaling every nondim
transport coefficient.

These tests cover:

  - ``make_graded_rectangle_mesh(domain_height_hat=...)`` builds a mesh
    with the requested y-extent and validates input.
  - ``make_bv_solver_params(l_eff_m=...)`` plumbs ``domain_height_hat``
    onto the ``bv_convergence`` config block.
  - The default (``domain_height_hat=1.0``) is byte-equivalent to the
    legacy unit-cube mesh — no production-target regression.
  - A slow anchor build at ``L_eff = 16 µm`` (smallest planned point in
    the sweep) on the 3sp + Bikerman + Stern stack converges so that
    Q3 (the actual driver) does not blow up at the smallest L_eff.
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

from tests.conftest import skip_without_firedrake


# ===================================================================
# Q1: Config knob plumbing — pure-Python, no Firedrake required
# ===================================================================


class TestLEffConfigPlumbing:
    """``l_eff_m`` flows through ``make_bv_solver_params`` correctly."""

    def _build(self, **overrides):
        from scripts._bv_common import (
            DEFAULT_CLO4_BOLTZMANN_COUNTERION,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
        )
        kwargs = dict(
            eta_hat=0.0,
            dt=0.25,
            t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation="logc",
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        )
        kwargs.update(overrides)
        return make_bv_solver_params(**kwargs)

    def test_default_l_eff_is_l_ref(self):
        """Default = L_REF: domain_height_hat=1.0 (legacy back-compat)."""
        sp = self._build()
        bv_conv = sp[10]["bv_convergence"]
        assert bv_conv["domain_height_hat"] == pytest.approx(1.0)

    def test_l_eff_smaller_than_l_ref(self):
        """L_eff = 16 µm => domain_height_hat = 0.16."""
        sp = self._build(l_eff_m=16e-6)
        bv_conv = sp[10]["bv_convergence"]
        assert bv_conv["domain_height_hat"] == pytest.approx(0.16, rel=1e-9)

    def test_l_eff_larger_than_l_ref(self):
        """L_eff = 200 µm => domain_height_hat = 2.0 (still in sanity range)."""
        sp = self._build(l_eff_m=200e-6)
        bv_conv = sp[10]["bv_convergence"]
        assert bv_conv["domain_height_hat"] == pytest.approx(2.0, rel=1e-9)


# ===================================================================
# Q2: Mesh y-extent validation and behavior — needs Firedrake
# ===================================================================


@skip_without_firedrake
class TestMeshDomainHeightHat:
    """``make_graded_rectangle_mesh`` y-extent is honored and validated."""

    def test_default_mesh_y_max_is_one(self):
        """domain_height_hat=1.0 (default) reproduces legacy [0,1] mesh."""
        from Forward.bv_solver import make_graded_rectangle_mesh
        mesh = make_graded_rectangle_mesh(Nx=4, Ny=10, beta=2.0)
        coords = mesh.coordinates.dat.data_ro
        y_max = float(np.max(coords[:, 1]))
        assert y_max == pytest.approx(1.0, abs=1e-12)

    def test_explicit_unit_height_matches_default(self):
        """Passing domain_height_hat=1.0 explicitly yields the same mesh."""
        from Forward.bv_solver import make_graded_rectangle_mesh
        mesh_default = make_graded_rectangle_mesh(Nx=4, Ny=10, beta=2.0)
        mesh_explicit = make_graded_rectangle_mesh(
            Nx=4, Ny=10, beta=2.0, domain_height_hat=1.0,
        )
        np.testing.assert_array_equal(
            mesh_default.coordinates.dat.data_ro,
            mesh_explicit.coordinates.dat.data_ro,
        )

    def test_short_mesh_y_max_matches_request(self):
        """domain_height_hat=0.16 => y_max = 0.16 (16 µm at L_REF=100 µm)."""
        from Forward.bv_solver import make_graded_rectangle_mesh
        mesh = make_graded_rectangle_mesh(
            Nx=4, Ny=10, beta=2.0, domain_height_hat=0.16,
        )
        coords = mesh.coordinates.dat.data_ro
        y_max = float(np.max(coords[:, 1]))
        y_min = float(np.min(coords[:, 1]))
        assert y_max == pytest.approx(0.16, abs=1e-12)
        assert y_min == pytest.approx(0.0, abs=1e-12)

    def test_grading_preserved_under_scaling(self):
        """Smaller mesh has tighter spacing near y=0 (relative ratio preserved)."""
        from Forward.bv_solver import make_graded_rectangle_mesh
        beta = 3.0
        mesh_unit = make_graded_rectangle_mesh(
            Nx=4, Ny=10, beta=beta, domain_height_hat=1.0,
        )
        mesh_short = make_graded_rectangle_mesh(
            Nx=4, Ny=10, beta=beta, domain_height_hat=0.16,
        )
        # Relative y-coord should be identical (short mesh is just a scale).
        y_unit_sorted = np.sort(np.unique(mesh_unit.coordinates.dat.data_ro[:, 1]))
        y_short_sorted = np.sort(np.unique(mesh_short.coordinates.dat.data_ro[:, 1]))
        np.testing.assert_allclose(
            y_short_sorted / 0.16, y_unit_sorted, rtol=1e-12, atol=1e-14,
        )

    def test_rejects_zero_height(self):
        from Forward.bv_solver import make_graded_rectangle_mesh
        with pytest.raises(ValueError, match="positive"):
            make_graded_rectangle_mesh(
                Nx=4, Ny=10, beta=2.0, domain_height_hat=0.0,
            )

    def test_rejects_negative_height(self):
        from Forward.bv_solver import make_graded_rectangle_mesh
        with pytest.raises(ValueError, match="positive"):
            make_graded_rectangle_mesh(
                Nx=4, Ny=10, beta=2.0, domain_height_hat=-0.5,
            )

    def test_rejects_too_small(self):
        """Below 1e-3 nondim (= 0.1 µm at L_REF=100 µm) — fail loudly."""
        from Forward.bv_solver import make_graded_rectangle_mesh
        with pytest.raises(ValueError, match="sanity"):
            make_graded_rectangle_mesh(
                Nx=4, Ny=10, beta=2.0, domain_height_hat=1e-6,
            )

    def test_rejects_too_large(self):
        """Above 10 nondim (= 1 mm at L_REF=100 µm) — fail loudly."""
        from Forward.bv_solver import make_graded_rectangle_mesh
        with pytest.raises(ValueError, match="sanity"):
            make_graded_rectangle_mesh(
                Nx=4, Ny=10, beta=2.0, domain_height_hat=100.0,
            )


# ===================================================================
# Q2 slow smoke: anchor build at smallest L_eff
# ===================================================================


@pytest.mark.slow
@skip_without_firedrake
def test_anchor_at_l_eff_16um():
    """Anchor build at +0.55 V succeeds at the smallest planned L_eff.

    This is the smoke check that gates Q3 (the sweep driver): if the
    smallest L_eff (16 µm) cannot anchor at +0.55 V on the multi-ion
    (Cs+/SO4--) Bikerman+Stern stack, the L_eff sweep should not run.
    """
    import firedrake.adjoint as adj

    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E, K0_HAT_R4E,
        ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V,
        C_HP_HAT, V_T,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        solve_anchor_with_continuation,
    )

    L_EFF_M = 16e-6
    L_REF = 1e-4
    domain_height_hat = L_EFF_M / L_REF

    mesh = make_graded_rectangle_mesh(
        Nx=8, Ny=80, beta=3.0, domain_height_hat=domain_height_hat,
    )

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })
    rxns = [
        {
            "k0": float(K0_HAT_R2E),
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
            "k0": float(K0_HAT_R4E) * 1e-18,  # ratio = 1e-18, the ref point
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
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc_muh", log_rate=True,
        u_clamp=100.0,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer="debye_boltzmann",
        l_eff_m=L_EFF_M,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    sp_anchor = sp.with_phi_applied(0.55 / float(V_T))
    k0_targets = {0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E) * 1e-18}

    try:
        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
                max_inserts_per_step=4,
                ic_at_target=True,
            )
    except LadderExhausted as exc:
        pytest.fail(
            f"anchor build LadderExhausted at L_eff=16 µm: {exc}.  "
            f"This is the gate test for the L_eff sweep; if it fails, "
            f"raise MAX_INSERTS_PER_STEP or insert a 1e-15 ladder rung."
        )

    assert result.converged, (
        f"anchor build did not converge at L_eff=16 µm; "
        f"ladder_history={result.ladder_history!r}"
    )
