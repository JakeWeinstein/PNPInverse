"""Bit-exact-on-attractor regression: linear_phi vs debye_boltzmann.

The IC is a path-only change: the converged Newton state must match
between initializers to within solver tolerance, regardless of which
basin Newton walked through to get there.

This test catches an IC that nudges Newton into a parallel basin (which
``test_solver_equivalence.py`` at REL_TOL=5e-3 might not catch -- two
different basins could happen to give CD/PC within 0.5%).

We cold-converge a small subset of the production V grid with each
initializer and assert max relative error on CD and PC is below 1e-6
(or whatever the solver's converged-state numerical noise floor is).

Voltage choice: ``V_RHE in {-0.5, -0.3, 0.0, +0.05, +0.1}`` -- five
points spanning the existing equivalence-test window.  At low V the
analytical IC is expected to fall back to linear-phi (Picard
oscillates against the H+ mass-transport limit), so the comparison is
fair: both paths land at the same SS regardless.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


def _run_with_initializer(initializer: str, V_RHE_grid: np.ndarray, mesh_ny: int):
    """Cold-converge V_RHE_grid with the given initializer flag."""
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN, DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(mesh_ny), beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc", log_rate=True,
        u_clamp=100.0,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer=initializer,
    )

    V_RHE_grid = np.asarray(V_RHE_grid, dtype=float)
    NV = len(V_RHE_grid)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = V_RHE_grid / V_T
    t0 = time.time()
    with adj.stop_annotating():
        solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=4,
            bisect_depth_warm=3,
            per_point_callback=_grab,
        )
    elapsed = time.time() - t0
    return {"cd": cd, "pc": pc, "wall_seconds": elapsed}


@skip_without_firedrake
@pytest.mark.slow
class TestInitializerAttractorEquivalence:
    """Converged states must match between initializers to ~1e-6."""

    V_RHE_GRID = np.array([-0.5, -0.3, 0.0, 0.05, 0.1])
    MESH_NY = 100
    REL_TOL = 1e-4
    ABS_FLOOR_mA_cm2 = 1e-7

    @pytest.fixture(scope="class")
    def both_results(self):
        linear = _run_with_initializer(
            "linear_phi", self.V_RHE_GRID, self.MESH_NY,
        )
        debye = _run_with_initializer(
            "debye_boltzmann", self.V_RHE_GRID, self.MESH_NY,
        )
        return {"linear": linear, "debye": debye}

    def test_both_converge(self, both_results):
        for name in ("linear", "debye"):
            cd = both_results[name]["cd"]
            pc = both_results[name]["pc"]
            assert np.all(np.isfinite(cd)), (
                f"{name}: not all CDs finite: {cd}"
            )
            assert np.all(np.isfinite(pc)), (
                f"{name}: not all PCs finite: {pc}"
            )

    def test_cd_attractor_matches(self, both_results):
        cd_lin = both_results["linear"]["cd"]
        cd_dbe = both_results["debye"]["cd"]
        denom = np.maximum(np.abs(cd_lin), self.ABS_FLOOR_mA_cm2)
        rel_err = np.abs(cd_dbe - cd_lin) / denom
        max_err = float(np.max(rel_err))
        assert max_err < self.REL_TOL, (
            f"CD attractor mismatch beyond {self.REL_TOL}: max_rel_err={max_err:.3e}\n"
            f"  V={self.V_RHE_GRID}\n"
            f"  linear={cd_lin}\n"
            f"  debye ={cd_dbe}\n"
            f"  rel_err={rel_err}"
        )

    def test_pc_attractor_matches(self, both_results):
        pc_lin = both_results["linear"]["pc"]
        pc_dbe = both_results["debye"]["pc"]
        denom = np.maximum(np.abs(pc_lin), self.ABS_FLOOR_mA_cm2)
        rel_err = np.abs(pc_dbe - pc_lin) / denom
        max_err = float(np.max(rel_err))
        assert max_err < self.REL_TOL, (
            f"PC attractor mismatch beyond {self.REL_TOL}: max_rel_err={max_err:.3e}\n"
            f"  V={self.V_RHE_GRID}\n"
            f"  linear={pc_lin}\n"
            f"  debye ={pc_dbe}\n"
            f"  rel_err={rel_err}"
        )
