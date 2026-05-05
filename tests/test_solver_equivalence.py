"""Production-codepath equivalence: 3sp + Boltzmann vs 4sp dynamic.

Verifies that the production 3-species + analytic Boltzmann ClO4-
counterion stack and the 4-species fully-dynamic stack (ClO4- as a 4th
NP species, no Boltzmann reduction) produce equivalent observables
(CD, PC) on the same production wiring.

Why they should agree
---------------------
For ClO4- in steady state with no BV reaction (stoichiometry 0 in both
R1 and R2) and the production BCs (Dirichlet at the bulk, natural
no-flux at the electrode and side walls), J = 0 everywhere is the
unique steady-state solution.  From J = -D*c*(grad(u) + z*grad(phi)) = 0
and c > 0::

    grad(u + z*phi) = 0  =>  u = const - z*phi  =>  c = c_bulk*exp(-z*phi)

So the Boltzmann formula isn't a modeling approximation -- it's the
closed-form analytic steady-state solution of the dynamic NP equation
under the production BCs.  The 3sp + Boltzmann reduction is therefore
mathematically *identical* to a 4sp dynamic formulation in continuum,
and we expect the discrete observables to match to within
discretization-error-difference noise.

What this test exercises
------------------------
The same call sequence ``scripts/plot_iv_curve_unified.py`` uses:

  - ``make_bv_solver_params`` factory (production wiring).
  - ``make_graded_rectangle_mesh`` (production graded mesh).
  - ``solve_grid_per_voltage_cold_with_warm_fallback`` (C+D orchestrator).
  - ``_build_bv_observable_form`` (CD, PC extraction).

The only configuration that differs between the two passes is::

  3sp+Boltzmann:  species=THREE_SPECIES_LOGC_BOLTZMANN
                  boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION]
  4sp dynamic:    species=FOUR_SPECIES_LOGC_DYNAMIC
                  boltzmann_counterions=None

Both run through the exact same dispatcher -> forms_logc -> orchestrator
path; ``boltzmann.py:add_boltzmann_counterion_residual`` is a clean
no-op when ``boltzmann_counterions=None``.

Voltage range
-------------
``V_RHE in [-0.5, +0.1]`` -- the historical convergence window for the
4sp dynamic formulation.  The 3sp+Boltzmann stack converges over a
wider range (up to +1.2 V via the warm-walk), but we restrict to the
common window where both formulations are known to converge.
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


# ---------------------------------------------------------------------------
# Production-codepath solve: same call sequence as
# scripts/plot_iv_curve_unified.py:main(), parameterized on species/Boltzmann.
# ---------------------------------------------------------------------------
def _run_production_solve(*, species, boltzmann_counterions, V_RHE_grid, mesh_ny):
    """Run factory + orchestrator + observable extraction.  No standalone
    solvers; every call routes through the production wiring."""
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
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

    # Production E_eq (V vs RHE).
    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(mesh_ny), beta=3.0)

    # Production SNES options (mirrors scripts/plot_iv_curve_unified.py:128).
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
        species=species,
        snes_opts=snes_opts,
        formulation="logc",
        log_rate=True,
        u_clamp=100.0,
        boltzmann_counterions=boltzmann_counterions,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
    )

    V_RHE_grid = np.asarray(V_RHE_grid, dtype=float)
    NV = len(V_RHE_grid)
    cd_mA_cm2 = np.full(NV, np.nan)
    pc_mA_cm2 = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density",  reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_mA_cm2[orig_idx] = float(fd.assemble(f_cd))
        pc_mA_cm2[orig_idx] = float(fd.assemble(f_pc))

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

    return {
        "V_RHE": V_RHE_grid,
        "cd_mA_cm2": cd_mA_cm2,
        "pc_mA_cm2": pc_mA_cm2,
        "wall_seconds": elapsed,
        "n_species": species.n_species,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
@skip_without_firedrake
@pytest.mark.slow
class TestSolverEquivalence4spVs3spBoltzmann:
    """Verify production-codepath equivalence between the two formulations."""

    # Historical convergence window for the 4sp dynamic formulation.
    # The 3sp+Boltzmann stack converges over a wider range (warm-walk up
    # to +1.2 V), but we test the common window.
    V_RHE_GRID = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1])

    # Production uses Ny=200 (matches v18, v24, writeup).  We use 100 to
    # keep the slow-test budget reasonable -- two full sweeps at Ny=100
    # take ~2-3 minutes; at Ny=200 it'd be ~5-6 minutes.  Both
    # formulations route through identical mesh/discretization, so the
    # equivalence comparison is mesh-independent in principle (any
    # discretization-error difference between the two formulations comes
    # from the extra c_ClO4 NP equation in 4sp, which integrates into
    # phi via Poisson coupling).
    MESH_NY = 100

    # Equivalence threshold (hybrid abs/rel).  In continuum the two
    # formulations are mathematically identical; discrete differences
    # are bounded by the discretization error of the c_ClO4 NP equation
    # in 4sp (analytic in 3sp+Boltzmann), which couples back to phi
    # through Poisson.  Observed at Ny=100 (May 2026 baseline):
    #   CD max rel err: 2.8e-3
    #   PC max rel err: 1.0e-3 (using hybrid abs/rel below)
    #   PC pure-rel err: 8.7e-3 at V=-0.1 V where PC ~ 3e-6 mA/cm^2
    #     (the H2O2 production/consumption transition voltage; absolute
    #      diff there is ~3e-8 mA/cm^2, well inside SNES tolerance).
    # The hybrid metric uses absolute tolerance when |observable| <
    # ABS_FLOOR (where relative becomes meaningless near zero crossings),
    # so the threshold doesn't get artificially loosened by zero crossings.
    REL_TOL = 5e-3
    ABS_FLOOR_mA_cm2 = 1e-4   # well below the cathodic CD peak of ~0.18 mA/cm^2

    @pytest.fixture(scope="class")
    def both_results(self):
        """Run both formulations once for the class (fixture caches results)."""
        from scripts._bv_common import (
            THREE_SPECIES_LOGC_BOLTZMANN,
            FOUR_SPECIES_LOGC_DYNAMIC,
            DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        )

        print("\n[3sp + Boltzmann ClO4-] running production solve...")
        r3 = _run_production_solve(
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
            V_RHE_grid=self.V_RHE_GRID,
            mesh_ny=self.MESH_NY,
        )
        print(
            f"  -> {(~np.isnan(r3['cd_mA_cm2'])).sum()}/{len(r3['V_RHE'])} converged "
            f"in {r3['wall_seconds']:.1f}s"
        )

        print("\n[4sp dynamic (ClO4- explicit)] running production solve...")
        r4 = _run_production_solve(
            species=FOUR_SPECIES_LOGC_DYNAMIC,
            boltzmann_counterions=None,
            V_RHE_grid=self.V_RHE_GRID,
            mesh_ny=self.MESH_NY,
        )
        print(
            f"  -> {(~np.isnan(r4['cd_mA_cm2'])).sum()}/{len(r4['V_RHE'])} converged "
            f"in {r4['wall_seconds']:.1f}s"
        )

        return {"3sp": r3, "4sp": r4}

    def _common_converged_mask(self, r3, r4):
        return ~(np.isnan(r3["cd_mA_cm2"]) | np.isnan(r4["cd_mA_cm2"]))

    def _hybrid_err(self, a, b):
        """Hybrid abs/rel error: |a-b| / max(|a|, |b|, ABS_FLOOR).

        Reduces to relative error when both magnitudes are >> ABS_FLOOR;
        becomes ``|a-b| / ABS_FLOOR`` (absolute-style) near zero crossings
        where pure relative error is ill-defined.
        """
        denom = np.maximum.reduce([
            np.abs(a), np.abs(b),
            np.full_like(a, self.ABS_FLOOR_mA_cm2),
        ])
        return np.abs(a - b) / denom

    # ---- Convergence-coverage check ----

    def test_both_converge_in_window(self, both_results):
        """Both formulations should converge at most voltages in the window."""
        r3, r4 = both_results["3sp"], both_results["4sp"]
        n_total = len(self.V_RHE_GRID)
        n3 = int((~np.isnan(r3["cd_mA_cm2"])).sum())
        n4 = int((~np.isnan(r4["cd_mA_cm2"])).sum())
        n_common = int(self._common_converged_mask(r3, r4).sum())

        assert n3 == n_total, (
            f"3sp+Boltzmann converged at only {n3}/{n_total} voltages: "
            f"{r3['V_RHE'][np.isnan(r3['cd_mA_cm2'])].tolist()} failed"
        )
        assert n4 == n_total, (
            f"4sp dynamic converged at only {n4}/{n_total} voltages: "
            f"{r4['V_RHE'][np.isnan(r4['cd_mA_cm2'])].tolist()} failed"
        )
        assert n_common == n_total, (
            f"Only {n_common}/{n_total} voltages converged in both formulations"
        )

    # ---- Observable equivalence ----

    def test_current_density_equivalent(self, both_results):
        """CD agrees to within REL_TOL across the common-converged window."""
        r3, r4 = both_results["3sp"], both_results["4sp"]
        mask = self._common_converged_mask(r3, r4)
        cd3, cd4 = r3["cd_mA_cm2"][mask], r4["cd_mA_cm2"][mask]
        v_grid = r3["V_RHE"][mask]

        err = self._hybrid_err(cd3, cd4)
        max_err = float(err.max())

        assert max_err < self.REL_TOL, (
            f"CD disagreement: max hybrid error {max_err:.3e} exceeds "
            f"threshold {self.REL_TOL:.3e}.\n"
            f"  V:       {v_grid.tolist()}\n"
            f"  3sp CD:  {cd3.tolist()}\n"
            f"  4sp CD:  {cd4.tolist()}\n"
            f"  err:     {err.tolist()}"
        )

    def test_peroxide_current_equivalent(self, both_results):
        """PC agrees to within REL_TOL across the common-converged window."""
        r3, r4 = both_results["3sp"], both_results["4sp"]
        mask = self._common_converged_mask(r3, r4)
        pc3, pc4 = r3["pc_mA_cm2"][mask], r4["pc_mA_cm2"][mask]
        v_grid = r3["V_RHE"][mask]

        err = self._hybrid_err(pc3, pc4)
        max_err = float(err.max())

        assert max_err < self.REL_TOL, (
            f"PC disagreement: max hybrid error {max_err:.3e} exceeds "
            f"threshold {self.REL_TOL:.3e}.\n"
            f"  V:       {v_grid.tolist()}\n"
            f"  3sp PC:  {pc3.tolist()}\n"
            f"  4sp PC:  {pc4.tolist()}\n"
            f"  err:     {err.tolist()}"
        )
