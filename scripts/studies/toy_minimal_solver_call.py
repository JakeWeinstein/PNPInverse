"""Toy minimal example: how to call the PNP-BV forward solver.

READ THIS FIRST if you're new to the repo. It is the smallest honest
end-to-end call of the production stack — build params, build a mesh,
converge an anchor, walk a tiny V_RHE grid, read the current density.

It deliberately mirrors the production driver
``scripts/studies/drivers/solver_demo_slide15_dual_pathway_cs.py`` but
strips out the dual-pathway routes, OCP shift, route ledger, and full
25-point grid. For anything real, copy that driver, not this file.

Run from PNPInverse/ with the firedrake venv active:

    source ../venv-firedrake/bin/activate
    MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 \
      FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1 \
      python -u scripts/studies/toy_minimal_solver_call.py

Canonical recipe (see CLAUDE.md "Calling the production solver"):
    make_bv_solver_params  ->  anchor  ->  Stern bump  ->  grid walk.
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    # -- 0. Constants + Firedrake environment -----------------------------
    # _bv_common is the single source of truth for presets, scales, and
    # nondim constants. setup_firedrake_env() sets the cache env vars.
    from scripts._bv_common import (
        setup_firedrake_env,
        make_bv_solver_params,
        THREE_SPECIES_LOGC_BOLTZMANN,          # 3 dynamic species: O2, H2O2, H+
        PARALLEL_2E_4E_REACTIONS,              # Ruggiero R2e (0.695V) + R4e (1.23V)
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,   # analytic Cs+ counterion
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,  # analytic SO4(2-) counterion
        K0_HAT_R2E, K0_HAT_R4E,                # production rate constants (nondim)
        V_T,                                   # thermal voltage (V) -> phi = V/V_T
        I_SCALE,                               # current-density scale (mA/cm^2)
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation,
        extract_preconverged_anchor,
        set_stern_capacitance_model,
    )
    from Forward.bv_solver.grid_per_voltage import (
        snapshot_U, solve_grid_with_anchor,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    # -- 1. Build solver params (the "what physics" object) ---------------
    # This is the canonical factory. Don't hand-wire flags; pass presets.
    # NOTE the two-Stern trick below: we build the ANCHOR params at the
    # easy C_S = 0.10 F/m^2 and the GRID params at the production
    # C_S = 0.20 F/m^2. The anchor's Newton solve does not converge if you
    # ask for 0.20 cold (see CLAUDE.md / memory project_no_stern_bump_ladder).
    def _make_sp(stern_c_s: float):
        return make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=80.0,         # pseudo-transient -> steady state
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation="logc_muh",                    # proton electrochem-potential var
            log_rate=True,                             # log-rate Butler-Volmer
            bv_reactions=PARALLEL_2E_4E_REACTIONS,
            boltzmann_counterions=[
                DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
                DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            ],
            multi_ion_enabled=True,                    # required for >=2 counterions
            stern_capacitance_f_m2=stern_c_s,
            initializer="debye_boltzmann",             # composite-psi + multispecies-gamma IC
            l_eff_m=100e-6,                            # RRDE diffusion length (100 um)
        )

    STERN_ANCHOR, STERN_PROD = 0.10, 0.20
    sp_anchor = _make_sp(STERN_ANCHOR).with_phi_applied(0.0 / V_T)  # anchor at V_RHE = 0
    sp_grid = _make_sp(STERN_PROD)

    # k0_targets tells the continuation which production rate to ramp UP to,
    # keyed by reaction index into PARALLEL_2E_4E_REACTIONS.
    k0_targets = {0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)}

    # -- 2. Mesh ----------------------------------------------------------
    # domain_height_hat MUST match l_eff_m / L_REF (100um / 100um = 1.0 here).
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=300, beta=2.0,
                                      domain_height_hat=1.0)

    # -- 3. Anchor: cold-start one voltage with k0 continuation -----------
    # stop_annotating() keeps the continuation off the adjoint tape.
    with adj.stop_annotating():
        anchor_result = solve_anchor_with_continuation(
            sp_anchor, mesh=mesh, k0_targets=k0_targets,
        )
    if not anchor_result.converged:
        raise SystemExit("anchor failed to converge")
    print(f"anchor converged ({len(anchor_result.ladder_history)} ladder rungs)")

    # -- 4. Stern bump: 0.10 -> 0.20 F/m^2 on the live ctx ----------------
    ctx = anchor_result.ctx
    with adj.stop_annotating():
        set_stern_capacitance_model(ctx, STERN_PROD)
        ctx["_last_solver"].solve()
    print(f"Stern bumped to {STERN_PROD} F/m^2")

    # Freeze the bumped state into a portable anchor. We snapshot U AFTER
    # the bump (extract_preconverged_anchor would capture the pre-bump
    # 0.10 state), so we build the frozen anchor by hand from the ctx.
    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=0.0 / V_T,
        k0_targets=k0_targets,
        mesh_dof_count=int(ctx["U"].function_space().dim()),
    )
    # overwrite the snapshot with the post-bump state:
    anchor = anchor.__class__(
        phi_applied_eta=anchor.phi_applied_eta,
        U_snapshot=tuple(np.asarray(a).copy() for a in snapshot_U(ctx["U"])),
        k0_targets=anchor.k0_targets,
        mesh_dof_count=anchor.mesh_dof_count,
        ladder_history=anchor.ladder_history,
    )

    # -- 5. Grid walk: warm-walk a few voltages off the anchor ------------
    # Solver convention: V_M = V_RHE directly (psi_bulk = 0, no OCP shift).
    v_rhe = np.linspace(-0.1, 0.3, 5)
    phi_grid = v_rhe / float(V_T)          # solver works in eta = V / V_T

    cd = []

    def _capture(orig_idx, _phi, ctx_pt):
        # Total electron-weighted current density, mA/cm^2 (cathodic < 0).
        f_cd = _build_bv_observable_form(
            ctx_pt, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        cd.append((orig_idx, float(fd.assemble(f_cd))))

    with adj.stop_annotating():
        grid_result = solve_grid_with_anchor(
            sp_grid, anchor=anchor, phi_applied_values=phi_grid, mesh=mesh,
            per_point_callback=_capture,
        )

    n_ok = sum(p.converged for p in grid_result.points)
    print(f"grid: {n_ok}/{len(v_rhe)} converged\n")
    print(f"{'V_RHE (V)':>10}  {'current density (mA/cm^2)':>26}")
    for idx, val in sorted(cd):
        print(f"{v_rhe[idx]:>10.3f}  {val:>26.5f}")


if __name__ == "__main__":
    main()
