"""Anchor-only smoke for fast-realignment plan §4 — validates the
multi-ion stack converges at V_RHE = +0.55 V before launching the
full grid.

Pass A only (pure-2e), Ny=80 (instead of 200) for fast turnaround.
If this converges with finite gross R_2e, the full driver in
``peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`` is safe to launch.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


ANCHOR_V_RHE = +0.55
MESH_NY = 80
EXPONENT_CLIP = 100.0


def main():
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        V_T, I_SCALE,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E, K0_HAT_R4E, ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V, C_HP_HAT,
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
            "k0": 0.0,    # disabled (pure-2e Pass A)
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
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    cd = np.full(1, np.nan)
    R2e = np.full(1, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_R2e = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        R2e[orig_idx] = float(fd.assemble(f_R2e))

    phi_hat_grid = np.array([ANCHOR_V_RHE]) / V_T
    print(f"Anchor smoke @ V_RHE={ANCHOR_V_RHE:+.3f} V (Ny={MESH_NY})")
    print(f"  multi-ion:  Cs+ + SO4-- + H+ at I=0.3 M")
    print(f"  formulation: logc_muh + log_rate + Stern (C_S=0.10) + Bikerman (multi)")
    print(f"  IC: debye_boltzmann (multi-ion shared-theta closure)")
    t0 = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=8,
            bisect_depth_warm=5,
            per_point_callback=_grab,
        )
    elapsed = time.time() - t0
    pt = result.points[0]
    print(f"\n  converged = {pt.converged}  method = {pt.method}")
    print(f"  z_achieved = {pt.achieved_z_factor}")
    print(f"  cd = {float(cd[0]):+.4e} mA/cm²")
    print(f"  R_2e = {float(R2e[0]):+.4e} mA/cm²")
    print(f"  wall = {elapsed:.1f}s")

    out_dir = os.path.join(_ROOT, "StudyResults", "fast_realignment_2026-05-08", "anchor_smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anchor_smoke.json")
    with open(out_path, "w") as f:
        json.dump({
            "anchor_v_rhe": float(ANCHOR_V_RHE),
            "mesh_Ny": int(MESH_NY),
            "exponent_clip": float(EXPONENT_CLIP),
            "converged": bool(pt.converged),
            "method": str(pt.method),
            "z_achieved": float(pt.achieved_z_factor),
            "cd_mA_cm2": float(cd[0]) if np.isfinite(cd[0]) else None,
            "R_2e_mA_cm2": float(R2e[0]) if np.isfinite(R2e[0]) else None,
            "wall_seconds": float(elapsed),
            "diagnostics": pt.diagnostics,
        }, f, indent=2, default=str)
    print(f"\n  output -> {out_path}")


if __name__ == "__main__":
    main()
