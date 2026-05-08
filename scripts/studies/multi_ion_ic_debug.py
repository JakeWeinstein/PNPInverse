"""Direct probe of the multi-ion IC at V_RHE = +0.55 V to see why
``_try_debye_boltzmann_ic_muh`` is failing for the Cs⁺/SO₄²⁻ stack.

Builds context, runs IC, prints picard iters + fallback reason +
spatial-IC field statistics so we can identify where the multi-ion
branch is going wrong.
"""
from __future__ import annotations

import os
import sys
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        V_T,
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
        build_context, build_forms,
    )
    from Forward.bv_solver.forms_logc_muh import _try_debye_boltzmann_ic_muh

    ANCHOR_V_RHE = +0.55
    rxns = [
        {"k0": float(K0_HAT_R2E), "alpha": float(ALPHA_R2E),
         "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
         "stoichiometry": [-1, +1, -2], "n_electrons": 2, "reversible": True,
         "E_eq_v": float(E_EQ_R2E_V),
         "cathodic_conc_factors": [
             {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)}]},
        {"k0": 0.0, "alpha": float(ALPHA_R4E),
         "cathodic_species": 0, "anodic_species": None, "c_ref": 0.0,
         "stoichiometry": [-1, 0, -4], "n_electrons": 4, "reversible": False,
         "E_eq_v": float(E_EQ_R4E_V),
         "cathodic_conc_factors": [
             {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)}]},
    ]
    sp = make_bv_solver_params(
        eta_hat=ANCHOR_V_RHE / V_T, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=SNES_OPTS_CHARGED,
        formulation="logc_muh", log_rate=True, u_clamp=100.0,
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
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0)
    print(f"Building context @ V_RHE={ANCHOR_V_RHE:+.3f} V (Ny=80)")
    with adj.stop_annotating():
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        print("\n--- Calling _try_debye_boltzmann_ic_muh directly ---")
        params = sp.params if hasattr(sp, "params") else sp.solver_options
        sp_list = sp.to_list() if hasattr(sp, "to_list") else list(sp)
        n_species = sp_list[0]
        c0 = sp_list[8]
        phi_applied = sp_list[7]
        ok, reason, picard_iters = _try_debye_boltzmann_ic_muh(
            ctx, sp_list, params, phi_applied, c0, n_species,
        )
        print(f"\nok={ok}  reason={reason!r}  picard_iters={picard_iters}")
        if "initializer_picard_state" in ctx:
            ps = ctx["initializer_picard_state"]
            print(f"\nPicard state:")
            for k in ["R_list", "c_s_list", "phi_o", "psi_D", "psi_S",
                      "phi_surface", "gamma_s", "eta_list"]:
                v = ps.get(k)
                print(f"  {k}: {v}")

        if ok:
            print("\n--- IC succeeded; field statistics ---")
            U = ctx["U"]
            for i in range(n_species + 1):
                sub = U.sub(i)
                arr = sub.dat.data_ro
                name = f"sub({i})"
                print(f"  {name}: min={float(np.min(arr)):+.3e} max={float(np.max(arr)):+.3e} mean={float(np.mean(arr)):+.3e}")


if __name__ == "__main__":
    main()
