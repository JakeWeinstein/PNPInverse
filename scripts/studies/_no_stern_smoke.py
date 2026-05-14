"""Tiny smoke to dump SNES diagnostics for the no-Stern anchor failure.

Builds the same SolverParams as solver_demo_slide15_no_speculative_cs.py
with --no-stern and tries the anchor at V=0V with snes_monitor on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = str(_THIS_DIR.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
import numpy as np

from scripts._bv_common import (
    ALPHA_R1, ALPHA_R2E, ALPHA_R4E,
    C_HP_HAT, C_O2_HAT, H2O2_SEED_NONDIM,
    D_H2O2_HAT, D_HP_HAT, D_O2_HAT,
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    E_EQ_R2E_V, E_EQ_R4E_V,
    K0_HAT_R1, K0_HAT_R2E, K0_HAT_R4E,
    SNES_OPTS_CHARGED,
    SpeciesConfig,
    V_T,
    make_bv_solver_params,
    setup_firedrake_env,
)

setup_firedrake_env()

import firedrake as fd
import firedrake.adjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.anchor_continuation import (
    LadderExhausted,
    solve_anchor_with_continuation,
)


C_SCALE_LOCAL = 1.2
_N_A = 6.02214076e23


def _a_nondim_from_radius_m(r_m: float) -> float:
    return (4.0 / 3.0) * math.pi * r_m ** 3 * _N_A * C_SCALE_LOCAL


A_O2_PHYSICAL = _a_nondim_from_radius_m(1.70e-10)
A_H2O2_PHYSICAL = _a_nondim_from_radius_m(2.00e-10)
A_HP_PHYSICAL = _a_nondim_from_radius_m(2.80e-10)


def main(no_stern: bool, anchor_v: float, initializer: str) -> int:
    species = SpeciesConfig(
        n_species=3,
        z_vals=[0, 0, 1],
        d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
        a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL],
        c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT],
        stoichiometry_r1=[-1, +1, -2],
        stoichiometry_r2=[0, -1, -2],
        k0_legacy=[K0_HAT_R1] * 3,
        alpha_legacy=[ALPHA_R1] * 3,
        stoichiometry_legacy=[-1, -1, -1],
        c_ref_legacy=[1.0, 0.0, 1.0],
        roles=["neutral", "neutral", "proton"],
    )

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        # Diagnostics:
        "snes_monitor":              None,
        "snes_converged_reason":     None,
        "snes_view":                 None,
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
            "k0": float(K0_HAT_R4E),
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

    if no_stern == "large":
        stern_kw = {"stern_capacitance_f_m2": 10.0}
    elif no_stern is True:
        stern_kw = {}
    else:
        stern_kw = {"stern_capacitance_f_m2": 0.10}

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=species,
        snes_opts=snes_opts,
        formulation="logc_muh",
        log_rate=True,
        u_clamp=100.0,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        initializer=initializer,
        l_eff_m=100e-6,
        **stern_kw,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    sp = sp.with_phi_applied(float(anchor_v) / V_T)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0, domain_height_hat=1.0)

    print(f"=== anchor smoke: no_stern={no_stern}, V={anchor_v:+.3f}V, "
          f"initializer={initializer} ===", flush=True)
    print(f"k0_R2e={K0_HAT_R2E:.3e}, k0_R4e={K0_HAT_R4E:.3e}", flush=True)

    try:
        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp, mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
                max_inserts_per_step=4,
                ic_at_target=True,
            )
        print(f"\nresult.converged = {result.converged}", flush=True)
        print(f"ladder_history = {list(result.ladder_history)!r}", flush=True)
    except LadderExhausted as exc:
        print(f"\nLadderExhausted: {exc}", flush=True)
    except Exception as exc:
        print(f"\nException: {type(exc).__name__}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stern", action="store_true")
    parser.add_argument("--stern-large", action="store_true",
                        help="Use C_S=10 F/m^2 (near no-Stern approximation)")
    parser.add_argument("--anchor-v", type=float, default=0.0)
    parser.add_argument("--initializer", default="linear_phi",
                        choices=("linear_phi", "debye_boltzmann"))
    args = parser.parse_args()
    mode = "large" if args.stern_large else args.no_stern
    sys.exit(main(mode, args.anchor_v, args.initializer))
