"""Sanity check: does 4sp dynamic + linear_phi + Ny=200 still converge
at the cathodic voltages where it historically worked?

The equivalence test (``tests/test_solver_equivalence.py``) confirms 4sp
dynamic agrees with 3sp+Boltzmann at Ny=100 over V_RHE ∈ [-0.5, +0.1].
The peroxide_window_4sp run from 14:57 today reported 0/29 across V ∈
[-0.5, +1.2] at Ny=200 — but with u_clamp=200 (current default is 100).

This script repeats just the legacy convergence window at Ny=200 with
the current production stack: 4sp dynamic, linear_phi IC, no Stern,
exponent_clip=100, u_clamp=100.  ~3 min wall.

If it converges → the previous 0/29 was config-specific (u_clamp).
If it doesn't → 4sp at Ny=200 is regressed; the IC/clamp work won't
fix it.
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


V_TEST = (-0.50, -0.30, -0.10, 0.00, 0.10)
# Override via positional argv: `python ... 100` runs at Ny=100.
MESH_NY = int(sys.argv[1]) if len(sys.argv) > 1 else 200
EXPONENT_CLIP = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
U_CLAMP = 100.0
OUT_SUBDIR = f"peroxide_window_4sp_cathodic_check_ny{MESH_NY}_clip{int(EXPONENT_CLIP)}"


def main() -> None:
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        FOUR_SPECIES_LOGC_DYNAMIC,
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

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("  4sp dynamic — cathodic sanity check (Ny=200, u_clamp=100)")
    print("=" * 72)
    print(f"  V_RHE grid    = {list(V_TEST)}")
    print(f"  species       = FOUR_SPECIES_LOGC_DYNAMIC")
    print(f"  initializer   = linear_phi (default)")
    print(f"  Stern         = OFF")
    print(f"  Ny            = {MESH_NY}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  u_clamp       = {U_CLAMP}")
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

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
        species=FOUR_SPECIES_LOGC_DYNAMIC,
        snes_opts=snes_opts,
        formulation="logc", log_rate=True,
        u_clamp=U_CLAMP,
        boltzmann_counterions=None,
        stern_capacitance_f_m2=None,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=0.68,        E_eq_r2=1.78,
        initializer="linear_phi",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(V_TEST)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = np.array(V_TEST) / V_T
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

    print()
    print(f"Converged at {sum(result.points[i].converged for i in range(NV))}/{NV} "
          f"in {elapsed:.1f}s")
    print()
    rows = []
    for i, v in enumerate(V_TEST):
        ok = bool(result.points[i].converged)
        method = result.points[i].method
        z = float(result.points[i].achieved_z_factor)
        d = result.points[i].diagnostics or {}
        c_clo4 = d.get("c3_surface_mean")
        c_clo4_s = f"{c_clo4:.2e}" if c_clo4 is not None else "n/a"
        cd_v = cd[i] if np.isfinite(cd[i]) else None
        pc_v = pc[i] if np.isfinite(pc[i]) else None
        cd_s = f"{cd_v:+.3e}" if cd_v is not None else "(none)"
        pc_s = f"{pc_v:+.3e}" if pc_v is not None else "(none)"
        print(f"  V={v:+.3f}  ok={ok}  cd={cd_s}  pc={pc_s}  "
              f"method={method}  z={z:.3f}  c_ClO4={c_clo4_s}")
        rows.append({
            "v_rhe": v, "converged": ok, "method": method, "z_achieved": z,
            "cd_mA_cm2": cd_v, "pc_mA_cm2": pc_v,
            "c_clo4_surface": c_clo4,
        })

    out_path = os.path.join(out_dir, "iv_curve.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "v_rhe": list(V_TEST), "mesh_Ny": MESH_NY,
                "exponent_clip": EXPONENT_CLIP, "u_clamp": U_CLAMP,
                "species": "FOUR_SPECIES_LOGC_DYNAMIC",
                "initializer": "linear_phi",
                "stern_capacitance_f_m2": None,
            },
            "wall_seconds": float(elapsed),
            "rows": rows,
        }, f, indent=2)
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
