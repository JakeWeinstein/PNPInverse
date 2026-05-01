"""Test: z=0 (neutral) solutions across the full voltage range.

z=0 eliminates the Poisson coupling, giving pure diffusion + BV kinetics.
This should converge everywhere and capture the onset shape for k0 identification.
"""
from __future__ import annotations
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def main():
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.sweep_order import _build_sweep_order

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    # Full range including onset
    V_RHE = np.sort(np.array([
        -0.50, -0.40, -0.30, -0.20, -0.10,
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.50, 0.60, 0.70,
    ]))[::-1]
    PHI_HAT = V_RHE / V_T

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 400,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
        },
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    n = n_species

    print(f"{'='*70}")
    print(f"  NEUTRAL (z=0) FULL RANGE TEST")
    print(f"  {len(V_RHE)} points, V_RHE: [{V_RHE.min():.2f}, {V_RHE.max():.2f}]")
    print(f"{'='*70}\n")

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    observable_scale = -I_SCALE

    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)
        set_initial_conditions(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func")
    dtc = ctx.get("dt_const"); dti = float(dt)

    # Force z=0 (neutral)
    for zci in zc:
        zci.assign(0.0)

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)
    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

    dt_max = dti * 20.0

    def _ss(max_steps):
        dc = dti; dtc.assign(dti); pf = pd = None; sc = 0
        for s in range(1, max_steps+1):
            try: sol.solve()
            except: return False, s-1
            Up.assign(U)
            fv = float(fd.assemble(of))
            if pf is not None:
                d = abs(fv-pf); sv = max(abs(fv),abs(pf),1e-8)
                sc = sc+1 if (d/sv<=1e-4 or d<=1e-8) else 0
                if pd and d>0:
                    r = pd/d
                    dc = min(dc*min(r,4.0),dt_max) if r>1 else max(dc*0.5,dti)
                    dtc.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, max_steps

    sweep = _build_sweep_order(PHI_HAT)
    cd = np.full(len(V_RHE), np.nan)
    pc = np.full(len(V_RHE), np.nan)
    hub = None; pe = 0.0

    print("Solving (z=0, neutral)...\n")
    t0 = time.time()

    with adj.stop_annotating():
        for p, oi in enumerate(sweep):
            ei = float(PHI_HAT[oi])
            vi = V_RHE[oi]

            if p > 0 and np.sign(ei) != np.sign(pe) and np.sign(ei) != 0 and hub:
                for s, d in zip(hub, U.dat): d.data[:] = s
                Up.assign(U)

            if p > 0 and abs(ei-pe) > 2.0:
                for br in np.linspace(pe, ei, max(2, int(abs(ei-pe)/2.0))+1)[1:-1]:
                    paf.assign(br); _ss(20)

            paf.assign(ei)
            ok, steps = _ss(100 if p==0 else 20)
            if ok or steps > 0:
                cd[oi] = float(fd.assemble(ocd))
                pc[oi] = float(fd.assemble(opc))
            if not hub: hub = tuple(d.data_ro.copy() for d in U.dat)
            pe = ei

    elapsed = time.time() - t0
    n_ok = sum(1 for x in cd if not np.isnan(x))

    print(f"\n{'='*70}")
    print(f"  RESULTS: {n_ok}/{len(V_RHE)} converged ({elapsed:.1f}s)")
    print(f"{'='*70}")

    sort_idx = np.argsort(V_RHE)
    for i in sort_idx:
        print(f"  V={V_RHE[i]:+.3f}V: cd={cd[i]:+.6f}  pc={pc[i]:+.6f}")

    # Also compare with z=1 at a few cathodic points to see the difference
    print(f"\n  z=0 vs z=1 comparison (cathodic points from earlier tests):")
    print(f"  V=-0.50V: z=0 cd={cd[list(V_RHE).index(-0.50)]:+.4f}, z=1 cd=-0.1839")
    print(f"  V= 0.00V: z=0 cd={cd[list(V_RHE).index( 0.00)]:+.4f}, z=1 cd=-0.1733")
    print(f"  V=+0.10V: z=0 cd={cd[list(V_RHE).index( 0.10)]:+.4f}, z=1 cd=-0.1629")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(_ROOT, "StudyResults", "neutral_full_range")
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    mask = ~np.isnan(cd)
    ax1.plot(V_RHE[sort_idx][mask[sort_idx]], cd[sort_idx][mask[sort_idx]], "o-",
             linewidth=2, markersize=5, label="z=0 (neutral)")
    ax1.set_xlabel("Applied Voltage (V vs RHE)", fontsize=12)
    ax1.set_ylabel("Total Current Density (mA/cm²)", fontsize=12)
    ax1.set_title("Total Current (z=0)", fontsize=13)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.axvline(E_EQ_R1, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label=f"E_eq_r1={E_EQ_R1}V")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    mask_pc = ~np.isnan(pc)
    ax2.plot(V_RHE[sort_idx][mask_pc[sort_idx]], pc[sort_idx][mask_pc[sort_idx]], "o-",
             linewidth=2, markersize=5, color="darkorange", label="z=0 (neutral)")
    ax2.set_xlabel("Applied Voltage (V vs RHE)", fontsize=12)
    ax2.set_ylabel("Peroxide Current Density (mA/cm²)", fontsize=12)
    ax2.set_title("Peroxide Current (z=0)", fontsize=13)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.axvline(E_EQ_R1, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label=f"E_eq_r1={E_EQ_R1}V")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "neutral_full_range.png"), dpi=150)
    print(f"\n  Plot saved to {out_dir}/neutral_full_range.png")


if __name__ == "__main__":
    main()
