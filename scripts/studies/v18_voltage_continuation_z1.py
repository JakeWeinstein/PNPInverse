"""V18: Voltage continuation at z=1 from a converged anchor point.

Strategy: Instead of z-ramping at each voltage independently,
1. Get z=1 solution at V_RHE=0.10V (known to converge)
2. Voltage-continue outward in small steps, staying at z=1
3. This changes only a boundary condition, not the PDE structure

Output: StudyResults/v18_voltage_continuation/
"""
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    FOUR_SPECIES_CHARGED, make_bv_solver_params,
    SNES_OPTS_CHARGED,
)
setup_firedrake_env()

import numpy as np
import time
import json
import firedrake as fd
import pyadjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
from Forward.bv_solver.observables import _build_bv_observable_form

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_voltage_continuation")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Anchor voltage where z=1 is known to converge
    V_ANCHOR = 0.10  # V vs RHE
    phi_anchor = V_ANCHOR / V_T

    # Target voltage range (onset region)
    V_TARGETS = np.arange(0.10, 0.75, 0.025)  # 0.10 to 0.725V in 25mV steps

    sp = make_bv_solver_params(
        eta_hat=phi_anchor,
        dt=0.25,
        t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
        E_eq_r1=E_EQ_R1,
        E_eq_r2=E_EQ_R2,
    )

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(list(sp), mesh=mesh)
        ctx = build_forms(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))

    U = ctx["U"]
    Up = ctx["U_prev"]
    zc = ctx["z_consts"]
    n = ctx["n_species"]
    dt_const = ctx["dt_const"]
    paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

    dt_init = 0.25

    def run_ss(max_steps=100, label=""):
        dt_val = dt_init
        dt_const.assign(dt_val)
        prev_flux = None
        prev_delta = None
        steady_count = 0
        for s in range(1, max_steps + 1):
            try:
                sol.solve()
            except Exception as e:
                return False, s - 1, str(e)
            Up.assign(U)
            fv = float(fd.assemble(ocd))
            if prev_flux is not None:
                delta = abs(fv - prev_flux)
                scale = max(abs(fv), abs(prev_flux), 1e-8)
                if delta / scale <= 1e-4 or delta <= 1e-8:
                    steady_count += 1
                else:
                    steady_count = 0
                if prev_delta and delta > 0:
                    r = prev_delta / delta
                    if r > 1:
                        dt_val = min(dt_val * min(r, 4.0), dt_init * 20)
                    else:
                        dt_val = max(dt_val * 0.5, dt_init)
                    dt_const.assign(dt_val)
                prev_delta = delta
            prev_flux = fv
            if steady_count >= 4:
                return True, s, "converged"
        return False, max_steps, "budget_exhausted"

    # ===================================================================
    # Step 1: Get z=1 solution at anchor voltage using z-ramp
    # ===================================================================
    print(f"Step 1: Getting z=1 solution at V_RHE={V_ANCHOR}V (anchor)")
    z_nominal = [float(sp[4][i]) for i in range(n)]  # [0, 0, 1, -1]

    # Phase 1: z=0
    for zci in zc:
        zci.assign(0.0)
    paf.assign(phi_anchor)

    with adj.stop_annotating():
        ok, steps, msg = run_ss(100, "z=0")
    print(f"  z=0: ok={ok}, steps={steps}, cd={float(fd.assemble(ocd)):.6f}")

    # Phase 2: z-ramp 0→1
    z_schedule = np.linspace(0.0, 1.0, 21)[1:]
    with adj.stop_annotating():
        for z_val in z_schedule:
            for i in range(n):
                zc[i].assign(z_nominal[i] * z_val)
            ok, steps, msg = run_ss(60)
            if not (ok or steps > 0):
                print(f"  z={z_val:.2f}: FAILED - anchor point failed!")
                return

    cd_anchor = float(fd.assemble(ocd))
    pc_anchor = float(fd.assemble(opc))
    print(f"  z=1: cd={cd_anchor:.6f}, pc={pc_anchor:.6f}")

    # ===================================================================
    # Step 2: Voltage continuation from anchor outward
    # ===================================================================
    print(f"\nStep 2: Voltage continuation at z=1 from V_RHE={V_ANCHOR}V")

    results = [{
        "V_RHE": V_ANCHOR, "phi_hat": phi_anchor,
        "cd": cd_anchor, "pc": pc_anchor,
        "ok": True, "steps": 0,
    }]

    # Save anchor state as checkpoint
    U_ckpt_anchor = tuple(d.data_ro.copy() for d in U.dat)

    last_good_V = V_ANCHOR
    last_good_U = tuple(d.data_ro.copy() for d in U.dat)

    with adj.stop_annotating():
        for V_target in V_TARGETS[1:]:  # skip the anchor itself
            phi_target = V_target / V_T

            # Checkpoint current state
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)

            paf.assign(phi_target)
            ok, steps, msg = run_ss(40, f"V={V_target:.3f}")

            cd = float(fd.assemble(ocd))
            pc = float(fd.assemble(opc))

            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]

            if ok or (steps > 0 and all(c > -1e-3 for c in c_min)):
                print(f"  V_RHE={V_target:.3f}V: ok={ok}, steps={steps}, "
                      f"cd={cd:.6f}, pc={pc:.6f}, c_min={[f'{c:.2e}' for c in c_min]}")
                results.append({
                    "V_RHE": float(V_target), "phi_hat": float(phi_target),
                    "cd": cd, "pc": pc, "ok": ok, "steps": steps,
                    "c_min": c_min,
                })
                last_good_V = V_target
                last_good_U = tuple(d.data_ro.copy() for d in U.dat)
            else:
                print(f"  V_RHE={V_target:.3f}V: FAILED ({msg}), "
                      f"c_min={[f'{c:.2e}' for c in c_min]}")

                # Restore and try smaller voltage steps
                for src, dst in zip(U_ckpt, U.dat):
                    dst.data[:] = src
                Up.assign(U)

                # Try intermediate steps from last good
                V_step = (V_target - last_good_V) / 4.0
                recovered = False
                for sub in range(1, 5):
                    V_mid = last_good_V + sub * V_step
                    phi_mid = V_mid / V_T

                    U_sub_ckpt = tuple(d.data_ro.copy() for d in U.dat)
                    paf.assign(phi_mid)
                    ok_s, steps_s, msg_s = run_ss(40)
                    cd_s = float(fd.assemble(ocd))
                    pc_s = float(fd.assemble(opc))
                    c_min_s = [float(U.dat[i].data_ro.min()) for i in range(n)]

                    if ok_s or (steps_s > 0 and all(c > -1e-3 for c in c_min_s)):
                        print(f"    sub V={V_mid:.4f}: ok={ok_s}, steps={steps_s}, "
                              f"cd={cd_s:.6f}, c_min={[f'{c:.2e}' for c in c_min_s]}")
                        if abs(V_mid - V_target) < 0.002:
                            results.append({
                                "V_RHE": float(V_target), "phi_hat": float(phi_target),
                                "cd": cd_s, "pc": pc_s, "ok": ok_s, "steps": steps_s,
                            })
                            recovered = True
                        last_good_V = V_mid
                        last_good_U = tuple(d.data_ro.copy() for d in U.dat)
                    else:
                        for src, dst in zip(U_sub_ckpt, U.dat):
                            dst.data[:] = src
                        Up.assign(U)
                        print(f"    sub V={V_mid:.4f}: FAILED")
                        break

                if not recovered:
                    # Log the failure boundary
                    results.append({
                        "V_RHE": float(V_target), "phi_hat": float(phi_target),
                        "cd": float("nan"), "pc": float("nan"),
                        "ok": False, "boundary": True,
                    })
                    print(f"  >> Voltage continuation boundary: V_RHE={last_good_V:.3f}V")
                    # Try to continue from last good point
                    for src, dst in zip(last_good_U, U.dat):
                        dst.data[:] = src
                    Up.assign(U)

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("VOLTAGE CONTINUATION AT Z=1 (Physical E_eq)")
    print("=" * 70)
    print(f"{'V_RHE':>8} {'phi_hat':>8} {'cd':>10} {'pc':>10} {'status':>8}")
    print("-" * 50)
    for r in results:
        V = r["V_RHE"]
        phi = r["phi_hat"]
        cd = r.get("cd", float("nan"))
        pc = r.get("pc", float("nan"))
        ok = r.get("ok", False)
        status = "OK" if ok else "FAIL"
        print(f"{V:8.3f} {phi:8.2f} {cd:10.6f} {pc:10.6f} {status:>8}")

    with open(os.path.join(OUT_DIR, "voltage_continuation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR}/voltage_continuation_results.json")


if __name__ == "__main__":
    main()
