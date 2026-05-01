"""Voltage-chain warm-start for the 4sp cathodic range.

Motivation: the hybrid forward study showed cold-start 4sp fails at V<=-0.4
(z=0 steady diverges). If we instead solve at V=-0.10 cold, then walk
downward in small steps (using each solution as the IC for the next), we
should be able to reach deep cathodic (v13 used V=-1.2V) and capture the
transport-limited plateau + non-trivial R_2 regime where PC carries
information about k0_2/alpha_2.

Strategy:
  1. Cold-start at V=V_START via z-ramp.
  2. Walk downward in ΔV steps, each time:
       - checkpoint current U
       - assign new phi_applied
       - run pseudo-transient to new steady state
       - on success: grow step, record (cd, pc), continue
       - on failure: restore U, shrink step, retry (bisection)
  3. Report the deepest V reached + I-V curve.

Saves StudyResults/v19_vchain_4sp/iv_cathodic.npz.
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


E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

V_START = -0.10        # start of chain (cold-start here)
V_TARGET = -1.20       # try to reach this depth (v13 used -1.2V)
DV_INIT = 0.05         # initial step size (magnitude)
DV_MIN = 1.0e-3        # bisection floor
DV_GROW = 1.3          # step growth on success
MIN_STEP_SUCCESSES_BEFORE_GROW = 2


def main():
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        FOUR_SPECIES_CHARGED, make_bv_solver_params, SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms import (
        build_context, build_forms, set_initial_conditions,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    out_dir = os.path.join(_ROOT, "StudyResults", "v19_vchain_4sp")
    os.makedirs(out_dir, exist_ok=True)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # Build the standard 4sp context at true kinetics, starting voltage.
    sp = make_bv_solver_params(
        eta_hat=V_START / V_T, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    with adj.stop_annotating():
        ctx = build_context(list(sp), mesh=mesh)
        ctx = build_forms(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    sp_dict["snes_error_if_not_converged"] = True
    # Slightly more conservative line search for deep cathodic.
    sp_dict["snes_linesearch_maxlambda"] = 0.3

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

    def run_ss(max_steps=60, rel_tol=1e-4, abs_tol=1e-8):
        dt_val = 0.25; dt_const.assign(dt_val)
        prev_flux = None; prev_delta = None; sc = 0
        for _ in range(max_steps):
            try:
                sol.solve()
            except Exception:
                return False
            Up.assign(U)
            fv = float(fd.assemble(form_cd))
            if prev_flux is not None:
                d = abs(fv - prev_flux)
                sv = max(abs(fv), abs(prev_flux), 1e-8)
                if d / sv <= rel_tol or d <= abs_tol:
                    sc += 1
                else:
                    sc = 0
                if prev_delta and d > 0:
                    r = prev_delta / d
                    dt_val = (min(dt_val * min(r, 4), 5.0) if r > 1
                              else max(dt_val * 0.5, 0.25))
                    dt_const.assign(dt_val)
                prev_delta = d
            prev_flux = fv
            if sc >= 4:
                return True
        return False

    # --- Cold start at V_START: z=0 then z-ramp ---
    print("=" * 70)
    print(f"V-CHAIN 4sp CATHODIC: start={V_START:+.3f}V, target={V_TARGET:+.3f}V, "
          f"dV_init={DV_INIT:.3f}")
    print("=" * 70)

    with adj.stop_annotating():
        for zci in zc: zci.assign(0.0)
        paf.assign(V_START / V_T)
        print(f"[cold start] V={V_START:+.3f}  z=0 steady...")
        if not run_ss(max_steps=100):
            print("FAILED z=0 steady at V_START — aborting")
            return

        z_nominal = [float(sp[4][i]) for i in range(n)]
        achieved_z = 0.0
        print(f"[cold start] V={V_START:+.3f}  z-ramp 0 -> 1...")
        for z_val in np.linspace(0, 1, 21)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n): zc[i].assign(z_nominal[i] * z_val)
            if run_ss(max_steps=60):
                achieved_z = z_val
            else:
                for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                Up.assign(U)
                print(f"  z-ramp failed at z={z_val:.3f}")
                break

        if achieved_z < 1.0 - 1e-3:
            print(f"FAILED z-ramp at V_START (only reached z={achieved_z:.3f}) — aborting")
            return

    cd0 = float(fd.assemble(form_cd))
    pc0 = float(fd.assemble(form_pc))
    print(f"[cold start] V={V_START:+.3f}  cd={cd0:+.6f}  pc={pc0:+.6f}  [OK]")

    # --- Voltage chain walk ---
    results = [{"V": V_START, "cd": cd0, "pc": pc0,
                "U_arrays": tuple(d.data_ro.copy() for d in U.dat)}]

    V_cur = V_START
    step = DV_INIT
    success_streak = 0
    n_attempts = 0
    n_bisections = 0

    t_chain_start = time.time()

    while V_cur > V_TARGET + DV_MIN:
        n_attempts += 1
        V_next = max(V_cur - step, V_TARGET)

        # Checkpoint current U (so we can restore on failure).
        U_ckpt = tuple(d.data_ro.copy() for d in U.dat)

        paf.assign(V_next / V_T)

        t0 = time.time()
        with adj.stop_annotating():
            ok = run_ss(max_steps=60, rel_tol=1e-4)
        dt = time.time() - t0

        if ok:
            cd = float(fd.assemble(form_cd))
            pc = float(fd.assemble(form_pc))
            results.append({
                "V": V_next, "cd": cd, "pc": pc,
                "U_arrays": tuple(d.data_ro.copy() for d in U.dat),
            })
            print(f"  V={V_next:+.4f}  cd={cd:+.6f}  pc={pc:+.6f}  "
                  f"step={step:.4f}  t={dt:.1f}s  [OK]")
            V_cur = V_next
            success_streak += 1
            if success_streak >= MIN_STEP_SUCCESSES_BEFORE_GROW:
                step = min(step * DV_GROW, DV_INIT * 2)
                success_streak = 0
        else:
            # Restore checkpoint; shrink step.
            for src, dst in zip(U_ckpt, U.dat):
                dst.data[:] = src
            Up.assign(U)
            paf.assign(V_cur / V_T)
            success_streak = 0
            step *= 0.5
            n_bisections += 1
            print(f"  V={V_next:+.4f}  FAILED  shrinking step -> {step:.4f}  t={dt:.1f}s")
            if step < DV_MIN:
                print(f"  step below DV_MIN={DV_MIN}; stopping at V={V_cur:+.4f}")
                break

    t_chain = time.time() - t_chain_start
    V_reached = V_cur
    print(f"\n[chain done] deepest V={V_reached:+.4f}V  attempts={n_attempts}  "
          f"bisections={n_bisections}  time={t_chain:.1f}s")

    # Summary + save
    print("\n" + "=" * 70)
    print("I-V CURVE (4sp cathodic chain)")
    print("=" * 70)
    print(f"  {'V_RHE':>8}  {'cd':>12}  {'pc':>12}  {'|PC/CD|':>9}")
    for r in results:
        ratio = f"{abs(r['pc']/r['cd']):.4f}" if abs(r["cd"]) > 1e-12 else "NaN"
        print(f"  {r['V']:+8.4f}  {r['cd']:+12.6f}  {r['pc']:+12.6f}  {ratio:>9}")

    V_arr = np.array([r["V"] for r in results])
    cd_arr = np.array([r["cd"] for r in results])
    pc_arr = np.array([r["pc"] for r in results])
    # Save U arrays too (for potential reuse in inference)
    U_arrays_per_V = [r["U_arrays"] for r in results]

    out_path = os.path.join(out_dir, "iv_cathodic.npz")
    np.savez(
        out_path, V_RHE=V_arr, cd=cd_arr, pc=pc_arr,
        V_start=V_START, V_target=V_TARGET, V_reached=V_reached,
        k0_r1=K0_HAT_R1, k0_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
    )
    # Also pickle the U_arrays for later inference seeding
    import pickle
    with open(os.path.join(out_dir, "U_arrays.pkl"), "wb") as f:
        pickle.dump({
            "V_RHE": V_arr.tolist(),
            "U_arrays_per_V": [[np.asarray(a) for a in tup]
                               for tup in U_arrays_per_V],
        }, f)
    print(f"\nSaved I-V to {out_path}")
    print(f"Saved U arrays to {os.path.join(out_dir, 'U_arrays.pkl')}")


if __name__ == "__main__":
    main()
