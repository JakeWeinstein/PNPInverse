"""How far is the debye_boltzmann IC from the converged SS at each V?

For each V_RHE in the production grid this script:

  1. Runs the C+D orchestrator to obtain the converged steady-state field
     state U_ss at every V (uses the per-point callback to snapshot U
     before the next voltage rebuilds the forms).
  2. Independently rebuilds a fresh context per V, runs
     ``set_initial_conditions`` with the debye_boltzmann + bikerman IC,
     snapshots U_ic, and assembles ``||F(U_ic; z=1)||`` --- the exact
     residual Newton sees at the start of Path B's direct-z=1 attempt
     in ``_solve_cold``.
  3. Compares U_ic against U_ss subfunction-by-subfunction (relative L2,
     absolute L2, peak deviation), plus the per-equation residual norm
     so we can see which block is dragging Newton out of basin.

All settings (logc_muh, debye_boltzmann, steric=bikerman, Stern=0.10
F/m^2, exponent_clip=100) match the production stack run in
``plot_iv_curve_unified.py``.

Run from PNPInverse/::

    ../venv-firedrake/bin/python scripts/diagnose_db_ic_distance.py

Output (under StudyResults/diagnose_db_ic_distance/):

    summary.json   --- per-V records (config + numbers)
    summary.csv    --- flat table for spreadsheet inspection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


V_GRID = np.array([
    -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10,
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
    0.70, 0.80, 0.90, 1.00,
])
NY = 200
E_EQ_R1, E_EQ_R2 = 0.68, 1.78


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formulation", choices=("logc", "logc_muh"), default="logc_muh")
    p.add_argument("--stern-capacitance", type=float, default=0.10,
                   help="Stern Cs in F/m^2 (default 0.10). Pass <=0 to disable Stern.")
    p.add_argument("--out-subdir", default=None,
                   help="StudyResults subdir (default diagnose_db_ic_distance_<formulation>).")
    args, passthrough = p.parse_known_args()
    sys.argv = [sys.argv[0], *passthrough]
    return args


def main():
    args = _parse_args()
    SUBFN_H_LABEL = "mu_H" if args.formulation == "logc_muh" else "u_H"
    SUBFN_LABELS = ["u_O2", "u_H2O2", SUBFN_H_LABEL, "phi"]
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
        build_context,
        build_forms,
        set_initial_conditions,
    )

    out_subdir = (
        args.out_subdir
        if args.out_subdir
        else (
            f"diagnose_db_ic_distance_{args.formulation}"
            + ("" if args.stern_capacitance > 0 else "_nostern")
        )
    )
    OUT = os.path.join(_ROOT, "StudyResults", out_subdir)
    os.makedirs(OUT, exist_ok=True)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=NY, beta=3.0)

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
    counterion = DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=args.formulation,
        log_rate=True,
        boltzmann_counterions=[counterion],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
        initializer="debye_boltzmann",
        stern_capacitance_f_m2=(args.stern_capacitance if args.stern_capacitance > 0 else None),
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    print("=" * 72)
    print("Diagnostic: debye_boltzmann IC distance from converged SS")
    print("=" * 72)
    print(f"  V_RHE grid:  {V_GRID.tolist()}")
    print(f"  mesh Ny:     {NY}")
    print(f"  formulation: {args.formulation}  initializer: debye_boltzmann")
    print(f"  steric:      bikerman  Stern: 0.10 F/m^2  clip: 100")
    print()

    # ------------------------------------------------------------------
    # Phase A: run the orchestrator to get U_ss at every V.
    # ------------------------------------------------------------------
    NV = len(V_GRID)
    ss_snaps: list = [None] * NV
    z_factor_at_ss: list = [float("nan")] * NV
    method_at_ss: list = ["MISSING"] * NV

    def _grab_ss(orig_idx, _phi_eta, ctx):
        U = ctx["U"]
        ss_snaps[orig_idx] = tuple(d.data_ro.copy() for d in U.dat)

    print("Phase A: running C+D orchestrator to obtain converged SS...")
    phi_hat_grid = V_GRID / V_T
    t0 = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp, phi_applied_values=phi_hat_grid, mesh=mesh,
            max_z_steps=20, n_substeps_warm=4, bisect_depth_warm=3,
            per_point_callback=_grab_ss,
        )
    for idx, point in result.points.items():
        z_factor_at_ss[idx] = point.achieved_z_factor
        method_at_ss[idx] = point.method
    n_ok = sum(1 for s in ss_snaps if s is not None)
    print(f"  done in {time.time()-t0:.1f}s  ({n_ok}/{NV} converged)")
    print()

    # ------------------------------------------------------------------
    # Phase B: independently build the IC at every V, snapshot U_ic, and
    # assemble ||F(U_ic; z=1)||.
    # ------------------------------------------------------------------
    print("Phase B: building debye_boltzmann IC + assembling F(U_ic; z=1)...")
    rows = []
    t0 = time.time()
    for i, V in enumerate(V_GRID):
        phi_target = float(V) / V_T
        sp_v = sp.with_phi_applied(phi_target)

        ctx = build_context(sp_v, mesh=mesh)
        ctx = build_forms(ctx, sp_v)
        set_initial_conditions(ctx, sp_v)

        U_ic = tuple(d.data_ro.copy() for d in ctx["U"].dat)

        # Sanity: confirm the fresh ctx is already at z = 1 (no homotopy).
        z_at_build = [float(zc) for zc in ctx["z_consts"]]
        bz_at_build = (
            float(ctx["boltzmann_z_scale"])
            if "boltzmann_z_scale" in ctx
            else None
        )

        # Residual at the IC, with full charge coupling.
        F_res = ctx["F_res"]
        bcs = ctx.get("bcs", None)
        b = fd.assemble(F_res, bcs=bcs)
        F_norms_per_block = [
            float(np.linalg.norm(d.data_ro)) for d in b.dat
        ]
        F_norm_total = float(np.sqrt(sum(n * n for n in F_norms_per_block)))

        ic_fallback = bool(ctx.get("initializer_fallback", False))
        ic_fallback_reason = ctx.get("initializer_fallback_reason", None)
        ic_picard_iters = ctx.get("initializer_picard_iters", None)

        # Rate-consistency check (Codex's verification protocol):
        # mean boundary BV rate from the residual expressions at the IC,
        # compared against Picard's converged R1, R2.  If gamma/Stern
        # consistency holds, the ratio should be ~1.
        picard_state = ctx.get("initializer_picard_state", None) or {}
        bv_rate_exprs = ctx.get("bv_rate_exprs", []) or []
        rate_consistency = []
        if not ic_fallback and bv_rate_exprs and picard_state:
            try:
                electrode_marker = ctx["bv_settings"]["electrode_marker"]
                ds = fd.Measure("ds", domain=ctx["mesh"])
                area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
                for j, R_expr in enumerate(bv_rate_exprs):
                    num = float(fd.assemble(R_expr * ds(electrode_marker)))
                    R_residual_mean = num / max(area, 1e-30)
                    R_picard = picard_state.get(f"R{j+1}", float("nan"))
                    if not (np.isfinite(R_picard) and abs(R_picard) > 1e-30):
                        ratio = float("nan")
                    else:
                        ratio = R_residual_mean / R_picard
                    rate_consistency.append({
                        "j": j,
                        "R_picard": R_picard,
                        "R_residual_mean": R_residual_mean,
                        "ratio": ratio,
                    })
            except Exception as exc:
                rate_consistency = [{"error": repr(exc)}]

        # Field-by-field comparison against U_ss.
        ssn = ss_snaps[i]
        rel_per_block = [float("nan")] * len(SUBFN_LABELS)
        abs_per_block = [float("nan")] * len(SUBFN_LABELS)
        inf_per_block = [float("nan")] * len(SUBFN_LABELS)
        ss_norm_per_block = [float("nan")] * len(SUBFN_LABELS)
        ic_norm_per_block = [float(np.linalg.norm(arr)) for arr in U_ic]
        if ssn is not None:
            for j in range(len(SUBFN_LABELS)):
                ic_arr = U_ic[j]
                ss_arr = ssn[j]
                ss_norm = float(np.linalg.norm(ss_arr))
                diff = ic_arr - ss_arr
                abs_per_block[j] = float(np.linalg.norm(diff))
                inf_per_block[j] = float(np.max(np.abs(diff)))
                rel_per_block[j] = abs_per_block[j] / max(ss_norm, 1e-30)
                ss_norm_per_block[j] = ss_norm

        rows.append({
            "V_RHE": float(V),
            "method": method_at_ss[i],
            "z_factor_ss": float(z_factor_at_ss[i]),
            "ic_fallback": ic_fallback,
            "ic_fallback_reason": ic_fallback_reason,
            "ic_picard_iters": ic_picard_iters,
            "z_at_build": z_at_build,
            "boltzmann_z_at_build": bz_at_build,
            "F_norm_total": F_norm_total,
            "F_norms_per_block": F_norms_per_block,
            "ic_norm_per_block": ic_norm_per_block,
            "ss_norm_per_block": ss_norm_per_block,
            "abs_diff_per_block": abs_per_block,
            "rel_diff_per_block": rel_per_block,
            "inf_diff_per_block": inf_per_block,
            "picard_state": {k: float(v) for k, v in picard_state.items()
                             if isinstance(v, (int, float))},
            "rate_consistency": rate_consistency,
        })

        rate_str = ""
        if rate_consistency and "error" not in rate_consistency[0]:
            ratios = [rc.get("ratio", float("nan")) for rc in rate_consistency]
            rate_str = "  ratio=[" + ", ".join(
                f"{r:+.3f}" if np.isfinite(r) else " nan"
                for r in ratios
            ) + "]"
        print(
            f"  V={V:+.3f}  fb={str(ic_fallback):>5}  "
            f"||F||={F_norm_total:.3e}  per-blk F: ["
            + ", ".join(f"{n:.2e}" for n in F_norms_per_block)
            + "]"
            + rate_str
        )

    print(f"  done in {time.time()-t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Phase C: emit a tidy comparison table.
    # ------------------------------------------------------------------
    print("Phase C: per-V IC vs SS comparison")
    print()
    hdr = (
        f"{'V_RHE':>7} | {'method':>13} | {'fb':>5} | "
        f"{'||F(U_ic)||':>11} | "
        + " | ".join(f"rel_{lab:>5}" for lab in SUBFN_LABELS)
        + " | "
        + " | ".join(f"inf_{lab:>5}" for lab in SUBFN_LABELS)
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['V_RHE']:+7.3f} | {r['method']:>13} | "
            f"{str(r['ic_fallback']):>5} | "
            f"{r['F_norm_total']:>11.3e} | "
            + " | ".join(f"{x:>9.3e}" for x in r["rel_diff_per_block"])
            + " | "
            + " | ".join(f"{x:>9.3e}" for x in r["inf_diff_per_block"])
        )
    print()

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({
            "config": {
                "v_rhe_grid": V_GRID.tolist(),
                "mesh_ny": NY,
                "formulation": args.formulation,
                "initializer": "debye_boltzmann",
                "counterion_mode": "steric (bikerman)",
                "stern_capacitance_f_m2": 0.10,
                "exponent_clip": 100.0,
                "subfunction_labels": SUBFN_LABELS,
            },
            "rows": rows,
        }, f, indent=2)

    csv_path = os.path.join(OUT, "summary.csv")
    with open(csv_path, "w") as f:
        cols = (
            ["V_RHE", "method", "z_factor_ss", "ic_fallback",
             "ic_picard_iters", "F_norm_total"]
            + [f"F_blk_{lab}" for lab in SUBFN_LABELS]
            + [f"rel_{lab}" for lab in SUBFN_LABELS]
            + [f"abs_{lab}" for lab in SUBFN_LABELS]
            + [f"inf_{lab}" for lab in SUBFN_LABELS]
            + [f"ic_norm_{lab}" for lab in SUBFN_LABELS]
            + [f"ss_norm_{lab}" for lab in SUBFN_LABELS]
        )
        f.write(",".join(cols) + "\n")
        for r in rows:
            row = [
                f"{r['V_RHE']:.4f}", r["method"], f"{r['z_factor_ss']:.4f}",
                str(r["ic_fallback"]),
                "" if r["ic_picard_iters"] is None else str(r["ic_picard_iters"]),
                f"{r['F_norm_total']:.6e}",
            ]
            row += [f"{x:.6e}" for x in r["F_norms_per_block"]]
            row += [f"{x:.6e}" for x in r["rel_diff_per_block"]]
            row += [f"{x:.6e}" for x in r["abs_diff_per_block"]]
            row += [f"{x:.6e}" for x in r["inf_diff_per_block"]]
            row += [f"{x:.6e}" for x in r["ic_norm_per_block"]]
            row += [f"{x:.6e}" for x in r["ss_norm_per_block"]]
            f.write(",".join(row) + "\n")

    print(f"Saved: {OUT}/{{summary.json, summary.csv}}")


if __name__ == "__main__":
    main()
