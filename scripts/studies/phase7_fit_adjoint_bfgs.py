"""Phase 7 stage-5b fit, ADJOINT edition: L-BFGS-B with pyadjoint gradients.

Replaces the Nelder-Mead harness per the standing adjoint-first rule.
theta = (log10 f_k0_w2e, log10 f_k0_w4e, alpha_w2e, alpha_w4e), water
routes only, coarse 13-pt grid per iteration.

Gradient architecture (steady-state implicit-function adjoint)
--------------------------------------------------------------
1. OFF-TAPE (``adj.stop_annotating()``): anchor at V_solver=0 (full Kw,
   linear_phi, k0 AdaptiveLadder) -> Stern bump -> robust grid walk via
   ``solve_grid_with_anchor`` capturing a U-snapshot per converged V.
2. ON-TAPE: ONE shared ctx (same mesh/space); per V: restore the walk
   snapshot (off-tape, initial guess only), assign phi (on-tape, not a
   control), set ``dt_const = 1e12`` so the transient term vanishes,
   one annotated Newton solve (converges in ~1-2 its from the warm
   state), assemble the role-resolved pc observable.  The tape then
   holds 13 independent solve blocks whose implicit-function adjoint is
   the EXACT steady-state sensitivity — no taping of the SER walk.
3. J = sum_i w_i (pc_i - t_i)^2 with the v2 target interpolated onto
   the grid V's (weights fixed, off-tape).  Thresholded-zero tail bins
   are scored two-sided with the sigma floor (smooth; the canonical
   one-sided scorer is still used for REPORTING).
4. ``ReducedFunctional(J, [k0_w2e, k0_w4e, a_w2e, a_w4e])`` ->
   ``rf.derivative()`` -> chain rule d/dlog10 k0 = k0 ln(10) dJ/dk0.
5. Outer ``scipy.optimize.minimize(method='L-BFGS-B', jac=True)`` with
   box bounds; per-iteration JSON checkpoints; tape cleared per
   iteration.

FD verification gate (per feedback_adjoint_fd_verification: every FD
point is a FRESH cold anchor+walk): ``--fd-check`` compares the adjoint
gradient at x0 against central differences on chosen components and
exits.  Run this BEFORE trusting a fit.

Usage:
    python -u scripts/studies/phase7_fit_adjoint_bfgs.py --fd-check
    python -u scripts/studies/phase7_fit_adjoint_bfgs.py --maxiter 25
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

X0 = (np.log10(1e-3), np.log10(4e-15), 0.627, 0.325)
BOUNDS = [(-8.0, 1.0), (-22.0, -8.0), (0.05, 0.95), (0.05, 0.95)]
WATER_2E_IDX, WATER_4E_IDX = 1, 3   # reaction indices in the dual preset
LN10 = float(np.log(10.0))
DT_STEADY = 1.0e12
FAILED_SCORE = 1.0e6


def _interp_target_onto(v_deck_grid):
    """Fixed (t_i, w_i, keep_i) at the model grid V's from the v2 bins."""
    from scipy.interpolate import PchipInterpolator
    from calibration.phase7_wls import load_target

    tgt = load_target(
        os.path.join(_ROOT, "data", "mangan_deck_p15_h2o2_current_v2.csv")
    )
    f_j = PchipInterpolator(tgt.v, tgt.j, extrapolate=False)
    f_s = PchipInterpolator(tgt.v, tgt.sigma, extrapolate=False)
    t, w, keep = [], [], []
    for v in v_deck_grid:
        jv, sv = f_j(v), f_s(v)
        if jv is None or not np.isfinite(float(jv)):
            keep.append(False); t.append(0.0); w.append(0.0)
        else:
            keep.append(True)
            t.append(float(jv))
            w.append(1.0 / float(sv) ** 2)
    return t, w, keep


class ForwardProblem:
    """One theta evaluation: off-tape anchor+walk, on-tape J + gradient."""

    def __init__(self, opts_base):
        self.opts_base = opts_base
        import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
        self.dp = dp
        self.v_deck = list(dp.V_RHE_DECK_GRID_COARSE)
        self.v_solver = [round(v - dp.V_OCP_RHE, 6) for v in self.v_deck]
        self.t_i, self.w_i, self.keep_i = _interp_target_onto(self.v_deck)

    def _opts(self, theta):
        lg2, lg4, a2, a4 = [float(x) for x in theta]
        o = SimpleNamespace(**vars(self.opts_base))
        o.k0_water_2e_factor = 10.0 ** lg2
        o.k0_water_4e_factor = 10.0 ** lg4
        o.alpha_water_2e = a2
        o.alpha_water_4e = a4
        return o

    def evaluate(self, theta, *, want_grad=True):
        """Returns (J, grad_theta (or None), info dict)."""
        import firedrake as fd
        import firedrake.adjoint as adj
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.anchor_continuation import (
            PreconvergedAnchor,
            set_stern_capacitance_model,
            solve_anchor_with_continuation,
        )
        from Forward.bv_solver.dispatch import (
            build_context, build_forms, set_initial_conditions,
        )
        from Forward.bv_solver.grid_per_voltage import (
            restore_U, snapshot_U, solve_grid_with_anchor,
        )
        from Forward.bv_solver.observables import _build_bv_observable_form
        from scripts._bv_common import I_SCALE, V_T, SNES_OPTS_CHARGED

        dp = self.dp
        opts = self._opts(theta)
        reactions = dp._build_reactions(opts)
        k2 = float(reactions[WATER_2E_IDX]["k0"])
        k4 = float(reactions[WATER_4E_IDX]["k0"])

        sp_base, k0_targets = dp._make_sp(
            opts, reactions,
            stern_capacitance_f_m2=dp.STERN_BASELINE,
            initializer=dp.ANCHOR_INITIALIZER,
        )
        sp_anchor_cs, _ = dp._make_sp(
            opts, reactions,
            stern_capacitance_f_m2=dp.STERN_ANCHOR,
            initializer=dp.ANCHOR_INITIALIZER,
        )
        sp_anchor = sp_anchor_cs.with_phi_applied(
            float(dp.ANCHOR_V_RHE) / float(V_T))

        mesh = make_graded_rectangle_mesh(
            Nx=dp.MESH_NX, Ny=dp.MESH_NY, beta=dp.MESH_BETA,
            domain_height_hat=float(opts.l_eff_um) * 1e-6 / 1.0e-4,
        )

        # ---- stage 1+2 (off-tape): anchor + Stern bump + grid walk ----
        snaps: dict[int, tuple] = {}

        def _snap(orig_idx, _phi_eta, ctx):
            snaps[orig_idx] = tuple(
                np.asarray(a).copy() for a in snapshot_U(ctx["U"])
            )

        with adj.stop_annotating():
            try:
                ar = solve_anchor_with_continuation(
                    sp_anchor, mesh=mesh, k0_targets=k0_targets,
                    initial_scales=dp.INITIAL_SCALES,
                    max_inserts_per_step=dp.MAX_INSERTS_PER_STEP,
                    ic_at_target=dp.IC_AT_TARGET, kw_eff_ladder=None,
                )
            except Exception as exc:
                return FAILED_SCORE, None, {"error": f"anchor: {exc}"}
            if not ar.converged:
                return FAILED_SCORE, None, {"error": "anchor not converged"}
            for cs in dp._stern_bump_ladder(dp.STERN_BASELINE):
                set_stern_capacitance_model(ar.ctx, float(cs))
                ar.ctx["_last_solver"].solve()
            anchor = PreconvergedAnchor(
                phi_applied_eta=float(dp.ANCHOR_V_RHE) / float(V_T),
                U_snapshot=tuple(
                    np.asarray(a).copy() for a in snapshot_U(ar.ctx["U"])
                ),
                k0_targets=tuple(sorted(
                    (int(j), float(k)) for j, k in k0_targets.items())),
                mesh_dof_count=int(ar.ctx["U"].function_space().dim()),
                ladder_history=tuple(
                    (float(s), str(o)) for s, o in ar.ladder_history),
            )
            grid = solve_grid_with_anchor(
                sp_base, anchor=anchor,
                phi_applied_values=np.array(self.v_solver) / float(V_T),
                mesh=mesh,
                n_substeps_warm=dp.N_SUBSTEPS_WARM,
                bisect_depth_warm=dp.BISECT_DEPTH_WARM,
                per_point_callback=_snap,
            )

            conv = [bool(grid.points[i].converged)
                    for i in range(len(self.v_solver))]
            usable = [i for i, (c, k) in enumerate(zip(conv, self.keep_i))
                      if c and k and i in snaps]
            if len(usable) < 6:
                return FAILED_SCORE, None, {
                    "error": f"only {len(usable)} usable points"}

            # ---- annotated-pass ctx (shared controls), built off-tape ----
            ctx = build_context(sp_base, mesh=mesh)
            ctx = build_forms(ctx, sp_base)
            set_initial_conditions(ctx, sp_base)
            snes = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
                    if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
            prob = fd.NonlinearVariationalProblem(
                ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
            solver = fd.NonlinearVariationalSolver(
                prob, solver_parameters=snes)
            form_pc = _build_bv_observable_form(
                ctx, mode="reaction_sum", reaction_index=None,
                scale=-I_SCALE)
            ctx["dt_const"].assign(DT_STEADY)

        controls_f = [
            ctx["bv_k0_funcs"][WATER_2E_IDX],
            ctx["bv_k0_funcs"][WATER_4E_IDX],
            ctx["bv_alpha_funcs"][WATER_2E_IDX],
            ctx["bv_alpha_funcs"][WATER_4E_IDX],
        ]

        # ---- stage 3 (on-tape): steady re-solves + J ----
        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()
        J = 0.0
        pc_vals = {}
        try:
            for i in usable:
                with adj.stop_annotating():
                    restore_U(snaps[i], ctx["U"], ctx["U_prev"])
                ctx["phi_applied_func"].assign(
                    float(self.v_solver[i]) / float(V_T))
                solver.solve()
                pc_i = fd.assemble(form_pc)
                pc_vals[i] = float(pc_i)
                J = J + self.w_i[i] * (pc_i - self.t_i[i]) ** 2
        except Exception as exc:
            adj.pause_annotation()
            return FAILED_SCORE, None, {"error": f"annotated pass: {exc}"}

        n_used = len(usable)
        J_val = float(J) / n_used
        grad = None
        if want_grad:
            rf = adj.ReducedFunctional(
                J, [adj.Control(c) for c in controls_f])
            gs = rf.derivative()

            def _x(g):
                if hasattr(g, "dat"):
                    return float(g.dat.data_ro[0])
                if hasattr(g, "values"):
                    return float(g.values()[0])
                return float(g)

            g_k2, g_k4, g_a2, g_a4 = (_x(g) for g in gs)
            grad = np.array([
                k2 * g_k2 * LN10, k4 * g_k4 * LN10, g_a2, g_a4,
            ]) / n_used
        adj.pause_annotation()
        tape.clear_tape()

        info = {
            "n_used": n_used,
            "n_converged": sum(conv),
            "pc_model": {self.v_deck[i]: pc_vals[i] for i in usable},
        }
        return J_val, grad, info


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--fd-check", action="store_true")
    parser.add_argument("--fd-components", default="1,3",
                        help="comma indices of theta for the FD check")
    parser.add_argument("--out-name", default="phase7_fit_adjoint")
    args = parser.parse_args()

    out_dir = Path(_ROOT) / "StudyResults" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()

    opts_base = SimpleNamespace(
        routes="water",
        k0_water_2e_factor=1.0, k0_water_4e_factor=1.0,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=None, alpha_water_4e=None,
        l_eff_um=15.4, bulk_h_mol_m3=0.1,
        enable_water_ionization=True, coarse_grid=True,
    )
    fp = ForwardProblem(opts_base)
    x0 = np.array(X0, dtype=float)

    if args.fd_check:
        print("== FD verification of adjoint gradient at x0 ==", flush=True)
        t0 = time.time()
        J0, g, info = fp.evaluate(x0, want_grad=True)
        print(f"J(x0)={J0:.6f}  grad={np.array2string(g, precision=4)}  "
              f"({time.time()-t0:.0f}s, n_used={info.get('n_used')})",
              flush=True)
        steps = {0: 0.02, 1: 0.02, 2: 0.005, 3: 0.005}
        results = {"x0": list(x0), "J0": J0, "grad_adjoint": list(map(float, g))}
        comps = [int(c) for c in args.fd_components.split(",")]
        for k in comps:
            h = steps[k]
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h; xm[k] -= h
            Jp, _, _ = fp.evaluate(xp, want_grad=False)
            Jm, _, _ = fp.evaluate(xm, want_grad=False)
            g_fd = (Jp - Jm) / (2 * h)
            rel = abs(g_fd - g[k]) / max(abs(g_fd), 1e-12)
            print(f"  theta[{k}]: adjoint={g[k]:+.6e}  fd={g_fd:+.6e}  "
                  f"rel_err={rel:.3e}", flush=True)
            results[f"fd_{k}"] = {"h": h, "J_plus": Jp, "J_minus": Jm,
                                  "g_fd": g_fd, "g_adjoint": float(g[k]),
                                  "rel_err": rel}
        with open(out_dir / "fd_check.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {out_dir / 'fd_check.json'}", flush=True)
        return 0

    # ---- L-BFGS-B ----
    from calibration.phase7_wls import load_target, score_iv_json
    iter_count = [0]
    history = []

    def fun(x):
        n = iter_count[0]
        iter_count[0] += 1
        t0 = time.time()
        J, g, info = fp.evaluate(x, want_grad=True)
        wall = time.time() - t0
        if g is None:
            g = np.zeros(4)
            print(f"[it {n:03d}] FAILED {info.get('error')}", flush=True)
        else:
            print(f"[it {n:03d}] J={J:10.4f}  "
                  f"x=({x[0]:+.3f},{x[1]:+.3f},{x[2]:.3f},{x[3]:.3f})  "
                  f"|g|={np.linalg.norm(g):.3e}  {wall:.0f}s", flush=True)
        rec = {"n": n, "x": list(map(float, x)), "J": float(J),
               "grad": list(map(float, g)), "wall_seconds": wall,
               "info": {k: v for k, v in info.items() if k != "pc_model"},
               "pc_model": info.get("pc_model")}
        history.append(rec)
        with open(out_dir / f"iter_{n:03d}.json", "w") as fh:
            json.dump(rec, fh, indent=1)
        return float(J), np.asarray(g, dtype=float)

    from scipy.optimize import minimize
    t0 = time.time()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=BOUNDS,
                   options={"maxiter": args.maxiter, "ftol": 1e-8,
                            "gtol": 1e-7})
    print(f"\nL-BFGS-B: {res.message}  nit={res.nit} nfev={res.nfev}  "
          f"J*={res.fun:.4f}  x*={np.round(res.x, 4)}", flush=True)

    # final: best theta on the fine grid via the driver (full report+score)
    import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
    opts_fine = fp._opts(res.x)
    opts_fine.coarse_grid = False
    report = dp._run(opts_fine)
    summary = {
        "x0": list(X0), "x_star": list(map(float, res.x)),
        "J_star_coarse": float(res.fun), "nit": int(res.nit),
        "nfev": int(res.nfev), "message": str(res.message),
        "wall_seconds": time.time() - t0,
    }
    with open(out_dir / "best_theta_fine_grid.json", "w") as fh:
        json.dump({"theta": list(map(float, res.x)), "report": report},
                  fh, indent=1)
    if report.get("anchor_converged"):
        target = load_target(os.path.join(
            _ROOT, "data", "mangan_deck_p15_h2o2_current_v2.csv"))
        sres = score_iv_json(report, target)
        summary["fine_chi2_per_point"] = float(sres.chi2_per_point)
        summary["fine_hinge"] = float(sres.hinge_penalty)
        summary["fine_validity"] = sres.validity_failures
        summary["fine_n_converged"] = report.get("n_converged")
        print(f"fine-grid: chi2/pt={sres.chi2_per_point:.2f} "
              f"({report.get('n_converged')}/{report.get('n_total')})",
              flush=True)
    with open(out_dir / "fit_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out_dir / 'fit_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
