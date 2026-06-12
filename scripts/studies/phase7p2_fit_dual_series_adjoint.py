"""Phase 7.2 dual-series adjoint fit — K2SO4 pH 6.39 (disk + raw ring).

Session-43-approved plan (~/.claude/plans/phase7p2-k2so4-dual-series-fit.md),
Stage 3/4 harness.  Same steady-state implicit-function adjoint
architecture as ``phase7_fit_adjoint_bfgs.py`` (off-tape anchor+walk,
on-tape per-V steady re-solve with dt -> inf), with:

* TWO taped observables per V:
    form_cd = electron-weighted total current (mode="current_density")
    form_pc = role-resolved H2O2 production current (= escape flux in
              the producing-only primary config; equality is a Stage-3
              test in tests/test_phase7p2_observables.py)
* Taped objective (optimization form; raw chi2 logged separately):
    J = (1/n) sum_i [ w_d_i (cd_i - d_i)^2
                      + w_ring_scale * w_r_i (jr_i - r_i)^2 ]
    with the collection model ON TAPE on the model side:
    jr_i = -pc_i * N * A_d / A_r          (R2#4)
* Convergence discipline (R2#1): a failed evaluation RAISES
  EvalFailure — never returns a penalty (f, g) to L-BFGS-B.  In-eval
  retry: one full re-walk with doubled warm substeps / +2 bisect
  depth.  The outer runner restarts L-BFGS-B from the last valid
  iterate with a fresh Hessian and a bounds box shrunk around it
  (step-halving); > 3 restarts aborts (stop and diagnose).
* Grids: --grid iter -> adaptive 17-pt iteration grid (predeclared:
  dense at onset + ring peak); --grid bins -> the 30 bin centers
  (final polish / scoring / FD / profiles).
* FD gate: --fd-check at theta* (NONSTATIONARY primary point, R3#5)
  in scaled optimizer variables, every component, fresh cold
  anchor+walk per FD point (feedback_adjoint_fd_verification).

Conventions: V axis = Stage-0 PHYSICAL iR axis; OCP shift 1.019 V
(0.47 + measured 0.549 cal); acid routes k0=0; sigma = conservative
single-observation scale (raw chi2 never interpreted absolutely).

Usage:
    python -u scripts/studies/phase7p2_fit_dual_series_adjoint.py --fd-check
    python -u scripts/studies/phase7p2_fit_dual_series_adjoint.py \
        --maxiter 30 [--grid iter] [--polish]
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

from calibration.phase7_wls import (
    A_DISK_CM2, A_RING_CM2, N_COLL_DEFAULT, load_dual_target,
)

# theta* from the slide-15 Cs+/pH4 adjoint fit = X0 here (and the
# FD-gate point — nonstationary for the NEW objective).
X0 = (-3.68327646668887, -13.537374774885187,
      0.5499855248013807, 0.2854405505612447)
BOUNDS = [(-8.0, 1.0), (-22.0, -8.0), (0.05, 0.95), (0.05, 0.95)]
WATER_2E_IDX, WATER_4E_IDX = 1, 3
LN10 = float(np.log(10.0))
DT_STEADY = 1.0e12
MAX_RESTARTS = 3

V_OCP_PH639 = 1.019           # 0.47 + measured 0.549 cal (session 43)
BULK_H_PH639 = 4.07e-4        # mol/m3 (concentration-pH convention)
BINNED_CSV = os.path.join(_ROOT, "data", "k2so4_ph6p39_rrde_binned.csv")

# Adaptive 17-pt iteration grid (predeclared from Stage-0 features):
# dense around the ring peak (~0.42) and onset (0.45..0.56), sparser
# on the plateau (0.13..0.35).
ITER_GRID = tuple(np.round(np.concatenate([
    np.linspace(0.1317, 0.34, 6, endpoint=False),
    np.linspace(0.34, 0.475, 7, endpoint=False),
    np.linspace(0.475, 0.5591, 4),
]), 6).tolist())


class EvalFailure(RuntimeError):
    """Solve failure inside an evaluation; carries theta."""

    def __init__(self, msg, theta=None):
        super().__init__(msg)
        self.theta = None if theta is None else list(map(float, theta))


class DualForwardProblem:
    """One theta evaluation against the dual-series target."""

    def __init__(self, opts_base, *, grid="iter",
                 n_coll=N_COLL_DEFAULT, w_ring_scale=1.0,
                 w_disk_scale=1.0, target_csv=BINNED_CSV):
        self.opts_base = opts_base
        self.w_disk_scale = float(w_disk_scale)
        import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
        self.dp = dp
        self.target = load_dual_target(target_csv)
        self.n_coll = float(n_coll)
        self.w_ring_scale = float(w_ring_scale)
        if grid == "bins":
            self.v_rhe = list(self.target.v)
        elif grid == "iter":
            self.v_rhe = list(ITER_GRID)
        else:
            raise ValueError(f"grid must be iter|bins, got {grid!r}")
        self.grid_mode = grid
        self.v_solver = [round(v - V_OCP_PH639, 6) for v in self.v_rhe]
        # Fixed (t, w) per series at the model V's.  On the bin grid
        # these ARE the bins (exact); on the iteration grid the target
        # is interpolated ONCE here (fixed thereafter).
        from scipy.interpolate import PchipInterpolator
        tv = np.asarray(self.target.v)
        self.d_i = PchipInterpolator(tv, self.target.j_disk)(self.v_rhe)
        self.sd_i = PchipInterpolator(tv, self.target.sigma_disk)(self.v_rhe)
        self.r_i = PchipInterpolator(tv, self.target.j_ring)(self.v_rhe)
        self.sr_i = PchipInterpolator(tv, self.target.sigma_ring)(self.v_rhe)
        self.wd_i = 1.0 / np.asarray(self.sd_i) ** 2
        self.wr_i = 1.0 / np.asarray(self.sr_i) ** 2

    def _opts(self, theta):
        lg2, lg4, a2, a4 = [float(x) for x in theta]
        o = SimpleNamespace(**vars(self.opts_base))
        o.k0_water_2e_factor = 10.0 ** lg2
        o.k0_water_4e_factor = 10.0 ** lg4
        o.alpha_water_2e = a2
        o.alpha_water_4e = a4
        return o

    def evaluate(self, theta, *, want_grad=True):
        """Returns (J, grad, info).  Raises EvalFailure on any solve
        failure (after the in-eval retry) — never a penalty value."""
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

        snaps: dict[int, tuple] = {}

        def _snap(orig_idx, _phi_eta, ctx):
            snaps[orig_idx] = tuple(
                np.asarray(a).copy() for a in snapshot_U(ctx["U"]))

        def _walk(n_sub, bisect):
            snaps.clear()
            return solve_grid_with_anchor(
                sp_base, anchor=anchor,
                phi_applied_values=np.array(self.v_solver) / float(V_T),
                mesh=mesh,
                n_substeps_warm=n_sub, bisect_depth_warm=bisect,
                per_point_callback=_snap,
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
                raise EvalFailure(f"anchor: {exc}", theta)
            if not ar.converged:
                raise EvalFailure("anchor not converged", theta)
            for cs in dp._stern_bump_ladder(dp.STERN_BASELINE):
                set_stern_capacitance_model(ar.ctx, float(cs))
                ar.ctx["_last_solver"].solve()
            anchor = PreconvergedAnchor(
                phi_applied_eta=float(dp.ANCHOR_V_RHE) / float(V_T),
                U_snapshot=tuple(
                    np.asarray(a).copy()
                    for a in snapshot_U(ar.ctx["U"])),
                k0_targets=tuple(sorted(
                    (int(j), float(k)) for j, k in k0_targets.items())),
                mesh_dof_count=int(ar.ctx["U"].function_space().dim()),
                ladder_history=tuple(
                    (float(s), str(o)) for s, o in ar.ladder_history),
            )
            grid = _walk(dp.N_SUBSTEPS_WARM, dp.BISECT_DEPTH_WARM)
            conv = [bool(grid.points[i].converged)
                    for i in range(len(self.v_solver))]
            retried = False
            if not all(conv):
                # in-eval retry ladder (R2#1): one harder re-walk
                retried = True
                grid = _walk(2 * dp.N_SUBSTEPS_WARM,
                             dp.BISECT_DEPTH_WARM + 2)
                conv = [bool(grid.points[i].converged)
                        for i in range(len(self.v_solver))]
            if not all(conv) or any(i not in snaps
                                    for i in range(len(self.v_solver))):
                bad = [self.v_rhe[i] for i, c in enumerate(conv)
                       if not c]
                raise EvalFailure(
                    f"non-converged objective V (post-retry): {bad}",
                    theta)

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
            form_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None,
                scale=-I_SCALE)
            ctx["dt_const"].assign(DT_STEADY)

        controls_f = [
            ctx["bv_k0_funcs"][WATER_2E_IDX],
            ctx["bv_k0_funcs"][WATER_4E_IDX],
            ctx["bv_alpha_funcs"][WATER_2E_IDX],
            ctx["bv_alpha_funcs"][WATER_4E_IDX],
        ]
        ring_map = self.n_coll * A_DISK_CM2 / A_RING_CM2

        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()
        J = 0.0
        cd_vals, pc_vals = {}, {}
        n = len(self.v_solver)
        try:
            for i in range(n):
                with adj.stop_annotating():
                    restore_U(snaps[i], ctx["U"], ctx["U_prev"])
                ctx["phi_applied_func"].assign(
                    float(self.v_solver[i]) / float(V_T))
                solver.solve()
                cd_i = fd.assemble(form_cd)
                pc_i = fd.assemble(form_pc)
                cd_vals[i] = float(cd_i)
                pc_vals[i] = float(pc_i)
                jr_i = -pc_i * ring_map          # collection model on tape
                J = J + (self.w_disk_scale * self.wd_i[i]
                         * (cd_i - self.d_i[i]) ** 2
                         + self.w_ring_scale * self.wr_i[i]
                         * (jr_i - self.r_i[i]) ** 2)
        except EvalFailure:
            adj.pause_annotation()
            raise
        except Exception as exc:
            adj.pause_annotation()
            raise EvalFailure(f"annotated pass: {exc}", theta)

        J_val = float(J) / n
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
            ]) / n
        adj.pause_annotation()
        tape.clear_tape()

        # raw chi2 on the fixed vector (statistics; bins grid only is
        # exact — on the iteration grid this is the surrogate's chi2)
        chi_d = sum(self.wd_i[i] * (cd_vals[i] - self.d_i[i]) ** 2
                    for i in range(n))
        chi_r = sum(self.wr_i[i]
                    * (-pc_vals[i] * ring_map - self.r_i[i]) ** 2
                    for i in range(n))
        info = {
            "n_used": n, "retried_walk": retried,
            "grid_mode": self.grid_mode,
            "chi2_raw": float(chi_d + chi_r),
            "score_disk": float(chi_d) / n,
            "score_ring": float(chi_r) / n,
            "cd_model": {self.v_rhe[i]: cd_vals[i] for i in range(n)},
            "pc_model": {self.v_rhe[i]: pc_vals[i] for i in range(n)},
        }
        return J_val, grad, info


def run_lbfgsb(fp, x0, out_dir, maxiter, fixed=None):
    """L-BFGS-B with the raise-and-restart failed-eval policy.

    fixed: optional {component_index: value} — those components are
    pinned (profile protocol R1#9); scipy optimizes the free subset.
    """
    from scipy.optimize import minimize

    fixed = dict(fixed or {})
    free = [k for k in range(4) if k not in fixed]
    if not free:
        raise ValueError("all components fixed")

    def full_x(xf):
        x = np.empty(4)
        for j, k in enumerate(free):
            x[k] = xf[j]
        for k, v in fixed.items():
            x[k] = v
        return x

    state = {"k": 0, "last_valid_x": None, "last_valid_J": None}
    log_path = Path(out_dir) / "eval_log.jsonl"

    def fun(xf):
        x = full_x(xf)
        J, g, info = fp.evaluate(x, want_grad=True)   # may raise
        state["k"] += 1
        state["last_valid_x"] = np.array(xf, dtype=float)
        state["last_valid_J"] = J
        rec = {"eval": state["k"], "theta": list(map(float, x)),
               "J": J, "grad": list(map(float, g)),
               "fixed": {str(k): float(v) for k, v in fixed.items()},
               **{k: v for k, v in info.items()
                  if k in ("chi2_raw", "score_disk", "score_ring",
                           "retried_walk")}}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  eval {state['k']:3d}  J={J:.4f}  "
              f"chi2_raw={info['chi2_raw']:.1f}  "
              f"(disk {info['score_disk']:.2f} / ring "
              f"{info['score_ring']:.2f})", flush=True)
        return J, np.asarray(g, dtype=float)[free]

    bounds = [list(BOUNDS[k]) for k in free]
    x_start = np.array([np.array(x0, dtype=float)[k] for k in free])
    restarts = 0
    while True:
        try:
            res = minimize(fun, x_start, jac=True, method="L-BFGS-B",
                           bounds=[tuple(b) for b in bounds],
                           options={"maxiter": maxiter,
                                    "ftol": 1e-10, "gtol": 1e-8})
            res.x = full_x(res.x)
            return res, restarts
        except EvalFailure as exc:
            restarts += 1
            print(f"!! EvalFailure: {exc} (restart {restarts}/"
                  f"{MAX_RESTARTS})", flush=True)
            if state["last_valid_x"] is None or restarts > MAX_RESTARTS:
                raise
            x_fail_full = np.array(exc.theta if exc.theta is not None
                                   else full_x(x_start), dtype=float)
            x_start = state["last_valid_x"]
            for j, k in enumerate(free):
                half = max(abs(x_fail_full[k] - x_start[j]) / 2.0,
                           0.05 * (BOUNDS[k][1] - BOUNDS[k][0]) / 10.0)
                bounds[j][0] = max(BOUNDS[k][0], x_start[j] - half)
                bounds[j][1] = min(BOUNDS[k][1], x_start[j] + half)


def main() -> int:
    global V_OCP_PH639
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--grid", choices=("iter", "bins"),
                        default="iter")
    parser.add_argument("--polish", action="store_true",
                        help="after the iter-grid fit, polish on the "
                             "bin-center objective (R2#2)")
    parser.add_argument("--fd-check", action="store_true")
    parser.add_argument("--predict", action="store_true",
                        help="Stage 2: single gradient-free "
                             "evaluation at x0 on the chosen grid; "
                             "dump prediction JSON and exit (no "
                             "tuning).")
    parser.add_argument("--fd-components", default="0,1,2,3",
                        help="comma list; 'k' or 'k=h' entries")
    parser.add_argument("--x0", default=None,
                        help="comma 4-tuple start (default theta*)")
    parser.add_argument("--n-coll", type=float, default=N_COLL_DEFAULT)
    parser.add_argument("--w-ring-scale", type=float, default=1.0)
    parser.add_argument("--w-disk-scale", type=float, default=1.0,
                        help="0 -> ring-only (pc-only) objective for "
                             "the profile comparison (R1#9 iii)")
    parser.add_argument("--fix", default=None,
                        help="profile protocol: 'k=v[,k=v]' pins "
                             "theta components; the rest reoptimize")
    parser.add_argument("--bulk-h-mol-m3", type=float,
                        default=BULK_H_PH639)
    parser.add_argument("--l-eff-um", type=float, default=15.4)
    parser.add_argument("--v-ocp-rhe", type=float, default=V_OCP_PH639)
    parser.add_argument("--out-name", default="phase7p2_fit_dual")
    args = parser.parse_args()

    V_OCP_PH639 = float(args.v_ocp_rhe)

    out_dir = Path(_ROOT) / "StudyResults" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()

    opts_base = SimpleNamespace(
        routes="water",
        k0_water_2e_factor=1.0, k0_water_4e_factor=1.0,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=None, alpha_water_4e=None,
        l_eff_um=float(args.l_eff_um),
        bulk_h_mol_m3=float(args.bulk_h_mol_m3),
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_ocp_rhe=float(args.v_ocp_rhe),
        v_grid_lo=None, v_grid_hi=None,
    )
    fixed = {}
    if args.fix:
        for tok in args.fix.split(","):
            kk, vv = tok.split("=")
            fixed[int(kk)] = float(vv)
    x0 = (np.array([float(t) for t in args.x0.split(",")])
          if args.x0 else np.array(X0, dtype=float))

    if args.fd_check:
        fp = DualForwardProblem(opts_base, grid="bins",
                                n_coll=args.n_coll,
                                w_ring_scale=args.w_ring_scale,
                                w_disk_scale=args.w_disk_scale)
        print("== FD verification (dual objective, bins grid, "
              "theta*) ==", flush=True)
        t0 = time.time()
        J0, g, info = fp.evaluate(x0, want_grad=True)
        print(f"J(x0)={J0:.6f} chi2_raw={info['chi2_raw']:.1f} "
              f"grad={np.array2string(g, precision=4)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        results = {"x0": list(map(float, x0)), "J0": J0,
                   "grad_adjoint": list(map(float, g)),
                   "info": {k: v for k, v in info.items()
                            if not k.endswith("_model")},
                   "fd": []}
        default_h = {0: 0.02, 1: 0.02, 2: 0.005, 3: 0.005}
        comps = []
        for tok in args.fd_components.split(","):
            if "=" in tok:
                kk, hh = tok.split("=")
                comps.append((int(kk), float(hh)))
            else:
                comps.append((int(tok), None))
        for k, h_over in comps:
            h = h_over if h_over is not None else default_h[k]
            xp, xm = x0.copy(), x0.copy()
            xp[k] += h
            xm[k] -= h
            Jp, _, _ = fp.evaluate(xp, want_grad=False)
            Jm, _, _ = fp.evaluate(xm, want_grad=False)
            g_fd = (Jp - Jm) / (2 * h)
            tau = max(0.05 * abs(g_fd),
                      1e-3 * float(np.max(np.abs(g))),
                      1e-6 * (1.0 + abs(J0)))
            rec = {"component": k, "h": h, "g_fd": g_fd,
                   "g_adj": float(g[k]),
                   "abs_err": abs(float(g[k]) - g_fd),
                   "tol": tau,
                   "pass": bool(abs(float(g[k]) - g_fd) <= tau)}
            results["fd"].append(rec)
            print(f"  comp {k} h={h:g}: g_fd={g_fd:+.5e} "
                  f"g_adj={float(g[k]):+.5e} "
                  f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
        with open(out_dir / "fd_check.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out_dir / 'fd_check.json'}", flush=True)
        return 0

    if args.predict:
        fp = DualForwardProblem(opts_base, grid="bins",
                                n_coll=args.n_coll,
                                w_ring_scale=args.w_ring_scale,
                                w_disk_scale=args.w_disk_scale)
        print("== Stage 2 prediction at x0 (NO tuning) ==", flush=True)
        t0 = time.time()
        J0, _, info = fp.evaluate(x0, want_grad=False)
        out = {"theta": list(map(float, x0)), "J": float(J0),
               "v_ocp_rhe": float(args.v_ocp_rhe),
               "l_eff_um": float(args.l_eff_um),
               "n_coll": float(args.n_coll),
               "wall_s": time.time() - t0, **info}
        with open(out_dir / "prediction.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"J={J0:.4f} chi2_raw={info['chi2_raw']:.1f} "
              f"(disk {info['score_disk']:.2f} / ring "
              f"{info['score_ring']:.2f})  wrote "
              f"{out_dir / 'prediction.json'}", flush=True)
        return 0

    fp = DualForwardProblem(opts_base, grid=args.grid,
                            n_coll=args.n_coll,
                            w_ring_scale=args.w_ring_scale,
                            w_disk_scale=args.w_disk_scale)
    print(f"== L-BFGS-B dual-series fit (grid={args.grid}) ==",
          flush=True)
    res, restarts = run_lbfgsb(fp, x0, out_dir, args.maxiter,
                               fixed=fixed)
    summary = {"stage": args.grid, "x_star": list(map(float, res.x)),
               "J_star": float(res.fun), "nit": int(res.nit),
               "restarts": restarts, "message": str(res.message),
               "fixed": {str(k): v for k, v in fixed.items()},
               "w_disk_scale": float(args.w_disk_scale),
               "w_ring_scale": float(args.w_ring_scale),
               "l_eff_um": float(args.l_eff_um),
               "v_ocp_rhe": float(args.v_ocp_rhe),
               "n_coll": float(args.n_coll),
               "bulk_h_mol_m3": float(args.bulk_h_mol_m3)}
    if args.polish and args.grid == "iter":
        print("== bin-center polish (R2#2) ==", flush=True)
        fp2 = DualForwardProblem(opts_base, grid="bins",
                                 n_coll=args.n_coll,
                                 w_ring_scale=args.w_ring_scale,
                                 w_disk_scale=args.w_disk_scale)
        res2, restarts2 = run_lbfgsb(fp2, res.x, out_dir,
                                     args.maxiter, fixed=fixed)
        summary["polish"] = {
            "x_star": list(map(float, res2.x)),
            "J_star": float(res2.fun), "nit": int(res2.nit),
            "restarts": restarts2, "message": str(res2.message)}
        # grid-convergence check (R1#14)
        J_iter_at_opt, _, _ = fp.evaluate(res2.x, want_grad=False)
        summary["grid_convergence"] = {
            "J_iter_at_final": float(J_iter_at_opt),
            "J_bins_at_final": float(res2.fun)}
    with open(out_dir / "fit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
