"""How much Newton work does each IC need to refine to steady state?

For each voltage in the existing converging window
``V_RHE in [-0.5, +0.6] V``, build a fresh context with each
initializer (``linear_phi`` and ``debye_boltzmann``), force charges to
full coupling (z=1, boltzmann_z_scale=1), and time-step toward steady
state without any z-ramp.  Record per-step Newton iteration counts and
plateau-detector progress so we can see how much "refinement" each IC
actually needs.

Metrics per (V, initializer):

* ``picard_iters`` -- IC construction cost (debye_boltzmann only;
  None for linear_phi).
* ``cd_IC``, ``pc_IC`` -- observables assembled from the IC field
  before any Newton step.  Compared against ``cd_truth`` from the
  production orchestrator to get the IC's static observable error.
* ``first_step_snes_iters`` -- Newton iterations Newton needs on the
  *first* SS time-step starting from the IC.  This is the canonical
  "how close to SS" metric -- 1 means the IC is already a Newton-fixed
  point at this dt; high values mean Newton has work to do.
* ``total_ss_steps`` -- transient time-steps until the CD plateau
  detector fires (or max-steps cap).
* ``total_snes_iters`` -- sum of Newton iters across all time-steps.
* ``converged_z1`` -- bool: did direct z=1 SS work from this IC?
  False at high V for ``linear_phi`` (Newton can't bridge the basin)
  is itself a useful signal.
* ``cd_SS``, ``pc_SS`` -- final converged observable from the direct
  z=1 path (matches truth when ``converged_z1=True``).

Truth reference: a single-pass production orchestrator
(linear_phi + standard z=0 -> z-ramp -> z=1 SS) at every V.  We use
its CD/PC as the canonical SS observable.

Outputs:

    StudyResults/ic_refinement_study/data.json     per-V/per-IC records
    StudyResults/ic_refinement_study/refinement.png  4-panel plot

Run from PNPInverse/ with ../venv-firedrake/bin/activate active::

    python scripts/studies/ic_refinement_study.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


V_GRID = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.10,
    0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
])
MESH_NY = 200
EXPONENT_CLIP = 100.0
MAX_SS_STEPS = 200
SS_REL_TOL = 1e-4
SS_ABS_TOL = 1e-8
SS_CONSEC = 4
DT_INIT = 0.25
DT_GROWTH_CAP = 4.0
DT_MAX_RATIO = 20.0
INITIALIZERS = ("linear_phi", "debye_boltzmann")
OUT_SUBDIR = "ic_refinement_study"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh-ny", type=int, default=MESH_NY,
        help=f"Graded mesh Ny (default {MESH_NY}).",
    )
    p.add_argument(
        "--clip", type=float, default=EXPONENT_CLIP,
        help=f"BV exponent_clip (default {EXPONENT_CLIP}).",
    )
    return p.parse_args()


def _make_sp(initializer: str, *, V_RHE: float, exponent_clip: float):
    from scripts._bv_common import (
        V_T, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
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
    })

    sp = make_bv_solver_params(
        eta_hat=V_RHE / V_T, dt=DT_INIT, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc", log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer=initializer,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(exponent_clip)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _measure_ic_refinement(
    V_RHE: float,
    initializer: str,
    *,
    mesh,
    exponent_clip: float,
) -> dict[str, Any]:
    """Build ctx with the given IC, force z=1, run instrumented SS."""
    import firedrake as fd
    import firedrake.adjoint as adj
    from scripts._bv_common import I_SCALE, V_T
    from Forward.bv_solver import build_context, build_forms, set_initial_conditions
    from Forward.bv_solver.observables import _build_bv_observable_form

    sp = _make_sp(initializer, V_RHE=V_RHE, exponent_clip=exponent_clip)
    n_s, _, _, _, z_v, _, _, _, _, _, params = sp

    z_nominal = [float(v) for v in (
        [z_v] * n_s if np.isscalar(z_v) else list(z_v)
    )][:n_s]

    with adj.stop_annotating():
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        set_initial_conditions(ctx, sp)

        picard = ctx.get("initializer_picard_iters")
        fallback = bool(ctx.get("initializer_fallback", False))
        fallback_reason = ctx.get("initializer_fallback_reason", "")

        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        of_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )

        # IC-only observables (no SNES yet).  These reflect what BV would
        # report if the IC field were the SS solution.
        cd_ic = float(fd.assemble(of_cd))
        pc_ic = float(fd.assemble(of_pc))

        problem = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"],
        )
        _NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
        _items = params.items() if isinstance(params, dict) else []
        solve_opts = {k: v for k, v in _items if k not in _NON_PETSC_KEYS}
        solve_opts.setdefault("snes_error_if_not_converged", True)
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=solve_opts,
        )

        # Force full coupling.
        for i in range(n_s):
            ctx["z_consts"][i].assign(z_nominal[i])
        boltz = ctx.get("boltzmann_z_scale")
        if boltz is not None:
            boltz.assign(1.0)
        ctx["phi_applied_func"].assign(float(V_RHE / V_T))

        U = ctx["U"]
        U_prev = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_max = DT_INIT * DT_MAX_RATIO

        history: list[dict[str, Any]] = []
        dt_val = DT_INIT
        dt_const.assign(dt_val)
        prev_cd = None
        prev_delta = None
        steady_count = 0
        converged_z1 = False
        for k in range(1, MAX_SS_STEPS + 1):
            try:
                solver.solve()
            except Exception as exc:
                history.append({
                    "step": k, "snes_iters": None, "cd": None,
                    "dt": float(dt_val),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                break
            snes_step = int(solver.snes.getIterationNumber())
            U_prev.assign(U)
            cd = float(fd.assemble(of_cd))
            pc = float(fd.assemble(of_pc))
            history.append({
                "step": k, "snes_iters": snes_step,
                "dt": float(dt_val), "cd": cd, "pc": pc,
            })

            if prev_cd is not None:
                delta = abs(cd - prev_cd)
                sv = max(abs(cd), abs(prev_cd), SS_ABS_TOL)
                is_steady = (delta / sv <= SS_REL_TOL) or (delta <= SS_ABS_TOL)
                steady_count = steady_count + 1 if is_steady else 0
                if prev_delta is not None and delta > 0:
                    ratio = prev_delta / delta
                    if ratio > 1.0:
                        grow = min(ratio, DT_GROWTH_CAP)
                        dt_val = min(dt_val * grow, dt_max)
                    else:
                        dt_val = max(dt_val * 0.5, DT_INIT)
                    dt_const.assign(dt_val)
                prev_delta = delta
            prev_cd = cd
            if steady_count >= SS_CONSEC:
                converged_z1 = True
                break

        cd_ss = float(fd.assemble(of_cd)) if converged_z1 else (
            history[-1].get("cd") if history else None
        )
        pc_ss = float(fd.assemble(of_pc)) if converged_z1 else (
            history[-1].get("pc") if history else None
        )

    snes_iters_total = sum(
        h["snes_iters"] for h in history if h.get("snes_iters") is not None
    )
    first_step = history[0] if history else {}

    return {
        "V_RHE": float(V_RHE),
        "initializer": initializer,
        "picard_iters": picard,
        "initializer_fallback": fallback,
        "initializer_fallback_reason": fallback_reason,
        "cd_IC": cd_ic,
        "pc_IC": pc_ic,
        "first_step_snes_iters": first_step.get("snes_iters"),
        "first_step_cd": first_step.get("cd"),
        "total_ss_steps": len(history),
        "total_snes_iters": int(snes_iters_total),
        "converged_z1": bool(converged_z1),
        "cd_SS": cd_ss,
        "pc_SS": pc_ss,
        "history": history,
    }


def _truth_pass(*, mesh, exponent_clip: float) -> dict[int, dict[str, float]]:
    """Production orchestrator (linear_phi + z-ramp) -> canonical CD/PC."""
    import firedrake as fd
    import firedrake.adjoint as adj
    from scripts._bv_common import I_SCALE, V_T
    from Forward.bv_solver import (
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    NV = len(V_GRID)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    sp = _make_sp("linear_phi", V_RHE=0.0, exponent_clip=exponent_clip)
    phi_hat_grid = V_GRID / V_T
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
    out: dict[int, dict[str, float]] = {}
    for i, point in result.points.items():
        out[i] = {
            "V_RHE": float(V_GRID[i]),
            "cd_truth": float(cd[i]) if np.isfinite(cd[i]) else None,
            "pc_truth": float(pc[i]) if np.isfinite(pc[i]) else None,
            "converged": bool(point.converged),
            "method": str(point.method),
        }
    return out


def _make_plot(records: list[dict], truth: dict, png_path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    by_init: dict[str, list[dict]] = {init: [] for init in INITIALIZERS}
    for r in records:
        by_init[r["initializer"]].append(r)
    for init in by_init:
        by_init[init].sort(key=lambda r: r["V_RHE"])

    colors = {"linear_phi": "#377eb8", "debye_boltzmann": "#e41a1c"}
    markers = {"linear_phi": "o", "debye_boltzmann": "s"}

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax_first = fig.add_subplot(gs[0, 0])
    ax_total = fig.add_subplot(gs[0, 1])
    ax_steps = fig.add_subplot(gs[1, 0])
    ax_obs = fig.add_subplot(gs[1, 1])

    def _xy(records, key):
        xs = [r["V_RHE"] for r in records]
        ys = [r.get(key) for r in records]
        return xs, ys

    for init in INITIALIZERS:
        xs, ys_first = _xy(by_init[init], "first_step_snes_iters")
        ys_first_plot = [np.nan if v is None else float(v) for v in ys_first]
        ax_first.plot(xs, ys_first_plot, marker=markers[init], color=colors[init],
                      linestyle="-", label=init)

        ys_total = [r["total_snes_iters"] for r in by_init[init]]
        ax_total.plot(xs, ys_total, marker=markers[init], color=colors[init],
                      linestyle="-", label=init)

        ys_steps = [r["total_ss_steps"] for r in by_init[init]]
        ax_steps.plot(xs, ys_steps, marker=markers[init], color=colors[init],
                      linestyle="-", label=init)

        # IC observable error vs truth
        ic_err = []
        for r in by_init[init]:
            cd_ic = r["cd_IC"]
            t = truth.get(int(np.argmin(np.abs(V_GRID - r["V_RHE"]))), {})
            cd_truth = t.get("cd_truth")
            if cd_truth is None or cd_truth == 0:
                ic_err.append(np.nan)
            else:
                ic_err.append(abs(cd_ic - cd_truth) / max(abs(cd_truth), 1e-12))
        ax_obs.plot(xs, ic_err, marker=markers[init], color=colors[init],
                    linestyle="-", label=init)

        # Mark non-converged direct-z=1 attempts with an X
        for r in by_init[init]:
            if not r["converged_z1"]:
                ax_first.plot(r["V_RHE"], (r["first_step_snes_iters"] or 0),
                              marker="x", markersize=12, color=colors[init],
                              markeredgewidth=2.5)
                ax_total.plot(r["V_RHE"], r["total_snes_iters"],
                              marker="x", markersize=12, color=colors[init],
                              markeredgewidth=2.5)
                ax_steps.plot(r["V_RHE"], r["total_ss_steps"],
                              marker="x", markersize=12, color=colors[init],
                              markeredgewidth=2.5)

    ax_first.set_title("Newton iters on first SS time-step (lower = closer IC)")
    ax_first.set_xlabel("V vs RHE (V)")
    ax_first.set_ylabel("SNES iterations")
    ax_first.grid(True, alpha=0.3)
    ax_first.legend(fontsize=9)

    ax_total.set_title("Total Newton iters across all SS time-steps")
    ax_total.set_xlabel("V vs RHE (V)")
    ax_total.set_ylabel("Sum of SNES iterations")
    ax_total.set_yscale("log")
    ax_total.grid(True, alpha=0.3, which="both")
    ax_total.legend(fontsize=9)

    ax_steps.set_title("SS time-steps to plateau (CD-delta < 1e-4 for 4 in a row)")
    ax_steps.set_xlabel("V vs RHE (V)")
    ax_steps.set_ylabel("SS time-steps")
    ax_steps.grid(True, alpha=0.3)
    ax_steps.legend(fontsize=9)

    ax_obs.set_title("|CD_IC - CD_truth| / |CD_truth|  (static observable error)")
    ax_obs.set_xlabel("V vs RHE (V)")
    ax_obs.set_ylabel("relative CD error at IC")
    ax_obs.set_yscale("log")
    ax_obs.grid(True, alpha=0.3, which="both")
    ax_obs.legend(fontsize=9)

    fig.suptitle(
        f"IC refinement study: how much Newton work each IC needs to reach SS\n"
        f"(Ny={MESH_NY}, exponent_clip={EXPONENT_CLIP:.0f}; X = direct z=1 SS did not converge)",
        y=0.995,
    )

    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()
    return None


def main() -> None:
    cli = _parse_args()

    OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("  IC refinement study: linear_phi vs debye_boltzmann")
    print("=" * 78)
    print(f"  V_GRID         = {V_GRID.tolist()}")
    print(f"  initializers   = {list(INITIALIZERS)}")
    print(f"  exponent_clip  = {cli.clip}")
    print(f"  mesh_Ny        = {cli.mesh_ny}")
    print(f"  output         = {OUT_DIR}")
    print()

    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    from Forward.bv_solver import make_graded_rectangle_mesh

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(cli.mesh_ny), beta=3.0)

    print("--- truth pass: production orchestrator (linear_phi + z-ramp) ---")
    t0 = time.time()
    truth = _truth_pass(mesh=mesh, exponent_clip=cli.clip)
    print(f"  truth pass: {time.time() - t0:.1f}s; "
          f"{sum(1 for v in truth.values() if v.get('cd_truth') is not None)}/"
          f"{len(V_GRID)} converged")
    print()

    records: list[dict] = []
    for init in INITIALIZERS:
        print(f"--- pass: initializer={init} ---")
        for V_RHE in V_GRID:
            t0 = time.time()
            r = _measure_ic_refinement(
                float(V_RHE), init, mesh=mesh, exponent_clip=cli.clip,
            )
            r["wall_seconds"] = float(time.time() - t0)
            records.append(r)
            cd_ic_s = (
                f"{r['cd_IC']:+.3e}" if r["cd_IC"] is not None else "(none)"
            )
            print(f"  V={V_RHE:+.3f}  ok_z1={r['converged_z1']}  "
                  f"first_snes={r['first_step_snes_iters']}  "
                  f"total_snes={r['total_snes_iters']}  "
                  f"ss_steps={r['total_ss_steps']}  "
                  f"picard={r['picard_iters']}  "
                  f"cd_IC={cd_ic_s}  "
                  f"({r['wall_seconds']:.1f}s)")
        print()

    data_path = os.path.join(OUT_DIR, "data.json")
    with open(data_path, "w") as f:
        json.dump({
            "V_GRID": V_GRID.tolist(),
            "exponent_clip": float(cli.clip),
            "mesh_Ny": int(cli.mesh_ny),
            "max_ss_steps": int(MAX_SS_STEPS),
            "ss_rel_tol": float(SS_REL_TOL),
            "ss_consec": int(SS_CONSEC),
            "truth": truth,
            "records": records,
        }, f, indent=2)
    print(f"  data.json -> {data_path}")

    png_path = os.path.join(OUT_DIR, "refinement.png")
    err = _make_plot(records, truth, png_path)
    if err is None:
        print(f"  refinement.png -> {png_path}")
    else:
        print(f"  plot skipped: {err}")

    # Quick textual summary
    print()
    print("=" * 78)
    print("  Summary")
    print("=" * 78)
    print(f"  {'V':>6s}  {'init':>16s}  {'first':>5s}  {'tot_snes':>9s}  "
          f"{'steps':>5s}  {'picard':>6s}  {'ok_z1':>6s}")
    for r in sorted(records, key=lambda r: (r["V_RHE"], r["initializer"])):
        ok = "Y" if r["converged_z1"] else "N"
        print(f"  {r['V_RHE']:+.3f}  {r['initializer']:>16s}  "
              f"{str(r['first_step_snes_iters']):>5s}  "
              f"{r['total_snes_iters']:>9d}  "
              f"{r['total_ss_steps']:>5d}  "
              f"{str(r['picard_iters']):>6s}  {ok:>6s}")


if __name__ == "__main__":
    main()
