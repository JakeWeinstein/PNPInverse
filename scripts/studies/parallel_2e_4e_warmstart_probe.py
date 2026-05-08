"""Escape-hatch probe: warm-start the parallel R_2e/R_4e residual from
a converged sequential state at V_RHE = 0.0 V.

Per ``docs/ruggiero_realignment_plan.md`` § "M3a.3 escape hatch" and the
M3a.3 plan handoff: tests whether the parallel 2e/4e residual can
converge from a physically-reasonable initial guess (the converged
sequential 3sp+Bikerman+muh state at V=0) WITHOUT first generalizing
the matched-asymptotic Picard.

Step A
    Build the production sequential 3sp+Bikerman+muh ctx at V=0.0 V
    with the ``debye_boltzmann`` IC (the known-good Run-C reference).
    Cold-solve via the C+D orchestrator restricted to a single-element
    voltage grid; capture ``ctx["U"]`` snapshot in a per-point callback.

Step B
    Build a FRESH parallel-residual ctx at the same V (same mesh, same
    species config; only ``bv_reactions = PARALLEL_2E_4E_REACTIONS``
    differs).  Skip ``set_initial_conditions`` -- the dispatcher would
    otherwise hit the topology gate, fall back to ``linear_phi`` IC,
    and erase Step A's state.  Manually copy ``U`` from the Step A
    snapshot, single-shot SNES.

Step B converges
    -> The M3a.2 diagnostic becomes runnable on a tight V grid via
       custom warm-walking from V=0; the full Picard generalization
       (§3 of M3a.3) is useful but no longer urgent.

Step B diverges
    -> Even a physically-reasonable IC isn't enough; the matched-
       asymptotic Picard rewrite is on the critical path.  Either way,
       the result informs M3a.3 scope.

Output: ``StudyResults/parallel_2e_4e_warmstart_probe/probe.json``
plus a console summary.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

V_PROBE = 0.0          # V vs RHE — hard production cold anchor
MESH_NY = 200
# Historical convergence regime (matches peroxide_window_3sp_bikerman_muh.py).
# clip=100 cold-fails more often per CLAUDE.md hard rule 2; the analytical
# IC fallback path in grid_per_voltage.py hardcodes the LOGC linear-phi IC
# and writes wrong-backend state into a muh ctx, so the analytical path
# MUST land directly at z=1 -- which clip=50 reliably allows at V=0.
# This is an IC-quality probe; the clip=50 PC-fidelity caveat (R_2 unclips
# only at V > +0.495 V) is irrelevant at V_PROBE = 0.0.
EXPONENT_CLIP = 50.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER_SEQ = "debye_boltzmann"
FORMULATION = "logc_muh"
OUT_SUBDIR = "parallel_2e_4e_warmstart_probe"


def _build_solver_options(eta_clip: float, u_clamp: float) -> dict[str, Any]:
    """Mirror peroxide_window_3sp_bikerman_muh.py SNES options."""
    from scripts._bv_common import SNES_OPTS_CHARGED

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update(
        {
            "snes_max_it": 400,
            "snes_atol": 1e-7,
            "snes_rtol": 1e-10,
            "snes_stol": 1e-12,
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "snes_error_if_not_converged": True,
        }
    )
    return snes_opts


def _build_sequential_params(*, snes_opts: dict[str, Any]):
    from scripts._bv_common import (
        K0_HAT_R1,
        K0_HAT_R2,
        ALPHA_R1,
        ALPHA_R2,
        E_EQ_R1_V,
        E_EQ_R2_V,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        THREE_SPECIES_LOGC_BOLTZMANN,
        make_bv_solver_params,
    )

    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        stern_capacitance_f_m2=0.10,
        k0_hat_r1=K0_HAT_R1,
        k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,
        alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1_V,
        E_eq_r2=E_EQ_R2_V,
        initializer=INITIALIZER_SEQ,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _build_parallel_params(*, snes_opts: dict[str, Any]):
    from scripts._bv_common import (
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        THREE_SPECIES_LOGC_BOLTZMANN,
        PARALLEL_2E_4E_REACTIONS,
        make_bv_solver_params,
    )

    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        stern_capacitance_f_m2=0.10,
        # The parallel preset overrides k0/alpha/E_eq via bv_reactions;
        # the legacy keyword bundle is ignored when bv_reactions != None.
        bv_reactions=PARALLEL_2E_4E_REACTIONS,
        # The dispatcher will route through the topology-gated Picard and
        # fall back to linear_phi -- which would clobber our warm copy.
        # Set linear_phi here as a no-op default; we skip set_initial_conditions
        # entirely below.
        initializer="linear_phi",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _step_a_sequential_cold(mesh, *, snes_opts: dict[str, Any]):
    """Cold-solve sequential at V=0.0 V; return (U_data_snapshot, diagnostics)."""
    from scripts._bv_common import V_T, I_SCALE
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback
    from Forward.bv_solver.observables import _build_bv_observable_form

    sp = _build_sequential_params(snes_opts=snes_opts)
    phi_target = float(V_PROBE) / V_T

    captured: dict[str, Any] = {}

    def _grab(orig_idx, _phi_eta, ctx):
        # Snapshot U + a few sanity observables.  ``U`` is on a
        # MixedFunctionSpace, so ``dat`` is a tuple of per-subspace
        # DataSets; copy each (matches Forward.bv_solver.grid_per_voltage
        # ``_snapshot_U``).
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE
        )
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE
        )
        captured["cd_mA_cm2"] = float(fd.assemble(f_cd))
        captured["pc_mA_cm2"] = float(fd.assemble(f_pc))
        captured["U_snap"] = tuple(d.data_ro.copy() for d in ctx["U"].dat)
        captured["dof_count"] = int(ctx["U"].function_space().dim())

    t0 = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=np.array([phi_target]),
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=N_SUBSTEPS_WARM,
            bisect_depth_warm=BISECT_DEPTH_WARM,
            per_point_callback=_grab,
        )
    elapsed = time.time() - t0

    converged = bool(result.points[0].converged)
    diagnostics = {
        "converged": converged,
        "method": result.points[0].method,
        "achieved_z_factor": float(result.points[0].achieved_z_factor),
        "wall_seconds": float(elapsed),
    }
    if "cd_mA_cm2" in captured:
        diagnostics["cd_mA_cm2"] = captured["cd_mA_cm2"]
        diagnostics["pc_mA_cm2"] = captured["pc_mA_cm2"]
        diagnostics["dof_count"] = captured["dof_count"]
    if not converged or "U_snap" not in captured:
        return None, diagnostics
    return captured["U_snap"], diagnostics


def _step_b_parallel_warmstart(mesh, U_snap_seq, *, snes_opts: dict[str, Any]):
    """Build parallel ctx, copy U from sequential, run single-shot SNES.

    ``U_snap_seq`` is a tuple of per-subspace numpy arrays produced by
    ``_step_a_sequential_cold`` (matches the Forward.bv_solver.grid_per_voltage
    snapshot convention).
    """
    from scripts._bv_common import V_T, I_SCALE
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver.dispatch import build_context, build_forms
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.diagnostics import collect_diagnostics

    sp = _build_parallel_params(snes_opts=snes_opts)
    phi_target = float(V_PROBE) / V_T
    sp = sp.with_phi_applied(phi_target)

    diagnostics: dict[str, Any] = {"v_rhe": float(V_PROBE), "phi_applied_hat": phi_target}

    t0 = time.time()
    with adj.stop_annotating():
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)

        # Sanity: subspace count must match (same mesh, same species config).
        dof_count = int(ctx["U"].function_space().dim())
        diagnostics["dof_count"] = dof_count
        par_subspaces = len(list(ctx["U"].dat))
        if par_subspaces != len(U_snap_seq):
            diagnostics["error"] = (
                f"Subspace mismatch: parallel ctx has {par_subspaces}, "
                f"sequential snapshot has {len(U_snap_seq)} -- species "
                f"config changed between Step A and Step B"
            )
            return None, diagnostics

        # Manual warm-start: copy per-subspace data from the sequential
        # snapshot into ctx["U"] and ctx["U_prev"].  Skips the dispatcher's
        # set_initial_conditions, which would otherwise hit the topology
        # gate and fall back to linear_phi (clobbering the warm copy).
        for src, dst in zip(U_snap_seq, ctx["U"].dat):
            dst.data[:] = src
        for src, dst in zip(U_snap_seq, ctx["U_prev"].dat):
            dst.data[:] = src
        ctx["phi_applied_func"].assign(phi_target)
        # Force full charge (z=1).  build_forms initialises boltzmann_z_scale
        # to 1.0 by default, so this is belt-and-suspenders only.
        if "boltzmann_z_scale" in ctx:
            ctx["boltzmann_z_scale"].assign(1.0)
        for z_const in ctx.get("z_consts", []):
            # z_consts are already at z_nominal[i] from build_forms; no scaling
            # needed (the orchestrator's _set_z_factor multiplies by the ramp
            # value, which is 1.0 for full charge).
            pass

        problem = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
        )
        params_block = sp[10] if hasattr(sp, "__getitem__") else {}
        _NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
        solve_opts = {
            k: v
            for k, v in (params_block.items() if isinstance(params_block, dict) else [])
            if k not in _NON_PETSC_KEYS
        }
        solve_opts.setdefault("snes_error_if_not_converged", True)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)

        snes_converged = False
        snes_error: Optional[str] = None
        try:
            solver.solve()
            snes_converged = True
        except Exception as exc:
            snes_error = f"{type(exc).__name__}: {exc}"
            diagnostics["snes_traceback_tail"] = traceback.format_exc().splitlines()[-3:]

        diagnostics["snes_converged"] = snes_converged
        if snes_error is not None:
            diagnostics["snes_error"] = snes_error

        # Attempt to extract observables either way (NaN-safe via try/except).
        try:
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None, scale=-I_SCALE
            )
            diagnostics["cd_mA_cm2"] = float(fd.assemble(f_cd))
        except Exception as exc:
            diagnostics["cd_mA_cm2_error"] = f"{type(exc).__name__}: {exc}"

        try:
            f_h2o2 = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=None, scale=-I_SCALE
            )
            diagnostics["gross_h2o2_current_mA_cm2"] = float(fd.assemble(f_h2o2))
        except Exception as exc:
            diagnostics["gross_h2o2_current_mA_cm2_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        # Per-reaction rate (raw, no scale) for diagnostic transparency.
        for j in (0, 1):
            try:
                f_rj = _build_bv_observable_form(
                    ctx, mode="reaction", reaction_index=j, scale=-I_SCALE
                )
                diagnostics[f"R_{j}_mA_cm2"] = float(fd.assemble(f_rj))
            except Exception as exc:
                diagnostics[f"R_{j}_error"] = f"{type(exc).__name__}: {exc}"

        try:
            full_diag = collect_diagnostics(
                ctx,
                phase="warmstart_probe_parallel",
                params=params_block if isinstance(params_block, dict) else {},
            )
            diagnostics["solver_diagnostics"] = full_diag
        except Exception as exc:
            diagnostics["solver_diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    diagnostics["wall_seconds"] = time.time() - t0
    return (None if not snes_converged else ctx["U"].dat.data_ro.copy()), diagnostics


def main() -> int:
    from scripts._bv_common import setup_firedrake_env

    setup_firedrake_env()

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    snes_opts = _build_solver_options(EXPONENT_CLIP, U_CLAMP)

    print("=" * 78)
    print("  Parallel 2e/4e warm-start probe @ V_RHE = +0.000 V")
    print("=" * 78)
    print(f"  mesh_Ny       = {MESH_NY}")
    print(f"  formulation   = {FORMULATION}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  output        = {out_dir}")
    print()

    from Forward.bv_solver import make_graded_rectangle_mesh

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    print("--- Step A: cold-solve sequential 3sp+Bikerman+muh @ V=0.0 ---")
    seq_U, seq_diag = _step_a_sequential_cold(mesh, snes_opts=snes_opts)
    print(f"  converged   = {seq_diag['converged']}")
    print(f"  method      = {seq_diag.get('method')}")
    print(f"  cd_mA_cm2   = {seq_diag.get('cd_mA_cm2')}")
    print(f"  pc_mA_cm2   = {seq_diag.get('pc_mA_cm2')}")
    print(f"  wall (s)    = {seq_diag.get('wall_seconds'):.1f}")
    print()

    if seq_U is None:
        report = {
            "v_rhe": float(V_PROBE),
            "step_a": seq_diag,
            "step_b": {"skipped": "step_a_did_not_converge"},
        }
        out_path = os.path.join(out_dir, "probe.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  [Step A failed] report -> {out_path}")
        return 1

    print("--- Step B: warm-start parallel R_2e/R_4e from Step A's U ---")
    par_U, par_diag = _step_b_parallel_warmstart(mesh, seq_U, snes_opts=snes_opts)
    print(f"  snes_converged          = {par_diag.get('snes_converged')}")
    print(f"  cd_mA_cm2               = {par_diag.get('cd_mA_cm2')}")
    print(f"  gross_h2o2_current_mA_cm2 = {par_diag.get('gross_h2o2_current_mA_cm2')}")
    print(f"  R_0 (R_2e)  mA/cm²      = {par_diag.get('R_0_mA_cm2')}")
    print(f"  R_1 (R_4e)  mA/cm²      = {par_diag.get('R_1_mA_cm2')}")
    print(f"  wall (s)                = {par_diag.get('wall_seconds'):.1f}")
    if par_diag.get("snes_error"):
        print(f"  snes_error              = {par_diag['snes_error']}")
    print()

    report = {
        "v_rhe": float(V_PROBE),
        "mesh_Ny": int(MESH_NY),
        "exponent_clip": float(EXPONENT_CLIP),
        "u_clamp": float(U_CLAMP),
        "formulation": FORMULATION,
        "initializer_sequential": INITIALIZER_SEQ,
        "step_a": seq_diag,
        "step_b": par_diag,
        "verdict": (
            "step_b_converged_warmstart_works"
            if par_diag.get("snes_converged")
            else "step_b_diverged_picard_rewrite_needed"
        ),
    }
    out_path = os.path.join(out_dir, "probe.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("=" * 78)
    print(f"  Verdict: {report['verdict']}")
    print(f"  Report:  {out_path}")
    print("=" * 78)
    return 0 if par_diag.get("snes_converged") else 2


if __name__ == "__main__":
    sys.exit(main())
