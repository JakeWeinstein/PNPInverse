"""K0 continuation ladder probe — Phase 2 / Stage 4 of the rescue ladder.

Forks ``parallel_2e_4e_warmstart_probe.py``: keeps Step A (sequential
cold solve at V=0 with debye_boltzmann IC) verbatim, replaces Step B's
single SNES solve with a continuation loop that ramps ``k0_R4e`` from a
tiny positive value to the production ``K0_HAT_R4E`` via
``ctx["bv_k0_funcs"][1].assign(...)`` between rungs.

The Phase-1 ``ln(0)`` guard makes ``k0_R4e=0`` a non-singular but
mathematically-disabled limit; here we instead use small **positive**
values so R_4e contributes a tiny but nonzero rate, matching the
handoff's "tiny → full" continuation idea exactly.

If the bottom rung (``1e-30 × K0_HAT_R4E``) converges, the warm-start +
continuation path is structurally viable and the next plan refines the
ladder + walks voltage.  If even the bottom rung fails, R_4e stiffness
isn't the binding issue and M3a.3 (Picard generalization) is on the
critical path.

Output: ``StudyResults/parallel_2e_4e_k0_ladder_probe/probe.json``
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

V_PROBE = 0.0
MESH_NY = 200
EXPONENT_CLIP = 50.0   # matches warmstart_probe; analytical IC fallback path
                       # in grid_per_voltage.py mandates clip=50 at V=0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER_SEQ = "debye_boltzmann"
FORMULATION = "logc_muh"
OUT_SUBDIR = "parallel_2e_4e_k0_ladder_probe"

# Sanity ladder: three rungs spanning 30 orders of magnitude.  Bottom is
# tame enough that R_4e × clipped_exp ≈ k0_scale × K0_HAT_R4E × exp(100)
# ≈ 1e-30 × 1 × 2.7e43 ≈ 2.7e13 -- still large in absolute terms but
# capped by the eta clip; the continuation contract is "if 1e-30
# converges, reduce step size and walk up; if it doesn't, R_4e ramping
# alone won't save us."
K0_R4E_LADDER = (1.0e-30, 1.0e-15, 1.0)


def _build_solver_options() -> dict[str, Any]:
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


def _build_parallel_params_with_initial_k0(
    *, snes_opts: dict[str, Any], k0_R4e_initial: float
):
    """Build parallel SolverParams with an explicit ``k0_R4e`` value.

    The form-builder reads ``rxn["k0_model"]`` once at construction time
    (forms_logc_muh.py:397: ``k0_j.assign(float(rxn["k0_model"]))``); we
    set the bottom-of-ladder value here so the FIRST solve is at the
    suppressed R_4e regime, then mutate ``ctx["bv_k0_funcs"][1]``
    in-place between rungs.
    """
    from scripts._bv_common import (
        K0_HAT_R2E,
        ALPHA_R2E,
        ALPHA_R4E,
        E_EQ_R2E_V,
        E_EQ_R4E_V,
        C_HP_HAT,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        THREE_SPECIES_LOGC_BOLTZMANN,
        make_bv_solver_params,
    )

    bv_reactions = [
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
            "k0": float(k0_R4e_initial),
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
        bv_reactions=bv_reactions,
        initializer="linear_phi",   # no-op; we skip set_initial_conditions
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _step_a_sequential_cold(mesh, *, snes_opts: dict[str, Any]):
    """Cold-solve sequential at V=0; return (U_snapshot, diagnostics)."""
    from scripts._bv_common import V_T, I_SCALE
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback
    from Forward.bv_solver.observables import _build_bv_observable_form

    sp = _build_sequential_params(snes_opts=snes_opts)
    phi_target = float(V_PROBE) / V_T

    captured: dict[str, Any] = {}

    def _grab(orig_idx, _phi_eta, ctx):
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


def _solve_one_rung(
    ctx, sp_params_block, scale: float, k0_R4e_unit: float
) -> dict[str, Any]:
    """Set ``ctx["bv_k0_funcs"][1]`` to ``scale * k0_R4e_unit``, solve.

    Returns per-rung diagnostics including snes_converged, observables,
    and surface-state telemetry.
    """
    import firedrake as fd
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.diagnostics import collect_diagnostics
    from scripts._bv_common import I_SCALE

    rung: dict[str, Any] = {
        "k0_R4e_scale": float(scale),
        "k0_R4e_value": float(scale * k0_R4e_unit),
    }

    ctx["bv_k0_funcs"][1].assign(float(scale * k0_R4e_unit))

    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
    )
    _NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
    solve_opts = {
        k: v
        for k, v in (
            sp_params_block.items() if isinstance(sp_params_block, dict) else []
        )
        if k not in _NON_PETSC_KEYS
    }
    solve_opts.setdefault("snes_error_if_not_converged", True)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)

    t0 = time.time()
    snes_converged = False
    snes_error: Optional[str] = None
    try:
        solver.solve()
        snes_converged = True
    except Exception as exc:
        snes_error = f"{type(exc).__name__}: {exc}"
        rung["snes_traceback_tail"] = traceback.format_exc().splitlines()[-3:]
    rung["snes_converged"] = snes_converged
    rung["wall_seconds"] = time.time() - t0
    if snes_error is not None:
        rung["snes_error"] = snes_error

    # Observables (NaN-safe).
    for label, mode in (
        ("cd_mA_cm2", "current_density"),
        ("gross_h2o2_current_mA_cm2", "gross_h2o2_current"),
    ):
        try:
            f = _build_bv_observable_form(
                ctx, mode=mode, reaction_index=None, scale=-I_SCALE
            )
            rung[label] = float(fd.assemble(f))
        except Exception as exc:
            rung[f"{label}_error"] = f"{type(exc).__name__}: {exc}"

    for j in (0, 1):
        try:
            f_rj = _build_bv_observable_form(
                ctx, mode="reaction", reaction_index=j, scale=-I_SCALE
            )
            rung[f"R_{j}_mA_cm2"] = float(fd.assemble(f_rj))
        except Exception as exc:
            rung[f"R_{j}_error"] = f"{type(exc).__name__}: {exc}"

    # Surface state telemetry — c_H2O2 blow-up was the failure signature
    # in the prior probe; track it explicitly.
    try:
        full_diag = collect_diagnostics(
            ctx,
            phase=f"k0_ladder_scale_{scale:.0e}",
            params=sp_params_block if isinstance(sp_params_block, dict) else {},
        )
        rung["c0_surface_mean"] = full_diag.get("c0_surface_mean")
        rung["c1_surface_mean"] = full_diag.get("c1_surface_mean")
        rung["c2_surface_mean"] = full_diag.get("c2_surface_mean")
        rung["min_phi"] = full_diag.get("min_phi")
        rung["snes_iters"] = full_diag.get("snes_iters")
    except Exception as exc:
        rung["solver_diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    return rung


def _step_b_continuation(mesh, U_snap_seq, *, snes_opts: dict[str, Any]):
    """Build parallel ctx with k0_R4e seeded TINY, warm-copy U from
    sequential, then walk the ladder via in-place k0 reassignment.

    Returns (final_U_or_None, diagnostics).
    """
    from scripts._bv_common import V_T, K0_HAT_R4E
    import firedrake.adjoint as adj
    from Forward.bv_solver.dispatch import build_context, build_forms

    diagnostics: dict[str, Any] = {
        "v_rhe": float(V_PROBE),
        "k0_R4e_unit": float(K0_HAT_R4E),
        "ladder": list(K0_R4E_LADDER),
        "rungs": [],
    }

    # Seed parallel ctx at the BOTTOM of the ladder so build_forms
    # captures k0_R4e = scale[0] * K0_HAT_R4E.  Subsequent rungs mutate
    # the Function in place.
    sp = _build_parallel_params_with_initial_k0(
        snes_opts=snes_opts, k0_R4e_initial=K0_R4E_LADDER[0] * float(K0_HAT_R4E)
    )
    sp = sp.with_phi_applied(float(V_PROBE) / V_T)

    with adj.stop_annotating():
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)

        diagnostics["dof_count"] = int(ctx["U"].function_space().dim())
        if len(list(ctx["U"].dat)) != len(U_snap_seq):
            diagnostics["error"] = (
                f"subspace mismatch: parallel has {len(list(ctx['U'].dat))}, "
                f"sequential snapshot has {len(U_snap_seq)}"
            )
            return None, diagnostics

        # Manual warm-start (skip set_initial_conditions to avoid
        # topology-gate fallback to linear_phi).
        for src, dst in zip(U_snap_seq, ctx["U"].dat):
            dst.data[:] = src
        for src, dst in zip(U_snap_seq, ctx["U_prev"].dat):
            dst.data[:] = src
        ctx["phi_applied_func"].assign(float(V_PROBE) / V_T)
        if "boltzmann_z_scale" in ctx:
            ctx["boltzmann_z_scale"].assign(1.0)

        params_block = sp[10] if hasattr(sp, "__getitem__") else {}

        for scale in K0_R4E_LADDER:
            print(f"  [rung] k0_R4e scale = {scale:.3e}  "
                  f"(value = {scale * float(K0_HAT_R4E):.3e})", flush=True)
            rung = _solve_one_rung(
                ctx, params_block, scale=scale, k0_R4e_unit=float(K0_HAT_R4E)
            )
            print(f"    converged = {rung['snes_converged']}  "
                  f"cd = {rung.get('cd_mA_cm2')}  "
                  f"R_0 = {rung.get('R_0_mA_cm2')}  "
                  f"R_1 = {rung.get('R_1_mA_cm2')}", flush=True)
            print(f"    c0/c1/c2 surf mean = "
                  f"{rung.get('c0_surface_mean')} / "
                  f"{rung.get('c1_surface_mean')} / "
                  f"{rung.get('c2_surface_mean')}", flush=True)
            if not rung["snes_converged"]:
                print(f"    snes_error = {rung.get('snes_error', '(none)')}",
                      flush=True)
            diagnostics["rungs"].append(rung)
            if not rung["snes_converged"]:
                # Stop ladder on first failure -- no point ramping a
                # diverged state.
                diagnostics["stopped_on_failure"] = True
                diagnostics["stopped_at_scale"] = float(scale)
                return None, diagnostics

    diagnostics["stopped_on_failure"] = False
    return None, diagnostics  # final U not needed for verdict


def main() -> int:
    from scripts._bv_common import setup_firedrake_env
    from Forward.bv_solver import make_graded_rectangle_mesh

    setup_firedrake_env()

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    snes_opts = _build_solver_options()

    print("=" * 78)
    print("  Parallel 2e/4e k0 continuation ladder probe @ V_RHE = +0.000 V")
    print("=" * 78)
    print(f"  mesh_Ny       = {MESH_NY}")
    print(f"  formulation   = {FORMULATION}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  ladder        = {K0_R4E_LADDER}")
    print(f"  output        = {out_dir}")
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    print("--- Step A: cold-solve sequential 3sp+Bikerman+muh @ V=0.0 ---")
    seq_U, seq_diag = _step_a_sequential_cold(mesh, snes_opts=snes_opts)
    print(f"  converged = {seq_diag['converged']}")
    print(f"  method    = {seq_diag.get('method')}")
    print(f"  cd_mA_cm2 = {seq_diag.get('cd_mA_cm2')}")
    print(f"  pc_mA_cm2 = {seq_diag.get('pc_mA_cm2')}")
    print(f"  wall (s)  = {seq_diag.get('wall_seconds'):.1f}")
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

    print("--- Step B: continuation ladder over k0_R4e ---")
    _, par_diag = _step_b_continuation(mesh, seq_U, snes_opts=snes_opts)
    print()

    n_converged = sum(1 for r in par_diag["rungs"] if r.get("snes_converged"))
    n_total = len(par_diag["rungs"])
    if n_converged == n_total and n_total == len(K0_R4E_LADDER):
        verdict = "all_rungs_converged_continuation_viable"
    elif n_converged == 0:
        verdict = "even_bottom_rung_failed_picard_rewrite_needed"
    else:
        last_ok = par_diag["rungs"][n_converged - 1]["k0_R4e_scale"]
        first_fail = par_diag.get("stopped_at_scale")
        verdict = (
            f"converged_through_scale_{last_ok:.0e}_failed_at_{first_fail:.0e}_"
            f"refine_ladder_between_them"
        )

    report = {
        "v_rhe": float(V_PROBE),
        "mesh_Ny": int(MESH_NY),
        "exponent_clip": float(EXPONENT_CLIP),
        "u_clamp": float(U_CLAMP),
        "formulation": FORMULATION,
        "initializer_sequential": INITIALIZER_SEQ,
        "step_a": seq_diag,
        "step_b": par_diag,
        "n_converged": n_converged,
        "n_total": n_total,
        "verdict": verdict,
    }
    out_path = os.path.join(out_dir, "probe.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=" * 78)
    print(f"  Verdict: {verdict}")
    print(f"  Converged rungs: {n_converged}/{n_total}")
    print(f"  Report:  {out_path}")
    print("=" * 78)
    return 0 if n_converged == n_total else 2


if __name__ == "__main__":
    sys.exit(main())
