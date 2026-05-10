"""Phase 6β v9 Gate 2 — cathodic dynamic-K⁺ smoke study.

Cathodic K2SO4 stack with K⁺ promoted to a DYNAMIC NP species (z = +1)
so the v9 cation-hydrolysis residual at the OHP can read c_K(0) at
solve time.  Sulfate stays as the analytic Bikerman counterion.

Goal: prove the orchestrator converges through V_RHE = -0.40 V on the
4sp dynamic-K⁺ stack (the load-bearing feasibility risk per CLAUDE.md
Hard Rule #5 — see ``.claude/plans/write-up-the-formal-joyful-papert.md``
§Risk callouts).  If the cold + warm-walk ladder fails, the script
escalates through the R4#7 fallbacks:

  1. Cold + z-ramp + warm-walk at C_S=0.10 over the production grid.
  2. Decimal V_RHE refinement (-0.10 .. -0.40 in 0.05 V steps).
  3. C_S ladder (1.0 → 0.10 F/m²) via solve_anchor_with_continuation.
  4. k0 floor reduction (existing k0_continuation).
  5. Re-plan + queue another GPT round (manual).

Outputs:
  StudyResults/phase6b_v9_gate2_smoke/iv_curve.json
  StudyResults/phase6b_v9_gate2_smoke/diagnostics.json

Usage::

    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    source ../venv-firedrake/bin/activate
    export MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 \
           FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1
    python -u scripts/studies/phase6b_v9_gate2_dynamic_k2so4_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

V_RHE_GRID_PRIMARY = (-0.10, -0.20, -0.30, -0.40)
V_RHE_GRID_DECIMAL = (-0.10, -0.20, -0.30, -0.35, -0.40)
C_S_LADDER = (1.0, 0.5, 0.25, 0.10)         # F/m²; production = 0.10
K0_FLOOR_LADDER = (1e-9, 1e-6, 1e-3, 1.0)   # multiplicative scales

MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0
L_EFF_M = 16e-6                              # Levich-thinned 16 µm
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"
STERN_PROD_F_M2 = 0.10
ENABLE_WATER_IONIZATION = True

OUT_SUBDIR = "phase6b_v9_gate2_smoke"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sp(*, stern_capacitance_f_m2: Optional[float] = STERN_PROD_F_M2):
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        PARALLEL_2E_4E_REACTIONS_4SP,
        K0_HAT_R2E,
        K0_HAT_R4E,
        KW_HAT,
        D_OH_HAT,
        A_OH_HAT,
        make_bv_solver_params,
    )
    setup_firedrake_env()

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
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=PARALLEL_2E_4E_REACTIONS_4SP,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=stern_capacitance_f_m2,
        initializer=INITIALIZER,
        l_eff_m=L_EFF_M,
        enable_water_ionization=ENABLE_WATER_IONIZATION,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    return sp


def _make_mesh():
    from Forward.bv_solver import make_graded_rectangle_mesh
    return make_graded_rectangle_mesh(
        Nx=int(MESH_NX), Ny=int(MESH_NY), beta=float(MESH_BETA),
        domain_height_hat=L_EFF_M / 1.0e-4,  # L_REF
    )


def _grid_per_voltage_run(v_rhe_grid, *, label: str) -> Dict[str, Any]:
    """Step 1/2: Cold + z-ramp + warm-walk via the production C+D dispatcher."""
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import V_T, I_SCALE

    sp = _build_sp(stern_capacitance_f_m2=STERN_PROD_F_M2)
    mesh = _make_mesh()
    NV = len(v_rhe_grid)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = np.array(v_rhe_grid) / V_T
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
    pts = [result.points[i] for i in range(NV)]
    return {
        "label": label,
        "v_rhe": list(v_rhe_grid),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc],
        "converged": [bool(p.converged) for p in pts],
        "method": [p.method for p in pts],
        "z_achieved": [float(p.achieved_z_factor) for p in pts],
        "diagnostics": [p.diagnostics for p in pts],
        "n_converged": int(sum(p.converged for p in pts)),
        "n_total": int(NV),
        "wall_seconds": float(elapsed),
    }


def _c_s_ladder_run(*, v_target: float, label: str) -> Dict[str, Any]:
    """Step 3: anchor at v_target with the C_S ladder ramp."""
    import firedrake.adjoint as adj
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation,
        LadderExhausted,
    )
    from scripts._bv_common import V_T, K0_HAT_R2E, K0_HAT_R4E

    sp = _build_sp(stern_capacitance_f_m2=STERN_PROD_F_M2)
    sp = sp.with_phi_applied(v_target / V_T)
    mesh = _make_mesh()
    t0 = time.time()
    converged = False
    cs_history = []
    error_msg: Optional[str] = None
    try:
        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp, mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=K0_FLOOR_LADDER,
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                c_s_ladder=C_S_LADDER,
            )
        converged = bool(result.converged)
        cs_history = list(result.ctx.get("c_s_ladder_history", []))
    except LadderExhausted as exc:
        error_msg = f"LadderExhausted: {exc}"
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0
    return {
        "label": label,
        "v_target": float(v_target),
        "converged": bool(converged),
        "c_s_ladder_history": [
            (float(s), str(o)) for s, o in cs_history
        ],
        "error": error_msg,
        "wall_seconds": float(elapsed),
    }


def _serialize(obj):
    """JSON helper: numpy → native, NaN → null."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int, bool)):
        return obj if isinstance(obj, bool) else int(obj)
    if obj is None or isinstance(obj, (str,)):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    out_dir = os.path.join(
        _ROOT, "StudyResults", OUT_SUBDIR,
    )
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "stack": "phase6b_v9_gate2_4sp_dynamic_k2so4_cathodic",
        "v_rhe_grid_primary": list(V_RHE_GRID_PRIMARY),
        "v_rhe_grid_decimal": list(V_RHE_GRID_DECIMAL),
        "c_s_ladder": list(C_S_LADDER),
        "k0_floor_ladder": list(K0_FLOOR_LADDER),
        "mesh_ny": int(MESH_NY),
        "l_eff_m": float(L_EFF_M),
        "stern_prod_f_m2": float(STERN_PROD_F_M2),
        "enable_water_ionization": bool(ENABLE_WATER_IONIZATION),
        "passes": [],
    }

    # --- Step 1: cold + z-ramp + warm-walk on the primary V_RHE grid.
    print(f"[gate2] Step 1: primary grid {V_RHE_GRID_PRIMARY}", flush=True)
    pass1 = _grid_per_voltage_run(V_RHE_GRID_PRIMARY, label="step1_primary")
    summary["passes"].append(pass1)
    primary_ok_at_target = pass1["converged"][-1]

    if primary_ok_at_target:
        print("[gate2] Step 1 converged at V=-0.40 V — stopping.", flush=True)
    else:
        # --- Step 2: decimal V_RHE refinement.
        print(
            f"[gate2] Step 1 missed -0.40 V; Step 2 decimal refinement "
            f"{V_RHE_GRID_DECIMAL}",
            flush=True,
        )
        pass2 = _grid_per_voltage_run(
            V_RHE_GRID_DECIMAL, label="step2_decimal_refinement",
        )
        summary["passes"].append(pass2)
        decimal_ok_at_target = pass2["converged"][-1]

        if decimal_ok_at_target:
            print("[gate2] Step 2 converged at V=-0.40 V — stopping.", flush=True)
        else:
            # --- Step 3: C_S ladder anchor at V=-0.40 V.
            print(
                f"[gate2] Step 2 missed; Step 3 C_S ladder {C_S_LADDER} at "
                f"V=-0.40 V",
                flush=True,
            )
            pass3 = _c_s_ladder_run(v_target=-0.40, label="step3_c_s_ladder")
            summary["passes"].append(pass3)

            if not pass3["converged"]:
                print(
                    "[gate2] Step 3 failed — the next step (k0 floor) is "
                    "already exercised inside C_S ladder via initial_scales; "
                    "no separate Step 4 needed.  See R4#7 plan.",
                    flush=True,
                )

    iv_path = os.path.join(out_dir, "iv_curve.json")
    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(iv_path, "w") as f:
        json.dump(_serialize(summary), f, indent=2)

    # Diagnostics file: same payload but also pull the per-pass diagnostics.
    diag_payload = {
        "stack": summary["stack"],
        "passes": [
            {
                "label": p["label"],
                "diagnostics": p.get("diagnostics", []),
            }
            for p in summary["passes"] if "diagnostics" in p
        ],
    }
    with open(diag_path, "w") as f:
        json.dump(_serialize(diag_payload), f, indent=2)

    last_pass = summary["passes"][-1]
    if "converged" in last_pass:
        target_ok = (
            isinstance(last_pass["converged"], list)
            and last_pass["converged"][-1]
        ) or (
            isinstance(last_pass["converged"], bool) and last_pass["converged"]
        )
    else:
        target_ok = False

    print(
        f"[gate2] DONE.  iv_curve={iv_path}\n"
        f"[gate2]        diagnostics={diag_path}\n"
        f"[gate2]        target_converged={target_ok}",
        flush=True,
    )
    return 0 if target_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
