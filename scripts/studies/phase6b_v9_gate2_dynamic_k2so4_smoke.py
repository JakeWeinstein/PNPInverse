"""Phase 6β v9 Gate 2 — cathodic dynamic-K⁺ smoke study (production pattern).

Cathodic K2SO4 stack with K⁺ promoted to a DYNAMIC NP species (z = +1)
so the v9 cation-hydrolysis residual at the OHP can read c_K(0) at
solve time.  Sulfate stays as the analytic Bikerman counterion.

Goal: prove the orchestrator converges through V_RHE = -0.40 V on the
4sp dynamic-K⁺ stack (the load-bearing feasibility risk per CLAUDE.md
Hard Rule #5 — see ``.claude/plans/write-up-the-formal-joyful-papert.md``
§Risk callouts).

**Production-style flow** (mirrors CLAUDE.md "Calling the production
solver" multi-ion example):

  1. Anchor at V=+0.55 V (anodic, easy regime) via
     ``solve_anchor_with_continuation`` with k0 ladder + Phase 6α
     Kw_eff ladder.  K⁺ at z=+1 *depletes* near the anodic electrode,
     so the matched-asymptotic IC is well-conditioned here.
  2. Extract a ``PreconvergedAnchor`` from the converged state.
  3. Warm-walk through the production V_RHE grid (anodic → 0 → cathodic)
     via ``solve_grid_with_anchor``.  Each step inherits the prior
     converged state, so the cathodic accumulation of K⁺ at the OHP
     evolves smoothly rather than exploding from a cold IC.

Fallbacks (if the production warm-walk drops a voltage):

  4. Decimal V_RHE refinement around the failure region.
  5. C_S ladder anchor at +0.55 V (relaxes Stern, then warm-walks).
  6. Re-plan + queue another GPT round (manual).

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

# Production V_RHE grid: anchor at +0.55 V (anodic, easy regime), warm-walk
# through 0 V then cathodically to -0.40 V.  Order matters — the warm-walk
# steps each inherit the prior converged state.
V_ANCHOR = 0.55                                # V_RHE, anodic anchor (CLAUDE.md production)
V_RHE_GRID_PRIMARY = (
    0.55, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00,
    -0.10, -0.20, -0.30, -0.40,
)
V_RHE_GRID_DECIMAL = (
    0.55, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.00,
    -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40,
)

# k0 ladder per CLAUDE.md production multi-ion example.
K0_INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)

# Phase 6α Kw_eff ladder per CLAUDE.md (kw=0 floor → physical KW_HAT).
# Built lazily inside _build_kw_ladder() so we can reference KW_HAT.

# C_S fallback ladder (F/m²; production = 0.10).
C_S_LADDER = (1.0, 0.5, 0.25, 0.10)

MESH_NX = 8
MESH_NY = 80                                   # smoke at coarser mesh; production runs use Ny=200
MESH_BETA = 3.0
L_EFF_M = 16e-6                                # Levich-thinned 16 µm
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
INITIALIZER = "linear_phi"   # debye_boltzmann Picard cosh-overflows for asymmetric K2SO4 salt
FORMULATION = "logc_muh"
STERN_PROD_F_M2 = 0.10
ENABLE_WATER_IONIZATION = True

OUT_SUBDIR = "phase6b_v9_gate2_smoke_ny80"


def _build_kw_ladder():
    from scripts._bv_common import KW_HAT
    return (0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT)


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


def _anchor_then_warm_walk(
    *, v_anchor: float, v_rhe_grid, label: str,
    use_kw_ladder: bool = True,
    use_c_s_ladder: bool = False,
) -> Dict[str, Any]:
    """Production-style anchor + warm-walk pass.

    1. solve_anchor_with_continuation at v_anchor with k0 ladder
       (+ optional kw ladder OR optional C_S ladder; the orchestrator
       rejects combining the two).
    2. extract_preconverged_anchor.
    3. solve_grid_with_anchor over the v_rhe_grid (warm-walk).
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation,
        extract_preconverged_anchor,
        LadderExhausted,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import V_T, I_SCALE, K0_HAT_R2E, K0_HAT_R4E

    sp = _build_sp(stern_capacitance_f_m2=STERN_PROD_F_M2)
    sp_anchor = sp.with_phi_applied(v_anchor / V_T)
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
        v_here = v_rhe_grid[orig_idx]
        print(
            f"  [walk] {label} V={v_here:+.3f} V → "
            f"cd={cd[orig_idx]:+.3e} pc={pc[orig_idx]:+.3e} mA/cm²",
            flush=True,
        )

    def _rung_cb(scale, ok, _ctx, rung_diag):
        rung_label = rung_diag.get("rung_label", f"scale={scale:.3e}")
        wall = rung_diag.get("wall_seconds", 0.0)
        cd_obs = rung_diag.get("cd_observable")
        cd_str = f"cd={cd_obs:+.3e}" if cd_obs is not None else "cd=n/a"
        outcome = "OK " if ok else "FAIL"
        print(
            f"  [anchor] {label} {rung_label}  {outcome}  "
            f"{cd_str}  wall={wall:.1f}s",
            flush=True,
        )

    # -- Step A: anchor at v_anchor.
    print(
        f"[anchor] {label} starting at V={v_anchor:+.3f} V "
        f"(use_kw_ladder={use_kw_ladder and not use_c_s_ladder}, "
        f"use_c_s_ladder={use_c_s_ladder})",
        flush=True,
    )
    t0 = time.time()
    anchor_kwargs: Dict[str, Any] = dict(
        mesh=mesh,
        k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
        initial_scales=K0_INITIAL_SCALES,
        max_inserts_per_step=4,
        max_ss_steps_per_rung=300,
        ic_at_target=True,
        rung_callback=_rung_cb,
    )
    if use_c_s_ladder:
        anchor_kwargs["c_s_ladder"] = C_S_LADDER
    elif use_kw_ladder:
        anchor_kwargs["kw_eff_ladder"] = _build_kw_ladder()

    anchor_error: Optional[str] = None
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, **anchor_kwargs,
            )
    except LadderExhausted as exc:
        anchor_error = f"LadderExhausted: {exc}"
        anchor_result = None
    except Exception as exc:
        anchor_error = f"{type(exc).__name__}: {exc}"
        anchor_result = None

    anchor_walltime = time.time() - t0
    anchor_converged = bool(
        anchor_result is not None and anchor_result.converged
    )
    print(
        f"[anchor] {label} DONE  converged={anchor_converged}  "
        f"wall={anchor_walltime:.1f}s",
        flush=True,
    )

    if not anchor_converged:
        return {
            "label": label,
            "v_anchor": float(v_anchor),
            "v_rhe": list(v_rhe_grid),
            "use_kw_ladder": bool(use_kw_ladder and not use_c_s_ladder),
            "use_c_s_ladder": bool(use_c_s_ladder),
            "anchor_converged": False,
            "anchor_error": anchor_error,
            "wall_seconds_anchor": float(anchor_walltime),
            "wall_seconds_walk": 0.0,
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "converged": [False] * NV,
            "method": ["anchor-failed"] * NV,
            "z_achieved": [0.0] * NV,
            "diagnostics": [None] * NV,
            "n_converged": 0,
            "n_total": int(NV),
        }

    # -- Step B: extract anchor.
    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=v_anchor / V_T,
        k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
        mesh_dof_count=anchor_result.ctx["U"].function_space().dim(),
    )

    # -- Step C: warm-walk to the v_rhe_grid.
    print(
        f"[walk] {label} starting warm-walk over {len(v_rhe_grid)} V_RHE points",
        flush=True,
    )
    phi_hat_grid = np.array(v_rhe_grid) / V_T
    t1 = time.time()
    with adj.stop_annotating():
        result = solve_grid_with_anchor(
            sp, mesh=mesh, anchor=anchor,
            phi_applied_values=phi_hat_grid,
            per_point_callback=_grab,
        )
    walk_walltime = time.time() - t1
    print(
        f"[walk] {label} DONE  wall={walk_walltime:.1f}s  "
        f"converged={int(sum(result.points[i].converged for i in range(NV)))}/{NV}",
        flush=True,
    )

    pts = [result.points[i] for i in range(NV)]
    return {
        "label": label,
        "v_anchor": float(v_anchor),
        "v_rhe": list(v_rhe_grid),
        "use_kw_ladder": bool(use_kw_ladder and not use_c_s_ladder),
        "use_c_s_ladder": bool(use_c_s_ladder),
        "anchor_converged": True,
        "anchor_error": None,
        "wall_seconds_anchor": float(anchor_walltime),
        "wall_seconds_walk": float(walk_walltime),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc],
        "converged": [bool(p.converged) for p in pts],
        "method": [p.method for p in pts],
        "z_achieved": [float(p.achieved_z_factor) for p in pts],
        "diagnostics": [p.diagnostics for p in pts],
        "n_converged": int(sum(p.converged for p in pts)),
        "n_total": int(NV),
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


def _target_converged(pass_dict: Dict[str, Any], target_v: float) -> bool:
    """True iff the pass converged at v_rhe == target_v."""
    if not pass_dict.get("anchor_converged"):
        return False
    v_list = pass_dict.get("v_rhe", [])
    conv = pass_dict.get("converged", [])
    for v, ok in zip(v_list, conv):
        if abs(float(v) - target_v) < 1e-6 and bool(ok):
            return True
    return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "stack": "phase6b_v9_gate2_4sp_dynamic_k2so4_cathodic",
        "anchor_strategy": "production_anchor_+0p55V_then_warm_walk",
        "v_anchor": float(V_ANCHOR),
        "v_rhe_grid_primary": list(V_RHE_GRID_PRIMARY),
        "v_rhe_grid_decimal": list(V_RHE_GRID_DECIMAL),
        "k0_initial_scales": list(K0_INITIAL_SCALES),
        "kw_eff_ladder": list(_build_kw_ladder()),
        "c_s_ladder": list(C_S_LADDER),
        "mesh_ny": int(MESH_NY),
        "l_eff_m": float(L_EFF_M),
        "stern_prod_f_m2": float(STERN_PROD_F_M2),
        "enable_water_ionization": bool(ENABLE_WATER_IONIZATION),
        "passes": [],
    }

    target_v = -0.40

    # --- Step 1: production anchor at +0.55 V with kw ladder + warm-walk
    # over the primary V_RHE grid (+0.55 → 0 → -0.40).
    print(
        f"[gate2] Step 1: anchor +{V_ANCHOR:.2f} V (kw ladder) → "
        f"warm-walk {V_RHE_GRID_PRIMARY}",
        flush=True,
    )
    pass1 = _anchor_then_warm_walk(
        v_anchor=V_ANCHOR,
        v_rhe_grid=V_RHE_GRID_PRIMARY,
        label="step1_anchor_kw_warm_walk",
        use_kw_ladder=True,
        use_c_s_ladder=False,
    )
    summary["passes"].append(pass1)

    if _target_converged(pass1, target_v):
        print(f"[gate2] Step 1 converged at V={target_v} V — stopping.",
              flush=True)
    else:
        # --- Step 2: decimal V_RHE refinement (denser warm-walk).
        print(
            f"[gate2] Step 1 missed V={target_v} V; "
            f"Step 2 decimal refinement {V_RHE_GRID_DECIMAL}",
            flush=True,
        )
        pass2 = _anchor_then_warm_walk(
            v_anchor=V_ANCHOR,
            v_rhe_grid=V_RHE_GRID_DECIMAL,
            label="step2_anchor_kw_decimal_walk",
            use_kw_ladder=True,
            use_c_s_ladder=False,
        )
        summary["passes"].append(pass2)

        if _target_converged(pass2, target_v):
            print(f"[gate2] Step 2 converged at V={target_v} V — stopping.",
                  flush=True)
        else:
            # --- Step 3: C_S ladder anchor at +0.55 V then warm-walk.
            #
            # Note: c_s_ladder + kw_eff_ladder cannot be combined (the
            # orchestrator raises NotImplementedError).  Step 3 trades the
            # Kw_eff continuation for the C_S relaxation; the form was
            # built with enable_water_ionization=True so Kw_eff stays at
            # the production KW_HAT throughout.
            print(
                f"[gate2] Step 2 missed; Step 3 anchor +{V_ANCHOR:.2f} V "
                f"with C_S ladder {C_S_LADDER} → warm-walk decimal grid",
                flush=True,
            )
            pass3 = _anchor_then_warm_walk(
                v_anchor=V_ANCHOR,
                v_rhe_grid=V_RHE_GRID_DECIMAL,
                label="step3_anchor_cs_decimal_walk",
                use_kw_ladder=False,
                use_c_s_ladder=True,
            )
            summary["passes"].append(pass3)
            if not _target_converged(pass3, target_v):
                print(
                    f"[gate2] Step 3 also missed V={target_v} V — see "
                    "StudyResults/phase6b_v9_gate2_smoke/FAILURE.md and "
                    "queue another GPT round (R4#7 step 6).",
                    flush=True,
                )

    iv_path = os.path.join(out_dir, "iv_curve.json")
    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(iv_path, "w") as f:
        json.dump(_serialize(summary), f, indent=2)

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
    target_ok = _target_converged(last_pass, target_v)
    print(
        f"[gate2] DONE.  iv_curve={iv_path}\n"
        f"[gate2]        diagnostics={diag_path}\n"
        f"[gate2]        target_converged={target_ok} (V={target_v} V)",
        flush=True,
    )
    return 0 if target_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
