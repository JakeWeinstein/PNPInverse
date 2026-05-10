"""Phase 5γ Pass A grid driver — anchor + warm-walk over V_RHE [+0.10, +0.80].

Anchored at V_RHE = +0.55 V on the multi-ion stack (Cs⁺ + SO₄²⁻ + parallel-2e/4e
+ Stern + logc_muh + debye_boltzmann), this driver:

  1. Builds an anchor via :func:`solve_anchor_with_continuation` (the
     Phase 5γ MVP path that converges Newton at the gate-failure voltage).
  2. Wraps the converged state in a :class:`PreconvergedAnchor`.
  3. Walks outward from the anchor across V_RHE ∈ {+0.10, +0.20, +0.30,
     +0.40, +0.50, +0.60, +0.70, +0.80} via :func:`solve_grid_with_anchor`.
  4. Captures total current density (``mode="current_density"``) and
     the deck-aligned gross peroxide current
     (``mode="gross_h2o2_current"``, i.e. single-rate R_2e — what
     Mangan's "Peroxide Current Density" actually maps to per the
     post-Ruggiero parallel-2e/4e topology audit).
  5. Emits an iv-curve JSON consumed by :mod:`plot_pass_a_grid`.

Pass criteria for exit code 0:
  - the anchor converges, AND
  - ``n_converged >= 7`` of 8 grid voltages.

Usage::

    python -u scripts/studies/pass_a_grid_driver_csplus_so4.py
    echo "Pass A exit: $?"

The driver explicitly uses :func:`solve_grid_with_anchor` rather than C+D
(``solve_grid_per_voltage_cold_with_warm_fallback``); the latter's per-V
cold-start hits the Phase 5α convergence wall that this anchor was built
to bridge.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

ANCHOR_V_RHE = +0.55
V_RHE_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

MESH_NY = 80
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"

# Phase 5γ MVP M5 ladder (proven 25 s anchor build at +0.55 V).
INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP = 4
IC_AT_TARGET = True

OUT_DIR = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08" / "pass_a_grid"
)
OUT_JSON_NAME = "pass_a_iv_curve.json"


# ---------------------------------------------------------------------------
# SolverParams build (mirrors anchor_smoke_csplus_so4_continuation.py)
# ---------------------------------------------------------------------------

def _make_sp():
    from scripts._bv_common import (
        SNES_OPTS_CHARGED,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E, K0_HAT_R4E,
        ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V,
        C_HP_HAT,
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
    rxns = [
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
            "k0": float(K0_HAT_R4E),
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
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION, log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer=INITIALIZER,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)}
    return sp, k0_targets


def main() -> int:
    from scripts._bv_common import setup_firedrake_env, V_T, I_SCALE
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_with_anchor,
    )
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        extract_preconverged_anchor,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp, k0_targets = _make_sp()
    sp_anchor = sp.with_phi_applied(float(ANCHOR_V_RHE) / V_T)
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    print("=" * 78)
    print(f"  Pass A grid driver — multi-ion (Cs⁺ + SO₄²⁻ + parallel-2e/4e)")
    print("=" * 78)
    print(f"  anchor V_RHE  = {ANCHOR_V_RHE:+.3f} V")
    print(f"  grid V_RHE    = {list(V_RHE_GRID)!r}")
    print(f"  mesh_Ny       = {MESH_NY}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  formulation   = {FORMULATION}")
    print(f"  initializer   = {INITIALIZER}")
    print(f"  k0 targets    = R2e={k0_targets[0]:.3e}, R4e={k0_targets[1]:.3e}")
    print(f"  output        = {OUT_DIR}")
    print()

    # ----- 1. Build anchor at +0.55 V via k0 continuation -----
    print(f"--- step 1/2: building anchor at V_RHE={ANCHOR_V_RHE:+.3f} V ---")
    t0 = time.time()
    anchor_converged = False
    anchor_ladder_history: list = []
    anchor_err: str | None = None
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
            )
        anchor_converged = bool(anchor_result.converged)
        anchor_ladder_history = list(anchor_result.ladder_history)
        anchor_dof = int(
            anchor_result.ctx["U"].function_space().dim()
        ) if anchor_result.converged else 0
    except LadderExhausted as exc:
        anchor_err = f"LadderExhausted: {exc}"
        print(f"  anchor build FAILED: {anchor_err}", flush=True)
    anchor_wall = time.time() - t0

    if not anchor_converged:
        # Cannot proceed without an anchor — emit diagnostic JSON and exit 1.
        out_path = OUT_DIR / OUT_JSON_NAME
        with open(out_path, "w") as f:
            json.dump({
                "v_rhe": list(V_RHE_GRID),
                "phi_applied_hat": [
                    float(v) / float(V_T) for v in V_RHE_GRID
                ],
                "cd_mA_cm2": [None] * len(V_RHE_GRID),
                "pc_mA_cm2": [None] * len(V_RHE_GRID),
                "converged": [False] * len(V_RHE_GRID),
                "method": ["anchor-build-failed"] * len(V_RHE_GRID),
                "n_converged": 0,
                "n_total": len(V_RHE_GRID),
                "anchor": {
                    "v_rhe": float(ANCHOR_V_RHE),
                    "converged": False,
                    "ladder_history": [
                        [float(s), str(o)] for s, o in anchor_ladder_history
                    ],
                    "wall_seconds": float(anchor_wall),
                    "error": anchor_err,
                },
                "config": _config_dict(),
            }, f, indent=2)
        print(f"  output -> {out_path}")
        return 1

    print(f"  anchor converged in {anchor_wall:.1f}s "
          f"(ladder={anchor_ladder_history!r})")

    # ----- 2. Extract anchor + walk grid -----
    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=float(ANCHOR_V_RHE) / float(V_T),
        k0_targets=k0_targets,
        mesh_dof_count=anchor_dof,
    )

    NV = len(V_RHE_GRID)
    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)

    def _grab(orig_idx: int, _phi_eta: float, ctx: dict) -> None:
        try:
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None,
                scale=-I_SCALE,
            )
            cd_arr[orig_idx] = float(fd.assemble(f_cd))
        except Exception as exc:
            print(f"  cd capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")
        try:
            # Deck-aligned gross R_2e (Mangan's "Peroxide Current Density");
            # mode="gross_h2o2_current" defaults to reaction_index=0 which
            # is R_2e in the parallel preset.
            f_pc = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=0,
                scale=-I_SCALE,
            )
            pc_arr[orig_idx] = float(fd.assemble(f_pc))
        except Exception as exc:
            print(f"  pc capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")

    print(f"\n--- step 2/2: walking grid from anchor ---")
    phi_grid_eta = np.array(V_RHE_GRID, dtype=float) / float(V_T)
    t0 = time.time()
    grid_result = solve_grid_with_anchor(
        sp,
        anchor=anchor,
        phi_applied_values=phi_grid_eta,
        mesh=mesh,
        n_substeps_warm=N_SUBSTEPS_WARM,
        bisect_depth_warm=BISECT_DEPTH_WARM,
        per_point_callback=_grab,
    )
    grid_wall = time.time() - t0

    # Build outputs in the canonical iv_curve.json schema.
    converged = [bool(grid_result.points[i].converged) for i in range(NV)]
    method = [str(grid_result.points[i].method) for i in range(NV)]
    n_converged = sum(1 for c in converged if c)

    cd_json = [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(cd_arr)
    ]
    pc_json = [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(pc_arr)
    ]

    out = {
        "v_rhe": list(V_RHE_GRID),
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": cd_json,
        "pc_mA_cm2": pc_json,
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
        "anchor": {
            "v_rhe": float(ANCHOR_V_RHE),
            "phi_applied_eta": float(ANCHOR_V_RHE) / float(V_T),
            "k0_targets": {str(j): float(k) for j, k in k0_targets.items()},
            "converged": True,
            "ladder_history": [
                [float(s), str(o)] for s, o in anchor_ladder_history
            ],
            "wall_seconds": float(anchor_wall),
        },
        "grid_wall_seconds": float(grid_wall),
        "config": _config_dict(),
    }

    out_path = OUT_DIR / OUT_JSON_NAME
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("=" * 78)
    print(f"  Pass A: {n_converged}/{NV} converged  "
          f"(grid wall {grid_wall:.1f}s, anchor {anchor_wall:.1f}s)")
    for i, v in enumerate(V_RHE_GRID):
        tag = "OK" if converged[i] else "FAIL"
        cd_s = f"{cd_arr[i]:+.3e}" if np.isfinite(cd_arr[i]) else "  nan   "
        pc_s = f"{pc_arr[i]:+.3e}" if np.isfinite(pc_arr[i]) else "  nan   "
        print(f"    V_RHE={v:+.3f}  {tag}  cd={cd_s} pc={pc_s} {method[i]}")
    print(f"  output -> {out_path}")
    print("=" * 78)

    return 0 if n_converged >= 7 else 1


def _config_dict() -> dict:
    return {
        "mesh_Ny": int(MESH_NY),
        "exponent_clip": float(EXPONENT_CLIP),
        "u_clamp": float(U_CLAMP),
        "stern_capacitance_f_m2": 0.10,
        "ic": INITIALIZER,
        "formulation": FORMULATION,
        "initial_scales": list(INITIAL_SCALES),
        "max_inserts_per_step": int(MAX_INSERTS_PER_STEP),
        "ic_at_target": bool(IC_AT_TARGET),
        "n_substeps_warm": int(N_SUBSTEPS_WARM),
        "bisect_depth_warm": int(BISECT_DEPTH_WARM),
    }


if __name__ == "__main__":
    sys.exit(main())
