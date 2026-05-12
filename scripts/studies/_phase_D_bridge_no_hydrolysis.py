"""Phase D bridge diagnostic — 6β stack MINUS cation hydrolysis MINUS kw.

Tests whether removing cation hydrolysis + water self-ionization
from the deck-baseline V10B stack recovers a Butler-Levich anodic
shape (as in Phase 6α multi-ion) instead of Phase D's plateau-cliff.

Config:
  - Species:       FOUR_SPECIES_LOGC_DYNAMIC_K2SO4 (deck-baseline K⁺/SO₄²⁻)
  - Stern:         0.20 F/m² (V10B baseline; two-stage anchor 0.10→0.20)
  - L_eff:         16 μm (Phase D baseline)
  - k0_R4e_factor: 1e-14 (Phase D production)
  - Cation hyd:    OFF
  - Water ion:     OFF (no kw ladder)
  - V grid:        slide-15 range V_RHE ∈ [-0.50, +0.55], dV = 0.05 (22 pts)
  - Anchor:        V = +0.55 (top of slide-15 band)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = str(_HERE.parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
V_RHE_GRID = tuple(np.round(np.arange(-0.50, 0.5501, 0.05), 4).tolist())
ANCHOR_V_RHE = +0.55

MESH_NX = 8
MESH_NY = 80
MESH_BETA = 3.0
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"

INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP = 4
IC_AT_TARGET = True

K0_R4E_FACTOR = 1e-14   # Phase D production
L_EFF_M = 16e-6         # Phase D baseline
STERN_ANCHOR = 0.10     # two-stage anchor build stage
STERN_BASELINE = 0.20   # V10B production target

N_COLLECTION = 0.224
H_SPECIES_INDEX = 2   # H+ index in 4sp K2SO4 stack

OUT_DIR = Path(_ROOT) / "StudyResults" / "phase6b_step10_phase_D_no_hydrolysis_bridge"


def _make_sp(*, stern_capacitance_f_m2: float):
    from scripts._bv_common import (
        ALPHA_R2E, ALPHA_R4E,
        C_HP_HAT,
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        E_EQ_R2E_V, E_EQ_R4E_V,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        K0_HAT_R2E, K0_HAT_R4E,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
        setup_firedrake_env,
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

    k0_r4e_target = float(K0_HAT_R4E) * float(K0_R4E_FACTOR)
    rxns = [
        {  # R2e: O2 + 2 e- + 2 H+ → H2O2
            "k0": float(K0_HAT_R2E),
            "alpha": float(ALPHA_R2E),
            "cathodic_species": 0,
            "anodic_species": 1,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2, 0],
            "n_electrons": 2,
            "reversible": True,
            "E_eq_v": float(E_EQ_R2E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {  # R4e: O2 + 4 e- + 4 H+ → 2 H2O
            "k0": k0_r4e_target,
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4, 0],
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
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=float(stern_capacitance_f_m2),
        initializer=INITIALIZER,
        l_eff_m=float(L_EFF_M),
        # enable_cation_hydrolysis and enable_water_ionization both default to False
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: float(K0_HAT_R2E), 1: k0_r4e_target}
    return sp, k0_targets


def main() -> int:
    from scripts._bv_common import C_SCALE, I_SCALE, V_T
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_with_anchor,
    )
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        PreconvergedAnchor,
        extract_preconverged_anchor,
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  Phase D bridge — 6β stack MINUS cation hydrolysis MINUS kw")
    print("=" * 78)
    print(f"  V grid     = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)")
    print(f"  anchor V   = {ANCHOR_V_RHE:+.3f} V")
    print(f"  k0_R4e_fac = {K0_R4E_FACTOR:g}")
    print(f"  L_eff      = {L_EFF_M * 1e6:.1f} μm")
    print(f"  Stern path = {STERN_ANCHOR:.3f} → {STERN_BASELINE:.3f} F/m² (two-stage)")
    print(f"  output     = {OUT_DIR}")

    domain_height_hat = L_EFF_M / 1.0e-4
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    sp_baseline, k0_targets = _make_sp(stern_capacitance_f_m2=STERN_BASELINE)
    sp_anchor_cs, _ = _make_sp(stern_capacitance_f_m2=STERN_ANCHOR)
    sp_anchor = sp_anchor_cs.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

    # ---- Stage 1: anchor at C_S = STERN_ANCHOR ----
    print(f"\nStage 1: anchor build at V={ANCHOR_V_RHE:+.3f} V, "
          f"C_S={STERN_ANCHOR:.3f} F/m²", flush=True)
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
            )
    except LadderExhausted as exc:
        raise RuntimeError(f"Anchor failed: {exc}") from exc
    if not anchor_result.converged:
        raise RuntimeError("Anchor did not converge at C_S=0.10.")
    print(f"  anchor done in {time.time() - t0:.1f}s; "
          f"ladder={list(anchor_result.ladder_history)!r}", flush=True)

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = ctx_anchor["U"].function_space().dim()

    # ---- Stage 2: bump C_S 0.10 → 0.20 and Newton-resolve ----
    print(f"\nStage 2: bumping C_S {STERN_ANCHOR:.3f} → {STERN_BASELINE:.3f} F/m²",
          flush=True)
    t_bump = time.time()
    set_stern_capacitance_model(ctx_anchor, float(STERN_BASELINE))
    with adj.stop_annotating():
        ctx_anchor["_last_solver"].solve()
    print(f"  Stern bump re-solved in {time.time() - t_bump:.1f}s", flush=True)

    # Capture the post-bump U as the PreconvergedAnchor.
    U_post_bump = snapshot_U(ctx_anchor["U"])
    anchor = PreconvergedAnchor(
        phi_applied_eta=float(ANCHOR_V_RHE) / V_T,
        U_snapshot=tuple(np.asarray(arr).copy() for arr in U_post_bump),
        k0_targets=tuple(
            (int(j), float(k))
            for j, k in sorted(k0_targets.items())
        ),
        mesh_dof_count=int(mesh_dof_count),
        ladder_history=tuple(
            (float(s), str(o)) for s, o in anchor_result.ladder_history
        ),
    )

    # ---- Grid walk ----
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
            print(f"    cd capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")
        try:
            f_pc = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=0,
                scale=-I_SCALE,
            )
            pc_arr[orig_idx] = float(fd.assemble(f_pc))
        except Exception as exc:
            print(f"    pc capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")

    phi_grid_eta = np.array(V_RHE_GRID, dtype=float) / float(V_T)
    print(f"\nStage 3: grid walk over {NV} V points", flush=True)
    t0 = time.time()
    grid_result = solve_grid_with_anchor(
        sp_baseline,
        anchor=anchor,
        phi_applied_values=phi_grid_eta,
        mesh=mesh,
        n_substeps_warm=N_SUBSTEPS_WARM,
        bisect_depth_warm=BISECT_DEPTH_WARM,
        per_point_callback=_grab,
    )
    grid_wall = time.time() - t0

    converged = [bool(grid_result.points[i].converged) for i in range(NV)]
    method = [str(grid_result.points[i].method) for i in range(NV)]
    n_converged = sum(1 for c in converged if c)
    print(f"  grid: {n_converged}/{NV} converged in {grid_wall:.1f}s", flush=True)

    # Ring-side observables
    j_ring_arr = np.full(NV, np.nan)
    s_h2o2_arr = np.full(NV, np.nan)
    n_e_arr = np.full(NV, np.nan)
    surface_pH_arr = np.full(NV, np.nan)
    c_H_surf_arr = np.full(NV, np.nan)
    h_diag_key = f"c{H_SPECIES_INDEX}_surface_mean"
    for i in range(NV):
        if not converged[i]:
            continue
        diag_i = grid_result.points[i].diagnostics or {}
        c_H_i = diag_i.get(h_diag_key)
        if c_H_i is None or not np.isfinite(c_H_i):
            continue
        c_H_surf_arr[i] = float(c_H_i)
        if not (np.isfinite(cd_arr[i]) and np.isfinite(pc_arr[i])):
            continue
        rrde = assemble_rrde_observables(
            j_disk=float(cd_arr[i]),
            j_h2o2_disk=float(pc_arr[i]),
            c_H_surface_nondim=float(c_H_i),
            C_scale_mol_m3=float(C_SCALE),
            N_collection=float(N_COLLECTION),
        )
        j_ring_arr[i] = rrde.j_ring_model
        s_h2o2_arr[i] = rrde.S_H2O2_percent
        n_e_arr[i] = rrde.n_e_rrde
        surface_pH_arr[i] = rrde.surface_pH_proxy

    def _to_json_list(arr):
        return [
            float(x) if (np.isfinite(x) and converged[i]) else None
            for i, x in enumerate(arr)
        ]

    report = {
        "label": "phase_D_bridge_no_hydrolysis_no_kw",
        "config": {
            "species_preset": "FOUR_SPECIES_LOGC_DYNAMIC_K2SO4",
            "stern_anchor": STERN_ANCHOR,
            "stern_baseline": STERN_BASELINE,
            "l_eff_m": L_EFF_M,
            "k0_r4e_factor": K0_R4E_FACTOR,
            "enable_cation_hydrolysis": False,
            "enable_water_ionization": False,
            "v_rhe_grid": list(V_RHE_GRID),
            "anchor_v_rhe": float(ANCHOR_V_RHE),
            "mesh": {"Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
                     "domain_height_hat": float(domain_height_hat)},
            "formulation": FORMULATION,
            "initializer": INITIALIZER,
            "exponent_clip": EXPONENT_CLIP,
            "u_clamp": U_CLAMP,
            "n_substeps_warm": N_SUBSTEPS_WARM,
            "bisect_depth_warm": BISECT_DEPTH_WARM,
        },
        "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
        "v_rhe": list(V_RHE_GRID),
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": _to_json_list(cd_arr),
        "pc_mA_cm2": _to_json_list(pc_arr),
        "j_ring_mA_cm2": _to_json_list(j_ring_arr),
        "S_H2O2_percent": _to_json_list(s_h2o2_arr),
        "n_e_rrde": _to_json_list(n_e_arr),
        "surface_pH_proxy": _to_json_list(surface_pH_arr),
        "c_H_surface_nondim": _to_json_list(c_H_surf_arr),
        "N_collection_used": float(N_COLLECTION),
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
        "anchor": {
            "v_rhe": float(ANCHOR_V_RHE),
            "phi_applied_eta": float(ANCHOR_V_RHE) / V_T,
            "stern_anchor_f_m2": STERN_ANCHOR,
            "stern_baseline_f_m2": STERN_BASELINE,
            "ladder_history": [
                [float(s), str(o)] for s, o in anchor_result.ladder_history
            ],
        },
        "grid_wall_seconds": float(grid_wall),
    }
    out_path = OUT_DIR / "iv_curve.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")

    # Quick numerical summary
    cd = np.array([x if x is not None else np.nan for x in report["cd_mA_cm2"]])
    pc = np.array([x if x is not None else np.nan for x in report["pc_mA_cm2"]])
    v_arr = np.array(V_RHE_GRID)
    finite = np.isfinite(pc)
    if finite.any():
        i_peak = int(np.nanargmin(pc))  # most negative pc (slide convention)
        print(f"  V_RHE peak peroxide (most cathodic pc): V = {v_arr[i_peak]:+.3f} V, "
              f"pc = {pc[i_peak]:.4f} mA/cm²")
        print(f"  max |j_disk| at V = {v_arr[np.nanargmax(np.abs(cd))]:+.3f} V: "
              f"{np.nanmax(np.abs(cd)):.4f} mA/cm²")
        # Onset (where pc magnitude exceeds 0.045 ≈ ring 0.01/N)
        thresh = 0.01 / N_COLLECTION
        onset_idx = np.where(np.abs(pc) >= thresh)[0]
        if len(onset_idx):
            v_onset = v_arr[onset_idx[-1]]  # most positive V exceeding threshold
            print(f"  onset (pc magnitude >= {thresh:.3f} mA/cm²) at V = {v_onset:+.3f} V")

    return 0


if __name__ == "__main__":
    sys.exit(main())
