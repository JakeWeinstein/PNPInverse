"""Phase D bridge with PHYSICAL per-species a_nondim — Cs⁺/SO₄²⁻ variant.

Same delta vs the uncorrected ``_phase_D_bridge_no_hydrolysis_cs.py``:
swap O₂/H₂O₂/H⁺ ``a_vals_hat`` from placeholder ``A_DEFAULT = 0.01`` to
physical hard-sphere values from Marcus / Stokes radii.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = str(_HERE.parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

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

K0_R4E_FACTOR = 1e-14
L_EFF_M = 16e-6
STERN_ANCHOR = 0.10
STERN_BASELINE = 0.20

N_COLLECTION = 0.224
H_SPECIES_INDEX = 2

OUT_DIR = Path(_ROOT) / "StudyResults" / "phase6b_step10_phase_D_bridge_corrected_a_Cs"

_C_SCALE = 1.2
_N_A = 6.02214076e23


def _a_nondim_from_radius_m(r_m: float) -> float:
    a_phys = (4.0 / 3.0) * math.pi * r_m ** 3 * _N_A
    return a_phys * _C_SCALE


A_O2_PHYSICAL = _a_nondim_from_radius_m(1.70e-10)
A_H2O2_PHYSICAL = _a_nondim_from_radius_m(2.00e-10)
A_HP_PHYSICAL = _a_nondim_from_radius_m(2.80e-10)


def _make_sp(*, stern_capacitance_f_m2: float):
    from scripts._bv_common import (
        ALPHA_R1, ALPHA_R2E, ALPHA_R4E,
        C_HP_HAT, C_O2_HAT, H2O2_SEED_NONDIM,
        D_H2O2_HAT, D_HP_HAT, D_O2_HAT,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        E_EQ_R2E_V, E_EQ_R4E_V,
        K0_HAT_R1, K0_HAT_R2E, K0_HAT_R4E,
        SNES_OPTS_CHARGED,
        SpeciesConfig,
        make_bv_solver_params,
        setup_firedrake_env,
    )
    setup_firedrake_env()

    species = SpeciesConfig(
        n_species=3,
        z_vals=[0, 0, 1],
        d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
        a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL],
        c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT],
        stoichiometry_r1=[-1, +1, -2],
        stoichiometry_r2=[0, -1, -2],
        k0_legacy=[K0_HAT_R1] * 3,
        alpha_legacy=[ALPHA_R1] * 3,
        stoichiometry_legacy=[-1, -1, -1],
        c_ref_legacy=[1.0, 0.0, 1.0],
        roles=["neutral", "neutral", "proton"],
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

    k0_r4e_target = float(K0_HAT_R4E) * float(K0_R4E_FACTOR)
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
            "k0": k0_r4e_target,
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
        species=species,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=float(stern_capacitance_f_m2),
        initializer=INITIALIZER,
        l_eff_m=float(L_EFF_M),
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
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  Phase D bridge — Cs⁺/SO₄²⁻ + PHYSICAL per-species a_nondim")
    print("=" * 78)
    print(f"  V grid     = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)")
    print(f"  Realistic a_nondim:")
    print(f"    O₂   r=1.7Å → a = {A_O2_PHYSICAL:.3e}   (was A_DEFAULT=0.01 = ~14.9Å)")
    print(f"    H₂O₂ r=2.0Å → a = {A_H2O2_PHYSICAL:.3e}")
    print(f"    H⁺   r=2.8Å → a = {A_HP_PHYSICAL:.3e}   (H₃O⁺ Stokes)")
    print(f"  output     = {OUT_DIR}")

    domain_height_hat = L_EFF_M / 1.0e-4
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    sp_baseline, k0_targets = _make_sp(stern_capacitance_f_m2=STERN_BASELINE)
    sp_anchor_cs, _ = _make_sp(stern_capacitance_f_m2=STERN_ANCHOR)
    sp_anchor = sp_anchor_cs.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

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
    print(f"  anchor done in {time.time() - t0:.1f}s", flush=True)

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = ctx_anchor["U"].function_space().dim()

    print(f"\nStage 2: bumping C_S {STERN_ANCHOR:.3f} → {STERN_BASELINE:.3f} F/m²",
          flush=True)
    t_bump = time.time()
    set_stern_capacitance_model(ctx_anchor, float(STERN_BASELINE))
    with adj.stop_annotating():
        ctx_anchor["_last_solver"].solve()
    print(f"  Stern bump re-solved in {time.time() - t_bump:.1f}s", flush=True)

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
            print(f"    cd capture failed @ idx={orig_idx}: {type(exc).__name__}: {exc}")
        try:
            f_pc = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=0,
                scale=-I_SCALE,
            )
            pc_arr[orig_idx] = float(fd.assemble(f_pc))
        except Exception as exc:
            print(f"    pc capture failed @ idx={orig_idx}: {type(exc).__name__}: {exc}")

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
        "label": "phase_D_bridge_corrected_a_Cs",
        "config": {
            "species_preset": "THREE_SPECIES_LOGC_BOLTZMANN (custom a_vals)",
            "counterions": ["Cs+", "SO4(2-)"],
            "a_vals_hat_dynamic": {
                "O2": A_O2_PHYSICAL,
                "H2O2": A_H2O2_PHYSICAL,
                "Hp": A_HP_PHYSICAL,
            },
            "radii_AA": {"O2": 1.7, "H2O2": 2.0, "Hp": 2.8},
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

    cd = np.array([x if x is not None else np.nan for x in report["cd_mA_cm2"]])
    pc = np.array([x if x is not None else np.nan for x in report["pc_mA_cm2"]])
    v_arr = np.array(V_RHE_GRID)
    if np.isfinite(pc).any():
        i_peak = int(np.nanargmin(pc))
        print(f"  most-cathodic pc: V = {v_arr[i_peak]:+.3f} V, "
              f"pc = {pc[i_peak]:.4f} mA/cm²")
        print(f"  max |j_disk| at V = {v_arr[np.nanargmax(np.abs(cd))]:+.3f} V: "
              f"{np.nanmax(np.abs(cd)):.4f} mA/cm²")
        thresh = 0.01 / N_COLLECTION
        onset_idx = np.where(np.abs(pc) >= thresh)[0]
        if len(onset_idx):
            v_onset = v_arr[onset_idx[-1]]
            print(f"  onset (|pc| >= {thresh:.3f}) at V = {v_onset:+.3f} V")

    return 0


if __name__ == "__main__":
    sys.exit(main())
