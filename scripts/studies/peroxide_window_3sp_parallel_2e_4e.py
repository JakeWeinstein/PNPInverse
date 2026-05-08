"""3sp + parallel 2e/4e ORR diagnostic — M3a.2 (Ruggiero 2022 alignment).

Diagnostic-grade driver: replaces the legacy sequential R_0 (O₂→H₂O₂) +
R_1 (H₂O₂→H₂O) BV reaction set with the Ruggiero/Mangan parallel set:

    R_2e:  O₂ + 2H⁺ + 2e⁻ → H₂O₂   E_eq = 0.695 V_RHE   stoich [-1, +1, -2]
    R_4e:  O₂ + 4H⁺ + 4e⁻ → 2H₂O   E_eq = 1.23  V_RHE   stoich [-1,  0, -4]

Three passes per call:

    (A)  pure-2e limit (k0_R4e = 0)         — must reproduce gross R_2e
                                              from the existing 3sp+Bikerman+muh
                                              reference at the same V points.
    (B)  pure-4e limit (k0_R2e = 0)         — peroxide channel must be 0,
                                              n_e_apparent = 4, total cd
                                              bounded by 4e Levich limit.
    (C)  mixed 2e + 4e (both k0 active)     — gross R_2e finite/correct sign;
                                              total cd bounded by 4e Levich.

Initializer is ``linear_phi`` (the conservative IC) because the
matched-asymptotic Picard at ``picard_ic.py:picard_outer_loop`` is
hardcoded to the sequential 2x2 surface-rate algebra and will reject
parallel topology via the M3a.2 topology gate (returns
``non_sequential_topology`` and the orchestrator falls through to
``linear_phi``).  M3a.3 generalizes the Picard.

Output: StudyResults/parallel_2e_4e_diagnostic_m3a2/

NOT production-comparable: ``comparison_status="diagnostic_only"``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# 8-point V_RHE grid spanning the page-15 window.  Includes the
# experimental left-plateau voltage (-0.32 V) and the experimental peak
# location (+0.10 V) as specific anchors for the diagnostic checks.
V_TEST = (-0.40, -0.32, -0.20, 0.00, +0.10, +0.20, +0.30, +0.45)

PASSES = (
    # (label, k0_factor_R2e, k0_factor_R4e)
    ("pure_2e_k04e_zero",   1.0, 0.0),
    ("pure_4e_k02e_zero",   0.0, 1.0),
    ("mixed_equal_priors",  1.0, 1.0),
)

MESH_NY = 200
EXPONENT_CLIP = 100.0   # CLAUDE.md hard rule 2 — clip=100 for PC trustworthiness
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER = "linear_phi"   # diagnostic IC; debye_boltzmann is gated for parallel
FORMULATION = "logc_muh"
OUT_SUBDIR = "parallel_2e_4e_diagnostic_m3a2"

# Audit-anchored RRDE constants (Ruggiero §2.4).  C_O2 = 1.2 mol/m³
# applied 2026-05-07 (M3a.2.1); I_SCALE is 2.4× the legacy ceiling
# (≈ 0.44 mA/cm² at L_REF=100 µm).
N_COLLECTION = 0.224
H_SPECIES_INDEX = 2


def _build_parallel_reactions(k0_factor_R2e: float, k0_factor_R4e: float):
    """Return a parallel-2e/4e reactions list with k0 multiplicatively scaled.

    Reads K0_HAT_R2E and K0_HAT_R4E from _bv_common and applies the
    per-pass factors so we can run pure-2e (factor_R4e=0) and pure-4e
    (factor_R2e=0) limiting tests from the same driver.
    """
    from scripts._bv_common import (
        K0_HAT_R2E, K0_HAT_R4E,
        ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V,
        C_HP_HAT,
    )
    return [
        {
            "k0": float(K0_HAT_R2E) * float(k0_factor_R2e),
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
            "k0": float(K0_HAT_R4E) * float(k0_factor_R4e),
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1,  0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]


def _run_one_pass(
    label: str, k0_factor_R2e: float, k0_factor_R4e: float, *, v_rhe_grid,
) -> dict[str, Any]:
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE, C_SCALE,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )

    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

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

    parallel_reactions = _build_parallel_reactions(k0_factor_R2e, k0_factor_R4e)

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION, log_rate=True,
        u_clamp=U_CLAMP,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        stern_capacitance_f_m2=0.10,
        initializer=INITIALIZER,
        bv_reactions=parallel_reactions,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(v_rhe_grid)
    cd = np.full(NV, np.nan)              # electron-weighted total disk current
    gross_R2e = np.full(NV, np.nan)       # mode='reaction', reaction_index=0
    gross_R4e = np.full(NV, np.nan)       # mode='reaction', reaction_index=1

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_R2e = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=-I_SCALE)
        f_R4e = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=1, scale=-I_SCALE)
        cd[orig_idx]        = float(fd.assemble(f_cd))
        gross_R2e[orig_idx] = float(fd.assemble(f_R2e))
        gross_R4e[orig_idx] = float(fd.assemble(f_R4e))

    phi_hat_grid = np.array(v_rhe_grid) / V_T
    t0 = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=N_SUBSTEPS_WARM,
            bisect_depth_warm=BISECT_DEPTH_WARM,
            per_point_callback=_grab,
        )
    elapsed = time.time() - t0

    # Derived RRDE observables: ring current uses gross_R2e (per the
    # post-Ruggiero observable definition, peroxide current = gross 2e
    # production = single-rate R_2e, NOT the legacy R_0 - R_1 net).
    surface_pH_arr = np.full(NV, np.nan)
    j_ring_arr = np.full(NV, np.nan)
    s_h2o2_arr = np.full(NV, np.nan)
    n_e_arr = np.full(NV, np.nan)
    c_H_surf_arr = np.full(NV, np.nan)
    h_diag_key = f"c{H_SPECIES_INDEX}_surface_mean"
    for i in range(NV):
        diag_i = result.points[i].diagnostics or {}
        c_H_i = diag_i.get(h_diag_key)
        if c_H_i is None or not np.isfinite(c_H_i):
            continue
        c_H_surf_arr[i] = float(c_H_i)
        if not (np.isfinite(cd[i]) and np.isfinite(gross_R2e[i])):
            continue
        rrde = assemble_rrde_observables(
            j_disk=float(cd[i]),
            j_h2o2_disk=float(gross_R2e[i]),
            c_H_surface_nondim=float(c_H_i),
            C_scale_mol_m3=float(C_SCALE),
            N_collection=float(N_COLLECTION),
        )
        surface_pH_arr[i] = rrde.surface_pH_proxy
        j_ring_arr[i] = rrde.j_ring_model
        s_h2o2_arr[i] = rrde.S_H2O2_percent
        n_e_arr[i] = rrde.n_e_rrde

    def _to_json_list(arr):
        return [float(x) if np.isfinite(x) else None for x in arr]

    return {
        "label": label,
        "k0_factor_R2e": float(k0_factor_R2e),
        "k0_factor_R4e": float(k0_factor_R4e),
        "wall_seconds": float(elapsed),
        "v_rhe": list(v_rhe_grid),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "cd_mA_cm2": _to_json_list(cd),
        "gross_R2e_mA_cm2": _to_json_list(gross_R2e),
        "gross_R4e_mA_cm2": _to_json_list(gross_R4e),
        "surface_pH_proxy": _to_json_list(surface_pH_arr),
        "j_ring_mA_cm2": _to_json_list(j_ring_arr),
        "S_H2O2_percent": _to_json_list(s_h2o2_arr),
        "n_e_rrde": _to_json_list(n_e_arr),
        "c_H_surface_nondim": _to_json_list(c_H_surf_arr),
        "N_collection_used": float(N_COLLECTION),
        "converged": [bool(result.points[i].converged) for i in range(NV)],
        "method": [result.points[i].method for i in range(NV)],
        "z_achieved": [float(result.points[i].achieved_z_factor) for i in range(NV)],
        "diagnostics": [result.points[i].diagnostics for i in range(NV)],
        "n_converged": int(sum(result.points[i].converged for i in range(NV))),
        "n_total": int(NV),
    }


def main() -> None:
    import dataclasses
    from scripts._bv_common import make_experiment_metadata, I_SCALE

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    experiment_metadata = make_experiment_metadata(
        catalyst="CMK-3",
        geometry="RRDE",
        pH_bulk=4.0,
        cation="Cs+",
        anion_model="ClO4_protonic_surrogate",
        rotation_rate_rpm=1600.0,
        L_eff_m=None,
        N_collection=N_COLLECTION,
        electrolyte_model="pH_countercharge_surrogate",
        comparison_status="diagnostic_only",  # NEW sentinel — see plan
        source_authority="Ruggiero_manuscript",
        target_curve="Mangan2025_deck_p15_H2O2_current_density_pH4_Csplus",
        acceptance_tier="trend",  # diagnostic = no quant claim
    )
    metadata_dict = dataclasses.asdict(experiment_metadata)

    print("=" * 78)
    print("  Parallel 2e/4e ORR diagnostic — M3a.2 (Ruggiero realignment)")
    print("=" * 78)
    print(f"  V_RHE grid (n={len(V_TEST)}): {list(V_TEST)}")
    print(f"  passes        = {[p[0] for p in PASSES]}")
    print(f"  mesh_Ny       = {MESH_NY}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  initializer   = {INITIALIZER} (parallel topology gates Picard)")
    print(f"  formulation   = {FORMULATION}")
    from scripts._bv_common import C_O2 as _C_O2_PHYS
    print(f"  I_SCALE       = {I_SCALE:.4g} mA/cm²  (= 2e Levich at C_O2={_C_O2_PHYS} mol/m³)")
    print(f"  4e Levich     ≈ {2.0 * I_SCALE:.4g} mA/cm²")
    print(f"  comparison    = {experiment_metadata.comparison_status}")
    print(f"  output        = {out_dir}")
    print()

    reports: list[dict] = []
    t_start = time.time()
    for label, kf2e, kf4e in PASSES:
        print(f"--- pass: {label} (k0_R2e×{kf2e}, k0_R4e×{kf4e}) ---")
        r = _run_one_pass(label, kf2e, kf4e, v_rhe_grid=list(V_TEST))
        reports.append(r)
        print(f"  converged {r['n_converged']}/{r['n_total']}  "
              f"in {r['wall_seconds']:.1f}s")
        for i, v in enumerate(V_TEST):
            ok = r["converged"][i]
            cd_i  = r["cd_mA_cm2"][i]
            R2e_i = r["gross_R2e_mA_cm2"][i]
            R4e_i = r["gross_R4e_mA_cm2"][i]
            sel_i = r["S_H2O2_percent"][i]
            n_e_i = r["n_e_rrde"][i]
            cd_s  = f"{cd_i:+.3e}"  if cd_i  is not None else "(none)"
            R2e_s = f"{R2e_i:+.3e}" if R2e_i is not None else "(none)"
            R4e_s = f"{R4e_i:+.3e}" if R4e_i is not None else "(none)"
            sel_s = f"{sel_i:5.1f}%" if sel_i is not None else "  n/a "
            n_e_s = f"{n_e_i:.2f}" if n_e_i is not None else "n/a "
            print(f"    V={v:+.3f}  ok={ok}  cd={cd_s}  R2e={R2e_s}  R4e={R4e_s}  "
                  f"sel={sel_s}  n_e={n_e_s}")
        print()

    iv_path = os.path.join(out_dir, "iv_curve.json")
    with open(iv_path, "w") as f:
        json.dump({
            "experiment_metadata": metadata_dict,
            "v_rhe": list(V_TEST),
            "passes": [{"label": p[0], "k0_factor_R2e": p[1], "k0_factor_R4e": p[2]}
                       for p in PASSES],
            "mesh_Ny": int(MESH_NY),
            "exponent_clip": float(EXPONENT_CLIP),
            "u_clamp": float(U_CLAMP),
            "initializer": INITIALIZER,
            "formulation": FORMULATION,
            "i_scale_mA_cm2": float(I_SCALE),
            "reports": [
                {k: v for k, v in r.items() if k != "diagnostics"}
                for r in reports
            ],
        }, f, indent=2)
    print(f"  iv_curve.json    -> {iv_path}")

    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({
            "experiment_metadata": metadata_dict,
            "v_rhe": list(V_TEST),
            "reports": [
                {
                    "label": r["label"],
                    "k0_factor_R2e": r["k0_factor_R2e"],
                    "k0_factor_R4e": r["k0_factor_R4e"],
                    "diagnostics_at_v": r["diagnostics"],
                }
                for r in reports
            ],
        }, f, indent=2, default=str)
    print(f"  diagnostics.json -> {diag_path}")

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Total wall time: {elapsed:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
