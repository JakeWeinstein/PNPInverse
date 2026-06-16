"""Yash-matched hydrolysis-ON forward run (Cs+/SO4, L_eff=6 µm).

Fork of ``_run_cs_pH4_k0fit_vs_slide15.py`` with:
  * L_eff overridable via --l-eff-m (defaults to 6e-6 m, Yash's value)
  * K0_R4e factor overridable via --k0-r4e-factor
  * Output subdir per K0 factor under StudyResults/yash_match_hydrolysis_on/

C_S baseline is kept at 0.20 F/m^2 (the existing two-stage anchor path
bumps from 0.10 to STERN_F_M2_BASELINE=0.20 in a single step).  Yash's
implied C_S ≈ 1.38 F/m^2 (L_Stern=0.5 nm, eps_r=78) requires a
multi-step bump-ladder which is non-trivial to wire here; plot title
flags the C_S difference.

CLI::

    python -u scripts/studies/_run_yash_match_hydrolysis_on.py \\
        --k0-r4e-factor 2.52e-18 \\
        --l-eff-m 6e-6 \\
        --out-name yash_match_hydrolysis_on

Hydrolysis stack: water ionization on (kw_eff ladder),
cation hydrolysis on for Cs+ (Singh Eq.4 with Cs+ pKa_bulk=14.8,
r_M=170 pm, z_eff=0.93), lambda_hydrolysis ramped 0->1 per V.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# Configuration
V_ANCHOR: float = +0.55
V_RHE_GRID: Tuple[float, ...] = tuple(
    round(v, 4) for v in [
        -0.40, -0.36, -0.32, -0.28, -0.24, -0.20, -0.16, -0.12, -0.08,
        -0.04, -0.00, 0.04, 0.08, 0.12, 0.15, 0.19, 0.23, 0.27, 0.31,
        0.35, 0.39, 0.43, 0.47, 0.51, 0.55,
    ]
)
LAMBDA_TARGET: float = 1.0
DELTA_BETA_PM2: float = 0.0
SIGMA_MAPPING: str = "stern"

# Cs+ diffusivity (matches _run_cs_pH4_k0fit_vs_slide15)
D_CSPLUS: float = 2.06e-9


def _build_sp_cs(*, k0_r4e_factor: float, l_eff_m: float):
    """Build SolverParams for Cs+/SO4 + hydrolysis at λ=0, with overridable
    L_eff and K0_R4e factor.  Mirrors the _build_sp_cs helper in
    _run_cs_pH4_k0fit_vs_slide15, but with explicit overrides.
    """
    from scripts._bv_common import (
        A_CSPLUS_HAT, A_H2O2_HAT, A_HP_HAT, A_OH_HAT, A_O2_HAT,
        ALPHA_R2E, ALPHA_R4E, ALPHA_R1,
        C_CSPLUS_HAT, C_HP_HAT, C_O2_HAT, H2O2_SEED_NONDIM,
        D_H2O2_HAT, D_HP_HAT, D_O2_HAT, D_OH_HAT, D_REF,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        E_EQ_R2E_V, E_EQ_R4E_V,
        K0_HAT_R1, K0_HAT_R2E, K0_HAT_R4E, KW_HAT,
        SNES_OPTS_CHARGED,
        SpeciesConfig,
        make_bv_solver_params,
        make_cation_hydrolysis_config,
        setup_firedrake_env,
    )
    from calibration.v10b import V10B_KINETICS
    setup_firedrake_env()

    d_csplus_hat = D_CSPLUS / D_REF
    species = SpeciesConfig(
        n_species=4,
        z_vals=[0, 0, 1, 1],
        d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT, d_csplus_hat],
        a_vals_hat=[A_O2_HAT, A_H2O2_HAT, A_HP_HAT, A_CSPLUS_HAT],
        c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT, C_CSPLUS_HAT],
        stoichiometry_r1=[-1, +1, -2, 0],
        stoichiometry_r2=[0, -1, -2, 0],
        k0_legacy=[K0_HAT_R1] * 4,
        alpha_legacy=[ALPHA_R1] * 4,
        stoichiometry_legacy=[-1, -1, -1, 0],
        c_ref_legacy=[1.0, 0.0, 1.0, 1.0],
        roles=["neutral", "neutral", "proton", "counterion"],
    )

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
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
            "stoichiometry": [-1, +1, -2, 0],
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
            "stoichiometry": [-1, 0, -4, 0],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]

    cation_cfg = make_cation_hydrolysis_config(
        k_hyd=float(V10B_KINETICS["k_hyd_nondim"]),
        k_prot=float(V10B_KINETICS["k_prot_nondim"]),
        k_des=float(V10B_KINETICS["k_des_nondim"]),
        delta_ohp_hat=float(V10B_KINETICS["delta_ohp_hat"]),
        cation="Cs+",
        r_H_El_pm=None,
        pka_shift_form="singh_2016_eq_4",
        gamma_max_nondim=float(V10B_KINETICS["gamma_max_nondim"]),
    )

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=species,
        snes_opts=snes_opts,
        formulation="logc_muh",
        log_rate=True,
        u_clamp=100.0,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
        l_eff_m=float(l_eff_m),
        enable_water_ionization=True,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
        enable_cation_hydrolysis=True,
        cation_hydrolysis_config=cation_cfg,
        lambda_hydrolysis=0.0,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Yash-matched (Cs+/SO4, L_eff=6um) hydrolysis-ON forward run."
    )
    parser.add_argument("--k0-r4e-factor", type=float, required=True,
                        help="K0_R4e/K0_R2e ratio (e.g. 2.52e-18).")
    parser.add_argument("--l-eff-m", type=float, default=6e-6,
                        help="Diffusion-layer thickness (m).  Default 6e-6 (Yash).")
    parser.add_argument("--out-name", type=str, default="yash_match_hydrolysis_on",
                        help="StudyResults subdir name.")
    args = parser.parse_args()

    K0_R4E_FACTOR = float(args.k0_r4e_factor)
    L_EFF_M = float(args.l_eff_m)
    OUT_SUBDIR = str(args.out_name)

    from scripts._bv_common import I_SCALE, L_REF
    from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
        _walk_lambda_zero_capture_snapshots,
        _i_lim_4e_mA_cm2,
    )
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import _per_v_lambda_ramp
    from Forward.bv_solver import make_graded_rectangle_mesh

    factor_label = f"factor_{K0_R4E_FACTOR:.2e}".replace("+", "p").replace("-", "n")
    out_dir = Path(_ROOT) / "StudyResults" / OUT_SUBDIR / factor_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("  Yash-matched HYDROLYSIS-ON forward run", flush=True)
    print(f"  K0_R4e_factor = {K0_R4E_FACTOR:.3e}", flush=True)
    print(f"  L_eff        = {L_EFF_M * 1e6:.2f} um (Yash: 6.0 um)", flush=True)
    print(f"  cation       = Cs+", flush=True)
    print(f"  C_S baseline = 0.20 F/m^2 (production; Yash equiv: 1.38)", flush=True)
    print(f"  lambda target= {LAMBDA_TARGET}", flush=True)
    print(f"  output       = {out_dir}", flush=True)
    print("=" * 78, flush=True)

    sp = _build_sp_cs(k0_r4e_factor=K0_R4E_FACTOR, l_eff_m=L_EFF_M)
    mesh = make_graded_rectangle_mesh(
        Nx=8, Ny=80, beta=3.0,
        domain_height_hat=L_EFF_M / L_REF,
    )
    domain_height_hat = L_EFF_M / L_REF
    i_lim_4e_mA_cm2 = _i_lim_4e_mA_cm2(L_EFF_M)

    # Pass 1: anchor + warm-walk at lambda=0
    print(f"\n[Pass 1] anchor + warm-walk over {len(V_RHE_GRID)} V pts at lambda=0...",
          flush=True)
    t0 = time.time()
    walk_records, snapshots, mesh_dof, electrode_area, electrode_marker = (
        _walk_lambda_zero_capture_snapshots(
            sp=sp, mesh=mesh,
            v_rhe_grid=V_RHE_GRID,
            v_anchor=V_ANCHOR,
            k0_r4e_factor=K0_R4E_FACTOR,
            walk_n_substeps=4,
            walk_max_ss_steps=60,
            walk_ss_rel_tol=1e-3,
        )
    )
    n_lam0_ok = sum(1 for r in walk_records if r.get("lambda_zero_converged"))
    print(f"  Pass 1 done in {time.time()-t0:.1f}s; {n_lam0_ok}/{len(walk_records)} lambda=0 OK",
          flush=True)

    # Pass 2: per-V lambda ramp 0 -> 1
    print(f"\n[Pass 2] per-V lambda ramp 0 -> {LAMBDA_TARGET}...", flush=True)
    t1 = time.time()
    per_v_records: List[Dict[str, Any]] = []
    n_failed = 0
    for grid_idx, voltage in enumerate(V_RHE_GRID):
        if grid_idx not in snapshots:
            per_v_records.append({
                "v_rhe": float(voltage),
                "snes_converged": False,
                "skip_reason": "lambda_zero_warm_walk_failed",
                "cd_mA_cm2": None,
                "pc_gross_mA_cm2": None,
                "gross_h2o2_pct": None,
            })
            n_failed += 1
            continue
        ramp_result = _per_v_lambda_ramp(
            sp_template=sp, mesh=mesh, voltage=float(voltage),
            U_warmstart=snapshots[grid_idx],
            delta_beta_pm2=DELTA_BETA_PM2,
            i_scale=I_SCALE,
            i_lim_4e_mA_cm2=i_lim_4e_mA_cm2,
            electrode_area_nondim=float(electrode_area),
            domain_height_hat=domain_height_hat,
            lambda_target=LAMBDA_TARGET,
            k0_r4e_factor=K0_R4E_FACTOR,
        )
        lam1 = ramp_result.get("lambda1_rung")
        if lam1 is None:
            per_v_records.append({
                "v_rhe": float(voltage),
                "snes_converged": False,
                "skip_reason": "lambda_target_rung_missing",
                "cd_mA_cm2": None,
                "pc_gross_mA_cm2": None,
                "gross_h2o2_pct": None,
            })
            n_failed += 1
            continue
        r2e = lam1.get("R_2e_current_nondim")
        r4e = lam1.get("R_4e_current_nondim")
        cd = lam1.get("cd_mA_cm2")
        # Gross peroxide current (slide-15 convention): -I_SCALE * R_2e
        pc_gross_mA_cm2 = None
        if r2e is not None:
            pc_gross_mA_cm2 = -float(I_SCALE) * float(r2e)
        gross_h2o2_pct = None
        if r2e is not None and r4e is not None:
            denom = float(r2e) + float(r4e)
            if abs(denom) > 1e-12 and float(r2e) >= 0 and float(r4e) >= 0:
                gross_h2o2_pct = 100.0 * float(r2e) / denom
        per_v_records.append({
            "v_rhe": float(voltage),
            "snes_converged": bool(lam1.get("snes_converged", False)),
            "picard_status": lam1.get("picard_status"),
            "R_2e_current_nondim": r2e,
            "R_4e_current_nondim": r4e,
            "cd_mA_cm2": cd,
            "pc_gross_mA_cm2": pc_gross_mA_cm2,
            "gross_h2o2_pct": gross_h2o2_pct,
            "theta": lam1.get("theta"),
            "gamma_final": lam1.get("gamma_final"),
            "sigma_S_C_per_m2": lam1.get("sigma_S_C_per_m2"),
            "pka_shift_avg": lam1.get("pka_shift_avg"),
            "mass_balance_residual_rel": lam1.get("mass_balance_residual_rel"),
        })
    print(f"  Pass 2 done in {time.time()-t1:.1f}s; "
          f"{len(per_v_records)-n_failed}/{len(per_v_records)} V OK at lambda=1",
          flush=True)

    result = {
        "label": "yash_match_hydrolysis_on",
        "config": {
            "cation": "Cs+",
            "anion": "SO4(2-)",
            "pH_bulk": 4.0,
            "v_anchor": V_ANCHOR,
            "v_rhe_grid": list(V_RHE_GRID),
            "k0_r4e_factor": K0_R4E_FACTOR,
            "lambda_target": LAMBDA_TARGET,
            "delta_beta_pm2": DELTA_BETA_PM2,
            "sigma_mapping": SIGMA_MAPPING,
            "stern_anchor_f_m2": 0.10,
            "stern_baseline_f_m2": 0.20,
            "l_eff_m": L_EFF_M,
            "enable_water_ionization": True,
            "enable_cation_hydrolysis": True,
            "i_scale_mA_cm2": float(I_SCALE),
        },
        "per_v_records": per_v_records,
    }
    out_json = out_dir / "iv_curve.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  result JSON saved -> {out_json}", flush=True)
    print(f"  total wall = {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
