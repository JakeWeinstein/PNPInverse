"""Single Cs⁺/SO₄²⁻ pH 4 forward run at K0=1e-15 — Phase 6β step-10 follow-up.

Goal: apples-to-apples comparison vs slide-15 envelope (which is Cs⁺
pH 4 RRDE).  Builds the Cs+ dynamic + hydrolysis stack inline (no
production-preset modifications), runs anchor + warm-walk + per-V
λ ramp to λ=1, extracts gross peroxide current pc_gross =
−I_SCALE·R_2e per V (slide-15 convention: cathodic = negative),
and overlays vs the digitized slide-15 envelope.

Outputs:
  StudyResults/phase6b_cs_pH4_K0fit_vs_slide15/
    eval_lambda_1p0000_cs_K0_1e-15.json
    cs_pH4_vs_slide15.png
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Configuration (matches Phase D fit_eval defaults)
# ---------------------------------------------------------------------------

V_ANCHOR: float = +0.55
V_RHE_GRID: Tuple[float, ...] = (
    -0.10,
    -0.06, -0.01, 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34,
    0.39, 0.44, 0.49, 0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84,
    0.89, 0.94, 0.99, 1.00,
)
K_HYD_BASELINE: float = 1e-3
LAMBDA_TARGET: float = 1.0
K0_R4E_FACTOR: float = 1e-15        # corrected deck fit (1 OOM smaller than v10b)
DELTA_BETA_PM2: float = 0.0
SIGMA_MAPPING: str = "stern"

OUT_SUBDIR: str = "phase6b_cs_pH4_K0fit_vs_slide15"

# Cs⁺ diffusivity in water at 25 °C, CRC handbook / Atkins
# (very close to K⁺ which is 1.96e-9; Cs⁺ is slightly larger D due to
# smaller hydrated radius / higher mobility)
D_CSPLUS: float = 2.06e-9    # m²/s


def _build_sp_cs():
    """Build the SolverParams for Cs⁺/SO₄²⁻ + hydrolysis stack at λ=0.

    Mirrors phase6b_v10a_v_sweep_diagnostic._build_sp but swaps K⁺
    for Cs⁺ as the 4th dynamic NP species, and threads cation="Cs+"
    into make_cation_hydrolysis_config so Singh Table S1 / Cu r_H_El
    values are pulled from the Cs⁺ row, not the K⁺ row.
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

    # 4-species dynamic stack: O₂, H₂O₂, H⁺, Cs⁺ (Cs⁺ at index 3 with z=+1).
    # Roles explicit (H⁺ and Cs⁺ both at z=+1; needed for resolvers).
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

    # Parallel 2e + 4e reactions, Ruggiero topology
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
            "k0": float(K0_HAT_R4E),   # base; K0_R4E_FACTOR applied via k0_targets
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

    # Hydrolysis config — note cation="Cs+" pulls Cs⁺-specific Singh
    # Table S1 row (r_M=170, z_eff=0.930, pKa_bulk=14.8) and the Cu
    # r_H_El back-fit value (232.97 pm).  This is the actual cation
    # difference from the K⁺ Phase D fit.
    cation_cfg = make_cation_hydrolysis_config(
        k_hyd=float(V10B_KINETICS["k_hyd_nondim"]),
        k_prot=float(V10B_KINETICS["k_prot_nondim"]),
        k_des=float(V10B_KINETICS["k_des_nondim"]),
        delta_ohp_hat=float(V10B_KINETICS["delta_ohp_hat"]),
        cation="Cs+",                                  # <-- the swap
        r_H_El_pm=None,                                # fall back to Cs+ Cu back-fit
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
        stern_capacitance_f_m2=0.10,       # anchor; will bump to 0.20 post-anchor
        initializer="linear_phi",
        l_eff_m=16e-6,
        enable_water_ionization=True,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
        enable_cation_hydrolysis=True,
        cation_hydrolysis_config=cation_cfg,
        lambda_hydrolysis=0.0,            # walk starts at λ=0; per-V ramp later
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def main() -> int:
    from scripts._bv_common import I_SCALE, L_REF
    from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
        _walk_lambda_zero_capture_snapshots,
        _i_lim_4e_mA_cm2,
        L_EFF_M_BASELINE,
    )
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import _per_v_lambda_ramp
    from Forward.bv_solver import make_graded_rectangle_mesh

    out_dir = Path(_ROOT) / "StudyResults" / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("  Single Cs⁺/SO₄²⁻ pH 4 forward run at K0=1e-15, λ=1, Δβ=0", flush=True)
    print("  (corrected deck-fit K0; apples-to-apples vs slide-15 envelope)", flush=True)
    print("=" * 78, flush=True)
    print(f"  V grid: [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V ({len(V_RHE_GRID)} pts)", flush=True)
    print(f"  V anchor: {V_ANCHOR:+.3f} V", flush=True)
    print(f"  K0_R4E_FACTOR: {K0_R4E_FACTOR:.0e}", flush=True)
    print(f"  Δ_β: {DELTA_BETA_PM2} pm²", flush=True)
    print(f"  cation: Cs⁺ (Singh Table S1: r_M=170 pm, z_eff=0.930, pKa_bulk=14.8)", flush=True)
    print(f"  output: {out_dir}", flush=True)

    sp = _build_sp_cs()
    mesh = make_graded_rectangle_mesh(
        Nx=8, Ny=80, beta=3.0,
        domain_height_hat=L_EFF_M_BASELINE / L_REF,
    )
    domain_height_hat = L_EFF_M_BASELINE / L_REF
    i_lim_4e_mA_cm2 = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)

    # Pass 1: anchor + warm-walk over V grid at λ=0
    print(f"\n[Pass 1] Anchor + warm-walk over {len(V_RHE_GRID)} V pts at λ=0...", flush=True)
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
    print(
        f"  Pass 1 done in {time.time()-t0:.1f}s; "
        f"{sum(1 for r in walk_records if r.get('lambda_zero_converged'))}/{len(walk_records)} converged",
        flush=True,
    )

    # Pass 2: per-V λ ramp to λ=1
    print(f"\n[Pass 2] Per-V λ ramp 0→{LAMBDA_TARGET} for each V...", flush=True)
    t1 = time.time()
    per_v_records: List[Dict[str, Any]] = []
    n_failed = 0
    for grid_idx, voltage in enumerate(V_RHE_GRID):
        if grid_idx not in snapshots:
            per_v_records.append({
                "v_rhe": float(voltage),
                "snes_converged": False,
                "skip_reason": "lambda_zero_warm_walk_failed",
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
            })
            n_failed += 1
            continue
        # Extract gross peroxide current.  R_2e is the 2e reaction rate
        # (positive = forward peroxide production).  Gross peroxide
        # current in slide-15 convention (cathodic = negative):
        #   pc_gross_mA_cm2 = -I_SCALE * R_2e
        r2e = lam1.get("R_2e_current_nondim")
        r4e = lam1.get("R_4e_current_nondim")
        cd = lam1.get("cd_mA_cm2")
        # The pc stored in fit_eval is -I_SCALE*(R_2e - R_4e), the deprecated
        # NET form.  We want the GROSS form: -I_SCALE*R_2e (slide-15).
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
            "pc_gross_mA_cm2": pc_gross_mA_cm2,
            "pc_net_mA_cm2_DEPRECATED": lam1.get("pc_mA_cm2"),
            "cd_mA_cm2": cd,
            "gross_h2o2_pct": gross_h2o2_pct,
            "theta": lam1.get("theta"),
            "gamma_final": lam1.get("gamma_final"),
            "sigma_S_C_per_m2": lam1.get("sigma_S_C_per_m2"),
            "pka_shift_avg": lam1.get("pka_shift_avg"),
            "mass_balance_residual_rel": lam1.get("mass_balance_residual_rel"),
        })
        print(
            f"  V={voltage:+.3f}: pc_gross={pc_gross_mA_cm2:+.4f} mA/cm², "
            f"cd={cd:+.4f}, R_2e={r2e:+.4f}, R_4e={r4e:+.4f}, "
            f"H2O2%={gross_h2o2_pct if gross_h2o2_pct is None else f'{gross_h2o2_pct:5.1f}'}",
            flush=True,
        )
    print(
        f"  Pass 2 done in {time.time()-t1:.1f}s; "
        f"{len(per_v_records) - n_failed}/{len(per_v_records)} V converged at λ=1",
        flush=True,
    )

    # Save JSON
    result = {
        "label": "phase6b_cs_pH4_K0fit_vs_slide15",
        "config": {
            "cation": "Cs+",
            "anion": "SO4²⁻",
            "pH_bulk": 4.0,
            "v_anchor": V_ANCHOR,
            "v_rhe_grid": list(V_RHE_GRID),
            "k0_r4e_factor": K0_R4E_FACTOR,
            "lambda_target": LAMBDA_TARGET,
            "delta_beta_pm2": DELTA_BETA_PM2,
            "sigma_mapping": SIGMA_MAPPING,
            "k_hyd_nondim": K_HYD_BASELINE,
            "stern_anchor": 0.10,
            "stern_baseline": 0.20,
            "l_eff_m": L_EFF_M_BASELINE,
            "i_scale_mA_cm2": float(I_SCALE),
        },
        "per_v_records": per_v_records,
    }
    out_json = out_dir / "eval_lambda_1p0000_cs_K0_1e-15.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result JSON saved → {out_json}", flush=True)
    print(f"  Total wall: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
