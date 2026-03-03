"""Forward solver feasibility study: charge configuration sweep.

Tests different charge-vector configurations for the PNP-BV system to
understand how charge assignment affects convergence range and failure modes.

Configurations tested:
    1. z=[0,0,1,-1]  -- full charged 4-species (baseline)
    2. z=[0,1]       -- 2-species, O2 neutral + H2O2 charged (no counter-ion)
                        NOTE: breaks electroneutrality, expect early failure
    3. z=[0,0,1,0]   -- 4-species, only H+ charged (no anion charge)
    4. z=[0,0,0,-1]  -- 4-species, only ClO4- charged (no cation charge)

For each: sweep voltage continuation, record max eta_hat, electrode
concentrations, current density, and failure modes.

Usage (from PNPInverse/ directory)::

    python scripts/studies/forward_solver_feasibility_study.py

Output: StudyResults/forward_solver_feasibility_study/
"""

from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    F_CONST, V_T, N_ELECTRONS,
    D_O2, D_H2O2, D_HP, D_CLO4,
    C_O2, C_H2O2, C_HP, C_CLO4,
    K0_PHYS_R1 as K0_1_PHYS,
    K0_PHYS_R2 as K0_2_PHYS,
    ALPHA_R1 as ALPHA_1,
    ALPHA_R2 as ALPHA_2,
    L_REF, D_REF, C_SCALE,
    SNES_OPTS_CHARGED as SNES_OPTS,
)
setup_firedrake_env()

import firedrake as fd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Forward.bv_solver import (
    build_context,
    build_forms,
    set_initial_conditions,
    make_graded_rectangle_mesh,
)
from Forward.params import SolverParams


# ---------------------------------------------------------------------------
# Script-local constants
# ---------------------------------------------------------------------------

E_EQ_RHE = 0.695

ETA_TARGET = (-0.5 - E_EQ_RHE) / V_T

# Steric a for all feasibility tests (moderate value that helps convergence)
STERIC_A = 0.05


# ---------------------------------------------------------------------------
# Charge configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChargeConfig:
    """Specifies a charge configuration to test."""
    label: str
    n_species: int
    z_vals: List[int]
    D_vals_phys: List[float]        # Physical diffusivities [m2/s]
    c_bulk_phys: List[float]        # Physical bulk concentrations [mol/m3]
    stoi_R1: List[int]              # R1 stoichiometry (per species)
    stoi_R2: List[int]              # R2 stoichiometry (per species)
    # Which species index has cathodic_conc_factors for H+ dependence
    # (None if no H+ dependence in this config)
    hp_species_index: Optional[int] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Define charge configurations to test
# ---------------------------------------------------------------------------

CONFIG_FULL_CHARGED = ChargeConfig(
    label="z=[0,0,+1,-1] (full charged)",
    n_species=4,
    z_vals=[0, 0, 1, -1],
    D_vals_phys=[D_O2, D_H2O2, D_HP, D_CLO4],
    c_bulk_phys=[C_O2, C_H2O2, C_HP, C_CLO4],
    stoi_R1=[-1, +1, -2, 0],
    stoi_R2=[0, -1, -2, 0],
    hp_species_index=2,
    description="Baseline 4-species with full Poisson coupling. "
                "H+ and ClO4- are charged, driving EDL formation.",
)

CONFIG_2SP_CHARGED = ChargeConfig(
    label="z=[0,+1] (2-sp, H2O2 charged)",
    n_species=2,
    z_vals=[0, 1],
    D_vals_phys=[D_O2, D_H2O2],
    c_bulk_phys=[C_O2, C_H2O2],
    stoi_R1=[-1, +1],
    stoi_R2=[0, -1],
    hp_species_index=None,
    description="2-species with H2O2 charged (z=+1). "
                "Breaks electroneutrality — no counterion. "
                "Expect Poisson to create strong electric field. "
                "Diagnostic only.",
)

CONFIG_HP_ONLY = ChargeConfig(
    label="z=[0,0,+1,0] (H+ only charged)",
    n_species=4,
    z_vals=[0, 0, 1, 0],
    D_vals_phys=[D_O2, D_H2O2, D_HP, D_CLO4],
    c_bulk_phys=[C_O2, C_H2O2, C_HP, C_CLO4],
    stoi_R1=[-1, +1, -2, 0],
    stoi_R2=[0, -1, -2, 0],
    hp_species_index=2,
    description="4-species but only H+ carries charge. ClO4- is neutral. "
                "Breaks electroneutrality in one direction. "
                "Tests whether cation-only charge helps or hurts.",
)

CONFIG_CLO4_ONLY = ChargeConfig(
    label="z=[0,0,0,-1] (ClO4- only charged)",
    n_species=4,
    z_vals=[0, 0, 0, -1],
    D_vals_phys=[D_O2, D_H2O2, D_HP, D_CLO4],
    c_bulk_phys=[C_O2, C_H2O2, C_HP, C_CLO4],
    stoi_R1=[-1, +1, -2, 0],
    stoi_R2=[0, -1, -2, 0],
    hp_species_index=2,
    description="4-species but only ClO4- carries charge. H+ is neutral. "
                "Breaks electroneutrality in the other direction.",
)

ALL_CONFIGS = [CONFIG_FULL_CHARGED, CONFIG_2SP_CHARGED,
               CONFIG_HP_ONLY, CONFIG_CLO4_ONLY]


# ---------------------------------------------------------------------------
# Build SolverParams from a ChargeConfig
# ---------------------------------------------------------------------------

def _make_sp(
    charge_cfg: ChargeConfig,
    eta_hat: float,
    *,
    dt: float = 0.5,
    max_ss_steps: int = 80,
    Nx_mesh: int = 8,
    Ny_mesh: int = 300,
    beta_mesh: float = 3.0,
    steric_a: float = STERIC_A,
) -> Tuple[SolverParams, Any]:
    """Build (SolverParams, mesh) for a given charge configuration."""
    n = charge_cfg.n_species
    k_scale = D_REF / L_REF
    k0_1_hat = K0_1_PHYS / k_scale
    k0_2_hat = K0_2_PHYS / k_scale

    # Nondimensionalize
    D_hat = [d / D_REF for d in charge_cfg.D_vals_phys]
    c_hat = [c / C_SCALE for c in charge_cfg.c_bulk_phys]
    a_vals = [steric_a] * n

    mesh = make_graded_rectangle_mesh(Nx_mesh, Ny_mesh, beta_mesh)

    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent":            True,
        "exponent_clip":            50.0,
        "regularize_concentration": True,
        "conc_floor":               1e-12,
        "use_eta_in_bv":            True,
    }
    params["nondim"] = {
        "enabled":                              True,
        "diffusivity_scale_m2_s":               D_REF,
        "concentration_scale_mol_m3":           C_SCALE,
        "length_scale_m":                       L_REF,
        "potential_scale_v":                     V_T,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }

    # Build cathodic_conc_factors if this config has H+ dependence
    cat_factors = []
    if charge_cfg.hp_species_index is not None:
        hp_idx = charge_cfg.hp_species_index
        hp_c_hat = c_hat[hp_idx]
        if hp_c_hat > 0:
            cat_factors = [{"species": hp_idx, "power": 2, "c_ref_nondim": hp_c_hat}]

    params["bv_bc"] = {
        "reactions": [
            {
                "k0": k0_1_hat,
                "alpha": ALPHA_1,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": 1.0,
                "stoichiometry": charge_cfg.stoi_R1,
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": list(cat_factors),
            },
            {
                "k0": k0_2_hat,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": charge_cfg.stoi_R2,
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": list(cat_factors),
            },
        ],
        # Legacy keys for marker config
        "k0": [k0_1_hat] * n,
        "alpha": [ALPHA_1] * n,
        "stoichiometry": charge_cfg.stoi_R1,
        "c_ref": [1.0] * n,
        "E_eq_v": 0.0,
        "electrode_marker":      3,
        "concentration_marker":  4,
        "ground_marker":         4,
    }

    sp = SolverParams.from_list([
        n,
        1,
        dt,
        dt * max_ss_steps,
        charge_cfg.z_vals,
        D_hat,
        a_vals,
        eta_hat,
        c_hat,
        0.0,
        params,
    ])
    return sp, mesh


# ---------------------------------------------------------------------------
# Single configuration voltage sweep
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityResult:
    """Result of one charge configuration test."""
    config_label: str
    z_vals: List[int]
    max_eta_hat: float
    max_v_rhe: float
    n_converged: int
    n_attempted: int
    failure_mode: str
    elapsed_seconds: float
    electrode_concentrations: Dict[str, float]
    eta_hat_values: List[float] = field(default_factory=list)
    I_total_values: List[float] = field(default_factory=list)
    I_peroxide_values: List[float] = field(default_factory=list)


def run_feasibility_test(
    charge_cfg: ChargeConfig,
    *,
    eta_steps: int = 300,
    dt: float = 0.5,
    max_ss_steps: int = 80,
    ss_tol: float = 1e-5,
    Ny_mesh: int = 300,
    beta_mesh: float = 3.0,
) -> FeasibilityResult:
    """Run a voltage continuation sweep for a given charge configuration."""
    i_scale = N_ELECTRONS * F_CONST * (D_REF * C_SCALE / L_REF) * 0.1

    eta_path = np.linspace(0.0, ETA_TARGET, eta_steps + 1)[1:]

    t_start = time.time()

    # Build context at eta=0
    sp_eq, mesh = _make_sp(charge_cfg, 0.0, dt=dt, max_ss_steps=max_ss_steps,
                           Ny_mesh=Ny_mesh, beta_mesh=beta_mesh)
    ctx = build_context(sp_eq, mesh=mesh)
    ctx = build_forms(ctx, sp_eq)
    set_initial_conditions(ctx, sp_eq, blob=False)

    ds = fd.Measure("ds", domain=mesh)
    electrode_marker = 3

    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(F_res, ctx["U"], bcs=ctx["bcs"], J=J_form)

    snes_opts_clean = dict(SNES_OPTS)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=snes_opts_clean)

    U_scratch = ctx["U"].copy(deepcopy=True)
    bv_rate_exprs = ctx.get("bv_rate_exprs", [])
    n = charge_cfg.n_species

    eta_hat_list = []
    I_total_list = []
    I_peroxide_list = []

    max_eta = 0.0
    failure_mode = "none"
    n_converged = 0
    electrode_concs: Dict[str, float] = {}

    for i_step, eta in enumerate(eta_path):
        ctx["phi_applied_func"].assign(float(eta))

        ss_reached = False
        try:
            for k in range(max_ss_steps):
                U_scratch.assign(ctx["U"])
                solver.solve()
                delta_norm = fd.errornorm(ctx["U"], U_scratch, norm_type="L2")
                ref_norm = fd.norm(U_scratch, norm_type="L2")
                rel_change = delta_norm / max(ref_norm, 1e-14)
                ctx["U_prev"].assign(ctx["U"])
                if rel_change < ss_tol:
                    ss_reached = True
                    break
        except fd.ConvergenceError as e:
            ctx["U"].assign(ctx["U_prev"])
            failure_mode = f"ConvergenceError at eta={eta:.2f}: {str(e)[:100]}"
            break
        except Exception as e:
            ctx["U"].assign(ctx["U_prev"])
            failure_mode = f"{type(e).__name__} at eta={eta:.2f}: {str(e)[:100]}"
            break

        # Check for NaN
        has_nan = False
        for d in ctx["U"].dat:
            if np.any(np.isnan(d.data_ro)):
                has_nan = True
                break
        if has_nan:
            failure_mode = f"NaN at eta={eta:.2f}"
            break

        # Current density
        if len(bv_rate_exprs) >= 2:
            R1_val = float(fd.assemble(bv_rate_exprs[0] * ds(electrode_marker)))
            R2_val = float(fd.assemble(bv_rate_exprs[1] * ds(electrode_marker)))
            I_total = -(R1_val + R2_val) * i_scale
            I_peroxide = -(R1_val - R2_val) * i_scale
        else:
            I_total = 0.0
            I_peroxide = 0.0

        # Electrode concentrations
        try:
            for i in range(n):
                electrode_concs[f"c_{i}"] = float(ctx["U"].sub(i).at((0.5, 0.0)))
        except Exception:
            pass

        max_eta = eta
        n_converged = i_step + 1
        eta_hat_list.append(float(eta))
        I_total_list.append(float(I_total))
        I_peroxide_list.append(float(I_peroxide))

        if (i_step + 1) % 20 == 0:
            conc_str = ", ".join(f"c{i}={electrode_concs.get(f'c_{i}', 0):.4f}"
                                 for i in range(n))
            print(f"    [{charge_cfg.label}] step {i_step+1}/{eta_steps}  "
                  f"eta={eta:+7.2f}  I_tot={I_total:+.4e}  {conc_str}")

        # 5-minute timeout
        if time.time() - t_start > 300:
            failure_mode = f"Timeout (300s) at eta={eta:.2f}"
            break
    else:
        failure_mode = "none (reached target)"

    elapsed = time.time() - t_start
    max_v_rhe = E_EQ_RHE + max_eta * V_T

    return FeasibilityResult(
        config_label=charge_cfg.label,
        z_vals=charge_cfg.z_vals,
        max_eta_hat=float(max_eta),
        max_v_rhe=float(max_v_rhe),
        n_converged=n_converged,
        n_attempted=eta_steps,
        failure_mode=failure_mode,
        elapsed_seconds=elapsed,
        electrode_concentrations=electrode_concs,
        eta_hat_values=eta_hat_list,
        I_total_values=I_total_list,
        I_peroxide_values=I_peroxide_list,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = "StudyResults/forward_solver_feasibility_study"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Forward Solver Feasibility Study")
    print(f"  Testing {len(ALL_CONFIGS)} charge configurations")
    print(f"  Target: eta_hat = {ETA_TARGET:.1f} (V_RHE = -0.5 V)")
    print(f"{'='*70}\n")

    results: List[FeasibilityResult] = []

    for i, cfg in enumerate(ALL_CONFIGS):
        print(f"\n--- Config {i+1}/{len(ALL_CONFIGS)}: {cfg.label} ---")
        print(f"    z_vals = {cfg.z_vals}")
        print(f"    {cfg.description}")

        try:
            result = run_feasibility_test(cfg)
            results.append(result)
            print(f"    => max eta_hat = {result.max_eta_hat:.2f} "
                  f"(V_RHE = {result.max_v_rhe:.3f} V), "
                  f"{result.n_converged}/{result.n_attempted} points, "
                  f"{result.elapsed_seconds:.1f}s")
            if result.failure_mode != "none (reached target)":
                print(f"    => Failure: {result.failure_mode}")
        except Exception as e:
            print(f"    => EXCEPTION: {e}")
            traceback.print_exc()
            results.append(FeasibilityResult(
                config_label=cfg.label,
                z_vals=cfg.z_vals,
                max_eta_hat=0.0,
                max_v_rhe=E_EQ_RHE,
                n_converged=0,
                n_attempted=300,
                failure_mode=f"Exception: {str(e)[:200]}",
                elapsed_seconds=0.0,
                electrode_concentrations={},
            ))

    # Save results CSV
    csv_path = os.path.join(out_dir, "feasibility_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "z_vals", "max_eta_hat", "max_v_rhe",
            "n_converged", "n_attempted", "elapsed_s",
            "failure_mode", "electrode_concentrations",
        ])
        for r in results:
            writer.writerow([
                r.config_label, str(r.z_vals),
                f"{r.max_eta_hat:.4f}", f"{r.max_v_rhe:.4f}",
                r.n_converged, r.n_attempted,
                f"{r.elapsed_seconds:.1f}",
                r.failure_mode,
                str(r.electrode_concentrations),
            ])
    print(f"\n[csv] Results saved -> {csv_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  Feasibility Study Results")
    print(f"{'='*90}")
    print(f"{'Configuration':<35} {'max eta':>8} {'V_RHE':>7} {'pts':>4} {'time':>6}  Failure")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: x.max_eta_hat):
        print(f"{r.config_label:<35} {r.max_eta_hat:>+8.2f} {r.max_v_rhe:>+7.3f} "
              f"{r.n_converged:>4d} {r.elapsed_seconds:>5.0f}s  "
              f"{r.failure_mode[:35]}")
    print(f"{'='*90}")

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        if not r.eta_hat_values:
            continue
        V_rhe = [E_EQ_RHE + e * V_T for e in r.eta_hat_values]
        axes[0].plot(V_rhe, r.I_total_values, linewidth=1.5, label=r.config_label)
        axes[1].plot(V_rhe, r.I_peroxide_values, linewidth=1.5, label=r.config_label)

    for ax, title in zip(axes, ["Total Current", "Peroxide Current"]):
        ax.set_xlabel("V vs RHE (V)")
        ax.set_ylabel("Current Density (mA/cm²)")
        ax.set_title(f"Feasibility: {title}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "feasibility_comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Comparison saved -> {png_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r.config_label for r in results]
    etas = [abs(r.max_eta_hat) for r in results]
    colors = ['#2ca02c' if e >= abs(ETA_TARGET) else '#1f77b4' for e in etas]
    ax.barh(range(len(labels)), etas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("|eta_hat| (V_T units)")
    ax.set_title("Feasibility: Maximum |eta_hat| Reached")
    ax.axvline(abs(ETA_TARGET), color="red", linestyle="--", linewidth=1.5,
               label=f"Target: |eta| = {abs(ETA_TARGET):.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "feasibility_max_eta_bar.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Bar chart saved -> {png_path}")

    # Agent log
    log_path = os.path.join(out_dir, "agent_log.md")
    with open(log_path, "w") as f:
        f.write("# Forward Solver Feasibility Study - Agent Log\n\n")
        f.write(f"Target: eta_hat = {ETA_TARGET:.1f} (V_RHE = -0.5 V)\n\n")
        f.write("## Results\n\n")
        for r in sorted(results, key=lambda x: x.max_eta_hat):
            f.write(f"### {r.config_label}\n")
            f.write(f"- z_vals: {r.z_vals}\n")
            f.write(f"- Max eta_hat: {r.max_eta_hat:.2f} (V_RHE = {r.max_v_rhe:.3f} V)\n")
            f.write(f"- Converged: {r.n_converged}/{r.n_attempted}\n")
            f.write(f"- Time: {r.elapsed_seconds:.1f}s\n")
            f.write(f"- Failure: {r.failure_mode}\n")
            f.write(f"- Electrode concentrations: {r.electrode_concentrations}\n\n")

    print(f"\n=== Feasibility Study Complete ===")
    print(f"Output directory: {out_dir}/")


if __name__ == "__main__":
    main()
