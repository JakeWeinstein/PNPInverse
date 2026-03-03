"""Charged 4-species BV solver: iterative voltage range extension study.

Goal: Push the charged 4-species PNP-BV solver from its current limit
(eta_hat ~ -10) toward the experimental range (V_RHE = -0.5 V, eta_hat ~ -46.5).

The study iterates over parameter configurations to find the combination of
steric exclusion (Bikerman a), mesh resolution, Newton damping, time-stepping,
and continuation strategy that maximizes the convergence range.

The key hypothesis is that increasing steric `a` prevents electrode-surface
concentration collapse (c -> 0), which is the dominant failure mode at large
cathodic overpotentials.

Usage (from PNPInverse/ directory)::

    python scripts/studies/charged_voltage_range_study.py
    python scripts/studies/charged_voltage_range_study.py --round 1
    python scripts/studies/charged_voltage_range_study.py --round 2

Output: StudyResults/charged_voltage_range_study/
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
    D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT,
    C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT,
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
from Forward.steady_state import compute_bv_reaction_rates


# ---------------------------------------------------------------------------
# Script-local constants
# ---------------------------------------------------------------------------

E_EQ_RHE = 0.695

# Experimental target: V_RHE = -0.5 V -> eta_hat = (-0.5 - 0.695) / V_T
ETA_TARGET = (-0.5 - E_EQ_RHE) / V_T  # ~ -46.5


# ---------------------------------------------------------------------------
# Configuration dataclass for a single test run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Configuration for one voltage-range test run."""
    label: str                        # Human-readable label
    steric_a: float = 0.0             # Bikerman steric parameter (all species)
    Ny_mesh: int = 300                # Mesh cells in y
    beta_mesh: float = 3.0            # Mesh grading exponent
    Nx_mesh: int = 8                  # Mesh cells in x
    lambda_max: float = 0.5           # SNES line-search max step
    exponent_clip: float = 50.0       # BV exponent clipping bound
    conc_floor: float = 1e-12         # Concentration regularization floor
    dt: float = 0.5                   # Nondim time step
    max_ss_steps: int = 80            # Max time steps per voltage point
    eta_steps: int = 200              # Number of continuation steps
    ss_tol: float = 1e-5              # Steady-state convergence tolerance
    snes_max_it: int = 300            # SNES max iterations per time step
    linesearch_type: str = "l2"       # Line-search type: l2, bt, cp, basic
    l_ref: float = L_REF              # Domain length [m]
    # Per-species steric a (overrides steric_a if set)
    steric_a_vals: Optional[List[float]] = None

    def get_a_vals(self) -> List[float]:
        """Return per-species steric a values."""
        if self.steric_a_vals is not None:
            return list(self.steric_a_vals)
        return [self.steric_a] * 4


@dataclass
class RunResult:
    """Result of one voltage-range test run."""
    config: RunConfig
    max_eta_hat: float                # Most negative eta_hat reached
    max_v_rhe: float                  # Corresponding V_RHE
    n_points_converged: int           # Number of voltage points solved
    n_points_attempted: int           # Total points attempted
    failure_mode: str                 # Description of the failure
    elapsed_seconds: float            # Wall time
    final_c_O2_electrode: float       # O2 conc at electrode at failure/end
    final_c_HP_electrode: float       # H+ conc at electrode at failure/end
    eta_hat_values: List[float] = field(default_factory=list)
    I_total_values: List[float] = field(default_factory=list)
    I_peroxide_values: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build SolverParams for a given config
# ---------------------------------------------------------------------------

def _make_sp(config: RunConfig, eta_hat: float) -> Tuple[SolverParams, Any]:
    """Build (SolverParams, mesh) for the given configuration."""
    k_scale = D_REF / config.l_ref
    k0_1_hat = K0_1_PHYS / k_scale
    k0_2_hat = K0_2_PHYS / k_scale

    c_hp_hat = C_HP / C_SCALE
    c_clo4_hat = C_CLO4 / C_SCALE

    mesh = make_graded_rectangle_mesh(config.Nx_mesh, config.Ny_mesh, config.beta_mesh)

    a_vals = config.get_a_vals()

    params = {
        "snes_type":                 "newtonls",
        "snes_max_it":               config.snes_max_it,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      config.linesearch_type,
        "snes_linesearch_maxlambda": config.lambda_max,
        "snes_divergence_tolerance": 1e12,
        "ksp_type":                  "preonly",
        "pc_type":                   "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8":         77,
        "mat_mumps_icntl_14":        80,
    }

    params["bv_convergence"] = {
        "clip_exponent":            True,
        "exponent_clip":            config.exponent_clip,
        "regularize_concentration": True,
        "conc_floor":               config.conc_floor,
        "use_eta_in_bv":            True,
    }

    params["nondim"] = {
        "enabled":                              True,
        "diffusivity_scale_m2_s":               D_REF,
        "concentration_scale_mol_m3":           C_SCALE,
        "length_scale_m":                       config.l_ref,
        "potential_scale_v":                     V_T,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }

    params["bv_bc"] = {
        "reactions": [
            {
                "k0": k0_1_hat,
                "alpha": ALPHA_1,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": 1.0,
                "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
            {
                "k0": k0_2_hat,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
        ],
        "k0": [k0_1_hat] * 4,
        "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0],
        "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker":      3,
        "concentration_marker":  4,
        "ground_marker":         4,
    }

    sp = SolverParams.from_list([
        4,
        1,
        config.dt,
        config.dt * config.max_ss_steps,
        [0, 0, 1, -1],
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
        a_vals,
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, c_hp_hat, c_clo4_hat],
        0.0,
        params,
    ])
    return sp, mesh


# ---------------------------------------------------------------------------
# Single test run: voltage continuation sweep
# ---------------------------------------------------------------------------

def run_single_config(config: RunConfig) -> RunResult:
    """Run a voltage continuation sweep with the given configuration.

    Returns a RunResult with the maximum eta_hat reached before failure.
    The sweep goes from eta=0 toward ETA_TARGET in config.eta_steps steps.
    """
    i_scale = N_ELECTRONS * F_CONST * (D_REF * C_SCALE / config.l_ref) * 0.1

    eta_path = np.linspace(0.0, ETA_TARGET, config.eta_steps + 1)[1:]

    t_start = time.time()

    # Build context at eta=0
    sp_eq, mesh = _make_sp(config, 0.0)
    ctx = build_context(sp_eq, mesh=mesh)
    ctx = build_forms(ctx, sp_eq)
    set_initial_conditions(ctx, sp_eq, blob=False)

    scaling = ctx["nondim"]
    ds = fd.Measure("ds", domain=mesh)
    electrode_marker = 3

    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(F_res, ctx["U"], bcs=ctx["bcs"], J=J_form)

    snes_opts = dict(sp_eq[10])
    # Remove non-PETSc keys
    for k in ["bv_convergence", "nondim", "bv_bc"]:
        snes_opts.pop(k, None)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=snes_opts)

    U_scratch = ctx["U"].copy(deepcopy=True)
    bv_rate_exprs = ctx.get("bv_rate_exprs", [])

    eta_hat_list = []
    I_total_list = []
    I_peroxide_list = []

    max_eta = 0.0
    c_O2_elec = float(C_O2_HAT)
    c_HP_elec = float(C_HP_HAT)
    failure_mode = "none"
    n_converged = 0

    for i_step, eta in enumerate(eta_path):
        ctx["phi_applied_func"].assign(float(eta))
        V_rhe = E_EQ_RHE + eta * V_T

        ss_reached = False
        n_taken = 0

        try:
            for k in range(config.max_ss_steps):
                U_scratch.assign(ctx["U"])
                solver.solve()

                delta_norm = fd.errornorm(ctx["U"], U_scratch, norm_type="L2")
                ref_norm = fd.norm(U_scratch, norm_type="L2")
                rel_change = delta_norm / max(ref_norm, 1e-14)

                ctx["U_prev"].assign(ctx["U"])
                n_taken = k + 1

                if rel_change < config.ss_tol:
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

        # Check for NaN in solution (can happen with steric model)
        u_data = ctx["U"].dat.data_ro
        if hasattr(u_data, '__iter__'):
            has_nan = False
            for d in ctx["U"].dat:
                if np.any(np.isnan(d.data_ro)):
                    has_nan = True
                    break
        else:
            has_nan = np.any(np.isnan(u_data))

        if has_nan:
            failure_mode = f"NaN in solution at eta={eta:.2f}"
            break

        # Compute current density
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
            c_O2_elec = float(ctx["U"].sub(0).at((0.5, 0.0)))
            c_HP_elec = float(ctx["U"].sub(2).at((0.5, 0.0)))
        except Exception:
            c_O2_elec = float("nan")
            c_HP_elec = float("nan")

        # Check for negative concentrations (non-physical)
        if c_O2_elec < -0.01 or c_HP_elec < -0.01:
            failure_mode = f"Negative concentration at eta={eta:.2f}: c_O2={c_O2_elec:.4f}, c_HP={c_HP_elec:.4f}"
            break

        max_eta = eta
        n_converged = i_step + 1
        eta_hat_list.append(float(eta))
        I_total_list.append(float(I_total))
        I_peroxide_list.append(float(I_peroxide))

        # Print progress every 20 steps
        if (i_step + 1) % 20 == 0:
            print(f"    [{config.label}] step {i_step+1}/{config.eta_steps}  "
                  f"eta={eta:+7.2f}  I_tot={I_total:+.4e}  "
                  f"c_O2={c_O2_elec:.4f}  c_H+={c_HP_elec:.4f}")

        # 5-minute timeout per config
        elapsed = time.time() - t_start
        if elapsed > 300:
            failure_mode = f"Timeout (300s) at eta={eta:.2f}"
            break

    else:
        # Completed all steps without break
        failure_mode = "none (reached target)"

    elapsed = time.time() - t_start
    max_v_rhe = E_EQ_RHE + max_eta * V_T

    return RunResult(
        config=config,
        max_eta_hat=float(max_eta),
        max_v_rhe=float(max_v_rhe),
        n_points_converged=n_converged,
        n_points_attempted=config.eta_steps,
        failure_mode=failure_mode,
        elapsed_seconds=elapsed,
        final_c_O2_electrode=float(c_O2_elec),
        final_c_HP_electrode=float(c_HP_elec),
        eta_hat_values=eta_hat_list,
        I_total_values=I_total_list,
        I_peroxide_values=I_peroxide_list,
    )


# ---------------------------------------------------------------------------
# Round 1: Baseline parameter sweep
# ---------------------------------------------------------------------------

def round1_configs() -> List[RunConfig]:
    """Generate Round 1 configurations for the initial parameter sweep.

    Tests: steric a, mesh resolution, Newton damping, time-stepping, continuation.
    """
    configs = []

    # Baseline: no steric, current defaults
    configs.append(RunConfig(
        label="baseline_no_steric",
        steric_a=0.0,
        Ny_mesh=300, beta_mesh=3.0,
        lambda_max=0.5, exponent_clip=50.0,
        dt=0.5, max_ss_steps=80, eta_steps=200,
    ))

    # Steric a sweep (main hypothesis)
    for a_val in [0.01, 0.05, 0.10, 0.20]:
        configs.append(RunConfig(
            label=f"steric_a={a_val:.2f}",
            steric_a=a_val,
            Ny_mesh=300, beta_mesh=3.0,
            lambda_max=0.5, exponent_clip=50.0,
            dt=0.5, max_ss_steps=80, eta_steps=200,
        ))

    # Mesh refinement (Ny=500, beta=3.5 for better Debye layer resolution)
    configs.append(RunConfig(
        label="mesh_Ny500_b3.5",
        steric_a=0.05,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.5, exponent_clip=50.0,
        dt=0.5, max_ss_steps=80, eta_steps=200,
    ))

    # More conservative Newton
    for lam in [0.3, 0.2]:
        configs.append(RunConfig(
            label=f"lambda_max={lam:.1f}_a=0.05",
            steric_a=0.05,
            lambda_max=lam,
            dt=0.5, max_ss_steps=80, eta_steps=200,
        ))

    # Exponent clipping
    for clip in [30.0, 20.0]:
        configs.append(RunConfig(
            label=f"clip={clip:.0f}_a=0.05",
            steric_a=0.05,
            exponent_clip=clip,
            dt=0.5, max_ss_steps=80, eta_steps=200,
        ))

    # Smaller time step + more SS steps
    configs.append(RunConfig(
        label="dt=0.2_ss=150_a=0.05",
        steric_a=0.05,
        dt=0.2, max_ss_steps=150, eta_steps=200,
    ))

    # More continuation steps (finer voltage increments)
    configs.append(RunConfig(
        label="steps=500_a=0.05",
        steric_a=0.05,
        eta_steps=500,
        dt=0.5, max_ss_steps=80,
    ))

    # Combined: more continuation + smaller dt
    configs.append(RunConfig(
        label="steps=500_dt=0.2_a=0.10",
        steric_a=0.10,
        eta_steps=500,
        dt=0.2, max_ss_steps=150,
    ))

    return configs


# ---------------------------------------------------------------------------
# Round 2: Targeted improvements based on Round 1 findings
# ---------------------------------------------------------------------------

def round2_configs(best_a: float = 0.10) -> List[RunConfig]:
    """Generate Round 2 configs combining best-performing settings from Round 1.

    Uses the best steric a value found in Round 1 as the base.
    """
    configs = []

    # Best combo from Round 1 with refinements
    configs.append(RunConfig(
        label=f"r2_best_combo_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        exponent_clip=30.0,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    # Extreme steric values
    for a_val in [0.30, 0.40]:
        configs.append(RunConfig(
            label=f"r2_steric_a={a_val:.2f}",
            steric_a=a_val,
            Ny_mesh=500, beta_mesh=3.5,
            lambda_max=0.3,
            exponent_clip=30.0,
            dt=0.2, max_ss_steps=200,
            eta_steps=500,
        ))

    # Very conservative Newton (lambda_max=0.1)
    configs.append(RunConfig(
        label=f"r2_lam=0.1_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.1,
        exponent_clip=30.0,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    # BT (backtracking) line search instead of L2
    configs.append(RunConfig(
        label=f"r2_bt_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.5,
        linesearch_type="bt",
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    # Very small conc_floor
    configs.append(RunConfig(
        label=f"r2_floor=1e-6_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        conc_floor=1e-6,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    # Higher conc_floor (more aggressive regularization)
    configs.append(RunConfig(
        label=f"r2_floor=1e-4_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        conc_floor=1e-4,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    # 1000 continuation steps (very fine voltage increments)
    configs.append(RunConfig(
        label=f"r2_steps=1000_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        dt=0.2, max_ss_steps=200,
        eta_steps=1000,
    ))

    # Clip exponent at 15 (very aggressive clipping)
    configs.append(RunConfig(
        label=f"r2_clip=15_a={best_a}",
        steric_a=best_a,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        exponent_clip=15.0,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    # Larger domain L_ref=300um (reduces (lambda_D/L)^2 stiffness)
    configs.append(RunConfig(
        label=f"r2_L=300um_a={best_a}",
        steric_a=best_a,
        l_ref=3e-4,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        exponent_clip=30.0,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    return configs


# ---------------------------------------------------------------------------
# Round 3: Final push combining all insights
# ---------------------------------------------------------------------------

def round3_configs(best_result: RunResult) -> List[RunConfig]:
    """Generate Round 3 configs based on the best result from Round 2.

    Tries extreme combinations to push as far as possible.
    """
    cfg = best_result.config
    configs = []

    # Take the best config from Round 2 and push with even more steps
    configs.append(RunConfig(
        label="r3_extreme_steps",
        steric_a=cfg.steric_a,
        steric_a_vals=cfg.steric_a_vals,
        Ny_mesh=cfg.Ny_mesh,
        beta_mesh=cfg.beta_mesh,
        lambda_max=cfg.lambda_max,
        exponent_clip=cfg.exponent_clip,
        conc_floor=cfg.conc_floor,
        linesearch_type=cfg.linesearch_type,
        dt=cfg.dt,
        max_ss_steps=cfg.max_ss_steps,
        eta_steps=2000,  # Very fine continuation
        l_ref=cfg.l_ref,
    ))

    # Even more conservative Newton
    configs.append(RunConfig(
        label="r3_ultra_conservative",
        steric_a=cfg.steric_a,
        steric_a_vals=cfg.steric_a_vals,
        Ny_mesh=cfg.Ny_mesh,
        beta_mesh=cfg.beta_mesh,
        lambda_max=0.05,
        exponent_clip=cfg.exponent_clip,
        conc_floor=cfg.conc_floor,
        linesearch_type=cfg.linesearch_type,
        dt=0.1,
        max_ss_steps=300,
        eta_steps=1000,
        snes_max_it=500,
        l_ref=cfg.l_ref,
    ))

    # Higher steric + fine mesh + fine continuation
    configs.append(RunConfig(
        label="r3_steric_0.5",
        steric_a=0.50,
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.2,
        exponent_clip=20.0,
        conc_floor=1e-6,
        dt=0.2, max_ss_steps=200,
        eta_steps=1000,
    ))

    # Asymmetric steric: larger a for charged species (H+, ClO4-)
    # to prevent their collapse while keeping neutral species less affected
    configs.append(RunConfig(
        label="r3_asymmetric_steric",
        steric_a=0.0,
        steric_a_vals=[0.02, 0.02, 0.20, 0.20],  # larger for ions
        Ny_mesh=500, beta_mesh=3.5,
        lambda_max=0.3,
        exponent_clip=30.0,
        dt=0.2, max_ss_steps=200,
        eta_steps=500,
    ))

    return configs


# ---------------------------------------------------------------------------
# Result I/O and summary
# ---------------------------------------------------------------------------

def save_results_csv(results: List[RunResult], csv_path: str) -> None:
    """Save all run results to CSV for analysis."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    header = [
        "label", "steric_a", "Ny_mesh", "beta_mesh", "lambda_max",
        "exponent_clip", "conc_floor", "dt", "max_ss_steps", "eta_steps",
        "linesearch_type", "l_ref",
        "max_eta_hat", "max_v_rhe", "n_converged", "n_attempted",
        "elapsed_s", "c_O2_final", "c_HP_final", "failure_mode",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            c = r.config
            writer.writerow([
                c.label, c.steric_a, c.Ny_mesh, c.beta_mesh, c.lambda_max,
                c.exponent_clip, c.conc_floor, c.dt, c.max_ss_steps, c.eta_steps,
                c.linesearch_type, c.l_ref,
                f"{r.max_eta_hat:.4f}", f"{r.max_v_rhe:.4f}",
                r.n_points_converged, r.n_points_attempted,
                f"{r.elapsed_seconds:.1f}",
                f"{r.final_c_O2_electrode:.6f}",
                f"{r.final_c_HP_electrode:.6f}",
                r.failure_mode,
            ])


def print_summary_table(results: List[RunResult], round_name: str) -> None:
    """Print a formatted summary table of results."""
    print(f"\n{'='*100}")
    print(f"  {round_name} Results Summary")
    print(f"{'='*100}")
    print(f"{'Label':<35} {'max eta':>8} {'V_RHE':>7} {'pts':>4} {'time':>6} "
          f"{'c_O2':>8} {'c_H+':>8}  Failure")
    print(f"{'-'*100}")

    # Sort by max_eta_hat (most negative = best)
    sorted_results = sorted(results, key=lambda r: r.max_eta_hat)

    for r in sorted_results:
        print(f"{r.config.label:<35} {r.max_eta_hat:>+8.2f} {r.max_v_rhe:>+7.3f} "
              f"{r.n_points_converged:>4d} {r.elapsed_seconds:>5.0f}s "
              f"{r.final_c_O2_electrode:>8.4f} {r.final_c_HP_electrode:>8.4f}  "
              f"{r.failure_mode[:40]}")

    print(f"{'='*100}")
    print(f"  Target: eta_hat = {ETA_TARGET:.1f} (V_RHE = -0.5 V)")
    best = sorted_results[0]
    print(f"  Best:   eta_hat = {best.max_eta_hat:.2f} "
          f"(V_RHE = {best.max_v_rhe:.3f} V) -- {best.config.label}")
    print(f"{'='*100}\n")


def plot_convergence_comparison(results: List[RunResult], out_dir: str, round_name: str) -> None:
    """Plot I-V curves from all configs overlaid for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sorted_results = sorted(results, key=lambda r: r.max_eta_hat)

    for r in sorted_results:
        if not r.eta_hat_values:
            continue
        V_rhe = [E_EQ_RHE + e * V_T for e in r.eta_hat_values]
        axes[0].plot(V_rhe, r.I_total_values, linewidth=1.0, label=r.config.label)
        axes[1].plot(V_rhe, r.I_peroxide_values, linewidth=1.0, label=r.config.label)

    for ax, title in zip(axes, ["Total Current", "Peroxide Current"]):
        ax.set_xlabel("V vs RHE (V)")
        ax.set_ylabel("Current Density (mA/cm²)")
        ax.set_title(f"{round_name}: {title}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{round_name.lower().replace(' ', '_')}_comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Comparison saved -> {png_path}")


def plot_max_eta_bar_chart(results: List[RunResult], out_dir: str, round_name: str) -> None:
    """Bar chart of max eta_hat reached by each configuration."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.4)))

    sorted_results = sorted(results, key=lambda r: r.max_eta_hat, reverse=True)
    labels = [r.config.label for r in sorted_results]
    etas = [abs(r.max_eta_hat) for r in sorted_results]

    colors = ['#2ca02c' if abs(r.max_eta_hat) >= abs(ETA_TARGET) else '#1f77b4'
              for r in sorted_results]

    bars = ax.barh(range(len(labels)), etas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|eta_hat| (V_T units)")
    ax.set_title(f"{round_name}: Maximum |eta_hat| Reached")
    ax.axvline(abs(ETA_TARGET), color="red", linestyle="--", linewidth=1.5,
               label=f"Target: |eta| = {abs(ETA_TARGET):.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{round_name.lower().replace(' ', '_')}_max_eta_bar.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Bar chart saved -> {png_path}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_round(configs: List[RunConfig], round_name: str, out_dir: str) -> List[RunResult]:
    """Execute all configs in a round and save results."""
    print(f"\n{'#'*60}")
    print(f"  {round_name}: {len(configs)} configurations")
    print(f"{'#'*60}")

    results: List[RunResult] = []

    for i, cfg in enumerate(configs):
        print(f"\n--- [{round_name}] Config {i+1}/{len(configs)}: {cfg.label} ---")
        print(f"    a={cfg.get_a_vals()}, Ny={cfg.Ny_mesh}, beta={cfg.beta_mesh}, "
              f"lam={cfg.lambda_max}, clip={cfg.exponent_clip}, "
              f"dt={cfg.dt}, ss={cfg.max_ss_steps}, steps={cfg.eta_steps}")

        try:
            result = run_single_config(cfg)
            results.append(result)
            print(f"    => max eta_hat = {result.max_eta_hat:.2f} "
                  f"(V_RHE = {result.max_v_rhe:.3f} V), "
                  f"{result.n_points_converged} points, "
                  f"{result.elapsed_seconds:.1f}s")
            if result.failure_mode != "none (reached target)":
                print(f"    => Failure: {result.failure_mode}")
        except Exception as e:
            print(f"    => EXCEPTION: {e}")
            traceback.print_exc()
            results.append(RunResult(
                config=cfg,
                max_eta_hat=0.0,
                max_v_rhe=E_EQ_RHE,
                n_points_converged=0,
                n_points_attempted=cfg.eta_steps,
                failure_mode=f"Exception: {str(e)[:200]}",
                elapsed_seconds=0.0,
                final_c_O2_electrode=float("nan"),
                final_c_HP_electrode=float("nan"),
            ))

    # Save and summarize
    csv_path = os.path.join(out_dir, f"{round_name.lower().replace(' ', '_')}_results.csv")
    save_results_csv(results, csv_path)
    print(f"\n[csv] Results saved -> {csv_path}")

    print_summary_table(results, round_name)
    plot_convergence_comparison(results, out_dir, round_name)
    plot_max_eta_bar_chart(results, out_dir, round_name)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Charged 4-species BV solver voltage range extension study"
    )
    parser.add_argument("--round", type=int, default=0,
                        help="Round to run (0=all, 1=baseline sweep, 2=targeted, 3=final push)")
    parser.add_argument("--out-dir", type=str,
                        default="StudyResults/charged_voltage_range_study",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Write agent log header
    log_path = os.path.join(out_dir, "agent_log.md")
    with open(log_path, "w") as f:
        f.write("# Charged Voltage Range Study - Agent Log\n\n")
        f.write(f"Target: eta_hat = {ETA_TARGET:.1f} (V_RHE = -0.5 V)\n\n")

    all_results: Dict[str, List[RunResult]] = {}

    if args.round == 0 or args.round == 1:
        r1_configs = round1_configs()
        r1_results = run_round(r1_configs, "Round 1", out_dir)
        all_results["Round 1"] = r1_results

        # Find best steric a from Round 1
        best_r1 = min(r1_results, key=lambda r: r.max_eta_hat)
        best_a = best_r1.config.steric_a

        with open(log_path, "a") as f:
            f.write(f"## Round 1 Results\n")
            f.write(f"- Best config: {best_r1.config.label}\n")
            f.write(f"- Max eta_hat: {best_r1.max_eta_hat:.2f}\n")
            f.write(f"- Best steric a: {best_a}\n")
            f.write(f"- Failure: {best_r1.failure_mode}\n\n")

        if args.round == 1:
            return

    if args.round == 0 or args.round == 2:
        # Use Round 1 best_a if available, else default
        if "Round 1" in all_results:
            best_r1 = min(all_results["Round 1"], key=lambda r: r.max_eta_hat)
            best_a = best_r1.config.steric_a
        else:
            best_a = 0.10  # reasonable default

        r2_configs = round2_configs(best_a)
        r2_results = run_round(r2_configs, "Round 2", out_dir)
        all_results["Round 2"] = r2_results

        best_r2 = min(r2_results, key=lambda r: r.max_eta_hat)

        with open(log_path, "a") as f:
            f.write(f"## Round 2 Results\n")
            f.write(f"- Best config: {best_r2.config.label}\n")
            f.write(f"- Max eta_hat: {best_r2.max_eta_hat:.2f}\n")
            f.write(f"- Failure: {best_r2.failure_mode}\n\n")

        if args.round == 2:
            return

    if args.round == 0 or args.round == 3:
        if "Round 2" in all_results:
            best_r2 = min(all_results["Round 2"], key=lambda r: r.max_eta_hat)
        else:
            # Create a dummy best result to generate Round 3 configs
            dummy_cfg = RunConfig(label="dummy", steric_a=0.10,
                                  Ny_mesh=500, beta_mesh=3.5,
                                  lambda_max=0.3, exponent_clip=30.0,
                                  dt=0.2, max_ss_steps=200, eta_steps=500)
            best_r2 = RunResult(
                config=dummy_cfg, max_eta_hat=-10.0, max_v_rhe=0.0,
                n_points_converged=0, n_points_attempted=0,
                failure_mode="dummy", elapsed_seconds=0,
                final_c_O2_electrode=0, final_c_HP_electrode=0,
            )

        r3_configs = round3_configs(best_r2)
        r3_results = run_round(r3_configs, "Round 3", out_dir)
        all_results["Round 3"] = r3_results

        best_r3 = min(r3_results, key=lambda r: r.max_eta_hat)

        with open(log_path, "a") as f:
            f.write(f"## Round 3 Results\n")
            f.write(f"- Best config: {best_r3.config.label}\n")
            f.write(f"- Max eta_hat: {best_r3.max_eta_hat:.2f}\n")
            f.write(f"- Failure: {best_r3.failure_mode}\n\n")

    # Final summary across all rounds
    if all_results:
        all_flat = [r for rlist in all_results.values() for r in rlist]
        best_overall = min(all_flat, key=lambda r: r.max_eta_hat)

        print(f"\n{'#'*60}")
        print(f"  FINAL SUMMARY")
        print(f"{'#'*60}")
        print(f"  Best overall: {best_overall.config.label}")
        print(f"  Max eta_hat: {best_overall.max_eta_hat:.2f} "
              f"(V_RHE = {best_overall.max_v_rhe:.3f} V)")
        print(f"  Target:       {ETA_TARGET:.1f} "
              f"(V_RHE = -0.500 V)")
        pct = (abs(best_overall.max_eta_hat) / abs(ETA_TARGET)) * 100
        print(f"  Progress:     {pct:.1f}% of target range")
        print(f"{'#'*60}\n")

        with open(log_path, "a") as f:
            f.write(f"## Final Summary\n")
            f.write(f"- Best overall: {best_overall.config.label}\n")
            f.write(f"- Max eta_hat: {best_overall.max_eta_hat:.2f}\n")
            f.write(f"- Target: {ETA_TARGET:.1f}\n")
            f.write(f"- Progress: {pct:.1f}% of target range\n")
            f.write(f"- Best config details:\n")
            c = best_overall.config
            f.write(f"  - steric_a: {c.get_a_vals()}\n")
            f.write(f"  - Ny={c.Ny_mesh}, beta={c.beta_mesh}\n")
            f.write(f"  - lambda_max={c.lambda_max}, clip={c.exponent_clip}\n")
            f.write(f"  - dt={c.dt}, max_ss={c.max_ss_steps}, steps={c.eta_steps}\n")
            f.write(f"  - linesearch={c.linesearch_type}\n")


if __name__ == "__main__":
    main()
