"""Extended Voltage Range Study - Push charged BV solver beyond V_RHE=-0.5.

Prior work showed the 4-species charged PNP-BV solver (z=[0,0,1,-1])
can reach V_RHE=-0.5 (eta_hat=-46.5) with 200 continuation steps in 85s
using the baseline (no steric, a=0) or moderate steric (a<=0.15).

This study pushes toward V_RHE=-0.8 to -1.0 (eta_hat=-58 to -66) by
extending the continuation range, testing adaptive stepping, asymmetric
steric parameters, and convergence tricks.

Usage (from PNPInverse/ directory)::

    python scripts/studies/extended_voltage_range_study.py
    python scripts/studies/extended_voltage_range_study.py --group A
    python scripts/studies/extended_voltage_range_study.py --group B C

Output: StudyResults/extended_voltage_range/
"""

from __future__ import annotations

import argparse
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

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
# Physical constants (same as charged_voltage_range_study.py)
# ---------------------------------------------------------------------------

F_CONST   = 96485.3329
R_GAS     = 8.31446
T_REF     = 298.15
V_T       = R_GAS * T_REF / F_CONST   # 0.025693 V
E_EQ_RHE  = 0.695
N_ELECTRONS = 2

D_O2   = 1.9e-9
C_O2   = 0.5
D_H2O2 = 1.6e-9
C_H2O2 = 0.0
D_HP   = 9.311e-9
C_HP   = 0.1
D_CLO4 = 1.792e-9
C_CLO4 = 0.1

K0_1_PHYS  = 2.4e-8
ALPHA_1    = 0.627
K0_2_PHYS  = 1e-9
ALPHA_2    = 0.5
L_REF      = 1.0e-4

D_REF   = D_O2
C_SCALE = C_O2

D_O2_HAT   = D_O2 / D_REF
D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT   = D_HP / D_REF
D_CLO4_HAT = D_CLO4 / D_REF

C_O2_HAT   = C_O2 / C_SCALE
C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT   = C_HP / C_SCALE
C_CLO4_HAT = C_CLO4 / C_SCALE

# Extended targets
ETA_TARGET_05  = (-0.5 - E_EQ_RHE) / V_T   # ~ -46.51 (V_RHE = -0.5)
ETA_TARGET_08  = (-0.8 - E_EQ_RHE) / V_T   # ~ -58.16 (V_RHE = -0.8)
ETA_TARGET_10  = (-1.0 - E_EQ_RHE) / V_T   # ~ -65.98 (V_RHE = -1.0)

TIMEOUT_SECONDS = 600  # 10 minutes per configuration


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Configuration for one extended voltage-range test run."""
    label: str
    group: str                            # Group identifier (A, B, C, D)
    steric_a: float = 0.0
    Ny_mesh: int = 300
    beta_mesh: float = 3.0
    Nx_mesh: int = 8
    lambda_max: float = 0.5
    exponent_clip: float = 50.0
    conc_floor: float = 1e-12
    dt: float = 0.5
    max_ss_steps: int = 80
    eta_steps: int = 300
    eta_target: float = ETA_TARGET_10     # Default: push to V_RHE = -1.0
    ss_tol: float = 1e-5
    snes_max_it: int = 300
    linesearch_type: str = "l2"
    l_ref: float = L_REF
    # Per-species steric a (overrides steric_a if set)
    steric_a_vals: Optional[List[float]] = None
    # Adaptive stepping: if set, use this schedule instead of linear
    # Format: list of (fraction_of_sweep, eta_step_size_fraction)
    # e.g. [(0.7, 1.0), (1.0, 0.3)] = uniform steps for 70%, then 3x finer
    adaptive_schedule: Optional[List[Tuple[float, float]]] = None
    # Two-stage clip: first solve with clip_stage1, then refine with clip_stage2
    clip_stage1: Optional[float] = None
    clip_stage2: Optional[float] = None

    def get_a_vals(self) -> List[float]:
        if self.steric_a_vals is not None:
            return list(self.steric_a_vals)
        return [self.steric_a] * 4


@dataclass
class RunResult:
    """Result of one extended voltage-range test run."""
    config: RunConfig
    max_eta_hat: float
    max_v_rhe: float
    n_points_converged: int
    n_points_attempted: int
    failure_mode: str
    elapsed_seconds: float
    final_c_O2_electrode: float
    final_c_HP_electrode: float
    eta_hat_values: List[float] = field(default_factory=list)
    I_total_values: List[float] = field(default_factory=list)
    I_peroxide_values: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Eta path generation
# ---------------------------------------------------------------------------

def make_eta_path(config: RunConfig) -> np.ndarray:
    """Build the eta continuation path for a config.

    If adaptive_schedule is set, creates a non-uniform path with finer steps
    in specified regions. Otherwise, uniform linear spacing.
    """
    if config.adaptive_schedule is not None:
        # Build piecewise eta path
        segments = []
        prev_frac = 0.0
        total_pts = config.eta_steps
        for end_frac, step_scale in config.adaptive_schedule:
            frac_span = end_frac - prev_frac
            # Number of points in this segment (proportional to span / step_scale)
            n_seg = max(1, int(total_pts * frac_span / step_scale))
            eta_start = prev_frac * config.eta_target
            eta_end = end_frac * config.eta_target
            seg = np.linspace(eta_start, eta_end, n_seg + 1)
            if len(segments) > 0:
                seg = seg[1:]  # avoid duplicating junction point
            segments.append(seg)
            prev_frac = end_frac
        path = np.concatenate(segments)
        # Remove eta=0 if present
        if len(path) > 0 and path[0] == 0.0:
            path = path[1:]
        return path
    else:
        return np.linspace(0.0, config.eta_target, config.eta_steps + 1)[1:]


# ---------------------------------------------------------------------------
# Build SolverParams
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
# Single test run: voltage continuation sweep (extended)
# ---------------------------------------------------------------------------

def run_single_config(config: RunConfig) -> RunResult:
    """Run a voltage continuation sweep with the given configuration.

    Returns a RunResult with the maximum eta_hat reached.
    Supports adaptive stepping and two-stage exponent clipping.
    """
    i_scale = N_ELECTRONS * F_CONST * (D_REF * C_SCALE / config.l_ref) * 0.1

    eta_path = make_eta_path(config)

    t_start = time.time()

    # Build context at eta=0
    sp_eq, mesh = _make_sp(config, 0.0)
    ctx = build_context(sp_eq, mesh=mesh)
    ctx = build_forms(ctx, sp_eq)
    set_initial_conditions(ctx, sp_eq, blob=False)

    ds = fd.Measure("ds", domain=mesh)
    electrode_marker = 3

    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(F_res, ctx["U"], bcs=ctx["bcs"], J=J_form)

    snes_opts = dict(sp_eq[10])
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

    # Two-stage clip: track whether we've switched
    two_stage_active = (config.clip_stage1 is not None and config.clip_stage2 is not None)
    stage1_done = False

    for i_step, eta in enumerate(eta_path):
        # Two-stage clip: switch from stage1 to stage2 after passing V_RHE=-0.5
        if two_stage_active and not stage1_done and eta < ETA_TARGET_05:
            # We've passed V_RHE=-0.5 in stage1 clip; rebuild with stage2 clip
            bv_conv = ctx.get("bv_convergence_cfg", {})
            if hasattr(ctx.get("exponent_clip_const"), "assign"):
                ctx["exponent_clip_const"].assign(config.clip_stage2)
            stage1_done = True
            print(f"    [{config.label}] Switching to stage2 clip={config.clip_stage2} "
                  f"at eta={eta:.2f}")

        ctx["phi_applied_func"].assign(float(eta))

        ss_reached = False

        try:
            for k in range(config.max_ss_steps):
                U_scratch.assign(ctx["U"])
                solver.solve()

                delta_norm = fd.errornorm(ctx["U"], U_scratch, norm_type="L2")
                ref_norm = fd.norm(U_scratch, norm_type="L2")
                rel_change = delta_norm / max(ref_norm, 1e-14)

                ctx["U_prev"].assign(ctx["U"])

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

        # Check for NaN
        has_nan = False
        for d in ctx["U"].dat:
            if np.any(np.isnan(d.data_ro)):
                has_nan = True
                break

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

        # Check for negative concentrations
        if c_O2_elec < -0.01 or c_HP_elec < -0.01:
            failure_mode = (f"Negative concentration at eta={eta:.2f}: "
                          f"c_O2={c_O2_elec:.4f}, c_HP={c_HP_elec:.4f}")
            break

        max_eta = eta
        n_converged = i_step + 1
        eta_hat_list.append(float(eta))
        I_total_list.append(float(I_total))
        I_peroxide_list.append(float(I_peroxide))

        # Print progress every 20 steps
        if (i_step + 1) % 20 == 0:
            elapsed = time.time() - t_start
            v_rhe = E_EQ_RHE + eta * V_T
            print(f"    [{config.label}] step {i_step+1}/{len(eta_path)}  "
                  f"eta={eta:+8.2f}  V_RHE={v_rhe:+.3f}  I_tot={I_total:+.4e}  "
                  f"c_O2={c_O2_elec:.4f}  c_H+={c_HP_elec:.4f}  "
                  f"({elapsed:.0f}s)")

        # Timeout check
        elapsed = time.time() - t_start
        if elapsed > TIMEOUT_SECONDS:
            failure_mode = f"Timeout ({TIMEOUT_SECONDS}s) at eta={eta:.2f}"
            break

    else:
        failure_mode = "none (reached target)"

    elapsed = time.time() - t_start
    max_v_rhe = E_EQ_RHE + max_eta * V_T

    return RunResult(
        config=config,
        max_eta_hat=float(max_eta),
        max_v_rhe=float(max_v_rhe),
        n_points_converged=n_converged,
        n_points_attempted=len(eta_path),
        failure_mode=failure_mode,
        elapsed_seconds=elapsed,
        final_c_O2_electrode=float(c_O2_elec),
        final_c_HP_electrode=float(c_HP_elec),
        eta_hat_values=eta_hat_list,
        I_total_values=I_total_list,
        I_peroxide_values=I_peroxide_list,
    )


# ---------------------------------------------------------------------------
# Group A: Extend baseline
# ---------------------------------------------------------------------------

def group_a_configs() -> List[RunConfig]:
    """Group A: Extend the working baseline (a=0) to V_RHE=-1.0."""
    configs = []

    # 1. baseline_extended: a=0, 300 steps, eta target=-66, lambda_max=0.5
    configs.append(RunConfig(
        label="baseline_extended",
        group="A",
        steric_a=0.0,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
        lambda_max=0.5,
    ))

    # 2. baseline_extended_lam02: Same but lambda_max=0.2 (best from R1)
    configs.append(RunConfig(
        label="baseline_extended_lam02",
        group="A",
        steric_a=0.0,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
        lambda_max=0.2,
    ))

    # 3. baseline_extended_400steps: a=0, 400 steps to -66
    configs.append(RunConfig(
        label="baseline_extended_400steps",
        group="A",
        steric_a=0.0,
        eta_steps=400,
        eta_target=ETA_TARGET_10,
        lambda_max=0.5,
    ))

    return configs


# ---------------------------------------------------------------------------
# Group B: Steric at extended range
# ---------------------------------------------------------------------------

def group_b_configs() -> List[RunConfig]:
    """Group B: Moderate steric at extended voltage range."""
    configs = []

    # 4. steric_005_extended
    configs.append(RunConfig(
        label="steric_005_extended",
        group="B",
        steric_a=0.05,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
    ))

    # 5. steric_010_extended
    configs.append(RunConfig(
        label="steric_010_extended",
        group="B",
        steric_a=0.10,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
    ))

    # 6. steric_015_extended
    configs.append(RunConfig(
        label="steric_015_extended",
        group="B",
        steric_a=0.15,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
    ))

    return configs


# ---------------------------------------------------------------------------
# Group C: Asymmetric steric
# ---------------------------------------------------------------------------

def group_c_configs() -> List[RunConfig]:
    """Group C: Asymmetric steric (different a for neutrals vs ions)."""
    configs = []

    # 7. asym_steric_low: small for neutrals, moderate for ions
    configs.append(RunConfig(
        label="asym_steric_low",
        group="C",
        steric_a_vals=[0.01, 0.01, 0.05, 0.05],
        eta_steps=300,
        eta_target=ETA_TARGET_10,
    ))

    # 8. asym_steric_med
    configs.append(RunConfig(
        label="asym_steric_med",
        group="C",
        steric_a_vals=[0.02, 0.02, 0.10, 0.10],
        eta_steps=300,
        eta_target=ETA_TARGET_10,
    ))

    # 9. asym_steric_high
    configs.append(RunConfig(
        label="asym_steric_high",
        group="C",
        steric_a_vals=[0.02, 0.02, 0.15, 0.15],
        eta_steps=300,
        eta_target=ETA_TARGET_10,
    ))

    return configs


# ---------------------------------------------------------------------------
# Group D: Convergence tricks
# ---------------------------------------------------------------------------

def group_d_configs() -> List[RunConfig]:
    """Group D: Convergence tricks (adaptive dt, two-stage clip, graded steps)."""
    configs = []

    # 10. adaptive_dt: a=0.05, start with dt=0.5, finer for last 30%
    #     Implemented via adaptive_schedule: first 70% uniform, last 30% 3x finer
    configs.append(RunConfig(
        label="adaptive_dt",
        group="D",
        steric_a=0.05,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
        dt=0.2,
        max_ss_steps=120,
        adaptive_schedule=[(0.7, 1.0), (1.0, 0.33)],
    ))

    # 11. two_stage_clip: First solve with clip=20, then refine to clip=50
    configs.append(RunConfig(
        label="two_stage_clip",
        group="D",
        steric_a=0.05,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
        exponent_clip=20.0,
        clip_stage1=20.0,
        clip_stage2=50.0,
    ))

    # 12. exp_graded_steps: Exponentially graded eta steps
    #     Finer near equilibrium (where dynamics change fast), coarser at extremes
    configs.append(RunConfig(
        label="exp_graded_steps",
        group="D",
        steric_a=0.05,
        eta_steps=300,
        eta_target=ETA_TARGET_10,
        adaptive_schedule=[(0.3, 0.5), (0.7, 1.0), (1.0, 0.5)],
    ))

    return configs


# ---------------------------------------------------------------------------
# Result I/O and plotting
# ---------------------------------------------------------------------------

def save_results_csv(results: List[RunResult], csv_path: str) -> None:
    """Save all run results to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    header = [
        "label", "group", "steric_a", "steric_a_vals",
        "Ny_mesh", "beta_mesh", "lambda_max",
        "exponent_clip", "conc_floor", "dt", "max_ss_steps", "eta_steps",
        "linesearch_type", "l_ref", "eta_target",
        "max_eta_hat", "max_v_rhe", "n_converged", "n_attempted",
        "elapsed_s", "c_O2_final", "c_HP_final", "failure_mode",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            c = r.config
            writer.writerow([
                c.label, c.group, c.steric_a,
                str(c.steric_a_vals) if c.steric_a_vals else "",
                c.Ny_mesh, c.beta_mesh, c.lambda_max,
                c.exponent_clip, c.conc_floor, c.dt, c.max_ss_steps, c.eta_steps,
                c.linesearch_type, c.l_ref, f"{c.eta_target:.4f}",
                f"{r.max_eta_hat:.4f}", f"{r.max_v_rhe:.4f}",
                r.n_points_converged, r.n_points_attempted,
                f"{r.elapsed_seconds:.1f}",
                f"{r.final_c_O2_electrode:.6f}",
                f"{r.final_c_HP_electrode:.6f}",
                r.failure_mode,
            ])


def print_summary_table(results: List[RunResult], title: str) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"{'Label':<30} {'Grp':>3} {'max eta':>8} {'V_RHE':>7} {'pts':>4} "
          f"{'time':>6} {'c_O2':>8} {'c_H+':>8}  Failure")
    print(f"{'-'*110}")

    sorted_results = sorted(results, key=lambda r: r.max_eta_hat)

    for r in sorted_results:
        print(f"{r.config.label:<30} {r.config.group:>3} "
              f"{r.max_eta_hat:>+8.2f} {r.max_v_rhe:>+7.3f} "
              f"{r.n_points_converged:>4d} {r.elapsed_seconds:>5.0f}s "
              f"{r.final_c_O2_electrode:>8.4f} {r.final_c_HP_electrode:>8.4f}  "
              f"{r.failure_mode[:40]}")

    print(f"{'='*110}")
    print(f"  Targets: V_RHE=-0.5 -> eta={ETA_TARGET_05:.1f} | "
          f"V_RHE=-0.8 -> eta={ETA_TARGET_08:.1f} | "
          f"V_RHE=-1.0 -> eta={ETA_TARGET_10:.1f}")
    if results:
        best = sorted_results[0]
        print(f"  Best:    eta = {best.max_eta_hat:.2f} "
              f"(V_RHE = {best.max_v_rhe:.3f} V) -- {best.config.label}")
    print(f"{'='*110}\n")


def plot_iv_comparison(results: List[RunResult], out_dir: str, title: str) -> None:
    """Plot I-V curves from all configs overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sorted_results = sorted(results, key=lambda r: r.max_eta_hat)
    group_colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

    for r in sorted_results:
        if not r.eta_hat_values:
            continue
        V_rhe = [E_EQ_RHE + e * V_T for e in r.eta_hat_values]
        color = group_colors.get(r.config.group, "gray")
        axes[0].plot(V_rhe, r.I_total_values, linewidth=1.0,
                     label=r.config.label, color=color, alpha=0.8)
        axes[1].plot(V_rhe, r.I_peroxide_values, linewidth=1.0,
                     label=r.config.label, color=color, alpha=0.8)

    for ax, sub_title in zip(axes, ["Total Current", "Peroxide Current"]):
        ax.set_xlabel("V vs RHE (V)")
        ax.set_ylabel("Current Density (mA/cm2)")
        ax.set_title(f"{title}: {sub_title}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.5, label="E_eq")
        # Mark voltage targets
        for v_target, lbl in [(-0.5, "V=-0.5"), (-0.8, "V=-0.8"), (-1.0, "V=-1.0")]:
            ax.axvline(v_target, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    fname = title.lower().replace(" ", "_").replace(":", "")
    png_path = os.path.join(out_dir, f"{fname}_iv_comparison.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] I-V comparison saved -> {png_path}")


def plot_max_eta_bar_chart(results: List[RunResult], out_dir: str, title: str) -> None:
    """Bar chart of max eta_hat reached by each config."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.45)))

    sorted_results = sorted(results, key=lambda r: r.max_eta_hat, reverse=True)
    labels = [r.config.label for r in sorted_results]
    etas = [abs(r.max_eta_hat) for r in sorted_results]

    group_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    colors = [group_colors.get(r.config.group, "gray") for r in sorted_results]

    bars = ax.barh(range(len(labels)), etas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|eta_hat| (V_T units)")
    ax.set_title(f"{title}: Maximum |eta_hat| Reached")

    # Draw target lines
    for eta_t, lbl, ls in [
        (abs(ETA_TARGET_05), "V_RHE=-0.5", "--"),
        (abs(ETA_TARGET_08), "V_RHE=-0.8", "-."),
        (abs(ETA_TARGET_10), "V_RHE=-1.0", "-"),
    ]:
        ax.axvline(eta_t, color="red", linestyle=ls, linewidth=1.0,
                   alpha=0.7, label=f"{lbl} (|eta|={eta_t:.1f})")

    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fname = title.lower().replace(" ", "_").replace(":", "")
    png_path = os.path.join(out_dir, f"{fname}_max_eta_bar.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Bar chart saved -> {png_path}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_group(configs: List[RunConfig], group_name: str, out_dir: str) -> List[RunResult]:
    """Execute all configs in a group and save results."""
    print(f"\n{'#'*60}")
    print(f"  {group_name}: {len(configs)} configurations")
    print(f"{'#'*60}")

    results: List[RunResult] = []

    for i, cfg in enumerate(configs):
        print(f"\n--- [{group_name}] Config {i+1}/{len(configs)}: {cfg.label} ---")
        a_str = str(cfg.steric_a_vals) if cfg.steric_a_vals else f"{cfg.steric_a}"
        print(f"    a={a_str}, Ny={cfg.Ny_mesh}, beta={cfg.beta_mesh}, "
              f"lam={cfg.lambda_max}, clip={cfg.exponent_clip}, "
              f"dt={cfg.dt}, ss={cfg.max_ss_steps}, steps={cfg.eta_steps}, "
              f"eta_target={cfg.eta_target:.2f}")

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

    # Save per-group CSV
    csv_path = os.path.join(out_dir, f"group_{group_name.lower()}_results.csv")
    save_results_csv(results, csv_path)
    print(f"\n[csv] Group {group_name} results saved -> {csv_path}")

    return results


def write_agent_log(all_results: Dict[str, List[RunResult]], out_dir: str) -> None:
    """Write comprehensive agent_log.md summarizing all results."""
    log_path = os.path.join(out_dir, "agent_log.md")

    all_flat = [r for rlist in all_results.values() for r in rlist]
    if not all_flat:
        return

    best = min(all_flat, key=lambda r: r.max_eta_hat)
    reached_05 = [r for r in all_flat if r.max_eta_hat <= ETA_TARGET_05]
    reached_08 = [r for r in all_flat if r.max_eta_hat <= ETA_TARGET_08]
    reached_10 = [r for r in all_flat if r.max_eta_hat <= ETA_TARGET_10]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Extended Voltage Range Study - Agent Log\n\n")
        f.write(f"Date: 2026-02-28\n\n")

        f.write("## Targets\n\n")
        f.write(f"| Target | V_RHE | eta_hat |\n")
        f.write(f"|--------|-------|---------|\n")
        f.write(f"| Baseline | -0.5 V | {ETA_TARGET_05:.1f} |\n")
        f.write(f"| Extended | -0.8 V | {ETA_TARGET_08:.1f} |\n")
        f.write(f"| Maximum  | -1.0 V | {ETA_TARGET_10:.1f} |\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total configurations tested: {len(all_flat)}\n")
        f.write(f"- Reached V_RHE=-0.5 (eta<={ETA_TARGET_05:.1f}): "
                f"{len(reached_05)}/{len(all_flat)}\n")
        f.write(f"- Reached V_RHE=-0.8 (eta<={ETA_TARGET_08:.1f}): "
                f"{len(reached_08)}/{len(all_flat)}\n")
        f.write(f"- Reached V_RHE=-1.0 (eta<={ETA_TARGET_10:.1f}): "
                f"{len(reached_10)}/{len(all_flat)}\n")
        f.write(f"- Best overall: **{best.config.label}** "
                f"(eta={best.max_eta_hat:.2f}, V_RHE={best.max_v_rhe:.3f})\n\n")

        # Per-group results
        for group_name, results in all_results.items():
            f.write(f"## Group {group_name}\n\n")

            sorted_results = sorted(results, key=lambda r: r.max_eta_hat)
            f.write(f"| Config | eta_hat | V_RHE | Points | Time | Failure |\n")
            f.write(f"|--------|---------|-------|--------|------|--------|\n")
            for r in sorted_results:
                f.write(f"| {r.config.label} | {r.max_eta_hat:.2f} | "
                        f"{r.max_v_rhe:.3f} | {r.n_points_converged} | "
                        f"{r.elapsed_seconds:.0f}s | {r.failure_mode[:50]} |\n")
            f.write("\n")

            if sorted_results:
                grp_best = sorted_results[0]
                f.write(f"**Best in group:** {grp_best.config.label} "
                        f"(eta={grp_best.max_eta_hat:.2f})\n\n")

        # Best config details
        f.write("## Best Configuration Details\n\n")
        c = best.config
        f.write(f"- Label: {c.label}\n")
        f.write(f"- Group: {c.group}\n")
        f.write(f"- Steric a: {c.get_a_vals()}\n")
        f.write(f"- Mesh: Ny={c.Ny_mesh}, beta={c.beta_mesh}\n")
        f.write(f"- lambda_max: {c.lambda_max}\n")
        f.write(f"- Exponent clip: {c.exponent_clip}\n")
        f.write(f"- dt: {c.dt}, max_ss_steps: {c.max_ss_steps}\n")
        f.write(f"- eta_steps: {c.eta_steps}\n")
        f.write(f"- Linesearch: {c.linesearch_type}\n")
        f.write(f"- L_ref: {c.l_ref}\n")
        if c.adaptive_schedule:
            f.write(f"- Adaptive schedule: {c.adaptive_schedule}\n")
        if c.clip_stage1 is not None:
            f.write(f"- Two-stage clip: {c.clip_stage1} -> {c.clip_stage2}\n")
        f.write(f"\n")

        # Conclusions
        f.write("## Conclusions\n\n")
        pct_of_10 = (abs(best.max_eta_hat) / abs(ETA_TARGET_10)) * 100
        f.write(f"- Achieved {pct_of_10:.1f}% of the V_RHE=-1.0 target range\n")
        if best.max_eta_hat <= ETA_TARGET_10:
            f.write("- Successfully reached the full V_RHE=-1.0 target\n")
        elif best.max_eta_hat <= ETA_TARGET_08:
            f.write("- Reached V_RHE=-0.8 but not -1.0; "
                    "further work needed for full range\n")
        elif best.max_eta_hat <= ETA_TARGET_05:
            f.write("- Reached V_RHE=-0.5 but not -0.8; "
                    "extended range is challenging for charged species\n")
        else:
            f.write("- Could not reach V_RHE=-0.5; "
                    "fundamental convergence issues remain\n")

        f.write(f"- Final electrode concentrations: "
                f"c_O2={best.final_c_O2_electrode:.4f}, "
                f"c_H+={best.final_c_HP_electrode:.4f}\n")

    print(f"[log] Agent log written -> {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extended Voltage Range Study: push charged BV solver beyond V_RHE=-0.5"
    )
    parser.add_argument("--group", type=str, nargs="*", default=None,
                        help="Groups to run (A, B, C, D). Default: all groups.")
    parser.add_argument("--out-dir", type=str,
                        default="StudyResults/extended_voltage_range",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    groups_to_run = args.group if args.group else ["A", "B", "C", "D"]
    groups_to_run = [g.upper() for g in groups_to_run]

    group_generators = {
        "A": ("A - Extend Baseline", group_a_configs),
        "B": ("B - Steric Extended", group_b_configs),
        "C": ("C - Asymmetric Steric", group_c_configs),
        "D": ("D - Convergence Tricks", group_d_configs),
    }

    all_results: Dict[str, List[RunResult]] = {}

    for grp_key in groups_to_run:
        if grp_key not in group_generators:
            print(f"[warn] Unknown group '{grp_key}', skipping.")
            continue

        grp_name, gen_fn = group_generators[grp_key]
        configs = gen_fn()
        results = run_group(configs, grp_name, out_dir)
        all_results[grp_name] = results

    # Combined results
    all_flat = [r for rlist in all_results.values() for r in rlist]
    if all_flat:
        # Save combined CSV
        combined_csv = os.path.join(out_dir, "all_results.csv")
        save_results_csv(all_flat, combined_csv)
        print(f"\n[csv] Combined results saved -> {combined_csv}")

        # Print combined summary
        print_summary_table(all_flat, "Extended Voltage Range Study - All Groups")

        # Combined plots
        plot_iv_comparison(all_flat, out_dir, "Extended Voltage Range")
        plot_max_eta_bar_chart(all_flat, out_dir, "Extended Voltage Range")

        # Write agent log
        write_agent_log(all_results, out_dir)

    print(f"\nAll done. Output in: {out_dir}/")


if __name__ == "__main__":
    main()
