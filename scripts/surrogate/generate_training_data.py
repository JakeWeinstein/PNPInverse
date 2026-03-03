"""Generate training data for BV surrogate model.

Replicates the physics setup from Infer_BVMaster_charged_v7.py, then runs
the BV solver at LHS-sampled parameter points to produce training I-V curves.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/generate_training_data.py --n-samples 200 \\
        --output StudyResults/surrogate/training_data.npz

    # Resume from checkpoint:
    python scripts/surrogate/generate_training_data.py --n-samples 200 \\
        --output StudyResults/surrogate/training_data.npz --resume
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig
from Forward.bv_solver import make_graded_rectangle_mesh
from Surrogate.sampling import ParameterBounds, generate_lhs_samples, generate_multi_region_lhs_samples
from Surrogate.training import generate_training_dataset

# ---------------------------------------------------------------------------
# Physical constants (IDENTICAL to v7)
# ---------------------------------------------------------------------------
F_CONST = 96485.3329
R_GAS = 8.31446
T_REF = 298.15
V_T = R_GAS * T_REF / F_CONST
N_ELECTRONS = 2

D_O2 = 1.9e-9;  C_O2 = 0.5
D_H2O2 = 1.6e-9; C_H2O2 = 0.0
D_HP = 9.311e-9; C_HP = 0.1
D_CLO4 = 1.792e-9; C_CLO4 = 0.1

K0_PHYS = 2.4e-8; ALPHA_1 = 0.627
K0_2_PHYS = 1e-9; ALPHA_2 = 0.5

L_REF = 1.0e-4; D_REF = D_O2; C_SCALE = C_O2; K_SCALE = D_REF / L_REF

D_O2_HAT = D_O2 / D_REF; D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT = D_HP / D_REF; D_CLO4_HAT = D_CLO4 / D_REF
C_O2_HAT = C_O2 / C_SCALE; C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT = C_HP / C_SCALE; C_CLO4_HAT = C_CLO4 / C_SCALE

K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_SCALE / L_REF * 0.1

SNES_OPTS = {
    "snes_type":                 "newtonls",
    "snes_max_it":               300,
    "snes_atol":                 1e-7,
    "snes_rtol":                 1e-10,
    "snes_stol":                 1e-12,
    "snes_linesearch_type":      "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type":                  "preonly",
    "pc_type":                   "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8":         77,
    "mat_mumps_icntl_14":        80,
}


def _make_bv_solver_params(eta_hat, dt, t_end):
    """Build SolverParams for 4-species charged BV (identical to v7)."""
    params = dict(SNES_OPTS)
    bv_conv = {
        "clip_exponent": True, "exponent_clip": 50.0,
        "regularize_concentration": True, "conc_floor": 1e-12,
        "use_eta_in_bv": True,
    }
    params["bv_convergence"] = bv_conv
    params["nondim"] = {
        "enabled": True,
        "diffusivity_scale_m2_s": D_REF,
        "concentration_scale_mol_m3": C_SCALE,
        "length_scale_m": L_REF,
        "potential_scale_v": V_T,
        "kappa_inputs_are_dimensionless": True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless": True,
        "time_inputs_are_dimensionless": True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_HAT, "alpha": ALPHA_1,
                "cathodic_species": 0, "anodic_species": 1,
                "c_ref": 1.0, "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2, "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
            {
                "k0": K0_2_HAT, "alpha": ALPHA_2,
                "cathodic_species": 1, "anodic_species": None,
                "c_ref": 0.0, "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2, "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
        ],
        "k0": [K0_HAT] * 4, "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0], "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    return SolverParams.from_list([
        4, 1, dt, t_end, [0, 0, 1, -1],
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
        [0.01, 0.01, 0.01, 0.01],
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
        0.0, params,
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Generate BV surrogate training data"
    )
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of LHS parameter samples (default 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LHS sampling")
    parser.add_argument("--output", type=str,
                        default="StudyResults/surrogate/training_data.npz",
                        help="Output .npz file path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Save checkpoint every N samples")
    parser.add_argument("--min-converged", type=float, default=0.8,
                        help="Min fraction of converged points per sample (default 0.8)")
    parser.add_argument("--n-focused", type=int, default=0,
                        help="Number of focused-region LHS samples (0 = no focused region)")
    parser.add_argument("--focused-seed", type=int, default=99,
                        help="Random seed for focused-region LHS (default 99)")
    args = parser.parse_args()

    # ===================================================================
    # Union of all phase voltage grids (from v7)
    # ===================================================================
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0, -0.5,
        -1.0, -2.0, -3.0, -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])
    eta_shallow = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0, -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])

    # Union of unique voltages, sorted
    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]  # Descending (most positive first)

    # Estimate total time based on ~60s/sample
    n_total_expected = args.n_samples + args.n_focused
    est_seconds = n_total_expected * 60
    est_hours = est_seconds / 3600.0

    print(f"\n{'#'*70}", flush=True)
    print(f"  SURROGATE TRAINING DATA GENERATION", flush=True)
    print(f"  N base samples  : {args.n_samples}", flush=True)
    print(f"  N focused samples: {args.n_focused}", flush=True)
    print(f"  N total          : {n_total_expected}", flush=True)
    print(f"  Voltage points   : {len(all_eta)} (union of P1+P2+P3 grids)", flush=True)
    print(f"  Voltage range    : [{all_eta.min():.1f}, {all_eta.max():.1f}]", flush=True)
    print(f"  Output           : {args.output}", flush=True)
    print(f"  Estimated time   : ~{est_hours:.1f} hours ({est_seconds:.0f}s at ~60s/sample)", flush=True)
    print(f"  Seed (base)      : {args.seed}", flush=True)
    print(f"  Seed (focused)   : {args.focused_seed}", flush=True)
    print(f"  Checkpoint every : {args.checkpoint_interval} samples", flush=True)
    print(f"  Min converged    : {args.min_converged*100:.0f}%", flush=True)
    print(f"  Resume           : {'yes' if args.resume else 'no'}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # ===================================================================
    # Parameter bounds (centered around true values, covering search space)
    # ===================================================================
    # k0 bounds match v7's k0_lower=1e-8, k0_upper=100.0 (nondim)
    # but we use a tighter range around the true values for better coverage
    wide_bounds = ParameterBounds(
        k0_1_range=(K0_HAT * 0.01, K0_HAT * 100.0),   # 2 orders of magnitude around true
        k0_2_range=(K0_2_HAT * 0.01, K0_2_HAT * 100.0),
        alpha_1_range=(0.1, 0.9),
        alpha_2_range=(0.1, 0.9),
    )

    print(f"  Wide parameter bounds (nondim):", flush=True)
    print(f"    k0_1: [{wide_bounds.k0_1_range[0]:.4e}, {wide_bounds.k0_1_range[1]:.4e}]  "
          f"(true: {K0_HAT:.4e})", flush=True)
    print(f"    k0_2: [{wide_bounds.k0_2_range[0]:.4e}, {wide_bounds.k0_2_range[1]:.4e}]  "
          f"(true: {K0_2_HAT:.4e})", flush=True)
    print(f"    alpha_1: [{wide_bounds.alpha_1_range[0]:.2f}, {wide_bounds.alpha_1_range[1]:.2f}]  "
          f"(true: {ALPHA_1:.3f})", flush=True)
    print(f"    alpha_2: [{wide_bounds.alpha_2_range[0]:.2f}, {wide_bounds.alpha_2_range[1]:.2f}]  "
          f"(true: {ALPHA_2:.3f})", flush=True)

    if args.n_focused > 0:
        focused_bounds = ParameterBounds(
            k0_1_range=(K0_HAT * 0.1, K0_HAT * 10.0),   # 1 order of magnitude around true
            k0_2_range=(K0_2_HAT * 0.1, K0_2_HAT * 10.0),
            alpha_1_range=(0.3, 0.9),
            alpha_2_range=(0.2, 0.8),
        )
        print(f"\n  Focused parameter bounds (nondim):", flush=True)
        print(f"    k0_1: [{focused_bounds.k0_1_range[0]:.4e}, {focused_bounds.k0_1_range[1]:.4e}]", flush=True)
        print(f"    k0_2: [{focused_bounds.k0_2_range[0]:.4e}, {focused_bounds.k0_2_range[1]:.4e}]", flush=True)
        print(f"    alpha_1: [{focused_bounds.alpha_1_range[0]:.2f}, {focused_bounds.alpha_1_range[1]:.2f}]", flush=True)
        print(f"    alpha_2: [{focused_bounds.alpha_2_range[0]:.2f}, {focused_bounds.alpha_2_range[1]:.2f}]", flush=True)

        samples = generate_multi_region_lhs_samples(
            wide_bounds=wide_bounds,
            focused_bounds=focused_bounds,
            n_base=args.n_samples,
            n_focused=args.n_focused,
            seed_base=args.seed,
            seed_focused=args.focused_seed,
            log_space_k0=True,
        )
        total_samples = args.n_samples + args.n_focused
        print(f"\n  Generated {total_samples} LHS samples "
              f"({args.n_samples} wide + {args.n_focused} focused)", flush=True)
    else:
        samples = generate_lhs_samples(
            wide_bounds, args.n_samples, seed=args.seed, log_space_k0=True,
        )
        total_samples = args.n_samples
        print(f"\n  Generated {args.n_samples} LHS samples", flush=True)

    # ===================================================================
    # Solver setup (identical to v7)
    # ===================================================================
    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    observable_scale = -I_SCALE

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    print(f"  Mesh: 8x200 (beta=3.0)", flush=True)

    # ===================================================================
    # Generate training data
    # ===================================================================
    resume_path = args.output + ".checkpoint.npz" if args.resume else None

    t_start = time.time()
    result = generate_training_dataset(
        parameter_samples=samples,
        phi_applied_values=all_eta,
        base_solver_params=base_sp,
        steady=steady,
        observable_scale=observable_scale,
        mesh=mesh,
        max_eta_gap=3.0,
        output_path=args.output,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=resume_path,
        min_converged_fraction=args.min_converged,
        verbose=True,
    )
    total_time = time.time() - t_start

    print(f"\n  Total generation time: {total_time:.0f}s", flush=True)
    print(f"  Valid samples: {result['n_valid']}/{result['n_total']}", flush=True)
    print(f"  Saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
