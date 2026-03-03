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
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, K_SCALE, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
    print_redimensionalized_results,
)
setup_firedrake_env()

# Backward-compat aliases used throughout this script
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np

from Forward.steady_state import SteadyStateConfig
from Forward.bv_solver import make_graded_rectangle_mesh
from Surrogate.sampling import ParameterBounds, generate_lhs_samples, generate_multi_region_lhs_samples
from Surrogate.training import generate_training_dataset


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
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
    )

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
