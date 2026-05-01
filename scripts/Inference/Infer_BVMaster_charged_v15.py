"""Master inference protocol v15 for BV kinetics.

Streamlined from v13:
  - Target generation uses charge continuation (z-ramping) at every voltage
  - Single surrogate phase: multistart only (no S1/S2/S3/S5)
  - P1 unchanged (eta_shallow, 10 pts)
  - P2 expanded to full all_eta range (including positive voltages)

Three-phase pipeline:
    S: MultiStart 20K (surrogate)       ~3-8s
    P1: PDE joint on SHALLOW cathodic   ~80-120s
    P2: PDE joint on FULL all_eta       ~200-300s

Usage (from PNPInverse/ directory)::

    # Full pipeline
    python scripts/Inference/Infer_BVMaster_charged_v15.py

    # Surrogate-only (fast, ~10s)
    python scripts/Inference/Infer_BVMaster_charged_v15.py --no-pde

    # Use RBF instead of NN ensemble
    python scripts/Inference/Infer_BVMaster_charged_v15.py --model-type rbf

    # Custom charge continuation steps
    python scripts/Inference/Infer_BVMaster_charged_v15.py --eta-steps 30 --charge-steps 15
"""

from __future__ import annotations

# Firedrake + PyTorch coexistence
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_redimensionalized_results,
)
setup_firedrake_env()

# Backward-compat aliases
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np

from Surrogate.io import load_surrogate
from Surrogate.multistart import MultiStartConfig, run_multistart_inference

# ---------------------------------------------------------------------------
# Fallback training-data bounds (used only if model lacks training_bounds)
# ---------------------------------------------------------------------------
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90

# ---------------------------------------------------------------------------
# Model paths relative to PNPInverse root
# ---------------------------------------------------------------------------
_SURROGATE_DIR = os.path.join("data", "surrogate_models")
_MODEL_PATHS = {
    "rbf": os.path.join(_SURROGATE_DIR, "model_rbf_baseline.pkl"),
    "pod-rbf-log": os.path.join(_SURROGATE_DIR, "model_pod_rbf_log.pkl"),
    "pod-rbf-nolog": os.path.join(_SURROGATE_DIR, "model_pod_rbf_nolog.pkl"),
}


def _load_pickle_model(path: str) -> object:
    """Load any surrogate model from a pickle file."""
    import pickle
    with open(path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "training_bounds"):
        model.training_bounds = None
    print(f"  Loaded {type(model).__name__} from: {path} (n_eta={model.n_eta})")
    if model.training_bounds is not None:
        print(f"  Training bounds: {model.training_bounds}")
    return model


def _load_model(model_type: str, args) -> object:
    """Load a surrogate model based on --model-type."""
    if model_type == "nn-ensemble":
        from Surrogate.ensemble import load_nn_ensemble
        ensemble_dir = os.path.join(args.nn_dir, args.design)
        print(f"  Loading NN ensemble from: {ensemble_dir}")
        return load_nn_ensemble(ensemble_dir, n_members=5, device="cpu")

    if model_type == "nn-single":
        from Surrogate.nn_model import NNSurrogateModel
        print(f"  Loading single NN from: {args.model}")
        return NNSurrogateModel.load(args.model, device="cpu")

    if model_type in _MODEL_PATHS:
        path = _MODEL_PATHS[model_type]
        return _load_pickle_model(path)

    # Fallback: treat --model as a pickle path
    return _load_pickle_model(args.model)


def _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr):
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return k0_err, alpha_err


def _print_phase_result(name, k0, alpha, true_k0_arr, true_alpha_arr, loss, elapsed):
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


def _log_ensemble_uncertainty(surrogate, k0, alpha, phase_name):
    """If surrogate supports predict_with_uncertainty, log mean std."""
    if not hasattr(surrogate, "predict_with_uncertainty"):
        return
    unc = surrogate.predict_with_uncertainty(k0[0], k0[1], alpha[0], alpha[1])
    cd_std_mean = float(np.mean(unc["current_density_std"]))
    pc_std_mean = float(np.mean(unc["peroxide_current_std"]))
    print(f"    [{phase_name}] Ensemble uncertainty: mean CD std={cd_std_mean:.4e}, "
          f"mean PC std={pc_std_mean:.4e}")


_TARGET_CACHE_DIR = os.path.join("StudyResults", "target_cache")


def _target_cache_path(phi_applied_values, observable_scale, eta_steps, charge_steps):
    """Build a cache file path based on ALL inputs that affect the PDE solution.

    Includes method=charge_continuation to avoid cache collision with v13.
    """
    import hashlib
    species = FOUR_SPECIES_CHARGED
    parts = [
        phi_applied_values.tobytes(),
        str(observable_scale).encode(),
        # True kinetic parameters (the 4 being inferred)
        f"k0={K0_HAT},{K0_2_HAT}".encode(),
        f"alpha={ALPHA_1},{ALPHA_2}".encode(),
        # Species transport config
        f"n_species={species.n_species}".encode(),
        f"z={species.z_vals}".encode(),
        f"D={species.d_vals_hat}".encode(),
        f"a={species.a_vals_hat}".encode(),
        f"c0={species.c0_vals_hat}".encode(),
        f"stoi_r1={species.stoichiometry_r1}".encode(),
        f"stoi_r2={species.stoichiometry_r2}".encode(),
        # Mesh settings
        f"Nx=8,Ny=200,beta=3.0".encode(),
        # Charge continuation method marker + parameters
        f"method=charge_continuation".encode(),
        f"eta_steps={eta_steps},charge_steps={charge_steps}".encode(),
    ]
    key = hashlib.sha256(b"|".join(parts)).hexdigest()[:16]
    return os.path.join(_TARGET_CACHE_DIR, f"clean_targets_{key}.npz")


def _solve_clean_targets_charge_cont(phi_applied_values, observable_scale,
                                     eta_steps, charge_steps):
    """Generate noise-free target I-V curves using charge continuation at each eta.

    For each voltage in phi_applied_values:
      1. Build solver_params with true kinetic params at that eta
      2. Call solve_bv_with_charge_continuation(..., return_ctx=True)
      3. Extract CD and PC observables from the returned ctx
    """
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver import solve_bv_with_charge_continuation
    from FluxCurve.bv_observables import _build_bv_observable_form

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps

    # Shared mesh to avoid re-meshing for every voltage point
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    n_eta = len(phi_applied_values)
    clean_cd = np.zeros(n_eta, dtype=float)
    clean_pc = np.zeros(n_eta, dtype=float)

    print(f"  [charge-cont targets] Solving {n_eta} voltage points "
          f"(eta_steps={eta_steps}, charge_steps={charge_steps})")

    # Disable adjoint annotation so charge-continuation solves don't
    # pollute the tape used later by PDE inference phases.
    with adj.stop_annotating():
        for i, eta in enumerate(phi_applied_values):
            t0 = time.time()
            sp = make_bv_solver_params(
                eta_hat=float(eta), dt=dt, t_end=t_end,
                species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
            )

            ctx = solve_bv_with_charge_continuation(
                sp,
                eta_target=float(eta),
                eta_steps=eta_steps,
                charge_steps=charge_steps,
                print_interval=max(eta_steps, 1),
                mesh=mesh,
                return_ctx=True,
            )

            # Extract observables
            form_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None, scale=observable_scale,
            )
            cd_val = float(fd.assemble(form_cd))

            form_pc = _build_bv_observable_form(
                ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale,
            )
            pc_val = float(fd.assemble(form_pc))

            clean_cd[i] = cd_val
            clean_pc[i] = pc_val

            elapsed = time.time() - t0
            print(f"    eta={eta:+7.2f}: CD={cd_val:+.6e}, PC={pc_val:+.6e} ({elapsed:.1f}s)")

    return clean_cd, clean_pc


def _generate_targets_with_charge_cont(phi_applied_values, observable_scale,
                                       noise_percent, noise_seed,
                                       eta_steps, charge_steps):
    """Generate target I-V curves using charge continuation at true parameters.

    Caches the clean (noise-free) solution so repeated runs skip the solve.
    """
    from Forward.steady_state import add_percent_noise

    cache_path = _target_cache_path(phi_applied_values, observable_scale,
                                    eta_steps, charge_steps)

    # Try loading cached clean targets
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            clean_cd = cached["current_density"]
            clean_pc = cached["peroxide_current"]
            cached_eta = cached["phi_applied"]
            if (clean_cd.shape[0] == len(phi_applied_values)
                    and np.allclose(cached_eta, phi_applied_values)):
                print(f"  Using cached clean targets from {cache_path}")
            else:
                print(f"  Cache shape mismatch, regenerating...")
                clean_cd, clean_pc = _solve_clean_targets_charge_cont(
                    phi_applied_values, observable_scale, eta_steps, charge_steps)
                _save_clean_targets(cache_path, phi_applied_values, clean_cd, clean_pc)
        except Exception as exc:
            print(f"  Cache load failed ({exc}), regenerating...")
            clean_cd, clean_pc = _solve_clean_targets_charge_cont(
                phi_applied_values, observable_scale, eta_steps, charge_steps)
            _save_clean_targets(cache_path, phi_applied_values, clean_cd, clean_pc)
    else:
        clean_cd, clean_pc = _solve_clean_targets_charge_cont(
            phi_applied_values, observable_scale, eta_steps, charge_steps)
        _save_clean_targets(cache_path, phi_applied_values, clean_cd, clean_pc)

    # Apply noise
    results = {}
    if noise_percent > 0:
        results["current_density"] = add_percent_noise(clean_cd, noise_percent, seed=noise_seed)
        results["peroxide_current"] = add_percent_noise(clean_pc, noise_percent, seed=noise_seed + 1)
    else:
        results["current_density"] = clean_cd.copy()
        results["peroxide_current"] = clean_pc.copy()

    return results


def _save_clean_targets(cache_path, phi_applied_values, clean_cd, clean_pc):
    """Save clean targets to disk cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path,
             current_density=clean_cd,
             peroxide_current=clean_pc,
             phi_applied=phi_applied_values)
    print(f"  Saved clean targets to {cache_path}")


def _subset_targets(target_cd, target_pc, all_eta, subset_eta):
    """Extract target values for a subset of voltages from the full grid."""
    idx = []
    for eta in subset_eta:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(matches[0])
    idx = np.array(idx, dtype=int)
    return target_cd[idx], target_pc[idx]


def _make_parallel_config(base_sp, steady, *, mesh_Nx, mesh_Ny, mesh_beta,
                          observable_mode, observable_reaction_index,
                          observable_scale, control_mode, n_controls,
                          blob_initial_condition=False, fail_penalty=1e9,
                          secondary_observable_mode=None,
                          secondary_observable_reaction_index=None,
                          secondary_observable_scale=None):
    """Build a BVParallelPointConfig from solver params."""
    from FluxCurve.bv_point_solve import (
        _WARMSTART_MAX_STEPS,
        _SER_GROWTH_CAP,
        _SER_SHRINK,
        _SER_DT_MAX_RATIO,
    )
    from FluxCurve.bv_parallel import BVParallelPointConfig

    return BVParallelPointConfig(
        base_solver_params=list(base_sp),
        ss_relative_tolerance=float(steady.relative_tolerance),
        ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
        ss_consecutive_steps=int(steady.consecutive_steps),
        ss_max_steps=int(steady.max_steps),
        mesh_Nx=mesh_Nx,
        mesh_Ny=mesh_Ny,
        mesh_beta=mesh_beta,
        blob_initial_condition=blob_initial_condition,
        fail_penalty=fail_penalty,
        warmstart_max_steps=_WARMSTART_MAX_STEPS,
        observable_mode=observable_mode,
        observable_reaction_index=observable_reaction_index,
        observable_scale=observable_scale,
        control_mode=control_mode,
        n_controls=n_controls,
        ser_growth_cap=_SER_GROWTH_CAP,
        ser_shrink=_SER_SHRINK,
        ser_dt_max_ratio=_SER_DT_MAX_RATIO,
        secondary_observable_mode=secondary_observable_mode,
        secondary_observable_reaction_index=secondary_observable_reaction_index,
        secondary_observable_scale=secondary_observable_scale,
    )


def _get_training_bounds(surrogate):
    """Extract training bounds from surrogate, falling back to defaults."""
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        K0_1_LO = tb["k0_1"][0]
        K0_1_HI = tb["k0_1"][1]
        K0_2_LO = tb["k0_2"][0]
        K0_2_HI = tb["k0_2"][1]
        ALPHA_LO = tb["alpha_1"][0]
        ALPHA_HI = tb["alpha_1"][1]
        ALPHA_LO = min(ALPHA_LO, tb["alpha_2"][0])
        ALPHA_HI = max(ALPHA_HI, tb["alpha_2"][1])
        return K0_1_LO, K0_1_HI, K0_2_LO, K0_2_HI, ALPHA_LO, ALPHA_HI, True
    return (K0_1_TRAIN_LO_DEFAULT, K0_1_TRAIN_HI_DEFAULT,
            K0_2_TRAIN_LO_DEFAULT, K0_2_TRAIN_HI_DEFAULT,
            ALPHA_TRAIN_LO_DEFAULT, ALPHA_TRAIN_HI_DEFAULT, False)


def _extend_voltage_cache_for_p2(
    all_eta: np.ndarray,
    warm_k0: np.ndarray,
    warm_alpha: np.ndarray,
    base_sp,
    *,
    eta_steps: int = 20,
    charge_steps: int = 10,
) -> int:
    """Pre-solve all P2 voltage points via charge continuation and populate cache.

    After P1 completes and ``_clear_caches()`` wipes the old cache, this
    function re-solves every voltage at P1's best parameters so that P2's
    first optimizer evaluation can use the fast-path (cached ICs) instead
    of a cold sequential sweep.

    Returns the number of fully-converged (z=1.0) cached points.
    """
    import warnings
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver import solve_bv_with_charge_continuation
    from FluxCurve.bv_point_solve import populate_cache_entry, mark_cache_populated_if_complete

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    n_points = len(all_eta)
    n_full = 0
    partial_points = []

    with adj.stop_annotating():
        for idx, eta in enumerate(all_eta):
            t0 = time.time()
            sp = make_bv_solver_params(
                eta_hat=float(eta), dt=dt, t_end=t_end,
                species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
                k0_hat_r1=float(warm_k0[0]),
                k0_hat_r2=float(warm_k0[1]),
                alpha_r1=float(warm_alpha[0]),
                alpha_r2=float(warm_alpha[1]),
            )

            ctx = solve_bv_with_charge_continuation(
                sp,
                eta_target=float(eta),
                eta_steps=eta_steps,
                charge_steps=charge_steps,
                print_interval=max(eta_steps, 1),
                mesh=mesh,
                return_ctx=True,
            )

            # Extract achieved z_factor from z_consts
            z_consts = ctx["z_consts"]
            z_nominal = [float(sp[4][i]) if isinstance(sp[4], (list, tuple)) else float(sp[4])
                         for i in range(int(sp[0]))]
            achieved_z = 1.0
            for i, zc in enumerate(z_consts):
                zc_val = float(zc)
                if abs(z_nominal[i]) > 1e-14:
                    achieved_z = min(achieved_z, abs(zc_val / z_nominal[i]))

            U_data = tuple(d.data_ro.copy() for d in ctx["U"].dat)
            mesh_dof_count = ctx["U"].function_space().dim()
            populate_cache_entry(idx, U_data, mesh_dof_count)

            elapsed = time.time() - t0
            if achieved_z >= 1.0 - 1e-6:
                n_full += 1
                print(f"    [ext] eta={eta:+7.2f}: cached (z=1.000, {elapsed:.1f}s)")
            else:
                partial_points.append((eta, achieved_z))
                warnings.warn(
                    f"Extension: eta={eta:+.2f} cached at z={achieved_z:.3f} "
                    f"(incomplete charge coupling)"
                )
                print(f"    [ext] eta={eta:+7.2f}: cached (z={achieved_z:.3f}, {elapsed:.1f}s)")

    mark_cache_populated_if_complete(n_points)

    if partial_points:
        pts_str = ", ".join(f"eta={e:+.2f} z={z:.3f}" for e, z in partial_points)
        print(f"  WARNING: partial-z points: {pts_str}")

    return n_full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BV Master Inference v15 (Charge Continuation + Multistart + PDE)"
    )
    # Surrogate model selection
    parser.add_argument("--model", type=str,
                        default=os.path.join(_SURROGATE_DIR, "model_rbf_baseline.pkl"),
                        help="Path to surrogate model .pkl (for rbf/nn-single)")
    parser.add_argument("--model-type", type=str, default="nn-ensemble",
                        choices=["nn-ensemble", "rbf", "pod-rbf-log", "pod-rbf-nolog", "nn-single"],
                        help="Surrogate model type (default: nn-ensemble)")
    parser.add_argument("--design", type=str, default="D3-deeper",
                        help="NN ensemble design variant (default: D3-deeper)")
    parser.add_argument("--nn-dir", type=str,
                        default=os.path.join(_SURROGATE_DIR, "nn_ensemble"),
                        help="Base directory for NN ensembles")

    # Multistart controls
    parser.add_argument("--multistart-n", type=int, default=20000,
                        help="MultiStart grid size (default: 20000)")
    parser.add_argument("--multistart-k", type=int, default=20,
                        help="MultiStart top candidates to polish (default: 20)")

    # Charge continuation controls
    parser.add_argument("--eta-steps", type=int, default=20,
                        help="Voltage continuation steps for target generation (default: 20)")
    parser.add_argument("--charge-steps", type=int, default=10,
                        help="Charge ramping steps for target generation (default: 10)")

    # PDE phase controls
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip PDE phases P1-P2 (surrogate-only)")
    parser.add_argument("--pde-cold-start", action="store_true",
                        help="Skip surrogate phases; run PDE from cold initial guesses")
    parser.add_argument("--skip-p1", action="store_true",
                        help="Skip P1 (shallow PDE); P2 warm-starts from surrogate best")
    parser.add_argument("--pde-p1-maxiter", type=int, default=25,
                        help="P1 (shallow) max L-BFGS-B iterations")
    parser.add_argument("--pde-p2-maxiter", type=int, default=20,
                        help="P2 (full range) max L-BFGS-B iterations")
    parser.add_argument("--pde-secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current for PDE phases")
    parser.add_argument("--max-anodic-eta", type=float, default=5.0,
                        help="Max positive eta for PDE phases (default: 5.0; try 4.0 if +5 fails)")
    parser.add_argument("--workers", type=int, default=0,
                        help="PDE parallel workers (0=auto)")

    # Target/noise
    parser.add_argument("--noise-percent", type=float, default=2.0,
                        help="Target noise level (0.0 for noise-free)")
    parser.add_argument("--noise-seed", type=int, default=20260226,
                        help="Noise seed")

    # Surrogate-phase secondary weight
    parser.add_argument("--secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current for surrogate phases")
    args = parser.parse_args()

    if args.pde_cold_start and args.no_pde:
        parser.error("--pde-cold-start and --no-pde are mutually exclusive")

    # ===================================================================
    # Voltage grids
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

    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]

    # PDE-phase eta: optionally cap the max anodic voltage
    pde_eta = all_eta[all_eta <= args.max_anodic_eta + 1e-10]
    if len(pde_eta) < len(all_eta):
        dropped = all_eta[all_eta > args.max_anodic_eta + 1e-10]
        print(f"  [v15] --max-anodic-eta={args.max_anodic_eta}: "
              f"dropping {len(dropped)} points > {args.max_anodic_eta} from PDE phases: {dropped}")

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    initial_k0_guess = [0.0015, 6e-05]
    initial_alpha_guess = [0.6, 0.48]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "master_inference_v15")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}
    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v15")
    print(f"  (Charge Continuation + Multistart + PDE Full Range)")
    print(f"{'#'*70}")
    print(f"  Model type:          {args.model_type}")
    if args.model_type == "nn-ensemble":
        print(f"  Design:              {args.design}")
    print(f"  True k0:             {true_k0}")
    print(f"  True alpha:          {true_alpha}")
    print(f"  Initial k0 guess:    {initial_k0_guess}")
    print(f"  Initial alpha guess: {initial_alpha_guess}")
    if args.pde_cold_start:
        print(f"  MODE:                PDE COLD-START (no surrogate)")
    print(f"  Secondary weight:    {args.secondary_weight}")
    print(f"  PDE sec. weight:     {args.pde_secondary_weight}")
    print(f"  Noise: {args.noise_percent}% (seed={args.noise_seed})")
    print(f"  Charge cont:         eta_steps={args.eta_steps}, charge_steps={args.charge_steps}")
    if not args.no_pde:
        print(f"  PDE P1 maxiter:      {args.pde_p1_maxiter}")
        print(f"  PDE P2 maxiter:      {args.pde_p2_maxiter}")
    print(f"  all_eta: {len(all_eta)} pts [{all_eta.min():.1f}, {all_eta.max():.1f}]")
    print(f"  pde_eta: {len(pde_eta)} pts [{pde_eta.min():.1f}, {pde_eta.max():.1f}]")
    print(f"  eta_shallow: {len(eta_shallow)} pts [{eta_shallow.min():.1f}, {eta_shallow.max():.1f}]")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Generate targets using charge continuation
    # ===================================================================
    print(f"\nGenerating target I-V curves with charge continuation at true parameters...")
    t_target = time.time()
    targets = _generate_targets_with_charge_cont(
        all_eta, observable_scale, args.noise_percent, args.noise_seed,
        args.eta_steps, args.charge_steps,
    )
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    # ===================================================================
    # PDE cold-start shortcut: skip surrogate, use initial guesses
    # ===================================================================
    if args.pde_cold_start:
        print(f"\n  --pde-cold-start: skipping surrogate phases")
        print(f"  PDE will warm-start from cold initial guesses:")
        print(f"    k0   = {initial_k0_guess}")
        print(f"    alpha= {initial_alpha_guess}")

        surrogate_time = 0.0
        surr_best_k0 = np.asarray(initial_k0_guess)
        surr_best_alpha = np.asarray(initial_alpha_guess)
        surr_best_loss = float("inf")
        surrogate = None

    else:
        # ===================================================================
        # Load surrogate model
        # ===================================================================
        print(f"Loading surrogate model ({args.model_type})...")
        surrogate = _load_model(args.model_type, args)
        surrogate_eta = surrogate.phi_applied
        print(f"  Surrogate voltage points: {surrogate.n_eta}")
        print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

        if surrogate.n_eta != len(all_eta):
            raise ValueError(
                f"Surrogate has {surrogate.n_eta} voltage points but all_eta has {len(all_eta)}. "
                f"Ensure the surrogate was trained on the same voltage grid."
            )

    # Subset targets for PDE phases (may exclude high anodic eta)
    pde_idx = np.array([i for i, e in enumerate(all_eta) if e <= args.max_anodic_eta + 1e-10])
    pde_target_cd = target_cd_full[pde_idx]
    pde_target_pc = target_pc_full[pde_idx]

    if not args.pde_cold_start:
        # ===================================================================
        # PHASE S: MultiStart surrogate optimization (only surrogate phase)
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"  PHASE S: MultiStart surrogate optimization ({args.model_type})")
        print(f"{'='*70}")

        (K0_1_LO, K0_1_HI, K0_2_LO, K0_2_HI,
         ALPHA_LO, ALPHA_HI, from_model) = _get_training_bounds(surrogate)

        if from_model:
            print(f"  Using training bounds FROM MODEL")
        else:
            print(f"  WARNING: model lacks training_bounds, using defaults")

        print(f"    k0_1 log10: [{np.log10(max(K0_1_LO, 1e-30)):.2f}, {np.log10(K0_1_HI):.2f}]")
        print(f"    k0_2 log10: [{np.log10(max(K0_2_LO, 1e-30)):.2f}, {np.log10(K0_2_HI):.2f}]")
        print(f"    alpha:      [{ALPHA_LO:.4f}, {ALPHA_HI:.4f}]")

        # Compute shallow subset indices for multistart
        shallow_idx = []
        for eta in eta_shallow:
            matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
            if len(matches) > 0:
                shallow_idx.append(matches[0])
        shallow_idx = np.array(shallow_idx, dtype=int)

        print(f"    n_grid={args.multistart_n}, n_top={args.multistart_k}")
        t_s = time.time()

        ms_config = MultiStartConfig(
            n_grid=args.multistart_n,
            n_top_candidates=args.multistart_k,
            polish_maxiter=60,
            secondary_weight=args.secondary_weight,
            fd_step=1e-5,
            use_shallow_subset=True,
            seed=42,
            verbose=True,
        )

        ms_result = run_multistart_inference(
            surrogate=surrogate,
            target_cd=target_cd_full,
            target_pc=target_pc_full,
            bounds_k0_1=(K0_1_LO, K0_1_HI),
            bounds_k0_2=(K0_2_LO, K0_2_HI),
            bounds_alpha=(ALPHA_LO, ALPHA_HI),
            config=ms_config,
            subset_idx=shallow_idx,
        )

        surr_best_k0 = np.array([ms_result.best_k0_1, ms_result.best_k0_2])
        surr_best_alpha = np.array([ms_result.best_alpha_1, ms_result.best_alpha_2])
        surr_best_loss = ms_result.best_loss
        s_time = time.time() - t_s

        _print_phase_result("S (multistart)", surr_best_k0, surr_best_alpha,
                            true_k0_arr, true_alpha_arr, surr_best_loss, s_time)
        _log_ensemble_uncertainty(surrogate, surr_best_k0, surr_best_alpha, "S")

        phase_results["S (multistart)"] = {
            "k0": surr_best_k0.tolist(), "alpha": surr_best_alpha.tolist(),
            "loss": surr_best_loss, "time": s_time,
        }

        surrogate_time = time.time() - t_total_start - t_target_elapsed

        print(f"\n  Surrogate best: S (multistart)")
        print(f"    k0   = {surr_best_k0.tolist()}")
        print(f"    alpha= {surr_best_alpha.tolist()}")

    # ===================================================================
    # PDE PHASES P1-P2 (warm-started by multistart result)
    # ===================================================================
    pde_secondary_weight = args.pde_secondary_weight

    if not args.no_pde:
        from Forward.steady_state import SteadyStateConfig
        from FluxCurve import (
            BVFluxCurveInferenceRequest,
            run_bv_multi_observable_flux_curve_inference,
        )
        from FluxCurve.bv_point_solve import (
            _clear_caches,
            set_parallel_pool,
            close_parallel_pool,
        )
        from FluxCurve.bv_parallel import BVPointSolvePool

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

        recovery = make_recovery_config(max_it_cap=600)

        n_pde_workers = args.workers
        if n_pde_workers <= 0:
            if args.skip_p1:
                max_phase_points = len(pde_eta)
            else:
                max_phase_points = max(len(eta_shallow), len(pde_eta))
            n_pde_workers = min(max_phase_points, max(1, (os.cpu_count() or 4) - 1))

        n_joint_controls = 4

        shared_config = _make_parallel_config(
            base_sp, steady,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="joint",
            n_controls=n_joint_controls,
            secondary_observable_mode="peroxide_current",
            secondary_observable_reaction_index=None,
            secondary_observable_scale=observable_scale,
        )
        shared_pool = BVPointSolvePool(shared_config, n_workers=n_pde_workers)
        set_parallel_pool(shared_pool)
        print(f"\n  [v15] Shared parallel pool: {n_pde_workers} workers")

        # --- Extract PDE target subsets ---
        p1_target_cd, p1_target_pc = _subset_targets(
            pde_target_cd, pde_target_pc, pde_eta, eta_shallow,
        )
        # P2 uses pde_eta (may exclude high anodic voltages)

        # --- P1: PDE shallow (skippable) ---
        warm_source_label = "cold-start" if args.pde_cold_start else "S (multistart)"
        if args.skip_p1:
            print(f"\n  [v15] --skip-p1: skipping P1, P2 warm-starts from {warm_source_label}")
            p2_warm_k0 = surr_best_k0
            p2_warm_alpha = surr_best_alpha
            p2_warm_source = warm_source_label

            # Voltage extension: pre-solve all P2 points at surrogate best params
            _clear_caches()
            t_ext = time.time()
            n_cached = _extend_voltage_cache_for_p2(
                pde_eta, p2_warm_k0, p2_warm_alpha, base_sp,
                eta_steps=args.eta_steps, charge_steps=args.charge_steps,
            )
            ext_time = time.time() - t_ext
            print(f"  [v15] Extension: {n_cached}/{len(pde_eta)} cached ({ext_time:.1f}s)")
        else:
            print(f"\n{'='*70}")
            print(f"  P1: PDE joint on SHALLOW")
            print(f"  Warm-start from: {warm_source_label}")
            print(f"  k0={surr_best_k0.tolist()}, alpha={surr_best_alpha.tolist()}")
            print(f"  {len(eta_shallow)}-pt shallow [{eta_shallow.min():.1f}, {eta_shallow.max():.1f}]")
            print(f"  maxiter={args.pde_p1_maxiter}, secondary_weight={pde_secondary_weight}")
            print(f"{'='*70}")
            t_p1 = time.time()

            _clear_caches()
            p1_dir = os.path.join(base_output, "P1_pde_shallow")

            request_p1 = BVFluxCurveInferenceRequest(
                base_solver_params=base_sp,
                steady=steady,
                true_k0=true_k0,
                initial_guess=surr_best_k0.tolist(),
                phi_applied_values=eta_shallow.tolist(),
                target_csv_path=os.path.join(p1_dir, "target_primary.csv"),
                output_dir=p1_dir,
                regenerate_target=True,
                target_noise_percent=args.noise_percent,
                target_seed=args.noise_seed,
                observable_mode="current_density",
                current_density_scale=observable_scale,
                observable_label="current density (mA/cm2)",
                observable_title=f"P1: PDE shallow (warm from {warm_source_label})",
                secondary_observable_mode="peroxide_current",
                secondary_observable_weight=pde_secondary_weight,
                secondary_current_density_scale=observable_scale,
                secondary_target_csv_path=os.path.join(p1_dir, "target_peroxide.csv"),
                control_mode="joint",
                true_alpha=true_alpha,
                initial_alpha_guess=surr_best_alpha.tolist(),
                alpha_lower=0.05, alpha_upper=0.95,
                k0_lower=1e-8, k0_upper=100.0,
                log_space=True,
                mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
                max_eta_gap=3.0,
                optimizer_method="L-BFGS-B",
                optimizer_options={
                    "maxiter": args.pde_p1_maxiter,
                    "ftol": 1e-8,
                    "gtol": 5e-6,
                    "disp": True,
                },
                max_iters=args.pde_p1_maxiter,
                live_plot=False,
                forward_recovery=recovery,
                parallel_fast_path=True,
                parallel_workers=n_pde_workers,
            )

            result_p1 = run_bv_multi_observable_flux_curve_inference(
                request_p1,
                precomputed_targets={
                    "primary": p1_target_cd,
                    "secondary": p1_target_pc,
                },
            )
            p1_k0 = np.asarray(result_p1["best_k0"])
            p1_alpha = np.asarray(result_p1["best_alpha"])
            p1_loss = float(result_p1["best_loss"])
            p1_time = time.time() - t_p1

            _print_phase_result("P1 (PDE shallow)", p1_k0, p1_alpha,
                                true_k0_arr, true_alpha_arr, p1_loss, p1_time)
            phase_results["P1 (PDE shallow)"] = {
                "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
                "loss": p1_loss, "time": p1_time,
            }

            _clear_caches()
            p2_warm_k0 = p1_k0
            p2_warm_alpha = p1_alpha
            p2_warm_source = "P1"

            # Voltage extension: pre-solve all P2 points at P1's best params
            t_ext = time.time()
            n_cached = _extend_voltage_cache_for_p2(
                pde_eta, p2_warm_k0, p2_warm_alpha, base_sp,
                eta_steps=args.eta_steps, charge_steps=args.charge_steps,
            )
            ext_time = time.time() - t_ext
            print(f"  [v15] Extension: {n_cached}/{len(pde_eta)} cached ({ext_time:.1f}s)")

        # --- P2: PDE on pde_eta range ---
        print(f"\n{'='*70}")
        print(f"  P2: PDE joint on pde_eta range")
        print(f"  Warm-start from {p2_warm_source}: k0={p2_warm_k0.tolist()}, alpha={p2_warm_alpha.tolist()}")
        print(f"  {len(pde_eta)}-pt [{pde_eta.min():.1f}, {pde_eta.max():.1f}]")
        print(f"  maxiter={args.pde_p2_maxiter}, secondary_weight={pde_secondary_weight}")
        print(f"{'='*70}")
        t_p2 = time.time()

        p2_dir = os.path.join(base_output, "P2_pde_full_range")

        request_p2 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=p2_warm_k0.tolist(),
            phi_applied_values=pde_eta.tolist(),
            target_csv_path=os.path.join(p2_dir, "target_primary.csv"),
            output_dir=p2_dir,
            regenerate_target=True,
            target_noise_percent=args.noise_percent,
            target_seed=args.noise_seed,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"P2: PDE full range (warm from {p2_warm_source})",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=pde_secondary_weight,
            secondary_current_density_scale=observable_scale,
            secondary_target_csv_path=os.path.join(p2_dir, "target_peroxide.csv"),
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=p2_warm_alpha.tolist(),
            alpha_lower=0.05, alpha_upper=0.95,
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={
                "maxiter": args.pde_p2_maxiter,
                "ftol": 1e-8,
                "gtol": 5e-6,
                "disp": True,
            },
            max_iters=args.pde_p2_maxiter,
            live_plot=False,
            forward_recovery=recovery,
            parallel_fast_path=True,
            parallel_workers=n_pde_workers,
        )

        result_p2 = run_bv_multi_observable_flux_curve_inference(
            request_p2,
            precomputed_targets={
                "primary": pde_target_cd,
                "secondary": pde_target_pc,
            },
        )
        p2_k0 = np.asarray(result_p2["best_k0"])
        p2_alpha = np.asarray(result_p2["best_alpha"])
        p2_loss = float(result_p2["best_loss"])
        p2_time = time.time() - t_p2

        close_parallel_pool()
        _clear_caches()
        print(f"  [v15] Shared parallel pool closed")

        _print_phase_result("P2 (PDE full range)", p2_k0, p2_alpha,
                            true_k0_arr, true_alpha_arr, p2_loss, p2_time)
        phase_results["P2 (PDE full range)"] = {
            "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
            "loss": p2_loss, "time": p2_time,
        }

        # Best-of selection
        if args.skip_p1:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "P2 (PDE full range)"
            p2_k0_err, p2_alpha_err = _compute_errors(p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
            best_max_err = max(p2_k0_err.max(), p2_alpha_err.max())
            print(f"\n  [--skip-p1] Best = P2 (max err = {best_max_err*100:.2f}%)")
        else:
            p1_k0_err, p1_alpha_err = _compute_errors(p1_k0, p1_alpha, true_k0_arr, true_alpha_arr)
            p2_k0_err, p2_alpha_err = _compute_errors(p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)

            p1_max_err = max(p1_k0_err.max(), p1_alpha_err.max())
            p2_max_err = max(p2_k0_err.max(), p2_alpha_err.max())

            if p2_max_err <= p1_max_err:
                best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
                best_source = "P2 (PDE full range)"
            else:
                best_k0, best_alpha = p1_k0.copy(), p1_alpha.copy()
                best_source = "P1 (PDE shallow)"

            best_max_err = min(p1_max_err, p2_max_err)
            print(f"\n  P1 vs P2: best is {best_source} (max err = {best_max_err*100:.2f}%)")
    else:
        best_k0, best_alpha = surr_best_k0.copy(), surr_best_alpha.copy()
        best_source = "S (multistart)"

    total_time = time.time() - t_total_start
    pde_time = (total_time - t_target_elapsed - surrogate_time) if not args.no_pde else 0.0

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*95}")
    print(f"  MASTER INFERENCE v15 SUMMARY")
    print(f"  (Charge Continuation + Multistart + PDE Full Range)")
    print(f"{'#'*95}")
    print(f"  Model type:        {args.model_type}")
    print(f"  True k0:           {true_k0}")
    print(f"  True alpha:        {true_alpha}")
    print(f"  K0_HAT = {K0_HAT:.6e},  K0_2_HAT = {K0_2_HAT:.6e}")
    print()

    header = (f"{'Phase':<40} | {'k0_1 err':>10} {'k0_2 err':>10} "
              f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12} | {'time':>6}")
    print(header)
    print(f"{'-'*110}")

    for name, ph in phase_results.items():
        k0_err, alpha_err = _compute_errors(
            ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
        )
        print(f"{name:<40} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
              f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.1f}s")

    print(f"{'-'*110}")
    print(f"  Total time: {total_time:.1f}s (target gen: {t_target_elapsed:.1f}s, "
          f"surrogate phase: {surrogate_time:.1f}s, PDE phases: {pde_time:.1f}s)")

    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)
    best_max_err = max(best_k0_err.max(), best_alpha_err.max())

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    print_redimensionalized_results(
        best_k0, true_k0_arr,
        best_alpha=best_alpha, true_alpha=true_alpha_arr,
    )

    # Cross-version comparison
    print(f"\n  {'='*80}")
    print(f"  v13 / v15 COMPARISON:")
    print(f"  {'='*80}")
    print(f"  {'Metric':<25} {'v13':>10} {'v15':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Total time (s)':<25} {'--':>10} {total_time:>10.1f}")
    print(f"  {'k0_1 err (%)':<25} {'--':>10} {best_k0_err[0]*100:>10.1f}")
    print(f"  {'k0_2 err (%)':<25} {'--':>10} {best_k0_err[1]*100:>10.1f}")
    print(f"  {'alpha_1 err (%)':<25} {'--':>10} {best_alpha_err[0]*100:>10.1f}")
    print(f"  {'alpha_2 err (%)':<25} {'--':>10} {best_alpha_err[1]*100:>10.1f}")
    print(f"  {'max err (%)':<25} {'--':>10} {best_max_err*100:>10.1f}")
    print(f"  {'='*80}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Surrogate phase:    {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE phases (P1+P2): {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*95}")

    # Save main results CSV
    csv_path = os.path.join(base_output, "master_comparison_v15.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
            "loss", "time_s",
        ])
        for name, ph in phase_results.items():
            k0_err, alpha_err = _compute_errors(
                ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
            )
            writer.writerow([
                name,
                f"{ph['k0'][0]:.8e}", f"{ph['k0'][1]:.8e}",
                f"{ph['alpha'][0]:.6f}", f"{ph['alpha'][1]:.6f}",
                f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                f"{ph['loss']:.12e}", f"{ph['time']:.1f}",
            ])
    print(f"\n  Results CSV saved -> {csv_path}")
    print(f"\n  Output: {base_output}/")
    print(f"\n=== Master Inference v15 Complete ===")


if __name__ == "__main__":
    main()
