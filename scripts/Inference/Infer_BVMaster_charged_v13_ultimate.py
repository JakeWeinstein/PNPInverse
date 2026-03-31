"""Master inference protocol v13 ULTIMATE for BV kinetics.

Extends v12 with ALL surrogate strategies (cascade + multistart + joint)
to find the best possible warm-start, then runs full PDE refinement.

Seven-phase pipeline:
    S1: Alpha init (surrogate)           ~0.1s
    S2: Joint L-BFGS-B (surrogate)       ~0.5s   (v12 Phase 2)
    S3: Cascade 3-pass (surrogate)       ~3-8s   NEW: per-observable weighting
    S4: MultiStart 20K (surrogate)       ~3-8s   NEW: global basin verification
    S5: Best surrogate selection          ~0s
    P1: PDE joint on SHALLOW cathodic    ~80-120s
    P2: PDE joint on FULL CATHODIC       ~200-300s

Target: all 4 params < 10% error, total < 7 min.

Usage (from PNPInverse/ directory)::

    # Full pipeline (~7 min)
    python scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py

    # Surrogate-only (fast, ~15s)
    python scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py --no-pde

    # Individual surrogate strategies
    python scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py --no-pde --surr-strategy cascade
    python scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py --no-pde --surr-strategy multistart

    # Use RBF instead of NN ensemble
    python scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py --model-type rbf

    # Compare ALL model types
    python scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py --compare --no-pde
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
from scipy.optimize import minimize

from Surrogate.io import load_surrogate
from Surrogate.objectives import AlphaOnlySurrogateObjective, SubsetSurrogateObjective
from Surrogate.cascade import CascadeConfig, run_cascade_inference
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


def _target_cache_path(phi_applied_values, observable_scale):
    """Build a cache file path based on ALL inputs that affect the PDE solution."""
    import hashlib
    species = FOUR_SPECIES_CHARGED
    parts = [
        phi_applied_values.tobytes(),
        str(observable_scale).encode(),
        # True kinetic parameters (the 4 being inferred)
        f"k0={K0_HAT},{K0_2_HAT}".encode(),
        f"alpha={ALPHA_1},{ALPHA_2}".encode(),
        # Species transport config (diffusion, charge, concentrations, stoichiometry)
        f"n_species={species.n_species}".encode(),
        f"z={species.z_vals}".encode(),
        f"D={species.d_vals_hat}".encode(),
        f"a={species.a_vals_hat}".encode(),
        f"c0={species.c0_vals_hat}".encode(),
        f"stoi_r1={species.stoichiometry_r1}".encode(),
        f"stoi_r2={species.stoichiometry_r2}".encode(),
        # Mesh and solver settings baked into _solve_clean_targets
        f"Nx=8,Ny=200,beta=3.0,dt=0.5,max_ss=100".encode(),
    ]
    key = hashlib.sha256(b"|".join(parts)).hexdigest()[:16]
    return os.path.join(_TARGET_CACHE_DIR, f"clean_targets_{key}.npz")


def _generate_targets_with_pde(phi_applied_values, observable_scale, noise_percent, noise_seed):
    """Generate target I-V curves using the PDE solver at true parameters.

    Caches the clean (noise-free) PDE solution so that repeated runs with
    different noise seeds / model types skip the ~70s PDE solve.
    """
    from Forward.steady_state import add_percent_noise

    cache_path = _target_cache_path(phi_applied_values, observable_scale)

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
                clean_cd, clean_pc = _solve_clean_targets(phi_applied_values, observable_scale)
                _save_clean_targets(cache_path, phi_applied_values, clean_cd, clean_pc)
        except Exception as exc:
            print(f"  Cache load failed ({exc}), regenerating...")
            clean_cd, clean_pc = _solve_clean_targets(phi_applied_values, observable_scale)
            _save_clean_targets(cache_path, phi_applied_values, clean_cd, clean_pc)
    else:
        clean_cd, clean_pc = _solve_clean_targets(phi_applied_values, observable_scale)
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


def _solve_clean_targets(phi_applied_values, observable_scale):
    """Run the PDE solver to get noise-free target I-V curves."""
    from Forward.steady_state import SteadyStateConfig
    from Forward.bv_solver import make_graded_rectangle_mesh
    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )

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

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    recovery = make_recovery_config(max_it_cap=600)
    dummy_target = np.zeros_like(phi_applied_values, dtype=float)

    clean = {}
    for obs_mode in ["current_density", "peroxide_current"]:
        _clear_caches()
        points = solve_bv_curve_points_with_warmstart(
            base_solver_params=base_sp,
            steady=steady,
            phi_applied_values=phi_applied_values,
            target_flux=dummy_target,
            k0_values=[K0_HAT, K0_2_HAT],
            blob_initial_condition=False,
            fail_penalty=1e9,
            forward_recovery=recovery,
            observable_mode=obs_mode,
            observable_reaction_index=None,
            observable_scale=observable_scale,
            mesh=mesh,
            alpha_values=[ALPHA_1, ALPHA_2],
            control_mode="joint",
            max_eta_gap=3.0,
        )
        clean[obs_mode] = np.array([float(p.simulated_flux) for p in points], dtype=float)

    _clear_caches()
    return clean["current_density"], clean["peroxide_current"]


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
    """Build a BVParallelPointConfig from solver params (mirrors v7)."""
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


def _run_surrogate_phases(
    surrogate, model_label, args,
    target_cd_surr, target_pc_surr,
    target_cd_full, target_pc_full,
    all_eta, eta_shallow,
    initial_k0_guess, initial_alpha_guess,
    true_k0_arr, true_alpha_arr,
    secondary_weight,
):
    """Run surrogate phases S1-S5 and return results dict.

    Returns
    -------
    dict with keys: s1_alpha, s2_k0/alpha/loss/time,
                    s3_k0/alpha/loss/time (if cascade),
                    s4_k0/alpha/loss/time (if multistart),
                    surr_best_k0/alpha/loss/source,
                    phase_results
    """
    (K0_1_LO, K0_1_HI, K0_2_LO, K0_2_HI,
     ALPHA_LO, ALPHA_HI, from_model) = _get_training_bounds(surrogate)

    if from_model:
        print(f"  [{model_label}] Using training bounds FROM MODEL")
    else:
        print(f"  [{model_label}] WARNING: model lacks training_bounds, using defaults")

    print(f"    k0_1 log10: [{np.log10(max(K0_1_LO, 1e-30)):.2f}, {np.log10(K0_1_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(K0_2_LO, 1e-30)):.2f}, {np.log10(K0_2_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_LO:.4f}, {ALPHA_HI:.4f}]")

    phase_results = {}
    surr_strategy = args.surr_strategy

    # -- S1: Alpha-only --
    print(f"\n  [{model_label}] S1: alpha-only")
    t_s1 = time.time()

    s1_obj = AlphaOnlySurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_surr,
        target_pc=target_pc_surr,
        fixed_k0=initial_k0_guess,
        secondary_weight=secondary_weight,
        fd_step=1e-5,
    )

    x0_s1 = np.array(initial_alpha_guess, dtype=float)
    bounds_s1 = [(ALPHA_LO, ALPHA_HI), (ALPHA_LO, ALPHA_HI)]

    result_s1 = minimize(
        s1_obj.objective,
        x0_s1,
        jac=s1_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_s1,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
    )

    s1_alpha = result_s1.x.copy()
    s1_k0 = np.asarray(initial_k0_guess)
    s1_loss = float(result_s1.fun)
    s1_time = time.time() - t_s1

    _print_phase_result(f"S1 alpha ({model_label})", s1_k0, s1_alpha,
                        true_k0_arr, true_alpha_arr, s1_loss, s1_time)
    _log_ensemble_uncertainty(surrogate, s1_k0, s1_alpha, "S1")

    phase_results[f"S1 alpha ({model_label})"] = {
        "k0": s1_k0.tolist(), "alpha": s1_alpha.tolist(),
        "loss": s1_loss, "time": s1_time,
    }

    # -- Compute shallow subset indices (shared by S2/S3/S4) --
    target_cd_shallow, target_pc_shallow = _subset_targets(
        target_cd_full, target_pc_full, all_eta, eta_shallow,
    )
    shallow_idx = []
    for eta in eta_shallow:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            shallow_idx.append(matches[0])
    shallow_idx = np.array(shallow_idx, dtype=int)

    # Candidate results: (k0, alpha, loss, source_name)
    candidates = []

    # -- S2: Joint 4-param L-BFGS-B (always run if strategy is 'all' or 'joint') --
    run_joint = surr_strategy in ("all", "joint")
    if run_joint:
        print(f"\n  [{model_label}] S2: joint 4-param (shallow)")
        t_s2 = time.time()

        s2_obj = SubsetSurrogateObjective(
            surrogate=surrogate,
            target_cd=target_cd_shallow,
            target_pc=target_pc_shallow,
            subset_idx=shallow_idx,
            secondary_weight=secondary_weight,
            fd_step=1e-5,
            log_space_k0=True,
        )

        x0_s2 = np.array([
            np.log10(initial_k0_guess[0]),
            np.log10(initial_k0_guess[1]),
            s1_alpha[0],
            s1_alpha[1],
        ], dtype=float)

        bounds_s2 = [
            (np.log10(K0_1_LO), np.log10(K0_1_HI)),
            (np.log10(K0_2_LO), np.log10(K0_2_HI)),
            (ALPHA_LO, ALPHA_HI),
            (ALPHA_LO, ALPHA_HI),
        ]

        result_s2 = minimize(
            s2_obj.objective,
            x0_s2,
            jac=s2_obj.gradient,
            method="L-BFGS-B",
            bounds=bounds_s2,
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        )

        s2_k0 = np.array([10.0**result_s2.x[0], 10.0**result_s2.x[1]])
        s2_alpha = result_s2.x[2:4].copy()
        s2_loss = float(result_s2.fun)
        s2_time = time.time() - t_s2

        _print_phase_result(f"S2 joint ({model_label})", s2_k0, s2_alpha,
                            true_k0_arr, true_alpha_arr, s2_loss, s2_time)
        _log_ensemble_uncertainty(surrogate, s2_k0, s2_alpha, "S2")

        phase_results[f"S2 joint ({model_label})"] = {
            "k0": s2_k0.tolist(), "alpha": s2_alpha.tolist(),
            "loss": s2_loss, "time": s2_time,
        }
        candidates.append((s2_k0, s2_alpha, s2_loss, f"S2 joint ({model_label})"))

    # -- S3: Cascade per-observable inference --
    run_cascade = surr_strategy in ("all", "cascade")
    if run_cascade:
        print(f"\n  [{model_label}] S3: Cascade per-observable inference")
        print(f"    pass1_weight={args.cascade_w1}, pass2_weight={args.cascade_w2}")
        t_s3 = time.time()

        cascade_config = CascadeConfig(
            pass1_weight=args.cascade_w1,
            pass2_weight=args.cascade_w2,
            pass1_maxiter=60,
            pass2_maxiter=60,
            polish_maxiter=30,
            polish_weight=1.0,
            skip_polish=False,
            fd_step=1e-5,
            verbose=True,
        )

        cascade_result = run_cascade_inference(
            surrogate=surrogate,
            target_cd=target_cd_full,
            target_pc=target_pc_full,
            initial_k0=initial_k0_guess,
            initial_alpha=list(s1_alpha),
            bounds_k0_1=(K0_1_LO, K0_1_HI),
            bounds_k0_2=(K0_2_LO, K0_2_HI),
            bounds_alpha=(ALPHA_LO, ALPHA_HI),
            config=cascade_config,
            subset_idx=shallow_idx,
        )

        s3_k0 = np.array([cascade_result.best_k0_1, cascade_result.best_k0_2])
        s3_alpha = np.array([cascade_result.best_alpha_1, cascade_result.best_alpha_2])
        s3_loss = cascade_result.best_loss
        s3_time = time.time() - t_s3

        _print_phase_result(f"S3 cascade ({model_label})", s3_k0, s3_alpha,
                            true_k0_arr, true_alpha_arr, s3_loss, s3_time)
        _log_ensemble_uncertainty(surrogate, s3_k0, s3_alpha, "S3")

        phase_results[f"S3 cascade ({model_label})"] = {
            "k0": s3_k0.tolist(), "alpha": s3_alpha.tolist(),
            "loss": s3_loss, "time": s3_time,
        }
        candidates.append((s3_k0, s3_alpha, s3_loss, f"S3 cascade ({model_label})"))

    # -- S4: MultiStart grid search --
    run_multistart = surr_strategy in ("all", "multistart")
    if run_multistart:
        print(f"\n  [{model_label}] S4: MultiStart grid search")
        print(f"    n_grid={args.multistart_n}, n_top={args.multistart_k}")
        t_s4 = time.time()

        ms_config = MultiStartConfig(
            n_grid=args.multistart_n,
            n_top_candidates=args.multistart_k,
            polish_maxiter=60,
            secondary_weight=secondary_weight,
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

        s4_k0 = np.array([ms_result.best_k0_1, ms_result.best_k0_2])
        s4_alpha = np.array([ms_result.best_alpha_1, ms_result.best_alpha_2])
        s4_loss = ms_result.best_loss
        s4_time = time.time() - t_s4

        _print_phase_result(f"S4 multistart ({model_label})", s4_k0, s4_alpha,
                            true_k0_arr, true_alpha_arr, s4_loss, s4_time)
        _log_ensemble_uncertainty(surrogate, s4_k0, s4_alpha, "S4")

        phase_results[f"S4 multistart ({model_label})"] = {
            "k0": s4_k0.tolist(), "alpha": s4_alpha.tolist(),
            "loss": s4_loss, "time": s4_time,
        }
        candidates.append((s4_k0, s4_alpha, s4_loss, f"S4 multistart ({model_label})"))

    # -- S5: Best surrogate selection --
    if not candidates:
        # Defensive guard — currently unreachable (all strategy choices add candidates)
        surr_best_k0 = s1_k0.copy()
        surr_best_alpha = s1_alpha.copy()
        surr_best_loss = s1_loss
        surr_best_source = f"S1 alpha ({model_label})"
    elif len(candidates) == 1:
        surr_best_k0, surr_best_alpha, surr_best_loss, surr_best_source = candidates[0]
    else:
        print(f"\n  [{model_label}] S5: Best surrogate selection")
        print(f"    {'Strategy':<35} {'Loss':>14} {'k0_1 err':>10} {'k0_2 err':>10} "
              f"{'a1 err':>10} {'a2 err':>10}")
        print(f"    {'-'*95}")
        best_idx = 0
        best_loss = candidates[0][2]
        for i, (c_k0, c_alpha, c_loss, c_name) in enumerate(candidates):
            c_k0_err, c_alpha_err = _compute_errors(c_k0, c_alpha, true_k0_arr, true_alpha_arr)
            print(f"    {c_name:<35} {c_loss:>14.6e} {c_k0_err[0]*100:>9.2f}% "
                  f"{c_k0_err[1]*100:>9.2f}% {c_alpha_err[0]*100:>9.2f}% "
                  f"{c_alpha_err[1]*100:>9.2f}%")
            if c_loss < best_loss:
                best_loss = c_loss
                best_idx = i

        surr_best_k0 = candidates[best_idx][0].copy()
        surr_best_alpha = candidates[best_idx][1].copy()
        surr_best_loss = candidates[best_idx][2]
        surr_best_source = candidates[best_idx][3]

        print(f"\n    Winner: {surr_best_source} (loss={surr_best_loss:.6e})")
        _log_ensemble_uncertainty(surrogate, surr_best_k0, surr_best_alpha, "S5-best")

    return {
        "s1_k0": s1_k0, "s1_alpha": s1_alpha, "s1_loss": s1_loss, "s1_time": s1_time,
        "surr_best_k0": surr_best_k0, "surr_best_alpha": surr_best_alpha,
        "surr_best_loss": surr_best_loss, "surr_best_source": surr_best_source,
        "phase_results": phase_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BV Master Inference v13 ULTIMATE (All Surrogate Strategies + PDE)"
    )
    # Surrogate model selection (from v12)
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

    # Surrogate strategy controls (NEW)
    parser.add_argument("--surr-strategy", type=str, default="all",
                        choices=["all", "joint", "cascade", "multistart"],
                        help="Surrogate strategy: all, joint, cascade, multistart (default: all)")
    parser.add_argument("--cascade-w1", type=float, default=0.5,
                        help="Cascade Pass 1 secondary_weight (CD-dominant)")
    parser.add_argument("--cascade-w2", type=float, default=2.0,
                        help="Cascade Pass 2 secondary_weight (PC-dominant)")
    parser.add_argument("--multistart-n", type=int, default=20000,
                        help="MultiStart grid size (default: 20000)")
    parser.add_argument("--multistart-k", type=int, default=20,
                        help="MultiStart top candidates to polish (default: 20)")

    # PDE phase controls
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip PDE phases P1-P2 (surrogate-only)")
    parser.add_argument("--skip-p1", action="store_true",
                        help="Skip P1 (shallow PDE); P2 warm-starts from surrogate best")
    parser.add_argument("--pde-p1-maxiter", type=int, default=25,
                        help="P1 (shallow) max L-BFGS-B iterations")
    parser.add_argument("--pde-p2-maxiter", type=int, default=20,
                        help="P2 (full cathodic) max L-BFGS-B iterations")
    parser.add_argument("--pde-secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current for PDE phases")
    parser.add_argument("--pde-cold-start", action="store_true",
                        help="Skip surrogate phases; run PDE from cold initial guesses")
    parser.add_argument("--workers", type=int, default=0,
                        help="PDE parallel workers (0=auto)")

    # Comparison mode (from v12)
    parser.add_argument("--compare", action="store_true",
                        help="Run ALL model types and produce comparison CSV")

    # Target/noise
    parser.add_argument("--noise-percent", type=float, default=2.0,
                        help="Target noise level (0.0 for noise-free)")
    parser.add_argument("--noise-seed", type=int, default=20260226,
                        help="Noise seed")

    # Surrogate-phase secondary weight (for S1/S2)
    parser.add_argument("--secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current for surrogate phases")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grids (IDENTICAL to v7/v9/v11/v12)
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

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "master_inference_v13")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}
    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v13 ULTIMATE")
    print(f"  (All Surrogate Strategies + PDE Refinement)")
    print(f"{'#'*70}")
    print(f"  Model type:          {args.model_type}")
    if args.model_type == "nn-ensemble":
        print(f"  Design:              {args.design}")
    print(f"  Surr strategy:       {args.surr_strategy}")
    print(f"  Compare mode:        {args.compare}")
    print(f"  True k0:             {true_k0}")
    print(f"  True alpha:          {true_alpha}")
    print(f"  Initial k0 guess:    {initial_k0_guess}")
    print(f"  Initial alpha guess: {initial_alpha_guess}")
    print(f"  Secondary weight:    {args.secondary_weight}")
    print(f"  PDE sec. weight:     {args.pde_secondary_weight}")
    print(f"  Noise: {args.noise_percent}% (seed={args.noise_seed})")
    if not args.no_pde:
        print(f"  PDE P1 maxiter:      {args.pde_p1_maxiter}")
        print(f"  PDE P2 maxiter:      {args.pde_p2_maxiter}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # PDE cold-start shortcut: skip surrogate, use initial guesses
    # ===================================================================
    if args.pde_cold_start:
        print(f"\n  --pde-cold-start: skipping surrogate phases")
        print(f"  PDE will warm-start from cold initial guesses:")
        print(f"    k0   = {initial_k0_guess}")
        print(f"    alpha= {initial_alpha_guess}")

        # Still need targets generated via PDE
        print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
        t_target = time.time()
        targets = _generate_targets_with_pde(all_eta, observable_scale, args.noise_percent, args.noise_seed)
        target_cd_full = targets["current_density"]
        target_pc_full = targets["peroxide_current"]
        t_target_elapsed = time.time() - t_target
        print(f"  Target generation: {t_target_elapsed:.1f}s")

        # Surrogate targets not available in cold-start mode
        target_cd_surr = target_cd_full
        target_pc_surr = target_pc_full

        surrogate_time = 0.0
        surr_best_k0 = np.asarray(initial_k0_guess)
        surr_best_alpha = np.asarray(initial_alpha_guess)
        surr_best_source = "cold-start"

        if args.compare:
            print("  WARNING: --compare is incompatible with --pde-cold-start; skipping comparison.")
            args.compare = False

    else:
        # ===================================================================
        # Load primary surrogate model
        # ===================================================================
        print(f"Loading primary surrogate model ({args.model_type})...")
        surrogate = _load_model(args.model_type, args)
        surrogate_eta = surrogate.phi_applied
        print(f"  Surrogate voltage points: {surrogate.n_eta}")
        print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

        if surrogate.n_eta != len(all_eta):
            raise ValueError(
                f"Surrogate has {surrogate.n_eta} voltage points but all_eta has {len(all_eta)}. "
                f"Ensure the surrogate was trained on the same voltage grid."
            )

        # ===================================================================
        # Generate targets using PDE solver
        # ===================================================================
        print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
        t_target = time.time()
        targets = _generate_targets_with_pde(all_eta, observable_scale, args.noise_percent, args.noise_seed)
        target_cd_full = targets["current_density"]
        target_pc_full = targets["peroxide_current"]
        t_target_elapsed = time.time() - t_target
        print(f"  Target generation: {t_target_elapsed:.1f}s")

        target_cd_surr = target_cd_full
        target_pc_surr = target_pc_full

        # ===================================================================
        # PHASES S1-S5: Surrogate optimization (primary model)
        # ===================================================================
        print(f"\n{'='*70}")
        print(f"  PHASES S1-S5: Surrogate optimization ({args.model_type})")
        print(f"  Strategy: {args.surr_strategy}")
        print(f"{'='*70}")

        primary_result = _run_surrogate_phases(
            surrogate=surrogate,
            model_label=args.model_type,
            args=args,
            target_cd_surr=target_cd_surr,
            target_pc_surr=target_pc_surr,
            target_cd_full=target_cd_full,
            target_pc_full=target_pc_full,
            all_eta=all_eta,
            eta_shallow=eta_shallow,
            initial_k0_guess=initial_k0_guess,
            initial_alpha_guess=initial_alpha_guess,
            true_k0_arr=true_k0_arr,
            true_alpha_arr=true_alpha_arr,
            secondary_weight=args.secondary_weight,
        )
        phase_results.update(primary_result["phase_results"])

        surr_best_k0 = primary_result["surr_best_k0"].copy()
        surr_best_alpha = primary_result["surr_best_alpha"].copy()
        surr_best_source = primary_result["surr_best_source"]

    print(f"\n  Surrogate best: {surr_best_source}")
    print(f"    k0   = {surr_best_k0.tolist()}")
    print(f"    alpha= {surr_best_alpha.tolist()}")

    # Record surrogate time BEFORE comparison block
    if not args.pde_cold_start:
        t_surrogate_end = time.time()
        surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    # ===================================================================
    # COMPARISON MODE: re-run surrogate phases with alternative models
    # ===================================================================
    comparison_results = {}
    if args.compare:
        print(f"\n{'='*70}")
        print(f"  COMPARISON MODE: running alternative models")
        print(f"{'='*70}")

        alt_models = ["nn-ensemble", "rbf", "pod-rbf-log", "pod-rbf-nolog"]
        alt_models = [m for m in alt_models if m != args.model_type]

        for alt_type in alt_models:
            print(f"\n--- Loading {alt_type} ---")
            try:
                alt_surr = _load_model(alt_type, args)
            except Exception as e:
                print(f"  SKIP {alt_type}: {e}")
                continue

            alt_result = _run_surrogate_phases(
                surrogate=alt_surr,
                model_label=alt_type,
                args=args,
                target_cd_surr=target_cd_surr,
                target_pc_surr=target_pc_surr,
                target_cd_full=target_cd_full,
                target_pc_full=target_pc_full,
                all_eta=all_eta,
                eta_shallow=eta_shallow,
                initial_k0_guess=initial_k0_guess,
                initial_alpha_guess=initial_alpha_guess,
                true_k0_arr=true_k0_arr,
                true_alpha_arr=true_alpha_arr,
                secondary_weight=args.secondary_weight,
            )
            comparison_results[alt_type] = alt_result
            phase_results.update(alt_result["phase_results"])

        # Write comparison CSV
        comp_csv_path = os.path.join(base_output, "model_comparison_v13.csv")
        with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
                "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
                "loss", "time_s",
            ])
            for name, ph in phase_results.items():
                k0_err, alpha_err = _compute_errors(
                    ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
                )
                model_label_csv = name.split("(")[-1].rstrip(")") if "(" in name else "unknown"
                writer.writerow([
                    model_label_csv, name,
                    f"{ph['k0'][0]:.8e}", f"{ph['k0'][1]:.8e}",
                    f"{ph['alpha'][0]:.6f}", f"{ph['alpha'][1]:.6f}",
                    f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                    f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                    f"{ph['loss']:.12e}", f"{ph['time']:.1f}",
                ])
        print(f"\n  Comparison CSV saved -> {comp_csv_path}")

    # Comparison block done; record comparison end time
    t_comparison_end = time.time()

    # ===================================================================
    # PDE PHASES P1-P2 (warm-started by best surrogate result)
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
                max_phase_points = len(eta_cathodic)
            else:
                max_phase_points = max(len(eta_shallow), len(eta_cathodic))
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
        print(f"\n  [v13] Shared parallel pool: {n_pde_workers} workers")

        # --- Extract PDE target subsets for P1/P2 from pre-generated targets ---
        p1_target_cd, p1_target_pc = _subset_targets(
            target_cd_full, target_pc_full, all_eta, eta_shallow,
        )
        p2_target_cd, p2_target_pc = _subset_targets(
            target_cd_full, target_pc_full, all_eta, eta_cathodic,
        )

        # --- P1: PDE shallow cathodic (skippable) ---
        if args.skip_p1:
            print(f"\n  [v13] --skip-p1: skipping P1, P2 warm-starts from surrogate best")
            p2_warm_k0 = surr_best_k0
            p2_warm_alpha = surr_best_alpha
            p2_warm_source = surr_best_source
        else:
            print(f"\n{'='*70}")
            print(f"  P1: PDE joint on SHALLOW cathodic")
            print(f"  Warm-start from: {surr_best_source}")
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
                observable_title="P1: PDE shallow (warm from surrogate best)",
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

        # --- P2: PDE full cathodic ---
        print(f"\n{'='*70}")
        print(f"  P2: PDE joint on FULL CATHODIC range")
        print(f"  Warm-start from {p2_warm_source}: k0={p2_warm_k0.tolist()}, alpha={p2_warm_alpha.tolist()}")
        print(f"  {len(eta_cathodic)}-pt [{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
        print(f"  maxiter={args.pde_p2_maxiter}, secondary_weight={pde_secondary_weight}")
        print(f"{'='*70}")
        t_p2 = time.time()

        p2_dir = os.path.join(base_output, "P2_pde_full_cathodic")

        request_p2 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=p2_warm_k0.tolist(),
            phi_applied_values=eta_cathodic.tolist(),
            target_csv_path=os.path.join(p2_dir, "target_primary.csv"),
            output_dir=p2_dir,
            regenerate_target=True,
            target_noise_percent=args.noise_percent,
            target_seed=args.noise_seed,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"P2: PDE full cathodic (warm from {p2_warm_source})",
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
                "primary": p2_target_cd,
                "secondary": p2_target_pc,
            },
        )
        p2_k0 = np.asarray(result_p2["best_k0"])
        p2_alpha = np.asarray(result_p2["best_alpha"])
        p2_loss = float(result_p2["best_loss"])
        p2_time = time.time() - t_p2

        close_parallel_pool()
        _clear_caches()
        print(f"  [v13] Shared parallel pool closed")

        _print_phase_result("P2 (PDE full cathodic)", p2_k0, p2_alpha,
                            true_k0_arr, true_alpha_arr, p2_loss, p2_time)
        phase_results["P2 (PDE full cathodic)"] = {
            "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
            "loss": p2_loss, "time": p2_time,
        }

        # Best-of selection
        if args.skip_p1:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "P2 (PDE full cathodic)"
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
                best_source = "P2 (PDE full cathodic)"
            else:
                best_k0, best_alpha = p1_k0.copy(), p1_alpha.copy()
                best_source = "P1 (PDE shallow)"

            best_max_err = min(p1_max_err, p2_max_err)
            print(f"\n  P1 vs P2: best is {best_source} (max err = {best_max_err*100:.2f}%)")
    else:
        best_k0, best_alpha = surr_best_k0.copy(), surr_best_alpha.copy()
        best_source = surr_best_source

    total_time = time.time() - t_total_start
    pde_time = (total_time - t_target_elapsed - surrogate_time) if not args.no_pde else 0.0

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*95}")
    print(f"  MASTER INFERENCE v13 ULTIMATE SUMMARY")
    print(f"  (All Surrogate Strategies + PDE Refinement)")
    print(f"{'#'*95}")
    print(f"  Primary model:     {args.model_type}")
    print(f"  Surr strategy:     {args.surr_strategy}")
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
          f"surrogate phases: {surrogate_time:.1f}s, PDE phases: {pde_time:.1f}s)")

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
    print(f"  v7 / v9 / v11 / v12 / v13 COMPARISON:")
    print(f"  {'='*80}")
    print(f"  {'Metric':<25} {'v7':>10} {'v9':>10} {'v11':>10} {'v12':>10} {'v13':>10}")
    print(f"  {'-'*78}")
    print(f"  {'Total time (s)':<25} {'415':>10} {'--':>10} {'--':>10} {'--':>10} {total_time:>10.1f}")
    print(f"  {'k0_1 err (%)':<25} {'10.9':>10} {'8.76':>10} {'--':>10} {'--':>10} {best_k0_err[0]*100:>10.1f}")
    print(f"  {'k0_2 err (%)':<25} {'2.6':>10} {'--':>10} {'0.82':>10} {'--':>10} {best_k0_err[1]*100:>10.1f}")
    print(f"  {'alpha_1 err (%)':<25} {'5.6':>10} {'--':>10} {'9.43':>10} {'3.2':>10} {best_alpha_err[0]*100:>10.1f}")
    print(f"  {'alpha_2 err (%)':<25} {'8.8':>10} {'--':>10} {'--':>10} {'3.9':>10} {best_alpha_err[1]*100:>10.1f}")
    print(f"  {'max err (%)':<25} {'10.9':>10} {'--':>10} {'9.43':>10} {'11.8':>10} {best_max_err*100:>10.1f}")
    print(f"  {'='*80}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Surrogate phases:   {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE phases (P1+P2): {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*95}")

    # Save main results CSV
    csv_path = os.path.join(base_output, "master_comparison_v13.csv")
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
    print(f"\n=== Master Inference v13 ULTIMATE Complete ===")


if __name__ == "__main__":
    main()
