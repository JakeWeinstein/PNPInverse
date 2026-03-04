#!/usr/bin/env python
"""Overnight surrogate training pipeline v11.

Usage:
    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python \
        scripts/surrogate/overnight_train_v11.py 2>&1 | tee StudyResults/surrogate_v11/run.log

Phases:
    1. Data generation (parallel, warm-started, ~7h)
    2. Model training (~1h)
    3. Evaluation (~20min)

Resume: Re-run the same command. Checkpoints are loaded automatically.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time

# Fix libomp conflict between Firedrake/PETSc and PyTorch on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
)
setup_firedrake_env()

import numpy as np

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v11")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Phase flags (set to False to skip) ----
RUN_PHASE_1 = False  # Already completed (12h 24m, 3194 merged samples)
RUN_PHASE_2 = True
RUN_PHASE_3 = True

# ---- Shared constants ----
MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0
N_WORKERS = 8
DT = 0.5
MAX_SS_STEPS = 100
T_END = DT * MAX_SS_STEPS  # 50.0


def _build_voltage_grid() -> np.ndarray:
    """Build the union voltage grid from v7 phases."""
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
    return np.sort(all_eta)[::-1]  # descending


def _build_base_solver_params():
    """Build base SolverParams for training data generation."""
    return make_bv_solver_params(
        eta_hat=0.0,
        dt=DT,
        t_end=T_END,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
    )


def _build_steady_config():
    """Build SteadyStateConfig for training."""
    from Forward.steady_state import SteadyStateConfig
    return SteadyStateConfig(
        relative_tolerance=1e-4,
        absolute_tolerance=1e-8,
        consecutive_steps=4,
        max_steps=MAX_SS_STEPS,
        flux_observable="total_species",
        verbose=False,
    )


def _fmt_duration(seconds: float) -> str:
    """Format duration as Xh Ym Zs."""
    if seconds < 0:
        return "???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# ==================================================================
# Phase 1: Data Generation
# ==================================================================

def phase_1_data_generation():
    """Generate ~3000 new training samples using parallel grouped workers."""
    from Surrogate.sampling import (
        ParameterBounds,
        generate_multi_region_lhs_samples,
    )
    from Surrogate.training import generate_training_dataset_parallel

    t0 = time.time()

    new_data_path = os.path.join(OUTPUT_DIR, "training_data_new_3000.npz")
    merged_path = os.path.join(OUTPUT_DIR, "training_data_merged.npz")
    split_path = os.path.join(OUTPUT_DIR, "split_indices.npz")
    checkpoint_path = new_data_path + ".checkpoint.npz"

    # Check if we already have final merged data
    if os.path.exists(merged_path):
        print(f"Phase 1: merged data already exists at {merged_path}, skipping.",
              flush=True)
        return

    # ---- 1.2 Sampling ----
    wide_bounds = ParameterBounds(
        k0_1_range=(1e-6, 1.0),
        k0_2_range=(1e-7, 0.1),
        alpha_1_range=(0.1, 0.9),
        alpha_2_range=(0.1, 0.9),
    )
    focused_bounds = ParameterBounds(
        k0_1_range=(1e-4, 1e-1),
        k0_2_range=(1e-5, 1e-2),
        alpha_1_range=(0.2, 0.7),
        alpha_2_range=(0.2, 0.7),
    )

    new_samples = generate_multi_region_lhs_samples(
        wide_bounds=wide_bounds,
        focused_bounds=focused_bounds,
        n_base=2000,
        n_focused=1000,
        seed_base=200,
        seed_focused=300,
        log_space_k0=True,
    )
    print(f"Generated {new_samples.shape[0]} parameter samples", flush=True)

    # ---- 1.3-1.5 Parallel data generation ----
    phi_applied = _build_voltage_grid()
    base_sp = _build_base_solver_params()
    steady = _build_steady_config()

    result = generate_training_dataset_parallel(
        new_samples,
        phi_applied_values=phi_applied,
        base_solver_params=base_sp,
        steady=steady,
        observable_scale=-I_SCALE,
        mesh_params=(MESH_NX, MESH_NY, MESH_BETA),
        output_path=new_data_path,
        n_workers=N_WORKERS,
        min_converged_fraction=0.8,
        verbose=True,
        resume_from=checkpoint_path,
    )

    n_new_valid = result["n_valid"]
    new_valid_params = result["parameters"]
    new_valid_cd = result["current_density"]
    new_valid_pc = result["peroxide_current"]

    print(f"\nPhase 1 new data: {n_new_valid} valid samples", flush=True)

    # ---- 1.6 Data merging ----
    v9_path = os.path.join(_ROOT, "StudyResults", "surrogate_v9", "training_data_500.npz")
    if os.path.exists(v9_path):
        v9 = np.load(v9_path)
        v9_params = v9["parameters"]
        v9_cd = v9["current_density"]
        v9_pc = v9["peroxide_current"]
        v9_phi = v9["phi_applied"]

        # Verify voltage grids match
        if np.allclose(v9_phi, phi_applied):
            merged_params = np.concatenate([v9_params, new_valid_params], axis=0)
            merged_cd = np.concatenate([v9_cd, new_valid_cd], axis=0)
            merged_pc = np.concatenate([v9_pc, new_valid_pc], axis=0)
            n_v9 = len(v9_params)
            print(f"Merged v9 ({n_v9}) + new ({n_new_valid}) = {len(merged_params)} total",
                  flush=True)
        else:
            print("WARNING: v9 phi_applied grid does not match! Using only new data.",
                  flush=True)
            merged_params = new_valid_params
            merged_cd = new_valid_cd
            merged_pc = new_valid_pc
            n_v9 = 0
    else:
        print(f"WARNING: v9 data not found at {v9_path}. Using only new data.",
              flush=True)
        merged_params = new_valid_params
        merged_cd = new_valid_cd
        merged_pc = new_valid_pc
        n_v9 = 0

    np.savez_compressed(
        merged_path,
        parameters=merged_params,
        current_density=merged_cd,
        peroxide_current=merged_pc,
        phi_applied=phi_applied,
        n_v9=n_v9,
        n_new=n_new_valid,
    )

    # ---- 1.7 Train/test split ----
    rng = np.random.default_rng(seed=777)
    N_total = len(merged_params)
    n_test = max(50, int(N_total * 0.15))
    perm = rng.permutation(N_total)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    np.savez_compressed(
        split_path,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    elapsed = time.time() - t0
    print(f"\n{'#'*78}", flush=True)
    print(f"  PHASE 1 COMPLETE: DATA GENERATION", flush=True)
    print(f"  Total time     : {_fmt_duration(elapsed)}", flush=True)
    print(f"  Total merged   : {N_total}", flush=True)
    print(f"  Train / Test   : {len(train_idx)} / {len(test_idx)}", flush=True)
    print(f"  Saved to       : {merged_path}", flush=True)
    print(f"{'#'*78}\n", flush=True)


# ==================================================================
# Phase 2: Model Training
# ==================================================================

def phase_2_model_training():
    """Train RBF, POD-RBF, and NN ensemble models."""
    t0 = time.time()

    merged_path = os.path.join(OUTPUT_DIR, "training_data_merged.npz")
    split_path = os.path.join(OUTPUT_DIR, "split_indices.npz")

    if not os.path.exists(merged_path):
        print("ERROR: merged training data not found. Run Phase 1 first.", flush=True)
        return
    if not os.path.exists(split_path):
        print("ERROR: split indices not found. Run Phase 1 first.", flush=True)
        return

    data = np.load(merged_path)
    params = data["parameters"]
    cd = data["current_density"]
    pc = data["peroxide_current"]
    phi = data["phi_applied"]

    splits = np.load(split_path)
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    train_params = params[train_idx]
    train_cd = cd[train_idx]
    train_pc = pc[train_idx]
    test_params = params[test_idx]
    test_cd = cd[test_idx]
    test_pc = pc[test_idx]

    print(f"Training set: {len(train_params)} samples", flush=True)
    print(f"Test set:     {len(test_params)} samples", flush=True)

    fit_times = {}

    # ---- Model A: Baseline RBF ----
    print(f"\n{'='*60}", flush=True)
    print(f"  Model A: Baseline RBF", flush=True)
    print(f"{'='*60}", flush=True)

    from Surrogate.surrogate_model import BVSurrogateModel, SurrogateConfig

    t_fit = time.time()
    config_rbf = SurrogateConfig(
        smoothing_cd=0.0,
        smoothing_pc=1e-3,
        log_space_k0=True,
        normalize_inputs=True,
    )
    model_rbf = BVSurrogateModel(config=config_rbf)
    model_rbf.fit(
        parameters=train_params,
        current_density=train_cd,
        peroxide_current=train_pc,
        phi_applied=phi,
    )
    fit_times["RBF-baseline"] = time.time() - t_fit
    print(f"  RBF fit time: {fit_times['RBF-baseline']:.1f}s", flush=True)

    rbf_path = os.path.join(OUTPUT_DIR, "model_rbf_baseline.pkl")
    with open(rbf_path, "wb") as f:
        pickle.dump(model_rbf, f)

    # ---- Model B: POD-RBF with log PC ----
    print(f"\n{'='*60}", flush=True)
    print(f"  Model B: POD-RBF (log PC transform)", flush=True)
    print(f"{'='*60}", flush=True)

    from Surrogate.pod_rbf_model import PODRBFSurrogateModel, PODRBFConfig

    t_fit = time.time()
    config_pod_log = PODRBFConfig(
        variance_threshold=0.999,
        kernel="thin_plate_spline",
        degree=1,
        log_space_k0=True,
        normalize_inputs=True,
        optimize_smoothing=True,
        n_smoothing_candidates=30,
        smoothing_range=(1e-8, 1e0),
        log_transform_pc=True,
        max_modes=None,
    )
    model_pod_log = PODRBFSurrogateModel(config=config_pod_log)
    model_pod_log.fit(
        parameters=train_params,
        current_density=train_cd,
        peroxide_current=train_pc,
        phi_applied=phi,
        verbose=True,
    )
    fit_times["POD-RBF-log"] = time.time() - t_fit
    print(f"  POD-RBF-log fit time: {fit_times['POD-RBF-log']:.1f}s", flush=True)

    pod_log_path = os.path.join(OUTPUT_DIR, "model_pod_rbf_log.pkl")
    with open(pod_log_path, "wb") as f:
        pickle.dump(model_pod_log, f)

    # ---- Model C: POD-RBF without log PC ----
    print(f"\n{'='*60}", flush=True)
    print(f"  Model C: POD-RBF (no log PC transform)", flush=True)
    print(f"{'='*60}", flush=True)

    t_fit = time.time()
    config_pod_nolog = PODRBFConfig(
        variance_threshold=0.999,
        kernel="thin_plate_spline",
        degree=1,
        log_space_k0=True,
        normalize_inputs=True,
        optimize_smoothing=True,
        n_smoothing_candidates=30,
        smoothing_range=(1e-8, 1e0),
        log_transform_pc=False,
        max_modes=None,
    )
    model_pod_nolog = PODRBFSurrogateModel(config=config_pod_nolog)
    model_pod_nolog.fit(
        parameters=train_params,
        current_density=train_cd,
        peroxide_current=train_pc,
        phi_applied=phi,
        verbose=True,
    )
    fit_times["POD-RBF-nolog"] = time.time() - t_fit
    print(f"  POD-RBF-nolog fit time: {fit_times['POD-RBF-nolog']:.1f}s", flush=True)

    pod_nolog_path = os.path.join(OUTPUT_DIR, "model_pod_rbf_nolog.pkl")
    with open(pod_nolog_path, "wb") as f:
        pickle.dump(model_pod_nolog, f)

    # ---- Model D: NN Ensemble ----
    print(f"\n{'='*60}", flush=True)
    print(f"  Model D: NN Ensemble (5 configs x 5 members)", flush=True)
    print(f"{'='*60}", flush=True)

    from Surrogate.nn_training import NNTrainingConfig, train_nn_ensemble

    nn_configs = {
        "D1-default": NNTrainingConfig(
            epochs=5000, lr=1e-3, weight_decay=1e-4, patience=500,
            T_0=500, T_mult=2, eta_min=1e-6,
            hidden=128, n_blocks=4,
            monotonicity_weight=0.01, smoothness_weight=0.001,
        ),
        "D2-wider": NNTrainingConfig(
            epochs=5000, lr=1e-3, weight_decay=1e-4, patience=500,
            T_0=500, T_mult=2, eta_min=1e-6,
            hidden=256, n_blocks=4,
            monotonicity_weight=0.01, smoothness_weight=0.001,
        ),
        "D3-deeper": NNTrainingConfig(
            epochs=5000, lr=1e-3, weight_decay=1e-4, patience=500,
            T_0=500, T_mult=2, eta_min=1e-6,
            hidden=128, n_blocks=6,
            monotonicity_weight=0.01, smoothness_weight=0.001,
        ),
        "D4-no-physics": NNTrainingConfig(
            epochs=5000, lr=1e-3, weight_decay=1e-4, patience=500,
            T_0=500, T_mult=2, eta_min=1e-6,
            hidden=128, n_blocks=4,
            monotonicity_weight=0.0, smoothness_weight=0.0,
        ),
        "D5-strong-physics": NNTrainingConfig(
            epochs=5000, lr=1e-3, weight_decay=1e-4, patience=500,
            T_0=500, T_mult=2, eta_min=1e-6,
            hidden=128, n_blocks=4,
            monotonicity_weight=0.1, smoothness_weight=0.01,
        ),
    }

    nn_ensemble_results = {}

    for cfg_name, nn_cfg in nn_configs.items():
        print(f"\n  Training NN ensemble: {cfg_name}", flush=True)
        ensemble_dir = os.path.join(OUTPUT_DIR, "nn_ensemble", cfg_name)

        t_fit = time.time()
        try:
            models, meta = train_nn_ensemble(
                parameters=train_params,
                current_density=train_cd,
                peroxide_current=train_pc,
                phi_applied=phi,
                n_ensemble=5,
                config=nn_cfg,
                output_dir=ensemble_dir,
                base_seed=42,
                val_fraction=0.15,
                verbose=True,
            )
            fit_times[f"NN-{cfg_name}"] = time.time() - t_fit
            nn_ensemble_results[cfg_name] = {
                "models": models,
                "meta": meta,
            }
            print(f"  {cfg_name} fit time: {fit_times[f'NN-{cfg_name}']:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"  ERROR training {cfg_name}: {e}", flush=True)
            fit_times[f"NN-{cfg_name}"] = time.time() - t_fit

    # Save fit times
    np.savez(
        os.path.join(OUTPUT_DIR, "fit_times.npz"),
        **{k: v for k, v in fit_times.items()},
    )

    elapsed = time.time() - t0
    print(f"\n{'#'*78}", flush=True)
    print(f"  PHASE 2 COMPLETE: MODEL TRAINING", flush=True)
    print(f"  Total time: {_fmt_duration(elapsed)}", flush=True)
    print(f"  Models trained: {len(fit_times)}", flush=True)
    print(f"{'#'*78}\n", flush=True)


# ==================================================================
# Phase 3: Evaluation
# ==================================================================

class EnsembleMeanWrapper:
    """Wraps an ensemble of NNSurrogateModels to match the validate_surrogate API."""

    def __init__(self, models):
        self.models = models
        self.phi_applied = models[0].phi_applied

    def predict_batch(self, parameters):
        from Surrogate.nn_training import predict_ensemble
        ens = predict_ensemble(self.models, parameters)
        return {
            "current_density": ens["current_density_mean"],
            "peroxide_current": ens["peroxide_current_mean"],
            "phi_applied": ens["phi_applied"],
        }

    def predict(self, k0_1, k0_2, alpha_1, alpha_2):
        params = np.array([[k0_1, k0_2, alpha_1, alpha_2]])
        batch = self.predict_batch(params)
        return {
            "current_density": batch["current_density"][0],
            "peroxide_current": batch["peroxide_current"][0],
            "phi_applied": batch["phi_applied"],
        }


def phase_3_evaluation():
    """Evaluate all models and generate comparison table."""
    t0 = time.time()

    merged_path = os.path.join(OUTPUT_DIR, "training_data_merged.npz")
    split_path = os.path.join(OUTPUT_DIR, "split_indices.npz")

    if not os.path.exists(merged_path) or not os.path.exists(split_path):
        print("ERROR: data not found. Run Phase 1 first.", flush=True)
        return

    data = np.load(merged_path)
    params = data["parameters"]
    cd_data = data["current_density"]
    pc_data = data["peroxide_current"]
    phi = data["phi_applied"]

    splits = np.load(split_path)
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    test_params = params[test_idx]
    test_cd = cd_data[test_idx]
    test_pc = pc_data[test_idx]
    train_params = params[train_idx]

    # ---- Load models ----
    models_to_evaluate = {}

    # RBF baseline
    rbf_path = os.path.join(OUTPUT_DIR, "model_rbf_baseline.pkl")
    if os.path.exists(rbf_path):
        with open(rbf_path, "rb") as f:
            models_to_evaluate["RBF-baseline"] = pickle.load(f)

    # POD-RBF variants
    for name, fname in [("POD-RBF-log", "model_pod_rbf_log.pkl"),
                        ("POD-RBF-nolog", "model_pod_rbf_nolog.pkl")]:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models_to_evaluate[name] = pickle.load(f)

    # NN ensembles
    from Surrogate.nn_model import NNSurrogateModel
    nn_ensemble_dir = os.path.join(OUTPUT_DIR, "nn_ensemble")
    if os.path.exists(nn_ensemble_dir):
        for cfg_name in sorted(os.listdir(nn_ensemble_dir)):
            cfg_dir = os.path.join(nn_ensemble_dir, cfg_name)
            if not os.path.isdir(cfg_dir):
                continue
            member_models = []
            for i in range(5):
                member_path = os.path.join(cfg_dir, f"member_{i}", "saved_model")
                if os.path.exists(member_path):
                    try:
                        m = NNSurrogateModel.load(member_path)
                        member_models.append(m)
                    except Exception as e:
                        print(f"  Warning: failed to load {member_path}: {e}",
                              flush=True)
            if member_models:
                models_to_evaluate[f"NN-{cfg_name}"] = EnsembleMeanWrapper(member_models)

    if not models_to_evaluate:
        print("ERROR: No models found. Run Phase 2 first.", flush=True)
        return

    print(f"Evaluating {len(models_to_evaluate)} models on {len(test_params)} test samples",
          flush=True)

    # ---- 3.1 Surrogate accuracy ----
    from Surrogate.validation import validate_surrogate, print_validation_report

    accuracy_results = {}
    for name, model in models_to_evaluate.items():
        print(f"\n--- {name} ---", flush=True)
        try:
            metrics = validate_surrogate(model, test_params, test_cd, test_pc)
            accuracy_results[name] = metrics
            print_validation_report(metrics)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)

    # ---- 3.2 Parameter recovery ----
    print(f"\n{'='*60}", flush=True)
    print(f"  PARAMETER RECOVERY TEST (10 synthetic cases)", flush=True)
    print(f"{'='*60}", flush=True)

    from Surrogate.cascade import run_cascade_inference, CascadeConfig

    cascade_cfg = CascadeConfig(
        pass1_weight=0.5,
        pass2_weight=2.0,
        pass1_maxiter=60,
        pass2_maxiter=60,
        polish_maxiter=30,
        polish_weight=1.0,
        skip_polish=False,
        verbose=False,
    )

    test_true_params = [
        (1e-3, 1e-4, 0.5, 0.5),
        (1e-2, 1e-3, 0.4, 0.6),
        (1e-5, 1e-6, 0.3, 0.3),
        (5e-4, 5e-5, 0.5, 0.5),
        (1e-3, 1e-5, 0.6, 0.4),
        (1e-4, 1e-3, 0.4, 0.5),
        (1e-3, 1e-4, 0.2, 0.8),
        (1e-3, 1e-4, 0.7, 0.3),
        (5e-2, 5e-3, 0.45, 0.55),
        (5e-5, 5e-6, 0.55, 0.45),
    ]

    recovery_results = {}
    for name, model in models_to_evaluate.items():
        errors_k0_1, errors_k0_2 = [], []
        errors_a1, errors_a2 = [], []
        n_success = 0

        for true_k0_1, true_k0_2, true_a1, true_a2 in test_true_params:
            try:
                target_pred = model.predict(true_k0_1, true_k0_2, true_a1, true_a2)
                target_cd = target_pred["current_density"].copy()
                target_pc = target_pred["peroxide_current"].copy()

                # Add small noise
                rng_noise = np.random.default_rng(42)
                target_cd += rng_noise.normal(
                    0, 0.01 * np.abs(target_cd).max(), target_cd.shape)
                target_pc += rng_noise.normal(
                    0, 0.01 * np.abs(target_pc).max(), target_pc.shape)

                initial_k0 = [true_k0_1 * 3.0, true_k0_2 * 0.3]
                initial_alpha = [0.5, 0.5]

                result = run_cascade_inference(
                    surrogate=model,
                    target_cd=target_cd,
                    target_pc=target_pc,
                    initial_k0=initial_k0,
                    initial_alpha=initial_alpha,
                    bounds_k0_1=(1e-6, 1.0),
                    bounds_k0_2=(1e-7, 0.1),
                    bounds_alpha=(0.1, 0.9),
                    config=cascade_cfg,
                )

                errors_k0_1.append(abs(result.best_k0_1 - true_k0_1) / true_k0_1)
                errors_k0_2.append(abs(result.best_k0_2 - true_k0_2) / true_k0_2)
                errors_a1.append(abs(result.best_alpha_1 - true_a1))
                errors_a2.append(abs(result.best_alpha_2 - true_a2))
                n_success += 1
            except Exception as e:
                print(f"  {name}: recovery failed for "
                      f"({true_k0_1:.1e},{true_k0_2:.1e}): {e}", flush=True)

        if errors_k0_2:
            recovery_results[name] = {
                "k0_1_mean": float(np.mean(errors_k0_1)),
                "k0_1_max": float(np.max(errors_k0_1)),
                "k0_2_mean": float(np.mean(errors_k0_2)),
                "k0_2_max": float(np.max(errors_k0_2)),
                "a1_mean": float(np.mean(errors_a1)),
                "a2_mean": float(np.mean(errors_a2)),
                "n_success": n_success,
            }
            print(f"\n{name} -- Parameter Recovery ({n_success}/10 cases):", flush=True)
            print(f"  k0_1 error: mean={np.mean(errors_k0_1)*100:.1f}%, "
                  f"max={np.max(errors_k0_1)*100:.1f}%", flush=True)
            print(f"  k0_2 error: mean={np.mean(errors_k0_2)*100:.1f}%, "
                  f"max={np.max(errors_k0_2)*100:.1f}%", flush=True)
            print(f"  alpha_1 abs error: mean={np.mean(errors_a1):.4f}", flush=True)
            print(f"  alpha_2 abs error: mean={np.mean(errors_a2):.4f}", flush=True)

    # ---- 3.4 Comparison table ----
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "n_train", "n_test",
            "cd_rmse", "pc_rmse",
            "cd_nrmse_%", "pc_nrmse_%",
            "cd_max_err", "pc_max_err",
            "k0_2_recovery_mean_%", "k0_2_recovery_max_%",
        ])
        for name in accuracy_results:
            m = accuracy_results[name]
            rec = recovery_results.get(name, {})
            writer.writerow([
                name,
                len(train_params),
                m.get("n_test", len(test_params)),
                f"{m['cd_rmse']:.6e}",
                f"{m['pc_rmse']:.6e}",
                f"{m['cd_mean_relative_error']*100:.2f}",
                f"{m['pc_mean_relative_error']*100:.2f}",
                f"{m['cd_max_abs_error']:.6e}",
                f"{m['pc_max_abs_error']:.6e}",
                f"{rec.get('k0_2_mean', float('nan'))*100:.1f}",
                f"{rec.get('k0_2_max', float('nan'))*100:.1f}",
            ])
    print(f"\nComparison table saved to {csv_path}", flush=True)

    # ---- 3.6 Select best model ----
    if accuracy_results:
        # Simple selection: lowest PC RMSE
        best_name = min(accuracy_results,
                        key=lambda n: accuracy_results[n]["pc_rmse"])
        print(f"\nBest model by PC RMSE: {best_name}", flush=True)
        print(f"  PC RMSE = {accuracy_results[best_name]['pc_rmse']:.6e}", flush=True)

        if best_name in recovery_results:
            rec = recovery_results[best_name]
            print(f"  k0_2 recovery: mean={rec['k0_2_mean']*100:.1f}%, "
                  f"max={rec['k0_2_max']*100:.1f}%", flush=True)

        # Copy best model
        best_dir = os.path.join(OUTPUT_DIR, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        with open(os.path.join(best_dir, "best_model_name.txt"), "w") as f:
            f.write(best_name)

        # Copy the model file
        if best_name == "RBF-baseline":
            src = os.path.join(OUTPUT_DIR, "model_rbf_baseline.pkl")
        elif best_name == "POD-RBF-log":
            src = os.path.join(OUTPUT_DIR, "model_pod_rbf_log.pkl")
        elif best_name == "POD-RBF-nolog":
            src = os.path.join(OUTPUT_DIR, "model_pod_rbf_nolog.pkl")
        else:
            src = None  # NN ensemble -- already saved in nn_ensemble/

        if src and os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(best_dir, "model.pkl"))

        print(f"Best model info saved to {best_dir}/", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'#'*78}", flush=True)
    print(f"  PHASE 3 COMPLETE: EVALUATION", flush=True)
    print(f"  Total time: {_fmt_duration(elapsed)}", flush=True)
    print(f"{'#'*78}\n", flush=True)


# ==================================================================
# Main
# ==================================================================

if __name__ == "__main__":
    t_grand = time.time()

    if RUN_PHASE_1:
        print("=" * 78, flush=True)
        print("  PHASE 1: DATA GENERATION", flush=True)
        print("=" * 78, flush=True)
        phase_1_data_generation()

    if RUN_PHASE_2:
        print("=" * 78, flush=True)
        print("  PHASE 2: MODEL TRAINING", flush=True)
        print("=" * 78, flush=True)
        phase_2_model_training()

    if RUN_PHASE_3:
        print("=" * 78, flush=True)
        print("  PHASE 3: EVALUATION", flush=True)
        print("=" * 78, flush=True)
        phase_3_evaluation()

    total = time.time() - t_grand
    print(f"\nTotal elapsed: {total/3600:.1f}h ({_fmt_duration(total)})", flush=True)
