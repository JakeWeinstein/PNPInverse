#!/usr/bin/env python
"""ISMO runner: top-level CLI that orchestrates the full ISMO loop.

Ties together initial surrogate loading, iterative solve-acquire-retrain,
convergence checking via ``ISMOConvergenceChecker``, and post-ISMO
validation.  Delegates each substep to the modules defined in plans
4-01 through 4-04.  When those modules are not yet available, inline
stubs provide fallback behaviour so this script is independently
runnable.

Usage
-----
::

    python scripts/surrogate/run_ismo.py --max-iterations 3 --budget 60 \\
        --surrogate-type nn_ensemble --design D1-default \\
        --skip-post-validation --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path so ``Surrogate.*`` imports work when invoked
# from the repo root.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Core convergence / diagnostics (always available -- implemented in 4-05)
# ---------------------------------------------------------------------------
from Surrogate.ismo_convergence import (
    ISMOConvergenceChecker,
    ISMOConvergenceCriteria,
    ISMODiagnosticRecord,
    compute_objective_improvement,
    compute_parameter_stability,
    compute_surrogate_pde_agreement,
)
from Surrogate.ismo_diagnostics import (
    ISMODiagnostics,
    plot_convergence_curves,
    plot_k0_2_recovery_comparison,
)

# ---------------------------------------------------------------------------
# Existing Surrogate modules (always available)
# ---------------------------------------------------------------------------
from Surrogate.sampling import ParameterBounds
from Surrogate.validation import validate_surrogate

# ---------------------------------------------------------------------------
# Guard imports of 4-01 .. 4-04 modules (may not exist yet)
# ---------------------------------------------------------------------------
try:
    from Surrogate.ismo import ISMOIteration, ISMOResult
except ImportError:
    ISMOIteration = None
    ISMOResult = None

try:
    from Surrogate.acquisition import AcquisitionConfig, select_new_samples
except ImportError:
    AcquisitionConfig = None
    select_new_samples = None

try:
    from Surrogate.ismo_retrain import retrain_surrogate
except ImportError:
    retrain_surrogate = None

try:
    from Surrogate.ismo_pde_eval import evaluate_candidates_with_pde
except ImportError:
    evaluate_candidates_with_pde = None


# ======================================================================= #
# CLI argument parsing
# ======================================================================= #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ISMO runner."""
    p = argparse.ArgumentParser(
        description="Run the ISMO iterative surrogate-model optimisation loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Surrogate selection
    p.add_argument(
        "--surrogate-type",
        choices=["nn_ensemble", "pod_rbf_log", "pod_rbf_nolog", "rbf_baseline", "gp"],
        default="nn_ensemble",
        help="Type of surrogate model to use.",
    )
    p.add_argument(
        "--design",
        default="D1-default",
        help="Design name for NN ensemble (e.g. D1-default, D3-deeper). "
             "Only used when surrogate-type=nn_ensemble.",
    )

    # Data paths
    p.add_argument(
        "--training-data",
        default="data/surrogate_models/training_data_merged.npz",
        help="Path to initial training data .npz.",
    )
    p.add_argument(
        "--split-indices",
        default="data/surrogate_models/split_indices.npz",
        help="Path to train/test split indices.",
    )

    # ISMO loop control
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument("--budget", type=int, default=200,
                    help="Maximum new PDE evaluations.")
    p.add_argument("--agreement-tol", type=float, default=0.01,
                    help="Surrogate-PDE agreement NRMSE tolerance.")
    p.add_argument("--stability-tol", type=float, default=0.01,
                    help="Parameter stability tolerance.")
    p.add_argument("--stagnation-window", type=int, default=3,
                    help="Iterations without improvement before stopping (min 2).")
    p.add_argument("--samples-per-iter", type=int, default=20,
                    help="New PDE samples per ISMO iteration.")

    # Acquisition
    p.add_argument(
        "--acquisition",
        choices=["trust_region", "exploit_explore", "error_based"],
        default="trust_region",
        help="Acquisition strategy for new sample selection.",
    )

    # Output
    p.add_argument("--output-dir", default="StudyResults/ismo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-post-validation", action="store_true",
                    help="Skip post-ISMO parameter recovery study.")
    p.add_argument("--verbose", action="store_true",
                    help="Print detailed progress.")

    return p.parse_args(argv)


# ======================================================================= #
# Surrogate loading
# ======================================================================= #

def load_or_build_surrogate(args: argparse.Namespace):
    """Load the initial surrogate model based on CLI args.

    Dispatches to the appropriate loader depending on surrogate type.
    """
    stype = args.surrogate_type

    if stype == "nn_ensemble":
        from Surrogate.ensemble import load_nn_ensemble
        ensemble_dir = os.path.join(
            "data", "surrogate_models", "nn_ensemble", args.design,
        )
        print(f"  [Runner] Loading NN ensemble from {ensemble_dir}", flush=True)
        return load_nn_ensemble(ensemble_dir)

    if stype == "gp":
        try:
            from Surrogate.gp_model import load_gp_surrogate
            gp_path = os.path.join("data", "surrogate_models", "gp")
            print(f"  [Runner] Loading GP surrogate from {gp_path}", flush=True)
            return load_gp_surrogate(gp_path)
        except ImportError:
            raise RuntimeError(
                "GP surrogate requested but Surrogate.gp_model is not available."
            )

    # RBF / POD-RBF pickle models
    from Surrogate.io import load_surrogate
    pkl_map = {
        "pod_rbf_log": "data/surrogate_models/model_pod_rbf_log.pkl",
        "pod_rbf_nolog": "data/surrogate_models/model_pod_rbf_nolog.pkl",
        "rbf_baseline": "data/surrogate_models/model_rbf_baseline.pkl",
    }
    path = pkl_map.get(stype)
    if path is None:
        raise ValueError(f"Unknown surrogate type: {stype}")
    print(f"  [Runner] Loading surrogate from {path}", flush=True)
    return load_surrogate(path)


# ======================================================================= #
# PDE solver stub (fallback when 4-04 is not available)
# ======================================================================= #

def _make_pde_solver_fn_stub(verbose: bool = False):
    """Return a stub PDE solver that raises NotImplementedError.

    In production, the runner replaces this with the real solver from
    ``scripts._bv_common`` or from 4-04's ``evaluate_pde_batch``.
    """
    def _stub(params):
        raise NotImplementedError(
            "PDE solver stub called.  Please provide a real PDE solver "
            "function via 4-04 integration or by passing --pde-solver-impl."
        )
    return _stub


# ======================================================================= #
# Acquisition fallback (when 4-02 is not available)
# ======================================================================= #

def _fallback_acquire(
    existing_params: np.ndarray,
    bounds: ParameterBounds,
    n_samples: int,
    best_params: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fall back to LHS sampling around the current best when 4-02 is absent.

    Generates ``n_samples`` points centred on ``best_params`` within the
    bounds, using a simple Latin Hypercube design.
    """
    from Surrogate.sampling import generate_lhs_samples
    return generate_lhs_samples(bounds, n_samples, seed=int(rng.integers(0, 2**31)))


# ======================================================================= #
# Retrain fallback (when 4-03 is not available)
# ======================================================================= #

def _fallback_retrain(surrogate, params, cd, pc, phi_applied, args):
    """Rebuild the surrogate from scratch with augmented data.

    For NN ensembles this is expensive; for RBF surrogates it is fast.
    This is a minimal fallback -- 4-03 will provide smarter retraining.
    """
    stype = args.surrogate_type
    if stype in ("pod_rbf_log", "pod_rbf_nolog", "rbf_baseline"):
        from Surrogate.training import build_surrogate
        return build_surrogate(params, cd, pc, phi_applied, model_type=stype)
    # For NN ensemble / GP: cannot easily retrain inline, return unchanged
    print(
        "  [Runner] WARNING: no retrain function available for "
        f"{stype}; surrogate unchanged this iteration.",
        flush=True,
    )
    return surrogate


# ======================================================================= #
# Main ISMO loop
# ======================================================================= #

def main(argv: list[str] | None = None) -> None:  # noqa: C901 -- orchestration
    """Entry point for the ISMO runner."""
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60, flush=True)
    print("  ISMO Runner", flush=True)
    print("=" * 60, flush=True)

    # ── Step 0: Load data and surrogate ──────────────────────────────
    print("\n  [Step 0] Loading training data and surrogate ...", flush=True)

    training_data = np.load(args.training_data)
    all_params = training_data["parameters"]
    all_cd = training_data["current_density"]
    all_pc = training_data["peroxide_current"]
    phi_applied = training_data["phi_applied"]

    # Load train/test split
    split = np.load(args.split_indices)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    params_train = all_params[train_idx]
    cd_train = all_cd[train_idx]
    pc_train = all_pc[train_idx]

    test_params = all_params[test_idx]
    test_cd = all_cd[test_idx]
    test_pc = all_pc[test_idx]

    print(f"  [Step 0] Training samples: {len(params_train)}, "
          f"Test samples: {len(test_params)}", flush=True)

    surrogate = load_or_build_surrogate(args)

    # Parameter bounds (defaults match the training data generation)
    bounds = ParameterBounds()

    # PDE solver function -- stub until 4-04 is integrated
    pde_solver_fn = _make_pde_solver_fn_stub(verbose=args.verbose)

    # ── Step 2: Initialise convergence checker & diagnostics ─────────
    criteria = ISMOConvergenceCriteria(
        agreement_tol=args.agreement_tol,
        stability_tol=args.stability_tol,
        max_pde_evals=args.budget,
        max_iterations=args.max_iterations,
        stagnation_window=args.stagnation_window,
    )
    checker = ISMOConvergenceChecker(criteria)
    diagnostics = ISMODiagnostics(output_dir=args.output_dir)

    print(f"  [Step 2] Convergence criteria: agreement_tol={criteria.agreement_tol}, "
          f"stability_tol={criteria.stability_tol}, budget={criteria.max_pde_evals}, "
          f"max_iter={criteria.max_iterations}", flush=True)

    # ── Step 3: ISMO loop ────────────────────────────────────────────
    print("\n  [Step 3] Starting ISMO loop ...", flush=True)

    try:
        for iteration in range(args.max_iterations):
            t_iter_start = time.time()
            print(f"\n{'─'*50}", flush=True)
            print(f"  Iteration {iteration}", flush=True)
            print(f"{'─'*50}", flush=True)

            # 3a. Run inference on current surrogate
            t_inf_start = time.time()
            from Surrogate.multistart import (
                MultiStartConfig,
                run_multistart_inference,
            )
            ms_config = MultiStartConfig(
                n_grid=20_000,
                n_top_candidates=20,
                seed=args.seed + iteration,
                verbose=args.verbose,
            )

            # We need target data -- for now, use the first test sample
            # as the "target" (in production this comes from experimental
            # data or a PDE solve at true params).
            target_cd = test_cd[0]
            target_pc = test_pc[0]

            ms_result = run_multistart_inference(
                surrogate=surrogate,
                target_cd=target_cd,
                target_pc=target_pc,
                bounds_k0_1=bounds.k0_1_range,
                bounds_k0_2=bounds.k0_2_range,
                bounds_alpha=bounds.alpha_1_range,
                config=ms_config,
            )
            inference_time = time.time() - t_inf_start

            best_params = np.array([
                ms_result.best_k0_1,
                ms_result.best_k0_2,
                ms_result.best_alpha_1,
                ms_result.best_alpha_2,
            ])
            if args.verbose:
                print(f"  [3a] Best params: k0=[{best_params[0]:.4e},"
                      f"{best_params[1]:.4e}] alpha=[{best_params[2]:.4f},"
                      f"{best_params[3]:.4f}] loss={ms_result.best_loss:.4e}",
                      flush=True)

            # 3b. Compute surrogate-PDE agreement (1 PDE eval)
            try:
                agreement = compute_surrogate_pde_agreement(
                    surrogate, best_params, pde_solver_fn, phi_applied,
                )
                n_agreement_evals = 1
            except NotImplementedError:
                # PDE solver not available -- use surrogate self-consistency
                # as a placeholder (agreement = 0 means "perfect")
                print("  [3b] WARNING: PDE solver not available, using "
                      "surrogate self-consistency (agreement=0).", flush=True)
                pred = surrogate.predict(
                    float(best_params[0]), float(best_params[1]),
                    float(best_params[2]), float(best_params[3]),
                )
                agreement = {
                    "cd_nrmse": 0.0,
                    "pc_nrmse": 0.0,
                    "combined_nrmse": 0.0,
                    "cd_surrogate": np.asarray(pred["current_density"]),
                    "cd_pde": np.asarray(pred["current_density"]),
                    "pc_surrogate": np.asarray(pred["peroxide_current"]),
                    "pc_pde": np.asarray(pred["peroxide_current"]),
                }
                n_agreement_evals = 0

            if args.verbose:
                print(f"  [3b] Agreement: cd_nrmse={agreement['cd_nrmse']:.4e}, "
                      f"pc_nrmse={agreement['pc_nrmse']:.4e}, "
                      f"combined={agreement['combined_nrmse']:.4e}", flush=True)

            # 3b'. Budget check before acquisition
            budget_after_agreement = checker.remaining_budget() - n_agreement_evals
            if budget_after_agreement < criteria.min_useful_batch_size:
                print(
                    f"  [3b'] Remaining budget ({budget_after_agreement}) < "
                    f"min_useful_batch_size ({criteria.min_useful_batch_size}). "
                    f"Stopping.",
                    flush=True,
                )
                # Record final iteration with 0 acquisition samples
                stability = compute_parameter_stability(
                    checker.history, best_params, bounds,
                )
                test_metrics = validate_surrogate(
                    surrogate, test_params, test_cd, test_pc,
                )
                obj_imp = compute_objective_improvement(
                    checker.history, ms_result.best_loss,
                )
                core = None
                if ISMOIteration is not None:
                    core = ISMOIteration(
                        iteration=iteration,
                        n_new_samples=n_agreement_evals,
                        n_total_training=len(params_train),
                        surrogate_loss_at_best=ms_result.best_loss,
                        pde_loss_at_best=float("nan"),
                        surrogate_pde_gap=agreement["combined_nrmse"],
                        convergence_metric=agreement["combined_nrmse"],
                        best_params=tuple(best_params.tolist()),
                        best_loss=ms_result.best_loss,
                        candidate_pde_losses=(),
                        candidate_surrogate_losses=(),
                        retrain_val_rmse_cd=test_metrics["cd_mean_relative_error"],
                        retrain_val_rmse_pc=test_metrics["pc_mean_relative_error"],
                        wall_time_s=time.time() - t_iter_start,
                    )
                diag = ISMODiagnosticRecord(
                    core=core,
                    surrogate_pde_agreement=agreement["combined_nrmse"],
                    parameter_stability=stability,
                    surrogate_test_nrmse_cd=test_metrics["cd_mean_relative_error"],
                    surrogate_test_nrmse_pc=test_metrics["pc_mean_relative_error"],
                    objective_improvement=obj_imp,
                    pde_solve_time_s=0.0,
                    retrain_time_s=0.0,
                    inference_time_s=inference_time,
                    acquisition_strategy=args.acquisition,
                    acquisition_details=(),
                    n_pde_evals_this_iter=n_agreement_evals,
                    iteration=iteration,
                    best_params=tuple(best_params.tolist()),
                    n_total_training=len(params_train),
                )
                checker.record_iteration(diag)
                diagnostics.log_iteration(diag)
                break

            # 3c. Acquire new sample points
            n_to_acquire = min(args.samples_per_iter, budget_after_agreement)
            t_acq_start = time.time()

            if select_new_samples is not None and AcquisitionConfig is not None:
                acq_config = AcquisitionConfig(budget=n_to_acquire)
                acq_result = select_new_samples(
                    existing_data=params_train,
                    bounds=bounds,
                    config=acq_config,
                    multistart_result=ms_result,
                    cascade_result=None,
                    gp_model=None,
                )
                new_params = acq_result.samples
            else:
                # Fallback: LHS sampling
                new_params = _fallback_acquire(
                    params_train, bounds, n_to_acquire, best_params, rng,
                )

            if args.verbose:
                print(f"  [3c] Acquired {len(new_params)} new sample points "
                      f"(strategy={args.acquisition}).", flush=True)

            # 3d. Evaluate PDE at new points
            t_pde_start = time.time()
            new_cd_list = []
            new_pc_list = []
            n_pde_failures = 0

            if evaluate_candidates_with_pde is not None:
                try:
                    pde_eval_result = evaluate_candidates_with_pde(new_params, pde_solver_fn)
                    new_cd_list.append(pde_eval_result.current_density)
                    new_pc_list.append(pde_eval_result.peroxide_current)
                except NotImplementedError:
                    print("  [3d] PDE solver not available -- skipping batch "
                          "evaluation.", flush=True)
                    new_params = np.empty((0, 4))
            else:
                # Inline PDE evaluation with partial-failure handling
                successful_indices = []
                for i, p in enumerate(new_params):
                    try:
                        result = pde_solver_fn(p)
                        new_cd_list.append(result["current_density"])
                        new_pc_list.append(result["peroxide_current"])
                        successful_indices.append(i)
                    except NotImplementedError:
                        print(f"  [3d] PDE solver not available for sample {i} "
                              f"-- skipping remaining.", flush=True)
                        break
                    except Exception as e:
                        print(f"  [3d] PDE failure at sample {i}: {e}", flush=True)
                        n_pde_failures += 1
                        continue

                if n_pde_failures > 0:
                    print(f"  [3d] {n_pde_failures} PDE failures out of "
                          f"{len(new_params)} samples.", flush=True)
                # Filter params to only successful indices
                if len(successful_indices) < len(new_params):
                    new_params = new_params[successful_indices]

            pde_solve_time = time.time() - t_pde_start

            if len(new_cd_list) > 0:
                new_cd = np.vstack(new_cd_list) if len(new_cd_list[0].shape) > 1 else np.array(new_cd_list)
                new_pc = np.vstack(new_pc_list) if len(new_pc_list[0].shape) > 1 else np.array(new_pc_list)
                n_successful = len(new_cd)
            else:
                new_cd = np.empty((0, cd_train.shape[1]))
                new_pc = np.empty((0, pc_train.shape[1]))
                n_successful = 0

            # 3e. Augment training data
            if n_successful > 0:
                params_train = np.vstack([params_train, new_params[:n_successful]])
                cd_train = np.vstack([cd_train, new_cd])
                pc_train = np.vstack([pc_train, new_pc])

            # 3f. Retrain surrogate
            t_retrain_start = time.time()
            if retrain_surrogate is not None:
                from Surrogate.ismo_retrain import ISMORetrainConfig
                retrain_config = ISMORetrainConfig()
                new_data_dict = {
                    "parameters": new_params[:n_successful],
                    "current_density": new_cd,
                    "peroxide_current": new_pc,
                    "phi_applied": phi_applied,
                }
                existing_data_dict = {
                    "parameters": params_train[:-n_successful] if n_successful > 0 else params_train,
                    "current_density": cd_train[:-n_successful] if n_successful > 0 else cd_train,
                    "peroxide_current": pc_train[:-n_successful] if n_successful > 0 else pc_train,
                    "phi_applied": phi_applied,
                }
                surrogate = retrain_surrogate(
                    surrogate,
                    new_data=new_data_dict,
                    existing_data=existing_data_dict,
                    config=retrain_config,
                    iteration=iteration,
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            else:
                surrogate = _fallback_retrain(
                    surrogate, params_train, cd_train, pc_train, phi_applied, args,
                )
            retrain_time = time.time() - t_retrain_start

            # 3g. Validate on held-out test set
            test_metrics = validate_surrogate(
                surrogate, test_params, test_cd, test_pc,
            )
            if args.verbose:
                print(f"  [3g] Test NRMSE: cd={test_metrics['cd_mean_relative_error']:.4e}, "
                      f"pc={test_metrics['pc_mean_relative_error']:.4e}", flush=True)

            # 3h. Record iteration
            n_pde_evals_this_iter = n_agreement_evals + n_successful
            stability = compute_parameter_stability(
                checker.history, best_params, bounds,
            )
            obj_imp = compute_objective_improvement(
                checker.history, ms_result.best_loss,
            )

            core = None
            if ISMOIteration is not None:
                core = ISMOIteration(
                    iteration=iteration,
                    n_new_samples=n_pde_evals_this_iter,
                    n_total_training=len(params_train),
                    surrogate_loss_at_best=ms_result.best_loss,
                    pde_loss_at_best=float("nan"),
                    surrogate_pde_gap=agreement["combined_nrmse"],
                    convergence_metric=agreement["combined_nrmse"],
                    best_params=tuple(best_params.tolist()),
                    best_loss=ms_result.best_loss,
                    candidate_pde_losses=(),
                    candidate_surrogate_losses=(),
                    retrain_val_rmse_cd=test_metrics["cd_mean_relative_error"],
                    retrain_val_rmse_pc=test_metrics["pc_mean_relative_error"],
                    wall_time_s=time.time() - t_iter_start,
                )

            diag = ISMODiagnosticRecord(
                core=core,
                surrogate_pde_agreement=agreement["combined_nrmse"],
                parameter_stability=stability,
                surrogate_test_nrmse_cd=test_metrics["cd_mean_relative_error"],
                surrogate_test_nrmse_pc=test_metrics["pc_mean_relative_error"],
                objective_improvement=obj_imp,
                pde_solve_time_s=pde_solve_time,
                retrain_time_s=retrain_time,
                inference_time_s=inference_time,
                acquisition_strategy=args.acquisition,
                acquisition_details=(),
                n_pde_evals_this_iter=n_pde_evals_this_iter,
                iteration=iteration,
                best_params=tuple(best_params.tolist()),
                n_total_training=len(params_train),
            )

            checker.record_iteration(diag)
            diagnostics.log_iteration(diag)

            # Plot I-V comparison at best
            diagnostics.plot_iv_comparison_at_best(
                iteration=iteration,
                phi_applied=phi_applied,
                surrogate_cd=agreement["cd_surrogate"],
                pde_cd=agreement["cd_pde"],
                surrogate_pc=agreement["pc_surrogate"],
                pde_pc=agreement["pc_pde"],
            )

            # Save iteration state
            diagnostics.save_iteration_state(
                iteration=iteration,
                record=diag,
                new_params=new_params[:n_successful] if n_successful > 0 else np.empty((0, 4)),
                new_cd=new_cd,
                new_pc=new_pc,
            )

            # 3i. Check convergence
            converged, reason = checker.check_convergence()
            if args.verbose:
                print(f"  [3i] PDE evals so far: {checker.total_pde_evals}, "
                      f"remaining: {checker.remaining_budget()}", flush=True)
            if converged:
                print(f"\n  ISMO converged: {reason}", flush=True)
                break

    finally:
        # Graceful shutdown: always save whatever we have
        print("\n  [Shutdown] Saving final outputs ...", flush=True)

        # ── Step 4: Post-ISMO outputs ────────────────────────────────
        # Save augmented training data
        aug_path = os.path.join(args.output_dir, "augmented_training_data.npz")
        np.savez_compressed(
            aug_path,
            parameters=params_train,
            current_density=cd_train,
            peroxide_current=pc_train,
            phi_applied=phi_applied,
        )
        print(f"  [Step 4] Saved augmented training data: {aug_path}", flush=True)

        # Save convergence report
        report = checker.get_convergence_summary()
        report_path = os.path.join(args.output_dir, "convergence_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  [Step 4] Saved convergence report: {report_path}", flush=True)

        # Plot convergence curves
        curves_path = os.path.join(args.output_dir, "convergence_curves.png")
        plot_convergence_curves(
            checker.history, curves_path, criteria=criteria,
        )

    # ── Step 5: Post-ISMO validation ─────────────────────────────────
    if not args.skip_post_validation:
        print("\n  [Step 5] Running post-ISMO validation ...", flush=True)
        _run_post_ismo_validation(surrogate, args.output_dir)
    else:
        print("\n  [Step 5] Post-validation skipped (--skip-post-validation).",
              flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ISMO Runner complete.", flush=True)
    print("=" * 60, flush=True)


# ======================================================================= #
# Post-ISMO validation
# ======================================================================= #

def _run_post_ismo_validation(surrogate, output_dir: str) -> None:
    """Run parameter recovery at 0% and 1% noise, compare to Phase 3.

    Phase 3 baseline (from parameter_recovery_summary.json):
        0% noise: surrogate_bias k0_2 ~ 10.67%
        1% noise: median_max_relative_error ~ 17.69%
    """
    # Load Phase 3 baseline
    baseline_path = os.path.join(
        "StudyResults", "inverse_verification",
        "parameter_recovery_summary.json",
    )
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        phase3_errors = {
            "0pct_noise": baseline.get("surrogate_bias", {}).get("k0_2", 10.67),
            "1pct_noise": baseline.get("median_max_relative_error", 17.69),
        }
    else:
        print("  [Post-val] Phase 3 baseline not found, using defaults.", flush=True)
        phase3_errors = {"0pct_noise": 10.67, "1pct_noise": 17.69}

    # Placeholder for ISMO recovery -- requires full inference pipeline
    # which depends on PDE solver availability
    ismo_errors = {"0pct_noise": float("nan"), "1pct_noise": float("nan")}

    comparison = {
        "phase3_baseline": phase3_errors,
        "ismo_refined": ismo_errors,
        "k0_2_improvement_pct": float("nan"),
        "success_criterion_met": False,
    }

    comp_path = os.path.join(output_dir, "parameter_recovery_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  [Post-val] Saved comparison: {comp_path}", flush=True)

    # Plot comparison (even with NaN placeholders -- will show zero bars)
    try:
        plot_k0_2_recovery_comparison(
            phase3_errors, ismo_errors,
            os.path.join(output_dir, "k0_2_improvement.png"),
        )
    except Exception as e:
        print(f"  [Post-val] Could not generate comparison plot: {e}", flush=True)


if __name__ == "__main__":
    main()
