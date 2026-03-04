"""End-to-end PDE tests for the v11 surrogate-warm-started inference pipeline.

Tests exercise the full v11 pipeline: surrogate Phases 1-2, PDE Phases 3-4,
and variant configurations (different seeds, noise levels, true parameters).

All tests are marked @pytest.mark.slow and require Firedrake to be available.
Module-scoped fixtures share expensive PDE resources across tests.

Reference values from:
    StudyResults/master_inference_v11/master_comparison_v11.csv
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Ensure PNPInverse root is importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.conftest import FIREDRAKE_AVAILABLE, skip_without_firedrake

# Import v11 helpers
sys.path.insert(0, os.path.join(_ROOT, "scripts", "surrogate"))
from Infer_BVMaster_charged_v11_surrogate_pde import (
    _compute_errors,
    _subset_targets,
)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1,
    K0_HAT_R2,
    ALPHA_R1,
    ALPHA_R2,
    I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
)

from Surrogate.io import load_surrogate
from Surrogate.objectives import AlphaOnlySurrogateObjective

setup_firedrake_env()

# ---------------------------------------------------------------------------
# Markers: all tests in this module are slow and need Firedrake
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.slow,
    skip_without_firedrake,
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

SURROGATE_MODEL_PATH = os.path.join(
    _ROOT, "StudyResults", "surrogate_v9", "surrogate_model.pkl"
)

# Voltage grids (identical to v11 script)
ETA_SYMMETRIC = np.array([
    +5.0, +3.0, +1.0, -0.5,
    -1.0, -2.0, -3.0, -5.0, -8.0,
    -10.0, -15.0, -20.0,
])
ETA_SHALLOW = np.array([
    -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
    -10.0, -11.5, -13.0,
])
ETA_CATHODIC = np.array([
    -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
    -10.0, -13.0, -17.0, -22.0, -28.0,
    -35.0, -41.0, -46.5,
])
ALL_ETA = np.sort(
    np.unique(np.concatenate([ETA_SYMMETRIC, ETA_SHALLOW, ETA_CATHODIC]))
)[::-1]


# ---------------------------------------------------------------------------
# SubsetSurrogateObjective (re-defined since it's nested in main())
# ---------------------------------------------------------------------------
class SubsetSurrogateObjective:
    """Surrogate objective on a subset of voltage points (from v9/v11)."""

    def __init__(self, surrogate, target_cd, target_pc, subset_idx,
                 secondary_weight=1.0, fd_step=1e-5, log_space_k0=True):
        self.surrogate = surrogate
        self.target_cd = np.asarray(target_cd, dtype=float)
        self.target_pc = np.asarray(target_pc, dtype=float)
        self.subset_idx = subset_idx
        self._valid_cd = ~np.isnan(self.target_cd)
        self._valid_pc = ~np.isnan(self.target_pc)
        self.secondary_weight = secondary_weight
        self.fd_step = fd_step
        self.log_space_k0 = log_space_k0
        self._n_evals = 0

    def _x_to_params(self, x):
        x = np.asarray(x, dtype=float)
        if self.log_space_k0:
            k0_1, k0_2 = 10.0 ** x[0], 10.0 ** x[1]
        else:
            k0_1, k0_2 = x[0], x[1]
        return k0_1, k0_2, x[2], x[3]

    def objective(self, x):
        k0_1, k0_2, a1, a2 = self._x_to_params(x)
        pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
        cd_sim = pred["current_density"][self.subset_idx]
        pc_sim = pred["peroxide_current"][self.subset_idx]
        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]
        J_cd = 0.5 * np.sum(cd_diff ** 2)
        J_pc = 0.5 * np.sum(pc_diff ** 2)
        self._n_evals += 1
        return float(J_cd + self.secondary_weight * J_pc)

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        grad = np.zeros(len(x), dtype=float)
        h = self.fd_step
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (self.objective(xp) - self.objective(xm)) / (2 * h)
        return grad


# ---------------------------------------------------------------------------
# Helper: generate PDE targets (mirrors v11 _generate_targets_with_pde)
# ---------------------------------------------------------------------------
def _generate_targets_with_pde(
    phi_applied_values, observable_scale, noise_percent, noise_seed,
    *,
    k0_hat=K0_HAT,
    k0_2_hat=K0_2_HAT,
    alpha_1=ALPHA_1,
    alpha_2=ALPHA_2,
):
    """Generate target I-V curves using the PDE solver at given parameters."""
    from Forward.steady_state import SteadyStateConfig, add_percent_noise
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

    results = {}
    for obs_mode in ["current_density", "peroxide_current"]:
        _clear_caches()
        seed_offset = 0 if obs_mode == "current_density" else 1

        points = solve_bv_curve_points_with_warmstart(
            base_solver_params=base_sp,
            steady=steady,
            phi_applied_values=phi_applied_values,
            target_flux=dummy_target,
            k0_values=[k0_hat, k0_2_hat],
            blob_initial_condition=False,
            fail_penalty=1e9,
            forward_recovery=recovery,
            observable_mode=obs_mode,
            observable_reaction_index=None,
            observable_scale=observable_scale,
            mesh=mesh,
            alpha_values=[alpha_1, alpha_2],
            control_mode="joint",
            max_eta_gap=3.0,
        )

        clean_flux = np.array([float(p.simulated_flux) for p in points], dtype=float)
        if noise_percent > 0:
            noisy_flux = add_percent_noise(
                clean_flux, noise_percent, seed=noise_seed + seed_offset
            )
        else:
            noisy_flux = clean_flux.copy()
        results[obs_mode] = noisy_flux

    _clear_caches()
    return results


# ---------------------------------------------------------------------------
# Helper: run surrogate Phases 1-2
# ---------------------------------------------------------------------------
def _run_surrogate_phases_1_2(surrogate, target_cd, target_pc):
    """Run surrogate Phases 1-2, return dict with results."""
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    # Get training bounds from model
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        K0_1_TRAIN_LO = tb["k0_1"][0]
        K0_1_TRAIN_HI = tb["k0_1"][1]
        K0_2_TRAIN_LO = tb["k0_2"][0]
        K0_2_TRAIN_HI = tb["k0_2"][1]
        ALPHA_TRAIN_LO = min(tb["alpha_1"][0], tb["alpha_2"][0])
        ALPHA_TRAIN_HI = max(tb["alpha_1"][1], tb["alpha_2"][1])
    else:
        K0_1_TRAIN_LO = K0_HAT * 0.01
        K0_1_TRAIN_HI = K0_HAT * 100.0
        K0_2_TRAIN_LO = K0_2_HAT * 0.01
        K0_2_TRAIN_HI = K0_2_HAT * 100.0
        ALPHA_TRAIN_LO = 0.10
        ALPHA_TRAIN_HI = 0.90

    # Phase 1: alpha-only
    p1_obj = AlphaOnlySurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd,
        target_pc=target_pc,
        fixed_k0=initial_k0_guess,
        secondary_weight=1.0,
        fd_step=1e-5,
    )

    x0_p1 = np.array(initial_alpha_guess, dtype=float)
    bounds_p1 = [(ALPHA_TRAIN_LO, ALPHA_TRAIN_HI)] * 2

    result_p1 = minimize(
        p1_obj.objective, x0_p1, jac=p1_obj.gradient,
        method="L-BFGS-B", bounds=bounds_p1,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
    )
    p1_alpha = result_p1.x.copy()
    p1_k0 = np.asarray(initial_k0_guess)
    p1_loss = float(result_p1.fun)

    # Phase 2: joint 4-param on shallow subset
    target_cd_shallow, target_pc_shallow = _subset_targets(
        target_cd, target_pc, ALL_ETA, ETA_SHALLOW,
    )
    shallow_idx = []
    for eta in ETA_SHALLOW:
        matches = np.where(np.abs(ALL_ETA - eta) < 1e-10)[0]
        if len(matches) > 0:
            shallow_idx.append(matches[0])
    shallow_idx = np.array(shallow_idx, dtype=int)

    p2_obj = SubsetSurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_shallow,
        target_pc=target_pc_shallow,
        subset_idx=shallow_idx,
        secondary_weight=1.0,
        fd_step=1e-5,
        log_space_k0=True,
    )

    x0_p2 = np.array([
        np.log10(initial_k0_guess[0]),
        np.log10(initial_k0_guess[1]),
        p1_alpha[0],
        p1_alpha[1],
    ], dtype=float)
    bounds_p2 = [
        (np.log10(K0_1_TRAIN_LO), np.log10(K0_1_TRAIN_HI)),
        (np.log10(K0_2_TRAIN_LO), np.log10(K0_2_TRAIN_HI)),
        (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
        (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
    ]

    result_p2 = minimize(
        p2_obj.objective, x0_p2, jac=p2_obj.gradient,
        method="L-BFGS-B", bounds=bounds_p2,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
    )
    p2_k0 = np.array([10.0 ** result_p2.x[0], 10.0 ** result_p2.x[1]])
    p2_alpha = result_p2.x[2:4].copy()
    p2_loss = float(result_p2.fun)

    return {
        "p1_k0": p1_k0,
        "p1_alpha": p1_alpha,
        "p1_loss": p1_loss,
        "p2_k0": p2_k0,
        "p2_alpha": p2_alpha,
        "p2_loss": p2_loss,
    }


# ---------------------------------------------------------------------------
# Helper: run PDE Phases 3-4
# ---------------------------------------------------------------------------
def _run_pde_phases_3_4(
    surr_best_k0,
    surr_best_alpha,
    *,
    true_k0,
    true_alpha,
    observable_scale,
    noise_percent=2.0,
    noise_seed=20260226,
    p3_maxiter=30,
    p4_maxiter=25,
    n_workers=1,
):
    """Run PDE Phases 3-4 warm-started from surrogate results."""
    from Forward.steady_state import SteadyStateConfig
    from Forward.bv_solver import make_graded_rectangle_mesh
    from FluxCurve import (
        BVFluxCurveInferenceRequest,
        run_bv_multi_observable_flux_curve_inference,
    )
    from FluxCurve.bv_point_solve import (
        _clear_caches,
        set_parallel_pool,
        close_parallel_pool,
    )
    from FluxCurve.bv_parallel import BVPointSolvePool, BVParallelPointConfig
    from FluxCurve.bv_point_solve import (
        _WARMSTART_MAX_STEPS,
        _SER_GROWTH_CAP,
        _SER_SHRINK,
        _SER_DT_MAX_RATIO,
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

    recovery = make_recovery_config(max_it_cap=600)
    n_joint_controls = 4

    # Create shared parallel pool
    shared_config = BVParallelPointConfig(
        base_solver_params=list(base_sp),
        ss_relative_tolerance=float(steady.relative_tolerance),
        ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
        ss_consecutive_steps=int(steady.consecutive_steps),
        ss_max_steps=int(steady.max_steps),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        blob_initial_condition=False,
        fail_penalty=1e9,
        warmstart_max_steps=_WARMSTART_MAX_STEPS,
        observable_mode="current_density",
        observable_reaction_index=None,
        observable_scale=observable_scale,
        control_mode="joint",
        n_controls=n_joint_controls,
        ser_growth_cap=_SER_GROWTH_CAP,
        ser_shrink=_SER_SHRINK,
        ser_dt_max_ratio=_SER_DT_MAX_RATIO,
        secondary_observable_mode="peroxide_current",
        secondary_observable_reaction_index=None,
        secondary_observable_scale=observable_scale,
    )
    shared_pool = BVPointSolvePool(shared_config, n_workers=n_workers)
    set_parallel_pool(shared_pool)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Phase 3: PDE Joint on SHALLOW cathodic
            _clear_caches()
            p3_dir = os.path.join(tmpdir, "phase3_pde_shallow")

            request_p3 = BVFluxCurveInferenceRequest(
                base_solver_params=base_sp,
                steady=steady,
                true_k0=true_k0,
                initial_guess=surr_best_k0.tolist(),
                phi_applied_values=ETA_SHALLOW.tolist(),
                target_csv_path=os.path.join(p3_dir, "target_primary.csv"),
                output_dir=p3_dir,
                regenerate_target=True,
                target_noise_percent=noise_percent,
                target_seed=noise_seed,
                observable_mode="current_density",
                current_density_scale=observable_scale,
                observable_label="current density (mA/cm2)",
                observable_title="Phase 3: PDE shallow (warm from surrogate)",
                secondary_observable_mode="peroxide_current",
                secondary_observable_weight=1.0,
                secondary_current_density_scale=observable_scale,
                secondary_target_csv_path=os.path.join(p3_dir, "target_peroxide.csv"),
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
                    "maxiter": p3_maxiter,
                    "ftol": 1e-8,
                    "gtol": 5e-6,
                    "disp": True,
                },
                max_iters=p3_maxiter,
                live_plot=False,
                forward_recovery=recovery,
                parallel_fast_path=True,
                parallel_workers=n_workers,
            )

            result_p3 = run_bv_multi_observable_flux_curve_inference(request_p3)
            p3_k0 = np.asarray(result_p3["best_k0"])
            p3_alpha = np.asarray(result_p3["best_alpha"])
            p3_loss = float(result_p3["best_loss"])

            # Phase 4: PDE Joint on FULL CATHODIC
            _clear_caches()
            p4_dir = os.path.join(tmpdir, "phase4_pde_full_cathodic")

            request_p4 = BVFluxCurveInferenceRequest(
                base_solver_params=base_sp,
                steady=steady,
                true_k0=true_k0,
                initial_guess=p3_k0.tolist(),
                phi_applied_values=ETA_CATHODIC.tolist(),
                target_csv_path=os.path.join(p4_dir, "target_primary.csv"),
                output_dir=p4_dir,
                regenerate_target=True,
                target_noise_percent=noise_percent,
                target_seed=noise_seed,
                observable_mode="current_density",
                current_density_scale=observable_scale,
                observable_label="current density (mA/cm2)",
                observable_title="Phase 4: PDE full cathodic (warm from P3)",
                secondary_observable_mode="peroxide_current",
                secondary_observable_weight=1.0,
                secondary_current_density_scale=observable_scale,
                secondary_target_csv_path=os.path.join(p4_dir, "target_peroxide.csv"),
                control_mode="joint",
                true_alpha=true_alpha,
                initial_alpha_guess=p3_alpha.tolist(),
                alpha_lower=0.05, alpha_upper=0.95,
                k0_lower=1e-8, k0_upper=100.0,
                log_space=True,
                mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
                max_eta_gap=3.0,
                optimizer_method="L-BFGS-B",
                optimizer_options={
                    "maxiter": p4_maxiter,
                    "ftol": 1e-8,
                    "gtol": 5e-6,
                    "disp": True,
                },
                max_iters=p4_maxiter,
                live_plot=False,
                forward_recovery=recovery,
                parallel_fast_path=True,
                parallel_workers=n_workers,
            )

            result_p4 = run_bv_multi_observable_flux_curve_inference(request_p4)
            p4_k0 = np.asarray(result_p4["best_k0"])
            p4_alpha = np.asarray(result_p4["best_alpha"])
            p4_loss = float(result_p4["best_loss"])
        finally:
            close_parallel_pool()
            _clear_caches()

    return {
        "p3_k0": p3_k0,
        "p3_alpha": p3_alpha,
        "p3_loss": p3_loss,
        "p4_k0": p4_k0,
        "p4_alpha": p4_alpha,
        "p4_loss": p4_loss,
    }


# ---------------------------------------------------------------------------
# Helper: full pipeline (targets + surr P1-P2 + PDE P3-P4)
# ---------------------------------------------------------------------------
def _run_full_pipeline(
    *,
    noise_percent=2.0,
    noise_seed=20260226,
    p3_maxiter=30,
    p4_maxiter=25,
    n_workers=1,
    k0_hat=K0_HAT,
    k0_2_hat=K0_2_HAT,
    alpha_1=ALPHA_1,
    alpha_2=ALPHA_2,
):
    """Run the complete v11 pipeline: targets + surrogate P1-P2 + PDE P3-P4."""
    true_k0 = [k0_hat, k0_2_hat]
    true_alpha = [alpha_1, alpha_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)
    observable_scale = -I_SCALE

    # Load surrogate
    surrogate = load_surrogate(SURROGATE_MODEL_PATH)

    # Generate targets
    targets = _generate_targets_with_pde(
        ALL_ETA, observable_scale, noise_percent, noise_seed,
        k0_hat=k0_hat, k0_2_hat=k0_2_hat,
        alpha_1=alpha_1, alpha_2=alpha_2,
    )
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]

    # Surrogate Phases 1-2
    surr = _run_surrogate_phases_1_2(surrogate, target_cd_full, target_pc_full)

    # PDE Phases 3-4
    pde = _run_pde_phases_3_4(
        surr["p2_k0"], surr["p2_alpha"],
        true_k0=true_k0,
        true_alpha=true_alpha,
        observable_scale=observable_scale,
        noise_percent=noise_percent,
        noise_seed=noise_seed,
        p3_maxiter=p3_maxiter,
        p4_maxiter=p4_maxiter,
        n_workers=n_workers,
    )

    # Compute errors
    def _errs(k0, alpha):
        k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
        return {
            "k0_1_err": float(k0_err[0]),
            "k0_2_err": float(k0_err[1]),
            "alpha_1_err": float(alpha_err[0]),
            "alpha_2_err": float(alpha_err[1]),
            "max_err": float(max(k0_err.max(), alpha_err.max())),
        }

    # Best-of selection
    p3_errs = _errs(pde["p3_k0"], pde["p3_alpha"])
    p4_errs = _errs(pde["p4_k0"], pde["p4_alpha"])

    if p4_errs["max_err"] <= p3_errs["max_err"]:
        best_source = "Phase 4"
        best_errs = p4_errs
    else:
        best_source = "Phase 3"
        best_errs = p3_errs

    return {
        "surrogate": surr,
        "pde": pde,
        "p1_errs": _errs(surr["p1_k0"], surr["p1_alpha"]),
        "p2_errs": _errs(surr["p2_k0"], surr["p2_alpha"]),
        "p3_errs": p3_errs,
        "p4_errs": p4_errs,
        "best_source": best_source,
        "best_errs": best_errs,
        "true_k0_arr": true_k0_arr,
        "true_alpha_arr": true_alpha_arr,
    }


# ===================================================================
# Module-scoped fixtures
# ===================================================================

@pytest.fixture(scope="module")
def surrogate_model():
    """Load the surrogate_v9 model (shared across all tests in module)."""
    return load_surrogate(SURROGATE_MODEL_PATH)


@pytest.fixture(scope="module")
def reference_true_params():
    """True k0/alpha from _bv_common."""
    return {
        "k0": np.array([K0_HAT, K0_2_HAT]),
        "alpha": np.array([ALPHA_1, ALPHA_2]),
    }


@pytest.fixture(scope="module")
def pde_solver_infrastructure():
    """Mesh, SteadyStateConfig, base SolverParams for PDE tests."""
    from Forward.steady_state import SteadyStateConfig
    from Forward.bv_solver import make_graded_rectangle_mesh

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

    return {
        "base_sp": base_sp,
        "steady": steady,
        "mesh": mesh,
        "dt": dt,
        "max_ss_steps": max_ss_steps,
    }


@pytest.fixture(scope="module")
def reference_targets(pde_solver_infrastructure):
    """PDE-generated noisy targets at true params (noise=2%, seed=20260226).

    This is expensive (~120s) but shared across all reference tests.
    """
    observable_scale = -I_SCALE
    targets = _generate_targets_with_pde(
        ALL_ETA, observable_scale,
        noise_percent=2.0, noise_seed=20260226,
    )
    return {
        "target_cd": targets["current_density"],
        "target_pc": targets["peroxide_current"],
        "observable_scale": observable_scale,
    }


@pytest.fixture(scope="module")
def surrogate_p1p2_result(surrogate_model, reference_targets):
    """Surrogate Phases 1-2 result (~0.3s)."""
    return _run_surrogate_phases_1_2(
        surrogate_model,
        reference_targets["target_cd"],
        reference_targets["target_pc"],
    )


@pytest.fixture(scope="module")
def pde_p3p4_result(surrogate_p1p2_result, reference_targets):
    """PDE Phases 3-4 warm-started from surrogate (~300-600s)."""
    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    observable_scale = reference_targets["observable_scale"]

    return _run_pde_phases_3_4(
        surrogate_p1p2_result["p2_k0"],
        surrogate_p1p2_result["p2_alpha"],
        true_k0=true_k0,
        true_alpha=true_alpha,
        observable_scale=observable_scale,
        noise_percent=2.0,
        noise_seed=20260226,
        p3_maxiter=30,
        p4_maxiter=25,
        n_workers=1,
    )


# ===================================================================
# TestReferenceReproduction
# ===================================================================

class TestReferenceReproduction:
    """Verify pipeline reproduces reference values from master_comparison_v11.csv.

    Reference:
        P1: k0=[5e-3, 5e-4], alpha=[0.3525, 0.1762]
        P2: k0=[1.374e-3, 4.865e-5], alpha=[0.5972, 0.4682]
        P3: k0=[1.471e-3, 4.643e-5], alpha=[0.5734, 0.4627]
        P4: k0=[1.382e-3, 5.220e-5], alpha=[0.5948, 0.4608]
    """

    def test_phase1_surrogate_alpha_reproduces_reference(
        self, surrogate_p1p2_result, reference_true_params,
    ):
        """P1 alpha is approximately [0.3525, 0.1762]."""
        p1_alpha = surrogate_p1p2_result["p1_alpha"]
        # P1 k0 is fixed at initial guess [0.005, 0.0005]
        np.testing.assert_allclose(
            surrogate_p1p2_result["p1_k0"],
            [0.005, 0.0005],
            rtol=1e-10,
            err_msg="P1 k0 should be the initial guess (fixed)",
        )
        np.testing.assert_allclose(
            p1_alpha, [0.3525, 0.1762],
            rtol=0.05,
            err_msg="P1 alpha should match reference within 5%",
        )

    def test_phase2_surrogate_joint_reproduces_reference(
        self, surrogate_p1p2_result, reference_true_params,
    ):
        """P2 k0 is approximately [1.374e-3, 4.865e-5], alpha approximately [0.597, 0.468]."""
        p2_k0 = surrogate_p1p2_result["p2_k0"]
        p2_alpha = surrogate_p1p2_result["p2_alpha"]

        np.testing.assert_allclose(
            p2_k0, [1.374e-3, 4.865e-5],
            rtol=0.10,
            err_msg="P2 k0 should match reference within 10%",
        )
        np.testing.assert_allclose(
            p2_alpha, [0.597, 0.468],
            rtol=0.10,
            err_msg="P2 alpha should match reference within 10%",
        )

    def test_full_pipeline_phase4_reproduces_reference(
        self, pde_p3p4_result, reference_true_params,
    ):
        """P4 k0 is approximately [1.382e-3, 5.220e-5], k0_2 error < 5%."""
        p4_k0 = pde_p3p4_result["p4_k0"]
        p4_alpha = pde_p3p4_result["p4_alpha"]
        true_k0 = reference_true_params["k0"]
        true_alpha = reference_true_params["alpha"]

        k0_err, alpha_err = _compute_errors(p4_k0, p4_alpha, true_k0, true_alpha)

        # k0_2 was the hardest parameter -- v11 reference achieves 0.82%
        assert k0_err[1] < 0.05, (
            f"P4 k0_2 error {k0_err[1]*100:.2f}% should be < 5%"
        )

        # All errors should be bounded
        np.testing.assert_allclose(
            p4_k0, [1.382e-3, 5.220e-5],
            rtol=0.15,
            err_msg="P4 k0 should match reference within 15%",
        )
        np.testing.assert_allclose(
            p4_alpha, [0.595, 0.461],
            rtol=0.15,
            err_msg="P4 alpha should match reference within 15%",
        )

    def test_best_of_selection_matches_reference(
        self, pde_p3p4_result, reference_true_params,
    ):
        """Best = Phase 4 (or P3), max error < 15%."""
        true_k0 = reference_true_params["k0"]
        true_alpha = reference_true_params["alpha"]

        p3_k0_err, p3_alpha_err = _compute_errors(
            pde_p3p4_result["p3_k0"], pde_p3p4_result["p3_alpha"],
            true_k0, true_alpha,
        )
        p4_k0_err, p4_alpha_err = _compute_errors(
            pde_p3p4_result["p4_k0"], pde_p3p4_result["p4_alpha"],
            true_k0, true_alpha,
        )

        p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
        p4_max_err = max(p4_k0_err.max(), p4_alpha_err.max())
        best_max_err = min(p3_max_err, p4_max_err)

        assert best_max_err < 0.15, (
            f"Best max error {best_max_err*100:.2f}% should be < 15%"
        )


# ===================================================================
# TestPDERefinementImprovesSurrogate
# ===================================================================

class TestPDERefinementImprovesSurrogate:
    """Verify PDE phases improve upon surrogate results."""

    def test_pde_p4_k0_2_better_than_surrogate(
        self, surrogate_p1p2_result, pde_p3p4_result, reference_true_params,
    ):
        """PDE k0_2 error < surrogate k0_2 error."""
        true_k0 = reference_true_params["k0"]
        true_alpha = reference_true_params["alpha"]

        surr_k0_err, _ = _compute_errors(
            surrogate_p1p2_result["p2_k0"], surrogate_p1p2_result["p2_alpha"],
            true_k0, true_alpha,
        )
        pde_k0_err, _ = _compute_errors(
            pde_p3p4_result["p4_k0"], pde_p3p4_result["p4_alpha"],
            true_k0, true_alpha,
        )

        # PDE should refine k0_2 (the hardest parameter)
        # In reference: surrogate k0_2 err = 7.57%, PDE P4 = 0.82%
        assert pde_k0_err[1] < surr_k0_err[1], (
            f"PDE P4 k0_2 err ({pde_k0_err[1]*100:.2f}%) should be < "
            f"surrogate P2 k0_2 err ({surr_k0_err[1]*100:.2f}%)"
        )

    def test_pde_p4_max_err_bounded(
        self, pde_p3p4_result, reference_true_params,
    ):
        """All 4 errors < 15% for Phase 4."""
        true_k0 = reference_true_params["k0"]
        true_alpha = reference_true_params["alpha"]

        k0_err, alpha_err = _compute_errors(
            pde_p3p4_result["p4_k0"], pde_p3p4_result["p4_alpha"],
            true_k0, true_alpha,
        )

        all_errors = np.concatenate([k0_err, alpha_err])
        for i, name in enumerate(["k0_1", "k0_2", "alpha_1", "alpha_2"]):
            assert all_errors[i] < 0.15, (
                f"P4 {name} error {all_errors[i]*100:.2f}% should be < 15%"
            )

    def test_pde_p3_loss_finite_and_reasonable(
        self, pde_p3p4_result,
    ):
        """P3 loss is finite and < 1.0."""
        p3_loss = pde_p3p4_result["p3_loss"]
        assert np.isfinite(p3_loss), "P3 loss should be finite"
        assert p3_loss < 1.0, f"P3 loss {p3_loss:.6e} should be < 1.0"


# ===================================================================
# TestMinimalPDE
# ===================================================================

class TestMinimalPDE:
    """Minimal PDE smoke test (~70s): single PDE solve at one voltage."""

    def test_single_pde_eval_at_true_params(self, pde_solver_infrastructure):
        """Single PDE solve at one voltage produces finite negative flux."""
        from FluxCurve.bv_point_solve import (
            solve_bv_curve_points_with_warmstart,
            _clear_caches,
        )

        infra = pde_solver_infrastructure
        observable_scale = -I_SCALE

        _clear_caches()

        # Single voltage point: mild cathodic
        phi_test = np.array([-5.0])
        dummy_target = np.zeros(1, dtype=float)

        recovery = make_recovery_config(max_it_cap=600)

        points = solve_bv_curve_points_with_warmstart(
            base_solver_params=infra["base_sp"],
            steady=infra["steady"],
            phi_applied_values=phi_test,
            target_flux=dummy_target,
            k0_values=[K0_HAT, K0_2_HAT],
            blob_initial_condition=False,
            fail_penalty=1e9,
            forward_recovery=recovery,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            mesh=infra["mesh"],
            alpha_values=[ALPHA_1, ALPHA_2],
            control_mode="joint",
            max_eta_gap=3.0,
        )

        _clear_caches()

        assert len(points) == 1, "Should get exactly one result"
        flux = float(points[0].simulated_flux)
        assert np.isfinite(flux), f"Flux should be finite, got {flux}"
        # Cathodic overpotential: current density should be negative
        assert flux < 0.0, (
            f"Current density at eta=-5 should be negative, got {flux:.6e}"
        )


# ===================================================================
# TestDifferentSeeds
# ===================================================================

class TestDifferentSeeds:
    """Test pipeline with different noise seeds (~400s each)."""

    def test_seed_42_pde_refinement_bounded(self):
        """noise_seed=42, all errors < 20%."""
        result = _run_full_pipeline(
            noise_percent=2.0,
            noise_seed=42,
            p3_maxiter=10,
            p4_maxiter=10,
            n_workers=1,
        )
        assert result["best_errs"]["max_err"] < 0.20, (
            f"Seed 42 max error {result['best_errs']['max_err']*100:.2f}% "
            f"should be < 20%"
        )

    def test_seed_99999_pde_refinement_bounded(self):
        """noise_seed=99999, all errors < 25% (reduced maxiter=10)."""
        result = _run_full_pipeline(
            noise_percent=2.0,
            noise_seed=99999,
            p3_maxiter=10,
            p4_maxiter=10,
            n_workers=1,
        )
        assert result["best_errs"]["max_err"] < 0.25, (
            f"Seed 99999 max error {result['best_errs']['max_err']*100:.2f}% "
            f"should be < 25%"
        )


# ===================================================================
# TestDifferentNoiseLevels
# ===================================================================

class TestDifferentNoiseLevels:
    """Test pipeline with different noise levels."""

    def test_noise_1pct_pde_refinement(self):
        """noise=1%, all errors < 20% (reduced maxiter=10), k0_2 < 10%."""
        result = _run_full_pipeline(
            noise_percent=1.0,
            noise_seed=20260226,
            p3_maxiter=10,
            p4_maxiter=10,
            n_workers=1,
        )
        best = result["best_errs"]
        assert best["max_err"] < 0.20, (
            f"1% noise max error {best['max_err']*100:.2f}% should be < 20%"
        )
        assert best["k0_2_err"] < 0.10, (
            f"1% noise k0_2 error {best['k0_2_err']*100:.2f}% should be < 10%"
        )

    def test_noise_5pct_pde_refinement(self):
        """noise=5%, all errors < 40% (reduced maxiter=10, high noise)."""
        result = _run_full_pipeline(
            noise_percent=5.0,
            noise_seed=20260226,
            p3_maxiter=10,
            p4_maxiter=10,
            n_workers=1,
        )
        assert result["best_errs"]["max_err"] < 0.40, (
            f"5% noise max error {result['best_errs']['max_err']*100:.2f}% "
            f"should be < 40%"
        )


# ===================================================================
# TestDifferentTrueParams
# ===================================================================

class TestDifferentTrueParams:
    """Test pipeline with shifted true parameters."""

    def test_shifted_alpha_recovery(self):
        """alpha=[0.45, 0.35], errors < 30% (reduced maxiter=10, shifted params)."""
        result = _run_full_pipeline(
            noise_percent=2.0,
            noise_seed=20260226,
            p3_maxiter=10,
            p4_maxiter=10,
            n_workers=1,
            alpha_1=0.45,
            alpha_2=0.35,
        )
        assert result["best_errs"]["max_err"] < 0.30, (
            f"Shifted alpha max error {result['best_errs']['max_err']*100:.2f}% "
            f"should be < 30%"
        )

    def test_shifted_k0_recovery(self):
        """k0 shifted 2x/0.5x, errors < 30%."""
        result = _run_full_pipeline(
            noise_percent=2.0,
            noise_seed=20260226,
            p3_maxiter=10,
            p4_maxiter=10,
            n_workers=1,
            k0_hat=K0_HAT * 2.0,
            k0_2_hat=K0_2_HAT * 0.5,
        )
        assert result["best_errs"]["max_err"] < 0.30, (
            f"Shifted k0 max error {result['best_errs']['max_err']*100:.2f}% "
            f"should be < 30%"
        )
