"""Adjoint-gradient inference of Robin kappa from a redimensionalized current-density curve."""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(_THIS_DIR)
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

# Keep Firedrake cache paths writable in sandboxed/restricted environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt

from FluxCurve import (
    ForwardRecoveryConfig,
    RobinFluxCurveInferenceRequest,
    run_robin_kappa_flux_curve_inference,
)
from Inverse import build_default_solver_params
from Utils.current_density_scaling import build_physical_scales, build_solver_options
from Forward.steady_state import SteadyStateConfig


RHE_REFERENCE_V = 0.695
A_PER_M2_TO_MA_PER_CM2 = 0.1


def main() -> None:
    scales = build_physical_scales()

    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=1e-1,
        t_end=20.0,
        z_vals=[1, -1],
        d_vals=[float(scales["d_species_m2_s"][0]), float(scales["d_species_m2_s"][1])],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[
            float(scales["bulk_concentration_mol_m3"]),
            float(scales["bulk_concentration_mol_m3"]),
        ],
        phi0=0.05,
        solver_options=build_solver_options(scales),
    )

    steady = SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        max_steps=120,
        # Keep convergence behavior close to historical runs by using the
        # non-Faraday proxy during optimization/steady checks.
        flux_observable="charge_proxy_no_f",  # enforced by request.observable_mode
        verbose=False,
        print_every=10,
    )

    request = RobinFluxCurveInferenceRequest(
        base_solver_params=base_solver_params,
        steady=steady,
        true_value=[1.0, 2.0],
        initial_guess=[5.0, 5.0],
        phi_applied_values=np.linspace(0.0, 0.04, 15),
        target_csv_path=os.path.join(
            "StudyResults",
            "robin_current_density_experiment",
            "phi_applied_vs_steady_current_density_synthetic.csv",
        ),
        output_dir=os.path.join("StudyResults", "robin_current_density_experiment"),
        regenerate_target=True,
        target_noise_percent=5.0,
        target_seed=20260222,
        # Keep objective in proxy units; convert to physical current at the end.
        observable_mode="charge_proxy_no_f",
        observable_species_index=None,
        observable_scale=1.0,
        observable_label="charge-weighted boundary flux proxy (model units)",
        observable_title="Robin kappa inference from charge proxy (rescaled after fit)",
        kappa_lower=1e-6,
        kappa_upper=20.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 80,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": True,
        },
        max_iters=8,
        gtol=1e-4,
        fail_penalty=1e9,
        print_point_gradients=True,
        # Use simple IC (uniform concentrations + linear potential), not blob.
        blob_initial_condition=False,
        live_plot=True,
        live_plot_pause_seconds=0.001,
        live_plot_eval_lines=True,
        live_plot_eval_line_alpha=0.30,
        live_plot_eval_max_lines=120,
        live_plot_export_gif_path=None,
        live_plot_export_gif_seconds=5.0,
        live_plot_export_gif_frames=50,
        live_plot_export_gif_dpi=140,
        # Replay temporarily disabled pending validity fixes.
        replay_mode_enabled=False,
        # Retained for future replay re-enable; currently inert while replay is off.
        replay_reenable_after_successes=1,
        # Optional process-parallel execution across phi_applied points.
        # Safe for adjoints because each worker has its own process + tape.
        parallel_point_solves_enabled=True,
        parallel_point_workers=4,
        parallel_point_min_points=4,
        parallel_start_method="spawn",
        anisotropy_trigger_failed_points=4,
        anisotropy_trigger_failed_fraction=0.25,
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=8,
            max_it_only_attempts=2,
            anisotropy_only_attempts=1,
            tolerance_relax_attempts=2,
            max_it_growth=1.5,
            max_it_cap=500,
            atol_relax_factor=10.0,
            rtol_relax_factor=10.0,
            ksp_rtol_relax_factor=10.0,
            line_search_schedule=("bt", "l2", "cp", "basic"),
            anisotropy_target_ratio=3.0,
            anisotropy_blend=0.5,
        ),
    )

    result = run_robin_kappa_flux_curve_inference(request)

    kappa_scale_m_s = float(scales["kappa_scale_m_s"])
    true_kappa_dimless = np.asarray(request.true_value, dtype=float)
    initial_kappa_dimless = np.asarray(request.initial_guess, dtype=float)
    estimated_kappa_dimless = np.asarray(result.best_kappa, dtype=float)

    true_kappa_m_s = (kappa_scale_m_s * true_kappa_dimless).tolist()
    initial_kappa_m_s = (kappa_scale_m_s * initial_kappa_dimless).tolist()
    estimated_kappa_m_s = (kappa_scale_m_s * estimated_kappa_dimless).tolist()

    # Redimensionalize proxy flux to physical current density after optimization.
    proxy_to_current_a_m2 = float(scales["current_density_scale_a_m2"])
    target_current_a_m2 = proxy_to_current_a_m2 * np.asarray(result.target_flux, dtype=float)
    simulated_current_a_m2 = proxy_to_current_a_m2 * np.asarray(result.best_simulated_flux, dtype=float)
    target_current_mA_cm2 = A_PER_M2_TO_MA_PER_CM2 * target_current_a_m2
    simulated_current_mA_cm2 = A_PER_M2_TO_MA_PER_CM2 * simulated_current_a_m2
    applied_voltage_v_vs_rhe = RHE_REFERENCE_V - np.asarray(result.phi_applied_values, dtype=float)

    # Persist a physical-units fit curve for downstream plotting/reporting.
    output_dir = os.path.join("StudyResults", "robin_current_density_experiment")
    os.makedirs(output_dir, exist_ok=True)
    redim_fit_csv_path = os.path.join(output_dir, "robin_kappa_fit_current_density_A_per_m2.csv")
    np.savetxt(
        redim_fit_csv_path,
        np.column_stack(
            [
                np.asarray(result.phi_applied_values, dtype=float),
                target_current_a_m2,
                simulated_current_a_m2,
            ]
        ),
        delimiter=",",
        header="phi_applied,target_current_density_A_per_m2,simulated_current_density_A_per_m2",
        comments="",
    )
    slide_units_csv_path = os.path.join(
        output_dir, "robin_kappa_fit_current_density_mA_per_cm2_vs_rhe.csv"
    )
    np.savetxt(
        slide_units_csv_path,
        np.column_stack(
            [
                applied_voltage_v_vs_rhe,
                target_current_mA_cm2,
                simulated_current_mA_cm2,
            ]
        ),
        delimiter=",",
        header=(
            "applied_voltage_v_vs_rhe,"
            "target_peroxide_current_density_mA_per_cm2,"
            "simulated_peroxide_current_density_mA_per_cm2"
        ),
        comments="",
    )
    slide_units_plot_path = os.path.join(
        output_dir, "robin_kappa_fit_current_density_mA_per_cm2_vs_rhe.png"
    )
    plt.figure(figsize=(7, 4))
    plt.plot(
        applied_voltage_v_vs_rhe,
        target_current_mA_cm2,
        marker="o",
        linewidth=2,
        label="target",
    )
    plt.plot(
        applied_voltage_v_vs_rhe,
        simulated_current_mA_cm2,
        marker="s",
        linewidth=2,
        label="best-fit simulated",
    )
    plt.xlabel("Applied Voltage (V vs RHE)")
    plt.ylabel("Peroxide Current Density (mA/cm^2)")
    plt.title("Current density of H$_2$O$_2$ production")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(slide_units_plot_path, dpi=170)
    plt.close()

    current_density_scale_a_m2 = float(scales["current_density_scale_a_m2"])
    current_density_scale_a_cm2 = current_density_scale_a_m2 / 1.0e4

    print("\n=== Robin Kappa Inference (Redimensionalized Current-Density Curve) ===")
    print(
        "Observable mode: charge_proxy_no_f "
        "(optimizer uses proxy units; converted to A/m^2 after fit)"
    )
    print(f"Length scale L_ref: {float(scales['length_scale_m']):.6e} m")
    print(f"Time scale t_ref: {float(scales['time_scale_s']):.6e} s")
    print(
        f"Debye ratio lambda_D/L_ref: {float(scales['debye_to_length_ratio']):.6e} "
        f"(lambda_D={float(scales['debye_length_m']):.6e} m)"
    )
    print(f"Kappa scale: {kappa_scale_m_s:.6e} m/s per model unit")
    print(
        "Current-density scale: "
        f"{current_density_scale_a_m2:.6e} A/m^2 "
        f"({current_density_scale_a_cm2:.6e} A/cm^2) per model unit"
    )
    print(f"True kappa (model): {true_kappa_dimless.tolist()}")
    print(f"True kappa (m/s): {true_kappa_m_s}")
    print(f"Initial guess (model): {initial_kappa_dimless.tolist()}")
    print(f"Initial guess (m/s): {initial_kappa_m_s}")
    print(f"Estimated kappa (model): {estimated_kappa_dimless.tolist()}")
    print(f"Estimated kappa (m/s): {estimated_kappa_m_s}")
    print(f"Final objective value: {result.best_loss:.12e}")
    print(
        "Final-curve current-density range (mA/cm^2): "
        f"target=[{float(np.min(target_current_mA_cm2)):.6e}, {float(np.max(target_current_mA_cm2)):.6e}] "
        f"sim=[{float(np.min(simulated_current_mA_cm2)):.6e}, {float(np.max(simulated_current_mA_cm2)):.6e}]"
    )
    print(f"Saved redimensionalized fit CSV: {redim_fit_csv_path}")
    print(f"Saved slide-units fit CSV: {slide_units_csv_path}")
    print(f"Saved slide-units fit plot: {slide_units_plot_path}")
    print(f"SciPy success: {result.optimization_success}")
    print(f"SciPy message: {result.optimization_message}")


if __name__ == "__main__":
    main()
