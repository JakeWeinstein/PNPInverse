"""Compare I-V curves: old (n_e=1 in exponent) vs new (n_e=2 in exponent).

Runs the BV forward solver at nominal parameters from -1000 to +1000 mV,
extracting both total current density and peroxide current density.
Sweeps from 0 outward in both directions for continuation stability.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
)
setup_firedrake_env()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Forward.steady_state import SteadyStateConfig
from Forward.steady_state.bv import sweep_phi_applied_steady_bv_flux
from Forward.bv_solver import make_graded_rectangle_mesh

V_T_mV = 25.693  # thermal voltage in mV


def _extract(results):
    """Extract voltages, total current, peroxide current from sweep results."""
    voltages = []
    total_current = []
    peroxide_current = []
    for r in results:
        voltages.append(r.phi_applied)
        if r.converged and len(r.species_flux) >= 2:
            R0, R1 = r.species_flux[0], r.species_flux[1]
            total_current.append(r.observed_flux)
            peroxide_current.append((R0 - R1) * I_SCALE)
        else:
            total_current.append(float("nan"))
            peroxide_current.append(float("nan"))
    return np.array(voltages), np.array(total_current), np.array(peroxide_current)


def run_sweep(n_electrons_override=None):
    """Run I-V sweep from -1000 to +1000 mV.

    Sweeps cathodic (0 -> -1000 mV) and anodic (0 -> +1000 mV) separately
    for continuation stability, then merges.
    """
    step_mV = 25.0
    # Cathodic: 0, -25, -50, ..., -1000 mV  (descending eta)
    eta_cathodic = np.arange(0, -1000 - step_mV / 2, -step_mV) / V_T_mV
    # Anodic: use fine 5 mV steps for better continuation convergence
    anodic_step_mV = 5.0
    eta_anodic = np.arange(anodic_step_mV, 1000 + anodic_step_mV / 2, anodic_step_mV) / V_T_mV

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
    )

    if n_electrons_override is not None:
        opts = sp.solver_options
        for rxn in opts["bv_bc"]["reactions"]:
            rxn["n_electrons"] = n_electrons_override

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # Cathodic sweep (0 -> -1000 mV)
    print(f"    Cathodic sweep: {len(eta_cathodic)} points")
    results_cat = sweep_phi_applied_steady_bv_flux(
        base_solver_params=sp,
        phi_applied_values=eta_cathodic,
        steady=steady,
        mesh=mesh,
        i_scale=-I_SCALE,
    )
    v_cat, i_tot_cat, i_pxd_cat = _extract(results_cat)

    # Anodic sweep (0 -> +1000 mV) — needs fresh mesh/context
    print(f"    Anodic sweep: {len(eta_anodic)} points")
    mesh2 = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    results_anod = sweep_phi_applied_steady_bv_flux(
        base_solver_params=sp,
        phi_applied_values=eta_anodic,
        steady=steady,
        mesh=mesh2,
        i_scale=-I_SCALE,
    )
    v_anod, i_tot_anod, i_pxd_anod = _extract(results_anod)

    # Merge and sort by voltage (ascending mV)
    v_all = np.concatenate([v_cat, v_anod])
    i_tot_all = np.concatenate([i_tot_cat, i_tot_anod])
    i_pxd_all = np.concatenate([i_pxd_cat, i_pxd_anod])
    order = np.argsort(v_all)
    return v_all[order], i_tot_all[order], i_pxd_all[order]


if __name__ == "__main__":
    print("=" * 60)
    print("Running with n_electrons=1 (OLD behavior)...")
    print("=" * 60)
    v_old, i_tot_old, i_pxd_old = run_sweep(n_electrons_override=1)
    n_conv = np.sum(~np.isnan(i_tot_old))
    print(f"  Done: {n_conv}/{len(i_tot_old)} converged")

    print("=" * 60)
    print("Running with n_electrons=2 (NEW/correct behavior)...")
    print("=" * 60)
    v_new, i_tot_new, i_pxd_new = run_sweep(n_electrons_override=None)
    n_conv = np.sum(~np.isnan(i_tot_new))
    print(f"  Done: {n_conv}/{len(i_tot_new)} converged")

    # Convert to mV for plotting
    v_old_mV = v_old * V_T_mV
    v_new_mV = v_new * V_T_mV

    # Two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(-v_old_mV, -i_tot_old, "-", color="tab:blue", label="Old (n_e=1 in exponent)", linewidth=1.5)
    ax1.plot(-v_new_mV, -i_tot_new, "-", color="tab:red", label="New (n_e=2 in exponent)", linewidth=1.5)
    ax1.set_xlabel("$-$Overpotential (mV vs E$_{eq}$)")
    ax1.set_ylabel("$-$Total current density (mA/cm$^2$)")
    ax1.set_title("Total Current Density  $I_{CD}$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(-v_old_mV, -i_pxd_old, "-", color="tab:blue", label="Old (n_e=1 in exponent)", linewidth=1.5)
    ax2.plot(-v_new_mV, -i_pxd_new, "-", color="tab:red", label="New (n_e=2 in exponent)", linewidth=1.5)
    ax2.set_xlabel("$-$Overpotential (mV vs E$_{eq}$)")
    ax2.set_ylabel("$-$Peroxide current density (mA/cm$^2$)")
    ax2.set_title("Peroxide Current  $I_{PC}$")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("n_electrons Bug Fix: -1000 to +1000 mV (nominal parameters)", fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(_ROOT, "StudyResults", "ne_comparison_iv.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")

    # Print table
    print(f"\n{'mV':>8}  {'I_CD old':>10}  {'I_CD new':>10}  {'I_PC old':>10}  {'I_PC new':>10}")
    print("-" * 54)
    for v, ito, itn, ipo, ipn in zip(v_old_mV, i_tot_old, i_tot_new, i_pxd_old, i_pxd_new):
        print(f"{v:8.1f}  {ito:10.4f}  {itn:10.4f}  {ipo:10.4f}  {ipn:10.4f}")
