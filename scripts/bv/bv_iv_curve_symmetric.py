"""Symmetric (anodic + cathodic) I-V curve sweep for 4-species charged PNP-BV.

Sweeps from equilibrium (eta=0) outward in BOTH directions:
  - Anodic (positive eta): R1 reverse, pure R1 signal
  - Cathodic (negative eta): R1 + R2 forward

This validates forward solver convergence at positive overpotentials
and generates reference I-V data for inference with symmetric voltage placement.

Usage (from PNPInverse/ directory)::

    python scripts/bv/bv_iv_curve_symmetric.py
    python scripts/bv/bv_iv_curve_symmetric.py --eta-anodic-max 6.0 --eta-cathodic-max 30.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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


# ---------------------------------------------------------------------------
# Physical constants (same as bv_iv_curve_charged.py)
# ---------------------------------------------------------------------------

F_CONST   = 96485.3329
R_GAS     = 8.31446
T_REF     = 298.15
V_T       = R_GAS * T_REF / F_CONST
E_EQ_RHE  = 0.695
N_ELECTRONS = 2

D_O2   = 1.9e-9;   C_O2   = 0.5
D_H2O2 = 1.6e-9;   C_H2O2 = 0.0
D_HP   = 9.311e-9;  C_HP   = 0.1
D_CLO4 = 1.792e-9;  C_CLO4 = 0.1

K0_1_PHYS  = 2.4e-8;  ALPHA_1 = 0.627
K0_2_PHYS  = 1e-9;    ALPHA_2 = 0.5

L_REF = 1.0e-4
D_REF = D_O2;  C_SCALE = C_O2;  K_SCALE = D_REF / L_REF

D_O2_HAT   = D_O2 / D_REF;   D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT   = D_HP / D_REF;   D_CLO4_HAT = D_CLO4 / D_REF
C_O2_HAT   = C_O2 / C_SCALE; C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT   = C_HP / C_SCALE; C_CLO4_HAT = C_CLO4 / C_SCALE

I_SCALE_MA_CM2 = N_ELECTRONS * F_CONST * (D_REF * C_SCALE / L_REF) * 0.1


SNES_OPTS = {
    "snes_type":                "newtonls",
    "snes_max_it":              300,
    "snes_atol":                1e-7,
    "snes_rtol":                1e-10,
    "snes_stol":                1e-12,
    "snes_linesearch_type":     "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type":                 "preonly",
    "pc_type":                  "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8":        77,
    "mat_mumps_icntl_14":       80,
}


def _make_sp(eta_hat: float, dt: float, t_end: float,
             *, l_ref: float = L_REF) -> SolverParams:
    """Build SolverParams for 4-species charged BV."""
    k_scale = D_REF / l_ref
    k0_1_hat = K0_1_PHYS / k_scale
    k0_2_hat = K0_2_PHYS / k_scale

    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent": True, "exponent_clip": 50.0,
        "regularize_concentration": True, "conc_floor": 1e-12,
        "use_eta_in_bv": True,
    }
    params["nondim"] = {
        "enabled": True,
        "diffusivity_scale_m2_s": D_REF,
        "concentration_scale_mol_m3": C_SCALE,
        "length_scale_m": l_ref,
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
                "k0": k0_1_hat, "alpha": ALPHA_1,
                "cathodic_species": 0, "anodic_species": 1,
                "c_ref": 1.0, "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2, "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
            {
                "k0": k0_2_hat, "alpha": ALPHA_2,
                "cathodic_species": 1, "anodic_species": None,
                "c_ref": 0.0, "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2, "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
        ],
        "k0": [k0_1_hat] * 4, "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0], "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    return SolverParams.from_list([
        4, 1, dt, t_end, [0, 0, 1, -1],
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
        [0.0, 0.0, 0.0, 0.0],
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
        0.0, params,
    ])


def _sweep_one_direction(ctx, solver, bv_rate_exprs, ds, electrode_marker,
                         eta_path, *, ss_tol, max_ss_steps, i_scale, label):
    """Sweep eta in one direction, returning list of result tuples."""
    results = []
    U_scratch = ctx["U"].copy(deepcopy=True)

    for i_step, eta in enumerate(eta_path):
        ctx["phi_applied_func"].assign(float(eta))

        n_taken = 0
        ss_reached = False
        try:
            for k in range(max_ss_steps):
                U_scratch.assign(ctx["U"])
                solver.solve()
                delta_norm = fd.errornorm(ctx["U"], U_scratch, norm_type="L2")
                ref_norm = fd.norm(U_scratch, norm_type="L2")
                rel_change = delta_norm / max(ref_norm, 1e-14)
                ctx["U_prev"].assign(ctx["U"])
                n_taken = k + 1
                if rel_change < ss_tol:
                    ss_reached = True
                    break

        except fd.ConvergenceError as e:
            ctx["U"].assign(ctx["U_prev"])
            print(f"  [FAIL] {label} eta={eta:+.2f}: {str(e)[:60]}")
            print(f"         Stopping {label} sweep at step {i_step+1}/{len(eta_path)}")
            break

        R1_val = float(fd.assemble(bv_rate_exprs[0] * ds(electrode_marker)))
        R2_val = float(fd.assemble(bv_rate_exprs[1] * ds(electrode_marker)))
        I_total    = -(R1_val + R2_val) * i_scale
        I_peroxide = -(R1_val - R2_val) * i_scale

        c_O2_e  = float(ctx["U"].sub(0).at((0.5, 0.0)))
        c_H2O2_e = float(ctx["U"].sub(1).at((0.5, 0.0)))
        c_HP_e  = float(ctx["U"].sub(2).at((0.5, 0.0)))

        results.append((
            eta, eta * V_T, I_peroxide, I_total,
            n_taken, int(ss_reached),
            R1_val, R2_val,
            c_O2_e, c_H2O2_e, c_HP_e,
        ))

        if (i_step + 1) % 5 == 0 or i_step == 0:
            ss_mark = "+" if ss_reached else "!"
            print(f"  {ss_mark} {label} step {i_step+1:3d}/{len(eta_path)}  "
                  f"eta={eta:+7.2f}  "
                  f"steps={n_taken:2d}  "
                  f"I_pxd={I_peroxide:+.4e}  I_tot={I_total:+.4e}  "
                  f"c_O2={c_O2_e:.4f}  c_H2O2={c_H2O2_e:.4f}  c_H+={c_HP_e:.4f}")

    return results


def run_symmetric_sweep(
    *,
    eta_anodic_max: float = 6.0,
    eta_cathodic_max: float = 28.0,
    anodic_steps: int = 20,
    cathodic_steps: int = 60,
    dt: float = 0.5,
    ss_tol: float = 1e-5,
    max_ss_steps: int = 80,
    Nx_mesh: int = 8,
    Ny_mesh: int = 200,
    beta_mesh: float = 3.0,
    l_ref: float = L_REF,
    out_dir: str = "StudyResults/bv_iv_curve_symmetric",
) -> dict:
    """Sweep from equilibrium (eta=0) outward in both directions.

    Phase 1: Anodic (eta = 0 -> +eta_anodic_max)
    Phase 2: Reset to eta=0, then cathodic (eta = 0 -> -eta_cathodic_max)
    """
    os.makedirs(out_dir, exist_ok=True)
    i_scale = N_ELECTRONS * F_CONST * (D_REF * C_SCALE / l_ref) * 0.1

    print(f"\n{'='*70}")
    print(f"  Symmetric I-V Curve: Anodic + Cathodic")
    print(f"{'='*70}")
    print(f"  Anodic range:   0 -> +{eta_anodic_max:.1f} V_T "
          f"(0 -> +{eta_anodic_max*V_T*1000:.0f} mV)")
    print(f"  Cathodic range: 0 -> -{eta_cathodic_max:.1f} V_T "
          f"(0 -> -{eta_cathodic_max*V_T*1000:.0f} mV)")
    print(f"  Steps: anodic={anodic_steps}, cathodic={cathodic_steps}")
    print(f"  Mesh: {Nx_mesh}x{Ny_mesh} (beta={beta_mesh:.1f})")
    print(f"{'='*70}\n")

    # Build at eta=0
    sp0 = _make_sp(0.0, dt=dt, t_end=dt * max_ss_steps, l_ref=l_ref)
    mesh = make_graded_rectangle_mesh(Nx_mesh, Ny_mesh, beta_mesh)
    ctx = build_context(sp0, mesh=mesh)
    ctx = build_forms(ctx, sp0)
    set_initial_conditions(ctx, sp0, blob=False)

    ds = fd.Measure("ds", domain=mesh)
    electrode_marker = 3
    bv_rate_exprs = ctx.get("bv_rate_exprs", [])

    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(F_res, ctx["U"], bcs=ctx["bcs"], J=J_form)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

    # Phase 1: Anodic sweep
    print(f"--- Phase 1: Anodic sweep (eta = 0 -> +{eta_anodic_max:.1f}) ---\n")
    anodic_path = np.linspace(0.0, eta_anodic_max, anodic_steps + 1)[1:]
    t0 = time.time()
    anodic_results = _sweep_one_direction(
        ctx, solver, bv_rate_exprs, ds, electrode_marker,
        anodic_path, ss_tol=ss_tol, max_ss_steps=max_ss_steps,
        i_scale=i_scale, label="ANODIC",
    )
    print(f"\n  Anodic phase: {len(anodic_results)} points, {time.time()-t0:.1f}s\n")

    # Phase 2: Reset to eta=0, then cathodic sweep
    print(f"--- Phase 2: Reset + Cathodic sweep (eta = 0 -> -{eta_cathodic_max:.1f}) ---\n")
    ctx["phi_applied_func"].assign(0.0)
    set_initial_conditions(ctx, sp0, blob=False)

    cathodic_path = np.linspace(0.0, -eta_cathodic_max, cathodic_steps + 1)[1:]
    t0 = time.time()
    cathodic_results = _sweep_one_direction(
        ctx, solver, bv_rate_exprs, ds, electrode_marker,
        cathodic_path, ss_tol=ss_tol, max_ss_steps=max_ss_steps,
        i_scale=i_scale, label="CATHODIC",
    )
    print(f"\n  Cathodic phase: {len(cathodic_results)} points, {time.time()-t0:.1f}s\n")

    # Combine results
    all_results = anodic_results + cathodic_results
    if not all_results:
        print("[ERROR] No results obtained")
        return {"eta_hat": np.array([]), "I_peroxide": np.array([]),
                "I_total": np.array([])}

    arr = np.array(all_results)
    # Sort by eta_hat (ascending: most negative first)
    sort_idx = np.argsort(arr[:, 0])
    arr_sorted = arr[sort_idx]

    # Save CSV
    csv_path = os.path.join(out_dir, "bv_iv_curve_symmetric.csv")
    header = ("eta_hat,eta_V,I_peroxide_mA_cm2,I_total_mA_cm2,"
              "ss_steps,ss_converged,R1,R2,"
              "c_O2_electrode,c_H2O2_electrode,c_HP_electrode")
    np.savetxt(csv_path, arr_sorted, delimiter=",", header=header, comments="")
    print(f"[sweep] CSV saved -> {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Total points: {len(arr_sorted)} "
          f"(anodic: {len(anodic_results)}, cathodic: {len(cathodic_results)})")
    print(f"  eta range: [{arr_sorted[0,0]:+.2f}, {arr_sorted[-1,0]:+.2f}] V_T")
    print(f"  I_peroxide range: [{arr_sorted[:,2].min():.4f}, "
          f"{arr_sorted[:,2].max():.4f}] mA/cm2")

    result = {
        "eta_hat": arr_sorted[:, 0],
        "eta_V": arr_sorted[:, 1],
        "I_peroxide": arr_sorted[:, 2],
        "I_total": arr_sorted[:, 3],
        "raw": arr_sorted,
    }

    # Plot
    _plot_symmetric_iv(result, out_dir=out_dir)

    return result


def _plot_symmetric_iv(result: dict, out_dir: str = "StudyResults/bv_iv_curve_symmetric"):
    """Plot the symmetric I-V curve."""
    eta = result["eta_hat"]
    eta_mV = result["eta_V"] * 1000  # mV
    I_pxd = result["I_peroxide"]
    I_tot = result["I_total"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: I vs eta_hat (nondimensional)
    ax = axes[0]
    ax.plot(eta, I_pxd, "o-", color="#c0408a", linewidth=1.5, markersize=2,
            label=r"$I_{\mathrm{peroxide}}$")
    ax.plot(eta, I_tot, "s-", color="#1f77b4", linewidth=1.0, markersize=1.5,
            linestyle="--", label=r"$I_{\mathrm{total}}$")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\hat{\eta}$ (dimensionless overpotential)", fontsize=11)
    ax.set_ylabel("Current density (mA/cm$^2$)", fontsize=11)
    ax.set_title("Symmetric I-V: anodic + cathodic", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Tafel plot (log|I| vs eta_hat)
    ax = axes[1]
    mask_anodic = (eta > 0.3) & (np.abs(I_pxd) > 1e-8)
    mask_cathodic = (eta < -0.3) & (np.abs(I_pxd) > 1e-8)

    if mask_anodic.any():
        ax.plot(eta[mask_anodic], np.log10(np.abs(I_pxd[mask_anodic])),
                "o-", color="#ff7f0e", markersize=3, linewidth=1.5,
                label="Anodic (R1 reverse)")
    if mask_cathodic.any():
        ax.plot(eta[mask_cathodic], np.log10(np.abs(I_pxd[mask_cathodic])),
                "s-", color="#2ca02c", markersize=3, linewidth=1.5,
                label="Cathodic (R1+R2)")

    # Expected Tafel slopes
    eta_fit = np.linspace(1.0, 5.0, 50)
    if mask_anodic.any():
        I_anodic_ref = np.abs(I_pxd[mask_anodic])[0]
        eta_anodic_ref = eta[mask_anodic][0]
        log_I_fit = np.log10(I_anodic_ref) + (1 - ALPHA_1) * (eta_fit - eta_anodic_ref) * np.log10(np.e)
        ax.plot(eta_fit, log_I_fit, "--", color="#ff7f0e", alpha=0.5,
                label=f"slope = (1-alpha) = {1-ALPHA_1:.3f}")

    eta_fit_cat = np.linspace(-5.0, -1.0, 50)
    if mask_cathodic.any():
        idx0 = np.argmin(np.abs(eta[mask_cathodic] - (-1.0)))
        I_cat_ref = np.abs(I_pxd[mask_cathodic])[idx0]
        eta_cat_ref = eta[mask_cathodic][idx0]
        log_I_fit_cat = np.log10(I_cat_ref) + ALPHA_1 * (-(eta_fit_cat - eta_cat_ref)) * np.log10(np.e)
        ax.plot(eta_fit_cat, log_I_fit_cat, "--", color="#2ca02c", alpha=0.5,
                label=f"slope = alpha = {ALPHA_1:.3f}")

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\hat{\eta}$ (dimensionless overpotential)", fontsize=11)
    ax.set_ylabel(r"$\log_{10}|I|$ (mA/cm$^2$)", fontsize=11)
    ax.set_title("Tafel plot: anodic and cathodic branches", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "bv_iv_curve_symmetric.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] PNG saved -> {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Symmetric (anodic + cathodic) I-V curve sweep"
    )
    parser.add_argument("--eta-anodic-max", type=float, default=6.0,
                        dest="eta_anodic_max",
                        help="Max anodic eta_hat (default 6.0)")
    parser.add_argument("--eta-cathodic-max", type=float, default=28.0,
                        dest="eta_cathodic_max",
                        help="Max cathodic |eta_hat| (default 28.0)")
    parser.add_argument("--anodic-steps", type=int, default=20,
                        dest="anodic_steps",
                        help="Number of anodic steps (default 20)")
    parser.add_argument("--cathodic-steps", type=int, default=60,
                        dest="cathodic_steps",
                        help="Number of cathodic steps (default 60)")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--ss-tol", type=float, default=1e-5, dest="ss_tol")
    parser.add_argument("--max-ss-steps", type=int, default=80, dest="max_ss_steps")
    parser.add_argument("--Ny-mesh", type=int, default=200, dest="Ny_mesh")
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--out-dir", type=str, default="StudyResults/bv_iv_curve_symmetric",
                        dest="out_dir")
    args = parser.parse_args()

    run_symmetric_sweep(
        eta_anodic_max=args.eta_anodic_max,
        eta_cathodic_max=args.eta_cathodic_max,
        anodic_steps=args.anodic_steps,
        cathodic_steps=args.cathodic_steps,
        dt=args.dt,
        ss_tol=args.ss_tol,
        max_ss_steps=args.max_ss_steps,
        Ny_mesh=args.Ny_mesh,
        beta_mesh=args.beta,
        out_dir=args.out_dir,
    )
    print(f"\n=== Symmetric I-V Curve Complete ===")
    print(f"Output directory: {args.out_dir}/")


if __name__ == "__main__":
    main()
