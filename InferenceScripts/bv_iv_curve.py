"""Butler-Volmer I-V curve: O₂ reduction current density vs. applied voltage (V vs RHE).

Supports two modes:

**1-species** (``--n-species 1``):
  Single neutral O₂ with per-species BV kinetics (legacy).

**2-species** (``--n-species 2``, default):
  Coupled O₂ + H₂O₂ with multi-reaction BV:
    R₁: O₂ + 2H⁺ + 2e⁻ → H₂O₂   (reversible, rate constant k₀₁, transfer coeff α₁)
    R₂: H₂O₂ + 2H⁺ + 2e⁻ → H₂O  (irreversible, rate constant k₀₂, transfer coeff α₂)

  Current observables:
    I_total    = 2F(R₁ + R₂)     — total electronic current
    I_peroxide = 2F(R₁ − R₂)     — net peroxide current (what Mangan2025 plots)

Physical parameters from Mangan2025 (pH 4 case):
  - E_eq = 0.695 V vs RHE           (O₂/H₂O₂ equilibrium potential)
  - D_O2 = 2.10 × 10⁻⁹ m²/s       (O₂ diffusivity in water at 25 °C)
  - D_H2O2 = 1.60 × 10⁻⁹ m²/s     (H₂O₂ diffusivity in water at 25 °C)
  - c_bulk = 0.5 mol/m³             (dissolved O₂ concentration)
  - k₀₁ = 2.4 × 10⁻⁸ m/s          (R₁ exchange rate constant)
  - α₁ = 0.627                      (R₁ charge-transfer coefficient)
  - k₀₂ = 1 × 10⁻⁹ m/s            (R₂ rate constant — sweep parameter)
  - α₂ = 0.5                        (R₂ charge-transfer coefficient — sweep parameter)

Usage (from PNPInverse/ directory)::

    # Default 2-species run
    python InferenceScripts/bv_iv_curve.py

    # Legacy 1-species
    python InferenceScripts/bv_iv_curve.py --n-species 1

    # Parameter studies
    python InferenceScripts/bv_iv_curve.py --k0-2 1e-8
    python InferenceScripts/bv_iv_curve.py --alpha-2 0.8
    python InferenceScripts/bv_iv_curve.py --l-ref 3e-4
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import firedrake as fd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt

from Forward.bv_solver import build_context, build_forms, set_initial_conditions
from Forward.params import SolverParams


# ---------------------------------------------------------------------------
# Physical constants and parameters
# ---------------------------------------------------------------------------

F_CONST   = 96485.3329   # C/mol
V_T       = 0.025693     # thermal voltage at 25 °C, V
E_EQ_RHE  = 0.695        # equilibrium potential for O₂/H₂O₂ at pH 4, V vs RHE
N_ELECTRONS = 2          # O₂ + 2H⁺ + 2e⁻ → H₂O₂

# Species 0: O₂ (z = 0, neutral)
D_O2   = 2.10e-9    # m²/s  (O₂ diffusivity in water at 25 °C)
C_BULK = 0.5        # mol/m³ (dissolved O₂ concentration)

# Species 1: H₂O₂ (z = 0, neutral)
D_H2O2 = 1.60e-9    # m²/s  (H₂O₂ diffusivity in water at 25 °C)

# R₁ kinetics (Mangan2025 fit): O₂ → H₂O₂
K0_PHYS  = 2.4e-8   # m/s  (exchange rate constant)
ALPHA    = 0.627     # charge-transfer coefficient

# R₂ kinetics: H₂O₂ → H₂O (irreversible)
K0_2_PHYS = 1e-9     # m/s  (default — sweep parameter)
ALPHA_2   = 0.5      # charge-transfer coefficient (default — sweep parameter)

# Reference scales
L_REF = 1.0e-4      # m — 100 µm diffusion layer

# Steric excluded-volume parameter (Bikerman, dimensionless = a_phys × c_scale)
A_O2_HAT   = 0.01   # dimensionless steric parameter for O₂
A_H2O2_HAT = 0.01   # dimensionless steric parameter for H₂O₂

D_REF   = D_O2                       # nondim: D_O2 maps to 1.0
K_SCALE = D_REF / L_REF             # transfer-coefficient scale (m/s)
K0_HAT  = K0_PHYS / K_SCALE        # dimensionless rate constant (R₁)

# Current density conversion
J_SCALE_MOL_M2_S = D_REF * C_BULK / L_REF
I_SCALE_A_M2     = N_ELECTRONS * F_CONST * J_SCALE_MOL_M2_S
I_SCALE_MA_CM2   = I_SCALE_A_M2 * 0.1

D_O2_HAT   = D_O2 / D_REF     # = 1.0 by construction
D_H2O2_HAT = D_H2O2 / D_REF   # ≈ 0.762

print(f"[params] Species 0    = O₂    (z=0, D={D_O2:.3e} m²/s)")
print(f"[params] Species 1    = H₂O₂  (z=0, D={D_H2O2:.3e} m²/s)")
print(f"[params] D_ref         = {D_REF:.3e} m²/s")
print(f"[params] kappa_scale   = {K_SCALE:.3e} m/s")
print(f"[params] k0_hat (R₁)   = {K0_HAT:.5f}")
print(f"[params] I_scale       = {I_SCALE_MA_CM2:.4f} mA/cm²  (= I_lim at full depletion)")


# ---------------------------------------------------------------------------
# SNES options — conservative Newton
# ---------------------------------------------------------------------------

SNES_OPTS = {
    "snes_type":                "newtonls",
    "snes_max_it":              200,
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
    "mat_mumps_icntl_14":       60,
}


# ---------------------------------------------------------------------------
# Build SolverParams
# ---------------------------------------------------------------------------

def _make_sp(
    eta_hat: float,
    dt: float,
    t_end: float,
    *,
    n_species: int = 2,
    a_hat: float = A_O2_HAT,
    k0_phys: float = K0_PHYS,
    alpha: float = ALPHA,
    k0_2_phys: float = K0_2_PHYS,
    alpha_2: float = ALPHA_2,
    c_bulk: float = C_BULK,
    l_ref: float = L_REF,
) -> SolverParams:
    """Return a SolverParams for the BV problem at the given eta_hat.

    When n_species=2, uses the multi-reaction config with two coupled reactions.
    When n_species=1, uses the legacy per-species config (backward compat).
    """
    k_scale_local = D_O2 / l_ref
    k0_hat_local  = k0_phys / k_scale_local
    k0_2_hat_local = k0_2_phys / k_scale_local

    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent":            True,
        "exponent_clip":            50.0,
        "regularize_concentration": True,
        "conc_floor":               1e-12,
        "use_eta_in_bv":            True,
    }
    params["nondim"] = {
        "enabled":                              True,
        "diffusivity_scale_m2_s":               D_O2,
        "concentration_scale_mol_m3":           c_bulk,
        "length_scale_m":                       l_ref,
        "potential_scale_v":                    V_T,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }

    if n_species == 2:
        params["bv_bc"] = {
            "reactions": [
                {
                    "k0": k0_hat_local,
                    "alpha": alpha,
                    "cathodic_species": 0,    # O₂ consumed
                    "anodic_species": 1,      # H₂O₂ produced
                    "c_ref": 1.0,             # nondim reference (= bulk O₂)
                    "stoichiometry": [-1, +1],
                    "n_electrons": 2,
                    "reversible": True,
                },
                {
                    "k0": k0_2_hat_local,
                    "alpha": alpha_2,
                    "cathodic_species": 1,    # H₂O₂ consumed
                    "anodic_species": None,   # irreversible
                    "c_ref": 0.0,
                    "stoichiometry": [0, -1],
                    "n_electrons": 2,
                    "reversible": False,
                },
            ],
            # Per-species fallback (needed by _get_bv_cfg for markers):
            "k0": [k0_hat_local, k0_2_hat_local],
            "alpha": [alpha, alpha_2],
            "stoichiometry": [-1, -1],
            "c_ref": [1.0, 0.0],
            "E_eq_v": 0.0,
            "electrode_marker":      1,
            "concentration_marker":  3,
            "ground_marker":         3,
        }
        return SolverParams.from_list([
            2,                                   # n_species
            1,                                   # FE order
            dt,                                  # Δt (nondim)
            t_end,                               # t_end (nondim)
            [0, 0],                              # z_vals — both neutral
            [D_O2_HAT, D_H2O2_HAT],             # D_vals (nondim)
            [a_hat, A_H2O2_HAT],                 # a_vals (Bikerman steric)
            eta_hat,                             # phi_applied (= η̂ = η / V_T)
            [1.0, 0.0],                          # c0_vals: O₂=1 (bulk), H₂O₂=0
            0.0,                                 # phi0
            params,
        ])
    else:
        # Legacy 1-species
        params["bv_bc"] = {
            "k0":               [k0_hat_local],
            "alpha":            [alpha],
            "stoichiometry":    [-1],
            "c_ref":            [1.0],
            "E_eq_v":           0.0,
            "electrode_marker":      1,
            "concentration_marker":  3,
            "ground_marker":         3,
        }
        return SolverParams.from_list([
            1,                           # n_species
            1,                           # FE order
            dt,
            t_end,
            [0],                         # z_vals — neutral
            [D_O2_HAT],                  # D_vals (nondim, = 1.0)
            [a_hat],                     # a_vals
            eta_hat,
            [1.0],                       # c0_vals
            0.0,
            params,
        ])


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_iv_sweep(
    *,
    n_species: int = 2,
    eta_steps: int = 100,
    dt: float = 0.5,
    ss_tol: float = 1e-5,
    max_ss_steps: int = 50,
    a_hat: float = A_O2_HAT,
    k0_phys: float = K0_PHYS,
    alpha: float = ALPHA,
    k0_2_phys: float = K0_2_PHYS,
    alpha_2: float = ALPHA_2,
    c_bulk: float = C_BULK,
    l_ref: float = L_REF,
    e_eq_rhe: float = E_EQ_RHE,
    out_dir: str = "StudyResults/bv_iv_curve",
) -> dict:
    """Sweep V_RHE from equilibrium to -0.5 V and compute I(V).

    Returns a dict with keys:
      V_rhe, I_peroxide, I_total, I_4e  (all sorted by V_rhe ascending)
    For n_species=1, I_total == I_peroxide and I_4e == 0.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Per-run scales
    i_scale_ma_cm2 = N_ELECTRONS * F_CONST * (D_O2 * c_bulk / l_ref) * 0.1
    k_scale_local  = D_O2 / l_ref
    k0_hat_local   = k0_phys / k_scale_local

    eta_min  = (-0.5 - e_eq_rhe) / V_T
    eta_path = np.linspace(0.0, eta_min, eta_steps + 1)[1:]

    mode_str = f"{n_species}-species" + (" (O₂+H₂O₂)" if n_species == 2 else " (O₂)")
    print(f"\n[sweep] {mode_str}, {eta_steps} steps, η̂=0 → {eta_min:.1f} V_T  "
          f"(V_RHE: {e_eq_rhe:.3f} → -0.500 V)")
    print(f"[sweep] dt={dt}, ss_tol={ss_tol:.0e}, max_ss_steps={max_ss_steps}")
    print(f"[sweep] R₁: k0={k0_phys:.2e} m/s (k0_hat={k0_hat_local:.5f}), α={alpha:.3f}")
    if n_species == 2:
        k0_2_hat = k0_2_phys / k_scale_local
        print(f"[sweep] R₂: k0_2={k0_2_phys:.2e} m/s (k0_2_hat={k0_2_hat:.5f}), α₂={alpha_2:.3f}")
    print(f"[sweep] c_bulk={c_bulk} mol/m³, L_ref={l_ref*1e6:.0f} µm, "
          f"I_scale={i_scale_ma_cm2:.4f} mA/cm²")

    # Build context once at η̂=0
    sp_eq = _make_sp(0.0, dt=dt, t_end=dt * max_ss_steps,
                     n_species=n_species, a_hat=a_hat,
                     k0_phys=k0_phys, alpha=alpha,
                     k0_2_phys=k0_2_phys, alpha_2=alpha_2,
                     c_bulk=c_bulk, l_ref=l_ref)
    ctx = build_context(sp_eq)
    ctx = build_forms(ctx, sp_eq)
    set_initial_conditions(ctx, sp_eq, blob=False)

    scaling  = ctx["nondim"]
    dt_model = float(scaling["dt_model"])

    mesh  = ctx["mesh"]
    ds    = fd.Measure("ds", domain=mesh)
    n_vec = fd.FacetNormal(mesh)

    F_res   = ctx["F_res"]
    J_form  = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(F_res, ctx["U"], bcs=ctx["bcs"], J=J_form)
    solver  = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

    U_scratch = ctx["U"].copy(deepcopy=True)

    bv_rate_exprs = ctx.get("bv_rate_exprs", [])
    has_reactions = len(bv_rate_exprs) > 0

    # results: (eta, V_rhe, I_peroxide, I_total, I_4e, n_steps, ss_converged)
    results_cat = []

    D0_model = float(scaling["D_model_vals"][0])
    electrode_marker = 1

    for i_step, eta in enumerate(eta_path):
        ctx["phi_applied_func"].assign(float(eta))
        V_rhe = e_eq_rhe + eta * V_T

        n_taken    = 0
        ss_reached = False

        try:
            for k in range(max_ss_steps):
                U_scratch.assign(ctx["U"])
                solver.solve()

                delta_norm = fd.errornorm(ctx["U"], U_scratch, norm_type="L2")
                ref_norm   = fd.norm(U_scratch, norm_type="L2")
                rel_change = delta_norm / max(ref_norm, 1e-14)

                ctx["U_prev"].assign(ctx["U"])
                n_taken = k + 1

                if rel_change < ss_tol:
                    ss_reached = True
                    break

        except fd.ConvergenceError:
            ctx["U"].assign(ctx["U_prev"])
            print(f"  [sweep] FAILED at η̂={eta:.2f} (V_RHE={V_rhe:+.3f} V) — stopping sweep")
            break

        if not ss_reached:
            print(f"  [warn] η̂={eta:+7.2f}: steady state NOT reached after "
                  f"{n_taken} steps (last Δrel={rel_change:.2e})")

        # Compute current density from BV reaction rates.
        # Sign convention: cathodic (reduction) current is negative.
        # R_j > 0 for cathodic direction, so I = -R * i_scale.
        # i_scale already includes N_ELECTRONS=2, so no extra factor needed.
        if has_reactions and n_species == 2:
            R1_val = float(fd.assemble(bv_rate_exprs[0] * ds(electrode_marker)))
            R2_val = float(fd.assemble(bv_rate_exprs[1] * ds(electrode_marker)))
            I_total_mA    = -(R1_val + R2_val) * i_scale_ma_cm2
            I_peroxide_mA = -(R1_val - R2_val) * i_scale_ma_cm2
            I_4e_mA       = I_total_mA - I_peroxide_mA  # = -2*R2*i_scale
        else:
            # 1-species: use diffusive flux (legacy path)
            c_species = fd.split(ctx["U"])[0]
            Jflux     = D0_model * fd.grad(c_species)
            flux_nd   = float(fd.assemble(fd.dot(Jflux, n_vec) * ds(electrode_marker)))
            I_peroxide_mA = flux_nd * i_scale_ma_cm2
            I_total_mA    = I_peroxide_mA
            I_4e_mA       = 0.0

        results_cat.append((eta, V_rhe, I_peroxide_mA, I_total_mA, I_4e_mA,
                            n_taken, int(ss_reached)))
        if (i_step + 1) % 5 == 0 or i_step == 0:
            ss_marker = "✓" if ss_reached else "!"
            if n_species == 2:
                print(f"  step {i_step+1:3d}/{eta_steps} {ss_marker} "
                      f"η̂={eta:+7.2f}  V_RHE={V_rhe:+6.3f} V  "
                      f"steps={n_taken:2d}  I_pxd={I_peroxide_mA:+.4e}  "
                      f"I_tot={I_total_mA:+.4e} mA/cm²")
            else:
                print(f"  step {i_step+1:3d}/{eta_steps} {ss_marker} "
                      f"η̂={eta:+7.2f}  V_RHE={V_rhe:+6.3f} V  "
                      f"steps={n_taken:2d}  I={I_peroxide_mA:+.4e} mA/cm²")

    # -----------------------------------------------------------------------
    # Assemble full I-V arrays (anodic stub at I=0 + cathodic sweep)
    # -----------------------------------------------------------------------
    V_anodic = np.linspace(1.25, e_eq_rhe, 8, endpoint=False)[::-1]
    n_anodic = len(V_anodic)

    arr_cat = np.array(results_cat)
    V_cat = arr_cat[:, 1]
    I_pxd_cat = arr_cat[:, 2]
    I_tot_cat = arr_cat[:, 3]
    I_4e_cat  = arr_cat[:, 4]

    V_full     = np.concatenate([V_anodic, V_cat])
    I_pxd_full = np.concatenate([np.zeros(n_anodic), I_pxd_cat])
    I_tot_full = np.concatenate([np.zeros(n_anodic), I_tot_cat])
    I_4e_full  = np.concatenate([np.zeros(n_anodic), I_4e_cat])

    sort_idx    = np.argsort(V_full)
    V_rhe_arr   = V_full[sort_idx]
    I_pxd_arr   = I_pxd_full[sort_idx]
    I_tot_arr   = I_tot_full[sort_idx]
    I_4e_arr    = I_4e_full[sort_idx]

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    n_steps_arr     = arr_cat[:, 5].astype(int)
    ss_conv_arr     = arr_cat[:, 6].astype(bool)
    n_not_converged = int((~ss_conv_arr).sum())

    # Build full data array with anodic padding
    anodic_data = np.column_stack([
        V_anodic, np.zeros(n_anodic), np.zeros(n_anodic), np.zeros(n_anodic),
        np.zeros(n_anodic), np.ones(n_anodic),
    ])
    cat_data = np.column_stack([
        arr_cat[:, 1], arr_cat[:, 2], arr_cat[:, 3], arr_cat[:, 4],
        arr_cat[:, 5], arr_cat[:, 6],
    ])
    full_data = np.concatenate([anodic_data, cat_data], axis=0)
    sort_idx  = np.argsort(full_data[:, 0])
    full_data = full_data[sort_idx]

    csv_path = os.path.join(out_dir, "bv_iv_curve.csv")
    np.savetxt(
        csv_path,
        full_data,
        delimiter=",",
        header=("applied_voltage_v_vs_rhe,peroxide_current_density_mA_cm2,"
                "total_current_density_mA_cm2,four_electron_current_density_mA_cm2,"
                "ss_steps,ss_converged"),
        comments="",
    )
    print(f"\n[sweep] CSV saved → {csv_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n[ss summary] Points reaching steady state: "
          f"{ss_conv_arr.sum()}/{len(ss_conv_arr)} "
          f"({100*ss_conv_arr.mean():.0f}%)")
    if len(n_steps_arr):
        print(f"[ss summary] Steps per point: min={n_steps_arr.min()}, "
              f"max={n_steps_arr.max()}, mean={n_steps_arr.mean():.1f}")
    if n_not_converged:
        print(f"[ss summary] WARNING: {n_not_converged} points did NOT reach "
              f"steady state — consider increasing max_ss_steps or decreasing dt")
    print(f"\n[summary] I_peroxide range: {I_pxd_arr.min():.4f} to {I_pxd_arr.max():.4f} mA/cm²")
    if n_species == 2:
        print(f"[summary] I_total range:    {I_tot_arr.min():.4f} to {I_tot_arr.max():.4f} mA/cm²")
    print(f"[summary] nondim I_scale = {i_scale_ma_cm2:.4f} mA/cm² per unit flux")

    return {
        "V_rhe": V_rhe_arr,
        "I_peroxide": I_pxd_arr,
        "I_total": I_tot_arr,
        "I_4e": I_4e_arr,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_iv_curve(
    result: dict,
    *,
    n_species: int = 2,
    a_hat: float = A_O2_HAT,
    l_ref: float = L_REF,
    out_dir: str = "StudyResults/bv_iv_curve",
) -> str:
    """Save a figure of the I-V curve."""
    V = result["V_rhe"]
    I_pxd = result["I_peroxide"]
    I_tot = result["I_total"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if n_species == 2:
        ax.plot(V, I_pxd, color="#c0408a", linewidth=2.0,
                label="$I_{\\mathrm{peroxide}}$ = $2F(R_1 - R_2)$")
        ax.plot(V, I_tot, color="#1f77b4", linewidth=1.5, linestyle="--",
                label="$I_{\\mathrm{total}}$ = $2F(R_1 + R_2)$")
        title_mode = "O₂+H₂O₂ coupled BV"
    else:
        ax.plot(V, I_pxd, color="#c0408a", linewidth=2.0,
                label=f"BV model (O₂, z=0, $a={a_hat}$)")
        title_mode = "O₂ only, per-species BV"

    ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$E_{{eq}}$ = {E_EQ_RHE:.3f} V vs RHE")
    ax.axhline(0.0, color="black", linewidth=0.5)

    I_lim_approx = float(np.nanmin(I_pxd))
    ax.annotate(
        f"$I_{{lim}}$ ≈ {I_lim_approx:.3f} mA/cm²",
        xy=(0.05, I_lim_approx),
        xytext=(0.5, I_lim_approx * 0.6),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=8,
        color="gray",
    )

    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=11)
    ax.set_ylabel("Current Density (mA/cm²)", fontsize=11)
    ax.set_title(f"{title_mode} — Bikerman steric ($a={a_hat}$)\n"
                 f"$L_\\mathrm{{ref}}={l_ref*1e6:.0f}\\,\\mu$m", fontsize=10)
    ax.set_xlim(-0.5, 1.25)
    y_min = min(float(np.nanmin(I_pxd)), float(np.nanmin(I_tot))) * 1.15
    ax.set_ylim(min(y_min, -0.01), 0.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "bv_iv_curve.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] PNG saved → {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Comparison plot: model vs. Mangan2025 simulation
# ---------------------------------------------------------------------------

def plot_iv_comparison(
    result: dict,
    *,
    n_species: int = 2,
    a_hat: float = A_O2_HAT,
    l_ref: float = L_REF,
    out_dir: str = "StudyResults/bv_iv_curve",
    exp_image_path: str = "writeups/WeekOfFeb25/assets/mangan_slide15.png",
) -> str:
    """Side-by-side: model I-V curve vs. Mangan2025 slide-15 simulation."""
    V = result["V_rhe"]
    I_pxd = result["I_peroxide"]
    I_tot = result["I_total"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: model
    if n_species == 2:
        axes[0].plot(V, I_pxd, color="#c0408a", linewidth=2,
                     label="$I_{\\mathrm{peroxide}}$")
        axes[0].plot(V, I_tot, color="#1f77b4", linewidth=1.5, linestyle="--",
                     label="$I_{\\mathrm{total}}$")
        model_title = f"Coupled BV — O₂+H₂O₂ ($a={a_hat}$)"
    else:
        axes[0].plot(V, I_pxd, color="#c0408a", linewidth=2,
                     label=f"BV model (O₂, $a={a_hat}$)")
        model_title = f"BV model — O₂ only ($a={a_hat}$)"

    axes[0].axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                    label=f"$E_{{eq}}$ = {E_EQ_RHE:.3f} V")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Applied Voltage (V vs RHE)")
    axes[0].set_ylabel("Current Density (mA/cm²)")
    axes[0].set_title(f"{model_title}\n"
                      f"$L_\\mathrm{{ref}}={l_ref*1e6:.0f}\\,\\mu$m")
    axes[0].set_xlim(-0.5, 1.25)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Right: Mangan2025 slide 15
    if os.path.exists(exp_image_path):
        img = plt.imread(exp_image_path)
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Mangan2025 slide 15 — MPB+BV simulation\n"
                          "(varying ion radius; spectral methods)")
    else:
        axes[1].text(0.5, 0.5, f"Image not found:\n{exp_image_path}",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=9)
        axes[1].set_title("Mangan2025 slide 15 — MPB+BV simulation")

    plt.suptitle("BV model (this work) vs. Mangan2025 MPB+BV reference simulation",
                 fontsize=11)
    plt.tight_layout()

    cmp_path = os.path.join(out_dir, "bv_iv_comparison.png")
    fig.savefig(cmp_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Comparison PNG saved → {cmp_path}")
    return cmp_path


# ---------------------------------------------------------------------------
# Steric overlay: two curves on one axes
# ---------------------------------------------------------------------------

def plot_steric_overlay(
    V1: np.ndarray,
    I1: np.ndarray,
    a1: float,
    V2: np.ndarray,
    I2: np.ndarray,
    a2: float,
    out_dir: str = "StudyResults/bv_iv_curve",
) -> str:
    """Overlay two I-V curves with different steric coefficients."""
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(V1, I1, color="#1f77b4", linewidth=2.0, label=f"$a = {a1}$")
    ax.plot(V2, I2, color="#c0408a", linewidth=2.0, label=f"$a = {a2}$")

    ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$E_{{eq}}$ = {E_EQ_RHE:.3f} V")
    ax.axhline(0.0, color="black", linewidth=0.5)

    I_lim1, I_lim2 = float(np.nanmin(I1)), float(np.nanmin(I2))
    ax.annotate(f"$I_{{lim}}={I_lim1:.3f}$", xy=(-0.4, I_lim1),
                xytext=(-0.1, I_lim1 * 0.75),
                arrowprops=dict(arrowstyle="->", color="#1f77b4"), fontsize=8, color="#1f77b4")
    ax.annotate(f"$I_{{lim}}={I_lim2:.3f}$", xy=(-0.4, I_lim2),
                xytext=(-0.1, I_lim2 * 0.5),
                arrowprops=dict(arrowstyle="->", color="#c0408a"), fontsize=8, color="#c0408a")

    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=11)
    ax.set_ylabel("Current Density (mA/cm²)", fontsize=11)
    ax.set_title("Steric effect on I-V curve (O₂ BV model, $z=0$)\n"
                 "Bikerman: $J_i = -D_i[\\nabla c_i + c_i\\nabla\\ln(1-ac_i)]$", fontsize=10)
    ax.set_xlim(-0.5, 1.25)
    y_min = min(I_lim1, I_lim2) * 1.15
    ax.set_ylim(min(y_min, -0.01), 0.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "bv_iv_steric_overlay.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Steric overlay PNG saved → {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Generic parameter-sweep overlay: load CSVs and compare side-by-side with exp.
# ---------------------------------------------------------------------------

def plot_param_sweep_overlay(
    entries: list[tuple[str, str]],
    title: str,
    out_path: str,
    *,
    current_col: int = 1,
    e_eq_rhe: float = E_EQ_RHE,
    exp_image_path: str = "writeups/WeekOfFeb25/assets/mangan_slide15.png",
) -> str:
    """Load CSVs for a parameter sweep and overlay them with the experimental reference.

    Parameters
    ----------
    entries : list of (label, csv_path)
    title : str
    out_path : str
    current_col : int
        CSV column index for current density (1=peroxide, 2=total, 3=4e).
    """
    cmap   = plt.cm.plasma
    n      = len(entries)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]

    for (label, csv_path), color in zip(entries, colors):
        if not os.path.exists(csv_path):
            print(f"[overlay] skipping '{label}': CSV not found at {csv_path}")
            continue
        data  = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        V, I  = data[:, 0], data[:, current_col]
        I_lim = float(np.nanmin(I))
        ax.plot(V, I, color=color, linewidth=2.0,
                label=f"{label}  ($I_{{lim}}={I_lim:.3f}$)")

    ax.axvline(e_eq_rhe, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$E_{{eq}}={e_eq_rhe:.3f}$ V")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=11)
    ax.set_ylabel("Current Density (mA/cm²)", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, 1.25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    if os.path.exists(exp_image_path):
        img = plt.imread(exp_image_path)
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Mangan2025 slide 15 — MPB+BV simulation\n"
                          "(pH 4, Cs⁺ ions; experimental target)", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "Mangan2025 image not found",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Mangan2025 slide 15")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Parameter overlay saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BV I-V curve (O₂ reduction current density vs V vs RHE)"
    )
    # Model mode
    parser.add_argument("--n-species", type=int, default=2, choices=[1, 2],
                        dest="n_species",
                        help="1 = single O₂ (legacy), 2 = coupled O₂+H₂O₂ (default)")
    # Sweep controls
    parser.add_argument("--steps",        type=int,   default=100,
                        help="Number of continuation steps (default 100)")
    parser.add_argument("--dt",           type=float, default=0.5,
                        help="Nondim time step (default 0.5)")
    parser.add_argument("--ss-tol",       type=float, default=1e-5,
                        dest="ss_tol",
                        help="Steady-state relative tolerance (default 1e-5)")
    parser.add_argument("--max-ss-steps", type=int,   default=50,
                        dest="max_ss_steps",
                        help="Maximum time steps per voltage point (default 50)")
    # R₁ parameters
    parser.add_argument("--steric",  type=float, default=A_O2_HAT,
                        help=f"Bikerman steric parameter a_hat (default {A_O2_HAT})")
    parser.add_argument("--k0",      type=float, default=K0_PHYS,
                        help=f"R₁ BV exchange rate constant [m/s] (default {K0_PHYS:.2e})")
    parser.add_argument("--alpha",   type=float, default=ALPHA,
                        help=f"R₁ charge-transfer coefficient (default {ALPHA})")
    # R₂ parameters (2-species only)
    parser.add_argument("--k0-2",    type=float, default=K0_2_PHYS, dest="k0_2",
                        help=f"R₂ rate constant [m/s] (default {K0_2_PHYS:.2e})")
    parser.add_argument("--alpha-2", type=float, default=ALPHA_2, dest="alpha_2",
                        help=f"R₂ charge-transfer coefficient (default {ALPHA_2})")
    # Common physical parameters
    parser.add_argument("--c-bulk",  type=float, default=C_BULK, dest="c_bulk",
                        help=f"Bulk O₂ concentration [mol/m³] (default {C_BULK})")
    parser.add_argument("--l-ref",   type=float, default=L_REF,  dest="l_ref",
                        help=f"Diffusion layer thickness [m] (default {L_REF:.2e})")
    parser.add_argument("--out-dir", type=str,   default=None,   dest="out_dir",
                        help="Output directory (default: auto-named from non-default params)")
    args = parser.parse_args()

    # Auto-name output directory from whichever params differ from defaults.
    if args.out_dir is None:
        tags = []
        if args.n_species == 1:
            tags.append("1sp")
        if abs(args.steric - A_O2_HAT) > 1e-9:
            tags.append(f"a{args.steric:.2f}")
        if abs(args.l_ref - L_REF) / L_REF > 0.01:
            tags.append(f"L{args.l_ref * 1e6:.0f}um")
        if abs(args.k0 - K0_PHYS) / K0_PHYS > 0.01:
            tags.append(f"k0_{args.k0:.2e}")
        if abs(args.alpha - ALPHA) > 0.01:
            tags.append(f"al{args.alpha:.2f}")
        if args.n_species == 2:
            if abs(args.k0_2 - K0_2_PHYS) / max(K0_2_PHYS, 1e-15) > 0.01:
                tags.append(f"k02_{args.k0_2:.2e}")
            if abs(args.alpha_2 - ALPHA_2) > 0.01:
                tags.append(f"al2_{args.alpha_2:.2f}")
        if abs(args.c_bulk - C_BULK) / C_BULK > 0.01:
            tags.append(f"c{args.c_bulk:.3f}")
        args.out_dir = ("StudyResults/bv_iv_curve" if not tags
                        else f"StudyResults/bv_iv_study/{'_'.join(tags)}")

    result = run_iv_sweep(
        n_species=args.n_species,
        eta_steps=args.steps,
        dt=args.dt,
        ss_tol=args.ss_tol,
        max_ss_steps=args.max_ss_steps,
        a_hat=args.steric,
        k0_phys=args.k0,
        alpha=args.alpha,
        k0_2_phys=args.k0_2,
        alpha_2=args.alpha_2,
        c_bulk=args.c_bulk,
        l_ref=args.l_ref,
        out_dir=args.out_dir,
    )

    plot_iv_curve(result, n_species=args.n_species, a_hat=args.steric,
                  l_ref=args.l_ref, out_dir=args.out_dir)
    plot_iv_comparison(result, n_species=args.n_species, a_hat=args.steric,
                       l_ref=args.l_ref, out_dir=args.out_dir)

    i_scale = N_ELECTRONS * F_CONST * (D_O2 * args.c_bulk / args.l_ref) * 0.1
    print("\n=== BV I-V Curve Summary ===")
    mode = "2-species (O₂+H₂O₂)" if args.n_species == 2 else "1-species (O₂)"
    print(f"Mode: {mode}")
    print(f"Cathodic sweep: V_RHE from {E_EQ_RHE:.3f} V to -0.500 V  ({args.steps} steps)")
    print(f"R₁: k0={args.k0:.2e} m/s, α={args.alpha:.3f}")
    if args.n_species == 2:
        print(f"R₂: k0_2={args.k0_2:.2e} m/s, α₂={args.alpha_2:.3f}")
    print(f"c_bulk={args.c_bulk} mol/m³, L_ref={args.l_ref*1e6:.0f} µm, a={args.steric}")
    print(f"I_scale={i_scale:.4f} mA/cm²  |  I_pxd_lim≈{result['I_peroxide'].min():.4f} mA/cm²")
    print(f"Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
