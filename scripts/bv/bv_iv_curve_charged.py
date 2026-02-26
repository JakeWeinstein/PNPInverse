"""4-species charged PNP Butler-Volmer I-V curve (full Poisson, 2D).

Solves the full Poisson-Nernst-Planck system with 4 transported species on a
2D rectangle mesh graded in y (normal to electrode):

    Species 0: O2     (z=0,  D=1.9e-9 m2/s, c_bulk=0.5 mol/m3)
    Species 1: H2O2   (z=0,  D=1.6e-9 m2/s, c_bulk=0)
    Species 2: H+     (z=+1, D=9.311e-9 m2/s, c_bulk=0.1 mol/m3 at pH 4)
    Species 3: ClO4-  (z=-1, D=1.792e-9 m2/s, c_bulk=0.1 mol/m3)

Reactions:
    R1: O2 + 2H+ + 2e- -> H2O2   (reversible)
    R2: H2O2 + 2H+ + 2e- -> H2O  (irreversible)

Both R1 and R2 have a (c_H+/c_ref_H+)^2 dependence in the cathodic branch,
implemented via the ``cathodic_conc_factors`` mechanism in bv_solver.

Domain: 2D rectangle [0,1] x [0,1], mapped to physical L_ref (default 100 um).
    y=0 (marker 3): electrode (BV BC, phi=eta)
    y=1 (marker 4): bulk (Dirichlet: c=c_bulk, phi=0)
    x=0,1 (markers 1,2): natural zero-flux BCs

At pH 4: lambda_D ~ 30 nm, L = 100 um -> (lambda_D/L)^2 ~ 9e-8.
Mesh grading beta=3 with Ny=300 gives ~20 points in the Debye layer.

Usage (from PNPInverse/ directory)::

    python scripts/bv/bv_iv_curve_charged.py
    python scripts/bv/bv_iv_curve_charged.py --Ny-mesh 500 --beta 3.5
    python scripts/bv/bv_iv_curve_charged.py --l-ref 3e-4 --steps 200
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

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
# Physical constants
# ---------------------------------------------------------------------------

F_CONST   = 96485.3329   # C/mol
R_GAS     = 8.31446      # J/(mol K)
T_REF     = 298.15       # K
V_T       = R_GAS * T_REF / F_CONST   # 0.025693 V
E_EQ_RHE  = 0.695        # V vs RHE (O2/H2O2 at pH 4)
N_ELECTRONS = 2

# --- Species data ---
# Species 0: O2 (neutral)
D_O2   = 1.9e-9     # m2/s
C_O2   = 0.5        # mol/m3 (dissolved O2)

# Species 1: H2O2 (neutral)
D_H2O2 = 1.6e-9     # m2/s
C_H2O2 = 0.0        # mol/m3 (initially zero)

# Species 2: H+ (z=+1)
D_HP   = 9.311e-9   # m2/s
C_HP   = 0.1        # mol/m3 (pH 4 = 10^-4 M = 0.1 mol/m3)

# Species 3: ClO4- (z=-1)
D_CLO4 = 1.792e-9   # m2/s
C_CLO4 = 0.1        # mol/m3 (electroneutrality: equals H+ at pH 4)

# Kinetics
K0_1_PHYS  = 2.4e-8   # m/s (R1)
ALPHA_1    = 0.627
K0_2_PHYS  = 1e-9     # m/s (R2)
ALPHA_2    = 0.5

# Default domain
L_REF = 1.0e-4   # 100 um — diffusion-layer scale

# Reference scales
D_REF = D_O2
C_SCALE = C_O2   # 0.5 mol/m3

# Nondimensional species quantities
D_O2_HAT   = D_O2 / D_REF       # 1.0
D_H2O2_HAT = D_H2O2 / D_REF     # ~0.842
D_HP_HAT   = D_HP / D_REF       # ~4.9
D_CLO4_HAT = D_CLO4 / D_REF     # ~0.943

C_O2_HAT   = C_O2 / C_SCALE     # 1.0
C_H2O2_HAT = C_H2O2 / C_SCALE   # 0.0
C_HP_HAT   = C_HP / C_SCALE     # 0.2
C_CLO4_HAT = C_CLO4 / C_SCALE   # 0.2

# Kappa (transfer coefficient) scale
K_SCALE = D_REF / L_REF

# Current density scale
J_SCALE_MOL_M2_S = D_REF * C_SCALE / L_REF
I_SCALE_A_M2     = N_ELECTRONS * F_CONST * J_SCALE_MOL_M2_S
I_SCALE_MA_CM2   = I_SCALE_A_M2 * 0.1   # A/m2 -> mA/cm2


# ---------------------------------------------------------------------------
# SNES options — conservative Newton for charged system
# ---------------------------------------------------------------------------

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
    "mat_mumps_icntl_8":        77,     # auto-scaling
    "mat_mumps_icntl_14":       80,     # extra memory
}


# ---------------------------------------------------------------------------
# Build SolverParams for charged 4-species problem
# ---------------------------------------------------------------------------

def _make_sp_charged(
    eta_hat: float,
    dt: float,
    t_end: float,
    *,
    Nx_mesh: int = 8,
    Ny_mesh: int = 300,
    beta_mesh: float = 3.0,
    k0_1_phys: float = K0_1_PHYS,
    alpha_1: float = ALPHA_1,
    k0_2_phys: float = K0_2_PHYS,
    alpha_2: float = ALPHA_2,
    l_ref: float = L_REF,
    c_hp: float = C_HP,
    c_support: float = 0.0,
) -> tuple:
    """Return (SolverParams, mesh) for the 4-species charged EDL problem.

    Uses a 2D rectangle mesh graded in y (normal to electrode).
    Electrode at y=0 (bottom, marker 3), bulk at y=1 (top, marker 4).
    Left/right sides (markers 1, 2) get natural zero-flux BCs.

    Parameters
    ----------
    eta_hat : float
        Dimensionless overpotential (eta / V_T).
    dt, t_end : float
        Time-stepping parameters (nondimensional).
    Nx_mesh : int
        Number of cells in x (tangential, uniform).
    Ny_mesh : int
        Number of cells in y (normal to electrode, graded).
    beta_mesh : float
        Power-law grading exponent for mesh (y-direction).
    k0_1_phys, alpha_1 : float
        R1 kinetic parameters.
    k0_2_phys, alpha_2 : float
        R2 kinetic parameters.
    l_ref : float
        Domain length [m].
    c_hp : float
        Bulk H+ concentration [mol/m3].
    c_support : float
        Additional supporting electrolyte (ClO4-) concentration [mol/m3].
        Total ClO4- = c_hp + c_support (for electroneutrality).
    """
    k_scale = D_REF / l_ref
    k0_1_hat = k0_1_phys / k_scale
    k0_2_hat = k0_2_phys / k_scale

    # Nondimensional bulk concentrations
    c_hp_hat = c_hp / C_SCALE
    c_clo4 = c_hp + c_support   # electroneutrality
    c_clo4_hat = c_clo4 / C_SCALE

    # Build 2D mesh: graded in y (electrode at y=0, bulk at y=1)
    mesh = make_graded_rectangle_mesh(Nx_mesh, Ny_mesh, beta_mesh)

    # Solver options
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
        "diffusivity_scale_m2_s":               D_REF,
        "concentration_scale_mol_m3":           C_SCALE,
        "length_scale_m":                       l_ref,
        "potential_scale_v":                    V_T,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }

    # Stoichiometry matrix (species x reactions):
    #          R1    R2
    # O2      -1     0
    # H2O2   +1    -1
    # H+     -2    -2
    # ClO4-   0     0
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": k0_1_hat,
                "alpha": alpha_1,
                "cathodic_species": 0,        # O2 consumed
                "anodic_species": 1,          # H2O2 produced
                "c_ref": 1.0,                 # nondim ref for anodic (= c_O2_bulk)
                "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
            {
                "k0": k0_2_hat,
                "alpha": alpha_2,
                "cathodic_species": 1,        # H2O2 consumed
                "anodic_species": None,       # irreversible
                "c_ref": 0.0,
                "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
        ],
        # Markers for bv_cfg (also read by legacy path for marker info):
        # RectangleMesh: 3=bottom (y=0), 4=top (y=1)
        "k0": [k0_1_hat] * 4,
        "alpha": [alpha_1] * 4,
        "stoichiometry": [-1, +1, -2, 0],
        "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker":      3,   # bottom (y=0)
        "concentration_marker":  4,   # top (y=1)
        "ground_marker":         4,   # top (y=1)
    }

    sp = SolverParams.from_list([
        4,                                                 # n_species
        1,                                                 # FE order
        dt,
        t_end,
        [0, 0, 1, -1],                                    # z_vals
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],   # D_vals (nondim)
        [0.0, 0.0, 0.0, 0.0],                             # a_vals (no steric)
        eta_hat,                                           # phi_applied (= eta_hat)
        [C_O2_HAT, C_H2O2_HAT, c_hp_hat, c_clo4_hat],   # c0_vals (nondim)
        0.0,                                               # phi0
        params,
    ])
    return sp, mesh


# ---------------------------------------------------------------------------
# Main I-V sweep
# ---------------------------------------------------------------------------

def run_iv_sweep_charged(
    *,
    eta_steps: int = 100,
    dt: float = 0.5,
    ss_tol: float = 1e-5,
    max_ss_steps: int = 80,
    Nx_mesh: int = 8,
    Ny_mesh: int = 300,
    beta_mesh: float = 3.0,
    k0_1_phys: float = K0_1_PHYS,
    alpha_1: float = ALPHA_1,
    k0_2_phys: float = K0_2_PHYS,
    alpha_2: float = ALPHA_2,
    l_ref: float = L_REF,
    c_hp: float = C_HP,
    c_support: float = 0.0,
    e_eq_rhe: float = E_EQ_RHE,
    out_dir: str = "StudyResults/bv_iv_curve_charged",
) -> dict:
    """Sweep V_RHE from equilibrium to -0.5 V on the charged 4-species EDL model.

    Returns dict with V_rhe, I_peroxide, I_total arrays.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Current density scale: n_e * F * D_ref * c_scale / l_ref * (A/m2 -> mA/cm2)
    i_scale = N_ELECTRONS * F_CONST * (D_REF * C_SCALE / l_ref) * 0.1

    eta_min = (-0.5 - e_eq_rhe) / V_T
    eta_path = np.linspace(0.0, eta_min, eta_steps + 1)[1:]

    c_clo4 = c_hp + c_support
    lambda_D_m = np.sqrt(8.854e-12 * 78.5 * V_T / (F_CONST * (c_hp + c_clo4)))
    ratio_sq = (lambda_D_m / l_ref) ** 2

    print(f"\n{'='*70}")
    print(f"  4-Species Charged EDL Butler-Volmer I-V Curve")
    print(f"{'='*70}")
    print(f"  Species: O2 (z=0), H2O2 (z=0), H+ (z=+1), ClO4- (z=-1)")
    print(f"  2D mesh: {Nx_mesh}x{Ny_mesh} (graded in y),  beta = {beta_mesh:.1f}")
    print(f"  L_ref = {l_ref*1e6:.2f} um")
    print(f"  lambda_D = {lambda_D_m*1e9:.1f} nm,  (lambda_D/L)^2 = {ratio_sq:.2e}")
    print(f"  c_H+ = {c_hp:.4f} mol/m3 (pH {-np.log10(c_hp/1000):.1f})")
    print(f"  c_ClO4- = {c_clo4:.4f} mol/m3")
    print(f"  R1: k0={k0_1_phys:.2e} m/s, alpha={alpha_1:.3f}")
    print(f"  R2: k0={k0_2_phys:.2e} m/s, alpha={alpha_2:.3f}")
    print(f"  I_scale = {i_scale:.4f} mA/cm2")
    print(f"  eta sweep: 0 -> {eta_min:.1f} V_T ({eta_steps} steps)")
    print(f"  dt={dt}, ss_tol={ss_tol:.0e}, max_ss_steps={max_ss_steps}")
    print(f"{'='*70}\n")

    # Build context at eta=0
    sp_eq, mesh = _make_sp_charged(
        0.0, dt=dt, t_end=dt * max_ss_steps,
        Nx_mesh=Nx_mesh, Ny_mesh=Ny_mesh, beta_mesh=beta_mesh,
        k0_1_phys=k0_1_phys, alpha_1=alpha_1,
        k0_2_phys=k0_2_phys, alpha_2=alpha_2,
        l_ref=l_ref, c_hp=c_hp, c_support=c_support,
    )
    ctx = build_context(sp_eq, mesh=mesh)
    ctx = build_forms(ctx, sp_eq)
    set_initial_conditions(ctx, sp_eq, blob=False)

    scaling = ctx["nondim"]
    dt_model = float(scaling["dt_model"])

    ds = fd.Measure("ds", domain=mesh)
    electrode_marker = 3   # bottom (y=0)

    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    problem = fd.NonlinearVariationalProblem(F_res, ctx["U"], bcs=ctx["bcs"], J=J_form)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

    U_scratch = ctx["U"].copy(deepcopy=True)
    bv_rate_exprs = ctx.get("bv_rate_exprs", [])

    results = []

    for i_step, eta in enumerate(eta_path):
        ctx["phi_applied_func"].assign(float(eta))
        V_rhe = e_eq_rhe + eta * V_T

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
            print(f"  [FAIL] eta={eta:+.2f} V_T (V_RHE={V_rhe:+.3f} V): {e}")
            print(f"         Stopping sweep at step {i_step+1}/{eta_steps}")
            break

        # Compute current density from BV rates
        if len(bv_rate_exprs) >= 2:
            R1_val = float(fd.assemble(bv_rate_exprs[0] * ds(electrode_marker)))
            R2_val = float(fd.assemble(bv_rate_exprs[1] * ds(electrode_marker)))
            I_total_mA    = -(R1_val + R2_val) * i_scale
            I_peroxide_mA = -(R1_val - R2_val) * i_scale
        else:
            I_total_mA = 0.0
            I_peroxide_mA = 0.0

        # Diagnostic: electrode concentrations at (0.5, 0.0) — bottom centre
        c_O2_elec = float(ctx["U"].sub(0).at((0.5, 0.0)))
        c_HP_elec = float(ctx["U"].sub(2).at((0.5, 0.0)))

        results.append((
            eta, V_rhe, I_peroxide_mA, I_total_mA,
            n_taken, int(ss_reached),
            R1_val if len(bv_rate_exprs) >= 2 else 0.0,
            R2_val if len(bv_rate_exprs) >= 2 else 0.0,
            c_O2_elec, c_HP_elec,
        ))

        if (i_step + 1) % 5 == 0 or i_step == 0:
            ss_marker = "+" if ss_reached else "!"
            print(f"  {ss_marker} step {i_step+1:3d}/{eta_steps}  "
                  f"eta={eta:+7.2f}  V_RHE={V_rhe:+6.3f}  "
                  f"steps={n_taken:2d}  "
                  f"I_pxd={I_peroxide_mA:+.4e}  I_tot={I_total_mA:+.4e}  "
                  f"c_O2={c_O2_elec:.4f}  c_H+={c_HP_elec:.4f}")

    if not results:
        print("[ERROR] No results obtained — all steps failed")
        return {"V_rhe": np.array([]), "I_peroxide": np.array([]),
                "I_total": np.array([])}

    # Assemble arrays
    arr = np.array(results)
    V_anodic = np.linspace(1.25, e_eq_rhe, 8, endpoint=False)[::-1]
    n_anodic = len(V_anodic)

    V_cat = arr[:, 1]
    I_pxd_cat = arr[:, 2]
    I_tot_cat = arr[:, 3]

    V_full = np.concatenate([V_anodic, V_cat])
    I_pxd_full = np.concatenate([np.zeros(n_anodic), I_pxd_cat])
    I_tot_full = np.concatenate([np.zeros(n_anodic), I_tot_cat])

    sort_idx = np.argsort(V_full)
    V_rhe_arr = V_full[sort_idx]
    I_pxd_arr = I_pxd_full[sort_idx]
    I_tot_arr = I_tot_full[sort_idx]

    # Save CSV
    csv_path = os.path.join(out_dir, "bv_iv_curve_charged.csv")
    header = ("eta_hat,V_RHE,I_peroxide_mA_cm2,I_total_mA_cm2,"
              "ss_steps,ss_converged,R1,R2,c_O2_electrode,c_HP_electrode")
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    print(f"\n[sweep] CSV saved -> {csv_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    ss_arr = arr[:, 5].astype(bool)
    print(f"  Points: {len(arr)} / {eta_steps}")
    print(f"  SS converged: {ss_arr.sum()} / {len(arr)} "
          f"({100*ss_arr.mean():.0f}%)")
    print(f"  I_peroxide range: {I_pxd_arr.min():.4f} to {I_pxd_arr.max():.4f} mA/cm2")
    print(f"  I_total range:    {I_tot_arr.min():.4f} to {I_tot_arr.max():.4f} mA/cm2")

    # Electrode concentrations at final point
    c_O2_final = arr[-1, 8]
    c_HP_final = arr[-1, 9]
    print(f"  Final electrode: c_O2={c_O2_final:.4f}, c_H+={c_HP_final:.4f} (nondim)")

    return {
        "V_rhe": V_rhe_arr,
        "I_peroxide": I_pxd_arr,
        "I_total": I_tot_arr,
        "raw": arr,
        "ctx": ctx,
        "mesh": mesh,
    }


# ---------------------------------------------------------------------------
# Plot: I-V curve
# ---------------------------------------------------------------------------

def plot_iv_curve_charged(
    result: dict,
    *,
    l_ref: float = L_REF,
    c_hp: float = C_HP,
    out_dir: str = "StudyResults/bv_iv_curve_charged",
) -> str:
    """Plot the charged EDL I-V curve."""
    V = result["V_rhe"]
    I_pxd = result["I_peroxide"]
    I_tot = result["I_total"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(V, I_pxd, color="#c0408a", linewidth=2.0,
            label="$I_{\\mathrm{peroxide}}$ = $2F(R_1 - R_2)$")
    ax.plot(V, I_tot, color="#1f77b4", linewidth=1.5, linestyle="--",
            label="$I_{\\mathrm{total}}$ = $2F(R_1 + R_2)$")

    ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"$E_{{eq}}$ = {E_EQ_RHE:.3f} V vs RHE")
    ax.axhline(0.0, color="black", linewidth=0.5)

    I_lim = float(np.nanmin(I_pxd)) if len(I_pxd) > 0 else 0.0
    if I_lim < -0.001:
        ax.annotate(
            f"$I_{{lim}}$ = {I_lim:.3f} mA/cm$^2$",
            xy=(0.05, I_lim),
            xytext=(0.4, I_lim * 0.6),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8, color="gray",
        )

    pH_val = -np.log10(c_hp / 1000.0)
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=11)
    ax.set_ylabel("Current Density (mA/cm$^2$)", fontsize=11)
    ax.set_title(
        f"4-species charged EDL: O$_2$ + H$_2$O$_2$ + H$^+$ + ClO$_4^-$\n"
        f"$L_\\mathrm{{ref}}={l_ref*1e6:.1f}\\,\\mu$m, "
        f"pH {pH_val:.1f}, Poisson coupled",
        fontsize=10,
    )
    ax.set_xlim(-0.5, 1.25)
    y_min = min(float(np.nanmin(I_pxd)), float(np.nanmin(I_tot))) * 1.15 if len(I_pxd) > 0 else -1.0
    ax.set_ylim(min(y_min, -0.01), 0.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "bv_iv_curve_charged.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] PNG saved -> {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Plot: species profiles at selected voltages
# ---------------------------------------------------------------------------

def plot_species_profiles(
    result: dict,
    *,
    l_ref: float = L_REF,
    out_dir: str = "StudyResults/bv_iv_curve_charged",
) -> str:
    """Plot species concentration and potential profiles from the final state.

    Extracts a 1D y-slice at x=0.5 from the 2D solution.
    """
    ctx = result.get("ctx")
    mesh = result.get("mesh")
    if ctx is None or mesh is None:
        print("[warn] No ctx/mesh in result — skipping profile plot")
        return ""

    U = ctx["U"]
    n = 4
    species_names = ["O$_2$", "H$_2$O$_2$", "H$^+$", "ClO$_4^-$"]
    species_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Extract 1D slice along y at x=0.5 (centre of domain)
    n_pts = 200
    y_pts = np.linspace(0.0, 1.0, n_pts)
    x_fixed = 0.5
    probe_pts = [(x_fixed, y_val) for y_val in y_pts]
    y_phys_um = y_pts * l_ref * 1e6   # y in micrometres

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: concentration profiles
    for i in range(n):
        ci = U.sub(i)
        vals = np.array([ci.at(pt) for pt in probe_pts])
        axes[0].plot(y_phys_um, vals, color=species_colors[i],
                     linewidth=1.5, label=species_names[i])

    axes[0].set_xlabel("$y$ ($\\mu$m)", fontsize=11)
    axes[0].set_ylabel("$\\hat{c}$ (nondim)", fontsize=11)
    axes[0].set_title("Species profiles at final voltage (slice $x=0.5$)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: potential profile
    phi = U.sub(n)
    phi_vals = np.array([phi.at(pt) for pt in probe_pts])
    axes[1].plot(y_phys_um, phi_vals * V_T * 1000,
                 color="black", linewidth=1.5)
    axes[1].set_xlabel("$y$ ($\\mu$m)", fontsize=11)
    axes[1].set_ylabel("$\\phi$ (mV)", fontsize=11)
    axes[1].set_title("Potential profile at final voltage (slice $x=0.5$)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "species_profiles.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Profiles PNG saved -> {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Plot: comparison with neutral 2-species model
# ---------------------------------------------------------------------------

def plot_comparison_with_neutral(
    result_charged: dict,
    *,
    neutral_csv: str = "StudyResults/bv_iv_curve/bv_iv_curve.csv",
    out_dir: str = "StudyResults/bv_iv_curve_charged",
) -> str:
    """Compare charged 4-species I-V with neutral 2-species."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    V_ch = result_charged["V_rhe"]
    I_ch = result_charged["I_peroxide"]
    ax.plot(V_ch, I_ch, color="#c0408a", linewidth=2.0,
            label="Charged EDL (4-species, H$^+$-dependent)")

    if os.path.exists(neutral_csv):
        data = np.loadtxt(neutral_csv, delimiter=",", skiprows=1)
        V_n, I_n = data[:, 0], data[:, 1]
        ax.plot(V_n, I_n, color="#1f77b4", linewidth=1.5, linestyle="--",
                label="Neutral (2-species, no H$^+$)")
    else:
        print(f"[warn] Neutral CSV not found: {neutral_csv}")

    ax.axvline(E_EQ_RHE, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=11)
    ax.set_ylabel("Current Density (mA/cm$^2$)", fontsize=11)
    ax.set_title("Charged EDL vs Neutral: I$_{\\mathrm{peroxide}}$ comparison",
                 fontsize=10)
    ax.set_xlim(-0.5, 1.25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    png_path = os.path.join(out_dir, "charged_vs_neutral.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Comparison PNG saved -> {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="4-species charged EDL BV I-V curve (O2 + H2O2 + H+ + ClO4-)"
    )
    parser.add_argument("--Nx-mesh", type=int, default=8, dest="Nx_mesh",
                        help="Mesh cells in x (tangential, default 8)")
    parser.add_argument("--Ny-mesh", type=int, default=300, dest="Ny_mesh",
                        help="Mesh cells in y (normal to electrode, default 300)")
    parser.add_argument("--beta", type=float, default=3.0,
                        help="Mesh grading exponent in y (default 3.0)")
    parser.add_argument("--l-ref", type=float, default=L_REF, dest="l_ref",
                        help=f"Domain length [m] (default {L_REF:.0e})")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of voltage steps (default 100)")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="Nondim time step (default 0.5)")
    parser.add_argument("--ss-tol", type=float, default=1e-5, dest="ss_tol",
                        help="Steady-state tolerance (default 1e-5)")
    parser.add_argument("--max-ss-steps", type=int, default=80, dest="max_ss_steps",
                        help="Max time steps per voltage (default 80)")
    parser.add_argument("--k0-1", type=float, default=K0_1_PHYS, dest="k0_1",
                        help=f"R1 rate constant [m/s] (default {K0_1_PHYS:.2e})")
    parser.add_argument("--alpha-1", type=float, default=ALPHA_1, dest="alpha_1",
                        help=f"R1 transfer coefficient (default {ALPHA_1})")
    parser.add_argument("--k0-2", type=float, default=K0_2_PHYS, dest="k0_2",
                        help=f"R2 rate constant [m/s] (default {K0_2_PHYS:.2e})")
    parser.add_argument("--alpha-2", type=float, default=ALPHA_2, dest="alpha_2",
                        help=f"R2 transfer coefficient (default {ALPHA_2})")
    parser.add_argument("--c-hp", type=float, default=C_HP, dest="c_hp",
                        help=f"Bulk H+ concentration [mol/m3] (default {C_HP})")
    parser.add_argument("--c-support", type=float, default=0.0, dest="c_support",
                        help="Supporting electrolyte [mol/m3] (default 0)")
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir",
                        help="Output directory (default auto)")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = "StudyResults/bv_iv_curve_charged"

    result = run_iv_sweep_charged(
        eta_steps=args.steps,
        dt=args.dt,
        ss_tol=args.ss_tol,
        max_ss_steps=args.max_ss_steps,
        Nx_mesh=args.Nx_mesh,
        Ny_mesh=args.Ny_mesh,
        beta_mesh=args.beta,
        k0_1_phys=args.k0_1,
        alpha_1=args.alpha_1,
        k0_2_phys=args.k0_2,
        alpha_2=args.alpha_2,
        l_ref=args.l_ref,
        c_hp=args.c_hp,
        c_support=args.c_support,
        out_dir=args.out_dir,
    )

    if len(result["V_rhe"]) > 0:
        plot_iv_curve_charged(result, l_ref=args.l_ref, c_hp=args.c_hp,
                              out_dir=args.out_dir)
        plot_species_profiles(result, l_ref=args.l_ref, out_dir=args.out_dir)
        plot_comparison_with_neutral(result, out_dir=args.out_dir)

    print(f"\n=== Charged EDL I-V Curve Complete ===")
    print(f"Output directory: {args.out_dir}/")


if __name__ == "__main__":
    main()
