"""MMS convergence test for 3-species + Boltzmann ClO4- log-c PNP-BV solver.

Verifies correctness of the forward solver used in v18/v19 onset inference:
  - Three dynamic species: O2 (z=0), H2O2 (z=0), H+ (z=+1)
  - ClO4- replaced by Boltzmann background: c_ClO4 = c_bulk * exp(phi)
  - Log-concentration transform: unknown is u_i = ln(c_i), c_i = exp(u_i)

Follows writeups/WeekOfFeb25/mms_butler_volmer.tex. Manufactured solution:

  c_i^ex(x,y) = c0_i + A_i * cos(pi*x) * (1 - exp(-beta_i * y))
  phi^ex(x,y) = eta0 * (1-y) + B * cos(pi*x) * y*(1-y)
  u_i^ex(x,y) = ln(c_i^ex)     (positive by construction if |A_i| < c0_i)

Volume sources (UFL auto-differentiation):
  S_c_i = -div[D_i * (grad(c_i^ex) + em * z_i * c_i^ex * grad(phi^ex))]
  S_phi = -eps_hat * div(grad(phi^ex))
          - charge_rhs * sum_i(z_i * c_i^ex)
          + charge_rhs * c_ClO4_bulk * exp(phi^ex)    <- Boltzmann term

Boundary correction at electrode (y=0): g_i = flux_outward - sum_j(s_ij * R_j^ex).

Expected: L2 rate -> 2.0, H1 rate -> 1.0 (CG1) for both u_i and phi.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import pi, log

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

from Forward.bv_solver.forms_logc import (
    build_context_logc, build_forms_logc,
)
from Forward.params import SolverParams


SNES_OPTS = {
    "snes_type":                   "newtonls",
    "snes_max_it":                 200,
    "snes_atol":                   1e-12,
    "snes_rtol":                   1e-12,
    "snes_stol":                   1e-14,
    "snes_linesearch_type":        "l2",
    "snes_linesearch_maxlambda":   1.0,
    "snes_divergence_tolerance":   1e12,
    "ksp_type":                    "preonly",
    "pc_type":                     "lu",
    "pc_factor_mat_solver_type":   "mumps",
    "mat_mumps_icntl_14":          100,
    "mat_mumps_icntl_8":           77,
}


# ---------------------------------------------------------------------------
# MMS parameters (3 species + Boltzmann ClO4-)
# ---------------------------------------------------------------------------
N_SPECIES = 3
SPECIES_NAMES = ["O2", "H2O2", "H+"]

Z_VALS = [0, 0, 1]
D_VALS = [1.0, 0.85, 4.9]           # nondim (matches production for O2-ref)
# NOTE: We deliberately DO NOT use the production H2O2 seed of 1e-4 here.
# With c_0,H2O2=1e-4, u_1 = ln(c_1) has gradients of order 1/c_1 ≈ 1e4,
# which CG1 cannot resolve; Newton converges to a wrong root.
# MMS checks the solver's implementation of the log-c PDE -- we want a
# well-behaved manufactured solution. The seed×physics interaction at
# small c is a separate physical modeling question, not a solver correctness issue.
C0_VALS = [1.0, 0.5, 0.2]           # bulk concs (nondim)
A_VALS = [0.2, 0.1, 0.05]           # amplitudes: |A_i| < C0_i (positivity)
BETA_VALS = [3.0, 3.0, 3.0]

ETA0 = -2.0                         # moderate cathodic overpotential
B_VAL = 0.1

# Boltzmann ClO4- parameters
C_CLO4_BULK = 0.2                   # nondim bulk ClO4- concentration

# BV kinetic parameters (for MMS; generic moderate values)
K0_1 = 0.5
ALPHA_1 = 0.5
C_REF_ANODIC = 1.0
K0_2 = 0.1
ALPHA_2 = 0.5
C_REF_H_PLUS = 0.2                  # matches C0_VALS[2] so H_factor at surface = 1.0

# Stoichiometry: R1 (O2 -> H2O2), R2 (H2O2 -> H2O)
STOI_R1 = [-1, +1, -2]              # per-species for R1 (consumes O2, produces H2O2, consumes 2 H+)
STOI_R2 = [ 0, -1, -2]              # per-species for R2


# ---------------------------------------------------------------------------
# Build SolverParams
# ---------------------------------------------------------------------------

def make_sp_mms(eta_hat: float) -> SolverParams:
    """3-species SolverParams with nondim coefficients chosen for clean MMS values."""
    D_REF = 1.9e-9
    C_SCALE = 0.5
    L_REF = 1.0e-4
    R_GAS = 8.314462618
    F_CONST = 96485.3329
    T_REF = 298.15
    V_T = R_GAS * T_REF / F_CONST

    # Target eps_hat ~ 0.01 for moderate Debye length
    target_eps_hat = 0.01
    perm_needed = target_eps_hat * F_CONST * C_SCALE * L_REF**2 / V_T

    dt = 1e15
    t_end = 1e15

    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent":            False,
        "exponent_clip":            50.0,
        "regularize_concentration": False,
        "conc_floor":               1e-8,
        "use_eta_in_bv":            True,
    }
    params["nondim"] = {
        "enabled":                              True,
        "diffusivity_scale_m2_s":               D_REF,
        "concentration_scale_mol_m3":           C_SCALE,
        "length_scale_m":                       L_REF,
        "potential_scale_v":                    V_T,
        "permittivity_f_m":                     perm_needed,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_1, "alpha": ALPHA_1,
                "cathodic_species": 0, "anodic_species": 1,
                "c_ref": C_REF_ANODIC,
                "stoichiometry": STOI_R1,
                "n_electrons": 2, "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_REF_H_PLUS},
                ],
            },
            {
                "k0": K0_2, "alpha": ALPHA_2,
                "cathodic_species": 1, "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": STOI_R2,
                "n_electrons": 2, "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_REF_H_PLUS},
                ],
            },
        ],
        "k0":            [K0_1] * N_SPECIES,
        "alpha":         [ALPHA_1] * N_SPECIES,
        "stoichiometry": STOI_R1,
        "c_ref":         [1.0] * N_SPECIES,
        "E_eq_v":        0.0,
        "electrode_marker":     3,      # bottom y=0
        "concentration_marker": 4,      # top y=1
        "ground_marker":        4,
    }

    sp = SolverParams.from_list([
        N_SPECIES, 1, dt, t_end,
        Z_VALS, D_VALS, [0.0] * N_SPECIES,  # no steric
        eta_hat, C0_VALS, 0.0, params,
    ])
    return sp


# ---------------------------------------------------------------------------
# MMS convergence study
# ---------------------------------------------------------------------------

def run_mms(N_vals: list[int], verbose: bool = True) -> dict:
    n = N_SPECIES
    electrode_marker = 3
    bulk_marker = 4

    print("=" * 80)
    print("  MMS Convergence: 3-Species + Boltzmann ClO4- log-c PNP-BV")
    print("=" * 80)
    print(f"  Species: {SPECIES_NAMES}")
    print(f"  z = {Z_VALS}, D = {D_VALS}, c0 = {C0_VALS}")
    print(f"  A  = {A_VALS}, beta = {BETA_VALS}")
    print(f"  eta0 = {ETA0}, B = {B_VAL}, c_ClO4_bulk = {C_CLO4_BULK}")
    print(f"  R1: k0={K0_1}, alpha={ALPHA_1}, stoi={STOI_R1}")
    print(f"  R2: k0={K0_2}, alpha={ALPHA_2}, stoi={STOI_R2}")
    print(f"  c_ref_H+ = {C_REF_H_PLUS}")
    print(f"  Mesh sizes: {N_vals}")
    print("=" * 80)

    results = {"N": [], "h": []}
    for i in range(n):
        results[f"u{i}_L2"] = []
        results[f"u{i}_H1"] = []
        results[f"c{i}_L2"] = []
    results["phi_L2"] = []
    results["phi_H1"] = []

    for N in N_vals:
        t0 = time.time()
        h = 1.0 / N

        # Mesh with marker 3 = bottom, 4 = top (use UnitSquareMesh and remap)
        # forms_logc expects electrode_marker=3 (bottom), concentration_marker=4 (top)
        # UnitSquareMesh: 1=left, 2=right, 3=bottom, 4=top. That matches.
        mesh = fd.UnitSquareMesh(N, N)
        x, y = fd.SpatialCoordinate(mesh)

        # Build forms via the production pipeline
        sp = make_sp_mms(ETA0)
        ctx = build_context_logc(sp, mesh=mesh)
        ctx = build_forms_logc(ctx, sp)

        # Read nondim coefficients actually used by the solver
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        eps_hat = float(scaling["poisson_coefficient"])
        charge_rhs = float(scaling["charge_rhs_prefactor"])
        D_model = [float(scaling["D_model_vals"][i]) for i in range(n)]
        z_float = [float(Z_VALS[i]) for i in range(n)]

        if N == N_vals[0]:
            print(f"\n  [nondim] em={em:.4f}, eps_hat={eps_hat:.4f}, "
                  f"charge_rhs={charge_rhs:.4f}")
            print(f"  [nondim] D_model = {D_model}")

        # -- Add Boltzmann ClO4- background to F_res --
        # Matches add_boltzmann from v19_hybrid_forward:
        # z_ClO4 = -1, so ρ contribution = -c_bulk*exp(phi). Residual subtracts
        # (charge_rhs * z * c_ClO4) -> subtract (charge_rhs * (-1) * c_bulk * exp(phi))
        W = ctx["W"]; U = ctx["U"]
        phi_sym = fd.split(U)[-1]
        w_test = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=mesh)
        ds_m = fd.Measure("ds", domain=mesh)
        n_vec = fd.FacetNormal(mesh)

        # Note: NO clipping on phi_sym here — MMS uses moderate phi. Clipping would
        # add a nonlinearity that doesn't match the smooth manufactured phi.
        ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * C_CLO4_BULK * fd.exp(phi_sym) * w_test * dx

        # ---- Manufactured solutions ----
        c_exact = [
            C0_VALS[i] + A_VALS[i] * fd.cos(pi * x) * (1.0 - fd.exp(-BETA_VALS[i] * y))
            for i in range(n)
        ]
        u_exact = [fd.ln(c_exact[i]) for i in range(n)]
        phi_exact = ETA0 * (1.0 - y) + B_VAL * fd.cos(pi * x) * y * (1.0 - y)

        # ---- Volume sources ----
        # NP: S_c_i = -div(D_i * (grad(c_i^ex) + em * z_i * c_i^ex * grad(phi_ex)))
        S_c = []
        for i in range(n):
            J_i = D_model[i] * (
                fd.grad(c_exact[i]) + em * z_float[i] * c_exact[i] * fd.grad(phi_exact)
            )
            S_c.append(-fd.div(J_i))

        # Poisson (with Boltzmann ClO4-):
        # F_res_phi_residual = -eps * laplacian(phi) - charge_rhs * sum(z_i c_i)
        #                     + charge_rhs * c_bulk * exp(phi)
        # S_phi matches this residual at U_manuf so F_res_MMS[U_manuf] = 0:
        S_phi = (
            -eps_hat * fd.div(fd.grad(phi_exact))
            - charge_rhs * sum(z_float[i] * c_exact[i] for i in range(n))
            + charge_rhs * C_CLO4_BULK * fd.exp(phi_exact)
        )

        # ---- Manufactured BV rates at electrode surface ----
        # MUST match forms_logc.py build_forms_logc rate expression:
        #   cathodic = k0 * c_cat_surf * exp(-alpha * n_e * eta)  [* H_factor if present]
        #   anodic   = k0 * c_anod_surf * exp((1-alpha) * n_e * eta)   (anodic_species != None)
        #              OR k0 * c_ref * exp((1-alpha) * n_e * eta)      (anodic_species = None, reversible)
        #              OR 0                                             (irreversible)
        # At y=0: c_i^ex(x,0) = C0_VALS[i], phi^ex(x,0) = ETA0.
        N_E = 2.0
        H_factor = (C0_VALS[2] / C_REF_H_PLUS) ** 2

        # R1: reversible, anodic_species=1 (H2O2). Anodic uses c_H2O2_surf = C0_VALS[1].
        R1_exact = (
            K0_1 * C0_VALS[0] * H_factor * fd.exp(-ALPHA_1 * N_E * ETA0)
            - K0_1 * C0_VALS[1] * fd.exp((1.0 - ALPHA_1) * N_E * ETA0)
        )
        # R2: irreversible (anodic = 0). Cathodic uses c_H2O2_surf = C0_VALS[1] with H_factor.
        R2_exact = K0_2 * C0_VALS[1] * H_factor * fd.exp(-ALPHA_2 * N_E * ETA0)
        R_exact = [R1_exact, R2_exact]
        STOI = [STOI_R1, STOI_R2]      # reactions index
        # per-species stoichiometric contribution
        # STOI_PER_SPECIES[i][j] = stoichiometry of species i in reaction j
        STOI_PER_SPECIES = [[STOI_R1[i], STOI_R2[i]] for i in range(n)]

        # ---- Boundary correction g_i on electrode ----
        # g_i = [D*(grad(c)+em*z*c*grad(phi))]·n_outward - sum_j(s_ij * R_j^ex)
        g_corr = []
        for i in range(n):
            J_i_exact = D_model[i] * (
                fd.grad(c_exact[i]) + em * z_float[i] * c_exact[i] * fd.grad(phi_exact)
            )
            flux_outward = fd.dot(J_i_exact, n_vec)
            bv_sum = sum(float(STOI_PER_SPECIES[i][j]) * R_exact[j] for j in range(2))
            g_corr.append(flux_outward - bv_sum)

        # ---- Inject MMS sources (SUBTRACT from F_res) ----
        F_res = ctx["F_res"]
        v_tests = fd.TestFunctions(W)
        v_list = v_tests[:-1]

        for i in range(n):
            F_res -= S_c[i] * v_list[i] * dx
            F_res -= g_corr[i] * v_list[i] * ds_m(electrode_marker)
        F_res -= S_phi * w_test * dx

        # ---- Replace Dirichlet BCs ----
        # u_i on bulk = ln(c_i^ex(x,1)) -- space-varying (depends on x)
        # phi on bulk = 0 (phi_ex(x,1) = 0 by construction)
        # phi on electrode = ETA0 (phi_ex(x,0) = ETA0 by construction)
        bcs_new = []
        for i in range(n):
            bcs_new.append(fd.DirichletBC(W.sub(i), u_exact[i], bulk_marker))
        bcs_new.append(fd.DirichletBC(W.sub(n), fd.Constant(ETA0), electrode_marker))
        bcs_new.append(fd.DirichletBC(W.sub(n), fd.Constant(0.0), bulk_marker))
        ctx["bcs"] = bcs_new

        # ---- Initial guess at manufactured solution ----
        for i in range(n):
            U.sub(i).interpolate(u_exact[i])
        U.sub(n).interpolate(phi_exact)
        ctx["U_prev"].assign(U)

        # ---- Solve ----
        J_form = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs_new, J=J_form)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)
        try:
            solver.solve()
        except fd.ConvergenceError as e:
            print(f"  [FAIL] N={N}: Newton failed: {e}")
            continue

        # ---- Errors ----
        V_scalar = ctx["V_scalar"]
        errs_u_L2, errs_u_H1, errs_c_L2 = [], [], []
        for i in range(n):
            u_ex_f = fd.Function(V_scalar); u_ex_f.interpolate(u_exact[i])
            u_L2 = fd.errornorm(u_ex_f, U.sub(i), norm_type="L2")
            u_H1 = fd.errornorm(u_ex_f, U.sub(i), norm_type="H1")
            errs_u_L2.append(u_L2); errs_u_H1.append(u_H1)
            # Also check c = exp(u) L2 error (for interpretability)
            c_ex_f = fd.Function(V_scalar); c_ex_f.interpolate(c_exact[i])
            c_h = fd.Function(V_scalar); c_h.interpolate(fd.exp(U.sub(i)))
            c_L2 = fd.errornorm(c_ex_f, c_h, norm_type="L2")
            errs_c_L2.append(c_L2)

        phi_ex_f = fd.Function(V_scalar); phi_ex_f.interpolate(phi_exact)
        phi_L2 = fd.errornorm(phi_ex_f, U.sub(n), norm_type="L2")
        phi_H1 = fd.errornorm(phi_ex_f, U.sub(n), norm_type="H1")

        elapsed = time.time() - t0
        results["N"].append(N)
        results["h"].append(h)
        for i in range(n):
            results[f"u{i}_L2"].append(errs_u_L2[i])
            results[f"u{i}_H1"].append(errs_u_H1[i])
            results[f"c{i}_L2"].append(errs_c_L2[i])
        results["phi_L2"].append(phi_L2)
        results["phi_H1"].append(phi_H1)

        if verbose:
            parts = [f"N={N:4d}  h={h:.5f}"]
            for i in range(n):
                parts.append(f"u{i}_L2={errs_u_L2[i]:.3e}")
            parts.append(f"phi_L2={phi_L2:.3e}")
            parts.append(f"({elapsed:.1f}s)")
            print("  " + "  ".join(parts))

    return results


# ---------------------------------------------------------------------------
# Rate computation + summary
# ---------------------------------------------------------------------------

def compute_rates(h_list, err_list):
    rates = [None]
    for k in range(1, len(h_list)):
        if err_list[k] > 0 and err_list[k - 1] > 0:
            rates.append(log(err_list[k - 1] / err_list[k]) / log(h_list[k - 1] / h_list[k]))
        else:
            rates.append(None)
    return rates


def format_summary(results: dict) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  3-sp + Boltzmann log-c MMS: Convergence Rate Summary")
    lines.append("=" * 80)

    h_list = results["h"]
    n = N_SPECIES
    all_pass = True

    for i in range(n):
        for norm in ("L2", "H1"):
            key = f"u{i}_{norm}"
            rates = compute_rates(h_list, results[key])
            final = rates[-1] if rates[-1] is not None else 0.0
            expected = 2.0 if norm == "L2" else 1.0
            lo = 1.85 if norm == "L2" else 0.85
            hi = 2.15 if norm == "L2" else 999.0
            status = "PASS" if lo <= final <= hi else "FAIL"
            if status == "FAIL":
                all_pass = False
            lines.append(
                f"  {SPECIES_NAMES[i]:>5s} u{i} {norm}: rate = {final:.4f}  "
                f"(expected ~{expected:.1f})  [{status}]"
            )
        # Also report c_i L2 rate
        key = f"c{i}_L2"
        rates = compute_rates(h_list, results[key])
        final = rates[-1] if rates[-1] is not None else 0.0
        lines.append(
            f"  {SPECIES_NAMES[i]:>5s} c{i} L2 (=exp(u)): rate = {final:.4f}"
        )

    for norm in ("L2", "H1"):
        key = f"phi_{norm}"
        rates = compute_rates(h_list, results[key])
        final = rates[-1] if rates[-1] is not None else 0.0
        expected = 2.0 if norm == "L2" else 1.0
        lo = 1.85 if norm == "L2" else 0.85
        hi = 2.15 if norm == "L2" else 999.0
        status = "PASS" if lo <= final <= hi else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  {'phi':>5s}    {norm}: rate = {final:.4f}  "
            f"(expected ~{expected:.1f})  [{status}]"
        )

    lines.append("-" * 80)
    lines.append(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_table(results: dict) -> str:
    lines = ["", "=" * 100]
    lines.append("  Full Error Table (u_i = ln(c_i) primary unknown)")
    lines.append("=" * 100)
    h_list = results["h"]
    n = N_SPECIES

    fields = []
    for i in range(n):
        fields.append(f"u{i}_L2"); fields.append(f"u{i}_H1")
    fields.append("phi_L2"); fields.append("phi_H1")

    header = f"  {'N':>4} {'h':>8}  "
    for fn in fields:
        header += f"{fn:>10}  {'rate':>5}  "
    lines.append(header)
    lines.append("-" * 100)

    rates = {fn: compute_rates(h_list, results[fn]) for fn in fields}
    for i in range(len(results["N"])):
        row = f"  {results['N'][i]:>4} {results['h'][i]:>8.4f}  "
        for fn in fields:
            r = rates[fn][i]
            r_str = f"{r:.2f}" if r is not None else "---"
            row += f"{results[fn][i]:>10.3e}  {r_str:>5}  "
        lines.append(row)
    lines.append("=" * 100)
    return "\n".join(lines)


def plot_convergence(results: dict, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = np.array(results["h"])
    n = N_SPECIES
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    # L2
    ax = axes[0]
    for i in range(n):
        ax.loglog(h, results[f"u{i}_L2"], "o-", color=colors[i], linewidth=1.5,
                  markersize=5, label=f"{SPECIES_NAMES[i]} $u_{i}$ $L^2$")
    ax.loglog(h, results["phi_L2"], "s-", color=colors[3], linewidth=1.5,
              markersize=5, label="$\\phi$ $L^2$")
    h_ref = np.array([h[0], h[-1]])
    scale = results["u0_L2"][0] / h[0] ** 2
    ax.loglog(h_ref, scale * h_ref ** 2, "k:", linewidth=0.8, label="$O(h^2)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("$L^2$ error")
    ax.set_title("$L^2$ Convergence (log-c 3sp + Boltzmann)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # H1
    ax = axes[1]
    for i in range(n):
        ax.loglog(h, results[f"u{i}_H1"], "o-", color=colors[i], linewidth=1.5,
                  markersize=5, label=f"{SPECIES_NAMES[i]} $u_{i}$ $H^1$")
    ax.loglog(h, results["phi_H1"], "s-", color=colors[3], linewidth=1.5,
              markersize=5, label="$\\phi$ $H^1$")
    scale = results["u0_H1"][0] / h[0] ** 1
    ax.loglog(h_ref, scale * h_ref ** 1, "k-.", linewidth=0.8, label="$O(h^1)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("$H^1$ error")
    ax.set_title("$H^1$ Convergence")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("MMS: 3sp + Boltzmann ClO4- log-c PNP-BV", fontsize=11)
    plt.tight_layout()
    png = os.path.join(out_dir, "mms_3sp_logc_boltzmann.png")
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nvals", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128])
    parser.add_argument("--out-dir", default="StudyResults/mms_3sp_logc_boltzmann")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = run_mms(args.Nvals)
    if len(results["N"]) < 2:
        print("\n[ERROR] Not enough successful solves for convergence analysis.")
        sys.exit(1)

    print(format_table(results))
    print(format_summary(results))

    summary_path = os.path.join(args.out_dir, "mms_3sp_logc_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"3sp + Boltzmann log-c MMS study\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(format_table(results) + "\n\n")
        f.write(format_summary(results) + "\n")
    print(f"\n[MMS] Summary saved -> {summary_path}")

    png = plot_convergence(results, args.out_dir)
    print(f"[MMS] Plot saved -> {png}")


if __name__ == "__main__":
    main()
