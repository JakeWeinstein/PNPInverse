"""4-Species MMS convergence test using the PRODUCTION solver pipeline.

Closes all five verification gaps between the existing MMS suite and the
production 4-species charged PNP Butler-Volmer solver:

  1. cathodic_conc_factors -- (c_H+/c_ref_H+)^2 in BV rates
  2. Neutral + charged species in the same MixedFunctionSpace (z=[0,0,+1,-1])
  3. Uses the actual bv_solver.build_context/build_forms pipeline (not hand-built forms)
  4. Stoichiometry magnitude |s|=2 (H+ consumed with coefficient -2)
  5. 4-species, 5-component MixedFunctionSpace

Manufactured solutions on [0,1]^2:

  c_i^ex(x,y) = c0_i + A_i * cos(pi*x) * (1 - exp(-beta_i * y))
  phi^ex(x,y) = eta0 * (1-y) + B * cos(pi*x) * y*(1-y)

Electrode y=0 (marker 3), bulk y=1 (marker 4).

Volume sources computed via UFL auto-differentiation.
Boundary corrections account for the mismatch between manufactured flux
and manufactured BV rate at the electrode.

Expected: L2 rate ~ 2.0, H1 rate ~ 1.0 (CG1).

Usage (from PNPInverse/ directory)::

    python InferenceScripts/mms_bv_4species.py
    python InferenceScripts/mms_bv_4species.py --Nvals 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import pi, log

from Forward.bv_solver import build_context, build_forms
from Forward.params import SolverParams


# ---------------------------------------------------------------------------
# SNES options -- direct solver (MUMPS), tight tolerances for MMS
# ---------------------------------------------------------------------------
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
# MMS parameters
# ---------------------------------------------------------------------------
N_SPECIES = 4
SPECIES_NAMES = ["O2", "H2O2", "H+", "ClO4-"]

# Charges
Z_VALS = [0, 0, 1, -1]

# Nondimensional diffusivities
D_VALS = [1.0, 0.85, 4.9, 0.94]

# Background concentrations (nondim)
C0_VALS = [1.0, 0.5, 0.2, 0.2]

# Manufactured solution perturbation amplitudes
A_VALS = [0.2, 0.1, 0.05, 0.05]

# Boundary-layer steepness parameters
BETA_VALS = [3.0, 2.5, 2.0, 2.0]

# Potential parameters
ETA0 = -2.0    # nondim overpotential at electrode (moderate cathodic)
B_VAL = 0.1    # potential perturbation amplitude

# BV kinetic parameters
K0_1 = 0.5     # nondim R1 rate constant
ALPHA_1 = 0.5  # R1 transfer coefficient
C_REF_ANODIC = 1.0  # nondim ref conc for R1 anodic branch
K0_2 = 0.1     # nondim R2 rate constant
ALPHA_2 = 0.5  # R2 transfer coefficient
C_REF_H_PLUS = 0.2  # nondim ref conc for H+ in cathodic_conc_factors

# Stoichiometry matrix [species][reaction]
#       R1   R2
# O2   -1    0
# H2O2 +1   -1
# H+   -2   -2
# ClO4- 0    0
STOI = [[-1, 0], [+1, -1], [-2, -2], [0, 0]]


# ---------------------------------------------------------------------------
# Build SolverParams for the MMS problem
# ---------------------------------------------------------------------------

def _make_sp_mms(eta_hat: float) -> SolverParams:
    """Build SolverParams for the 4-species MMS problem.

    Uses all *_inputs_are_dimensionless = True so that the nondim
    pipeline passes model-space values through unchanged.  The
    Poisson coefficient (eps_hat) is set explicitly via permittivity
    and scales so it comes out to a moderate value.

    Key: potential_scale = V_T so electromigration_prefactor = 1.0.
    """
    # Physical scales -- chosen so the nondim pipeline produces
    # clean, controllable coefficients.
    D_REF = 1.9e-9       # m2/s (O2 diffusivity)
    C_SCALE = 0.5        # mol/m3
    L_REF = 1.0e-4       # 100 um
    R_GAS = 8.314462618
    F_CONST = 96485.3329
    T_REF = 298.15
    V_T = R_GAS * T_REF / F_CONST  # ~0.02569 V

    # We want eps_hat = permittivity * V_T / (F * c_scale * L^2)
    # Choose permittivity to give a reasonable eps_hat ~ 0.01
    # eps_hat = perm * 0.02569 / (96485 * 0.5 * 1e-8)
    #         = perm * 0.02569 / 4.824e-4
    #         = perm * 53.26
    # For eps_hat = 0.01: perm = 0.01/53.26 = 1.878e-4
    target_eps_hat = 0.01
    perm_needed = target_eps_hat * F_CONST * C_SCALE * L_REF**2 / V_T

    # Very large dt for steady state (1/dt ~ 0)
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
                "k0": K0_1,
                "alpha": ALPHA_1,
                "cathodic_species": 0,        # O2 consumed
                "anodic_species": 1,          # H2O2 produced
                "c_ref": C_REF_ANODIC,       # nondim ref for anodic
                "stoichiometry": STOI[0] + STOI[1] + STOI[2] + STOI[3],
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_REF_H_PLUS},
                ],
            },
            {
                "k0": K0_2,
                "alpha": ALPHA_2,
                "cathodic_species": 1,        # H2O2 consumed
                "anodic_species": None,       # irreversible
                "c_ref": 0.0,
                "stoichiometry": STOI[0] + STOI[1] + STOI[2] + STOI[3],
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_REF_H_PLUS},
                ],
            },
        ],
        # Legacy per-species fields needed by _get_bv_cfg for marker info:
        "k0": [K0_1] * N_SPECIES,
        "alpha": [ALPHA_1] * N_SPECIES,
        "stoichiometry": [s[0] for s in STOI],
        "c_ref": [1.0] * N_SPECIES,
        "E_eq_v": 0.0,
        "electrode_marker":      3,   # bottom (y=0)
        "concentration_marker":  4,   # top (y=1)
        "ground_marker":         4,   # top (y=1)
    }

    sp = SolverParams.from_list([
        N_SPECIES,           # n_species
        1,                   # order (CG1)
        dt,
        t_end,
        Z_VALS,              # z_vals
        D_VALS,              # D_vals (nondim)
        [0.0] * N_SPECIES,   # a_vals (no steric)
        eta_hat,             # phi_applied (= eta_hat)
        C0_VALS,             # c0_vals (nondim)
        0.0,                 # phi0
        params,
    ])
    return sp


# ---------------------------------------------------------------------------
# Fix stoichiometry: the production code flattens STOI into a single list
# for "stoichiometry" per reaction. We need [s_00, s_10, s_20, s_30] for R1.
# Looking at the config: stoichiometry length must == n_species.
# So stoichiometry for R1 = [-1, +1, -2, 0] (per-species for that reaction)
# and for R2 = [0, -1, -2, 0].
# ---------------------------------------------------------------------------

def _make_sp_mms_fixed(eta_hat: float) -> SolverParams:
    """Build SolverParams with correctly formatted stoichiometry."""
    D_REF = 1.9e-9
    C_SCALE = 0.5
    L_REF = 1.0e-4
    R_GAS = 8.314462618
    F_CONST = 96485.3329
    T_REF = 298.15
    V_T = R_GAS * T_REF / F_CONST

    target_eps_hat = 0.01
    perm_needed = target_eps_hat * F_CONST * C_SCALE * L_REF**2 / V_T

    dt = 1e15
    t_end = 1e15

    # Per-reaction stoichiometry (each is a list of length n_species):
    # R1: O2(-1), H2O2(+1), H+(-2), ClO4-(0)
    stoi_R1 = [-1, +1, -2, 0]
    # R2: O2(0), H2O2(-1), H+(-2), ClO4-(0)
    stoi_R2 = [0, -1, -2, 0]

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
                "k0": K0_1,
                "alpha": ALPHA_1,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": C_REF_ANODIC,
                "stoichiometry": stoi_R1,
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_REF_H_PLUS},
                ],
            },
            {
                "k0": K0_2,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": stoi_R2,
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_REF_H_PLUS},
                ],
            },
        ],
        "k0": [K0_1] * N_SPECIES,
        "alpha": [ALPHA_1] * N_SPECIES,
        "stoichiometry": stoi_R1,
        "c_ref": [1.0] * N_SPECIES,
        "E_eq_v": 0.0,
        "electrode_marker":      3,
        "concentration_marker":  4,
        "ground_marker":         4,
    }

    sp = SolverParams.from_list([
        N_SPECIES,
        1,
        dt,
        t_end,
        Z_VALS,
        D_VALS,
        [0.0] * N_SPECIES,
        eta_hat,
        C0_VALS,
        0.0,
        params,
    ])
    return sp


# ---------------------------------------------------------------------------
# MMS convergence study
# ---------------------------------------------------------------------------

def run_mms_4species(
    N_vals: list[int],
    *,
    verbose: bool = True,
) -> dict:
    """Run MMS convergence study for the 4-species charged BV solver.

    For each mesh size N:
      1. Build a RectangleMesh(N, N, 1, 1).
      2. Call build_context and build_forms from the production solver.
      3. Construct manufactured solutions and compute MMS sources via UFL.
      4. Inject sources into F_res, replace BCs, set initial guess.
      5. Solve and compute L2/H1 errors for all 5 fields.

    Returns dict with N, h, and errors for each field.
    """
    n = N_SPECIES
    electrode_marker = 3
    bulk_marker = 4

    print("\n" + "=" * 75)
    print("  MMS Convergence Study: 4-Species Charged PNP + 2 BV Reactions")
    print("  (Production Solver Pipeline)")
    print("=" * 75)
    print(f"  Species: {SPECIES_NAMES}")
    print(f"  z = {Z_VALS}")
    print(f"  D = {D_VALS}")
    print(f"  c0 = {C0_VALS}")
    print(f"  eta0 = {ETA0}, B = {B_VAL}")
    print(f"  k0_1 = {K0_1}, alpha_1 = {ALPHA_1}")
    print(f"  k0_2 = {K0_2}, alpha_2 = {ALPHA_2}")
    print(f"  c_ref_H+ = {C_REF_H_PLUS}")
    print(f"  Stoichiometry: {STOI}")
    print(f"  Mesh sizes: {N_vals}")
    print("=" * 75 + "\n")

    results = {"N": [], "h": []}
    for i in range(n):
        results[f"c{i}_L2"] = []
        results[f"c{i}_H1"] = []
    results["phi_L2"] = []
    results["phi_H1"] = []

    for N in N_vals:
        t0 = time.time()
        h = 1.0 / N

        # --- Build mesh ---
        mesh = fd.RectangleMesh(N, N, 1.0, 1.0)
        x, y = fd.SpatialCoordinate(mesh)

        # --- Build SolverParams and call production solver pipeline ---
        sp = _make_sp_mms_fixed(ETA0)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)

        # --- Read back actual nondim coefficients from the solver ---
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        eps_hat = float(scaling["poisson_coefficient"])
        charge_rhs = float(scaling["charge_rhs_prefactor"])

        # Read D model values (these came through exp(logD))
        D_model = [float(scaling["D_model_vals"][i]) for i in range(n)]
        z_float = [float(Z_VALS[i]) for i in range(n)]

        if N == N_vals[0]:
            print(f"  [nondim] em = {em:.6f}")
            print(f"  [nondim] eps_hat = {eps_hat:.6f}")
            print(f"  [nondim] charge_rhs = {charge_rhs:.6f}")
            print(f"  [nondim] D_model = {D_model}")
            print(f"  [nondim] dt_model = {float(scaling['dt_model']):.2e}")
            print(f"  [nondim] phi_applied_model = {float(scaling['phi_applied_model']):.6f}")
            print()

        W = ctx["W"]
        U = ctx["U"]

        n_vec = fd.FacetNormal(mesh)
        ds = fd.Measure("ds", domain=mesh)
        dx = fd.Measure("dx", domain=mesh)

        # --- Manufactured solutions (UFL expressions on the mesh) ---
        c_exact = []
        for i in range(n):
            c_ex_i = C0_VALS[i] + A_VALS[i] * fd.cos(pi * x) * (1.0 - fd.exp(-BETA_VALS[i] * y))
            c_exact.append(c_ex_i)
        phi_exact = ETA0 * (1.0 - y) + B_VAL * fd.cos(pi * x) * y * (1.0 - y)

        # --- Volume source terms (UFL auto-diff) ---
        # NP flux: J_i = D_i * (grad(c_i) + em * z_i * c_i * grad(phi))
        # Source: S_i = -div(J_i)
        S_c = []
        for i in range(n):
            J_i = D_model[i] * (
                fd.grad(c_exact[i]) + em * z_float[i] * c_exact[i] * fd.grad(phi_exact)
            )
            S_c.append(-fd.div(J_i))

        # Poisson source: -eps_hat * laplacian(phi) - charge_rhs * sum(z_i * c_i) = 0
        # => S_phi = -eps_hat * div(grad(phi_exact)) - charge_rhs * sum(z_i * c_exact[i])
        S_phi = -eps_hat * fd.div(fd.grad(phi_exact)) - charge_rhs * sum(
            z_float[i] * c_exact[i] for i in range(n)
        )

        # --- Manufactured BV rates at the electrode surface ---
        # At y=0: c_i^ex(x,0) = c0_i (constant), phi^ex(x,0) = eta0.
        # H+ factor: (c0_H+/c_ref_H+)^2
        H_factor = (C0_VALS[2] / C_REF_H_PLUS) ** 2  # = (0.2/0.2)^2 = 1.0

        # R1 = k0_1 * c_O2(y=0) * H_factor * exp(-alpha1*eta0)
        #     - k0_1 * c_ref_anodic * exp((1-alpha1)*eta0)
        R1_exact = K0_1 * (
            C0_VALS[0] * H_factor * fd.exp(-ALPHA_1 * ETA0)
            - C_REF_ANODIC * fd.exp((1.0 - ALPHA_1) * ETA0)
        )

        # R2 = k0_2 * c_H2O2(y=0) * H_factor * exp(-alpha2*eta0)
        R2_exact = K0_2 * C0_VALS[1] * H_factor * fd.exp(-ALPHA_2 * ETA0)

        R_exact = [R1_exact, R2_exact]

        # --- Boundary correction ---
        # g_i = D_i * dot(J_i_exact, n_outward) - sum_j(stoi[i][j] * R_j_exact)
        # where J_i_exact includes electromigration for charged species
        g_corr = []
        for i in range(n):
            # Full flux including electromigration
            J_i_exact = D_model[i] * (
                fd.grad(c_exact[i]) + em * z_float[i] * c_exact[i] * fd.grad(phi_exact)
            )
            flux_outward = fd.dot(J_i_exact, n_vec)

            # BV stoichiometric sum for species i
            bv_sum = sum(float(STOI[i][j]) * R_exact[j] for j in range(2))

            g_i = flux_outward - bv_sum
            g_corr.append(g_i)

        # --- Get test functions (same symbolic objects as build_forms used) ---
        v_tests = fd.TestFunctions(W)
        v_list = v_tests[:-1]
        w_test = v_tests[-1]

        # --- Inject MMS sources into the solver's F_res ---
        F_res = ctx["F_res"]

        for i in range(n):
            # Volume source
            F_res -= S_c[i] * v_list[i] * dx
            # Boundary correction at electrode
            F_res -= g_corr[i] * v_list[i] * ds(electrode_marker)

        # Poisson volume source
        F_res -= S_phi * w_test * dx

        # --- Replace Dirichlet BCs ---
        # The manufactured solution at y=1 depends on x, so we must use
        # the full expression (not a constant).
        bcs_new = []
        for i in range(n):
            bcs_new.append(fd.DirichletBC(W.sub(i), c_exact[i], bulk_marker))
        # phi Dirichlet at electrode (y=0): phi_exact(x,0) = eta0
        bcs_new.append(fd.DirichletBC(W.sub(n), fd.Constant(ETA0), electrode_marker))
        # phi Dirichlet at bulk (y=1): phi_exact(x,1) = 0
        bcs_new.append(fd.DirichletBC(W.sub(n), fd.Constant(0.0), bulk_marker))
        ctx["bcs"] = bcs_new

        # --- Set initial guess to manufactured solution ---
        for i in range(n):
            U.sub(i).interpolate(c_exact[i])
        U.sub(n).interpolate(phi_exact)
        ctx["U_prev"].assign(U)

        # --- Solve ---
        J_form = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs_new, J=J_form)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

        try:
            solver.solve()
        except fd.ConvergenceError as e:
            print(f"  [FAIL] N={N}: Newton failed: {e}")
            continue

        # --- Compute errors ---
        V_scalar = ctx["V_scalar"]

        errs_L2 = []
        errs_H1 = []
        for i in range(n):
            c_ex_f = fd.Function(V_scalar)
            c_ex_f.interpolate(c_exact[i])
            c_L2 = fd.errornorm(c_ex_f, U.sub(i), norm_type="L2")
            c_H1 = fd.errornorm(c_ex_f, U.sub(i), norm_type="H1")
            errs_L2.append(c_L2)
            errs_H1.append(c_H1)

        phi_ex_f = fd.Function(V_scalar)
        phi_ex_f.interpolate(phi_exact)
        phi_L2 = fd.errornorm(phi_ex_f, U.sub(n), norm_type="L2")
        phi_H1 = fd.errornorm(phi_ex_f, U.sub(n), norm_type="H1")

        elapsed = time.time() - t0

        results["N"].append(N)
        results["h"].append(h)
        for i in range(n):
            results[f"c{i}_L2"].append(errs_L2[i])
            results[f"c{i}_H1"].append(errs_H1[i])
        results["phi_L2"].append(phi_L2)
        results["phi_H1"].append(phi_H1)

        if verbose:
            parts = [f"N={N:4d}  h={h:.5f}"]
            for i in range(n):
                parts.append(f"c{i}_L2={errs_L2[i]:.4e}")
            parts.append(f"phi_L2={phi_L2:.4e}")
            parts.append(f"({elapsed:.1f}s)")
            print("  " + "  ".join(parts))

    return results


# ---------------------------------------------------------------------------
# Convergence rate computation
# ---------------------------------------------------------------------------

def compute_rates(h_list, err_list):
    """Compute convergence rates: rate_k = log(e_{k-1}/e_k) / log(h_{k-1}/h_k)."""
    rates = [None]
    for k in range(1, len(h_list)):
        if err_list[k] > 0 and err_list[k-1] > 0:
            rates.append(log(err_list[k-1] / err_list[k]) / log(h_list[k-1] / h_list[k]))
        else:
            rates.append(None)
    return rates


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def format_table(results: dict) -> str:
    """Format the convergence table for the 4-species test."""
    lines = []
    lines.append("")
    lines.append("=" * 150)
    lines.append("  4-Species Charged PNP + 2 BV Reactions -- MMS Convergence Table")
    lines.append("  (Production solver pipeline: build_context + build_forms)")
    lines.append("=" * 150)

    h_list = results["h"]
    n = N_SPECIES

    # Build column names
    field_names = []
    for i in range(n):
        field_names.append(f"c{i}_L2")
        field_names.append(f"c{i}_H1")
    field_names.append("phi_L2")
    field_names.append("phi_H1")

    # Header
    header = f"{'N':>6s}  {'h':>9s}  "
    for fn in field_names:
        header += f"{fn:>10s}  {'rate':>6s}  "
    lines.append(header)
    lines.append("-" * 150)

    all_rates = {}
    for fn in field_names:
        all_rates[fn] = compute_rates(h_list, results[fn])

    for i in range(len(results["N"])):
        row = f"{results['N'][i]:6d}  {results['h'][i]:9.5f}  "
        for fn in field_names:
            r = all_rates[fn][i]
            r_str = f"{r:.2f}" if r is not None else "---"
            row += f"{results[fn][i]:10.4e}  {r_str:>6s}  "
        lines.append(row)

    lines.append("-" * 150)
    lines.append("  Expected: L2 rate -> 2.00, H1 rate -> 1.00 (CG1)")
    lines.append("=" * 150)
    return "\n".join(lines)


def format_summary_table(results: dict) -> str:
    """Format a compact summary showing only the final convergence rates."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  Final Convergence Rates (finest mesh pair)")
    lines.append("=" * 70)

    h_list = results["h"]
    n = N_SPECIES
    all_pass = True

    for i in range(n):
        for norm in ("L2", "H1"):
            key = f"c{i}_{norm}"
            rates = compute_rates(h_list, results[key])
            final_rate = rates[-1] if rates[-1] is not None else 0.0
            expected = 2.0 if norm == "L2" else 1.0
            lo = 1.90 if norm == "L2" else 0.90
            hi = 2.10 if norm == "L2" else 999.0
            status = "PASS" if lo <= final_rate <= hi else "FAIL"
            if status == "FAIL":
                all_pass = False
            lines.append(
                f"  {SPECIES_NAMES[i]:>5s} {norm:>2s}: rate = {final_rate:.4f}  "
                f"(expected ~{expected:.1f})  [{status}]"
            )

    for norm in ("L2", "H1"):
        key = f"phi_{norm}"
        rates = compute_rates(h_list, results[key])
        final_rate = rates[-1] if rates[-1] is not None else 0.0
        expected = 2.0 if norm == "L2" else 1.0
        lo = 1.90 if norm == "L2" else 0.90
        hi = 2.10 if norm == "L2" else 999.0
        status = "PASS" if lo <= final_rate <= hi else "FAIL"
        if status == "FAIL":
            all_pass = False
        lines.append(
            f"  {'phi':>5s} {norm:>2s}: rate = {final_rate:.4f}  "
            f"(expected ~{expected:.1f})  [{status}]"
        )

    lines.append("-" * 70)
    lines.append(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(results: dict, out_dir: str) -> str:
    """Create log-log convergence plot for all 5 fields."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = np.array(results["h"])
    n = N_SPECIES

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Left plot: L2 errors
    ax = axes[0]
    for i in range(n):
        key = f"c{i}_L2"
        err = np.array(results[key])
        ax.loglog(h, err, "o-", color=colors[i], linewidth=1.5, markersize=5,
                  label=f"{SPECIES_NAMES[i]} $L^2$")
    err_phi = np.array(results["phi_L2"])
    ax.loglog(h, err_phi, "s-", color=colors[4], linewidth=1.5, markersize=5,
              label="$\\phi$ $L^2$")

    # O(h^2) reference
    h_ref = np.array([h[0], h[-1]])
    scale = results["c0_L2"][0] / h[0]**2
    ax.loglog(h_ref, scale * h_ref**2, "k:", linewidth=0.8, label="$O(h^2)$")

    ax.set_xlabel("$h$", fontsize=12)
    ax.set_ylabel("$L^2$ Error", fontsize=12)
    ax.set_title("$L^2$ Convergence", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, which="both")

    # Right plot: H1 errors
    ax = axes[1]
    for i in range(n):
        key = f"c{i}_H1"
        err = np.array(results[key])
        ax.loglog(h, err, "o-", color=colors[i], linewidth=1.5, markersize=5,
                  label=f"{SPECIES_NAMES[i]} $H^1$")
    err_phi = np.array(results["phi_H1"])
    ax.loglog(h, err_phi, "s-", color=colors[4], linewidth=1.5, markersize=5,
              label="$\\phi$ $H^1$")

    scale = results["c0_H1"][0] / h[0]**1
    ax.loglog(h_ref, scale * h_ref**1, "k-.", linewidth=0.8, label="$O(h^1)$")

    ax.set_xlabel("$h$", fontsize=12)
    ax.set_ylabel("$H^1$ Error", fontsize=12)
    ax.set_title("$H^1$ Convergence", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "4-Species MMS: O$_2$ + H$_2$O$_2$ + H$^+$ + ClO$_4^-$\n"
        "(Production pipeline: build_context + build_forms + cathodic_conc_factors + |s|=2)",
        fontsize=10,
    )
    plt.tight_layout()

    png_path = os.path.join(out_dir, "mms_bv_4species.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="4-species MMS convergence test (production solver pipeline)"
    )
    parser.add_argument(
        "--Nvals", type=int, nargs="+", default=[8, 16, 32, 64, 128],
        help="Mesh sizes (default: 8 16 32 64 128)"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="StudyResults/mms_bv_4species",
        dest="out_dir",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = run_mms_4species(args.Nvals)

    if len(results["N"]) < 2:
        print("\n[ERROR] Not enough successful solves for convergence analysis.")
        sys.exit(1)

    # Print tables
    table = format_table(results)
    print(table)
    summary = format_summary_table(results)
    print(summary)

    # Save
    summary_path = os.path.join(args.out_dir, "mms_4species_summary.txt")
    with open(summary_path, "w") as f:
        f.write("4-Species MMS Convergence Study (Production Pipeline)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(table + "\n\n")
        f.write(summary + "\n")
    print(f"\n[MMS] Summary saved -> {summary_path}")

    png_path = plot_convergence(results, args.out_dir)
    print(f"[MMS] Plot saved -> {png_path}")

    print("\n=== 4-Species MMS Convergence Study Complete ===")


if __name__ == "__main__":
    main()
