"""Method of Manufactured Solutions (MMS) convergence study for PNP-BV solver.

Verifies the Butler-Volmer PNP solver by demonstrating optimal convergence
rates on problems with known exact solutions.

Three test cases:
  1. Single neutral species (z=0) + single irreversible BV reaction.
  2. Two neutral species + 2 BV reactions (O2, H2O2 system).
  3. Two charged species (z=+1, z=-1) + Poisson coupling + 1 BV reaction.

The manufactured solutions are (nondimensional, on unit square):

  c_i^ex(x,y) = c0_i + A_i * cos(pi*x) * (1 - exp(-beta_i * y))
  phi^ex(x,y) = eta0 * (1 - y) + B * cos(pi*x) * y * (1 - y)

These satisfy:
  - Zero x-flux at x=0,1 (compatible with zero-flux side walls)
  - phi^ex(x,0) = eta0 (electrode Dirichlet)
  - phi^ex(x,1) = 0 (ground)
  - c_i^ex(x,0) = c0_i (uniform at electrode)
  - Nonzero y-flux at y=0 (exercises BV BC)
  - Positivity: |A_i| < c0_i
  - Non-polynomial (trig + exp, so CG1 cannot represent exactly)

Volume source S_i computed via UFL auto-differentiation.
Boundary correction g_i accounts for mismatch between manufactured flux
and manufactured BV rate at the electrode.

SIGN CONVENTION (critical):
  The solver assembles F_res -= stoi * R * v * ds(electrode), which via IBP
  implements the BC:  D * grad(c) . n_outward = -stoi * R  at the electrode.
  (n_outward points out of domain, i.e., (0,-1) at y=0.)

  The IBP boundary integral from the diffusion bilinear form contributes:
    + D * dot(grad(c), n_outward) * v * ds    (at the electrode)

  So at the exact solution, the residual's boundary part is:
    [D * dot(grad(c_exact), n_outward) - stoi * R_bv_exact] * v * ds

  The correction g makes this zero:
    g = D * dot(grad(c_exact), n_outward) - stoi * R_bv_exact

  We add F_res -= g * v * ds(electrode) to absorb this mismatch.

Expected rates for CG1:  L2 -> O(h^2),  H1 -> O(h).

Usage (from PNPInverse/ directory)::

    python scripts/verification/mms_bv_convergence.py
    python scripts/verification/mms_bv_convergence.py --case all
    python scripts/verification/mms_bv_convergence.py --case single --Nvals 8 16 32 64 128
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

from math import pi, log


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
}


# ===========================================================================
# Helper: compute boundary correction g for one species
# ===========================================================================

def _boundary_correction_neutral(
    D_hat, c_exact, n_vec, stoi_list, R_bv_exact_list
):
    """Compute g = D*dot(grad(c_exact), n_outward) - sum_j(s_ij * R_j_exact).

    This is the mismatch between the manufactured diffusive flux at the
    electrode (in the outward-normal direction) and the BV contribution
    that the solver applies.  Subtracting g*v*ds from F_res ensures the
    manufactured solution satisfies the modified problem exactly.

    Parameters
    ----------
    D_hat : float
        Nondimensional diffusivity.
    c_exact : UFL expression
        Manufactured solution for this species.
    n_vec : UFL FacetNormal
        Outward facet normal.
    stoi_list : list of float
        Stoichiometric coefficients [s_i1, s_i2, ...] for this species.
    R_bv_exact_list : list of UFL expressions
        Manufactured BV rates [R1_exact, R2_exact, ...].

    Returns
    -------
    UFL expression for g_i.
    """
    # Manufactured outward flux: D * dot(grad(c), n_outward)
    flux_outward = D_hat * fd.dot(fd.grad(c_exact), n_vec)

    # Sum of stoichiometric BV contributions
    bv_sum = sum(s * R for s, R in zip(stoi_list, R_bv_exact_list))

    # g = flux_outward - sum(s_ij * R_j)
    return flux_outward - bv_sum


def _boundary_correction_charged(
    D_hat, z_i, c_exact, phi_exact, n_vec, stoi_list, R_bv_exact_list
):
    """Like _boundary_correction_neutral but includes electromigration flux."""
    J_exact = D_hat * (fd.grad(c_exact) + z_i * c_exact * fd.grad(phi_exact))
    flux_outward = fd.dot(J_exact, n_vec)
    bv_sum = sum(s * R for s, R in zip(stoi_list, R_bv_exact_list))
    return flux_outward - bv_sum


# ===========================================================================
# Case 1: Single neutral species, single irreversible BV reaction
# ===========================================================================

def run_mms_single_species(
    N_vals: list[int],
    *,
    verbose: bool = True,
) -> dict:
    """MMS convergence study: 1 species (z=0), 1 irreversible BV reaction.

    PDE (steady state, nondim):
        -D * laplacian(c) = S  in Omega
        BV flux at electrode (y=0)
        c = c_exact(x,1) at bulk (y=1)
        zero flux at sides (x=0,1)

    BV rate: R = k0 * c_surf * exp(-alpha * eta_hat)  (cathodic only)

    Domain: unit square [0,1]^2.
    RectangleMesh markers: 1=left, 2=right, 3=bottom, 4=top.
    Electrode = bottom (marker 3), bulk = top (marker 4).
    """
    # --- MMS parameters ---
    D_hat   = 1.0     # nondim diffusivity
    c0_val  = 1.0     # background concentration
    A_val   = 0.2     # perturbation amplitude (|A| < c0 for positivity)
    beta_c  = 3.0     # boundary layer steepness
    eta0    = -2.0    # nondim overpotential (moderate cathodic)
    B_val   = 0.1     # potential perturbation amplitude
    k0_hat  = 0.5     # nondim BV rate constant
    alpha   = 0.5     # transfer coefficient
    eps_hat = 0.01    # Poisson coefficient (lambda_D/L)^2

    # Stoichiometry: species 0 consumed by reaction 1
    stoi = -1.0

    print("\n" + "=" * 70)
    print("  MMS Convergence Study: Single Neutral Species + BV")
    print("=" * 70)
    print(f"  D_hat={D_hat}, c0={c0_val}, A={A_val}, beta={beta_c}")
    print(f"  eta0={eta0}, B={B_val}, k0={k0_hat}, alpha={alpha}")
    print(f"  eps_hat={eps_hat}, stoi={stoi}")
    print(f"  Mesh sizes: {N_vals}")
    print("=" * 70 + "\n")

    results = {
        "N": [], "h": [],
        "c_L2": [], "c_H1": [],
        "phi_L2": [], "phi_H1": [],
    }

    for N in N_vals:
        t0 = time.time()
        h = 1.0 / N

        # --- Build mesh ---
        mesh = fd.RectangleMesh(N, N, 1.0, 1.0)
        x, y = fd.SpatialCoordinate(mesh)

        # Markers: 1=left, 2=right, 3=bottom (electrode), 4=top (bulk)
        electrode_marker = 3
        bulk_marker = 4

        # --- Function spaces ---
        V = fd.FunctionSpace(mesh, "CG", 1)
        W = fd.MixedFunctionSpace([V, V])  # [c, phi]
        U = fd.Function(W, name="U")
        v_c, v_phi = fd.TestFunctions(W)

        c_h, phi_h = fd.split(U)

        # --- Manufactured solutions (UFL expressions) ---
        c_exact = c0_val + A_val * fd.cos(pi * x) * (1.0 - fd.exp(-beta_c * y))
        phi_exact = eta0 * (1.0 - y) + B_val * fd.cos(pi * x) * y * (1.0 - y)

        # --- Volume source ---
        # S_c = -div(D * grad(c_exact))  [z=0, no electromigration]
        J_exact = D_hat * fd.grad(c_exact)
        S_c = -fd.div(J_exact)

        # Poisson source: -eps * laplacian(phi_exact) = 0 + S_phi  [z=0]
        S_phi = -eps_hat * fd.div(fd.grad(phi_exact))

        # --- Boundary correction ---
        n_vec = fd.FacetNormal(mesh)
        ds = fd.Measure("ds", domain=mesh)
        dx = fd.Measure("dx", domain=mesh)

        # Manufactured BV rate at surface values (constants):
        # c_exact(x,0) = c0_val, phi_exact(x,0) = eta0
        R_bv_exact = k0_hat * c0_val * fd.exp(-alpha * eta0)

        # g = D * dot(grad(c_exact), n_outward) - stoi * R_bv_exact
        g_0 = _boundary_correction_neutral(
            D_hat, c_exact, n_vec,
            stoi_list=[stoi], R_bv_exact_list=[R_bv_exact],
        )

        # --- Assemble weak form ---
        # Diffusion bilinear form (NP for z=0):
        F_res = D_hat * fd.dot(fd.grad(c_h), fd.grad(v_c)) * dx

        # BV flux at electrode (on NUMERICAL unknowns):
        # R_bv = k0 * c_h * exp(-alpha * eta0)  (irreversible cathodic)
        R_bv_numerical = k0_hat * c_h * fd.exp(-alpha * eta0)
        F_res -= stoi * R_bv_numerical * v_c * ds(electrode_marker)

        # MMS volume source
        F_res -= S_c * v_c * dx

        # MMS boundary correction
        F_res -= g_0 * v_c * ds(electrode_marker)

        # Poisson equation:
        F_res += eps_hat * fd.dot(fd.grad(phi_h), fd.grad(v_phi)) * dx
        F_res -= S_phi * v_phi * dx

        # --- Dirichlet BCs ---
        bc_c_bulk = fd.DirichletBC(W.sub(0), c_exact, bulk_marker)
        bc_phi_elec = fd.DirichletBC(W.sub(1), fd.Constant(eta0), electrode_marker)
        bc_phi_bulk = fd.DirichletBC(W.sub(1), fd.Constant(0.0), bulk_marker)
        bcs = [bc_c_bulk, bc_phi_elec, bc_phi_bulk]

        # --- Initial guess: interpolate manufactured solution ---
        U.sub(0).interpolate(c_exact)
        U.sub(1).interpolate(phi_exact)

        # --- Solve ---
        J_form = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

        try:
            solver.solve()
        except fd.ConvergenceError as e:
            print(f"  [FAIL] N={N}: Newton failed: {e}")
            continue

        # --- Compute errors ---
        c_h_sol = U.sub(0)
        phi_h_sol = U.sub(1)

        c_exact_func = fd.Function(V, name="c_exact")
        c_exact_func.interpolate(c_exact)
        phi_exact_func = fd.Function(V, name="phi_exact")
        phi_exact_func.interpolate(phi_exact)

        c_L2 = fd.errornorm(c_exact_func, c_h_sol, norm_type="L2")
        c_H1 = fd.errornorm(c_exact_func, c_h_sol, norm_type="H1")
        phi_L2 = fd.errornorm(phi_exact_func, phi_h_sol, norm_type="L2")
        phi_H1 = fd.errornorm(phi_exact_func, phi_h_sol, norm_type="H1")

        elapsed = time.time() - t0

        results["N"].append(N)
        results["h"].append(h)
        results["c_L2"].append(c_L2)
        results["c_H1"].append(c_H1)
        results["phi_L2"].append(phi_L2)
        results["phi_H1"].append(phi_H1)

        if verbose:
            print(f"  N={N:4d}  h={h:.5f}  "
                  f"c_L2={c_L2:.4e}  c_H1={c_H1:.4e}  "
                  f"phi_L2={phi_L2:.4e}  phi_H1={phi_H1:.4e}  "
                  f"({elapsed:.1f}s)")

    return results


# ===========================================================================
# Case 2: Two neutral species, two reactions (O2 + H2O2 system)
# ===========================================================================

def run_mms_two_species(
    N_vals: list[int],
    *,
    verbose: bool = True,
) -> dict:
    """MMS convergence study: 2 neutral species, 2 BV reactions.

    Species 0: O2  (z=0)  consumed by R1
    Species 1: H2O2 (z=0) produced by R1, consumed by R2

    Reactions:
      R1 = k01 * [c_O2 * exp(-alpha1*eta) - c_ref1 * exp((1-alpha1)*eta)]
      R2 = k02 * c_H2O2 * exp(-alpha2 * eta)  (irreversible)

    Stoichiometry:  s = [[-1, 0], [+1, -1]]
    """
    D0_hat = 1.0
    D1_hat = 0.85
    c0_0   = 1.0      # O2 background
    c0_1   = 0.5      # H2O2 background (nonzero for MMS)
    A0     = 0.2
    A1     = 0.15
    beta0  = 3.0
    beta1  = 2.5
    eta0   = -2.0
    B_val  = 0.1
    eps_hat = 0.01
    k01    = 0.5
    alpha1 = 0.5
    c_ref1 = 1.0
    k02    = 0.1
    alpha2 = 0.5

    # Stoichiometry matrix [species][reaction]
    stoi = [[-1.0, 0.0],   # species 0 (O2)
            [+1.0, -1.0]]  # species 1 (H2O2)

    print("\n" + "=" * 70)
    print("  MMS Convergence Study: Two Neutral Species + 2 BV Reactions")
    print("=" * 70)
    print(f"  D0={D0_hat}, D1={D1_hat}")
    print(f"  c0_O2={c0_0}, c0_H2O2={c0_1}, A0={A0}, A1={A1}")
    print(f"  beta0={beta0}, beta1={beta1}, eta0={eta0}, B={B_val}")
    print(f"  k01={k01}, alpha1={alpha1}, c_ref1={c_ref1}")
    print(f"  k02={k02}, alpha2={alpha2}, eps_hat={eps_hat}")
    print(f"  Mesh sizes: {N_vals}")
    print("=" * 70 + "\n")

    results = {
        "N": [], "h": [],
        "c0_L2": [], "c0_H1": [],
        "c1_L2": [], "c1_H1": [],
        "phi_L2": [], "phi_H1": [],
    }

    for N in N_vals:
        t0 = time.time()
        h = 1.0 / N

        mesh = fd.RectangleMesh(N, N, 1.0, 1.0)
        x, y = fd.SpatialCoordinate(mesh)
        electrode_marker = 3
        bulk_marker = 4

        V = fd.FunctionSpace(mesh, "CG", 1)
        W = fd.MixedFunctionSpace([V, V, V])  # [c0, c1, phi]
        U = fd.Function(W, name="U")
        v0, v1, w = fd.TestFunctions(W)
        c0_h, c1_h, phi_h = fd.split(U)

        # --- Manufactured solutions ---
        c0_exact = c0_0 + A0 * fd.cos(pi * x) * (1.0 - fd.exp(-beta0 * y))
        c1_exact = c0_1 + A1 * fd.cos(pi * x) * (1.0 - fd.exp(-beta1 * y))
        phi_exact = eta0 * (1.0 - y) + B_val * fd.cos(pi * x) * y * (1.0 - y)

        # --- Volume sources (z=0 for both) ---
        S_c0 = -fd.div(D0_hat * fd.grad(c0_exact))
        S_c1 = -fd.div(D1_hat * fd.grad(c1_exact))
        S_phi = -eps_hat * fd.div(fd.grad(phi_exact))  # z=0: no charge

        # --- Boundary correction ---
        n_vec = fd.FacetNormal(mesh)
        ds = fd.Measure("ds", domain=mesh)
        dx = fd.Measure("dx", domain=mesh)

        # Manufactured BV rates at surface: c0_exact(x,0)=c0_0, c1_exact(x,0)=c0_1
        R1_exact = k01 * (c0_0 * fd.exp(-alpha1 * eta0)
                          - c_ref1 * fd.exp((1.0 - alpha1) * eta0))
        R2_exact = k02 * c0_1 * fd.exp(-alpha2 * eta0)

        g_0 = _boundary_correction_neutral(
            D0_hat, c0_exact, n_vec,
            stoi_list=stoi[0], R_bv_exact_list=[R1_exact, R2_exact],
        )
        g_1 = _boundary_correction_neutral(
            D1_hat, c1_exact, n_vec,
            stoi_list=stoi[1], R_bv_exact_list=[R1_exact, R2_exact],
        )

        # --- Weak form ---
        F_res = D0_hat * fd.dot(fd.grad(c0_h), fd.grad(v0)) * dx
        F_res += D1_hat * fd.dot(fd.grad(c1_h), fd.grad(v1)) * dx

        # BV flux (numerical unknowns)
        R1_num = k01 * (c0_h * fd.exp(-alpha1 * eta0)
                        - c_ref1 * fd.exp((1.0 - alpha1) * eta0))
        R2_num = k02 * c1_h * fd.exp(-alpha2 * eta0)

        for j, R_num in enumerate([R1_num, R2_num]):
            F_res -= stoi[0][j] * R_num * v0 * ds(electrode_marker)
            F_res -= stoi[1][j] * R_num * v1 * ds(electrode_marker)

        # MMS sources
        F_res -= S_c0 * v0 * dx
        F_res -= S_c1 * v1 * dx
        F_res -= g_0 * v0 * ds(electrode_marker)
        F_res -= g_1 * v1 * ds(electrode_marker)

        # Poisson
        F_res += eps_hat * fd.dot(fd.grad(phi_h), fd.grad(w)) * dx
        F_res -= S_phi * w * dx

        # --- BCs ---
        bcs = [
            fd.DirichletBC(W.sub(0), c0_exact, bulk_marker),
            fd.DirichletBC(W.sub(1), c1_exact, bulk_marker),
            fd.DirichletBC(W.sub(2), fd.Constant(eta0), electrode_marker),
            fd.DirichletBC(W.sub(2), fd.Constant(0.0), bulk_marker),
        ]

        # --- Initial guess ---
        U.sub(0).interpolate(c0_exact)
        U.sub(1).interpolate(c1_exact)
        U.sub(2).interpolate(phi_exact)

        # --- Solve ---
        J_form = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

        try:
            solver.solve()
        except fd.ConvergenceError as e:
            print(f"  [FAIL] N={N}: Newton failed: {e}")
            continue

        # --- Errors ---
        c0_h_sol = U.sub(0)
        c1_h_sol = U.sub(1)
        phi_h_sol = U.sub(2)

        c0_ex_f = fd.Function(V); c0_ex_f.interpolate(c0_exact)
        c1_ex_f = fd.Function(V); c1_ex_f.interpolate(c1_exact)
        phi_ex_f = fd.Function(V); phi_ex_f.interpolate(phi_exact)

        c0_L2 = fd.errornorm(c0_ex_f, c0_h_sol, norm_type="L2")
        c0_H1 = fd.errornorm(c0_ex_f, c0_h_sol, norm_type="H1")
        c1_L2 = fd.errornorm(c1_ex_f, c1_h_sol, norm_type="L2")
        c1_H1 = fd.errornorm(c1_ex_f, c1_h_sol, norm_type="H1")
        phi_L2 = fd.errornorm(phi_ex_f, phi_h_sol, norm_type="L2")
        phi_H1 = fd.errornorm(phi_ex_f, phi_h_sol, norm_type="H1")

        elapsed = time.time() - t0

        results["N"].append(N)
        results["h"].append(h)
        results["c0_L2"].append(c0_L2)
        results["c0_H1"].append(c0_H1)
        results["c1_L2"].append(c1_L2)
        results["c1_H1"].append(c1_H1)
        results["phi_L2"].append(phi_L2)
        results["phi_H1"].append(phi_H1)

        if verbose:
            print(f"  N={N:4d}  h={h:.5f}  "
                  f"c0_L2={c0_L2:.4e}  c0_H1={c0_H1:.4e}  "
                  f"c1_L2={c1_L2:.4e}  c1_H1={c1_H1:.4e}  "
                  f"phi_L2={phi_L2:.4e}  phi_H1={phi_H1:.4e}  "
                  f"({elapsed:.1f}s)")

    return results


# ===========================================================================
# Case 3: Two charged species + Poisson coupling
# ===========================================================================

def run_mms_charged(
    N_vals: list[int],
    *,
    verbose: bool = True,
) -> dict:
    """MMS convergence study: 2 charged species (z=+1, z=-1) + Poisson + 1 BV reaction.

    Species 0: cation (z=+1)  consumed by reaction
    Species 1: anion  (z=-1)  no reaction (stoi=0)

    Reaction:
      R = k0 * c0_surf * exp(-alpha * eta)  (irreversible cathodic)
    Stoichiometry: s = [-1, 0]

    This tests electromigration coupling and the Poisson source term.
    """
    D0_hat = 1.0
    D1_hat = 0.8
    c0_0   = 1.0     # cation background
    c0_1   = 1.0     # anion background
    A0     = 0.15
    A1     = 0.12
    beta0  = 3.0
    beta1  = 2.5
    eta0   = -1.0
    B_val  = 0.05
    eps_hat = 0.01
    z0     = 1.0
    z1     = -1.0
    em     = 1.0     # electromigration prefactor (=1 for V_T scaling)
    k0_hat = 0.3
    alpha  = 0.5
    stoi_vals = [-1.0, 0.0]

    print("\n" + "=" * 70)
    print("  MMS Convergence Study: Two Charged Species + Poisson + BV")
    print("=" * 70)
    print(f"  D0={D0_hat}, D1={D1_hat}, z0={z0}, z1={z1}")
    print(f"  c0_cat={c0_0}, c0_an={c0_1}, A0={A0}, A1={A1}")
    print(f"  beta0={beta0}, beta1={beta1}, eta0={eta0}, B={B_val}")
    print(f"  k0={k0_hat}, alpha={alpha}, eps_hat={eps_hat}")
    print(f"  Mesh sizes: {N_vals}")
    print("=" * 70 + "\n")

    results = {
        "N": [], "h": [],
        "c0_L2": [], "c0_H1": [],
        "c1_L2": [], "c1_H1": [],
        "phi_L2": [], "phi_H1": [],
    }

    for N in N_vals:
        t0 = time.time()
        h = 1.0 / N

        mesh = fd.RectangleMesh(N, N, 1.0, 1.0)
        x, y = fd.SpatialCoordinate(mesh)
        electrode_marker = 3
        bulk_marker = 4

        V = fd.FunctionSpace(mesh, "CG", 1)
        W = fd.MixedFunctionSpace([V, V, V])
        U = fd.Function(W, name="U")
        v0, v1, w = fd.TestFunctions(W)
        c0_h, c1_h, phi_h = fd.split(U)

        # --- Manufactured solutions ---
        c0_exact = c0_0 + A0 * fd.cos(pi * x) * (1.0 - fd.exp(-beta0 * y))
        c1_exact = c0_1 + A1 * fd.cos(pi * x) * (1.0 - fd.exp(-beta1 * y))
        phi_exact = eta0 * (1.0 - y) + B_val * fd.cos(pi * x) * y * (1.0 - y)

        # --- Volume sources ---
        J0_exact = D0_hat * (fd.grad(c0_exact) + em * z0 * c0_exact * fd.grad(phi_exact))
        J1_exact = D1_hat * (fd.grad(c1_exact) + em * z1 * c1_exact * fd.grad(phi_exact))
        S_c0 = -fd.div(J0_exact)
        S_c1 = -fd.div(J1_exact)

        # Poisson: -eps*laplacian(phi) = sum(z*c) + S_phi
        S_phi = -eps_hat * fd.div(fd.grad(phi_exact)) - (z0 * c0_exact + z1 * c1_exact)

        # --- Boundary correction ---
        n_vec = fd.FacetNormal(mesh)
        ds = fd.Measure("ds", domain=mesh)
        dx = fd.Measure("dx", domain=mesh)

        R_bv_exact = k0_hat * c0_0 * fd.exp(-alpha * eta0)

        g_0 = _boundary_correction_charged(
            D0_hat, em * z0, c0_exact, phi_exact, n_vec,
            stoi_list=[stoi_vals[0]], R_bv_exact_list=[R_bv_exact],
        )
        g_1 = _boundary_correction_charged(
            D1_hat, em * z1, c1_exact, phi_exact, n_vec,
            stoi_list=[stoi_vals[1]], R_bv_exact_list=[R_bv_exact],
        )

        # --- Weak form ---
        F_res = D0_hat * fd.dot(
            fd.grad(c0_h) + em * z0 * c0_h * fd.grad(phi_h), fd.grad(v0)
        ) * dx
        F_res += D1_hat * fd.dot(
            fd.grad(c1_h) + em * z1 * c1_h * fd.grad(phi_h), fd.grad(v1)
        ) * dx

        # BV (numerical)
        R_bv_num = k0_hat * c0_h * fd.exp(-alpha * eta0)
        F_res -= stoi_vals[0] * R_bv_num * v0 * ds(electrode_marker)
        F_res -= stoi_vals[1] * R_bv_num * v1 * ds(electrode_marker)

        # Poisson
        F_res += eps_hat * fd.dot(fd.grad(phi_h), fd.grad(w)) * dx
        F_res -= (z0 * c0_h + z1 * c1_h) * w * dx

        # MMS sources
        F_res -= S_c0 * v0 * dx
        F_res -= S_c1 * v1 * dx
        F_res -= S_phi * w * dx
        F_res -= g_0 * v0 * ds(electrode_marker)
        F_res -= g_1 * v1 * ds(electrode_marker)

        # --- BCs ---
        bcs = [
            fd.DirichletBC(W.sub(0), c0_exact, bulk_marker),
            fd.DirichletBC(W.sub(1), c1_exact, bulk_marker),
            fd.DirichletBC(W.sub(2), fd.Constant(eta0), electrode_marker),
            fd.DirichletBC(W.sub(2), fd.Constant(0.0), bulk_marker),
        ]

        # --- Initial guess ---
        U.sub(0).interpolate(c0_exact)
        U.sub(1).interpolate(c1_exact)
        U.sub(2).interpolate(phi_exact)

        # --- Solve ---
        J_form = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=SNES_OPTS)

        try:
            solver.solve()
        except fd.ConvergenceError as e:
            print(f"  [FAIL] N={N}: Newton failed: {e}")
            continue

        # --- Errors ---
        c0_ex_f = fd.Function(V); c0_ex_f.interpolate(c0_exact)
        c1_ex_f = fd.Function(V); c1_ex_f.interpolate(c1_exact)
        phi_ex_f = fd.Function(V); phi_ex_f.interpolate(phi_exact)

        c0_L2 = fd.errornorm(c0_ex_f, U.sub(0), norm_type="L2")
        c0_H1 = fd.errornorm(c0_ex_f, U.sub(0), norm_type="H1")
        c1_L2 = fd.errornorm(c1_ex_f, U.sub(1), norm_type="L2")
        c1_H1 = fd.errornorm(c1_ex_f, U.sub(1), norm_type="H1")
        phi_L2 = fd.errornorm(phi_ex_f, U.sub(2), norm_type="L2")
        phi_H1 = fd.errornorm(phi_ex_f, U.sub(2), norm_type="H1")

        elapsed = time.time() - t0

        results["N"].append(N)
        results["h"].append(h)
        results["c0_L2"].append(c0_L2)
        results["c0_H1"].append(c0_H1)
        results["c1_L2"].append(c1_L2)
        results["c1_H1"].append(c1_H1)
        results["phi_L2"].append(phi_L2)
        results["phi_H1"].append(phi_H1)

        if verbose:
            print(f"  N={N:4d}  h={h:.5f}  "
                  f"c0_L2={c0_L2:.4e}  c0_H1={c0_H1:.4e}  "
                  f"c1_L2={c1_L2:.4e}  c1_H1={c1_H1:.4e}  "
                  f"phi_L2={phi_L2:.4e}  phi_H1={phi_H1:.4e}  "
                  f"({elapsed:.1f}s)")

    return results


# ===========================================================================
# Convergence rate computation
# ===========================================================================

def compute_rates(h_list, err_list):
    """Compute convergence rates: rate_k = log(e_{k-1}/e_k) / log(h_{k-1}/h_k)."""
    rates = [None]
    for k in range(1, len(h_list)):
        if err_list[k] > 0 and err_list[k-1] > 0:
            rates.append(log(err_list[k-1] / err_list[k]) / log(h_list[k-1] / h_list[k]))
        else:
            rates.append(None)
    return rates


# ===========================================================================
# Pretty-print tables
# ===========================================================================

def format_table_single(results: dict) -> str:
    """Format convergence table for single-species case."""
    lines = []
    lines.append("")
    lines.append("=" * 95)
    lines.append("  Single Neutral Species MMS Convergence Table")
    lines.append("=" * 95)
    lines.append(f"{'N':>6s}  {'h':>9s}  "
                 f"{'c_L2':>10s}  {'rate':>6s}  {'c_H1':>10s}  {'rate':>6s}  "
                 f"{'phi_L2':>10s}  {'rate':>6s}  {'phi_H1':>10s}  {'rate':>6s}")
    lines.append("-" * 95)

    h_list = results["h"]
    c_L2_rates = compute_rates(h_list, results["c_L2"])
    c_H1_rates = compute_rates(h_list, results["c_H1"])
    phi_L2_rates = compute_rates(h_list, results["phi_L2"])
    phi_H1_rates = compute_rates(h_list, results["phi_H1"])

    for i in range(len(results["N"])):
        r_c_L2 = f"{c_L2_rates[i]:.2f}" if c_L2_rates[i] is not None else "---"
        r_c_H1 = f"{c_H1_rates[i]:.2f}" if c_H1_rates[i] is not None else "---"
        r_phi_L2 = f"{phi_L2_rates[i]:.2f}" if phi_L2_rates[i] is not None else "---"
        r_phi_H1 = f"{phi_H1_rates[i]:.2f}" if phi_H1_rates[i] is not None else "---"

        lines.append(
            f"{results['N'][i]:6d}  {results['h'][i]:9.5f}  "
            f"{results['c_L2'][i]:10.4e}  {r_c_L2:>6s}  "
            f"{results['c_H1'][i]:10.4e}  {r_c_H1:>6s}  "
            f"{results['phi_L2'][i]:10.4e}  {r_phi_L2:>6s}  "
            f"{results['phi_H1'][i]:10.4e}  {r_phi_H1:>6s}"
        )

    lines.append("-" * 95)
    lines.append("  Expected: L2 rate -> 2.00, H1 rate -> 1.00 (CG1)")
    lines.append("=" * 95)
    return "\n".join(lines)


def format_table_multi(results: dict, title: str) -> str:
    """Format convergence table for multi-species case."""
    lines = []
    lines.append("")
    lines.append("=" * 130)
    lines.append(f"  {title}")
    lines.append("=" * 130)

    h_list = results["h"]

    fields = []
    for key in sorted(results.keys()):
        if key in ("N", "h"):
            continue
        fields.append(key)

    header = f"{'N':>6s}  {'h':>9s}  "
    for f in fields:
        header += f"{f:>10s}  {'rate':>6s}  "
    lines.append(header)
    lines.append("-" * 130)

    all_rates = {}
    for f in fields:
        all_rates[f] = compute_rates(h_list, results[f])

    for i in range(len(results["N"])):
        row = f"{results['N'][i]:6d}  {results['h'][i]:9.5f}  "
        for f in fields:
            r = all_rates[f][i]
            r_str = f"{r:.2f}" if r is not None else "---"
            row += f"{results[f][i]:10.4e}  {r_str:>6s}  "
        lines.append(row)

    lines.append("-" * 130)
    lines.append("  Expected: L2 rate -> 2.00, H1 rate -> 1.00 (CG1)")
    lines.append("=" * 130)
    return "\n".join(lines)


# ===========================================================================
# Convergence plot
# ===========================================================================

def plot_convergence(all_results: dict, out_dir: str) -> str:
    """Create log-log convergence plot for all test cases."""
    n_cases = len(all_results)
    fig, axes = plt.subplots(1, n_cases, figsize=(7 * n_cases, 5), squeeze=False)

    for idx, (case_name, results) in enumerate(all_results.items()):
        ax = axes[0][idx]
        h = np.array(results["h"])

        L2_fields = [k for k in sorted(results.keys()) if k.endswith("_L2")]
        H1_fields = [k for k in sorted(results.keys()) if k.endswith("_H1")]

        colors_L2 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        colors_H1 = ["#17becf", "#bcbd22", "#9467bd", "#8c564b"]

        for i, f in enumerate(L2_fields):
            err = np.array(results[f])
            label = f.replace("_L2", "") + " $L^2$"
            ax.loglog(h, err, "o-", color=colors_L2[i % len(colors_L2)],
                      linewidth=1.5, markersize=5, label=label)

        for i, f in enumerate(H1_fields):
            err = np.array(results[f])
            label = f.replace("_H1", "") + " $H^1$"
            ax.loglog(h, err, "s--", color=colors_H1[i % len(colors_H1)],
                      linewidth=1.5, markersize=5, label=label)

        # Reference slopes
        h_ref = np.array([h[0], h[-1]])
        scale_L2 = results[L2_fields[0]][0] / h[0] ** 2
        scale_H1 = results[H1_fields[0]][0] / h[0] ** 1
        ax.loglog(h_ref, scale_L2 * h_ref ** 2, "k:", linewidth=0.8,
                  label="$O(h^2)$ ref")
        ax.loglog(h_ref, scale_H1 * h_ref ** 1, "k-.", linewidth=0.8,
                  label="$O(h^1)$ ref")

        ax.set_xlabel("$h$", fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.set_title(case_name, fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    png_path = os.path.join(out_dir, "mms_bv_convergence.png")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MMS convergence study for PNP-BV solver"
    )
    parser.add_argument(
        "--case", type=str, default="all",
        choices=["single", "two", "charged", "all"],
        help="Which test case to run (default: all)"
    )
    parser.add_argument(
        "--Nvals", type=int, nargs="+", default=[8, 16, 32, 64, 128],
        help="Mesh sizes (default: 8 16 32 64 128)"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="StudyResults/mms_bv_convergence",
        dest="out_dir",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    all_tables = []

    if args.case in ("single", "all"):
        r1 = run_mms_single_species(args.Nvals)
        all_results["1-species neutral + BV"] = r1
        table1 = format_table_single(r1)
        print(table1)
        all_tables.append(table1)

    if args.case in ("two", "all"):
        r2 = run_mms_two_species(args.Nvals)
        all_results["2-species neutral + 2 BV reactions"] = r2
        table2 = format_table_multi(r2, "Two Neutral Species MMS Convergence Table")
        print(table2)
        all_tables.append(table2)

    if args.case in ("charged", "all"):
        r3 = run_mms_charged(args.Nvals)
        all_results["2-species charged + Poisson + BV"] = r3
        table3 = format_table_multi(
            r3, "Two Charged Species + Poisson MMS Convergence Table"
        )
        print(table3)
        all_tables.append(table3)

    # Save summary
    if all_results:
        summary_path = os.path.join(args.out_dir, "mms_convergence_summary.txt")
        with open(summary_path, "w") as f:
            f.write("MMS Convergence Study for PNP-BV Solver\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for t in all_tables:
                f.write(t + "\n\n")
        print(f"\n[MMS] Summary saved -> {summary_path}")

        png_path = plot_convergence(all_results, args.out_dir)
        print(f"[MMS] Plot saved -> {png_path}")

    print("\n=== MMS Convergence Study Complete ===")


if __name__ == "__main__":
    main()
