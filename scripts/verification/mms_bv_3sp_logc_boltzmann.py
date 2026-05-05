"""Production-faithful MMS convergence test for the production PNP-BV solver.

Verifies the *exact* production stack used by ``scripts/plot_iv_curve_unified.py``:

  - 3 dynamic species (O2, H2O2, H+) + analytic Boltzmann ClO4-
  - log-concentration primary variables ``u_i = ln(c_i)``
  - log-rate Butler-Volmer evaluation (``bv_log_rate=True``)
  - Physical E_eq (R1 = 0.68 V, R2 = 1.78 V vs RHE)
  - Real D, k0, alpha, c0 from ``scripts/_bv_common.py`` (Mangan2025, pH 4)
  - Steric (Bikerman) with a_vals = [0.01]*3 (production default)
  - eta-clip (+/-50) and u-clamp (100, production-widened) left active --
    inactive *by manufactured-solution design* so MMS verifies the smooth
    operator without crossing a non-smooth threshold.

The factory call mirrors the production driver
``scripts/plot_iv_curve_unified.py:139`` exactly.

Manufactured solution (in nondim units)::

    c_i^ex(x, y) = c0_HAT[i] * (1 + delta_i * cos(pi x) * (1 - y)^2)
    phi^ex(x, y) = eta_hat * (1 - y) + B_phi * cos(pi x) * y * (1 - y)

These exactly satisfy production Dirichlet BCs::

    u_i(x, 1) = ln(c0_HAT[i])    <- production concentration BC
    phi(x, 1) = 0                 <- production ground BC
    phi(x, 0) = eta_hat           <- production electrode BC

so no BC overrides are needed.

V_RHE = +0.55 V chosen so:

  - eta_R1 = (V - 0.68) / V_T = -5.06   (clip inactive, |eta| << 50)
  - eta_R2 = (V - 1.78) / V_T = -47.87  (clip inactive, |eta| < 50)
  - Inside production warm-walk regime (V_RHE in [+0.50, +1.20] V).

Source generation (term-by-term continuum residual at u_exact):

  Standard MMS pattern: compute the *continuum* residual of each
  production weak-form term at U=u_exact, inject as forcing on F_res.
  At u_exact in continuum the residual vanishes; the discrete operator
  has O(h^p) error vs continuum, so the discrete solution u_h satisfies
  ||u_h - u_exact||_L2 = O(h^(k+1)) and ||u_h - u_exact||_H1 = O(h^k).

  Important: we deliberately do NOT use ``ufl.replace`` to mirror the
  discrete F_res at U_manuf -- that would make U_manuf an exact discrete
  solution by construction, hiding any wiring bug that's symmetric on
  both sides of the subtraction.  By writing the source against an
  independent continuum statement of the PDE, any inconsistency in
  F_res's discrete implementation shows up as a convergence-rate violation.

Mesh: ``UnitSquareMesh(N, N)``.  Departure from production's graded
rectangle (Nx=8, Ny=200, beta=3): graded refinement breaks h^p
convergence.  The operator under test is mesh-topology independent.

Expected: L2 rate ~ 2.0, H1 rate ~ 1.0 (CG1) for u_i and phi.
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

# Production code path: dispatcher -> forms_logc when formulation="logc".
from Forward.bv_solver import build_context, build_forms, make_graded_rectangle_mesh

from scripts._bv_common import (
    V_T,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
    A_DEFAULT,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
)


# ---------------------------------------------------------------------------
# Production constants (mirror ``plot_iv_curve_unified.py``)
# ---------------------------------------------------------------------------
E_EQ_R1 = 0.68    # V vs RHE
E_EQ_R2 = 1.78    # V vs RHE

V_RHE_TEST = 0.55                      # V vs RHE; production warm-walk regime
ETA_HAT_TEST = V_RHE_TEST / V_T        # nondim, ~21.41

# Manufactured-shape parameters.  delta_i < 1 keeps c_i^ex > 0; B_phi
# small keeps |phi^ex| inside Boltzmann's phi_clamp = 50.
DELTA_PERTURB = (0.30, 0.30, 0.30)     # one per species (O2, H2O2, H+)
B_PHI         = 0.5                    # nondim

# Quadrature degree for source-term integration.  CG1 default quadrature
# is too low for non-polynomial integrands like ln, exp, cos; bump up so
# the source contributes negligibly to the convergence rate.
SRC_QUAD_DEGREE = 8


# ---------------------------------------------------------------------------
# SolverParams factory call (mirrors ``plot_iv_curve_unified.py:139``)
# ---------------------------------------------------------------------------
def make_sp_production(eta_hat: float, *, counterion_entry=None):
    """Build SolverParams matching the production driver, with two changes:

    (1) ``eta_hat`` is fixed at the test voltage.
    (2) ``dt`` is large to neutralize the time term -- we initialise
        ``U_prev = U_manuf`` so ``(c - c_old)/dt`` is exactly zero.
    """
    # Tolerances are looser than the production driver (atol=1e-7) because
    # the manufactured-solution test problem has large absolute source
    # magnitude at V_RHE = +0.55 V -- the Boltzmann counterion contributes
    # ~1e8 to F_res near the electrode (Σz·c_bulk·exp(-z·η) at η ~ 21).
    # The FP-noise floor on source evaluation is therefore ~1e-8 absolute,
    # below which Newton spins.  Relative tolerance (rtol=1e-8) is what
    # actually drives Newton to the discretization-error floor; that's
    # what verifies the operator.  Initial residual norm reaches O(1e9)
    # at U_manuf, so rtol=1e-8 means Newton drives residual to O(1e1),
    # which is below the smallest discretization error of interest at
    # N=128 (O(h^2) ~ 6e-5).
    snes_opts = {
        **SNES_OPTS_CHARGED,
        "snes_max_it":               60,
        "snes_atol":                 1e-5,
        "snes_rtol":                 1e-8,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",   # production setting
        "snes_linesearch_maxlambda": 0.3,    # production setting
        "snes_divergence_tolerance": 1e10,
    }
    return make_bv_solver_params(
        eta_hat=eta_hat,
        dt=1e15, t_end=1e15,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc",
        log_rate=True,
        u_clamp=100.0,
        boltzmann_counterions=(
            [counterion_entry] if counterion_entry is not None
            else [DEFAULT_CLO4_BOLTZMANN_COUNTERION]
        ),
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
    )


def _extract_solver_parameters(sp):
    """Pull the SNES/KSP/PC/MUMPS options from the SolverParams 11-tuple."""
    params = sp[10]
    keep_prefixes = ("snes_", "ksp_", "pc_", "mat_")
    return {k: v for k, v in params.items() if k.startswith(keep_prefixes)}


# ---------------------------------------------------------------------------
# Manufactured-source construction (term-by-term continuum residual)
# ---------------------------------------------------------------------------
def _build_manufactured_source(
    ctx,
    c_exact,
    u_exact,
    phi_exact,
    params_for_logging=None,
    *,
    clip_source: bool = False,
    bikerman_counterion=None,
):
    """Inject the continuum residual of every production weak-form term at
    U=u_exact into ``ctx['F_res']`` as a forcing.

    The PDE we mirror in continuum (matching forms_logc.py + boltzmann.py):

      Per species i:
        d c_i / dt + nabla . J_i  =  0  (interior)
        J_i . n  =  sum_j s_ij R_j      (electrode)
        J_i  =  -D_i * c_i * (grad u_i + em*z_i*grad phi + grad mu_steric)
        mu_steric  =  ln(1 - sum_i a_i * c_i)
        R_j = exp(log_cathodic_j) - exp(log_anodic_j)  (log-rate form)

      Poisson:
        -eps * laplacian(phi)  -  charge_rhs * sum_i z_i * c_i
                                -  z_scale * charge_rhs * sum_k z_k * c_bulk_k * exp(-z_k * phi)
        =  0

    The corresponding discrete weak form (production):
        sum_i [ ((c_i - c_old_i)/dt) v_i dx + J_i . grad(v_i) dx
                - sum_j s_ij R_j v_i ds(electrode) ]
        + eps * grad(phi) . grad(w) dx
        - charge_rhs * sum_i z_i c_i w dx
        - sum_k z_scale * charge_rhs * z_k * c_bulk_k * exp(-z_k phi_clamped) w dx
        =  0

    Adding the continuum source (subtracting from F_res) makes u_exact a
    continuum solution; the discrete solver finds u_h that approximates
    u_exact at the predicted h^(k+1) / h^k rate.
    """
    n = ctx["n_species"]
    mesh = ctx["mesh"]
    W = ctx["W"]

    # --- Pull production scaling (so D, k0, alpha, em, eps, charge_rhs all
    # match the dispatched production residual) ---
    scaling = ctx["nondim"]
    em = float(scaling["electromigration_prefactor"])
    eps = float(scaling["poisson_coefficient"])
    charge_rhs = float(scaling["charge_rhs_prefactor"])
    D_model = [float(scaling["D_model_vals"][i]) for i in range(n)]
    z_vals_float = [float(z) for z in THREE_SPECIES_LOGC_BOLTZMANN.z_vals]
    phi_app_model = float(scaling["phi_applied_model"])  # nondim eta_hat
    bv_exp_scale = float(scaling["bv_exponent_scale"])

    # Steric (Bikerman) coefficients.  Production a_vals_hat is
    # [A_DEFAULT]*3 = [0.01]*3.  packing_floor only fires near
    # close-packing -- our manufactured c_i is well below that.
    a_vals_float = [float(A_DEFAULT)] * n

    # Steric-aware Boltzmann counterion (bikerman closure) — when
    # supplied, the manufactured Poisson source uses ``c_steric_manuf``
    # in place of ideal ``c_b * exp(-z*phi)``, AND the dynamic species'
    # packing fraction includes the counterion's contribution
    # (mirroring forms_logc.py's wiring).
    if bikerman_counterion is not None:
        z_b_steric = float(bikerman_counterion["z"])
        c_b_steric = float(bikerman_counterion["c_bulk_nondim"])
        a_b_steric = float(bikerman_counterion["a_nondim"])
        # Bulk packing fraction (the helper in boltzmann.py validates this).
        c0_dyn_bulk = list(scaling.get("c0_model_vals", []))[:n]
        A_dyn_bulk = sum(a_vals_float[i] * float(c0_dyn_bulk[i])
                         for i in range(min(n, len(c0_dyn_bulk))))
        theta_b_steric = 1.0 - A_dyn_bulk - a_b_steric * c_b_steric
        # Closure expression at the manufactured fields.  The phi clamp
        # is irrelevant on our smooth manufactured solution (well below
        # the ±50 limit), so we omit the clamp here.
        q_manuf = fd.exp(-fd.Constant(z_b_steric) * phi_exact)
        A_dyn_manuf = sum(
            fd.Constant(a_vals_float[i]) * c_exact[i] for i in range(n)
        )
        c_steric_manuf = (
            fd.Constant(c_b_steric) * q_manuf
            * (fd.Constant(1.0) - A_dyn_manuf)
            / (fd.Constant(theta_b_steric)
               + fd.Constant(a_b_steric * c_b_steric) * q_manuf)
        )
        packing_manuf = (
            fd.Constant(1.0) - A_dyn_manuf
            - fd.Constant(a_b_steric) * c_steric_manuf
        )
    else:
        c_steric_manuf = None
        packing_manuf = fd.Constant(1.0) - sum(
            fd.Constant(a_vals_float[i]) * c_exact[i] for i in range(n)
        )
    mu_steric_manuf = -fd.ln(packing_manuf)

    # --- Manufactured fluxes J_i = D_i * c_i * (grad u + em z grad phi + grad mu_steric) ---
    # NOTE: forms_logc.py line 287-291 uses the log-c-friendly form
    #   Jflux = D[i] * c_i * (grad(u_i) + grad(em*z_i*phi) + grad(mu_steric))
    # mathematically equivalent to D*(grad(c) + em*z*c*grad(phi) + c*grad(mu_steric))
    # since c*grad(u) = grad(c) and c*grad(em*z*phi) = em*z*c*grad(phi).
    # Wrap em*z_i as fd.Constant so z_i = 0 species keep mesh-aware UFL terms
    # (a Python float 0.0 multiplied into a UFL expression simplifies to scalar 0
    # and fd.grad(0) fails to find the geometric dimension).
    em_z_const = [fd.Constant(em * z_vals_float[i]) for i in range(n)]
    J_manuf = [
        D_model[i] * c_exact[i] * (
            fd.grad(u_exact[i])
            + em_z_const[i] * fd.grad(phi_exact)
            + fd.grad(mu_steric_manuf)
        )
        for i in range(n)
    ]

    # --- Manufactured BV rates (log-rate form, mirror of forms_logc.py:334-366) ---
    rxns = scaling["bv_reactions"]
    R_manuf = []
    for j_idx, rxn in enumerate(rxns):
        k0 = float(rxn["k0_model"])
        alpha = float(rxn["alpha"])
        n_e = float(rxn["n_electrons"])
        cat_idx = rxn["cathodic_species"]
        E_eq_j_model = float(rxn.get("E_eq_model", 0.0))
        # eta is *spatially constant* on the electrode (depends only on
        # phi_applied_func, not phi).  forms_logc.py line 235-241 uses
        # use_eta_in_bv=True path: eta_raw = phi_applied_func - E_eq.
        eta_j = bv_exp_scale * (phi_app_model - E_eq_j_model)
        if clip_source:
            # Mirror the production eta-clip so the source assumes the same
            # numeric eta as the discrete operator -- isolates discrete
            # truncation error from clip-induced model error.
            eta_j = max(min(eta_j, 50.0), -50.0)

        log_cath = log(k0) + u_exact[cat_idx] - alpha * n_e * eta_j
        for f in rxn.get("cathodic_conc_factors", []):
            sp_idx = f["species"]
            power = float(f["power"])
            c_ref_f = max(float(f["c_ref_nondim"]), 1e-12)
            log_cath = log_cath + power * (u_exact[sp_idx] - log(c_ref_f))
        cathodic = fd.exp(log_cath)

        if rxn["reversible"] and rxn["anodic_species"] is not None:
            anod_idx = rxn["anodic_species"]
            log_anod = log(k0) + u_exact[anod_idx] + (1.0 - alpha) * n_e * eta_j
            anodic = fd.exp(log_anod)
        elif rxn["reversible"] and float(rxn.get("c_ref_model", 0.0)) > 1e-30:
            c_ref_j = float(rxn["c_ref_model"])
            log_anod = log(k0) + log(c_ref_j) + (1.0 - alpha) * n_e * eta_j
            anodic = fd.exp(log_anod)
        else:
            anodic = fd.Constant(0.0)

        R_manuf.append(cathodic - anodic)

    # --- Test functions and measures (high quadrature for non-polynomial integrands) ---
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w_test = v_tests[-1]
    n_vec = fd.FacetNormal(mesh)
    dx_q = fd.dx(domain=mesh, degree=SRC_QUAD_DEGREE)
    ds_q = fd.ds(domain=mesh, degree=SRC_QUAD_DEGREE)

    bv_settings = ctx["bv_settings"]
    electrode_marker = bv_settings["electrode_marker"]

    # --- Inject sources into F_res ---
    # Pattern: production residual term ``+ ∫ X · v · dx`` requires source
    # ``- ∫ X_continuum_at_u_exact · v · dx`` to make u_exact a continuum
    # solution.  We use the divergence theorem to write everything in
    # divergence form on dx + boundary on ds.
    F_res = ctx["F_res"]

    for i in range(n):
        # NP interior source.  Production: +∫ J_i · ∇v_i · dx
        # IBP:  +∫ J_i · ∇v_i dx  =  -∫ div(J_i) v_i dx + ∫ J_i·n v_i ds_boundary
        # Continuum residual at u_exact: -div(J_manuf_i) on interior,
        # J_manuf_i·n - sum_j s_ij R_manuf_j on electrode, J_manuf_i·n on bulk side.
        # The bulk side has Dirichlet BC so v_i = 0 there; only electrode + bulk
        # natural-BC sides contribute.  All sides except electrode have zero
        # natural BC in production (no flux on x-walls; concentration BC on bulk).
        # Inject sources for interior and electrode only.
        S_c_i = -fd.div(J_manuf[i])
        F_res -= S_c_i * v_list[i] * dx_q

        # Electrode boundary correction:
        #   ∫ J_i·n v_i ds_elec - ∫ Σ s_ij R_j v_i ds_elec  ===> needs
        # source g_i = J_manuf_i·n - Σ s_ij R_manuf_j on electrode.
        flux_outward = fd.dot(J_manuf[i], n_vec)
        bv_sum = sum(
            float(rxns[j_idx]["stoichiometry"][i]) * R_manuf[j_idx]
            for j_idx in range(len(rxns))
            if rxns[j_idx]["stoichiometry"][i] != 0
        )
        # If no reaction touches species i, bv_sum is 0; flux_outward source still applies.
        if isinstance(bv_sum, int) and bv_sum == 0:
            bv_sum = fd.Constant(0.0)
        g_i = flux_outward - bv_sum
        F_res -= g_i * v_list[i] * ds_q(electrode_marker)

    # --- Poisson source ---
    # Production residual (signs match forms_logc.py + boltzmann.py):
    #   F_res += eps * grad(phi) . grad(w) * dx
    #   F_res -= charge_rhs * sum_i z_i * c_i * w * dx
    #   F_res -= z_scale * charge_rhs * z_C * c_bulk * exp(-z_C * phi_clamped) * w * dx
    #
    # IBP first: eps * grad(phi) . grad(w) dx = -eps * lap(phi) * w dx + boundary.
    # On Dirichlet boundaries (electrode + bulk for phi) w=0; on x-walls natural
    # BC has grad(phi).n = 0 for our manufactured phi (cos(pi x) vanishes derivatively).
    # So F_res_phi at u_manuf integrates to:
    #   ∫ [ -eps*lap(phi_manuf) - charge_rhs*Σ z_i c_i_manuf
    #       - charge_rhs * Σ_k z_k c_bulk_k * exp(-z_k phi_manuf) ] * w * dx
    # which we want to make zero by subtracting that integrand as a source.
    S_phi = (
        -eps * fd.div(fd.grad(phi_exact))
        - charge_rhs * sum(z_vals_float[i] * c_exact[i] for i in range(n))
    )
    if bikerman_counterion is not None:
        # Bikerman closure replaces the ideal Boltzmann source.
        S_phi = S_phi - charge_rhs * fd.Constant(z_b_steric) * c_steric_manuf
    else:
        # Ideal Boltzmann counterions (z_scale defaults to 1.0; clamp inactive at our phi).
        for entry in [DEFAULT_CLO4_BOLTZMANN_COUNTERION]:
            z_c = float(entry["z"])
            c_bulk_c = float(entry["c_bulk_nondim"])
            S_phi = S_phi - charge_rhs * z_c * c_bulk_c * fd.exp(-z_c * phi_exact)
    F_res -= S_phi * w_test * dx_q

    ctx["F_res"] = F_res
    return ctx


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
SPECIES_NAMES = ["O2", "H2O2", "H+"]


def compute_rates(h_list, err_list):
    rates = [None]
    for k in range(1, len(h_list)):
        if err_list[k] > 0 and err_list[k - 1] > 0:
            rates.append(log(err_list[k - 1] / err_list[k]) / log(h_list[k - 1] / h_list[k]))
        else:
            rates.append(None)
    return rates


# ---------------------------------------------------------------------------
# MMS convergence study
# ---------------------------------------------------------------------------
def _ufl_l2_error(u_ufl, u_h, mesh, degree=SRC_QUAD_DEGREE):
    """L2 error of u_h vs UFL expression u_ufl, using high-degree quadrature."""
    dx_q = fd.dx(domain=mesh, degree=degree)
    return float(fd.sqrt(fd.assemble((u_ufl - u_h) ** 2 * dx_q)))


def _ufl_h1_error(u_ufl, u_h, mesh, degree=SRC_QUAD_DEGREE):
    """H1 error of u_h vs UFL expression u_ufl, using high-degree quadrature."""
    dx_q = fd.dx(domain=mesh, degree=degree)
    diff = u_ufl - u_h
    grad_diff = fd.grad(u_ufl) - fd.grad(u_h)
    integrand = diff ** 2 + fd.inner(grad_diff, grad_diff)
    return float(fd.sqrt(fd.assemble(integrand * dx_q)))


def _solve_mms_on_mesh(
    mesh,
    sp,
    snes_params,
    *,
    eta_hat: float = ETA_HAT_TEST,
    clip_source: bool = False,
    bikerman_counterion=None,
) -> dict:
    """Set up the MMS problem on a given mesh, solve, and return error dict.

    Returns a dict with per-field L2/H1 errors against the continuum
    manufactured solution (UFL-based, high quadrature)::

        {"u0_L2", "u0_H1", "u1_L2", ..., "phi_L2", "phi_H1",
         "c0_L2", "c1_L2", "c2_L2",
         "newton_converged": bool, "newton_iterations": int}

    Caller is responsible for printing diagnostics; this helper only
    raises on Newton divergence (sets newton_converged=False and returns
    NaN errors so the caller can still report).
    """
    n = THREE_SPECIES_LOGC_BOLTZMANN.n_species
    c0_HAT = list(THREE_SPECIES_LOGC_BOLTZMANN.c0_vals_hat)

    x, y = fd.SpatialCoordinate(mesh)

    # ---- Build the production residual via the dispatcher ----
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)

    # ---- Manufactured solution as UFL expressions in (x, y) ----
    c_exact = [
        c0_HAT[i] * (
            fd.Constant(1.0)
            + fd.Constant(DELTA_PERTURB[i]) * fd.cos(pi * x) * (1.0 - y) ** 2
        )
        for i in range(n)
    ]
    u_exact = [fd.ln(c_exact[i]) for i in range(n)]
    phi_exact = (
        fd.Constant(eta_hat) * (1.0 - y)
        + fd.Constant(B_PHI) * fd.cos(pi * x) * y * (1.0 - y)
    )

    # ---- Inject continuum source into F_res (term-by-term mirror) ----
    ctx = _build_manufactured_source(
        ctx, c_exact, u_exact, phi_exact,
        clip_source=clip_source,
        bikerman_counterion=bikerman_counterion,
    )

    W = ctx["W"]; U = ctx["U"]; U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]; bcs = ctx["bcs"]

    # ---- Build U_manuf for initial guess ----
    U_manuf = fd.Function(W)
    for i in range(n):
        U_manuf.sub(i).interpolate(u_exact[i])
    U_manuf.sub(n).interpolate(phi_exact)

    # Production BCs already match U_manuf by manufactured-shape design.
    U.assign(U_manuf)
    U_prev.assign(U_manuf)

    # ---- Solve ----
    J_form  = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
    solver  = fd.NonlinearVariationalSolver(
        problem, solver_parameters=snes_params,
    )
    out: dict = {"newton_converged": False, "newton_iterations": -1}
    try:
        solver.solve()
        out["newton_converged"] = True
        out["newton_iterations"] = int(solver.snes.getIterationNumber())
    except fd.ConvergenceError as exc:
        out["newton_error"] = str(exc)
        for i in range(n):
            out[f"u{i}_L2"] = float("nan")
            out[f"u{i}_H1"] = float("nan")
            out[f"c{i}_L2"] = float("nan")
        out["phi_L2"] = float("nan")
        out["phi_H1"] = float("nan")
        return out

    # ---- Errors (UFL-based, high quadrature against continuum manuf) ----
    for i in range(n):
        out[f"u{i}_L2"] = _ufl_l2_error(u_exact[i], U.sub(i), mesh)
        out[f"u{i}_H1"] = _ufl_h1_error(u_exact[i], U.sub(i), mesh)
        out[f"c{i}_L2"] = _ufl_l2_error(c_exact[i], fd.exp(U.sub(i)), mesh)
    out["phi_L2"] = _ufl_l2_error(phi_exact, U.sub(n), mesh)
    out["phi_H1"] = _ufl_h1_error(phi_exact, U.sub(n), mesh)
    return out


def _print_config_banner(
    *,
    mesh_sizes_str: str,
    v_rhe: float = V_RHE_TEST,
    eta_hat: float = ETA_HAT_TEST,
    clip_source: bool = False,
) -> None:
    n = THREE_SPECIES_LOGC_BOLTZMANN.n_species
    c0_HAT = list(THREE_SPECIES_LOGC_BOLTZMANN.c0_vals_hat)
    eta_R1 = (v_rhe - E_EQ_R1) / V_T
    eta_R2 = (v_rhe - E_EQ_R2) / V_T
    R1_status = "CLIPPED" if abs(eta_R1) > 50.0 else "unclipped"
    R2_status = "CLIPPED" if abs(eta_R2) > 50.0 else "unclipped"
    print("=" * 80)
    print("  Production-Faithful MMS: 3sp + Boltzmann + log-c + log-rate BV")
    print("=" * 80)
    print(f"  V_RHE          = {v_rhe} V vs RHE")
    print(f"  eta_hat        = {eta_hat:.4f} (nondim)")
    print(f"  E_eq R1/R2     = {E_EQ_R1}/{E_EQ_R2} V")
    print(f"  c0_HAT         = {c0_HAT}")
    print(f"  delta_i        = {DELTA_PERTURB}")
    print(f"  B_phi          = {B_PHI}")
    print(f"  log-rate BV    = ON (bv_log_rate=True)")
    cl04 = DEFAULT_CLO4_BOLTZMANN_COUNTERION
    print(f"  Boltzmann ClO4-= z={cl04['z']}, c_bulk_hat={cl04['c_bulk_nondim']}")
    print(f"  steric a_vals  = {[A_DEFAULT]*n}")
    print(f"  u-clamp        = 100 (active wiring)")
    print(
        f"  eta-clip       = +/-50  "
        f"R1: eta={eta_R1:+.2f} ({R1_status}), "
        f"R2: eta={eta_R2:+.2f} ({R2_status})"
    )
    print(f"  clip_source    = {clip_source}")
    print(f"  Source quadrature degree = {SRC_QUAD_DEGREE}")
    print(f"  Mesh           = {mesh_sizes_str}")
    print("=" * 80)


def run_mms(
    N_vals: list[int],
    verbose: bool = True,
    *,
    eta_hat: float = ETA_HAT_TEST,
    v_rhe: float = V_RHE_TEST,
    clip_source: bool = False,
    bikerman_counterion=None,
) -> dict:
    """Run the production-faithful MMS convergence study on a chain of
    UnitSquareMesh(N, N) meshes for h^p rate verification.

    Parameters
    ----------
    eta_hat, v_rhe : float
        Voltage to test (nondim eta and physical V_RHE; should satisfy
        ``eta_hat = v_rhe / V_T``). Defaults preserve the legacy
        V_RHE = +0.55 V test point.
    clip_source : bool
        If True, the manufactured BV source applies the same eta-clip
        (+/-50) as the discrete operator. Useful for isolating discrete
        truncation error from clip-induced model error when ``v_rhe``
        falls in the production clipped regime (R2 below +0.495 V).

    For single-mesh recovery on the production graded mesh, see
    :func:`verify_on_graded_production_mesh`.
    """
    n = THREE_SPECIES_LOGC_BOLTZMANN.n_species

    if verbose:
        _print_config_banner(
            mesh_sizes_str=f"UnitSquareMesh sweep N={N_vals}",
            v_rhe=v_rhe,
            eta_hat=eta_hat,
            clip_source=clip_source,
        )

    sp = make_sp_production(eta_hat, counterion_entry=bikerman_counterion)
    snes_params = _extract_solver_parameters(sp)

    results: dict = {"N": [], "h": []}
    for i in range(n):
        results[f"u{i}_L2"] = []
        results[f"u{i}_H1"] = []
        results[f"c{i}_L2"] = []
    results["phi_L2"] = []
    results["phi_H1"] = []

    for N in N_vals:
        t0 = time.time()
        h = 1.0 / N
        # UnitSquareMesh markers: 3=bottom (electrode), 4=top (bulk).
        mesh = fd.UnitSquareMesh(N, N)
        errs = _solve_mms_on_mesh(
            mesh, sp, snes_params,
            eta_hat=eta_hat, clip_source=clip_source,
            bikerman_counterion=bikerman_counterion,
        )
        if not errs.get("newton_converged", False):
            print(f"  [FAIL] N={N}: Newton failed: {errs.get('newton_error', '?')}")
            continue

        results["N"].append(N)
        results["h"].append(h)
        for i in range(n):
            results[f"u{i}_L2"].append(errs[f"u{i}_L2"])
            results[f"u{i}_H1"].append(errs[f"u{i}_H1"])
            results[f"c{i}_L2"].append(errs[f"c{i}_L2"])
        results["phi_L2"].append(errs["phi_L2"])
        results["phi_H1"].append(errs["phi_H1"])

        if verbose:
            elapsed = time.time() - t0
            parts = [f"N={N:4d}  h={h:.5f}"]
            for i in range(n):
                parts.append(f"u{i}_L2={errs[f'u{i}_L2']:.3e}")
            parts.append(f"phi_L2={errs['phi_L2']:.3e}")
            parts.append(f"({elapsed:.1f}s)")
            print("  " + "  ".join(parts))

    return results


def verify_on_graded_production_mesh(verbose: bool = True) -> dict:
    """Single-mesh MMS recovery test on the production graded rectangle.

    Mirrors ``scripts/plot_iv_curve_unified.py:118``: ``Nx=8``, ``Ny=200``,
    ``beta=3.0`` -- exactly the mesh production solves use.  Sanity check
    that the solver recovers the manufactured solution to within the
    expected discretization error of the *production* mesh (not the
    asymptotic regime).

    The dominant error here is ``h_x = 1/8 = 0.125``: our manufactured
    solution has cos(pi x) variation in x, and Nx=8 is coarse for it.
    Expected L2 errors for u_i and phi: O(h_x^2) ~ O(1.6e-2).  Newton
    should still converge cleanly from the U_manuf initial guess.

    Returns the error dict from :func:`_solve_mms_on_mesh` plus a
    ``mesh_label`` field.
    """
    Nx, Ny, beta = 8, 200, 3.0
    if verbose:
        _print_config_banner(
            mesh_sizes_str=f"graded rectangle Nx={Nx}, Ny={Ny}, beta={beta} (production)"
        )

    sp = make_sp_production(ETA_HAT_TEST)
    snes_params = _extract_solver_parameters(sp)

    t0 = time.time()
    mesh = make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta)
    errs = _solve_mms_on_mesh(mesh, sp, snes_params)
    elapsed = time.time() - t0
    errs["mesh_label"] = f"graded Nx={Nx}, Ny={Ny}, beta={beta}"
    errs["elapsed_seconds"] = float(elapsed)

    if verbose:
        if errs.get("newton_converged", False):
            print(
                f"  [graded {Nx}x{Ny}, beta={beta}]  "
                f"Newton iterations: {errs['newton_iterations']}  "
                f"({elapsed:.1f}s)"
            )
            n = THREE_SPECIES_LOGC_BOLTZMANN.n_species
            for i in range(n):
                print(
                    f"    {SPECIES_NAMES[i]:>5s} u{i}: "
                    f"L2={errs[f'u{i}_L2']:.3e}  H1={errs[f'u{i}_H1']:.3e}"
                )
            print(
                f"    {'phi':>5s}   : "
                f"L2={errs['phi_L2']:.3e}  H1={errs['phi_H1']:.3e}"
            )
        else:
            print(f"  [graded {Nx}x{Ny}] Newton FAILED: {errs.get('newton_error', '?')}")

    return errs


# ---------------------------------------------------------------------------
# Reporting (pretty-print + plot)
# ---------------------------------------------------------------------------
def format_summary(results: dict) -> str:
    lines = [""]
    lines.append("=" * 80)
    lines.append("  Production-Faithful MMS: Convergence Rate Summary")
    lines.append("=" * 80)

    h_list = results["h"]
    n = THREE_SPECIES_LOGC_BOLTZMANN.n_species
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
        rates = compute_rates(h_list, results[f"c{i}_L2"])
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
    n = THREE_SPECIES_LOGC_BOLTZMANN.n_species

    fields: list[str] = []
    for i in range(n):
        fields.append(f"u{i}_L2")
        fields.append(f"u{i}_H1")
    fields.append("phi_L2")
    fields.append("phi_H1")

    header = f"  {'N':>4} {'h':>8}  "
    for fn in fields:
        header += f"{fn:>10}  {'rate':>5}  "
    lines.append(header)
    lines.append("-" * 100)

    rates = {fn: compute_rates(h_list, results[fn]) for fn in fields}
    for k in range(len(results["N"])):
        row = f"  {results['N'][k]:>4} {results['h'][k]:>8.4f}  "
        for fn in fields:
            r = rates[fn][k]
            r_str = f"{r:.2f}" if r is not None else "---"
            row += f"{results[fn][k]:>10.3e}  {r_str:>5}  "
        lines.append(row)
    lines.append("=" * 100)
    return "\n".join(lines)


def plot_convergence(results: dict, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h = np.array(results["h"])
    n = THREE_SPECIES_LOGC_BOLTZMANN.n_species
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    ax = axes[0]
    for i in range(n):
        ax.loglog(h, results[f"u{i}_L2"], "o-", color=colors[i], linewidth=1.5,
                  markersize=5, label=f"{SPECIES_NAMES[i]} $u_{i}$ $L^2$")
    ax.loglog(h, results["phi_L2"], "s-", color=colors[3], linewidth=1.5,
              markersize=5, label=r"$\phi$ $L^2$")
    h_ref = np.array([h[0], h[-1]])
    scale = results["u0_L2"][0] / h[0] ** 2
    ax.loglog(h_ref, scale * h_ref ** 2, "k:", linewidth=0.8, label=r"$O(h^2)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("$L^2$ error")
    ax.set_title("$L^2$ Convergence (production-faithful)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    for i in range(n):
        ax.loglog(h, results[f"u{i}_H1"], "o-", color=colors[i], linewidth=1.5,
                  markersize=5, label=f"{SPECIES_NAMES[i]} $u_{i}$ $H^1$")
    ax.loglog(h, results["phi_H1"], "s-", color=colors[3], linewidth=1.5,
              markersize=5, label=r"$\phi$ $H^1$")
    scale = results["u0_H1"][0] / h[0] ** 1
    ax.loglog(h_ref, scale * h_ref ** 1, "k-.", linewidth=0.8, label=r"$O(h^1)$")
    ax.set_xlabel("$h$"); ax.set_ylabel("$H^1$ error")
    ax.set_title("$H^1$ Convergence (production-faithful)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"MMS: 3sp + Boltzmann + log-c + log-rate BV   "
        f"(V_RHE = {V_RHE_TEST} V, production stack)",
        fontsize=11,
    )
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
        f.write("Production-faithful MMS study\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(format_table(results) + "\n\n")
        f.write(format_summary(results) + "\n")
    print(f"\n[MMS] Summary saved -> {summary_path}")

    png = plot_convergence(results, args.out_dir)
    print(f"[MMS] Plot saved -> {png}")


if __name__ == "__main__":
    main()
