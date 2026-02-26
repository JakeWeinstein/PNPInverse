"""Butler-Volmer BC PNP forward solver with full nondimensionalization.

Supports two BV configuration modes:

**Per-species mode** (legacy, backward-compatible)::

    "bv_bc": {
        "k0":               [2.4e-8, 2.4e-8],   # one per species
        "alpha":            [0.627, 0.373],
        "stoichiometry":    [-1, +1],
        "c_ref":            [0.5, 0.5],
        "E_eq_v":           0.0,
        "electrode_marker":      1,
        "concentration_marker":  3,
        "ground_marker":         3,
    }

**Multi-reaction mode** (new — for coupled reactions)::

    "bv_bc": {
        "reactions": [
            {
                "k0": 2.4e-8,
                "alpha": 0.627,
                "cathodic_species": 0,    # species consumed
                "anodic_species": 1,      # species produced (None if irreversible)
                "c_ref": 1.0,            # reference conc for anodic term
                "stoichiometry": [-1, +1],
                "n_electrons": 2,
                "reversible": True,
            },
            ...
        ],
        "electrode_marker": 1,
        "concentration_marker": 3,
        "ground_marker": 3,
    }

Multi-reaction mode enables coupled chemistry — e.g. O₂→H₂O₂ (R₁)
and H₂O₂→H₂O (R₂) share the H₂O₂ species.  Each reaction j contributes::

    R_j = k0_j · [c_cat · exp(−α_j · η̂)  −  c_ref_j · exp((1−α_j) · η̂)]

to the weak form via the stoichiometry matrix: for species i,
``F_res -= s_ij · R_j · v_i · ds(electrode)``.

The solved ``ctx["bv_rate_exprs"]`` list stores UFL expressions for each R_j,
allowing post-solve current-density computation.

Convergence strategies
----------------------
``clip_exponent``, ``regularize_concentration``, ``use_eta_in_bv`` —
see ``bv_convergence`` config block.

Public API
----------
build_context(solver_params) → dict
build_forms(ctx, solver_params) → dict
set_initial_conditions(ctx, solver_params, blob=False) → None
forsolve_bv(ctx, solver_params, print_interval=100) → Function
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import firedrake as fd

from Nondim.transform import build_model_scaling, _get_nondim_cfg, _as_list, _bool, _pos
from Nondim.constants import FARADAY_CONSTANT, GAS_CONSTANT


# ---------------------------------------------------------------------------
# Graded mesh utilities
# ---------------------------------------------------------------------------

def make_graded_interval_mesh(N: int = 300, beta: float = 2.0) -> fd.Mesh:
    """Create a 1D interval mesh on [0, 1] with power-law grading.

    Points are placed at ``x_i = (i/N)^beta`` for ``i = 0, ..., N``.
    ``beta > 1`` clusters nodes near ``x = 0`` (electrode).

    Boundary markers:
        1 = left  (x=0, electrode)
        2 = right (x=1, bulk)

    Parameters
    ----------
    N : int
        Number of cells (elements).
    beta : float
        Grading exponent.  beta=1 gives uniform spacing; beta=2 gives
        quadratic clustering near x=0.
    """
    mesh = fd.IntervalMesh(N, 1.0)
    coords = mesh.coordinates.dat.data
    # Apply power-law stretching: x -> x^beta
    coords[:] = coords[:] ** beta
    return mesh


def make_graded_rectangle_mesh(
    Nx: int = 8,
    Ny: int = 300,
    beta: float = 2.0,
) -> fd.Mesh:
    """Create a 2D rectangle mesh on [0, 1]^2 with power-law grading in y.

    The y-coordinate is stretched: ``y -> y^beta``.  ``beta > 1`` clusters
    nodes near ``y = 0`` (electrode / bottom).  The x-direction is uniform.

    Boundary markers (firedrake ``RectangleMesh`` convention):
        1 = left   (x=0, zero-flux)
        2 = right  (x=1, zero-flux)
        3 = bottom (y=0, electrode)
        4 = top    (y=1, bulk)

    Parameters
    ----------
    Nx : int
        Number of cells in x (tangential, uniform).
    Ny : int
        Number of cells in y (normal to electrode, graded).
    beta : float
        Grading exponent for the y-direction.
    """
    mesh = fd.RectangleMesh(Nx, Ny, 1.0, 1.0)
    coords = mesh.coordinates.dat.data
    coords[:, 1] = coords[:, 1] ** beta
    return mesh


# ---------------------------------------------------------------------------
# BV config helpers
# ---------------------------------------------------------------------------

def _get_bv_cfg(params: Any, n_species: int) -> dict:
    """Parse and validate the ``bv_bc`` block from solver_params[10]."""
    if not isinstance(params, dict):
        raise ValueError("solver_params[10] must be a dict containing 'bv_bc'.")
    raw = params.get("bv_bc", {})
    if not isinstance(raw, dict):
        raise ValueError("solver_params[10]['bv_bc'] must be a dict.")

    k0 = raw.get("k0", 1e-5)
    alpha = raw.get("alpha", 0.5)
    stoichiometry = raw.get("stoichiometry", [-1] * n_species)
    c_ref = raw.get("c_ref", raw.get("c_inf", 1.0))
    E_eq_v = float(raw.get("E_eq_v", 0.0))
    electrode_marker = int(raw.get("electrode_marker", 1))
    concentration_marker = int(raw.get("concentration_marker", 3))
    ground_marker = int(raw.get("ground_marker", 3))

    return {
        "k0_vals":            _as_list(k0, n_species, "bv_bc.k0"),
        "alpha_vals":         _as_list(alpha, n_species, "bv_bc.alpha"),
        "stoichiometry":      [int(s) for s in _as_list(stoichiometry, n_species, "bv_bc.stoichiometry")],
        "c_ref_vals":         _as_list(c_ref, n_species, "bv_bc.c_ref"),
        "E_eq_v":             E_eq_v,
        "electrode_marker":   electrode_marker,
        "concentration_marker": concentration_marker,
        "ground_marker":      ground_marker,
    }


def _get_bv_convergence_cfg(params: Any) -> dict:
    """Parse optional BV convergence-strategy settings."""
    if not isinstance(params, dict):
        return {}
    raw = params.get("bv_convergence", {})
    if not isinstance(raw, dict):
        return {}
    return {
        "clip_exponent":              _bool(raw.get("clip_exponent", True)),
        "exponent_clip":              float(raw.get("exponent_clip", 50.0)),
        "regularize_concentration":   _bool(raw.get("regularize_concentration", True)),
        "conc_floor":                 float(raw.get("conc_floor", 1e-8)),
        "use_eta_in_bv":              _bool(raw.get("use_eta_in_bv", True)),
    }


def _add_bv_scaling_to_transform(
    scaling: dict,
    bv_cfg: dict,
    *,
    n_species: int,
    nondim_enabled: bool,
    kappa_inputs_dimless: bool,
    concentration_inputs_dimless: bool = False,
) -> dict:
    """Augment the ModelScaling dict with BV-specific nondimensional quantities.

    Returns a new dict (does not mutate scaling).

    Parameters
    ----------
    concentration_inputs_dimless:
        If True, ``bv_bc.c_ref`` values are already dimensionless (divided by the
        concentration scale) and must NOT be scaled again.  If False (default),
        c_ref values are in physical units (mol/m³) and will be divided by
        ``concentration_scale_mol_m3``.  Mirror of ``kappa_inputs_dimless``.
    """
    kappa_scale = scaling.get("kappa_scale_m_s", 1.0)
    thermal_voltage_v = scaling.get("thermal_voltage_v", 0.02569)
    conc_scale = scaling.get("concentration_scale_mol_m3", 1.0)
    potential_scale = scaling.get("potential_scale_v", 1.0)

    k0_raw = bv_cfg["k0_vals"]
    c_ref_raw = bv_cfg["c_ref_vals"]
    E_eq_raw = bv_cfg["E_eq_v"]

    if not nondim_enabled:
        # Dimensional: keep physical values, apply F/(RT) as EM prefactor for exponents
        k0_model = k0_raw
        c_ref_model = c_ref_raw
        # Exponent in BV: exp(±α · F · η / RT).  In dimensional mode phi is in Volts.
        # The BV exponent prefactor = F/(RT).
        bv_exponent_scale = FARADAY_CONSTANT / (GAS_CONSTANT * scaling["temperature_k"])
        E_eq_model = E_eq_raw
    else:
        # Nondimensional: k0 → k0/κ_scale, c_ref → c_ref/c_scale
        # Exponent: exp(±α · η̂) where η̂ = η/V_T (dimensionless potential already)
        # Since potential is already in V_T units, exponent prefactor = 1.
        bv_exponent_scale = 1.0
        if kappa_inputs_dimless:
            k0_model = k0_raw  # already dimensionless
        else:
            k0_model = [v / kappa_scale for v in k0_raw]
        if concentration_inputs_dimless:
            c_ref_model = c_ref_raw  # already dimensionless, do not divide by c_scale
        else:
            c_ref_model = [v / conc_scale for v in c_ref_raw]
        # E_eq in thermal voltage units
        E_eq_model = E_eq_raw / potential_scale

    out = dict(scaling)
    out["bv_k0_model_vals"] = k0_model
    out["bv_c_ref_model_vals"] = c_ref_model
    out["bv_exponent_scale"] = bv_exponent_scale
    out["bv_E_eq_model"] = E_eq_model
    out["bv_alpha_vals"] = bv_cfg["alpha_vals"]
    out["bv_stoichiometry"] = bv_cfg["stoichiometry"]
    return out


# ---------------------------------------------------------------------------
# Multi-reaction BV config helpers
# ---------------------------------------------------------------------------

def _get_bv_reactions_cfg(params: Any, n_species: int) -> list[dict] | None:
    """Parse multi-reaction BV config from ``bv_bc.reactions``.

    Returns a list of reaction dicts if present, or None to signal
    that the caller should use the legacy per-species path.
    """
    if not isinstance(params, dict):
        return None
    raw = params.get("bv_bc", {})
    if not isinstance(raw, dict):
        return None
    reactions_raw = raw.get("reactions")
    if reactions_raw is None:
        return None
    if not isinstance(reactions_raw, list) or len(reactions_raw) == 0:
        return None

    reactions = []
    for j, rxn in enumerate(reactions_raw):
        cat = rxn.get("cathodic_species")
        anod = rxn.get("anodic_species")
        if cat is None:
            raise ValueError(f"Reaction {j}: 'cathodic_species' is required")
        stoi = rxn.get("stoichiometry")
        if stoi is None:
            raise ValueError(f"Reaction {j}: 'stoichiometry' is required")
        if len(stoi) != n_species:
            raise ValueError(
                f"Reaction {j}: stoichiometry length {len(stoi)} != n_species {n_species}"
            )
        # Optional cathodic concentration factors: e.g. (c_H+/c_ref)^2
        cat_conc_factors_raw = rxn.get("cathodic_conc_factors", [])
        cat_conc_factors = []
        for f_cfg in cat_conc_factors_raw:
            sp_idx = f_cfg.get("species")
            if sp_idx is None:
                raise ValueError(
                    f"Reaction {j}: cathodic_conc_factors entry missing 'species'"
                )
            if int(sp_idx) < 0 or int(sp_idx) >= n_species:
                raise ValueError(
                    f"Reaction {j}: cathodic_conc_factors species {sp_idx} "
                    f"out of range [0, {n_species})"
                )
            cat_conc_factors.append({
                "species": int(sp_idx),
                "power": int(f_cfg.get("power", 1)),
                "c_ref_nondim": float(f_cfg.get("c_ref_nondim", 1.0)),
            })

        reactions.append({
            "k0": float(rxn.get("k0", 1e-5)),
            "alpha": float(rxn.get("alpha", 0.5)),
            "cathodic_species": int(cat),
            "anodic_species": int(anod) if anod is not None else None,
            "c_ref": float(rxn.get("c_ref", 1.0)),
            "stoichiometry": [int(s) for s in stoi],
            "n_electrons": int(rxn.get("n_electrons", 2)),
            "reversible": _bool(rxn.get("reversible", True)),
            "cathodic_conc_factors": cat_conc_factors,
        })
    return reactions


def _add_bv_reactions_scaling_to_transform(
    scaling: dict,
    reactions: list[dict],
    *,
    nondim_enabled: bool,
    kappa_inputs_dimless: bool,
    concentration_inputs_dimless: bool = False,
    E_eq_v: float = 0.0,
) -> dict:
    """Augment ModelScaling with multi-reaction BV quantities.

    Each reaction's k0 and c_ref are nondimensionalized individually.
    """
    kappa_scale = scaling.get("kappa_scale_m_s", 1.0)
    conc_scale = scaling.get("concentration_scale_mol_m3", 1.0)
    potential_scale = scaling.get("potential_scale_v", 1.0)

    if not nondim_enabled:
        bv_exponent_scale = FARADAY_CONSTANT / (GAS_CONSTANT * scaling["temperature_k"])
        E_eq_model = E_eq_v
    else:
        bv_exponent_scale = 1.0
        E_eq_model = E_eq_v / potential_scale

    scaled_reactions = []
    for rxn in reactions:
        srxn = dict(rxn)
        if not nondim_enabled:
            srxn["k0_model"] = rxn["k0"]
            srxn["c_ref_model"] = rxn["c_ref"]
        else:
            if kappa_inputs_dimless:
                srxn["k0_model"] = rxn["k0"]
            else:
                srxn["k0_model"] = rxn["k0"] / kappa_scale
            if concentration_inputs_dimless:
                srxn["c_ref_model"] = rxn["c_ref"]
            else:
                srxn["c_ref_model"] = rxn["c_ref"] / conc_scale

        # Scale cathodic_conc_factors c_ref_nondim values
        scaled_factors = []
        for f_cfg in rxn.get("cathodic_conc_factors", []):
            sf = dict(f_cfg)
            if nondim_enabled and not concentration_inputs_dimless:
                sf["c_ref_nondim"] = f_cfg["c_ref_nondim"] / conc_scale
            else:
                sf["c_ref_nondim"] = f_cfg["c_ref_nondim"]
            scaled_factors.append(sf)
        srxn["cathodic_conc_factors"] = scaled_factors

        scaled_reactions.append(srxn)

    out = dict(scaling)
    out["bv_reactions"] = scaled_reactions
    out["bv_exponent_scale"] = bv_exponent_scale
    out["bv_E_eq_model"] = E_eq_model
    return out


# ---------------------------------------------------------------------------
# Public solver API
# ---------------------------------------------------------------------------

def build_context(solver_params, *, mesh=None) -> dict:
    """Build mesh and function spaces for the Butler-Volmer PNP solver.

    Parameters
    ----------
    solver_params : list or SolverParams
        11-element parameter list.
    mesh : firedrake.Mesh, optional
        Pre-built mesh.  If ``None`` (default), a ``UnitSquareMesh(32, 32)``
        is created for backward compatibility.  Pass a
        :func:`make_graded_interval_mesh` for 1-D EDL problems.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    if not (len(z_vals) == len(D_vals) == len(a_vals) == n_species):
        raise ValueError(
            f"z_vals, D_vals, and a_vals must all have length n_species ({n_species}); "
            f"got lengths {len(z_vals)}, {len(D_vals)}, {len(a_vals)}"
        )

    if mesh is None:
        mesh = fd.UnitSquareMesh(32, 32)
    V_scalar = fd.FunctionSpace(mesh, "CG", order)
    W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])
    U = fd.Function(W)
    U_prev = fd.Function(W)

    return {
        "mesh": mesh,
        "V_scalar": V_scalar,
        "W": W,
        "U": U,
        "U_prev": U_prev,
        "n_species": n_species,
    }


def build_forms(ctx: dict, solver_params) -> dict:
    """Assemble weak forms for the Butler-Volmer PNP problem.

    The BV boundary condition is applied at the electrode boundary::

        J_i · n  =  s_i · k̂₀_i · [max(ĉ_i, ε) · exp(−α_i · clip(η̂ − Ê_eq, ±50))
                                   − ĉ_ref_i     · exp((1−α_i) · clip(η̂ − Ê_eq, ±50))]

    The overpotential η̂ is taken from ``phi_applied_func`` (the Dirichlet BC value),
    not from the interior φ field — this keeps the Newton Jacobian well-conditioned.

    Parameters
    ----------
    ctx:
        Context dict from :func:`build_context`.
    solver_params:
        11-element SolverParams / list.  ``solver_params[10]`` must contain a
        ``"bv_bc"`` key (see module docstring).

    Returns
    -------
    dict
        Updated context including ``F_res``, ``J_form``, ``bcs``,
        ``bv_settings``, ``bv_scaling``, ``nondim``.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    c0_raw = [float(v) for v in (
        [c0] * n_species if np.isscalar(c0) else list(c0)
    )][:n_species]

    mesh = ctx["mesh"]
    W = ctx["W"]
    n = ctx["n_species"]

    # Parse BV config and convergence options.
    bv_cfg = _get_bv_cfg(params, n)
    conv_cfg = _get_bv_convergence_cfg(params)
    reactions_cfg = _get_bv_reactions_cfg(params, n)
    use_reactions = reactions_cfg is not None

    # Nondimensionalization (standard path, reusing robin transform).
    # Pass a dummy robin dict so the function doesn't crash; BV params are
    # handled separately via _add_bv_*_scaling_to_transform.
    dummy_robin = {
        "kappa_vals": [1.0] * n,
        "c_inf_vals": bv_cfg["c_ref_vals"],
        "electrode_marker": bv_cfg["electrode_marker"],
        "concentration_marker": bv_cfg["concentration_marker"],
        "ground_marker": bv_cfg["ground_marker"],
    }
    base_scaling = build_model_scaling(
        params=params,
        n_species=n,
        dt=dt,
        t_end=t_end,
        D_vals=D_vals,
        c0_vals=c0_raw,
        phi_applied=phi_applied,
        phi0=phi0,
        robin=dummy_robin,
    )

    nondim_cfg = _get_nondim_cfg(params)
    nondim_enabled = _bool(nondim_cfg.get("enabled", False))
    kappa_inputs_dimless = _bool(nondim_cfg.get("kappa_inputs_are_dimensionless", True))
    concentration_inputs_dimless = _bool(
        nondim_cfg.get("concentration_inputs_are_dimensionless", False)
    )

    if use_reactions:
        E_eq_v = float(bv_cfg.get("E_eq_v", 0.0))
        scaling = _add_bv_reactions_scaling_to_transform(
            base_scaling, reactions_cfg,
            nondim_enabled=nondim_enabled,
            kappa_inputs_dimless=kappa_inputs_dimless,
            concentration_inputs_dimless=concentration_inputs_dimless,
            E_eq_v=E_eq_v,
        )
    else:
        scaling = _add_bv_scaling_to_transform(
            base_scaling, bv_cfg,
            n_species=n,
            nondim_enabled=nondim_enabled,
            kappa_inputs_dimless=kappa_inputs_dimless,
            concentration_inputs_dimless=concentration_inputs_dimless,
        )

    electrode_marker = bv_cfg["electrode_marker"]
    concentration_marker = bv_cfg["concentration_marker"]
    ground_marker = bv_cfg["ground_marker"]

    dx = fd.Measure("dx", domain=mesh)
    ds = fd.Measure("ds", domain=mesh)
    R_space = fd.FunctionSpace(mesh, "R", 0)

    # Log-diffusivity controls (same as robin_solver).
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(scaling["D_model_vals"][i])))
    D = [fd.exp(m[i]) for i in range(n)]

    z = [fd.Constant(float(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    ci_prev = fd.split(U_prev)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    em = float(scaling["electromigration_prefactor"])
    dt_m = float(scaling["dt_model"])

    # Electrode potential constant (for BV exponent).
    phi_applied_func = fd.Function(R_space, name="phi_applied")
    phi_applied_func.assign(float(scaling["phi_applied_model"]))
    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(float(scaling["phi0_model"]))

    # E_eq offset in model space.
    E_eq_model = fd.Constant(float(scaling["bv_E_eq_model"]))
    bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))

    # Build the η̂ expression used in BV exponent.
    if conv_cfg["use_eta_in_bv"]:
        eta_hat_raw = phi_applied_func - E_eq_model
    else:
        eta_hat_raw = phi - E_eq_model

    eta_hat_scaled = bv_exp_scale * eta_hat_raw

    # Exponent clipping to prevent float overflow.
    if conv_cfg["clip_exponent"]:
        clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
        eta_clipped = fd.min_value(fd.max_value(eta_hat_scaled, -clip_val), clip_val)
    else:
        eta_clipped = eta_hat_scaled

    # Steric chemical potential (Bikerman model): mu_steric = ln(1 - sum_j a_j c_j)
    a_vals_list = [float(v) for v in a_vals]
    steric_active = any(v != 0.0 for v in a_vals_list)
    if steric_active:
        a_consts = [fd.Constant(v) for v in a_vals_list]
        packing = fd.max_value(
            fd.Constant(1.0) - sum(a_consts[j] * ci[j] for j in range(n)),
            fd.Constant(1e-8),
        )
        mu_steric = fd.ln(packing)

    # Regularized surface concentrations (shared by both BV paths).
    if conv_cfg["regularize_concentration"]:
        eps_c = fd.Constant(float(conv_cfg["conc_floor"]))
        c_surf = [fd.max_value(ci[i], eps_c) for i in range(n)]
    else:
        c_surf = list(ci)

    # Residual: Nernst-Planck time-stepping + BV electrode BC.
    F_res = 0

    for i in range(n):
        c_i = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]

        # Electromigration drift (zero for neutral species, z=0).
        drift = em * z[i] * phi
        if steric_active:
            Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift) + c_i * fd.grad(mu_steric))
        else:
            Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift))

        # Time-stepping residual.
        F_res += ((c_i - c_old) / dt_m) * v * dx
        F_res += fd.dot(Jflux, fd.grad(v)) * dx

    # Butler-Volmer boundary flux.
    if use_reactions:
        # ---- Multi-reaction path ----
        # Each reaction j has its own rate R_j; stoichiometry matrix couples
        # reactions to species.
        bv_rate_exprs = []
        rxns_scaled = scaling["bv_reactions"]
        for j, rxn in enumerate(rxns_scaled):
            k0_j = fd.Constant(float(rxn["k0_model"]))
            alpha_j = fd.Constant(float(rxn["alpha"]))
            cat_idx = rxn["cathodic_species"]

            # Cathodic term: k0 * c_cat * exp(-alpha * eta)
            cathodic = k0_j * c_surf[cat_idx] * fd.exp(-alpha_j * eta_clipped)

            # Apply optional cathodic concentration factors: e.g. (c_H+/c_ref)^power
            for factor in rxn.get("cathodic_conc_factors", []):
                sp_idx = factor["species"]
                power = factor["power"]
                c_ref_f = fd.Constant(float(factor["c_ref_nondim"]))
                cathodic = cathodic * (c_surf[sp_idx] / c_ref_f) ** power

            # Anodic term (if reversible)
            if rxn["reversible"] and rxn["anodic_species"] is not None:
                c_ref_j = fd.Constant(float(rxn["c_ref_model"]))
                anodic = k0_j * c_ref_j * fd.exp(
                    (fd.Constant(1.0) - alpha_j) * eta_clipped
                )
            else:
                anodic = fd.Constant(0.0)

            R_j = cathodic - anodic
            bv_rate_exprs.append(R_j)

            # Apply stoichiometry: F_res -= s_ij * R_j * v_i * ds(electrode)
            stoi = rxn["stoichiometry"]
            for i in range(n):
                if stoi[i] != 0:
                    F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)

    else:
        # ---- Legacy per-species path ----
        bv_rate_exprs = []
        for i in range(n):
            alpha_i = fd.Constant(float(scaling["bv_alpha_vals"][i]))
            stoi_i = int(scaling["bv_stoichiometry"][i])
            k0_i = fd.Constant(float(scaling["bv_k0_model_vals"][i]))
            c_ref_i = fd.Constant(float(scaling["bv_c_ref_model_vals"][i]))

            bv_flux_i = k0_i * (
                c_surf[i] * fd.exp(-alpha_i * eta_clipped)
                - c_ref_i * fd.exp((fd.Constant(1.0) - alpha_i) * eta_clipped)
            )
            bv_rate_exprs.append(bv_flux_i)
            # Weak-form sign: F_res -= stoi_i × bv_flux_i × v × ds
            F_res -= fd.Constant(float(stoi_i)) * bv_flux_i * v_list[i] * ds(electrode_marker)

    # Poisson equation.
    suppress_poisson_source = _bool(nondim_cfg.get("suppress_poisson_source", False))
    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
    if not suppress_poisson_source:
        F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx

    # Dirichlet BCs.
    bc_phi_electrode = fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    bc_ci = [
        fd.DirichletBC(
            W.sub(i),
            fd.Constant(float(scaling["c0_model_vals"][i])),
            concentration_marker,
        )
        for i in range(n)
    ]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]

    J_form = fd.derivative(F_res, U)

    ctx.update({
        "F_res": F_res,
        "J_form": J_form,
        "bcs": bcs,
        "logD_funcs": m,
        "D_consts": D,
        "z_consts": z,
        "phi_applied_func": phi_applied_func,
        "phi0_func": phi0_func,
        "bv_settings": bv_cfg,
        "bv_convergence": conv_cfg,
        "bv_rate_exprs": bv_rate_exprs,
        "nondim": scaling,
    })
    return ctx


def set_initial_conditions(ctx: dict, solver_params, blob: bool = False) -> None:
    """Set initial conditions for the BV solver.

    Default ``blob=False``: uniform concentrations + linear potential profile.
    For BV problems, starting near equilibrium (η ≈ 0) significantly helps
    Newton convergence at the first timestep.

    Parameters
    ----------
    blob:
        If True, superimpose a Gaussian concentration perturbation.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    scaling = ctx.get("nondim", {})

    c0_raw = [float(v) for v in ([c0] * n if np.isscalar(c0) else list(c0))][:n]
    c0_model = scaling.get("c0_model_vals", c0_raw)
    phi_applied_model = scaling.get("phi_applied_model", float(phi_applied))

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()

    if blob and ndim >= 2:
        x, y = coords[0], coords[1]
        A = fd.Constant(0.1)
        x0 = fd.Constant(0.5)
        y0 = fd.Constant(0.2)
        sigma = fd.Constant(0.08)
        gaussian_blob = A * fd.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        for i in range(n):
            U_prev.sub(i).interpolate(fd.Constant(float(c0_model[i])) + gaussian_blob)
    else:
        for i in range(n):
            U_prev.sub(i).assign(fd.Constant(float(c0_model[i])))

    # Linear potential profile: phi_applied at electrode, 0 at bulk.
    # 1D: electrode at x=0, bulk at x=1 → phi = phi_applied * (1 - x)
    # 2D: electrode at y=0, bulk at y=1 → phi = phi_applied * (1 - y)
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]
    U_prev.sub(n).interpolate(fd.Constant(float(phi_applied_model)) * (1.0 - spatial_var))
    ctx["U"].assign(U_prev)


def forsolve_bv(
    ctx: dict,
    solver_params,
    print_interval: int = 20,
) -> fd.Function:
    """Run the BV PNP time-stepping loop to steady state.

    Parameters
    ----------
    ctx:
        Context dict from :func:`build_forms`.
    solver_params:
        11-element list / SolverParams.
    print_interval:
        Print progress every N steps.

    Returns
    -------
    Function
        Final solved state ``U``.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]
    scaling = ctx.get("nondim", {})

    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))
    num_steps = max(1, int(round(t_end_model / dt_model)))

    J = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print(f"[bv_solver] step {step}/{num_steps}  eta_hat={float(ctx['phi_applied_func'].dat.data[0]):.4f}")
        solver.solve()
        U_prev.assign(U)

    return U


def solve_bv_with_continuation(
    solver_params,
    *,
    eta_target: float,
    eta_steps: int = 20,
    print_interval: int = 20,
    mesh=None,
) -> fd.Function:
    """Solve BV problem using potential continuation from η̂ = 0 to ``eta_target``.

    This is the primary convergence strategy for large overpotentials.
    At each continuation step:

    1. Set ``phi_applied_func`` to the current η̂.
    2. Run the time-stepping loop.
    3. Use the result as the initial condition for the next step.

    If Newton diverges at a step, the step size is halved (bisection up to 4 times).

    Parameters
    ----------
    solver_params:
        11-element list / SolverParams.  The ``phi_applied`` field is overridden
        during continuation — its value sets the **final** target, not the starting
        value.  The starting value is always 0 (equilibrium).
    eta_target:
        Final dimensionless overpotential to reach.  In nondim mode this is
        φ_electrode / V_T.  In dimensional mode this is φ_electrode in Volts.
    eta_steps:
        Number of continuation steps between 0 and eta_target.
    print_interval:
        Passed to :func:`forsolve_bv`.

    Returns
    -------
    Function
        The solved state at ``eta_target``.
    """
    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    # Build context at eta = 0.
    params_0 = _clone_params_with_phi(solver_params, phi_applied=0.0)
    ctx = build_context(params_0, mesh=mesh)
    ctx = build_forms(ctx, params_0)
    set_initial_conditions(ctx, params_0, blob=False)

    # Build the continuation path.
    path = np.linspace(0.0, eta_target, eta_steps + 1)[1:]  # skip 0 (already at IC)

    scaling = ctx["nondim"]
    J = fd.derivative(ctx["F_res"], ctx["U"])
    problem = fd.NonlinearVariationalProblem(ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))
    num_steps = max(1, int(round(t_end_model / dt_model)))

    def _try_timestep(eta_try: float) -> bool:
        """Attempt time-stepping at eta_try.  Returns True on success.

        On failure the context state (U, U_prev, phi_applied_func) is NOT
        guaranteed to be consistent — caller must restore from a checkpoint.
        """
        ctx["phi_applied_func"].assign(float(eta_try))
        try:
            for _ in range(num_steps):
                solver.solve()
                ctx["U_prev"].assign(ctx["U"])
            return True
        except fd.ConvergenceError:
            return False

    # eta currently solved (start = 0, already in ctx)
    eta_current = 0.0
    max_sub = 6   # maximum recursive bisection depth per target step

    for i_cont, eta_target_step in enumerate(path):
        print(f"[continuation] step {i_cont+1}/{len(path)}  eta_hat → {eta_target_step:.4f}")

        # Checkpoint the last known-good state before attempting this step.
        U_ckpt = fd.Function(ctx["U"])
        U_prev_ckpt = fd.Function(ctx["U_prev"])

        # Try the full step directly.
        if _try_timestep(eta_target_step):
            eta_current = eta_target_step
            continue  # success — proceed to next planned step

        # Direct step failed — adaptive sub-stepping between eta_current and eta_target_step.
        print(f"  [sub-step] direct solve failed; inserting sub-steps "
              f"from eta={eta_current:.4f} to eta={eta_target_step:.4f}")

        eta_lo = eta_current
        eta_hi = eta_target_step
        sub_count = 0

        while sub_count < max_sub:
            # Restore to last good state and try the midpoint.
            ctx["U"].assign(U_ckpt)
            ctx["U_prev"].assign(U_prev_ckpt)

            eta_mid = (eta_lo + eta_hi) / 2.0
            print(f"  [sub-step {sub_count+1}/{max_sub}] eta_mid={eta_mid:.4f}")

            if _try_timestep(eta_mid):
                # Midpoint converged — update checkpoint and try the full target.
                U_ckpt.assign(ctx["U"])
                U_prev_ckpt.assign(ctx["U_prev"])
                eta_lo = eta_mid

                ctx["U"].assign(U_ckpt)
                ctx["U_prev"].assign(U_prev_ckpt)
                if _try_timestep(eta_target_step):
                    eta_current = eta_target_step
                    break  # reached the target
                # Target still fails — bisect again from eta_mid toward eta_hi
                sub_count += 1
            else:
                # Midpoint failed too — halve the lower half
                eta_hi = eta_mid
                sub_count += 1
        else:
            raise RuntimeError(
                f"[continuation] Failed to converge at eta={eta_target_step:.4f} after "
                f"{max_sub} sub-stepping attempts (reached eta={eta_lo:.4f})."
            )

    return ctx["U"]


def _clone_params_with_phi(solver_params, *, phi_applied: float):
    """Return a new SolverParams-like list with phi_applied replaced."""
    lst = list(solver_params)   # works for both list and SolverParams
    lst[7] = float(phi_applied)
    return lst
