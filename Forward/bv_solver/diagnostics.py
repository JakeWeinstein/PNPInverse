"""Per-voltage diagnostics for the BV-PNP forward solver.

Used by the C+D orchestrator and study scripts to capture failure-mode
information (max phi, surface concentrations, SNES reason / iters,
Bikerman steric saturation) without mutating the residual.

The orchestrator stashes the active solver as ``ctx['_last_solver']`` in
``_build_for_voltage`` so SNES diagnostics can be retrieved here without
threading a solver argument through ``per_point_callback``.

All routines treat missing ctx keys gracefully — diagnostics are
best-effort, especially on the failure path where the orchestrator has
already rolled back to the last accepted state.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import firedrake as fd
import numpy as np

from .config import _get_bv_boltzmann_counterions_cfg


def surface_field_means(ctx: dict[str, Any]) -> dict[str, float]:
    """Mean ``log(c_i)`` and ``c_i`` over the electrode boundary, plus phi.

    Returns a flat dict with keys ``electrode_area_nondim``,
    ``u{i}_surface_mean``, ``c{i}_surface_mean``, ``phi_surface_mean``,
    plus ``mu{i}_surface_mean`` for any species treated as muh primary
    variables (``i in ctx['mu_species']``).

    For the production logc formulation, ``u{i}_surface_mean`` is the
    surface mean of raw ``U.sub(i)`` (== ``u_i`` = ``log(c_i)``).

    For the muh formulation (``ctx['logc_muh_transform']`` is True), the
    surface mean of ``log(c_i)`` for the proton species is computed from
    the muh-aware expression ``ctx['u_exprs'][i] = mu_H - em*z_H*phi``,
    so ``u{i}_surface_mean`` is still semantically the surface mean of
    ``log(c_i)`` -- the contract is consistent across formulations.
    Additionally, ``mu{i}_surface_mean`` reports the surface mean of the
    raw muh primary variable ``U.sub(i)``.

    NB: ``c{i}_surface_mean = exp(u_mean)`` retains the existing
    ``exp(mean(u))`` Jensen-fault for both formulations -- a separate
    correctness fix is tracked.  Overflowing ``exp(u)`` values are
    reported as ``float('inf')``.
    """
    bv_settings = ctx["bv_settings"]
    elec_marker = int(bv_settings["electrode_marker"])
    mesh = ctx["mesh"]
    U = ctx["U"]
    n = ctx["n_species"]

    # Prefer ctx-stored log-c expressions when available (handles muh).
    # Falls back to raw U.sub(i) for backwards compat with logc-only ctx.
    u_exprs = ctx.get("u_exprs", None)
    mu_species = frozenset(int(i) for i in ctx.get("mu_species", []))

    ds_e = fd.ds(domain=mesh, subdomain_id=elec_marker)
    area = float(fd.assemble(fd.Constant(1.0) * ds_e))
    out: dict[str, float] = {"electrode_area_nondim": area}
    if area <= 0:
        return out

    for i in range(n):
        log_c_expr = u_exprs[i] if u_exprs is not None else U.sub(i)
        u_mean = float(fd.assemble(log_c_expr * ds_e)) / area
        out[f"u{i}_surface_mean"] = u_mean
        try:
            out[f"c{i}_surface_mean"] = float(np.exp(u_mean))
        except OverflowError:
            out[f"c{i}_surface_mean"] = float("inf")
        if i in mu_species:
            # Raw muh primary variable surface mean (offset from u_mean by em*z*phi_surf).
            out[f"mu{i}_surface_mean"] = float(fd.assemble(U.sub(i) * ds_e)) / area
    out["phi_surface_mean"] = float(fd.assemble(U.sub(n) * ds_e)) / area
    return out


def collect_diagnostics(
    ctx: dict[str, Any],
    *,
    phase: str,
    params: Optional[dict[str, Any]] = None,
    picard_iters: Optional[int] = None,
    steric_cap: float = 100.0,
) -> dict[str, Any]:
    """Assemble a per-voltage diagnostic record.

    Parameters
    ----------
    phase:
        Pipeline stage that produced the state, e.g. ``"cold_z_ramp"``,
        ``"cold_no_z_ramp"``, ``"warm_walk"``, or ``"failed"``.
    params:
        Optional ``solver_params[10]`` dict.  When provided, analytic
        Boltzmann counterions are parsed from ``params['bv_bc']`` and the
        per-counterion surface concentration ``c_bulk · exp(-z·phi_surf)``
        is computed.  When omitted, those fields are absent.
    picard_iters:
        Number of Picard iterations the analytical IC took, if
        applicable.  ``None`` for the linear-phi initializer.
    steric_cap:
        Bikerman saturation threshold on Boltzmann counterion surface
        concentration (nondim).
    """
    n = ctx["n_species"]
    U = ctx["U"]
    out: dict[str, Any] = {"phase": phase}

    phi_data = U.dat[n].data_ro
    out["max_phi"] = float(phi_data.max()) if phi_data.size else None
    out["min_phi"] = float(phi_data.min()) if phi_data.size else None

    # Per-volume min/max of log(c_i): for non-mu species this is raw
    # U.dat[i] (== u_i); for mu species this is the muh-reconstructed
    # log(c_i) = mu_H - em*z_H*phi at dofs.  Both formulations expose
    # ``min_u{i}/max_u{i}`` consistently as log-concentration extremes.
    # Additionally report ``min_mu{i}/max_mu{i}`` for muh species so the
    # raw muh primary-variable extremes are observable.
    em = float(ctx.get("nondim", {}).get("electromigration_prefactor", 1.0))
    z_consts = ctx.get("z_consts", [])
    mu_species = frozenset(int(i) for i in ctx.get("mu_species", []))

    for i in range(n):
        u_data_raw = U.dat[i].data_ro
        if u_data_raw.size == 0:
            continue
        if i in mu_species:
            try:
                z_i = float(z_consts[i])
            except (IndexError, TypeError):
                z_i = 0.0
            log_c_data = u_data_raw - em * z_i * phi_data
            out[f"max_u{i}"] = float(log_c_data.max())
            out[f"min_u{i}"] = float(log_c_data.min())
            out[f"max_mu{i}"] = float(u_data_raw.max())
            out[f"min_mu{i}"] = float(u_data_raw.min())
        else:
            out[f"max_u{i}"] = float(u_data_raw.max())
            out[f"min_u{i}"] = float(u_data_raw.min())

    try:
        out.update(surface_field_means(ctx))
    except Exception as exc:
        out["surface_fields_error"] = f"{type(exc).__name__}: {exc}"

    if params is not None:
        try:
            counterions = _get_bv_boltzmann_counterions_cfg(params)
        except Exception:
            counterions = []
        phi_surf = out.get("phi_surface_mean")
        if phi_surf is not None and counterions:
            # For bikerman entries we need bulk dynamic packing
            # ``A_dyn_bulk`` and the surface ``A_dyn_surf`` to evaluate
            # the closure at the surface.  Pull from ctx when available
            # (steric_a_funcs published by build_forms_logc); fall back
            # to ``A_dyn_surf = A_dyn_bulk`` if the steric block was
            # never built (purely-ideal config).
            a_dyn_floats: list[float] = []
            steric_a_funcs = ctx.get("steric_a_funcs") or []
            for af in steric_a_funcs:
                try:
                    a_dyn_floats.append(float(af.dat.data_ro[0]))
                except Exception:
                    a_dyn_floats.append(0.0)
            scaling = ctx.get("nondim", {}) or {}
            c0_dyn = list(scaling.get("c0_model_vals", [])) or []
            A_dyn_bulk = (
                sum(a * c for a, c in zip(a_dyn_floats, c0_dyn[:len(a_dyn_floats)]))
                if a_dyn_floats and c0_dyn else 0.0
            )
            # Surface A_dyn from the per-species c{i}_surface_mean already
            # in `out` (populated by surface_field_means).
            A_dyn_surf = 0.0
            for i, a_i in enumerate(a_dyn_floats):
                c_i_surf = out.get(f"c{i}_surface_mean")
                if c_i_surf is None or not np.isfinite(c_i_surf):
                    A_dyn_surf = A_dyn_bulk  # conservative fallback
                    break
                A_dyn_surf += a_i * float(c_i_surf)

            within = True
            for j, ci_spec in enumerate(counterions):
                z = int(ci_spec["z"])
                c_bulk = float(ci_spec["c_bulk_nondim"])
                steric_mode = ci_spec.get("steric_mode", "ideal")
                if steric_mode == "bikerman":
                    a_b = float(ci_spec["a_nondim"])
                    phi_clamp = float(ci_spec.get("phi_clamp", 50.0))
                    theta_b = 1.0 - A_dyn_bulk - a_b * c_bulk
                    if theta_b <= 0:
                        c_surf = float("inf")
                    else:
                        phi_eff = max(-phi_clamp, min(phi_clamp, float(phi_surf)))
                        try:
                            q = float(np.exp(-z * phi_eff))
                        except OverflowError:
                            q = float("inf")
                        free_dyn = max(1.0 - A_dyn_surf, 1e-10)
                        denom = theta_b + a_b * c_bulk * q
                        c_surf = (
                            c_bulk * q * free_dyn / denom
                            if np.isfinite(q) else (1.0 - A_dyn_surf) / a_b
                        )
                else:
                    try:
                        c_surf = c_bulk * float(np.exp(-z * phi_surf))
                    except OverflowError:
                        c_surf = float("inf")
                out[f"c_counterion{j}_surface_mean"] = c_surf
                out[f"c_counterion{j}_z"] = z
                out[f"c_counterion{j}_steric_mode"] = steric_mode
                if not (c_surf <= steric_cap):
                    within = False
            out["surface_counterion_within_steric"] = within

    out["picard_iters"] = picard_iters
    out["initializer_fallback"] = bool(ctx.get("initializer_fallback", False))

    solver = ctx.get("_last_solver")
    if solver is not None:
        try:
            out["snes_reason"] = str(solver.snes.getConvergedReason())
            out["snes_iters"] = int(solver.snes.getIterationNumber())
        except Exception as exc:
            out["snes_reason"] = f"unavailable: {type(exc).__name__}"
            out["snes_iters"] = None
    else:
        out["snes_reason"] = "no_solver_on_ctx"
        out["snes_iters"] = None

    return out


def check_steric_saturation(
    ctx: dict[str, Any],
    *,
    params: dict[str, Any],
    cap: float = 100.0,
    emit_warning: bool = True,
) -> dict[str, Any]:
    """Bikerman-cap watch for the analytic Boltzmann counterion residual.

    The Poisson-Boltzmann ClO4- residual in ``boltzmann.py`` does not
    enforce steric saturation, so at high anodic phi the analytic
    surface concentration ``c_bulk · exp(+phi)`` can exceed the
    Bikerman cap (~100 nondim) where the *physical* state is sterically
    forbidden.  Emit a ``UserWarning`` when this happens; the converged
    Newton state is non-physical even if the residual is satisfied.

    Returns a dict with per-counterion surface concentrations and an
    overall ``within_steric`` flag.
    """
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    if not counterions:
        return {"within_steric": True, "counterions": []}

    fields = surface_field_means(ctx)
    phi_surf = fields.get("phi_surface_mean")
    if phi_surf is None:
        return {"within_steric": True, "counterions": []}

    # Same surface/bulk dynamic-packing reconstruction as
    # collect_diagnostics — needed to evaluate the closure for any
    # bikerman entries.
    a_dyn_floats: list[float] = []
    for af in (ctx.get("steric_a_funcs") or []):
        try:
            a_dyn_floats.append(float(af.dat.data_ro[0]))
        except Exception:
            a_dyn_floats.append(0.0)
    scaling = ctx.get("nondim", {}) or {}
    c0_dyn = list(scaling.get("c0_model_vals", [])) or []
    A_dyn_bulk = (
        sum(a * c for a, c in zip(a_dyn_floats, c0_dyn[:len(a_dyn_floats)]))
        if a_dyn_floats and c0_dyn else 0.0
    )
    A_dyn_surf = 0.0
    for i, a_i in enumerate(a_dyn_floats):
        c_i_surf = fields.get(f"c{i}_surface_mean")
        if c_i_surf is None or not np.isfinite(c_i_surf):
            A_dyn_surf = A_dyn_bulk
            break
        A_dyn_surf += a_i * float(c_i_surf)

    entries: list[dict[str, Any]] = []
    within = True
    for j, ci_spec in enumerate(counterions):
        z = int(ci_spec["z"])
        c_bulk = float(ci_spec["c_bulk_nondim"])
        steric_mode = ci_spec.get("steric_mode", "ideal")
        if steric_mode == "bikerman":
            a_b = float(ci_spec["a_nondim"])
            phi_clamp = float(ci_spec.get("phi_clamp", 50.0))
            theta_b = 1.0 - A_dyn_bulk - a_b * c_bulk
            if theta_b <= 0:
                c_surf = float("inf")
            else:
                phi_eff = max(-phi_clamp, min(phi_clamp, float(phi_surf)))
                try:
                    q = float(np.exp(-z * phi_eff))
                except OverflowError:
                    q = float("inf")
                free_dyn = max(1.0 - A_dyn_surf, 1e-10)
                denom = theta_b + a_b * c_bulk * q
                c_surf = (
                    c_bulk * q * free_dyn / denom
                    if np.isfinite(q) else (1.0 - A_dyn_surf) / a_b
                )
        else:
            try:
                c_surf = c_bulk * float(np.exp(-z * phi_surf))
            except OverflowError:
                c_surf = float("inf")
        entries.append({"index": j, "z": z, "c_bulk_nondim": c_bulk,
                        "steric_mode": steric_mode, "c_surf": c_surf})
        if not (c_surf <= cap):
            within = False
            if emit_warning:
                warnings.warn(
                    f"steric saturation exceeded: counterion {j} "
                    f"(z={z}, mode={steric_mode}) surface c={c_surf:.4g} "
                    f"> cap={cap:.4g} at phi_surf={phi_surf:.4g}; the "
                    f"analytic Boltzmann residual exceeds the Bikerman cap "
                    f"despite Newton convergence — investigate.",
                    UserWarning,
                    stacklevel=2,
                )
    return {"within_steric": within, "counterions": entries,
            "phi_surf": phi_surf, "cap": cap}
