"""Nondimensionalization transform for PNP solver parameters.

This module converts physical (dimensional) or model-space solver parameters
into a unified ``ModelScaling`` dict consumed by :mod:`Forward.robin_solver`.

Nondimensional PDE form
-----------------------
Starting from the dimensional PNP system in a domain of length L_ref with
reference diffusivity D_ref, concentration c_ref, and thermal voltage V_T = RT/F:

**Nernst-Planck** (per species i, nondimensional)::

    ∂ĉᵢ/∂t̂  +  ∇̂ · [ -D̂ᵢ (∇̂ĉᵢ  +  ẑᵢ · ĉᵢ · ∇̂φ̂) ]  =  0

where  D̂ᵢ = Dᵢ/D_ref,  ĉᵢ = cᵢ/c_ref,  φ̂ = φ/V_T,  t̂ = t/t_ref,
t_ref = L_ref²/D_ref.  The electromigration coefficient is implicitly 1
because we scaled by V_T.

**Poisson** (nondimensional)::

    (λ_D/L_ref)² · Δφ̂  =  − Σᵢ zᵢ ĉᵢ

where  λ_D = √(ε·V_T / (F·c_ref))  is the Debye length.

In the code the Poisson residual is assembled as::

    eps_coeff * (∇φ̂, ∇w)  −  rhs_coeff * Σᵢ zᵢ ĉᵢ w  =  0

so  eps_coeff  corresponds to  (λ_D/L_ref)²  and  rhs_coeff = 1  in the
nondimensional case.  See below for the dimensional (``enabled=False``) case.

Dimensional mode (``enabled=False``)
-------------------------------------
When nondimensionalization is disabled, no input rescaling is applied.
The PDE still must be assembled with correct coefficients:

    ε · Δφ  =  F · Σᵢ zᵢ cᵢ          (SI)

so the weak-form coefficients are::

    eps_coeff  = ε  (absolute permittivity, e.g. ~6.95e-10 F/m for water)
    rhs_coeff  = F  (Faraday constant, 96485 C/mol)

**Note on previous behaviour**: prior to this refactor, ``enabled=False``
incorrectly set ``eps_coeff=1.0``.  This had no effect for the symmetric
two-species case (z=[+1,−1], equal concentrations) because the RHS is
identically zero in that case.  It would produce a wrong Poisson equation
for non-symmetric species.  The corrected default now uses the actual
permittivity of water.  If you relied on ``eps_coeff=1.0`` in a non-symmetric
setup, pass ``permittivity_f_m=1.0`` explicitly in ``solver_params[10]["nondim"]``.

Input flags
-----------
All flags live in ``solver_params[10]["nondim"]``:

``diffusivity_inputs_are_dimensionless`` (default ``False``)
    If True, D_vals are already in model units (D̂ = D/D_ref).
``concentration_inputs_are_dimensionless`` (default ``False``)
    If True, c0_vals and c_inf are already in model units.
``potential_inputs_are_dimensionless`` (default ``False``)
    If True, phi_applied and phi0 are already in model units.
``time_inputs_are_dimensionless`` (default ``False``)
    If True, dt and t_end are already in model units.
``kappa_inputs_are_dimensionless`` (default ``True``)
    If True, kappa values are already dimensionless (model units).
    **Default True** because the Robin kappa optimizer works in model space.
    Set to False when passing physical kappa values in m/s.
"""

from __future__ import annotations

import copy
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from Nondim.constants import (
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    DEFAULT_TEMPERATURE_K,
    VACUUM_PERMITTIVITY_F_PER_M,
    DEFAULT_RELATIVE_PERMITTIVITY_WATER,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_list(values: Any, n: int, name: str) -> List[float]:
    if np.isscalar(values):
        return [float(values)] * n
    try:
        out = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a scalar or sequence of length {n}") from exc
    if len(out) != n:
        raise ValueError(f"{name} must have length {n}; got {len(out)}")
    return out


def _pos(value: Any, name: str) -> float:
    v = float(value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0; got {v}.")
    return v


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _get_nondim_cfg(params: Any) -> dict:
    if not isinstance(params, dict):
        return {}
    cfg = params.get("nondim", {})
    return cfg if isinstance(cfg, dict) else {}


def _get_robin_cfg(params: Any, n_species: int) -> dict:
    """Extract validated Robin BC settings from solver_params[10]."""
    robin_raw: dict = {}
    if isinstance(params, dict):
        maybe = params.get("robin_bc", {})
        if isinstance(maybe, dict):
            robin_raw = maybe
        kappa = robin_raw.get("kappa", params.get("robin_kappa", 1.0))
        c_inf = robin_raw.get("c_inf", params.get("robin_c_inf", 0.01))
        electrode_marker = robin_raw.get(
            "electrode_marker", params.get("robin_electrode_marker", 1)
        )
        concentration_marker = robin_raw.get(
            "concentration_marker", params.get("robin_concentration_marker", 3)
        )
        ground_marker = robin_raw.get(
            "ground_marker", params.get("robin_ground_marker", 3)
        )
    else:
        kappa = 1.0
        c_inf = 0.01
        electrode_marker = 1
        concentration_marker = 3
        ground_marker = 3

    return {
        "kappa_vals": _as_list(kappa, n_species, "robin_kappa"),
        "c_inf_vals": _as_list(c_inf, n_species, "robin_c_inf"),
        "electrode_marker": int(electrode_marker),
        "concentration_marker": int(concentration_marker),
        "ground_marker": int(ground_marker),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model_scaling(
    *,
    params: Any,
    n_species: int,
    dt: float,
    t_end: float,
    D_vals: Sequence[float],
    c0_vals: Sequence[float],
    phi_applied: float,
    phi0: float,
    robin: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Convert physical solver inputs to model-space values for PDE assembly.

    This function is the single authoritative place where nondimensionalization
    is applied.  It reads the ``solver_params[10]["nondim"]`` config block and
    returns a ``ModelScaling`` dict used by :func:`Forward.robin_solver.build_forms`.

    Parameters
    ----------
    params:
        ``solver_params[10]`` (the options dict).
    n_species:
        Number of ionic species.
    dt, t_end:
        Physical or model-space time parameters.
    D_vals:
        Per-species diffusivities.
    c0_vals:
        Per-species bulk concentrations.
    phi_applied:
        Applied boundary voltage.
    phi0:
        Reference potential (echoed into context for inspection).
    robin:
        Pre-parsed Robin settings dict (keys: kappa_vals, c_inf_vals, markers).
        If None, parsed from ``params["robin_bc"]``.

    Returns
    -------
    dict
        ModelScaling dict with keys: enabled, temperature_k, thermal_voltage_v,
        potential_scale_v, length_scale_m, concentration_scale_mol_m3,
        diffusivity_scale_m2_s, time_scale_s, kappa_scale_m_s, permittivity_f_m,
        electromigration_prefactor, poisson_coefficient, charge_rhs_prefactor,
        dt_model, t_end_model, D_model_vals, c0_model_vals, c_inf_model_vals,
        kappa_model_vals, phi_applied_model, phi0_model,
        flux_scale_mol_m2_s, current_density_scale_a_m2,
        debye_length_m, debye_to_length_ratio.
    """
    nondim_cfg = _get_nondim_cfg(params)
    enabled = _bool(nondim_cfg.get("enabled", False))

    if robin is None:
        robin = _get_robin_cfg(params, n_species)

    temperature_k = _pos(
        nondim_cfg.get("temperature_k", DEFAULT_TEMPERATURE_K), "nondim.temperature_k"
    )
    thermal_voltage_v = (GAS_CONSTANT * temperature_k) / FARADAY_CONSTANT

    D_raw = _as_list(D_vals, n_species, "D_vals")
    if any(v <= 0.0 for v in D_raw):
        raise ValueError("All D_vals must be strictly positive.")

    c_inf_raw = _as_list(robin["c_inf_vals"], n_species, "robin_c_inf")
    kappa_raw = _as_list(robin["kappa_vals"], n_species, "robin_kappa")
    c0_raw = _as_list(c0_vals, n_species, "c0")

    # ------------------------------------------------------------------
    # Dimensional mode  (enabled=False)
    # ------------------------------------------------------------------
    if not enabled:
        # Permittivity for the Poisson equation:
        #   ε · Δφ  =  F · Σᵢ zᵢ cᵢ   (SI, see module docstring)
        # Previously this was hardcoded to 1.0 which is incorrect for physical
        # units.  We now read (or default) the permittivity from the config.
        permittivity_f_m = _pos(
            nondim_cfg.get(
                "permittivity_f_m",
                DEFAULT_RELATIVE_PERMITTIVITY_WATER * VACUUM_PERMITTIVITY_F_PER_M,
            ),
            "nondim.permittivity_f_m",
        )
        return {
            "enabled": False,
            "temperature_k": temperature_k,
            "thermal_voltage_v": thermal_voltage_v,
            "potential_scale_v": 1.0,
            "length_scale_m": 1.0,
            "concentration_scale_mol_m3": 1.0,
            "diffusivity_scale_m2_s": 1.0,
            "time_scale_s": 1.0,
            "kappa_scale_m_s": 1.0,
            "permittivity_f_m": permittivity_f_m,
            # Corrected: use actual permittivity (not 1.0).
            "electromigration_prefactor": FARADAY_CONSTANT / (GAS_CONSTANT * temperature_k),
            "poisson_coefficient": permittivity_f_m,
            "charge_rhs_prefactor": FARADAY_CONSTANT,
            "dt_model": _pos(dt, "dt"),
            "t_end_model": _pos(t_end, "t_end"),
            "D_model_vals": D_raw,
            "c0_model_vals": c0_raw,
            "c_inf_model_vals": c_inf_raw,
            "kappa_model_vals": kappa_raw,
            "phi_applied_model": float(phi_applied),
            "phi0_model": float(phi0),
            "flux_scale_mol_m2_s": 1.0,
            "current_density_scale_a_m2": 1.0,
            "debye_length_m": None,
            "debye_to_length_ratio": None,
        }

    # ------------------------------------------------------------------
    # Nondimensional mode  (enabled=True)
    # ------------------------------------------------------------------
    d_inputs_dimless = _bool(nondim_cfg.get("diffusivity_inputs_are_dimensionless", False))
    c_inputs_dimless = _bool(nondim_cfg.get("concentration_inputs_are_dimensionless", False))
    phi_inputs_dimless = _bool(nondim_cfg.get("potential_inputs_are_dimensionless", False))
    time_inputs_dimless = _bool(nondim_cfg.get("time_inputs_are_dimensionless", False))
    # Default True: the Robin kappa optimizer works in dimensionless model units.
    # Set to False when passing physical kappa values in m/s.
    kappa_inputs_dimless = _bool(nondim_cfg.get("kappa_inputs_are_dimensionless", True))

    # Reference scales (auto-computed from inputs if not provided)
    D_arr = np.asarray(D_raw, dtype=float)
    c_all = np.asarray(c0_raw + c_inf_raw, dtype=float)

    diffusivity_scale = _pos(
        nondim_cfg.get(
            "diffusivity_scale_m2_s",
            float(np.exp(np.mean(np.log(D_arr)))),  # geometric mean
        ),
        "nondim.diffusivity_scale_m2_s",
    )
    concentration_scale = _pos(
        nondim_cfg.get(
            "concentration_scale_mol_m3",
            max(1e-16, float(np.max(np.abs(c_all)))),
        ),
        "nondim.concentration_scale_mol_m3",
    )
    length_scale = _pos(
        nondim_cfg.get("length_scale_m", 1.0e-4), "nondim.length_scale_m"
    )
    potential_scale = _pos(
        nondim_cfg.get("potential_scale_v", thermal_voltage_v), "nondim.potential_scale_v"
    )
    permittivity_f_m = _pos(
        nondim_cfg.get(
            "permittivity_f_m",
            DEFAULT_RELATIVE_PERMITTIVITY_WATER * VACUUM_PERMITTIVITY_F_PER_M,
        ),
        "nondim.permittivity_f_m",
    )
    time_scale = _pos(
        nondim_cfg.get(
            "time_scale_s", (length_scale * length_scale) / diffusivity_scale
        ),
        "nondim.time_scale_s",
    )
    kappa_scale = _pos(
        nondim_cfg.get("kappa_scale_m_s", diffusivity_scale / length_scale),
        "nondim.kappa_scale_m_s",
    )

    # Apply scaling to each input type
    D_model = D_raw if d_inputs_dimless else [v / diffusivity_scale for v in D_raw]
    if any(v <= 0.0 for v in D_model):
        raise ValueError("All model-space diffusivities must be strictly positive.")

    c0_model = c0_raw if c_inputs_dimless else [v / concentration_scale for v in c0_raw]
    c_inf_model = c_inf_raw if c_inputs_dimless else [v / concentration_scale for v in c_inf_raw]

    dt_raw = _pos(dt, "dt")
    t_end_raw = _pos(t_end, "t_end")
    dt_model = dt_raw if time_inputs_dimless else dt_raw / time_scale
    t_end_model = t_end_raw if time_inputs_dimless else t_end_raw / time_scale
    if dt_model <= 0.0 or t_end_model <= 0.0:
        raise ValueError("dt and t_end must map to strictly positive model-space values.")

    phi_applied_raw = float(phi_applied)
    phi0_raw = float(phi0)
    phi_applied_model = phi_applied_raw if phi_inputs_dimless else phi_applied_raw / potential_scale
    phi0_model = phi0_raw if phi_inputs_dimless else phi0_raw / potential_scale

    kappa_model = kappa_raw if kappa_inputs_dimless else [v / kappa_scale for v in kappa_raw]
    if any(v < 0.0 for v in kappa_model):
        raise ValueError("Robin kappa values must be nonneg in model space.")

    # PDE coefficients — see module docstring for derivation
    #
    # Nernst-Planck electromigration factor:
    #   drift = (F/RT) · V_T · z · φ̂  →  prefactor = (F/RT) · V_T
    #   When potential_scale = V_T this equals 1.0 exactly.
    electromigration_prefactor = (FARADAY_CONSTANT / (GAS_CONSTANT * temperature_k)) * potential_scale

    # Nondimensional Poisson equation:
    #   (λ_D/L)² · Δφ̂ = −Σᵢ zᵢ ĉᵢ  →  poisson_coefficient = (λ_D/L)²
    #   equivalently: ε·V_T/(F·c_ref·L²)
    poisson_coefficient = (
        permittivity_f_m * potential_scale
    ) / (FARADAY_CONSTANT * concentration_scale * (length_scale * length_scale))

    debye_length_m = float(
        np.sqrt(
            (permittivity_f_m * potential_scale) / (FARADAY_CONSTANT * concentration_scale)
        )
    )

    flux_scale = diffusivity_scale * concentration_scale / length_scale
    current_density_scale = FARADAY_CONSTANT * flux_scale

    scaling = {
        "enabled": True,
        "temperature_k": temperature_k,
        "thermal_voltage_v": thermal_voltage_v,
        "potential_scale_v": potential_scale,
        "length_scale_m": length_scale,
        "concentration_scale_mol_m3": concentration_scale,
        "diffusivity_scale_m2_s": diffusivity_scale,
        "time_scale_s": time_scale,
        "kappa_scale_m_s": kappa_scale,
        "permittivity_f_m": permittivity_f_m,
        "electromigration_prefactor": electromigration_prefactor,
        "poisson_coefficient": poisson_coefficient,
        "charge_rhs_prefactor": 1.0,
        "dt_model": dt_model,
        "t_end_model": t_end_model,
        "D_model_vals": D_model,
        "c0_model_vals": c0_model,
        "c_inf_model_vals": c_inf_model,
        "kappa_model_vals": kappa_model,
        "phi_applied_model": phi_applied_model,
        "phi0_model": phi0_model,
        "flux_scale_mol_m2_s": flux_scale,
        "current_density_scale_a_m2": current_density_scale,
        "debye_length_m": debye_length_m,
        "debye_to_length_ratio": debye_length_m / length_scale,
    }

    verify_model_params(scaling, n_species=n_species)
    return scaling


def verify_model_params(scaling: dict, *, n_species: int = 2) -> None:
    """Warn if model-space parameters look suspicious.

    Checks are heuristic, not exhaustive.  A warning is printed (not raised)
    so the solver can still attempt a solve.
    """
    if not scaling.get("enabled", False):
        return

    D_model = scaling.get("D_model_vals", [])
    for i, d in enumerate(D_model):
        if not (1e-8 <= float(d) <= 1e4):
            print(
                f"[nondim warn] D_model[{i}]={float(d):.3e} is outside [1e-8, 1e4]. "
                "Check that diffusivity_scale_m2_s matches your physical inputs."
            )

    c0_model = scaling.get("c0_model_vals", [])
    for i, c in enumerate(c0_model):
        if float(c) < 0.0:
            print(f"[nondim warn] c0_model[{i}]={float(c):.3e} is negative.")
        elif float(c) > 1e3:
            print(
                f"[nondim warn] c0_model[{i}]={float(c):.3e} is very large. "
                "Check concentration_scale_mol_m3."
            )

    ratio = scaling.get("debye_to_length_ratio")
    if ratio is not None and (float(ratio) < 1e-5 or float(ratio) > 1.0):
        print(
            f"[nondim warn] Debye-to-length ratio={float(ratio):.3e}. "
            "Very small ratio means the double layer is poorly resolved; "
            "very large means the domain is sub-Debye-length scale."
        )
