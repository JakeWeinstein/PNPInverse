"""BV configuration helpers for parsing solver_params."""

from __future__ import annotations

from typing import Any

from Nondim.transform import _as_list, _bool


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
    alpha_val = float(alpha) if not isinstance(alpha, (list, tuple)) else None
    if alpha_val is not None and not (0.0 < alpha_val <= 1.0):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha_val}")
    stoichiometry = raw.get("stoichiometry", [-1] * n_species)
    c_ref = raw.get("c_ref", raw.get("c_inf", 1.0))
    E_eq_v = float(raw.get("E_eq_v", 0.0))
    # Defaults match RectangleMesh convention: 3=bottom (electrode), 4=top (bulk).
    # For IntervalMesh, override to electrode_marker=1, concentration_marker=2, ground_marker=2.
    electrode_marker = int(raw.get("electrode_marker", 3))
    concentration_marker = int(raw.get("concentration_marker", 4))
    ground_marker = int(raw.get("ground_marker", 4))

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
        return {
            "clip_exponent":            True,
            "exponent_clip":            50.0,
            "regularize_concentration": True,
            "conc_floor":               1e-8,
            "use_eta_in_bv":            True,
            "packing_floor":            1e-8,
            "softplus_regularization":  False,
        }
    raw = params.get("bv_convergence", {})
    if not isinstance(raw, dict):
        return {
            "clip_exponent":            True,
            "exponent_clip":            50.0,
            "regularize_concentration": True,
            "conc_floor":               1e-8,
            "use_eta_in_bv":            True,
            "packing_floor":            1e-8,
            "softplus_regularization":  False,
        }
    exponent_clip = float(raw.get("exponent_clip", 50.0))
    conc_floor = float(raw.get("conc_floor", 1e-8))
    packing_floor = float(raw.get("packing_floor", 1e-8))
    if exponent_clip <= 0:
        raise ValueError(f"exponent_clip must be positive; got {exponent_clip}")
    if conc_floor <= 0:
        raise ValueError(f"conc_floor must be positive; got {conc_floor}")
    if packing_floor <= 0:
        raise ValueError(f"packing_floor must be positive; got {packing_floor}")
    return {
        "clip_exponent":              _bool(raw.get("clip_exponent", True)),
        "exponent_clip":              exponent_clip,
        "regularize_concentration":   _bool(raw.get("regularize_concentration", True)),
        "conc_floor":                 conc_floor,
        "use_eta_in_bv":              _bool(raw.get("use_eta_in_bv", True)),
        "packing_floor":              packing_floor,
        "softplus_regularization":    _bool(raw.get("softplus_regularization", False)),
    }


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

        alpha_val = float(rxn.get("alpha", 0.5))
        if not (0.0 < alpha_val <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha_val}")
        n_e = int(rxn.get("n_electrons", 2))
        if n_e <= 0:
            raise ValueError(f"Reaction {j}: n_electrons must be positive; got {n_e}")
        reactions.append({
            "k0": float(rxn.get("k0", 1e-5)),
            "alpha": alpha_val,
            "cathodic_species": int(cat),
            "anodic_species": int(anod) if anod is not None else None,
            "c_ref": float(rxn.get("c_ref", 1.0)),
            "stoichiometry": [int(s) for s in stoi],
            "n_electrons": n_e,
            "reversible": _bool(rxn.get("reversible", True)),
            "cathodic_conc_factors": cat_conc_factors,
        })
    return reactions
