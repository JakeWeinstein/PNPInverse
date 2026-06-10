"""BV configuration helpers for parsing solver_params."""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

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
    if alpha_val is None:
        for k, a in enumerate(alpha):
            try:
                a_f = float(a)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"bv_bc.alpha[{k}] must be a float; got {a!r}"
                ) from exc
            if not (0.0 < a_f <= 1.0):
                raise ValueError(
                    f"bv_bc.alpha[{k}] must be in (0, 1]; got {a_f}"
                )
    stoichiometry = raw.get("stoichiometry", [-1] * n_species)
    c_ref = raw.get("c_ref", raw.get("c_inf", 1.0))
    E_eq_v = float(raw.get("E_eq_v", 0.0))
    # Defaults match RectangleMesh convention: 3=bottom (electrode), 4=top (bulk).
    # For IntervalMesh, override to electrode_marker=1, concentration_marker=2, ground_marker=2.
    electrode_marker = int(raw.get("electrode_marker", 3))
    concentration_marker = int(raw.get("concentration_marker", 4))
    ground_marker = int(raw.get("ground_marker", 4))
    # Stern layer capacitance [F/m²].  When set (> 0), enables the Frumkin-
    # corrected overpotential model:
    #   eta = phi_m - phi_s - E_eq     (matches LaTeX formulation)
    # where phi_m = phi_applied (metal potential) and phi_s = solution potential
    # at the electrode surface (free to float, determined by the Poisson
    # equation + Stern layer Robin BC).  The Dirichlet BC for phi at the
    # electrode is replaced with:
    #   epsilon * grad(phi) . n = C_stern * (phi_m - phi)
    # When None or 0 (default), the classical model is used:
    #   eta = phi_applied - E_eq       (phi_s = phi_applied via Dirichlet BC)
    stern_raw = raw.get("stern_capacitance_f_m2", None)
    stern_capacitance = float(stern_raw) if stern_raw is not None else None
    if stern_capacitance is not None and stern_capacitance < 0:
        raise ValueError(
            f"stern_capacitance_f_m2 must be non-negative; got {stern_capacitance}"
        )

    return {
        "k0_vals":            _as_list(k0, n_species, "bv_bc.k0"),
        "alpha_vals":         _as_list(alpha, n_species, "bv_bc.alpha"),
        "stoichiometry":      [int(s) for s in _as_list(stoichiometry, n_species, "bv_bc.stoichiometry")],
        "c_ref_vals":         _as_list(c_ref, n_species, "bv_bc.c_ref"),
        "E_eq_v":             E_eq_v,
        "electrode_marker":   electrode_marker,
        "concentration_marker": concentration_marker,
        "ground_marker":      ground_marker,
        "stern_capacitance_f_m2": stern_capacitance,
    }


_VALID_FORMULATIONS = ("concentration", "logc", "logc_muh")
_VALID_INITIALIZERS = ("linear_phi", "debye_boltzmann")


def _validate_formulation(value: Any) -> str:
    """Coerce/validate the formulation config field.

    The ``"concentration"`` backend was removed in the May 2026 cleanup;
    the dispatcher silently falls through to ``"logc"`` for that value to
    keep old configs parseable.  Emit a deprecation warning so callers
    know they are getting the log-c backend rather than the named one.
    """
    if value is None:
        return "logc"
    s = str(value).strip().lower()
    if s not in _VALID_FORMULATIONS:
        raise ValueError(
            f"bv_convergence.formulation must be one of {_VALID_FORMULATIONS}; got {value!r}"
        )
    if s == "concentration":
        warnings.warn(
            "bv_convergence.formulation='concentration' refers to a backend "
            "removed in the May 2026 cleanup; the dispatcher will use 'logc' "
            "instead.  Update the config to 'logc' or 'logc_muh' explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
    return s


def _validate_initializer(value: Any) -> str:
    """Coerce/validate the initializer config field."""
    if value is None:
        return "linear_phi"
    s = str(value).strip().lower()
    if s not in _VALID_INITIALIZERS:
        raise ValueError(
            f"bv_convergence.initializer must be one of {_VALID_INITIALIZERS}; got {value!r}"
        )
    return s


def _default_bv_convergence_cfg() -> dict:
    """Default BV convergence config (used when params is missing the block).

    ``exponent_clip = 100.0`` is the only PC-trustworthy default.  At
    clip=50 (production until 2026-05-04) the peroxide-current observable
    is fictitious below V_RHE = -0.1 V (sign-flipped, 3-4 OOM off; CD is
    approximately correct).  At clip=100 the production V grid
    V_RHE in [-0.5, +1.0] V is unclipped for both reactions, exposing
    true mass-transport-limited PC.  Do not lower this below 100 for
    forward runs whose PC will be compared against experiment or used
    for inverse fitting.  Trade-off: cold-start Newton basin shrinks
    (10/13 -> 3/13 voltages on a typical sweep), but C+D warm-walk
    continuation rescues every voltage.  See
    ``docs/clip_observable_investigation.md`` §5.2 and
    ``docs/clipping_conventions.md`` for the operational rule.
    """
    return {
        "clip_exponent":            True,
        "exponent_clip":            100.0,
        "regularize_concentration": True,
        "conc_floor":               1e-8,
        "use_eta_in_bv":            True,
        "packing_floor":            1e-8,
        "softplus_regularization":  False,
        "bv_log_rate":              False,
        "u_clamp":                  100.0,
        "formulation":              "logc",
        "initializer":              "linear_phi",
        # Phase 6β step 6 plumbing-ablation defaults (preserve
        # byte-equivalence with v9/v10a/v10a'/A.2).
        "apply_h_source":           True,
        "apply_k_sink":             True,
        "override_sigma_singh_counts_pm2": None,
    }


def _get_bv_convergence_cfg(params: Any) -> dict:
    """Parse optional BV convergence-strategy settings."""
    if not isinstance(params, dict):
        return _default_bv_convergence_cfg()
    raw = params.get("bv_convergence", {})
    if not isinstance(raw, dict):
        return _default_bv_convergence_cfg()
    exponent_clip = float(raw.get("exponent_clip", 100.0))
    conc_floor = float(raw.get("conc_floor", 1e-8))
    packing_floor = float(raw.get("packing_floor", 1e-8))
    u_clamp = float(raw.get("u_clamp", 100.0))
    formulation = _validate_formulation(raw.get("formulation", "logc"))
    initializer = _validate_initializer(raw.get("initializer", "linear_phi"))
    if exponent_clip <= 0:
        raise ValueError(f"exponent_clip must be positive; got {exponent_clip}")
    if conc_floor <= 0:
        raise ValueError(f"conc_floor must be positive; got {conc_floor}")
    if packing_floor <= 0:
        raise ValueError(f"packing_floor must be positive; got {packing_floor}")
    if u_clamp <= 0:
        raise ValueError(f"u_clamp must be positive; got {u_clamp}")

    # Phase 6α — water self-ionization keys.  Default to a disabled
    # configuration (enable_water_ionization=False, zero kw/d_oh/a_oh)
    # so callers that never set these keys are byte-equivalent to the
    # pre-Phase-6α stack.  Validated when enable_water_ionization is on.
    enable_water_ionization = _bool(raw.get("enable_water_ionization", False))
    kw_eff_hat = float(raw.get("kw_eff_hat", 0.0))
    d_oh_hat = float(raw.get("d_oh_hat", 0.0))
    a_oh_hat = float(raw.get("a_oh_hat", 0.0))
    if enable_water_ionization:
        if kw_eff_hat < 0.0:
            raise ValueError(
                "kw_eff_hat must be non-negative when "
                f"enable_water_ionization=True; got {kw_eff_hat}"
            )
        if d_oh_hat <= 0.0:
            raise ValueError(
                "d_oh_hat must be positive when "
                f"enable_water_ionization=True; got {d_oh_hat}"
            )
        if a_oh_hat < 0.0:
            raise ValueError(
                "a_oh_hat must be non-negative when "
                f"enable_water_ionization=True; got {a_oh_hat}"
            )

    # Phase 6β v9 Gate 3 — cation hydrolysis keys.  The flag gates the
    # mixed-space Γ slot (Gate 3A) and the cation_hydrolysis bundle
    # (Gate 3B).  ``cation_hydrolysis_config`` carries the per-cation
    # Singh parameters + finite-rate kinetics (filled in by Gate 4A);
    # at Gate 3 the flag is enough to extend the mixed space.
    enable_cation_hydrolysis = _bool(raw.get("enable_cation_hydrolysis", False))
    cation_hydrolysis_config = raw.get("cation_hydrolysis_config", None)
    if enable_cation_hydrolysis:
        if cation_hydrolysis_config is not None and not isinstance(
            cation_hydrolysis_config, dict
        ):
            raise ValueError(
                "cation_hydrolysis_config must be a dict (or None) when "
                f"enable_cation_hydrolysis=True; got "
                f"{type(cation_hydrolysis_config).__name__}"
            )
    # Manufactured-source override for Gate 3D unit tests: when set,
    # the form replaces R_net with this Constant value.  ``None`` (the
    # default) leaves the production R_net wired up.  Surfaced via the
    # parser so the form-build code can read it from conv_cfg.
    manufactured_R_inj_raw = raw.get("manufactured_R_inj", None)
    manufactured_R_inj = (
        None if manufactured_R_inj_raw is None
        else float(manufactured_R_inj_raw)
    )

    # Phase 6β step 6 — plumbing-ablation flags.  Defaults preserve
    # byte-equivalence with v9/v10a/v10a'/A.2.
    #
    #   apply_h_source : gates the cation-hydrolysis proton residual
    #       term.  Valid only with manufactured_R_inj is not None.
    #   apply_k_sink : gates the cation-hydrolysis K+ residual term.
    #       Valid only with manufactured_R_inj is not None.
    #   override_sigma_singh_counts_pm2 : when set (non-negative finite
    #       scalar in Singh counts/pm² units), replaces σ_S in
    #       build_pka_shift with a fake-signed σ that, after the
    #       existing anode clamp + 6.2415e-6 conversion, equals the
    #       override.  Valid only with manufactured_R_inj is None.
    apply_h_source = _bool(raw.get("apply_h_source", True))
    apply_k_sink = _bool(raw.get("apply_k_sink", True))
    override_raw = raw.get("override_sigma_singh_counts_pm2", None)
    if override_raw is None:
        override_sigma_singh_counts_pm2 = None
    else:
        override_sigma_singh_counts_pm2 = float(override_raw)
        import math
        if (
            not math.isfinite(override_sigma_singh_counts_pm2)
            or override_sigma_singh_counts_pm2 < 0.0
        ):
            raise ValueError(
                "override_sigma_singh_counts_pm2 must be a finite "
                "non-negative scalar (post-clamp σ_singh is "
                f"non-negative by convention); got {override_raw!r}"
            )
    half_physical = (not apply_h_source) or (not apply_k_sink)
    if half_physical and manufactured_R_inj is None:
        raise ValueError(
            "apply_h_source=False or apply_k_sink=False requires "
            "manufactured_R_inj to be set; half-physical ablations "
            "would otherwise give Picard a physically inconsistent "
            "Γ update.  The manufactured path's closed-form Γ_ss "
            "is well-defined regardless."
        )
    if (
        override_sigma_singh_counts_pm2 is not None
        and manufactured_R_inj is not None
    ):
        raise ValueError(
            "override_sigma_singh_counts_pm2 is a physical-path "
            "imposition; manufactured_R_inj bypasses the physical "
            "path entirely.  Set at most one of the two."
        )

    return {
        "clip_exponent":              _bool(raw.get("clip_exponent", True)),
        "exponent_clip":              exponent_clip,
        "regularize_concentration":   _bool(raw.get("regularize_concentration", True)),
        "conc_floor":                 conc_floor,
        "use_eta_in_bv":              _bool(raw.get("use_eta_in_bv", True)),
        "packing_floor":              packing_floor,
        "softplus_regularization":    _bool(raw.get("softplus_regularization", False)),
        "bv_log_rate":                _bool(raw.get("bv_log_rate", False)),
        "u_clamp":                    u_clamp,
        "formulation":                formulation,
        "initializer":                initializer,
        "enable_water_ionization":    enable_water_ionization,
        "kw_eff_hat":                 kw_eff_hat,
        "d_oh_hat":                   d_oh_hat,
        "a_oh_hat":                   a_oh_hat,
        "enable_cation_hydrolysis":   enable_cation_hydrolysis,
        "cation_hydrolysis_config":   cation_hydrolysis_config,
        # Phase 6β v9 Gate 3C — λ continuation knob; ramps cation
        # hydrolysis source from 0 → 1.  Default 0.0 so a freshly-
        # built form with the Γ slot but no λ override is the same
        # as the disabled-feature baseline (Γ pinned to 0).
        "lambda_hydrolysis":          float(raw.get("lambda_hydrolysis", 0.0)),
        # Phase 6β v9 Gate 3D — manufactured-source override for
        # unit tests.  ``None`` (default) leaves the production
        # R_net wired up.
        "manufactured_R_inj":         manufactured_R_inj,
        # Phase 6β step 6 plumbing-ablation flags (see prose above).
        "apply_h_source":             apply_h_source,
        "apply_k_sink":               apply_k_sink,
        "override_sigma_singh_counts_pm2": override_sigma_singh_counts_pm2,
    }


# ---------------------------------------------------------------------------
# Phase 6β v9 Gate 1 — role-aware species lookup
# ---------------------------------------------------------------------------

def _get_species_roles(params: Any, n_species: int) -> Optional[List[str]]:
    """Read optional per-species role labels from ``bv_bc.species_roles``.

    Phase 6β v9 Gate 1.  Returns the explicit roles list when set on the
    config, or ``None`` to signal callers should fall back to legacy
    z-inference.  Validates length matches ``n_species`` so a stale config
    fails LOUDLY at form-build rather than silently mis-wiring the proton
    index.
    """
    if not isinstance(params, dict):
        return None
    bv_raw = params.get("bv_bc", {})
    if not isinstance(bv_raw, dict):
        return None
    roles_raw = bv_raw.get("species_roles", None)
    if roles_raw is None:
        return None
    if not isinstance(roles_raw, (list, tuple)):
        raise ValueError(
            "bv_bc.species_roles must be a list of strings or None; "
            f"got {type(roles_raw).__name__}"
        )
    roles = [str(r) for r in roles_raw]
    if len(roles) != n_species:
        raise ValueError(
            f"bv_bc.species_roles length {len(roles)} does not match "
            f"n_species {n_species}"
        )
    return roles


# ---------------------------------------------------------------------------
# Multi-reaction BV config helpers
# ---------------------------------------------------------------------------

def _get_bv_boltzmann_counterions_cfg(params: Any) -> list[dict]:
    """Parse ``bv_bc.boltzmann_counterions`` — analytic Boltzmann ions in Poisson.

    Each entry represents an inert ion that is treated as locally
    equilibrated with the electrostatic potential (Poisson--Boltzmann
    reduction) instead of being solved as a dynamic Nernst--Planck
    unknown.  The ion contributes an additional charge density term
    ``z * c_bulk * exp(-z * phi)`` (in nondimensional units, sign
    consistent with the Poisson convention used by the forms modules)
    to the right-hand side of Poisson's equation.

    The exponent is symmetrically clamped at ``|phi| <= phi_clamp`` to
    prevent floating-point overflow during Newton iteration.

    Each list entry is a dict with keys:
        z              (int)           — counterion charge number (e.g. -1).
        c_bulk_nondim  (float)         — bulk concentration in nondim units.
        phi_clamp      (float, optional, default 50.0) — exponent clamp.
        steric_mode    (str, optional, default ``"ideal"``) — one of
                       ``"ideal"`` (unbounded ``c_b * exp(-z*phi)``) or
                       ``"bikerman"`` (steric-aware closure that respects
                       the Bikerman packing fraction; see
                       ``docs/steric_analytic_clo4_reduction_handoff.md``).
        a_nondim       (float, required when ``steric_mode='bikerman'``)
                       — the counterion's Bikerman steric size (nondim).
                       Ignored when ``steric_mode='ideal'``.

    The ``theta_b > 0`` precondition for the bikerman closure is NOT
    validated here (this parser does not see the dynamic-species
    ``a_vals`` or ``c0`` needed to compute it); that validation lives
    in the helper that constructs the closure expression.

    Returns ``[]`` when not configured (no Boltzmann counterions).
    """
    if not isinstance(params, dict):
        return []
    bv_raw = params.get("bv_bc", {})
    if not isinstance(bv_raw, dict):
        return []
    raw = bv_raw.get("boltzmann_counterions", [])
    if raw in (None, [], ()):
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "bv_bc.boltzmann_counterions must be a list of dicts."
        )
    out: list[dict] = []
    for j, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"bv_bc.boltzmann_counterions[{j}] must be a dict."
            )
        if "z" not in entry:
            raise ValueError(
                f"bv_bc.boltzmann_counterions[{j}] missing required key 'z'."
            )
        if "c_bulk_nondim" not in entry:
            raise ValueError(
                f"bv_bc.boltzmann_counterions[{j}] missing required key 'c_bulk_nondim'."
            )
        z_val = int(entry["z"])
        c_bulk = float(entry["c_bulk_nondim"])
        phi_clamp = float(entry.get("phi_clamp", 50.0))
        if c_bulk < 0:
            raise ValueError(
                f"boltzmann_counterions[{j}].c_bulk_nondim must be non-negative; got {c_bulk}"
            )
        if phi_clamp <= 0:
            raise ValueError(
                f"boltzmann_counterions[{j}].phi_clamp must be positive; got {phi_clamp}"
            )
        steric_mode = str(entry.get("steric_mode", "ideal")).strip().lower()
        if steric_mode not in ("ideal", "bikerman"):
            raise ValueError(
                f"boltzmann_counterions[{j}].steric_mode must be 'ideal' or "
                f"'bikerman'; got {steric_mode!r}"
            )
        a_nondim_raw = entry.get("a_nondim", None)
        if steric_mode == "bikerman":
            if a_nondim_raw is None:
                raise ValueError(
                    f"boltzmann_counterions[{j}] with steric_mode='bikerman' "
                    f"requires 'a_nondim'."
                )
            a_nondim = float(a_nondim_raw)
            if a_nondim <= 0:
                raise ValueError(
                    f"boltzmann_counterions[{j}].a_nondim must be positive; "
                    f"got {a_nondim}"
                )
        else:
            a_nondim = float(a_nondim_raw) if a_nondim_raw is not None else None
        out.append({
            "z": z_val,
            "c_bulk_nondim": c_bulk,
            "phi_clamp": phi_clamp,
            "steric_mode": steric_mode,
            "a_nondim": a_nondim,
        })
    return out


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
        cat_idx = int(cat)
        if cat_idx < 0 or cat_idx >= n_species:
            raise ValueError(
                f"Reaction {j}: cathodic_species {cat_idx} out of range "
                f"[0, {n_species})"
            )
        anod_idx: int | None
        if anod is None:
            anod_idx = None
        else:
            anod_idx = int(anod)
            if anod_idx < 0 or anod_idx >= n_species:
                raise ValueError(
                    f"Reaction {j}: anodic_species {anod_idx} out of range "
                    f"[0, {n_species})"
                )
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

        reversible = _bool(rxn.get("reversible", True))

        # Phase 7 (v11): water-as-proton-donor route marker.  "water"
        # selects the alkaline-route rate law shape: no H+ concentration
        # factor (water activity == 1) and cathodic-only (the anodic
        # branch has no product-activity support, so a reversible water
        # route would be thermodynamically inconsistent).  The
        # stoichiometry is unchanged: with the proton-condition variable
        # E = c_H - c_OH, producing m OH- is algebraically identical to
        # consuming m H+ (both lower E by m per event).
        proton_donor = str(rxn.get("proton_donor", "hydronium")).strip().lower()
        if proton_donor not in ("hydronium", "water"):
            raise ValueError(
                f"Reaction {j}: proton_donor must be 'hydronium' or "
                f"'water'; got {rxn.get('proton_donor')!r}"
            )
        if proton_donor == "water":
            if cat_conc_factors:
                raise ValueError(
                    f"Reaction {j}: proton_donor='water' forbids "
                    "cathodic_conc_factors (water activity == 1 in v11)"
                )
            if reversible:
                raise ValueError(
                    f"Reaction {j}: proton_donor='water' requires "
                    "reversible=False (no product-activity support in the "
                    "anodic branch)"
                )

        # Role flags for index-stable observable resolution.  Validated
        # against the stoichiometry so a mislabeled flag cannot silently
        # corrupt the peroxide observable.  h2o2_species defaults to 1
        # (production-stack convention: species = [O2, H2O2, H+]).
        label = str(rxn.get("label", f"reaction_{j}"))
        produces_h2o2 = _bool(rxn.get("produces_h2o2", False))
        consumes_h2o2 = _bool(rxn.get("consumes_h2o2", False))
        if produces_h2o2 and consumes_h2o2:
            raise ValueError(
                f"Reaction {j} ('{label}'): produces_h2o2 and "
                "consumes_h2o2 cannot both be True"
            )
        h2o2_idx = int(rxn.get("h2o2_species", 1))
        if produces_h2o2 or consumes_h2o2:
            if h2o2_idx < 0 or h2o2_idx >= n_species:
                raise ValueError(
                    f"Reaction {j} ('{label}'): h2o2_species {h2o2_idx} "
                    f"out of range [0, {n_species})"
                )
            if produces_h2o2 and int(stoi[h2o2_idx]) <= 0:
                raise ValueError(
                    f"Reaction {j} ('{label}'): produces_h2o2=True but "
                    f"stoichiometry[{h2o2_idx}] = {stoi[h2o2_idx]} is not > 0"
                )
            if consumes_h2o2 and int(stoi[h2o2_idx]) >= 0:
                raise ValueError(
                    f"Reaction {j} ('{label}'): consumes_h2o2=True but "
                    f"stoichiometry[{h2o2_idx}] = {stoi[h2o2_idx]} is not < 0"
                )

        reactions.append({
            "k0": float(rxn.get("k0", 1e-5)),
            "alpha": alpha_val,
            "cathodic_species": cat_idx,
            "anodic_species": anod_idx,
            "c_ref": float(rxn.get("c_ref", 1.0)),
            "stoichiometry": [int(s) for s in stoi],
            "n_electrons": n_e,
            "reversible": reversible,
            "cathodic_conc_factors": cat_conc_factors,
            "E_eq_v": float(rxn.get("E_eq_v", 0.0)),
            "proton_donor": proton_donor,
            "label": label,
            "produces_h2o2": produces_h2o2,
            "consumes_h2o2": consumes_h2o2,
        })
    return reactions


def _validate_reactions_vs_convergence(
    reactions_cfg: list[dict] | None, conv_cfg: dict | None
) -> None:
    """Cross-config validation shared by BOTH form backends.

    Lives here (NOT only in dispatch) because ``anchor_continuation``
    and several tests call ``build_forms_logc`` /
    ``build_forms_logc_muh`` directly — every entry path must hit this
    check.

    Rule: an ACTIVE water route (``proton_donor='water'`` and k0 > 0)
    requires ``enable_water_ionization=True``.  Its stoichiometry sinks
    proton-equivalents that only the proton-condition variable
    E = c_H − c_OH can source from water; without it the same
    stoichiometry would wrongly drain the finite c_H pool.  Inactive
    (k0 == 0) water entries are allowed so ablation/provenance configs
    do not force the Kw closure on.
    """
    if not reactions_cfg:
        return
    water_on = bool((conv_cfg or {}).get("enable_water_ionization", False))
    for j, rxn in enumerate(reactions_cfg):
        if (
            rxn.get("proton_donor", "hydronium") == "water"
            and float(rxn.get("k0", 0.0)) > 0.0
            and not water_on
        ):
            raise ValueError(
                f"Reaction {j} ('{rxn.get('label', f'reaction_{j}')}'): "
                "active water-route reaction (proton_donor='water', k0>0) "
                "requires enable_water_ionization=True in bv_convergence; "
                "enable it or disable the reaction (k0=0)."
            )
