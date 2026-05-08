"""Observable form builders for BV-boundary current-density computation.

Moved from FluxCurve/bv_observables.py so that Forward-layer modules
(e.g. grid_charge_continuation) can use observable-based convergence
without importing from FluxCurve.

Electron-weighted accounting (M3a.1, 2026-05-07)
------------------------------------------------
Total disk current and selectivity must respect per-reaction
``n_electrons`` once the reaction list contains heterogeneous electron
counts (e.g. parallel 2e + 4e ORR per Ruggiero 2022).  ``current_density``
mode therefore weights each reaction by ``n_e_j / N_ELECTRONS_REF`` where
``N_ELECTRONS_REF = 2`` matches the reference baked into ``I_SCALE`` in
``scripts/_bv_common.compute_i_scale``.  For a uniform-2e reaction list
this reduces to the legacy ``Σ R_j`` form to floating-point tolerance.

Per-reaction ``n_electrons`` is read from
``ctx["nondim"]["bv_reactions"][j]["n_electrons"]`` (populated by
``_add_bv_reactions_scaling_to_transform`` in ``nondim.py``).  When the
reactions list is unavailable (legacy per-species path) the form falls
back to the unweighted sum and emits no warning, matching pre-M3a.1
behavior.

The ``peroxide_current`` mode (``R_0 - R_1``) is retained for
back-compat with v13/v15/v16 inverse scripts but is **deprecated** for
new work: per the post-Ruggiero analysis (see
``docs/Ruggiero2022_JCatal_source_paper.md``) the deck's "Peroxide
Current Density" maps to the gross 2e production current, which is the
single-reaction R_2e rate, not the net difference R_0 - R_1.  Use
``gross_h2o2_current`` (or equivalently ``mode="reaction",
reaction_index=R_2e_idx``) for new comparisons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Reference electron count: ``I_SCALE`` in ``scripts/_bv_common.compute_i_scale``
# is built around ``n_electrons = 2``.  ``current_density`` mode weights
# each reaction by ``n_e_j / N_ELECTRONS_REF`` so the I_SCALE multiplier
# (applied externally via the ``scale`` argument) yields physical units.
N_ELECTRONS_REF = 2


def _get_reaction_n_electrons(ctx: Dict[str, Any]) -> Optional[List[float]]:
    """Return per-reaction ``n_electrons`` from ctx, or None if unavailable.

    Reads ``ctx["nondim"]["bv_reactions"]`` (the scaled reactions list
    populated by ``_add_bv_reactions_scaling_to_transform``).  Returns
    None when the list is missing (legacy per-species BV path or
    incomplete ctx) so callers can fall back to the unweighted form.
    """
    nondim = ctx.get("nondim", {})
    reactions = nondim.get("bv_reactions") if isinstance(nondim, dict) else None
    if not reactions:
        return None
    out: List[float] = []
    for rxn in reactions:
        try:
            out.append(float(rxn["n_electrons"]))
        except (KeyError, TypeError, ValueError):
            return None
    return out


def _build_bv_observable_form(
    ctx: Dict[str, object],
    *,
    mode: str,
    reaction_index: Optional[int],
    scale: float,
) -> object:
    """Build scalar UFL observable from BV reaction rate expressions.

    The observable is current density = scale * (combination of R_j) * ds(electrode).

    Supported modes
    ---------------
    - ``"current_density"``: total faradaic current respecting per-reaction
      electron count: ``scale * Σ_j (n_e_j / N_ELECTRONS_REF) * R_j * ds``.
      For uniform-2e reaction lists this reduces to the legacy
      ``scale * Σ_j R_j * ds`` to floating-point tolerance.  When
      per-reaction ``n_electrons`` is unavailable (legacy ctx without the
      scaled-reactions list) falls back to the unweighted sum.
    - ``"gross_h2o2_current"``: gross 2e peroxide production current,
      equivalent to ``mode="reaction", reaction_index=R_2e_idx``.  Defaults
      ``R_2e_idx = 0`` (R_2e is at index 0 in the parallel preset).
      Override via ``reaction_index`` for non-default layouts.
    - ``"peroxide_current"`` (DEPRECATED): net peroxide ``scale * (R_0 - R_1)
      * ds``.  Retained for back-compat with v13/v15/v16 inverse scripts
      that target the legacy sequential model.  Per the post-Ruggiero
      analysis the deck maps to gross 2e production, not the net — use
      ``gross_h2o2_current`` for new comparisons.
    - ``"reaction"``: single reaction ``scale * R_{reaction_index} * ds``.
    """
    import firedrake as fd

    mode_norm = str(mode).strip().lower()
    bv_rate_exprs = list(ctx["bv_rate_exprs"])
    bv_cfg = ctx.get("bv_settings", {})
    electrode_marker = int(bv_cfg.get("electrode_marker", 1))
    ds = fd.Measure("ds", domain=ctx["mesh"])
    scale_const = fd.Constant(float(scale))

    if mode_norm == "current_density":
        # Electron-weighted total current.  Each reaction contributes
        # (n_e_j / N_ELECTRONS_REF) * R_j so the external I_SCALE
        # (anchored to n_e=2) yields physical units.
        n_e_list = _get_reaction_n_electrons(ctx)
        rate_sum = 0
        if n_e_list is not None and len(n_e_list) == len(bv_rate_exprs):
            ref = float(N_ELECTRONS_REF)
            for n_e_j, R_j in zip(n_e_list, bv_rate_exprs):
                weight = fd.Constant(float(n_e_j) / ref)
                rate_sum = rate_sum + weight * R_j
        else:
            # Legacy fallback: unweighted sum (pre-M3a.1 behavior).
            for R_j in bv_rate_exprs:
                rate_sum = rate_sum + R_j
        return scale_const * rate_sum * ds(electrode_marker)

    elif mode_norm == "gross_h2o2_current":
        # Gross 2e peroxide production current.  Maps to the single
        # R_2e rate.  Default reaction_index = 0 (R_2e is at index 0 in
        # the parallel preset and was R_0 in the legacy sequential
        # preset, so this also gives the deck-aligned gross 2e for
        # both topologies).
        idx = 0 if reaction_index is None else int(reaction_index)
        if idx < 0 or idx >= len(bv_rate_exprs):
            raise ValueError(
                f"gross_h2o2_current reaction_index {idx} out of bounds "
                f"for {len(bv_rate_exprs)} reactions."
            )
        return scale_const * bv_rate_exprs[idx] * ds(electrode_marker)

    elif mode_norm == "peroxide_current":
        # DEPRECATED: net peroxide R_0 - R_1.  Kept for back-compat with
        # v13/v15/v16 inverse scripts.  The deck's "Peroxide Current
        # Density" actually maps to gross 2e production (single rate);
        # use ``gross_h2o2_current`` for new comparisons.
        if len(bv_rate_exprs) < 2:
            raise ValueError(
                "peroxide_current mode requires at least 2 reactions; "
                f"got {len(bv_rate_exprs)}."
            )
        return scale_const * (bv_rate_exprs[0] - bv_rate_exprs[1]) * ds(electrode_marker)

    elif mode_norm == "reaction":
        if reaction_index is None:
            raise ValueError("reaction_index must be set when mode='reaction'.")
        idx = int(reaction_index)
        if idx < 0 or idx >= len(bv_rate_exprs):
            raise ValueError(
                f"reaction_index {idx} out of bounds for {len(bv_rate_exprs)} reactions."
            )
        return scale_const * bv_rate_exprs[idx] * ds(electrode_marker)

    else:
        raise ValueError(
            f"Unknown BV observable mode '{mode}'. "
            "Use 'current_density', 'gross_h2o2_current', 'peroxide_current', or 'reaction'."
        )


def assemble_observable_validated(
    form: object,
    *,
    I_lim: float,
    phi_applied: float,
    V_T: float,
    mode: str = "current_density",
) -> tuple[float, object]:
    """Assemble an observable form and validate the result.

    Parameters
    ----------
    form : ufl.Form
        The observable form to assemble.
    I_lim : float
        Diffusion-limited current (same units as assembled value).
    phi_applied : float
        Applied potential (nondimensional).
    V_T : float
        Thermal voltage for sign convention.
    mode : str
        ``"current_density"``, ``"gross_h2o2_current"``, or
        ``"peroxide_current"`` — affects which checks run.

    Returns
    -------
    tuple[float, ValidationResult]
        (assembled_value, validation_result)
    """
    import firedrake as fd
    from .validation import ValidationResult

    value = float(fd.assemble(form))

    # F2 (magnitude exceeds diffusion limit) for single-observable
    # validation.  Cross-observable checks (F3, F7) require both cd and
    # peroxide quantities, which happens at the pipeline level
    # (training.py, bv_curve_eval.py).
    failures = []
    warnings_list = []

    if mode in ("current_density", "peroxide_current", "gross_h2o2_current"):
        if abs(value) > abs(I_lim) * 1.05:
            failures.append(f"F2: |{mode}|={abs(value):.4g} exceeds I_lim={abs(I_lim):.4g}")

    result = ValidationResult(valid=len(failures) == 0, failures=failures, warnings=warnings_list)
    return value, result
