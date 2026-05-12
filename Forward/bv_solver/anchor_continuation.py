"""k0-continuation primitives for the BV-PNP forward solver (Phase 5γ MVP).

This module provides the building blocks for **k0 continuation**: the
strategy of ramping ``k0`` upward from a small floor to its production
target so each Newton solve can warm-start from a converged predecessor.
The motivation is the Phase 5α gate failure at ``V_RHE = +0.55 V`` on
the multi-ion stack, where the Picard predicts ``psi_S ≈ 19.83`` and
the resulting ``exp(0.5 * 4 * 7.22) ≈ 1.8e6`` rate cannot be matched
by H⁺ mass-transport supply at production ``k0``.

The MVP exports four objects:

  * :func:`set_reaction_k0_model` — write reaction ``j``'s nondim ``k0``
    to **both** the metadata dict (read by Picard) and the live UFL
    Function (read by the FE residual). Both layers must agree.
  * :func:`get_reaction_k0_model` — read the authoritative ``k0`` value
    from the metadata dict.
  * :class:`AdaptiveLadder` — geometric scale ladder controller with
    insert-on-failure rollback (sqrt(prev * curr) midpoints) and a
    per-step insert cap.
  * :func:`solve_anchor_with_continuation` — orchestrates the full
    continuation: builds context+forms+IC at the target voltage, ramps
    ``k0`` down to the ladder floor, then walks back up via SS solves,
    rolling back on failure.

Companion data classes :class:`AnchorContinuationResult` and
exception :class:`LadderExhausted` describe the success/failure shapes.

References
----------
* ``StudyResults/fast_realignment_2026-05-08/PHASE_5_ALPHA_GATE_FAILURE.md``
  diagnoses the convergence wall at production ``k0``.
* ``scripts/studies/parallel_2e_4e_k0_ladder_probe.py`` is the
  reference linear-ladder pattern this module factors into reusable
  primitives.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Solver-options keys that are config metadata, not PETSc options. These
# must be filtered before the dict is passed to a NonlinearVariationalSolver
# so they don't pollute the PETSc options database with "unknown option"
# warnings. Centralized here so M3 callers and external callers agree on
# the filter (Risk R2: drift between scripts).
NON_PETSC_KEYS: frozenset[str] = frozenset(
    {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
)


# ---------------------------------------------------------------------------
# Result + exception types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AnchorContinuationResult:
    """Outcome of :func:`solve_anchor_with_continuation`.

    Attributes
    ----------
    converged
        ``True`` iff the ladder reached ``scale = 1.0`` and the final
        SS solve returned ``True``.
    U_data
        Snapshot of ``ctx['U'].dat`` at the last successful rung
        (``tuple(d.data_ro.copy() for d in U.dat)``); ``None`` if no
        rung succeeded.
    ladder_history
        ``(scale, "ok"|"fail")`` tuples in chronological attempt order.
    rungs
        Per-rung diagnostics dicts (one entry per attempted rung,
        successful or not).
    ctx
        Live Firedrake context. Caller may extract observables before
        the next ``build_context``. Treat as borrowed; do not stash
        across long-running outer loops (see Risk R6).
    """

    converged: bool
    U_data: Optional[tuple]
    ladder_history: List[Tuple[float, str]]
    rungs: List[Dict[str, Any]]
    ctx: Dict[str, Any]


class LadderExhausted(RuntimeError):
    """Raised when :class:`AdaptiveLadder` cannot insert more midpoints.

    Indicates the ladder failed at a rung and ``max_inserts_per_step``
    is exhausted. Caller should report the ladder history and consider
    densifying ``initial_scales`` or raising ``max_inserts_per_step``.
    """


# ---------------------------------------------------------------------------
# PreconvergedAnchor — frozen handoff object for warm-walk grid drivers
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PreconvergedAnchor:
    """Frozen handoff: a converged Newton state seeding warm walks.

    The minimum information needed to reproduce a converged state on a
    fresh ctx of the same mesh:

      * ``phi_applied_eta`` -- nondim ``V/V_T`` of the anchor.
      * ``U_snapshot`` -- ``tuple(d.data_ro.copy() for d in U.dat)``.
        Must match ``mesh_dof_count`` per-subspace shape on the
        receiving ctx.
      * ``k0_targets`` -- production nondim ``k0`` per reaction index;
        the Newton state was converged with these.
      * ``mesh_dof_count`` -- total ``ctx['U'].function_space().dim()``
        at extract time. Receiver asserts equality before
        ``restore_U``.
      * ``ladder_history`` -- frozen tuple of the continuation ladder
        that produced this anchor (provenance, not used at runtime).

    Frozen: no field is mutable post-construction. Pass by value;
    safe to capture in long-lived dispatchers.
    """

    phi_applied_eta: float
    U_snapshot: tuple
    k0_targets: tuple                                    # ((j, k0), ...)
    mesh_dof_count: int
    ladder_history: tuple                                # ((scale, "ok"|"fail"), ...)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.phi_applied_eta)):
            raise ValueError(
                f"phi_applied_eta must be finite "
                f"(got {self.phi_applied_eta!r})"
            )
        if not isinstance(self.mesh_dof_count, int) or self.mesh_dof_count <= 0:
            raise ValueError(
                f"mesh_dof_count must be a positive int "
                f"(got {self.mesh_dof_count!r})"
            )
        if not isinstance(self.U_snapshot, tuple) or len(self.U_snapshot) == 0:
            raise ValueError(
                "U_snapshot must be a non-empty tuple of numpy arrays"
            )
        for arr in self.U_snapshot:
            if not isinstance(arr, np.ndarray):
                raise ValueError(
                    f"U_snapshot entries must be numpy arrays "
                    f"(got {type(arr).__name__})"
                )
        if not isinstance(self.k0_targets, tuple) or len(self.k0_targets) == 0:
            raise ValueError(
                "k0_targets must be a non-empty tuple of (int, float) pairs"
            )
        for entry in self.k0_targets:
            if (not isinstance(entry, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"k0_targets entries must be (int, float) tuples "
                    f"(got {entry!r})"
                )
            j, k = entry
            if not isinstance(j, int):
                raise ValueError(
                    f"k0_targets reaction index must be int "
                    f"(got {type(j).__name__} for entry {entry!r})"
                )
            if not (float(k) > 0.0):
                raise ValueError(
                    f"k0_targets[{j}] must be > 0 "
                    f"(got {k!r}); k <= 0 disables the rate UFL term."
                )

    def k0_targets_dict(self) -> Dict[int, float]:
        """Defensive-copy view as a regular dict.

        For orchestrator loops that prefer dict semantics. Mutating the
        returned dict has no effect on the frozen anchor.
        """
        return {int(j): float(k) for j, k in self.k0_targets}


def extract_preconverged_anchor(
    result: AnchorContinuationResult,
    *,
    phi_applied_eta: float,
    k0_targets: Mapping[int, float],
    mesh_dof_count: int,
) -> PreconvergedAnchor:
    """Build a frozen anchor from a successful continuation result.

    Parameters
    ----------
    result
        Outcome of :func:`solve_anchor_with_continuation`. Must be
        converged with a non-``None`` ``U_data``.
    phi_applied_eta
        Nondim voltage at which ``result`` was produced.
    k0_targets
        Per-reaction production ``k0`` mapping at which the result
        converged. Used to drive subsequent warm walks at the same
        rate.
    mesh_dof_count
        ``ctx['U'].function_space().dim()`` of the live ctx that
        produced ``result``. Caller passes it because
        :class:`AnchorContinuationResult` does not capture it.

    Raises
    ------
    ValueError
        ``result.converged is False`` or ``result.U_data is None``.
    """
    if not result.converged:
        raise ValueError(
            "cannot extract anchor: result.converged is False"
        )
    if result.U_data is None:
        raise ValueError(
            "cannot extract anchor: result.U_data is None"
        )
    return PreconvergedAnchor(
        phi_applied_eta=float(phi_applied_eta),
        U_snapshot=tuple(np.asarray(arr).copy() for arr in result.U_data),
        k0_targets=tuple(
            (int(j), float(k)) for j, k in sorted(k0_targets.items())
        ),
        mesh_dof_count=int(mesh_dof_count),
        ladder_history=tuple(
            (float(s), str(o)) for s, o in result.ladder_history
        ),
    )


# ---------------------------------------------------------------------------
# k0 metadata helpers
# ---------------------------------------------------------------------------

def set_reaction_k0_model(ctx: dict, j: int, k0_model_value: float) -> None:
    """Set reaction ``j``'s nondim ``k0`` in BOTH the metadata dict and the FE Function.

    Two layers read ``k0`` independently and must agree:

    * **Picard** reads ``ctx['nondim']['bv_reactions'][j]['k0_model']``
      (``picard_ic.py:954``).
    * **FE residual** reads ``ctx['bv_k0_funcs'][j]`` (UFL Function
      assigned at form-build, ``forms_logc_muh.py:401``).

    Order: dict first, Function second. Either failing leaves an
    inconsistent state; callers should treat any exception as fatal.

    Parameters
    ----------
    ctx
        Firedrake context built by ``build_context_logc_muh`` /
        ``build_forms_logc_muh``.
    j
        Reaction index into ``ctx['nondim']['bv_reactions']`` and
        ``ctx['bv_k0_funcs']``.
    k0_model_value
        New nondim ``k0``. Must be ``> 0`` — see *Raises*.

    Raises
    ------
    ValueError
        ``k0_model_value <= 0``. ``forms_logc_muh.py:410-412``
        short-circuits ``R_j = fd.Constant(0.0)`` at ``k0 <= 0``;
        once disabled, the rate UFL is a constant zero and
        ``bv_k0_funcs[j].assign`` cannot reactivate it. Continuation
        floors must be ``ε > 0``.
    IndexError
        ``j`` is out of range for ``bv_reactions`` or ``bv_k0_funcs``.
    """
    if not (k0_model_value > 0.0):
        raise ValueError(
            f"k0_model_value must be > 0 (got {k0_model_value!r}); "
            "k0 <= 0 disables the rate UFL term irreversibly."
        )
    nondim = ctx.get("nondim", {})
    rxns = nondim.get("bv_reactions", [])
    if j < 0 or j >= len(rxns):
        raise IndexError(
            f"reaction index {j} out of range (N={len(rxns)})"
        )
    funcs = ctx.get("bv_k0_funcs", [])
    if j >= len(funcs):
        raise IndexError(
            f"reaction index {j} out of range for bv_k0_funcs "
            f"(N={len(funcs)})"
        )

    # Replace the per-reaction dict (defensive copy) so a caller's
    # earlier reference to the old dict isn't mutated underfoot. The
    # outer list is reassigned to keep nondim['bv_reactions'] pointing
    # at the new sequence (guards against aliasing if the dict was
    # captured elsewhere).
    new_rxn = {**rxns[j], "k0_model": float(k0_model_value)}
    new_rxns = list(rxns)
    new_rxns[j] = new_rxn
    nondim["bv_reactions"] = new_rxns
    ctx["nondim"] = nondim

    funcs[j].assign(float(k0_model_value))


def get_reaction_k0_model(ctx: dict, j: int) -> float:
    """Read reaction ``j``'s current ``k0_model`` from the metadata dict.

    The metadata dict is the authoritative source for Picard. The FE
    Function should always agree (set by :func:`set_reaction_k0_model`)
    but is a *display* of the same value — read the dict.
    """
    return float(ctx["nondim"]["bv_reactions"][j]["k0_model"])


# ---------------------------------------------------------------------------
# Phase 6α — water self-ionization Kw_eff continuation helpers
# ---------------------------------------------------------------------------

def set_reaction_kw_eff_model(ctx: dict, kw_eff_value: float) -> None:
    """Set the water-ionization ``K_w_eff`` in the live FE Function + bv_convergence dict.

    Mirrors :func:`set_reaction_k0_model`: writes to BOTH the metadata
    layer and the live UFL Function so the form residual sees the same
    value Picard / diagnostics see.

    Two layers read ``K_w_eff`` independently and must agree:

    * **Form residual** reads
      ``ctx['water_ionization'].kw_eff_func`` — the R-space Function
      assigned at form-build by :func:`build_water_ionization_terms`.
    * **Picard / Q3 IC** and downstream diagnostics read
      ``ctx['nondim']['bv_convergence']['kw_eff_hat']`` (or the
      equivalent on ``ctx['bv_convergence']``).

    Order: dict first, Function second.  Either failing leaves an
    inconsistent state; callers should treat any exception as fatal.

    Parameters
    ----------
    ctx
        Firedrake context built by ``build_forms_logc`` /
        ``build_forms_logc_muh`` with ``enable_water_ionization=True``.
    kw_eff_value
        New nondim ``Kw_eff``.  Must be ``>= 0`` (the ladder may start
        at 0; physical KW_HAT is positive).

    Raises
    ------
    ValueError
        ``kw_eff_value < 0`` or no water-ionization bundle on ``ctx``
        (``enable_water_ionization`` was not True at form-build).
    """
    if kw_eff_value < 0.0:
        raise ValueError(
            f"kw_eff_value must be non-negative (got {kw_eff_value!r}); "
            "Kw_eff = 0 corresponds to a fully-disabled water source."
        )
    bundle = ctx.get("water_ionization")
    if bundle is None:
        raise ValueError(
            "ctx has no 'water_ionization' bundle — was the context built "
            "with enable_water_ionization=True on bv_convergence?  "
            "set_reaction_kw_eff_model is a no-op without that flag."
        )

    # Update the bv_convergence cfg copy on ctx so downstream readers
    # (Picard, diagnostics) see the new value.  Both ctx['bv_convergence']
    # and ctx['nondim']['bv_convergence'] are kept in sync if present.
    val = float(kw_eff_value)
    if "bv_convergence" in ctx and isinstance(ctx["bv_convergence"], dict):
        new_cfg = dict(ctx["bv_convergence"])
        new_cfg["kw_eff_hat"] = val
        ctx["bv_convergence"] = new_cfg

    # The FE Function is the single source of truth for the residual.
    bundle.kw_eff_func.assign(val)


def get_reaction_kw_eff_model(ctx: dict) -> float:
    """Read the current ``Kw_eff`` from the live FE Function.

    The FE Function is the residual-side authoritative value;
    diagnostics that need to confirm what Newton actually used should
    read this rather than the metadata copy.

    Raises
    ------
    ValueError
        No water-ionization bundle on ``ctx``.
    """
    bundle = ctx.get("water_ionization")
    if bundle is None:
        raise ValueError(
            "ctx has no 'water_ionization' bundle; was the context built "
            "with enable_water_ionization=True?"
        )
    return float(bundle.kw_eff_func)


# ---------------------------------------------------------------------------
# Phase 6β v9 Gate 2 — Stern capacitance continuation helpers
# ---------------------------------------------------------------------------

def set_stern_capacitance_model(ctx: dict, c_s_f_m2: float) -> None:
    """Set the Stern capacitance in BOTH metadata and the live FE Constant.

    Phase 6β v9 Gate 2 helper.  Mirrors ``set_reaction_k0_model`` and
    ``set_reaction_kw_eff_model``: writes both the metadata layer
    (``ctx['nondim']['bv_stern_capacitance_model']``) and the live UFL
    Constant (``ctx['stern_coeff_const']``) so the form residual sees
    the same value Picard / diagnostics see.

    The argument is the **physical** Stern capacitance in F/m² (matching
    the original ``stern_capacitance_f_m2`` config field).  This is
    different from ``set_reaction_k0_model`` (nondim) but consistent
    with how callers think about Stern values.  The phys→nondim
    conversion factor was stashed in ctx at form-build time
    (``ctx['nondim']['bv_stern_phys_to_nondim_factor']``).

    Parameters
    ----------
    ctx
        Firedrake context built by ``build_forms_logc`` /
        ``build_forms_logc_muh`` with ``stern_capacitance_f_m2 > 0``.
    c_s_f_m2
        New Stern capacitance in F/m².  Must be ``>= 0``; the C_S = 0
        limit is the no-Stern Dirichlet (build-time decision, not
        runtime), so this setter rejects ``< 0`` only.

    Raises
    ------
    ValueError
        ``c_s_f_m2 < 0`` or no live Stern Constant on ``ctx``
        (``stern_capacitance_f_m2`` was not set positive at build).
    """
    if c_s_f_m2 < 0.0:
        raise ValueError(
            f"c_s_f_m2 must be non-negative (got {c_s_f_m2!r})"
        )
    stern_const = ctx.get("stern_coeff_const")
    if stern_const is None:
        raise ValueError(
            "ctx has no 'stern_coeff_const' — was the form built with "
            "stern_capacitance_f_m2 > 0?  set_stern_capacitance_model is "
            "a no-op without an active Stern Robin BC."
        )
    scaling = ctx.get("nondim", {})
    factor = float(scaling.get("bv_stern_phys_to_nondim_factor", 1.0))
    nondim_value = float(c_s_f_m2) * factor

    # Update the metadata layer first so any downstream readers
    # (Picard's stern_split, diagnostics) see the new value.
    if isinstance(scaling, dict):
        new_scaling = dict(scaling)
        new_scaling["bv_stern_capacitance_model"] = nondim_value
        ctx["nondim"] = new_scaling

    # Then update the live Constant so the residual reflects it.
    stern_const.assign(nondim_value)


def get_stern_capacitance_model(ctx: dict) -> float:
    """Read the current Stern capacitance in F/m² from the live FE Constant.

    Inverts the phys→nondim factor stashed at form-build time.

    Raises
    ------
    ValueError
        No live Stern Constant on ``ctx``.
    """
    stern_const = ctx.get("stern_coeff_const")
    if stern_const is None:
        raise ValueError(
            "ctx has no 'stern_coeff_const'; was the form built with "
            "stern_capacitance_f_m2 > 0?"
        )
    scaling = ctx.get("nondim", {})
    factor = float(scaling.get("bv_stern_phys_to_nondim_factor", 1.0))
    if factor == 0.0:
        return 0.0
    return float(stern_const) / factor


# ---------------------------------------------------------------------------
# Phase 6β v9 Gate 3C — cation hydrolysis continuation accessors
# ---------------------------------------------------------------------------


def _require_cation_hydrolysis_bundle(ctx: dict):
    """Fetch ``ctx['cation_hydrolysis']`` or raise with a helpful message."""
    bundle = ctx.get("cation_hydrolysis")
    if bundle is None:
        raise ValueError(
            "ctx has no 'cation_hydrolysis' bundle — was the form built "
            "with enable_cation_hydrolysis=True on bv_convergence?  The "
            "Gate 3C accessors are no-ops without that flag."
        )
    return bundle


def set_reaction_lambda_hydrolysis_model(ctx: dict, lambda_value: float) -> None:
    """Set the cation-hydrolysis ``λ`` continuation knob.

    Mirrors :func:`set_reaction_kw_eff_model`: writes to BOTH the
    metadata layer (``ctx['bv_convergence']['lambda_hydrolysis']``)
    and the live UFL Function
    (``ctx['cation_hydrolysis'].lambda_hydrolysis_func``) so the form
    residual sees the same value Picard / diagnostics see.

    Parameters
    ----------
    ctx
        Firedrake context built with ``enable_cation_hydrolysis=True``.
    lambda_value
        New activation knob value, must lie in ``[0, 1]``.

    Raises
    ------
    ValueError
        ``lambda_value`` outside ``[0, 1]`` or no cation_hydrolysis
        bundle on ``ctx``.
    """
    if not (0.0 <= float(lambda_value) <= 1.0):
        raise ValueError(
            f"lambda_value must lie in [0, 1] (got {lambda_value!r}); "
            "outside this range the Γ smooth-blend residual loses "
            "its monotone Newton path."
        )
    bundle = _require_cation_hydrolysis_bundle(ctx)
    val = float(lambda_value)
    if "bv_convergence" in ctx and isinstance(ctx["bv_convergence"], dict):
        new_cfg = dict(ctx["bv_convergence"])
        new_cfg["lambda_hydrolysis"] = val
        ctx["bv_convergence"] = new_cfg
    bundle.lambda_hydrolysis_func.assign(val)


def get_reaction_lambda_hydrolysis_model(ctx: dict) -> float:
    """Read the current ``λ`` value from the live FE Function."""
    bundle = _require_cation_hydrolysis_bundle(ctx)
    return float(bundle.lambda_hydrolysis_func)


def _set_kinetic_rate(
    ctx: dict, attr_name: str, config_key: str, value: float,
    *, allow_zero: bool = True,
) -> None:
    """Internal helper: write a kinetic-rate Function + metadata in lockstep.

    Mirrors the ``set_reaction_kw_eff_model`` two-layer update: writes
    the dict layer (``ctx['bv_convergence']['cation_hydrolysis_config'][config_key]``)
    AND the live FE Function (``getattr(bundle, attr_name)``).
    """
    val = float(value)
    if val < 0.0:
        raise ValueError(
            f"{config_key} must be non-negative (got {val!r})"
        )
    if not allow_zero and val == 0.0:
        raise ValueError(
            f"{config_key} must be positive (got {val!r}); zero leaves "
            f"the corresponding kinetic branch undefined."
        )
    bundle = _require_cation_hydrolysis_bundle(ctx)
    if "bv_convergence" in ctx and isinstance(ctx["bv_convergence"], dict):
        new_cfg = dict(ctx["bv_convergence"])
        sub_cfg = dict(new_cfg.get("cation_hydrolysis_config") or {})
        sub_cfg[config_key] = val
        new_cfg["cation_hydrolysis_config"] = sub_cfg
        ctx["bv_convergence"] = new_cfg
    getattr(bundle, attr_name).assign(val)


def set_reaction_k_hyd_model(ctx: dict, k_hyd_value: float) -> None:
    """Set the forward hydrolysis rate ``k_hyd`` in BOTH metadata + Function."""
    _set_kinetic_rate(ctx, "k_hyd_func", "k_hyd", k_hyd_value)


def set_reaction_k_prot_model(ctx: dict, k_prot_value: float) -> None:
    """Set the reverse protonation rate ``k_prot`` in BOTH metadata + Function."""
    _set_kinetic_rate(ctx, "k_prot_func", "k_prot", k_prot_value)


def set_reaction_k_des_model(ctx: dict, k_des_value: float) -> None:
    """Set the desorption rate ``k_des`` in BOTH metadata + Function.

    ``k_des = 0`` is rejected because the Γ residual at λ=1 reduces to
    ``R_net = 0`` (forces ``R_net = 0`` for any γ), giving Newton an
    indeterminate state.
    """
    _set_kinetic_rate(
        ctx, "k_des_func", "k_des", k_des_value, allow_zero=False,
    )


def set_reaction_delta_ohp_model(ctx: dict, delta_ohp_value: float) -> None:
    """Set the OHP thickness ``δ_OHP`` (nondim) in BOTH metadata + Function."""
    _set_kinetic_rate(
        ctx, "delta_ohp_func", "delta_ohp_hat", delta_ohp_value,
        allow_zero=False,
    )


def set_reaction_gamma_max_model(ctx: dict, gamma_max_value: float) -> None:
    """Set the Langmuir saturation cap Γ_max in BOTH metadata + Function.

    Phase 6β v10a accessor — mirrors the kinetic-rate setters.
    Writes the live FE Function plus the metadata layer
    (``cation_hydrolysis_config['gamma_max_nondim']``) so downstream
    readers (Picard, diagnostics) see the current value.

    Raises
    ------
    ValueError
        ``gamma_max_value <= 0`` (Γ_max = 0 forces the vacancy factor
        ``(1 − Γ/Γ_max)`` to diverge) or no cation_hydrolysis bundle
        on ``ctx``.
    """
    _set_kinetic_rate(
        ctx, "gamma_max_func", "gamma_max_nondim", gamma_max_value,
        allow_zero=False,
    )


def set_reaction_r_H_El_pm_model(ctx: dict, r_H_El_pm_value: float) -> None:
    """Set Singh's hydration-shell H to electrode distance r_H_El (pm).

    Phase 6β v9 Gate 4B sweep accessor.  Mirrors the kinetic-rate
    setters: writes the live FE Function plus the metadata layer
    (``cation_hydrolysis_config['pka_shift_params']['r_H_El_pm']``)
    so downstream readers see the current value.

    Raises
    ------
    ValueError
        ``r_H_El_pm_value <= 0`` (geometric distance must be positive)
        or no cation_hydrolysis bundle on ``ctx``.
    """
    val = float(r_H_El_pm_value)
    if val <= 0.0:
        raise ValueError(
            f"r_H_El_pm must be positive (got {val!r}); the Singh "
            "geometric factor diverges as r_H_El → 0."
        )
    bundle = _require_cation_hydrolysis_bundle(ctx)
    if "bv_convergence" in ctx and isinstance(ctx["bv_convergence"], dict):
        new_cfg = dict(ctx["bv_convergence"])
        sub_cfg = dict(new_cfg.get("cation_hydrolysis_config") or {})
        # Mutate the nested pka_shift_params if present.
        nested = sub_cfg.get("pka_shift_params")
        if isinstance(nested, dict):
            new_nested = dict(nested)
            new_nested["r_H_El_pm"] = val
            sub_cfg["pka_shift_params"] = new_nested
        new_cfg["cation_hydrolysis_config"] = sub_cfg
        ctx["bv_convergence"] = new_cfg
    bundle.r_H_El_pm_func.assign(val)
    # Keep the bundle's cation_params dict in sync — readers (Picard
    # update, diagnostics) may inspect it.
    new_params = dict(bundle.cation_params)
    new_params["r_H_El_pm"] = val
    object.__setattr__(bundle, "cation_params", new_params)


def set_reaction_beta_offset_pm2_model(
    ctx: dict, beta_offset_value: float,
) -> None:
    """Set the Phase D Δ_β carbon-vs-Cu offset (pm²) on residual + metadata.

    Phase 6β step 10 Phase D sweep accessor.  Mirrors
    :func:`set_reaction_r_H_El_pm_model`: writes the live FE Function
    plus the metadata layer
    (``cation_hydrolysis_config['beta_offset_pm2']``) and the
    ``bundle.cation_params`` dict so Picard / diagnostics consumers see
    the current value.

    Sign convention: positive Δ_β raises ``β_carbon`` (less negative
    under Singh's cathodic geometry) and reduces ``|ΔpKa|``; the Phase
    D fit's bracket construction (plan §3.1) caps Δ_β strictly below
    ``-β_K_Cu = +45.608 pm²`` to preserve ``β_carbon < 0``.

    Raises
    ------
    ValueError
        No cation_hydrolysis bundle on ``ctx``.  Any sign of
        ``beta_offset_value`` is accepted (the residual + bracket
        construction is responsible for sign-flip safety).
    """
    val = float(beta_offset_value)
    bundle = _require_cation_hydrolysis_bundle(ctx)
    if "bv_convergence" in ctx and isinstance(ctx["bv_convergence"], dict):
        new_cfg = dict(ctx["bv_convergence"])
        sub_cfg = dict(new_cfg.get("cation_hydrolysis_config") or {})
        sub_cfg["beta_offset_pm2"] = val
        new_cfg["cation_hydrolysis_config"] = sub_cfg
        ctx["bv_convergence"] = new_cfg
    bundle.beta_offset_pm2_func.assign(val)
    # Mirror into bundle.cation_params (NOT bundle.params — the
    # frozen dataclass has no `params` attribute; cation_params is
    # the public diagnostics-readable dict).
    new_params = dict(bundle.cation_params)
    new_params["beta_offset_pm2"] = val
    object.__setattr__(bundle, "cation_params", new_params)


# ---------------------------------------------------------------------------
# AdaptiveLadder — geometric scale ramp with sqrt-mean rollback
# ---------------------------------------------------------------------------

class AdaptiveLadder:
    """Geometric scale ramp with insert-on-failure rollback.

    Manages a sequence of monotonic-increasing positive scales ending
    at ``1.0``. The orchestrator queries ``current_scale`` for the
    next rung, attempts the SS solve, and reports back via
    :meth:`record_success` or :meth:`record_failure_and_insert`.

    Failure inserts a geometric mean ``sqrt(prev * curr)`` between the
    previous successful scale (``previous_scale``) and the current
    failing scale, halving the **log-distance**. The orchestrator must
    rollback ``U`` to the previous-success snapshot before retrying;
    that rollback is the orchestrator's responsibility, not the
    ladder's.

    The insert counter is per-step: :meth:`record_success` resets it
    so each newly-attempted scale gets a fresh budget.

    First-rung failures (``previous_scale is None``) normally return
    ``False`` from :meth:`record_failure_and_insert` because there is
    no previous success to interpolate from. When ``warm_start_floor``
    is set, this behavior changes: first-rung failures **arithmetic-
    bisect** between ``warm_start_floor`` and the failed scale. This
    is the appropriate operation for λ-from-warm-start ramps, where
    the "previous" state is the warm-start (e.g. λ=0) and the search
    space ``[0, 1]`` is linearly spaced rather than log-spaced.

    Parameters
    ----------
    initial_scales
        Strictly-increasing positive scales ending at ``1.0``.
        Must be non-empty.
    max_inserts_per_step
        Cap on midpoints inserted between any single
        previous_scale → current_scale gap. Default 4.
    warm_start_floor
        Optional arithmetic-bisection floor for first-rung failures.
        When ``None`` (default), first-rung failures return ``False``
        immediately. When set to a float (typically ``0.0``), first-
        rung failures insert the arithmetic midpoint
        ``0.5 * (warm_start_floor + scale)``. Must satisfy
        ``warm_start_floor < initial_scales[0]`` (a strictly-lower
        floor); otherwise ``ValueError`` is raised.
    """

    def __init__(
        self,
        *,
        initial_scales: tuple[float, ...],
        max_inserts_per_step: int = 4,
        warm_start_floor: Optional[float] = None,
    ) -> None:
        if not initial_scales:
            raise ValueError("initial_scales must be non-empty")
        if any(s <= 0.0 for s in initial_scales):
            raise ValueError(
                f"initial_scales entries must all be > 0 "
                f"(got {initial_scales!r})"
            )
        for a, b in zip(initial_scales[:-1], initial_scales[1:]):
            if not (b > a):
                raise ValueError(
                    f"initial_scales must be strictly monotonic increasing "
                    f"(got {initial_scales!r})"
                )
        if initial_scales[-1] != 1.0:
            raise ValueError(
                f"initial_scales must end at 1.0 "
                f"(got {initial_scales!r})"
            )
        if max_inserts_per_step < 0:
            raise ValueError(
                f"max_inserts_per_step must be >= 0 "
                f"(got {max_inserts_per_step})"
            )
        if warm_start_floor is not None:
            if not (float(warm_start_floor) < float(initial_scales[0])):
                raise ValueError(
                    f"warm_start_floor must be strictly less than "
                    f"initial_scales[0] (got floor={warm_start_floor!r}, "
                    f"first scale={initial_scales[0]!r})"
                )

        self._planned: List[float] = list(initial_scales)
        self._idx: int = 0
        self._inserts_at_current_step: int = 0
        self._max_inserts_per_step: int = int(max_inserts_per_step)
        self._history: List[Tuple[float, str]] = []
        self._warm_start_floor: Optional[float] = (
            float(warm_start_floor) if warm_start_floor is not None else None
        )

    @property
    def current_scale(self) -> float:
        """Scale of the rung the orchestrator is about to attempt."""
        if self.is_done():
            raise IndexError("ladder exhausted: is_done() is True")
        return self._planned[self._idx]

    @property
    def previous_scale(self) -> Optional[float]:
        """Last successful scale, or ``None`` if at the floor."""
        if self._idx == 0:
            return None
        return self._planned[self._idx - 1]

    def is_done(self) -> bool:
        """``True`` once every planned rung has been recorded as ``ok``."""
        return self._idx >= len(self._planned)

    def record_success(self) -> None:
        """Mark the current rung successful; advance to the next."""
        if self.is_done():
            raise RuntimeError("cannot record success after is_done()")
        scale = self._planned[self._idx]
        self._history.append((float(scale), "ok"))
        self._idx += 1
        self._inserts_at_current_step = 0

    def record_failure_and_insert(self) -> bool:
        """Record the current rung as failed; insert a midpoint.

        When ``previous_scale`` is available (post-first-rung), uses
        the geometric midpoint ``sqrt(prev * scale)``. When at the
        first rung, returns ``False`` unless ``warm_start_floor`` is
        configured; in that case, inserts the arithmetic midpoint
        ``0.5 * (warm_start_floor + scale)``.

        Returns
        -------
        bool
            ``True`` if a midpoint was inserted and the orchestrator
            should rollback ``U`` and retry the new ``current_scale``.
            ``False`` if ``max_inserts_per_step`` is exhausted, if no
            previous-scale anchor exists and no ``warm_start_floor``
            is set, or if the arithmetic-bisection midpoint would
            collapse to a numerical floor; the caller should raise
            :class:`LadderExhausted`.
        """
        if self.is_done():
            raise RuntimeError("cannot record failure after is_done()")
        scale = self._planned[self._idx]
        self._history.append((float(scale), "fail"))
        if self._inserts_at_current_step >= self._max_inserts_per_step:
            return False
        prev = self.previous_scale
        if prev is None:
            # No previous success — there's nothing to interpolate from
            # using the geometric path.
            if self._warm_start_floor is None:
                # Existing behavior: fail fast.
                return False
            # NEW: arithmetic-bisect toward the warm-start floor. λ in
            # [0, 1] is linearly spaced, and sqrt(0 * x) = 0 is
            # ill-defined; arithmetic midpoint is the correct probe.
            midpoint = 0.5 * (self._warm_start_floor + scale)
            # Guard: midpoint must be strictly between floor and scale.
            # If the gap has collapsed below float precision, decline
            # to subdivide further (numerical floor; no infinite-loop
            # risk).
            if not (self._warm_start_floor < midpoint < scale):
                return False
            self._planned.insert(self._idx, float(midpoint))
            self._inserts_at_current_step += 1
            return True
        midpoint = math.sqrt(prev * scale)
        self._planned.insert(self._idx, float(midpoint))
        self._inserts_at_current_step += 1
        return True

    def history(self) -> List[Tuple[float, str]]:
        """Defensive copy of the (scale, outcome) log."""
        return list(self._history)

    def planned_scales(self) -> List[float]:
        """Defensive copy of the current planned scale sequence (with inserts)."""
        return list(self._planned)


# ---------------------------------------------------------------------------
# solve_anchor_with_continuation — MVP orchestrator
# ---------------------------------------------------------------------------

def solve_anchor_with_continuation(
    sp,
    *,
    mesh,
    k0_targets: Dict[int, float],
    initial_scales: Tuple[float, ...] = (1e-12, 1e-9, 1e-6, 1e-3, 1.0),
    max_inserts_per_step: int = 4,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
    max_ss_steps_per_rung: int = 200,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
    ic_at_target: bool = True,
    rung_callback: Optional[Callable] = None,
    kw_eff_ladder: Optional[Tuple[float, ...]] = None,
    c_s_ladder: Optional[Tuple[float, ...]] = None,
    lambda_hydrolysis_ladder: Optional[Tuple[float, ...]] = None,
) -> AnchorContinuationResult:
    """Solve a single voltage with k0 continuation.

    Builds the context once, sets the IC, then ramps every reaction's
    ``k0`` down to ``initial_scales[0] * k0_targets[j]`` and walks back
    up to production ``k0_targets[j]`` via geometric rungs. Each rung
    runs the SER adaptive-dt SS loop; on failure the ladder inserts a
    geometric midpoint and the orchestrator rolls back ``U``.

    The full ladder loop is wrapped in
    ``firedrake.adjoint.stop_annotating()`` so the adjoint tape does
    not accumulate continuation history (Risk R7).

    Parameters
    ----------
    sp
        :class:`Forward.params.SolverParams` (or compatible 11-tuple)
        already configured for the target voltage. The caller should
        have called ``sp.with_phi_applied(...)`` upstream.
    mesh
        Firedrake mesh.
    k0_targets
        ``{j: k0_target_nondim}`` mapping. The dict's keys must be
        valid indices into ``ctx['nondim']['bv_reactions']``.
    initial_scales
        Geometric ladder, strictly increasing, ending at ``1.0``. The
        first element is the floor.
    max_inserts_per_step
        Per-step cap on ladder midpoint inserts.
    ss_rel_tol, ss_abs_tol, ss_consec
        SS plateau-detection tolerances.
    max_ss_steps_per_rung
        Step cap on the SS loop at each rung.
    dt_init, dt_growth_cap, dt_max_ratio
        SER adaptive-dt knobs.
    ic_at_target
        If ``True`` (default), build the IC at the production
        ``k0_targets`` (Picard sees production rates; may fall back
        to linear-φ on its own oscillation, which is OK because the
        ladder ramps from the floor anyway). If ``False``, build the
        IC after first ramping ``k0`` to the floor — diagnostic mode
        for isolating Picard from rung-1 Newton.
    rung_callback
        Optional ``callback(scale, ok, ctx, rung_diag)`` invoked after
        each rung completes (success or failure).
    kw_eff_ladder
        Optional Phase 6α water-self-ionization ``Kw_eff`` continuation
        ladder.  When ``None`` (default), no Kw_eff ramp is performed
        — required and only meaningful when the form was built with
        ``enable_water_ionization=True`` on ``bv_convergence``.  When
        provided, must be a strictly-increasing positive sequence whose
        last value is ``KW_HAT`` (the production target).  The
        orchestrator walks this ladder OUTSIDE the k0 ladder: at each
        Kw_eff rung the full k0 ladder runs, with each k0 success warm-
        starting the next.  Recommended pattern:
        ``(KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT)``; a
        starting floor of ``0`` is accepted (no Kw_eff continuation
        needed at the floor since the residual reduces to the standard
        H⁺ NP residual).
    c_s_ladder
        Optional Phase 6β v9 Gate 2 Stern capacitance continuation
        ladder, in **physical F/m²** values.  When ``None`` (default),
        no C_S ramp is performed — Stern stays pinned at whatever value
        was set in ``stern_capacitance_f_m2`` at form-build.  When
        provided, must be a strictly-decreasing positive sequence whose
        last value matches the production C_S (e.g.
        ``(1.0, 0.5, 0.25, 0.10)`` for production C_S = 0.10 F/m²).
        The orchestrator walks this ladder OUTER to the k0 ladder and
        INNER to the Kw_eff ladder: at each C_S rung the full k0 ladder
        runs.  When ``kw_eff_ladder`` is also provided, the ordering is
        Kw_eff (outermost) > C_S > k0 (innermost) — but the MVP only
        supports C_S without Kw_eff (combining both is deferred).
        Requires the form to have been built with
        ``stern_capacitance_f_m2 > 0`` (otherwise the helper raises).
    lambda_hydrolysis_ladder
        Optional Phase 6β v9 Gate 3C cation-hydrolysis activation
        ladder.  When ``None`` (default), no λ ramp is performed —
        λ stays pinned at whatever value was on
        ``bv_convergence['lambda_hydrolysis']`` (typically 0).  When
        provided, must be a strictly-increasing positive sequence
        whose last value is ``1.0`` (the production target).  The
        orchestrator walks this ladder OUTER to the k0 ladder; at the
        first rung the full k0 ladder runs (so k0 reaches production
        at λ=floor), then subsequent λ rungs warm-start from the
        previous successful λ at production k0.  Combining with
        ``kw_eff_ladder`` or ``c_s_ladder`` is deferred (raises
        ``NotImplementedError``); each ladder is individually
        supported.  Requires the form to have been built with
        ``enable_cation_hydrolysis=True``.

    Returns
    -------
    AnchorContinuationResult
        ``converged=True`` only if every rung up to ``scale=1.0``
        succeeded. ``ctx`` is live; do not retain past the next
        ``build_context``.

    Raises
    ------
    LadderExhausted
        Mid-ladder if the orchestrator exhausted
        ``max_inserts_per_step`` at a failing rung. The caller should
        catch this and inspect the partial result, which is *not*
        constructed; the caller may interrogate ``ladder.history()``
        if they hold a reference. (For MVP, the result is constructed
        only on success.)
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    from .dispatch import build_context, build_forms, set_initial_conditions
    from .observables import _build_bv_observable_form
    from .grid_per_voltage import snapshot_U, restore_U, make_run_ss

    # ----- 1. Validate ladder shapes up-front (cheap before context build).
    if not k0_targets:
        raise ValueError("k0_targets must be non-empty")
    for j, k_target in k0_targets.items():
        if not (k_target > 0.0):
            raise ValueError(
                f"k0_targets[{j}] = {k_target!r} must be > 0"
            )

    # Phase 6β v9 Gate 2 — disallow combining the C_S and Kw_eff
    # ladders in the MVP (both individually supported).  Validate
    # before building the (expensive) Firedrake context so misuses
    # surface as a fast error.
    if c_s_ladder is not None and kw_eff_ladder is not None:
        raise NotImplementedError(
            "Combining c_s_ladder + kw_eff_ladder is deferred for the "
            "Gate 2 MVP.  Run them independently."
        )
    # Phase 6β v9 Gate 3C — disallow combining lambda_hydrolysis ladder
    # with the other two.  Each is independently supported.
    if lambda_hydrolysis_ladder is not None and (
        kw_eff_ladder is not None or c_s_ladder is not None
    ):
        raise NotImplementedError(
            "Combining lambda_hydrolysis_ladder with kw_eff_ladder or "
            "c_s_ladder is deferred for the Gate 3C MVP.  Run them "
            "independently — typical pattern is anchor + warm-walk "
            "with kw_eff_ladder at λ=0, then a separate λ ramp at the "
            "Gate 4 target voltage."
        )

    # ----- 2. Build context + forms, set IC at target k0 (or floor).
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)

    # Cross-check that every requested reaction index is in range.
    rxn_count = len(ctx.get("nondim", {}).get("bv_reactions", []))
    for j in k0_targets:
        if j < 0 or j >= rxn_count:
            raise IndexError(
                f"k0_targets index {j} out of range "
                f"(N_reactions={rxn_count})"
            )

    if not ic_at_target:
        # Diagnostic mode: ramp k0 down BEFORE the IC so Picard sees
        # tiny rates (converges trivially), then build IC at the floor.
        floor = initial_scales[0]
        for j, k_target in k0_targets.items():
            set_reaction_k0_model(ctx, j, floor * k_target)
    set_initial_conditions(ctx, sp)

    # ----- 3. Ramp k0 to the floor for the first Newton solve. Even if
    # the IC ran at production k0, the FIRST SS must see floor-k0 so the
    # ramp gains traction.
    floor = initial_scales[0]
    for j, k_target in k0_targets.items():
        set_reaction_k0_model(ctx, j, floor * k_target)

    # ----- 4. Build the Newton solver + observable for SS detection.
    params_block = sp[10] if hasattr(sp, "__getitem__") else {}
    items = (
        params_block.items() if isinstance(params_block, dict) else []
    )
    solve_opts = {k: v for k, v in items if k not in NON_PETSC_KEYS}
    solve_opts.setdefault("snes_error_if_not_converged", True)
    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
    )
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solve_opts
    )
    ctx["_last_solver"] = solver
    of_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0
    )

    run_ss = make_run_ss(
        ctx=ctx,
        solver=solver,
        of_cd=of_cd,
        dt_init=dt_init,
        dt_growth_cap=dt_growth_cap,
        dt_max_ratio=dt_max_ratio,
        ss_rel_tol=ss_rel_tol,
        ss_abs_tol=ss_abs_tol,
        ss_consec=ss_consec,
    )

    # ----- 5. Initial baseline snapshot for first-rung rollback.
    last_success_snap = snapshot_U(ctx["U"])

    rungs: List[Dict[str, Any]] = []

    def _run_gamma_picard(label: str) -> None:
        """Outer Picard for ``Γ_MOH`` after a successful inner solve.

        Phase 6β v9 Gate 3: Γ is an external R-space coefficient
        updated by ``update_gamma_from_solution`` between continuation
        rungs.  The closed-form formula has Γ depending on
        boundary-averaged c_M, c_H — so once Γ changes, the FE
        solution must be re-solved.  We iterate until either Γ
        stabilises (relative tolerance ``1e-4``) or the cap of
        ``8`` Picard iterations is hit.

        At λ=0 (the disabled-feature pin) the formula returns Γ=0
        immediately and the loop is a no-op.
        """
        if ctx.get("cation_hydrolysis") is None:
            return
        from .cation_hydrolysis import update_gamma_from_solution
        picard_max_iters = 8
        picard_rel_tol = 1e-4
        gamma_history = []
        for _it in range(picard_max_iters):
            gamma_old = float(ctx["cation_hydrolysis"].gamma_func)
            gamma_new = update_gamma_from_solution(ctx)
            gamma_history.append(float(gamma_new))
            lam_active = float(
                ctx["cation_hydrolysis"].lambda_hydrolysis_func
            )
            if lam_active == 0.0:
                break
            rel = abs(gamma_new - gamma_old) / max(
                abs(gamma_new), abs(gamma_old), 1e-30
            )
            if rel < picard_rel_tol:
                break
            ok = run_ss(max_ss_steps_per_rung)
            if not ok:
                # Picard step de-stabilised; stop and accept the
                # previous Γ.  Continuation orchestrator will surface
                # this as a non-converged outcome.
                break
        ctx[f"gamma_picard_history_{label}"] = gamma_history

    def _run_k0_ladder(
        rung_label: str,
    ) -> Tuple[bool, AdaptiveLadder, Optional[tuple]]:
        """Run a fresh k0 ladder under the *current* Kw_eff.

        Mutates ``rungs`` (appends per-rung diagnostics, prefixed with
        ``rung_label`` so Kw_eff outer rungs are traceable).  Updates
        ``last_success_snap`` via the closure.

        Returns ``(converged, ladder, last_snap)``.  ``last_snap`` is the
        snapshot at the last successful k0 rung, or ``None`` if no rung
        succeeded — the caller uses it to roll back when the outer
        Kw_eff ladder rolls back.
        """
        nonlocal last_success_snap
        k0_lad = AdaptiveLadder(
            initial_scales=tuple(initial_scales),
            max_inserts_per_step=max_inserts_per_step,
        )
        k0_last_ok_snap: Optional[tuple] = None
        try:
            while not k0_lad.is_done():
                k0_scale = k0_lad.current_scale
                for j, k_target in k0_targets.items():
                    set_reaction_k0_model(ctx, j, k0_scale * k_target)
                ok = run_ss(max_ss_steps_per_rung)
                rung_diag: Dict[str, Any] = {
                    "rung_label": rung_label,
                    "scale": float(k0_scale),
                    "snes_converged": bool(ok),
                }
                if ok:
                    try:
                        rung_diag["cd_observable"] = float(fd.assemble(of_cd))
                    except Exception as exc:
                        rung_diag["cd_observable_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                rungs.append(rung_diag)

                if rung_callback is not None:
                    try:
                        rung_callback(k0_scale, ok, ctx, rung_diag)
                    except Exception as exc:
                        rung_diag["rung_callback_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                if ok:
                    snap = snapshot_U(ctx["U"])
                    last_success_snap = snap
                    k0_last_ok_snap = snap
                    k0_lad.record_success()
                else:
                    if not k0_lad.record_failure_and_insert():
                        return False, k0_lad, k0_last_ok_snap
                    # Rollback to the previous k0-rung snapshot.
                    restore_U(last_success_snap, ctx["U"], ctx["U_prev"])
        except RuntimeError:
            # AdaptiveLadder raises on misuse only; treat as ladder-exhausted.
            return False, k0_lad, k0_last_ok_snap
        return True, k0_lad, k0_last_ok_snap

    # ----- 6. Phase 6α — optional Kw_eff outer ladder.  Kw_eff = 0 is
    # treated as a "no water source" floor (residual reduces to standard
    # H+ NP) and runs the full k0 ladder as if water-ionization were off.
    # Only the positive ladder rungs are walked by AdaptiveLadder.
    #
    # Phase 6β v9 Gate 2 — optional C_S ladder is OUTER to k0 ladder
    # and INNER to Kw_eff ladder.  Combining both is deferred (raises
    # NotImplementedError); each can be used independently.
    converged_to_target = False
    final_k0_ladder: Optional[AdaptiveLadder] = None

    with adj.stop_annotating():
        if c_s_ladder is not None:
            # Validate C_S ladder shape.
            cs_seq = [float(v) for v in c_s_ladder]
            if not cs_seq:
                raise ValueError(
                    "c_s_ladder must be non-empty when provided"
                )
            for v in cs_seq:
                if v <= 0.0:
                    raise ValueError(
                        f"c_s_ladder entries must be > 0 (got {v!r}); "
                        "the C_S = 0 limit is the no-Stern Dirichlet, a "
                        "build-time decision."
                    )
            for a, b in zip(cs_seq[:-1], cs_seq[1:]):
                if not (b < a):
                    raise ValueError(
                        "c_s_ladder must be strictly monotonic decreasing "
                        f"(got {cs_seq!r}); the ladder ramps from a "
                        "Stern-relaxed start down to the production target."
                    )

            # Walk C_S as outer loop, run a fresh k0 ladder at each rung.
            cs_history: List[Tuple[float, str]] = []
            cs_last_success_snap = last_success_snap
            for cs_val in cs_seq:
                set_stern_capacitance_model(ctx, cs_val)
                t_rung = time.time()
                ok, k0_lad, _ = _run_k0_ladder(f"cs={cs_val:.3e}")
                cs_rung_diag: Dict[str, Any] = {
                    "rung_label": f"cs_outer={cs_val:.3e}",
                    "c_s_f_m2": float(cs_val),
                    "k0_ladder_converged": bool(ok),
                    "wall_seconds": float(time.time() - t_rung),
                }
                if ok:
                    try:
                        cs_rung_diag["cd_observable"] = float(
                            fd.assemble(of_cd)
                        )
                    except Exception as exc:
                        cs_rung_diag["cd_observable_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                rungs.append(cs_rung_diag)
                cs_history.append((float(cs_val), "ok" if ok else "fail"))
                if rung_callback is not None:
                    try:
                        rung_callback(cs_val, ok, ctx, cs_rung_diag)
                    except Exception as exc:
                        cs_rung_diag["rung_callback_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                if not ok:
                    # Roll back to the last successful C_S rung; the
                    # MVP does NOT auto-insert C_S midpoints.  Caller
                    # can densify ``c_s_ladder`` and retry.
                    if cs_last_success_snap is not None:
                        restore_U(
                            cs_last_success_snap, ctx["U"], ctx["U_prev"]
                        )
                    final_k0_ladder = k0_lad
                    ctx["c_s_ladder_history"] = list(cs_history)
                    raise LadderExhausted(
                        f"c_s ladder exhausted at C_S={cs_val:.3e} F/m²; "
                        f"cs_history={cs_history!r}"
                    )
                cs_last_success_snap = snapshot_U(ctx["U"])
                last_success_snap = cs_last_success_snap
                final_k0_ladder = k0_lad
            converged_to_target = True
            ctx["c_s_ladder_history"] = list(cs_history)
        elif lambda_hydrolysis_ladder is not None:
            # Phase 6β v9 Gate 3C — cation hydrolysis activation ladder.
            # Mirrors the kw_eff_ladder pattern: floor branch optionally
            # at λ=0, then walks positive λ rungs at production k0.  At
            # each positive λ rung we run an OUTER PICARD on Γ (the
            # cation hydrolysis surface coverage) — Newton solves the
            # FE system at fixed Γ, then ``update_gamma_from_solution``
            # recomputes Γ from the converged boundary integrals; the
            # iteration runs until Γ stops changing.
            from .cation_hydrolysis import update_gamma_from_solution

            lam_seq = list(lambda_hydrolysis_ladder)
            if not lam_seq:
                raise ValueError(
                    "lambda_hydrolysis_ladder must be non-empty when provided"
                )
            lam_target = float(lam_seq[-1])
            if lam_target != 1.0:
                raise ValueError(
                    f"lambda_hydrolysis_ladder must end at the production "
                    f"target 1.0 (got {lam_target!r})"
                )
            for v in lam_seq:
                if float(v) < 0.0:
                    raise ValueError(
                        f"lambda_hydrolysis_ladder entries must be >= 0 "
                        f"(got {v!r})"
                    )
                if float(v) > 1.0:
                    raise ValueError(
                        f"lambda_hydrolysis_ladder entries must be <= 1 "
                        f"(got {v!r})"
                    )
            for a, b in zip(lam_seq[:-1], lam_seq[1:]):
                if not (float(b) > float(a)):
                    raise ValueError(
                        "lambda_hydrolysis_ladder must be strictly "
                        f"monotonic increasing (got {lam_seq!r})"
                    )

            # Floor branch: if the ladder includes 0.0, run the full k0
            # ladder there first (λ=0 → Γ pinned to 0 by the explicit
            # branch in update_gamma_from_solution; matches the
            # disabled-feature baseline).
            lam_positive = [float(v) for v in lam_seq if float(v) > 0.0]
            if float(lam_seq[0]) == 0.0:
                set_reaction_lambda_hydrolysis_model(ctx, 0.0)
                # Force Γ = 0 explicitly before the floor solve.
                ctx["cation_hydrolysis"].gamma_func.assign(0.0)
                ok, k0_lad, _ = _run_k0_ladder("lambda=0.0")
                final_k0_ladder = k0_lad
                if not ok:
                    raise LadderExhausted(
                        f"k0 ladder exhausted at λ=0; "
                        f"history={k0_lad.history()!r}"
                    )

            lam_scales = tuple(v / lam_target for v in lam_positive)
            if not lam_scales or lam_scales[-1] != 1.0:
                raise ValueError(
                    "lambda_hydrolysis_ladder positive entries must end "
                    f"at the target (last entry / target != 1.0; "
                    f"ladder={lam_seq!r})"
                )
            lam_ladder = AdaptiveLadder(
                initial_scales=lam_scales,
                max_inserts_per_step=max_inserts_per_step,
            )
            lam_last_success_snap = last_success_snap

            # Pin k0 at production for every λ>0 rung — same rationale
            # as the kw_eff ladder.
            for j, k_target in k0_targets.items():
                set_reaction_k0_model(ctx, j, float(k_target))

            picard_max_iters = 8
            picard_rel_tol = 1e-4

            while not lam_ladder.is_done():
                lam_scale = lam_ladder.current_scale
                lam_val = lam_scale * lam_target
                set_reaction_lambda_hydrolysis_model(ctx, lam_val)

                t_rung = time.time()
                # Outer Picard for Γ.  Each Newton solve at fixed Γ
                # produces a converged FE state; the Picard step then
                # updates Γ from the boundary integrals.  Convergence
                # is measured on the relative change in Γ.
                gamma_picard_history: List[float] = []
                ok = False
                for picard_iter in range(picard_max_iters):
                    ok = run_ss(max_ss_steps_per_rung)
                    if not ok:
                        break
                    gamma_old = float(
                        ctx["cation_hydrolysis"].gamma_func
                    )
                    gamma_new = update_gamma_from_solution(ctx)
                    gamma_picard_history.append(float(gamma_new))
                    rel = abs(gamma_new - gamma_old) / max(
                        abs(gamma_new), abs(gamma_old), 1e-30
                    )
                    if rel < picard_rel_tol:
                        break
                rung_diag: Dict[str, Any] = {
                    "rung_label": f"lambda={lam_val:.3e}",
                    "scale": float(lam_scale),
                    "lambda_hydrolysis": float(lam_val),
                    "snes_converged": bool(ok),
                    "gamma_picard_iters": len(gamma_picard_history),
                    "gamma_picard_history": gamma_picard_history,
                    "gamma_final": (
                        gamma_picard_history[-1]
                        if gamma_picard_history
                        else None
                    ),
                    "wall_seconds": float(time.time() - t_rung),
                }
                if ok:
                    try:
                        rung_diag["cd_observable"] = float(fd.assemble(of_cd))
                    except Exception as exc:
                        rung_diag["cd_observable_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                if ok:
                    # Phase 6β v10a diagnostics — collected after Newton
                    # converged so boundary integrals reflect the final
                    # FE state (not a stale Picard mid-step).
                    try:
                        from .cation_hydrolysis import (
                            collect_v10a_rung_diagnostics,
                        )
                        rung_diag.update(collect_v10a_rung_diagnostics(ctx))
                    except Exception as exc:
                        rung_diag["v10a_diagnostics_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                rungs.append(rung_diag)

                if rung_callback is not None:
                    try:
                        rung_callback(lam_scale, ok, ctx, rung_diag)
                    except Exception as exc:
                        rung_diag["rung_callback_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                if ok:
                    lam_last_success_snap = snapshot_U(ctx["U"])
                    last_success_snap = lam_last_success_snap
                    lam_ladder.record_success()
                    if lam_ladder.is_done():
                        converged_to_target = True
                else:
                    if not lam_ladder.record_failure_and_insert():
                        raise LadderExhausted(
                            f"λ_hydrolysis ladder exhausted at "
                            f"λ={lam_val:.3e}; "
                            f"lam_history={lam_ladder.history()!r}"
                        )
                    restore_U(
                        lam_last_success_snap, ctx["U"], ctx["U_prev"]
                    )

            ctx["lambda_hydrolysis_ladder_history"] = lam_ladder.history()
        elif kw_eff_ladder is None:
            # No water-ionization continuation requested.  Run a single
            # k0 ladder under whatever Kw_eff the form was built with
            # (typically 0 if water-ionization is disabled).
            ok, k0_lad, _ = _run_k0_ladder("kw=default")
            final_k0_ladder = k0_lad
            converged_to_target = ok
            if not ok:
                raise LadderExhausted(
                    f"k0 ladder exhausted; history={k0_lad.history()!r}"
                )
            _run_gamma_picard("bare")
        else:
            kw_lad_seq = list(kw_eff_ladder)
            if not kw_lad_seq:
                raise ValueError("kw_eff_ladder must be non-empty when provided")
            kw_target = float(kw_lad_seq[-1])
            if not (kw_target > 0.0):
                raise ValueError(
                    f"kw_eff_ladder must end at a positive Kw_eff target "
                    f"(got {kw_target!r})"
                )
            for v in kw_lad_seq:
                if float(v) < 0.0:
                    raise ValueError(
                        f"kw_eff_ladder entries must be >= 0 (got {v!r})"
                    )
            for a, b in zip(kw_lad_seq[:-1], kw_lad_seq[1:]):
                if not (float(b) > float(a)):
                    raise ValueError(
                        "kw_eff_ladder must be strictly monotonic increasing "
                        f"(got {kw_lad_seq!r})"
                    )

            # Floor branch: if the ladder includes 0.0, run the full k0
            # ladder there first (Kw_eff = 0 → standard H+ NP residual).
            kw_positive = [float(v) for v in kw_lad_seq if float(v) > 0.0]
            if float(kw_lad_seq[0]) == 0.0:
                set_reaction_kw_eff_model(ctx, 0.0)
                ok, k0_lad, _ = _run_k0_ladder("kw=0.0")
                final_k0_ladder = k0_lad
                if not ok:
                    raise LadderExhausted(
                        f"k0 ladder exhausted at Kw_eff=0; "
                        f"history={k0_lad.history()!r}"
                    )

            kw_scales = tuple(v / kw_target for v in kw_positive)
            if not kw_scales or kw_scales[-1] != 1.0:
                raise ValueError(
                    "kw_eff_ladder positive entries must end at the target "
                    f"(last entry / target != 1.0; ladder={kw_lad_seq!r})"
                )
            kw_ladder = AdaptiveLadder(
                initial_scales=kw_scales,
                max_inserts_per_step=max_inserts_per_step,
            )
            kw_last_success_snap = last_success_snap

            # Pin k0 at production for every kw>0 rung — at this point
            # the kw=0 floor has already converged at production k0, so
            # incremental water turns into a small Newton perturbation
            # rather than a full restart.  An inner k0 ladder per kw
            # rung is not necessary (and adds 4-5× wall per rung).
            for j, k_target in k0_targets.items():
                set_reaction_k0_model(ctx, j, float(k_target))

            while not kw_ladder.is_done():
                kw_scale = kw_ladder.current_scale
                kw_val = kw_scale * kw_target
                set_reaction_kw_eff_model(ctx, kw_val)

                t_rung = time.time()
                ok = run_ss(max_ss_steps_per_rung)
                rung_diag: Dict[str, Any] = {
                    "rung_label": f"kw={kw_val:.3e}",
                    "scale": float(kw_scale),
                    "kw_eff": float(kw_val),
                    "snes_converged": bool(ok),
                    "wall_seconds": float(time.time() - t_rung),
                }
                if ok:
                    try:
                        rung_diag["cd_observable"] = float(fd.assemble(of_cd))
                    except Exception as exc:
                        rung_diag["cd_observable_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
                rungs.append(rung_diag)

                if rung_callback is not None:
                    try:
                        rung_callback(kw_scale, ok, ctx, rung_diag)
                    except Exception as exc:
                        rung_diag["rung_callback_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                if ok:
                    kw_last_success_snap = snapshot_U(ctx["U"])
                    last_success_snap = kw_last_success_snap
                    kw_ladder.record_success()
                    if kw_ladder.is_done():
                        converged_to_target = True
                else:
                    if not kw_ladder.record_failure_and_insert():
                        raise LadderExhausted(
                            f"Kw_eff ladder exhausted at Kw_eff={kw_val:.3e}; "
                            f"kw_history={kw_ladder.history()!r}"
                        )
                    # Roll back U to the last successful Kw_eff rung
                    # before retrying the midpoint-inserted Kw_eff.
                    restore_U(
                        kw_last_success_snap, ctx["U"], ctx["U_prev"]
                    )

            # Stash the Kw_eff history alongside the k0 history so callers
            # can reconstruct provenance.
            ctx["kw_eff_ladder_history"] = kw_ladder.history()
            # Phase 6β v9 Gate 3D — outer Picard for Γ at the final
            # Kw_eff target.  At λ=0 (the default initial value when
            # not running the lambda ladder explicitly) this is a no-op.
            if converged_to_target:
                _run_gamma_picard("kw")

    return AnchorContinuationResult(
        converged=bool(converged_to_target),
        U_data=last_success_snap if converged_to_target else None,
        ladder_history=(
            final_k0_ladder.history() if final_k0_ladder is not None else []
        ),
        rungs=rungs,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Phase 6β v9 Gate 4B — solve_lambda_ramp_from_warm_start
# ---------------------------------------------------------------------------


def solve_lambda_ramp_from_warm_start(
    sp,
    *,
    mesh,
    U_warmstart: tuple,
    k0_targets: Dict[int, float],
    lambda_hydrolysis_ladder: Tuple[float, ...],
    parameter_overrides: Optional[Dict[str, Any]] = None,
    reconverge_at_ss: bool = True,
    max_inserts_per_step: int = 4,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
    max_ss_steps_per_rung: int = 200,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
    rung_callback: Optional[Callable] = None,
) -> "AnchorContinuationResult":
    """Run a λ_hydrolysis ramp on a fresh ctx warm-started from a U snapshot.

    Phase 6β v9 Gate 4B optimization helper.  Designed for sweeps
    where the anchor + warm-walk pipeline has already produced a
    converged solution at some target voltage and we want to amortize
    that cost across many parameter combinations.  Each combination
    gets:

      1. A fresh ctx + form (cheap: ~10–30 s)
      2. ``U`` restored from the cached snapshot (no IC, no k0 ramp)
      3. Optional parameter overrides applied to the new ctx (Stern
         capacitance, kinetic rates, r_H_El, etc.)
      4. Optional SS re-converge (default ``True``) — needed when
         parameter changes affect the FE residual at fixed V_RHE
         (e.g. ``C_S`` change)
      5. λ ladder walked with outer Picard for Γ at every rung

    Note: the form is rebuilt because some parameters (the bv_bc
    config block, the IC routine path) are baked at form-build time.
    Reusing the existing solver_params with parameter overrides via
    ``with_solver_options`` is the cleanest way to swap them.

    Parameters
    ----------
    sp
        ``SolverParams`` already configured for the target voltage
        and combination parameters.  This is the fresh sp built per
        combination — caller threads in the new C_S / r_H_El /
        k_des etc. via the standard ``make_bv_solver_params`` factory.
    mesh
        Firedrake mesh shared with the cached snapshot.  Must match
        ``ctx['U'].function_space().dim()`` of the snapshot or the
        restore will raise.
    U_warmstart
        ``tuple(d.data_ro.copy() for d in U.dat)`` from a
        prior converged solve.  Restored verbatim; must match the
        new ctx's mixed-space layout.
    k0_targets
        Production nondim k0 per reaction index.  Pinned at production
        before the SS reconverge (no ramp).
    lambda_hydrolysis_ladder
        Strictly-increasing positive sequence ending at 1.0.  Same
        validation rules as ``solve_anchor_with_continuation``.
    parameter_overrides
        Optional ``{accessor_name: value}`` dict applied AFTER the
        warm-start and BEFORE the optional SS re-converge.  Supported
        accessor names map to ``set_reaction_*_model``:
        ``"stern_capacitance_f_m2"``, ``"k_hyd"``, ``"k_prot"``,
        ``"k_des"``, ``"delta_ohp_hat"``, ``"r_H_El_pm"``,
        ``"kw_eff_hat"``.  Useful when the same ``sp`` is reused
        across combinations and only specific knobs need swapping.
    reconverge_at_ss
        When True (default), runs ``run_ss(max_ss_steps_per_rung)``
        before the λ ramp to absorb any parameter-induced residual
        change.  Set False to skip when the warm-start is exact for
        the new parameters (e.g. only k_des changed — k_des doesn't
        appear in the FE residual).
    rung_callback
        Optional ``callback(scale, ok, ctx, rung_diag)`` invoked
        after each λ rung.

    Returns
    -------
    AnchorContinuationResult
        Same shape as ``solve_anchor_with_continuation`` so callers
        can reuse the diagnostic plumbing.

    Raises
    ------
    LadderExhausted
        If the SS re-converge or any λ rung fails.
    ValueError
        Mesh DOF mismatch, ladder validation failure, or unknown
        ``parameter_overrides`` key.
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    from .dispatch import build_context, build_forms
    from .observables import _build_bv_observable_form
    from .grid_per_voltage import snapshot_U, restore_U, make_run_ss
    from .cation_hydrolysis import (
        clamp_gamma_to_max,
        collect_v10a_rung_diagnostics,
        update_gamma_from_solution,
    )

    # ----- Validate ladder up-front
    lam_seq = list(lambda_hydrolysis_ladder)
    if not lam_seq:
        raise ValueError(
            "lambda_hydrolysis_ladder must be non-empty when provided"
        )
    lam_target = float(lam_seq[-1])
    if lam_target != 1.0:
        raise ValueError(
            f"lambda_hydrolysis_ladder must end at 1.0 (got {lam_target!r})"
        )
    for v in lam_seq:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(
                f"lambda_hydrolysis_ladder entries must be in [0, 1] "
                f"(got {v!r})"
            )
    for a, b in zip(lam_seq[:-1], lam_seq[1:]):
        if not (float(b) > float(a)):
            raise ValueError(
                "lambda_hydrolysis_ladder must be strictly monotonic "
                f"increasing (got {lam_seq!r})"
            )

    # ----- Build fresh ctx + form
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)

    # ----- Restore U from snapshot (mesh-DOF assertion)
    U = ctx["U"]
    U_prev = ctx["U_prev"]
    if U.function_space().dim() != sum(arr.size for arr in U_warmstart):
        raise ValueError(
            "solve_lambda_ramp_from_warm_start: U_warmstart DOF count "
            f"({sum(arr.size for arr in U_warmstart)}) does not match "
            f"new ctx U DOFs ({U.function_space().dim()})"
        )
    restore_U(U_warmstart, U, U_prev)

    # Phase 6β v10a — silent warm-restart clamp on Γ to absorb the case
    # where the cached snapshot was taken under a different Γ_max
    # (e.g. parameter-sweep combinations).  Picard runs after this with
    # a Γ already inside [0, Γ_max], so the first vacancy factor is
    # well-posed.  No-op when feature disabled.
    clamp_gamma_to_max(ctx)

    # ----- Apply parameter overrides
    if parameter_overrides is None:
        parameter_overrides = {}
    _OVERRIDE_DISPATCH = {
        "stern_capacitance_f_m2": set_stern_capacitance_model,
        "kw_eff_hat": set_reaction_kw_eff_model,
        "lambda_hydrolysis": set_reaction_lambda_hydrolysis_model,
        "k_hyd": set_reaction_k_hyd_model,
        "k_prot": set_reaction_k_prot_model,
        "k_des": set_reaction_k_des_model,
        "delta_ohp_hat": set_reaction_delta_ohp_model,
        "r_H_El_pm": set_reaction_r_H_El_pm_model,
        "gamma_max_nondim": set_reaction_gamma_max_model,
        "beta_offset_pm2": set_reaction_beta_offset_pm2_model,
    }
    for key, value in parameter_overrides.items():
        if key not in _OVERRIDE_DISPATCH:
            raise ValueError(
                f"Unknown parameter_overrides key {key!r}; supported: "
                f"{list(_OVERRIDE_DISPATCH)}"
            )
        _OVERRIDE_DISPATCH[key](ctx, float(value))

    # Re-clamp Γ after parameter overrides — if the override included
    # ``gamma_max_nondim`` and the new cap is tighter than the cached
    # Γ, Newton needs a feasible vacancy factor on entry.
    clamp_gamma_to_max(ctx)

    # ----- Pin k0 at production (no ramp; we're warm-started)
    rxn_count = len(ctx.get("nondim", {}).get("bv_reactions", []))
    for j, k_target in k0_targets.items():
        if j < 0 or j >= rxn_count:
            raise IndexError(
                f"k0_targets index {j} out of range "
                f"(N_reactions={rxn_count})"
            )
        set_reaction_k0_model(ctx, j, float(k_target))

    # ----- Build solver + SS loop
    params_block = sp[10] if hasattr(sp, "__getitem__") else {}
    items = (
        params_block.items() if isinstance(params_block, dict) else []
    )
    solve_opts = {k: v for k, v in items if k not in NON_PETSC_KEYS}
    solve_opts.setdefault("snes_error_if_not_converged", True)
    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
    )
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solve_opts
    )
    ctx["_last_solver"] = solver
    of_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0
    )
    run_ss = make_run_ss(
        ctx=ctx, solver=solver, of_cd=of_cd,
        dt_init=dt_init, dt_growth_cap=dt_growth_cap,
        dt_max_ratio=dt_max_ratio,
        ss_rel_tol=ss_rel_tol, ss_abs_tol=ss_abs_tol,
        ss_consec=ss_consec,
    )

    rungs: List[Dict[str, Any]] = []
    last_success_snap = snapshot_U(ctx["U"])

    with adj.stop_annotating():
        # ----- Optional SS re-converge to absorb parameter changes
        if reconverge_at_ss:
            t0 = time.time()
            ok = run_ss(max_ss_steps_per_rung)
            rungs.append({
                "rung_label": "warm_reconverge",
                "snes_converged": bool(ok),
                "wall_seconds": float(time.time() - t0),
            })
            if not ok:
                raise LadderExhausted(
                    "warm-start SS re-converge failed; the parameter "
                    "overrides may have moved the system too far from "
                    "the cached snapshot."
                )
            last_success_snap = snapshot_U(ctx["U"])

        # ----- λ ladder
        lam_positive = [float(v) for v in lam_seq if float(v) > 0.0]
        if float(lam_seq[0]) == 0.0:
            set_reaction_lambda_hydrolysis_model(ctx, 0.0)
            ctx["cation_hydrolysis"].gamma_func.assign(0.0)
            t0 = time.time()
            ok = run_ss(max_ss_steps_per_rung)
            rungs.append({
                "rung_label": "lambda=0.0",
                "lambda_hydrolysis": 0.0,
                "snes_converged": bool(ok),
                "wall_seconds": float(time.time() - t0),
            })
            if not ok:
                raise LadderExhausted(
                    "λ=0 floor solve failed after warm restore"
                )
            last_success_snap = snapshot_U(ctx["U"])

        if not lam_positive:
            return AnchorContinuationResult(
                converged=True,
                U_data=last_success_snap,
                ladder_history=[],
                rungs=rungs,
                ctx=ctx,
            )

        lam_scales = tuple(v / lam_target for v in lam_positive)
        if not lam_scales or lam_scales[-1] != 1.0:
            raise ValueError(
                "lambda_hydrolysis_ladder positive entries must end "
                f"at the target (last entry / target != 1.0; "
                f"ladder={lam_seq!r})"
            )
        lam_ladder = AdaptiveLadder(
            initial_scales=lam_scales,
            max_inserts_per_step=max_inserts_per_step,
            warm_start_floor=0.0,
        )

        picard_max_iters = 8
        picard_rel_tol = 1e-4

        while not lam_ladder.is_done():
            lam_scale = lam_ladder.current_scale
            lam_val = lam_scale * lam_target
            set_reaction_lambda_hydrolysis_model(ctx, lam_val)

            t_rung = time.time()
            gamma_history: List[float] = []
            ok = False
            for _picard_iter in range(picard_max_iters):
                ok = run_ss(max_ss_steps_per_rung)
                if not ok:
                    break
                gamma_old = float(ctx["cation_hydrolysis"].gamma_func)
                gamma_new = update_gamma_from_solution(ctx)
                gamma_history.append(float(gamma_new))
                rel = abs(gamma_new - gamma_old) / max(
                    abs(gamma_new), abs(gamma_old), 1e-30
                )
                if rel < picard_rel_tol:
                    break

            try:
                cd_obs = float(fd.assemble(of_cd))
            except Exception:
                cd_obs = None

            rung_diag: Dict[str, Any] = {
                "rung_label": f"lambda={lam_val:.3e}",
                "scale": float(lam_scale),
                "lambda_hydrolysis": float(lam_val),
                "snes_converged": bool(ok),
                "gamma_picard_iters": len(gamma_history),
                "gamma_picard_history": gamma_history,
                "gamma_final": (
                    gamma_history[-1] if gamma_history else None
                ),
                "cd_observable": cd_obs,
                "wall_seconds": float(time.time() - t_rung),
            }
            if ok:
                try:
                    rung_diag.update(collect_v10a_rung_diagnostics(ctx))
                except Exception as exc:
                    rung_diag["v10a_diagnostics_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            rungs.append(rung_diag)

            if rung_callback is not None:
                try:
                    rung_callback(lam_scale, ok, ctx, rung_diag)
                except Exception as exc:
                    rung_diag["rung_callback_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            if ok:
                last_success_snap = snapshot_U(ctx["U"])
                lam_ladder.record_success()
            else:
                if not lam_ladder.record_failure_and_insert():
                    raise LadderExhausted(
                        f"λ_hydrolysis ramp exhausted at λ={lam_val:.3e}; "
                        f"history={lam_ladder.history()!r}"
                    )
                restore_U(last_success_snap, ctx["U"], ctx["U_prev"])

        ctx["lambda_hydrolysis_ladder_history"] = lam_ladder.history()

    return AnchorContinuationResult(
        converged=True,
        U_data=last_success_snap,
        ladder_history=lam_ladder.history(),
        rungs=rungs,
        ctx=ctx,
    )


__all__ = [
    "NON_PETSC_KEYS",
    "AnchorContinuationResult",
    "LadderExhausted",
    "PreconvergedAnchor",
    "extract_preconverged_anchor",
    "AdaptiveLadder",
    "set_reaction_k0_model",
    "get_reaction_k0_model",
    "set_reaction_kw_eff_model",
    "get_reaction_kw_eff_model",
    "set_stern_capacitance_model",
    "get_stern_capacitance_model",
    # Phase 6β v9 Gate 3C — cation hydrolysis accessors.
    "set_reaction_lambda_hydrolysis_model",
    "get_reaction_lambda_hydrolysis_model",
    "set_reaction_k_hyd_model",
    "set_reaction_k_prot_model",
    "set_reaction_k_des_model",
    "set_reaction_delta_ohp_model",
    "set_reaction_r_H_El_pm_model",
    # Phase 6β v10a — Langmuir saturation cap accessor.
    "set_reaction_gamma_max_model",
    # Phase 6β step 10 Phase D — Δ_β offset accessor.
    "set_reaction_beta_offset_pm2_model",
    "solve_lambda_ramp_from_warm_start",
    "solve_anchor_with_continuation",
]
