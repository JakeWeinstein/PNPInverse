"""Outer-Picard wrapper for the Jithin Eq 4.31 boundary closure.

Continuum-MPNP analog of Jithin's spectral integro-differential closure
for the neutral O₂ boundary supply.  Wraps ``make_run_ss`` with a Picard
factory that interleaves boundary-supply ξ updates between steady-state
Newton solves at every warm-walk substep and every anchor/Stern/k₀
ladder rung.

Math (z=0 cathodic species, code convention):

    ξ = c_OHP_hat / θ_OHP                              (supply variable)
    c_eff_hat (in BV)  = θ_OHP · ξ
    log_c_cat (in BV)  = ln(packing) + ln(ξ)           (forms_logc_muh wiring)

    Eq A':  ξ = c_b_hat/θ_b − R_O2_hat · I_hat / D_O2_hat,   R_O2_hat ≥ 0
    Eq B :  ξ_target = (c_b_hat/θ_b) / (1 + K_old · I_hat · θ_OHP / D_O2_hat)
                       where K_old = R_O2_hat / c_eff_hat_old
    Residual: |ξ + R_O2_hat·I_hat/D_O2_hat − c_b_hat/θ_b|   (no θ_OHP factor)

For full design rationale + GPT-critique-loop trace see
``docs/handoffs/CHATGPT_HANDOFF_40_picard-closure-cliff/``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .grid_per_voltage import make_run_ss, restore_U, snapshot_U

# ---------------------------------------------------------------------------
# Configuration + state dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PicardConfig:
    """Ctx-invariant Picard runtime + deck-invariant physical knobs.

    Captures the **constants** of the Picard wrap — knobs that don't
    change between anchor / Stern rungs / per-V continuation.  All
    ctx-dependent objects (xi_funcs, packing_expr, closure_theta_b,
    etc.) are pulled FRESH from each ctx at factory-call time via
    :func:`make_picard_run_ss_factory`.

    Attributes
    ----------
    D_per_species_hat
        Nondim diffusivity per cathodic species index.  Treated as
        deck-invariant for this feature.  The factory adapter asserts
        ``ctx logD`` matches this at call time; raises if stale (e.g.,
        a future logD continuation rung would invalidate the closure).
    electrode_marker
        Mesh facet marker for the OHP boundary (read from
        ``ctx["bv_settings"]["electrode_marker"]`` by the adapter; see
        :func:`make_picard_run_ss_factory`).  Stored here as a sanity
        cross-check.
    Lx_hat
        Nondim x-extent of the 2D mesh (used to convert the volume
        integral ``∫(1/packing) dx_2d`` to the per-y-line integral
        ``I_hat = ∫₀^Ly dy/packing(y)``).
    max_picard_iters, tol_residual, tol_step, tol_state
        Outer-Picard convergence knobs.
    damping_init, damping_min, max_damping_retries
        Log-space damping for the ξ update, halved on inner-Newton
        failure with target (max ``max_damping_retries`` retries before
        the wrap returns False with reason="run_ss_failed_with_target").
    strict_floor
        If True (default), packing-floor engagement (area fraction >
        floor_tol) returns False (the closure formula no longer reflects
        physics when packing is clamped).  False (only for floor-
        sensitivity experiments) lets iteration continue and reports the
        floor hit in the iter record.
    floor_tol
        Threshold on the area fraction of the domain where
        ``theta_inner <= packing_floor·(1+1e-6)``.  Default 1e-10
        (essentially zero, with quadrature-roundoff tolerance).
    xi_floor
        Lower clamp on ξ before ``log(ξ)`` to avoid overflow.  Default
        ``exp(-50) ≈ 2e-22``.  Persistent ξ-floor across ≥2 iters in
        strict mode → False (reason="xi_floored_persistent").
    """

    D_per_species_hat: tuple  # tuple of (species_idx, D_hat) pairs (hashable for frozen dataclass)
    electrode_marker: int
    Lx_hat: float
    max_picard_iters: int = 15
    tol_residual: float = 1e-3
    tol_step: float = 1e-3
    tol_state: float = 1e-4
    damping_init: float = 0.5
    damping_min: float = 0.05
    max_damping_retries: int = 3
    strict_floor: bool = True
    floor_tol: float = 1e-10
    xi_floor: float = math.exp(-50.0)

    def D_dict(self) -> dict[int, float]:
        """Return D_per_species_hat as a dict for indexed access."""
        return dict(self.D_per_species_hat)

    def runtime_kwargs(self) -> dict[str, Any]:
        """Picard runtime knobs as kwargs (for forwarding to make_picard_run_ss)."""
        return {
            "max_picard_iters": self.max_picard_iters,
            "tol_residual": self.tol_residual,
            "tol_step": self.tol_step,
            "tol_state": self.tol_state,
            "damping_init": self.damping_init,
            "damping_min": self.damping_min,
            "max_damping_retries": self.max_damping_retries,
            "strict_floor": self.strict_floor,
            "floor_tol": self.floor_tol,
            "xi_floor": self.xi_floor,
        }


@dataclass(frozen=True)
class StateSnapshot:
    """Picard rollback snapshot: U_snap (via existing snapshot_U) + xi_snap.

    Reuses ``Forward.bv_solver.grid_per_voltage.snapshot_U`` / ``restore_U``
    semantics (per-subfunction tuple of numpy arrays; restore also sets
    ``U_prev = U`` for time-stepping consistency).  No parallel
    serialization path.
    """

    U_snap: tuple
    xi_snap: tuple  # tuple of (species_idx, dof_values_tuple) pairs


@dataclass(frozen=True)
class PicardIterRecord:
    """Per-iter diagnostics for one outer-Picard iteration."""

    iter: int
    xi_per_species: tuple  # ((s, xi_value), ...)
    R_per_reaction: tuple  # ((j, R_j_mean_hat), ...)
    R_O2_total_per_species: tuple  # ((s, R_s_hat), ...)
    theta_OHP_mean: float
    I_hat: float
    residual_per_species: tuple  # ((s, residual), ...)
    step_per_species: tuple  # ((s, |log(new) - log(old)|), ...)
    state_norm: Optional[float]  # None on iter 0 (no prior Picard iter)
    damping: float
    floor_hit_area_frac: float
    min_theta_inner: float
    h_closure_rel_err: Optional[float]  # H+ closure quality diagnostic; None if not computed


@dataclass(frozen=True)
class PicardResult:
    """Outcome of one ``picard_run_ss(max_steps)`` invocation."""

    converged: bool
    n_iters: int
    reason: str  # "converged" | "run_ss_failed_before_picard_target" | "packing_floored"
                 # | "xi_floored_persistent" | "run_ss_failed_with_target" | "max_picard_iters"
    iter_history: tuple  # tuple[PicardIterRecord, ...]


# ---------------------------------------------------------------------------
# State snapshot helpers (wrap existing snapshot_U / restore_U)
# ---------------------------------------------------------------------------


def snapshot_xi(xi_funcs: dict) -> tuple:
    """Copy ξ Function values per species.

    Parameters
    ----------
    xi_funcs : dict[int, fd.Function]
        Mapping species_idx → R-space ``fd.Function`` holding ``log(ξ)``.

    Returns
    -------
    tuple
        Tuple of ``(species_idx, tuple_of_dof_values)`` pairs, sorted
        by species_idx for deterministic ordering.
    """
    out = []
    for s in sorted(xi_funcs.keys()):
        out.append((int(s), tuple(float(v) for v in xi_funcs[s].dat.data_ro)))
    return tuple(out)


def restore_xi(xi_funcs: dict, snap: tuple) -> None:
    """Restore ξ Function values from a snapshot."""
    snap_dict = {int(s): vals for s, vals in snap}
    for s, func in xi_funcs.items():
        vals = snap_dict[int(s)]
        func.dat.data[:] = np.asarray(vals, dtype=float)


def snapshot_state(ctx, xi_funcs: dict) -> StateSnapshot:
    """Snapshot U (via existing snapshot_U) + ξ Functions."""
    return StateSnapshot(
        U_snap=snapshot_U(ctx["U"]),
        xi_snap=snapshot_xi(xi_funcs),
    )


def restore_state(ctx, xi_funcs: dict, snap: StateSnapshot) -> None:
    """Restore U (via existing restore_U, which also sets U_prev=U) + ξ."""
    restore_U(snap.U_snap, ctx["U"], ctx["U_prev"])
    restore_xi(xi_funcs, snap.xi_snap)


# ---------------------------------------------------------------------------
# Picard diagnostics + target update
# ---------------------------------------------------------------------------


def compute_picard_diagnostics(
    *,
    ctx,
    electrode_marker: int,
    Lx_hat: float,
    packing_floor: float,
    packing_expr,
    theta_inner_expr,
    bv_rate_exprs,
) -> dict:
    """Extract surface and volume diagnostics needed for the Picard update.

    Computes:
      - ``theta_OHP_mean = assemble(packing · ds(em)) / assemble(1 · ds(em))``
      - ``I_hat = assemble((1/packing) · dx) / Lx_hat``
      - ``R_j_mean_hat = assemble(R_j · ds(em)) / assemble(1 · ds(em))``
        for every reaction j (one entry per bv_rate_expr)
      - ``floor_hit_area_frac`` over the (theta_inner ≤ floor·(1+ε)) region
      - ``min_theta_inner`` (sampled at electrode mesh nodes)

    All values are returned in nondim units (the solver's native scale).
    """
    import firedrake as fd

    mesh = ctx["mesh"]
    ds = fd.Measure("ds", domain=mesh)
    dx = fd.Measure("dx", domain=mesh)

    ds_em = ds(int(electrode_marker))
    electrode_area = float(fd.assemble(fd.Constant(1.0) * ds_em))
    if electrode_area <= 0.0:
        raise RuntimeError(
            f"electrode boundary measure is {electrode_area} for marker "
            f"{electrode_marker}; cannot compute surface means."
        )

    # Surface mean of packing at electrode
    theta_OHP_mean = float(fd.assemble(packing_expr * ds_em)) / electrode_area

    # Per-y-line integral of 1/packing.  In 2D with x-uniform setup,
    # assemble((1/packing) · dx_2d) = Lx_hat · ∫₀^Ly dy/packing(y).
    domain_area = float(fd.assemble(fd.Constant(1.0) * dx))
    if domain_area <= 0.0:
        raise RuntimeError(f"domain measure is {domain_area}; cannot compute I_hat.")
    I_hat = float(fd.assemble((fd.Constant(1.0) / packing_expr) * dx)) / float(Lx_hat)

    # Per-reaction mean rate at electrode
    R_means = []
    for j, R_j in enumerate(bv_rate_exprs):
        # bv_rate_exprs may contain fd.Constant(0.0) for disabled reactions
        try:
            R_j_total = float(fd.assemble(R_j * ds_em))
        except Exception:
            R_j_total = 0.0
        R_means.append((int(j), R_j_total / electrode_area))

    # Floor-hit area fraction (use theta_inner, the UNCAPPED expression)
    floor_threshold = float(packing_floor) * (1.0 + 1e-6)
    floor_hit_form = fd.conditional(
        theta_inner_expr <= fd.Constant(floor_threshold),
        fd.Constant(1.0),
        fd.Constant(0.0),
    )
    floor_hit_area_frac = float(fd.assemble(floor_hit_form * dx)) / domain_area

    # Min theta_inner: sample over the electrode boundary using a small
    # negative-log proxy.  For diagnostic purposes only.
    try:
        # max_value(packing_floor_neg) doesn't exist in UFL; use a Function
        # interpolation if available.  Fallback: surface integral of -log gives
        # an inverse-weighted average.  Either is fine for diagnostics.
        min_theta_form = (
            fd.assemble(theta_inner_expr * ds_em) / electrode_area
        )
        min_theta_inner = float(min_theta_form)
    except Exception:
        min_theta_inner = float("nan")

    return {
        "theta_OHP_mean": theta_OHP_mean,
        "I_hat": I_hat,
        "R_per_reaction": tuple(R_means),
        "floor_hit_area_frac": floor_hit_area_frac,
        "min_theta_inner": min_theta_inner,
    }


def compute_R_O2_per_species(
    *,
    R_per_reaction: tuple,
    cathodic_stoich: dict,
) -> dict[int, float]:
    """Aggregate molar consumption rate per cathodic species.

    For each species s consumed by reactions j with stoichiometry
    ``stoich[s, j]`` (negative for consumption):

        R_s_hat = Σ_j (−stoich[s, j]) · R_j_mean_hat

    Always ≥ 0 for irreversible cathodic consumption.

    Parameters
    ----------
    R_per_reaction : tuple
        Sequence of (j, R_j_mean_hat) pairs from
        :func:`compute_picard_diagnostics`.
    cathodic_stoich : dict[int, dict[int, int]]
        Maps species_idx → (rxn_idx → stoich).  From
        ``ctx["closure_cathodic_stoich"]``.

    Returns
    -------
    dict[int, float]
        Map species_idx → R_s_hat (nondim molar rate).
    """
    R_by_rxn = {int(j): float(R) for j, R in R_per_reaction}
    out: dict[int, float] = {}
    for s, stoich_for_s in cathodic_stoich.items():
        total = 0.0
        for j, stoich in stoich_for_s.items():
            if int(j) in R_by_rxn:
                total += (-float(stoich)) * R_by_rxn[int(j)]
        out[int(s)] = total
    return out


def compute_picard_target(
    *,
    c_b_hat: float,
    theta_b: float,
    theta_OHP: float,
    R_O2_hat: float,
    I_hat: float,
    D_O2_hat: float,
    xi_old: float,
) -> float:
    """Compute the semi-implicit Picard target for ξ (Eq B).

        K_old = R_O2_hat / (θ_OHP · ξ_old)
        ξ_target = (c_b_hat/θ_b) / (1 + K_old · I_hat · θ_OHP / D_O2_hat)

    Raises
    ------
    ValueError
        If ``R_O2_hat < 0`` (irreversible cathodic scope only).
    """
    if R_O2_hat < 0.0:
        raise ValueError(
            f"compute_picard_target: R_O2_hat = {R_O2_hat} < 0.  This "
            f"closure supports irreversible cathodic only "
            f"(R_O2_hat ≥ 0).  For reversible/anodic, implement signed "
            f"closure (out of scope)."
        )
    if R_O2_hat == 0.0:
        return float(c_b_hat) / float(theta_b)
    c_eff_old = float(theta_OHP) * float(xi_old)
    if c_eff_old <= 0.0:
        # Pathological state; return no-flux equilibrium.
        return float(c_b_hat) / float(theta_b)
    K_old = float(R_O2_hat) / c_eff_old
    denom = 1.0 + K_old * float(I_hat) * float(theta_OHP) / float(D_O2_hat)
    return (float(c_b_hat) / float(theta_b)) / denom


# ---------------------------------------------------------------------------
# State norm (Picard iter-to-iter, L∞ on DOF arrays)
# ---------------------------------------------------------------------------


def _state_l_inf_delta(U, U_prev_picard_snap: tuple) -> tuple[float, float]:
    """L∞ delta between current U and a prior snapshot.

    Returns (dU_inf, U_inf).  state_norm = dU_inf / max(U_inf, 1e-12).
    """
    dU_inf = 0.0
    U_inf = 0.0
    for cur, prev in zip(U.dat, U_prev_picard_snap):
        cur_arr = np.asarray(cur.data_ro)
        prev_arr = np.asarray(prev)
        d = float(np.max(np.abs(cur_arr - prev_arr))) if cur_arr.size else 0.0
        u = float(np.max(np.abs(cur_arr))) if cur_arr.size else 0.0
        if d > dU_inf:
            dU_inf = d
        if u > U_inf:
            U_inf = u
    return dU_inf, U_inf


# ---------------------------------------------------------------------------
# H+ closure-quality diagnostic (gating for a potential v3 H+ substitute)
# ---------------------------------------------------------------------------


def _h_closure_quality(
    *,
    ctx,
    electrode_marker: int,
    packing_expr,
    closure_theta_b: float,
    closure_bulk_c_hat: dict,
) -> Optional[float]:
    """Relative error between PDE H+ at OHP and closure estimate.

        c_H_closure_est = c_H_bulk · exp(-z_H · phi_OHP) · θ_OHP / θ_b
        rel_err = |c_H_PDE − c_H_closure_est| / c_H_closure_est

    Returns None when no H+ species (z=+1, mu species index) is detected,
    or when diagnostic computation fails.
    """
    try:
        import firedrake as fd

        mu_species = ctx.get("mu_species", [])
        if not mu_species:
            return None
        h_idx = int(mu_species[0])  # H+ is the muh species
        if h_idx not in closure_bulk_c_hat:
            return None
        z_h = float(ctx["nondim"]["z_vals"][h_idx]) if "z_vals" in ctx["nondim"] else 1.0

        mesh = ctx["mesh"]
        ds = fd.Measure("ds", domain=mesh)
        ds_em = ds(int(electrode_marker))
        electrode_area = float(fd.assemble(fd.Constant(1.0) * ds_em))
        if electrode_area <= 0.0:
            return None

        # c_H_PDE at OHP via surface mean of ci_exprs[h_idx]
        ci_exprs = ctx["ci_exprs"]
        c_h_pde = float(fd.assemble(ci_exprs[h_idx] * ds_em)) / electrode_area

        # phi at OHP
        phi_form = fd.split(ctx["U"])[ctx["n_species"]]
        phi_OHP = float(fd.assemble(phi_form * ds_em)) / electrode_area

        # theta at OHP
        theta_OHP = float(fd.assemble(packing_expr * ds_em)) / electrode_area

        c_h_bulk = float(closure_bulk_c_hat[h_idx])
        c_h_closure = c_h_bulk * math.exp(-z_h * phi_OHP) * theta_OHP / float(closure_theta_b)
        if c_h_closure <= 0.0:
            return None
        return abs(c_h_pde - c_h_closure) / c_h_closure
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main factory: make_picard_run_ss
# ---------------------------------------------------------------------------


def make_picard_run_ss(
    *,
    # make_run_ss-compatible kwargs (forwarded verbatim to inner run_ss)
    ctx,
    solver,
    of_cd,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
    # Picard-specific:
    xi_funcs: dict,
    closure_theta_b: float,
    closure_bulk_c_hat: dict,
    cathodic_species_set: frozenset,
    cathodic_stoich: dict,
    D_per_species_hat: dict,
    packing_expr,
    theta_inner_expr,
    electrode_marker: int,
    Lx_hat: float,
    packing_floor: float,
    max_picard_iters: int = 15,
    tol_residual: float = 1e-3,
    tol_step: float = 1e-3,
    tol_state: float = 1e-4,
    damping_init: float = 0.5,
    damping_min: float = 0.05,
    max_damping_retries: int = 3,
    strict_floor: bool = True,
    floor_tol: float = 1e-10,
    xi_floor: float = math.exp(-50.0),
) -> Callable[[int], bool]:
    """Build a Picard-wrapped steady-state closure with the same ``(max_steps) → bool``
    signature as :func:`Forward.bv_solver.grid_per_voltage.make_run_ss`.

    Contract for the returned ``picard_run_ss(max_steps)``:

    1. Snapshot entry state ``(U, U_prev=via restore_U, xi)``.
    2. Build inner ``run_ss = make_run_ss(...)`` with the forwarded SS knobs.
    3. Call ``run_ss(max_steps)``.
    4. If False (run_ss_failed_before_picard_target): append PicardResult to
       ``ctx["_picard_run_ss_history"]``, **restore entry snap**, return False.
    5. Picard loop (iter 0..max_picard_iters):
       - Compute diagnostics.
       - Strict floor check: if floor_hit > floor_tol → append, restore, False.
       - Negative-R check: ValueError.
       - Compute ξ_target per species; persistent ξ_floor (≥2 iters) → append,
         restore, False.
       - **Iter 0:** convergence check on (residual + step + run_ss_ok).
         If all clear: append "converged", return True (leave state).
       - **Iter N≥1:** convergence check on (residual + step + state_norm +
         run_ss_ok).
       - Damped log update; assign new ξ.
       - Snapshot pre-solve iter_snap.
       - Call ``run_ss(max_steps)``.  On False with target: restore iter_snap,
         halve damping, retry (up to max_damping_retries).  Persistent fail:
         append "run_ss_failed_with_target", **restore entry snap**, return False.
    6. Max iters without convergence: append "max_picard_iters", **restore
       entry snap**, return False.

    **Every False return restores entry state**.  True return leaves the
    converged state.  Every wrap invocation appends to
    ``ctx["_picard_run_ss_history"]``.
    """
    # Initialize history list if not present
    if "_picard_run_ss_history" not in ctx:
        ctx["_picard_run_ss_history"] = []

    def picard_run_ss(max_steps: int) -> bool:
        history_list = ctx.setdefault("_picard_run_ss_history", [])

        # Step 1: Snapshot entry state
        entry_snap = snapshot_state(ctx, xi_funcs)
        iter_records: list[PicardIterRecord] = []

        # Step 2: Build inner run_ss
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

        # Step 3: Initial run_ss with inherited ξ
        ok = run_ss(max_steps)

        if not ok:
            # Step 4: failed before any target computed
            history_list.append(
                PicardResult(
                    converged=False,
                    n_iters=0,
                    reason="run_ss_failed_before_picard_target",
                    iter_history=tuple(iter_records),
                )
            )
            restore_state(ctx, xi_funcs, entry_snap)
            return False

        # ---- Picard loop ----
        damping = float(damping_init)
        xi_floor_consecutive = 0
        prev_U_snap: Optional[tuple] = None  # for state_norm computation

        for picard_iter in range(max_picard_iters + 1):  # iter 0 .. max
            # Capture snapshot of U after the most recent successful run_ss.
            # This is what state_norm at the NEXT iter will compare against.
            cur_U_snap = snapshot_U(ctx["U"])

            # Compute diagnostics
            diag = compute_picard_diagnostics(
                ctx=ctx,
                electrode_marker=electrode_marker,
                Lx_hat=Lx_hat,
                packing_floor=packing_floor,
                packing_expr=packing_expr,
                theta_inner_expr=theta_inner_expr,
                bv_rate_exprs=ctx["bv_rate_exprs"],
            )

            # Strict floor check
            if strict_floor and diag["floor_hit_area_frac"] > floor_tol:
                # Build a record of the failure
                iter_records.append(
                    _build_iter_record(
                        picard_iter=picard_iter,
                        xi_funcs=xi_funcs,
                        diag=diag,
                        cathodic_stoich=cathodic_stoich,
                        cathodic_species_set=cathodic_species_set,
                        closure_theta_b=closure_theta_b,
                        closure_bulk_c_hat=closure_bulk_c_hat,
                        D_per_species_hat=D_per_species_hat,
                        damping=damping,
                        state_norm=None,
                        h_closure_rel_err=_h_closure_quality(
                            ctx=ctx,
                            electrode_marker=electrode_marker,
                            packing_expr=packing_expr,
                            closure_theta_b=closure_theta_b,
                            closure_bulk_c_hat=closure_bulk_c_hat,
                        ),
                    )
                )
                history_list.append(
                    PicardResult(
                        converged=False,
                        n_iters=picard_iter,
                        reason="packing_floored",
                        iter_history=tuple(iter_records),
                    )
                )
                restore_state(ctx, xi_funcs, entry_snap)
                return False

            # R_O2 per species (raises ValueError on R<0)
            R_O2_per_sp = compute_R_O2_per_species(
                R_per_reaction=diag["R_per_reaction"],
                cathodic_stoich=cathodic_stoich,
            )
            for s, R_s in R_O2_per_sp.items():
                if R_s < 0.0:
                    raise ValueError(
                        f"make_picard_run_ss: R_O2_hat for species {s} = {R_s} < 0 "
                        f"at Picard iter {picard_iter}.  Irreversible cathodic only."
                    )

            # Compute targets per species
            xi_targets: dict[int, float] = {}
            xi_old_values: dict[int, float] = {}
            any_floored = False
            for s in cathodic_species_set:
                xi_old = float(xi_funcs[s].dat.data_ro[0])
                xi_old_lin = math.exp(xi_old)
                xi_old_values[s] = xi_old_lin
                xi_target = compute_picard_target(
                    c_b_hat=closure_bulk_c_hat[s],
                    theta_b=closure_theta_b,
                    theta_OHP=diag["theta_OHP_mean"],
                    R_O2_hat=R_O2_per_sp[s],
                    I_hat=diag["I_hat"],
                    D_O2_hat=D_per_species_hat[s],
                    xi_old=xi_old_lin,
                )
                # Floor check (linear-space)
                if xi_target < xi_floor:
                    any_floored = True
                    xi_targets[s] = xi_floor
                else:
                    xi_targets[s] = xi_target

            if any_floored:
                xi_floor_consecutive += 1
            else:
                xi_floor_consecutive = 0

            if strict_floor and xi_floor_consecutive >= 2:
                iter_records.append(
                    _build_iter_record(
                        picard_iter=picard_iter,
                        xi_funcs=xi_funcs,
                        diag=diag,
                        cathodic_stoich=cathodic_stoich,
                        cathodic_species_set=cathodic_species_set,
                        closure_theta_b=closure_theta_b,
                        closure_bulk_c_hat=closure_bulk_c_hat,
                        D_per_species_hat=D_per_species_hat,
                        damping=damping,
                        state_norm=None,
                        h_closure_rel_err=_h_closure_quality(
                            ctx=ctx,
                            electrode_marker=electrode_marker,
                            packing_expr=packing_expr,
                            closure_theta_b=closure_theta_b,
                            closure_bulk_c_hat=closure_bulk_c_hat,
                        ),
                    )
                )
                history_list.append(
                    PicardResult(
                        converged=False,
                        n_iters=picard_iter,
                        reason="xi_floored_persistent",
                        iter_history=tuple(iter_records),
                    )
                )
                restore_state(ctx, xi_funcs, entry_snap)
                return False

            # Residual and step per species
            residual_per_sp = {
                s: xi_old_values[s]
                + R_O2_per_sp[s] * diag["I_hat"] / D_per_species_hat[s]
                - closure_bulk_c_hat[s] / closure_theta_b
                for s in cathodic_species_set
            }
            # Damped log-space update
            log_xi_new_per_sp = {
                s: (1.0 - damping) * math.log(xi_old_values[s])
                + damping * math.log(max(xi_targets[s], xi_floor))
                for s in cathodic_species_set
            }
            step_per_sp = {
                s: abs(log_xi_new_per_sp[s] - math.log(xi_old_values[s]))
                for s in cathodic_species_set
            }

            # state_norm: iter-to-iter at SAME voltage; undefined on iter 0
            state_norm_val: Optional[float] = None
            if prev_U_snap is not None:
                dU_inf, U_inf = _state_l_inf_delta(ctx["U"], prev_U_snap)
                state_norm_val = dU_inf / max(U_inf, 1e-12)

            # Build iter record
            h_closure_err = _h_closure_quality(
                ctx=ctx,
                electrode_marker=electrode_marker,
                packing_expr=packing_expr,
                closure_theta_b=closure_theta_b,
                closure_bulk_c_hat=closure_bulk_c_hat,
            )
            iter_records.append(
                PicardIterRecord(
                    iter=picard_iter,
                    xi_per_species=tuple(
                        (int(s), float(xi_old_values[s])) for s in sorted(cathodic_species_set)
                    ),
                    R_per_reaction=diag["R_per_reaction"],
                    R_O2_total_per_species=tuple(
                        (int(s), float(R_O2_per_sp[s])) for s in sorted(cathodic_species_set)
                    ),
                    theta_OHP_mean=float(diag["theta_OHP_mean"]),
                    I_hat=float(diag["I_hat"]),
                    residual_per_species=tuple(
                        (int(s), float(residual_per_sp[s])) for s in sorted(cathodic_species_set)
                    ),
                    step_per_species=tuple(
                        (int(s), float(step_per_sp[s])) for s in sorted(cathodic_species_set)
                    ),
                    state_norm=state_norm_val,
                    damping=float(damping),
                    floor_hit_area_frac=float(diag["floor_hit_area_frac"]),
                    min_theta_inner=float(diag["min_theta_inner"]),
                    h_closure_rel_err=h_closure_err,
                )
            )

            # Convergence check
            residual_ok = all(
                abs(residual_per_sp[s]) < tol_residual for s in cathodic_species_set
            )
            step_ok = all(step_per_sp[s] < tol_step for s in cathodic_species_set)
            run_ss_ok = True  # we got here because prior run_ss returned True
            if picard_iter == 0:
                # Iter 0: no prior Picard iter exists to compute state_norm.
                # Gate on three only.
                converged = residual_ok and step_ok and run_ss_ok
            else:
                state_ok = (state_norm_val is not None) and (state_norm_val < tol_state)
                converged = residual_ok and step_ok and state_ok and run_ss_ok

            if converged:
                history_list.append(
                    PicardResult(
                        converged=True,
                        n_iters=picard_iter,
                        reason="converged",
                        iter_history=tuple(iter_records),
                    )
                )
                return True

            if picard_iter >= max_picard_iters:
                # Reached max without convergence
                history_list.append(
                    PicardResult(
                        converged=False,
                        n_iters=picard_iter,
                        reason="max_picard_iters",
                        iter_history=tuple(iter_records),
                    )
                )
                restore_state(ctx, xi_funcs, entry_snap)
                return False

            # Assign damped update
            for s in cathodic_species_set:
                xi_funcs[s].dat.data[:] = float(log_xi_new_per_sp[s])

            # Snapshot pre-solve for damping rollback
            iter_snap = snapshot_state(ctx, xi_funcs)
            prev_U_snap = cur_U_snap  # capture for next iter's state_norm

            # Re-solve with new ξ
            retry_count = 0
            cur_damping = damping
            ok = run_ss(max_steps)
            while not ok and retry_count < max_damping_retries:
                # Damping rollback: restore U/U_prev/xi to iter_snap,
                # halve damping, recompute the update with that damping, retry.
                restore_state(ctx, xi_funcs, iter_snap)
                cur_damping = max(cur_damping * 0.5, damping_min)
                for s in cathodic_species_set:
                    new_log = (
                        (1.0 - cur_damping) * math.log(xi_old_values[s])
                        + cur_damping * math.log(max(xi_targets[s], xi_floor))
                    )
                    xi_funcs[s].dat.data[:] = float(new_log)
                ok = run_ss(max_steps)
                retry_count += 1

            if not ok:
                history_list.append(
                    PicardResult(
                        converged=False,
                        n_iters=picard_iter + 1,
                        reason="run_ss_failed_with_target",
                        iter_history=tuple(iter_records),
                    )
                )
                restore_state(ctx, xi_funcs, entry_snap)
                return False

            damping = cur_damping  # carry forward (may have been halved)

        # Should not reach here (loop has explicit max_picard_iters break),
        # but defensive return.
        history_list.append(
            PicardResult(
                converged=False,
                n_iters=max_picard_iters,
                reason="max_picard_iters",
                iter_history=tuple(iter_records),
            )
        )
        restore_state(ctx, xi_funcs, entry_snap)
        return False

    return picard_run_ss


def _build_iter_record(
    *,
    picard_iter: int,
    xi_funcs: dict,
    diag: dict,
    cathodic_stoich: dict,
    cathodic_species_set: frozenset,
    closure_theta_b: float,
    closure_bulk_c_hat: dict,
    D_per_species_hat: dict,
    damping: float,
    state_norm: Optional[float],
    h_closure_rel_err: Optional[float],
) -> PicardIterRecord:
    """Build a PicardIterRecord for a failure path (where we don't compute
    residual/step normally).  Used by floor-hit and xi-floor failure paths.
    """
    xi_per_sp = tuple(
        (int(s), float(math.exp(float(xi_funcs[s].dat.data_ro[0]))))
        for s in sorted(cathodic_species_set)
    )
    R_O2_per_sp = compute_R_O2_per_species(
        R_per_reaction=diag["R_per_reaction"],
        cathodic_stoich=cathodic_stoich,
    )
    return PicardIterRecord(
        iter=picard_iter,
        xi_per_species=xi_per_sp,
        R_per_reaction=diag["R_per_reaction"],
        R_O2_total_per_species=tuple(
            (int(s), float(R_O2_per_sp.get(s, 0.0))) for s in sorted(cathodic_species_set)
        ),
        theta_OHP_mean=float(diag["theta_OHP_mean"]),
        I_hat=float(diag["I_hat"]),
        residual_per_species=tuple(),
        step_per_species=tuple(),
        state_norm=state_norm,
        damping=float(damping),
        floor_hit_area_frac=float(diag["floor_hit_area_frac"]),
        min_theta_inner=float(diag["min_theta_inner"]),
        h_closure_rel_err=h_closure_rel_err,
    )


# ---------------------------------------------------------------------------
# Ctx-aware factory adapter: make_picard_run_ss_factory
# ---------------------------------------------------------------------------


def make_picard_run_ss_factory(picard_config: PicardConfig) -> Callable:
    """Return a ``make_run_ss``-compatible factory.

    The returned callable has the signature:

        factory(ctx, solver, of_cd, **ss_kwargs) -> Callable[[int], bool]

    matching :func:`Forward.bv_solver.grid_per_voltage.make_run_ss`.
    Inside, it pulls ctx-specific Picard objects (xi_funcs, packing_expr,
    closure_theta_b, etc.) FRESH from the passed ctx, so the same
    ``picard_config`` is reused safely across rebuilt ctx instances
    (anchor build, Stern bump rungs, per-V continuation).

    Includes a sanity assertion that the ctx's diffusivity values match
    ``picard_config.D_per_species_hat`` (catches future logD-continuation
    mutation that would invalidate the cached Picard config).
    """

    def _factory(ctx, solver, of_cd, **ss_kwargs):
        # Verify ctx has Picard-mode wiring
        required_keys = {
            "picard_log_xi_funcs",
            "packing_expr",
            "theta_inner_expr",
            "closure_theta_b",
            "closure_bulk_c_hat",
            "closure_cathodic_species_set",
            "closure_cathodic_stoich",
            "closure_packing_floor",
        }
        missing = required_keys - set(ctx.keys())
        if missing:
            raise RuntimeError(
                f"make_picard_run_ss_factory: ctx is missing Picard keys "
                f"{sorted(missing)}.  Did you set bv_picard_mode=True and "
                f"bv_jithin_closure_form=True in the solver params?"
            )

        # Staleness check: ctx logD values should match picard_config.
        # logD_funcs hold log(D_hat), so D_hat = exp(value).
        D_dict = picard_config.D_dict()
        for s in ctx["closure_cathodic_species_set"]:
            ctx_logD = float(ctx["logD_funcs"][s].dat.data_ro[0])
            ctx_D = math.exp(ctx_logD)
            cfg_D = D_dict.get(int(s))
            if cfg_D is None:
                raise RuntimeError(
                    f"PicardConfig.D_per_species_hat missing entry for "
                    f"cathodic species {s} (ctx has logD = {ctx_logD})."
                )
            if abs(cfg_D - ctx_D) > 1e-12 * max(abs(cfg_D), abs(ctx_D), 1e-30):
                raise RuntimeError(
                    f"PicardConfig.D_per_species_hat[{s}] = {cfg_D} stale "
                    f"vs ctx D = {ctx_D}.  Did a D-continuation rung run "
                    f"between PicardConfig construction and this factory call?"
                )

        # Electrode marker from ctx (PicardConfig.electrode_marker is a
        # sanity cross-check; if they disagree, prefer ctx and warn).
        ctx_em = int(ctx["bv_settings"]["electrode_marker"])
        if ctx_em != int(picard_config.electrode_marker):
            raise RuntimeError(
                f"PicardConfig.electrode_marker ({picard_config.electrode_marker}) "
                f"!= ctx['bv_settings']['electrode_marker'] ({ctx_em})."
            )

        return make_picard_run_ss(
            ctx=ctx,
            solver=solver,
            of_cd=of_cd,
            **ss_kwargs,
            xi_funcs=ctx["picard_log_xi_funcs"],
            packing_expr=ctx["packing_expr"],
            theta_inner_expr=ctx["theta_inner_expr"],
            closure_theta_b=float(ctx["closure_theta_b"]),
            closure_bulk_c_hat=dict(ctx["closure_bulk_c_hat"]),
            cathodic_species_set=frozenset(ctx["closure_cathodic_species_set"]),
            cathodic_stoich=dict(ctx["closure_cathodic_stoich"]),
            D_per_species_hat=D_dict,
            electrode_marker=ctx_em,
            Lx_hat=float(picard_config.Lx_hat),
            packing_floor=float(ctx["closure_packing_floor"]),
            **picard_config.runtime_kwargs(),
        )

    return _factory
