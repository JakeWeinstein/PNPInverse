"""Jithin (2024 thesis Fig 4.36) emulation — outer-Picard wrap on closure form.

Extends ``_run_jithin_closure_exact.py`` (v1 equilibrium-only closure
substitute) with an outer Picard loop that adds the continuum-MPNP
analog of Jithin's Eq 4.31 flux-supply correction:

  c_OHP_hat = θ_OHP · ξ
  ξ = c_b_hat/θ_b − R_O2_hat · I_hat / D_O2_hat,   R_O2_hat ≥ 0

The Picard wrap interleaves ξ updates between every steady-state
Newton solve in warm-walk / anchor / Stern-bump.  See APPROVED PLAN.md
at ``.planning/jithin_picard_plan/PLAN.md`` and the 7-round GPT critique
loop at ``docs/handoffs/CHATGPT_HANDOFF_40_picard-closure-cliff/``.

Predicted result (per pre-analysis): smooth S-curve from kinetic onset
to ~Levich plateau, NO cliff, because continuum-Bikerman flux correction
is only ~6% at our geometry (sat layer ~Debye-length thin vs L=10 µm).
If true → continuum-Bikerman ruled out as cliff mechanism, pivot to
path 2 (surface-coverage / non-covalent cation effect).
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = str(_HERE.parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Jithin Fig 4.36 configuration (same as closure_exact for direct comparison)
# ---------------------------------------------------------------------------
V_RHE_GRID: Tuple[float, ...] = tuple(
    round(float(v), 4) for v in np.linspace(-0.40, +0.55, 25).tolist()
)
ANCHOR_V_RHE: float = +0.55

MESH_NX: int = 8
MESH_NY: int = 80
MESH_BETA: float = 3.0
EXPONENT_CLIP: float = 100.0
U_CLAMP: float = 100.0
N_SUBSTEPS_WARM: int = 8
BISECT_DEPTH_WARM: int = 5
INITIALIZER: str = "debye_boltzmann"
FORMULATION: str = "logc_muh"

INITIAL_SCALES: Tuple[float, ...] = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP: int = 4
IC_AT_TARGET: bool = True

L_EFF_M: float = 10e-6
STERN_ANCHOR: float = 0.10
STERN_TARGET: float = 1.16
_STERN_BUMP_LADDER_VERIFIED: Tuple[float, ...] = (
    0.20, 0.35, 0.50, 0.70, 0.85, 1.0, 1.16, 1.5, 2.0, 5.0, 10.0, 100.0
)

K0_R2E_JITHIN_FACTOR: float = 1e-25
D_O2_JITHIN_PHYS: float = 1.5e-9
PACKING_FLOOR_EXPERIMENT_A: float = 1e-15

C_O2_JITHIN_MOL_M3: float = 0.25
C_HP_JITHIN_MOL_M3: float = 10.0
C_CS_JITHIN_MOL_M3: float = 190.0
C_SO4_JITHIN_MOL_M3: float = 100.0
C_H2O2_SEED_MOL_M3: float = 1e-6

V_O2_JITHIN_NM3: float = 0.064
V_H2O2_JITHIN_NM3: float = 0.16638
V_HP_JITHIN_NM3: float = 0.175616
V_CS_JITHIN_NM3: float = 0.28489
V_SO4_JITHIN_NM3: float = 0.43552

A_TAFEL_JITHIN_FIG436_MV_DEC: float = 26.2
N_ELECTRONS_TAFEL: int = 2
_ALPHA_TARGET: float = (
    59.16 / A_TAFEL_JITHIN_FIG436_MV_DEC / float(N_ELECTRONS_TAFEL)
)
ALPHA_TAFEL_USED: float = min(_ALPHA_TARGET, 1.0)
A_TAFEL_USED_MV_DEC: float = (
    59.16 / (ALPHA_TAFEL_USED * float(N_ELECTRONS_TAFEL))
)

# Picard wrap on closure form (vs v1 closure_exact equilibrium-only).
BV_STERIC_ACTIVITY: bool = False
BV_JITHIN_CLOSURE_FORM: bool = True
BV_PICARD_MODE: bool = True
BV_PICARD_STRICT_FLOOR: bool = True

# Picard runtime knobs (per APPROVED PLAN.md §3 defaults).
PICARD_MAX_ITERS: int = 15
PICARD_TOL_RESIDUAL: float = 1e-3
PICARD_TOL_STEP: float = 1e-3
PICARD_TOL_STATE: float = 1e-4
PICARD_DAMPING_INIT: float = 0.5
PICARD_DAMPING_MIN: float = 0.05
PICARD_MAX_DAMPING_RETRIES: int = 3
PICARD_FLOOR_TOL: float = 1e-10

OUT_DIR = Path(_ROOT) / "StudyResults" / "jithin_closure_picard_emulation"

_N_AVOGADRO: float = 6.02214076e23


def _jithin_a_nondim(volume_nm3: float, c_scale: float) -> float:
    a_phys = volume_nm3 * 1e-27 * _N_AVOGADRO
    return a_phys * c_scale


def _stern_bump_ladder(target: float) -> List[float]:
    if target <= STERN_ANCHOR:
        return [float(target)]
    rungs: List[float] = []
    for rung in _STERN_BUMP_LADDER_VERIFIED:
        if rung >= target:
            rungs.append(float(target))
            return rungs
        rungs.append(float(rung))
    if rungs[-1] < target:
        rungs.append(float(target))
    return rungs


def _make_sp(*, stern_capacitance_f_m2: float):
    from scripts._bv_common import (
        C_SCALE,
        D_H2O2_HAT,
        D_HP_HAT,
        D_REF,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        H2O2_SEED_NONDIM,
        K0_HAT_R2E,
        SNES_OPTS_CHARGED,
        THREE_SPECIES_LOGC_BOLTZMANN,
        make_bv_solver_params,
        setup_firedrake_env,
    )
    setup_firedrake_env()

    c_o2_hat = float(C_O2_JITHIN_MOL_M3) / float(C_SCALE)
    c_hp_hat = float(C_HP_JITHIN_MOL_M3) / float(C_SCALE)
    c_h2o2_seed_hat = max(
        float(H2O2_SEED_NONDIM), float(C_H2O2_SEED_MOL_M3) / float(C_SCALE)
    )

    a_o2_hat = _jithin_a_nondim(V_O2_JITHIN_NM3, float(C_SCALE))
    a_h2o2_hat = _jithin_a_nondim(V_H2O2_JITHIN_NM3, float(C_SCALE))
    a_hp_hat = _jithin_a_nondim(V_HP_JITHIN_NM3, float(C_SCALE))
    a_cs_hat = _jithin_a_nondim(V_CS_JITHIN_NM3, float(C_SCALE))
    a_so4_hat = _jithin_a_nondim(V_SO4_JITHIN_NM3, float(C_SCALE))

    d_o2_hat_jithin = float(D_O2_JITHIN_PHYS) / float(D_REF)

    species = dataclasses.replace(
        THREE_SPECIES_LOGC_BOLTZMANN,
        d_vals_hat=[d_o2_hat_jithin, float(D_H2O2_HAT), float(D_HP_HAT)],
        a_vals_hat=[a_o2_hat, a_h2o2_hat, a_hp_hat],
        c0_vals_hat=[c_o2_hat, c_h2o2_seed_hat, c_hp_hat],
    )

    cs_entry: Dict[str, Any] = {
        **DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        "c_bulk_nondim": float(C_CS_JITHIN_MOL_M3) / float(C_SCALE),
        "a_nondim": a_cs_hat,
    }
    so4_entry: Dict[str, Any] = {
        **DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        "c_bulk_nondim": float(C_SO4_JITHIN_MOL_M3) / float(C_SCALE),
        "a_nondim": a_so4_hat,
    }

    snes_opts = {
        **SNES_OPTS_CHARGED,
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    }

    k0_jithin = float(K0_HAT_R2E) * float(K0_R2E_JITHIN_FACTOR)
    rxns: List[Dict[str, Any]] = [
        {
            "k0": k0_jithin,
            "alpha": float(ALPHA_TAFEL_USED),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": int(N_ELECTRONS_TAFEL),
            "reversible": False,
            "E_eq_v": 0.695,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(c_hp_hat)},
            ],
        },
    ]

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=species,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[cs_entry, so4_entry],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=float(stern_capacitance_f_m2),
        initializer=INITIALIZER,
        l_eff_m=float(L_EFF_M),
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_bv["packing_floor"] = float(PACKING_FLOOR_EXPERIMENT_A)
    new_bv["bv_steric_activity"] = bool(BV_STERIC_ACTIVITY)
    new_bv["bv_jithin_closure_form"] = bool(BV_JITHIN_CLOSURE_FORM)
    new_bv["bv_picard_mode"] = bool(BV_PICARD_MODE)
    new_bv["bv_picard_strict_floor"] = bool(BV_PICARD_STRICT_FLOOR)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: k0_jithin}
    return sp, k0_targets


def _picard_result_to_dict(pr) -> Dict[str, Any]:
    """Serialize a PicardResult dataclass to a JSON-friendly dict."""
    return {
        "converged": bool(pr.converged),
        "n_iters": int(pr.n_iters),
        "reason": str(pr.reason),
        "iter_history": [
            {
                "iter": int(ir.iter),
                "xi_per_species": list(ir.xi_per_species),
                "R_per_reaction": list(ir.R_per_reaction),
                "R_O2_total_per_species": list(ir.R_O2_total_per_species),
                "theta_OHP_mean": float(ir.theta_OHP_mean),
                "I_hat": float(ir.I_hat),
                "residual_per_species": list(ir.residual_per_species),
                "step_per_species": list(ir.step_per_species),
                "state_norm": (
                    float(ir.state_norm) if ir.state_norm is not None else None
                ),
                "damping": float(ir.damping),
                "floor_hit_area_frac": float(ir.floor_hit_area_frac),
                "min_theta_inner": float(ir.min_theta_inner),
                "h_closure_rel_err": (
                    float(ir.h_closure_rel_err)
                    if ir.h_closure_rel_err is not None
                    else None
                ),
            }
            for ir in pr.iter_history
        ],
    }


def main() -> int:
    from scripts._bv_common import C_SCALE, I_SCALE, L_REF, V_T
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_with_anchor,
    )
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        PreconvergedAnchor,
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.closure_picard import (
        PicardConfig,
        make_picard_run_ss_factory,
        snapshot_xi,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  Jithin Fig 4.36 closure-form + outer Picard — Tafel R2e, Cs/SO4 pH 2")
    print("=" * 78)
    print(f"  V grid     = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)")
    print(f"  anchor V   = {ANCHOR_V_RHE:+.3f} V")
    alpha_neff = ALPHA_TAFEL_USED * N_ELECTRONS_TAFEL
    print(f"  α (R2e)    = {ALPHA_TAFEL_USED:.3f}  "
          f"(α·n_e = {alpha_neff:.3f}, "
          f"effective Tafel slope = {A_TAFEL_USED_MV_DEC:.1f} mV/dec)")
    print(f"  L_eff      = {L_EFF_M * 1e6:.1f} μm")
    bump_ladder = _stern_bump_ladder(STERN_TARGET)
    print(f"  C_S        = {STERN_ANCHOR:.3f} → "
          f"{STERN_TARGET:.3f} F/m² via {bump_ladder!r}")
    print(f"  k0 (R2e)   = {K0_R2E_JITHIN_FACTOR:.2e} × K0_HAT_R2E")
    print(f"  D_O2       = {D_O2_JITHIN_PHYS:.2e} m²/s (Jithin Table 4.1)")
    print(f"  packing_floor = {PACKING_FLOOR_EXPERIMENT_A:.0e}")
    print(f"  BV_STERIC_ACTIVITY      = {BV_STERIC_ACTIVITY}")
    print(f"  BV_JITHIN_CLOSURE_FORM  = {BV_JITHIN_CLOSURE_FORM}")
    print(f"  BV_PICARD_MODE          = {BV_PICARD_MODE}  "
          f"<-- continuum-MPNP analog of Jithin κ_5·φ·g flux-supply correction")
    print(f"  BV_PICARD_STRICT_FLOOR  = {BV_PICARD_STRICT_FLOOR}")
    print(f"  Picard: max_iters={PICARD_MAX_ITERS}, "
          f"tol_residual={PICARD_TOL_RESIDUAL}, "
          f"tol_step={PICARD_TOL_STEP}, tol_state={PICARD_TOL_STATE}")
    print(f"  Picard: damping_init={PICARD_DAMPING_INIT}, "
          f"max_damping_retries={PICARD_MAX_DAMPING_RETRIES}")
    print(f"  output     = {OUT_DIR}")
    print("=" * 78)

    domain_height_hat = float(L_EFF_M) / float(L_REF)
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    # Compute Lx_hat (nondim mesh x-extent) for the Picard I_hat integral.
    coords = mesh.coordinates.dat.data_ro
    Lx_hat = float(coords[:, 0].max() - coords[:, 0].min())

    sp_baseline, k0_targets = _make_sp(stern_capacitance_f_m2=STERN_TARGET)
    sp_anchor_cs, _ = _make_sp(stern_capacitance_f_m2=STERN_ANCHOR)
    sp_anchor = sp_anchor_cs.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

    # Build Picard config (ctx-invariant knobs).
    d_o2_hat = float(D_O2_JITHIN_PHYS) / float(_make_sp.__globals__.get("D_REF", 1.0))
    # Recompute D_O2_hat from the actual scaling for safety
    from scripts._bv_common import D_REF as _D_REF
    d_o2_hat = float(D_O2_JITHIN_PHYS) / float(_D_REF)
    picard_config = PicardConfig(
        D_per_species_hat=((0, d_o2_hat),),     # O₂ at species index 0
        electrode_marker=3,                       # default for our 2D rectangular mesh
                                                  # (see scripts/_bv_common.py:572)
        Lx_hat=Lx_hat,
        max_picard_iters=PICARD_MAX_ITERS,
        tol_residual=PICARD_TOL_RESIDUAL,
        tol_step=PICARD_TOL_STEP,
        tol_state=PICARD_TOL_STATE,
        damping_init=PICARD_DAMPING_INIT,
        damping_min=PICARD_DAMPING_MIN,
        max_damping_retries=PICARD_MAX_DAMPING_RETRIES,
        strict_floor=BV_PICARD_STRICT_FLOOR,
        floor_tol=PICARD_FLOOR_TOL,
    )
    picard_factory = make_picard_run_ss_factory(picard_config)

    print(f"\nStage 1: anchor build at V={ANCHOR_V_RHE:+.3f} V, "
          f"C_S={STERN_ANCHOR:.3f} F/m² (Picard-wrapped)", flush=True)
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
                make_run_ss_factory=picard_factory,
            )
    except LadderExhausted as exc:
        raise RuntimeError(f"Anchor failed: {exc}") from exc
    if not anchor_result.converged:
        raise RuntimeError("Anchor did not converge at C_S=0.10.")
    anchor_picard_history = list(
        anchor_result.ctx.get("_picard_run_ss_history", [])
    )
    print(f"  anchor done in {time.time() - t0:.1f}s; "
          f"ladder={list(anchor_result.ladder_history)!r}", flush=True)
    print(f"  anchor Picard wraps: {len(anchor_picard_history)} "
          f"(reasons: {[pr.reason for pr in anchor_picard_history]})", flush=True)

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = ctx_anchor["U"].function_space().dim()

    print(f"\nStage 2: Stern bump ladder {STERN_ANCHOR:.3f} → "
          f"{STERN_TARGET:.3f} F/m² via {bump_ladder!r}", flush=True)
    t_bump = time.time()
    bump_history: List[Tuple[float, str]] = []
    bump_picard_history_per_rung: List[List[Any]] = []
    for cs_target in bump_ladder:
        t_rung = time.time()
        try:
            set_stern_capacitance_model(ctx_anchor, float(cs_target))
            # Clear Picard history for this rung
            ctx_anchor["_picard_run_ss_history"] = []
            with adj.stop_annotating():
                # The bump path uses ctx_anchor's _last_solver, which was
                # built without Picard.  After set_stern_capacitance_model,
                # we need to re-solve.  For Picard wrap to apply here, we
                # need to call picard_factory ourselves.
                run_ss_picard = picard_factory(
                    ctx=ctx_anchor,
                    solver=ctx_anchor["_last_solver"],
                    of_cd=_build_bv_observable_form(
                        ctx_anchor, mode="current_density",
                        reaction_index=None, scale=1.0,
                    ),
                )
                ok = run_ss_picard(200)
            if not ok:
                raise RuntimeError(f"Picard wrap failed at Stern rung {cs_target}")
            bump_history.append((float(cs_target), "ok"))
            rung_picard = list(ctx_anchor.get("_picard_run_ss_history", []))
            bump_picard_history_per_rung.append(rung_picard)
            print(f"  rung C_S={cs_target:.3f} F/m² ok "
                  f"({time.time() - t_rung:.1f}s; "
                  f"Picard wraps={len(rung_picard)})", flush=True)
        except Exception as exc:
            bump_history.append((float(cs_target), "fail"))
            raise RuntimeError(
                f"Stern bump to C_S={cs_target:.3f} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    print(f"  Stern ladder done in {time.time() - t_bump:.1f}s "
          f"(reached C_S = {bump_ladder[-1]:.3f} F/m²)", flush=True)

    U_post_bump = snapshot_U(ctx_anchor["U"])
    xi_post_bump = snapshot_xi(ctx_anchor["picard_log_xi_funcs"])
    anchor = PreconvergedAnchor(
        phi_applied_eta=float(ANCHOR_V_RHE) / V_T,
        U_snapshot=tuple(np.asarray(arr).copy() for arr in U_post_bump),
        k0_targets=tuple(
            (int(j), float(k))
            for j, k in sorted(k0_targets.items())
        ),
        mesh_dof_count=int(mesh_dof_count),
        ladder_history=tuple(
            (float(s), str(o)) for s, o in anchor_result.ladder_history
        ),
        xi_snapshots=xi_post_bump,
    )

    NV = len(V_RHE_GRID)
    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)
    c_o2_ohp = np.full(NV, np.nan)
    c_h2o2_ohp = np.full(NV, np.nan)
    c_h_ohp = np.full(NV, np.nan)
    phi_ohp = np.full(NV, np.nan)
    picard_per_V: Dict[int, List[Dict[str, Any]]] = {}
    picard_converged_per_V: Dict[int, bool] = {}

    def _grab(orig_idx: int, _phi_eta: float, ctx: dict) -> None:
        # Standard cd/pc captures
        try:
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None,
                scale=-I_SCALE,
            )
            cd_arr[orig_idx] = float(fd.assemble(f_cd))
        except Exception as exc:
            print(f"    cd capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")
        try:
            f_pc = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=0,
                scale=-I_SCALE,
            )
            pc_arr[orig_idx] = float(fd.assemble(f_pc))
        except Exception as exc:
            print(f"    pc capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")
        # OHP diagnostics (from solver-emitted ctx state via grid walker)
        # Capture Picard history for this V.
        history = list(ctx.get("_picard_run_ss_history", []))
        picard_per_V[orig_idx] = [_picard_result_to_dict(pr) for pr in history]
        # Per APPROVED PLAN.md §6: converged_overall_per_V = warm_walk True
        # AND final accepted Picard result converged.  The final accepted
        # one is the last entry in history (warm-walk accepts the most
        # recent True returns; failed attempts are recorded but rejected).
        if history and history[-1].converged:
            picard_converged_per_V[orig_idx] = True
        else:
            picard_converged_per_V[orig_idx] = False

    phi_grid_eta = np.array(V_RHE_GRID, dtype=float) / float(V_T)
    print(f"\nStage 3: grid walk over {NV} V points (Picard-wrapped)", flush=True)
    t0 = time.time()
    grid_result = solve_grid_with_anchor(
        sp_baseline,
        anchor=anchor,
        phi_applied_values=phi_grid_eta,
        mesh=mesh,
        n_substeps_warm=N_SUBSTEPS_WARM,
        bisect_depth_warm=BISECT_DEPTH_WARM,
        per_point_callback=_grab,
        make_run_ss_factory=picard_factory,
    )
    grid_wall = time.time() - t0

    grid_converged = [bool(grid_result.points[i].converged) for i in range(NV)]
    method = [str(grid_result.points[i].method) for i in range(NV)]
    converged_overall = [
        bool(grid_converged[i] and picard_converged_per_V.get(i, False))
        for i in range(NV)
    ]
    n_converged = sum(1 for c in converged_overall if c)
    n_grid_only = sum(1 for c in grid_converged if c)
    print(f"  grid: {n_grid_only}/{NV} grid-converged, "
          f"{n_converged}/{NV} also Picard-converged "
          f"in {grid_wall:.1f}s", flush=True)

    # Extract per-V OHP-surface diagnostics
    for i in range(NV):
        if not converged_overall[i]:
            continue
        diag = grid_result.points[i].diagnostics or {}
        if (v := diag.get("c0_surface_mean")) is not None:
            c_o2_ohp[i] = float(v)
        if (v := diag.get("c1_surface_mean")) is not None:
            c_h2o2_ohp[i] = float(v)
        if (v := diag.get("c2_surface_mean")) is not None:
            c_h_ohp[i] = float(v)
        if (v := diag.get("phi_surface_mean")) is not None:
            phi_ohp[i] = float(v)

    def _to_json_list(arr: np.ndarray) -> List[Any]:
        return [
            float(x) if (np.isfinite(x) and converged_overall[i]) else None
            for i, x in enumerate(arr)
        ]

    report: Dict[str, Any] = {
        "label": "jithin_closure_picard_emulation",
        "config": {
            "topology": "single_tafel_R2e_only_closure_picard",
            "alpha_tafel": float(ALPHA_TAFEL_USED),
            "alpha_times_n_e": float(alpha_neff),
            "A_tafel_mV_dec": float(A_TAFEL_USED_MV_DEC),
            "n_electrons_tafel": int(N_ELECTRONS_TAFEL),
            "counterions": ["Cs+", "SO4(2-)"],
            "stern_anchor": float(STERN_ANCHOR),
            "stern_target": float(STERN_TARGET),
            "stern_bump_ladder": list(bump_ladder),
            "l_eff_m": float(L_EFF_M),
            "d_o2_phys_jithin_m2_s": float(D_O2_JITHIN_PHYS),
            "k0_r2e_jithin_factor": float(K0_R2E_JITHIN_FACTOR),
            "C_SCALE_mol_m3": float(C_SCALE),
            "c_o2_mol_m3": float(C_O2_JITHIN_MOL_M3),
            "c_hp_mol_m3": float(C_HP_JITHIN_MOL_M3),
            "c_cs_mol_m3": float(C_CS_JITHIN_MOL_M3),
            "c_so4_mol_m3": float(C_SO4_JITHIN_MOL_M3),
            "v_h_nm3": float(V_HP_JITHIN_NM3),
            "v_o2_nm3": float(V_O2_JITHIN_NM3),
            "v_h2o2_nm3": float(V_H2O2_JITHIN_NM3),
            "v_cs_nm3": float(V_CS_JITHIN_NM3),
            "v_so4_nm3": float(V_SO4_JITHIN_NM3),
            "v_rhe_grid": list(V_RHE_GRID),
            "anchor_v_rhe": float(ANCHOR_V_RHE),
            "mesh": {
                "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
                "domain_height_hat": float(domain_height_hat),
                "Lx_hat": float(Lx_hat),
            },
            "formulation": FORMULATION,
            "initializer": INITIALIZER,
            "exponent_clip": float(EXPONENT_CLIP),
            "packing_floor": float(PACKING_FLOOR_EXPERIMENT_A),
            "bv_steric_activity": bool(BV_STERIC_ACTIVITY),
            "bv_jithin_closure_form": bool(BV_JITHIN_CLOSURE_FORM),
            "bv_picard_mode": bool(BV_PICARD_MODE),
            "bv_picard_strict_floor": bool(BV_PICARD_STRICT_FLOOR),
            "picard_max_iters": int(PICARD_MAX_ITERS),
            "picard_tol_residual": float(PICARD_TOL_RESIDUAL),
            "picard_tol_step": float(PICARD_TOL_STEP),
            "picard_tol_state": float(PICARD_TOL_STATE),
            "picard_damping_init": float(PICARD_DAMPING_INIT),
            "picard_max_damping_retries": int(PICARD_MAX_DAMPING_RETRIES),
            "picard_floor_tol": float(PICARD_FLOOR_TOL),
            "u_clamp": float(U_CLAMP),
            "n_substeps_warm": int(N_SUBSTEPS_WARM),
            "bisect_depth_warm": int(BISECT_DEPTH_WARM),
        },
        "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
        "v_rhe": list(V_RHE_GRID),
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": _to_json_list(cd_arr),
        "pc_mA_cm2": _to_json_list(pc_arr),
        "c_O2_OHP_nondim": _to_json_list(c_o2_ohp),
        "c_H2O2_OHP_nondim": _to_json_list(c_h2o2_ohp),
        "c_H_OHP_nondim": _to_json_list(c_h_ohp),
        "phi_OHP_nondim": _to_json_list(phi_ohp),
        "grid_converged": grid_converged,
        "picard_converged_per_V": [
            picard_converged_per_V.get(i, False) for i in range(NV)
        ],
        "converged_overall": converged_overall,
        "method": method,
        "n_converged": int(n_converged),
        "n_grid_only": int(n_grid_only),
        "n_total": int(NV),
        "picard_per_V": {
            str(i): picard_per_V.get(i, []) for i in range(NV)
        },
        "anchor": {
            "v_rhe": float(ANCHOR_V_RHE),
            "phi_applied_eta": float(ANCHOR_V_RHE) / V_T,
            "stern_anchor_f_m2": float(STERN_ANCHOR),
            "stern_target_f_m2": float(STERN_TARGET),
            "stern_bump_ladder": list(bump_ladder),
            "stern_bump_history": [
                [float(c), str(o)] for c, o in bump_history
            ],
            "ladder_history": [
                [float(s), str(o)] for s, o in anchor_result.ladder_history
            ],
            "anchor_picard_wraps": len(anchor_picard_history),
            "anchor_picard_reasons": [pr.reason for pr in anchor_picard_history],
        },
        "grid_wall_seconds": float(grid_wall),
        "jithin_reference": {
            "exp_plateau_mA_cm2": -0.386,
            "fig436_simulated_plateau_mA_cm2": -0.36,
            "fig436_far_cathodic_cliff_mA_cm2": -0.15,
            "diffusion_limit_calc_L10um_mA_cm2": -0.724,
            "source": "Jithin George 2024 thesis Fig 4.36 + p.138",
            "note": (
                "Picard-wrap closure test (continuum-MPNP analog of "
                "Jithin Eq 4.31).  Predicted: smooth S-curve to ~Levich, "
                "NO cliff (continuum correction ≈6% at L=10µm).  If "
                "this prediction holds, continuum-Bikerman ruled out as "
                "cliff mechanism — pivot to path 2 (surface coverage)."
            ),
        },
    }
    out_path = OUT_DIR / "iv_curve.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")

    cd = np.array(
        [x if x is not None else np.nan for x in report["cd_mA_cm2"]]
    )
    v_arr = np.array(V_RHE_GRID)
    finite = np.isfinite(cd)
    if finite.any():
        i_min = int(np.nanargmin(cd))
        print(f"  simulated plateau (most cathodic cd): "
              f"V = {v_arr[i_min]:+.3f} V, cd = {cd[i_min]:.4f} mA/cm²")
        if converged_overall[0]:
            cd_far = float(cd[0])
            v_mask = (v_arr >= -0.3) & (v_arr <= -0.05)
            cd_mid_idxs = np.where(v_mask & finite)[0]
            if len(cd_mid_idxs) > 0:
                cd_mid_min_mag = float(np.min(np.abs(cd[cd_mid_idxs])))
                cliff_ratio = (
                    abs(cd_far) / cd_mid_min_mag
                    if cd_mid_min_mag > 1e-12 else float("nan")
                )
                print(f"  cliff diagnostic: |cd(V=-0.4)|={abs(cd_far):.4f}, "
                      f"min |cd| in V∈[-0.3,-0.05]={cd_mid_min_mag:.4f}, "
                      f"ratio={cliff_ratio:.3f} "
                      f"(<1 = cliff present; ≈1 = no cliff)")
        print(f"  Jithin Fig 4.36 simulated plateau     ≈ -0.36  mA/cm² "
              f"(L_diff=10μm, A_Tafel=26.2)")
        print(f"  Jithin Fig 4.36 far-cathodic cliff    ≈ -0.15  mA/cm² @ V=-0.4")
        print(f"  experimental plateau                  ≈ -0.386 mA/cm²")
        print(f"  v1 closure-exact most cathodic cd     ≈ -0.27  mA/cm² @ V=-0.16")
        print(f"  continuum Levich at L=10μm            ≈ -0.724 mA/cm²")

    return 0


if __name__ == "__main__":
    sys.exit(main())
