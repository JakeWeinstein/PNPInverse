"""Solver demo — Cs+/SO4(2-) baseline with deck OCP shift applied.

Forked from ``solver_demo_slide15_no_speculative_cs.py`` to apply the
canonical deck OCP convention shift documented in CLAUDE.md Hard
Rule #8 / memory note ``project_deck_ocp_convention.md``.

At pH 4: V_OCP_RHE = 0.47 + 0.197 + 0.059·4 = 0.903 V vs RHE.

Convention:
    V_RHE_solver = V_RHE_deck − V_OCP_RHE
    E°_R2e_solver = 0.695 − V_OCP_RHE
    E°_R4e_solver = 1.23  − V_OCP_RHE

Preserves η_BV = V_RHE − E° for both R2e and R4e while moving the
diffuse-layer driving into the deck OCP regime (V_RHE_deck = V_OCP
maps to V_M − ψ_bulk = 0 — flat double layer).

What is OFF (same as parent driver)
-----------------------------------
* enable_water_ionization = False
* enable_cation_hydrolysis not enabled
* lambda_hydrolysis = 0.0

What is ON (same as parent driver)
----------------------------------
* THREE_SPECIES_LOGC_BOLTZMANN with PHYSICAL a_nondim for O2/H2O2/H+
* Cs+/SO4(2-) Bikerman counterions, multi-ion
* C_S = 0.20 F/m² via two-stage anchor (build at 0.10, bump to 0.20)
* L_eff = 100 µm; logc_muh formulation; debye_boltzmann IC
* exponent_clip = 100, u_clamp = 100

V grid
------
Deck axis: linspace(-0.40, +0.55, 25) V vs RHE.
Solver axis (after −0.903 V shift): linspace(-1.303, -0.353, 25).
Anchor at deck V_RHE = +0.55 (solver V_RHE = -0.353).

JSON output (per K0_R4e factor):
    v_rhe        — solver convention (shifted)
    v_rhe_deck   — deck convention (= v_rhe + 0.903)
    cd_mA_cm2, pc_mA_cm2 — unchanged

Usage
-----
::

    cd PNPInverse
    source ../venv-firedrake/bin/activate
    python -u scripts/studies/solver_demo_slide15_ocp_shifted_cs.py \\
        --factors=1e-18
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# OCP shift (pH 4 Cs+/SO4 baseline, per project_deck_ocp_convention)
# ---------------------------------------------------------------------------

PH_DECK = 4.0
V_OCP_RHE = 0.47 + 0.197 + 0.059 * PH_DECK   # = 0.903 V at pH 4

# Deck axis (what the deck reports — pH-4 Cs+/SO4 RRDE):
V_RHE_DECK_GRID = tuple(np.linspace(-0.40, +0.55, 25).round(4).tolist())
ANCHOR_V_RHE_DECK = +0.55

# Solver axis (after −V_OCP_RHE shift):
V_RHE_GRID = tuple(
    round(v - V_OCP_RHE, 6) for v in V_RHE_DECK_GRID
)

# Anchor at V_solver = 0 (= deck V_OCP_RHE = +0.903 V at pH 4).
# Physically: this is the rest state — no diffuse-layer driving, both
# Bikerman caps relaxed.  The original baseline could anchor at the
# top-of-grid (V_solver=+0.55) because there ψ_bulk=0 meant +0.55 V was
# mildly anodic — bearable Bikerman saturation.  After the OCP shift,
# top-of-grid V_solver=-0.353 is mildly cathodic in solver convention,
# which pulls Cs+ into the OHP cap and breaks Newton even at k0=1e-12.
# Anchoring at V_solver=0 sidesteps this.  Grid walker then descends
# from 0 down to V_solver=-1.303 (deck -0.40) via warm-walk + bisection.
ANCHOR_V_RHE = 0.0
ANCHOR_V_RHE_DECK_ACTUAL = ANCHOR_V_RHE + V_OCP_RHE  # = +0.903 (rest)


K0_R4E_FACTORS = (1.0, 1e-6, 1e-12, 1e-18)

MESH_NX = 8
MESH_NY = 80
MESH_BETA = 3.0
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"

INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP = 4
IC_AT_TARGET = True

L_EFF_M = 100e-6
STERN_ANCHOR = 0.10
STERN_BASELINE = 0.20

H_SPECIES_INDEX = 2


# Physical hard-sphere a_nondim for dynamic species O2/H2O2/H+.
_C_SCALE = 1.2
_N_A = 6.02214076e23


def _a_nondim_from_radius_m(r_m: float) -> float:
    a_phys = (4.0 / 3.0) * math.pi * r_m ** 3 * _N_A
    return a_phys * _C_SCALE


A_O2_PHYSICAL = _a_nondim_from_radius_m(1.70e-10)
A_H2O2_PHYSICAL = _a_nondim_from_radius_m(2.00e-10)
A_HP_PHYSICAL = _a_nondim_from_radius_m(2.80e-10)


# ---------------------------------------------------------------------------
# Solver params factory
# ---------------------------------------------------------------------------


def _make_sp(
    *,
    stern_capacitance_f_m2,
    k0_r4e_factor: float,
    initializer: str = INITIALIZER,
    enable_water_ionization: bool = False,
):
    """Build SolverParams + k0_targets for one (Stern, K0_R4e_factor) pair.

    Reaction E_eq values are shifted by −V_OCP_RHE so η_BV is preserved
    when V_RHE_solver = V_RHE_deck − V_OCP_RHE.

    Phase 7 step 0b: ``enable_water_ionization=True`` switches to the
    proton-condition residual E = c_H − c_OH with fast-equilibrium
    closure c_OH = Kw_eff/c_H (Phase 6α machinery); the anchor build
    must then ramp Kw via ``kw_eff_ladder`` (see ``_run_one_factor``).
    """
    from scripts._bv_common import (
        ALPHA_R1, ALPHA_R2E, ALPHA_R4E,
        C_HP_HAT, C_O2_HAT, H2O2_SEED_NONDIM,
        D_H2O2_HAT, D_HP_HAT, D_O2_HAT,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        E_EQ_R2E_V, E_EQ_R4E_V,
        K0_HAT_R1, K0_HAT_R2E, K0_HAT_R4E,
        SNES_OPTS_CHARGED,
        SpeciesConfig,
        make_bv_solver_params,
        setup_firedrake_env,
    )
    setup_firedrake_env()

    species = SpeciesConfig(
        n_species=3,
        z_vals=[0, 0, 1],
        d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
        a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL],
        c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT],
        stoichiometry_r1=[-1, +1, -2],
        stoichiometry_r2=[0, -1, -2],
        k0_legacy=[K0_HAT_R1] * 3,
        alpha_legacy=[ALPHA_R1] * 3,
        stoichiometry_legacy=[-1, -1, -1],
        c_ref_legacy=[1.0, 0.0, 1.0],
        roles=["neutral", "neutral", "proton"],
    )

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    # OCP shift applied to E° so η_BV = V_RHE_solver - E_eq_shifted is
    # invariant when V_RHE_solver = V_RHE_deck - V_OCP_RHE.
    e_eq_r2e_shifted = float(E_EQ_R2E_V) - V_OCP_RHE
    e_eq_r4e_shifted = float(E_EQ_R4E_V) - V_OCP_RHE

    k0_r4e_target = float(K0_HAT_R4E) * float(k0_r4e_factor)
    rxns = [
        {
            "k0": float(K0_HAT_R2E),
            "alpha": float(ALPHA_R2E),
            "cathodic_species": 0,
            "anodic_species": 1,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": 2,
            "reversible": True,
            "E_eq_v": e_eq_r2e_shifted,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {
            "k0": k0_r4e_target,
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": e_eq_r4e_shifted,
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]

    stern_kw: dict = {}
    if stern_capacitance_f_m2 is not None:
        stern_kw["stern_capacitance_f_m2"] = float(stern_capacitance_f_m2)
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=species,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        initializer=str(initializer),
        l_eff_m=float(L_EFF_M),
        enable_water_ionization=bool(enable_water_ionization),
        **stern_kw,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: float(K0_HAT_R2E), 1: k0_r4e_target}
    return sp, k0_targets


def _factor_label(factor: float) -> str:
    return f"factor_{factor:g}"


def _config_dict(
    factor: float, *,
    anchor_v_rhe: float = ANCHOR_V_RHE,
    initializer: str = INITIALIZER,
    stern_final: float = STERN_BASELINE,
) -> dict:
    return {
        "K0_R4e_factor": float(factor),
        "K0_R2e_basis_m_s": 2.4e-8,
        "alpha_R2e": 0.627,
        "alpha_R4e": 0.5,
        "E_eq_R2e_V_solver": 0.695 - V_OCP_RHE,
        "E_eq_R4e_V_solver": 1.23 - V_OCP_RHE,
        "E_eq_R2e_V_real_RHE": 0.695,
        "E_eq_R4e_V_real_RHE": 1.23,
        "n_electrons_R2e": 2,
        "n_electrons_R4e": 4,
        "stern_mode": "bump_ladder",
        "stern_anchor_f_m2": STERN_ANCHOR,
        "stern_final_f_m2": float(stern_final),
        "l_eff_m": L_EFF_M,
        "mesh_Nx": MESH_NX,
        "mesh_Ny": MESH_NY,
        "mesh_beta": MESH_BETA,
        "exponent_clip": EXPONENT_CLIP,
        "u_clamp": U_CLAMP,
        "n_substeps_warm": N_SUBSTEPS_WARM,
        "bisect_depth_warm": BISECT_DEPTH_WARM,
        "initializer": str(initializer),
        "formulation": FORMULATION,
        "v_rhe_grid_n": len(V_RHE_GRID),
        "v_rhe_grid_min_solver": float(V_RHE_GRID[0]),
        "v_rhe_grid_max_solver": float(V_RHE_GRID[-1]),
        "v_rhe_grid_min_deck": float(V_RHE_DECK_GRID[0]),
        "v_rhe_grid_max_deck": float(V_RHE_DECK_GRID[-1]),
        "anchor_v_rhe_solver": float(anchor_v_rhe),
        "anchor_v_rhe_deck": float(anchor_v_rhe + V_OCP_RHE),
        "counterions": ["Cs+", "SO4(2-)"],
        "a_vals_hat_dynamic": {
            "O2_r_A": 1.70, "H2O2_r_A": 2.00, "Hp_r_A_H3Op_Stokes": 2.80,
            "O2_a_nondim": A_O2_PHYSICAL,
            "H2O2_a_nondim": A_H2O2_PHYSICAL,
            "Hp_a_nondim": A_HP_PHYSICAL,
        },
        "enable_water_ionization": False,
        "enable_cation_hydrolysis": False,
        "lambda_hydrolysis": 0.0,
        "ocp_shift": {
            "pH": PH_DECK,
            "V_OCP_RHE": V_OCP_RHE,
            "formula": "0.47 + 0.197 + 0.059*pH",
            "applied_to": ["V_RHE", "E_eq_R2e", "E_eq_R4e"],
            "source": "project_deck_ocp_convention.md / "
                      "Yash-Trends/Data and Plotting.zip cell 4",
        },
    }


_STERN_BUMP_LADDER_VERIFIED = (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)


def _stern_bump_ladder(target: float) -> list[float]:
    if target <= STERN_ANCHOR:
        return [float(target)]
    rungs: list[float] = []
    for rung in _STERN_BUMP_LADDER_VERIFIED:
        if rung >= target:
            rungs.append(float(target))
            return rungs
        rungs.append(float(rung))
    if rungs[-1] < target:
        rungs.append(float(target))
    return rungs


def _run_one_factor(
    factor: float,
    *,
    mesh,
    anchor_v_rhe: float = ANCHOR_V_RHE,
    initializer: str = INITIALIZER,
    stern_final: float = STERN_BASELINE,
    enable_water_ionization: bool = False,
) -> dict:
    """Anchor + Stern bump-ladder + grid-walk for a single K0_R4e factor."""
    from scripts._bv_common import C_SCALE, I_SCALE, V_T
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        PreconvergedAnchor,
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form

    NV = len(V_RHE_GRID)
    stern_final_v = float(stern_final)
    bump_ladder = _stern_bump_ladder(stern_final_v)

    sp_baseline, k0_targets = _make_sp(
        stern_capacitance_f_m2=stern_final_v,
        k0_r4e_factor=factor,
        initializer=initializer,
        enable_water_ionization=enable_water_ionization,
    )
    sp_anchor_cs, _ = _make_sp(
        stern_capacitance_f_m2=STERN_ANCHOR,
        k0_r4e_factor=factor,
        initializer=initializer,
        enable_water_ionization=enable_water_ionization,
    )
    sp_anchor = sp_anchor_cs.with_phi_applied(float(anchor_v_rhe) / V_T)

    # Phase 6α pattern (l_eff_transport_sweep_csplus_so4.py): Kw must be
    # ramped from 0 inside the anchor build, jointly with the k0 ladder.
    if enable_water_ionization:
        from scripts._bv_common import KW_HAT
        kw_eff_ladder_arg = (
            0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT,
        )
        print(f"  enable_water_ionization=True "
              f"(Kw_eff target = {KW_HAT:.3e}, 5-rung ladder)", flush=True)
    else:
        kw_eff_ladder_arg = None

    print(f"\n===== factor = {factor:g} =====", flush=True)
    print(f"  k0_R2e target = {k0_targets[0]:.3e}", flush=True)
    print(f"  k0_R4e target = {k0_targets[1]:.3e}", flush=True)
    print(f"  Stage 1: anchor at V_solver={anchor_v_rhe:+.3f} V "
          f"(V_deck={anchor_v_rhe + V_OCP_RHE:+.3f} V), "
          f"C_S={STERN_ANCHOR:.3f} F/m^2", flush=True)
    t0 = time.time()
    anchor_converged = False
    anchor_err: str | None = None
    anchor_result = None
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
                kw_eff_ladder=kw_eff_ladder_arg,
            )
        anchor_converged = bool(anchor_result.converged)
    except LadderExhausted as exc:
        anchor_err = f"LadderExhausted: {exc}"
    except Exception as exc:
        anchor_err = f"{type(exc).__name__}: {exc}"
    anchor_wall = time.time() - t0

    if not anchor_converged:
        msg = anchor_err or "anchor build did not converge"
        print(f"  anchor FAILED in {anchor_wall:.1f}s: {msg}", flush=True)
        return {
            "factor": float(factor),
            "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
            "v_rhe": list(V_RHE_GRID),
            "v_rhe_deck": list(V_RHE_DECK_GRID),
            "phi_applied_hat": [float(v) / float(V_T) for v in V_RHE_GRID],
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "converged": [False] * NV,
            "method": ["anchor-build-failed"] * NV,
            "n_converged": 0,
            "n_total": NV,
            "anchor": {
                "v_rhe_solver": float(anchor_v_rhe),
                "v_rhe_deck": float(anchor_v_rhe + V_OCP_RHE),
                "stern_anchor_f_m2": STERN_ANCHOR,
                "stern_final_f_m2": stern_final_v,
                "stern_bump_ladder": list(bump_ladder),
                "converged": False,
                "wall_seconds": float(anchor_wall),
                "error": msg,
                "ladder_history": [],
            },
            "grid_wall_seconds": 0.0,
            "config": _config_dict(
                factor, anchor_v_rhe=anchor_v_rhe,
                initializer=initializer, stern_final=stern_final_v,
            ),
        }

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = int(ctx_anchor["U"].function_space().dim())
    print(f"  anchor ok in {anchor_wall:.1f}s; "
          f"ladder={list(anchor_result.ladder_history)!r}", flush=True)

    print(f"  Stage 2: Stern bump ladder {STERN_ANCHOR:.3f} -> "
          f"{stern_final_v:.3f} F/m^2 via {bump_ladder!r}", flush=True)
    t_bump = time.time()
    bump_history: list[tuple[float, str]] = []
    bump_err: str | None = None
    for cs_target in bump_ladder:
        try:
            set_stern_capacitance_model(ctx_anchor, float(cs_target))
            with adj.stop_annotating():
                ctx_anchor["_last_solver"].solve()
            bump_history.append((float(cs_target), "ok"))
        except Exception as exc:
            bump_err = f"bump to C_S={cs_target} failed: {type(exc).__name__}: {exc}"
            bump_history.append((float(cs_target), "fail"))
            break
    if bump_err is not None:
        print(f"  {bump_err}", flush=True)
        return {
            "factor": float(factor),
            "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
            "v_rhe": list(V_RHE_GRID),
            "v_rhe_deck": list(V_RHE_DECK_GRID),
            "phi_applied_hat": [float(v) / float(V_T) for v in V_RHE_GRID],
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "converged": [False] * NV,
            "method": ["stern-bump-failed"] * NV,
            "n_converged": 0,
            "n_total": NV,
            "anchor": {
                "v_rhe_solver": float(anchor_v_rhe),
                "v_rhe_deck": float(anchor_v_rhe + V_OCP_RHE),
                "stern_anchor_f_m2": STERN_ANCHOR,
                "stern_final_f_m2": stern_final_v,
                "stern_bump_ladder": list(bump_ladder),
                "stern_bump_history": [
                    [float(c), str(o)] for c, o in bump_history
                ],
                "converged": False,
                "wall_seconds": float(anchor_wall),
                "error": bump_err,
                "ladder_history": [
                    [float(s), str(o)]
                    for s, o in anchor_result.ladder_history
                ],
            },
            "grid_wall_seconds": 0.0,
            "config": _config_dict(
                factor, anchor_v_rhe=anchor_v_rhe,
                initializer=initializer, stern_final=stern_final_v,
            ),
        }
    print(f"  Stern ladder converged at C_S={stern_final_v:.3f} F/m^2 "
          f"in {time.time() - t_bump:.1f}s", flush=True)

    U_post_bump = snapshot_U(ctx_anchor["U"])
    anchor = PreconvergedAnchor(
        phi_applied_eta=float(anchor_v_rhe) / V_T,
        U_snapshot=tuple(np.asarray(arr).copy() for arr in U_post_bump),
        k0_targets=tuple(
            (int(j), float(k))
            for j, k in sorted(k0_targets.items())
        ),
        mesh_dof_count=mesh_dof_count,
        ladder_history=tuple(
            (float(s), str(o)) for s, o in anchor_result.ladder_history
        ),
    )

    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)

    def _grab(orig_idx: int, _phi_eta: float, ctx: dict) -> None:
        try:
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None,
                scale=-I_SCALE,
            )
            cd_arr[orig_idx] = float(fd.assemble(f_cd))
        except Exception as exc:
            print(f"    cd capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
        try:
            f_pc = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=0,
                scale=-I_SCALE,
            )
            pc_arr[orig_idx] = float(fd.assemble(f_pc))
        except Exception as exc:
            print(f"    pc capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    phi_grid_eta = np.array(V_RHE_GRID, dtype=float) / float(V_T)
    print(f"  Stage 3: grid walk over {NV} V points", flush=True)
    t0 = time.time()
    grid_result = solve_grid_with_anchor(
        sp_baseline,
        anchor=anchor,
        phi_applied_values=phi_grid_eta,
        mesh=mesh,
        n_substeps_warm=N_SUBSTEPS_WARM,
        bisect_depth_warm=BISECT_DEPTH_WARM,
        per_point_callback=_grab,
    )
    grid_wall = time.time() - t0

    converged = [bool(grid_result.points[i].converged) for i in range(NV)]
    method = [str(grid_result.points[i].method) for i in range(NV)]
    n_converged = sum(1 for c in converged if c)
    print(f"  grid: {n_converged}/{NV} converged in {grid_wall:.1f}s",
          flush=True)

    def _to_json_list(arr):
        return [
            float(x) if (np.isfinite(x) and converged[i]) else None
            for i, x in enumerate(arr)
        ]

    return {
        "factor": float(factor),
        "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
        "v_rhe": list(V_RHE_GRID),
        "v_rhe_deck": list(V_RHE_DECK_GRID),
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": _to_json_list(cd_arr),
        "pc_mA_cm2": _to_json_list(pc_arr),
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
        "anchor": {
            "v_rhe_solver": float(anchor_v_rhe),
            "v_rhe_deck": float(anchor_v_rhe + V_OCP_RHE),
            "phi_applied_eta": float(anchor_v_rhe) / float(V_T),
            "stern_anchor_f_m2": STERN_ANCHOR,
            "stern_final_f_m2": stern_final_v,
            "stern_bump_ladder": list(bump_ladder),
            "stern_bump_history": [
                [float(c), str(o)] for c, o in bump_history
            ],
            "converged": True,
            "wall_seconds": float(anchor_wall),
            "ladder_history": [
                [float(s), str(o)] for s, o in anchor_result.ladder_history
            ],
        },
        "grid_wall_seconds": float(grid_wall),
        "config": _config_dict(
            factor, anchor_v_rhe=anchor_v_rhe,
            initializer=initializer, stern_final=stern_final_v,
        ),
    }


def _parse_factor_list(arg: str) -> tuple[float, ...]:
    factors: list[float] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        factors.append(float(tok))
    if not factors:
        raise ValueError(f"--factors must be non-empty (got {arg!r})")
    return tuple(factors)


def main() -> int:
    import argparse

    global L_EFF_M

    parser = argparse.ArgumentParser(
        description=(
            "Solver-works demo with deck OCP shift applied "
            "(V_OCP=0.903 V at pH 4).  Cs+/SO4(2-), no speculative physics."
        )
    )
    parser.add_argument(
        "--factors", type=_parse_factor_list,
        default=K0_R4E_FACTORS,
        help=(
            "Comma-separated list of K0_R4e/K0_R2e factors to sweep. "
            f"Default: {','.join(repr(f) for f in K0_R4E_FACTORS)}."
        ),
    )
    parser.add_argument(
        "--out-name", default="solver_demo_slide15_ocp_shifted_cs",
        help="Subdirectory name under StudyResults/.",
    )
    parser.add_argument(
        "--l-eff-um", type=float, default=L_EFF_M * 1e6,
        help=(
            "Diffusion-film thickness in microns (default 100). "
            "Phase 7: 15.4 = O2 Levich-equivalent at 1600 rpm. Feeds both "
            "l_eff_m and domain_height_hat so mesh y-extent stays consistent."
        ),
    )
    parser.add_argument(
        "--enable-water-ionization", action="store_true",
        help=(
            "Phase 6α proton condition E = c_H - c_OH with Kw fast "
            "equilibrium; anchor ramps Kw via a 5-rung kw_eff_ladder."
        ),
    )
    args = parser.parse_args()

    L_EFF_M = float(args.l_eff_um) * 1e-6

    factors_to_run: tuple[float, ...] = tuple(args.factors)
    out_dir = Path(_ROOT) / "StudyResults" / args.out_name

    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()

    from Forward.bv_solver import make_graded_rectangle_mesh

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("  Solver demo - OCP-SHIFTED slide-15, Cs+/SO4(2-), no speculative",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  OCP shift   = -{V_OCP_RHE:.3f} V (pH {PH_DECK:.0f} formula)",
          flush=True)
    print(f"  V_RHE deck  = [{V_RHE_DECK_GRID[0]:+.3f}, {V_RHE_DECK_GRID[-1]:+.3f}] V"
          f" ({len(V_RHE_DECK_GRID)} points)", flush=True)
    print(f"  V_RHE solver= [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V"
          f" (shifted)", flush=True)
    print(f"  anchor V    = solver {ANCHOR_V_RHE:+.3f} V / deck "
          f"{ANCHOR_V_RHE_DECK_ACTUAL:+.3f} V (= V_OCP_RHE, rest state)",
          flush=True)
    print(f"  E_eq R2e    = solver {0.695 - V_OCP_RHE:+.3f} V / real_RHE +0.695 V",
          flush=True)
    print(f"  E_eq R4e    = solver {1.23 - V_OCP_RHE:+.3f} V / real_RHE +1.230 V",
          flush=True)
    print(f"  factors     = {list(factors_to_run)!r}", flush=True)
    print(f"  L_eff       = {L_EFF_M * 1e6:.1f} um", flush=True)
    print(f"  Stern path  = {STERN_ANCHOR:.3f} -> {STERN_BASELINE:.3f} F/m^2",
          flush=True)
    print(f"  output      = {out_dir}", flush=True)

    domain_height_hat = L_EFF_M / 1.0e-4
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    sweep_t0 = time.time()
    summary_per_factor: list[dict] = []
    for factor in factors_to_run:
        report = _run_one_factor(
            factor, mesh=mesh,
            enable_water_ionization=bool(args.enable_water_ionization),
        )
        out_subdir = out_dir / _factor_label(factor)
        out_subdir.mkdir(parents=True, exist_ok=True)
        with open(out_subdir / "iv_curve.json", "w") as f:
            json.dump(report, f, indent=2)
        summary_per_factor.append({
            "factor": float(factor),
            "label": _factor_label(factor),
            "anchor_converged": bool(report["anchor"]["converged"]),
            "n_converged": int(report["n_converged"]),
            "n_total": int(report["n_total"]),
        })

    sweep_wall = time.time() - sweep_t0
    summary = {
        "factors": [float(f) for f in factors_to_run],
        "anchor_v_rhe_solver": float(ANCHOR_V_RHE),
        "anchor_v_rhe_deck": float(ANCHOR_V_RHE_DECK_ACTUAL),
        "v_rhe_grid_solver": list(V_RHE_GRID),
        "v_rhe_grid_deck": list(V_RHE_DECK_GRID),
        "stern_anchor_f_m2": STERN_ANCHOR,
        "stern_final_f_m2": float(STERN_BASELINE),
        "stern_bump_ladder": _stern_bump_ladder(float(STERN_BASELINE)),
        "ocp_shift": {
            "pH": PH_DECK,
            "V_OCP_RHE": V_OCP_RHE,
        },
        "l_eff_um": float(L_EFF_M * 1e6),
        "enable_water_ionization": bool(args.enable_water_ionization),
        "per_factor": summary_per_factor,
        "wall_seconds": float(sweep_wall),
        "config": _config_dict(0.0),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary -> {summary_path}", flush=True)
    print(f"total wall = {sweep_wall:.1f}s "
          f"= {sweep_wall / 60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
