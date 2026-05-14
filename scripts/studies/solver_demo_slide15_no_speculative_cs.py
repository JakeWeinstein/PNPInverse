"""Solver demo — Cs+/SO4(2-) baseline I-V curves over slide-15 V_RHE band.

Purpose
-------
Demonstrate that the production PNP+BV forward stack produces sensible
I-V curves over the Mangan slide-15 voltage range *without* any of the
Phase-6 speculative-physics machinery enabled.  Intended as a "the
solver works" sanity demo — not a fitting exercise.

What is OFF
-----------
* ``enable_water_ionization = False``       (no Kw split / proton-condition residual)
* ``enable_cation_hydrolysis`` not enabled  (no Singh cation hydrolysis source)
* ``lambda_hydrolysis = 0.0``               (no Singh sigma-pKa ramp)

What is ON (production defaults)
--------------------------------
* THREE_SPECIES_LOGC_BOLTZMANN (O2, H2O2, H+) with PHYSICAL hard-sphere
  ``a_vals_hat`` for the dynamic species — replaces A_DEFAULT=0.01 with
  Marcus/Stokes radii (O2 r=1.70 A, H2O2 r=2.00 A, H+ r=2.80 A as H3O+).
  Counterions Cs+ and SO4(2-) keep their existing A_*_HAT entries.
* Parallel-2e/4e Butler-Volmer (Ruggiero 2022): E0_R2e = 0.695 V,
  E0_R4e = 1.23 V.  K0_R2e = K0_HAT_R2E (= K0_PHYS_R1 = 2.4e-8 m/s).
  Sweeps ``K0_R4e_factor`` in {1, 1e-6, 1e-12, 1e-18} — 4 separate
  anchor+grid-walk runs (4e channel from "as strong as 2e" down to
  "suppressed 18 orders of magnitude").
* Cs+ + SO4(2-) Bikerman counterions, multi-ion enabled.
* Stern C_S = 0.20 F/m^2 (literature, Bohra-Koper-Choi consensus) via
  two-stage anchor: build at C_S=0.10 (anchor-friendly), then Stern-bump
  to C_S=0.20 (production) and re-solve in place.
* L_eff = 100 um (production), domain_height_hat=1.0.
* formulation = "logc_muh", log_rate = True, initializer = "debye_boltzmann".
* exponent_clip = 100.0 (Hard Rule #2), u_clamp = 100.0.

V grid
------
linspace(-0.40, +0.55, 25) V vs RHE — Mangan deck page-15 band.
Anchor at +0.55 V (top of grid, weakest cathodic drive).

Outputs
-------
For each K0_R4e factor, writes
``StudyResults/solver_demo_slide15_no_speculative_cs/factor_{f:g}/iv_curve.json``
with per-voltage current-density ``cd_mA_cm2`` and gross-H2O2 current
``pc_mA_cm2`` arrays plus convergence diagnostics.

Usage
-----
::

    cd PNPInverse
    source ../venv-firedrake/bin/activate
    python -u scripts/studies/solver_demo_slide15_no_speculative_cs.py
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
# Configuration
# ---------------------------------------------------------------------------

V_RHE_GRID = tuple(np.linspace(-0.40, +0.55, 25).round(4).tolist())
ANCHOR_V_RHE = +0.55

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
# a_phys = (4/3) pi r^3 N_A m^3/mol; a_nondim = a_phys * C_SCALE.
_C_SCALE = 1.2
_N_A = 6.02214076e23


def _a_nondim_from_radius_m(r_m: float) -> float:
    a_phys = (4.0 / 3.0) * math.pi * r_m ** 3 * _N_A
    return a_phys * _C_SCALE


A_O2_PHYSICAL = _a_nondim_from_radius_m(1.70e-10)    # ~1.487e-5
A_H2O2_PHYSICAL = _a_nondim_from_radius_m(2.00e-10)  # ~2.422e-5
A_HP_PHYSICAL = _a_nondim_from_radius_m(2.80e-10)    # ~6.645e-5 (H3O+ Stokes)


# ---------------------------------------------------------------------------
# Solver params factory
# ---------------------------------------------------------------------------


def _make_sp(
    *,
    stern_capacitance_f_m2,
    k0_r4e_factor: float,
    initializer: str = INITIALIZER,
):
    """Build SolverParams + k0_targets for one (Stern, K0_R4e_factor) pair.

    Pass ``stern_capacitance_f_m2 = None`` for the no-Stern (idealised
    C_S -> infinity) limit with Dirichlet ``phi_s = phi_m`` at the
    electrode.  Pass a positive float for the production Robin BC.

    ``initializer`` selects the BV IC: ``debye_boltzmann`` (composite-psi
    + multispecies-gamma; production default) or ``linear_phi`` (straight
    interpolation; required for the no-Stern path because the Debye IC
    builds an EDL profile that assumes the Stern capacitor exists).
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

    # Three-species log-c stack with PHYSICAL per-species a_vals_hat
    # (overrides the THREE_SPECIES_LOGC_BOLTZMANN preset's A_DEFAULT=0.01).
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
            "E_eq_v": float(E_EQ_R2E_V),
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
            "E_eq_v": float(E_EQ_R4E_V),
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
    no_stern: bool = False,
    anchor_v_rhe: float = ANCHOR_V_RHE,
    initializer: str = INITIALIZER,
    stern_final: float = STERN_BASELINE,
) -> dict:
    return {
        "K0_R4e_factor": float(factor),
        "K0_R2e_basis_m_s": 2.4e-8,
        "alpha_R2e": 0.627,
        "alpha_R4e": 0.5,
        "E_eq_R2e_V": 0.695,
        "E_eq_R4e_V": 1.23,
        "n_electrons_R2e": 2,
        "n_electrons_R4e": 4,
        "stern_mode": (
            "near_no_stern_bump_ladder" if no_stern else "bump_ladder"
        ),
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
        "v_rhe_grid_min": float(V_RHE_GRID[0]),
        "v_rhe_grid_max": float(V_RHE_GRID[-1]),
        "anchor_v_rhe": float(anchor_v_rhe),
        "counterions": ["Cs+", "SO4(2-)"],
        "a_vals_hat_dynamic": {
            "O2_r_A": 1.70,
            "H2O2_r_A": 2.00,
            "Hp_r_A_H3Op_Stokes": 2.80,
            "O2_a_nondim": A_O2_PHYSICAL,
            "H2O2_a_nondim": A_H2O2_PHYSICAL,
            "Hp_a_nondim": A_HP_PHYSICAL,
        },
        "enable_water_ionization": False,
        "enable_cation_hydrolysis": False,
        "lambda_hydrolysis": 0.0,
    }


_STERN_BUMP_LADDER_VERIFIED = (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)


def _stern_bump_ladder(target: float) -> list[float]:
    """Pick intermediate C_S steps from STERN_ANCHOR (0.10) up to target.

    Returns the list of bump targets (excluding the 0.10 starting point).
    Uses the verified ladder ``(0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)``
    truncated at the first rung >= ``target``; if ``target`` is not in
    the verified ladder it is appended as the final entry.
    """
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
    no_stern: bool = False,
    anchor_v_rhe: float = ANCHOR_V_RHE,
    initializer: str = INITIALIZER,
    stern_final: float | None = None,
) -> dict:
    """Anchor + Stern bump-ladder + grid-walk for a single K0_R4e factor.

    Build the anchor at C_S=STERN_ANCHOR=0.10 F/m^2 (known-good cold-start
    voltage; CLAUDE.md hard-rule + v10a' two-stage memory), then ladder
    C_S up to ``stern_final`` via :func:`_stern_bump_ladder` (intermediate
    steps capped at 5x growth per rung so Newton stays inside its trust
    region), then grid-walk at the final C_S.

    When ``no_stern`` is True, ``stern_final`` defaults to 100 F/m^2 —
    the practical "no Stern" limit (Stern voltage drop sub-mV).  True
    C_S -> infinity Dirichlet would require ``stern_final=None`` but
    that path doesn't converge with the Cs+/SO4 Bikerman stack: the
    OHP SO4 saturation produces an initial Newton residual of ~1e26
    even at k0_scale=1e-12, far outside the basin of attraction.  The
    bump ladder reaches C_S=100 F/m^2 (500x production) cleanly, which
    is visually indistinguishable from true no-Stern in the I-V plot.
    """
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

    if stern_final is None:
        if no_stern:
            stern_final_v = 100.0
        else:
            stern_final_v = STERN_BASELINE
    else:
        stern_final_v = float(stern_final)
    bump_ladder = _stern_bump_ladder(stern_final_v)

    sp_baseline, k0_targets = _make_sp(
        stern_capacitance_f_m2=stern_final_v,
        k0_r4e_factor=factor,
        initializer=initializer,
    )
    sp_anchor_cs, _ = _make_sp(
        stern_capacitance_f_m2=STERN_ANCHOR,
        k0_r4e_factor=factor,
        initializer=initializer,
    )
    sp_anchor = sp_anchor_cs.with_phi_applied(float(anchor_v_rhe) / V_T)

    print(f"\n===== factor = {factor:g} =====", flush=True)
    print(f"  k0_R2e target = {k0_targets[0]:.3e}", flush=True)
    print(f"  k0_R4e target = {k0_targets[1]:.3e}", flush=True)
    print(f"  Stage 1: anchor at V={anchor_v_rhe:+.3f} V, "
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
            "phi_applied_hat": [float(v) / float(V_T) for v in V_RHE_GRID],
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "converged": [False] * NV,
            "method": ["anchor-build-failed"] * NV,
            "n_converged": 0,
            "n_total": NV,
            "anchor": {
                "v_rhe": float(anchor_v_rhe),
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
                factor, no_stern=no_stern, anchor_v_rhe=anchor_v_rhe,
                initializer=initializer, stern_final=stern_final_v,
            ),
        }

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = int(ctx_anchor["U"].function_space().dim())
    print(f"  anchor ok in {anchor_wall:.1f}s; "
          f"ladder={list(anchor_result.ladder_history)!r}", flush=True)

    # Stage 2: Stern bump ladder STERN_ANCHOR -> stern_final.
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
            "phi_applied_hat": [float(v) / float(V_T) for v in V_RHE_GRID],
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "converged": [False] * NV,
            "method": ["stern-bump-failed"] * NV,
            "n_converged": 0,
            "n_total": NV,
            "anchor": {
                "v_rhe": float(anchor_v_rhe),
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
                factor, no_stern=no_stern, anchor_v_rhe=anchor_v_rhe,
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

    # Stage 3: grid walk over V_RHE_GRID.
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
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": _to_json_list(cd_arr),
        "pc_mA_cm2": _to_json_list(pc_arr),
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
        "anchor": {
            "v_rhe": float(anchor_v_rhe),
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
            factor, no_stern=no_stern, anchor_v_rhe=anchor_v_rhe,
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

    parser = argparse.ArgumentParser(
        description=(
            "Solver-works demo: 25 V_RHE points x N K0_R4e factors on "
            "Cs+/SO4(2-) baseline. No water/cation hydrolysis."
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
        "--no-stern", action="store_true",
        help=(
            "Approximate no-Stern via a Stern bump ladder to C_S=100 F/m^2 "
            "(500x production; voltage drop sub-mV; visually indistinguishable "
            "from the true Dirichlet C_S -> infinity limit which doesn't "
            "converge with the Cs+/SO4 Bikerman stack)."
        ),
    )
    parser.add_argument(
        "--stern-final", type=float, default=None,
        help=(
            "Final Stern capacitance after the bump ladder.  Defaults to "
            "0.20 F/m^2 (production) for Stern runs and 100.0 F/m^2 "
            "(near-no-Stern) for --no-stern runs."
        ),
    )
    parser.add_argument(
        "--out-name", default=None,
        help=(
            "Subdirectory name under StudyResults/.  Defaults to "
            "'solver_demo_slide15_no_speculative_cs' for the Stern run "
            "and '...cs_noStern' for the no-Stern run."
        ),
    )
    parser.add_argument(
        "--anchor-voltage", type=float, default=None,
        help=(
            "V_RHE for the cold-start anchor build.  Default depends on "
            "Stern mode: +0.55 V for Stern (top of grid), 0.0 V for "
            "no-Stern (the cold anchor at +0.55 V brittles on the "
            "Bikerman SO4 saturation; the warm-walk then reaches both "
            "band edges from 0 V)."
        ),
    )
    parser.add_argument(
        "--initializer", choices=("debye_boltzmann", "linear_phi"),
        default=None,
        help=(
            "BV IC choice.  Default depends on Stern mode: "
            "'debye_boltzmann' for Stern (production), 'linear_phi' for "
            "no-Stern (the Debye/Boltzmann composite-psi IC assumes a "
            "Stern compact layer to absorb voltage drop; with C_S -> "
            "infinity it builds an EDL Newton can't unwind, so the "
            "straight-line IC is the friendlier choice)."
        ),
    )
    args = parser.parse_args()

    factors_to_run: tuple[float, ...] = tuple(args.factors)
    no_stern: bool = bool(args.no_stern)
    if args.anchor_voltage is None:
        # Bump-ladder path always anchors at C_S=0.10 (which works at
        # the top-of-grid voltage), even for no-Stern.
        anchor_v_rhe = ANCHOR_V_RHE
    else:
        anchor_v_rhe = float(args.anchor_voltage)
    if args.initializer is None:
        initializer = INITIALIZER
    else:
        initializer = str(args.initializer)
    if args.stern_final is None:
        stern_final = 100.0 if no_stern else STERN_BASELINE
    else:
        stern_final = float(args.stern_final)
    if args.out_name is None:
        out_name = (
            "solver_demo_slide15_no_speculative_cs_noStern" if no_stern
            else "solver_demo_slide15_no_speculative_cs"
        )
    else:
        out_name = str(args.out_name)
    out_dir = Path(_ROOT) / "StudyResults" / out_name

    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()

    from Forward.bv_solver import make_graded_rectangle_mesh

    out_dir.mkdir(parents=True, exist_ok=True)

    stern_path_msg = (
        f"{STERN_ANCHOR:.3f} -> {stern_final:.3f} F/m^2 (bump ladder)"
        + ("  [near-no-Stern]" if no_stern else "")
    )
    print("=" * 78, flush=True)
    print("  Solver demo - slide-15 V_RHE, Cs+/SO4(2-), no speculative physics",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  V_RHE band   = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)", flush=True)
    print(f"  anchor V_RHE = {anchor_v_rhe:+.3f} V", flush=True)
    print(f"  factors      = {list(factors_to_run)!r}", flush=True)
    print(f"  L_eff        = {L_EFF_M * 1e6:.1f} um (production default)",
          flush=True)
    print(f"  Stern path   = {stern_path_msg}", flush=True)
    print(f"  initializer  = {initializer}", flush=True)
    print(f"  mesh         = Nx={MESH_NX}, Ny={MESH_NY}, beta={MESH_BETA}",
          flush=True)
    print(f"  dynamic a    = O2 {A_O2_PHYSICAL:.3e}, H2O2 {A_H2O2_PHYSICAL:.3e},"
          f" H+ {A_HP_PHYSICAL:.3e} (physical)", flush=True)
    print(f"  output       = {out_dir}", flush=True)

    domain_height_hat = L_EFF_M / 1.0e-4
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    sweep_t0 = time.time()
    summary_per_factor: list[dict] = []
    for factor in factors_to_run:
        report = _run_one_factor(
            factor, mesh=mesh, no_stern=no_stern,
            anchor_v_rhe=anchor_v_rhe,
            initializer=initializer,
            stern_final=stern_final,
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
        "anchor_v_rhe": float(anchor_v_rhe),
        "v_rhe_grid": list(V_RHE_GRID),
        "stern_anchor_f_m2": STERN_ANCHOR,
        "stern_final_f_m2": float(stern_final),
        "stern_bump_ladder": _stern_bump_ladder(float(stern_final)),
        "per_factor": summary_per_factor,
        "wall_seconds": float(sweep_wall),
        "config": _config_dict(
            0.0, no_stern=no_stern, anchor_v_rhe=anchor_v_rhe,
            initializer=initializer, stern_final=stern_final,
        ),
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
