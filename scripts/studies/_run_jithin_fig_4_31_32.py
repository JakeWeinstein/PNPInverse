"""Jithin (2024 thesis) Section 4.8 species-volume sweep — Fig 4.31 / 4.32.

Reproduces the two simulator-only figures from Jithin Chapter 4 §4.8 that
sweep the Bikerman steric volumes of every species:

* Fig 4.31 — c*_O2 / c^b_O2 vs applied V, one curve per volume scale
* Fig 4.32 — simulated jV curves, one curve per volume scale

Volume scales (Jithin legend): a_k³, a_k³/4, a_k³/10, a_k³/100.  Jithin's
text (p. 134-135): the full Table 4.1 volumes make the steric occlusion so
dominant that the simulated current is ≈ 0 everywhere (blue); dividing the
volumes by 4 / 10 / 100 progressively lets more O₂ reach the OHP at high
overpotential (Fig 4.31) and recovers current (Fig 4.32).  The a_k³/100 case
reaches a -0.14 mA/cm² plateau (his L=50 µm Levich estimate), still short of
the experimental -0.386 mA/cm².

Same §4.8 default settings as ``_run_jithin_fig_4_26_28.py`` (the Fig
4.26/4.27/4.28 study lives in the same §4.8 sequence), EXCEPT the V grid
spans Jithin [-0.40, +0.71] V (Fig 4.31/4.32 x-axis) rather than [0.05, 0.71]
(Fig 4.26).

The Bikerman steric exclusion of O₂ at the OHP — the mechanism behind
Jithin's volume ordering — is handled entirely by the PDE: the gradient-form
MPNP residual carries ``mu_steric = -ln(1 - Σ a·c)`` in every dynamic
species' flux, so c_O2(OHP) already reflects cation crowding and the BV rate
reads that excluded concentration directly.  No extra Butler-Volmer steric
multiplier is applied — the old ``bv_steric_activity`` "jmode" knob was an
experiment that was ditched (it double-counted what the PDE already does) and
is a no-op in the current solver.

Per scale we run the full production pipeline (anchor → Stern bump ladder →
warm-walked grid) and record c_O2(OHP)/bulk (Fig 4.31) + cd (Fig 4.32).
Each scale's result is written immediately to its own JSON; one stiff scale
failing does not abort the others.

Forked from ``scripts/studies/_run_jithin_fig_4_26_28.py``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = str(_HERE.parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Jithin §4.8 (Fig 4.31/4.32) configuration
# ---------------------------------------------------------------------------
# Convention shift: Jithin's model uses Ψ_bulk = 0.785 V (his open-circuit
# potential, p. 128); our solver uses ψ_bulk = 0.  Shift V_RHE AND E_eq by
# -OCP_OFFSET_V so the diffuse-layer driving field matches while the BV
# overpotential η = V_RHE - E_eq is preserved.  (Same trick as the 4.26 run.)
OCP_OFFSET_V: float = 0.785

# Jithin Fig 4.31/4.32 x-axis is "Applied Voltage (vs RHE)" ∈ ~[-0.40, +0.71].
# Mapped to our solver's V_RHE (raw electrode potential, ψ_bulk = 0):
#   V_RHE_ours = V_RHE_jithin - OCP_OFFSET_V
# Grid range/density and the warm-walk retry knobs are env-overridable so a
# quick diagnostic can run without editing the file, e.g.:
#   J4312_NPTS=13 J4312_VMIN_JITHIN=0.35 J4312_VMAX_JITHIN=0.71 python ... ak3_div10
# 25 pts over Jithin [-0.40, +0.71] (~0.046 V/step) matches the proven
# solver_demo_slide15_ocp_shifted recipe.  With the OCP anchor + deep
# warm-walk bisection (BISECT_DEPTH_WARM=5) each step subdivides finely, so
# this density is ample.  (The earlier hangs came from anchoring at the
# mildly-cathodic top-of-grid, NOT from grid density — see ANCHOR below.)
N_V_POINTS: int = int(os.environ.get("J4312_NPTS", "25"))
_VMIN_JITHIN: float = float(os.environ.get("J4312_VMIN_JITHIN", "-0.40"))
_VMAX_JITHIN: float = float(os.environ.get("J4312_VMAX_JITHIN", "0.71"))
V_RHE_GRID_JITHIN: Tuple[float, ...] = tuple(
    round(float(v), 4)
    for v in np.linspace(_VMIN_JITHIN, _VMAX_JITHIN, N_V_POINTS).tolist()
)
V_RHE_GRID: Tuple[float, ...] = tuple(
    round(float(v - OCP_OFFSET_V), 4) for v in V_RHE_GRID_JITHIN
)
# Anchor at the OCP (rest state): V_ours = 0 = V_jithin + OCP_OFFSET_V.
# Per solver_demo_slide15_ocp_shifted_cs.py: anchoring at the mildly-cathodic
# top-of-grid (V_ours < 0) pulls Cs⁺/H⁺ into the OHP Bikerman cap and breaks
# Newton; the OCP rest state (zero diffuse-layer field, both caps relaxed) is
# the robust seed.  The grid walker descends from V_ours=0 down through the
# grid via warm-walk + bisection.
ANCHOR_V_RHE: float = 0.0
ANCHOR_V_RHE_JITHIN: float = ANCHOR_V_RHE + OCP_OFFSET_V  # = +0.785 (OCP)

# Volume scales swept (Jithin Fig 4.31/4.32 legend).  Applied as a multiplier
# to every species' molecular volume (→ a_nondim scales linearly).
VOLUME_SCALES: Tuple[float, ...] = (1.0, 0.25, 0.10, 0.01)
SCALE_TAGS: Tuple[str, ...] = ("ak3", "ak3_div4", "ak3_div10", "ak3_div100")
SCALE_LABELS: Tuple[str, ...] = (
    r"$a_k^3$", r"$a_k^3/4$", r"$a_k^3/10$", r"$a_k^3/100$",
)

MESH_NX: int = 8
MESH_NY: int = 80
MESH_BETA: float = 3.0
EXPONENT_CLIP: float = 100.0
U_CLAMP: float = 100.0
# Warm-walk retry config (env-overridable).  Defaults match the proven
# working scripts (_run_jithin_fig_4_26_28, solver_demo_slide15_ocp_shifted):
# deep bisection (depth 5) lets the walker *cross* a stiff voltage by fine
# subdivision rather than grinding-then-failing.  The earlier hangs were NOT
# caused by deep retry — they were the mildly-cathodic anchor (see ANCHOR).
# With the OCP anchor, deep retry is the right tool and rarely engages.
N_SUBSTEPS_WARM: int = int(os.environ.get("J4312_NSUB", "8"))
BISECT_DEPTH_WARM: int = int(os.environ.get("J4312_BISECT", "5"))
MAX_SS_STEPS_WARM: int = int(os.environ.get("J4312_MAXSS", "150"))
SNES_MAX_IT: int = int(os.environ.get("J4312_SNES_MAXIT", "400"))
INITIALIZER: str = "debye_boltzmann"
FORMULATION: str = "logc_muh"

INITIAL_SCALES: Tuple[float, ...] = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP: int = 4
IC_AT_TARGET: bool = True

L_EFF_M: float = 50e-6                # Jithin §4.8 default L_bulk
STERN_ANCHOR: float = 0.10
# L_Stern = 0.4 nm, ε_r = 78.5, ε_0 = 8.854e-12 F/m → C_S ≈ 1.738 F/m²
STERN_TARGET: float = 1.738
_STERN_BUMP_LADDER_VERIFIED: Tuple[float, ...] = (
    0.20, 0.35, 0.50, 0.70, 0.85, 1.0, 1.16, 1.5, 1.738,
)

# Jithin §4.8 doesn't state j₀; use our default K0_HAT_R2E (same as 4.26).
K0_R2E_DEFAULT_FACTOR: float = 1.0

# Jithin Table 4.3 diffusivity (m²/s); only D_O2 differs from our default.
D_O2_JITHIN_PHYS: float = 1.5e-9

# Deep packing floor for the cathodic saturation regime (matches the 4.26
# and Fig 4.36 runs; default 1e-8 caps μ_steric near the singularity).
PACKING_FLOOR: float = 1e-15

# Jithin Table 4.3 bulk concentrations (mol/m³)
C_O2_JITHIN_MOL_M3: float = 0.25
C_HP_JITHIN_MOL_M3: float = 10.0      # pH 2
C_CS_JITHIN_MOL_M3: float = 190.0
C_SO4_JITHIN_MOL_M3: float = 100.0
C_H2O2_SEED_MOL_M3: float = 1e-6      # Jithin sets 0 exactly

# Jithin Table 4.1/4.3 ionic volumes (nm³/molecule) at full scale.
V_O2_JITHIN_NM3: float = 0.064
V_H2O2_JITHIN_NM3: float = 0.16638
V_HP_JITHIN_NM3: float = 0.175616
V_CS_JITHIN_NM3: float = 0.28489
V_SO4_JITHIN_NM3: float = 0.43552

# Jithin §4.8 default A_Tafel = 142 mV/dec → α·n_e ≈ 0.417 → α ≈ 0.208 (n_e=2).
A_TAFEL_MV_DEC: float = 142.0
N_ELECTRONS_TAFEL: int = 2
_ALPHA_TARGET: float = (
    59.16 / A_TAFEL_MV_DEC / float(N_ELECTRONS_TAFEL)
)  # ≈ 0.208
ALPHA_TAFEL_USED: float = min(_ALPHA_TARGET, 1.0)
A_TAFEL_USED_MV_DEC: float = (
    59.16 / (ALPHA_TAFEL_USED * float(N_ELECTRONS_TAFEL))
)

OUT_DIR = Path(_ROOT) / "StudyResults" / "jithin_fig_4_31_4_32"

_N_AVOGADRO: float = 6.02214076e23


def _jithin_a_nondim(volume_nm3: float, c_scale: float) -> float:
    """Convert Jithin Table 4.1 volume (nm³/molecule) → Bikerman a_nondim."""
    a_phys = volume_nm3 * 1e-27 * _N_AVOGADRO
    return a_phys * c_scale


def _stern_bump_ladder(target: float) -> List[float]:
    """Pick intermediate C_S rungs from STERN_ANCHOR up to ``target``."""
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


def _make_sp(*, stern_capacitance_f_m2: float, volume_scale: float):
    """Build solver params with all 5 species volumes multiplied by
    ``volume_scale`` (Jithin Fig 4.31/4.32 sweep)."""
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

    s = float(volume_scale)

    c_o2_hat = float(C_O2_JITHIN_MOL_M3) / float(C_SCALE)
    c_hp_hat = float(C_HP_JITHIN_MOL_M3) / float(C_SCALE)
    c_h2o2_seed_hat = max(
        float(H2O2_SEED_NONDIM), float(C_H2O2_SEED_MOL_M3) / float(C_SCALE)
    )

    a_o2_hat = _jithin_a_nondim(V_O2_JITHIN_NM3 * s, float(C_SCALE))
    a_h2o2_hat = _jithin_a_nondim(V_H2O2_JITHIN_NM3 * s, float(C_SCALE))
    a_hp_hat = _jithin_a_nondim(V_HP_JITHIN_NM3 * s, float(C_SCALE))
    a_cs_hat = _jithin_a_nondim(V_CS_JITHIN_NM3 * s, float(C_SCALE))
    a_so4_hat = _jithin_a_nondim(V_SO4_JITHIN_NM3 * s, float(C_SCALE))

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
        "snes_max_it": int(SNES_MAX_IT),
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    }

    k0_jithin = float(K0_HAT_R2E) * float(K0_R2E_DEFAULT_FACTOR)
    e_eq_v_shifted = 0.695 - OCP_OFFSET_V  # = -0.090 V
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
            "E_eq_v": float(e_eq_v_shifted),
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
    new_bv["packing_floor"] = float(PACKING_FLOOR)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: k0_jithin}
    return sp, k0_targets


def _seed_v_ours() -> List[float]:
    """Multi-seed anchor voltages (V_ours), env-overridable via J4312_SEEDS
    (comma-separated V_ours).  Default: OCP (0.0) plus seeds every ~0.20 V down
    to the grid floor, so each grid point is within ~0.10 V of a seed and the
    warm-walk between seeds is short (≤ ~2 grid steps)."""
    env = os.environ.get("J4312_SEEDS", "").strip()
    if env:
        return [float(x) for x in env.split(",") if x.strip()]
    lo = float(V_RHE_GRID[0])             # grid floor (most cathodic), e.g. -1.185
    seeds = [0.0]                         # OCP rest state (above grid)
    v = -0.20
    while v > lo + 0.06:
        seeds.append(round(v, 3))
        v -= 0.20
    seeds.append(round(lo, 3))            # bracket the grid floor
    return seeds


def _build_seed_anchor(*, scale: float, v_ours: float, mesh,
                       bump_ladder: List[float]):
    """Build one anchor at V_ours via solve_anchor_with_continuation
    (ic_at_target) + Stern bump.  Returns (anchor|None, status, seconds)."""
    import firedrake.adjoint as adj
    from scripts._bv_common import V_T
    from Forward.bv_solver.anchor_continuation import (
        PreconvergedAnchor,
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U

    sp_anchor_cs, k0_targets = _make_sp(
        stern_capacitance_f_m2=STERN_ANCHOR, volume_scale=scale,
    )
    sp_anchor = sp_anchor_cs.with_phi_applied(float(v_ours) / float(V_T))
    t0 = time.time()
    try:
        with adj.stop_annotating():
            ar = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh, k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
            )
        if not ar.converged:
            return None, "anchor_not_converged", time.time() - t0
        ctx = ar.ctx
        mesh_dof = int(ctx["U"].function_space().dim())
        for cs in bump_ladder:
            set_stern_capacitance_model(ctx, float(cs))
            with adj.stop_annotating():
                ctx["_last_solver"].solve()
        U_snap = snapshot_U(ctx["U"])
        anchor = PreconvergedAnchor(
            phi_applied_eta=float(v_ours) / float(V_T),
            U_snapshot=tuple(np.asarray(a).copy() for a in U_snap),
            k0_targets=tuple(
                (int(j), float(k)) for j, k in sorted(k0_targets.items())
            ),
            mesh_dof_count=mesh_dof,
            ladder_history=tuple(
                (float(s), str(o)) for s, o in ar.ladder_history
            ),
        )
        return anchor, "ok", time.time() - t0
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", time.time() - t0


def _run_one_scale(
    *, scale: float, tag: str, label: str, mesh, bump_ladder: List[float],
) -> Dict[str, Any]:
    """Multi-seed anchoring → per-seed short warm-walks for one volume scale.

    Builds a robust solve_anchor_with_continuation anchor at each seed voltage
    (ic_at_target + Stern bump), assigns every grid point to its nearest
    successful seed, and warm-walks only the short distance from that seed.
    This routes the stiff cathodic region through the k0-ramp + physics-IC
    builder instead of warm-walking into it.  Returns a JSON-serializable dict.
    """
    import firedrake as fd
    from scripts._bv_common import C_SCALE, I_SCALE, V_T
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.observables import _build_bv_observable_form

    sp_baseline, _ = _make_sp(
        stern_capacitance_f_m2=STERN_TARGET, volume_scale=scale,
    )

    seeds_v = _seed_v_ours()
    print(f"\n[{tag}] Multi-seed build (volume_scale={scale:g}); seeds V_ours="
          f"{[round(s, 3) for s in seeds_v]}", flush=True)
    built: List[Tuple[float, Any]] = []
    seed_status: List[Dict[str, Any]] = []
    t_seed = time.time()
    for sv in seeds_v:
        anchor, st, secs = _build_seed_anchor(
            scale=scale, v_ours=float(sv), mesh=mesh, bump_ladder=bump_ladder,
        )
        ok = anchor is not None
        seed_status.append({
            "v_ours": float(sv), "v_jithin": float(sv + OCP_OFFSET_V),
            "status": st, "build_seconds": round(secs, 1),
        })
        print(f"[{tag}]   seed V_ours={sv:+.3f} (V_jithin={sv + OCP_OFFSET_V:+.3f}): "
              f"{'OK' if ok else 'FAIL'} [{st}] ({secs:.1f}s)", flush=True)
        if ok:
            built.append((float(sv), anchor))
    print(f"[{tag}]   {len(built)}/{len(seeds_v)} seeds built in "
          f"{time.time() - t_seed:.1f}s", flush=True)
    if not built:
        raise RuntimeError(f"[{tag}] no seed anchors built")

    NV = len(V_RHE_GRID)
    cd_arr = np.full(NV, np.nan)
    c_o2_ohp = np.full(NV, np.nan)
    c_h_ohp = np.full(NV, np.nan)
    c_cs_ohp = np.full(NV, np.nan)
    phi_ohp = np.full(NV, np.nan)
    converged = [False] * NV
    method = [""] * NV

    # Assign each grid point to the nearest successful seed.
    seed_vs = np.array([b[0] for b in built], dtype=float)
    assign: Dict[int, List[int]] = {si: [] for si in range(len(built))}
    for gi, gv in enumerate(V_RHE_GRID):
        si = int(np.argmin(np.abs(seed_vs - float(gv))))
        assign[si].append(gi)

    t0 = time.time()
    for si, (sv, anchor) in enumerate(built):
        gis = assign[si]
        if not gis:
            continue
        sub_phi = np.array(
            [float(V_RHE_GRID[g]) / float(V_T) for g in gis], dtype=float
        )

        def _grab(local_idx: int, _phi_eta: float, ctx: dict, _gis=gis) -> None:
            g = _gis[local_idx]
            try:
                f_cd = _build_bv_observable_form(
                    ctx, mode="current_density", reaction_index=None,
                    scale=-I_SCALE,
                )
                cd_arr[g] = float(fd.assemble(f_cd))
            except Exception as exc:
                print(f"[{tag}]     cd capture failed @ g={g}: "
                      f"{type(exc).__name__}: {exc}")

        print(f"[{tag}] walk from seed V_ours={sv:+.3f}: {len(gis)} pts "
              f"(V_jithin {V_RHE_GRID_JITHIN[gis[0]]:+.3f}.."
              f"{V_RHE_GRID_JITHIN[gis[-1]]:+.3f})", flush=True)
        gr = solve_grid_with_anchor(
            sp_baseline, anchor=anchor, phi_applied_values=sub_phi, mesh=mesh,
            n_substeps_warm=N_SUBSTEPS_WARM, bisect_depth_warm=BISECT_DEPTH_WARM,
            max_ss_steps_warm=MAX_SS_STEPS_WARM, per_point_callback=_grab,
        )
        for j, g in enumerate(gis):
            pt = gr.points[j]
            converged[g] = bool(pt.converged)
            method[g] = str(pt.method)
            if not pt.converged:
                continue
            diag = pt.diagnostics or {}
            if (v := diag.get("c0_surface_mean")) is not None:
                c_o2_ohp[g] = float(v)
            if (v := diag.get("c2_surface_mean")) is not None:
                c_h_ohp[g] = float(v)
            if (v := diag.get("c_counterion0_surface_mean")) is not None:
                c_cs_ohp[g] = float(v)
            if (v := diag.get("phi_surface_mean")) is not None:
                phi_ohp[g] = float(v)
    grid_wall = time.time() - t0
    n_converged = sum(1 for c in converged if c)
    print(f"[{tag}]   grid: {n_converged}/{NV} converged in {grid_wall:.1f}s "
          f"(multi-seed: {len(built)} seeds)", flush=True)

    c_o2_bulk_hat = float(C_O2_JITHIN_MOL_M3) / float(C_SCALE)
    ratio_o2 = c_o2_ohp / c_o2_bulk_hat

    def _jl(arr: np.ndarray) -> List[Optional[float]]:
        return [
            float(x) if (np.isfinite(x) and converged[i]) else None
            for i, x in enumerate(arr)
        ]

    return {
        "scale": float(scale),
        "tag": tag,
        "label": label,
        "v_rhe": list(V_RHE_GRID),
        "v_rhe_jithin": list(V_RHE_GRID_JITHIN),
        "cd_mA_cm2": _jl(cd_arr),
        "c_O2_OHP_nondim": _jl(c_o2_ohp),
        "c_O2_OHP_mol_m3": _jl(c_o2_ohp * float(C_SCALE)),
        "ratio_O2_OHP_over_bulk": _jl(ratio_o2),
        "c_H_OHP_nondim": _jl(c_h_ohp),
        "c_Cs_OHP_nondim": _jl(c_cs_ohp),
        "phi_OHP_nondim": _jl(phi_ohp),
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
        "grid_wall_seconds": float(grid_wall),
        "seeds": seed_status,
    }


def _parse_requested(argv: List[str]) -> List[int]:
    """Map CLI scale selectors → indices into VOLUME_SCALES.

    Accepts tags (ak3, ak3_div4, ak3_div10, ak3_div100), the integer divisor
    (1, 4, 10, 100), or 'all'.  Default (no args) = all scales.
    """
    p = argparse.ArgumentParser(description="Jithin Fig 4.31/4.32 volume sweep")
    p.add_argument(
        "scales", nargs="*", default=["all"],
        help="scale selectors: tag, divisor (1/4/10/100), or 'all'",
    )
    ns = p.parse_args(argv)
    sel = ns.scales or ["all"]
    if any(s.lower() == "all" for s in sel):
        return list(range(len(VOLUME_SCALES)))
    divisor_map = {"1": 0, "4": 1, "10": 2, "100": 3}
    out: List[int] = []
    for s in sel:
        key = s.strip()
        if key in SCALE_TAGS:
            out.append(SCALE_TAGS.index(key))
        elif key in divisor_map:
            out.append(divisor_map[key])
        else:
            raise SystemExit(
                f"unknown scale selector {s!r}; "
                f"use one of {SCALE_TAGS} / {list(divisor_map)} / 'all'"
            )
    # De-dup, preserve canonical order.
    return sorted(set(out))


def _assemble_combined(
    *, config: Dict[str, Any], wall: float, bump_ladder: List[float],
) -> Dict[str, Any]:
    """Build the combined report from every per-scale JSON on disk, in
    canonical scale order (so incremental runs accumulate)."""
    scales: List[Dict[str, Any]] = []
    for tag in SCALE_TAGS:
        f = OUT_DIR / f"iv_curve_{tag}.json"
        if f.exists():
            scales.append(json.loads(f.read_text()))
    return {
        "label": "jithin_fig_4_31_4_32",
        "config": config,
        "scales": scales,
        "wall_seconds": float(wall),
        "jithin_reference": {
            "fig_4_32_div100_plateau_mA_cm2": -0.14,
            "experimental_plateau_mA_cm2": -0.386,
            "diffusion_limit_calc_L50um_mA_cm2": -0.145,
            "source": "Jithin George 2024 thesis Fig 4.31/4.32 + p.135",
        },
    }


def main() -> int:
    from scripts._bv_common import C_SCALE, L_REF
    from Forward.bv_solver import make_graded_rectangle_mesh

    requested = _parse_requested(sys.argv[1:])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  Jithin §4.8 species-volume sweep (Fig 4.31 / 4.32)")
    print("=" * 78)
    print(f"  V grid (Jithin) = [{V_RHE_GRID_JITHIN[0]:+.3f}, "
          f"{V_RHE_GRID_JITHIN[-1]:+.3f}] V ({len(V_RHE_GRID_JITHIN)} points)")
    print(f"  V grid (ours)   = [{V_RHE_GRID[0]:+.3f}, "
          f"{V_RHE_GRID[-1]:+.3f}] V (Jithin - {OCP_OFFSET_V:.3f} V offset)")
    _seeds = _seed_v_ours()
    print(f"  multi-seed anchors V_ours = {[round(s, 3) for s in _seeds]} "
          f"(OCP + cathodic; ic_at_target builds, short warm-walks between)")
    print(f"  E_eq shifted = {0.695 - OCP_OFFSET_V:+.3f} V (was 0.695)")
    alpha_neff = ALPHA_TAFEL_USED * N_ELECTRONS_TAFEL
    print(f"  α (R2e)    = {ALPHA_TAFEL_USED:.3f}  (α·n_e = {alpha_neff:.3f}, "
          f"effective Tafel slope = {A_TAFEL_USED_MV_DEC:.1f} mV/dec)")
    print(f"  L_eff      = {L_EFF_M * 1e6:.1f} μm")
    bump_ladder = _stern_bump_ladder(STERN_TARGET)
    print(f"  C_S        = {STERN_ANCHOR:.3f} → "
          f"{STERN_TARGET:.3f} F/m² via {bump_ladder!r}")
    print("  steric exclusion: handled by PDE (mu_steric in flux); "
          "no BV steric multiplier")
    requested_tags = [SCALE_TAGS[i] for i in requested]
    v_step = (
        abs(V_RHE_GRID_JITHIN[1] - V_RHE_GRID_JITHIN[0])
        if len(V_RHE_GRID_JITHIN) > 1 else 0.0
    )
    print(f"  volume scales = {VOLUME_SCALES}  (legend {SCALE_TAGS})")
    print(f"  running this invocation: {requested_tags}")
    print(f"  grid: {N_V_POINTS} pts, step {v_step:.4f} V; retry: "
          f"nsub={N_SUBSTEPS_WARM} bisect={BISECT_DEPTH_WARM} "
          f"max_ss={MAX_SS_STEPS_WARM} snes_max_it={SNES_MAX_IT}")
    print(f"  bulk: O₂ {C_O2_JITHIN_MOL_M3}, H⁺ {C_HP_JITHIN_MOL_M3} (pH 2), "
          f"Cs⁺ {C_CS_JITHIN_MOL_M3}, SO₄²⁻ {C_SO4_JITHIN_MOL_M3} mol/m³")
    print(f"  output     = {OUT_DIR}")
    print("=" * 78, flush=True)

    domain_height_hat = float(L_EFF_M) / float(L_REF)
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    t_all = time.time()
    for idx in requested:
        scale, tag, label = (
            VOLUME_SCALES[idx], SCALE_TAGS[idx], SCALE_LABELS[idx],
        )
        try:
            res = _run_one_scale(
                scale=scale, tag=tag, label=label,
                mesh=mesh, bump_ladder=bump_ladder,
            )
            res["status"] = "ok"
        except Exception as exc:
            print(f"[{tag}] SCALE FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            res = {
                "scale": float(scale), "tag": tag, "label": label,
                "status": f"failed:{type(exc).__name__}", "error": str(exc),
                "v_rhe": list(V_RHE_GRID),
                "v_rhe_jithin": list(V_RHE_GRID_JITHIN),
            }
        # Write per-scale JSON immediately so partial progress is durable.
        (OUT_DIR / f"iv_curve_{tag}.json").write_text(json.dumps(res, indent=2))
        print(f"[{tag}] wrote {OUT_DIR / f'iv_curve_{tag}.json'}", flush=True)

    wall = time.time() - t_all

    config = {
        "topology": "single_tafel_R2e_only",
        "section": "4.8 species-volume sweep",
        "alpha_tafel": float(ALPHA_TAFEL_USED),
        "alpha_times_n_e": float(alpha_neff),
        "A_tafel_mV_dec": float(A_TAFEL_USED_MV_DEC),
        "A_tafel_jithin_target_mV_dec": float(A_TAFEL_MV_DEC),
        "n_electrons_tafel": int(N_ELECTRONS_TAFEL),
        "counterions": ["Cs+", "SO4(2-)"],
        "ocp_offset_v": float(OCP_OFFSET_V),
        "e_eq_v_shifted": float(0.695 - OCP_OFFSET_V),
        "e_eq_v_jithin": 0.695,
        "stern_anchor": float(STERN_ANCHOR),
        "stern_target": float(STERN_TARGET),
        "stern_bump_ladder": list(bump_ladder),
        "l_eff_m": float(L_EFF_M),
        "d_o2_phys_jithin_m2_s": float(D_O2_JITHIN_PHYS),
        "k0_r2e_factor": float(K0_R2E_DEFAULT_FACTOR),
        "C_SCALE_mol_m3": float(C_SCALE),
        "c_o2_mol_m3": float(C_O2_JITHIN_MOL_M3),
        "c_hp_mol_m3": float(C_HP_JITHIN_MOL_M3),
        "c_cs_mol_m3": float(C_CS_JITHIN_MOL_M3),
        "c_so4_mol_m3": float(C_SO4_JITHIN_MOL_M3),
        "v_o2_nm3_full": float(V_O2_JITHIN_NM3),
        "v_h2o2_nm3_full": float(V_H2O2_JITHIN_NM3),
        "v_hp_nm3_full": float(V_HP_JITHIN_NM3),
        "v_cs_nm3_full": float(V_CS_JITHIN_NM3),
        "v_so4_nm3_full": float(V_SO4_JITHIN_NM3),
        "volume_scales": list(VOLUME_SCALES),
        "scale_tags": list(SCALE_TAGS),
        "scale_labels": list(SCALE_LABELS),
        "v_rhe_grid": list(V_RHE_GRID),
        "v_rhe_grid_jithin": list(V_RHE_GRID_JITHIN),
        "anchor_v_rhe": float(ANCHOR_V_RHE),
        "anchor_v_rhe_jithin": float(ANCHOR_V_RHE_JITHIN),
        "seeds_v_ours": [round(float(s), 3) for s in _seed_v_ours()],
        "anchoring": "multi_seed_ic_at_target",
        "mesh": {
            "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
            "domain_height_hat": float(domain_height_hat),
        },
        "formulation": FORMULATION,
        "initializer": INITIALIZER,
        "exponent_clip": float(EXPONENT_CLIP),
        "packing_floor": float(PACKING_FLOOR),
        "bv_steric_note": "steric exclusion via PDE mu_steric; no BV multiplier",
        "u_clamp": float(U_CLAMP),
        "n_v_points": int(N_V_POINTS),
        "n_substeps_warm": int(N_SUBSTEPS_WARM),
        "bisect_depth_warm": int(BISECT_DEPTH_WARM),
        "max_ss_steps_warm": int(MAX_SS_STEPS_WARM),
        "snes_max_it": int(SNES_MAX_IT),
    }
    # Reassemble combined JSON from every per-scale file on disk (so this run
    # accumulates with any earlier per-scale runs).
    report = _assemble_combined(config=config, wall=wall, bump_ladder=bump_ladder)
    out_path = OUT_DIR / "iv_curve.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path} ({len(report['scales'])} scale(s) on disk)")
    print(f"this run wall {wall:.1f}s")

    print("\nSummary (Jithin V_RHE; cd in mA/cm²):")
    for res in report["scales"]:
        if res.get("status") != "ok":
            print(f"  {res['tag']:>12}: {res.get('status')}")
            continue
        cd = np.array([np.nan if x is None else x for x in res["cd_mA_cm2"]])
        ratio = np.array(
            [np.nan if x is None else x for x in res["ratio_O2_OHP_over_bulk"]]
        )
        if np.isfinite(cd).any():
            i_min = int(np.nanargmin(cd))
            vj = res["v_rhe_jithin"][i_min]
            print(f"  {res['tag']:>12}: {res['n_converged']}/{res['n_total']} "
                  f"conv; min cd = {cd[i_min]:.4f} @ V_jithin={vj:+.3f}; "
                  f"ratio range [{np.nanmin(ratio):.3f}, {np.nanmax(ratio):.3f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
