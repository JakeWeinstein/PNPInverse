"""Jithin (2024 thesis Fig 4.36) emulation — single Tafel R2e, Cs+/SO4 pH 2.

Reproduces the parameter set Jithin used in his best-fit Fig 4.36 to test
whether our solver can recover his curve when configured equivalently:

  * Single peroxide-pathway Tafel reaction (R2e only, no R4e, no anodic branch)
  * A_Tafel = 26.2 mV/dec  →  α·n_e ≈ 2.26  (α = 1.13, n_e = 2)
  * Cs⁺/SO₄²⁻ electrolyte at pH 2 (Jithin Table 4.1)
  * Bulk: O₂ 0.25, H⁺ 10, Cs⁺ 190, SO₄²⁻ 100, H₂O₂ ≈ 0 (mol/m³)
  * Jithin's Table 4.1 nm³ volumes converted to Bikerman a_nondim
  * L_eff = 10 μm (Jithin Fig 4.36 fitted L_diff)
  * C_S target 1.16 F/m² (Jithin L_Stern = 0.6 nm); we use 0.20 F/m²
    (production two-stage anchor reach) and flag the gap.
  * No hydrolysis, no water ionization

Reference comparison against Jithin's Fig 4.36:
  * Experimental plateau           ≈ -0.386 mA/cm²
  * Jithin simulated plateau       ≈ -0.14  mA/cm²  (transport-limited)
  * Jithin's diffusion-limit calc  ≈ -0.145 mA/cm²

Pattern forked from ``scripts/studies/_phase_D_bridge_no_hydrolysis_cs.py``.
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
# Jithin Fig 4.36 configuration
# ---------------------------------------------------------------------------
# Grid restricted to [-0.40, +0.55] V_RHE: the kinetic + plateau region.
# Above +0.55 (≈ -0.14 V from E°=0.695), j ≈ 0 from the previous full-grid
# run, so dropping V > 0.55 loses no IV-curve structure.
# Anchor at +0.55 V (above E° — kinetic term suppressed) keeps the SO₄²⁻
# Bikerman pile-up tractable as C_S bumps up to 1.16 F/m².  Anchoring at
# +1.00 V failed Stern bumps beyond C_S = 0.7 due to OHP polarization
# saturation; the lower anchor sits closer to PZC.
V_RHE_GRID: Tuple[float, ...] = tuple(
    round(float(v), 4) for v in np.linspace(-0.40, +0.55, 25).tolist()
)
ANCHOR_V_RHE: float = +0.55          # top of grid; warm-walk descends

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

L_EFF_M: float = 10e-6                # Jithin Fig 4.36 fitted L_diff
STERN_ANCHOR: float = 0.10
# Jithin L_Stern = 0.6 nm with ε_r = 78.5, ε_0 = 8.854e-12  →  C_S = 1.157 F/m²
STERN_TARGET: float = 1.16
# Denser ladder than solver_demo_slide15_no_speculative_cs.py because
# Jithin's pH-2 stack with large a-volumes (Cs⁺ ~6× standard, SO₄²⁻ ~7×)
# is much stiffer than the v10b validation regime.  At anchor V=+1.0 V
# the coarser 0.5 → 1.0 step diverges; intermediate 0.7/0.85 keep Newton
# inside the basin.
_STERN_BUMP_LADDER_VERIFIED: Tuple[float, ...] = (
    0.20, 0.35, 0.50, 0.70, 0.85, 1.0, 1.16, 1.5, 2.0, 5.0, 10.0, 100.0
)

# Jithin Fig 4.36 j₀ is ~10 orders of magnitude smaller than our K0_HAT_R2E
# (back-of-envelope from his ΔV_50% ≈ 0.3 V shift relative to ours,
# divided by ln(10)·V_T/(α·n_e) ≈ 0.030 V per decade).  Adjust if the
# half-wave still doesn't land near +0.25 V vs RHE.
K0_R2E_JITHIN_FACTOR: float = 1e-25

# Jithin Table 4.1 diffusivities (m²/s).  Only D_O2 differs from our
# default; D_HP / D_H2O2 match within rounding.
D_O2_JITHIN_PHYS: float = 1.5e-9

# Gap 2 investigation (Experiment A): default packing_floor=1e-8 caps
# μ_steric at -ln(1e-8) ≈ +18.4, which may bound the steric singularity
# at near-saturation and mask Jithin's far-cathodic cliff.  Drop to
# 1e-15 → μ_steric_max ≈ +34.5.  If cliff appears at far cathodic V
# with the lower floor, the floor was the culprit.
PACKING_FLOOR_EXPERIMENT_A: float = 1e-15

# Jithin Table 4.1 bulk concentrations (mol/m³)
C_O2_JITHIN_MOL_M3: float = 0.25
C_HP_JITHIN_MOL_M3: float = 10.0      # pH 2
C_CS_JITHIN_MOL_M3: float = 190.0
C_SO4_JITHIN_MOL_M3: float = 100.0
C_H2O2_SEED_MOL_M3: float = 1e-6      # Jithin sets 0 exactly

# Jithin Table 4.1 ionic volumes (nm³/molecule).
V_O2_JITHIN_NM3: float = 0.064
V_H2O2_JITHIN_NM3: float = 0.16638
V_HP_JITHIN_NM3: float = 0.175616
V_CS_JITHIN_NM3: float = 0.28489
V_SO4_JITHIN_NM3: float = 0.43552

# Tafel slope → α·n_e: b = (ln10·V_T)/(α·n_e); at 298 K → α·n_e = 59.2/b_mV_dec
# Jithin Fig 4.36 fitted A_Tafel = 26.2 mV/dec → α·n_e ≈ 2.26 (α ≈ 1.13 at n_e=2),
# which violates the production validator's α ∈ (0,1] Marcus bound.  We clamp
# α at 1.0 here — gives effective Tafel slope ≈ 29.6 mV/dec (vs his 26.2).
# This is acceptable for the sanity check because Jithin's plateau is mass-
# transport-limited (his own Levich calc on p.138: -0.145 mA/cm²); the Tafel
# slope only affects the rising-portion shape, not the plateau magnitude.
A_TAFEL_JITHIN_FIG436_MV_DEC: float = 26.2     # Jithin Fig 4.36 fitted value
N_ELECTRONS_TAFEL: int = 2
_ALPHA_TARGET: float = (
    59.16 / A_TAFEL_JITHIN_FIG436_MV_DEC / float(N_ELECTRONS_TAFEL)
)  # ≈ 1.13
ALPHA_TAFEL_USED: float = min(_ALPHA_TARGET, 1.0)
A_TAFEL_USED_MV_DEC: float = (
    59.16 / (ALPHA_TAFEL_USED * float(N_ELECTRONS_TAFEL))
)

OUT_DIR = Path(_ROOT) / "StudyResults" / "jithin_emulation_fig436_jmode_k0_1e-25"
# Set True to apply Jithin's closure-form multiplicative θ(OHP) factor
# to the BV rate (Forward.bv_solver.forms_logc_muh:bv_steric_activity).
# This is the "Jithin-mode" steric activity coefficient — when packing
# at the OHP → 0 from cation saturation, BV rate is suppressed below
# Levich, reproducing his far-cathodic cliff.
BV_STERIC_ACTIVITY: bool = True

_N_AVOGADRO: float = 6.02214076e23


def _jithin_a_nondim(volume_nm3: float, c_scale: float) -> float:
    """Convert Jithin Table 4.1 volume (nm³/molecule) → Bikerman a_nondim.

    a_phys [m³/mol] = V_nm³ · 1e-27 · N_A
    a_nondim       = a_phys · C_SCALE     [unitless]
    """
    a_phys = volume_nm3 * 1e-27 * _N_AVOGADRO
    return a_phys * c_scale


def _stern_bump_ladder(target: float) -> List[float]:
    """Pick intermediate C_S rungs from STERN_ANCHOR up to ``target``.

    Mirrors ``solver_demo_slide15_no_speculative_cs._stern_bump_ladder``
    (memory: project_no_stern_bump_ladder.md).  Truncates the verified
    ladder at the first rung ≥ target and appends the exact target.
    """
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

    # Override species c0 / a values with Jithin's Table 4.1 numbers
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

    # Jithin Table 4.1 D_O2 = 1.5e-9; D_HP/D_H2O2 match our defaults
    d_o2_hat_jithin = float(D_O2_JITHIN_PHYS) / float(D_REF)

    species = dataclasses.replace(
        THREE_SPECIES_LOGC_BOLTZMANN,
        d_vals_hat=[d_o2_hat_jithin, float(D_H2O2_HAT), float(D_HP_HAT)],
        a_vals_hat=[a_o2_hat, a_h2o2_hat, a_hp_hat],
        c0_vals_hat=[c_o2_hat, c_h2o2_seed_hat, c_hp_hat],
    )

    # Counterion entries — pH-2 Cs⁺/SO₄²⁻ with Jithin's a values
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

    # Tafel-only single peroxide-pathway reaction (Jithin Eq 3.46 / 4.32):
    #   anodic_species=None + reversible=False  ⇒  cathodic-only BV branch
    # k0 scaled by K0_R2E_JITHIN_FACTOR (~1e-10) to push half-wave 0.3 V
    # cathodic — matches Jithin's onset near V_RHE ≈ +0.25 V.
    k0_jithin = float(K0_HAT_R2E) * float(K0_R2E_JITHIN_FACTOR)
    rxns: List[Dict[str, Any]] = [
        {
            "k0": k0_jithin,
            "alpha": float(ALPHA_TAFEL_USED),
            "cathodic_species": 0,            # O₂
            "anodic_species": None,           # Tafel — no anodic branch
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],    # 3sp: O₂, H₂O₂, H⁺
            "n_electrons": int(N_ELECTRONS_TAFEL),
            "reversible": False,
            "E_eq_v": 0.695,                  # peroxide thermodynamic E°
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
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: k0_jithin}
    return sp, k0_targets


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
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  Jithin (2024) Fig 4.36 emulation — Tafel-only R2e, Cs⁺/SO₄ pH 2")
    print("=" * 78)
    print(f"  V grid     = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)")
    print(f"  anchor V   = {ANCHOR_V_RHE:+.3f} V")
    alpha_neff = ALPHA_TAFEL_USED * N_ELECTRONS_TAFEL
    print(f"  α (R2e)    = {ALPHA_TAFEL_USED:.3f}  "
          f"(α·n_e = {alpha_neff:.3f}, "
          f"effective Tafel slope = {A_TAFEL_USED_MV_DEC:.1f} mV/dec)")
    if ALPHA_TAFEL_USED < _ALPHA_TARGET - 1e-6:
        print(f"             [Jithin Fig 4.36 fitted α≈{_ALPHA_TARGET:.3f} "
              f"({A_TAFEL_JITHIN_FIG436_MV_DEC} mV/dec) clamped to "
              f"α=1.0 by validator]")
    print(f"  L_eff      = {L_EFF_M * 1e6:.1f} μm")
    bump_ladder = _stern_bump_ladder(STERN_TARGET)
    print(f"  C_S        = {STERN_ANCHOR:.3f} → "
          f"{STERN_TARGET:.3f} F/m² via {bump_ladder!r}")
    print(f"  k0 (R2e)   = {K0_R2E_JITHIN_FACTOR:.2e} × K0_HAT_R2E")
    print(f"  D_O2       = {D_O2_JITHIN_PHYS:.2e} m²/s (Jithin Table 4.1)")
    print(f"  packing_floor = {PACKING_FLOOR_EXPERIMENT_A:.0e} "
          f"(Experiment A; default is 1e-8)")
    print(f"  bv_steric_activity = {BV_STERIC_ACTIVITY}  "
          f"(Jithin-mode θ multiplier on BV rate)")
    print(f"  bulk O₂    = {C_O2_JITHIN_MOL_M3} mol/m³")
    print(f"  bulk H⁺    = {C_HP_JITHIN_MOL_M3} mol/m³ (pH 2)")
    print(f"  bulk Cs⁺   = {C_CS_JITHIN_MOL_M3} mol/m³")
    print(f"  bulk SO₄²⁻ = {C_SO4_JITHIN_MOL_M3} mol/m³")
    print(f"  output     = {OUT_DIR}")
    print("=" * 78)

    domain_height_hat = float(L_EFF_M) / float(L_REF)
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    sp_baseline, k0_targets = _make_sp(stern_capacitance_f_m2=STERN_TARGET)
    sp_anchor_cs, _ = _make_sp(stern_capacitance_f_m2=STERN_ANCHOR)
    sp_anchor = sp_anchor_cs.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

    print(f"\nStage 1: anchor build at V={ANCHOR_V_RHE:+.3f} V, "
          f"C_S={STERN_ANCHOR:.3f} F/m²", flush=True)
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
            )
    except LadderExhausted as exc:
        raise RuntimeError(f"Anchor failed: {exc}") from exc
    if not anchor_result.converged:
        raise RuntimeError("Anchor did not converge at C_S=0.10.")
    print(f"  anchor done in {time.time() - t0:.1f}s; "
          f"ladder={list(anchor_result.ladder_history)!r}", flush=True)

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = ctx_anchor["U"].function_space().dim()

    print(f"\nStage 2: Stern bump ladder {STERN_ANCHOR:.3f} → "
          f"{STERN_TARGET:.3f} F/m² via {bump_ladder!r}", flush=True)
    t_bump = time.time()
    bump_history: List[Tuple[float, str]] = []
    for cs_target in bump_ladder:
        t_rung = time.time()
        try:
            set_stern_capacitance_model(ctx_anchor, float(cs_target))
            with adj.stop_annotating():
                ctx_anchor["_last_solver"].solve()
            bump_history.append((float(cs_target), "ok"))
            print(f"  rung C_S={cs_target:.3f} F/m² ok "
                  f"({time.time() - t_rung:.1f}s)", flush=True)
        except Exception as exc:
            bump_history.append((float(cs_target), "fail"))
            raise RuntimeError(
                f"Stern bump to C_S={cs_target:.3f} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    print(f"  Stern ladder done in {time.time() - t_bump:.1f}s "
          f"(reached C_S = {bump_ladder[-1]:.3f} F/m²)", flush=True)

    U_post_bump = snapshot_U(ctx_anchor["U"])
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
    )

    NV = len(V_RHE_GRID)
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

    phi_grid_eta = np.array(V_RHE_GRID, dtype=float) / float(V_T)
    print(f"\nStage 3: grid walk over {NV} V points", flush=True)
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

    # Extract per-V OHP-surface diagnostics for Gap 2 investigation
    # (does steric saturation of Cs⁺/H⁺ at the OHP occlude O₂ access?)
    c_o2_ohp = np.full(NV, np.nan)        # species 0 (O₂)
    c_h2o2_ohp = np.full(NV, np.nan)      # species 1 (H₂O₂)
    c_h_ohp = np.full(NV, np.nan)         # species 2 (H⁺)
    phi_ohp = np.full(NV, np.nan)
    for i in range(NV):
        if not converged[i]:
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
            float(x) if (np.isfinite(x) and converged[i]) else None
            for i, x in enumerate(arr)
        ]

    report: Dict[str, Any] = {
        "label": "jithin_emulation_fig436",
        "config": {
            "topology": "single_tafel_R2e_only",
            "alpha_tafel": float(ALPHA_TAFEL_USED),
            "alpha_times_n_e": float(alpha_neff),
            "A_tafel_mV_dec": float(A_TAFEL_USED_MV_DEC),
            "alpha_jithin_target": float(_ALPHA_TARGET),
            "A_tafel_jithin_target_mV_dec": float(
                A_TAFEL_JITHIN_FIG436_MV_DEC
            ),
            "n_electrons_tafel": int(N_ELECTRONS_TAFEL),
            "counterions": ["Cs+", "SO4(2-)"],
            "stern_anchor": float(STERN_ANCHOR),
            "stern_target": float(STERN_TARGET),
            "stern_bump_ladder": list(bump_ladder),
            "stern_note": (
                f"Jithin L_Stern = 0.6 nm → C_S = 1.16 F/m² reached via "
                f"verified bump ladder {list(bump_ladder)!r}."
            ),
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
            "enable_cation_hydrolysis": False,
            "enable_water_ionization": False,
            "v_rhe_grid": list(V_RHE_GRID),
            "anchor_v_rhe": float(ANCHOR_V_RHE),
            "mesh": {
                "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
                "domain_height_hat": float(domain_height_hat),
            },
            "formulation": FORMULATION,
            "initializer": INITIALIZER,
            "exponent_clip": float(EXPONENT_CLIP),
            "packing_floor": float(PACKING_FLOOR_EXPERIMENT_A),
            "bv_steric_activity": bool(BV_STERIC_ACTIVITY),
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
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
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
        },
        "grid_wall_seconds": float(grid_wall),
        "jithin_reference": {
            "exp_plateau_mA_cm2": -0.386,
            "fig436_simulated_plateau_mA_cm2": -0.36,
            "fig434_simulated_plateau_mA_cm2": -0.14,
            "diffusion_limit_calc_L50um_mA_cm2": -0.145,
            "diffusion_limit_calc_L10um_mA_cm2": -0.724,
            "source": "Jithin George 2024 thesis Fig 4.34/4.36 + p.138",
            "note": (
                "Fig 4.36 (his fitted-best, L_diff=10μm, A_Tafel=26.2): "
                "simulated plateau ≈ -0.36 mA/cm² matches experimental "
                "-0.386 mA/cm² closely.  Fig 4.34 (L_diff=50μm): -0.14 "
                "mA/cm² which is his Levich estimate at L=50μm."
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
        print(f"  Jithin Fig 4.36 simulated plateau     ≈ -0.36  mA/cm² "
              f"(L_diff=10μm, A_Tafel=26.2)")
        print(f"  experimental plateau                  ≈ -0.386 mA/cm²")
        print(f"  diffusion-limit calc (Jithin p.138)   ≈ -0.145 mA/cm²")

    return 0


if __name__ == "__main__":
    sys.exit(main())
