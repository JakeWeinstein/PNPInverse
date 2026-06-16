"""Jithin (2024 thesis) Section 4.8 default-settings emulation.

Reproduces three simulator-only figures from Jithin's Chapter 4 §4.8:

* Fig 4.26 — c*_O2 / c^b_O2 vs applied V (single curve, V ∈ [0.05, 0.71])
* Fig 4.27 — c_O2(y) at V = 0.3 and V = 0.66 (zoom 0–14 nm)
* Fig 4.28 — c_H+(y) at V = 0.3 and V = 0.66 (zoom 0–14 nm)

Section 4.8 Table 4.3 default settings (NOT the Fig 4.36 fitted overlay):

  * Single 2e Tafel R2e at E° = 0.695 V (NO R4e, NO anodic branch)
  * Cs⁺ / SO₄²⁻ Bikerman counterions (Table 4.3 = Table 4.1 entries)
  * Bulk: H⁺ 10, Cs⁺ 190, SO₄²⁻ 100, O₂ 0.25, H₂O₂ ≈ 0 (mol/m³, pH 2)
  * L_bulk = 50 µm  (Jithin §4.8 default; Fig 4.36 fitted 10 µm)
  * L_Stern = 0.4 nm → C_S = ε_r·ε_0 / L_Stern ≈ 1.74 F/m²
    (Jithin §4.8 default; Fig 4.36 fitted 0.6 nm → 1.16 F/m²)
  * A_Tafel = 142 mV/dec → α·n_e ≈ 0.417, so α ≈ 0.21 for n_e = 2
    (Jithin §4.8 default low-overpotential slope; Fig 4.36 fitted 26.2)
  * k0_R2E = default K0_HAT_R2E (Jithin doesn't state j₀; default gives
    j₀ ≈ 1.2e-4 mA/cm² which keeps the system kinetic-limited through
    V ≈ 0.5, same regime as Jithin's Fig 4.25 simulated jV peak.)

Why this comparison is informative:

The c_O2(OHP)/bulk → 0 transition in Fig 4.26 is driven by Bikerman steric
occlusion (cations + H⁺ saturate the OHP cavity → O₂ excluded), NOT by O₂
consumption.  Fig 4.27 + 4.28 show the matching nanoscale physics: at V=0.3
H⁺ pins at the steric ceiling and O₂ drops to zero at the OHP; at V=0.66
both species are near-bulk because the cathodic driving force is weak.

If our solver reproduces these three curves we have direct simulator-vs-
simulator confirmation that the MPNP-Bikerman-BV coupling is implemented
correctly.

Forked from ``scripts/studies/_run_jithin_emulation_fig436.py``.
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
# Jithin §4.8 (Fig 4.26/4.27/4.28) configuration
# ---------------------------------------------------------------------------
# Convention shift: Jithin's model uses Ψ_bulk = 0.785 V (his open-circuit
# potential, p. 128), while our solver implicitly uses ψ_bulk = 0.  To make
# the diffuse-layer driving field (= ψ_electrode - ψ_bulk) match between the
# two models, we shift our V_RHE grid AND our E_eq_v by -OCP_OFFSET_V.  This
# keeps the BV overpotential η = V_RHE - E_eq the same in both conventions
# (so kinetics are equivalent), while moving the double-layer driving field
# into Jithin's regime (where his "V_RHE = 0.785" is PZC-like).
OCP_OFFSET_V: float = 0.785

# Jithin Fig 4.26/4.27/4.28 x-axis is "Applied Voltage (vs RHE)" ∈ [0.05, 0.71].
# Mapped to our solver's V_RHE (raw electrode potential, ψ_bulk = 0):
#   V_RHE_ours = V_RHE_jithin - OCP_OFFSET_V  ∈  [-0.735, -0.075]
V_RHE_GRID_JITHIN: Tuple[float, ...] = tuple(
    round(float(v), 4) for v in np.linspace(0.05, 0.71, 25).tolist()
)
V_RHE_GRID: Tuple[float, ...] = tuple(
    round(float(v - OCP_OFFSET_V), 4) for v in V_RHE_GRID_JITHIN
)
ANCHOR_V_RHE_JITHIN: float = +0.71
ANCHOR_V_RHE: float = ANCHOR_V_RHE_JITHIN - OCP_OFFSET_V  # = -0.075

# Profile-capture targets: pick grid points nearest to Jithin's chosen V's
# (Fig 4.27/4.28 caption: 0.3 V and 0.66 V vs RHE).
PROFILE_V_TARGETS_JITHIN: Tuple[float, ...] = (0.30, 0.66)
PROFILE_V_TARGETS: Tuple[float, ...] = tuple(
    round(float(v - OCP_OFFSET_V), 4) for v in PROFILE_V_TARGETS_JITHIN
)

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

L_EFF_M: float = 50e-6                # Jithin §4.8 default L_bulk
STERN_ANCHOR: float = 0.10
# L_Stern = 0.4 nm, ε_r = 78.5, ε_0 = 8.854e-12 F/m → C_S ≈ 1.738 F/m²
STERN_TARGET: float = 1.738
# Stern bump ladder reused from _run_jithin_emulation_fig436 (pH-2 stiff
# Cs⁺/SO₄ stack); extend the top end to reach 1.74.
_STERN_BUMP_LADDER_VERIFIED: Tuple[float, ...] = (
    0.20, 0.35, 0.50, 0.70, 0.85, 1.0, 1.16, 1.5, 1.738,
)

# Jithin §4.8 doesn't state j₀ explicitly; use our default K0_HAT_R2E.
# Back-of-envelope: K0_PHYS_R2E = 2.4e-8 m/s → j₀ = n·F·k·c_O2_bulk ≈
# 2 · 96485 · 2.4e-8 · 0.25 ≈ 1.16e-3 A/m² ≈ 1.16e-4 mA/cm².  This keeps
# us kinetic-limited through V ≈ 0.5 (same regime as Jithin's Fig 4.25).
K0_R2E_DEFAULT_FACTOR: float = 1.0

# Jithin Table 4.3 diffusivities (m²/s).  Only D_O2 differs from our
# default; D_HP / D_H2O2 match within rounding.
D_O2_JITHIN_PHYS: float = 1.5e-9

# Match jithin_emulation_fig436 packing_floor (deeper floor for cathodic
# saturation cliff; default 1e-8 caps μ_steric near the singularity).
PACKING_FLOOR: float = 1e-15

# Jithin Table 4.3 bulk concentrations (mol/m³)
C_O2_JITHIN_MOL_M3: float = 0.25
C_HP_JITHIN_MOL_M3: float = 10.0      # pH 2
C_CS_JITHIN_MOL_M3: float = 190.0
C_SO4_JITHIN_MOL_M3: float = 100.0
C_H2O2_SEED_MOL_M3: float = 1e-6      # Jithin sets 0 exactly

# Jithin Table 4.1/4.3 ionic volumes (nm³/molecule).
V_O2_JITHIN_NM3: float = 0.064
V_H2O2_JITHIN_NM3: float = 0.16638
V_HP_JITHIN_NM3: float = 0.175616
V_CS_JITHIN_NM3: float = 0.28489
V_SO4_JITHIN_NM3: float = 0.43552

# Tafel slope → α·n_e: b = (ln10·V_T)/(α·n_e); at 298 K → α·n_e = 59.16/b_mV_dec
# Jithin §4.8 default A_Tafel = 142 mV/dec → α·n_e ≈ 0.417 → α ≈ 0.208 at n_e=2.
A_TAFEL_MV_DEC: float = 142.0
N_ELECTRONS_TAFEL: int = 2
_ALPHA_TARGET: float = (
    59.16 / A_TAFEL_MV_DEC / float(N_ELECTRONS_TAFEL)
)  # ≈ 0.208
ALPHA_TAFEL_USED: float = min(_ALPHA_TARGET, 1.0)
A_TAFEL_USED_MV_DEC: float = (
    59.16 / (ALPHA_TAFEL_USED * float(N_ELECTRONS_TAFEL))
)

OUT_DIR = Path(_ROOT) / "StudyResults" / "jithin_fig_4_26_4_27_4_28"

# NOTE: bv_steric_activity remains OFF here.  Jithin's §4.8 default model
# does NOT multiply the BV rate by θ(OHP) — that's the Fig 4.36 closure-
# form patch.  The Section 4.8 figures show pure steric exclusion through
# the Boltzmann/Bikerman concentration field, not via the rate prefactor.
BV_STERIC_ACTIVITY: bool = False

# Profile sampling: 60 y-positions, log-spaced from y=1e-4 (= 10 nm if
# L_REF=100 µm) up to the bulk (y=domain_height_hat).
N_PROFILE_SAMPLES: int = 60
PROFILE_Y_NM_MAX: float = 14.0        # Jithin Fig 4.27/4.28 x-axis range
# To resolve the EDL we want ~30 points in 0–14 nm + ~30 points 14 nm → bulk.
# Computed at runtime once L_REF is imported.

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


def _profile_y_grid_hat(domain_height_hat: float, l_ref_m: float) -> np.ndarray:
    """Build the y-sampling grid: dense near the OHP (Jithin zoom 0–14 nm),
    then log-spaced out to the bulk for the full-domain plot."""
    y_zoom_nm_max = float(PROFILE_Y_NM_MAX)
    # Inner 30 points: log from 1e-3 nm to 14 nm
    inner_nm = np.geomspace(1e-3, y_zoom_nm_max, num=30)
    # Outer 30 points: log from 14 nm to L_eff
    outer_nm = np.geomspace(
        y_zoom_nm_max, float(l_ref_m * domain_height_hat) * 1e9, num=30
    )[1:]
    nm = np.concatenate([[0.0], inner_nm, outer_nm])
    # Convert to nondim: y_hat = y_m / L_REF
    y_hat = nm * 1e-9 / float(l_ref_m)
    # Clamp at the top to avoid querying just outside the mesh
    y_hat = np.clip(y_hat, 0.0, float(domain_height_hat))
    return y_hat


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

    k0_jithin = float(K0_HAT_R2E) * float(K0_R2E_DEFAULT_FACTOR)
    # E_eq_v shifted by -OCP_OFFSET_V to match the V_RHE shift (see grid
    # construction).  Keeps η = V_RHE_ours - E_eq_v_ours = V_RHE_jithin -
    # 0.695, so kinetics are byte-identical to Jithin's BV residual.
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
    new_bv["bv_steric_activity"] = bool(BV_STERIC_ACTIVITY)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: k0_jithin}
    return sp, k0_targets


def _profile_at(U, species_idx: int, x_query: float,
                y_hat_grid: np.ndarray, *,
                mu_correction: bool = False,
                n_species: int = 3,
                em: float = 1.0, z: float = 0.0) -> np.ndarray:
    """Sample one species concentration in non-dim units along (x, y).

    For non-mu species (z=0 or logc formulation): c = exp(U.sub(i)(y)).
    For muh-mode mu species (H+ in our setup): the primary variable stored
    in U.sub(i) is μ_H = u_H + em·z_H·φ, so we must subtract the EM term
    to recover log(c).  φ lives at U.sub(n_species).
    """
    pts = np.column_stack([
        np.full_like(y_hat_grid, x_query),
        y_hat_grid,
    ])
    u_field = U.sub(species_idx)
    mu_or_u = np.asarray(u_field.at(pts.tolist(), tolerance=1e-9), dtype=float)
    if mu_correction:
        phi_field = U.sub(n_species)
        phi_vals = np.asarray(
            phi_field.at(pts.tolist(), tolerance=1e-9), dtype=float
        )
        log_c = mu_or_u - em * z * phi_vals
    else:
        log_c = mu_or_u
    return np.exp(log_c)


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
    print("  Jithin §4.8 default-settings emulation (Fig 4.26/4.27/4.28)")
    print("=" * 78)
    print(f"  V grid (Jithin) = [{V_RHE_GRID_JITHIN[0]:+.3f}, "
          f"{V_RHE_GRID_JITHIN[-1]:+.3f}] V ({len(V_RHE_GRID_JITHIN)} points)")
    print(f"  V grid (ours)   = [{V_RHE_GRID[0]:+.3f}, "
          f"{V_RHE_GRID[-1]:+.3f}] V (Jithin - {OCP_OFFSET_V:.3f} V offset)")
    print(f"  anchor V (Jithin/ours) = "
          f"{ANCHOR_V_RHE_JITHIN:+.3f} / {ANCHOR_V_RHE:+.3f} V")
    print(f"  E_eq shifted = {0.695 - OCP_OFFSET_V:+.3f} V (was 0.695)")
    alpha_neff = ALPHA_TAFEL_USED * N_ELECTRONS_TAFEL
    print(f"  α (R2e)    = {ALPHA_TAFEL_USED:.3f}  "
          f"(α·n_e = {alpha_neff:.3f}, "
          f"effective Tafel slope = {A_TAFEL_USED_MV_DEC:.1f} mV/dec)")
    print(f"  L_eff      = {L_EFF_M * 1e6:.1f} μm")
    bump_ladder = _stern_bump_ladder(STERN_TARGET)
    print(f"  C_S        = {STERN_ANCHOR:.3f} → "
          f"{STERN_TARGET:.3f} F/m² via {bump_ladder!r}")
    print(f"  k0 (R2e)   = {K0_R2E_DEFAULT_FACTOR:.2e} × K0_HAT_R2E (default)")
    print(f"  D_O2       = {D_O2_JITHIN_PHYS:.2e} m²/s (Jithin Table 4.3)")
    print(f"  bulk O₂    = {C_O2_JITHIN_MOL_M3} mol/m³")
    print(f"  bulk H⁺    = {C_HP_JITHIN_MOL_M3} mol/m³ (pH 2)")
    print(f"  bulk Cs⁺   = {C_CS_JITHIN_MOL_M3} mol/m³")
    print(f"  bulk SO₄²⁻ = {C_SO4_JITHIN_MOL_M3} mol/m³")
    print(f"  profile V (Jithin/ours) = "
          f"{PROFILE_V_TARGETS_JITHIN} / {PROFILE_V_TARGETS} V (Fig 4.27/4.28)")
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
    c_o2_ohp = np.full(NV, np.nan)
    c_h_ohp = np.full(NV, np.nan)
    c_cs_ohp = np.full(NV, np.nan)
    phi_ohp = np.full(NV, np.nan)

    # Pick the grid-V indices closest to each profile target.
    v_arr = np.asarray(V_RHE_GRID, dtype=float)
    profile_indices: Dict[int, float] = {}
    for v_target in PROFILE_V_TARGETS:
        idx = int(np.argmin(np.abs(v_arr - v_target)))
        profile_indices[idx] = float(v_target)
    print(f"\nProfile capture at indices: "
          f"{[(i, V_RHE_GRID[i]) for i in profile_indices]}")

    # y-grid for profile sampling (dense near electrode + log out to bulk)
    y_hat_grid = _profile_y_grid_hat(domain_height_hat, float(L_REF))

    # Per-V profile storage (only filled for indices in profile_indices)
    profiles: Dict[int, Dict[str, Any]] = {}

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

        # Capture full y-profile if this is a target voltage
        if orig_idx in profile_indices:
            try:
                U = ctx["U"]
                n_sp = int(ctx["n_species"])
                mu_set = frozenset(int(i) for i in ctx.get("mu_species", []))
                em = float(ctx.get("nondim", {})
                           .get("electromigration_prefactor", 1.0))
                # species 0 = O₂ (z=0, never muh), species 2 = H⁺ (z=+1,
                # muh primary in logc_muh formulation).
                c_o2_hat = _profile_at(
                    U, 0, 0.5, y_hat_grid,
                    mu_correction=False, n_species=n_sp,
                )
                z_h = 1.0
                c_hp_hat = _profile_at(
                    U, 2, 0.5, y_hat_grid,
                    mu_correction=(2 in mu_set), n_species=n_sp,
                    em=em, z=z_h,
                )
                profiles[orig_idx] = {
                    "v_rhe": float(V_RHE_GRID[orig_idx]),
                    "v_rhe_jithin": float(V_RHE_GRID_JITHIN[orig_idx]),
                    "v_target": float(profile_indices[orig_idx]),
                    "v_target_jithin": float(
                        profile_indices[orig_idx] + OCP_OFFSET_V
                    ),
                    "y_hat": y_hat_grid.tolist(),
                    "y_m": (y_hat_grid * float(L_REF)).tolist(),
                    "y_nm": (y_hat_grid * float(L_REF) * 1e9).tolist(),
                    "c_O2_hat": c_o2_hat.tolist(),
                    "c_H_hat": c_hp_hat.tolist(),
                    "c_O2_mol_m3": (c_o2_hat * float(C_SCALE)).tolist(),
                    "c_H_mol_m3": (c_hp_hat * float(C_SCALE)).tolist(),
                }
                print(f"    profile captured @ idx={orig_idx} V={V_RHE_GRID[orig_idx]:+.3f}V "
                      f"({len(y_hat_grid)} y-samples)")
            except Exception as exc:
                print(f"    profile capture failed @ idx={orig_idx}: "
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

    # OHP surface diagnostics
    for i in range(NV):
        if not converged[i]:
            continue
        diag = grid_result.points[i].diagnostics or {}
        if (v := diag.get("c0_surface_mean")) is not None:
            c_o2_ohp[i] = float(v)
        if (v := diag.get("c2_surface_mean")) is not None:
            c_h_ohp[i] = float(v)
        if (v := diag.get("c_counterion0_surface_mean")) is not None:
            c_cs_ohp[i] = float(v)
        if (v := diag.get("phi_surface_mean")) is not None:
            phi_ohp[i] = float(v)

    # Compute c_O2(OHP)/bulk ratio (Fig 4.26 y-axis)
    c_o2_bulk_hat = float(C_O2_JITHIN_MOL_M3) / float(C_SCALE)
    ratio_o2_ohp_bulk = c_o2_ohp / c_o2_bulk_hat

    def _to_json_list(arr: np.ndarray) -> List[Any]:
        return [
            float(x) if (np.isfinite(x) and converged[i]) else None
            for i, x in enumerate(arr)
        ]

    report: Dict[str, Any] = {
        "label": "jithin_fig_4_26_4_27_4_28",
        "config": {
            "topology": "single_tafel_R2e_only",
            "section": "4.8 default settings",
            "alpha_tafel": float(ALPHA_TAFEL_USED),
            "alpha_times_n_e": float(alpha_neff),
            "A_tafel_mV_dec": float(A_TAFEL_USED_MV_DEC),
            "alpha_jithin_target": float(_ALPHA_TARGET),
            "A_tafel_jithin_target_mV_dec": float(A_TAFEL_MV_DEC),
            "n_electrons_tafel": int(N_ELECTRONS_TAFEL),
            "counterions": ["Cs+", "SO4(2-)"],
            "ocp_offset_v": float(OCP_OFFSET_V),
            "e_eq_v_shifted": float(0.695 - OCP_OFFSET_V),
            "e_eq_v_jithin": 0.695,
            "convention_note": (
                "Our V_RHE and E_eq are shifted by -OCP_OFFSET_V (= "
                f"-{OCP_OFFSET_V:.3f}) relative to Jithin's so the diffuse-"
                "layer driving field (= ψ_electrode - ψ_bulk) matches.  "
                "BV η = V_RHE_ours - E_eq_ours = V_RHE_jithin - 0.695."
            ),
            "stern_anchor": float(STERN_ANCHOR),
            "stern_target": float(STERN_TARGET),
            "stern_bump_ladder": list(bump_ladder),
            "stern_note": (
                "Jithin §4.8 L_Stern = 0.4 nm → C_S = ε_r·ε_0/L_Stern ≈ "
                "1.738 F/m² (ε_r = 78.5)"
            ),
            "l_eff_m": float(L_EFF_M),
            "d_o2_phys_jithin_m2_s": float(D_O2_JITHIN_PHYS),
            "k0_r2e_factor": float(K0_R2E_DEFAULT_FACTOR),
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
            "v_rhe_grid_jithin": list(V_RHE_GRID_JITHIN),
            "anchor_v_rhe": float(ANCHOR_V_RHE),
            "anchor_v_rhe_jithin": float(ANCHOR_V_RHE_JITHIN),
            "mesh": {
                "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
                "domain_height_hat": float(domain_height_hat),
            },
            "formulation": FORMULATION,
            "initializer": INITIALIZER,
            "exponent_clip": float(EXPONENT_CLIP),
            "packing_floor": float(PACKING_FLOOR),
            "bv_steric_activity": bool(BV_STERIC_ACTIVITY),
            "u_clamp": float(U_CLAMP),
            "profile_v_targets": list(PROFILE_V_TARGETS),
            "profile_v_targets_jithin": list(PROFILE_V_TARGETS_JITHIN),
        },
        "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
        "v_rhe": list(V_RHE_GRID),
        "v_rhe_jithin": list(V_RHE_GRID_JITHIN),
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": _to_json_list(cd_arr),
        "c_O2_OHP_nondim": _to_json_list(c_o2_ohp),
        "c_O2_OHP_mol_m3": _to_json_list(c_o2_ohp * float(C_SCALE)),
        "c_O2_bulk_nondim": float(c_o2_bulk_hat),
        "c_O2_bulk_mol_m3": float(C_O2_JITHIN_MOL_M3),
        "ratio_O2_OHP_over_bulk": _to_json_list(ratio_o2_ohp_bulk),
        "c_H_OHP_nondim": _to_json_list(c_h_ohp),
        "c_Cs_OHP_nondim": _to_json_list(c_cs_ohp),
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
        "profiles": [
            {
                "v_rhe": p["v_rhe"],
                "v_rhe_jithin": p["v_rhe_jithin"],
                "v_target": p["v_target"],
                "v_target_jithin": p["v_target_jithin"],
                "y_nm": p["y_nm"],
                "y_m": p["y_m"],
                "y_hat": p["y_hat"],
                "c_O2_mol_m3": p["c_O2_mol_m3"],
                "c_H_mol_m3": p["c_H_mol_m3"],
                "c_O2_hat": p["c_O2_hat"],
                "c_H_hat": p["c_H_hat"],
            }
            for _idx, p in sorted(profiles.items())
        ],
    }
    out_path = OUT_DIR / "iv_curve.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")

    cd = np.array(
        [x if x is not None else np.nan for x in report["cd_mA_cm2"]]
    )
    ratio = np.array(
        [x if x is not None else np.nan
         for x in report["ratio_O2_OHP_over_bulk"]]
    )
    v_arr_np = np.array(V_RHE_GRID)
    finite = np.isfinite(cd)
    if finite.any():
        i_min = int(np.nanargmin(cd))
        print(f"  most cathodic |cd|: V = {v_arr_np[i_min]:+.3f} V, "
              f"cd = {cd[i_min]:.4f} mA/cm²")
        # Print ratio at Jithin's reference voltages
        v_jithin_np = v_arr_np + OCP_OFFSET_V
        print(f"  c_O2(OHP)/c^b_O2 sample (Jithin V_RHE):")
        for v_jithin_check in [0.05, 0.30, 0.50, 0.66, 0.71]:
            i = int(np.argmin(np.abs(v_jithin_np - v_jithin_check)))
            print(f"    V_jithin={v_jithin_check:+.2f} (V_ours={v_arr_np[i]:+.3f}): "
                  f"ratio={ratio[i]:.4f}")

    print(f"\n  profiles captured: {len(report['profiles'])} V points")
    for p in report["profiles"]:
        c_o2_ohp_val = p["c_O2_mol_m3"][0]
        c_h_ohp_val = p["c_H_mol_m3"][0]
        print(f"    V_jithin={p['v_rhe_jithin']:+.3f}V "
              f"(V_ours={p['v_rhe']:+.3f}): "
              f"c_O2(OHP)={c_o2_ohp_val:.4e} mol/m³, "
              f"c_H(OHP)={c_h_ohp_val:.4e} mol/m³")

    return 0


if __name__ == "__main__":
    sys.exit(main())
