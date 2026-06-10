"""Phase 7 dual-pathway driver — acid + water-donor ORR vs slide-15.

Forked from ``solver_demo_slide15_ocp_shifted_cs.py``.  Adds the
water-as-proton-donor routes (``PARALLEL_2E_4E_DUAL_PATHWAY``) with the
verified water-route anchor recipe, per-reaction route ledger, and
first-class local-pH diagnostics.

Water-route anchor recipe (Phase 7 finding, locked by
``tests/test_phase7_dual_pathway.py::TestWaterRouteEscapesLevichCap``):
  * anchor at V_solver = 0 (deck +0.903 V, flat double layer)
  * ``initializer='linear_phi'`` — the debye_boltzmann Picard IC
    mis-seeds water-route reactions (rung-1 divergence, verified)
  * NO kw_eff_ladder — anchor at FULL Kw; the Kw=0 floor is unphysical
    for water routes (H+-equivalent sink with no c_H damping and no
    water reservoir -> mu_H blowup)
  * k0 AdaptiveLadder with ``max_inserts_per_step=6`` (hard band
    k0_scale ~ 1e-8..1e-6 where water demand crosses the O2 limit)
  * Stern bump 0.10 -> 0.20 after the anchor, then grid walk.

Defaults: Cs+/SO4(2-), pH 4, OCP shift ON, L_eff = 15.4 um (O2
Levich-equivalent at 1600 rpm).

Usage (from PNPInverse/, venv-firedrake):
    python -u scripts/studies/solver_demo_slide15_dual_pathway_cs.py \\
        --k0-water-2e-factor=1.0 --k0-water-4e-factor=1.0 \\
        --routes=acid,water --coarse-grid --out-name=phase7_dp_smoke
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
# Conventions (OCP shift per CLAUDE.md Hard Rule #8)
# ---------------------------------------------------------------------------

PH_DECK = 4.0
V_OCP_RHE = 0.47 + 0.197 + 0.059 * PH_DECK   # 0.903 V at pH 4

V_RHE_DECK_GRID_FINE = tuple(np.linspace(-0.40, +0.55, 25).round(4).tolist())
V_RHE_DECK_GRID_COARSE = tuple(np.linspace(-0.40, +0.55, 13).round(4).tolist())

ANCHOR_V_RHE = 0.0           # solver convention: deck +0.903 V (rest state)

MESH_NX = 8
MESH_NY = 80
MESH_BETA = 3.0
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
FORMULATION = "logc_muh"
ANCHOR_INITIALIZER = "linear_phi"     # water-route recipe (see module doc)

INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP = 6
IC_AT_TARGET = True

L_EFF_UM_DEFAULT = 15.4
STERN_ANCHOR = 0.10
STERN_BASELINE = 0.20

H_SPECIES_INDEX = 2
O2_SPECIES_INDEX = 0

_C_SCALE = 1.2
_N_A = 6.02214076e23


def _a_nondim_from_radius_m(r_m: float) -> float:
    a_phys = (4.0 / 3.0) * math.pi * r_m ** 3 * _N_A
    return a_phys * _C_SCALE


A_O2_PHYSICAL = _a_nondim_from_radius_m(1.70e-10)
A_H2O2_PHYSICAL = _a_nondim_from_radius_m(2.00e-10)
A_HP_PHYSICAL = _a_nondim_from_radius_m(2.80e-10)


# ---------------------------------------------------------------------------
# Reactions + SolverParams factory
# ---------------------------------------------------------------------------


def _build_reactions(opts) -> list[dict]:
    """OCP-shifted dual-pathway reaction list with CLI overrides.

    E° values come UNSHIFTED from the preset (repo invariant) and are
    shifted by −V_OCP_RHE here — the single shift point.  Route
    disabling is via k0=0 (the parser drops nothing; ablation
    convention — ``"enabled"`` does not survive the parser).
    """
    from scripts._bv_common import PARALLEL_2E_4E_DUAL_PATHWAY

    routes = {tok.strip() for tok in opts.routes.split(",") if tok.strip()}
    if not routes <= {"acid", "water"}:
        raise SystemExit(f"--routes must be subset of acid,water; got {opts.routes!r}")

    rxns = []
    for rxn in PARALLEL_2E_4E_DUAL_PATHWAY:
        r = dict(rxn)
        r["E_eq_v"] = float(r["E_eq_v"]) - V_OCP_RHE
        is_water = r["proton_donor"] == "water"
        if is_water:
            if "water" not in routes:
                r["k0"] = 0.0
            else:
                factor = (opts.k0_water_2e_factor if r["n_electrons"] == 2
                          else opts.k0_water_4e_factor)
                r["k0"] = float(r["k0"]) * float(factor)
                alpha = (opts.alpha_water_2e if r["n_electrons"] == 2
                         else opts.alpha_water_4e)
                if alpha is not None:
                    r["alpha"] = float(alpha)
        else:
            if "acid" not in routes:
                r["k0"] = 0.0
            elif r["n_electrons"] == 4:
                r["k0"] = float(r["k0"]) * float(opts.k0_acid_4e_factor)
        rxns.append(r)
    return rxns


def _make_sp(opts, reactions, *, stern_capacitance_f_m2, initializer):
    from scripts._bv_common import (
        C_HP_HAT, C_O2_HAT, H2O2_SEED_NONDIM,
        D_H2O2_HAT, D_HP_HAT, D_O2_HAT,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R1, ALPHA_R1,
        SNES_OPTS_CHARGED,
        SpeciesConfig,
        make_bv_solver_params,
        setup_firedrake_env,
    )
    setup_firedrake_env()

    c_hp_hat = float(opts.bulk_h_mol_m3) / _C_SCALE

    species = SpeciesConfig(
        n_species=3,
        z_vals=[0, 0, 1],
        d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
        a_vals_hat=[A_O2_PHYSICAL, A_H2O2_PHYSICAL, A_HP_PHYSICAL],
        c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, c_hp_hat],
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
        bv_reactions=reactions,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        initializer=str(initializer),
        l_eff_m=float(opts.l_eff_um) * 1e-6,
        enable_water_ionization=bool(opts.enable_water_ionization),
        **stern_kw,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {
        j: float(r["k0"]) for j, r in enumerate(reactions) if float(r["k0"]) > 0.0
    }
    return sp, k0_targets


_STERN_BUMP_LADDER_VERIFIED = (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)


def _stern_bump_ladder(target: float) -> list[float]:
    rungs: list[float] = []
    for rung in _STERN_BUMP_LADDER_VERIFIED:
        if rung >= target:
            rungs.append(float(target))
            return rungs
        rungs.append(float(rung))
    if rungs[-1] < target:
        rungs.append(float(target))
    return rungs


# ---------------------------------------------------------------------------
# Per-V capture: currents, route ledger, local pH
# ---------------------------------------------------------------------------


def _capture_point(ctx, reactions) -> dict:
    """Assemble per-reaction rates, route ledger, and local-pH state."""
    import firedrake as fd
    from scripts._bv_common import I_SCALE
    from Forward.bv_solver.observables import (
        N_ELECTRONS_REF, _build_bv_observable_form,
    )
    from Forward.bv_solver.diagnostics import surface_ph

    out: dict = {}
    bv_cfg = ctx.get("bv_settings", {})
    marker = int(bv_cfg.get("electrode_marker", 1))
    ds_e = fd.Measure("ds", domain=ctx["mesh"])(marker)

    # Total electron-weighted current density (mA/cm2, cathodic negative).
    f_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
    )
    out["cd_mA_cm2"] = float(fd.assemble(f_cd))

    # Gross H2O2 production current: role-resolved sum of the 2e channels
    # (raw 2e-units == physical for n_e=2).
    f_pc = _build_bv_observable_form(
        ctx, mode="reaction_sum", reaction_index=None, scale=-I_SCALE,
    )
    out["pc_mA_cm2"] = float(fd.assemble(f_pc))

    # Per-reaction: raw rate (2e-current units) + electron-weighted current
    # + cathodic/anodic branch decomposition.
    rates_raw, rates_ew, branches = [], [], []
    branch_exprs = ctx.get("bv_rate_branch_exprs") or []
    for j, rxn in enumerate(reactions):
        if float(rxn["k0"]) <= 0.0:
            rates_raw.append(0.0)
            rates_ew.append(0.0)
            branches.append({"cathodic": 0.0, "anodic": 0.0})
            continue
        f_raw = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=j, scale=-I_SCALE,
        )
        raw = float(fd.assemble(f_raw))
        rates_raw.append(raw)
        rates_ew.append(raw * float(rxn["n_electrons"]) / float(N_ELECTRONS_REF))
        if j < len(branch_exprs):
            cat, anod = branch_exprs[j]
            branches.append({
                "cathodic": float(fd.assemble(-I_SCALE * cat * ds_e)),
                "anodic": float(fd.assemble(-I_SCALE * anod * ds_e)),
            })
        else:
            branches.append({"cathodic": None, "anodic": None})
    out["per_reaction"] = [
        {
            "label": rxn.get("label", f"reaction_{j}"),
            "proton_donor": rxn.get("proton_donor", "hydronium"),
            "n_electrons": int(rxn["n_electrons"]),
            "rate_2e_units_mA_cm2": rates_raw[j],
            "current_mA_cm2": rates_ew[j],
            "cathodic_mA_cm2": branches[j]["cathodic"],
            "anodic_mA_cm2": branches[j]["anodic"],
        }
        for j, rxn in enumerate(reactions)
    ]

    # Route ledger.
    ew_sum = sum(rates_ew)
    out["ledger"] = {
        # Electron-current consistency: cd must equal the electron-weighted
        # per-reaction sum (regression on the n_e/2 weighting).
        "electron_consistency_residual_mA_cm2": out["cd_mA_cm2"] - ew_sum,
        # Acid vs water share of the proton-equivalent (E) sink.
        "e_sink_acid_mA_cm2": sum(
            rates_ew[j] for j, r in enumerate(reactions)
            if r.get("proton_donor", "hydronium") == "hydronium"
        ),
        "e_sink_water_mA_cm2": sum(
            rates_ew[j] for j, r in enumerate(reactions)
            if r.get("proton_donor", "hydronium") == "water"
        ),
        # O2 consumption (mol-flux in 2e-current units) per reaction sums;
        # |stoich_O2| = 1 for every ORR channel here.
        "o2_consumption_2e_units_mA_cm2": sum(
            rates_raw[j] for j, r in enumerate(reactions)
            if r["stoichiometry"][O2_SPECIES_INDEX] != 0
        ),
        # 2e-channel anodic share (topology check: must stay < 1%).
        "anodic_share_2e": _anodic_share(branches, reactions),
    }
    out.update(surface_ph(ctx, h_species_index=H_SPECIES_INDEX,
                          c_scale_mol_m3=_C_SCALE))
    return out


def _anodic_share(branches, reactions) -> float | None:
    cat_tot, anod_tot = 0.0, 0.0
    for j, rxn in enumerate(reactions):
        if not rxn.get("produces_h2o2"):
            continue
        b = branches[j]
        if b["cathodic"] is None:
            return None
        cat_tot += abs(b["cathodic"])
        anod_tot += abs(b["anodic"])
    if cat_tot <= 0.0:
        return None
    return anod_tot / cat_tot


# ---------------------------------------------------------------------------
# Anchor + Stern bump + grid walk
# ---------------------------------------------------------------------------


def _run(opts) -> dict:
    from scripts._bv_common import V_T
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        PreconvergedAnchor,
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.grid_per_voltage import (
        snapshot_U, solve_grid_with_anchor,
    )

    reactions = _build_reactions(opts)
    v_deck_grid = (V_RHE_DECK_GRID_COARSE if opts.coarse_grid
                   else V_RHE_DECK_GRID_FINE)
    v_grid = tuple(round(v - V_OCP_RHE, 6) for v in v_deck_grid)
    NV = len(v_grid)

    sp_baseline, k0_targets = _make_sp(
        opts, reactions,
        stern_capacitance_f_m2=STERN_BASELINE,
        initializer=ANCHOR_INITIALIZER,
    )
    sp_anchor_cs, _ = _make_sp(
        opts, reactions,
        stern_capacitance_f_m2=STERN_ANCHOR,
        initializer=ANCHOR_INITIALIZER,
    )
    sp_anchor = sp_anchor_cs.with_phi_applied(float(ANCHOR_V_RHE) / float(V_T))

    print(f"  routes={opts.routes!r}  k0_targets={ {j: f'{k:.3e}' for j, k in k0_targets.items()} }",
          flush=True)
    print(f"  L_eff={opts.l_eff_um:.1f} um  bulk_H+={opts.bulk_h_mol_m3:g} mol/m3  "
          f"water_ionization={opts.enable_water_ionization}", flush=True)

    domain_height_hat = float(opts.l_eff_um) * 1e-6 / 1.0e-4
    mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=float(domain_height_hat),
    )

    # Stage 1: anchor at rest V, full Kw, k0 ladder.
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
                kw_eff_ladder=None,
            )
        anchor_ok = bool(anchor_result.converged)
        anchor_err = None
    except (LadderExhausted, Exception) as exc:
        anchor_ok = False
        anchor_err = f"{type(exc).__name__}: {exc}"
    anchor_wall = time.time() - t0
    if not anchor_ok:
        print(f"  anchor FAILED in {anchor_wall:.1f}s: {anchor_err}", flush=True)
        return {"anchor_converged": False, "anchor_error": anchor_err,
                "config": _config_dict(opts, reactions, v_grid, v_deck_grid)}
    print(f"  anchor ok in {anchor_wall:.1f}s "
          f"({len(anchor_result.ladder_history)} rungs)", flush=True)

    # Stage 2: Stern bump.
    ctx_anchor = anchor_result.ctx
    bump_ladder = _stern_bump_ladder(STERN_BASELINE)
    for cs_target in bump_ladder:
        set_stern_capacitance_model(ctx_anchor, float(cs_target))
        with adj.stop_annotating():
            ctx_anchor["_last_solver"].solve()
    print(f"  Stern bump -> {STERN_BASELINE} F/m^2 ok", flush=True)

    anchor = PreconvergedAnchor(
        phi_applied_eta=float(ANCHOR_V_RHE) / float(V_T),
        U_snapshot=tuple(
            np.asarray(a).copy() for a in snapshot_U(ctx_anchor["U"])
        ),
        k0_targets=tuple(sorted((int(j), float(k)) for j, k in k0_targets.items())),
        mesh_dof_count=int(ctx_anchor["U"].function_space().dim()),
        ladder_history=tuple(
            (float(s), str(o)) for s, o in anchor_result.ladder_history
        ),
    )

    # Stage 3: grid walk with full capture.
    points: dict[int, dict] = {}

    def _grab(orig_idx, _phi_eta, ctx):
        try:
            points[orig_idx] = _capture_point(ctx, reactions)
        except Exception as exc:
            print(f"    capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    phi_grid_eta = np.array(v_grid, dtype=float) / float(V_T)
    t0 = time.time()
    with adj.stop_annotating():
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
    n_conv = sum(converged)
    print(f"  grid: {n_conv}/{NV} converged in {grid_wall:.1f}s", flush=True)

    def _col(key, default=None):
        return [
            points.get(i, {}).get(key, default) if converged[i] else None
            for i in range(NV)
        ]

    return {
        "v_rhe": list(v_grid),
        "v_rhe_deck": list(v_deck_grid),
        "cd_mA_cm2": _col("cd_mA_cm2"),
        "pc_mA_cm2": _col("pc_mA_cm2"),
        "surface_pH": _col("surface_pH"),
        "surface_pOH": _col("surface_pOH"),
        "ph_poh_minus_pkw": _col("ph_poh_minus_pkw"),
        "per_reaction": _col("per_reaction"),
        "ledger": _col("ledger"),
        "converged": converged,
        "n_converged": int(n_conv),
        "n_total": int(NV),
        "anchor_converged": True,
        "anchor_wall_seconds": float(anchor_wall),
        "anchor_ladder_history": [
            [float(s), str(o)] for s, o in anchor_result.ladder_history
        ],
        "grid_wall_seconds": float(grid_wall),
        "config": _config_dict(opts, reactions, v_grid, v_deck_grid),
    }


def _config_dict(opts, reactions, v_grid, v_deck_grid) -> dict:
    return {
        "driver": "solver_demo_slide15_dual_pathway_cs",
        "routes": opts.routes,
        "k0_water_2e_factor": float(opts.k0_water_2e_factor),
        "k0_water_4e_factor": float(opts.k0_water_4e_factor),
        "k0_acid_4e_factor": float(opts.k0_acid_4e_factor),
        "alpha_water_2e": opts.alpha_water_2e,
        "alpha_water_4e": opts.alpha_water_4e,
        "l_eff_um": float(opts.l_eff_um),
        "bulk_h_mol_m3": float(opts.bulk_h_mol_m3),
        "enable_water_ionization": bool(opts.enable_water_ionization),
        "coarse_grid": bool(opts.coarse_grid),
        "ocp_shift": {"pH": PH_DECK, "V_OCP_RHE": V_OCP_RHE,
                      "applied_to": ["V_RHE", "all reaction E_eq_v"]},
        "anchor": {"v_rhe_solver": ANCHOR_V_RHE,
                   "initializer": ANCHOR_INITIALIZER,
                   "kw_eff_ladder": None,
                   "max_inserts_per_step": MAX_INSERTS_PER_STEP},
        "stern": {"anchor_f_m2": STERN_ANCHOR, "final_f_m2": STERN_BASELINE},
        "mesh": {"Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA},
        "exponent_clip": EXPONENT_CLIP,
        "u_clamp": U_CLAMP,
        "formulation": FORMULATION,
        "reactions": [
            {k: v for k, v in r.items()} for r in reactions
        ],
        "v_grid_n": len(v_grid),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--routes", default="acid,water",
                        help="Comma subset of acid,water (default acid,water).")
    parser.add_argument("--k0-water-2e-factor", type=float, default=1.0)
    parser.add_argument("--k0-water-4e-factor", type=float, default=1.0)
    parser.add_argument("--k0-acid-4e-factor", type=float, default=1e-15,
                        help="K0_R4e_acid multiplier (v10b production 1e-15"
                             " band; acid 4e is H+-starved anyway).")
    parser.add_argument("--alpha-water-2e", type=float, default=None)
    parser.add_argument("--alpha-water-4e", type=float, default=None)
    parser.add_argument("--l-eff-um", type=float, default=L_EFF_UM_DEFAULT)
    parser.add_argument("--bulk-h-mol-m3", type=float, default=0.1,
                        help="Bulk H+ (mol/m3). 0.1 = pH 4; 1.1 = bisulfate"
                             " stress-test upper bracket.")
    parser.add_argument("--coarse-grid", action="store_true",
                        help="13-pt deck grid (fit iterations) vs 25-pt.")
    parser.add_argument("--no-water-ionization", dest="enable_water_ionization",
                        action="store_false", default=True)
    parser.add_argument("--out-name", default="phase7_dual_pathway_smoke")
    opts = parser.parse_args()

    if "water" in opts.routes and not opts.enable_water_ionization:
        raise SystemExit("--routes=water requires water ionization "
                         "(drop --no-water-ionization)")

    out_dir = Path(_ROOT) / "StudyResults" / "phase7_dual_pathway" / opts.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("  Phase 7 dual-pathway driver — slide-15 (Cs+/SO4, pH 4, OCP-shifted)",
          flush=True)
    print("=" * 78, flush=True)

    t0 = time.time()
    report = _run(opts)
    report["wall_seconds"] = time.time() - t0

    out_path = out_dir / "iv_curve.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)
    print(f"total wall = {report['wall_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
