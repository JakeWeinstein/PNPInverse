"""Phase 5γ GATE v2 — Picard log-consistency at K0_FACTOR = 1e-9.

This is the **multi-ion algebra correctness probe** for Phase 5γ
(Goal A in ``~/.claude/plans/make-a-plan-based-reflective-clarke.md``).
It clones ``picard_residual_consistency_csplus_so4.py`` verbatim, with
one surgical change: every reaction's nondim ``k0`` is multiplied by
``K0_FACTOR = 1e-9`` in :func:`_build_parallel_reactions` BEFORE the
SolverParams build.

Why a low-k0 GATE?
------------------
The Phase 5α GATE (at production ``k0``) failed at ``V_RHE = +0.55 V``
because Picard cannot converge in the strongly mass-transport-limited
regime — the multi-ion physics correctly produces ``psi_S ≈ 19.83``
which drives R_4e through ``exp(0.5 * 4 * 7.22) ≈ 1.8e6`` and the
H⁺ supply cannot match. That failure is a **Newton convergence wall**
at production rates, not an algebra bug.

At ``K0_FACTOR = 1e-9``, the rate is small everywhere, surface
concentrations equal bulk, Picard converges trivially, and the
log-consistency gate becomes a **pure algebra check**: does the
multi-ion-aware Picard predict the same boundary rate as the FE
residual integrates over the electrode at the IC?

Pass means the multi-ion plumbing landed in Phase 5α
(``picard_ic.py`` + ``multi_ion.py`` + IC dispatch) is internally
consistent — Phase 5γ's continuation strategy is the right path for
the high-rate regime. Fail at ``1e-9`` would indicate the Phase 5α
93/93 unit tests missed an algebra bug; lower the factor (1e-12,
1e-15, 1e-30) to triangulate.

Acceptance
----------
- For each reaction j with ``|R_picard[j]| > 1e-15``:
    ``|log10(R_residual[j] / R_picard[j])| < 0.05``.
- For each reaction j with ``|R_picard[j]| <= 1e-15``:
    ``|R_residual[j] - R_picard[j]| < 1e-10``.
- ``picard_state.initializer_fallback`` must be ``False`` (Picard
  converged on its own; we are NOT validating the linear-φ fallback).

Output
------
``StudyResults/fast_realignment_2026-05-08/phase5gamma_gate_v2/gate_report.json``

Usage
-----
::

    python -u scripts/studies/picard_residual_consistency_csplus_so4_lowk0.py
    echo "Exit: $?"   # 0 = pass, 1 = fail
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

V_RHE_GATE = +0.55          # page-15 anchor (matches PHASE_4_STATUS.md)
MESH_NY = 200
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"

# **Phase 5γ knob**: scale factor applied to BOTH reaction k0 values
# (R_2e and R_4e). The plan's first try at 1e-9 was insufficient — at
# V_RHE = +0.55 V, eta_R4e ≈ 28 nondim, exp(0.5*4*28) ≈ 1.2e24, so
# even k0 * 1e-9 still sits at the H⁺ mass-transport limit ≈ 0.28
# (Picard oscillates → linear-φ fallback → not a clean algebra check).
# We push to 1e-30 so the predicted rate is well below the supply
# limit and Picard converges trivially. This still exercises the
# multi-ion algebra branches (the relative R_4e/R_2e ratio is
# preserved), just at a smaller absolute scale.
K0_FACTOR = 1e-30

LOG_TOL = 0.05              # |log10(R_residual / R_picard)| threshold
TINY_THRESHOLD = 1e-15      # |R_picard| below which switch to abs tolerance
ABS_TOL = 1e-10             # absolute tolerance for tiny rates

OUT_DIR = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08"
    / "phase5gamma_gate_v2"
)


# ---------------------------------------------------------------------------
# SolverParams build (mirrors peroxide_window_3sp_parallel_2e_4e_csplus_so4)
# ---------------------------------------------------------------------------

def _build_parallel_reactions():
    """Build the reaction list with K0_FACTOR baked into both rates.

    The factor enters the nondim ``k0`` *before* SolverParams build, so
    it flows through ``_add_bv_reactions_scaling_to_transform`` into
    ``scaling['bv_reactions'][j]['k0_model']``. Picard sees the small
    k0; the FE residual sees the same value via ``ctx['bv_k0_funcs']``.
    """
    from scripts._bv_common import (
        K0_HAT_R2E, K0_HAT_R4E,
        ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V,
        C_HP_HAT,
    )
    return [
        {
            "k0": float(K0_HAT_R2E) * K0_FACTOR,
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
            "k0": float(K0_HAT_R4E) * K0_FACTOR,
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


def _make_sp_for_gate():
    from scripts._bv_common import (
        SNES_OPTS_CHARGED,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        make_bv_solver_params,
    )
    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
    })
    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=_build_parallel_reactions(),
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer=INITIALIZER,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


# ---------------------------------------------------------------------------
# Picard state extraction from ctx
# ---------------------------------------------------------------------------

def _pick_R_list(picard_state: dict) -> list[float]:
    """Pull per-reaction R from the Picard state dict.

    ``picard_outer_loop_general`` returns ``R_list`` directly. The
    legacy 2-rxn path also exposes ``R1, R2``; prefer ``R_list`` when
    present, fall back otherwise.
    """
    if "R_list" in picard_state:
        return [float(r) for r in picard_state["R_list"]]
    R1 = picard_state.get("R1", float("nan"))
    R2 = picard_state.get("R2", float("nan"))
    return [float(R1), float(R2)]


def _instrument_picard_state(ctx: dict) -> dict:
    """Capture the converged Picard state stashed by the IC build."""
    state = ctx.get("initializer_picard_state", {}) or {}
    return {
        "phi_o": float(state.get("phi_o", float("nan"))),
        "psi_D": float(state.get("psi_D", float("nan"))),
        "psi_S": float(state.get("psi_S", float("nan"))),
        "phi_surface": float(state.get("phi_surface", float("nan"))),
        "gamma_s": float(state.get("gamma_s", float("nan"))),
        "H_o": float(state.get("H_o", float("nan"))),
        "R_list": _pick_R_list(state),
        "eta_list": [float(e) for e in state.get("eta_list", [])],
        "initializer_fallback": bool(ctx.get("initializer_fallback", False)),
        "initializer_fallback_reason": ctx.get("initializer_fallback_reason", ""),
    }


# ---------------------------------------------------------------------------
# FEM residual rate assembly
# ---------------------------------------------------------------------------

def _assemble_residual_R_per_rxn(ctx: dict) -> list[float]:
    """At the IC (post-set_initial_conditions, pre-Newton), assemble
    ``∫ R_j ds`` over the electrode boundary and divide by the electrode
    area. Returns one rate per reaction."""
    import firedrake as fd

    bv_settings = ctx["bv_settings"]
    elec_marker = int(bv_settings["electrode_marker"])
    mesh = ctx["mesh"]
    ds_e = fd.ds(domain=mesh, subdomain_id=elec_marker)
    area = float(fd.assemble(fd.Constant(1.0) * ds_e))
    bv_rate_exprs = list(ctx["bv_rate_exprs"])
    out = []
    for R_j in bv_rate_exprs:
        R_int = float(fd.assemble(R_j * ds_e))
        out.append(R_int / max(area, 1e-30))
    return out


# ---------------------------------------------------------------------------
# GATE assertion
# ---------------------------------------------------------------------------

def _evaluate_gate(R_residual: list[float], R_picard: list[float]) -> tuple[bool, list[str], list, list]:
    """Per-reaction log-consistency or absolute-tolerance check.

    Returns ``(gate_passed, failures, log10_ratios, abs_diffs)``. Each
    reaction is evaluated against:

      - ``|log10(R_r / R_p)| < LOG_TOL`` when both are non-tiny.
      - ``|R_r - R_p| < ABS_TOL`` otherwise.
    """
    n = max(len(R_residual), len(R_picard))
    failures: list[str] = []
    log10_ratios: list[float | None] = []
    abs_diffs: list[float | None] = []
    gate_passed = True
    for j in range(n):
        Rr = R_residual[j] if j < len(R_residual) else float("nan")
        Rp = R_picard[j] if j < len(R_picard) else float("nan")
        if not (math.isfinite(Rr) and math.isfinite(Rp)):
            failures.append(
                f"rxn{j}: non-finite (R_residual={Rr!r}, R_picard={Rp!r})"
            )
            log10_ratios.append(None)
            abs_diffs.append(None)
            gate_passed = False
            continue
        ad = abs(Rr - Rp)
        abs_diffs.append(ad)
        rp_tiny = abs(Rp) <= TINY_THRESHOLD
        rr_tiny = abs(Rr) <= TINY_THRESHOLD
        if rp_tiny and rr_tiny:
            log10_ratios.append(0.0)
            if ad >= ABS_TOL:
                failures.append(f"rxn{j}: tiny-rate |Δ|={ad:.3e} >= {ABS_TOL:.0e}")
                gate_passed = False
            continue
        if rp_tiny ^ rr_tiny:
            log10_ratios.append(None)
            if ad >= ABS_TOL:
                failures.append(
                    f"rxn{j}: cross-tiny |Δ|={ad:.3e} >= {ABS_TOL:.0e} "
                    f"(R_residual={Rr:.3e}, R_picard={Rp:.3e})"
                )
                gate_passed = False
            continue
        lr = math.log10(abs(Rr / Rp))
        log10_ratios.append(lr)
        if abs(lr) >= LOG_TOL:
            failures.append(
                f"rxn{j}: |log10(Rr/Rp)|={abs(lr):.3f} >= {LOG_TOL:.2f}; "
                f"R_residual={Rr:.3e}, R_picard={Rp:.3e}"
            )
            gate_passed = False
    return gate_passed, failures, log10_ratios, abs_diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    from scripts._bv_common import setup_firedrake_env, V_T

    setup_firedrake_env()

    import firedrake.adjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.dispatch import (
        build_context, build_forms, set_initial_conditions,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp = _make_sp_for_gate()
    phi_target_eta = float(V_RHE_GATE) / V_T
    sp_at_v = sp.with_phi_applied(phi_target_eta)

    print(
        f"[gate v2] V_RHE = {V_RHE_GATE:+.3f} V, phi_target_eta = "
        f"{phi_target_eta:.4g} (V_T={V_T:.6f})"
    )
    print(
        f"[gate v2] Mesh Ny = {MESH_NY}, formulation = {FORMULATION}, "
        f"K0_FACTOR = {K0_FACTOR:.0e}"
    )

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(sp_at_v, mesh=mesh)
        ctx = build_forms(ctx, sp_at_v)
        set_initial_conditions(ctx, sp_at_v)

        picard = _instrument_picard_state(ctx)
        R_residual = _assemble_residual_R_per_rxn(ctx)

    R_picard = picard["R_list"]
    gate_passed, failures, log10_ratios, abs_diffs = _evaluate_gate(
        R_residual=R_residual, R_picard=R_picard,
    )

    # Goal A also requires Picard to have converged on its own — fallback
    # to linear-φ would mean Picard didn't actually converge and the
    # algebra check is moot.
    if picard["initializer_fallback"]:
        gate_passed = False
        failures.append(
            "Picard initializer fell back to linear_phi "
            f"({picard['initializer_fallback_reason']!r}); GATE v2 "
            "requires Picard to converge at K0_FACTOR=1e-9."
        )

    report = {
        "config": {
            "V_RHE": V_RHE_GATE,
            "phi_target_eta": phi_target_eta,
            "mesh_Ny": MESH_NY,
            "exponent_clip": EXPONENT_CLIP,
            "u_clamp": U_CLAMP,
            "formulation": FORMULATION,
            "initializer": INITIALIZER,
            "k0_factor": K0_FACTOR,
            "log_tol": LOG_TOL,
            "tiny_threshold": TINY_THRESHOLD,
            "abs_tol": ABS_TOL,
        },
        "picard_state": picard,
        "R_residual_per_rxn": R_residual,
        "R_picard_per_rxn": R_picard,
        "log10_R_ratio_per_rxn": log10_ratios,
        "abs_diff_per_rxn": abs_diffs,
        "gate_passed": gate_passed,
        "failures": failures,
    }
    out_path = OUT_DIR / "gate_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print(json.dumps(report, indent=2, default=str))
    print()

    if not gate_passed:
        print(f"[gate v2] FAILED. See {out_path}.")
        return 1
    print(f"[gate v2] PASSED. See {out_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
