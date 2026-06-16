"""Phase 5γ anchor smoke (Goal B) — k0 continuation at V_RHE = +0.55 V.

Clones ``anchor_smoke_csplus_so4.py`` SolverParams setup verbatim and
swaps the single-shot orchestrator call for
:func:`Forward.bv_solver.anchor_continuation.solve_anchor_with_continuation`.
This is the **continuation correctness probe** for Phase 5γ Goal B:
does ramping ``k0`` from a small floor to production targets via
geometric rungs converge Newton at the failing-voltage anchor?

The companion GATE v2 (Goal A) at ``K0_FACTOR = 1e-9`` validates the
multi-ion algebra at low rate. This script validates that the
continuation strategy carries that converged low-rate state up to
production rates.

Configuration mirrors the page-15 anchor in
``StudyResults/fast_realignment_2026-05-08/PHASE_5_ALPHA_GATE_FAILURE.md``:

* ``V_RHE = +0.55 V``
* ``Ny = 80`` (fast turnaround; full Pass A grid waits for follow-up)
* Both reactions enabled (parallel-2e + parallel-4e), at production
  ``K0_HAT_R2E`` / ``K0_HAT_R4E``.

Pass criteria
-------------
- exit code 0
- ``report['converged'] == True``
- ``report['ladder_history'][-1] == [1.0, 'ok']``
- ``report['rungs'][-1]['snes_converged'] == True``

Output
------
``StudyResults/fast_realignment_2026-05-08/phase5gamma_anchor_smoke/anchor_smoke_continuation.json``

Usage
-----
::

    python -u scripts/studies/anchor_smoke_csplus_so4_continuation.py
    echo "Exit: $?"   # 0 = pass, 1 = fail
"""
from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANCHOR_V_RHE = +0.55
MESH_NY = 80
EXPONENT_CLIP = 100.0

# Geometric ladder: 5 rungs spanning 12 decades (1e-12 → 1.0). Phase 5γ
# MVP — denser ladders are deferred to follow-up if Goal B fails on
# this baseline.
INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP = 4

# Build IC at production k0 (Picard sees the production rate; may fall
# back to linear-φ on its own, which is fine because the ladder ramps
# from the floor). ``ic_at_target=False`` is the diagnostic mode and
# is reserved for follow-up if Goal B fails with ``True``.
IC_AT_TARGET = True

OUT_DIR = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08"
    / "phase5gamma_anchor_smoke"
)


def main() -> int:
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        V_T, I_SCALE,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E, K0_HAT_R4E, ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V, C_HP_HAT,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            "k0": float(K0_HAT_R4E),
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

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc_muh", log_rate=True,
        u_clamp=100.0,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer="debye_boltzmann",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    sp_at_v = sp.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    print(f"Anchor smoke (Phase 5γ continuation) @ V_RHE={ANCHOR_V_RHE:+.3f} V "
          f"(Ny={MESH_NY})")
    print(f"  multi-ion:  Cs+ + SO4-- + H+ at I=0.3 M")
    print(f"  formulation: logc_muh + log_rate + Stern (C_S=0.10) + "
          f"Bikerman (multi)")
    print(f"  IC: debye_boltzmann (ic_at_target={IC_AT_TARGET})")
    print(f"  ladder: {INITIAL_SCALES} (max_inserts_per_step="
          f"{MAX_INSERTS_PER_STEP})")
    print(f"  k0 targets: R2e={float(K0_HAT_R2E):.3e}, "
          f"R4e={float(K0_HAT_R4E):.3e}")

    rung_log: list[dict] = []

    def _rung_callback(scale, ok, ctx, rung_diag):
        # Decorate per-rung diagnostics with observables for the JSON
        # report (the orchestrator already captures snes_converged
        # and cd_observable; we add R_2e and R_4e for inspection).
        try:
            f_R2e = _build_bv_observable_form(
                ctx, mode="reaction", reaction_index=0, scale=-I_SCALE
            )
            rung_diag["R_2e_mA_cm2"] = float(fd.assemble(f_R2e))
        except Exception as exc:
            rung_diag["R_2e_error"] = f"{type(exc).__name__}: {exc}"
        try:
            f_R4e = _build_bv_observable_form(
                ctx, mode="reaction", reaction_index=1, scale=-I_SCALE
            )
            rung_diag["R_4e_mA_cm2"] = float(fd.assemble(f_R4e))
        except Exception as exc:
            rung_diag["R_4e_error"] = f"{type(exc).__name__}: {exc}"
        rung_log.append({**rung_diag})
        print(
            f"  [rung] scale={scale:.3e}  ok={ok}  "
            f"R_2e={rung_diag.get('R_2e_mA_cm2')}  "
            f"R_4e={rung_diag.get('R_4e_mA_cm2')}",
            flush=True,
        )

    t0 = time.time()
    converged = False
    ladder_history: list[tuple[float, str]] = []
    ladder_exhausted = False
    err_msg = None
    cd_final = None
    R_2e_final = None
    R_4e_final = None
    ic_fallback = False
    ic_fallback_reason = ""

    try:
        result = solve_anchor_with_continuation(
            sp_at_v,
            mesh=mesh,
            k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
            initial_scales=INITIAL_SCALES,
            max_inserts_per_step=MAX_INSERTS_PER_STEP,
            ic_at_target=IC_AT_TARGET,
            rung_callback=_rung_callback,
        )
        converged = bool(result.converged)
        ladder_history = list(result.ladder_history)
        ic_fallback = bool(result.ctx.get("initializer_fallback", False))
        ic_fallback_reason = str(
            result.ctx.get("initializer_fallback_reason", "")
        )
        if converged:
            try:
                f_cd = _build_bv_observable_form(
                    result.ctx, mode="current_density",
                    reaction_index=None, scale=-I_SCALE,
                )
                cd_final = float(fd.assemble(f_cd))
                f_R2e = _build_bv_observable_form(
                    result.ctx, mode="reaction", reaction_index=0,
                    scale=-I_SCALE,
                )
                R_2e_final = float(fd.assemble(f_R2e))
                f_R4e = _build_bv_observable_form(
                    result.ctx, mode="reaction", reaction_index=1,
                    scale=-I_SCALE,
                )
                R_4e_final = float(fd.assemble(f_R4e))
            except Exception as exc:
                err_msg = f"final observable error: {type(exc).__name__}: {exc}"
    except LadderExhausted as exc:
        ladder_exhausted = True
        err_msg = f"LadderExhausted: {exc}"
        print(f"  LadderExhausted: {exc}", flush=True)

    elapsed = time.time() - t0

    print()
    print(f"  converged = {converged}")
    print(f"  ladder_exhausted = {ladder_exhausted}")
    print(f"  ladder_history = {ladder_history!r}")
    if ic_fallback:
        print(f"  IC fallback: {ic_fallback_reason!r}")
    if cd_final is not None:
        print(f"  cd (final) = {cd_final:+.4e} mA/cm²")
        print(f"  R_2e (final) = {R_2e_final:+.4e} mA/cm²")
        print(f"  R_4e (final) = {R_4e_final:+.4e} mA/cm²")
    print(f"  wall = {elapsed:.1f}s")
    if err_msg:
        print(f"  err: {err_msg}")

    out_path = OUT_DIR / "anchor_smoke_continuation.json"
    with open(out_path, "w") as f:
        json.dump({
            "anchor_v_rhe": float(ANCHOR_V_RHE),
            "mesh_Ny": int(MESH_NY),
            "exponent_clip": float(EXPONENT_CLIP),
            "initial_scales": list(INITIAL_SCALES),
            "max_inserts_per_step": int(MAX_INSERTS_PER_STEP),
            "ic_at_target": bool(IC_AT_TARGET),
            "k0_targets": {
                "0": float(K0_HAT_R2E),
                "1": float(K0_HAT_R4E),
            },
            "converged": bool(converged),
            "ladder_exhausted": bool(ladder_exhausted),
            "ladder_history": [
                [float(s), str(o)] for s, o in ladder_history
            ],
            "rungs": rung_log,
            "ic_fallback": bool(ic_fallback),
            "ic_fallback_reason": ic_fallback_reason,
            "cd_mA_cm2": cd_final,
            "R_2e_mA_cm2": R_2e_final,
            "R_4e_mA_cm2": R_4e_final,
            "wall_seconds": float(elapsed),
            "error": err_msg,
        }, f, indent=2, default=str)
    print(f"\n  output -> {out_path}")

    # Pass requires (a) converged AND (b) last ladder entry is (1.0, "ok").
    if not converged:
        return 1
    if not ladder_history:
        return 1
    last_scale, last_outcome = ladder_history[-1]
    if not (float(last_scale) == 1.0 and last_outcome == "ok"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
