"""Phase 1.2 audit: picard_outer_loop_general at V_RHE=+0.55 V (parallel 2e/4e).

Calls the generalized Picard outer loop directly (no Firedrake) at the
weakest cathodic drive within the page-15 grid, using the production
3sp + ClO4 single-counterion + parallel-2e/4e configuration.  Prints
per-iter verbose state and final converged tuple.

Three sub-runs:
  (A) pure-2e:   k0_R4e disabled (=0)
  (B) pure-4e:   k0_R2e disabled (=0)
  (C) mixed:     both k0 active

Output: StudyResults/fast_realignment_2026-05-08/picard_audit/{A,B,C}.log
"""
from __future__ import annotations

import json
import math
import os
import sys
from contextlib import redirect_stdout

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Forward.bv_solver.picard_ic import picard_outer_loop_general

from scripts._bv_common import (
    V_T,
    C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT, C_CLO4_HAT,
    D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
    K0_HAT_R2E, K0_HAT_R4E,
    ALPHA_R2E, ALPHA_R4E,
    E_EQ_R2E_V, E_EQ_R4E_V,
    A_DEFAULT,
)


V_RHE_ANCHOR = +0.55     # plan §1.2: weakest cathodic drive within page-15
EXPONENT_CLIP = 100.0
PHI_APPLIED_MODEL = V_RHE_ANCHOR / V_T   # nondim


def make_reactions(*, k0_R2e: float, k0_R4e: float):
    return [
        {
            "k0_model": float(k0_R2e),
            "alpha": float(ALPHA_R2E),
            "n_electrons": 2,
            "E_eq_model": float(E_EQ_R2E_V) / V_T,
            "cathodic_species": 0,
            "anodic_species": 1,
            "stoichiometry": [-1, +1, -2],
            "reversible": True,
            "c_ref_model": 1.0,                    # nondim H2O2 reference
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {
            "k0_model": float(k0_R4e),
            "alpha": float(ALPHA_R4E),
            "n_electrons": 4,
            "E_eq_model": float(E_EQ_R4E_V) / V_T,
            "cathodic_species": 0,
            "anodic_species": None,
            "stoichiometry": [-1, 0, -4],
            "reversible": False,
            "c_ref_model": 0.0,
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]


def run_pass(label: str, k0_R2e: float, k0_R4e: float, *, out_dir: str):
    out_path = os.path.join(out_dir, f"{label}.log")
    print(f"\n=== Picard audit pass: {label} ===")
    print(f"   k0_R2e_model={k0_R2e:.3e}  k0_R4e_model={k0_R4e:.3e}")
    print(f"   V_RHE={V_RHE_ANCHOR:+.3f} V  phi_applied_model={PHI_APPLIED_MODEL:+.3e}")
    print(f"   E_eq_R2e_model={E_EQ_R2E_V/V_T:+.3e}  E_eq_R4e_model={E_EQ_R4E_V/V_T:+.3e}")
    print(f"   eta_R2e_at_anchor={(V_RHE_ANCHOR - E_EQ_R2E_V)/V_T:+.3e} "
          f"eta_R4e_at_anchor={(V_RHE_ANCHOR - E_EQ_R4E_V)/V_T:+.3e}")
    print(f"   exponent_clip={EXPONENT_CLIP}  log -> {out_path}")
    rxns = make_reactions(k0_R2e=k0_R2e, k0_R4e=k0_R4e)
    bulk_concs = [float(C_O2_HAT), float(H2O2_SEED_NONDIM), float(C_HP_HAT)]
    diffusivities = [float(D_O2_HAT), float(D_H2O2_HAT), float(D_HP_HAT)]
    species_floors = [1e-300, 1e-30, 1e-300]
    a_h = 0.0      # no Bikerman in audit (single-ClO4 ideal counterion)
    a_cl = 0.0
    with open(out_path, "w") as f:
        with redirect_stdout(f):
            print(f"audit pass {label}", flush=True)
            print(f"  rxns = {json.dumps(rxns, indent=2)}", flush=True)
            print(f"  bulk_concs={bulk_concs}  diffusivities={diffusivities}",
                  flush=True)
            print(f"  V_RHE={V_RHE_ANCHOR}  phi_applied_model={PHI_APPLIED_MODEL}",
                  flush=True)
            print(f"  c_clo4_bulk={float(C_CLO4_HAT)}  a_h={a_h} a_cl={a_cl}",
                  flush=True)
            print("---- Picard iterations ----", flush=True)
            ok, reason, iters, state = picard_outer_loop_general(
                reactions=rxns,
                bulk_concs=bulk_concs,
                diffusivities=diffusivities,
                species_floors=species_floors,
                h_idx=2,
                c_clo4_bulk=float(C_CLO4_HAT),
                phi_applied_model=PHI_APPLIED_MODEL,
                bv_exp_scale=1.0,
                exponent_clip=EXPONENT_CLIP,
                clip_exponent=True,
                a_h=a_h, a_cl=a_cl,
                c_cl_anchor_kind="bulk",
                stern_split=None,                 # no Stern in audit
                omega=0.5,
                max_iters=50,
                tol=1e-6,
                topology_hint="general",
                verbose=True,
            )
            print("---- Final ----", flush=True)
            print(f"  ok={ok}  reason={reason!r}  iters={iters}", flush=True)
            print(f"  state={json.dumps({k: v for k, v in state.items() if not isinstance(v, list) or len(v) <= 8}, indent=2, default=float)}", flush=True)
    print(f"   ok={ok}  reason={reason!r}  iters={iters}")
    if ok:
        print(f"   converged: R={state.get('R_list')}  c_s={state.get('c_s_list')} "
              f"phi_o={state.get('phi_o'):+.3e}  psi_D={state.get('psi_D'):+.3e}")
    return {"label": label, "ok": ok, "reason": reason, "iters": iters,
            "state_summary": {
                "R_list": state.get("R_list"),
                "c_s_list": state.get("c_s_list"),
                "phi_o": state.get("phi_o"),
                "psi_D": state.get("psi_D"),
                "psi_S": state.get("psi_S"),
                "gamma_s": state.get("gamma_s"),
                "eta_list": state.get("eta_list"),
            }}


def main():
    out_dir = os.path.join(
        _ROOT, "StudyResults", "fast_realignment_2026-05-08", "picard_audit"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Picard audit (parallel-2e/4e) at V_RHE={V_RHE_ANCHOR:+.3f} V")
    print(f"  out_dir = {out_dir}")
    summaries = []
    summaries.append(run_pass("A_pure_2e", float(K0_HAT_R2E), 0.0, out_dir=out_dir))
    summaries.append(run_pass("B_pure_4e", 0.0, float(K0_HAT_R4E), out_dir=out_dir))
    summaries.append(run_pass("C_mixed",   float(K0_HAT_R2E), float(K0_HAT_R4E),
                              out_dir=out_dir))
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"V_RHE": V_RHE_ANCHOR, "passes": summaries}, f, indent=2,
                  default=float)
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
