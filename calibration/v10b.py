"""Phase 6 beta v10b literature calibration of Gamma_max + k_des + C_S.

Locked numeric constants plus the V10B_CALIBRATION_METADATA record
that every v10b result JSON emits verbatim.  This module is
deliberately Firedrake-free; importing it must not pull
``firedrake`` (or any ``Forward.bv_solver`` submodule) into
``sys.modules``.  Pure Python only.

Source of truth: ``.claude/plans/phase6b-step8-v10b-calibration.md``
(v7-FINAL, APPROVED by GPT critique session 36).  Companion writeup:
``docs/phase6/v10b_calibration_summary.md``.

Outcome summary (2026-05-10):

* ``C_S`` -- locked at step 7 to ``0.20 F/m^2`` per
  ``docs/phase6/CMK3_capacitance_literature.md`` (Bohra-Koper-Choi
  consensus).  v10b carries the 4-rung sensitivity bracket {0.05, 0.10,
  0.20, 0.30} F/m^2 and three open asks (Bohra 2019 EES pull, Risk #5
  sigma_S re-derivation with Stern-only C_S = 20 uF/cm^2, Yash
  convention disposition).
* ``Gamma_max_nondim`` -- 4-test compatibility check (mechanism /
  electrode / electrolyte / dimensional) finds no peer-reviewed source
  that reports MOH adsorbate coverage at the OHP in K2SO4 / sp2-carbon
  / aqueous.  Singh 2016 reports the equilibrium constant K_eq for
  hydrolysis (constraining k_hyd / k_prot, not coverage);
  Iamprasertkun 2019 reports HOPG specific capacitance (4.7 - 9.4
  uF/cm^2 across Li+ -> Cs+) which is a different quantity from MOH
  coverage; Bohra 2019 uses variable Booth permittivity which does not
  yield MOH coverage.  Decision rule outcome: tighten V10A derivation
  chain (V10B = V10A = 0.047 nondim).  The V10A chain is the one-
  monolayer hard-sphere packing at the OHP with K+ effective hard-
  sphere radius r ~ 2.3 Angstrom, giving Gamma_max_phys ~ 5.6e-6
  mol/m^2 and Gamma_max_nondim = 5.6e-6 / (C_SCALE * L_REF)
  = 5.6e-6 / (1.2 * 1e-4) ~= 0.047.  Sensitivity bracket: 3-rung
  Gamma_max in {V10A/2, V10A, V10A*2} as the three Gamma_max rungs
  of the D7-D4 coupled matrix.  Engineering choice flag NOT set --
  this is a documented derivation, not an undocumented engineering
  pick.  See writeup section 2 for the full 4-test outcome and the
  K+ hydrated-radius caveats (Marcus / Volkov bond-distance literature
  reports K-O 2.65 - 2.97 Angstrom; r = 2.3 Angstrom is a closer-
  packing effective hard-sphere radius for monolayer estimation,
  intentionally smaller than the bare K-O bond distance so the
  monolayer area is not over-counted).
* ``k_des_nondim`` -- engineering choice with documented Eyring prior.
  Strategy 1 (analogous OH* desorption from sp2 carbon, Nørskov-
  Viswanathan 2012 + Co-Billy 2017) does not yield a transportable
  order of magnitude with electrode / electrolyte transferability:
  the published OH* binding energies are for metal cathodes in
  CO2R / HER conditions, not MOH adsorbate at the OHP in K2SO4 / ORR.
  Strategy 2 (Eyring estimate from cation-OH bond energy) yields the
  bracket prior k_des_nondim in [1e-2, 1e2] corresponding to
  Delta G_des in [0.69, 0.94] eV at 298 K (k_des_phys = (k_B T / h)
  exp(-Delta G_des / RT) and k_des_nondim = k_des_phys * tau_REF
  with tau_REF = L_REF^2 / D_REF ~= 5 s).  Strategy 3 -- engineering
  choice -- is the honest fallback: lock central value at k_des_nondim
  = 1.0 (smoke value; smoke 1.0 nondim corresponds to Delta G_des
  ~= 0.80 eV via Eyring at 298 K) and run D7-D3 + D7-D4 sensitivity
  brackets at {0.01, 0.1, 1.0, 10.0, 100.0}.  Engineering choice
  flag IS set on this parameter; future scope expansion (post-v10b)
  could use this as initial prior for a Phase D-style fit.

Hard invariants (do NOT touch in v10b -- see plan section 2 table):

* V_kin = -0.10 V, K0_R4e_factor = 1e-14, k_hyd_baseline = 1e-3
  nondim, k_hyd_route = 1e-1 nondim, WARM_WALK_GRID
  (+0.55, +0.40, +0.20, +0.10, -0.10), LAMBDA_LADDER (0.0, 0.25,
  0.50, 0.75, 1.0), R2e E_0 = 0.695 V, R4e E_0 = 1.23 V,
  exponent_clip = 100.0, STERN_F_M2_ANCHOR = 0.10 F/m^2,
  STERN_F_M2_BASELINE = 0.20 F/m^2, c_s_ladder + kw_eff_ladder combo
  unsupported, tau_REF = L_REF^2 / D_REF ~= 5 s.

References
----------
.. plan v7-FINAL: /Users/jakeweinstein/.claude/plans/phase6b-step8-v10b-calibration.md
.. CMK-3 writeup: docs/phase6/CMK3_capacitance_literature.md
.. v10b writeup: docs/phase6/v10b_calibration_summary.md
.. CLAUDE.md hard rules #2, #6
"""
from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Locked v10b numeric constants
# ---------------------------------------------------------------------------

# Stern compact-layer capacitance in F/m^2.  Per
# docs/phase6/CMK3_capacitance_literature.md (Bohra-Koper-Choi consensus):
# L_S = 5 Angstrom, eps_S = 11.3, C_S = eps_S * eps_0 / L_S = 20 uF/cm^2
# = 0.20 F/m^2.  Pillai 2024 ``safe band'' is 10-50 uF/cm^2.  This is
# the *production* target; the residual reads it directly in SI units
# (no nondim rescaling).
C_S_F_M2_V10B: float = 0.20

# Surface-coverage cap at the OHP, nondim.  V10B = V10A.  Derivation
# chain: one monolayer of hydrated MOH at the OHP, hard-sphere areal
# coverage with effective r ~ 2.3 Angstrom (closer-packing than the
# K-O bond distance to avoid over-counting monolayer area), giving
# Gamma_max_phys ~ 1 / (pi * (2.3e-10 m)^2 * N_A) ~= 5.6e-6 mol/m^2 and
# Gamma_max_nondim = Gamma_max_phys / (C_SCALE * L_REF)
# = 5.6e-6 / (1.2 * 1e-4) ~= 0.047.
GAMMA_MAX_HAT_V10B: float = 0.047

# MOH desorption-rate constant at the OHP, nondim.  Engineering choice
# per plan section 3.3 strategy 3.  Smoke central value preserved
# from V10A (no anchor in V10A); v10b documents the Eyring prior
# Delta G_des in [0.69, 0.94] eV mapping to k_des_nondim in
# [1e-2, 1e2] (5-rung bracket).  D7-D3 + D7-D4 sweeps close this
# evidentially; future scope expansion could use this as initial
# prior for a Phase D-style fit.
K_DES_NONDIM_V10B: float = 1.0


# ---------------------------------------------------------------------------
# Calibration metadata block -- emitted verbatim in every v10b result
# JSON.  Schema documented in plan section D1.
# ---------------------------------------------------------------------------


def _gamma_max_metadata() -> Dict[str, Any]:
    return {
        "value": GAMMA_MAX_HAT_V10B,
        "units": "nondim",
        "is_nondim": True,
        "source_type": "literature_chain",
        "engineering_choice": False,
        "citation": (
            "V10A derivation chain tightened in v10b: one-monolayer "
            "hard-sphere packing at the OHP, K+ effective hard-sphere "
            "radius r = 2.3 Angstrom (Marcus / Volkov K-O bond "
            "distance 2.65 - 2.97 Angstrom is an upper bound for the "
            "hydration shell, not the monolayer packing radius); "
            "Gamma_max_phys = 1 / (pi * r^2 * N_A) ~= 5.6e-6 mol/m^2; "
            "Gamma_max_nondim = Gamma_max_phys / (C_SCALE * L_REF) "
            "= 5.6e-6 / (1.2 * 1e-4) ~= 0.047.  "
            "Singh 2016 (10.1021/jacs.6b07612) reports K_eq for "
            "M(H2O)n+ <=> M(OH)0 + H+, not adsorbate coverage; "
            "Iamprasertkun 2019 JPCL (10.1021/acs.jpclett.8b03523) "
            "reports HOPG specific capacitance (4.7 - 9.4 uF/cm^2), "
            "a different quantity from MOH coverage; Bohra 2019 EES "
            "(10.1039/c9ee02485a) uses variable Booth permittivity "
            "and does not report MOH coverage at the OHP.  See "
            "docs/phase6/v10b_calibration_summary.md section 2 for "
            "the 4-test compatibility audit."
        ),
        "bracket": [
            GAMMA_MAX_HAT_V10B / 2.0,
            GAMMA_MAX_HAT_V10B,
            GAMMA_MAX_HAT_V10B * 2.0,
        ],
        "prior": None,
        "compatibility": {
            "mechanism": (
                "MOH adsorbate coverage at the OHP, hard-sphere "
                "monolayer packing derivation"
            ),
            "electrode": (
                "sp2-carbon (CMK-3); derivation is electrode-agnostic "
                "in the limit of monolayer packing at the OHP -- the "
                "underlying counterion-OHP geometry sets r, not the "
                "carbon DOS"
            ),
            "electrolyte": "aqueous K+ (deck baseline; K2SO4 pH 4-6)",
            "dimensional": (
                "Gamma_max_nondim = Gamma_max_phys / (C_SCALE * L_REF) "
                "= 5.6e-6 mol/m^2 / (1.2 mol/m^3 * 1e-4 m) = 0.047"
            ),
        },
    }


def _k_des_metadata() -> Dict[str, Any]:
    return {
        "value": K_DES_NONDIM_V10B,
        "units": "nondim",
        "is_nondim": True,
        "source_type": "engineering",
        "engineering_choice": True,
        "citation": None,
        "bracket": [0.01, 0.1, 1.0, 10.0, 100.0],
        "prior": (
            "Eyring at 298 K: k_des_phys = (k_B T / h) * "
            "exp(-Delta G_des / RT), k_des_nondim = k_des_phys * "
            "tau_REF with tau_REF = L_REF^2 / D_REF ~= 5 s.  Bracket "
            "k_des_nondim in [1e-2, 1e2] maps to Delta G_des in "
            "[0.69, 0.94] eV; central value 1.0 nondim corresponds to "
            "Delta G_des ~= 0.80 eV.  Strategies 1 and 2 were attempted "
            "(analogous OH* desorption from sp2-carbon, Norskov-"
            "Viswanathan 2012 + Co-Billy 2017; Eyring estimate from "
            "cation-OH bond energy) and rejected because no source "
            "transports the order of magnitude to the v10b mechanism "
            "(MOH adsorbate at the OHP in K2SO4 / ORR) with documented "
            "electrode / electrolyte compatibility.  Engineering "
            "choice fallback is per plan section 3.3 strategy 3; "
            "D7-D3 (10 rungs) + D7-D4 (30 rungs) sensitivity sweeps "
            "close this parameter evidentially within v10b scope."
        ),
        "compatibility": {
            "mechanism": (
                "MOH desorption from the OHP (M(OH)0 -> M(OH)0(aq)); "
                "engineering choice, not a literature anchor"
            ),
            "electrode": "sp2-carbon (CMK-3) -- engineering prior",
            "electrolyte": "aqueous K+ -- engineering prior",
            "dimensional": (
                "k_des_nondim = k_des_phys * tau_REF; tau_REF = "
                "L_REF^2 / D_REF ~= 5 s; smoke 1.0 nondim ~ 0.20/s "
                "physical desorption rate ~ Delta G_des = 0.80 eV via "
                "Eyring at 298 K"
            ),
        },
    }


def _c_s_metadata() -> Dict[str, Any]:
    return {
        "value": C_S_F_M2_V10B,
        "units": "F/m^2",
        "is_nondim": False,
        "source_type": "literature",
        "engineering_choice": False,
        "citation": (
            "Bohra-Koper-Choi PNP-modeling consensus: Bohra et al. 2024 "
            "JPC C (PMC11215773); Choi et al. 2024 JPC C "
            "(10.1021/acs.jpcc.4c03469) -- explicit L_S = 5 Angstrom, "
            "eps_S = 11.3 derivation; Pillai et al. 2024 JPC C "
            "(10.1021/acs.jpcc.3c05364) methodological critique of "
            "the 100 - 200 uF/cm^2 CO2R-modeler convention; CatINT "
            "(Ringe/Bell, Stanford) default config; Kilic-Bazant 2007 "
            "Phys Rev E 75:021503 foundational mPNP-Stern.  See "
            "docs/phase6/CMK3_capacitance_literature.md for the full "
            "citation chain and caveats (per-local-surface-element "
            "interpretation, Singh 51 uF/cm^2 Cu vs Stern-only, "
            "carbon-specific narrowing, constant-C_S as field-averaged "
            "approximation)."
        ),
        "bracket": [0.05, 0.10, 0.20, 0.30],
        "prior": None,
        "compatibility": {
            "mechanism": (
                "Stern compact-layer capacitance entering the PNP-Stern "
                "boundary condition (residual)"
            ),
            "electrode": (
                "sp2-carbon (CMK-3); per-local-surface-element "
                "interpretation -- roughness factor (RF ~ 6000 for "
                "the deck) is implicit in fitted k_0, not in the BC"
            ),
            "electrolyte": "aqueous K2SO4 pH 4 - 6 (deck baseline)",
            "dimensional": (
                "C_S = eps_S * eps_0 / L_S = 11.3 * 8.854e-12 F/m / "
                "5e-10 m = 0.200 F/m^2 = 20 uF/cm^2"
            ),
        },
    }


V10B_CALIBRATION_METADATA: Dict[str, Dict[str, Any]] = {
    "gamma_max": _gamma_max_metadata(),
    "k_des": _k_des_metadata(),
    "C_S": _c_s_metadata(),
}


# Module-level kinetics convenience dict mirroring SMOKE_KINETICS_V10A
# in scripts/studies/phase6b_v10a_v_sweep_diagnostic.py.  The v-sweep
# driver imports V10B_KINETICS at module top level so factory defaults
# refer to V10B at import time.
V10B_KINETICS: Dict[str, float] = {
    "k_hyd_nondim":     1e-3,
    "k_prot_nondim":    1e-3,
    "k_des_nondim":     K_DES_NONDIM_V10B,
    "delta_ohp_hat":    4e-6,                  # 0.40 nm / 100 um, locked
    "gamma_max_nondim": GAMMA_MAX_HAT_V10B,
    "r_H_El_pm":        200.98,                # Singh Cu prior, locked
}


__all__ = [
    "C_S_F_M2_V10B",
    "GAMMA_MAX_HAT_V10B",
    "K_DES_NONDIM_V10B",
    "V10B_CALIBRATION_METADATA",
    "V10B_KINETICS",
]
