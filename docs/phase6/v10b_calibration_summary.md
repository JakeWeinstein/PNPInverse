# Phase 6 beta v10b Literature Calibration of Gamma_max + k_des + C_S

**Status:** v10b shipped (2026-05-10).  Acceptance-bundle step 8 of
the "v10a -> E sequence"
(`PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`).

**Question.** What values should the project use for the three
parameters introduced by the Phase 6 beta v10a Langmuir cap plus
step 7 Stern-capacitance work?

* `Gamma_max_nondim` -- the v10a saturation cap on Gamma at the OHP.
* `k_des_nondim` -- the MOH desorption rate at the OHP (= `R_net` /
  Gamma at steady state by the closed-form Langmuir residual).
* `stern_capacitance_f_m2` (`C_S`) -- the Stern compact-layer
  capacitance entering the PNP-Stern BC.

**Recommendation (v10b production target):**

```
GAMMA_MAX_HAT_V10B   = 0.047 nondim   (= GAMMA_MAX_HAT_V10A_SMOKE; tightened V10A chain)
K_DES_NONDIM_V10B    = 1.0  nondim    (engineering choice; Eyring prior)
C_S_F_M2_V10B        = 0.20 F/m^2     (locked at step 7)
```

Source of truth: `calibration/v10b.py` (Firedrake-free top-level
package).  Plan: `.claude/plans/phase6b-step8-v10b-calibration.md`
(v7-FINAL).  Critique trail:
`docs/handoffs/CHATGPT_HANDOFF_36_phase6b-v10b-calibration/` (R1-R7 +
FINAL_REVISION, 53 accepted issues across 7 rounds).

---

## 1. Citation chain per parameter

### 1.1 `C_S = 0.20 F/m^2`

Already published in
`docs/phase6/CMK3_capacitance_literature.md` (step 7, 2026-05-10).
**Bohra-Koper-Choi consensus**, derived from `L_S = 5 Angstrom` and
`eps_S = 11.3` (Booth-saturated):

```
C_S = eps_S * eps_0 / L_S = 11.3 * 8.854e-12 F/m / 5e-10 m
    = 0.200 F/m^2 = 20 uF/cm^2
```

Independent sources converging on this value:

| Source | DOI / Link | Contribution |
|---|---|---|
| Bohra et al. 2024 *JPC C* | [PMC11215773](https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/) | "Experimentally found values of C_Stern are often reported in the range of 20 to 25 uF cm^-2." |
| Choi et al. 2024 *JPC C* | [10.1021/acs.jpcc.4c03469](https://pubs.acs.org/doi/10.1021/acs.jpcc.4c03469) | Explicit derivation: `L_S = 5 A`, `eps_S = 11.3` => `C_S = 20 uF/cm^2`. |
| Pillai et al. 2024 *JPC C* | [10.1021/acs.jpcc.3c05364](https://pubs.acs.org/doi/10.1021/acs.jpcc.3c05364) | Methodological critique; identifies "safe band" 10-50 uF/cm^2 and flags the 100-200 uF/cm^2 CO2R-modeler convention as wrong. |
| CatINT (Ringe/Bell, Stanford) | [github.com/sringe/CatINT](https://github.com/sringe/CatINT/blob/master/docs/source/tutorials/co2r_au_catmap/catint_input.rst) | Default config: `Stern capacitance: 20. # micro F/cm2`. |
| Kilic, Bazant, Ajdari 2007 *Phys Rev E* | [arXiv physics/0611030](https://arxiv.org/pdf/physics/0611030) | Foundational mPNP-Stern; phenomenological band 10-50 uF/cm^2. |

Full step 7 trail in `.research/cmk3-stern-capacitance/SUMMARY.md`.

### 1.2 `Gamma_max_nondim = 0.047`

**v10b decision rule outcome:** tighten V10A derivation chain
(V10B = V10A; no value change; document the derivation more
carefully than v10a did).

The 4-test compatibility check (mechanism / electrode / electrolyte /
dimensional) was applied to the candidate sources from plan section
3.2:

| Source | Reports | 4-test outcome |
|---|---|---|
| Singh et al. 2016 *JACS* [10.1021/jacs.6b07612](https://pubs.acs.org/doi/10.1021/jacs.6b07612) | Field-dependent pKa + `K_eq` for `M(H2O)n+ <=> M(OH)0 + H+`.  Table S1 reports partial-coverage estimates derived from spectroscopic IR.  But the partial-coverage is for water at the metal Stern layer (Cu / Ag), NOT MOH adsorbate coverage at the OHP. | FAIL test 1 (mechanism): not MOH coverage.  Also FAIL test 2 (electrode): polycrystalline Cu / Ag, not sp2-carbon. |
| Iamprasertkun 2019 *JPCL* [10.1021/acs.jpclett.8b03523](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.8b03523) | HOPG basal-plane specific capacitance 4.7-9.4 uF/cm^2 across Li+ -> Cs+ at fixed potential vs PZC. | FAIL test 1: specific capacitance is a different quantity from MOH adsorbate coverage.  PASS test 2 (electrode: sp2 carbon). |
| Bohra 2019 EES [10.1039/c9ee02485a](https://doi.org/10.1039/c9ee02485a) | Variable Booth permittivity in GMPNP; no explicit MOH adsorbate coverage at the OHP. | FAIL test 1: not MOH coverage.  Paper not pulled into `Articles/`; open ask per `docs/phase6/CMK3_capacitance_literature.md` section 6. |
| Co-Zhang 2019 *Angewandte* | Local-pH change product / ring spectroscopy on Au; no MOH coverage at the OHP. | FAIL test 1: not MOH coverage. |
| Yash modeling code (`data/.../Yash-Trends/`) | PNP+BV cross-validation reference using parallel 6-species (K+/SO4(2-) explicit) with `L_Stern` thickness, not Gamma_max cap.  No saturation cap. | FAIL test 1: not Gamma_max. |
| Parameters_Seitz_Mangan.xlsx | Group convention spreadsheet; no Gamma_max entry. | FAIL test 1: no entry. |

**Conclusion:** no peer-reviewed source reports MOH adsorbate coverage
at the OHP in K2SO4 / sp2-carbon / aqueous K+ that passes all four
tests.  Per plan section 3.2 decision rule, the outcome is
"tighten V10A chain (V10B = V10A)" with a documented derivation.

V10B derivation (= V10A): one monolayer of hydrated MOH at the OHP,
hard-sphere areal coverage:

```
Gamma_max_phys ~ 1 / (pi * r^2 * N_A)
```

with `r = 2.3 Angstrom`, the effective hard-sphere radius for K+
monolayer packing at the OHP.  Note that the K-O bond distance from
Marcus / Volkov literature is 2.65 - 2.97 Angstrom (see
[Persson et al. 2012 *Inorg Chem* 51](https://pubs.acs.org/doi/10.1021/ic2018693)
and related solvation-shell references); `r = 2.3 A` is a closer
packing radius for monolayer-area estimation (intentionally smaller
than bond distance to avoid over-counting monolayer area).

```
Gamma_max_phys ~ 5.6e-6 mol/m^2
Gamma_max_nondim = Gamma_max_phys / (C_SCALE * L_REF)
                  = 5.6e-6 / (1.2 * 1e-4)
                  ~= 0.047
```

### 1.3 `k_des_nondim = 1.0`

**v10b decision rule outcome:** engineering choice with documented
Eyring prior `k_des_nondim in [1e-2, 1e2]`.

Strategy 1 (analogous reactions: OH* desorption from sp2-carbon).
Targets reviewed:

* Nørskov-Viswanathan 2012 *JPCL* (in
  `data/.../Articles/2012-Norskov-Viswanathan-Unifying the 2e and 4e
  Reduction of Oxygen on Metal Surfaces-JPCL.pdf`) -- OH* on metal
  cathodes, not sp2-carbon.  No transportable order of magnitude for
  the OHP MOH desorption rate.
* Co-Billy 2017 *ACS Catal* (in
  `data/.../Articles/2017-Co-Billy-Experimental Parameters Influencing
  Hydrocarbon Selectivity during the Electrochemical Conversion of
  CO2-ACS Catal.pdf`) -- CO2R / Cu kinetics, not MOH at sp2-carbon.

Outcome: strategy 1 yields no transportable order of magnitude
with documented electrode / electrolyte compatibility.

Strategy 2 (Eyring estimate from cation-OH bond energy).  The
cation-OH binding energy literature for K-OH and similar alkali
hydroxides bracket the desorption barrier in a wide range
(0.5-1.5 eV depending on solvation context).  Eyring at 298 K:

```
k_des_phys = (k_B T / h) * exp(-Delta G_des / RT)
```

with `k_B T = 0.0257 eV` and `k_B T / h ~= 6.21e12 / s` gives the
mapping:

| Delta G_des | k_phys [1/s] | k_des_nondim |
|---|---|---|
| 0.7 eV  | 9.2     | 46    |
| 0.8 eV  | 0.19    | 0.93  |
| 0.9 eV  | 0.0040  | 0.020 |
| 1.0 eV  | 8.0e-5  | 4.0e-4 |

with `tau_REF = L_REF^2 / D_REF ~= 5 s`.  Outcome: strategy 2 cannot
bracket Delta G_des within +/- 0.06 eV (the rough criterion for "lock
with bracket").

Strategy 3 -- engineering-choice fallback -- is the honest fallback
per plan section 3.3:

```
k_des_nondim in [1e-2, 1e2]   <=>   Delta G_des in [0.69, 0.94] eV
                                    at 298 K
```

Central value `K_DES_NONDIM_V10B = 1.0` corresponds to
`Delta G_des ~= 0.80 eV`.  The D7-D3 (10 rungs) and D7-D4 (30 rungs)
sensitivity sweeps close this parameter evidentially within v10b
scope.  Future scope expansion (post-v10b) could use this calibration
as the initial prior for a Phase D-style fit; Phase D's locked scope
is K-only Delta_beta and does NOT include `k_des` (see
`docs/phase6/phase6b_next_steps_plan.md`).

### 1.4 Nondim mapping summary

`C_S` is dimensional [F/m^2]; the residual reads it directly.

`Gamma_max_hat = Gamma_max_phys / (C_SCALE * L_REF)
              = Gamma_max_phys / (1.2e-4 mol/m^2)`

`k_des_nondim = k_des_phys * tau_REF`, `tau_REF = L_REF^2 / D_REF
              ~= 5 s` (per `scripts/_bv_common.py:131-134`).

---

## 2. Per-parameter regimes / decision rules / engineering-choice flags

| Parameter | Source type | Engineering choice? | Bracket |
|---|---|---|---|
| `C_S` | literature | False | {0.05, 0.10, 0.20, 0.30} F/m^2 |
| `Gamma_max` | literature_chain (V10A tightened) | False | {V10B/2, V10B, V10B*2} nondim |
| `k_des` | engineering | True | {0.01, 0.1, 1.0, 10.0, 100.0} nondim |

Decision rules per plan section 9:

* `C_S`: locked at 0.20 F/m^2; D7-D1 bracket + carry open asks.
* `Gamma_max`: 4-test compatibility check fails for all sources ->
  tighten V10A chain (V10B = V10A) -> document derivation.
* `k_des`: strategy 1 + 2 fail -> engineering choice with
  Eyring prior; D7-D3 + D7-D4 close evidentially.

---

## 3. Caveats per parameter

### 3.1 `C_S = 0.20 F/m^2`

Caveats inherited from step 7 (full discussion in
`docs/phase6/CMK3_capacitance_literature.md` section 3):

* `C_S` is **per-local-surface-element**, not per-geometric-area.
  CMK-3 roughness factor (RF ~= 6000) is implicit in fitted `k_0`,
  never explicit in the Stern BC.
* Singh's 51 uF/cm^2 Cu is total `C_dl`, not Stern-only.  The
  earlier "sigma_S = 226 uC/cm^2 / C_S = 0.10 F/m^2 implies
  unphysical Delta phi_Stern" Risk #5 concern is being re-derived
  with Stern-only `C_S = 20 uF/cm^2` -- see Open Asks section 6.
* Carbon-specific narrowing pulls slightly below 20 uF/cm^2 for
  sp2 carbon EDLC measurements; the 1D-slab Stern-only `C_S`
  converges on 0.10-0.20 F/m^2 (sp2-carbon Stern is at the low
  edge of the metal-electrode consensus).
* Constant `C_S` is field-averaged.  Variable-`eps_S` / Booth-
  equation refinement is out of scope for v10b; could land
  post-v10b if the sensitivity sweep shows constant-`C_S` is a
  meaningful systematic.

### 3.2 `Gamma_max_nondim = 0.047`

* The V10A derivation assumes hard-sphere packing with K+ effective
  radius `r = 2.3 A`.  Marcus / Volkov experimental K-O bond
  distances are 2.65 - 2.97 A (Persson 2012 *Inorg Chem*), giving
  `Gamma_max_nondim ~ 0.029-0.036` if `r` were taken at the bond
  distance.  `r = 2.3 A` is a closer-packing effective radius, not a
  measured bond distance -- the smaller value avoids over-counting
  monolayer area.  Sensitivity bracket {V10B/2, V10B, V10B*2}
  spans this uncertainty.
* The MOH adsorbate at the OHP is a **modeled species** -- there is
  no direct experimental measurement of MOH coverage at the OHP in
  the deck's K2SO4 / sp2-carbon / ORR conditions.  The model treats
  MOH as a Langmuir-cap-saturated reservoir at the OHP per the
  Singh 2016 hydrolysis mechanism, but the value of `Gamma_max`
  itself is a derivation, not a measurement.
* If a future literature pull resolves the open ask (Bohra 2019 EES,
  or a direct MOH XPS / IR coverage measurement), the V10A chain
  could shift.  In the meantime the documented derivation is the
  reference.

### 3.3 `k_des_nondim = 1.0`

* Engineering choice -- not literature-anchored.  The Eyring prior
  spans 4 decades in `k_des_nondim` and 0.25 eV in `Delta G_des`;
  bracket sweeps D7-D3 + D7-D4 close this evidentially within v10b
  scope.
* The Singh 2016 K_eq for `M(H2O)n+ <=> M(OH)0 + H+` constrains
  `k_hyd / k_prot` (after deriving the full dimensional identity
  carrying `c_H_avg`, `delta_OHP_hat`, the Gamma-normalization, and
  `tau_REF`), NOT `k_des`.  A bare ratio `K_eq = k_hyd / k_prot` is
  shorthand; the v10b `k_des` parameter is the desorption-from-OHP
  rate, mechanistically distinct from `k_prot` (the reverse
  hydrolysis step).  An audit of the `k_hyd / k_prot` consistency
  with Singh K_eq is a separate task carried as an open ask in
  section 6.
* A2 diagnostic risk flag (handoff section 7): K+ Boltzmann pile-up
  at V_kin (`c_K_boundary_avg ~ 291 * c_K_bulk`) means sentinel-scale
  R_inj perturbations cannot dent boundary c_K by 5%.  v10b uses
  `theta` AND `R_net` at cap-saturated `k_hyd_route = 1e-1` as the
  k_des diagnostic, NOT boundary-c_K perturbation.

---

## 4. Sensitivity brackets and bracket-sweep numeric results

### 4.1 D7-D1 -- `C_S` sensitivity bracket (4 rungs)

```
C_S in {0.05, 0.10, 0.20, 0.30} F/m^2
```

Run at V_kin = -0.10 V, lambda = 1.0, k_hyd_baseline = 1e-3 nondim.
Two-stage anchor pattern at every rung (build at
STERN_F_M2_ANCHOR = 0.10, runtime-bump to target).

Hard gates (failure -> escalate to v10c; 4/4 mandatory):

* `cd_mA_cm^2 < 0` (cathodic) at V_kin for each rung.
* No R_4e sign flip: `R_4e_current_nondim > 0` where
  `|R_4e_current_nondim| > 1e-6`.
* `R_net >= 0` at every rung (positive by construction).
* Per-rung analytic-vs-solver mass-balance: rel <= 5e-3 via
  `gamma_ss_langmuir` using per-rung state.

**Output:** `StudyResults/phase6b_v10b_cs_bracket/cs_bracket.{json,png}`
(see also "Implementation status" below for run state).

### 4.2 D7-D4 -- coupled `Gamma_max` x `k_des` matrix (30 rungs)

```
Gamma_max in {V10B/2, V10B, V10B*2}   (3 values)
k_des     in {0.01, 0.1, 1.0, 10.0, 100.0}   (5 values)
k_hyd     in {1e-3, 1e-1}              (2 values; baseline + route)
                                       = 30 rungs total
```

Two-stage anchor pattern at every rung.  Same HARD gates as D7-D1.

**Output:** `StudyResults/phase6b_v10b_gamma_kdes_matrix/matrix.{json,png}`.

### 4.3 D7-D3 -- `k_des` 1D bracket (conditional, NOT triggered)

Per plan section D7-D3, this 10-rung sweep is conditional on `k_des`
landing on engineering choice.  v10b's `k_des` IS an engineering
choice (section 1.3), but D7-D3 is subsumed by D7-D4 (30 rungs
already vary k_des in {0.01, 0.1, 1.0, 10.0, 100.0} across BOTH
k_hyd in {1e-3, 1e-1}).  Per plan note "D7-D2 is dropped because
D7-D4 is the single Gamma_max sweep", the same logic applies to
D7-D3 -- it is subsumed.  **D7-D3 not run as a separate driver.**

### 4.4 D7-D2 -- 1D `Gamma_max` bracket (DROPPED per plan section
D7-D4)

Per plan note: "D7-D2 is DROPPED.  D7-D4 is the single Gamma_max
sweep -- it varies Gamma_max across 3 points x 5 k_des x 2 k_hyd; a
1D Gamma_max sweep would be redundant."

---

## 5. Implementation status

### 5.1 Source-of-truth metadata block

```python
# calibration/v10b.py
V10B_CALIBRATION_METADATA = {
    "gamma_max": {
        "value": 0.047,
        "units": "nondim",
        "is_nondim": True,
        "source_type": "literature_chain",
        "engineering_choice": False,
        "citation": "(see section 1.2 -- tightened V10A chain)",
        "bracket": [0.0235, 0.047, 0.094],
        "prior": None,
        "compatibility": {
            "mechanism": "MOH adsorbate coverage at OHP, hard-sphere monolayer",
            "electrode": "sp2-carbon (CMK-3), electrode-agnostic in monolayer limit",
            "electrolyte": "aqueous K+ (deck baseline)",
            "dimensional": "0.047 nondim = 5.6e-6 mol/m^2 / (1.2 * 1e-4) mol/m^2",
        },
    },
    "k_des": {
        "value": 1.0,
        "units": "nondim",
        "is_nondim": True,
        "source_type": "engineering",
        "engineering_choice": True,
        "citation": None,
        "bracket": [0.01, 0.1, 1.0, 10.0, 100.0],
        "prior": "Eyring at 298 K; k_des_nondim in [1e-2, 1e2] <=> Delta G_des in [0.69, 0.94] eV",
        "compatibility": {
            "mechanism": "MOH desorption from OHP -- engineering choice",
            "electrode": "sp2-carbon (CMK-3) -- engineering prior",
            "electrolyte": "aqueous K+ -- engineering prior",
            "dimensional": "k_des_nondim = k_des_phys * tau_REF; tau_REF ~ 5 s",
        },
    },
    "C_S": {
        "value": 0.20,
        "units": "F/m^2",
        "is_nondim": False,
        "source_type": "literature",
        "engineering_choice": False,
        "citation": "Bohra-Koper-Choi consensus -- see CMK3_capacitance_literature.md",
        "bracket": [0.05, 0.10, 0.20, 0.30],
        "prior": None,
        "compatibility": {
            "mechanism": "Stern compact-layer capacitance in PNP-Stern BC",
            "electrode": "sp2-carbon (CMK-3); per-local-surface-element",
            "electrolyte": "aqueous K2SO4 pH 4-6 (deck baseline)",
            "dimensional": "C_S = eps_S * eps_0 / L_S = 11.3 * 8.854e-12 / 5e-10 = 0.20 F/m^2",
        },
    },
}
```

### 5.2 Code-touch summary

| File | Change |
|---|---|
| `calibration/__init__.py` | New package (empty) |
| `calibration/v10b.py` | New module with V10B constants + metadata + V10B_KINETICS.  Pure Python; Firedrake-free |
| `Forward/bv_solver/cation_hydrolysis.py` | Freeze `GAMMA_MAX_HAT_V10A_SMOKE = 0.047`; import `GAMMA_MAX_HAT_V10B`, `K_DES_NONDIM_V10B`, `V10B_CALIBRATION_METADATA` from `calibration.v10b`; factory defaults at lines 293/295 switch to V10B; ASCII deprecation comment for `GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10A_SMOKE` alias |
| `scripts/_bv_common.py` | Mirror freeze + import + alias pattern; `make_cation_hydrolysis_config` default switches to `GAMMA_MAX_HAT_V10B` |
| `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` | Rename `SMOKE_KINETICS` to `SMOKE_KINETICS_V10A`; add `V10B_KINETICS` import; factory signature defaults switch to `V10B_KINETICS`; `SMOKE_KINETICS = SMOKE_KINETICS_V10A` deprecation alias; result JSON emits `"v10b_kinetics"` |
| `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` | Imports switch to `V10B_KINETICS` + `V10B_CALIBRATION_METADATA`; `_convergence_audit` refactored to `hard_gates` / `soft_deltas` split (overall_pass driven by hard_gates only); `R_net` field added to per-rung augmentation; result JSON emits `v10b_kinetics` + `v10b_calibration_metadata` |
| `scripts/studies/phase6b_step6_plumbing_ablation.py` | New `--a2-baseline-json` CLI flag; `_baseline_reproduction_audit` keys include `R_net`; imports switch to V10B; result JSON emits v10b kinetics + metadata |
| `scripts/studies/phase6b_v10b_cs_bracket.py` | New driver (D7-D1).  Lazy imports for Firedrake + `gamma_ss_langmuir` + `Forward.bv_solver.*`.  Per-rung HARD gates emitting cd, R_4e sign, R_net, analytic-vs-solver Gamma mismatch |
| `scripts/studies/phase6b_v10b_gamma_kdes_matrix.py` | New driver (D7-D4).  Same lazy-import policy; 30-rung enumeration; same HARD-gate evaluator |
| `tests/test_phase6b_v10b_calibration.py` | 13 new fast tests (metadata schema, V10A/V10B coexistence, V10B numeric consistency, AST-aware production-driver import audit, convergence-audit HARD/SOFT split, step 6 `--a2-baseline-json`, R_net audit field) |
| `tests/test_phase6b_v10b_bracket_matrix.py` | 14 new fast tests (Firedrake-free module imports, CLI parse, target grid, output schema, sign-convention regressions) |
| `tests/test_phase6b_v10a_langmuir_cap.py` | Literal `0.047` refs at lines 90, 123, 132, 147 refactored to `GAMMA_MAX_HAT_V10A_SMOKE`; new deprecation-alias semantics test |

### 5.3 Hard invariants (do NOT touch in v10b)

Per plan section 2:

| Constant | Value | Source |
|---|---|---|
| `V_kin` | -0.10 V | step 4 |
| `K0_R4e_factor` | 1e-14 | step 4 |
| `k_hyd_baseline` | 1e-3 nondim | step 5 |
| `k_hyd_route` | 1e-1 nondim | A.2 |
| `WARM_WALK_GRID` | (+0.55, +0.40, +0.20, +0.10, -0.10) | A.2 + step 6 |
| `LAMBDA_LADDER` | (0.0, 0.25, 0.50, 0.75, 1.0) | v10a |
| `R2e E_0` | 0.695 V | Ruggiero 2022 |
| `R4e E_0` | 1.23 V | Ruggiero 2022 |
| `exponent_clip` | 100.0 | CLAUDE.md hard rule #2 |
| `STERN_F_M2_ANCHOR` | 0.10 F/m^2 | two-stage anchor |
| `STERN_F_M2_BASELINE` | 0.20 F/m^2 | step 7 |
| `tau_REF` | ~5 s | `_bv_common.py:131-134` |

Breaking any of these in v10b -> escalate to v10c.  All preserved.

---

## 6. Open asks (carry post-v10b)

* **Bohra 2019 EES pull** (`10.1039/c9ee02485a`) into `Articles/`.
  Cited by Ruggiero 2022 ref 71 and Linsey 2025 deck slide 13; the
  absence is a true open ask.  Code at
  [github.com/divyabohra/GMPNP](https://github.com/divyabohra/GMPNP)
  (FEniCS + Bikerman).  Carried from step 7.
* **Re-derive Risk #5 sigma_S mismatch** using Stern-only
  `C_S = 20 uF/cm^2`.  If the mismatch persists, treat the Singh
  `sigma_S = 226 uC/cm^2` target as non-transportable from
  polycrystalline Cu in CO2R to CMK-3 in K2SO4 / ORR.  Carried from
  step 7.
* **Yash convention cross-check**: `L_Stern = 0.6 nm` with
  `eps_S = 11.3` (Choi-consistent) => `C_S = 0.17 F/m^2`.  With
  `eps_S = 6` (Conway oriented-water) => `C_S = 0.088 F/m^2`.
  Decide which convention the cross-stack equivalence uses.
  Carried from step 7.
* **Field-dependent `eps_S` / variable-`C_S`** (Storey-Bazant 2012,
  Bohra 2019).  Out of scope for v10b.  Could land post-v10b if the
  Pillai-safe-band sensitivity sweep shows constant-`C_S` is a
  meaningful systematic.  Carried from step 7.
* **Singh `k_prot` consistency audit** (Singh 2016 K_eq = k_hyd /
  k_prot rather than k_hyd / k_des).  Different parameter from
  v10b's `k_des` scope; out of v10b scope.  Carried as separate task.
* **Legacy 348-line `C_S = 0.10 F/m^2` cleanup**.  Carried as
  separate post-v10b cleanup task (NOT v10b's responsibility per
  plan section 6).
* **MOH adsorbate coverage measurement** at the OHP in K2SO4 /
  sp2-carbon (XPS / surface-enhanced IR / impedance).  No
  peer-reviewed source currently reports this; if one emerges, the
  V10A chain anchor for Gamma_max would be replaced.

---

## 7. Cross-references

* Plan: `.claude/plans/phase6b-step8-v10b-calibration.md` (v7-FINAL).
* Critique trail:
  `docs/handoffs/CHATGPT_HANDOFF_36_phase6b-v10b-calibration/`
  (R1-R7 + FINAL_REVISION).
* CMK-3 capacitance writeup:
  `docs/phase6/CMK3_capacitance_literature.md` (step 7).
* Acceptance bundle:
  `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`
  (section Status step 8).
* Solver module: `Forward/bv_solver/cation_hydrolysis.py`.
* Calibration package: `calibration/v10b.py`.
* CLAUDE.md hard rules #2 (exponent_clip), #4 (Ruggiero topology),
  #6 (C_S anchor).
