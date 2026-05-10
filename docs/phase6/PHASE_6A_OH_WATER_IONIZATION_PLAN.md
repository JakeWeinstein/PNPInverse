# Phase 6α — OH⁻ / Water Self-Ionization Implementation Plan

**Status**: GPT-loop converged at round 5 / 5 (verdict APPROVED).
**Author**: Claude
**Date**: 2026-05-09 (initial draft) — 2026-05-10 (revised after
critique-loop convergence; see
`docs/CHATGPT_HANDOFF_25_phase-6a-water-ionization/`).
**Triggered by**: `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/verdict.json`

> **Major revision from initial draft**: the original plan recommended
> Option B (algebraic `c_OH = Kw / c_H` slaving in Poisson only) as the
> cheapest first cut.  GPT's adversarial review showed Option B is
> physically inconsistent — substituting OH⁻ into Poisson without
> sourcing H⁺ leaves the H⁺ NP equation unmodified, so the surface
> c_H still crashes regardless of how OH⁻ contributes to charge
> density.  The only mass-conserving fast-water limit is the
> proton-condition variable `E = c_H − c_OH` (originally Option C).
> This plan therefore lands Option C as the primary path; Option B is
> dropped, Option D (full dynamic OH⁻ + finite-rate R_w) is reserved
> as the escalation path if Option C falls short.  See
> `R1_from_gpt.md` issues #3–#5, #23 for the architectural pivot.

## 1. Context — what just happened

The L_eff transport-domain sweep
(`.claude/plans/l-eff-transport-sweep.md`) completed 8/8 combos
13/13-converged in 44.4 min and emitted 3 falsifiable predictions:

| # | Prediction | Verdict | Detail |
|---|---|---|---|
| 1 | `\|cd_plateau\| ∝ 1/L_eff`, slope ∈ [0.85, 1.15] | **PASS** | Slope = 1.000, both ratios. Plateau matches `F·D_H·c_H_bulk/L_eff·0.1` to 4 digits. |
| 2 | No peak appears at any L_eff in V_RHE ∈ [-0.40, +0.55] V | **PASS** | No interior local cd minimum at any (L_eff, ratio). |
| 3 | `max_surface_pH < 9` at smallest L_eff (16 µm) | **FAIL** | pH = 13.72 at L_eff=16 µm; drops only 0.42 units across 6× L_eff reduction. |

**Plateau magnitudes vs deck (-0.18 mA/cm² target):**

| L_eff | plateau (mA/cm²) | residual vs deck |
|------:|-----------------:|-----------------:|
| 100 µm | -0.0899 | -50.1 % |
| 66 µm | -0.1362 | -24.4 % |
| 21 µm | -0.4279 | +137.7 % |
| 16 µm | -0.5617 | +212.0 % |

The deck plateau magnitude is matched at L_eff ≈ 50 µm by Levich.
But picking that L_eff alone won't make the model deck-comparable
because the surface pH would still pin at ~14 — far outside the
plausible RRDE range (4–9 by experimental consensus).

## 2. Diagnosis — what the model is missing

The current production stack (3 dynamic NP species O₂/H₂O₂/H⁺ +
analytic Bikerman counterions Cs⁺/SO₄²⁻ + parallel 2e/4e BV +
finite Stern compact layer) provides **no mechanism for H⁺
replenishment near the surface beyond diffusion from the bulk
top**.  When ORR consumes H⁺ at the cathode, the local
concentration crashes by ~14 orders of magnitude (surface pH =
13.72 ⟹ c_H_surface ≈ 1.9 × 10⁻¹⁴ M = 1.9 × 10⁻¹¹ mol/m³).

In a real aqueous electrolyte at pH 4, the homogeneous water
self-ionization

```
H₂O ⇌ H⁺ + OH⁻        Kw = [H⁺][OH⁻] = 10⁻¹⁴ M² = 10⁻⁸ (mol/m³)²
```

immediately re-establishes equilibrium pointwise.  Forward
dissociation rate `k_f · [H₂O] = k_r · Kw ≈ 1.4 × 10⁻³ M/s = 1.4
mol/m³·s` (correctly derived from detailed balance with `k_r ≈
1.4 × 10¹¹ M⁻¹s⁻¹`).  Relaxation timescale
`τ_water_eq ≈ 1/(k_r · (c_H + c_OH))` ranges from **71 ns at bulk**
(c_H ≈ 10⁻⁴ M) down to **7 ps at pH 14** (c_OH ≈ 1 M).  Compare to
diffusive transport timescale `τ_diff = L²/D_H ≈ 1.07 s at L_REF=100 µm`,
`27 ms at L_eff=16 µm`.  Damköhler ranges from `1.5 × 10⁵` (worst
case: small L_eff AND bulk c_H) to `3.9 × 10¹⁴` (best case),
uniformly Da ≫ 1.

The current model has no water source/sink and no OH⁻ field, so c_H
crashes unbounded → surface pH = 14.

## 3. The reduction we are landing — Option C (proton-condition variable)

### 3.1 Why not pure-Boltzmann (REJECTED)

Treating OH⁻ identically to Cs⁺ / SO₄²⁻ via inert Bikerman would
give `c_OH(y) = c_OH_bulk · exp(+φ(y)) · γ_steric(y)` (z = -1, so
`exp(-z·φ)` becomes `exp(+φ)`).  At a cathode φ < 0 ⟹ c_OH
**decreases**.  Empirically/via water equilibrium the opposite is
true.  Pure-Boltzmann is the right distribution for an *inert*
species in an EDL (homogeneous reaction equilibrium NOT enforced);
it's the wrong one for OH⁻ because the dominant driver is the
homogeneous water-equilibrium response to depleted c_H, not the
EDL field.  See R1 issue #10 for the scope clarification.

### 3.2 Why not algebraic-only c_OH = Kw/c_H slaving (REJECTED)

This was the original plan's "Option B".  Substituting `c_OH(y) =
Kw_hat · exp(-u_H(y))` into Poisson (and the Bikerman packing
closure) adds OH⁻ charge but does **NOT** modify the H⁺ NP
equation.  Surface c_H still crashes because BV consumes it
without replacement.  The only mass-conserving fast-water limit
that actually replenishes H⁺ requires the implied source
`R_w_implicit = -∂c_OH/∂t + ∇·J_OH` to enter the H⁺ residual,
which after symmetry collapses to the proton-condition variable.
See R1 issues #3–#5.

### 3.3 Option C — proton-condition variable (PRIMARY LANDING)

Define

```
E(y) ≡ c_H(y) − c_OH(y)
```

Then `∂E/∂t + ∇·J_E = R_w − R_w = 0` (water source cancels in the
difference — both reactions produce/consume H⁺ and OH⁻ in 1:1).
**One PDE replaces two**, and the water reaction term drops out
entirely by construction.  Constitutive closure via the
equilibrium constraint `c_H · c_OH = Kw_hat`:

```
c_H(E) = (E + √(E² + 4·Kw_hat)) / 2
c_OH(E) = (-E + √(E² + 4·Kw_hat)) / 2 = c_H − E
```

(Both formulae expressed in nondim concentration; Kw_hat = Kw_phys
/ C_SCALE² = 1 × 10⁻⁸ (mol/m³)² / 1.44 (mol/m³)² ≈ 6.94 × 10⁻⁹.)

Adopting Nernst-Planck with diffusivity in the migration term
(R2 issue #1): `J_i = -D_i · (∇c_i + z_i · c_i · ∇φ)`.  With c_OH =
Kw_hat/c_H:

```
J_E = J_H − J_OH
    = -[D_H · c_H + D_OH · c_OH] · (∇u_H + ∇φ)        (logc primary)
    = -[D_H · c_H + D_OH · c_OH] · ∇μ_H               (muh primary,
                                                       em·z_H = +1)
```

The migration coefficient `D_H · c_H + D_OH · c_OH = D_H · exp(u_H)
+ D_OH · Kw_hat · exp(-u_H)` is **well-conditioned in log/muh
primary**: at low pH the first term dominates with smooth
`exp(u_H)`; at high pH the second term dominates with smooth
`exp(-u_H)`.  No `exp(-2 u_H)` stiffness (R2 issue #8).

Conservative weak form (R2 issue #2, R3 issue #3):

```
∫ v · ∂E/∂t  dx  -  ∫ ∇v · J_E  dx  +  ∫ v · J_E·n  ds  =  0
```

For steady-state (which is the production target — every anchor
build and warm-walk solves SS at each V_RHE), the time term drops:

```
- ∫ ∇v · J_E  dx  +  ∫ v · J_E·n  ds  =  0
```

For transient runs (cold-ramp), the muh storage term is
`(c_H + c_OH) · (∂μ_H/∂t − ∂φ/∂t)` and there is a `∂φ/∂t`
coupling that the logc form doesn't have (R2 issue #4).

Boundary conditions (R2 issue #3, R3 issue #4):

| Boundary | Marker | Condition |
|---|---|---|
| Electrode (y=0) | 3 | `J_E·n = J_H,BV·n` (acidic-form BV; J_OH not separately constrained — see §6 R9 for validation) |
| Bulk top (y=L) | 4 | Dirichlet `c_H = c_H_bulk_hat`, `φ_top = 0` ⟹ `μ_H_top = ln c_H_bulk_hat` |
| Sides (x=0, x=1) | 1, 2 | No-flux `J_E·n = 0` |

Steric / activity convention: **concentration-Kw model**.  Sterics
applied only via Poisson source and packing-fraction closure;
**not** as an activity correction on water equilibrium.  This
matches Yash's reference implementation (to be verified in the
cross-check).  Activity-Kw with `γ_H · γ_OH · c_H · c_OH = Kw_thermo`
is out of Phase 6α scope.  See R2 issue #12.

### 3.4 Option D — full dynamic OH⁻ + finite-rate R_w (ESCALATION)

If Option C fails any acceptance gate (in particular Gate 4: fast-
water validity `max |R_w,req_hat| · 0.163 < 0.1`), escalate to
Option D: add OH⁻ as a 4th NP species with explicit finite-rate
source/sink `R_w = k_r · (Kw - c_H · c_OH)` in both H⁺ and OH⁻
equations.  Matches Yash's 6-species reference exactly.  Adds
~25 % function-space DOFs and a stiff source term requiring
implicit treatment.  Out of scope for Phase 6α; specified for
phase 6α.5 if needed.

## 4. Damköhler / fast-water validity

| Quantity | Bulk (pH 4) | Surface pH 14 (failed sweep) |
|---|---|---|
| c_H (M) | 10⁻⁴ | 10⁻¹⁴ |
| c_OH (M) | 10⁻¹⁰ | 1 |
| 1/(k_r·(c_H+c_OH)) | ≈ 71 ns | ≈ 7 ps |
| τ_diffusion = L²/D_H |  | 1.07 s (L=100 µm), 27 ms (L=16 µm) |
| Da = τ_diff / τ_water | 1.5 × 10⁷ | 3.9 × 10¹⁴ |

Da ≫ 1 uniformly across the planned sweep.  Fast-water assumption
is robust as a *zeroth-order* model; the Gate-4 validity check
(see §5 below) tests it post-solve via the deviation-from-
equilibrium metric.

**Finite water-source capacity** (R2 issue #6, R3 issue #1):

```
i_max [mA/cm²] = R_w_max_phys · L_eff · F · 0.1
              = 1.4 (mol/m³·s) · L_eff (m) · 96485 (C/mol) · 0.1
              = 1.35 · (L_eff/100 µm) mA/cm²
```

| L_eff | i_max water-only | deck plateau (-0.18) | deck peak (-0.40) |
|------:|----------------:|---------------------:|------------------:|
| 100 µm | 1.35 mA/cm² | 7.5× headroom | 3.4× headroom |
| 66 µm  | 0.89        | 5.0×           | 2.2×           |
| 21 µm  | 0.28        | 1.6×           | **insufficient** (0.7×) |
| 16 µm  | 0.22        | 1.2× (very tight) | **insufficient** (0.55×) |

Comfortable for the deck plateau and Phase 6α's surface-pH-lift goal.
Insufficient for the deck peak at L_eff ≤ 21 µm — but the deck peak
is not a Phase 6α goal; recovering it is Phase 6β (sulfate buffering)
or Phase 6γ.

Sustained sulfate transport ceiling (R3 issue #6, R4 issue #3) for
comparison:

| L_eff | sulfate transport (HSO₄⁻ from bulk) | water source |
|------:|--------------------------------------:|-------------:|
| 100 µm | 0.097 mA/cm² | 1.35 mA/cm² |
| 16 µm  | 0.60          | 0.22         |

At L_eff = 100 µm (the gating Yash-comparison condition), water
source dominates (1.35 vs 0.097).  At small L_eff, especially
≤ 21 µm, sulfate transport dominates.  Phase 6α targets
L_eff = 100 µm + plateau, where water alone suffices.

## 5. Implementation plan

### Q1 — Constants and config knobs (no Firedrake, ~30 LoC)

`scripts/_bv_common.py`:

```python
KW_MOLAR_SQUARED = 1.0e-14   # M², CRC handbook, 25 °C
                              # M² = (mol/L)² = (1000 mol/m³)² = 1e6 (mol/m³)²
KW_PHYS = KW_MOLAR_SQUARED * 1.0e6   # = 1e-8 (mol/m³)²
KW_HAT = KW_PHYS / (C_SCALE ** 2)    # = 6.94e-9 (dimensionless)

C_H_BULK_M3 = 0.1                    # mol/m³ at pH 4
C_OH_BULK_M3 = KW_PHYS / C_H_BULK_M3 # = 1e-7 mol/m³ ≡ 1e-10 M
C_OH_BULK_HAT = C_OH_BULK_M3 / C_SCALE  # = 8.33e-8

D_OH = 5.273e-9       # m²/s, CRC handbook (Marcus mobility)
D_OH_HAT = D_OH / D_REF
A_OH_HAT = (4.0 / 3.0) * math.pi * (1.76e-10)**3 * 6.022e23 * C_SCALE
# = 1.65e-5  (Marcus crystallographic radius 1.76 Å)
```

Test: `KW_HAT == C_H_BULK_HAT * C_OH_BULK_HAT` to 1e-12 relative.

A new species-config flag at solver-params construction:
`enable_water_ionization: bool = False`.  Default off.  When True,
`make_bv_solver_params` registers an additional residual term
(see Q2).  The flag gates the entire Phase 6α path so existing
regression tests run untouched.

### Q2 — Proton-condition residual in `forms_logc.py` and `forms_logc_muh.py`

Replace the H⁺ NP residual contribution with the proton-condition
residual.  Touch surfaces (R2 issue #12):

| File | Function | Change |
|---|---|---|
| forms_logc.py | `_build_np_residual` (H⁺ branch) | Replace `J_H` flux with `J_E = -(D_H exp(u_H) + D_OH Kw_hat exp(-u_H)) · (∇u_H + ∇φ)`; replace storage `c_H · ∂u_H/∂t` with `(c_H + c_OH) · ∂u_H/∂t` |
| forms_logc.py | BV cathodic concentration factor | Already uses `c_H = exp(u_H)`; no change |
| forms_logc.py | Poisson source | Add `(-z_OH) · c_OH = +Kw_hat · exp(-u_H)` term (using `c_OH` reconstructed from same u_H) |
| forms_logc.py | Bikerman shared-θ closure | Add `a_OH_hat · Kw_hat · exp(-u_H)` to packing fraction |
| forms_logc.py | `_try_debye_boltzmann_ic` | See Q3 |
| forms_logc_muh.py | `_resolve_mu_h_index` and `build_forms_logc_muh` | Same changes, but flux is `-(D_H c_H + D_OH c_OH) · ∇μ_H` (muh form is cleaner — no separate ∇φ term in flux) |
| forms_logc_muh.py | Top BC | Document `μ_H_top = ln c_H_bulk_hat` requires `φ_top = 0` (R3 issue #7) |

Continuation parameter `Kw_eff` ramped from 0 to Kw_hat during
anchor build (R2 issue #15):

```
Kw_eff schedule = [0, Kw_hat * 1e-6, Kw_hat * 1e-3, Kw_hat * 0.1, Kw_hat]
```

Mirrors the existing k0 ladder in
`solve_anchor_with_continuation`.  All Newton-stability concerns
(R1 issue #14, R3 issue #7) are addressed via continuation, not
clamps.  The existing `u_clamp = 100` is retained (it bounds
`exp(u_H)` at low pH and `Kw_hat · exp(-u_H)` at high pH
symmetrically because the muh/logc primary is well-conditioned).

> **Clamp-inactive policy** (R5 minor #3): any accepted production
> or validation run must report that the `u_clamp` is inactive at
> all quadrature points; otherwise the run is diagnostic only and
> may not be cited as evidence of Phase 6α physics.

### Q3 — IC reconstruction (Path B, approximate)

`Forward/bv_solver/forms_logc.py:_try_debye_boltzmann_ic` and the
muh sibling.  The existing Picard outer loop is **counterion-aware
but not water-aware**; making it water-aware (Path A) requires a
substantial rewrite of the Picard surface-rate algebra to use the
proton-condition flux form.  Phase 6α adopts **Path B**: keep the
existing Picard loop, label its IC as approximate, and rely on
Newton + Kw_eff continuation to relax the IC into the true
E-equation steady state.

The Phase 6α IC implementation:

1. Run the existing Picard outer loop (unchanged) → c_H(y) profile.
2. Compute `c_OH(y) = Kw_eff · exp(-u_H(y))` pointwise (Kw_eff is
   the active continuation-parameter value).
3. Update Poisson charge density to include OH⁻.
4. Update Bikerman shared-θ closure to include OH⁻ packing.
5. Re-evaluate the proton-condition residual norm
   `||F_E|| / ||u_H||`; track per-rung in continuation logs.

This IC violates the proton-condition equation in absolute terms
but lands Newton in the right basin of attraction at each Kw_eff
rung.  Acceptance: the residual norm at IC must be < 1e-2 at
Kw_eff = 0 (recovers the existing IC) and decrease as continuation
progresses.

> **Hard-trigger fallback to Path A** (R5 minor #5): if the anchor
> Newton fails at any Kw_eff rung, OR the post-IC proton-condition
> residual stays above `1e-2 · ||u_H||` after two ladder
> refinements (insertion of mid-points), the IC is declared
> insufficient and Phase 6α.5 (Path A — water-aware Picard
> rewrite) is required.

### Q4 — Tests

New test file `tests/test_water_ionization_phase_6a.py`:

**Unit (fast)**:
- `test_kw_hat_consistency`: `KW_HAT == C_H_BULK_HAT * C_OH_BULK_HAT`
  to relative 1e-12.
- `test_bulk_oh_concentration`: at pH 4 in mol/m³, `c_OH_bulk = 1e-7`
  exact.
- `test_solver_params_default_off`:
  `make_bv_solver_params(...)` (no `enable_water_ionization` kwarg)
  produces the existing 3sp+counterion solver-params dict
  byte-equivalent to the pre-Phase-6α baseline.
- `test_solver_params_on_routes_correctly`:
  with `enable_water_ionization=True`, the solver-params dict has
  the new Kw_hat field and at least one `kind="water_ionization"`
  entry in the residual config.
- `test_no_water_ionization_residual_unchanged`: with the flag off,
  the assembled residual UFL has zero new terms (programmatic
  residual-form inspection — replaces the brittle expression-shape
  regression of R2 issue #14).

**Slow (Firedrake)**:
- `test_anchor_at_l_eff_100um_with_water`: anchor build at +0.55 V
  on the 3sp + multi-ion + water-ionization stack converges
  through the 5-rung Kw_eff continuation.
- `test_mms_proton_condition`: MMS verification with manufactured
  smooth `(u_H(y), φ(y))` pair, derived `c_OH = Kw_hat · exp(-u_H)`,
  and analytic forcing `s(y) = ∇·J_E` (using the convention from
  the conservative weak form).  The forcing is added to the weak
  residual as `∫ v · s(y) dx`.  Mesh-refinement: Ny ∈ {20, 40, 80,
  160}, β = 3.0.  Acceptance: L²-error in u_H decays at order ≥ 2
  for CG-1 elements (smooth manufactured solution).
- `test_disabled_path_numerical_equivalence`: with the flag off,
  `mangan_full_grid_csplus_so4.py` produces cd / pc / surface_pH /
  j_ring values within 1e-10 relative tolerance of the
  pre-Phase-6α baseline (checked-in JSON snapshot).

### Q5 — Validation sweep

Re-run the L_eff transport sweep with water ionization enabled:

```bash
python -u scripts/studies/l_eff_transport_sweep_csplus_so4.py \
       --enable-water-ionization
python -u scripts/studies/score_l_eff_sweep.py
```

The driver gets a `--enable-water-ionization` flag that flips the
new solver-params kwarg.  Per-combo iv_curve.json gains:

- `R_w_req_hat_max[i]`: max over domain of weakly-projected
  `R_w,req_hat` at V_RHE[i] (per R3 issue #5,
  R4 issue #5; "R_w,req_hat is the mass-matrix projection of the
  weak H+ residual into the chosen scalar space" — R5 minor #4).
- `epsilon[i] = R_w_req_hat_max[i] · 0.163` (nondim-to-physical
  conversion factor; R4 issue #2).
- `cd_E_balance[i]`: independent assembly of cd from the
  E-equation flux at the electrode boundary, reusing
  `_build_bv_observable_form` machinery (R3 issue #8).
- `J_OH_n_inferred_at_electrode[i]`: post-hoc evaluation of OH⁻
  flux at the electrode, assembled directly via Firedrake
  `FacetNormal` (no hand-expansion of signs — R5 minor #1).

The post-solve `score_l_eff_sweep.py` evaluates 5 acceptance gates:

1. **Surface pH (P3)**: `max_surface_pH < 9` at L_eff = 16 µm.
2. **Plateau direction-of-change at L=100 µm**: cd at deepest
   cathodic V_RHE more negative than the pre-Phase-6α value
   (-0.0899 mA/cm²), trending toward deck -0.18.
3. **E conservation per V_RHE**:
   `|cd_solver - cd_E_balance| / |cd_solver| < 1e-3`.
4. **Fast-water validity per V_RHE**:
   `max |epsilon[i]| < 0.1` over (L_eff, V_RHE) combos.
5. **Yash cross-check** at L=100 µm, V_RHE=-0.40 V, pH 4,
   Cs⁺/SO₄²⁻ matched, mesh height 100 µm:
   - c_OH(y) profile within 1 OOM at every node, within 50%
     relative error at the OHP.
   - `|J_OH_n_inferred| / |J_H,BV·n| < 0.05` at the electrode (R3
     issue #7 — the reduced-BC validation).

Pass conditions:
- Gates 1-3 PASS + Gate 4 PASS + Gate 5 PASS → Phase 6α complete;
  proceed to Phase 6β (cation-dependent selectivity) if deck peak
  recovery is the next goal, else Phase 6γ (alkaline-form ORR).
- Gate 4 FAIL anywhere → Option C is breaking down; escalate to
  Option D (Phase 6α.5).
- Gate 5 J_OH check FAIL → reduced BC is hiding a boundary-layer
  artifact; escalate to Option D (Phase 6α.5).
- Gates 1, 2, or 3 FAIL with Gate 4 PASS → physics gap beyond
  water alone (sulfate buffering, cation effect, alkaline-form
  ORR); document and reassess.

## 6. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Option C fails Gate 1 (P3) | Escalate to Option D (full dynamic OH⁻). |
| R2 | Newton stiffness during Kw_eff continuation | Tighten the 5-rung schedule to 8 rungs; mid-point insertion at the failing rung. |
| R3 | c_OH packing dominance at high pH (~1 M at pH 14) | Continuation handles it (start at Kw_eff=0, ramp slowly); if Bikerman θ saturates, document as Phase 6α.5 trigger. |
| R4 | IC composite-ψ closure with water inconsistent | Path B labeled approximate; hard-trigger fallback to Path A per Q3. |
| R5 | Default-off flag not actually preserving existing regression | `test_solver_params_default_off` + `test_disabled_path_numerical_equivalence` enforce both byte and numerical equivalence. |
| R6 | Damköhler/fast-water assumption invalid in some regime | Gate 4 (`max |ε| < 0.1`) tests it directly per V_RHE point. |
| R7 | log/muh primary flux coefficient stiffness | Coefficient is `D_H exp(u_H) + D_OH Kw exp(-u_H)`, well-conditioned by construction (no exp(-2u) form anywhere). |
| R8 | Finite water source capacity insufficient at small L_eff | Acceptance gates target L=100 µm; small L_eff regimes flagged in §4 as out of scope for water alone. |
| R9 | Reduced BC J_OH·n hides Faradaic OH⁻ artifact | Gate 5 sub-check `|J_OH·n_inferred| / |J_H,BV·n| < 0.05` catches this. |
| R10 | MMS forcing sign mismatch with weak form | MMS protocol explicitly derives `s(y)` from the conservative weak form sign convention (`∂E/∂t − ∇v·J_E + boundary` form) per R3 issue #3 and R4 issue #8. |

## 7. Out of scope (deferred to later phases)

- **Sulfate buffering** (HSO₄⁻ ⇌ SO₄²⁻ + H⁺): at L_eff = 100 µm
  (gating condition), water source (1.35 mA/cm²) >> sulfate
  transport (0.097 mA/cm²), so sulfate is not load-bearing for
  Phase 6α's surface-pH-lift goal.  At small L_eff (especially
  ≤ 21 µm), sulfate transport (0.60 at 16 µm) dominates and
  becomes necessary for deck-peak recovery.  Phase 6β.
- **HSO₄⁻ as a separate dynamic species**: scope multiplier; defer
  to Phase 6β.
- **Cation-dependent selectivity factor**: orthogonal to local-pH;
  Phase 6γ.
- **Alkaline-form ORR pathway**: at pH 4 the acidic-form rate
  expressions dominate; would change the J_E·n electrode BC
  (Faradaic OH⁻ flux becomes nonzero).  Phase 6δ.
- **Activity-Kw model** (`a_H · a_OH = Kw_thermo` with γ_i activity
  corrections): out of Phase 6 scope; sticking with concentration-
  Kw + sterics-in-Poisson convention per R2 issue #12.

## 8. Open questions (finalized after critique loop)

1. **Kw_eff continuation rung schedule**: 5 rungs is the initial
   spec.  If Newton struggles at any rung, insert mid-points
   (8-10 rungs).  Empirical tuning during Q4 slow tests.
2. **MMS convergence order**: expect ≥ p+1 = 2 for CG-1 with
   smooth manufactured solution.  If observed order is < 1.5,
   debug the residual sign / boundary terms.
3. **IC residual tolerance threshold**: 1e-2 relative for "good
   enough" before Newton; 1e-3 if Newton is iteration-budget
   limited.

## 9. Implementation cost estimate

| Task | LoC estimate | Wall-time (build + test) |
|---|---|---|
| Q1 (constants) | ~50 LoC | 1-2 h |
| Q2 (residual rewrite, 2 backends) | ~300 LoC | 1-2 days |
| Q3 (IC + continuation plumbing) | ~150 LoC | 1 day |
| Q4 (tests, MMS, regression) | ~400 LoC | 1-2 days |
| Q5 (validation sweep + scorer extension) | ~100 LoC | 0.5 day + ~1 h sweep wall |
| **Total** | **~1000 LoC** | **5-7 days** |

The critical path is Q2 (residual rewrite); everything else can
proceed in parallel once Q2 lands.

## Cross-reference

Critique session: `docs/CHATGPT_HANDOFF_25_phase-6a-water-ionization/`.

Key issue ledger:
- R1 #3-#5, #23 → Option B → Option C pivot.
- R2 #1, #2, #4 → NP flux + weak form + muh derivation.
- R2 #6 → finite water source capacity gate.
- R3 #1 → 10× arithmetic correction.
- R3 #5, R4 #2, #5 → fast-water validity metric (Gate 4).
- R5 minor #1-#5 → revision TODOs (FacetNormal, sulfate text,
  clamp-inactive policy, Gate 4 projection note, hard-trigger
  fallback).
