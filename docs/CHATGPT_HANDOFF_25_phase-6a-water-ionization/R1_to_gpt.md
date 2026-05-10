# Round 1 — Phase 6α OH⁻ / water self-ionization plan, adversarial review

## 1. Context bundle

### 1.1 Project at a glance

PNP+BV forward solver for ORR at a rotating ring-disk electrode in
aqueous K₂SO₄ / Cs₂SO₄ at pH 4. Goal: forward-match the page-15
deck of Mangan 2025 (cd vs V_RHE on V_RHE ∈ [-0.40, +0.55] V) so
that downstream inverse work has a defensible forward.

Ruggiero 2022 (the source paper underlying the Mangan deck — Mangan
is a coauthor) describes parallel 2e/4e ORR at acidic pH:

```
R_2e :  O₂ + 2H⁺ + 2e⁻ → H₂O₂        E°_2e = 0.695 V
R_4e :  O₂ + 4H⁺ + 4e⁻ → 2H₂O        E°_4e = 1.23  V
```

Production stack (the May 2026 cleanup target, now landed):

* **3 dynamic NP species** (`THREE_SPECIES_LOGC_BOLTZMANN`):
  - O₂  (z=0, D=1.9e-9 m²/s, c_bulk=1.2 mol/m³)
  - H₂O₂ (z=0, D=1.6e-9, seeded at 1e-4 in nondim units)
  - H⁺   (z=+1, D=9.311e-9, c_bulk=0.1 mol/m³)
* **Analytic Bikerman counterions** (multi-ion shared-θ closure):
  - Cs⁺   (z=+1, c_bulk=199.9 mol/m³, r=2.2 Å, a_nondim=3.23e-5)
  - SO₄²⁻ (z=-2, c_bulk=100.0,         r=2.4 Å, a_nondim=4.20e-5)
* **Parallel 2e/4e BV at electrode** (`PARALLEL_2E_4E_REACTIONS`)
* **Stern compact layer**, C_S = 0.10 F/m²
* **Primary variables**: `u_i = ln c_i` (logc) for all dynamic
  species; for the muh formulation H+ is stored as `μ_H = u_H +
  em·z_H·φ` with `em·z_H = +1` at production scaling.
* **Initializer**: `debye_boltzmann` — a Picard outer loop
  (ambipolar 2D_H proton transport + 2x2 algebraic surface-rate
  Picard) followed by a Gouy-Chapman / BKSA matched-asymptotic
  composite-ψ inner profile; multispecies γ closure on the dynamic
  species.

Nondimensionalization: L_REF = 100 µm, D_REF = D_O₂ = 1.9e-9 m²/s,
C_SCALE = C_O₂ = 1.2 mol/m³, V_T = RT/F ≈ 25.69 mV. Time scale
τ_ref = L_REF² / D_REF ≈ 5.26 s.
`I_SCALE = n_e · F · D_REF · C_SCALE / L_REF · 0.1 = 0.439 mA/cm²`
(per electron-equivalent).

H+ Levich limit (electron-equivalent, n_e = 1):
`F · D_H · c_H_bulk / L_REF · 0.1 = 96485 · 9.311e-9 · 0.1 / 1e-4 ·
0.1 = 0.0898 mA/cm²` — the smoking gun.

### 1.2 Recent landings that shape the plan

* **L_eff transport-domain knob landed** (today): mesh y-extent
  decoupled from L_REF via `domain_height_hat = l_eff_m / L_REF`;
  IC routines normalize their outer linear interpolation by
  `y_norm = y / domain_height_hat`. Default 1.0 reproduces legacy
  unit-cube. See `Forward/bv_solver/mesh.py:make_graded_rectangle_mesh`,
  `scripts/_bv_common.py:make_bv_solver_params(l_eff_m=...)`,
  `tests/test_l_eff_smoke.py`.
* **Parallel 2e/4e topology landed** (M3a.2, 2026-05-07):
  `PARALLEL_2E_4E_REACTIONS` replaces the legacy sequential R_1+R_2.
* **Multi-ion shared-θ Bikerman landed** (Phase 5α, 2026-05-09):
  `Forward/bv_solver/multi_ion.py` solves for outer-region phi_o
  numerically with multiple bikerman counterions; `boltzmann.py`
  builds the residual.
* **Anchor + warm-walk continuation** (Phase 5γ): all current
  studies land via `solve_anchor_with_continuation` then
  `solve_grid_with_anchor`.

### 1.3 What just falsified

L_eff transport sweep (`.claude/plans/l-eff-transport-sweep.md`):
8 combos (4 L_eff × 2 K0_R4e/K0_R2e ratios), 13 V_RHE points each,
all 8/8 anchor-converged + 13/13 grid-converged in 44.4 min wall.

Verdict (`StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/verdict.json`):

| # | Prediction | Verdict | Detail |
|---|---|---|---|
| 1 | `\|cd_plateau\| ∝ 1/L_eff`, slope ∈ [0.85, 1.15] | **PASS** | Slope = 1.00001 for both ratios. |
| 2 | No peak in V_RHE band | **PASS** | No interior local cd minimum. |
| 3 | `max_surface_pH < 9` at L_eff=16 µm | **FAIL** | 13.72 at L_eff=16 µm (drop only 0.4 pH from 14.14 at 100 µm across 6× L_eff reduction). |

Plateau magnitudes:

| L_eff | plateau (mA/cm²) | F·D_H·c_H/L_eff·0.1 (theory) |
|------:|-----------------:|------------------------------:|
| 100 µm | -0.0899 | -0.0898 |
| 66 µm  | -0.1362 | -0.1361 |
| 21 µm  | -0.4279 | -0.4280 |
| 16 µm  | -0.5617 | -0.5617 |

Numerical agreement to 4 digits, identical across both K0_R4e/K0_R2e
ratios. Confirms: cathodic plateau is **purely H+ Levich-limited**,
independent of R_4e kinetics. Picking L_eff ≈ 50 µm by Levich
matches the deck plateau MAGNITUDE (-0.18 mA/cm²), but P3 says
surface pH would still pin at 14 — **deck says pH locally must be
in 4-9 range, so even Levich-correct magnitude would leave the
model structurally non-deck-comparable**.

### 1.4 Yash's reference simulation (the alkaline truth)

Independent parallel reference at
`data/EChem Reactor Modeling-Seitz-Mangan/Yash-Trends and Data and
Plotting.zip`. 6-species dynamic PNP+BV: H⁺, K⁺, **OH⁻**, SO₄²⁻,
O₂, H₂O₂. Per-voltage `.npy` snapshots store `conc_OH-` as a
121-point profile along with cd, selectivity, surface c_H, etc.
We have it, we just haven't ported the OH⁻ side yet.

### 1.5 Constants relevant to the plan

* Kw = 10⁻¹⁴ M² at 25 °C (water self-ionization).
* H⁺ + OH⁻ → H₂O rate: k_r ≈ 1.4 × 10¹¹ M⁻¹·s⁻¹ (Eigen 1964).
* Forward dissociation: k_f = k_r · Kw ≈ 1.4 × 10⁻³ s⁻¹ (per M of
  H₂O, so effective k_f for c_H production is k_f · c_H₂O ≈
  1.4 × 10⁻³ · 55.5 ≈ 0.078 mol/m³·s in water).
* D_OH ≈ 5.273 × 10⁻⁹ m²/s.
* Marcus crystallographic radius r_OH ≈ 1.76 Å.

Nondim: with L_REF = 100 µm and D_REF = 1.9e-9, the diffusive
timescale τ_ref = 5.26 s and the **transport Damköhler**
`Da = k_r · c_typ · L_REF² / D_REF`. With c_typ taken as the
larger of c_H and c_OH (call it 1 mol/m³ ≈ bulk H+, or 1e-5 mol/m³
near the cathode for OH⁻):
* At c_typ = 0.1 mol/m³ (= 1e-4 M, bulk H+): Da ≈ 1.4 × 10¹¹ ·
  1e-4 · (1e-4)² / 1.9e-9 ≈ 7.4 × 10⁵.  Da ≫ 1 ✓.
* At c_typ = 1e-9 M (worst-case surface c_H): Da ≈ 7.4. Marginal.
  But c_OH at the same point is Kw/c_H = 1e-5 M giving Da ≈ 7.4 ×
  10⁴ for c_OH. So at least one of (H+, OH-) is always Da ≫ 1.

The fast-water assumption looks safe. But this is an order-of-
magnitude argument — please pressure-test it.

### 1.6 What's already in the codebase that the new species would
extend (so reviewer knows where the plumbing already exists)

* `scripts/_bv_common.py:DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`,
  `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC` — the existing
  inert Bikerman counterion entries; the new OH⁻ entry will be a
  sibling.
* `Forward/bv_solver/boltzmann.py:add_boltzmann_counterion_residual`
  and `build_steric_boltzmann_expressions` — wire the analytic
  counterion into Poisson (charge density) and into the Bikerman
  γ closure (packing fraction). These are the touch sites.
* `Forward/bv_solver/multi_ion.py:solve_outer_phi_multiion`,
  `effective_debye_length_local` — where the Picard-IC outer-region
  phi_o is solved with multiple counterions; the IC for the new
  stack would extend this to include OH⁻ slaved to H+.
* `Forward/bv_solver/forms_logc.py:_try_debye_boltzmann_ic` and
  `forms_logc_muh.py:_try_debye_boltzmann_ic_muh` — composite-ψ +
  multispecies γ IC seeding. Only the multispecies-γ denominator
  needs to know about the OH⁻ contribution.

## 2. The artifact under review

The plan as written is a single doc. Pasting it in full below for
GPT to mark up by section number.

---

```markdown
# Phase 6α — OH⁻ / Water Self-Ionization Implementation Plan

**Status**: draft, pending GPT adversarial review
**Author**: Claude (with GPT-loop critique to follow)
**Date**: 2026-05-09
**Triggered by**: `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/verdict.json`

## 1. Context — what just happened

The L_eff transport-domain sweep
(`.claude/plans/l-eff-transport-sweep.md`) completed 8/8 combos
13/13-converged in 44.4 min and emitted 3 falsifiable predictions:

| # | Prediction | Verdict | Detail |
|---|---|---|---|
| 1 | `|cd_plateau| ∝ 1/L_eff`, slope ∈ [0.85, 1.15] | **PASS** | Slope = 1.000, both ratios. Plateau matches `F · D_H · c_H_bulk / L_eff · 0.1` to 4 digits. |
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

The current production stack:

* **3 dynamic NP species**: O₂, H₂O₂, H⁺
* **Analytic Bikerman counterions**: Cs⁺, SO₄²⁻
* **Parallel 2e/4e BV** at the electrode
* **Finite Stern compact layer**

provides **no mechanism for H⁺ replenishment near the surface beyond
diffusion from the bulk top**.  When ORR consumes H⁺ at the cathode,
the local concentration crashes by ~10⁵× to ~10⁻⁹ mol/m³.  In a real
aqueous electrolyte at pH 4, the homogeneous water self-ionization

```
H₂O ⇌ H⁺ + OH⁻        Kw = [H⁺][OH⁻] = 10⁻¹⁴ M²
```

immediately re-establishes equilibrium pointwise (Damköhler ≫ 1
because k_r ≈ 1.4 × 10¹¹ M⁻¹ s⁻¹).  This caps local pH around
7-9 in moderate cathodic regimes by sourcing fresh H⁺ from water,
producing OH⁻ in 1:1 ratio.

The current model has no water source/sink and no OH⁻ field, so c_H
crashes unbounded → surface pH = 14.

## 3. Shortcut options — which one to land

### Option A — Pure analytic Boltzmann (REJECTED)

Treat OH⁻ identically to Cs⁺ / SO₄²⁻:

```
c_OH(y) = c_OH_bulk · exp(+φ(y)) · γ_steric(y)
```

**Wrong**: predicts c_OH **decreases** at the cathode (φ < 0
⟹ exp(+φ) ≪ 1 ⟹ c_OH(surface) ≪ c_OH_bulk = 10⁻¹⁰ M).
Empirically and via water equilibrium, c_OH should **increase** by
~10⁵× near a cathode under ORR.  Pure Boltzmann is the right
distribution for an **inert** species in an EDL; it's the wrong one
for OH⁻ because the dominant driver is the water-equilibrium
response to depleted c_H, not the EDL field.

### Option B — Mass-action slaving (RECOMMENDED FIRST PASS)

Take the fast-water limit and bind OH⁻ algebraically to dynamic c_H:

```
c_OH(y) = Kw / c_H(y)        pointwise, no separate NP equation
```

In log-c primary variable: `c_OH(y) = Kw · exp(-u_H(y))`.

**Implementation cost** (analogous to existing Bikerman counterion
plumbing):

1. Poisson source: add `(−e) · Kw · exp(−u_H)` to charge density.
2. Bikerman shared-theta closure: add `a_OH · Kw · exp(−u_H)` to θ.
3. Bulk consistency: c_OH_bulk = Kw / c_H_bulk = 10⁻¹⁰ M.
4. Bulk Dirichlet for the algebraic field is automatic (slaved to
   the c_H Dirichlet at mesh top).

**Drops**:
- OH⁻ Nernst-Planck transport (OH⁻ produced near the cathode does
  not independently diffuse away).  Instead, OH⁻ piles up wherever
  c_H crashes; the spatial profile is dictated entirely by c_H.

**Cost**: 1 new algebraic field, no extra dynamic DOFs, no new
Newton iteration cost.  Very close in code shape to the existing
`build_steric_boltzmann_expressions` path.

### Option C — Proton-condition variable

Define the proton-excess scalar:

```
E(y) ≡ c_H(y) − c_OH(y)
```

Then ∂E/∂t + ∇·J_E = R_w − R_w = 0 (water-source terms cancel
symmetrically).  Solve **one** PDE for E; back-solve c_H and c_OH:

```
c_H = (E + √(E² + 4 Kw)) / 2
c_OH = (-E + √(E² + 4 Kw)) / 2
```

**Pros**:
- Eliminates water reaction term entirely (no stiff source).
- Recovers OH⁻ Nernst-Planck transport correctly through J_E.
- One dynamic species replaces two (H⁺ + OH⁻).

**Cons**:
- Requires changing the H⁺ primary variable from `u_H = ln c_H`
  (or `μ_H = u_H + em·z·φ` in muh) to E.
- Affects **every** H⁺-touching site in `forms_logc.py` /
  `forms_logc_muh.py`: NP flux, BV cathodic factor, Bikerman
  packing, Poisson source, IC.
- Loses byte-compatibility with existing tests.
- More invasive than Option B; not a drop-in shortcut.

### Option D — Full dynamic OH⁻ + R_w source (Yash's reference)

Add OH⁻ as a 4th NP species; add stiff bulk source/sink

```
R_w = k_f − k_r · c_H · c_OH        with k_r/k_f = 1/Kw
∂c_H/∂t + ∇·J_H = … + R_w
∂c_OH/∂t + ∇·J_OH = R_w
```

**Pros**: most physical; matches Yash's 6-species reference code
exactly.

**Cons**:
- One extra dynamic species → +25 % function-space DOFs.
- Stiff R_w (k_r·c_H·c_OH ~ k_r·Kw ≈ 1.4 × 10⁻³ s⁻¹·M; with
  Da = k_r · L²/D ~ 10⁹ at L_REF=100 µm) requires either
  implicit treatment or operator-splitting.
- Most expensive landing.

## 4. Recommendation

Land **Option B (mass-action slaving)** as Phase 6α.  Reasons:

1. **Cheapest to implement**: one new algebraic field plumbed into
   Poisson + Bikerman, mirroring the existing analytic-counterion
   path.  No new primary variable, no new dynamic species, no new
   Newton-stiff source term.
2. **Falsifiable test**: same sweep we just ran will tell us whether
   Option B reaches the deck-target surface pH.  If yes → ship.  If
   no → escalate to C.
3. **Damköhler order-of-magnitude check supports it**: with
   k_r·c_H_bulk ≈ 1.4 × 10⁻³ s⁻¹·(0.1 mol/m³) = 1.4 × 10⁻⁴ s⁻¹ at
   bulk and c_H_surface ≈ 10⁻⁹ M giving k_r·c_H_surf ≈ 1.4×10⁻⁸ s⁻¹
   forward and ≈ 1.4×10⁻⁴ s⁻¹ reverse, both ≫ steady-state
   approach time τ_ss ≈ L_eff² / D_H ~ 1 ms at L_eff = 100 µm.
   Local water equilibrium is essentially instantaneous on the
   transport timescale.
4. **OH⁻ transport drop is bounded**: in the fast-water limit, OH⁻
   production rate equals c_H consumption rate; the difference
   between Option B and Option C is a second-order correction
   from OH⁻ flux divergence.  Should be small for L_eff in the
   16-100 µm range.

If Option B fails P3 (surface pH still > 9 after water term),
escalate to Option C.  Don't go directly to Option D — too
expensive given the milder physics of Option C.

## 5. Implementation plan — Option B mass-action shortcut

### Q1 — Constants and config knobs (no Firedrake, ~30 LoC)

`scripts/_bv_common.py`:

* Add Kw constant: `KW_M2 = 1e-14` (M²).  Convert to nondim:
  `KW_NONDIM = KW_M2 * (1000.0 / C_SCALE) ** 2`
  (factor 1000 converts mol/L → mol/m³).
* Add D_OH: `D_OH = 5.273e-9` (m²/s, from CRC handbook).
* Add OH⁻ Bikerman radius: `A_OH_HAT = (4/3) · π · r_OH³ · N_A · C_SCALE`
  with `r_OH = 1.76e-10` m (Marcus crystallographic).
* Add `WATER_IONIZATION_DEFAULT` entry mirroring
  `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`:
  ```python
  WATER_IONIZATION_DEFAULT: Dict[str, Any] = {
      "z": -1,
      "c_bulk_nondim": KW_NONDIM / C_HP_HAT,
      "phi_clamp": 50.0,
      "steric_mode": "bikerman",
      "a_nondim": A_OH_HAT,
      "label": "OH-",
      "kind": "water_ionization",   # NEW: distinguishes from inert counterions
      "Kw_nondim": KW_NONDIM,
      "slaved_to_species": 2,        # H+ index
  }
  ```

The new `kind` and `Kw_nondim` keys signal that this entry is
**slaved to the dynamic H+ field via mass-action**, not a free
inert counterion.  All current Bikerman entries default to
`kind="inert_counterion"`.

### Q2 — Algebraic c_OH wired into Poisson + Bikerman

`Forward/bv_solver/boltzmann.py`:

* Extend `add_boltzmann_counterion_residual` (and the
  `build_steric_boltzmann_expressions` path) to handle
  `kind="water_ionization"` entries.
* For these entries, the analytic concentration becomes
  `c_OH(y) = Kw · exp(−u_H(y))` instead of
  `c_OH(y) = c_OH_bulk · exp(−z · φ(y))`.
* Rest of the plumbing (Bikerman γ closure, Poisson source) is
  identical to the inert path.

`Forward/bv_solver/forms_logc.py` and `forms_logc_muh.py`:

* No changes needed if `boltzmann.py` exposes the right
  expressions — the residual builders already accept
  pre-built expression dicts.
* For muh: c_OH uses the H⁺ field via
  `c_H = exp(μ_H − em·z_H·φ)`, then `c_OH = Kw / c_H`.

### Q3 — IC support (debye_boltzmann initializer)

`Forward/bv_solver/forms_logc.py:_try_debye_boltzmann_ic`
and the muh sibling:

* The Picard outer loop (`picard_outer_loop_general`) currently
  treats counterions as inert; extend it to recognize
  `kind="water_ionization"` and use mass-action slaving in the
  θ closure: `θ_b += a_OH · Kw / c_H_bulk`.
* The matched-asymptotic profile (BKSA composite-ψ) needs to
  include OH⁻ in `gamma_psi` denominator and in `H_outer`-derived
  fields.  At equilibrium with H⁺ + OH⁻ and fast water, the
  gamma closure becomes:
  ```
  gamma(ψ) = 1 / [ 1 + a_H · H_outer · (e^{−ψ} − 1)
                     + a_Cs · c_Cs_bulk · (e^{−ψ} − 1)
                     + a_SO4 · c_SO4_bulk · (e^{2ψ} − 1)
                     + a_OH · Kw / H_outer · (e^{ψ} − 1) ]
  ```

### Q4 — Tests

New test file `tests/test_water_ionization_oh_minus.py`:

* **Unit (fast)**:
  * Bulk consistency: at c_H = c_H_bulk, c_OH = Kw / c_H_bulk
    matches the configured bulk OH⁻ concentration.
  * Charge density agreement: Poisson source picks up the
    expected −Kw·exp(−u_H) term.
  * Bikerman θ closure: OH⁻ contributes `a_OH · Kw · exp(−u_H)`
    to packing.
  * Config validation: `kind` field defaults sensibly; missing
    `Kw_nondim` for `kind="water_ionization"` entries fails fast.
* **Slow**:
  * Anchor at +0.55 V on the 3sp + multi-ion + OH⁻ stack converges.
  * MMS convergence: a manufactured solution with c_H · c_OH = Kw
    pointwise verifies the shortcut at order of accuracy.
  * Regression: with `WATER_IONIZATION_DEFAULT` *off*,
    `mangan_full_grid_csplus_so4.py` produces byte-identical
    output to the pre-Phase-6α run (snapshot).

### Q5 — Validation sweep

Re-run the L_eff transport sweep with water ionization enabled:

```bash
python -u scripts/studies/l_eff_transport_sweep_csplus_so4.py \
       --enable-water-ionization
python -u scripts/studies/score_l_eff_sweep.py
```

Acceptance:

1. **P3 PASSES**: `max_surface_pH < 9` at L_eff=16 µm.
2. **Plateau magnitude**: at L_eff=100 µm, plateau lifts toward
   the deck −0.18 mA/cm² (the fresh H⁺ source raises the supply
   ceiling).  Direction-of-change check, not a tight target.
3. **Peak emergence (bonus)**: a peak near +0.10 V may emerge if
   local-pH dynamics drive a kinetic crossover.  If yes → check
   whether magnitude matches the deck −0.40 mA/cm².

If P3 still fails → escalate to Option C
(proton-condition variable).

## 6. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Mass-action slaving's drop of OH⁻ transport is significant — surface pH stays > 9 even with water term | Escalate to Option C (proton-condition variable).  Don't add OH⁻ as a separate dynamic species first. |
| R2 | The new algebraic c_OH expression introduces Newton stiffness near the surface where c_H is tiny (Kw/c_H is large) | The same `u_clamp = 100` that protects the existing exp(u_H) reconstruction also bounds Kw·exp(-u_H).  Should be safe.  Add a `c_OH_clamp` if regression tests fail. |
| R3 | Bikerman θ closure with c_OH included pushes packing fraction over the saturation threshold near the cathode | Steric saturation is already in place for Cs⁺ + SO₄²⁻ + Bikerman counterions.  The OH⁻ contribution in physical units is small (~10⁻⁵ M peak vs Cs⁺ at 0.2 M), so should not dominate. |
| R4 | The IC's matched-asymptotic profile needs reworking to include OH⁻; if Picard doesn't converge, anchor build fails | Fallback: `linear_phi` IC for the new stack.  Document and ship; revisit IC once dynamics are validated. |
| R5 | Snapshot regression tests for the existing single-ion ClO4 path break because the new OH⁻ entry shifts the bulk pH self-consistency | Gate the entire water-ionization path behind a new `enable_water_ionization=False` flag; default off; existing tests untouched. |
| R6 | Fast-water Damköhler assumption breaks down at small L_eff where convection/diffusion timescales become comparable to k_r·c_H | Order-of-magnitude check in §4 shows Da ≫ 1 at all planned L_eff.  If validation P3 fails AND a separate transport-timescale anomaly appears, this is the tell. |

## 7. Out of scope (deferred)

* **Sulfate buffering** (HSO₄⁻ ⇌ SO₄²⁻ + H⁺, pKa ≈ 2): at pH 4
  it's almost fully dissociated, weak buffer.  Add only if
  validation P3 still doesn't pass after water ionization.
* **HSO₄⁻ as a separate dynamic species**: huge complication for
  marginal benefit; skip.
* **Cation-dependent selectivity factor** (Phase 6β): orthogonal
  to local-pH; address after Phase 6α lands.
* **OH⁻-form ORR pathway**: at pH 4 the acidic-form rate
  expressions dominate; alkaline-form correction is a Phase 6γ
  consideration.

## 8. Open questions for GPT-loop

1. **Damköhler analysis correctness**: is the "fast-water"
   limit (Da ≫ 1) really uniformly justified across the
   L_eff sweep range?  Specifically at L_eff = 16 µm where
   transport timescales are short, does k_r · c_H_typ still
   dominate?
2. **OH⁻ transport drop validity**: is the second-order
   correction from OH⁻ flux divergence (Option B vs Option C)
   actually small, or does it matter for surface pH?
3. **Bikerman θ closure**: should OH⁻ enter steric packing
   (Marcus radius) or just enter Poisson?  The OH⁻ molar volume
   is small but the Bikerman path is already wired up — what
   are the gotchas?
4. **Proton-condition fall-back**: if Option B fails, is the
   proton-condition variable (C) the right next step, or should
   we go directly to full dynamic OH⁻ (D)?  What's the
   cost-vs-fidelity tradeoff?
5. **MMS verification**: what's the right manufactured solution
   for the new water-ionization term?  c_H · c_OH = Kw pointwise
   plus a smooth bulk profile is the obvious choice — anything
   GPT thinks we should add?
6. **IC robustness**: the `debye_boltzmann` IC's BKSA
   composite-ψ closure currently assumes a single-counterion
   electroneutrality (or multi-counterion shared-θ).  Adding a
   c_H-slaved species — is the composite-ψ derivation still
   valid, or does the IC need a fundamental rework?
7. **Sanity check on the magnitude**: the surface pH should drop
   from ~14 to ~7-9 with water ionization.  Order-of-magnitude:
   c_H_surf needs to rise from 10⁻⁹ M to ~10⁻⁵ M.  Water source
   rate must supply ~(10⁻⁵ − 10⁻⁹) mol/m³ × J_BV / F per unit
   time.  Does the Kw / c_H slaving deliver enough source?
   (This is the main physics check.)
```

## 3. Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
