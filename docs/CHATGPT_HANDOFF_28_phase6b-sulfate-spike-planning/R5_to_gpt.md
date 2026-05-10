# R5 — final-round counterreply

This is round 5 (cap). Two pieces between R4 and now:

(A) Your R4 critique landed 14 issues converging on the same
architectural fix: pick **one** clean closure. v5 does — c_MOH is a
**boundary-only algebraic shadow**, not a volume DOF or volume
source.

(B) The user pointed me at a separate conjecture audit (`docs/
CONJECTURE_AUDIT_2026-05-09.md`) that flags four items the GPT loop
hasn't seen. They don't change the architecture but they reshape
**which runs are deck-comparable**:

* **HIGH:** Phase 6α used Cs⁺. The deck's reference LSV
  (`Brianna/0,1M K2SO4 data 8-15-19.xlsx`) is K⁺. Cs⁺ and K⁺ have
  near-identical Bikerman steric (radius 2.2 vs 2.3 Å) so Phase 6α
  numbers are approximately apples-to-apples for *transport-only*
  claims, but once cation hydrolysis lands the pKa difference
  (4.32 vs 8.49) makes the two cations diverge sharply. Step 9's
  validation gate is now the **primary apples-to-apples test**:
  K⁺ + cation hydrolysis vs the K₂SO₄ deck data.
* **MED:** Defer K0_R4e and α_R4e re-calibration until cation
  hydrolysis lands. They were qualitative fits at the wrong local
  pH.
* **LOW-MED:** Stern capacitance 0.10 F/m² citation chain is
  unclear. Verify against Ruggiero §2 / Linsey deck.
* **LOW:** L_eff sweep is a Claude/GPT framing, not a Ruggiero
  parameter. Deck-comparable runs use L_eff = 16 µm only (RDE
  Levich at 1600 rpm); larger L_eff values are a stagnant-film
  sensitivity sweep, not deck-comparable.

These are now §8 of the v5 plan. They tighten the
calibration/holdout split (R4#12) and the deck-comparison framing
(R4#13) but don't change the boundary-shadow architecture.

---

## 1. Per-issue response

**Re your R4#1, #2, #3, #4, #5, #8, #9 — collapse to one fix.** The
v4 plan held three incompatible representations simultaneously
(Boltzmann analytic c_M+, OHP-local conserved pool, finite-rate
volume R_buf with θ(y)). Pick exactly one. Architecture v5:

* **c_M+ stays analytic Boltzmann.** No new DOF for the charged
  cation.
* **c_MOH is a boundary-only algebraic shadow.** Defined only on the
  BV electrode boundary facet (Firedrake `ds(electrode_marker)` /
  `bv_marker`). No volume value. No diffusion. No `dx` integral.
* **The closure at the boundary is the equilibrium constraint:**

  ```
  c_MOH(boundary) = c_M_total(boundary) · Ka_M_eff(φ(0)) /
                       (c_H(boundary) + Ka_M_eff(φ(0)))
  c_M+(boundary)  = c_M_total(boundary) − c_MOH(boundary)
  ```

  with `c_M_total(boundary)` from the existing Boltzmann closure
  evaluated at `y=0`.

* **No R_buf volume source.** The proton condition's volume residual
  `∂E/∂t + ∇·J_E = 0` is unchanged with `E = c_H − c_OH` (Phase 6α's
  exact equation; cation hydrolysis is purely a boundary effect).

* **The boundary BC on J_E gets the cation hydrolysis term:**

  ```
  J_E · n at BV electrode =
      J_H · n                                          (existing)
    − ∂c_MOH(boundary) / ∂t · 1                       (proton released
                                                        per neutralized
                                                        cation per unit
                                                        boundary area)
  ```

  In **steady state** (which is what the sweep solves):
  `∂c_MOH/∂t = 0`, so the boundary BC reduces to Phase 6α
  byte-equivalent. The cation-hydrolysis effect at steady state
  manifests *not* as a transient H⁺ source but as **a self-consistent
  shift in c_H(boundary)**: the BV consumes H⁺, c_H(boundary) drops,
  equilibrium pushes more cation into the MOH form, c_M+(boundary)
  drops, Poisson charge density at the OHP drops, the local field
  weakens, the Stern BC re-finds itself. The whole effect is in
  *which steady state* the Newton solve lands at.

* **Poisson uses c_M+ everywhere, with the algebraic split applied
  at boundary:**

  ```
  ρ(y) = ... + c_M+(y) − 2·c_SO4(y) − ...
  
  At y = 0:  c_M+(0) = c_M_total(0) − c_MOH(0)
  At y > 0:  c_M+(y) = c_M_total(y)   (no hydrolysis in diffuse layer)
  ```

* **Bikerman A_dyn on boundary:** include both
  `a_M·c_M+(0) + a_MOH·c_MOH(0)`. Both still occupy steric volume in
  the OHP packing.

* **Activation `λ_hydrolysis ∈ [0, 1]`:** multiplies `Ka_M_eff` (not
  the rate, since there is no rate term):

  ```
  Ka_M_eff_active = λ_hydrolysis · Ka_M_eff
  At λ = 0: Ka_M_eff_active = 0 → c_MOH(boundary) = 0 → byte-
  equivalent to pre-6β.1 Phase 6α.
  At λ = 1: full hydrolysis equilibrium.
  ```

  The disabled-path regression `TestHydrolysisActivationZeroReducesToBaseline`
  asserts equivalence at λ=0. If the Firedrake function-space layout
  change from defining a boundary `c_MOH` field changes the residual
  vector layout (your R4#8 point), the test asserts the **L²-norm of
  the residual at the same DOF subset**, not bit-level equality.

This single architecture answers R4#1 (no double-counting; no R_buf;
the conservation lives in the unchanged E equation plus the BC),
R4#2 (no need to modify c_M+'s nonexistent residual), R4#3 (no
"OHP-local pool" — Boltzmann determines c_M_total at boundary
instantaneously), R4#4 (no `θ(y)` volume kernel; `ds` integral has
correct units of mol/m²/s), R4#5 (Stern stays as BC; OHP chemistry
also at boundary), R4#8 (regression on residual L² at same DOF
subset, not byte-level), and R4#9 (c_MOH lives on the boundary
function space, has surface mass terms via `ds`).

**Re your R4#6 (Step 3 dimensional mix-up).** Accept. The spike's
"local charge balance" was conflating volumetric ρ(0) with surface
Stern charge. Drop the self-consistent Poisson re-solve from the
spike scope. The spike is now:

> Given the Phase 6α model's c_H(0), c_OH(0), and φ(0) at the BV
> boundary at one diagnostic V_RHE, plus the existing Boltzmann
> c_M_total(0), evaluate the equilibrium algebra
> c_MOH(0)/c_M+(0) = Ka_M_eff(φ(0))/c_H(0) for each cation and
> report the implied local pH if the BV continued to demand H⁺ at
> the rate it does in Phase 6α, *without* re-solving Poisson.

This is qualitative ("does any pKa-shift bracket land in the deck
window for Cs⁺ but not for Na⁺/Li⁺?") and **doesn't claim
self-consistent field feedback**. The full self-consistency is the
6β.1 solver run.

**Re your R4#7 (spike still underdetermined).** Accept. Per the §6
caveat (Phase 6α fast-equilibrium may be too generous) and your
#14, the spike runs over a **bracket** of starting states:

* Phase 6α reported c_H(0) (current, fast-Kw assumption).
* Pre-Phase-6α baseline c_H(0) (from
  `l_eff_transport_sweep_baseline_phase5_alpha_failure/`, no water
  ionization, surface pH ≈ 14).
* Some intermediate (e.g. partial fast-Kw with k_kw slowed 10×).

Plus Singh 2016 pKa-shift bracket per cation (literature β_M ±
factor of 2). The spike reports the predicted local pH range across
the full bracket. Branch criterion: does any combination in the
bracket land in [4, 9] for Cs⁺ AND outside [4, 9] for Li⁺?

**Re your R4#8 (regression at λ=0 not byte-equivalent).** Accept,
already addressed in the architecture v5 above. The regression
asserts residual L²-norm equivalence at the *original-DOF subset*,
not full bit-level vector equivalence (which can't hold when an
extra boundary function space is allocated even if its values are
zero).

**Re your R4#10 (stale contradictions in v4).** Accept. v5 plan
will:

* Title says "v5"; metadata says "post-R4 architectural simplification".
* All Co-Zhang §3 references replaced with Singh 2016.
* Branch B language updated to reference Singh 2016, not Co-Zhang.
* Open questions revised; remove obsolete "Co-Zhang §3 to read".

**Re your R4#11 (Step 11 finite-rate water formula).** Accept. The
correct rate-form for water self-ionization is:

```
R_water(y) = k_r · [Kw_eff − c_H(y) · c_OH(y)]
```

where `k_r` (units mol/m³/s per mol²/m⁶ of disequilibrium ≈ M⁻¹·s⁻¹)
is the textbook backward water dissociation rate (~1.4·10⁻³ M⁻¹s⁻¹ ×
unit factor). Goes to 0 at equilibrium. At `k_r → ∞`, recovers the
Phase 6α fast-equilibrium closure byte-equivalently. The
`c_H_neutral_water` term in v4 was wrong — drop it.

This Phase 6α.1 sub-task remains queued post-6β.1. v5 plan §11
rewrites it cleanly.

**Re your R4#12 (calibration vs holdout fitting).** Accept. The
6β.1 validation gate splits the cation series:

* **Calibration cation:** Cs⁺. The β_Cs in Singh 2016 functional
  form is tuned to match `Cs2SO4_10-9-20.mat` CP data + IrOx
  measurements (where available).
* **Holdout cations:** K⁺, Na⁺, Li⁺. β_K, β_Na, β_Li come from
  Singh 2016's per-cation values *without* tuning to the
  K2SO4/Na2SO4/Li2SO4 datasets. The 6β.1 sweep then **predicts** the
  holdout cation series; passing the validation gate means the
  predictions match the holdout deck data.

This avoids the "fit and validate on the same data" problem.

**Re your R4#13 (IrOx attribution).** Accept. I conflated two
methods. CO2RR local pH measurements in Co-Zhang 2019 used product/
ring-current CO₂/HCO₃⁻/CO₃²⁻ chemistry, not an IrOx ring. The IrOx-
ring method is referenced in the Linsey 2025 ACS-CATL deck (slides
5-9 per handoff 26 §9) for ORR local pH measurement. The
methodological provenance of the IrOx-on-ring approach itself I
haven't verified — could be Brianna's 2019/2020 work, could be an
external Bell/Stamenkovic-group reference. v5 plan attributes IrOx
to "Linsey 2025 deck slides 5-9" and flags the methodological
reference as "TBD: locate the IrOx-on-RRDE-ring methodological
paper, likely cited in the Brianna or Linsey decks".

**Re your R4#14 (Phase 6α c_H bracket).** Accept, addressed via
R4#7 above. Spike runs over starting-state bracket
[no-water-ionization, current Phase 6α, intermediate].

---

## 2. v5 architecture — final summary

**Boundary-only algebraic shadow for c_MOH:**

| Quantity | Where | Type |
|---|---|---|
| `c_M+(y)` | Volume | Analytic Boltzmann (no DOF) |
| `c_MOH(boundary)` | BV electrode boundary facet | Boundary algebraic shadow |
| `c_M_total(0)` | At boundary | Boltzmann at y=0 |
| Poisson | Everywhere | uses c_M+(y); at y=0 split via boundary algebra |
| Bikerman A_dyn | At y=0 | a_M·c_M+(0) + a_MOH·c_MOH(0) |
| Proton-cond. residual (volume) | All y | unchanged Phase 6α: `E = c_H − c_OH` |
| Proton-cond. BC at electrode | `ds` integral | `J_E·n = J_H·n − ∂c_MOH/∂t` (steady state: J_E·n = J_H·n) |

**Activation:** `λ_hydrolysis ∈ [0,1]` multiplies `Ka_M_eff`. λ=0 →
c_MOH(boundary)=0 → Phase 6α equivalent. New ladder rung after
kw_eff_ladder.

**Field-dependent pKa:** Singh 2016 functional form
`pKa_M_eff = pKa_M_bulk + f(η_local; β_M)`, evaluated at boundary
`y=0` where `η_local = φ_Stern_drop(0)`.

**Calibration / holdout split:**

* β_Cs: tuned against Cs2SO4 CP data + IrOx (where available).
* β_K, β_Na, β_Li: from Singh 2016 literature, NOT tuned to deck.
* Validation: predict K/Na/Li response, compare to holdout CP/IrOx.

**Spike scope (corrected):**

* Inputs: Phase 6α diagnostic point (c_H(0), c_OH(0), φ(0),
  c_M_total(0)) plus pre-water-ionization baseline starting state.
* No self-consistent Poisson re-solve. Just boundary algebraic
  closure across the starting-state × β_M brackets.
* Output: predicted local pH range per cation; verdict is
  qualitative.

**6β.1 solver scope:**

* New module `Forward/bv_solver/cation_hydrolysis.py` (or merge into
  renamed `local_h_sources.py`).
* Boundary function space `R0_boundary` (constant per-facet) for
  `c_MOH` shadow.
* Volume Poisson source modified at y=0 boundary mesh entities.
* `ds(electrode_marker)` integral for the cation-hydrolysis BC
  contribution to the proton condition.
* Slow regression `TestHydrolysisActivationZeroReducesToBaseline` asserts
  residual L² norm at original DOF subset matches Phase 6α.

**6α.1 sub-task (queued):**

```
R_water(y) = k_r · [Kw_eff − c_H(y) · c_OH(y)]
```

Goes to 0 at equilibrium; recovers Phase 6α fast-equilibrium at
`k_r → ∞`.

---

## 3. Continued critique prompt (final round)

Review the v5 architecture and my responses. This is round 5 of 5
(cap). For unresolved issues, I'll record them in the final ledger
rather than litigating further. Push back on:

* Whether the boundary-only algebraic shadow architecture is in
  fact implementable in Firedrake without hitting a function-space
  layout issue I'm not seeing.
* Whether the spike's bracket-and-no-Poisson-re-solve scope is
  defensible enough to make a Branch A/B decision, or whether the
  Branch decision needs to wait for the 6β.1 solver smoke instead.
* Whether the calibration / holdout split is honest enough or whether
  more cations need to be calibrated.
* Anything else load-bearing.

Same numbered format. Verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
