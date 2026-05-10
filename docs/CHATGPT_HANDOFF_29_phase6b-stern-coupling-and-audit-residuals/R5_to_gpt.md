# R5 — final-round counterreply (v9 as staged-investigation plan + ledger)

Round 5 of 5 (cap). Round 4's #12 closing observation is the right
framing: v9 is **a staged investigation plan, not an approved
architecture**. Three load-bearing physics items remain unresolved
and depend on either literature lookup or Gate 4 empirical evidence.
This round accepts all 12 R4 issues, commits the staged plan + ledger
to the v9 artifact, and exits.

## 1. Per-issue response

**Re #1 (k_des dimension wrong, not literature-anchored).** Accept.
v9 §7: `k_des [1/s]`, **labeled phenomenological**. No Bohra
citation for MOH desorption (Bohra 2019 doesn't give it). Sweep
range `k_des ∈ {1e3, 1e5, 1e7} 1/s` at Gate 4 (per #3).

**Re #2 (open-reservoir, not full mass-conserving).** Accept.
v9 §7 explicit caveat: "Bulk electrolyte is treated as an infinite
reservoir for desorbed neutral MOH⁰. Full cation mass conservation
requires tracking c_MOH(y) as a 5th species; deferred to a later
phase if the smoke validates the architecture." Adds to unresolved
ledger.

**Re #3 (K_s alone insufficient with k_des).** Accept. Gate 4
calibration extends to **two grouped parameters**: `K_s = δ ·
Ka_eff` and `Da = k_des · δ / D_M_OHP_supply` (a Damköhler-style
ratio of desorption rate to OHP supply rate). Both varied at Gate 4;
sensitivity-only (no fitting) at the smoke V_RHE.

**Re #4 (Singh formula guessed, missing ln(10), sign suspect).**
Accept. **v9 §7 does NOT implement a guessed formula.** Status:
* Singh 2016 main text only states "dependence on cation
  charge/size and cathode charge density" — qualitative.
* Exact functional form requires Singh SI — **must extract before
  Gate 4 implementation**.
* Until then v9 places the pKa shift in the unresolved-physics
  ledger; Gate 4 uses a placeholder shift `ΔpKa = β_M · sgn(σ_S) ·
  |σ_S|^p` with p ∈ {0.5, 1.0, 2.0} sensitivity, β_M and p both
  unphysical-by-fiat. The smoke is therefore **falsification-
  oriented** in the Singh-formula sense: if no choice of (β_M, p,
  k_des, K_s) brings predicted-vs-realized Δ ln R within 30%, the
  architecture is wrong; if some combination does, that's a
  necessary-but-not-sufficient pass.

**Re #5 (σ_S including Γ correction risks circular feedback).**
Accept. Gate 4 includes a branch diagnostic:
* Run pKa-driven by **bare** σ_S (no Γ correction): diagnostic A.
* Run pKa-driven by **corrected** σ_S = stern + F·Γ: diagnostic B.
* If diagnostics yield qualitatively different Γ_MOH (e.g. order-of-
  magnitude difference), the corrected σ_S has a positive-feedback
  branch — fail the gate.
* Add packing-near-1 detection (already in v9 §7); if packing > 0.95
  at any (V, λ), fail the gate.

**Re #6 (Stern correction may double-count electrostatics).**
Accept. v9 §7 will derive a one-paragraph Gauss balance for the
electrode/OHP interface BEFORE Gate 4 implementation:

```
Surface charge balance at electrode:
  σ_metal     (metal-side, set by Stern BC C_S·ψ_S)
  + Γ_MOH     (neutral surface inventory — does NOT add to charge)
  + ∫_OHP_layer ρ_diffuse dy   (diffuse charge inside δ_OHP)
                              = 0  (Gauss)

If c_K+(y) is dynamic NP, then ∫ ρ_diffuse dy already accounts for
the K+ depletion at the OHP. The neutral Γ_MOH contributes 0 to
the Stern σ-balance. The "+F·Γ_hat" Stern correction in v8/v9 is
WRONG (R4#6) — Γ is neutral, doesn't enter Stern σ.

The actual electrostatic effect of hydrolysis: depleted c_K+(0)
shifts ∫ρ_diffuse dy via the Boltzmann tail change, which already
shifts ψ_S via the dynamic-NP Poisson coupling. NO additional
Stern correction needed.
```

**This kills the v6/v7/v8 Stern surface-charge correction
entirely.** v9 §7 removes it. The hydrolysis effect is purely:
* H+ source via R_net = k_des · Γ_MOH (proton boundary flux).
* M+ sink via the same R_net (cation NP flux).
* Indirect electrostatic shift via the dynamic K+ NP profile
  responding to the boundary sink.

This is a *cleaner* architecture than v6/v7/v8 attempted — the
Stern correction was always conceptually muddled. Round 4 #6 is
the load-bearing fix that makes v9 actually defensible.

**Re #7 (Gate 2 dynamic K+ feasibility).** Accept. v9 §7 documents
expected continuation fallbacks:
* z-ramp: start with z=0.1 K+ Boltzmann at λ_z=0, ramp z→1.
* k0 ramp: start with k0_R4e/R2e at low values.
* C_S ramp: start with C_S = 1.0 (effectively no Stern), ramp to
  0.10.
* Warm-start from less-cathodic V_RHE (e.g. -0.1 → -0.4).
* Decimal voltage refinement on the V_RHE grid.

Gate 2 failure is "expected, not exceptional" per #7. v9 §9 lists
this as a feasibility risk that may delay 6β.1 by weeks.

**Re #8 (h_index refactor scope).** Accept. v9 prereq audit:
* `Forward/bv_solver/forms_logc_muh.py:_resolve_mu_h_index`
* `Forward/bv_solver/water_ionization.py:79`
* `Forward/bv_solver/picard_ic.py` (debye/Boltzmann IC paths)
* `Forward/bv_solver/multi_ion.py` (multi-ion charge inference)
* All `z=+1` greps in IC, diagnostics, post-processing.
* New test: 4-species K2SO4-with-H stack runs cleanly; H+ uses μ_H,
  K+ does not.

Audit is the first 1-2 days of Gate 1.

**Re #9 (Constant ≠ Newton unknown).** Accept. Γ is an actual
mixed-space unknown via Real element. Constant only used in
manufactured-source unit tests.

**Re #10 (Γ residual area normalization).** Accept. v9 §7 explicit
form:

```
F_res += G_test * (
    (Γ - Γ_old) / dt_nondim
    − (R_net_hat - k_des_hat * Γ)
) * ds(electrode_marker)

# In steady-state, dt_nondim → ∞ so the time-derivative drops:
F_res += G_test * (- (R_net_hat - k_des_hat * Γ)) * ds(electrode_marker)

# Real element ⇒ G_test = G_test_constant; ds gives the electrode
# area as the test-function weighting. Verified by an
# area-doubling test (refine the mesh; Γ value should be invariant
# to electrode area).
```

**Re #11 (λ=0 hard-zero pin).** Accept. v9 §7: at λ=0, replace the
Γ residual with a Dirichlet-like algebraic constraint `F_res =
G_test * Γ * ds(electrode_marker)` (forces Γ = 0 exactly). Re-enable
hydrolysis residual at λ > 0 via continuation. Stern correction is
already dropped in #6; packing diagnostic is automatic.

**Re #12 (v9 staged investigation plan, not approved architecture).**
Accept. This is the right framing. v9 explicit unresolved-physics
ledger:

```
# v9 unresolved-physics ledger (must be closed during 6β.1, not before)

L1. Singh 2016 SI exact pKa-shift formula
    Status: TBD; placeholder β_M·sgn(σ_S)·|σ_S|^p with sensitivity
    Decision needed: read Singh SI; extract exact functional form
    Closure target: Gate 4 + cation series

L2. k_des desorption rate prior
    Status: phenomenological, no literature anchor
    Decision needed: literature search for MOH⁰ desorption from
                     polarized OHP; otherwise full-range sweep
    Closure target: Gate 4 sensitivity sweep

L3. Γ-Stern double-counting
    Status: REMOVED in v9 per R4#6; the only electrostatic effect
            is via dynamic K+ Poisson coupling. Σ-correction was
            an artifact of v6/v7/v8 conceptual muddle.
    Decision needed: Gauss-balance proof in §7 (write before Gate 4)
    Closure target: §7 derivation pre-Gate 4

L4. Open-reservoir desorption (not full mass conservation)
    Status: documented caveat
    Decision needed: track c_MOH(y) as 5th species if any post-
                     6β.1 evidence requires it
    Closure target: 6β.2 or later

L5. Dynamic K+ cathodic convergence (Gate 2 feasibility)
    Status: known risk; continuation fallbacks documented in §7
    Decision needed: Gate 2 outcome
    Closure target: Gate 2

L6. Cation-series transferability (β_M, p, K_s, Da)
    Status: deferred to 6β.2 cation-series validation
    Decision needed: 6β.2 holdout pass/fail
    Closure target: 6β.2

L7. Stern capacitance C_S = 0.10 F/m² (carry-forward from R1
    audit)
    Status: labelled tunable [0.05, 0.50] F/m²; sensitivity only
    Decision needed: deferred to 6β.2 calibration
    Closure target: 6β.2

L8. SO₄²⁻ Bikerman radius 2.4 Å (Marcus textbook prior)
    Status: documented carry-forward
    Decision needed: cross-check Linsey deck slide 13 anion radii
                     if/when delivered
    Closure target: post-6β.1 (low priority)

L9. Tafel slope xlsx (external)
    Status: blocked on data delivery
    Decision needed: re-request from Linsey/Brianna
    Closure target: 6β.2 K0_R4e + α_R4e calibration
```

Gate 4 is **falsification-oriented**: designed to expose any of
L1/L2/L3/L5 if the underlying physics is wrong. If Gate 4 fails,
re-queue a GPT round with the failure data as new evidence. If
Gate 4 passes within tolerance, that's necessary-but-not-sufficient
— 6β.2 is the actual physics validation.

---

## 2. v9 final summary (consolidated)

The v6 plan I started this loop with had three unresolved
architectural items. After 5 rounds we've established:

* **The boundary-only algebraic shadow architecture (v5/v6) is
  structurally impossible.** Algebraic equilibrium gives zero net
  steady-state proton release.
* **The Stern surface-charge correction (v6/v7/v8) was conceptually
  muddled.** Γ_MOH is neutral; doesn't enter Stern σ. The Gauss
  balance shows the hydrolysis electrostatic effect is purely
  via the dynamic-cation Poisson coupling, not a separate Stern
  σ correction.
* **The principled architecture (v9) requires:**
  - c_K+ promoted to dynamic NP species (4-DOF stack).
  - Γ_MOH as a global Real-element scalar with finite-rate
    hydrolysis kinetics.
  - Desorption R_des = k_des · Γ as the only steady-state
    turnover-allowing path (open-reservoir).
  - h_index refactor as prereq (logc_muh + water_ionization +
    IC paths must accept multiple z=+1 species).
  - 4-gate phased smoke: build / λ=0 dyn / Γ source unit test /
    finite hydrolysis at one V_RHE.
* **9-item unresolved-physics ledger** for in-execution determination.
* **Singh pKa formula is TBD-on-SI-extraction**, not implementable
  from the current handoff — gates 1-3 don't need it; gate 4
  needs it.

This is honestly scoped. The architecture is now the right *next
investigation*, not a finished design.

---

## 3. Final-round prompt

This is round 5 of 5. For unresolved issues, the ledger above is
the durable record. If you APPROVE the staged investigation plan
+ ledger framing, that exits the loop on a clean basis. If you
ISSUE_REMAIN, we record any remaining items as durable unresolved
and exit on cap.

Specifically, push back if:
1. The ledger is missing a load-bearing item.
2. The Stern-correction-removed framing (R4#6 derivation) is
   wrong; should the Stern correction stay in v9?
3. Any of the 4 gates is mis-scoped.
4. The Singh SI extraction prereq blocks Gate 4 in a way I'm
   under-stating.
5. Anything else.

Same numbered format. Verdict line at the end:
  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

If issues remain at round 5 they go to the unresolved ledger
verbatim and the loop exits.
