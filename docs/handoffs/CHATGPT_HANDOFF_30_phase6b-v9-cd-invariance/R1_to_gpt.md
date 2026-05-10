# R1 — Phase 6β v9 Phase A + B finding handoff

## Section 1: Context bundle

### Goal of Phase 6β
Reproduce the Linsey 2025 deck slide-27 cation pH-shift trend
(Cs⁺ ≪ K⁺ ≪ Na⁺ ≪ Li⁺ in surface pKa near the cathode) inside the
`PNPInverse` forward solver.  Mechanism in scope: Singh 2016 SI Eq.
(2)–(4) cation-hydrolysis at the OHP, where ``r_M(σ_S)`` causes
ΔpKa shifts that release H⁺ near the surface (or absorb H⁺ at
anodic bias).  Production target: 11-point V_RHE IV curve at λ=1
that reproduces the deck shape per cation, calibrated on K⁺, with
Cs/Na/Li as predictive holdouts.

### Stack (production-target, deck-aligned, 4 dynamic species)

* `formulation='logc_muh'`, `log_rate=True`, `u_clamp=100`,
  `exponent_clip=100`.
* Species (4): O₂ (idx 0), H₂O₂ (idx 1), H⁺ (idx 2), K⁺ (idx 3).
  K⁺ is dynamic NP (so the cation-hydrolysis residual can read
  ``c_K(0)`` at the OHP) — **not** analytic Bikerman.
* Analytic Bikerman counterion: SO₄²⁻ at C_SCALE-nondim = 83.33,
  ``a_nondim = 4.20e-5``.
* Reactions: parallel 2e⁻ (E°=0.695 V vs RHE) + 4e⁻ (E°=1.23 V vs
  RHE) per Ruggiero 2022 §1.
* Stern: ``stern_capacitance_f_m2 = 0.10`` F/m² (Cu prior; not
  deck-cited, sensitivity sweep treats as tunable).
* IC: `linear_phi`.
* L_eff: **16 µm boundary layer** (per gate4 driver; CLAUDE.md
  documents 100 µm as the broader production setting; gate4 used
  16 µm to ease convergence at high anodic V — this difference is
  load-bearing for the finding below).

### Cation-hydrolysis residual (Phase 6β v9 Gate 3 + 4 architecture)

Forward (Singh 2016 SI):

    R_net  =  k_hyd · c_M(0) · 10^(−ΔpKa(σ_S))
              −  k_prot · c_H(0) · Γ / δ_OHP_HAT                 (1)

with field-dependent ΔpKa per Singh Eq. (4):

    ΔpKa(σ)  =  +2 · A · z_eff · σ · r_H_El · (1 − r_M-O² / r_H_El²)   (2)

where σ_S is the surface-charge density (counts/pm²); A=620.32 pm,
z_eff(K⁺)=0.919, r_M-O = r_M + 63 pm = 138+63=201 pm,
r_H_El = 200.98 pm (Cu prior, also Phase D's calibration knob).

Outer Picard for Γ_MOH: at every continuation rung the orchestrator
solves

    Γ_ss(λ)  =  λ · k_hyd · ⟨c_M · 10^(−ΔpKa)⟩
                / (λ · k_des + (1 − λ) + λ · k_prot · ⟨c_H⟩ / δ_OHP)    (3)

and assigns it to the R-space coefficient ``bundle.gamma_func``.
Newton then solves the FE residual with Γ pinned.  Picard tolerance
1e-4 relative, max 8 iters per rung.

**Coupling to bulk equations:**
* Proton boundary residual gains a SOURCE ``+ λ · R_net``.
* Cation (K⁺) boundary residual gains a SINK ``− λ · R_net``.
* Γ does NOT enter the bulk Bikerman packing; it's only a boundary
  coefficient (the Bikerman packing is over O₂/H₂O₂/H⁺/K⁺ +
  analytic SO₄²⁻).

### Continuation orchestration (`Forward/bv_solver/anchor_continuation.py`)

* `solve_anchor_with_continuation` walks k0 ladder
  (1e-12 → 1e-9 → 1e-6 → 1e-3 → 1.0) at λ=0 + the optional kw_eff
  ladder (water self-ionization) for the anchor point V=+0.55 V.
* `extract_preconverged_anchor` / `solve_grid_with_anchor` warm-walks
  V_RHE points down to the target voltage at λ=0.
* `solve_lambda_ramp_from_warm_start` accepts a cached U snapshot at
  the target voltage, applies parameter overrides
  (k_hyd/k_prot/k_des/r_H_El/δ_OHP), reconverges SS at λ=0, then
  walks a λ ladder (default 0.0, 0.25, 0.50, 0.75, 1.00) with the
  outer Picard at every rung.
* Picard convergence check: ``|Γ_new − Γ_old| / max(|Γ_new|,
  |Γ_old|, 1e-30) < 1e-4`` for 1 iteration of agreement.

### Smoke baseline parameters

* `k_hyd = 1e-3` nondim (intentionally tame so Picard converges).
* `k_prot = 1e-3` nondim.
* `k_des = 1.0` nondim.
* `δ_OHP_HAT = 4e-6` (= 0.40 nm / L_REF=100 µm).
* `r_H_El_pm = 200.98` (Singh Cu prior).
* C_HP_HAT = 0.0833 (pH 4 bulk).
* C_KPLUS_HAT = 166.58 (199.9 mol/m³, electroneutrality).

### Gate 4B status (already landed before Phase A/B)

* 9-combination sensitivity sweep at V=−0.40 V completed in <45 s
  per combination.
* All 9 combinations converge and produce **identical cd =
  −5.532 mA/cm²** to ~6 sig figs.
* Picard hits the analytic Γ_ss formula exactly (Γ·k_des = const
  to 4 decimals across the k_des sweep at fixed k_hyd/k_prot) —
  validates the Picard machinery.

### What changed in Phases A + B (this round)

The `phase6b_v9_post_gate4_plan.md` v9 post-Gate-4 plan added
three flags to the gate4 driver and ran §A + §B per the plan's
decision tree.  **The plan's Risk Callouts §A and §B both fired
positive: cd plateau invariant + Phase B no-stable-window.**

### Test gap (load-bearing for the bug-vs-feature interpretation)

* `tests/test_phase6b_v9_gate4_finite_hydrolysis.py` only verifies
  λ=0 byte-equivalence to Gate 2 baseline (i.e. that disabling
  hydrolysis recovers the prior result).  **There is NO test
  verifying that λ=1 produces a measurable cd shift relative to
  λ=0 at any voltage.**  The cd-invariance finding could in
  principle be a residual-plumbing bug that this test gap would
  miss.

### Numerical sanity-check on R_net magnitude

Nondim R_net at k_hyd=1e-2, λ=1 (the boundary case before Picard
breaks):

    forward = k_hyd · c_K · 10^(−ΔpKa)
            ≈ 1e-2 · 166 · 10^(−ΔpKa)
    backward = k_prot · c_H · Γ / δ_OHP
            ≈ 1e-3 · 0.0833 · 3 / 4e-6  ≈ 62.5

Observed Γ ≈ 3 implies steady state ≈ forward / (k_des + backward/Γ),
so forward ≈ 3 · (1 + 62.5/3) ≈ 65.

That gives 10^(−ΔpKa) ≈ 65 / 1.66 ≈ 39 → ΔpKa ≈ −1.6.

Physical R_net (mol/m²/s):
``R_net_phys = R_net_nondim · C_SCALE · D_REF / L_REF``
``= 65 · 1.2 · 1e-9 / 1e-4 = 7.8e-4 mol/m²/s``

BV consumption rate at cd = −5.531 mA/cm²:
``flux_H = i / (n_e · F) = 55.3 A/m² / (2 · 96485) ≈ 2.9e-4 mol/m²/s``
(2e branch; 4e gives half this).

So **R_net source is comparable to (slightly above) BV H+
consumption**.  This is the contradiction we want GPT to interrogate:
how can R_net be the same order as BV consumption yet cd be
*invariant* across a 10× change in R_net (k_hyd 1e-3 → 1e-2)?

### Bikerman packing sanity-check

Bulk packing fraction (without MOH):

    a_K · c_K_HAT + a_SO4 · c_SO4_HAT
    = 4.27e-5 · 166.58 + 4.20e-5 · 83.33
    = 7.11e-3 + 3.50e-3
    ≈ 0.011 (NOT 0.99 — earlier comment in code mis-stated)

So the bulk is NOT packing-saturated.  Picard breakdown at k_hyd≥1e-1
must be Newton-stiffness, not Bikerman-saturation, contrary to
my initial hypothesis in the artifact.

### What we know about why Picard breaks at k_hyd ≥ 1e-1

At k_hyd = 1e-1, projected Γ at λ=0.25 would be ~0.25·forward/(k_des
+ k_prot·c_H/δ).  Forward = 0.1·166·10^(−ΔpKa).  At ΔpKa near the
σ_S=0 anchor value (~0), Γ projects to ~16/22 ≈ 0.7 at λ=0.25.  But
ΔpKa is field-dependent: at the cathodic V=−0.20 V the surface charge
σ_S is more negative → ΔpKa more negative → 10^(−ΔpKa) larger → Γ
balloons.  So Picard breaks because Γ jumps too far in the first
positive λ rung when forward's ΔpKa-dependent factor dominates.

### Key files

* Plan: `docs/phase6b_v9_post_gate4_plan.md`
* Phase A+B finding: `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`
* Phase A output JSON: `StudyResults/phase6b_v9_observability_v_minus_0_20/iv_curve.json`
* Phase B output JSON: `StudyResults/phase6b_v9_k_hyd_ramp/iv_curve.json`
* Cation hydrolysis impl: `Forward/bv_solver/cation_hydrolysis.py`
* Driver: `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`

---

## Section 2: The artifact under review

(See the parent file:
`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`.
The full text follows below for self-containment.)

````markdown
# Phase 6β v9 post-Gate-4 plan — Phases A + B results

## TL;DR

Phase A + Phase B together confirm the plan's worst-case
risk-callout: the **cd plateau is invariant** at V=−0.20 V (just like
at V=−0.40 V), and **Phase B finds no usable k_hyd window** —
between `k_hyd ≤ 1e-2` (cd unmoved) and `k_hyd ≥ 1e-1` (Picard
breaks at the first positive λ rung), there is no value where
hydrolysis produces a measurable cd shift while the architecture
remains numerically tractable. This is the **v9 R5#5 wording-guard
outcome**: architecture works, expresses a plausible branch, but
the branch has no measurable cd impact at the production parameter
set.

## Phase A — observability at V=−0.20 V

cd_observable at V=−0.20 V across λ ladder:

| λ | cd_observable | Γ |
|---|---|---|
| 0.25 | 12.572849749522968 | 0.07639157771401214 |
| 0.50 | 12.572851628193117 | 0.15277664154520300 |
| 0.75 | 12.572853502603921 | 0.22915519131001685 |
| 1.00 | 12.572855372992420 | 0.30552722681905870 |

Δcd from λ=0.25 → λ=1.00: 5.62e-6 (4.47e-7 relative).  Final cd
mA/cm²: −5.531 (same H⁺ Levich floor as Gate 4B at V=−0.40 V).

## Phase B — k_hyd ramp at V=−0.20 V

| k_hyd_nondim | converged | cd (mA/cm²) | pc (mA/cm²) | Γ_MOH | error |
|---|---|---|---|---|---|
| 1e-3 | True  | −5.531 | 2.766 | 0.306 | — |
| 1e-2 | True  | −5.531 | 2.766 | 3.051 | — |
| 1e-1 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+0 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+1 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+2 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+3 | False | —      | —     | —     | LadderExhausted at λ=0.25 |

cd is invariant across all converged rungs (−5.531 mA/cm² at
k_hyd=1e-3 AND 1e-2; cd_observable identical to 6 sig figs).  Γ
scales linearly with k_hyd (0.306 → 3.051; exactly 10×).  Picard
breaks at k_hyd ≥ 1e-1 at the first positive λ rung.

## Mechanism interpretation (artifact's claim)

The H⁺ Levich floor at the cathode is rate-limiting for cd.  Cation
hydrolysis at the OHP adds local H⁺, but BV consumes it as fast as
produced; bulk-side H⁺ supply through the boundary layer
(L_eff = 16 µm) pins the cd value at −5.531 mA/cm².

## Decision points (artifact's branches)

1. Mitigation A: thinner L_eff to escape the Levich floor.  Risk:
   deviates from the deck's RRDE rotation rate.
2. Mitigation B: Anderson acceleration on Picard to unblock
   k_hyd ≥ 1e-1.  Doesn't address cd-invariance at converged rungs.
3. Re-queue GPT round (current path).
````

---

## Section 3: Critique prompt

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

### Specific questions to ground the critique

In particular, please push hard on:

1. **Is cd-invariance a transport-saturation finding or a residual
   plumbing bug?**  The numerical sanity-check above shows R_net is
   the same order of magnitude as BV H+ consumption, yet cd doesn't
   move across a 10× change in k_hyd.  Possibilities:
   (a) Levich-floor argument is correct and the cd shift is just
       too small to see at the cd_observable form's precision (in
       which case: what observable WOULD show the effect?).
   (b) R_net source is being added to the proton boundary residual
       but the residual is dominated by something else that
       absorbs the source (electromigration of K+? The
       analytic-counterion residual closure?).
   (c) The +λR_net source on c_H is being silently cancelled by
       the −λR_net sink on c_K when Bikerman couples them via the
       analytic counterion residual.
   (d) The cd_observable form reads c_H(0) but the change in c_H(0)
       is being absorbed by the Stern layer φ-redistribution rather
       than by BV.

   What experimental probes would distinguish these?

2. **Is "thinner L_eff" physically defensible as a mitigation?**
   The Levich equation gives δ ≈ 1.61 · D^(1/3) · ν^(1/6) · ω^(−1/2).
   For ω = 1600 rpm and standard aqueous values, δ ≈ ~17 µm — close
   to the gate4 driver's 16 µm.  CLAUDE.md mentions 100 µm as the
   broader production setting.  Which is right for the deck?  And
   if 16 µm is the deck-correct Levich layer, what does it mean to
   "shrink it"?

3. **Reframing of Phase 6β scope.**  The deck shows cation pH-shift
   ordering (Cs⁺ < K⁺ < Na⁺ < Li⁺ in surface pKa) — this is a
   surface-charge / surface-pH effect, not necessarily a cd
   ordering.  The model could in principle reproduce the surface-pH
   ordering even if cd is invariant.  Is the right deliverable for
   6β.2 actually a *surface pH* prediction rather than a cd IV
   curve?  What would change in the Phase C/D/E plan if we shifted
   the observable?

4. **Test gap.**  The Gate 4 test only checks λ=0 byte-equivalence.
   There is no regression test that λ=1 moves cd.  Should we add
   a "λ ramp moves cd at synthetic k_hyd" test before continuing?
   If so, what's the right synthetic regime — small enough that
   Picard converges but large enough that cd shifts at least 1%?
   Does our k_hyd ≤ 1e-2 result bracket that regime, or is the
   architecture incapable of producing such a regime under any
   parameters?

5. **Picard breakdown at k_hyd=1e-1.**  Earlier I claimed it was
   Bikerman packing saturation; the corrected sanity-check shows
   bulk packing is only 0.011 so it's actually Newton-stiffness
   from Γ blowing up via the ΔpKa(σ_S) factor.  Is that the right
   diagnosis?  What numerical fixes (line search damping, Anderson
   acceleration, inner Newton on Γ) are most likely to unblock
   k_hyd ≥ 1e-1, and does that unblock matter if the cd-invariance
   finding stands at the converged rungs?

6. **Does the Singh 2016 Eq. (4) signage actually predict deck
   ordering at all?**  The deck claims Cs⁺ has the LOWEST surface
   pKa near Cu cathode (4.32 vs K⁺=8.49).  But Singh's z_eff for
   Cs⁺ (0.930) is the LARGEST in our cation params, which would
   give the LARGEST ΔpKa magnitude under Eq. (4) — which means
   Cs⁺ should have the LARGEST surface pKa shift, putting its
   surface pKa LOWEST.  Is this consistent with the deck?  Or does
   the model already have a sign-error somewhere that makes the
   observed cd-invariance also confound a hidden directional bug?

End with the verdict line.
