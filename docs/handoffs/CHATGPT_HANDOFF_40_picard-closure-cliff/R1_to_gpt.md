# Round 1: Outer-Picard implementation plan for Jithin Eq 4.31 closure

You are reviewing an implementation plan for completing the second piece
of Jithin's 2024 thesis Eq 4.31 closure inside an existing Firedrake-based
gradient-form MPNP-BV solver.  The first piece (equilibrium ceiling)
already landed and was validated.  This plan extends it with the
flux-supply correction via an outer Picard wrap.

Please be adversarial and find every hole — math, code, dimensional
analysis, convergence, scope, hidden assumptions.

---

## Section 1: Context

### 1.1 The problem

Jithin's 2024 thesis Fig 4.36 is the canonical reference jV curve for
ORR on the Seitz/Mangan deck (Cs⁺/SO₄²⁻ at pH 2 with parallel 2e⁻/4e⁻).
His simulated curve has two features our solver does not reproduce:

- **Sub-Levich plateau** at ≈ -0.36 mA/cm² (vs his own Levich estimate at
  L=10 µm of -0.724 mA/cm², which we reproduce exactly)
- **Far-cathodic cliff** at V_RHE < -0.3 V: cd magnitude drops from
  ≈ -0.36 plateau back to ≈ -0.15 at V_RHE = -0.4 V

The user has confirmed the cliff is **physically real in experiment**, not
a numerical artifact, so we need to find what mechanism in real PDE solvers
produces it.

### 1.2 Solver stack

- 3 dynamic species (O₂, H₂O₂, H⁺) with `formulation='logc_muh'`
  (proton electrochemical potential as primary variable for H⁺,
  log-concentration for O₂ and H₂O₂)
- Analytic-Boltzmann counterions (Cs⁺ and SO₄²⁻) via
  `build_steric_boltzmann_expressions` in
  `Forward/bv_solver/boltzmann.py`
- Bikerman steric on the dynamic species via `μ_steric = -ln(θ)` where
  `θ = max(1 - Σ a_k·c_k, packing_floor)` with `packing_floor = 1e-15`
  (Jithin emulation) or `1e-8` (production default)
- Steric flux: `J = D · c · ∇μ = D · c · (∇log(c) + ∇μ_steric)`
- Log-rate Butler–Volmer with Stern Robin BC (`C_S = 1.16 F/m²` for
  Jithin; `0.20 F/m²` for production)
- 2D mesh (Nx=8, Ny=80, β=3.0 graded) with finest cell ≈ 0.02 nm
  near OHP

### 1.3 Existing v1 patch (closure equilibrium term only)

Already landed and tested.  When `bv_jithin_closure_form=True`:
- In `Forward/bv_solver/forms_logc_muh.py` reaction loop (around line 559),
  for the log-rate branch, replaces `u_exprs[cat_idx]` (= ln(c_O₂(OHP))
  from PDE) with `ln(c_O₂_bulk · θ(OHP)/θ_bulk_const)`.
- `θ_bulk_const = steric_boltz[0].metadata["theta_b"]` (precomputed by
  `build_steric_boltzmann_expressions`) = `1 - A_dyn_bulk - Σ a_ion·c_ion_bulk`.
- Mutually exclusive with `bv_steric_activity` flag (mutual-exclusion
  check in `config.py`).
- Only patches the cathodic species (O₂); H⁺ as a `cathodic_conc_factor`
  stays at the PDE value because diagnostics showed it already matches
  Jithin's closure for charged species within ~10% (Boltzmann pile-up
  in the PDE captures z≠0 correctly).

### 1.4 v1 results (`StudyResults/jithin_closure_exact_emulation/`)

- 19/25 grid points converged (V_RHE +0.55 down to -0.163 V)
- 6/25 failed (V_RHE -0.20 to -0.40) — closure rate exceeds Levich, no
  PDE steady state exists
- Most cathodic converged cd = **-0.27 mA/cm²** (V=-0.163 V)
  - 38% of Levich (-0.72)
  - 75% of Jithin Fig 4.36 plateau (-0.36)
- c_O₂(OHP) tracks the closure constraint perfectly: 0.10 mol/m³ at V=-0.16
  (Cs⁺ pile-up) → 0.25 (≈bulk) at V≈0 → 7e-6 at V=+0.55 (SO₄²⁻ pile-up)
- φ(OHP) linear in V_RHE from -3 to +8 nondim (Stern partition)

So the v1 patch is wired correctly — the mechanism works.  Failures at
deep cathodic are exactly where Jithin's cliff is, and they occur
because v1 has no flux-supply term to keep the rate bounded.

### 1.5 What we already ruled out (per `docs/papers/jithin_thesis_emulation_findings.md`)

- `packing_floor` (1e-8 vs 1e-15): bit-exact identical IV curves — floor never engages
- Mesh resolution: first cell already sub-Angstrom, not the bottleneck
- The `bv_steric_activity` flag (multiplying BV rate by θ but keeping
  PDE c_O₂(OHP)): transport-balance compensation — system raises
  c_O₂(OHP) to satisfy mass balance, no cliff

### 1.6 My pre-analysis (the prediction this plan needs to test or invalidate)

Steady-state continuum Bikerman for neutral species:
`J = -D · θ · ∇(c/θ)`.  Integrating from OHP (y=0) to bulk (y=L) gives
`c_OHP/θ_OHP = c_bulk/θ_bulk - (J/D)·I` where `I = ∫₀^L dy/θ(y)`.

For our geometry:
- L = 10 µm, λ_Debye ≈ 0.5 nm at I_strength ≈ 0.3 M
- Saturation layer δ_sat ~ few λ_D ≈ 1-5 nm where θ → 0.034 (Bikerman cap)
- I ≈ δ_sat/θ_OHP + (L−δ_sat)/θ_bulk ≈ 5e-9/0.034 + 1e-5/0.94
       ≈ 1.5e-7 + 1.06e-5 ≈ 1.06e-5 m ≈ L/θ_bulk

So the continuum sub-Levich correction is only ≈ 6% — Levich limit
becomes -0.68 instead of -0.72.  NOT the factor-2 drop in Jithin's plot.

**Prediction:** the Picard wrap with rigorous flux correction will give
a smooth S-curve from kinetic onset to ~Levich plateau, NO cliff at any
V.  If true, this rules out continuum-Bikerman as the cliff mechanism
and points at extra physics needed (cation surface adsorption / Frumkin
isotherm, local-pH coupling, hard-sphere mixture D, or numerical artifact
in Jithin's spectral closure).  We are running this anyway because (a) it
validates the solver via an independent code path, (b) confirms or
refutes my pre-analysis, (c) gives us a clean kinetic-S-curve baseline
to compare against future surface-coverage patches.

---

## Section 2: The plan under review

```markdown
# Jithin Closure Outer-Picard Implementation Plan (v1, pre-GPT-critique)

## Goal

Implement the outer-Picard flux-correction wrap on top of the existing
`bv_jithin_closure_form` patch, completing Jithin's Eq 4.31 closure for our
gradient-form MPNP solver.  Test whether the **full** closure (equilibrium
term `A_k · (1−Σa·c)` + flux-supply term `κ_5·φ_k·g_k · (1−Σa·c)`) reproduces
his far-cathodic cliff.

## Mathematical derivation

### Continuum steady-state Bikerman MPNP

For a neutral species (z=0) in 1D steady state with Bikerman steric
chemical potential μ_steric = -ln(θ):

  J = -D · c · ∇μ = -D · c · ∇(ln(c) - ln(θ)) = -D · θ · ∇(c/θ)

Define ξ = c/θ.  Steady state ⇒ J constant:

  ∇ξ = -J / (D · θ(y))

Integrate from OHP (y=0) to bulk (y=L):

  ξ(L) - ξ(0) = -(J/D) · ∫₀^L dy/θ(y)
  c_bulk/θ_bulk - c_OHP/θ_OHP = -(J/D) · I    where I ≡ ∫₀^L dy/θ(y)

So:

  **c_OHP = θ_OHP · [c_bulk/θ_bulk - (J/D) · I]**            (Eq A)

This is the continuum equivalent of Jithin's Eq 4.31 for the neutral O₂.
The first bracketed term is the equilibrium ceiling (Jithin's `A_k`); the
second is the flux-supply correction (Jithin's `κ_5·φ_k·g_k`).

### Implicit fixed point with kinetic BC

BV rate: R_BV = k₀ · c_OHP · exp(α·n·η/V_T).  Flux at OHP: J = R_BV / (F·n_e).
Substituting J into Eq A:

  c_OHP = θ_OHP · c_bulk/θ_bulk - θ_OHP · R_BV · I / (D · F · n_e)

Substituting R_BV(c_OHP):

  c_OHP · [1 + k₀·exp(η)·I·θ_OHP/(D·F·n_e)] = θ_OHP · c_bulk/θ_bulk
  **c_OHP = θ_OHP · c_bulk / [θ_bulk · (1 + A)]**            (Eq B)
  where A ≡ k₀·exp(η)·I·θ_OHP / (D·F·n_e)

R_BV at fixed point:

  R_BV = k₀·exp(η) · θ_OHP·c_bulk/[θ_bulk·(1+A)]
       = (k₀·exp(η)·c_bulk·θ_OHP/θ_bulk) / (1 + A)

Limits:
- Kinetic regime (A << 1): R_BV ≈ k₀·c_bulk·θ_OHP/θ_bulk·exp(η)   (= v1 patch)
- Transport regime (A >> 1): R_BV ≈ F·n_e·D·c_bulk/(θ_bulk·I)     (= continuum Levich)

### Pre-analysis prediction

For our geometry:
- L = 10 µm, λ_D ≈ 0.5 nm (Cs⁺/SO₄²⁻ at 0.3 M)
- Saturation layer δ_sat ~ few λ_D ≈ 1-5 nm
- θ_OHP ≈ 0.034 (Bikerman cap), θ_bulk ≈ 0.94

Estimate:
- I_diffuse ≈ δ_sat · avg(1/θ) ≈ 5e-9 · 30 ≈ 1.5e-7 m  (negligible)
- I_bulk ≈ (L - δ_sat)/θ_bulk ≈ 10e-6/0.94 ≈ 1.06e-5 m  (dominant)
- I ≈ 1.06e-5 m ≈ L/θ_bulk

Continuum Levich: R_max = F·n·D·c_bulk/(θ_bulk·I) ≈ F·n·D·c_bulk/L = standard Levich

**Predicted outcome:** smooth S-curve from kinetic onset to ~Levich plateau,
NO cliff.

## Implementation

### Files modified

1. **`Forward/bv_solver/config.py`** (~5 lines)
   - Add `bv_picard_mode` flag to `_get_bv_convergence_cfg` (default False).
   - Validate: `bv_picard_mode=True` requires `bv_jithin_closure_form=True`
     and `bv_log_rate=True`.

2. **`Forward/bv_solver/forms_logc_muh.py`** (~30 lines)
   - In the BV builder, when `bv_jithin_closure_form=True`:
     - If `bv_picard_mode=True`:
       - Allocate one `fd.Function` in `R_space` per reaction:
         `picard_log_c_O2_eff_{rxn_idx}`
       - Use this Function as `log_c_cat_closure` in BV rate
         (instead of inline equilibrium expression `ln(c_bulk·packing/packing_bulk)`)
       - Expose via `ctx['picard_log_c_O2_eff_funcs']` (dict: rxn_idx → Function)
       - Initialize each Function to `log(c_cat_bulk * theta_b_init)` where
         `theta_b_init` is `theta_b_bulk_val` (well-defined at build time).
     - Else (existing v1 behavior): inline equilibrium expression unchanged.

3. **New script `scripts/studies/_run_jithin_closure_picard.py`** (~400 lines)
   - Fork `_run_jithin_closure_exact.py`.
   - Add `PICARD_MODE=True`, `PICARD_MAX_ITERS=15`, `PICARD_TOL=1e-3`,
     `PICARD_UNDER_RELAX=0.5` (damping).
   - Per-V callback `_grab_with_picard` performs outer Picard:
     a. Read current state (Newton just converged with current
        `picard_log_c_O2_eff_func`).
     b. Compute diagnostics via UFL assemble:
        - `I_y_nondim = assemble(1/packing * dx) / Lx_nondim`
        - `theta_OHP = assemble(packing * ds(electrode_marker))
                        / assemble(1 * ds(electrode_marker))`
        - `R_BV_nondim = assemble(bv_rate_observable_form)`
     c. Convert R_BV_nondim → dimensional via I_SCALE.
     d. Compute continuum closure (dimensional):
        `c_OHP_new_dimensional = θ_OHP · c_bulk_dimensional / θ_bulk
                                  - R_BV_dimensional · I_y_dimensional
                                    · θ_OHP / (D_O2_phys · F · n_e)`
        (where I_y_dimensional = I_y_nondim · L_REF)
        Floor: `c_OHP_new = max(c_OHP_new, c_O2_bulk * 1e-30)`
     e. Convert to nondim: `c_OHP_new_nondim = c_OHP_new / C_SCALE`
     f. Damped update:
        `log_c_new = (1 - α_relax) · log_c_old + α_relax · log(c_OHP_new_nondim)`
     g. Convergence check: `|log_c_new - log_c_old| < PICARD_TOL`.
     h. If converged: capture cd/pc as in v1, append Picard trajectory to JSON.
     i. If not: assign log_c_new to picard Function, call
        `ctx['_last_solver'].solve()`, recurse.
     j. Max iters: capture final state, mark as `picard_converged=False`.
   - JSON additions: per-V Picard trajectory (list of c_OHP_nondim, R_BV,
     I_y, theta_OHP per iter); per-V `picard_n_iters` and `picard_converged`.

### Run script orchestration

Maintain v1's three-stage architecture (anchor → Stern bump → grid walk).
Inside the grid walk, modify the per-V callback to do Picard.

## Diagnostics captured

Per V_RHE point:
- `picard_iters`: list of dicts `{iter, c_O2_eff_nondim, R_BV_dim,
  I_y_nondim, theta_OHP, residual}`
- `picard_n_iters`: total iters
- `picard_converged`: bool
- `picard_final_residual`: final |log_c_new - log_c_old|
- Existing cd, pc, c_O2_OHP, c_H_OHP, phi_OHP (post-converged state)

## Risk areas / failure modes

1. **Picard non-convergence (oscillation or divergence).**
   Mitigation: under-relaxation (default α=0.5).  If still oscillating,
   bisect on α.  If still diverging: report failure, dump trajectory.

2. **c_O_eff goes negative.**  Possible if R·I·θ_OHP/(D·F·n_e) > c_bulk·θ_OHP/θ_bulk.
   Means kinetic forcing exceeds even the corrected Levich.  Floor at
   `c_O2_bulk · 1e-30`; if hit, this is the "system can't support steady
   state at this k₀/V" regime — expected at most cathodic V.

3. **Dimensional analysis errors.**  R_BV from observable is in nondim
   (then scaled by -I_SCALE for mA/cm² output).  D_O2 is dimensional.
   I_y from assemble is in nondim length units (relative to L_REF).
   Must convert carefully.  Concrete check: at Picard fixed point in
   ideal-mixture limit (θ=1), continuum prediction should match standard
   Levich within numerical accuracy.

4. **The grid walker callback may not support re-invoking Newton.**
   Need to verify `ctx['_last_solver'].solve()` is safe to call mid-callback
   (it's used in the Stern bump ladder, so should be fine, but the grid
   walker context may differ).

5. **Initial guess matters.**  Inline equilibrium gives `c_O2_eff = c_bulk · θ_OHP/θ_bulk`.
   This is the no-flux limit; with kinetic consumption it overestimates
   c_O2_eff, leading to over-large initial R_BV.  Picard may need several
   iters to descend.  Mitigation: warm-start each V's Picard from previous
   V's converged c_O2_eff_nondim (when V step is small).

6. **Surface integral conventions.**  `assemble(packing * ds(electrode_marker))`
   requires the right facet marker.  Need to verify which marker is the
   electrode in `make_graded_rectangle_mesh`.

7. **The Picard loop runs PER VOLTAGE.**  With 25 voltages × 5-15 Picard
   iters × Newton solve, this could be 5-10× longer than v1 baseline.
   Expected wall time: 30 min - 2 hours.

8. **Form rebuild on Function value change.**  If `picard_log_c_O2_eff_func`
   is referenced in the residual via UFL, updating its value via `.assign()`
   should propagate to the next solve without form rebuild.  Verify Newton
   actually sees the updated value.

9. **Per-reaction Picard Functions.**  We only have one reaction in Jithin
   emulation (R2e), but the infrastructure should be per-reaction list
   for generality.

10. **My continuum derivation might be wrong somewhere.**  Steps to verify:
    - Limit check: θ ≡ 1 → recovers standard Fick/Levich.
    - Limit check: J = 0 → c_OHP = c_bulk · θ_OHP/θ_bulk (equilibrium).
    - Sign convention: cathodic consumption → c_OHP < c_bulk → J flowing
      bulk→OHP.  I used J = R/(F·n_e) > 0 for cathodic.  Re-verify in
      implementation.

## Expected outcomes

### Success (Picard converges at all V)

- Each V converges in ~5-10 Picard iters.
- IV curve: smooth S-curve from kinetic onset (V≈0) to plateau.
- Plateau magnitude: likely close to Levich (~-0.7 mA/cm²) per pre-analysis.
- No cliff at V_RHE < -0.3.
- Solver-correctness verdict: validated (Picard + Newton path = continuum
  steady state, matches direct Newton baseline).

### Partial success (Picard converges some V, diverges others)

- Convergence at moderate V, divergence at deep cathodic.
- Indicates rate-Levich crossover regime where Picard oscillation defeats
  damping.  May need adaptive α or different formulation (e.g., solve in
  R_BV instead of c_O2_eff).

### Unexpected (cliff actually appears)

- Plateau drops at V_RHE < -0.3.
- Would invalidate pre-analysis: I integral must be much larger than my
  estimate, OR the saturation profile extends much deeper than I expect.
```

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
