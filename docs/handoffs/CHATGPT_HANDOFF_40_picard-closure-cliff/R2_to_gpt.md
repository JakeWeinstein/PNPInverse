# Round 2: Counterreply on the outer-Picard plan

Most of your 25 points land.  Many were genuine bugs; a handful had
implications I had not internalized.  The plan needs a structural rewrite,
not patching.  Below: per-issue accept/defend/clarify, then the rewritten plan.

## Per-issue responses

**Re point 1** (sign-flip mid-derivation, `c_eff_hat = θ_s · (c_b_hat/θ_b − R_hat·I_hat/D_hat)`):
**Accept.**  The boxed Eq A is correct but the prose lost the sign on the
integration step.  Restating cleanly in nondim with `R_hat > 0` as the
positive consumption-rate of O₂ at the OHP, the closure is

  `c_eff_hat(OHP) = θ_OHP · (c_bulk_hat/θ_bulk − R_O2_hat · I_hat / D_O2_hat)`

where `R_O2_hat = Σ_j (−stoich[O₂, j]) · R_j_hat` (positive for net
consumption).  Drop `R_BV/(F·n_e)` entirely; work in molar rate (issue 2).

**Re point 2** (current density vs molar rate, electron weighting):
**Accept.**  Verified `observables.py:_build_bv_observable_form` exposes
`mode="reaction", reaction_index=j, scale=1.0` that returns the raw
nondim `R_j` scalar.  Will use that for the Picard update and avoid the
F·n_e/I_SCALE roundtrip.  `current_density` mode still used for the
final cd diagnostic (separate purpose).

**Re point 3** (Picard in callback runs only after warm-walk converges,
deep-cathodic failures bypass it):
**Accept — this is the most consequential issue.**  Looked at
`Forward/bv_solver/grid_per_voltage.py:613,686,734,825,1112`: callback
fires after a successful Newton convergence within warm-walk substeps.
For a point that warm-walk gives up on, the callback never runs with a
useful state.  The plan's callback-Picard architecture cannot rescue the
6 failed v1 points, which is the entire target region for cliff
investigation.  Fix: replace `solve_grid_with_anchor` with a custom
per-V Picard wrapper that owns the outer loop.  See rewritten plan §3.

**Re point 4** (grid source snapshot taken before callback):
**Accept.**  Tied to issue 3 — custom wrapper takes the snapshot after
Picard convergence.  No mutation issue under the rewrite.

**Re point 5** (Picard coefficient not in continuation state):
**Accept.**  Will extend `PreconvergedAnchor` (or a parallel
`PreconvergedAnchorWithSupply` struct) to carry the supply variable
ξ alongside `U_snapshot`.  Per-V results store `xi_converged_value` so
next V warm-starts from previous V's ξ.

**Re point 6** (anchor + Stern bump not Picard-converged):
**Accept.**  The custom wrapper runs the Picard loop after anchor build
and after each Stern bump rung.  Initial ξ at anchor V=+0.55 V is near-
no-flux (η anodic, rate suppressed), so Picard converges in 1–2 iters.
Same for Stern bumps (each rung is small perturbation; ξ from prior
rung is a good initial guess).

**Re point 7** (`ctx["_last_solver"].solve()` is pseudo-transient,
not steady-state; `make_run_ss` is the steady-state):
**Accept.**  Looked at `forms_logc_muh.py:284,1007` — `U_prev` is the
time-stepping previous state.  A single `solve()` advances one
pseudo-time step, not to steady state.  For Picard, each inner solve
must be steady-state.  Two options:
  (a) Call `make_run_ss(ctx, ...)` then iterate `U_prev.assign(U); solver.solve()`
      until `|U − U_prev| < tol` per Picard iter.
  (b) Use the existing time-march to steady state convention (currently
      used in anchor + grid walk).  Per-V uses `t_end=80.0 · dt=0.25 = 320 substeps`,
      which is enough for steady state.
  Will use (b) for consistency with v1; effectively the existing Newton
  solve already marches to steady state via the dt loop.  Within Picard,
  each inner solve calls the steady-state-march path, not a single Newton step.

**Re point 8** (replacing inline `ln(c_cat)` with a scalar freezes/loses
`ln(packing)` Newton coupling):
**Accept — critical design flaw caught.**  Will introduce a scalar
supply variable `ξ` (R-space `fd.Function`) and define
  `ln(c_cat_for_BV) = ln(packing) + ln(ξ)`
where `ln(packing)` stays as the inline UFL expression (Newton-coupled,
recomputed each iter as concentrations evolve) and `ln(ξ)` is the
Picard-controlled scalar.  Picard updates ξ between Newton solves;
within Newton, packing evolves naturally and the BV rate tracks it.

**Re point 9** (initialization `log(c_cat_bulk · θ_b)` is wrong; no-flux
limit is `c_bulk · θ_OHP/θ_b`):
**Accept.**  With the `ln(c_cat) = ln(packing) + ln(ξ)` design from
issue 8, the natural initial value is `ξ = c_bulk/θ_b` (which makes
c_cat = c_bulk · θ_OHP/θ_b, the no-flux equilibrium).  Init now
unambiguous in the supply-variable formulation.

**Re point 10** (`packing` not exposed in ctx; script can't assemble it):
**Accept.**  Will add `ctx["packing_expr"]` and `ctx["theta_inner_expr"]`
in the form-build, both at the same point where they're constructed in
`forms_logc_muh.py:447-452`.  Script imports and assembles directly,
no duplicated algebra.

**Re point 11** (per-reaction Picard fns wrong for shared O₂ across
parallel 2e/4e):
**Accept.**  Picard supply variable ξ is **per cathodic species**, not
per reaction.  Dict keyed by species_idx with value = `(Function, list
of (rxn_idx, stoich))`.  For Jithin single-R2e: one species, one entry.
For parallel 2e/4e: one ξ for O₂, two reactions consume it.

**Re point 12** (stoichiometry: `R/(F·n_e)` only works for single
1-O₂-per-event reaction):
**Accept.**  Folded into point 11: total O₂ flux at OHP is
`J_O2_hat = Σ_j (−stoich[O₂, j]) · R_j_hat`.  Use this in the closure
update.  Drop the F·n_e accounting entirely.

**Re point 13** (2D averaging mean(θ)·mean(I) ≠ pointwise streamline
closure):
**Accept** with a defense layer.  Our mesh (Nx=8, Ny=80, β=3.0 graded
in y) and BC structure (uniform x, Stern + BV on bottom, Dirichlet bulk
on top) yield an x-invariant solution to within rounding.  Will assert
this empirically in diagnostics: compute the x-variance of
`assemble(1/packing * dx) / Lx` by splitting the domain into per-x
strips, and assert `std/mean < 1e-3`.  If x-invariance fails, fall back
to per-x-strip closure or abort with a clear message.

**Re point 14** (convergence on damped residual underestimates the
true fixed-point residual):
**Accept.**  Check both:
  - Damped step: `|log(ξ_damped) − log(ξ_old)| < tol_step` (progress)
  - Undamped target: `|log(ξ_target) − log(ξ_old)| < tol_residual`
    (true fixed-point residual)
Only call converged when BOTH are below `tol = 1e-3`.

**Re point 15** (flooring negative `c_eff` can manufacture a cliff;
positivity-preserving update needed):
**Accept — important.**  Will use a semi-implicit positivity-preserving
update.  Define `K = R_O2_hat / c_eff_hat_old` (effective rate constant
at current state).  Then the closed-form update:

  `c_eff_new = θ_OHP · (c_bulk_hat/θ_b) / (1 + K · I_hat · θ_OHP / D_O2_hat)`

guarantees positive `c_eff_new` and is exact at the linearized fixed
point of `c_eff = θ_OHP · (c_bulk_hat/θ_b − R/(c_eff_old) · c_eff · I / D)`.
Floor only on the truly degenerate case `K → ∞`, with explicit logging
"system has no positive fixed point at this V".

**Re point 16** (no rollback for failed Picard inner solves):
**Accept.**  Snapshot `U`, `U_prev`, all ξ Functions before each Picard
iter.  On Newton failure (`SNES_DIVERGED` or exception): restore
snapshot, halve relaxation, retry once.  Persistent failure: abort that
V, mark `picard_converged=False`, do not use diagnostics.

**Re point 17** (max-iter Picard still reported as converged in JSON):
**Accept.**  Reported `converged` flag becomes
`grid_converged AND picard_converged`.  Maintain separate
`picard_converged` array for diagnostics.  Plotter masks cd/pc on the
combined flag.

**Re point 18** (Eq B is overclaimed — full R has H+ factors, η is
state-coupled, etc.):
**Accept.**  Eq B will be labeled as the **single-cathodic-species
scalar Picard update** (the only place it's used).  Real fixed point is
the coupled (PDE + Picard) system; Eq B is the explicit Picard target
formula for ξ.  Will rewrite that section to clarify.

**Re point 19** (BV exponent sign in math vs code):
**Accept.**  Code uses `exp(-α·n·eta_clipped)` for cathodic where
`eta_clipped = (V_applied − E_eq)/V_T` (cathodic → eta < 0).
Will restate the math using exactly that convention.

**Re point 20** ("continuum equivalent of Jithin Eq 4.31" not proven):
**Accept.**  Will demote the claim.  Re-titled: "Continuum-MPNP
boundary closure derived from steady-state Bikerman transport, which
yields a structurally analogous form to Jithin's Eq 4.31 (equilibrium
term + flux-supply correction) without claiming term-by-term mapping."
The plan acknowledges that without Jithin's exact spectral
discretization of κ_5·φ·g, we are testing **the continuum-MPNP analog**,
not the literal Jithin closure.  If the analog doesn't reproduce his
cliff (per pre-analysis), that's evidence his cliff is either a
non-continuum spectral artifact OR requires extra physics beyond what
his closure encodes.

**Re point 21** (single-R2e script is insufficient for Fig 4.36 claims):
**Accept.**  Plan now has two run stages:
  (a) **Single R2e** (this plan): structural test of the Picard
      infrastructure + verify pre-analysis prediction (smooth S-curve
      to ~Levich).
  (b) **Parallel 2e/4e** (follow-up): required before any claim about
      Jithin Fig 4.36.  Out of scope for THIS plan but explicitly noted
      as the next step gated on (a) succeeding.

**Re point 22** (no hard tests):
**Accept.**  Will add a `tests/forward/bv/test_jithin_picard_closure.py`
with:
  - `test_no_flux_equivalent_to_v1`: at anodic V (R≈0), Picard converges
    in 1 iter; cd matches v1 closure-exact run to 6 digits.
  - `test_theta_unity_recovers_levich`: with `a_vals_hat = 0` for all
    species (turns off Bikerman), Picard plateau == standard Levich
    within 1%.
  - `test_function_update_no_rebuild`: after `xi_func.assign(new_value)`,
    rate observable assembles with new value; no form rebuild required.
  - `test_fixed_point_residual_at_convergence`: at converged ξ,
    residual `|c_eff/θ + R·I/D − c_bulk/θ_b| < 1e-6`.
  - `test_raw_rate_to_molar_flux_conversion`: extract R via mode="reaction",
    multiply by stoichiometry, check against known Levich at θ=1.

**Re point 23** (packing floor not re-checked under Picard):
**Accept.**  Per Picard iter, record `min(theta_inner)` (without floor)
and the area-fraction where `theta_inner ≤ packing_floor`.  If
floor-hit-area > 1% anywhere, log warning.  If > 10%, flag the Picard
iter as unreliable.

**Re point 24** (cliff diagnostic ratio uses wrong sign convention):
**Accept.**  Fix: `cliff_ratio = abs(cd_far_cathodic) / abs(cd_mid_min)`.
A ratio < 1 indicates a cliff (current magnitude decreases at deep
cathodic); ratio ≈ 1 means flat plateau (no cliff).

**Re point 25** (H+ closure assumption is stale under Picard states):
**Accept.**  Per Picard iter, also compute and store:
  - `c_H_OHP_pde` — current PDE value
  - `c_H_closure_estimate = c_H_bulk · exp(-z_H · phi_OHP) · θ_OHP/θ_b`
  - `h_closure_relative_error = |c_H_OHP_pde − c_H_closure_estimate| / c_H_closure_estimate`
If h_closure_relative_error > 25% at any V, flag — substitute c_H+ too
becomes warranted.  (For this plan iteration, still leave H+ at PDE
value; track the diagnostic to determine if a v3 H+-also-substituted
patch is needed.)

## Updated artifact

See full rewrite below.  Changes vs v1:

- **Math:** Rewritten in code-convention nondim, single positive rate
  symbol R_O2_hat for O₂ consumption, sign-checked end-to-end.
- **Closure design:** Introduce scalar supply variable ξ (R-space
  Function), use `ln(c_cat) = ln(packing) + ln(ξ)` (keep packing
  inline, Newton-coupled).  Per cathodic species, not per reaction.
- **Architecture:** Custom per-V Picard wrapper replaces
  `solve_grid_with_anchor`.  Anchor + Stern bump also Picard-
  converged.  Per-V state carries ξ alongside U.
- **Updates:** Semi-implicit positivity-preserving:
  `c_eff_new = θ·(c_b/θ_b) / (1 + K·I·θ/D)` with `K = R/c_eff_old`.
  Damping kept as fallback.
- **Convergence:** Both damped step + undamped target residual
  must clear tol.
- **Rollback:** Snapshot before each Picard iter; restore on Newton
  fail; halve damping; retry once.
- **Reporting:** `converged = grid_AND_picard`.  Cliff ratio uses
  abs().  H+ closure-quality diagnostic per iter.  Floor-hit fraction
  per iter.
- **Validation:** 2D x-invariance assertion; demoted "Jithin literal
  closure" claim to "continuum-MPNP analog"; scoped this plan to
  single R2e with parallel 2e/4e as gated follow-up.
- **Tests:** Hard tests for no-flux v1 equivalence, θ=1 Levich,
  function-update-no-rebuild, fixed-point residual, rate-to-flux
  conversion.

```markdown
# Jithin Closure Outer-Picard Implementation Plan (v2, post-GPT-round-1)

## Goal

Implement an outer-Picard wrap on top of the existing
`bv_jithin_closure_form` patch.  Test whether the **continuum-MPNP
analog** of Jithin's Eq 4.31 closure (equilibrium ceiling +
steady-state flux-supply correction derived from continuum Bikerman
transport) reproduces his Fig 4.36 cliff in our gradient-form solver.

**Scope:** single-reaction R2e (Jithin emulation) only.  Parallel 2e/4e
validation is a gated follow-up.  Validates the Picard infrastructure
and tests pre-analysis prediction that continuum-Bikerman correction
is too small to produce the cliff.

## Math (rewritten in code convention)

### Symbols and conventions

- `c_b_hat` = bulk O₂ concentration (nondim, = `c_O2_bulk_mol_m3 / C_SCALE`)
- `c_eff_hat` = effective c_O₂ at OHP (nondim) used in BV rate
- `θ_OHP`, `θ_b` = Bikerman packing fraction at OHP / bulk (dimensionless)
- `R_O2_hat` = molar rate of O₂ consumption at OHP (nondim, positive)
  = `Σ_j (−stoich[O₂, j]) · R_j_hat`
- `I_hat` = `∫₀^L_hat dy_hat / packing(y_hat)` (nondim length)
- `D_O2_hat` = O₂ diffusivity (nondim)
- `η = (V_applied − E_eq) / V_T` (cathodic → η < 0)
- BV cathodic rate: `R_j_hat = k₀_hat · c_cat_hat · exp(−α·n·η)`
  (cathodic V → η < 0, exp positive, rate grows)
- `ξ` = scalar supply variable, R-space `fd.Function`,
  related to c_eff by `c_eff_hat = θ_OHP · ξ`

### Continuum-MPNP boundary closure for neutral cathodic species

Steady-state Bikerman 1D for z=0 species:

  J_hat = −D_hat · θ · ∇(c/θ)

Integrate from OHP (y=0) to bulk (y=L_hat) with constant flux:

  c_OHP_hat / θ_OHP − c_bulk_hat / θ_b = −(J_hat / D_hat) · I_hat

For cathodic O₂ consumption, J_hat at OHP is the net molar flux INTO
the OHP from bulk = R_O2_hat (positive).  So:

  **c_OHP_hat = θ_OHP · (c_b_hat / θ_b − R_O2_hat · I_hat / D_O2_hat)**  (Eq A)

Equivalently using ξ = c_OHP_hat / θ_OHP:

  **ξ = c_b_hat / θ_b − R_O2_hat · I_hat / D_O2_hat**                    (Eq A')

This is the continuum-MPNP closure for the boundary supply variable.
Structurally analogous to Jithin's Eq 4.31 (equilibrium term + flux-
supply correction) but derived from continuum transport, not from his
spectral integro-diff closure.  We are NOT claiming term-by-term
identity with his κ_5·φ·g.

### Picard fixed-point structure

The BV rate uses `c_eff_hat = θ_OHP · ξ` where ξ is Picard-controlled.
Within Newton, θ_OHP evolves with the concentration profile (coupled
through the residual).  Between Picard iters, ξ is held fixed, then
updated.

Picard update (semi-implicit, positivity-preserving):

Define `K_old = R_O2_hat_old / c_eff_hat_old = R_O2_hat_old / (θ_OHP_old · ξ_old)`.
Then the fixed-point of Eq A':

  ξ = c_b_hat/θ_b − K_old · θ_OHP · ξ · I_hat / D_O2_hat

solved for ξ:

  **ξ_new = (c_b_hat / θ_b) / (1 + K_old · I_hat · θ_OHP_avg / D_O2_hat)**  (Eq B)

where `θ_OHP_avg` is the surface-mean of packing from the just-converged
state.  This guarantees `ξ_new > 0` always (no floor needed in the
typical regime).  Limits:

- Kinetic regime (K · I · θ / D << 1): ξ → c_b_hat/θ_b (no-flux equilibrium)
- Transport regime (K · I · θ / D >> 1): ξ → D_O2_hat / (K · I · θ_OHP_avg)
  → R = K · c_eff = K · θ · ξ = D_O2_hat / I_hat = continuum Levich

Note: this is the **explicit-Picard update target**; the **actual coupled
fixed point** of (PDE + Picard) includes all the other terms in R_j
(H+ factor, η-dependence on phi, etc.) and is solved by iterating Eq B
to fixed point.  Eq B is NOT the full coupled fixed point of the system —
it is the Picard target formula for ξ given the current PDE state.

### Pre-analysis prediction

For our geometry (L=10 µm, λ_D ≈ 0.5 nm, sat layer ~few nm,
θ_OHP ≈ 0.034, θ_b ≈ 0.94):
- I_hat ≈ L_hat / θ_b (dominant bulk contribution, tiny sat correction)
- Continuum Levich ≈ standard Levich (correction ~6%)
- Predicted curve: smooth S-curve from kinetic onset (V≈0) to ~Levich
  plateau (~-0.7 mA/cm²), NO cliff

If true → rules out continuum-Bikerman as cliff mechanism; pivot to
surface-coverage / non-covalent cation effect (Strmcnik) as path 2.
If unexpected (sub-Levich plateau or cliff): re-examine I integral
and saturation profile assumptions.

## Implementation

### Files modified

1. **`Forward/bv_solver/config.py`** (~10 lines)
   - Add `bv_picard_mode: bool` to `_get_bv_convergence_cfg` (default False).
   - Validate: `bv_picard_mode=True` requires `bv_jithin_closure_form=True`
     AND `bv_log_rate=True`.

2. **`Forward/bv_solver/forms_logc_muh.py`** (~50 lines)
   - In the BV builder, when `bv_jithin_closure_form=True` AND
     `bv_picard_mode=True`:
     - Per cathodic species index `s` (collected from
       `rxn["cathodic_species"]` across all reactions, deduped):
       - Allocate `xi_func_s = fd.Function(R_space, name=f"picard_log_xi_sp{s}")`
       - Initialize: `log(c_bulk_hat_s / θ_b_const)` (no-flux equilibrium)
     - Store dict: `ctx["picard_log_xi_funcs"][s] = xi_func_s`
     - In log_cathodic construction, replace
         `log_c_cat = u_exprs[cat_idx]`
       with
         `log_c_cat = ln(packing) + ctx["picard_log_xi_funcs"][cat_idx]`
       (keeps packing inline, ξ Picard-controlled).
   - Expose `ctx["packing_expr"] = packing` and `ctx["theta_inner_expr"] = theta_inner`
     (uncapped version) at line ~452.

3. **New `Forward/bv_solver/closure_picard.py`** (~250 lines)
   - `class PicardState`: holds ξ values per species + Newton snapshot.
   - `solve_with_picard_closure(ctx, sp, *, max_iters=15, tol=1e-3,
        damping=0.5, max_damping_retries=3)`:
     - Snapshot U, U_prev, all xi_funcs.
     - Inner steady-state Newton solve via existing dt-march to t_end.
       (Verify "single ctx['_last_solver'].solve()" gives steady state;
       if not, wrap with steady-state convergence loop.)
     - Extract diagnostics via UFL assemble:
       - `theta_OHP_avg = assemble(packing_expr * ds(electrode))
                          / assemble(Constant(1) * ds(electrode))`
       - `I_hat = assemble((1/packing_expr) * dx) / Lx_hat`
       - x-invariance check: split domain, compute per-strip I, assert
         `std/mean < 1e-3`; abort with diagnostic if fails.
       - `R_j_hat = assemble(_build_bv_observable_form(ctx, mode="reaction",
                                reaction_index=j, scale=1.0))`
         for each j with `bv_jithin_closure_form` cathodic species.
     - Compute total molar consumption: `R_O2_hat = Σ_j (−stoich[O₂, j]) · R_j_hat`
     - For each cathodic species s (typically one, O₂):
       - `c_eff_old_hat = θ_OHP_avg · exp(xi_func_s)` (current state)
       - `K_old = R_s_hat / c_eff_old_hat`
       - `xi_target = (c_bulk_hat_s / θ_b_const) / (1 + K_old · I_hat · θ_OHP_avg / D_s_hat)`
       - Damped update: `xi_damped = (1−α)·ξ_old + α·xi_target`
       - Convergence: BOTH `|log(xi_damped) − log(xi_old)| < tol_step`
         AND `|log(xi_target) − log(xi_old)| < tol_residual`
     - If Newton failed: restore snapshot, halve damping, retry.
       Persistent failure after `max_damping_retries`: mark
       `picard_converged=False`, return.
     - Diagnostic capture per iter:
       `(iter, xi_per_species, c_eff_per_species, R_per_rxn, R_O2_total,
         theta_OHP_avg, I_hat, min_theta_inner, floor_hit_area_frac,
         c_H_OHP_pde, c_H_closure_est, h_closure_rel_err, damping_used)`

4. **`PreconvergedAnchorWithPicard` dataclass** (extends `PreconvergedAnchor`):
   - Adds `xi_snapshots: tuple[(int, np.ndarray), ...]` per species.

5. **New script `scripts/studies/_run_jithin_closure_picard.py`** (~500 lines)
   - Fork `_run_jithin_closure_exact.py`.
   - `BV_PICARD_MODE = True`, `PICARD_MAX_ITERS = 15`,
     `PICARD_TOL_STEP = 1e-3`, `PICARD_TOL_RESIDUAL = 1e-3`,
     `PICARD_DAMPING_INIT = 0.5`.
   - Custom three-stage:
     - Stage 1: anchor build + Picard converge at V=+0.55 V (1-2 iters
       expected since η anodic).
     - Stage 2: Stern bump ladder, each rung followed by Picard converge.
     - Stage 3: custom per-V loop replacing `solve_grid_with_anchor`:
       - For each V_RHE in grid:
         - Cold or warm Newton init from previous V's converged state.
         - Picard converge via `solve_with_picard_closure(...)`.
         - Capture diagnostics + ξ for warm-start.
         - On failure: report partial result, do NOT use as warm-start.
   - JSON output: full per-V Picard trajectory + final state.
   - `converged = grid_converged_per_V AND picard_converged_per_V`.

### Diagnostics captured per V_RHE

- `cd_mA_cm2`, `pc_mA_cm2` (from final converged state)
- `c_O2_OHP_nondim`, `c_H2O2_OHP_nondim`, `c_H_OHP_nondim`, `phi_OHP_nondim`
- `picard_iters`: list of per-iter dicts (see above)
- `picard_n_iters`, `picard_converged`, `picard_final_residual_step`,
  `picard_final_residual_target`
- `min_theta_inner_per_iter` array
- `h_closure_quality_per_iter` array
- `grid_converged` AND `picard_converged` → `converged` (single mask)

## Tests (new `tests/forward/bv/test_jithin_picard_closure.py`)

- `test_no_flux_equivalent_to_v1`: at anodic V where R≈0,
  Picard converges in 1-2 iters; cd matches v1 to 6 digits.
- `test_theta_unity_recovers_levich`: `a_vals_hat = [0,0,0]` and
  no counterions → standard Fick, plateau at standard Levich within 1%.
- `test_function_update_no_rebuild`: `xi_func.assign(np.log(0.5))`
  then `assemble(rate_form)` changes; no `solve()` invocation.
- `test_fixed_point_residual_at_convergence`: at converged ξ at one
  test V, residual `|ξ + R·I·θ/D − c_b/θ_b| < 1e-6`.
- `test_raw_rate_to_molar_flux_conversion`: extract R via
  `mode="reaction", scale=1.0`, multiply by stoich; check matches
  expected Levich molar rate at θ=1.

## Risk areas

1. **Picard divergence at deep cathodic.**  Semi-implicit positivity-
   preserving update minimizes this; damping is fallback.
2. **x-invariance check fires.**  If x-variance > 1e-3, the scalar
   closure is invalid.  Fallback: per-x-strip ξ (one ξ per Nx cell,
   plumbed through Function on a Lagrange P0 in x); for v2 just abort.
3. **Pseudo-transient vs steady state.**  Need to verify the existing
   dt-march path actually reaches steady state at every V.  Will add
   `dU_norm` diagnostic per outer Newton solve; if not converging,
   either extend t_end or switch to `make_run_ss`.
4. **Newton failure in Picard inner.**  Snapshot/restore/halve-damping
   protocol covers transient failures.  Persistent failure → abort V.
5. **ξ becomes very small at deep cathodic.**  Numerical issue if
   `log(ξ)` underflows.  Add floor at `ξ_min = exp(-50)` ≈ 2e-22 with
   warning.

## Expected outcomes (unchanged from v1)

Per pre-analysis: smooth S-curve to ~Levich, NO cliff.  Valuable as
solver correctness validation via independent code path.  If cliff
DOES appear: pre-analysis was wrong, I integral is larger than
estimated; investigate θ(y) profile.

## Out-of-scope for this plan (explicit follow-ups)

- Parallel 2e/4e validation (gated on this plan completing).
- Surface-coverage / Frumkin isotherm patch (path 2).
- Hard-sphere mixture D(packing) (path 3).
- H+ closure-form substitute (if h_closure_rel_err diagnostic shows > 25%).
- Per-x-strip ξ (if x-invariance assertion fails).
- Custom from-scratch Chebyshev spectral solver (separate effort).
```

## Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
