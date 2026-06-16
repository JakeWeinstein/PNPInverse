# Round 4 — Counterreply

Caught two real sign errors in my R3. Verified against the code:
picard_ic.py:421 says `eta_drop = phi_applied_model` for no-Stern,
and picard_ic.py:263 confirms large C_S → psi_D → full_drop. Both
your R3 corrections are right.

## Section 1 — Acknowledgment per issue

**Re your R3 point 1** (no-Stern `eta_drop = phi_applied_model`,
not `phi_applied - phi_o`):
**Accept fully.** Verified at picard_ic.py:419-421 — for no-Stern,
the residual sees `eta_raw = phi_applied_func - E_eq` (not `-phi`),
and Picard mirrors that with `eta_drop = phi_applied_model`. My
multi-ion update_electrostatics no-Stern branch must use
`eta_drop = phi_applied_model` to match. The Stern branch uses
`eta_drop = psi_S` because the residual then has `eta_raw = phi_applied_func
- phi - E_eq` and the Stern Robin BC fixes `phi(0) = phi_applied - psi_S`,
giving `eta_at_OHP = phi_applied - (phi_applied - psi_S) - E_eq = psi_S - E_eq`.

  Concrete:
  ```python
  if multi_ion_ctx is not None and stern_split is not None:
      eta_drop = psi_S
  elif multi_ion_ctx is not None:  # no Stern
      eta_drop = phi_applied_model         # NOT phi_applied - phi_o
  ```

**Re your R3 point 2** (large C_S → psi_D → full_drop, not
psi_S → full_drop):
**Accept fully.** I had the Stern physics inverted. Verified at
picard_ic.py:263:
  ```
  psi_D = stern_coeff * full_drop * lambda_D / (eps + stern_coeff * lambda_D)
  ```
  At large stern_coeff: numerator and denominator both grow, ratio
  → `lambda_D / lambda_D = 1`, so `psi_D → full_drop`. At small
  stern_coeff: `psi_D → stern_coeff · full_drop · lambda_D / eps → 0`.

  So large C_S means "rigid" Stern (small voltage absorbed), large
  diffuse drop. Small C_S means "compliant" Stern (absorbs voltage),
  small diffuse drop.

  For Stern homotopy as an EASE path: start at LARGE C_S (psi_D
  large → strong H+ depletion via Boltzmann → bounded BV rate),
  ramp DOWN to literature 0.10 F/m². Production target is C_S =
  0.10 = "moderate compliance, moderate psi_D".

  Defer this; not first priority per R3 conclusion, but framing now
  correct.

**Re your R3 point 3** (refactor _update_electrostatics into smaller
helpers):
**Accept fully.** Three sub-helpers:
  ```python
  def _solve_phi_o(c_s, c_clo4_bulk, multi_ion_ctx, *, phi_o_prev=None):
      """Returns phi_o for the outer-region anchor.
      Single-ion: log(H_o / c_clo4_bulk).
      Multi-ion: solve_outer_phi_multiion with local bracket."""

  def _solve_picard_stern_split(phi_applied, phi_o, lambda_D_or_eff,
                                 c_clo4_bulk, a_cl, stern_coeff, eps,
                                 multi_ion_ctx=None):
      """Returns (psi_S, psi_D, phi_surface).
      Single-ion: solve_stern_split (BKSA composite).
      Multi-ion: linear-Debye matching with effective_debye_length_local."""

  def _compute_picard_gamma_s(H_o, psi_D, a_h, a_cl, c_cl_anchor,
                               multi_ion_ctx=None, phi_o=None, c_s=None):
      """Returns gamma_s.
      Single-ion: compute_surface_gamma.
      Multi-ion: compute_surface_gamma_multiion with per_ion_outer_concs."""
  ```

  Top-level wrapper preserves single-ion call order for byte-equivalence.

**Re your R3 point 4** (monotonicity test fails outside clamp):
**Accept fully.** Test only on the unclamped operating interval
(say |phi| ≤ 0.9 · min(phi_clamp_per_ion) ≈ ±45 for default
clamp=50). Plus an assert in Picard that `|phi_o|` stays comfortably
inside that interval throughout the loop. Two separate properties.

**Re your R3 point 5** (local bracket fallback — log diagnostic,
don't silently fall back to global):
**Accept fully.** When `solve_outer_phi_multiion` is called with a
local bracket and that fails, raise/log a diagnostic (`local_bracket_failed`)
with the previous phi_o, the new c_dyn, and the residual values at
bracket endpoints. Then fall back to global bisection ONLY if the
monotonicity test for this ctx already passed, otherwise abort.

**Re your R3 point 6** (Phase 5γ ordering: assign k0 BEFORE IC):
**Accept fully.** Correct sequence:
  1. `build_context(sp)` — context with target K0_HAT_R2E in
     ctx["nondim"]["bv_reactions"][j]["k0_model"] AND in bv_k0_funcs.
  2. `build_forms(ctx, sp)` — forms reference bv_k0_funcs.
  3. `set_reaction_k0_model(ctx, j=0, new_k0_model=1e-12)` —
     mutates BOTH ctx["nondim"]["bv_reactions"][0]["k0_model"]
     AND ctx["bv_k0_funcs"][0].assign(1e-12). Picard now sees
     1e-12 in metadata; residual now sees 1e-12 in mutable function.
  4. `set_initial_conditions(ctx, sp)` — Picard runs with 1e-12 k0,
     seeds IC consistent with that.
  5. SS solve at k0=1e-12, dt=1e-4 (small both).
  6. Continuation loop: per step, ramp k0 (and dt independently after
     k0 reaches target).

**Re your R3 point 7** (warm-start preserves state; don't rerun IC at
every k0 step; rename to `set_reaction_k0_model`):
**Accept fully.** Once the FIRST converged solve at small k0 + small
dt is done, subsequent k0 ramps don't need IC rebuild. Just:
  ```python
  for new_k0 in k0_ladder:
      set_reaction_k0_model(ctx, j=0, new_k0_model=new_k0)
      success = run_ss_with_rollback(ctx, ...)
      if not success: rollback k0 multiplier and retry
  ```
  Picard metadata is irrelevant during continuation (no new IC); only
  needed for diagnostic consistency. Rename function accordingly.

**Re your R3 point 8** (`ctx["dt_const"]` not `ctx["dt_constant"]`):
**Accept fully.** Verified at forms_logc_muh.py:581. Use
`ctx["dt_const"].assign(new_dt)`. Same pattern as bv_k0_funcs.

**Re your R3 point 9** (C+D returns immediately if no cold successes;
need preconverged_anchors API or separate warm-walk):
**Accept fully.** Two clean options:

  Option A: extend C+D with `preconverged_anchors: dict[int, dict]`
  parameter mapping `orig_idx → snapshot dict`. When provided, C+D
  treats those points as if Phase 1 cold-converged (skipping its own
  cold attempt for those indices) and uses them as warm-walk anchors.

  Option B: write a standalone `warm_walk_from_anchor(sp, anchor_ctx,
  v_rhe_grid, anchor_idx, ...)` that uses C+D's existing `_warm_walk`
  internals but starts from an externally-built anchor.

  Option A is more orthogonal — preserves C+D as the single entry
  point. Option B is less invasive (no API changes to C+D). Going
  with A; the changeset is small (~30 lines): a new dict parameter,
  a check in Phase 1 loop to skip cold-attempt for anchor indices,
  and snapshot copy-in for those indices.

**Re your R3 point 10** (k0 + dt: BOTH small initially; ramp k0
first keeping dt small; then ramp dt):
**Accept fully.** Order:
  1. Initial state: k0=1e-12, dt=1e-4. SS solve.
  2. k0 ramp: 1e-12 → 32× → 1e-10 → ... → K0_HAT_R2E (8 steps).
     dt stays at 1e-4 throughout.
  3. dt ramp: 1e-4 → 10× → 0.001 → ... → 0.25 (4 steps).
     k0 stays at K0_HAT_R2E.
  4. Final state has both at production values; this is the anchor
     for C+D warm-walk.

**Re your R3 point 11** (z=0 phi_surface not guaranteed = phi_applied;
measure empirically):
**Accept fully.** I'll measure `phi_surface_mean` and assembled
`bv_rate_exprs[0]` at the z=0 fallback IC after Phase 5β fix and
report. Quantitative not speculative.

**Re your R3 point 12** (rate-consistency tolerance: 1e-3 rel +
1e-10 abs floor; 1e-2 for spatial; divide by electrode area):
**Accept fully.** Concrete tolerances codified in test suite:
  ```python
  def assert_rate_consistent(picard_R, residual_assembled, *,
                              electrode_area_nondim,
                              rel_tol=1e-3, abs_tol=1e-10,
                              tag="constant_state"):
      residual_per_unit_area = residual_assembled / electrode_area_nondim
      diff = abs(picard_R - residual_per_unit_area)
      bound = max(rel_tol * abs(picard_R), abs_tol)
      assert diff <= bound, ...
  ```
  Spatial-IC variant uses rel_tol=1e-2 to allow for FE quadrature
  drift on the graded mesh.

**Re your R3 point 13** (logc_muh baseline regen is yellow flag;
keep old + new side by side):
**Accept fully.** Versioned baselines:
  - `tests/baselines/logc_muh_single_clo4_legacy_pre5β.json` (current)
  - `tests/baselines/logc_muh_single_clo4_legacy_post5β.json` (new)
  - regression test asserts `direct debye path` baselines unchanged
    AND `linear_phi fallback path` baselines match the new file.

  Document the intentional shift in CHANGELOG-equivalent (ours is
  STUDY_LOG.md).

**Re your R3 point 14** (use `params` dict, not `sp.solver_options`,
for SolverParams + 11-tuple compatibility):
**Accept fully.** grid_per_voltage builds `params = ...` from sp at
line 175, accessible throughout. Use `params['bv_convergence']['formulation']`
not `sp.solver_options['bv_convergence']['formulation']`.

  The orchestrator already does this for other config reads; my fix
  follows the existing pattern.

## Section 2 — Updated artifact (the recommended path, v3)

Plan unchanged in structure (5α, 5ζ, 5β, 5γ, 5δ, 5ε), but spec
tightened with all R3 corrections:

### Phase 5α: Patch Picard for multi-ion

(unchanged from v2 in structure; refactored per R3 §3-5)

  - Three helpers `_solve_phi_o`, `_solve_picard_stern_split`,
    `_compute_picard_gamma_s`.
  - Top-level `_update_electrostatics(c_s, ..., multi_ion_ctx=None)`
    wrapper preserving single-ion call order.
  - `eta_drop = phi_applied_model` for no-Stern (R3 §1).
  - Local-bracket bisection in `solve_outer_phi_multiion` with
    diagnostic on local-bracket failure (R3 §5).
  - Monotonicity test on unclamped operating interval only (R3 §4).
  - Per-ion phi_clamp consistency between multi_ion.py and
    boltzmann.py (R2 §5).

  Tests:
  - `test_picard_multi_ion_csplus_so4_phi_o_at_bulk`
  - `test_picard_single_ion_byte_equivalent` (regression)
  - `test_picard_residual_rate_consistency_csplus_so4` with
    tolerance per R3 §12.
  - `test_multi_ion_electroneutrality_monotone_unclamped`
  - `test_picard_phi_o_stays_inside_phi_clamp`

### Phase 5ζ: Patch multi-ion diagnostics

(unchanged; lands BEFORE 5β so steering uses correct numbers)

### Phase 5β: Patch grid_per_voltage.py:350 + logc_muh linear IC

  1. grid_per_voltage.py:350 — use `params` not `sp.solver_options`
     for formulation lookup (R3 §14):
     ```python
     formulation = params['bv_convergence'].get('formulation', 'logc')
     if formulation == 'logc_muh':
         set_initial_conditions_logc_muh(ctx, _params_with_phi(V_target_eta))
     else:
         set_initial_conditions_logc(ctx, _params_with_phi(V_target_eta))
     ```

  2. forms_logc_muh.py:662 — gate multi-ion linear-phi IC on
     `len(bikerman) > 1`. Use `solve_outer_phi_multiion` +
     `effective_debye_length_local` for the multi-ion linear case.

  3. Versioned baselines (R3 §13):
     - logc + single-ClO4 byte-equivalent (no behavior change).
     - logc_muh + single-ClO4: legacy 15/15 sweep regression test
       with regenerated baseline as POST-5β reference.

### Phase 5γ: Anchor builder with k0 + dt continuation

  1. `set_reaction_k0_model(ctx, j, new_k0_model)` helper (R3 §7
     rename) mutates BOTH metadata AND `bv_k0_funcs`.
  2. dt mutation via `ctx["dt_const"].assign(new_dt)` (R3 §8).
  3. Sequence (R3 §6, §10):
     ```python
     ctx = build_context(sp)
     ctx = build_forms(ctx, sp)
     set_reaction_k0_model(ctx, j=0, new_k0_model=1e-12)
     ctx["dt_const"].assign(1e-4)
     set_initial_conditions(ctx, sp)
     run_ss(ctx, ...)
     # k0 ramp
     for k0 in [1e-10, 3.2e-9, 1e-7, 3.2e-6, 1e-4, 3.2e-3, K0_HAT_R2E]:
         set_reaction_k0_model(ctx, j=0, new_k0_model=k0)
         success = run_ss_with_rollback(ctx, ...)
     # dt ramp (warm-start preserves state)
     for dt in [1e-3, 1e-2, 0.1, 0.25]:
         ctx["dt_const"].assign(dt)
         success = run_ss_with_rollback(ctx, ...)
     ```
  4. C+D extension: `preconverged_anchors: dict[int, dict] | None = None`
     parameter (R3 §9), maps `anchor_idx → ctx_snapshot`. C+D's
     Phase 1 skips cold-attempt for anchor indices and uses provided
     snapshot.

### Phase 5δ (only if 5α-5γ insufficient): empirical voltage homotopy
### Phase 5ε (last resort): legacy-ClO4 IC swap or 2-parameter homotopy

## Section 3 — Continued critique prompt

Round 4 review. The plan is now in implementation-spec territory.
At this point I'd like to know:

  1. Are there any remaining BLOCKERS (issues that would prevent the
     plan from achieving ≥ 15/25 V_RHE convergence) — vs nitpicks
     that are good-to-fix but won't block?

  2. Estimate of probability of success: with this plan, what fraction
     of {Pass A, Pass B, Pass D} do you think reaches ≥ 15/25 in,
     say, 3 days of focused engineering? 50%? 80%? <30%? Calibrate me.

  3. The single highest-risk step in the plan, and what I should
     instrument to detect failure early.

  4. Any final gotchas in the Picard refactor that aren't addressed
     yet (e.g., the legacy `solve_stern_split` BKSA path has
     pre-bisection bracket scans at picard_ic.py:285 that the
     multi-ion linear-Debye doesn't need — should the helper assert
     bracket consistency or just trust the multi-ion shortcut?).

If there are no further blockers, please verdict APPROVED. Minor
nitpicks alone don't justify ISSUES_REMAIN per the original prompt.

Same numbered format and same verdict line at the end.
