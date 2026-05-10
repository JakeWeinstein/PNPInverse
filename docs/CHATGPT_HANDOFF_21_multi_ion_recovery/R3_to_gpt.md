# Round 3 — Counterreply

Sharp round. Most points accept; one needs verification; one I want to
push back on the framing of.

## Section 1 — Acknowledgment per issue

**Re your point 1** (set_initial_conditions routes back to analytical
if sp says debye_boltzmann):
**Accept fully.** I had `set_initial_conditions(ctx, sp)` reading the
unchanged `initializer="debye_boltzmann"` from sp.bv_convergence and
routing right back to the analytical IC, defeating the fallback's
purpose. Correct fix: at the orchestrator's fallback branch, build
a `_params_with_phi_linear` that clones `sp.solver_options` with
`bv_convergence.initializer = "linear_phi"`, then call the dispatched
`set_initial_conditions(ctx, sp_linear)` — OR call the formulation-
specific linear-phi function directly based on
`sp.solver_options['bv_convergence']['formulation']`.

  Concrete: the orchestrator already has `_params_with_phi(eta_target)`
  at line 219 (cloning params with new phi_applied). I'll add a
  `_params_with_linear_phi_init(eta_target)` that ALSO swaps the
  initializer flag, then call `set_initial_conditions(ctx, _params_...)`.

**Re your point 2** (factual error: `_set_z_factor` mutates SAME
z Constants used in muh reconstruction; at z=0, c_H = exp(mu_H), not
exp(mu_H - phi)):
**Accept fully.** Critical correction. The muh reconstruction at the
form level uses the same `z[i]` Constant that gets z-ramped. So at
z=0:
  - electromigration drift in transport residual: `em·z·∇φ → 0` ✓
  - muh reconstruction: `c_H = exp(mu_H - em·z_H·phi)` with z_H=0 →
    `c_H = exp(mu_H)`, which for the linear-phi fallback IC's
    `mu_H = ln(H_b)` constant gives `c_H = c_H_b = 0.0833` everywhere.

So at z=0 the H+ concentration IS bulk everywhere, NOT artificially
depleted. My speculation about "wrong fallback giving accidental H+
depletion" was wrong — the Robin BC at z=0 with no Poisson source
gives phi(0)=phi_applied=21.4, but c_H_at_electrode=0.0833 (bulk).
Then BV evaluates with c_H = 0.0833, (c_H/c_H_ref)² = 1, full
exponent factor exp(α·n_e·η)=exp(33.9). R_2e cathodic ≈ 1.263e-3 ·
1.0 · 1 · 5e14 = 6.3e11 NONDIM. Truly unbounded. THAT's why z=0 fails.

And ramping z from 0 to 1 doesn't help linearly: at any z<1, the BV
rate has full exp factor (z doesn't enter the BV rate computation
directly — only through the dynamic-species charges that affect phi
via Poisson). So z-ramp's z=0 step might never converge for this
configuration regardless of fallback IC quality.

  Plan: re-evaluate after Phase 5α patch. With multi-ion-correct
  Picard → multi-ion-correct linear-phi IC → c_H_at_electrode might
  end up at exp(-psi_D)·c_H_b after the IC walks through the multi-ion
  Boltzmann shift, in which case z=0's BV rate is bounded.

**Re your point 3** (Phase 5α must patch the FIRST Picard
electrostatics computation at lines 1102-1126 too, not just the loop
update at lines 1138-1228):
**Accept fully.** I missed the pre-loop initialization. Same structural
edit: factor a `_update_electrostatics(c_s, multi_ion_ctx, stern_split,
phi_applied_model, ...)` helper that returns `(phi_o, psi_D, psi_S,
phi_surface, gamma_s, eta_drop)` and call it both at the pre-loop
init (lines 1102-1126) AND at each loop iteration (lines 1138-1228).
Cleaner than sprinkling `if multi_ion_ctx` branches.

  Plan: refactor before the multi-ion patch. The helper takes
  `multi_ion_ctx=None` for single-ion (current behavior, unchanged
  call signature) and uses the multi-ion branch when provided.

**Re your point 4** (sprinkling `if multi_ion_ctx is not None` will
make the loop brittle; prefer small helpers/closures):
**Accept fully.** Same conclusion as point 3. The helper-based
refactor is the right abstraction.

**Re your point 5** (multi_ion.py uses unclamped `_safe_exp` while
boltzmann.py clamps each ion's phi at phi_clamp):
**Accept partially with a clarification.** boltzmann.py's
`phi_clamped = fd.min_value(fd.max_value(phi, -phi_clamp), phi_clamp)`
is a UFL expression that clamps the field-point phi BEFORE evaluating
the closure exp. multi_ion.py's `_safe_exp` clamps the EXPONENT
(±cap=700.0). Different things; both prevent overflow but at
different layers.

The Picard's bisection in `solve_outer_phi_multiion` operates over
phi_o in [-50, +50] with `_safe_exp`. The boltzmann.py UFL has
`phi_clamp_val` per entry (default 50). For consistency at extreme
phi_o, the bisection should respect the same per-ion clamp the UFL
uses.

  Concrete: in `_electroneutrality_residual` and related helpers,
  apply per-ion phi clamp (not just _safe_exp) so the multi_ion
  bisection sees the same closure the residual sees. This is more
  important for higher-order calls like `effective_debye_length_local`
  where dρ/dφ is taken at phi=phi_o (potentially close to clamp).

  For the production page-15 grid (V_RHE ∈ [-0.5, +1.0] V → phi_o
  expected to stay well below ±10 in nondim units after the full
  solve), the clamp shouldn't bite. But I'll add an assertion in
  the Picard that |phi_o| stays well below phi_clamp during the
  loop, and abort early if it ever approaches.

**Re your point 6** (bisection might pick wrong root if monotonicity
breaks; use local bracket):
**Accept fully.** Two fixes:
  1. Add a monotonic scan test: at Cs/SO4 representative bulk and
     across phi ∈ [-50, +50] in steps of 0.5, verify `dρ/dφ < 0`
     everywhere. If non-monotone, error out at ctx-build time.
  2. In the Picard's per-iteration phi_o solve, pass a local bracket
     `(phi_o_prev - 5, phi_o_prev + 5)` instead of the global default.
     If that fails to bracket, fall back to global bisection (with
     the monotone test passing, this should be safe).

**Re your point 7** (1000x k0 ramp is too coarse; use 10× or 32×
with adaptive rollback/bisect):
**Accept fully.** 32× per step (= 5 steps from 1e-12 to ~3e-5, plus
2-3 more to reach K0_HAT_R2E ≈ 1.26e-3 = 8 steps total). Adaptive:
on Newton failure, rollback to previous k0, halve the multiplier,
retry. Standard continuation pattern.

**Re your point 8** (Picard reads nondim["bv_reactions"] static k0,
residual uses mutable bv_k0_funcs; metadata divergence on ramp):
**Accept fully.** This is a real bug not a hypothetical. Two paths:
  1. Build with target K0_HAT_R2E, then `bv_k0_funcs[j].assign(1e-12)`
     BEFORE first solve. Picard sees target k0 in scaled bv_reactions
     (its prefactor calc would use target k0 and compute "correct"
     Picard rates at full k0), but the residual uses the assigned
     1e-12. So Picard's R/c_s would be inconsistent with the
     residual rate. BAD.
  2. Build a `update_picard_k0(ctx, j, new_k0_model)` helper that
     mutates BOTH `ctx["nondim"]["bv_reactions"][j]["k0_model"]` AND
     `ctx["bv_k0_funcs"][j].assign(new_k0_model)`. Then per ramp
     step, update both. Picard re-runs with new k0 and produces
     consistent R/c_s. Residual sees consistent k0.

  Plan: implement option 2. Add the helper to bv_solver/multi_ion.py
  (or a new continuation.py module).

**Re your point 9** (current C+D has no k0 homotopy hook; rebuilds
context per voltage, no driver-side assignment):
**Accept fully.** Need a separate anchor-builder that does k0+dt
continuation OUTSIDE the C+D orchestrator, returns a converged
state, and feeds that as the anchor for C+D's warm-walk.

  Plan: build `solve_anchor_with_continuation(sp, V_anchor, mesh)`
  in scripts/_bv_common or as a helper in the new driver. It:
    1. Build context at V_anchor with k0_R2e_target = K0_HAT_R2E
       (no continuation in setup yet — context holds full bv_k0_funcs).
    2. `update_picard_k0(ctx, 0, 1e-12)` (assign tiny down).
    3. Set dt = 1e-4 (override `ctx["dt_constant"]` — need to check
       how dt is stored in the form).
    4. Run SS at this small-k0 small-dt state.
    5. Geometric ramp k0_R2e: 1e-12 → 1e-10 → 1e-8 → ... → K0_HAT_R2E
       with adaptive rollback (32x per step).
    6. Then ramp dt: 1e-4 → 1e-3 → 1e-2 → 0.25.
    7. Return the final converged ctx state as the anchor.

  Then C+D warm-walks from this anchor across the V_RHE grid.

**Re your point 10** (dt_init=1e-4 won't ramp to 0.25 under
default SER ratio of 20):
**Accept fully.** SER default `dt_max = dt_init · dt_max_ratio` at
grid_per_voltage.py:258. With `dt_init=1e-4 · 20 = 2e-3` max. Need
explicit override.

  Concrete: pass `dt_max_ratio=2500` (or just `dt_max_abs=0.25` if
  the orchestrator supports that param; check). Or implement a manual
  dt ladder OUTSIDE the SER mechanism in the anchor-builder.

**Re your point 11** (above-Eeq homotopy isn't a clean zero
crossing; treat as empirical voltage homotopy):
**Accept fully.** I oversimplified. The actual rate at V=E_eq has
cathodic term k0·c_O·(c_H/c_H_ref)² and anodic term k0·c_H2O2·1
(no overpotential exp factor). Net zero only if those balance, which
requires specific surface concs not the ideal. Empirical homotopy
+0.85 → +0.55 might be useful or might not — need to measure.

  Plan: keep above-Eeq as a TENTATIVE 5δ; falsify empirically if
  5α+5β+5γ doesn't reach ≥15/25.

**Re your point 12** (Stern homotopy is not one-parameter smooth;
no-Stern changes residual form):
**Accept fully.** No-Stern path uses Dirichlet BC at the electrode
(phi=phi_applied), Stern path uses Robin BC. Different residuals.
Within Stern Robin path: large finite C_S → eta ≈ -E_eq (Stern
absorbs all of phi_applied, ψ_S = phi_applied = 21.4). At
V_RHE=+0.55, eta_R2e_at_surface = -E_eq_R2e = -27.05 nondim
(strong cathodic, not bounded by H+ depletion via large psi_D
because psi_D ≈ 0 at large C_S).

  So Stern continuation is NOT useful here. Defer.

**Re your point 13** (Phase 5β byte-equivalence: logc OK if
linear_phi forced; logc_muh intentionally changed; need regression
tests):
**Accept fully.** Two test categories:
  1. logc + single-ClO4 (legacy): byte-equivalent before/after.
     Existing snapshot test (test_stern_no_stern_snapshot.py).
  2. logc_muh + single-ClO4 (legacy): semantically changed
     (linear_phi fallback now correctly routes through logc_muh
     IC builder). Should still pass the legacy 15/15 sweep.

  Plan: extend snapshot tests to cover the muh fallback path.
  May require regenerating the muh baseline if the bug fix shifts
  values within tolerance.

**Re your point 14** (Phase 5ζ diagnostics shouldn't be post-only):
**Accept partially.** Patch BEFORE relying on diagnostics for
triage; AFTER relying on Picard correctness for steering. Order:
  - Phase 5α (Picard patch) lands first.
  - Phase 5ζ (diagnostics patch) lands next, before Phase 5β.
  - Phase 5β/γ use the corrected diagnostics for failure analysis.

**Re your point 15** ("sensible R/c_s" test is too vague; add
rate-consistency test):
**Accept fully.** Concrete test: at Cs/SO4 multi-ion at +0.55 V
with k0_R2e = K0_HAT_R2E:
  1. Run patched Picard, get `R_list_picard, c_s_list_picard`.
  2. Build the residual context with the same params (no orchestrator).
  3. Set IC to bulk-flat (or Picard's c_s_list).
  4. Evaluate the BV rate expression at the boundary (assemble
     `bv_rate_exprs[j] * fd.ds(electrode_marker)`) using the IC's c_s.
  5. Compare to Picard's R_list[j] within rel_tol=1e-3 (for the
     case where Picard's c_s_list IS the IC).

  This pins that Picard's flux balance and the residual's BV rate
  agree on the same surface state. Without this test, "sensible"
  is in the eye of the beholder.

## Section 2 — Updated artifact (the recommended path, v2)

### Phase 5α: Patch Picard for multi-ion (HIGHEST PRIORITY)

Refactor: extract `_update_electrostatics(c_s, *, c_clo4_bulk,
phi_applied_model, multi_ion_ctx=None, stern_split=None, a_h, a_cl,
c_cl_anchor_kind, ...)` helper from picard_ic.py:1102-1228 returning
`(phi_o, psi_D, psi_S, phi_surface, gamma_s, eta_drop, eta_list)`.

When `multi_ion_ctx is not None`:
  - phi_o ← `solve_outer_phi_multiion(ctx=multi_ion_ctx, c_dyn_outer=c_s,
            bracket=(phi_o_prev-5, phi_o_prev+5))`
  - Per-ion outer concs ← `per_ion_outer_concs(ctx, c_s, phi_o)`
  - λ_eff ← `effective_debye_length_local(phi_o, multi_ion_ctx, c_s,
            poisson_coeff=eps)`
  - psi_D ← linear-Debye Stern matching with λ_eff
  - psi_S ← phi_applied - phi_o - psi_D
  - phi_surface ← phi_o + psi_D
  - gamma_s ← `compute_surface_gamma_multiion(H_o, psi_D, a_h,
              ions_with_outer_concs)`
  - eta_drop ← psi_S (Stern) or phi_applied - phi_o (no Stern)
  - eta_list ← `_eta_list_from_drop(eta_drop, ...)`

When `multi_ion_ctx is None` (legacy single-ion): unchanged calls
to `compute_surface_gamma`, `solve_stern_split`, `phi_o = log(H_o /
c_clo4_bulk)`. Byte-equivalence guaranteed.

Add tests:
  - `test_picard_multi_ion_csplus_so4_phi_o_at_bulk`: with c_dyn = bulk,
    Picard phi_o → 0 (within 1e-9).
  - `test_picard_single_ion_byte_equivalent`: re-run all existing
    Picard tests through the refactored path with multi_ion_ctx=None.
  - `test_picard_residual_rate_consistency_csplus_so4`: per
    your point 15. Picard's R agrees with residual's BV rate at
    the same surface state.

Also adds a monotonicity test:
  - `test_multi_ion_electroneutrality_monotone`: scan ρ(phi) over
    [-50, +50] for Cs/SO4 bulk; assert `dρ/dφ < 0` everywhere.

Per your point 5: in `_electroneutrality_residual` and all
`multi_ion.py` exp callsites, apply per-ion `phi_clamp` consistent
with the boltzmann.py UFL clamp. Add an assert in Picard that
|phi_o| < 0.9 · min(phi_clamp_per_ion) during iteration.

### Phase 5ζ: Patch multi-ion diagnostics

Refactor `surface_field_means` and the saturation diagnostic in
`diagnostics.py:195` to use `build_counterion_ctx` +
`per_ion_outer_concs`. Single producer of theta_b, shared across
counterions. Tests verify the diagnostic now reports correct
saturation for Cs/SO4.

### Phase 5β: Patch grid_per_voltage.py:350 + logc_muh linear IC

  1. grid_per_voltage.py:350: don't call dispatcher with unchanged
     sp. Instead:
     ```python
     formulation = sp.solver_options['bv_convergence'].get('formulation', 'logc')
     if formulation == 'logc_muh':
         set_initial_conditions_logc_muh(ctx, _params_with_phi(V_target_eta))
     else:
         set_initial_conditions_logc(ctx, _params_with_phi(V_target_eta))
     ```
     Both functions internally know to skip the analytical IC and
     write linear-phi (they're the LOGC-name-suffix linear-phi
     variants, distinct from `set_initial_conditions_debye_boltzmann_*`).

  2. forms_logc_muh.py:662 (linear-phi IC for muh): gate on
     `len(bikerman) > 1` and call `solve_outer_phi_multiion()` +
     `effective_debye_length_local()` for the multi-ion linear
     case. Mirror the Phase 2.4 _try_debye_boltzmann_ic_muh
     multi-ion branch but as the linear path (no Picard call).

  3. Tests:
     - logc + single-ClO4: byte-equivalent (existing snapshots).
     - logc_muh + single-ClO4: regenerate baseline once; should
       converge as before.
     - logc_muh + Cs/SO4: NEW path; tested via the smoke driver.

### Phase 5γ: Anchor builder with k0 + dt continuation

`solve_anchor_with_continuation(sp, V_anchor, mesh)` in
scripts/_bv_common.py:

  1. Build context at V_anchor with target K0_HAT_R2E.
  2. Helper `update_picard_k0(ctx, j, new_k0_model)`:
     - Mutates `ctx["nondim"]["bv_reactions"][j]["k0_model"] = new_k0_model`
     - Calls `ctx["bv_k0_funcs"][j].assign(new_k0_model)`
     Both Picard input and residual see the new k0 consistently.
  3. Initial state: `update_picard_k0(ctx, 0, 1e-12)` (Pass A pure-2e:
     R_2e is the only active rxn; R_4e = 0 hardcoded by Phase 1.1
     guard or starts at 1e-12 for Pass D).
  4. Override dt to 1e-4: mutate `ctx["dt_constant"]` (UFL Constant)
     directly.
  5. SS solve at k0_R2e=1e-12, dt=1e-4. If diverges, halve dt and
     retry.
  6. k0 ramp: 1e-12 → 32× → ... → K0_HAT_R2E (8 steps). Adaptive:
     on failure, rollback k0 and halve multiplier (16×, 8×, ...).
     Hard cap on rollback iterations.
  7. dt ramp: 1e-4 → 10× → ... → 0.25 (4 steps). Adaptive same as k0.
  8. Return `ctx` with all fields at production k0 and dt.

C+D orchestrator then warm-walks from this anchor across the V_RHE
grid using its existing warm-walk code path.

### Phase 5δ (only if 5α-5γ insufficient): empirical voltage homotopy

Anchor at V_RHE=+0.85 (above E_eq_R2E; rate dominated by anodic
direction with bounded c_H2O2). Walk to +0.55. Empirical, not
"clean zero crossing".

### Phase 5ε (last resort): legacy-ClO4 IC swap or 2-parameter
homotopy. Cheap one-hour falsification at most.

## Section 3 — Continued critique prompt

Review the v2 escalation plan. Push back on my acceptances if I
backtracked weakly, and re-issue any earlier point I addressed
shallowly.

Specifically:
  - Phase 5α refactor: is `_update_electrostatics` helper the right
    name and signature? Or should it be split into
    `_solve_phi_o`, `_solve_stern_split`, `_compute_gamma_s`
    (three separate helpers)?
  - The k0+dt continuation order (k0 first, then dt) — is k0-first
    correct? Or should dt go first (small dt = transient damping)
    and then k0? Argument for dt-first: at small dt the residual
    has a strong (c-c_old)/dt term that effectively biases toward
    the previous state, helping Newton. Argument for k0-first:
    BV rate is the actual blow-up source; reducing it directly is
    surgical.
  - Is there a case for combining: dt small AND k0 small simultaneously
    at step 1, then ramp them on independent schedules?
  - The rate-consistency test (your point 15): is "Picard R agrees
    with residual BV rate at the same surface state to rel_tol=1e-3"
    too tight? The Picard uses scalar arithmetic, the residual uses
    UFL+Firedrake with FEM quadrature on a graded mesh. Some drift
    is expected. Suggest a tolerance.

  - Regression test discipline for Phase 5β logc_muh fallback:
    do you trust regenerating the muh baseline as part of this
    patch, or is that a yellow flag we should treat with care
    (the existing 15/15 sweep result is load-bearing for
    docs/4sp_bikerman_ic_option_2b_results.md)?

Same numbered format and same verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
