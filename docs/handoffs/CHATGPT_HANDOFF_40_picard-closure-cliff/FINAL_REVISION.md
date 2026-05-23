# Critique Session 40 — Final Revision Ledger (post-round-7 APPROVED)

- **Session:** `picard-closure-cliff`
- **Rounds completed:** 7 (extended cap from 5 → 10; APPROVED at round 7)
- **Final verdict:** APPROVED with 3 non-blocking nitpicks (all addressed)
- **Original artifact:** `.planning/jithin_picard_plan/PLAN.md` (revised in-place)
- **Total issues across rounds:** 81 (R1: 25 + R2: 18 + R3: 12 + R4: 10 + R5: 6 + R6: 4 + R7: 3 + minor wrap-up)
- **All accepted:** 81 / 81 ; 0 defended ; 0 unresolved

## Round-by-round verdict trajectory

| Round | Issues raised | Verdict | Theme |
|-------|---------------|---------|-------|
| 1 | 25 | ISSUES_REMAIN | Math/sign/units; architecture (callback-Picard wrong); closure design (lost packing); plumbing (ctx access) |
| 2 | 18 | ISSUES_REMAIN | `make_run_ss` not invented dt-march; fork not replace; algebra slip on residual; integrated-vs-averaged rate; θ=1 path; sign conventions; tests |
| 3 | 12 | ISSUES_REMAIN | Wrap `run_ss` not `warm_walk_phi`; recovery without target; explicit factory args; PreconvergedAnchor frozen-extension; signed-R policy; state norm units |
| 4 | 10 | ISSUES_REMAIN | `warm_walk_phi` owns `make_run_ss` (not `solve_grid_with_anchor`); factory kwargs match exactly; PicardResult propagation; ctx checkpoint includes U_prev; anchor ladder also needs factory; layer separation |
| 5 | 6 | ISSUES_REMAIN | warm_walk rollback includes ξ; anchor rollback includes ξ; converged_overall uses final accepted attempt only; entry-state restore contract; None normalization at every entry; self-generated fixture |
| 6 | 4 | ISSUES_REMAIN | Adapter for ctx-specific factory args; state_norm iter-to-iter not entry-to-current; reuse `snapshot_U`/`restore_U`; byte-equiv unit assertion |
| 7 | 3 | **APPROVED** | Capture post-normalization in spy test; D-config staleness assert; narrow non-Picard inertness test to Picard-specific keys |

## Addressed (every R1-R7 accepted issue, ordered by category)

### Math / sign / units (R1: #1, #2, #12, #19, #20, #24; R2: #5)
All seven incorporated.  Final math § uses positive `R_O2_hat ≥ 0`,
correct residual `ξ + R·I/D − c_b/θ_b` (no spurious θ_OHP factor),
code-convention `exp(−α·n·η)` with cathodic η<0, `R_j_mean_hat = ∫R·ds/∫1·ds`
surface mean, `R_O2_hat = Σ_j (−stoich[O₂,j])·R_j_mean`, demoted "Jithin
Eq 4.31" to "continuum-MPNP analog", `abs()` cliff diagnostic ratio.

### Closure design (R1: #8, #9, #11)
Scalar supply variable ξ with `log_c_cat = ln(packing) + xi_func_s`;
init at `log(c_b_hat / θ_b_const)` (no-flux equilibrium); per cathodic
species (shared across reactions) not per reaction.

### Architecture / insertion point (R1: #3, #4, #5, #6, #7, #10; R2: #1, #2, #3, #4, #8; R3: #1, #3; R4: #1, #2, #6; R5: #1, #2, #3, #4, #5; R6: #1, #3)
Final architecture: factory pattern wrapping `make_run_ss` itself.
`make_picard_run_ss_factory(picard_config)` returns a
`make_run_ss`-compatible adapter that pulls ctx-specific Picard
objects (xi_funcs, packing_expr, etc.) FRESH from each ctx at call
time.  `_normalize_make_run_ss_factory` helper handles None → bare at
every public entry point.  `warm_walk_phi` is the actual call site
(not `solve_grid_with_anchor`); ξ-aware `ckpt_inner`/`ckpt_outer`;
anchor ladder factory plumbing through all 10 sites in
`anchor_continuation.py`; `PreconvergedAnchor` extended with defaulted
`xi_snapshots: tuple = ()` field for frozen-dataclass compatibility.
Reuses existing `snapshot_U`/`restore_U` rather than inventing a
parallel `.dat` serialization.

### Convergence / robustness (R1: #14, #15, #16, #17, #23; R2: #6, #10, #11, #16, #17; R3: #2, #4, #5; R4: #3; R5: #6; R6: #2)
Four-way convergence check (residual + step + state_norm + run_ss_ok)
with iter-0 special case (skip state_norm on first post-initial-solve
check, since no prior Picard iter exists).  Semi-implicit positivity-
preserving Eq B `ξ = (c_b/θ_b)/(1+K·I·θ/D)` avoids flooring needs.
State_norm defined as Picard-iter-to-iter L∞ DOF norm at SAME voltage
(not entry-to-current).  Strict-floor mode with `floor_tol = 1e-10`.
Rollback on Newton fail snapshots `(U, U_prev=via restore_U, xi)`;
halve damping, retry up to 3.  No damping retry on first-iter failure
(warm-walk bisects instead).  PicardResult propagation via
`ctx["_picard_run_ss_history"]` list (cleared per V).  Self-generated
test fixture (not untracked StudyResults files).

### Signed-R / charged-species safeguards (R2: #9, #12; R3: #6)
formulation="logc_muh" required; z=0 cathodic species required;
R_O2_hat < 0 → ValueError (irreversible cathodic scope).

### Diagnostics + reporting (R1: #18, #22; R2: #13, #14)
Eq B labeled as Picard target for ξ (not full coupled fixed point);
14 tests planned + 2 from R7; runtime x-invariance check demoted to
setup-level assertion (DG0 binning out of scope).

### Test correctness (R1: #21, #25; R2: #7, #15, #18; R3: #7, #8, #9; R4: #4, #5, #7, #8, #9, #10; R6: #4)
Scope adjusted: parallel 2e/4e is smoke test in this plan, full deck
follow-up.  H+ closure quality recorded per Picard iter as gating
diagnostic.  no-flux test uses `enabled=False` reaction.  V naming
corrected (V=+0.60 is cathodic).  L∞ on raw DOF arrays for state norm.
`PreconvergedAnchor` defaulted xi_snapshots tuple.  Source pool entries
captured AFTER Picard converges.  No file I/O in `closure_picard.py`.
Tests use self-contained fixtures.  Parallel test runs to convergence
before asserting.  Complementary tests cover both enabled and disabled
reaction paths.  Unit test on normalization helper added (R7-aligned).

### R6 deltas
- **R6#1 (interface adapter):** `PicardConfig` dataclass holds
  ctx-invariant Picard knobs; `make_picard_run_ss_factory(picard_config)`
  returns adapter that pulls ctx-specific objects fresh on each call.
- **R6#2 (state_norm iter-to-iter):** explicit iter-0 vs iter-N≥1
  convergence checks; state_norm only applies once a prior Picard
  iter exists.
- **R6#3 (reuse snapshot_U/restore_U):** StateSnapshot wraps
  `snapshot_U(ctx["U"])` + xi tuple; restore via `restore_U` (which
  also sets U_prev=U for time-stepping consistency).
- **R6#4 (byte-equiv unit assertion):** module-level
  `_normalize_make_run_ss_factory` helper + dedicated unit test
  asserting identity check on `is make_run_ss`.

### R7 deltas (non-blocking nitpicks, all addressed)
- **R7#1 (spy test post-normalization):** test the
  `_normalize_make_run_ss_factory` helper directly via identity
  assertion; bypasses spy-timing ambiguity in `warm_walk_phi`.
- **R7#2 (D-config staleness):** factory adapter asserts
  `picard_config.D_per_species_hat[s] == ctx logD value` at call
  time; raises if stale (catches future logD mutation).
- **R7#3 (narrow inertness test):** test 16 only asserts
  Picard-specific keys (`picard_log_xi_funcs`, `closure_*`,
  `_picard_run_ss_history`) absent in non-Picard mode; `packing_expr`
  and `theta_inner_expr` may be exposed by other diagnostics without
  breaking the contract.

## Defended

None — every issue across 7 rounds was substantive and accepted.

## Unresolved

None.  All issues addressed in PLAN.md.

## Notable observations about the extended loop

- **Critique substance converged steadily**: 25 → 18 → 12 → 10 → 6 → 4 → 3.
  Each round surfaced finer-grained issues; by round 7 the remaining
  concerns were narrow contracts (test mechanics, staleness asserts,
  inertness scope).
- **No defenses needed**: every objection was either a real bug, a real
  under-spec, or a real test-correctness concern.  This is consistent
  with the loop's purpose — gap-finding rather than appeasement.
- **GPT excelled at**: catching architecture mistakes (Picard insertion
  point was wrong twice before settling on `make_run_ss` wrap);
  algebra slips (residual θ_OHP factor); unit conflations (current
  density vs molar rate); contract gaps (entry-state restoration); and
  unexpected coupling (state_norm semantics depending on what's being
  compared to what).
- **Where GPT was wrong**: no instances in this session — all 81 issues
  held up under counterreply.  (Notable absence of false positives.)

## Implementation gate

The revised PLAN.md is approved and ready for implementation.

**Estimated effort:** ~4 days of focused work.
- Day 1: `closure_picard.py` with `PicardConfig`, `StateSnapshot`,
  `make_picard_run_ss_factory`, helpers; unit tests for these in isolation.
- Day 2: form-build wiring in `forms_logc_muh.py` (closure substitute,
  ctx population, z=0 gate); test 2 (function update no rebuild),
  test 4 (θ=1 Levich), test 8 (z!=0 reject).
- Day 3: factory plumbing in `grid_per_voltage.py` and
  `anchor_continuation.py`; byte-equivalence tests 10-11 +
  `_normalize_make_run_ss_factory` test 15.
- Day 4: study script `_run_jithin_closure_picard.py`; run + analyze;
  iterate on observed Picard behavior.

**Decision point after run:** if smooth-S-curve-to-Levich (predicted),
write up as solver-correctness validation + close out continuum-Bikerman
as the cliff mechanism; pivot to path 2 (surface-coverage / non-covalent
cation effect).  If cliff appears: re-examine I integral and θ(y)
profile assumptions; further iterations.
