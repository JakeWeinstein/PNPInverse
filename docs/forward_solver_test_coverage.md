# Forward-solver testing: coverage and known holes

**Last updated:** 2026-05-03
**Scope:** What our test suite verifies about the production 3sp + Boltzmann
ClO4- + log-c + log-rate BV stack, and what it does not.

## Current state

After the May 2026 cleanup the forward-correctness test suite consists of:

| Test file | What it verifies | Runtime |
|---|---|---|
| `tests/test_mms_convergence.py::TestMMSConvergence` | h^p convergence (L2 ~ h^2, H1 ~ h^1) of the production weak form on `UnitSquareMesh(N, N)` for N=[8,16,32,64] | ~5 s |
| `tests/test_mms_convergence.py::TestMMSProductionGradedMesh` | Single-mesh recovery on the production graded rectangle (Nx=8, Ny=200, beta=3) | ~5 s |
| `tests/test_solver_equivalence.py::TestSolverEquivalence4spVs3spBoltzmann` | 3sp+Boltzmann vs 4sp dynamic produce equivalent CD/PC on V_RHE in [-0.5, +0.1] V | ~2 min |
| `docs/forward_cleanup_may2026.md` (bit-exact baseline) | Production driver at Ny=200 reproduces pre-cleanup output bit-for-bit | one-shot |

These collectively exercise:

- The production code path through `Forward.bv_solver` dispatcher → `forms_logc.py`.
- The `make_bv_solver_params` factory in `scripts/_bv_common.py` with both
  `THREE_SPECIES_LOGC_BOLTZMANN` and `FOUR_SPECIES_LOGC_DYNAMIC` presets.
- The C+D orchestrator (`solve_grid_per_voltage_cold_with_warm_fallback`) end-to-end.
- Log-rate BV evaluation (bv_log_rate=True wiring).
- Boltzmann counterion residual (`boltzmann.py`) sign and magnitude.
- Physical E_eq scaling (R1=0.68, R2=1.78 V) into the nondim transform.
- Real production parameter values (D, k0, alpha, c0, A_DEFAULT).

## Tier 1 — Holes that actually leave doubt

These are gaps where we cannot rule out a forward-solver bug from existing tests
alone.  Listed in order of severity / leverage.

### 1. Clipping/clamping behavior is unverified

The eta-clip at +/-50 fires for R2 below `V_RHE = +0.495 V` (see
`docs/clipping_conventions.md`).  The production grid extends down to
`V_RHE = -0.5 V`, so most production solves are in the clipped regime for R2.

MMS deliberately keeps the clip inactive (it's non-smooth and would break
h^p convergence).  The bit-exact baseline only verifies "current
implementation == prior implementation" — it does not verify "implementation
is correct".  A clipping bug introduced before the baseline would persist.

**Closing test:** Tafel-slope assertion.  At V_RHE much less than +0.495 V the
cathodic Tafel slope `d(log|CD|)/dV` should reflect only R1's transfer
coefficient (R2's exponent is frozen by clip).  At V_RHE much greater than
+0.495 V it should reflect both.  Three voltage points + linear regression
gives the slope; the prediction is fully analytic from BV theory and
production constants.

**Cost:** ~50 lines of test, runs in seconds (uses cached equivalence-test
results or a separate small-grid solve).

### 2. Mass / current consistency is never asserted

`Forward.bv_solver.observables._build_bv_observable_form` has multiple
modes (`current_density`, `peroxide_current`, etc.) that compute
observables from BV rates.  At steady state, CD computed from BV rates
must equal CD computed from electrode flux of charge carriers,
`Σ z_i F · J_i · n` — that is the discrete charge-conservation law.

We assemble one mode in our tests and never cross-check it against the
other.  A bug in one mode (or a mismatch between BV rates and the
species-flux summation) would not be detected.

**Closing test:** One voltage, two assembly modes, assert agreement to
floating-point precision (~1e-12).  Catches an entire class of wiring
bugs in observable extraction.

**Cost:** ~30 lines, runs in seconds.

### 3. Adjoint gradients have 6 pre-existing failing tests

Per `docs/forward_cleanup_may2026.md`:

> `pytest -m "not slow"`: 315 pass; 6 fail in `test_autograd_gradient.py`
> + `test_multistart.py` — confirmed pre-existing by re-running on
> `pre-cleanup-2026-05-02`

The entire inverse pipeline rides on these gradients.  CLAUDE.md hard
rule #1:

> Always use adjoint gradients for inference. Do not switch to
> derivative-free as a workaround when an adjoint-based optimizer fails.

But we don't actually verify the adjoints work in CI.  The cold-ramp
finite-difference check in `scripts/studies/v19_lograte_extended_adjoint_check.py`
exists as a manual study, not as a regression test.

**Closing test:** Diagnose the 6 fails and either fix or document.  Until
those pass, "the forward solver gives correct adjoints" is unverified —
which means the inverse pipeline's correctness is unverified by the test
suite.

**Cost:** Unknown; depends on what the failures are.  This is the
biggest item on the list.

## Tier 2 — Probably fine but not directly tested

### 4. Time-stepping wiring

MMS uses `dt = 1e15` to neutralize the `(c - c_old)/dt` term.  Production
runs a transient toward steady state with `dt = 0.25` and adaptive growth.
A sign or scaling bug in the time term would not be detected by either
the MMS tests (where the term vanishes) or the equivalence test (which
compares only steady-state observables).

The orchestrator's adaptive transient does exercise the term, but only
the *endpoint* is verified — not the trajectory.

**Closing test:** Short transient MMS — manufacture
`u_i(x, y, t) = u_steady(x, y) + delta_t * t * sin(...)`, run a few time
steps, verify trajectory L2 error converges in dt at the expected order.

**Cost:** ~150 lines; would require a small refactor of the MMS source
construction to handle time-dependent manufactured fields.

### 5. Internal field correctness

The equivalence test compares CD and PC only.  If the forward solver
gets internal fields (`phi`, `c_O2`, `c_H2O2`, `c_H+` profiles) wrong but
observables happen to come out right (canceling errors at the
electrode), we would not notice.

**This matters specifically for the inverse pipeline** — adjoint
gradients propagate through the internals, so observable agreement does
not imply gradient agreement.

**Closing test:** Extend the equivalence test to compare full fields
between 3sp+Boltzmann and 4sp dynamic at each voltage, not just CD/PC.
Cheap addition; the fields are already in `ctx['U']` after each
converged solve.

**Cost:** ~20 lines added to the existing equivalence test.

### 6. Cold-start basin of attraction

MMS initializes Newton at `U_manuf` and converges in 2–3 steps.
Production uses a real cold start (constant `u_i = ln(c0_i)`, linear `phi`).
The equivalence test does use cold start through the orchestrator, so
this is partly covered — but only at TRUE parameters.

A perturbed-parameter cold start (which the inverse pipeline does on
*every* gradient eval) might fail in ways we do not catch.

## Tier 3 — Lower-priority structural concerns

### 7. Symmetric-error masking in MMS source

The MMS source reads parameter values directly from
`THREE_SPECIES_LOGC_BOLTZMANN` and `_bv_common.py` constants (D, k0,
alpha, c0, A_DEFAULT).  If any of those have a bug — say a wrong
`D_HP_HAT` value — both production and MMS source see the same wrong
value, and the test passes despite the bug.

The test verifies *the discrete operator implements the configured math
correctly*, not *the configuration is right*.  Partial mitigation
because the residual *structure* is derived from first principles
independently in the test.

**Closing test:** A "physical sanity" assertion that uses an independent
analytic ground truth — e.g., assert `i_lim` at V_RHE = -0.5 V is within
a few percent of `2 * F * D_O2 * c_O2_bulk / L_REF` (the
mass-transport-limited current density).  Doesn't read solver constants
beyond the published physical values.

### 8. Multi-parameter convergence robustness

All tests run at TRUE `(k0, alpha)`.  The inverse pipeline needs the
forward to converge robustly at *perturbed* parameters, sometimes far
from TRUE.  No test sweeps parameters and asserts the C+D orchestrator
still hits all expected voltages.

### 9. Graded-mesh refinement

The graded-mesh test is a single solve.  We can't verify h^p convergence
on the production mesh topology — only on `UnitSquareMesh`.  If graded
refinement breaks the FE order (e.g., through an aspect-ratio interaction
with the integrator), we wouldn't detect it.

### 10. Wrong-marker BC silent pass

The MMS manufactured solution `c_i(x, 1) = c0_HAT[i]` and `phi(x, 1) = 0`,
`phi(x, 0) = eta` happens to be consistent with the production Dirichlet
BCs at the *correct* markers.  If `concentration_marker` and
`electrode_marker` were swapped in the dispatcher, the wrong BC would
impose the same numeric values at the wrong boundaries — and our
manufactured solution might still satisfy them by coincidence.

**Closing test:** Asymmetric manufactured solution where the BC values
differ between top and bottom in a way that distinguishes the two
markers.

## Tier 4 — Structural, not testable by MMS

### 11. Continuum-equivalence proofs

We trust that:
- `c_ClO4 = c_bulk · exp(-z*phi)` is the exact analytic solution of the
  dynamic NP equation for the non-reactive counterion under the
  production BCs (used by Boltzmann reduction).
- The log-c and log-rate transforms are mathematically equivalent to
  the concentration / direct-rate forms.
- The IBP boundary terms vanish where production has Dirichlet (because
  test functions vanish there).

These are mathematical identities, not testable as code behavior.  The
equivalence test (`tests/test_solver_equivalence.py`) verifies the first
one *empirically* by comparing 3sp+Boltzmann to 4sp dynamic.

## Recommended order of operations

If triaging by leverage / cost ratio:

1. **#2 (mass/current consistency)** — cheapest, catches an entire class
   of observable-extraction bugs.  ~30 lines, seconds of runtime.
2. **#5 (internal-field equivalence)** — cheap addition to existing test,
   probes regimes the inverse pipeline cares about.  ~20 lines.
3. **#1 (clip Tafel slope)** — moderate cost, probes the most-used
   production regime that MMS structurally cannot reach.  ~50 lines.
4. **#3 (adjoint failing tests)** — biggest project, but most important
   for the inverse pipeline's credibility.

The remaining items in Tier 2 and Tier 3 are real but lower priority
than these four.

## Pointers

- Production driver: `scripts/plot_iv_curve_unified.py`
- Production stack API: `docs/bv_solver_unified_api.md`
- Cleanup post-mortem (incl. failing tests): `docs/forward_cleanup_may2026.md`
- Clipping conventions / threshold derivation: `docs/clipping_conventions.md`
- Adjoint verification study: `scripts/studies/v19_lograte_extended_adjoint_check.py`
- Hard rules: `CLAUDE.md`
