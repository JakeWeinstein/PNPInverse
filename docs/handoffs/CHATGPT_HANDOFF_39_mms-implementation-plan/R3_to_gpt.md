# Critique session 39 — Round 3

## Section 1: Acknowledgment

All 14 R2 issues accepted. Issues 1, 4, 7, 8 are pushback on R1 fixes I
over-corrected; that's fair — the right move is to walk those back to
the more targeted versions GPT proposed.

### Re point 1 — drop `newton_iterations ≥ 2` floor on unperturbed runs

**Accept.** The perturbed-IC pilot (R1 point 7's mitigation) is the
right defense; piggybacking an iteration floor on the normal runs adds
noise without value. Removed. Iteration counts are still recorded in
`run_mms_result["newton_iterations"]` for diagnostic logging.

### Re point 2 — deterministic smooth perturbation, not random

**Accept.** Replaced perturbed IC spec:

```python
def _perturbed_initial_guess(U_manuf, mesh, eps=1e-3):
    """Deterministic smooth perturbation that vanishes on essential-BC boundaries."""
    x, y = fd.SpatialCoordinate(mesh)
    # Perturbations vanish at y=0 and y=1 (electrode + bulk) so Dirichlet BCs
    # on u_O2, u_H2O2, μ_H, φ at y=1 are not violated.
    # (Electrode has no Dirichlet on c_i or φ in the Stern-on branch, so
    # any perturbation is admissible there, but using sin(π·y) is cleaner.)
    pert = fd.Constant(eps) * fd.sin(pi*x) * fd.sin(pi*y)
    U_init = U_manuf.copy(deepcopy=True)
    for i in range(ctx['n_species']):
        U_init.sub(i).interpolate(U_init.sub(i) + pert)
    # phi component: also perturbed by the same expression
    U_init.sub(phi_idx).interpolate(U_init.sub(phi_idx) + pert)
    # Apply Dirichlet BCs to U_init so essential constraints are respected.
    for bc in ctx['bcs']:
        bc.apply(U_init)
    return U_init
```

Applied only in the perturbed-IC pilot (step 10.5), not in the normal
convergence-test runs.

### Re point 3 — residual reduction > 100× only in perturbed pilot

**Accept.** Moved from "every mesh in run_mms" to "perturbed-IC pilot
only". Step 10.7 reworded:

> **10.7 Residual-reduction check (perturbed-IC pilot only):** at N=32
> with the perturbed IC from 10.5, record
> `||F_res||_L2(U_init) / ||F_res||_L2(U_final)`. Reduction must be > 100×.
> Skip this check if `||F_res||_L2(U_init) < 1e3 · snes_atol` (initial
> residual is already small enough that the ratio is noise).

### Re point 4 — closure algebra smoke test on same inputs

**Accept.** For the smoke test, build a SECOND version of the
independent closure that takes interpolated fields as inputs (not
analytic). Compare the production bundle against this
"interpolated-input independent closure" to within 1e−9. The actual
MMS source builder still uses the analytic-input version.

```python
# Smoke test only — separate from MMS source builder
def _independent_closure_smoke(ctx, counter_cfg):
    """Recompute the multi-ion shared-θ closure using ctx['ci_exprs']
    and ctx['phi'] (interpolated). Compares to ctx['steric_boltzmann']
    bundle to verify algebraic agreement."""
    ci_h    = ctx['ci_exprs']             # interpolated c_i UFL
    phi_h   = fd.split(ctx['U'])[ctx['mixed_space_indices'].phi_index]
    a_dyn   = [...]; c0_dyn = [...]; z_dyn = [...]
    closure = _build_shared_theta_closure_ex(
        counterions_cfg=counter_cfg, a_dyn=a_dyn, c0_dyn=c0_dyn,
        phi_ex=phi_h, c_dyn_ex=ci_h,
    )
    return closure
```

After `U.assign(U_manuf)`, compare bundle outputs to closure outputs
on the same `(U_manuf_h, phi_h)`. Demand `1e−9` relative agreement (FP
noise floor).

### Re point 5 — bundle exposes only c_steric/packing/charge_density

**Accept.** Limited the smoke test to those three exposed quantities:

- `c_steric_ex` vs `bundle.c_steric_expr`
- `P_k_ex = a_k · c_steric_ex` vs `bundle.packing_contribution`
- `ρ_k_ex = z_k · c_steric_ex` vs `bundle.charge_density`

θ_inner_ex and μ_steric_ex are derived quantities; if c_steric agrees,
they automatically agree (modulo the `max(·, packing_floor)` clamp,
which is identity at u_exact per the margin check). Document this in
the smoke-test docstring.

### Re point 6 — UFL independence check via base mixed function

**Accept.** Reworked the check:

```python
from ufl.algorithms.analysis import extract_coefficients

ALLOWED_LIVE_COEFFS = {
    ctx['phi_applied_func'],
    ctx.get('stern_coeff_const'),
    *ctx.get('bv_k0_funcs', []),
    *ctx.get('bv_alpha_funcs', []),
    ctx.get('boltzmann_z_scale'),
}
ALLOWED_LIVE_COEFFS.discard(None)
FORBIDDEN_COEFFS = {ctx['U'], ctx['U_prev']}

for label, S in zip(SOURCE_LABELS, S_terms):
    coeffs = set(extract_coefficients(S))
    bad = coeffs & FORBIDDEN_COEFFS
    assert not bad, f"Source {label} depends on {bad} — violates independence"
    # Inform-only: log any non-whitelisted coefficient for audit
    unknown = coeffs - ALLOWED_LIVE_COEFFS - {fd.Constant(0.0), ...}
    if unknown:
        print(f"[MMS] source {label}: unrecognized coefficients {unknown}")
```

### Re point 7 — DG projection smooths extrema; use `fd.interpolate`

**Accept.** All `_dg_proj_min/max` helpers replaced with `fd.interpolate`
versions:

```python
def _expr_min(expr, mesh, *, degree=4):
    """Pointwise min via DG-k interpolation. Avoids the L2-projection
    smoothing that fd.project introduces."""
    P = fd.interpolate(expr, fd.FunctionSpace(mesh, 'DG', degree))
    return float(P.dat.data_ro.min())

def _expr_max(expr, mesh, *, degree=4):
    P = fd.interpolate(expr, fd.FunctionSpace(mesh, 'DG', degree))
    return float(P.dat.data_ro.max())

def _expr_abs_max(expr, mesh, *, degree=4):
    P = fd.interpolate(expr, fd.FunctionSpace(mesh, 'DG', degree))
    return float(abs(P.dat.data_ro).max())
```

Plus the quadrature-indicator companion check (catches small-region
violations between DG DOFs):

```python
def _expr_indicator_measure(expr, threshold, mesh, *, comparison='lt', degree=8):
    """Measure of the set where expr {<,>} threshold, via high-degree quadrature."""
    cond = fd.lt(expr, threshold) if comparison == 'lt' else fd.gt(expr, threshold)
    return float(fd.assemble(
        fd.conditional(cond, fd.Constant(1.0), fd.Constant(0.0))
        * fd.dx(domain=mesh, degree=degree)
    ))
```

Used together: `assert _expr_min(...) > T and _expr_indicator_measure(..., T) < 1e-12 · vol`.

### Re point 8 — centralized helpers, no `fd.assemble(fd.dot(...))` scalar tricks

**Accept.** All clamp/floor margin checks use the helpers from point 7
above. Concrete rewrite for η-margin:

```python
phi_app_model = float(scaling['phi_applied_model'])
E_eq_R2e = float(r2e['E_eq_model'])
E_eq_R4e = float(r4e['E_eq_model'])
bv_exp_scale = float(scaling['bv_exponent_scale'])

# eta_raw_j on the electrode is x-dependent through phi_ex(x,0); evaluate
# the scalar UFL and take its absolute max via interpolation.
eta_R2e_expr = bv_exp_scale * (phi_app_model - phi_ex - E_eq_R2e)
eta_R4e_expr = bv_exp_scale * (phi_app_model - phi_ex - E_eq_R4e)
exp_clip = float(conv_cfg['exponent_clip'])
assert _expr_abs_max(eta_R2e_expr, mesh, degree=4) < 0.9 * exp_clip
assert _expr_abs_max(eta_R4e_expr, mesh, degree=4) < 0.9 * exp_clip
```

(Note: at the recommended envelope, |η_R4e| ≈ 27, so 0.9·100 = 90
gives ~3.3× safety margin.)

### Re point 9 — assert R2e/R4e `k0_model` and `enabled`

**Accept.** Added to reaction-identity asserts:

```python
assert pytest.approx(float(r2e['k0_model']), rel=1e-9) == K0_HAT_R2E
assert float(r2e['k0_model']) > 0.0
assert bool(r2e.get('enabled', True)) is True

assert pytest.approx(float(r4e['k0_model']), rel=1e-9) == K0_HAT_R4E * K0_R4E_FACTOR_MMS
assert float(r4e['k0_model']) > 0.0
assert bool(r4e.get('enabled', True)) is True
```

(Also addressing point 11 below — these use `pytest.approx` only in the
test class. The source-builder copy uses `_assert_close`.)

### Re point 10 — assert dynamic species identity

**Accept.** Added:

```python
assert ctx['n_species'] == 3
z_vals = list(scaling.get('z_vals', []))
assert z_vals == [0, 0, 1]
roles  = list(scaling.get('species_roles', []))
assert roles == ['neutral', 'neutral', 'proton']

a_vals = list(scaling.get('a_vals_hat', []))
assert pytest.approx(a_vals[0], rel=1e-9) == A_O2_PHYSICAL
assert pytest.approx(a_vals[1], rel=1e-9) == A_H2O2_PHYSICAL
assert pytest.approx(a_vals[2], rel=1e-9) == A_HP_PHYSICAL

c0_vals = list(scaling.get('c0_model_vals', []))
assert pytest.approx(c0_vals[0], rel=1e-9) == C_O2_HAT
assert pytest.approx(c0_vals[1], rel=1e-9) == H2O2_SEED_NONDIM
assert pytest.approx(c0_vals[2], rel=1e-9) == C_HP_HAT
```

(Exact ctx key names verified at code time.)

### Re point 11 — `pytest.approx` dependency leak in verification script

**Accept.** In `scripts/verification/mms_pnpbv_muh_multi_ion_stern.py`,
use a local helper:

```python
import math

def _assert_close(name: str, got: float, expected: float,
                  *, rel: float = 1e-9, abs_tol: float = 0.0) -> None:
    if not math.isclose(got, expected, rel_tol=rel, abs_tol=abs_tol):
        raise AssertionError(
            f"_assert_close({name}): got {got!r}, expected {expected!r} "
            f"(rel_tol={rel}, abs_tol={abs_tol})"
        )
```

The test files (`tests/test_mms_logc_muh_multi_ion_stern.py`) can still
use `pytest.approx` since they're pytest-only.

### Re point 12 — rate-test NaN handling unsafe

**Accept.** The shared fixture asserts all_converged BEFORE returning:

```python
@pytest.fixture(scope="class")
def mms_results(self):
    res = run_mms(self.MESH_SIZES, verbose=True)
    if not all(res["newton_converged"]):
        failed = [N for N, c in zip(res["N"], res["newton_converged"]) if not c]
        pytest.fail(f"MMS Newton failed on meshes: {failed}")
    return res
```

Downstream `test_l2_convergence_rates`, `test_h1_convergence_rates`,
etc. inherit the precondition: by the time they see `res`, all meshes
have converged. No NaN-skip path.

(Diagnostic callers that explicitly want partial results can call
`run_mms` directly; the fixture is for the convergence test only.)

### Re point 13 — TestMMSAsserts directly hits invariant harness

**Accept.** Factored a thin harness:

```python
def _prepare_mms_context_for_asserts(sp, mesh):
    """Build ctx + forms + manuf + closure + rxn_rates, then call
    _assert_stack_invariants(...). Stops before solver construction —
    intended for parametrized invariant-coverage tests, not for solving."""
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    manuf = _make_manufactured_fields(mesh, ctx, sp)
    closure = _build_shared_theta_closure_ex(...)
    rxn_rates = _build_bv_rates_ex(...)
    _assert_stack_invariants(ctx, sp, manuf=manuf, closure=closure,
                              rxn_rates=rxn_rates)
    # NO solver setup, NO solver.solve() — invariant failures raise here.
    return ctx, manuf, closure, rxn_rates


class TestMMSAsserts:
    @pytest.mark.parametrize("label,mutate,expected_match", BROKEN_CONFIGS)
    def test_invariant_catches(self, label, mutate, expected_match):
        sp = make_sp_production_muh()
        sp = mutate(sp)
        mesh = fd.UnitSquareMesh(8, 8)
        with pytest.raises(AssertionError, match=expected_match):
            _prepare_mms_context_for_asserts(sp, mesh)
```

`expected_match` is a regex per broken config, so the test verifies the
right invariant fired (not just any AssertionError).

For configs that fail in `build_forms` rather than at the invariant
layer (e.g. `missing_reaction` may raise `ValueError` from
`forms_logc_muh.py`), use `pytest.raises((AssertionError, ValueError))`
and assert the message names the broken aspect.

### Re point 14 — xdist: pick one now

**Accept.** Decision: **mark the test module serial; no xdist support**.
Rationale: the existing slow MMS tests aren't xdist-safe either, the
benefit of xdist on a 4-mesh convergence sweep is marginal, and per-worker
cache dirs add complexity for a low payoff.

Concrete:

```python
# tests/test_mms_logc_muh_multi_ion_stern.py
pytestmark = pytest.mark.serial   # honored by pytest-forked if used, ignored otherwise
```

In `conftest.py` (existing): if `--forked` is set or xdist is detected,
honor the `serial` marker by routing the test to a single worker. If
the project doesn't already have such a mechanism, document in the
test file that the test is not xdist-safe:

```python
"""...
NOTE: This test is NOT xdist-safe. Run as `pytest -p no:xdist ...`
or accept that under xdist it may contend with itself over
/tmp/firedrake-tsfc/ and StudyResults/ outputs.
"""
```

`os.environ.setdefault(...)` removed — was a half-measure.

---

## Section 2: Updated artifact deltas

Beyond the per-issue fixes above, the consolidated plan now has:

- **Single source for clamp/floor margin helpers**: `_expr_min`, `_expr_max`,
  `_expr_abs_max`, `_expr_indicator_measure`. All margin checks use these.
- **Tightened reaction asserts**: identity (E_eq, α, n_e, reversible,
  stoichiometry, conc-factor, anodic species) + magnitudes (k0_model,
  enabled).
- **Tightened species asserts**: n_species, z_vals, roles, a_vals_hat,
  c0_model_vals.
- **Smoke-test comparison done on interpolated inputs** (not analytic),
  with bundle's exposed quantities only (c_steric, packing_contribution,
  charge_density).
- **Independence check** via `extract_coefficients` against base mixed
  `ctx['U']`, with allowed-live whitelist.
- **No `pytest.approx` in `scripts/verification/`** — local `_assert_close`.
- **Convergence-fixture pre-asserts Newton on all meshes** before
  returning; no NaN-skip path.
- **`TestMMSAsserts` calls direct invariant harness** (`_prepare_mms_context_for_asserts`),
  no solver invocation.
- **Module marked serial / not xdist-safe** with docstring note.
- **Perturbed-IC pilot uses deterministic smooth perturbation** (sin·sin),
  BCs reapplied to U_init.
- **Residual-reduction check confined to perturbed-IC pilot**, with absolute
  initial-residual floor.
- **Unperturbed iteration-count floor REMOVED**.

(The fully-revised PLAN.md will land via the auto-revise step after the
loop exits. For this round, the deltas above need pressure-testing.)

---

## Section 3: Continued critique prompt

Review my responses to your 14 issues. Push back where I addressed
something superficially. Raise any new issues these changes create.
Re-issue any earlier issue you don't think I addressed. Same numbered
format and same verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
