# Critique session 39 — Round 5 (final per cap)

## Section 1: Acknowledgment

All 6 R4 issues accepted. Tight cleanup-level fixes only.

### Re point 1 — `missing_reaction` → "static"

**Accept.** Production parsing accepts any non-empty reactions list, so
the static `len(rxns) == 2` invariant is what catches a missing
reaction. Recategorized.

```python
("missing_reaction",    _drop_reaction("R4e"),  "static",  r"len\(rxns\)|n_reactions"),
```

This **adds 1 to MMS-invariant-coverage count** (was attributed to
production validation; now correctly attributed to MMS invariants).

### Re point 2 — `k0_r4e_wrong` → "static"

**Accept.** The static reaction-identity asserts already check
`k0_model == K0_HAT_R4E * K0_R4E_FACTOR_MMS`, so this fires at the
static layer. Recategorized:

```python
("k0_r4e_wrong",  _force_k0_r4e_factor(1.0),  "static",  r"k0_model.*R4e|K0_R4e"),
```

The `R_ratio` runtime invariant remains as a defense-in-depth check
(catches K0_R4e mis-set when factor differs from the static-asserted
value, e.g. someone bumped `K0_HAT_R4E` itself). Both layers
contribute coverage; the test's `expected_layer` is whichever fires
first.

After R4 points 1 + 2: 12 MMS-invariant-coverage cases (was 10),
0 production-validation-only cases (was 1). All `BROKEN_CONFIGS` rows
are now under the "static" or "runtime" expected_layer.

### Re point 3 — closure smoke: unscaled per-ion vs bundle; z_scale on totals

**Accept.** Production bundle exposes raw per-ion `c_steric_expr,
packing_contribution = a_k * c_steric, charge_density = z_k * c_steric`;
`z_scale` is applied later in `forms_logc_muh.py:445–449` (μ_steric
build) and `:656–660` (Poisson source). Closure smoke compares
**unscaled** per-ion quantities; `z_scale` only enters when
reconstructing the totals (μ_steric and Poisson charge).

```python
def _independent_closure_smoke(ctx, sp):
    """Algebra-equivalence diagnostic. Compares UNSCALED per-ion
    c_steric, a·c_steric, z·c_steric against bundle.c_steric_expr,
    bundle.packing_contribution, bundle.charge_density on the same
    interpolated input. z_scale is NOT applied to the per-ion
    quantities (mirrors bundle semantics)."""
    ...
    return {
        "per_ion_c_steric":            [...],     # unscaled
        "per_ion_packing_contribution":[...],     # = a_k * c_steric, unscaled
        "per_ion_charge_density":      [...],     # = z_k * c_steric, unscaled
    }


# Smoke comparison test
def _smoke_compare_to_bundle(ctx, smoke):
    bundles = ctx['steric_boltzmann']
    for k, (bundle, smoke_c, smoke_P, smoke_rho) in enumerate(zip(
        bundles,
        smoke['per_ion_c_steric'],
        smoke['per_ion_packing_contribution'],
        smoke['per_ion_charge_density'],
    )):
        # Compare via DG-projection .dat.data_ro for each pair.
        # Demand max(|diff|) / max(|bundle|) < 1e-9.
        _check_close(smoke_c,   bundle.c_steric_expr,        rel=1e-9)
        _check_close(smoke_P,   bundle.packing_contribution, rel=1e-9)
        _check_close(smoke_rho, bundle.charge_density,       rel=1e-9)
```

For the derived totals (μ_steric_ex, Poisson source ρ_total_ex), use
the same `z_scale` value that production reads — read it live as
`ctx['boltzmann_z_scale']` and apply it identically in the source
builder and (if compared) in the smoke. At MMS runtime `z_scale = 1.0`
so this is a no-op numerically; but the algebra is preserved for
future-proofing.

### Re point 4 — owned-constant helper, no subclassing

**Accept.** Replaced the `class _OwnedConstant(fd.Constant)` subclass
approach with a factory + set:

```python
_OWNED_COEFFS: set = set()       # populated as builders create Constants

def _owned_constant(value, label: str = "") -> fd.Constant:
    """Create an fd.Constant whose object identity is recorded so the
    independence check can whitelist it. Subclassing fd.Constant is
    avoided per Firedrake idioms; we use object identity instead."""
    c = fd.Constant(float(value))
    _OWNED_COEFFS.add(c)
    return c

def _clear_owned_coeffs() -> None:
    """Call at the start of each new mesh setup so test isolation holds."""
    _OWNED_COEFFS.clear()
```

The source builder uses `_owned_constant(z_k)` etc. in place of
`fd.Constant(z_k)` everywhere it needs a literal scalar. The
independence check is then:

```python
unknown = coeffs - ALLOWED_LIVE_COEFFS - _OWNED_COEFFS
# also filter to Functions+Constants if extract_coefficients returns
# stray base UFL objects
unknown = {c for c in unknown if isinstance(c, (fd.Function, fd.Constant))}
assert not unknown, f"Source {label} has unrecognized live deps: {unknown}"
```

`_clear_owned_coeffs()` is called at the start of `_build_manufactured_source`
on each mesh, so prior-mesh literals don't pollute the next mesh's check.

### Re point 5 — `_perturbed_initial_guess` and `Function.sub` index

**Accept.** `mixed_space_indices.phi_index` may be `-1` (relative) in
the no-Γ layout, which is valid for tuple indexing on `fd.split(U)`
but problematic for `U.sub(i)`. Resolve to a nonnegative absolute
index before `Function.sub`:

```python
def _phi_sub_index(ctx: dict) -> int:
    """Return the non-negative subspace index for the φ component.
    Resolves the relative `mixed_space_indices.phi_index` (which may
    be -1) into an absolute index for `Function.sub(i)` use."""
    indices = ctx['mixed_space_indices']
    raw = indices.phi_index
    n_subs = ctx['W'].num_sub_spaces()
    return raw if raw >= 0 else n_subs + raw


def _perturbed_initial_guess(
    ctx: dict, U_manuf: fd.Function, mesh: fd.Mesh, *, eps: float = 1e-3,
) -> fd.Function:
    x, y = fd.SpatialCoordinate(mesh)
    pert = fd.Constant(eps) * fd.sin(pi*x) * fd.sin(pi*y)
    U_init = U_manuf.copy(deepcopy=True)
    phi_idx_abs = _phi_sub_index(ctx)
    for i in range(ctx['n_species']):
        U_init.sub(i).interpolate(U_manuf.sub(i) + pert)
    U_init.sub(phi_idx_abs).interpolate(U_manuf.sub(phi_idx_abs) + pert)
    for bc in ctx['bcs']:
        bc.apply(U_init)
    return U_init
```

`_phi_sub_index` is reused everywhere the test code needs the absolute
φ subspace index for `U.sub(...)` calls.

### Re point 6 — sp[8] is already nondim

**Accept.** Reworded the docstring around the c0 assert:

```python
# c0_model_vals (nondim) lives on ctx['nondim'] after build_forms.
# sp[8] is ALSO nondim in this factory because we call
# make_bv_solver_params with concentration_inputs_are_dimensionless=True
# (default for the _bv_common factory path). We assert against
# ctx['nondim']['c0_model_vals'] because that's the value the residual
# actually reads at form-build time.
c0_vals = list(ctx['nondim']['c0_model_vals'])
_assert_close('c0[O2]',   c0_vals[0], C_O2_HAT,         rel=1e-9)
_assert_close('c0[H2O2]', c0_vals[1], H2O2_SEED_NONDIM, rel=1e-9)
_assert_close('c0[H]',    c0_vals[2], C_HP_HAT,         rel=1e-9)
```

(Stripped the misleading "sp[8] is dimensional" remark from R4
acknowledgment.)

---

## Section 2: Plan state

After 4 rounds of fixes (18 + 14 + 8 + 6 = 46 issues, all accepted),
the plan now has:

- Two-phase invariant harness (static + runtime).
- 12 parametrized broken-config tests (`TestMMSAsserts`), all expected
  to fire at the MMS invariant layer.
- 9-step pilot (quadrature sweep over two meshes, Newton-convergence
  sanity, R_ratio check, θ_inner discrete-min + indicator,
  perturbed-IC pilot with deterministic smooth perturbation, residual
  reduction guarded by absolute floor, SNES-tolerance sensitivity
  sweep, closure-algebra smoke with bundle).
- Stack-invariant asserts: formulation, mu_species, log-rate, Stern
  on, suppress_poisson_source, water/cation hydrolysis off, n_species,
  reaction identity (E_eq, α, n_e, reversible, stoichiometry, k0, conc
  factors, enabled), species identity (z_vals, roles, a_vals_hat, c0),
  no Γ slot, dt threshold, SNES tolerance bounds.
- Runtime margin asserts: η-clip, u-clamp, ion phi-clamp, free_dyn_floor,
  packing_floor (DG-interpolated min/max + quadrature indicator).
- UFL independence check with owned-constant whitelist via identity.
- `run_mms` returns `newton_converged, newton_iterations, snes_reason,
  F_res_l2_initial/final, c_H_L2` plus per-field errors.
- Per-field graded-mesh thresholds (post-pilot).
- Live continuation-coeff snapshot + post-solve verification.
- Module-level skip on `PYTEST_XDIST_WORKER`.
- Lazy `_import_run_mms` / `_import_graded_verifier` pattern.

(Full revised PLAN.md lands via auto-revise after this round closes.)

---

## Section 3: Continued critique prompt

This is round 5, the configured cap. Whatever issues remain at the end
of this round will be listed as unresolved in the auto-revise step
unless they're trivial enough to fold in immediately. Push back where
appropriate, but be aware the loop closes after your verdict.

Review my responses. Same numbered format and verdict line:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
