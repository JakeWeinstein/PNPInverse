1. WHAT: The dynamic-species asserts read from non-existent `scaling` keys.
WHY: `ctx["nondim"]` does not publish `z_vals`, `species_roles`, or `a_vals_hat`. Those live on `sp[4]`, `sp[6]`, and `sp[10]["bv_bc"]["species_roles"]`. As written, the asserts will see `[]` and fail, or get “fixed” later by weakening them.
WHAT TO DO: Assert species identity from the actual sources:
`sp[4]` for charges, `sp[6]` for `a_vals`, `sp[8]` or `ctx["nondim"]["c0_model_vals"]` for concentrations, and `sp[10]["bv_bc"]["species_roles"]` for roles.

2. WHAT: The `serial` marker is not enforced in this repo.
WHY: `pyproject.toml` only registers `slow`; `tests/conftest.py` has no xdist/serial hook. `pytest.mark.serial` is currently just metadata and xdist will ignore it.
WHAT TO DO: Either implement enforcement in `conftest.py` and register the marker, or add a module-level skip/fail when `PYTEST_XDIST_WORKER` is present. Documentation alone does not make it serial.

3. WHAT: `_prepare_mms_context_for_asserts` builds manufactured fields, closure, and BV rates before running stack invariants.
WHY: Broken configs like missing/wrong reactions can fail inside helper construction before the invariant test fires. That defeats the purpose of `TestMMSAsserts`.
WHAT TO DO: Split invariants into static and field-dependent phases. Run static stack asserts immediately after `build_forms`; only then build closure/rates and run runtime checks.

4. WHAT: Allowing `ValueError` from `build_forms` to satisfy `TestMMSAsserts` weakens criterion 5.
WHY: The acceptance criterion says the MMS/source-builder invariants catch drift. If production validation catches it earlier, that is useful, but it is not evidence your invariant layer works.
WHAT TO DO: Categorize broken configs into “expected production validation failure” vs “expected MMS invariant failure.” Only count the latter toward source-builder invariant coverage.

5. WHAT: `_perturbed_initial_guess` is under-specified and uses out-of-scope variables.
WHY: The function references `ctx` and `phi_idx` but does not accept them. The snippet also interpolates from `U_init.sub(i)` into itself, which is fragile.
WHAT TO DO: Make the signature `def _perturbed_initial_guess(ctx, U_manuf, mesh, eps=...)`, derive indices inside it, interpolate from `U_manuf.sub(i) + pert`, then apply BCs.

6. WHAT: The closure smoke still needs an explicit clamp/floor alignment statement.
WHY: Production uses `phi_clamp`, `free_dyn_floor`, `packing_floor`, and `z_scale`; your independent closure may omit some because margins say inactive. For a `1e-9` algebra comparison, both sides must either include the same branches or prove the inactive branches are bit-identical on the sampled input.
WHAT TO DO: In the smoke helper, mirror production branch structure exactly, including clamps/floors/z_scale. Keep the analytic MMS source free to use unclamped expressions only after the margin asserts pass.

7. WHAT: The coefficient whitelist logging is too loose if “unknown coefficients” are only printed.
WHY: A source term can accidentally depend on a helper coefficient that is not `U` but still violates the intended source contract. Printing will disappear in CI noise.
WHAT TO DO: Make unknown coefficients fail unless explicitly whitelisted. Keep the whitelist small and named.

8. WHAT: The indicator helper may still be ambiguous about quadrature domain and threshold type.
WHY: Clamp/floor checks are only useful if they use the same mesh/domain and sufficiently high quadrature everywhere. A helper default can silently regress.
WHAT TO DO: Require `degree` explicitly at call sites or bind it to `quad_degree`; use `fd.Constant(threshold)` inside comparisons; compute `vol` with the same `dx_q`.

VERDICT: ISSUES_REMAIN