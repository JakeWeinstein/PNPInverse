1. WHAT: `missing_reaction` is wrongly categorized as `build_forms`.
WHY: Production parsing accepts any non-empty `bv_bc.reactions` list; it does not require exactly two reactions. A one-reaction config will likely build forms successfully and should be caught by your static invariant `len(rxns) == 2`.
WHAT TO DO: Move `missing_reaction` to `"static"` and count it toward MMS invariant coverage.

2. WHAT: `k0_r4e_wrong` is categorized as `runtime`, but your static reaction-identity asserts already check `k0_model`.
WHY: The expected failure layer should match the invariant that actually fires. Otherwise the test is less precise than the table claims.
WHAT TO DO: Mark `k0_r4e_wrong` as `"static"` unless you intentionally remove the static `k0_model` assert, which you should not.

3. WHAT: The closure smoke description puts `z_scale` “on packing/charge,” but production bundle fields are unscaled.
WHY: In `boltzmann.py`, `bundle.packing_contribution = a_k*c_k` and `bundle.charge_density = z_k*c_k`; `z_scale` is applied later by the residual/packing assembly. If your smoke helper multiplies per-ion `P` or `rho` by `z_scale`, it will not match the bundle and will test the wrong object.
WHAT TO DO: Compare unscaled per-ion `c_steric`, `P`, and `charge_density` to the bundle. Apply `z_scale` only when reconstructing derived `theta_inner`/Poisson totals.

4. WHAT: The owned-constant handling in the independence check is still unresolved.
WHY: `class _OwnedConstant(fd.Constant)` is probably not a valid Firedrake pattern, and “verify at code time” leaves a critical invariant underspecified. Locally created `fd.Constant`s may appear in `extract_coefficients` and cause false failures.
WHAT TO DO: Use a concrete helper like `_owned_constant(value, name)` that creates `fd.Constant`, records the exact object in `_OWNED_COEFFS`, and returns it. Or explicitly filter only `fd.Function` plus known live `fd.Constant` objects. Do not rely on subclassing.

5. WHAT: `_perturbed_initial_guess` uses `indices.phi_index` with `Function.sub(...)`.
WHY: `indices.phi_index` is `-1` in the current layout. Negative indexing is safe for tuples from `fd.split`, but not something to assume for Firedrake `Function.sub`.
WHAT TO DO: Use a nonnegative subspace index for assignment, e.g. `phi_sub_index = ctx["n_species"]` when `gamma_index is None`, or add a helper that converts layout indices for `Function.sub`.

6. WHAT: The `c0` explanation is inaccurate.
WHY: In this factory, `sp[8]` already carries nondimensional concentrations because `make_bv_solver_params` is called with dimensionless species values and `concentration_inputs_are_dimensionless=True`.
WHAT TO DO: Keep the assert against `ctx["nondim"]["c0_model_vals"]`, but remove the claim that `sp[8]` is dimensional.

VERDICT: ISSUES_REMAIN