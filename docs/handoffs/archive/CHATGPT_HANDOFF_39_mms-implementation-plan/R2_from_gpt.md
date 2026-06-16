1. WHAT: The new `newton_iterations >= 2` requirement on every unperturbed MMS run is an overcorrection.
WHY: A valid MMS solve can exit in 0-1 iterations if `U_manuf` is already below tolerance. That is not itself a failure once you have a separate perturbed-start test.
WHAT TO DO: Drop the unperturbed iteration-count assertion. Keep the perturbed-start pilot and tolerance sweep. For normal runs, record iteration counts but do not require a minimum.

2. WHAT: The perturbed IC uses a seeded random Function.
WHY: Random FE noise can violate Dirichlet BCs, introduce mesh-scale gradients, vary under MPI/DOF ordering, and make Newton behavior noisy.
WHAT TO DO: Use deterministic smooth analytic perturbations that vanish on essential-BC boundaries, e.g. `eps*sin(pi*x)*sin(pi*y)` variants per component. Apply BCs to the initial guess before solve.

3. WHAT: “Residual reduction > 100x at every mesh” is still too brittle.
WHY: If the initial residual is already small, the ratio is noise-dominated. This recreates the same false-failure risk as the iteration-count floor.
WHAT TO DO: Apply residual-reduction criteria to the perturbed-start pilot only, or guard with an absolute initial-residual floor.

4. WHAT: The closure algebra smoke test compares production expressions evaluated at interpolated `U_manuf` against independent analytic exact expressions, then demands `1e-9`.
WHY: Interpolation error alone makes that fail. The production bundle sees `ci_exprs(U_manuf_h)` and interpolated `phi_h`; the independent closure sees analytic `c_ex, phi_ex`.
WHAT TO DO: Compare both closures using the same inputs. Either build the independent closure from `ctx["ci_exprs"]`/`phi` for this diagnostic only, or compare analytic-vs-production with an interpolation-error-scaled tolerance, not `1e-9`.

5. WHAT: The production bundle does not expose every object you say you will compare.
WHY: `StericBoltzmannBundle` exposes `c_steric_expr`, `packing_contribution`, and `charge_density`; it does not directly expose `theta_inner_ex` or `mu_steric_ex`.
WHAT TO DO: Reconstruct production-side `theta_inner` from `ctx["ci_exprs"]`, `ctx["steric_a_funcs"]`, `ctx["boltzmann_z_scale"]`, and bundle packing terms, or limit the smoke test to exposed quantities.

6. WHAT: The UFL independence check is probably checking the wrong objects.
WHY: `extract_coefficients(S)` will usually return the base mixed `ctx["U"]` coefficient, not the split components from `fd.split(ctx["U"])`.
WHAT TO DO: Check `ctx["U"]` and `ctx["U_prev"]` directly in extracted coefficients. Whitelist expected live coefficients like `phi_applied_func`, `stern_coeff_const`, `bv_k0_funcs`, `bv_alpha_funcs`, and `boltzmann_z_scale`.

7. WHAT: DG `project(...).dat.data_ro.min()` is not a reliable min/max check.
WHY: L2 projection can smooth extrema. It does not “catch nodal violations” as claimed.
WHAT TO DO: Use `fd.interpolate(expr, DGk)` for sampled DOF checks, plus high-degree quadrature indicator forms. Set explicit `dx(..., degree=...)` on indicator checks.

8. WHAT: The eta/u/phi clamp checks still mix scalar-integral logic with max logic.
WHY: The shown eta code uses an integral-style assembly and even `fd.dot` on scalar expressions. That does not compute a max.
WHAT TO DO: Centralize real helpers: `_expr_min`, `_expr_max`, `_expr_abs_max`, with interpolation-based sampling and optional quadrature indicators. Use them consistently.

9. WHAT: Reaction asserts still omit `R2e` `k0_model` and reaction `enabled` status.
WHY: A disabled reaction or wrong R2e rate can pass most identity checks and destroy the manufactured BV source.
WHAT TO DO: Assert both reaction `k0_model` values, both `k0_model > 0`, and `rxn.get("enabled", True) is True`.

10. WHAT: You still do not assert the dynamic species identity.
WHY: The source assumes index 0 O2, index 1 H2O2, index 2 H+, charges `[0,0,+1]`, physical `a_nondim`, and bulk `c0`. Reaction asserts alone do not protect that.
WHAT TO DO: Add hard asserts for `ctx["n_species"] == 3`, `z_vals == [0,0,1]`, roles, `a_vals`, `D_vals` if relevant, and `c0_model_vals`.

11. WHAT: Using `pytest.approx` inside the verification script/source module is a bad dependency leak.
WHY: `scripts/verification/...` should run as a script outside pytest.
WHAT TO DO: Use `math.isclose` or a tiny local `_assert_close(name, got, expected, rel)` helper.

12. WHAT: The rate tests “skip rather than fail on NaN” is unsafe.
WHY: Running only `-k l2_convergence` could skip and appear acceptable without the Newton test running.
WHAT TO DO: The fixture should assert all solves converged before returning normal convergence results. Only special diagnostic callers should allow failed meshes.

13. WHAT: `TestMMSAsserts` routed through `run_mms([8], _force_sp=sp)` is too indirect.
WHY: Some broken configs may fail in `build_forms` with `ValueError`, some may be caught by your invariant, and some may accidentally start expensive setup. The test intent becomes muddy.
WHAT TO DO: Factor a `_prepare_mms_context(sp, mesh)` or direct `_assert_stack_invariants(...)` harness. For each broken config, assert the expected failure location and message before solve construction.

14. WHAT: xdist handling is still deferred.
WHY: “Suffix artifacts if xdist is active OR mark serial” leaves the plan unresolved. Also `setdefault` will not override inherited shared cache paths.
WHAT TO DO: Choose one now. Easiest: mark the module serial/no-xdist and document it. If supporting xdist, override cache dirs when `PYTEST_XDIST_WORKER` is present and always use worker-scoped artifact paths.

VERDICT: ISSUES_REMAIN