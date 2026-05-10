1. WHAT: Your proposed `set_initial_conditions(ctx, sp)` fix at `grid_per_voltage.py:350` is wrong as written. The dispatcher reads `initializer`; if `sp` still says `debye_boltzmann`, it routes right back to the analytical IC.
WHY: The existing code intentionally bypasses dispatch there to force linear fallback, even though it picked the wrong formulation-specific function.
WHAT TO DO: Dispatch with a cloned params object whose `bv_convergence.initializer = "linear_phi"`, or directly call `set_initial_conditions_logc_muh` / `set_initial_conditions_logc` by formulation. Do not call the raw dispatcher with unchanged `sp`.

2. WHAT: Your z=0 analysis has a factual error: `logc_muh` reconstruction does not use immutable “global z”. It uses the same `z` Constants stored as `ctx["z_consts"]`; `_set_z_factor()` mutates them at `grid_per_voltage.py:299-304`.
WHY: At z=0, `c_H = exp(mu_H - em*z_H*phi)` becomes `exp(mu_H)`, not `exp(mu_H - phi)`. So the wrong LOGC fallback does not artificially deplete H at z=0; it likely leaves H bulk-like while BV/Stern still generate a hard boundary problem.
WHAT TO DO: Re-evaluate z=0 diagnostics after fixing fallback. Your explanation of why z=0 “should work” is not reliable.

3. WHAT: Phase 5α must patch the initialization electrostatics too, not only the loop update. Current Picard computes the first `psi_D/eta_list` with `solve_stern_split()` before the loop at `picard_ic.py:1102-1126`.
WHY: If the first Picard rate prefactors are wrong by many orders, relaxation can settle into the wrong basin or report misleading convergence.
WHAT TO DO: Factor a helper like `update_electrostatics(c_s)` and use it both before `eta_list` initialization and after each `c_s` update.

4. WHAT: Optional `multi_ion_ctx` is acceptable, but only if you isolate the branch behind helpers. Sprinkling `if multi_ion_ctx is not None` at `1138`, `1212`, and `1228` will make this loop brittle.
WHY: Single-ion byte-equivalence depends on preserving update order exactly; mixed branch edits inside the loop are easy to get subtly wrong.
WHAT TO DO: Keep one `picard_outer_loop_general`, but add small closures/helpers for `phi_o`, `gamma_s`, and Stern split. Separate full duplicate function is worse unless the helper approach becomes unreadable.

5. WHAT: The multi-ion helper is not fully residual-equivalent for large potentials. `boltzmann.py` clamps each ion’s `phi` at `phi_clamp` in the residual, while `multi_ion.py` uses unclamped `_safe_exp`.
WHY: If any root search or surface reconstruction approaches `|phi| > 50`, Picard and residual solve different analytic-ion laws.
WHAT TO DO: Either apply the same per-ion `phi_clamp` in `multi_ion.py` or assert/log that all Picard/IC potentials stay well inside clamp for the page-15 grid.

6. WHAT: Multiple roots are probably not the Cs/SO4 production problem, but monotonicity is not guaranteed for arbitrary unequal steric sizes.
WHY: Bisection on a global `[-50,+50]` bracket can return an unintended root if the residual ever becomes non-monotone or if exp caps flatten it.
WHAT TO DO: Add a monotonic scan test for Cs/SO4 over the expected phi range, check `dρ/dφ < 0` at the root, and in Picard prefer a local bracket around the previous `phi_o`.

7. WHAT: `k0_R2e_init = 1e-12` is small enough. The bigger issue is the 1000x ramp.
WHY: The hard transition is near mass-transport limitation; a 1000x step can jump directly across the basin boundary.
WHAT TO DO: Use adaptive continuation: start with factors 10 or 32, rollback and bisect on failure. Four hard-coded 1000x jumps are too coarse.

8. WHAT: Building the solver with tiny `k0` has metadata consistency risks. Picard reads `ctx["nondim"]["bv_reactions"]`, while the residual uses mutable `bv_k0_funcs`.
WHY: If you later assign `bv_k0_funcs` upward, residual and Picard metadata diverge unless you rebuild or explicitly override Picard k0 inputs.
WHAT TO DO: Prefer building with target positive `k0`, then assign the mutable function down before the first solve; or add a real k0-continuation hook that updates both Picard input and residual.

9. WHAT: Current C+D does not have a k0 homotopy hook.
WHY: It rebuilds a fresh context per voltage at `_build_for_voltage()` and then immediately solves. A driver-side assignment has nowhere to run unless you extend the orchestrator or write a custom anchor solve.
WHAT TO DO: Add a pre-solve homotopy callback or create a separate anchor builder that performs k0/dt continuation and returns the converged snapshot to C+D.

10. WHAT: `dt_init=1e-4` will not ramp to `0.25` under current SER defaults. `dt_max = dt_init * dt_max_ratio` at `grid_per_voltage.py:258`.
WHY: With default ratio 20, max dt is `2e-3`, not `0.25`.
WHAT TO DO: Add `dt_max_abs`, pass `dt_max_ratio=2500`, or implement an explicit dt ladder.

11. WHAT: Above-Eeq homotopy is plausible, but your “zero crossing at E_eq” language is too clean. The implemented reversible rate has cathodic O/H factors and anodic H2O2, so net zero is concentration-dependent.
WHY: At `V=E_eq`, net R2e is not automatically zero unless surface concentrations satisfy the configured reaction quotient.
WHAT TO DO: Treat `+0.85 → +0.55` as empirical voltage homotopy, not as guaranteed anodic-to-cathodic crossing.

12. WHAT: Stern homotopy is still under-specified. “No Stern” changes the residual form and BV eta semantics; large finite Stern keeps the Robin form but can make `eta ≈ -E_eq`.
WHY: This is not a one-parameter smooth path unless you stay inside the positive-Stern Robin branch.
WHAT TO DO: If you use it, use finite positive `C_S`: start very large, ramp down to `0.10`, and expose `stern_coeff` as mutable or accept rebuild/state-copy. Defer behind k0/dt.

13. WHAT: Phase 5β byte-equivalence is safe only for logc if you force linear dispatch correctly. For legacy `logc_muh`, behavior will change.
WHY: Current fallback is wrong for muh; fixing it may alter old convergence baselines.
WHAT TO DO: Add regression tests: logc single-ClO4 fallback remains byte-equivalent; logc_muh single-ClO4 fallback is intentionally changed and must still pass the legacy sweep.

14. WHAT: Phase 5ζ diagnostics should not be purely post-convergence if you use failure JSON to steer decisions.
WHY: Current multi-ion counterion diagnostics use per-ion theta, not shared theta, so saturation and surface counterion numbers can mislead failure triage.
WHAT TO DO: Patch diagnostics before relying on them to choose between 5γ/5δ/5ε. It need not block the first Picard/fallback patch.

15. WHAT: Your Phase 5α test criterion “sensible R/c_s” is too vague.
WHY: A wrong Picard can look sensible while being inconsistent with the assembled boundary residual.
WHAT TO DO: Add a rate-consistency test: evaluate the residual BV rate at the IC surface and compare to Picard `R_list` within tolerance for Cs/SO4 at `+0.55`.

VERDICT: ISSUES_REMAIN