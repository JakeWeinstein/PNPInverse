1. **WHAT:** A3 override is not wired through all active paths. `forms_logc[_muh].py` would use the override, but `update_gamma_from_solution` and `collect_v10a_rung_diagnostics` rebuild `pka_shift_expr` from `ctx["_cation_hydrolysis_sigma_S_expr"]` and ignore the proposed override.
**WHY:** Residual, Γ Picard update, and diagnostics can disagree. A3 can report the wrong `pka_shift_avg`, or worse, solve one model and classify another.
**WHAT TO DO:** Centralize “active pKa shift” construction and use it in forms, Γ update, diagnostics, and F0 decomposition. Store the active override on `ctx`.

2. **WHAT:** The plan contradicts itself on A3 coupling. It says the override “does not affect residual-side `R_net`”, but `R_net_default = build_proton_boundary_source(..., pka_shift_expr=...)` means overriding `pka_shift_expr` necessarily changes physical `R_net`.
**WHY:** The A3 interpretation is currently incoherent.
**WHAT TO DO:** Decide explicitly: either A3 is a residual-path imposed-σ ablation, or it is diagnostics-only. For plumbing, it should be residual-path, and the pass criteria must say that.

3. **WHAT:** A1/A2 cannot catch a broken physical `build_proton_boundary_source`. `manufactured_R_inj` bypasses the physical source.
**WHY:** A bug where `build_proton_boundary_source` returns `0.0` can pass A0 and pass A1/A2, exactly as you suspected.
**WHAT TO DO:** Add a physical-path gate: assemble the actual H/K hydrolysis residual contribution at the A0 solution and assert nonzero equal-and-opposite integrated flux, or add an A0-vs-physical-both-off comparison with a defined expected delta.

4. **WHAT:** The manufactured path still builds `R_net_default` before replacing it.
**WHY:** A manufactured plumbing test should not depend on physical pKa/source construction. A physical-path bug can crash A1/A2 before the manufactured constant is used.
**WHAT TO DO:** If `manufactured_R_inj is not None`, skip building `R_net_default` unless diagnostics explicitly request it.

5. **WHAT:** `apply_h_source` and `apply_k_sink` gate only FE residual terms, not the Γ Picard equation.
**WHY:** Any physical half-ablation has an inconsistent Γ update: Γ is computed as if the full coupled source/sink is active.
**WHAT TO DO:** Either declare these flags valid only with `manufactured_R_inj`, or make `update_gamma_from_solution` flag-aware.

6. **WHAT:** Reusing A.2 diagnostics for A1/A2 is invalid. `collect_v10a_rung_diagnostics` computes physical F0/mass-balance while Γ may come from manufactured `R_inj`.
**WHY:** A1/A2 mass-balance and F0 decomposition become meaningless and can create false failures.
**WHAT TO DO:** Add manufactured-aware diagnostics, or suppress physical mass-balance gates for manufactured ablations.

7. **WHAT:** Your corrected σ conversion is right, but it exposes a repo inconsistency. `2.26 C/m² * 6.241509e-6 = 1.41058e-5 counts/pm²`. The existing test using `0.226 C/m²` is a factor-10 mismatch for 226 µC/cm², and `docs/phase6/singh_2016_pka_formula.md` lists `0.141 1/pm²`, which is 10,000x larger than true counts/pm².
**WHY:** A3 cannot be both “deck-Cu Singh value” and “current code convention” until this is resolved.
**WHAT TO DO:** Fix the citation chain. If using current code units, `SIGMA_SINGH_K_CU_OVERRIDE = 1.41058e-5`, active post-clamp. The equivalent fake signed `sigma_S` is `-2.26 C/m²`.

8. **WHAT:** With current code units, the Singh deck override is tiny. For K/Cu, β is about `-45.61 per counts/pm²`, so `β * 1.41058e-5 ≈ -6.43e-4`, not the Singh table-scale `-6`.
**WHY:** A3 may pass while barely perturbing anything, making the “plumbing” conclusion weak.
**WHAT TO DO:** Separate the deck-value audit from a plumbing sentinel. Use a larger synthetic override for plumbing response, and keep the deck value as a unit/convention check.

9. **WHAT:** The proposed `build_pka_shift_from_override` sketch uses wrong parameter names: `A_pka_per_count_pm2`, `z`, and `r_M_O_pm` do not match current `A_pm`, `z_eff`, `r_M_pm`, `r_O_pm`.
**WHY:** The implementation sketch will either fail or silently diverge from the production formula.
**WHAT TO DO:** Do not duplicate the formula. Implement override by passing fake signed `sigma_S = -override / factor` into existing `build_pka_shift`, or factor out one shared internal helper.

10. **WHAT:** A3 pass criterion is mostly tautological. `pka_shift_avg == βσ_override` only tests the helper/diagnostic expression.
**WHY:** It does not prove the residual used the override. `Δσ_S < 10%` can also pass if the override never reached `R_net`.
**WHAT TO DO:** Add a residual-path check: `F0_A3 / F0_no_singh` or active `pka_factor_avg` must match `10^(-βσ_override)`.

11. **WHAT:** The 5% primary and 1% secondary thresholds are not defensible as written.
**WHY:** H and K are coupled through Poisson, sterics, BV H-dependence, and electroneutrality. “K unchanged” and “H unchanged” are not pure plumbing invariants. Also “3σ < 1%” is undefined for a single run.
**WHAT TO DO:** Use primary sign/magnitude gates plus convergence/positivity gates. Treat secondary species as bounded-response diagnostics, not hard 1% invariants, unless replicated noise estimates exist.

12. **WHAT:** R-inj bracketing only uses A1 and has no upper bound.
**WHY:** The selected value may overdrive A2, leave linear response, deplete K badly, or force Newton failure.
**WHAT TO DO:** Bracket A1 and A2 together. Require convergence, positive concentrations, and primary shift in a band like 5-25%, not merely `>=5%`.

13. **WHAT:** The proposed soft clamp on `c_K(0)` is a bad mitigation.
**WHY:** It changes the PDE and can hide the very K-sink sign/wiring bug A2 is supposed to expose.
**WHAT TO DO:** Lower `R_inj`, add λ continuation for the manufactured ablation, or mark the result inconclusive. Do not pass a clamped residual as plumbing-verified.

14. **WHAT:** Full Firedrake byte-equivalence at `rel=1e-12` is too brittle, and routing for `1e-9` vs `1e-12` is undefined.
**WHY:** PETSc solves are not reliable byte-level regression artifacts across runs.
**WHAT TO DO:** Keep `1e-12` for pure helper/form same-process checks. For end-to-end A0, define per-observable tolerances and a gray band, e.g. pass <= `1e-9`, rerun/investigate `1e-9..1e-6`, block above that.

15. **WHAT:** The “fast tests no Firedrake” list overclaims. UFL form comparisons, residual zeroing, and byte-equivalence need Firedrake.
**WHY:** The plan promises guardrails that cannot run in the stated tier.
**WHAT TO DO:** Keep pure Python tests to parsing, classification, unit conversion, and override dict construction. Move form/residual tests to marked slow Firedrake tests.

16. **WHAT:** Boolean and override validation is underspecified.
**WHY:** Raw config strings like `"False"` can become truthy if read directly from `conv_cfg.get`. Negative or NaN override counts would invert the post-clamp meaning.
**WHAT TO DO:** Parse `apply_h_source` and `apply_k_sink` with `_bool` in config, and reject non-finite or negative `override_sigma_singh_counts_pm2`.

17. **WHAT:** The A3 `10% σ_S leak` rule is not a clean leak detector.
**WHY:** If A3 intentionally changes residual `R_net`, some σ movement is expected through the solved fields. If the corrected override is tiny, 10% is too loose; if the old Singh-scale value is used, a clean run can fail.
**WHAT TO DO:** Test leak structurally: assert the Stern expression still depends only on PNP fields, while the pKa expression uses the override. Do not infer leak solely from `Δσ_S`.

VERDICT: ISSUES_REMAIN