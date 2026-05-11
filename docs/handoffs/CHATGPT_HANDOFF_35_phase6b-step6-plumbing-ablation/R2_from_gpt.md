1. **WHAT:** Your A3 sign analysis is wrong. At V_kin, `sigma_S ≈ -0.017 C/m²` is cathodic under the code convention, not “anodic-clamped to 0”. The active Singh count is about `1.07e-7 counts/pm²`, matching A.2’s `pka_shift_avg ≈ -4.88e-6`.
**WHY:** This reverses the expected A3 behavior.
**WHAT TO DO:** Update the plan: current K/Cu has `r_H_El=200.98 < r_M+r_O=201`, so `G < 0`, `β ≈ -45.61`, and positive `sigma_singh` lowers pKa.

2. **WHAT:** `SIGMA_SINGH_PLUMBING_SENTINEL = 0.022` does not make `pka_factor ≈ 0.1`. It gives `ΔpKa ≈ -1.003`, so `10^(-ΔpKa) ≈ 10.08`.
**WHY:** Your A3 expected R_net direction is backward. The sentinel amplifies the forward hydrolysis branch about 10x; it does not suppress it.
**WHAT TO DO:** Change the expected A3 response to amplification, or choose a different sentinel only if you intentionally want suppression. With the current clamp, positive override cannot suppress K/Cu unless β is positive.

3. **WHAT:** A0b’s `>1% c_H` gate is not supported by A.2 data. In the landed A.2 JSON, going from `k_hyd=1e-5` to `k_hyd=1e-3` changes `c_H_avg` only from `1.8054157548e-08` to `1.8074774657e-08`, about `0.114%`, while Γ changes about 15x. Across the whole k_hyd grid, `c_H_avg` range is only about `0.134%`.
**WHY:** A0b can fail cleanly even if the physical R_net path is wired.
**WHAT TO DO:** Do not use solution-level `c_H` shift as the physical-path sentinel. Assemble/probe the physical H/K residual contribution directly and assert nonzero equal-and-opposite flux.

4. **WHAT:** “`R_net_A0 = 0.0405`, so A0b should move c_H by >1%” is dimensionally unjustified.
**WHY:** Boundary flux magnitude does not translate directly to relative OHP concentration shift, and A.2 already shows the c_H response is much smaller than that intuition.
**WHAT TO DO:** Replace that mitigation with an empirical precheck from the A.2 record or, better, a residual assembly test.

5. **WHAT:** The proposed `ctx["_cation_hydrolysis_sigma_singh_override_counts_pm2"]` accessor will not affect already-built UFL forms unless the override is represented by a mutable Firedrake coefficient embedded in the form.
**WHY:** A Python `None`/not-`None` branch at form-build time is baked into the form. A later `set_reaction_sigma_singh_override_model(ctx, value)` cannot switch paths.
**WHAT TO DO:** Either rebuild forms per ablation with static config, or add mutable `override_active` and `override_counts` coefficients and use a UFL conditional. Do not pretend this is analogous to existing mutable `k_hyd_func` unless you implement it that way.

6. **WHAT:** A3 gate `R_net_A3 / R_net_A0 ≈ pka_factor_A3 / pka_factor_A0` is false for this model.
**WHY:** Net `R_net` includes forward, reverse, Γ, and the Langmuir vacancy factor. With a 10x pKa-factor amplification, Γ may move toward the cap, changing `(1 - Γ/Γ_max)` and the reverse term.
**WHAT TO DO:** Compare the uncapped forward branch/F0 or `pka_factor_avg` directly. If comparing capped/net rates, include the vacancy and reverse terms explicitly.

7. **WHAT:** The A3 pka-factor formula in Issue 10 is garbled. `forward_avg_no_k_hyd` is already `⟨c_K * pka_factor⟩`; it is not something to divide by `k_hyd` again.
**WHY:** The diagnostic gate can be numerically wrong while looking principled.
**WHAT TO DO:** For override A3, pKa factor is constant, so use `F0_decomposition.pka_factor_avg` directly, or verify `c_K_pka_product_avg / c_K_avg ≈ 10^(-βσ_override)`.

8. **WHAT:** The R_inj bracket only searches upward from `1e-2`.
**WHY:** With the new 25% upper bound, `1e-2` may already overdrive A1. Then the plan escalates upward and declares inconclusive, missing valid smaller values like `1e-3`.
**WHAT TO DO:** Search both directions or use adaptive log bracketing. Include at least `{1e-4, 1e-3, 1e-2, 1e-1, 1.0}` before escalation.

9. **WHAT:** Requiring the same `R_INJ_MFG` for A1 and A2 may be unnecessarily brittle.
**WHY:** H and K sensitivities are vastly different. A single value may overdrive H while barely moving K, even with correct plumbing.
**WHAT TO DO:** Prefer separate bracketed sentinels for A1 and A2, unless “same magnitude” is itself a required scientific constraint. Plumbing only needs sign-controlled measurable response.

10. **WHAT:** The driver-only validation for invalid flag combos is too weak.
**WHY:** These flags live in solver config. Other tests/scripts can build invalid physical half-ablations directly and get incoherent Γ behavior.
**WHAT TO DO:** Validate in `config.py` or form-build as well, not only in the Step 6 driver.

11. **WHAT:** “Physical diagnostics skipped” for A1/A2 is not automatic. `solve_lambda_ramp_from_warm_start` calls `collect_v10a_rung_diagnostics(ctx)` before the driver callback.
**WHY:** Manufactured runs will still emit physical diagnostics or diagnostic errors unless you modify the collector/anchor path.
**WHAT TO DO:** Make `collect_v10a_rung_diagnostics` manufactured-aware, or strip/mark those fields after collection and ensure classifier ignores diagnostic errors for A1/A2.

12. **WHAT:** The UFL structural test as written is overbroad. The full residual must depend on `ctx["U"]` through `c_M` and `c_H` even when pKa override is constant.
**WHY:** `extract_coefficients(F_res)` on the whole residual will still find solution coefficients.
**WHAT TO DO:** Store `ctx["_cation_hydrolysis_pka_shift_expr"]` or `ctx["_cation_hydrolysis_sigma_S_active_expr"]` and inspect that expression specifically.

13. **WHAT:** The “1% single-run noise floor” is asserted, not established.
**WHY:** SNES rtol and Picard rel_tol do not imply observable noise. A.2 shows real physical c_H changes below 1%.
**WHAT TO DO:** Either run duplicate A0 solves to estimate numerical repeatability, or avoid solution-level 1% gates for physical-path detection.

14. **WHAT:** If the σ-unit or Singh-sign audit changes code or calibrated constants, Step 6 can no longer use commit `2f5f071` A.2 as the baseline.
**WHY:** Any formula/sign/unit correction changes A.2 physics and invalidates the locked sequence state.
**WHAT TO DO:** Add an explicit branch: if audit changes only docs/tests, continue; if it changes solver math or Singh params, rerun v10a'/A.2 before Step 6.

VERDICT: ISSUES_REMAIN