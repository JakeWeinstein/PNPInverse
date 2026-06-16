1. **`dR_net/dσ_S` is mislabeled.**

WHAT: The `C_S -> C_S(1±ε)` column is not `dR_net/dσ_S`. Because `σ_S = C_S(φ_applied - φ) * scale` in [forms_logc_muh.py](</Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py:700>), and the solve relaxes `φ`, concentrations, BV rates, and `Γ`, the estimate is:

`(dR_net/dlog C_S) / (dσ_S/dlog C_S)` at fixed `V_RHE`, along the re-solved Stern-capacitance manifold.

WHY: Calling this a σ partial derivative will mislead Phase E. It folds transport, surface pH, cation enrichment/depletion, Γ saturation, and current changes into the quotient.

WHAT: Keep it only if renamed explicitly, e.g. `dRnet_dsigma_along_stern_capacitance`, and also log `dRnet_dlogCs`, `dsigma_dlogCs`, `epsilon`, fixed variables, and relaxed variables. A cleaner σ-only partial would require a separate postprocess/ablation that perturbs σ only in the Singh/pKa path while holding solved fields fixed.

2. **The “C_S is equivalent to fitting `r_H_El`” justification is wrong.**

WHAT: Perturbing `C_S` changes the electrostatic boundary condition and the whole PNP/BV state. Perturbing `r_H_El_pm` changes the Singh geometric factor in `ΔpKa`; it does not change `σ_S`.

WHY: These are different calibration knobs. Treating them as equivalent would make later identifiability claims invalid.

WHAT: Document `C_S` perturbation as Stern-capacitance leverage only. If Phase E fits Singh geometry, add a separate `dRnet_dr_H_El` or fixed-field `dRnet_dsigma_pka_only` diagnostic.

3. **The 50% FD-vs-perturb exclusion is not defensible.**

WHAT: FD over voltage measures `dR/dV / dσ/dV`; the perturbation column measures `dR/dlogC_S / dσ/dlogC_S`. Disagreement is path dependence, not numerical noise. Excluding it also adds a hidden fourth filter beyond the locked filters in [PHASE_6B_V9...](</Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md:355>).

WHY: You can discard the actual max Stern-capacitance leverage just because voltage motion is controlled by another state variable. That changes the locked selection rule.

WHAT: Do not exclude candidates on FD mismatch. Log `path_mismatch` informationally. For numerical quality, use the perturbation itself: require finite converged solves, a minimum `|σ_+ - σ_-|`, and one-sided slope agreement `|S_+ - S_-| / max(|S_+|, |S_-|) <= 0.25` for primary candidates, relaxed to `0.50` only in fallback.

4. **The Levich helper must not hardcode `D_O2 = 2.18e-9`.**

WHAT: The plan’s helper uses the wrong diffusivity. The codebase constant is `D_O2 = 1.9e-9` in [_bv_common.py](</Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py:73>), with `C_O2 = 1.2`, `L_REF = 1e-4`, and `I_SCALE` defined at [_bv_common.py](</Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py:213>).

WHY: The wrong D shifts `I_lim` by about 15%, enough to flip points near the `0.9` cutoff.

WHAT: Import `F_CONST`, `D_O2`, and `C_O2` from `_bv_common`. Use `I_lim_4e = 4 F D_O2 C_O2 / l_eff_m * 0.1`. At `l_eff = 16e-6 m`, this is about `5.50 mA/cm²`. The `0.1` conversion factor is correct.

5. **`|cd|/I_lim_4e` is not a general O₂-plateau test for a mixed 2e/4e cathode.**

WHAT: The observable weights current as `R_2e + 2 R_4e` in [observables.py](</Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/observables.py:106>). At O₂-flux limit, `|cd|/I_lim_4e = (1 + x_4e)/2`, where `x_4e = R_4e/(R_2e + R_4e)`. Pure 2e plateau gives `0.5`; 50/50 plateau gives `0.75`; only strongly 4e-rich plateaus exceed `0.9`.

WHY: The plan’s “non-plateau” interpretation is false in 2e-rich or mixed regimes. A transport-limited point can pass the locked current-ratio filter.

WHAT: Use `I_lim_4e` for the locked rule if the rule literally means “current below 90% of pure-4e maximum.” But also log `o2_flux_over_levich = (R_2e + R_4e)/(L_REF/l_eff)` or equivalently divide `|cd|` by `I_lim_4e * (1 + x_4e)/2`. Do not silently substitute that into the locked filter without amending the rule.

VERDICT: ISSUES_REMAIN