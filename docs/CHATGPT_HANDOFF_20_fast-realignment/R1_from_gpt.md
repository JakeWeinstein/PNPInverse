1. WHAT: Phase 1.2’s cathodic prefactor sign is wrong: `exp(α_j · n_e_j · η_j_clipped)` should match the residual/Picard convention, i.e. cathodic `exp(-α_j n_e_j η_j)`.
WHY: This flips voltage dependence and makes the anchor logic meaningless.
WHAT TO DO: Mirror `_build_picard_prefactors`: cathodic `-α n η`; anodic/reverse `+(1-α)nη`.

2. WHAT: Phase 1.2 omits the reversible anodic H2O2 term for `R_2e`, even though `PARALLEL_2E_4E_REACTIONS` has `reversible=True`, `anodic_species=1`.
WHY: The 2e channel is not just `A·O_s·H^2`; ignoring `B·P_s` changes peroxide balance and can fake convergence.
WHAT TO DO: Either explicitly make `R_2e` irreversible for the fast run, or include the `-B_2e P_s` branch in the Picard system.

3. WHAT: The Phase 1.2 rate expression double-counts/miscounts γ and drops the proton Boltzmann factor. `R_j = A_j · O_s · H_o^n · γ_s` plus `A_j = k0·γ_s·...` is not the existing log-rate algebra.
WHY: Correct cathodic rate uses `O_s γ · (H_o exp(-ψ_D) γ / c_ref)^p`; for p=2 that is γ³ and includes `exp(-2ψ_D)`.
WHAT TO DO: Derive `α̂_j` from the actual `cathodic_conc_factors` path, not hand-written `H_o^n`.

4. WHAT: The transport balance in Phase 1.2 adds `· L_eff` even though the existing nondimensional Picard uses `O_s = O_b - ΣR/D` with no `L_eff`.
WHY: This is a dimensional mismatch unless every rate and diffusivity is re-derived.
WHAT TO DO: Use the existing nondimensional flux balance exactly, or explicitly rederive and propagate a length factor everywhere.

5. WHAT: The plan is stale against the local tree: `picard_outer_loop_general` already exists and both log-c form files call it.
WHY: Implementing a new hardcoded `picard_ic_parallel_2rate.py` risks duplicating and bypassing the live generic Picard machinery.
WHAT TO DO: Change Phase 1.2 to audit/fix the existing generic Picard for parallel 2e/4e, not replace it blindly.

6. WHAT: Disabled-reaction handling is only specified for form construction, not for Picard dispatch/scaling.
WHY: A disabled reaction can be zero in the residual but still appear in the initializer topology/rate solve, especially with `enabled=False` and nonzero `k0`.
WHAT TO DO: Normalize disabled reactions centrally and make the Picard use the same disabled mask.

7. WHAT: H2O2 stoichiometry `[+1, 0]` is too weak as topology detection.
WHY: It does not verify O2 consumption, H+ stoich `[-2,-4]`, electron counts `[2,4]`, reversibility, reaction order, or disabled branches.
WHAT TO DO: Add a strict predicate for the exact parallel 2e/4e config or use explicit reaction IDs.

8. WHAT: Phase 2.1’s “drop the guard and sum γ terms” is not a valid implementation of multi-steric analytic ions in the current code.
WHY: `build_steric_boltzmann_expressions` returns one bundle and the forms expect one packing contribution; two Bikerman ions require a new shared-denominator closure and API changes.
WHAT TO DO: Either implement the full multi-ion closure end-to-end, or fast-path one ion as steric and the other as ideal.

9. WHAT: The `a_nondim` values are not merely “probably wrong”; they make bulk packing invalid. With `C_SCALE=1.2`, `0.0044*199.9/1.2 + 0.0048*100/1.2 > 1`.
WHY: `theta_b <= 0` will throw or produce nonphysical packing before Newton starts.
WHAT TO DO: Recompute `a_nondim` from excluded volume units before implementation; do not tune this after wiring.

10. WHAT: Phase 5b says to start continuation at `A_DEFAULT=0.01` for Cs and sulfate.
WHY: That is even worse: bulk packing is multiple times larger than 1.
WHAT TO DO: Start steric continuation from `a=0` or physically converted `~O(1e-5)` values, not `A_DEFAULT`.

11. WHAT: The whole IC/Picard stack still assumes one monovalent anion via `phi_o = ln(H_o / c_clo4_bulk)`.
WHY: There is no meaningful `c_clo4_bulk` for Cs+ + SO4²⁻; outer electroneutrality is a multi-ion algebraic solve, not a log ratio.
WHAT TO DO: Derive `phi_o` from `H_o + c_Cs exp(-φ) - 2 c_SO4 exp(2φ) = 0` or equivalent steric form.

12. WHAT: `compute_surface_gamma` only supports H+ plus one ClO4-like anion.
WHY: Passing Cs/SO4 through it gives the wrong charge signs, wrong exponent powers, and wrong steric denominator.
WHAT TO DO: Replace it with a list-based multi-ion γ helper before using Cs/SO4 in Picard.

13. WHAT: Deferring asymmetric composite-ψ and saying “let Picard absorb the shape mismatch” is self-deception.
WHY: Picard only fixes scalar surface balances; Newton still gets a 1:1 Gouy-Chapman/BKSA profile for a 2:1 electrolyte.
WHAT TO DO: Use a numeric 1D PB pre-solve, a linearized multi-ion Debye seed, or derive the 2:1 first integral before the production sweep.

14. WHAT: Stern splitting also remains 1:1/symmetric through `solve_stern_split`.
WHY: The Stern drop controls η; with 2:1 electrolyte and I=0.3 M, the surface slope and diffuse capacitance are wrong.
WHAT TO DO: Update Stern split to use the same multi-ion first integral, or disable Stern only if explicitly labeled as a numerical crutch.

15. WHAT: The Debye length used by the IC is not updated for ionic strength. `sqrt(poisson_coefficient)` is a base scale, not the 0.3 M effective screening length.
WHY: At I=0.3 M the layer is ~0.55 nm; using the old scale misplaces the entire initial charge layer.
WHAT TO DO: Use `sqrt(eps / Σ z_i² c_i_bulk)` for linearized ICs or the full multi-ion PB profile.

16. WHAT: Anchor logic is backwards/unsupported. `+0.45 V` is not “weak reaction side” relative to `E_2e=0.695 V`; lower V is more cathodic in the current sign convention.
WHY: The chosen anchor may be harder than `+0.55 V`, especially with `E_4e=1.23 V`.
WHAT TO DO: Start at `+0.55 V`, then use reaction-strength and ionic-strength continuation before scanning downward.

17. WHAT: `K0_PHYS_R4E = K0_PHYS_R1` can make the 4e channel dominate by construction.
WHY: With `n=4` and `E_eq=1.23 V`, the exponential drive can swamp the 2e channel; a “non-trivial R2e curve” may be arbitrary garbage.
WHAT TO DO: Include a tiny fast bracket sweep for `k0_4e` or label the run as structural convergence only, not page-15 interpretable.

18. WHAT: Acceptance is internally inconsistent: Phase 4 accepts `≥1 V`, final done accepts `≥15/25 V`.
WHY: One converged point is not an end-to-end page-15 sweep.
WHAT TO DO: Make the blocking acceptance one criterion: `≥15/25` or whatever the user actually wants, and do not call Phase 4 structurally done at one point.

VERDICT: ISSUES_REMAIN