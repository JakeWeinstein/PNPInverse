# Domain Pitfalls

**Domain:** PDE Verification & Validation for Electrochemical Inference Pipeline
**Researched:** 2026-03-06

## Critical Pitfalls

Mistakes that cause rewrites, incorrect publications, or invalidated results.

### Pitfall 1: MMS Source Terms Computed Incorrectly (Sign or Boundary Mismatch)

**What goes wrong:** The manufactured source term `S` or boundary correction `g` has the wrong sign, causing MMS to "pass" with incorrect convergence rates or to fail entirely. This is the most common MMS error.

**Why it happens:** The sign convention for the boundary integral depends on whether the solver uses integration-by-parts (IBP) with the outward or inward normal. The existing code documents this carefully (see `SIGN CONVENTION (critical)` comment in `mms_bv_convergence.py`), but any modification to the weak form requires re-deriving the correction.

**Consequences:** A sign error in `g` means the MMS test is verifying the wrong problem. The convergence rates may still look correct (O(h^2) for L2), giving false confidence.

**Prevention:**
- Verify the boundary correction formula algebraically: at the exact solution, the total boundary residual (diffusive flux + BV flux + correction) must be exactly zero.
- Test with a simple 1D case where the correction can be computed by hand.
- The existing implementation is correct (I verified the IBP derivation in the code comments). Do not modify without re-deriving.

**Detection:** If L2 rates are correct but H1 rates are degraded (e.g., 1.5 instead of 1.0), suspect a boundary correction error. The H1 norm is more sensitive to boundary errors because it includes gradient accuracy.

### Pitfall 2: Testing the MMS Code Path Instead of the Production Code Path

**What goes wrong:** The MMS test builds its own weak form that mirrors the production solver. If the MMS form has a different structure than the production solver, MMS verifies the wrong code. Conversely, if MMS calls the production solver directly, a bug in the solver could cancel out in the MMS framework.

**Why it happens:** There is a fundamental tension in MMS: the test must be independent of the production code (to catch bugs) but must solve the same equations (to be relevant).

**Consequences:** MMS passes but the production solver has a bug that the MMS test does not exercise.

**Prevention:**
- The existing approach (separate MMS weak form that mirrors but does not call the production solver) is correct.
- After MMS passes, do a manual code review comparing the MMS weak form to the production `bv_solver.py` weak form. Check: same stoichiometry signs, same electromigration terms, same BV rate expression.
- Add a comment in the MMS test listing which production code elements it is designed to verify.

**Detection:** If MMS passes but the forward solver produces physically unreasonable results (e.g., negative concentrations, wrong current sign), the MMS test is verifying the wrong equations.

### Pitfall 3: Nondimensionalization Errors Producing Silent Wrong Results

**What goes wrong:** A dimensional quantity is used where a nondimensional one is expected (or vice versa). The solver converges normally but produces answers that are off by factors of physical constants.

**Why it happens:** The codebase has a past bug of this type (hardcoded value that should have been a physical constant in `Nondim/constants.py`). The nondimensionalization involves multiple interacting scales (length, voltage, concentration, diffusivity, reaction rate), and any mismatch is invisible to convergence tests.

**Consequences:** All downstream results (I-V curves, inferred parameters) are quantitatively wrong. The error is proportional to the ratio of the mismatched scales, which can be orders of magnitude.

**Prevention:**
- Dimensional analysis tests: for every nondimensional quantity, verify `nondim_value * scale_factor == dimensional_value` with known physical values.
- The MMS tests use nondimensional quantities directly, so they do NOT catch nondimensionalization errors. Separate dimensional analysis tests are essential.
- Pin physical constants in test fixtures and compare against NIST/CODATA values.

**Detection:** Compare PDE solver output (dimensional) against analytical solutions or literature values for standard electrochemical systems. If the limiting current is off by a factor of ~26 (V_T at 25C), suspect a missing V_T scaling.

### Pitfall 4: Convergence Rate "Verification" in the Pre-Asymptotic Regime

**What goes wrong:** The mesh is too coarse to reach the asymptotic convergence regime. Observed rates look approximately correct (e.g., 1.7 instead of 2.0 for L2) but this is coincidence, not verification.

**Why it happens:** Nonlinear problems (like PNP-BV) with boundary layers require fine meshes to resolve the layer before asymptotic convergence sets in. The existing MMS results show clean rates even at N=8, suggesting the manufactured solutions were well-chosen (smooth, no sharp boundary layers). But if the MMS parameters are changed (e.g., larger `beta_c` for steeper boundary layers), pre-asymptotic effects will appear.

**Consequences:** False confidence that the solver is verified. In reality, the solver may have bugs that only manifest at fine resolution.

**Prevention:**
- Use the R-squared metric from log-log linear regression. R-squared > 0.99 indicates clean asymptotic behavior.
- Always use at least 4 mesh levels (existing code uses 5: N=8,16,32,64,128).
- If R-squared < 0.99, either refine further or choose smoother manufactured solutions.

**Detection:** Convergence rates that vary significantly between mesh levels (e.g., 1.5 at N=8->16 but 1.9 at N=64->128) indicate pre-asymptotic behavior. The consecutive log-ratio rates should stabilize as h decreases.

## Moderate Pitfalls

### Pitfall 5: Surrogate Fidelity Tested Only at Training Points

**What goes wrong:** The surrogate-vs-PDE comparison is performed at the exact parameter values used to train the surrogate. The surrogate trivially matches (it was optimized to fit these points), so the test gives false confidence about interpolation accuracy.

**Prevention:** Use Latin Hypercube samples that span the parameter space, including points between training samples. The existing `TestSurrogateVsPDEConsistency` test uses 3 parameter sets (truth + 2 perturbations) -- this is minimal. Expand to 20-50 LHS samples for publication.

### Pitfall 6: Parameter Recovery Tolerance Too Generous

**What goes wrong:** The recovery test allows 5-10% relative error, which masks systematic bias. The inferred parameters may consistently be biased in one direction (e.g., k0 always overestimated by 3%).

**Prevention:** Report bias (signed error) separately from accuracy (unsigned error). If all 5 multistart runs overestimate k0 by ~3%, there is a systematic issue even if each individual error is within tolerance. Check the sign distribution of errors across runs.

### Pitfall 7: Ignoring PETSc/SNES Solver Tolerance Contamination

**What goes wrong:** The nonlinear solver tolerances (`snes_atol`, `snes_rtol`) are too loose, so the "computed solution" has solver error on top of discretization error. The convergence rate study then measures a mix of discretization and solver error.

**Prevention:** The existing MMS script uses very tight tolerances (`snes_atol=1e-12`, `snes_rtol=1e-12`, direct solver MUMPS). This is correct. Do not relax these tolerances for MMS tests even though they make solves slower. The tight tolerances ensure solver error is negligible compared to discretization error.

**Detection:** If convergence rates plateau (stop improving with mesh refinement), solver tolerance is contaminating the results.

### Pitfall 8: Firedrake errornorm Projection Artifacts

**What goes wrong:** `fd.errornorm` internally projects the exact solution into the same function space as the computed solution before computing the norm. If the exact solution involves expressions that cannot be exactly represented in the function space (e.g., cos(pi*x) in CG1), the projection introduces interpolation error that is NOT part of the solver error.

**Prevention:** The existing code projects the exact solution via `c_exact_func.interpolate(c_exact)` and then calls `fd.errornorm(c_exact_func, c_h_sol, ...)`. This is the correct approach because both arguments are in the same function space, avoiding re-projection inside `errornorm`. Keep this pattern.

## Minor Pitfalls

### Pitfall 9: Hardcoded MMS Parameters Hiding Edge Cases

**What goes wrong:** The manufactured solution parameters (A=0.2, beta=3.0, etc.) are chosen to be "nice" and may not exercise edge cases in the solver (e.g., near-zero concentrations, large gradients).

**Prevention:** Consider adding one MMS case with more extreme parameters (larger A, smaller c0) to test robustness. But do not make the manufactured solution so extreme that it violates positivity (|A| must be less than c0).

### Pitfall 10: pytest-regressions Baseline Drift

**What goes wrong:** Baselines are regenerated after a code change, accepting new (potentially wrong) values as the reference. The regression test then passes even though results changed.

**Prevention:** Baseline regeneration (`--force-regen`) should be a deliberate, reviewed action. Document the expected values in comments alongside the test so a human can sanity-check the baseline.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| MMS pytest wrappers | Pitfall 1 (sign errors) if MMS code is modified | Do not modify MMS weak forms unless re-deriving corrections from scratch |
| Convergence rate assertions | Pitfall 4 (pre-asymptotic) | Require R-squared > 0.99, not just rate ~ expected |
| Surrogate fidelity testing | Pitfall 5 (training points only) | Use LHS samples spanning full parameter space |
| Parameter recovery expansion | Pitfall 6 (too-generous tolerance) | Report bias and accuracy separately |
| GCI computation | Richardson extrapolation assumes monotone convergence | Verify monotonicity before applying formula |
| Reproducibility baselines | Pitfall 10 (baseline drift) | Review baselines before accepting regeneration |
| Nondim verification expansion | Pitfall 3 (silent dimensional errors) | Test against known analytical solutions with physical units |

## Sources

- Existing codebase analysis: `mms_bv_convergence.py` sign convention comments, `test_v13_verification.py` tolerance rationale
- [Roache, P.J. "Code Verification by MMS"](https://www.researchgate.net/publication/278408318_Code_Verification_by_the_Method_of_Manufactured_Solutions) -- MMS methodology and common errors (HIGH confidence)
- [ASME V&V 20-2009](https://www.osti.gov/servlets/purl/1368927) -- Richardson extrapolation and GCI (HIGH confidence)
- [Firedrake errornorm source](https://www.firedrakeproject.org/_modules/firedrake/norms.html) -- projection behavior (HIGH confidence)
- PROJECT.md constraint: "The entire codebase was built quickly with AI assistance, including an MMS reference that hasn't been independently verified"
