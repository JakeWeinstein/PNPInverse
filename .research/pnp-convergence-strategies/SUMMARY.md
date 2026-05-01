# PNP Convergence Strategies: Research Synthesis

## Executive Summary

The Firedrake-based PNP-BV solver fails at dimensionless electrode potentials phi_hat > ~4 V_T. The root cause is **not** the Butler-Volmer boundary (which uses applied potential, not phi, making its Jacobian well-conditioned) and **not** insufficient mesh resolution (Ny=400, beta=5 does not help). The root cause is the **singular perturbation in the Poisson equation**: the coefficient (lambda_D/L)^2 ~ 1.8e-7 forces a razor-thin boundary layer where co-ion concentrations drop to exp(-4) ~ 0.018 of bulk, creating a near-singular Jacobian that crashes the MUMPS direct solver.

**The single highest-impact fix is the log-concentration transform** (u_i = ln(c_i)), which converts exponential concentration variation into linear variation, guarantees positivity, and is straightforward to implement in Firedrake UFL. This is the consensus recommendation across all four research sources.

### Ranked Recommendations

| Priority | Strategy | Expected Impact | Implementation Effort |
|----------|----------|----------------|----------------------|
| 1 | Log-concentration transform (u_i = ln(c_i)) | Very High | Low-Medium |
| 2 | Block fieldsplit preconditioning (PCFIELDSPLIT) | High | Low |
| 3 | Adaptive voltage continuation (SER strategy) | High | Low |
| 4 | SNES diagnostic instrumentation | Medium | Very Low |
| 5 | Analytical EDL boundary conditions | High | Medium-High |
| 6 | Voltage-dependent initial dt for PTC | Medium | Very Low |
| 7 | EAFE/Scharfetter-Gummel stabilization | Medium | Medium |
| 8 | Steric corrections (Bikerman model) | Medium | Medium |

---

## 1. Root Cause Analysis

Four independent investigations converge on the same diagnosis.

### 1.1 What the Codebase Shows

The solver uses standard concentration variables c_i with the drift-diffusion form:
```python
Jflux = D[i] * (grad(c_i) + c_i * grad(drift))    # drift = em * z[i] * phi
```

The Poisson equation has coefficient eps_coeff = (lambda_D/L)^2 ~ 1.8e-7:
```python
F_res += eps_coeff * dot(grad(phi), grad(w)) * dx
F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
```

This is a textbook singular perturbation. The Poisson Laplacian is multiplied by ~10^-7 while the charge source is O(1), forcing the solution to develop a boundary layer of width ~lambda_D where phi changes rapidly.

**Key finding from codebase analysis**: BV clipping IS triggering (eta clips to 50, meaning exp(50) ~ 5e21), but BV is not the stiffness source. With `use_eta_in_bv=True`, the overpotential is computed from the applied potential (a constant), not from the solution variable phi. The BV flux depends only on surface concentrations, not phi. The stiffness lives in the interior Poisson equation's coupling to near-zero concentrations in the depletion zone.

### 1.2 Why Mesh Refinement Does Not Help

The extended voltage tests tried Ny=400/beta=5, which produces a first cell of ~0.01 pm -- far below the Debye length. Yet convergence did not improve. This is because the problem is algebraic, not spatial:

- The mesh already resolves the EDL (even Ny=200/beta=3 gives h_min ~ 12.5 nm vs lambda_D ~ 43 nm)
- The condition number of the stiffness matrix scales with exp(z*phi_hat), independent of mesh size
- Near-zero concentrations (c ~ 0.018 at phi_hat=4) create near-singular rows in the Jacobian
- MUMPS encounters near-zero pivots, produces garbage Newton directions, line search fails

As the web-mesh agent notes: "the stiffness matrix conditioning in PNP problems depends more on the exponential variation of concentrations across the domain than on the mesh size alone."

### 1.3 Why Reducing z to 0.9 Helps (+70 mV Extension)

Setting z=0.9 reduces the charge coupling by 10%, which has two effects:
1. The Poisson source term is 0.9x weaker, slightly relaxing the singular perturbation
2. The Boltzmann factor exp(-z*phi) becomes exp(-0.9*phi), so depletion is less extreme

At phi_hat=4: concentration drops from exp(-4)=0.018 at z=1.0 to exp(-3.6)=0.027 at z=0.9 -- a 50% increase in the minimum concentration. This is enough to push the Jacobian from singular to barely non-singular for another ~70 mV.

### 1.4 The Failure Cascade

From the codebase analysis:
1. Phase 1 (z=0) succeeds easily -- no charge coupling
2. Phase 2 z-ramp: direct jump to z=1.0 fails; binary search finds z ~ 0.8-0.95
3. At critical z, EDL depletion makes the Jacobian nearly singular
4. MUMPS produces a bad Newton direction
5. L2 line search (maxlambda=0.5) cannot find descent
6. SNES throws ConvergenceError; z-ramp gets stuck below z=1.0

---

## 2. The Slotboom Contradiction -- Resolved

The web-transforms agent says Slotboom is bad (ill-conditioning from exponentially varying diffusion coefficients). The literature agent ranks it #1. Both are correct, but they are talking about different contexts.

### When Slotboom Helps (Literature Agent's Context)

The Slotboom transform n_i = c_i * exp(z_i * phi) eliminates the convection term entirely, converting the NP equation into pure diffusion: div(D_i * exp(-z_i*phi) * grad(n_i)) = 0. This is beneficial because:
- The operator becomes self-adjoint (symmetric), enabling CG instead of GMRES
- No upwinding or SUPG stabilization is needed
- Structure-preserving properties (positivity, energy dissipation) are proven
- In the **Scharfetter-Gummel finite volume** context, the exponential coefficients are handled analytically via the Bernoulli function B(x) = x/(exp(x)-1), so they never enter the stiffness matrix

The literature references (Ding et al. 2022, Xie & Lu 2020, Liu & Wang 2020) work primarily with SG-type discretizations or harmonic-mean coefficient averaging, which tame the exponential variation.

### When Slotboom Hurts (Web-Transforms Agent's Context)

In a **standard Galerkin FEM** context (which is what the Firedrake solver uses), the Slotboom diffusion coefficient D_bar_i = D_i * exp(-z_i*phi) enters the stiffness matrix directly. At phi_hat=4 with monovalent ions, these coefficients span exp(8) ~ 3000x. As Lu et al. (2010) state: "the transformed formulation will always lead to an ill-conditioned stiffness matrix."

### Resolution

**Slotboom is the wrong transform for this codebase.** The solver uses standard CG elements with Galerkin assembly and a direct MUMPS solve. Without SG-type exponential fitting or harmonic-mean coefficient treatment, the Slotboom transform trades advection instability for ill-conditioning -- not a net gain at phi_hat > 4. The log-concentration transform is the correct choice for this solver architecture.

---

## 3. The Log-Concentration Transform (Consensus Recommendation)

All four research sources agree: switching from c_i to u_i = ln(c_i) is the most promising single change. This is the intersection of the web-transforms analysis, the literature's energy-stable discretization theory (Metti, Xu, Liu 2016), the mesh agent's recommendation, and the codebase analysis's identification of near-zero concentrations as the failure mode.

### 3.1 Why It Works for This Problem

| Regime | Standard c_i | Log-transformed u_i = ln(c_i) |
|--------|-------------|-------------------------------|
| Bulk (c ~ 1) | c = 1.0 | u = 0.0 |
| Depletion at phi_hat=4 | c = 0.018 (near-zero) | u = -4.0 (moderate) |
| Depletion at phi_hat=8 | c = 0.00034 (near-singular) | u = -8.0 (still moderate) |
| Positivity | Not guaranteed; clipping needed | Automatic: exp(u) > 0 always |
| Jacobian near depletion | Near-singular (1/c terms) | Well-conditioned (u varies linearly) |

The key insight from Metti, Xu, Liu (2016): u_i = ln(c_i) is the entropy variable, and the singularity of ln(c) as c -> 0 acts as a natural barrier. The discrete energy minimizer cannot approach zero concentration. This is precisely the regularization needed for the EDL depletion zone.

### 3.2 Implementation in Firedrake

The transform is straightforward in UFL. The current weak form:
```python
# Current: c_i as primary unknown
Jflux = D[i] * (grad(c_i) + c_i * grad(drift))
F_NP = inner(Jflux, grad(v)) * dx
```

Becomes:
```python
# Transformed: u_i = ln(c_i) as primary unknown
u_i = Function(V)      # primary unknown
c_i = exp(u_i)         # physical concentration, for Poisson and BV
Jflux = D[i] * c_i * (grad(u_i) + grad(drift))
F_NP = inner(Jflux, grad(v)) * dx
```

The Poisson source term uses c_i = exp(u_i) directly. BV boundary terms become:
```python
j_BV = k0 * exp(alpha_c * u_O) * exp((1-alpha_c) * u_R) * [exp(alpha*f*eta) - exp(-(1-alpha)*f*eta)]
```

All terms are exponentials of the primary variables. The Jacobian contributions are smooth and well-behaved.

### 3.3 Caveats

1. **Mass matrix becomes exp(u)-weighted**: The time derivative term is exp(u) * du/dt, so the mass matrix varies with the solution. Preconditioning with a mass-weighted Laplacian is recommended.

2. **Initial guess**: The initial guess must be converted: u_init = ln(c_init). For c_init = 1 (bulk), u_init = 0. Ensure c_init > 0 everywhere.

3. **BV surface concentrations**: Surface c values are recovered as exp(u_surface). The BV expression may need careful handling to avoid intermediate overflow when both u and eta are large.

4. **Convergence rate**: Newton convergence may change because the nonlinearity shifts from the coupling term to the exp(u) coefficient. The system is still nonlinear, but the nonlinearity is smoother.

### 3.4 Related: Electrochemical Potential Variable

If log-concentration alone is insufficient, the next step is to solve for mu_i = ln(c_i) + z_i * phi (the electrochemical potential). This is the Slotboom variable in log space: mu_i = ln(n_i). At equilibrium, mu_i = const everywhere, so the solver trivially converges. Near equilibrium, mu_i varies smoothly even when c_i and phi individually vary wildly.

The NP equation becomes: div(D_i * exp(mu_i - z_i*phi) * grad(mu_i)) = 0. The Poisson equation still involves c_i = exp(mu_i - z_i*phi), which reintroduces exponential coupling. A mixed formulation may be needed.

---

## 4. Block Preconditioning with PCFIELDSPLIT

The current solver uses a monolithic direct solve (MUMPS) for the full (4-species + Poisson) system. This is robust at low voltages but does not exploit the block structure of the PNP system.

### 4.1 The Block Structure

The PNP Jacobian has a natural 2x2 block form:
```
[ A_phi    B_phi_c  ] [ delta_phi ]   [ r_phi ]
[ B_c_phi  A_c      ] [ delta_c   ] = [ r_c   ]
```
where A_phi is the Poisson block (well-conditioned), A_c is the NP block (potentially ill-conditioned in depletion), and B are the coupling blocks.

### 4.2 PETSc Implementation

Firedrake exposes PETSc's PCFIELDSPLIT directly. A Schur complement or multiplicative splitting can precondition the Poisson and NP blocks separately:
```python
solver_parameters = {
    "snes_type": "newtonls",
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",  # or "schur"
    "fieldsplit_phi": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
    "fieldsplit_species": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
}
```

Ying et al. (2021) proved that block preconditioning for PNP yields eigenvalues bounded independently of mesh size. Schneider et al. (2021) demonstrated 6x speedup over monolithic direct solvers for electrochemical systems with BV coupling.

### 4.3 Why This Helps

Even with a direct solve, the fieldsplit approach can be more robust because:
- Each block is factored separately with better conditioning
- The multiplicative update applies corrections sequentially (Poisson first, then NP), which respects the one-way dominance of phi on c
- The Schur complement captures the coupling more accurately than a monolithic pivot

---

## 5. Adaptive Continuation Strategies

### 5.1 Current Approach: z-Ramp with Binary Search

The solver uses a two-phase approach:
1. Phase 1: Solve at z=0 (decoupled, easy)
2. Phase 2: Ramp z from 0 to 1 with geometric acceleration and binary search fallback

When _try_z(z_target) fails, the code bisects and retries. This is ad-hoc but functional. The solver gets stuck when the EDL depletion at z ~ 0.8-0.95 makes the Jacobian singular.

### 5.2 Improvements from the Literature

**Switched Evolution Relaxation (SER)** (Kelley & Keyes 1998): The pseudo-transient continuation time step should be updated as dt = dt_0 / ||F||, where ||F|| is the residual norm. This automatically shrinks dt when the residual is large (stiff regime) and grows it when the residual is small (near convergence). The current solver uses SER but the initial dt is not voltage-dependent.

**Recommendation**: Make dt_initial inversely proportional to exp(phi_hat) or to the estimated condition number. At phi_hat=4, start with dt_initial ~ dt_0 * exp(-phi_hat) to give the first Newton solve a chance to succeed.

**Adaptive voltage stepping** (Yan, Pugh & Dawson 2021): Instead of fixed z-ramp increments, use local truncation error estimates to control the voltage step. The VSBDF2 method handles the wide range of time scales automatically. Code available at: https://github.com/daveboat/vssimex_pnp

**Continuation in exchange current**: Instead of (or in addition to) ramping z, ramp k0 (exchange current density) from a small value to the physical value. This gives the solver time to develop the correct EDL structure before full BV kinetics are imposed.

### 5.3 Voltage-Dependent Initial dt (Quick Win)

The current code uses the same dt_initial regardless of phi_hat. From the codebase analysis: "If the SNES fails on the FIRST step of a z-ramp attempt, the code immediately declares failure without any recovery attempt."

Fix: Set dt_initial = dt_base * min(1, C / exp(z * phi_hat)) for some calibration constant C. This is a one-line change that could extend convergence by several V_T.

---

## 6. SNES Diagnostic Instrumentation (Quick Win)

The codebase catches ConvergenceError but cannot distinguish between:
- MUMPS factorization failure (near-singular Jacobian)
- Line search failure (bad Newton direction)
- SNES divergence (residual blowup)

Adding monitoring flags is trivial and essential for debugging:
```python
solver_parameters.update({
    "snes_monitor": None,
    "snes_linesearch_monitor": None,
    "snes_converged_reason": None,
    "ksp_converged_reason": None,
})
```

This will print the SNES residual at each iteration, the line search step length, and the specific reason for convergence/divergence. No code changes beyond adding these keys.

---

## 7. Analytical EDL Boundary Conditions (Medium-Term)

Taherkhani, Brogioli & La Mantia (2022) propose replacing the numerically-resolved EDL with an analytical Gouy-Chapman-Stern solution. The analytical EDL solution provides boundary conditions at the outer Helmholtz plane for the bulk FEM calculation, eliminating the need to mesh the Debye layer.

### 7.1 How It Works

1. Define the computational domain from the outer Helmholtz plane (OHP) outward, not from the electrode surface
2. The analytical GCS theory provides concentrations and potential at the OHP as functions of the applied voltage and bulk concentrations
3. These become Dirichlet or Robin BCs for the bulk FEM solve
4. The bulk domain is electroneutral (or nearly so), removing the singular perturbation entirely

### 7.2 Benefits

- Eliminates the eps_coeff ~ 1.8e-7 singular perturbation from the FEM system
- No need for Debye-length mesh resolution
- The BV kinetics are absorbed into the analytical boundary condition
- The bulk PNP solve is well-conditioned at any voltage

### 7.3 Limitations

- Requires deriving the analytical EDL solution for 4 charged species (nontrivial for non-binary electrolytes)
- Assumes the EDL is in quasi-equilibrium with the bulk (valid for steady-state)
- Does not capture transient EDL dynamics during pseudo-transient continuation
- Loses spatial resolution within the EDL (cannot output concentration profiles there)

---

## 8. Additional Strategies

### 8.1 Energy-Stable Discretization

The Metti-Xu-Liu (2016) framework provides unconditional energy stability for PNP. The discrete free energy decreases monotonically, preventing spurious oscillations. Combined with the log-concentration transform, this ensures:
- Positivity of concentrations (entropy barrier)
- Energy dissipation (no runaway growth)
- Unique solvability of the discrete problem

For the current solver, the log-transform alone captures most of this benefit. Full energy-stable discretization adds theoretical guarantees but requires more extensive code changes.

### 8.2 Steric Corrections (Bikerman Model)

At high potentials, the Gouy-Chapman model predicts unphysically compressed ion layers. Adding finite ion size effects via the Bikerman excess chemical potential mu_ex = -ln(1 - sum v_j * c_j) bounds concentrations from above, smoothing the profile and improving Newton convergence.

For dilute solutions typical in PEM fuel cells, the steric correction is small and the primary benefit comes from the log-transform. But for concentrated electrolytes or very high potentials, steric effects provide an additional regularization.

### 8.3 XFEM Enrichment

If the analytical form of the boundary layer is known (approximately exp(-x/lambda_D)), the FE basis can be enriched with this function. The mesh no longer needs to resolve the layer explicitly. An XFEM for PNP has been demonstrated in 1D and 2D (ScienceDirect, 2024). This is high-effort but could be transformative for extreme parameter regimes.

### 8.4 FAS Nonlinear Multigrid

If monolithic Newton continues to fail after the log-transform, Liu, Yang & Shu (2023) propose Full Approximation Storage multigrid as an alternative to Newton. FAS provides global convergence where Newton may stagnate. This is a fallback architecture, not a first-line strategy.

---

## 9. Mesh Strategies -- When They Matter

Although mesh refinement alone does not fix the convergence failure, proper meshing remains important for solution accuracy and complements the algebraic strategies above.

### 9.1 Current Mesh Assessment

The power-law grading formula x_i = (i/N)^beta produces:

| Config | h_min (nondim) | h_min (physical) | Elements in EDL |
|--------|---------------|-----------------|----------------|
| Ny=200, beta=3 | 1.25e-7 | 12.5 nm | ~3-4 |
| Ny=400, beta=3 | 1.56e-8 | 1.56 nm | ~7-8 |
| Ny=200, beta=5 | 3.13e-12 | 0.31 pm | many, but numerically problematic |
| Ny=400, beta=5 | 9.77e-14 | 0.01 pm | extreme; floating-point issues likely |

The beta=5 configurations produce sub-angstrom cells that can cause floating-point problems in FE assembly. The recommended range from the literature is 5-10 elements per Debye length, with grading ratio r = 1.15-1.25. Ny=400/beta=3 is adequate; going beyond this has diminishing returns.

### 9.2 When Mesh Refinement Does Help

After implementing the log-transform, if the solver converges but the solution has spatial accuracy issues (oscillatory profiles, inaccurate currents), then mesh refinement becomes relevant. The Animate toolkit (metric-based anisotropic adaptation) and Goalie (goal-oriented AMR targeting electrode current error) are available in Firedrake for this purpose.

---

## 10. Implementation Roadmap

### Phase 1: Quick Wins (hours, no architecture changes)

1. **Add SNES diagnostics**: snes_monitor, snes_linesearch_monitor, snes_converged_reason, ksp_converged_reason. Identify whether failure is MUMPS factorization, line search, or residual blowup.

2. **Voltage-dependent initial dt**: Set dt_initial = dt_base * min(1, C / exp(z * phi_hat)) with C calibrated from the phi_hat=4 failure threshold.

3. **Tighten BV exponent clip**: Reduce from 50 to 20-30. At clip=50, exp(50) ~ 5e21 is still extreme. With the BV using applied potential (not phi), the clip mainly affects the concentration-dependent prefactor.

### Phase 2: Log-Concentration Transform (days, moderate code changes)

4. **Implement u_i = ln(c_i) transform in forms.py**: Replace c_i as primary unknown with u_i. Reconstruct c_i = exp(u_i) for Poisson source and BV terms. Update initial conditions, BCs, and output postprocessing.

5. **Update continuation logic**: The z-ramp initial guess must be converted to log-space. The observable (integrated BV flux) still uses physical concentrations, so observable_form needs exp(u_i) substitution.

6. **Validate**: Run the extended voltage test suite. The log-transform should extend convergence well past phi_hat = 4 V_T since u varies linearly where c varied exponentially.

### Phase 3: Solver Architecture (days-week)

7. **Block fieldsplit preconditioning**: Split into {phi} and {u_1,...,u_4} blocks. Use multiplicative or Schur complement factorization. Keep MUMPS for each block's inner solve initially; later explore AMG.

8. **Adaptive continuation**: Replace fixed z-ramp increments with SER-controlled stepping. Add exchange current continuation (k0 ramp) as a third continuation dimension.

### Phase 4: Advanced Strategies (if needed)

9. **Analytical EDL BCs**: Derive GCS solution for the 4-species system and implement as Robin BCs at the OHP.

10. **EAFE stabilization**: Implement edge-averaged finite elements for the log-transformed NP equation to handle residual advection dominance.

---

## 11. Key References (Grouped by Strategy)

### Log-Concentration / Energy-Stable Methods
- Metti, Xu, Liu (2016). Energetically stable discretizations for charge transport. J Comput Phys 306:1-18.
- Sun & Sun (2022). High-order space-time FEM for PNP: positivity and unconditional energy stability. CMAME 401:115515.
- Liu & Wang (2020). Fully discrete positivity-preserving FD scheme for PNP. Numer Math 145:77-115.

### Slotboom / Scharfetter-Gummel
- Ding et al. (2022). Convergence analysis of Slotboom-based methods for PNP. arXiv:2202.10931.
- Xie & Lu (2020). Effective FE iterative solver for PNP ion channel model. SIAM J Sci Comput 42(6):B1490-B1516.
- Lu et al. (2010). PNP equations for biomolecular diffusion-reaction. J Comput Phys 229:6979-6994. [Slotboom ill-conditioning warning]

### Block Preconditioning
- Ying et al. (2021). Block preconditioner for PNP. J Comput Phys 428:110015.
- Schneider et al. (2021). Segregated approach for Li-ion battery with block preconditioners. J Sci Comput 86:11.

### Analytical EDL / Multi-Scale
- Bazant, Chu & Bayly (2005). Current-voltage relations for electrochemical thin films. SIAM J Appl Math 65(5):1463-1484.
- Taherkhani et al. (2022). Coupling analytical DDL with FEM bulk. Electroanalysis 34:e202200257.

### Pseudo-Transient Continuation
- Kelley & Keyes (1998). Convergence analysis of pseudo-transient continuation. SIAM J Numer Anal 35(2):508-523.
- Yan, Pugh & Dawson (2021). Adaptive time-stepping for PNP. Appl Numer Math 163:254-269. Code: https://github.com/daveboat/vssimex_pnp

### Firedrake-Specific
- Roy et al. (2024). EchemFEM: Firedrake-based electrochemical transport. JOSS 9(97):6531. https://github.com/LLNL/echemfem
- Vetter et al. (2026). Thermodynamically consistent electrolyte model in FEniCSx. Comput Phys Commun 319.

### Mesh Adaptation
- Flux-based moving mesh for PNP (2024). J Comput Phys. ScienceDirect.
- XFEM for Nernst-Planck-Poisson (2024). ScienceDirect.
- Animate/Goalie/Movement: https://mesh-adaptation.github.io/
