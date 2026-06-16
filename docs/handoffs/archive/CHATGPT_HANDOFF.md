# PNP-BV Inverse Solver — Brainstorming Handoff

**Goal of this document:** Give you (ChatGPT) enough physics, math, numerical, and history context to brainstorm new ideas for making k0 (and k0_2) identifiable in the inverse problem. Everything below is honest current state from a multi-month investigation.

---

## 1. Physical system

We're modeling a rotating-disk-electrode-style oxygen reduction reaction (ORR) experiment. A platinum electrode in aqueous perchloric acid (HClO4) solution at pH 4 reduces dissolved O2 via two coupled Butler–Volmer (BV) electron-transfer reactions:

- **R1:** O2 + 2H+ + 2e- → H2O2,  E_eq,1 = 0.68 V vs RHE
- **R2:** H2O2 + 2H+ + 2e- → 2 H2O, E_eq,2 = 1.78 V vs RHE

Species (4-species formulation): O2, H2O2, H+, ClO4- (the supporting electrolyte counter-ion). 1-D domain of length L_ref = 100 µm normal to the electrode; bulk values held fixed at x = L_ref by Dirichlet BC; BV flux BC at the electrode (x = 0).

Bulk concentrations (mol/m³): c_O2 = 0.5, c_H2O2 = 0.0, c_H+ = c_ClO4- = 0.1 (electroneutrality).

Diffusivities (m²/s): D_O2 = 1.9e-9, D_H2O2 = 1.6e-9, D_H+ = 9.31e-9, D_ClO4 = 1.79e-9.

The experimental observable is current density (CD = total electron flux at the electrode) and "peroxide current" (PC = the part of the current going to R1/R2 partitioning) as a function of applied voltage. Goal: from a measured I–V curve, infer k0_1, k0_2, α_1, α_2 (4 unknowns).

---

## 2. Forward PDE (Poisson–Nernst–Planck + BV)

### 2.1 Strong form (dimensional)

For each species i:

    ∂c_i/∂t = ∇ · J_i,           J_i = D_i (∇c_i + (z_i F / RT) c_i ∇φ)

Poisson:

    -ε ∇²φ = F · Σ_i z_i c_i

BC at electrode (x = 0):

    J_i · n = -Σ_j s_{i,j} · r_j,    where r_j is the BV rate of reaction j

with the Butler–Volmer rate

    r_j = k0_j · ( c_red^surf · exp(-α_j n_e η_j / V_T)  -  c_ox^surf · exp((1-α_j) n_e η_j / V_T) )

where η_j = φ_applied - E_eq,j (overpotential of reaction j), n_e = 2 electrons, V_T = RT/F = 25.69 mV, and s_{i,j} are stoichiometric coefficients.

BC at bulk (x = L_ref): c_i = c_i^bulk and φ = 0 (Dirichlet). Discretization: CG1 finite elements via Firedrake on a graded interval mesh with finest element ≈ 0.01 nm at the electrode (Debye length is ≈ 30 nm at this ionic strength, so the EDL is over-resolved geometrically).

### 2.2 Nondimensionalization

Length L_ref, diffusivity D_ref = D_O2, concentration c_scale = c_O2_bulk. Dimensionless k̂0 = k0 · L_ref / D_ref. With these scales:

- K0_HAT_R1 ≈ 1.263e-3 (from k0_1 = 2.4e-8 m/s)
- K0_HAT_R2 ≈ 5.263e-5 (from k0_2 = 1.0e-9 m/s)
- The Poisson singular perturbation parameter ε̂ ≈ 1.8e-7. **This is the source of all numerical pain.**

Time-stepping: backward Euler, monolithic Newton on (c, φ) with MUMPS LU.

### 2.3 The inverse problem

Given a target I–V curve cd_target(V_k), k = 1..N_V, find θ = (k0_1, k0_2, α_1, α_2) minimizing

    J(θ) = Σ_k (cd_sim(V_k; θ) - cd_target(V_k))²    [+ optional PC residual + Tikhonov]

Adjoint gradients via dolfin-adjoint / pyadjoint work and have been verified against finite differences to ~1% (when the forward solver converges cleanly).

---

## 3. Numerical machinery and "charge continuation"

The Poisson equation is **singularly perturbed** (ε̂ ~ 1e-7). Cold-starting Newton on the fully charged system from c = c_bulk, φ = 0 diverges immediately. The solver uses two-stage continuation:

- **Phase 1 (z = 0, neutral):** Solve with all z_i set to 0 (no electromigration coupling, Poisson decouples). This converges robustly everywhere.
- **Phase 2 (z-ramp):** Continuation parameter z ∈ [0, 1] scales electromigration and Poisson source. Step from z = 0 → z = 1 in adaptive increments using the previous z's solution as IC. The "charge continuation" terminates at z_achieved ≤ 1; if z_achieved < 1.0 - 1e-6, we declare partial convergence.

This pattern (along with adaptive dt, exponent clipping at η/V_T = ±50, surface concentration floor c_surf = max(c, ε_c), and softplus regularization options) is implemented in `Forward/bv_solver/`.

### 3.1 Where charge continuation breaks (the central numerical obstacle)

| V_RHE      | z_achieved (standard 4sp) | Status   |
|------------|--------------------------|----------|
| -0.30 V    | 1.00                     | FULL     |
|  0.00 V    | 1.00                     | FULL     |
|  0.10 V    | 1.00                     | FULL     |
|  0.15 V    | 0.79                     | PARTIAL  |
|  0.20 V    | 0.60                     | PARTIAL  |
|  0.30 V    | 0.40                     | PARTIAL  |
|  0.40 V    | 0.30                     | PARTIAL  |
|  0.60 V    | 0.20                     | PARTIAL  |

**Root cause confirmed by diagnostics:** ClO4- (the co-ion) goes catastrophically negative in the unresolved Debye-layer profile during the z-ramp. At z = 0.80, V = 0.15 V, c_ClO4- reaches **-6.1e+04** (should be ~1e-5). CG1 is not positivity-preserving; the depletion zone (where c_ClO4- → 0 in the EDL at anodic potentials) causes the linearization to step to negative values, and then BV exponential rates blow up nonlinearly and Newton diverges.

Even tiny z-steps from a partial solution (e.g. z = 0.79 → z = 0.7905) fail. SNES line search saturates at λ = 0.21 with residual *divergence* at every step — this is a genuine Jacobian near-singularity from the Poisson singular perturbation, not just bad damping.

**The full charge-continuation z-convergence window is V_RHE ∈ [-0.50, +0.10] V.** That window is **cathodic + onset edge only**, not the onset itself.

### 3.2 Why this matters for inversion

- **α (Tafel slope) is identifiable mostly in the kinetic regime:** V near the onset (V_RHE 0.15–0.4 V), where current is rate-limited by BV.
- **k0 sets the absolute Tafel intercept:** identifiable in the same kinetic regime.
- **Cathodic region is transport-limited:** current saturates at the diffusion-limited value ~ -0.18 mA/cm², which depends on D and c_bulk, *not* on k0/α. Loss surface w.r.t. k0 is essentially flat there.

So the regime where the parameters are identifiable is exactly the regime where the standard solver fails at z = 1.

---

## 4. Things tried (chronological, with outcomes)

### 4.1 v13 — Surrogate + multi-observable PDE refinement (BASELINE)
- Surrogate (NN ensemble, RBF, POD-RBF, GP, PCE) trained on ~5000 forward solves over a parameter box.
- Inverse: surrogate-based global search → PDE-refined local optimization on a full-cathodic grid (V_RHE ∈ [-1.2, -0.03] V, ~40 points).
- **Result: 4–5% k0 error at low noise but 5–28% range across 5 noise seeds (median 23%).** Pipeline works but k0_2 is consistently the worst-recovered parameter and the 99th-percentile error fails publication thresholds.
- v13 audit fixed 14 pipeline bugs (commit `0fe7446`).

### 4.2 v14 — Pipeline redesign (PARTIAL)
- Phase 7 (baseline diagnostics): 3 plans completed — multi-seed wrapper, profile likelihood study (showed χ² flat ridges in (k0, α) plane), 1-D sensitivity sweeps.
- Phases 8–11 (ablation, objective redesign, multi-pH, pipeline build) deferred. 11 of 15 requirements unfinished.

### 4.3 v15 — Codex-audited fixes
- 9 numerical bugs found and fixed in parallel Claude+Codex audit (commit `6cff9da`). Mostly indexing, scaling, IC cache invalidation issues. Pipeline didn't fundamentally change behavior.

### 4.4 v16 — Adjoint gradients + shared-memory IC cache
- Adjoint via pyadjoint, verified against FD to ~1%.
- Shared-memory IC cache across optimizer evals (continuation seeds reused across (k0, α) trials within a small ball).
- Overnight training pipeline for surrogate retrain at fresh parameters.

### 4.5 v17 — Physical E_eq fix and identifiability diagnosis
**Voltage mapping bug found:** earlier code had E_eq = 0 for both reactions, which accidentally placed the onset region in the v16 grid. With physical E_eq (0.68, 1.78), the onset is at η_1 = 0, i.e., V_RHE ≈ 0.68 V — far outside the z-convergence window.

Recovery diagnostics on the cathodic grid V_RHE ∈ [-0.5, +0.1] V (the only place the solver converges):

| Init     | k0_1 err | k0_2 err | α_1 err | α_2 err |
|----------|---------:|---------:|--------:|--------:|
| TRUE     | 2.7%     | 0.6%     | **67%** | **62%** |
| 20% off  | 13%      | 21%      | 42%     | 39%     |
| 2× off   | 93%      | 50%      | 48%     | 48%     |

Optimizer drifted α from 0.627 → 0.207 *even when started at TRUE parameters*. This is the transport-limited regime: current saturates and α loses its gradient signal.

A "robust forward solver" (parallelized z-ramp with IC cache reused across param perturbations) ran cleanly at 10/10 voltage points; with proper warm-start, **α is recovered to <1% in [-0.5, +0.1] V at 10% offset, but k0 is stuck at the initial offset (loss is flat w.r.t. k0).**

A "hybrid z=0/z=1" solver (z=0 above V_RHE = 0.10 V, z=1 below) covered onset shape, but the z=0 currents are ~50% of z=1 currents (missing electromigration), so the curve has a discontinuity at the seam.

Conclusion at end of v17: **k0 is non-identifiable from cathodic I–V data alone, α is identifiable but only with a working solver across the kinetic-to-transport transition.**

### 4.6 v18 — Try to extend the z=1 window
Investigation `StudyResults/v18_convergence_extension_log.md` (2026-04-07).

**Approaches that failed:**
- Naïve log-concentration transform u = ln c: H2O2 starts at c = 0, so u = ln(1e-20) = -46 creates extreme stiffness from the start.
- Voltage continuation at z = 1 from the V = 0.10 V edge: fails at V = 0.118 V (25 mV step), even at 1 mV steps.
- Aggressive Newton damping (maxlambda = 0.005): fails at step 1.
- Trust-region Newton: 274 iterations, diverges.
- Small dt = 0.01: fails at step 1.
- Gummel operator splitting (NP / Poisson alternating): same oscillations.
- Positivity penalty on c_ClO4-: makes things worse (penalty gradients fight physics).
- Lower BV exponent clip (30, 20): distorts R_1 physics at cathodic voltages.

**Approach that "worked" but distorted physics — artificial diffusion:**
Streamline artificial diffusion on the co-ion only:

    D_art = d_art_scale · h · |z_i D_i em| · |∇φ|

with `d_art_scale = 0.001` on ClO4- only: 35/39 voltage points reach z = 1 across V_RHE ∈ [-0.30, +0.725] V. **But the resulting onset I–V curve is FLAT** — artificial diffusion smears out the kinetic transition. Adjoint gradients work through the stabilization (verified at V_RHE = 0.20 V: gradient ≈ 0 at TRUE params, finite and correctly signed at 20% offset for all 4 parameters).

Inference test (L-BFGS-B on V_RHE ∈ [-0.20, +0.40] V, 9 pts, 20% offset):
- α_1: 0.8% error ✓
- α_2: 0.4% error ✓
- k0_1: **21% (stuck at init)** ✗
- k0_2: **20% (stuck at init)** ✗

**Best v18 forward model — 3-species + Boltzmann ClO4- background:**
Drop ClO4- as a dynamic species; replace its contribution to Poisson with the Boltzmann equilibrium

    -ε ∇²φ ∝ z_H+ c_H+  +  z_ClO4- · c_ClO4-_bulk · exp(-z_ClO4- φ̂)

This is exact in the limit where the co-ion has no kinetic role (it doesn't react), is in local thermodynamic equilibrium with φ, and resolves the EDL self-consistently via the analytic Boltzmann factor instead of needing to discretize a steep CG1 profile that goes negative.

Forward results: 11/12 points converge at z = 1 over V_RHE ∈ [-0.30, +0.60] V. Onset shape is *correct* (cd from -0.18 at V = -0.3 to -0.14 at V = 0.15 to -0.02 at V = 0.25). 0.1–0.3% error vs the 4sp standard solver where they overlap. **Physics is exact and onset is captured for the first time.**

But adjoint inference on this model (9 points [-0.10, +0.50] V, 20% offset):
- α_1: 0.8% ✓
- α_2: 3.6% ✓
- k0_1: **21.3% (stuck)** ✗
- k0_2: **20.2% (stuck)** ✗

The optimizer reduced loss 860× by adjusting α while *barely moving k0*. **Confirms: even with correct onset physics over the full kinetic region, the data alone has a k0–α ridge degeneracy.**

### 4.7 The latest breakthrough (2026-04-13) — log-c + seeded H2O2 + Boltzmann + Tikhonov
Investigation `docs/k0_inference_status.md` and `docs/HANDOFF.md`.

**Diagnosis of why "log-c didn't work" before:** R2 has E_eq,2 = 1.78 V, so η_2 = φ_applied - E_eq,2 is very negative everywhere in the operating window. With BV exponent clip at ±50, `exp(-α n_e η_2)` saturates at exp(50) ≈ 5e21 across the full V range. Combined with the c_surf concentration floor (1e-12), the spurious R2 sink rate is ≈ k0_2 · exp(50) · (c_H+/c_ref)² · 1e-12 ≈ 0.5 per unit time. When c_H2O2 → 0 at anodic V (R1 produces almost no H2O2), this spurious sink has nothing to balance and Newton finds a "solution" with c_H2O2 ≈ -0.69. CG1 oscillation around the near-zero state amplifies it.

**Recipe that works:**
1. **Log-c transform:** u_i = ln c_i for *all* dynamic species. c = exp(u) > 0 by construction.
2. **3-species + Boltzmann ClO4- background** (as above; ClO4- doesn't need positivity preservation because it's analytic).
3. **Seeded H2O2 IC:** u_H2O2 = ln(1e-4) = -9.2 instead of ln(0). Avoids log-of-zero singularity. 1e-4 is small enough to be physically negligible but large enough to be numerically benign.
4. **Voltage range V_RHE ∈ [-0.10, +0.30] V** (the onset region). Outside this, log-c without warm-starting fails (V = -0.30 has no good IC; V ≥ +0.40 has H2O2 runaway from BV mismatch).
5. **Tikhonov regularization on log k0:** J_total = J_data + λ · (log k0 - log k0_prior)² with λ = 0.01.

**Result on a single test case (k0_true = K0_HAT_R1, α_true = 0.627; 6 voltages; 2% Gaussian noise; init k0 +20%, α -10%):**

| Approach                          | k0 error    | α error |
|-----------------------------------|------------:|--------:|
| Unregularized Nelder-Mead         | **+1580%** (17× true) | -12% |
| Regularized NM (λ = 0.01, prior = true) | **-0.3%** | **+0.7%** |

The optimizer trace under regularization showed J_data and J_prior both decreasing smoothly. Without regularization, NM walks the (log k0, α) ridge unboundedly — slope d(log k0)/d(α) ≈ -47 (a 1% α change is offset by a 60% k0 change at the same data residual).

**This is the first time we have <1% recovery on both k0 and α.** Caveats:
- Only one noise seed (42).
- Prior is set to the true value (cheating). Real-world prior would be off by some factor (literature k0 for similar catalysts can be 0.3× to 3× off).
- Only k0_1 / α_1 inferred. k0_2 / α_2 untested in this recipe.
- Adjoint not yet verified through `forms_logc` + Boltzmann (NM was used, ~30–40 evals, ~30 min). L-BFGS-B with adjoint would be ~3× faster but adjoint correctness through the log transform isn't confirmed.

---

## 5. Summary of identifiability findings

This is the most important table:

| Voltage regime      | What converges        | What's identifiable from data |
|---------------------|----------------------|------------------------------|
| Cathodic [-0.5, -0.1] V | Standard 4sp z=1 | α (with warm-start IC cache); k0 NOT (transport-limited) |
| Onset [+0.1, +0.3] V | 3sp + Boltzmann (concentration formulation) | α; k0 still NOT (ridge degeneracy) |
| Onset [+0.1, +0.3] V | 3sp + Boltzmann + log-c + seeded H2O2 | α; k0 only with Tikhonov prior |
| Anodic > +0.4 V     | Nothing converges at z=1 (BV exp + EDL collapse) | — |

**The fundamental issue:** the BV form `r = k0 · exp(-α n_e η / V_T)` has a structural near-degeneracy. A change in k0 can be absorbed by a change in α through the identity `log r = log k0 + (-α n_e η / V_T)`. Over a narrow η range (the onset is only ~150 mV wide), the Hessian has a near-zero eigenvalue along the direction d(log k0)/d(α) = -n_e η_center / V_T ≈ -50, exactly matching the empirically observed ridge slope.

**Breaking this requires extra information beyond the I–V curve:**
- **Independent k0 measurement** (Tafel analysis on a subset of data, EIS exchange current measurement, literature value) → Tikhonov prior.
- **Multi-experiment fitting** with varying L_ref (rotation rate), c_O2_bulk, T, or pH. k0 is intrinsic; α is intrinsic; the diffusion-limited current scales differently. Joint fit breaks the ridge.
- **Multi-observable** — simultaneously fit cd(V) and the peroxide selectivity PC(V)/CD(V). PC depends on k0_2/k0_1 ratio differently than total CD does. v13 used this and got the best k0 recovery.
- **Wider voltage window** — extending z=1 convergence to higher V (where the second BV exponential term starts to matter and the curvature changes) would change the Hessian.
- **Mass-transport-coupled features** — the H+ depletion feedback (which makes the surface c_H+ shift with k0) introduces some k0 dependence into the transport-limited current. Currently weak signal because c_H+ is buffered by ClO4- electroneutrality.

---

## 6. What I'd love brainstorming on

**Primary question: what could make k0 (especially k0_2) identifiable in this system, given:**
- We can only get the standard PNP-BV solver to z = 1 in V_RHE ∈ [-0.50, +0.10] V.
- We can get a physics-correct solver (3sp + Boltzmann) to z = 1 across V_RHE ∈ [-0.30, +0.60] V.
- We can extend even further with log-c + seeded H2O2 (V_RHE up to ~+0.30 cleanly) but with constraints on what can be tested.
- 2% Gaussian noise on cd is the realistic data quality.
- Adjoint gradients work through the standard 4sp and the 3sp+Boltzmann concentration formulations; not yet verified through log-c.

**Specific sub-questions I'm stuck on:**

1. **Physics-side ideas:** Are there ORR-specific tricks people in electrochemistry use to break the (k0, α) ridge that we should be using? (Tafel intercept extrapolation, Koutecký–Levich, EIS-derived j_0 priors, Tafel slope from Allen–Hickling plot, etc.) Which of these can be turned into either an extra term in J or an extra residual in a multi-experiment fit?

2. **Multi-experiment fitting:** What's the minimum set of experimental perturbations (varying L_ref via rotation rate, varying c_O2 via gas pressure, varying T) needed to break the ridge cleanly? What does the joint Fisher information matrix look like?

3. **Better numerical formulations:** Beyond log-c, are there standard tricks for singularly-perturbed PNP that we're missing? (Discontinuous Galerkin with positivity-preserving fluxes? Scharfetter–Gummel? Mixed formulations? Asymptotic matched-expansion solvers that hard-code the Boltzmann EDL and only solve the outer problem?) The concrete failure is CG1 not preserving c_ClO4- ≥ 0 in a depletion zone.

4. **Adjoint gradients through log-c:** pyadjoint should tape `exp(u)` correctly, but we haven't verified. Are there gotchas? If we get adjoint working through log-c we can move from Nelder-Mead (30+ evals) to L-BFGS-B (~5 iter × 2 = 10 evals). 3× speedup unlocks much wider parameter studies.

5. **Tikhonov design:** Our λ = 0.01 was eyeballed. L-curve or discrepancy principle would let us pick it from the data. But more fundamentally — what's the right space to regularize in? log k0 is what we're using. Should we regularize the Tafel intercept j_0 = k0 · n_e F · c_bulk instead (more physical)? Should the regularizer be anisotropic (loose along the ridge direction, tight perpendicular)?

6. **Bayesian / MCMC alternative:** Profile likelihood says (k0, α) has a long thin ridge. A flat prior on α with a weakly-informative prior on log k0 (centered on literature with 10× width) would give a posterior whose maximum is well-defined and whose marginals quantify uncertainty honestly. Is this a more defensible approach for a publication than point estimation + Tikhonov?

7. **Surrogate hybrid:** v13's approach of "cheap surrogate global search → PDE refinement" got 4–5% k0 error in the cathodic region. Can we revive this with a surrogate trained over the working window of the new 3sp+Boltzmann+log-c forward model, then use Tikhonov refinement on the PDE? The surrogate provides the prior the Tikhonov term needs.

8. **Forward solver R&D:** The single biggest enabler would be getting z = 1 convergence for V > +0.30 V. The fundamental obstruction is the singular perturbation ε̂ ~ 1e-7 plus CG1 positivity violations on the depleting co-ion. We've tried many tactical fixes (artificial diffusion, log-c, seeded IC, voltage continuation, damping). Is there a structural change to the formulation — e.g., solve in terms of (φ, c_H+) only and reconstruct ClO4- from local Boltzmann? That's exactly what 3sp+Boltzmann does and it works. Can we generalize: solve a **reduced 2-equation system** (φ, c_O2 + c_H2O2 collapsed via reaction stoichiometry, with H+ either Boltzmann or transported)? What does the literature on EDL asymptotics suggest as the right small-ε model?

9. **Ill-posedness as a feature:** if k0 is genuinely non-identifiable from a single I–V curve at 2% noise, and we can prove this via Fisher information, then the publishable story might shift from "we recover k0" to "we quantify what is and isn't recoverable, and provide a Bayesian interval for k0." Does that reframing hold up given how these papers usually look?

---

## 7. Reference numbers and context

- **Solver runtime:** one 6-voltage I-V curve takes ~80–90 s. NM inference ~30–60 min.
- **Adjoint per-eval:** 26 s. Adjoint L-BFGS-B inference: 10–15 evals × ~30 s ≈ 5–8 min if everything works.
- **Discretization:** CG1 on graded mesh (finest h = 0.01 nm at electrode), 256 elements.
- **Noise:** 2% Gaussian on each I-V point (the realistic experimental level).
- **k0 prior:** literature values for ORR on Pt are within ~1 order of magnitude of true — so a Tikhonov prior with σ_log k0 ≈ 0.5–1 is realistic.
- **Repository:** Firedrake (FEniCS-fork) + pyadjoint for gradients. Inference scripts use scipy.optimize (Nelder-Mead, L-BFGS-B). No JAX, no autograd outside pyadjoint.

---

## 8. The current "best recipe" (state of art as of 2026-04-13)

```
forward = forms_logc      # log-concentration transform
species = [O2, H2O2, H+]  # 3 dynamic species
boltzmann = ClO4-         # background, in Poisson source
h2o2_ic_seed = 1e-4       # log-of-zero protection
v_grid = [-0.10, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30] V_RHE
optimizer = scipy.optimize.minimize(method="Nelder-Mead")
J = data_residual + 0.01 * (log k0 - log k0_prior)^2
```

Recovers k0_1, α_1 to <1% at 2% noise, single seed, with prior = truth. **Untested for k0_2, untested for wrong priors, untested for multiple noise seeds, no adjoint, no real experimental data.**
