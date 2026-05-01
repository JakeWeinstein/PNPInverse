# Definitive multi-experiment FIM result + proposal for next path

## Headline finding

**No multi-experiment design rotates the k0_2 weak direction by even one degree.** The complete FIM screen ran 6 designs spanning all combinations of (single-rotation, multi-rotation, H2O2-cofed) × (CD-only, CD+PC). Every single design ends with ridge_cos = 1.000 and a weak eigenvector aligned with pure log_k0_2.

## Complete result table

```
Design                       #rows   sv_min      cond(F)    ridge_cos   weak eigvec
A. CD only, L0                  5    4.08e-08    1.02e+16    1.000       k0_2-pure
B. CD+PC, L0                   10    2.35e-02    2.03e+11    1.000       k0_2-pure
C. CD+PC, L0/2L0/4L0           30    3.51e-02    2.72e+11    1.000       k0_2-pure
D. CD+PC, H2O2-cofed only       8    2.77e-07    2.71e+34    1.000       k0_2-pure
E. ORR L0 + H2O2-cofed L0      18    2.35e-02    2.03e+11    1.000       k0_2-pure
F. Full design                 38    3.51e-02    2.72e+11    1.000       k0_2-pure
```

(All experiments at TRUE k0; whitened by σ = 2% × max|target|; ridge_cos is the cosine similarity between weak eigvec of F and the canonical (k0_2, α_2) ridge direction predicted by the breakthrough analysis.)

The smoking gun for the mechanism is the parameter sensitivity invariance:

```
                  |dcd/dlog_k0_2|    |dpc/dlog_k0_2|
ORR L0            1.58e-08           2.78e-05
ORR 2L0           1.62e-08           2.78e-05
ORR 4L0           1.40e-08           2.78e-05
H2O2-cofed L0     1.83e-08           3.58e-05
```

`dpc/dlog_k0_2` is **identical to 3 sig figs across all ORR experiments** regardless of L_ref. Co-feeding H2O2 only bumps it 30%. No information-theoretically distinct sensitivity direction emerges from any of these experimental perturbations.

## Mechanism (which I should have argued more carefully in the original handoff)

R2 has E_eq,2 = 1.78 V vs RHE. The R2 overpotential at any solver-reachable voltage V_RHE ∈ [-0.10, +0.20] V is η_2 = V - 1.78 ∈ [-1.88, -1.58] V. The cathodic BV exponent `α_2 · n_e · η_2 / V_T = 0.5 · 2 · (-73 to -62) ≈ -73 to -62`, far past the clip `±50`. So `exp(-α_2 n_e η_2 / V_T)` is *frozen at exp(50)* at every voltage in every experiment we can simulate. Consequences:

- `∂r_2 / ∂α_2 = 0` exactly (clipped derivative). α_2 is invisible to first-order kinetics.
- `∂r_2 / ∂k0_2 = r_2 / k0_2` (linear). The *direction* of this sensitivity is the same in every experiment; only the *magnitude* (set by c_H2O2_surf) varies.
- Changing c_H2O2_bulk, c_O2_bulk, or L_ref shifts the magnitude of r_2 but not the direction in (k0_2, α_2) sensitivity space.

The k0_2 weak direction is therefore irreducible from any steady-state experiment in our solver-accessible voltage window. **The information loss is physical, not a discretization artifact.**

## What this means for your "Recommended Next Path" plan

Two of the three paths in your plan are now closed:

- ✗ **Multi-rotation (L_ref variation):** ruled out by Designs C and F. ω/ω_0 ratios from 1× to 16× did not lift the k0_2 sensitivity direction.
- ✗ **H2O2-fed at c_O2 ≈ 0:** the forward solver fails to converge in this regime (log-c degeneracy when ∂/∂u_O2 ~ exp(u_O2) → 0). The "co-fed" variant we could simulate (c_O2 = 1.0, c_H2O2 = 0.1) preserves the ORR ridge structure unchanged.

The third path you mentioned was Tikhonov / Bayesian prior. That's available and proven to work, but it doesn't recover k0_2 from data — it just selects a MAP point on the ridge given a prior. The publishable story under Tikhonov is: "α_1, α_2, k0_1 are recovered from data alone; k0_2 requires an independent prior."

## Proposal: target the anodic voltage frontier instead

The structural cause of our k0_2 problem is the BV exponent clip. The clip exists because we can't reach V_RHE high enough (≥ ~1.5 V) for R2 to operate in its kinetic regime. R2 unclips when |α_2 · n_e · η_2 / V_T| < 50, i.e., η_2 > -0.64 V, i.e., V_RHE > 1.14 V.

**If we could solve the forward problem at V_RHE ∈ [+0.5, +1.5] V, R2 would gradually unclip across that range, providing voltage-dependent shape information about α_2 directly. k0_2 would similarly become identifiable.** No multi-experiment, no prior, no transient observables — just the standard ORR I-V curve at higher voltages.

The blocker is forward-solver convergence. We've cataloged the failures (`docs/HANDOFF.md`, `StudyResults/v18_convergence_extension_log.md`):

- Standard 4sp: fails at V > +0.10 V (ClO4⁻ depletion → CG1 oscillates negative → SNES diverges)
- Stabilized 4sp (artificial diffusion on co-ion): converges to V = +0.73 V but **destroys onset physics** (cd profile flat, not the kinetic transition)
- 3sp + Boltzmann (concentration formulation): converges to V = +0.60 V but H2O2 oscillates to -0.69 (CG1 around near-zero state)
- Log-c + 3sp + Boltzmann (current breakthrough): converges to V = +0.30 V; fails at V ≥ +0.40 V due to H2O2 runaway
- Voltage continuation, Newton damping, trust region, Gummel splitting: all fail past V = +0.10 V

The fundamental obstruction is the singular Poisson perturbation (ε ≈ 1.8e-7) combined with **CG1 finite elements not preserving positivity in the Debye-layer depletion zone**. The depleting species — ClO4⁻ at high V (4sp) or H2O2 at high V (3sp) — goes negative locally, then BV exponentials applied to those negative values blow up Newton.

## Specific technical questions for you

1. **Positivity-preserving discretizations for PNP at high overpotential:** what's the standard tool kit? I know:
   - DG (discontinuous Galerkin) with slope limiters
   - Scharfetter-Gummel finite differences (semiconductor literature standard for high-bias drift-diffusion)
   - Mixed formulation (flux variable + concentration variable, possibly with positivity penalty on concentration)
   - Monotone schemes (M-matrix structure preserved at all overpotentials)
   - DG with `c = exp(u)` AND limiters on u (combine log-c with DG)
   
   For our system specifically (continuous Galerkin Firedrake, mesh ~ 200 graded cells with finest h ≈ 0.01 nm at electrode), which approach has the best chance of working at V_RHE = 1.0+ V without destroying onset physics? DG seems heaviest; Scharfetter-Gummel is FV not FE; mixed formulations can be assembled in Firedrake but the combination with adjoint isn't tested.

2. **Matched-asymptotic Boltzmann boundary layer:** the EDL is a singularly-perturbed boundary layer at the electrode. Could we solve a *reduced* problem where the EDL is replaced by an analytic Boltzmann profile and only the outer (electroneutral) problem is discretized? The Boltzmann factor `c_i = c_i_outer · exp(-z_i (φ - φ_outer))` could be applied as an effective BC. This would eliminate the singular perturbation entirely. We're already doing this for ClO4⁻ in the 3sp+Boltzmann setup — could it generalize to all charged species?

3. **Are there published PNP-BV solvers that have demonstrated convergence at η > 1 V on multi-electron-transfer reactions?** I'd love a literature pointer. Most ORR modeling I've seen stops well below E_eq,2 = 1.78 V and treats k0_2 as input from EIS rather than fitting it. If the consensus is "no published solver does this," that's important to know — it shifts the framing from "we should do this" to "we'd be doing something publishable in numerical methods, not just electrochemistry."

4. **Should we drop the BV exponent clip entirely?** The clip is currently `±50` on `α n_e η / V_T`. Removing it means `exp(73)` and bigger appears in the residual at high η. With double-precision floats we have ~700 of overhead before overflow, so `exp(73) ≈ 5e31` is fine numerically. The clip was added to prevent SNES from diverging when concentrations are tiny (because the rate `k0 · exp(huge) · c_tiny` may still be moderate but its derivative w.r.t. c_tiny is enormous). Maybe with a positivity-preserving discretization, we don't need the clip and can solve directly?

5. **Inverse-problem-side question:** if we *did* succeed at V_RHE = 1.0+ V and got direct R2 sensitivity, the data would be very strong — likely no longer ridge-limited at all. The TRF + adjoint Jacobian + IC cache pipeline we built should handle it cleanly. Is there value in *first* validating the full inverse pipeline by running at synthetic high-V data (i.e., having a hypothetical solver hand us cd/pc curves at V_RHE up to 1.5 V) before investing in the actual numerical work? This would de-risk the pipeline and let us estimate what accuracy improvement to expect from the new solver.

## What I'm asking you to choose between

A. **Implement Tikhonov on k0_2 + run honest prior-sensitivity tests** (1-2 days). Get a publishable result with the framing "α and k0_1 are data-driven; k0_2 is prior-assisted." Defensible electrochemistry paper, not novel numerics.

B. **Tackle the anodic-voltage frontier with a positivity-preserving discretization** (weeks to months). Risk: solver might still not converge at V_RHE = 1.5 V even with DG+limiters. Reward: if it works, k0_2 becomes genuinely data-identifiable and the contribution is publishable in numerical methods AND electrochemistry.

C. **Hybrid: pursue (A) immediately for an electrochemistry result, then (B) on a longer timeline for the methods paper.**

I lean toward (C). (A) is needed regardless because the k0_2 prior story is the honest scientific report of where we are. (B) is the genuinely interesting research path that the FIM analysis has now cleanly motivated — *we know exactly what physics is missing and why no clever experimental design can replace it*.

Your call on which to prioritize, plus thoughts on questions 1–5 above.

## Files for reference

- `scripts/studies/v18_logc_multiexperiment_fim.py` — the FIM screening script
- `StudyResults/v18_logc_multiexperiment_fim/{summary.json,experiments.json,run_v4.log}` — full FIM data
- `StudyResults/v18_logc_lsq_inverse/lsq_trf_noise_0.0pct_prectight_ny200/result.json` — tight-precision TRF result (k0_1 to -11% improvement)
- `StudyResults/v18_logc_sensitivity_fim/fim_results.json` — earlier FIM at TRUE for CD/PC/both at single L
- `docs/CHATGPT_HANDOFF.md` — original (math + history)
- `docs/CHATGPT_HANDOFF_2_PEROXIDE_RESULT.md` — initial peroxide negative result (now partially superseded)
- `docs/CHATGPT_HANDOFF_3_LSQ_RIDGE_RESULT.md` — TRF+adjoint result, refined ridge story
- `docs/CHATGPT_HANDOFF_4_PRECISION_AND_NEXT.md` — precision diagnostic (k0_1: -65% → -11% with tight tolerance)
- `docs/k0_inference_status.md` — breakthrough recipe
- `StudyResults/v18_convergence_extension_log.md` — detailed catalog of past anodic-voltage solver failures
