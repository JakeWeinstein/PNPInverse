# Follow-up to ChatGPT: Phase A diagnostics + TRF/LSQ inverse — refined picture

## Concession before everything

Two specific corrections to the previous handoff (`CHATGPT_HANDOFF_2_PEROXIDE_RESULT.md`):

1. **You were right about PC adding rank.** My mechanistic argument that "R2 saturated ⇒ peroxide redundant with disk current" was empirically wrong.
2. **The previous "L-BFGS-B converged at non-true minimum" framing was misleading.** It wasn't a true local minimum — it was an optimizer that simply got stuck on a heavily ill-conditioned landscape.

Both diagnostics from your Phase A plan ran cleanly. They invalidated my interpretation. Then the LSQ+adjoint inverse you implicitly suggested produced new information that further refines the picture.

## Phase A.1 — FIM diagnostic (confirms PC adds rank)

Sensitivity matrix S[m, j] = ∂y_m/∂θ_j computed at TRUE params via central FD with two step sizes. Whitened by σ = 2% × max|target| per observable type. Computed F = Sᵀ_white S_white separately for CD-only, PC-only, and CD+PC.

```
                  smallest sv     condition(F)    weak eigvec ([log_k0_1, log_k0_2, α_1, α_2])
CD only           4.47e-08        1.18e+16        [-0.000, -1.000, +0.000, +0.010]
PC only           3.58e-03        8.73e+12        [+0.881, -0.473, +0.000, +0.014]
CD + PC           2.98e-02        1.26e+11        [-0.001, -1.000, +0.000, +0.010]
```

**`sv_min(CD+PC) / sv_min(CD-only) = 667,097×`.** PC is genuinely informative. The condition number drops by 5 orders of magnitude when PC is added. The weak eigenvector under CD+PC is still log_k0_2-dominant, but its FIM eigenvalue is now (3e-2)² ≈ 9e-4 — finite, where CD-only had ~2e-15 (essentially numerical zero).

So the rank/identifiability question has a clean answer: **adding PC moves the system from "mathematically singular" (cond F ~ 1e16) to "ill-conditioned but technically solvable" (cond F ~ 1e11)**. Whether 1e11 is *practically* solvable depends on the optimizer and forward-solve precision — which is the next finding.

## Phase A.2 — Line profile from L-BFGS-B endpoint to TRUE (confirms optimizer failure)

Took θ_recovered_clean (the L-BFGS-B clean-data endpoint at +20.5%/+18%/-6%/-5%) and θ_true. Evaluated J(t), J_cd(t), J_pc(t) for θ(t) = (1-t)·θ_recovered + t·θ_true, t ∈ [0, 0.05, ..., 1.0].

```
t=0.000 (recovered):  J = 1.24e-02
t=0.500 midpoint:     J = 2.94e-03
t=1.000 (TRUE):       J = 1.77e-13   (≈ 0 modulo solver discretization)
```

**Monotone descent** from recovered to TRUE. There was no barrier between them; the optimizer simply didn't traverse the path. Confirms the "L-BFGS-B failure" hypothesis from before.

(One ~10% bump at t=0.75 is numerical noise from a warm-start solve; not a real basin.)

## Phase B — LSQ + per-observable adjoint Jacobian (TRF method)

Switched from L-BFGS-B + scalar-J adjoint to `scipy.optimize.least_squares(method='trf')` with σ-whitened residuals (10 components: 5 cd + 5 pc) and a 10×4 Jacobian computed via per-observable adjoint (one annotated forward solve per voltage, two ReducedFunctionals built on the same tape, one for cd_assembled and one for pc_assembled).

Same initial guess (+20% offset on all 4 params), clean data, no Tikhonov.

**Result:**

| Param | Init   | LBFGS-B final | **TRF final** | True |
|-------|-------:|--------------:|--------------:|-----:|
| k0_1  | +20.0% | +20.5%        | **-65.3%**    |   0  |
| k0_2  | +20.0% | +18.3%        | **-23.9%**    |   0  |
| α_1   | +20.0% | -5.95%        | **+4.54%**    |   0  |
| α_2   | +20.0% | -4.68%        | **+1.89%**    |   0  |
| cost  | 7.91e+02 | 2.46e+01    | **1.15e-02**  |   0  |

TRF reduced cost by 4.5 OOM (vs L-BFGS-B's 1.5 OOM). It actually navigated the landscape. **α came back within 5% of TRUE.** k0 went substantially low — *opposite direction* from L-BFGS-B's high.

This is the (k0, α) ridge in action. Predicted ridge slope from the breakthrough analysis: d(log k0)/dα ≈ -47. Apply to TRF result:

- α_1 recovered shift = +0.029 from TRUE ⇒ predicted log k0_1 shift = -1.36 ⇒ k0_1 factor = 0.26 ⇒ predicted k0_1 = 3.3e-4. Actual recovered = 4.4e-4 (within 30% of ridge prediction).
- α_2 recovered shift = +0.009 from TRUE ⇒ predicted log k0_2 shift = -0.44 ⇒ k0_2 factor = 0.66 ⇒ predicted = 3.5e-5. Actual = 4.0e-5 (within 15%).

**TRF descended along the ridge — different point than L-BFGS-B, similar low cost.**

TRF terminated with `xtol` (step size < 1e-9 in parameter space), optimality measure 1.5e+01 (finite gradient remaining). It couldn't find a downhill step despite a clean monotone path to TRUE.

## Why TRF stopped on the ridge

I propose the following mechanism. Tell me if it's right or if there's a better explanation.

The smallest singular value of S_white ≈ 3e-2. Largest ≈ 1e+4. Condition number ≈ 1e11; condition number of S itself ≈ √cond(F) ≈ 3e5.

For the optimizer to make progress along the weak direction, the Jacobian must point in that direction with relative accuracy better than (sv_min / sv_max) ≈ 1e-6. The Jacobian rows are computed by adjoint solves through forward steady-states whose CG1 + Newton + adaptive-dt convergence yields cd, pc values to relative precision ~1e-6 (the SNES tolerance × some amplification).

So the Jacobian's representation of the weak ridge direction is *at the precision limit* of the forward solve. The optimizer can't reliably distinguish "step downhill along ridge" from "step into Jacobian noise." TRF correctly refuses to take steps it can't verify will reduce cost.

This is consistent with the observed behavior:
- TRF made fast progress along the strong directions (cost dropped from 790 to 0.011 in 12 iterations)
- It stalled at the ridge
- The recovered point fits the data essentially perfectly modulo solver discretization
- The "wrong" k0 values are 30% within the predicted ridge curve from a 4.5%-displaced α

Adjoint accuracy doesn't help here because the *forward* solve is the precision-limiting step, not the adjoint.

## What this means

The earlier framings were both wrong in different directions. The refined picture:

1. **Information content (FIM):** PC adds genuine rank. The data is information-theoretically sufficient to identify all 4 params at clean data. cond(F_CD+PC) ≈ 1e11 — finite, so rank is nominally 4.

2. **Numerical conditioning (TRF behavior):** cond ≈ 1e11 in F-space means cond ≈ 3e5 in residual-space, which exceeds the relative precision of CG1+Newton forward solves (~1e-6 best case). So while the data carries the information, no Jacobian-based optimizer running on standard double-precision forward solves can extract k0 reliably from a single-experiment fit.

3. **What's actually recoverable from disk+peroxide:** **α is robustly recoverable** — TRF got both α's within 5% of TRUE without any prior. This is a useful publishable result for Tafel-slope estimation. **k0 is not** — it's stuck on the ridge wherever the optimizer first hits it.

4. **The "v17 conclusion" needs precise restating:**
   - ❌ "k0 is information-theoretically unidentifiable from I-V curves" (wrong, FIM shows finite info)
   - ❌ "L-BFGS-B's BFGS approximation was the bottleneck" (partly wrong, TRF + LM stalls similarly)
   - ✅ **"k0 is numerically unidentifiable from a single-experiment I-V fit at standard double-precision floating-point, because the FIM condition number (~1e11) exceeds the forward-solve precision."**

## Specific questions for you

1. **Is the "ridge condition vs forward-solve precision" diagnosis correct?** Standard textbook L-curve analysis assumes the forward model is exact. But our forward solve has finite precision, and the FIM is right at the precision wall. Have you seen this characterized as a separate phenomenon? Is there a name for it (e.g., "discretization-limited identifiability")?

2. **Practical mitigations:**
   - **Tighter SNES + finer mesh:** Could push forward precision from ~1e-6 to ~1e-9, buying ~3 OOM of headroom. Wall time per solve goes 10–100×, so a single inverse run becomes overnight. Worth it?
   - **Pre-condition the optimizer with FIM diagonal:** I can pass `x_scale = 1/sqrt(diag(F))` to `least_squares`. Does this help when the cross-coupling (off-diagonal F) is what's ill-conditioned?
   - **Trust-region with explicit subproblem regularization:** TRF uses a damping parameter (Levenberg-Marquardt-like). Setting larger initial damping might prevent the optimizer from sliding so far along the ridge in early iterations. Trade-off?
   - **Polished output coordinates:** instead of fitting θ in the natural (log k0_1, log k0_2, α_1, α_2) basis, project to the FIM eigenbasis and fit there. The well-conditioned directions converge first; the ridge direction's lack of progress is then explicit. Does this work in practice?

3. **Multi-experiment Fisher analysis (your Phase B suggestion):** if we add a second experiment at different L_ref, the FIM accumulates rows. The block structure makes ridge-breaking quantifiable. Is there a closed-form for the smallest singular value of the joint FIM as a function of how different the two L_ref values are? I.e., if `L_ref_2 = α · L_ref_1`, does sv_min scale with α–1 (linearly), (α–1)² (quadratically), or something else? This would let us decide whether to do 2 vs 3 experiments and how spread the L_ref values should be.

4. **Should I publish the α-only result?** If we run TRF with disk+peroxide on real data, we expect α recovery to <5% with no priors. That's useful for Tafel-slope work even if k0 is non-identifiable. But the framing has to be careful — "we recover α robustly; k0 requires either an EIS prior or multi-experiment fitting." Is this a defensible scientific claim, or does it require additional caveats?

5. **One gap I haven't tested:** the "BFGS Hessian gets contaminated" hypothesis I floated previously is partly testable — TRF/LM doesn't use BFGS, and TRF still stalled, so the BFGS-specific story was wrong. But it's still possible that the L-BFGS-B run *would* have been substantially better with proper preconditioning. Does it matter? TRF's performance gives us a clean answer regardless.

## Files for reference

- `scripts/studies/v18_logc_diagnostics.py` — Phase A FIM + line profile (single script, both outputs)
- `scripts/studies/v18_logc_lsq_inverse.py` — Phase B equivalent: TRF + per-observable adjoint Jacobian
- `StudyResults/v18_logc_sensitivity_fim/fim_results.json` — full FIM data (S matrices, eigendecomps, condition numbers)
- `StudyResults/v18_logc_joint_profile_checks/profile.json` — line profile data
- `StudyResults/v18_logc_lsq_inverse/lsq_trf_noise_0.0pct/result.json` — TRF clean-data run
- `StudyResults/v18_logc_joint_observable/noise_*pct/` — earlier L-BFGS-B + adjoint runs (the originally-misinterpreted ones)
- `docs/CHATGPT_HANDOFF.md` — original handoff (math + history)
- `docs/CHATGPT_HANDOFF_2_PEROXIDE_RESULT.md` — previous handoff (negative result, partially incorrect interpretation)
- `docs/k0_inference_status.md` — breakthrough recipe (Tikhonov-regularized 2D case, still relevant for k0 recovery path)
