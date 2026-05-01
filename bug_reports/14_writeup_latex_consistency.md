# Bug Report: Writeup/LaTeX Consistency

**Focus:** Documentation accuracy, numbers matching data, equation-code consistency
**Agent:** Writeup/LaTeX Consistency

---

## ISSUE 1: Surrogate fidelity test set size mismatch
**File:** `writeups/vv_report/vv_report.tex:223, 240`
**Severity:** HIGH
**Description:** Text states "479 Latin Hypercube-sampled parameter sets" but JSON source (`StudyResults/surrogate_fidelity/fidelity_summary.json`) reports `"n_test": 524`.
**Suggested fix:** Update both instances of "479" to "524".

## ISSUE 2: Butler-Volmer equation in writeup doesn't match code
**File:** `writeups/vv_report/vv_report.tex:96-101`
**Severity:** HIGH
**Description:** Writeup presents standard symmetric BV form `j = j_0 [exp(alpha_a F eta / RT) - exp(-alpha_c F eta / RT)]`. Code uses asymmetric cathodic-dominant formulation with:
1. Concentration-dependent cathodic kinetics (not exchange-current-density form)
2. Single transfer coefficient alpha with `(1-alpha)` for anodic branch
3. Optional cathodic concentration factors (proton dependence)
4. Nondimensionalized overpotential with scaling factor
**Suggested fix:** Rewrite Eq. (3) to match actual implementation.

## ISSUE 3: Pipeline stage count mismatch
**File:** `writeups/vv_report/vv_report.tex:114-115`
**Severity:** MEDIUM
**Description:** Text says "six stages" but TikZ figure defines eight nodes.
**Suggested fix:** Change "six stages" to "eight stages".

## ISSUE 4: "Seven-phase pipeline" undefined
**File:** `writeups/vv_report/vv_report.tex:356`
**Severity:** MEDIUM
**Description:** References "full seven-phase pipeline" but this is not defined or reconciled with the architecture section.

## ISSUE 5: GP model name inconsistency
**File:** `writeups/vv_report/tables/surrogate_fidelity.tex:9`
**Severity:** MEDIUM
**Description:** Table labels GP as "GP (GPyTorch)" but JSON key is "gp_fixed". If trained with fixed hyperparameters, name may be misleading.

## ISSUE 6: Gradient consistency text oversells results
**File:** `writeups/vv_report/vv_report.tex:305`
**Severity:** MEDIUM
**Description:** Text says "analytic and FD gradients agree to machine precision" but doesn't mention that FD convergence rates for `log10_k0_1` (-0.134) and `log10_k0_2` (0.146) are essentially zero. The gradient landscape is extremely flat in those dimensions.
**Suggested fix:** Note that log10_k0 parameters have near-zero gradients making FD rate estimation unreliable.

## ISSUE 7: Nernst-Planck equation missing steric term present in code
**File:** `writeups/vv_report/vv_report.tex:73-78`
**Severity:** MEDIUM
**Description:** Eq. (1) shows standard drift-diffusion without the Bikerman steric chemical potential term. Code includes optional steric term: `D[i] * (grad(c_i) + c_i * grad(drift) + c_i * grad(mu_steric))`.
**Suggested fix:** If steric effects are used in production, add term to equation. If disabled, add footnote.

## ISSUE 8: Report only discusses +10% evaluation point, not -10%
**File:** `writeups/vv_report/vv_report.tex:306-309`
**Severity:** LOW
**Description:** JSON contains results for "true", "+10%", and "-10%" points. The -10% point shows poor FD convergence rates but is not discussed.

## ISSUE 9: Unused figure file
**File:** `writeups/vv_report/figures/surrogate_comparison.pdf`
**Severity:** LOW
**Description:** File exists but never referenced in `vv_report.tex`.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 2     |
| MEDIUM   | 5     |
| LOW      | 2     |

**Top priority:** Issue 1 (wrong test set size) and Issue 2 (BV equation mismatch) are factual errors that should be corrected before publication.
