**Audience:** Applied math and computational science research group (peers).
**Tone:** Technical, candid, understated. No marketing language, no superlatives, no exclamation marks. Speak as a researcher explaining to peers.
**Design:** Minimalist, white background, clean sans-serif typography. No decorative elements. Data visualizations should be the visual focus.
**Target length:** ~15 minute research group meeting (8-9 slides).

---

## Slide-by-Slide Outline

1. **The Problem** — ORR kinetics inverse problem
2. **Why k0_2 Is Hard** — Anti-correlation, cancellation, signal burial
3. **The Surrogate-Then-PDE Idea** — Cheap exploration, expensive refinement
4. **Surrogate Selection** — NN ensemble chosen for noise robustness
5. **The Continuation Insight** — P1 selects the basin, P2 resolves k0_2
6. **Results** — All 4 parameters under 5.3% error
7. **Verification** — MMS convergence, gradient checks, inverse crime elimination
8. **Reflections: AI-Assisted Research** — What AI accelerated, what required human judgment
9. **(Optional) Open Questions** — Noise robustness as the frontier

---

## Slide 1: The Problem

**Title:** Recovering ORR Kinetic Parameters from I-V Curves

- Oxygen reduction at a rotating disk electrode: two coupled Butler-Volmer reactions
- Forward model: Poisson-Nernst-Planck PDE system solved at each applied voltage
- Two observables: disk current density (I_CD) and peroxide current (I_PC)
- Goal: recover 4 parameters (k0_1, k0_2, alpha_1, alpha_2) from measured I-V curves
- Each forward evaluation requires a full PDE solve across 10-15 voltage points

The inverse problem is PDE-constrained. We observe current-voltage curves from a rotating disk electrode experiment and want to infer the kinetic parameters governing two coupled oxygen reduction reactions. The forward model solves a 4-species Poisson-Nernst-Planck system with Butler-Volmer boundary conditions. Every evaluation of the objective function is expensive — a full PDE solve at each voltage in the sweep. We are searching for four parameters simultaneously: two rate constants and two transfer coefficients.

**Visual:** `plot1_iv_curve_fit.png` — Side-by-side I-V curves showing target vs fitted disk current and peroxide current.

---

## Slide 2: Why k0_2 Is Hard

**Title:** The k0 Trade-Off

- k0_2 is 24x smaller than k0_1 — its signal is buried in the dominant R1 contribution
- Peroxide current involves R1 minus R2: small errors in either rate get amplified in the difference
- k0_1 and k0_2 are anti-correlated — improving one historically degraded the other
- Every prior pipeline version showed this seesaw behavior
- Brute-force parameter search is infeasible: each I-V curve evaluation takes ~20 seconds

The core difficulty is that Reaction 2 is much slower than Reaction 1. The peroxide current, which is the observable most sensitive to k0_2, is computed as a difference of two nearly-equal reaction rates. This cancellation amplifies any error in k0_2. In 12 prior pipeline versions, improving k0_2 accuracy always came at the expense of k0_1, or vice versa. The anti-correlation between these parameters made simultaneous recovery of both a persistent open problem.

**Visual:** No dedicated plot. Slide text carries the message.

---

## Slide 3: The Surrogate-Then-PDE Idea

**Title:** Separating Exploration from Refinement

- Surrogate model (neural network ensemble) approximates I-V curves in milliseconds
- Scan 20,000 candidate parameter sets in under 10 seconds
- Surrogate finds the right neighborhood but hits a ceiling around 12% error on k0_2
- Switch to full PDE solver for final optimization, warm-started from surrogate's best guess
- Total pipeline: surrogate phases S1-S5 (~12s) then PDE phases P1-P2 (~460s)

The insight is that cheap and expensive models serve different purposes. The surrogate's job is warm-starting, not final accuracy. It scans broadly and cheaply, identifying the basin of attraction. The PDE solver then refines from that starting point with full physics. This two-stage approach means the expensive optimizer begins close to the answer, needing far fewer iterations than a cold start.

**Visual:** `plot2_pipeline_architecture.png` — Pipeline flow diagram showing S1-S5 feeding into P1-P2 with timing annotations.

---

## Slide 4: Surrogate Selection

**Title:** Choosing the Right Surrogate

- Tested 4 surrogate types: RBF, POD-RBF (log and nolog), NN ensemble
- Rankings invert between noise-free and noisy conditions (bias-variance tradeoff):

| Surrogate | 0% noise (max err) | 2% noise (mean max err) |
|-----------|-------------------|------------------------|
| RBF | 0.3% | 22.7% |
| POD-RBF-nolog | 3.2% | 27.2% |
| POD-RBF-log | 4.2% | 27.6% |
| NN Ensemble | 4.7% | 19.7% |

- NN ensemble's implicit smoothing acts as regularization under noise
- 20,000-point multistart confirms single global basin — surrogate ceiling is fundamental, not a search failure

Exact interpolation (RBF) excels without noise but overfits noise realizations. The neural network ensemble's smoothing makes it the most robust choice for PDE warm-starting under realistic conditions. The single-basin finding means we cannot escape the ~12% k0_2 ceiling by searching harder — we need a different model.

**Visual:** No plot. Table in slide text is the visual.

---

## Slide 5: The Continuation Insight

**Title:** Two-Stage PDE Refinement Breaks the k0 Trade-Off

- P1 (shallow cathodic, 10 voltage points): locks optimizer into the correct basin
- P2 (full cathodic, 15 voltage points): extends to deep overpotentials where R2 signal emerges
- k0_2 error drops from 12.01% (P1) to 4.23% (P2) — a threefold improvement
- Ablation: skipping P1 degrades max error from 5.3% to 17%
- P1 functions as a continuation method — solve the easier problem first to constrain the harder one

The key architectural decision in v13. At shallow overpotentials, R1 dominates and k0_2 is nearly invisible. P1 uses this regime to lock in k0_1 and the transfer coefficients without disturbing k0_2 too much. P2 then extends to deeper voltages where R2 becomes relatively significant, allowing the optimizer to resolve k0_2 without losing the gains from P1. This is what finally broke the seesaw.

**Visual:** `plot3_phase_progression.png` — Line chart showing parameter errors across S2, S3, S4, P1, P2 phases.

---

## Slide 6: Results

**Title:** v13: All Four Parameters Under 5.3% Error

- k0_1: 4.78% error
- k0_2: 4.23% error
- alpha_1: 2.95% error
- alpha_2: 5.26% error
- Max error: 5.26% (alpha_2) — first version where all parameters are simultaneously below 6%
- k0 error sum (k0_1 + k0_2 = 9.0%) is the lowest achieved across 13 pipeline versions
- Prior versions all showed the seesaw: v11 had k0_2 at 0.8% but k0_1 at 9.4%; v12 had k0_1 at 6.8% but k0_2 at 11.8%

The improvement came from the two-stage PDE architecture, not from a better surrogate. The surrogate is the same as v12; only the PDE refinement strategy changed.

**Visual:** `plot4_results_fit.png` — Target vs fitted I-V curves from P2, showing quality of fit on both observables.

---

## Slide 7: Verification

**Title:** Solver and Gradient Verification

- Method of Manufactured Solutions (MMS): all 4 species + potential achieve 2nd-order convergence
  - L2 rates: O2 = 2.01, H2O2 = 1.99, H+ = 2.01, ClO4- = 2.00, phi = 1.98
- Gradient verification via finite differences: adjoint gradients match FD to machine precision
- Inverse crime elimination: surrogate and PDE targets generated independently
- These checks are table stakes — without them, no inverse result is trustworthy

Verification is non-negotiable for PDE-constrained inverse problems. The MMS convergence study confirms the forward solver produces correct convergence rates across all species and the potential. Gradient accuracy ensures the optimizer sees a correct search direction.

**Visual:** `plot5_mms_convergence.png` — Log-log convergence plot showing 2nd-order rates for all fields.

---

## Slide 8: Reflections: AI-Assisted Research

**Title:** Developing with Claude Code

- Pipeline developed with Claude Code (AI coding assistant) over ~3 weeks
- AI accelerated: rapid prototyping of pipeline phases, parameter studies, infrastructure (CLI, caching, plotting)
- Human judgment required: physics correctness (nondimensionalization, weak forms), architecture decisions (the two-stage PDE idea), interpreting whether numerical results are meaningful
- Verification imperative: AI code generation speed makes rigorous verification even more critical
- Too easy to produce plausible but incorrect numerical results without systematic checks

The speed of AI-assisted development is real, but it shifts the bottleneck from writing code to verifying code. The MMS study, gradient checks, and inverse crime elimination were all essential — not optional extras.

**Visual:** No plot. Slide text carries the message.

---

## Slide 9 (Optional): Open Questions

**Title:** What Remains

- Noise robustness is the frontier: at 2% noise, performance becomes seed-dependent
- Noise-free max error of 5.3% amplifies to ~27% median under noise
- Potential strategies: Tikhonov regularization on PDE objective, restricted voltage ranges, multi-seed selection
- Open question: can the continuation strategy be extended to handle noise more gracefully?
- This is still early-stage work with clear next steps

The gap between best-case (5.3%) and noisy performance (~27%) defines the open research direction. The pipeline architecture is sound; the challenge is making it robust to realistic measurement uncertainty.

**Visual:** No plot. Brief closing slide.

---

**SAFETY RULES:** Use ONLY the information provided in this document. Do not introduce external examples, statistics, or results not present in the source material. All numerical results are from the v13 pipeline (March 2026). Accompanying plots are in `writeups/WeekOfMar4/presentation_plots/`.
