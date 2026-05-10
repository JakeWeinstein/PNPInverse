# Follow-up to ChatGPT: Peroxide-as-second-observable experiment — negative result

## Context

In the prior handoff (`CHATGPT_HANDOFF.md`) you correctly pointed out that the 3-species + Boltzmann ClO4⁻ background solver still has both R1 and R2 reactions and therefore can compute peroxide current from `r1 - r2` directly. You hypothesized that adding peroxide current as a second observable to the joint inverse objective would break the (k0, α) ridge degeneracy that has been blocking k0 identifiability — because disk current sees `r1 + r2` while peroxide sees `r1 - r2`, providing independent constraints on the partitioning between R1 and R2.

We implemented this rigorously and ran it. **The hypothesis did not hold.** Below is everything you need to think about why and what to try next.

## Implementation (what was actually run)

Forward physics: the breakthrough recipe from `k0_inference_status.md` exactly as documented:
- `forms_logc.py` (log-concentration transform `u = ln c`)
- 3 dynamic species (O2, H2O2, H+)
- Boltzmann ClO4⁻ background in Poisson source (`-ε∇²φ` includes `+c_ClO4_bulk · exp(φ)` term)
- H2O2 IC seeded at u = ln(1e-4) = -9.2
- Working voltage window V_RHE ∈ {-0.10, 0.00, +0.10, +0.15, +0.20} V (5 points; dropped V=+0.25 because at +20% offset on all 4 params it falls past the convergence cliff)

Observables built from `bv_rate_exprs[0] - bv_rate_exprs[1]` exactly as you recommended (the existing `_build_bv_observable_form(..., mode="peroxide_current", ...)` already does this — confirmed in code, this was infrastructure already present).

Inverse:
- 4 free parameters: log(k0_1), log(k0_2), α_1, α_2
- Initial guess: +20% offset on all four (assumed surrogate-model warm-start accuracy)
- Joint objective with **range-normalized** residuals (1/cd_scale² and 1/pc_scale²) — we tried inverse-variance first but σ_pc ≈ 3e-7 made the gradient explode by 10⁸×, breaking L-BFGS-B
- **No Tikhonov regularization** (this was the controlled test of the structural argument)

Optimizer:
- L-BFGS-B with adjoint gradients via pyadjoint
- Adjoint verified to <0.1% against finite differences on 3 of 4 components (k0_2 component matched within FD's double-precision noise floor — adjoint is correct)
- Per-voltage tape with summed gradients across V_GRID (linear since J decomposes by voltage)
- Per-voltage IC cache that persists across optimizer evals
- Cascading fallback: cached U → adjacent voltage's U → TRUE-cache U → cold ramp
- 5 annotated SNES steps after warm-start convergence (matches `v18_adjoint_simple.py` pattern that worked previously)

## Result

**Clean data run (no noise; J = 0 at TRUE by construction):**

| Param | Init  | Recovered | Δ from init |
|-------|------:|----------:|------------:|
| k0_1  | +20%  | +20.52%   | +0.52%      |
| k0_2  | +20%  | +18.25%   | -1.75%      |
| α_1   | +20%  |  -5.95%   | **-25.95%** |
| α_2   | +20%  |  -4.68%   | **-24.68%** |

J went from 0.316 → 0.012 (26× reduction over 48 evals, 3.3 min wall-clock with IC cache).

**Pattern is unambiguous:** both k0's stayed within 2% of their +20% init while both α's moved 25% — overshooting TRUE by ~5% in the same direction. This is the (k0, α) ridge degeneracy expressed mechanistically: the optimizer descends J by sliding α downward to compensate for inflated k0, finding a local minimum *on the ridge* rather than at TRUE.

That this happens at clean data — where the global minimum J = 0 sits exactly at TRUE — confirms the optimizer found a genuine stationary point. Per the breakthrough script's analysis, the ridge has slope d(log k0)/dα ≈ -47, so when ∂J/∂α = 0 at the ridge bottom, ∂J/∂log k0 = 0 there too. The ridge is in (log k0_1, α_1, log k0_2, α_2) space; both reactions exhibit it.

**Noisy run (2% Gaussian, single seed) for comparison:**

| Param | Init  | Recovered (noisy) | Recovered (clean) |
|-------|------:|------------------:|------------------:|
| k0_1  | +20%  | +20.87%           | +20.52%           |
| k0_2  | +20%  | +18.97%           | +18.25%           |
| α_1   | +20%  | +19.05%           |  -5.95%           |
| α_2   | +20%  | +10.52%           |  -4.68%           |

Noise masks the α gradient signal further (only α_2 made meaningful progress under noise). Clean data shows the full extent of α movement that the joint observable enables — but k0 still doesn't move in either case.

## Mechanistic interpretation of why peroxide didn't help

R2 has E_eq,2 = 1.78 V vs RHE. At every voltage in our working window V_RHE ∈ [-0.10, +0.20] V, the R2 overpotential η_2 = V - E_eq,2 ∈ [-1.88, -1.58] V — extremely cathodic. With α_2 = 0.5, n_e = 2, V_T = 25.69 mV, the R2 BV exponent is α_2·n_e·η_2/V_T ∈ [-73, -62]. The exponent clip at ±50 saturates this to exp(50) for *every voltage* in the window.

So R2's rate at the electrode is `r_2 = k0_2 · exp(50) · (c_H+/c_ref)² · c_H2O2_surf`. The exponential saturation means **r_2 depends only on c_H2O2_surf and k0_2 multiplicatively at every voltage in our window** — there is no voltage-dependent BV "shape" information to disentangle k0_2 from α_2 (or to disentangle R1 contributions from R2 contributions).

c_H2O2_surf in turn is set by R1 production rate (which is much larger and hits the surface H2O2) minus R2 consumption (small) minus diffusive escape. So peroxide current `i_pc = -2F(r_1 - r_2)` ≈ `-2F·r_1` to leading order in this window — peroxide is essentially a slightly noisier copy of disk current, not an independent observable.

The premise that "peroxide carries `r_1 - r_2` info that disk doesn't" only holds when both reactions are operating in their kinetic (non-saturated) regimes. In our current window, R2 is saturated everywhere. Disk current sees k0_1 mainly (since r_2 << r_1 in absolute terms), and peroxide also sees k0_1 mainly (same reason). The two observables are nearly redundant.

**This explains the result quantitatively:** peroxide adds essentially zero new information about k0_2, so the inverse remains underdetermined exactly where v17 originally found it.

## What this means

The peroxide-as-second-observable strategy *would* work if we could move R2 out of its saturated regime — i.e., access voltages closer to E_eq,2 = 1.78 V. But the solver doesn't converge anywhere near that anodic regime (the convergence cliff is around V_RHE ≈ +0.30 V even in log-c+Boltzmann). So under the current forward model, peroxide can't break the ridge regardless of objective weighting or optimizer choice.

Three viable paths forward, ranked by effort:

1. **Tikhonov + peroxide combined** (low effort, ~1 hour). The breakthrough script (`v18_logc_regularized.py`) already shows Tikhonov on log(k0) recovers k0_1 to <1% in 2D. Adding peroxide as a second observable to the *same* objective might tighten that result and/or extend it cleanly to 4 params. The ridge isn't broken by peroxide alone, but combined with a weak prior, peroxide + Tikhonov could give better recovery than Tikhonov alone.

2. **Multi-experiment fitting** (medium effort, ~1 week). Vary L_ref (rotation rate) or c_O2_bulk between experiments. The diffusion-limited current scales as D · c_bulk / L_ref while the BV-controlled regime depends on k0 · c_surf — they shift differently with the experimental knob. Joint fitting across 2-3 experiments breaks the ridge from a *different* physical principle than peroxide. This is standard practice in electrochemistry (Koutecký-Levich analysis is essentially this).

3. **Push the solver past V=+0.30 V** (high effort, possibly impossible). The fundamental obstruction is the singular Poisson perturbation ε ~ 1e-7 combined with CG1's lack of positivity preservation in the depletion zone. Many tactical fixes have been tried (artificial diffusion, log-c, seeded IC, voltage continuation, damping) — see `docs/HANDOFF.md` for the catalog. The structural fix would be moving to a positivity-preserving discretization (DG with limiters, or a Scharfetter-Gummel scheme), which is research-grade work.

## Specific questions for you

1. **Is the saturated-R2 mechanistic explanation above correct?** Did your original argument assume R2 was operating non-saturated, or did you account for the BV exponent clip? If you assumed non-saturated, the argument needs updating for the actual operating regime.

2. **Would extending the voltage window to access *less* anodic R2 conditions help?** R2's E_eq is so high (1.78 V) that we'd need V_RHE near or above 1.78 V to get η_2 in the kinetic regime — but we can't push the forward solver past +0.30 V in any tested formulation. Is there a clever observable construction that exposes R2 kinetics from data taken in the saturated regime?

3. **For multi-experiment fitting (option 2 above), what's the minimum set of experimental perturbations** (L_ref, c_O2, T, pH) that would break the ridge cleanly? The Fisher information argument would help — if you can sketch what the joint Hessian looks like for, say, 2 L_ref values vs 3, that would help us decide how many experiments to plan.

4. **Is there a dimensionally-different observable we're missing?** E.g., DC vs AC, time-resolved transient response vs steady-state, EIS spectrum at specific frequencies. We're currently only fitting steady-state I-V; expanding to transient or frequency-domain might add k0 information that disentangles from α.

## Numerical artifacts worth noting (not the main story)

A few details from the run that don't affect the conclusion but might affect your model of what's happening:

- The IC cache + warm-start design works: per-voltage solves at warm-start cost ~1-3 s vs ~80 s cold ramp. Total per-eval cost (5 voltages forward + 5 voltages adjoint) is ~5-12 s, making 48 L-BFGS-B evals fit in 3.3 min instead of multi-hour.
- Adjoint through `forms_logc` + Boltzmann monkey-patch verified correct against FD to ~0.1% (k0_2 component disagreed at "5%" but that's pure FD double-precision noise — adjoint dJ/dk0_2 = 1.011e-3, FD = 1.065e-3, expected FD step magnitude is at machine epsilon).
- Cold-start at +20% offset is unreliable in log-c (failed at V=-0.10 V even on the easiest cathodic point). The fix is to seed the IC cache from the TRUE-params solution as a one-time anchor — every subsequent eval warm-starts from there.
- L-BFGS-B with `bounds=` and `jac=callable` works fine once the gradient scale is reasonable. Inverse-variance weighting (1/σ²) was 10⁸× too steep; range-normalized weighting (1/scale²) gave |g| ~ 6 at init which L-BFGS-B handled cleanly.

## Files for reference

- `scripts/studies/v18_logc_joint_observable.py` — the inversion script
- `scripts/studies/v18_logc_adjoint_check.py` — FD-vs-adjoint verification
- `StudyResults/v18_logc_joint_observable/noise_0.0pct/result.json` — full per-eval history clean data
- `StudyResults/v18_logc_joint_observable/noise_2.0pct/result.json` — same with 2% noise
- `StudyResults/v18_logc_joint_observable/noise_0.0pct_run.log` — clean run log with all 48 evals
- `docs/CHATGPT_HANDOFF.md` — original handoff (math + history)
- `docs/k0_inference_status.md` — breakthrough recipe (Tikhonov+regularized 2D case that worked)
