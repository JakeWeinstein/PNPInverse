# Round 1 — Adversarial Review of Fast Realignment Plan

Date: 2026-05-08
Reviewer: GPT (via Codex CLI, default xhigh)
Reviewee: Claude
Scope: Pre-implementation plan for a fast/aggressive realignment of the
        PNP-BV forward solver to the actual Seitz/Mangan experimental
        setup.

## Section 1 — Context bundle

### 1.1 What this plan is

The user said, verbatim:

> okay, I want to say fuck it. this realignment plan is too slow. read
> the realignment plan and read the docs that show what the real
> experimental setup is. Make a new plan to adjust everything to the
> real experimental setup then run the gpt-critique-loop on it until
> approved. This plan should be fast, no regression testing, no
> extensive unit tests, I just want everything implemented and I'll
> fix it later. Be sure the beginning of it adds some git checkpointing.

So the deliverable is a *fast* plan (5-10 days target) that
deliberately defers regression-vs-Run-C, comprehensive unit tests,
M3a substaging, M3b multi-stage breakdown, M4 calibration, M5 L_eff
retune, and M6 Stern+cation joint sensitivity. Acceptance: the user
gets *one* end-to-end converged page-15 sweep with the structurally
correct experimental setup, and fixes quantitative gaps after the
fact.

### 1.2 What "real experimental setup" means

Source authority hierarchy (per the artifact under review):

1. **Ruggiero 2022 J. Catal.** — peer-reviewed, Mangan co-author, the
   source paper for the deck. (`docs/Ruggiero2022_JCatal_source_paper.md`,
   PDF at `docs/Ruggiero2022_JCatal_manuscript.pdf`.)
2. **Seitz/Mangan data folder audit** — multi-document confirmation
   2019→2025. (`docs/seitz_mangan_data_folder_audit_2026-05-08.md`.)
3. **Linsey 2025 ACS-CATL deck slide 13** — cation pKa & hydrated radii.

**Hard mismatches between current PNPInverse production stack and the
real experiment, exhaustively (as of 2026-05-08):**

| Current production (CLAUDE.md "production stack") | Real experimental setup | Severity |
|---|---|---|
| Analytic ClO₄⁻ Bikerman counterion (z=-1, c_bulk=0.1 mol/m³) | Sulfate (SO₄²⁻, z=-2, c_bulk=100 mol/m³) + alkali cation (Cs⁺ for page 15, z=+1, c_bulk=199.9 mol/m³ by electroneutrality) at I=0.3 M; λ_D≈0.55 nm | HIGH (structural) |
| Sequential R_0 (O₂→H₂O₂ at E°=0.68 V) + R_1 (H₂O₂→2H₂O at E°=1.78 V) | Parallel R_2e (O₂→H₂O₂ at E°=0.695 V) + R_4e (O₂→2H₂O at E°=1.23 V); free H₂O₂ never forms in the 4e channel | HIGH (structural) |
| Hardcoded 2×2 sequential Picard for IC | No parallel-topology Picard exists; topology gate already added (rejects parallel) and orchestrator falls through to `linear_phi`, which empirically cold-fails at every V tested | HIGH (blocks convergence) |
| C_O2 = 1.2 mol/m³ (post-M3a.2.1 fix) | 1.2 mol/m³ at pH 5-13 per Ruggiero §2.4 ✓ | OK (already aligned) |
| Disk area, Levich constants, V vs RHE conversion | All match Ruggiero §2 ✓ | OK |
| Single-cation hard-coded (currently ClO₄⁻; was 4sp dynamic ClO₄⁻ before that) | Cation should be configurable Li/Na/K/Cs | MEDIUM (deferred to post-fast-realignment) |
| HSO₄⁻ ↔ H⁺ + SO₄²⁻ acid-base equilibrium | At pH 4 with pKa₂(SO₄)=1.99, sulfate is ~99% SO₄²⁻ | LOW (negligible at pH 4; deferred) |
| Constant ε_r = 80 throughout | Trienens 2023 Aim 3 flags ε≈6 near surface; field-dependent | MEDIUM (deferred) |
| Symmetric Bikerman (single `a_i`) | Cation hydrated radii: Li⁺ 3.4 / Na⁺ 2.8 / K⁺ 2.3 / Cs⁺ 2.2 Å (Linsey 2025 deck slide 13); asymmetric | MEDIUM (deferred to M4-M6) |

The *fast* plan addresses the three HIGH items only.

### 1.3 Production solver call site (current, working for ClO₄⁻ +
sequential)

From `CLAUDE.md`:

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,   # steric_mode='bikerman'
)
from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback

sp = make_bv_solver_params(
    ...,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh",
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,
    initializer="debye_boltzmann",
)
```

Reaches V_RHE = +1.0 V at 15/15 cold/warm convergence on the C+D
orchestrator. **This is the current production baseline; the fast
plan replaces all three structural pieces.**

### 1.4 What's already landed (M3a.0/M3a.1/M3a.2/M3a.2.1)

- ✅ `scripts/_bv_common.py` has `PARALLEL_2E_4E_REACTIONS`,
  `K0_HAT_R{2E,4E}`, `E_EQ_R{2E,4E}_V`, `ALPHA_R{2E,4E}`. Wired
  through `make_bv_solver_params(bv_reactions=...)` and
  `_make_bv_bc_cfg`.
- ✅ `Forward/bv_solver/observables.py` has electron-weighted
  `current_density` and new `gross_h2o2_current` mode.
- ✅ Topology gate at `forms_logc{,_muh}.py` rejects parallel from
  the matched-asymptotic Picard with `non_sequential_topology` reason.
- ✅ C_O2 = 1.2 mol/m³ migrated; `C_HP_HAT` and `C_CLO4_HAT`
  auto-track via `C_SCALE`.

**What hasn't landed (the gap the plan closes):**

- ⏳ Disabled-reaction guard for `ln(k0)` in log-rate forms (H19 §1).
- ⏳ Parallel-topology Picard (any form — minimal or generic).
- ⏳ Multi-counterion Bikerman closure (currently
  `boltzmann.py:159-165` raises `NotImplementedError`).
- ⏳ Cs⁺ + SO₄²⁻ analytic-counterion entries.
- ⏳ Asymmetric multi-ion IC closure (composite-ψ for 2:1 sulfate +
  1:1 cation + 1:1 H⁺).

### 1.5 Existing sequential 2x2 Picard (math reference)

From `Forward/bv_solver/picard_ic.py:294-624`. Inside the outer loop
(picard_iter = 1..max_iters):

```python
# 1. Compute multispecies γ_s using current Picard iterates of H_o, ψ_D
gamma_s = compute_surface_gamma(H_o, c_clo4_bulk, psi_D, a_h, a_cl, c_cl_anchor)
log_gamma = log(gamma_s)

# 2. Compute reaction-plane log-concentrations using current iterates
log_O_rxn = log(O_s) + log_gamma
log_P_rxn = log(P_s) + log_gamma
log_H_rxn = log(H_o) - psi_D + log_gamma

# 3. Pre-compute the H+ stoichiometric factor per reaction (these
#    depend on H_o through log_H_rxn but treated as constants this iter)
log_h_factor1 = power_R1 * (log_H_rxn - log(c_HP_ref))
log_h_factor2 = power_R2 * (log_H_rxn - log(c_HP_ref))

# 4. Solve a *linear* 2x2 in (R1, R2) with constant coefficients:
log_A1 = log(k1) + log_gamma + log_h_factor1 - α1 * n_e * η1
log_B1 = log(k1) + log_gamma                  + (1 - α1) * n_e * η1
log_A2 = log(k2) + log_gamma + log_h_factor2 - α2 * n_e * η2

A1, B1, A2 = exp(log_A1), exp(log_B1), exp(log_A2)

#  Diffusion-flux <-> rate balance (linearized in O_s, P_s):
#    O_s = O_b - R_1 / D_O                       (R_2 doesn't consume O₂)
#    P_s = P_b + (R_1 - R_2) / D_P
#  R_1 = A_1 O_s - B_1 P_s
#  R_2 = A_2 P_s
#  Sub: 2x2 in (R_1, R_2) closed-form via det = m_11 m_22 - m_12 m_21.

m11 = 1.0 + A1/D_O + B1/D_P
m12 = -B1/D_P
m21 = -A2/D_P
m22 = 1.0 + A2/D_P
rhs1 = A1 * O_b - B1 * P_b
rhs2 = A2 * P_b
det  = m11 * m22 - m12 * m21
R1_new = (m22 * rhs1 - m12 * rhs2) / det
R2_new = (-m21 * rhs1 + m11 * rhs2) / det

# 5. Relaxation
R1 = (1-ω) R1 + ω R1_new
R2 = (1-ω) R2 + ω R2_new

# 6. Update outer-region scalars
O_s = max(O_b - R1/D_O, 1e-300)
P_s = max(P_b + (R1 - R2)/D_P, P_FLOOR)
H_o = max(H_b - (R1 + R2)/D_H, 1e-300)        # see ambipolar comment
phi_o = log(H_o / c_clo4_bulk)

# 7. Update ψ_D (Stern split or no-Stern)
# ...

# 8. Convergence check on |ΔR_1| + |ΔR_2| < tol
```

**Important comment in the code** about the ambipolar factor for the
sequential case:

```
# Ambipolar 2*D_H factor and -2 proton stoichiometry cancel
# (PNP_BV_Analytical_Simplifications.md lines 240-244). The
# denominator is bare D_H, NOT 2*D_H; do not "correct".
H_o = max(H_b - (R1 + R2) / D_H, 1e-300)
```

This is consistent with the signed ambipolar form
`H_o = H_b + Σ_j s_{H,j} R_j / (2 D_H)` once you fold in `s_H,1 = s_H,2 = -2`:
both cancel the 1/(2 D_H) factor to give bare D_H.

For parallel 2e/4e: `s_H,2e = -2`, `s_H,4e = -4`. So:
`H_o = H_b + (-2 R_2e - 4 R_4e) / (2 D_H) = H_b - (R_2e + 2 R_4e) / D_H`.

### 1.6 Multi-counterion Bikerman: structural blocker in current code

From `Forward/bv_solver/boltzmann.py:159-165` (verbatim):

```python
if len(bikerman) > 1:
    raise NotImplementedError(
        "multi-counterion bikerman closure not supported: when more than "
        "one counterion is steric-aware the closure algebra couples "
        "(each appears in the others' denominator).  See "
        "docs/steric_analytic_clo4_reduction_handoff.md caveats."
    )
```

The single-counterion analytic reduction is:

```
c(x) = c_b * exp(-z * φ) * (1 - A_dyn) / (theta_b + a_b * c_b * exp(-z * φ))
```

For two steric counterions (e.g. Cs⁺ and SO₄²⁻), each contributes to
the OTHER's denominator:

```
c_Cs(φ)  = c_b_Cs  * exp(-φ)  * (1 - A_dyn) / (theta_local(φ))
c_SO4(φ) = c_b_SO4 * exp(+2φ) * (1 - A_dyn) / (theta_local(φ))
theta_local(φ) = 1 - A_dyn - a_Cs · c_Cs(φ) - a_SO4 · c_SO4(φ)
```

The system is **implicit in (c_Cs, c_SO4) at each spatial point**.
There is no closed-form analytic reduction. The plan's "drop the
single-counterion guard and generalize" understates this; the
correct generalization requires either (a) solving a 2×2 local
nonlinear system at each spatial Newton iterate, or (b) one steric
+ one ideal, or (c) a different algebraic ansatz entirely.

### 1.7 Convergence rescue ladder from H19 (relevant context)

H19 prescribed an incremental ladder before doing M3a.3
(generic Picard rewrite):

- **Stage 1**: `ln(k0=0)` guard (handled in fast plan Phase 1.1).
- **Stage 2**: pure-channel reduced reaction list probes.
- **Stage 3**: warm-start parallel ctx from a freshly solved legacy
  ctx at the same V (in-memory, no disk).
- **Stage 4**: R_4e `k0` continuation via
  `ctx["bv_k0_funcs"][1].assign(...)`.
- **Stage 5**: voltage continuation from the weak-reaction side
  (V_RHE = +0.45 → +0.30 → … → -0.40).
- **Stage 6**: minimal 2-rate Picard *only if 1-5 fail*.

**The fast plan's Phase 1.2 inverts this**: it goes straight to a
hardcoded 2-rate Picard, bypassing Stages 3-5 entirely. The
rationale is "fail fast on the actual production target rather than
chase incremental rescues that may not generalize." GPT should
challenge whether this inversion is wise.

### 1.8 Page-15 voltage grid and target

From `StudyResults/mangan_p15_comparison/m3a0_observables.json`:

- 25 V_RHE points spanning [-0.40, +0.55] V.
- Acceptance bands are advisory (the fast plan deliberately doesn't
  enforce them).
- Anchor candidate per H19 §5: V_RHE = +0.45 V (weak-reaction side,
  far from R_2e onset at 0.695 V).

### 1.9 What the user explicitly DOESN'T want feedback on

- "This plan is too aggressive." (User said exactly this is what
  they want.)
- "You should regression-test against Run C." (User said no.)
- "You should add comprehensive unit tests for X." (User said no.)
- "You should split this into smaller phases." (Original plan had
  M3a.0/M3a.1/M3a.2/M3a.2.1/M3a.3/M3b.1-5; user said too slow.)

### 1.10 What GPT should hammer (per user's invocation message)

Verbatim from user's invocation of the critique-loop skill:

> GPT should hammer especially on:
> (a) the minimal 2-rate Picard math (signed ambipolar H_o equation,
>     the (H_o)^n nonlinearity vs the H19 §6 "linear 2x2" claim,
>     omega/tol choices),
> (b) the deferred asymmetric composite-ψ multi-ion IC ("let the
>     Picard absorb the shape mismatch") — is this actually viable
>     or self-deception?,
> (c) anchor-relocation strategy at I=0.3 M / 2:1 charge asymmetry,
> (d) the a_nondim values for Cs+ and SO4²⁻ (likely wrong as written),
> (e) whether running with K0_PHYS_R4E = K0_PHYS_R1 placeholder
>     produces anything physically interpretable or just garbage,
> (f) the disabled-reaction guard's interaction with the existing
>     topology gate / Picard dispatch.

Push hard on these. Other holes welcome too — the fast pace
maximizes review value.

## Section 2 — The artifact under review

The plan being critiqued, in full, is below. (Path:
`docs/fast_realignment_plan_2026-05-08.md`.)

```markdown
# Fast Realignment Plan — Real Experimental Setup

Date: 2026-05-08
Status: Replaces the M3a/M3b cascade in `docs/ruggiero_realignment_plan.md`
        for purposes of *getting the forward solver onto the actual
        Seitz/Mangan experimental setup as fast as possible*. Not a
        substitute for the long-form plan once the dust settles — it
        deliberately defers regression testing, comprehensive unit
        tests, the asymmetric multi-ion IC re-derivation, and the M4
        cation series.

## TL;DR

Three structural mismatches separate the current production stack
from the real Seitz/Mangan experiment:

1. Counterion identity. ClO₄⁻ analytic → Cs⁺ + SO₄²⁻ multi-ion
   (I=0.3 M, λ_D ≈ 0.55 nm).
2. Reaction topology. Sequential R_0 + R_1 → parallel R_2e + R_4e.
3. Initial condition / Picard. Sequential 2×2 Picard hardcoded;
   parallel topology rejected; orchestrator falls through to
   linear_phi which cold-fails universally.

The "fast" plan rips all three together, anchored on a single
warm-start strategy, with a one-shot driver that runs the page-15
voltage grid and prints what it gets.

## Hard constraints

- Begin with git checkpointing.
- No regression testing.
- No extensive unit tests.
- venv-firedrake activation, C+D orchestrator, exponent_clip=100,
  physical E_eq, all per CLAUDE.md hard rules.

## Phase 0 — Git checkpointing (10 min, FIRST)

1. `git status`.
2. `git add -A` selectively: `Forward/`, `scripts/`, `tests/`,
   `docs/`. Exclude `StudyResults/`.
3. Commit "chore: checkpoint pre-fast-realignment state ...".
4. Tag: `git tag pre-fast-realignment-2026-05-08`.
5. Branch: `git checkout -b fast-realignment-2026-05-08`.
6. No push.

Acceptance: `git log -1` + `git tag -l` + `git branch --show-current`.

## Phase 1 — Disabled-reaction guard + minimal 2-rate Picard (1-2 days)

### 1.1 Disabled-reaction guard for log-rate forms

In `forms_logc.py` and `forms_logc_muh.py`, before any `fd.ln(k0_j)`:

    k0_model_j = float(rxn["k0_model"])
    if k0_model_j <= 0.0 or bool(rxn.get("enabled", True)) is False:
        R_j = fd.Constant(0.0)
        bv_rate_exprs.append(R_j)
        bv_k0_funcs.append(_zero_k0_placeholder(R_space, j))
        bv_alpha_funcs.append(_zero_alpha_placeholder(R_space, j))
        continue

Smoke test: build 2-reaction config with k0_R4e = 0; confirm no
NaN/exception.

### 1.2 Minimal 2-rate parallel Picard (hardcoded)

For fixed H_o, psi_D, gamma_s, BV coefficients (the Picard outer
state):

    R_2e = A_2e · O_s · (H_o)^2 · gamma_s
    R_4e = A_4e · O_s · (H_o)^4 · gamma_s
    A_j  = k0_j · gamma_s · exp(α_j · n_e_j · η_j_clipped) · stoich_factors

    O_s  = O_b - (R_2e + R_4e) / D_O · L_eff
    P_s  = max(P_b + R_2e         / D_P · L_eff, P_FLOOR)
    H_o  = max(H_b - (R_2e + 2·R_4e) / D_H, 1e-300)
                                                # signed ambipolar

omega=0.5, max_iters=50, tol=1e-6.

Code surface:
- New `Forward/bv_solver/picard_ic_parallel_2rate.py` (~150 LOC).
- `forms_logc_muh.py:704-908` and `forms_logc.py:623-811`: replace
  topology gate with dispatch (sequential vs parallel-2e-4e).

Topology detection: H₂O₂ stoichiometry signature
- sequential: [+1, -1]
- parallel:   [+1,  0]

Smoke tests: pure-2e probe at V=+0.30; pure-4e probe; mixed.

## Phase 2 — Multi-ion Bikerman (Cs⁺ + SO₄²⁻) (1-2 days)

Bulk concentrations from Ruggiero §2:
- [H+] = 0.1 mol/m³
- [SO₄²⁻] = 100 mol/m³
- [Cs+] = 199.9 mol/m³
- I = 300 mol/m³ = 0.3 M, λ_D ≈ 0.55 nm

HSO₄⁻ speciation deferred (negligible at pH 4).

### 2.1 Drop single-counterion guard in `boltzmann.py`

Current: `boltzmann.py:90-` has
`assert len(boltzmann_counterions) == 1`. Drop, generalize residual
loop. Multispecies γ:

    γ_s = 1 / (1 + Σ_k a_k · c_b_k · (e^(-z_k·ψ) − 1)
                 + Σ_dyn a_i · c_i_dyn · packing_term)

### 2.2 Cs⁺ + SO₄²⁻ analytic-counterion entries

Add to `scripts/_bv_common.py`:

    A_CSPLUS_HAT = 0.0044   # 2.2 Å Linsey deck slide 13
    A_SO4_HAT    = 0.0048   # 2.4 Å Marcus radius (placeholder)
    C_CSPLUS = 199.9        # mol/m³
    C_SO4    = 100.0        # mol/m³

    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC: ...
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC: ...

`a_nondim` values flagged as probably wrong; will tune during
Phase 3.

### 2.3 Bulk electroneutrality check

`Σ z_k c_b_k_nondim + Σ z_i c_i_dyn_bulk_nondim = 0` to within 1e-12.

### 2.4 What's deferred

- Asymmetric multi-ion IC composite-ψ derivation. Workaround: call
  Picard with 2:1 Bikerman γ in place; let Picard absorb shape
  mismatch.
- Cation-swap parameterization (Li/Na/K). Hardcode Cs⁺.
- HSO₄⁻ speciation. Deferred.
- OH⁻ as tracked species. K_w-coupled.

## Phase 3 — One-shot driver (0.5 day)

New `scripts/studies/peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`:

    sp = make_bv_solver_params(
        ...,
        bv_reactions=PARALLEL_2E_4E_REACTIONS,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        stern_capacitance_f_m2=0.10,
        initializer="debye_boltzmann",
        exponent_clip=100.0,
        u_clamp=100.0,
    )
    result = solve_grid_per_voltage_cold_with_warm_fallback(
        sp,
        v_rhe_grid=PAGE_15_V_RHE_GRID,
        anchor_v_rhe=+0.45,
        max_z_steps=20,
        n_substeps_warm=8,
        bisect_depth_warm=5,
    )

Output: `StudyResults/fast_realignment_2026-05-08/run_A/`.

## Phase 4 — First production sweep (1 day if it converges)

Run the driver. Watch for Picard / Newton convergence, NaN, anchor
behavior. Acceptance: ≥1 V converges end-to-end.

## Phase 5 — Fix breaks (variable; expected longest)

Failure-mode triage:

- 5a. Picard non-convergence: drop omega → 0.3 / 0.2; max_iters →
  100; reaction-strength continuation; warm-start hack from legacy.
- 5b. Newton non-convergence at multi-ion EDL: continuation in I;
  continuation in a_nondim; mesh refinement Ny 200 → 400 → 800.
- 5c. Anchor relocation: scan V ∈ {+0.55, +0.50, +0.45, +0.40,
  +0.35, +0.30, +0.20, 0.0}.
- 5d. a_nondim calibration: reduce until pack-fraction stays bounded.
- 5e. Legacy driver regression: leave broken, user fixes later.

## Risks

| Risk | Where | Likelihood | Worst case | Triage |
|---|---|---|---|---|
| 2-rate Picard fails to converge with multi-ion γ | 1.2+2.1 | Medium | Need M3a.3 (full generic) | 5a fallbacks |
| Multi-ion bulk anchor at I=0.3 M doesn't cold-converge | 4 | Med-high | Zero converged points | 5b/5c |
| Disabled-reaction guard regresses legacy | 1.1 | Low | v13/v15/v16 break | Acknowledged |
| `a_nondim` literature radii violate pack-fraction | 4 | Medium | Newton fails | 5d |
| Symmetric cosh(ψ) IC closure incompatible with 2:1 Bikerman | 2.4 | Medium | Picard non-convergence | 5a + last-resort: derive asymmetric closure |
| K0_PHYS_R4E placeholder is OoM wrong | 4 | High | Shape and magnitude both off | Acknowledged |
| Anchor at +0.45 V cold-fails (legacy was 0.0 V) | 4 | High | Need anchor scan | 5c |

## Sequencing

```
Phase 0 (10 min)   →   Phase 1.1 (hours)   →   Phase 1.2 (1-2 days)
   →   Phase 2.1-2.3 (1-2 days)   →   Phase 3 (0.5 day)
   →   Phase 4 (1 day)   →   Phase 5 (variable)
```

Total: 5-10 days first end-to-end converged page-15 sweep.

## Acceptance for "done" (single criterion)

≥ 15/25 V_RHE converged with the new stack and a non-trivial gross
R_2e curve. No quantitative bands. User iterates after.
```

(End of artifact.)

## Section 3 — Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
