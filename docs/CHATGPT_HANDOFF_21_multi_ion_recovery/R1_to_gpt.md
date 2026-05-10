# Handoff to GPT — Multi-Ion Convergence Recovery (Phase 4 wall)

## Goal

Recommend the best next path forward for the fast-realignment effort.
We have a working multi-ion analytic-counterion infrastructure
(Cs⁺ + SO₄²⁻ shared-theta Bikerman closure) and a parallel-2e/4e
reaction set, but Newton diverges when we try to solve the coupled
PNP-BV system with this stack at any V_RHE in the page-15 grid
(±0.55, 0, ...). The plan lists three candidate escalation paths;
I want adversarial review on all of them and identification of any
better path I'm missing.

## Section 1 — Context bundle

### What this project is

`PNPInverse/` is a Firedrake-backed Poisson-Nernst-Planck + Butler-Volmer
forward solver for ORR (O₂ → H₂O₂ → H₂O) at pH 4 on RRDE / RDE
geometries. The production stack (May 2026) is:

  3 dynamic species (O₂, H₂O₂, H⁺)  +
  analytic counterion via Boltzmann residual on Poisson  +
  log-c primary variable (`u_i = ln c_i`)  +
  proton electrochemical-potential transformation
      (`mu_H = u_H + em·z_H·phi`)  +
  log-rate Butler-Volmer  +
  finite Stern compact layer (Robin BC)  +
  C+D orchestrator (per-V cold + z-ramp + warm-fallback).

This stack converges 15/15 over V_RHE ∈ [-0.5, +1.0] V on the LEGACY
single-counterion ClO₄⁻ + sequential R₁(O₂→H₂O₂) + R₂(H₂O₂→H₂O)
configuration.

### What we just changed (the fast-realignment plan)

Per Ruggiero 2022 J. Catal. (the deck's source paper), the experiment
is actually:
  - Cs₂SO₄ at I=0.3 M (NOT ClO₄⁻; verified by Seitz/Mangan data folder
    audit), so Cs⁺ at 199.9 mol/m³ + SO₄²⁻ at 100 mol/m³ + H⁺ at 0.1
    mol/m³ (pH 4).
  - PARALLEL ORR topology: R_2e (E°=0.695 V) + R_4e (E°=1.23 V),
    NOT sequential R₁+R₂.

Phases 0-3 of the plan landed:
  - Phase 1: disabled-reaction guard, Picard audit (works), helpers.
  - Phase 2.1: multi-counterion shared-theta closed-form in
    `Forward/bv_solver/boltzmann.py`. Replaces the old
    `len(bikerman) > 1` NotImplementedError.
  - Phase 2.2: Cs⁺/SO₄²⁻ entries in `scripts/_bv_common.py`:
        A_CSPLUS_HAT = 3.23e-5  (Cs⁺ at r=2.2 Å)
        A_SO4_HAT    = 4.20e-5  (SO₄²⁻ at r=2.4 Å)
        C_CSPLUS_HAT = 199.9 / C_SCALE  (= 166.58 with C_SCALE=1.2)
        C_SO4_HAT    = 100.0 / C_SCALE  (= 83.33)
    Verified: θ_b = 1 - A_dyn_bulk - Σ a_k c_b_k ≈ 0.991 (positive).
  - Phase 2.3: new `Forward/bv_solver/multi_ion.py` module with:
        build_counterion_ctx()  — single producer of theta_b
        solve_outer_phi_multiion()  — bisection on ρ_total(phi)=0
        compute_surface_gamma_multiion()  — multi-ion γ_s
        effective_debye_length_local()  — central-FD λ_eff
  - Phase 2.4: new multi-ion IC branch in
    `Forward/bv_solver/forms_logc{,_muh}.py:_try_debye_boltzmann_ic*`.
    Triggered when `len(bikerman) > 1`.
    Uses ψ-vs-φ split, linear-Debye Stern matching, multi-ion γ_psi(y)
    from shared-theta closure.
  - Phase 3: driver `scripts/studies/peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`.

Single-counterion ClO₄⁻ legacy code path is byte-equivalent (12/12
debye_boltzmann tests pass on logc + logc_muh).

### Hard rules from CLAUDE.md (non-negotiable)

1. C+D orchestrator (`solve_grid_per_voltage_cold_with_warm_fallback`),
   not Strategy B.
2. `exponent_clip = 100.0` (clip is on `α·n_e·η` AFTER multiplication;
   this is the only PC-trustworthy setting). Lowering this is a
   physics regression — see `docs/clipping_conventions.md`.
3. The IC and the residual must agree about steric saturation
   (composite-ψ + multispecies-γ both seeded). A bikerman IC without
   the matching residual cold-fails on the saturated manifold.
4. Solver convergence window: V_RHE ∈ [-0.5, +1.0] V via C+D
   (production reference: docs/4sp_bikerman_ic_option_2b_results.md).
   Use physical E_eq (R₁=0.68 V, R₂=1.78 V), never E_eq=0.

### Phase 4 result

Anchor smoke at V_RHE = +0.55 V (Ny=80) FAILS for the multi-ion stack.
Cold solve at z=1 (direct from analytic IC) diverges. Orchestrator
falls back to linear-phi + z-ramp; z=0 step ALSO diverges
(`snes_reason=-9 = DIVERGED_LINE_SEARCH`).

Full diagnostic dump:
  StudyResults/fast_realignment_2026-05-08/anchor_smoke/anchor_smoke.json

Key fields from that JSON:
```
"max_phi": 3.34,           # phi at electrode (linear-phi fallback)
"max_u1": +50.35,          # log H₂O₂ blew up (post-Newton-fail state)
"c1_surface_mean": 7.24e21,
"snes_reason": "-9",       # DIVERGED_LINE_SEARCH
"snes_iters": 1,           # diverged on first Newton iter
"picard_iters": 40,
"initializer_fallback": true,
```

The blown-up `u1 = +50` is the post-divergence state on the linear-phi
fallback IC (which the orchestrator falls back to AFTER the multi-ion
analytic IC's z=1 attempt fails). The actual multi-ion IC build is
sane — see Diagnostic 2 below.

### Diagnostic 1: legacy single-counterion + parallel-2e/4e WORKS

Same script with `boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]`
and parallel-2e/4e reactions converges:
  - cold success in 33.7 s
  - cd = -1.79e-6 mA/cm²
  - R_2e = -1.79e-6 mA/cm²

So: parallel-2e/4e topology + Stern (C_S=0.10 F/m²) + log_rate +
log_c_muh + log-rate BV all WORK. The regression is multi-ion-specific.

### Diagnostic 2: multi-ion IC build is sane in isolation

`scripts/studies/multi_ion_ic_debug.py` calls
`_try_debye_boltzmann_ic_muh` directly (no orchestrator) at +0.55 V.

Picard returns ok=True at iter 40 with state:
  R_list  = [-5.732e-13, 0.0]    (R_2e tiny, R_4e=0 (k0=0))
  c_s_list = [1.000, 1.000e-04, 0.0833]  (essentially bulk)
  phi_o   = -7.600           (Picard's 1:1-closed-form value)
  psi_D   =  10.87           (Picard's Stern-split with 1:1 c_clo4_b)
  psi_S   =  18.14
  gamma_s =  0.0035

NOTE: Picard's `phi_o = log(H_o / c_clo4_bulk)` is HARDCODED to
single-counterion 1:1 algebra. With c_clo4_bulk = first counterion's
bulk = C_CSPLUS_HAT = 166.58 (Cs⁺), this gives phi_o = log(0.0833/166.58)
= -7.60. That is the WRONG number for multi-ion electroneutrality.

After Picard, my Phase 2.4 multi-ion IC branch OVERRIDES phi_o using
solve_outer_phi_multiion(c_dyn_outer=Picard's c_s_list ≈ bulk):
  phi_o_local ≈ 0.0  (correct — bulk anchor for multi-ion)
  λ_eff      ≈ 1.75e-6 nondim (much smaller than single-ClO4
              λ_D=3.9e-5 because I=0.3 M strongly screens)
  psi_D_local ≈ 1.58 (linear-Debye Stern matching: stern·full_drop·λ_eff
                 / (eps + stern·λ_eff))

Resulting spatial IC fields (from debug script, after assemble):
  u_O2  ∈ [-0.073, 0.0]    (slight surface depletion)
  u_H2O2 ∈ [-9.28, -9.21]  (essentially bulk seed = 1e-4)
  mu_H  ∈ [-2.56, -2.49]   (≈ log(0.0833) = -2.49 everywhere)
  phi   ∈ [0.0, +1.58]     (psi_D_local at electrode → 0 at bulk)

These are physically sensible. log_gamma at electrode ≈ -0.072 (mild,
gamma_psi ≈ 0.93). c_H_at_OHP via mu_H reconstruction
= exp(mu_H - em·z_H·phi) = exp(-2.49 - 1·1·1.58) = exp(-4.07) ≈ 0.017
(c_H/c_H_b ≈ 0.21 — mild depletion, NOT artificial 11-orders-of-mag
collapse like the single-ClO4 case).

### Root cause of Newton divergence

With multi-ion physics correctly giving ψ_D ≈ 1.58, the BV cathodic
rate at +0.55 V is:

  η_R2e_at_surface = phi_applied - phi(0) - E_eq_R2e
                   = 21.4 - 1.58 - 27.05
                   = -7.23  (cathodic, exp factor exp(α·n_e·|η|))

  exp(0.627 · 2 · 7.23) = exp(9.07) ≈ 8700  (well below clip=100)

  R_2e_cathodic = k₀_R2e · c_O · (c_H/c_H_ref)² · exp(...)
                = 1.263e-3 · 1.0 · (0.017/0.0833)² · 8700
                = 1.263e-3 · 1.0 · 0.041 · 8700
                ≈ 0.45  (nondim)

Mass-transport limit (at L_REF=100 µm, D_O2): c_O_b - R/D_O2 = 1 - 0.45 = 0.55.
The "true" SS solution likely sits in this strongly-depleted regime, but Newton
from a nearly-bulk IC (c_O ≈ 1) cannot bridge there robustly when the rate
exponent doesn't have the 11-order-of-magnitude H⁺-depletion safety bound.

### Why the legacy single-ClO₄⁻ stack works

The Picard's `phi_o = log(H_o / c_clo4_bulk)` is exactly 0 when
c_clo4_bulk = c_H_bulk (the legacy 1:1 ClO4 case at pH 4 with both
nondim'd to 0.0833). With phi_o ≈ 0 and Stern+1:1-Bikerman split,
ψ_D ≈ 11 (Stern absorbs ~half of phi_applied=21.4, the rest is in the
diffuse layer).

ψ_D ≈ 11 → c_H/c_H_b = exp(-11) ≈ 1.7e-5 → (c_H/c_H_ref)² ≈ 3e-10.
That single factor BOUNDS the cathodic rate by 11 orders of magnitude
relative to the bulk-evaluated rate. Newton finds a self-consistent
state where R is tiny and c_O ≈ c_O_b. EASY.

But — this works ONLY because the "wrong" c_clo4_bulk = 0.0833 (matched
to H⁺ at pH 4) coincidentally makes phi_o = 0. Try Cs⁺ (c_bulk = 166.58
mol/m³) and the same code gives phi_o = log(0.0833/166.58) = -7.60,
which is mathematically wrong (NOT the multi-ion electroneutrality
solution) AND physically misleading. The resulting IC works in the
single-counterion case only because phi_o ≈ 0 happens to be both right
(by coincidence with bulk-pH electroneutrality) AND helpful (large
ψ_D bounds the rate).

Multi-ion at I=0.3 M is the realistic experiment. There the diffuse
layer is much thinner (λ_eff ≪ λ_D_single_ClO4) and ψ_D is genuinely
small. The BV rate at the production V_RHE grid genuinely is in the
"big and mass-transport-limited" regime. Newton has to find that
state, not be handed an artifact-friendly version of it.

### Relevant file paths and code

Multi-ion machinery:
  - `Forward/bv_solver/multi_ion.py` (NEW, Phase 2.3)
  - `Forward/bv_solver/boltzmann.py:90-265` (multi-counterion closure)
  - `Forward/bv_solver/forms_logc.py:949-1057` (multi-ion IC branch)
  - `Forward/bv_solver/forms_logc_muh.py:1003-1109` (muh multi-ion IC)
  - `Forward/bv_solver/picard_ic.py:1142` (the hardcoded 1:1 phi_o
    `phi_o = math.log(max(H_o, 1e-300) / max(c_clo4_bulk, 1e-300))`
    — currently used by the Picard EVEN in multi-ion mode; my Phase 2.4
    code overrides post-Picard but the Picard's INTERNAL eta_drop and
    psi_D inside the iteration are still computed with the wrong phi_o)

Orchestrator:
  - `Forward/bv_solver/grid_per_voltage.py:319-369` (cold-start with
    z-ramp + warm-fallback). Path B: try direct z=1 from analytic IC;
    if Newton diverges, reset to linear-phi + z-ramp.
  - LINE 350: `set_initial_conditions_logc(...)` — note: this calls the
    LOGC fallback even when formulation is logc_muh. Pre-existing bug
    that the multi-ion path now exposes (linear-phi fallback for muh
    formulation routes through wrong IC builder).

Solver options:
  - SNES_OPTS_CHARGED + linesearch_type=l2, maxlambda=0.3,
    snes_max_it=400, atol=1e-7, rtol=1e-10.

Experimental constants (from Ruggiero §2):
  - C_O2 = 1.2 mol/m³ (ORR-saturated water at pH 4-13)
  - D_O2 = 1.9e-9 m²/s, D_H2O2 = 1.6e-9, D_HP = 9.311e-9
  - C_HP = 0.1 mol/m³ (pH 4)
  - C_CSPLUS = 199.9, C_SO4 = 100.0 mol/m³ (electroneutrality + I=0.3 M)
  - K0_PHYS_R2E = 2.4e-8 m/s (= K0_PHYS_R1, placeholder)
  - K0_PHYS_R4E = 2.4e-8 m/s (PRIOR-SELECTED, same as R2E)
  - α_R2E = 0.627 (Tafel-fit), α_R4E = 0.5 (placeholder)
  - L_REF = 100 µm, V_T = 0.025693 V at 25°C

### Things already ruled out

  - `_try_debye_boltzmann_ic_muh` itself is NOT broken — it returns
    ok=True with sensible state.
  - `picard_outer_loop_general` for parallel topology IS NOT broken
    — Phase 1.2 audit confirmed geometric convergence for parallel-2e/4e.
  - The multi-counterion Bikerman closure in boltzmann.py is correct
    by algebra (single-counterion case byte-equivalent; 19/19 tests
    pass for multi-ion variants).
  - The parallel-2e/4e topology + Stern + log_rate stack DOES converge
    when paired with the legacy single-counterion ClO₄⁻ entry (proven
    in Diagnostic 1).

So the wall is specifically the COUPLING between:
  - Multi-ion's correct ψ_D ≈ 1.58 (small)
  - The BV rate's cathodic exponent at production V_RHE
  - Newton's ability to find the strongly-mass-transport-limited SS
    from any reasonable IC.

## Section 2 — Artifact under review

### Three candidate paths from plan §5

**Path 5b (I-ramp continuation)**:
Ramp ionic strength from low to literature values across pseudo-time
steps. Concretely: scale `C_CSPLUS_HAT, C_SO4_HAT` by α ∈ {0.01, 0.1,
0.5, 1.0} across N continuation steps, with Newton repair at each
step. The idea: at low α the EDL is closer to the single-ClO₄⁻ regime
(thicker diffuse layer, larger ψ_D, friendly H⁺ depletion), and at
each α step Newton has a warm-start from the previous α. Hopefully
the path "low-I large-ψ_D" → "high-I small-ψ_D" stays in a
solver-friendly basin throughout.

  Within fast-realignment scope.
  Costs: ~hours of solver time per V (multiple α per V).
  Risks:
    - The path may not actually be smooth; somewhere between α=0.1 and
      α=1.0 the rate transitions from "tiny" (1:1-friendly) to "0.45"
      (mass-transport-limited) and Newton may still struggle.
    - Need to re-instantiate context per α (because c_bulk_nondim is
      baked into the analytic counterion entries → the form), which
      means warm-start has to copy state across contexts. Not done in
      this codebase yet for analytic-counterion bulk concentrations.

**Path 5g (smarter spatial IC)**:
Replace linear `phi_outer(y)` with `solve_outer_phi_multiion()`
evaluated at multiple y nodes plus interpolation between them. Higher-
fidelity IC closer to the SS solution. Specifically: solve the
multi-ion electroneutrality at y nodes {0, λ_eff, 5·λ_eff, L_REF/2,
L_REF}, with c_dyn_outer(y) interpolated as needed, and build a
piecewise-cubic phi_outer(y).

  Out of fast-realignment scope (1-2 days per the plan).
  Costs: implementation effort + risk that even higher-fidelity IC
  doesn't help if the SS Newton basin is narrow.
  Risks:
    - We don't have evidence that the IC quality is the issue. The IC
      is already physically sensible (Diagnostic 2). Newton diverges
      because the BV rate at the IC is much larger than the SS rate,
      not because the IC is far from a Newton-friendly version of the
      SS in some norm we can refine away.
    - Adding more Newton-internal nonlinearity (cubic-spline-of-
      bisection-output evaluated at each quadrature point) may make
      assembly slower and harder to diagnose.

**Path "Outside the plan" (softplus-bounded BV rate)**:
Add a softplus or similar smooth saturation to the BV rate so that
during Newton iteration the rate is bounded above by a physically
motivated limit (e.g., k_Lm * c_O_b, the mass-transport limit).
Specifically, instead of `R = k₀ · c · exp(...) - k₀ · c · exp(...)`,
use something like
  R = R_lim · tanh( (k₀ · c · exp(...) - k₀ · c · exp(...)) / R_lim )
with R_lim chosen as a multiple of the diffusion-limited current.

  Out of plan, addresses root cause directly, but invasive.
  Risks:
    - Distorts the physical SS unless R_lim is chosen carefully and
      taken to ∞ at convergence (continuation in R_lim).
    - PC trustworthiness implications (CLAUDE.md hard rule 2 explicitly
      forbids modifying the BV exponent clip; a tanh saturation on the
      rate is a different bound but may still trip downstream
      assumptions about clip physics in `docs/clipping_conventions.md`).
    - Have to build it for parallel-2e/4e in both `forms_logc.py` and
      `forms_logc_muh.py`, plus handle log-rate vs concentration-rate
      branches.

### What I haven't done yet but plausibly should

  1. Patch `picard_outer_loop_general` to use multi-ion-aware phi_o
     (not the hardcoded 1:1 `log(H_o/c_clo4_bulk)`). The Picard's
     INTERNAL state is still wrong for multi-ion even though my
     Phase 2.4 IC builder overrides it post-loop. If the Picard's
     own eta_drop and psi_D were correct, the c_s_list it produces
     would also reflect mass-transport limitation, and the IC
     downstream would seed a much-more-depleted O_s and H_o, which
     could be the difference Newton needs.

  2. Pre-relax via a transient. Instead of asking SS Newton to solve
     directly, run a few transient time steps from a benign IC (e.g.,
     bulk-flat) and let the system find its own EDL via diffusion +
     electromigration. Then SS-solve from the relaxed state. The
     existing solver supports transient stepping (dt, t_end set in
     SolverParams).

  3. Smaller Stern coefficient continuation: ramp Stern from a "weaker"
     value (e.g., C_S = 0.01 F/m², which would absorb less of the
     drop and produce larger ψ_D) up to the literature value. This
     would mimic the single-ClO₄⁻ "large ψ_D" situation at the start
     and walk down to the realistic Stern.

  4. Reduce K0_R2e at first, walk up. Same logic as Stern: at very
     small k₀, the BV rate is bounded by k₀ even at full overpotential,
     so Newton has an easy SS. Walk k₀ from 0 to literature.

  5. Combine: cold-start at low V_RHE (e.g., -0.5 V where the Tafel
     direction puts R_2e in anodic direction with strong inhibition
     from c_H2O2 bulk = 1e-4 — actually anodic rate ∝ c_H2O2 which
     is tiny, so net rate is tiny → easy SS), then warm-walk to
     +0.55 V. The C+D orchestrator has warm-walk built in. We just
     need a cold success somewhere.

  6. Hybrid: use the ARTIFACT-BENEFICIAL single-ClO₄⁻ IC as a
     starting state (where ψ_D ≈ 11 → strong H⁺ depletion → Newton-
     friendly), then SWAP to multi-ion residual and let Newton find
     the correct multi-ion SS from this strongly-EDL IC. This relies
     on the "wrong-but-friendly" IC and the "right" residual coupling
     to a basin Newton can navigate.

### Question for GPT

  - Are the three plan §5 paths (5b, 5g, softplus) actually viable?
    Find every hole.
  - Are the 6 alternatives I sketched above viable? Find every hole.
  - Is there a path I'm missing that would be better than any of
    these?
  - Specifically: which path has the highest probability of
    ≥ 15/25 V_RHE converged with non-zero R_2e in the smallest
    number of additional engineering hours?
  - Is "Patch Picard for multi-ion phi_o" (option 1 in my list)
    actually the ROOT cause fix that enables everything else, and
    therefore should land BEFORE any continuation strategy?

I want a recommendation with reasoning, not a vague "try them all".
Prioritize by signal-to-implementation-cost ratio. Be willing to
say "the plan §5 paths are all wrong, here's what's actually
needed" if that's what the algebra says.

Cite specific lines / formulas / code paths. Don't be vague. If you
think my BV rate calculation is wrong (R ≈ 0.45 nondim), say where.

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
