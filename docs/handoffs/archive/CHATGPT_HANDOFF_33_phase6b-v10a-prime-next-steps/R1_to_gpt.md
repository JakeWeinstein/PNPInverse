# Round 1 — Phase 6β v10a' next-steps plan critique

## 1. Context bundle

### What just happened (the v10a V-sweep diagnostic outcome)

The Phase 6β v10a V-sweep diagnostic landed and ran cleanly through a
7-point V_RHE grid (+0.55, +0.40, +0.20, +0.10, -0.10, -0.30, -0.50)
at smoke kinetics + the production K2SO4 stack.  End-to-end Firedrake
wiring works, 25 unit tests on V_kin selection pass.

**Verdict: `no_candidate_passed_locked_rule`** — no V satisfied the
locked three-filter set even in fallback.

Per-V table from `iv_diagnostic.json`:

```
   V        σ_S      cd      R_2e      R_4e   x_4e  σ<0 cd_ok br_ok    θ      F0    R_net   sensS    sensV   o2lev
+0.55   +0.052   -0.59 -6.4e-10    0.669   1.00  F    T     F   0.525  0.052  0.0247  -0.189  -0.203  0.107
+0.40   +0.038   -0.94  6.9e-10    1.07    1.00  F    T     F   0.587  0.067  0.0276  -0.216  -0.224  0.171
+0.20   +0.019   -5.47  1.3e-09    6.21    1.00  F    F     F   0.683  0.101  0.0321  -0.246  -0.243  0.994
+0.10   +0.009   -5.53  5.5e-11    6.29    1.00  F    F     F   0.733  0.129  0.0345  -0.250  -0.242  1.006
-0.10   -0.009   -5.53  8.3e-14    6.29    1.00  T    F     F   0.828  0.226  0.0389  -0.217  -0.209  1.006
-0.30   -0.028   -5.53  1.1e-16    6.29    1.00  T    F     F   0.898  0.414  0.0422  -0.140  -0.143  1.006
-0.50   -0.046   -5.53  1.6e-19    6.29    1.00  T    F     F   0.940  0.732  0.0442  -0.076  -0.106  1.006
```

Where:
- `σ_S` is signed Stern surface charge density [C/m²]
- `cd` is total cathodic current density [mA/cm²], scale=−I_SCALE
- `R_2e`, `R_4e` are boundary-integrated nondim reaction rates
- `x_4e = R_4e/(R_2e+R_4e)` — branch selectivity
- `σ<0`, `cd_ok`, `br_ok` are the three LOCKED filter booleans:
  - `σ<0`: σ_S(V) < 0
  - `cd_ok`: |cd|/I_lim_4e < 0.9 with I_lim_4e = 5.50 mA/cm² at l_eff=16 µm
  - `br_ok`: 0.05 ≤ x_4e ≤ 0.95
- `θ = Γ/Γ_max` — Langmuir fractional coverage
- `F0` is the uncapped forward forcing (= k_hyd · ⟨c_M · 10^(-ΔpKa)⟩)
- `R_net = k_des·Γ` — H⁺ source from cation hydrolysis
- `sensS` is the perturbation-column derivative (Stern-cap manifold)
- `sensV` is the FD column (V_RHE manifold)
- `o2lev` is the branch-selectivity-independent O₂ Levich indicator

### The empty-intersection finding

- `σ<0` is True only at V ≤ −0.10 V.
- `cd_ok` is True only at V ≥ +0.40 V.
- `br_ok` is False at every V (pure-4e selectivity at the codebase
  defaults: K0_HAT_R4E = K0_HAT_R2E, `_bv_common.py:836`:
  > k0_R4e is a PRIOR-SELECTED placeholder (= K0_HAT_R2E) — not
  > calibrated to any data.

So the σ_S<0 region and the cd_ok region have **empty intersection**
at this parameter setup.

### Codebase parameter values (current)

```
# scripts/_bv_common.py
E_EQ_R2E_V = 0.695           # V vs RHE (Ruggiero §1, peer-reviewed)
E_EQ_R4E_V = 1.23            # V vs RHE (Ruggiero §1, peer-reviewed)
ALPHA_R2E  = ALPHA_R1 = 0.5  # (legacy default; "placeholder" per comment)
ALPHA_R4E  = 0.5             # explicit comment: "default placeholder; revisit in M4 with Tafel"
K0_PHYS_R4E = K0_PHYS_R1     # explicit comment: "PRIOR-SELECTED placeholder ... not calibrated"

# Stern compact layer
stern_capacitance_f_m2 = 0.10     # legacy; new production target 0.20
                                  # per .research/cmk3-stern-capacitance/SUMMARY.md
                                  # (Bohra-Koper-Choi convergence on Stern_thickness=5Å, ε_S=11.3)

# Transport
l_eff_m = 16e-6              # boundary-layer thickness; production
C_O2    = 1.2 mol/m³         # bulk O2 (= C_SCALE because reference)
D_O2    = 1.9e-9 m²/s
L_REF   = 1e-4 m             # nondim length scale

# Smoke cation hydrolysis (Phase 6β v10a)
k_hyd  = 1e-3 nondim
k_prot = 1e-3 nondim
k_des  = 1.0 nondim
delta_ohp_hat = 4e-6         # 0.40 nm / L_REF
gamma_max_nondim = 0.047     # 1-monolayer smoke baseline
r_H_El_pm = 200.98           # Singh Cu prior
```

### BV math (parallel 2e/4e cathodic branch)

Cathodic BV rate for each reaction j:

    R_j  =  k0_j · c_O2 · (c_H/c_H_ref)^{n_e_j} · exp(−α_j · n_e_j · η_j / V_T)
    η_j  =  V_RHE − E_eq_j   (η < 0 is cathodic; rate grows as exp(+...))

Voltage sensitivity of the cathodic rate (since η < 0):

    ∂ ln R_j / ∂ V_RHE  =  −α_j · n_e_j / V_T

At α_j = 0.5, n_e_j = 4, V_T = 0.0257 V:
    ∂ ln R_4e / ∂ V = −0.5 · 4 / 0.0257 ≈ **−78 / V**

So R_4e grows by `e^78` per 1 V more cathodic, by `e^(78·0.3) ≈ e²³ ≈ 1e10`
per 0.3 V more cathodic.

### Plateau-onset condition

cd plateau when R_4e (BV) hits the O₂ Levich limit:

    k0_R4e · c_O2 · c_H^4 · exp(−α·n_e·η/V_T)  ≈  D_O2 · c_O2 / l_eff

The V at which this happens depends on k0_R4e, α, l_eff (Levich denom),
and the H⁺ stoichiometric factor.  From the diagnostic table:
plateau onset is between V=+0.40 (cd=−0.94) and V=+0.20 (cd=−5.47),
i.e. around V ≈ +0.30 V.  We need plateau onset to move to V < 0 V
(roughly a 0.3 V shift more cathodic) so the σ_S<0 region overlaps
with the cd_ok region.

### The three knob candidates I considered

#### Knob A — Reduce K0_R4e (multiplicative on R_4e BV)

Plateau shift from reducing K0_R4e by factor F:
At fixed V, R_4e' = F · R_4e.  Plateau condition unchanged in form,
so the plateau-onset V satisfies:
    F · (BV exp at new plateau)  =  (BV exp at old plateau)

⇒ log F = −78 · ΔV    (for α=0.5, n_e=4)
⇒ For ΔV = 0.3 V: F = e^(−78·0.3) = e^(−23.4) ≈ **7e−11** (factor ≈ 1e−10).
⇒ For ΔV = 0.5 V: F ≈ 1e−17.

Project memory `project_k0_r4e_ratio_regimes`:
> K0_R4e/K0_R2e ratio sweep: three regimes; Butler shape + Mangan-like
> ~35-50% peroxide selectivity emerges at ratio ≈ 1e-18; pure-2e
> saturates at ratio ≤ 1e-24.

i.e. **factor ≈ 1e-18 on K0_R4e** at codebase defaults brings the
selectivity into the mixed regime (35-50% H₂O₂) and would push the
4e plateau-onset to V ≈ −0.5 V.

#### Knob B — Lower α_R4E (transfer coefficient)

α_R4E is documented as a placeholder; experimentally constrained but
not yet calibrated against Tafel slopes for the deck.  Math: at fixed
V, the plateau condition becomes
    α'·n_e·η_plateau' = α·n_e·η_plateau_old
    ⇒ η_plateau' = (α/α') · η_plateau_old

For plateau-onset at V_new = 0 (so η_new = −1.23 V at the 4e channel):
    α' = α · η_old / η_new  =  0.5 · (−0.93) / (−1.23)  ≈ 0.378

So lowering α_R4E from 0.5 to 0.38 (a 24% reduction, within the
experimental Tafel uncertainty for ORR which spans α ∈ [0.30, 0.50])
would shift plateau onset by ~0.30 V.

#### Knob C — L_eff (Levich denominator)

I_lim_4e = 4·F·D_O2·c_O2/l_eff scales linearly in 1/l_eff.  BUT |cd|
in the BV-controlled regime also doesn't depend on l_eff (it's set by
BV kinetics × c_O2_surface).  The plateau onset V satisfies
    R_4e(V_plateau) = D·c/l_eff
    ⇒ log(K0_R4e) − 78·V_plateau + log(c_H^4) = log(D·c/l)
    ⇒ V_plateau_new − V_plateau_old = (1/78) · log(l_old/l_new)
    ⇒ ΔV per 2× reduction in l_eff = log(2)/78 = **0.009 V (9 mV)**

To shift plateau by 0.3 V via l_eff alone: l_eff_new/l_eff_old = e^(−78·0.3) ≈ 1e−11.
With l_eff_old = 16 µm, l_eff_new ≈ 0.16 femtometers.  **Sub-Angstrom.
Not physical.**  L_eff cannot be the knob.

#### Knob D — c_O2_bulk

Both R_4e (BV cathodic rate) and I_lim_4e (Levich limit) scale linearly
in c_O2.  Ratio |cd|/I_lim is invariant.  **Not a knob.**

#### Knob E — E_eq

Thermodynamic, pinned by Ruggiero 2022 §1 (E_eq_4e = 1.23 V vs RHE
peer-reviewed).  Cannot be tuned without misrepresenting the physics.
**Not a knob.**

### What the plan proposes

1. Bump `STERN_F_M2_BASELINE` from 0.10 to **0.20** (the new
   literature-anchored CMK-3 value from
   `.research/cmk3-stern-capacitance/SUMMARY.md` — 2026-05-10).
2. Add a `--k0-r4e-factor` CLI flag (default 1.0, dimensionless
   multiplier on K0_HAT_R4E in the bv_reactions list).
3. Re-run the 7-point grid at C_S = 0.20 + factor = 1e-18 (project
   memory's calibrated mixed-selectivity point).  ~1.5 h wall.
4. Bracket-sweep fallback {1, 1e-6, 1e-12, 1e-18, 1e-24} if the
   single-factor run still doesn't yield a clean V_kin.  ~7 h wall.

### The three focus points I want critiqued

**Focus 1: K0_R4e the right knob?**
My math says l_eff is ~9 mV per 2× → wildly insufficient for ~300 mV.
K0_R4e via BV exponential gives the right OOM (~e²³ for 0.3 V shift).
α_R4E is the only other knob with the right OOM, and it's also a
"placeholder" in the codebase.  c_O2 and L_eff are invariant for
|cd|/I_lim_4e.  E_eq is locked.  Is there another knob I'm missing?
And — if α_R4E is also viable, why pick K0_R4e instead?  My
reasoning is: (a) project memory already characterized the K0 ratio
sweep, (b) α has tighter experimental constraints than k0 (Tafel
slopes are measured), but both are "placeholders" in the codebase.

**Focus 2: C_S = 0.10 → 0.20 × Langmuir-cap interaction.**
The new C_S = 0.20 doubles σ_S magnitude at every V.  At fixed V,
larger |σ_S| → larger ΔpKa (Singh formula scales ~linearly with σ_S
in cathodic regime) → larger F₀ → larger Γ_ss via the closed form.
With the cap active, Γ_ss → Γ_max more aggressively.  But the
plan also reduces K0_R4e by 1e18 (knob A), which doesn't directly
affect F₀ (cation hydrolysis is K⁺-driven, not 4e-coupled).
Question: does the C_S bump push the OPERABLE σ_S<0 region (where
sensS is high) deeper into cap-saturation (where sensS collapses)?
The v10a run already showed sensS collapse from 0.250 at V=+0.10
(θ=0.73) down to 0.076 at V=−0.50 (θ=0.94) — i.e. sensS drops
3× as θ climbs from 0.73 to 0.94.  With C_S=0.20, all these θ
values shift upward.  Could the answer be that the sensS maximum
ALWAYS lies in the anodic σ_S>0 region (where the locked filter
rejects), regardless of which knob we turn?

**Focus 3: Bracket-sweep coverage {1, 1e-6, 1e-12, 1e-18, 1e-24}.**
Project memory characterizes three regimes:
- "Butler shape + Mangan-like 35-50% peroxide at ratio ≈ 1e-18"
- "pure-2e at ratio ≤ 1e-24"
- (implicitly) pure-4e at ratio = 1.

The bracket spans 5 orders of magnitude (3 of them on log scale).
The transitions between regimes are presumably exponential in the
ratio (since BV rates are exponential in V_eq).  Is 5 points
sufficient to characterize the transitions, or should I densify
around 1e-18 (the calibrated point)?  What's the right design —
geometric spacing around 1e-18, or wide spacing across the three
regimes?

### Out of scope (not under critique)

- Driver edit mechanics (the `--k0-r4e-factor` CLI flag wiring).
- Test list (the V_kin selection test class still applies; new
  factor-flag tests are mechanical).
- File modifications + decision tree.
- The v10a Langmuir cap formulation itself (approved in session 31).
- The V_kin selection rule (approved in session 32; locked by
  acceptance bundle).

## 2. The plan under review

(Full plan from `~/.claude/plans/sparkly-gilded-pasteur.md`.)

# Plan — Phase 6β v10a' V-sweep diagnostic with corrected parameters

[... full plan body inlined here ...]

[ELIDED for brevity — see the linked file path; the structural
parts (file modifications, test list, decision tree, risks, after-
v10a' sequence) are out-of-critique-scope per session-32 + this
session's focus.]

## 3. Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

**Scope of this review (narrow):** only the three subtle points
called out above:
  1. Is K0_R4e the right knob to shift cd plateau onset by ~0.3 V?
     Or is α_R4E (or something else) cleaner / more defensible?
  2. C_S 0.10 → 0.20 doubles σ_S → faster Γ cap saturation; does
     this kill σ_S-sensitivity in the σ_S<0 region we're trying
     to recover?
  3. Bracket sweep {1, 1e-6, 1e-12, 1e-18, 1e-24} — dense enough?
     Should we densify around 1e-18 (the calibrated point)?

Do NOT critique:
  - The locked V_kin selection rule (approved in session 32,
    locked by acceptance bundle).
  - Driver structure, CLI wiring, test list, file mods.
  - The v10a Langmuir cap residual / Picard formula (approved in
    session 31, landed today).
  - The general flow of the plan (run diagnostic, evaluate, route).

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
