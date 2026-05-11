# Round 1 — Phase 6β v10a V-sweep diagnostic plan critique

## 1. Context bundle

### Project + state

Research code for Poisson–Nernst–Planck / Butler–Volmer simulation of
ORR (O₂ → H₂O₂ → H₂O) on a Seitz/Mangan-deck-aligned stack:
K₂SO₄ electrolyte at pH 4–6, parallel 2e/4e ORR (Ruggiero 2022:
E°_2e = 0.695 V vs RHE, E°_4e = 1.23 V vs RHE).  Stack is 4 species
(O₂, H₂O₂, H⁺, K⁺ dynamic; SO₄²⁻ analytic Bikerman counterion),
log-c + μ_H proton primary variable, Stern compact-layer C_S = 0.10
F/m², `l_eff = 16e-6 m`.

**Phase 6β v10a landed today** (2026-05-10): a Langmuir capacity cap
`(1 − Γ/Γ_max)` on the cation-hydrolysis forward branch.  Without
the cap, v9 had Γ growing to 6+ monolayers (unphysical).  v10a
saturates at `Γ_max = 0.047` nondim (≈ 1 monolayer of MOH at the
OHP).

### Cation hydrolysis residual (post-v10a)

For the cation `M⁺ = K⁺` at the electrode boundary:

    R_net = k_hyd · c_M⁺(0) · 10^(−ΔpKa(σ_S)) · (1 − Γ/Γ_max)
            − k_prot · c_H(0) · Γ / δ_OHP
    R_des = k_des · Γ                                          (desorption)

Steady state for Γ (closed form solved by outer Picard between
Newton solves):

    Γ_ss(λ) = λ · F₀ / ((1 − λ) + λ·k_des + λ·k_prot·⟨c_H⟩/δ_OHP
                        + λ·F₀/Γ_max)

where `F₀ = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩` is the boundary-area-averaged
uncapped forward forcing.  At steady state, by mass balance
`R_net = k_des · Γ`.  This is the H⁺ source term added to the proton
boundary residual (and the K⁺ sink added to the cation residual).

### ΔpKa(σ_S) (Singh 2016 SI Eq. 4)

    ΔpKa(σ_S) = +2 · A · z · σ_singh(σ_S) · r_H_El · G

    σ_singh = max(0, −σ_S) · (N_A/F) · 1e-24   counts/pm² (anode-clamp)
    G = 1 − (r_M-O / r_H_El)²                   geometric factor

K⁺ Singh Table S1 + Cu r_H_El back-fit:
- z_eff = 0.919, r_M = 138 pm, r_O = 63 pm, r_H_El = 200.98 pm.
- r_M-O = r_M + r_O = 201 pm; r_M-O > r_H_El by 0.02 pm.
- G < 0  ⇒ cathodic σ_S → ΔpKa < 0 (lowers hydrolysis pKa, more H⁺).

### Stern Robin BC + σ_S definition

The Robin BC at the electrode (nondim):

    C_S · (φ_applied − φ_s) = σ_S_nondim

`stern_coeff_const` on ctx is the nondim Stern capacitance (in units
that absorb the V_T / L_REF factors).  The form-build code wires:

```python
# forms_logc_muh.py:670–704 (paraphrased)
length_scale = float(scaling.get("length_scale_m", 1.0))      # = L_REF = 1e-4 m
concentration_scale = float(scaling.get("concentration_scale_mol_m3", 1.0))  # = C_SCALE = 1.2 mol/m³
sigma_phys_per_nondim = float(_F) * concentration_scale * length_scale  # = 96485 · 1.2 · 1e-4 ≈ 11.58 C/m²

sigma_S_expr = stern_coeff * (phi_applied_func - phi) * fd.Constant(sigma_phys_per_nondim)
# this is the *signed physical* C/m² that the Singh helper consumes
```

So *both* `stern_coeff` (live FE Constant) and `(φ_applied − φ)`
determine σ_S.  Perturbing `C_S` changes `stern_coeff`; Newton then
re-equilibrates `(φ_applied − φ)` to satisfy charge balance.

### v10a rung diagnostics (already implemented)

`collect_v10a_rung_diagnostics(ctx)` returns per-rung dict:
- `gamma` (Γ), `gamma_max`, `theta = Γ/Γ_max`
- `F0_avg`, `forward_avg_no_k_hyd`, `c_H_avg`, `pka_shift_avg`
- `R_forward_capped = F₀·(1−θ)`
- `denominator_constant`, `denominator_kdes`, `denominator_kprot`,
  `denominator_cap`, `denominator_total`, `numerator`
- `sigma_S_C_per_m2` (signed, physical)
- `sigma_S_counts_per_pm2` (via 1/e × 1e-24 factor)
- `R_2e_current_nondim`, `R_4e_current_nondim`
  (boundary integrals of `R_j` per reaction; multiply by I_SCALE-like
  factor for physical mA/cm²)

### Codebase constants relevant to this critique

From `scripts/_bv_common.py`:

```
F_CONST     = 96485.3329 C/mol   (Faraday)
D_O2        = 1.9e-9  m²/s       (O₂ diffusivity — codebase value)
C_O2        = 1.2     mol/m³     (O₂ bulk; = C_SCALE because reference)
L_REF       = 1.0e-4  m          (length scale = 100 µm = D_O2 boundary layer)
C_SCALE     = C_O2 = 1.2
D_REF       = D_O2

N_ELECTRONS = 2                  (BV reference electron count)

I_SCALE     = n_e × F × D_ref × c_scale / L_ref × 0.1
            = 2 × 96485 × 1.9e-9 × 1.2 / 1e-4 × 0.1
            ≈ 0.4400 mA/cm²        (× nondim rate ⇒ mA/cm²; the 0.1
                                    converts A/m² → mA/cm²)
```

The `current_density` observable mode (in `observables.py`) weights
each reaction by `(n_e_j / N_ELECTRONS_REF)` with `N_ELECTRONS_REF = 2`:

```python
# observables.py:106–121
if mode_norm == "current_density":
    n_e_list = _get_reaction_n_electrons(ctx)
    rate_sum = 0
    if n_e_list is not None and len(n_e_list) == len(bv_rate_exprs):
        ref = float(N_ELECTRONS_REF)
        for n_e_j, R_j in zip(n_e_list, bv_rate_exprs):
            weight = fd.Constant(float(n_e_j) / ref)
            rate_sum = rate_sum + weight * R_j
    return scale_const * rate_sum * ds(electrode_marker)
```

For parallel 2e/4e: cd = `-I_SCALE · ((2/2)·R_2e + (4/2)·R_4e) · ds`
                  = `-I_SCALE · (R_2e + 2·R_4e) · ds`

So the cd value is in mA/cm² physical, anchored to `I_SCALE` (2e).
The factor `n_e_j / N_ELECTRONS_REF` re-weights each branch so the
total respects per-reaction electron count.

### l_eff convention

`l_eff_m = 16e-6 m` (16 µm) is the H⁺/O₂ boundary-layer thickness
for the production stack.  Mesh is built with
`domain_height_hat = l_eff_m / L_REF = 0.16`.  The bulk Dirichlet
BC sits at y = domain_height_hat (= 16 µm physical), not at L_REF.

### What this loop is critiquing

The plan introduces a V-sweep diagnostic driver (Phase 6β v10a
step 3) that walks V_RHE at smoke kinetics, runs both λ=0 baseline
and λ=1 cap-active passes, and emits a V_kin candidate list filtered
+ scored per a *locked* selection rule.

The locked V_kin rule is verbatim from the acceptance bundle:

    argmax(|dR_net/dσ_S|) subject to filters {
        σ_S < 0,
        |cd|/I_lim_4e < 0.9,
        R_2e/(R_2e + R_4e) ∈ [0.05, 0.95],
    }
    with a fallback rule and a fail-stop to v10c if no V has σ_S < 0.

The rule itself is locked — *don't* critique it.

What I want critiqued is *how the driver computes the inputs* to
that rule.  Three subtle points:

#### Point 1: perturbation column actually measures dR_net/dσ_S?

The plan proposes computing the sensitivity by re-solving at C_S ·
(1 ± ε) with ε = 0.05, then:

    dR_net/dσ_S ≈ (R_net(C_S·(1+ε)) − R_net(C_S·(1−ε)))
                  / (σ_S(C_S·(1+ε)) − σ_S(C_S·(1−ε)))

i.e. we *vary C_S* but *measure d/dσ_S* by using the actual σ_S
values that come out of the solve, not C_S itself.

Conceptual concern: when we perturb C_S, ΔR_net contains contributions
from changes in c_H(0), c_M(0), Γ, etc., not just from σ_S.  So what
the formula returns is a *total derivative along the C_S constraint
manifold*, not a partial derivative.  Define:

    R_net = R_net(σ_S, c_H(σ_S), c_M(σ_S), Γ(σ_S, c_H, c_M), ...)

    ΔR_net/Δσ_S = (∂R_net/∂σ_S)|_others
                  + (∂R_net/∂c_H)·(dc_H/dσ_S)
                  + (∂R_net/∂c_M)·(dc_M/dσ_S)
                  + (∂R_net/∂Γ)·(dΓ/dσ_S along C_S manifold)
                  + ...

Possible justifications:

(a) The total derivative along the C_S manifold *is* what we want
    for V_kin selection, because in Phase E we'll be calibrating
    against experimental cation-pH curves *by tuning C_S* (or by
    fitting Singh r_H_El which equivalently shifts σ_singh).  So
    sensitivity-along-C_S = sensitivity-against-calibration-knob,
    which is the relevant question for "where in V do we get the
    best fitting handle?".

(b) Alternatively, perturb φ_applied directly (the literal voltage)
    — but that's just the FD column by another name.

(c) Or perturb r_H_El_pm in `cation_hydrolysis_config`'s pka_shift_params,
    which would shift ΔpKa via the σ_singh-independent geometric
    factor.  Does NOT directly shift σ_S, however — only shifts F₀
    via 10^(-ΔpKa).  So this measures dR_net/dr_H_El, not dR_net/dσ_S.

I'm inclined toward (a): keep the C_S perturbation, accept that
it's a total-derivative-along-C_S-manifold, document it explicitly.
But I want this challenged.

**Question for GPT**: is the total-derivative-along-C_S interpretation
defensible for V_kin selection?  Should the plan switch to a cleaner
perturbation knob?  If keeping C_S, what's the most rigorous way to
describe what the column measures so downstream readers (Phase E
calibrators) don't get confused?

#### Point 2: 50 % FD/perturbation disagreement threshold

Plan says: if FD and perturbation columns disagree by > 50 % relative
at a given V, flag that V as `sensitivity_ok = False` and exclude
from V_kin candidates.  Reasoning was loose: "noisy spot mid-saturation".

Re-reading after writing the plan: FD measures the *total derivative
along V_RHE*, perturbation measures the *total derivative along C_S*.
These should agree only when σ_S is the dominant R_net driver at
that V.  If they disagree, it means some other factor (Γ saturation,
c_H surface depletion, cd plateau) is dominating ΔR_net.  That's
genuinely useful information: it filters out V where R_net is *not*
σ_S-controlled — which is exactly the right filter for "where do we
get clean fitting handle on σ_S parameters?"

But the threshold value 50 % is still hand-waved.  Options:

(a) Hold at 50 %, document the reasoning above as the rationale.
(b) Set threshold = median(|FD − perturb|/max(|FD|, |perturb|))
    across V, drop V > 2× median.  Adaptive.
(c) Use a tighter threshold (20 %) in the "primary" candidate set
    but loosen in fallback.
(d) Drop the cross-check entirely; trust perturbation column as
    primary, log FD as informational.

**Question for GPT**: pick a defensible threshold, or propose a
better criterion.

#### Point 3: Levich cross-check

Plan's draft formula:

    I_lim_4e_phys = 4 · F · D_O2 · c_O2 / l_eff_m            [A/m²]
                  · 0.1                                       [→ mA/cm²]

This is the standard 4-electron Levich limit (planar, fully developed
diffusion layer).  The cd values come from observables.py weighted
sum `Σ (n_e_j / N_ELECTRONS_REF) · R_j` multiplied by
`scale = -I_SCALE` where `I_SCALE` uses `N_ELECTRONS = 2`.

For a pure-4e cathode (R_2e = 0):
    cd_phys = -I_SCALE · (4/2) · R_4e = -2 · I_SCALE · R_4e [mA/cm²]
            = -2 · (2·F·D·c/L_REF · 0.1) · R_4e
            = -(4·F·D·c/L_REF · 0.1) · R_4e

At Levich limit (R_4e = O₂ flux / O₂_consumed_per_R = N_O2 / 1 per
R_4e), R_4e_lim ≈ N_O2_lim = D_O2·c_O2/l_eff  (in nondim, c_O2/l_eff
since D=1, c=1, L=1 normalisation).  Wait this is getting confusing.

Let me sanity-check via numbers.  D_O2 = 1.9e-9 m²/s, c_O2 = 1.2
mol/m³, l_eff = 16e-6 m.

    I_lim_4e = 4·F·D·c/l_eff
             = 4 · 96485 · 1.9e-9 · 1.2 / 16e-6
             = 54.97 A/m²
             = 5.50 mA/cm²

The plan wrote D_O2 = 2.18e-9 — that's the WATER LITERATURE VALUE,
not the codebase's D_O2 = 1.9e-9.  Plan needs fixing: use the
codebase's D_O2 so cd/Levich uses consistent constants.  Corrected:
I_lim_4e ≈ 5.50 mA/cm² at l_eff = 16 µm.

For cross-checking dimensions of the `|cd|/I_lim_4e < 0.9` ratio:

    cd_value (mA/cm²) / I_lim_4e (mA/cm²)   ⇒ dimensionless ✓

But the cd value is a *signed* current (negative cathodic), and
includes BOTH 2e and 4e contributions:

    cd_phys = -I_SCALE · (R_2e + 2·R_4e)  [mA/cm²]

For pure 4e at Levich: cd_phys = -I_lim_4e ≈ -5.50 mA/cm² ⇒ ratio = 1.0 ✓
For pure 2e at Levich: cd_phys = -I_SCALE · R_2e_lim ⇒ what is R_2e_lim?
    R_2e_lim consumes 1 O₂ per reaction (stoich -1 on O₂), but only 2e.
    Same O₂ flux limit ⇒ R_2e_lim has the same magnitude as R_4e_lim
    in the *O₂ flux* sense.  So at pure 2e Levich:
    cd_phys = -I_SCALE · R_2e_lim = -(2·F·D·c/L_REF · 0.1) · (c·D/l_eff in nondim)
            = -2·F·D·c/l_eff · 0.1 (after the nondim cancels)
            ≈ -2.75 mA/cm²    (= I_lim_4e / 2)

So `|cd|/I_lim_4e` ranges from 0 to 1.0 at pure 4e, but only to 0.5
at pure 2e.  The threshold 0.9 effectively requires partial 4e to be
"on the plateau" — V with pure 2e selectivity will *never* hit the
0.9 ratio regardless of how Levich-limited they are.

**Question for GPT**: is `I_lim_4e` the right Levich reference?
Should it instead be `I_lim_O₂_consumed_max = 4·F·D_O2·c_O2/l_eff`
(the maximum O₂-flux-limited current assuming pure 4e — same number
as above), or branch-specific Levich, or O₂-flux Levich (no F/n_e
factor, just `D·c/l`)?  What does the locked rule's "non-plateau"
test actually mean for a parallel-2e/4e cathode?

Also: dimensional sanity on the `·0.1` factor I added — A/m² →
mA/cm² is `× 0.1`?
  1 A/m² = 1 A · m⁻²
        = 1000 mA · (10000 cm²)⁻¹
        = 0.1 mA/cm²   ✓

### Working files relevant to GPT's review

- `Forward/bv_solver/cation_hydrolysis.py:540–595` — Langmuir
  closed form
- `Forward/bv_solver/cation_hydrolysis.py:984–1110` — diagnostics
  helper
- `Forward/bv_solver/forms_logc_muh.py:670–710` — σ_S expression
- `Forward/bv_solver/observables.py:106–121` — cd weighting
- `scripts/_bv_common.py:200–226` — I_SCALE definition
- `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md:155–172` —
  locked V_kin rule

## 2. The plan under review

(Pasted verbatim from `~/.claude/plans/whimsical-tumbling-hoare.md`.)

# Plan — Phase 6β v10a Minimum V-sweep diagnostic + V_kin selection

## Context

Phase 6β v10a (Langmuir capacity cap on the cation-hydrolysis residual)
landed 2026-05-10. The acceptance bundle locked the post-v10a sequence
(`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` §
"Sequenced re-do plan"): step 3 is the **minimum V-sweep diagnostic**
and step 4 is **V_kin selection by a predeclared rule**. Together
they produce the V_kin that Phase A.2 and the plumbing-ablation matrix
will then run at.

This plan covers steps 3 + 4 only. It is **not** Phase A.2 — that's the
densified λ + k_hyd ramp _at_ V_kin, run after this driver lands.

**Estimated scope:** 1 day of focused implementation (per the
acceptance lock), ~10-30 min wall per smoke run.

## What the driver produces

1. A `σ_S(V)` curve at λ=0 (baseline, cap inactive — sanity reference).
2. A `σ_S(V)` curve at λ=1 with the v10a Langmuir cap active.
3. Per-V (λ=1): `cd`, `pc`, per-reaction `R_2e` / `R_4e` currents,
   `c_H(0)`, `c_K(0)`, `Γ`, `θ=Γ/Γ_max`, `R_net=k_des·Γ`, `σ_S` in both
   C/m² and counts/pm², `dR_net/dσ_S` estimate (two methods, see
   below).
4. A V_kin candidate list filtered + scored per the locked rule, plus
   the chosen V_kin (or `abort_to_v10c` flag if no V has σ_S < 0).
5. `iv_diagnostic.json` with everything above + run config + ladder
   provenance.
6. Optional matplotlib plots if `matplotlib` is importable.

## Sensitivity computation (robust, dual-method)

Both methods computed and logged side-by-side; the perturbation
column is primary for V_kin selection because it is a local per-V
estimate independent of V_RHE grid spacing. FD is the cross-check.

| Method | What it does | Cost |
|---|---|---|
| **FD (secondary)** | `dR_net/dσ_S(V_i) ≈ (R_net(V_{i+1}) - R_net(V_{i-1})) / (σ_S(V_{i+1}) - σ_S(V_{i-1}))` | Free (post-process). |
| **Perturbation (primary)** | At each V, re-solve with `C_S → C_S · (1+ε)` and `C_S · (1-ε)` (ε=0.05), warm-restarted from the unperturbed converged U. Compute `dR_net/dσ_S` from the local pair. | ~2× wall on the λ=1 pass. |

If the two columns disagree by more than 50% relative at a given V,
flag that V in the JSON and **exclude it from V_kin candidates** —
that's a noisy spot where the cap is mid-saturation and the V_kin
rule should not anchor there.

## V_kin selection rule (locked, NOT under critique)

[ELIDED — locked verbatim from acceptance bundle. See full plan file.]

## Driver layout — single new script

**File:** `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` (new).

[Full code skeleton elided — see plan file. The structure is:
solve_anchor_with_continuation at V=+0.55 (kw_eff_ladder, λ=0) →
solve_grid_with_anchor warm-walk → per-V solve_lambda_ramp_from_warm_start
for the λ ramp → optional ±5% C_S perturbation second solve →
post-process FD column → select_v_kin() applies the locked rule.]

## Levich limit (under critique)

```python
def _i_lim_4e_mA_cm2(l_eff_m):
    """I_lim_4e_phys = 4 · F · D_O2 · c_O2 / l_eff_m (mA/cm²).
    F = 96485 C/mol, D_O2 = 2.18e-9 m²/s [WRONG: codebase = 1.9e-9],
    c_O2 = 1.2 mol/m³ at L_REF.
    """
    F = 96485.3329
    D_O2 = 2.18e-9    # ← TYPO. Should be 1.9e-9 from _bv_common.D_O2.
    c_O2 = 1.2
    return 4 * F * D_O2 * c_O2 / l_eff_m * 0.1     # A/m² → mA/cm²
```

[Rest of plan elided — out of scope.]

## 3. Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

**Scope of this review (narrow):** only the three subtle points
called out in the context bundle:
  1. Perturbation column physics (chain rule / total vs partial
     derivative interpretation).
  2. 50 % FD/perturbation disagreement threshold (defensible value
     or better criterion).
  3. Levich cross-check (formula, factor of 0.1, parallel-2e/4e
     interpretation of the |cd|/I_lim ratio, D_O2 = 1.9e-9 vs
     2.18e-9 typo confirmation).

Do NOT critique:
  - The locked V_kin selection rule (acceptance bundle).
  - Driver structure (solve_grid_with_anchor + lambda_ramp pattern).
  - Test layout / V_RHE grid resolution.
  - The v10a Langmuir cap residual / Picard formula (already
     reviewed in session 31 and landed).

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
