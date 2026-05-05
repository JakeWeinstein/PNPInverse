# 4sp Bikerman-corrected IC — handoff for GPT review

> **Status (2026-05-04):** Superseded by
> `docs/4sp_bikerman_ic_option_2a_plan.md` (execution plan, with
> corrections applied) and `docs/4sp_drop_boltzmann_investigation.md` §12
> (resolution).  This handoff contained two errors caught by GPT review
> in `docs/4sp_bikerman_ic_gpt_assessment.md`:
> (1) §1.2 misstated the existing `u_3` seed as `ln(c_ClO₄_bulk) + ψ`;
> the actual code seeds `u_3 = ln(H_outer) + ψ`.
> (2) §8 used constant bulk anchors in γ; local `H_outer` anchors are
> the correct choice (matched-asymptotic structure, ~4× wider packing
> margin).  Open question 3 ("does φ pick up ln γ?") was answered No
> — `θ = θ_bulk · γ` already pairs γ in `c_i` with `−γ` in `−ln(θ)`,
> they cancel in the chemical potential.  Read the plan and resolution
> for the as-implemented final form.

**Date:** 2026-05-04
**Audience:** GPT, picking this up cold.
**Goal:** critique the specific IC modification proposed below. Catch math
mistakes, identify weaker constraints than the ones cited, and either
endorse Option 2a′ or recommend a better alternative.

The repo is a Firedrake-based Poisson–Nernst–Planck / Butler–Volmer
forward solver for a two-step ORR. The 4-species dynamic preset tracks
O₂, H₂O₂, H⁺, ClO₄⁻ as primary variables with log-concentration
transform `u_i = ln c_i`. ClO₄⁻ has `z=-1` and is the only counterion;
all four species share `a_j = 0.01` (nondim Bikerman lattice site
size); bulk concentrations are
`c_O₂_bulk = 1.0`, `c_H₂O₂_bulk ≈ 1e-4`, `c_H_bulk = c_ClO₄_bulk = 0.2`.

## 1. State of the system

### 1.1 Residual — sign correction has been applied

The modified Nernst–Planck flux in `Forward/bv_solver/forms_logc.py:266`
now reads:

```python
mu_steric = -fd.ln(packing)        # was +fd.ln(packing) — wrong sign
Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift) + fd.grad(mu_steric))
```

with `packing = max(1 − Σ_j a_j·c_j, packing_floor)` and
`packing_floor = 1e-8`. This is the Borukhov-Andelman-Orland (1997)
eq (3) / Bazant-Kilic-Storey-Ajdari (2009) eq (20) convention. The
electrochemical potential is

```
μ_i = ln(c_i) + z_i·ψ − ln(θ),    θ := 1 − Σ_j a_j c_j.
```

Zero-flux equilibrium `∇μ_i = 0` therefore gives, for any species i:

```
c_i · θ⁻¹ · exp(z_i·ψ) = const_i
```

with const_i fixed by bulk values.

### 1.2 IC — currently a pure-Boltzmann seed

`_try_debye_boltzmann_ic` (forms_logc.py:571–807) builds:

* a Picard outer-loop estimate of `O_s, P_s, H_o, ψ_D` driven by
  drift–diffusion–reaction balance (no steric);
* a Gouy–Chapman ψ(y) profile,
  `ψ(y) = 4·atanh(tanh(ψ_D/4)·exp(−y/λ_D))`;
* outer linear envelopes `O_outer(y), P_outer(y), H_outer(y)` from the
  picard surface values to bulk;
* species seeds:
  * `u_0 = ln(O_outer)` (linear)
  * `u_1 = ln(P_outer)` (linear)
  * `u_2 = ln(H_outer) − ψ` (Boltzmann × outer envelope, z=+1)
  * `u_3 = ln(c_ClO₄_bulk) + ψ` (Boltzmann, z=−1)
  * `u_n = ln(H_outer / c_ClO₄_bulk) + ψ` (φ on the matched-asymptotic
    profile, internally consistent with the H⁺ seed).

### 1.3 Failure mode under the new (sign-corrected) residual

At V_RHE = +0.3 V, ψ_D ≈ 11.5. The pure-Boltzmann IC yields
`c_ClO₄(surface) ≈ 0.2·exp(11.5) ≈ 2·10⁴`. With the sign-corrected
residual:

* `Σ a_j c_j ≈ a · c_ClO₄ ≈ 200`,
* `packing → packing_floor = 1e-8`,
* `μ_steric = −ln(1e-8) ≈ +18.4` (large positive — repulsive, as
  expected post sign fix),
* `∂μ_steric / ∂c_ClO₄ = a / packing ≈ 1·10⁶`.

Newton evaluates a near-singular Jacobian at the very first residual
build; the direct z=1 SS solve diverges; the orchestrator falls back
to linear_phi z-ramp; the warm-walk reaches the same partial-z it
would have reached without any IC at all. The sign fix is necessary
for a positive-packing SS to *exist* at this voltage, but it does
nothing to bring the *initial* state into the Newton basin.

This is observed in `StudyResults/peroxide_window_4sp_extended_*`
and is the prediction that motivates the IC change discussed below.

## 2. The proposal under review (Option 2a, from `4sp_drop_boltzmann_investigation.md` §8)

```python
# 5-line patch in _try_debye_boltzmann_ic, replacing the u_3 seeding:
gamma = fd.Constant(1.0) / (
    fd.Constant(1.0)
    + fd.Constant(2.0 * a * c_ClO4_bulk) * (fd.cosh(psi) - fd.Constant(1.0))
)
U_prev.sub(3).interpolate(
    fd.ln(fd.Constant(c_ClO4_bulk) * gamma) + psi          # i.e. ln(c_bulk · γ · exp(+ψ))
)
# u_0, u_1, u_2 unchanged
```

Apply γ-correction to ClO₄⁻ only. Leave O₂, H₂O₂, H⁺ on their existing
profiles. Keep the standard Gouy–Chapman ψ(y).

## 3. Math review

### 3.1 The γ formula is correct for a binary z=±1 system

For a binary symmetric electrolyte (only H⁺ and ClO₄⁻ in the lattice
gas, equal bulk concentrations, equal a), the Borukhov 1997 eq (5)
saturating distribution gives, after rearrangement,

```
c_+(ψ) = c_bulk · exp(−ψ) / [1 + 2·a·c_bulk · (cosh ψ − 1)]
       =: c_bulk · γ_bin(ψ) · exp(−ψ).
```

This `γ_bin` matches the formula in the proposal. ✓

### 3.2 Multispecies generalisation (the relevant case here)

Solve the joint zero-flux conditions
`∇[ln(c_i) + z_i ψ − ln θ] = 0` for all i, with bulk anchors
`c_i(0) = c_bulk_i`, `θ(0) = θ_bulk = 1 − Σ_j a_j c_bulk_j`.

Derivation. From `c_i · θ⁻¹ · exp(z_i ψ) = c_bulk_i / θ_bulk`,

```
c_i(ψ) = (c_bulk_i / θ_bulk) · θ(ψ) · exp(−z_i ψ).             (1)
```

Substitute into `Σ_j a_j c_j(ψ) = 1 − θ(ψ)`:

```
1 − θ = Σ_j a_j · (c_bulk_j / θ_bulk) · θ · exp(−z_j ψ)
      = (θ / θ_bulk) · S(ψ),         S(ψ) := Σ_j a_j c_bulk_j exp(−z_j ψ).
```

Solve for θ(ψ):

```
θ(ψ) = θ_bulk / (θ_bulk + S(ψ)).                              (2)
```

Define

```
γ(ψ) := 1 / (θ_bulk + S(ψ))
      = 1 / (1 + Σ_j a_j c_bulk_j · (exp(−z_j ψ) − 1)).        (3)
```

Then (1) becomes

```
c_i(ψ) = c_bulk_i · γ(ψ) · exp(−z_i ψ).                       (4)
```

This (3)–(4) is the multispecies Bikerman/Kornyshev γ; specialising
to a binary z=±1 system reproduces `γ_bin` from §3.1. ✓

### 3.3 Important consequence: z=0 species do appear in θ_bulk but not in S(ψ)

For neutral species `exp(−0·ψ) = 1`, so each z=0 contribution to the
sum in (3) is `a_j c_bulk_j · (1 − 1) = 0`. **Algebraically, γ(ψ) for
our 4sp setup is identical to the binary-symmetric formula in the
proposal, even though the lattice contains O₂ and H₂O₂.**

What is *not* the same is the species concentrations: by (4), z=0
species also follow γ —

```
c_O₂(ψ)   = c_O₂_bulk   · γ(ψ),          (z=0)
c_H₂O₂(ψ) = c_H₂O₂_bulk · γ(ψ),          (z=0)
c_H(ψ)    = c_H_bulk    · γ(ψ) · exp(−ψ),
c_ClO₄(ψ) = c_ClO₄_bulk · γ(ψ) · exp(+ψ).
```

Saturation limit at ψ → ∞: `S → a c_ClO₄_bulk · e^ψ`,
`γ → 1/S`, so `c_ClO₄ → 1/a` and every other species → 0. Total
`Σ a c → 1` from below. Strictly positive packing at all finite ψ.

### 3.4 Diagnosis of Option 2a as written — concrete numbers at V=+0.3

Using `ψ_D = 11.5`, `a = 0.01`, `c_O₂_bulk = 1.0`,
`c_H_bulk = c_ClO₄_bulk = 0.2`:

```
exp(11.5)        ≈ 9.87 × 10⁴
S(ψ_D)           ≈ a·1·1 + 0 + a·0.2·exp(−11.5) + a·0.2·exp(+11.5)
                ≈ 0.01 + 0 + 2·10⁻⁸ + 197.4
                ≈ 197.4
θ_bulk           ≈ 1 − a·(1 + 1e-4 + 0.2 + 0.2) ≈ 0.986
γ(ψ_D)           ≈ 1/(0.986 + 197.4) ≈ 5.04 × 10⁻³
c_ClO₄(surface)  ≈ 0.2 · γ · e^{+ψ} ≈ 99.5
```

Option 2a γ-corrects only `u_3`, so `c_ClO₄(surface) ≈ 99.5` (the
intended cap). But:

```
a · c_O₂(surface)   ≈ 0.01 · O_s     ≈ 0.01 · 0.7   ≈ 7.0 × 10⁻³
a · c_ClO₄(surface) ≈ 0.01 · 99.5                  ≈ 0.995
                                                      --------
Σ a_j c_j(surface)                                  ≈ 1.002
1 − Σ                                              ≈ −2 × 10⁻³.
```

`packing_floor` clamps to 1e-8, `μ_steric ≈ +18.4`, gradient ≈ 1e+6.
**Same Newton pathology as the pre-sign-fix case, just shifted by
~1% in occupancy.** The 1% margin is real and binding because, by
(4), `c_ClO₄ → 1/a` is the *unique* zero-flux saturation under the
implemented residual; you cannot keep ClO₄⁻ on its zero-flux manifold
*and* leave room for the O₂ that the linear profile leaves at the
electrode.

### 3.5 Diagnosis of "Option 2a′" — γ-correct every species

Replace u_0…u_3 by

```python
log_gamma = fd.ln(gamma_psi)        # gamma_psi as in (3)
U_prev.sub(0).interpolate(fd.ln(O_outer) + log_gamma)
U_prev.sub(1).interpolate(fd.ln(P_outer) + log_gamma)
U_prev.sub(2).interpolate(fd.ln(H_outer) - psi + log_gamma)
U_prev.sub(3).interpolate(
    fd.Constant(math.log(c_ClO4_bulk)) + psi + log_gamma
)
```

i.e. `c_i(y) = c_outer_i(y) · γ(ψ(y)) · exp(−z_i ψ(y))` for every
species, where the picard `c_outer_i(y)` are kept for O₂/H₂O₂/H⁺ (so
the macroscopic transport–reaction state is unchanged at y ≫ λ_D,
where γ → 1) and `c_outer_3 = c_ClO₄_bulk` is constant (no reactive
sink for ClO₄⁻).

Surface budget at V=+0.3, with γ ≈ 5×10⁻³:

```
a · c_O₂(0)   ≈ 0.01 · O_s     · γ ≈ 3.5 × 10⁻⁵
a · c_H₂O₂(0) ≈ 0.01 · P_s     · γ ≈ ~0
a · c_H(0)    ≈ 0.01 · H_o     · γ · e^{−ψ_D} ≈ ~10⁻⁹
a · c_ClO₄(0) ≈ 0.01 · c_bulk  · γ · e^{+ψ_D} ≈ 0.995
                                              --------
Σ a c                                          ≈ 0.995
1 − Σ (= θ(ψ_D))                               ≈ 5 × 10⁻³ = γ·θ_bulk.
```

Strictly positive, matches `θ(ψ)` from (2), Jacobian entries are
O(1). Newton-healthy at the IC.

At y ≫ λ_D, ψ → 0, γ → 1, so all four species recover their picard
`c_outer_i(y)` profile and the macroscopic outer state is preserved.

## 4. What 2a′ does *not* fix

The Gouy–Chapman ψ(y) inside the EDL is wrong by Bikerman: the true
first integral of the modified Poisson equation is

```
(dψ/dy)² = (2 / (ν λ_D²)) · ln(1 + ν · (cosh ψ − 1)),    ν = 2 a c_bulk,
```

which softens dψ/dy in the saturated zone (the counterion can't pile
up further, so ψ falls more slowly). 2a′ uses the standard GC
`4·atanh(tanh(ψ_D/4)·exp(−y/λ_D))`, so the IC's ψ is correct at
y=0 (= ψ_D) and y=∞ (= 0) but not at intermediate y. This means c_i
inside the EDL deviates from the true SS by a multiplicative factor
of γ(ψ_GC(y)) / γ(ψ_Bik(y)).

Empirically (the §8 conjecture, untested) this is OK for V around
+0.3–0.5, may be the next blocker around +0.5–0.7, and is plausibly
the limiter at +1.0. Recovery options if 2a′ stalls:

* **2b** — composite asymptotic ψ(y): two-zone closed form with
  saturated linear-decay zone and outer GC zone, matched at
  ψ_sat ≈ ln(2/ν).
* **2c** — numerical integration of dψ/dy via `solve_ivp`, then
  interpolation onto FE nodes.

I'd not preempt 2b/2c; do 2a′ first, see how far the warm-walk gets,
escalate based on the failure voltage.

## 5. Test changes

The current test suite (`tests/test_initializer_debye_boltzmann_4sp.py`)
asserts `c_ClO₄(y) = c_bulk · exp(+ψ(y))` to FE tolerance — pure
Boltzmann. Replace with:

1. **Bulk recovery**: at y = 1, `|c_i(y) − c_bulk_i| < FE_tol` for all
   i (γ → 1, ψ → 0).
2. **Total packing positive**: at every node,
   `1 − Σ_j a_j c_j(y) > margin` with margin ~ 10⁻³.
3. **Borukhov match**: pose a binary z=±1 sub-problem (drop reactions,
   drop neutral species) and check `c_+(y), c_-(y)` against the
   Borukhov 1997 eq (5) closed form within FE tolerance.
4. **Saturation visible at high ψ_D**: at V=+0.3, surface
   `c_ClO₄(0) ∈ [50, 1/a)`.
5. **3sp+Boltzmann regression unchanged**: existing
   `TestRegression3spStillWorks::test_3sp_still_fires` continues
   byte-identical (γ branch is gated on `synthesised_4sp_counterion`).

## 6. Interaction with the steric_sign_correction_plan

That plan (`docs/steric_sign_correction_plan.md`) §5 anticipates a
sweep S1 in which the post-sign-fix solver, with the *existing*
`debye_boltzmann` initializer, converges at V=+0.3 and saturating
c_ClO₄. **That expectation is too optimistic**: the existing
initializer seeds c_ClO₄ ≈ 2·10⁴ at the surface, which the sign-
corrected residual rejects through packing_floor, exactly as the
pre-fix residual rejected it through the bare logarithm. The sign
fix and the IC fix are independent and *both* required.

Suggested updated sequencing:

1. Sign fix — done.
2. Implement 2a′ (this doc).
3. Update the 4sp IC tests per §5.
4. Re-run `peroxide_window_4sp_extended.py debye_boltzmann`. Expected:
   convergence at V ∈ {+0.3, +0.5}, surface c_ClO₄ saturating just
   below 1/a.
5. If V ≥ +0.7 still partial — escalate to 2b.

## 7. Open questions for GPT

1. **Is the multispecies γ derivation in §3.2 right?** I derived it
   directly from `∇μ_i = 0` for the implemented residual. I'd value
   an independent rederivation; please flag any algebra slip.

2. **Is "γ on every species" actually the best IC, or is it putting
   O₂/H₂O₂ on a manifold that's far from the BV-reaction-balanced
   SS?** The picard outer values for O₂, H₂O₂ encode the
   reaction–transport balance, not zero-flux. Multiplying by γ at
   y=0 reduces O₂ at the surface from `O_s ≈ 0.7` to `O_s · γ ≈
   0.0035`. That's well below the picard prediction but well *above*
   the steric-limited maximum, and it makes the steric residual
   happy. Is there a better hybrid that respects both balances?
   (E.g.: γ-correct only the species whose `a_j c_j` would otherwise
   push total packing below a margin; leave others on linear?)

3. **Stern compatibility.** The 4sp + Stern stack adds a Robin BC at
   the electrode that imposes a finite voltage drop in a Stern
   layer. The IC's φ is currently
   `ln(H_outer/c_ClO₄_bulk) + ψ`. With γ on H⁺, should φ also pick
   up `ln γ` to remain internally consistent? My instinct says yes —
   the φ profile is built from the H⁺ profile, and if H⁺ now carries
   a `+ ln γ` term, φ should too. But I'd like a double-check.

4. **Is there any voltage where the binary-symmetric γ is *worse*
   than pure Boltzmann?** I expect not — the binary γ only attenuates,
   never amplifies — but the proposal note in §8 of
   `4sp_drop_boltzmann_investigation.md` had a vague worry about
   "γ correction too aggressive, c_ClO₄ < ~50". That's a calibration
   question against the multispecies-Bikerman analytical SS; my
   numbers above suggest 2a′ saturates just below 1/a (the cap), and
   "below 50" would only happen at much smaller a·c_bulk than what
   we use. I'd appreciate a sanity check.

5. **Does the FE interpolation introduce a spurious negative-packing
   risk?** The interpolation of `ln(c_outer · γ) − z_i ψ` onto a
   piecewise-linear FE basis is not pointwise-equal to the symbolic
   expression. Near the electrode, where ψ has steep gradients, the
   nodal interpolant may locally violate `Σ a_j c_j ≤ 1` even when
   the symbolic IC satisfies it. The standard answer is to refine
   the mesh near y=0 (we already do — `Ny=200` with a clustered
   grid). Worth checking explicitly with a node-level packing
   assertion in the IC tests.

## 8. Minimal code diff (for reference / orientation)

In `Forward/bv_solver/forms_logc.py`, the relevant block is the
existing IC body around lines 793–806:

```python
# CURRENT (pure Boltzmann × outer envelope)
U_prev.sub(0).interpolate(fd.ln(O_outer))
U_prev.sub(1).interpolate(fd.ln(P_outer))
U_prev.sub(2).interpolate(fd.ln(H_outer) - psi)
phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi
U_prev.sub(n).interpolate(phi_init_expr)
if synthesised_4sp_counterion:
    U_prev.sub(3).interpolate(
        fd.Constant(math.log(c_clo4_bulk)) + phi_init_expr
    )
```

becomes (Option 2a′):

```python
# Multispecies Bikerman γ — matches the sign-corrected residual's
# zero-flux equilibrium on a domain where every species carries a
# nonzero a_j.
a_steric  = float(a_vals_list[3])
c_h_b     = c0_model[2]
gamma_psi = fd.Constant(1.0) / (
    fd.Constant(1.0)
    + fd.Constant(a_steric * c_h_b)        * (fd.exp(-psi) - fd.Constant(1.0))
    + fd.Constant(a_steric * c_clo4_bulk)  * (fd.exp(+psi) - fd.Constant(1.0))
)
log_gamma = fd.ln(gamma_psi)

U_prev.sub(0).interpolate(fd.ln(O_outer) + log_gamma)
U_prev.sub(1).interpolate(fd.ln(P_outer) + log_gamma)
U_prev.sub(2).interpolate(fd.ln(H_outer) - psi + log_gamma)
phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi
U_prev.sub(n).interpolate(phi_init_expr)
if synthesised_4sp_counterion:
    U_prev.sub(3).interpolate(
        fd.Constant(math.log(c_clo4_bulk)) + psi + log_gamma
    )
```

Branch is already gated by `if synthesised_4sp_counterion` for the
ClO₄⁻ seed; the changes to u_0/u_1/u_2 should also be gated so the
3sp+Boltzmann path (where γ would be defined with `c_ClO₄_bulk` from
the analytic counterion config but the residual has no Bikerman
term) is byte-identical to the current behaviour. One possibility:
move the entire `log_gamma` block inside `if synthesised_4sp_counterion`
and also gate the u_0/u_1/u_2 modifications on it. Cleaner: introduce
a local boolean `apply_steric_ic = synthesised_4sp_counterion or
(steric_active and any(a_j > 0))` and gate everything on that.

## 9. Pointers

* Investigation log: `docs/4sp_drop_boltzmann_investigation.md`
  (failure mode under pre-sign-fix code, §8 = original Option 2a/2b/2c).
* External review: `docs/4sp_bikerman_corrected_ic_review.md`
  (the "1/a cap is not enough, you need total-packing cap" critique).
* Sign fix: `docs/steric_sign_correction_plan.md`; code at
  `Forward/bv_solver/forms_logc.py:266`.
* Variational derivation: `docs/steric_sign_correction_plan.md` §1
  (variational derivative of Borukhov entropy density).
* Literature:
  * Borukhov, Andelman, Orland (1997) *PRL* 79, 435 (eqs 2, 3, 5).
  * Bazant, Kilic, Storey, Ajdari (2009) *Adv. Colloid Interface
    Sci.* 152, 48 (eqs 20–22).
  * Kornyshev (2007) *J. Phys. Chem. B* 111, 5545.
