# Round 4 — counterreply on Phase 6α plan

## 1. Acknowledgment

### Issue 1 — Finite-source arithmetic off by 10×: **Accept**

Embarrassing. Recomputing:

```
i_max [mA/cm²] = R_w_max · L_eff · F · 0.1 (m²→cm² unit factor)
              = 1.4 (mol/m³·s) · L_eff (m) · 96485 (C/mol) · 0.1
              = 1.35 · (L_eff / 100 µm) mA/cm²
```

Corrected headroom table:

| L_eff | i_max water-only | deck plateau (-0.18) | deck peak (-0.40) |
|------:|----------------:|---------------------:|------------------:|
| 100 µm | 1.35 mA/cm² | 7.5× headroom | 3.4× headroom |
| 66 µm  | 0.89        | 5.0×           | 2.2×           |
| 21 µm  | 0.28        | 1.6×           | **insufficient** (0.7×) |
| 16 µm  | 0.22        | 1.2× (very tight) | **insufficient** (0.55×) |

Two material consequences:

(a) **At L_eff ≤ 21 µm, water-only Option C cannot sustain the deck
    peak (-0.40 mA/cm²).** Either the deck peak never emerges in
    our model at small L_eff (consistent with P2 PASS — no peak
    materialized in the L_eff sweep), OR sulfate buffering / Option
    D is required to support it.

(b) **At L_eff = 16 µm, even the deck plateau has only 22% headroom
    above the water-source ceiling.** Comfortable for the verdict's
    "lift cd toward -0.18" goal, but tight — Newton may need
    smaller continuation rungs near the boundary.

I'll record this in §4 of the revised plan as a quantitative
prediction: at L_eff ∈ {21, 16} µm, plateau magnitude should
**decrease** (be less negative) below the current Levich-only
sweep result of -0.428, -0.562 mA/cm², toward the water-source
ceiling 0.28, 0.22 mA/cm² respectively. **This is a falsifiable
direction-of-change prediction**, not just for surface pH.

### Issue 2 — Finite-source metric conceptually wrong: **Accept**

You're right; comparing total cd against R_w_max·L_eff over-counts
because the bulk-side Levich diffusion already supplies most of the
H⁺ flux. The right metric is the local source residual:

```
R_w,req(y) = ∇·J_H(y)   (steady state: H⁺ NP equation residual that
                          water must supply at each point)
```

Acceptance criterion replacing my earlier `required_to_max_ratio`:

```
ε(y) = R_w,req(y) / (k_r · Kw_phys)        # local fast-water validity
       = R_w,req(y) / (1.4 mol/m³·s)       # at standard k_r, Kw

max |ε(y)| over the domain  <  0.1         # 10% departure from
                                              equilibrium → fast-water
                                              limit credible
∫ ε(y) dy / domain_height  <  0.05         # spatial-average
                                              tighter cap
```

This needs to be evaluated post-solve from the converged solution.
Per-V_RHE reporting; flag any combo where threshold is exceeded.

### Issue 3 — Weak-form sign flipped: **Accept**

Correct conservative weak form for `∂E/∂t + ∇·J_E = 0`:

```
∫ v · ∂E/∂t dx  +  ∫ v · ∇·J_E dx  =  0

  ⟹ (integrate second term by parts)

∫ v · ∂E/∂t dx  -  ∫ ∇v · J_E dx  +  ∫ v · J_E · n  ds  =  0
```

Steady-state form:

```
- ∫ ∇v · J_E dx  +  ∫ v · J_E · n  ds  =  0
```

With outward normal at ∂Ω. At the electrode (n = -ŷ), the boundary
term contributes `∫ v · J_E · (-ŷ)  ds`. Specifying J_E·n =
J_H,BV·n at the electrode (BV is positive cathodic-current convention
in our existing code: positive J_H,BV·n = H⁺ flowing INTO electrode
to be consumed, which is `-ŷ` for our geometry, so J_H,BV·n is the
positive surface-flux observable).

I'll align with the existing `_build_bv_observable_form` sign
convention rather than rederiving. The acceptance criterion #10
(current balance) reuses that existing observable.

### Issue 4 — OH⁻ boundary language ambiguity: **Accept**

Clarifying. The reduced E equation imposes ONE BC on J_E·n at the
electrode. We do NOT separately constrain J_OH·n. The wording in
the revised plan:

> Electrode (y=0): `J_E · n = J_H,BV · n` (the existing acidic-form
> BV cathodic flux, with H⁺ Faradaic consumption from R_2e and
> R_4e). The proton-condition variable does not separately specify
> a J_OH boundary flux — that's recovered post-hoc from the
> equilibrium constraint c_OH = Kw_hat / c_H. Total OH⁻ flux at
> the electrode is implied, not imposed.

If we ever add an alkaline-form ORR pathway (Phase 6γ), J_E·n at
the electrode picks up a Faradaic-OH⁻ contribution and the BC
form changes.

### Issue 5 — Finite-rate validity = deviation from equilibrium: **Accept**

This is the right metric and replaces my "required-to-max" check.
Stating it explicitly:

In the finite-rate model, `R_w = k_r · (Kw - c_H · c_OH) =
k_r · Kw · (1 - c_H · c_OH / Kw)`. Option C enforces
`c_H · c_OH / Kw = 1 + δ` where `|δ|` is the fractional
deviation from equilibrium. The required source rate is
`R_w,req = R_w,Option_C` = the source we'd need water to provide
to support the Option-C solution. From the H⁺ equation:
`R_w,req = ∇·J_H + ∂c_H/∂t`, i.e., the divergence of the H⁺
NP flux at steady state.

In Option C with c_OH = Kw/c_H, the implied δ is whatever satisfies
the conservative residual. The validity check is:

```
ε ≡ R_w,req / (k_r · Kw)
```

If `max |ε| < 0.1`, fast-water limit holds (deviation < 10 %). If
`max |ε| > 0.5`, fast-water is breaking down → escalate to Option D
(finite-rate water reaction).

Adopting this as the primary fast-water-validity gate. Adding it
to the post-solve verdict.json schema.

### Issue 6 — Sulfate not "one-shot": **Accept**

Right; the EDL inventory is replenished from bulk by HSO₄⁻
diffusion. Steady-state sustained sulfate buffer flux:

```
J_HSO4_max = D_HSO4 · c_HSO4_bulk / L_eff
           ≈ 1e-9 (m²/s) · 1 (mol/m³) / L_eff
           = 1e-5 / (L_eff/100 µm)  mol/(m²·s)
           ≈ 0.97 / (L_eff/100 µm)  mA/cm² of equivalent H⁺ supply

| L_eff   | J_HSO4_max equivalent |
| 100 µm  | 0.97 mA/cm² |
| 16 µm   | 6.07 mA/cm² |  (much larger than water source!)
```

Wait, that's inverted — at smaller L_eff, the diffusion-driven
sulfate supply is LARGER (because the gradient is steeper), while
the water source is SMALLER (less domain volume to source from).
At L_eff = 16 µm:
- Water source ceiling: 0.22 mA/cm²
- Sulfate transport ceiling: 6.07 mA/cm²

So at small L_eff, sulfate buffering vastly exceeds water capacity
in steady state. This makes sulfate not deferrable for the full
deck-target plan — but our immediate Phase 6α goal (lift surface
pH from 14 to <9) requires only proton supply at deck-plateau
magnitudes, not deck-peak. At L_eff = 100 µm where the comparison
to Yash is cleanest, water (1.35) >> sulfate (0.97), so water
dominates.

**Updated deferral rationale**: sulfate is deferred for Phase 6α
because (a) at L_eff=100 µm (the gating Yash-comparison condition),
water > sulfate; (b) the immediate goal is surface pH lift, which
only needs ~0.18 mA/cm² of supply that water comfortably
provides. Phase 6β picks up sulfate when we want to recover the
deck PEAK, not just the plateau. Adding this comparison to §7
of the revised plan.

### Issue 7 — muh top Dirichlet under-specified: **Accept**

In the muh primary, `μ_H = u_H + em·z_H·φ = ln(c_H/c_ref) + φ`
(em·z_H = +1). At the bulk top:
- `c_H_top = c_H_bulk` (Dirichlet on concentration)
- `φ_top = 0` (ground reference, existing convention)

Therefore `μ_H_top = ln(c_H_bulk / c_ref) + 0 = ln(c_H_bulk_hat)`.

This is what the existing muh top BC machinery does (as long as φ
is grounded at the top). Documenting explicitly in the revised
plan to avoid ambiguity.

### Issue 8 — Current-balance scaling ambiguity: **Accept**

Reusing `_build_bv_observable_form(ctx, mode="current_density")`
for the current-balance check. That function already handles the
ds normalisation and I_SCALE scaling consistently with the cd
output we plot. The acceptance criterion becomes:

```python
cd_solver = build_bv_observable(ctx, mode="current_density")
cd_E_balance = -build_E_flux_at_electrode(ctx)  # NEW helper
assert abs(cd_solver - cd_E_balance) / abs(cd_solver) < 1e-3
```

The new helper `build_E_flux_at_electrode` reuses the existing ds
machinery (same boundary marker, same V_T scaling).

### Issue 9 — IC water-aware Picard primary, not post-hoc: **Accept**

Restructuring the IC plan. The Picard outer loop in
`picard_ic.picard_outer_loop_general` is currently
counterion-aware but H⁺ remains the only dynamic field. To make it
water-aware:

1. Modify the Picard surface-rate equations to include OH⁻ slaving:
   when computing the outer-region phi_o and surface concentrations,
   add `c_OH(y) = Kw_hat / c_H(y)` to the charge density and
   packing-fraction closures.
2. The Picard outer loop's residual now self-consistently solves
   for both c_H_outer profile AND φ profile WITH water equilibrium
   active. The continuation parameter Kw_eff scales the water term
   from 0 to Kw_hat.
3. Post-Picard residual check on the proton-condition equation:
   ||F_E|| / ||u_H|| < 1e-3 before handing off to Newton.

This is a more invasive change to picard_ic.py than I'd estimated.
Tagging as part of Q3 in the revised plan with appropriate scope.

The "post-hoc reconstruction" path I described earlier is now a
fallback (e.g., if water-aware Picard fails to converge during the
ramp), not the primary path.

### Issue 10 — MMS spec incomplete: **Accept**

Full MMS protocol:

```
Manufactured solution:
   c_H(y) = a₀ + a₁·exp(-y/λ) + a₂·y    (smooth, positive)
   φ(y)  = b₀·(1 - y) + b₁·sin(πy)
   c_OH(y) = Kw / c_H(y)                (slaved by definition)

Compute analytically:
   J_E(y) = -[D_H · c_H(y) + D_OH · c_OH(y)] · (∂_y u_H + ∂_y φ)
         where ∂_y u_H = ∂_y c_H / c_H

Plug into ∇·J_E = source(y), get:
   source(y) = ∂_y[ -[D_H·c_H + D_OH·c_OH] · (∂_y u_H + ∂_y φ) ]

Boundary terms:
   J_E·n at top:    J_E(y=L) · ŷ
   J_E·n at electrode: -J_E(y=0) · ŷ  ← imposed BV cathodic flux

Solver call:
   1. Run with manufactured BV-flux BC at y=0 and Dirichlet at y=L,
      and an interior forcing term equal to source(y) added to F_E.
   2. Refine mesh: Ny ∈ {20, 40, 80, 160}.
   3. Verify L²-error in u_H decays as h^(p+1) for CG-p elements.
```

This tests:
- Conservative residual (interior + boundary integrals).
- Flux-coefficient correctness in log/muh form.
- BV-flux BC sign convention (integrate by parts correctly).
- Source/sink term wiring.

I'll add this as the canonical Phase 6α MMS test.

## 2. Updated artifact summary

The plan's quantitative claims are tightened by:
- **Corrected i_max table** (Issue 1) — water-source ceiling
  is 1.35 mA/cm² at 100 µm, dropping to 0.22 at 16 µm.
- **Falsifiable plateau prediction** at L_eff ≤ 21 µm (Issue 1):
  cd magnitude should DECREASE toward the water-supply ceiling.
- **Deviation-from-equilibrium gate** (Issues 2, 5):
  `max |R_w,req / (k_r · Kw)| < 0.1` per (L_eff, V_RHE) combo.
- **Conservative weak form sign** (Issue 3): canonical form
  `∂E/∂t + ∇·J_E = 0` ⟹ `∫ v·E_t − ∫ ∇v·J_E + ∫ v·J_E·n = 0`,
  used consistently throughout.
- **Water-aware Picard IC** (Issue 9) as primary path; post-hoc
  reconstruction as fallback.
- **Full MMS protocol** (Issue 10) with manufactured u_H, φ, c_OH,
  J_E, source(y), boundary fluxes.
- **Sulfate deferral rationale** (Issue 6) updated with sustained
  diffusion-supply comparison instead of "one-shot reservoir".

The plan now has a coherent set of acceptance gates:

1. **P3 surface pH**: max_surface_pH < 9 at L_eff = 16 µm.
2. **Plateau direction-of-change**: at L_eff = 100 µm, cd at deepest
   cathodic V_RHE moves toward deck -0.18 (more negative).
3. **Water-supply self-limiting**: at L_eff ≤ 21 µm, cd magnitude
   decreases toward water-source ceiling (0.28, 0.22 respectively).
4. **E conservation per V_RHE**:
   `|cd_solver - cd_E_balance| / |cd_solver| < 1e-3`.
5. **Fast-water validity per V_RHE**:
   `max |R_w,req / (k_r·Kw)| < 0.1` over domain.
6. **Yash cross-check** at gated test condition (L=100 µm,
   V_RHE=-0.40 V, pH 4, Cs⁺/SO₄²⁻ matched).

If gate 3 or 5 fails → escalate to Option D (finite-rate water +
full dynamic OH⁻).

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
