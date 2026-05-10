# Round 5 — counterreply on Phase 6α plan

## 1. Acknowledgment

### Issue 1 — Gate 3 wrong; water isn't a current cap: **Accept**

Right. Total H⁺ supply at the electrode = bulk Levich diffusion +
distributed water source. At L_eff=16 µm, Levich alone is 0.56
mA/cm², already over the water-only ceiling of 0.22 mA/cm². So
adding water cannot CAP cd at 0.22 — it adds to the supply or it
self-limits via the local equilibrium constraint. Removing Gate 3.

The acceptance gates collapse to 5:
1. **P3 surface pH**: max_surface_pH < 9 at L_eff = 16 µm.
2. **Plateau direction-of-change at L=100 µm**: cd at deepest
   cathodic V_RHE moves toward deck -0.18 (direction check).
3. **E conservation per V_RHE**:
   `|cd_solver - cd_E_balance| / |cd_solver| < 1e-3`.
4. **Fast-water validity per V_RHE**:
   `max |R_w,req / (k_r·Kw)| < 0.1` over domain.
5. **Yash cross-check** at L=100 µm, V_RHE=-0.40 V: c_OH(y)
   profile within 1 OOM at every node, within 50% relative error
   at the OHP.

Gate 4 is now the primary gate for "Option C is the right
reduction" (per #7 below); if it fails, Option D is required.

### Issue 2 — Finite-rate validity needs nondim-to-physical: **Accept**

Explicit conversion. The solver works in nondim with the existing
nondim recipe `Nondim/transform.py`:

```
R_w,req has units of [c]/[t] = mol/m³ / s in physical
Nondim: R_hat = R_phys · L_REF² / (D_REF · C_SCALE)
       = R_phys / (D_REF · C_SCALE / L_REF²)
       = R_phys / (1.9e-9 · 1.2 / (1e-4)²)
       = R_phys / 0.228 (mol/m³/s)
```

So:
```
R_phys = R_hat · 0.228 (mol/m³/s)
ε     = R_phys / (k_r · Kw_phys)
       = R_hat · 0.228 / 1.4
       = R_hat · 0.163
```

Acceptance gate stated in nondim:
```
max |R_w,req_hat| · 0.163 < 0.1
⟺ max |R_w,req_hat| < 0.614
```

I'll bake the conversion factor into the post-solve scorer rather
than asking the user to remember it.

### Issue 3 — Sulfate transport off by 10×: **Accept**

Recomputing carefully:

```
J_HSO4 [mol/m²/s] = D · c_HSO4_bulk / L_eff
                  = 1e-9 (m²/s) · 1 (mol/m³) / L_eff (m)

i_HSO4 [mA/cm²]   = J_HSO4 · F · 0.1
                  = 1e-9 · 1 · 96485 · 0.1 / L_eff
                  = 9.65e-6 / L_eff (m)
                  = 0.0965 (L_eff/100 µm)⁻¹ mA/cm²
```

Corrected sulfate table:

| L_eff | sulfate transport ceiling | water source ceiling |
|------:|--------------------------:|---------------------:|
| 100 µm | 0.0965 mA/cm² | 1.35 mA/cm² |
| 66 µm  | 0.146         | 0.89        |
| 21 µm  | 0.460         | 0.28        |
| 16 µm  | 0.603         | 0.22        |

So at L_eff ≥ 21 µm, sulfate transport ≥ water source. At small
L_eff, sulfate dominates. The deferral rationale needs revising:

**Updated sulfate deferral**: at L_eff = 100 µm (Yash-comparison
gate), water source (1.35) >> sulfate transport (0.0965), so water
is dominant for the surface-pH-lift goal. At L_eff = 16 µm, sulfate
transport (0.60) is comparable to deck peak (0.40); if the model
needs to recover the deck PEAK at small L_eff, sulfate becomes
necessary. Phase 6α targets L_eff = 100 µm + plateau, where water
alone suffices. Phase 6β picks up sulfate when targeting the deck
peak at small L_eff.

### Issue 4 — Water-aware Picard transport: **Accept (defer with explicit label)**

You're right that just adding OH⁻ to charge density isn't enough
— the H⁺ profile in Picard is governed by the new E-flux form,
not the old ambipolar 2D_H proton transport. Two implementation
paths:

**Path A** (heaviest, most correct): Rewrite
`picard_outer_loop_general` to solve the proton-condition equation
for H⁺ during the outer loop. Significant rewrite; affects all
existing IC paths.

**Path B** (lighter, accepted as "approximate"): Keep the existing
Picard loop unchanged (solves old ambipolar transport) and label
the IC explicitly as "approximate — does not satisfy proton-
condition flux balance". Rely on Newton + Kw_eff continuation to
relax the IC into the true E-equation steady state.

I'm going with **Path B** for Phase 6α. Justification:
- The IC's job is to give Newton a starting point on the same
  side of the basin of attraction. Approximate IC + 5-rung Kw_eff
  continuation should be enough.
- Path A doubles the scope of Phase 6α and has its own
  convergence risks (rewriting the Picard surface-rate algebra
  for the E-equation).
- If Path B fails (anchor doesn't converge through continuation),
  Path A becomes Phase 6α.5.

I'll add an explicit "IC is approximate" disclaimer in §5 Q3 and
make the proton-condition residual norm at IC the primary
diagnostic during continuation.

### Issue 5 — `R_w,req` pointwise noise: **Accept**

Computing ∇·J_H from CG fields requires second derivatives. Two
robust approaches:

**Option (a)**: Weak-residual reconstruction. The H⁺ NP residual
in weak form:
```
F_H(v) = ∫ ∂c_H/∂t · v - ∫ J_H · ∇v + ∫ J_H · n · v ds
```
For the Option-C model, this residual is exactly the implied water
source `R_w,req` integrated against test functions:
```
F_H(v) = ∫ R_w,req · v dx
```
We can extract a CG projection of `R_w,req` by solving a mass
matrix:
```
∫ M · R_w,req · v = F_H(v)  ∀v
```
This is well-defined and avoids second derivatives.

**Option (b)**: DG projection of `∇·J_H` cellwise. Element-by-element
weak divergence; numerically cleaner near boundaries.

I'll go with **(a)** since it's cheaper (one mass solve) and
matches the existing residual machinery. Reporting both
`max |R_w,req_hat|` and `||R_w,req||_L²` (RMS) for sanity.

### Issue 6 — Purge stale water-rate constants: **Accept**

You're right; I had two contradictory values in flight. The
correct rate, derived from `k_r = 1.4e11 M⁻¹·s⁻¹` and `Kw = 1e-14
M²`:

```
k_r · Kw = 1.4e-3 M·s⁻¹ = 1.4 mol/m³·s
```

(With units check: M⁻¹·s⁻¹ · M² = M·s⁻¹, then 1 M = 1000 mol/m³,
so 1.4e-3 M·s⁻¹ = 1.4 mol/m³·s.)

The earlier `0.078 mol/m³·s` came from an erroneous attempt to
factor out the H₂O activity (k_f · [H₂O]); the rate `k_r · Kw`
already absorbs the water-activity factor through detailed
balance.

Purging from §1.5 of the revised plan. One number throughout:
`R_w,max_phys = 1.4 mol/m³·s`.

### Issue 7 — Validate reduced BC against full OH⁻ no-flux: **Accept**

Adding to Yash cross-check (gate 5). The reduced E equation at the
electrode imposes `J_E·n = J_H,BV·n`, which leaves J_OH·n
implicit (recovered from c_OH = Kw/c_H gradient at the wall).
Reconstruct from the converged solution:

```
J_OH·n_inferred = -D_OH · ∂c_OH/∂y |_{y=0} + D_OH · c_OH · ∂φ/∂y |_{y=0}
                = D_OH · (Kw/c_H²) · ∂c_H/∂y - D_OH · (Kw/c_H) · ∂φ/∂y
```

The full-dynamic Yash run should report J_OH·n ≈ 0 at the electrode
(no Faradaic OH⁻ flux for acidic ORR). If our reconstructed
J_OH·n_inferred is comparable to J_H,BV at the surface,  Option C
hides a boundary-layer artifact and we need Option D.

Acceptance threshold: `|J_OH·n_inferred| / |J_H,BV·n| < 0.05` (5%
of the H⁺ flux). Otherwise reduction is invalid.

This goes into Gate 5 (Yash cross-check) as a sub-criterion.

### Issue 8 — MMS forcing sign matches weak form: **Accept**

Explicit derivation. The forced weak residual we're testing is:

```
∫ v · ∂E/∂t dx  -  ∫ ∇v · J_E dx  +  ∫ v · J_E · n  ds
   =  ∫ v · s(y) dx                           (interior forcing)
   +  ∫_top    v · g_top  ds                  (boundary forcing top)
   +  ∫_elec   v · g_elec ds                  (boundary forcing electrode)
```

For the manufactured solution to satisfy this:
```
s(y) = ∂E/∂t + ∇·J_E       (interior, pointwise; from manufactured fields)
g_top = J_E·n |_top         (Dirichlet on c_H, boundary flux follows)
g_elec = J_E·n |_elec       (matches the manufactured c_H near surface)
```

Steady-state version (∂E/∂t = 0):
```
s(y) = ∇·J_E [from manufactured u_H, φ, c_OH = Kw_hat/c_H]
```

For the smooth manufactured fields:
```
u_H(y) = α + β · exp(-y/λ) + γ · y       (smooth, gives c_H = exp(u_H))
φ(y)   = δ · (1 - y) + η · sin(πy)
c_OH(y)= Kw_hat / c_H(y) = Kw_hat · exp(-u_H(y))
```

Then evaluate symbolically (or via SymPy) to get `s(y)` matching
the weak residual sign convention. The MMS test runs the solver
with this `s(y)` as an explicit forcing in the residual form
above; the converged u_H should match the manufactured u_H to
within `O(h^{p+1})` for CG-p elements.

Documenting this protocol in §5 Q4 with an explicit reference to
the weak-form sign convention from Issue 3.

## 2. Updated artifact summary

The plan now has the following coherent acceptance structure:

**Gates** (5):
1. Surface pH: max_surface_pH < 9 at L_eff = 16 µm.
2. Plateau direction at L=100 µm: cd more negative toward deck -0.18.
3. E conservation per V_RHE: |cd_solver - cd_E_balance|/|cd_solver| < 1e-3.
4. Fast-water validity (primary gate for Option C correctness):
   `max |R_w,req_hat| · 0.163 < 0.1` per (L_eff, V_RHE).
5. Yash cross-check at gated condition + reduced-BC sub-check
   `|J_OH·n_inferred| / |J_H,BV·n| < 0.05`.

**Implementation**:
- Constants: `KW_MOLAR_SQUARED = 1e-14`, `KW_HAT = Kw_phys / C_SCALE²`,
  `D_OH_HAT`, `A_OH_HAT`, all derived from one canonical physical
  baseline.
- Proton-condition flux: `J_E = -(D_H·c_H + D_OH·c_OH) · ∇μ_H` in
  muh, `... · (∇u_H + ∇φ)` in logc.
- Weak form: `∫ v·E_t  - ∫ ∇v·J_E  + ∫ v·J_E·n  = 0`
  with explicit BCs (Dirichlet on c_H at top, J_E·n = J_H,BV·n at
  electrode, no-flux on sides).
- Continuation: 5-rung Kw_eff schedule
  `[0, Kw_hat·1e-6, Kw_hat·1e-3, Kw_hat·0.1, Kw_hat]`.
- IC: existing Picard outer loop (counterion-aware, NOT
  water-aware). Documented as approximate; relies on Newton +
  continuation. Path-B labeled.
- u_H clamping: extend the existing symmetric u_clamp = 100 to
  also bound `c_OH = Kw_hat · exp(-u_H)` (lower bound on u_H,
  not a separate c_OH clamp).
- Tests: MMS with manufactured (u_H, φ) and forcing s(y) per #8.
  Conservative weak-form assembly. Disabled-path numerical
  regression at 1e-10 relative tolerance.

**Risk register** (final):
- R1: Option C fails P3 → Option D (full dynamic OH⁻ + finite-rate R_w).
- R2: Newton stiffness during Kw_eff continuation → tighten ladder.
- R3: c_OH packing dominance at high pH → continuation handles it.
- R4: IC composite-ψ closure with water → labeled approximate;
  Newton + continuation primary.
- R5: Default-off flag preserves existing regression.
- R6: Damköhler validity verified across L_eff range (#1, #2, #5).
- R7: log/muh primary gives well-conditioned flux coefficients (#8).
- R8: Finite water source capacity tracked via Gate 4 nondim
  metric (#2, #5).
- R9 (NEW): Reduced-BC J_OH·n validation against Yash (#7) — if
  inferred J_OH·n at electrode > 5% of J_H,BV, escalate.

**Out of scope**:
- Sulfate buffering (Phase 6β; Option C with water alone covers
  surface-pH-lift goal at L=100 µm).
- HSO₄⁻ as a dynamic species (Phase 6β).
- Cation-dependent selectivity factor (Phase 6γ).
- Alkaline-form ORR pathway (Phase 6δ; would change J_E·n
  electrode BC).
- Activity-Kw model (out of Phase 6 scope; sticking with
  concentration-Kw + sterics-in-Poisson per #12 from R1).

## 3. Continued critique prompt

This is round 5 (the cap per the skill spec). If you verdict APPROVED,
the loop terminates and Claude auto-revises the plan artifact with
the accepted feedback. If ISSUES_REMAIN, the loop terminates anyway
and remaining issues are flagged as unresolved.

Re-evaluate: are the remaining concerns blocking, or are they
nitpicks that fit on a follow-up TODO list? Same numbered format
and same verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
