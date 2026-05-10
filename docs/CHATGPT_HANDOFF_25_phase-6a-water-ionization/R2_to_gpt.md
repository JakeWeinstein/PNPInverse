# Round 2 — counterreply on Phase 6α plan

## 1. Acknowledgment of issues raised in R1

### Issue 1 — Damköhler arithmetic wrong: **Accept**

You're right; I dropped 11 orders of magnitude on `k_r · c_H_bulk`.

Correct units:
- `k_r = 1.4 × 10¹¹ M⁻¹ s⁻¹`, `c_H_bulk = 1 × 10⁻⁴ M`.
- `k_r · c_H_bulk = 1.4 × 10⁷ s⁻¹`. Reverse rate.
- `k_f = k_r · Kw = 1.4 × 10¹¹ · 10⁻¹⁴ M·s⁻¹ = 1.4 × 10⁻³ M·s⁻¹` (zero-order
  in c_H, since H₂O activity ≈ 1).
- `τ_water_eq ≈ 1 / k_r·(c_H + c_OH)` ranges from `~0.7 µs` at bulk
  (c_H ≈ 10⁻⁴ M, c_OH ≈ 10⁻¹⁰ M) to `~7 ns` at pH 14 (both ~1 M).
- `τ_diffusion = L²/D_H = (10⁻⁴)² / 9.3 × 10⁻⁹ ≈ 1.07 s` at L_REF=100 µm;
  `≈ 27 ms` at L_eff=16 µm.
- Da = τ_diffusion / τ_water_eq ranges from `1.5 × 10⁵` (worst case:
  L_eff=16 µm AND bulk c_H, where water rate is slowest) to `3.9 × 10¹⁴`
  (best case: L_eff=100 µm AND high pH).

Da ≫ 1 uniformly across the planned sweep. The conclusion stands; the
arithmetic doesn't.

### Issue 2 — Option D Damköhler missing concentration: **Accept**

Same fix as #1. The corrected per-species Da:
- `Da_H = k_r · c_H · L²/D_H`. At bulk: `1.4e11 · 1e-4 · (1e-4)² / 9.3e-9 ≈ 1.5e7`. At surface (pH 9, c_H = 1e-9 M): `1.5e2`.
- `Da_OH = k_r · c_OH · L²/D_OH`. At bulk: `1.4e11 · 1e-10 · (1e-4)² / 5.27e-9 ≈ 2.7e1`. At surface (pH 9): `2.7e5`.

Both species have Da ≫ 1 in the regions where their respective
concentrations matter. The fast-water assumption is robust.

### Issues 3, 4, 5, 23 — Option B is physically broken: **Accept (large architectural change)**

You're right. Algebraic `c_OH = Kw/c_H` substituted into Poisson does
not source H⁺ — the H⁺ NP equation is unmodified, so c_H still crashes
at the cathode regardless of how OH⁻ contributes to charge density and
steric packing.

Working through the consistency requirement:

1. The OH⁻ NP equation must hold: `∂c_OH/∂t + ∇·J_OH = R_w`.
2. If we enforce `c_OH = Kw/c_H` algebraically, then
   `∂c_OH/∂t = -Kw/c_H² · ∂c_H/∂t` and `R_w` is determined by the
   constraint, not free.
3. Mass conservation forces R_w_H = R_w_OH (water reaction is symmetric).
4. Substituting back into the H⁺ equation:
   ```
   ∂c_H/∂t + ∇·J_H = R_w
   = -Kw/c_H² · ∂c_H/∂t + ∇·J_OH
   ```
   Rearranging:
   ```
   ∂(c_H − Kw/c_H)/∂t + ∇·(J_H − J_OH) = 0
   ⟺  ∂E/∂t + ∇·J_E = 0   (E = c_H − c_OH)
   ```

So the **only physically consistent fast-water limit IS Option C**.
There's no separate cheaper "Option B" that can satisfy the
diagnosis — algebraic-only c_OH in Poisson is not the limit of any
mass-conserving model. It's a phantom shortcut.

I am pivoting the recommendation to make **Option C the first landing**.
Option B is reframed as a diagnostic-only sanity check ("what
happens if we add OH⁻ to Poisson without sourcing H⁺?") that we may
not actually need to run.

### Issue 6 — Option C requires explicit PDE derivation: **Accept**

Working it out properly. Using c_H as primary and c_OH = Kw/c_H as
slaved:

Conservation laws (with em·z_H = +1 in muh scaling, em·z_OH = -1):
```
J_H_hat  = -D_H_hat  · ∇c_H_hat  - c_H_hat · ∇φ_hat
J_OH_hat = -D_OH_hat · ∇c_OH_hat + c_OH_hat · ∇φ_hat
```

Substituting `c_OH_hat = Kw_hat / c_H_hat`:
```
∇c_OH_hat = -Kw_hat / c_H_hat² · ∇c_H_hat
J_OH_hat = D_OH_hat · Kw_hat / c_H_hat² · ∇c_H_hat
         + Kw_hat / c_H_hat · ∇φ_hat
```

The proton-condition equation:
```
∂(c_H_hat - Kw_hat/c_H_hat)/∂t + ∇·(J_H_hat - J_OH_hat) = 0
```

Equivalently in c_H_hat alone, multiplying through by `c_H_hat²`:
```
(c_H_hat² + Kw_hat) · ∂(ln c_H_hat)/∂t  -- via chain rule
   + c_H_hat² · ∇·J_H_hat
   - c_H_hat² · ∇·J_OH_hat
   = 0
```

In log-c primary `u_H = ln(c_H_hat)`, `c_H_hat = exp(u_H)`:
```
[exp(2 u_H) + Kw_hat] · ∂u_H/∂t  
   + ∇·[ -D_H_hat · exp(u_H) · ∇u_H · exp(u_H) - exp(u_H) · ∇φ · exp(u_H) ]
   - ∇·[ D_OH_hat · Kw_hat · exp(-u_H) · ∇u_H · (-1) - Kw_hat · exp(-u_H) · ∇φ ]
   = 0
```

Cleaner: the flux for the proton-condition variable is:
```
J_E_hat = -[D_H_hat · c_H_hat + D_OH_hat · c_OH_hat] · ∇c_H_hat / c_H_hat
        - [c_H_hat + c_OH_hat] · ∇φ_hat
```

Wait, that's not quite right. Let me redo. With c_OH = Kw/c_H:
```
∇c_OH = -Kw/c_H² · ∇c_H
```

So:
```
J_H = -D_H · ∇c_H - c_H · ∇φ                       (z_H = +1)
J_OH = -D_OH · ∇c_OH + c_OH · ∇φ
     = D_OH · Kw/c_H² · ∇c_H + (Kw/c_H) · ∇φ       (z_OH = -1)
```

Then:
```
J_E = J_H - J_OH
    = -D_H · ∇c_H - c_H · ∇φ - D_OH · Kw/c_H² · ∇c_H - (Kw/c_H) · ∇φ
    = -[D_H + D_OH · Kw/c_H²] · ∇c_H - [c_H + Kw/c_H] · ∇φ
```

So the proton-condition flux has:
- An effective diffusivity `D_eff = D_H + D_OH · Kw/c_H²` (water-buffer
  diffusivity).
- An electromigration coefficient `c_H + c_OH = c_H + Kw/c_H`.

Time derivative:
```
∂E/∂t = (1 + Kw/c_H²) · ∂c_H/∂t
```

So in log-c form:
```
[exp(2 u_H) + Kw_hat] · ∂u_H/∂t  +  exp(u_H) · ∇·J_E_hat  = 0
```

Or equivalently, divide through by `c_H + c_OH` to get a normalised
form. The point is: the PDE is in c_H (or u_H), with effective
diffusivity and electromigration coefficient that depend on c_H
itself.

This is the actual residual to implement for Option C. It's roughly
2× the line count of the existing H⁺ residual but uses the same
primary variable.

### Issue 7 — Units in quadratic formula hidden: **Accept**

Explicit nondim:
```
E_hat = c_H_hat − Kw_hat / c_H_hat
c_H_hat = (E_hat + √(E_hat² + 4 · Kw_hat)) / 2
c_OH_hat = (-E_hat + √(E_hat² + 4 · Kw_hat)) / 2
```

Where `E_hat = E / C_SCALE`, `Kw_hat = Kw_M2 · (1000/C_SCALE)² /
C_SCALE` … wait, let me redo this carefully.

Define everything in mol/m³ (the project's "physical-but-nondim" base):
```
[Kw] = (mol/m³)²,  Kw = 1e-14 M² · (1000 mol/m³ / M)² = 1e-8 (mol/m³)²
```

Then in nondim concentration `c_hat = c / C_SCALE`:
```
c_OH = Kw / c_H
=> C_SCALE · c_OH_hat = Kw / (C_SCALE · c_H_hat)
=> c_OH_hat = Kw / (C_SCALE² · c_H_hat) = (Kw / C_SCALE²) / c_H_hat
=> Kw_hat ≡ Kw / C_SCALE² 
=> c_OH_hat = Kw_hat / c_H_hat
```

With Kw = 1e-8 (mol/m³)² and C_SCALE = 1.2 mol/m³, `Kw_hat = 1e-8 /
1.44 ≈ 6.94 × 10⁻⁹` (dimensionless).

Sanity: at bulk c_H_hat = 0.1/1.2 ≈ 0.0833, c_OH_hat = 6.94e-9 /
0.0833 ≈ 8.33 × 10⁻⁸, so c_OH = 1e-7 mol/m³ = 1e-10 M ✓ (matches
pH 4 bulk OH⁻).

I'll embed this derivation in the plan as a unit-consistency table.

### Issue 8 — `KW_M2` naming: **Accept**

Renaming to `KW_MOLAR_SQUARED` (not `KW_MOLAR2` per your suggestion
because `M2` and `MOLAR2` both look like "squared" but `M2` is
ambiguous-with-meters; spelling it out kills the ambiguity).
Adding test: bulk pH 4 ⟹ `c_OH_hat = Kw_hat / c_H_hat` matches
`1e-7 mol/m³ / C_SCALE`.

### Issue 9 — `c_bulk_nondim = KW_NONDIM / C_HP_HAT`: **Accept**

Compute everything from one canonical physical bulk:
```python
C_H_BULK_M3 = 0.1   # mol/m³ at pH 4
C_OH_BULK_M3 = (KW_MOLAR_SQUARED * 1e6) / C_H_BULK_M3
# = 1e-14 · 1e6 / 0.1 = 1e-7 mol/m³
C_OH_BULK_HAT = C_OH_BULK_M3 / C_SCALE  # = 8.33e-8

KW_HAT = (KW_MOLAR_SQUARED * 1e6) / (C_SCALE ** 2)  # = 6.94e-9
```

(The `1e6` is `(1000 mol/m³ per M)²`.) Test asserts both derived
constants and their consistency `KW_HAT == C_H_BULK_HAT * C_OH_BULK_HAT`
to 1e-12 relative.

### Issue 10 — Option A rejection scope: **Clarify (mostly defend)**

You're right that within the EDL inert Boltzmann is locally valid for
a *fixed* c_OH_bulk — that distinction is real. The reason I rejected
Option A is exactly the one you state: "outer electroneutral pH
response is not an EDL Boltzmann problem". Pure-Boltzmann with
c_OH_bulk = 10⁻¹⁰ M doesn't cap surface pH because it can't *source*
new OH⁻ from water dissociation. I'll rewrite the Option A rejection
to scope-clarify: "Pure-Boltzmann-with-fixed-bulk-c_OH cannot satisfy
the homogeneous reaction equilibrium, only the EDL one."

### Issue 11 — IC formula sign convention: **Accept**

I'll redo the IC closure from primaries. In the `debye_boltzmann` IC
the inner ψ profile satisfies `c_H_hat(y) ≈ H_outer · exp(-ψ(y))` (in
the dynamic 3sp + Bikerman Picard outer; OHP-anchored). With water
slaving, c_OH(y) = Kw_hat / c_H_hat(y) = (Kw_hat / H_outer) · exp(+ψ).
So OH⁻ does follow Boltzmann-like inside the EDL with effective bulk
`Kw_hat / H_outer` — but only because H⁺ itself is in EDL Boltzmann
equilibrium *at the IC*. If H⁺ is NOT in Boltzmann equilibrium (not
true for the IC because Picard picks a non-Boltzmann surface c_H),
then the closure breaks.

Concrete fix: derive the IC closure for the *Picard-state* c_H_surface
directly, then enforce `c_H · c_OH = Kw_hat` pointwise via composite-ψ
+ multispecies γ. The γ denominator term I wrote
`a_OH · Kw_hat / H_outer · (e^ψ − 1)` was assuming H⁺ Boltzmann for the
IC; replace with the explicit form using the Picard-state H+ profile.

I'll add a numerical sanity test: after IC seeding, evaluate
`c_H_hat * c_OH_hat / Kw_hat` at every node — must be ≈ 1.

### Issue 12 — `forms_logc[_muh].py` touch surface: **Accept**

I was wrong that only `boltzmann.py` would change. The new equation is
the proton-condition equation, which replaces the existing H⁺ NP
residual entirely. So forms_logc.py's NP-flux loop, BV cathodic
factor (which uses c_H), Bikerman packing, and Poisson source all
need updating. The muh sibling needs the same. I'll inventory the
specific touch sites in the revised plan:
- forms_logc.py `_build_np_residual` (H⁺ branch becomes proton-condition)
- forms_logc.py BV cathodic concentration factor (uses c_H_hat = exp(u_H))
- forms_logc.py Poisson source (already uses c_H; add c_OH)
- forms_logc.py `_try_debye_boltzmann_ic` (IC closure update)
- forms_logc_muh.py `_resolve_mu_h_index` and the muh proton-residual
  path (the proton-condition equation needs to be rewritten in μ_H
  variables; em*z_H cancellation differs)

### Issue 13 — Jacobian/derivative discussion: **Accept**

The proton-condition equation involves `c_H = exp(u_H)` and `c_OH =
Kw_hat · exp(-u_H)`. Both enter the residual symbolically; UFL's
`fd.derivative` will produce the Jacobian automatically. But:
- The effective diffusivity `D_H + D_OH · Kw_hat · exp(-2 u_H)`
  introduces a `exp(-2 u_H)` term that's huge near the cathode.
- Newton may struggle at pH > 12 where this term dominates.

Mitigation: continuation parameter `Kw_eff` ramped from 0 to Kw_hat
during anchor build (similar to the existing k0 ladder).

### Issue 14 — `u_clamp` clamp directionality: **Accept**

You're right. The existing `u_clamp = 100` is a symmetric clamp on
u_H, but `exp(-u_H)` blows up when u_H is very negative (depleted
H+). In the failed sweep, surface u_H ≈ -32 (c_H ≈ 1e-14 M / 1.2
mol/m³ ≈ exp(-32)), so `exp(-2 u_H) ≈ exp(64)` ~ 6e27 — Kw_hat ≈
6.94e-9 means the OH⁻ contribution `Kw_hat · exp(-2u_H)` ~ 4e19,
which is enormous.

Two clamp options:
(a) Lower-clamp u_H so c_H_hat ≥ 1e-30 (i.e., u_H ≥ -69). Test
    sensitivity of Newton convergence to this clamp level.
(b) Direct clamp on c_OH_hat ≤ some max (e.g., 1000 mol/m³ / C_SCALE
    = 833 nondim, corresponding to pH 14). Matches physical sanity:
    you can't have c_OH > total water concentration.

I'll add (b) as the primary clamp and (a) as a safety belt.

### Issue 15 — c_OH magnitude at pH 14: **Accept (largest physical concern)**

Critical point. The failed-sweep state has surface pH 14, meaning
c_H_surface = 1e-14 M = 1e-11 mol/m³, and slaving would give
c_OH_surface = Kw / c_H = 1e-14 M² / 1e-14 M = 1 M = 1000 mol/m³.

This is HUGE. Compared to:
- Cs⁺ bulk: 200 mol/m³
- SO₄²⁻ bulk: 100 mol/m³
- Steric saturation packing fraction: ~1 (i.e., volume occupied)

OH⁻ at 1000 mol/m³ with `a_OH_hat ≈ 1.6 × 10⁻⁵` gives packing
contribution `1.6e-5 · 1000/1.2 = 0.013` — actually small. The
Marcus radius is small enough that OH⁻ doesn't dominate steric.

But Poisson charge density: `(-e) · 1000 mol/m³` is 10× larger than
the largest dynamic species. The local potential gradient would
respond strongly. Newton may struggle.

**Continuation strategy** (new in revised plan):
1. Anchor at +0.55 V on the EXISTING (no water) stack (already done).
2. Introduce `Kw_eff = α · Kw` continuation parameter, α ∈ [0, 1].
3. Ramp α from 0 to 1 in steps (e.g., α ∈ {0, 1e-6, 1e-3, 0.1, 0.5, 1.0}),
   solving Newton at each rung.
4. As α grows, the "missing source" turns on, surface c_H rises,
   surface c_OH = α Kw / c_H stays bounded by the equilibrium.

This is structurally analogous to the k0 ladder we already use for
anchor builds. Should work.

### Issue 16 — "Lift" wording: **Accept**

Changing acceptance criterion 2 to "plateau magnitude increases
toward deck −0.18 mA/cm² (cd becomes more negative at the deepest
cathodic V_RHE)".

### Issue 17 — P3 alone insufficient: **Accept**

Adding to acceptance criteria:
- **Mass conservation**: integrate ∂E/∂t over the domain; should
  equal net surface BV current scaled by F·n. Tolerance: 1e-3.
- **Local equilibrium**: `c_H · c_OH / Kw` at every node must be in
  [0.99, 1.01] (allowing Newton-iteration tolerance).
- **Yash cross-check**: at one (V_RHE, pH=4, Cs⁺) point, plot
  c_OH(y) profile from our solver against Yash's `conc_OH-` snapshot.
  Order-of-magnitude agreement is the success target (we're doing a
  shortcut, not a re-derivation).
- **Charge neutrality far from EDL**: at y = 0.5 · domain_height_hat,
  the integrated charge density should be < 1% of typical EDL value.

### Issue 18 — pH ↔ concentration consistency: **Accept**

I conflated two different states. The original failed sweep reports
surface pH 13.72 → c_H_surf = 10⁻¹³·⁷² M = 1.9e-14 M = 1.9e-11 mol/m³.
The "10⁻⁹ M" I quoted in the plan was a hypothetical "deeper-cathodic"
estimate, not the actual result. I'll standardize to one tabulated
column with both M and mol/m³ side-by-side throughout the revised plan.

### Issue 19 — Sulfate buffering glib: **Defend (with quantification)**

At pH 4, [SO₄²⁻]/[HSO₄⁻] = 10^(pH − pKa) = 10^(4 − 1.99) = 10²·⁰¹ ≈
102. So ~99% sulfate, ~1% bisulfate. With [SO₄²⁻] = 0.1 M, [HSO₄⁻] ≈
1 mM. Buffer capacity β = 2.303 · C_buffer · α · (1−α) with α ≈
0.99 gives β ≈ 2.3 · 0.1 · 0.01 ≈ 2 mM/pH — not huge, but
non-negligible.

The reason I deferred sulfate is not that it's irrelevant — it's that
at the *surface* (where the action is), the local pH crosses pKa = 2
so sulfate buffering becomes significant only well into the alkaline
regime where water self-ionization dominates. So I'd argue:
1. Water self-ionization is the dominant proton-supply mechanism in
   the regimes we care about (pH 4 to 14 surface).
2. Sulfate buffering becomes load-bearing only in a narrow band
   (pH 1-3) which we don't touch in the cathodic sweep.

But you're right I should *quantify* this rather than gesture. I'll
add a buffer-capacity comparison in the revised plan's "out of scope"
section. If validation P3 still fails after water ionization,
sulfate goes back on the table.

### Issue 20 — Byte-identity regression brittle: **Accept**

Switching to numerical regression with relative tolerance 1e-10 on
cd / pc / surface c_H values. The byte-identity test is replaced
with an "expression-shape" test (assert that with the water-ionization
flag off, the residual UFL expression set has no `Kw_hat` constant
and no slaved-OH term).

### Issue 21 — MMS scope: **Accept**

Dropping MMS from the Option B path (which is dropped entirely). For
Option C, MMS is meaningful: a manufactured `c_H(y, t)` profile with
`c_OH = Kw/c_H` and the corresponding analytic E and J_E gives a
closed-form residual to verify the implementation at the right order
(p+1 for CG-p elements with a smooth solution).

### Issue 22 — Second-order claim unsupported: **Accept**

Withdrawing the "OH⁻ flux divergence is second-order" claim. There's
no scaling argument that supports it; D_OH · ∇c_OH can be very large
near the cathode. I'll remove this claim from the revised plan.

### Issue 23 — Escalation logic backwards: **Accept (see #3-#5 fix)**

Pivoting Option C to first landing.

### Issue 24 — D_OH unused in Option B: **Accept (moot)**

Option B is dropped; D_OH is needed for Option C (in the effective
diffusivity term `D_H + D_OH · Kw / c_H²`).

### Issue 25 — Nondim θ closure: **Accept**

Rewriting steric closure as
`θ_b += a_OH_hat · (Kw_hat / c_H_hat_bulk)` with all quantities nondim.

## 2. Updated artifact (revised plan)

The revised plan reframes around Option C as the recommended landing.
Posting the section-by-section diff (changes only):

**§1 Context** — keep, but standardize all pH values with both M and
mol/m³.

**§2 Diagnosis** — keep.

**§3 Shortcut options** — restructured:
- Option A (pure Boltzmann): rejection scope-clarified per #10.
- **Option B (algebraic-only c_OH)**: reframed as a *diagnostic-only*
  experiment — implement only if we want to confirm the diagnosis
  ("does adding OH⁻ to Poisson without sourcing H⁺ change anything?").
  Predicted result: surface pH stays ≈ 14 because c_H still has no
  source. We may skip the diagnostic and go straight to Option C.
- **Option C (proton-condition variable, recommended first landing)**:
  expanded with explicit PDE derivation per #6, #7. The
  implementation keeps `u_H = ln c_H` as the primary variable but
  swaps the H⁺ NP residual for the proton-condition residual. The
  effective diffusivity and electromigration coefficient depend on
  c_H itself.
- Option D (full dynamic OH⁻): unchanged; reserved for if Option C
  fails.

**§4 Recommendation** — pivoted to Option C. Damköhler arithmetic
fixed per #1, #2.

**§5 Implementation plan** — restructured around Option C:
- Q1: Constants and config knobs (renamed per #8, recomputed per #9).
- Q2: Replace H⁺ NP residual with proton-condition residual in
  `forms_logc.py` and `forms_logc_muh.py`. Touch surface enumerated
  per #12. Continuation parameter `Kw_eff` plumbed through per #15.
- Q3: IC closure rederived per #11. Add post-IC sanity test
  `c_H · c_OH / Kw ∈ [0.99, 1.01]`.
- Q4: Tests rewritten per #20 (numerical regression), #21 (MMS for
  Option C, not B).
- Q5: Acceptance criteria expanded per #17 (mass conservation, local
  equilibrium, Yash cross-check, far-field charge neutrality).

**§6 Risk register** — updated:
- R1: Option C fails P3 → escalate to Option D (full dynamic).
- R2: Newton stiffness from `exp(-2u_H)` near cathode → continuation
  parameter `Kw_eff` (per #14, #15).
- R3: c_OH magnitude at pH 14 → bounded by `c_OH_clamp` and
  continuation (per #15).
- R4: IC composite-ψ closure for Option C → rederived per #11.
- R5: Default-off flag preserves regression (per #20).
- R6: Damköhler validity uniformly ≫ 1 across L_eff sweep (per #1, #2).
- R7 (NEW): Effective diffusivity term `D_OH · Kw/c_H²` introduces
  Newton stiffness near surface; continuation strategy is the
  primary mitigation.

**§7 Out of scope** — sulfate buffering quantified per #19 (defer
unless P3 still fails after Option C).

**§8 Open questions** — three new ones:
- Should `Kw_eff` continuation be a multi-rung ladder analogous to k0,
  or a single-shot homotopy?
- Does the proton-condition residual remain Newton-friendly with the
  `exp(-2u_H)` term, or do we need a different primary variable
  (e.g., E_hat directly) at high pH?
- Is the post-IC sanity test `c_H · c_OH / Kw ∈ [0.99, 1.01]` the
  right tolerance, or do we need tighter at the surface?

I've not yet rewritten the plan markdown file; the diff above is
the spec. Will write the actual artifact after the loop converges.

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
