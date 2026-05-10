# Round 3 — counterreply on Phase 6α plan

## 1. Acknowledgment

### Issue 1 — NP flux missing diffusivity in migration: **Accept**

You're right. Standard Nernst-Planck has D_i on both terms:
`J_i = -D_i (∇c_i + z_i c_i ∇φ)`. I had `c_H · ∇φ` (no D_H), which
is wrong. Re-derived properly:

```
J_H  = -D_H · ∇c_H  - D_H · c_H · ∇φ                (z_H = +1)
J_OH = -D_OH · ∇c_OH + D_OH · c_OH · ∇φ              (z_OH = -1)
```

With c_OH = Kw_hat / c_H:

```
J_E = J_H − J_OH
    = -[D_H + D_OH · Kw_hat / c_H²] · ∇c_H
      - [D_H · c_H + D_OH · Kw_hat / c_H] · ∇φ
    = -[D_H · c_H + D_OH · c_OH] · ∇(ln c_H)
      - [D_H · c_H + D_OH · c_OH] · ∇φ
    = -[D_H · c_H + D_OH · c_OH] · (∇u_H + ∇φ)
```

In muh primary (μ_H = u_H + em·z_H·φ; em·z_H = +1 in production):

```
J_E_muh = -[D_H · c_H + D_OH · c_OH] · ∇μ_H
```

This is exactly your simplified expression. Adopting it.

### Issue 2 — Conservative weak form: **Accept**

Adopting your weak form. In log primary `u_H = ln c_H`:

```
∫ v · (c_H + c_OH) · ∂u_H/∂t  dx
+ ∫ ∇v · J_E  dx
- ∫ v · J_E · n  ds = 0
```

For steady-state runs (which is what we run for the L_eff sweep
acceptance) the time term drops and we just have:

```
∫ ∇v · J_E  dx - ∫ v · J_E · n  ds = 0
```

with J_E from #1. Conservation is automatic in this form.

### Issue 3 — Boundary conditions: **Accept**

Explicit BCs:

| Boundary | Marker | Condition |
|---|---|---|
| Electrode (y=0) | 3 | `J_E · n = J_H,BV · n − J_OH,Faradaic · n = J_H,BV · n` (J_OH Faradaic = 0 at pH 4 acidic ORR) |
| Bulk top (y=L) | 4 | Dirichlet `c_H = c_H_bulk` ⟹ `E = c_H_bulk − Kw_hat / c_H_bulk` (pinned via the existing top-boundary Dirichlet machinery on u_H or μ_H) |
| Sides (x=0, x=1) | 1, 2 | No-flux `J_E · n = 0` (zero-flux on sides) |

H⁺ Faradaic flux at the electrode follows the existing
parallel-2e/4e formula: H⁺ consumed at 1 per electron in both R_2e
(2 H⁺ per 2 e⁻) and R_4e (4 H⁺ per 4 e⁻).

J_OH = 0 at the electrode in this regime (pH 4, acidic-form ORR
only — no alkaline-form OH⁻ generation reaction). If we ever add
the alkaline-form pathway (Phase 6γ), J_OH,Faradaic gets a Faradaic
component and the boundary changes.

### Issue 4 — muh derivation separately: **Accept**

For steady state (which is the actual production target — both the
anchor build and the warm-walk grid solver call SS at each V_RHE),
∂μ_H/∂t = 0 and the storage term drops, giving:

```
∇·J_E = 0 in interior
J_E = -[D_H · c_H + D_OH · c_OH] · ∇μ_H
```

For transient runs (used during the cold ramp-up to anchor), the
storage term in muh is:

```
∂E/∂t = ∂(c_H − c_OH)/∂t
      = c_H · (∂μ_H/∂t − ∂φ/∂t) + c_OH · (∂μ_H/∂t − ∂φ/∂t)
      ... wait, let me redo
```

In muh: c_H = exp(μ_H − em·z_H·φ) = exp(μ_H − φ) (em·z_H = +1).
c_OH = Kw_hat / c_H = Kw_hat · exp(φ − μ_H).

∂c_H/∂t = c_H · (∂μ_H/∂t − ∂φ/∂t)
∂c_OH/∂t = c_OH · (∂φ/∂t − ∂μ_H/∂t) = -c_OH · (∂μ_H/∂t − ∂φ/∂t)

∂E/∂t = ∂c_H/∂t − ∂c_OH/∂t = (c_H + c_OH) · (∂μ_H/∂t − ∂φ/∂t)

So the muh transient storage term is `(c_H + c_OH) · (∂μ_H/∂t −
∂φ/∂t)`. There IS a `∂φ/∂t` coupling that the logc form doesn't
have. I'll write this explicitly in the residual.

(For our actual production runs: Phase 6α validation is steady-
state only. The cold-ramp transient doesn't necessarily need
exact water-equilibrium dynamics; the SS rungs after each ramp
step are what matter. But documenting the transient form
defensively.)

### Issue 5 — Damköhler arithmetic still sloppy: **Accept**

Fixing the actual numbers:
- `1/(k_r · c_H_bulk) = 1/(1.4e11 · 1e-4) = 1/(1.4e7) ≈ 71 ns` at
  bulk. (I had said 0.7 µs.)
- At pH 14, c_OH = 1 M not "both ~1 M" (c_H = 1e-14 M, c_OH = 1 M).
  Relaxation time `1/(k_r · c_OH) = 1/(1.4e11 · 1) = 7 ps`.

Rewriting §1.5 of the revised plan with one consistent table.

### Issue 6 — NEW BLOCKER, finite water source capacity: **Accept (critical addition)**

This is a real constraint and a falsifiable prediction Option C
should be evaluated against. Working it through:

Maximum volumetric water source rate (when c_H · c_OH ≪ Kw):
```
R_w_max = k_f · [H₂O] = k_r · Kw ≈ 1.4 × 10⁻³ M/s = 1.4 mol/m³·s
```

Total H⁺ supply capacity per unit electrode area, distributed over
transport-domain depth L_eff:
```
N_H_supply_max = R_w_max · L_eff      (mol/m²·s)
```

Equivalent maximum BV current (per electron, accounting for n_H+ = 1
per electron in acidic ORR):
```
i_max [mA/cm²] = N_H_supply_max · F · 0.1
              = 1.4 · L_eff · 96485 · 0.1
              = 13.5 · (L_eff / 100 µm) mA/cm²
```

Comparing against deck targets:

| L_eff | i_max (water source only) | deck plateau (-0.18) | deck peak (-0.40) | headroom (plateau / peak) |
|------:|--------------------------:|---------------------:|------------------:|--------------------------:|
| 100 µm | 13.5 mA/cm² | 0.18 | 0.40 | 75× / 34× ✓ |
| 66 µm  | 8.9         | 0.18 | 0.40 | 49× / 22× ✓ |
| 21 µm  | 2.83        | 0.18 | 0.40 | 16× / 7× ✓ |
| 16 µm  | 2.16        | 0.18 | 0.40 | 12× / 5× ✓ |

Water source has comfortable headroom across all L_eff for the
deck-aligned current densities. Even at L_eff=16 µm the deck peak
(0.40 mA/cm²) is 5× under the water-supply ceiling.

But for the model's CURRENT (without-water) overshoot, e.g. -0.5617
mA/cm² at L_eff=16 µm — that's only ~4× under the ceiling. So
Option C is expected to PULL the cathodic plateau MAGNITUDE
DOWN at small L_eff (toward water-supply-limited ~0.5 mA/cm² rather
than today's Levich-only 0.5617). At L_eff=100 µm there's so much
headroom that the source merely caps surface c_H without limiting
the plateau.

**New validation scalar to track in the post-sweep verdict.json:**

```python
required_to_max_ratio[i] = abs(cd[i]) / (1.4 · L_eff_m · F · 1e-1)
# fail if max(required_to_max_ratio) > 0.5 (50% of source capacity)
```

> If `max(required_to_max_ratio) > 0.5` over the sweep, water alone
> may be insufficient at deeper cathodic V_RHE; sulfate buffering or
> Option D (full dynamic OH⁻ with finite-rate R_w) becomes
> necessary.

I'll add this to acceptance criteria.

### Issue 7 — c_OH_clamp breaks the model: **Accept**

Removing the c_OH_clamp from the plan. Continuation parameter
`Kw_eff` is the primary stabilization (start with Kw_eff = 0,
ramp to Kw, with Newton at each rung). If a clamp is ever
triggered, the run is invalid; document and fail the validation.

For the IC continuation specifically:
```
Kw_eff schedule = [0, Kw_hat * 1e-6, Kw_hat * 1e-3, Kw_hat * 0.1, Kw_hat]
```

(Five rungs; mirrors the existing k0 ladder.)

### Issue 8 — exp(-2u) stiffness is self-inflicted: **Accept (large win)**

Implementing in log/muh primary directly. The flux coefficient
`D_H · c_H + D_OH · c_OH = D_H · exp(u_H) + D_OH · Kw_hat ·
exp(-u_H)` is well-conditioned: at high pH, the second term
dominates with a smooth `exp(-u_H)` (not `exp(-2u_H)`); at low pH,
the first term dominates. Newton sees a smooth coefficient
function. This is a major win and removes one of the largest
stiffness concerns.

R7 in the risk register now reads: "Effective flux coefficient
`D_H exp(u_H) + D_OH Kw exp(-u_H)` is well-conditioned in
log/muh primary; no stiffness from exp(-2u) since that form is
not used."

### Issue 9 — Local equilibrium check tautological: **Accept**

Dropping `c_H · c_OH / Kw ∈ [0.99, 1.01]` from acceptance. It's
trivially true by construction. Real validations:

a) **E conservation per V_RHE point**:
   `|J_H,BV(electrode) − J_E·n(top)| / |J_H,BV| < 1e-3`
b) **Surface BV current matches volume integral**: from the weak
   form's identity, the assembled `cd` should equal the boundary
   integral evaluated independently.
c) **Yash cross-check** (per #15 below): match conditions and
   compare c_OH(y) profile.
d) **Finite-source capacity check** (per #6): track
   `max_required_to_max_ratio` per combo.

### Issue 10 — Mass-conservation acceptance under-specified: **Accept**

Explicit nondim identity. With `J_E_hat` as defined in #1 (sign
convention: outward normal n at all boundaries), steady-state
divergence theorem:

```
∮_∂Ω  J_E_hat · n  ds = 0

= ∫_top   J_E_hat · n  ds         (Dirichlet, computed)
+ ∫_sides J_E_hat · n  ds = 0     (no-flux)
+ ∫_electrode  J_E_hat · n  ds    (= surface BV current)
```

So at the electrode (n = -ŷ), J_E·n must equal J_H,BV·n (with sign).
Dimensionally: J_E_hat has units `(c̃·m̃/s̃) = (C_SCALE · L_REF /
τ_ref) = C_SCALE · D_REF / L_REF`. Multiplying by F · 0.1 gives
mA/cm². The acceptance is:

```
|cd(V_RHE) − I_SCALE · ∫_electrode J_E_hat · n  ds| / |cd(V_RHE)| < 1e-3
```

I'll add this as a per-point assertion to the validation script.

### Issue 11 — Sulfate defense weak: **Accept**

Quantifying before deferring. Sulfate buffer reservoir at pH 4 with
[SO₄²⁻] = 100 mol/m³ and pKa(HSO₄⁻) = 1.99:
```
[HSO₄⁻] / [SO₄²⁻] = 10^(pKa − pH) = 10^(-2.01) ≈ 0.0098
[HSO₄⁻] ≈ 0.98 mol/m³ ≈ 1 mM
```

Compare to bulk H⁺ = 0.1 mol/m³ = 0.1 mM. So sulfate reservoir is
~10× larger than free proton pool. **However**, sulfate buffering
acts only when local pH approaches pKa = 2 (i.e., during the
deeply cathodic regime where surface pH is rising AND sulfate
reservoir is being depleted by H⁺ release). For the regime our
sweep targets (V_RHE ≥ -0.40, deck peak at +0.10 V, deck plateau
~-0.18 mA/cm²), surface pH should be pulled from ~14 (current) to
~7-9 with water, well outside sulfate's pKa window.

Worst-case sulfate release rate (instantaneous): if all 1 mM
HSO₄⁻ in the EDL releases its proton, the supply per area is
`0.001 mol/L · 1000 L/m³ · L_EDL ≈ 1 mol/m³ · 1e-7 m = 1e-7
mol/m²·s` (over τ_release ~ 1 ns), giving an instantaneous flux
~1e-7 / 1e-9 mol/m²·s = 100 A/m² = 10 mA/cm². Comparable to
water source. But this is a one-shot reservoir; sustained
buffer release requires SO₄²⁻ → HSO₄⁻ recombination, which
is slow.

So sulfate is a transient buffer, not a steady-state supply.
Defer Phase 6β unless validation P3 still fails after Option C
or unless sulfate transients show up as Newton instability
during continuation.

I'll add this quantitative analysis to §7 and explicitly
condition on "if Option C falls short."

### Issue 12 — Thermodynamic consistency with Bikerman: **Accept**

Adopting the **concentration-Kw model** (sterics applied only via
Poisson source and packing fraction, NOT as an activity correction
on the water equilibrium). This is the simplest defensible model:
- Water equilibrium uses bare concentrations: `c_H · c_OH = Kw_hat`.
- Steric effects enter Poisson via packing-fraction-modulated
  charge density (existing mechanism for Cs⁺/SO₄²⁻).
- No γ_i activity correction on the water reaction.

Documenting this as a model assumption in §3 of the revised plan
with the explicit alternative noted ("activity-Kw with γ_H · γ_OH ·
c_H · c_OH = Kw_thermo would require fully consistent thermodynamic
treatment of all species — out of scope for Phase 6α").

This matches what Yash's reference implementation likely does
(I'll verify in the cross-check step) and avoids opening up the
activity vs. concentration can of worms in a phase whose primary
goal is just lifting surface pH from 14 to <9.

### Issue 13 — IC plan still hand-wavy: **Accept**

Explicit IC reconstruction order:

```
Step 1 (existing): Run Picard outer loop with H+ as primary.
  Result: c_H(y) profile, Picard surface c_H_s, ψ_D, OHP phi_o.

Step 2 (NEW): Compute c_OH(y) = Kw_hat / c_H(y) pointwise,
  with Kw_eff = Kw_hat (or the active continuation-parameter value).

Step 3 (NEW): Recompute charge density:
  ρ(y) = Σ_dyn z_i · c_i(y) + Σ_inert z_j · c_j_bulk · ζ_j(ψ)
       + (-1) · c_OH(y)         ← new term

Step 4 (NEW): Re-solve linearised Poisson with the updated ρ(y)
  for an updated φ(y); update u_H to maintain c_H = exp(u_H)
  consistent with Picard surface state.

Step 5 (NEW): Verify the new IC satisfies the proton-condition
  residual numerically: assemble F_E from the muh weak form and
  check ||F_E||_L2 < 1e-3 · ||u_H||_L2.

Step 6 (NEW): If residual norm too large, iterate (Picard outer
  loop with the now-water-aware Poisson).
```

This is a minor extension to `_try_debye_boltzmann_ic` — adds
steps 2-6 after the existing Picard loop converges. Documenting
in §5 Q3 of the revised plan.

### Issue 14 — Expression-shape regression fragile: **Accept**

Dropping the "expression has no Kw_hat constant" check. Replacing
with:
- Disabled-path numerical equivalence: same `mangan_full_grid`
  output (cd, pc, j_ring, surface_pH) within 1e-10 relative
  tolerance to the pre-Phase-6α baseline when the
  `enable_water_ionization` flag is False.
- Config-level assertion: `enable_water_ionization=False` ⟹ no
  `kind="water_ionization"` entry ever reaches the residual
  builder (assert at solver-params construction time).

### Issue 15 — Yash cross-check needs comparability gates: **Accept**

Explicit gating before declaring Yash agreement. The test
condition will lock:
- V_RHE = -0.40 V (one cathodic point, where pH gap is largest)
- pH = 4 bulk
- Cation = Cs⁺ (matched), supporting anion = SO₄²⁻ (matched)
- L_eff = 100 µm (matched to Yash's mesh extent — verify in his
  npy snapshots)
- Coordinate scaling: convert Yash's 121-pt y-grid to nondim
  L_REF=100 µm units
- Bulk concentrations: H⁺=0.1, Cs⁺=199.9, SO₄²⁻=100, OH⁻=1e-7 mol/m³
- Boundary conditions: bulk top Dirichlet on c_H, OH⁻

Acceptance: c_OH(y) profile from our solver vs Yash's `conc_OH-`
profile match within 1 OOM at every node, and within 50% relative
error at the surface OHP (the most sensitive point).

A larger discrepancy is investigated as a model-side question
(e.g., does Yash include sulfate buffering?), not a solver-bug
question.

## 2. Updated artifact (revised plan summary)

Section-by-section diff vs R2 spec:

**§1 Context** — keep, with #5 arithmetic fixes and #18 unit
standardization (one table, both M and mol/m³).

**§2 Diagnosis** — keep.

**§3 Shortcut options** — Option C as primary:
- Equation: `∂E/∂t + ∇·J_E = 0` with E = c_H − c_OH
- Flux: `J_E = -(D_H c_H + D_OH c_OH) · (∇u_H + ∇φ)` in logc
        `J_E = -(D_H c_H + D_OH c_OH) · ∇μ_H` in muh
- Weak form (#2 conservative).
- BCs (#3 explicit).
- Concentration-Kw (#12).

**§4 Recommendation** — Option C primary; Damköhler arithmetic per
#1, #2, #5; finite-source capacity check per #6.

**§5 Implementation plan** — restructured:
- Q1: Constants (`KW_MOLAR_SQUARED`, `D_OH_HAT`, `A_OH_HAT`,
  bulk values from one canonical physical baseline).
- Q2: Replace H⁺ NP residual with proton-condition residual in
  log/muh primary form (no exp(-2u)), per #1, #2, #4, #8.
  Continuation `Kw_eff` per #7. Touch surfaces per #12.
- Q3: IC reconstruction order per #13.
- Q4: Tests:
  - Unit: Kw_hat consistency (#9), bulk OH consistency, expression
    routing (#14).
  - Slow: anchor at +0.55 V on the new stack converges through the
    Kw_eff continuation. MMS for the proton-condition equation
    (manufactured smooth c_H(y) → c_OH = Kw_hat / c_H, plug into
    J_E to get residual; verify p+1 convergence).
  - Regression: numerical-tolerance disabled-path match (#14, #20).
- Q5: Validation sweep with five acceptance gates:
  1. P3: max_surface_pH < 9 at L_eff=16 µm.
  2. Plateau: cd at L_eff=100 µm becomes more negative
     (toward deck −0.18); direction-of-change check.
  3. E conservation: per-V_RHE check `|J_E·n(electrode) +
     J_E·n(top)| / |J_H,BV| < 1e-3` (#10).
  4. Finite-source capacity: max(`required_to_max_ratio`) < 0.5
     across all (L_eff, V_RHE) combos (#6).
  5. Yash cross-check at the gated test condition (#15).

**§6 Risk register** — updated:
- R1: Option C fails P3 → Option D (full dynamic OH⁻ with
  finite-rate R_w).
- R2: Newton stiffness during Kw_eff continuation → tighten the
  schedule, add intermediate rungs.
- R3: c_OH packing dominance at high pH → continuation handles it;
  if Bikerman θ saturates, document and consider activity-Kw.
- R4: IC composite-ψ closure with water → reconstruction order
  per #13 + numerical residual check.
- R5: Default-off flag preserves regression.
- R6: Damköhler validity verified (#1, #2, #5).
- R7: Effective flux coefficient `D_H exp(u_H) + D_OH Kw exp(-u_H)`
  is well-conditioned in log/muh primary (#8).
- R8 (NEW): Finite water source capacity (#6) — track per-combo
  ratio; if > 50%, escalate to Option D or add sulfate buffering.

**§7 Out of scope** — sulfate quantified per #11; deferred unless
P3 still fails after Option C OR finite-source ratio > 50 %.

**§8 Open questions** — converged to:
1. Kw_eff continuation schedule rungs (5 from current spec; tune
   if Newton struggles).
2. transient form for muh in cold-ramp transients (per #4) —
   probably only matters if SS rungs converge slowly.
3. IC residual tolerance threshold (1e-3 vs 1e-4) for "good
   enough" before Newton picks up.

## 3. Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
