# R4 — counterreply (v9 with desorption removal path + phased gating)

Round 4 of 5. v8 still had load-bearing flaws — most importantly
the steady-state-zero-source recurrence (#1), the
two-z=+1-species build blocker (#4), the Γ plumbing reality (#7),
and the still-too-ambitious smoke (#12). v9 fixes these.

## 1. Per-issue response

**Re #1 (steady-state R_net=0 without removal path).** Accept,
fundamental. Pure reversible hydrolysis at the surface gives
R_f = R_r at steady state regardless of whether Γ is a DOF. v9
adds an explicit MOH⁰ removal path:

```
∂_t Γ_MOH = R_f − R_r − R_des
R_des = k_des · Γ_MOH                    [units: mol/m²/s]

steady state: R_f − R_r = R_des = R_net  (nonzero turnover)
H+ source rate at boundary: R_net = R_des = k_des · Γ_MOH
M+ sink rate at boundary:    R_net = k_des · Γ_MOH
                             (cation comes in, MOH leaves, H+ released)
```

The desorption path returns MOH⁰ to the bulk diffuse layer (where
it diffuses away as a neutral species — no electromigration). For
the diffuse-layer model to remain tractable, MOH⁰ is treated as
a "released" species that **does not enter** the analytic Bikerman
or Poisson source — it's a sink for OHP-MOH that disappears into
the bulk before having a measurable effect on the diffuse-layer
electrostatics. This is a controlled approximation: MOH⁰ in the
bulk effectively becomes part of the "bulk water"; we don't track
its concentration.

If the user later needs MOH⁰ tracking (e.g. for product analysis),
v9 can be extended with c_MOH as a 5th NP species, but that's not
6β.1 scope.

The k_des rate constant is a tunable; per Singh-style hydrolysis
literature, MOH⁰ desorption from a polarized OHP is fast (the
neutral species has no electrostatic binding). v9 fixes k_des to
a literature-anchored prior (~10⁶ 1/s × 1/δ_OHP, or per Bohra 2019
if cited there); flag for sensitivity sweep at 6β.2.

**Re #2 (mass balance prose chemically wrong).** Accept. v9 §7
uses the clean R_net form:

```
R_net = R_f − R_r = k_hyd · c_M+(0) − k_prot · c_H(0) · Γ_MOH

Boundary residuals:
  c_M+ NP flux at electrode: J_M+·n |_{y=0} = R_net  (sink)
  Γ inventory:                ∂_t Γ_MOH = R_net − k_des·Γ_MOH
  Proton condition source:    H+ source at boundary = R_net
                              (forward consumes M+, releases H+; reverse
                              consumes H+, releases M+; net is R_net both ways)
```

(I had mis-stated this in R3 §1.)

**Re #3 (k_prot units).** Accept, k_prot is m³/(mol·s), not
m⁴/(mol·s). Equilibrium gives `Γ_eq = δ_OHP · Ka_eff_volume · c_M+
/ c_H` if we interpret `Γ/δ_OHP` as the local volume MOH⁰
concentration and Ka_eff_volume as the bulk-equivalent Ka. The
deck pKa transfers cleanly under this convention.

**Re #4 (logc_muh hard-fails with two z=+1 species).** Accept,
this is a **direct implementation blocker**. v9 implementation
order pre-prep:

1. Refactor `Forward/bv_solver/forms_logc_muh.py:_resolve_mu_h_index`
   to take an explicit `h_index` argument from species config
   instead of inferring from `z=+1`.
2. Same for `Forward/bv_solver/water_ionization.py:79`.
3. Update `_bv_common.py` species config to include explicit
   `role: "proton" | "counterion" | "neutral"` fields.
4. Add a regression test: 4-species K2SO4-with-H stack builds
   without z=+1 inference errors.

These are **prerequisite** for v9. They land before any hydrolysis
work; they're not part of the "physics" pivot, they're infrastructure
that the prior 3sp/Boltzmann stack didn't need.

**Re #5 (dynamic K+ at cathodic = dynamic ClO4 at anodic in
stiffness).** Accept. v9's first phase smoke is *just*
build+converge for 4sp K+ NP (no hydrolysis):

```
Smoke gate 1: build/form test
  - Refactored _resolve_mu_h_index + water_ionization w/ explicit
    h_index. Tests pass at 549 fast + 10 slow.
  - 4sp K2SO4 stack assembles; assembly time within 2× of 3sp.

Smoke gate 2: λ=0 dynamic-K equilibrium smoke
  - 4sp K+ NP + analytic SO4 + Phase 6α water-ionization, λ=0.
  - V_RHE = -0.40 V × C_S = 0.10 F/m² × L_eff = 16 µm.
  - Converges to within numerical tolerance of the existing
    3sp+Boltzmann-K stack at the same V_RHE.
  - Verifies that c_M+ NP profile matches Boltzmann to within a
    semantic tolerance (e.g. L²-norm difference < 1% at boundary).
  - DOES NOT NEED to converge across full V_RHE grid. Just one
    cathodic point.
```

Gates 3 and 4 below.

**Re #6 (λ=0 byte-equivalent impossible).** Accept. v9 §7 replaces
"byte-equivalent" with **semantic tolerances** at λ=0:

* H+ residual L²-norm matches Phase 6α to within tolerance ε ≈ 1e-6
  on c_H, c_O2, c_H2O2 observables.
* c_M+ NP profile matches analytic Boltzmann profile to within ε ≈ 1e-3
  on the boundary value c_M+(0). Larger tolerance because the NP
  solver introduces small numerical drift away from the analytic
  closed-form.
* Γ_MOH stays at 0.0 exactly (boundary scalar initialized to 0 +
  no source at λ=0 by hard-zeros per R3#10).

The "TestHydrolysisActivationZeroReducesToBaseline" regression
test name updates to "TestHydrolysisActivationZeroReproducesPhase6aSemantics".

**Re #7 (Γ as facet-supported Function not current plumbing).**
Accept. v9 uses **one global R scalar Γ** (a single mol/m² constant
value across the BV electrode marker) coupled through `ds`. This
is appropriate for the 1D RDE / RRDE model where the electrode
surface is "uniform" in the sense that there's no spatial variation
across the disk (the model is 1D, y-direction only).

In Firedrake: a `Function` on a mesh-wide `Real` element OR a
single `Constant` updated at each Newton iteration. Easiest is a
0-form (single scalar in the solution space). Implementation:

```python
# In set_up_function_space:
W_volume = MixedFunctionSpace([V_phi, V_logc_O2, V_logc_H2O2,
                                V_logc_H, V_logc_K])  # 4sp + phi
W_global = FunctionSpace(mesh, "Real", 0)              # global scalar
# Mixed: W = W_volume × W_global
```

The Γ residual:

```python
F_res += (G_test * (R_net - k_des*Gamma) - dGamma_dt * G_test) * ds(electrode_marker) / area_electrode
       # or equivalent, with proper weighting; defer exact form to implementation
```

For 1D this assembly is straightforward. v9 §7 documents that the
Γ-as-global-scalar choice limits the model to spatially uniform
electrodes; defer facet-supported Γ to a future phase if spatially
non-uniform OHP chemistry is ever needed.

**Re #8 (Stern correction in Γ units).** Accept. The cleaner form:

```
Physical:    σ_hydrolysis_correction = F · Γ_MOH         [C/m²]
Nondim:      σ_hat_corr = Γ_hat   (with Γ_hat = Γ · F / scale)
                                  ↳ scale = F·C_REF·L_SCALE/V_T
                                          = same as stern_coeff scale
Implementation:
F_res -= (stern_coeff_hat·(φ_applied − φ) + Γ_hat) · w · ds(electrode_marker)
```

This avoids mixing Γ_MOH (areal mol/m²) with c_MOH_hat (volumetric
nondim) and δ at all. δ is needed only for packing diagnostics
(`a_MOH·Γ_MOH/δ` for boundary cell packing) and the equilibrium
algebra (`Γ_eq = δ · Ka_eff_volume · c_M+ / c_H`).

**Re #9 (hybrid dynamic K+ + analytic SO4 closure consistency
proof).** Accept. v9 smoke gate 2 explicitly includes the
manufactured-equilibrium test: at λ=0, dynamic K+ + analytic SO4
must reproduce the analytic-K+/analytic-SO4 charge density and
packing on a fixed φ profile, to within numerical tolerance. Test
is `TestDynamicKplusAnalyticSO4MatchesAnalyticBaseline`.

**Re #10 (calibration underdetermined).** Accept. v9 freezes
literature priors:

* `δ_OHP = 0.40 nm` (Bohra 2019 fixed prior)
* `r_K = 0.23 nm` (Linsey deck slide 13)
* `C_S = 0.10 F/m²` (existing production tunable; sensitivity at
  {0.05, 0.10, 0.20})
* `k_des = 10⁶ /s × 1/δ_OHP` (literature MOH⁰ desorption from
  polarized OHP; one-time-fit at 6β.2 calibration)
* `β_M_K = 1.0` (Singh 2016 default; sensitivity at {0.5, 1.0, 2.0})
* `k_hyd = 10² m/s × ratio` (placeholder; absorbed into K_s)

v9 fits **only one grouped parameter at 6β.1**: K_s = δ · Ka_eff
× (k_hyd/k_prot) × cation-specific factor. K_s is calibrated against
LSV+CP K⁺ data at the smoke voltage. Other parameters held fixed
or sensitivity-swept. Cation-series validation (6β.2) tests
transferability of K_s scaling across cations using literature-derived
β_M only.

**Re #11 (pKa/field model still hand-wavy).** Accept. v9 §7 writes
the explicit formula:

```
ΔpKa(σ_S, r_M, δ_OHP, ε_OHP) =
  − (e · σ_S · r_M) / (k_B · T · ε_OHP · ε_0)
  + β_M
       
Equivalently:
log10(Ka_M_eff / Ka_M_bulk) = −ΔpKa = (e · σ_S · r_M) / (k_B · T · ε_OHP · ε_0)
                                       − β_M

with:
  σ_S = stern_coeff · ψ_S   (signed: cathodic σ_S < 0)
  r_M = cation Stokes radius
  δ_OHP = OHP layer thickness (Bohra 2019, 0.4 nm)
  ε_OHP = effective OHP dielectric (typically 6-10 for compact
          layer; v9 uses 8 as Bohra-aligned prior)
  β_M = cation-specific solvation contribution (Singh 2016; only
        cation-specific tunable).

Driver: σ_S includes the hydrolysis correction Γ_hat — but at
the smoke voltage that's a self-consistent fixed point, not a
chicken-and-egg problem (Newton handles it).

Sign check: at cathodic σ_S < 0 → Ka_M_eff > Ka_M_bulk → pKa shift
DOWN → more hydrolysis at cathode. Matches the deck.
```

**Re #12 (smoke plan too ambitious — split into gates).** Accept.
v9 6β.1 is now staged in **4 gates**, each independently passable:

```
Gate 1: Build/form test (1-2 days)
  - h_index/role refactor of forms_logc_muh.py + water_ionization.py
  - Existing 549 fast + 10 slow tests pass.
  - 4sp K2SO4 + dynamic-K + analytic-SO4 stack assembles.

Gate 2: λ=0 dynamic-K equilibrium smoke (3-5 days)
  - V_RHE = -0.40 V, C_S = 0.10, L_eff = 16 µm, K2SO4.
  - 4sp dynamic stack converges to within semantic tolerance of
    Phase 6α 3sp+Boltzmann-K stack.
  - TestDynamicKplusAnalyticSO4MatchesAnalyticBaseline passes.

Gate 3: Γ-only manufactured source test (1-2 days)
  - Γ as global scalar; impose a fixed R_inj source at boundary;
    verify it correctly increases c_H by the predicted ratio.
  - One-cell unit test for sign convention (R3#5 carry-forward).

Gate 4: Finite-rate hydrolysis at one voltage (5-7 days)
  - V_RHE = -0.40 V, λ ∈ {0, 0.5, 1}.
  - Newton converges; Γ_MOH > 0; surface pH drops in right direction;
    packing < 1; predicted-vs-realized Δ ln R within 30%.
  - C_S sensitivity {0.05, 0.10, 0.20} at λ ∈ {0, 1}.

If all 4 gates pass → 6β.1.b: full V_RHE grid + cation series.
If any gate fails → diagnose, re-plan, possibly request another
GPT round.
```

The 6β.1 scope as previously written (full 13-V grid + cation
series in one phase) is too aggressive. v9 narrows 6β.1 to gate 1+2
+ smoke at one voltage; full sweep is 6β.1.b; cation series 6β.2.

---

## 2. v9 architecture summary

| Item | v8 (R3) | v9 (R4) |
|---|---|---|
| Steady-state H+ source | R_f = R_r → 0 (broken) | R_net = R_des = k_des·Γ_MOH (nonzero turnover) |
| Boundary mass balance | "M+ flux in = ∂tΓ + R_prot" (wrong) | M+ sink = R_net; Γ inventory = R_net − R_des; H+ source = R_net |
| Γ plumbing | "Boundary scalar Function" (vague) | Single global R-element scalar (coupled via ds) |
| logc_muh species roles | Unaware (will hard-fail w/ 2 z=+1) | Explicit `h_index` role, built before hydrolysis |
| λ=0 equivalence | Byte-equivalent (impossible) | Semantic tolerance on observables |
| Stern correction nondim | `+(δ/L)·c_MOH_hat` (wrong scale) | `+Γ_hat` (with Γ_hat scale matching stern_coeff scale) |
| pKa formula | "Singh-style f(ψ_S)" (vague) | Explicit `(e·σ_S·r_M)/(k_B·T·ε_OHP·ε_0) − β_M` |
| Calibration scope | All ~7 tunables at 6β.1 | Just K_s grouped param; others frozen at literature priors |
| Smoke phasing | One big smoke (13 V) | 4 gates: build / λ=0 dyn / Γ source test / 1-V hydrolysis |
| 6β.1 scope | Full sweep + cation series | Gate 1+2 + smoke at one V_RHE |

**Implementation prereqs (must land BEFORE hydrolysis work):**

1. h_index role refactor.
2. 4sp dynamic K+ + analytic SO4 stack feasibility verification.
3. Global-scalar R-element Γ in mixed function space.

---

## 3. Continued critique prompt

Round 4 of 5. v9 is the first version where the architecture
*should* work. Push back on:

1. **Whether the desorption removal path is physically defensible**
   for a polarized OHP. MOH⁰ is neutral so has no electromigration,
   but it's still hydrogen-bonded into the inner-Helmholtz solvent
   structure. Is k_des = 10⁶ /s a defensible literature anchor or
   am I making it up?
2. **Whether the Singh-style pKa formula** with ε_OHP = 8 and the
   given form is actually what Singh 2016 uses, or whether I'm
   conflating it with a different formulation. If the latter, what
   does v9 §7 actually need?
3. **The K_s grouped-parameter calibration scope** — is fitting just
   one composite parameter at 6β.1 actually enough to decide
   whether the architecture is sound, or do I need at least k_des
   sensitivity too?
4. **Gate 2 feasibility risk** — is there any reason to expect 4sp
   dynamic K+ at cathodic V_RHE = -0.40 V to converge cleanly when
   the analogous 4sp dynamic ClO4- at anodic doesn't, or is this a
   load-bearing risk that may require its own continuation work?
5. **Anything else load-bearing.**

This is the second-to-last round. APPROVE if v9 is ready to
execute as scoped; otherwise final issues stay in the unresolved
ledger.

Same numbered format. Verdict line at the end:
  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
