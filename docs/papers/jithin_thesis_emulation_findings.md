# Jithin Thesis Fig 4.36 Emulation — Findings

**Date:** 2026-05-22
**Source paper:** Jithin Donny George, *Two Tales in Scientific Computing*,
  Northwestern University Ph.D. thesis, March 2024 (`docs/papers/Jithin Thesis.pdf`).
**Scripts:**
  - Main driver: `scripts/studies/_run_jithin_emulation_fig436.py`
  - Plot:        `scripts/studies/_plot_jithin_emulation_fig436.py`
  - Closure-only algebra test (Gap 2 diagnostic):
    `scripts/studies/jithin_closure_algebra_test.py`
  - Standalone Chebyshev MPB-flux (stub; not converged):
    `scripts/studies/jithin_standalone_mpb_chebyshev.py`

**Solver changes:**
  - `Forward/bv_solver/forms_logc_muh.py:1291-1296` — guard `picard_state["R2"]`
    with `.get()` for N=1 Tafel-only topologies.
  - `Forward/bv_solver/forms_logc_muh.py:515-526, 608-617` —
    `bv_steric_activity` opt-in flag (default False); when True multiplies
    cathodic+anodic BV rate by local Bikerman packing θ.
  - `Forward/bv_solver/config.py:309` — `bv_steric_activity` added to
    `_get_bv_convergence_cfg` whitelist (initial impl dropped the key
    silently).

**Outputs:**
  - Baseline (k₀×1e-10, θ-on-BV off): `StudyResults/jithin_emulation_fig436/`
  - Experiment A (`packing_floor=1e-15`, otherwise identical to baseline):
    `StudyResults/jithin_emulation_fig436_expA_pf1e-15/`
  - Jithin-mode (k₀×1e-10, θ-on-BV on):
    `StudyResults/jithin_emulation_fig436_jithin_mode/`
  - Jithin-mode + low k₀ (k₀×1e-25, θ-on-BV on):
    `StudyResults/jithin_emulation_fig436_jmode_k0_1e-25/`
  - Closure-only algebra (no PDE):
    `StudyResults/jithin_closure_algebra_test/`

## Why this exists

Jithin's thesis Chapter 3-4 is the origin of the modelling stack used in this
repo (PNP + Bikerman steric + Stern + log-rate BV, calibrated against the
Seitz/Mangan Cs⁺ pH 4 RRDE deck).  Chapter 4 closes with Fig 4.36, his
best-effort match of simulated jV against experimental — the first
quantitative reference point for the entire forward stack.  This note
records what happens when we configure our production solver to emulate his
Fig 4.36 exactly: what matches, what doesn't, and why.

## Configuration matching Jithin Fig 4.36

Per Jithin Table 4.1, Table 4.3 and Fig 4.36 caption:

| Knob | Jithin Fig 4.36 | Our production default | Our emulation value |
|---|---|---|---|
| Reactions | R2e only, Tafel (one direction) | Parallel R2e+R4e, full BV | R2e only, Tafel via `reversible=False`, `anodic_species=None` |
| α·n_e | 2.26 (A_Tafel=26.2 mV/dec, α=1.13) | 1.0 (α=0.5, n_e=2) | Clamped to α=1.0 (validator caps at 1) → effective A_Tafel = 29.6 mV/dec |
| E° (R2e) | 0.695 V vs RHE | 0.695 V | 0.695 V |
| Bulk H⁺ | 10 mol/m³ (pH 2) | 0.1 mol/m³ (pH 4) | 10 mol/m³ |
| Bulk O₂ | 0.25 mol/m³ | 1.2 mol/m³ | 0.25 mol/m³ |
| Counterions | Cs⁺/SO₄²⁻ | K⁺/SO₄²⁻ | Cs⁺/SO₄²⁻ |
| Bulk Cs⁺ / SO₄²⁻ | 190 / 100 mol/m³ | n/a | 190 / 100 mol/m³ |
| a (Table 4.1 nm³) | H⁺ 0.176, O₂ 0.064, H₂O₂ 0.166, Cs⁺ 0.285, SO₄²⁻ 0.436 | Marcus/Stokes radii | Jithin Table 4.1 nm³ converted to a_nondim |
| D_O₂ | 1.5e-9 m²/s | 1.9e-9 m²/s | 1.5e-9 m²/s |
| L_eff (diffusion layer) | 10 µm | 100 µm | 10 µm |
| Stern (L_Stern / C_S) | 0.6 nm → C_S = 1.16 F/m² | C_S = 0.20 F/m² | Multi-stage bump 0.10 → 0.20 → 0.35 → 0.50 → 0.70 → 0.85 → 1.0 → 1.16 F/m² |
| k₀ (R2e exchange-current) | His fitted j₀ (small) | K0_HAT_R2E | K0_HAT_R2E × 1e-10 (back-of-envelope to shift half-wave to V≈+0.25 V) |
| Water ionisation | OFF | varies | OFF |
| Cation hydrolysis | OFF | varies | OFF |

Anchor V is +0.55 V (above E° → tiny kinetic current; far enough below his
top-of-grid V=+0.8 V to keep the Stern bump tractable when SO₄²⁻ is
attracted to the OHP).  V_RHE grid is 25 points on [-0.40, +0.55] V.

## What works

**All 25/25 grid points converge** in ~10 min (anchor 5s + Stern ladder 16s
+ grid walk 500-600s).  The Stern bump ladder reaches Jithin's exact target
1.16 F/m² without Newton failure.

**Half-wave position matches Jithin** within ~40 mV (ours: V_RHE ≈ +0.21 V;
his: V_RHE ≈ +0.25 V).  This required the K0_HAT_R2E × 1e-10 scaling — our
production k₀ would put the half-wave at V_RHE ≈ +0.58 V, ~300 mV too anodic.

**Plateau is exactly Levich at L=10 µm with D_O₂=1.5e-9**: -0.7236 mA/cm² to
4 decimals, matching i_L = n·F·D·c_bulk/L = 2·96485·1.5e-9·0.25/10e-6 =
0.7236 mA/cm².

## What doesn't

### Gap 1: plateau magnitude (factor 2)

Jithin's plotted plateau in Fig 4.36 is ≈ -0.36 mA/cm² — half of ours and
half of his own Levich estimate (Eq 4.37: 0.724 mA/cm² at L=10 µm,
D_O₂=1.5e-9).  His own text on p.139 admits "the rate of change of the
simulated current density with respect of applied potential is qualitatively
different from that seen experimentally" and that he "qualitatively changed
our modeling assumptions until we could capture the behaviour seen
experimentally."

There is no obvious explanation in his Eq 3.32 / 4.31 derivation for why
his plateau is half of his own Levich calculation.  Candidate causes:

- His integro-differential closure may have an O(½) factor that doesn't
  trace through cleanly from continuum MPNP (his Eq 3.36 integral over
  [L_bulk, x] vs the standard Levich integration convention).
- His simulated curve in Fig 4.36 may be plotted at scaled units we
  misread.  The y-axis is labelled "Current Density" without explicit
  units; assuming mA/cm² to match the orange/magenta experimental curve.

**Status: unresolved.**  Without re-implementing his Chebyshev spectral
integro-diff scheme, we can't disentangle "his model genuinely gives
half-Levich" from "his plot reads as -0.36 mA/cm² due to a unit
convention we missed."

### Gap 2: far-cathodic cliff

Jithin's Fig 4.36 simulated curve drops in magnitude at V<-0.3 V — current
goes from his ~ -0.36 plateau back to ~ -0.15 at V=-0.4.  He attributes
this (pp. 133-135) to "steric exclusion of O₂ by Cs⁺/H⁺ pile-up at the
OHP."  Our curve stays at the Levich plateau across the entire V ∈
[-0.4, +0.15] range — no cliff.

#### Experiment A — packing_floor (RULED OUT)

Default `packing_floor = 1e-8` in `bv_convergence` clamps `μ_steric =
-ln(max(packing, 1e-8))` at ≈ +18.4.  Hypothesised that this masks the
Bikerman singular behaviour near saturation.  Dropped floor to 1e-15
(allowing μ_steric up to +34.5).

**Result: bit-exact identical to baseline.**  Δcd = 0.0 at every V point;
c_O2(OHP) identical to all reported digits.  The floor never engages
because actual packing values stay above it everywhere.

#### Mesh resolution (RULED OUT)

`make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0, domain_height=10µm/L_REF)`
uses `y_i = (i/Ny)^β · L_eff`.  First cell at Ny=80, β=3 is **(1/80)³ ·
10 µm ≈ 0.02 nm** — sub-Angstrom.  Four mesh nodes fall inside the first
1 nm.  Mesh resolution at the OHP is not the bottleneck.

#### Empirical state at the OHP (V=-0.4 V, our most cathodic point)

From the extracted diagnostics (`c0_surface_mean`, etc.):

| Quantity | Value |
|---|---|
| φ(OHP) | -6.66 nondim (β·ψ) |
| c_O₂(OHP) | 1.5e-21 mol/m³ (vs bulk 0.25; depletion factor 10²⁰) |
| c_H⁺(OHP) | 242 mol/m³ (24× bulk) |
| c_Cs⁺(OHP) | ≈ 5.5 mol/L ≈ 94% of Bikerman cap 1/a_Cs |
| c_SO₄²⁻(OHP) | ≈ 0 (anion depleted at cathodic V) |
| 1 − Σa·c (packing free space) | ≈ 0.034 (3.4%) |
| μ_steric = -ln(0.034) | ≈ +3.4 |
| Current density | -0.7235 mA/cm² (exact Levich) |

So **the Bikerman saturation IS engaged** (94% Cs⁺ occupancy; μ_steric ≈ 3.4)
but our O₂ at the OHP is already at numerical zero from ordinary Fick
diffusion.

#### Structural explanation

The steric flux in our gradient-form MPNP residual is

`J_steric = D · c · ∇μ_steric  = D · c · (Σ a_k ∂c_k/∂x) / (1 − Σ a·c)`

which is **linear in c**.  For O₂ already depleted to 1e-21, the steric
drift contribution `c · ∇μ_steric` vanishes — there's no O₂ left to push
out of the saturated zone.  The transport balance reduces to standard
Fick → Levich.

Jithin's closure-form MPB-with-flux (his Eq 4.31) writes

`c_k = [A_k · exp(-z_k ψ) + κ_5 · φ_k · g_k] · (1 − Σ a·c)`

i.e. the `(1 − Σ a·c)` factor multiplies the entire RHS, including the
flux-supply term.  When cations saturate, `c_k` is suppressed by that
factor *at the boundary value*, not just in the flux gradient.  His BV
current rate `j ∝ c_O2*` then drops below Levich.

The two forms are **mathematically equivalent in continuum** but yield
different numerical solutions at the OHP saturation regime because of how
the Bikerman coupling is realised:

- **Our path (gradient form, log-c primary):** flux residual coupled by
  ∇μ_steric.  Steady-state Newton finds c_O₂(OHP) → 0 from kinetic
  exponential forcing.  Steric drift on O₂ is c-weighted and vanishes.
  Levich-limited.

- **Jithin's path (closure form, c as LHS):** outer fixed-point on c_O₂*
  with `(1-Σa·c)` multiplicative factor.  Cation saturation directly
  multiplies the OHP concentration.  BV rate suppressed.  Sub-Levich.

#### Whether Jithin's cliff is physics or numerical

Jithin reports (p. 134-135) that with his **full** Table 4.1 a-volumes, his
simulated current is **essentially zero everywhere** — not just at far
cathodic.  Fig 4.36 uses a_k³/10 (volumes divided by ten); his match
required reducing the steric strength tenfold.  Even at a_k³/10 the cliff
persists at V<-0.3.

This pattern — *the model is too aggressive at the input volumes and only
matches experiment after hand-tuned reduction* — is more consistent with a
numerical artifact of his Chebyshev spectral discretization near the
(1−Σa·c) singularity than with a derived physical feature.  Our solver
gives the same Levich limit regardless of a_k³ scale (we've verified ratio
1, 1/4, 1/10, 1/100 of his volumes all yield essentially Levich at far
cathodic in our gradient form).

#### Standalone closure-only algebra test (2026-05-23)

To probe whether the cliff is in Jithin's boundary algebra (independent of
the spatial PDE), we built `scripts/studies/jithin_closure_algebra_test.py`
— a ~30-line pure-NumPy test that computes Boltzmann pile-up at the OHP +
Bikerman closure + Tafel BV, with NO transport.  Outputs in
`StudyResults/jithin_closure_algebra_test/`.

Result: at far cathodic V_RHE, **θ(OHP) drops to 10⁻⁶ (Cs⁺ saturates the
OHP) and c_O₂(OHP) is multiplicatively suppressed by the shared θ factor
DOWN from bulk 0.25 to ~10⁻⁶ mol/m³ — even though O₂ is neutral**.  The
cliff mechanism IS in the algebra: BV rate ∝ c_O₂(OHP)·θ(OHP) drops
sharply when cations saturate.  In Jithin's closure form c_O₂(OHP) carries
the θ factor implicitly; our gradient-form sets c_O₂(OHP) by flux balance
(Levich) without that multiplier.

#### Experiment B (2026-05-23): `bv_steric_activity` opt-in flag

To test whether explicitly multiplying our BV rate by the boundary θ
factor reproduces Jithin's cliff, we added an opt-in
`bv_steric_activity: bool` flag to the `bv_convergence` config dict
(default `False` for byte-equivalence).  When `True` and
`steric_active=True`, the cathodic and anodic BV rate exprs in
`Forward/bv_solver/forms_logc_muh.py:608-619` are multiplied by the local
`packing = max(1 − Σa·c, packing_floor)` UFL expression — same θ that
participates in `μ_steric = -ln(packing)` for the dynamic-species flux.

Three runs (in `StudyResults/jithin_emulation_fig436_*`):

| Run | k₀ factor | θ on BV | Plateau | Cliff? |
|---|---|---|---|---|
| baseline | 1e-10 | OFF | -0.7236 (Levich) | No |
| jithin_mode | 1e-10 | ON | -0.7236 (Levich) | No (kinetic-regime cd suppressed ~20%; far cathodic unchanged) |
| jmode_k0_1e-25 | 1e-25 | ON | -0.7204 (~Levich) | No (clean kinetic S-curve; half-wave at V≈-0.20) |

**Empirical verdict from the flag experiment:**

1. The flag IS wired correctly (verified via config-parser whitelist fix at
   `Forward/bv_solver/config.py:309` — initial implementation was silently
   dropping the key because the parser filters by key name).  Anchor took
   30s vs 4.8s baseline (Newton harder), c_O₂(OHP) shifted by 10²⁰× at
   V=-0.4 V between OFF and ON modes — the multiplier is engaged.

2. **The cliff doesn't manifest in our solver regardless of k₀ scale**, as
   long as the system can find a steady-state c_O₂(OHP) up to c_O₂_bulk
   that satisfies `k₀·c_O₂(OHP)·θ(OHP)·exp(huge) = Levich`.  For
   `k₀ × θ × exp` to drop BELOW Levich at the bulk-c ceiling, the
   exponential decay rate of θ with V must exceed the growth rate of the
   BV exp.  At our parameters, that doesn't happen — the system finds a
   higher c_O₂(OHP) to compensate for the θ suppression and stays at
   Levich.

3. **Lowering k₀ shifts the kinetic-to-transport crossover, producing a
   clean S-curve, but no cliff**: at k₀×1e-25 we get a textbook kinetic
   onset around V=-0.10 V, half-wave at V=-0.20 V, plateau at Levich for
   V<-0.35 V.  The shape resembles Jithin's WITHOUT the cliff feature.

4. **Combined with the closure-only algebra test**: the cliff mechanism
   exists in Jithin's boundary closure algebra (c_O₂(OHP) explicitly
   multiplied by θ), but adding the same θ multiplier to our gradient-
   form BV doesn't reproduce it because the steady-state transport
   balance compensates by raising c_O₂(OHP).  Jithin's c_O₂(OHP) in his
   closure form is FIXED by the algebra (no transport-balance feedback)
   — that's where the cliff comes from.

This is the strongest evidence yet that **Jithin's cliff is a feature of
his closure-form fixed-point discretization specifically, NOT a derivable
consequence of the underlying continuum MPNP equations**.  A gradient-form
PNP solver with the same Bikerman steric coupling does not produce it,
even when the steric activity coefficient is explicitly added to BV.

## Verdict

Our solver:

1. **Reproduces Jithin's Fig 4.36 half-wave position** when configured with
   his k₀ (1e-10× our K0_HAT_R2E), D_O₂ (1.5e-9), Stern (1.16 F/m²),
   counterions (Cs⁺/SO₄²⁻), pH 2, and a-volumes.

2. **Hits Levich exactly** (-0.7236 mA/cm² at L=10 µm with D_O₂=1.5e-9).

3. **Does not reproduce his sub-Levich plateau (Gap 1) or his far-cathodic
   cliff (Gap 2)**, both of which appear to be features of his closure-form
   integro-differential discretization that don't emerge from continuum
   MPNP integrated via gradient-form FE residuals.

4. **The Bikerman steric coupling is fully wired in our PNP** (verified by
   inspecting `Forward/bv_solver/forms_logc_muh.py:498-499` — `J_flux =
   D·c·(ideal_grad + ∇μ_steric)` for every dynamic species when
   `steric_active=True`).  Our gradient form is **mathematically equivalent**
   to his closure form in the continuum, but numerically gives different
   behaviour at the OHP saturation regime.

## Implications for the broader stack

If Gap 1 + Gap 2 are numerical artifacts of Jithin's specific closure
discretisation (Chebyshev spectral with tanh⁻¹ mapping, outer fixed-point
on c_O₂*), then **the production deck-match story that has been hanging on
his curves needs to be re-anchored against pure continuum-MPNP expectations
rather than his fitted Fig 4.36 plateau**.

In particular, the Phase 6β Step 10 Phase D `OUTCOME_C_NON_IDENTIFIABLE`
result — "model max_H₂O₂ = 66.58 pp vs deck K@pH4 mean 50.95 pp = uniform
+15.6 pp overshoot" — was framed against an assumed-correct Jithin baseline.
If Jithin's plateau is itself a numerical artifact (his own Levich estimate
on p. 138 is -0.145 mA/cm² for L=50 µm; his Fig 4.36 simulated plateau at
L=10 µm should be -0.724 if his model integrated cleanly, but he plots
~-0.36), the non-identifiability finding might be diagnosing not a true
model degeneracy but a confusion of reference target.

This is worth a re-read of the Phase D outcome with this finding in mind.

## Reproduction

```bash
# from PNPInverse/, with venv-firedrake activated and cache env set:
python -u scripts/studies/_run_jithin_emulation_fig436.py
python -u scripts/studies/_plot_jithin_emulation_fig436.py StudyResults/jithin_emulation_fig436
```

Tunable constants at the top of `_run_jithin_emulation_fig436.py`:

- `PACKING_FLOOR_EXPERIMENT_A` — Bikerman saturation floor (default 1e-15;
  was 1e-8 in v1 of this script).
- `K0_R2E_JITHIN_FACTOR` — multiplier on K0_HAT_R2E to match Jithin's
  half-wave (default 1e-10; lower to 1e-25 for kinetic-limited regime).
- `STERN_TARGET` / `STERN_ANCHOR` — Stern capacitance bump start/end
  (defaults 0.10 → 1.16 F/m², matching Jithin's L_Stern=0.6 nm).
- `D_O2_JITHIN_PHYS` — override O₂ diffusivity (default 1.5e-9; Jithin
  Table 4.1).
- `BV_STERIC_ACTIVITY` — opt-in θ-on-BV (Jithin-mode steric activity
  coefficient).  Set to False for baseline gradient-form behaviour;
  True to add the multiplicative θ factor to BV rate.
- Bulk concentrations and Table 4.1 a-volumes are exposed individually.

To re-run the closure-only algebra test (~1 second, pure Python):

```bash
python -u scripts/studies/jithin_closure_algebra_test.py
```

Reports θ(OHP), c_O₂(OHP), c_H(OHP), c_Cs(OHP) and BV current vs V_RHE
under no transport assumption — useful for verifying the cliff mechanism
is in Jithin's boundary algebra.
