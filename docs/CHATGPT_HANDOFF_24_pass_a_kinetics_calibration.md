# ChatGPT Handoff 24 — Pass A landed; kinetics calibration & cathodic-peak gap

Date: 2026-05-09
Branch: `fast-realignment-2026-05-08`
Author: Claude Opus 4.7 (1M context) for Jake

## What's new since prior handoffs (HANDOFF 22 / 23)

Two new orchestrators landed in `Forward/bv_solver/`:

* `solve_anchor_with_continuation` (Phase 5γ MVP, prior session) — k0
  geometric ladder from a small floor up to production, with adaptive
  midpoint insertion on Newton failure. Solves the Phase 5α
  convergence wall at V_RHE = +0.55 V on the multi-ion stack.
* `solve_grid_with_anchor` + `PreconvergedAnchor` (this session, Phase
  5γ post-MVP, P1+P2+P3) — takes a converged anchor and walks any
  V_RHE grid by warm-stepping outward (closest-first) without any
  cold-starts or z-ramps.

These together replace the C+D path
(`solve_grid_per_voltage_cold_with_warm_fallback`) for the multi-ion
parallel-2e/4e stack, where C+D's per-V cold-start fails 13/13 around
V_RHE ≈ +0.55 V. Tests: 95 fast + 5 slow + 27 phase-5α regression
(all green).

Three drivers built and run on the production stack:

1. `scripts/studies/pass_a_grid_driver_csplus_so4.py` — Pass A on
   V_RHE ∈ {+0.10, +0.20, …, +0.80} V (8 points). **8/8 converged.**
2. `scripts/studies/k0_r4e_ratio_sweep_csplus_so4.py` — sweep over
   K0_R4e/K0_R2e ∈ {1.0, 1e-6, 1e-12, 1e-18, 1e-24, 1e-30}. **48/48
   converged** (6 ratios × 8 V).
3. `scripts/studies/mangan_full_grid_csplus_so4.py` — promising ratios
   {1e-18, 1e-24} on the deck-aligned page-15 grid `linspace(-0.40,
   +0.55, 25)`. **50/50 converged.** This driver also wires
   `assemble_rrde_observables` so ring-side `j_ring`,
   `S_H2O2_percent`, `n_e_rrde`, and `surface_pH_proxy` are captured
   per-voltage.

Outputs: `StudyResults/fast_realignment_2026-05-08/{pass_a_grid,
k0_r4e_ratio_sweep, mangan_full_grid}/` (per-ratio JSON, summary.json,
overlay PNGs).

## Production stack we are running

| Component | Setting |
|---|---|
| Dynamic species (3) | O₂, H₂O₂, H⁺ (`THREE_SPECIES_LOGC_BOLTZMANN`) |
| Multi-ion | `multi_ion_enabled=True` |
| Boltzmann counterions (analytic, Bikerman steric) | Cs⁺ + SO₄²⁻ (`I = 0.3 M` Cs₂SO₄, Ruggiero §2) |
| Reactions | parallel R_2e + R_4e (Ruggiero parallel topology) |
| E_eq | R_2e = 0.695 V, R_4e = 1.23 V vs RHE (Ruggiero §1 Eqs 1-2) |
| α | α_R2e = 0.627, α_R4e = **0.5 (placeholder, M4 deferred)** |
| K0 | K0_R2e = 2.4e-8 m/s, K0_R4e = K0_R2e × ratio (sweep variable) |
| Formulation | `logc_muh` (μ_H = u_H + e_m·z·φ) |
| BV | log-rate (`log_rate=True`) |
| Stern compact layer | `stern_capacitance_f_m2 = 0.10` |
| Initializer | `debye_boltzmann` (composite-ψ + multispecies-γ) |
| Mesh | graded rectangle Nx=8, Ny=80, β=3.0 |
| Clips | `exponent_clip=100.0`, `u_clamp=100.0` |
| Bulk concs | C_O2 = 1.2 mol/m³, c_H⁺ = 0.1 (pH 4), Cs⁺ = 199.9, SO₄²⁻ = 100 mol/m³ |
| Anchor strategy | `solve_anchor_with_continuation` at V_RHE=+0.55 V; ladder = (1e-12, 1e-9, 1e-6, 1e-3, 1.0); typical history ends with `(1.0, 'fail') → (0.0316, 'ok') → (1.0, 'ok')` (one adaptive midpoint insert) |
| Grid orchestrator | `solve_grid_with_anchor` (warm-walk closest-first); `n_substeps_warm=8`, `bisect_depth_warm=5` |
| Catalyst | placeholder; deck = CMK-3 (carbon RRDE) per Ruggiero/Mangan |
| RRDE | N_collection = 0.224 (Ruggiero §2) |

## Key finding 1 — Three-regime structure of the K0_R4e/K0_R2e ratio

Sweep at V_RHE ∈ [+0.10, +0.80] V revealed:

* **Ratio ≥ 1e-12** — R_4e dominates. cd flat at -0.0899 mA/cm²
  across the band; pc gross R_2e ~10⁻⁹ to 10⁻³ mA/cm². No Butler shape;
  H⁺ floor saturated by R_4e at every voltage.
* **Ratio = 1e-18** — transition / Mangan-like. Butler shape emerges.
  cd ramps from -0.0899 to ~0 over the anodic shoulder. Selectivity
  S_H₂O₂ ≈ 100% deep cathodic, decreasing smoothly to 41.5% at +0.55V.
* **Ratio ≤ 1e-24** — R_4e fully off. cd ≡ pc (R_2e is the only
  active channel). Pure 2e Butler curve. Output identical at 1e-30.

**Why so extreme?** The structural cause is the (E_eq_R4e − E_eq_R2e)
gap of 0.535 V, which is ~21 nondim units / V_T. In the cathodic
exponential, R_4e's `exp(α·n_e·|η|)` factor outruns R_2e's by 1e20
just from this E_eq separation when α and n_e go in. With equal K0,
R_4e wins by 1e20. To compensate, K0 must be reduced 1e18-1e22.
Physically extreme as a raw kinetic gap, suggesting α_R4e=0.5 is also
miscalibrated.

## Key finding 2 — Full deck-aligned curves

V_RHE ∈ [-0.40, +0.55] V (Mangan page-15 grid, 25 points). At ratio
1e-18:

| V_RHE | cd (mA/cm²) | pc disk (mA/cm²) | j_ring (mA/cm²) | S_H₂O₂ (%) | pH_surf |
|---|---|---|---|---|---|
| -0.400 | -0.0899 | -0.0899 | +0.0201 | 100.0 | **14.14** |
| -0.083 | -0.0899 | -0.0899 | +0.0201 | 100.0 | 11.03 |
|  0.000 (≈) | -0.0899 | -0.0899 | +0.0201 | 100.0 | 10.25 |
| +0.273 | -0.0898 | -0.0832 | +0.0186 | 96.2 | 7.51 |
| +0.392 | -0.0890 | -0.0589 | +0.0132 | 79.7 | 6.39 |
| +0.471 | -0.0851 | -0.0361 | +0.0081 | 59.6 | 5.70 |
| +0.510 | -0.0790 | -0.0260 | +0.0058 | 49.5 | 5.37 |
| +0.550 | -0.0661 | -0.0173 | +0.0039 | 41.5 | 5.06 |

For ratio 1e-24, S_H₂O₂ ≡ 100% across the entire band (R_4e fully
off); cd = pc = -0.0899 in the cathodic regime, ramping up the same
way on the anodic shoulder.

**Match vs deck**:
* MATCH (qualitatively): anodic Butler activation onset around
  V_RHE ≈ +0.3 V; carbon-like decreasing peroxide selectivity moving
  anodic; total cd magnitude (within order of magnitude).
* MISS: the deck shows a **peak in peroxide current at moderate
  cathodic V** (~+0.1 to +0.3 V) followed by decline at very negative
  V. Our model has a flat cathodic plateau, no peak.

## Key finding 3 — Surface pH dynamic range exposes Nernst-thickness mismatch

The diagnostic capture (`c{H_SPECIES_INDEX}_surface_mean`, post-processed
through `compute_surface_pH_proxy`) shows surface pH spans **5.06 to
14.14** across V_RHE ∈ [+0.55, -0.40] at bulk pH = 4. A surface pH of
14 implies c_H⁺_surf is 1e-10 mol/m³ — effectively zero. The deck's
experimental data at pH 4 does not show such extreme surface
alkalinization, since real Nernst layers (~14-16 µm at 1600 rpm by
Levich) and bulk buffering buffer the surface pH closer to bulk.

This is direct empirical evidence that the documented "1.86× short on
plateau" finding from M3a.0 is δ_N (Nernst layer thickness) mismatch.
Our domain's bulk Dirichlet BC sits at the mesh edge with implicit δ
that does not match real RRDE hydrodynamics.

## Sequenced gap analysis (Ruggiero parallel-topology only)

Per the user's instruction, we are NOT adding sequential R_3 (peroxide
reduction) — Ruggiero's analysis is that the deck's mechanism is
parallel-2e/4e only.

That leaves the following ordered gap-closure plan:

1. **RRDE ring observables** — DONE (this session). Plumb
   `assemble_rrde_observables` so disk-vs-ring observables aren't
   conflated. Confirmed: ring observable is just `N × |pc_disk|`, so
   it scales not transforms — but the *selectivity* `S_H2O2_%` is the
   correct deck-aligned y-axis.
2. **Nernst-layer thickness calibration** — NEXT. Diagnose what δ_N
   is implicit in the current mesh + BC, parametrize it, and tune
   against the deck's plateau magnitude. Hydrodynamics: δ ≈
   1.61·D^(1/3)·ν^(1/6)·ω^(-1/2) ≈ 14-16 µm at 1600 rpm. Without this
   the kinetic calibration in step 3 would be fitting α to absorb δ
   error.
3. **ALPHA_R4e + K0 joint Tafel calibration (M4)**. With ring
   observables and correct δ in place, extract the deck's experimental
   Tafel slope at activation onset and use it to pin α_R4e. Then
   sweep K0_R4e/K0_R2e to match deck selectivity at one anchor V. The
   current empirical 1e-18 ratio should land at a much more physical
   1e-3 to 1e-6 range with calibrated α_R4e (~0.7-1.0 instead of 0.5).
4. **Local-pH physics** (M5+ structural). Only if 1-3 don't close the
   cathodic-peak gap. Means relaxing the bulk-Dirichlet H⁺ BC to allow
   surface H⁺ to be lifted above bulk via Stern/diffuse-layer pumping,
   or possibly adding hydroxide explicitly with self-consistent local
   pH. High structural risk.

## Open questions for GPT

1. **Is K0_R4e/K0_R2e ≈ 1e-18 a defensible empirical operating point
   pre-calibration?** Or does the 18-decade gap kill physical
   plausibility even as a placeholder, demanding step 3 (α calibration)
   before any further deck comparison?

2. **Is the surface-pH overshoot to 14 evidence enough that δ_N is
   the dominant gap?** Could there be other mechanisms (e.g., implicit
   bulk-pH BC behavior in `forms_logc_muh`, or a Stern-layer coupling
   that's wrong for H⁺) producing the same extreme surface alkalinization?

3. **Tafel slope extraction methodology** — given the deck's data
   structure (LSV scans + ring at 1600 rpm at multiple pH), what is the
   correct way to pin α_R4e? Per-pH Tafel? Use the disk-only or
   selectivity slope? What's the noise floor?

4. **Is parallel-only really the right topology?** The deck shows a
   peroxide *peak* and decline — parallel-2e/4e alone (without
   sequential reduction) cannot produce this on a flat-pH boundary
   layer. Either we need (a) δ_N correction so the peak appears via
   transport, (b) local-pH physics, or (c) reconsider the parallel-only
   reading. Where does Ruggiero's analysis depend on the absence of R_3?

5. **Convergence-wall structural cause** — why does the production
   stack at +0.55 V need k0 ramped from 1e-12 to 1.0 (12 decades), with
   a midpoint insert at the final rung? Is this a sign of an underlying
   ill-posedness in the multi-ion + Bikerman + Stern coupling that
   would also affect Newton at intermediate V on a thicker δ?

## Files of interest for the reviewer

* `Forward/bv_solver/anchor_continuation.py` — anchor + ladder + new
  `PreconvergedAnchor` dataclass.
* `Forward/bv_solver/grid_per_voltage.py` — `warm_walk_phi`,
  `solve_grid_with_anchor`.
* `scripts/studies/mangan_full_grid_csplus_so4.py` — driver currently
  producing the deck-aligned curves.
* `Forward/bv_solver/rrde_observables.py` — `assemble_rrde_observables`
  + per-component formulas.
* `StudyResults/fast_realignment_2026-05-08/mangan_full_grid/` — the
  data being interpreted.
* `docs/Mangan2025_experimental_alignment.md` and
  `docs/Ruggiero2022_JCatal_source_paper.md` — the deck/source paper
  alignment audit.
* `docs/seitz_mangan_data_folder_audit_2026-05-08.md` — the actual
  experimental data drop audit (K₂SO₄ RRDE, pH 1-6, parallel-2e/4e per
  Ruggiero).

## What we want from GPT

A second-opinion review on questions 1–5 above, with explicit
identification of any *physical* objection to step 2 (Nernst-thickness
calibration) being the right next move, and any *kinetic* concern
about α_R4e=0.5 + K0_R4e=2.4e-8 m/s being miscalibrated together
(rather than either alone). If GPT finds a structural issue we haven't
noticed (e.g., a missing physical mechanism, a sign error in the H⁺
boundary handling, or a topology critique), we want to hear it before
investing days into δ_N + Tafel calibration work.
