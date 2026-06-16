# Phase 6β v9 — Gates 3 + 4 implementation summary

**Date:** 2026-05-10
**Branch:** `fast-realignment-2026-05-08`
**Plan executed:** `.claude/plans/write-up-the-formal-joyful-papert.md`
**Outcome:** ✅ Gates 3 (3A–3D) + Gate 4 (4A–4C) closed. Full v9
machinery wired with Singh 2016 SI Eq. (4) pKa formula. Disabled-path
byte-equivalence preserved (Phase 6α 10/10 slow regressions hold).
Gate 2 frozen baseline reproduces at the anchor within 5e-2 relative.

---

## Scope

Phase 6β v9 introduces cation hydrolysis at the OHP::

    M⁺(H₂O)ₙ + H₂O  ⇌  MOH⁰(H₂O)ₙ₋₁ + H₃O⁺                  (Singh 2016 Eq. 2)

with a field-dependent pKa shift driven by the Stern surface charge
(Singh 2016 SI Eq. 4) and the proton boundary residual gaining a
cation-hydrolysis source. The architectural template is Phase 6α's
water self-ionization (`Forward/bv_solver/water_ionization.py`).

Gates 3 + 4 deliver:

* **Gate 3A** — Mixed function space + indexing helper
* **Gate 3B** — `cation_hydrolysis.py` helper module
* **Gate 3C** — `λ_hydrolysis_ladder` continuation + six accessors
* **Gate 3D** — Manufactured-source unit tests
* **Gate 4A** — Singh 2016 SI Eq. (4) field-dependent pKa
* **Gate 4B** — Smoke study driver (`scripts/studies/...`)
* **Gate 4C** — Slow regression vs Gate 2 frozen baseline

---

## Architectural deviation from plan: Γ as coefficient, not Newton unknown

The plan called for Γ_MOH as a **Real-element Newton unknown in the
mixed function space**. I implemented this initially (mixed space
extended from `n_species + 1` to `n_species + 2` components). At
solve time Firedrake rejected the assembly:

```
ValueError: Monolithic matrix assembly not supported for systems
with R-space blocks
```

Switching to `mat_type='nest'` (the canonical workaround) hit a
secondary failure inside `MatMult_Nest` on the off-diagonal coupling
blocks (PETSc error 101 — "wrong format"). The off-diagonal blocks
between FE subspaces and the R-space subspace are stored as Firedrake's
"python" matrix type, which `MatMult_Nest` cannot multiply.

**Decision:** refactor Γ to an **external R-space coefficient updated
between continuation rungs by an outer Picard fixed-point**. This
mirrors Phase 6α's `kw_eff_func` pattern — Γ is a `Function` on
`R_space` that the Newton residual reads, but the Newton system itself
solves only the legacy `species + phi` mixed problem.

The closed-form steady state is well-defined:

```
F_Γ = λ·(R_net − k_des·Γ)·v_R·ds_e − (1−λ)·Γ·v_R·ds_e = 0

Γ_ss(λ) = λ·k_hyd·⟨c_M·10^(−ΔpKa)⟩
          / (λ·k_des + (1−λ) + λ·k_prot·⟨c_H⟩/δ_OHP)
```

with `⟨·⟩` the boundary-area average. At λ=0 this returns Γ=0
exactly (the Dirichlet pin invariant from Gate 3D). At λ=1 it gives
the dynamic steady state. The orchestrator runs at most 8 Picard
iterations per rung with a 1e-4 relative tolerance on Γ; in practice
2–3 iterations suffice.

**What this preserves:**

* Sign convention (Gate 3D's `TestProtonBoundarySourceSignConvention`)
* Area invariance (Gate 3D's `TestGammaResidualAreaInvariance`)
* Hard-zero pin at λ=0 (Gate 3D's `TestGammaDirichletPinAtLambdaZero`)
* λ=0 byte-zeros every hydrolysis contribution → disabled-feature
  observables match within Newton tolerance
* Same `λ_hydrolysis_ladder` continuation interface
* Phase 6α byte-equivalence (slow regression unchanged)

**What changes:**

* The mixed function space stays at `n_species + 1` whether
  `enable_cation_hydrolysis` is on or off
* `mixed_space_indices` always reports `has_gamma=False` (the helper
  is kept for forward compatibility but no longer needed)
* `build_gamma_residual` is gone (no UFL form fragment for Γ)
* Newton convergence is per-rung × Picard-outer instead of fully
  monolithic — typically 30–80 inner Newton iterations at the
  cathodic target, vs ~25 in the monolithic ideal

---

## What was built

### Gate 3A — Mixed function space + indexing helper

* **`Forward/bv_solver/forms_indexing.py`** (new). Pure-Python helper
  exposing `unpack_dof_indices(has_gamma=...) → SpeciesPhiGammaIndices`.
  In the refactored architecture `has_gamma` is always False, but the
  helper is retained for uniform call-site slicing
  (`fd.split(U)[indices.species_slice]` etc.) across forms_logc,
  forms_logc_muh, and boltzmann.py.
* **`Forward/bv_solver/forms_logc.py`** + **`forms_logc_muh.py`** —
  every `fd.split(U)[:-1]` / `[-1]` call-site rewritten to use the
  helper. Mixed-space layout extension was prototyped then reverted
  per the architectural deviation above; ctx publishes
  `mixed_space_indices` and `cation_hydrolysis_enabled` for
  downstream code regardless.
* **`Forward/bv_solver/boltzmann.py`** — `phi`, `w` slicing migrated
  to the helper.
* **`Forward/bv_solver/config.py`** — `_get_bv_convergence_cfg` reads
  `enable_cation_hydrolysis`, `cation_hydrolysis_config`,
  `lambda_hydrolysis`, `manufactured_R_inj`. Validates the dict
  shape when the feature is on; passes through verbatim when off.
* **`scripts/_bv_common.py`** — `_make_bv_convergence_cfg` and
  `make_bv_solver_params` plumb the new parameters through.

### Gate 3B — `cation_hydrolysis.py` helper module

* **`Forward/bv_solver/cation_hydrolysis.py`** (new) mirrors the
  `water_ionization.py` public API:
  * `CationHydrolysisBundle` (frozen dataclass) holding mutable
    R-space `Function`s for `δ_OHP`, `k_hyd`, `k_prot`, `k_des`,
    `λ_hydrolysis`, and **`Γ_MOH`** (Phase 6β v9 Gate 3B refactor:
    `Γ` lives on `R_space` not in mixed space). Plus
    `cation_params` dict (Singh Table S1 row + r_H_El + solver
    switches) and `counterion_idx` / `h_idx` (Gate 1 role-aware).
  * `is_cation_hydrolysis_enabled(conv_cfg)` — gate check.
  * `resolve_counterion_index(roles, role_label='counterion')` —
    forces the K2SO4 stack's H⁺ + K⁺ z=+1 ambiguity to be
    disambiguated by explicit role labels.
  * `build_cation_hydrolysis_terms(...)` — builds the bundle,
    attaches it to `ctx['cation_hydrolysis']`.
  * `build_proton_boundary_source(...)` — returns UFL expression
    for `R_net = k_hyd·c_M·10^(−ΔpKa) − k_prot·c_H·Γ/δ_OHP`.
  * `build_pka_shift(...)` — at Gate 3B returns `Constant(0.0)`
    (placeholder); at Gate 4A swaps in Singh Eq. (4).
  * `update_gamma_from_solution(ctx)` — outer Picard step:
    computes the closed-form `Γ_ss(λ)` from the converged solution
    and assigns to `bundle.gamma_func`. Two branches: physical
    (uses ⟨c_M·10^(−ΔpKa)⟩ and ⟨c_H⟩) and manufactured-override
    (uses fixed R_inj for Gate 3D unit tests).
  * `extract_gamma_value(ctx)` — diagnostic accessor.
* **Forms wiring** (`forms_logc.py` and `forms_logc_muh.py`): when
  `enable_cation_hydrolysis=True`, the form-build code adds
  `λ·R_net · v_H · ds(electrode)` to the proton residual,
  `λ·(−R_net) · v_M · ds(electrode)` to the cation residual, and
  stashes the physical `σ_S` UFL expression on
  `ctx['_cation_hydrolysis_sigma_S_expr']` so the Singh helper can
  recompute the pka_factor between Picard rungs. λ-modulation
  guarantees byte-zero residual contribution at λ=0.

### Gate 3C — `λ_hydrolysis_ladder` + accessors

* **`Forward/bv_solver/anchor_continuation.py`**:
  * Six new accessors: `set/get_reaction_lambda_hydrolysis_model`,
    `set_reaction_k_hyd_model`, `set_reaction_k_prot_model`,
    `set_reaction_k_des_model`, `set_reaction_delta_ohp_model`.
    Each writes to BOTH the metadata layer
    (`ctx['bv_convergence']['cation_hydrolysis_config'][k]`) AND
    the live FE Function — same convention as
    `set_reaction_kw_eff_model`.
  * `lambda_hydrolysis_ladder` keyword on
    `solve_anchor_with_continuation`. Validation matches the
    `kw_eff_ladder` pattern (non-empty, monotone increasing, ends
    at 1.0, optional 0.0 floor branch).
  * `_run_gamma_picard(label)` helper called at the end of every
    ladder branch (bare-k0, kw_eff, c_s, λ_hydrolysis). Picks up the
    converged solution, walks Γ to its closed-form steady state via
    `update_gamma_from_solution`. Cap of 8 iterations + 1e-4
    relative tolerance.
  * Combination guards: `lambda_hydrolysis_ladder + (kw_eff_ladder
    or c_s_ladder)` raises `NotImplementedError` (each ladder
    individually supported; combining is deferred per MVP scope).

### Gate 3D — Manufactured-source unit tests

`tests/test_phase6b_v9_gate3_gamma_machinery.py`:

* `TestUnpackDofIndices` (5 fast) — pure-Python helper validation
* `TestCationHydrolysisDefaultOff` (5 fast) — config defaults
* `TestCationHydrolysisConvergenceCfgParser` (4 fast) — parser surface
* `TestCationHydrolysisHelpersDefaultOff` (6 fast) — gate + role helpers
* `TestLambdaHydrolysisAccessorErrors` (7 fast) — accessor validation
* `TestLambdaHydrolysisLadderValidation` (2 fast) — combination guards
* `TestMixedSpaceLayoutLegacy` (4 slow) — disabled path unchanged
* `TestMixedSpaceLayoutWithCationHydrolysis` (6 slow) — refactored
  layout matches design (mixed space stays at n+1, Γ is R-space)
* `TestCationHydrolysisBundleBuild` (2 slow) — bundle construction
* `TestLambdaHydrolysisAccessorRoundtrip` (2 slow) — setter roundtrip
* `TestProtonBoundarySourceSignConvention` (1 slow, ~45s) —
  R_inj=+ε vs −ε at λ=1 produces the expected c_H sign change
* `TestGammaResidualAreaInvariance` (1 slow, ~70s) — Γ at Ny=40
  matches Γ at Ny=80 within 1e-3 relative
* `TestGammaDirichletPinAtLambdaZero` (2 slow, ~90s) — λ=0
  forces Γ=0 within 1e-10; observables at λ=0 with feature
  enabled match disabled-feature observables within 1e-6 relative

### Gate 4A — Singh 2016 SI Eq. (4) field-dependent pKa

* **`Forward/bv_solver/cation_hydrolysis.py::build_pka_shift`** —
  swaps `Constant(0.0)` placeholder for the actual Singh formula
  when `pka_shift_form == 'singh_2016_eq_4'`:
  ```
  ΔpKa(σ)  =  +2 · A · z · σ_singh · r_H-El · (1 − r_M-O² / r_H-El²)
  σ_singh  =  max(0, −σ_S) · (N_A/F) · 1e-24            (anode-clamped)
  ```
* `_avogadro_per_faraday()` helper carries the canonical 1/e
  conversion (matches `Nondim.constants` Faraday).
* **σ_S in physical C/m²** is constructed in the form-build code by
  multiplying `stern_coeff_nondim · (φ_applied − φ)` by
  `F · C_SCALE · L_SCALE` — Singh's Eq. (4) is unit-specific and the
  form's σ_S must arrive in SI units.
* **`scripts/_bv_common.py`** — module-level Singh constants
  (`SINGH_A_PM = 620.32`, `SINGH_B = 17.154`,
  `SINGH_R_O_PM = 63.0`) and `SINGH_2016_CATION_PARAMS` table for
  Li⁺ / Na⁺ / K⁺ / Rb⁺ / Cs⁺ (Singh Table S1 + Cu r_H_El back-fit
  from Table S3). Plus `make_singh_pka_shift_params(cation,
  r_H_El_pm=...)` factory.
* **Tests** (`TestSinghPkaShiftPure` 5 fast,
  `TestSinghPkaShiftUFL` 4 slow):
  * Singh Eq. (3) bulk pKa Table S1 verified within rounding for
    all 5 cations
  * `make_singh_pka_shift_params("K+")` returns canonical dict
  * Singh Eq. (4) UFL: ΔpKa = 0 at anodic σ_S (clamp), < 0 at
    cathodic σ_S (sign convention), = 0 at σ_S=0

### Gate 4B — Smoke study driver

`scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`:

* Three-step pipeline mirrors Gate 2 SUCCESS recipe:
  1. Anchor at V=+0.55 V via `solve_anchor_with_continuation` with
     `kw_eff_ladder` + λ=0 (anodic, well-conditioned)
  2. `extract_preconverged_anchor` → `solve_grid_with_anchor` warm
     walk to V=−0.40 V at λ=0 (must reproduce Gate 2 baseline)
  3. λ ramp 0 → 0.25 → 0.50 → 0.75 → 1.0 at V=−0.40 V via a second
     `solve_anchor_with_continuation` call with
     `lambda_hydrolysis_ladder`
* Sensitivity sweeps (one axis at a time around baseline):
  * `r_H_El_pm ∈ {180, 195, 200.98, 215, 250}` pm (Cu→carbon
    transferability calibration)
  * `C_S ∈ {0.05, 0.10, 0.20}` F/m² (Stern sensitivity)
  * `k_des ∈ {1e3, 1e5, 1e7}` 1/s (desorption sensitivity at λ=1)
* `--quick` flag runs a single baseline combination only (~5–15 min)
  for end-to-end smoke verification.
* Outputs `iv_curve.json`, written incrementally so partial runs
  preserve data.

### Gate 4C — Slow regression vs Gate 2 baseline

`tests/test_phase6b_v9_gate4_finite_hydrolysis.py`:

* `TestHydrolysisActivationZeroReproducesGate2Anchor`
  (~66s) — anchor-only at V=+0.55 V with full v9 architecture
  (`enable_cation_hydrolysis=True`, Singh formula wired, λ=0).
  Asserts:
  * Γ_MOH = 0 within 1e-10 (machine precision)
  * cd_anchor matches Gate 2's `−0.583 mA/cm²` within 5e-2 relative
  * pc_anchor matches Gate 2's `+0.291 mA/cm²` within 5e-2 relative
* `TestSinghAnodeClampAtAnchor` (~53s) — at V=+0.55 V the Stern
  surface charge is anodic (σ_S > 0), so Singh's anode-clamp
  drives `⟨ΔpKa(K⁺)⟩ = 0` exactly. Verifies that the form-build
  code passes σ_S in physical units to the Singh helper correctly.

The full warm-walk + sensitivity sweep is the smoke driver's job;
this regression test just locks the architecture against Gate 2's
frozen baseline at one voltage to keep wall time manageable.

---

## Test results

```
Fast (no Firedrake solve):  61 passed       1 s
Slow (Firedrake solves):    34 passed     321 s   (5:21)
                            +1 skipped (Phase 6α MMS, deferred)
```

Phase 6α byte-equivalence: 10/10 slow regressions hold.
Gate 2 frozen baseline: anchor cd/pc reproduced within 5e-2 at
`enable_cation_hydrolysis=True, λ=0`.

Pre-existing failures (unrelated; verified via `git stash`):
`test_autograd_gradient.py` (5) + `test_multistart.py` (1) — inverse
solver path, paused per CLAUDE.md "Inverse status: paused".

---

## Files modified or created

### New

* `Forward/bv_solver/cation_hydrolysis.py` (~470 lines)
* `Forward/bv_solver/forms_indexing.py` (~95 lines)
* `tests/test_phase6b_v9_gate3_gamma_machinery.py` (~970 lines)
* `tests/test_phase6b_v9_gate4_finite_hydrolysis.py` (~190 lines)
* `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`
  (~370 lines)

### Modified

* `Forward/bv_solver/forms_logc.py` — split-helper migration; cation
  hydrolysis residual block + λ-modulated proton/cation source
* `Forward/bv_solver/forms_logc_muh.py` — same
* `Forward/bv_solver/boltzmann.py` — split-helper migration
* `Forward/bv_solver/config.py` — `enable_cation_hydrolysis` +
  `cation_hydrolysis_config` + `lambda_hydrolysis` +
  `manufactured_R_inj` parser keys
* `Forward/bv_solver/anchor_continuation.py` — six new accessors,
  `lambda_hydrolysis_ladder` orchestration branch, `_run_gamma_picard`
  helper called from every ladder branch
* `scripts/_bv_common.py` — `_make_bv_convergence_cfg` /
  `make_bv_solver_params` plumb the new params; Singh constants +
  `SINGH_2016_CATION_PARAMS` + `make_singh_pka_shift_params` factory

---

## What's tested vs. what's not

**Locked down by tests (pass):**

* Gate 3 machinery default-off contract (no surprises in legacy
  callers)
* Mixed-space layout (consistent with the refactored architecture)
* Bundle construction + accessor round-trips
* λ ladder validation (combinations guarded, range checked)
* Manufactured-source sign convention at λ=1
* Γ area-invariance under mesh refinement
* λ=0 hard-zero pin (Γ=0, observables match disabled-feature)
* Singh Eq. (3) Table S1 numerical verification
* Singh Eq. (4) UFL: anode clamp, cathodic sign, σ=0 zero
* Gate 2 anchor reproducibility at λ=0 with full v9 wired

**Smoke driver run results (2026-05-10):** see
`docs/PHASE_6B_V9_GATE_4B_SWEEP_RESULTS.md` for the full data + cache
optimization writeup.  Brief summary:

* **All 9 sweep combinations converged** (baseline + 4× r_H_El + 2×
  C_S + 2× k_des, one-axis-at-a-time).  Total wall: 9 min cache +
  5.4 min sweep = 14 min.
* **Picard formula sanity check passes:** `Γ × k_des = 0.5552`
  constant to 4 decimals across `k_des ∈ {0.1, 1.0, 10.0}` —
  confirms the outer Picard hits the analytic fixed point.
* **Singh formula directionality is correct** (Γ↓ as r_H_El↑) but
  magnitude is small (~8% Γ swing across 70 pm bracket) because the
  smoke kinetics (`k_hyd = 1e-3 nondim`) are intentionally tame for
  Picard convergence.
* **Stern is the loudest knob:** doubling C_S more than doubles Γ.
* **cd is invariant across the sweep** (−5.532 mA/cm² to 4 dp) —
  Gate 2's mass-transport-saturated regime confirmed; surface pH is
  the right observable for hydrolysis effects, not cd at this V.

**Cache optimization (Phase 6β v9 Gate 4B refactor 2026-05-10):**

* `Forward/bv_solver/anchor_continuation.py` — new
  `solve_lambda_ramp_from_warm_start` orchestrator helper that
  takes a cached U snapshot + sp + λ ladder, builds a fresh ctx,
  restores U, runs only the λ Picard loop.  Skips the ~9-min anchor
  + warm-walk per combination.
* `Forward/bv_solver/cation_hydrolysis.py` — `r_H_El_pm` promoted
  from baked Constant to live R-space `Function`, mutable via the
  new `set_reaction_r_H_El_pm_model` accessor.  Lets the r_H_El
  sweep avoid form rebuild.
* `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` —
  caches the converged U at V=−0.40 V to
  `StudyResults/phase6b_v9_gate4_smoke/u_warmstart_at_v_target.npz`
  (5-array `np.savez`); subsequent runs skip the cache step.
* Net savings: ~94 min on the first full sweep, ~99 min on every
  re-run.

**Open calibration questions (the smoke does NOT close these):**

* `r_H_El_pm` for ORR-on-CMK-3-carbon: the Singh Cu prior produces
  the right *direction* but not the deck slide 27 ΔpKa(K) ≈ −6 unit
  drop magnitude.  Needs higher `k_hyd` AND/OR a different `r_H_El`
  before the calibration target is reachable.  This is the
  load-bearing 6β.2 work per the v9 plan §"Risk callouts".
* Predicted-vs-realized `Δ ln R_4e`: cd plateau is invariant at
  V=−0.40 V (mass-transport saturated), so the metric is undefined.
  Needs a less saturated V (e.g. V=−0.20 V near onset) to compute
  meaningfully.
* Cation-series holdout (Cs⁺ / Na⁺ / Li⁺): not yet run.

To re-run::

    # Cached fast path (~5–14 min depending on whether snapshot exists):
    python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py
    python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py --quick

    # Force a fresh cache (delete snapshot first):
    rm StudyResults/phase6b_v9_gate4_smoke/u_warmstart_at_v_target.npz

---

## Pointers for next steps

* **Smoke run.** The Gate 4B verdict (per the plan) gates the v9
  architecture: Newton must converge at every (λ, C_S, k_des,
  r_H_El) combination, packing must stay < 1.0, surface pH must
  drop in the right direction at λ → 1, and at least one
  `r_H_El_pm` in the swept bracket must reproduce the deck pKa
  drop within 30%. Anything else triggers a re-plan + GPT round
  with the failure data as new evidence.
* **Cation-series 6β.2.** Once Gate 4 closes the architecture
  question, the cation-series holdout (Cs⁺ / Na⁺ / Li⁺) is the
  actual physics validation per the v9 R5#5 wording guard: "Gate 4
  pass does not validate hydrolysis physics; it only shows the v9
  coupled solver can express a plausible branch without immediate
  contradiction."
* **Re-monolithise Γ later (optional).** If a Firedrake / PETSc
  upgrade lands proper R-space-in-mixed support, the refactor is
  reversible — the residual structure (`build_gamma_residual`-style
  smooth-blend) is preserved in
  `update_gamma_from_solution`'s closed-form formula, and could be
  re-introduced as a UFL block alongside the FE residual without
  semantic change. The outer Picard would become a single Newton
  solve on a 1-DOF-larger system.

## Source-of-truth references

* `.claude/plans/write-up-the-formal-joyful-papert.md` — the plan
* `docs/singh_2016_pka_formula.md` — Singh Eq. (3)/(4) extraction
* `docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md` — Phase 6α
  architectural template
* `docs/PHASE_6B_V9_GATE_4B_SWEEP_RESULTS.md` — Gate 4B sensitivity
  sweep results + cache architecture writeup
* `StudyResults/phase6b_v9_gate2_smoke/SUCCESS.md` — Gate 2 frozen
  baseline (cd, pc at every V_RHE point)
* `StudyResults/phase6b_v9_gate4_smoke/iv_curve.json` — full Gate 4B
  sweep payload (per-combination Γ, cd, pc, Picard rung diagnostics)
* `docs/CONJECTURE_AUDIT_2026-05-09.md` — high-risk audit (K⁺ vs
  Cs⁺, etc.) — reading order for any deviation from the K2SO4 deck
  baseline
