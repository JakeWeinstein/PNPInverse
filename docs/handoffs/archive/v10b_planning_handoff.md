# v10b Planning Handoff — Γ_max + k_des + C_S Literature Calibration

**Target reader.** Planner agent (sci-planner, GPT critique loop,
or another Claude session) that will produce the v10b execution
plan. The plan should be substantive (multi-day, multi-phase)
because there are real literature interpretation calls to make.

**Status.** Steps 1–7 of the acceptance-bundle "v10a → E sequence"
have landed. **Step 8 = v10b** is the next milestone and is
mandatory in all routing branches per
`docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` § "v10a
→ E sequence".

**TL;DR.** v9/v10a used smoke values for `Γ_max = 0.047` and
`k_des = 1.0` and a convergence-pinned `C_S = 0.10 F/m²`. v10b's
job is to replace these with literature-anchored values, lock them
into `scripts/_bv_common.py`, run the prerequisite sensitivity +
regression suite, and produce a defensible writeup. Step 7
(`docs/phase6/CMK3_capacitance_literature.md`) already locks
`C_S = 0.20 F/m²`; v10b inherits that recommendation. The hard
calibration work is `Γ_max` and **especially `k_des`** (no obvious
literature anchor exists yet).

---

## 1. Where this sits in the sequence

```
[done]  Step 1  Lock acceptance bundle
[done]  Step 2  v10a Langmuir cap (Γ saturates at Γ_max)
[done]  Step 3  Anchor ladder + warm-walk infrastructure
[done]  Step 4  V_kin selection (v10a' V-sweep → V_kin = −0.10 V)
[done]  Step 5  Phase A.2 (k_hyd × λ ramp at V_kin)
[done]  Step 6  Plumbing-ablation matrix (all 5 ablations pass)
[done]  Step 7  CMK-3 capacitance literature note (C_S = 0.20 F/m²)
[NEXT]  Step 8  v10b — Γ_max + k_des + C_S literature calibration
[future] Step 9  B.2 — densified k_hyd × λ at v10b params
[future] Step 10 Phase D — K-only Δ_β fit
[future] Step 11 Phase E — predictive holdout (4 cations)
```

Reference: `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`
§ "v10a → E sequence" + § Status.

---

## 2. What v10b must deliver

### 2.1. Hard outputs

1. **Literature-anchored numeric values** for `Γ_max`, `k_des`, `C_S`
   with citation chains and derivation steps documented.
2. **Code change** updating `scripts/_bv_common.py` defaults +
   `Forward/bv_solver/cation_hydrolysis.py:GAMMA_MAX_HAT_SMOKE`
   constant (rename or replace).
3. **Regression**: Phase A.2 baseline reproduction at v10b params
   (re-run `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` with
   the v10b defaults; commit the new baseline JSON; replace
   `BASELINE_REPRODUCTION_REL_TOL` reference numbers).
4. **Step 6 plumbing matrix regression** at v10b params (re-run
   `scripts/studies/phase6b_step6_plumbing_ablation.py`; confirm
   all 5 ablations still pass — A0 byte-equivalent to the new
   v10b A.2 baseline).
5. **Sensitivity bracket sweep** across at least `C_S` (the four
   locked points: 0.05, 0.10, 0.20, 0.30 F/m²) and ideally
   across `Γ_max` and `k_des` if the literature points to a
   single value with uncertainty.
6. **Writeup**: `docs/phase6/v10b_calibration_summary.md` (analog
   of `CMK3_capacitance_literature.md` for the three parameters
   together) + acceptance-bundle § Status update.

### 2.2. Pass criteria (suggested; planner can refine)

- All three parameters cited to peer-reviewed sources (or
  explicitly flagged as estimates with uncertainty bracket).
- Phase A.2 at v10b params: 10/10 k_hyd rungs converge cleanly at
  λ=1.0; Picard converges everywhere; mass-balance residual at
  machine precision.
- Step 6 plumbing matrix at v10b params: all 5 ablations pass.
- C_S sensitivity sweep: solver convergence + qualitative
  observable behaviour stable across `{0.05, 0.10, 0.20, 0.30}`.
- Selectivity gap (single-V, V_kin = −0.10): improvement vs A.2
  baseline `single_v_selectivity_gap_pp = +5.09 pp` is *welcome*
  but **not required** (A.2 already flagged this as v10b LOW
  priority; the calibration is about literature anchoring, not
  about hitting the H₂O₂ deck band).

---

## 3. The three parameters — what's known, what's needed

### 3.1. `C_S` — Stern compact-layer capacitance

**Status:** **Recommendation locked at step 7.** `C_S = 0.20 F/m²`
(20 µF/cm²) per the Bohra-Koper-Choi consensus.

**Source of truth:** `docs/phase6/CMK3_capacitance_literature.md`
(citation chain, three Pillai-2024 regimes, four load-bearing
caveats, sensitivity bracket).

**v10b work:**
- **Lock `C_S = 0.20 F/m²`** as the production default. Update
  `scripts/_bv_common.py` callers + driver `STERN_F_M2_BASELINE`
  in `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py`. (The
  driver constant is already 0.20; the issue is the wider call-
  site sprawl — see "Code touch points" §4.)
- **Run sensitivity bracket sweep** at `{0.05, 0.10, 0.20, 0.30}`
  F/m². Two-stage anchor pattern from Phase A.2 / step 6 (anchor
  build at 0.10, runtime-bump to target) is required because of
  the `c_s_ladder + kw_eff_ladder NotImplementedError` in
  `solve_anchor_with_continuation`.
- **Carry forward open asks** from step 7: pull Bohra 2019 EES
  (`10.1039/c9ee02485a`) into `Articles/`; re-derive Risk #5
  σ_S mismatch with Stern-only C_S; decide Yash convention
  (L_Stern + ε_S vs C_S).

### 3.2. `Γ_max` — Langmuir saturation cap on cation-hydrolysis coverage

**Current (smoke):** `GAMMA_MAX_HAT_SMOKE = 0.047 nondim`. Defined
in `Forward/bv_solver/cation_hydrolysis.py:242`. Derivation
(lines 232–239):

```
Γ_max_phys  ≈ 1 / (π · (2.3e-10 m)² · N_A) ≈ 5.6e-6 mol/m²
            (hard-sphere monolayer, r ≈ K⁺ hydrated radius 2.3 Å)
Γ_max_hat   = Γ_max_phys / (C_SCALE · L_REF)
            = 5.6e-6 / (1.2 · 1e-4)
            ≈ 0.047
```

So the smoke value = 1 monolayer of MOH at the OHP using hard-
sphere packing with K⁺ hydrated radius.

**Why it matters:** controls the cap term `λ·F₀/Γ_max` in the
closed-form `Γ_ss(λ)` denominator. At A.2's `k_hyd_route = 1e-1`,
`θ ≡ Γ/Γ_max = 0.998` (cap-saturated). The Phase A.2 record
shows mass-balance residual at machine precision at this value;
moving `Γ_max` should rescale the saturation k_hyd but not break
convergence.

**Literature search targets (planner: research these):**
1. **Sub-monolayer / partial-coverage corrections.** Hard-sphere
   monolayer overestimates coverage if water/co-adsorbate competes
   for OHP space. Look at:
   - Singh 2016 *JACS* `10.1021/jacs.6b07612` — does it give a
     partial-coverage estimate from CV?
   - Cation-specific adsorption at sp² carbon (NOT Hg/Au; the
     metal-coverage literature is much more developed than the
     carbon one).
   - Iamprasertkun 2019 *JPCL* `10.1021/acs.jpclett.8b03523` HOPG
     basal + alkali cations — may give coverage estimates.
   - Bohra 2019 EES `10.1039/c9ee02485a` cation-hydration steric
     model (open ask: pull to Articles/).
2. **Cation-specific hydrated radius** (vs the K⁺ default 2.3 Å).
   `scripts/_bv_common.py:921-925` lists per-cation `r_M_pm`
   (K⁺: 138 pm naked; Cs⁺: 170 pm); the *hydrated* radius is what
   sets Γ_max via hard-sphere packing. Singh 2016 Table S1 likely
   has these.
3. **CMK-3-specific** considerations: pore confinement may
   change effective Γ_max if MOH formation is confined to specific
   surface sites (e.g. defect edges vs basal plane).

**Cross-check via deck data:** Linsey 2025 ACS-CATL deck slide 27
gives per-cation effective pKa near Cu (Li 13.16, Na 11.44, K
8.49, Cs 4.32). If the *equilibrium* hydrolysis at saturation
matches a known surface density, Γ_max can be back-fit.

**Sensitivity bracket** (suggested): `Γ_max ∈ {0.02, 0.047, 0.1,
0.2} nondim` — half-monolayer, hard-sphere monolayer (smoke),
two-monolayer (allows local enrichment), high stretching test.
Planner: pick narrower range if literature settles.

### 3.3. `k_des` — desorption rate of MOH → bulk

**Current (smoke):** `1.0 nondim` (rate units `1/τ_REF`).
Default from `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py:
SMOKE_KINETICS = {..., "k_des_nondim": 1.0, ...}`.

**Why it matters:** appears in the `Γ_ss(λ)` denominator
`λ·k_des + ...`. At λ=1, the k_des term is the *dominant*
denominator contributor in v10a (A.2 record at k_hyd=1e-3:
`denominator_kdes = 1.0`, `denominator_cap = 0.0374`,
`denominator_kprot ≈ 0.025`). So `k_des` directly sets the steady-
state Γ for cap-not-yet-saturated rungs and the F₀/k_des prefactor
for cap-saturated rungs.

**The hard problem.** No obvious peer-reviewed `k_des` value
exists for `M(OH)⁰ → M⁺ + OH⁻` at the polarized OHP. Singh 2016's
field-dependent pKa is an *equilibrium* quantity (`K_eq =
k_hyd/k_des`); rate constants are not directly extractable from
the equilibrium alone.

**Research strategies (planner: pick one or more):**
1. **Detailed balance from `K_eq + k_hyd`.** If a literature
   value for `k_hyd` (forward hydrolysis rate constant) exists,
   detailed balance gives `k_des = k_hyd / 10^(pKa_eff)`. Singh's
   reported `k_hyd` (if any) + pKa table → back-out `k_des`.
2. **Analogous reactions.** Adsorbate desorption rates from sp²
   carbon are documented for `OH*`, `O*`, `CO*` etc. in the ORR /
   CO2R literature. MOH desorption might be analogous to `OH*`
   desorption with cation-stabilization correction.
3. **Diffusion-limited estimate.** If desorption is barrier-less,
   `k_des ~ D_M+/δ_OHP` would be the upper bound. Lower bound
   from `k_BT/h · exp(-ΔG_des/RT)` Eyring with ΔG_des estimated
   from cation-OH binding energy literature.
4. **Treat k_des as a free fit parameter** anchored to a
   plausible prior (e.g. `10^{-1}–10^{+1}` nondim) and let
   Phase A.2 / B.2 update it via the steady-state Γ observation.
   **This is the most honest fallback if the literature is
   silent.**

**Sensitivity bracket** (suggested): `k_des ∈ {0.1, 1.0, 10.0}
nondim` — three decades centered on the smoke value. Planner:
this is wide because the literature is genuinely uncertain.

**Risk flag.** If `k_des` cannot be literature-anchored, v10b
should explicitly mark it as an *engineering choice* with a
documented prior + the bracket sweep as evidence that the
plumbing is robust across plausible values. Phase D's K-only
β fit (step 10) is where `k_des` actually gets data-constrained.

---

## 4. Code touch points

### 4.1. Files the planner's executor will modify

| File | What changes |
|---|---|
| `Forward/bv_solver/cation_hydrolysis.py` | Update or replace `GAMMA_MAX_HAT_SMOKE = 0.047` with the literature value. Keep the smoke constant for backward compatibility OR rename to `GAMMA_MAX_HAT_V10B` with an explicit citation. |
| `scripts/_bv_common.py` | `make_cation_hydrolysis_config` `gamma_max_nondim` default. Also any in-line `stern_capacitance_f_m2=0.10` callers that should switch to 0.20. |
| `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` | `SMOKE_KINETICS` dict (`k_hyd_nondim`, `k_prot_nondim`, `k_des_nondim`, `gamma_max_nondim`). Rename to `V10B_KINETICS` or similar. `STERN_F_M2_BASELINE` already 0.20 — verify. |
| `docs/phase6/v10b_calibration_summary.md` | NEW. Mirrors `CMK3_capacitance_literature.md` structure: citation chain + caveats per parameter + bracket + provenance. |
| `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` § Status | Append v10b landing paragraph. |
| `CLAUDE.md` § "Recent progress" | Mention v10b landing + reference the new summary. Keep CLAUDE.md under 200 lines (currently 199). |
| `tests/test_phase6b_v10a_*.py` | Update reference numbers in `compute_mass_balance_residual_rel` and `baseline_reproduction` thresholds where they reference smoke values. |
| `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.json` | REGENERATE with v10b params (commit the new file). |
| `StudyResults/phase6b_step6_plumbing_ablation/ablation_matrix.json` | REGENERATE with v10b params (commit the new file). |

### 4.2. Existing flags the planner can use

`bv_convergence` flags (from `Forward/bv_solver/config.py:_get_bv_convergence_cfg`):

- `enable_cation_hydrolysis: True`
- `cation_hydrolysis_config` (sub-dict with `k_hyd`, `k_prot`,
  `k_des`, `delta_ohp_hat`, `cation`, `r_H_El_pm`,
  `pka_shift_form="singh_2016_eq_4"`, `gamma_max_nondim`)
- `stern_capacitance_f_m2` (Stern coefficient)
- `apply_h_source`, `apply_k_sink`, `override_sigma_singh_counts_pm2`
  (step 6 ablation flags — preserve byte-equivalence at defaults)

Runtime accessors (from `Forward/bv_solver/anchor_continuation.py`):

- `set_reaction_gamma_max_model(ctx, val)` — Γ_max runtime
- `set_reaction_k_des_model(ctx, val)` — k_des runtime
- `set_stern_capacitance_model(ctx, val)` — C_S runtime (needed
  for the two-stage anchor pattern)

---

## 5. Validation strategy

### 5.1. Regression cascade (run in order)

1. **Unit tests.** `pytest tests/ -k "phase6b or cation" -m "not slow"`
   should still pass after the constant changes.
2. **Phase A.2 regression** at v10b params. Re-run
   `scripts/studies/phase6b_v10a_phase_A2_v_kin.py --v-kin -0.10
   --k0-r4e-factor 1e-14 --out-subdir phase6b_v10b_phase_A2_v_kin`.
   Pass criterion: all 10 k_hyd rungs converge cleanly at λ=1.0;
   Picard converges everywhere; mass-balance residual < 5e-3.
3. **Step 6 plumbing-ablation regression** at v10b params. Re-run
   `scripts/studies/phase6b_step6_plumbing_ablation.py` with
   v10b defaults loaded. Pass criterion: all 5 ablations pass.
4. **C_S sensitivity sweep** across `{0.05, 0.10, 0.20, 0.30}`.
   For each value, run a 1-V cathodic point (V_RHE = −0.10 V) +
   λ=1 + the central k_hyd. Pass criterion: solver convergence
   + no spurious sign flips on observables.
5. **(optional) Γ_max sensitivity sweep** across the suggested
   bracket. Same pattern.
6. **(optional) k_des sensitivity sweep** across the suggested
   bracket. Same pattern.

### 5.2. What to compare against

- **A.2 baseline** (smoke params): the committed
  `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.json`.
  Numbers move with v10b — *that's expected* — but the qualitative
  shape (θ rises from 0.058 → 0.998 as k_hyd ramps; cd ≈ −3.12
  mA/cm²; x_2e ≈ 0.199 across the k_hyd grid) should be
  preserved within ~20%.
- **Deck observables**: Linsey 2025 ACS-CATL deck slide 15 gives
  the page-15 V-band IV curve. Should compare cd vs V curve
  qualitatively (not for v10b, but to flag if v10b makes things
  obviously worse).

### 5.3. Selectivity is **NOT** a v10b pass criterion

A.2's `single_v_selectivity_gap_pp = +5.09 pp` (H₂O₂% = 19.91%
vs deck band [25, 50]%) was within the 10pp routing cutoff, so
v10b k_des/Γ_max priority was flagged **LOW**. If v10b doesn't
move the gap into the band, that's fine. Phase D (K-only β fit)
is where selectivity gets data-fitted.

---

## 6. Hard invariants (do NOT touch in v10b)

These were locked by steps 4–6 and breaking them invalidates
A.2 and step 6:

| Constant | Value | Source |
|---|---|---|
| `V_kin` | `−0.10 V` | v10a' V-sweep diagnostic (step 4) |
| `K0_R4e_factor` | `1e-14` | v10a' branch-pass probe (step 4) |
| `k_hyd_baseline` | `1e-3 nondim` | Phase A.2 baseline (step 5) |
| `WARM_WALK_GRID` | `(+0.55, +0.40, +0.20, +0.10, −0.10)` | A.2 + step 6 driver |
| `LAMBDA_LADDER` | `(0.0, 0.25, 0.50, 0.75, 1.0)` | v10a |
| Parallel topology | `R2e (E°=0.695 V)` + `R4e (E°=1.23 V)` | Ruggiero 2022; CLAUDE.md hard rule #4 |
| `exponent_clip` | `100.0` | CLAUDE.md hard rule #2 |

If a v10b parameter change forces a re-derivation of any of
these (e.g. C_S = 0.20 changes the σ_S manifold enough that V_kin
shifts), that's a re-trigger of step 4 + 5 + 6, not a v10b
in-scope change.

---

## 7. Known risks / open questions

1. **`k_des` may not have a literature value.** If the planner's
   research turns up nothing, v10b should explicitly mark it as
   an engineering choice with documented prior + bracket sweep.
   Don't fabricate a citation.
2. **`Γ_max` smoke value (hard-sphere K⁺ monolayer) may already
   be a defensible approximation.** If the literature search
   confirms 1 monolayer is the right order, v10b's `Γ_max` change
   may be a tightening of the citation chain rather than a new
   numeric value. **That's still a valid v10b deliverable**;
   the writeup is the load-bearing part.
3. **A2 finding (step 6) — K⁺ Boltzmann pile-up at V_kin makes
   `c_K_boundary_avg ≈ 291·c_K_bulk`.** Sentinel-scale `R_inj`
   perturbations cannot dent boundary c_K by 5%. **v10b should
   NOT use boundary-c_K perturbation at V_kin as a `k_des`
   calibration diagnostic.** Use cap-saturated θ instead (e.g.
   `θ(k_hyd=1e-1)` is sensitive to `Γ_max` and `k_des` via the
   closed-form Γ_ss).
4. **Two-stage anchor pattern requirement.** Any C_S change
   requires the anchor build at `STERN_F_M2_ANCHOR = 0.10` then
   runtime-bump to target via `set_stern_capacitance_model` +
   Newton resolve. Combining `c_s_ladder + kw_eff_ladder` still
   raises `NotImplementedError` in `solve_anchor_with_continuation`
   (`anchor_continuation.py:1689` _PARAMETER_OVERRIDE_SETTERS).
5. **Cs⁺ vs K⁺.** Deck baseline is K⁺/SO₄²⁻ (Linsey deck slide
   9). Any cross-cation extrapolation needs explicit accounting
   per `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md`. v10b should
   stay on K⁺ unless the planner explicitly extends scope.
6. **Inverse status is paused** (CLAUDE.md). v10b is a forward-
   solver calibration step; do not introduce adjoint-tape work.
   If parameter inference is implicit anywhere (e.g. fitting
   `k_des` to data), wrap in `with adj.stop_annotating():` or
   defer to Phase D.

---

## 8. References (organized by topic)

### 8.1. Project canonical docs (read these first)

- `CLAUDE.md` — operational guide, hard rules, source-of-truth
  table, calling the production solver.
- `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`
  — acceptance bundle + § Status with the full Phase 6 chronology.
- `docs/phase6/CMK3_capacitance_literature.md` — step 7 output;
  the model for what v10b's calibration summary should look like.
- `docs/phase6/phase6b_next_steps_plan.md` — live Phase 6β plan
  (predecessor of the acceptance bundle).
- `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md` — Phase 6α
  water-ionization outcome + Phase 6β scoping handoff.
- `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` — Cs⁺ vs K⁺
  baseline audit.
- `docs/phase6/singh_2016_pka_formula.md` — Singh 2016 ΔpKa
  derivation + σ-mapping convention.
- `docs/solver/bv_solver_unified_api.md` — how to call the
  production solver.

### 8.2. Per-parameter literature (for v10b research)

**`C_S` — already done (step 7):**
- Bohra et al. 2024 *JPC C* PMC11215773
- Choi et al. 2024 *JPC C* `10.1021/acs.jpcc.4c03469`
- Pillai et al. 2024 *JPC C* `10.1021/acs.jpcc.3c05364`
- CatINT default (Stanford Bell group)
- Kilic, Bazant, Ajdari 2007 *Phys Rev E* 75:021503

**`Γ_max` — investigate:**
- Singh et al. 2016 *JACS* 138:13006 `10.1021/jacs.6b07612` —
  primary paper for the cation-hydrolysis formulation; check
  for partial-coverage estimates.
- Iamprasertkun 2019 *JPCL* `10.1021/acs.jpclett.8b03523` —
  HOPG basal + alkali cation series; coverage info.
- Bohra 2019 EES `10.1039/c9ee02485a` — open ask (not in
  `Articles/`); cation-hydration steric model.
- Linsey 2025 ACS-CATL deck slide 27 — per-cation pKa-near-Cu
  table (Li 13.16, Na 11.44, K 8.49, Cs 4.32) for cross-check.
- `data/EChem Reactor Modeling-Seitz-Mangan/` — group's
  documents may have coverage estimates (per CLAUDE.md hard
  rule #5, **check this folder first**).

**`k_des` — investigate (this is the hard one):**
- Singh 2016 *JACS* — does it report a `k_hyd` rate constant?
  If yes, detailed balance gives `k_des`.
- ORR/CO2R adsorbate-desorption literature for `OH*` at sp²
  carbon, with cation-stabilization correction. Look at:
  - Sabatier-volcano papers from the Nørskov group
  - Koper group adsorbate kinetics papers
- Eyring-style ΔG_des estimate from cation-OH bond energy.
- **Fallback:** treat as engineering choice with documented
  prior and bracket sweep.

### 8.3. Codebase entry points

- `Forward/bv_solver/cation_hydrolysis.py` — the bundle,
  `build_pka_shift`, `build_proton_boundary_source`,
  `gamma_ss_langmuir`, `update_gamma_from_solution`,
  `collect_v10a_rung_diagnostics`. Smoke baseline at line 242.
- `Forward/bv_solver/anchor_continuation.py` —
  `solve_anchor_with_continuation`,
  `solve_lambda_ramp_from_warm_start`,
  `_PARAMETER_OVERRIDE_SETTERS` at line 1689 (parameter dispatch).
- `Forward/bv_solver/config.py` —
  `_get_bv_convergence_cfg`, validates the step 6 ablation
  flags + the `cation_hydrolysis_config` block.
- `Forward/bv_solver/forms_logc.py` /
  `Forward/bv_solver/forms_logc_muh.py` — residual wiring +
  step 6 canonical artifacts (`_cation_hydrolysis_R_net_expr`,
  `_pka_shift_expr`, `_H_residual_term`, `_K_residual_term`,
  scalar flux forms).
- `scripts/_bv_common.py` — factory functions, deck-aligned
  defaults (`PARALLEL_2E_4E_REACTIONS_4SP`,
  `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`,
  `DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4`, etc.).
- `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` —
  SMOKE_KINETICS, `_build_sp`, `_make_mesh`,
  `_walk_lambda_zero_capture_snapshots`,
  `_compute_o2_flux_levich_ratio`, `_i_lim_4e_mA_cm2`.
- `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` — Phase A.2
  driver; convergence audit + v10b priorities block.
- `scripts/studies/phase6b_step6_plumbing_ablation.py` — step 6
  driver; reference for the override consumers.

### 8.4. Test entry points

- `tests/test_phase6b_v10a_langmuir_cap.py` — Langmuir cap +
  σ-unit conversion + Singh sign convention.
- `tests/test_phase6b_v10a_phase_A2_driver.py` — A.2 driver
  helpers + classification logic.
- `tests/test_phase6b_step6_plumbing_ablation.py` /
  `tests/test_phase6b_step6_plumbing_ablation_slow.py` — step 6
  driver helpers + slot wiring + override path assembly.
- `tests/test_phase6b_v9_gate3_gamma_machinery.py` — Γ
  bundle build + accessor round-trip tests.

### 8.5. Project memory (for cross-cutting findings)

`~/.claude/projects/-Users-jakeweinstein-Desktop-ResearchForwardSolverClone-FireDrakeEnvCG-PNPInverse/memory/MEMORY.md`
— index; relevant entries for v10b:
- `project_v10a_phase_A2_outcome.md` — A.2 numbers, k_hyd_route,
  selectivity gap.
- `project_v10a_prime_outcome.md` — v10a' V_kin selection.
- `project_v10a_prime_two_stage_anchor.md` — two-stage C_S
  anchor pattern.
- `project_step6_plumbing_ablation_outcome.md` — step 6 results,
  A2 K⁺ Boltzmann pile-up finding.
- `project_k0_r4e_ratio_regimes.md` — K0_R4e/K0_R2e regime
  characterization.

### 8.6. Critique provenance (for context)

Recent step plans + GPT critique rounds:
- `~/.claude/plans/phase6b-step6-plumbing-ablation.md` (step 6,
  5 rounds APPROVED)
- `~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md` (Phase A.2,
  4 rounds APPROVED)
- `~/.claude/plans/sparkly-gilded-pasteur.md` (v10a' V-sweep)
- `docs/handoffs/CHATGPT_HANDOFF_35_phase6b-step6-plumbing-ablation/`
  (step 6 critique trail)

---

## 9. Suggested phase breakdown for the planner

(The planner should refine; this is a starting skeleton.)

**Phase v10b.A — Literature pass (~3 days, parallelizable).**
Independent research on each parameter; deliverable is
`docs/phase6/v10b_calibration_summary.md` draft with citation
chains and bracket recommendations. May benefit from
`/sci-research` or `/gpt-critique-loop` for the harder `k_des`
question.

**Phase v10b.B — Code change + unit-test regression (~1 day).**
Update constants, update tests that reference smoke values, run
fast suite (`pytest -m "not slow"`).

**Phase v10b.C — A.2 + step 6 regression (~1 day wall).** Re-run
both drivers with v10b defaults; commit new JSON baselines;
verify all gates pass.

**Phase v10b.D — Sensitivity bracket sweep (~2 days wall).** C_S
sweep is the highest priority (already locked literature); Γ_max
and k_des optionally (if the literature didn't settle them).

**Phase v10b.E — Writeup + acceptance-bundle update (~1 day).**
Append v10b § Status to PHASE_0_ACCEPTANCE_BUNDLE_LOCK; update
CLAUDE.md (keep under 200 lines); add memory entry.

**Total estimate: 1–2 weeks** (matches the acceptance bundle's
estimate).

---

## 10. Out of scope for v10b

Explicitly **NOT** in v10b:
- Phase D (K-only Δ_β fit) — separate step 10.
- Cs⁺ / Li⁺ / Na⁺ / Rb⁺ extension — Phase E, step 11.
- Variable-`ε_S` / Booth-equation refinement — post-v10b option.
- Inverse / adjoint work — paused project-wide.
- V_kin re-selection / `K0_R4e_factor` retune — these are step 4
  invariants; only retrigger if v10b causes a convergence
  regression at V_kin.
- L_Stern parameterization vs `C_S` parameterization choice —
  carry as open ask from step 7.

---

## 11. Definition of done

v10b is done when **all** of these are true:

- [ ] `Γ_max`, `k_des`, `C_S` have literature-anchored values
      (or explicit engineering-choice flags with prior +
      sensitivity bracket).
- [ ] `docs/phase6/v10b_calibration_summary.md` is published
      and cross-referenced from CLAUDE.md + acceptance bundle.
- [ ] Phase A.2 regenerated at v10b params; new JSON committed;
      all 10 k_hyd rungs converge cleanly.
- [ ] Step 6 plumbing-ablation regenerated at v10b params;
      all 5 ablations pass.
- [ ] C_S sensitivity bracket sweep across {0.05, 0.10, 0.20,
      0.30} F/m² produces solver convergence + qualitative
      stability evidence.
- [ ] `tests/test_phase6b_*.py` pass at the new constants.
- [ ] Acceptance bundle § Status updated with v10b paragraph.
- [ ] CLAUDE.md updated and under 200 lines.

After v10b, the sequence proceeds to **step 9 (B.2)** —
densified k_hyd × λ ramp at V_kin with v10b params, ~2 days.

---

**End of handoff.** Hand this to a planner via `/sci-planner`,
`/gpt-critique-loop`, or a fresh Claude session. The planner
should produce a detailed multi-phase plan (likely with a
critique-loop pass on the `k_des` question specifically).
