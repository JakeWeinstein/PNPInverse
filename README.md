# PNPInverse

Research code for Poisson-Nernst-Planck / Butler-Volmer (PNP-BV)
forward simulation and (eventually) inverse kinetic-parameter
inference for the oxygen reduction reaction (ORR) on the
Seitz/Mangan deck (K₂SO₄, pH 4–6, parallel 2e⁻/4e⁻ topology per
Ruggiero 2022). The code is built around Firedrake finite
elements, Pyadjoint adjoints, direct PDE inverse studies, and a
separate surrogate inference stack. **The forward solver is the
active surface; all inverse and surrogate work is paused** until
the forward pipeline is mature enough for a clean re-entry.

The ORR is modeled as **parallel** 2e⁻ and 4e⁻ branches
(Ruggiero 2022 §1):

```text
R_2e: O2 + 2H+ + 2e-  ->  H2O2     E°_R2e = 0.695 V_RHE
R_4e: O2 + 4H+ + 4e-  ->  2 H2O    E°_R4e = 1.23  V_RHE
```

Target kinetic parameters (per-reaction): `[log_k0_2e, log_k0_4e,
alpha_2e, alpha_4e]`. The original sequential R1/R2 topology
(R1 = O₂→H₂O₂ at E°=0.68; R2 = H₂O₂→H₂O at E°=1.78) was wrong
per the Mangan deck and was retired in M3a.2 (2026-05-07).

Experimental context and physical constants tie to the
Seitz/Mangan 2019–2025 ORR datasets under
`data/EChem Reactor Modeling-Seitz-Mangan/` (gitignored; lives
outside git) and `docs/papers/Ruggiero2022_JCatal_source_paper.md`.

## Current State (2026-05-10)

The production forward model is the **3-dynamic-species + analytic
Bikerman counterion + proton electrochemical potential + log-rate
parallel 2e⁻/4e⁻ Butler–Volmer + finite Stern + Phase 6α water
self-ionization** stack. The deck baseline electrolyte is
**K⁺/SO₄²⁻** (Linsey 2025 ACS-CATL deck slide 9; Ruggiero 2022 §2).
Cs⁺/SO₄²⁻, Na⁺/SO₄²⁻, Li⁺/SO₄²⁻ are studied as part of the
cation-comparison study (slide 27). The legacy ClO₄⁻ single-
counterion stack is preserved as a backward-compat reference. The
legacy 4-species concentration backend was removed in the May 2026
cleanup.

**Phase 6α landed (2026-05-09):** water self-ionization residual
`E = c_H − c_OH` added via `Forward/bv_solver/water_ionization.py`.
New `solve_anchor_with_continuation` + `solve_grid_with_anchor` in
`Forward/bv_solver/anchor_continuation.py` with a `kw_eff_ladder`
outer-loop continuation. Default-off path is byte-equivalent to
pre-Phase-6α.

**Phase 6β v9 in progress (cation hydrolysis at OHP):** Singh 2016
SI Eq. (4) field-dependent pKa with finite-rate Γ_MOH outer Picard.
Gate 3 (Γ machinery) and Gate 4 (Singh formula + 9-combination
sweep) landed. **Phases A/B post-Gate-4 found cd plateau is at the
4e O₂ Levich limit (5.50 mA/cm² with code constants); v9 lacks a
Langmuir capacity on Γ so converged k_hyd ≥ 1e-3 is at ≥6
monolayers of MOH — physically invalid above ~1 monolayer.**
Phase 6β v10a (Langmuir cap + integrated diagnostics) is the next
executable step; the revised plan is locked in
`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` (post
two critique sessions: CHATGPT_HANDOFF_30 cd-invariance, capped at
ISSUES_REMAIN; CHATGPT_HANDOFF_31 strategic pivot, **APPROVED**).

**All inverse work is paused** until the forward solver is mature
enough for a clean re-entry — the v13–v24 study scripts, the FIM
work, and the FluxCurve adjoint pipeline are reference-only.

### Production forward stack

1. **3 dynamic species** (O₂, H₂O₂, H⁺) plus **analytic Bikerman
   counterion(s)** via `bv_bc.boltzmann_counterions` with
   `steric_mode="bikerman"`. Deck baseline uses **two** entries
   (K⁺ + SO₄²⁻) and requires `multi_ion_enabled=True`; the legacy
   ClO₄⁻ single-counterion stack is preserved as a reference. The
   K2SO4 stack additionally promotes K⁺ to a fully-dynamic NP
   species (`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`) so the Phase 6β
   cation-hydrolysis residual can read `c_K(0)` at the OHP, leaving
   SO₄²⁻ as the analytic Bikerman counterion. The closure is the
   steric-aware
   `c_b · exp(−z·φ) · (1 − A_dyn) / (θ_b + a_b · c_b · exp(−z·φ))`,
   matched between the IC seed and the residual side
   (`Forward/bv_solver/boltzmann.py:build_steric_boltzmann_expressions`).
2. **Proton electrochemical-potential primary variable**
   (`bv_convergence.formulation = "logc_muh"`):
   `mu_H = u_H + em·z_H·φ`. Keeps Newton smooth in deep-ψ regions
   where `u_H` and `φ` each vary by tens of log units. The other
   species use plain `u_i = ln c_i`.
3. **Log-rate Butler–Volmer with parallel 2e/4e** (`bv_log_rate =
   True`, `bv_reactions = PARALLEL_2E_4E_REACTIONS`). For each
   reaction `log_R = log(k0) + u_cat + Σ p·(u_sp − ln c_ref) −
   α·n_e·η`. The clip is on `η_scaled` *before* the `α·n_e`
   multiplication; `exponent_clip = 100` is the only PC-trustworthy
   default (clip=50 produces a fictitious peroxide current — see
   `docs/solver/clipping_conventions.md`). E°_R2e = 0.695 V vs RHE
   (Ruggiero 2022 §1); E°_R4e = 1.23 V vs RHE.
4. **Phase 6α water self-ionization**
   (`enable_water_ionization=True`, `kw_eff_hat=KW_HAT`): residual
   `E = c_H − c_OH = 0` with `c_OH = Kw_eff / c_H` closure. The
   orchestrator walks a `kw_eff_ladder` outside the k0 ladder
   (typical: `(0, KW_HAT·1e-6, KW_HAT·1e-3, KW_HAT·0.1, KW_HAT)`).
   Default-off path is byte-equivalent to pre-Phase-6α.
5. **Bikerman-consistent IC** (`initializer = "debye_boltzmann"` or
   `"linear_phi"`): composite-ψ (BKSA matched-asymptotic, saturated
   zone + outer exponential) plus multispecies-γ. The IC's surface
   activity and the residual's Bikerman closure agree on the
   saturated counterion concentration.
6. **Finite Stern compact layer**
   (`stern_capacitance_f_m2 ≈ 0.10` F/m²): absorbs ≈10–13 V_T of
   applied potential at high anodic V_RHE so the diffuse-layer drop
   ψ_D stays modest and the proton supply does not underflow the
   BV cathodic terms. **Note:** the 0.10 F/m² value is uncited
   (see `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md`); v10b's
   `docs/phase6/CMK3_capacitance_literature.md` is a prerequisite
   for any literature anchor.

This stack reaches **V_RHE = +1.0 V at 15/15** (cold ceiling
+0.60 V; warm-walk to +1.00 V) on a 15-voltage grid spanning
V_RHE ∈ [−0.5, +1.0] via the C+D orchestrator
(`solve_grid_per_voltage_cold_with_warm_fallback`). For the
multi-ion + Stern + Phase 6α stack, **prefer the newer
`solve_anchor_with_continuation` + `solve_grid_with_anchor`** in
`Forward/bv_solver/anchor_continuation.py` (C+D's Phase-1 cold-
start fails 13/13 around V ≈ +0.55 V on the multi-ion stack).
Cross-stack equivalence with the 4sp ClO₄⁻ dynamic reference
holds to ~10⁻⁹ in the cathodic regime and ~5·10⁻³ at the +0.5 V
edge. See `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md`
for the legacy sweep, `StudyResults/phase6b_v9_gate4_smoke/` for
the deck-aligned Phase 6β v9 Gate 4 baseline, and
`docs/PHASE_6B_V9_GATE_4B_SWEEP_RESULTS.md` for the 9-combination
sensitivity sweep.

### Recent timeline (2026-04-27 → 2026-05-10)

- **2026-05-10 — Phase 6β v9 Phases A/B/F + critique sessions
  (this commit).** Two GPT critique loops (CHATGPT_HANDOFF_30
  cd-invariance + CHATGPT_HANDOFF_31 strategic pivot) converged
  on a revised 11-step plan. Key findings: cd plateau is **O₂
  Levich-limited** (not H⁺ Levich); v9 Γ has no Langmuir capacity
  (~64-monolayer Γ at converged k_hyd=1e-2 — physically invalid);
  slide 27 IS Singh's Cu pKa table reproduced (not an independent
  experimental target); real validation target is per-cation
  experimental Cation Summary Table from `Summary Data-Error.xlsx`.
  Output: `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`
  + `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`. Also
  delivered: K⁺ Tafel slopes from Brianna 2019 LSV
  (`scripts/derive/extract_k_plus_tafel_slopes.py`; pH 6.39 only,
  3 cycles, 270–310 mV/decade R²>0.995; scope caveat in
  `docs/phase6/missing_data.md` M1).
- **2026-05-09 — Phase 5γ + Phase 6α (Gate 1/2).** Phase 5γ added
  `solve_anchor_with_continuation` + `solve_grid_with_anchor` in
  `Forward/bv_solver/anchor_continuation.py` to side-step C+D's
  Phase-1 cold-fail on the multi-ion + Stern stack. Phase 6α added
  water self-ionization residual `E = c_H − c_OH` via
  `Forward/bv_solver/water_ionization.py` with the kw_eff outer-loop
  ladder. Default-off; byte-equivalent to pre-Phase-6α. See
  `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md`.
- **2026-05-09 — Phase 6β Gate 3 + Gate 4 architecture.** Gate 3:
  R-space Γ_MOH coefficient + outer Picard
  (`Forward/bv_solver/cation_hydrolysis.py`); λ_hydrolysis activation
  knob; default-off contract. Gate 4A: Singh 2016 SI Eq. (4)
  field-dependent pKa extraction with per-cation Singh Table S1 +
  Cu r_H_El back-fit values (`SINGH_2016_CATION_PARAMS`). Gate 4B:
  9-combination sensitivity sweep at V=−0.40 V. See
  `docs/PHASE_6B_V9_GATES_3_4_SUMMARY.md` and
  `docs/PHASE_6B_V9_GATE_4B_SWEEP_RESULTS.md`.
- **2026-05-08 — Ruggiero deck audit + parallel 2e/4e topology.**
  `data/EChem Reactor Modeling-Seitz-Mangan/` audit confirmed deck
  baseline is K₂SO₄ (not ClO₄⁻ as the legacy code assumed) and
  parallel 2e (E°=0.695 V) + 4e (E°=1.23 V) ORR (not sequential
  R₀ + R₁). The legacy sequential R1/R2 (E°=0.68 V / 1.78 V) was
  retired in M3a.2. See
  `docs/papers/seitz_mangan_data_folder_audit_2026-05-08.md` and
  `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md`. New constants:
  `PARALLEL_2E_4E_REACTIONS`, `PARALLEL_2E_4E_REACTIONS_4SP`,
  `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`,
  `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC`,
  `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`.
- **2026-05-07 — IC/Picard bugfix + factory hard-rule defaults.**
  Stern-η inconsistency (`forms_logc.py`, `forms_logc_muh.py`,
  `Forward/bv_solver/picard_ic.py`) and Bikerman-γ inconsistency
  in Picard were fixed; shared scalar Picard outer loop extracted
  into `picard_ic.py`. `make_bv_solver_params` now defaults
  `E_eq_r1=0.68`, `E_eq_r2=1.78` (legacy sequential constants kept
  for backward compat; parallel-2e/4e overrides via `bv_reactions`).
  `bv_convergence.formulation` defaults to `"logc"`. C+D
  orchestrator gains a Phase-2 interior warm-walk for cold-failed
  interior gaps. Validator W1 (clip saturation) + W5
  (cation depletion at CG2+ orders) fixed. See
  `docs/handoffs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md` and
  `docs/handoffs/CHATGPT_HANDOFF_13_RESPONSE_TO_CODEX_REVIEW.md`.
- **2026-05-07 — Mangan 2025 alignment scaffolding.** Study runs
  emit an `experiment_metadata` block
  (`scripts/_bv_common.py:ExperimentMetadata`) with honest
  placeholders for deferred M0 fields. RRDE-style observables
  (surface-pH proxy, ring current, S_H₂O₂%, n_e) computed in
  `Forward/bv_solver/rrde_observables.py`; tests in
  `tests/test_rrde_observables.py`. See
  `docs/realignment/Mangan2025_experimental_alignment.md` and
  `docs/realignment/m0_target_extraction.md`.

### Next executable step

**Phase 6β v10a** — add Langmuir `(1 − Γ/Γ_max)` cap to cation-
hydrolysis residual + Picard formula; Γ clamp `[0, Γ_max]` with
`RuntimeWarning` on out-of-bounds; integrated rung diagnostics
(F₀, Γ, θ, R_forward_capped, denominator, R_2e/R_4e separately,
σ_S from solved fields); new helper
`Forward/bv_solver/units.py:sigma_C_m2_to_counts_pm2`; regression
tests for Γ → Γ_max and Γ_max → ∞ byte-equivalence to v9.
Estimated 5–7 days. See
`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` §
"Sequenced re-do plan" steps 2.

### Direct PDE Inverse Status — paused

All inverse scripts are **legacy / non-operational**. No inverse work
is currently running. The pipeline is held until the forward solver
is mature enough for a clean re-entry; treat the v13–v24 study
scripts (`scripts/studies/v*.py`), the FluxCurve adjoint pipeline,
and the FIM tooling as historical reference only.

When the inverse work resumes, start from
`docs/inverse/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`,
`docs/inverse/Next Steps After Basin Geometry.md`, and
`docs/inverse/noise_model_conventions.md`. The headline result going into
the pause was:

- Local Fisher information is good on the log-rate G0 grid
  (`cond(F) ≈ 1.8·10⁷`, `ridge_cos ≈ 0.03`); transfer coefficients
  α₁/α₂ recover to ~0.02–2% from clean starts.
- A single steady-state CD+PC experiment still has a multi-basin
  Tafel-ridge objective; **joint k0₁ / k0₂ recovery from one
  experiment is initialization-dependent**.
- The pre-existing multi-experiment Fisher screen plan (bulk O₂
  variation → H₂O₂-fed R2 isolation → L_ref / rotation variation)
  is the first thing to revisit when the pipeline restarts.

### Surrogate Pipeline Status — paused with the rest of inverse

`Surrogate/` (RBF, NN, NN ensemble, GP, PCE, POD-RBF, multistart, BCD,
cascade, ISMO) is a real, separately useful framework, but it is
gated on the inverse pipeline and is therefore also paused. The V&V
report in `writeups/vv_report/` documents passing surrogate-era
checks (MMS convergence, hold-out fidelity, 0–2% noise gates,
gradient consistency); read it as historical V&V on the surrogate
stack, not a current operational claim about the direct-PDE inverse.

## What To Read First

Source-of-truth docs, in roughly the order you'd hit them coming
back to the project cold:

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project-specific conventions, hard rules (E_eq, clip, C+D, IC/residual saturation match, parallel 2e/4e topology, K⁺ vs Cs⁺ deck baseline). |
| `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` | **Current revised plan** for Phase 6β v9 after critique sessions 30 + 31. Sequenced 11-step roadmap (Phase 0 → v10a → V-sweep → V_kin → A.2 → ablations → CMK-3 lit → v10b → B.2 → D → E). |
| `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` | Locked acceptance bundle (primary: H₂O₂ selectivity ±10 pp; secondaries; mechanism check; data reduction protocol). |
| `docs/papers/Ruggiero2022_JCatal_source_paper.md` | Peer-reviewed source paper for the Mangan deck physics: K₂SO₄ (not ClO₄⁻), parallel 2e (0.695 V) + 4e (1.23 V) ORR (not sequential), N=0.224, 1600 rpm, I=0.3 M. PDF at `docs/papers/Ruggiero2022_JCatal_manuscript.pdf`. |
| `docs/papers/seitz_mangan_data_folder_audit_2026-05-08.md` | Deep audit of the experimental data folder. |
| `docs/realignment/Mangan2025_experimental_alignment.md` | Gap audit between the model and the Mangan 2025 deck. |
| `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` | Audit of `fast-realignment` for Claude/GPT-conjecture vs. grounded changes. Flags HIGH-risk Cs⁺ vs deck-baseline K⁺ mismatch. |
| `docs/phase6/singh_2016_pka_formula.md` | Singh 2016 SI Section 1 extraction: Eq. (3) bulk pKa + Eq. (4) field-dependent ΔpKa + §5.2 σ-mapping convention. |
| `docs/phase6/missing_data.md` | Missing-data ledger (M1: Tafel slope xlsx; M2: C_S CMK-3 carbon; etc.). |
| `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md` | State of Phase 6α (water self-ionization landed; 8/8 sweep convergence; P3 surface-pH gate fails at 10.58). |
| `docs/PHASE_6B_V9_GATES_3_4_SUMMARY.md` | Phase 6β v9 Gate 3 (Γ machinery) + Gate 4A (Singh formula) status. |
| `docs/PHASE_6B_V9_GATE_4B_SWEEP_RESULTS.md` | 9-combination sensitivity sweep at V=−0.40 V (architecturally PASS; calibration OPEN). |
| `docs/phase6b_v9_post_gate4_plan.md` | Original post-Gate-4 plan (superseded by `PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`). |
| `docs/handoffs/CHATGPT_HANDOFF_30_phase6b-v9-cd-invariance/FINAL_REVISION.md` | Critique session 30 ledger (63 issues; cd-invariance finding capped at ISSUES_REMAIN). |
| `docs/handoffs/CHATGPT_HANDOFF_31_phase6b-v9-strategic-pivot/FINAL_REVISION.md` | Critique session 31 ledger (52 issues; APPROVED on R5). |
| `docs/solver/bv_solver_unified_api.md` | How to call the dispatcher and configure the production stack. |
| `docs/solver/clipping_conventions.md` | Three distinct BV-related clips and the operational rule that PC is fictitious at clip=50. |
| `docs/solver/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B for the logc+counterion stack; and why `solve_anchor_with_continuation` for the multi-ion + Stern + Phase 6α stack. |
| `docs/solver/steric_analytic_clo4_reduction_handoff.md` | Derivation of the Bikerman analytic-counterion residual closure. |
| `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md` | Legacy ClO₄⁻ reference sweep (3sp + Bikerman + Stern + muh + debye_boltzmann IC = 15/15 over V_RHE [−0.5, +1.0]). |
| `docs/handoffs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md` | Phase 6α outcome + Phase 6β scoping (the document upstream of the v9 architecture). |
| `writeups/ForwardSolverChangesMay26/forward_solver_changes_may2026.pdf` | May 2026 production-target writeup. |
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Forward-solver rebuild narrative. |
| `.verification/REPORT.md` | Multi-agent correctness verification of the production codepath. |

When the inverse pipeline resumes, also read:
`docs/inverse/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`,
`docs/inverse/Next Steps After Basin Geometry.md`,
`docs/inverse/noise_model_conventions.md`,
`docs/inverse/TODO_extend_inverse_v_range_negative.md`. Older
`CHATGPT_HANDOFF*` files (1–17) are useful chronology but reflect
the pre-Phase-6 state.

## Repository Layout

| Path | Role |
|---|---|
| `Forward/` | Forward solvers, parameters, noise, plotting, and steady-state utilities. |
| `Forward/bv_solver/` | Main PNP-BV package: log-c forms (`forms_logc.py`) and the muh variant (`forms_logc_muh.py`), log-rate BV with parallel 2e/4e reactions, ideal + Bikerman analytic counterions (`boltzmann.py`), shared scalar Picard for the `debye_boltzmann` IC (`picard_ic.py`), per-voltage diagnostics (`diagnostics.py`), observables, RRDE post-processing (`rrde_observables.py`), validation, the legacy C+D continuation orchestrator (`grid_per_voltage.py`), and the newer Phase 5γ + 6α anchor-and-grid orchestrator (`anchor_continuation.py`). Phase 6α water-ionization closure in `water_ionization.py`. Phase 6β cation-hydrolysis machinery in `cation_hydrolysis.py` (Γ_MOH outer Picard, Singh Eq. 4 pKa shift). The legacy concentration backend was removed in the May 2026 cleanup. |
| `Inverse/` | Generic Pyadjoint inverse framework and objective factories. **Inverse paused.** |
| `FluxCurve/` | Adjoint-gradient curve-fitting framework for Robin and BV flux/current curves. **Inverse paused — reference only.** |
| `Nondim/` | Physical constants, scaling transforms, and compatibility wrappers. |
| `Surrogate/` | Surrogate models (RBF, NN, GP, PCE, POD-RBF, multistart, BCD, cascade, ISMO). **Paused with the inverse pipeline.** |
| `scripts/studies/` | Forward-solver study scripts and diagnostics. **Current deck-aligned drivers:** `l_eff_transport_sweep_csplus_so4.py` (most recent Cs⁺/SO₄²⁻ + parallel 2e/4e + Phase 6α opt-in flag); `mangan_full_grid_csplus_so4.py` (deck-page-15 V_RHE band); `phase6b_v9_gate2_dynamic_k2so4_smoke.py` and `phase6b_v9_gate4_finite_hydrolysis_smoke.py` (Phase 6β v9 gates). **Legacy reference:** `peroxide_window_3sp_bikerman_muh.py` (ClO₄⁻ single-counterion + sequential R1/R2). `v*` are legacy inverse studies. |
| `scripts/derive/` | Data-derivation scripts. `extract_k_plus_tafel_slopes.py` extracts Tafel slopes from Brianna 2019 LSV (Phase F output goes to `data/derived/` which is gitignored). |
| `scripts/verification/` | MMS and BV forward strategy verification scripts. |
| `scripts/profile/` | Performance-profile runners for the production sweep. |
| `scripts/surrogate/` | Surrogate training, validation, GP/PCE/NN drivers, ISMO drivers (paused). |
| `scripts/Inference/` | Older master inverse entry points and wrappers. Kept for reproducibility (uppercase `Inference`). |
| `data/` | **Gitignored.** Experimental data drop from the Seitz/Mangan group (`EChem Reactor Modeling-Seitz-Mangan/` ~273 MB) + derived outputs (`derived/`). See `docs/papers/data_folder_code_inventory.md` for the per-file inventory. |
| `docs/` | Handoffs, plans, conventions, equations, literature inputs, and current status notes. Organized into `docs/phase6/` (Phase 6α/6β), `docs/handoffs/` (CHATGPT_HANDOFF_*), `docs/papers/` (Ruggiero, Singh, data-folder audits), `docs/realignment/` (Mangan deck alignment), `docs/solver/` (API, continuation, clipping conventions), `docs/inverse/` (paused). |
| `writeups/` | PDF/TeX reports (Apr 27 solver writeup, May 2026 forward-solver-changes writeup, May 4 IC walkthrough, V&V report). |
| `StudyResults/` | Generated results, summaries, plots, JSON, CSV, and run logs. Working research record, not a clean build-artifact directory. Historical diagnostic results nested under `StudyResults/diagnostics/`; fast-realignment outputs under `StudyResults/fast_realignment/`; Phase 6β v9 results in `StudyResults/phase6b_v9_*/`. |
| `tests/` | Pytest regression and verification tests. Firedrake tests are marked `slow`. Phase 6β v9 tests: `test_phase6b_v9_gate{1_roles,2_dynamic_k,3_gamma_machinery,4_finite_hydrolysis}.py`. Phase 6α tests: `test_water_ionization_phase_6a.py`. |
| `archive/` | Old results/code for reference, not the active implementation surface. |

There is no current `scripts/bv/` directory and no lowercase
`scripts/inference/` directory. Use `scripts/studies/`,
`scripts/Inference/`, `scripts/surrogate/`, and `scripts/derive/`.

## Core Forward-Solver Configuration

The active production stack is controlled through
`solver_params[10]`. The factory is
`scripts/_bv_common.py:make_bv_solver_params`. There are two
canonical call shapes today.

### Deck-aligned multi-ion stack (current production target)

The deck baseline is K⁺/SO₄²⁻; the script below uses Cs⁺/SO₄²⁻
(one of the four cations in the slide-27 comparison study). Swap
`DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` for a K⁺ entry for
apples-to-apples deck baselines (see CLAUDE.md gotchas).

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    PARALLEL_2E_4E_REACTIONS,                    # Ruggiero §1
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,  # ⚠ deck baseline is K⁺
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    K0_HAT_R2E, K0_HAT_R4E, V_T,
)
from Forward.bv_solver.anchor_continuation import (
    solve_anchor_with_continuation,
    extract_preconverged_anchor,
)
from Forward.bv_solver import (
    solve_grid_with_anchor, make_graded_rectangle_mesh,
)

sp = make_bv_solver_params(
    eta_hat=0.0, dt=0.25, t_end=80.0,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    bv_reactions=PARALLEL_2E_4E_REACTIONS,        # parallel 2e/4e
    boltzmann_counterions=[                       # multi-ion shared-θ closure
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    ],
    multi_ion_enabled=True,                       # required for ≥2 entries
    stern_capacitance_f_m2=0.10,
    initializer="debye_boltzmann",
    l_eff_m=100e-6,
    enable_water_ionization=False,                # Phase 6α opt-in
)

mesh = make_graded_rectangle_mesh(
    Nx=8, Ny=80, beta=3.0,
    domain_height_hat=sp.solver_options["bv_convergence"]["domain_height_hat"],
)

anchor_result = solve_anchor_with_continuation(
    sp.with_phi_applied(0.55 / V_T), mesh=mesh,
    k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
    initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
)
anchor = extract_preconverged_anchor(
    anchor_result, phi_applied_eta=0.55 / V_T,
    k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
    mesh_dof_count=anchor_result.ctx["U"].function_space().dim(),
)
grid = solve_grid_with_anchor(sp, mesh=mesh, anchor=anchor,
                              v_rhe_grid=V_RHE_GRID)
```

Reference drivers:

| Driver | Stack |
|---|---|
| `scripts/studies/l_eff_transport_sweep_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e + `--enable-water-ionization` flag (most recent; Phase 6α validation) |
| `scripts/studies/mangan_full_grid_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e at the deck V_RHE band |
| `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` | Phase 6β v9 Gate 4 + post-Gate-4 plan (Phases A/B at V=−0.20 V); supports `--voltage`, `--lambda-only`, `--k-hyd-ramp`, `--k-hyd`, `--k-prot`, `--out-subdir` |

### Legacy single-counterion stack (ClO₄⁻ reference)

For backward-compat / equivalence-checking against historical
results, use a single ClO₄⁻ counterion + the C+D dispatcher (no
`bv_reactions` keyword → defaults to legacy sequential R1/R2):

```python
from scripts._bv_common import DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC
from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback

sp = make_bv_solver_params(
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10, initializer="debye_boltzmann",
    # parallel-2e/4e omitted → defaults to legacy sequential R1/R2
)
```

Reference driver: `scripts/studies/peroxide_window_3sp_bikerman_muh.py`.

### Full API shape

The dispatcher in `Forward/bv_solver/dispatch.py` routes
`build_context()`, `build_forms()`, and `set_initial_conditions()`
to the right backend (`forms_logc.py` or `forms_logc_muh.py`) based
on `bv_convergence.formulation`, then to the right IC routine
(linear-φ or debye_boltzmann) based on `bv_convergence.initializer`.
The Bikerman residual-side closure is built by
`Forward/bv_solver/boltzmann.py:build_steric_boltzmann_expressions`
and enters both the Poisson source and the dynamic-species packing
fraction. The shared scalar Picard outer loop and Stern split for
the `debye_boltzmann` IC live in `Forward/bv_solver/picard_ic.py`.
The Phase 6α water-ionization closure lives in
`Forward/bv_solver/water_ionization.py`; the Phase 6β v9 cation-
hydrolysis machinery lives in `Forward/bv_solver/cation_hydrolysis.py`.
See `docs/solver/bv_solver_unified_api.md` for the full API
including all `solver_options` keys.

Gotchas (see CLAUDE.md for the full list):

- **K⁺ vs Cs⁺**: `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` is
  the Cs⁺ entry, but the deck *baseline* is K⁺/SO₄²⁻ (Linsey 2025
  deck slide 9). Apples-to-apples deck comparisons need a K⁺ entry.
- **Parallel 2e/4e**: must pass `bv_reactions=PARALLEL_2E_4E_REACTIONS`
  (or `PARALLEL_2E_4E_REACTIONS_4SP` for the K2SO4 stack). Omitting
  it falls back to the legacy sequential R1/R2 — don't use legacy
  for deck-aligned work.
- **`multi_ion_enabled=True` is required** when passing ≥2 bikerman
  counterions.
- **Phase 6α opt-in**: `enable_water_ionization=True` plus the
  `kw_eff_ladder` outer loop on `solve_anchor_with_continuation`.
- **Phase 6β v9 cation hydrolysis** is **not** yet production-trusted.
  `enable_cation_hydrolysis=True` activates the Γ machinery but
  there is no Langmuir capacity in v9, so converged k_hyd ≥ 1e-3
  is unphysical (>6 monolayers of MOH). v10a is the fix; track in
  `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`.
- `set_initial_conditions(ctx, sp, blob=True)` is silently ignored
  in log-c mode (no blob IC for `u_i = ln c_i`).
- `validate_solution_state` needs `is_logc=...` for log-c contexts;
  on the muh backend also pass `mu_species=ctx.get('mu_species')`,
  `em=ctx['nondim'].get('electromigration_prefactor', 1.0)`, and
  (for the W1 clip-saturation check) `reaction_e_eq` and
  `bv_exp_scale` from the live scaling dict.
- The `debye_boltzmann` IC requires either a `synthesised_4sp` ClO₄⁻
  counterion *or* a `steric_mode="bikerman"` entry; with
  `steric_mode="ideal"` it falls back to the tanh-Gouy-Chapman seed.
- `H2O2_SEED_NONDIM = 1e-4` is the finite seed for `ln c_H2O2` at
  the bulk Dirichlet BC, not a physics tweak.
- **C+D vs anchor-and-grid**: use C+D
  (`solve_grid_per_voltage_cold_with_warm_fallback`) for the
  legacy ClO₄⁻ single-counterion stack; use anchor-and-grid
  (`solve_anchor_with_continuation` + `solve_grid_with_anchor`)
  for the multi-ion + Stern + Phase 6α/6β stack (C+D's Phase-1
  cold-start fails 13/13 around V ≈ +0.55 V on the multi-ion stack).
- `l_eff_m` is read at form-build time via
  `bv_convergence['domain_height_hat']`; the mesh y-extent must match
  (`make_graded_rectangle_mesh(domain_height_hat=...)`) or the IC
  and residual disagree on the bulk anchor location.

## Setup

The forward solver depends on Firedrake, which is not pip-installable
from PyPI — it has its own installer that builds PETSc, MPI,
PyOP2/TSFC kernels, and friends. The supported workflow is:

1. **Install Firedrake into its own virtual environment.**
   Follow the official instructions at
   [firedrakeproject.org/install.html](https://www.firedrakeproject.org/install.html).
   The installer creates a venv (the path is up to you) with
   Firedrake, `firedrake.adjoint` (Pyadjoint), PETSc, MPI, NumPy,
   SciPy, and matplotlib already wired up. Conda environments are
   not supported by the Firedrake installer at the time of writing
   — use the venv path.

2. **Activate the Firedrake venv and install this package on top.**
   From the `PNPInverse/` directory:

   ```bash
   source /path/to/firedrake-venv/bin/activate
   python -m pip install -e ".[dev]"
   ```

   `pyproject.toml` declares NumPy, SciPy, matplotlib, and h5py as
   runtime deps and pytest/pytest-cov as `[dev]` extras; Firedrake
   itself is intentionally *not* in the dependency list because it
   must come from its own installer.

3. **(Optional) Install surrogate-stack dependencies.** The
   surrogate pipeline (`Surrogate/`) is paused, but if you want to
   run its scripts you'll additionally need PyTorch/GPyTorch (NN +
   GP work), ChaosPy (PCE), imageio, and Pillow. Install them into
   the same venv with `pip install`.

4. **Set Firedrake / PyOP2 cache paths and thread count.** PyOP2 and
   TSFC cache compiled kernels on disk; matplotlib also wants a
   writable config directory. The defaults can collide with each
   other on multi-user systems, so this project standardises on:

   ```bash
   export MPLCONFIGDIR=/tmp
   export XDG_CACHE_HOME=/tmp
   export PYOP2_CACHE_DIR=/tmp/pyop2
   export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc
   export OMP_NUM_THREADS=1
   ```

   Add these to your shell profile or set them per-session before
   running PDE work. `OMP_NUM_THREADS=1` matters: PETSc + MPI +
   OpenMP threading interact badly in this configuration and a
   single OMP thread per MPI rank is what the production sweep is
   tuned against.

5. **Verify the install.** Run the lightweight tests (no Firedrake
   needed):

   ```bash
   python -m pytest -m "not slow"
   ```

   then the slow tests (Firedrake required):

   ```bash
   python -m pytest -m slow
   ```

   and finally a smoke test of the production sweep on a few
   voltages:

   ```bash
   python scripts/studies/peroxide_window_3sp_bikerman_muh.py
   ```

   The full sweep takes minutes-to-hours; see
   `scripts/profile/profile_production_sweep.py` for a profile-only
   variant.

All commands in this README assume the Firedrake venv is active and
the working directory is `PNPInverse/`.

## Common Commands

Lightweight tests (no Firedrake):

```bash
python -m pytest -m "not slow"
```

Firedrake-dependent verification:

```bash
python -m pytest -m slow
python scripts/verification/mms_bv_3sp_logc_boltzmann.py
```

The deck-aligned multi-ion sweep (most recent; Cs⁺/SO₄²⁻ +
parallel 2e/4e, with optional Phase 6α water-ionization):

```bash
python -u scripts/studies/l_eff_transport_sweep_csplus_so4.py \
    [--enable-water-ionization]
```

The deck-page-15 V_RHE band sweep (Cs⁺/SO₄²⁻ + parallel 2e/4e,
single L_eff):

```bash
python -u scripts/studies/mangan_full_grid_csplus_so4.py
```

Phase 6β v9 Gate 4 cation-hydrolysis smoke study (the driver used
for Phases A/B in the post-Gate-4 plan):

```bash
# Phase A observability (with cached snapshot once it exists)
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py \
    --voltage -0.20 --lambda-only

# Phase B k_hyd ramp at the same voltage
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py \
    --voltage -0.20 --k-hyd-ramp
```

K⁺ Tafel slope extraction (Phase F, parallel-safe):

```bash
python -u scripts/derive/extract_k_plus_tafel_slopes.py
# outputs go to data/derived/ (gitignored)
```

Legacy ClO₄⁻ reference sweep (single-counterion + sequential R1/R2,
for backward-compat checking only — **not** a deck-aligned run):

```bash
python -u scripts/studies/peroxide_window_3sp_bikerman_muh.py
```

IC-distance diagnostic (the script that surfaced the May 2026-05-07
Stern-η + Bikerman-γ Picard bugs):

```bash
python scripts/diagnose_db_ic_distance.py
```

Profile the production sweep:

```bash
python scripts/profile/profile_production_sweep.py
```

Inverse and surrogate scripts are paused. Re-running them is not
part of the current workflow; treat their command lines in handoff
documents as historical reference only.

## Noise Models (paused inverse pipeline)

Recorded for when the inverse work resumes. Default to
`local_rel + abs_floor`; report `global_max`, `local_rel`, and
`local_rel + abs_floor` together when feasible. `global_max`
rotations are suspect when CD/PC spans many decades across the
voltage grid (the negative-V FIM study is the cautionary example).
See `docs/inverse/noise_model_conventions.md`.

## Known Gotchas

- Run scripts from `PNPInverse/` (this directory), not from
  `Forward/` or `scripts/`. The Firedrake venv must be active —
  see the Setup section above.
- Forward studies are expensive — minutes to hours depending on
  mesh, voltage grid, and Phase-2 fill behaviour.
  `StudyResults/` is part of the working research record, not a
  clean build-artifact directory; check existing `summary.md`
  files before regenerating.
- `scripts/Inference/` is uppercase. Older README text and notes
  may refer to paths that no longer exist.
- **Use the parallel 2e/4e ORR topology** (Ruggiero 2022 §1)
  via `bv_reactions=PARALLEL_2E_4E_REACTIONS` for any deck-aligned
  work. The legacy sequential R1/R2 (E°=0.68/1.78 V) is preserved
  only for backward-compat against pre-M3a.2 results.
- **Deck baseline electrolyte is K⁺/SO₄²⁻**, not ClO₄⁻ or Cs⁺
  (Linsey 2025 deck slide 9). Cs⁺ is one of four cations in the
  comparison study (slide 27). Use a K⁺ entry for deck baselines;
  use Cs⁺/Na⁺/Li⁺ entries for cation-comparison runs.
- **`multi_ion_enabled=True`** is required when passing ≥2 bikerman
  counterions; the factory raises if it's not set.
- **C+D vs anchor-and-grid**: use C+D
  (`solve_grid_per_voltage_cold_with_warm_fallback`) for legacy
  ClO₄⁻ single-counterion stack; use **anchor-and-grid**
  (`solve_anchor_with_continuation` + `solve_grid_with_anchor`)
  for the multi-ion + Stern + Phase 6α/6β stack. C+D's Phase-1
  cold-start fails 13/13 around V ≈ +0.55 V on the multi-ion stack;
  Strategy B (`solve_grid_with_charge_continuation`) also fails
  on the logc + counterion stack (3/13 at production resolution).
- The Bikerman residual closure and the IC's matched-asymptotic
  seed must agree about steric saturation; mixing a bikerman IC
  with an ideal-counterion residual (or vice-versa) cold-fails
  on the saturated manifold.
- **`exponent_clip = 100`** is the only PC-trustworthy default. The
  clip is on `eta_scaled = (V_RHE − E_eq)/V_T` *before* the
  `α·n_e` multiplication. Older results at `clip = 50` produce a
  fictitious peroxide current; do not compare them against
  experiment. Some configs cold-fail more often at clip=100;
  recover with anchor-and-grid warm-walk or Stern, not by lowering
  the clip. `u_clamp = 100` for the same reason.
- **Phase 6α water-ionization is opt-in.** When using
  `enable_water_ionization=True`, also pass the `kw_eff_ladder`
  outer-loop to `solve_anchor_with_continuation` (typical:
  `(0.0, KW_HAT·1e-6, KW_HAT·1e-3, KW_HAT·0.1, KW_HAT)`).
- **Phase 6β v9 cation hydrolysis is a numerical/architectural
  diagnostic only, not a physics-trustworthy production path.**
  v9 lacks a Langmuir capacity on Γ_MOH, so converged k_hyd ≥ 1e-3
  produces ≥6 monolayers of MOH at the OHP — physically invalid.
  v10a (Langmuir cap) is the production fix; until v10a lands,
  no physics conclusion from Γ-dependent observables is defensible.
  See `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`.
- **`l_eff_m` is read at form-build time** via
  `bv_convergence['domain_height_hat']`; the mesh y-extent must
  match. The current gate4 driver uses 16 µm (Ruggiero 2022 1600 rpm
  Levich layer); CLAUDE.md still mentions 100 µm as a broader
  setting — match to the deck's documented rotation rate.
- **cd at V ≤ +0.10 V is the 4e O₂ Levich limit** (5.50 mA/cm²
  with code constants D_O₂=1.9e-9, C_O₂=1.2 mol/m³, L_eff=16 µm).
  Adding H⁺ at the OHP via cation hydrolysis cannot move what's
  already O₂-bottlenecked; cation-effect validation should target
  per-cation H₂O₂ selectivity / ring current / surface pH, not cd
  per se. See `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`.
- The 4sp dynamic stack is a validation reference (cathodic
  agreement to ~10⁻⁹; +0.5 V edge ~5·10⁻³). Its anodic ceiling is
  bound by the dynamic c_ClO₄ NP equation, not the IC; "go fully
  dynamic" is *not* a fix for the anodic ceiling.
- **Singh-σ-to-model-σ_S mapping is an assumption, not a fact.**
  `docs/phase6/singh_2016_pka_formula.md` §5.2 chose the local
  Stern σ_S convention; the alternative (imposed-Singh cell-level
  σ) is available as the `pka_override_ablation` flag. Phase D
  must bracket calibration under both conventions if their results
  diverge by > 30%.
- Inverse status is paused; do not claim single-experiment
  four-parameter recovery is solved.
