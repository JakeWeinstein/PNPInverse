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

## Current State (2026-06-15)

> **New here? Read [`docs/INTRO_TO_THIS_REPO.md`](docs/INTRO_TO_THIS_REPO.md)
> first** for the broad-strokes tour (stages, where to start reading code, how
> to call the production solver), and [`REPO_LAYOUT.md`](REPO_LAYOUT.md) for the
> file map. This README is the detailed research narrative.

> **Active frontier — Phase 7 (dual-pathway).** Phase 6β's Phase D was
> root-caused to a transport artifact: total current was pinned at the H⁺ Levich
> cap, so every fit knob was tuning a transport-capped current. Phase 7 pivoted to
> **dual-pathway water-as-proton-donor kinetics + RRDE-correct `L_eff` + slide-15
> volcano fit**. **Phase 7.2** locked a K₂SO₄ pH-6.39 disk+ring dual-series fit
> against real LSV data (water-route model fits; kinetics transfer; ring-determined
> partition). **Phase 7.3** (in flight) ranks the missing mechanism behind
> pH-flatness — onset shifts +41 mV/pH on the RHE scale ⇒ a proton-uncoupled first
> electron transfer; peroxide-consumption owns selectivity. See
> `docs/handoffs/CHATGPT_HANDOFF_4{1..5}_*` and `tasks/todo.md`.
>
> The repo was reorganized on **2026-06-15** (`scripts/studies/` split into
> `drivers/ plot/ extract/`; closed-phase results archived; 225 MB `archive/`
> moved off-git to `~/PNPInverse-archive`). Paths below reflect the new layout.

The production forward model is the **3-dynamic-species + analytic
Bikerman counterion(s) + proton electrochemical potential + log-rate
parallel 2e⁻/4e⁻ Butler–Volmer + finite Stern** stack, with optional
**Phase 6α water self-ionization** and **Phase 6β v10b cation
hydrolysis** layered on as opt-in physics. The deck baseline
electrolyte is **K⁺/SO₄²⁻** (Linsey 2025 ACS-CATL deck slide 9;
Ruggiero 2022 §2); Cs⁺/SO₄²⁻, Na⁺/SO₄²⁻, Li⁺/SO₄²⁻ are studied as
part of the cation-comparison study (slide 27). The legacy ClO₄⁻
single-counterion stack and the legacy 4-species concentration backend
are gone except as backward-compat references in tests.

**Stern compact-layer Robin BC (production: `C_S = 0.20 F/m²`).**
The Bonnefont–Argoul–Bazant (2001) Stern model is implemented as a
Robin BC on the Poisson equation at the electrode; derivation +
exact code mapping in `writeups/May13th/stern_robin_bc_derivation.tex`.
The value is literature-locked at the Bohra–Koper–Choi consensus
(`docs/phase6/CMK3_capacitance_literature.md`, step 7, 2026-05-10);
it replaces the uncited `C_S = 0.10` used in earlier v9 work.
Convergence at `C_S = 0.20` requires the **Stern bump ladder** —
build the anchor cold at `C_S = 0.10` then ramp through verified
rungs `(0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0)` via
`set_stern_capacitance_model` without rebuilding forms.

**Phase 6α (water self-ionization)** — *retired as primary mechanism
(2026-05-09 → 2026-05-13 writeup).* Residual `E = c_H − c_OH` is
plumbed via `Forward/bv_solver/water_ionization.py` with a
`kw_eff_ladder` outer loop, but the gate P3 failed (max surface pH
10.58 at L_eff=16 µm, L-invariant) and a bulk-rate estimate puts
water dissociation ~20× too slow to backfill the cathodic H⁺ demand.
Plumbing preserved as opt-in (`enable_water_ionization=True`); default
path is byte-equivalent to pre-6α.

**Phase 6β v10b (cation hydrolysis at the OHP) — calibrated and
shipped (2026-05-10), Phase D fit blocked (2026-05-12).** Singh 2016
SI Eq. (4) field-dependent pKa with finite-rate Γ_MOH outer Picard
and Langmuir `(1 − Γ/Γ_max)` cap. K⁺ is promoted to a fourth dynamic
NP species (`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`) so the hydrolysis
residual can read `c_K(0)` at the OHP; SO₄²⁻ stays analytic. Three
parameters locked from literature in
`docs/phase6/v10b_calibration_summary.md` (live in `calibration/v10b.py`,
Firedrake-free):

| Symbol | Value | Origin |
|---|---|---|
| `GAMMA_MAX_HAT_V10B` | 0.047 (nondim) | One-monolayer hard-sphere areal coverage; V10A chain tightened |
| `K_DES_NONDIM_V10B` | 1.0 | Eyring prior, ΔG_des ∈ [0.69, 0.94] eV |
| `C_S_F_M2_V10B` | 0.20 F/m² | Bohra/Koper/Choi consensus (step 7) |

**Step 10 Phase D K-only Δ_β fit (2026-05-12) returned
`OUTCOME_C_NON_IDENTIFIABLE_flagged`:** scanning the carbon-vs-Cu
β-offset Δ_β through 11 orders of magnitude leaves the loss flat at
15.629 pp² under Stern σ-mapping; Ablation σ-override also flat.
Model max H₂O₂% = 66.58 pp vs deck K@pH 4 mean 50.95 pp → uniform
**+15.6 pp overshoot** that Δ_β alone cannot close. **Phase E
predictive screen MUST NOT launch on this Δ_β.** Open scope:
re-fit `k_des`/`Γ_max`, sweep `r_H_El`, examine local-pH /
mass-transport coupling.

**Known discrepancy — Bikerman `a_nondim` for dynamic species.**
Both `THREE_SPECIES_LOGC_BOLTZMANN` and
`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4` seed O₂/H₂O₂/H⁺ with
`A_DEFAULT = 0.01` (≈ hard-sphere radius 14.9 Å, ~150× larger than
the physical values: O₂ 1.7 Å, H₂O₂ 2.0 Å, H₃O⁺ 2.8 Å Stokes).
Counterion entries (`A_KPLUS_HAT`, `A_CSPLUS_HAT`, `A_SO4_HAT`,
`A_OH_HAT`) use real radii. H⁺ in particular is Bikerman-clamped
~150× tighter than its physical cap, so anything that depends on
the H⁺ Levich plateau is suspect until the physical-`a` bridge runs
in `scripts/studies/_phase_D_bridge_corrected_a*.py` disambiguate.
The Phase D non-identifiability verdict is independent of this, but
the +15.6 pp overshoot might not be.

**MMS verification of the muh + multi-ion + Stern stack landed
2026-05-14.** Four manufactured-solution source terms (NP interior +
NP electrode + Poisson interior + Stern Robin) recover textbook CG1
rates ($L^2$≈2.0, $H^1$≈1.0; $R^2$>0.999) for $(u_{O_2}, u_{H_2O_2},
\mu_H, \varphi)$ on the production stack used by the
`solver_demo_slide15_no_speculative_cs.py` Cs⁺/SO₄²⁻ baseline.
Driver: `scripts/verification/mms_pnpbv_muh_multi_ion_stern.py`
(~1100 LOC). Tests: `tests/test_mms_logc_muh_multi_ion_stern.py`
(18 cases incl. 12 broken-config invariants). Algebraic derivation:
`writeups/May13th/mms_source_terms_derivation.tex` +
`docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md`.

**All inverse work is paused** until the forward solver is mature
enough for a clean re-entry — the v13–v24 study scripts, the FIM
work, and the FluxCurve adjoint pipeline are reference-only.

### Production forward stack

1. **3 dynamic species** (O₂, H₂O₂, H⁺) plus **analytic Bikerman
   counterion(s)** via `bv_bc.boltzmann_counterions` with
   `steric_mode="bikerman"`. Deck baseline uses **two** entries
   (K⁺ + SO₄²⁻) and requires `multi_ion_enabled=True`. For the
   Phase 6β hydrolysis path, K⁺ is additionally promoted to a
   fully-dynamic NP species (`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`) so
   the cation-hydrolysis residual can read `c_K(0)` at the OHP,
   leaving SO₄²⁻ as the analytic Bikerman counterion. The closure
   is the shared-θ multi-ion Tresset extension
   `c_k = c_{b,k} · q_k · (1 − A_dyn) / (θ_b + Σ a_{k'} c_{b,k'} q_{k'})`
   with `q_k = exp(−z_k φ)`, matched between the IC seed and the
   residual side
   (`Forward/bv_solver/boltzmann.py:build_steric_boltzmann_expressions`).
   Full derivation including the hybrid `(1−A_dyn)` pre-factor
   (original to this work): `writeups/May13th/analytic_counterion_derivation.tex`.
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
4. **Bikerman-consistent IC** (`initializer = "debye_boltzmann"` or
   `"linear_phi"`): composite-ψ (BKSA matched-asymptotic, saturated
   zone + outer exponential) plus multispecies-γ. The IC's surface
   activity and the residual's Bikerman closure agree on the
   saturated counterion concentration.
5. **Stern compact-layer Robin BC**
   (`stern_capacitance_f_m2 = 0.20` F/m², literature-locked at
   Bohra/Koper/Choi consensus). Imposes
   `ε_b · ∂_n φ = C_S · (V_app − φ_OHP)` at the electrode marker via
   `F_res -= stern_coeff · (φ_app − φ) · w · ds`; partitions the
   applied voltage between compact and diffuse layers, keeping the
   diffuse drop ψ_D modest at high V_RHE. Cold-start at `C_S = 0.20`
   is unreliable on the multi-ion stack — build the anchor at
   `C_S = 0.10` and use the Stern bump ladder
   `(0.10 → 0.20 → 0.50 → 1.0 → 2.0 → 5.0 → 10.0 → 100.0)` via
   `set_stern_capacitance_model(ctx, c_s)` (no form rebuild).
   Derivation + BAB(2001) code mapping:
   `writeups/May13th/stern_robin_bc_derivation.tex`.
6. **Phase 6α water self-ionization (opt-in)**
   `enable_water_ionization=True`, `kw_eff_hat=KW_HAT`: residual
   `E = c_H − c_OH = 0` with `c_OH = Kw_eff / c_H` closure. The
   orchestrator walks a `kw_eff_ladder` outside the k0 ladder
   (typical: `(0, KW_HAT·1e-6, KW_HAT·1e-3, KW_HAT·0.1, KW_HAT)`).
   Default-off path is byte-equivalent to pre-Phase-6α. Retired as
   primary peak mechanism (Phase 6α P3 gate fail; see Phase 6α
   summary).
7. **Phase 6β v10b cation hydrolysis (opt-in)**
   `enable_cation_hydrolysis=True` with v10b literature parameters
   (`GAMMA_MAX_HAT_V10B = 0.047`, `K_DES_NONDIM_V10B = 1.0`,
   `C_S_F_M2_V10B = 0.20`; `calibration/v10b.py`). Singh Eq. (4)
   field-dependent pKa shift drives a global Γ_MOH Real-element
   scalar on the electrode boundary via an outer Picard; Langmuir
   `(1 − Γ/Γ_max)` cap prevents super-monolayer Γ. Step 10 Phase D
   fit returned `OUTCOME_C_NON_IDENTIFIABLE_flagged`; the
   architecture and machinery pass all hard gates, but the fit
   step is blocked. Don't claim physics conclusions from
   Γ-dependent observables yet. See Phase D summary.

This stack reaches **V_RHE = +1.0 V at 15/15** (cold ceiling
+0.60 V; warm-walk to +1.00 V) on a 15-voltage grid spanning
V_RHE ∈ [−0.5, +1.0] via the C+D orchestrator
(`solve_grid_per_voltage_cold_with_warm_fallback`). For the
multi-ion + Stern + Phase 6α/6β stack, **prefer the newer
`solve_anchor_with_continuation` + `solve_grid_with_anchor`** in
`Forward/bv_solver/anchor_continuation.py` (C+D's Phase-1 cold-
start fails 13/13 around V ≈ +0.55 V on the multi-ion stack). The
anchor+grid orchestrator wraps an `AdaptiveLadder` (k₀ continuation
with `sqrt`-midpoint failure recovery, optional `warm_start_floor`
arithmetic bisection for λ-from-warm-start added in step 9.5) +
recursive `warm_walk_phi` (8 substeps × depth-5 bisection,
32× refinement max) to absorb the Frumkin cliff near V ≈ 0 V
caused by SO₄²⁻ Bikerman saturation. Visual call graph of this
codepath: `writeups/May13th/forward_codepath_demo_slide15.tex`.

### Recent timeline (2026-04-27 → 2026-05-14)

- **2026-05-14 — MMS verification for muh + multi-ion + Stern stack
  (this commit).** Four manufactured-solution source terms (NP
  interior + NP electrode + Poisson interior + Stern Robin) recover
  textbook CG1 rates on the production stack used by
  `solver_demo_slide15_no_speculative_cs.py`. Driver:
  `scripts/verification/mms_pnpbv_muh_multi_ion_stern.py` (~1100
  LOC); 18-test suite at `tests/test_mms_logc_muh_multi_ion_stern.py`
  including 12 broken-config invariants. Algebra:
  `writeups/May13th/mms_source_terms_derivation.tex`. The MMS
  baseline rescales `c_{0,H₂O₂}^bulk = 1` (instead of the production
  seed `1e-4`) to keep Newton in basin without continuation; an
  identity check confirms this leaves the operator unchanged.
- **2026-05-12 — Phase 6β step 10 Phase D K-only Δ_β fit.**
  Verdict `OUTCOME_C_NON_IDENTIFIABLE_flagged`: scanning Δ_β across
  11 orders of magnitude leaves loss flat at 15.629 pp² under
  Stern σ-mapping (local σ ~10⁻⁷ counts/pm²); Ablation σ-override
  also flat. Model max H₂O₂% = 66.58 pp vs deck K@pH4 mean
  50.95 pp = uniform +15.6 pp overshoot. Phase E predictive screen
  blocked. Also surfaced during bridge diagnostics: dynamic species
  Bikerman `a_nondim` discrepancy (O₂/H₂O₂/H⁺ use `A_DEFAULT = 0.01`
  ≈ r 14.9 Å, ~150× tighter than physical for H₃O⁺). Bridge runs
  in `scripts/studies/_phase_D_bridge_corrected_a*.py`. Writeup:
  `docs/phase6/phase6b_step10_phase_D_summary.md`.
- **2026-05-11 — Phase 6β step 9 B.2 (densified k_hyd × λ ramp).**
  14×10 = 140-rung grid at V_kin = −0.10 V on v10b parameters;
  14/14 k_hyd × 10/10 λ converge after step 9.5 added
  `warm_start_floor` arithmetic bisection to `AdaptiveLadder` for
  λ-from-warm-start rescue. Mass balance ≤ 5×10⁻¹³. Summary:
  `docs/phase6/phase6b_step9_B2_summary.md`.
- **2026-05-10 — Phase 6β v10b literature calibration shipped.**
  Three parameters locked from literature: `GAMMA_MAX_HAT_V10B =
  0.047` (one-monolayer hard-sphere areal coverage; V10A chain
  tightened), `K_DES_NONDIM_V10B = 1.0` (Eyring prior ΔG_des ∈
  [0.69, 0.94] eV), `C_S_F_M2_V10B = 0.20` F/m² (Bohra/Koper/Choi
  consensus, step 7). Source of truth: `calibration/v10b.py`
  (Firedrake-free). D7-D1 (C_S bracket) 4/4 + D7-D4 (Γ_max ×
  k_des matrix) 30/30 hard gates pass. Summary:
  `docs/phase6/v10b_calibration_summary.md`,
  `docs/phase6/CMK3_capacitance_literature.md`.
- **2026-05-10 — Phase 6β v10a (Langmuir cap) + v10a' V-sweep
  + Phase A.2 k_hyd ramp.** Langmuir `(1 − Γ/Γ_max)` cap added to
  hydrolysis residual + Picard formula; Γ clamp `[0, Γ_max]` with
  RuntimeWarning. V10a' V-sweep at `C_S = 0.20` +
  `K0_R4e_factor = 1e-14` returned V_kin = −0.10 V (primary path:
  σ_S < 0, branch active, not transport-artifact, not
  cap-dominated). 10-rung k_hyd ladder at V_kin × λ = 1.0:
  10/10 converge; selectivity gap at V_kin = +5 pp so
  Γ_max/k_des retune flagged LOW priority going into v10b.
- **2026-05-10 — Phase 6β v9 Phases A/B/F + critique sessions.**
  Two GPT critique loops (CHATGPT_HANDOFF_30 cd-invariance +
  CHATGPT_HANDOFF_31 strategic pivot) converged on a revised
  11-step plan. Key findings: cd plateau is **O₂ Levich-limited**
  (not H⁺ Levich); v9 Γ has no Langmuir capacity (~64-monolayer Γ
  at converged k_hyd=1e-2 — physically invalid); slide 27 IS
  Singh's Cu pKa table reproduced (not an independent experimental
  target); real validation target is per-cation experimental Cation
  Summary Table from `Summary Data-Error.xlsx`. Output:
  `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`
  + `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`.
  Also delivered: K⁺ Tafel slopes from Brianna 2019 LSV
  (`scripts/studies/extract/extract_k_plus_tafel_slopes.py`; pH 6.39 only,
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

**Phase D′ / Phase 6γ scoping — the selectivity gap is open.** With
Δ_β alone ruled out, candidate next moves in rough order of
likelihood (`docs/phase6/phase6b_step10_phase_D_summary.md` §7,
`writeups/May13th/phase_6_overview.tex` §6):

1. **Physical-`a` Bikerman bridge runs disambiguate the
   +15.6 pp overshoot.** Four runs in flight at
   `scripts/studies/_phase_D_bridge_corrected_a*.py`; resolves
   whether the H⁺ Levich plateau is silently set by the dynamic
   `A_DEFAULT = 0.01` clamp.
2. **Re-fit `k_des` and `Γ_max`.** Phase D locked these as v10b
   literature priors; Phase D non-identifiability re-opens them as
   fit parameters.
3. **`r_H_El` sensitivity sweep.** Cu prior (200.98 pm) may not
   transfer cleanly to CMK-3 carbon.
4. **Local-pH / mass-transport coupling.** Model selectivity is
   V-flat → transport-limited; H⁺ Levich convention may need
   re-examination.
5. **Phase 6δ (parallel alkaline ORR kinetics)** likely needed
   downstream — 6β fixes the local-pH *source*, not the alkaline
   decay mechanism.

**Phase E predictive screen (Cs⁺/Na⁺/Li⁺ holdout) MUST NOT launch
on the current Δ_β** until items (1)–(4) resolve.

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
| `docs/INTRO_TO_THIS_REPO.md` | **Start here cold.** Broad-strokes tour: research arc (phases 5→7), where to start reading code, where the production solver lives + how to call it, the mental model. |
| `REPO_LAYOUT.md` | One-page file map (source packages, `scripts/` taxonomy, `StudyResults/`, `docs/`, `tests/`) after the 2026-06-15 reorg. |
| `CLAUDE.md` | Project-specific conventions, hard rules (E_eq, clip, C+D vs anchor-and-grid, IC/residual saturation match, parallel 2e/4e topology, K⁺ vs Cs⁺ deck baseline, `a_nondim` discrepancy for dynamic species). |
| `writeups/May13th/phase_6_overview.pdf` | **Short story of Phase 6α/6β:** the selectivity gap, the two hypotheses, sub-step chronology (v10a → v10a' → A.2 → step 6 → v10b → step 9/9.5 → step 10 Phase D), and the current verdict + open issues. |
| `writeups/May13th/forward_codepath_demo_slide15.pdf` | Visual call graph of `solver_demo_slide15_no_speculative_cs.py`: anchor build → Stern bump ladder → grid walk + 9-layer defense-in-depth around the Newton/SNES core. |
| `writeups/May13th/analytic_counterion_derivation.pdf` | Derivation of the multi-ion shared-θ Bikerman counterion closure from electrochemical-potential first principles, incl. hybrid `(1−A_dyn)` extension to Tresset 2008. |
| `writeups/May13th/stern_robin_bc_derivation.pdf` | Bonnefont–Argoul–Bazant (2001) Stern Robin BC derivation with exact code mapping (`forms_logc_muh.py:668`). |
| `writeups/May13th/mms_source_terms_derivation.pdf` | Algebra of the four MMS source terms (NP interior + NP electrode + Poisson interior + Stern Robin) for the production stack. |
| `docs/phase6/v10b_calibration_summary.md` | v10b literature lock of `Γ_max = 0.047`, `k_des = 1.0`, `C_S = 0.20 F/m²`; 4-test compatibility audit. |
| `docs/phase6/CMK3_capacitance_literature.md` | C_S = 0.20 F/m² citation chain (Bohra/Choi/Pillai/CatINT/Kilic). |
| `docs/phase6/phase6b_step9_B2_summary.md` | Step 9 B.2 (140-rung k_hyd × λ ramp at V_kin on v10b parameters); step 9.5 `warm_start_floor` extension. |
| `docs/phase6/phase6b_step10_phase_D_summary.md` | Phase D K-only Δ_β fit verdict `OUTCOME_C_NON_IDENTIFIABLE_flagged`; bridge diagnostics; open scope. |
| `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` | Original 11-step roadmap for Phase 6β (Phase 0 → v10a → V-sweep → V_kin → A.2 → step 6 → CMK-3 lit → v10b → B.2 → D → E). |
| `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` | Locked acceptance bundle (primary: H₂O₂ selectivity ±10 pp; secondaries; mechanism check; § Status with full chronology). |
| `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md` | State of Phase 6α (water self-ionization plumbed; P3 surface-pH gate fails at 10.58; retired as primary). |
| `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` | Audit of `fast-realignment` for Claude/GPT-conjecture vs. grounded changes. Flags HIGH-risk Cs⁺ vs deck-baseline K⁺ mismatch. |
| `docs/phase6/singh_2016_pka_formula.md` | Singh 2016 SI Section 1: Eq. (3) bulk pKa + Eq. (4) field-dependent ΔpKa + §5.2 σ-mapping convention. |
| `docs/phase6/missing_data.md` | Missing-data ledger (M1: Tafel slope xlsx; M2: C_S CMK-3 carbon; etc.). |
| `docs/papers/Ruggiero2022_JCatal_source_paper.md` | Peer-reviewed source paper for the Mangan deck physics: K₂SO₄, parallel 2e (0.695 V) + 4e (1.23 V) ORR, N=0.224, 1600 rpm, I=0.3 M. PDF at `docs/papers/Ruggiero2022_JCatal_manuscript.pdf`. |
| `docs/papers/seitz_mangan_data_folder_audit_2026-05-08.md` | Deep audit of the experimental data folder. |
| `docs/realignment/Mangan2025_experimental_alignment.md` | Gap audit between the model and the Mangan 2025 deck. |
| `docs/solver/bv_solver_unified_api.md` | How to call the dispatcher and configure the production stack. |
| `docs/solver/clipping_conventions.md` | Three distinct BV-related clips and the operational rule that PC is fictitious at clip=50. |
| `docs/solver/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B for the logc+counterion stack; and why `solve_anchor_with_continuation` for the multi-ion + Stern + Phase 6α/6β stack. |
| `docs/solver/steric_analytic_clo4_reduction_handoff.md` | Code-side notes on the Bikerman analytic-counterion residual closure. |
| `docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md` | Full MMS coverage table + runtime invariants + broken-config matrix (companion to the May13th writeup). |
| `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md` | Legacy ClO₄⁻ reference sweep (3sp + Bikerman + Stern + muh + debye_boltzmann IC = 15/15 over V_RHE [−0.5, +1.0]). |
| `docs/handoffs/CHATGPT_HANDOFF_30_phase6b-v9-cd-invariance/FINAL_REVISION.md` | Critique session 30 ledger (63 issues; cd-invariance finding). |
| `docs/handoffs/CHATGPT_HANDOFF_31_phase6b-v9-strategic-pivot/FINAL_REVISION.md` | Critique session 31 ledger (52 issues; APPROVED). |
| `docs/handoffs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md` | Phase 6α outcome + Phase 6β scoping (upstream of the v9 architecture). |
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
| `Forward/bv_solver/` | Main PNP-BV package: log-c forms (`forms_logc.py`) and the muh variant (`forms_logc_muh.py`), log-rate BV with parallel 2e/4e reactions, ideal + Bikerman analytic counterions (`boltzmann.py`), multi-ion shared-θ closure (`multi_ion.py`), shared scalar Picard for the `debye_boltzmann` IC (`picard_ic.py`), per-voltage diagnostics (`diagnostics.py`), observables (`observables.py`, `rrde_observables.py`), unit conversions (`units.py`), validation, the legacy C+D continuation orchestrator (`grid_per_voltage.py`), and the newer Phase 5γ+6α/6β anchor-and-grid orchestrator (`anchor_continuation.py`) with `AdaptiveLadder` and `warm_start_floor` arithmetic bisection. Phase 6α water-ionization closure in `water_ionization.py`. Phase 6β v10b cation-hydrolysis machinery in `cation_hydrolysis.py` (Γ_MOH outer Picard, Singh Eq. 4 pKa shift, Langmuir cap). The legacy concentration backend was removed in the May 2026 cleanup. |
| `calibration/` | **Firedrake-free** literature-locked constants. `v10b.py`: `GAMMA_MAX_HAT_V10B = 0.047`, `K_DES_NONDIM_V10B = 1.0`, `C_S_F_M2_V10B = 0.20`. `singh2016.py`: per-cation Singh Table S1 + Cu r_H_El values. Deprecation alias `SMOKE = V10A_SMOKE` (never `SMOKE = V10B`). |
| `Inverse/` | Generic Pyadjoint inverse framework and objective factories. **Inverse paused.** |
| `FluxCurve/` | Adjoint-gradient curve-fitting framework for Robin and BV flux/current curves. **Inverse paused — reference only.** |
| `Nondim/` | Physical constants, scaling transforms, and compatibility wrappers. |
| `Surrogate/` | Surrogate models (RBF, NN, GP, PCE, POD-RBF, multistart, BCD, cascade, ISMO). **Paused with the inverse pipeline.** |
| `scripts/_bv_common.py` | Shared constants, scales, species presets, reaction bundles, and the `make_bv_solver_params` factory. Imported by most drivers and ~30 tests. |
| `scripts/studies/drivers/` | **Re-runnable reference drivers** (since the 2026-06-15 reorg). **Phase 6β v10b:** `phase6b_step10_phase_D_orchestrate.py` + `phase6b_step10_phase_D_fit_eval.py` (Δ_β fit + identifiability gate), `phase6b_v10a_phase_A2_v_kin.py` (k_hyd ramp at V_kin), `phase6b_v10a_v_sweep_diagnostic.py` (V_kin selection), `phase6b_step6_plumbing_ablation.py` (4-path wiring). **Deck reference:** `l_eff_transport_sweep_csplus_so4.py` (Phase 6α validation), `mangan_full_grid_csplus_so4.py` (deck-page-15 band), `pass_a_grid_driver_csplus_so4.py`. **Phase 7:** `solver_demo_slide15_dual_pathway_cs.py`. **Reproductions:** `_run_jithin_*.py`. |
| `scripts/studies/plot/` | Plot generators (read results, write figures), e.g. `sensitivity_visualization.py`. |
| `scripts/studies/extract/` | Experimental-data digitizers, e.g. `extract_k_plus_tafel_slopes.py` (Tafel slopes from Brianna 2019 LSV; output to gitignored `data/derived/`). |
| `scripts/studies/*.py` (top level) | Active/ad-hoc studies not yet promoted to drivers — current **Phase 7.3** work (`phase7p3_m1a_onset_selection.py`, `phase7p3_m3_*`, …), the solver-baseline demos `solver_demo_slide15_no_speculative_cs.py` / `_ocp_shifted_cs.py`, and the ClO₄⁻ legacy reference `peroxide_window_3sp_bikerman_muh.py` (sequential R1/R2). Note: the `v*` legacy inverse studies and `_phase_D_bridge_*` throwaways were pruned in the reorg (recoverable from git history). |
| `scripts/verification/` | MMS and BV forward strategy verification scripts — **imported by the `tests/test_mms_*` suite as the MMS engine**. `mms_pnpbv_muh_multi_ion_stern.py` is the current production-stack MMS (~1100 LOC; muh + Cs⁺/SO₄²⁻ multi-ion + Stern Robin + parallel 2e/4e). `mms_bv_3sp_logc_boltzmann.py` covers the simpler 3sp stack. |
| `scripts/profile/` | Performance-profile runners for the production sweep. |
| `data/` | **Gitignored.** Experimental data drop from the Seitz/Mangan group (`EChem Reactor Modeling-Seitz-Mangan/` ~273 MB) + derived outputs (`derived/`). See `docs/papers/data_folder_code_inventory.md` for the per-file inventory. |
| `docs/` | Handoffs, plans, conventions, equations, literature inputs, and current status notes. Organized into `docs/phase6/` (Phase 6α/6β, including v10b calibration + Phase D summary + CMK-3 capacitance lit), `docs/handoffs/` (CHATGPT_HANDOFF_*), `docs/papers/` (Ruggiero, Singh, data-folder audits), `docs/realignment/` (Mangan deck alignment), `docs/solver/` (API, continuation, clipping conventions, MMS derivation), `docs/inverse/` (paused). |
| `writeups/` | PDF/TeX reports. **`May13th/`** (current): `phase_6_overview` (selectivity-gap story + Phase D verdict), `forward_codepath_demo_slide15` (visual call graph), `analytic_counterion_derivation` (multi-ion shared-θ Bikerman closure), `stern_robin_bc_derivation` (BAB 2001), `mms_source_terms_derivation` (production-stack MMS). Earlier writeups under `WeekOfApr27/`, `ForwardSolverChangesMay26/`, `vv_report/`. |
| `StudyResults/` | Generated results, summaries, plots, JSON, CSV, run logs — the working research record. Since the 2026-06-15 reorg: **active Phase 7 results stay at top level** (`phase7*`, `phase7p2_*`, `phase7p3_*`, `solver_demo_*`); closed phases are under `archive/` (`archive/phase6b/`, `archive/phase5/`, `archive/scratch/`, `archive/legacy/`); inverse series under `inverse/`, methodology under `methodology/`, simulator reproductions under `reproductions/`, loose logs under `_logs/`. |
| `tests/` | Pytest regression and verification tests. Firedrake tests are marked `slow`. MMS suite: `test_mms_logc_muh_multi_ion_stern.py` (18 cases, ~12 s; the production-stack MMS), `test_mms_steric_boltzmann_convergence.py`, `test_mms_convergence.py`. Phase 6β v9 gates: `test_phase6b_v9_gate{1_roles,2_dynamic_k,3_gamma_machinery,4_finite_hydrolysis}.py`. Phase 6β v10a/v10b: `test_phase6b_v10a_langmuir_cap.py`, `test_phase6b_v10a_phase_A2_driver.py`, `test_phase6b_v10a_v_kin_selection.py`, `test_phase6b_v10b_calibration.py`, `test_phase6b_v10b_bracket_matrix.py`. Phase D: `test_phase6b_step10_phase_D_{plumbing,fit_eval,orchestrate}.py`. Stern: `test_stern_no_stern_snapshot.py`. Multi-ion: `test_multi_ion_csplus_so4.py`. Phase 6α: `test_water_ionization_phase_6a.py`. |
| `~/PNPInverse-archive/` | **Off-git** (moved out of the repo in the 2026-06-15 reorg, history retained): old superseded runs + model checkpoints. Reference only, not the active surface. |

The paused inverse/surrogate **scripts** (`scripts/Inference/`,
`scripts/surrogate/`, the `v*` studies) were pruned in the 2026-06-15 reorg and
are recoverable from git history; the inverse **source packages** (`Inverse/`,
`Surrogate/`, `FluxCurve/`) remain in place. Re-entry point when inverse resumes:
`docs/inverse/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`.

## Core Forward-Solver Configuration

The active production stack is controlled through
`solver_params[10]`. The factory is
`scripts/_bv_common.py:make_bv_solver_params`. There are two
canonical call shapes today.

### Deck-aligned multi-ion stack (current production target)

The deck baseline is K⁺/SO₄²⁻; the snippet below uses Cs⁺/SO₄²⁻
(one of the four cations in the slide-27 comparison study). Swap
`DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` for a K⁺ entry for
apples-to-apples deck baselines (see CLAUDE.md gotchas). The
anchor is built at `C_S = 0.10` and ramped up to the production
`C_S = 0.20` via the Stern bump ladder (no form rebuild).

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
    set_stern_capacitance_model,
)
from Forward.bv_solver import (
    solve_grid_with_anchor, make_graded_rectangle_mesh,
)

# Anchor sp at C_S=0.10 (cold-convergeable); production sp at C_S=0.20.
common = dict(
    eta_hat=0.0, dt=0.25, t_end=80.0,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    bv_reactions=PARALLEL_2E_4E_REACTIONS,        # parallel 2e/4e
    boltzmann_counterions=[                       # multi-ion shared-θ closure
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    ],
    multi_ion_enabled=True,                       # required for ≥2 entries
    initializer="debye_boltzmann",
    l_eff_m=100e-6,
    enable_water_ionization=False,                # Phase 6α opt-in
)
sp_anchor   = make_bv_solver_params(**common, stern_capacitance_f_m2=0.10)
sp_baseline = make_bv_solver_params(**common, stern_capacitance_f_m2=0.20)

mesh = make_graded_rectangle_mesh(
    Nx=8, Ny=80, beta=3.0,
    domain_height_hat=sp_anchor.solver_options["bv_convergence"]["domain_height_hat"],
)

# Stage 1: cold anchor at C_S = 0.10 via k0 AdaptiveLadder
anchor_result = solve_anchor_with_continuation(
    sp_anchor.with_phi_applied(0.55 / V_T), mesh=mesh,
    k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
    initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
)

# Stage 2: Stern bump ladder 0.10 → 0.20 (no form rebuild, reuses solver)
ctx = anchor_result.ctx
for c_s in (0.20,):                               # extend rung list for >0.20
    set_stern_capacitance_model(ctx, c_s)
    ctx["_last_solver"].solve()

# Stage 3: grid walk warm-started from the bumped anchor
anchor = extract_preconverged_anchor(
    anchor_result, phi_applied_eta=0.55 / V_T,
    k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
    mesh_dof_count=anchor_result.ctx["U"].function_space().dim(),
)
grid = solve_grid_with_anchor(sp_baseline, mesh=mesh, anchor=anchor,
                              v_rhe_grid=V_RHE_GRID)
```

To enable Phase 6β v10b cation hydrolysis, set
`enable_cation_hydrolysis=True` and use
`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4` (K⁺ as a dynamic NP species),
plus the v10b literature constants from `calibration.v10b`:

```python
from calibration.v10b import (
    GAMMA_MAX_HAT_V10B,    # 0.047
    K_DES_NONDIM_V10B,     # 1.0
    C_S_F_M2_V10B,         # 0.20
)
from scripts._bv_common import FOUR_SPECIES_LOGC_DYNAMIC_K2SO4, PARALLEL_2E_4E_REACTIONS_4SP
```

Reference drivers:

| Driver | Stack |
|---|---|
| `scripts/studies/solver_demo_slide15_no_speculative_cs.py` | **Solver-works baseline.** Cs⁺/SO₄²⁻ + parallel 2e/4e + Stern bump, default-off Phase 6α/6β; 4-factor `K0_R4e` × 25 V_RHE. Visual call graph: `writeups/May13th/forward_codepath_demo_slide15.pdf`. (Stayed at `studies/` top level.) |
| `scripts/studies/drivers/phase6b_step10_phase_D_orchestrate.py` | Phase D K-only Δ_β fit + identifiability gate (`OUTCOME_C_NON_IDENTIFIABLE_flagged` on K@pH4 deck data). |
| `scripts/studies/drivers/phase6b_v10a_phase_A2_v_kin.py` | 10-rung k_hyd ramp at V_kin = −0.10 V on v10a/v10b parameters; `--lambda-ladder` CLI per step 9.A. |
| `scripts/studies/drivers/phase6b_v10a_v_sweep_diagnostic.py` | V_kin selection diagnostic (primary route: σ_S<0, branch active, not transport-artifact, not cap-dominated). |
| `scripts/studies/drivers/phase6b_step6_plumbing_ablation.py` | 4-path wiring verification (form-build residual, pKa-shift context, Picard Γ update, diagnostics). |
| `scripts/studies/drivers/l_eff_transport_sweep_csplus_so4.py` | Phase 6α validation: 8 L_eff × 13 V_RHE with `--enable-water-ionization`. |
| `scripts/studies/drivers/mangan_full_grid_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e at the deck V_RHE band. |

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
- **Stern bump ladder for `C_S = 0.20`**: cold-start at `C_S = 0.20`
  on the multi-ion stack is unreliable. Build the anchor at
  `C_S = 0.10`, then ramp via verified rungs
  `(0.10 → 0.20 → 0.50 → 1.0 → 2.0 → 5.0 → 10.0 → 100.0)` using
  `set_stern_capacitance_model(ctx, c_s)` + `ctx["_last_solver"].solve()`
  (no form rebuild). The bump-ladder helper truncates the rung list
  at the first rung ≥ target.
- **Bikerman `a_nondim` for dynamic species is unphysical**:
  O₂/H₂O₂/H⁺ are seeded with `A_DEFAULT = 0.01` (≈ r 14.9 Å,
  ~150× larger than physical). Only counterion entries
  (`A_KPLUS_HAT`, `A_CSPLUS_HAT`, `A_SO4_HAT`, `A_OH_HAT`) use real
  radii. H⁺ Bikerman cap is clamped ~150× tighter than its r=2.8 Å
  H₃O⁺ Stokes value — treat any plateau-set-by-Levich finding as
  suspect until the `_phase_D_bridge_corrected_a*.py` runs land.
- **Phase 6α opt-in**: `enable_water_ionization=True` plus the
  `kw_eff_ladder` outer loop on `solve_anchor_with_continuation`.
  Retired as primary peak mechanism (P3 fail) but plumbing remains.
- **Phase 6β v10b cation hydrolysis** is calibrated and convergent
  but the **Δ_β fit step (Phase D) returned non-identifiable**. All
  hard gates (cd<0, R_4e sign preserved, R_net≥0, mass-balance
  ≤ 5×10⁻³) pass, but model max H₂O₂% overshoots deck mean by a
  uniform +15.6 pp. Don't claim Δ_β-derived cation-trend conclusions
  yet; don't launch Phase E. See `phase6b_step10_phase_D_summary.md`.
- **`AdaptiveLadder.warm_start_floor`** (step 9.5) is opt-in
  arithmetic bisection at the first ladder rung when warming from a
  prior state — required for λ ladders at high `k_hyd`. k0 and
  `kw_eff` ladders keep `warm_start_floor=None` (byte-equivalent
  to pre-9.5 behavior).
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
- `H2O2_SEED_NONDIM = 1e-4` is the production seed for `ln c_H2O2`
  at the bulk Dirichlet BC, not a physics tweak. **MMS tests rescale
  this to `1.0`** to keep Newton in basin without continuation; the
  operator under test is unchanged (the seed enters only the bulk
  Dirichlet, an `O(10⁻⁵)` `θ_b` contribution, and the anodic
  `c_ref` reference which neither reaction enters).
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

   and finally a smoke test of the solver-works baseline (Cs⁺/SO₄²⁻
   + parallel 2e/4e + Stern bump, default-off Phase 6α/6β):

   ```bash
   python scripts/studies/solver_demo_slide15_no_speculative_cs.py
   ```

   The full 4-factor × 25-V sweep takes 15-30 min; see
   `scripts/profile/profile_production_sweep.py` for a profile-only
   variant.

All commands in this README assume the Firedrake venv is active and
the working directory is `PNPInverse/`.

## Common Commands

Lightweight tests (no Firedrake):

```bash
python -m pytest -m "not slow"
```

Firedrake-dependent verification (MMS for the production stack):

```bash
# Production stack: muh + Cs⁺/SO₄²⁻ multi-ion + Stern Robin + parallel 2e/4e
python -m pytest -m slow tests/test_mms_logc_muh_multi_ion_stern.py

# Legacy single-counterion stack (3sp + Bikerman)
python -m pytest -m slow tests/test_mms_steric_boltzmann_convergence.py
python scripts/verification/mms_bv_3sp_logc_boltzmann.py
```

Solver-works baseline (current canonical demo; Cs⁺/SO₄²⁻ + parallel
2e/4e + Stern bump, default-off Phase 6α/6β; visual call graph in
`writeups/May13th/forward_codepath_demo_slide15.pdf`):

```bash
python -u scripts/studies/solver_demo_slide15_no_speculative_cs.py
# --no-stern variant ramps Stern through 0.10 → 100 F/m²
```

Phase 6β v10b drivers (cation hydrolysis on calibrated parameters):

```bash
# Step 10 Phase D K-only Δ_β fit (returns OUTCOME_C_NON_IDENTIFIABLE_flagged)
python -u scripts/studies/drivers/phase6b_step10_phase_D_orchestrate.py

# Step 9 B.2 — 14×10 = 140-rung k_hyd × λ ramp at V_kin
python -u scripts/studies/drivers/phase6b_v10a_phase_A2_v_kin.py --lambda-ladder

# Step 6 plumbing ablation (4-path wiring verification at V_kin)
python -u scripts/studies/drivers/phase6b_step6_plumbing_ablation.py
```

Deck-aligned reference sweeps:

```bash
# Phase 6α validation: 8 L_eff × 13 V_RHE, --enable-water-ionization opt-in
python -u scripts/studies/drivers/l_eff_transport_sweep_csplus_so4.py \
    [--enable-water-ionization]

# Deck-page-15 V_RHE band (Cs⁺/SO₄²⁻ + parallel 2e/4e, single L_eff)
python -u scripts/studies/drivers/mangan_full_grid_csplus_so4.py
```

K⁺ Tafel slope extraction (Phase F, parallel-safe):

```bash
python -u scripts/studies/extract/extract_k_plus_tafel_slopes.py
# outputs go to data/derived/ (gitignored)
```

Legacy ClO₄⁻ reference sweep (single-counterion + sequential R1/R2,
for backward-compat checking only — **not** a deck-aligned run):

```bash
python -u scripts/studies/peroxide_window_3sp_bikerman_muh.py
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
- After the 2026-06-15 reorg, re-runnable drivers live under
  `scripts/studies/drivers/`, plotters under `scripts/studies/plot/`,
  extractors under `scripts/studies/extract/`. Older handoff/notes may
  cite pre-reorg paths (`scripts/studies/<driver>.py`, `scripts/Inference/`,
  `scripts/surrogate/`, `scripts/derive/`) — recoverable from git history.
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
- **Stern bump ladder for `C_S = 0.20`**: cold-start at the
  production `C_S = 0.20` is unreliable on the multi-ion stack.
  Build the anchor at `C_S = 0.10` and ramp via the verified rung
  sequence `(0.10 → 0.20 → 0.50 → 1.0 → 2.0 → 5.0 → 10.0 → 100.0)`
  using `set_stern_capacitance_model` (no form rebuild). The
  100 F/m² rung approximates the no-Stern Dirichlet limit; a true
  `C_S = None` Dirichlet doesn't converge on the multi-ion stack.
- **Bikerman `a_nondim` for dynamic species is unphysical** —
  O₂/H₂O₂/H⁺ use `A_DEFAULT = 0.01` (≈ r 14.9 Å, ~150× larger than
  physical). Only counterion entries use real radii. Anything that
  depends on the H⁺ Levich plateau is suspect until the
  `_phase_D_bridge_corrected_a*.py` runs disambiguate.
- **Phase 6α water-ionization is opt-in** but retired as primary
  peak mechanism. When using `enable_water_ionization=True`, pass
  the `kw_eff_ladder` outer loop to `solve_anchor_with_continuation`
  (typical: `(0.0, KW_HAT·1e-6, KW_HAT·1e-3, KW_HAT·0.1, KW_HAT)`).
- **Phase 6β v10b cation hydrolysis** is calibrated and convergent
  (Γ_max=0.047, k_des=1.0, C_S=0.20 from `calibration/v10b.py`), all
  hard gates pass, but **step 10 Phase D K-only Δ_β fit returned
  `OUTCOME_C_NON_IDENTIFIABLE_flagged`** with a uniform +15.6 pp
  H₂O₂% overshoot vs deck K@pH 4. Phase E predictive screen
  (Cs/Na/Li holdout) MUST NOT launch on the current Δ_β. See
  `docs/phase6/phase6b_step10_phase_D_summary.md`.
- **`AdaptiveLadder.warm_start_floor`** (step 9.5) is opt-in
  arithmetic bisection at the first ladder rung when warming from
  a prior state — required for λ ladders at high `k_hyd`. k0 and
  `kw_eff` ladders keep `warm_start_floor=None` (byte-equivalent
  to pre-9.5 behavior).
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
