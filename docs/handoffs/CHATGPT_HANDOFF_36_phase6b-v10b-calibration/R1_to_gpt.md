# R1 → GPT: Critique session 36 — Phase 6β step 8 (v10b) plan

You are an adversarial reviewer for a multi-day execution plan in a
PNP-BV research codebase.  Your job: find every hole.  Verdict at
the end is binary (APPROVED / ISSUES_REMAIN); use APPROVED only if
nothing remaining is blocking.

---

## Section 1 — Context bundle

### 1.1. Project shape

**Repo:** `PNPInverse` (FireDrakeEnvCG branch).  Research code for
**Poisson–Nernst–Planck / Butler–Volmer (PNP-BV)** forward simulation
for ORR (O₂ → H₂O₂ → H₂O) on the Seitz/Mangan deck (**K₂SO₄ at pH
4–6, parallel 2e⁻ / 4e⁻ topology per Ruggiero 2022**).

Inverse work is paused; v10b is forward-only.

**Production stack (May 2026):**
- 3 dynamic species (O₂, H₂O₂, H⁺) + analytic Bikerman counterion(s).
- `formulation='logc_muh'` — proton electrochemical-potential primary
  variable `μ_H = u_H + em·z_H·φ`.
- Log-rate Butler–Volmer, **parallel R2e (E° = 0.695 V) + R4e
  (E° = 1.23 V vs RHE)**.
- Stern capacitance `C_S = 0.20 F/m²` (Bohra-Koper-Choi consensus,
  locked in step 7).
- `debye_boltzmann` IC; reaches V_RHE = +1.0 V at 15/15 via C+D.

### 1.2. Phase 6β arc — chronology

Phase 6β scope is **cation hydrolysis at the polarized OHP** per
Singh 2016 (`M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺` with field-dependent pKa).
Locked acceptance bundle:
`docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`.

Step sequence (per acceptance bundle § "v10a → E sequence"):

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

### 1.3. The closed-form Γ_ss formula (v10a Langmuir cap)

The v10a Picard-converged closed form for surface coverage is:

```
Γ_ss(λ) = λ·F₀ / ((1−λ) + λ·k_des + λ·B + λ·F₀/Γ_max)
```

where:
- `F₀ = k_hyd · F_avg` is the forward hydrolysis driver
  (`F_avg` aggregates K⁺ enrichment + Singh pKa shift + B-related
  proton concentration).
- `k_des` is the desorption rate of MOH → bulk.
- `B = k_prot · c_H_avg · δ_OHP_hat` is the proton-mediated back-
  reaction term.
- `Γ_max` is the Langmuir saturation cap.
- `λ` is the homotopy parameter (0 → no hydrolysis, 1 → full).
- `θ = Γ/Γ_max` is the dimensionless coverage.

Mass-balance residual is the difference between the closed-form
`Γ_ss` and the solver's recovered `Γ` after Picard.  At v10a A.2
this was at machine precision (1e-14 to 1e-16) across the grid.

### 1.4. Smoke values that v10b is replacing

| Name | Smoke value | Source |
|---|---|---|
| `Γ_max_hat` | 0.047 nondim | 1 monolayer of K⁺ MOH at the OHP, hard-sphere with r ≈ 2.3 Å |
| `k_des_nondim` | 1.0 | No literature anchor; pure engineering choice |
| `C_S` | 0.10 F/m² (legacy) → 0.20 F/m² (step 7 lock) | Bohra/Choi/Pillai 2024 consensus |

**Γ_max derivation (in `Forward/bv_solver/cation_hydrolysis.py:232-242`):**
```
Γ_max_phys  ≈ 1 / (π · (2.3e-10 m)² · N_A) ≈ 5.6e-6 mol/m²
            (hard-sphere monolayer, r ≈ K⁺ hydrated radius 2.3 Å)
Γ_max_hat   = Γ_max_phys / (C_SCALE · L_REF)
            = 5.6e-6 / (1.2 · 1e-4)
            ≈ 0.047
```
where C_SCALE = 1.2 mol/m³ and L_REF = 1e-4 m (100 µm).

### 1.5. Hard invariants (CANNOT be touched in v10b without triggering re-derivation cascades)

- `V_kin = −0.10 V` (locked at step 4; v10a' V-sweep).
- `K0_R4e_factor = 1e-14` (locked at step 4; branch-pass probe).
- `k_hyd_baseline = 1e-3 nondim` (locked at step 5).
- `WARM_WALK_GRID = (+0.55, +0.40, +0.20, +0.10, −0.10)`.
- `LAMBDA_LADDER = (0.0, 0.25, 0.50, 0.75, 1.0)`.
- Parallel topology R2e (E°=0.695 V) + R4e (E°=1.23 V).
- `exponent_clip = 100.0` (PC-trustworthy setting; not 50).
- `STERN_F_M2_ANCHOR = 0.10 F/m²` — convergence-friendly value used
  in the two-stage anchor pattern.  The pattern: build anchor at
  0.10, then runtime-bump to target via `set_stern_capacitance_model`
  + Newton resolve.  This is required because
  `solve_anchor_with_continuation` raises `NotImplementedError` when
  `c_s_ladder` is combined with `kw_eff_ladder` (the proton-side
  Phase 6α ladder).

### 1.6. Key file paths

- `Forward/bv_solver/cation_hydrolysis.py` — defines
  `GAMMA_MAX_HAT_SMOKE = 0.047` at line 242; the bundle build
  function `build_cation_hydrolysis_terms`; `gamma_ss_langmuir`;
  `update_gamma_from_solution`.
- `Forward/bv_solver/anchor_continuation.py` —
  `solve_anchor_with_continuation` (with `c_s_ladder`,
  `kw_eff_ladder`, `lambda_hydrolysis` outer loops);
  `solve_grid_with_anchor`;
  `_PARAMETER_OVERRIDE_SETTERS` at line 1689; runtime setters
  `set_reaction_gamma_max_model`, `set_reaction_k_des_model`,
  `set_stern_capacitance_model`.
- `scripts/_bv_common.py` — defines a **second copy** of
  `GAMMA_MAX_HAT_SMOKE = 0.047` at line 944; the factory
  `make_cation_hydrolysis_config` at line 957 (default
  `gamma_max_nondim = GAMMA_MAX_HAT_SMOKE`); 12 legacy scripts
  that still pass `stern_capacitance_f_m2 = 0.10`.
- `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` — defines
  `SMOKE_KINETICS` dict at line 174 (includes `k_des_nondim=1.0`,
  `gamma_max_nondim=0.047`); `STERN_F_M2_BASELINE = 0.20` at line
  89 and `STERN_F_M2_ANCHOR = 0.10` at line 106; the two-stage
  anchor pattern in `_build_sp_at_cs` at line 905.
- `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` — Phase A.2
  driver.  Imports `SMOKE_KINETICS` from the V-sweep driver
  (line 1293).  Computes `v10b_priorities` block including
  `k_hyd_route`, `single_v_selectivity_gap_pp`, `max_amp_from_singh`,
  `transport_re_entry_first_k_hyd`.
- `scripts/studies/phase6b_step6_plumbing_ablation.py` — step 6
  driver (5 ablations: H source, K sink, σ override, etc.).
- `tests/test_phase6b_v10a_langmuir_cap.py` — has a cross-file
  sanity test at lines 46-58 asserting the two
  `GAMMA_MAX_HAT_SMOKE` copies agree.  Pins literal `0.047` at
  lines 90, 123, 132, 147.
- `tests/test_phase6b_v10a_phase_A2_driver.py` — 43 fast tests.
- `tests/test_phase6b_step6_plumbing_ablation*.py` — step 6 tests.
- `docs/phase6/CMK3_capacitance_literature.md` — step 7 writeup;
  v10b writeup mirrors its structure (7 sections: citation chain
  + 3 regimes + caveats + sensitivity bracket + implementation
  status + open asks + cross-references).

### 1.7. Phase A.2 v10a baseline numbers (for v10b regression comparison)

At V_kin = −0.10 V, λ=1, k_hyd_baseline = 1e-3, smoke kinetics:
- `cd_mA_cm² ≈ −3.12`.
- `x_2e ≈ 0.199`.
- `θ(k_hyd=1e-3) ≈ 0.86` (transition rung).
- `θ(k_hyd=1e-1) ≈ 0.998` (cap-saturated).
- `o2_flux_levich_ratio ≈ 0.63` (no transport re-entry).
- `single_v_selectivity_gap_pp = +5.09 pp` (H₂O₂% = 19.91% vs
  deck band [25, 50]%).
- Mass-balance residual 1e-14 to 1e-16 across all 10 rungs.
- `convergence_audit.overall_pass = False` is a **threshold-
  narrowness artifact** (max θ in transition grid = 0.9253; cutoff
  0.93).  Known issue, not a real regression.

### 1.8. Step 6 plumbing-ablation contract

All 5 ablations pass.  A0 (no ablation) is byte-equivalent (rel ≤
1e-6 in cd, R_net, σ_S, θ) to the Phase A.2 baseline at λ=1,
k_hyd=1e-3, V=V_kin.  A2 finding: K⁺ Boltzmann pile-up at V_kin
makes `c_K_boundary_avg ≈ 291 · c_K_bulk` — sentinel-scale R_inj
perturbations cannot dent boundary c_K by 5%.  **v10b must NOT
use boundary-c_K perturbation at V_kin as a k_des calibration
diagnostic.**  Use cap-saturated θ instead.

### 1.9. Literature pointers (for the v10b literature pass)

In `data/EChem Reactor Modeling-Seitz-Mangan/Articles/` (confirmed
via `ls`):
- `2019-Co-Zhang-Direct Evidence of Local pH Change … CO2 Electroreduction
  in Aqueous Media-Angewandte.pdf` (cation-pH support).
- `2012-Nørskov-Viswanathan-Unifying the 2e– and 4e– Reduction of Oxygen
  on Metal Surfaces-JPCL.pdf` (ORR Sabatier framework).
- `2017-Co-Billy-Experimental Parameters Influencing Hydrocarbon
  Selectivity during the Electrochemical Conversion of CO2-ACS Catal.pdf`.
- 2015/2017 Lewis-Singh papers on near-neutral electrolyte schemes.
- **Bohra 2019 EES (`10.1039/c9ee02485a`) is NOT present** — an
  open ask from step 7.

In `data/.../Linsey/`:
- `20200407_Electrochemical Double Layer Modeling_LSeitz.pdf`
  (group's earlier EDL modeling doc).
- ButlerVolmer MATLAB code.

In `data/.../Yash-Trends/`:
- `Data and Plotting.zip` containing a parallel 6-species PNP+BV
  with K⁺/SO₄²⁻ explicit (`reference_yash_modeling_code.md`
  memory).  Cross-validation reference.

`Parameters_Seitz_Mangan.xlsx` in the data folder root — may
contain group-internal Γ_max convention.

`docs/phase6/singh_2016_pka_formula.md` — Singh 2016 SI Eq. (3)/(4)
derivation + σ-mapping convention.  Already integrated for ΔpKa;
v10b's research questions for Singh 2016: (a) does Table S1 give
partial-coverage estimates from CV for K⁺?  (b) does Singh report
a forward `k_hyd` rate constant from which detailed balance gives
`k_des`?

### 1.10. Cross-cutting decisions already made

- **Selectivity gap is NOT a v10b pass criterion.**  Phase A.2
  flagged the gap (+5.09 pp) as v10b LOW priority routing.  v10b
  is about literature anchoring; Phase D (step 10) is where
  selectivity gets data-fitted.
- **v10b is MANDATORY** regardless of A.2 priority routing (per
  acceptance bundle § "v10a → E sequence").
- **k_hyd_baseline = 1e-3** is the canonical λ=1 anchor.
  `k_hyd_route = 1e-1` (the highest k_hyd satisfying θ>0.95 etc.)
  is the cap-saturated rung used for diagnostics.

---

## Section 2 — The artifact under review

The plan below was just written.  It targets all of step 8.

````````````````````````markdown
# Phase 6β Step 8 — v10b Literature Calibration of Γ_max + k_des + C_S

**Author:** Claude (planner).  **Date:** 2026-05-10.
**Source handoff:** `docs/handoffs/v10b_planning_handoff.md`.
**Status:** Draft v1 — entering GPT critique loop (≤ 7 rounds).

---

## 0. One-paragraph framing

v9/v10a used three smoke values that have not been pinned to peer-
reviewed literature: `Γ_max_hat = 0.047` (1-monolayer K⁺ MOH at the
OHP, hard-sphere derivation), `k_des_nondim = 1.0` (no anchor at
all), and `C_S = 0.10 F/m²` (a convergence-pinned engineering value).
Step 7 already locked `C_S = 0.20 F/m²` via
`docs/phase6/CMK3_capacitance_literature.md`.  v10b's job is to
deliver literature-anchored numeric values (or explicit engineering-
choice flags with documented priors) for **all three** parameters,
update the code defaults, regenerate the Phase A.2 + step 6 plumbing
baselines at v10b parameters, run a C_S sensitivity bracket sweep,
and publish `docs/phase6/v10b_calibration_summary.md`.  Selectivity-
gap improvement is **not** a v10b pass criterion (that's Phase D).

---

## 1. Definition of done (lifted from handoff §11, refined)

v10b is **done** iff **every** box ticks:

- [ ] **D1.** `Γ_max`, `k_des`, `C_S` each carry one of:
      *(a)* peer-reviewed literature anchor with citation chain, **or**
      *(b)* explicit `engineering_choice = True` flag with documented
      prior, bracket sweep evidence, and "data-constrained in Phase
      D" note.
- [ ] **D2.** `docs/phase6/v10b_calibration_summary.md` published.
      Three-parameter analog of `CMK3_capacitance_literature.md`:
      per-parameter citation chain + caveats + sensitivity bracket +
      implementation status + open asks + cross-references.
- [ ] **D3.** `GAMMA_MAX_HAT_SMOKE` constant in
      `Forward/bv_solver/cation_hydrolysis.py:242` **and** in
      `scripts/_bv_common.py:944` updated (the cross-file sanity test
      at `tests/test_phase6b_v10a_langmuir_cap.py:46-58` must still
      pass).  Rename to `GAMMA_MAX_HAT_V10B` with citation comment;
      keep `GAMMA_MAX_HAT_SMOKE` as a deprecated alias mapped to the
      v10b value for one cycle, then delete after step 9 (B.2) lands.
- [ ] **D4.** `SMOKE_KINETICS` dict in
      `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py:174-182`
      renamed to `V10B_KINETICS` with the new numeric values and the
      same key set.  Backward-compat shim
      `SMOKE_KINETICS = V10B_KINETICS` kept for one cycle so importers
      (`phase6b_v10a_phase_A2_v_kin.py:1293`,
      `phase6b_step6_plumbing_ablation.py`) don't break atomically.
- [ ] **D5.** Phase A.2 driver re-run at v10b params (anchor at
      `STERN_F_M2_ANCHOR=0.10`, bump to `STERN_F_M2_BASELINE=0.20`,
      new `Γ_max` + `k_des`).  10/10 k_hyd rungs converge at λ=1.0;
      Picard converges everywhere; **mass-balance residual < 5e-3
      across the grid**.  New JSON committed at
      `StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json`.
      Qualitative shape (θ rises from ≈ pre-cap to ≈ saturation; cd
      ≈ −3.12 mA/cm²; x_2e ≈ 0.20) preserved within ±20%.
- [ ] **D6.** Step 6 plumbing-ablation driver re-run at v10b params.
      All 5 ablations pass; A0 byte-equivalent to the new v10b A.2
      baseline.  New JSON at
      `StudyResults/phase6b_v10b_step6_plumbing_ablation/ablation_matrix.json`.
- [ ] **D7.** C_S sensitivity bracket sweep across
      `C_S ∈ {0.05, 0.10, 0.20, 0.30} F/m²` at V_kin = −0.10 V, λ=1,
      k_hyd_baseline = 1e-3.  All four rungs converge cleanly; no
      spurious sign flips on R_net, R_2e, R_4e, σ_S.  Single new
      driver `scripts/studies/phase6b_v10b_cs_bracket.py` +
      JSON at `StudyResults/phase6b_v10b_cs_bracket/cs_bracket.json`.
- [ ] **D8.** `pytest -m "not slow" -k "phase6b or cation"` green at
      the new constants.  Tests that pinned literal 0.047 in
      `test_phase6b_v10a_langmuir_cap.py:90,123,132,147` refactored
      to reference the constant (`GAMMA_MAX_HAT_V10B`), not a
      literal, so future calibration cycles don't churn the tests.
- [ ] **D9.** Acceptance bundle § Status appended with the v10b
      paragraph at
      `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`.
- [ ] **D10.** `CLAUDE.md` "Recent progress" updated; total file
      length ≤ 200 lines (current is 199 — must consolidate).
- [ ] **D11.** Memory entry `project_v10b_calibration_outcome.md`
      added to
      `~/.claude/projects/-Users-jakeweinstein-Desktop-ResearchForwardSolverClone-FireDrakeEnvCG-PNPInverse/memory/`
      with index pointer in `MEMORY.md`.

**Selectivity gap (`single_v_selectivity_gap_pp`) is NOT a v10b pass
criterion.**  Phase A.2 baseline at +5.09 pp is documented; if v10b
moves it closer to or further from the deck band [25, 50] %, that's
data for Phase D, not a v10b verdict.

---

## 2. Hard invariants (do NOT touch in v10b)

From handoff §6; breaking any of these invalidates A.2 and step 6
and triggers a step-4/5/6 re-derivation cascade.

| Constant | Value | Source |
|---|---|---|
| `V_kin` | `−0.10 V` | step 4 (v10a' V-sweep diagnostic) |
| `K0_R4e_factor` | `1e-14` | step 4 (v10a' branch-pass probe) |
| `k_hyd_baseline` | `1e-3 nondim` | step 5 (Phase A.2) |
| `WARM_WALK_GRID` | `(+0.55, +0.40, +0.20, +0.10, −0.10)` | A.2 + step 6 driver |
| `LAMBDA_LADDER` | `(0.0, 0.25, 0.50, 0.75, 1.0)` | v10a |
| Parallel topology | `R2e (E°=0.695 V)` + `R4e (E°=1.23 V)` | Ruggiero 2022; CLAUDE.md hard rule #4 |
| `exponent_clip` | `100.0` | CLAUDE.md hard rule #2 |
| `STERN_F_M2_ANCHOR` | `0.10 F/m²` | two-stage anchor pattern (v10a') |
| `c_s_ladder + kw_eff_ladder` combo | unsupported | `_PARAMETER_OVERRIDE_SETTERS` at `anchor_continuation.py:1689` (still raises `NotImplementedError`) |

If v10b's new `C_S = 0.20 F/m²` (or any new `Γ_max` / `k_des`)
shifts the σ_S manifold enough that V_kin or `K0_R4e_factor` no
longer satisfy the locked rule, that is a **step 4 + 5 + 6 re-
trigger, not a v10b in-scope change**.  Detection: A.2 regression
in D5 fails the qualitative-shape check, or step 6 ablation 1/5
fails.  Action: escalate, document as v10c, do not push past.

---

## 3. The three parameters — decision rules

### 3.1. `C_S` — already locked at step 7

**Recommendation locked: `C_S = 0.20 F/m²`** per
`docs/phase6/CMK3_capacitance_literature.md`.

**v10b work for C_S:**
1. **No literature search needed** — step 7 did it.
2. **Audit and update legacy call-sites** that still pass
   `stern_capacitance_f_m2 = 0.10` in deck-aligned multi-ion scripts.
   From grep: 12 call-sites total.  Classify each:
   * **Deck-aligned multi-ion (K⁺/SO₄²⁻ or Cs⁺/SO₄²⁻) →** update to
     0.20 F/m² (with two-stage anchor pattern if convergence
     regresses).  Candidates:
     `pass_a_grid_driver_csplus_so4.py`,
     `mangan_full_grid_csplus_so4.py`,
     `peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`,
     `picard_residual_consistency_csplus_so4.py`,
     `picard_residual_consistency_csplus_so4_lowk0.py`,
     `l_eff_transport_sweep_csplus_so4.py`,
     `k0_r4e_ratio_sweep_csplus_so4.py`,
     `anchor_smoke_csplus_so4.py`,
     `anchor_smoke_csplus_so4_continuation.py`,
     `multi_ion_ic_debug.py`.
   * **Legacy ClO₄⁻ single-counterion (CLAUDE.md hard rule #6
     explicitly keeps 0.10 for byte-equivalence):** leave alone.
     Candidates:
     `peroxide_window_3sp_bikerman_muh.py` and similar.
   * **Branch-probe / parallel-2e/4e warm-start probe:** classify
     case-by-case based on whether the StudyResults snapshot is
     load-bearing.
   For each updated script: add a one-line comment "v10b: C_S
   bumped 0.10 → 0.20 F/m², Bohra-Koper-Choi consensus
   (`docs/phase6/CMK3_capacitance_literature.md`); convergence
   verified at … if applicable."  No silent updates.
3. **Sensitivity bracket sweep** across `{0.05, 0.10, 0.20, 0.30}`
   F/m² at V_kin = −0.10 V (D7).  Two-stage anchor pattern:
   * Anchor at `STERN_F_M2_ANCHOR = 0.10` (this is the locked
     convergence-friendly value; do NOT replace).
   * Bump to target C_S via `set_stern_capacitance_model(ctx, …)`
     + Newton resolve per `phase6b_v10a_phase_A2_v_kin.py:1025-1035`
     pattern.
   * For C_S = 0.10 rung specifically: still go through the two-
     stage pattern (anchor at 0.10, "bump" to 0.10 is a Newton
     no-op) — keeps the four rungs methodologically uniform.
4. **Carry forward step 7's three open asks:**
   * Pull Bohra 2019 EES (`10.1039/c9ee02485a`) into
     `data/EChem Reactor Modeling-Seitz-Mangan/Articles/` (it is
     **not** there — confirmed via `ls Articles/`).  Cited by
     Ruggiero 2022 ref 71 *and* Linsey 2025 deck slide 13.
   * Re-derive Risk #5 σ_S mismatch with Stern-only C_S = 20
     µF/cm² (not Singh's 51 µF/cm² Cu CV-slope total C_dl).
   * Decide Yash convention (L_Stern + ε_S=11.3 ⇒ C_S=0.17 F/m²
     vs C_S=0.20 F/m² Choi).  At ε_S = 6 (Conway oriented-water)
     this drops to 0.088 F/m² — flag the implicit ε_S assumption
     in the writeup.

### 3.2. `Γ_max` — literature search + likely tighten citation chain

**Current smoke:** `0.047 nondim` =
`Γ_max_phys / (C_SCALE · L_REF) = 5.6e-6 mol/m² / (1.2 · 1e-4)`
where `5.6e-6 mol/m² ≈ 1/(π · (2.3e-10 m)² · N_A)` is one monolayer
of K⁺ MOH at the OHP using hard-sphere packing with K⁺ hydrated
radius 2.3 Å.

**Likely v10b outcome:** the literature confirms 1 monolayer is the
right order of magnitude; v10b is a **citation-chain tightening**
deliverable, not a value change.  If true, the writeup is the load-
bearing artifact (the constant doesn't move; D3 is mostly a
docstring + comment update).

**Literature search targets (decision rule below):**
1. **Singh et al. 2016 *JACS* 138:13006**
   (`10.1021/jacs.6b07612`) — does Table S1 give partial-coverage
   estimates from CV for K⁺?  Does it give a `k_hyd` rate constant
   (relevant to §3.3)?  This paper is already integrated as the
   pKa-shift mechanism source per
   `docs/phase6/singh_2016_pka_formula.md`.
2. **Iamprasertkun 2019 *JPCL* `10.1021/acs.jpclett.8b03523`**
   — HOPG basal + alkali cation specific-C measurements (cited in
   step 7's CMK3 writeup as 4.7–9.4 µF/cm² range).  Look for
   surface-coverage estimates at the cathodic edge.
3. **Bohra 2019 EES `10.1039/c9ee02485a`** (still an open ask;
   pull into `Articles/` first).  Cation-hydration steric model
   may give an alternative Γ_max derivation.
4. **Co-Zhang 2019 Angewandte** (already in `Articles/`) — cation-
   pH experimental support.  Probably no Γ_max directly, but
   contextual.
5. **Yash modeling code** at `data/EChem Reactor Modeling-Seitz-
   Mangan/Yash-Trends/` (per memory entry
   `reference_yash_modeling_code.md`) — parallel 6-species PNP+BV
   with K⁺/SO₄²⁻ explicit.  Check what surface-coverage convention
   Yash uses.
6. **`Parameters_Seitz_Mangan.xlsx`** in the data folder root —
   may have group-internal Γ_max convention.

**Decision rule (commit before research):**
* IF a peer-reviewed source gives a transportable Γ_max for K⁺ at
  sp²-carbon-cathode OHP within a factor of ~3 of 0.047 nondim
  (i.e. between 0.015 and 0.15) → **lock the cited value**.
* ELSE IF the hard-sphere monolayer derivation can be defended
  with a tighter citation chain (e.g. Singh Table S1 confirms 2.3
  Å hydrated radius for K⁺) → **keep 0.047 nondim**, tighten the
  derivation chain in the comment + writeup.
* ELSE → **mark as engineering choice** with prior `Γ_max ∈
  {0.02, 0.047, 0.1}` (half-monolayer, hard-sphere monolayer, two-
  monolayer-allows-local-enrichment); run sensitivity bracket
  sweep.

**Sensitivity bracket (optional D7 extension):**
`Γ_max ∈ {0.02, 0.047, 0.1, 0.2}` nondim — half-monolayer, hard-
sphere monolayer, two-monolayer, high-stretching test.  Run **only
if** the decision tree above lands on "keep 0.047" or "engineering
choice"; skip if a single literature value is locked.

### 3.3. `k_des` — research + likely fall back to engineering choice

**Current smoke:** `1.0 nondim` (no anchor at all).

**The hard problem.** No obvious peer-reviewed `k_des` value exists
for `M(OH)⁰ → M⁺ + OH⁻` at the polarized OHP.  Singh 2016's pKa is
an equilibrium quantity (`K_eq = k_hyd / k_des`); rates require
extra information.

**Research strategies (commit to order of attempt):**
1. **Detailed balance from `K_eq + k_hyd` (highest priority).**  If
   Singh 2016 reports `k_hyd` (forward hydrolysis rate constant) at
   any field strength, detailed balance gives `k_des = k_hyd /
   10^(pKa_eff)`.  Scan Singh 2016 SI specifically for kinetics.
2. **Analogous reactions from CO2R/ORR adsorbate-desorption
   literature.**  `OH*` desorption rates on sp²-carbon with cation
   stabilization correction.  Specific papers to grep:
   * Nørskov-Viswanathan 2012 *JPCL* (in `Articles/`,
     `2012-Nørskov-Viswanathan-Unifying the 2e– and 4e– Reduction
     of Oxygen on Metal Surfaces-JPCL.pdf`) — has the unified
     ORR Sabatier framework.
   * Co-Billy 2017 ACS Catal (in `Articles/`,
     `2017-Co-Billy-Experimental Parameters Influencing
     Hydrocarbon Selectivity during the Electrochemical Conversion
     of CO2-ACS Catal.pdf`).
   * Sabatier-volcano references via Nørskov group.
3. **Diffusion-limited upper bound + Eyring lower bound.**  Upper
   bound: `k_des ~ D_M+ / δ_OHP` (barrier-less limit).  Lower
   bound: `k_BT/h · exp(−ΔG_des/RT)` with ΔG_des from cation-OH
   binding energy literature.
4. **Engineering-choice fallback (most honest if 1-3 silent).**
   Mark as `engineering_choice = True`; prior centered at
   `k_des_nondim ∈ [10^{-1}, 10^{+1}]` (three decades); bracket
   sweep at `{0.1, 1.0, 10.0}`; explicit "data-constrained in
   Phase D" note.

**Decision rule (commit before research):**
* IF strategy 1 gives a defensible `k_des` (Singh `k_hyd`
  reported + pKa table available) → **lock the derived value**.
* ELSE IF strategy 2 gives a `k_des` order of magnitude with
  documented uncertainty (1–2 decades) → **lock at the central
  value** + bracket sweep.
* ELSE IF strategy 3 gives a defensible Eyring estimate → **lock**
  + bracket sweep.
* ELSE → **fall back to strategy 4**.  Do **not** fabricate a
  citation.  Document the search trail.

**Sensitivity bracket (D7 extension; required if engineering
choice):** `k_des ∈ {0.1, 1.0, 10.0}` nondim — three decades
centered on the smoke value.

**Risk flag, copied from handoff §7:** the A2 finding (step 6) —
K⁺ Boltzmann pile-up at V_kin makes `c_K_boundary_avg ≈ 291 ·
c_K_bulk` — means sentinel-scale R_inj perturbations cannot dent
boundary c_K by 5%.  v10b must **NOT** use boundary-c_K
perturbation at V_kin as a k_des calibration diagnostic.  Use
cap-saturated θ instead (e.g. `θ(k_hyd = 1e-1)` is sensitive to
Γ_max and k_des via the closed-form Γ_ss).

---

## 4. Phase breakdown (with explicit dependencies)

Total estimate: **8–12 working days**.  Parallelizable where
indicated.

### Phase v10b.A — Literature pass (parallelizable; ~3 days)

**Goal:** decide each of the three parameters' D1 outcome (literature
anchor or engineering choice).  Produces draft writeup.

**Sub-phases (parallel):**
* **A1. Γ_max literature pass.**  Read Singh 2016 SI, Iamprasertkun
  2019, Bohra 2019 (after pulling), `Parameters_Seitz_Mangan.xlsx`,
  Yash modeling code.  Apply §3.2 decision rule.  Deliverable:
  draft §3 of `docs/phase6/v10b_calibration_summary.md`.
* **A2. k_des literature pass.**  Strategies 1→2→3, then fall back
  to 4 if needed.  Apply §3.3 decision rule.  Deliverable: draft
  §4 of the writeup.
* **A3. C_S follow-up.**  Pull Bohra 2019 EES into `Articles/`;
  re-derive Risk #5 σ_S mismatch with Stern-only 20 µF/cm²;
  document Yash convention disposition.  Deliverable: draft §5
  of the writeup, plus optional update to step 7's open-asks
  list in `CMK3_capacitance_literature.md`.

**A1 + A2 + A3 can run in parallel as three Agent calls.**  Each
agent gets its decision rule + targets + the writeup section
template (mirror of `CMK3_capacitance_literature.md`).

**Gate at end of A:** consolidated writeup draft + numeric values
committed for D3 + D4.  IF k_des lands on engineering choice,
proceed; do not stall on infinite literature search.

### Phase v10b.B — Code change + unit-test regression (~1 day, serial)

**Depends on:** A1, A2 final values.

**Steps (all in one PR or one set of commits):**
1. Update `Forward/bv_solver/cation_hydrolysis.py:232-242` — rename
   `GAMMA_MAX_HAT_SMOKE` to `GAMMA_MAX_HAT_V10B`; update value;
   rewrite the docstring with the v10b citation.  Keep
   `GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10B` alias for one cycle.
2. Update `scripts/_bv_common.py:944` — same rename + value.
   Sanity test at `tests/test_phase6b_v10a_langmuir_cap.py:46-58`
   should still pass (the test asserts the two values agree).
3. Update `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py:174-182`
   — rename `SMOKE_KINETICS` → `V10B_KINETICS`; update
   `k_des_nondim` + `gamma_max_nondim`; keep `SMOKE_KINETICS =
   V10B_KINETICS` alias for one cycle.
4. Refactor test literals in
   `tests/test_phase6b_v10a_langmuir_cap.py:90,123,132,147` to
   reference `GAMMA_MAX_HAT_V10B` instead of the literal `0.047`.
   Tests at `tests/test_phase6b_v10a_phase_A2_driver.py` and
   `tests/test_phase6b_step6_plumbing_ablation*.py` audited for
   similar literals.
5. Run `pytest -m "not slow" -k "phase6b or cation" -s -vv` —
   green at the new constants.  D8 gate.
6. Audit `scripts/_bv_common.py` and `_v_sweep_diagnostic.py`
   docstrings for any other smoke-value references; update for
   consistency.

**Out of scope for B:**
* The 12 legacy script `stern_capacitance_f_m2=0.10` call-sites
  audit (§3.1 step 2) — that's a separate concurrent task
  classified per-script; doesn't block A.2 regression because the
  v10a-prime drivers already use `STERN_F_M2_BASELINE=0.20`.  Land
  in Phase v10b.E as part of the "writeup + cleanup" bundle.

### Phase v10b.C — A.2 + step 6 regression (~1 day wall, serial)

**Depends on:** Phase B complete.

**Steps:**
1. Re-run `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` with
   default flags (V_kin = −0.10, k0_r4e_factor = 1e-14) and
   `--out-subdir phase6b_v10b_phase_A2_v_kin`.  Wall: ~22 min per
   the v10a' record.  Output:
   `StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.{json,png}`.
2. **Inspect convergence audit.**  If `transition_grid` θ-threshold
   (0.93) false-flags overall_pass = False due to threshold-
   narrowness artifact (per A.2 outcome memory note), document it
   but do not change the threshold without a separate critique
   round.
3. **Qualitative-shape check (D5):**
   * 10/10 rungs converge at λ=1.0.
   * Picard converges everywhere.
   * Mass-balance residual < 5e-3 across the grid.
   * `θ(k_hyd=1e-1)` ≈ 0.998 ± 0.05 (cap saturation preserved).
   * `cd_mA_cm² ≈ −3.12` ± 20% at λ=1.
   * `x_2e ≈ 0.20` ± 20%.
   * `k_hyd_route` exists (highest k_hyd with θ>0.95 etc.).
   If any of these regress > 20% of the v10a' record, ESCALATE
   (likely step 4 re-trigger; see §2).
4. Re-run `scripts/studies/phase6b_step6_plumbing_ablation.py`
   with `--out-subdir phase6b_v10b_step6_plumbing_ablation`.  All
   5 ablations pass.  A0 byte-equivalent (rel ≤ 1e-6 in cd,
   R_net, σ_S, θ) to the new v10b A.2 baseline at λ=1, k_hyd=1e-3,
   V=V_kin.
5. Commit both regenerated JSONs + PNGs.  D5 + D6 gates close.

### Phase v10b.D — Sensitivity bracket sweep (~2 days wall, parallelizable)

**Depends on:** Phase B complete.  Can run **in parallel with Phase C**
since it uses the same constants but a different driver.

**D1. C_S sensitivity bracket (always required).**
* New driver `scripts/studies/phase6b_v10b_cs_bracket.py`:
  * Single V (V_kin = −0.10), λ=1, k_hyd_baseline = 1e-3.
  * Loop C_S ∈ {0.05, 0.10, 0.20, 0.30}.
  * Two-stage anchor pattern: build at STERN_F_M2_ANCHOR = 0.10,
    bump to target C_S via `set_stern_capacitance_model`.
  * Per-rung: report convergence (Newton OK), σ_S, R_net, R_2e,
    R_4e, cd, θ, mass-balance residual.  Plus C_S = 0.10 rung as
    a "shape baseline" (the legacy value).
  * Pass criterion: 4/4 rungs converge; no sign flip on R_net,
    R_2e, R_4e between adjacent rungs; σ_S monotonic in C_S; θ
    smooth.
* Output:
  `StudyResults/phase6b_v10b_cs_bracket/cs_bracket.{json,png}`.
* Wall: ~4 × 5 min = 20 min plus warm-walk overhead.

**D2. Γ_max sensitivity bracket (conditional, see §3.2).**
* Skip if the literature search locks a single value.
* Run if engineering choice or if "keep 0.047 with bracket
  evidence."
* Loop `Γ_max ∈ {0.02, 0.047, 0.1, 0.2}` nondim at V_kin, λ=1,
  k_hyd_baseline.  Anchor at C_S=0.10 + bump to 0.20 (the
  Phase A.2 default).
* Same convergence + smoothness criteria as D1.

**D3. k_des sensitivity bracket (conditional but likely required).**
* Required if §3.3 decision rule lands on engineering choice.
* Loop `k_des ∈ {0.1, 1.0, 10.0}` nondim.  Anchor at C_S=0.10 →
  0.20.
* Critical: use `θ(k_hyd=1e-1)` (cap-saturated rung) as the
  diagnostic, NOT `c_K_boundary_avg` perturbation (per A2
  finding; §3.3 risk flag).

**D7 gate closes when D1 passes; D2 and D3 close conditionally.**

### Phase v10b.E — Writeup + acceptance-bundle update (~1 day)

**Depends on:** A, B, C, D all complete.

**Steps:**
1. Finalize `docs/phase6/v10b_calibration_summary.md`.  Structure
   mirrors `CMK3_capacitance_literature.md`:
   * §1 — Citation chain per parameter.
   * §2 — Per-parameter regimes / decision rules / engineering-
     choice flags.
   * §3 — Caveats per parameter (analog of CMK3 §3a/3b/3c/3d).
   * §4 — Sensitivity brackets + bracket-sweep numeric results.
   * §5 — Implementation status (constants + alias schedule).
   * §6 — Open asks (Singh `k_hyd` if not found; Bohra 2019
     hand-off if relevant; Phase D `k_des` data fit; Yash
     convention).
   * §7 — Cross-references (CLAUDE.md hard rules, acceptance
     bundle § Status, memory entry, MEMORY.md index entry).
2. Append v10b paragraph to acceptance bundle § Status (D9).
3. Update `CLAUDE.md` "Recent progress" line (D10).  Length
   budget: ≤ 200 lines.  Current is 199; consolidate the v10a' /
   Phase A.2 / step 6 narrative paragraphs into 1–2 lines each;
   v10b gets a similar 1–2 line entry that points to the writeup.
4. Audit the 12 legacy `stern_capacitance_f_m2=0.10` call-sites
   per §3.1 step 2 (deck-aligned multi-ion → 0.20; ClO₄⁻ legacy
   → keep 0.10).  Land as part of this commit.
5. Write `project_v10b_calibration_outcome.md` memory entry
   (D11).  Pointer in `MEMORY.md`.

---

## 5. Risk + mitigation register

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | `k_des` literature pass yields nothing | HIGH | LOW (fallback exists) | Engineering-choice flag per §3.3 strategy 4; bracket sweep in D3 closes evidentially. |
| R2 | New `Γ_max` shifts σ_S manifold enough that V_kin or `K0_R4e_factor` no longer satisfy the locked rule | LOW | HIGH (step 4/5/6 re-trigger) | Detect in D5 qualitative-shape check; if regression > 20%, ESCALATE to v10c, do not push past. |
| R3 | `C_S = 0.20` sensitivity sweep fails to converge at one rung (likely 0.30 or 0.05) | MEDIUM | MEDIUM | Two-stage anchor pattern (anchor at 0.10 always; bump to target).  Document rung-specific failure; if 4/4 unachievable, fall back to 3/4 with note. |
| R4 | Convergence-audit threshold-narrowness artifact (overall_pass=False at θ_max=0.9253) re-appears in v10b A.2 baseline | HIGH | LOW (known issue) | Document in writeup; do not change `transition_grid_threshold` without a separate critique cycle. |
| R5 | `tests/test_phase6b_v10a_langmuir_cap.py` literal-`0.047` references break when constant changes | CERTAIN | LOW | Phase B step 4 explicitly refactors literals → constant references. |
| R6 | Legacy ClO₄⁻ script `stern_capacitance_f_m2=0.10` byte-equivalence broken by a hasty global update | MEDIUM | MEDIUM | §3.1 step 2 explicitly classifies per-script; do NOT batch-update.  Audit in Phase E. |
| R7 | CLAUDE.md exceeds 200-line budget after v10b "Recent progress" addition | MEDIUM | LOW | Consolidate v10a' / A.2 / step 6 narrative in Phase E step 3; budget verified pre-commit. |
| R8 | Adjoint-tape contamination if any v10b code touches inverse paths | LOW | HIGH | v10b is forward-only; no `scripts/Inference/` modifications.  Verified by file-touched audit at end of E. |
| R9 | Bohra 2019 EES PDF download fails or paper is paywalled | LOW | LOW | Document open ask; do not block v10b on it (step 7 already shipped without it).  Carry to post-v10b. |
| R10 | Step 6 plumbing-ablation A0 rung fails byte-equivalence vs. new A.2 baseline (rel > 1e-6) | LOW | HIGH | A0 byte-equivalence is the step 6 contract.  If broken: indicates a smoke-value-to-driver leak somewhere; debug before declaring D6 pass. |

---

## 6. Out of scope (handoff §10, reaffirmed)

- **Phase D (K-only Δ_β fit)** — separate step 10 after step 9 (B.2).
- **Cs⁺ / Li⁺ / Na⁺ / Rb⁺ extension** — Phase E, step 11.
- **Variable-`ε_S` / Booth-equation refinement** — post-v10b option.
- **Inverse / adjoint work** — paused project-wide.
- **V_kin re-selection / `K0_R4e_factor` retune** — step 4 invariants;
  only retrigger if v10b causes a convergence regression at V_kin (R2).
- **L_Stern parameterization vs `C_S` parameterization choice** —
  carry as open ask from step 7 (Yash convention).
- **Selectivity-gap improvement** as a v10b pass criterion — that's
  Phase D's job (LOW priority per A.2 outcome).

---

## 7. Dependency graph

```
v10b.A (parallel)
├── A1 Γ_max research ──┐
├── A2 k_des research ──┼──► A end gate (writeup draft + values)
└── A3 C_S follow-up ───┘
                         │
                         ▼
                    v10b.B (serial)
                    Code changes + unit tests
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
       v10b.C (serial)          v10b.D (parallel)
       A.2 + step 6              C_S/Γ_max/k_des
       regression                bracket sweeps
            │                         │
            └────────────┬────────────┘
                         ▼
                    v10b.E (serial)
                    Writeup + acceptance bundle
                    + CLAUDE.md + memory entry
```

**Critical path:** A → B → C → E.  D parallel with C.
**Bottlenecks:** A (literature pass — may stall on k_des per R1);
B (single-day serial work, no obvious parallelization);
C (single Phase A.2 + step 6 wall ~30 min each).

---

## 8. Validation checkpoints

1. **End of A:** writeup draft has citation chain for each parameter
   OR explicit engineering-choice flag.  Numeric values committed
   for D3 + D4.  If A2 (k_des) silent, fallback flag set.
2. **End of B:** `pytest -m "not slow" -k "phase6b or cation"`
   green.  D8 gate.
3. **End of C:** D5 + D6 gates close.  Qualitative-shape preserved;
   step 6 A0 byte-equivalent to v10b A.2.  If not, escalate per R2.
4. **End of D:** D7 gate closes (C_S 4/4 + conditional Γ_max,
   k_des).  If any rung fails per R3, document fallback in writeup.
5. **End of E:** D9 + D10 + D11 gates close.  v10b shipped.
   B.2 (step 9) unblocked.

---

## 9. Decision rules summary (for quick reference)

| Parameter | Outcome path |
|---|---|
| `C_S` | Already locked at 0.20 F/m²; v10b runs sweep + carries open asks. |
| `Γ_max` | Try literature anchor → tighten citation chain → engineering choice with bracket. |
| `k_des` | Strategy 1 → 2 → 3 → engineering choice with bracket (probably 4). |

**If in doubt, prefer the engineering-choice flag with documented
prior and bracket sweep evidence over a fabricated citation.**

---

## 10. Open questions for the GPT critique loop

1. Is the alias / one-cycle-deprecation strategy for
   `GAMMA_MAX_HAT_SMOKE` and `SMOKE_KINETICS` correct, or should the
   rename be atomic (break callers in one PR)?
2. Should the C_S sensitivity sweep at {0.05, 0.10, 0.20, 0.30}
   include a fifth high-end excursion (e.g. 0.50) to probe Pillai's
   "safe band" upper edge?
3. Should Γ_max bracket sweep be run unconditionally for evidential
   completeness, even when a literature anchor is locked?
4. Is the k_des engineering-choice prior `[10^{-1}, 10^{+1}]` the
   right range, or should it be tighter (e.g. `[0.5, 5]`) given the
   diffusion-limited upper bound is implicitly ~D_M+/δ_OHP ~ O(1)
   in nondim?
5. Should the legacy ClO₄⁻ script C_S audit (§3.1 step 2) be
   deferred entirely out of v10b (separate cleanup task) to avoid
   blast-radius creep?
6. Is the qualitative-shape ±20% tolerance in D5 the right
   threshold, or should it be tighter for cd (which has more
   significant figures) and looser for x_2e (which is selectivity-
   sensitive)?
7. Should v10b.A spawn **three** parallel research agents or one
   sequential agent with explicit per-parameter sub-tasks?

---

**End of plan v1.**  Entering GPT critique loop.
````````````````````````

---

## Section 3 — Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.

Specific lenses for this artifact:
- Missing decision rules (where the plan defers to the executor
  rather than committing up front).
- Ungrounded literature claims (anything stated as fact without a
  citation or with a hand-wave).
- Hidden coupling between Γ_max, k_des, C_S (e.g. detailed-balance
  identities, denominator structure of Γ_ss).
- Incorrect dependency ordering (e.g. D7 depending on B but not
  noticing it also needs C if a baseline is shared).
- Scope creep risks (anything that should be deferred to v10c, step
  9, Phase D, or post-v10b).
- Test/regression coverage gaps (especially around the alias
  schedule and the byte-equivalence contract).
- Risk register coverage (anything material that's missing).
