# CLAUDE.md — PNPInverse

Short operational guide for Claude when working in this repo. The
`README.md` has the research narrative and the full architecture; this
file is the project-specific conventions, invariants, and lessons that
are easy to get wrong. Read both.

## What this repo is

Research code for Poisson–Nernst–Planck / Butler–Volmer (PNP-BV) forward
simulation and (eventually) inverse kinetic inference for ORR
(O₂ → H₂O₂ → H₂O) on the Seitz/Mangan deck (K₂SO₄ at pH 4–6, parallel
2e⁻/4e⁻ topology per Ruggiero 2022).

The production forward stack (May 2026):

- **3 dynamic species** (O₂, H₂O₂, H⁺) + **analytic Bikerman
  counterion(s)** (`steric_mode='bikerman'`, residual-side closure +
  Bikerman-consistent IC). Reference scripts use ClO₄⁻; deck-aligned
  sweeps use Cs⁺ + SO₄²⁻ (deck-baseline is **K⁺**, not Cs⁺ — see
  `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md`).
- **`formulation='logc_muh'`** with proton electrochemical-potential
  primary variable (`mu_H = u_H + em·z_H·phi`).
- **Log-rate Butler–Volmer** with **parallel R2e + R4e** (E°_R2e =
  0.695 V, E°_R4e = 1.23 V vs RHE; Ruggiero 2022 §1).
- **Finite Stern compact layer** — production target
  `stern_capacitance_f_m2 = 0.20` F/m² (= 20 µF/cm²) per the
  Bohra-Koper-Choi PNP-modeling consensus
  (`.research/cmk3-stern-capacitance/SUMMARY.md`). Derived from
  Stern thickness `L_S = 5 Å` and Booth-saturated
  `ε_S = 11.3`. Citation chain: Bohra et al. 2024 *JPC C*
  (PMC11215773), Choi et al. 2024 (`10.1021/acs.jpcc.4c03469`),
  Pillai et al. 2024 (`10.1021/acs.jpcc.3c05364`,
  the methodological-error critical review that flags the
  100–200 µF/cm² CO2R-modeler convention as wrong), CatINT
  default (Stanford Bell group), Kilic-Bazant 2007 *Phys Rev E*
  75:021503 (foundational mPNP-Stern). The legacy
  `0.10 F/m² (10 µF/cm²)` value (pre-2026-05-10) was a
  convergence-pinned engineering choice with no CMK-3 anchor
  (see `.research/cmk3-stern-capacitance/codebase-cs-provenance.md`);
  it still sits in Pillai's "safe band" (10–50 µF/cm²) and is
  defensible as carbon-conservative, but `0.20 F/m²` is the
  production target going forward.
- **`debye_boltzmann` IC** (composite-ψ + multispecies-γ).

Reaches V_RHE = +1.0 V at 15/15 via C+D (cold ceiling +0.60 V;
warm-walk to +1.00 V).

**Phase 6α landed (2026-05-09):** water self-ionization residual
`E = c_H − c_OH` via `Forward/bv_solver/water_ionization.py`; new
`solve_anchor_with_continuation` + `solve_grid_with_anchor` in
`Forward/bv_solver/anchor_continuation.py`. 8/8 sweep convergence
but P3 surface-pH gate fails at 10.58. See
`docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md`.

**Phase 6β scope (current):** cation hydrolysis at the OHP
(`M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺` with field-dependent pKa per Singh 2016
JACS). **Not sulfate buffering** — retired after grepping the data
folder. See `docs/phase6/phase6b_next_steps_plan.md`.

**Phase 6β v10a landed (2026-05-10):** Langmuir saturation cap
`(1 − Γ/Γ_max)` on the cation-hydrolysis forward branch — the v9
formulation let Γ grow unboundedly (≈ 6+ monolayers at `k_hyd ≥ 1e-3`,
~64 monolayers at `k_hyd = 1e-2`). The v10a closed form
`Γ_ss(λ) = λ·F₀ / ((1−λ) + λ·k_des + λ·B + λ·F₀/Γ_max)` reduces to v9
as `Γ_max → ∞` and saturates at one monolayer otherwise. Default
`gamma_max_nondim = 0.047` is a smoke baseline (1 monolayer of MOH at
the OHP, `5.6e-6 mol/m² / (C_SCALE · L_REF)`); v10b replaces it with
the literature-calibrated value once
`docs/phase6/CMK3_capacitance_literature.md` lands. Wired through
`Forward/bv_solver/cation_hydrolysis.py` + `Forward/bv_solver/units.py`
+ `set_reaction_gamma_max_model` accessor; diagnostic plumbing emits
`F₀`, `Γ`, `θ`, `R_forward_capped`, denominator decomposition, R_2e/R_4e
per-reaction currents, and σ_S in counts/pm² per rung. Regression
suite at `tests/test_phase6b_v10a_langmuir_cap.py`.

**Phase 6β v10a' landed (2026-05-10):** V-sweep diagnostic at the
production target (`C_S = 0.20 F/m²` per Bohra-Koper-Choi consensus,
`K0_R4e_factor = 1e-14` V=−0.10 branch-pass probe per
`project_k0_r4e_ratio_regimes`).  The initial v10a sweep returned
`no_candidate_passed_locked_rule`; v10a' adds two driver-side fixes:
(a) two-stage anchor (`STERN_F_M2_ANCHOR = 0.10` for the k0/Kw_eff
ladder, then runtime bump to `STERN_F_M2_BASELINE = 0.20` via
`set_stern_capacitance_model` + Newton resolve — needed because the
existing solver's `c_s_ladder` raises `NotImplementedError` when
combined with `kw_eff_ladder`); and (b) `--k0-r4e-factor` CLI flag
with deep-copy reaction rescaling.  **Result: V_kin = −0.10 V via
the primary path (no fallback).**  Decision tree → Case A → Phase
A.2 at V_kin = −0.10 V.  Per-V breakdown shows V=−0.30 / V=−0.50
are cap-dominated (denom_cap/total > 0.8 AND θ > 0.9 AND |sensS| <
0.10) — v10b prerequisite signal is present at the deepest cathodic
V's but not at V_kin itself, so v10b routing is not triggered.  K+
enrichment is the dominant F₀ amplifier in the cathodic region
(amp_from_c_K = 0.16 → 11.6 across V_RHE; amp_from_singh ≈ 1.0
everywhere).  New diagnostic emissions: `F0_decomposition`,
`R_4e_decomposition_log`, `denominator_cap_to_total_ratio` (folded
into per-V records).  Output:
`StudyResults/phase6b_v10a_prime_k0r4e_1e-14/iv_diagnostic.{json,png}`.
Wall: 27 min.  Plan + critique provenance:
`~/.claude/plans/sparkly-gilded-pasteur.md`.

**Phase 6β v10a Phase A.2 landed (2026-05-10):** densified k_hyd × λ
ramp at V_kin = −0.10 V (10 points spanning `k_hyd ∈ {1e-5 … 1e-1}`)
with the v10a Langmuir cap + v10a' two-stage anchor + full
diagnostics.  All 10 rungs converge cleanly at λ=1.0; Picard
converged everywhere (no `iter_cap_hit_unconverged` rungs);
**mass-balance residual at machine precision (1e-14 to 1e-16)**
across the grid, confirming the closed-form Γ_ss residual-side
closure is consistent.  θ tracks from 0.058 (k_hyd=1e-5, pre-cap)
through the onset transition (θ=0.86 at k_hyd=1e-3 baseline,
reproducing v10a' record within rel 1e-3) to full saturation
(θ=0.998 at k_hyd=1e-1; v9 Phase B failed Picard at this point
without a cap).  **k_hyd_route = 1e-1** (highest k_hyd satisfying
θ>0.95, slope<0.05, picard_ok, mass_balance_ok, transport_ok).
**v10b routing: LOW priority for k_des/Γ_max calibration**
(`single_v_selectivity_gap_pp = +5.09 pp` — H₂O₂% = 19.91% sits
5pp below the deck band [25, 50]%, within the 10pp routing
cutoff).  **rH_El recalibration NOT required**
(`max_amp_from_singh = 1.0000112`; Singh ΔpKa contribution is
negligible at V_kin).  No transport re-entry across the k_hyd
grid (`o2_flux_levich_ratio ≈ 0.63` everywhere).  σ_S, cd_mA_cm²,
x_2e are all k_hyd-independent at this V — confirming cation
hydrolysis is decoupled from the parallel-2e/4e branch split and
from the Stern-cap manifold at V_kin (Γ affects the proton
boundary source but not the field-driven Bikerman packing).
Convergence audit `overall_pass = False` is a threshold-narrowness
artifact: `max(θ_λ=1) = 0.9253` in the plan-defined transition
grid {1e-5 … 2e-3} just below the 0.93 cutoff, but the next
k_hyd up (5e-3, in the saturation grid by construction) hits
θ=0.969 — the audit's transition_grid threshold was set 0.005
above the closed-form prediction without margin.  Output:
`StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.{json,png}`.
Tests: `tests/test_phase6b_v10a_phase_A2_driver.py` (43 fast).
Wall: 22 min.  Plan + critique provenance:
`~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md` (4 rounds of GPT
critique, APPROVED).  Next: step 6 plumbing ablation matrix at
V_kin or step 8 v10b literature calibration of Γ_max + k_des + C_S
(v10b is MANDATORY in all routing branches; A.2 informs priority,
never cancels v10b).

## Inverse status: paused

All inverse scripts (`scripts/studies/v*.py`, `scripts/Inference/`) are
non-operational. When work resumes, start from
`docs/inverse/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md` +
`docs/inverse/noise_model_conventions.md`.

## Source-of-truth docs (read before opining on status)

| File | Purpose |
|---|---|
| `data/EChem Reactor Modeling-Seitz-Mangan/` | **PRIMARY EXPERIMENTAL DATA DROP** from the Seitz/Mangan group. Read this BEFORE conjecturing about deck physics. Per-file inventory in `docs/papers/data_folder_code_inventory.md`; deep audit in `docs/papers/seitz_mangan_data_folder_audit_2026-05-08.md` |
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Forward-solver rebuild narrative |
| `writeups/ForwardSolverChangesMay26/forward_solver_changes_may2026.pdf` | May 2026 production-target writeup |
| `docs/solver/bv_solver_unified_api.md` | How to call the production stack |
| `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md` | Production-target reference sweep (15/15 V_RHE [−0.5, +1.0]) |
| `docs/solver/steric_analytic_clo4_reduction_handoff.md` | Derivation of the Bikerman analytic-counterion residual closure |
| `docs/solver/clipping_conventions.md` | The three distinct BV-related clips and the `exponent_clip` 50 → 100 raise |
| `docs/realignment/Mangan2025_experimental_alignment.md` | Gap audit between the model and the Mangan 2025 deck |
| `docs/papers/Ruggiero2022_JCatal_source_paper.md` | Peer-reviewed source paper underlying the Mangan deck. Load-bearing: K₂SO₄ not ClO₄⁻, parallel 2e⁻ (0.695 V) + 4e⁻ (1.23 V) ORR not sequential R₀/R₁, N=0.224, 1600 rpm, I=0.3 M. PDF at `docs/papers/Ruggiero2022_JCatal_manuscript.pdf` |
| `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md` | State of Phase 6α (2026-05-09): water-ionization landed, 8/8 convergence, P3 surface-pH gate fails at 10.58, mechanism correction (cation hydrolysis, not water, not sulfate). Expanded planning in `docs/handoffs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md` §9 |
| `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` | Audit of `fast-realignment-2026-05-08` for Claude/GPT-conjecture vs. grounded changes. HIGH-risk: Cs⁺ vs deck-baseline K⁺. Read before scoping new physics or kinetics calibration |
| `docs/phase6/phase6b_next_steps_plan.md` | Current Phase 6β plan (cation hydrolysis at OHP). Survived 5-round GPT critique; two architectural items unresolved |
| `docs/solver/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B for the logc+counterion stack |
| `docs/solver/forward_solver_test_coverage.md` | What the bv_solver test suite covers |

## Environment

- **Activate `../venv-firedrake/bin/activate` from `PNPInverse/`.**
  Conda envs (e.g. `FireDrakeEnv`) will not run Firedrake correctly.
- Useful cache settings: `MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1`.
- Tests: `pytest -m "not slow"` for fast unit tests; `pytest -m slow`
  requires Firedrake.
- **Always stream test output** (`pytest -s -vv`, `--log-cli-level=INFO`,
  `python -u`, no buffering wrappers). Tests can stall for minutes
  inside a single Firedrake solve.

## Hard rules (lessons that cost real time)

1. **Use C+D, not Strategy B**: call
   `Forward.bv_solver.solve_grid_per_voltage_cold_with_warm_fallback`,
   not `solve_grid_with_charge_continuation`. B's Phase-1 V-sweep at
   z=0 hands bisection a mismatched species IC it can't recover from
   on the logc+counterion stack (3/13 at production resolution). For
   Phase 6α/β work prefer the newer
   `solve_anchor_with_continuation` + `solve_grid_with_anchor` pair
   in `Forward/bv_solver/anchor_continuation.py`.

2. **`exponent_clip = 100` is the only PC-trustworthy setting.** The
   clip is on `eta_scaled = (V_RHE − E_eq)/V_T` *before* the α·n_e
   multiplication (`forms_logc.py:_build_eta_clipped`). At clip=100,
   R2 unclips at V_RHE > −0.79 V (production grid fully unclipped).
   At clip=50 (historical), clipped R2 produces fictitious peroxide
   current — don't trust PC there. `u_clamp = 100` for the same
   reason. Full breakdown in `docs/solver/clipping_conventions.md`.

3. **The IC and the residual must agree about steric saturation.**
   `set_initial_conditions_debye_boltzmann_*` seeds composite-ψ +
   multispecies-γ; the residual side picks up
   `build_steric_boltzmann_expressions` so the dynamic-species
   packing fraction *and* the Poisson source agree on the saturated
   counterion concentration. A bikerman IC without the matching
   residual (or vice-versa) cold-fails on the saturated manifold.

4. **Use the Ruggiero parallel-topology E_eq values**, not the legacy
   sequential ones. The original ORR formulation in this repo (R1 =
   0.68 V, R2 = 1.78 V, sequential R₀→R₁) was **wrong** — it didn't
   match the deck's actual parallel 2e⁻/4e⁻ structure (Ruggiero 2022
   §1). The current model uses **R2e (E° = 0.695 V vs RHE)** and
   **R4e (E° = 1.23 V vs RHE)** in parallel, configured in
   `scripts/_bv_common.py:660-690`. Solver convergence window is
   V_RHE ∈ [−0.5, +1.0] V via C+D (cold ceiling +0.60 V; warm-walk
   to +1.00 V — see `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md`).
   Never use `E_eq = 0`.

5. **Check `data/EChem Reactor Modeling-Seitz-Mangan/` BEFORE
   conjecturing about the electrochemistry.** Whenever you're about
   to claim *what's happening physically* in the cell — buffer
   mechanism, local pH driver, rate law, cation effect, transport
   bottleneck, missing physics — first grep the group's documents
   and the cited literature in that folder. Per-file inventory in
   `docs/papers/data_folder_code_inventory.md`.

   The group has a documented hypothesis for buffering chemistry:
   **cation hydrolysis at the polarized OHP** (`M(H₂O)ₙ⁺ ⇌ M(OH)⁰ +
   H⁺` with field-dependent pKa; Cs⁺ ≈ 4.3 near Cu per Linsey deck
   slide 27). It is **not** sulfate buffering, **not** carbonate
   buffering, **not** water self-ionization. Don't reinvent the
   chemistry from textbook intuition without checking — the
   2026-05-09 Phase 6α handoff briefly pointed Phase 6β at HSO₄⁻/SO₄²⁻
   buffering before this folder was checked, and that was wrong. See
   `docs/handoffs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md`
   §9 for the corrected scope. Reading order for Phase 6β background:
   Singh 2016 JACS (`10.1021/jacs.6b07612`, functional form for
   field-dependent pKa) → Co-Zhang 2019 Angewandte (cation-pH
   experimental support; product/ring electrochemistry, *not* IrOx)
   → Linsey 2025 ACS-CATL deck slides 5–9 + 27.

6. **`stern_capacitance_f_m2 = 0.20` F/m² is the literature-anchored
   production target.** Derived from Stern thickness `L_S = 5 Å` and
   Booth-saturated permittivity `ε_S = 11.3`, giving
   `C_S = ε_S·ε₀/L_S = 20 µF/cm²` per the Bohra-Koper-Choi
   PNP-modeling consensus (Bohra 2024, Choi 2024, Pillai 2024,
   CatINT default, Kilic-Bazant 2007 foundational).
   **Three caveats that matter:**
   (a) The legacy `0.10 F/m²` (pre-2026-05-10) is at the *low* edge
       of Pillai 2024's "safe band" (10–50 µF/cm²) — defensible as
       carbon-conservative but not citation-anchored; existing
       StudyResults at `0.10 F/m²` remain interpretable as a
       sensitivity-bracket low rung.
   (b) **C_S is per-local-surface-element**, not per-geometric-
       electrode-area. The 1D-slab PNP-Stern formulation
       (Bazant/Kilic/Storey/Jithin lineage) has no roughness factor
       in the BC. The deck's CMK-3 RF ≈ 6000 (0.5 mg/cm² × 1200
       m²/g BET) is *implicit in fitted k₀*. Cross-stack comparison
       to Yash (uses `L_Stern` thickness directly, not C_S) or
       Bohra 2019 (variable ε via Booth) requires explicit RF
       accounting on the kinetic side.
   (c) The earlier "σ_S mismatch vs Singh 51 µF/cm²" concern
       (`PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` Risk #5)
       was a misapplication: Singh's 51 µF/cm² Cu was an in-house
       CV-slope total C_dl measurement (Stern + diffuse + roughness
       + specific adsorption), not a Stern-only quantity. At
       `C_S = 0.20 F/m²`, the σ_S mismatch may not even be the
       right comparison anymore — re-derive with Stern-only values
       (Bohra/Koper 20–25 µF/cm² for metal; ~10–20 µF/cm² for sp²
       carbon) before treating Risk #5 as load-bearing.
   Sensitivity bracket for Gate 4B / v10b:
   `C_S ∈ {0.05, 0.10, 0.20, 0.30}` F/m². Full literature trail in
   `.research/cmk3-stern-capacitance/SUMMARY.md`.

## Calling the production solver

Use the canonical factory + dispatcher; don't reinvent flag wiring.
**Don't** add the inline `add_boltzmann(ctx)` helper while *also*
setting `bv_bc.boltzmann_counterions` — that double-counts the
counterion. **Don't** pass `bv_reactions=...` while also passing
`k0_hat_r1`/`k0_hat_r2`/etc. — the reactions list takes precedence
and the legacy keyword bundle gets silently ignored.

### Multi-ion deck-aligned stack (current production target)

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    PARALLEL_2E_4E_REACTIONS,                    # Ruggiero §1 parallel topology
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,  # ⚠ deck baseline is K⁺ — see audit
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
)
from Forward.bv_solver.anchor_continuation import (
    solve_anchor_with_continuation,
    extract_preconverged_anchor,
)
from Forward.bv_solver import solve_grid_with_anchor, make_graded_rectangle_mesh

sp = make_bv_solver_params(
    eta_hat=0.0, dt=0.25, t_end=80.0,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    bv_reactions=PARALLEL_2E_4E_REACTIONS,        # parallel 2e/4e
    boltzmann_counterions=[                       # multi-ion shared-θ closure
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    ],
    multi_ion_enabled=True,                       # required for ≥2 bikerman entries
    stern_capacitance_f_m2=0.20,                  # Bohra/Choi/Pillai 2024 anchor — see hard rule #6
    initializer="debye_boltzmann",
    l_eff_m=100e-6,                               # boundary-layer thickness
    enable_water_ionization=False,                # Phase 6α opt-in (default off)
)

mesh = make_graded_rectangle_mesh(
    Nx=8, Ny=80, beta=3.0,
    domain_height_hat=sp.solver_options["bv_convergence"]["domain_height_hat"],
)
```

Then anchor + grid (the **preferred** path for the multi-ion +
Stern stack — C+D's Phase-1 cold-start fails 13/13 around V ≈ +0.55 V
on this stack):

```python
anchor_result = solve_anchor_with_continuation(
    sp.with_phi_applied(0.55 / V_T), mesh=mesh,
    k0_targets={0: K0_HAT_R2E, 1: K0_HAT_R4E},
    initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
    # Phase 6α only: kw_eff_ladder=(0.0, KW_HAT*1e-6, KW_HAT*1e-3, KW_HAT*0.1, KW_HAT)
)
anchor = extract_preconverged_anchor(
    anchor_result, phi_applied_eta=0.55/V_T,
    k0_targets={0: K0_HAT_R2E, 1: K0_HAT_R4E},
    mesh_dof_count=anchor_result.ctx["U"].function_space().dim(),
)
grid = solve_grid_with_anchor(sp, mesh=mesh, anchor=anchor, v_rhe_grid=V_RHE_GRID)
```

### Legacy single-counterion stack (ClO₄⁻ reference)

For backward-compat / equivalence-checking against historical results,
use a single ClO₄⁻ counterion + the C+D dispatcher:

```python
from scripts._bv_common import DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC
from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback

sp = make_bv_solver_params(
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,                  # legacy ClO₄⁻ runs keep 0.10
                                                  # for byte-equivalence with
                                                  # historical StudyResults;
                                                  # use 0.20 for deck-aligned
                                                  # multi-ion stack (hard rule #6)
    initializer="debye_boltzmann",
    # parallel-2e/4e omitted → defaults to legacy sequential R1/R2
)
```

### Reference drivers

| Driver | Stack | Use when |
|---|---|---|
| `scripts/studies/l_eff_transport_sweep_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e + `--enable-water-ionization` flag | **Most recent**; Phase 6α validation, L_eff sweep |
| `scripts/studies/mangan_full_grid_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e | Deck page-15 V_RHE band, single L_eff |
| `scripts/studies/peroxide_window_3sp_bikerman_muh.py` | ClO₄⁻ single-counterion + sequential R1/R2 | Legacy ClO₄⁻ reference |

Full API in `docs/solver/bv_solver_unified_api.md`.

### Gotchas

- **K⁺ vs Cs⁺**: `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` is one of
  the four cations the deck investigates, but the deck *baseline* is
  K⁺/SO₄²⁻ (Linsey 2025 deck slide 9: `[SO₄²⁻]=0.1 M & [K⁺]=0.2 M`).
  Apples-to-apples deck comparisons should swap to a K⁺ entry. See
  `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md`.
- **Phase 6α opt-in**: `enable_water_ionization=True` plus the
  `kw_eff_ladder` outer loop on `solve_anchor_with_continuation` —
  see `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md`. Default-off path is
  byte-equivalent to pre-Phase-6α.
- **`multi_ion_enabled=True` is required** when passing ≥2 bikerman
  counterions (the factory raises if not set; the multi-ion shared-θ
  closure has different math than the single-counterion 1:1 closure).
- **`debye_boltzmann` IC requires** either a `synthesised_4sp` ClO₄⁻
  counterion *or* a `steric_mode='bikerman'` entry; with
  `steric_mode='ideal'` it falls back to tanh-Gouy-Chapman.
- **`validate_solution_state` on muh** needs `is_logc=...`,
  `mu_species=ctx.get('mu_species')`, and
  `em=ctx['nondim'].get('electromigration_prefactor', 1.0)`.
- **`H2O2_SEED_NONDIM = 1e-4`** is a finite-seed for `ln c_H2O2` at
  bulk Dirichlet, not a physics tweak.
- **`set_initial_conditions(ctx, sp, blob=True)`** is silently ignored
  in log-c mode (no blob IC for `u_i = ln c_i`).
- **`l_eff_m` is read at form-build time** via
  `bv_convergence['domain_height_hat']`; the mesh y-extent must
  match (`make_graded_rectangle_mesh(domain_height_hat=...)`) or
  the IC and residual disagree on the bulk anchor location.

## Path conventions

Run scripts from `PNPInverse/`, not from `Forward/` or `scripts/`.
`scripts/Inference/` is **uppercase**. `scripts/studies/v*.py` are
legacy inverse scripts (not operational). `StudyResults/` is part of
the working research record — check existing `summary.md` files before
regenerating expensive studies. `archive/` is reference-only.

## Workflow notes

- **Plan before non-trivial forward-solver changes.** Live backends:
  `forms_logc.py` and `forms_logc_muh.py` (concentration backend
  removed May 2026). Decide which validation scripts to run
  (`scripts/verification/` MMS, `peroxide_window_3sp_bikerman_muh.py`)
  before implementing.
- **Long-running studies cost minutes-to-hours.** Confirm before
  regenerating expensive runs.
- **Adjoint tape hygiene** (when inverse resumes): wrap unannotated
  cold-ramp / continuation work in `with adj.stop_annotating():`.
