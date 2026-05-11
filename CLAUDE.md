# CLAUDE.md — PNPInverse

Operational guide for Claude. `README.md` has the research
narrative; this file is the conventions, invariants, and
lessons-that-cost-time. Read both.

## What this repo is

Research code for Poisson–Nernst–Planck / Butler–Volmer (PNP-BV)
forward simulation and (eventually) inverse kinetic inference for
ORR (O₂ → H₂O₂ → H₂O) on the Seitz/Mangan deck (K₂SO₄ at pH 4–6,
parallel 2e⁻/4e⁻ per Ruggiero 2022).

Production forward stack (May 2026):

- **3 dynamic species** (O₂, H₂O₂, H⁺) + **analytic Bikerman
  counterion(s)**. Deck baseline is **K⁺/SO₄²⁻** (not Cs⁺ — see
  `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md`).
- **`formulation='logc_muh'`** — proton electrochemical-potential
  primary variable (`mu_H = u_H + em·z_H·phi`).
- **Log-rate Butler–Volmer**, parallel **R2e (E°=0.695 V)** + **R4e
  (E°=1.23 V vs RHE)** per Ruggiero 2022 §1.
- **Stern capacitance `C_S = 0.20 F/m²`** (Bohra-Koper-Choi
  consensus; canonical reference
  `docs/phase6/CMK3_capacitance_literature.md`).
- **`debye_boltzmann` IC** (composite-ψ + multispecies-γ).
- Reaches V_RHE = +1.0 V at 15/15 via C+D.

## Recent progress (Phase 6α/6β)

Latest landed: **v10b literature calibration (2026-05-10)** —
`GAMMA_MAX_HAT_V10B = 0.047` (tightened V10A chain; 4-test
compatibility check found no peer-reviewed MOH-coverage anchor),
`K_DES_NONDIM_V10B = 1.0` (engineering choice; Eyring prior
`[1e-2, 1e2]` ↔ `ΔG_des ∈ [0.69, 0.94] eV`), `C_S_F_M2_V10B = 0.20`
(locked at step 7). New top-level Firedrake-free
`calibration/v10b.py`. Deprecation alias `SMOKE = V10A_SMOKE`
(NEVER `SMOKE = V10B`). Writeup:
`docs/phase6/v10b_calibration_summary.md`. Full Phase 6α/6β
chronology (v9 A/B → v10a → v10a' → Phase A.2 → step 6 → v10b) in
`docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md § Status`.
Current scope: **cation hydrolysis at polarized OHP** (Singh 2016
field-dependent pKa). Live plan:
`docs/phase6/phase6b_next_steps_plan.md`. Per-investigation
findings: `~/.claude/projects/.../memory/MEMORY.md`.

## Inverse status: paused

Inverse scripts (`scripts/studies/v*.py`, `scripts/Inference/`) are
non-operational. When resumed, start from
`docs/inverse/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`.

## Source-of-truth docs (read before opining on status)

| File | Purpose |
|---|---|
| `data/EChem Reactor Modeling-Seitz-Mangan/` | **PRIMARY EXPERIMENTAL DATA** from Seitz/Mangan. Inventory: `docs/papers/data_folder_code_inventory.md` |
| `docs/solver/bv_solver_unified_api.md` | How to call the production stack |
| `docs/solver/clipping_conventions.md` | The three BV clips + `exponent_clip` 50 → 100 raise |
| `docs/solver/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B for logc+counterion |
| `docs/papers/Ruggiero2022_JCatal_source_paper.md` | Source paper underlying Mangan deck (K₂SO₄, parallel 2e/4e, N=0.224, 1600 rpm, I=0.3 M) |
| `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` | Acceptance bundle + § Status with full Phase 6 chronology |
| `docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md` | Phase 6α water-ionization outcome + Phase 6β scoping handoff |
| `docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` | Cs⁺ vs deck-baseline K⁺ audit. Read before new physics or kinetics calibration |
| `docs/phase6/phase6b_next_steps_plan.md` | Live Phase 6β plan |
| `docs/phase6/singh_2016_pka_formula.md` | Singh 2016 SI Eq. (3)/(4) + σ-mapping |
| `docs/phase6/CMK3_capacitance_literature.md` | `C_S = 0.20 F/m²` citation chain |
| `docs/phase6/v10b_calibration_summary.md` | v10b Γ_max + k_des + C_S calibration |

## Environment

- **Activate `../venv-firedrake/bin/activate` from `PNPInverse/`.**
  Conda envs (e.g. `FireDrakeEnv`) will NOT run Firedrake correctly.
- Cache env: `MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1`.
- Tests: `pytest -m "not slow"` for fast; `pytest -m slow` needs
  Firedrake. **Always stream** (`-s -vv --log-cli-level=INFO`,
  `python -u`); tests can stall minutes inside a single solve.

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
   current. `u_clamp = 100` for the same reason. Details in
   `docs/solver/clipping_conventions.md`.

3. **The IC and the residual must agree about steric saturation.**
   `set_initial_conditions_debye_boltzmann_*` seeds composite-ψ +
   multispecies-γ; the residual side picks up
   `build_steric_boltzmann_expressions`. A bikerman IC without the
   matching residual (or vice-versa) cold-fails on the saturated
   manifold.

4. **Use Ruggiero parallel-topology E_eq values**: **R2e (E°=0.695 V)**
   + **R4e (E°=1.23 V)** in parallel (configured in
   `scripts/_bv_common.py:660-690`). The legacy sequential R₀→R₁
   (R1=0.68, R2=1.78) was **wrong**. Never use `E_eq = 0`.
   Convergence window V_RHE ∈ [−0.5, +1.0] V via C+D.

5. **Check `data/EChem Reactor Modeling-Seitz-Mangan/` BEFORE
   conjecturing about the electrochemistry.** Group's documented
   hypothesis is **cation hydrolysis at the polarized OHP** (Singh
   2016 field-dependent pKa) — **not** sulfate, carbonate, or
   water self-ionization. Reading order + corrected scope in
   `docs/handoffs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md` §9.

6. **`stern_capacitance_f_m2 = 0.20` F/m² is the literature-anchored
   production target** (Bohra-Koper-Choi consensus). `C_S` is
   **per-local-surface-element**, not per-geometric area (CMK-3
   RF ≈ 6000 is implicit in fitted k₀). Sensitivity bracket for
   v10b: `C_S ∈ {0.05, 0.10, 0.20, 0.30}` F/m². Citation chain +
   caveats: `docs/phase6/CMK3_capacitance_literature.md`.

## Calling the production solver

Use canonical factory + dispatcher; don't reinvent flag wiring.
**Don't** add inline `add_boltzmann(ctx)` while setting
`bv_bc.boltzmann_counterions` (double-counts). **Don't** pass
`bv_reactions=...` with `k0_hat_r1`/`k0_hat_r2`/etc. (reactions list
takes precedence; legacy bundle silently ignored).

Multi-ion deck-aligned stack (current production target):

```python
sp = make_bv_solver_params(
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh", log_rate=True,
    bv_reactions=PARALLEL_2E_4E_REACTIONS,
    boltzmann_counterions=[
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,    # ⚠ deck baseline is K⁺
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    ],
    multi_ion_enabled=True,
    stern_capacitance_f_m2=0.20,                       # Hard rule #6
    initializer="debye_boltzmann",
    l_eff_m=100e-6,
    enable_water_ionization=False,                     # Phase 6α opt-in
)
```

Then **anchor + grid** (preferred for multi-ion + Stern — C+D's
Phase-1 cold-start fails 13/13 around V ≈ +0.55 V):
`solve_anchor_with_continuation` → `extract_preconverged_anchor` →
`solve_grid_with_anchor`. Full API + Phase 6α/β extensions
(`kw_eff_ladder`, `lambda_hydrolysis`, override flags, etc.) in
`docs/solver/bv_solver_unified_api.md`.

### Reference drivers

| Driver | Stack | Use when |
|---|---|---|
| `scripts/studies/phase6b_step6_plumbing_ablation.py` | K⁺/SO₄²⁻ + cation hydrolysis + step 6 ablation flags | Plumbing verification at V_kin |
| `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` | K⁺/SO₄²⁻ + v10a Langmuir cap, k_hyd ramp | Phase A.2 baseline reproduction |
| `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` | K⁺/SO₄²⁻ + v10a' V-sweep | V_kin selection diagnostic |
| `scripts/studies/l_eff_transport_sweep_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion + parallel 2e/4e + `--enable-water-ionization` | Phase 6α validation |
| `scripts/studies/mangan_full_grid_csplus_so4.py` | Cs⁺/SO₄²⁻ multi-ion | Deck page-15 V_RHE band |

### Gotchas

- **K⁺ vs Cs⁺**: deck baseline is K⁺/SO₄²⁻ (Linsey 2025 deck slide 9:
  `[SO₄²⁻]=0.1 M & [K⁺]=0.2 M`). Use K⁺ entry for apples-to-apples
  comparisons. See `CONJECTURE_AUDIT_2026-05-09.md`.
- **Phase 6α opt-in**: `enable_water_ionization=True` + `kw_eff_ladder`
  outer loop on `solve_anchor_with_continuation`. Default-off path is
  byte-equivalent to pre-Phase-6α.
- **`multi_ion_enabled=True` required** for ≥2 bikerman counterions
  (multi-ion shared-θ closure has different math).
- **`debye_boltzmann` IC requires** `steric_mode='bikerman'` entry;
  `steric_mode='ideal'` falls back to tanh-Gouy-Chapman.
- **`l_eff_m` is read at form-build time** via
  `bv_convergence['domain_height_hat']`; mesh y-extent must match.
- **`validate_solution_state` on muh** needs `is_logc=...`,
  `mu_species=ctx.get('mu_species')`,
  `em=ctx['nondim'].get('electromigration_prefactor', 1.0)`.
- **Step 6 ablation flags** (`apply_h_source`, `apply_k_sink`,
  `override_sigma_singh_counts_pm2`) default to v9 byte-equivalent.
  Cross-validation in `Forward/bv_solver/config.py:_get_bv_convergence_cfg`.
- **v10b constants live in `calibration/v10b.py`** (top-level
  Firedrake-free). `GAMMA_MAX_HAT_V10B = 0.047`, `K_DES_NONDIM_V10B
  = 1.0`, `C_S_F_M2_V10B = 0.20`. Deprecation alias `SMOKE =
  V10A_SMOKE` (NEVER `SMOKE = V10B`).

## Path + workflow conventions

- Run scripts from `PNPInverse/`. `scripts/Inference/` is **uppercase**.
  `StudyResults/` is the working record — check existing `summary.md`
  before regenerating expensive studies. `archive/` is reference-only.
- **Plan before non-trivial forward-solver changes.** Live backends:
  `forms_logc.py` + `forms_logc_muh.py` (conc backend removed May 2026).
- **Long-running studies cost minutes-to-hours.** Confirm first.
- **Adjoint tape hygiene** (when inverse resumes): wrap unannotated
  cold-ramp / continuation work in `with adj.stop_annotating():`.
