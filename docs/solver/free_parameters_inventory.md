# Free parameters of the PNP+BV forward model

Comprehensive inventory of every knob that can be tuned to change
model behavior, separated into:

* **Group A** — shared between no-hydrolysis and hydrolysis-ON models
* **Group B** — only active when hydrolysis is enabled
* **Group C** — solver / numerics (rarely physically meaningful)
* **Group D** — what we have been actively varying

Each entry cites the source-of-truth file. "Locked" means a hard rule
(`CLAUDE.md` or a Phase 6β invariant) — don't touch without an
explicit plan. "Free" means it's a legitimate tuning knob.

Compiled 2026-05-22 from the codebase + recent K0 sweeps.


## Group A — shared parameters (no-hydrolysis AND hydrolysis-ON)

### A.1 Butler–Volmer kinetics

Source: `scripts/_bv_common.py` lines 93–124, `PARALLEL_2E_4E_REACTIONS_4SP`.

| Parameter | Symbol / variable | Default | Status | Notes |
|---|---|---|---|---|
| 2e rate constant | `K0_PHYS_R2E` | 2.4e-8 m/s | Locked | = `K0_PHYS_R1` (Ruggiero 2022) |
| 4e rate constant | `K0_PHYS_R4E` | same baseline × `K0_R4e_factor` | **Free** | Primary K0_R4e_factor lever, swept across decades |
| 2e transfer coeff | `ALPHA_R2E` | 0.627 | Free | M4 Tafel calibration planned |
| 4e transfer coeff | `ALPHA_R4E` | 0.5 | Free | Placeholder; revisit with Tafel data |
| 2e equilibrium V | `E_EQ_R2E_V` | 0.695 V vs RHE | **Locked** | CLAUDE.md Hard Rule #4 |
| 4e equilibrium V | `E_EQ_R4E_V` | 1.23 V vs RHE | **Locked** | Ruggiero 2022 §1 |
| n_electrons | n_R2e, n_R4e | 2, 4 | **Locked** | Stoichiometry |
| Reversibility | `reversible` flag | R2e=True, R4e=False | Locked | Stoichiometric |
| Stoichiometry per reaction | `stoichiometry` lists | per `PARALLEL_2E_4E_REACTIONS_4SP` | Locked | |
| η_scaled exponent clip | `exponent_clip` | 100.0 | **Locked** | CLAUDE.md Hard Rule #2 — never lower |

### A.2 Bulk concentrations

Source: `scripts/_bv_common.py` lines 86–90, 747–748, 793.

| Parameter | Default | Status | Notes |
|---|---|---|---|
| C_O2_bulk | 1.2 mol/m³ | **Locked** | Ruggiero §2.4 salt-corrected; never 5 mol/m³ (Hard rule) |
| C_H+_bulk (sets pH) | 0.1 mol/m³ (pH 4) | Free per study | Bulk-pH lever (the 4-cation × 3-pH CP grid) |
| C_K+_bulk | 199.9 mol/m³ | Locked by electroneutrality | |
| C_Cs+_bulk | 199.9 mol/m³ | Locked by electroneutrality | |
| C_SO4²⁻_bulk | 100.0 mol/m³ | Locked | Ruggiero §2 baseline |
| C_H2O2_seed | 1e-4 nondim | Locked | IC only |
| C_OH⁻_bulk | 1e-10 (pH 4) | Auto from KW | Only when water ionization on |

### A.3 Diffusion coefficients

Source: `scripts/_bv_common.py` lines 73–76, 220, 796, and `D_CSPLUS=2.06e-9` in `_run_cs_pH4_k0fit_vs_slide15.py:53`.

| Species | D (m²/s) | Status |
|---|---|---|
| O₂ | 1.9e-9 | Locked |
| H₂O₂ | 1.6e-9 | Locked |
| H⁺ | 9.311e-9 | Locked — sets the H⁺ Levich limit we keep hitting |
| K⁺ | 1.96e-9 | Locked |
| Cs⁺ | 2.06e-9 | Locked |
| SO₄²⁻ | 1.065e-9 | Locked |
| OH⁻ | 5.273e-9 | Locked — only when water ionization on |

### A.4 Bikerman steric parameters

Source: `scripts/_bv_common.py` lines 166–188, 744–745, 804.
Bikerman sets `c_max ≈ 1/a` per species.

| Species | Default `a_nondim` | Status | Notes |
|---|---|---|---|
| O₂ | `A_O2_HAT ≈ 1.49e-5` (r=1.70 Å) | Free (physical) | Used by solver_demo |
| H₂O₂ | `A_H2O2_HAT ≈ 2.42e-5` (r=2.00 Å) | Free (physical) | |
| H⁺ | `A_HP_HAT ≈ 6.64e-5` (r=2.80 Å H₃O⁺) **OR** `A_DEFAULT=0.01` (≈14.9 Å) | **Open caveat** | **CLAUDE.md Hard Rule #7** — solver_demo uses physical, v10a/A.2/Phase D use A_DEFAULT. Big lever, unresolved 2026-05-12 |
| K⁺ | `A_KPLUS_HAT` (Marcus radius) | Locked | |
| Cs⁺ | `A_CSPLUS_HAT ≈ 3.23e-5` (r=2.2 Å) | Locked | |
| SO₄²⁻ | `A_SO4_HAT ≈ 4.20e-5` (r=2.4 Å) | Locked | |
| OH⁻ | `A_OH_HAT` | Locked | |

### A.5 Stern compact-layer capacitance

Source: `calibration/v10b.py:64`, `phase6b_v10a_v_sweep_diagnostic.py:89,106`,
`docs/phase6/CMK3_capacitance_literature.md`.

| Parameter | Default | Status | Notes |
|---|---|---|---|
| `stern_capacitance_f_m2` (C_S) | **0.20 F/m²** | Free | Bohra-Koper-Choi consensus; bracket {0.05, 0.10, 0.20, 0.30}. Yash equiv ≈1.38 from L_S=0.5 nm × ε_r=78. CLAUDE.md Hard Rule #6 |
| `STERN_F_M2_ANCHOR` | 0.10 F/m² | **Hard invariant** | Anchor build value; bump-ladder afterward |

### A.6 Geometry and mesh

Source: `phase6b_v10a_v_sweep_diagnostic.py:135,203–205`.

| Parameter | Default | Status |
|---|---|---|
| `l_eff_m` (diffusion-layer thickness) | 100e-6 m | Free — production, also tested 6e-6 (Yash), 16e-6, 300e-6 |
| `domain_height_hat` | `l_eff_m / 1e-4` | Auto |
| `MESH_NX` | 8 | Free (numerical) |
| `MESH_NY` | 80 | Free (numerical) |
| `MESH_BETA` | 3.0 | Free (numerical) |
| `electrode_marker` | int (boundary id) | Locked by mesh |
| `electrode_area_nondim` | assembled from mesh | Locked |

### A.7 Formulation / BC / initializer

Source: `make_bv_solver_params()` in `scripts/_bv_common.py`.

| Parameter | Choices | Status | Notes |
|---|---|---|---|
| `formulation` | "logc_muh" / "conc" (removed) | **Locked** | Production stack |
| `log_rate` | True | **Locked** | |
| `u_clamp` | 100.0 | **Locked** | Hard Rule #2 |
| `initializer` | "debye_boltzmann" / "linear_phi" | Free | Debye assumes Stern; linear_phi for no-Stern |
| `multi_ion_enabled` | True | Locked when ≥2 Bikerman counterions | |


## Group B — hydrolysis-specific knobs (only when hydrolysis ON)

### B.1 Water self-ionization

Source: `scripts/_bv_common.py:218–237`, `phase6b_v10a_v_sweep_diagnostic.py:_build_kw_ladder` (line ~950).

| Parameter | Default | Status | Notes |
|---|---|---|---|
| `enable_water_ionization` | **True** for hydrol-ON | Toggle | Adds Kw split + proton-condition residual |
| `KW_PHYS` | 1.0e-8 (mol/m³)² | Locked | Standard at 25 °C |
| `KW_HAT` | nondim (=KW_PHYS / C_SCALE²) | Auto | |
| `kw_eff_ladder` | (0, KW_HAT·1e-6, KW_HAT·1e-3, KW_HAT·0.1, KW_HAT) | Free (continuation) | Tunable rung schedule |

### B.2 Cation hydrolysis (Singh-style pKa shift)

Source: `calibration/v10b.py` (`V10B_KINETICS`); `Forward/bv_solver/cation_hydrolysis.py` lines 228–240.

| Parameter | Default (V10B) | Status | Notes |
|---|---|---|---|
| `enable_cation_hydrolysis` | True for hydrol-ON | Toggle | |
| `k_hyd_nondim` | 1e-3 | Free | Forward rate (cation + H₂O → MOH + H⁺) |
| `k_prot_nondim` | 1e-3 | Free | Reverse protonation rate |
| `k_des_nondim` | 1.0 | **Engineering choice** | Eyring prior bracket [1e-2, 1e2] ↔ ΔG_des [0.69, 0.94] eV; sensitivity rungs {0.01, 0.1, 1, 10, 100} |
| `delta_ohp_hat` | 4e-6 (= 0.40 nm / 100 µm) | **Locked** | OHP thickness |
| `gamma_max_nondim` | 0.047 (V10B) | Free | Langmuir saturation cap; bracket {V10A/2, V10A, V10A·2} |
| `k_hyd_baseline` | 1e-3 | **Hard invariant** | |
| `k_hyd_route` | 1e-1 | **Hard invariant** | Per A.2 / v10b |
| `lambda_hydrolysis` | 0 → 1 ladder | Free (homotopy) | Big toggle: λ=0 disables source, λ=1 = full |
| `LAMBDA_LADDER` | (0, 0.25, 0.50, 0.75, 1.0) | **Locked** | 5-rung ramp per V |

### B.3 Singh pKa shift parameters

Source: `Forward/bv_solver/cation_hydrolysis.py:228–240`, `scripts/_bv_common.py:make_singh_pka_shift_params`.

| Parameter | Default (K⁺) | Status | Notes |
|---|---|---|---|
| `cation` | "K+" production, "Cs+" slide-15 | Free | Selects Singh Table S1 row |
| `z_eff` | 0.919 (K⁺), 0.930 (Cs⁺) | Per-cation | Effective charge |
| `r_M_pm` | 138 (K⁺), 170 (Cs⁺) | Per-cation | Cation hard-sphere radius |
| `pKa_bulk` | 14.5 (K⁺), 14.8 (Cs⁺) | Per-cation | Bulk pKa |
| `r_H_El_pm` | 200.98 (Cu Singh prior) | Free | Hydration-H-to-electrode distance — M4 Tafel-calibration target |
| `A_pm` | 620.32 | Locked | Singh global constant |
| `B` | 17.154 | Locked | Singh global constant |
| `r_O_pm` | 63.0 | Locked | Singh global constant |
| `anode_clamp` | True | Free | Clamp Δ_pKa to 0 at anodic bias |
| `pka_shift_form` | "singh_2016_eq_4" | Free | Form of pKa shift; "placeholder" available |

### B.4 Phase D / σ-mapping experimental knobs

Source: `phase6b_step10_phase_D_fit_eval.py`, `Forward/bv_solver/config.py:_get_bv_convergence_cfg`.

| Parameter | Default | Status | Notes |
|---|---|---|---|
| `delta_beta_pm2` | 0.0 | Free (was sweep target) | Phase D found non-identifiable; bracket [−10, +10] explored |
| `sigma_mapping` | "stern" | Free | Two modes: "stern" or "ablation_singh_0.141" |
| `override_sigma_singh_counts_pm2` | None | Free | Step-6 ablation override |
| `apply_h_source` | True | Step-6 ablation | Include H source in Poisson |
| `apply_k_sink` | True | Step-6 ablation | Include K sink |


## Group C — solver / numerics

Source: `phase6b_v10a_v_sweep_diagnostic.py` (SNES + ladders), `scripts/_bv_common.py:SNES_OPTS_CHARGED`.

| Parameter | Default | Status |
|---|---|---|
| `snes_max_it` | 400 | Free (numerical) |
| `snes_atol` | 1e-7 | Free (numerical) |
| `snes_rtol` | 1e-10 | Free (numerical) |
| `snes_stol` | 1e-12 | Free (numerical) |
| `snes_linesearch_type` | "l2" | Free |
| `snes_linesearch_maxlambda` | 0.3 | Free |
| `K0_INITIAL_SCALES` | (1e-12, 1e-9, 1e-6, 1e-3, 1.0) | Free (continuation ladder) |
| `max_inserts_per_step` | 4 | Free |
| `max_ss_steps_per_rung` | 300 | Free |
| Warm-walk: `walk_n_substeps` | 8 (production) | Free |
| Warm-walk: `walk_max_ss_steps` | 150 (production) | Free |
| Warm-walk: `walk_ss_rel_tol` | 1e-4 (production) | Free |
| Grid-walk: `n_substeps_warm` | 8 | Free |
| Grid-walk: `bisect_depth_warm` | 5 | Free |
| Stern bump-ladder rungs (verified) | (0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0) | Free |


## Group D — what we are actively varying (May 2026)

| Lever | Recent sweep values | Where |
|---|---|---|
| `K0_R4e_factor` | {1, 1e-6, 1e-12, 1e-15, 1e-16, 3e-17, 1e-17, 2.52e-18, 3e-18, 1e-18} | yash_match_no_hydrolysis_25pt/, phase6b_k0_r4e_sweep_*/, yash_match_hydrolysis_on/ |
| `l_eff_m` | {6e-6 (Yash), 16e-6 (v10a default), 100e-6 (production)} | solver_demo, phase D |
| `stern_capacitance_f_m2` | {0.10 anchor, 0.20 production, 1.38 Yash-equiv, 100 near-no-Stern} | solver_demo --stern-final |
| `cation` | K⁺ (deck) vs Cs⁺ (slide-15, Yash, recent runs) | per-Singh-row |
| `lambda_hydrolysis` | 0 (off) vs 1 (full) | yash_match_no_hydrolysis vs yash_match_hydrolysis_on |
| `gamma_max_nondim` | sensitivity bracket {0.0235, 0.047, 0.094} | Phase D D7-D4 |
| `delta_beta_pm2` | Phase D scan {−10 … +10 pm²} → non-identifiable | StudyResults/phase6b_step10_phase_D/ |
| H⁺ Bikerman `a` | A_DEFAULT=0.01 (v10a) vs physical A_HP_HAT=6.64e-5 (solver_demo) | Hard Rule #7 open caveat |
| bulk pH | {4, 6.39} | CP_data.csv, Brianna 8-15-19 |


## Important caveats

1. **Hard-Rule-locked parameters**: `E_eq_R2e/R4e`, `exponent_clip=100`, `u_clamp=100`, `formulation="logc_muh"`, `C_O2=1.2 mol/m³`, `delta_ohp_hat=4e-6`. Don't touch without an explicit plan + GPT critique pass.

2. **Engineering-choice flag set**: `k_des_nondim`. Documented free knob with declared Eyring prior bracket.

3. **Open caveat (Hard Rule #7)**: H⁺ Bikerman `a` is the largest unresolved physics knob. solver_demo uses physical (≈6.6e-5); v10a/A.2/Phase D use `A_DEFAULT=0.01` (≈14.9 Å effective radius). Differs by ~150× in c_max. Treat any plateau-set-by-Levich finding as suspect until physical-a bridge runs disambiguate.

4. **Singh constants are per-cation**: swapping `cation="K+"` → `"Cs+"` automatically changes z_eff, r_M_pm, pKa_bulk — three knobs at once. To probe each independently, override per-row values directly.

5. **`r_H_El_pm = 200.98 pm` is the Cu Singh prior**, not a deck-anchored measurement. Phase 6β plans an r_H_El sweep to recalibrate for sp²-carbon (CMK-3).

6. **`stern_capacitance_f_m2` is per-local-surface-element, NOT per-geometric-area** for CMK-3 (RF ≈ 6000 implicit in fitted k₀). See `docs/phase6/CMK3_capacitance_literature.md`.

7. **C_S bump ladder**: `STERN_F_M2_ANCHOR=0.10 → STERN_F_M2_BASELINE=0.20` is a single-step bump in the v10a path. Reaching higher C_S (e.g., 1.38 Yash equiv) requires the multi-step verified ladder (0.20 → 0.50 → 1.0 → target).

8. **`c_s_ladder + kw_eff_ladder` combo is `NotImplementedError`**: cannot ramp both simultaneously. Use Stern bump-ladder post-anchor for C_S; kw_eff_ladder during anchor for Kw.


## Cross-references

- `docs/solver/bv_solver_unified_api.md` — how to call the production stack
- `docs/solver/clipping_conventions.md` — exponent_clip + u_clamp details
- `docs/phase6/v10b_calibration_summary.md` — v10b Γ_max, k_des, C_S decision rules
- `docs/phase6/CMK3_capacitance_literature.md` — C_S = 0.20 F/m² provenance
- `docs/phase6/singh_2016_pka_formula.md` — Singh Eq. 3/4 and σ-mapping
- `calibration/v10b.py` — Firedrake-free V10B_KINETICS dict
- `CLAUDE.md` — hard rules and operational invariants
