# PNPInverse

Research code for Poisson-Nernst-Planck / Butler-Volmer (PNP-BV)
forward simulation and (eventually) inverse kinetic-parameter
inference for a two-step oxygen reduction reaction (O₂ → H₂O₂ → H₂O,
pH 4). The code is built around Firedrake finite elements,
Pyadjoint adjoints, direct PDE inverse studies, and a separate
surrogate inference stack. **The forward solver is the active
surface; all inverse and surrogate work is paused** until the
forward pipeline is mature enough for a clean re-entry.

The target kinetic parameters are

```text
[log_k0_1, log_k0_2, alpha_1, alpha_2]
```

for

```text
R1: O2   + 2H+ + 2e- -> H2O2      E_eq ~= 0.68 V_RHE
R2: H2O2 + 2H+ + 2e- -> 2H2O      E_eq ~= 1.78 V_RHE
```

Experimental context and physical constants are tied to the 
2025 ORR dataset and the pH 4 setup documented under `docs/`.

## Current State

As of 2026-05-07, the production forward model is the
3-dynamic-species + analytic Bikerman ClO4⁻ + proton electrochemical
potential + log-rate Butler–Volmer + Stern stack. The legacy
4-species concentration backend was removed in the May 2026 cleanup.
**All inverse work is paused** until the forward solver is mature
enough for a clean re-entry — the v13–v24 study scripts, the FIM
work, and the FluxCurve adjoint pipeline are reference-only.

### Production forward stack

1. **3 dynamic species** (O2, H2O2, H+) plus an
   **analytic Bikerman counterion** for ClO4⁻ added via
   `bv_bc.boltzmann_counterions` with `steric_mode="bikerman"`.
   Closure is the steric-aware
   `c_b · exp(−z·φ) · (1 − A_dyn) / (θ_b + a_b · c_b · exp(−z·φ))`,
   matched between the IC seed and the residual side
   (`Forward/bv_solver/boltzmann.py:build_steric_boltzmann_expressions`).
2. **Proton electrochemical-potential primary variable**
   (`bv_convergence.formulation = "logc_muh"`):
   `mu_H = u_H + em·z_H·φ`. Keeps Newton smooth in deep-ψ regions
   where `u_H` and `φ` each vary by tens of log units. The other
   species use plain `u_i = ln c_i`.
3. **Log-rate Butler–Volmer evaluation** (`bv_log_rate = True`):
   `log_R = log(k0) + u_cat + Σ p·(u_sp − ln c_ref) − α·n_e·η`. The
   clip is on `η_scaled` *before* the `α·n_e` multiplication;
   `exponent_clip = 100` is the only PC-trustworthy default
   (clip=50 produces a fictitious peroxide current — see
   `docs/clipping_conventions.md`).
4. **Bikerman-consistent IC** (`initializer = "debye_boltzmann"`):
   composite-ψ (BKSA matched-asymptotic, saturated zone + outer
   exponential) plus multispecies-γ. The IC's surface activity and
   the residual's Bikerman closure agree on the saturated counterion
   concentration; an ideal-counterion fallback path is preserved for
   `steric_mode="ideal"` configs.
5. **Finite Stern compact layer**
   (`stern_capacitance_f_m2 ≈ 0.10`): absorbs ≈10–13 V_T of applied
   potential at high anodic V_RHE so the diffuse-layer drop ψ_D
   stays modest and the proton supply does not underflow the BV
   cathodic terms.

This stack reaches **V_RHE = +1.0 V at 15/15** (cold ceiling
+0.60 V; warm-walk to +1.00 V) on a 15-voltage grid spanning
V_RHE ∈ [−0.5, +1.0] via the C+D orchestrator
(`solve_grid_per_voltage_cold_with_warm_fallback`). Cross-stack
equivalence with the 4sp dynamic reference holds to ~10⁻⁹ in the
cathodic regime and ~5·10⁻³ at the +0.5 V edge. See
`docs/4sp_bikerman_ic_option_2b_results.md` for the sweep and
`docs/4sp_drop_boltzmann_investigation.md` for the investigation log.

### Recent forward-solver bugfixes (2026-05-07)

A diagnostic pass on the production stack (`scripts/diagnose_db_ic_distance.py`,
plus the verification report at `.verification/REPORT.md`) surfaced two
independent bugs in the `debye_boltzmann` IC and several smaller
defensive-validation gaps. All have been resolved in the current tree:

- **Bug #1 — Stern-η inconsistency (`forms_logc.py`, `forms_logc_muh.py`,
  `Forward/bv_solver/picard_ic.py`).** With Stern on, the residual's
  `η_raw = phi_applied − φ(0) − E_eq`, but every IC path was anchoring
  `φ(0) = phi_applied` so `η_raw|_IC` collapsed to `−E_eq`
  (V-independent, with `α·n_e·E_eq_R2 ≈ 50`, hence
  `‖F‖ ≈ exp(50) ≈ 5·10²¹` at every cathodic IC). Fixed by solving the
  Stern split `psi_S + psi_D = phi_applied − phi_o` at IC time and
  anchoring the linear-φ fallback and Picard's `η` at the OHP-side
  `phi_surface = phi_applied − psi_S`. Picard's outer loop now uses
  the Stern-consistent η as well, so its R1/R2 match what the residual
  evaluates at the same IC.
- **Bug #2 — Bikerman-γ inconsistency in Picard (`picard_ic.py`).**
  Commit `77ceff3` added `+ log_gamma` to the IC's `u_i` seeds without
  updating the Picard outer loop, so Picard solved against a γ-free
  surface c_O2 while the residual saw `O_s · γ(0)` — three orders of
  magnitude off at V_RHE = +0.5 V no-Stern. The Picard is now γ-aware
  (`a_h`, `a_cl`, `c_cl_anchor_kind` propagate into the surface
  `gamma_s` per iter; `a_h = a_cl = 0` reduces back to the legacy
  γ-free path).
- **`P_FLOOR` over-clamping (`forms_logc.py:_try_debye_boltzmann_ic`,
  muh counterpart).** The legacy `P_FLOOR = max(P_b, 1e-30)` clamped
  the Picard's surface H₂O₂ at the bulk seed, which broke the
  diffusion-limited matched-asymptotic balance `P_s ≈ R2/A2 ≪ P_b` at
  high anodic V_RHE. Reduced to a pure 1e-30 numerical floor.
- **Linear-φ fallback (`set_initial_conditions_logc[_muh]`).** When the
  Picard fails, the linear-φ IC is now also Stern-aware: it anchors
  φ(0) at `phi_surface` rather than `phi_applied`, so the fallback
  rows do not see the same `η_raw = −E_eq` collapse.

These fixes are extracted into a shared
`Forward/bv_solver/picard_ic.py` module (scalar Picard outer loop,
`solve_stern_split`, γ-aware activity coefficient,
numerically-safe `_eta_clipped` / `_safe_exp`) so the logc and
logc_muh backends call the same code path. Unit tests for the helpers
live in `tests/test_picard_ic_helpers.py`.

Two adjacent improvements landed alongside the IC fixes:

- **Phase 2 interior warm-walk (`grid_per_voltage.py`).** The C+D
  orchestrator's outer cathodic/anodic walks only visit indices
  outside `[anchor_lo, anchor_hi]`. Cold-failed *interior* gaps
  (e.g. cold succeeded at idx 2 and idx 7 but failed at 3–6) now get a
  fixed-point Phase-2 pass that warm-walks each interior failure from
  the nearest already-converged neighbour, retrying as new anchors
  appear.
- **Validator W1 + W5 fixes (`Forward/bv_solver/validation.py`,
  `FluxCurve/bv_point_solve/`).** The previously-dead W1 ("clip
  saturation") check is now implemented and uses the live
  `(phi_min, phi_max)` over the domain. W5 cation-depletion now
  interpolates `y` onto each species' subspace so the mask aligns at
  CG2+ orders, not just CG1. The two FluxCurve callers now pass
  `mu_species`, `em`, `reaction_e_eq`, and `bv_exp_scale` so muh
  contexts get correct H⁺ reconstruction inside the validator.

### Other recent updates

- **Factory defaults follow CLAUDE.md hard rules.**
  `scripts/_bv_common.py:make_bv_solver_params` now defaults
  `E_eq_r1 = 0.68`, `E_eq_r2 = 1.78` (RHE), surfaced as the named
  constants `E_EQ_R1_V`/`E_EQ_R2_V`. The previous `E_eq = 0.0`
  defaults silently bypassed Hard Rule 4 for any script that omitted
  the kwargs.
- **`config.py` cleanup.** `bv_convergence.formulation` defaults to
  `"logc"` (the legacy `"concentration"` value emits
  `DeprecationWarning` and falls through to the log-c backend).
  `bv_bc.alpha` list entries and `cathodic_species` /
  `anodic_species` indices are now bounds-checked at parse time
  rather than failing as obscure UFL `IndexError`s.
- **Mangan 2025 alignment scaffolding.** Study runs now emit an
  `experiment_metadata` block (`scripts/_bv_common.py:ExperimentMetadata`)
  with honest placeholders for the deferred M0 fields
  (`source_authority="memory"`,
  `comparison_status="internal_baseline_only"`, `N_collection`
  user-provided). RRDE-style observables (surface-pH proxy, model
  ring current, S_H₂O₂%, n_e) are computed in
  `Forward/bv_solver/rrde_observables.py` and serialised by
  `scripts/studies/peroxide_window_3sp_bikerman_muh.py`. Tests live in
  `tests/test_rrde_observables.py`. See
  `docs/Mangan2025_experimental_alignment.md` and
  `docs/m0_target_extraction.md` for the deck-alignment plan.

### Direct PDE Inverse Status — paused

All inverse scripts are **legacy / non-operational**. No inverse work
is currently running. The pipeline is held until the forward solver
is mature enough for a clean re-entry; treat the v13–v24 study
scripts (`scripts/studies/v*.py`), the FluxCurve adjoint pipeline,
and the FIM tooling as historical reference only.

When the inverse work resumes, start from
`docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`,
`docs/Next Steps After Basin Geometry.md`, and
`docs/noise_model_conventions.md`. The headline result going into
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
| `CLAUDE.md` | Project-specific conventions, hard rules (E_eq, clip, C+D, IC/residual saturation match). |
| `writeups/ForwardSolverChangesMay26/forward_solver_changes_may2026.pdf` | May 2026 production-target writeup. |
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Forward-solver rebuild narrative. |
| `docs/bv_solver_unified_api.md` | How to call the dispatcher and configure the production stack. |
| `docs/4sp_bikerman_ic_option_2b_results.md` | Production-target reference sweep: bikerman + Stern + muh + debye_boltzmann IC = 15/15 over V_RHE [−0.5, +1.0]. |
| `docs/4sp_drop_boltzmann_investigation.md` | §11–§13: investigation log behind the production target. |
| `docs/steric_analytic_clo4_reduction_handoff.md` | Derivation of the Bikerman analytic-counterion residual closure. |
| `docs/clipping_conventions.md` | Three distinct BV-related clips and the operational rule that PC is fictitious at clip=50. |
| `docs/CONTINUATION_STRATEGY_HANDOFF.md` | Why C+D over A/B for the logc+counterion stack. |
| `docs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md` | Diagnosis of the Stern-η + Bikerman-γ Picard bugs (the May 2026-05-07 IC bugfix). |
| `docs/CHATGPT_HANDOFF_13_RESPONSE_TO_CODEX_REVIEW.md` | Resolution plan and review response for those bugs. |
| `.verification/REPORT.md` | Multi-agent correctness verification of the production codepath. |
| `docs/Mangan2025_experimental_alignment.md` | Gap audit between the model and the Mangan 2025 deck. |
| `docs/m0_target_extraction.md` | M0 target-extraction plan for deck-quantitative comparison. |
| `docs/forward_solver_test_coverage.md` | What the bv_solver test suite covers. |
| `StudyResults/peroxide_window_3sp_bikerman_muh_2b/` | Reference 2b sweep on the production target stack. |
| `StudyResults/iv_curve_post_fix123/` | I-V curve regenerated after the IC bugfix. |
| `writeups/WeekOfMay4/debye_boltzmann_ic_walkthrough.pdf` | Walkthrough of the composite-ψ + multispecies-γ IC. |

When the inverse pipeline resumes, also read:
`docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md`,
`docs/Next Steps After Basin Geometry.md`,
`docs/noise_model_conventions.md`,
`docs/TODO_extend_inverse_v_range_negative.md`. Older
`CHATGPT_HANDOFF*` files are useful chronology but reflect the
pre-pause inverse state.

## Repository Layout

| Path | Role |
|---|---|
| `Forward/` | Forward solvers, parameters, noise, plotting, and steady-state utilities. |
| `Forward/bv_solver/` | Main PNP-BV package: log-c forms (`forms_logc.py`) and the muh variant (`forms_logc_muh.py`), log-rate BV, ideal + Bikerman analytic counterions (`boltzmann.py`), shared scalar Picard for the `debye_boltzmann` IC (`picard_ic.py`), per-voltage diagnostics (`diagnostics.py`), observables, RRDE post-processing (`rrde_observables.py`), validation, and the C+D continuation orchestrator (`grid_per_voltage.py`). The legacy concentration backend was removed in the May 2026 cleanup. |
| `Inverse/` | Generic Pyadjoint inverse framework and objective factories. **Inverse paused.** |
| `FluxCurve/` | Adjoint-gradient curve-fitting framework for Robin and BV flux/current curves. **Inverse paused — reference only.** |
| `Nondim/` | Physical constants, scaling transforms, and compatibility wrappers. |
| `Surrogate/` | Surrogate models (RBF, NN, GP, PCE, POD-RBF, multistart, BCD, cascade, ISMO). **Paused with the inverse pipeline.** |
| `scripts/studies/` | Forward-solver study scripts and diagnostics. The current driver is `peroxide_window_3sp_bikerman_muh.py`; v13–v24 are legacy inverse studies. |
| `scripts/verification/` | MMS and BV forward strategy verification scripts. |
| `scripts/profile/` | Performance-profile runners for the production sweep. |
| `scripts/surrogate/` | Surrogate training, validation, GP/PCE/NN drivers, ISMO drivers (paused). |
| `scripts/Inference/` | Older master inverse entry points and wrappers. Kept for reproducibility (uppercase `Inference`). |
| `docs/` | Handoffs, plans, conventions, equations, literature inputs, and current status notes. |
| `writeups/` | PDF/TeX reports (Apr 27 solver writeup, May 2026 forward-solver-changes writeup, May 4 IC walkthrough, V&V report). |
| `StudyResults/` | Generated results, summaries, plots, JSON, CSV, and run logs. Working research record, not a clean build-artifact directory. |
| `tests/` | Pytest regression and verification tests. Firedrake tests are marked `slow`. |
| `archive/` | Old results/code for reference, not the active implementation surface. |

There is no current `scripts/bv/` directory and no lowercase
`scripts/inference/` directory. Use `scripts/studies/`, `scripts/Inference/`,
and `scripts/surrogate/`.

## Core Forward-Solver Configuration

The active production stack is controlled through `solver_params[10]`.
The factory is `scripts/_bv_common.py:make_bv_solver_params`. The
high-level call shape for the production target is:

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,   # steric_mode='bikerman'
)
from Forward.bv_solver import solve_grid_per_voltage_cold_with_warm_fallback

sp = make_bv_solver_params(
    ...,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    formulation="logc_muh",
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,
    initializer="debye_boltzmann",
    # E_eq_r1=0.68, E_eq_r2=1.78 are now the factory defaults
    # (CLAUDE.md Hard Rule 4).
)
```

Reference driver:
`scripts/studies/peroxide_window_3sp_bikerman_muh.py`. The full API
shape produced by the factory is:

```python
solver_options = {
    "bv_convergence": {
        "formulation": "logc_muh",      # or "logc" for the plain log-c backend
        "initializer": "debye_boltzmann",  # or "linear_phi"
        "bv_log_rate": True,
        "clip_exponent": True,
        "exponent_clip": 100.0,         # only PC-trustworthy default
        "u_clamp": 100.0,
    },
    "bv_bc": {
        "reactions": [
            {"k0": k0_1, "alpha": alpha_1, "cathodic_species": 0,
             "anodic_species": 1, "stoichiometry": [-1, +1, -2],
             "n_electrons": 2, "reversible": True, "E_eq_v": 0.68,
             "cathodic_conc_factors": [
                 {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat}]},
            {"k0": k0_2, "alpha": alpha_2, "cathodic_species": 1,
             "anodic_species": None, "stoichiometry": [0, -1, -2],
             "n_electrons": 2, "reversible": False, "E_eq_v": 1.78,
             "cathodic_conc_factors": [
                 {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat}]},
        ],
        "boltzmann_counterions": [
            {"z": -1, "c_bulk_nondim": c_clo4_hat, "phi_clamp": 50.0,
             "steric_mode": "bikerman", "a_nondim": 0.01},
        ],
        "stern_capacitance_f_m2": 0.10,    # omit for no-Stern Dirichlet BC
        "electrode_marker": 3,
        "concentration_marker": 4,
        "ground_marker": 4,
    },
}
```

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

Gotchas (see CLAUDE.md for the full list):

- `set_initial_conditions(ctx, sp, blob=True)` is silently ignored in
  log-c mode (no blob IC for `u_i = ln c_i`).
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
- Use the C+D orchestrator
  (`solve_grid_per_voltage_cold_with_warm_fallback`), not Strategy
  B (`solve_grid_with_charge_continuation`); B's Phase-1 V-sweep at
  z=0 hands bisection a mismatched species IC on the
  logc+counterion stack.

## Environment

Run from the repository root:

```bash
cd /path/to/PNPInverse
```

Firedrake is installed separately. The local development convention in this
workspace is a Firedrake virtual environment in the parent directory:

```bash
source ../venv-firedrake/bin/activate
python -m pip install -e ".[dev]"
```

Useful cache settings for Firedrake/PyOP2 runs:

```bash
export MPLCONFIGDIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYOP2_CACHE_DIR=/tmp/pyop2
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc
export OMP_NUM_THREADS=1
```

Core dependencies are Firedrake, `firedrake.adjoint`, NumPy, SciPy,
Matplotlib, h5py, imageio, and Pillow. Surrogate models additionally need
PyTorch/GPyTorch for NN/GP work and ChaosPy for PCE.

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

The production target sweep (3sp + bikerman + muh + Stern +
debye_boltzmann IC over V_RHE ∈ [−0.5, +1.0]):

```bash
python scripts/studies/peroxide_window_3sp_bikerman_muh.py
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

Inverse and surrogate scripts are paused. Re-running them is not part
of the current workflow; treat their command lines in handoff
documents as historical reference only.

## Noise Models (paused inverse pipeline)

Recorded for when the inverse work resumes. Default to
`local_rel + abs_floor`; report `global_max`, `local_rel`, and
`local_rel + abs_floor` together when feasible. `global_max`
rotations are suspect when CD/PC spans many decades across the
voltage grid (the negative-V FIM study is the cautionary example).
See `docs/noise_model_conventions.md`.

## Known Gotchas

- Run scripts from `PNPInverse/` (this directory), not from
  `Forward/` or `scripts/`. The Firedrake venv is in the parent
  directory: `source ../venv-firedrake/bin/activate`.
- Forward studies are expensive — minutes to hours depending on
  mesh, voltage grid, and Phase-2 fill behaviour.
  `StudyResults/` is part of the working research record, not a
  clean build-artifact directory; check existing `summary.md`
  files before regenerating.
- `scripts/Inference/` is uppercase. Older README text and notes
  may refer to paths that no longer exist.
- Use the C+D orchestrator
  (`solve_grid_per_voltage_cold_with_warm_fallback`), not the
  Strategy-B z-ramp continuation. The Bikerman residual closure
  and the IC's matched-asymptotic seed must agree about steric
  saturation; mixing a bikerman IC with an ideal-counterion residual
  (or vice-versa) cold-fails on the saturated manifold.
- `exponent_clip = 100` is the only PC-trustworthy default. Older
  results at `clip = 50` produce a fictitious peroxide current
  across the cathodic grid; do not compare them against experiment.
  Some configs cold-fail more often at clip=100 than at clip=50;
  recover with C+D warm-walk or Stern, not by lowering the clip.
- The 4sp dynamic stack is a validation reference (cathodic
  agreement to ~10⁻⁹; +0.5 V edge ~5·10⁻³). Its anodic ceiling is
  bound by the dynamic c_ClO₄ NP equation, not the IC; "go fully
  dynamic" is *not* a fix for the anodic ceiling.
- Inverse status is paused; do not claim single-experiment
  four-parameter recovery is solved.
