# PNPInverse

Research code for Poisson-Nernst-Planck / Butler-Volmer (PNP-BV) forward
solves and inverse kinetic-parameter studies for a two-step oxygen reduction
reaction (O2/H2O2, pH 4). The code is built around Firedrake finite elements,
Pyadjoint adjoints, direct PDE inverse studies, and a separate surrogate
inference stack.

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

As of 2026-05-01, the production forward model is not the original full
4-species concentration-formulation solver. The active solver stack for the
latest inverse work is:

1. **3 explicit transported species:** O2, H2O2, H+.
2. **Analytic Boltzmann counterion:** inert ClO4- is removed as a dynamic
   Nernst-Planck unknown and added to Poisson through
   `bv_bc.boltzmann_counterions`.
3. **Log-concentration primary variables:** `u_i = ln(c_i)`, selected by
   `bv_convergence.formulation = "logc"`.
4. **Log-rate Butler-Volmer evaluation:** selected by
   `bv_convergence.bv_log_rate = True`.
5. **Two-observable inverse data:** current density (CD) plus peroxide
   current (PC), both assembled from the per-reaction BV rates.

The most important recent validation result is V24:

```text
StudyResults/v24_3sp_logc_vs_4sp_validation/summary.md
```

On the 8-voltage overlap window where both the canonical 4-species solver
and current 3sp+Boltzmann log-c solver converge, all points pass the 5%
F2-style observable tolerance. The max differences were 0.269% of CD max
and 0.104% of PC max.

### Direct PDE Inverse Status

The log-rate BV change fixed the old local Fisher-information failure:

```text
old low-voltage / clipped model:
  cond(F) ~= 2.03e11
  weak direction ~= log_k0_2

current log-rate G0 grid:
  V_GRID = [-0.10, +0.10, +0.20, +0.30, +0.40, +0.50, +0.60] V_RHE
  cond(F) ~= 1.79e7
  ridge_cos ~= 0.031
  weak direction ~= log_k0_1
```

That is a real improvement, but it does **not** mean the single-experiment
four-parameter inverse is globally solved. The current conclusion from
V19-V23 is:

- The adjoint/FIM pipeline is reliable. The apparent factor-of-two
  adjoint-vs-finite-difference mismatch was a warm-start FD artifact;
  cold-ramp FD matches the adjoint at the relevant voltages.
- `alpha_1` and `alpha_2` are robustly data-identifiable in several clean
  starts, often to about 0.02-2%.
- Each `k0` can be reached from some starts, but **joint recovery of both
  `k0_1` and `k0_2` from one CD+PC experiment is initialization-dependent**.
- LM is worse than bounded TRF on this problem.
- Log-`k0` Tikhonov priors with literature-scale uncertainty
  `sigma = log(3)` or `log(10)` move endpoints within a basin but do not
  cross the basin barriers.
- Negative-voltage extension gives only a weak local FIM improvement under
  local-relative noise and does not rotate the remaining weak direction.
- Restart-with-perturbation can sometimes escape a wrong basin, so it is a
  diagnostic/practical tactic, not a structural identifiability fix.

The honest project statement is:

```text
The log-rate BV formulation removes the old clipped-R2 local
identifiability failure, but a single steady-state CD+PC experiment still
has a multi-basin Tafel-ridge objective. Transfer coefficients are robustly
identifiable; joint exchange-rate recovery needs tighter prior information
than is usually defensible or additional independent experiments/observables.
```

The next main path is multi-experiment Fisher screening, starting with bulk
O2 variation, then H2O2-fed R2 isolation, then L_ref / rotation variation.
Do not run noisy synthetic seeds until clean multi-experiment recovery works
from multiple initializations.

### Surrogate Pipeline Status

`Surrogate/` is a real, separately useful framework. It includes RBF, neural
network, NN ensemble, GP, PCE, POD-RBF, multistart, block-coordinate descent,
cascade inference, and ISMO components.

The V&V report in `writeups/vv_report/` documents passing checks for the
surrogate-era pipeline:

- MMS forward convergence: `L2 ~ O(h^2)`, `H1 ~ O(h)`.
- Surrogate hold-out fidelity across six model families.
- Parameter recovery through 0-2% noise gates.
- Gradient consistency and pipeline reproducibility.

Do not confuse that report with the latest direct-PDE inverse conclusion
above. The newer late-April work is about the 3sp+Boltzmann/log-c/log-rate
solver and the remaining single-experiment basin geometry.

## What To Read First

Use these as the current source of truth:

| File | Purpose |
|---|---|
| `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` | Main narrative of the forward-solver rebuild. |
| `writeups/WeekOfApr27/literature_provenance.md` | Provenance and novelty audit for PBNP, log-c, and log-rate BV. |
| `docs/Next Steps After Basin Geometry.md` | Current inverse-problem plan after V20-V22. |
| `docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md` | LM and Tikhonov results; basin-geometry conclusion. |
| `docs/noise_model_conventions.md` | Current FIM/inverse noise-model convention. |
| `docs/TODO_extend_inverse_v_range_negative.md` | Negative-voltage extension rationale and caveats. |
| `StudyResults/v24_3sp_logc_vs_4sp_validation/summary.md` | Current 3sp vs 4sp overlap validation. |
| `StudyResults/v23_negative_v_fim_ablation/summary.md` | Negative-V FIM result and noise-model artifact warning. |
| `writeups/vv_report/vv_report.pdf` | Surrogate-era V&V report. |

Older `CHATGPT_HANDOFF*` files are useful chronology, but the newest docs
above supersede earlier optimism that the single-experiment inverse was fully
fixed.

## Repository Layout

| Path | Role |
|---|---|
| `Forward/` | Forward solvers, parameters, noise, plotting, and steady-state utilities. |
| `Forward/bv_solver/` | Main PNP-BV package: concentration/log-c forms, log-rate BV, Boltzmann counterions, observables, validation, continuation, hybrid/stabilized/robust/Gummel helpers. |
| `Inverse/` | Generic Pyadjoint inverse framework and objective factories. |
| `FluxCurve/` | Adjoint-gradient curve-fitting framework for Robin and BV flux/current curves, including point-solve caches and parallel helpers. |
| `Nondim/` | Physical constants, scaling transforms, and compatibility wrappers. |
| `Surrogate/` | Surrogate models, training, validation, multistart, cascade, BCD, ISMO, acquisition, retraining, and PDE evaluation integration. |
| `scripts/studies/` | Current direct-PDE study scripts and diagnostics. This is where the v18-v24 inverse/FIM/validation work lives. |
| `scripts/verification/` | MMS and BV forward strategy verification scripts. |
| `scripts/surrogate/` | Surrogate training, validation, GP/PCE/NN drivers, ISMO drivers. |
| `scripts/Inference/` | Older master inverse entry points and wrappers. Kept for reproducibility. |
| `docs/` | Handoffs, plans, conventions, equations, literature inputs, and current status notes. |
| `writeups/` | PDF/TeX reports, including the Apr 27 solver writeup and V&V report. |
| `StudyResults/` | Generated results, summaries, plots, JSON, CSV, and run logs. |
| `tests/` | Pytest regression and verification tests. Firedrake tests are marked `slow`. |
| `archive/` | Old results/code for reference, not the active implementation surface. |

There is no current `scripts/bv/` directory and no lowercase
`scripts/inference/` directory. Use `scripts/studies/`, `scripts/Inference/`,
and `scripts/surrogate/`.

## Core Forward-Solver Configuration

The active production stack is controlled through `solver_params[10]`:

```python
solver_options = {
    "bv_convergence": {
        "formulation": "logc",
        "bv_log_rate": True,
        "clip_exponent": True,
        "exponent_clip": 50.0,
        "u_clamp": 30.0,
    },
    "bv_bc": {
        "reactions": [
            {
                "k0": k0_1,
                "alpha": alpha_1,
                "cathodic_species": 0,       # O2
                "anodic_species": 1,         # H2O2
                "stoichiometry": [-1, +1, -2],
                "n_electrons": 2,
                "reversible": True,
                "E_eq_v": 0.68,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
            {
                "k0": k0_2,
                "alpha": alpha_2,
                "cathodic_species": 1,       # H2O2
                "anodic_species": None,
                "stoichiometry": [0, -1, -2],
                "n_electrons": 2,
                "reversible": False,
                "E_eq_v": 1.78,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
        ],
        "boltzmann_counterions": [
            {"z": -1, "c_bulk_nondim": c_clo4_hat, "phi_clamp": 50.0},
        ],
        "electrode_marker": 3,
        "concentration_marker": 4,
        "ground_marker": 4,
    },
}
```

The dispatcher in `Forward/bv_solver/__init__.py` routes
`build_context()`, `build_forms()`, and `set_initial_conditions()` to the
concentration or log-c backend based on `bv_convergence.formulation`.

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

Lightweight tests that do not require full PDE solves:

```bash
python -m pytest -m "not slow"
```

Firedrake-dependent verification:

```bash
python -m pytest -m slow
python scripts/verification/mms_bv_4species.py
python scripts/verification/mms_bv_3sp_logc_boltzmann.py
```

Regenerate the current 3sp+Boltzmann vs 4sp overlap validation:

```bash
python scripts/studies/v24_3sp_logc_vs_4sp_validation.py
```

Generate the TRUE 3sp log-c/log-rate I-V scan over the wide voltage range:

```bash
python scripts/plot_iv_curves_3sp_true.py
```

Run the current direct-PDE clean inverse on the G0 grid:

```bash
python scripts/studies/v18_logc_lsq_inverse.py \
  --method trf \
  --log-rate \
  --v-grid -0.10 0.10 0.20 0.30 0.40 0.50 0.60 \
  --init minus20
```

Aggregate late-April diagnostics that already have local per-run outputs:

```bash
python scripts/studies/v23_restart_perturbation.py --aggregate-only
python scripts/studies/v23_anchored_tafel_lsq_inverse.py --aggregate-only
```

Run the next multi-experiment FIM screen from scratch:

```bash
python scripts/studies/v23_multiexperiment_fim.py
```

Reuse the existing negative-voltage FIM sensitivities without recomputing
PDE solves:

```bash
python scripts/studies/v23_negative_v_fim_ablation.py \
  --skip-solves \
  --extended-json StudyResults/v23_negative_v_fim_ablation/sensitivities_extended.json
```

Surrogate examples:

```bash
python scripts/surrogate/train_gp.py \
  --training-data data/surrogate_models/training_data_merged.npz \
  --output-dir data/surrogate_models/gp

python scripts/surrogate/validate_surrogate.py \
  --model data/surrogate_models/model_pod_rbf_nolog.pkl \
  --test-data data/surrogate_models/training_data_merged.npz \
  --output-dir StudyResults/surrogate_fidelity

python scripts/surrogate/run_ismo.py \
  --surrogate-type nn_ensemble \
  --design D1-default \
  --max-iterations 3 \
  --budget 60 \
  --skip-post-validation
```

## Noise Models

For FIM and inverse analysis, the current convention is:

1. Report all three models when possible: `global_max`, `local_rel`, and
   `local_rel + abs_floor`.
2. Use `local_rel + abs_floor` as the recommended default.
3. Treat `global_max` rotations as suspect when CD/PC spans many decades
   across the voltage grid.

The negative-voltage study is the cautionary example: adding V <= -0.30
appears to rotate the weak direction under `global_max`, but the condition
number gets much worse because large negative-V PC inflates `sigma_pc` and
demotes the positive-V PC rows. Under local-relative noise, the same grid
only sharpens the log_k0_1 curvature modestly.

See `docs/noise_model_conventions.md`.

## Known Gotchas

- The active inverse scripts are expensive. Many direct PDE study runs take
  minutes to hours depending on mesh, voltage grid, and number of starts.
- `StudyResults/` is part of the working research record, not a clean build
  artifact directory. Check existing summaries before rerunning expensive
  studies.
- `scripts/Inference/` is uppercase. Older README text and notes may refer
  to paths that no longer exist.
- The full 4-species CG1 concentration solver still has real positivity
  trouble in the stiff anodic/onset regimes. The current production route
  avoids that by using the 3sp+Boltzmann + log-c + log-rate formulation.
- Do not claim single-experiment four-parameter recovery is solved. The
  correct current result is local Fisher improvement plus persistent global
  multi-basin geometry.
