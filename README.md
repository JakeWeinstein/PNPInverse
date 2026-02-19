# PNPInverse

Inverse and forward experiments for a Poisson-Nernst-Planck (PNP) model using Firedrake and `firedrake.adjoint` (Pyadjoint).

## Unified Inverse Interface

The inverse workflow is now centralized in `UnifiedInverse/`.

You can:
- Plug in any forward solver module that exposes the expected contract (`build_context`, `build_forms`, `set_initial_conditions`, and `forsolve`/equivalent solve function).
- Choose a parameter target from a registry (`diffusion`, `dirichlet_phi0`, `robin_kappa`).
- Set true value, noise amount, and initial guess.
- Automatically generate synthetic data and run optimization end-to-end.

### Core modules

- `UnifiedInverse/solver_interface.py`
  - Forward-solver adapter loading and contract handling.
- `UnifiedInverse/parameter_targets.py`
  - Parameter target definitions and registry.
- `UnifiedInverse/inference_runner.py`
  - Synthetic data generation, objective construction, and inference runner.
- `InferenceScripts/Infer_parameter_from_data.py`
  - Unified CLI-style entrypoint.

## Folder Layout

- `InferenceScripts/`
  - Active inference entry scripts.
- `Studies/`
  - Tests, probes, and diagnostic/benchmark studies.
- `Helpers/`
  - Backward-compatible helper wrappers for legacy helper import paths.
- `Utils/`
  - Forward-solver and data/plot utility modules.
- `UnifiedInverse/`
  - Modular inverse framework used by active scripts.
- `Old/`
  - Archived legacy scripts kept for reference.

## Primary Scripts

### Inference scripts

- `InferenceScripts/Infer_parameter_from_data.py`
  - Fully configurable unified inference interface.
- `InferenceScripts/Infer_DirichletBC_from_data.py`
  - Example problem using unified interface for `dirichlet_phi0` inference.
- `InferenceScripts/Infer_D_from_data.py`
  - Example problem using unified interface for diffusion inference.
- `InferenceScripts/Infer_RobinBC_from_data.py`
  - Example problem using unified interface for Robin `kappa` inference.
- `InferenceScripts/Infer_D_from_data_robin.py`
  - Example problem using unified interface for diffusion inference with Robin forward solver.
- `InferenceScripts/Infer_RobinKappa_from_flux_curve.py`
  - Infers Robin `kappa` by minimizing least-squares mismatch between target and
    simulated `phi_applied`-flux curves.

### Study/probe scripts

- `Studies/Probe_RobinFlux_steady_state.py`
  - Measures steps/time required to reach steady-state Robin flux for a `phi_applied` sweep.
- `Studies/Generate_RobinFlux_vs_phi0_data.py`
  - Generates synthetic experimental-style `phi_applied` vs steady-state flux data.
- `Studies/Test_RobinFlux_kappa_overlay.py`
  - Overlays steady-state flux curves for multiple Robin `kappa` pairs.
- `Studies/optimization_method_study.py`
  - Compares optimization methods across inverse tasks.
- `Studies/bfgs_lbfgsb_diffusion_failure_study.py`
  - Focused BFGS vs L-BFGS-B failure analysis.
- `Studies/forward_solver_D_stability_study.py`
  - Maps forward-solver convergence across diffusion pairs.

## Backward Compatibility

Legacy helper import paths are preserved as wrappers that call the unified engine:
- `Helpers/Infer_D_from_data_helpers.py`
- `Helpers/Infer_DirichletBC_from_data_helpers.py`
- `Helpers/Infer_RobinBC_from_data_helpers.py`

Original implementations are kept and marked with `(old)` in their filenames.

## Legacy Files (Not Used by Default)

Legacy standalone inverse scripts and helpers have been retained and renamed:
- `Old/Infer_D_from_data (old).py`
- `Old/Infer_DirichletBC_from_data (old).py`
- `Old/Infer_D_from_data_robin (old).py`
- `Old/Infer_RobinBC_from_data (old).py`
- `Old/Infer_D_from_data_helpers (old).py`
- `Old/Infer_DirichletBC_from_data_helpers (old).py`
- `Old/Infer_RobinBC_from_data_helpers (old).py`

## Quick Start

From `FireDrakeEnvCG/PNPInverse`:

```bash
python InferenceScripts/Infer_DirichletBC_from_data.py
```

Run the generic interface directly:

```bash
python InferenceScripts/Infer_parameter_from_data.py \
  --target dirichlet_phi0 \
  --true-value 1.0 \
  --initial-guess 10.0
```

## Requirements

This project depends on a working Firedrake install with adjoint support.

Core runtime dependencies:
- Firedrake
- `firedrake.adjoint` (Pyadjoint)
- `numpy`
- `matplotlib`
- `imageio` (for animation export in plotting utility)

Optional for report workflows:
- A LaTeX toolchain (`latexmk`, `pdflatex`) to build PDFs from generated `.tex`.

Recommended setup:
- Use official Firedrake installation instructions: https://www.firedrakeproject.org/install.html
- Or run inside the Firedrake Docker image.

## Study Workflows

### Experimental-style Robin flux workflow

1) Probe steady-state horizon with coarse `dt`:

```bash
python Studies/Probe_RobinFlux_steady_state.py
```

2) Generate synthetic `phi_applied` vs steady-state flux data:

```bash
python Studies/Generate_RobinFlux_vs_phi0_data.py
```

3) Infer `kappa` from that curve using least squares:

```bash
python InferenceScripts/Infer_RobinKappa_from_flux_curve.py
```

Outputs are written under:

```text
StudyResults/robin_flux_experiment/
```

### 1) Optimizer comparison across inverse tasks

```bash
python Studies/optimization_method_study.py
```

Default outputs (under `StudyResults/optimization_methods/`):
- `opt_method_study_results.csv`
- `opt_method_study_summary.csv`
- `opt_method_study_summary.md`

Generate figures + LaTeX report source:

```bash
python StudyResults/optimization_methods/generate_latex_report.py
```

Build PDF:

```bash
cd StudyResults/optimization_methods
latexmk -pdf opt_method_study_report.tex
```

### 2) BFGS vs L-BFGS-B diffusion failure study

```bash
python Studies/bfgs_lbfgsb_diffusion_failure_study.py
```

Default outputs (under `StudyResults/bfgs_lbfgsb_diffusion_failure_study/`):
- `bfgs_lbfgsb_diffusion_results.csv`
- `bfgs_lbfgsb_diffusion_summary.csv`
- `bfgs_lbfgsb_diffusion_failure_cases.csv`
- `bfgs_lbfgsb_diffusion_failure_analysis.md`

Generate figures + LaTeX report source:

```bash
python StudyResults/bfgs_lbfgsb_diffusion_failure_study/generate_latex_report.py
```

Build PDF:

```bash
cd StudyResults/bfgs_lbfgsb_diffusion_failure_study
latexmk -pdf bfgs_lbfgsb_diffusion_failure_report.tex
```

### 3) Forward solver D-stability map

```bash
python Studies/forward_solver_D_stability_study.py
```

Anisotropy-focused dense grid:

```bash
python Studies/forward_solver_D_stability_study.py --study-mode anisotropy_dense --anis-ratio-threshold 8.0
```

Default outputs:
- `StudyResults/forward_solver_D_stability*/forward_solver_d_stability_results.csv`
- `StudyResults/forward_solver_D_stability*/forward_solver_d_stability_map.png`
- `StudyResults/forward_solver_D_stability*/forward_solver_d_stability_report.tex`

## Solver Parameter Convention

Forward solver parameter list format remains:

```text
[n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params]
```

`UnifiedInverse/build_default_solver_params(...)` is the recommended way to construct this list.
