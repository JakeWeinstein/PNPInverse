# PNPInverse

Inverse and forward experiments for a Poisson-Nernst-Planck (PNP) model using Firedrake and `firedrake.adjoint` (Pyadjoint).

This repository contains:
- A forward PNP solve on a 2D unit-square mesh.
- Inverse parameter estimation for:
  - diffusion coefficients `D` (`infer_D`)
  - Dirichlet boundary value for electric potential `phi0` (`infer_phi0`)
- Reproducible benchmark/stability studies and generated reports.

## What Is In This Repo

- `Utils/forsolve.py`: core forward model setup and nonlinear solve loop.
- `Utils/generate_noisy_data.py`: synthetic data generator (clean + noisy fields).
- `Helpers/Infer_D_from_data_helpers.py`: objective and controls for diffusion inference.
- `Helpers/Infer_DirichletBC_from_data_helpers.py`: objective and control for `phi0` inference.
- `Infer_D_from_data.py`: simple end-to-end diffusion inference example.
- `Infer_DirichletBC_from_data.py`: simple end-to-end Dirichlet-BC inference example.
- `optimization_method_study.py`: benchmark multiple optimizers for `infer_D` and `infer_phi0`.
- `bfgs_lbfgsb_diffusion_failure_study.py`: focused failure analysis of BFGS vs L-BFGS-B for `infer_D`.
- `forward_solver_D_stability_study.py`: forward-solver convergence map over `(D0, D1)` grids.
- `StudyResults/`: generated CSV/Markdown/Tex/PDF reports and figures.
- `Renders/`: sample visualization outputs.

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

Recommended way to get started:
- Use the official Firedrake installation instructions: https://www.firedrakeproject.org/install.html
- Or run inside the Firedrake Docker image for a consistent environment.

## Quick Start

From repo root run the basic inverse examples:

```bash
python Infer_D_from_data.py
python Infer_DirichletBC_from_data.py
```

Notes:
- These scripts generate both clean and noisy synthetic data.
- The current examples optimize against clean targets by default; switch to noisy vectors in the script if you want noisy-data inversion.

## Study Workflows

### 1) Optimizer comparison across inverse tasks

Runs a sweep over methods/noise/seeds for both `infer_D` and `infer_phi0`.

```bash
python optimization_method_study.py
```

Default outputs (under `StudyResults/optimization_methods/`):
- `opt_method_study_results.csv`
- `opt_method_study_summary.csv`
- `opt_method_study_summary.md`

Generate figures + LaTeX report source:

```bash
python StudyResults/optimization_methods/generate_latex_report.py
```

Optionally build PDF:

```bash
cd StudyResults/optimization_methods
latexmk -pdf opt_method_study_report.tex
```

### 2) BFGS vs L-BFGS-B diffusion failure study

```bash
python bfgs_lbfgsb_diffusion_failure_study.py
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

Optionally build PDF:

```bash
cd StudyResults/bfgs_lbfgsb_diffusion_failure_study
latexmk -pdf bfgs_lbfgsb_diffusion_failure_report.tex
```

### 3) Forward solver D-stability map

Default grid:

```bash
python forward_solver_D_stability_study.py
```

Anisotropy-focused dense grid:

```bash
python forward_solver_D_stability_study.py --study-mode anisotropy_dense --anis-ratio-threshold 8.0
```

Default outputs:
- `StudyResults/forward_solver_D_stability*/forward_solver_d_stability_results.csv`
- `StudyResults/forward_solver_D_stability*/forward_solver_d_stability_map.png`
- `StudyResults/forward_solver_D_stability*/forward_solver_d_stability_report.tex`

Optionally build PDF from the generated tex report.

## Model/Implementation Notes

- Solver parameter convention is a list:
  `[n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params]`
- Diffusion parameters are optimized in log-space (`logD`) and exponentiated, enforcing `D > 0`.
- Current study setup is mostly 2-species (`z = [1, -1]`) with first-order CG elements and a `32 x 32` mesh.
- Boundary conditions currently applied in `Utils/forsolve.py`:
  - `phi`: Dirichlet BC on boundary id `1`
  - concentrations `c_i`: Dirichlet BC on boundary id `3`
- `phi_applied` and `a_vals` are present in parameter lists for compatibility, but are not active terms in the current weak form.

