# PNPInverse

Inverse and forward experiments for a Poisson-Nernst-Planck (PNP) model using Firedrake and `firedrake.adjoint` (Pyadjoint).

This directory now has two complementary workflows:
- A **unified inverse interface** (`UnifiedInverse/`) for standard parameter inference from state data.
- An **experimental-style Robin workflow** where the measured signal is a `phi_applied` vs steady-state curve (flux or charge-weighted current-density proxy), used to infer Robin transfer coefficients `kappa`.

## Current Folder Layout

- `InferenceScripts/`
  - Active entry scripts.
- `Studies/`
  - Probes, tests, overlays, and method studies.
- `Helpers/`
  - Helper modules used by inference scripts.
- `Utils/`
  - Forward solver and Robin experiment utilities.
- `UnifiedInverse/`
  - Modular inverse framework.
- `Old/`
  - Archived legacy scripts kept for reference.
- `writeups/`
  - Weekly reports and report assets.

## Environment Setup

Use any environment where Firedrake + adjoint dependencies are installed.
For example:

```bash
conda activate <your_firedrake_env>
```

Then run from:

```bash
cd PNPInverse
```

Core dependencies:
- Firedrake
- `firedrake.adjoint`
- `numpy`
- `matplotlib`
- `scipy`
- `imageio`
- `Pillow` (for GIF export)

Optional for writeup builds:
- `latexmk`, `pdflatex`

## Unified Inverse Interface

The generic inverse engine is in `UnifiedInverse/`:
- `UnifiedInverse/solver_interface.py`
- `UnifiedInverse/parameter_targets.py`
- `UnifiedInverse/inference_runner.py`

Entry script:
- `InferenceScripts/Infer_parameter_from_data.py`

You can configure:
- forward solver module contract,
- parameter target,
- true value, noise level, and initial guess,
- optimizer options.

Backward-compatible wrappers are kept in:
- `Helpers/Infer_D_from_data_helpers.py`
- `Helpers/Infer_DirichletBC_from_data_helpers.py`
- `Helpers/Infer_RobinBC_from_data_helpers.py`

## Robin Flux-Curve Inference Experiment

### Physical observable and boundary flux

For species $i$, the Robin boundary condition used in this project is:

$$
J_i \cdot n = \kappa_i \left(c_i - c_{\infty,i}\right)
$$

Species flux through the Robin electrode boundary:

$$
F_i = \int_{\Gamma_{\mathrm{electrode}}} \kappa_i \left(c_i - c_{\infty,i}\right)\, ds
$$

Default scalar measured signal in the experiment:

$$
F_{\mathrm{obs}} = \sum_i F_i
$$

Implementation reference:
- `Utils/robin_flux_experiment.py` (`compute_species_flux_on_robin_boundary`, `observed_flux_from_species_flux`)

### Steady-state definition

At time step $n$, with boundary flux $F_i^{(n)}$:

$$
\Delta_i^{(n)} = \left|F_i^{(n)} - F_i^{(n-1)}\right|
$$

$$
\mathrm{rel}^{(n)} =
\max_i \frac{\Delta_i^{(n)}}{\max\left(\left|F_i^{(n)}\right|,\left|F_i^{(n-1)}\right|,\varepsilon_{\mathrm{abs}}\right)}
$$

$$
\mathrm{abs}^{(n)} = \max_i \Delta_i^{(n)}
$$

A step is marked steady if:

$$
\mathrm{rel}^{(n)} \le \varepsilon_{\mathrm{rel}}
\quad \text{or} \quad
\mathrm{abs}^{(n)} \le \varepsilon_{\mathrm{abs}}
$$

Steady state is declared after `consecutive_steps` steady steps in a row.

Implementation reference:
- `Utils/robin_flux_experiment.py` (`solve_to_steady_state_for_phi_applied`)

### Synthetic noise model

Noise is additive Gaussian with:

$$
\sigma = \left(\frac{p}{100}\right)\mathrm{RMS}\!\left(F_{\mathrm{clean}}\right)
$$

where $p$ is `noise_percent`.

This is RMS-scaled noise, not a strict pointwise $\pm p\%$ cap.

Implementation reference:
- `Utils/robin_flux_experiment.py` (`add_percent_noise`)

### Inverse objective and adjoint gradient

For sweep points $\phi_j$ with target flux $F_j^\star$:

$$
L_j(\kappa) = \frac{1}{2}\left(F_j(\kappa) - F_j^\star\right)^2
$$

$$
J(\kappa) = \sum_{j=1}^{m} L_j(\kappa)
$$

Per-point adjoint gradients are computed with Firedrake-adjoint and summed:

$$
\nabla J(\kappa) = \sum_{j \in \mathcal{C}} \nabla_{\kappa} L_j(\kappa)
$$

where $\mathcal{C}$ is the set of converged sweep points.

Optimization uses SciPy `minimize` (default `L-BFGS-B`) with analytic Jacobian.

Implementation reference:
- `InferenceScripts/Infer_RobinKappa_from_flux_curve.py`
- `Helpers/Infer_RobinKappa_from_flux_curve_helpers.py`

### Current-density proxy mode

The current-density branch fits a charge-weighted proxy observable:

$$
\sum_i z_i F_i
$$

using `observable_mode="charge_proxy_no_f"`.

Notes:
- Faraday scaling is intentionally omitted for now (proxy units are arbitrary).
- This keeps magnitudes comparable to flux-space studies while unit calibration is being finalized.

Entry script:
- `InferenceScripts/Infer_RobinKappa_from_current_density_curve.py`

### Forward-solve resilience

When a point solve diverges, recovery stages are applied:
1. Increase `snes_max_it`.
2. Try anisotropy-reduced `kappa`.
3. Relax `snes_atol`, `snes_rtol`, and `ksp_rtol` and vary line search.

This prevents immediate failure on difficult iterates and improves robustness.

### Parallel point solves (adjoint-safe)

To speed up each objective/gradient evaluation across the 15 sweep points,
point solves can run in parallel worker processes.

Important: this uses **processes**, not threads, so each worker has isolated
firedrake-adjoint tape state.

User-configurable options in `RobinFluxCurveInferenceRequest`:
- `parallel_point_solves_enabled`
- `parallel_point_workers`
- `parallel_point_min_points`
- `parallel_start_method` (`spawn` recommended)

Example (inside an inference script request block):

```python
parallel_point_solves_enabled=True,
parallel_point_workers=4,
parallel_point_min_points=4,
parallel_start_method="spawn",
```

Current status:
- Replay mode is temporarily disabled in the helper due validity issues.
- Process-parallel point solves are active and supported with serial fallback if
  worker initialization/execution fails.
- Latest no-noise current-density benchmark on this machine:
  serial `279.642 s` vs process-parallel (`4` workers) `111.948 s`
  (~`2.50x` speedup) with matching fitted `kappa`/loss.

## Active Scripts

### Inference entry scripts

- `InferenceScripts/Infer_parameter_from_data.py`
  - Unified interface entrypoint.
- `InferenceScripts/Infer_DirichletBC_from_data.py`
  - Dirichlet example.
- `InferenceScripts/Infer_D_from_data.py`
  - Diffusion example.
- `InferenceScripts/Infer_D_from_data_robin.py`
  - Diffusion inference with Robin forward solver.
- `InferenceScripts/Infer_RobinBC_from_data.py`
  - Robin `kappa` inference on state data.
- `InferenceScripts/Infer_RobinKappa_from_flux_curve.py`
  - Robin `kappa` inference from `phi_applied` vs steady-state flux curve.
- `InferenceScripts/Infer_RobinKappa_from_current_density_curve.py`
  - Robin `kappa` inference from `phi_applied` vs charge-weighted steady-state proxy.

### Studies/probes

- `Studies/Probe_RobinFlux_steady_state.py`
  - Steady-state horizon probe over `phi_applied`.
- `Studies/Generate_RobinFlux_vs_phi0_data.py`
  - Generates synthetic flux-curve data (script name has legacy `phi0`; workflow uses `phi_applied`).
- `Studies/Test_RobinFlux_kappa_overlay.py`
  - Overlays no-noise curves for arbitrary `kappa` combinations.
- `Studies/optimization_method_study.py`
- `Studies/bfgs_lbfgsb_diffusion_failure_study.py`
- `Studies/forward_solver_D_stability_study.py`
- `Studies/benchmark_current_density_replay_runtime.py`
  - Runtime benchmark driver (supports serial and process-parallel point solves).

## Quick Start Commands

### 1) Unified interface example

```bash
python InferenceScripts/Infer_DirichletBC_from_data.py
```

Or generic entrypoint:

```bash
python InferenceScripts/Infer_parameter_from_data.py \
  --target dirichlet_phi0 \
  --true-value 1.0 \
  --initial-guess 10.0
```

### 2) Robin flux experiment workflow

Probe steady-state behavior:

```bash
python Studies/Probe_RobinFlux_steady_state.py
```

Generate synthetic curve:

```bash
python Studies/Generate_RobinFlux_vs_phi0_data.py
```

Infer `kappa` from curve:

```bash
python InferenceScripts/Infer_RobinKappa_from_flux_curve.py
```

Infer `kappa` from current-density proxy curve:

```bash
python InferenceScripts/Infer_RobinKappa_from_current_density_curve.py
```

Main outputs are written to:

```text
StudyResults/robin_flux_experiment/
```

Typical files:
- `phi_applied_vs_steady_flux_synthetic.csv`
- `phi_applied_vs_steady_observable_fit.csv`
- `robin_kappa_gradient_optimization_history.csv`
- `robin_kappa_point_gradients.csv`
- `phi_applied_vs_steady_observable_fit.png`
- `robin_kappa_fit_convergence.gif`

Current-density workflow outputs are written under:

```text
StudyResults/robin_current_density_experiment/
```

Typical files:
- `phi_applied_vs_steady_current_density_synthetic.csv`
- `phi_applied_vs_steady_observable_fit.csv`
- `robin_kappa_gradient_optimization_history.csv`
- `robin_kappa_point_gradients.csv`
- `phi_applied_vs_steady_observable_fit.png`

### 3) Overlay arbitrary no-noise `kappa` curves

```bash
python Studies/Test_RobinFlux_kappa_overlay.py \
  --kappa-list "2,2;1,1;2,1;1,2;1,5;5,1" \
  --phi-min 0.0 --phi-max 0.04 --n-points 15 \
  --output-dir writeups/assets \
  --output-prefix robin_kappa_no_noise_overlay
```

## Solver Parameter Convention

Forward solver parameter list:

```text
[n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params]
```

Use:
- `UnifiedInverse/build_default_solver_params(...)`

to construct this consistently.

## Legacy and Archived Files

Old script implementations are retained under `Old/` and marked with `(old)` in filename where applicable. They are reference-only and not used by default workflows.
