# PNPInverse

Forward and inverse solvers for the Poisson-Nernst-Planck (PNP) equations with Butler-Volmer (BV) electrochemical boundary conditions, built on the [Firedrake](https://www.firedrakeproject.org/) finite element library and `firedrake.adjoint` (Pyadjoint).

The target application is modeling oxygen reduction at pH 4 (O2/H2O2 system), with current-voltage curves compared against experimental data from Mangan et al. (2025).

---

## Recent Highlights (through Apr 27, 2026)

### Surrogate-based inverse pipeline (v11 - v18)

A full surrogate-accelerated inverse workflow now sits beside the direct adjoint pipeline. The `Surrogate/` package provides five interchangeable surrogate families behind a common API (`fit`, `predict`, `predict_batch`, `training_bounds`):

| Family | Module | Notes |
|--------|--------|-------|
| RBF | `surrogate_model.py` | Thin-plate spline interpolant, log-`k0` transform |
| NN (ResNet-MLP) | `nn_model.py`, `nn_training.py` | 4-block ResNet on `[log10(k0_1), log10(k0_2), alpha_1, alpha_2]` -> 44 outputs (22 CD + 22 PC) |
| NN ensemble | `ensemble.py` | Mean wrapper over multi-seed NN runs |
| Gaussian Process | `gp_model.py` | 44 independent GPyTorch exact GPs (Matern-2.5, ARD); `predict_with_uncertainty`, autograd gradients via `predict_torch` |
| PCE | `pce_model.py` | ChaosPy Legendre PCE with hyperbolic truncation, Sobol sensitivity |
| POD-RBF | `pod_rbf_model.py` | POD basis on the I-V curve + RBF in coefficient space |

Around them: LHS / multi-region sampling (`sampling.py`), training-data generation (`training.py`), validation reports (`validation.py`), block-coordinate descent (`bcd.py`), multistart inference (`multistart.py`), cascade inference (`cascade.py`), and **ISMO** -- iterative surrogate-model optimization (`ismo.py`, Lye-Mishra-Ray 2020) with adaptive acquisition (`acquisition.py`) and incremental retraining (`ismo_retrain.py`, `ismo_pde_eval.py`).

Cross-model gradient and inverse-recovery benchmarks live in `StudyResults/gradient_benchmark/` and `StudyResults/inverse_benchmark/`; the multi-criteria surrogate ranking (CD/PC accuracy, gradient accuracy, recovery, runtime) is in `scripts/studies/surrogate_ranking_report.py`.

### Forward-solver hardening (`Forward/bv_solver/`)

`Forward/bv_solver.py` is now a sub-package. Major additions:

| Module | Role |
|--------|------|
| `forms.py`, `forms_log.py`, `forms_logc.py`, `forms_mixed_logc.py` | Variational forms in `c`, `log c`, mixed neutral/log spaces |
| `grid_charge_continuation.py` | Unified neutral-then-charged grid sweep (z=0 sweep once, then per-point z-ramp) |
| `hybrid_forward.py` | Per-voltage selection between z=0 (onset) and z=1 (cathodic transport-limited) |
| `stabilized_forward.py`, `stabilization.py` | ClO4-only artificial diffusion for z=1 convergence in the onset region |
| `robust_forward.py` | Wrapper layering continuation, line-search, and fallback strategies |
| `gummel_solver.py` | Decoupled Gummel iteration as a non-monolithic fallback |
| `observables.py` | Current density, peroxide current, and per-reaction rate post-processing |

These let the inverse pipeline run on physically realistic 3-species (O2, H2O2, H+ with Boltzmann ClO4-) and 4-species (adds explicit ClO4-) targets across the full V_RHE range used for inference.

### V&V report and literature provenance

- `writeups/vv_report/vv_report.pdf` -- six-surrogate verification & validation report with prediction-accuracy, gradient, and inverse-recovery figures (`generate_figures.py` regenerates from `StudyResults/`).
- `writeups/WeekOfApr27/literature_provenance.pdf` -- formal provenance audit of three numerical choices (PBNP hybrid; `u = ln c` log-density; log-rate BV), with corrected citations and a defensible novelty assessment.
- `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` -- revised solver writeup.
- `writeups/WeekOfMar4/v13_pipeline_paper.pdf` -- v13 pipeline paper.

### MMS verification (still passing)

The four MMS cases from Feb 25 continue to pass on the current solver:

| Case | Description | Script |
|------|-------------|--------|
| 1 | Single neutral species + 1 irreversible BV reaction | `mms_bv_convergence.py --case single` |
| 2 | Two neutral species + 2 BV reactions (O2 + H2O2) | `mms_bv_convergence.py --case two_species` |
| 3 | Two charged species + Poisson coupling + 1 BV reaction | `mms_bv_convergence.py --case charged` |
| 4 | Full 4-species system through the production solver pipeline | `mms_bv_4species.py` |
| 5 | 3-species `log c` Boltzmann variant | `mms_bv_3sp_logc_boltzmann.py` |

All cases achieve O(h^2) L2 / O(h) H1 convergence. Outputs: `StudyResults/mms_bv_convergence/`, `StudyResults/mms_bv_4species/`, `StudyResults/mms_3sp_logc_boltzmann/`. Derivation: `writeups/WeekOfFeb25/mms_butler_volmer.pdf`.

---

## Repository Structure

### Canonical packages (source of truth)

| Package | Contents |
|---------|----------|
| `Nondim/` | Physical constants, nondimensionalization scales (`NondimScales` dataclass), `build_model_scaling()`, and `compat.py` (dict wrappers + solver-option builder). |
| `Forward/` | Forward solvers: `bv_solver/` (sub-package, see above), `dirichlet_solver`, `robin_solver`, `steady_state/`; plus `params.py` (`SolverParams`), `noise.py`, `plotter.py`. |
| `Inverse/` | Modular inverse engine: `solver_interface`, `parameter_targets`, `inference_runner/` (sub-package: `config`, `data`, `objective`, `recovery`, `formatting`), `objectives` (pre-built objective factories). |
| `Surrogate/` | RBF / NN / NN-ensemble / GP / PCE / POD-RBF surrogate models, sampling, training, validation, multistart + BCD + cascade inference, and ISMO. |
| `FluxCurve/` | 9-module adjoint-gradient Robin-kappa inference pipeline: `config`, `results`, `recovery`, `observables`, `point_solve`, `curve_eval`, `replay`, `plot`, `run`. |

### Scripts

| Directory | Contents |
|-----------|----------|
| `scripts/bv/` | BV I-V curve generation (`bv_iv_curve.py`, `bv_iv_curve_charged.py`) |
| `scripts/verification/` | MMS convergence tests (`mms_bv_convergence.py`, `mms_bv_4species.py`, `mms_bv_3sp_logc_boltzmann.py`) and BV solver strategy tests |
| `scripts/Inference/` | Inverse-problem entry points: `Infer_BVMaster_charged_v13...v18`, `v17/v18_robust_inference.py`, 3-species and adjoint variants |
| `scripts/surrogate/` | Training (`train_gp.py`, `train_pce.py`, `train_nn_surrogate.py`, `build_surrogate.py`), overnight pipelines (`overnight_train_v11..v16.py`), ISMO drivers (`run_ismo.py`, `run_ismo_live.py`), multistart inference, validation |
| `scripts/studies/` | Gradient & inverse benchmarks, surrogate ranking, profile-likelihood, FIM ablations, k0_2 stratified analysis, hybrid-solver tests, voltage-grid extension studies |
| `scripts/inference/` | Earlier Robin-kappa / diffusivity / Dirichlet-BC entry points |

### Other directories

| Directory | Contents |
|-----------|----------|
| `StudyResults/` | All run outputs (CSV, PNG, JSON, GIF) organized by study. Key collections: `master_inference_v13..v17`, `v18_*`, `surrogate_v12/v15`, `ismo`, `inverse_benchmark`, `gradient_benchmark`, `surrogate_fidelity`. |
| `writeups/` | Weekly PDF reports (`WeekOfFeb16`, `WeekOfFeb25`, `WeekOfMar4`, `WeekOfApr27`), the V&V report (`vv_report/`), MMS derivation, and literature-provenance audit. |
| `docs/` | Reference documents: PNP formulations (TeX), Mangan2025 (PDF), parameters (xlsx), handoff notes, `noise_model_conventions.md`. |
| `tests/` | Pytest suite covering MMS, autograd gradients, multistart, ISMO, profile likelihood, surrogate fidelity, pipeline reproducibility, sensitivity visualization, and v13 verification. |
| `archive/` | Legacy code for reference only: `shims/` (old Utils/UnifiedInverse/Helpers), `old/` (pre-restructure scripts), `renders/`. |

---

## Environment

Requires a [Firedrake](https://www.firedrakeproject.org/) virtual environment (not conda). Core dependencies: Firedrake, `firedrake.adjoint`, NumPy, SciPy, Matplotlib, imageio, Pillow. Surrogates additionally use PyTorch + GPyTorch (NN, GP) and ChaosPy (PCE).

**Working directory for all scripts:** `PNPInverse/` (so package imports resolve correctly).

```bash
cd /path/to/PNPInverse
/path/to/venv-firedrake/bin/python scripts/bv/bv_iv_curve.py
```

---

## Scripts

### BV I-V curve generation (`scripts/bv/`)

| Script | Description |
|--------|-------------|
| `bv_iv_curve.py` | Neutral 2-species BV I-V sweep (O2/H2O2) with parameter study CLI |
| `bv_iv_curve_charged.py` | Full 4-species charged PNP-BV I-V sweep (O2, H2O2, H+, ClO4-) |

### Verification (`scripts/verification/`)

| Script | Description |
|--------|-------------|
| `mms_bv_convergence.py` | MMS convergence study: Cases 1-3 (single/two neutral, two charged) |
| `mms_bv_4species.py` | MMS convergence study: Case 4 (4-species production pipeline) |
| `mms_bv_3sp_logc_boltzmann.py` | MMS convergence study: 3-species `log c` Boltzmann variant |
| `test_bv_forward.py` | BV solver strategy tests |

### Inverse problem (`scripts/Inference/`)

| Script | Description |
|--------|-------------|
| `Infer_BVMaster_charged_v18.py` | Latest master inference script -- stabilized z=1 forward, full onset coverage |
| `v18_robust_inference.py` / `v17_robust_inference.py` | Adjoint L-BFGS-B with onset stabilization (ClO4 artificial diffusion) |
| `v18_3sp_inference.py` | 3-species Boltzmann inference (no artificial diffusion) |
| `v18_adjoint_inference.py` / `v18_adjoint_simple.py` | Adjoint-only entry points |
| `v18_fast_recovery.py` | Quick offset-recovery sanity check |
| `v17_nelder_mead.py` / `test_v17_recovery.py` | Derivative-free fallback and v17 recovery test |

### Surrogate workflow (`scripts/surrogate/`)

| Script | Description |
|--------|-------------|
| `generate_training_data.py` | LHS / multi-region training-data sweeps via the true PDE |
| `train_gp.py` / `train_pce.py` / `train_nn_surrogate.py` | Per-family training entry points |
| `build_surrogate.py` / `train_improved_surrogate.py` | RBF surrogate training |
| `overnight_train_v15.py` / `overnight_train_v16.py` | End-to-end overnight pipelines (training -> validation -> adjoint gradients -> inversion) |
| `compute_adjoint_gradients_v16.py` | Pre-compute adjoint gradients for surrogate-vs-true comparison |
| `multistart_inference.py` | Multistart surrogate inversion |
| `run_ismo.py` / `run_ismo_live.py` | Iterative surrogate-model optimization driver |
| `validate_surrogate.py` | Surrogate validation report (`StudyResults/surrogate_fidelity/`) |
| `pce_sensitivity_report.py` | Sobol sensitivity report from a PCE surrogate |

### Studies (`scripts/studies/`)

| Script | Description |
|--------|-------------|
| `surrogate_ranking_report.py` | Multi-criteria ranking across surrogate families |
| `gradient_benchmark.py` | Cross-model gradient accuracy and timing |
| `inverse_benchmark_all_models.py` | Cross-model parameter recovery under noise |
| `parameter_recovery_all_models.py` | Companion to `inverse_benchmark_all_models.py` |
| `k02_stratified_analysis.py` | k0_2 stratified error analysis |
| `profile_likelihood_pde.py` / `profile_likelihood_study.py` | Profile-likelihood diagnostics on the PDE forward and on surrogates |
| `v23_*.py` | Anchored-Tafel LSQ inverse, multi-experiment FIM, negative-V FIM ablation |
| `v19_*.py`, `v18_*.py` | Voltage-continuation, FIM, log-rate, log-c diagnostics |
| `optimization_method_study.py` | Compare BFGS / L-BFGS-B / CG / SLSQP / TNC / Newton-CG |
| `forward_solver_D_stability_study.py` | Forward solver D-stability map |
| `training_data_audit.py` | Training-data audit |

(Older Robin-kappa / diffusivity / Dirichlet-BC entry points remain in `scripts/inference/`.)

---

## Running Tests

### Pytest suite

```bash
pytest tests/
```

Covers MMS convergence, autograd-gradient checks, multistart, ISMO retraining, profile likelihood, surrogate fidelity, and pipeline reproducibility.

### MMS verification (recommended first check after any solver changes)

```bash
# All 3 cases (single neutral, two neutral, two charged)
python scripts/verification/mms_bv_convergence.py --case all

# 4-species production pipeline test
python scripts/verification/mms_bv_4species.py

# 3-species log-c Boltzmann variant
python scripts/verification/mms_bv_3sp_logc_boltzmann.py
```

Output: convergence tables to stdout; plots and summary files to `StudyResults/mms_bv_convergence/`, `StudyResults/mms_bv_4species/`, and `StudyResults/mms_3sp_logc_boltzmann/`.

### I-V curve generation

```bash
# Neutral 2-species (fast)
python scripts/bv/bv_iv_curve.py

# Full 4-species charged (slower, requires graded mesh)
python scripts/bv/bv_iv_curve_charged.py

# Parameter studies
python scripts/bv/bv_iv_curve_charged.py --l-ref 6.5e-5 --Ny-mesh 300 --beta 3.0
```

### End-to-end inverse pipelines

```bash
# Direct adjoint inversion (latest)
python scripts/Inference/v18_robust_inference.py

# Surrogate-based: build, validate, invert
python scripts/surrogate/generate_training_data.py
python scripts/surrogate/train_gp.py
python scripts/surrogate/validate_surrogate.py
python scripts/surrogate/multistart_inference.py

# ISMO loop (true PDE in the loop, surrogate retrained each iteration)
python scripts/surrogate/run_ismo.py
```

---

## Butler-Volmer Forward Solver

`Forward/bv_solver/` solves the nondimensional PNP system with multi-reaction Butler-Volmer electrode boundary conditions. The package supports both per-species (legacy) and multi-reaction configurations, plus three Galerkin variable choices (`c`, `log c`, mixed) and four convergence wrappers (continuation, hybrid z=0/z=1, stabilized, robust).

### Nondimensionalization

| Quantity | Scale | Dimensionless form |
|----------|-------|--------------------|
| Concentration | c_ref = C_bulk | c_hat = c / c_ref |
| Potential | V_T = RT/F | phi_hat = phi / V_T |
| Length | L_ref | x_hat = x / L_ref |
| Time | L_ref^2 / D_ref | t_hat = t * D_ref / L_ref^2 |
| Rate constant | D_ref / L_ref | k0_hat = k0 * L_ref / D_ref |
| Current density | n * F * D_ref * c_ref / L_ref | I = J_hat * I_scale |

### Multi-reaction BV configuration

```python
"bv_bc": {
    "reactions": [
        {
            "k0": 2.4e-8,           # m/s
            "alpha": 0.627,
            "cathodic_species": 0,   # O2 consumed
            "anodic_species": 1,     # H2O2 produced
            "c_ref": 1.0,           # nondim reference for anodic term
            "stoichiometry": [-1, +1, -2, 0],
            "n_electrons": 2,
            "cathodic_conc_factors": [{"species": 2, "c_ref": 0.1, "power": 2}],
        },
        # ... additional reactions
    ],
    "electrode_marker": 3,
    "concentration_marker": 4,
    "ground_marker": 4,
}
```

Each reaction j contributes to the weak form:

```
R_j = k0_j * [c_cat * prod(conc_factors) * exp(-alpha_j * eta_hat) - c_ref_j * exp((1-alpha_j) * eta_hat)]
F_res -= s_ij * R_j * v_i * ds(electrode)
```

### Convergence strategies

The base solver layers seven strategies for robustness at large overpotentials (|eta_hat| up to ~46):

1. **Voltage continuation** -- uniform steps from eta=0; each step warm-starts from the previous converged state
2. **Inner time-stepping** -- BDF-1 pseudo-transient continuation until relative change < 1e-5
3. **Exponent clipping** -- BV exponent clamped to +/-50 via UFL `min_value`/`max_value`
4. **Concentration floor** -- `max(c_surf, 1e-12)` removes the c=0 singularity
5. **`use_eta_in_bv`** -- uses the Dirichlet constant `phi_applied_func` instead of interior phi field (exact for z=0)
6. **l2 linesearch, lambda_max=0.5** -- prevents Newton from driving c negative
7. **Direct LU (MUMPS)** -- eliminates Krylov stagnation risk; `mat_mumps_icntl_8: 77` for auto-scaling

On top of those, `grid_charge_continuation.py` (z=0 sweep then per-point z-ramp), `hybrid_forward.py` (per-voltage z=0/z=1 selection), `stabilized_forward.py` (ClO4-only artificial diffusion), and `gummel_solver.py` (decoupled fallback) handle the regimes where monolithic Newton with z=1 alone stalls.

---

## Robin Flux-Curve Inference

Infers Robin transfer coefficients kappa from a phi_applied vs. steady-state flux curve using adjoint gradients and L-BFGS-B.

### Key modules

| Module | Role |
|--------|------|
| `FluxCurve/config.py` | `RobinFluxCurveInferenceRequest`, `ForwardRecoveryConfig` |
| `FluxCurve/run.py` | `run_robin_kappa_flux_curve_inference` -- top-level entry point |
| `FluxCurve/point_solve.py` | Per-voltage-point adjoint solve (parallel-safe) |
| `FluxCurve/replay.py` | Replay-mode curve evaluation for fast objective re-evaluation |
| `Forward/steady_state/` | `SteadyStateConfig`, `solve_to_steady_state_for_phi_applied` (BV / Robin variants) |

### Quick start

```bash
# Flux curve inference
python scripts/inference/Infer_RobinKappa_from_flux_curve.py

# Current-density proxy inference
python scripts/inference/Infer_RobinKappa_from_current_density_curve.py
```

Outputs go to `StudyResults/robin_flux_experiment/` and `StudyResults/robin_current_density_experiment/`.

Supports process-parallel point solves (4-worker spawn mode gives ~2.5x speedup).

---

## SolverParams

`Forward/params.py` defines `SolverParams`, a `list` subclass that adds named attribute access while remaining fully backward-compatible with all index/unpack-based forward solver code.

```python
from Forward.params import SolverParams

sp = SolverParams.from_list([
    n_species, order, dt, t_end,
    z_vals, D_vals, a_vals,
    phi_applied, c0_vals, phi0,
    params_dict,
])

# Named access:
sp.D_vals, sp.phi_applied, sp.solver_options

# Index access (unchanged):
sp[5], sp[7], sp[10]
```

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `n_species` | int | Number of ionic species |
| 1 | `order` | int | FE polynomial order |
| 2 | `dt` | float | Time step |
| 3 | `t_end` | float | Final time |
| 4 | `z_vals` | list | Per-species charge numbers |
| 5 | `D_vals` | list | Per-species diffusivities |
| 6 | `a_vals` | list | Per-species steric parameters |
| 7 | `phi_applied` | float | Applied boundary voltage |
| 8 | `c0_vals` | list | Initial/bulk concentrations |
| 9 | `phi0` | float | Reference potential |
| 10 | `solver_options` | dict | PETSc/SNES/BV/nondim options |

Build with `Inverse.build_default_solver_params(...)` or `SolverParams.from_list([...])`.

---

## Nondimensionalization Package

`Nondim/` is the single source of truth for all physical constants and scaling logic.

```python
from Nondim import build_physical_scales, build_model_scaling, NondimScales
from Nondim.constants import FARADAY_CONSTANT, GAS_CONSTANT, DEFAULT_TEMPERATURE_K
```

`NondimScales` computes: reference diffusivity (geometric mean), thermal voltage, Debye length, time/flux/current-density scales. `build_model_scaling()` handles both dimensional and dimensionless modes and is called internally by all three forward solvers.

---

## Writeups

| Document | Location |
|----------|----------|
| V&V report (six surrogate models) | `writeups/vv_report/vv_report.pdf` |
| Literature provenance audit | `writeups/WeekOfApr27/literature_provenance.pdf` |
| Revised solver writeup (Apr 27, 2026) | `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf` |
| v13 pipeline paper | `writeups/WeekOfMar4/v13_pipeline_paper.pdf` |
| Weekly report (Mar 4, 2026) | `writeups/WeekOfMar4/week_of_march_4_2026.pdf` |
| Weekly report (Feb 25, 2026) | `writeups/WeekOfFeb25/week_of_february_25_2026.pdf` |
| MMS derivation for PNP-BV | `writeups/WeekOfFeb25/mms_butler_volmer.pdf` |
| Weekly report (Feb 16, 2026) | `writeups/WeekOfFeb16/` |
