# PNPInverse

Forward and inverse solvers for the Poisson-Nernst-Planck (PNP) equations with Butler-Volmer (BV) electrochemical boundary conditions, built on the [Firedrake](https://www.firedrakeproject.org/) finite element library and `firedrake.adjoint` (Pyadjoint).

The target application is modeling oxygen reduction at pH 4 (O2/H2O2 system), with current-voltage curves compared against experimental data from Mangan et al. (2025).

---

## Recent Highlights (Week of Feb 25, 2026)

### MMS Verification (4 test cases, all passing)

A rigorous Method of Manufactured Solutions study proves O(h^2) L2 convergence and O(h) H1 convergence for the full PNP-BV solver across four cases of increasing complexity:

| Case | Description | Script |
|------|-------------|--------|
| 1 | Single neutral species + 1 irreversible BV reaction | `mms_bv_convergence.py --case single` |
| 2 | Two neutral species + 2 BV reactions (O2 + H2O2) | `mms_bv_convergence.py --case two_species` |
| 3 | Two charged species + Poisson coupling + 1 BV reaction | `mms_bv_convergence.py --case charged` |
| 4 | Full 4-species system through the production solver pipeline | `mms_bv_4species.py` |

Case 4 exercises the entire production code path (`build_context` / `build_forms`) including `cathodic_conc_factors`, stoichiometry magnitude |s|=2, and mixed neutral/charged species in the same `MixedFunctionSpace`. All 10 field components (4 concentrations + potential, L2 and H1) achieve their expected convergence rates. Results and convergence plots are in `StudyResults/mms_bv_convergence/` and `StudyResults/mms_bv_4species/`.

The MMS derivation, including the boundary correction source term for BV flux BCs, is documented in `writeups/WeekOfFeb25/mms_butler_volmer.pdf`.

### 4-Species Charged BV Solver

`Forward/bv_solver.py` now supports:
- **Multi-reaction Butler-Volmer kinetics** via a `"reactions"` configuration, enabling coupled chemistry (e.g., O2 -> H2O2 and H2O2 -> H2O sharing the H2O2 species)
- **`cathodic_conc_factors`** for concentration-dependent BV rates (e.g., (c_H+/c_ref)^2)
- **Voltage continuation** with pseudo-transient inner time-stepping for robust convergence at large overpotentials
- **Graded meshes** (`make_graded_rectangle_mesh`) with power-law cell distribution concentrating resolution in the Debye layer
- **Log-diffusivity parameterization** for inference compatibility
- **Full Poisson coupling** for charged species with (lambda_D/L)^2 ~ 10^-8

### 29-Run Convergence & Parameter Study

A systematic study (`StudyResults/convergence_study_summary.txt`) spanning 29 runs of the 4-species charged solver demonstrates:
- **Mesh convergence**: Converged to 5+ significant figures at Ny=50 with beta=3.0 grading; Ny=200 recommended for production
- **L_ref controls I_lim**: Limiting current scales as 1/L_ref; L_ref=65 um matches experimental I_lim ~ -0.27 mA/cm^2
- **k0 shifts V_1/2 only**: 5 decades of k0 shift V_1/2 by ~120 mV total; I_lim is transport-controlled
- **Solver robustness**: Converges on meshes up to Ny=1000 (h_min ~ 0.3 nm) and beta=4.0 (aspect ratio ~2000:1)

---

## Repository Structure

### Canonical packages (source of truth)

| Package | Contents |
|---------|----------|
| `Nondim/` | Physical constants, nondimensionalization scales (`NondimScales` dataclass), `build_model_scaling()`, and `compat.py` (dict wrappers + solver-option builder). |
| `Forward/` | All forward solvers: `bv_solver`, `dirichlet_solver`, `robin_solver`, `steady_state`; plus `params.py` (`SolverParams`), `noise.py`, `plotter.py`. |
| `Inverse/` | Modular inverse engine: `solver_interface`, `parameter_targets`, `inference_runner`, `objectives` (pre-built objective factories). |
| `FluxCurve/` | 9-module adjoint-gradient Robin-kappa inference pipeline: `config`, `results`, `recovery`, `observables`, `point_solve`, `curve_eval`, `replay`, `plot`, `run`. |

### Scripts

| Directory | Contents |
|-----------|----------|
| `scripts/bv/` | BV I-V curve generation (`bv_iv_curve.py`, `bv_iv_curve_charged.py`) |
| `scripts/verification/` | MMS convergence tests + solver strategy tests |
| `scripts/inference/` | Parameter inference entry points (Robin kappa, diffusivity, Dirichlet BC) |
| `scripts/studies/` | Benchmarks & parameter sweeps (9 study scripts) |

### Other directories

| Directory | Contents |
|-----------|----------|
| `StudyResults/` | All run outputs (CSV, PNG, GIF) organized by study. |
| `writeups/` | Weekly PDF reports, LaTeX source, and MMS derivation document. |
| `docs/` | Reference documents: PNP formulations (TeX), Mangan2025 (PDF), parameters (xlsx). |
| `archive/` | Legacy code for reference only: `shims/` (old Utils/UnifiedInverse/Helpers), `old/` (pre-restructure scripts), `renders/`. |

---

## Environment

Requires a [Firedrake](https://www.firedrakeproject.org/) virtual environment (not conda). Core dependencies: Firedrake, `firedrake.adjoint`, NumPy, Matplotlib, SciPy, imageio, Pillow.

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
| `test_bv_forward.py` | BV solver strategy tests (7 strategies: A-G) |

### Inference (`scripts/inference/`)

| Script | Description |
|--------|-------------|
| `Infer_RobinKappa_from_flux_curve.py` | Robin kappa from phi vs. flux curve (adjoint L-BFGS-B) |
| `Infer_RobinKappa_from_current_density_curve.py` | Robin kappa from phi vs. current-density proxy |
| `Infer_parameter_from_data.py` | Unified inverse entrypoint (CLI) |
| `Infer_DirichletBC_from_data.py` | Dirichlet BC inference example |
| `Infer_D_from_data.py` | Diffusivity inference example |
| `Infer_D_from_data_robin.py` | Diffusivity with Robin forward solver |
| `Infer_RobinBC_from_data.py` | Robin kappa inference from state data |

### Studies (`scripts/studies/`)

| Script | Description |
|--------|-------------|
| `optimization_method_study.py` | Compare BFGS/L-BFGS-B/CG/SLSQP/TNC/Newton-CG |
| `bfgs_lbfgsb_diffusion_failure_study.py` | Detailed BFGS vs L-BFGS-B failure analysis |
| `forward_solver_D_stability_study.py` | Forward solver D-stability map |
| `Generate_RobinFlux_vs_phi0_data.py` | Generate phi vs. flux data |
| `Test_RobinFlux_kappa_overlay.py` | Overlay kappa curves |
| `Probe_RobinFlux_steady_state.py` | Steady-state horizon probing |
| `benchmark_robin_kappa_ic_runtime.py` | IC choice runtime benchmark |
| `benchmark_current_density_replay_runtime.py` | Replay mode runtime benchmark |
| `Generate_RobinCurrentDensity_report_plots.py` | Report-ready current-density plots |

---

## Running Tests

### MMS verification (recommended first check after any solver changes)

```bash
# All 3 cases (single neutral, two neutral, two charged)
python scripts/verification/mms_bv_convergence.py --case all

# 4-species production pipeline test
python scripts/verification/mms_bv_4species.py

# Custom mesh refinement sequence
python scripts/verification/mms_bv_convergence.py --case single --Nvals 8 16 32 64 128
```

Output: convergence tables printed to stdout; plots and summary files saved to `StudyResults/mms_bv_convergence/` and `StudyResults/mms_bv_4species/`.

### BV solver strategy tests

```bash
# Run all 7 strategies (A through G)
python scripts/verification/test_bv_forward.py --strategy all

# Run a specific strategy
python scripts/verification/test_bv_forward.py --strategy A
```

### I-V curve generation

```bash
# Neutral 2-species (fast)
python scripts/bv/bv_iv_curve.py

# Full 4-species charged (slower, requires graded mesh)
python scripts/bv/bv_iv_curve_charged.py

# Parameter studies
python scripts/bv/bv_iv_curve_charged.py --l-ref 6.5e-5 --Ny-mesh 300 --beta 3.0
```

---

## Butler-Volmer Forward Solver

`Forward/bv_solver.py` solves the nondimensional PNP system with multi-reaction Butler-Volmer electrode boundary conditions.

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

Seven strategies are layered for robustness at large overpotentials (|eta_hat| up to ~46):

1. **Voltage continuation** -- uniform steps from eta=0; each step warm-starts from the previous converged state
2. **Inner time-stepping** -- BDF-1 pseudo-transient continuation until relative change < 1e-5
3. **Exponent clipping** -- BV exponent clamped to +/-50 via UFL `min_value`/`max_value`
4. **Concentration floor** -- `max(c_surf, 1e-12)` removes the c=0 singularity
5. **`use_eta_in_bv`** -- uses the Dirichlet constant `phi_applied_func` instead of interior phi field (exact for z=0)
6. **l2 linesearch, lambda_max=0.5** -- prevents Newton from driving c negative
7. **Direct LU (MUMPS)** -- eliminates Krylov stagnation risk; `mat_mumps_icntl_8: 77` for auto-scaling

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
| `Forward/steady_state.py` | `SteadyStateConfig`, `solve_to_steady_state_for_phi_applied` |

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
| Weekly report (Feb 25, 2026) | `writeups/WeekOfFeb25/week_of_february_25_2026.pdf` |
| MMS derivation for PNP-BV | `writeups/WeekOfFeb25/mms_butler_volmer.pdf` |
| Weekly report (Feb 16, 2026) | `writeups/WeekOfFeb16/` |
