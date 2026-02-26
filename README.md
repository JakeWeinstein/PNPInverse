# PNPInverse

Inverse and forward experiments for the Poisson-Nernst-Planck (PNP) equations using Firedrake and `firedrake.adjoint` (Pyadjoint).

Two main workflows:
- **Forward I–V sweeps** (`InferenceScripts/bv_iv_curve.py`): Butler-Volmer kinetics, voltage continuation, parameter studies against experimental data.
- **Adjoint inverse inference** (`InferenceScripts/Infer_RobinKappa_from_flux_curve.py`): fit Robin transfer coefficients from a φ-applied vs. steady-state flux curve using adjoint gradients and L-BFGS-B.

---

## Folder Layout

### Canonical packages (source of truth)

| Package | Contents |
|---|---|
| `Nondim/` | Single source of truth for physical constants, nondimensionalization scales (`NondimScales`), and `build_model_scaling()`. |
| `Forward/` | All forward solvers: `bv_solver`, `dirichlet_solver`, `robin_solver`, `steady_state`; plus `params.py` (`SolverParams`), `noise.py`, `plotter.py`. |
| `Inverse/` | Modular inverse engine: `solver_interface`, `parameter_targets`, `inference_runner` (`build_default_solver_params`). |
| `FluxCurve/` | 9-module split of the Robin-kappa flux-curve adjoint inference pipeline: `config`, `results`, `recovery`, `observables`, `point_solve`, `curve_eval`, `replay`, `plot`, `run`. |

### Backward-compatibility shims (re-export only)

| Directory | Points to |
|---|---|
| `Utils/` | `Forward.*` |
| `UnifiedInverse/` | `Inverse.*` |
| `Helpers/Infer_RobinKappa_from_flux_curve_helpers.py` | `FluxCurve.*` |

All existing `InferenceScripts/` and `Studies/` scripts continue to work through these shims.

### Other directories

| Directory | Contents |
|---|---|
| `InferenceScripts/` | Active entry scripts. |
| `Studies/` | Probes, overlays, method studies. |
| `StudyResults/` | All run outputs (CSV, PNG, GIF). |
| `writeups/` | Weekly PDF reports and their LaTeX source. |
| `Old/` | Archived legacy scripts (reference only). |

---

## Environment

**Python executable:**
```bash
/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python
```

**Working directory for all scripts:** `PNPInverse/` (so package imports resolve correctly).

```bash
cd /path/to/PNPInverse
/path/to/venv-firedrake/bin/python InferenceScripts/bv_iv_curve.py
```

Core dependencies: Firedrake, `firedrake.adjoint`, `numpy`, `matplotlib`, `scipy`, `imageio`, `Pillow`.

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

# Named access (new):
sp.D_vals, sp.phi_applied, sp.solver_options

# Index access (unchanged):
sp[5], sp[7], sp[10]
```

Use `Inverse.build_default_solver_params(...)` to construct a correctly populated `SolverParams` for inverse problems.

---

## Butler-Volmer Forward Solver

`Forward/bv_solver.py` solves the nondimensional PNP system with a Butler-Volmer electrode boundary condition.

### Nondimensionalization

| Quantity | Scale | Dimensionless form |
|---|---|---|
| Concentration | $c_\mathrm{ref} = C_\mathrm{bulk}$ | $\hat c = c / c_\mathrm{ref}$ |
| Potential | $V_T = RT/F$ | $\hat\phi = \phi / V_T$ |
| Length | $L_\mathrm{ref}$ | $\hat x = x / L_\mathrm{ref}$ |
| Time | $L_\mathrm{ref}^2 / D_\mathrm{ref}$ | $\hat t = t D_\mathrm{ref} / L_\mathrm{ref}^2$ |
| Rate constant | $D_\mathrm{ref} / L_\mathrm{ref}$ | $\hat k_0 = k_0 L_\mathrm{ref} / D_\mathrm{ref}$ |
| Current density | $n F D_\mathrm{ref} c_\mathrm{ref} / L_\mathrm{ref}$ | $I = \hat J \times I_\mathrm{scale}$ |

For a neutral species ($z=0$) the governing equations reduce to diffusion + Bikerman steric + Butler-Volmer BC:

$$
\frac{\partial\hat c}{\partial\hat t} = \hat\nabla\cdot\!\left(\hat D\,\hat\nabla\hat c + \hat D\,\hat c\,\hat\nabla\ln(1-\hat a\hat c)\right)
$$

$$
-\hat D\,\hat\nabla\hat c\cdot\hat n = s\,\hat k_0\bigl[\hat c_\mathrm{surf}\,e^{-\alpha\hat\eta} - \hat c_\mathrm{ref}\,e^{(1-\alpha)\hat\eta}\bigr] \quad \text{on } \Gamma_\mathrm{el}
$$

### Convergence strategies

Seven strategies are layered to handle $|\hat\eta| \sim 46$ without failure:

1. **Voltage continuation** — 100 uniform steps from $\hat\eta=0$; each step warm-starts from the previous converged state.
2. **Inner time-stepping** — BDF-1 pseudo-transient continuation until $\|U^{n+1}-U^n\|_{L^2}/\|U^n\|_{L^2}<10^{-5}$ (up to 50 steps).
3. **Exponent clipping** — BV exponent clamped to $\pm50$ via UFL `min_value`/`max_value`.
4. **Concentration floor** — `max(c_surf, 1e-12)` removes the $c=0$ singularity from the cathodic term.
5. **`use_eta_in_bv`** — uses the Dirichlet constant `phi_applied_func` instead of the interior $\hat\phi$ field, zeroing the $\partial(\mathrm{BV})/\partial\hat\phi$ Jacobian block (exact for $z=0$).
6. **$l_2$ linesearch, $\lambda_\mathrm{max}=0.5$** — prevents Newton from driving $\hat c$ negative.
7. **Direct LU (MUMPS)** — eliminates Krylov stagnation risk for the small 2D mesh.

Convergence fails (and is not expected) when: $\hat k_0 \gg 1$ (unresolved boundary layer), $\alpha \lesssim 0.43$ (no transport-limited plateau), $\alpha \gtrsim 0.8$ with a 2D corner (singularity amplification), or $\Delta\hat t \gg 1$ (BDF-1 regularization negligible near transport limit).

---

## I–V Curve Study (`bv_iv_curve.py`)

`InferenceScripts/bv_iv_curve.py` sweeps applied voltage from the equilibrium potential down to $-0.5$ V vs. RHE and computes the O₂ reduction current density, targeting the Mangan2025 experimental curve.

### Default physical parameters (O₂/H₂O₂, pH 4)

| Parameter | Value |
|---|---|
| $D_{\mathrm{O}_2}$ | $2.10\times10^{-9}$ m²/s |
| $C_\mathrm{bulk}$ | $0.5$ mol/m³ |
| $k_0$ | $2.4\times10^{-8}$ m/s |
| $\alpha$ | $0.627$ (Tafel slope $\approx94$ mV/dec) |
| $L_\mathrm{ref}$ | $100\,\mu$m (default) |
| $E_\mathrm{eq}$ | $0.695$ V vs. RHE |
| $\hat a_{\mathrm{O}_2}$ | $0.01$ (Bikerman steric) |

### CLI usage

```bash
# Default run
python InferenceScripts/bv_iv_curve.py

# Parameter sensitivity study
python InferenceScripts/bv_iv_curve.py --l-ref 3e-4   # 300 µm → I_lim ≈ -0.33 mA/cm²
python InferenceScripts/bv_iv_curve.py --k0 1e-7       # shifts V_1/2 only
python InferenceScripts/bv_iv_curve.py --alpha 0.5

# Control sweep resolution and steady-state tolerance
python InferenceScripts/bv_iv_curve.py --steps 100 --max-ss-steps 50
```

Outputs (CSV + PNG) go to `StudyResults/bv_iv_curve/` for default runs, or `StudyResults/bv_iv_study/<tag>/` for parameter studies.

### Best-fit parameters (vs. Mangan2025)

| Parameter | Experimental target | Best fit |
|---|---|---|
| $I_\mathrm{lim}$ | $\approx-0.35$ mA/cm² | $-0.332$ mA/cm² ($L_\mathrm{ref}=300\,\mu$m) |
| $V_{1/2}$ | $0.20$–$0.25$ V vs. RHE | $0.233$ V |

Parameter roles: $L_\mathrm{ref}$ controls $I_\mathrm{lim}$ (scales as $1/L_\mathrm{ref}$); $k_0$ shifts $V_{1/2}$ only; $\alpha$ controls whether a transport limit is reached within the sweep range.

---

## Robin Flux-Curve Inference

Infers Robin transfer coefficients $\kappa$ from a $\phi_\mathrm{applied}$ vs. steady-state flux curve using adjoint gradients.

### Robin BC and observable

$$
J_i \cdot n = \kappa_i(c_i - c_{\infty,i}) \qquad F_\mathrm{obs} = \sum_i \int_{\Gamma_\mathrm{el}} \kappa_i(c_i - c_{\infty,i})\,ds
$$

### Adjoint objective

$$
J(\kappa) = \sum_{j=1}^{m} \tfrac{1}{2}(F_j(\kappa) - F_j^\star)^2
\qquad
\nabla J(\kappa) = \sum_{j \in \mathcal{C}} \nabla_\kappa L_j(\kappa)
$$

Optimized with SciPy `L-BFGS-B` using analytic adjoint Jacobian from `firedrake.adjoint`.

### Key modules

| Module | Role |
|---|---|
| `FluxCurve/config.py` | `RobinFluxCurveInferenceRequest`, `ForwardRecoveryConfig` |
| `FluxCurve/run.py` | `run_robin_kappa_flux_curve_inference` — top-level entry point |
| `FluxCurve/point_solve.py` | Per-voltage-point adjoint solve (parallel-safe) |
| `FluxCurve/replay.py` | Replay-mode curve evaluation for fast objective re-evaluation |
| `Forward/steady_state.py` | `SteadyStateConfig`, `solve_to_steady_state_for_phi_applied` |

### Steady-state criterion

$$
\mathrm{rel}^{(n)} = \max_i \frac{|F_i^{(n)} - F_i^{(n-1)}|}{\max(|F_i^{(n)}|, |F_i^{(n-1)}|, \varepsilon_\mathrm{abs})} \le \varepsilon_\mathrm{rel}
$$

declared after `consecutive_steps` successive steady steps.

### Synthetic noise model

$$
\sigma = \tfrac{p}{100} \cdot \mathrm{RMS}(F_\mathrm{clean})
$$

### Forward-solve resilience

On solver divergence, recovery stages are applied in order:
1. Increase `snes_max_it`.
2. Reduce kappa anisotropy.
3. Relax `snes_atol`, `snes_rtol`, `ksp_rtol`; try alternate linesearch.

### Parallel point solves

```python
RobinFluxCurveInferenceRequest(
    ...
    parallel_point_solves_enabled=True,
    parallel_point_workers=4,
    parallel_point_min_points=4,
    parallel_start_method="spawn",   # required for adjoint tape isolation
)
```

Benchmark: 4-worker process-parallel gives ~2.5× speedup over serial (279 s → 112 s) on the current machine.

### Quick start

```bash
# Flux curve inference
python InferenceScripts/Infer_RobinKappa_from_flux_curve.py

# Current-density proxy inference
python InferenceScripts/Infer_RobinKappa_from_current_density_curve.py
```

Outputs go to `StudyResults/robin_flux_experiment/` and `StudyResults/robin_current_density_experiment/`.

---

## Nondimensionalization Package

`Nondim/` is the single source of truth for all physical constants and scaling logic.

```python
from Nondim import build_physical_scales, build_model_scaling, NondimScales
from Nondim.constants import FARADAY_CONSTANT, GAS_CONSTANT, DEFAULT_TEMPERATURE_K
```

`build_model_scaling()` handles both dimensional (`enabled=False`) and dimensionless (`enabled=True`) modes and is called internally by all three forward solvers. Known fix applied: the `enabled=False` path previously set `poisson_coefficient=1.0`, silently dropping the physical permittivity; it now correctly uses `permittivity_f_m`.

---

## Active Inference Scripts

| Script | Description |
|---|---|
| `bv_iv_curve.py` | Butler-Volmer I–V sweep and parameter study |
| `Infer_RobinKappa_from_flux_curve.py` | Robin kappa from φ vs. flux curve |
| `Infer_RobinKappa_from_current_density_curve.py` | Robin kappa from φ vs. current-density proxy |
| `Infer_parameter_from_data.py` | Unified inverse entrypoint |
| `Infer_DirichletBC_from_data.py` | Dirichlet BC inference example |
| `Infer_D_from_data.py` | Diffusivity inference example |
| `Infer_D_from_data_robin.py` | Diffusivity with Robin forward solver |
| `Infer_RobinBC_from_data.py` | Robin kappa inference from state data |

---

## Solver Parameter Convention

```text
SolverParams([n_species, order, dt, t_end, z_vals, D_vals, a_vals,
              phi_applied, c0_vals, phi0, params_dict])
```

| Index | Name | Type | Description |
|---|---|---|---|
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
