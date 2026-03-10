# Technology Stack

**Project:** PNP-BV v14 Pipeline Redesign -- Robust Surrogate-Assisted Inverse Parameter Recovery
**Researched:** 2026-03-09
**Scope:** NEW capabilities only. Existing validated stack (Firedrake, numpy, scipy, torch, pytest, matplotlib) is out of scope.

## What Already Exists (DO NOT CHANGE)

These are validated, working, and should not be upgraded or replaced:

| Technology | Current Use | Status |
|------------|-------------|--------|
| Firedrake | PDE forward solver (PNP + Butler-Volmer) | MMS-verified, keep as-is |
| NumPy | Array operations throughout | Core dependency, keep |
| SciPy (`optimize.minimize`, `stats.qmc.LatinHypercube`) | L-BFGS-B optimization, LHS sampling | Working in multistart/cascade, keep |
| PyTorch | NN surrogate model training and inference | Validated, keep |
| matplotlib | Plotting | Keep |
| pytest + pytest-cov | Testing | Keep |
| h5py | Data storage | Keep |

## Recommended Stack Additions

### 1. Sensitivity Analysis

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| SALib | >=1.5.1 | Sobol and Morris sensitivity analysis on surrogate objectives | SALib is the standard Python library for global sensitivity analysis. Sobol indices quantify which parameters (k0_1, k0_2, alpha_1, alpha_2) most influence the objective, informing whether all 4 parameters are identifiable from I-V data at 2% noise. Morris screening is cheaper for initial factor prioritization. Both methods are well-cited in the inverse problems literature. |

**Integration:** SALib works directly with numpy arrays. The surrogate `predict_batch` function already returns numpy arrays, so the SALib workflow is: (1) `SALib.sample.sobol.sample()` generates parameter samples, (2) evaluate each via existing `BVSurrogateModel.predict_batch()`, (3) `SALib.analyze.sobol.analyze()` computes first-order and total-effect Sobol indices. No adapter code needed beyond a thin wrapper.

**Confidence:** HIGH -- SALib is the de facto standard (2000+ citations), actively maintained (v1.5.2 released Oct 2025), pure Python with numpy/scipy dependencies only.

### 2. Statistical Comparison of Pipeline Variants

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `scipy.stats` (already installed) | >=1.12 | Wilcoxon signed-rank test, bootstrap CIs for paired pipeline comparisons | Already available. `scipy.stats.wilcoxon` tests whether pipeline A beats pipeline B across noise seeds (paired, non-parametric). `scipy.stats.bootstrap` (added in scipy 1.7) computes BCa confidence intervals on median relative error. No new dependency needed. |
| pandas | >=2.1 | Tabular comparison of pipeline variants across seeds, parameters, and metrics | The project already uses CSV files for results (see `StudyResults/master_inference_v13/`). pandas provides groupby aggregation (median/IQR by parameter, by seed), pivot tables for variant comparison, and clean CSV I/O. The alternative is raw numpy, which becomes unwieldy for multi-factor comparisons. |

**Why not statsmodels:** Overkill. The comparisons needed are simple paired non-parametric tests across seeds (Wilcoxon, bootstrap CI). statsmodels adds a large dependency for features that won't be used.

**Why not a dedicated experiment tracking framework (MLflow, Weights & Biases):** Out of scope. The number of pipeline variants is small (10-20 configurations), all local. CSV + pandas is sufficient and keeps the stack simple.

**Confidence:** HIGH -- scipy.stats is already installed; pandas is the universal Python data wrangling tool.

### 3. Robust Optimization Alternatives

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `scipy.optimize.differential_evolution` (already installed) | >=1.12 | Global optimizer as alternative to multistart L-BFGS-B | Already available in scipy. Differential evolution is gradient-free, handles bounds natively, and is well-suited to 4D surrogate optimization where the objective is cheap to evaluate (surrogate calls are sub-millisecond). Worth benchmarking against the current LHS + L-BFGS-B approach for robustness across noise seeds. |
| `scipy.optimize.least_squares` (already installed) | >=1.12 | Gauss-Newton / Trust Region Reflective for residual-based objectives | Already partially used in `FluxCurve/bv_run/optimization.py`. The residual formulation (minimize sum of squared residuals per voltage point) gives natural Jacobian structure. SciPy 1.16 added a `callback` argument for trf/dogbox methods, useful for convergence tracking. |
| `scipy.optimize.dual_annealing` (already installed) | >=1.12 | Simulated annealing variant as additional global optimizer candidate | Available in scipy. Combines classical simulated annealing with local search. Potentially more robust than DE for low-dimensional problems with narrow basins. Worth benchmarking but likely inferior to multistart L-BFGS-B for this problem. |

**Why not pymoo/pysamoo:** These are multi-objective optimization frameworks. The inverse problem is single-objective (weighted sum of CD + PC residuals). Adding pymoo introduces framework overhead with no benefit over scipy.optimize.

**Why not Optuna/hyperopt:** These are hyperparameter tuning frameworks designed for ML model selection, not scientific optimization. They add unnecessary abstraction over what scipy.optimize already does.

**Why not nlopt:** Redundant with scipy.optimize for this problem size (4 parameters). nlopt's advantage is in high-dimensional problems or when specific algorithms (COBYLA variants, MMA) are needed.

**Confidence:** HIGH -- all recommendations are already in scipy, no new dependencies.

### 4. Profile Likelihood and Identifiability

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| lmfit | >=1.3.0 | Profile likelihood confidence intervals, parameter identifiability analysis | lmfit wraps scipy.optimize with a Parameter class that supports bounds, fixing, and constraints. Its `conf_interval()` function computes profile likelihood CIs using the F-test, which is exactly what `Infer_BVProfileLikelihood_charged.py` does manually. Using lmfit would standardize this and add covariance matrix CIs for free. |

**Integration:** lmfit's `minimize()` accepts a residual function returning an array (like scipy least_squares). The existing surrogate objective can be wrapped: define lmfit `Parameters` with bounds matching the current `[(lb, ub)]` tuples, write a residual function calling `surrogate.predict()`, and lmfit handles the rest. Profile likelihood plots come from `ci_report()` and `plot_ci()`.

**Alternative considered -- ci-rvm:** The Venzon-Moolgavkar algorithm (ci-rvm package) computes profile likelihood CIs more efficiently than grid search, but it is a niche package with limited maintenance. lmfit is battle-tested (15+ years, 3000+ citations) and does the same thing.

**Whether to add this:** OPTIONAL. The existing hand-rolled profile likelihood script works. lmfit is worth adding only if profile likelihood analysis becomes a routine part of the pipeline comparison workflow (e.g., checking identifiability for every pipeline variant). If it is a one-off analysis, keep the existing script.

**Confidence:** MEDIUM -- lmfit is well-established, but the existing manual implementation may be sufficient.

### 5. Experiment Orchestration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| (none -- use scripts) | -- | Running pipeline variant comparisons | The v14 work involves running ~10-20 pipeline configurations across ~10-20 noise seeds. This is 100-400 runs, each taking seconds (surrogate) to minutes (PDE). A simple Python script with nested loops and CSV output is appropriate. No experiment framework needed. |

**Why not Hydra/Sacred/DVC:** Massive overkill for 100-400 runs. These frameworks solve configuration management and reproducibility for ML experiments with thousands of runs. A dataclass-based config + CSV results file is simpler and sufficient.

## Recommended Stack (Summary)

### Required Additions

```bash
# Within Firedrake venv
pip install SALib>=1.5.1
pip install pandas>=2.1
```

### Optional Addition

```bash
# Only if profile likelihood becomes routine
pip install lmfit>=1.3.0
```

### Already Available (No Install Needed)

- `scipy.optimize.differential_evolution` -- benchmark as global optimizer alternative
- `scipy.optimize.least_squares` -- benchmark residual-based formulation
- `scipy.optimize.dual_annealing` -- benchmark as SA-based global optimizer
- `scipy.stats.wilcoxon` -- paired non-parametric test for pipeline comparison
- `scipy.stats.bootstrap` -- confidence intervals on comparison metrics
- `scipy.stats.qmc.LatinHypercube` -- already used for multistart

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| pymoo / pysamoo | Multi-objective framework; this is a single-objective problem | `scipy.optimize` (already available) |
| Optuna / hyperopt | ML hyperparameter tuners, wrong abstraction for scientific inverse problems | `scipy.optimize` + manual experiment scripts |
| statsmodels | Heavy dependency for simple paired tests | `scipy.stats.wilcoxon` + `scipy.stats.bootstrap` |
| MLflow / W&B | Experiment tracking overkill for <500 local runs | pandas DataFrame + CSV files |
| nlopt | Redundant with scipy for 4-parameter problems | `scipy.optimize` |
| hIPPYlib / occamypy | PDE-based inverse problem libraries assuming different solver architectures | Existing Firedrake + surrogate pipeline |
| Hydra / Sacred / DVC | Configuration management overkill | Python dataclasses + argparse |
| emcee / PyMC | Bayesian MCMC samplers; not needed for point estimation with noise robustness | Multistart + profile likelihood for uncertainty |

## Version Compatibility Notes

| Package | Compatible With | Constraint |
|---------|-----------------|------------|
| SALib 1.5.x | numpy >=1.20, scipy >=1.7, matplotlib, pandas | SALib depends on pandas already, so adding pandas is not an extra dependency beyond what SALib brings |
| pandas 2.x | numpy >=1.23, Python >=3.9 | Use 2.x not 3.x; pandas 3.0 (Jan 2026) requires Python >=3.11 and has breaking changes. Firedrake environments may still be on Python 3.10. |
| lmfit 1.3.x | numpy, scipy, asteval, uncertainties | Light dependency chain. asteval is a safe expression evaluator (no security concern). |
| scipy 1.15-1.17 | Already constrained by Firedrake environment | Do not upgrade scipy independently of Firedrake. Use whatever version Firedrake provides (likely 1.12-1.15). All recommended features exist in 1.12+. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Sensitivity analysis | SALib | Custom Sobol implementation | SALib is 50 lines to use vs 500 to reimplement. Well-tested, well-cited. |
| Global optimization | scipy differential_evolution | CMA-ES (pycma) | DE is already in scipy; CMA-ES adds a dependency for marginal benefit in 4D. |
| Statistical testing | scipy.stats.wilcoxon | permutation_test (scipy 1.15+) | Wilcoxon is sufficient for paired comparisons across seeds. Permutation test is an option if more power needed but requires scipy >=1.15. |
| Data wrangling | pandas | polars | pandas is the standard, already a SALib dependency, and the data volumes are tiny (<1MB). |
| Profile likelihood | lmfit (optional) | Keep existing manual script | Manual script works. lmfit only worth it if this becomes routine. |

## Sources

- [SALib documentation](https://salib.readthedocs.io/) -- v1.5.2, sensitivity analysis methods (HIGH confidence)
- [SALib GitHub releases](https://github.com/SALib/SALib/releases) -- v1.5.2 released Oct 2025 (HIGH confidence)
- [scipy.optimize.differential_evolution docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) -- SciPy v1.17.0 (HIGH confidence)
- [scipy.stats.bootstrap docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) -- BCa method, paired support (HIGH confidence)
- [SciPy 1.16.0 release notes](https://docs.scipy.org/doc/scipy/release/1.16.0-notes.html) -- least_squares callback, L-BFGS-B hess_inv improvement (HIGH confidence)
- [lmfit confidence intervals docs](https://lmfit.github.io/lmfit-py/confidence.html) -- profile likelihood F-test method (HIGH confidence)
- [pandas 3.0.0 release notes](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v3.0.0.html) -- breaking changes, Python >=3.11 requirement (HIGH confidence)
- Existing codebase analysis: `Surrogate/multistart.py`, `Surrogate/cascade.py`, `FluxCurve/bv_run/optimization.py`, `Infer_BVProfileLikelihood_charged.py`, `pyproject.toml` (HIGH confidence)

---
*Stack research for: PNP-BV v14 Pipeline Redesign*
*Researched: 2026-03-09*
