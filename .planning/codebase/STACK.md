# Technology Stack

**Analysis Date:** 2026-03-06

## Languages

**Primary:**
- Python >=3.10 - All source code, scripts, and tests

**Secondary:**
- LaTeX - Paper writeups in `writeups/WeekOfMar4/v13_pipeline_paper.tex`

## Runtime

**Environment:**
- CPython (Firedrake-managed virtual environment under `FireDrakeEnvCG/`)
- Firedrake FEM framework installed separately via its own installer (not pip)

**Package Manager:**
- pip / setuptools (build backend: `setuptools.backends._legacy:_Backend`)
- No lockfile present (dependencies are loose in `pyproject.toml`)

## Frameworks

**Core:**
- Firedrake (FEM) - Finite element PDE solver; provides mesh generation, function spaces, variational forms, nonlinear solvers (SNES/PETSc), and adjoint differentiation (`firedrake.adjoint`)
- NumPy - Array operations, numerical computations throughout
- SciPy - Optimization (`scipy.optimize.minimize` with L-BFGS-B), RBF interpolation (`scipy.interpolate.RBFInterpolator`), Latin Hypercube sampling (`scipy.stats.qmc.LatinHypercube`)
- PyTorch (optional) - Neural network surrogate model (`Surrogate/nn_model.py`, `Surrogate/nn_training.py`); conditionally imported with `_TORCH_AVAILABLE` guard

**Testing:**
- pytest - Test runner, configured in `pyproject.toml` `[tool.pytest.ini_options]`
- pytest-cov - Coverage reporting (dev dependency)

**Visualization:**
- matplotlib - Plotting I-V curves, loss histories, fit comparisons (`FluxCurve/plot.py`, `Forward/plotter.py`)

**Build/Dev:**
- setuptools >=64 - Package build system (`pyproject.toml`)

## Key Dependencies

**Critical (declared in pyproject.toml):**
- `numpy` - Core numerical arrays; used in every module
- `scipy` - Optimization (L-BFGS-B), RBF surrogate interpolation, LHS sampling
- `matplotlib` - Visualization of results and solver output
- `h5py` - Listed as dependency; HDF5 file I/O for Firedrake checkpointing

**Critical (installed separately, not in pyproject.toml):**
- `firedrake` - FEM solver framework; provides `firedrake`, `firedrake.adjoint`, `DMPLEX`, PETSc/SNES solvers. Installed via Firedrake's own installer, not pip.
- `PETSc` / `petsc4py` - Underlying linear/nonlinear solver backend used by Firedrake

**Optional:**
- `torch` (PyTorch) - Neural network surrogate model training and inference; guarded by `try/except ImportError` in `Surrogate/nn_model.py` and `Surrogate/nn_training.py`

**Dev:**
- `pytest` - Test framework
- `pytest-cov` - Coverage analysis

## Configuration

**Environment:**
- Firedrake manages its own Python environment with PETSc and all FEM dependencies
- Worker processes set `OMP_NUM_THREADS=1` to prevent thread oversubscription (see `FluxCurve/bv_parallel.py`)
- Firedrake cache directories configured via `FIREDRAKE_TSFC_KERNEL_CACHE_DIR`, `PYOP2_CACHE_DIR`, `XDG_CACHE_HOME` environment variables
- No `.env` files detected; no secrets or API keys required

**Build:**
- `pyproject.toml` - Package metadata, dependencies, pytest config, setuptools package discovery
- Packages included: `Forward*`, `Inverse*`, `FluxCurve*`, `Nondim*`, `Surrogate*`
- Pytest markers: `slow` for tests requiring Firedrake FEM environment

**Physical Constants:**
- Centralized in `Nondim/constants.py`: Faraday constant, gas constant, temperature, permittivity
- Shared BV inference config in `scripts/_bv_common.py`: species properties, solver options, scaling

## Platform Requirements

**Development:**
- macOS (Darwin) or Linux with Firedrake installed
- Firedrake requires PETSc, MPI, and a C compiler (managed by Firedrake installer)
- Python >=3.10
- Optional: PyTorch for NN surrogate (CPU-only sufficient; install via `pip install torch --index-url https://download.pytorch.org/whl/cpu`)

**Production/Research:**
- Multi-core CPU for parallel BV point solves (`FluxCurve/bv_parallel.py` uses `ProcessPoolExecutor` with spawn context)
- No GPU required (PyTorch uses CPU; Firedrake solves are CPU-based)
- No external services, databases, or cloud infrastructure

---

*Stack analysis: 2026-03-06*
