# Testing Patterns

**Analysis Date:** 2026-03-06

## Test Framework

**Runner:**
- pytest (via `pytest` package in `[project.optional-dependencies] dev`)
- Python 3.12 (based on `.pyc` cache files: `cpython-312-pytest-9.0.2`)
- Config: `pyproject.toml` `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest built-in `assert` statements
- `pytest.approx()` for floating-point comparisons with relative tolerance
- `np.testing.assert_allclose()` for array comparisons with `rtol` and `err_msg`

**Run Commands:**
```bash
pytest tests/                              # Run all fast tests
pytest tests/ -m "not slow"                # Exclude Firedrake/PyTorch tests
pytest tests/ -m slow --tb=short           # Run slow integration tests only
pytest tests/ --cov                        # With coverage (pytest-cov)
pytest tests/test_cascade.py -v            # Single test module
```

## Test File Organization

**Location:**
- All tests in a dedicated `tests/` directory (not co-located with source)
- Shared fixtures in `tests/conftest.py`

**Naming:**
- `test_<module_or_feature>.py` pattern
- Test classes: `TestClassName` (PascalCase, no `Test` suffix on the noun)

**Current test files:**
```
tests/
├── conftest.py                    # Shared fixtures, Firedrake availability flag
├── test_bcd.py                    # Block Coordinate Descent tests
├── test_bv_forward.py             # Butler-Volmer forward solver tests
├── test_cascade.py                # Cascade inference strategy tests
├── test_cascade_pde_hybrid.py     # Cascade + PDE hybrid tests
├── test_ensemble_and_v12.py       # NN ensemble wrapper + v12 CLI tests
├── test_fixed_pde.py              # Fixed-parameter PDE tests
├── test_inference_config.py       # Inference configuration tests
├── test_inference_robustness.py   # Inference robustness tests
├── test_multistart.py             # Multi-start optimizer tests
├── test_nondim.py                 # Nondimensionalization tests
├── test_params.py                 # SolverParams dataclass tests
├── test_steady_state_common.py    # Steady-state common utilities tests
├── test_v11_e2e_pde.py            # v11 end-to-end PDE tests
├── test_v11_surrogate_pde.py      # v11 surrogate vs PDE tests
├── test_v13_verification.py       # v13 mathematical verification tests
└── test_weight_sweep.py           # Weight sweep strategy tests
```

## Test Structure

**Suite Organization:**
```python
# Test classes group related tests by component or behavior
class TestCascadeConfig:
    """Tests for the CascadeConfig frozen dataclass."""

    def test_defaults(self):
        cfg = CascadeConfig()
        assert cfg.pass1_weight == 0.5
        assert cfg.pass2_weight == 2.0

    def test_frozen(self):
        cfg = CascadeConfig()
        with pytest.raises(AttributeError):
            cfg.pass1_weight = 0.1

    def test_custom_values(self):
        cfg = CascadeConfig(pass1_weight=0.1, pass2_weight=5.0)
        assert cfg.pass1_weight == 0.1
```

**Section Separators:**
- Test files use `# ===================================================================` block separators between test classes
- `# ---------------------------------------------------------------------------` separators between logical sections (Fixtures, Tests, Helpers)

**Patterns:**
- One test class per logical component or behavior
- Test methods named `test_<behavior_description>`
- Private helper methods on test classes prefixed with `_`: `_make_objective()`, `_get_parser()`
- Each test class has a class-level docstring explaining what it covers

## Test Markers

**Custom Markers (defined in `pyproject.toml`):**
```ini
[tool.pytest.ini_options]
markers = [
    "slow: marks tests requiring Firedrake FEM environment (deselect with '-m \"not slow\"')",
]
```

**Usage:**
```python
@pytest.mark.slow
class TestZeroNoiseIdentity:
    """With zero noise, the surrogate optimizer should recover true parameters."""
    ...

@pytest.mark.slow
def test_loads_d3_deeper(self):
    """Load actual D3-deeper ensemble (requires PyTorch + saved models)."""
    ...
```

**Conditional Skips:**
```python
# Skip if Firedrake is not available (defined in conftest.py)
skip_without_firedrake = pytest.mark.skipif(
    not FIREDRAKE_AVAILABLE,
    reason="Firedrake is not installed or not importable in this environment",
)

# Skip if specific data files are missing
if not os.path.isdir(ensemble_dir):
    pytest.skip("D3-deeper ensemble not found on disk")

# Skip if Firedrake not importable (inline)
try:
    import firedrake
except ImportError:
    pytest.skip("Firedrake not available")
```

## Fixtures

**conftest.py Shared Fixtures:**
```python
@pytest.fixture()
def two_species_diffusivities() -> tuple[float, float]:
    """Return representative O2 / H2O2 diffusivities in m^2/s."""
    return (1.5e-9, 1.6e-9)

@pytest.fixture()
def default_nondim_kwargs(two_species_diffusivities) -> dict:
    """Keyword arguments for build_physical_scales with sensible defaults."""
    return dict(d_species_m2_s=two_species_diffusivities, ...)

@pytest.fixture()
def sample_solver_params():
    """Return a SolverParams instance for a simple 2-species neutral problem."""
    return SolverParams(n_species=2, order=1, dt=0.1, ...)

@pytest.fixture()
def sample_steady_state_results():
    """Return a small list of SteadyStateResult instances for testing."""
    return [SteadyStateResult(...), ...]
```

**Module-scoped Fixtures (for expensive resources):**
```python
@pytest.fixture(scope="module")
def nn_ensemble():
    """Load the real NN ensemble surrogate (shared across the module)."""
    if not os.path.isdir(_ENSEMBLE_DIR):
        pytest.skip("D3-deeper ensemble not found on disk")
    from Surrogate.ensemble import load_nn_ensemble
    return load_nn_ensemble(_ENSEMBLE_DIR, n_members=5, device="cpu")
```

**Per-test-file Fixtures (in test modules, not conftest.py):**
```python
@pytest.fixture()
def fitted_surrogate() -> BVSurrogateModel:
    """Build and fit a small BVSurrogateModel for testing."""
    rng = np.random.default_rng(42)
    config = SurrogateConfig(kernel="thin_plate_spline", ...)
    model = BVSurrogateModel(config)
    # ... fit with synthetic data ...
    model.fit(params, cd, pc, phi_applied)
    return model

@pytest.fixture()
def target_data(fitted_surrogate):
    """Generate target I-V curves at a known parameter set."""
    pred = fitted_surrogate.predict(0.02, 0.002, 0.6, 0.5)
    return {"target_cd": pred["current_density"], "k0_1_true": 0.02, ...}
```

## Mocking

**Framework:** `unittest.mock` (standard library)

**Patterns:**
```python
from unittest.mock import MagicMock

# Custom mock classes that satisfy the surrogate API
class _MockSurrogateModel:
    """Minimal mock that satisfies the surrogate API."""
    def __init__(self, n_eta: int = 22, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.training_bounds = {"k0_1": (1e-5, 1e-1), ...}

    def predict(self, k0_1, k0_2, alpha_1, alpha_2):
        scale = np.log10(max(k0_1, 1e-30)) + alpha_1
        cd = scale * np.sin(self._phi_applied) + self._rng.normal(0, 0.001, self._n_eta)
        return {"current_density": cd, "peroxide_current": pc, "phi_applied": ...}
```

**What to Mock:**
- Surrogate models (for unit tests that don't need real NN/RBF models)
- Firedrake FEM solver (tests that exercise pure-Python logic)
- CLI argument parsers (test structure without running main())

**What NOT to Mock:**
- NumPy/SciPy operations (test actual numerical behavior)
- Frozen dataclass construction (test real immutability)
- Integration tests use real fitted surrogates with synthetic polynomial response functions

## Fixtures and Factories

**Test Data Pattern:**
```python
# Synthetic surrogate with known polynomial response (identifiable parameters)
rng = np.random.default_rng(42)  # Fixed seed for reproducibility
n_samples = 40
n_eta = 8
phi_applied = np.linspace(-10, 5, n_eta)

# Non-separable response (mimics coupled BV kinetics)
cd = (np.outer(alpha_1, -phi_applied) * k0_1[:, None] * 10
      + np.outer(alpha_2 * k0_2, phi_applied ** 2) * 0.1)

# Generate targets at known truth for recovery tests
pred = model.predict(k0_1_true, k0_2_true, alpha_1_true, alpha_2_true)
target_cd = pred["current_density"]
```

**Location:**
- Shared fixtures: `tests/conftest.py`
- Test-specific fixtures: inline in each test file
- No separate fixtures directory

## Coverage

**Requirements:** pytest-cov available as dev dependency, no enforced threshold detected

**View Coverage:**
```bash
pytest tests/ --cov=Forward --cov=Surrogate --cov=Nondim --cov=Inverse --cov=FluxCurve
pytest tests/ --cov --cov-report=html
```

## Test Types

**Unit Tests (fast, no external dependencies):**
- Dataclass construction, frozen enforcement, property accessors: `test_params.py`, `test_cascade.py` (TestCascadeConfig)
- Nondimensionalization math: `test_nondim.py`
- Objective function evaluation with mock surrogates: `test_ensemble_and_v12.py`
- Input validation (ValueError checks): `test_nondim.py` (TestBuildPhysicalScalesValidation)
- CLI argument parsing: `test_ensemble_and_v12.py` (TestV12CLI)

**Integration Tests (use real surrogates, may need PyTorch):**
- Multistart inference end-to-end: `test_multistart.py` (TestRunMultistartInference)
- Cascade inference end-to-end: `test_cascade.py` (TestRunCascadeInference)
- Batch vs single-point objective consistency: `test_multistart.py` (TestBatchObjectiveConsistency)
- Recovery with perfect targets (surrogate-generated data): `test_cascade.py`, `test_multistart.py`

**Slow Tests (require Firedrake FEM and/or real NN models, marked `@pytest.mark.slow`):**
- Mathematical verification: `test_v13_verification.py` (7 test classes)
  - Zero-noise identity recovery
  - Known-parameter PDE roundtrip
  - Gradient verification via finite differences
  - Observable sign and scale checks
  - Surrogate vs PDE consistency
  - Sensitivity monotonicity
  - Multistart convergence basin
- End-to-end PDE solves: `test_v11_e2e_pde.py`
- Surrogate-PDE comparison: `test_v11_surrogate_pde.py`
- Real NN ensemble loading: `test_ensemble_and_v12.py` (test_loads_d3_deeper)

**E2E Tests:**
- No dedicated E2E framework. Slow integration tests in `test_v13_verification.py` serve as the closest equivalent, exercising the full surrogate-to-PDE pipeline.

## Common Patterns

**Frozen Dataclass Testing:**
```python
def test_defaults(self):
    cfg = CascadeConfig()
    assert cfg.pass1_weight == 0.5

def test_frozen(self):
    cfg = CascadeConfig()
    with pytest.raises(AttributeError):
        cfg.pass1_weight = 0.1  # type: ignore[misc]

def test_custom_values(self):
    cfg = CascadeConfig(pass1_weight=0.1)
    assert cfg.pass1_weight == 0.1
```

**Recovery/Roundtrip Testing:**
```python
def test_recovery_with_perfect_targets(self, fitted_surrogate, target_data):
    """With surrogate-generated targets, recover within tolerance."""
    result = run_cascade_inference(surrogate=fitted_surrogate, ...)
    k0_1_err = abs(result.best_k0_1 - target_data["k0_1_true"]) / target_data["k0_1_true"]
    assert result.best_loss < 1e-2
    assert k0_2_err < 0.50, f"k0_2 error {k0_2_err*100:.1f}% too large"
```

**Verbose Suppression Testing:**
```python
def test_verbose_false(self, fitted_surrogate, target_data, capsys):
    """verbose=False should suppress all print output."""
    config = CascadeConfig(verbose=False)
    run_cascade_inference(surrogate=fitted_surrogate, config=config, ...)
    captured = capsys.readouterr()
    assert captured.out == "", f"Expected no output, got: {captured.out!r}"
```

**Error Testing:**
```python
def test_empty_raises(self):
    with pytest.raises(ValueError, match="at least one model"):
        EnsembleMeanWrapper([])

def test_negative_diffusivity_raises(self):
    with pytest.raises(ValueError, match="strictly positive diffusivities"):
        build_physical_scales(d_species_m2_s=(-1e-9, 1e-9))
```

**Numerical Tolerance Assertions:**
```python
# Relative tolerance with pytest.approx
assert scales.d_ref_m2_s == pytest.approx(expected_d_ref, rel=1e-12)

# Array comparison with np.testing
np.testing.assert_allclose(grad_h5, manual_grad, rtol=1e-10,
    err_msg="gradient does not match manual FD computation")

# Custom relative error with informative failure messages
k0_1_relerr = abs(k0_1_rec - K0_HAT_R1) / K0_HAT_R1
assert k0_1_relerr < 0.05, (
    f"k0_1 relative error {k0_1_relerr:.4f} exceeds 5%. "
    f"Recovered {k0_1_rec:.6e}, true {K0_HAT_R1:.6e}"
)
```

**Mathematical Verification Test Pattern (from `test_v13_verification.py`):**
```python
@pytest.mark.slow
class TestGradientVerification:
    """Verify FD gradients by comparing two step sizes and manual FD check.

    Mathematical property tested:
        [Detailed mathematical justification]

    Tolerance:
        [Justification for each tolerance value]
    """

    def test_alpha_only_gradient(self, nn_ensemble, true_predictions):
        # Compare gradients at two step sizes
        grad_h4 = obj_h4.gradient(x_test)
        grad_h5 = obj_h5.gradient(x_test)
        for i in range(2):
            if abs(grad_h5[i]) > 1e-12:
                relerr = abs(grad_h4[i] - grad_h5[i]) / abs(grad_h5[i])
                assert relerr < 5e-2
```

## Test Data and Seeds

- Fixed random seeds for reproducibility: `np.random.default_rng(42)`, `seed=123`, `seed=20260226`
- Synthetic polynomial surrogate responses (not random noise) ensure identifiable parameter recovery
- `tmp_path` fixture used for filesystem tests: `test_missing_member_raises(self, tmp_path)`

---

*Testing analysis: 2026-03-06*
