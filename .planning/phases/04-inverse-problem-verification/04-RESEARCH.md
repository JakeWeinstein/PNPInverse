# Phase 4: Inverse Problem Verification - Research

**Researched:** 2026-03-07
**Domain:** Inverse problem V&V -- parameter recovery, gradient consistency, multistart convergence
**Confidence:** HIGH

## Summary

Phase 4 verifies the v13 inverse inference pipeline by proving parameter recovery from synthetic data, validating gradient consistency, and demonstrating multistart optimizer convergence. The codebase already contains substantial infrastructure for all three requirements: `SurrogateObjective` classes with FD gradients (INV-02 surrogate path), `run_multistart_inference()` (INV-03), and the full 7-phase v13 pipeline script (INV-01). Existing Tests 1, 2, and 7 in `test_v13_verification.py` cover simpler versions of these tests and will be consolidated/replaced.

The primary new work involves: (1) running parameter recovery at 4 noise levels with 3 realizations each using PDE-generated targets, (2) updating the noise model from percent-of-RMS (additive) to percent-of-signal (multiplicative), (3) adding a PDE adjoint-vs-FD gradient test, (4) upgrading the multistart test to use the production `run_multistart_inference()` with full 20K LHS config, and (5) saving structured JSON/CSV artifacts for Phase 6 report consumption.

**Primary recommendation:** Build one new test file `tests/test_inverse_verification.py` that consolidates INV-01/02/03, following the Phase 3 `test_surrogate_fidelity.py` pattern for artifact output and soft gates. Update `add_percent_noise()` to support multiplicative (percent-of-signal) mode before running any tests.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **INV-01 scope**: Replace and consolidate existing Tests 1, 2, and 7 from `test_v13_verification.py` into a new Phase 4 test file. Run the full 7-phase v13 pipeline (S1-S5 surrogate + P1-P2 PDE refinement) at each noise level: 0%, 1%, 2%, 5%. Soft gates scaled by noise level: 0% noise < 5% error, 1% < 10%, 2% < 15%, 5% < 30% (per-parameter max relative error). Actual errors saved as artifacts. Output: JSON summary + CSV per-run details to `StudyResults/inverse_verification/`.
- **INV-02 gradient consistency**: PDE adjoint vs FD: Compare Firedrake automatic adjoint gradient against central FD on PDE objective function. Tolerance: component-wise relative agreement within 1%. Evaluate at 3 points: true parameters + 2 random perturbations. Surrogate FD self-consistency: Verify FD gradient convergence rate at step sizes h=1e-3, 1e-5, 1e-7. Central FD should show O(h^2) convergence. Fast test (no Firedrake). PDE gradient test is `@pytest.mark.slow`; surrogate FD convergence is a fast test.
- **INV-03 multistart**: Use full production `run_multistart_inference()` config: 20K LHS grid + top-20 candidate polish. Soft gate: >50% of polished candidates recover all 4 parameters within 10% of truth. Report 4 statistics: % converging, loss distribution, parameter spread, best-vs-worst gap. All saved to JSON.
- **Noise handling**: Always use PDE-generated synthetic targets (never surrogate-generated). Cache clean PDE solutions. Switch noise model from percent-of-range (additive) to percent-of-signal (multiplicative): `noise_std = noise_percent/100 * |signal|`. 3 noise realizations per noise level with fixed seeds (42, 43, 44). Report mean/std of recovery error across realizations. Scaled soft gates apply to mean error.

### Claude's Discretion
- Exact test file organization (single class vs multiple classes per requirement)
- How to structure the PDE adjoint gradient extraction from Firedrake's ReducedFunctional
- FD step size selection for the PDE adjoint-vs-FD test (optimal sqrt(eps) or similar)
- Plot styling for any diagnostic figures
- How to handle the noise model update across the codebase (backward compatibility)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INV-01 | Parameter recovery from v13 synthetic data at multiple noise levels (0%, 1%, 2%, 5%) | Full 7-phase pipeline exists in `Infer_BVMaster_charged_v13_ultimate.py`; PDE target generation via `_generate_targets_with_pde()`; noise model update needed in `add_percent_noise()` |
| INV-02 | Gradient consistency verification (finite-difference vs adjoint) for the v13 objective function | Surrogate FD gradient classes exist in `Surrogate/objectives.py`; PDE adjoint available via `Inverse/inference_runner/objective.py:build_reduced_functional()`; Firedrake ReducedFunctional provides `.derivative()` |
| INV-03 | Multistart convergence basin analysis showing the v13 optimizer finds the correct minimum | `Surrogate/multistart.py:run_multistart_inference()` is the production function; existing Test 7 uses manual 5-start L-BFGS-B instead |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (project version) | Array operations, noise generation, error computation | Universal numerical Python |
| scipy.optimize | (project version) | L-BFGS-B optimizer used in surrogate phases | Already used throughout pipeline |
| pytest | (project version) | Test framework | Project standard |
| matplotlib | (project version) | Diagnostic plots (Agg backend) | Phase 3 precedent |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| firedrake | (env version) | PDE forward solves + adjoint gradients | INV-01 full pipeline, INV-02 PDE gradient test |
| firedrake.adjoint | (env version) | ReducedFunctional for adjoint gradients | INV-02 PDE adjoint vs FD test |
| scipy.stats.qmc | (project version) | LatinHypercube sampler | INV-03 multistart |

### Alternatives Considered
None -- all libraries are locked by existing codebase usage.

## Architecture Patterns

### Recommended Project Structure
```
tests/
  test_inverse_verification.py     # NEW: INV-01, INV-02, INV-03 consolidated
  test_v13_verification.py         # MODIFIED: Tests 1, 2, 7 removed (3, 4, 6 remain)
Forward/steady_state/common.py     # MODIFIED: add_percent_noise updated for multiplicative mode
StudyResults/inverse_verification/  # NEW: output directory
  parameter_recovery_summary.json
  parameter_recovery_details.csv
  gradient_consistency.json
  multistart_basin.json
```

### Pattern 1: Noise Model Update (Backward-Compatible)
**What:** Update `add_percent_noise()` to support both additive (RMS-based) and multiplicative (per-point signal-based) noise modes.
**When to use:** Called from all noise injection paths in the project.
**Example:**
```python
def add_percent_noise(
    values: Sequence[float],
    noise_percent: float,
    *,
    seed: int | None = None,
    mode: str = "rms",  # "rms" (legacy) or "signal" (multiplicative)
) -> np.ndarray:
    """Add Gaussian noise.

    mode="rms": sigma = pct/100 * RMS(values)  [legacy, global sigma]
    mode="signal": sigma_i = pct/100 * |values_i|  [per-point multiplicative]
    """
    v = np.asarray(values, dtype=float)
    pct = float(noise_percent)
    if pct <= 0:
        return v.copy()
    rng = np.random.default_rng(seed)
    finite_mask = np.isfinite(v)
    out = v.copy()

    if mode == "signal":
        # Per-point multiplicative noise
        sigma = (pct / 100.0) * np.abs(v)
        sigma = np.where(sigma < 1e-12, 1e-12, sigma)
        noise = rng.normal(0.0, 1.0, size=v.shape) * sigma
    else:
        # Legacy RMS-based additive noise
        v_finite = v[finite_mask]
        if v_finite.size == 0:
            return v.copy()
        rms = float(np.sqrt(np.mean(v_finite * v_finite)))
        sigma = (pct / 100.0) * max(rms, 1e-12)
        noise = rng.normal(0.0, sigma, size=v.shape)

    out[finite_mask] += noise[finite_mask]
    return out
```

### Pattern 2: PDE Target Caching (Reuse Existing)
**What:** Cache clean PDE solutions and apply noise separately.
**When to use:** INV-01 parameter recovery tests.
**Example:** Follow `_generate_targets_with_pde()` from the v13 script. The cache key encodes `phi_applied_values` and `observable_scale`. For non-true parameter values, extend the cache key with the parameter hash.

### Pattern 3: Test Structure (Phase 3 Precedent)
**What:** Module-level fixtures, JSON+CSV artifact output, soft gates.
**When to use:** All Phase 4 tests.
**Example:**
```python
@pytest.fixture(scope="module")
def nn_ensemble():
    """Load NN ensemble surrogate (module-scoped, shared across tests)."""
    ...

class TestParameterRecovery:
    """INV-01: Parameter recovery at multiple noise levels."""

    @pytest.mark.slow
    def test_recovery_at_noise_levels(self, nn_ensemble):
        ...
        # Save artifacts
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "parameter_recovery_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
```

### Pattern 4: PDE Adjoint Gradient Extraction
**What:** Use Firedrake's ReducedFunctional to get adjoint gradients for comparison with FD.
**When to use:** INV-02 PDE gradient test.
**Example:**
```python
# Build ReducedFunctional using existing infrastructure
from Inverse.inference_runner import build_reduced_functional

rf = build_reduced_functional(
    adapter=adapter,
    target=target,
    solver_params=solver_params,
    concentration_targets=c_targets,
    phi_target=phi_target,
    blob_initial_condition=True,
)

# Get adjoint gradient
import firedrake
controls = [firedrake.Constant(v) for v in param_values]
dJ = rf.derivative()  # Returns adjoint gradient w.r.t. controls

# Compare with central FD
h = 1e-5  # ~sqrt(machine_eps) for float64
for i, ctrl in enumerate(controls):
    ctrl_plus = ctrl + h
    ctrl_minus = ctrl - h
    fd_grad_i = (rf(ctrl_plus) - rf(ctrl_minus)) / (2 * h)
    adjoint_grad_i = float(dJ[i])
    rel_err = abs(fd_grad_i - adjoint_grad_i) / max(abs(adjoint_grad_i), 1e-30)
    assert rel_err < 0.01  # 1% tolerance
```

### Anti-Patterns to Avoid
- **Surrogate-on-surrogate inverse crime:** Never generate synthetic targets from the surrogate and then infer parameters using the same surrogate. Always use PDE-generated targets.
- **RNG seed sharing across noise realizations:** Use distinct seeds (42, 43, 44) per realization, not a single RNG that advances.
- **Gradient testing at the minimum:** At theta*, gradients are zero and relative error is meaningless. Always evaluate at perturbed points.
- **Hardcoded absolute thresholds for noisy recovery:** Error scales with noise level; use noise-scaled soft gates (0% < 5%, 1% < 10%, 2% < 15%, 5% < 30%).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDE adjoint gradients | Manual chain rule through PDE solver | `Inverse/inference_runner:build_reduced_functional()` + `rf.derivative()` | Firedrake's tape-based AD handles PDE chain rule correctly |
| LHS sampling | Custom random grid | `Surrogate/multistart.py:run_multistart_inference()` with `MultiStartConfig` | Production code with proper LHS spacing, vectorized batch eval |
| Full 7-phase pipeline | Manual sequence of S1-S5, P1-P2 stages | Import from `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` or refactor key functions | Complex multi-stage pipeline with many configuration details |
| PDE target generation | Manual `solve_bv_curve_points_with_warmstart` setup | `_generate_targets_with_pde()` from v13 script with caching | Caching logic, warmstart, recovery config |
| Noise injection | Custom noise function | Updated `add_percent_noise()` from `Forward/steady_state/common.py` | NaN handling, RNG reproducibility |

**Key insight:** The codebase has mature implementations for all three requirements. The tests should exercise the production code, not reimplement it.

## Common Pitfalls

### Pitfall 1: PyTorch/MPI Segfault in predict_batch
**What goes wrong:** `run_multistart_inference()` uses `predict_batch()` which can trigger PyTorch/MPI segfaults in test environments.
**Why it happens:** Threading/forking conflicts between PyTorch and Firedrake MPI in the same process.
**How to avoid:** Test 7 in the existing suite specifically avoids `run_multistart_inference()` for this reason. If the production function segfaults, fall back to loop-based `predict()` evaluation (as in existing Test 7). Mark multistart tests appropriately and test in isolation.
**Warning signs:** Segfaults or SIGABRT during batch surrogate evaluation.

### Pitfall 2: Noise Model Migration Breaking Existing Tests
**What goes wrong:** Changing `add_percent_noise()` behavior globally could break other tests/scripts that depend on the RMS-based noise.
**Why it happens:** The function is imported from `Forward/steady_state/__init__.py` and used in 26+ files.
**How to avoid:** Add a `mode` parameter with default `"rms"` (backward compatible). Phase 4 tests explicitly pass `mode="signal"`. This preserves all existing behavior while enabling the new model.
**Warning signs:** Existing noise tests or scripts producing different outputs.

### Pitfall 3: FD Step Size for PDE Gradient Comparison
**What goes wrong:** FD gradient agrees poorly with adjoint despite both being correct.
**Why it happens:** Step too large = truncation error dominates. Step too small = cancellation error dominates (especially with iterative PDE solves that have their own convergence tolerance).
**How to avoid:** Use h ~ 1e-5 to 1e-4 (well above PDE solver tolerance of ~1e-7, well below O(1) parameter scales). The k0 parameters are in log-space where a step of 1e-5 in log10(k0) is a ~0.002% change in k0 -- appropriate. Alpha parameters are O(0.5), so h=1e-5 gives a ~0.002% perturbation. Both are reasonable for central FD with O(h^2) error.
**Warning signs:** FD-vs-adjoint disagreement > 5% at all evaluation points.

### Pitfall 4: Parameter Recovery at 5% Noise Hitting 30% Gate
**What goes wrong:** High noise causes optimizer to converge to wrong basin, failing the 30% gate.
**Why it happens:** Noise realizations can push the objective landscape enough to shift the minimum.
**How to avoid:** Gate is on the *mean* error across 3 realizations (seeds 42, 43, 44). Individual realizations may exceed 30%; only the mean must pass. The 30% threshold is generous. If it still fails, investigate whether the full 7-phase pipeline (with PDE refinement P1/P2) is actually able to recover from surrogate-phase errors under noise.
**Warning signs:** High variance across noise realizations at 5% level.

### Pitfall 5: Firedrake Cache Stale Across Gradient Evaluations
**What goes wrong:** Adjoint gradient computed from stale forward solve state.
**Why it happens:** Firedrake's annotation tape can accumulate state from previous evaluations.
**How to avoid:** Clear caches with `_clear_caches()` between independent forward solves. Use `adj.stop_annotating()` context when generating targets, then re-enable for the gradient-tested forward pass.
**Warning signs:** Adjoint gradient that doesn't change when parameters change.

## Code Examples

### Existing noise model (current, to be extended)
```python
# Source: Forward/steady_state/common.py lines 136-163
def add_percent_noise(values, noise_percent, *, seed=None):
    """sigma = pct/100 * RMS(finite values) -- global sigma, additive."""
    rms = float(np.sqrt(np.mean(v_finite * v_finite)))
    sigma = (pct / 100.0) * max(rms, 1e-12)
    noise = rng.normal(0.0, sigma, size=v.shape)
```

### Target noise model (per CONTEXT.md decision)
```python
# Multiplicative noise: noise_std = noise_percent/100 * |signal| per point
sigma_per_point = (noise_percent / 100.0) * np.abs(values)
sigma_per_point = np.maximum(sigma_per_point, 1e-12)  # floor for near-zero values
noise = rng.normal(0.0, 1.0, size=values.shape) * sigma_per_point
noisy = values + noise
```

### Production multistart invocation
```python
# Source: Surrogate/multistart.py
from Surrogate.multistart import run_multistart_inference, MultiStartConfig

config = MultiStartConfig(
    n_grid=20_000,
    n_top_candidates=20,
    polish_maxiter=60,
    secondary_weight=1.0,
    fd_step=1e-5,
    use_shallow_subset=True,
    seed=42,
    verbose=False,
)

result = run_multistart_inference(
    surrogate=nn_ensemble,
    target_cd=target_cd,
    target_pc=target_pc,
    bounds_k0_1=bounds_k0_1,
    bounds_k0_2=bounds_k0_2,
    bounds_alpha=(0.1, 0.9),
    config=config,
    subset_idx=np.arange(nn_ensemble.n_eta),
)
# result.candidates is tuple of MultiStartCandidate
# Each has: k0_1, k0_2, alpha_1, alpha_2, polished_loss
```

### Artifact output pattern (Phase 3 precedent)
```python
# Source: tests/test_surrogate_fidelity.py pattern
_OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "inverse_verification")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "requirement": "INV-01",
    "noise_levels": [0.0, 1.0, 2.0, 5.0],
    "realizations_per_level": 3,
    "results": { ... },
}
with open(os.path.join(_OUTPUT_DIR, "parameter_recovery_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Tests 1,2,7 in test_v13_verification.py | Consolidated test_inverse_verification.py | Phase 4 | Single authoritative source for all inverse V&V |
| RMS-based additive noise | Per-point multiplicative noise | Phase 4 | More physically realistic noise model |
| Manual 5-start L-BFGS-B (Test 7) | Production run_multistart_inference (20K LHS) | Phase 4 | Tests actual production code path |
| Surrogate-generated targets (inverse crime) | PDE-generated targets always | Phase 4 | Statistically valid recovery tests |

**Deprecated/outdated:**
- Tests 1, 2, 7 in `test_v13_verification.py`: Will be removed after Phase 4 tests are in place
- `add_percent_noise()` with only RMS mode: Extended with `mode` parameter

## Open Questions

1. **PDE adjoint availability for BV objectives**
   - What we know: `Inverse/objectives.py` provides `build_reduced_functional()` factories for diffusion, Dirichlet phi0, and Robin kappa objectives. The BV I-V curve objective used in v13 is NOT built via `build_reduced_functional()` -- it uses manual FD on the `solve_bv_curve_points_with_warmstart()` forward pass.
   - What's unclear: Whether Firedrake's tape-based adjoint can differentiate through the BV steady-state solver chain (multi-voltage-point warm-started solves). The v13 PDE phases (P1/P2) use FD gradients, not adjoints.
   - Recommendation: The INV-02 PDE adjoint test may need to use a simpler single-voltage-point PDE objective where the adjoint tape works (similar to Test 2's structure). If the BV curve objective is not adjoint-differentiable, the "PDE adjoint vs FD" test should document this and test at the single-solve level. Alternatively, if the v13 pipeline only uses FD for PDE gradients, the PDE gradient consistency test becomes "FD convergence rate on PDE objective" (analogous to the surrogate FD convergence test).

2. **Full 7-phase pipeline accessibility from tests**
   - What we know: The 7-phase pipeline lives in `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` as a monolithic `main()` function with argparse configuration.
   - What's unclear: Whether the pipeline can be invoked programmatically from a test, or if key stage functions need to be extracted.
   - Recommendation: The test should import and call the individual stage functions (S1-S5, P1-P2) directly rather than invoking the full CLI. The v13 script's `_generate_targets_with_pde()` and individual optimization stages are importable.

3. **Multistart predict_batch segfault risk**
   - What we know: Existing Test 7 explicitly avoids `run_multistart_inference()` due to "PyTorch/MPI segfaults in some environments."
   - What's unclear: Whether the current test environment has this issue.
   - Recommendation: Try `run_multistart_inference()` first. If it segfaults, fall back to manual loop-based evaluation (compute grid objectives via loop, then polish top candidates individually). Document the fallback in the test.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | tests/conftest.py (existing, provides skip_without_firedrake, fixtures) |
| Quick run command | `pytest tests/test_inverse_verification.py -m "not slow" -x` |
| Full suite command | `pytest tests/test_inverse_verification.py -m slow --tb=short` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INV-01 | Parameter recovery at 0/1/2/5% noise | integration (slow, Firedrake) | `pytest tests/test_inverse_verification.py::TestParameterRecovery -m slow --tb=short -x` | Wave 0 |
| INV-02a | PDE adjoint vs FD gradient | integration (slow, Firedrake) | `pytest tests/test_inverse_verification.py::TestGradientConsistencyPDE -m slow --tb=short -x` | Wave 0 |
| INV-02b | Surrogate FD convergence rate | unit (fast, no Firedrake) | `pytest tests/test_inverse_verification.py::TestSurrogateFDConvergence -x` | Wave 0 |
| INV-03 | Multistart convergence basin | integration (slow) | `pytest tests/test_inverse_verification.py::TestMultistartBasin -m slow --tb=short -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_inverse_verification.py -m "not slow" -x`
- **Per wave merge:** `pytest tests/test_inverse_verification.py -m slow --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_inverse_verification.py` -- covers INV-01, INV-02, INV-03
- [ ] `StudyResults/inverse_verification/` directory -- output location for artifacts
- [ ] `add_percent_noise()` update in `Forward/steady_state/common.py` -- multiplicative noise mode

## Sources

### Primary (HIGH confidence)
- `tests/test_v13_verification.py` -- existing tests to be replaced (read in full)
- `Surrogate/objectives.py` -- SurrogateObjective, AlphaOnlySurrogateObjective, SubsetSurrogateObjective (read in full)
- `Surrogate/multistart.py` -- run_multistart_inference, MultiStartConfig (read in full)
- `scripts/_bv_common.py` -- true parameter values, PDE target generation (read in full)
- `Forward/steady_state/common.py` -- add_percent_noise implementation (read in full)
- `Inverse/objectives.py` -- PDE objective factories (read in full)
- `Inverse/inference_runner/__init__.py` -- build_reduced_functional, run_inverse_inference (read in full)
- `tests/conftest.py` -- skip_without_firedrake, shared fixtures (read in full)
- `tests/test_surrogate_fidelity.py` -- Phase 3 artifact output pattern (read header)

### Secondary (MEDIUM confidence)
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` -- 7-phase pipeline, `_generate_targets_with_pde()` (grep-searched for noise/target patterns)

### Tertiary (LOW confidence)
- PDE adjoint tape compatibility with BV steady-state multi-voltage-point solves -- not verified; based on codebase structure showing FD-only gradients in PDE phases

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use in the codebase
- Architecture: HIGH -- following established Phase 3 patterns, test consolidation
- Pitfalls: HIGH -- identified from existing test comments and codebase inspection
- PDE adjoint for BV objectives: LOW -- unclear if Firedrake tape handles the multi-solve BV curve objective

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable codebase, no external dependency changes expected)
