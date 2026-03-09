# Phase 5: Pipeline Reproducibility - Research

**Researched:** 2026-03-09
**Domain:** pytest regression testing, numerical reproducibility, pipeline orchestration
**Confidence:** HIGH

## Summary

Phase 5 proves the v13 inference pipeline produces deterministic, regression-tested results. The core challenge is not algorithmic -- it is plumbing: invoking the 7-phase pipeline (S1-S5 surrogate + P1-P2 PDE refinement) programmatically from pytest, capturing intermediate outputs (S1, P2), and comparing them against saved JSON baselines with solver-tolerance-aware thresholds.

The existing codebase provides strong foundations: `_run_surrogate_phases()` returns a dict with all phase results, `SurrogateObjective` and `run_multistart_inference()` accept explicit seeds, and Phase 4 already established the subprocess-based PDE target generation pattern to avoid the Firedrake/PyTorch segfault. The primary technical risk is correctly extracting intermediate values from the v13 pipeline without modifying production code, and ensuring the `--update-baselines` conftest plugin integrates cleanly with existing conftest.py.

**Primary recommendation:** Import `_run_surrogate_phases()` directly for the fast surrogate-only test (S1+S2); for the full 7-phase test, run the v13 script via subprocess with `--no-pde` for surrogate phases then a separate subprocess for P1-P2, parsing the saved CSV artifacts. Use `pytest_addoption` in conftest.py for `--update-baselines`.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Run the full 7-phase v13 pipeline end-to-end (S1-S5 + P1-P2), not just individual layer outputs
- Use PDE-generated synthetic targets (same approach as Phase 4, reuse pde_targets_cache.npz pattern)
- Compare one fresh run against saved baseline values (run-vs-baseline, not run-vs-run)
- Compare: final inferred parameters (k0_1, k0_2, alpha_1, alpha_2) + final loss + key intermediates (S1 output and P2 output)
- Baselines stored as JSON in `StudyResults/pipeline_reproducibility/regression_baselines.json`
- Custom `--update-baselines` pytest flag to regenerate baselines (test fails if baselines don't exist and flag not passed)
- Baseline JSON includes metadata: generation timestamp, git commit hash, Python/NumPy versions
- Baselines cover full-pipeline outputs only (Phases 2-4 tests already assert on their own artifacts)
- Solver-tolerance-aware: surrogate-only outputs rel=1e-10 (float64 deterministic), PDE-refined outputs rel=1e-4 (solver tolerance)
- Final parameters: rel=1e-4 (dominated by PDE refinement precision)
- Objective function values: relative tolerance with absolute floor (pytest.approx with rel=1e-4, abs=1e-8)
- On failure: detailed diff table showing parameter name, baseline value, current value, absolute diff, relative diff, tolerance, pass/fail
- Pass/fail only -- no warning band
- New file: `tests/test_pipeline_reproducibility.py`
- Two test classes: fast surrogate-only (S1+S2, no Firedrake) and full 7-phase (`@pytest.mark.slow` + `@skip_without_firedrake`)
- Single `--update-baselines` flag updates both surrogate-only and full-pipeline baselines
- Follows existing marker conventions from Phases 2-4

### Claude's Discretion
- How to invoke the 7-phase pipeline programmatically (import from v13 script or wrap subprocess)
- How to implement the --update-baselines conftest plugin
- Fixture organization for PDE target generation and caching
- Exact JSON schema for baseline storage
- How to extract S1 and P2 intermediate outputs from the pipeline

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PIP-01 | End-to-end v13 reproducibility test: same inputs produce same outputs across runs | Surrogate-only test (direct import, deterministic at rel=1e-10) + full 7-phase test (subprocess, deterministic within solver tolerance rel=1e-4). Seeds are already controlled (seed=42 in multistart, fixed initial guesses). |
| PIP-02 | Numerical regression baselines with saved reference values to catch future breakage | JSON baselines in `StudyResults/pipeline_reproducibility/regression_baselines.json` with `--update-baselines` conftest plugin. Diff table on failure for immediate diagnosis. |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | (existing) | Test framework | Already in use across all phases |
| numpy | (existing) | Numerical arrays and comparison | Already in use |
| json | stdlib | Baseline storage/loading | Simple, human-readable, diffable |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| subprocess | stdlib | Run v13 pipeline for PDE phases | Avoids Firedrake/PyTorch segfault (established Phase 4 pattern) |
| pytest.approx | (built-in) | Tolerance-aware comparison | All numerical assertions |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON baselines | npz baselines | JSON is human-readable, diffable in git, and includes metadata naturally; npz is more compact but opaque |
| subprocess for full pipeline | Direct import | Firedrake/PyTorch segfault makes direct import unsafe for PDE phases; subprocess is the established pattern |

## Architecture Patterns

### Recommended Project Structure
```
tests/
├── conftest.py                       # Add --update-baselines option here
├── test_pipeline_reproducibility.py  # NEW: PIP-01, PIP-02
StudyResults/
└── pipeline_reproducibility/
    └── regression_baselines.json     # Saved baselines
```

### Pattern 1: Conftest Plugin for --update-baselines

**What:** Add `pytest_addoption` to existing conftest.py to register `--update-baselines` flag. Expose via fixture.
**When to use:** Any test that needs to regenerate baselines.
**Example:**
```python
# In tests/conftest.py -- add to existing file

def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Regenerate regression baselines instead of comparing against them",
    )

@pytest.fixture()
def update_baselines(request):
    return request.config.getoption("--update-baselines")
```

### Pattern 2: Surrogate-Only Test (Fast, Direct Import)

**What:** Import `_run_surrogate_phases()` from v13 script and run S1+S2 with fixed seeds. Compare against baselines at rel=1e-10.
**When to use:** Fast reproducibility check without Firedrake.
**Key insight:** The v13 script's `_run_surrogate_phases()` function returns a dict with all phase results including k0, alpha, loss for each phase. It can be called with `args.surr_strategy = "joint"` to run only S1+S2 (the fast path). Surrogate model predictions are deterministic (float64 math, no PDE solver).

```python
# Pseudo-pattern for surrogate-only test
from scripts.surrogate.Infer_BVMaster_charged_v13_ultimate import _run_surrogate_phases

# Load nn-ensemble, generate surrogate targets, call _run_surrogate_phases
# with strategy="joint" (S1+S2 only), fixed seeds, deterministic inputs
# Compare s1_alpha, s2 k0/alpha/loss against baselines at rel=1e-10
```

### Pattern 3: Full Pipeline Test (Subprocess)

**What:** Run the full v13 pipeline via subprocess, parse output CSV from `StudyResults/master_inference_v13/master_comparison_v13.csv`.
**When to use:** Full 7-phase reproducibility with PDE refinement.
**Key insight:** The v13 script saves `master_comparison_v13.csv` with all phase results. The subprocess pattern avoids the Firedrake/PyTorch segfault. Use `--noise-percent 0` and fixed seed for deterministic targets.

```python
result = subprocess.run(
    [sys.executable, "scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py",
     "--noise-percent", "0", "--noise-seed", "42"],
    capture_output=True, text=True, timeout=900,
    cwd=_ROOT,
)
# Parse master_comparison_v13.csv for all phase results
# Compare against baselines at rel=1e-4
```

### Pattern 4: Baseline JSON Schema

**What:** Structured JSON with metadata + results sections.
**Example:**
```json
{
    "metadata": {
        "generated_at": "2026-03-09T12:00:00Z",
        "git_commit": "abc123",
        "python_version": "3.11.5",
        "numpy_version": "1.26.4"
    },
    "surrogate_only": {
        "s1_alpha": [0.35, 0.25],
        "s2_k0": [0.01, 0.001],
        "s2_alpha": [0.35, 0.25],
        "s2_loss": 1.23e-6
    },
    "full_pipeline": {
        "s1_alpha": [0.35, 0.25],
        "p2_k0": [0.01, 0.001],
        "p2_alpha": [0.35, 0.25],
        "p2_loss": 1.23e-6,
        "final_k0": [0.01, 0.001],
        "final_alpha": [0.35, 0.25],
        "final_loss": 1.23e-6
    }
}
```

### Pattern 5: Diff Table on Failure

**What:** Format a human-readable comparison table when values drift beyond tolerance.
**Example output:**
```
Parameter      | Baseline      | Current       | Abs Diff  | Rel Diff  | Tol       | Pass
k0_1           | 1.234567e-02  | 1.234890e-02  | 3.23e-06  | 2.62e-04  | 1.00e-04  | FAIL
k0_2           | 5.678901e-04  | 5.678901e-04  | 0.00e+00  | 0.00e+00  | 1.00e-04  | PASS
alpha_1        | 3.500000e-01  | 3.500000e-01  | 0.00e+00  | 0.00e+00  | 1.00e-04  | PASS
alpha_2        | 2.500000e-01  | 2.500000e-01  | 0.00e+00  | 0.00e+00  | 1.00e-04  | PASS
```

### Anti-Patterns to Avoid
- **Importing Firedrake in fast tests:** The Firedrake/PyTorch segfault will crash the test process. Use subprocess for any PDE work.
- **Run-vs-run comparison:** Non-deterministic due to floating-point ordering, parallel worker scheduling. Always compare against saved baseline.
- **Tight tolerances on PDE outputs:** PDE solver convergence is iterative with rel=1e-4 tolerance. Asserting tighter than this will cause flaky tests.
- **Modifying the v13 production script:** The script is the system under test. Don't add test hooks to it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tolerance comparison | Custom abs/rel logic | `pytest.approx(val, rel=..., abs=...)` | Handles edge cases (zero values, NaN) correctly |
| Git commit hash | Shell out to git | `subprocess.check_output(["git", "rev-parse", "HEAD"])` | One-liner, already a pattern in the codebase |
| CSV parsing | Manual string splitting | `csv.DictReader` | Handles quoting, escaping correctly |
| JSON serialization with numpy | Custom encoder | `default=lambda o: o.tolist() if hasattr(o, 'tolist') else o` | Simple numpy array serialization |

## Common Pitfalls

### Pitfall 1: Firedrake/PyTorch Segfault
**What goes wrong:** Importing Firedrake (PETSc) in the same process as PyTorch corrupts memory, causing segfaults in torch.tensor operations.
**Why it happens:** PETSc and PyTorch both use OpenMP/MKL with conflicting library states.
**How to avoid:** Use subprocess for any code path that touches Firedrake. The surrogate-only test can safely import PyTorch (no Firedrake needed). The full pipeline test MUST use subprocess.
**Warning signs:** Segfault during test collection or fixture setup.

### Pitfall 2: Non-Deterministic Multistart LHS
**What goes wrong:** Multistart uses Latin Hypercube Sampling which is seed-dependent.
**Why it happens:** `LatinHypercube(d=4, seed=seed)` in `Surrogate/multistart.py` is deterministic given same seed, but if the seed or grid size changes, all results change.
**How to avoid:** Fix `seed=42` and `n_points=20000` in test invocations (matches production defaults). These are already the defaults in MultiStartConfig.
**Warning signs:** Surrogate-only baselines changing without code changes.

### Pitfall 3: PDE Target Caching Stale Data
**What goes wrong:** The v13 script caches PDE targets in `.npz` files. If the cache exists from a different voltage grid or parameter set, stale targets are used.
**Why it happens:** Cache key is based on voltage grid shape and values, but not on parameter values.
**How to avoid:** For reproducibility tests, use noise_percent=0 with a known seed. The cache will be deterministic given the same voltage grid and true parameters (which are constants in `_bv_common.py`).
**Warning signs:** Baseline values changing after first run but stable on subsequent runs.

### Pitfall 4: Baselines Missing on First Run
**What goes wrong:** Test fails because baselines don't exist and `--update-baselines` wasn't passed.
**Why it happens:** Expected behavior by design -- forces explicit baseline generation.
**How to avoid:** Clear error message: "Baselines not found. Run with --update-baselines to generate."
**Warning signs:** N/A -- this is intentional.

### Pitfall 5: CSV Parsing Floating-Point Precision
**What goes wrong:** Values parsed from CSV lose precision compared to in-memory values.
**Why it happens:** The v13 script writes k0 with `:.8e` and alpha with `:.6f` format, which limits precision.
**How to avoid:** For the surrogate-only test, use direct import (no CSV parsing). For the full pipeline test, the CSV precision (8 significant digits for k0, 6 decimal places for alpha) is well within the rel=1e-4 tolerance.
**Warning signs:** Surrogate-only test failing at rel=1e-10 if using CSV output.

## Code Examples

### Loading Baselines
```python
def _load_baselines(path: str) -> dict:
    """Load regression baselines from JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _save_baselines(path: str, data: dict) -> None:
    """Save regression baselines to JSON file with metadata."""
    import subprocess as sp
    data["metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": sp.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)
```

### Diff Table Formatter
```python
def _format_diff_table(name_val_pairs: list[tuple[str, float, float, float]]) -> str:
    """Format a comparison table: [(name, baseline, current, tolerance), ...]"""
    lines = []
    header = f"{'Parameter':<15} | {'Baseline':>14} | {'Current':>14} | {'Abs Diff':>10} | {'Rel Diff':>10} | {'Tol':>10} | Pass"
    lines.append(header)
    lines.append("-" * len(header))
    any_fail = False
    for name, baseline, current, tol in name_val_pairs:
        abs_diff = abs(current - baseline)
        rel_diff = abs_diff / max(abs(baseline), 1e-30)
        passed = rel_diff <= tol
        if not passed:
            any_fail = True
        lines.append(
            f"{name:<15} | {baseline:>14.6e} | {current:>14.6e} | {abs_diff:>10.2e} | {rel_diff:>10.2e} | {tol:>10.2e} | {'PASS' if passed else 'FAIL'}"
        )
    return "\n".join(lines), any_fail
```

### Invoking Surrogate-Only Pipeline (Direct Import)
```python
# Key: create a mock args object with the right attributes
from types import SimpleNamespace

args = SimpleNamespace(
    surr_strategy="joint",  # S1+S2 only
    # ... other needed attributes from argparse
)

result = _run_surrogate_phases(
    surrogate=model,
    model_label="nn-ensemble",
    args=args,
    target_cd_surr=target_cd,
    target_pc_surr=target_pc,
    target_cd_full=target_cd,
    target_pc_full=target_pc,
    all_eta=phi_applied,
    eta_shallow=eta_shallow,
    initial_k0_guess=[0.005, 0.0005],
    initial_alpha_guess=[0.4, 0.3],
    true_k0_arr=np.array([K0_HAT_R1, K0_HAT_R2]),
    true_alpha_arr=np.array([ALPHA_R1, ALPHA_R2]),
    secondary_weight=1.0,
)
# result["s1_alpha"], result["surr_best_k0"], etc.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| In-process PDE + PyTorch | Subprocess PDE isolation | Phase 4 (04-03) | Avoids segfault; established pattern to reuse |
| Surrogate-on-surrogate targets | PDE-generated targets | Phase 4 (04-02) | Eliminates inverse crime; same approach for Phase 5 |

## Open Questions

1. **How to extract S1 intermediate from full pipeline subprocess?**
   - What we know: The CSV output has S1 results as a row. `_run_surrogate_phases()` returns `s1_k0`, `s1_alpha`.
   - What's unclear: Whether the subprocess CSV reliably contains S1 row vs being overwritten.
   - Recommendation: Parse `master_comparison_v13.csv` which includes all phase rows. For surrogate-only test, use direct import which gives exact values.

2. **Will _run_surrogate_phases import cleanly from test code?**
   - What we know: The function is module-level in the v13 script. The script does `setup_firedrake_env()` at import time (sets env vars only, does NOT import Firedrake).
   - What's unclear: Whether importing the v13 module triggers any side effects beyond env var setup.
   - Recommendation: Test the import path. The module-level imports are: numpy, scipy.optimize, Surrogate.io, Surrogate.objectives, Surrogate.cascade, Surrogate.multistart -- all safe (no Firedrake). The Firedrake imports happen inside `main()` and inside `_generate_targets_with_pde()` only.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | none (default discovery) |
| Quick run command | `pytest tests/test_pipeline_reproducibility.py -m "not slow" -x` |
| Full suite command | `pytest tests/test_pipeline_reproducibility.py -x --tb=short` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PIP-01 | Surrogate-only reproducibility (S1+S2 deterministic) | unit | `pytest tests/test_pipeline_reproducibility.py::TestSurrogateReproducibility -x` | No -- Wave 0 |
| PIP-01 | Full 7-phase pipeline reproducibility | integration | `pytest tests/test_pipeline_reproducibility.py::TestFullPipelineReproducibility -m slow -x` | No -- Wave 0 |
| PIP-02 | Baselines exist and are compared | unit | `pytest tests/test_pipeline_reproducibility.py -x` | No -- Wave 0 |
| PIP-02 | --update-baselines regenerates baselines | unit | `pytest tests/test_pipeline_reproducibility.py --update-baselines -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_pipeline_reproducibility.py -m "not slow" -x`
- **Per wave merge:** `pytest tests/ -x --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline_reproducibility.py` -- covers PIP-01, PIP-02
- [ ] `tests/conftest.py` update -- add `--update-baselines` option and fixture
- [ ] `StudyResults/pipeline_reproducibility/regression_baselines.json` -- generated on first `--update-baselines` run

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` (1257 lines) -- pipeline structure, `_run_surrogate_phases()` API, CSV output format
- Direct code inspection of `tests/test_inverse_verification.py` (1136 lines) -- subprocess PDE pattern, fixture structure, `pde_targets_cache.npz` caching
- Direct code inspection of `tests/conftest.py` -- existing fixtures, `skip_without_firedrake`, `FIREDRAKE_AVAILABLE`
- Direct code inspection of `Surrogate/multistart.py` -- LHS seed control, `MultiStartConfig.seed=42`
- Direct code inspection of `scripts/_bv_common.py` -- true parameter values, `setup_firedrake_env()` (env vars only, no imports)

### Secondary (MEDIUM confidence)
- pytest documentation for `pytest_addoption` and custom flags -- well-established pattern

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in use in the project
- Architecture: HIGH - direct code inspection of pipeline, established patterns from Phase 4
- Pitfalls: HIGH - Firedrake/PyTorch segfault is documented and solved in Phase 4; other pitfalls from code inspection

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable domain, no external dependencies changing)
