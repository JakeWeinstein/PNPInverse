# Phase 7: Baseline Diagnostics - Research

**Researched:** 2026-03-10
**Domain:** Statistical diagnostics for inverse problem pipelines (multi-seed robustness, profile likelihood identifiability, sensitivity analysis)
**Confidence:** HIGH

## Summary

Phase 7 is a measurement-only phase: run the existing v13 pipeline across noise seeds, assess parameter identifiability via profile likelihood, and visualize sensitivity of observables to parameters. No pipeline modifications are made. The codebase already contains all necessary infrastructure: the v13 script accepts `--noise-seed` and `--noise-percent` CLI args, the FluxCurve PDE objective supports joint control mode with warm-starting, and an existing profile likelihood script (`scripts/inference/Infer_BVProfileLikelihood_charged.py`) provides a reference implementation (though per CONTEXT.md decisions, a new implementation is required).

The three diagnostic tools map cleanly to three scripts: (1) a multi-seed wrapper that invokes v13 per seed and aggregates results, (2) a PDE-only profile likelihood script with 30 grid points per parameter, and (3) a sensitivity visualization script combining 1D parameter sweeps with a Jacobian heatmap. All outputs go to `StudyResults/v14/` with JSON metadata sidecars for AUDT-04 compliance.

**Primary recommendation:** Build three standalone scripts following existing codebase patterns (frozen dataclass configs, CSV + PNG output, `[tag]` print logging). Reuse v13 infrastructure directly -- the multi-seed wrapper can import and call v13's `main()` machinery or invoke it via subprocess. The profile likelihood script should use `run_bv_multi_observable_flux_curve_inference` in joint control mode with one parameter fixed at each grid point. The sensitivity script should use `solve_bv_curve_points_with_warmstart` directly for I-V curve generation at perturbed parameter values.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 20 noise seeds, sequential integers 0-19
- Full 7-phase v13 pipeline (S1-S5 + P1-P2) per seed
- New wrapper script that calls v13 per seed, collects results, and generates summary CSV -- keeps v13 script unchanged
- 2% noise level throughout
- New profile likelihood implementation (do not reuse existing scripts)
- PDE-only profile likelihood -- no surrogate profiles
- 30 profile points per parameter (k0_1, k0_2, alpha_1, alpha_2)
- Chi-squared 95% CI threshold (delta-chi2 = 3.84 for 1 DOF) for identifiability determination
- ~120 PDE re-optimizations total across 4 parameters
- Both 1D parameter sweeps AND Jacobian heatmap
- 1D sweeps: 5 values per parameter (e.g., 0.5x, 0.75x, 1x, 1.5x, 2x of true value)
- Extended voltage range beyond v13 default
- Use voltage continuation/warm-starting and bridge points for extended voltages
- Allow more SNES iterations or smaller dt for convergence at extreme voltages
- Jacobian heatmap via central finite differences (h=1e-5)
- Summary statistics: median relative error, IQR (25th/75th percentile), worst-case (max) per parameter
- Auto-generated plots: box plots of relative error per parameter across seeds, per-seed scatter
- JSON metadata sidecar for each diagnostic tool (AUDT-04 compliance)
- Output directory: `StudyResults/v14/` with sub-folders

### Claude's Discretion
- Exact parameter sweep ranges for 1D sensitivity (multiplicative factors around true values)
- Extended voltage range bounds (how far beyond v13 default to go)
- Profile likelihood parameter range selection
- Plot styling (colors, layout, figure sizes)
- Multi-seed wrapper script architecture details
- JSON metadata schema details beyond the required fields

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DIAG-01 | Run v13 pipeline across 10+ noise seeds at 2% noise, report per-parameter median/worst-case relative error | Multi-seed wrapper calls v13 with `--noise-seed` for seeds 0-19, parses phase_results dict, computes median/IQR/max from numpy |
| DIAG-02 | Profile likelihood analysis for each of k0_1, k0_2, alpha_1, alpha_2 to determine practical identifiability | New PDE-only profile likelihood with 30 grid points, joint re-optimization with one parameter fixed, delta-chi2=3.84 threshold |
| DIAG-03 | Extended voltage sweep visualization of total and peroxide current across parameter values | 1D sweeps at 5 multiplicative factors + Jacobian heatmap via central FD (h=1e-5), extended voltage range with warm-starting |
| AUDT-04 | Every new component must have justification entry documented | JSON metadata sidecar per diagnostic tool with tool_name, justification_type, reference, rationale fields |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | existing | Array operations, statistics (median, percentile) | Already used throughout codebase |
| scipy.optimize | existing | L-BFGS-B optimizer for profile likelihood re-optimizations | Already used in v13 pipeline |
| matplotlib | existing | Box plots, scatter plots, heatmaps, profile likelihood curves | Already used for all visualization |
| json | stdlib | JSON metadata sidecar files for AUDT-04 | Standard library, no dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| csv | stdlib | Read/write result CSVs | All summary output |
| subprocess | stdlib | Invoke v13 script per seed (alternative to direct import) | If direct import has Firedrake memory issues |
| argparse | stdlib | CLI for all three diagnostic scripts | Consistent with codebase pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| subprocess v13 invocation | Direct import of v13 functions | Direct import avoids subprocess overhead but may cause Firedrake memory accumulation across 20 seeds; subprocess is safer for long-running batch |
| Central FD Jacobian | Adjoint-based gradient | FD is consistent with codebase convention (h=1e-5) and simpler; adjoint would be faster but not implemented for this use case |

**Installation:**
No new packages needed -- everything is already in the environment.

## Architecture Patterns

### Recommended Project Structure
```
scripts/studies/
    run_multi_seed_v13.py          # DIAG-01: multi-seed wrapper
    profile_likelihood_pde.py      # DIAG-02: profile likelihood
    sensitivity_visualization.py   # DIAG-03: sweeps + heatmap

StudyResults/v14/
    multi_seed/
        seed_results.csv           # Per-seed, per-parameter results
        summary_statistics.csv     # Median, IQR, worst-case
        boxplot_errors.png
        scatter_per_seed.png
        metadata.json              # AUDT-04
    profile_likelihood/
        profile_k0_1.csv + .png
        profile_k0_2.csv + .png
        profile_alpha_1.csv + .png
        profile_alpha_2.csv + .png
        identifiability_summary.csv
        metadata.json              # AUDT-04
    sensitivity/
        sweep_k0_1.csv + .png
        sweep_k0_2.csv + .png
        sweep_alpha_1.csv + .png
        sweep_alpha_2.csv + .png
        jacobian_heatmap.csv + .png
        metadata.json              # AUDT-04
```

### Pattern 1: Multi-Seed Wrapper via Subprocess
**What:** A wrapper script that runs v13 20 times with different `--noise-seed` values, collects results, and generates summary statistics.
**When to use:** DIAG-01 implementation.
**Why subprocess:** Each v13 run takes ~7 minutes and involves Firedrake PDE solves. Running 20 seeds sequentially in-process risks memory accumulation from Firedrake's compiled kernel caches. Subprocess isolation ensures clean state per seed.
**Example:**
```python
import subprocess
import csv
import json
import numpy as np

SEEDS = list(range(20))
TRUE_PARAMS = {"k0_1": K0_HAT_R1, "k0_2": K0_HAT_R2, "alpha_1": ALPHA_R1, "alpha_2": ALPHA_R2}

for seed in SEEDS:
    cmd = [
        sys.executable,
        "scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py",
        "--noise-seed", str(seed),
        "--noise-percent", "2.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    # Parse result from stdout or from output CSV
```

### Pattern 2: PDE Profile Likelihood with Fixed Parameter
**What:** For each of 4 parameters, fix at 30 grid points, re-optimize remaining 3 parameters using PDE joint objective.
**When to use:** DIAG-02 implementation.
**Key insight:** The existing profile likelihood script fixes one parameter type (k0 or alpha) and optimizes the other type. The new implementation should fix one specific parameter (e.g., k0_1) and jointly optimize k0_2, alpha_1, alpha_2. This requires `control_mode="joint"` with bounds that pin the profiled parameter.
**Implementation approach:** Use `run_bv_multi_observable_flux_curve_inference` with modified bounds: set lower=upper=fixed_value for the profiled parameter. This naturally reduces the optimizer to 3 free dimensions.
**Example:**
```python
# Profile k0_1: fix k0_1, optimize k0_2, alpha_1, alpha_2
for fixed_val in grid_k0_1:
    # Set bounds so k0_1 is pinned
    # In log-space: log10(fixed_val) for both lower and upper of k0_1
    request = BVFluxCurveInferenceRequest(
        ...
        control_mode="joint",
        k0_lower=[fixed_val * 0.999, 1e-8],  # pin k0_1
        k0_upper=[fixed_val * 1.001, 100.0],  # pin k0_1
        initial_guess=[fixed_val, best_k0_2],
        initial_alpha_guess=[best_alpha_1, best_alpha_2],
        ...
    )
    result = run_bv_multi_observable_flux_curve_inference(request)
```

### Pattern 3: Sensitivity Sweep with Warm-Starting
**What:** For each parameter, evaluate I-V curves at 5 multiplicative factors, using voltage continuation for stability.
**When to use:** DIAG-03 implementation.
**Key insight:** At extreme voltages, the PDE solver needs warm-starting from adjacent voltage solutions. The existing `solve_bv_curve_points_with_warmstart` handles this automatically. For extended voltage range, simply pass a larger voltage array.
**Example:**
```python
from FluxCurve.bv_point_solve import solve_bv_curve_points_with_warmstart

# Extended voltage range: go beyond v13's [-46.5, +5.0] range
eta_extended = np.array([
    +5.0, +3.0, +1.0, -0.5, -1.0, -2.0, -3.0, -5.0, -8.0,
    -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -46.5,
    -55.0, -65.0, -75.0,  # extended beyond v13
])
eta_extended = np.sort(eta_extended)[::-1]  # decreasing order for warm-start

for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
    perturbed_k0 = [K0_HAT_R1 * factor, K0_HAT_R2]
    points = solve_bv_curve_points_with_warmstart(
        base_solver_params=base_sp,
        steady=steady,
        phi_applied_values=eta_extended,
        target_flux=np.zeros_like(eta_extended),  # dummy
        k0_values=list(perturbed_k0),
        alpha_values=[ALPHA_R1, ALPHA_R2],
        ...
    )
```

### Pattern 4: JSON Metadata Sidecar (AUDT-04)
**What:** Each diagnostic script writes a `metadata.json` alongside its outputs documenting the tool's justification.
**When to use:** Every script in this phase.
**Example:**
```python
metadata = {
    "tool_name": "Multi-Seed Pipeline Robustness Assessment",
    "phase": "07-baseline-diagnostics",
    "requirement": "DIAG-01",
    "justification_type": "empirical",
    "reference": "Standard practice in inverse problems literature: test parameter recovery across noise realizations to assess estimator variance",
    "rationale": "Running across 20 noise seeds at 2% noise quantifies the pipeline's sensitivity to noise realization and identifies parameters with high variance (potential identifiability issues)",
    "parameters": {
        "n_seeds": 20,
        "noise_percent": 2.0,
        "pipeline_version": "v13"
    },
    "generated": "2026-03-10T...",
}
with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
```

### Anti-Patterns to Avoid
- **Modifying v13 pipeline:** This phase is measurement-only. The wrapper must call v13 unchanged.
- **Using surrogate for profile likelihood:** The decision explicitly requires PDE-only profiles for gold-standard identifiability assessment.
- **Running all 20 seeds in-process:** Firedrake kernel caches accumulate; use subprocess isolation.
- **Ignoring solver convergence at extended voltages:** Must use warm-starting and allow more SNES iterations; otherwise solver will fail at extreme cathodic voltages.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDE I-V curve evaluation | Custom forward solve loop | `solve_bv_curve_points_with_warmstart` | Handles warm-starting, bridge points, recovery strategies automatically |
| Multi-observable objective | Custom loss computation | `run_bv_multi_observable_flux_curve_inference` | Handles CD + PC weighting, log-space, bounds, L-BFGS-B wrapping |
| Noise injection | Custom noise code | `add_percent_noise` from `Forward.steady_state` | Consistent with codebase convention, handles RMS-based sigma |
| Solver parameter construction | Manual dict building | `make_bv_solver_params` from `scripts._bv_common` | Handles nondim, BV BC config, SNES opts correctly |
| Recovery from SNES failures | Custom retry logic | `make_recovery_config` / `ForwardRecoveryConfig` | Implements 6-attempt strategy with line search cycling, tolerance relaxation |

**Key insight:** The entire PDE forward solve + objective infrastructure is mature and well-tested. The diagnostic scripts are thin wrappers around existing FluxCurve and Forward APIs.

## Common Pitfalls

### Pitfall 1: Firedrake Memory Accumulation Across Seeds
**What goes wrong:** Running 20 v13 instances in the same process causes Firedrake's JIT kernel cache and PETSc objects to accumulate, eventually causing OOM or slowdown.
**Why it happens:** Firedrake compiles kernels on first use and caches them in memory. Each seed creates new mesh/function space objects.
**How to avoid:** Run each seed as a separate subprocess. The v13 script already supports `--noise-seed` CLI arg.
**Warning signs:** Increasing per-seed runtime, or crash on seed 10+.

### Pitfall 2: PDE Solver Failure at Extended Voltages
**What goes wrong:** SNES divergence at very cathodic voltages (beyond eta=-50) due to numerical stiffness.
**Why it happens:** At extreme overpotentials, Butler-Volmer exponentials become very large, creating ill-conditioned systems.
**How to avoid:** Use voltage warm-starting (decreasing eta order), bridge points (already in `bv_point_solve`), and allow more SNES iterations (`snes_max_it=400+`) via `SNES_OPTS_CHARGED` override. Use `ForwardRecoveryConfig` with generous `max_it_cap=600`.
**Warning signs:** NaN in simulated flux values, `ConvergenceError` in point solve output.

### Pitfall 3: Profile Likelihood Range Too Narrow
**What goes wrong:** Profile likelihood curve appears parabolic (identifiable) but the range didn't explore far enough to see the flat region.
**Why it happens:** Choosing grid bounds too close to the MLE.
**How to avoid:** For k0 parameters (log-scale), use at least 2 orders of magnitude around the MLE. For alpha parameters (linear scale 0-1), span at least [0.1, 0.9]. Check that the profile curve rises above the chi-squared threshold at both ends.
**Warning signs:** Profile curve has not crossed the 3.84 threshold at the grid boundaries.

### Pitfall 4: Target Cache Invalidation Across Seeds
**What goes wrong:** All seeds get the same target data because the cache key doesn't include the noise seed.
**Why it happens:** The existing `_target_cache_path` in v13 caches the *clean* (noise-free) PDE solution, which is correct -- the clean solution is the same for all seeds; noise is applied after loading the cache. This is actually correct behavior.
**How to avoid:** Understand that the cache stores clean targets. The noise seed only affects the noise added *after* cache load. Verify by checking that different seeds produce different `target_cd` values but identical pre-noise values.

### Pitfall 5: Incorrect Chi-Squared Threshold Application
**What goes wrong:** Comparing raw objective values instead of likelihood ratios.
**Why it happens:** The objective function J(x) is a sum-of-squares, not a negative log-likelihood. The chi-squared threshold applies to 2*(L_max - L_profile), which for Gaussian errors equals delta_J / sigma^2.
**How to avoid:** The profile likelihood threshold should be applied as: parameter is identifiable if min(J_profile) - J_global_min > threshold, where threshold = 0.5 * delta_chi2 * sigma^2 for known noise variance, or simply plot the normalized profile chi2 = (J_profile - J_min) / (J_min / (n_obs - n_params)) and compare against 3.84.
**Warning signs:** All parameters appear identifiable or all appear non-identifiable regardless of the data.

## Code Examples

### Multi-Seed Result Parsing
```python
# After running v13 per seed, parse the master_comparison_v13.csv
# The v13 script writes phase results including P2 (final) k0 and alpha values.
# Key output: the last row of the CSV contains P2 (PDE full cathodic) results.

import csv
import numpy as np

def parse_v13_output(csv_path):
    """Parse v13 master_comparison CSV and extract P2 final results."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"].startswith("P2"):
                return {
                    "k0_1": float(row["k0_1"]),
                    "k0_2": float(row["k0_2"]),
                    "alpha_1": float(row["alpha_1"]),
                    "alpha_2": float(row["alpha_2"]),
                    "k0_1_err_pct": float(row["k0_1_err_pct"]),
                    "k0_2_err_pct": float(row["k0_2_err_pct"]),
                    "alpha_1_err_pct": float(row["alpha_1_err_pct"]),
                    "alpha_2_err_pct": float(row["alpha_2_err_pct"]),
                }
    return None
```

### Profile Likelihood Grid Construction
```python
# For k0 parameters (log-scale): span ~2 orders of magnitude around true value
# For alpha parameters (linear scale): span reasonable physical range

import numpy as np

# k0_1 profile: true value = K0_HAT_R1 = 1.2631578e-03
# Log-space grid: 30 points from 0.01x to 100x true value
k0_1_grid = np.logspace(
    np.log10(K0_HAT_R1 * 0.01),
    np.log10(K0_HAT_R1 * 100.0),
    30
)

# alpha_1 profile: true value = 0.627
# Linear grid from 0.1 to 0.95 (physical bounds)
alpha_1_grid = np.linspace(0.1, 0.95, 30)
```

### Chi-Squared Identifiability Check
```python
# Profile likelihood identifiability determination
# Based on Raue et al. (2009) and Wilks' theorem

DELTA_CHI2_95 = 3.84  # chi2(1) 95th percentile, 1 DOF

def assess_identifiability(profile_losses, global_min_loss, n_obs, n_params):
    """Determine if parameter is practically identifiable from profile likelihood.

    For least-squares with Gaussian noise:
    chi2_profile = (J_profile - J_min) * n_obs / J_min

    Identifiable if chi2_profile exceeds DELTA_CHI2_95 on both sides.
    """
    # Normalize to chi-squared scale
    sigma2_hat = global_min_loss / (n_obs - n_params)  # estimated noise variance
    chi2_profile = (np.array(profile_losses) - global_min_loss) / sigma2_hat

    # Find minimum of profile (should be near global_min_loss)
    min_idx = np.argmin(chi2_profile)

    # Check if threshold is exceeded on both sides
    left_exceeds = np.any(chi2_profile[:min_idx] > DELTA_CHI2_95)
    right_exceeds = np.any(chi2_profile[min_idx:] > DELTA_CHI2_95)

    identifiable = left_exceeds and right_exceeds

    return {
        "identifiable": identifiable,
        "left_bounded": left_exceeds,
        "right_bounded": right_exceeds,
        "chi2_profile": chi2_profile.tolist(),
    }
```

### Extended Voltage Range with Solver Hardening
```python
# Extended voltage range for sensitivity analysis
# v13 default range: [+5.0, ..., -46.5]
# Extended range: push to -75 or -80 with more SNES iterations

# Solver options for extended voltages
SNES_OPTS_EXTENDED = {
    **SNES_OPTS_CHARGED,
    "snes_max_it": 400,  # more iterations for extreme voltages
}

# Smaller dt for better convergence at extreme voltages
dt_extended = 0.25  # half of standard 0.5
max_ss_steps_extended = 200  # double steps to compensate
t_end_extended = dt_extended * max_ss_steps_extended

base_sp_extended = make_bv_solver_params(
    eta_hat=0.0,
    dt=dt_extended,
    t_end=t_end_extended,
    species=FOUR_SPECIES_CHARGED,
    snes_opts=SNES_OPTS_EXTENDED,
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-seed testing | Multi-seed robustness assessment | Standard practice | Reveals variance across noise realizations; single seed can be misleading |
| Fisher Information Matrix only | Profile likelihood for identifiability | Raue et al. 2009 | FIM assumes local linearity; profile likelihood is exact for practical identifiability |
| Visual inspection of I-V curves | Quantitative Jacobian heatmaps | Common in sensitivity analysis | Heatmap shows which voltage regions carry information about which parameters |

**Key references:**
- Profile likelihood for practical identifiability: Raue et al., Bioinformatics 2009 -- established the profile likelihood approach for determining structural and practical identifiability in dynamical models
- Chi-squared threshold for profile confidence intervals: Based on Wilks' theorem -- the likelihood ratio statistic is asymptotically chi-squared distributed

## Open Questions

1. **Extended voltage range bounds**
   - What we know: v13 goes to eta=-46.5. The PDE solver uses warm-starting and bridge points.
   - What's unclear: How far can we push the voltage before the solver consistently fails even with recovery strategies?
   - Recommendation: Start with eta=-60, extend to -75 if convergence holds. Log any failed voltage points and report the usable range. Claude's discretion area.

2. **Profile likelihood grid range for k0_2**
   - What we know: k0_2 is the highest-risk parameter for identifiability (per STATE.md blockers). Its true value (K0_HAT_R2 = 5.2631e-05) is small.
   - What's unclear: Whether the profile will be flat (non-identifiable) or have a clear minimum.
   - Recommendation: Use a wide range (0.001x to 1000x true value, 30 points log-spaced) to ensure we capture the full profile shape. This is the key diagnostic result.

3. **Multi-seed wrapper: subprocess vs direct import**
   - What we know: v13 takes ~7 minutes per seed. 20 seeds = ~140 minutes total.
   - What's unclear: Whether Firedrake memory accumulation is a real problem for 20 sequential runs.
   - Recommendation: Use subprocess for safety. If runtime is too long, consider parallelizing across seeds (though Firedrake is single-process).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | existing pytest configuration |
| Quick run command | `python -m pytest tests/ -x -q --timeout=60` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DIAG-01 | Multi-seed wrapper produces CSV with expected columns and 20 rows | smoke | Manual: run wrapper with 1-2 seeds, check CSV format | No -- Wave 0 |
| DIAG-02 | Profile likelihood produces 30-point profiles for 4 parameters | smoke | Manual: run profile for 1 parameter with 3 grid points | No -- Wave 0 |
| DIAG-03 | Sensitivity plots generated for 4 parameters + Jacobian heatmap | smoke | Manual: run sensitivity with 2 voltage points, 2 factors | No -- Wave 0 |
| AUDT-04 | Each script produces metadata.json with required fields | unit | `python -m pytest tests/test_diagnostic_metadata.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** Validate metadata JSON schema, check CSV output format
- **Per wave merge:** Run each diagnostic script with reduced parameters (few seeds, few grid points) to verify end-to-end
- **Phase gate:** Full runs complete with all outputs in `StudyResults/v14/`

### Wave 0 Gaps
- [ ] `tests/test_diagnostic_metadata.py` -- validates JSON metadata schema for AUDT-04 compliance
- [ ] Smoke test helper: a fixture that creates mock v13 output CSV for testing the aggregation logic without running the full pipeline

*(Note: Full DIAG-01/02/03 validation is inherently manual due to ~2.5 hour runtime for multi-seed and ~2 hour runtime for profile likelihood. Tests should validate format and schema, not full numerical results.)*

## Sources

### Primary (HIGH confidence)
- Existing codebase: `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` -- v13 pipeline with `--noise-seed`, `--noise-percent` CLI args
- Existing codebase: `scripts/inference/Infer_BVProfileLikelihood_charged.py` -- reference profile likelihood implementation (not reused, but pattern reference)
- Existing codebase: `FluxCurve/bv_point_solve/` -- warm-started PDE curve evaluation
- Existing codebase: `scripts/_bv_common.py` -- shared constants, solver param factories
- Existing codebase: `Surrogate/objectives.py` -- objective function patterns with FD gradient (h=1e-5)

### Secondary (MEDIUM confidence)
- [Profile-Wise Analysis: PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011515) -- profile likelihood workflow
- [Raue et al. (2009) Structural and Practical Identifiability](https://pubmed.ncbi.nlm.nih.gov/19505944/) -- foundational profile likelihood identifiability paper
- [Wilks' theorem](https://en.wikipedia.org/wiki/Wilks'_theorem) -- chi-squared threshold justification (delta-chi2 = 3.84 for 95% CI, 1 DOF)
- [Practical identifiability of electrochemical P2D models](https://link.springer.com/article/10.1007/s10800-021-01579-5) -- identifiability in electrochemical context

### Tertiary (LOW confidence)
- Extended voltage range feasibility: based on codebase inspection of bridge point logic and recovery config; actual convergence limits need empirical testing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use in codebase
- Architecture: HIGH -- patterns directly observed in existing scripts (v13, profile likelihood, FluxCurve)
- Pitfalls: HIGH -- memory accumulation and solver convergence at extreme voltages are well-known issues in this codebase
- Profile likelihood method: MEDIUM -- chi-squared threshold application needs care in mapping from sum-of-squares objective to chi-squared statistic

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable -- no external dependencies changing)
