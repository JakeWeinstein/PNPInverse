# Architecture Patterns

**Domain:** Surrogate-assisted inverse parameter recovery pipeline redesign (v13 -> v14)
**Researched:** 2026-03-09
**Confidence:** HIGH (based on direct codebase analysis of v13 pipeline, v7 PDE-only pipeline, and all supporting modules)

## Current Architecture (v13)

The v13 pipeline is a 7-phase sequential script (`Infer_BVMaster_charged_v13_ultimate.py`) that chains surrogate-only phases (S1-S5) into PDE refinement phases (P1-P2). It lives entirely in `scripts/surrogate/` as a ~1100-line monolithic script.

```
S1: Alpha-only init (surrogate)          ~0.1s   -> alpha warm-start
S2: Joint L-BFGS-B (surrogate, shallow)  ~0.5s   -> 4-param estimate
S3: Cascade 3-pass (surrogate)           ~3-8s   -> 4-param estimate
S4: MultiStart 20K LHS (surrogate)       ~3-8s   -> 4-param estimate
S5: Best surrogate selection             ~0s     -> picks min-loss from S2/S3/S4
P1: PDE joint on SHALLOW cathodic        ~80s    -> 4-param refinement
P2: PDE joint on FULL cathodic           ~200s   -> final 4-param estimate
```

### Current Component Boundaries

| Component | Responsibility | Location |
|-----------|---------------|----------|
| v13 master script | Phase orchestration, CLI args, CSV output | `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` |
| `Surrogate.objectives` | Objective functions: `SurrogateObjective`, `AlphaOnlySurrogateObjective`, `SubsetSurrogateObjective`, `ReactionBlockSurrogateObjective` | `Surrogate/objectives.py` |
| `Surrogate.cascade` | Per-observable cascade (3 passes: CD-dominant, PC-dominant, polish) | `Surrogate/cascade.py` |
| `Surrogate.multistart` | 20K LHS grid search + top-K polish | `Surrogate/multistart.py` |
| `Surrogate.ensemble` | NN ensemble loading and prediction | `Surrogate/ensemble.py` |
| `Surrogate.surrogate_model` | RBF surrogate model (`.predict()`, `.predict_batch()`) | `Surrogate/surrogate_model.py` |
| `FluxCurve` | PDE-based I-V curve inference: `BVFluxCurveInferenceRequest`, parallel pool, point solve | `FluxCurve/` |
| `FluxCurve.bv_parallel` | `BVPointSolvePool` for multiprocessing PDE solves | `FluxCurve/bv_parallel.py` |
| `Forward` | Firedrake PDE solver, steady-state config, noise generation | `Forward/` |
| `Nondim` | Nondimensionalization transforms and physical constants | `Nondim/` |
| `scripts/_bv_common.py` | Shared constants (K0_HAT, ALPHA, SNES_OPTS), param builders | `scripts/_bv_common.py` |
| `Inverse` | Firedrake-adjoint reduced functional framework (not used by v13 surrogate path) | `Inverse/` |

### Current Data Flow

```
True params -> PDE solver (subprocess) -> clean I-V curves
                                       -> add_percent_noise() -> noisy targets
                                       -> cache to .npz

Surrogate model (.pkl or nn ensemble dir) -> .predict(k0_1, k0_2, a1, a2)
                                          -> {"current_density": [...], "peroxide_current": [...]}

Objective = 0.5 * ||CD_sim - CD_target||^2 + w * 0.5 * ||PC_sim - PC_target||^2
Gradient = central finite differences (8 surrogate evals for 4 params)

scipy.optimize.minimize(L-BFGS-B) on surrogate objective -> warm-start
FluxCurve PDE inference (L-BFGS-B) -> final estimate

Results -> CSV files in StudyResults/master_inference_v13/
```

### Known Issues with v13

1. **S2, S3, S4 converge to the same surrogate minimum.** Evidence: S2 loss = 2.466e-6, S3 loss = 2.466e-6, S4 loss = 2.466e-6 (identical to 4 significant figures). The multistart (S4) confirms the basin is unique but costs 9.5s for no new information beyond S2.

2. **~11% surrogate bias from PDE truth.** Surrogate optimum is ~3-5% error on all params; PDE refinement reduces to ~2-4%. The surrogate bias is a fundamental accuracy limit.

3. **S1 alpha-only initialization may be unnecessary.** S2 already starts from initial guesses and finds the correct basin in 0.5s. S1's value is providing a marginally better alpha warm-start, but S2 could work without it.

4. **P1 (shallow PDE) may be redundant.** P1 at 73s produces nearly identical results to the surrogate optimum (errors change by <0.1%). P2 (full cathodic) does the real refinement.

5. **No seed-robustness testing.** v13 runs with a single noise seed (20260226). Pipeline behavior across seeds is unknown.

## Recommended Architecture for v14

### Design Principles

1. **Every component justified.** Literature, empirical comparison, or simplest-that-works.
2. **Comparison experiments first, then redesign.** Baseline v13, ablate components, measure impact.
3. **Minimal pipeline.** Remove phases that don't improve robustness. Fewer phases = fewer failure modes.
4. **Seed-robustness is the metric.** Pipeline must achieve <10% error on all 4 params across multiple noise seeds at 2% noise.

### Proposed Pipeline Structure

Based on v13 analysis, the likely minimal pipeline is:

```
Phase 1: Surrogate warm-start (single strategy, ~1s)
Phase 2: PDE refinement on full cathodic range (~200-300s)
```

But this must be validated through systematic ablation experiments before committing.

### Component Architecture

```
                    ExperimentRunner
                    /              \
           PipelineConfig      ResultsCollector
              /      \              |
     SurrogatePhase  PDEPhase    ComparisonTable
         |               |
    SurrogateObjective   FluxCurve (existing)
         |
    Surrogate models (existing)
```

#### New Components

| Component | Responsibility | Integration Point |
|-----------|---------------|-------------------|
| `ExperimentRunner` | Run pipeline variants across seeds, collect results | Orchestrates existing surrogate/PDE modules |
| `PipelineConfig` | Declarative pipeline definition (which phases, which objectives, which optimizer) | Replaces v13's hardcoded phase sequence |
| `ResultsCollector` | Aggregate per-seed results into comparison tables | Writes to StudyResults/ |
| `ComparisonTable` | Standardized metric computation (max relative error, per-param error) | Used by both experiments and tests |

#### Modified Components

| Component | Change | Reason |
|-----------|--------|--------|
| `Surrogate.objectives` | No changes needed | Already well-factored with `SurrogateObjective`, `SubsetSurrogateObjective`, etc. |
| `Surrogate.cascade` | May be removed from pipeline | Experiment will determine if cascade adds value over joint L-BFGS-B |
| `Surrogate.multistart` | May be simplified to diagnostic-only | 20K LHS confirms basin uniqueness but likely unnecessary every run |
| `FluxCurve` | No changes needed | PDE inference phases work correctly |
| `scripts/_bv_common.py` | Add noise seed iteration support | Currently hardcodes seed |

#### Unchanged Components

| Component | Reason |
|-----------|--------|
| `Forward/` | Verified correct via MMS in v1.0 |
| `Nondim/` | Verified correct via roundtrip tests |
| `Surrogate.surrogate_model` | Surrogate accuracy is what it is; retraining is out of scope |
| `Surrogate.ensemble` | NN ensemble loading works fine |
| `Inverse/` | Firedrake-adjoint path not used by BV kinetics pipeline |

### Data Flow (v14)

```
ExperimentRunner(config) -> for each seed in seeds:
  1. Generate noisy targets (PDE at true params + noise)
     - Reuse cached clean targets (existing _target_cache_path logic)
     - Apply noise with current seed
  2. Run pipeline variant:
     a. Surrogate warm-start (objective + optimizer from config)
     b. PDE refinement (voltage grid + optimizer from config)
  3. Compute metrics:
     - Per-param relative error
     - Max relative error across params
     - Objective value
     - Runtime
  4. Store results -> ResultsCollector

ResultsCollector -> comparison CSV/JSON with:
  - Per-seed breakdown
  - Aggregate statistics (mean, median, worst-case across seeds)
  - Pipeline variant label for comparison
```

## Patterns to Follow

### Pattern 1: Declarative Pipeline Configuration

**What:** Define pipeline variants as data (config dicts or dataclasses), not code.

**When:** For all comparison experiments. Each experiment is a PipelineConfig, not a new script.

**Why:** v13 has ~45 scripts in `scripts/inference/` because each pipeline variant became a new file. Declarative configs prevent script sprawl.

**Example:**
```python
@dataclass(frozen=True)
class PipelineConfig:
    """Declarative definition of an inference pipeline variant."""
    name: str
    surrogate_strategy: str  # "joint", "cascade", "multistart", "none"
    surrogate_model_type: str  # "nn-ensemble", "rbf", etc.
    pde_phases: list[PDEPhaseConfig]  # empty list = surrogate-only
    noise_seeds: list[int]
    noise_percent: float = 2.0
    secondary_weight: float = 1.0

@dataclass(frozen=True)
class PDEPhaseConfig:
    """Configuration for a single PDE refinement phase."""
    name: str
    voltage_grid: np.ndarray
    maxiter: int
    gtol: float = 5e-6
    ftol: float = 1e-8
    secondary_weight: float = 1.0
```

### Pattern 2: Seed-Sweep Experiment Structure

**What:** Run each pipeline variant across N noise seeds, report worst-case and median.

**When:** For all robustness evaluations. Single-seed results are misleading.

**Why:** v13 was optimized for seed=20260226. A pipeline that works on one seed but fails on others is not robust.

**Example:**
```python
def run_seed_sweep(config: PipelineConfig) -> SweepResult:
    """Run pipeline across all configured seeds, collect per-seed metrics."""
    results = []
    for seed in config.noise_seeds:
        targets = generate_noisy_targets(seed=seed, noise_percent=config.noise_percent)
        result = run_single_pipeline(config, targets, seed)
        results.append(result)
    return SweepResult(
        config=config,
        per_seed=results,
        worst_case_max_err=max(r.max_relative_error for r in results),
        median_max_err=np.median([r.max_relative_error for r in results]),
    )
```

### Pattern 3: Ablation-First Redesign

**What:** Before adding or removing pipeline components, run controlled ablation experiments that isolate the contribution of each component.

**When:** During the audit phase (before implementing v14).

**Why:** Intuition about what matters is often wrong. S1 "seems unnecessary" but maybe it prevents failure on certain seeds. Only data settles this.

**Ablation matrix:**
```
Variant 0: Full v13 (baseline)
Variant 1: Skip S1 (no alpha init)
Variant 2: Skip S3 (no cascade)
Variant 3: Skip S4 (no multistart)
Variant 4: Skip S3+S4 (joint only)
Variant 5: Skip P1 (surrogate -> P2 directly)
Variant 6: Skip S1+S3+S4+P1 (minimal: S2 -> P2)
```

Each variant runs across 10+ noise seeds. The variant with the best worst-case performance wins.

### Pattern 4: PDE Target Caching with Seed Separation

**What:** Cache clean PDE targets separately from noise application. Apply noise per-seed at runtime.

**When:** Always. PDE target generation costs ~70s and is deterministic given true params.

**Why:** v13 already does this via `_target_cache_path` and `.npz` caching. The v14 seed sweep reuses the same clean targets with different noise realizations.

**Key detail:** Clean targets depend on: true params, voltage grid, mesh, solver config. Noise depends on: noise_percent, seed. These must be factored apart.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Optimizing for a Single Seed
**What:** Tuning hyperparameters (secondary_weight, maxiter, voltage grid) to minimize error on one specific noise realization.
**Why bad:** Produces a pipeline that overfits to that particular noise pattern. Different seeds may hit different local minima or convergence issues.
**Instead:** Optimize for worst-case or median performance across 10+ seeds. Use boxplots, not point estimates.

### Anti-Pattern 2: Adding Phases to Fix Edge Cases
**What:** When one seed fails, adding a new pipeline phase specifically to handle that case.
**Why bad:** Increases pipeline complexity and runtime without addressing root cause. Each new phase is a new failure mode.
**Instead:** Investigate why the seed fails. Is it a bad surrogate warm-start? A PDE solver convergence issue? Fix the root cause with a more robust objective or optimizer, not more phases.

### Anti-Pattern 3: Comparing Losses Across Different Objectives
**What:** Comparing S2 loss (surrogate objective) with P2 loss (PDE objective) to decide which is "better."
**Why bad:** The objectives are fundamentally different. The surrogate objective has ~11% bias. A low surrogate loss does not mean low PDE error.
**Instead:** Always compare parameter recovery error (relative error vs true params). Loss is useful within a single objective type for convergence monitoring, not across objective types.

### Anti-Pattern 4: Script Sprawl for Variants
**What:** Creating a new Python script for each pipeline variant (`Infer_BVMaster_charged_v14a.py`, `v14b.py`, `v14c.py`, ...).
**Why bad:** v1-v13 already accumulated ~45 inference scripts. Each is slightly different, making comparison and maintenance nightmarish.
**Instead:** Use the declarative PipelineConfig approach. One runner script, many configs.

### Anti-Pattern 5: Subprocess Isolation When Not Needed
**What:** Running surrogate inference in a subprocess to avoid Firedrake/PyTorch conflicts.
**Why bad for surrogate-only:** The subprocess boundary exists because Firedrake and PyTorch PETSc conflict. But surrogate-only phases (S1-S5) use only NumPy/SciPy -- no Firedrake needed. The v13 script already handles this correctly with `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`.
**Instead:** Keep subprocess isolation only for PDE phases. Surrogate phases run in-process.

## Build Order for v14 Components

The build order follows dependency chains and the principle of validating each piece before building the next.

### Phase 1: Baseline and Metrics (no new components)

**Build:**
1. Seed-sweep harness (run v13 across 10 seeds, collect per-seed results)
2. Standardized metrics module (max relative error, per-param error, timing)
3. v13 baseline results across seeds

**Rationale:** You cannot improve what you have not measured. The seed sweep reveals whether v13 is already robust or has seed-dependent failures.

**Dependencies:** None new. Uses existing v13 script.

**Integration:** Wraps v13 script invocation, parses output CSVs.

### Phase 2: Ablation Experiments (minimal new code)

**Build:**
1. Pipeline ablation configs (skip S1, skip S3+S4, skip P1, etc.)
2. Ablation runner that modifies v13 CLI args
3. Comparison table generator

**Rationale:** Ablation data determines which components matter. Cannot design v14 without this.

**Dependencies:** Phase 1 metrics module.

**Integration:** Uses v13 `--skip-p1`, `--surr-strategy` flags. May need minor v13 modifications to support all ablation variants.

### Phase 3: Alternative Component Experiments

**Build:**
1. Alternative objective experiments (different secondary_weight, different loss formulations)
2. Alternative optimizer experiments (Nelder-Mead, differential evolution, basin-hopping)
3. Voltage grid experiments (different eta grids for PDE phase)

**Rationale:** Each component (objective, optimizer, voltage grid) has alternatives worth testing. But test one at a time against the ablated baseline.

**Dependencies:** Phase 2 results (know which components to keep).

**Integration:** New objective classes extend `Surrogate.objectives`. New optimizer calls replace `scipy.optimize.minimize` in the relevant phase.

### Phase 4: v14 Pipeline Implementation

**Build:**
1. `PipelineConfig` and `PDEPhaseConfig` dataclasses
2. `ExperimentRunner` that executes a `PipelineConfig`
3. `ResultsCollector` for aggregation
4. v14 pipeline as a specific `PipelineConfig`

**Rationale:** Only build the final pipeline after empirical evidence determines the optimal component set.

**Dependencies:** Phase 2 + Phase 3 results.

**Integration:** Uses existing `Surrogate.objectives`, `Surrogate.multistart`, `FluxCurve` modules unchanged. New orchestration code replaces the v13 monolithic script.

### Phase 5: Robustness Validation

**Build:**
1. 20+ seed validation sweep
2. Comparison of v14 vs v13 across all seeds
3. Test for `tests/test_v14_robustness.py`

**Rationale:** The pipeline is only "done" when it passes the robustness criterion (<10% error on all params across all seeds).

**Dependencies:** Phase 4 pipeline.

**Integration:** Automated test in pytest. Uses the same `ExperimentRunner`.

## Scalability Considerations

| Concern | Current v13 | v14 (10 seeds) | v14 (50 seeds) |
|---------|-------------|-----------------|-----------------|
| Surrogate warm-start | ~10s | ~100s total (10 seeds x 10s) | ~500s |
| PDE refinement | ~260s (P1+P2) | ~2600s (if P1+P2 per seed) or ~2000s (P2-only) | ~10000-15000s |
| Clean target generation | ~70s (cached) | ~70s (cached, reused) | ~70s (cached, reused) |
| Noise application | <1s | <1s per seed | <1s per seed |
| Total per pipeline variant | ~340s | ~2700s (~45 min) | ~10000s (~3 hr) |
| Ablation matrix (7 variants x 10 seeds) | N/A | ~5 hours | N/A |

**Mitigation:** Prioritize experiments that answer the highest-value questions first. Run overnight. Cache aggressively.

## Sources

- Direct codebase analysis: `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` (1100+ lines, 7-phase pipeline) -- HIGH confidence
- Direct codebase analysis: `scripts/inference/Infer_BVMaster_charged_v7.py` (3-phase PDE-only pipeline) -- HIGH confidence
- v13 results: `StudyResults/master_inference_v13/master_comparison_v13.csv` -- HIGH confidence
- `Surrogate/objectives.py`, `Surrogate/cascade.py`, `Surrogate/multistart.py` -- HIGH confidence
- `FluxCurve/bv_config.py`, `FluxCurve/bv_parallel.py` -- HIGH confidence
- `Inverse/inference_runner/recovery.py` (resilient minimizer) -- HIGH confidence
- `.planning/PROJECT.md` (project constraints, known issues) -- HIGH confidence
