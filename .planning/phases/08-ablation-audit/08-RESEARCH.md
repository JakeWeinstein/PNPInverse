# Phase 8: Ablation Audit - Research

**Researched:** 2026-03-12
**Domain:** Scientific computing / empirical pipeline audit / statistical testing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Ablation configurations (4 total, all 20 seeds 0-19):**
1. Full v13 (S1-S5 → P1 → P2) — reuse Phase 7 results, no re-run needed
2. S4-only → P1 → P2 — multistart 20K LHS replaces cascade/joint warm-up
3. S4-only → P2 — skip P1, multistart straight to full cathodic PDE
4. Cold-start → P1 → P2 — no surrogate at all, default initial guess into PDE

- Surrogate stages S1-S3 treated as a single block — do not ablate individually
- S4 multistart uses the same 20K LHS grid as v13 (isolate warm-start, not grid size)
- S5 (best selection) is implicit logic, not a real stage — not ablated separately

**Statistical comparison method:**
- Wilcoxon signed-rank test on paired per-parameter relative errors across 20 seeds
- p < 0.05 threshold: stage is "redundant" if removal does NOT significantly worsen results
- Per-parameter testing: separate Wilcoxon test for each of k0_1, k0_2, alpha_1, alpha_2 (4 p-values per ablation config)
- Report both median error (Wilcoxon) and worst-case (max) error (descriptive comparison)

**Justification criteria format (AUDT-03):**
- Table + narrative in `StudyResults/v14/ablation/`
- Table columns: Stage | Status (justified/redundant/unjustified) | Criterion (literature/empirical/simplest) | Evidence summary
- 1-2 sentence narrative per stage
- S4 (multistart warm-starting) gets literature credit from surrogate-assisted optimization literature
- S1-S3 need empirical justification from ablation results or are marked redundant
- S5 listed as "N/A: selection logic"
- P1/P2 assessed by ablation results

**Minimal pipeline spec output:**
- Markdown spec document at `StudyResults/v14/ablation/minimal_pipeline_spec.md`
- Lists only stages that survived ablation with justification
- Stages removed if Wilcoxon p >= 0.05
- Includes timing data (wall-clock per stage per seed) for cost/benefit analysis
- If ALL surrogate stages are redundant, spec recommends PDE-only pipeline
- Spec states what survived only — no Phase 9 recommendations
- Not a runnable script — blueprint for Phase 9 and Phase 11

### Claude's Discretion

- Ablation script architecture (new script vs modifying v13 with flags)
- How to implement the cold-start initial guess (zeros, parameter-space center, or other)
- Plot styling for ablation comparison figures
- Narrative tone and detail level in justification table
- Whether to generate comparison box plots or other visualizations

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUDT-01 | Ablation study removing S3/S4 surrogate stages, quantifying per-stage contribution across 10+ seeds | Configurations 2-4 vs config 1 baseline; Wilcoxon test pattern established |
| AUDT-02 | Ablation study of P1 (shallow PDE), quantifying contribution vs direct S2→P2 | Config 3 (S4-only→P2) vs config 2 (S4-only→P1→P2) paired comparison |
| AUDT-03 | Document justification status of each v13 stage against 3 criteria | Justification table template; evidence from ablation results + literature |
| AUDT-04 | Every new component added must pass the 3-criteria test | metadata.json sidecar pattern; ablation script itself must be justified |
</phase_requirements>

---

## Summary

Phase 8 is a pure measurement phase: run 3 new ablation configurations (each across 20 seeds, sequential execution), compare against the Phase 7 baseline, apply Wilcoxon signed-rank tests, and produce a justification table plus a minimal pipeline specification. No new algorithmic components are introduced. The primary risk is execution cost — each new config runs 20 seeds of a 5-7 minute inference pipeline (~2-3 hours per config, ~6-8 hours total for all 3 new configs).

The v13 script already exposes all the flags needed. Config 2 (S4-only → P1 → P2) maps to `--surr-strategy multistart`. Config 3 (S4-only → P2) maps to `--surr-strategy multistart --skip-p1`. Config 4 (cold-start → P1 → P2) maps to `--pde-cold-start`. The multi-seed wrapper pattern from Phase 7 (`scripts/studies/run_multi_seed_v13.py`) is the direct template for the ablation wrapper — it handles sequential seed execution, CSV parsing, result aggregation, and the AUDT-04 metadata sidecar.

The Phase 7 baseline data shows median k0_1 error of 22.1% and k0_2 error of 27.8% with wide worst-case spreads (max 70.7% and 54.8% respectively). This high variance in the baseline means the Wilcoxon test will need the full 20-seed pairing to achieve meaningful power. The alpha parameters are substantially better (median ~7-10%), providing a clearer signal for those dimensions.

**Primary recommendation:** Build a new script `scripts/studies/run_ablation_v14.py` following the exact multi-seed wrapper pattern, dispatching to v13 with different flag combinations per config. Keep the analysis (Wilcoxon + justification table + spec writing) in a companion `scripts/studies/analyze_ablation_v14.py`.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | project standard | Array ops, error computation | Already in all scripts |
| scipy.stats | project standard | Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) | Standard nonparametric test, already a dependency |
| matplotlib | project standard | Box plots / comparison figures | All Phase 7 plots use it |
| csv / json | stdlib | Output CSVs and metadata.json sidecars | Established pattern in codebase |
| subprocess | stdlib | Invoke v13 script per seed (isolates Firedrake process) | Established pattern in run_multi_seed_v13.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses (frozen) | stdlib | Config + result containers | Follow frozen dataclass pattern established in codebase |
| argparse | stdlib | CLI for num-seeds, timeout, output-dir flags | All study scripts use argparse |

### Key scipy.stats API
```python
from scipy.stats import wilcoxon
# paired signed-rank test
stat, p_value = wilcoxon(errors_config_ablated, errors_config_full, alternative='greater')
# alternative='greater': tests that ablated is worse (removal hurts)
# p >= 0.05 -> removal does NOT significantly worsen -> stage is "redundant"
```

**Installation:** All dependencies already installed. No new packages required.

---

## Architecture Patterns

### Recommended Project Structure

```
scripts/studies/
├── run_ablation_v14.py          # Multi-seed runner for 3 new configs
└── analyze_ablation_v14.py      # Wilcoxon analysis + justification table + spec

StudyResults/v14/ablation/
├── config2_s4only_p1p2/         # Per-seed outputs for config 2
│   ├── seed_results.csv
│   ├── summary_statistics.csv
│   └── metadata.json
├── config3_s4only_p2/           # Per-seed outputs for config 3
│   ├── seed_results.csv
│   └── ...
├── config4_coldstart_p1p2/      # Per-seed outputs for config 4
│   ├── seed_results.csv
│   └── ...
├── ablation_table.csv           # Full comparison table (all configs x all params x all seeds)
├── wilcoxon_results.csv         # p-values per config x parameter
├── justification_table.md       # Stage | Status | Criterion | Evidence
├── minimal_pipeline_spec.md     # Survived stages + timing
└── metadata.json                # AUDT-04 sidecar
```

### Pattern 1: Config-Dispatched Multi-Seed Runner

The ablation runner follows `run_multi_seed_v13.py` exactly, adding a `--config` flag that selects which ablation configuration to run. Each config maps to a specific set of flags passed to the v13 script.

```python
# Source: scripts/studies/run_multi_seed_v13.py (adapted)
from dataclasses import dataclass, field

ABLATION_CONFIGS = {
    "config2_s4only_p1p2": [
        "--surr-strategy", "multistart",
        # No --skip-p1, no --pde-cold-start
    ],
    "config3_s4only_p2": [
        "--surr-strategy", "multistart",
        "--skip-p1",
    ],
    "config4_coldstart_p1p2": [
        "--pde-cold-start",
        # No surrogate flags needed; --pde-cold-start skips all surrogate stages
    ],
}

@dataclass(frozen=True)
class AblationConfig:
    """Frozen configuration for a single ablation run."""
    config_name: str
    num_seeds: int = 20
    noise_percent: float = 2.0
    seed_start: int = 0
    timeout_per_seed: int = 900
    output_dir: str = field(default_factory=lambda: ...)
```

The CSV output structure for each config matches `run_multi_seed_v13.py` exactly: a `seed_results.csv` with `seed, k0_1, k0_2, alpha_1, alpha_2, k0_1_err_pct, k0_2_err_pct, alpha_1_err_pct, alpha_2_err_pct` columns. This allows the analysis script to load the Phase 7 baseline from `StudyResults/v14/multi_seed/seed_results.csv` using the identical parser.

### Pattern 2: Paired Wilcoxon Analysis

The analysis script loads all 4 configs' `seed_results.csv` files, aligns by seed index (seeds 0-19), and runs paired Wilcoxon tests. Config 1 baseline is loaded from the Phase 7 output directory directly.

```python
# Source: scipy.stats.wilcoxon docs + Wilcoxon signed-rank test conventions
from scipy.stats import wilcoxon
import numpy as np

def run_wilcoxon_comparison(baseline_errors: np.ndarray, ablated_errors: np.ndarray,
                            param_name: str) -> dict:
    """Paired Wilcoxon signed-rank test: tests if ablation worsens recovery.

    alternative='greater': H1 is that ablated_errors > baseline_errors (ablation hurts)
    p >= 0.05 -> fail to reject H0 -> stage is 'redundant' (removal doesn't hurt)
    p < 0.05  -> reject H0 -> stage is 'justified' (removal significantly hurts)
    """
    # Ensure seeds are paired: load both CSVs sorted by seed column
    stat, p_value = wilcoxon(ablated_errors, baseline_errors, alternative='greater')
    return {
        "param": param_name,
        "stat": stat,
        "p_value": p_value,
        "verdict": "redundant" if p_value >= 0.05 else "justified",
        "median_baseline": float(np.median(baseline_errors)),
        "median_ablated": float(np.median(ablated_errors)),
        "max_baseline": float(np.max(baseline_errors)),
        "max_ablated": float(np.max(ablated_errors)),
    }
```

**Critical alignment note:** The Phase 7 seed_results.csv has seeds [0,1,2,4,5,6,8,10,11,12,13,14,15,16,17,18,19] — seed 3, 7, and 9 are missing (see actual CSV: only 17 seeds ran successfully). The ablation configs must be restricted to exactly the seeds present in the baseline CSV when doing paired tests. Use `set(baseline_seeds) & set(ablated_seeds)` to find the intersection before running Wilcoxon.

### Pattern 3: Justification Table Writer

```python
# Source: CONTEXT.md locked decisions + AUDT-04 pattern
STAGE_JUSTIFICATIONS = {
    "S1": {"stage": "S1: Alpha init (surrogate)", "criterion": "empirical"},
    "S2": {"stage": "S2: Joint L-BFGS-B (surrogate)", "criterion": "empirical"},
    "S3": {"stage": "S3: Cascade 3-pass (surrogate)", "criterion": "empirical"},
    "S4": {"stage": "S4: MultiStart 20K (surrogate)", "criterion": "literature"},
    "S5": {"stage": "S5: Best selection", "criterion": "N/A"},
    "P1": {"stage": "P1: PDE shallow cathodic", "criterion": "empirical"},
    "P2": {"stage": "P2: PDE full cathodic", "criterion": "simplest"},
}
# Status is filled in by ablation results: "justified" / "redundant" / "unjustified"
# S1-S3 status derived from config2 vs config1 comparison
# P1 status derived from config3 vs config2 comparison
# S4 status is "justified" (literature: surrogate-assisted multistart is the standard
#   approach for global basin coverage in expensive black-box optimization)
# P2 status is "justified" (simplest: it is the core objective, always required)
```

### Pattern 4: AUDT-04 Metadata Sidecar

Follow the exact schema from `run_multi_seed_v13.py`:

```python
metadata = {
    "tool_name": "Ablation Audit v14",
    "phase": "08-ablation-audit",
    "requirement": "AUDT-01,AUDT-02,AUDT-03",
    "justification_type": "empirical",
    "reference": "Wilcoxon signed-rank test for paired nonparametric comparison",
    "rationale": "...",
    "parameters": {
        "n_seeds": 20,
        "noise_percent": 2.0,
        "configs": list(ABLATION_CONFIGS.keys()),
        "statistical_test": "Wilcoxon signed-rank, p < 0.05 threshold",
    },
    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
```

### Anti-Patterns to Avoid

- **Running ablation configs in parallel processes:** Firedrake/PETSc process conflicts. Each seed must run in a separate subprocess sequentially, identical to the Phase 7 multi-seed pattern.
- **Pairing seeds by index rather than seed value:** The Phase 7 baseline CSV is missing 3 seeds. Always align by the seed integer value, not by row index.
- **Using unpaired Wilcoxon (Mann-Whitney U):** The 20 seeds are the same noise realizations across configs — this is a paired test. Use `scipy.stats.wilcoxon`, not `scipy.stats.mannwhitneyu`.
- **Declaring a stage "redundant" from descriptive statistics alone:** The CONTEXT.md locks Wilcoxon p >= 0.05 as the decision rule. Descriptive comparison of medians is supplementary only.
- **Generating a new surrogate or re-training:** Phase 8 uses the existing surrogate (NN ensemble) without modification.
- **Ablating S1, S2, S3 individually:** The CONTEXT.md locks S1-S3 as a single block. Configs 2-4 test "no S1-S3" vs "with S1-S3", not individual stages.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Nonparametric paired comparison | Custom permutation test | `scipy.stats.wilcoxon` | Wilcoxon is the locked decision (CONTEXT.md); scipy implements it correctly with continuity correction |
| Latin Hypercube sampling | Custom LHS grid | `Surrogate.multistart.MultiStartConfig` with `n_grid=20_000` | Already implemented with `scipy.stats.qmc.LatinHypercube`; same grid as v13 |
| Subprocess-based seed isolation | Threading or multiprocessing | `subprocess.run(...)` sequential pattern | Firedrake/PETSc state cannot be shared across processes |
| CSV result aggregation | Custom data loading | Reuse `parse_v13_csv` and `aggregate_seed_results` from `run_multi_seed_v13.py` | Already tested (test_multi_seed_aggregation.py) |

**Key insight:** This phase is analysis, not construction. Nearly all infrastructure already exists in the codebase. The value is in the empirical comparison, not in new code.

---

## Common Pitfalls

### Pitfall 1: Seed Alignment Failure in Wilcoxon Pairing

**What goes wrong:** The Phase 7 baseline seed_results.csv contains 17 seeds (not 20) — seeds 3, 7, and 9 are absent. If the ablation configs successfully run all 20 seeds, a naive row-to-row pairing would compare different noise realizations, invalidating the paired test.

**Why it happens:** Phase 7 multi-seed runner skips failed seeds without error in the summary. The CSV only contains successful runs.

**How to avoid:** Load both baseline and ablated CSVs as `{seed: errors}` dicts keyed by the `seed` column. Compute `common_seeds = sorted(set(baseline.keys()) & set(ablated.keys()))`. Run Wilcoxon only on `common_seeds`. Report the count used.

**Warning signs:** If Wilcoxon returns p-values below 1e-10 for all parameters and configs, row alignment was probably off.

### Pitfall 2: Wrong Alternative Hypothesis in Wilcoxon

**What goes wrong:** Using `alternative='two-sided'` (default) tests whether errors differ in either direction. The correct test is directional: does removing a stage make things *worse*?

**Why it happens:** `scipy.stats.wilcoxon` defaults to `alternative='two-sided'`. The phase logic (p >= 0.05 → redundant) assumes a one-sided test for "ablation worsens performance."

**How to avoid:** Use `wilcoxon(ablated_errors, baseline_errors, alternative='greater')`. Verify: for any stage that clearly helps (e.g., S4 vs cold-start), this should return p << 0.05.

### Pitfall 3: Config 4 Cold-Start Initial Guess Choice

**What goes wrong:** Using a pathologically bad initial guess (e.g., all zeros) for config 4 makes cold-start look artificially terrible, overstating the surrogate contribution.

**Why it happens:** The CONTEXT.md leaves cold-start implementation to Claude's discretion. The v13 script uses `initial_k0_guess = [0.005, 0.0005]` and `initial_alpha_guess = [0.4, 0.3]` as cold-start values when `--pde-cold-start` is set.

**How to avoid:** Use the v13 script's hardcoded default initial guess ([0.005, 0.0005] for k0, [0.4, 0.3] for alpha) as the cold-start — these are already the values that `--pde-cold-start` uses. Do not change them. This gives the fairest comparison: same starting point available to a user with no surrogate.

**Warning signs:** Cold-start produces uniform 90%+ errors across all seeds — likely a bad initial guess was used.

### Pitfall 4: CSV Phase-Row Mismatch for Config 3 (S4-only → P2)

**What goes wrong:** Config 3 uses `--skip-p1`, so the v13 output CSV has no P2 row labeled starting with "P2" — the final result is still labeled "P2: PDE joint on FULL CATHODIC" in the v13 CSV, but this is now warm-started from S4 directly. The existing `parse_v13_csv` correctly picks up the P2 row regardless.

**Why it happens:** The v13 script always writes the phase name to CSV. `parse_v13_csv` looks for `phase.startswith("P2")` which matches.

**How to avoid:** Verify by running config 3 on a single seed first and inspecting the output CSV to confirm the P2 row is present and warm-started from S4.

### Pitfall 5: Wall-Clock Timing Attribution

**What goes wrong:** Timing recorded by v13 for each phase includes queue/setup time that varies with system load. The ablation runner captures total elapsed time per seed via subprocess, not per-phase timing.

**Why it happens:** The per-phase timing in v13's CSV is from `time.time()` around each optimization call inside the process. The ablation wrapper measures wall-clock from process start to end.

**How to avoid:** For the minimal_pipeline_spec.md timing data, parse the per-phase timing from the v13 output CSV rows (`time_s` column) rather than relying on subprocess wall-clock. Aggregate `time_s` across seeds for each phase label to get median/IQR stage cost.

---

## Code Examples

### Wilcoxon Paired Test Implementation

```python
# Source: scipy.stats.wilcoxon documentation
from scipy.stats import wilcoxon
import numpy as np

def compare_configs(baseline_csv: str, ablated_csv: str) -> list[dict]:
    """Run per-parameter paired Wilcoxon tests between two ablation configs."""
    baseline = load_seed_results(baseline_csv)   # {seed: {param: error}}
    ablated = load_seed_results(ablated_csv)

    common_seeds = sorted(set(baseline.keys()) & set(ablated.keys()))
    params = ["k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct"]
    results = []

    for param in params:
        b_errs = np.array([baseline[s][param] for s in common_seeds])
        a_errs = np.array([ablated[s][param] for s in common_seeds])
        stat, p = wilcoxon(a_errs, b_errs, alternative='greater')
        results.append({
            "param": param,
            "n_pairs": len(common_seeds),
            "stat": stat,
            "p_value": p,
            "verdict": "redundant" if p >= 0.05 else "justified",
            "median_baseline": float(np.median(b_errs)),
            "median_ablated": float(np.median(a_errs)),
            "max_baseline": float(np.max(b_errs)),
            "max_ablated": float(np.max(a_errs)),
        })
    return results
```

### Config Flag Dispatch

```python
# Source: scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py arg inspection
ABLATION_CONFIGS = {
    "config1_full_v13": [],           # reuse Phase 7 results, no new runs
    "config2_s4only_p1p2": [
        "--surr-strategy", "multistart",
        # NOTE: --multistart-n 20000 is default, explicitly set for clarity
        "--multistart-n", "20000",
    ],
    "config3_s4only_p2": [
        "--surr-strategy", "multistart",
        "--multistart-n", "20000",
        "--skip-p1",
    ],
    "config4_coldstart_p1p2": [
        "--pde-cold-start",
    ],
}

def build_seed_cmd(v13_script: str, seed: int, noise_percent: float,
                   extra_flags: list[str]) -> list[str]:
    return [
        sys.executable, v13_script,
        "--noise-seed", str(seed),
        "--noise-percent", str(noise_percent),
    ] + extra_flags
```

### Box Plot Comparison Figure

```python
# Source: scripts/studies/run_multi_seed_v13.py generate_plots() (adapted)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_ablation_comparison(config_results: dict[str, list[dict]],
                             output_path: str) -> None:
    """4-panel box plot: one panel per parameter, one box per config."""
    params = ["k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct"]
    config_labels = list(config_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, param in zip(axes.flatten(), params):
        data = [[r[param] for r in config_results[cfg]] for cfg in config_labels]
        ax.boxplot(data, labels=[c.replace("config", "C") for c in config_labels])
        ax.set_ylabel("Relative Error (%)")
        ax.set_title(param.replace("_err_pct", ""))
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation Comparison: Per-Parameter Recovery Error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Single-run ablation (1 seed) | 20-seed paired comparison with Wilcoxon | Statistical validity — single-seed results are dominated by noise |
| Comparing medians only | Wilcoxon + worst-case (max) comparison | Captures tail behavior critical for worst-case guarantees |
| Ablating all stages individually | Blocking S1-S3 as one unit | Reduces combinatorial explosion; focus on the decision that matters |

---

## Open Questions

1. **Missing Phase 7 seeds (3, 7, 9)**
   - What we know: The Phase 7 seed_results.csv has 17 rows, not 20. Seeds 3, 7, 9 were not recorded (likely failed or timed out).
   - What's unclear: Whether these 3 seeds should be re-run for the baseline before ablation, or whether 17 paired seeds is sufficient for Wilcoxon power.
   - Recommendation: 17 pairs is adequate for Wilcoxon power at these effect sizes. Do not re-run Phase 7 seeds. Note the effective n=17 in the report. If the ablation configs also fail on some seeds, report the actual paired count.

2. **Config 2 (S4-only → P1 → P2) — does `--surr-strategy multistart` also run S1-S2?**
   - What we know: The v13 `_run_surrogate_phases` function checks `surr_strategy in ("all", "joint")` to decide whether to run S2, and checks `surr_strategy in ("all", "cascade")` for S3. For `--surr-strategy multistart`, S2 and S3 are skipped, but S1 (alpha-only init) is always run as it precedes the strategy check.
   - What's unclear: S1 is always executed in v13 regardless of `--surr-strategy`. This means config 2 includes S1 + S4, not purely S4.
   - Recommendation: Accept this — S1 is a 0.1s alpha-only init, negligible cost and contribution. The ablation question is "does the cascade/joint block (S2+S3) add value beyond S4's global search?" S1 is not meaningfully ablatable given the script structure. Document this in the justification table.

3. **Timing data availability per phase**
   - What we know: The v13 master_comparison_v13.csv has a `time_s` column per phase row (S1 alpha, S2 joint, S3 cascade passes, S4 multistart, P1, P2).
   - What's unclear: The per-seed subprocess output goes to STDOUT only; the existing CSV parser (`parse_v13_csv`) extracts only the P2 row. Timing for individual stages requires parsing ALL rows, not just P2.
   - Recommendation: Extend the ablation CSV parser to extract all stage rows (not just P2) for timing analysis. The minimal_pipeline_spec.md requires per-stage timing.

---

## Validation Architecture

> nyquist_validation is enabled in .planning/config.json.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml: `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`, testpaths=["tests"]) |
| Quick run command | `pytest tests/test_ablation_analysis.py -x -m "not slow"` |
| Full suite command | `pytest tests/ -m "not slow"` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUDT-01 | Wilcoxon comparison logic correctly identifies redundant vs justified stages given known error arrays | unit | `pytest tests/test_ablation_analysis.py::TestWilcoxonComparison -x` | Wave 0 |
| AUDT-01 | Seed alignment uses set intersection, not row index | unit | `pytest tests/test_ablation_analysis.py::TestSeedAlignment -x` | Wave 0 |
| AUDT-02 | P1 contribution comparison (config3 vs config2) produces distinct result rows | unit | `pytest tests/test_ablation_analysis.py::TestP1Contribution -x` | Wave 0 |
| AUDT-03 | Justification table writer produces all required columns and stage entries | unit | `pytest tests/test_ablation_analysis.py::TestJustificationTable -x` | Wave 0 |
| AUDT-03 | minimal_pipeline_spec.md contains timing data section | unit | `pytest tests/test_ablation_analysis.py::TestMinimalPipelineSpec -x` | Wave 0 |
| AUDT-04 | AUDT-04 metadata.json sidecar written with required keys | unit | `pytest tests/test_ablation_analysis.py::TestAudt04Metadata -x` | Wave 0 |
| AUDT-01 | Ablation runner config flag dispatch produces correct CLI args per config | unit | `pytest tests/test_ablation_runner.py::TestConfigDispatch -x` | Wave 0 |
| AUDT-01 | Multi-row CSV parser extracts all phase rows (not just P2) for timing analysis | unit | `pytest tests/test_ablation_runner.py::TestMultiRowParser -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_ablation_analysis.py tests/test_ablation_runner.py -x -m "not slow"`
- **Per wave merge:** `pytest tests/ -m "not slow"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_ablation_analysis.py` — covers AUDT-01, AUDT-02, AUDT-03, AUDT-04 analysis logic
- [ ] `tests/test_ablation_runner.py` — covers AUDT-01 runner config dispatch and CSV parsing
- [ ] No framework install needed — pytest already in `dev` extras

*(Existing test infrastructure covers framework and surrogate modules; new test files cover ablation-specific logic only.)*

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` — confirmed flags `--surr-strategy multistart`, `--skip-p1`, `--pde-cold-start`; confirmed S1 always runs regardless of strategy
- Direct code inspection of `scripts/studies/run_multi_seed_v13.py` — confirmed subprocess pattern, CSV schema, metadata sidecar format
- Direct inspection of `StudyResults/v14/multi_seed/seed_results.csv` — confirmed 17 successful seeds (not 20), confirmed error magnitude ranges
- Direct inspection of `pyproject.toml` — confirmed pytest infrastructure, `testpaths=["tests"]`, `slow` marker
- Direct inspection of `Surrogate/multistart.py` — confirmed `MultiStartConfig.n_grid=20_000`, `LatinHypercube` from `scipy.stats.qmc`

### Secondary (MEDIUM confidence)
- scipy.stats.wilcoxon documentation: paired signed-rank test, `alternative` parameter options including `'greater'`
- Wilcoxon signed-rank test standard methodology: for nonparametric paired comparison of two conditions across matched samples

### Tertiary (LOW confidence)
- None needed — all required information was obtainable from existing codebase and scipy stdlib

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use; no new dependencies
- Architecture: HIGH — ablation patterns are direct extensions of existing run_multi_seed_v13.py with confirmed flag mappings
- Pitfalls: HIGH — seed alignment issue confirmed from inspecting actual CSV (17 rows, not 20); S1-always-runs confirmed from code; Wilcoxon directionality from scipy API docs

**Research date:** 2026-03-12
**Valid until:** 2026-06-12 (stable domain; scipy Wilcoxon API is stable)
