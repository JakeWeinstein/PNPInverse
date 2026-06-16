# Training Data Audit Report

**Date:** 2026-03-16
**Total samples:** 3194
**v9 batch:** 445  |  **v11 batch:** 2749

## 1. Summary Statistics

| Parameter | Min | Max | Mean | Std |
|-----------|-----|-----|------|-----|
| k0_1 | 1.003e-06 | 9.897e-01 | 3.954e-02 | 1.219e-01 |
| k0_2 | 1.029e-07 | 9.991e-02 | 4.840e-03 | 1.395e-02 |
| alpha_1 | 1.000e-01 | 9.000e-01 | 4.733e-01 | 2.036e-01 |
| alpha_2 | 1.001e-01 | 8.997e-01 | 5.057e-01 | 2.020e-01 |

## 2. Marginal Distributions

See `marginal_histograms.png`.

## 3. Coverage Analysis (k0_2)

### Per-decade sample counts

| Decade | Count | Status |
|--------|-------|--------|
| [1e-7, 1e-6) | 268 | OK |
| [1e-6, 1e-5) | 357 | OK |
| [1e-5, 1e-4) | 764 | OK |
| [1e-4, 1e-3) | 772 | OK |
| [1e-3, 1e-2) | 714 | OK |
| [1e-2, 1e-1) | 319 | OK |

### k0_2 vs k01
- Empty bins: 0 / 144
- Sparse bins (<5): 1
- Min count: 3, Max count: 66

### k0_2 vs alpha1
- Empty bins: 0 / 96
- Sparse bins (<5): 0
- Min count: 8, Max count: 78

### k0_2 vs alpha2
- Empty bins: 0 / 96
- Sparse bins (<5): 2
- Min count: 3, Max count: 73

## 4. Error-Density Correlation

### r=0.25
- CD: Spearman rho=-0.570, p=1.36e-42
- PC: Spearman rho=-0.347, p=5.43e-15

### r=0.5
- CD: Spearman rho=-0.556, p=3.37e-40
- PC: Spearman rho=-0.307, p=6.64e-12

### r=1.0
- CD: Spearman rho=-0.481, p=4.69e-29
- PC: Spearman rho=-0.203, p=7.51e-06

### Top 5% worst CD samples
- N worst: 24
- Worst mean density: 295.8
- Rest mean density: 901.7
- Worst in low-density half: 24

### Top 5% worst PC samples
- N worst: 24
- Worst mean density: 483.0
- Rest mean density: 891.8
- Worst in low-density half: 24

## 5. Sensitivity Analysis

- Weak-signal k0_2 region: log10 in [-2.2, -1.0]
  (physical: [6.0e-03, 1.0e-01])
- Samples in weak-signal region: 466
- PC signal range: [4.96e-03, 1.82e-01]

## 6. Max-Empty-Ball

- Radius: 0.5846
- Location: {'k0_1': 0.8738845478997729, 'k0_2': 1.5801462798621793e-07, 'alpha_1': 0.839323521211998, 'alpha_2': 0.10731451266218511}
- Threshold: 0.15
- Status: **EXCEEDS THRESHOLD**

## 7. Convergence Failures

- Intended: 3000
- Converged: 2749
- Failed: 251 (8.4%)

| Decade | Failures |
|--------|----------|
| [1e-7, 1e-6) | 84 |
| [1e-6, 1e-5) | 64 |
| [1e-5, 1e-4) | 45 |
| [1e-4, 1e-3) | 22 |
| [1e-3, 1e-2) | 22 |
| [1e-2, 1e-1) | 14 |

## 8. Go/No-Go Decision

### DECISION: AUGMENTATION NEEDED

Reasons:
- Max-empty-ball radius (0.5846) exceeds 0.15

### Augmentation Plan

**Strategy:** Targeted LHS in under-sampled corner regions + wide-coverage supplement

The max-empty-ball is centered at k0_1=0.87, k0_2=1.6e-7, alpha_1=0.84, alpha_2=0.11 --
a corner of the parameter space where high k0_1, low k0_2, high alpha_1, and low alpha_2
combine. The LHS design covers marginals well (all k0_2 decades have 268+ samples) but
leaves 4D corners sparse. Convergence failures also cluster heavily in the low k0_2 decades
(84 failures in [1e-7, 1e-6) alone), further depleting these regions.

**Focused region 1 (largest gap corner):**
- k0_1: [0.1, 1.0]
- k0_2: [1e-7, 1e-5]
- alpha_1: [0.6, 0.9]
- alpha_2: [0.1, 0.4]
- Samples: ~500

**Focused region 2 (low k0_2 decade, failure-depleted):**
- k0_1: [1e-6, 1.0] (full range)
- k0_2: [1e-7, 1e-6]
- alpha_1: [0.1, 0.9] (full range)
- alpha_2: [0.1, 0.9] (full range)
- Samples: ~300

**Wide coverage supplement:**
- Default ParameterBounds (full ranges)
- Samples: ~200

- **Total new samples:** ~1,000
- **Estimated runtime:** ~0.8 hours (at ~3s/sample)

**Sampling function:** `generate_multi_region_lhs_samples()` with focused bounds above
**Note:** Use fresh seeds (e.g., seed_base=400, seed_focused=500) to avoid overlap with existing LHS designs
