# Phase 2f: Polynomial Chaos Expansion Surrogate with Sobol Sensitivity

## Objective

Build a PCE surrogate using ChaosPy that (a) implements the existing surrogate predict/predict_batch API, (b) computes Sobol sensitivity indices from the PCE coefficients, and (c) answers the key research question: what fraction of I-V variance does each kinetic parameter (especially k0_2) contribute?

## Deliverables

| Deliverable | Path |
|---|---|
| PCE surrogate module | `Surrogate/pce_model.py` |
| Training/fitting script | `scripts/Surrogate/train_pce.py` |
| Sensitivity report script | `scripts/Surrogate/pce_sensitivity_report.py` |
| Trained artifacts | `data/surrogate_models/pce/` |
| Sensitivity results | `StudyResults/surrogate_fidelity/pce_sobol_indices.json` |

---

## Step 1: Install ChaosPy

**Action**: Install chaospy into the venv-firedrake environment.

```bash
source ../venv-firedrake/bin/activate
pip install chaospy
```

**Verify**: `python -c "import chaospy; print(chaospy.__version__)"` succeeds.

---

## Step 2: Implement `Surrogate/pce_model.py`

This is the core deliverable. The module contains two classes: `PCEConfig` (dataclass) and `PCESurrogateModel`.

### 2a: `PCEConfig` dataclass

```python
@dataclass
class PCEConfig:
    max_degree: int = 6            # Maximum total polynomial degree
    sparse_truncation: bool = True # Use hyperbolic truncation (q < 1.0)
    q_norm: float = 0.75           # Hyperbolic truncation parameter (0 < q <= 1)
    fitting_method: str = "lstsq"  # "lstsq" or "lars" (sparse regression)
    log_space_k0: bool = True      # Transform k0_1, k0_2 to log10 before fitting
    cross_validation: bool = True  # Use LOO error to select degree
    degree_candidates: tuple = (3, 4, 5, 6, 7, 8)  # Degrees to try if CV enabled
```

**Key decisions**:
- `log_space_k0 = True`: Matches POD-RBF convention. Parameters span decades in k0, so log-transform is essential for polynomial accuracy.
- `q_norm = 0.75`: Hyperbolic truncation reduces the number of basis terms from O(p^d) to a smaller set, avoiding overfitting. With 4 dimensions and degree 6, full tensor basis = C(10,4) = 210 terms; hyperbolic truncation reduces to ~80-120 terms, well within the 3194-sample budget.
- `fitting_method = "lstsq"`: Start with ordinary least squares. LARS/OMP can be added as an option but with 3194 samples and ~100-200 basis terms, OLS is well-conditioned.

### 2b: `PCESurrogateModel` class

**Constructor** stores config, initializes empty state:
- `_expansion_cd`: chaospy expansion for current_density (list of 22 per-output-dim expansions, or one joint)
- `_expansion_pc`: same for peroxide_current
- `_joint_dist`: chaospy Joint distribution object
- `_phi_applied`, `_n_eta`, `_is_fitted`, `training_bounds`
- `_sobol_indices`: cached dict of computed Sobol indices

**Strategy: One PCE per output dimension (22 for CD, 22 for PC = 44 total).**
Rationale: Each voltage point may have different parameter sensitivity. Per-dimension PCE is straightforward with ChaosPy's `fit_regression` and gives per-voltage Sobol indices, which is more informative than a single joint PCE.

### 2c: `_build_distribution(self) -> chaospy.Distribution`

Constructs the joint input distribution for the 4 parameters in their transformed space:

```python
import chaospy as cp

# After log-transform of k0: uniform in log10-space
log_k0_1 = cp.Uniform(log10(k0_1_min), log10(k0_1_max))
log_k0_2 = cp.Uniform(log10(k0_2_min), log10(k0_2_max))
alpha_1  = cp.Uniform(alpha_1_min, alpha_1_max)
alpha_2  = cp.Uniform(alpha_2_min, alpha_2_max)
joint = cp.J(log_k0_1, log_k0_2, alpha_1, alpha_2)
```

Legendre polynomials are automatically selected by ChaosPy for Uniform distributions (Askey scheme).

### 2d: `_build_basis(self) -> chaospy.Expansion`

```python
if self.config.sparse_truncation:
    expansion = cp.generate_expansion(
        order=degree,
        dist=self._joint_dist,
        cross_truncation=self.config.q_norm,
    )
else:
    expansion = cp.generate_expansion(order=degree, dist=self._joint_dist)
```

### 2e: `fit(self, parameters, current_density, peroxide_current, phi_applied, verbose=True) -> self`

Algorithm:
1. Store `phi_applied`, `training_bounds`, `_n_eta`.
2. Transform inputs: apply log10 to k0 columns if `log_space_k0`.
3. Build joint distribution from training bounds via `_build_distribution()`.
4. If `cross_validation` is enabled:
   - For each candidate degree in `degree_candidates`:
     - Build basis expansion
     - Fit PCE via `cp.fit_regression(expansion, samples.T, outputs)` for a representative output (e.g., CD at median voltage index)
     - Compute LOO error using the closed-form hat-matrix formula: `e_loo_i = residual_i / (1 - h_ii)` where H = X(X^T X)^{-1} X^T
     - Select degree with minimum LOO error
   - Print selected degree
5. Build final basis at selected degree.
6. For each of the 22 voltage points:
   - Fit CD PCE: `self._pce_cd[j] = cp.fit_regression(expansion, X.T, cd[:, j])`
   - Fit PC PCE: `self._pce_pc[j] = cp.fit_regression(expansion, X.T, pc[:, j])`
7. Compute and cache Sobol indices (Step 3).
8. Set `_is_fitted = True`.

**Important**: ChaosPy's `fit_regression` expects samples as shape `(n_dims, n_samples)` -- the transpose of our convention.

### 2f: `predict(self, k0_1, k0_2, alpha_1, alpha_2) -> dict`

```python
x = self._transform_single(k0_1, k0_2, alpha_1, alpha_2)  # shape (4,1)
cd = np.array([float(self._pce_cd[j](*x)) for j in range(self._n_eta)])
pc = np.array([float(self._pce_pc[j](*x)) for j in range(self._n_eta)])
return {"current_density": cd, "peroxide_current": pc, "phi_applied": self._phi_applied.copy()}
```

### 2g: `predict_batch(self, parameters) -> dict`

Vectorized version: transform all parameters, evaluate each PCE on the batch.

```python
X = self._transform_inputs(parameters)  # (N, 4)
cd = np.column_stack([self._pce_cd[j](*X.T) for j in range(self._n_eta)])  # (N, 22)
pc = np.column_stack([self._pce_pc[j](*X.T) for j in range(self._n_eta)])
```

### 2h: `predict_gradient(self, k0_1, k0_2, alpha_1, alpha_2) -> dict`

Analytic gradient via polynomial differentiation. This is a unique advantage of PCE over RBF/NN.

For each output dimension j:
- Differentiate the polynomial w.r.t. each of the 4 transformed inputs
- If `log_space_k0`, apply chain rule: d/d(k0) = d/d(log10_k0) * 1/(k0 * ln(10))

Return dict with `"grad_cd"` shape (4, 22) and `"grad_pc"` shape (4, 22).

Implementation uses `cp.Poly.diff()` or manual coefficient differentiation.

### 2i: Properties

Match the existing surrogate API:
- `n_eta -> int`
- `phi_applied -> np.ndarray`
- `is_fitted -> bool`
- `training_bounds -> dict`
- `n_terms -> int` (number of PCE basis terms, unique to PCE)
- `selected_degree -> int`

---

## Step 3: Sobol Sensitivity Analysis

This is the primary scientific value of the PCE surrogate. Sobol indices decompose output variance into contributions from each input parameter and their interactions.

### 3a: Theory

For a PCE `f(x) = sum_alpha c_alpha * Psi_alpha(x)`, the Sobol indices are:

- **First-order** S_i = (sum of c_alpha^2 where multi-index alpha has only dimension i nonzero) / (total variance)
- **Second-order** S_ij = (sum of c_alpha^2 where alpha has only dims i,j nonzero) / (total variance)
- **Total-order** ST_i = (sum of c_alpha^2 where alpha has dimension i nonzero) / (total variance)

Total variance = sum of all c_alpha^2 for alpha != 0 (excluding the mean term).

### 3b: `compute_sobol_indices(self) -> dict`

Method on `PCESurrogateModel`. Uses ChaosPy's built-in Sobol computation:

```python
cp.Sens_m(self._pce_cd[j], self._joint_dist)  # First-order for CD at voltage j
cp.Sens_t(self._pce_cd[j], self._joint_dist)  # Total-order
cp.Sens_m2(self._pce_cd[j], self._joint_dist) # Second-order (4x4 matrix)
```

Compute for all 22 voltage points, for both CD and PC. Store as:

```python
{
    "parameter_names": ["log10_k0_1", "log10_k0_2", "alpha_1", "alpha_2"],
    "current_density": {
        "first_order": np.ndarray (4, 22),   # S_i at each voltage
        "total_order": np.ndarray (4, 22),    # ST_i at each voltage
        "second_order": np.ndarray (4, 4, 22),# S_ij at each voltage
        "mean_first_order": np.ndarray (4,),  # Averaged over voltages
        "mean_total_order": np.ndarray (4,),
    },
    "peroxide_current": {
        # same structure
    },
}
```

### 3c: `print_sensitivity_report(self)`

Pretty-printed table showing:
1. Mean first-order and total-order Sobol indices for each parameter, averaged over voltages
2. Top 3 two-way interactions (S_ij)
3. The specific answer to "what fraction of I-V variance does k0_2 explain?" (first-order and total-order)
4. Voltage-resolved sensitivity: at which voltages is k0_2 most/least influential?

Format:
```
================================================================
  PCE SOBOL SENSITIVITY ANALYSIS
  Basis terms: 120, Degree: 6, Samples: 3194
================================================================
  CURRENT DENSITY — Mean Sobol Indices (averaged over 22 voltages)
  ----------------------------------------------------------------
  Parameter      S_i (1st)    ST_i (total)   Interaction
  log10_k0_1     0.XXXX       0.XXXX         0.XXXX
  log10_k0_2     0.XXXX       0.XXXX         0.XXXX
  alpha_1        0.XXXX       0.XXXX         0.XXXX
  alpha_2        0.XXXX       0.XXXX         0.XXXX
  ----------------------------------------------------------------
  Top interactions:
    k0_1 x alpha_1: S_12 = 0.XXXX
    k0_2 x alpha_2: S_24 = 0.XXXX
    ...
  ================================================================
  KEY FINDING: k0_2 accounts for XX.X% of total I-V variance
  (first-order) and XX.X% including interactions (total-order).
  ================================================================
```

### 3d: `save_sobol_indices(self, path: str)`

Export the full Sobol index structure to JSON for downstream use in the V&V report.

---

## Step 4: Training Script (`scripts/Surrogate/train_pce.py`)

Standalone script that:
1. Loads `data/surrogate_models/training_data_merged.npz`
2. Loads `data/surrogate_models/split_indices.npz` for train/test split (same split as POD-RBF to enable fair comparison)
3. Instantiates `PCESurrogateModel` with default config
4. Calls `fit()` on training data
5. Runs `validate_surrogate()` from `Surrogate.validation` on test data
6. Prints validation report + sensitivity report
7. Saves model to `data/surrogate_models/pce/pce_model.pkl`
8. Saves Sobol indices to `StudyResults/surrogate_fidelity/pce_sobol_indices.json`
9. Generates comparison table: PCE vs POD-RBF vs NN ensemble errors

**Estimated runtime**: < 30 seconds (PCE fitting is fast -- just a least-squares solve).

---

## Step 5: Sensitivity Report Script (`scripts/Surrogate/pce_sensitivity_report.py`)

Standalone script for generating publication-quality sensitivity outputs:
1. Loads fitted PCE model from pickle
2. Generates voltage-resolved Sobol index plots:
   - Line plot: S_i(voltage) for each parameter, one panel for CD, one for PC
   - Stacked bar chart: variance decomposition at each voltage
3. Saves plots to `StudyResults/surrogate_fidelity/`
4. Exports LaTeX table for the V&V report
5. Prints the key k0_2 variance fraction finding

---

## Step 6: Register in `Surrogate/__init__.py`

Add PCE exports:
```python
from Surrogate.pce_model import PCEConfig, PCESurrogateModel
```

Add to `__all__`.

---

## Step 7: Validation and Comparison

### 7a: Prediction accuracy

Run `validate_surrogate()` on the PCE model using the same test split. Compare against POD-RBF and NN ensemble:

| Metric | RBF Baseline | POD-RBF (log) | NN Ensemble | PCE |
|--------|-------------|---------------|-------------|-----|
| CD RMSE | ? | ? | ? | ? |
| PC RMSE | ? | ? | ? | ? |
| CD NRMSE | ? | ? | ? | ? |
| PC NRMSE | ? | ? | ? | ? |

PCE is not expected to beat NN ensemble on raw accuracy, but should be competitive with POD-RBF. The value is in the sensitivity indices, not prediction accuracy.

### 7b: Sobol index validation

Sanity checks:
- [ ] Sum of all first-order indices <= 1.0 for each output dimension
- [ ] Total-order indices >= first-order indices for every parameter
- [ ] Sum of total-order indices >= 1.0 (interactions present)
- [ ] No negative indices (would indicate numerical issues)
- [ ] Results stable across degree 5 vs 6 vs 7 (convergence check)

### 7c: Gradient validation

Compare PCE analytic gradients against finite differences at 5 random test points:
- FD step: 1e-6 in transformed space
- Expected agreement: relative error < 1e-4

---

## Step 8: Degree Convergence Study

Fit PCE at degrees 3, 4, 5, 6, 7, 8. For each:
- Record LOO error
- Record test-set RMSE
- Record Sobol indices

Plot convergence of both accuracy and Sobol indices vs degree. This validates that the selected degree is sufficient and that sensitivity results are stable.

Include in the sensitivity report script output.

---

## Implementation Order

1. **Step 1**: Install chaospy (1 min)
2. **Step 2a-2e**: Core PCEConfig + fit() (main implementation, ~200 lines)
3. **Step 2f-2g**: predict/predict_batch (50 lines)
4. **Step 3**: Sobol computation + report (100 lines)
5. **Step 6**: Register in __init__.py (2 lines)
6. **Step 4**: Training script (80 lines)
7. **Step 7**: Validate, compare, sanity-check
8. **Step 2h**: Analytic gradients (optional, 50 lines)
9. **Step 5**: Publication plots (80 lines)
10. **Step 8**: Degree convergence study

**Total estimated LOC**: ~550 in `pce_model.py`, ~80 in `train_pce.py`, ~80 in `pce_sensitivity_report.py`.

---

## Success Criteria

- [ ] `PCESurrogateModel.predict()` and `predict_batch()` match the API contract (same keys, same shapes)
- [ ] Prediction NRMSE within 2x of POD-RBF (log) on the same test split
- [ ] Sobol indices computed for all 4 parameters and all 6 two-way interactions
- [ ] Sobol indices pass sanity checks (non-negative, sum constraints)
- [ ] Quantitative answer: "k0_2 explains X% of I-V variance (first-order) and Y% including interactions"
- [ ] Sobol indices stable across degrees 5-7 (max variation < 0.05 in any S_i)
- [ ] Sensitivity report clearly shows which parameters dominate at which voltages
- [ ] Model serializable to pickle, loadable from `data/surrogate_models/pce/`

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| ChaosPy `fit_regression` fails for vector output | Fit 44 scalar PCEs (one per output dim) instead of joint |
| Degree selection overfits | LOO cross-validation with hat-matrix formula; compare test error |
| Sobol indices noisy at high degree | Hyperbolic truncation (q=0.75) limits spurious high-order terms |
| ChaosPy API differs from expected | Pin version; check `cp.generate_expansion` vs `cp.orth_ttr` naming |
| PCE accuracy poor in log-k0 tails | log-transform already compresses the range; validate at extremes |

---

## Dependencies

- `chaospy` (new, must install)
- `numpy`, `scipy` (already available)
- `matplotlib` (for plots, already available)
- No PyTorch dependency (unlike NN surrogate)
