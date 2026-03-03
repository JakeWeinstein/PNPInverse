# Extended Voltage Range Inference Plan

## Objective

Enable adjoint-gradient inference of Bikerman steric exclusion parameters across the full experimental voltage range eta_hat in [-1, -46.5] (V_RHE from +0.67 V to -0.50 V) while keeping per-evaluation wall time under 60 seconds.

The current inference operates on only 10 linearly spaced points in [-1, -10], which covers less than 20% of the I-V curve and misses the limiting-current plateau where steric effects are most pronounced.

---

## 1. Empirical I-V Curve Analysis

Before designing the point placement, we need to understand the I-V curve shape quantitatively. From the 100-point forward sweep in `StudyResults/bv_iv_curve_charged/bv_iv_curve_charged.csv`:

### Current density vs eta_hat (peroxide current, key observable)

| Region | eta_hat range | I_peroxide range (mA/cm2) | dI/d(eta) | Character |
|--------|--------------|--------------------------|-----------|-----------|
| Onset | -0.5 to -2.0 | -0.0006 to -0.018 | Large (exponential) | Tafel slope regime |
| Transition | -2.0 to -8.0 | -0.018 to -0.143 | Moderate (decelerating) | Mass-transport onset |
| Knee | -8.0 to -15.0 | -0.143 to -0.156 | Small (flattening) | Approach to plateau |
| Plateau | -15.0 to -46.5 | -0.156 to -0.176 | Very small (~6e-4 per unit eta) | Diffusion-limited |

### Key quantitative observations

1. **87% of the current change occurs before eta_hat = -10**: I_peroxide goes from -0.0006 to -0.145 mA/cm2 in this range.
2. **The remaining 13% spans eta_hat = -10 to -46.5**: I_peroxide changes by only 0.031 mA/cm2 over 36.5 units of eta.
3. **Electrode concentrations at eta_hat = -46.5**: c_O2 = 0.0034 (depleted to 0.34% of bulk), c_H+ = 4.5e-5 (depleted to 0.02% of bulk).
4. **Steric signal**: At these extreme depletions, the Bikerman term mu_steric = ln(1 - sum(a_j * c_j)) approaches ln(1) = 0 at the electrode. The steric effect is largest in the TRANSITION region (eta_hat ~ -3 to -10) where concentrations are intermediate and the sum a_j*c_j is non-negligible.
5. **PTC steps decrease with |eta|**: 14-22 steps in onset, 9-10 in knee, 4-6 on plateau (from CSV ss_steps column). This confirms warm-start becomes very effective on the plateau.

### Steric sensitivity analysis

The steric chemical potential is mu_steric = ln(1 - sum_j a_j c_j). For the 4-species system:

- sum(a_j * c_j) = a*(c_O2 + c_H2O2 + c_H+ + c_ClO4-)
- At bulk (y=1): sum = 0.05*(1.0 + 0.0 + 0.2 + 0.2) = 0.07, mu_steric = ln(0.93) = -0.073
- At electrode, eta=-5: c_O2 ~ 0.33, c_H2O2 ~ 1.86, sum ~ 0.05*(0.33+1.86+0.05+0.2) = 0.122, mu_steric = -0.130
- At electrode, eta=-10: c_O2 ~ 0.12, c_H2O2 ~ 0.82, sum ~ 0.05*(0.12+0.82+0.01+0.2) = 0.058, mu_steric = -0.060
- At electrode, eta=-46.5: c_O2 ~ 0.003, c_H2O2 ~ 0.004, sum ~ 0.05*(0.003+0.004+4e-5+0.2) = 0.010, mu_steric = -0.010

**Critical insight**: The steric signal peaks in the TRANSITION region (eta ~ -3 to -8), not on the plateau. On the plateau, concentrations are so depleted that steric exclusion is negligible. However, the plateau height IS affected by steric modification of the flux in the boundary layer INTERIOR (not just at the electrode surface). The gradient of mu_steric drives extra flux. So both regions contain information, but the transition zone carries more signal per point.

---

## 2. Optimal Point Placement Strategy

### 2.1. Recommended placement: hybrid logarithmic + linear (15 inference points)

Based on the I-V curve shape and steric sensitivity analysis:

```
Region 1 — Onset/Tafel (3 points):
  eta_hat = [-1.0, -2.0, -3.0]

Region 2 — Transition (6 points, densest spacing):
  eta_hat = [-4.0, -5.0, -6.5, -8.0, -10.0, -13.0]

Region 3 — Plateau (6 points, sparser spacing):
  eta_hat = [-17.0, -22.0, -28.0, -35.0, -41.0, -46.5]
```

Total: 15 inference points.

### 2.2. Rationale

- **Dense in transition (eta -3 to -13)**: This is where the I-V curvature is highest and where the steric parameter has the strongest effect on the observable. 6 points here provide good shape constraint.
- **Moderate on plateau (eta -17 to -46.5)**: The curve is nearly flat, so closely-spaced points carry redundant information. However, we include 6 points to constrain the plateau HEIGHT, which shifts with steric parameter. The spacing (~5-7 units of eta) is large enough to require hidden continuation points.
- **Onset (eta -1 to -3)**: Kinetic regime dominated by k0 and alpha. Needed for completeness if doing full inference (k0+alpha+steric), less critical for steric-only.

### 2.3. Alternative: 12-point placement (faster, recommended for initial testing)

```
eta_hat = [-1.0, -2.5, -4.0, -5.5, -7.0, -9.0, -12.0, -17.0, -23.0, -30.0, -38.0, -46.5]
```

Spacing: 1.5, 1.5, 1.5, 1.5, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 8.5

### 2.4. Alternative: 20-point placement (most signal, when time permits)

```
eta_hat = [-1.0, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.5,
           -11.0, -13.0, -16.0, -20.0, -25.0, -30.0, -35.0, -40.0, -44.0, -46.5]
```

### 2.5. Why not Chebyshev nodes?

Chebyshev nodes on [-1, -46.5] cluster at the endpoints, which is suboptimal here. The I-V curve curvature is concentrated near eta = -3 to -10, not at the endpoints. Chebyshev nodes would under-sample the transition region and over-sample the near-equilibrium onset.

---

## 3. Convergence Strategy: Hidden Continuation Points

### 3.1. The warm-start gap problem

The current inference pipeline (`FluxCurve/bv_point_solve.py`, line 189-494) solves points sequentially in ascending |eta|. With 15 inference points, the largest gap is between eta = -13 and eta = -17 (4 units) and between plateau points (5-8 units).

From the forward sweep data, warm-started solves at adjacent eta values (delta_eta ~ 0.465) converge in 4-9 PTC steps. When the gap is 4-8 units, the predictor step (P4, line 240-261) may extrapolate too aggressively, and the solution change may be too large for the warm-start to handle within the _WARMSTART_MAX_STEPS=15 cap.

### 3.2. Solution: hidden bridge points (no adjoint tape)

Insert intermediate eta values between inference points that are solved FORWARD-ONLY (no adjoint gradient). These "bridge" the gap for warm-start purposes but do NOT contribute to the objective function or gradient computation.

**Implementation approach A (preferred): Script-level, no pipeline changes**

The inference script can provide a DENSE array of phi_applied_values that includes both inference points AND bridge points. A parallel array `is_inference_point` flags which points contribute to the objective. The curve evaluator sums only over flagged points.

This requires changes to:
- `FluxCurve/bv_config.py`: Add `inference_point_mask: Optional[Sequence[bool]]` field to `BVFluxCurveInferenceRequest` (line 26).
- `FluxCurve/bv_point_solve.py`: Skip adjoint tape recording for non-inference points. In the sweep loop (line 189), when `is_inference[orig_idx]` is False, solve forward-only under `adj.stop_annotating()` and skip the adjoint derivative computation (lines 396-456). Still update `carry_U_data` and predictor state.
- `FluxCurve/bv_curve_eval.py`: Sum objective/gradient only over inference points.

**Implementation approach B (simpler but less efficient): Purely in bv_point_solve.py**

Add a `bridge_points` parameter to `solve_bv_curve_points_with_warmstart()`. Before solving each inference point, if the gap from the previous solved eta exceeds a threshold, insert automatic bridge points:

```python
# Pseudocode for bridge point insertion
gap_threshold = 3.0  # max allowed eta gap for warm-start
for sweep_idx, orig_idx in enumerate(sorted_indices):
    phi_i = phi_applied_values[orig_idx]
    if carry_U_data is not None:
        gap = abs(phi_i - prev_solved_eta)
        if gap > gap_threshold:
            # Insert bridge points
            n_bridges = int(np.ceil(gap / gap_threshold)) - 1
            bridge_etas = np.linspace(prev_solved_eta, phi_i, n_bridges + 2)[1:-1]
            for eta_b in bridge_etas:
                _solve_bridge_point_forward_only(ctx, eta_b, ...)
                carry_U_data = ...  # update warm-start
```

### 3.3. Bridge point requirements

Based on the forward sweep convergence data:
- Gap of 2.0 in eta: 8-10 PTC steps (safe with _WARMSTART_MAX_STEPS=15)
- Gap of 4.0 in eta: 12-20 PTC steps (marginal, may need increased cap)
- Gap of 8.0 in eta: likely 25-40 PTC steps (will fail with _WARMSTART_MAX_STEPS=15)

Recommended maximum gap: **3.0 units of eta_hat** for warm-start reliability.

For the 15-point placement, bridge points needed:
- Between eta=-13 and eta=-17: 1 bridge at -15
- Between eta=-17 and eta=-22: 1 bridge at -19.5
- Between eta=-22 and eta=-28: 1 bridge at -25
- Between eta=-28 and eta=-35: 2 bridges at -30.5, -33
- Between eta=-35 and eta=-41: 1 bridge at -38
- Between eta=-41 and eta=-46.5: 1 bridge at -43.75

Total bridge points: ~7. Each bridge point requires ~8-12 PTC steps (forward only, no adjoint), ~0.3-0.5s each.

### 3.4. Bridge point cost estimate

- 7 bridge points x 10 PTC steps x ~0.04s/step = ~2.8s per evaluation
- This is negligible compared to the 15 inference points (each with adjoint), so Approach B is acceptable.

---

## 4. Enhanced Predictor Step

### 4.1. Current predictor (P4, linear, 2-point)

Location: `FluxCurve/bv_point_solve.py`, lines 240-261.

The current predictor uses linear extrapolation from two prior converged solutions:
```python
slope = (phi_i - eta_curr) / (eta_curr - eta_prev)
U_predict = U_curr + slope * (U_curr - U_prev)
```

This is a first-order Taylor prediction: U(eta_i) ~ U(eta_curr) + dU/deta * (eta_i - eta_curr).

### 4.2. Proposed enhancement: quadratic predictor (3-point)

With 3 prior solutions (eta_a, U_a), (eta_b, U_b), (eta_c, U_c) where |eta_a| < |eta_b| < |eta_c|, use Lagrange interpolation:

```python
# Quadratic (Lagrange) predictor
L_a = ((eta_i - eta_b)*(eta_i - eta_c)) / ((eta_a - eta_b)*(eta_a - eta_c))
L_b = ((eta_i - eta_a)*(eta_i - eta_c)) / ((eta_b - eta_a)*(eta_b - eta_c))
L_c = ((eta_i - eta_a)*(eta_i - eta_b)) / ((eta_c - eta_a)*(eta_c - eta_b))
U_predict = L_a*U_a + L_b*U_b + L_c*U_c
```

**Risk**: Quadratic extrapolation can OVERSHOOT, producing negative concentrations. A safeguard is needed:
```python
# Clamp predicted concentrations to [eps, inf)
for k in range(n_species):
    dat = U_predict.sub(k).dat.data
    dat[:] = np.maximum(dat, 1e-10)
```

**Recommendation**: Implement quadratic predictor but fall back to linear when extrapolation distance exceeds 2x the largest prior gap:
```python
extrap_dist = abs(eta_i - eta_c)
max_prior_gap = max(abs(eta_c - eta_b), abs(eta_b - eta_a))
if extrap_dist > 2.0 * max_prior_gap:
    use_linear_predictor()
else:
    use_quadratic_predictor()
```

### 4.3. Implementation location

File: `FluxCurve/bv_point_solve.py`
- Line 186-187: Add `predictor_prev2: Optional[tuple] = None` for the third history point.
- Lines 240-261: Replace the predictor logic block with the quadratic/linear hybrid.
- Lines 449-451: Update predictor state rotation to maintain 3 points.

---

## 5. SER Adaptive dt Improvements for Large Gaps

### 5.1. Current SER parameters

Location: `FluxCurve/bv_point_solve.py`, lines 52-54.
```python
_SER_GROWTH_CAP = 4.0    # max dt multiplier per step
_SER_SHRINK = 0.5        # dt multiplier when residual grows
_SER_DT_MAX_RATIO = 20.0  # max dt / dt_initial ratio
```

### 5.2. Problem with large gaps

When jumping from eta=-13 to eta=-17 (gap=4), the first PTC step sees a large residual change. The SER mechanism correctly SHRINKS dt, but:
1. `_SER_DT_MAX_RATIO = 20.0` limits dt to 20*dt_initial. Once dt grows to this cap, the system may not yet be at steady state but dt cannot grow further.
2. `_WARMSTART_MAX_STEPS = 15` may be insufficient.

### 5.3. Recommended changes

For bridge points (forward-only, no adjoint constraint):
```python
_BRIDGE_MAX_STEPS = 40       # more relaxed for bridge points
_BRIDGE_SER_DT_MAX_RATIO = 50.0  # allow larger dt growth
```

For inference points with warm-start from a nearby bridge:
```python
_WARMSTART_MAX_STEPS = 20    # increase from 15 to 20 (line 43)
```

### 5.4. dt reset policy

Currently (line 388), dt is reset to dt_initial after each point. This is correct for inference points (each must start with a fresh adjoint tape). For bridge points, consider carrying the converged dt forward:
```python
if is_bridge_point:
    # Don't reset dt -- the next bridge/inference point benefits from
    # the large dt that this point converged at.
    pass
else:
    dt_const.assign(dt_initial)
```

This optimization avoids the initial "ramp up" phase of SER at each bridge point.

---

## 6. Computational Cost Analysis

### 6.1. Current cost (10 points, eta in [-1, -10])

| Component | PTC steps | Time per step | Time |
|-----------|-----------|--------------|------|
| Point 1 (cold start from cross-eval cache) | 9 | ~0.5s | 4.5s |
| Points 2-10 (warm-start) | 7-9 each | ~0.12s | 7.5s |
| **Total per eval** | | | **~12s** |
| Optimizer (40 evals) | | | **~8 min** |

### 6.2. Projected cost: 15 inference + 7 bridge points (eta in [-1, -46.5])

| Component | PTC steps | Time per step | Time |
|-----------|-----------|--------------|------|
| Point 1 (cold from cache, eta=-1) | 9 | ~0.5s | 4.5s |
| Points 2-9 (inference, warm-start, eta=-2 to -13) | 7-9 each | ~0.12s | 7.0s |
| Bridge points (7, forward-only, no adjoint overhead) | 8-12 each | ~0.04s | 2.8s |
| Points 10-15 (inference, warm-start from bridges, eta=-17 to -46.5) | 5-8 each | ~0.12s | 4.0s |
| **Total per eval** | | | **~18-20s** |
| Optimizer (40 evals) | | | **~12-14 min** |

### 6.3. Projected cost: 12 inference + 10 bridge points

| Component | Time |
|-----------|------|
| 12 inference points | ~14s |
| 10 bridge points | ~4s |
| **Total per eval** | **~18s** |

### 6.4. Projected cost: 20 inference + 5 bridge points

| Component | Time |
|-----------|------|
| 20 inference points | ~22s |
| 5 bridge points | ~2s |
| **Total per eval** | **~24s** |

### 6.5. Budget summary

All three placements are well within the 60s budget. The 15-point placement with 7 bridges at ~20s/eval is the recommended default.

---

## 7. Steric Convergence at Large a

### 7.1. The steric divergence problem

From the convergence study (`StudyResults/charged_voltage_range_study/`):
- a=0.05: converges to eta=-46.5 in 124s
- a=0.10: converges to eta=-46.5 in 133s
- a=0.20: DIVERGES at eta=-1.63 (DIVERGED_DTOL after 8 Newton iters)
- a=0.30 (round 2): DIVERGES at eta=-0.47
- a=0.40 (round 2): DIVERGES at eta=-0.09

### 7.2. Root cause

The Bikerman steric term mu_steric = ln(1 - sum(a_j c_j)) has a singularity when sum(a_j c_j) -> 1. For a=0.20 with bulk c_O2=1.0, c_H2O2 initially near 0, c_H+=0.2, c_ClO4-=0.2:
- sum = 0.20*(1.0 + 0.0 + 0.2 + 0.2) = 0.28
- The gradient of mu_steric = -sum(a_j grad(c_j)) / (1 - sum(a_j c_j)) has a 1/(1-0.28) = 1.39 amplification.
- As H2O2 is produced at the electrode (c_H2O2 ~ 1.8 at moderate eta), sum -> 0.20*(0.33+1.86+0.05+0.2) = 0.488, and the amplification becomes 1/(1-0.488) = 1.95.
- For a=0.20, if c_H2O2 transiently exceeds 3.0 during Newton iteration, sum > 0.8 and the amplification exceeds 5x, causing the Newton Jacobian to blow up.

### 7.3. Safeguard: floor the packing fraction

Location: `Forward/bv_solver.py`, line 572-573.

Current implementation already floors:
```python
packing = fd.max_value(
    fd.Constant(1.0) - sum(steric_a_funcs[j] * ci[j] for j in range(n)),
    fd.Constant(1e-8),
)
```

The floor of 1e-8 is sufficient to prevent log(0), but does NOT prevent the gradient blow-up during Newton iteration when packing transiently approaches zero.

**Recommendation**: Raise the packing floor to 0.01 (1% minimum void fraction):
```python
packing = fd.max_value(
    fd.Constant(1.0) - sum(steric_a_funcs[j] * ci[j] for j in range(n)),
    fd.Constant(0.01),
)
```

This caps the steric amplification at 1/0.01 = 100x, which is large but bounded.

### 7.4. Safeguard: bound the optimizer

For steric inference, the optimizer should be bounded:
```python
steric_a_lower = 0.001
steric_a_upper = 0.15   # reduced from current 0.5
```

At a=0.15, max sum = 0.15*(1+2+0.2+0.2) = 0.51, packing = 0.49 — still safe.
At a=0.20, max sum = 0.20*(1+2+0.2+0.2) = 0.68, packing = 0.32 — marginal.
At a=0.50, max sum > 1.0 — guaranteed singularity.

### 7.5. Continuation in a-space

For robust inference with large steric parameters, consider a STAGED approach:
1. Run optimizer with a_upper=0.10 first.
2. If converged, increase a_upper to 0.15 using the previous solution as warm-start.
3. If the forward solver diverges at any candidate a value during inference, return fail_penalty (already implemented).

---

## 8. Implementation Plan

### 8.1. Phase 1 — Minimal viable extension (Approach B, ~2-3 hours)

**Goal**: Get inference working at 15 points spanning [-1, -46.5] with automatic bridge point insertion.

**Files to modify**:

1. **`FluxCurve/bv_point_solve.py`**
   - Add a `max_eta_gap` parameter to `solve_bv_curve_points_with_warmstart()` (line 57, add to signature; default 3.0).
   - Before the main sweep loop (line 189), compute automatic bridge point locations based on `max_eta_gap`.
   - In the sweep loop, when processing a bridge point: solve under `adj.stop_annotating()`, update `carry_U_data` and predictor state, skip adjoint. Return `PointAdjointResult` with `converged=True, objective=0.0, gradient=zeros`.
   - Increase `_WARMSTART_MAX_STEPS` from 15 to 20 (line 43).
   - Increase `_SER_DT_MAX_RATIO` from 20.0 to 40.0 (line 54) for bridge points.

2. **`FluxCurve/bv_curve_eval.py`**
   - When summing objective and gradient over PointAdjointResults, skip entries where `phi_applied` is not in the original `phi_applied_values` (bridge points).
   - Or: mark bridge points with a flag in PointAdjointResult.

3. **`FluxCurve/results.py`**
   - Add `is_bridge: bool = False` field to `PointAdjointResult`.

4. **Inference scripts** (e.g., `scripts/inference/Infer_BVSteric_charged_from_current_density_curve.py`)
   - Change `eta_values = np.linspace(-1.0, -10.0, 10)` to the 15-point placement.
   - Add `max_eta_gap=3.0` parameter to the request (passed through to bv_point_solve).

### 8.2. Phase 2 — Quadratic predictor and dt carry-forward (~1-2 hours)

**Files to modify**:

1. **`FluxCurve/bv_point_solve.py`**
   - Add `predictor_prev2` state variable (before line 186).
   - Replace predictor logic (lines 240-261) with quadratic/linear hybrid.
   - Add concentration clamp after prediction.
   - Carry converged dt from bridge points to next point (skip reset at line 388 for bridges).

### 8.3. Phase 3 — Approach A for production (Approach A, ~3-4 hours)

**Files to modify**:

1. **`FluxCurve/bv_config.py`**
   - Add `inference_point_mask: Optional[Sequence[bool]]` to `BVFluxCurveInferenceRequest` (after line 26).
   - Add `bridge_eta_gap: float = 3.0` for automatic bridge insertion.

2. **`FluxCurve/bv_point_solve.py`**
   - Accept `inference_mask` parameter. When False for a point, solve forward-only (no tape, no adjoint).
   - This is cleaner than auto-inserting bridge points because the user controls exactly which points are inference vs bridge.

3. **`FluxCurve/bv_curve_eval.py`**
   - Filter results by inference mask before summing.

4. **`FluxCurve/bv_run.py`**
   - Pass mask through to curve evaluator.

### 8.4. Phase 4 — Steric robustness improvements (~1 hour)

1. **`Forward/bv_solver.py`**
   - Raise packing floor from 1e-8 to 0.01 (line 573).
   - Or: make the floor configurable via `bv_convergence.packing_floor` (alongside `conc_floor`).

2. **Inference scripts**
   - Reduce `steric_a_upper` from 0.5 to 0.15.

---

## 9. Recommended Configuration for Initial Test

### 9.1. 15-point steric inference script

```python
# Point placement
eta_values = np.array([
    -1.0, -2.0, -3.0,           # onset
    -4.0, -5.0, -6.5, -8.0,     # transition
    -10.0, -13.0,                # knee
    -17.0, -22.0, -28.0,        # plateau (sparse)
    -35.0, -41.0, -46.5,        # deep plateau
])

# Solver settings
dt = 0.5
max_ss_steps = 100       # generous for cold/first point
t_end = dt * max_ss_steps

steady = SteadyStateConfig(
    relative_tolerance=1e-4,
    absolute_tolerance=1e-8,
    consecutive_steps=4,
    max_steps=max_ss_steps,
    verbose=False,
)

# Bridge point auto-insertion
max_eta_gap = 3.0  # auto-insert bridge points for gaps > 3.0

# Steric bounds
steric_a_lower = 0.001
steric_a_upper = 0.15

# Optimizer
optimizer_options = {"maxiter": 50, "ftol": 1e-12, "gtol": 1e-6, "disp": True}
```

### 9.2. Expected bridge points (auto-generated with max_eta_gap=3.0)

Between the 15 inference points, gaps exceeding 3.0:
- eta=-13 to -17 (gap=4.0): 1 bridge at -15.0
- eta=-17 to -22 (gap=5.0): 1 bridge at -19.5
- eta=-22 to -28 (gap=6.0): 1 bridge at -25.0
- eta=-28 to -35 (gap=7.0): 2 bridges at -30.33, -32.67
- eta=-35 to -41 (gap=6.0): 1 bridge at -38.0
- eta=-41 to -46.5 (gap=5.5): 1 bridge at -43.75

Total: 7 bridge points. Total points solved per eval: 22.

### 9.3. Expected wall time per evaluation

- 15 inference points at ~1.1s each (setup+forward+adjoint) = 16.5s
- 7 bridge points at ~0.4s each (forward-only, no adjoint overhead) = 2.8s
- Total: ~19s per eval

With 50 optimizer iterations: ~16 minutes total. Well within the 60s/eval budget.

---

## 10. Risk Assessment

### 10.1. Low risk (expected to work)

- **Bridge points converging**: Forward sweep data shows all 100 densely-spaced points converge easily. Bridge points with gap <= 3.0 should converge in 8-12 warm-started PTC steps.
- **Adjoint at large eta**: The adjoint linear solve uses the same MUMPS factorization as the forward solve. Since the forward converges on the plateau (4 PTC steps, well-conditioned), the adjoint should be fine.
- **Cross-eval warm-start**: The P2 cache (`_cross_eval_cache`, line 49) already handles optimizer iteration N -> N+1. This works unchanged with the extended range since it caches the first point (eta=-1).

### 10.2. Medium risk (may need tuning)

- **Predictor overshoot at large gaps**: The linear predictor may overshoot between eta=-13 and -15 (bridge), producing negative concentrations. Mitigation: clamp concentrations after prediction. Worst case: fall back to simple warm-start (copy U, no extrapolation).
- **SER dt dynamics**: If the SER shrinks dt too aggressively after a large gap, the system may need more than _WARMSTART_MAX_STEPS to recover. Mitigation: increase cap to 20 for inference, 40 for bridge.
- **Steric a > 0.1 combined with large eta**: The forward solver may diverge at intermediate eta values (not the inference points themselves, but the bridge points). Mitigation: if a bridge point fails, subdivide the gap (insert more bridges).

### 10.3. High risk (may require significant work)

- **Steric a > 0.20**: The packing fraction singularity makes the forward solver fundamentally unstable at large a. Current optimizer bound of a_upper=0.5 allows the optimizer to probe catastrophically bad parameter values. Mitigation: reduce upper bound to 0.15.
- **Gradient accuracy at large eta**: On the flat plateau, the gradient dJ/da is very small. The adjoint may return noisy gradients due to finite-precision arithmetic. If the optimizer struggles on the plateau, consider FINITE DIFFERENCES for gradient verification (`FluxCurve/bv_curve_eval.py` could implement FD gradient check mode).
- **Total wall time scaling**: If the optimizer needs MORE iterations with 15 points (more parameters in play), the total time could exceed the budget. Mitigation: keep maxiter at 40-50, use warm-start from the 10-point solution as initial guess.

---

## 11. Validation Strategy

### 11.1. Forward solve validation

Before running inference, verify the forward sweep converges at all 22 points (15 inference + 7 bridge) with the target steric parameters:

```bash
# Test forward sweep with a=0.05
python scripts/bv/bv_iv_curve_charged.py --steps 100
```

### 11.2. Gradient verification

At the converged parameter values, compare adjoint gradient to finite differences:

```python
# In inference script, after optimization:
eps = 1e-6
for i in range(n_controls):
    a_plus = a_best.copy(); a_plus[i] += eps
    a_minus = a_best.copy(); a_minus[i] -= eps
    J_plus = forward_eval(a_plus)
    J_minus = forward_eval(a_minus)
    fd_grad_i = (J_plus - J_minus) / (2*eps)
    print(f"a[{i}]: adjoint={adj_grad[i]:.6e}, FD={fd_grad_i:.6e}, "
          f"ratio={adj_grad[i]/fd_grad_i:.4f}")
```

### 11.3. Point-by-point timing

Enable timing output (already present, line 460-468) and verify:
- Bridge points complete in < 0.5s each
- Inference points complete in < 1.5s each (setup+forward+adjoint)
- Total per-eval is < 25s

---

## 12. Summary of Changes by File

| File | Change | Priority |
|------|--------|----------|
| `FluxCurve/bv_point_solve.py` | Auto bridge points, increase _WARMSTART_MAX_STEPS, _SER_DT_MAX_RATIO | Phase 1 |
| `FluxCurve/results.py` | Add `is_bridge` flag to PointAdjointResult | Phase 1 |
| `FluxCurve/bv_curve_eval.py` | Filter bridge points from objective/gradient sum | Phase 1 |
| `scripts/inference/Infer_BVSteric_*.py` | Update eta_values to 15-point placement | Phase 1 |
| `FluxCurve/bv_point_solve.py` | Quadratic predictor, dt carry-forward | Phase 2 |
| `FluxCurve/bv_config.py` | Add inference_point_mask, bridge_eta_gap fields | Phase 3 |
| `FluxCurve/bv_run.py` | Pass mask through to evaluator | Phase 3 |
| `Forward/bv_solver.py` | Raise packing floor to 0.01 | Phase 4 |
| `scripts/inference/Infer_BV*.py` | Reduce steric_a_upper to 0.15 | Phase 4 |

---

## Appendix A: Full I-V Curve Data Points (from forward sweep, a=0)

Sampled at every 10th point from the 100-point sweep:

| eta_hat | I_peroxide (mA/cm2) | ss_steps | c_O2 at electrode |
|---------|---------------------|----------|-------------------|
| -0.47 | -0.00058 | 14 | 0.997 |
| -4.65 | -0.116 | 17 | 0.333 |
| -9.30 | -0.145 | 9 | 0.121 |
| -13.95 | -0.154 | 9 | 0.085 |
| -18.60 | -0.161 | 8 | 0.061 |
| -23.26 | -0.166 | 7 | 0.043 |
| -27.91 | -0.170 | 6 | 0.029 |
| -32.56 | -0.173 | 5 | 0.018 |
| -37.21 | -0.175 | 4 | 0.011 |
| -41.86 | -0.176 | 4 | 0.006 |
| -46.51 | -0.176 | 4 | 0.003 |

## Appendix B: Steric Sensitivity at Key Voltage Points

For a = [0.05, 0.05, 0.05, 0.05]:

| eta_hat | sum(a_j*c_j) at electrode | mu_steric | grad(mu_steric) relative magnitude |
|---------|--------------------------|-----------|-----------------------------------|
| -1.0 | 0.068 | -0.070 | Moderate |
| -5.0 | 0.122 | -0.130 | Peak |
| -10.0 | 0.058 | -0.060 | Moderate (concentrations depleting) |
| -20.0 | 0.015 | -0.015 | Small |
| -46.5 | 0.010 | -0.010 | Negligible at electrode |

Note: The steric effect also operates through the BULK-to-electrode concentration gradient. The gradient of mu_steric drives additional flux throughout the boundary layer interior, not just at the electrode surface. This means plateau points DO carry some steric information, even though the electrode-surface steric signal is weak.
