# V17 Investigation Log — Physical E_eq Inverse Pipeline

**Date**: 2026-04-06
**Investigator**: Claude (autonomous research session)

## Executive Summary

The v16 inference pipeline had a fundamental voltage-mapping error: E_eq was set to 0 for both reactions, causing the dimensionless voltage grid to misalign with the physical operating regime. This investigation diagnoses the issue, maps the solver convergence boundaries, and tests parameter recovery with corrected physics.

---

## 1. Root Cause: Voltage Grid / E_eq Mismatch

### The Problem
- `_bv_common.py` sets `E_eq_r1 = 0.0, E_eq_r2 = 0.0` (defaults)
- Physical values: E_eq_r1 = 0.68V vs RHE (O2 → H2O2), E_eq_r2 = 1.78V vs RHE (H2O2 → H2O)
- The BV overpotential: η_j = phi_applied_hat - E_eq_j / V_T
- V_T = 0.02569V, so E_eq_r1/V_T = 26.5, E_eq_r2/V_T = 69.3

### Impact on v16
- v16's grid: phi_hat from -46.5 to +5.0
- With E_eq=0: η_r1 = phi_hat (range -46.5 to +5.0, spanning cathodic and anodic)
- With physical E_eq: η_r1 = phi_hat - 26.5 (range -73 to -21.5, ALL cathodic!)
- **The v16 grid NEVER reaches equilibrium for either reaction when E_eq is physical**
- This explains why E_eq=0 was "working" — it accidentally placed the onset in the grid

### The Fix
- Map voltage as V_RHE / V_T for phi_hat
- Experimental range: -0.5V to 1.25V vs RHE → phi_hat from -19.5 to +48.7
- With physical E_eq, the onset region (near E_eq_r1=0.68V) is at phi_hat ≈ 26.5

---

## 2. Forward Solver Diagnostics

### Script: `scripts/studies/diagnostic_eeq_voltage_sweep.py`

**Cathodic grid (-0.5V to 0.3V vs RHE, 14 points)**:
- E_eq=0: 14/14 converged, 10/14 full z=1.0
- Physical E_eq: 14/14 converged, 10/14 full z=1.0 (but partial z at positive phi_hat)
- Key finding: Physical E_eq gives correct onset at ~0.1V (near where data becomes non-trivial)

**Full grid (-0.5V to 1.25V vs RHE, 31 points)**:
- E_eq=0: STUCK on z-ramp at phi_hat=+48.7 (deep anodic, solver diverges)
- Physical E_eq: 31/31 values obtained, but only 12/31 full z=1.0

### Charge Continuation z-Convergence Window
- **Full z=1.0 convergence**: V_RHE ∈ [-0.50V, +0.10V] (phi_hat -19.5 to +3.9)
- **Partial z (0.0-0.4)**: V_RHE > +0.10V (solver reaches ~40% charge coupling)
- **z=0 (neutral only)**: V_RHE < -0.60V (extremely stiff BV exponentials)
- Convergence window is the same for all parameter combinations tested

---

## 3. Curve Shape Analysis

### Script: `scripts/studies/diagnostic_eeq_shape_search.py`

Experimental data (DataPlot.png) features:
- Onset ~0.3V vs RHE
- Peak (most negative peroxide) ~-0.35 mA/cm² near 0V
- Dip and recovery (less negative) at cathodic voltages
- Plateau at ~-0.15 to -0.2 mA/cm² at -0.5V

Model with physical E_eq (baseline k0 values):
- Onset near 0.7V (at E_eq_r1, thermodynamically correct)
- Transport-limited current: ~0.18 mA/cm² (matches experiment at -0.5V)
- No "dip and recovery" — peroxide current monotonically increases toward transport limit
- Total current roughly flat at -0.17 to -0.18 mA/cm²

### Shape Match v2: `scripts/studies/shape_match_v2.py`
Tested 6 parameter combinations with extended cathodic range (-1V to 0.3V):
- Increasing k0_r2 up to 1000x: reduces peroxide current but no dip shape
- E_eq_r2=0.4V: subtle hint of dip at -0.4V (closest to experimental shape)
- alpha_r2=0.8: all peroxide consumed (pc→0 everywhere)
- The "dip" requires reaction 2 to consume H2O2 at moderate potentials faster than reaction 1 produces it

### Why the Model Doesn't Show the Experimental "Dip"
1. H2O2 surface concentration is determined by production-diffusion balance
2. With L_ref=100µm, H2O2 diffuses away efficiently, keeping surface concentration low
3. Reaction 2 (H2O2 consumption) is rate-limited by H2O2 surface concentration
4. Need shorter L_ref or higher k0_r2 to see significant H2O2 consumption
5. The paper's model (Mangan2025) achieves the dip with L_diff=66-86µm

---

## 4. Inference Recovery Test (v17)

### Script: `scripts/Inference/Infer_BVMaster_charged_v17.py`

First test (0% noise, 2x offset initial guess):
- k0_1 error: 91.5%
- k0_2 error: 49.3%  
- alpha_1 error: 52.1%
- alpha_2 error: 44.2%
- **POOR RECOVERY even with perfect data**

### Diagnosis
The poor recovery is NOT due to E_eq handling (verified correct through full pipeline trace). Possible causes:
1. **Solver failures at trial parameters**: optimizer tries k0/alpha combos that cause SNES divergence → penalty → corrupted gradient
2. **Identifiability**: in the transport-limited regime, total current is nearly flat, providing minimal gradient information
3. **k0-alpha tradeoff**: BV kinetics have a known k0 × exp(α * η) degeneracy — different k0/α combos can produce similar rates

### Recovery test (running): tests convergence from true, 20%, and 2x initial guesses

---

## 5. Key Decisions

| Decision | Rationale |
|----------|-----------|
| E_eq_r1 = 0.68V, E_eq_r2 = 1.78V | Standard thermodynamic values for ORR |
| Voltage grid: -0.5V to +0.1V | Reliable z=1.0 convergence window |
| PDE cold start (no surrogate) | Surrogates trained on E_eq=0 data |
| SNES maxiter=400, maxlambda=0.3 | Tuned for physical E_eq robustness |
| dt=0.25, t_end=80 | Smaller dt + longer time for stiff points |

---

## 5b. IC Cache Discovery (critical)

The charge continuation IC cache is **parameter-specific**. When the optimizer changes k0/alpha, the cached ICs don't transfer, and the sequential solver fails at V_RHE >= 0V (phi_hat >= 0). This is because:
- The IC is the steady-state solution at specific k0/alpha
- Different parameters produce different steady-state profiles
- At positive phi_hat, the BV kinetics are sensitive to parameter changes
- At negative phi_hat, the transport-limited regime is parameter-insensitive (current is capped by diffusion)

**Fix**: Restrict inference to V_RHE < 0V where the solver converges reliably across parameter space.

## 5c. Recovery Test Results (test_v17_recovery.py)

8-point cathodic grid (V_RHE = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1]):

| Init | k0_1 err | k0_2 err | α1 err | α2 err | Loss |
|------|---------|---------|--------|--------|------|
| TRUE | **2.7%** | **0.6%** | 67.0% | 61.9% | 2.7e-2 |
| 20% off | 13.0% | 21.5% | 42.0% | 38.6% | 3.6e-3 |
| 2x off | 93.0% | 49.6% | 47.6% | 47.9% | 7.5e-3 |

**Key finding**: Alpha is non-identifiable in the transport-limited regime. Even from TRUE parameters, the optimizer DRIFTS alpha from 0.627 to 0.207 (67% error). k0 is identifiable (2.7% from true).

**Physics explanation**: In the transport-limited regime, current is set by diffusion (D, c_bulk, L_ref), not kinetics (k0, alpha). Alpha only matters at the onset where the kinetics→transport transition happens. That transition is at V_RHE > 0V where the solver can't handle parameter perturbations.

**Implication for surrogate warm-start**: A good surrogate can get k0 within 10-20%, but alpha recovery requires extending the solver to the onset region OR fixing alpha and only inferring k0.

## 5d. Robust Forward Solver + Inference Result (BREAKTHROUGH)

**New module**: `Forward/bv_solver/robust_forward.py`
- Parallelizes Phase 2 (z-ramp) across voltage points via multiprocessing
- 10/10 converged, numerically identical to sequential (diff < 1e-9)
- Single function call: `solve_curve_robust(sp, phi_hat, scale)`
- IC cache integration: `populate_ic_cache_robust(sp, phi_hat, scale)`

**v17_robust_inference.py** (10% offset, 0% noise):
```
eval  1:  J=5.74e-06  (initial)
eval 12:  J=1.70e-07  (alpha converging)
eval 15:  J=5.79e-10  (converged)
eval 16:  J=5.23e-10  (final)

k0_1:    err 9.9%  (stuck at initial offset — non-identifiable)
k0_2:    err 10.1% (stuck at initial offset — non-identifiable)
alpha_1: err 0.0%  (RECOVERED PERFECTLY)
alpha_2: err 0.0%  (RECOVERED PERFECTLY)
```

**Key insight**: With proper IC cache (from charge continuation at initial guess params):
- Adjoint fast-path works: 16/16 evaluations converge, no penalties
- Alpha IS identifiable in [-0.5V, +0.1V] (peroxide selectivity ratio encodes alpha)
- k0 is NOT identifiable (transport-limited, loss flat w.r.t. k0)
- Surrogate should be designed to estimate k0 (from onset region shape), then PDE refines alpha

## 5e. Hybrid z=0/z=1 Solver (WORKING)

**Module**: `Forward/bv_solver/hybrid_forward.py`

- z=0 (neutral) for V > 0.10V: captures onset shape, BV kinetics, converges everywhere
- z=1 (charged) for V <= 0.10V: accurate electromigration, parallel charge continuation
- 16/17 points across -0.5V to +0.7V in 25 seconds
- Discontinuity at transition (z=0 currents ~50% of z=1 due to missing electromigration)
- The onset region (0.3-0.7V) is captured by z=0 points

**Log-transform finding**: Does NOT extend the z=1 convergence boundary. The barrier is a genuine Jacobian singularity from the Poisson singular perturbation (ε~1.8e-7), confirmed by:
- Even 0.001 z-step from z=0.79 solution fails (test_push_z_from_partial.py)
- SNES diagnostics show residual DIVERGENCE at every Newton step (line search saturates at lambda=0.21)
- The depletion zone creates near-singular MUMPS pivots

## 6. Open Questions

1. **Can parameter recovery work at all in [-0.5, +0.1]V?** The recovery test will tell us.
2. **Is L_ref the key to matching the experimental curve shape?** Paper suggests yes (66-86µm).
3. **Should we extend the voltage window?** Requires improving charge continuation for positive eta.
4. **Are 13 voltage points enough?** May need denser sampling in the transition region.
5. **Would fixing alpha and only inferring k0 improve identifiability?**

---

## 7. Files Created (no existing files modified)

- `scripts/studies/diagnostic_eeq_voltage_sweep.py` — Forward sweep with V vs RHE mapping
- `scripts/studies/diagnostic_eeq_shape_search.py` — k0_r2 sweep for curve shape
- `scripts/studies/shape_match_v2.py` — Extended cathodic range + multi-parameter study
- `scripts/Inference/Infer_BVMaster_charged_v17.py` — Corrected inference pipeline
- `scripts/Inference/test_v17_recovery.py` — Parameter recovery diagnostic
- `StudyResults/diagnostic_eeq_sweep/` — Forward sweep results and plots
- `StudyResults/shape_match_v2/` — Shape matching results and z-convergence map
- `StudyResults/master_inference_v17/` — v17 inference output
