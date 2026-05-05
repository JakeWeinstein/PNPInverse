# Peroxide-window proton-electrochemical-potential (muh) sweep

Study script: `scripts/studies/peroxide_window_muh_test.py`.
Plan: `~/.claude/plans/look-at-docs-electrochemical-potential-s-misty-trinket.md`.
Forms: `Forward/bv_solver/forms_logc_muh.py`.

## Configuration

- Formulation: `logc_muh` (Phase 2 hybrid: H+ as mu_H)
- V_RHE grid: [0.6, 0.66, 0.68, 0.7, 0.75, 0.8, 1.0]
- C_S grid (F/m²): ['None', 0.05, 0.1, 0.2, 0.4, 1.0]
- Mesh Ny: 200 (graded, beta=3, Nx=8)
- exponent_clip: 100.0
- Initializer: debye_boltzmann
- Stack: 3sp + Boltzmann ClO4- + log-c + log-rate + muh(H+)
- Orchestrator: `solve_grid_per_voltage_cold_with_warm_fallback` (C+D, n_substeps_warm=8, bisect_depth_warm=5)

## Convergence matrix (rows: C_S, cols: V_RHE)

| C_S \ V_RHE | +0.60 | +0.66 | +0.68 | +0.70 | +0.75 | +0.80 | +1.00 |
|---|---|---|---|---|---|---|---|
| None | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 0.05 | ✓ | ✓ | ✓ | ✓ | ✓ (warm) | ✓ (warm) | ✓ (warm) |
| 0.1 | ✓ | ✓ | ✓ | ✓ | ✓ (warm) | ✓ (warm) | ✓ (warm) |
| 0.2 | ✓ | ✓ | ✓ | ✓ | ✓ (warm) | ✓ (warm) | ✓ (warm) |
| 0.4 | ✓ | ✓ | ✓ | ✓ | ✓ (warm) | ✓ (warm) | ✓ (warm) |
| 1 | ✓ | ✓ | ✓ | ✓ | ✓ (warm) | ✓ (warm) | ✓ (warm) |

## Validation gates

### Gate 1 — Code regression: muh vs logc at C_S=None (HARD): **PASS**

PASS: muh C_S=None reproduces logc C_S=None CD/PC at all converged voltages within rel_tol=1e-06.  The muh transform is algebraic; this is the load-bearing check that mu_H reconstruction recovers the same physical c_H.

### Gate 2 — Numerical convergence at peroxide V (REQUIRED): **PASS**

PASS: muh at finite C_S converged at all of [0.68, 0.7, 0.75] V.
  - C_S=0.05 F/m²: all converged.
  - C_S=0.1 F/m²: all converged.
  - C_S=0.2 F/m²: all converged.
  - C_S=0.4 F/m²: all converged.
  - C_S=1 F/m²: all converged.

### Gate 3 — Physical validity: c_ClO4 ≤ steric cap (PHYSICS): **WARN**

WARN: 5 row(s) Newton-converged but exceeded steric cap.  These states are non-physical; use only as numerical diagnostics, NOT inverse-problem ground truth.
  - C_S=0.05: V=+0.68: False, V=+0.70: False, V=+0.75: False
  - C_S=0.1: V=+0.68: False, V=+0.70: False, V=+0.75: False
  - C_S=0.2: V=+0.68: False, V=+0.70: False, V=+0.75: False
  - C_S=0.4: V=+0.68: False, V=+0.70: False, V=+0.75: False
  - C_S=1: V=+0.68: False, V=+0.70: False, V=+0.75: False

### Gate 4 — Cross-formulation comparison (INFO): **INFO**

INFO: no overlapping converged points to compare.

## Decision

**Decision: muh extends Newton convergence but does not solve physical validity.** The analytic Boltzmann ClO4- residual still produces c_ClO4_surface > steric cap at peroxide voltages.  This was the predicted outcome (per `~/.claude/plans/look-at-docs-electrochemical-potential-s-misty-trinket.md` Phase 7 caveat).  Proceed to Phase 7 (charged-species mu on 4sp+Bikerman) for a physically valid solver at peroxide V.

## Peroxide-window observables (CD, mA/cm²)

| C_S | +0.66 | +0.68 | +0.70 | +0.75 | +0.80 | +1.00 |
|---|---|---|---|---|---|---|
| None | +1.292e-08 | — | — | — | — | — |
| 0.05 | -2.043e-05 | -1.733e-05 | -1.619e-05 | -1.549e-05 | -1.545e-05 | -1.544e-05 |
| 0.1 | -1.753e-05 | -1.625e-05 | -1.580e-05 | -1.546e-05 | -1.545e-05 | -1.544e-05 |
| 0.2 | -1.634e-05 | -1.583e-05 | -1.565e-05 | -1.545e-05 | -1.545e-05 | -1.544e-05 |
| 0.4 | -1.587e-05 | -1.566e-05 | -1.559e-05 | -1.545e-05 | -1.545e-05 | -1.543e-05 |
| 1 | -1.565e-05 | -1.559e-05 | -1.556e-05 | -1.545e-05 | -1.545e-05 | -1.540e-05 |

## Surface c_ClO4 (nondim; steric cap ~100)

| C_S | +0.60 | +0.66 | +0.68 | +0.70 | +0.75 | +0.80 | +1.00 |
|---|---|---|---|---|---|---|---|
| None | **2.77e+09** | **2.87e+10** | **6.24e+10** | **1.36e+11** | **9.52e+11** | **6.67e+12** | **1.60e+16** |
| 0.05 | **2.50e+02** | **3.19e+02** | **3.44e+02** | **3.70e+02** | **4.39e+02** | **5.14e+02** | **8.82e+02** |
| 0.1 | **8.51e+02** | **1.10e+03** | **1.19e+03** | **1.29e+03** | **1.55e+03** | **1.83e+03** | **3.21e+03** |
| 0.2 | **2.86e+03** | **3.77e+03** | **4.11e+03** | **4.46e+03** | **5.40e+03** | **6.45e+03** | **1.16e+04** |
| 0.4 | **9.46e+03** | **1.27e+04** | **1.39e+04** | **1.52e+04** | **1.87e+04** | **2.25e+04** | **4.17e+04** |
| 1 | **4.46e+04** | **6.20e+04** | **6.85e+04** | **7.53e+04** | **9.40e+04** | **1.15e+05** | **2.22e+05** |

Bold values exceed the Bikerman steric scale and indicate a non-physical converged state.

## mu_H surface mean (muh-specific)

Raw mu_H = u_H + em*z_H*phi at the electrode.  The smaller the y-range of mu_H over the domain, the closer the diffuse layer is to Boltzmann equilibrium (analytic-cancellation property).  Bulk mu_H = log(c_H_bulk) ≈ -1.6 nondim.

| C_S | +0.60 | +0.66 | +0.68 | +0.70 | +0.75 | +0.80 | +1.00 |
|---|---|---|---|---|---|---|---|
| None | -1.612 | -1.665 | -1.614 | -1.616 | -1.611 | -1.610 | -1.610 |
| 0.05 | -1.613 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 |
| 0.1 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 |
| 0.2 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 |
| 0.4 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 |
| 1 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 | -1.612 |

## Stern voltage drop, phi_m - phi_s (nondim)

| C_S | +0.60 | +0.66 | +0.68 | +0.70 | +0.75 | +0.80 | +1.00 |
|---|---|---|---|---|---|---|---|
| 0.05 | +16.221 | +18.314 | +19.018 | +19.724 | +21.498 | +23.285 | +30.530 |
| 0.1 | +14.997 | +17.074 | +17.772 | +18.474 | +20.238 | +22.017 | +29.237 |
| 0.2 | +13.785 | +15.844 | +16.537 | +17.234 | +18.987 | +20.757 | +27.951 |
| 0.4 | +12.589 | +14.627 | +15.315 | +16.006 | +17.747 | +19.507 | +26.673 |
| 1 | +11.038 | +13.044 | +13.723 | +14.406 | +16.130 | +17.875 | +25.004 |

## Artifacts

- `StudyResults/peroxide_window_muh_test/iv_curve.json` — per-C_S CD/PC and convergence.
- `StudyResults/peroxide_window_muh_test/diagnostics.json` — full per-(C_S, V) diagnostic dump.
- `StudyResults/peroxide_window_muh_test/results.csv` — flat per-row dataset.
- `StudyResults/peroxide_window_muh_test/comparison.png` — CD/PC + Stern drop vs V (when matplotlib is available).

