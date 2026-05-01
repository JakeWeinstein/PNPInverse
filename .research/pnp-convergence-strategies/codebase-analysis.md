# Codebase Analysis: PNP Convergence at High Applied Potentials

## 1. How z_consts Are Used in the Forms

**Files:** `Forward/bv_solver/forms.py` (L197, L309-314), `Forward/bv_solver/grid_charge_continuation.py` (L435-436)

The `z_consts` are `fd.Constant` objects stored in `ctx["z_consts"]` (forms.py L197):
```python
z = [fd.Constant(float(z_vals[i])) for i in range(n)]
```

They appear in exactly two places in the weak form:

1. **Nernst-Planck electromigration drift** (forms.py L310):
   ```python
   drift = em * z[i] * phi
   Jflux = D[i] * (grad(c_i) + c_i * grad(drift))
   ```
   When z=0, `drift=0` and `Jflux = D[i] * grad(c_i)` -- pure diffusion, no electric field coupling.

2. **Poisson charge source** (forms.py L415):
   ```python
   F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
   ```
   When z=0, the Poisson RHS is zero. The Poisson equation becomes `eps_coeff * laplacian(phi) = 0` (decoupled from species).

**Key insight:** Setting z=0 eliminates both the electromigration drift AND the Poisson source term. This decouples the Poisson equation from transport, removing the dominant stiffness source (the tiny `eps_coeff ~ 9e-8` multiplying the Laplacian against order-1 charge terms). The z-ramp from 0 to 1 gradually re-introduces this coupling.

## 2. Exact Poisson Equation Weak Form

**File:** `Forward/bv_solver/forms.py` (L411-415)

```python
eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))   # (lambda_D/L)^2
charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"])) # 1.0 in nondim mode
F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
```

In nondimensional mode (from `Nondim/transform.py` L401-403):
```
eps_coeff = (epsilon * V_T) / (F * c_ref * L^2) = (lambda_D / L)^2
```

**What becomes stiff at positive phi_hat:** At large positive phi_hat (anodic), cations are repelled from the electrode while anions accumulate. The EDL depletion zone forms where the co-ion concentration drops to near zero. The Poisson equation must resolve a boundary layer of width ~lambda_D with the tiny coefficient `eps_coeff ~ (lambda_D/L)^2`. For typical parameters:
- `lambda_D ~ 43 nm`, `L = 100 um` gives `eps_coeff ~ 1.8e-7`
- This creates a singular perturbation problem: the Poisson equation has a very thin boundary layer where `phi` changes rapidly.

The `verify_model_params` function (transform.py L476) warns when `debye_to_length_ratio < 1e-5` or `> 1.0`, confirming this is a known concern.

## 3. Slotboom / Log-Variable Transform

**No Slotboom or log-variable transform exists in the solver code.** Grep for `slotboom`, `log_var`, `log_transform`, `exponential_variable` found zero hits in the Forward/ directory. The only references are in planning/research documents discussing potential future work.

The current transport equation uses the standard drift-diffusion form:
```python
Jflux = D[i] * (grad(c_i) + c_i * grad(drift))
```

A Slotboom variable `u_i = c_i * exp(z_i * phi)` would transform this to `Jflux = D[i] * exp(-z_i*phi) * grad(u_i)`, which can improve conditioning when `z_i * phi` is large by absorbing the exponential growth into the variable itself.

## 4. Mesh Grading Formula and Smallest Cell Size

**File:** `Forward/bv_solver/mesh.py` (L12-34, L37-66)

Power-law grading: `x_i = (i/N)^beta` for `i = 0, ..., N`.

Smallest cell size (first cell at electrode, i=0 to i=1):
```
h_min = (1/N)^beta
```

Typical configurations used in the codebase:
| Config | N (Ny) | beta | h_min | h_min * L (if L=100um) |
|--------|--------|------|-------|------------------------|
| Standard | 200 | 3.0 | 1.25e-7 | 12.5 nm |
| Fine | 400 | 3.0 | 1.56e-8 | 1.56 nm |
| High grading | 200 | 5.0 | 3.13e-12 | 0.31 pm |
| Fine+graded | 400 | 5.0 | 9.77e-14 | 0.01 pm |

With `lambda_D ~ 43 nm`, the standard mesh (Ny=200, beta=3) has `h_min ~ 12.5 nm`, which is sub-Debye. The extended voltage test (test_extended_voltage_v3.py) tries Ny=400 and beta=5 configurations to see if finer meshes help.

**Note:** beta=5 produces astronomically small cells, which may cause floating-point issues in the finite element assembly.

## 5. BV Exponent Clipping Mechanism

**Files:** `Forward/bv_solver/forms.py` (L236-248), `Forward/bv_solver/config.py` (L65-105)

```python
def _build_eta_clipped(E_eq_const):
    eta_raw = phi_applied_func - E_eq_const   # (use_eta_in_bv=True, no Stern)
    eta_scaled = bv_exp_scale * eta_raw        # bv_exp_scale=1.0 in nondim mode
    if conv_cfg["clip_exponent"]:
        clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))  # default: 50.0
        return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)
    return eta_scaled
```

Default clip at +/-50. In nondim mode with `bv_exp_scale=1.0`, the exponent is clipped at `eta_scaled = 50`.

**At phi_hat ~ 4 V_T:** The overpotential `eta = phi_hat - E_eq`. For E_eq_r1 = 0.68V/V_T ~ 26.4, E_eq_r2 = 1.78V/V_T ~ 69.2. So at phi_hat=4/V_T ~ 155.6:
- Reaction 1: eta ~ 155.6 - 26.4 = 129 (clips to 50)
- Reaction 2: eta ~ 155.6 - 69.2 = 86.4 (clips to 50)

**Yes, the clipping is actively triggering.** The BV exponents are clamped at exp(50) ~ 5.2e21 for the cathodic term and exp(-50) ~ 1.9e-22 for the anodic term. This prevents literal overflow but does NOT help with the EDL depletion stiffness in the Poisson equation, which is the actual failing mechanism.

**Important nuance:** With `use_eta_in_bv=True` (default), `eta = phi_applied - E_eq` is computed from the *applied* potential (a constant), NOT from the solution variable phi. This makes the BV boundary flux a function of surface concentrations only (not phi), which should keep the BV Jacobian well-conditioned. The stiffness is in the interior Poisson equation, not the BV boundary terms.

## 6. SNES Divergence Diagnosis

**Files:** `Forward/bv_solver/grid_charge_continuation.py` (L253-261), `Forward/bv_solver/solvers.py` (L62-64)

The solver catches `fd.ConvergenceError` and `PETSc.Error` but does NOT distinguish between:
- KSP failure (linear solve diverged)
- Line search failure (SNES couldn't reduce residual along Newton direction)
- SNES divergence (residual blew up past `snes_divergence_tolerance`)

The code only returns a boolean success/failure:
```python
except Exception as exc:
    if isinstance(exc, fd.ConvergenceError):
        return False, -1  # sentinel: SNES failure
    if isinstance(exc, PETSc.Error):
        return False, -1
    raise
```

**SNES configuration** (from `scripts/_bv_common.py` L179-193):
```python
"snes_type":                 "newtonls",
"snes_max_it":               200,
"snes_linesearch_type":      "l2",
"snes_linesearch_maxlambda": 0.5,   # max step fraction
"snes_divergence_tolerance": 1e12,
"ksp_type":                  "preonly",
"pc_type":                   "lu",
"pc_factor_mat_solver_type": "mumps",
```

Since `ksp_type=preonly` with `pc_type=lu` (direct solve via MUMPS), the KSP cannot fail in the usual iterative sense. Failure is either:
1. **MUMPS factorization failure** -- Jacobian is numerically singular (near-zero pivot due to EDL depletion)
2. **Line search failure** -- L2 line search with maxlambda=0.5 cannot find a descent direction
3. **SNES divergence** -- residual exceeds `snes_divergence_tolerance=1e12`

**Most likely cause at high phi_hat:** The Jacobian becomes nearly singular in the EDL depletion zone where concentrations approach zero, causing MUMPS to produce a garbage Newton direction, which the line search then rejects.

## 7. Adaptive Mesh Refinement / Different Mesh Strategies

**No AMR exists.** The codebase uses only static graded meshes (`make_graded_interval_mesh`, `make_graded_rectangle_mesh`). The test script `scripts/studies/test_extended_voltage_v3.py` explores different static mesh parameters (Ny, beta) but not dynamic refinement.

The `robust_forward.py` (L92) uses a hardcoded mesh:
```python
mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
```

The `test_extended_voltage_v3.py` tests these configurations:
- A: Ny=200, beta=3.0 (baseline)
- B: Ny=400, beta=3.0 (finer)
- C: Ny=200, beta=5.0 (more grading)
- D: Ny=400, beta=5.0 (both)
- E: z=0.9 (partial charge to avoid full EDL)
- F: dt=0.01 (smaller time step)

## 8. Steady-State Convergence Metric

**File:** `Forward/bv_solver/grid_charge_continuation.py` (L203-251)

The convergence criterion is observable-based (integrated BV flux), NOT residual-based:
```python
flux_val = float(fd.assemble(observable_form))  # integrated current density
delta = abs(flux_val - prev_flux_val)
scale_val = max(abs(flux_val), abs(prev_flux_val), _STEADY_ABS_TOL)
rel_metric = delta / scale_val
is_steady = (rel_metric <= 1e-4) or (delta <= 1e-8)
```

Requires 4 consecutive steady steps (`_STEADY_CONSEC = 4`).

**Potential premature divergence detection issue:** When returning `steps == -1` (SNES failure sentinel), the z-ramp treats this as a hard failure and restores the checkpoint. But the observable metric itself could oscillate near steady state, preventing the 4-consecutive-step criterion from being met, which leads to budget exhaustion (returned as `steps == max_steps`). Budget exhaustion is treated as "usable" (not a hard failure), so this path is actually more forgiving.

**The real risk:** If the SNES fails on the FIRST step of a z-ramp attempt, the code immediately declares failure without any recovery attempt. The PTC time-stepping acts as regularization, but if the initial dt is too large for the current stiffness level, the first Newton solve can fail before the SER adaptive dt has a chance to shrink.

## 9. What Exactly Fails

Based on the code architecture:

1. **The Newton step computation (KSP):** Unlikely to fail directly since MUMPS direct solve is used. But MUMPS can fail with a near-singular matrix when EDL concentrations approach machine zero.

2. **The Newton update acceptance (line search):** The L2 line search with `maxlambda=0.5` is conservative. At high phi_hat, the Newton direction from the nearly-singular Jacobian may be so poor that even a 0.5 step length exceeds the divergence tolerance.

3. **The nonlinear convergence (SNES):** With `snes_error_if_not_converged=True`, any of the above failures raises `ConvergenceError` which is caught by the `_run_to_steady_state` exception handler and mapped to `steps=-1`.

**The failure cascade:** At phi_hat > ~4 V_T:
- Phase 1 (z=0) succeeds easily (no charge coupling)
- Phase 2 z-ramp: `_try_z(1.0)` fails (direct jump too aggressive)
- Binary search: `_try_z(0.5)` may succeed; then geometric acceleration toward 1.0
- At some z_factor ~ 0.8-0.95, the EDL depletion becomes severe enough that MUMPS encounters near-zero pivots, the Newton direction is garbage, line search fails, SNES throws `ConvergenceError`
- The z-ramp gets stuck at the largest achievable z_factor < 1.0

## Key Gaps and Potential Improvements

1. **No Slotboom/log-variable transform** -- This is the most promising missing feature for extending convergence. It would absorb the exponential concentration profiles in the EDL into the variable itself.

2. **No diagnostic output on SNES failure** -- The code cannot distinguish KSP/line-search/residual-blowup failures, making debugging difficult. Adding `snes_monitor`, `snes_linesearch_monitor`, and `ksp_monitor` would help.

3. **No adaptive mesh refinement** -- Static meshes cannot adapt to the voltage-dependent EDL thickness. At high phi_hat, the depletion zone becomes thinner and sharper.

4. **Observable-based convergence may be too lenient** -- An integrated flux can appear steady while local residuals in the EDL are still large. A residual-norm-based criterion would be stricter but more reliable.

5. **The initial dt in PTC/SER is not voltage-dependent** -- The same `dt_initial` is used regardless of phi_hat. Smaller dt_initial at higher voltages would help the first Newton solve succeed.

6. **Exponent clip at 50 is very generous** -- In nondim mode, `exp(50) ~ 5e21`. A tighter clip (e.g., 20-30) combined with proper Slotboom variables could improve conditioning.

## Relevant File Summary

| File | Role |
|------|------|
| `Forward/bv_solver/forms.py` | Weak form assembly, z_consts, BV BC, Poisson |
| `Forward/bv_solver/grid_charge_continuation.py` | Two-phase z-ramp solver with SER adaptive dt |
| `Forward/bv_solver/mesh.py` | Power-law graded mesh construction |
| `Forward/bv_solver/nondim.py` | BV-specific nondimensionalization |
| `Forward/bv_solver/config.py` | Convergence config defaults (clip, floor, etc.) |
| `Forward/bv_solver/solvers.py` | Forward solve with continuation/PTC |
| `Forward/bv_solver/robust_forward.py` | Parallel Phase 2 worker |
| `Forward/bv_solver/observables.py` | Observable form for convergence metric |
| `Nondim/transform.py` | Core nondimensionalization (eps_coeff, scales) |
| `Nondim/constants.py` | Physical constants |
| `scripts/_bv_common.py` | SNES options, species configs |
| `scripts/studies/test_extended_voltage_v3.py` | Extended voltage mesh/strategy tests |
