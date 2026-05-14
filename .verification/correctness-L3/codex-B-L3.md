# Q1. BV eta clipping

## Finding 1
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:380-405`; `Forward/bv_solver/picard_ic.py:47-70`; `Forward/bv_solver/picard_ic.py:906-924`

TRIGGER: All BV configurations using clipped eta.

EVIDENCE: `_build_eta_clipped` builds `eta_scaled = bv_exp_scale * eta_raw` and clips that value with `fd.min_value(fd.max_value(...))` before returning it (`forms_logc_muh.py:393-403`). The BV exponent multiplies the returned `eta_j` by `alpha_j * n_e_j` later in both log-rate and standard branches (`forms_logc_muh.py:553-579`, `forms_logc_muh.py:584-604`). The scalar Picard twin does the same: `_eta_clipped` computes `eta = bv_exp_scale * (eta_drop - E)` and clips that scalar before `_eta_list_from_drop` returns per-reaction eta values (`picard_ic.py:47-70`, `picard_ic.py:906-924`). Therefore the clip is applied before multiplication by `alpha*n_e`.

## Finding 2
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:463-505`; `Forward/bv_solver/forms_logc_muh.py:609-615`; `Forward/bv_solver/forms_logc_muh.py:641-668`; `Forward/bv_solver/forms_logc_muh.py:839`

TRIGGER: Fully clipped eta regions where UFL `min_value/max_value` has zero derivative with respect to eta.

EVIDENCE: BV enters only through boundary source terms after `R_j = cathodic - anodic` (`forms_logc_muh.py:609-615`). The same residual also contains time and NP transport terms for every dynamic species (`forms_logc_muh.py:463-505`), Poisson volume terms (`forms_logc_muh.py:641-660`), and the Stern Robin term when enabled (`forms_logc_muh.py:666-668`). `J_form = fd.derivative(F_res, U)` differentiates the complete residual (`forms_logc_muh.py:839`). Thus clipping can zero the eta-dependent BV Jacobian contribution, but it does not zero the full Jacobian blocks for non-BV transport, Poisson, Stern, or concentration dependence in BV.

## Finding 3
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:198-233`; `Forward/bv_solver/forms_logc_muh.py:380-382`; `Forward/bv_solver/forms_logc_muh.py:1184-1186`

TRIGGER: Arithmetic check for production voltage range.

EVIDENCE: The targeted code reads model-scaled BV quantities from the nondimensionalization layer: `build_model_scaling(...)` is called at `forms_logc_muh.py:206-216`, reaction BV scaling is added at `forms_logc_muh.py:225-233`, and `bv_exp_scale` is read from `scaling["bv_exponent_scale"]` (`forms_logc_muh.py:380-382`, `forms_logc_muh.py:1184`). The default Picard clip is `exponent_clip = 100.0` (`forms_logc_muh.py:1185-1186`). Given the task's external values `V_T = 0.0257 V`, `E_eq = 0.695 V`, and `exponent_clip = 100`, unclipped cathodic eta requires `V_RHE > 0.695 - 100*0.0257 = -1.875 V`; the stated grid `[-0.4, +0.55] V` is not clipped. The literal `V_T = 0.0257`, `T = 298 K`, and the concrete production value of `bv_exp_scale` are not present in the targeted files; they are unverified from these reads.

# Q2. Bikerman closure

## Finding 1
SEVERITY: INFO

LOCATION: `Forward/bv_solver/boltzmann.py:141-159`; `Forward/bv_solver/boltzmann.py:203-254`

TRIGGER: Two or more `steric_mode="bikerman"` counterions, such as Cs+ and SO4^2-.

EVIDENCE: The function gathers all bikerman entries into one `bikerman` list (`boltzmann.py:141-145`), computes bulk occupied volume across all entries (`boltzmann.py:154-159`), builds one `q_k = exp(-z_k * phi_clamped_k)` per ion (`boltzmann.py:215-239`), and uses a single shared denominator `theta_b_const + sum(a_const*c_const*q for p in per_ion_q)` (`boltzmann.py:241-242`). Each counterion concentration is then `c_const*q*free_dyn/denom` using that shared denominator (`boltzmann.py:252-254`). This is a shared-theta, weighted partition function combining both counterion exponents, not independent per-ion closures.

## Finding 2
SEVERITY: MEDIUM

LOCATION: `Forward/bv_solver/boltzmann.py:203-207`; `Forward/bv_solver/forms_logc_muh.py:447-453`

TRIGGER: Local dynamic packing `A_dyn_local >= 1` or total packing `theta_inner <= packing_floor`, especially on coarse meshes or during bad Newton iterates.

EVIDENCE: The closure floors dynamic free volume with `free_dyn = fd.max_value(1.0 - A_dyn_local, 1e-10)` (`boltzmann.py:203-207`). The form later floors total packing with `packing = fd.max_value(theta_inner, packing_floor)` and `mu_steric = -fd.ln(packing)` (`forms_logc_muh.py:447-453`). Algebraically this prevents negative free volume and log singularities, but once the floor is active the derivative of the floored branch is zero. In an overpacked iterate, Newton loses the steric restoring derivative exactly where the physical barrier should be strongest.

## Finding 3
SEVERITY: INFO

LOCATION: `Forward/bv_solver/boltzmann.py:91-102`; `Forward/bv_solver/boltzmann.py:244-269`; `Forward/bv_solver/boltzmann.py:326-340`

TRIGGER: Verifying whether analytic counterions enter the Newton mixed state.

EVIDENCE: `build_steric_boltzmann_expressions` accepts `ci`, `phi`, and `R_space`, but no `W` argument (`boltzmann.py:91-102`). It creates only a shared `fd.Function(R_space, name="boltzmann_z_scale")` for continuation scaling (`boltzmann.py:244-250`) and returns `StericBoltzmannBundle` UFL expressions (`boltzmann.py:252-269`). `fd.TestFunctions(W)` appears only in `add_boltzmann_counterion_residual`, the residual-mutating path (`boltzmann.py:326-340`), not in the steric closure builder. Cs+ and SO4^2- therefore are closed-form UFL functionals of `phi` and dynamic `ci`, not mixed-state unknowns.

# Q3. Boltzmann counterion residual

## Finding 1
SEVERITY: INFO

LOCATION: `Forward/bv_solver/boltzmann.py:360-386`; `Forward/bv_solver/forms_logc_muh.py:414-428`; `Forward/bv_solver/forms_logc_muh.py:652-660`; `Forward/bv_solver/forms_logc_muh.py:879-881`

TRIGGER: All configured counterions are bikerman and `add_boltzmann_counterion_residual(..., skip_bikerman=True)` is called.

EVIDENCE: `add_boltzmann_counterion_residual` skips entries with `steric_mode == "bikerman"` when `skip_bikerman=True` (`boltzmann.py:360-366`). The bikerman source is instead wired in the main form: `build_steric_boltzmann_expressions(...)` is called before assembly (`forms_logc_muh.py:414-428`), and Poisson adds `F_res -= z_scale_shared * charge_rhs * charge_density_total * w * dx` for `steric_boltz` (`forms_logc_muh.py:652-660`). The post-assembly call is explicitly for ideal-path entries only (`forms_logc_muh.py:879-881`).

## Finding 2
SEVERITY: INFO

LOCATION: `Forward/bv_solver/boltzmann.py:224-228`; `Forward/bv_solver/boltzmann.py:252-258`; `Forward/bv_solver/forms_logc_muh.py:652-660`

TRIGGER: Cathodic polarization with `phi < 0` and a Cs+ bikerman counterion (`z=+1`).

EVIDENCE: The counterion factor is `q_k = exp(-z_k * phi_clamped_k)` (`boltzmann.py:224-228`). For Cs+ with `z=+1` and `phi<0`, `q_k>1`, so `c_steric_expr = c_bulk*q*free_dyn/denom` increases (`boltzmann.py:252-254`). Its charge contribution is `z_const * c_steric_expr` (`boltzmann.py:255-258`) and Poisson subtracts `charge_rhs * charge_density_total * w` from the weak residual (`forms_logc_muh.py:652-660`), matching the stated weak form `eps*grad(phi)*grad(w) - rho*w = 0`. The sign gives positive charge accumulation for Cs+ under negative potential.

# Q4. Jacobian freshness

## Finding 1
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:463-505`; `Forward/bv_solver/forms_logc_muh.py:609-668`; `Forward/bv_solver/forms_logc_muh.py:786-789`; `Forward/bv_solver/forms_logc_muh.py:839-881`; `Forward/bv_solver/boltzmann.py:384-386`

TRIGGER: Building `J_form = fd.derivative(F_res, U)`.

EVIDENCE: The NP residual is assembled before the derivative (`forms_logc_muh.py:463-505`), BV boundary terms are added before the derivative (`forms_logc_muh.py:609-636`), Poisson and Stern terms are added before the derivative (`forms_logc_muh.py:641-668`), and cation-hydrolysis residual terms are added before the derivative when enabled (`forms_logc_muh.py:786-789`). `J_form` is first derived at `forms_logc_muh.py:839`. The only residual mutator after `ctx.update` is `add_boltzmann_counterion_residual(...)` (`forms_logc_muh.py:879-881`), and that function re-derives `ctx["J_form"] = fd.derivative(F_res, U)` after its mutation (`boltzmann.py:384-386`). No stale Jacobian mutation was observed.

# Q5. Picard outer loop

## Finding 1
SEVERITY: INFO

LOCATION: `Forward/bv_solver/picard_ic.py:1016-1088`; `Forward/bv_solver/picard_ic.py:1108-1137`; `Forward/bv_solver/picard_ic.py:1416-1427`

TRIGGER: General N-reaction Picard solve.

EVIDENCE: The Picard loop assembles an `N x N` matrix `M` and vector `b` from all reactions (`picard_ic.py:1016-1088`). `_solve_linear_system` has special paths for `N=1`, `N=2`, and `N>=3`; for `N=2` it returns the direct 2x2 determinant (`picard_ic.py:1108-1137`). The caller marks the `N=2` solve singular if `det` is non-finite or `abs(det) < 1e-300` (`picard_ic.py:1416-1427`).

## Finding 2
SEVERITY: LOW

LOCATION: `Forward/bv_solver/picard_ic.py:1091-1105`; `Forward/bv_solver/picard_ic.py:1416-1423`

TRIGGER: Nearly singular but badly scaled 2x2 Picard matrices.

EVIDENCE: `_solve_2x2` guards only `abs(det) < 1e-300` and does not compare the determinant to the scale or condition number of `M` (`picard_ic.py:1091-1105`). The outer loop repeats the same absolute determinant test for `N=2` (`picard_ic.py:1416-1423`). A matrix with large entries and a determinant well above `1e-300` can still be numerically ill-conditioned without tripping this guard.

## Finding 3
SEVERITY: INFO

LOCATION: `Forward/bv_solver/picard_ic.py:1270-1272`; `Forward/bv_solver/picard_ic.py:1444-1445`; `Forward/bv_solver/picard_ic.py:1505-1527`; `Forward/bv_solver/forms_logc_muh.py:1267-1285`

TRIGGER: Picard convergence and damping.

EVIDENCE: `picard_outer_loop_general` defaults to `omega=0.5`, `max_iters=50`, and `tol=1e-6` (`picard_ic.py:1270-1272`). The call from the muh IC does not override `omega`, so the default applies (`forms_logc_muh.py:1267-1285`). The update is fixed relaxation `R = (1 - omega)*R_old + omega*R_solve` (`picard_ic.py:1444-1445`). The convergence metric is `sum(abs(R[j] - R_old[j]) / max(abs(R[j]), 1e-30))`, using the relaxed new rate in the denominator, not `abs(R_old)` (`picard_ic.py:1505-1507`). Non-convergence exits with `picard_max_iters_delta=...` (`picard_ic.py:1523-1527`). No adaptive omega logic appears in the function body.

## Finding 4
SEVERITY: MEDIUM

LOCATION: `Forward/bv_solver/forms_logc_muh.py:1028-1039`; `Forward/bv_solver/forms_logc_muh.py:1078-1083`; `Forward/bv_solver/forms_logc_muh.py:1155-1170`; `Forward/bv_solver/forms_logc_muh.py:1288-1289`; `Forward/bv_solver/picard_ic.py:1324-1339`; `Forward/bv_solver/picard_ic.py:1423-1476`; `Forward/bv_solver/picard_ic.py:1523-1527`

TRIGGER: Debye-Boltzmann IC failure.

EVIDENCE: The outer IC wrapper falls back to `set_initial_conditions_logc_muh` for any returned `ok=False` from `_try_debye_boltzmann_ic_muh` (`forms_logc_muh.py:1028-1039`). Returned failure reasons include insufficient species or empty reactions (`forms_logc_muh.py:1078-1083`), missing counterion or unsupported proton index (`forms_logc_muh.py:1155-1170`), a failed Picard return (`forms_logc_muh.py:1288-1289`), arity/index/reaction validation failures (`picard_ic.py:1324-1339`), singular/non-finite states (`picard_ic.py:1423-1476`), and max iterations (`picard_ic.py:1523-1527`). There is no oscillation-specific detector; fallback is broader than oscillation. Exceptions raised inside helper calls are not converted into `ok=False` by this wrapper, so those can bypass fallback.

# Q6. Stern split

## Finding 1
SEVERITY: LOW

LOCATION: `Forward/bv_solver/picard_ic.py:238-270`

TRIGGER: `solve_stern_split` cannot bracket a root because `f_lo * f_hi > 0`.

EVIDENCE: The bisection variable is `psi_D_signed`; the code sets `psi_S = full_drop - psi_D_signed` and residual `stern_coeff_nondim * psi_S - eps_nondim * slope_signed` (`picard_ic.py:238-251`). It brackets with `lo=0.0`, `hi=full_drop` (`picard_ic.py:253-258`). If `f_lo * f_hi > 0.0`, it does not fail; it returns a linear-Debye analytic fallback `psi_D_lin = stern_coeff*full_drop*lambda_D / (eps + stern_coeff*lambda_D)` (`picard_ic.py:259-270`). The fallback has no explicit check that `|psi_D|` is small, so a non-small bracket failure would silently use the linear approximation.

## Finding 2
SEVERITY: INFO

LOCATION: `Forward/bv_solver/picard_ic.py:142-178`; `Forward/bv_solver/picard_ic.py:246-287`; `Forward/bv_solver/forms_logc_muh.py:666-668`

TRIGGER: Stern-enabled IC and residual assembly.

EVIDENCE: `compute_surface_slope_signed` gives the surface slope the sign of `psi_D` (`picard_ic.py:142-178`). `solve_stern_split` enforces `stern_coeff * psi_S = eps * surface_slope_signed(psi_D)` with `psi_S = full_drop - psi_D` and returns `phi_surface = phi_applied_model - psi_S` (`picard_ic.py:246-287`). The weak residual adds the Stern Robin term as `F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)` (`forms_logc_muh.py:666-668`), which corresponds to the boundary condition `eps*partial_n(phi) = C_S*(phi_applied - phi_OHP)`. The bisection variable and sign convention are consistent.

# Q7. Sign conventions across the form

## Finding 1
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:641-660`; `Forward/bv_solver/forms_logc_muh.py:666-668`

TRIGGER: Poisson residual assembly.

EVIDENCE: The residual uses `F_res += eps_coeff * dot(grad(phi), grad(w)) * dx` and subtracts dynamic, water, and boltzmann charge sources (`forms_logc_muh.py:641-660`). With integration by parts this is the weak form of `-eps*Delta(phi) = charge_rhs*rho_charge`. The Stern term subtracts `C_S*(phi_applied - phi)*w` on the electrode (`forms_logc_muh.py:666-668`), giving the natural boundary equation `eps*partial_n(phi) = C_S*(phi_applied - phi_OHP)`. The Poisson and Stern signs are consistent.

## Finding 2
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:455-505`

TRIGGER: Nernst-Planck residual assembly for log-c and mu_H species.

EVIDENCE: The code comments state the physical fluxes as `J_i = -D_i*c_i*(grad(u_i)+em*z_i*grad(phi))` for non-mu species and `J_H = -D_H*c_H*grad(mu_H)` for the mu species (`forms_logc_muh.py:455-462`). The residual constructs the positive mobility-gradient quantity `Jflux = D*c*(...)` (`forms_logc_muh.py:470-501`) and adds `dot(Jflux, grad(v))` plus the time term (`forms_logc_muh.py:503-505`). This is consistent with weak form integration for physical flux `J_phys = -Jflux`.

## Finding 3
SEVERITY: INFO

LOCATION: `Forward/bv_solver/forms_logc_muh.py:540-615`; `Forward/bv_solver/forms_logc_muh.py:631-636`

TRIGGER: BV reaction boundary source with positive cathodic rate.

EVIDENCE: Per-reaction eta is selected before rate construction (`forms_logc_muh.py:540-546`). The cathodic rate uses `exp(-alpha*n_e*eta_j)` and the anodic branch uses `exp((1-alpha)*n_e*eta_j)` (`forms_logc_muh.py:548-607`), so negative cathodic overpotential increases the cathodic term. The net rate is `R_j = cathodic - anodic` (`forms_logc_muh.py:609`). Boundary residual terms are `F_res -= stoich_i * R_j * v_i * ds` (`forms_logc_muh.py:612-615`), and the legacy path uses the same sign pattern (`forms_logc_muh.py:631-636`). For negative stoichiometry reactants this adds a consumption flux; for positive stoichiometry products it adds a production flux into the electrolyte.

# Q8. Other numerical pathologies and edge cases

## Finding 1
SEVERITY: MEDIUM

LOCATION: `Forward/bv_solver/forms_logc_muh.py:326-340`

TRIGGER: Reconstructed log concentration outside `[-u_clamp, +u_clamp]`, with default `u_clamp=30`.

EVIDENCE: Concentrations are reconstructed as `exp(min(max(u_expr, -u_clamp), +u_clamp))` for both current and previous states (`forms_logc_muh.py:326-340`). This protects against overflow, but when the clamp is active the concentration derivative with respect to the underlying unknown is zero. In high-polarization Debye layers this can flatten NP, Poisson, and BV concentration sensitivity for the clipped species.

## Finding 2
SEVERITY: LOW

LOCATION: `Forward/bv_solver/boltzmann.py:360-389`

TRIGGER: Caller relies on the return value of `add_boltzmann_counterion_residual` while passing `skip_bikerman=True`.

EVIDENCE: The function skips bikerman entries in the loop (`boltzmann.py:360-366`) but returns `len(counterions)` rather than the number actually added (`boltzmann.py:384-389`). With all-bikerman counterions this reports a positive count even though the residual path did not add any ideal counterion term. This is not a form-sign bug because bikerman Poisson sources are added elsewhere, but it is a diagnostics/API edge case.

## Finding 3
SEVERITY: LOW

LOCATION: `Forward/bv_solver/picard_ic.py:171-178`; `Forward/bv_solver/picard_ic.py:238-270`

TRIGGER: Extremely large Stern diffuse drop passed to `solve_stern_split`.

EVIDENCE: The BKSA slope path evaluates `math.cosh(abs(psi_D))` directly (`picard_ic.py:171-178`). `solve_stern_split` can evaluate residuals at `psi_D = full_drop` before any bisection or fallback (`picard_ic.py:238-258`). For very large `|full_drop|`, this can overflow in Python before the linear-Debye fallback branch is reached.

# VERDICT

Overall: the audited PNP+BV ORR form assembly, eta clipping order, shared-theta Bikerman closure, counterion Poisson sign, Stern sign, and Jacobian refresh logic are broadly correct in the targeted code. The solver still has numerical pathologies: hard floors/clamps create zero-derivative regions, Picard singularity detection uses an absolute determinant threshold without conditioning, Stern fallback can silently linearize non-small bracket failures, and some helper exceptions can bypass IC fallback. These are pathologies and edge-case risks, not observed core sign errors in the main residual.
