# SONNET_03 — Initial Conditions Verification
## Scope: `debye_boltzmann` initializer, `picard_ic.py` + `forms_logc_muh.py`
## Date: 2026-05-22

---

## Bottom Line

The `debye_boltzmann` initializer (`set_initial_conditions_debye_boltzmann_logc_muh`) is **structurally correct** for composite-ψ construction, μ_H assignment, O₂/H₂O₂ seeding, Stern split, and IC↔residual consistency under the standard `steric_mode='bikerman'` branch. One resolved discrepancy (`A_DEFAULT` → physical `a_nondim` for O₂/H₂O₂/H⁺) is present as **uncommitted working-tree changes** in `scripts/_bv_common.py` (2026-05-21), visible in `git diff`. Until that diff is committed and propagates to calling scripts, the **current installed production run still uses `A_DEFAULT = 0.01` for H⁺ in the IC** if any calling script imports the old preset. One open concern is identified around the linear-Debye fallback in `solve_stern_split` failing to bracket at small `psi_D`: it falls to the analytical limit correctly, but this path is unlogged.

---

## 1. Composite-ψ Construction (Check 1)

**Verified CORRECT.**

The IC constructs ψ(y) in three branches inside `_try_debye_boltzmann_ic_muh`:

1. **Gouy-Chapman (GC) baseline** (`psi_gc`): always built as the tanh-angle form
   `psi_gc = 2*ln[(1 + T·exp(-y/λ_D)) / (1 - T·exp(-y/λ_D))]` with
   `T = tanh(psi_D/4)` and ε-clamped tanh to avoid log(0). Satisfies ψ(y→∞) = 0 analytically.

2. **BKSA matched-asymptotic (Bikerman, single-counterion)**: selects between
   - Debye-linear: `ψ(y) = psi_D · exp(-y/λ_D)` when `|psi_D| < psi_sat · (1-1e-3)`;
   - Saturated-zone: piecewise `psi_zone1` (linear in y near OHP) joined to
     `psi_zone2` (exponential tail) at `y_match = (|psi_D| - psi_sat)/α_d`.
   Both zones match sign conventions correctly.

3. **Multi-ion mode**: `psi(y) = psi_D_local · exp(-y/λ_eff)` using the effective
   Debye length from `multi_ion.effective_debye_length_local`. Shared-θ closure.

**y-coordinate normalization**: `y_norm = y / domain_height_hat` is used for the
outer linear profile (O₂, H₂O₂, H⁺ outer→bulk interpolation), but the Debye-layer
exponentials `exp(-y/λ_D)` correctly use the raw y coordinate, not y_norm. This is
consistent since λ_D shares nondim units with y. Verified explicitly at line 1314.

**ψ(y=0)**: By construction at y=0, `exp(-0/λ_D) = 1`, so `psi(0) = psi_D` (diffuse
drop). The `phi_init_expr = ln(H_outer/c_clo4_bulk) + psi` at y=0 gives
`phi_surface = phi_applied - psi_S`, consistent with the Stern split.

**ψ(y→∞)**: All three branch forms → 0. Verified analytically.

---

## 2. Multispecies-γ in the IC (Check 2)

**Verified CORRECT for counterions; see Known Issue for dynamic species.**

### Single Bikerman counterion path (`apply_bikerman_ic = True`)
```python
gamma_psi = 1 / (1 + a_h * H_outer * (exp(-psi) - 1)
                   + a_cl * c_cl_anchor * (exp(+psi) - 1))
```
This is the standard Bikerman activity coefficient with shared-θ from the matched
asymptotic: `γ(y) = θ(y) / θ_outer`. The formula matches the scalar Picard loop's
`compute_surface_gamma` with OHP-evaluated `psi_D` → spatial `psi(y)`. VERIFIED.

### Multi-ion path (Cs⁺/SO₄²⁻)
`gamma_psi = theta_y / theta_outer_const` where `theta_y` and `theta_outer_const`
are built from `build_counterion_ctx` with `a_dyn` fed from `solver_params[6][:n]`.
The multi-ion shared-θ closure is consistent with `build_steric_boltzmann_expressions`
in the residual builder. VERIFIED.

### Species treated with Bikerman vs ideal in IC
- Counterions with `steric_mode='bikerman'` (Cs⁺, SO₄²⁻, K⁺, ClO₄⁻-steric): **Bikerman** γ. CORRECT.
- Counterions with `steric_mode='ideal'` (default): γ=1. CORRECT (GC path, psi_gc used).
- Dynamic species (O₂, H₂O₂, H⁺): γ appears via `log_gamma` additive shift in `u_i_init`.

---

## 3. μ_H Initial Profile (Check 3)

**Verified CORRECT — analytic cancellation works.**

The IC explicitly computes:
```python
u_h_init_expr  = ln(H_outer) - psi + log_gamma      # u_H = log(c_H) at OHP
mu_h_init_expr = u_h_init_expr + em * z_h_val * phi_init_expr
```
With `em * z_h_val = 1` (production), `phi_init_expr = ln(H_outer/c_clo4_bulk) + psi`, so:
```
mu_H = ln(H_outer) - psi + log_gamma + ln(H_outer/c_clo4_bulk) + psi
     = 2*ln(H_outer) - ln(c_clo4_bulk) + log_gamma
```
The `psi` terms cancel exactly. In the ideal case (`log_gamma = 0`), `mu_H` is y-independent,
confirming Boltzmann equilibrium (H⁺ pile-up is all encoded in the residual's `exp(mu_H - phi)`
reconstruction, not in the IC primary variable). CORRECT and matches module docstring
comment at lines 1067–1068.

The multi-ion branch stores `mu_h_init = u_i_init + em*z_h * phi_init_expr` (lines 1431–1434)
with the same algebraic cancellation. CORRECT.

**Residual consistency**: the residual reconstructs `c_H = exp(mu_H - em*z_H*phi)`;
the IC seeds `mu_H` such that `exp(mu_H_init - em*z_H*phi_init) = H_outer * exp(-z_H * psi)
* gamma_psi` — a smooth Boltzmann profile. CORRECT.

---

## 4. O₂ / H₂O₂ Initial Profiles (Check 4)

**Verified CORRECT — no electromigration contamination.**

O₂ and H₂O₂ have `z = 0` (neutral). In the Bikerman path:
```python
U_prev.sub(0).interpolate(ln(O_outer) + log_gamma)
U_prev.sub(1).interpolate(ln(P_outer) + log_gamma)
```
`O_outer` and `P_outer` are linear interpolations between their surface (`O_s`, `P_s` from Picard)
and bulk values: `O_outer = O_s + (O_b - O_s) * y_norm`. No `psi` term, no electromigration.
`log_gamma` is the activity correction from excluded volume, not from charge.
In the ideal path, `log_gamma = 0` and the profiles are purely linear. CORRECT.

---

## 5. Bikerman Parameters Used in IC (Check 5)

**KNOWN ISSUE — partially resolved in uncommitted diff.**

### Current committed code (HEAD)
`THREE_SPECIES_LOGC_BOLTZMANN` was defined with `a_vals_hat=[A_DEFAULT] * 3` (all three
dynamic species at `A_DEFAULT = 0.01 ≈ 14.9 Å` hard-sphere). This is the Hard Rule #7
discrepancy: H⁺ cap clamped ~150× tighter than physical r=2.8 Å H₃O⁺ Stokes radius.

### Working-tree state (uncommitted, `scripts/_bv_common.py`)
The 2026-05-21 diff in `git diff scripts/_bv_common.py` replaces:
- `THREE_SPECIES_LOGC_BOLTZMANN`: `a_vals_hat = [A_O2_HAT, A_H2O2_HAT, A_HP_HAT]`
- `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`: `a_vals_hat = [A_O2_HAT, A_H2O2_HAT, A_HP_HAT, A_KPLUS_HAT]`
- `FOUR_SPECIES_LOGC_DYNAMIC` (ClO₄ equivalence test): `[A_O2_HAT, A_H2O2_HAT, A_HP_HAT, A_DEFAULT]`

Physical values: `A_O2_HAT ≈ 1.487e-5`, `A_H2O2_HAT ≈ 2.422e-5`, `A_HP_HAT ≈ 6.645e-5`.

**Status**: This change is **NOT yet committed**. The `A_HP_HAT` is ≈4.4× larger than
`A_DEFAULT` (not 150× — note: c_max ∝ 1/a, so the ~150× claim in CLAUDE.md refers to
the cap on molar concentration, not on `a` itself). The H⁺ Bikerman cap changes from
`1/A_DEFAULT = 100 nondim ≈ 120 mol/m³` to `1/A_HP_HAT ≈ 15040 nondim ≈ 18050 mol/m³`.

**IC impact**: `a_h_picard = float(a_vals_full_for_picard[2])` reads from `solver_params[6]`
(which comes from `species.a_vals_hat`). Before commit, `a_h_picard = A_DEFAULT` for the
Picard surface gamma; after commit, `a_h_picard = A_HP_HAT`. The IC's steric γ formula
changes accordingly.

**Counterion entries** (`DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`, `DEFAULT_KPLUS_BOLTZMANN_COUNTERION_STERIC`, `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC`) already use physical Linsey-deck
radii (`A_CSPLUS_HAT`, `A_SO4_HAT`, `A_KPLUS_HAT`) and are **unaffected** by the diff.

`DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC` retains `A_DEFAULT` as `a_nondim` — this is
intentional (not yet re-anchored) and documented.

---

## 6. IC↔Residual Consistency (Check 6)

**Verified CORRECT for the `steric_mode='bikerman'` branch.**

### Branch selection
IC branch: `apply_bikerman_ic = synthesised_4sp_counterion or bikerman_in_counterions`
where `bikerman_in_counterions = any(e.get("steric_mode", "ideal") == "bikerman" for e in counterions)`.

Residual branch: `build_steric_boltzmann_expressions` filters
`bikerman = [(j, e) for j, e in enumerate(counterions) if e.get("steric_mode", "ideal") == "bikerman"]`.

Both use identical `steric_mode` key-check logic. CONSISTENT.

### Fallback path (`steric_mode='ideal'`)
When no counterion has `steric_mode='bikerman'`, `apply_bikerman_ic = False` and the IC
uses `psi_gc` (Gouy-Chapman) with `log_gamma = 0`. The residual's
`add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)` builds ideal-Boltzmann
counterions (unbounded `c_k = c_bulk * exp(-z_k * phi)`). IC and residual both see no
steric packing. CONSISTENT.

### Hard Rule #3 compliance
The IC's Bikerman γ formula matches the residual's `build_steric_boltzmann_expressions`
shared-θ closure:
- IC: `gamma_psi = 1 / (1 + a_h * H * (exp(-psi)-1) + a_cl * c_cl * (exp(psi)-1))`
- Residual: `c_k(phi) = c_b_k * exp(-z_k*phi) * (1 - A_dyn(phi)) / (theta_b + sum_k a_k*c_b_k*exp(-z_k*phi))`

These are equivalent in the single-counterion 1:1 electrolyte limit. The IC γ is evaluated
at the OHP potential (via `psi_D` from the Picard loop, then extended spatially), while the
residual evaluates it at the FE-resolved `phi`. STRUCTURALLY CONSISTENT.

### Double-count guard
`add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)` at the end of
`build_forms_logc_muh` (line 881) skips bikerman entries; they were already handled
by `build_steric_boltzmann_expressions` at line 418 in `forms_logc.py`. NO double-count.

---

## 7. Anchor IC vs Grid IC (Check 7)

**Verified CORRECT.**

From `anchor_continuation.py`: `solve_anchor_with_continuation` calls the IC setter once
at the anchor voltage; the grid sweep via `solve_grid_with_anchor` copies the converged
anchor solution (`extract_preconverged_anchor`) as the warm start for all grid points.
The `debye_boltzmann` IC is only invoked at the anchor — subsequent grid points inherit
the warm-walk chain, not a new Picard run. This is correct and avoids the Picard
oscillation problem that occurs at cathodic voltages (see `ic_refinement_study.md`).

---

## Discrepancies

### D1 — Uncommitted `A_DEFAULT` → Physical Radii (HIGH PRIORITY, known, tracked)
`scripts/_bv_common.py` has an uncommitted diff (2026-05-21) replacing `A_DEFAULT` with
physical `A_O2_HAT`, `A_H2O2_HAT`, `A_HP_HAT` in three presets. Until committed, any
driver script importing these presets will use the wrong (oversized) `a_nondim` for H⁺
in the IC Picard gamma and in the residual Bikerman packing. Hard Rule #7 explicitly flags
this. The change is staged in working tree but not propagated to a commit.

**Risk**: IC gamma_s at the OHP uses `a_h_picard = A_DEFAULT` (pre-commit), producing a
tighter steric cap on H⁺ pile-up. The bridge runs referenced in CLAUDE.md §7 (`_phase_D_bridge_corrected_a*.py`) are the intended test vehicles.

### D2 — Stern-split fallback path unlogged (LOW, latent)
`solve_stern_split` uses a linear-Debye analytical fallback when the bisection bracket
fails to straddle zero (lines 260–270). This occurs at very small `|full_drop|` or
degenerate `stern_coeff`. The fallback is mathematically correct but does not emit any
log or set a ctx flag, so silent fallthrough is undetectable in production runs. Low risk
(production voltages produce non-zero `full_drop`), but worth a log statement.

---

## Open Questions

1. **Are the physical `a_nondim` changes in `_bv_common.py` covered by the existing
   snapshot tests?** The `test_stern_no_stern_snapshot.py` and
   `test_initializer_attractor_equivalence.py` tests likely have hardcoded tolerances
   calibrated to `A_DEFAULT`-era behavior. If these xfail after commit, the
   behavior change may be larger than the "negligible at hydrolysis-off" bridge-run
   finding suggests.

2. **The Picard `H_o` flux-balance formula** (line 767: `H_o = H_b - (R1+R2)/D_H`)
   carries a comment "ambipolar 2·D_H and −2 proton stoichiometry cancel." This
   cancelation is non-obvious for the parallel 4e topology (R4e stoichiometry = −4 H⁺).
   The comment references `PNP_BV_Analytical_Simplifications.md` lines 240–244; verify
   that the derivation there covers the parallel 2e/4e case explicitly, not just
   sequential.

3. **The `_solve_phi_o` multi-ion path** (line 326) calls `solve_outer_phi_multiion`
   with a `bracket=(phi_o_prev ± 5)` warm start. At the start of the Picard loop,
   `phi_o = 0` and `phi_o_prev = None`, so the first call always uses the global
   `(-50, +50)` bracket. This is fine but means the multi-ion Picard incurs one extra
   wide-bracket bisect at iteration 1. Not a correctness issue.

---

## Files Reviewed

- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/picard_ic.py` (full, 1647 lines)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/forms_logc_muh.py` (lines 889–1561)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/boltzmann.py` (lines 91–270)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py` (lines 155–854, plus git diff)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/solver/ic_refinement_study.md`
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/solver/stern_layer_physics_and_next_steps.md`
