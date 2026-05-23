# SONNET_10 — Water Self-Ionization Verification

**Scope**: `enable_water_ionization` Phase 6α opt-in layer.  
**Files reviewed**: `Forward/bv_solver/water_ionization.py` (338 lines), `scripts/_bv_common.py` (water-ionization block), `Forward/bv_solver/forms_logc_muh.py` (water-ion integration), `Forward/bv_solver/anchor_continuation.py` (kw_eff_ladder).  
**Date**: 2026-05-22  

---

## 1. K_w Value

**PASS — correctly scaled.**

- `KW_MOLAR_SQUARED = 1e-14` (mol/L)² — textbook 25°C value.  
- `KW_PHYS = 1e-14 × 1000² = 1e-8` (mol/m³)² — correct unit conversion.  
- `KW_HAT = KW_PHYS / C_SCALE²` where `C_SCALE = C_O2 = 1.2 mol/m³`; numerically `KW_HAT ≈ 6.944e-9` (nondim).  
- `_bv_common.py` includes a self-consistency check: at pH 4 (`C_HP_HAT ≈ 0.0833`), `C_OH_BULK_HAT = KW_HAT / C_HP_HAT ≈ 8.33e-8`, which back-converts to `c_OH_phys = 1e-7 mol/m³` (= 1e-10 mol/L) — consistent with `Kw` at pH 4.  Comment confirms this explicitly.

---

## 2. Rate Law / Closure Form

**PASS — correct fast-equilibrium closure; no explicit r_water rate law.**

The implementation does **not** use a finite-rate `r_water = k_w_eff · (K_w_eff − c_H · c_OH)` source term. Instead it uses the **fast-equilibrium (Da ≫ 1) reduction**: the OH⁻ concentration is a pointwise algebraic function of c_H via the closure

```
c_OH = K_w_eff · exp(−u_H)
```

This is the correct limit when the water equilibration timescale (71 ns–7 ps) is ≪ diffusion timescale (~1 s at L_ref=100 µm). The H⁺ NP equation is **replaced** by the proton-condition residual `∂E/∂t + ∇·J_E = 0` where `E = c_H − c_OH`. There is no separate H⁺ source term added; the sourcing occurs implicitly through the modified flux `J_E = J_H − J_OH`. Physically this is equivalent: divergence of `J_E` captures any spatial imbalance in the H⁺ ↔ OH⁻ exchange, relaxing the system toward `c_H · c_OH = K_w_eff` at every point.

Sign check: when `c_H · c_OH < K_w_eff`, `c_OH < K_w_eff/c_H` → OH⁻ flux draws c_OH up (H₂O dissociation increases) — correct direction. The closure automatically drives the system toward equilibrium.

---

## 3. Source Terms Applied to Species

**PASS — correct topology; both H⁺ and OH⁻ handled consistently.**

OH⁻ is NOT a separate dynamic NP species (no extra DOF added to the FE space). The H⁺ residual equation is **replaced** — not augmented — by the proton-condition residual on `E = c_H − c_OH`. The effect is equivalent to simultaneously sourcing both H⁺ and OH⁻ at the same rate, since `J_E = J_H − J_OH` and the time term is `∂(c_H − c_OH)/∂t`.

The flux `J_E` is built in `build_proton_condition_flux` (`water_ionization.py:317`):

```python
flux = (D_h * c_h + d_oh * c_oh) * ideal_grad_h
```

This matches the derivation: both H⁺ and OH⁻ carry current along `∇μ_H_ideal` (OH⁻'s electrochemical gradient equals the negative of H⁺'s), so their combined contribution to the proton-condition flux adds the diffusion coefficients weighted by their respective concentrations. The steric correction (`D_h*c_h - d_oh*c_oh` × ∇μ_steric) is included when `steric_active=True`.

---

## 4. OH⁻ Topology in the Stack

**PASS — OH⁻ is a "shadow species" (analytic closure), NOT a dynamic DOF.**

When `enable_water_ionization=True`:
- OH⁻ does **not** add a new FE unknown to the function space.  
- `c_oh_expr` is a UFL symbolic expression `kw_eff_func * exp(-u_H_clamped)`, built at form-construction time.  
- OH⁻ enters **Poisson** via `F_res -= charge_rhs * (−1.0) * c_oh_expr * w * dx` (`forms_logc_muh.py:648-651`). This is correct: z = −1 for OH⁻, same charge_rhs prefactor as dynamic species.  
- OH⁻ enters **Bikerman packing** via `A_dyn = A_dyn + a_oh_const * c_oh_expr` when `a_oh_const != 0` (`forms_logc_muh.py:438-440`).  
- No mesh/DOF count change → the disabled and enabled paths share the same FE topology. This is consistent with the "no extra NP equation for OH⁻" design choice.

The Poisson and steric wiring are both correct and gated properly.

---

## 5. `kw_eff_ladder` Status

**PARTIALLY IMPLEMENTED — individually operational; combination with other ladders raises `NotImplementedError`.**

Current status (updated from memory note `project_v10a_prime_two_stage_anchor.md` which recorded `NotImplementedError` for `kw_eff_ladder` as of v10a'):

- `kw_eff_ladder` is now **fully implemented** in `solve_anchor_with_continuation` (`anchor_continuation.py:1500-1606`). The ladder validates strict monotone ordering, handles a 0.0 floor rung (runs full k0 ladder at Kw_eff=0 first, then ramps Kw_eff), and uses `AdaptiveLadder` with rollback on failure.  
- `set_reaction_kw_eff_model` writes consistently to both the R-space Function (residual source of truth) and `ctx['nondim']['bv_convergence']['kw_eff_hat']` (Picard/diagnostics).  
- **Restriction still in effect**: `kw_eff_ladder + c_s_ladder` → `NotImplementedError` (`anchor_continuation.py:1048-1051`). `kw_eff_ladder + lambda_hydrolysis_ladder` → `NotImplementedError` (`anchor_continuation.py:1055-1063`). Each ladder must be used independently.

---

## 6. Default-Off Byte Equivalence

**PASS — confirmed by multiple layers.**

- `is_water_ionization_enabled` (`water_ionization.py:323`) returns `False` when key is absent or falsy — safe default.  
- `forms_logc_muh.py` gates all water-ionization code paths at `water_ion_enabled = is_water_ionization_enabled(conv_cfg)` (`line 348`); `water_bundle = None` when disabled.  
- Poisson source for OH⁻ is inside `if water_ion_enabled:` guard (`line 647`).  
- Bikerman packing addition is gated `if water_ion_enabled and float(water_bundle.a_oh_const) != 0.0:` (`line 438`).  
- The H⁺ residual replacement is gated `if water_ion_enabled and i == h_idx_water:` (`line 479`); when False, the standard `D*c*ideal_grad` flux is built unchanged.  
- Phase 6α summary confirms: "byte-equivalent to baseline when disabled. 549 fast tests + 10 slow tests pass."  
- In `water_ionization.py:308`: at `Kw_eff=0`, `c_oh_expr=0`, so `J_E` reduces algebraically to `D_h * c_h * ideal_grad_h` — the standard H⁺ NP flux. This provides a continuous bridge confirming byte-equivalence at the kw_eff=0 floor.

---

## 7. K_w_eff vs K_w — Field Dependence

**PASS — `kw_eff` is a continuation ramp parameter only; NOT field-dependent.**

`kw_eff_func` is a `firedrake.Function(R_space, ...)` — an R-space (globally constant) function. It is a single scalar value across the entire domain, updated between Newton solves by `set_reaction_kw_eff_model`. It is NOT spatially varying and has no dependence on the local electric potential `φ` or concentration fields.

This is distinct from the **cation hydrolysis** pKa shift (agent 09 scope), which uses Singh 2016 field-dependent pKa. The water-ionization Kw_eff is purely a continuation device: ramping from 0 → `KW_HAT` to keep Newton convergent as the OH⁻ coupling is switched on. At the final rung, `kw_eff = KW_HAT ≈ 6.94e-9` (the physical value, nondim) with no spatial variation.

If field-dependent Kw_eff were desired in the future (e.g., dielectric saturation suppressing K_w near the OHP), this would require replacing `kw_eff_func` with a spatially resolved expression — a non-trivial topology change, not a parameter adjustment.

---

## Findings Summary

| Item | Status | Notes |
|------|--------|-------|
| K_w value & scaling | PASS | 1e-14 mol²/L² → 1e-8 mol²/m⁶ → KW_HAT ≈ 6.94e-9 (nondim), self-consistent at pH 4 |
| Rate law (closure form) | PASS | Fast-equilibrium `c_OH = Kw_eff/c_H` via proton-condition residual; sign drives to equilibrium |
| Source terms topology | PASS | H⁺ eq replaced (not augmented); OH⁻ enters as shadow species, both sides accounted |
| OH⁻ species topology | PASS | Analytic closure (no extra DOF); correctly wired to Poisson (z=−1) + Bikerman packing |
| kw_eff_ladder | PARTIAL | Individually implemented; combining with c_s_ladder or lambda_ladder raises NotImplementedError |
| Default-off byte equivalence | PASS | Multi-layer gating confirmed; kw_eff=0 algebraically recovers standard NP flux |
| Field dependence of kw_eff | PASS | Global scalar ramp only; NOT field-dependent (distinct from Singh hydrolysis pKa) |

**No blocking bugs found.** One open architectural note: the ladder combination restrictions (`kw_eff + c_s`, `kw_eff + lambda`) are `NotImplementedError` by design (Gate 2 MVP constraint), not bugs. Any future study requiring simultaneous ramps must lift these guards.

---

## Concerns / Recommendations

1. **OH⁻ Bikerman size parameter `a_oh_hat`**: the `a_oh_const` enters the packing fraction `A_dyn`. `_bv_common.py` derives `A_OH_HAT` from Marcus radius r_OH = 1.76 Å — this is physically grounded (unlike dynamic species which use `A_DEFAULT = 0.01`, per Hard Rule #7). However, callers must pass `a_oh_hat=A_OH_HAT` explicitly; if `a_oh_hat=0` is passed (default behavior when not set), OH⁻ is excluded from packing — a silently reduced steric model. Confirm calling scripts pass non-zero `a_oh_hat` when steric mode is active.

2. **`kw_eff_initial = 0.0` is allowed** at form build time (`water_ionization.py:197-202`); the guard only rejects negative values. This means a context built with `kw_eff_hat=0` will have `c_oh_expr=0` initially — mathematically correct (OH⁻ absent until ladder ramps up), but callers must ensure `kw_eff_ladder` is used to ramp Kw_eff to the physical target. If `kw_eff_hat=0` and no ladder is provided, water ionization is silently no-op despite `enable_water_ionization=True`.
