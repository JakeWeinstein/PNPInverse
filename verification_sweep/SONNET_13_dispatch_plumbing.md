# SONNET_13 — Dispatch + Plumbing Verification

**Scope:** `Forward/bv_solver/__init__.py`, `config.py`, `dispatch.py`,
`forms_indexing.py`, `mesh.py`, `sweep_order.py`

**Date:** 2026-05-22

---

## 1. Dispatch routing (dispatch.py)

**PASS — with one stale-comment note.**

`_resolve_backend` maps:

- `formulation="logc_muh"` → `build_context_logc_muh` / `build_forms_logc_muh` / `set_initial_conditions_*_logc_muh`
- anything else → `build_context_logc` / `build_forms_logc` / `set_initial_conditions_*_logc`

The `formulation="concentration"` case never reaches `dispatch.py` as its own branch: `_resolve_backend` returns `"logc"` for it (defensive fall-through). The concentration backend file `forms.py` does not exist — confirmed deleted.

**Stale comment (low severity):** `dispatch.py` module docstring and `__all__` comment still call `logc_muh` "experimental" and `logc` "production". Per CLAUDE.md, `logc_muh` is the production target as of Phase 6. No functional impact; documentation only.

The `set_initial_conditions` dispatcher correctly two-keys on both `formulation` (backend) and `initializer` (`"linear_phi"` vs `"debye_boltzmann"`). The `blob=True` kwarg is accepted and silently ignored with a clear docstring explanation.

---

## 2. config.py defaults

**PASS.**

`_default_bv_convergence_cfg()` (the no-params fallback) returns:

| Key | Value | Hard rule |
|---|---|---|
| `exponent_clip` | `100.0` | Hard rule #2 ✓ |
| `u_clamp` | `100.0` | Hard rule #2 ✓ |
| `formulation` | `"logc"` | OK (production is `logc_muh` but this is the parser default) |
| `bv_log_rate` | `False` | Caller must set `True` explicitly |
| `apply_h_source` | `True` | v9 byte-equivalent ✓ |
| `apply_k_sink` | `True` | v9 byte-equivalent ✓ |
| `override_sigma_singh_counts_pm2` | `None` | v9 byte-equivalent ✓ |

`_get_bv_convergence_cfg` (the live parser) also defaults `exponent_clip=100.0` and `u_clamp=100.0` at parse-time (line 163, 166). Hard rule #2 is satisfied in both paths.

The ablation-flag cross-validation logic is correct: setting `apply_h_source=False` or `apply_k_sink=False` without `manufactured_R_inj` raises a `ValueError` at config-parse time (line 258–266). Setting both `override_sigma_singh_counts_pm2` and `manufactured_R_inj` also raises at parse time (lines 267–275). These are exactly the guards described in CLAUDE.md §Cross-validation.

**Minor concern:** `_VALID_FORMULATIONS` in config.py (line 78) still includes `"concentration"` as a valid string (triggers a `DeprecationWarning` but does not raise). This is deliberate — keeps old configs parseable. The warning text is correct and explicit.

---

## 3. forms_indexing.py — species/phi/Gamma DOF layout

**PASS.**

`unpack_dof_indices` is a single source of truth for the mixed-space layout. It returns a frozen dataclass with:

- `has_gamma=False`: `species_slice=slice(0,-1)`, `phi_index=-1`, `gamma_index=None` — matches legacy `fd.split(U)[:-1]` / `fd.split(U)[-1]` patterns byte-for-byte.
- `has_gamma=True`: `species_slice=slice(0,-2)`, `phi_index=-2`, `gamma_index=-1` — prevents the silent phi→Gamma mis-wire that motivated the module.

No Firedrake import: module is pure Python, testable without FE infrastructure. ✓

**Spot-check against forms_logc_muh.py:** `mu_h_idx` is resolved dynamically via `_resolve_mu_h_index(z_vals, roles=species_roles)`, falling back to z-inference when `species_roles` is not set. The comment at line 1158–1165 explicitly notes `mu_h_idx == 2` for the 3sp+Boltzmann production ordering (O₂=0, H₂O₂=1, H⁺=2) and raises if `mu_h_idx != 2` (line 1169). This is consistent with the species ordering implied everywhere in the codebase.

---

## 4. mesh.py — mesh generation

**PASS.**

`make_graded_interval_mesh(N=300, beta=2.0)`: 1D interval [0,1], power-law grading `x_i = (i/N)^beta`. `beta=2` clusters near electrode (x=0). Boundary markers: 1=electrode, 2=bulk. Correct.

`make_graded_rectangle_mesh(Nx=8, Ny=300, beta=2.0, domain_height_hat=1.0)`: 2D, y-direction graded by `y^beta * domain_height_hat`. Boundary markers: 3=bottom/electrode, 4=top/bulk (RectangleMesh convention). Production driver uses `Ny=200` or `Ny=300`. `_validate_domain_height_hat` enforces `[1e-3, 10.0]` sanity range, fails loudly otherwise.

**Note:** `domain_height_hat` sanity bounds (min 1e-3, max 10.0) are appropriate for L_REF=100µm (0.1µm to 1mm physical). The L_eff sweep is explicitly noted in the comment.

---

## 5. sweep_order.py — voltage ordering

**PASS — with a minor labeling note.**

`_build_sweep_order` produces a deterministic ordering:

- Single-sign case (all V ≥ 0 or all V ≤ 0): ascending `|eta|` — anchor-outward. ✓
- Mixed-sign case: splits into negative branch (ascending `|eta|`) and positive branch (ascending `eta`); the branch with the smaller minimum `|eta|` goes first.

This is NOT strictly "alternating + and − from a single anchor" — it is "one branch fully, then the other branch." The docstring documents this accurately as a "two-branch sweep." The effect is identical to alternating from the smallest |eta| point outward within each sign hemisphere, then switching hemispheres. The warm-start carry is continuous within each branch; the inter-branch transition carries the last-converged state of the first branch as the IC for the first point of the second branch.

`_apply_predictor` implements a quadratic Lagrange predictor (3-point) with fallback to linear (2-point) then simple warm-start. The `_MAX_PREDICTOR_RATIO = 10.0` guard prevents divergent extrapolation; validated by reverting to carry-state if any DOF deviates more than 10× from carry. Concentration clamp at ≥1e-10 applied post-prediction for all but the last component (phi). Correct.

---

## 6. __init__.py — public API exports

**PASS — with one missing export.**

Exported: `make_graded_interval_mesh`, `make_graded_rectangle_mesh`, `build_context`, `build_forms`, `set_initial_conditions`, all `*_logc` and `*_logc_muh` variants, `solve_grid_per_voltage_cold_with_warm_fallback`, `solve_grid_with_anchor`, `solve_anchor_with_continuation`, `extract_preconverged_anchor`, `AdaptiveLadder`, `PreconvergedAnchor`, `get/set_reaction_k0_model`, `get/set_reaction_kw_eff_model`, `get/set_stern_capacitance_model`, plus all supporting types.

**Missing export:** `validate_solution_state` is NOT exported from `Forward/bv_solver/__init__.py`. The function lives in `Forward/bv_solver/validation.py` but is not re-exported via `__init__.py`. CLAUDE.md documents it in the "Gotchas" section and callers (orchestrators) import it directly from `validation.py`. This is a documentation inconsistency — CLAUDE.md presents it as part of the public API but it is not in `__all__`. **Severity: low** (callers import it correctly from `validation.py`; not a runtime failure, but a completeness gap in the public surface).

**Stale doc in __init__.py module docstring (line 9):** The example shows `ClO4-` as the Boltzmann counterion. Deck baseline is K⁺/SO₄²⁻ per CLAUDE.md. This is documentation-only; no code is affected.

`make_bv_solver_params` lives in `scripts/_bv_common.py`, not in `Forward/bv_solver/`; it is not exported from `__init__.py`. This is by design (scripts layer, not library layer).

---

## 7. Step-6 ablation flags

**PASS.**

`apply_h_source=True`, `apply_k_sink=True`, `override_sigma_singh_counts_pm2=None` are the defaults in both `_default_bv_convergence_cfg()` and in `_get_bv_convergence_cfg()`. These preserve byte-equivalence with v9/v10a/v10a'/A.2. The cross-validation guards (ValueError on half-physical without manufactured injection; ValueError on conflicting overrides) are implemented at config-parse time. ✓

---

## 8. Legacy script risk: _make_bv_convergence_cfg called directly

**LOW-SEVERITY CONCERN.**

`_make_bv_convergence_cfg` in `scripts/_bv_common.py` defaults `formulation="concentration"` (line 456). This is the internal helper; `make_bv_solver_params` correctly passes `formulation=formulation` through to it (line 1292–1293).

However, ~15 legacy scripts (`v18_*`, `v19_*`, `v23_*`, `v25_*`, `plot_iv_curves_3sp_true.py`) import and call `_make_bv_convergence_cfg()` directly with no `formulation` argument. This will trigger the `DeprecationWarning` from `_validate_formulation` and then route to the `logc` backend. The `concentration` backend file is gone, so no silent wrong-backend run is possible. These scripts are documented as "legacy / non-operational" (CLAUDE.md) and are not used in Phase 6 production runs. The deprecation warning path in `config.py:_validate_formulation` is the correct safety net.

---

## Summary of findings

| # | Item | Severity | Status |
|---|---|---|---|
| 1 | `logc_muh` labeled "experimental" in dispatch.py comments | DOC | Low |
| 2 | `__init__.py` docstring example shows ClO4- counterion (stale) | DOC | Low |
| 3 | `validate_solution_state` not re-exported from `__init__.py` | API gap | Low |
| 4 | Legacy scripts call `_make_bv_convergence_cfg()` without `formulation` arg | Legacy | Low (docs say non-operational) |
| 5 | `exponent_clip=100`, `u_clamp=100` defaults ✓ | PASS | — |
| 6 | Step-6 ablation flag defaults (True/True/None) ✓ | PASS | — |
| 7 | `conc` backend removed, `forms.py` gone ✓ | PASS | — |
| 8 | `forms_indexing.py` single source of truth for phi/Gamma layout ✓ | PASS | — |
| 9 | Mesh grading correct, boundary markers consistent ✓ | PASS | — |
| 10 | Sweep ordering deterministic, well-documented ✓ | PASS | — |

**No blocking issues. All hard-rule-relevant defaults are correct.**
