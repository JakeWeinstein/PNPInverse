# Round 5 counterreply — Step 6 plumbing ablation plan

## Section 1 — Acknowledgments (per R4 issue)

### Issues 1–2 (A3 doesn't prove override reached actual R_net in F_res; "override-in-form" overstated)
**Accept both.**  Adding A3 gate 6 (residual-side check) and storing
the canonical R_net expression on ctx:

At form-build, add:
```python
ctx["_cation_hydrolysis_R_net_expr"] = R_net  # the actual UFL expr
ctx["_cation_hydrolysis_R_net_scalar_form"] = (
    lam_func * R_net * ds(electrode_marker)
)
ctx["_cation_hydrolysis_H_flux_scalar_form"] = (
    lam_func * R_net * ds(electrode_marker)
)  # same as R_net_scalar but stored under H name for symmetry
ctx["_cation_hydrolysis_K_flux_scalar_form"] = (
    lam_func * (-R_net) * ds(electrode_marker)
)
```

Both `_cation_hydrolysis_H_residual_term` (with `v_list[h_idx]`)
and `_cation_hydrolysis_K_residual_term` (with `v_list[k_idx]`)
that go into `F_res` MUST reference the same `R_net` object as
`_cation_hydrolysis_R_net_expr` so a split-brain bug is impossible
by construction (single source of truth at form-build time).

A3 gate 6 (NEW):
```python
flux_scalar_A3 = float(fd.assemble(
    ctx["_cation_hydrolysis_R_net_scalar_form"]
))
k_des = float(ctx["cation_hydrolysis"].k_des_func)
gamma = float(ctx["cation_hydrolysis"].gamma_func)
expected = k_des * gamma * electrode_area_nondim
assert (
    abs(flux_scalar_A3 - expected)
    / max(abs(expected), 1e-30)
    < 5e-3
)
```

This proves the override-aware R_net (used in `_R_net_scalar_form`,
the same R_net used in the residual) integrates to k_des·Γ·area at
SS.  If the residual used the OLD solved-σ pKa expression instead
of the override (split-brain bug), Γ_A3 would be Γ_predicted with
old σ (≈ Γ_A0 = 0.0405) while flux_scalar_A3 would reflect the
old R_net.  Either way, the closure fails.

**Reworded** for §A3 final pass criteria:
* Gates 1+2 prove **override-in-stored-pKa-expression + override-in-diagnostics**.
* Gate 3 proves **override-in-Picard**.
* Gate 4 (Newton+Picard converged) and gate 5 (positivity+finite) are
  hygiene.
* **Gate 6 (new) proves override-in-residual** via mass-balance
  closure at A3.

### Issue 3 (manufactured diagnostics missing top-level c_K_boundary_avg)
**Accept.**  `collect_v10a_rung_diagnostics` always emits top-level
`c_H_boundary_avg` and `c_K_boundary_avg` (both physical AND
manufactured paths), assembled directly:

```python
ds = fd.Measure("ds", domain=mesh)
area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
diag["c_H_boundary_avg"] = (
    float(fd.assemble(ci[bundle.h_idx] * ds(electrode_marker))) / area
)
diag["c_K_boundary_avg"] = (
    float(fd.assemble(ci[bundle.counterion_idx] * ds(electrode_marker))) / area
)
```

These two fields are populated BEFORE the manufactured-vs-physical
branch and live at the top level of diag.  Physical-path
`F0_decomposition.c_K_avg` remains for back-compat (it's the same
quantity but inside the F0 sub-dict; manufactured paths get None
there).

### Issue 4 ("all diagnostics finite" underspecified)
**Accept.**  Per-ablation required-key list:

```python
REQUIRED_NUMERIC_KEYS = {
    "A0": [
        "gamma", "theta", "sigma_S_C_per_m2",
        "c_H_boundary_avg", "c_K_boundary_avg",
        "R_2e_current_nondim", "R_4e_current_nondim",
        "cd_observable", "F0_avg", "denominator_total",
        "mass_balance_residual_rel",
    ],
    "A0b": [  # piggybacks on A0
        "phys_flux_scalar", "phys_h_flux_scalar",
        "phys_k_flux_scalar",
    ],
    "A1": [
        "gamma", "theta", "sigma_S_C_per_m2",
        "c_H_boundary_avg", "c_K_boundary_avg",
        "R_2e_current_nondim", "R_4e_current_nondim",
        "cd_observable", "manufactured_R_inj",
    ],
    "A2": [
        "gamma", "theta", "sigma_S_C_per_m2",
        "c_H_boundary_avg", "c_K_boundary_avg",
        "R_2e_current_nondim", "R_4e_current_nondim",
        "cd_observable", "manufactured_R_inj",
    ],
    "A3": [
        "gamma", "theta", "sigma_S_C_per_m2",
        "c_H_boundary_avg", "c_K_boundary_avg",
        "R_2e_current_nondim", "R_4e_current_nondim",
        "cd_observable", "F0_avg", "pka_factor_avg",
        "amp_from_singh", "phys_flux_scalar",
    ],
}

def classify_diagnostic_failure(record, ablation_id):
    for key in REQUIRED_NUMERIC_KEYS[ablation_id]:
        v = record.get(key)
        if v is None or not numpy.isfinite(v):
            return f"diagnostic_failure: {key} is {v!r}"
    return None
```

Non-numeric fields (strings, bools, dicts, lists) are not checked.

### Issue 5 (Cofunction.subfunctions API uncertain across Firedrake versions)
**Accept.**  Drop vector slot inspection from A0b.  Use **scalar
forms only**:

A0b (REVISED FINAL):
1. **Scalar source flux:** `flux_scalar = assemble(_R_net_scalar_form)`
   matches `k_des·Γ·area` within rel 5e-3.
2. **H+ flux scalar:** `h_flux = assemble(_H_flux_scalar_form)`
   matches `+flux_scalar` exactly (same form, same R_net object).
3. **K+ flux scalar:** `k_flux = assemble(_K_flux_scalar_form)`
   matches `-flux_scalar` within rel 1e-12 (FP).
4. **Anti-symmetry:** `|h_flux + k_flux| / max(|h_flux|, 1e-30) <
   1e-12` (FP precision).

Gates 2–4 are TAUTOLOGICAL by form construction (they read scalar
forms that share `R_net` with the canonical residual contributions).
**That's the point**: if these gates DON'T pass, the form-build
itself is broken (e.g., `R_net` not consistently used across the
stored artifacts).  This catches a development-time bug where the
form-build code drifts (e.g., someone refactors and inserts a
different R_net expression into one of the scalar forms).

Slot-wiring proof comes from a SEPARATE slow test, not from A0b:

```python
@pytest.mark.slow
def test_H_K_residual_terms_target_correct_slots(ctx_A0):
    """At A0 converged, inspect F_res's integrals_by_type to confirm
    the hydrolysis terms attach to v_h and v_k (not other slots)."""
    from ufl.algorithms.analysis import extract_arguments
    F_res = ctx_A0["F_res"]
    h_idx = ctx_A0["cation_hydrolysis"].h_idx
    k_idx = ctx_A0["cation_hydrolysis"].counterion_idx
    # Find the boundary integrals on electrode_marker:
    electrode_integrals = [
        intg for intg in F_res.integrals_by_type("exterior_facet")
        if intg.subdomain_id() == ctx_A0["bv_settings"]["electrode_marker"]
    ]
    # For each integral involving R_net, check which TestFunction
    # subspace it multiplies:
    # (specific UFL introspection — implementation in test fixture)
    ...
```

This slow test runs once at A0 and confirms slot wiring at the UFL
form level, not via vector slot inspection.

### Issue 6 (DOF sums fragile)
**Accept (folded into Issue 5 redesign).**  Scalar forms are the
primary check.  Vector DOF inspection is dropped.

### Issue 7 (`bundle.k_hyd_func.values()[0]` wrong pattern)
**Accept.**  Use `float(bundle.k_hyd_func)`.  Same pattern as
existing `collect_v10a_rung_diagnostics` at
`cation_hydrolysis.py:1029` which already does
`k_hyd = float(bundle.k_hyd_func)`.  No new code pattern.

---

## Section 2 — Updated plan (deltas)

### Form-build canonical stored expressions (FINAL)

```python
# At forms_logc[_muh].py build time, AFTER F_res is assembled:
ctx["_cation_hydrolysis_R_net_expr"] = R_net  # canonical UFL expr
ctx["_cation_hydrolysis_pka_shift_expr"] = pka_shift_expr
ctx["_cation_hydrolysis_sigma_S_expr"] = sigma_S_expr  # solved
ctx["_cation_hydrolysis_pka_sigma_S_expr"] = (
    fake_signed_sigma_S if sigma_singh_override is not None
    else sigma_S_expr
)
# Vector-form residual terms (for slot-wiring slow test):
ctx["_cation_hydrolysis_H_residual_term"] = (
    lam_func * R_net * v_list[h_idx_for_cation]
    * ds(electrode_marker)
)
ctx["_cation_hydrolysis_K_residual_term"] = (
    lam_func * (-R_net)
    * v_list[counterion_idx]
    * ds(electrode_marker)
)
# Scalar forms (for A0b + A3 gate 6 + downstream diagnostics):
ctx["_cation_hydrolysis_R_net_scalar_form"] = (
    lam_func * R_net * ds(electrode_marker)
)
ctx["_cation_hydrolysis_H_flux_scalar_form"] = (
    lam_func * R_net * ds(electrode_marker)
)
ctx["_cation_hydrolysis_K_flux_scalar_form"] = (
    lam_func * (-R_net) * ds(electrode_marker)
)
```

All stored artifacts share `R_net` (the canonical UFL expression
object).  No re-derivation.  Single source of truth.

### A0b final spec (scalar-only)

(See Issue 5 redesign — four scalar gates.)

### A3 final pass criteria (with new gate 6)

| # | Gate | Threshold | Catches |
|---|---|---|---|
| 1 | `\|pka_factor_avg_A3 - 10^(-β·σ_override)\| / 10^(-β·σ) < 5%` | rel 5% | Diagnostic rebuilds without override |
| 2 | `\|amp_from_singh_A3 - pka_factor_avg_A3\| / pka_factor_avg_A3 < 5%` | rel 5% | amp_from_singh-vs-pka_factor mismatch |
| 3 | `\|Γ_A3 - Γ_predicted_from_override_F0\| / Γ_max < 1e-3` | abs/Γ_max | Picard doesn't use override-aware F0 |
| 4 | Newton + Picard converged at λ=1 | — | — |
| 5 | All required-key diagnostics finite AND `c_K_boundary > 0.01·c_K_bulk` | — | NaN/Inf, severe depletion |
| 6 (NEW) | `\|flux_scalar_A3 - k_des·Γ_A3·area\| / max(\|k_des·Γ_A3·area\|, 1e-30) < 5e-3` | rel 5e-3 | Override doesn't reach residual R_net (split-brain) |

Gate 6 is the residual-side mass-balance closure, mirroring A0b
gate 1.  Together with gates 1-3, all four override consumers
(stored pKa expr, diagnostics, Picard, residual) are verified.

### Manufactured diagnostics top-level fields (FINAL)

(See Issue 3 ack — always emit `c_H_boundary_avg` and
`c_K_boundary_avg`.)

### Required-numeric-key per ablation (FINAL)

(See Issue 4 ack — `REQUIRED_NUMERIC_KEYS` dict.)

### Slot-wiring slow test (replaces A0b vector inspection)

(See Issue 5 redesign — `test_H_K_residual_terms_target_correct_slots`
inspects F_res's UFL integrals.)

### `float(bundle.k_hyd_func)` (FINAL)

(See Issue 7 ack.)  All ctx-bundle scalar accessors use
`float(bundle.X_func)`.  Confirmed against existing
`collect_v10a_rung_diagnostics` at `cation_hydrolysis.py:1029-1032`.

---

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

(Note: this is round 5 of a 5-round cap.  If your VERDICT remains
ISSUES_REMAIN, please clearly mark which of your issues are
blocking implementation vs. nits-with-suggested-fixes, so the
auto-revise step can prioritize correctly.)
