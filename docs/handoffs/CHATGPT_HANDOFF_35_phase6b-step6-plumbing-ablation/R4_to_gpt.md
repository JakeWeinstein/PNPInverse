# Round 4 counterreply — Step 6 plumbing ablation plan

## Section 1 — Acknowledgments (per R3 issue)

### Issues 1–4 (A0b residual-assembly is incorrect / weak / tautological)
**Accept all four.**  A0b is fully redesigned:

**A0b (final): four-part residual-wiring assertion** (all run at A0
converged state, no separate solve):

1. **Scalar source flux (magnitude gate):**
   At form-build, store
   `ctx["_cation_hydrolysis_R_net_scalar_form"] = λ_func · R_net · ds(electrode_marker)`
   (a scalar form, not vector — no test function).  At A0 converged
   state, assemble:
   ```python
   flux_scalar = float(fd.assemble(
       ctx["_cation_hydrolysis_R_net_scalar_form"]
   ))
   gamma = float(ctx["cation_hydrolysis"].gamma_func)
   k_des = float(ctx["cation_hydrolysis"].k_des_func)
   expected = k_des * gamma * electrode_area_nondim  # at λ=1
   ```
   Pass: `|flux_scalar - expected| / max(|expected|, 1e-30) < 5e-3`
   (mirrors the v10a mass-balance threshold; if R_net's
   form-side magnitude doesn't match k_des·Γ·area at SS, something
   is wrong residual-side).

2. **H+ residual slot wired (presence gate):**
   At form-build, store
   `ctx["_cation_hydrolysis_H_residual_term"]` = the EXACT linear
   form `λ_func · R_net · v_list[h_idx] · ds(electrode_marker)`
   that is subtracted into F_res.  At A0 converged state, assemble:
   ```python
   vec_H = fd.assemble(ctx["_cation_hydrolysis_H_residual_term"])
   # vec_H is a Cofunction (dual vector) in the mixed test space.
   # Extract per-subspace norms:
   h_slot_norm = vec_H.subfunctions[h_idx].dat.data_ro.copy()
   # h_slot_norm is the array of nodal values for the H+ slot.
   ```
   Pass:
   * `numpy.linalg.norm(h_slot_norm) > 0` (H+ slot populated;
     catches an `apply_h_source=False` not wired).
   * For each non-H subspace j ≠ h_idx:
     `numpy.linalg.norm(vec_H.subfunctions[j].dat.data_ro) < 1e-15`
     (only H slot is populated; catches a wrong-slot wiring bug
     where R_net is mistakenly added to, e.g., the c_O2 residual).

3. **K+ residual slot wired (presence + sign gate):**
   Same as gate 2 but for `_cation_hydrolysis_K_residual_term =
   λ_func · (-R_net) · v_list[k_idx] · ds(electrode_marker)`.
   Pass:
   * `numpy.linalg.norm(vec_K.subfunctions[k_idx].dat.data_ro) > 0`.
   * Off-slot norms < 1e-15.

4. **Form-construction anti-symmetry (sign gate):**
   ```python
   sum_h_dofs = float(numpy.sum(
       vec_H.subfunctions[h_idx].dat.data_ro
   ))
   sum_k_dofs = float(numpy.sum(
       vec_K.subfunctions[k_idx].dat.data_ro
   ))
   ```
   Pass: `|sum_h_dofs + sum_k_dofs| / max(|sum_h_dofs|, 1e-30) <
   1e-12` (the sum of H-slot DOF values for H form equals negative
   sum of K-slot DOF values for K form; this is the integrated
   anti-symmetry).

   GPT R3 issue #3's point ("anti-symmetry is tautological if I
   build R_net and -R_net manually") is addressed because **these
   are the EXACT forms that go into F_res** (stored on ctx during
   form-build at the canonical wiring site), not re-derived
   expressions.  The test reads the wiring; the form construction
   establishes anti-symmetry by virtue of using the same R_net and
   v_list[idx] from the same code path.

Together, gates 1–4 catch:
* `build_proton_boundary_source` returns 0 → flux_scalar < expected
  → gate 1 fails.
* H+ wiring broken (`apply_h_source=False` plumbed even at A0) →
  gate 2's h_slot_norm = 0 → fail.
* K+ wiring broken → gate 3 fails.
* Wrong-slot bug (R_net into, e.g., c_O2 residual) → gates 2-3
  off-slot norms exceed 1e-15 → fail.
* Sign error in `-R_net` for K slot → gate 4 fails.

### Issue 5 (`ctx["nondim"]["bv_convergence"]` is wrong path)
**Accept.**  Use `ctx.get("bv_convergence", {})` directly.  Fixed
in plan's `collect_v10a_rung_diagnostics` snippet.

### Issue 6 (A3 gate 4 `Γ_A3 > Γ_A0` too weak)
**Accept.**  Replace with **Picard-prediction match**:

```python
F0_A3 = (
    bundle.k_hyd_func.values()[0]  # k_hyd nondim
    * forward_avg_no_k_hyd_A3        # ⟨c_K · pka_factor_override⟩
)
# Closed-form Γ_ss at λ=1 with override-active pka_factor:
gamma_predicted = gamma_ss_langmuir(
    F0=F0_A3,
    k_des=k_des, k_prot=k_prot,
    c_H_avg=c_H_avg_A3,
    delta_ohp=delta_ohp_hat,
    gamma_max=gamma_max,
    lam=1.0,
)
# Picard outer tolerance is 1e-4 (anchor_continuation.py:1809).
assert abs(gamma_A3 - gamma_predicted) / gamma_max < 1e-3
```

Pass criterion: `|Γ_A3 - Γ_predicted_from_override_F0| / Γ_max <
1e-3` (loose Picard band).  Proves that Picard's Γ update used the
override-aware F0.

### Issue 7 (A3 gate 3 wrong denominator)
**Accept.**  Direct gate (cleaner):

A3 gate 3 (final): `|amp_from_singh_A3 - pka_factor_avg_A3| /
pka_factor_avg_A3 < 5%`.

In override mode, pka_factor is a Constant on the boundary, so
`⟨c_K · const⟩ / ⟨c_K⟩ = const` exactly (up to FP).  The gate
catches the case where diagnostics rebuild pka_factor independently
without seeing the override (Issue 8 ack below).

Removed the comparative `ratio / A0_ratio` form.

### Issue 8 (gates 1+2 don't prove "override-in-form" unless diagnostics share the form's pKa expression)
**Accept.**  `collect_v10a_rung_diagnostics` consumes the
ctx-stored expression:

```python
# In collect_v10a_rung_diagnostics:
pka_shift_expr = ctx.get(
    "_cation_hydrolysis_pka_shift_expr", None,
)
if pka_shift_expr is None:
    # Backward-compat fallback (existing path).
    pka_shift_expr = build_pka_shift(
        cation_params=bundle.cation_params,
        sigma_S=sigma_S_expr,
        r_H_El_func=bundle.r_H_El_pm_func,
    )
```

Single source of truth: forms build the pka_shift_expr (with or
without override), store it on ctx, and diagnostics consume it.
Picard's `update_gamma_from_solution` does the same.  All three
consumers agree by construction.

### Issue 9 (structural test "only Constants" wrong)
**Accept.**  Test asserts independence from `ctx["U"]` specifically:

```python
def test_override_pka_shift_independent_of_U(ctx_override):
    pka_expr = ctx_override["_cation_hydrolysis_pka_shift_expr"]
    coeffs = ufl.algorithms.extract_coefficients(pka_expr)
    U = ctx_override["U"]
    assert U not in coeffs, (
        "override-mode pka_shift_expr must not depend on solved U; "
        "found U in extract_coefficients output"
    )
    # r_H_El_pm_func is a Firedrake Function (mutable coefficient),
    # but NOT solved.  It IS allowed in coeffs.
```

### Issue 10 (sigma_S_active_expr ambiguous)
**Accept.**  Two stored names:

* `ctx["_cation_hydrolysis_sigma_S_expr"]` — Stern σ_S, ALWAYS
  solved-field-dependent (used for σ in residual residue, Bikerman
  packing, etc.).
* `ctx["_cation_hydrolysis_pka_sigma_S_expr"]` — pKa-side σ; fake
  signed Constant in override mode, solved expression otherwise.

Stored separately at form-build.  Tests inspect the right one.

### Issue 11 (positivity tautological in log-c form)
**Accept.**  Replace with severity gates:

* A1: `c_K_boundary_avg_A1 > 0.01 · c_K_bulk` (catches K depletion
  bleeding into A1 via electrostatic feedback; not expected but
  guard).
* A2: `c_K_boundary_avg_A2 > 0.01 · c_K_bulk` (the actual A2
  primary signal: K+ should fall but not below 1% of bulk; if
  c_K → 0, the manufactured break overdrove the system).
* All ablations: `numpy.isfinite(diag[key])` for every numeric
  diagnostic; NaN/Inf in any diagnostic ⇒ classify as "diagnostic
  failure", do not pass.

### Issue 12 (rel 1e-9 too strict for A0)
**Accept.**  Tiered table restored (was R2; collapsed in R3 by
mistake):

| Tier | Rel range | Action |
|---|---|---|
| Pass | ≤ 1e-9 | continue |
| Investigate | 1e-9 to 1e-6 | re-run once; if both ≤ 1e-6 and within rel 1e-6 of each other, document PETSc-determinism note and continue; if either > 1e-6, block |
| Block | > 1e-6 | block; debug residual-side wiring |

Per-observable basis (γ, θ, σ_S_C_per_m2, cd_mA_cm2, c_H_avg,
c_K_avg, R_2e_current_nondim, R_4e_current_nondim).  All must clear
the same tier for A0 to pass.

---

## Section 2 — Updated plan (deltas)

### Form-build canonical stored expressions

At `forms_logc[_muh].py` build time, the following ctx keys are
populated AFTER the residual is assembled:

```python
ctx["_cation_hydrolysis_pka_shift_expr"] = pka_shift_expr
ctx["_cation_hydrolysis_sigma_S_expr"] = sigma_S_expr  # solved
ctx["_cation_hydrolysis_pka_sigma_S_expr"] = (
    fake_signed_sigma_S if sigma_singh_override is not None
    else sigma_S_expr
)
ctx["_cation_hydrolysis_H_residual_term"] = (
    lam_func * R_net * v_list[h_idx_for_cation]
    * ds(electrode_marker)
)
ctx["_cation_hydrolysis_K_residual_term"] = (
    lam_func * (-R_net)
    * v_list[cation_hydrolysis_bundle.counterion_idx]
    * ds(electrode_marker)
)
ctx["_cation_hydrolysis_R_net_scalar_form"] = (
    lam_func * R_net * ds(electrode_marker)
)
```

These are the **canonical** wiring artifacts.  All downstream
consumers (Picard, diagnostics, A0b residual-assembly test) read
from here.

### A0b final spec

(See Issues 1-4 ack above — four-part residual-wiring assertion.)
A0b runs in the rung_callback after A0's λ=1 rung converges.
Wall-time cost: 4 boundary-form assembles + 4 norm-computations,
all on FP64.  <100 ms.

### A3 final pass criteria

| # | Gate | Threshold | Catches |
|---|---|---|---|
| 1 | `\|pka_factor_avg_A3 - 10^(-β_K_Cu·σ_override)\| / 10^(-β·σ) < 5%` | rel 5% | Override doesn't reach diagnostic-rebuilt pka_factor |
| 2 | `\|amp_from_singh_A3 - pka_factor_avg_A3\| / pka_factor_avg_A3 < 5%` | rel 5% | Override-mode amp_from_singh inconsistent with pka_factor |
| 3 | `\|Γ_A3 - Γ_predicted_from_override_F0\| / Γ_max < 1e-3` | abs/Γ_max < 1e-3 | Picard's Γ update doesn't use override-aware F0 |
| 4 | Newton + Picard converged at λ=1 | — | — |
| 5 | All `c_K_boundary_avg > 0.01 · c_K_bulk` AND all diagnostics finite | — | Severe depletion / NaN |

Gates 1+2 prove override-in-form + override-in-diagnostics.  Gate 3
proves override-in-Picard.  Together: the override is centralized
correctly via the ctx-stored pka_shift_expr.

### Manufactured-aware collect_v10a_rung_diagnostics (revised)

```python
def collect_v10a_rung_diagnostics(ctx, *, electrode_marker=None):
    # ... existing setup ...
    bv_conv = ctx.get("bv_convergence", {})  # GPT R3 #5 fix
    manufactured_R_inj = bv_conv.get("manufactured_R_inj", None)
    apply_h_source = bv_conv.get("apply_h_source", True)
    apply_k_sink = bv_conv.get("apply_k_sink", True)

    if manufactured_R_inj is not None:
        diag["manufactured_run"] = True
        diag["manufactured_R_inj"] = float(manufactured_R_inj)
        diag["apply_h_source_active"] = bool(apply_h_source)
        diag["apply_k_sink_active"] = bool(apply_k_sink)
        # Still emit basic surface concentrations + Γ + σ_S etc.
        # for A1/A2 c_H/c_K shift gates.
        # Physical-path fields → None:
        diag["F0_avg"] = None
        diag["forward_avg_no_k_hyd"] = None
        diag["denominator_kprot"] = None
        diag["denominator_cap"] = None
        diag["denominator_total"] = None
        diag["denominator_cap_to_total_ratio"] = None
        diag["R_forward_capped"] = None
        diag["pka_shift_avg"] = None
        diag["F0_decomposition"] = None
        diag["R_4e_decomposition_log"] = None
        return diag
    # Physical path: consume ctx-stored pka_shift_expr (GPT R3 #8).
    pka_shift_expr = ctx.get(
        "_cation_hydrolysis_pka_shift_expr", None,
    )
    if pka_shift_expr is None:
        # Backward-compat (legacy callers).
        pka_shift_expr = build_pka_shift(
            cation_params=bundle.cation_params,
            sigma_S=sigma_S_expr,
            r_H_El_func=bundle.r_H_El_pm_func,
        )
    # ... rest of existing physical-path collection ...
```

### Stored-expression structural tests (revised)

* `test_override_pka_shift_independent_of_U` — extract_coefficients
  on `ctx["_cation_hydrolysis_pka_shift_expr"]` in override mode;
  assert `ctx["U"] not in coeffs`.  Allow `r_H_El_pm_func` and
  other bundle Functions in coeffs.

* `test_stern_sigma_S_remains_solved_in_override_mode` —
  `extract_coefficients(ctx["_cation_hydrolysis_sigma_S_expr"])`
  should include `ctx["U"]` even when override is active (the
  Stern-side σ remains solved-field-dependent).

* `test_H_K_residual_terms_anti_symmetric_at_A0` — slow test;
  assembles vec_H, vec_K at A0 converged state and asserts the
  four conditions from Issues 1-4 ack.

### Severity gates (revised pass criteria for all ablations)

* All ablations: `numpy.isfinite(value)` for every assembled
  diagnostic.  Any NaN/Inf → "diagnostic failure" (separate from
  "ablation failed gate").
* A2 specifically: `c_K_boundary_avg_A2 > 0.01 · c_K_bulk`.  If
  violated, lower R_INJ_MFG_A2 and re-run pre-pass.

### Byte-equivalence tiered table (restored from R2)

(See Issue 12 ack — restored per-observable tiered table.)

---

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
