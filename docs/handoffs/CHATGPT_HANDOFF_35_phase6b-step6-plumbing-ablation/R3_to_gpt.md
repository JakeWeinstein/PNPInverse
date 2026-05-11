# Round 3 counterreply — Step 6 plumbing ablation plan

## Section 1 — Acknowledgments (per R2 issue)

### Issues 1–2 (sign convention + sentinel direction)
**Accept both.** I had the cathodic/anodic clamp logic backwards in R2.

Correct convention (from `cation_hydrolysis.py:516`):
* `sigma_singh = max(0, -sigma_S_signed)` (counts/pm²; non-negative).
* At V_kin, `sigma_S_signed ≈ -0.017 C/m²` (cathodic).  Then
  `-sigma_S_signed = +0.017 C/m²` (positive), passes the clamp,
  becomes `+0.017 × 6.2415e-6 ≈ +1.07e-7 counts/pm²`.  Matches A.2's
  `pka_shift_avg ≈ -4.88e-6`.
* K/Cu: `r_H_El = 200.98 < r_M + r_O = 201` ⇒ `G = (1 - r_M-O²/r_H_El²) < 0`.
* `ΔpKa = +2·A·z·r_H_El·G·sigma_singh`.  With G<0, A>0, z=+1, r_H_El>0,
  sigma_singh > 0: ΔpKa < 0 (cathodic pKa lowering ✓).

Sentinel value `0.022 counts/pm²` gives `β · σ ≈ -45.61 · 0.022 ≈ -1.003`
⇒ `pka_factor = 10^(-ΔpKa) = 10^(+1.003) ≈ 10.07`.  So R_net is
**amplified** by ~10× (capped by `(1 − Γ/Γ_max)` Langmuir factor as
Γ saturates).

Revised A3 expected behavior: forward branch amplifies, Γ moves
toward Γ_max, capped R_net plateaus.  The discriminator is no longer
"R_net suppression" but "R_net amplification on the uncapped forward
branch" (Issue 6 forces us to look at uncapped quantities anyway).

Singh sign-convention audit (Prerequisites) becomes simpler — the
sign is consistent; the audit just confirms the implementation
matches the convention I now describe correctly.

### Issues 3–4 (A0b 1% c_H gate unsupported; flux-magnitude logic wrong)
**Accept.**  A0b's solution-level c_H gate is replaced by a
**residual-assembly test**:

**A0b — Physical R_net assembly sanity (no solve required).**

After A0 form-build, before Newton, assemble:
```python
# Both should be ZERO at λ=0 (Picard hasn't ticked Γ yet).
# After running A0 to convergence at λ=1, assemble:
phys_h_flux = float(fd.assemble(
    cation_hydrolysis_bundle.lambda_hydrolysis_func
    * R_net_default_at_A0_state
    * v_h * ds(electrode_marker)
))
phys_k_flux = float(fd.assemble(
    cation_hydrolysis_bundle.lambda_hydrolysis_func
    * (-R_net_default_at_A0_state)
    * v_k * ds(electrode_marker)
))
```

Where `R_net_default_at_A0_state` is the physical R_net expression
(rebuilt from the A0 solution's `c_M`, `c_H`, `σ_S`, `pka_shift`).
`v_h` and `v_k` are unit test functions on the H+ and K+ residual
slots, restricted to the electrode marker.

Pass criteria:
* `|phys_h_flux| > 1e-30` (physical source actually fires).
* `|phys_k_flux + phys_h_flux| / max(|phys_h_flux|, 1e-30) < 1e-9`
  (equal and opposite within FP noise — confirms the K+ sink is
  symmetric to the H+ source).
* `phys_h_flux` sign matches expected (positive when net proton
  production cathodic, given β·σ < 0 at V_kin).

This is a **direct test of the residual-side wiring** that runs
independently of c_H's observable response.  The c_H sensitivity is
weak (A.2 shows ~0.1% range, per GPT's Issue 3) but the residual
integral is direct.

### Issue 5 (ctx accessor doesn't switch baked UFL form)
**Accept.**  Dropping the runtime accessor.  The plan moves to
**rebuild forms per ablation** (the existing SolverParams pattern):

Each ablation builds a fresh `sp` via `_build_sp_for_ablation(
ablation_id)` that sets the appropriate `bv_convergence` keys
(`manufactured_R_inj`, `apply_h_source`, `apply_k_sink`,
`override_sigma_singh_counts_pm2`).  The driver calls
`solve_lambda_ramp_from_warm_start` once per ablation; each call
gets `build_context + build_forms` fresh from the new sp.  No
runtime mutation of the override.

No new mutable Constant coefficients are needed.  The existing
`manufactured_R_inj` plumbing already follows this pattern
(`forms_logc_muh.py:728-732` reads at form-build time, not runtime).
The new flags do the same.

### Issues 6–7 (capped R_net comparison wrong; pka-factor formula garbled)
**Accept both.**  Revised A3 pass criteria (final):

A3 compares the **uncapped forward branch** quantities — these are
the σ-mapping-sensitive observables:

1. **`pka_factor_avg_A3 ≈ 10^(-β_K_Cu · σ_override) ≈ 10.07`**
   within rel 5%.  This is the constant pKa factor when override is
   active; it's straight algebra.
2. **`c_K_pka_product_avg / c_K_avg ≈ pka_factor_avg`** within rel
   5% (sanity: when pka_factor is a Constant via override, the
   boundary-averaged product equals `c_K_avg · pka_factor`).
3. **`F0_decomposition.amplification_from_singh_A3` ≈
   pka_factor_avg_A3 / pka_factor_avg_A0`** within rel 10% (the
   defined Singh amplification ratio should reflect the override).
4. **`Γ_A3` approaches `Γ_max`** monotonically vs A0's Γ=0.0405
   (because forward branch is amplified, Γ_ss saturates toward
   Γ_max; Picard's closed-form predicts the new Γ_ss).
5. **Convergence** (Newton + Picard) at λ=1.

Gates 1+2 prove the override reached the FORM (forms_logc[_muh]).
Gate 3 proves the override reached the DIAGNOSTICS
(collect_v10a_rung_diagnostics).
Gate 4 proves the override reached the PICARD update.
Together: the override is centralized correctly.

### Issue 8 (R_inj bracket should search both directions)
**Accept.**  Initial bracket extended to
**`{1e-4, 1e-3, 1e-2, 1e-1, 1.0}`** (5 values, half-decade-ish in
log).  Selection logic walks from smallest, picks the first that
satisfies 5–25% AND convergence AND positivity.  Only escalates to
`{2.0, 5.0, 10.0}` if all 5 initial values fail the upper bound
(too small) OR fail the lower bound (too large; only possible at
1e-4 or 1e-3 in the typical regime — then we go up, not down).

If 1e-4 already has |Δc| > 25% at A1, the system is super-sensitive
and the plumbing test is inconclusive at sentinel scales; report
"plumbing test inconclusive — solver sensitivity dominates" rather
than artificially passing with a too-small value.

### Issue 9 (A1 and A2 may need separate R_inj sentinels)
**Accept.**  Separate brackets:

* Pre-pass for A1: scan `{1e-4, 1e-3, 1e-2, 1e-1, 1.0}` at A1
  flags.  Pick smallest with 5–25% c_H rise + convergence +
  positivity.  Record as `R_INJ_MFG_A1`.
* Pre-pass for A2: same scan at A2 flags.  Record as `R_INJ_MFG_A2`.

The two sentinels are documented independently in the JSON config
block; "same magnitude" is not a load-bearing constraint.

### Issue 10 (driver-only validation too weak — move upstream)
**Accept.**  Validation moves to `Forward/bv_solver/config.py` (the
canonical raw-config parser):

```python
# In parse_solver_params(...) or wherever bv_convergence is built:
manufactured_R_inj_raw = raw.get("manufactured_R_inj", None)
manufactured_R_inj = ... (existing)
apply_h_source = _bool(raw.get("apply_h_source", True))
apply_k_sink   = _bool(raw.get("apply_k_sink",   True))
override_raw = raw.get("override_sigma_singh_counts_pm2", None)
# ... (existing validation from R2 Issue 16)

# Cross-validation:
half_h = (not apply_h_source)
half_k = (not apply_k_sink)
if (half_h or half_k) and manufactured_R_inj is None:
    raise ValueError(
        "apply_h_source=False / apply_k_sink=False require "
        "manufactured_R_inj to be set; otherwise the Γ Picard formula "
        "uses a physically inconsistent forcing.  See plan §"
        "'Revised flags table' for the matrix of valid combos."
    )
if (override_sigma_singh_counts_pm2 is not None
    and manufactured_R_inj is not None):
    raise ValueError(
        "override_sigma_singh_counts_pm2 + manufactured_R_inj are "
        "mutually exclusive: override is a physical-path imposition; "
        "manufactured bypasses the physical path."
    )
```

The Step 6 driver inherits this validation; ad-hoc external scripts
calling `make_bv_solver_params` also hit it.

### Issue 11 (collect_v10a_rung_diagnostics called by solver, not driver)
**Accept.**  `collect_v10a_rung_diagnostics` becomes
**manufactured-aware**:

```python
def collect_v10a_rung_diagnostics(ctx, *, electrode_marker=None):
    # ... existing ...
    manufactured_R_inj = ctx.get("nondim", {}).get(
        "bv_convergence", {}
    ).get("manufactured_R_inj", None)

    if manufactured_R_inj is not None:
        # Manufactured run: emit a different schema.
        diag["manufactured_R_inj"] = float(manufactured_R_inj)
        diag["manufactured_run"] = True
        # Still emit Γ, θ, c_H_avg, c_K_avg, σ_S_C_per_m2,
        # R_2e/R_4e for downstream comparison.  But mark physical-
        # path fields as inapplicable:
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
    # ... existing physical-path collection ...
```

The driver's `classify_ablation_status` for A1/A2 reads
`diag["manufactured_run"] is True` and skips F0/mass-balance gates.

Side effect: `solve_lambda_ramp_from_warm_start`'s emitted rung_diag
for manufactured runs now has explicit None for the physical fields
(instead of stale values or exceptions).  Step 6's driver expects
this; A.2/v10a' drivers are unaffected because they never set
`manufactured_R_inj`.

### Issue 12 (UFL structural test overbroad)
**Accept.**  Test inspects `ctx["_cation_hydrolysis_pka_shift_expr"]`
specifically, not the full F_res:

```python
# In forms_logc[_muh].py, store the active pka_shift_expr on ctx:
ctx["_cation_hydrolysis_pka_shift_expr"] = pka_shift_expr
ctx["_cation_hydrolysis_sigma_S_active_expr"] = (
    fake_signed_sigma_S if sigma_singh_override is not None
    else sigma_S_expr
)

# In the slow test:
from ufl.algorithms.analysis import extract_coefficients
def test_pka_shift_uses_override_when_active(ctx_with_override):
    pka_expr = ctx_with_override["_cation_hydrolysis_pka_shift_expr"]
    coeffs = extract_coefficients(pka_expr)
    # The override pka_shift should depend ONLY on Constants
    # (cation_params constants + sigma_singh override Constant), not
    # on U or its components.
    U = ctx_with_override["U"]
    assert U not in coeffs, (
        "override-mode pka_shift_expr should not depend on solved U"
    )

def test_residual_sigma_S_still_depends_on_solved_fields(ctx):
    sigma_S_in_residual = ctx["_cation_hydrolysis_sigma_S_active_expr"]
    # Even in override mode, the RESIDUAL-side σ_S (used in Stern BC,
    # Bikerman, etc.) must still be the solved one.  Only the
    # ΔpKa-side σ_S is overridden.
    # ... assert sigma_S_in_residual_for_stern_BC depends on U ...
```

Net: the test inspects two specific stored expressions, not the
whole residual.

### Issue 13 (1% noise floor asserted not established)
**Accept.**  A0/A0b drop the solution-level 1% gate; physical-path
detection moves entirely to the **residual-assembly test** (Issues
3-4 ack).

For A1/A2: the sentinel R_inj is chosen so that the c_H/c_K shifts
are 5–25% (Issue 8-9 ack), far above any plausible numerical-noise
floor.  No 1% gate needed.

For A3: solution-level gates are NOT used (Issues 6-7 ack); A3 uses
form-level pka_factor + diagnostic-level amplification_from_singh.

Net: no 1% solution-level gate remains in the plan.  All gates are
either residual-assembly (A0/A0b) or large-signal sentinel (A1/A2)
or form/diagnostic (A3).

### Issue 14 (audit branch logic — does it invalidate A.2 baseline?)
**Accept.**  Explicit branch:

* **σ-unit audit outcome A**: docs/tests have stale unit labels but
  `Forward/bv_solver/cation_hydrolysis.py:457-467` is correct.
  → Fix docs/tests only.  A.2 baseline remains valid.  Continue
    with step 6.
* **σ-unit audit outcome B**: `cation_hydrolysis.py:457-467`'s
  conversion is wrong.
  → Solver math change.  RE-RUN v10a' + A.2 with the corrected
    conversion before step 6.  Locked invariants
    (`K0_R4e_factor=1e-14`, `C_S=0.20`, `V_kin=-0.10`) may shift.
* **Singh sign audit outcome A**: `cation_hydrolysis.py:528`
  formula sign is consistent with the convention I described in
  Issue 1 ack.
  → No change.  A.2 baseline remains valid.
* **Singh sign audit outcome B**: Formula sign is inconsistent
  (cathodic σ_singh gives anti-Singh ΔpKa direction).
  → Solver math change.  RE-RUN v10a' + A.2 with the corrected
    sign before step 6.

Plan adds a Prerequisites step ordered before step 6's main run,
with explicit gate: "if audit changes solver math, re-run v10a' +
A.2 and re-establish locked invariants BEFORE proceeding to step 6".

---

## Section 2 — Updated plan (deltas)

The plan's R2-update sections are revised as follows.

### Prerequisites (revised)

1. **σ-conversion audit** (~1-2 hours):
   * Audit `tests/test_phase6b_v10a_langmuir_cap.py:297-304`.
   * Audit `docs/phase6/singh_2016_pka_formula.md:241`.
   * Confirm `Forward/bv_solver/cation_hydrolysis.py:457-467`'s
     conversion.
   * Decision tree (see Issue 14 ack).
2. **Singh sign convention audit** (~30 min):
   * Trace cathodic σ_S → σ_singh = max(0, -σ_S) > 0 → ΔpKa via
     `+2·A·z·r_H_El·G·σ_singh` with `G<0` at K/Cu (r_H_El=200.98 <
     r_M-O=201) → ΔpKa < 0 (Singh's cathodic lowering ✓).
   * Verify against A.2's `pka_shift_avg ≈ -4.88e-6` sign.
   * Decision tree (see Issue 14 ack).
3. **A.2 baseline preservation**: if either audit changes solver
   math, re-run v10a' + A.2 BEFORE step 6.

### Revised A3 (final)

* **A3 setup**: `apply_h_source=True`, `apply_k_sink=True`,
  `manufactured_R_inj=None`, `override_sigma_singh_counts_pm2 =
  SIGMA_SINGH_PLUMBING_SENTINEL = 0.022 counts/pm²` (gives
  `pka_factor_avg ≈ 10.07`; AMPLIFIES R_net 10×, capped by Langmuir).

* **A3 expected behavior** (REVISED — amplification, not suppression):
  * Uncapped forward branch (F0) amplifies ~10×.
  * Γ_A3 → Γ_max (Langmuir cap engages; Γ_A3 > Γ_A0 = 0.0405,
    likely 0.046–0.047).
  * Capped R_net at λ=1: bounded by k_des·Γ_max = 0.047 (vs A0's
    k_des·Γ_A0 = 0.0405).

* **A3 pass criteria** (final, replaces R2 Issue 10 list):
  1. `pka_factor_avg_A3` matches `10^(-β_K_Cu·σ_override)` within
     rel 5%.
  2. `c_K_pka_product_avg / c_K_avg` ≈ `pka_factor_avg_A3` within
     rel 5%.
  3. `amp_from_singh_A3 / amp_from_singh_A0` ≈ pka_factor_avg_A3 /
     pka_factor_avg_A0 within rel 10%.
  4. `Γ_A3 > Γ_A0` (monotone toward Γ_max).
  5. Newton + Picard converged at λ=1.

Gates 1+2 prove override-in-form; gate 3 proves
override-in-diagnostics; gate 4 proves override-in-Picard; gate 5 is
convergence.

### Revised A0b (final)

* **A0b — Physical-path residual-assembly sanity (no solve change)**:
  * Run A0 once to convergence at λ=1.
  * In the rung_callback, assemble the physical H/K residual
    contributions directly: `phys_h_flux = ∫ λ · R_net · v_h · ds`
    and `phys_k_flux = ∫ λ · (-R_net) · v_k · ds`.
  * Pass criteria:
    - `|phys_h_flux| > 1e-30` (physical source nonzero).
    - `|phys_h_flux + phys_k_flux| / max(|phys_h_flux|, 1e-30) <
      1e-9` (source/sink anti-symmetric).
    - `phys_h_flux` sign matches expected (positive at V_kin given
      ΔpKa<0).

A0b is **not a separate solve** — it's a post-A0 residual-assembly
check that runs in <1 ms.  No wall-time cost.

### Revised flag-validation surface (final)

* Validation moves to `Forward/bv_solver/config.py` (Issue 10 ack).
* Three rules:
  1. `apply_h_source=False` or `apply_k_sink=False` requires
     `manufactured_R_inj is not None`.
  2. `override_sigma_singh_counts_pm2 is not None` requires
     `manufactured_R_inj is None`.
  3. `override_sigma_singh_counts_pm2`, if set, must be finite and
     ≥ 0.

### Revised collect_v10a_rung_diagnostics (final)

* Manufactured-aware (Issue 11 ack).
* Emits `manufactured_run: bool` and `manufactured_R_inj: float | None`.
* For manufactured runs, sets all physical-path fields (F0_avg,
  denominator_*, etc.) to None.
* Driver's `classify_ablation_status` reads `manufactured_run` and
  skips physical-path gates for A1/A2.

### Revised UFL structural test (final)

* Stores `ctx["_cation_hydrolysis_pka_shift_expr"]` and
  `ctx["_cation_hydrolysis_sigma_S_active_expr"]` at form-build.
* Test inspects these specific expressions, not the full F_res
  (Issue 12 ack).

### Revised pass-criteria summary table

| Ablation | Primary gate | Convergence | Positivity | Physical-path check |
|---|---|---|---|---|
| A0 | reproduces A.2 baseline per-observable rel 1e-9 | Newton+Picard | c_H,c_K ≥ 0 | A0b |
| A0b | `|phys_h_flux|>1e-30` AND `|phys_h+phys_k|/|phys_h|<1e-9` AND sign correct | (uses A0 solve) | — | — |
| A1 | `5% ≤ Δc_H_vs_A0_rel ≤ 25%` | Newton+Picard | c_H,c_K ≥ 0 | — |
| A2 | `5% ≤ |Δc_K_vs_A0_rel| ≤ 25%` | Newton+Picard | c_H,c_K ≥ 0 | — |
| A3 | 5 gates (Issue 6-7 ack final list above) | Newton+Picard | c_H,c_K ≥ 0 | — |

(Removed: 1% noise-floor gates; 3σ language; tautological diagnostic
gate without residual check.)

---

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
