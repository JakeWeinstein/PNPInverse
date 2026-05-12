# R5 → GPT: counterreply + plan v5 patches

All 11 round-4 issues accepted.  Verified:
* JSON key is `rungs` (line `for rung in
  per_k_hyd_record.get('rungs', [])` in A.2 driver line 253).
* `grid_per_voltage.py:1039`: `visit_order = sorted(...)` —
  nearest-anchor ordering, you were right.
* A.2 flow is: `solve_anchor_with_continuation` builds anchor at
  V_anchor (e.g., +0.55) WITH λ ladder applied AT the anchor →
  warm-walk across V grid at fixed λ=1 (no per-V λ ramp).  Total
  per-Δ_β eval ≈ 1 anchor build (internally ~10 Newtons via kw +
  λ + C_S sub-ladders) + ~24 V points at fixed λ=1 = ~34 PDE
  solves.  So v4's ~30 estimate was approximately correct.

---

## Section 1 — Per-issue acknowledgments

**1. ACCEPT.**  A.2 lookup uses `target_record["rungs"]` (not
`lam_diagnostics`):
```python
matches = [rec for rec in payload["per_k_hyd_records"]
           if abs(float(rec["k_hyd_target"]) - 1e-3) < 1e-12]
assert len(matches) == 1, f"Expected unique k_hyd=1e-3; got {len(matches)}"
target_record = matches[0]

lam1_matches = [rung for rung in target_record["rungs"]
                if abs(float(rung["lambda_hydrolysis"]) - 1.0) < 1e-12]
assert len(lam1_matches) == 1, f"Expected unique λ=1.0 rung; got {len(lam1_matches)}"
lam1_diag = lam1_matches[0]
```

**2. ACCEPT.**  Continuation order is nearest-anchor sorted (the
existing `solve_grid_with_anchor` behavior).  v5 plan documents:
> "Each Δ_β evaluation: anchor at V_anchor = +0.55 V with the
> two-stage C_S, kw_eff, and λ ladders applied at the anchor.
> Then `solve_grid_with_anchor` visits the 24 V grid points in
> NEAREST-ANCHOR-FIRST order (sorted by `|V − V_anchor|` per
> `grid_per_voltage.py:1039`).  Each visit is a single Newton
> resolve at λ=1, warm-started from the nearest already-visited
> V (the anchor at first)."
No custom visit order; existing topology is reused as-is.

**3. ACCEPT.**  Per-V λ topology clarified in v5:
* **λ ladder is at the anchor (V=+0.55), not per-V.**  The
  anchor's converged state has λ=1.0 already.
* V-grid sweep with `solve_grid_with_anchor` does a single
  Newton resolve at each V at fixed λ=1.0 (no per-V λ ramp).
* Per-Δ_β cost confirmation: ~1 anchor (with internal ~10
  sub-ladder Newtons) + 24 V × 1 Newton = ~34 PDE solves.

**4. ACCEPT (no budget change).**  My v4 estimate of ~30 PDE
solves per eval was approximately correct; v5 updates the
estimate to ~34 PDE solves per eval.  Wall budget table:

| Activity | Δ_β evals | PDE solves | Wall |
|---|---|---|---|
| Δ_β=0 baseline + pre-screen σ map | 1 | ~34 | ~10 min |
| Stern pre-fit grid (7 new) | 7 | ~238 | ~70 min |
| Stern Brent | ~10 | ~340 | ~100 min |
| Ablation pre-fit grid (7 new) | 7 | ~238 | ~70 min |
| Ablation Brent | ~10 | ~340 | ~100 min |
| **Total** | **35 evals** | **~1190 PDE solves** | **~6 hours** |

(Δ_β=0 reused for Stern pre-fit grid — see #5 below.)

**5. ACCEPT.**  Δ_β=0 in Stern grid removed:
```python
delta_beta_grid_stern_additional = [
    -4.673e7,   # T ≈ -5
    -2.804e7,   # T ≈ -3
    -9.346e6,   # T ≈ -1
    -9.346e5,   # T ≈ -0.1
    -9.346e4,   # T ≈ -0.01
    -9.346e3,   # T ≈ -0.001
    -9.346e2,   # T ≈ -1e-4
]
# 7 points; Δ_β=0 baseline is INJECTED as the 8th from the
# byte-equiv baseline run, not re-computed.
```

**6. ACCEPT.**  Exact baseline_T computation:
```python
beta_K_Cu = -45.608196  # from compute_beta_per_cation("K")
override_sigma = 0.141   # Singh K-Cu cell-level
baseline_T_ablation = beta_K_Cu * override_sigma  # ≈ -6.43

# Lowermost ablation target safely inside [-14.9, +eps]:
T_lower_ablation = -14.9
delta_beta_lower_ablation = (T_lower_ablation - baseline_T_ablation) / override_sigma
# = (-14.9 - (-6.4308)) / 0.141 = -60.07 pm² (safely > -60.78)
```

**7. ACCEPT.**  All ablation grid values computed from exact
`compute_beta_per_cation("K")` and `override_sigma=0.141`:
```python
def ablation_delta_beta_for_target(T):
    return (T - beta_K_Cu * override_sigma) / override_sigma

delta_beta_grid_ablation_additional = [
    ablation_delta_beta_for_target(T) for T in
    [-14.9, -10.0, -8.0, -4.0, -1.0, -0.1]
]
# 6 additional points; Δ_β=0 baseline is INJECTED as the 7th
# from the byte-equiv baseline run.
```

**8. ACCEPT.**  Variable renamed `sigma_local_clamped_max_over_grid`
everywhere.  v5 driver uses this name consistently in
`eps_beta`, `xatol_beta_stern`, pre-screen bound checks, and the
identifiability slope estimator.

**9. ACCEPT.**  `gamma` fallback dropped.  Exact key
`gamma_final` (or whatever is in the A.2 JSON — verified in
Phase 10.A.0).  If the actual A.2 file uses a different name,
v5 driver `COMPARISON_FIELDS` is pinned to the actual name with
a regression test that breaks on schema rename.

**10. ACCEPT.**  Back-compat import tests for the moved
`SINGH_2016_CATION_PARAMS`:
```python
def test_singh_params_identity_singh2016_first(monkeypatch):
    import sys
    # Force singh2016 to load first
    if "scripts._bv_common" in sys.modules:
        del sys.modules["scripts._bv_common"]
    if "calibration.singh2016" in sys.modules:
        del sys.modules["calibration.singh2016"]
    from calibration.singh2016 import SINGH_2016_CATION_PARAMS as A
    from scripts._bv_common import SINGH_2016_CATION_PARAMS as B
    assert A is B

def test_singh_params_identity_bv_common_first(monkeypatch):
    # Force _bv_common to load first
    ... (analogous)

def test_singh_params_identity_both_imported():
    # Whichever order; assert is-identity
    ...
```
Three test variants for import-order permutations.

**11. ACCEPT.**  `beta_offset_pm2_func` representation:
* Lives in the cation hydrolysis bundle as a Firedrake
  `Function(R_space)` or `Constant(0.0)` (mirror of
  `r_H_El_pm_func`).
* Plumbed via `build_cation_hydrolysis_terms` →
  `cation_hydrolysis.py:294-380` (add line ~295:
  `beta_offset_init = float(raw_cfg.get("beta_offset_pm2", 0.0))`).
* Runtime setter `set_reaction_beta_offset_pm2_model(ctx, val)`:
  ```python
  def set_reaction_beta_offset_pm2_model(ctx: Dict[str, Any], val: float) -> None:
      bundle = ctx["cation_hydrolysis"]
      bundle.beta_offset_pm2_func.assign(float(val))
  ```
  Added to `_PARAMETER_OVERRIDE_SETTERS` at
  `anchor_continuation.py:1688`.
* Diagnostic emission (`collect_v10a_rung_diagnostics`):
  recomputes `β_K_carbon = β_K_Cu + beta_offset_pm2_func` from
  the live Firedrake Function value, NOT a closed-over capture.
* `_build_singh_2016_eq_4_pka_shift`: adds the bundle's
  `beta_offset_pm2_func` to the per-cation β at form-build
  time.  Form is rebuilt only when needed; live coefficient
  updates via `Function.assign()` propagate to the next Newton
  resolve.
* Tests:
  * `test_beta_offset_zero_byte_equivalent` — at offset=0, all
    diagnostics match v10a byte-for-byte at machine precision.
  * `test_beta_offset_nonzero_residual_diagnostic_agreement` —
    at offset=1e6 pm², emitted `pka_shift_avg` equals
    `(β_K_Cu + 1e6) · σ_local_clamped_avg` to rel < 1e-6.
  * `test_set_beta_offset_runtime_bump` — set + Newton resolve;
    verify ctx state updated and form sees new value.

---

## Section 2 — Plan v5 patches (against v4)

* **P50**: D5 A.2 lookup uses `rungs` not `lam_diagnostics`.
* **P51**: §4 continuation order: nearest-anchor sort per
  `solve_grid_with_anchor` existing topology.
* **P52**: §4 λ topology: ladder AT the anchor (V=+0.55), V-grid
  warm-walk at fixed λ=1.
* **P53**: §4 wall budget table updated to ~34 PDE solves per
  eval × 35 evals = ~1190 PDE solves × 0.5 min/PDE solve ≈
  ~10 hours wall.  (Single Newton solve in Phase A.2 records
  averaged ~5-10s for a converged warm-walk hop; conservative
  estimate is 30s.  Real wall depends on per-V convergence.)
* **P54**: §3.3 Stern pre-fit grid has 7 additional points;
  Δ_β=0 baseline is injected from the byte-equiv reproduction
  run, not re-computed.
* **P55**: §3.3 ablation grid uses exact arithmetic from
  `compute_beta_per_cation` and `override_sigma`; lowermost
  T = −14.9 inside `[-15, +eps]` safe domain.
* **P56**: Variable `sigma_local_max_at_V_kin` renamed to
  `sigma_local_clamped_max_over_grid` throughout.
* **P57**: D5 `COMPARISON_FIELDS` drops `gamma` fallback; uses
  exact `gamma_final` (or whatever is in A.2 JSON; verified in
  10.A.0).
* **P58**: D8 + new tests file
  `tests/test_calibration_singh2016_import_safety.py` with
  three import-order permutation tests.
* **P59**: D1 + D2 + D8 — `beta_offset_pm2_func` Firedrake
  `Function`, runtime setter, diagnostic emission consistency,
  three new fast tests.

---

## Section 3 — Continued critique prompt

Review v5.  Push back on poor defenses.  Raise new issues.
Verdict line at end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
