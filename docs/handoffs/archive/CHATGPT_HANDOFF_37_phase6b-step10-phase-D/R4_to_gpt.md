# R4 → GPT: counterreply + plan v4 patches (concrete final forms)

All 18 round-3 issues accepted.  Verified actual schema fields:
* Per-rung keys in `per_k_hyd_records` are `k_hyd_target` and
  `lambda_hydrolysis` (line 253, 487, 595 of A.2 driver).
* Observable keys are `cd_mA_cm2` (no Unicode), `theta` (no
  Greek) (lines 319, 400, 424).  My v3 used Unicode keys; v4
  fixed.

This round I'm committing to **specific final forms** to drive
to convergence.  No more hedging between Stern and ablation
budgets; no more grid-vs-bracket mismatch.

---

## Section 1 — Per-issue acknowledgments

**1. ACCEPT.**  T-grid contradiction fixed.  Final form:
* **T-target grid (production / Stern path):** explicit Δ_β
  values, NOT T values.  Driver iterates:
  ```python
  delta_beta_grid_stern = [
      -4.673e7,   # T_target ≈ -5  at σ_local(V_kin)
      -2.804e7,   # T_target ≈ -3
      -9.346e6,   # T_target ≈ -1
      -9.346e5,   # T_target ≈ -0.1
      -9.346e4,   # T_target ≈ -0.01
      -9.346e3,   # T_target ≈ -0.001
      -9.346e2,   # T_target ≈ -1e-4
      0.0,        # baseline (T_target ≈ -4.88e-6 at V_kin)
  ]
  # 8 evaluations.
  ```
* `Δ_β = 0` is the explicit V10B baseline anchor.  T = 0 is NEVER
  in the grid (it's the sign-flip boundary, hence the open upper
  bound of the optimizer).

**2. ACCEPT.**  Closed-bounds bracket:
```python
sigma_local_max_at_V_kin = abs(sigma_local_clamped_max)  # from baseline scan
eps_T = 1e-6   # ΔpKa-effect margin from the sign-flip boundary
eps_beta = eps_T / sigma_local_max_at_V_kin
upper = -beta_K_Cu - eps_beta   # closed; < +45.608 strictly
lower = -4.673e7   # corresponds to T = -5
```

**3. ACCEPT.**  Grid `T ∈ {−5, −3, −1, −0.1, −0.01, −0.001, −1e-4,
baseline}` (8 points) — covers `[−5, baseline]` continuously.
Bracket `[lower, upper] = [−4.673e7, −β_K_Cu − eps_beta]` aligns.

**4. ACCEPT.**  Ablation grid is **separately constructed**:
```python
# At override_sigma=0.141, Δ_β=0 ⇒ T = β_K_Cu·0.141 ≈ -6.43.
# Ablation T target shifts must overlap [-15, +eps] (safe domain).
# Grid centered around the Δ_β=0 ablation baseline:
delta_beta_grid_ablation = [
    # Lower (stronger negative T): keep within |T| < 15
    -(15 - 6.43) / 0.141,    # ≈ -60.78 pm² ⇒ T = -15
    -(10 - 6.43) / 0.141,    # ≈ -25.32 pm² ⇒ T = -10
    -(8 - 6.43) / 0.141,     # ≈ -11.13 pm² ⇒ T = -8
    0.0,                     # baseline T = -6.43
    +(6.43 - 4) / 0.141,     # ≈ +17.23 pm² ⇒ T = -4
    +(6.43 - 1) / 0.141,     # ≈ +38.51 pm² ⇒ T = -1
    +(6.43 - 0.1) / 0.141,   # ≈ +44.89 pm² ⇒ T = -0.1
    # Cap below sign-flip boundary β_K_Cu + Δ_β < 0 ⇒ Δ_β < 45.608
]
# 7 evaluations.
```
Both grids include `Δ_β=0`.  Both have sign-flip bounds applied
to the optimizer.

**5. ACCEPT.**  Mapping-specific grids confirmed (#4 above).

**6. ACCEPT.**  Pre-screen σ bound from a **full V-resolved Δ_β=0
scan** (not just V_kin).  Two purposes:
* (a) Compute `σ_local_max_over_V = max_V |σ_local_clamped(V)|`
  for the pre-screen bound check.
* (b) Compare the `(V_kin, k_hyd=1e-3, λ=1)` record against A.2
  for byte-equivalence.
The full scan IS the byte-equiv reproduction baseline + the
pre-screen σ profile, in one Δ_β=0 evaluation.

**7. ACCEPT.**  After each successful solve at Δ_β candidate:
verify per-V `|pka_shift_avg| ≤ 15`.  If violated:
* Mark candidate `pka_shift_overflow`.
* Exclude from finite loss (per #8 below).
* Solver did succeed; we just reject the candidate as out-of-
  domain ex post.

**8. ACCEPT.**  Two-array bookkeeping:
* `loss_all[i] = {loss, status}` where status ∈ {`finite_valid`,
  `solve_failed`, `solve_invalid_domain`, `pka_shift_overflow`,
  `sign_guard_violation`}.
* `loss_finite_valid[i] = loss` only where status =
  `finite_valid`.
* Identifiability gate and optimizer use `loss_finite_valid`.
* `loss_all` is traced + saved for the writeup.

**9. ACCEPT.**  V grid: constant + unit test are the source of
truth.  No prose count.  Driver constant:
```python
V_RHE_PRODUCTION_GRID: Tuple[float, ...] = (
    -0.10,
    -0.06, -0.01, 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34,
    0.39, 0.44, 0.49, 0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84,
    0.89, 0.94, 0.99, 1.00,
)
# 24 points.
```
Unit test asserts `len == 24`, `−0.10 in grid`, `1.00 in grid`,
endpoints present, no duplicates.

**10. ACCEPT.**  Observable extraction is **masked to the locked
overlap window `[−0.06, +1.0]`**:
* `max_H2O2_selectivity_in_window` only considers V ∈ [−0.06,
  +1.0].
* `ring_onset_V_at_0.01_mA_cm2` only considers V ∈ [−0.06, +1.0]
  (descending sort within mask).
* V_kin = −0.10 is solved and reported per-V but **excluded from
  the observable aggregation**.  V_kin solve is used for sign
  guard, byte-equiv reproduction, and σ_local diagnostics.

**11. ACCEPT.**  Continuation order spelled out:
```
1. Build anchor at V_anchor = +0.55 (two-stage C_S, kw_eff_ladder).
2. From anchor, warm-walk DESCENDING the V_grid:
     anchor (+0.55) → +1.0 (anodic edge, OFF-grid order but ASCENDING from anchor)
     OR
     anchor (+0.55) → +0.54, +0.49, ..., -0.06, -0.10 (DESCENDING from anchor).
```

The natural choice is **bidirectional from anchor**: from +0.55,
walk anodically up to +1.0 first (ascending), THEN walk
cathodically from +0.55 down to -0.10 (descending).  This is the
existing `solve_grid_with_anchor` topology.

Final order in v4: bidirectional bisecting from V_anchor.  The
24-point V grid is traversed in order +0.59, +0.64, ..., +1.0
(11 points ascending from anchor) then +0.54, +0.49, ..., -0.10
(13 points descending from anchor).

**12. ACCEPT.**  V_anchor = +0.55 is **anchor-only**, NOT a grid
point.  Excluded from observable aggregation.  Documented in
driver docstring + unit test.

**13. ACCEPT.**  Circular import risk avoided:
* New module `calibration/singh2016.py` (Firedrake-free)
  holds `SINGH_2016_CATION_PARAMS` (moved from `_bv_common.py`)
  and `compute_beta_per_cation()`.
* `calibration/v10b.py` imports from `calibration.singh2016`
  (one-way).
* `scripts/_bv_common.py` imports from `calibration.singh2016`
  (one-way; no cycle).
* `Forward/bv_solver/cation_hydrolysis.py` imports from
  `calibration.singh2016` for the per-cation table.
* Existing tests verify both `_bv_common.SINGH_2016_CATION_PARAMS`
  alias and `singh2016.SINGH_2016_CATION_PARAMS` are identical
  references (for one-cycle backward compatibility).

**14. ACCEPT.**  Comparison field names pinned to actual JSON
keys (no Unicode):
```python
COMPARISON_FIELDS = [
    "cd_mA_cm2",
    "x_2e",
    "theta",
    "gamma_final",       # verify exact name in A.2 JSON; fallback "gamma"
    "sigma_S_C_per_m2",
    "pka_shift_avg",
    "mass_balance_residual_rel",
    "R_2e_current_nondim",
    "R_4e_current_nondim",
]
# Tolerance: rel <= 1e-3 per field.
```
Phase 10.A.0 verifies these key names against the actual A.2 JSON
before committing; if any key is missing, the byte-equiv test
fails with `SCHEMA_MISMATCH` (not `PHYSICS_DEVIATION`).

**15. ACCEPT.**  Selection criterion for k_hyd=1e-3 record:
```python
target_record = next(
    rec for rec in payload["per_k_hyd_records"]
    if abs(float(rec["k_hyd_target"]) - 1e-3) < 1e-12
)
# Within that record, find lambda_hydrolysis == 1.0:
lam1_diag = next(
    diag for diag in target_record["lam_diagnostics"]
    if abs(float(diag["lambda_hydrolysis"]) - 1.0) < 1e-12
)
# Assert uniqueness:
matches = [rec for rec in payload["per_k_hyd_records"]
           if abs(float(rec["k_hyd_target"]) - 1e-3) < 1e-12]
assert len(matches) == 1, f"Expected unique k_hyd=1e-3 record; got {len(matches)}"
```
Verified in Phase 10.A.0 before the fit runs.

**16. ACCEPT.**  Wall budget table (no more inconsistencies):
| Activity | # Δ_β evals | PDE solves per eval | Total PDE solves | Wall |
|---|---|---|---|---|
| Δ_β=0 baseline | 1 | ~30 (24 V × Newton + anchor) | ~30 | ~9 min |
| Stern pre-fit grid | 8 | ~30 | ~240 | ~72 min |
| Stern Brent | ~10 | ~30 | ~300 | ~90 min |
| Ablation pre-fit grid | 7 | ~30 | ~210 | ~63 min |
| Ablation Brent | ~10 | ~30 | ~300 | ~90 min |
| Final σ-div check at fits | 0 (reuses last evals) | 0 | 0 | 0 |
| **Total** | **36 evals** | **~30/eval** | **~1080** | **~5.4 hours** |

**17. ACCEPT.**  Ablation Brent IS run (committed).  Required for
bundle-locked σ-divergence rel check.  Not "grid-best fallback".

**18. ACCEPT.**  Tolerance set in **target-ΔpKa space**:
```python
xatol_T = 0.05  # ΔpKa-effect units
# At each optimizer step, convert Δ_β → T using the active
# σ scale (σ_local_max_over_V for Stern; override_sigma=0.141 for
# ablation):
def tol_beta_for(sigma_scale):
    return xatol_T / abs(sigma_scale)
xatol_beta_stern = xatol_T / sigma_local_max_over_V  # ~5e5 pm² at σ=1e-7
xatol_beta_ablation = xatol_T / 0.141                # ~0.35 pm² at σ=0.141
```
The `minimize_scalar(method="bounded")` xatol is in Δ_β space, so
the per-mapping conversion is essential.  Same logic in identifi-
ability slope estimates.

---

## Section 2 — Plan v4 patches (against v3)

* **P36**: §1.5 — T-target grid (Stern) listed verbatim with 8
  explicit Δ_β values; `Δ_β = 0` is the baseline anchor.
* **P37**: §1.5 — Ablation target grid separately constructed
  (7 values; baseline T = −6.43 explicit).
* **P38**: D4 optimizer: closed bracket
  `[−4.673e7, −β_K_Cu − eps_beta]`; eps_beta from ΔpKa space.
* **P39**: D6 pre-screen σ bound from full Δ_β=0 V-scan (not just
  V_kin); same scan doubles as byte-equiv baseline.
* **P40**: D5/D6 two-array loss bookkeeping (`loss_all` +
  `loss_finite_valid`).
* **P41**: D1 V grid constant — explicit tuple, 24 points,
  V_kin = −0.10 first, V = +1.00 last; unit test verifies.
* **P42**: D1 observable extraction masked to overlap
  [−0.06, +1.0]; V_kin used only for sign / baseline.
* **P43**: D1 continuation order: anchor at +0.55 (anchor-only,
  not on grid), bidirectional from anchor across the 24-point
  V grid.
* **P44**: New module `calibration/singh2016.py` (Firedrake-free)
  holds Singh params + `compute_beta_per_cation`.  `_bv_common.py`
  and `cation_hydrolysis.py` import from here.
* **P45**: D5 byte-equiv comparison fields pinned to actual JSON
  schema (no Unicode); `lam_diagnostics` lookup with
  `lambda_hydrolysis == 1.0` uniqueness assertion.
* **P46**: §4 wall-budget table replaces all prior cost prose.
* **P47**: D5 ablation Brent committed (not grid-best fallback).
* **P48**: D4 optimizer tolerance set in target-ΔpKa space;
  per-mapping xatol conversion.
* **P49**: Risk register updates: R21 (circular import via
  `calibration/v10b.py`), R22 (V_kin in observable mask
  accidentally), R23 (cross-grid σ scale used in wrong σ for
  conversion).

---

## Section 3 — Continued critique prompt

Review v4 patches.  Push back on poor defenses.  Raise new
issues.  Verdict at end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
