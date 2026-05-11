# R5 → GPT: counterreply + plan v5 patches

All 5 round-4 issues accepted.  Verified:
* **#1 confirmed**: `Forward/__init__.py` re-exports BV-solver
  symbols (`SolverParams`, `build_context`, `forsolve`, …) at
  module import time → any submodule import pulls Firedrake.
* **`gamma_ss_langmuir`** lives at `cation_hydrolysis.py:638`,
  inside the heavy module.

---

## Section 1 — Per-issue acknowledgments

**1. ACCEPT.**  Move the leaf module OUT of `Forward/` entirely:
* New top-level package at repo root: `calibration/` (no parent
  in `Forward/`).
* `calibration/__init__.py` — empty.
* `calibration/v10b.py` — pure data: V10B numeric constants +
  `V10B_CALIBRATION_METADATA` dict.  No imports from `Forward`,
  `Nondim`, `scripts`, `firedrake`.  Only `from __future__ import
  annotations` and Python stdlib (`typing.Any`, `typing.Dict`).
* Test: `test_v10b_calibration_module_firedrake_free` confirms
  `sys.modules` delta after importing `calibration.v10b` contains
  zero `firedrake*` entries.
* `Forward/bv_solver/cation_hydrolysis.py` imports
  `from calibration.v10b import (GAMMA_MAX_HAT_V10B,
  K_DES_NONDIM_V10B, V10B_CALIBRATION_METADATA)`.
* `scripts/_bv_common.py` imports the same.
* Path: ensure `calibration` is importable from the repo root
  (it sits at the same level as `Forward/`, `scripts/`, `tests/`).

**2. ACCEPT.**  Analytic Γ_ss audit uses per-rung state, not
baseline state.  For each of the 30 (or N) solver rungs:
* Solver emits `F_avg`, `c_H_avg`, `k_prot`, `delta_ohp_hat`,
  `k_des`, `gamma_max`, and `Γ_solver` (the converged coverage)
  in the per-rung diagnostic record.
* Analytic step calls
  `gamma_ss_langmuir(lam=1.0, k_hyd=k_hyd_rung, k_prot=k_prot,
  k_des=k_des_rung, delta=delta_ohp_hat, F_avg=F_avg,
  c_H_avg=c_H_avg, gamma_max=gamma_max_rung)` → `Γ_analytic`.
* Mismatch metric: `abs(Γ_solver − Γ_analytic) / max(Γ_solver,
  Γ_analytic, 1e-12)`.  Threshold rel ≤ 1e-3 (Phase A.2 v10a
  baseline showed machine precision; 1e-3 is the conservative
  "everything is healthy" gate; 1e-6 would be ideal but margin
  protects against minor diagnostic-emission rounding).
* Per-rung mismatch logged; >1e-3 at any rung → soft flag in the
  writeup, NOT escalation (consistent with hard/soft split).

**3. ACCEPT.**  Subsumed by #2: the audit calls the existing
`gamma_ss_langmuir(...)` helper directly (imported in the driver,
which already pulls Firedrake — no extra cost).  No hand-rolled
formula.

**4. ACCEPT.**  `test_v10b_production_drivers_use_V10B_kinetics`
becomes AST-aware:
* Use `ast.parse` on each driver file in the v10b-production
  whitelist (A.2 driver, step 6 driver, v-sweep driver when
  used for v10b, new bracket / matrix drivers).
* Walk for `ast.ImportFrom` and `ast.Import` nodes; assert each
  imports `V10B_KINETICS` (or `V10B_CALIBRATION_METADATA`), not
  `SMOKE_KINETICS`.
* Whitelist the alias-definition line in the v-sweep file
  (`SMOKE_KINETICS = SMOKE_KINETICS_V10A`) by detecting it as a
  module-level `ast.Assign` with the V10A RHS — that's the
  preserved historical alias, not a v10b-production import.
* The historical / test path (v9 / v10a gate-3 / gate-4 scripts,
  langmuir cap tests) is OUT OF the whitelist — those legitimately
  use `SMOKE_KINETICS_V10A` (frozen historical).

**5. ACCEPT.**  Deprecation comment now ASCII-only:
```python
# DEPRECATED 2026-05-10: GAMMA_MAX_HAT_SMOKE is preserved as
# an alias to GAMMA_MAX_HAT_V10A_SMOKE (frozen historical 0.047)
# for one-cycle backward compatibility with v9/v10a callers.
# NEW callers MUST use GAMMA_MAX_HAT_V10B (production) or
# GAMMA_MAX_HAT_V10A_SMOKE (explicit historical reproduction).
# NEVER alias SMOKE to a V10B value -- that is silent provenance
# theft.  See plan v10b D3 (this commit).  Removal scheduled
# post-step-9 (B.2) after grep zero-caller audit.
GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10A_SMOKE
```

---

## Section 2 — Plan v5 patches (against v4)

* **P30**: Relocate metadata module to top-level `calibration/v10b.py`
  (not under `Forward/`).  Test confirms firedrake-free.
* **P31**: D7-D4 analytic audit uses per-rung state via
  `gamma_ss_langmuir` helper; mismatch threshold rel ≤ 1e-3
  (soft flag, no escalation).
* **P32**: `test_v10b_production_drivers_use_V10B_kinetics` is
  AST-aware; whitelists the v-sweep alias-definition line.
* **P33**: Deprecation comment is ASCII only.  No emoji.

---

## Section 3 — Continued critique prompt

Review the v5 patches.  Push back if any are inadequate.  Raise new
issues if v5 introduces them.  Verdict line at end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
