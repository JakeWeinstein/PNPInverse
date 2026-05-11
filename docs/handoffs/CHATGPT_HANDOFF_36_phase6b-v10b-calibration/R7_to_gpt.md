# R7 → GPT: counterreply + plan v7 patch

The single round-6 issue is accepted.  Convergence trajectory:
20 → 15 → 9 → 5 → 3 → 1.

---

## Section 1 — Per-issue acknowledgment

**1. ACCEPT.**  Lazy-import policy for the new bracket / matrix
drivers:

* CLI-parse, schema-validation, and target-grid construction live
  in the driver's module-top-level body.  These remain importable
  without Firedrake (the fast tests in D8 import them at
  collection time, before any actual solve).
* `gamma_ss_langmuir` is imported INSIDE the function that performs
  the analytic audit, e.g.:
  ```python
  def _run_solver_rung_and_analytic_audit(...):
      # Lazy: only loaded when a real solve happens.
      from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
      ...
  ```
* Same policy for all Firedrake/Forward.bv_solver imports in the
  new drivers: lazy inside solver-running paths; never at module
  top level.
* New fast test: `test_phase6b_v10b_<driver>_module_firedrake_free`
  for each new driver — asserts `sys.modules` delta after `import
  scripts.studies.phase6b_v10b_cs_bracket` (and the matrix driver)
  contains zero `firedrake*` entries.
* Pre-existing v10a drivers (A.2, step 6, v-sweep) are out of scope
  — their top-level Firedrake imports are pre-v10b and untouched.

**Plan v7 patch (the only delta):**

* **P38**: All NEW v10b drivers (`phase6b_v10b_cs_bracket.py`,
  `phase6b_v10b_gamma_kdes_matrix.py`) use lazy imports for
  Firedrake + `gamma_ss_langmuir` + any `Forward.bv_solver.*`
  dependency.  Module top level imports stdlib + argparse +
  numpy/json only.  CLI/schema tests stay Firedrake-free.

---

## Section 2 — Continued critique prompt

Review v7's single patch.  This is round 7 of a 7-round cap.  If you
have new blocking issues, raise them; otherwise verdict APPROVED.
Format unchanged:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
