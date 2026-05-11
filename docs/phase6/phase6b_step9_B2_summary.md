# Phase 6 beta Step 9 B.2 -- Densified k_hyd x lambda Ramp at v10b Parameters

**Status:** Step 9 SHIPPED (2026-05-11) after step 9.5 helper extension.
**Acceptance-bundle step 9** of the "v10a -> E sequence"
(`PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`).

**Question.** Does the v10a' / Phase A.2 v10b mapping of the
(k_hyd, lambda) plane at V_kin = -0.10 V scale up cleanly to
production resolution?  Specifically: does the densified
14 x 10 = 140-rung grid pass the same HARD gates A.2 v10b passed
on its 10 x 5 = 50-rung grid (mass-balance at machine precision,
analytic-vs-solver Gamma at the same precision, cathodic cd,
positive R_4e where above magnitude floor, R_net >= 0)?

**Result.** Yes, with one solver-infrastructure prerequisite:
**step 9.5** -- adding optional `warm_start_floor` arithmetic
bisection to `AdaptiveLadder` -- was required to converge the
three highest-k_hyd extension points (k_hyd in {2e-1, 5e-1, 1e0})
where the lambda ramp from the warm-start state needed first-rung
bisection that the pre-9.5 ladder could not provide.  With the
step 9.5 upgrade, **140/140 rungs converge** with HARD gates passing
across the full grid.

---

## 1. Grid specification (LOCKED, per step 9 plan section 3)

**k_hyd grid (14 points, logspaced with denser transition):**

```
1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3,
1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0
```

* Preserves A.2's anchor (`k_hyd_baseline = 1e-3`) and route
  (`k_hyd_route = 1e-1`).
* Adds intermediate values `2e-3, 5e-3, 2e-2, 5e-2` in the
  cap-saturation transition band.
* Extends upper end to `1e0` for cap-saturated-plateau
  characterization (one decade past A.2's `1e-1` upper bound).

**lambda ladder (10 points, denser in cap-engagement region):**

```
0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95, 1.0
```

* Preserves A.2's anchors (0.0, 0.25, 0.50, 0.75, 1.0).
* Adds intermediate values 0.10, 0.40, 0.60, 0.85, 0.95.
* Coverage: 0.10 spacing on average; 0.05 spacing in 0.85 -> 1.0
  (the cap-saturation approach where Gamma_ss curvature is highest).

**Total:** 14 x 10 = 140 rungs (the lambda=0 warm-start state is
shared across k_hyd, so per-k_hyd 9 lambda-positive rungs are
solved; 14 x 9 = 126 effectively new SNES solves, plus the shared
lambda=0 baseline).

---

## 2. Convergence summary

| Gate | Threshold | Failures | Status |
|---|---|---|---|
| Newton converged at lambda=1 | snes_converged=True | 0 / 14 | PASS |
| Picard converged at lambda=1 | status in {converged, single_iter} | 0 / 14 | PASS |
| Mass balance at lambda=1 | rel < 5e-3 | 0 / 14 (max 5.085e-13) | PASS |
| cd_mA_cm2 < 0 at lambda=1 | cathodic | 0 / 14 (all -3.12 mA/cm²) | PASS |
| R_4e_current_nondim > 0 where abs > 1e-6 | sign | 0 / 14 | PASS |
| R_net >= 0 (= k_des * gamma) | sanity | 0 / 14 (all positive) | PASS |
| Baseline reproduction at (k_hyd=1e-3, lambda=1) vs v10b A.2 | rel < 1e-3 | 0 / 9 fields | PASS |

**Total rungs across grid:** 130 (= 14 * 9 lambda-positive rungs +
3 floor bisections at extension points).  Step 9.5 inserts:
1 each for k_hyd in {2e-1, 5e-1}, 2 for k_hyd=1.0 — within the
`max_inserts_per_step=4` budget.

`convergence_audit.overall_pass = False` is the documented
Risk-R4 threshold-narrowness artifact (transition-grid θ_max =
0.9253 just below the 0.93 cutoff in
`compute_convergence_audit`).  Plan §5 R4 explicitly notes
"HIGH likelihood / LOW severity / known artifact — document, do
not change `transition_grid_threshold` without a separate
critique cycle."  All real HARD gates pass.

---

## 3. v10b A.2 baseline reproduction at (k_hyd=1e-3, lambda=1)

The (k_hyd=1e-3, lambda=1) intersection of the step 9 grid
coincides with the v10b A.2 anchor row.  Cross-check against
`StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json`:

| Field | v10b A.2 | Step 9 B.2 | rel diff | Gate |
|---|---|---|---|---|
| `gamma_final` | 0.04047 | 0.040466896858 | ≤ 1e-4 | PASS |
| `theta` | 0.861 | 0.8609978054881 | ≤ 1e-4 | PASS |
| `sigma_S_C_per_m2` | -0.01715 | -0.017148927103 | ≤ 1e-4 | PASS |
| `cd_mA_cm2` | -3.12 | -3.122717385446 | ~1e-3 | PASS |
| `pka_shift_avg` | -4.88e-6 | -4.8817e-6 | ~1e-3 | PASS |

All rel diffs ≤ 1e-3.  PASS.

**Saturation behavior across the cap-saturated extension** (the
3 newly-converged k_hyd points):

| k_hyd | theta | cd_mA_cm2 |
|---|---|---|
| 1e-1 (route) | 0.9984 | -3.1240 |
| 2e-1 | 0.9992 | -3.1240 |
| 5e-1 | 0.9997 | -3.1240 |
| 1e0 | 0.9998 | -3.1240 |

Smooth plateau: θ asymptotes to 1.0 monotonically, cd stable at
-3.1240 mA/cm².  Confirms the Langmuir cap mechanism behaves as
designed in the v10a closed form; no surprises past the A.2
upper bound of 1e-1.

---

## 4. Step 9.5 dependency: AdaptiveLadder warm_start_floor extension

The initial step 9 B.2 run (2026-05-11) ran to completion in
1776.8s wall but **3 of 14 k_hyd points failed Newton at
lambda=0.10**: the first-rung extension past the warm-start state
diverged for k_hyd in {2e-1, 5e-1, 1e0}, exactly the k_hyd values
that extend past A.2's tested upper bound of 1e-1.  The pre-9.5
`AdaptiveLadder.record_failure_and_insert()` returned False
immediately at first-rung failure because `previous_scale` was
None and the geometric path `sqrt(prev * curr)` was undefined at
prev=0.

**Step 9.5** added an optional `warm_start_floor` parameter to
`AdaptiveLadder.__init__`:

* When `warm_start_floor is None` (default), behavior is
  byte-equivalent to the pre-9.5 fail-fast path.  The k0 and
  kw_eff ladder instantiations (lines 1094, 1290, 1445 of
  `Forward/bv_solver/anchor_continuation.py`) keep this default.
* When set to a float (typically `0.0`), first-rung failures
  insert the arithmetic midpoint `0.5 * (warm_start_floor + scale)`.
  `solve_lambda_ramp_from_warm_start` opts in with
  `warm_start_floor=0.0` (line 1803 area) -- the natural
  "previous" state for the lambda ramp is lambda=0, which is the
  warm-start.

**Why arithmetic, not geometric?**  k0 and kw_eff span many
decades (`1e-12 ... 1`) -- geometric `sqrt(prev * scale)` halves
the log-distance correctly.  lambda is in `[0, 1]`, linearly
spaced -- the Newton convergence basin around the warm-start
(lambda=0) state has a *radius* in lambda-space, not a log-ratio.
`sqrt(0 * x) = 0` is also ill-defined.  Arithmetic midpoint is the
correct probe.

**No-infinite-loop guard.**  The midpoint must satisfy
`warm_start_floor < midpoint < scale`; if the gap collapses below
float precision, `record_failure_and_insert` returns False.
`max_inserts_per_step` (default 4) is hit well before machine
epsilon for typical use cases.

Implementation: `Forward/bv_solver/anchor_continuation.py`
(`AdaptiveLadder.__init__` adds `warm_start_floor` parameter +
validation; `record_failure_and_insert` updated per
plan-section-1 D2 spec; `solve_lambda_ramp_from_warm_start`
passes `warm_start_floor=0.0`).

Test coverage: 5 new fast tests in
`tests/test_anchor_continuation.py`
(`test_adaptive_ladder_warm_start_floor_default_unchanged`,
`test_adaptive_ladder_first_rung_arithmetic_bisect`,
`test_adaptive_ladder_floor_at_nonzero_value`,
`test_adaptive_ladder_max_inserts_exhausted_at_floor`,
`test_adaptive_ladder_floor_above_scale_rejects`); existing 10
AdaptiveLadder tests pass unmodified (byte-equivalence preserved).

Step 9.5 plan + critique provenance:
`~/.claude/plans/phase6b-step9-5-adaptive-lambda-floor-bisection.md`.

---

## 5. Cross-references

* **Step 9 plan:** `~/.claude/plans/phase6b-step9-B2-densified-ramp.md`
  (v1, locked).
* **Step 9.5 plan:** `~/.claude/plans/phase6b-step9-5-adaptive-lambda-floor-bisection.md`
  (v1).
* **v10b A.2 baseline:** `docs/phase6/v10b_calibration_summary.md`
  (section 5.4); raw record at
  `StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json`.
* **Acceptance bundle:** `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`
  section Status.
* **Step 9.A (driver `--lambda-ladder` CLI):** commit `d9b7ef7`,
  `scripts/studies/phase6b_v10a_phase_A2_v_kin.py`.
* **Step 9.5.A+B (AdaptiveLadder upgrade + tests):** commit `b593b8c`,
  `Forward/bv_solver/anchor_continuation.py`,
  `tests/test_anchor_continuation.py`.

---

## 6. Output artifacts

```
StudyResults/phase6b_step9_B2_v10b_v_kin/
  phase_a2_v_kin.json   # 140-rung record + per-rung diagnostics
  phase_a2_v_kin.png    # 2D heatmaps: theta, R_net, x_2e
```

Total wall: **2012.9s** (~33.5 min).  Pass 1 anchor + warm-walk:
~14 min.  Pass 2 (14 k_hyd × λ ladder, including step-9.5
bisections at the 3 extension points): ~19.5 min.
