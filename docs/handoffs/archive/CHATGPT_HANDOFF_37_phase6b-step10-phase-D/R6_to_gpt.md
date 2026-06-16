# R6 → GPT: counterreply + plan v6 patches

All 8 round-5 issues accepted.  Verified the topology against
actual code:
* `phase6b_v10a_phase_A2_v_kin.py:1264`:
  `_build_sp(lambda_hydrolysis=0.0, …)` — template at λ=0.
* `:1276`: `_walk_lambda_zero_capture_snapshots(...)` — warm-walk
  across V grid AT λ=0, capturing U snapshots.
* `:1316`: `_run_k_hyd_ramp(...)` calls
  `solve_lambda_ramp_from_warm_start` at V_kin from the λ=0
  snapshot — that's the per-V λ ramp.

So the validated topology is **anchor at λ=0 → V-grid warm-walk
at λ=0 → per-V λ ramp 0→1**.  v5's "λ ladder at anchor + V
warm-walk at λ=1" was an invented topology, not the validated
one.

---

## Section 1 — Per-issue acknowledgments

**1. ACCEPT.**  Committing to the **validated topology**:
* Anchor at V_anchor = +0.55 V with λ=0 (kw_eff + C_S sub-ladders
  only; NO λ ladder during anchor build).
* Warm-walk across the 24-point V grid at λ=0.  Each V gets a
  single Newton resolve at λ=0.  Captures U snapshots at every V.
* Per-V λ ramp: at EACH V in the grid, run
  `solve_lambda_ramp_from_warm_start` with the 5-rung λ ladder
  `(0.0, 0.25, 0.50, 0.75, 1.0)` from the V's λ=0 snapshot to
  λ=1.  Each V gets ~5 Newton resolves.

Per Δ_β evaluation:
* Anchor build: ~5-10 Newton-equivalents (kw + C_S ladders).
* λ=0 V-warm-walk: ~24 Newton resolves (one per V).
* Per-V λ ramps: 24 × 5 = ~120 Newton resolves.
* **Total: ~150-155 PDE solves per Δ_β eval.**

**2. ACCEPT.**  Wall budget recomputed for validated topology:

| Activity | Δ_β evals | PDE solves/eval | Total PDE | Wall @ 10s/Newton |
|---|---|---|---|---|
| Stern Δ_β=0 baseline | 1 | ~155 | ~155 | ~26 min |
| Stern pre-fit (7 additional) | 7 | ~155 | ~1085 | ~3.0 hr |
| Stern Brent (~10 evals) | 10 | ~155 | ~1550 | ~4.3 hr |
| Ablation Δ_β=0 baseline | 1 | ~155 | ~155 | ~26 min |
| Ablation pre-fit (6 additional) | 6 | ~155 | ~930 | ~2.6 hr |
| Ablation Brent (~10 evals) | 10 | ~155 | ~1550 | ~4.3 hr |
| **Total** | **35** | **~155** | **~5425** | **~15 hours** |

Single-Newton wall is ~5-15s in v10b records; using 10s as the
central estimate gives ~15 hours.  Phase D wall is **~12-20
hours** depending on per-V convergence quality.  This is an
**overnight execution** scope.

**3. ACCEPT.**  Δ_β=0 byte-equivalence comparison topology
matched:
* Phase D's Δ_β=0 baseline run uses the **same** anchor +
  warm-walk + per-V λ ramp topology as A.2.
* At V_kin in the V grid, the λ=1 state should match A.2's
  V_kin λ=1 record byte-for-byte (rel < 1e-3) — because BOTH
  are produced by the same residual + same topology.
* Phase D's V grid contains V_kin = −0.10 V; the per-V λ ramp
  at V_kin produces a directly-comparable λ=1 state.

**4. ACCEPT.**  Ablation Δ_β=0 is its OWN explicit evaluation:
```
Stern path baseline:    1 Δ_β=0 run with Stern σ mapping
Ablation path baseline: 1 Δ_β=0 run with override_sigma=0.141 ablation
```
Budget table above includes both.  Total baseline cost is ~52 min
(2 evals × 26 min each).

**5. ACCEPT.**  Pre-fit counts pinned:
* Stern: 7 additional + 1 Δ_β=0 baseline = **8 total**.
* Ablation: 6 additional + 1 Δ_β=0 baseline = **7 total**.

**6. ACCEPT.**  Stern Δ_β grid computed AFTER the Δ_β=0 V scan:
```python
# Step 1: run Stern Δ_β=0 baseline; emit per-V σ_local_clamped.
# Step 2: compute sigma_local_clamped_max_over_grid from that scan.
sigma_max = max(abs(σ_local_clamped(V)) for V in V_grid)

# Step 3: build Stern Δ_β grid in target-ΔpKa space:
def stern_delta_beta_for_target(T):
    return (T - beta_K_Cu * sigma_max) / sigma_max

T_targets = [-5.0, -3.0, -1.0, -0.1, -0.01, -0.001, -1e-4]
delta_beta_grid_stern_additional = [
    stern_delta_beta_for_target(T) for T in T_targets
]
```
Grid labels NOT hard-coded; computed from the actual σ profile
of the v10b model at Δ_β=0.

**7. ACCEPT.**  One wall estimate: **~15 hours total**, with
optimistic-conservative range **12-20 hours**.  Documented in
plan §4 wall budget table (single source of truth).

**8. ACCEPT.**  `beta_offset_pm2_func` is **exclusively
`fd.Function(R_space)`** (not `fd.Constant`).  Rationale: R_space
matches the existing `r_H_El_pm_func` pattern in the cation
hydrolysis bundle; runtime setter uses `.assign(float(val))`;
diagnostic emission reads via `float(beta_offset_pm2_func)`.  Tests
cover only the Function path.

---

## Section 2 — Plan v6 patches (against v5)

* **P60**: §4 topology pinned to validated anchor-at-λ=0 +
  V-warm-walk-at-λ=0 + per-V λ ramp pattern.
* **P61**: §4 wall budget table — single estimate ~15 hours;
  range 12-20 hours; overnight execution.
* **P62**: D5 byte-equivalence topology matched to A.2's anchor +
  warm-walk + per-V λ ramp pattern.
* **P63**: D5 ablation has its own Δ_β=0 baseline (1 ablation
  eval), not injected from Stern.
* **P64**: Pre-fit counts: Stern = 7+1 = 8; Ablation = 6+1 = 7.
* **P65**: §3.3 Stern Δ_β grid computed AFTER Δ_β=0 baseline,
  using `sigma_local_clamped_max_over_grid` (not V_kin σ).
* **P66**: D1 + D2 — `beta_offset_pm2_func` is exclusively
  `fd.Function(R_space)`; tests cover only this path.
* **P67**: Risk register updates: R24 (per-V λ ramp wall budget
  ~15 hours overnight; underestimate at single-Newton fast
  estimate); R25 (V grid warm-walk at λ=0 may fail at deep
  cathodic V if the v10b model's λ=0 manifold is fragile —
  fallback: per-V grid warm-walk at λ=ε small).

---

## Section 3 — Continued critique prompt

Review v6.  Push back on poor defenses.  Raise new issues.
Verdict at end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
