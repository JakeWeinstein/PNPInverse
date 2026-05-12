# R7 → GPT: counterreply + plan v7 patches (final round in cap)

All 4 round-6 issues accepted.  Confirmed cation keys in
`scripts/_bv_common.py:913-933` are charged-form (`"Li+"`, `"Na+"`,
`"K+"`, `"Rb+"`, `"Cs+"`).

---

## Section 1 — Per-issue acknowledgments

**1. ACCEPT.**  Phase D's Δ_β=0 baseline is split into TWO
distinct runs:
* **(a) A.2-compatible reproduction (HARD gate)**: uses the
  EXACT A.2 warm grid `(+0.55, +0.40, +0.20, +0.10, -0.10)` — 5
  points — to reach V_kin.  At V_kin, runs the λ ramp via
  `_run_k_hyd_ramp` with k_hyd=1e-3.  The λ=1 rung at V_kin must
  match the A.2 v10b record (rel < 1e-3) on all 9 comparison
  fields.  This is the ONLY HARD-stop check.
* **(b) Phase D production-grid baseline (DIAGNOSTIC)**: uses the
  24-point V grid + per-V λ ramp topology (the production path).
  Outputs σ_local_clamped profile across V, used to compute
  `sigma_local_clamped_max_over_grid` for grid construction and
  pre-screen bounds.  Mismatch vs (a) at V_kin is logged as
  `continuation_path_drift = {rel_diff_per_field}` but NOT a STOP.

Phase 10.A.0 (data audit) → 10.A.1 (`beta_offset_pm2_func` impl) →
10.A.2 (driver build) → 10.B.0 (a) → 10.B.1 (b) → 10.B.2-3 (Stern
+ ablation fits in parallel).

**2. ACCEPT.**  Hard-stop ONLY on (a) deviation.  (b) deviation
is diagnostic — recorded in the result JSON's
`continuation_path_drift` field for the writeup, no STOP.

**3. ACCEPT.**  `set_reaction_beta_offset_pm2_model()` updated to
mirror metadata everywhere the runtime state is reflected:
```python
def set_reaction_beta_offset_pm2_model(ctx: Dict[str, Any], val: float) -> None:
    bundle = ctx["cation_hydrolysis"]
    bundle.beta_offset_pm2_func.assign(float(val))
    # Mirror into convergence config metadata so result JSONs report
    # the actual value used:
    cfg = ctx.setdefault("bv_convergence", {})
    cation_cfg = cfg.setdefault("cation_hydrolysis_config", {})
    cation_cfg["beta_offset_pm2"] = float(val)
    # Mirror into the diagnostic-emitting bundle params if present:
    if hasattr(bundle, "params") and isinstance(bundle.params, dict):
        bundle.params["beta_offset_pm2"] = float(val)
```
New fast test:
```python
def test_set_beta_offset_pm2_mirrors_ctx_and_diagnostics():
    ctx, bundle = make_test_ctx()
    set_reaction_beta_offset_pm2_model(ctx, 1.5e6)
    assert float(bundle.beta_offset_pm2_func) == 1.5e6
    assert ctx["bv_convergence"]["cation_hydrolysis_config"]["beta_offset_pm2"] == 1.5e6
    # Diagnostic emission picks up live value:
    diag = collect_v10a_rung_diagnostics(ctx, ...)
    assert diag["beta_offset_pm2"] == 1.5e6
```

**4. ACCEPT.**  Cation key convention pinned to **charged form
(`"K+"`, `"Cs+"`, etc.)** — verified against
`scripts/_bv_common.py:913-933`:
* `SINGH_2016_CATION_PARAMS` keys: `"Li+"`, `"Na+"`, `"K+"`,
  `"Rb+"`, `"Cs+"`.
* `compute_beta_per_cation()` signature: `def compute_beta_per_cation(
  cation: str, r_H_El_pm: Optional[float] = None) -> float` with
  cation key validated against `SINGH_2016_CATION_PARAMS.keys()`.
  Invalid key → `ValueError` with the canonical-keys list.
* All docstrings + tests + plan text use `"K+"` (not `"K"`).
* No alias support — explicit canonical-only.  Reduces footgun.

---

## Section 2 — Plan v7 patches (against v6)

* **P68**: D5 split into Δ_β=0(a) HARD A.2-compatible reproduction
  + (b) DIAGNOSTIC full-grid production baseline.
* **P69**: D5 STOP rule scoped to (a) only.
* **P70**: D1 `set_reaction_beta_offset_pm2_model()` updates ctx
  metadata + bundle params mirror; new test
  `test_set_beta_offset_pm2_mirrors_ctx_and_diagnostics`.
* **P71**: Plan-wide cation references use canonical charged-form
  keys (`"K+"`, `"Cs+"`, etc.).  `compute_beta_per_cation`
  validates against `SINGH_2016_CATION_PARAMS.keys()` with
  ValueError on unknown.  No alias support.
* **P72**: Wall budget — (a) reproduction adds ~22 min (single
  A.2 driver run); total Phase D wall ~15.5 hours (negligible
  delta from v6's ~15 hour estimate).

---

## Section 3 — Continued critique prompt

This is round 7 (final round in the 7-cap).  Review v7.  If new
blocking issues remain, raise them; otherwise verdict APPROVED.

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
