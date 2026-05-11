# R3 → GPT: counterreply + plan v3 delta

Round 2 was clean — all 15 issues are valid.  Code verified for
the issues that name specific files:
* **#1 confirmed**: `_extract_R_net_nondim` at
  `phase6b_v10a_v_sweep_diagnostic.py:1202-1213` returns `k_des · Γ`,
  positive by construction.  My R_net-sign-cathodic gate was wrong.
* **#2 recomputed**: at 298 K, `k_BT = 0.0257 eV`,
  `k_BT/h = 6.21e12 /s`.
  * ΔG = 0.8 eV → exp(−31.13) ≈ 3.0e-14 → k_phys ≈ 0.19 /s →
    nondim ≈ **0.93**.  (Smoke value 1.0 nondim ↔ ≈ 0.8 eV.)
  * ΔG = 0.9 eV → nondim ≈ **0.020**.
  * ΔG = 0.7 eV → nondim ≈ **46**.
  * ΔG = 1.0 eV → nondim ≈ **4e-4**.
  Decade-spacing of nondim corresponds to ≈ 0.06 eV in ΔG.

**Accepting all 15.**  Below: per-issue response (terse — codex has
v2 memory), then v3 patches.  No full rewrite — only deltas.

---

## Section 1 — Per-issue acknowledgments

**1. ACCEPT.**  R_net is positive (= k_des · Γ ≥ 0).  Cathodic/anodic
sign checks belong to `cd_mA_cm²` (and to R_4e for branch
diagnostics).  D7-D1 and D7-D4 sign-flip gates now use:
* `cd_mA_cm² < 0` at V_kin (cathodic) — flip → escalate.
* For R_4e: magnitude-floor `|R_4e| > 1e-6` AND sign convention
  (R_4e cathodic at V_kin).
* R_net is monotonicity-checked only (positive scalar; no sign).

**2. ACCEPT.**  Eyring table corrected:
| ΔG_des | k_phys [1/s] | k_des_nondim |
|---|---|---|
| 0.7 eV | 9.2 | 46 |
| 0.8 eV | 0.19 | 0.93 |
| 0.9 eV | 0.0040 | 0.020 |
| 1.0 eV | 8.0e-5 | 4.0e-4 |
Smoke 1.0 nondim ≈ ΔG_des ≈ 0.80 eV.  Engineering prior
narrowed to `k_des_nondim ∈ [0.01, 100]` matching
`ΔG_des ∈ [0.69, 0.94] eV` — a defensible adsorbate-desorption
barrier range for OH-like species on sp²-carbon (cf. Nørskov-
Viswanathan 2012 Sabatier framework).

**3. ACCEPT.**  Bracket extended to cover the prior:
* D7-D3 (k_des only): `k_des ∈ {0.01, 0.1, 1.0, 10.0, 100.0}` — 5
  rungs × 2 k_hyd values = 10 rungs.
* D7-D4 (coupled Γ_max × k_des): keeps central 3 `k_des ∈ {0.1,
  1.0, 10.0}` × 3 Γ_max × 2 k_hyd = 18 rungs.  (Wider matrix
  prohibitive at 5×3×2 = 30 rungs; the 1D k_des sweep covers
  endpoints; coupled matrix tests interactions in the central
  region.)

**4. ACCEPT.**  Metadata schema gets `units` field:
```python
"value": float,
"units": "nondim" | "F/m^2" | "1/s" | …,
"is_nondim": bool,
```
For `C_S`: `"value": 0.20, "units": "F/m^2", "is_nondim": False`.
For `Γ_max`, `k_des`: nondim.

**5. ACCEPT.**  New test
`test_v10b_constants_solver_driver_metadata_consistency`
asserts:
* `cation_hydrolysis.K_DES_NONDIM_V10B ==
  v_sweep_diagnostic.V10B_KINETICS["k_des_nondim"]`
* `_bv_common.GAMMA_MAX_HAT_V10B ==
  cation_hydrolysis.GAMMA_MAX_HAT_V10B`
* `V10B_CALIBRATION_METADATA["k_des"]["value"] ==
  K_DES_NONDIM_V10B`
* `V10B_CALIBRATION_METADATA["gamma_max"]["value"] ==
  GAMMA_MAX_HAT_V10B`

**6. ACCEPT.**  You're right — banning ALL aliases overcorrects.
The forbidden alias is `SMOKE = V10B` (silent provenance theft).
A deprecation alias `SMOKE = frozen V10A` is fine.  Plan v3:
* Keep `GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10A_SMOKE` in
  `cation_hydrolysis.py` and `_bv_common.py` with comment
  `# DEPRECATED 2026-05-10: use GAMMA_MAX_HAT_V10A_SMOKE
  (frozen historical) or GAMMA_MAX_HAT_V10B (current production).`
  Removal scheduled post-step-9 (B.2) after grep zero-caller
  audit.
* Same for `K_DES_NONDIM_V10A_SMOKE` (formerly never named).
* No alias points at V10B values.

**7. ACCEPT.**  Same pattern: `SMOKE_KINETICS = SMOKE_KINETICS_V10A`
deprecation alias in `phase6b_v10a_v_sweep_diagnostic.py` so
existing importers don't break atomically.  Phase B step 7
**removes** the dual-mode CLI (per issue #8) and instead keeps
the deprecation aliases for one cycle.

**8. ACCEPT.**  Dual-mode CLI dropped — was an inconsistent half-
measure.  Replaced with:
* Drivers run V10B by default (the production behavior).
* JSON keys are always `"v10b_kinetics"` (no `"v10a_smoke_kinetics"`
  branch).
* Historical reproduction (if ever needed) is achieved by passing
  the V10A constants explicitly via the existing `--gamma-max`,
  `--k-des`, `--c-s` flags — no extra CLI mode.
* Deprecation aliases (per #6/#7) cover the silent-import path.

**9. ACCEPT.**  `_convergence_audit` refactor lands in Phase v10b.C
step 0 (before re-run):
* `hard_gates`: convergence count, mass-balance residual,
  V_kin σ_S sign, K0_R4e_factor branch-pass, Picard-OK.
* `soft_deltas`: cd_mA_cm², x_2e, θ-shape, selectivity_gap —
  numeric reporting with delta vs v10a' baseline, NO pass/fail.
* `overall_pass` driven only by `hard_gates`.
* Existing v10a' / step 6 baseline JSONs remain valid; the audit
  consumer reads the same fields but classifies them differently.
* New fast test
  `test_convergence_audit_hard_soft_separation` verifies the
  classification.

**10. ACCEPT.**  σ_S monotonicity is a sensitivity diagnostic, not
a pass/fail gate.  Plan v3 D7-D1:
* **Hard gates**: 4/4 convergence; cd sign cathodic; no R_4e
  sign-flip with magnitude > floor.
* **Soft diagnostics (logged not gated)**: σ_S trend across C_S
  (expected `|σ_S|` increasing in C_S; report deviations);
  R_net smoothness (no jumps > 50% between adjacent rungs is
  expected but not gated).
* Escalation requires hard-gate failure.

**11. ACCEPT.**  D7-D4 rewritten to use the two-stage anchor
pattern at EVERY rung: build anchor at `STERN_F_M2_ANCHOR = 0.10`,
runtime-bump to `STERN_F_M2_BASELINE = 0.20` per the v10a' /
Phase A.2 pattern.  "Anchored at C_S = 0.20" wording removed.

**12. ACCEPT.**  R_net sign-stability now uses `|R_net| > floor`
(not |R_4e|).  Suggested floor: `|R_net| > 1e-9` nondim — below
that we're in the proton-source noise regime.  R_4e magnitude
floor `|R_4e| > 1e-6` applies ONLY to R_4e sign-flip checks
(branch diagnostics).

**13. ACCEPT.**  Singh-pKa identity reworded in §3.3 note:
> Singh's `pKa_eff` controls the hydrolysis/protonation equilibrium
> *as expressed in the model's specific dimensional identity* —
> incorporating `c_H` boundary concentration, `δ_OHP_hat`, the
> Γ-normalization by `C_SCALE · L_REF`, and the nondim time scale
> `τ_REF`.  A bare ratio `K_eq = k_hyd / k_prot` is only a
> shorthand; any v10b consistency audit must derive the full
> identity from the residual equation in
> `Forward/bv_solver/cation_hydrolysis.py` (the `R_hyd` /
> `R_back` definitions at lines ~643-737).  This audit is OUT OF
> SCOPE for v10b and lives as an open ask in writeup §6.

**14. ACCEPT.**  Drop D7-D2 entirely.  D7-D4 is the single Γ_max
sweep (always required, coupled with k_des).  The Γ_max-only 1D
sweep is redundant given D7-D4 already varies Γ_max across 3
points × 3 k_des × 2 k_hyd.

**15. ACCEPT.**  Metadata moved out of `cation_hydrolysis.py`
(which imports Firedrake) to a new lightweight module:
* New file `Forward/bv_solver/v10b_calibration.py` — pure data
  module, no Firedrake imports.  Exports `V10B_CALIBRATION_METADATA`
  + V10B numeric constants (`GAMMA_MAX_HAT_V10B`,
  `K_DES_NONDIM_V10B`, `C_S_F_M2_V10B = 0.20`).
* `cation_hydrolysis.py` imports the constants from
  `v10b_calibration.py` (one-way; no circular).
* `_bv_common.py` also imports them.
* Result: scripts/tests that only need provenance metadata don't
  pull Firedrake.

---

## Section 2 — Plan v3 deltas (patches against v2)

### Patch P1 — §3.0 (nondim mapping)

Replace the k_des paragraph with corrected Eyring table:

```
k_des_nondim = k_des_phys · τ_REF, τ_REF = L_REF²/D_REF ≈ 5 s.
Smoke 1.0 nondim ↔ 0.2 /s ↔ ΔG_des ≈ 0.80 eV (Eyring at 298 K).
Decade in nondim ↔ ≈ 0.06 eV in ΔG_des.  Diffusion-limited upper
bound `D_K+/δ_OHP² ≈ 4e10` nondim is effectively instantaneous;
kinetics are barrier-limited.
```

### Patch P2 — §3.3 (k_des decision)

Replace the "Note" paragraph with the issue-13 reworded version.
Update strategy 3 prior to `k_des_nondim ∈ [0.01, 100]` matching
`ΔG_des ∈ [0.69, 0.94] eV`.

### Patch P3 — D1 (metadata schema)

Add `"units"` and `"is_nondim"` keys.  Specify:
```python
"C_S": {"value": 0.20, "units": "F/m^2", "is_nondim": False, …},
"gamma_max": {"value": 0.047, "units": "nondim", "is_nondim": True, …},
"k_des": {"value": 1.0, "units": "nondim", "is_nondim": True, …},
```

### Patch P4 — D3, D4, D4' (constants + aliases)

Plan v3:
* New module `Forward/bv_solver/v10b_calibration.py` holds:
  * `GAMMA_MAX_HAT_V10B`, `K_DES_NONDIM_V10B`, `C_S_F_M2_V10B`.
  * `V10B_CALIBRATION_METADATA` with the full schema.
* `Forward/bv_solver/cation_hydrolysis.py`:
  * Freezes `GAMMA_MAX_HAT_V10A_SMOKE = 0.047`,
    `K_DES_NONDIM_V10A_SMOKE = 1.0`.
  * Adds deprecation aliases `GAMMA_MAX_HAT_SMOKE =
    GAMMA_MAX_HAT_V10A_SMOKE`, with comment block.
  * Imports `GAMMA_MAX_HAT_V10B`, `K_DES_NONDIM_V10B` from
    `v10b_calibration`.
  * Updates `raw_cfg.get` defaults at lines 293/295 to use V10B.
* `scripts/_bv_common.py`:
  * Mirrors the freeze + alias + import pattern.
  * Factory default switches to `GAMMA_MAX_HAT_V10B`.
* `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py`:
  * Freezes `SMOKE_KINETICS_V10A = {…}`.
  * Adds `V10B_KINETICS = {…}`.
  * Deprecation alias `SMOKE_KINETICS = SMOKE_KINETICS_V10A`.
  * Factory signature defaults switch to V10B.
  * Driver reads V10B by default; no `--use-v10a-smoke` flag.

### Patch P5 — D5 split (audit refactor)

Plan v3 D5 explicitly requires:
* Phase v10b.C step 0: refactor `_convergence_audit` in
  `phase6b_v10a_phase_A2_v_kin.py` into `hard_gates` +
  `soft_deltas` classification (drop in `_baseline_reproduction_audit`
  in step 6 driver similarly).
* `overall_pass` driven only by `hard_gates`.
* `soft_deltas` logged with absolute and relative deltas vs the
  committed baseline JSON.
* New fast test `test_convergence_audit_hard_soft_separation`.

### Patch P6 — D7-D1 (C_S bracket)

Replace strict σ_S monotonicity:
* **Hard gates**: 4/4 convergence; cd_mA_cm² < 0 (cathodic) at
  V_kin for each rung; no R_4e sign flip with `|R_4e| > 1e-6`
  between adjacent rungs.
* **Soft diagnostics**: σ_S trend (expected monotonic |σ_S|
  increase in C_S — log deviations); R_net smoothness (no >
  50% jumps between adjacent rungs).

### Patch P7 — D7-D4 (coupled matrix)

* Uses two-stage anchor pattern at EVERY rung (anchor at
  STERN_F_M2_ANCHOR = 0.10 → bump to 0.20 via
  `set_stern_capacitance_model`).
* 18 rungs total.
* Sign-stability gate: `|R_net| > 1e-9` (R_net) and `|R_4e| >
  1e-6` (R_4e); R_net sign is always non-negative.
* DROP D7-D2.  D7-D4 is the single Γ_max sweep.

### Patch P8 — D7-D3 (k_des bracket, widened)

* 5 rungs: `k_des ∈ {0.01, 0.1, 1.0, 10.0, 100.0}` × 2 k_hyd =
  10 total.
* Both k_hyd values: `k_hyd_baseline = 1e-3` AND
  `k_hyd_route = 1e-1`.
* Two-stage anchor at every rung.

### Patch P9 — D8 (tests, expanded)

Add the additional tests called out by issues #5 and #9:
* `test_v10b_constants_solver_driver_metadata_consistency` —
  cross-file value equality.
* `test_convergence_audit_hard_soft_separation` — verifies
  audit refactor.
* `test_v10b_calibration_module_no_firedrake_import` — verifies
  the new lightweight module is FE-free.
* `test_metadata_schema_required_keys` — schema completeness.
* Existing planned tests (CLI parse, target grid, output
  schema) — unchanged.

### Patch P10 — §3.1 step 2 (C_S code touch)

No change to plan-v2 limited scope (5 files).  Reaffirm: 348
legacy occurrences are post-v10b cleanup, NOT in v10b DoD.

### Patch P11 — Risk register updates

* R11 reworded: "Sign convention or magnitude-floor misapplied"
  → resolved by patches P6, P7, P8 (explicit conventions for
  cd, R_4e, R_net).
* New R15: "Metadata module imports cation_hydrolysis (circular
  dep)" → mitigated by P4 (v10b_calibration.py is leaf).
* New R16: "Deprecation alias `SMOKE = V10A_SMOKE` accidentally
  imported by a future v10b script writer" → mitigated by
  module-level comment + lint-time deprecation warning at
  alias definition site.

---

## Section 3 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates.  Re-issue any earlier
issue you don't think I addressed.  Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
