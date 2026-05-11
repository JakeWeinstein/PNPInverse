# R2 → GPT: counterreply + plan v2

Thanks — round 1 was substantive.  I cross-checked the codebase to
verify your specific claims before deciding accept vs. defend.
Findings (so you can judge whether I'm being honest):

* **#3 confirmed**: `Forward/bv_solver/cation_hydrolysis.py:293`
  has `k_des_init = float(raw_cfg.get("k_des", 1.0))`, and line 295
  has `gamma_max_init = float(raw_cfg.get("gamma_max_nondim",
  GAMMA_MAX_HAT_SMOKE))`.  Both solver-layer defaults exist.
* **#4 partially confirmed**: `phase6b_step6_plumbing_ablation.py:
  1296` *does* accept `a2_baseline_json` as an argument; I need to
  verify whether the driver's `main()` exposes it via CLI and what
  the default path is.  Treating as accept regardless.
* **#13 confirmed**: a fresh grep for `stern_capacitance_f_m2.*0.10`
  (excluding StudyResults, .git, archive, Inference) returns 348
  lines, not the 12 I claimed.  Most are tests and docs; my §3.1
  scope was wrong.

I'm **accepting all 20 issues**.  Below: per-issue response, then a
delta summary, then the full revised plan (v2).

---

## Section 1 — Per-issue acknowledgments

**1. (k_des detailed-balance route is wrong) — ACCEPT.**
Singh's pKa constrains the equilibrium `M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺`
(hydrolysis ↔ protonation), i.e. `K_eq = k_hyd / k_prot`.  MOH
desorption `M(OH)⁰(ad) → M(OH)⁰(bulk)` is a separate kinetic step
not constrained by Singh.  Strategy 1 in §3.3 removed.  The
Singh-derived constraint now lives in a new audit-only section: if
the v10a `k_prot` value can be checked against Singh's `K_eq`, do
so — it's a `k_prot` consistency check, not a `k_des` anchor.

**2. (no nondim map for k_des) — ACCEPT.**
`τ_REF = L_REF² / D_REF` per `scripts/_bv_common.py:131-134` —
`L_REF = 1e-4 m`, `D_REF = D_O2 ≈ 2e-9 m²/s` → `τ_REF ≈ 5 s`.
So `k_des_nondim = k_des_phys · τ_REF` with `k_des_phys` in `1/s`.
Diffusion-limited upper bound at the OHP: `D_K+/δ_OHP² ≈ 4e10
nondim` — effectively instantaneous, so the kinetics are barrier-
limited.  Eyring: `k_des_phys = (k_BT/h) · exp(−ΔG_des/RT)`,
`k_BT/h ≈ 6e12 /s` at 298 K.  ΔG_des ≈ 0.9 eV → `k_des_phys ≈ 0.2
/s` ≈ 1.0 nondim (i.e. the smoke value implies a moderate-barrier
desorption).  Added §3.3.0 (nondim map) to plan v2.

**3. (solver default `k_des` not updated) — ACCEPT.**
Plan v2 D3 adds a `K_DES_NONDIM_V10B` constant in
`Forward/bv_solver/cation_hydrolysis.py` and updates the
`raw_cfg.get("k_des", 1.0)` default to `raw_cfg.get("k_des",
K_DES_NONDIM_V10B)`.  Symmetric to `GAMMA_MAX_HAT_V10B` handling.
Both go into the module `__all__`.

**4. (step 6 A0 audit baseline path) — ACCEPT.**
Plan v2 D6 requires explicit `--a2-baseline-json` CLI override
pointing at the new v10b A.2 JSON.  If the driver's `main()`
doesn't already expose this CLI flag, Phase v10b.C step 4 adds it.

**5. (D6 audit fields do not actually include R_net) — ACCEPT.**
Plan v2 D6 contract: verify `_baseline_reproduction_audit` keys
during Phase v10b.C step 4.  If `R_net` is not in the audit, add
it (the canonical λ=1 source-term diagnostic) — that's a small
edit to `_baseline_reproduction_audit` plus a regression test.
Adding to D8 fast-test scope.

**6. (`SMOKE_* = V10B_*` aliases destroy provenance) — ACCEPT FULLY.**
The alias plan was wrong.  Plan v2:
* **Freeze** `GAMMA_MAX_HAT_V10A_SMOKE = 0.047`,
  `K_DES_NONDIM_V10A_SMOKE = 1.0`, `SMOKE_KINETICS_V10A = {…}`
  in their *existing* locations.  These are immutable historical
  constants.
* **Introduce** `GAMMA_MAX_HAT_V10B`, `K_DES_NONDIM_V10B`,
  `V10B_KINETICS` with the v10b values.  Symmetric naming.
* **Update v10b production callers + factory defaults** to point
  at the V10B constants.  v9/v10a scripts and tests stay on V10A.
* **No `SMOKE = V10B` aliases anywhere.**  After step 9 (B.2)
  lands, optionally delete `*_SMOKE` from the public surface IF
  grep confirms zero callers — but never alias to v10b.

**7. (alias/export/test plan incomplete) — ACCEPT.**
Plan v2 D3, D4, D8 add explicit `__all__` updates + dedicated
tests:
* `test_cation_hydrolysis_v10a_v10b_constants_coexist` —
  asserts both constants importable, distinct, with the v10b
  value not equal to the v10a value (unless physics says they
  must agree; if it does, the test asserts the equality with
  citation).
* `test_factory_default_uses_v10b` — `make_cation_hydrolysis_config()`
  default `gamma_max_nondim` equals `GAMMA_MAX_HAT_V10B`.

**8. (k_des bracket diagnostic contradicts itself) — ACCEPT.**
Plan v2 D7-D3 runs BOTH `k_hyd_baseline = 1e-3` AND
`k_hyd_route = 1e-1` (cap-saturated) at each `k_des` rung; reports
θ AND R_net per rung.  `R_net = k_des · Γ` at λ=1 is the most
direct k_des sensitivity diagnostic.

**9. (no coupled Γ_max × k_des sensitivity) — ACCEPT.**
Plan v2 D7-D4 adds a coupled 3×3 matrix sweep:
`Γ_max ∈ {Γ_max_lo, Γ_max_v10b, Γ_max_hi}`
× `k_des ∈ {0.1, 1.0, 10.0}` at λ=1, V=V_kin, evaluated at
**both** `k_hyd_baseline = 1e-3` and `k_hyd_route = 1e-1`.
Report θ, R_net, R_2e, R_4e, σ_S per cell.  18 rungs total;
each rung is a Newton resolve from the warm anchor (~30 s wall
each) → ~10 min total.  Tease out ridges and compensating pairs.

**10. (D5 ±20% tolerance can reject valid calibration) — ACCEPT.**
Plan v2 D5 split into:
* **D5.HARD (escalate on failure):** convergence (10/10 at λ=1.0,
  Picard converges, no Newton failures); mass-balance < 5e-3;
  invariants preserved (V_kin σ_S sign is cathodic; the locked
  rule still applies — `σ_S<0`, `cd_ok`, `branch_ok` at V_kin).
* **D5.SOFT (document, no escalation):** cd, x_2e, θ-shape
  numeric deltas vs v10a' — logged in the writeup, no gate.
Escalation triggered ONLY by D5.HARD failures.  Selectivity-gap
movement is data, not a verdict.

**11. (C_S sweep monotonicity brittle) — ACCEPT.**
Plan v2 D7-D1 replaces "no sign flips, σ_S monotonic":
* **R_4e magnitude floor:** sign-flip detection requires
  `|R_4e| > 1e-6 nondim` at both adjacent C_S rungs; below floor,
  no flip claim (numerical noise at `K0_R4e_factor = 1e-14`).
* **Sign convention:** R_net should be cathodic at V_kin
  (negative for cathodic source).  A cathodic→anodic flip
  between adjacent C_S rungs DOES warrant escalation.
* **σ_S monotonicity prediction:** at fixed V_RHE, larger C_S
  shifts more of the drop to the diffuse layer → |σ_S| increases
  monotonically in C_S.  So `σ_S(C_S=0.05) > σ_S(C_S=0.10) >
  σ_S(C_S=0.20) > σ_S(C_S=0.30)` (all cathodic = negative;
  inequalities reverse on signed value).  Violation flags
  numerical/coupled-solve issue.

**12. (R3 contradicts D7) — ACCEPT.**
Plan v2 D7-D1: **4/4 C_S rungs converge is mandatory**.  R3
mitigation rewritten: if any rung fails, ESCALATE to v10c — do
not pass with 3/4.  Document the failing rung's diagnostic in
the writeup.

**13. (legacy C_S audit stale and too broad) — ACCEPT.**
Plan v2 §3.1 step 2 redone: a fresh grep returns 348 lines.
Per-occurrence classification is out of v10b scope (see #14).
v10b's C_S-related code touch is ONLY:
* `Forward/bv_solver/cation_hydrolysis.py` — constants + default.
* `scripts/_bv_common.py` — constants + factory default.
* `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` — already
  uses `STERN_F_M2_BASELINE = 0.20` (no change needed).
* `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` — already
  uses the v10a' two-stage anchor pattern (no change needed).
* `scripts/studies/phase6b_step6_plumbing_ablation.py` — same.
* New `scripts/studies/phase6b_v10b_cs_bracket.py` — uses the
  V10B constants and two-stage anchor.

**14. (legacy script updates in Phase E = scope creep) — ACCEPT.**
Plan v2 removes the legacy-script-audit from Phase E entirely.
Create a separate post-v10b cleanup task tracked as v10b-cleanup
(NOT part of v10b's definition of done; carried as an open ask
in the writeup).

**15. (`engineering_choice = True` flag has no schema) — ACCEPT.**
Plan v2 D1 defines a `v10b_calibration_metadata` block:
```python
V10B_CALIBRATION_METADATA: Dict[str, Dict[str, Any]] = {
    "gamma_max": {
        "value": GAMMA_MAX_HAT_V10B,
        "source_type": "literature" | "engineering" | "literature_chain",
        "engineering_choice": bool,
        "citation": str | None,   # e.g. "Singh 2016 Table S1" or None
        "bracket": list[float],
        "prior": str | None,      # e.g. "Eyring ΔG_des ∈ [0.7, 1.1] eV"
        "compatibility": {
            "mechanism": "MOH at OHP",
            "electrode": "sp²-carbon-cathode" | "metal-cathode-transferred",
            "electrolyte": "aqueous K2SO4",
            "dimensional": "Γ_phys / (C_SCALE · L_REF) verified",
        },
    },
    "k_des": {…same shape…},
    "C_S": {…same shape, sourced from CMK3_capacitance_literature.md…},
}
```
Stored in `Forward/bv_solver/cation_hydrolysis.py` module scope;
emitted in every v10b result JSON; quoted in the writeup §5
(Implementation status).  Public via `__all__`.

**16. ("data-constrained in Phase D" not justified) — ACCEPT.**
Phase D is K-only Δ_β fit per acceptance bundle § "v10a → E
sequence" step 10.  Plan v2 rewords engineering-choice note to
"open parameter for post-v10b; if a future scope expansion fits
`k_des` against per-cation data, this calibration becomes the
initial prior."  Explicitly NOT promised by Phase D as locked.

**17. (Γ_max decision rule too naive) — ACCEPT.**
Plan v2 §3.2 decision rule strengthened to require:
* **Mechanism compatibility:** source must report MOH adsorbate
  coverage or equivalently constrained surface site density.
  Not generic capacitance-derived packing.  Not bare-cation
  Stern occupancy.
* **Electrode compatibility:** sp²-carbon-cathode preferred;
  metal-cathode acceptable with explicit "site-density transfer"
  note (e.g. polycrystalline Au → CMK-3 not automatic).
* **Electrolyte compatibility:** aqueous K⁺-containing; alkali-
  cation sub-tabling acceptable if K⁺ entry present.
* **Dimensional equivalence:** source's Γ must be transportable
  to nondim Γ_max via `Γ_phys / (C_SCALE · L_REF) = Γ_phys /
  (1.2e-4 mol/m²)`.
Falling any of these → cannot lock; either tighten chain
(option B) or engineering choice (option C).

**18. ("Likely outcome" bias) — ACCEPT.**
Plan v2 §3.2 removes the "Likely v10b outcome" paragraph.  The
decision rule decides; no priors stated.

**19. (Result JSON "smoke_kinetics" provenance) — ACCEPT.**
Plan v2 D5 + D6 + D7 require renaming `"smoke_kinetics"` →
`"v10b_kinetics"` in driver result payloads:
* `phase6b_v10a_phase_A2_v_kin.py` line 1600 → `"v10b_kinetics"`
  when run under v10b.  (Driver becomes dual-mode: emit
  `"v10a_smoke_kinetics"` only if explicitly invoked with the
  historical constants via a CLI flag; otherwise `"v10b_kinetics"`.)
* `phase6b_step6_plumbing_ablation.py` line 1659 — same.
* `phase6b_v10a_v_sweep_diagnostic.py` — same.
* New bracket driver writes `"v10b_kinetics"`.

**20. (new bracket driver lacks test requirements) — ACCEPT.**
Plan v2 D8 + D7 add fast-test contract for the new driver:
* `test_phase6b_v10b_cs_bracket_cli_parses` — argparse smoke.
* `test_phase6b_v10b_cs_bracket_target_grid` — confirms the 4
  C_S rungs are scheduled with the two-stage anchor pattern.
* `test_phase6b_v10b_cs_bracket_output_schema` — JSON keys
  match the spec.
* For coupled D7-D4 matrix: `test_phase6b_v10b_gamma_kdes_matrix_*`
  three analogous tests.

---

## Section 2 — Delta summary (v1 → v2)

Major changes (driven by your issues):

* **§3.3 k_des**: detailed-balance strategy removed (was wrong);
  new §3.3.0 (nondim map: `k_des_nondim = k_des_phys · τ_REF`,
  τ_REF ≈ 5 s); strategy list reordered (analog adsorbate-desorption
  → diffusion/Eyring → engineering choice); engineering-choice
  fallback unchanged.
* **§3.2 Γ_max**: "likely outcome" bias removed; decision rule
  tightened with mechanism/electrode/electrolyte/dimensional
  compatibility tests.
* **§3.1 C_S legacy audit**: removed from v10b; deferred to post-
  v10b cleanup task.
* **D1**: new `V10B_CALIBRATION_METADATA` block schema specified.
* **D3, D4**: alias scheme rewritten — freeze V10A, introduce V10B,
  no SMOKE = V10B aliases.
* **D5**: split into HARD (escalate) and SOFT (document) gates.
* **D6**: explicit baseline-path CLI requirement; verify
  `_baseline_reproduction_audit` includes R_net (add if not).
* **D7**: monotonicity criterion replaced with magnitude-floor +
  sign-convention + signed-σ_S-monotonicity-prediction; 4/4
  mandatory; conditional matrix expanded.
* **D7-D3 / D7-D4**: coupled Γ_max × k_des matrix added; k_des
  rungs evaluated at BOTH `k_hyd_baseline` and `k_hyd_route`.
* **D8**: fast-test contract added for new driver + coexistence
  test for v10a / v10b constants.
* **D9, D10, D11**: unchanged.
* **Phase E**: legacy script audit removed entirely (moved to
  post-v10b cleanup).
* **R3, R10, R5**: updated to match new gate semantics.

Three open verification items the executor handles in Phase v10b.C:
1. Verify `_baseline_reproduction_audit` actually has the
   `--a2-baseline-json` CLI flag (or add it).
2. Verify the audit keys include R_net at λ=1 (add if not).
3. Verify Phase A.2 driver's result-JSON `"smoke_kinetics"` key
   path is renameable without breaking the convergence-audit
   consumer.

---

## Section 3 — Plan v2 (full revised text)

````````````````````````markdown
# Phase 6β Step 8 — v10b Literature Calibration of Γ_max + k_des + C_S (v2)

**Author:** Claude (planner).  **Date:** 2026-05-10.
**Status:** v2 — after R1 critique (20 issues accepted).
**Source handoff:** `docs/handoffs/v10b_planning_handoff.md`.

---

## 0. One-paragraph framing

v9/v10a used three smoke values that have not been pinned to peer-
reviewed literature: `Γ_max_hat = 0.047` (1-monolayer K⁺ MOH at the
OHP, hard-sphere derivation), `k_des_nondim = 1.0` (no anchor), and
`C_S = 0.10 F/m²` (a convergence-pinned engineering value).  Step 7
locked `C_S = 0.20 F/m²` per
`docs/phase6/CMK3_capacitance_literature.md`.  v10b delivers
literature-anchored numeric values (or explicit `engineering_choice`
flags with documented priors) for all three parameters, updates the
solver-layer + driver-layer defaults under a clean V10A/V10B
provenance scheme (no aliases), regenerates Phase A.2 + step 6
plumbing baselines at v10b parameters, runs a C_S sensitivity
bracket sweep and a coupled Γ_max × k_des matrix, and publishes
`docs/phase6/v10b_calibration_summary.md`.  Selectivity-gap
improvement is **not** a v10b pass criterion (Phase D's job).

---

## 1. Definition of done

v10b is **done** iff every box ticks:

- [ ] **D1.** Each of `Γ_max`, `k_des`, `C_S` carries a per-parameter
      record in `V10B_CALIBRATION_METADATA` with the schema:
      ```python
      {
        "value": float,                     # nondim
        "source_type": "literature" | "literature_chain" | "engineering",
        "engineering_choice": bool,
        "citation": str | None,
        "bracket": list[float],             # sensitivity-sweep grid
        "prior": str | None,                # required if engineering_choice
        "compatibility": {                  # required for literature/chain
            "mechanism": str,
            "electrode": str,
            "electrolyte": str,
            "dimensional": str,
        },
      }
      ```
      The metadata block lives in
      `Forward/bv_solver/cation_hydrolysis.py` module scope and is
      emitted verbatim in every v10b result JSON.
- [ ] **D2.** `docs/phase6/v10b_calibration_summary.md` published.
      Three-parameter analog of `CMK3_capacitance_literature.md`:
      per-parameter citation chain + caveats + sensitivity bracket
      + implementation status (metadata block quoted) + open asks
      + cross-references.
- [ ] **D3.** Solver-layer constants in
      `Forward/bv_solver/cation_hydrolysis.py`:
      * **Freeze** `GAMMA_MAX_HAT_V10A_SMOKE = 0.047` (renamed
        from `GAMMA_MAX_HAT_SMOKE`).  Add docstring "frozen
        historical".
      * **Add** `GAMMA_MAX_HAT_V10B` with the v10b value (or
        rationale = "kept = V10A; chain tightened").
      * **Add** `K_DES_NONDIM_V10A_SMOKE = 1.0` (frozen).
      * **Add** `K_DES_NONDIM_V10B` with the v10b value.
      * Update `raw_cfg.get("k_des", 1.0)` →
        `raw_cfg.get("k_des", K_DES_NONDIM_V10B)` at line 293.
      * Update `raw_cfg.get("gamma_max_nondim", GAMMA_MAX_HAT_SMOKE)`
        →
        `raw_cfg.get("gamma_max_nondim", GAMMA_MAX_HAT_V10B)` at
        line 295.
      * Add all four constants + `V10B_CALIBRATION_METADATA` to
        `__all__`.
      * **No `SMOKE = V10B` aliases.**  After step 9 (B.2) lands,
        the V10A-frozen constants may be deleted IF grep confirms
        zero callers (separate decision; not part of v10b DoD).
- [ ] **D4.** `scripts/_bv_common.py`:
      * **Freeze** `GAMMA_MAX_HAT_V10A_SMOKE = 0.047` at line 944
        (renamed).
      * **Add** `GAMMA_MAX_HAT_V10B` (mirror of solver-layer
        value; sanity-test enforces equality).
      * `make_cation_hydrolysis_config` default at line 957
        switches from `GAMMA_MAX_HAT_SMOKE` to
        `GAMMA_MAX_HAT_V10B`.
- [ ] **D4'.** `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py`:
      * **Freeze** `SMOKE_KINETICS_V10A = {…}` at line 174
        (renamed from `SMOKE_KINETICS`).
      * **Add** `V10B_KINETICS = {…}` with v10b values + same
        key set.
      * Update factory signature default args from
        `SMOKE_KINETICS["X"]` → `V10B_KINETICS["X"]` at lines
        821-826, 866-871.
      * Driver `main()` reads from `V10B_KINETICS` by default;
        keep a `--use-v10a-smoke` CLI flag for historical
        reproduction.
- [ ] **D5.HARD (escalation gates).**  Phase A.2 driver re-run at
      v10b params with output subdir
      `phase6b_v10b_phase_A2_v_kin`:
      * 10/10 k_hyd rungs converge at λ=1.0.
      * Picard converges everywhere (no `iter_cap_hit_unconverged`).
      * Mass-balance residual < 5e-3 across the grid.
      * V_kin = −0.10 V still satisfies the locked rule: at
        V_kin, `σ_S < 0` ∧ `cd_ok` ∧ `branch_ok` (i.e. the v10a'
        decision-tree precedence guards still hold).
      * `K0_R4e_factor = 1e-14` still produces non-zero R_4e at
        V_kin (`|R_4e| > 1e-9` nondim at λ=1).
      Failure of any HARD gate → ESCALATE to v10c; do not push past.
- [ ] **D5.SOFT (informative deltas, logged not gated).**
      * `cd_mA_cm²` at λ=1: report delta vs v10a' record.
      * `x_2e` at λ=1: report delta vs v10a' record.
      * `θ(k_hyd=1e-1)`: report value and delta vs v10a' record.
      * `single_v_selectivity_gap_pp`: report; movement is data
        for Phase D, not a v10b verdict.
- [ ] **D6.** Step 6 plumbing-ablation driver re-run at v10b
      params with output subdir
      `phase6b_v10b_step6_plumbing_ablation` and explicit
      `--a2-baseline-json
      StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json`.
      All 5 ablations pass.  A0 byte-equivalent (rel ≤ 1e-6 in
      cd, R_net, σ_S, θ) to the new v10b A.2 baseline.  Step 6
      driver: verify (and add if missing) the `--a2-baseline-json`
      CLI flag; verify `_baseline_reproduction_audit` keys
      include `R_net` at λ=1 (add if not).  Both modifications
      land in Phase v10b.C step 4.
- [ ] **D7-D1 (C_S sensitivity bracket; ALWAYS REQUIRED).**
      `C_S ∈ {0.05, 0.10, 0.20, 0.30} F/m²` at V_kin = −0.10 V,
      λ=1, k_hyd = `k_hyd_baseline = 1e-3`.  Pass criteria:
      * **4/4 mandatory.**  If any rung fails to converge,
        ESCALATE to v10c (no 3/4 fallback).
      * Sign convention: R_net cathodic at V_kin (negative).
        cathodic→anodic flip between adjacent C_S rungs → escalate.
      * Magnitude floor on R_4e: sign flips claimed only when
        `|R_4e| > 1e-6` nondim at both adjacent C_S rungs.
      * σ_S monotonicity prediction: `signed_σ_S` strictly
        decreasing (i.e. `|σ_S|` strictly increasing) in `C_S`.
        Violation flags coupled-solve issue → escalate.
- [ ] **D7-D2 (Γ_max sensitivity bracket; CONDITIONAL).**
      Skip if §3.2 locks a single value.  Run if engineering
      choice or "keep V10A value with bracket evidence".
      `Γ_max ∈ {Γ_max/2, Γ_max, Γ_max·2, Γ_max·4}` nondim, anchor
      at C_S = 0.10 + bump to 0.20.  Same convergence + smoothness
      criteria as D7-D1.
- [ ] **D7-D3 (k_des sensitivity bracket; CONDITIONAL).**
      Required if §3.3 lands on engineering choice (likely).
      `k_des ∈ {0.1, 1.0, 10.0}` nondim at V_kin, anchor at C_S =
      0.10 + bump to 0.20.  **Run at both** `k_hyd_baseline =
      1e-3` AND `k_hyd_route = 1e-1` (cap-saturated).  Report
      θ AND R_net (= k_des · Γ at λ=1) per rung.
- [ ] **D7-D4 (coupled Γ_max × k_des matrix; ALWAYS REQUIRED).**
      9-rung matrix at λ=1, V=V_kin, anchored at C_S = 0.20:
      `Γ_max ∈ {Γ_max/2, Γ_max, Γ_max·2}` × `k_des ∈ {0.1, 1.0,
      10.0}` evaluated at both `k_hyd ∈ {1e-3, 1e-1}` → 18
      total rungs.  Pass: 18/18 converge; no sign flip on R_net
      that crosses both `|R_4e| > 1e-6` and the sign convention.
      Output: 2D heatmap of θ and R_net.
- [ ] **D8.** `pytest -m "not slow" -k "phase6b or cation" -s -vv`
      green at the new constants.  New fast tests required:
      * `test_cation_hydrolysis_v10a_v10b_constants_coexist`
      * `test_factory_default_uses_v10b`
      * `test_v10b_calibration_metadata_schema` — asserts each
        of the 3 entries has all required keys.
      * `test_phase6b_v10b_cs_bracket_cli_parses`,
        `test_phase6b_v10b_cs_bracket_target_grid`,
        `test_phase6b_v10b_cs_bracket_output_schema`.
      * Analogous trio for the Γ_max × k_des matrix driver.
      * `test_step6_a2_baseline_json_cli_flag`,
        `test_step6_audit_keys_include_R_net`.
      Tests with literal `0.047` references in
      `test_phase6b_v10a_langmuir_cap.py:90,123,132,147`:
      * Keep the test (it's a v10a sanity check) and rename to
        `test_v10a_langmuir_cap_*` if not already.
      * Add parallel v10b tests using `GAMMA_MAX_HAT_V10B`.
      The two test groups are independent (v10a freezes the V10A
      constant; v10b tests the V10B constant) — no churn on
      future calibration cycles.
- [ ] **D9.** Acceptance bundle § Status appended with the v10b
      paragraph at
      `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`.
- [ ] **D10.** `CLAUDE.md` "Recent progress" updated; total file
      length ≤ 200 lines.
- [ ] **D11.** Memory entry `project_v10b_calibration_outcome.md`
      added with index pointer in `MEMORY.md`.

**Out of v10b's DoD (explicitly):**
* Selectivity gap improvement — Phase D job.
* Legacy script `stern_capacitance_f_m2 = 0.10` migration (~348
  occurrences) — separate post-v10b cleanup task.
* Phase D `k_des` data fit — step 10's scope is K-only Δ_β,
  unless explicitly expanded in a future plan.

---

## 2. Hard invariants (unchanged from v1)

| Constant | Value | Source |
|---|---|---|
| `V_kin` | `−0.10 V` | step 4 |
| `K0_R4e_factor` | `1e-14` | step 4 |
| `k_hyd_baseline` | `1e-3 nondim` | step 5 |
| `k_hyd_route` | `1e-1 nondim` | A.2 (cap-saturated rung) |
| `WARM_WALK_GRID` | `(+0.55, +0.40, +0.20, +0.10, −0.10)` | A.2 + step 6 |
| `LAMBDA_LADDER` | `(0.0, 0.25, 0.50, 0.75, 1.0)` | v10a |
| Parallel topology | `R2e (E°=0.695 V)` + `R4e (E°=1.23 V)` | Ruggiero 2022 |
| `exponent_clip` | `100.0` | CLAUDE.md hard rule #2 |
| `STERN_F_M2_ANCHOR` | `0.10 F/m²` | two-stage anchor pattern |
| `c_s_ladder + kw_eff_ladder` combo | unsupported | `_PARAMETER_OVERRIDE_SETTERS` |
| `τ_REF` | `L_REF² / D_REF ≈ 5 s` | `_bv_common.py:131-134` |

Breaking any of these in v10b → escalate to v10c; do not push past.

---

## 3. The three parameters — decision rules

### 3.0. Nondim mapping (new in v2)

`Γ_max_hat = Γ_max_phys / (C_SCALE · L_REF) = Γ_max_phys / (1.2e-4
mol/m²)`.  Diagnostic: literature `Γ_phys` of `~5.6e-6 mol/m²` →
`Γ_hat ≈ 0.047`.

`k_des_nondim = k_des_phys · τ_REF`, with `τ_REF = L_REF² / D_REF
≈ 5 s`.  So `k_des_phys [1/s] = k_des_nondim / 5`.  Smoke value
`k_des_nondim = 1.0` ↔ `k_des_phys = 0.2 /s` ↔ Eyring with
ΔG_des ≈ 0.9 eV.  Diffusion-limited upper bound at the OHP is
`~D_K+/δ_OHP² ≈ 4e10 nondim` (effectively instantaneous; kinetics
are barrier-limited).

`C_S` is dimensional [F/m²] in the residual; no rescaling.

### 3.1. `C_S` — locked at step 7

**Locked:** `C_S = 0.20 F/m²` per
`docs/phase6/CMK3_capacitance_literature.md`.

**v10b work:**
1. No literature search needed.
2. **Code touch (limited; do NOT batch-update 348 legacy
   occurrences):**
   * `Forward/bv_solver/cation_hydrolysis.py` — constants + default.
   * `scripts/_bv_common.py` — constants + factory default.
   * (`v_sweep_diagnostic.py`, `phase_A2_v_kin.py`,
     `step6_plumbing_ablation.py` already use `STERN_F_M2_BASELINE
     = 0.20` — no change needed.)
   * `phase6b_v10b_cs_bracket.py` (new) — uses V10B constants.
3. **Sensitivity bracket sweep** (D7-D1; always required).
4. **Carry forward step 7 open asks** as items in the v10b
   writeup §6 — do not block v10b on them:
   * Bohra 2019 EES pull into `Articles/`.
   * Risk #5 σ_S mismatch re-derivation with Stern-only 20 µF/cm².
   * Yash convention disposition.

### 3.2. `Γ_max` — literature search

**Current smoke (V10A-frozen):** `0.047 nondim`.

**Literature search targets:**
1. Singh 2016 *JACS* (`10.1021/jacs.6b07612`) — Table S1
   partial-coverage estimates for K⁺ (if any).
2. Iamprasertkun 2019 *JPCL* (`10.1021/acs.jpclett.8b03523`).
3. Bohra 2019 EES (`10.1039/c9ee02485a`) — pull first.
4. Co-Zhang 2019 Angewandte (in `Articles/`).
5. Yash modeling code at `data/.../Yash-Trends/`.
6. `Parameters_Seitz_Mangan.xlsx` — group-internal convention.

**Decision rule:**
* **LOCK to a cited value** ONLY IF the source passes all four
  compatibility tests:
  - Mechanism: source reports MOH adsorbate coverage or
    equivalently constrained surface site density (not generic
    cation packing, not bare-cation Stern occupancy).
  - Electrode: sp²-carbon-cathode preferred; metal-cathode
    acceptable with "site-density transfer" caveat documented.
  - Electrolyte: aqueous K⁺-containing OR alkali-cation sub-
    tabling with K⁺ entry present.
  - Dimensional: `Γ_phys / (C_SCALE · L_REF) = Γ_phys / (1.2e-4
    mol/m²)` produces a finite nondim value.
* **ELSE tighten the V10A derivation chain** (no value change;
  V10B = V10A; document the source for "K⁺ hydrated radius =
  2.3 Å" with peer-reviewed citation).
* **ELSE engineering choice** with prior
  `Γ_max_nondim ∈ {V10A/2, V10A, V10A·2}`; D7-D2 + D7-D4 bracket
  evidence.

### 3.3. `k_des` — research + likely engineering choice

**Current smoke (V10A-frozen):** `k_des_nondim = 1.0` (no anchor).

**Note (was strategy 1 in v1, removed):** Singh 2016 pKa is
the equilibrium `M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺`, i.e. `K_eq = k_hyd /
k_prot`.  This constrains `k_prot`, not `k_des`.  v10b uses
Singh pKa ONLY for `k_prot` consistency audit (separate from
this calibration); for `k_des` it has no bearing.

**Research strategies (in order of attempt):**

1. **Analogous reactions: `OH*` desorption from sp²-carbon.**
   Read:
   * Nørskov-Viswanathan 2012 *JPCL* (in `Articles/`).
   * Co-Billy 2017 ACS Catal (in `Articles/`).
   * Other Nørskov-group Sabatier-volcano papers via the data-
     folder index.
   If a transportable `k_des` order of magnitude with documented
   uncertainty (1–2 decades) emerges → lock central value +
   bracket sweep.
2. **Eyring estimate from cation-OH binding energy.**
   `k_des_phys = (k_BT/h) · exp(−ΔG_des/RT)`; ΔG_des estimated
   from cation-OH bond energy literature.  If ΔG_des can be
   bracketed within ±0.2 eV → corresponds to k_des within ~1
   decade → lock with bracket.
3. **Engineering-choice fallback (most honest).**  Mark
   `engineering_choice = True`; document prior:
   `k_des_nondim ∈ [10⁻², 10²]` from Eyring with ΔG_des ∈
   [0.7, 1.1] eV.  Run D7-D3 bracket at `{0.1, 1.0, 10.0}`.
   Document "open parameter for post-v10b; future scope
   expansion to fit `k_des` against data would use this
   calibration as the initial prior."  Do NOT promise Phase D
   will fit it (locked scope is K-only Δ_β).

**Decision rule:**
* IF strategy 1 yields a defensible order of magnitude with
  electrode/electrolyte transferability documented → lock
  central + bracket.
* ELSE IF strategy 2 yields ΔG_des within ±0.2 eV → lock with
  Eyring + bracket.
* ELSE → strategy 3 fallback.  Do NOT fabricate a citation.

**A2 diagnostic risk flag (from handoff §7, retained):** K⁺
Boltzmann pile-up at V_kin (`c_K_boundary_avg ≈ 291 · c_K_bulk`)
means sentinel-scale `R_inj` perturbations cannot dent boundary
c_K by 5%.  v10b MUST NOT use boundary-c_K perturbation as a
`k_des` calibration diagnostic.  Use `θ` AND `R_net` at cap-
saturated `k_hyd_route = 1e-1` instead.

---

## 4. Phase breakdown (with dependencies)

Total estimate: **8–12 working days**.

### Phase v10b.A — Literature pass (~3 days, parallelizable)

**Goal:** decide each parameter's D1 outcome.  Produces draft
writeup §1-§3 + numeric values for D3/D4.

**Sub-phases (parallel):**
* **A1. Γ_max literature pass.**  Decision rule §3.2.  Apply 4-
  test compatibility check before locking.
* **A2. k_des literature pass.**  Strategies 1→2→3.  Do not stall
  on infinite search; engineering choice is acceptable.
* **A3. C_S follow-up.**  Pull Bohra 2019 EES if accessible;
  re-derive Risk #5 σ_S; document Yash convention.

Run A1+A2+A3 as three parallel Agent calls.  Each agent gets the
decision rule, file targets, and the writeup section template
(mirror of `CMK3_capacitance_literature.md` structure).

**Gate at end of A:** writeup draft §1-§3 + `V10B_CALIBRATION_METADATA`
populated.

### Phase v10b.B — Code change + unit-test regression (~1 day)

**Depends on:** A1, A2 final values.

**Steps:**
1. Solver-layer constants (D3): freeze V10A, add V10B, update
   defaults at lines 293, 295.
2. `scripts/_bv_common.py` constants (D4): freeze V10A, add V10B,
   factory default switch at line 957.
3. V-sweep diagnostic constants (D4'): freeze
   `SMOKE_KINETICS_V10A`, add `V10B_KINETICS`, update factory
   signature defaults at lines 821-826, 866-871.
4. Add `V10B_CALIBRATION_METADATA` block (D1).
5. New fast tests (D8): constants coexistence, factory default,
   metadata schema, bracket driver schema, step 6 audit fields.
6. Refactor test literals in
   `tests/test_phase6b_v10a_langmuir_cap.py:90,123,132,147` to
   use `GAMMA_MAX_HAT_V10A_SMOKE` constant (frozen, so OK to
   reference); add parallel v10b tests using
   `GAMMA_MAX_HAT_V10B`.
7. Add `--use-v10a-smoke` CLI flag to v-sweep / A.2 / step 6
   drivers for historical reproduction (optional convenience;
   land if low-cost, defer otherwise).
8. Run `pytest -m "not slow" -k "phase6b or cation" -s -vv`.

### Phase v10b.C — A.2 + step 6 regression (~1 day wall)

**Depends on:** Phase B complete.

**Steps:**
1. Verify / add `--a2-baseline-json` CLI flag on
   `phase6b_step6_plumbing_ablation.py`.
2. Verify / add `R_net` to `_baseline_reproduction_audit` keys
   in step 6 driver (D6 contract).
3. Re-run Phase A.2 driver with `--out-subdir
   phase6b_v10b_phase_A2_v_kin`.
4. Apply D5.HARD gates; if fail → escalate to v10c.
5. Re-run step 6 driver with `--out-subdir
   phase6b_v10b_step6_plumbing_ablation
   --a2-baseline-json
   StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json`.
6. D6 byte-equivalence rel ≤ 1e-6 in {cd, R_net, σ_S, θ} verified.
7. Commit regenerated JSONs + PNGs.

### Phase v10b.D — Sensitivity bracket sweeps (~2 days wall, parallelizable with C)

**Depends on:** Phase B complete.

**D1 work:**
* D7-D1: new driver `scripts/studies/phase6b_v10b_cs_bracket.py`
  with the two-stage anchor pattern.  4 rungs.
* D7-D4: coupled Γ_max × k_des matrix driver
  `scripts/studies/phase6b_v10b_gamma_kdes_matrix.py`.  18 rungs
  (Γ_max ∈ {V10B/2, V10B, V10B·2} × k_des ∈ {0.1, 1.0, 10.0} ×
  k_hyd ∈ {1e-3, 1e-1}).
* D7-D2 (conditional): Γ_max-only bracket if not locked.
* D7-D3 (conditional, likely required): k_des bracket at both
  k_hyd values.

### Phase v10b.E — Writeup + acceptance bundle (~1 day)

**Depends on:** A, B, C, D complete.

**Steps:**
1. Finalize `docs/phase6/v10b_calibration_summary.md` mirroring
   `CMK3_capacitance_literature.md` structure.  Includes the
   verbatim `V10B_CALIBRATION_METADATA` block.
2. Append v10b paragraph to acceptance bundle § Status (D9).
3. Update `CLAUDE.md` "Recent progress" (D10).  Consolidate
   v10a' / A.2 / step 6 narrative; budget ≤ 200 lines.
4. Write `project_v10b_calibration_outcome.md` memory entry
   (D11) + `MEMORY.md` pointer.
5. **DO NOT** audit the 348 legacy `C_S = 0.10` occurrences.
   That's a separate post-v10b cleanup task; documented as an
   open ask in writeup §6 with the v10b-cleanup tag.

---

## 5. Risk + mitigation register (v2)

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | `k_des` literature pass yields nothing | HIGH | LOW | Engineering-choice flag §3.3 strategy 3; D7-D3 + D7-D4 close evidentially. |
| R2 | New `Γ_max` shifts σ_S manifold so V_kin or `K0_R4e_factor` no longer satisfy the locked rule | LOW | HIGH | Detect in D5.HARD; ESCALATE to v10c. |
| R3 | C_S sensitivity sweep fails 4/4 convergence | MEDIUM | HIGH | Two-stage anchor pattern.  4/4 IS the contract; failures ESCALATE to v10c (no 3/4 fallback). |
| R4 | Convergence-audit threshold-narrowness artifact (overall_pass=False at θ_max=0.9253) re-appears | HIGH | LOW (known) | Document; do not change `transition_grid_threshold`. |
| R5 | V10A test refs break when value changes / V10B coexistence test missing | LOW | LOW | D8 explicit tests; V10A constants frozen, never aliased. |
| R6 | A0 byte-equivalence broken vs. new v10b A.2 baseline | LOW | HIGH | Step 6 driver gets explicit `--a2-baseline-json` CLI; D6 verifies. |
| R7 | CLAUDE.md exceeds 200-line budget | MEDIUM | LOW | Consolidate v10a' / A.2 / step 6 narrative in Phase E. |
| R8 | Adjoint-tape contamination | LOW | HIGH | v10b is forward-only; no `scripts/Inference/` touches. |
| R9 | Bohra 2019 EES paywalled | LOW | LOW | Step 7 already shipped without it; open ask. |
| R10 | k_des/Γ_max calibration moves cd by >20% from v10a' | MEDIUM | LOW (data, not gate) | D5 split: HARD gates are convergence/invariants; SOFT gates (cd/x_2e) are informative, not escalation triggers. |
| R11 | Sign convention or magnitude-floor misapplied in D7 σ_S check | MEDIUM | MEDIUM | Explicit sign convention + 1e-6 floor on R_4e specified in D7-D1. |
| R12 | k_des bracket diagnostic at wrong k_hyd value | LOW | MEDIUM | D7-D3 explicitly evaluates BOTH `k_hyd_baseline = 1e-3` and `k_hyd_route = 1e-1`. |
| R13 | Singh pKa consistency check for `k_prot` not done (separate from k_des) | LOW | LOW | Out of v10b scope; carry as open ask in writeup §6. |
| R14 | Result JSON `smoke_kinetics` key collision in dual-mode drivers | LOW | LOW | Driver writes `v10b_kinetics` when run under V10B; `v10a_smoke_kinetics` only with `--use-v10a-smoke` flag. |

---

## 6. Out of scope (v2; mostly unchanged)

- Phase D (K-only Δ_β fit) — step 10.
- Cs⁺ / Li⁺ / Na⁺ / Rb⁺ extension — Phase E, step 11.
- Variable-`ε_S` / Booth-equation refinement — post-v10b.
- Inverse / adjoint work — paused.
- V_kin re-selection / `K0_R4e_factor` retune — step 4 invariants.
- L_Stern parameterization vs `C_S` parameterization choice —
  open ask from step 7.
- Selectivity-gap improvement — Phase D's job.
- **Legacy 348-line `C_S = 0.10` audit + script migration —
  separate post-v10b cleanup task; carried as open ask.**
- **Phase D `k_des` fitting — Phase D scope is K-only Δ_β;
  engineering-choice k_des is a post-v10b open parameter.**

---

## 7. Dependency graph (v2)

```
v10b.A (parallel)
├── A1 Γ_max research ──┐
├── A2 k_des research ──┼──► writeup draft + V10B_CALIBRATION_METADATA
└── A3 C_S follow-up ───┘                │
                                          ▼
                                    v10b.B (serial)
                                    Code changes + unit tests
                                          │
                          ┌──────────────┴──────────────┐
                          ▼                              ▼
                     v10b.C (serial)             v10b.D (parallel)
                     A.2 + step 6                 C_S bracket
                     regression                   + Γ_max × k_des matrix
                          │                              │
                          └──────────────┬──────────────┘
                                          ▼
                                    v10b.E (serial)
                                    Writeup + acceptance bundle
                                    + CLAUDE.md + memory entry
```

Critical path: A → B → C → E.  D parallel with C.

---

## 8. Validation checkpoints (v2)

1. **End of A:** writeup §1-§3 draft; `V10B_CALIBRATION_METADATA`
   populated; engineering-choice flags set where applicable.
2. **End of B:** `pytest -m "not slow" -k "phase6b or cation"`
   green (D8); new fast tests passing.
3. **End of C:** D5.HARD gates close; D6 byte-equivalence verified
   with new baseline path.  D5.SOFT deltas logged, not gated.
4. **End of D:** D7-D1 (4/4) closes; D7-D4 (18/18) closes;
   conditional D7-D2 / D7-D3 close as needed.
5. **End of E:** D9 + D10 + D11 close.  v10b shipped.

---

## 9. Decision rules summary (v2)

| Parameter | Outcome path |
|---|---|
| `C_S` | Locked at 0.20 F/m²; D7-D1 bracket + carry open asks. |
| `Γ_max` | Compatibility-checked lit anchor → tightened V10A chain (value unchanged) → engineering choice with bracket. |
| `k_des` | Analogous-reaction order of magnitude → Eyring with ΔG_des bracket → engineering choice with Eyring prior. |

**If in doubt, prefer engineering-choice flag with documented prior
and bracket evidence over a fabricated citation.**

---

**End of plan v2.**
````````````````````````

---

## Section 4 — Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
