# Critique handoff R1 — Phase 6β Step 6 plumbing ablation plan

You are reviewing Claude's plan for **Phase 6β Step 6 (plumbing
ablation matrix at V_kin = -0.10 V)** of a research PNP-BV solver
project (Poisson–Nernst–Planck + Butler–Volmer kinetics for ORR on
CMK-3 carbon).  Step 6 is the immediate next step after the
just-landed Phase A.2 work (commit `2f5f071`).

GPT critique sessions 32 (3 rounds), 33 (4 rounds), and 34 (4 rounds)
hardened the v10a / v10a' / A.2 plans; this is session 35 doing the
same for step 6.

## Section 1 — Context bundle

### Project state (just-landed)

Phase A.2 (commit `2f5f071`) landed clean at V_kin = -0.10 V with:
* 10/10 k_hyd rungs (k_hyd ∈ {1e-5 … 1e-1}) converged at λ=1.0.
* Picard converged everywhere; mass-balance residual at machine
  precision (1e-14 to 1e-16).
* Baseline at k_hyd=1e-3 reproduces v10a' record within rel 1e-3:
  γ=0.0405, θ=0.861, σ_S=−0.01715 C/m², cd=−3.12 mA/cm².
* **k_hyd_route = 1e-1** (highest k_hyd passing all five gates).
* v10b k_des/Γ_max priority **LOW** (single_v_selectivity_gap_pp =
  +5.09 pp; H₂O₂% = 19.91% sits within 10pp of deck band [25, 50]).
* rH_El recalibration **NOT required**
  (max_amp_from_singh = 1.0000112).

**Striking observation in A.2:** σ_S, cd_mA_cm², x_2e are
**k_hyd-independent** across the full 10-point grid at V_kin.  This
is the central scientific motivation for step 6: that decoupling
could be either (a) a clean physics result — cation hydrolysis Γ
affects the proton boundary source but not the Stern field or BV
branch split, OR (b) a residual-side wiring bug where the cation
hydrolysis source is silently dropped from the H+ residual (or
silently going somewhere wrong).  Step 6 discriminates between
these via three manufactured/imposed ablations.

### Step 6 in the locked sequence

Per `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` §
"v10a → E sequence":

1. ✅ Phase 0 (acceptance bundle).
2. ✅ v10a (Langmuir cap).
3. ✅ v10a' (V-sweep diagnostic, picks V_kin).
4. ✅ V_kin selection (= -0.10 V; primary path, no fallback).
5. ✅ Phase A.2 (densified k_hyd ramp).
6. **← THIS STEP — plumbing ablation matrix at V_kin (~2 days).**
7. CMK-3 capacitance literature note (v10b prerequisite, mostly landed).
8. v10b — Γ_max + k_des + C_S literature calibration (~1-2 weeks).
9. B.2 — densified k_hyd × λ at V_kin with v10b values (~2 days).
10. Phase D — K-only Δ_β fit (~1 day).
11. Phase E — predictive holdout (~3-5 days).

### Locked invariants (HARD precondition across steps 6 / 8 / 9)

* `K0_R4e_factor = 1e-14`
* `C_S = 0.20 F/m²` (Bohra-Koper-Choi consensus)
* `V_kin = -0.10 V`
* `k_hyd_baseline = 1e-3` (the v10a' / A.2 reference)

### Three new ctx flags proposed (don't exist in codebase yet)

| Flag | Type | Default | Effect |
|---|---|---|---|
| `apply_h_source` | `bool` | `True` | When `False`, omit the cation-hydrolysis proton residual term entirely. |
| `apply_k_sink` | `bool` | `True` | When `False`, omit the cation-hydrolysis K+ residual term entirely. |
| `override_sigma_singh_counts_pm2` | `Optional[float]` | `None` | When set, replace `sigma_S` in `build_pka_shift` with a constant such that the post-clamp σ_singh_counts = override. |

### Current solver wiring (where the new gates land)

`Forward/bv_solver/forms_logc_muh.py:715-749`:

```python
pka_shift_expr = build_pka_shift(
    cation_params=cation_hydrolysis_bundle.cation_params,
    sigma_S=sigma_S_expr,
    r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
)
R_net_default = build_proton_boundary_source(
    bundle=cation_hydrolysis_bundle,
    c_M_bdy_expr=c_M_bdy_expr,
    c_H_bdy_expr=c_H_bdy_expr,
    pka_shift_expr=pka_shift_expr,
)
manufactured_R_inj = conv_cfg.get("manufactured_R_inj", None)
if manufactured_R_inj is not None:
    R_net = fd.Constant(float(manufactured_R_inj))
else:
    R_net = R_net_default

lam_func = cation_hydrolysis_bundle.lambda_hydrolysis_func
F_res -= (
    lam_func * R_net
    * v_list[h_idx_for_cation]
    * ds(electrode_marker)
)
F_res -= (
    lam_func * (-R_net)
    * v_list[cation_hydrolysis_bundle.counterion_idx]
    * ds(electrode_marker)
)
```

The new gates would wrap the two `F_res -= ...` statements with
`if apply_h_source: ...` and `if apply_k_sink: ...`.

### σ-counts conversion (resolved during plan-writing review)

The plan's Risk #1 originally flagged ambiguity in the conversion
factor between σ_S in C/m² and σ in counts/pm² (counts of elementary
charges per square picometre).  The codebase has it nailed down in
`Forward/bv_solver/cation_hydrolysis.py:457-467`:

```
σ_singh_counts/pm² = max(0, −σ_S_C_per_m²) × (N_A / F) × 1e-24
Using N_A·e = F (definitional) ⇒ N_A/F = 1/e ≈ 6.2415e18 1/C
_INVERSE_ELEMENTARY_CHARGE = 1.0 / 1.602176634e-19
```

Concretely: `σ_singh_counts_pm² = σ_C_per_m² · (1/1.602e-19) · 1e-24`
= `σ_C_per_m² · 6.2415e-6`.  So for Singh's Cu calibration
σ ≈ 226 µC/cm² = 2.26 C/m², the counts/pm² value is
2.26 × 6.2415e-6 ≈ **1.41e-5** counts/pm².

The plan's Risk #1 (currently) says "6.243e-6 would give σ_S ≈ 14.1"
— that arithmetic is wrong by 6 orders of magnitude (units error;
dropped the 1e-24 factor).  The corrected `SIGMA_SINGH_K_CU_OVERRIDE`
is on the order of `1.41e-5` counts/pm², not 14.1.  Please re-check
this derivation and flag any unit/sign errors you find.

### What I want you to look for

Be argumentative.  Focus on:

1. **Plumbing design**: does the proposed flag wiring actually
   isolate the source from the sink?  Are there hidden couplings I'm
   missing (e.g., through `build_proton_boundary_source` reading
   pka_shift_expr internally)?
2. **Pass criteria thresholds**: the 5% c_H shift threshold, the 1%
   "unchanged" tolerance, the 10% σ_S leak tolerance for A3 — are
   these defensible?  Could a real bug pass these thresholds, or
   could a clean run fail them?
3. **σ-counts conversion + override value**: the corrected
   `SIGMA_SINGH_K_CU_OVERRIDE ≈ 1.41e-5` derivation above — verify.
   Is the anode-clamp `max(0, −σ_S)` correctly handled in the
   override path?
4. **R_INJ_MANUFACTURED bracketing logic**: pre-pass runs A1 at
   {1e-2, 1e-1, 1.0} nondim.  Is "smallest value clearing 5% with
   ≥2× headroom" the right criterion?  Should we also enforce an
   UPPER bound (e.g., 5–25% c_H shift) to stay in linear response?
5. **Newton convergence risk for A1/A2**: the manufactured break
   creates a one-sided source/sink without conservation.  Will
   Newton converge?  Can we distinguish "Newton failed because of
   the unphysical break" from "Newton failed because the plumbing is
   buggy"?  My mitigation (Risk #10) of "cap c_K(0) at 1% of bulk
   via soft clamp" feels hacky.
6. **A3 pass criterion's tautology risk**: requiring
   `|pka_shift_avg − β_K_Cu·σ_override|/scale < 0.05` is trivially
   true if `build_pka_shift_from_override` is implemented correctly.
   Is the second gate `|Δσ_S vs A0|/σ_S < 0.10` the real test?
7. **Decoupling-claim discriminator**: would a bug where
   `build_proton_boundary_source` returns `0.0` instead of the
   actual R_net produce a passing A0 byte-equivalence AND passing
   A1/A2 (because manufactured_R_inj bypasses
   build_proton_boundary_source)?  If so, the ablations would say
   "plumbing OK" while the actual physical-path R_net is broken.
   How to catch this?
8. **Byte-equivalence test scope**: A0 reproduces A.2 baseline at
   k_hyd=1e-3 within rel 1e-12.  Is that too tight for PETSc
   floating-point reproducibility?
9. **Tests that don't require Firedrake**: limitations on what we
   can guard purely in Python.
10. **Routing decisions table**: thresholds for "A0 fails" — the
    plan says "rel 1e-12" but doesn't specify what to do at rel
    1e-9 vs 1e-12.
11. **Other holes you find**: missing edge cases, sign errors, scope
    creep, untested assumptions, dimensional errors, off-by-one.

## Section 2 — The artifact under review

Full plan (539 lines) below.  Note line "### Section 2 — The
artifact" continues into the plan text:

---PLAN START---
# Plan — Phase 6β Step 6: Plumbing ablation matrix at V_kin

## Provenance

To be hardened by GPT critique loop (target: APPROVED).  Predecessor
A.2 plan at `~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md` (4
rounds, APPROVED; commit `2f5f071`).

## Context

Phase A.2 landed clean (2026-05-10): all 10/10 k_hyd rungs converge
at λ=1.0; baseline at k_hyd=1e-3 reproduces v10a' record within rel
1e-3; mass-balance residual at machine precision; **k_hyd_route =
1e-1**; v10b k_des/Γ_max priority LOW (`single_v_selectivity_gap_pp
= +5.09 pp`); rH_El recalibration NOT required.  See
`StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.json` and
`project_v10a_phase_A2_outcome` memory.

A.2 also surfaced a striking decoupling at V_kin = -0.10 V: **σ_S,
cd_mA_cm², and x_2e are k_hyd-independent across the entire 10-point
grid** (cation hydrolysis affects the proton boundary source but not
the FE residual's Stern or BV terms at this V).  Step 6 verifies
this decoupling is **physically meaningful and plumbed correctly**,
not a residual-side bug masquerading as a clean result, BEFORE v10b
spends 1–2 weeks calibrating Γ_max + k_des + C_S against literature.

Per the locked acceptance-bundle sequence
(`docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` § "v10a
→ E sequence", step 6), this step is the **plumbing ablation matrix
at V_kin = -0.10 V** with A1/A2/A3 manufactured-and-imposed
ablations.  Its purpose is fourfold:

1. **Confirm `apply_h_source` ⇒ surface c_H rises** (A1): a manufactured
   R_inj that flows ONLY into the proton residual should drive the
   boundary-averaged c_H up by ≥5%.  Catches a bug where the proton
   boundary source is mis-wired into K+ residual instead, or where
   it's silently zeroed.
2. **Confirm `apply_k_sink` ⇒ c_K(0) falls** (A2): a manufactured
   R_inj that flows ONLY into the K+ residual should deplete c_K at
   the OHP.  Catches a bug where the K+ sink is mis-signed.
3. **Confirm `override_sigma_singh_counts_pm2` decouples ΔpKa from σ_S**
   (A3): replacing the PNP-solved σ_S with a deck-Cu-cited Singh value
   in the ΔpKa formula should produce a Γ steady-state matching the
   imposed-σ closed-form prediction.  Catches a bug where the override
   doesn't actually bypass the PNP coupling.
4. **Establish manufactured/imposed baselines for v10b**: A.2's
   k_hyd-independence claim at V_kin presupposes the plumbing is
   correct; A1+A2 verify the plumbing carries R_net the way the math
   says; A3 verifies σ-mapping independence as a control before v10b
   does any literature calibration.

Step 6 is **NOT** v10b literature calibration (step 8) or Phase D
fitting (step 10).  A4 (sulfate-analytic-disabled) and A5 (physical
Singh hydrolysis at large k_hyd with v10b-calibrated capacity) are
**deferred** out of step 6 per the original 11-step roadmap
(`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` §6).

## Scope (Step 6 only)

### What's in scope

1. **New solver ctx flags** — added to `bv_convergence` config block:

   | Flag | Type | Default | Effect |
   |---|---|---|---|
   | `apply_h_source` | `bool` | `True` | When `False`, zero the cation-hydrolysis proton residual contribution (`F_res -= 0 · v_H · ds`).  Affects the FE residual ONLY at the electrode marker; bulk H+ Dirichlet BC unchanged. |
   | `apply_k_sink` | `bool` | `True` | When `False`, zero the cation-hydrolysis K+ residual contribution (`F_res -= 0 · v_K · ds`).  Affects the FE residual ONLY at the electrode marker; bulk K+ Dirichlet BC unchanged. |
   | `override_sigma_singh_counts_pm2` | `Optional[float]` | `None` | When set, replace `sigma_S` in `build_pka_shift` with a constant such that the post-clamp `σ_singh_counts = max(0, −σ_S_signed_counts) = override_value` (positive number in counts/pm²).  Bypasses the PNP/Stern coupling for the ΔpKa formula ONLY.  Does not affect the residual-side `R_net` σ_S coupling — only `pka_shift_expr`. |

   **Defaults preserve byte-equivalence** with v9/v10a/v10a'/A.2 (no
   numerical change at the established baseline).  CI byte-equivalence
   test required: run the A.2 baseline at k_hyd=1e-3 with all three
   flags at defaults; result must match the committed A.2 record
   within rel 1e-12 (FP-arithmetic agreement).

2. **Solver-side wiring** (both `forms_logc_muh.py` and `forms_logc.py`):

   * `apply_h_source` gates the proton residual `λ·R_net·v_H·ds` term.
   * `apply_k_sink` gates the K+ residual `λ·(-R_net)·v_K·ds` term.
   * `override_sigma_singh_counts_pm2` gates `pka_shift_expr` build:
     when set, `pka_shift_expr = build_pka_shift_from_override(...)`
     using a `Constant` σ_S that converts to the override counts value.

3. **New driver** `scripts/studies/phase6b_step6_plumbing_ablation.py`:

   * **Two-stage anchor at V=+0.55 V** (same pattern as A.2): build
     at C_S=0.10, runtime-bump to C_S=0.20.
   * **Warm-walk to V_kin = -0.10 V** at λ=0 (5-point grid identical
     to A.2: `{+0.55, +0.40, +0.20, +0.10, -0.10}`).
   * **For each ablation in {A0, A1, A2, A3}:**
     * Build sp_template with the appropriate ctx flags.
     * Run `solve_lambda_ramp_from_warm_start` from the V_kin snapshot
       with custom `rung_callback` augmenting v10a' diagnostics.
     * Hold `k_hyd_target = 1e-3` (the v10a' baseline) for all
       ablations — this gives a single clean comparison point.
     * Hold `manufactured_R_inj = R_INJ_MANUFACTURED` (a single
       bracketed value, NOT a sweep — see §4 below) for A1/A2 only.
     * Hold `override_sigma_singh_counts_pm2 = SIGMA_SINGH_K_CU_OVERRIDE`
       (deck-Cu-cited Singh K⁺ value) for A3 only.

4. **A0 (baseline)**: defaults (apply_h_source=True, apply_k_sink=True,
   override_sigma_singh_counts_pm2=None, no manufactured_R_inj).
   Reproduces A.2 baseline at k_hyd=1e-3.  **Required** for the
   byte-equivalence check (R0-R3 pass criteria below).

5. **A1 — Source-only manufactured:**
   * `apply_h_source = True`, `apply_k_sink = False`,
     `manufactured_R_inj = R_INJ_MANUFACTURED`,
     `override_sigma_singh_counts_pm2 = None`.
   * Expected: c_H(boundary) rises ≥ 5% relative to A0;
     c_K(boundary) unchanged within tolerance (3σ < 1%).
   * Captures: c_H_avg, c_K_avg, Γ, R_net, surface concentration
     deltas vs A0.

6. **A2 — Sink-only manufactured:**
   * `apply_h_source = False`, `apply_k_sink = True`,
     `manufactured_R_inj = R_INJ_MANUFACTURED`,
     `override_sigma_singh_counts_pm2 = None`.
   * Expected: c_K(boundary) falls ≥ 5% relative to A0;
     c_H(boundary) unchanged within tolerance.

7. **A3 — Imposed Singh σ:**
   * `apply_h_source = True`, `apply_k_sink = True`,
     `manufactured_R_inj = None` (PHYSICAL path),
     `override_sigma_singh_counts_pm2 = SIGMA_SINGH_K_CU_OVERRIDE`.
   * Expected: `pka_shift_avg` matches the closed-form prediction
     `β · σ_override` with β from `build_pka_shift` for K⁺ on Cu;
     PNP-solved σ_S(C/m²) at the OHP can be anything but ΔpKa is
     pinned.  Γ_ss should be close to A0 because Singh ΔpKa is
     small at V_kin (A.2 saw amp_from_singh ≈ 1.00001), so the
     dominant effect is sanity-checking that `pka_shift_avg`
     responds correctly.

8. **Custom rung_callback** — A.2's `augment_rung_diagnostics`
   PLUS:
   * `c_H_boundary_avg`, `c_K_boundary_avg` (already in A.2 via
     `c_H_avg` and `F0_decomposition.c_K_avg`).
   * **Δc_H_vs_A0_rel** and **Δc_K_vs_A0_rel** (computed post-run
     by the driver across all 4 ablations from the JSON records).
   * **σ_S_solved_at_OHP_C_per_m2** (already in `sigma_S_C_per_m2`).
   * **σ_singh_counts_active** — the actual value used by
     `build_pka_shift` after the override (for A3 audit).
   * **pka_shift_avg** (already in v10a diagnostics).

9. **`R_INJ_MANUFACTURED` bracketing** — the manufactured R_inj
   value used for A1+A2 must produce ≥5% surface c_H shift in A1
   AND ≥5% c_K shift in A2 at V_kin = -0.10 V with the v10a smoke
   kinetics held at A.2 baseline values.  Per A.2's record, the
   nondim Γ at k_hyd=1e-3 is 0.0405, so R_net = k_des·Γ = 0.0405
   nondim.  **Bracketing strategy:** start with
   `R_INJ_MANUFACTURED = 1e-2` nondim (≈ 25% of A0's R_net at λ=1).
   If A1's Δc_H_vs_A0_rel < 5%, escalate to 1e-1, 1.0 in
   half-decade steps up to a hard ceiling of 10.0 nondim
   (≈ 250× A0's R_net).  Document the final value in
   `R_INJ_MANUFACTURED_USED`.

   **Sub-decision**: bracket determination is done by a small
   pre-pass that runs A1 only at three R_inj values
   `{1e-2, 1e-1, 1.0}` and picks the smallest that clears 5% with
   ≥2× headroom.  This bracket value is then re-used for A2 to
   keep A1/A2 symmetric.

10. **`SIGMA_SINGH_K_CU_OVERRIDE`** — single deck-Cu-cited Singh K⁺
    value.  Per `docs/phase6/singh_2016_pka_formula.md` and Singh
    2016 SI Table S1, the K⁺-on-Cu σ value at the deck condition
    (typical experimental Δφ_cell = 4.4 V, C_dl = 51 µF/cm²) is
    σ ≈ 226 µC/cm² → counts/pm² via the 6.243e-6 conversion:
    `226 µC/cm² × (1 C / 1e6 µC) × (1 cm² / 1e16 pm²) × N_A` ...

    Wait, the conversion is: `σ_counts_per_pm² = σ_C_per_m² ·
    (1 m² / 1e24 pm²) / e_charge`.  At σ = 2.26 C/m², counts/pm² =
    2.26 × (1/1e24) / 1.602e-19 = 2.26 × 1e-24 / 1.602e-19 ≈
    1.41e-5 counts/pm² (i.e., ~1.4e-5 elementary charges per pm²).
    The cross-check value `6.243e-6` quoted in
    `PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` line 159
    suggests `σ_counts_per_pm² = σ_C_per_m² · 6.243e-6` which would
    give σ at 226 µC/cm² = 2.26 C/m² → counts/pm² ≈ 14.1.  This is
    a 6 order of magnitude discrepancy that needs resolution
    before A3 runs.  **Open: derive the exact conversion factor +
    re-derive the deck-Cu-cited Singh K⁺ value in counts/pm² with
    a clear citation chain.**  Place the resolution in the plan's
    Risks section.

11. **Output:**
    * `StudyResults/phase6b_step6_plumbing_ablation/ablation_matrix.json`:
      per-ablation records {A0, A1, A2, A3} with the full v10a'
      decomposition + per-ablation deltas + pass/fail flags.
    * `StudyResults/phase6b_step6_plumbing_ablation/ablation_matrix.png`:
      4-panel comparison:
      (A) c_H_boundary_avg per ablation (bar chart with A0 baseline).
      (B) c_K_boundary_avg per ablation.
      (C) Γ per ablation.
      (D) pka_shift_avg per ablation (highlights A3's override).

12. **Pass criteria** (structured pass criterion per ablation):

    | Ablation | Pass condition | Rationale |
    |---|---|---|
    | A0 | Reproduce A.2 baseline at k_hyd=1e-3 within rel 1e-3 AND byte-equivalence at rel 1e-12 with all-default flags | Plumbing additions must not break v10a' |
    | A1 | `Δc_H_vs_A0_rel ≥ +5%` AND `|Δc_K_vs_A0_rel| < 1%` | Source-only manufactured raises c_H, leaves c_K alone |
    | A2 | `Δc_K_vs_A0_rel ≤ -5%` AND `|Δc_H_vs_A0_rel| < 1%` | Sink-only manufactured falls c_K, leaves c_H alone |
    | A3 | `\|pka_shift_avg_A3 − β_K_Cu · σ_override\| / max(\|β_K_Cu · σ_override\|, 1e-30) < 0.05` AND `\|sigma_S_C_per_m2_A3 − sigma_S_C_per_m2_A0\| / max(\|sigma_S_C_per_m2_A0\|, 1e-30) < 0.10` (PNP σ_S only weakly perturbed) | Override pins ΔpKa to closed form; PNP coupling intact for residual side |

    All four must pass for step 6 to be considered "plumbing
    verified".  Any failure routes to a debug step.

13. **Tests** (`tests/test_phase6b_step6_plumbing_ablation.py`):
    * `apply_h_source=True, apply_k_sink=True, override=None` produces
      byte-equivalence with default code path (rel 1e-12).
    * `apply_h_source=False` zeroes the proton residual contribution
      (UFL form-level: verify the H+ test-function multiplier is 0).
    * `apply_k_sink=False` zeroes the K+ residual contribution.
    * `override_sigma_singh_counts_pm2=X` produces `pka_shift_avg`
      consistent with `β_K_Cu · X` at the OHP, regardless of
      PNP-solved σ_S.
    * `_parse_args` parses `--r-inj-manufactured 1e-2`,
      `--sigma-singh-override 14.1` (or whatever the resolved value is).
    * Bracket-determination pre-pass returns a single value clearing
      ≥2× headroom on 5% c_H shift.
    * `classify_ablation_status` (analogous to A.2's `classify_no_route_cause`)
      returns each of {A0_pass, A1_pass, A2_pass, A3_pass} or the
      appropriate `*_fail_reason` for each synthetic record.

### What's out of scope (deferred)

* **A4** — sulfate-analytic disabled with concrete replacement
  residual (requires sulfate-as-dynamic-species rewiring; separate scope).
* **A5** — physical Singh hydrolysis at large k_hyd with v10b-calibrated
  capacity (post v10b only).
* **v10b literature calibration** of Γ_max + k_des + C_S (step 8).
* **Multiple V_kin** — step 6 is single-V.
* **Multiple cations** — K⁺ only.
* **k_hyd sweep beyond baseline** — A1/A2/A3 are run at k_hyd=1e-3
  only (the v10a' baseline).
* **2D ablation grid** — no combined (apply_h_source=False AND
  apply_k_sink=False) test; these flag combinations are tested
  independently.

## Critical files to modify / create

| File | Type | Purpose |
|---|---|---|
| `Forward/bv_solver/forms_logc_muh.py` | MODIFY | Wire `apply_h_source` / `apply_k_sink` / `override_sigma_singh_counts_pm2` into the residual + ΔpKa form-build |
| `Forward/bv_solver/forms_logc.py` | MODIFY | Same wiring for the non-muh formulation (kept in sync) |
| `Forward/bv_solver/cation_hydrolysis.py` | MODIFY | Add `build_pka_shift_from_override` helper (or inline override in `build_pka_shift`) |
| `Forward/bv_solver/config.py` | MODIFY | Parse the three new keys from raw config + plumb defaults |
| `scripts/_bv_common.py` | MODIFY | (Optional) expose `SIGMA_SINGH_K_CU_OVERRIDE` constant; document the conversion factor |
| `scripts/studies/phase6b_step6_plumbing_ablation.py` | NEW | Step 6 driver |
| `tests/test_phase6b_step6_plumbing_ablation.py` | NEW | Unit tests for flags + byte-equivalence + classify_ablation_status |
| `tests/test_phase6b_v10a_phase_A2_driver.py` | (verify) | A.2 byte-equivalence test still passes |
| `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` | MODIFY | Append § Step 6 result to Status |
| `CLAUDE.md` | MODIFY | Append step 6 outcome paragraph |

## Implementation notes

### Solver-side wiring sketch

`forms_logc_muh.py` (mirrors `forms_logc.py`):

```python
manufactured_R_inj = conv_cfg.get("manufactured_R_inj", None)
apply_h_source     = conv_cfg.get("apply_h_source", True)
apply_k_sink       = conv_cfg.get("apply_k_sink",   True)
override_sigma_singh_counts_pm2 = conv_cfg.get(
    "override_sigma_singh_counts_pm2", None,
)

# ΔpKa build — gated by override.
if override_sigma_singh_counts_pm2 is not None:
    pka_shift_expr = build_pka_shift_from_override(
        cation_params=cation_hydrolysis_bundle.cation_params,
        sigma_singh_counts_override=fd.Constant(
            float(override_sigma_singh_counts_pm2),
        ),
        r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
    )
else:
    pka_shift_expr = build_pka_shift(
        cation_params=cation_hydrolysis_bundle.cation_params,
        sigma_S=sigma_S_expr,
        r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
    )

R_net_default = build_proton_boundary_source(
    bundle=cation_hydrolysis_bundle,
    c_M_bdy_expr=c_M_bdy_expr,
    c_H_bdy_expr=c_H_bdy_expr,
    pka_shift_expr=pka_shift_expr,
)
if manufactured_R_inj is not None:
    R_net = fd.Constant(float(manufactured_R_inj))
else:
    R_net = R_net_default

lam_func = cation_hydrolysis_bundle.lambda_hydrolysis_func
if apply_h_source:
    F_res -= (
        lam_func * R_net
        * v_list[h_idx_for_cation]
        * ds(electrode_marker)
    )
if apply_k_sink:
    F_res -= (
        lam_func * (-R_net)
        * v_list[cation_hydrolysis_bundle.counterion_idx]
        * ds(electrode_marker)
    )
```

When `apply_h_source=False`, the proton residual contribution is
completely omitted (not zeroed by multiplication — the term is not
added at all).  Same for `apply_k_sink`.  This is robust to any
finite-precision issues from `Constant(0.0)` propagation through
the form.

### `build_pka_shift_from_override` helper

```python
def build_pka_shift_from_override(
    *,
    cation_params,
    sigma_singh_counts_override,  # ufl.Constant in counts/pm²
    r_H_El_func,
):
    """Build ΔpKa with σ_singh pinned to a constant.

    Bypasses the PNP/Stern coupling for ΔpKa ONLY — the residual-side
    R_net coupling to σ_S (via `build_proton_boundary_source`) is
    UNCHANGED.  Used by A3 to verify σ-mapping independence.
    """
    A = fd.Constant(cation_params.A_pka_per_count_pm2)
    z = fd.Constant(cation_params.z)
    r_M_O_pm = fd.Constant(cation_params.r_M_O_pm)
    return (
        2.0 * A * z * sigma_singh_counts_override * r_H_El_func
        * (1.0 - r_M_O_pm**2 / r_H_El_func**2)
    )
```

This is the formula form from `build_pka_shift` minus the
σ_signed-counts → σ_singh anode clamp (because the override is
already a non-negative counts value).  Verify against
`docs/phase6/singh_2016_pka_formula.md` §4.1 before implementing.

### Wall-time budget

Per A.2 (1300s = 22 min for warm-walk + 10 k_hyd ramps), step 6's
expected wall:
* Pass 1 (anchor + warm-walk): same as A.2 ≈ 850 s.
* Pass 2 (4 ablations + 3-point R_inj bracket pre-pass = 7 ramps):
  ≈ 7 × 40 s = 280 s.
* Total: ≈ 1130 s ≈ 19 min.

Conservative budget: 30 min wall, includes 50% margin.

### Routing decisions

| Outcome | Action |
|---|---|
| All 4 pass (A0+A1+A2+A3) | Plumbing verified.  Proceed to step 7 (CMK-3 lit note) → step 8 (v10b). |
| A0 fails byte-equivalence | New flag plumbing broke v9/v10a/v10a' byte-equivalence.  Debug residual-side wiring; do NOT proceed. |
| A1 fails (c_H doesn't rise) | Proton boundary source wiring may be silently dropping the source term.  Debug `build_proton_boundary_source` + `apply_h_source` gate. |
| A2 fails (c_K doesn't fall) | K+ sink wiring may be mis-signed or dropping the term.  Debug `apply_k_sink` gate. |
| A3 fails (ΔpKa not pinned to override) | `build_pka_shift_from_override` doesn't bypass PNP coupling cleanly.  Debug. |
| A3 fails (PNP σ_S substantially perturbed by override) | The override is leaking into the residual side somehow.  Debug. |

A3-fail "PNP σ_S substantially perturbed" with `|Δσ_S/σ_S| > 10%`
indicates the override is coupling back into the FE residual,
which would mean the override is NOT a clean ablation — that's a
plumbing bug that v10b MUST know about before doing any calibration.

### Per-ablation record schema

```json
{
  "ablation_id": "A0" | "A1" | "A2" | "A3",
  "ladder_converged": true,
  "exception_phase": null,
  "lambda_one_rung": {
    "lambda_hydrolysis": 1.0,
    "snes_converged": true,
    "picard_status": "converged",
    "k_hyd": 1e-3,
    "k_des": 1.0,
    "gamma": ...,
    "gamma_max": 0.047,
    "theta": ...,
    "c_H_boundary_avg": ...,
    "c_K_boundary_avg": ...,
    "sigma_S_C_per_m2": ...,
    "sigma_singh_counts_active": ...,
    "pka_shift_avg": ...,
    "F0_decomposition": {...},
    "R_2e_current_nondim": ...,
    "R_4e_current_nondim": ...,
    "cd_mA_cm2": ...,
    "x_2e": ..., "x_4e": ...,
    "H2O2_selectivity_pct": ...,
    "mass_balance_residual_rel": ...
  },
  "ablation_flags": {
    "apply_h_source": ...,
    "apply_k_sink": ...,
    "override_sigma_singh_counts_pm2": ...,
    "manufactured_R_inj": ...
  },
  "deltas_vs_A0": {
    "delta_c_H_rel": ...,
    "delta_c_K_rel": ...,
    "delta_sigma_S_rel": ...,
    "delta_pka_shift_abs": ...
  },
  "pass": true,
  "pass_reason": "all gates passed" | <fail reason>
}
```

Top-level:
```json
{
  "config": {...},
  "lambda_zero_baseline_at_v_kin": {...},
  "r_inj_bracket_prepass": {
    "values_tested": [...],
    "delta_c_H_rel_per_value": [...],
    "selected_value": ...
  },
  "ablation_records": [A0, A1, A2, A3],
  "overall_pass": ...,
  "routing_decision": "proceed_to_step_7" | <debug step>
}
```

## Verification

### Fast tests (no Firedrake)

```bash
source ../venv-firedrake/bin/activate
python -m pytest tests/test_phase6b_step6_plumbing_ablation.py -v
```

Covered:
* CLI parsing (`--r-inj-manufactured`, `--sigma-singh-override`,
  `--ablations A0,A1,A2,A3`).
* `classify_ablation_status` boundary cases per ablation.
* `_build_ablation_sp_overrides` returns the right ctx flags per
  ablation_id.
* Default flags preserve byte-equivalence (UFL form comparison).

### Slow tests (Firedrake required, marked `slow`)

* `apply_h_source=False` zeroes the H+ residual: UFL form-level
  asserts the proton-residual coefficient is exactly 0 at the
  electrode marker.
* `apply_k_sink=False` zeroes the K+ residual: same form-level check.
* `override_sigma_singh_counts_pm2=X` → `pka_shift` UFL ≡ `β_K_Cu · X`
  symbolically (form comparison).
* End-to-end: A0 reproduces A.2 baseline at k_hyd=1e-3 within rel
  1e-12 byte-equivalence.

### Re-run

```bash
python -u scripts/studies/phase6b_step6_plumbing_ablation.py \
    --v-kin -0.10 --k0-r4e-factor 1e-14 \
    --r-inj-prepass 1e-2,1e-1,1.0 \
    --sigma-singh-override <resolved_counts_pm2_value> \
    --out-subdir phase6b_step6_plumbing_ablation
```

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **σ-counts conversion factor ambiguity** (Risk #11 from R0 draft) | Resolve in plan before A3 runs.  Cross-check 6.243e-6 in acceptance bundle line 159 against `Forward/bv_solver/units.py:sigma_C_m2_to_counts_pm2`.  If they disagree, fix one (the units helper is the authoritative source). |
| 2 | Manufactured `R_inj_const = 1e-2` may not produce 5% c_H shift | Bracketed pre-pass with 3 values; escalate to 1e-1 / 1.0 if needed.  Hard ceiling 10.0 (250× A0 R_net) to avoid Newton instability. |
| 3 | `apply_h_source=False` causes Newton instability | Run A1 (apply_h_source=True, apply_k_sink=False) before A2 (apply_h_source=False, apply_k_sink=True) so any FE singularity surfaces with diagnostic context.  If A2 won't converge, the unphysical break is the cause; mitigate by adding a tiny `H+ Robin term` for A2 only (NOT a real fix, just enables the ablation). |
| 4 | A3's override leaks into the residual via `R_net = build_proton_boundary_source(...)` since `R_net_default` uses `pka_shift_expr` | This is BY DESIGN — A3 overrides ΔpKa, which propagates into R_net.  Pass criterion is `|Δσ_S| < 10%` to confirm the leak doesn't significantly perturb the PNP-solved σ_S.  Document this as expected coupling. |
| 5 | Byte-equivalence test fails because new flag defaults break v9 (e.g., `None`-vs-missing distinction in `conv_cfg.get`) | Use `.get(key, default)` everywhere; explicit `None` is the "off" value (matches `manufactured_R_inj` precedent); test asserts rel < 1e-12 vs committed A.2 baseline. |
| 6 | A1/A2 break charge conservation (R goes into H+ but not K+) → solver complains | This IS the manufactured ablation; charge balance is intentionally broken.  Newton + linear solver should still converge because the H+ boundary residual is changed but the bulk Dirichlet BC pins the species mass.  If Newton diverges, escalate to a smaller `R_INJ_MANUFACTURED`. |
| 7 | Forms-logc-vs-muh wiring inconsistency | Mirror the wiring in both files identically; add a doc-comment cross-reference; the test suite should cover both. |
| 8 | Pre-pass bracket determination is wasted wall time | At most 3 extra ramps × ~40s = 2 min.  Cheap insurance vs. failing the 5% criterion mid-run. |
| 9 | A3 needs `SIGMA_SINGH_K_CU_OVERRIDE` resolved; ambiguity blocks the whole run | Resolve in plan; if still ambiguous, use a sensitivity bracket (3 values) and pick the median.  Document. |
| 10 | A2's "apply_h_source=False, apply_k_sink=True" + `manufactured_R_inj` creates a one-sided sink that drives c_K below bulk — possibly into negative concentration (unphysical) | Cap c_K(0) at 1% of bulk via a soft clamp in the residual OR drop `R_INJ_MANUFACTURED` if A2's c_K goes negative.  Document the cap mechanism in test fixtures. |

## After Step 6 lands

Per the locked sequence:

* Step 7 — CMK-3 capacitance literature note (mostly landed via
  `.research/cmk3-stern-capacitance/SUMMARY.md`; small lift to
  finalize `docs/phase6/CMK3_capacitance_literature.md`).
* Step 8 — v10b literature calibration of Γ_max + k_des + C_S
  (~1-2 weeks).  v10b inherits step 6's `routing_decision`
  outcome: if step 6 passes, v10b proceeds without plumbing
  caveats.  If step 6 fails any ablation, v10b is BLOCKED on
  the failing flag's bug fix.

## Config invariant (HARD precondition)

Step 6, v10b, and B.2 share:
* `K0_R4e_factor = 1e-14`
* `C_S = 0.20 F/m²`
* `V_kin = -0.10 V`
* `k_hyd_baseline = 1e-3` (the v10a' / A.2 reference)

If v10b's literature calibration motivates a different
`K0_R4e_factor`, step 6 + v10a' + A.2 must re-run before B.2.
Documented in `CLAUDE.md`, the acceptance bundle, and as a B.2
driver assertion.

## References

* `~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md` — A.2 plan.
* `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.json` —
  A.2 baseline record.
* `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` § "v10a
  → E sequence" step 6 — locked spec.
* `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` §6 —
  A1/A2/A3 description (original).
* `docs/phase6/singh_2016_pka_formula.md` — Singh 2016 SI Eq. (4)
  + §5.2 σ-mapping conventions.
* `docs/handoffs/CHATGPT_HANDOFF_30_phase6b-v9-cd-invariance/` —
  Session 30 critique that proposed apply_h_source / apply_k_sink
  / override_pka_sigma_S flags (originally R3 #8 + R5 #6).
* `Forward/bv_solver/cation_hydrolysis.py:728-749` — current proton
  + K residual wiring (where the new gates land).
* `Forward/bv_solver/forms_logc_muh.py:697-700` — current
  `manufactured_R_inj` plumb point.
* `Forward/bv_solver/units.py` — `sigma_C_m2_to_counts_pm2` helper.
* Memory: `project_v10a_phase_A2_outcome`,
  `project_v10a_prime_outcome`.

## Critique provenance

To be hardened by GPT critique loop.  Provenance directory:
`docs/handoffs/CHATGPT_HANDOFF_35_phase6b-step6-plumbing-ablation/`
(to be created at start of critique loop).

---PLAN END---

## Section 3 — Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
