# Final revision — Critique session 35 (Phase 6β Step 6)

## Artifact revised

`~/.claude/plans/phase6b-step6-plumbing-ablation.md`

## Final verdict

**APPROVED at R5** (round-cap, but final verdict APPROVED, not
cap-hit-with-ISSUES_REMAIN).  GPT: *"No blocking issues remain.
The revised plan now directly checks the physical source magnitude,
manufactured source/sink response, override diagnostics, Picard
closure, and residual-side A3 closure."*

## Issue ledger across 5 rounds (54 total)

### Addressed (all 54 accepted, applied to revised plan)

**R1 issues (17 accepted)**

1. **A3 override not wired through Picard + diagnostics** → Centralized via ctx-stored `_cation_hydrolysis_pka_shift_expr`; all three consumers (forms, Picard, diagnostics) read from it.  Lands in plan §Solver-side wiring + §Centralized override consumers.
2. **A3 self-contradictory on coupling** → Re-framed A3 as residual-path imposed-σ ablation; A3 §Scope clarifies override propagates into R_net via build_proton_boundary_source.
3. **A1/A2 can't catch broken physical R_net path** → Added A0b ablation (physical-path residual-assembly sanity); see plan §A0b spec.
4. **Manufactured path still builds R_net_default** → Skip R_net_default build when manufactured_R_inj set; plan §Solver-side wiring code snippet.
5. **apply_h_source/apply_k_sink not wired to Picard** → Cross-validation rule: half-flags require manufactured_R_inj; plan §Cross-validation.
6. **A.2 diagnostics invalid for A1/A2** → Manufactured-aware `collect_v10a_rung_diagnostics`; plan §Centralized override consumers.
7. **σ conversion repo inconsistencies** → Prerequisite P1 audit; plan §Prerequisites.
8. **Singh deck override tiny** → Split `SIGMA_SINGH_PLUMBING_SENTINEL = 0.022` (large response) vs `SIGMA_SINGH_K_CU_DECK = 1.41e-5` (unit audit); plan §Five ablations.
9. **build_pka_shift_from_override wrong parameter names** → Use fake-signed σ_S = -override/6.2415e-6 through existing `build_pka_shift`; no new helper.  Plan §Solver-side wiring.
10. **A3 pass criterion tautological** → Added residual-side gate 6 (mass-balance closure); plan §A3 final pass criteria.
11. **5%/1% thresholds not defensible** → Reframed with sign + magnitude + convergence + severity; plan §Pass criteria.
12. **R_inj bracket only A1, no upper bound** → Joint A1+A2 brackets, 5-25% window; plan §Pre-pass.
13. **Soft clamp on c_K bad mitigation** → Dropped; λ continuation instead; plan §Risk #5.
14. **Byte-equivalence 1e-12 too brittle** → Tiered tolerance table; plan §A0 pass criteria.
15. **Fast tests overclaim** → Reclassified: pure-Python in `tests/...py`, Firedrake-required in `tests/..._slow.py`.
16. **Bool / override validation underspecified** → Validation in `config.py` with `_bool` + finite/≥0 checks.
17. **A3 σ_S leak rule not clean detector** → Replaced with structural test on `_cation_hydrolysis_pka_shift_expr`.

**R2 issues (14 accepted)**

1. **A3 sign analysis wrong** → Corrected: at V_kin, σ_S<0 ⇒ σ_singh>0 ⇒ ΔpKa<0 (cathodic lowering).  Plan §Prerequisites P2.
2. **Sentinel direction backward** → SIGMA_SINGH_PLUMBING_SENTINEL=0.022 gives `pka_factor ≈ 10.07` (amplifies, doesn't suppress).  Plan §Five ablations.
3. **A0b 1% c_H gate unsupported by A.2 data** → A0b redesigned as residual-assembly test (scalar fluxes); plan §A0b spec.
4. **Flux-magnitude → c_H 1% logic dimensionally wrong** → Dropped; A0b uses mass-balance closure on assembled flux scalar.
5. **ctx-runtime-override won't work for built UFL forms** → Drop runtime accessor; rebuild forms per ablation (existing SolverParams pattern).
6. **A3 capped R_net comparison wrong** → A3 gates use uncapped pka_factor + Picard prediction.
7. **pka-factor formula garbled** → Direct comparison `amp_from_singh ≈ pka_factor_avg` in override mode.
8. **R_inj bracket only searches upward** → Bidirectional 5-value bracket starting at 1e-4.
9. **Same R_INJ for A1 and A2 brittle** → Separate `R_INJ_MFG_A1` and `R_INJ_MFG_A2` sentinels.
10. **Driver-only validation too weak** → Validation in `config.py`.
11. **collect_v10a called by solver before driver** → Manufactured-aware diagnostic.
12. **UFL structural test overbroad** → Inspect `_cation_hydrolysis_pka_shift_expr` specifically.
13. **1% single-run noise floor asserted not established** → Dropped; A0/A0b use residual-assembly, A1/A2 use 5-25% large-signal sentinels.
14. **Audit branch logic missing** → P3 explicit decision tree.

**R3 issues (12 accepted)**

1. **A0b doesn't test actual residual wiring** → Store canonical ctx artifacts; A0b assembles those.
2. **`R_net * v_h * ds` not a scalar** → Use `R_net * ds` for scalar; vector inspection separated.
3. **Anti-symmetry tautological if I build R_net and -R_net** → Read from canonical stored terms; tautology IS the point (consistency check).  Labeled as such in R5.
4. **`|phys_h_flux| > 1e-30` too weak** → Replaced with mass-balance closure (rel 5e-3).
5. **Wrong ctx path for bv_convergence** → Use `ctx.get("bv_convergence", {})`.
6. **A3 gate 4 `Γ_A3 > Γ_A0` too weak** → Replaced with Picard-prediction match (rel 1e-3 / Γ_max).
7. **A3 gate 3 wrong denominator** → Direct `amp_from_singh_A3 ≈ pka_factor_avg_A3` (no ratio).
8. **Gates 1+2 don't prove override-in-form** → `collect_v10a` consumes ctx-stored `_pka_shift_expr`.
9. **"Only Constants" language wrong** → Test specifically asserts `ctx["U"] not in coeffs`.
10. **`_sigma_S_active_expr` ambiguous** → Split into `_sigma_S_expr` (solved) and `_pka_sigma_S_expr` (fake-signed in override mode).
11. **Positivity tautological in log-c** → Replaced with severity gate `c_K > 0.01·c_K_bulk` + finite-check.
12. **rel 1e-9 too strict for A0** → Restored tiered table.

**R4 issues (7 accepted)**

1. **A3 doesn't prove override reached actual R_net in F_res** → Added A3 gate 6 (mass-balance closure via `_R_net_scalar_form`).
2. **"Override-in-form" overstated** → Reworded: gates 1+2 prove override-in-stored-pKa, gate 6 proves override-in-residual.
3. **Manufactured diagnostics missing top-level c_K_boundary_avg** → Always emit `c_H_boundary_avg` + `c_K_boundary_avg` top-level.
4. **"All diagnostics finite" underspecified** → Per-ablation `REQUIRED_NUMERIC_KEYS` dict.
5. **Cofunction.subfunctions API uncertain** → Drop vector slot inspection; use scalar forms only.
6. **DOF sums fragile proxy** → Scalar forms primary; vector inspection optional.
7. **`bundle.k_hyd_func.values()[0]` wrong pattern** → Use `float(bundle.k_hyd_func)`.

**R5 nits (4 incorporated, all non-blocking)**

1. **Label A0b gates 2-4 as consistency checks** → Plan §A0b table labels them; gate 1 is "load-bearing", 2-4 are "stored-artifact consistency checks".
2. **Slot-wiring slow test should inspect stored terms first** → Plan §Tests: `extract_arguments` on `_H_residual_term` / `_K_residual_term` first; full-F_res inspection secondary.
3. **REQUIRED_NUMERIC_KEYS dotted-path support** → `_get_dotted` helper + `F0_decomposition.amplification_from_singh` in A3 key list.
4. **ctx artifacts staged as locals then inserted** → Plan §Solver-side wiring shows staging via `_step6_artifacts` dict then `ctx.update`.

### Defended (none)

No issues were defended.  All 54 accepted.

### Unresolved (none)

All addressed.

## Where to start implementation

1. **Prerequisites first**: P1 σ-conversion audit + P2 Singh sign
   audit + P3 decision-tree.  ~1-2 hours.  Decides whether v10a' +
   A.2 need re-running.
2. **Solver-side plumbing** (Forward/bv_solver/forms_logc[_muh].py,
   config.py, cation_hydrolysis.py, anchor_continuation.py): add
   the three new ctx flags, canonical stored artifacts, and
   manufactured-aware diagnostics.
3. **Driver + tests** (scripts/studies/, tests/): the new step 6
   driver + pure-Python + Firedrake-slow tests.
4. **Run step 6** at V_kin = -0.10 V with `K0_R4e_factor=1e-14`,
   `C_S=0.20`, `k_hyd_baseline=1e-3`.  Wall ~24 min (35 min margin).
5. **Update CLAUDE.md + acceptance bundle** with step 6 outcome.

## Files

| Path | Status | Role |
|---|---|---|
| `R1_to_gpt.md` ... `R5_to_gpt.md` | committed | Round-by-round counterreplies |
| `R1_from_gpt.md` ... `R5_from_gpt.md` | committed | GPT responses |
| `.codex_log_R*.txt` | committed | Raw codex output |
| `.codex_session_id` | committed | Persistent session UUID |
| `STATUS.md` | committed | Live state |
| `FINAL_REVISION.md` (this file) | committed | Ledger |
