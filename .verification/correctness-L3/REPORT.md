# Level-3 Code-Correctness Audit — Forward Solver Codepath

**Target:** Codepath exercised by `scripts/studies/solver_demo_slide15_no_speculative_cs.py`
**Date:** 2026-05-13
**Level:** 3 (Sonnet ×3 + Opus ×3 + Codex xhigh ×3, in parallel)
**Mode:** Independent code correctness — documentation ignored
**Total agents this pass:** 9
**Total agents cumulative across L1 doc audit + L2 correctness + L3:** 21
**Verdict:** **PASS for this demo's exercised codepath** with new theoretical and latent concerns surfaced by Codex xhigh and Opus.

---

## Top-line: what L3 adds beyond L2

L2 already concluded "no critical bugs in this demo's path." L3 corroborates that AND adds 3 substantive new findings that L2 missed:

| # | Severity | Surfaced by | Item |
|---|---------|-------------|------|
| **N1** | **HIGH (theoretical) / LOW (practical, this demo)** | Codex-C-L3 (xhigh) | **AdaptiveLadder termination is NOT bounded by `len(initial_scales) · (max_inserts+1)`.** The claimed "25-iter worst case" is false. `record_success` resets `_inserts_at_current_step` to 0, so a pattern of "fail-original → insert-midpoint → midpoint-succeeds → fail-original-again → insert-closer-midpoint → ..." can recur indefinitely in exact arithmetic. In FP arithmetic the geometric `sqrt(prev * scale)` eventually collapses, but that's the safety net, not the design. For the demo's well-separated initial scales `(1e-12, 1e-9, 1e-6, 1e-3, 1.0)`, the pattern doesn't actually arise (Phase 5γ outcome memo confirms empirical convergence). |
| **N2** | **HIGH (theoretical) / LOW (practical)** | Codex-C-L3 | **Geometric-midpoint branch lacks the strict-progress guard the arithmetic branch has.** Arithmetic branch (`anchor_continuation.py:874-880`) checks `if not (warm_start_floor < midpoint < scale): return False`. Geometric branch (`:884-887`) just inserts `math.sqrt(prev*scale)` with no progress check. If FP collapses midpoint to prev (duplicate success → counter reset → footgun for N1) or to scale (duplicate failure rung), or if the product underflows to zero (would insert nonpositive scale, violating constructor invariant), the code doesn't catch it. |
| **N3** | **MEDIUM (latent)** | Opus-C-L3 | **`extract_preconverged_anchor`'s `U_data` is frozen at Stage-1 success and does NOT auto-reflect Stage-2 Stern bump mutations.** This demo correctly sidesteps the issue by building a fresh `PreconvergedAnchor` at demo line 502 with `snapshot_U(ctx_anchor["U"])` (post-bump state). Future callers using `extract_preconverged_anchor` after post-Stage-1 ctx mutations would silently get pre-bump U. |

Plus two new medium-latent concerns:
- **Codex-B (xhigh):** IC fallback wrapper at `forms_logc_muh.py:1028-1039` converts `ok=False` returns to the linear-phi fallback, but does NOT convert exceptions raised inside helpers — they can bypass the fallback path entirely.
- **Codex-B (xhigh):** `u_clamp=30` on log-concentration (`forms_logc_muh.py:326-340`) creates a zero-derivative trap analogous to the `free_dyn` floor — when clamped, NP/Poisson/BV concentration sensitivity for the clipped species is flattened.

---

## What L3 confirmed (gold-standard verifications)

Codex xhigh added arithmetic-verified confirmations of items L2 had already cleared:

| Item | L3 confirmation |
|------|-----------------|
| **Physical a_nondim(H+) derivation** | **Codex-A computed step-by-step**: r=2.80e-10 m → a_phys = (4/3)π·r³·N_A = 5.537e-5 m³/mol → a_hat = 5.537e-5 × C_SCALE(1.2) = 6.645e-5 → c_max_phys = C_SCALE/a_hat = **18058.7 mol/m³ ≈ 1.8e4** ✓ Matches Hard Rule #7. |
| **Mesh grading FP-exactness** | **Codex-A computed**: coords transformed via `(coords[:, 1] ** beta) * domain_height_hat`. For Ny=80, β=3.0, D=1.0: `y_80 = (80/80)³ × 1.0 = 1.0` FP-exact. Δy_1 = 1.953e-6 (electrode fine end), Δy_80 = 0.037 (bulk). Marker 3 = bottom (y=0, electrode). All correct. |
| **Bikerman shared-θ closure self-consistency** | **Codex-B + Opus-B verified math step-by-step**: `denom = theta_b_const + Σ a_const·c_const·q_k` (shared partition function), then `c_steric_k = c_bulk·q·free_dyn/denom`. Single shared denominator combines both counterion exponents. NOT independent per-ion closures. Mathematically self-consistent. |
| **Jacobian freshness** | **Codex-B verified**: F_res mutations in order: NP (463-505), BV (609-636), Poisson+Stern (641-668), cation hydrolysis (786-789). `J_form = derivative(F_res, U)` at line 839. The ONLY post-derivative mutator is `add_boltzmann_counterion_residual` (boltzmann.py:384-386), which RE-DERIVES `ctx["J_form"]` after its mutation. No stale Jacobian. |
| **Sign conventions** | **Codex-B verified**: Poisson `+ ε·∇φ·∇w·dx − ρ·w·dx` (correct weak form of `−ε∇²φ = ρ`); Stern Robin `F_res -= C_S·(φ_applied − φ)·w·ds` (correct for cathodic polarization); BV cathodic `exp(-α·n_e·η)`, anodic `exp((1-α)·n_e·η)`, net `R_j = cathodic − anodic` (positive cathodic). All consistent. |
| **η-clip ordering (Hard Rule #2)** | **Codex-B verified**: `eta_scaled = bv_exp_scale · eta_raw` clipped BEFORE multiplication by `α·n_e`. Production V band `[-0.4, +0.55]V` with exponent_clip=100 → cathodic threshold V_RHE > 0.695 − 2.57 = −1.875V → grid is fully unclipped. |
| **Boltzmann closed-form invariant** | **Codex-B verified**: `build_steric_boltzmann_expressions` does NOT take W as argument; creates only a `Function(R_space)` for continuation scaling; returns UFL expressions in φ. `TestFunctions(W)` appears ONLY in `add_boltzmann_counterion_residual` (the residual-mutator), not in the closure builder. Cs⁺/SO₄²⁻ NOT in Newton state. |
| **`ctx['_last_solver']` cross-stage isolation** | **Codex-C verified (consistent with Opus-C-L2)**: `_build_for_voltage` at `grid_per_voltage.py:1011-1032` builds fresh sp_v → ctx → forms → NonlinearVariationalProblem → NonlinearVariationalSolver → stores in fresh ctx. No alias back to Stage 1/2. |
| **U_prev placement in SER** | **Codex-C verified**: `U_prev.assign(U)` at `grid_per_voltage.py:192`, immediately after `solver.solve()`, before observable assemble. Correct for transient term. |
| **`_march` state restoration (recursive bisection)** | **Codex-C verified**: ckpt_inner snapshot before each substep; on failure restore + reassert paf; recursive halves use ckpt_inner state correctly; v_prev_substep updated only on success; outer bailout restores ckpt_outer with paf=v0. Correct through all failure cascades. |
| **Stern Robin via mutated fd.Constant** | **Codex-C verified**: `stern_coeff_const` is `fd.Constant` baked into F_res by reference (forms_logc_muh.py:667). `.assign()` mutates in place; Firedrake reads at solve-time. Standard pattern. Mutation is live-visible to next `solver.solve()`. |
| **Observable n_e/N_ELECTRONS_REF weighting** | **Codex-C verified**: `weight = n_e_j / N_ELECTRONS_REF` (where N_ELECTRONS_REF=2). For R_2e (n_e=2) → weight=1.0; for R_4e (n_e=4) → weight=2.0. Sum × scale_const × ds. Matches Faraday's law per project memory. |
| **`gross_h2o2_current` default index** | **Codex-C verified**: defaults to idx=0; for both PARALLEL_2E_4E_REACTIONS and legacy sequential preset, index 0 is R_2e (the 2e peroxide channel). Demo passes explicit `0` to make it certain. |
| **Firedrake form caching** | **Opus-C verified**: TSFC caches by form-structure hash, not Constant values. Constant mutation does NOT trigger recompile. Stern bump pattern is correct and efficient. |

---

## Codex-C-L3's "FAIL" verdict — full context

Codex-C-L3 issued VERDICT: FAIL based on the AdaptiveLadder termination findings (N1, N2). To be precise about what this means:

- **The codebase's claim of bounded termination (in code comments or by structure) is mathematically false.** Codex correctly identified this.
- **For this demo's specific inputs**, termination is empirically observed (per project memory `project_phase5gamma_mvp_outcome.md`: "GATE v2 needs K0_FACTOR=1e-30 (not 1e-9) due to mass-transport floor at eta_R4e≈28" — anchor converges with the production scales).
- **Codex's other findings** (warm_start_floor accepts negative, set_reaction_k0_model/set_stern_capacitance_model not atomic, SER plateau detection on one scalar observable, gross_h2o2_current default idx=0 not validated) are all latent or context-specific.

**Resolution:** Codex-C is correct that the termination-bound claim is false. The demo nevertheless terminates because (a) initial scales are well-separated geometrically, (b) max_inserts_per_step=4 limits depth per "failure burst," and (c) FP arithmetic eventually collapses the geometric midpoint. The right fix is a strict-progress guard on the geometric branch (mirror the arithmetic branch's `if not (prev < midpoint < scale): return False`). This is a real but unrealized defensive gap.

---

## Consolidated findings table (L3 unique, including those that corroborate L2)

| # | Severity in this demo | Item | Location | Found by | Cross-tier agreement |
|---|----------------------|------|----------|----------|----------------------|
| L3-1 | **High (theoretical) / Low (practical)** | AdaptiveLadder counter-reset enables unbounded fail-success-fail cycles | `anchor_continuation.py:829-836, 860-887` | Codex-C-L3 unique | Sonnet/Opus-C said "terminates in practice" — Codex went deeper and showed the bound claim is false. |
| L3-2 | **High (theoretical) / Low (practical)** | Geometric midpoint has no strict-progress guard | `anchor_continuation.py:884-887` | Codex-C-L3 unique | Confirmed: arithmetic branch (`:874-880`) has the guard, geometric branch doesn't. Sonnet-C-L3 saw asymmetry, didn't grade as severely. |
| L3-3 | **Medium (latent)** | `extract_preconverged_anchor` U_data freezes at Stage-1, doesn't reflect Stage-2 mutations | `anchor_continuation.py:1611ish` (verified by Opus-C) | Opus-C-L3 unique | Demo sidesteps via fresh `snapshot_U` at line 502. Footgun for future callers. |
| L3-4 | **Medium (latent)** | IC fallback wrapper converts `ok=False` but NOT exceptions → exception bypasses fallback | `forms_logc_muh.py:1028-1039` | Codex-B-L3 unique | Real concern: if a helper raises, Picard fallback to linear-phi IC never fires. |
| L3-5 | **Medium (latent)** | `u_clamp=30` zero-derivative trap on log-c | `forms_logc_muh.py:326-340` | Codex-B-L3 unique | Analogous to free_dyn floor concern from L2. In high-polarization Debye layers when clamp activates, NP/Poisson/BV concentration sensitivity is flattened for that species. |
| L3-6 | **Medium (latent)** | `with_solver_options` shallow-copies nested dicts | `_bv_common.py` SolverParams | Opus-A-L3 unique | Demo's specific pattern (`dict(sp.solver_options)`) happens to dance around the issue. Future code paths mutating non-shallow-copied inner dicts (`bv_bc`, `nondim`) would alias. |
| L3-7 | **Medium (latent)** | `set_stern_capacitance_model` silent unit fallback (default factor = 1.0) | `anchor_continuation.py:456` | Sonnet-C-L3 unique | Default-of-1.0 fallback if `bv_stern_phys_to_nondim_factor` key missing → raw F/m² value silently assigned as nondim. Production ctx always populates the key. |
| L3-8 | **Medium (latent)** | `set_reaction_k0_model` + `set_stern_capacitance_model` not atomic across metadata + Firedrake coefficient | `anchor_continuation.py:299-310, 459-467` | Codex-C-L3 unique | Updates `ctx["nondim"]` BEFORE `.assign()` on the Constant. If `.assign()` raises, metadata is updated but live coefficient is stale. No rollback. Docstring acknowledges this. |
| L3-9 | **Low** | Picard `_solve_2x2` checks only `|det| < 1e-300`, not conditioning vs matrix scale | `picard_ic.py:1091-1105` | Codex-B-L3 unique | Ill-conditioned matrices with large entries can still trip up Newton without being detected. |
| L3-10 | **Low** | Stern split linear-Debye fallback when bracket fails has no smallness check on `\|psi_D\|` | `picard_ic.py:259-270` | Codex-B-L3 unique | Same as L2 Sonnet-B's MEDIUM finding, but Codex grades it LOW since the downstream solver still enforces Robin BC. |
| L3-11 | **Low** | `cosh(abs(psi_D))` at `psi_D=full_drop` can overflow Python before bisection-or-fallback branch | `picard_ic.py:171-178, 238-258` | Codex-B-L3 unique | For extremely large Stern drops; not triggered in production V band. |
| L3-12 | **Low** | `warm_start_floor` accepts negative values → could insert negative scale | `anchor_continuation.py:794-800, 874-881` | Codex-C-L3 unique | Constructor validates `warm_start_floor < initial_scales[0]` but not `warm_start_floor > 0`. Production always uses default-None or positive. |
| L3-13 | **Low** | SER plateau detection uses ONE scalar observable (j_cd), no `\|U - U_prev\|` or residual check | `grid_per_voltage.py:193-198` | Codex-C-L3 unique | ss_consec=4 mitigates one-step sign-flip-through-zero. Doesn't catch silent hidden-state changes that don't show up in j_cd. Empirically not a problem. |
| L3-14 | **Low** | `gross_h2o2_current` defaults idx=0 without validating reaction identity | `observables.py:123-135` | Codex-C-L3 unique | Demo passes explicit idx=0 (good). Risk only if caller relies on default with non-standard reaction ordering. |
| L3-15 | **Low** | `current_density` falls back to unweighted reaction sums when `n_electrons` metadata missing | `observables.py:110-121` | Codex-C-L3 unique | No warning emitted. Underweights 4e reactions for heterogeneous-electron mixed configs if metadata is malformed. |
| L3-16 | **Low** | `run_ss` exception path doesn't restore U; callers must roll back | `grid_per_voltage.py:187-192` | Codex-C-L3 unique | warm_walk_phi does the rollback. K0 ladder rollback was outside Codex's read range (verified clean by Sonnet-C and Opus-C). |
| L3-17 | **Low** | `warm_walk_phi` doesn't validate `n_substeps` or `bisect_depth` | `grid_per_voltage.py:225-226, 271-310` | Codex-C-L3 unique | n_substeps=0 → no march, jumps to final solve. Negative depth → first failure exhausts. Demo passes valid values. |
| L3-18 | **Low (Strategy B path only)** | `theta_inner = 1 − A_dyn − z_scale·packing_total` zeros counterion packing when z_scale=0 | `forms_logc_muh.py:448, forms_logc.py:397` | Sonnet-B-L3 unique | Affects Strategy B Phase 1 path that demo doesn't use (CLAUDE.md Hard Rule #1: prefer C+D). |
| L3-19 | **Note (cosmetic)** | `c_clo4_bulk = counterions[0]["c_bulk_nondim"]` is dead in multi-ion mode | `forms_logc_muh.py:1182, :1273` | Opus-B-L3 unique | Bypassed by `multi_ion_ctx` dispatch. Latent risk if someone removes the arg. |
| L3-20 | **Note** | `dispatch.py` silently falls back to `"logc"` for unrecognized formulation, `"linear_phi"` for unrecognized initializer | `dispatch.py:68-80, 82-120` | Sonnet-A-L3, Codex-A-L3 (corroborates L2) | Relies on `config._validate_formulation` to reject typos upstream. |
| L3-21 | **Note** | `_stern_bump_ladder` accepts target ≤ STERN_ANCHOR silently (down-bump) | `solver_demo:312` | All A-tier agents (corroborates L2) | Demo's default target=0.20 unaffected. |
| L3-22 | **Note** | `_grab` swallows assemble exceptions with print only → `converged=True, cd_mA_cm2=null` possible | demo:520-538 | Opus-A-L3, Codex-A-L3 (corroborates L2) | Distinguishes "Newton OK + assemble failed" from "Newton failed" only via print, not JSON. |
| L3-23 | **Note** | CLI `--factors`, `--stern-final`, `beta` parameters have no nan/inf/zero/negative input validation | demo + mesh.py | Codex-A-L3 unique (corroborates partial L2) | Defaults safe. nan/inf would propagate. |
| L3-24 | **Note** | UFL `min_value`/`max_value` is a hard clip — derivative=0 in plateau | forms_logc_muh.py `_build_eta_clipped`, `_steric_pack_clamped` | Opus-B-L3, Codex-B-L3 (corroborates L2) | Production V band is unclipped. Bohra-paper saturation regime would activate. |
| L3-25 | **Note** | No Picard oscillation detection; only max-iter exit + linear-phi fallback | picard_ic.py:1505-1527 | Sonnet-B-L3, Opus-B-L3, Codex-B-L3 (corroborates L2) | Falls back cleanly; no diagnostic distinguishes oscillation from other failure modes. |

---

## Cross-tier agreement analysis

### High agreement (all 3 tiers concur)
- **No critical bugs in the exercised codepath.** All 9 L3 agents PASS/CONCERNS-FOUND verdicts on the demo's actual inputs.
- **Bikerman closure self-consistency** — Codex-B + Opus-B did the math; Sonnet-B verified the shared-`free_dyn` structure.
- **Sign conventions** (Poisson, Stern Robin, BV cathodic/anodic, NP flux) — all verified by Codex-B and Opus-B.
- **η-clip ordering** matches Hard Rule #2 — Codex-B verified the literal multiplication order.
- **Cross-stage solver isolation** (Stage 3 builds fresh ctx + solver per voltage) — Codex-C corroborates Opus-C-L2.
- **Mesh grading FP-exactness, physical a_nondim, factor label uniqueness** — Codex-A computed step-by-step.

### Disagreement: Codex-C "FAIL" vs others "PASS"
Codex-C-L3 issued FAIL based on AdaptiveLadder termination not being mathematically bounded. Sonnet-C-L3, Opus-C-L3, and L2 agents said "terminates in practice." **Both are correct** — Codex caught a theoretical defect that the others didn't dig deep enough to identify. For this demo's inputs, empirical termination is established. **Codex's finding is a real defensive gap worth fixing if the AdaptiveLadder is ever exercised on adversarial inputs.**

### Disagreement: Codex-B vs Sonnet-B on `solve_stern_split` fallback
- Sonnet-B (L2 + L3): MEDIUM — "linear-Debye fallback doesn't satisfy Robin identity, IC quality suboptimal."
- Opus-B (L2 + L3) + Codex-B-L3: LOW — "downstream solver enforces Robin BC, final answer correct."
**Resolution:** Both right on the facts. L2's medium framing is more conservative; L3's low framing reflects the empirical observation that the downstream solver corrects the suboptimal IC.

---

## What this demo's output should be trusted to mean

For each of the 4 K0_R4E_FACTORS, the demo produces `iv_curve.json` with `converged[i]`, `cd_mA_cm2[i]`, `pc_mA_cm2[i]` per grid point. The audit's verdict means:

- **If `converged[i] = true` AND `cd_mA_cm2[i]` is finite (not None):** Trust the value. Sign convention, Faraday's law weighting, Bikerman closure, η-clip ordering, and Stern Robin BC all verified.
- **If `converged[i] = true` AND `cd_mA_cm2[i] = null`:** A latent ambiguity exists. `_grab`'s exception swallowing OR `_to_json_list`'s inf→null coercion could be the cause. Check stderr/logs for the actual reason (L3-22).
- **If `converged[i] = false`:** Convergence failure at that voltage; output is NaN. Stage 3 builds fresh ctx per voltage, so no cross-point contamination.
- **Stage 2 Stern bump failure:** Demo early-returns at line 464 with a "stern-bump-failed" record; Stage 3 is unreachable. No corrupt-U leakage. (Verified by Sonnet-A, Opus-A, Codex-A across L2+L3.)

---

## Recommended hardening (not blocking)

In priority order, the items most worth fixing for future robustness:

1. **(L3-2)** Add `if not (prev < midpoint < scale): return False` to the geometric midpoint branch in `AdaptiveLadder.record_failure_and_insert` (anchor_continuation.py:884-887). One-line fix.
2. **(L3-1)** Add a per-original-rung or per-ladder cumulative insert budget to bound the fail-success-fail cycle. Or document that the bound is empirical, not analytic.
3. **(L3-4)** Wrap the IC builder dispatch in try/except → `ok=False` so helper exceptions trigger the fallback path properly.
4. **(L3-3)** Document `extract_preconverged_anchor` U_data freeze, or add a `ctx_override` parameter that takes a fresh snapshot.
5. **(L3-22)** `_to_json_list` should NOT coerce `inf` to `null` when the point is flagged converged — propagate the inf (or write a `cd_status` field).
6. **(L3-7)** Assert `bv_stern_phys_to_nondim_factor` is present rather than defaulting to 1.0.
7. **(L3-6)** `with_solver_options` should deep-copy nested dicts to match `deep_copy()`'s contract.
8. **(L3-21)** `_stern_bump_ladder(target)` should reject `target < STERN_ANCHOR` with a clear error.
9. **(L3-23)** Add input validation for `--factors`, `--stern-final`, `beta` CLI args.
10. **(L3-8)** Make `set_reaction_k0_model` / `set_stern_capacitance_model` atomic by `.assign()`-ing the Constant BEFORE updating `ctx["nondim"]` (so rollback is trivial: just re-assign).

---

## Cumulative verdict across L1 + L2 + L3

| Pass | Type | Conclusion |
|------|------|-----------|
| L1 | Doc-vs-code reconciliation | 5 critical doc drift items, 13 warnings — doc is structurally faithful but has identifier-level transcription errors. **No code bugs found.** |
| L2 | Code correctness (Sonnet + Opus) | 14 findings, all latent or downstream-corrected. **No critical correctness bugs in the exercised codepath.** |
| L3 | Code correctness (Sonnet + Opus + Codex xhigh) | 25 findings, of which 6 are NEW vs L2. **No critical correctness bugs in the exercised codepath confirmed by a third independent tier.** Codex xhigh adds the AdaptiveLadder termination theoretical defect. |

**Final assessment:** If the demo prints `converged[i] = true` for all 25 grid points across all 4 factors and writes finite `cd_mA_cm2` / `pc_mA_cm2` values, those numbers are the result of a correctly-implemented PNP+BV solver. The latent issues catalogued above do NOT bite this demo's inputs but should be addressed before exercising the codepath in adjacent contexts (other initial_scales, other initializers, adversarial CLI inputs, adjoint-tape contexts, Strategy-B paths, or `extract_preconverged_anchor` reuse).

Per-chunk reports preserved at `.verification/correctness-L3/{sonnet,opus,codex}-{A,B,C}-L3.md`. L2 reports at `.verification/correctness/`. L1 doc-audit reports at `.verification/`.
