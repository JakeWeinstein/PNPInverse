# Final Revision Ledger — Session 20 (fast-realignment)

- **Final verdict:** APPROVED at round 5 (cap reached coincidentally)
- **Total rounds:** 5
- **Total issues raised:** 41 (some re-issued / clarified across rounds; counted once each)
- **Revised artifact:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/fast_realignment_plan_2026-05-08.md`

## Addressed (Accepted) — 40 issues

### Round 1 (initial 18 issues — all accepted)

| # | Issue | Resolution in revised artifact |
|---|---|---|
| 1 | Cathodic prefactor sign wrong | Phase 1.2 dropped — uses existing `_build_picard_prefactors` which has correct `−α·n·η` sign |
| 2 | R_2e reversibility dropped | Phase 1.2 audit relies on existing three-branch dispatch in `_build_picard_prefactors` |
| 3 | γ powers / proton Boltzmann factor missing | Phase 1.2 dropped — existing code has γ³ structure with `log_h_factor` carrying H⁺ Boltzmann shift |
| 4 | Spurious `· L_eff` factor | Phase 1.2 dropped — existing `_surface_concs_from_rates` is fully nondim |
| 5 | `picard_outer_loop_general` already exists | Phase 1.2 replaced from "write" to "audit + instrument with `verbose` flag"; M3a.3 silently landed before fast plan was written |
| 6 | Disabled-rxn dispatch not centralized | Phase 1.3 adds `_is_reaction_disabled` helper used at all four sites (form builder, prefactor builder, system assembler, topology classifier) |
| 7 | Topology detection too weak | Phase 1.3 adds `_is_parallel_2e_4e` strict predicate |
| 8 | Multi-steric Bikerman API requires structural change | Phase 2.1 — multi-counterion shared-theta closed-form (closed-form per project derivation; comment in `boltzmann.py:159-165` was misleading) |
| 9 | `a_nondim` violates bulk packing | Phase 2.2 — hard-sphere derivation from hydrated radii (Cs⁺=3.23e-5, SO₄²⁻=4.20e-5; θ_b=0.991 ✓) |
| 10 | A_DEFAULT continuation start broken | Phase 5b — continuation in `a_nondim` from 0, not A_DEFAULT |
| 11 | `phi_o = ln(H_o/c_clo4_bulk)` is 1:1-specific | Phase 2.3 — `_solve_outer_phi_multiion` via bisection on multi-ion electroneutrality |
| 12 | `compute_surface_gamma` 1:1-specific | Phase 2.3 — `compute_surface_gamma_multiion` with outer anchors |
| 13 | "Picard absorbs 2:1 shape mismatch" self-deception | Phase 2.4 — proper ψ-vs-φ split + linearized-Debye seed; warm-start fallback (5f) |
| 14 | Stern split 1:1-specific | Phase 2.4 — linearized-Debye Stern split with `λ_eff_local` |
| 15 | λ_D not updated for I=0.3 M | Phase 2.3 — `effective_debye_length_local` (FD `−dρ/dφ`) |
| 16 | Anchor V backwards (+0.45 vs +0.55) | Phase 3 — anchor V_RHE = +0.55 (weakest cathodic drive within page-15 grid) |
| 17 | `K0_PHYS_R4E = K0_PHYS_R1` produces R_4e dominance | Phase 3 — four-pass structure (A pure-2e, B pure-4e, C exploratory mixed, D mixed reduced K0_R4e ladder); Pass C explicitly NOT required for done |
| 18 | Acceptance inconsistency | Phase 4 + "Done" — per-pass criteria, no combined-grid ambiguity |

### Round 2 (11 new issues — all accepted)

| # | Issue | Resolution |
|---|---|---|
| R2-1 | SO₄²⁻ ideal explodes at +0.55 V; "Cs steric / SO₄ ideal" wrong physics | Phase 2.1 multi-steric closed-form (rejected the partial-steric hack; full multi-steric is closed-form) |
| R2-2 | `_solve_outer_phi_multiion` ideal vs steric | Phase 2.3 uses steric closure consistently |
| R2-3 | γ uses bulk anchors, should use outer | Phase 2.3 — `compute_surface_gamma_multiion` takes `c_outer` per ion |
| R2-4 | Don't change `poisson_coefficient` | Confirmed — only IC/Stern helpers consume `λ_eff_local`, residual coefficient unchanged |
| R2-5 | Stern at IC dominates; can't disable | Phase 2.4 — linearized-Debye Stern split (existing fallback path), NOT disabled |
| R2-6 | Disabled-rxn topology classification | Phase 1.3 — classify from nominal config; disabled rxns get zero rows in linear system |
| R2-7 | Spatial IC hardcodes `ln(H_outer/c_clo4_bulk)` | Phase 2.4 — multi-ion `_solve_outer_phi_multiion` replaces it; `c_clo4_bulk` removed from spatial IC code path |
| R2-8 | `c_clo4_bulk` overloaded | Phase 2.3 — structured `counterion_ctx` via `build_counterion_ctx()`; single source of truth |
| R2-9 | "Done" lowered without approval | Restored to ≥ 15/25 across A/B/D (Phase 4 + "Done") |
| R2-10 | Pure-2e doesn't exercise mixed | Phase 3 — Pass D ladder (mixed with reduced K0_R4e) |
| R2-11 | Legacy warm-start spec underspecified | Phase 5f — concrete `_legacy_warmstart_to_target` with subfunction-by-subfunction copy |

### Round 3 (7 new issues — 6 accepted, 1 defended-then-accepted)

| # | Issue | Resolution |
|---|---|---|
| R3-1 | A_dyn_outer omits a_O2/a_H2O2 | Phase 2.3 `_solve_outer_phi_multiion` uses full `Σ a_i c_i_outer` |
| R3-2 | Legacy byte-equiv condition wrong (a_H ≠ 0 in production) | `multi_ion_enabled` opt-in flag with hard validation; legacy path engaged when flag is False (no a_H test) |
| R3-3 | `effective_debye_length` should use local dρ/dφ at outer | DEFENDED in R4 (bulk linearization for fast plan), CAPITULATED in R5 — implemented as FD `−dρ/dφ` (Phase 2.3 `effective_debye_length_local`) |
| R3-4 | Spatial IC neutral species need γ shift | Phase 2.4 — `gamma_psi_profile(y)` applied to all dynamic species (z=0 included) |
| R3-5 | Pass D fixed factor balances near anchor only | Phase 3 — Pass D becomes ladder {1e-12, 1e-15, 1e-18, 1e-21, 1e-24} |
| R3-6 | "Done" criterion ambiguous | Phase 4 — restated per-pass with explicit smoke vs done thresholds |
| R3-7 | `theta_b` duplication risk | Phase 2.3 `multi_ion.py` is single producer; downstream reads from ctx |

### Round 4 (5 new issues — all accepted)

| # | Issue | Resolution |
|---|---|---|
| R4-1 | A_dyn_outer silently zeros a_O2/a_H2O2 (re-issue of R3-1 with sharper detail) | Confirmed `THREE_SPECIES_LOGC_BOLTZMANN` has `a_vals_hat=[A_DEFAULT]*3`; Phase 2.3 uses full sum |
| R4-2 | "Byte-equivalent legacy condition" `a_H==0` doesn't fire | Phase 2.3 — `multi_ion_enabled=False` is the explicit opt-in flag, not predicated on `a_H` |
| R4-3 | `effective_debye_length` bulk Σz²c too weak | Phase 2.3 — `effective_debye_length_local` via FD `−dρ/dφ` |
| R4-4 | `multi_ion_enabled=False` with len>1 footgun | `make_bv_solver_params` — hard `ValueError` when `len(counterions) > 1 AND not multi_ion_enabled` |
| R4-5 | Pass C contradiction (≥5/25 OK vs never gates done) | Phase 4 + "Done" — Pass C explicitly NOT required for done; exploratory only |

### Round 5 (3 new issues — 2 accepted, 1 final approval)

| # | Issue | Resolution |
|---|---|---|
| R5-1 | Confused absolute φ vs diffuse-layer ψ | Phase 2.4 — explicit `phi(y) = phi_outer(y) + psi(y)` decomposition; Boltzmann shift uses ψ; Poisson primary uses φ; mu_H gets both |
| R5-2 | O₂/H₂O₂ outer values aren't bulk | Phase 2.4 — `c_dyn_outer_profile(y)` interpolates from Picard `O_s/P_s/H_o` at OHP edge to bulk over L_REF |
| R5-3 | Local λ_eff with FD dρ/dφ | Already accepted in R4-3 |
| R5-4 | `multi_ion_enabled=False` with len>1 (re-issue of R4-4) | Already accepted; hard validation in place |
| R5-5 | Pass C contradiction (re-issue of R4-5) | Already accepted; restated cleanly |

### Round 5 minor caveats (post-APPROVAL)

| # | Note | Resolution |
|---|---|---|
| R5-Note-1 | "Done" sweep is structural, not calibrated | Pre-implementation note + Phase 3 labeling caveat + final acceptance section all state "structural validation only; Pass D plots are NOT physical page-15 agreement; calibration is M4 work" |
| R5-Note-2 | `phi_outer(y) = phi_o · (1 − y/L_REF)` is IC approximation | Phase 5g — explicit escalation: solve `_solve_outer_phi_multiion` at several y nodes if Newton failures localize to initial EDL shape |

## Defended (none ultimately)

R3-3 (`effective_debye_length` local form) was defended in R4 with the
"fast plan accepts the bulk linearization as a documented
approximation" argument, but capitulated in R5 after GPT pointed
out that the linear-Debye Stern split (where λ_eff is consumed) is
itself sensitive to the form, and that the "Stern dominates" regime
is precisely where the bulk-vs-local discrepancy could put ψ_D on
the wrong branch. Final implementation uses local FD `−dρ/dφ`.

## Unresolved (none)

All 41 issues raised across the 5-round loop reached resolution.
The two R5 caveats are documentation/labeling notes, not blocking.

## Statistics

- 5 rounds
- 41 unique issues
- 40 accepted, 0 defended (1 defended-then-accepted), 0 unresolved
- 1 major plan revision discovered mid-loop:
  `picard_outer_loop_general` already exists, so the original
  Phase 1.2 ("write minimal 2-rate Picard") was replaced with an
  audit pass.
- 1 closed-form math discovery: multi-counterion Bikerman has a
  shared-theta closed form (per the project's own derivation in
  `docs/steric_analytic_clo4_reduction_handoff.md` generalized to
  multi-ion), invalidating the `boltzmann.py:159-165`
  `NotImplementedError` "couples" comment.

## Revised artifact summary

The plan is now substantially restructured from the round-1 draft:

- Phase 0 unchanged (git checkpointing).
- Phase 1 split into 1.1 (disabled-rxn guard), 1.2 (Picard audit
  rather than rewrite), 1.3 (centralized helpers).
- Phase 2 expanded into 2.1-2.4 with new `multi_ion.py` module as
  the single source of truth for multi-ion state (counterion_ctx,
  outer phi solve, multi-ion γ, local λ_eff).
- Phase 2.4 includes proper ψ-vs-φ decomposition and full
  c_dyn_outer interpolation from Picard surface values.
- Phase 3 has 4 passes (A pure-2e, B pure-4e, C exploratory
  literature mixed, D ladder of reduced K0_R4e).
- Phase 4 + "Done" use per-pass mechanically-checkable criteria.
- Phase 5 includes 7 sub-fallbacks (5a-5g), with concrete code for
  legacy warm-start and IC-approximation escalation.

Estimated total: 7-12 days (was 5-10 in the round-1 draft;
structural fixes added net ~2 days but several phases compressed
since `picard_outer_loop_general` already exists).

The plan is APPROVED but its delivered output is **structural, not
calibrated**. Calibration is M4 work, deferred.
