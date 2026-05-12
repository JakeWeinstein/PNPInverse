# FINAL_REVISION — Critique session 37

**Path revised:**
`/Users/jakeweinstein/.claude/plans/phase6b-step10-phase-D-deltaBeta-fit.md`
(v1 → v7-FINAL).

**Final verdict:** APPROVED at round 7.
**Issue trajectory:** R1=20, R2=22, R3=18, R4=11, R5=8, R6=4, R7=0.
**Total accepted:** 91.  **Defended:** 0.  **Unresolved:** 0.
**Non-blocking note:** 1 (cation_params attribute name).

---

## Critical structural corrections

The v1 plan had **two major errors** that were caught early in the
critique loop:

1. **Δ_β bracket was dimensionally wrong by ~7 decades.** v1 said
   `Δ_β ∈ [-2, +2]` with β in 1/pm² units.  Truth: β is in pm²
   (β_K_Cu = −45.608 pm²), local Stern σ at V_kin is
   ~1.07e-7 counts/pm², so ±2 in Δ_β shifts ΔpKa by ~2e-7 (flat-
   loss territory).  Bracket corrected to one-sided
   `[−4.673e7, −β_K_Cu − eps_beta]` pm², parameterized in
   target-ΔpKa-effect space.

2. **Continuation topology was invented, not validated.** v5 said
   "λ ladder at anchor → V warm-walk at λ=1".  Validated A.2
   topology is "anchor at λ=0 → V warm-walk at λ=0 → per-V λ
   ramp 0→1".  Switched to the validated path; wall budget jumped
   from ~5 hours to ~15 hours (overnight execution scope).

These two corrections alone shifted the plan from "fast 1D fit"
to "overnight derivative-free fit with careful topology choices".

---

## Addressed (91 accepted issues, by round)

### Round 1 (20 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | Δ_β ∈ [-2, +2] bracket wrong by ~7 decades | P1 unit fix, P23 target-ΔpKa parameterization |
| 2 | β unit error (1/pm² → pm²) | P2 unit derivation rewrite |
| 3 | r_H_El_pm fallback unacceptable for additive Δ_β | P3 mandatory beta_offset_pm2 |
| 4 | beta_offset implementation needs residual + diagnostic + tests | P4 |
| 5 | Data target wrong — must average pH bin rows | P2 4-row mean |
| 6 | V grid arithmetically inconsistent | P4 explicit 24-point grid |
| 7 | Ring onset sort direction missing | P5 anodic→cathodic sort |
| 8 | 0.05 V spacing too coarse for ±50 mV onset | P8 adaptive 4-point refinement |
| 9 | Unimodality alone insufficient | P7 sensitivity + noise floor + slope |
| 10 | argmax-V hidden in informational | P13 elevated to diagnostic |
| 11 | Brent mis-specified | P8 `minimize_scalar(method="bounded")` |
| 12 | 25-eval cap inconsistent with σ-mapping | P20 budget recount |
| 13 | Ablation σ underspecified | P10 V-independent 0.141 |
| 14 | 30% div formula breaks at zero | P9 symmetric + abs floor (later abs threshold dropped) |
| 15 | Sign guard ambiguous after max(0,-σ) clamp | P12 `pka_shift_avg < 0` |
| 16 | ΔpKa overflow via 10^(-ΔpKa) | P15 pre-screen (later) |
| 17 | D2 tests too weak | P6 observable extraction tests |
| 18 | No Δ_β=0 byte-equiv baseline | P14, P34 |
| 19 | Falsification + DoD contradictory | P11 OUTCOME_A/B/C split |
| 20 | Solver failure conflated with falsification | P11 separate verdicts |

### Round 2 (22 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | Widened gate violates locked bundle | P21 ±10 pp restored, NOISY flag |
| 2 | β_K_Cu = -45.18 vs -45.61 | P22 `compute_beta_per_cation` helper |
| 3 | Positive ΔpKa targets violate sign guard | P23 negative-only grid |
| 4 | Stern bracket upper bound contradicts sign guard | P24 one-sided closed |
| 5 | T=0 maps to Δ_β = +45.6, not baseline | P23 grid construction |
| 6 | T-grid too coarse over relevant decades | P23 log-spaced |
| 7 | +inf breaks identifiability | P40 two-array bookkeeping |
| 8 | Slope threshold meaningless in raw Δ_β | P31 target-ΔpKa space |
| 9 | ΔpKa_clip in residual changes locked formula | P25 pre-screen not clamp |
| 10 | loss=+inf self-contradictory after solve | P25 pre-screen verdict |
| 11 | V grid off-by-one (22 vs 23) | P41 explicit tuple + test |
| 12 | V_kin = -0.10 outside grid | P41 V_kin first |
| 13 | Adaptive refinement points off-by-one | P8 4 interior points |
| 14 | NaN handling reintroduces v1 bug | P27 production V-fail invalidates |
| 15 | OUTCOME_A adds non-locked gates | P28 split LOCKED_PASS vs E_READY |
| 16 | argmax/onset as falsification overclaims | P29 diagnostic mismatch |
| 17 | abs_div threshold invented | P30 only bundle-locked rel |
| 18 | σ-divergence almost guaranteed | P30 expected divergence note |
| 19 | xatol too loose in β units | P31 ΔpKa-space tolerance |
| 20 | Wall budget hides PDE count | P32 PDE-solve count table |
| 21 | Continuation topology unresolved | P33 per-Δ_β fresh anchor |
| 22 | Δ_β=0 byte-equiv underspecified | P34 file/V/fields/tolerance |

### Round 3 (18 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | T=0 baseline-vs-boundary contradiction | P36 Δ_β=0 explicit |
| 2 | Bounded optimizer needs closed bounds | P38 eps_beta cap |
| 3 | Grid-vs-bracket mismatch (T=-5 vs -3) | P36 grid+bracket aligned |
| 4 | Ablation grid wrong (Δ_β=0 maps to T=-6.43) | P37 mapping-specific grid |
| 5 | Same T grid can't be reused | P37 separate grids |
| 6 | Pre-screen needs full V scan | P39 full Δ_β=0 scan |
| 7 | σ may change with Δ_β | P39 per-V pka_shift verify |
| 8 | +inf and identifiability metric overlap | P40 loss_all + loss_finite_valid |
| 9 | V grid "22 points" prose vs 24 actual | P41 constant + test |
| 10 | V_kin outside locked overlap window | P42 mask extraction to [-0.06, +1.0] |
| 11 | Continuation order inconsistent | P43 nearest-anchor |
| 12 | +0.55 anchor not in grid | P43 anchor-only |
| 13 | Circular import via _bv_common ↔ calibration | P44 new `calibration/singh2016.py` |
| 14 | JSON keys are `cd_mA_cm2` not Unicode | P45 actual schema |
| 15 | k_hyd=1e-3 record selection underspecified | P45 uniqueness assertion |
| 16 | Wall budget arithmetic inconsistent | P46 single table |
| 17 | Ablation Brent decision unresolved | P47 committed |
| 18 | xatol assumes V_kin σ | P48 per-mapping conversion |

### Round 4 (11 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | A.2 lookup uses `rungs` not `lam_diagnostics` | P50 actual key |
| 2 | `solve_grid_with_anchor` topology is nearest-anchor | P51 documented |
| 3 | λ ramp omitted from per-Δ_β topology | P52 (later corrected in P60) |
| 4 | Wall budget undercounted | P53 (later replaced by P61) |
| 5 | Δ_β=0 double-counted in Stern grid + baseline | P54 baseline injected |
| 6 | First ablation grid violates safe domain | P55 -14.9 inside |
| 7 | Hard-coded 6.43 in ablation grid | P55 exact arithmetic |
| 8 | Variable name `_at_V_kin` misleading | P56 `_max_over_grid` |
| 9 | `gamma` fallback weakens schema | P57 exact key, no fallback |
| 10 | Back-compat import tests for SINGH params | P58 3 permutation tests |
| 11 | `beta_offset_pm2` representation ambiguous | P59 exclusively fd.Function |

### Round 5 (8 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | λ-topology verification wrong | P60 validated topology pinned |
| 2 | Wall budget depends on #1 | P61 ~15 hours, overnight |
| 3 | Byte-equivalence topology mismatch | P62 same topology |
| 4 | Ablation Δ_β=0 can't be injected | P63 separate ablation baseline |
| 5 | Ablation pre-fit count inconsistent | P64 7 total (1+6) |
| 6 | Stern grid hard-coded from V_kin σ | P65 computed from full scan |
| 7 | Wall estimate internally inconsistent | P61 single table |
| 8 | beta_offset_pm2_func type ambiguity | P66 exclusively Function |

### Round 6 (4 issues, all accepted)

| # | Issue | Patch |
|---|---|---|
| 1 | Δ_β=0 byte-equiv path differs from A.2 warm grid | P68 split (a) hard + (b) diagnostic |
| 2 | "Deviation → STOP" too strong for full-grid baseline | P69 STOP only on (a) |
| 3 | Runtime setter must update ctx metadata | P70 mirror + test |
| 4 | Cation keys "K" vs "K+" | P71 canonical charged-form |

### Round 7 (0 blocking issues — APPROVED)

GPT noted one non-blocking implementation correction:
`CationHydrolysisBundle` has `cation_params`, not `params`.  The
P70 metadata mirror should use `bundle.cation_params` (likely via
`object.__setattr__` with a copied dict, mirroring
`set_reaction_r_H_El_pm_model`).  GPT explicitly said "I would
not block Phase D on this."  Executor handles in Phase 10.A.1
when implementing the runtime setter.

---

## Defended (0)

None.  All 91 issues raised by GPT across 7 rounds were either:
* Verifiably correct against the codebase (cross-checked via
  grep, file reads against named symbols and line numbers); or
* Sound priors / risk identification that were straightforwardly
  worth accepting.

---

## Unresolved (0)

None — verdict APPROVED with one non-blocking implementation note.

---

## Provenance trail

Round-by-round files in this directory:
* `R1_to_gpt.md` (~857 lines including inline plan) / `R1_from_gpt.md` (20 issues)
* `R2_to_gpt.md` / `R2_from_gpt.md` (22 issues)
* `R3_to_gpt.md` / `R3_from_gpt.md` (18 issues)
* `R4_to_gpt.md` / `R4_from_gpt.md` (11 issues)
* `R5_to_gpt.md` / `R5_from_gpt.md` (8 issues)
* `R6_to_gpt.md` / `R6_from_gpt.md` (4 issues)
* `R7_to_gpt.md` / `R7_from_gpt.md` (APPROVED + 1 non-blocking note)

Codex session ID: `019e17c0-91c7-7791-945c-25064a110dfd` (a single
persistent codex session, resumed every round).

The revised plan
(`/Users/jakeweinstein/.claude/plans/phase6b-step10-phase-D-deltaBeta-fit.md`)
is the v7-FINAL artifact and the executor's source of truth.
Step 10 execution is **not yet started** — to be spawned once step
9 completes and the user signals to proceed with Phase D execution.

## Concurrent execution context

Step 9 (B.2 densified k_hyd × λ ramp) was running concurrently
during this critique loop via subagent `ac1c433d3161f93c8`.  Step
10 plan files do not modify step 9 files; the two are
independent.  At the time of APPROVED verdict (round 7), step 9
was at Pass 2.5/14 with θ=0.861 at k_hyd=1e-3 — reproducing the
v10b A.2 baseline anchor cleanly (~14 rungs remain in step 9).
