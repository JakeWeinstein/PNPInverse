# Final Revision Ledger — Session 25 (phase-6a-water-ionization)

**Final verdict**: APPROVED at round 5 / 5.
**Total issues raised across rounds**: 58 (R1: 25, R2: 15, R3: 10, R4: 8, R5: 5).
**Path**: revised plan at `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md`.

## 1. Architectural change

The single biggest result of the loop: **the original plan's "Option B"
(algebraic c_OH = Kw/c_H slaving in Poisson only) was physically
incoherent.** Substituting OH⁻ into Poisson without modifying the H⁺
NP equation leaves surface c_H unbounded — Option B can NEVER satisfy
the diagnostic it was intended to fix.  The only mass-conserving
fast-water limit is the proton-condition variable `E = c_H - c_OH`,
which the original plan listed as a fallback.

**Pivot**: Option C is now the primary landing.  Option D (full
dynamic OH⁻ + finite-rate R_w) is the escalation if Option C fails
the post-solve fast-water-validity gate.

Architectural issues driving the pivot: R1 #3, #4, #5, #23.

## 2. Issue ledger

### Addressed (52)

#### Architectural (R1)
- **R1 #3, #4, #5, #23**: Option B → Option C pivot.  Plan §3 entirely
  rewritten; §3.2 documents the rejection of algebraic-only slaving.
- **R1 #6, #7**: Option C PDE derivation explicit; closure
  `c_H(E) = (E + √(E²+4Kw_hat))/2` written in §3.3 with explicit nondim.
- **R1 #11**: IC closure rederived from primaries; Path-B (approximate)
  selected for Phase 6α with hard-trigger fallback to Path A.  §5 Q3.
- **R1 #12**: forms_logc[_muh].py touch surface enumerated.  §5 Q2 table.

#### Math/units (R1, R2, R3, R4)
- **R1 #1, #2, R2 #5, R3 #1**: Damköhler arithmetic corrected; final
  table in §4.  Water-source ceiling i_max = 1.35 mA/cm² at L=100 µm,
  not 13.5.
- **R1 #18**: pH↔concentration consistency in one tabulated column.
- **R1 #8**: Naming `KW_MOLAR_SQUARED` (avoid `_M2` ambiguity).  §5 Q1.
- **R1 #9, #25, R3 #2, R4 #2**: Nondim/units explicit; Kw_hat computed
  from one canonical physical baseline `KW_PHYS = 1e-8 (mol/m³)²`.
  §5 Q1.
- **R1 #14, R2 #7**: Symmetric `u_clamp` retained; bounds both
  `exp(u_H)` and `Kw_hat·exp(-u_H)` in log/muh primary.  c_OH_clamp
  rejected; continuation strategy primary.  §5 Q2 + clamp-inactive
  policy box.
- **R1 #15**: c_OH magnitude at pH 14 (~1 M) acknowledged; 5-rung
  Kw_eff continuation schedule mitigates Newton stiffness.  §5 Q2.
- **R2 #1**: NP migration term gets diffusivity:
  `J_E = -(D_H c_H + D_OH c_OH)·(∇u_H + ∇φ)`.  §3.3.
- **R2 #2, R3 #3**: Conservative weak form
  `∫ v·E_t − ∫ ∇v·J_E + ∫ v·J_E·n = 0`.  §3.3.
- **R2 #3, R3 #4, R4 #1**: BCs explicit; `J_OH,Faradaic = 0` at pH 4
  acidic ORR; reduced E equation only constrains `J_E·n`.  §3.3 + §6 R9.
- **R2 #4, R3 #7**: muh derivation separate; top BC `μ_H_top = ln c_H_bulk_hat`
  requires `φ_top = 0`.  §3.3.
- **R2 #6, R4 #1**: Finite water-source capacity table + headroom
  per L_eff in §4.
- **R2 #8**: log/muh primary avoids `exp(-2u_H)` stiffness; coefficient
  is `D_H exp(u_H) + D_OH Kw exp(-u_H)`, well-conditioned.  §3.3 + R7.
- **R2 #10, R3 #8**: Mass-conservation acceptance reuses
  `_build_bv_observable_form` for current-balance check.  Gate 3.
- **R2 #12**: Concentration-Kw model (sterics in Poisson only, no
  activity correction).  §3.3.
- **R2 #14**: Expression-shape regression replaced with config-level
  + numerical-tolerance regression.  §5 Q4.
- **R2 #15**: Yash cross-check gated on locked V_RHE/pH/species/coords.
  §5 Q5 Gate 5.
- **R3 #5**: Fast-water validity = `ε = R_w,req / (k_r·Kw)`; Gate 4.
- **R3 #6**: Sulfate buffer is steady-state (not "one-shot");
  quantitative comparison vs water source in §4 + §7.
- **R3 #9**: c_H · c_OH / Kw check is tautological; dropped.
  Conservation/current-balance/Yash-profile checks are the real gates.
  Gates 3-5.
- **R3 #10, R4 #8**: Full MMS protocol with manufactured (u_H, φ),
  derived c_OH and J_E, analytic forcing s(y) matching the
  conservative weak-form sign convention.  §5 Q4.
- **R4 #3**: Sulfate transport arithmetic corrected: 0.0965 mA/cm²
  at L=100 µm (not 0.97).
- **R4 #4, R3 #9**: Water-aware Picard rewrite is Path A; Phase 6α
  adopts Path B (approximate, relies on Newton + continuation) with
  hard-trigger fallback.
- **R4 #5**: `R_w,req` extracted from weak H⁺ residual via mass-matrix
  projection (avoids second derivatives).  §5 Q5.
- **R4 #6**: Stale `k_f·[H₂O] ≈ 0.078 mol/m³·s` purged; correct value
  `k_r · Kw = 1.4 mol/m³·s` used throughout.
- **R4 #7**: Reduced-BC validation: `|J_OH·n_inferred| / |J_H,BV·n| < 0.05`
  added to Gate 5.

#### R5 minor (5)
- **R5 #1**: J_OH·n inferred via Firedrake `FacetNormal`, no
  hand-expansion.  §5 Q5 sentence.
- **R5 #2**: Sulfate inequality typo fixed ("at small L_eff,
  especially ≤ 21 µm").  §4 + §7.
- **R5 #3**: Clamp-inactive policy box added in §5 Q2.
- **R5 #4**: Gate 4 includes "R_w,req_hat is the mass-matrix
  projection of the weak H+ residual".  §5 Q5.
- **R5 #5**: Hard trigger for Path-B fallback — anchor Newton
  failure at any Kw_eff rung OR post-IC residual stays above 1e-2
  after two ladder refinements.  §5 Q3.

### Defended (3)

- **R1 #10**: Option A rejection scope.  Defended that pure-Boltzmann
  is the right distribution for an inert species in an EDL but wrong
  for OH⁻ because the dominant driver is homogeneous reaction
  equilibrium, not EDL Boltzmann.  Plan §3.1.
- **R1 #19**: Sulfate-buffering significance.  Defended that the
  Phase 6α surface-pH-lift goal is achievable with water alone at
  the L=100 µm Yash-comparison gate; sulfate becomes load-bearing
  for the deck PEAK at small L_eff (Phase 6β).  Quantitative
  comparison added in §4 + §7 to support deferral.
- **R1 #21**: MMS scope.  Defended dropping MMS for Option B (which
  is dropped entirely).  MMS for Option C is the proton-condition
  PDE with manufactured (u_H, φ).  §5 Q4.

### Unresolved (3)

These are listed in §8 of the revised plan as "open questions" and
will be resolved empirically during implementation:

- **Kw_eff rung count**: 5 rungs initial spec; mid-point insertion
  if Newton struggles.  Empirical tuning during Q4 slow tests.
- **MMS convergence order**: expect ≥ p+1 = 2 for CG-1; if observed
  < 1.5, debug residual sign / boundary terms.
- **IC residual tolerance threshold**: 1e-2 vs 1e-3.  Tighter
  threshold may be needed if Newton is iteration-budget limited.

## 3. Implementation cost (revised plan estimate)

| Task | LoC | Wall-time |
|---|---|---|
| Q1 constants | ~50 | 1-2 h |
| Q2 residual rewrite (2 backends) | ~300 | 1-2 days |
| Q3 IC + continuation | ~150 | 1 day |
| Q4 tests + MMS | ~400 | 1-2 days |
| Q5 sweep + scorer | ~100 | 0.5 day + ~1 h sweep |
| **Total** | **~1000** | **5-7 days** |

Critical path: Q2.  Everything downstream of the residual rewrite
can proceed in parallel.

## 4. Files written by this critique session

```
docs/CHATGPT_HANDOFF_25_phase-6a-water-ionization/
├── STATUS.md
├── R1_to_gpt.md   R1_from_gpt.md   .codex_log_R1.txt
├── R2_to_gpt.md   R2_from_gpt.md   .codex_log_R2.txt
├── R3_to_gpt.md   R3_from_gpt.md   .codex_log_R3.txt
├── R4_to_gpt.md   R4_from_gpt.md   .codex_log_R4.txt
├── R5_to_gpt.md   R5_from_gpt.md   .codex_log_R5.txt
├── .codex_session_id (UUID = 019e0f4c-f1e7-7350-aeac-7bab073eaad7)
└── FINAL_REVISION.md  (this file)

docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md  (revised artifact)
```

## 5. Recommended next steps

1. **Pause for the user**: review the revised plan and confirm Option C
   as the landing strategy before any implementation starts.  The
   pivot from B to C is significant and worth a checkpoint.
2. **If approved by the user**: scope Phase 6α.0 as Q1 + Q2 (constants
   + residual rewrite, ~1.5 days).  This gates everything else.
3. **Validation gate before declaring Phase 6α complete**: all 5
   acceptance gates (Surface pH, plateau direction, E conservation,
   fast-water validity, Yash cross-check) must pass.  Any single gate
   failure escalates to Phase 6α.5 (Path A IC rewrite OR Option D
   full dynamic OH⁻).
