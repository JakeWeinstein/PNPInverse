# Round 4 counterreply — k2so4-dual-series-fit

All 7 accepted.

**Re 1 (mask removal still too permissive):** Accept — removed-bin
discipline added: any in-window bin removed solely for θ*-failure is
RETRIED at the polished θ, then either re-included (final scoring
re-run) or excluded with a documented scientific reason; original
count / removed bins / retry outcomes all reported; paper-grade
claims blocked while any bin remains excluded for purely numerical
reasons.

**Re 2 (c_O₂ bracket vs gas identity):** Accept — air-saturated
impossibility check added to the provenance note with numbers:
air-sat c_O₂ ≈ 0.25 mol/m³ ⇒ 4e ceiling ≈ 1.2 mA/cm² < measured
plateau ≈ 4.0 ⇒ air ruled out BY THE DATA (computed at both ends of
the L_eff bracket). If that check ever fails to exclude air, the O₂
protocol confirmation becomes blocking for all transport/kinetic
claims.

**Re 3 (lock blocked only on OCP):** Accept — lock tiers defined:
computational lock (fit + gates + sensitivities + profiles +
ablations, θ with all conditional labels) vs paper-grade claims with
an explicit claim → blocking-ask table (0.47 V OCP → absolute θ /
potentials; rpm + O₂ protocol → transport-free kinetics; catalyst →
transferability; ring hold/cal → absolute partition scale; raw Tafel
xlsx → 7.1 feature chemistry). No paper-grade claim ships with its
ask open.

**Re 4 (derived-g profiles rigor):** Accept — fixed-g refits get a
constraint tolerance |g(θ̂) − g₀| ≤ 0.01 (absolute, fraction scale),
penalty weight swept until the tolerance binds without distorting J
by >1%, and the g-ladder is EXTENDED until the profile crosses
Δχ² = 4 on both sides (or a bound is hit → reported one-sided).
Checkpoint envelopes demoted to exploratory plots, never cited
widths.

**Re 5 (FD tolerance collapses at stationarity):** Accept —
objective-scale absolute floor τ_abs = 1e-6·(1+|J|) per scaled unit
added to the tolerance; the PRIMARY adjoint/FD validation runs at a
nonstationary interior point (X0 = θ*); the final-θ check is a
regression smoke test only.

**Re 6 (n_e diagnostic misses 4e-dominated regimes):** Accept — n_e
evaluated wherever disk clears its threshold; ring σ propagated into
an n_e uncertainty band; low-ring regions flagged low-confidence
rather than excluded; band excursions outside [1.8, 4.2] still
trigger stop-and-diagnose.

**Re 7 (reduced-χ² naming):** Accept — renamed "per-series
standardized residual score" everywhere; reduced-χ² semantics
explicitly disclaimed.

## Updated artifact

Changes this round: removed-bin retry discipline; air-saturation
impossibility check + escalation rule; lock tiers with
claim→blocking-ask table; constrained-profile tolerance + Δχ²=4
crossing requirement + envelope demotion; FD absolute floor +
nonstationary primary validation point; n_e band evaluation
everywhere disk clears threshold; standardized-residual renaming.

---

# Phase 7.2 — Dual-series (disk + ring) fit to the K₂SO₄ pH 6.39 RRDE data

**Status: in GPT critique session 43 (R1: 18, R2: 11, R3: 7 issues —
all accepted; this version incorporates all three rounds).**
Predecessor: `~/.claude/plans/phase7p1-chemistry-gap.md` (session 42
APPROVED, never executed). This plan does NOT supersede 7.1 — it runs
a cleaner condition first, with data that resolves 7.1's biggest
structural weakness (R4: 2e/4e partition non-identified on
peroxide-only data). 7.1's slide-15 residual program resumes after,
informed by what the dual-series fit pins down.

## Why this data, why now

`data/EChem Reactor Modeling-Seitz-Mangan/Brianna/0,1M K2SO4 data
8-15-19.xlsx` contains REAL numeric RRDE LSV — no digitization:

- Sheets: processed main (2 column groups), `Exp Info`, raw
  `cycle 1/2/3`. Raw columns: `Edisk/V, Idisk/mA, Iring/mA` +
  in-sheet constants: **N = 0.224**, Ref Cal vs RHE = −0.549 V,
  Rs = 125 Ω, disk Ø 5 mm (A_d = 0.196 cm²), ring 7.5/6.5 mm OD/ID
  (A_r = 0.110 cm²). Processed: V_RHE iR-corrected, j_disk (per
  disk area), j_ring (per ring area), Sel%, n_e.
- Condition in-file: **0.1 M K₂SO₄, bulk pH 6.39** (H₂SO₄-titrated),
  1047 pts/cycle, V_RHE −0.06 … +1.14, j_disk −4.0 … +2.25,
  j_ring ≤ 0.36 (ring area), peak Sel ≈ 73% @ +0.30 V (their
  formula; we recompute canonically in Stage 0), n_e ≈ 2.8.
- `Exp Info` catalogs the FULL day: pH {6.39, 5.21, 4.21, 3.42,
  2.35, 1.65}, per-disk Rs, ring onset (0.472 V @ 0.01 mA/cm²_ring),
  max ring current, peak selectivity, capacitance band ±0.0008.
  Only the pH 6.39 LSVs are in this file — sibling files are a
  data ask.
- Cycles 2 and 3 are near-identical replicates (→ real σ). Cycle 1
  carries the remark "pH changed to 6.11 while ring cleaning" and
  differs grossly → excluded as conditioning/contaminated, documented.

Scientific leverage at pH 6.39: bulk c_H ≈ 4.07e-4 mol/m³ → H⁺
Levich cap ≈ 2.4e-3 mA/cm² (3 OOM below data) and
[HSO₄⁻]/[SO₄²⁻] = c_H/Ka2 ≈ 4e-5 → **acid routes and bisulfate
buffering are both negligible by construction**. The dual-pathway
water-route kinetics carry the entire curve, and the disk series
makes the 2e/4e partition identifiable (to be DEMONSTRATED via the
Stage 4 protocol, including derived-quantity profiles — not assumed).

## Inherited conventions (from 7.1, unchanged unless noted)

- V-axis mapping table; sign ledger (incl. anion-migration
  coordinate-free row + unit test); proton-source ledger;
  escape-flux scoring; mini-ablation cell at acceptance.
- **Statistics framework (R1#8, R2#10):** the OPTIMIZATION objective
  J may use per-series normalization, but all model comparison uses
  **raw χ² = Σ (r/σ)² over a FIXED data vector** (both series,
  frozen mask, no per-series normalization, no dropped points).
  θ*-vs-fit comparison is a **Δχ² predictive-improvement score** —
  NOT AIC, NOT a likelihood-ratio test. σ is declared a
  **conservative single-observation predictive-error scale**
  (replicate |Δ|/√2, not the SEM |Δ|/2 of the fitted two-cycle
  mean) ⇒ absolute reduced-χ² values are NOT interpreted; only
  relative comparisons on the fixed vector.
- **OCP shift (R1#1, R2#9):** the 0.47 V GC-OCP component is
  documented on the Ag/AgCl axis; this dataset's V_RHE axis was
  built with the file's MEASURED calibration (+0.549 V), not
  theoretical 0.197 + 0.059·pH = 0.574 V. Central shift
  **V_OCP = 0.47 + 0.549 = 1.019 V**; refit variants at 1.044 and
  0.994 (calibration-convention bracket) AND ±50 mV {0.969, 1.069}
  (0.47-component uncertainty). **θ is labeled OCP-conditional
  until the 0.47 V component is confirmed** — the confirmation ask
  is BLOCKING for paper-grade lock, not just for absolute-potential
  claims. Both E° shifted identically (η-preservation fast test
  re-run at this pH).
- **pH convention (R1#17):** reported pH interpreted as
  concentration-pH (c_H = 10^−pH M) for bulk BCs; ≤0.11 pH-unit
  activity caveat at I ≈ 0.3 M; bulk c_H ×{0.7, 1.4} refit
  sensitivity; γ_H/Kw_eff folded into the Kw 0.5× ablation;
  surface-pH claims carry the caveat verbatim.
- ALL acid branches at k0 = 0 in the primary fit (7.1 freeze-out;
  here also physics). **Acid-on ablation pre-registered (R1#15):**
  acid k0/α at the production values used in diagnostic A1, water
  params at accepted θ_K, NO reoptimization; metrics: acid share of
  cd and of E-sink per V, Δcd(V), Δpc(V), reported at max-share V.
  Purpose: quantify the Kw-laundering artifact at this condition.
- NOT inherited: 7.1 Stage B′ (bisulfate) — negligible at pH 6.39
  (ledger row verifies); D′ surface-transition arm — out of scope.

## Convergence + evaluation discipline (R1#2/#3, R2#1 — all stages)

- Fit grid and data mask FROZEN before optimization (Stage 1
  output). Persistent single-V failures at θ* (post escalation
  ladder) are removed from the mask BEFORE fitting, documented,
  never re-added mid-fit. **Removed-bin discipline (R3#1):** any
  in-window experimental bin removed solely because θ* failed is
  RETRIED at the polished θ; it is then either re-included (final
  scoring re-run) or excluded with a documented scientific reason.
  Original bin count, removed bins, and retry outcomes are reported.
  Paper-grade claims are blocked while any bin remains excluded for
  purely numerical reasons.
- **Failed-evaluation policy (exact, R2#1):** a solve failure at
  any objective V during an evaluation triggers an in-evaluation
  retry ladder: (i) re-walk that V warm-started from the nearest
  converged neighbor with max_inserts+2; (ii) half-step V-walk
  insertion. If still failed, the evaluation RAISES — it never
  returns (f, g) to L-BFGS-B. The outer driver catches, logs
  θ_fail, and restarts L-BFGS-B from the last valid checkpointed
  iterate with a fresh Hessian and the step that produced θ_fail
  halved (bounds box temporarily shrunk around the restart point).
  No constant-penalty values and no stale/NaN gradients ever enter
  the optimizer. Restart events are counted and reported; >3
  restarts ⇒ stop, diagnose robustness before continuing.
- The ACCEPTED θ must converge at every voltage of the final
  scoring grid.
- **FD verification points must themselves converge** (no penalty/
  retry contamination) and stay strictly inside bounds; any FD
  point that fails to converge invalidates that check (re-centered,
  not papered over).

## Stages

### Stage 0 — Extraction + provenance (Firedrake-free, ~0.5 day)
1. `scripts/studies/_extract_k2so4_ph6p39_rrde.py`: parse RAW
   `cycle 2/3` sheets; re-derive every processed column from raw +
   in-sheet constants; CROSS-CHECK against the processed main sheet
   with combined tolerance |Δ| ≤ max(1e-8 abs, 1e-6 rel) (R1#18) +
   separate explicit sign and unit assertions. Resolve and document
   the raw→processed sign flip (raw Idisk cathodic-positive;
   canonical output **cathodic-negative disk**, **anodic-positive
   raw ring**).
2. iR re-derivation: confirm the sheet formula `V + (I/1000)·Rs`
   with THEIR sign convention reproduces their iR-corrected axis;
   canonical V = their iR-corrected V_RHE.
3. **Ring baseline correction (R2#6):** per-cycle ring baseline =
   median raw j_ring over the predeclared no-H₂O₂ region (V anodic
   of the disk-signal window where the disk is non-cathodic);
   subtracted from j_ring; baseline spread propagated into σ_ring
   in quadrature; baseline value + region reported in provenance.
   An optional bounded ring-offset nuisance (±baseline spread) is
   pre-registered as a Stage 4 sensitivity, default OFF.
4. **Ring series enters the objective RAW (R2#4):** target =
   baseline-corrected j_ring (mA/cm²_ring, anodic-positive), σ_ring
   independent of N. The collection model lives on the MODEL side:
   j_ring_model = −pc_model·N·A_d/A_r (pc_model cathodic-negative ⇒
   j_ring_model ≥ 0). N variants are then pure model-side refits —
   no target/σ regeneration. The disk-equivalent conversion
   pc_data = −j_ring·A_r/(N·A_d) is still produced for PLOTS and
   slide-15 comparability only, never for the objective.
5. **Recompute Sel% and n_e canonically from raw CURRENTS (R1#11):**
   Sel = 200·(I_r/N)/(|I_d| + I_r/N), n_e = 4|I_d|/(|I_d| + I_r/N),
   using baseline-corrected I_r. Brianna's in-sheet Sel% formula
   reconciled in the provenance note; processed Sel%/n_e columns are
   PROVENANCE ONLY.
6. Binning + σ (R1#7, R2#10): common V grid ≈ 30 bins over the fit
   window; per-bin mean of cycles 2+3 per series. σ_bin per series
   = max( |cycle2 − cycle3|_interp/√2, σ_floor_series,
   f_model·|j_bin| ), f_model = 0.02; σ_floor_disk from the Exp
   Info capacitance band (units verified), σ_floor_ring from ring
   noise + baseline spread. Within-bin scatter = diagnostic only.
   σ framing per the statistics convention above.
7. Fit window (R1#6): lower edge −0.06 V. Upper edge predeclared
   dual threshold: largest V where |j_disk| > max(3·σ_floor_disk,
   2·Cap_band) sustained (5 consecutive raw points) — strictly
   cathodic of the zero crossing; anodic region excluded a
   fortiori. Onset-region bins get Cap-band σ inflation in
   quadrature. Upper edge ±0.05 V is a REFIT sensitivity.
8. Hydrodynamics + protocol asks (R1#13, R2#5, R3#2): rotation rate
   NOT in file → assume 1600 rpm (L_eff = 15.4 µm central; Stage 4
   refits at {12, 21.7}). O₂ protocol NOT in file → assume
   O₂-saturated, c_O₂ = 1.2 mol/m³ (salt-corrected; production
   value); if unconfirmed, c_O₂ refits {1.0, 1.3} and kinetic
   parameters labeled **transport-conditional** (k0–c_O₂ partial
   degeneracy noted explicitly). **Air-saturated impossibility
   check (R3#2), in the provenance note with numbers:** air-sat
   c_O₂ ≈ 0.25 mol/m³ ⇒ 4e transport ceiling ≈ 5.71·(0.25/1.2) ≈
   1.2 mA/cm² < measured plateau ≈ 4.0 ⇒ air-saturation is ruled
   out BY THE DATA (computed with the final L_eff bracket, both
   ends); if the check ever fails to exclude air, O₂-protocol
   confirmation becomes blocking for all transport/kinetic claims.
   DATA ASKS: rpm; gas/temperature/O₂-saturation protocol; catalyst
   identity (CMK-3 vs GC); ring hold potential + ring calibration;
   sibling pH files.
9. Outputs: `data/k2so4_ph6p39_rrde_{cycle2,cycle3,binned}.csv`
   (git add -f), QA overlays (extraction vs processed sheet;
   recomputed vs sheet Sel%), provenance note
   `docs/phase7/k2so4_ph6p39_provenance.md` (calibration note,
   Sel% reconciliation, ring baseline, sign table).
GATE: cross-check passes; QA overlays faithful; σ > 0 every bin.
**n_e diagnostic (R2#11, R3#6):** recomputed n_e(V) evaluated
wherever |I_d| > 10·σ_floor_disk, with ring σ PROPAGATED into an
n_e uncertainty band (low ring current ⇒ wide band, flagged
low-confidence, NOT excluded — a 4e-dominated regime is exactly
where near-zero ring matters). Excursions of the band outside
[1.8, 4.2] ⇒ STOP AND DIAGNOSE (documented investigation), not
automatic rejection.

### Stage 1 — K⁺/pH 6.39 model config (config-only, ~0.5 day)
- Driver `solver_demo_slide15_dual_pathway_cs.py` gains `--cation
  {cs,k}` (swaps DEFAULT_CSPLUS↔KPLUS_BOLTZMANN_COUNTERION_STERIC;
  preset verified, physical radius, D = 1.96e-9) and `--v-ocp-rhe`
  (pH 6.39 central 1.019). `--bulk-h-mol-m3 4.07e-4` (existing).
  Counterion bulk: K⁺ 200 / SO₄²⁻ 100 mol/m³.
- Water ionization ON; kw_eff_ladder = None (full Kw).
- Anchor recipe inherited (linear_phi IC, k0 AdaptiveLadder
  max_inserts ≥ 6, anchor at V_solver = 0, warm walk). Escalation
  ladder if anchor fails: (a) re-tune k0 band, (b) bulk-H
  continuation ramp 0.1 → 4.07e-4 (Stern-bump pattern), (c)
  coarse-mesh anchor + mesh-sequenced restart.
- Solver window: V_solver = V_RHE − 1.019 ≈ [−1.08, −0.17].
GATE: **13/13 on the smoke grid** at θ* water-only config (after
any documented mask freeze); ledger closures (E, O₂, electron)
pass; anodic share < 1%.

### Stage 2 — Pre-registered transferability diagnostic at θ*
(BEFORE any refit; R1#16 framing.) Run θ* = (−3.683, −13.537,
0.550, 0.285) at K⁺ pH 6.39, L = 15.4. Predict BOTH series; score
with the Stage 3 raw-χ² machinery; DO NOT tune. Uncertainty context
attached (θ* 4e params weakly identified; digitized-curve source);
"poor prediction may reflect θ* uncertainty, not failed
transferability" stated in the writeup. Feature metrics with
SEPARATE onsets (R1#12): ring onset = V at baseline-corrected
j_ring = 0.01 mA/cm²_ring (data: 0.472 V); disk onset = V at
|j_disk| = 0.05 mA/cm²_disk; disk plateau; ring peak
position/height; recomputed Sel% (derived QA). Archived as
`StudyResults/phase7p2_theta_star_prediction/`. No gate.

### Stage 3 — Dual-series objective + adjoint extension (~1 day)
- `calibration/phase7_wls.py` → `score_dual_series`: optimization
  J = (1/n_d)Σ w_d(cd_i − d_i)² + (1/n_r)Σ w_r(jr_i − r_i)²
  (disk: cathodic-negative mA/cm²_disk; ring: raw anodic-positive
  mA/cm²_ring with model-side N mapping), w = 1/σ²; raw χ² (fixed
  vector) logged at every evaluation; **per-series standardized
  residual score** surfaced (renamed per R3#7 — the σ framing
  forbids reduced-χ² semantics); failed-eval behavior per the
  discipline section; fast
  tests (σ handling, window clipping, frozen-mask immutability,
  retry/restart path, N-on-model-side mapping).
- **pc_model = OUTER-BOUNDARY H₂O₂ escape flux (R1#4)** — the
  observable the ring collects. Primary config (no consuming
  channels): escape ≡ production at steady state; explicit
  coarse-solve test |escape − production|/|production| ≤ 1% + sign.
  Consuming-channel ablations score escape, never production.
  j_ring_model = −pc_model·N·A_d/A_r.
- **form_cd reaction-by-reaction (R1#5):**
  cd = Σ_j (n_e_j/N_REF)·(−I_SCALE)·R_j_net (R_j_net = cathodic −
  anodic; water routes irreversible ⇒ anodic ≡ 0; N_REF = 2).
  Tests: (i) equals ledger electron-consistency cd, tolerance
  |Δ| ≤ max(1e-12 abs nondim, 1e-10 rel); (ii) isolated 2e-only
  (cd = pc) and 4e-only (pc = 0, cd carries 4e weight); (iii)
  ablation j-subset path test.
- **Grid discipline (R1#14, R2#2):** fit ITERATIONS on a
  predeclared adaptive ~17-pt grid (dense at onset + ring peak);
  then a **final L-BFGS-B POLISH on the full 30-bin-center
  objective**. FD gates, profiles, sensitivities, and the reported
  θ ALL use the final bin-center objective. Grid-convergence check
  (J + per-feature deltas at θ* and at the 17-pt optimum) within
  feature-gate tolerances, else iterations move to the denser grid
  outright.
- **FD gate (R2#3, R3#5):** central-FD h-convergence on all 4
  controls against the final objective, in SCALED optimizer
  variables; combined tolerance |g_adj − g_fd| ≤ max(0.05·|g_fd|,
  1e-3·max_k|g_k|, **τ_abs**) with τ_abs = 1e-6·(1+|J|) per scaled
  unit (objective-scale absolute floor — near-stationary gradients
  must not turn the tolerance into numerical zero); one
  directional-derivative check along a random unit vector in scaled
  space; every FD point converged and inside bounds (else
  re-centered). **The PRIMARY validation runs at a nonstationary
  interior point (X0 = θ*)**; the final-θ FD check is a regression
  smoke test only.
- Controls/bounds/X0: 4 water params, bounds inherited, X0 = θ*.
  Acid k0 = 0 (not controls).
GATE: FD + directional checks pass; escape-vs-production and
cd-ledger tests pass.

### Stage 4 — Fit + identifiability (~1.5 days + overnight)
- L-BFGS-B (adaptive grid → bin-center polish), per-eval JSON
  checkpoints; reported θ from the polished bin-center objective.
- Optimum robustness: 2 perturbed restarts (±0.5 log10 k0, ±0.05
  α); agreement within profile widths or flagged multi-modal.
- **Identifiability protocol (R1#9, R2#8):** (i) BFGS
  inverse-Hessian + central-FD diagonal check; (ii) TRUE profiles
  for the two 4e parameters (5-pt ladders, REOPTIMIZE the other 3,
  warm-started) under BOTH dual and pc-only objectives; (iii)
  **derived-quantity profiles for the partition itself (R3#4)**: 4e
  current fraction g at the plateau V and integrated over the
  window — TRUE constrained profiles: fixed-g refits with
  constraint tolerance |g(θ̂) − g₀| ≤ 0.01 (absolute, fraction
  scale; quadratic penalty with weight swept until the tolerance
  binds without distorting J by > 1%), on a g-ladder EXTENDED until
  the profile crosses Δχ² = 4 on BOTH sides of the optimum (or a
  parameter bound is hit, reported as one-sided). Checkpoint-cloud
  lower envelopes are exploratory plots only, never cited widths.
  The paper claim ("disk series identifies the partition") cites
  (ii)+(iii) widths under dual vs pc-only, nothing weaker.
  Fixed-rest slices, if plotted, carry a "not a profile" label.
- **Refit vs rescore table (R2#7):**
  | Perturbation | Treatment |
  |---|---|
  | Window upper edge ±0.05 V | REFIT |
  | OCP {0.969, 0.994, 1.044, 1.069} | REFIT |
  | L_eff {12, 21.7} µm | REFIT |
  | N {0.20, 0.25} (model-side) | REFIT |
  | bulk c_H ×{0.7, 1.4} | REFIT |
  | c_O₂ {1.0, 1.3} (if protocol unconfirmed) | REFIT |
  | Weights: w_r ×2/×½, unweighted, Huber | REFIT |
  | Ring-offset nuisance ON | REFIT |
  | Ablations: water-2e off, water-4e off, acid-on, Kw 0.5× | RESCORE |
  All refits warm-started from accepted θ; reported as parameter
  spread (structural error bars). If any structural spread exceeds
  the profile widths, the affected parameters are reported as
  conditional on that assumption (L_eff-conditional,
  OCP-conditional, transport-conditional).
- Feature gates (data axis, recomputed observables): disk onset
  ±0.05 V; |j_disk| at +0.10 V ±30%; ring peak position ±0.05 V,
  height ±30%; left-end |j_disk| ±30% of −4.0.
- Δχ² predictive-improvement vs Stage 2 reported (raw χ², fixed
  vector).
- Mini-ablation at accepted θ: water-2e off, water-4e off, acid-on
  (pre-registered probe), Kw 0.5×.

### Stage 5 — Cross-condition consistency + lock (~0.5 day)
- θ_K6.39 → re-predict slide-15 Cs⁺ pH 4 (no refit); compare
  parameter vectors. Pre-registered outcomes: (i) transferable
  within profile widths → one kinetic model; (ii) k0 shifts only →
  site-density / catalyst-history difference (catalyst ask gates
  interpretation); (iii) structural misfit → evidence handed to
  7.1's residual program.
- Joint two-condition fit ONLY if (i) or borderline (ii), with
  per-condition L_eff fixed.
- **Lock tiers (R3#3):**
  - **Computational lock** (achievable with no asks answered): fit,
    gates, sensitivities, profiles, ablations complete; θ reported
    with ALL conditional labels attached.
  - **Paper-grade claims, gated per ask:**
    | Claim | Blocking ask |
    |---|---|
    | absolute θ values / absolute-potential statements | 0.47 V OCP |
    | kinetic parameters as transport-free constants | rpm (L_eff) + O₂ protocol |
    | cross-condition transferability (vs slide-15 Cs⁺) | catalyst identity |
    | absolute partition (4e fraction) scale | ring hold + calibration (N) |
    | bump/feature chemistry follow-ons (7.1) | raw Tafel xlsx |
  No paper-grade claim ships while its blocking ask is open; the
  computational lock may.
- Lock artifacts: bin-center curves both conditions, residual
  panels, full ledger checks, FD re-gate at final θ, summary +
  `docs/phase7/phase7p2_dual_series_summary.md`, memory note.
- Standing data asks: sibling pH files; rpm; gas/T/O₂-saturation;
  catalyst; ring hold/calibration; raw Tafel xlsx; 0.47 V OCP.

## Risks
| # | Risk | Mitigation |
|---|---|---|
| 1 | Anchor recipe fails at pH 6.39 | Stage 1 escalation ladder |
| 2 | rpm unknown → L_eff wrong | REFITS {12, 21.7}; spread = structural error bar; L_eff-conditional labels; ask |
| 3 | 2019 catalyst ≠ paper CMK-3 | onset/Tafel cross-check vs Ruggiero pH 6; k0 condition-specific unless confirmed; ask |
| 4 | Non-ORR current in window | dual-threshold window + Cap σ inflation; n_e thresholded diagnostic; residual inspection |
| 5 | Ring baseline / N / hold bias | baseline subtraction + σ propagation; N model-side refits; ring-offset nuisance; asks |
| 6 | One series dominates objective | per-series reduced χ² every eval; weight-swap/unweighted/Huber REFITS |
| 7 | Kw-laundered acid artifact | acid k0 = 0 primary; pre-registered acid-on ablation |
| 8 | Optimizer corrupted by failed solves | raise-and-restart policy (no penalty values into L-BFGS-B); restart cap 3; FD points must converge |
| 9 | O₂ protocol unknown (gas/T/solubility) | c_O₂ REFITS; transport-conditional labels; ask |

---

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
