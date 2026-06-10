# Round 2 counterreply — k2so4-dual-series-fit

All 18 accepted. Per-issue disposition, then the updated artifact.

**Re 1 (OCP axis mismatch):** Accept — you are right that the file's
own measured calibration is the axis the data lives on, so the 0.47 V
component (documented vs Ag/AgCl) composes with +0.549, not the
theoretical +0.574. Central shift is now **1.019 V**; refit
sensitivities at 1.044 (theoretical-cal variant) and 0.994 (−25 mV);
both E° shifted identically with the η-preservation test re-run; the
0.47 V confirmation gate inherits unchanged.

**Re 2 (dropped non-converged points):** Accept — new "Convergence
discipline" section: frozen pre-fit mask; any objective-V convergence
failure ⇒ failed evaluation with predeclared large penalty (1e6) and
logged event; accepted θ must converge at every scoring V. No silent
drops anywhere in the objective.

**Re 3 (≥11/13 gate):** Accept — Stage 1 gate is now 13/13 on the
smoke grid; persistent single-V failures at θ* (post escalation
ladder) are removed from the mask BEFORE fitting, documented, never
re-added mid-fit.

**Re 4 (pc_model = production vs escape):** Accept — pc_model is now
defined as the outer-boundary H₂O₂ escape flux (the ring's actual
observable). In the primary config (no consuming channels) escape ≡
production at steady state, but that equality is now an explicit
Stage-3 test (≤1% + sign), not an assumption; all consuming-channel
ablations score escape, never production.

**Re 5 (form_cd underspecified):** Accept — written
reaction-by-reaction in the plan: cd = Σ_j (n_e_j/N_REF)·(−I_SCALE)·
R_j_net, R_j_net = cathodic − anodic (water routes irreversible ⇒
anodic ≡ 0), N_REF = 2, no consuming/side channels in the primary
config. Tests: ledger-cd equality to 1e-10 rel; isolated 2e-only
(cd = pc) and 4e-only (pc = 0, cd carries 4e weight); explicit
j-subset path test for ablations.

**Re 6 (window includes background):** Accept — window upper edge is
now a dual-threshold rule (largest V where |j_disk| > max(3·σ_floor,
2·Cap band) sustained over 5 raw points), which stops the window
where disk signal sinks into background, strictly cathodic of the
zero crossing. Onset-region bins get Cap-band σ inflation in
quadrature. Upper-edge ±0.05 V is a REFIT sensitivity.

**Re 7 (σ model too weak):** Accept — σ_bin = max(paired
|cycle2 − cycle3|/√2 interpolated at bin centers, per-series floors
(disk and ring floors set separately from the capacitance band and
ring noise, units verified), 2% fractional model-error floor).
Within-bin scatter demoted to diagnostic (slope-contaminated).
Unweighted + Huber refit sensitivity at accepted θ.

**Re 8 (AIC misuse):** Accept — ΔAIC language removed. Optimization
J keeps per-series normalization; all statistics use raw
χ² = Σ(r/σ)² on the fixed frozen data vector, logged at every eval;
θ*-vs-fit comparison is a "Δχ² predictive-improvement score",
explicitly not a likelihood-ratio statistic.

**Re 9 (slices ≠ profiles):** Accept — identifiability protocol is
now: BFGS inverse-Hessian + FD diagonal check; TRUE profiles
(reoptimize the other 3, warm-started 5-pt ladders) for the two 4e
parameters under BOTH the dual and pc-only objectives; the paper
claim is made from those profile widths only. Fixed-rest slices, if
plotted, carry a "not a profile" label.

**Re 10 (L_eff post-hoc):** Accept — full warm-started REFITS at 12
and 21.7 µm; across-L_eff spread reported as structural error bar;
if spread > profile widths, parameters are stated L_eff-conditional
pending the rpm confirmation.

**Re 11 (Sel%/n_e area-suspect):** Accept — Stage 0 recomputes Sel%
and n_e from raw CURRENTS with the canonical formulas
(Sel = 200·(I_r/N)/(|I_d| + I_r/N); n_e = 4|I_d|/(|I_d| + I_r/N));
processed columns demoted to provenance cross-checks; all QA and
gates use recomputed quantities; the n_e ∈ [2,4] gate moved onto the
recomputed series.

**Re 12 (onset conflation):** Accept — separate predeclared
definitions: ring onset = V at j_ring = 0.01 mA/cm²_ring (Exp Info
convention, 0.472 V); disk onset = V at |j_disk| = 0.05 mA/cm²_disk.
Gates target each separately.

**Re 13 (ring hold/N validity):** Accept — ring hold potential +
ring calibration added to the data asks; N handled by warm-started
REFITS at {0.20, 0.25} (not post-hoc rescoring); pc-scale
uncertainty propagated into the partition claim, which carries the
N band until the calibration ask is answered.

**Re 14 (13-pt grid too sparse):** Accept — fit iterations move to a
predeclared adaptive ~17-pt grid concentrated at onset + ring peak;
final scoring and all gates solve AT the ~30 bin centers (no
model-side interpolation); grid-convergence check (J and per-feature
deltas, θ* and optimum) must sit within feature-gate tolerances or
iterations move to the denser grid.

**Re 15 (acid-on ablation undefined):** Accept — pre-registered:
acid k0/α at the production values used in diagnostic A1, water
params at accepted θ_K, no reoptimization; metrics: acid share of cd
and of the E-sink per V, Δcd(V), Δpc(V), reported at max-share V.

**Re 16 (Stage 2 overclaimed):** Accept — relabeled "pre-registered
transferability diagnostic"; writeup attaches θ* profile-width and
digitization caveats and explicitly allows "poor prediction may
reflect θ* uncertainty, not failed transferability".

**Re 17 (pH activity convention):** Accept — declared
concentration-pH interpretation (c_H = 10^−pH M), ≤0.11 pH-unit
activity caveat at I ≈ 0.3 M; bulk c_H ×{0.7, 1.4} sensitivity at
accepted θ; γ_H/Kw_eff folded into the Kw 0.5× ablation; surface-pH
claims carry the caveat.

**Re 18 (brittle 1e-6 rel):** Accept — combined tolerance
|Δ| ≤ max(1e-8 abs, 1e-6 rel) with separate explicit sign and unit
assertions.

## Updated artifact

Changes this round: OCP central value 1.019 V + bracket; new
"Convergence discipline" section (frozen mask, failed-eval penalty,
13/13 gate); pc_model = escape flux + equality test; form_cd
explicit formula + isolated-reaction tests; dual-threshold window +
Cap-band σ inflation; replicate-difference σ model with per-series
floors + fractional floor; raw-χ² statistics, ΔAIC removed; true
profile likelihoods for the 4e pair under both objectives; L_eff and
N become warm-started refits; canonical Sel%/n_e recomputation;
separate onset definitions; adaptive 17-pt iteration grid + 30-pt
bin-center scoring + grid-convergence check; pre-registered acid-on
ablation; Stage 2 relabeled with uncertainty framing; pH-convention
declaration; abs+rel cross-check tolerance; risk table updated
(risk 8 added).

---

# Phase 7.2 — Dual-series (disk + ring) fit to the K₂SO₄ pH 6.39 RRDE data

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
  Only the pH 6.39 LSVs are in this file — sibling files for the
  other pHs are a data ask.
- Cycles 2 and 3 are near-identical replicates (→ real σ). Cycle 1
  carries the remark "pH changed to 6.11 while ring cleaning" and
  differs grossly → excluded as conditioning/contaminated, documented.

Scientific leverage at pH 6.39: bulk c_H ≈ 4.07e-4 mol/m³ → H⁺
Levich cap ≈ 2.4e-3 mA/cm² (3 OOM below data) and
[HSO₄⁻]/[SO₄²⁻] = c_H/Ka2 ≈ 4e-5 → **acid routes and bisulfate
buffering are both negligible by construction**. The dual-pathway
water-route kinetics carry the entire curve, and the disk series
makes the 2e/4e partition identifiable (to be DEMONSTRATED, not
assumed — Stage 4 deliverable).

## Inherited conventions (from 7.1, unchanged unless noted)

- V-axis mapping table; sign ledger (incl. anion-migration
  coordinate-free row + unit test); proton-source ledger;
  escape-flux scoring; FD h-convergence gates on every new control
  and on the accepted final model; OCP 0.47 V confirmation gate for
  absolute-potential claims; mini-ablation cell at acceptance.
- **Statistics framework (revised per R1#8):** the OPTIMIZATION
  objective J may use per-series normalization, but all model
  comparison uses **raw χ² = Σ (r/σ)² over a FIXED data vector**
  (both series, frozen mask, no per-series normalization, no
  dropped points). θ* prediction vs fitted θ is reported as
  **Δχ² predictive-improvement score** — explicitly NOT AIC, NOT a
  likelihood-ratio test (bins correlated, σ partly systematic).
- **OCP shift (revised per R1#1):** the 0.47 V GC-OCP component is
  documented on the Ag/AgCl axis; this dataset's V_RHE axis was
  built with the file's MEASURED calibration (+0.549 V), not the
  theoretical 0.197 + 0.059·pH = 0.574 V. Consistency with the
  data's own axis ⇒ central shift **V_OCP = 0.47 + 0.549 =
  1.019 V**; sensitivity runs at 1.044 V (theoretical-cal variant)
  and 0.994 V (−25 mV) bracket the convention uncertainty; both E°
  shifted identically (η-preservation fast test re-run). The 0.47 V
  confirmation gate inherits unchanged.
- **pH convention (per R1#17):** we interpret reported pH as
  concentration-pH (c_H = 10^−pH M) for bulk BCs. At I ≈ 0.3 M the
  activity correction is ≤ 0.11 pH units; sensitivity bulk c_H
  ×{0.7, 1.4} at accepted θ; Kw_eff/γ_H sensitivity folded into the
  Kw 0.5× ablation. Surface-pH claims carry this caveat verbatim.
- ALL acid branches at k0 = 0 in the primary fit (7.1 freeze-out;
  here also physics: H⁺ cap 3 OOM below data). **Acid-on ablation
  pre-registered (per R1#15):** acid k0/α at the production
  slide-15 values used in diagnostic A1, water params at accepted
  θ_K, NO reoptimization; metrics: acid share of cd and of E-sink
  per V, Δcd(V), Δpc(V); reported at the V of max share. Purpose:
  quantify the Kw-laundering artifact at this condition.
- NOT inherited: 7.1 Stage B′ (bisulfate local-buffer) — negligible
  at pH 6.39 (ledger row verifies); D′ surface-transition arm —
  out of scope here (returns via 7.1 if the bump survives).

## Convergence discipline (per R1#2, #3 — applies to all stages)

- The fit grid and data mask are FROZEN before optimization
  (Stage 1 output). Any evaluation in which any objective voltage
  fails to converge is a **failed evaluation**: J returns a large
  predeclared penalty (1e6), the gradient is not trusted, and the
  event is logged. Non-converged points are NEVER silently dropped
  inside the objective.
- Stage 1 gate is therefore **13/13 on the smoke grid**; if a
  specific V fails persistently at θ* after the escalation ladder,
  it is removed from the frozen mask BEFORE fitting, documented,
  and never re-added mid-fit.
- The ACCEPTED θ must converge at every voltage of the final
  scoring grid (no mask beyond the frozen one).

## Stages

### Stage 0 — Extraction + provenance (Firedrake-free, ~0.5 day)
1. `scripts/studies/_extract_k2so4_ph6p39_rrde.py`: parse RAW
   `cycle 2/3` sheets; re-derive every processed column from raw +
   in-sheet constants (cal offset, Rs, areas); CROSS-CHECK against
   the processed main sheet with **combined tolerance
   |Δ| ≤ max(1e-8 abs, 1e-6 rel)** (per R1#18; abs floor in mA/cm²
   resp. V), plus separate explicit sign and unit assertions.
   Resolve and document the raw→processed sign flip (raw Idisk is
   cathodic-positive; canonical output is **cathodic-negative** for
   both series).
2. iR re-derivation: confirm the sheet formula `V + (I/1000)·Rs`
   with THEIR sign convention reproduces their iR-corrected axis;
   our canonical V is their iR-corrected V_RHE.
3. Ring → disk-equivalent peroxide current:
   pc_data = −j_ring·A_r/(N·A_d) = −j_ring·0.110/(0.224·0.196)
   (mA/cm²_disk, cathodic-negative) — same convention as the
   slide-15 target. Unit line verbatim in the extractor docstring.
4. **Recompute Sel% and n_e canonically from raw CURRENTS** (per
   R1#11): Sel = 200·(I_r/N)/(|I_d| + I_r/N), n_e = 4|I_d|/(|I_d| +
   I_r/N). Brianna's in-sheet Sel% formula
   (200·j_ring/(N·|j_disk| + j_ring), densities on different areas)
   is reconciled in the provenance note; processed Sel%/n_e columns
   are PROVENANCE ONLY — every QA plot and feature gate uses the
   recomputed quantities.
5. Binning + σ (per R1#7): common V grid ≈ 30–40 bins over the fit
   window; per-bin mean of cycles 2+3. σ_bin per series =
   max( |cycle2 − cycle3|_interp/√2 (paired replicate difference at
   bin centers), σ_floor_series, f_model·|j_bin| ) with
   σ_floor_disk and σ_floor_ring set SEPARATELY from the Exp Info
   capacitance band (units verified in Stage 0) and the ring noise
   floor; f_model = 0.02 fractional model-error floor. Within-bin
   scatter is NOT used as σ (slope-contaminated); it is reported as
   a diagnostic only. Robustness: unweighted and Huber-weighted
   refit sensitivity at accepted θ (Stage 4).
6. Fit window (per R1#6): lower edge −0.06 V (data start). Upper
   edge predeclared by dual threshold: largest V where
   |j_disk| > max(3·σ_floor_disk, 2·Cap_band) SUSTAINED (5
   consecutive raw points) — i.e. the window stops where disk
   signal sinks into the capacitive/background band, NOT at the
   zero crossing; anodic carbon-oxidation/OER region excluded a
   fortiori. Onset-region bins additionally get σ inflated by the
   Cap band in quadrature. Window-edge sensitivity: refit at upper
   edge ±0.05 V (Stage 4).
7. Hydrodynamics: rotation rate is NOT in the file. Assume
   1600 rpm (Ruggiero protocol) → L_eff = 15.4 µm central. L_eff
   handling per R1#10 lives in Stage 4 (refits, not rescoring).
   DATA ASKS: rpm; catalyst identity (2019 disk: CMK-3 or GC?);
   ring hold potential + ring calibration (per R1#13); sibling pH
   files. Catalyst/ring asks gate transferability and absolute-N
   claims, not the fit itself.
8. Outputs: `data/k2so4_ph6p39_rrde_{cycle2,cycle3,binned}.csv`
   (git add -f), QA overlay plot (our extraction vs processed
   sheet; recomputed vs sheet Sel% overlay), provenance note
   `docs/phase7/k2so4_ph6p39_provenance.md` (incl. the −0.549 vs
   −0.574 calibration note and the Sel% formula reconciliation).
GATE: cross-check (1) passes; QA overlay faithful; σ > 0 every bin;
recomputed n_e(V) ∈ [2, 4] over the window (else the window or the
data understanding is wrong — stop and diagnose).

### Stage 1 — K⁺/pH 6.39 model config (config-only, ~0.5 day)
- Driver `solver_demo_slide15_dual_pathway_cs.py` gains `--cation
  {cs,k}` (swaps DEFAULT_CSPLUS↔KPLUS_BOLTZMANN_COUNTERION_STERIC;
  K⁺ preset verified present, physical radius, D = 1.96e-9) and
  `--v-ocp-rhe` (pH 6.39 central runs pass 1.019).
  `--bulk-h-mol-m3 4.07e-4` (existing flag). Counterion bulk: K⁺
  200 mol/m³ / SO₄²⁻ 100 (electroneutral to 4e-4).
- Water ionization ON (Kw closure); kw_eff_ladder = None (full Kw).
- Anchor recipe inherited (linear_phi IC, k0 AdaptiveLadder
  max_inserts ≥ 6, anchor at V_solver = 0 then warm walk). RISK:
  recipe was established at bulk pH 4 Cs⁺. Escalation ladder if the
  anchor fails: (a) re-tune k0 ladder band, (b) bulk-H continuation
  ramp 0.1 → 4.07e-4 mol/m³ at anchor V (Stern-bump pattern),
  (c) coarse-mesh anchor + mesh-sequenced restart.
- Solver window: V_solver = V_RHE − 1.019 → ≈ [−1.08, −0.17] —
  all-cathodic walk, span comparable to the slide-15 grid.
GATE (per R1#3): **13/13 on the smoke grid** at θ* water-only
config (after any documented mask freeze); ledger closures (E, O₂,
electron) pass; anodic share < 1%.

### Stage 2 — Pre-registered transferability diagnostic at θ*
(BEFORE any refit; relabeled per R1#16.) Run θ* = (−3.683, −13.537,
0.550, 0.285) — the slide-15 Cs⁺ pH 4 fit — at K⁺ pH 6.39,
L = 15.4. Predict BOTH series; score with the Stage 3 raw-χ²
machinery; DO NOT tune. Uncertainty context attached: θ*'s 4e
parameters are weakly identified on peroxide-only data (profile
widths from the Phase 7 fit) and θ* was fit to a digitized rendered
curve — a poor prediction may reflect θ* uncertainty rather than
failed transferability, and the writeup says so. Feature metrics
recorded with SEPARATE onset definitions (per R1#12): ring onset =
V at j_ring = 0.01 mA/cm²_ring (Exp Info convention; data: 0.472 V);
disk onset = V at |j_disk| = 0.05 mA/cm²_disk (predeclared); disk
plateau magnitude; ring-equiv peak position/height; recomputed Sel%
curve (derived QA). Archived as
`StudyResults/phase7p2_theta_star_prediction/`. No gate.

### Stage 3 — Dual-series objective + adjoint extension (~1 day)
- `calibration/phase7_wls.py` → `score_dual_series`: optimization
  J = (1/n_d)Σ w_d(cd_i − d_i)² + (1/n_r)Σ w_r(pc_i − r_i)²,
  w = 1/σ²; **raw χ² (fixed vector, both series) computed and
  logged at every evaluation** alongside J; per-series reduced χ²
  surfaced; failed-eval penalty per the convergence discipline;
  fast tests (σ handling, window clipping, frozen-mask
  immutability, penalty path).
- **pc_model defined as the OUTER-BOUNDARY H₂O₂ escape flux** (per
  R1#4), the observable the ring actually collects. In the primary
  fit (no consuming channels) escape ≡ reaction-sum production at
  steady state; Stage 3 adds the explicit coarse-solve test:
  |escape − production|/|production| ≤ 1% AND sign matches
  cathodic-negative convention (7.1 escape-flux conventions). All
  ablations with consuming channels score escape flux, never
  production.
- **form_cd written reaction-by-reaction (per R1#5):**
  cd = Σ_j (n_e_j/N_REF)·(−I_SCALE)·R_j_net over the BV reaction
  list (R_j_net = cathodic − anodic branch; water routes
  irreversible ⇒ anodic ≡ 0; N_REF = 2; no consuming/side channels
  exist in the primary config). Tests: (i) cd equals the ledger
  electron-consistency cd to 1e-10 rel on a smoke solve; (ii)
  isolated-reaction runs: 2e-only ⇒ cd = pc (electron-weighted
  equality), 4e-only ⇒ pc = 0 and cd carries the 4e weight; (iii)
  ablation configs exercise the j-subset path explicitly.
- **Model V-grid discipline (per R1#14):** fit ITERATIONS solve on
  an adaptive ~17-pt grid concentrated around the onset and
  ring-peak windows (predeclared spacing from Stage 0 features);
  final scoring and all gates solve AT the bin centers (~30 pts, no
  model-side interpolation). Grid-convergence check: |J_17 − J_30|
  and per-feature deltas at θ* and at the optimum must be within
  the feature-gate tolerances, else iterate on the denser grid.
- FD gate: fresh-walk central-FD h-convergence on ALL 4 controls
  against the NEW dual objective (the pc-only FD pass does not
  carry over).
- Controls/bounds/X0: the 4 water params, bounds inherited,
  X0 = θ*. Acid k0 = 0 (not controls).
GATE: FD rel err ≤ 0.05 at the h-converged point for all controls;
escape-vs-production and cd-ledger tests pass.

### Stage 4 — Fit + identifiability (~1.5 days + overnight)
- L-BFGS-B on the adaptive grid, per-eval JSON checkpoints; final
  θ re-scored at bin centers.
- Optimum robustness: 2 perturbed restarts (±0.5 in each log10 k0,
  ±0.05 in α); agreement within profile widths or flagged
  multi-modal.
- **Identifiability protocol (per R1#9):** (i) BFGS inverse-Hessian
  estimate + central-FD diagonal check at the optimum; (ii) TRUE
  profiles for the two 4e parameters (fix the parameter on a 5-pt
  ladder, REOPTIMIZE the other three, warm-started from neighbors);
  (iii) the same two profiles under the pc-only objective at this
  condition. The paper claim ("disk series identifies the
  partition") is made from (ii) vs (iii) curvature/width numbers,
  nothing weaker. 1-D fixed-rest slices may be plotted but carry a
  "not a profile" label.
- **L_eff structural handling (per R1#10):** full warm-started
  REFITS at 12 and 21.7 µm (not rescoring). Kinetic-parameter
  claims carry the across-L_eff spread as a structural error bar;
  if the spread exceeds the profile widths, parameter values are
  reported as L_eff-conditional pending the rpm ask.
- **N sensitivity (per R1#13):** warm-started refits at N = 0.20
  and 0.25; pc-scale uncertainty propagated into the partition
  claim.
- Feature gates (data axis, recomputed observables): disk onset
  within ±0.05 V; |j_disk| at +0.10 V within ±30%; ring-equiv peak
  position ±0.05 V, height ±30%; left-end |j_disk| within ±30% of
  −4.0. Sensitivities at accepted θ: window upper edge ±0.05 V
  (refit), weight swap (w_r ×2, ×½), unweighted/Huber, bulk c_H
  ×{0.7, 1.4}, OCP shift {0.994, 1.019, 1.044} (refit at the two
  variants).
- Δχ² predictive-improvement vs the Stage 2 θ* prediction reported
  (raw χ², fixed vector).
- Mini-ablation at accepted θ: water-2e off, water-4e off, acid-on
  (pre-registered artifact probe), Kw 0.5×.

### Stage 5 — Cross-condition consistency + lock (~0.5 day)
- θ_K6.39 → re-predict slide-15 Cs⁺ pH 4 (no refit) and compare
  parameter vectors. Three pre-registered outcomes: (i)
  transferable within profile widths → one kinetic model, strong
  paper claim; (ii) k0 shifts only (α stable) → site-density /
  catalyst-history difference, stated as such (catalyst ask gates
  the interpretation); (iii) structural misfit → conditions are NOT
  explained by shared water-route kinetics; 7.1's residual program
  inherits this as evidence.
- Joint two-condition fit ONLY if (i) or borderline (ii), and only
  with per-condition L_eff fixed.
- Lock: bin-center curves both conditions, residual panels, full
  ledger checks, FD re-gate at final θ, summary +
  `docs/phase7/phase7p2_dual_series_summary.md`, memory note.
- Standing data asks (gating follow-ons, not this fit): sibling pH
  xlsx files (5.21 … 1.65 → pH-series out-of-sample ladder), rpm +
  catalyst + ring-hold/calibration confirmation, raw Tafel xlsx
  (slide-15 exact), 0.47 V OCP component.

## Risks
| # | Risk | Mitigation |
|---|---|---|
| 1 | Anchor recipe fails at pH 6.39 (bulk u_H shift ln 240) | Stage 1 escalation ladder: re-tuned k0 band → bulk-H continuation ramp → coarse-mesh anchor |
| 2 | rpm unknown → L_eff wrong | 1600 rpm assumption flagged; REFITS at {12, 21.7}; spread = structural error bar; claims L_eff-conditional if spread > profile width; data ask |
| 3 | 2019 catalyst ≠ paper CMK-3 | Onset/Tafel cross-check vs Ruggiero pH 6 panel; k0 condition-specific unless confirmed; transferability claims gated on the ask |
| 4 | Disk series contains non-ORR current inside the window (background, H₂O₂ reduction, trace HER) | dual-threshold window + Cap-band σ inflation near onset; recomputed n_e(V) ∈ [2,4] gate; residual structure inspected before any new-physics claim |
| 5 | Ring collection/calibration bias (N, ring hold not in file) | N refits {0.20, 0.25}; ring onset vs Exp Info consistency; ring-hold data ask; partition claim carries N band |
| 6 | Dual-series weights let one series dominate | per-series reduced χ² at every eval; weight-swap + unweighted/Huber sensitivity at accepted θ |
| 7 | Kw-laundered acid artifact misattributed | acid k0 = 0 primary; pre-registered acid-on ablation quantifies the artifact |
| 8 | Optimizer exploits convergence failures | frozen mask + failed-eval penalty; accepted θ converges at every scoring V |

---

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
