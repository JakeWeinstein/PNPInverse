# Round 1 — Plan critique: Phase 7.2 dual-series (disk + ring) fit to K₂SO₄ pH 6.39 RRDE data

## Section 1: Context bundle

### The system and the model

Steady-state 2-D Firedrake FEM of Poisson–Nernst–Planck + Butler–Volmer ORR
(O₂ → H₂O₂ → H₂O) on a CMK-3 carbon RRDE disk. Production stack:

- 3 dynamic species (O₂, H₂O₂, H⁺) in `logc_muh` formulation (proton carried
  as electrochemical potential μ_H = u_H + em·z_H·φ; the proton-condition
  residual is E = c_H − c_OH with c_OH = Kw_eff·exp(−u_H), a fast-equilibrium
  water-ionization closure).
- Analytic Bikerman (steric Boltzmann) counterions: Cs⁺ or K⁺ plus SO₄²⁻,
  physical ion radii, multi-ion shared-θ closure.
- Stern Robin BC, C_S = 0.20 F/m² (anchor built at 0.10, runtime-bumped).
- Log-rate Butler–Volmer. **Dual-pathway kinetics** (Phase 7, implemented and
  tested): each of the 2e (O₂→H₂O₂, E° = 0.695 V) and 4e (O₂→H₂O,
  E° = 1.23 V) channels exists in TWO variants:
  - acid route: rate ∝ k0·c_O₂·c_H^m·exp(−α n η/V_T), m = 2 or 4
  - water route (water as proton donor, O₂ + 2H₂O + 2e⁻ → H₂O₂ + 2OH⁻ and
    4e analog): rate ∝ k0·c_O₂·exp(−α n η/V_T), NO c_H factor, irreversible
    empirical Tafel branch (E° kept as formal onset parameter; no
    thermodynamic-consistency claim). Stoichiometry routes proton consumption
    onto the E-residual (producing OH⁻ ≡ consuming H⁺ — algebraically exact).
- Robust solve recipe (locked by tests): anchor at V_solver = 0 with
  linear_phi initializer, full Kw (no kw ladder), adaptive k0 ladder
  (max_inserts_per_step ≥ 6), then warm-walk the V grid. Continuation /
  ladders are robustness machinery, NOT fitting.
- Adjoint machinery (working, FD-verified): off-tape anchor+walk, then per-V
  on-tape re-solve (steady-state implicit-function trick with dt → ∞),
  ReducedFunctional over R-space k0/α Functions, scipy L-BFGS-B on
  (log10 k0_w2e, log10 k0_w4e, α_w2e, α_w4e). FD h-convergence verification
  protocol exists (fresh cold walks per FD point).

### What has been fit so far (Phase 7 result, current state)

Target so far: deck "slide 15" — H₂O₂ production current (RING-derived,
disk-equivalent) vs V_RHE for **Cs₂SO₄ pH 4**, 1600 rpm. Exact data NOT
available; we fit a vector-SVG extraction of the rendered curve (754
vertices → 33 bins; thresholded zero tail; one deleted outlier known).
Adjoint fit result θ* = (log10 f_2w, log10 f_4w, α_2w, α_4w) =
(−3.683, −13.537, 0.550, 0.285), fine-grid χ²/pt ≈ 29.8, 25/25 converged.
Volcano shape reproduced; structured residuals remain (trough +0.08 V
anodic, missing small bump at +0.22–0.27, left-plateau slope, total
current −5.5 vs deck ~3 mA/cm²).

KEY STRUCTURAL WEAKNESS (motivates this new plan): the slide-15 target is
peroxide-only. Total disk current is only hinge-bounded, so the 2e/4e
partition is non-identified — the 4e parameters are constrained only
through O₂-competition shape. We label every result "peroxide-only,
partition-non-identified".

A predecessor plan ("Phase 7.1", GPT session 42, 5 rounds, APPROVED, never
executed) attacks the slide-15 residuals via staged diagnostics (D-scaling
probe, L_eff bracket, bounded H₂O₂ sink), a bisulfate local-buffer
sensitivity (the only right-signed lever for the pH-transition position at
bulk pH 4), an acid-2e handoff refit, and a gated surface-transition arm.
Its conventions (sign ledger, V-axis mapping table, proton-source ledger,
escape-flux scoring, statistics framework, FD gates, acid freeze-out,
feature-metric gates) are inherited here. Phase 7.1 is parked, not
cancelled; the new data motivates running a cleaner condition first.

### The OCP / V-axis convention (recurring subtlety — read carefully)

The experimental group references ψ_bulk = V_OCP_measured with
V_OCP_RHE = 0.47 + 0.197 + 0.059·pH (0.903 V at pH 4). Our solver uses
ψ_bulk = 0, V_M = V_solver directly. For deck comparisons we apply a single
uniform shift in the driver: V_solver = V_RHE_deck − V_OCP_RHE, AND shift
both E° identically, preserving η for every reaction (fast test asserts
(V_solver − E°_solver) == (V_RHE − E°_RHE)). The 0.47 V component is
documented only in a collaborator notebook → standing confirmation gate on
absolute-potential claims. BV flank shapes are η-invariant; double-layer
state (Stern charge, crowding, migration, surface pH) is NOT η-invariant —
it depends on where the rest point actually is.

### The NEW data (verified by direct parse, numbers below are from the file)

`data/EChem Reactor Modeling-Seitz-Mangan/Brianna/0,1M K2SO4 data
8-15-19.xlsx` (Brianna Ruggiero, 2019-08-15; the group's J. Catal. 2022
paper data campaign era):

- Raw cycle sheets (`cycle 1/2/3`): columns Edisk/V (vs Ag/AgCl, raw),
  Idisk/mA, Iring/mA; in-sheet constants: N = 0.224 (RRDE collection
  efficiency), "Ref Cal vs RHE" = −0.549 V, Rs = 125 Ω, disk Ø 5 mm →
  A_d = 0.196 cm², ring OD/ID 7.5/6.5 mm → A_r = 0.110 cm².
- Processed main sheet: V_RHE iR-corrected, j_disk = I_disk/A_d,
  j_ring = I_ring/A_r, Sel% = 200·j_ring/(N·|j_disk| + j_ring) (NOTE: mixes
  densities normalized by different areas — we suspect an A_d/A_r
  inconsistency vs the canonical current-based formula; we will reconcile
  but in any case fit only the primary series), n_e column.
- Condition: 0.1 M K₂SO₄ titrated with H₂SO₄, bulk pH 6.39. 1047 points
  per cycle. V_RHE −0.06 … +1.14 V. j_disk −4.0 … +2.25 mA/cm² (cathodic
  ORR negative; anodic current at V ≳ 0.9 = carbon oxidation/OER, outside
  our model). j_ring up to 0.36 mA/cm²_ring. Peak Sel ≈ 73% at +0.30 V.
  Apparent n_e ≈ 2.8 at the cathodic plateau (consistent with −4.0 vs our
  O₂ transport ceilings 2.86 (2e) / 5.71 (4e) mA/cm² at L_eff = 15.4 µm).
- `Exp Info` sheet catalogs SIX same-day conditions: pH 6.39, 5.21, 4.21,
  3.42, 2.35, 1.65 (each with own Rs, ring onset potential, max ring
  current, peak Sel%, n_e). Only pH 6.39 LSV traces are in THIS file;
  sibling files exist with the group (data ask).
- Cycles 2 and 3 agree closely (first-row Idisk 0.0684 vs 0.0686 mA);
  cycle 1 differs grossly and carries the in-sheet remark "pH changed to
  6.11 while ring cleaning" → we exclude cycle 1 as
  conditioning/contaminated and use cycles 2+3 as replicates for σ.
- Rotation rate is NOT recorded in the file. The group's protocol
  (Ruggiero 2022 J. Catal. 414, 33) is 1600 rpm → O₂ Levich film
  δ_O₂ = 15.4 µm. Catalyst identity for this 2019 disk (CMK-3 vs GC) also
  not in-file. Both are data asks; the plan brackets L_eff and gates
  transferability claims on the catalyst answer.

Why this condition is scientifically clean for the dual-pathway model:
bulk c_H = 10^(−6.39) M = 4.07e-4 mol/m³ → H⁺ Levich cap ≈ 2.4e-3 mA/cm²,
three orders below the measured current → acid routes are negligible by
construction (no freeze-out judgment call needed, though we keep them at
k0 = 0 and run one acid-on ablation to QUANTIFY the Kw-laundering artifact:
at bulk pH 4 a diagnostic showed the acid branch can draw −3.6 mA/cm² of
fictitious current at surface pH 9 because the Kw closure manufactures
c_H from water at the rate fast-equilibrium allows). Likewise bisulfate:
[HSO₄⁻]/[SO₄²⁻] = c_H/Ka2 ≈ 4e-5 (Ka2 = 10^−1.99 M = 10.23 mol/m³) →
the 7.1 buffer stage is irrelevant here. The water routes carry the whole
curve, and the DISK series finally identifies the 2e/4e partition.

### Environment / cost realities

Each converged 13-pt coarse grid ≈ 20–40 min; 25-pt fine grid ≈ 1–1.5 h;
one L-BFGS-B fit = O(30–50) evals = overnight. Firedrake venv, streamed
logs, separate cache dirs for parallel runs. Adjoint gradients mandatory
(user rule; no derivative-free fitting).

## Section 2: The artifact under review

The full plan follows verbatim.

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
  j_ring ≤ 0.36 (ring area), peak Sel ≈ 73% @ +0.30 V, n_e ≈ 2.8.
- `Exp Info` catalogs the FULL day: pH {6.39, 5.21, 4.21, 3.42,
  2.35, 1.65}, per-disk Rs, ring onset (0.472 V at pH 6.39), max
  ring current, peak selectivity. Only the pH 6.39 LSVs are in this
  file — sibling files for the other pHs are a data ask.
- Cycles 2 and 3 are near-identical replicates (→ real σ). Cycle 1
  carries the remark "pH changed to 6.11 while ring cleaning" and
  differs grossly → excluded as conditioning/contaminated, documented.

Scientific leverage at pH 6.39: bulk c_H = 4.07e-4 mol/m³ → H⁺
Levich cap ≈ 2.4e-3 mA/cm² (3 OOM below data) and
[HSO₄⁻]/[SO₄²⁻] = c_H/Ka2 ≈ 4e-5 → **acid routes and bisulfate
buffering are both negligible by construction**. The dual-pathway
water-route kinetics carry the entire curve. This is the cleanest
available test of the Phase 7 model, AND the disk series breaks the
4e-parameter sloppiness (partition becomes identified).

## Inherited conventions (from 7.1, unchanged unless noted)

- V-axis mapping table; sign ledger (incl. anion-migration
  coordinate-free row + unit test); proton-source ledger;
  escape-flux scoring; FD h-convergence gates on every new control
  and on the accepted final model; statistics framework
  (J = (1/n)Σw r², w = 1/σ²; ΔAIC pragmatic; feature metrics are
  the real gates); OCP 0.47 V confirmation gate for
  absolute-potential claims; mini-ablation cell at acceptance.
- OCP shift at pH 6.39: V_OCP_RHE = 0.47 + 0.197 + 0.059·6.39 =
  **1.0440 V**; presets store unshifted E°; shift applied once in
  the driver; η-preservation fast test re-run at this pH. NOTE the
  in-file Ag/AgCl→RHE cal is −0.549 V vs 0.574 theoretical
  (0.197 + 0.059·6.39) — 25 mV discrepancy documented; it is THEIR
  V-axis (we fit their reported V_RHE as-is); it does not touch our
  OCP-shift convention, but both numbers go in the provenance note.
- ALL acid branches at k0 = 0 in the primary fit (7.1 freeze-out,
  here it is also physics: cap 3 OOM below data). One ablation run
  re-enables production acid k0 to MEASURE the Kw-laundering
  artifact at this condition (A1 showed −3.6 mA/cm² at surface
  pH 9 at bulk pH 4; same mechanism is available here and must be
  quantified, not assumed away).
- NOT inherited: 7.1 Stage B′ (bisulfate local-buffer) — negligible
  at pH 6.39 (ledger row verifies); D′ surface-transition arm —
  out of scope here (returns via 7.1 if the bump survives).

## Stages

### Stage 0 — Extraction + provenance (Firedrake-free, ~0.5 day)
1. `scripts/studies/_extract_k2so4_ph6p39_rrde.py`: parse RAW
   `cycle 2/3` sheets; re-derive every processed column from raw +
   in-sheet constants (cal offset, Rs, areas); CROSS-CHECK against
   the processed main sheet to ≤1e-6 rel (catches silent sign/area
   errors). Resolve and document the raw→processed sign flip (raw
   Idisk is cathodic-positive; canonical output is
   **cathodic-negative** for both series).
2. iR re-derivation: confirm the sheet formula `V + (I/1000)·Rs`
   with THEIR sign convention reproduces their iR-corrected axis;
   our canonical V is their iR-corrected V_RHE.
3. Ring → disk-equivalent peroxide current:
   pc_data = −j_ring·A_r/(N·A_d) = −j_ring·0.110/(0.224·0.196)
   (mA/cm²_disk, cathodic-negative) — same convention as the
   slide-15 target. Unit line verbatim in the extractor docstring.
4. Reconcile Brianna's Sel% formula (`200·j_ring/(N·|j_disk| +
   j_ring)`, densities on DIFFERENT areas) against the canonical
   current-based RRDE formula — flag any A_d/A_r inconsistency in
   the provenance note. We fit the two PRIMARY series only; Sel% is
   derived QA, never fit.
5. Binning + σ: common V grid ≈ 40 bins over the fit window;
   per-bin mean over cycles 2+3; σ_bin = max(cycle spread, within-
   bin scatter, σ_floor). σ_floor from the Exp Info capacitance
   band (±0.0008 mA/cm²) per series.
6. Fit window: V ∈ [−0.06, V_zero] where V_zero = cathodic-side
   zero crossing of j_disk (≈ +0.85 V; exact value from data);
   anodic disk current (carbon oxidation/OER, not in model) is
   excluded; window-edge sensitivity ±0.05 V in Stage 4.
7. Hydrodynamics: rotation rate is NOT in the file. Assume
   1600 rpm (Ruggiero protocol) → L_eff = 15.4 µm central, bracket
   {12, 15.4, 21.7} as structural uncertainty. DATA ASK: confirm
   rpm + catalyst identity (2019 disk: CMK-3 or GC?) + sibling pH
   files. Catalyst ask gates k0-transferability claims only, not
   the fit itself.
8. Outputs: `data/k2so4_ph6p39_rrde_{cycle2,cycle3,binned}.csv`
   (git add -f), QA overlay plot (our extraction vs processed
   sheet), provenance note `docs/phase7/k2so4_ph6p39_provenance.md`.
GATE: cross-check (1) passes; QA overlay visually faithful;
σ > 0 every bin.

### Stage 1 — K⁺/pH 6.39 model config (config-only, ~0.5 day)
- Driver `solver_demo_slide15_dual_pathway_cs.py` gains `--cation
  {cs,k}` (swaps DEFAULT_CSPLUS↔KPLUS_BOLTZMANN_COUNTERION_STERIC;
  K⁺ preset verified present, physical radius, D = 1.96e-9) and
  `--v-ocp-rhe` (default keeps pH 4 value; pH 6.39 runs pass
  1.0440). `--bulk-h-mol-m3 4.07e-4` (existing flag). Counterion
  bulk: K⁺ 200 mol/m³ / SO₄²⁻ 100 (electroneutral to 4e-4).
- Water ionization ON (Kw closure) — surface goes alkaline under
  cathodic current from a near-neutral bulk; OH⁻ is the dominant
  E-carrier. kw_eff_ladder = None (full Kw; Phase 2 recipe).
- Anchor recipe inherited (linear_phi IC, k0 AdaptiveLadder
  max_inserts ≥ 6, anchor at V_solver = 0 then warm walk). RISK:
  recipe was established at bulk pH 4 Cs⁺; bulk u_H shifts by
  ln(240) here. First action is a 5-point smoke grid; if the
  anchor fails: (a) re-tune ladder band, (b) bulk-H continuation
  ramp 0.1 → 4.07e-4 mol/m³ at anchor V (new small utility,
  Stern-bump pattern), (c) coarse-mesh anchor + mesh-sequenced
  restart. Escalate in that order.
- Solver window: V_solver = V_RHE − 1.0440 → [−1.10, −0.19] for
  the data window — all-cathodic walk, span comparable to the
  slide-15 grid (no new territory).
GATE: 13-pt coarse grid ≥ 11/13 converged at θ* water-only config;
ledger closures (E, O₂, electron) pass; anodic share < 1%.

### Stage 2 — Out-of-sample prediction at θ* (BEFORE any refit)
Run θ* = (−3.683, −13.537, 0.550, 0.285) — the slide-15 Cs⁺ pH 4
fit — at K⁺ pH 6.39, L = 15.4. Predict BOTH series. Score with the
Stage 3 objective but DO NOT tune anything. Plot prediction vs
data; record feature metrics: disk onset V (data ring onset
0.472 V), disk plateau magnitude (−4.0), ring peak position/height,
Sel% curve (derived). This is the transferability test of
water-route kinetics across 2.2 pH units and a cation swap — a
paper result REGARDLESS of outcome, and it must be locked in before
fitting to avoid unconscious tuning. No gate; archived as
`StudyResults/phase7p2_theta_star_prediction/`.

### Stage 3 — Dual-series objective + adjoint extension (~1 day)
- `calibration/phase7_wls.py` → add `score_dual_series`: J =
  (1/n_d)Σ w_d(cd_i − d_i)² + (1/n_r)Σ w_r(pc_i − r_i)², w = 1/σ²
  from Stage 0; per-series reduced χ² reported separately so
  neither series silently dominates; non-converged V dropped (not
  zero-filled) with count surfaced; fast tests (σ handling, PCHIP
  onto bins, window clipping, dropped-point accounting).
- Fit harness `phase7_fit_adjoint_bfgs.py`: add taped
  `form_cd` = electron-weighted sum over ALL active reactions
  (n_e_j/N_REF weights — the existing reaction_sum machinery with
  role filter removed); assemble per V alongside form_pc; J as
  above on tape. The cd observable must equal the ledger
  electron-consistency cd to 1e-10 rel on a smoke solve (fast-ish
  slow test, coarse mesh).
- FD gate: fresh-walk central-FD h-convergence on ALL 4 controls
  against the NEW dual objective (the pc-only FD pass does not
  carry over).
- Controls/bounds/X0: the 4 water params, bounds inherited,
  X0 = θ*. Acid k0 = 0 (not controls).
GATE: FD rel err ≤ 0.05 at the h-converged point for all controls.

### Stage 4 — Fit + identifiability (~1 day + overnight)
- L-BFGS-B on coarse 13-pt grid (within window), per-eval JSON
  checkpoints; final θ re-scored on fine 25-pt grid.
- Robustness of the optimum: re-run from 2 perturbed starts
  (±0.5 in each log10 k0, ±0.05 in α); agreement ≤ 1σ_profile or
  flagged multi-modal.
- Identifiability deliverable (the POINT of this phase): 1-D
  profile slices for all 4 params under the dual objective vs the
  pc-only objective at the same condition — demonstrate (or
  refute) that the disk series converts the 4e direction from
  sloppy to identified. This claim, with curvature numbers, goes
  in the paper.
- Feature gates (data axis): disk onset within ±0.05 V; |j_disk|
  at +0.10 V within ±30%; ring-equiv peak position ±0.05 V,
  height ±30%; left-end |j_disk| within ±30% of −4.0. Window-edge
  ±0.05 V and L_eff bracket sensitivity on the accepted θ.
  ΔAIC vs the Stage 2 θ* prediction reported.
- Mini-ablation at accepted θ: water-2e off, water-4e off, acid-on
  (artifact probe), Kw 0.5×.

### Stage 5 — Cross-condition consistency + lock (~0.5 day)
- θ_K6.39 → re-predict slide-15 Cs⁺ pH 4 (no refit) and compare
  parameter vectors. Three pre-registered outcomes: (i)
  transferable within profile widths → one kinetic model, strong
  paper claim; (ii) k0 shifts only (α stable) → site-density /
  catalyst-history difference, stated as such; (iii) structural
  misfit → conditions are NOT explained by shared water-route
  kinetics; 7.1's residual program inherits this as evidence.
- Joint two-condition fit ONLY if (i) or borderline (ii), and only
  with per-condition L_eff fixed.
- Lock: fine-grid curves both conditions, residual panels, full
  ledger checks, FD re-gate at final θ, summary +
  `docs/phase7/phase7p2_dual_series_summary.md`, memory note.
- Standing data asks (gating follow-ons, not this fit): sibling pH
  xlsx files (5.21 … 1.65 → pH-series out-of-sample ladder), rpm +
  catalyst confirmation, raw Tafel xlsx (slide-15 exact), 0.47 V
  OCP component.

## Risks
| # | Risk | Mitigation |
|---|---|---|
| 1 | Anchor recipe fails at pH 6.39 (bulk u_H shift ln 240) | Stage 1 escalation ladder: re-tuned k0 band → bulk-H continuation ramp → coarse-mesh anchor |
| 2 | rpm unknown → L_eff wrong | 1600 rpm assumption flagged; {12, 15.4, 21.7} structural band on accepted θ; data ask |
| 3 | 2019 catalyst ≠ paper CMK-3 | Onset/Tafel cross-check vs Ruggiero pH 6 panel; k0 treated as condition-specific unless confirmed; transferability claims gated on the ask |
| 4 | Disk series contains non-ORR current inside the window (H₂O₂ reduction, trace HER) | n_e(V) from data stays in [2,4] check; window excludes anodic; residual structure inspected before any new-physics claim |
| 5 | Ring collection/calibration bias (N, ring potential not in file) | N sensitivity {0.20, 0.224, 0.25} on accepted θ; ring onset vs Exp Info consistency check |
| 6 | Dual-series weights let one series dominate | per-series reduced χ² reported at every eval; weight-swap sensitivity (×2, ×½ on w_r) at accepted θ |
| 7 | Kw-laundered acid artifact misattributed | acid k0 = 0 primary; explicit acid-on ablation quantifies the artifact at this condition |

---

## Section 3: Critique prompt

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
