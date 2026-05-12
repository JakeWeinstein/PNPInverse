# Phase 6β Step 10 — Phase D: K-only Δ_β fit summary

**Date:** 2026-05-12.  **Verdict:** `OUTCOME_C_NON_IDENTIFIABLE_flagged`.
**Plan:** `~/.claude/plans/phase6b-step10-phase-D-deltaBeta-fit.md` (v7-FINAL).
**Driver:** `scripts/studies/phase6b_step10_phase_D_fit_eval.py`.
**Orchestrator:** `scripts/studies/phase6b_step10_phase_D_orchestrate.py`.

---

## §1. Fit setup

* **Free parameter:** scalar `Δ_β` (carbon-vs-Cu offset in Singh 2016's
  pKa-shift coefficient).  Applied uniformly via
  `β_X_carbon = β_X_Cu + Δ_β` to the Singh Eq. (4) residual.  Phase E
  (step 11) was to hold `Δ_β` fixed and predict Cs/Na/Li.
* **Data target:** deck K₂SO₄ at pH ∈ [3.5, 4.5] (4 rows from
  Brianna xlsx).  Mean H₂O₂ selectivity = **50.95 pp**, std = **27.47 pp**
  (informational; `DATA_TARGET_NOISY=True`).  See
  `StudyResults/phase6b_step10_phase_D/data_audit_K_at_pH4.json`.
* **Loss:** `|max_H2O2%_model − 50.95|` over the locked 24-point V_RHE
  grid (mask [−0.06, +1.0]).  Primary gate: ≤ 10 pp.
* **Optimizer:** `scipy.optimize.minimize_scalar(method="bounded",
  xatol = 0.05 / σ_max, maxiter=16)`.  Two paths sequentially: Stern
  (production σ from PNP/Stern solve) and Ablation
  (`override_sigma_singh_counts_pm2 = 0.141`, V-independent).
* **Stack:** V10B-locked (Γ_max=0.047, k_des=1.0, C_S=0.20 F/m²),
  K0_R4e_factor=1e-14, K₂SO₄ 4-species + Stern + parallel 2e/4e
  Ruggiero, `λ_hydrolysis=1.0` ramped via 5-rung ladder at each V.

## §2. Pre-fit identifiability evidence

### Stern pre-fit grid (7 evals + 3 dup baselines at Δ_β=0)

| T (target ΔpKa) | Δ_β (pm²)    | loss (pp²)        | status         |
|----------------:|-------------:|------------------:|---------------:|
| baseline ×3     |          0.0 | **15.628839**     | finite_valid   |
| −5.0            |  −7.765e7    | inf               | solve_failed   |
| −3.0            |  −4.659e7    | inf               | solve_failed   |
| −1.0            |  −1.553e7    | **15.628839**     | finite_valid   |
| −0.1            |  −1.553e6    | **15.628839**     | finite_valid   |
| −0.01           |  −1.553e5    | **15.628839**     | finite_valid   |
| −0.001          |  −1.548e4    | **15.628839**     | finite_valid   |
| −1e-4           |  −1.507e3    | **15.628839**     | finite_valid   |

**Loss range across 8 finite_valid Stern evals: 0.0 pp².**
**Noise floor across 3 duplicate Δ_β=0 baselines: 0.0 pp²** (deterministic).
The Stern σ-mapping is exactly degenerate over 11 orders of magnitude
of Δ_β — confirming Plan Risk #4 (`non_identifiable under local
Stern σ`).  Mechanism: Singh Eq. (4) contribution to selectivity is
σ-coordinate-invariant; local Stern σ at the OHP is ~10⁻⁷ counts/pm²
so even Δ_β ≈ −1.55e7 gives `|ΔpKa_avg|` ≪ 15 (within domain).

### Ablation pre-fit grid (6 evals + Δ_β=0 baseline)

| T (target ΔpKa) | Δ_β (pm²)    | loss (pp²)        | converged    |
|----------------:|-------------:|------------------:|-------------:|
| baseline        |          0.0 | inf               | 0/24         |
| −14.9           |    −60.07    | inf               | 0/24         |
| −10.0           |    −25.31    | inf               | 0/24         |
| −8.0            |    −11.13    | inf               | 0/24         |
| −4.0            |    +17.24    | inf               | 19/24        |
| −1.0            |    +38.52    | **15.629456**     | 24/24        |
| −0.1            |    +44.90    | **15.628884**     | 24/24        |

Under Ablation, large negative Δ_β values drive the residual past the
solver's safe domain (cation hydrolysis source overpowers Newton).
Only positive Δ_β values near the bracket upper bound converge across
all 24 V's.  Loss values where they converge: 15.629–15.628, again
essentially flat.

### D7 identifiability gate (Stern, executed in main orchestrator)

| criterion       | result   | observed                                  |
|----------------:|---------:|------------------------------------------:|
| range Δ_loss ≥ 1 pp² | **FAIL** | 0.0 pp²                              |
| noise floor (3·σ)    | pass     | (trivially, since noise_std = 0)     |
| slope ≥ 0.01 pp²/ΔpKa| **FAIL** | 0.0                                  |
| unimodality          | pass     | 0 interior minima (flat function)    |

`overall_pass = False` ⇒ orchestrator emitted
`OUTCOME_C_NON_IDENTIFIABLE_flagged` at 02:01 (UTC-local), exited
before Stern Brent.  See
`StudyResults/phase6b_step10_phase_D/identifiability_report.json`.

## §3. Fit results

### Stern path

* **Δ_β_fit:** not computed.  Orchestrator exited at D7.
* Equivalent best-loss estimate: any Δ_β in `[−1.55e7, 0]` yields
  loss = 15.628839 pp.  Mathematical degeneracy.

### Ablation path (worker; ran independently to completion)

* **Δ_β_fit:** **+45.4054 pm²** (at upper bound −β_K_Cu − ε ≈ +45.61).
* **loss_at_fit:** **15.628851 pp**.
* **n_evals:** 13 (scipy bounded Brent).
* **bracket:** `(−60.066, +45.608)`.  Brent converged via
  `success=True, message="Solution found."`.
* See `StudyResults/phase6b_step10_phase_D/ablation_brent_summary.json`.

### σ-mapping divergence (informational; not consumed by verdict)

Cannot be computed in the standard form because the Stern fit is
degenerate.  Effective answer: every Stern Δ_β within the convergent
plateau gives the same observable as Δ_β_ablation=+45.4054 → loss
disagreement = `15.629 − 15.629 ≈ 0`.  Plan Risk #4 anticipated
divergence on the ~10⁶× σ-scale difference, but here the loss
itself is flat, so divergence in observable-space is zero.

### Primary acceptance gate (locked at ±10 pp)

* deck K@pH4 mean = **50.95 pp**
* model max_H2O2% at every finite_valid eval = **66.58 pp**
* gap = **+15.63 pp**
* primary gate (≤ 10 pp): **FAIL** uniformly across all Δ_β tested

The model overshoots the deck by ~16 pp throughout the bracket.  Even
if D7 had passed (which it did not), the primary gate would have
falsified the fit.

## §4. Outcome verdict

**`OUTCOME_C_NON_IDENTIFIABLE_flagged`** is locked.

Diagnostic: the cation-hydrolysis Δ_β degree of freedom does **not**
control selectivity at the (V10B kinetics) × (K₂SO₄ stack) × (V_RHE
[−0.06, +1.0] V) production point.  The Stern σ-coordinate is
exactly invariant under Δ_β rescaling (Plan Risk #4); the Ablation
path needs `Δ_β` near the geometric upper bound to converge at all,
and even there the converging plateau is flat.  Conclusively: Δ_β
alone cannot close the deck-vs-model gap.

Secondary observation (not part of the locked verdict): the primary
acceptance gate (≤ 10 pp) would also have failed by ~5.6 pp at every
finite_valid eval.  This is `B_FALSIFIED_documented` evidence in
addition to `C_NON_IDENTIFIABLE`.  Per Plan §D8, when both verdicts
apply, **C takes precedence** (D7 is the earlier gate in the orchestrator).

## §5. Phase E status

**Phase E (step 11) must NOT launch on this Δ_β.**

Plan §D9 specifies that Phase E launches conditional on
`OUTCOME_A_LOCKED_PASS`; this run produced
`OUTCOME_C_NON_IDENTIFIABLE_flagged` instead.  An
identifiability-report has been emitted at
`StudyResults/phase6b_step10_phase_D/identifiability_report.json`
documenting the loss-curve geometry.

## §6. Open asks (Phase D' / Phase 6γ scoping)

The Δ_β-alone fit does NOT explain the 15.6 pp deck-vs-model gap.
Future scope needs to add additional degrees of freedom or
reconsider the cation-hydrolysis structure.  Candidates:

1. **`k_des` or `Γ_max` re-fit.**  Plan §7 lists these as out of
   scope for Phase D ("V10B locked; falsification in Phase D opens
   a separate re-derivation, NOT a scope expansion").  This is the
   trigger: Phase D is falsified, so step 12+ should consider a
   data-driven Γ_max + k_des fit (Phase D' or Phase 6γ).
2. **r_H_El_pm sensitivity.**  Phase 6β v9 Gate 4B treated r_H_El
   as a calibration sweep parameter.  The Cu prior (200.98 pm for
   K+) may not transfer to CMK-3 carbon.  However, r_H_El affects
   `β_per_cation_Cu` and thus the Δ_β offset's effective magnitude;
   sweeping r_H_El concurrent with Δ_β would be needed.
3. **Local-pH / mass-transport coupling.**  The flat selectivity
   across V in the model (66.58% essentially V-independent in mask)
   suggests the model is in a transport-limited regime where the
   cation hydrolysis source is too weak to break selectivity-vs-V
   degeneracy.  Re-examine the H⁺ Levich limit and the σ_S
   coordinate convention.
4. **Singh formula structure validity.**  Plan §3.1 locked the
   Singh formula structure as a hard invariant.  If the Δ_β alone
   cannot match the deck, the formula structure (β · σ_singh) may
   need revisiting — though that is a bigger paradigm change
   beyond the original Phase D scope.

## §7. Known discrepancy — dynamic-species `a_nondim` placeholders

Surfaced 2026-05-12 during shape-diagnostic follow-up to this Phase D
verdict.  The Bikerman steric `a_nondim` is set per-species, but the
current `SpeciesConfig` presets only use physical radii for the
counterions:

| Species | Stack role | `a_nondim` used | Implied r (Å) | Physical? |
|---|---|---|---|---|
| O₂ | dynamic | `A_DEFAULT = 0.01` | **≈ 14.9** | ✗ (Marcus ≈ 1.7 Å, a ≈ 1.49e-5) |
| H₂O₂ | dynamic | `A_DEFAULT = 0.01` | **≈ 14.9** | ✗ (≈ 2.0 Å, a ≈ 2.42e-5) |
| H⁺ | dynamic | `A_DEFAULT = 0.01` | **≈ 14.9** | ✗ (H₃O⁺ Stokes 2.8 Å, a ≈ 6.65e-5) |
| K⁺ / Cs⁺ | counterion | `A_KPLUS_HAT` / `A_CSPLUS_HAT` | 2.3 / 2.2 | ✓ Linsey deck slide 13 |
| SO₄²⁻ | counterion | `A_SO4_HAT` | 2.4 | ✓ (placeholder) |
| OH⁻ (kw on) | analytic | `A_OH_HAT` | 1.76 | ✓ Marcus |

**Mechanism.** `A_DEFAULT = 0.01` is ~150× larger than the realistic
H⁺ value (6.65e-5), so the Bikerman cap on local H⁺ accumulation at
the OHP under cathodic polarization is `c_max ≈ 1/a` ≈ 100 nondim
(≈ 120 mol/m³) — clamped ~150× tighter than the physical r=2.8 Å cap
would give.  This directly throttles surface H⁺ concentration and
therefore the local-pH feedback into σ_singh and into the 2e/4e
Levich limits.  The Phase D plateau topology may carry an artifact
from this clamp, independent of the Δ_β identifiability finding.

**Status (2026-05-12):** four bridge runs queued at deck-baseline
config (V10B kinetics, Stern=0.20, no cation hydrolysis, no kw) to
disentangle.  Two carry the legacy `A_DEFAULT`, two carry physical
`a_O2 = 1.49e-5`, `a_H2O2 = 2.42e-5`, `a_HP = 6.65e-5`.  Outputs in
`StudyResults/phase6b_step10_phase_D_no_hydrolysis_bridge*` and
`StudyResults/phase6b_step10_phase_D_bridge_corrected_a*`.  See the
diagnostic scripts `scripts/studies/_phase_D_bridge_*.py`.

**Verdict implication.** The locked C-verdict (Δ_β non-identifiable
on Stern σ) is independent of this discrepancy: σ_singh is V-flat
under Stern at every Δ_β tested, regardless of how H⁺ packs.  But
the secondary B-falsified observation (uniform +15.6 pp overshoot of
deck K@pH4) may not be robust under physical a_HP, and the bridge
runs will say whether a fit re-attempt (Phase D' / 6γ) should use
the corrected steric.

## Artifacts

* `StudyResults/phase6b_step10_phase_D/data_audit_K_at_pH4.json` (10.A.0)
* `StudyResults/phase6b_step10_phase_D/identifiability_report.json` (10.B.5 verdict)
* `StudyResults/phase6b_step10_phase_D/ablation_brent_summary.json` (Ablation Brent)
* `StudyResults/phase6b_step10_phase_D/eval_db_*.json` (per-eval forward solves)
* `StudyResults/phase6b_step10_phase_D/anchor_cache_stern_fcae57e2.pkl` (anchor cache)
* `StudyResults/phase6b_step10_phase_D_no_hydrolysis_bridge*/iv_curve.json` (§7 bridges, legacy a)
* `StudyResults/phase6b_step10_phase_D_bridge_corrected_a*/iv_curve.json` (§7 bridges, physical a)
