# R1 — Adversarial review of Phase 6β v10a Phase A.2 plan

## Context bundle

### What this project is

PNPInverse is a Poisson–Nernst–Planck / Butler–Volmer (PNP-BV) forward
solver targeting ORR (O₂ → H₂O₂ → H₂O) on the Seitz/Mangan deck.
Production stack:

* **3 dynamic species** (O₂, H₂O₂, H⁺) + **K⁺ as a 4th NP species**
  (`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`) + analytic-Bikerman SO₄²⁻
  counterion.
* `formulation='logc_muh'` (proton electrochemical-potential primary
  variable: `μ_H = u_H + em·z_H·φ`).
* Log-rate Butler-Volmer with **parallel R2e (E°=0.695 V) + R4e
  (E°=1.23 V)**.
* Finite Stern compact layer; production target `C_S = 0.20 F/m²`
  (Bohra-Koper-Choi consensus, set today 2026-05-10).
* Phase 6α: water self-ionization residual `c_OH = K_w/c_H` enabled
  via `kw_eff_ladder`.
* Phase 6β: cation hydrolysis at the OHP, `M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺`,
  with field-dependent pKa per Singh 2016 JACS Eq. (4).
* Phase 6β v10a (2026-05-10): Langmuir cap `(1 − Γ/Γ_max)` on the
  cation-hydrolysis forward branch.

### What just happened: v10a' V-sweep diagnostic (2026-05-10)

Ran a 7-point V_RHE diagnostic at the production C_S=0.20 +
K0_R4e_factor=1e-14, with v10a smoke kinetics:

* `k_hyd_nondim = 1e-3, k_des_nondim = 1.0, k_prot_nondim = 1e-3,
  Γ_max_nondim = 0.047 (1 monolayer MOH), δ_OHP_hat = 4e-6,
  r_H_El_pm = 200.98 (K⁺ Cu prior)`.

Returned **V_kin = −0.10 V** via the primary path of the locked
acceptance-bundle V_kin selection rule (no fallback used).
Locked rule: argmax `|dRnet_dsigma_along_stern_capacitance|` subject to
σ_S<0 ∩ |cd|/I_lim_4e<0.9 ∩ x_2e ∈ [0.05, 0.95].

At V_kin = −0.10:

| Quantity | Value |
|---|---|
| σ_S | −0.017 C/m² |
| cd | −3.12 mA/cm² |
| |cd|/I_lim_4e | 0.567 |
| x_2e (R_2e/(R_2e+R_4e)) | 0.199 |
| θ (= Γ/Γ_max) | 0.861 |
| Γ | 0.0405 |
| F0_avg | 0.291 |
| denom_cap_to_total_ratio | 0.861 |
| o2_flux_levich_ratio | 0.631 |
| dRnet_dsigma_along_stern_capacitance | −0.187 |
| amplification_from_c_K (in F0 decomp) | 1.748 |
| amplification_from_singh | 1.000011 |
| pka_shift_avg | −4.88e-6 |

Per-V-grid summary (the σ<0 fallback-valid set is V ∈ {−0.10, −0.30, −0.50}):

| V_RHE | σ<0 | cd_ok | branch_ok | 3pass | sensS | denom_cap/T | θ | x_2e |
|---|---|---|---|---|---|---|---|---|
| +0.55 | F | T | F | F | −0.104 | 0.318 | 0.318 | 0.0024 |
| +0.40 | F | T | F | F | −0.154 | 0.446 | 0.446 | 0.0098 |
| +0.20 | F | T | T | F | −0.220 | 0.596 | 0.596 | 0.0593 |
| +0.10 | F | T | T | F | −0.249 | 0.687 | 0.687 | 0.130 |
| **−0.10** | **T** | **T** | **T** | **T** | **−0.187** | **0.861** | **0.861** | **0.199** |
| −0.30 | T | F | F | F | −0.064 | 0.948 | 0.948 | 0.0007 |
| −0.50 | T | F | F | F | −0.021 | 0.976 | 0.976 | <1e-5 |

Decision tree precedence guards:
1. is_transport_artifact at V_kin: o2levich = 0.631 < 0.9 ⇒ NOT artifact.
2. is_cap_dominated_routing: requires ALL σ<0 fallback-valid V to satisfy
   `denom_cap/total > 0.8 AND θ > 0.9 AND |sensS| < 0.10`.
   * V=−0.10: denom_cap/T=0.861 ✓, θ=0.861 ✗ (<0.9), |sensS|=0.187 ✗ (>0.10)
   * V=−0.30: 0.948 ✓, 0.948 ✓, 0.064 ✓
   * V=−0.50: 0.976 ✓, 0.976 ✓, 0.021 ✓
   * V=−0.10 fails → NOT cap-dominated routing
3. Both guards false ⇒ Case A → Phase A.2 at V_kin.

K+ enrichment is dominant F₀ amplifier in cathodic region:
amplification_from_c_K = 0.16 (V=+0.55) → 11.6 (V=−0.50).
Singh ΔpKa is essentially zero (amp_from_singh ≈ 1.000 to 1.000057;
pka_shift_avg ~ 1e-5).

### Background: v9 Phase A + Phase B (the original A/B at V=−0.20, no Langmuir cap)

Phase A (v9, 2026-05-08): observability check at V=−0.20V at smoke
kinetics. Result: cd is O₂-Levich-limited at all V≤+0.10; "kinetic
regime is V ≥ +0.30 V; transition near V=+0.20 V."

Phase B (v9, 2026-05-08): k_hyd ladder {1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2,
1e3} at V=−0.20V from cached Phase A snapshot.

| k_hyd | converged | Γ | error |
|---|---|---|---|
| 1e-3 | True  | 0.306 | — |
| 1e-2 | True  | 3.051 | — |
| 1e-1 | False | — | LadderExhausted at λ=0.25 |
| ≥1e+0 | False | — | LadderExhausted at λ=0.25 |

v9 Phase B's findings:
1. Γ scales linearly with k_hyd (0.306 → 3.051 = 10× for 10× k_hyd, at
   k_hyd ≤ 1e-2).
2. Γ has no Langmuir capacity in v9; converged k_hyd=1e-2 was
   ~64 monolayers of MOH (physically invalid).
3. Picard breaks at k_hyd ≥ 1e-1 because σ_S-driven ΔpKa exponent
   makes Γ_ss explode in the first positive λ rung.

v10a (landed 2026-05-10) added the Langmuir cap `(1 − Γ/Γ_max)` to the
forward branch; new closed form:

```
Γ_ss(λ) = λ·F₀ / ((1−λ) + λ·k_des + λ·B + λ·F₀/Γ_max)
```

where `F₀ = k_hyd·⟨c_M·10^(−ΔpKa)⟩` and `B = k_prot·⟨c_H⟩/δ_OHP`.
Reduces to v9 as Γ_max → ∞.  Default `Γ_max = 0.047` ≈ 1 monolayer MOH
at the OHP (`5.6e-6 mol/m² / (C_SCALE·L_REF)`).  k_des = 1.0 smoke
baseline. Both are placeholders awaiting v10b literature calibration.

### What Phase A.2 sits between

Per the locked acceptance-bundle sequence
(`docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`
§ "v10a → E sequence"):

1. Phase 0 — done.
2. v10a — Langmuir cap + integrated diagnostics. ✅ landed.
3. Minimum V-sweep diagnostic. ✅ delivered as v10a' (V_kin=−0.10).
4. V_kin selection. ✅ done.
5. **← Phase A.2 at V_kin (this plan, ~1 day).**
6. Plumbing ablation matrix at V_kin (~2 days; A1/A2/A3 manufactured-R_inj ablations BEFORE v10b).
7. CMK-3 capacitance literature note (v10b prerequisite).
8. v10b — Γ_max + k_des + C_S literature calibration (~1-2 weeks).
9. B.2 — densified k_hyd × λ ramp at V_kin (~2 days), with v10b-calibrated parameters.
10. Phase D — K-only Δ_β fit (~1 day).
11. Phase E — predictive holdout (~3-5 days).

So A.2 is the **first densified k_hyd × λ characterization at the new V_kin
with v10a's Langmuir cap, BEFORE plumbing ablation and BEFORE v10b
literature calibration**.  B.2 (step 9) is the same characterization
re-run with v10b's calibrated Γ_max + k_des + C_S.

The plan being reviewed below explicitly calls A.2 "the first densified
k_hyd × λ at V_kin (still using v10a smoke values)" and calls B.2 "a
re-run AFTER v10b has calibrated."

### Key prior decisions

1. **V_kin = −0.10 V, locked from v10a'.** The driver doesn't re-select
   V_kin; A.2 hard-codes V_kin=−0.10 (CLI overridable).

2. **Smoke kinetics held fixed at v10a baseline.** k_des=1.0, Γ_max=0.047,
   k_prot=1e-3, δ_OHP_hat=4e-6, r_H_El=200.98 (K⁺ Cu prior). v10b
   recalibrates these, not A.2.

3. **C_S = 0.20 F/m², K0_R4e_factor = 1e-14.** Inherited from v10a'.
   The driver reuses the v10a' two-stage anchor pattern (build at C_S=0.10
   convergence-pinned, runtime-bump to 0.20 via `set_stern_capacitance_model`
   + Newton resolve) because the existing solver's `c_s_ladder` raises
   NotImplementedError when combined with `kw_eff_ladder`.

4. **k_hyd grid for A.2: {1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1}** — decade-spaced,
   6 points, brackets the v9 Picard-failure point (1e-1).  v10a's Langmuir
   cap should make all 6 converge with Γ → Γ_max as k_hyd → ∞.

5. **B.2's k_hyd grid is half-decade-spaced {1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1}.**
   Different from A.2's. Why two grids? B.2 uses the *calibrated* Γ_max from
   v10b — it samples around the cap-engagement regime more finely. A.2 uses
   the wider decade-spaced grid because the v10a smoke Γ_max = 0.047 is a
   placeholder; the goal is to *find* where the cap engages, not refine
   resolution at a known cap-onset point.

6. **No perturbation column in A.2 by default.** v10a' driver had a ±C_S
   perturbation column for the V_kin selection score. A.2 doesn't need that
   (V_kin is already chosen). Default skip; CLI flag for opt-in.

7. **No plumbing ablation knobs in A.2.** No `apply_h_source` / `apply_k_sink`
   / `override_sigma_singh` flag toggling. That's step 6.

### Math: the closed-form Γ_ss

At λ=1, the steady-state Γ from the closed-form Picard solver is:

```
Γ_ss(λ=1) = F₀ / (k_des + B + F₀/Γ_max)
          = Γ_max · F₀ / (Γ_max·(k_des + B) + F₀)
          = Γ_max · F₀/(F₀ + Γ_max·k_des + Γ_max·B)
```

where:
* F₀ = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩ (boundary average)
* B  = k_prot · ⟨c_H⟩ / δ_OHP

Limits:
* Small F₀ (linear regime): Γ_ss ≈ F₀ / (k_des + B), i.e., Γ ∝ k_hyd.
* Large F₀ (saturated regime): Γ_ss ≈ Γ_max — and R_net = k_des·Γ_ss
  saturates at k_des·Γ_max.

denom_cap_to_total_ratio = `λ·F₀/Γ_max / (Γ_max·(k_des + B) + F₀ + (1−λ)·Γ_max)`
... wait, let me re-derive. The denominator decomposition the v10a code emits is:

```
denom_constant = (1 − λ)
denom_kdes     = λ · k_des
denom_kprot    = λ · k_prot · ⟨c_H⟩ / δ_OHP   = λ·B
denom_cap      = λ · F₀ / Γ_max
denom_total    = denom_constant + denom_kdes + denom_kprot + denom_cap
```

denom_cap_to_total_ratio = denom_cap / denom_total. At λ=1:

```
denom_cap/total = (F₀/Γ_max) / (k_des + B + F₀/Γ_max)
```

Limits at λ=1:
* F₀ << Γ_max·(k_des+B): denom_cap/total → 0, Γ ∝ F₀.
* F₀ >> Γ_max·(k_des+B): denom_cap/total → 1, Γ → Γ_max.

So at the cap saturation, denom_cap/total → 1 *and* Γ → Γ_max *and*
R_net = k_des·Γ → k_des·Γ_max.

At V_kin=−0.10, k_hyd=1e-3 baseline (v10a' record): denom_cap/total = 0.861.
This means F₀/Γ_max ≈ 0.861/(1−0.861) · (k_des + B). With k_des=1.0 and
B small (k_prot=1e-3, c_H ≈ 1e-8 nondim, δ_OHP=4e-6: B ≈ 1e-3·1e-8/4e-6 =
2.5e-6 — negligible). So F₀/Γ_max ≈ 0.861/0.139 · 1.0 ≈ 6.19. With
Γ_max = 0.047, F₀ ≈ 0.291 (matches v10a' record F0_avg = 0.291).

If we 10× k_hyd to 1e-2, F₀ → 2.91 (assuming c_K and 10^(−ΔpKa) don't
change much — they'll change a bit because the proton boundary source
shifts c_H, which feeds back via the Stern + Boltzmann coupling). Then:
* F₀/Γ_max ≈ 62
* denom_cap/total ≈ 62/63 ≈ 0.984
* Γ_ss ≈ Γ_max · 62/63 ≈ 0.0463
* R_net ≈ 0.0463 (saturated)

So at k_hyd ≥ 1e-2, the cap is near-saturated and R_net asymptotes at
k_des·Γ_max = 0.047. Beyond k_hyd = 1e-2, the diagnostic mostly tells us
"yes, cap saturates" — minimal additional information per rung.

### Math: the v10b prerequisite signal

If the cap saturates at the v10a smoke Γ_max but the saturated R_net
(= k_des · Γ_max = 0.047 nondim) is too small to produce the deck's
target H₂O₂ selectivity, then v10b's calibration target is to find a
larger Γ_max OR larger k_des (or both).  A.2's k_hyd ramp characterizes
the *current* saturation R_net at smoke values, which is the input to
the v10b calibration question: "what Γ_max·k_des does the system need?"

### Files relevant to A.2 implementation

* `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` — v10a' driver.
  A.2 reuses: `_build_sp`, `_build_sp_at_cs`, `_make_mesh`, `_build_kw_ladder`,
  `_walk_lambda_zero_capture_snapshots`, constants (LAMBDA_LADDER,
  K0_INITIAL_SCALES, SMOKE_KINETICS, MESH_*, EXPONENT_CLIP, U_CLAMP).
* `Forward/bv_solver/anchor_continuation.py:1533` —
  `solve_lambda_ramp_from_warm_start`. Accepts `parameter_overrides`.
* `Forward/bv_solver/anchor_continuation.py:1689` — the
  `_PARAMETER_OVERRIDE_SETTERS` dispatch table. Verified to support
  "k_hyd" key.
* `Forward/bv_solver/cation_hydrolysis.py:961` —
  `collect_v10a_rung_diagnostics(ctx)`. Emits per-rung:
  `gamma`, `gamma_max`, `theta`, `lambda_hydrolysis`, `k_hyd`, `k_prot`,
  `k_des`, `delta_ohp_hat`, `F0_avg`, `forward_avg_no_k_hyd`, `c_H_avg`,
  `pka_shift_avg`, `R_forward_capped`, denominator decomposition,
  `R_2e_current_nondim`, `R_4e_current_nondim`, `sigma_S_*`,
  `denominator_cap_to_total_ratio` (added 2026-05-10),
  `F0_decomposition` (added 2026-05-10),
  `R_4e_decomposition_log` (added 2026-05-10).

## Artifact under review

**File:** `/Users/jakeweinstein/.claude/plans/phase6b-v10a-phase-A2-v-kin.md`

```markdown
# Plan — Phase 6β v10a Phase A.2 at V_kin = −0.10 V

## Context

Phase 6β v10a' V-sweep diagnostic (2026-05-10) returned a clean
**V_kin = −0.10 V** via the primary path (no fallback):

* Config: `C_S = 0.20 F/m²` (Bohra-Koper-Choi), `K0_R4e_factor = 1e-14`
  (V=−0.10 branch-pass probe), v10a smoke kinetics (`k_hyd_nondim =
  1e-3`, `k_des_nondim = 1.0`, `Γ_max_nondim = 0.047`, `r_H_El_pm =
  200.98`, `δ_OHP_hat = 4e-6`).
* At V_kin: σ_S = −0.017 C/m², θ = 0.86, |sensS| = 0.187,
  o2_flux_levich = 0.63 (not transport-limited), denom_cap/total =
  0.86 (cap engaged, not saturated), x_2e = 0.20 (mixed branch).
* Decision tree → **Case A → Phase A.2 at V_kin**.
* See `~/.claude/plans/sparkly-gilded-pasteur.md` for the v10a' plan
  and `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` § v10a'
  Result for the per-V breakdown.

Per the locked sequence in the acceptance bundle (§ "v10a → E sequence",
step 5), Phase A.2 is the **densified k_hyd × λ ramp at V_kin** with
v10a Langmuir cap and full diagnostics.  Its purpose is twofold:

1. **Confirm v10a Langmuir cap performs as designed** at the new V_kin.
   Phase B (v9, 2026-05-08, no cap) failed Picard at `k_hyd ≥ 1e-1`
   because Γ exploded to ~64 monolayers.  v10a's `(1 − Γ/Γ_max)`
   cap should make `k_hyd ∈ {1e-4 … 1e+1}` convergent and Γ → Γ_max
   as `k_hyd → ∞`.
2. **Map the (k_hyd, λ) → (σ_S, c_H, Γ, θ, R_net, branch ratio,
   denom_cap_to_total_ratio) response surface** at V_kin.  Outputs
   feed v10b's literature calibration of `Γ_max + k_des` (those two
   determine the maximum sustainable R_net = k_des·Γ_max in the
   denom-cap-dominated regime).

Phase A.2 is **NOT** the plumbing ablation matrix (that's step 6;
`apply_h_source`, `apply_k_sink`, `override_sigma_singh` knobs are
out of scope here).  It's also **NOT** v10b literature calibration
or Phase D fitting.

## Scope (Phase A.2 only)

### What's in scope

1. **New driver** `scripts/studies/phase6b_v10a_phase_A2_v_kin.py`:
   * Two-stage anchor at V=+0.55 V (build at C_S=0.10, runtime-bump
     to C_S=0.20) — reuse the pattern from v10a' driver.
   * Warm-walk to V_kin = −0.10 V at λ=0.
   * For each `k_hyd_target` in the densified ramp: call
     `solve_lambda_ramp_from_warm_start` at V_kin with
     `parameter_overrides={"k_hyd": k_hyd_target}` and the standard
     5-point λ ladder.
   * Collect every rung's `collect_v10a_rung_diagnostics(ctx)` output
     (the v10a' decompositions: `F0_decomposition`,
     `R_4e_decomposition_log`, `denominator_cap_to_total_ratio`).
2. **k_hyd grid (decade-spaced, brackets v9 failure point):**
   `{1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1}` — 6 points.
   * 1e-3 = v10a smoke baseline (sanity-check vs. v10a' record).
   * 1e-1 / 1e0 / 1e1 = the v9 Picard-failure regime; the v10a cap
     should make these converge with Γ saturating at Γ_max.
   * 1e-4 = sub-smoke probe (does R_net trace the linear-in-F₀
     regime predicted by `R_net ≈ λ·F₀` when `F₀/Γ_max → 0`?).
3. **Held fixed at v10a smoke values:** `k_des_nondim = 1.0`,
   `k_prot_nondim = 1e-3`, `Γ_max_nondim = 0.047`, `δ_OHP_hat =
   4e-6`, `r_H_El_pm = 200.98 (K⁺ Cu prior)`, `C_S = 0.20`,
   `K0_R4e_factor = 1e-14`, `l_eff_m = 16 µm`.
4. **λ ladder:** existing `(0.0, 0.25, 0.50, 0.75, 1.0)` — 5 rungs.
5. **Output:**
   * `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.json`:
     per-(k_hyd, λ) records with full v10a' decompositions, plus
     config block + convergence audit.
   * `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.png`:
     4-panel summary at λ=1: `Γ(k_hyd)`, `θ(k_hyd)`,
     `denom_cap_to_total_ratio(k_hyd)`, `R_net(k_hyd)`.  Linear-x
     side-by-side with log-x where appropriate.
6. **Convergence audit:** print per-(k_hyd, λ) `snes_converged`
   flag and any `LadderExhausted` cause.  Pass criterion: ≥ 5/6
   k_hyd values converge through λ=1.0.  Document any non-convergence
   with diagnostics.
7. **Decision routing for downstream steps** (per the acceptance
   bundle § Status):
   * If ≥ 5/6 k_hyd converge AND `Γ → Γ_max` as `k_hyd → ∞` (cap
     engages) → proceed to **plumbing ablation matrix** (step 6).
   * If < 5/6 converge → diagnose (likely a λ ladder densification
     need; route to **B.2**'s patched AdaptiveLadder ahead of the
     ablation matrix).
   * If `Γ saturates BUT denom_cap/total < 0.8 at all k_hyd > 1e-2`
     → cap is engaged but not dominant; flag for v10b's k_des
     calibration (k_des may need to be *smaller* than smoke 1.0 to
     make the cap dominate).
8. **Tests** (`tests/test_phase6b_v10a_phase_A2_driver.py`): light
   unit tests for the k_hyd grid override threading (mock the
   solver call; assert `parameter_overrides["k_hyd"]` is the
   k_hyd_target for each grid point).

### What's out of scope (deferred)

* **Plumbing ablation matrix** (A1/A2/A3) — separate step.  No
  `apply_h_source` / `apply_k_sink` / `override_sigma_singh` flag
  toggling here.
* **K_des, Γ_max, C_S literature calibration** — that's v10b.
* **r_H_El sensitivity** — fixed at K⁺ Cu prior.
* **Multiple V_RHE** — A.2 is a single-V re-run at V_kin.
* **Multiple cations** — K⁺ only.  Cs⁺/Na⁺/Li⁺ wait for Phase E.
* **2D dense grid** beyond {k_hyd × λ_ladder}.  No k_des or k_prot
  variation here.
* **Phase D fitting** of Δ_β_K — separate step.

## Critical files to modify / create

| File | Type | Purpose |
|---|---|---|
| `scripts/studies/phase6b_v10a_phase_A2_v_kin.py` | NEW | Phase A.2 driver |
| `tests/test_phase6b_v10a_phase_A2_driver.py` | NEW | Unit tests for k_hyd override threading |
| `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` | MODIFY | Append § Phase A.2 result to the Status section |
| `CLAUDE.md` | MODIFY | Append the A.2 outcome to the Phase 6β v10a' paragraph |

No solver-side changes.  All v10a/v10a' plumbing (Langmuir cap,
two-stage anchor, k0_r4e_factor, decompositions) is unchanged.

## Implementation notes

### Driver structure (mirrors v10a' driver pattern)

[code sketch omitted for brevity — reuses v10a' driver helpers, builds sp,
runs two-stage anchor + warm-walk, then loops over k_hyd_target calling
solve_lambda_ramp_from_warm_start with parameter_overrides={"k_hyd": ...}
and the standard 5-point λ ladder.  Collects per-rung diagnostics.]

### k_hyd accessor wiring

`solve_lambda_ramp_from_warm_start` at
`Forward/bv_solver/anchor_continuation.py:1689` already supports
`"k_hyd"` in `parameter_overrides`.  No solver edit needed.

### Picard convergence at high k_hyd

v9 Phase B's k_hyd ≥ 1e-1 Picard-failure was specifically because
Γ had no cap.  v10a's `(1 − Γ/Γ_max)` factor caps Γ at Γ_max and the
closed-form Γ_ss(λ) handles arbitrary k_hyd analytically — no
Picard iteration needed within a rung.  Therefore:

* k_hyd = 1e-1 / 1e0 / 1e1 should converge with Γ → Γ_max = 0.047.
* If they don't, the failure is in the **λ ladder** (Newton at λ=0.25
  with high F₀), not Picard.  Diagnose with the per-rung
  `lambda_history` field.
* If λ ladder fails, B.2's `AdaptiveLadder` patch (`λ=0` floor +
  finer rungs) is the proper fix; document and route to B.2 ahead
  of schedule.

### Expected qualitative behavior at V_kin = −0.10 V

Predicted from v10a' record (k_hyd = 1e-3 baseline at V_kin):

| k_hyd | Γ_predicted | θ_predicted | denom_cap/T_predicted | R_net_predicted |
|---|---|---|---|---|
| 1e-4 | ~5e-3 | ~0.10 | ~0.30 | ~5e-3 |
| 1e-3 | 0.0405 (observed) | 0.86 (obs) | 0.86 (obs) | 0.0405 (obs) |
| 1e-2 | ~0.046 | ~0.98 | ~0.98 | ~0.046 |
| 1e-1 | ~0.047 | ~1.00 | ~0.997 | k_des·Γ_max = 0.047 |
| 1e0  | ~0.047 | ~1.00 | ~0.9997 | 0.047 |
| 1e1  | ~0.047 | ~1.00 | ~0.99997 | 0.047 |

(These are rough closed-form predictions assuming F₀ scales linearly with
k_hyd at fixed c_K; actual values will deviate due to coupling.)

### Per-(k_hyd, λ) record schema

[JSON schema — per-(k_hyd, λ) records with full v10a' decompositions]

### Plot specification

4-panel matplotlib at λ=1: Γ, θ, denom_cap/total, R_net all vs k_hyd
log-x; reference lines at Γ_max, θ=1, 0.8, k_des·Γ_max.

## Verification

### Fast tests
* `_parse_args(["--k-hyd-grid", "1e-4,1e-3,1e-2"])` parses correctly.
* Default = K_HYD_GRID_DEFAULT (6 points).
* CLI accepts `--v-kin -0.10` and `--k0-r4e-factor 1e-14` overrides.
* `_build_per_k_hyd_record` packages a converged ladder result correctly.

### Re-run
```bash
python -u scripts/studies/phase6b_v10a_phase_A2_v_kin.py \
    --v-kin -0.10 --k0-r4e-factor 1e-14 \
    --k-hyd-grid 1e-4,1e-3,1e-2,1e-1,1e0,1e1 \
    --out-subdir phase6b_v10a_phase_A2_v_kin
```

### Sanity checks on the run

1. **k_hyd = 1e-3 sanity (vs. v10a' record at V_kin = −0.10):** the
   λ=1.0 rung at k_hyd=1e-3 must reproduce v10a' record within rel 1e-3:
   `gamma=0.0405, theta=0.861, sigma_S=−0.017, cd_mA_cm2=−3.12`.
   Failure → driver wiring bug.
2. **Cap saturation:** Γ monotonically increases with k_hyd; asymptotes
   at Γ_max=0.047 as k_hyd → ∞.
3. **Mass balance at λ=1 SS:** `R_net = k_des · Γ` to numerical precision.
4. **σ_S response:** as k_hyd grows, R_net grows, c_H rises at OHP, and
   σ_S responds via the Stern + Boltzmann coupling. Sign check: σ_S
   should become more cathodic (more negative) as R_net grows. If σ_S
   grows more positive, flag as a sign bug.
5. **Branch ratio drift:** x_2e at V_kin baseline is 0.20. As k_hyd
   grows, c_H rises, R_4e grows faster (n_e=4 vs 2), so x_2e should
   decrease. If x_2e grows with k_hyd, flag as a kinetic-coupling anomaly.

### v10b prerequisite signal threshold

If at the highest converged k_hyd, all three of:
* `denom_cap/total > 0.8`
* `θ > 0.9`
* `|sensS| < 0.10` (vs. v10a' baseline 0.187 — OPTIONAL since perturbation
  is skipped by default)

are satisfied → v10b literature calibration is mandatory before B.2.
If none satisfied at any k_hyd → cap doesn't dominate even at k_hyd=1e+1;
Γ_max=0.047 is too generous; v10b should re-derive Γ_max from literature
MOH adsorption (likely smaller).

### Optional: Sensitivity column

Phase A.2 does NOT need the ±C_S perturbation column. v10a' record already
gave |sensS| at V_kin for k_hyd=1e-3. Default skip; CLI flag `--with-perturbation`
for opt-in (wall ~3× longer).

### Convergence-failure escalation tree

* k_hyd ≤ 1e-2 fails: unexpected (smoke regime). Stop; debug.
* k_hyd ∈ {1e-1, 1e0, 1e1} fails at λ ≥ 0.25 with Γ → Γ_max but Newton
  diverging: cap engaged but Newton stiff. Mitigate with finer λ inserts;
  if exhausted, escalate to B.2's patched AdaptiveLadder.
* k_hyd = 1e-4 fails: F₀ underflow at λ=0.25. Drop k_hyd=1e-4 from the
  grid; document.
* All λ > 0 fails at all k_hyd > 1e-2: Γ_max too small — cap clamps Γ
  before λ ramps. Document and escalate to v10b ahead of plumbing ablation.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | k_hyd = 1e+1 Newton stiffness causes LadderExhausted | Closed-form Γ_ss handles arbitrary k_hyd; only FE Newton at λ=0.25 with high F₀ at risk. If fails, route to B.2 patched ladder. Document. |
| 2 | k_hyd = 1e-4 F₀ underflow → ill-conditioned residual | Drop the rung; document. Sub-smoke regime not on critical path. |
| 3 | Cap saturates BEFORE the highest k_hyd in grid → diminishing returns from k_hyd > 1e-1 rungs | Acceptable — those rungs confirm saturation plateau. Not a failure. |
| 4 | k_hyd accessor `parameter_overrides["k_hyd"]` misnamed | Verified; unit test #4 guards. |
| 5 | σ_S sign convention mismatch | v10a' record at V_kin = −0.10 already shows correct sign (−0.017). Sanity check #4 explicitly checks. |
| 6 | `_run_lambda_ramp_collect_all_rungs` may double-count rung 0 | Rung 0 is same state as warm-start (mass-balance verification at λ=0); document, don't dedupe. |
| 7 | r_H_El = 200.98 may give unphysical Singh ΔpKa at high k_hyd | F0_decomposition `amplification_from_singh` is the load-bearing diagnostic; if grows above ~10, treat as unphysical and flag for v10b r_H_El recalibration. |
| 8 | Wall budget overrun | Hard-cap each lambda ramp at 5 min; if hit, mark failure and continue. |

## After Phase A.2 lands

Per the locked sequence: step 6 (plumbing ablation), step 7 (CMK-3 cap note),
step 8 (v10b calibration), step 9 (B.2), step 10 (Phase D), step 11 (Phase E).

## References

* `~/.claude/plans/sparkly-gilded-pasteur.md` — v10a' V-sweep plan
  (immediately preceding step).
* `StudyResults/phase6b_v10a_prime_k0r4e_1e-14/iv_diagnostic.json` —
  v10a' record establishing V_kin = −0.10.
* `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` — locked
  acceptance bundle.
* `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` — original
  v9 Phase A/B at V=−0.20 V (no cap; failed Picard at k_hyd ≥ 1e-1).
* `Forward/bv_solver/cation_hydrolysis.py:961` — `collect_v10a_rung_diagnostics`.
* `Forward/bv_solver/anchor_continuation.py:1533` — `solve_lambda_ramp_from_warm_start`.
* `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` — v10a' driver
  (helper functions A.2 reuses).
* Memory: `project_v10a_prime_outcome`, `project_v10a_prime_two_stage_anchor`,
  `project_k0_r4e_ratio_regimes`.
```

## Critique prompt

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

Specific focus areas (per user request):
1. Is k_hyd × λ at v10a smoke values actually answering the question that
   v10b will need answered? (Or is A.2 redundant with B.2?)
2. Is the 6-point k_hyd grid {1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1} the right
   bracket vs B.2's half-decade {1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1}?
   What's the right grid for "find where the cap engages"?
3. The "denom_cap dominates" routing rule (denom_cap/T > 0.8 AND θ > 0.9
   AND |sensS| < 0.10) — is this rule actually meaningful at A.2 (where
   |sensS| isn't computed by default), and what should it route to?
4. Convergence-failure escalation tree — does it cover the realistic
   failure modes? Anything missing?
5. Does the plan correctly bridge from v10a' to v10b? (i.e., does the
   output give v10b's calibration step the inputs it needs?)
6. Sanity checks #1-5 — are they sufficient? Anything missing?
7. Hidden coupling effects: at high k_hyd, c_H rises, which feeds back into
   F₀ via the `c_M · 10^(−ΔpKa)` factor (Singh ΔpKa increases with σ_S
   which depends on c_H), AND into B = k_prot·c_H/δ_OHP. Is the closed-form
   Γ_ss actually closed-form in F₀, or does it implicitly require Picard
   over c_H ↔ Γ ↔ σ_S ↔ ΔpKa coupling? Does the plan acknowledge this?
8. Inheritance from session 33's 22 issues — does A.2 inherit all the
   v10a'-hardened guardrails (Stern bump, K+ enrichment expectation,
   Jensen-safe averaging, Stern-aware η_raw)? Anything missing?
