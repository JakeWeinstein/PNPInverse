# Ruggiero Realignment — Forward-Solver Plan

Date: 2026-05-07
Status: Drafted while M3a.0 observable audit is in flight. This plan
covers what comes after the audit, regardless of its outcome.

## Why this exists

We hit a structural shift on 2026-05-07. The Mangan deck page-15 figure
that Plan B targeted is built on Ruggiero 2022 (J. Catal., Mangan
co-author). The Ruggiero paper's reaction model is **parallel 2e and 4e
ORR**, not the sequential R_0 → R_1 our solver implements:

```text
2e (current R_0 in our model):   O₂ + 2H⁺ + 2e⁻ → H₂O₂   E⁰ = 0.695 V_RHE
4e (NOT currently in our model): O₂ + 4H⁺ + 4e⁻ → 2H₂O   E⁰ = 1.23  V_RHE
```

Our R_1 (H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O at the disk surface) is **not** the 4e
pathway — that one goes through *-OOH → *-OH → H₂O via adsorbed
intermediates, with no free H₂O₂. The R_0/R_1 lock-in seen in Run D is a
model-structural artifact, not a physical effect.

This plan brings the forward solver into the Ruggiero formulation.
It is the consolidation of:

- `docs/Ruggiero2022_JCatal_source_paper.md` — the experimental + chemistry facts.
- `docs/CHATGPT_HANDOFF_17_RUGGIERO_REALIGNMENT_PLAN.md` — initial realignment plan.
- `docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md` — the pushback that splits M3a into substages and corrects mis-scoping. **H18 is the live structure** for M3a.0 → M3a.3.
- `docs/mangan_alignment_status_2026-05-07.md` — current status doc (the "read me first" pickup doc).

## Goal

Forward solver capable of producing a defensible quantitative comparison
against Ruggiero/Mangan deck p.15 (peroxide current density vs V_RHE,
pH 4, Cs⁺, CMK-3, RRDE) without reaction-set or observable-definition
artifacts.

## Non-goals (deferred)

- Inverse work (paused; see CLAUDE.md).
- Cation-specific kinetic modulation (would require chemistry
  beyond PNP+Bikerman).
- Re-deriving Mangan deck modeling parameters (k0, α, L_eff): the
  Ruggiero manuscript does not contain a BV parameter table. Pages 14,
  16-18 of the deck do; extraction is a separate open task.

## Source authority hierarchy (per H18)

1. **Ruggiero 2022 J. Catal.** — experimental protocol, electrolyte,
   RRDE constants, local-pH/cation-effect mechanism, qualitative ORR
   pathway chemistry.
2. **Mangan 2025 deck** — computational modeling intent, parameter
   sweeps, effective length/radius/kinetic narrative.
3. **Our solver** — implementation constraints and observable definitions.

Do not ask the Ruggiero paper to provide BV parameters it does not
contain (k0, α, L_eff). Use the deck for those once extracted; until
then, treat them as priors.

## Locked decisions

These are nailed down by Ruggiero data + H18 critique + CLAUDE.md hard
rules. Do not relitigate them in implementation:

1. **Replace sequential R_0 + R_1 with parallel R_2e + R_4e.** Free H₂O₂
   never forms in the 4e channel.
2. **The deck's "peroxide current density" maps to gross 2e production
   current (single R_2e rate × 2F)**, not the existing
   `peroxide_current = R_0 − R_1` observable.
3. **Total disk current must be electron-weighted**:
   `j_disk ∝ Σ_j n_e_j · R_j`. The legacy `current_density` observable
   mode is wrong by construction once R_4e is added (it sums R_j
   uniformly assuming n_e=2 across).
4. **Optional H₂O₂ sink is allowed but off by default.** Do not delete
   the structural ability to model peroxide consumption — Ruggiero
   discusses pH-dependent decomposition mechanisms. Just unbind it from
   the 4e pathway.
5. **Electrolyte: 0.1 M H₂SO₄ + 0.1 M MOH + 0.1 M M₂SO₄, I = 0.3 M, λ_D
   ≈ 0.55 nm.** Sulfate, not perchlorate. ClO₄⁻ surrogate is wrong by
   design for this comparison.
6. **CP validation currents are area-normalized**:
   `-0.10, -1.02, -2.04, -3.06, -3.32 mA/cm²` on a 0.196 cm² disk.
   The famous 4 → 8-9 local-pH excursion is at ~3.25 mA/cm².
7. **OH⁻ stays K_w-coupled to H⁺ unless an explicit alkaline rate law
   or hydrolysis chemistry is added.** Tracking OH⁻ as an independent
   species without acid/base closure adds a species but not the missing
   physics (per H18 V2).
8. **k0_4e is weakly identified from page 15 alone.** Calibration
   targets must include disk current / selectivity / electron number,
   not just the peroxide left-plateau magnitude (per H18 V4).
9. **PNP+Bikerman captures only part of the cation-buffering story**
   (OHP localization). Hydrolysis/activity chemistry may be required
   for cation ordering — failure to reproduce the cation series in M4
   is **not** automatically a multi-ion bug (per H18 V3).

## Bug to fix in passing — RESOLVED 2026-05-07 (M3a.2.1)

`scripts/_bv_common.py:79` previously held `C_O2 = 0.5 mol/m³`.
Ruggiero §2.4 gives `C_O2 = 1.2 mol/m³` at pH 5-13 (1.1 at pH 2).
The legacy value underestimated the dimensional Levich limit by ~2.4×.
Migrated to `C_O2 = 1.2 mol/m³` on 2026-05-07; fast pytest suite still
green (455/455 net of 6 pre-existing inverse-stack failures).
Snapshot tests `test_stern_no_stern_snapshot.py` and
`test_steric_boltzmann_closure.py::test_ideal_path_byte_identical`
marked `xfail(strict=True)` until their baselines are regenerated
(slow suite, requires Firedrake). See "Implementation status" §M3a.2.1
below for the resolution log.

## Milestone plan

### M3a.0 — Observable audit (DONE 2026-05-07)

Audit complete. Full report at
`StudyResults/mangan_p15_comparison/m3a0_audit.md`. No solver re-run
needed; pure algebraic post-processing of Run C.

**Verdict:** `m3a0_verdict = "magnitude_off_by_lock_in_shape_off_by_local_pH"`.

This is the second-branch ("magnitude off / shape off") outcome the
status doc anticipated, with two structural refinements:

- **Magnitude (gross R_0 short by 1.86× at V=−0.32 V, −0.092 vs
  experimental −0.17 mA/cm²)** is a mechanical artifact of the
  R_0/R_1 saturation lock-in. Replacing R_1 with parallel R_4e (M3a.2)
  unlocks gross R_2e to absorb the full O₂ flux at saturation. The
  total cd left-plateau "PASS" (−0.183 vs −0.17, 7.6%) is **not** a
  current match — it's the projected post-fix gross R_2e saturation
  envelope. Audit's prediction for M3a.2 is concrete and testable:
  gross R_2e plateau lifts from −0.092 → near −I_SCALE.
- **Shape (no peak in middle of V range)** is genuine missing physics:
  local-pH dynamics + cation buffering + multi-ion EDL screening. M3b
  + M3c work, not a forward-solver bug.

**Encouraging signal for M3c:** Audit notes that even under the ClO₄⁻
surrogate, the existing solver already produces a 4–6 pH-unit alkaline
excursion (`surface_pH_proxy` ~9.77 at V=−0.40 V, ~8.36 at V=+0.30 V),
in the ballpark of Ruggiero Fig 1B (~4 pH units at 3.25 mA/cm²). H⁺
NP transport alone gets the right qualitative behavior — the multi-ion
work is then about *modulating* the excursion via cation buffering,
not building it from scratch.

**Important interaction with M3a.2's C_O2 fix:** Audit also flagged
that experimental peak |j| = 0.40 mA/cm² is 2.18× our current 2e Levich
ceiling (I_SCALE = 0.183 mA/cm²) and concluded "M5 L_eff retune is
larger than ~10%, the ceiling itself needs to roughly double." This
finding is **implicitly conditional on `C_O2 = 0.5 mol/m³`** — the
buggy current value. The C_O2 = 0.5 → 1.2 mol/m³ fix already in M3a.2
lifts I_SCALE by 2.4× → new ceiling 0.44 mA/cm², ~10% inside
experimental peak. So the M5 retune scope stays small (~10% L_eff),
provided M3a.2 includes the C_O2 fix. See M3a.2 verification + M5
scope clarification below.

**Bands passing.** 0/5 quantitative + qualitative on `pc_net`; 0/5 on
`gross R_0`; 1/5 on total `cd` (left plateau only, where the
saturation envelope coincides with experimental peroxide at the
plateau by accident — see audit §"layered diagnostic
decomposition (2)").

### M3a.1 — Electron-weighted current observables

**Cost: small (~0.5–1 day). Fully isolated from physics.**

**Goal.** Total disk current and selectivity respect per-reaction
`n_electrons`. Pure-2e config gives 100% H₂O₂ selectivity and
`n_e_apparent = 2`; pure-4e config gives 0% and `n_e_apparent = 4`.

**Code surface.**

- `Forward/bv_solver/observables.py:13-68` (`_build_bv_observable_form`):
  - `current_density` mode: change `rate_sum = Σ R_j` to
    `rate_sum = Σ (n_e_j / N_ELECTRONS_REF) · R_j` where
    `N_ELECTRONS_REF = 2` (the existing `I_SCALE` reference). Pull
    `n_e_j` from `ctx["bv_settings"]["reactions"][j]["n_electrons"]`.
    For pure-2e configs this reduces to the legacy form (back-compat).
  - `peroxide_current` mode (R_0 − R_1): mark deprecated. Add a new
    mode `gross_h2o2_current` that maps to `mode="reaction"`,
    `reaction_index=R_2e_idx` (assemble single rate × 2F). Keep the
    legacy net-difference mode reachable for the existing v13/v15/v16
    scripts but flag it in the docstring with the H18 finding.
  - Add a new mode `disk_current_n_e_weighted` if not strictly
    redundant with the corrected `current_density` (likely is — pick
    one).
- `scripts/_bv_common.py:131-144` (`compute_i_scale`): document that
  the returned scale is anchored to `n_e=2` and that observable code
  multiplies by `(n_e_j / N_ELECTRONS_REF)`. No value change.
- Wire RRDE selectivity / `n_e_apparent` (already in
  `Forward/bv_solver/rrde_observables.py` from M1) to use the new
  electron-weighted disk current.

**Verification.**

- Pure-2e regression: keep two reactions, both with `n_electrons=2`
  (current production stack). Re-assemble `current_density` over Run C
  state. Result must equal the legacy assembly to floating-point
  tolerance.
- Pure-4e synthetic: build a one-reaction config with `n_electrons=4`,
  verify `current_density` = 2 × `Σ R_j` and selectivity computes 0%
  H₂O₂.
- Mixed synthetic: two reactions with n_e ∈ {2, 4}, verify
  proportionality.

**Tests.** Add unit tests in `tests/Forward/bv_solver/test_observables.py`
for the three cases above. No Firedrake solve required — synthesize
`bv_rate_exprs` directly from constants.

**Dependencies.** None — runs in parallel with M3a.0 if the audit drags.

**Risks.** Low. The change is local to the observable layer. The only
breakage is for legacy callers of `peroxide_current` mode; those will
see a deprecation note and continue working.

### M3a.2 — Diagnostic parallel R_2e/R_4e residual

**Cost: medium (~3–5 days). The `_bv_common.py` factory generalization
is the real work; the residual side is already generic.**

**Goal.** End-to-end converged run on a small voltage subset with
parallel 2e/4e reactions. Diagnostic: not yet production-comparable
because the IC is conservative.

**Code surface — config plumbing.**

- `scripts/_bv_common.py:204-238` (`THREE_SPECIES_LOGC_BOLTZMANN`):
  - Add a parallel preset, e.g. `THREE_SPECIES_LOGC_BOLTZMANN_PARALLEL`,
    that provides a `stoichiometry_list: List[List[int]]` instead of
    the hardcoded `stoichiometry_r1 / stoichiometry_r2` pair:
    - R_2e: `[-1, +1, -2]` (O₂ consumed, H₂O₂ produced, 2H⁺ consumed).
    - R_4e: `[-1, 0, -4]` (O₂ consumed, H₂O₂ untouched, 4H⁺ consumed).
  - Keep the existing sequential preset untouched for v13/v15/v16
    backward compat.
- `scripts/_bv_common.py:73-93` — add Ruggiero-aligned constants:
  - `E_EQ_R2E_V = 0.695` (Ruggiero §1; refines the legacy 0.68).
  - `E_EQ_R4E_V = 1.23` (Ruggiero §1).
  - `K0_PHYS_R2E = K0_PHYS_R1` (legacy 2e rate; alias for clarity).
  - `K0_PHYS_R4E` — **prior-selected**, no Ruggiero arbitrator.
    Default to `K0_PHYS_R1` and document that this is a placeholder
    pending M4 calibration against disk current / selectivity.
  - `ALPHA_R2E = ALPHA_R1 = 0.627`.
  - `ALPHA_R4E = 0.5` (default placeholder; revisit in M4 with Tafel).
- `scripts/_bv_common.py:79` — fix `C_O2 = 0.5 → 1.2 mol/m³`. **Ripple
  audit required**: every nondim ratio derived from `C_SCALE = C_O2`
  changes. Validate that I_SCALE moves correctly (it should — the
  `n·F·D·c/L` formula recovers the right physical scale once `C_SCALE`
  is correct). Re-run a single-V smoke vs Run C to confirm no
  regressions. Tag the migration with a memory note.
- `scripts/_bv_common.py:311-406` (`_make_bv_bc_cfg`):
  - Refactor to accept either the old `k0_hat_r1, k0_hat_r2, …`
    keyword set OR a generic
    `reactions: Optional[List[Dict[str, Any]]] = None` list. When the
    generic list is supplied, build `cfg["reactions"]` directly from it.
  - Auto-attach H⁺ stoichiometric concentration factors using each
    reaction's `n_electrons` to derive the H⁺ power
    (`power = n_e_j` since the proton stoichiometry follows the
    electron count for the acid-form ORR).
- `scripts/_bv_common.py:436-565` (`make_bv_solver_params`): add a
  `bv_reactions: Optional[List[Dict]] = None` parameter that, when set,
  takes precedence over the legacy R_1/R_2 keyword set.

**Code surface — IC handling (conservative path).**

- For M3a.2, **do NOT generalize the Picard.** The
  `_try_debye_boltzmann_ic_*` helpers in `forms_logc.py:623-811` and
  `forms_logc_muh.py:704-908` still index `bv_reactions[0/1]` and run
  the 2x2 sequential-rate algebra. Calling them on a parallel config
  produces a physically inconsistent seed.
- Two acceptable diagnostic paths:
  1. **Linear-phi fallback (preferred for M3a.2).** Pass
     `initializer="linear_phi"` so `set_initial_conditions_logc_muh`
     runs without invoking the Picard. This is the IC the C+D
     orchestrator already uses on warm-walks. Convergence rate will
     drop vs the matched-asymptotic IC.
  2. **Gate the Picard.** Add a check in `_try_debye_boltzmann_ic_muh`:
     if the reactions don't match the sequential template (R_0 produces
     P, R_1 consumes P), return
     `(False, "non_sequential_topology", 0)` so the orchestrator falls
     back automatically.
- Either way, tag the run output's `experiment_metadata` with
  `comparison_status="diagnostic_only"` (new sentinel) so downstream
  scripts know the IC was conservative.

**Driver script.** New
`scripts/studies/peroxide_window_3sp_parallel_2e_4e.py`. Modeled on
`peroxide_window_3sp_bikerman_muh.py` but:
- Uses the parallel preset.
- Voltage subset: 8-10 points spanning the page-15 window
  (V_RHE ∈ [−0.40, +0.55] V); not the full 25-point sweep.
- Tag run as diagnostic.

**Verification.**

- **Pure-2e limit.** Set `k0_R4e = 0`. Re-run. Result must reproduce
  `cd_mA_cm2` and gross R_2e from the existing 3sp+Bikerman+muh stack
  (Run C reference) to within ~10⁻⁶ relative.
- **Pure-4e limit.** Set `k0_R2e = 0`. Verify n_e_apparent = 4, peroxide
  current = 0, total disk current bounded by the 4e Levich limit
  (~5.7 mA/cm² at corrected C_O2 = 1.2 mol/m³).
- **Mixed config smoke.** Both rates active. Verify:
  - Gross H₂O₂ current is finite, has the correct sign, lies below
    the 2e Levich ceiling (~0.44 mA/cm² post-C_O2-fix at L_REF=100 µm,
    or ~2.85 mA/cm² at the bare-Levich δ if L_REF is also retuned —
    use the L_REF=100 µm budget for the diagnostic).
  - Total disk current is bounded by the 4e Levich ceiling
    (~0.88 mA/cm² post-C_O2-fix at L_REF=100 µm).
  - No NaNs or unbounded residuals.
  - Convergence: ≥ 6/10 V points converged. (The legacy 4sp dynamic
    ceiling was 5/15 / 7/15 — set ≥6/10 as a low bar consistent with
    the linear-phi IC being weaker than the matched-asymptotic Picard.)
- **C_O2-fix sanity check (M3a.0 prediction test).** With the fix
  applied (C_O2 = 0.5 → 1.2 mol/m³) and parallel topology (k0_R4e
  prior-selected), confirm gross R_2e at V_RHE = −0.32 V lifts from
  Run C's −0.092 mA/cm² toward saturation envelope. Audit predicts
  gross R_2e plateau ≈ −0.183 mA/cm² with C_O2 unchanged or ≈
  −0.44 mA/cm² with C_O2 fix; experimental left plateau is
  −0.17 mA/cm². If gross R_2e overshoots experimental by >2× (i.e.
  closer to 0.44 than 0.17), the parallel preset's k0_R4e prior is
  too small — selectivity is too high. Sweep `K0_PHYS_R4E ∈ {1, 5,
  10} × K0_PHYS_R1` to bracket. **This is the cheapest diagnostic of
  whether parallel topology + C_O2 fix lands the magnitude in the
  right OoM.**
- **Shape diagnostic.** Plot `gross_R_2e` vs page-15 target. **Loose
  acceptance**: peak voltage within ±100 mV, peak magnitude within
  factor of 2-3 (this is a diagnostic, not the production comparison).
  Peak structure is not expected to land here — that is M3b/M3c work.

**Acceptance gating.** Per H18: "peroxide nonzero" is too weak.
Quantitative checks above + pure-channel limiting tests.

**Dependencies.** M3a.1 must land first (selectivity acceptance
criteria depend on electron-weighted accounting).

**Risks.**

- **Convergence rate.** Without the matched-asymptotic Picard, more
  V points may cold-fail. Mitigation: rely on C+D warm-walking from
  one or two stable anchors. If <6/10 converges, escalate to M3a.3
  early.
- **k0_4e prior.** Defaulting to `K0_PHYS_R4E = K0_PHYS_R1` is a guess.
  If gross H₂O₂ current is way off magnitude, sweep
  `K0_PHYS_R4E ∈ {1e-1, 1, 10} × K0_PHYS_R1` to bracket. Document the
  sweep result in the study output.
- **C_O2 fix ripple.** Could perturb other anchors. Run a regression
  smoke on the existing 3sp+Bikerman+muh stack (Run C reference) at
  one V (e.g. V_RHE = 0.0 V) before going wide.

### M3a.3 — Production parallel-reaction IC

**Cost: large (~1–2 weeks). Load-bearing solver work.**

**Status (2026-05-07): NOW CRITICAL PATH.** The M3a.2 diagnostic
path (parallel residual + `linear_phi` IC) was validated empirically
and **does not converge**. Even the pure-2e limit (k0_R4e × 1e-30) at
V=0.0 with clip=50 cold-fails at z=0. The linear-phi IC cannot quench
c_O2 at the cathode against the `exp(α·n_e·η)` BV factor — that UFL
expression is built unconditionally for every reaction including
n_e=4 R_4e, regardless of k0. M3a.2's "diagnostic with conservative
IC" path is structurally unworkable; M3a.3 (this milestone) must land
before any parallel-topology run produces output.

**Empirical finding from M3a.2 attempt (2026-05-07):**

- Legacy sequential, debye_boltzmann IC, clip=50, V=0.0 → cold-converges
  ✓ (reproduces Run C reference cd = -0.175 mA/cm²).
- Legacy sequential, debye_boltzmann IC, clip=100, V=0.0 → cold-fails
  ✗ (matches CLAUDE.md hard rule 2 "no-Stern bikerman near +0.1 V"
  pattern; warm-walk from a stable anchor would recover it).
- Parallel R_2e/R_4e (any k0_R4e), debye_boltzmann IC → topology gate
  triggers correctly, fallback to linear_phi → linear_phi cold-fails
  at every V tested (+0.0, +0.45, -0.32, -0.40).

**Goal.** Generalize the matched-asymptotic Picard initializer to
arbitrary parallel reaction topology so the parallel 2e/4e stack
recovers the legacy 25/25 V_RHE convergence.

**Code surface — Picard generalization.**

- `Forward/bv_solver/picard_ic.py:294-440` (`picard_outer_loop`):
  Current signature takes scalars `k1, k2, a1, a2, n_e, E1, E2,
  h_factor1, h_factor2`. The body assumes:
  - `O_s = O_b - (R1 + R2)/D_O · scale` — both rates consume O₂ ✓ for
    parallel; not load-bearing for restructure.
  - `P_s = P_b + (R1 - R2)/D_P · scale` — assumes R_2 consumes the H₂O₂
    R_1 produces. **For parallel 2e/4e, P_s = P_b + R_2e/D_P · scale**
    (R_4e doesn't touch H₂O₂).
  - `H_o = H_b - (R1 + R2)/D_H · scale_factor_H` — assumes uniform 2-H⁺
    consumption per turnover. **For parallel** (corrected, see
    `docs/picard_general_topology_derivation.md:78-128` v3 §2): the
    outer-region proton flux balance is the **signed ambipolar** form
    ```
    H_o = H_b + Σ_j s_{H,j} · R_j / (2 D_H)
    ```
    where `s_{H,j}` is the **signed** stoichiometric coefficient (negative
    for proton consumption). Sequential 2×2 reduces to
    `H_b − (R_1 + R_2)/D_H` (matches `picard_ic.py:516`); parallel 2e/4e
    reduces to `H_b − (R_2e + 2·R_4e)/D_H`. The previous version of this
    note read `H_b − (2·R_2e + 4·R_4e)/D_H`, which dropped the
    ambipolar `1/2` factor and was wrong by `2×` whenever `s_H` is
    non-uniform across reactions; the absolute-value form
    `Σ_j |s_{H,j}|` also silently inverts the sign for any future
    proton-producing reaction. The signed ambipolar form above is the
    implementation contract.
- Refactor signature: replace scalar `k1, k2, …` with a `reactions: list
  of dicts` (`k0`, `alpha`, `n_e`, `E_eq`, `cathodic_conc_factors`,
  `stoichiometry`). Internal Picard state becomes `R_j: list[float]`.
- The 2×2 algebraic structure reduces to a generic flux-balance
  iteration (see `docs/picard_general_topology_derivation.md` v3 §2/§4
  for the full derivation; signed forms throughout, ambipolar `1/(2 D_H)`
  for protons):
  ```
  c_{i,s} = c_{i,b} + (Σ_j s_{i,j} · R_j) / D_i · L_eff      (i ∈ {O, P, …})
  P_s     = max(P_b + (Σ_j s_{P,j} · R_j) / D_P · L_eff, P_FLOOR)
  H_o     = max(H_b + (Σ_j s_{H,j} · R_j) / (2 D_H) · L_eff, 1e-300)
  ```
  with `R_j = R_j(O_s, P_s, H_o, η_j, γ_s)` from the BV expression for
  reaction j. All sums are over **signed** stoichiometries, not
  absolute values. The `1 / (2 D_H)` ambipolar correction applies
  only to H⁺ (matched-asymptotic outer transport law); ordinary
  species use `1 / D_i`.

**Code surface — adapter sites.**

- `Forward/bv_solver/forms_logc_muh.py:704-908` (`_try_debye_boltzmann_ic_muh`):
  - Drop the `len(bv_reactions) < 2` requirement (or generalize to
    `len(bv_reactions) >= 1`).
  - Pass the full reactions list (not just `rxn1, rxn2` scalars) to
    `picard_outer_loop`.
  - Drop the `mu_h_idx != 2` rejection if H⁺ stays at index 2 in the
    parallel preset (likely yes; no species reordering needed).
- `Forward/bv_solver/forms_logc.py:623-811` — same generalization for
  the logc backend.
- `Forward/bv_solver/boltzmann.py:90-` — verify the multispecies γ
  algebra still works with the generalized H⁺ outer state. The
  Bikerman formula
  `γ_s = 1/(1 + a_h·H_o·(e^-ψ_D − 1) + a_cl·c_anchor·(e^+ψ_D − 1))`
  doesn't reference reaction count, so no change expected.

**Verification.**

- **Pure-2e regression.** Parallel preset with `k0_R4e = 0`.
  Convergence on the full 25-V grid must match Run C (25/25 cold-or-
  warm via C+D).
- **IC residual diagnostic.** After Picard converges, compute
  `R_j(picard_state)` and compare to `R_j(SNES_solution)` post-solve.
  Match should be within Picard `tol=1e-6`.
- **Single-V cold anchor.** At a hard V (e.g. V_RHE = -0.40 V or
  +0.55 V), confirm IC alone (no SNES) gives a consistent state.
- **Full 25-V grid via C+D.** Goal: ≥ 22/25 converged. Drop in
  convergence vs Run C is acceptable if the new physics is in.
- **Comparison plot.** `gross_R_2e` vs page-15 target. Acceptance
  bands per Plan B B7 (peak ±50 mV, peak magnitude ±25%, etc.). This
  is the first "real" production comparison.

**Dependencies.** M3a.2 must succeed (config plumbing must be in
place; diagnostic results must justify the larger effort).

**Risks.**

- **Picard convergence at parallel topology.** The current 2×2 algebra
  has known stability properties (Codex review handoffs 12-13). The
  generalized iteration is an N×N flux balance — could oscillate if
  reactions span very different rates. Mitigations:
  - Keep Picard relaxation `omega = 0.5` (or lower).
  - Cap `max_iters = 50` (current default).
  - Fall back to linear-phi IC on Picard non-convergence (already
    wired).
- **Anchor relocation.** With corrected `E_EQ_R2E_V = 0.695` (was
  0.68), the cathodic onset shifts ~15 mV. Existing anchor V points
  may need rediscovery.
- **Tape hygiene.** Already wrapped in `adj.stop_annotating()`; the
  refactor must preserve this.

### M3b — Multi-ion electrolyte (split per H18)

**Cost: very large (multi-week). M3b.3 is load-bearing.**

**Goal.** Replace the protonic ClO₄⁻ surrogate with the deck-correct
Cs⁺/H⁺/SO₄²⁻ electrolyte at I = 0.3 M, λ_D ≈ 0.55 nm.

**Substages (sequential):**

- **M3b.1 — Ideal multi-ion electroneutrality.** Drop the single-
  counterion guard in `Forward/bv_solver/boltzmann.py`. Generalize the
  ideal Boltzmann residual loop to accept ≥1 entries with arbitrary
  charges. Add a multi-ion bulk electroneutrality check
  `Σ z_i c_b_i = 0` at config time. Test: collapse to single-ion when
  others → 0.
- **M3b.2 — Steric (Bikerman) multi-ion closure.** Generalize the
  multispecies γ algebra in `boltzmann.py:build_steric_boltzmann_expressions`
  and `picard_ic.py:compute_surface_gamma` to
  `γ_s = 1/(1 + Σ_k a_k · c_b_k · (e^(-z_k·ψ) − 1))` over all analytic
  ions. Pack-fraction check `Σ a_i c_i ≤ 1` across the domain.
- **M3b.3 — Asymmetric multi-ion IC.** Re-derive the composite-ψ first
  integral for an asymmetric mixture (1:1 H⁺ + 1:1 Cs⁺ + 1:2 SO₄²⁻).
  The Gouy-Chapman `cosh(ψ)` form is specific to symmetric 1:1; the
  asymmetric version has different shape. This is the load-bearing
  derivation. Document in
  `docs/multi_ion_ic_derivation_2026-XX.md` (new). Update
  `picard_outer_loop` to consume the new ψ_D closure.
- **M3b.4 — Sulfate/Cs⁺ deck condition.** Plug in the deck-correct
  constants:
  - `[SO₄²⁻] = 100 mol/m³`, z = -2.
  - `[H⁺] = 0.1 mol/m³` (pH 4).
  - `[Cs⁺] = 199.9 mol/m³` (electroneutrality:
    `[H⁺] + [Cs⁺] = 2·[SO₄²⁻]`).
  - Cs⁺ effective steric radius (literature hydrated 3.29 Å Marcus, or
    bare 1.67 Å Shannon). Default to hydrated; revisit in M4/M6.
  - Drop ClO₄⁻ entry.
- **M3b.5 — Cation-configurable electrolyte.** Accept any of Li⁺/Na⁺/
  K⁺/Cs⁺ as a parameter. Per-cation steric radii from literature
  hydrated radii (Marcus). Required for M3c K⁺ validation against
  Ruggiero Fig 1B if that target stays.

**Code surface highlights.**

- `Forward/bv_solver/boltzmann.py:90-` — multi-ion residual.
- `Forward/bv_solver/forms_logc_muh.py` and `forms_logc.py` — IC
  composite-ψ derivation in `_try_debye_boltzmann_ic_*`.
- `Forward/bv_solver/picard_ic.py:compute_surface_gamma` — multi-ion γ.
- `scripts/_bv_common.py` — new analytic counterion entries
  (`DEFAULT_CSPLUS_BOLTZMANN_COUNTERION`,
  `DEFAULT_SULFATE_BOLTZMANN_COUNTERION`), new bulk concentrations.

**Verification.**

- Bulk recovery: `Σ z_i c_i = 0` at φ = 0.
- Dilute limit: collapse to single-ion when other species → 0
  (regression to Run C surrogate behavior).
- Electroneutrality at all ψ in the EDL.
- Pack-fraction `Σ a_i c_i ≤ 1` across the domain.
- **Mesh refinement audit.** At λ_D ≈ 0.55 nm and L_REF = 100 µm with
  β = 3 grading, compute cells per Debye in the first few normal
  cells. Run Ny ∈ {200, 400, 800} convergence test on surface H⁺,
  potential drop, and currents. Acceptance: < 3% relative change
  Ny=400 → Ny=800.
- **Single-V smoke at deck-correct I = 0.3 M.** Single anchor point
  before the full sweep. Expected difficulty: high. Allocate up to a
  day per anchor.

**Dependencies.** M3a.3 must land first. Two simultaneous IC reworks
would be intractable to debug.

**Risks.**

- **Anchor fragility at I = 0.3 M.** Per `project_ic_stern_bug.md`,
  the production anchor structure is already fragile (18/19 V points
  are warm-walks from a single anchor). 1000× ionic strength likely
  re-localizes the anchor. Plan: explicit anchor-rediscovery sub-task
  inside M3b.4 — single V_RHE re-anchor, then bracket-walk outward,
  THEN full 25-V sweep.
- **Newton conditioning at near-saturation.** May need preconditioner
  work or smaller pseudo-time-steps. Out-of-scope until empirical
  failure shows up.
- **Mesh budget.** Ny may need to grow 4-8× from 200. Watch wall time;
  C+D is already ~15 min/run at Ny=200.

### M3c — Local-pH validation against Ruggiero Fig 1B

**Cost: small (~1-2 days runtime + analysis), conditional on M3b.5.**

**Goal.** Confirm M3b reproduces Ruggiero Figure 1B: at bulk pH 4 with
K⁺, surface pH swings from 4 → ~8-9 at disk current density of
~3.25 mA/cm².

**Test protocol.**

- CP (constant-current) protocol: hold disk current at
  -0.10, -1.02, -2.04, -3.06, -3.32 mA/cm² for 5 min each (steady
  state). **Use area-normalized values per H18 V7** — not the raw mA
  values mislabeled as mA/cm² in the Ruggiero text.
- Compute surface pH from `c_H_surface_mean` via `pH = -log10([H⁺])`.
- Plot surface pH vs |disk current density|. Compare to Fig 1B with
  K⁺.

**Acceptance.**

- Trend: surface pH increases monotonically with |I_disk| ✓/✗.
- Quantitative: at -3.06 to -3.32 mA/cm², surface pH ∈ [7, 9]
  (semi-quant tolerance per page-15 acceptance tier).
- Linear regime check: surface pH approximately linear in |I_disk|
  across [-1, -3.32] mA/cm² (per Fig 1B) ✓/✗.

**Branches.**

- **All checks pass**: M3b is delivering the expected local-pH
  physics. Proceed to M4.
- **Trend right, magnitude off**: investigate OH⁻ handling and H⁺
  diffusivity. May indicate K_w-coupling assumption (locked decision
  7) is too coarse — consider tracking OH⁻ explicitly.
- **Trend wrong**: deeper bug. Halt M4; return to M3b.

**Risks.** Per H18 V3: PNP+Bikerman may not capture the cation
ordering. K⁺ vs Cs⁺ differences may require hydrolysis chemistry.
**Failure here is a finding, not necessarily a bug.**

### M4 — Cation specificity sweep

**Cost: medium. Conditional on extracted cation-series data.**

**Open dependency.** Need cation-ordering data for selectivity / peak
shifts at fixed pH. Sources:

- Ruggiero §3+ (pages 16-end of `docs/Ruggiero2022_JCatal_manuscript.pdf`
  already extracted per H18; check for a Li/Na/K/Cs figure).
- Mangan deck pages 14, 16-18 (NOT YET extracted — open task).

**Approach.**

- Run M3b.5 with Li⁺/Na⁺/K⁺/Cs⁺ in turn. Hold all other parameters
  fixed.
- Compare predicted ordering of:
  - H₂O₂ selectivity at fixed V_RHE.
  - Peak voltage / shoulder voltage of the gross R_2e current.
  - Total disk current at fixed V_RHE.
- Per H18 V4, calibrate `k0_4e` here (not earlier) using disk current
  / selectivity / Tafel slope, not just peroxide left plateau.

**Risks.** Cation-specific kinetic modulation (e.g. Cs⁺-stabilized
*-OOH) cannot be captured without extending the BV. Worst case: the
cation ordering is wrong even with multi-ion electrolyte correct.
This would be a finding that motivates a chemistry/activity layer
(out of scope for this plan).

### M5 — L_eff alignment

**Cost: small (~1 day). Scope is *peak shape and position*, NOT
mass-transport ceiling magnitude.**

**Goal.** Retune `L_REF` from 100 µm → ~90 µm to match deck p.16
empirical bracket (66-86 µm). Bare Levich at 1600 rpm with Ruggiero
constants is ~21 µm; deck calibrates higher to absorb boundary-layer
thinning + finite-disk + mesh effects.

**Note on scope (M3a.0 audit refinement).** The M3a.0 audit found that
experimental peak (|j| = 0.40 mA/cm²) is 2.18× the *current* 2e Levich
ceiling (0.183 mA/cm² at C_O2 = 0.5 mol/m³, L_REF = 100 µm) and flagged
M5 as "larger than 10%, ceiling needs to double." That finding is
**resolved by the C_O2 = 0.5 → 1.2 mol/m³ fix in M3a.2** (Ruggiero §2.4
gives 1.1–1.2 mol/m³ at pH 4): the post-fix ceiling at L_REF = 100 µm
is 0.44 mA/cm², which is ~10% above experimental peak — adequate
headroom. M5 therefore stays small in scope, retuning L_REF for shape
agreement only.

**Approach.**

- Run a small L sweep `L_REF ∈ {66, 76, 86, 90, 100, 120} µm` on the
  full M3b.4 stack at the page-15 voltage grid.
- Pick the L that minimizes peak-position and shoulder-presence
  errors (NOT peak-magnitude error in isolation, since magnitude is
  pinned by C_O2 and selectivity).

**Dependencies.** M3b.4 stack must be working.

### M6 — Stern + cation joint sensitivity

**Cost: medium.**

**Goal.** Identifiability analysis between Stern thickness
`stern_capacitance_f_m2` and cation effective radius (`a_nondim`).

**Approach.** Per the existing inverse plan handoffs. Out of scope
for the immediate page-15 quantitative comparison; revisit when
multi-target inference is back on.

## Sequencing summary

```text
M3a.0 (in flight)              # observable audit, no code
   │
   ▼
M3a.1 (electron-weighted obs)  # ~0.5–1 day, isolated
   │
   ▼
M3a.2 (parallel residual,      # ~3–5 days
        diagnostic IC)
   │
   ▼
M3a.3 (production parallel IC) # ~1–2 weeks
   │
   ▼
M3b.1 → M3b.5 (multi-ion)      # multi-week
   │
   ▼
M3c (local-pH val)             # ~1–2 days
   │
   ▼
[deck p.14, 16-18 extraction]  # blocked open task
   │
   ▼
M4 (cation specificity)        # medium
   │
   ▼
M5 (L_eff retune)              # ~1 day
   │
   ▼
M6 (Stern+cation joint)        # medium, optional
```

Critical path is M3a.0 → M3a.1 → M3a.2 → M3a.3 → M3b.3 → M3b.4 →
production page-15 comparison. Everything else is downstream of the
production stack landing.

## Decision branches (where this plan adapts)

1. **M3a.0 outcome.** If gross R_0 matches both magnitude and shape,
   M3b becomes optional for page-15 specifically (still required for
   cation-ordering work). If gross R_0 matches nothing, halt and
   debug forward solver before M3a.1.
2. **M3a.2 convergence rate.** If <6/10 V converges, escalate M3a.3
   early (don't burn time chasing the diagnostic).
3. **M3a.2 mixed-config plausibility.** If gross R_2e is flat or
   sign-wrong at the midpoint of the V grid, the parallel reaction
   physics has a bug in our wiring (not a Ruggiero problem). Halt and
   debug.
4. **M3b.3 IC convergence at I = 0.3 M.** If single-V anchor doesn't
   converge after a week of mesh / preconditioner / pseudo-time-step
   tuning, consider:
   - Stepping ionic strength gradually (continuation in I).
   - Decoupling Cs⁺ from SO₄²⁻ (test with two ideal 1:1 + 1:1 first).
5. **M3c trend wrong.** Return to M3b — check OH⁻ handling. Don't
   advance to M4 until trend is qualitatively right.
6. **M4 cation ordering wrong.** This is a *finding*, not a bug.
   Document, escalate to a chemistry-layer plan, do not patch with
   hand-tuned per-cation k0.

## Risks (cross-cutting)

| Risk | Where | Mitigation |
|------|-------|------------|
| C_O2 fix ripples through all anchors | M3a.2 | Single-V regression smoke vs Run C before going wide |
| Picard divergence at parallel topology | M3a.3 | omega=0.5 cap, max_iters=50, linear-phi fallback wired |
| Anchor relocation post-E_EQ_R2E refinement | M3a.3 | Plan single-V re-anchor before 25-V sweep |
| Anchor fragility at I = 0.3 M | M3b.4 | Continuation in I; bracket-walk; explicit anchor sub-task |
| Mesh insufficient at λ_D ≈ 0.55 nm | M3b | Ny convergence audit before declaring M3b.4 done |
| Cation ordering not captured | M4 | Tier interpretation; a chemistry layer is out of scope but possible |
| Adjoint tape hygiene during Picard refactor | M3a.3 | Preserve `adj.stop_annotating()` wrapper; verify on inverse re-entry |
| k0_4e weakly identified | M3a.2, M4 | Document prior; calibrate against disk current + selectivity, not just peroxide left plateau |

## What this plan deliberately defers

- **Mangan deck pages 14, 16-18 extraction.** Required for k0/α/L_eff
  arbitration in M4 and M5. Open task; not part of forward-solver work.
- **Inverse work.** Paused until forward solver is mature.
- **OH⁻ as tracked species.** Locked decision 7 — start with K_w-coupled.
  Revisit only if M3c fails on local-pH magnitude.
- **Hydrolysis / activity chemistry layer.** Possibly required for
  cation ordering (M4); out of scope until empirical evidence demands it.
- **Multi-cation deck experimental targets.** Page 15 is single-cation
  (Cs⁺). Different deck pages with cation series would need their own M0.

## Implementation status (2026-05-07 session)

### What landed

- **M3a.1 — DONE.** Electron-weighted `current_density` mode and new
  `gross_h2o2_current` mode in `Forward/bv_solver/observables.py`.
  Pulls `n_electrons` per reaction from
  `ctx["nondim"]["bv_reactions"][j]`. Falls back to unweighted sum when
  the reactions list is unavailable (legacy ctx). Legacy
  `peroxide_current` (R_0 − R_1) retained for v13/v15/v16 back-compat
  with deprecation note. Test file:
  `tests/test_observables_electron_weighting.py` (10 new tests, all
  pass; 79 regression tests in adjacent suites pass).
- **M3a.2 plumbing — DONE (does not converge end-to-end; see M3a.3).**
  - `scripts/_bv_common.py`: added Ruggiero-aligned constants
    (`E_EQ_R2E_V=0.695`, `E_EQ_R4E_V=1.23`, `K0_PHYS_R2E`, `K0_PHYS_R4E`
    placeholder = `K0_PHYS_R1`, `ALPHA_R4E=0.5`, `K0_HAT_R{2E,4E}`),
    `PARALLEL_2E_4E_REACTIONS` reaction list literal.  (M3a.2 originally
    only *documented* `C_O2_PHYS_RUGGIERO = 1.2 mol/m³`; the actual
    `C_O2 = 0.5 → 1.2` flip landed in M3a.2.1 — see "DONE 2026-05-07"
    note below; the legacy value is retained as `C_O2_PHYS_LEGACY`.)
  - `make_bv_solver_params` accepts new `bv_reactions=None` parameter
    that overrides the legacy R_1/R_2 keyword bundle when set.
    `_make_bv_bc_cfg` deep-copies caller-supplied entries so internal
    nondim mutation doesn't leak. **Legacy path verified unchanged**
    by 79/79 regression tests.
  - `Forward/bv_solver/forms_logc{,_muh}.py`: topology gate that
    rejects parallel topology from the matched-asymptotic Picard with
    `non_sequential_topology` reason; orchestrator falls through to
    `linear_phi` IC. Gate triggers correctly in tests.
  - `scripts/studies/peroxide_window_3sp_parallel_2e_4e.py`: 3-pass
    diagnostic driver (pure-2e / pure-4e / mixed). Built but
    unrunnable until M3a.3 lands; convergence smoke fails universally.
- **M3a.2.1 — DONE 2026-05-07.** `C_O2 = 0.5 → 1.2 mol/m³` migration
  landed. Source flip in `scripts/_bv_common.py:79` (one line); the
  former `C_O2_PHYS_RUGGIERO = 1.2` constant was demoted to
  `C_O2_PHYS_LEGACY = 0.5` for any pre-fix comparison plots. All
  affected tests (`test_steric_boltzmann_closure_algebra.py`,
  `test_initializer_debye_boltzmann.py`,
  `test_initializer_debye_boltzmann_3sp_bikerman.py`,
  `test_initializer_debye_boltzmann_4sp.py`,
  `test_initializer_debye_boltzmann_4sp_muh.py`,
  `test_picard_ic_helpers.py`, `test_steric_psi_profile.py`,
  `test_rrde_observables.py`) now import `C_HP_HAT` / `C_CLO4_HAT` /
  `A_DEFAULT` from `scripts._bv_common` instead of hardcoding `0.2`,
  so they track future C_O2 changes automatically. Snapshot tests
  (`test_stern_no_stern_snapshot.py::test_no_stern_cd_pc_matches_baseline`
  and `test_steric_boltzmann_closure.py::test_ideal_path_byte_identical`)
  marked `xfail(strict=True)` pending baseline regeneration in the
  slow suite. Run C / m3a0 audit remain anchored to legacy `C_O2 = 0.5`
  by design (see header note in `mangan_p15_m3a0_audit.py`); they are
  pre-fix references and should not be reused for post-fix runs.
  Fast pytest delta vs baseline: 0 new failures (455 passed both
  pre- and post-flip, with the same 6 pre-existing inverse-stack
  failures unrelated to C_O2). The audit's prediction (post-fix
  I_SCALE ≈ 0.44 mA/cm² vs experimental peak 0.40 mA/cm²) is now
  the operative ceiling for M5 retune scoping — confirms M5 stays
  small (~10% L_eff) per the M3a.0 audit refinement note above.

### Parameters used in failed M3a.2 cold-convergence probe

For the next session's context — these are the full settings the
parallel topology was tested with:

| Knob | Value | Notes |
|------|-------|-------|
| `species` | `THREE_SPECIES_LOGC_BOLTZMANN` | 3sp (O₂, H₂O₂, H⁺) |
| `formulation` | `"logc_muh"` | proton μ primary variable |
| `log_rate` | `True` | log-rate BV |
| `boltzmann_counterions` | `[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]` | bikerman steric ClO₄⁻ |
| `stern_capacitance_f_m2` | `0.10` F/m² | Stern active |
| `u_clamp` | `100.0` | log-c bulk clamp |
| `exponent_clip` | tested at both `50.0` and `100.0` | both cold-fail for parallel |
| `initializer` | `"linear_phi"` (and `"debye_boltzmann"` which gates → linear_phi) | |
| Mesh | `Nx=8, Ny=200, beta=3.0` | production graded mesh |
| SNES | `SNES_OPTS_CHARGED` + `max_it=400, linesearch=l2, maxlambda=0.3, divergence_tol=1e10` | |
| Orchestrator | `solve_grid_per_voltage_cold_with_warm_fallback` (C+D), `max_z_steps=20, n_substeps_warm=8, bisect_depth_warm=5` | |
| Reactions | `PARALLEL_2E_4E_REACTIONS` (+ k0 factor sweep tested at 0, 1e-30, 1.0) | |
| C_O2 (deferred fix) | `0.5 mol/m³` (legacy unchanged) | M3a.2.1 will fix to 1.2 |

These are the exact same settings as the legacy
`peroxide_window_3sp_bikerman_muh.py` reference (which gets 25/25)
modulo the parallel `bv_reactions` override and the consequent
linear_phi fallback. The legacy stack at V=0.0, clip=50,
debye_boltzmann IC reproduces Run C `cd = -0.175 mA/cm²` cold ✓ —
control test passes, confirming env / mesh / SNES are sane.

### M3a.3 escape hatch (alternative to full Picard generalization)

Before committing to the 1–2 week Picard rewrite, worth ~½ day on a
**warm-from-Run-C shortcut**:

- Load `StudyResults/mangan_p15_comparison/run_C/` solver state at one
  stable V (e.g., V_RHE=0.0 V where legacy cold-converged at
  cd=-0.175).
- Hand-seed it as the IC for the parallel R_2e/R_4e residual at the
  same V. The Run C O₂/H₂O₂/H⁺/φ profiles are already physically
  reasonable; Newton has a much better starting point than `linear_phi`.
- Solve the parallel residual with this IC. If it converges, use it as
  the warm-walk anchor for the full V grid via C+D.

If this works, M3a.2 produces output without the Picard rewrite and
the audit's gross-R_2e prediction is testable. If it doesn't, the
generalized Picard is unavoidable. Either way the result informs
M3a.3 scoping.

The orchestrator doesn't natively support cross-reaction-set warm
starts; would be a custom one-off script that loads the converged
state from disk and feeds it to the parallel residual at a single V,
no warm-walk yet.

## Code-surface checklist (one-line summary)

- ✅ `scripts/_bv_common.py:73-93` — R_4e physical constants added
  (`E_EQ_R2E_V`, `E_EQ_R4E_V`, `K0_PHYS_R{2E,4E}`, `ALPHA_R{2E,4E}`,
  `K0_HAT_R{2E,4E}`); `C_O2 = 0.5 → 1.2 mol/m³` migration applied
  2026-05-07 (M3a.2.1).  Legacy retained as `C_O2_PHYS_LEGACY = 0.5`.
- ✅ `scripts/_bv_common.py:~430` — `PARALLEL_2E_4E_REACTIONS`
  reaction-list literal added; H⁺ stoichiometric factor with
  `power = n_electrons` per reaction.
- ✅ `scripts/_bv_common.py:_make_bv_bc_cfg + make_bv_solver_params` —
  accept arbitrary `bv_reactions=` list; deep-copies caller entries.
- ✅ `Forward/bv_solver/observables.py` — electron-weighted
  `current_density`; new `gross_h2o2_current` mode; legacy
  `peroxide_current` retained with deprecation note.
- ⏳ `Forward/bv_solver/picard_ic.py:294-440` — generalize
  `picard_outer_loop` to N reactions with stoichiometry-weighted flux
  balance. **NOW CRITICAL PATH** (M3a.3).
- ✅ `Forward/bv_solver/forms_logc_muh.py:782-` and
  `forms_logc.py:707-` — topology gate added (rejects parallel from
  matched-asymptotic Picard with `non_sequential_topology`). M3a.3
  will replace the gate with a generalized Picard.
- ⏳ `Forward/bv_solver/boltzmann.py:90-` — drop single-counterion
  guard; generalize multispecies γ. (M3b.)
- ⏳ `Forward/bv_solver/picard_ic.py:compute_surface_gamma` — multi-ion γ. (M3b.)

## References

- `StudyResults/mangan_p15_comparison/m3a0_audit.md` — **M3a.0 audit
  report (DONE 2026-05-07);** verdict and per-channel band tables.
- `StudyResults/mangan_p15_comparison/m3a0_observables.json` — derived
  observables JSON (cd, pc_net, gross_R0).
- `StudyResults/mangan_p15_comparison/m3a0_audit.png` — overlay plot.
- `docs/Ruggiero2022_JCatal_source_paper.md` — paper extraction.
- `docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md` —
  the live H18 substaging structure.
- `docs/CHATGPT_HANDOFF_17_RUGGIERO_REALIGNMENT_PLAN.md` — initial plan
  (superseded by H18, kept as audit trail).
- `docs/mangan_alignment_status_2026-05-07.md` — status doc; pickup
  point for the next session.
- `docs/mangan_p15_comparison_summary.md` — Run D verdict + reframing.
- `docs/m0_target_extraction.md` — page-15 M0 outputs (RRDE constants,
  acceptance bands).
- `docs/clipping_conventions.md` — clip=100 mandate (CLAUDE.md hard
  rule 2).
- `docs/4sp_bikerman_ic_option_2b_results.md` — production reference
  sweep (15/15 V_RHE, baseline for parallel-topology regression).
- `CLAUDE.md` — production stack flags, hard rules.
- `~/.claude/plans/swirling-crunching-wren.md` — Plan B (page-15
  comparison execution; superseded by this plan post-Ruggiero).
- `memory/project_mangan_m0_extraction_complete.md` — page-15 M0
  memory entry; should be updated after M3a.3 lands.
