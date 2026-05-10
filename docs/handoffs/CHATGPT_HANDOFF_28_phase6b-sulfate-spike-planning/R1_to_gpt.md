# R1 — Phase 6β planning handoff for adversarial review

## 1. Context bundle

### 1.1 Repo and production solver

**Repo:** `PNPInverse` — Firedrake-based PNP–BV forward solver for ORR
(O₂ → H₂O₂ → H₂O) with the longer-term goal of inverse kinetic
inference. The forward solver is currently the active surface; inverse
work is paused.

**Production stack (May 2026):**

* 3 dynamic species: O₂ (idx 0), H₂O₂ (idx 1), H⁺ (idx 2).
* Analytic Bikerman counterion for ClO₄⁻ (`steric_mode='bikerman'`,
  residual-side closure plus Bikerman-consistent IC). Currently the
  perchlorate counterion species; will need to become SO₄²⁻ /
  HSO₄⁻ in Phase 6β.
* Proton electrochemical-potential primary variable
  (`formulation='logc_muh'`, `μ_H = u_H + em·z_H·φ`).
* Log-rate Butler–Volmer (`bv_log_rate=True`).
* Finite Stern compact layer (`stern_capacitance_f_m2 ≈ 0.10`).
* `debye_boltzmann` IC (composite-ψ + multispecies-γ).
* C+D continuation (cold ceiling +0.60 V; warm-walk to +1.00 V).
  Strategy B is broken on this stack — do not propose it.

**Live backends:**
`Forward/bv_solver/forms_logc.py`, `Forward/bv_solver/forms_logc_muh.py`.
The legacy concentration backend was deleted in the May 2026 cleanup.

### 1.2 Reaction set as currently configured

From `scripts/_bv_common.py:660-690`:

```python
# R2e (parallel 2e ORR): O₂ + 2H⁺ + 2e⁻ ⇌ H₂O₂
{
    "k0": K0_HAT_R2E,
    "alpha": ALPHA_R2E,             # 0.5
    "cathodic_species": 0,          # O₂ consumed → cathodic = exp(... + u_O2 ...)
    "anodic_species": 1,            # H₂O₂ produced (reverse)
    "stoichiometry": [-1, +1, -2],  # consumes 2 H⁺
    "n_electrons": 2,
    "reversible": True,
    "E_eq_v": 0.695,                # E°_R2e (Ruggiero §1)
    "cathodic_conc_factors": [
        {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
    ],
},
# R4e (parallel 4e ORR): O₂ + 4H⁺ + 4e⁻ → 2H₂O (irreversible)
{
    "k0": K0_HAT_R4E,
    "alpha": ALPHA_R4E,
    "cathodic_species": 0,          # O₂
    "anodic_species": None,
    "stoichiometry": [-1,  0, -4],
    "n_electrons": 4,
    "reversible": False,
    "E_eq_v": 1.23,                 # E°_R4e (Ruggiero §1)
    "cathodic_conc_factors": [
        {"species": 2, "power": 4, "c_ref_nondim": C_HP_HAT},
    ],
},
```

This is the **parallel 2e + 4e topology** the deck requires (Ruggiero
2022 J.Catal). Already implemented in the current solver — the
"sequential R₀+R₁" framing in earlier docs is obsolete.

### 1.3 Reference concentrations and scaling

From `scripts/_bv_common.py:69-205`:

* `C_O2 = 1.2 mol/m³` (Ruggiero 2022 §2.4 — pH 5–13 deck-correct).
* `C_HP = 0.1 mol/m³` ≡ pH 4. This is **the** reference used in the
  cathodic concentration factor: `C_HP_HAT = 0.0833` after nondim
  rescale.
* `KW_HAT = KW_PHYS / C_SCALE² ≈ 6.944e-9`. At bulk pH 4,
  `C_OH_BULK_HAT = KW_HAT / C_HP_HAT ≈ 8.33e-8`.
* `c_ref_nondim = C_HP_HAT` for both reactions, with `power = 2` (R2e)
  and `power = 4` (R4e).

### 1.4 Log-rate BV residual (live)

From `Forward/bv_solver/forms_logc.py:411-462` (cathodic only shown):

```
log_r = ln(k0) + u_cat + Σ_factor [ power * (u_sp − ln c_ref) ]
        − α · n_e · η_clipped
R = exp(log_r)
```

So for R4e:

```
log_r_R4e = ln(k0_R4e) + u_O2(0,t)                          # O2 surface
          + 4 * (u_H(0,t) − ln C_HP_HAT)                     # H+ factor
          − 0.5 · 4 · η_clipped                              # α·n_e·η, η = (V_RHE − E°_R4e)/V_T
```

`η_clipped` is `(V − E_eq) / V_T` clipped to ±100 *before* the
`α·n_e` multiplication. `exponent_clip = 100` is the only
PC-trustworthy setting; `clip = 50` produces fictitious peroxide
current and must not be used.

`u_i = ln c_i_nondim`. So `u_H(0,t) − ln C_HP_HAT = ln(c_H_surf / c_H_ref)`.

For pH 10.6 surface (current Phase 6α 100 µm × 1e-18 case):
`c_H_surf / c_H_ref = 10^(4 − 10.6) ≈ 2.5·10^(−7)`.
R4e cathodic factor: `(2.5·10^(−7))^4 ≈ 4·10^(−27)`.
R2e cathodic factor: `(2.5·10^(−7))^2 ≈ 6·10^(−14)`.

This means the cathodic **concentration** factor is enormously
suppressed at surface pH 10.6, but the BV **overpotential** factor
`exp(α·n_e·|η|)` at deep cathodic V_RHE = −0.4 V is huge:
`η = (−0.4 − 1.23)/0.02569 = −63.4`, clipped at ±100, OK.
`exp(0.5·4·63.4) = exp(126.8) ≈ 4·10^54`.

So `R4e ∝ (k0_R4e) · 4·10^(−27) · 4·10^54 · c_O2_surf`. With
`k0_R4e = ratio · k0_R2e`, ratio = 1e-18, the rate ends up finite —
the magnitudes balance with overpotential dominating. The current
model is in a regime where cathodic concentration factor *is*
suppressed but overpotential exponential more than compensates.

### 1.5 Phase 6α outcome (water self-ionization)

Phase 6α added a proton-condition residual on `E = c_H − c_OH` with
the fast-equilibrium closure `c_OH = K_w_eff · exp(−u_H)`. Module:
`Forward/bv_solver/water_ionization.py`. Default-off via
`solver_options['bv_convergence']['enable_water_ionization']`;
disabled path is byte-equivalent to pre-Phase-6α.

Continuation pattern (used as the architectural template for any 6β
work): a 5-rung `kw_eff_ladder` in
`Forward/bv_solver/anchor_continuation.py` ramps `Kw_eff` from a tiny
floor up to `KW_HAT` *after* the k0 ladder converges at `Kw_eff = 0`.

Verified Phase 6α numerical state (post-water-ionization, currently
the running sweep 4/8 done at time of writing):

| combo | cd[V=−0.4] mA/cm² | surface pH[V=−0.4] | converged |
|---|---:|---:|---|
| L100µm × 1e-18 | −0.737 | 10.61 | 13/13 |
| L100µm × 1e-30 | −0.440 | 10.34 | 13/13 |
| L66µm × 1e-18 | −1.122 | 10.60 | 13/13 |
| L66µm × 1e-30 | −0.667 | 10.33 | 13/13 |
| L21µm × {1e-18,1e-30} | (running) | — | — |
| L16µm × {1e-18,1e-30} | (pending) | — | — |

100µm × 1e-18 voltage table (from
`StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/L100um_ratio_1e-18/iv_curve.json`):

| V_RHE | cd mA/cm² | surface pH proxy |
|---:|---:|---:|
| −0.4000 | −0.7373 | 10.6101 |
| −0.3208 | −0.5450 | 10.5539 |
| −0.2417 | −0.4553 | 10.5585 |
| −0.1625 | −0.4338 | 10.6331 |
| −0.0833 | −0.3127 | 10.5430 |
| −0.0042 | −0.1537 | 10.0935 |
| +0.0750 | −0.1014 | 9.4360 |
| +0.1542 | −0.0915 | 8.6768 |
| +0.2333 | −0.0901 | 7.8956 |
| +0.3125 | −0.0898 | 7.1242 |
| +0.3917 | −0.0890 | 6.3884 |
| +0.4708 | −0.0851 | 5.6984 |
| +0.5500 | −0.0661 | 5.0599 |

|cd| is monotonically increasing as V_RHE goes more cathodic. **No
peak**, no decay. Surface pH is also monotonic in V_RHE: drifts from
~5 (at +0.55 V, near R2e equilibrium) up to ~10.6 (at −0.40 V, deep
cathodic). The 10.6 ceiling matches the K_w / c_OH balance — model is
"saturated" at the OH-ionization-supplied cap.

### 1.6 Deck the model is targeting

Per `docs/Mangan2025_experimental_alignment.md` and
`docs/seitz_mangan_data_folder_audit_2026-05-08.md`:

* Real experimental data is **K₂SO₄** electrolyte, **not** ClO₄⁻.
* Parallel 2e⁻ (E° = 0.695 V) + 4e⁻ (E° = 1.23 V) ORR per Ruggiero
  2022 J.Catal §1.
* pH range tested 1–6.
* Deck shape (Cs⁺ pH 4 reference column from Yash's plotting workflow,
  via `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` — currently
  **missing** from the local data folder):
  * cathodic onset around +0.15 V_RHE
  * peak around +0.10 V_RHE
  * plateau / left magnitude around −0.18 mA/cm²
  * peak magnitude around −0.40 mA/cm² where applicable
  * decay at more cathodic potentials
  * cation identity changes peak height/shape
  * surface pH should remain in the experimentally plausible
    operating window, ~4–9 ideally 4–7

* Closest substitute available locally:
  `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/0,1M K2SO4 data 8-15-19.xlsx`
  (RRDE LSV at six pH values; columns include `E_disk (V vs RHE)`,
  `j_disk`, `j_ring`, `H2O2%`, `n_e`, `Overpotential`).

### 1.7 K0_R4e / K0_R2e ratio — what's known

* Memory + audit: ratio ≈ 1e-18 produces ~35–50 % peroxide selectivity
  at moderate cathodic V_RHE, qualitatively matching Mangan deck
  selectivity.
* ratio ≤ 1e-24 saturates pure-2e (almost no R4e current).
* Sweep tests both `1e-18` and `1e-30` to bracket. The handoff treats
  both as candidate inputs to the spike.

### 1.8 Hard solver invariants

* Use `solve_grid_per_voltage_cold_with_warm_fallback` (C+D), not
  `solve_grid_with_charge_continuation` (B).
* Keep `exponent_clip = 100`. clip = 50 produces fictitious peroxide
  current (sign-flipped at V_RHE < −0.1 V); not PC-trustworthy.
* Use physical `E_eq` (R1 = 0.68 V, R2 = 1.78 V; R2e = 0.695 V,
  R4e = 1.23 V).
* `set_initial_conditions(blob=True)` is silently ignored in log-c
  mode.
* Must run from `PNPInverse/`, with `venv-firedrake` activated.
* Cache env: `MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1`.

---

## 2. The artifact under review (full plan)

This is the plan I (Claude) produced inline and committed to disk at
`docs/phase6b_next_steps_plan.md`. The auto-revise step at end-of-loop
will edit this file in place based on accepted feedback.

```markdown
# Phase 6β next-steps plan (v1, pre-critique)

## Verified Phase 6α state (vs handoff 27 snapshot)

[as described in §1.5 above — the table + voltage curve. Phase 6α is
mid-flight, 4/8 combos refreshed, plotting/scoring deferred to step 2.]

## Step 1 — Let Phase 6α finish (passive)

ETA: ~30–40 min remaining. 21 µm runs faster than 66/100 µm
(~13 s anchors vs ~50 s), then 16 µm faster still. No action needed;
do not touch `l_eff_transport_sweep/` or the running process.

## Step 2 — When PID 2509 exits

1. Verify all 8 `iv_curve.json` timestamps ≥ 21:23 with the `stat`
   loop from §10 of the handoff.
2. Regenerate aggregates from the now-coherent directory:
   * `python scripts/studies/plot_l_eff_transport_sweep.py`
   * `python scripts/studies/score_l_eff_sweep.py`
3. Confirm `verdict.json` no longer reports the old `−0.0898 mA/cm²`
   H⁺ Levich numbers.
4. **Defer** the `--out-dir`/`--only-l-eff`/`--resume` CLI retrofit
   (handoff §9 task 1). Note as a follow-up if/when 6β requires it.

## Step 3 — Start the 6β algebra spike now (parallel to step 1)

Deliverable: a script + writeup that answers: **does fast HSO₄⁻ ⇌
SO₄²⁻ + H⁺ equilibrium, applied as a post-hoc closure on the
existing surface pH/c_H curves, raise or lower the predicted ORR
current factor `(c_H/c_H_ref)^n` at each grid voltage?**

Concrete output:

* New file: `scripts/studies/phase6b_sulfate_algebra_spike.py`
* Inputs: `L100um_ratio_{1e-18,1e-30}/iv_curve.json` (and 66µm once
  available).
* For each grid voltage:
  * Read `surface_pH_proxy` and `c_H_surface_nondim`.
  * Apply HSO₄⁻/SO₄²⁻ algebra at I = 0.3 M using literature
    `pKa2 ≈ 1.99`.
  * Compute the implied surface c_H if a fixed total sulfate
    inventory `[SO₄²⁻]_T = 0.3 M` is enforced via the equilibrium →
    predict pH shift toward 4–7 band.
  * Multiply the existing predicted current by the new
    `(c_H_new / c_H_old)^n` factor for `n ∈ {0.5, 1.0}`. **Note: in
    the current solver n is 2 (R2e) and 4 (R4e), not 0.5/1.0 — see
    §1.2. The plan as written has the wrong powers.**
* Output:
  `StudyResults/fast_realignment_2026-05-08/phase6b_sulfate_spike/`
  * `algebra_summary.json`, `pH_shift_overlay.png`,
    `cd_rate_factor_overlay.png`, `verdict.md`.

## Step 4 — Branch decision after step 3

* **Branch A — sulfate sign right** (peak appears in +0.0…+0.20 V_RHE):
  proceed to **6β.1**, algebraic equilibrium closure.
  * New module `Forward/bv_solver/sulfate_buffering.py` mirroring
    `water_ionization.py`.
  * Config keys: `enable_sulfate_buffering`, `pKa2`, total sulfate,
    activation factor, optional HSO₄⁻ diffusivity/steric.
  * Conservation: total sulfate (algebraic split between SO₄²⁻ and
    HSO₄⁻); proton condition coupled to sulfate acid–base state.
  * Poisson source includes SO₄²⁻ and HSO₄⁻ separately; Bikerman A_dyn
    includes both.
  * Activation continuation analogous to `kw_eff_ladder`.
  * Smoke-test on L100µm × 1e-18 first; only then full sweep.

* **Branch B — sulfate sign wrong** (rate factor goes wrong direction
  or amplifies cathodic current with no peak): write **6δ plan**
  before touching solver.
  * Required: alkaline-form ORR or local-pH-dependent BV form switch
    (acid → alkaline as surface pH rises).
  * Defer 6γ (cation-OHP) until 6δ separates kinetic-regime from
    sulfate buffering.

## Step 5 — Cross-check against the deck before any solver work

Compare algebra-spike output to
`data/.../0,1M K2SO4 data 8-15-19.xlsx` cd at V_RHE ∈ [−0.4, +0.55] at
pH 4. If algebraic prediction within ~30 % of deck peak/plateau ratio,
6β.1 is justified. If > 2× off in magnitude or sign, 6δ is mandatory.
```

(Full text in `docs/phase6b_next_steps_plan.md`.)

---

## 3. Specific points I want pressure-tested

For each, give me the actual answer / deeper view, not just "consider
this." If a question presupposes something false, name what's false.

1. **Equilibrium algebra.** Is fast-equilibrium HSO₄⁻ / SO₄²⁻ at
   I = 0.3 M with `pKa2 ≈ 1.99` actually defensible without a Davies
   or Pitzer activity correction? At I = 0.3 M the activity coefficient
   for SO₄²⁻ is roughly γ ≈ 0.1, so the effective Ka2 in the spike's
   simple `[SO₄²⁻][H⁺] / [HSO₄⁻]` ratio could be off by ~1 order of
   magnitude. Does this kill the spike's quantitative claim?

   At pH ~10.6 surface, `[SO₄²⁻] / [HSO₄⁻] = 10^(10.6 − 1.99) ≈ 4·10^8`,
   so essentially all sulfate is in the SO₄²⁻ form at the surface
   (even more so when the activity coefficient pushes Ka up). Buffering
   is one-sided in the sense that there's almost no HSO₄⁻ reservoir at
   pH 10.6 to release H⁺; the reservoir is bulk sulfate at pH 4 (where
   `[SO₄²⁻]/[HSO₄⁻] ≈ 10^2`, both forms substantial). Does this kill
   the spike's ability to predict a peak before it runs? Or is the
   relevant buffer not the surface speciation but the *gradient* in
   sulfate / proton between surface and bulk?

2. **Post-hoc closure assumption.** The spike treats existing surface
   c_H from a Phase 6α PNP solve as input, then re-derives an
   equilibrated c_H′ under sulfate. The real coupled solve will have
   *different* surface concentration fields because (a) Poisson charge
   changes when SO₄²⁻ / HSO₄⁻ enter at 0.3 M total, (b) the H⁺ proton
   condition changes when sulfate competes for charge balance, (c) the
   Bikerman packing changes with new species sizes, (d) the BV flux
   itself changes with the new c_H surface. Is the post-hoc
   approximation defensible enough to make a 6β.1-vs-6δ branch
   decision, or is the sign of the rate-factor change so coupled to
   the full PDE solve that the spike could mislead in either
   direction?

3. **Kinetic-form claim.** Handoff §4 says acid-form BV with
   `(c_H/c_H_ref)^n` cathodic factor *strengthens* cathodic rate as
   c_H rises (pH 10 → 6–7), so sulfate "merely amplifies" current. Is
   that exactly right? Or does the `(c_O2)^m` factor (m = 1 for both
   R2e and R4e per stoichiometry above) plus mass-transport coupling
   produce a peak even with monotonic kinetic factor?

   What mechanism actually produces decay past a peak in PNP-BV?
   Candidates I can think of:

   * c_O2 surface depletion at deep cathodic (mass-transport floor on
     O2). Currently the model shows monotonically increasing |cd| at
     V_RHE = −0.4 — so c_O2 depletion isn't binding here. Is that
     because L_eff = 100 µm is too short and Levich limit on O2
     hasn't kicked in?
   * c_H surface depletion (Levich limit on H⁺). Phase 6α removed
     this via water ionization.
   * Local-pH-driven kinetic-regime switch (acid form → alkaline form
     as surface pH rises through ~7). The alkaline-form rate would
     have *no* `(c_H)^n` factor and a different α; switching to it as
     surface pH rises would make the rate factor *decrease* through
     the transition.
   * Site-blocking adsorbed intermediates (not in the current model).

   Which of these is most likely to produce the deck's peak shape?

4. **Mangan / Ruggiero parallel-2e/4e structure.** The deck and the
   current solver both use *parallel* 2e⁻ + 4e⁻ (E°_R2e = 0.695 V,
   E°_R4e = 1.23 V). Some earlier handoffs say "sequential R₀+R₁" —
   that framing is **obsolete** as of the May 2026 reaction-config
   landed in `_bv_common.py:660-690`. So there's no structural
   mismatch to invalidate the spike. **Confirm this matches the
   evidence in §1.2 above, or call out where I'm wrong.**

5. **Operating window.** Deck peak at +0.10 V_RHE. The 100 µm × 1e-18
   table shows roughly monotonic |cd| with V_RHE; spacing is ~80 mV
   (13 voltages over a 0.95 V range). Could a peak in +0.05…+0.15 V be
   hidden by undersampling, or is the trend obviously monotonic from
   the table?

   Inspect the values:
   * +0.234 V → −0.0901
   * +0.155 V → −0.0915
   * +0.075 V → −0.1014
   * −0.004 V → −0.1537
   * +0.083 V → −0.3127

   The last two are almost identical V_RHE (−0.004 vs +0.083), but cd
   jumps 2× between them. **Wait — these are out of order. The table
   in §1.5 has them in order; let me recopy:**

   ```
   V_RHE          cd mA/cm²
   +0.5500    →  −0.0661
   +0.4708    →  −0.0851
   +0.3917    →  −0.0890
   +0.3125    →  −0.0898
   +0.2333    →  −0.0901
   +0.1542    →  −0.0915
   +0.0750    →  −0.1014
   −0.0042    →  −0.1537
   −0.0833    →  −0.3127
   −0.1625    →  −0.4338
   −0.2417    →  −0.4553
   −0.3208    →  −0.5450
   −0.4000    →  −0.7373
   ```

   Now the curve is clearly monotonic in V_RHE (descending |cd| as
   V_RHE → positive). The **only** interesting feature is the inflection
   between +0.0750 and −0.0833 where the slope `Δcd / ΔV` jumps from
   ~0.13 mA/V to ~2.7 mA/V. Could that inflection actually be the
   *foot* of a peak that the spike+model would resolve into a peak if
   the surface pH dropped 4 decades?

6. **Deck data validity.** Without the missing
   `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`, is
   `0,1M K2SO4 data 8-15-19.xlsx` an acceptable substitute? I don't
   know the cation in the K₂SO₄ file (likely K⁺); the Tafel-slope
   workbook had the Cs⁺ column. If only K⁺ is available, does the
   cation identity gap invalidate the comparison enough that step 5's
   30 % gate is meaningless?

7. **Ratio dependence.** Should the spike use ratio 1e-18, 1e-30, or
   both? Does sulfate buffering interact with K0_R2e/K0_R4e ratio in
   a way that changes the sign answer? Both ratios show the same
   monotonic-no-peak shape qualitatively; does that mean ratio is
   irrelevant for the sign question, or is it just irrelevant in the
   *current* (no sulfate) configuration?

8. **6β.1 conservation correctness.** The handoff warns against adding
   `R_buf = k * (c_HSO4 − c_SO4 · c_H / Ka)` only to the H⁺ residual
   without sulfate bookkeeping. The plan says "use algebraic
   speciation with total sulfate as conserved variable." Is that
   actually solvable as a PDE-with-algebraic-constraint inside
   Firedrake's Newton solver? Specifically:

   * Total sulfate `c_S_total = c_SO4 + c_HSO4` is conserved (no
     source / sink).
   * Speciation `c_HSO4 = c_S_total · c_H / (c_H + Ka)`,
     `c_SO4 = c_S_total · Ka / (c_H + Ka)`.
   * H⁺ residual gets a sulfate proton-condition extension: the
     proton condition becomes `E = c_H + c_HSO4 − c_OH − ...`?

   Or is the "right" form to introduce two NP equations — one for
   c_S_total, one for c_HSO4 (or c_SO4) — and let the solver find the
   speciation by minimizing the residual `c_HSO4 − c_S_total · c_H /
   (c_H + Ka)`? Would Newton converge?

9. **6δ scope.** If branch B fires, the plan calls 6δ "alkaline-form
   ORR or local-pH-dependent kinetic switch." Is that one phase or
   two? Concretely:

   * Adding alkaline-form reactions (e.g. R4e_alk: O₂ + 2H₂O + 4e⁻ →
     4OH⁻) is a new BV reaction with new k0, alpha, E_eq. That fits
     the existing reaction list. New species (OH⁻ explicitly) would
     reuse the Phase 6α water-ionization closure.
   * Switching α and n_e on local pH is a more invasive change to the
     log-rate residual.

   Which is the smaller change set? Can the local-pH switch be done
   as *two parallel acid + alkaline reactions* with no explicit
   switch (let Newton pick the dominant one based on which is faster
   at each surface pH)?

10. **`c_H_ref` convention and the rate-collapse question.** Confirmed
    `c_ref_nondim = C_HP_HAT` (bulk pH 4) for both R2e and R4e
    cathodic factors, with powers 2 and 4. At surface pH 10.6, factor
    = `(10^(−6.6))^4 ≈ 4·10^(−27)` for R4e; balanced against
    `exp(α·n_e·|η|) ≈ 4·10^54` at V_RHE = −0.4 V to give a finite
    cd ≈ −0.74 mA/cm².

    If sulfate brings surface pH from 10.6 to 6, factor becomes
    `(10^(−2))^4 = 10^(−8)`, an *amplification* of ~10^19 over the
    current rate. The BV exponential is unchanged. So the predicted
    cd at V_RHE = −0.4 V would be ~10^19 times the current value —
    unphysical. The spike would have to also model the c_O2 depletion
    that this amplification would cause; otherwise the prediction is
    nonsense.

    **Question:** does this immediately imply the spike must compute a
    self-consistent c_O2 surface concentration under the amplified
    cathodic rate (i.e. solve a 1D O2 diffusion problem at each
    voltage), or can the post-hoc closure get the *sign* of the shape
    change without that?

---

## 4. Critique prompt

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
