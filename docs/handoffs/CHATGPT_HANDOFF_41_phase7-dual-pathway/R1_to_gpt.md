# Handoff 41 — Adversarial review of the Phase 7 / v11 plan: dual-pathway (water-as-proton-donor) ORR kinetics to fit the slide-15 H₂O₂ volcano

## Section 1: Context bundle

### The system and the code

Research code (Firedrake FEM, Python) for steady-state Poisson–Nernst–Planck + Butler–Volmer simulation of ORR (O₂ → H₂O₂ → H₂O) on a CMK-3 carbon disk in K₂SO₄/Cs₂SO₄ at pH 4, modeling the Seitz/Mangan group's RRDE experiments (source paper: Ruggiero et al. 2022, J. Catal., parallel 2e⁻/4e⁻ ORR, 1600 rpm, I = 0.3 M).

Production stack:
- **3 dynamic species**: O₂ (z=0), H₂O₂ (z=0), H⁺ (z=+1), each a Nernst–Planck PDE on a 2D (x,y) graded mesh; y is distance from electrode, bulk Dirichlet at y = L_eff.
- **Formulation `logc_muh`**: O₂/H₂O₂ solved in u = ln(c); H⁺ solved via electrochemical potential μ_H = u_H + z_H·φ (smooth through the Debye layer).
- **Analytic Bikerman counterions** (Cs⁺ or K⁺, SO₄²⁻): Boltzmann-in-φ closures with steric (Bikerman) saturation entering Poisson's source and a steric drift term in transport.
- **Stern layer**: Robin BC on φ at the electrode, C_S = 0.20 F/m².
- **Butler–Volmer boundary reactions** at the electrode (log-rate form). Current production set `PARALLEL_2E_4E_REACTIONS`:
  - R2e: O₂ + 2H⁺ + 2e⁻ ⇌ H₂O₂, E° = 0.695 V vs RHE, stoichiometry [−1, +1, −2] on (O₂, H₂O₂, H⁺), reversible. Cathodic log-rate: ln k₀ + u_O₂ + 2·(u_H − ln c_ref) − α·n_e·η. So rate ∝ c_O₂·c_H²·exp(−α n η).
  - R4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O, E° = 1.23 V vs RHE, stoichiometry [−1, 0, −4], irreversible, rate ∝ c_O₂·c_H⁴·exp(−α n η).
  - η_j = (V_applied − φ_surface) − E°_j in thermal-voltage units, clipped at ±100 (clip applied to (V_RHE − E_eq)/V_T before α·n_e multiplication).
  - The c_H^m factors enter ONLY via a per-reaction config list `cathodic_conc_factors` (verified: `forms_logc_muh.py:557-563`). The species-stoichiometry boundary terms enter via a loop `F_res -= stoi[i]·R_j·v_i·ds(electrode)` (`forms_logc_muh.py:612-615`).
- **Water self-ionization module (opt-in, "Phase 6α")**: when `enable_water_ionization=True`, the H⁺ NP residual is replaced by a proton-condition residual on E = c_H − c_OH, with OH⁻ as a *shadow species* closed by fast equilibrium c_OH = Kw_eff·exp(−u_H) (i.e., c_H·c_OH = Kw_eff pointwise). OH⁻ contributes to Poisson's source, to an effective E-flux (−(D_H c_H + D_OH c_OH)∇μ + steric cross-term), and to Bikerman packing. Continuation: `kw_eff_ladder` ramps Kw from 0 to its physical value.
- **Cation hydrolysis module (opt-in, "Phase 6β")**: boundary source/sink pair modeling M⁺ + H₂O ⇌ MOH⁰ + H⁺ at the outer Helmholtz plane with a field-dependent pKa (Singh 2016 Eq. 3/4: ΔpKa ∝ β·σ_Stern), a Langmuir-capped surface reservoir Γ (Γ_max = 0.047 nondim, one MOH monolayer), desorption k_des = 1.0 nondim. Activation scalar λ ∈ [0,1] ramped by a ladder. This is the group's hypothesized cation-specific local-pH buffer.
- **Continuation infrastructure**: `solve_anchor_with_continuation` (k0-ladder cold start at an anchor voltage) → `solve_grid_with_anchor` (warm walk across the V grid with substeps + bisection). `AdaptiveLadder` with midpoint insertion. Known limitation: kw_eff ladder and C_S ladder cannot be combined (NotImplementedError); the working pattern is anchor at C_S=0.10 then a manual "Stern bump" to 0.20 via coefficient reassignment + re-solve.

### The target data

Deck "slide 15": H₂O₂ production current density vs applied V (V_RHE), Cs₂SO₄ pH 4, RRDE 1600 rpm, collection-efficiency-derived. Eyeballed digitization (to be replaced by a precision pixel re-extraction as Step 1 of the plan): left plateau ≈ −0.18 mA/cm² for V ≲ −0.3 V, volcano peak ≈ −0.40 mA/cm² at V ≈ +0.10 V, decays to ≈ 0 by V ≈ +0.45 V. Cathodic = negative. Deck total disk current at pH 4 reaches ≈ 3.25 mA/cm² (from the paper's local-pH figure); Levich ceilings at 1600 rpm: 2e ≈ 2.85, 4e ≈ 5.7 mA/cm².

### Established diagnostic facts (verified this session — do not re-litigate the numbers, but DO challenge interpretations)

1. With the current acid-route-only kinetics and water ionization OFF, the model's TOTAL current is pinned at the H⁺ Levich cap: −0.0898 mA/cm² at pH 4, L_eff = 100 µm, across the entire deck V window (verified in two independent 25-point study runs, including an OCP-shifted variant). Both R2e and R4e rates carry c_H^m factors, so once surface H⁺ depletes, total current saturates at F·D_H·c_H,bulk/L_eff. The H₂O₂ peak alone in the deck (−0.40) is ~4.5× this cap; deck total is ~36×.
2. Phase 6α results (water ionization ON, acid rate laws unchanged, L_eff sweep): cd plateau −0.737 mA/cm² at L=100 µm, −4.65 at L=16 µm (Levich 1/L scaling preserved, the limiting supply became OH⁻ removal); surface pH self-pins at ≈ 10.58 independent of L_eff; a peroxide-current volcano APPEARS but peaks at V ≈ −0.20 V — ~0.30 V more cathodic than the deck's +0.10 V. Documented root cause: at surface pH 10.6, the c_H⁴ factor punishes R4e ~16 orders of magnitude more than at deck-like pH ~6, so the 2e→4e branching transition (which sets the volcano peak) happens at far more cathodic V than it should.
3. A separate documented critique of the Phase 6α closure: the Kw fast-equilibrium closure effectively manufactures H⁺ at the surface ~20× faster than the real water-dissociation forward rate would allow (fast-equilibrium is an upper bound on H⁺ supply from water splitting).
4. Ruggiero 2022's central experimental claim: local pH (measured via IrOx ring) at bulk pH 4 rises ~linearly with disk current, reaching ~8–9 at 3.25 mA/cm²; cation identity modulates this excursion (buffering); the reaction topology is parallel 2e/4e on carbon with NO surface consumption of free H₂O₂ (4e goes *-OOH → *-OH → H₂O without free peroxide; once H₂O₂ leaves, nothing on the surface consumes it).
5. OCP convention: the deck and the prior thesis both reference ψ_bulk = V_OCP with V_OCP_RHE = 0.47 + 0.197 + 0.059·pH (= 0.903 V at pH 4). Our solver uses ψ_bulk = 0, V_M = V_RHE directly. The prior successful thesis-reproduction runs applied a uniform shift: V_solver = V_deck − V_OCP and E°_solver = E°_RHE − V_OCP for BOTH reactions, which preserves η = V − E° exactly while relocating the diffuse-layer "rest point" (φ_DL = 0 at deck OCP). The plan adopts this convention for all deck comparisons.
6. A long fit campaign (Phase 6β, "Phase D") against a *different* scalar target (deck K@pH4 max-H₂O₂-selectivity = 50.95 pp) returned OUTCOME_C_NON_IDENTIFIABLE: the fitted Δ_β (Singh pKa offset) had literally zero leverage (loss flat to 6 decimals over 11 orders of magnitude of Δ_β; Stern σ at the OHP ~1e-7 counts/pm² is too small for the β·σ pathway to matter), and model max selectivity sat at 66.58% vs target 50.95% uniformly. Our retrospective interpretation: every knob was tuning details on top of a current hard-capped by H⁺ transport — there was no mechanism freedom to exploit.

### Why the dual-pathway plan (the core physics proposal under review)

At pH 4 unbuffered, the experimental currents (≈3 mA/cm²) vastly exceed any possible bulk-H⁺ supply (≈0.09–0.58 mA/cm² depending on film thickness). The reaction must proceed with water as the proton donor, generating OH⁻ (equivalently: alkaline-route ORR), which is consistent with the measured local alkalinization. The plan adds **water-route variants of both reactions**:

- R2e_water: O₂ + 2H₂O + 2e⁻ → H₂O₂ + 2OH⁻, cathodic rate ∝ c_O₂·exp(−α n η) — NO c_H factor (water activity ≡ 1), irreversible (first cut).
- R4e_water: O₂ + 2H₂O + 4e⁻ → 4OH⁻ (net 4e with water donor), cathodic rate ∝ c_O₂·exp(−α n η), irreversible.

Key implementation claim (verified in code, and central to the plan's "zero weak-form edits" assertion): with the proton-condition variable E = c_H − c_OH active, producing 2 OH⁻ per event is algebraically identical to consuming 2 H⁺ per event — both decrement E by 2. The existing stoichiometry vector [−1, +1, −2] (and [−1, 0, −4]) routed onto the E-residual therefore covers the water route with no new plumbing; only the rate law differs (empty `cathodic_conc_factors`). A new per-reaction config field `proton_donor: "hydronium" | "water"` becomes schema/validation/provenance only. Water routes require `enable_water_ionization=True` (else OH⁻ production is unrepresentable and the same stoichiometry would wrongly drain a finite c_H), enforced at config validation.

E° for the water routes stays at the acid-route values (0.695 / 1.23 V vs RHE). Plan's argument: thermodynamics is path-independent — O₂/H₂O₂ and O₂/H₂O couples have one equilibrium potential each at given local activities; on the RHE scale at the *reference* pH these are 0.695/1.23 V; acid and water routes are kinetic pathways of the same net half-reactions (at pH 4: 2H₂O → H₂O₂ + 2H⁺ + 2e⁻ reversed, etc.), so they share E_eq. The α and k₀ of the water branches are free kinetic parameters (to be fit).

Mechanistic story for the volcano (to be tested, not assumed): anodic flank = R2e kinetic onset (η_2e); cathodic flank = local pH excursion driving 2e→4e branching shift (acid 2e starves as c_H falls; water-route 4e takes over). Cation hydrolysis buffering then modulates the local-pH excursion per cation (Cs vs K), which is the deck's cation story. A sequential H₂O₂-reduction reaction (R3) is implemented as a *falsifiable alternative* for the cathodic flank and expected to be rejected (Ruggiero topology says no surface H₂O₂ consumption).

### Constraints and conventions (already decided; challenge only if you see a real inconsistency)

- Solver: η uses absolute potentials; exponent_clip = 100; u_clamp = 100; these are locked production conventions with documented histories.
- Repo invariant: any new flag must be default-off byte-equivalent (residual-level) to the pre-change solver.
- Fit strategy (user-decided): manual coarse sweeps over k0_water factors FIRST (~6–10 runs); a scipy WLS fit (4 params: log10 k0_R2e_water, log10 k0_R4e_water, α_R2e_water, α_R4e_water; acid params frozen) only if the sweep lands the volcano in the right neighborhood.
- The 37-pt eyeball digitization gets replaced by a precision pixel-level re-extraction (with per-point σ from curve line thickness + calibration residual) BEFORE any fitting.
- L_eff for deck comparison changes from 100 µm to 15.4 µm (O₂ Levich-equivalent at 1600 rpm; single-film approximation, species-δ differences ∝ D^(1/3) ≤ 10%, documented).

## Section 2: The artifact under review

The full plan follows verbatim.

---

# Phase 7 / v11 — Dual-Pathway (Water-Donor) ORR: Fit the Slide-15 H₂O₂ Volcano

## Context — why this change

**Target:** deck slide-15 data — H₂O₂ production current density vs V_RHE, Cs₂SO₄ pH 4, RRDE 1600 rpm. Current 37-pt eyeball at `data/mangan_deck_p15_h2o2_current.csv` (left plateau −0.18 mA/cm², volcano peak −0.40 at V ≈ +0.10, zero by +0.45 V) is a **low-confidence hand digitization — it gets replaced by a precision re-extraction (Step 1) before anything is fit to it.** Ultimate goal: paper on model construction + chemical mechanism.

**Diagnosis (verified this session):**
1. Production stack (acid-route 2e/4e BV, rates ∝ c_O₂·c_H², c_O₂·c_H⁴; `enable_water_ionization=False`) pins **total** current at the H⁺ Levich cap −0.0898 mA/cm² (pH 4, L_eff=100 µm) — 5× below the deck H₂O₂ peak alone, ~36× below deck total disk current (~3.25 mA/cm²). Monotonic pc, no volcano (verified `StudyResults/solver_demo_slide15_no_speculative_cs/`, `..._ocp_shifted_cs/`). **This retroactively explains every flat-loss fit failure (Phase D Δ_β, λ sweeps, K0_R4e sweeps): all knobs tuned a current hard-capped by transport.**
2. Ruggiero 2022 (source paper): local pH story — bulk pH 4 swings to ~8–9 under ~3 mA/cm²; cation identity modulates local pH (buffering); parallel 2e/4e with **no surface consumption of free H₂O₂**. Levich-equivalent film at 1600 rpm: δ ≈ **15.4 µm** (our L_eff=100 µm is wrong for deck comparison; 2e/4e ceilings 2.85/5.7 mA/cm²).
3. Phase 6α already proved the escape route works: water ionization ON → cd −0.737 (L=100 µm) / −4.65 (L=16 µm), **PC volcano appears but ~0.3 V too cathodic** — because the c_H⁴ penalty on R4e at surface pH 10.58 is ~16 OOM (`docs/phase6/PHASE_6A_INVESTIGATION_SUMMARY.md`). Missing physics: **water-as-proton-donor (alkaline-route) kinetics** — O₂ + 2H₂O + 2e⁻ → H₂O₂ + 2OH⁻ (and 4e analog) with rate NOT gated by c_H. A water-route R4e is the lever that moves the peak anodically toward +0.10 V.
4. Cation hydrolysis (Singh pKa, Γ/k_des — existing machinery, v10b) gets its proper role: the **cation-specific buffer of the local-pH excursion**, not a standalone selectivity knob.

**Key implementation insight (verified in code):** the c_H^m factor enters the BV rate ONLY via `rxn["cathodic_conc_factors"]` (`forms_logc_muh.py:557-563`); the stoichiometry loop (`:612-615`) already routes "−2 per event" onto the proton-condition residual E = c_H − c_OH, which is algebraically identical to producing 2 OH⁻. **A water-route reaction is pure config — zero weak-form edits.** `proton_donor` becomes a schema/validation/provenance field. Caveat: `_get_bv_reactions_cfg` (`config.py:452-536`) rebuilds dicts from an explicit whitelist — the new key must be added there or it is silently dropped. (Also noted: `"enabled"` never survives this parser; ablations disable via `k0=0` or list omission.)

## Execution order

### Step −1 — GPT critique loop on this plan (user-approved)
Run `/gpt-critique-loop` on this plan before coding. [This handoff is that step.]

### Step 0 — Checkpoint commit
Commit in-flight work as a `chore: checkpoint` commit so Phase 7 diffs stay clean. Fix a stale CLAUDE.md hard-rule note.

### Step 1 — Precision re-digitization of the slide-15 target (before any fitting; ~0.5 day)
1. **Source hunt:** extract embedded images at native resolution from candidate original decks (pptx = zip → `ppt/media/*`) in the experimental-data folder; compare to the existing 251 KB screenshot; pick the best; record provenance.
2. **Visual structure pass (Claude reads the image directly):** confirm axis ranges/ticks, curve color vs gridlines, markers/error bars/multiple series — anything the v1 eyeball missed.
3. **Extraction script** `scripts/studies/_digitize_mangan_slide15_v2.py` (PIL + numpy): color-segment the curve, per-x-column median y → dense trace; pixel→data affine from tick calibration; per-point σ from local line thickness + calibration residual; downsample to ~60–100 pts.
4. **QA loop:** re-render extraction overlaid on the original (pixel space) → visual verification; v1-vs-v2 comparison quantifying the old eyeball's error.
5. **Output:** `data/mangan_deck_p15_h2o2_current_v2.csv` (v1 kept for provenance) with full metadata. **All downstream WLS uses v2 + per-point σ.**
Gate: overlay faithful; calibration residual ≲ 1% of axis span; σ documented. Error bars / multiple series, if present, change the loss definition — capture them.

### Phase 0 — Cheap reconnaissance (no new physics, ~1 h compute)
- **0a:** `--l-eff-um` CLI flag on the existing OCP-shifted demo driver; run at 15.4 µm. Gate: ≥20/25 converge; plateau ≈ −0.58 mA/cm² (H⁺ cap × 100/15.4). Documents transport correction alone is insufficient.
- **0b:** `--enable-water-ionization` flag + 5-rung kw_eff ladder in the anchor (pattern from the proven Phase 6α sweep driver); anchor at C_S=0.10 + manual Stern bump to 0.20. Run at L=100 and 15.4 µm. Gate: reproduces 6α behavior (|cd| ≫ 0.09; PC volcano exists ~0.3 V too cathodic). **This is the baseline the water route must beat on peak position.**

### Phase 1 — Config schema + reaction preset + fast tests (no Firedrake, ~1 day)
- **1a:** parse `proton_donor` in `_get_bv_reactions_cfg` (default "hydronium"; validate ∈ {hydronium, water}; include in output dict). Validation: water route forbids `cathodic_conc_factors` (water activity ≡ 1, documented v11 approximation) and requires `reversible=False` (anodic branch lacks concentration-factor support — a reversible water route would be thermodynamically wrong). Cross-flag check (water requires `enable_water_ionization=True`) in `dispatch.py::build_forms`.
- **1b:** `PARALLEL_2E_4E_DUAL_PATHWAY` preset (R2e_acid [index 0, byte-equal to existing + key], R2e_water [stoich [−1,+1,−2], n_e=2, irreversible, no conc factors], R4e_acid, R4e_water [stoich [−1,0,−4], n_e=4]). New constants K0_HAT_R2E_WATER etc. (init = acid values). E_eq stays 0.695/1.23 vs RHE for water routes. **Gross PC observable must now sum reaction indices 0 AND 1** (both 2e channels).
- **1c:** fast tests: parsing round-trip (whitelist regression), validation rules, preset shape, default-off equivalence of parsed dicts.

### Phase 2 — Slow Firedrake tests (~0.5 day)
- `TestWaterRouteRateLaw`: perturb u_H → water-route boundary rate invariant; acid-route scales ~c_H².
- `TestDefaultOffResidualNorm`: residual L²-norm with defaulted legacy reactions matches pre-change golden.
- `TestWaterRouteEscapesLevichCap`: water ionization ON, acid routes k0=0, water routes only, L=15.4 µm, strongly cathodic V; assert |cd| > 10× the 15.4 µm H⁺ cap — finite current with zero H⁺-supply dependence.

### Phase 3 — Dual-pathway driver + local-pH diagnostics (~1 day)
- Driver clone with CLI: `--k0-water-2e-factor`, `--k0-water-4e-factor`, `--alpha-water-2e`, `--alpha-water-4e`, `--routes`, `--enable-hydrolysis`, `--coarse-grid`. Output per-reaction currents (4 entries), pc = R0+R1.
- `surface_ph(ctx)` diagnostic + pOH consistency check; `local_ph_vs_v.json`. **Independent validation target (paper figure): Ruggiero Fig 1B — local pH ≈ linear in |j|, reaching 8–9 by 3.25 mA/cm².**
Gate: default run converges ≥20/25; PC non-monotonic; local-pH-vs-j roughly linear.

### Phase 4 — Robustness hardening (1–3 days; schedule risk center)
1. Anchor with 4 reactions: joint k0 ladder, floor → 1e-15; if exhausted: anchor with acid routes at production k0 + water routes at 1e-12× floor, then ramp water k0 post-anchor via coefficient reassignment + re-solve (same pattern as the Stern bump). Promote to a `k0_subset_ladder` kwarg only if needed in ≥2 drivers.
2. Alkaline surface blow-up (u_H ≪ 0 → c_OH = Kw_eff·exp(−u_H) huge → OH Bikerman saturation/Poisson stiffening): watch packing_floor hits; finer kw rungs; smaller linesearch maxlambda; last-resort diagnostic-only a_OH bump.
3. Thin-domain mesh: first-cell ≈30 nm at H_hat=0.154 — finer near wall than before; Ny=120 check only if needed.
Gate: dual-pathway driver ≥23/25 at L=15.4, both routes production k0, twice in a row, no manual intervention.

### Phase 5 — Coarse sweeps first, then optional optimizer (user decision: sweeps first)
- **5a (2–3 h):** ~6–10 runs over k0_water_2e/4e factors on a 13-pt coarse grid, OCP-shifted, L=15.4. Score vs the v2 re-digitized CSV with per-point σ (WLS objective in Firedrake-free `calibration/phase7_wls.py` + fast tests: σ handling, PCHIP interpolation onto data V's, non-converged points dropped not zero-filled).
  Decision gate: volcano in right neighborhood (left plateau within ±50%, peak position within ±0.15 V) → 5b. Shape qualitatively wrong at every corner → Phase 6 ablation diagnosis instead of optimizer compute.
- **5b (overnight, only after 5a passes):** scipy Nelder-Mead / differential_evolution, eval-capped, 4 params (log10 k0_R2e_water, log10 k0_R4e_water, α_R2e_water, α_R4e_water), acid params frozen; per-eval JSON checkpoints; anchor per-θ; final θ on fine 25-pt grid.
  Stage 2 (only if residuals show buffering-shaped misfit): add k_hyd/λ.
  Gate: left plateau ±30% of −0.18; peak position ±0.10 V of +0.10; zero-crossing ±0.10 V of +0.45.

### Phase 6 — Mechanism attribution / paper arm (~1 day)
Ablation matrix at best-fit θ: water route (k0=0/fitted); water ionization; cation hydrolysis (λ=0/1); **R3 sequential peroxide reduction** as falsifiable alternative for the cathodic flank — pure config 5th reaction {cathodic_species: H₂O₂, stoich [0,−1,−2], n_e=2, irreversible, proton_donor: water, E_eq 1.76 V OCP-shifted}; steric on/off; OCP shift on/off.
Decision rule: if R3 is required for the cathodic decline at k0 values that wreck the left plateau, reject R3 (contradicts Ruggiero topology); if water-route 2e/4e competition alone reproduces it, accept parallel topology. Optional out-of-sample: digitize deck K⁺ panel, swap Cs⁺→K⁺ + K⁺ Singh pKa (8.49 vs 4.32), predict with no refit. Second independent validation: local-pH-vs-j linearity.

## Risk register
| # | Risk | L | Mitigation |
|---|---|---|---|
| R1 | Anchor/grid divergence (4 rxns + Kw + thin domain) | High | Phase 4 sequence; manual k0 bump; rest-V anchor |
| R2 | Kw closure ~20× optimistic → k0_water absorbs unphysical H⁺ supply | Med | Report acid-route current share at fitted θ; if >30% cathodic of peak, flag finite-rate water dissociation as future scope |
| R3 | Volcano needs H₂O₂ consumption after all | Med | Phase 6 matrix detects exactly this; paper reports either outcome |
| R4 | Picard IC drops to "general" topology for 4 rxns | Med | linear_phi fallback automatic; extend matcher if persistent |
| R5 | Fit wall-clock blowout | Med | Sweeps-first gate; coarse grid; checkpointing; eval cap |
| R6 | Byte-equivalence regression via parser | Low | Zero forms edits; round-trip fast tests + residual-norm golden |
| R7 | Single-δ film vs species-dependent Levich δ (∝D^{1/3}) | Low | O₂-based 15.4 µm; document; others differ ≤10% |

## Conventions locked for this work
- OCP shift **ON** for all deck comparisons (V grid AND both E° shifted by −0.903 V at pH 4; η preserved). Caveat: the 0.47 V component is only documented in a collaborator's notebook — flagged for confirmation before any production recalibration claim in the paper.
- L_eff = 15.4 µm (O₂ Levich-equivalent, 1600 rpm) for deck comparison.
- Cs⁺/SO₄²⁻ stack (slide 15 is Cs₂SO₄); K⁺ only as out-of-sample prediction.
- Model "peroxide current" = gross 2e production (R2e_acid + R2e_water), per Ruggiero topology.

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
