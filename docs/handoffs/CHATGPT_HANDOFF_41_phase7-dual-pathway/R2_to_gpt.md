# Round 2 counterreply — phase7-dual-pathway

## Section 1: Acknowledgments

**Re point 1 (impossible 10× test threshold):** **Accept.** You're right: 10× the 15.4 µm H⁺ cap = 5.8 mA/cm² ≥ the 4e O₂ ceiling 5.71. New gate: `|cd| > 3 × H⁺ cap (= 1.74 mA/cm²)` AND `|cd| ≤ 1.05 × O₂ 4e ceiling (= 5.99 mA/cm²)`. The upper bound doubles as a sanity check that the water route doesn't manufacture O₂-free current.

**Re point 2 (species-δ ≤10% claim false for ions):** **Accept.** With repo constants D_H⁺ = 9.31e-9, D_O₂ = 1.9e-9, D_OH⁻ ≈ 5.3e-9 m²/s: δ_H/δ_O₂ = (4.9)^(1/3) ≈ 1.70, δ_OH/δ_O₂ ≈ 1.41. Risk R7 is rewritten: the single O₂-based film **overestimates H⁺ supply ~1.7× and OH⁻ removal ~1.4×** (partially canceling in pH, but biasing both the excursion onset and magnitude). Mitigation upgraded from "document" to: (a) honest statement in paper limitations; (b) cheap bracketing sensitivity in Phase 6 — rerun best-fit θ at L_eff ∈ {15.4, 21.7, 26.2} µm (OH-based, H-based equivalents) and report the spread of peak position/plateau.

**Re point 3 (bisulfate buffering absent):** **Accept the obligation to quantify; defend the v11 scope cut — with your requested flux-scale estimate done.** At pH 4 with 0.1 M total sulfate: [HSO₄⁻]/[SO₄²⁻] = 10^(pKa₂−pH) = 10^(1.99−4) ≈ 0.0098 → [HSO₄⁻] ≈ 1.0 mol/m³ (10× free H⁺). Levich-style ceiling on bisulfate proton delivery: F·D_HSO₄·c/δ ≈ 96485·1.33e-9·1.0/15.4e-6 ≈ **0.83 mA/cm²**. So: NOT negligible at the volcano-onset scale (peak −0.40), but small vs deck total 3.25, and Ruggiero's measured pH 4→8–9 excursion proves the real buffer is overwhelmed at operating currents. v11 decision: (i) the omission is promoted to an explicit Med risk + paper limitation with the 0.83 number; (ii) Phase 3 diagnostics gain a per-V comparison of modeled proton-equivalent boundary flux vs the 0.83 ceiling, flagging the V-range where the omission can matter (expected: onset/low-current region only); (iii) a proper HSO₄⁻ fast-equilibrium shadow-species closure (same algebraic pattern as the Kw closure, c_HSO₄ = c_SO₄·c_H/Ka₂ folded into an extended proton condition E = c_H + c_HSO₄ − c_OH) is specced as future scope — it fits the existing machinery but is a new closure + continuation surface we will not open mid-v11. If the fitted model systematically misses the onset region, this is the first suspect, and the paper will say so.

**Re point 4 (E_eq thermodynamic language overstated):** **Accept.** The water branches as specified (no product-activity factors, irreversible) are empirical cathodic Tafel branches, not a thermodynamically consistent alkaline BV law. Language changes everywhere (plan + paper): E_eq for water routes is a **formal onset/reference potential**, reused at 0.695/1.23 V for η-bookkeeping continuity; k0_water/α_water are empirical. A product-consistent alkaline BV form (c_OH, c_H₂O₂ factors in the anodic branch) is listed as future refinement. No code change.

**Re point 5 (observable layer misses the sum):** **Accept.** Instead of per-driver workarounds: extend `_build_bv_observable_form` with a `reaction_indices: list[int]` parameter (mode `"reaction_sum"`), unit-tested, and drivers/fit use it for pc = R2e_acid + R2e_water. This is an observables.py edit (not weak form) — the plan's "zero weak-form edits" claim stands but the "pure config" framing is corrected to "config + observable-layer".

**Re point 6 (per-reaction current 2× error for 4e):** **Accept — verified in code.** `mode="reaction"` returns unweighted `scale·R_j`; I_SCALE is anchored to n_e=2 (`N_ELECTRONS_REF`), and only `mode="current_density"` applies the n_e/2 weighting. Per-reaction outputs will carry BOTH the raw rate (2e-current units, existing convention) and the electron-weighted current `(n_e_j/2)·(−I_SCALE)·R_j`, with a test asserting Σ_j weighted = cd to tolerance. Driver JSON schema gains `per_reaction: [{label, n_e, rate_2e_units_mA_cm2, current_mA_cm2}]`.

**Re point 7 (F2 validation gate vs old I_lim):** **Accept, narrowed by verification.** The slide-15 demo drivers assemble raw forms (no `assemble_observable_validated` call), so Phase 0/3 runs aren't at risk. But the scoring/fit paths (Phase 5) and any pipeline reuse must pass route-aware I_lim: O₂-4e ceiling for cd, O₂-2e ceiling for pc — never an H⁺-based limit. These constants go in `calibration/phase7_wls.py` with a test; the plan gains an explicit line item.

**Re point 8 (reversible acid R2e consumes H₂O₂):** **Clarify + partial defend.** Microscopic reversibility requires the anodic branch near E_eq, and the anchor sits at the anodic end where it stabilizes the cold start — removing reversibility from acid R2e is a production-convention change with its own risk and no demonstrated need. Quantitatively: across the deck window η_2e ≤ −0.145 V ⇒ the anodic factor exp((1−α)·n·η/V_T) ≤ exp(−4.2) ≈ 1.5e-2, multiplied by surface c_H₂O₂ (small) — expected ≪1% of the cathodic branch. **Accept the instrumentation, not the surgery:** per-2e-channel outputs will expose cathodic-only, anodic-only, and net; the deck-comparison pc is **net production flux** (what escapes toward the ring); a diagnostic asserts anodic share < 1% in-window and flags if violated (which would force the topology discussion you're pointing at).

**Re point 9 (volcano may be Tafel competition, not local pH):** **Accept — this is the best catch in the review.** Both water routes being pH-independent means the cathodic flank could be set by (α_w2e·2 vs α_w4e·4) Tafel competition with no pH involvement. The plan's mechanism narrative is downgraded from a claim to three candidate flank mechanisms: (M1) acid-route starvation via local-pH excursion, (M2) water-route Tafel-slope competition, (M3) hydrolysis-buffer modulation of M1. Phase 6 gains the discriminating ablations: water-routes-only run (volcano persists ⇒ M2 sufficient), water-4e-off (flank vanishes ⇒ 4e competition is the flank), bulk-pH 7 vs 4 (M1 sensitivity), hydrolysis on/off (M3). Paper reports the attribution that survives, whichever it is.

**Re point 10 (hydrolysis continuation under-specified):** **Accept.** Stage-2 sequence is now explicit: (i) anchor with kw ladder at λ=0, C_S=0.10; (ii) manual Stern bump 0.10→0.20 + re-solve; (iii) λ ramp 0→1 as a separate per-V outer ladder (the proven A.2-driver pattern), warm-restarting between rungs; never two ladders in one continuation call. Kw + hydrolysis + Stern-bump simultaneously is flagged as untested territory in R1 of the risk register (Phase 6β ran hydrolysis with water ionization OFF).

**Re point 11 (R3 E_eq double-shift risk):** **Accept.** Convention codified: presets store unshifted E° vs RHE only; the OCP shift is applied in exactly one place (driver config assembly); new fast test asserts η-preservation `(V_solver − E°_solver) = (V_deck − E°_RHE)` for every reaction including R3, to 1e-12.

**Re point 12 (fit underdetermined):** **Partial accept.** The objective gains two soft constraint terms: (a) local-pH hinge penalty — pH_surf at |j| ≈ 3 mA/cm² must land in [7, 10] (Ruggiero band, generous); (b) total-current hinge — |cd| at the far-cathodic end within [1.5, 5.99] mA/cm². Step 1 is extended: while re-digitizing slide 15, check the deck for a digitizable **total disk current** panel at the same condition; if present, it becomes a second WLS series with its own σ (the strongest fix for partition non-identifiability). Identifiability reporting: final-simplex parameter spread + 1D profile slices from the checkpointed evals. **Defend the scope cut** on full profile-likelihood/Bayesian treatment — out of v11; the paper claims a calibrated demonstration, not a posterior.

**Re point 13 (k0-only sweep can falsely reject):** **Accept.** Before any "shape qualitatively wrong" verdict, the sweep stage runs an α mini-probe at the best k0 corner: α_w4e ∈ {0.35, 0.50, 0.65} (2 extra runs; α_w2e held), since the flank position is most sensitive to the 4e Tafel slope. Only after the probe can Phase 5a return a negative verdict.

**Re point 14 (digitization σ ≠ experimental uncertainty; correlated points):** **Accept.** Loss semantics restated: this is a **rendered-curve reproduction metric**, not a likelihood against experimental replicates. Concretely: (a) if the plot has discrete markers, marker centers become the fit points (preferred over trace); (b) trace-derived points are thinned/block-averaged to ≤ 40 effective points to cut correlation; (c) line-thickness σ is kept but labeled digitization-σ; (d) χ² values are used relatively (model/ablation comparison), never quoted as absolute goodness-of-fit. Plan Step 1 and Phase 5 text updated.

**Re point 15 (validation too late/too narrow):** **Partial defend, partial accept.** Defend the placement: `dispatch.py::build_forms` is the single shared entry for BOTH backends (logc and logc_muh both build through the dispatcher), so the cross-flag check there is not narrow — and it's the earliest point where reactions-cfg and convergence-cfg coexist. Accept the rest: everything checkable at parse time moves to parse time (water+conc-factors, water+reversible in `_get_bv_reactions_cfg`), and the test matrix covers both backends plus the failure paths (water route with water-ionization OFF must raise, on both formulations).

**Re point 16 (no global OH/E balance diagnostic):** **Accept.** Phase 3 gains a per-V **route ledger**: per-reaction boundary mol-flux integrals, acid vs water share of the E-sink, bulk-boundary E-flux, and the steady-state closure residual |Σ boundary E-sources − bulk E-outflux| (must be ≈ 0). Emitted into iv_curve.json; a test checks the closure residual on a converged coarse solve. This is also the diagnostic that backs Risk R2 (Kw-optimism absorbing into k0_water): the acid/water share is now measured, not inferred.

## Section 2: Updated artifact

Changes applied to the plan (full revision lands at loop end; deltas listed here):
1. Phase 2 cap-escape gate → `> 3× H⁺ cap` AND `≤ 1.05× O₂-4e ceiling`.
2. Risk R7 rewritten with δ_H/δ_O₂ = 1.70, δ_OH/δ_O₂ = 1.41; added L_eff ∈ {15.4, 21.7, 26.2} bracketing runs to Phase 6.
3. New Med risk + Phase 3 diagnostic + paper limitation: bisulfate buffer omission (ceiling ≈ 0.83 mA/cm² at pH 4, 0.1 M sulfate); future-scope HSO₄⁻ shadow-species closure specced (extended proton condition E = c_H + c_HSO₄ − c_OH).
4. All "thermodynamic path-independence" language for water-route E_eq replaced with "formal onset/reference potential; empirical Tafel branch".
5. Phase 1b: new `reaction_indices`/`"reaction_sum"` observable mode in observables.py (+tests); "pure config" framing corrected to "config + observable layer".
6. Per-reaction outputs carry raw rate AND electron-weighted current; test Σ_j weighted = cd.
7. Phase 5: route-aware I_lim constants (O₂-4e for cd, O₂-2e for pc) in phase7_wls.py (+test).
8. Per-2e-channel cathodic/anodic/net decomposition + anodic-share<1% diagnostic.
9. Mechanism narrative → three candidate flank mechanisms (M1 pH-starvation / M2 Tafel competition / M3 hydrolysis modulation); Phase 6 matrix extended with water-only, water-4e-off, bulk-pH-7 ablations.
10. Stage-2 continuation sequence (kw@λ=0 → Stern bump → separate λ ramp) explicit; Kw+hydrolysis+Stern flagged untested in R1.
11. E° stored unshifted; OCP shift applied once; η-preservation fast test incl. R3.
12. WLS objective + soft hinges (local-pH band, total-current band); Step 1 extended to hunt a digitizable total-current panel; identifiability via simplex spread + profile slices.
13. Phase 5a α_w4e mini-probe {0.35, 0.50, 0.65} required before any negative shape verdict.
14. Step 1/Phase 5 loss semantics: rendered-curve reproduction; markers preferred; ≤40 effective points; relative χ² only.
15. Parse-time validation maximized; cross-flag stays in dispatch (shared by both backends); failure-path tests for both formulations.
16. Phase 3 route ledger (per-reaction boundary integrals, acid/water E-sink share, closure residual) + test.

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
