1. WHAT: P2 widens the primary gate to `max(±10 pp, ±std)`. WHY: this directly violates the locked acceptance bundle, which says ±10 pp absolute, not “±10 unless the data are noisy” ([bundle](/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md:61)). DO: keep ±10 pp as the pass/fail gate. Report the 26 pp std as data-quality context or a separate `DATA_TARGET_NOISY` flag, not as a relaxed pass rule.

2. WHAT: The β value in the counterreply is numerically off. With `A=620.32`, `z=0.919`, `r_H=200.98`, `r_M-O=201`, β_K_Cu is about `-45.61 pm²`, not `-45.18`. WHY: small here, but this exact number defines the sign boundary and `T=0` conversion. DO: compute β in one shared helper and use that helper in docs/tests/driver.

3. WHAT: The target-ΔpKa grid includes positive targets `{+0.2,+0.5,+1,+2,+3,+5}`. WHY: positive cathodic ΔpKa violates the locked mechanism sign guard. For K at cathodic σ, `T >= 0` implies `β_K_carbon >= 0`, i.e. `Δ_β >= ~45.6 pm²`, which v2 itself says should falsify. DO: exclude positive total-ΔpKa targets from the optimizer domain. Treat sign-flip demand as a falsification, not as search space.

4. WHAT: The proposed Stern bracket `[-4.67e7,+4.67e7] pm²` contradicts the sign guard `Δ_β < +45 pm²`. WHY: half the bounded optimizer domain is invalid by construction. DO: upper-bound the Stern fit at `< -β_K_Cu` and use a one-sided negative ΔpKa-effect grid.

5. WHAT: `T=0` maps to `Δ_β ≈ +45.6 pm²`, not to the current V10B baseline. WHY: the pre-fit grid no longer includes the scientifically important prior point `Δ_β=0` except as a separate baseline check, so the loss curve can miss the transition from “no effect” to “large imposed pKa effect.” DO: include `Δ_β=0` in the fit grid and add log-spaced negative ΔpKa targets between `~5e-6` and `0.2`.

6. WHAT: The ΔpKa target grid is too coarse over the relevant decades. WHY: it jumps from baseline `~5e-6` to `0.2`, a factor of ~40,000. If the model response saturates at `1e-3` or `1e-2`, the grid will misdiagnose identifiability and bracket. DO: use log spacing in `|ΔpKa|`, e.g. baseline, `1e-5, 3e-5, 1e-4, ... , 5`.

7. WHAT: `Δ_loss = max(L)-min(L)` will be broken by `+inf` losses from sign-invalid or clamp-engaged candidates. WHY: `inf` makes the identifiability gate pass even when all finite points are flat. DO: compute identifiability only over finite, sign-valid, non-clamped evaluations; separately report invalid-domain coverage.

8. WHAT: The slope threshold `0.01 pp²/Δ_β_unit` is meaningless when `Δ_β` can be `1e7 pm²`. WHY: slope magnitude becomes unit-scale dependent and will look artificially tiny. DO: compute sensitivity in target-ΔpKa space or normalized fractional bracket coordinates.

9. WHAT: The proposed `ΔpKa_clip [-15,+15]` changes the locked Singh formula. WHY: clamping inside `_build_singh_2016_eq_4_pka_shift` means the solver evaluates a new model, even if you later assign `+inf` loss. DO: do not clamp the residual. Pre-screen or abort candidates whose predicted/assembled ΔpKa leaves a declared safe domain.

10. WHAT: `loss=+inf` after a clamp-engaged solve is self-contradictory. WHY: if the solve used the clamped residual, the result is not the candidate model; if it is invalid, the solve should not be trusted. DO: reject before solving where possible, or mark `SOLVE_INVALID_DOMAIN` without using the forward result.

11. WHAT: P4’s V grid is still off by one. `[-0.06,-0.01,...,0.99]` is already 22 points, and appending `1.0` makes 23, not 22. WHY: the exact grid feeds wall budget, endpoint tests, and extraction. DO: list all explicit values and count them in a test.

12. WHAT: The new grid still excludes `V_kin=-0.10 V`. WHY: v2 uses `V_kin` σ to map ΔpKa targets and wants byte-equivalence against v10b A.2, but the proposed grid starts `-0.06,-0.01,...`; there is no `-0.10` point. DO: add a dedicated `V_kin=-0.10` baseline/evaluation point or change the grid.

13. WHAT: The adaptive ring-onset refinement says “5 additional points at 0.01 V spacing” inside a 0.05 V bracket. WHY: there are only 4 interior 0.01-spaced points if endpoints already exist. DO: specify whether duplicates are allowed, or say “4 interior points.”

14. WHAT: `test_nan_handling_in_observables` says one V solve can fail and aggregation skips it. WHY: that reintroduces the v1 problem: dropping V points biases max/selectivity/onset and conflicts with v2’s `SOLVE_FAILED` rule. DO: only skip NaN observables in pure synthetic aggregation tests; any real hard-gate V failure invalidates the whole Δβ evaluation.

15. WHAT: `OUTCOME_A` adds gates not in the locked Phase D pass rule: ≥2/3 secondary, σ-divergence within threshold, argmax/onset tolerances. WHY: Phase D’s locked gate is K primary plus mechanism sign; σ divergence is a non-identifiability flag, not necessarily a hard fail. DO: distinguish “Phase D locked pass” from stricter “recommended Phase E readiness.”

16. WHAT: `argmax_V_mismatch_falsified` and ring-onset mismatch as “falsification” are overclaims. WHY: the bundle’s falsification path is primary-criterion failure; secondary failures are secondary failures unless the acceptance bundle is amended. DO: report them as serious diagnostic failures, not cation-hydrolysis falsification.

17. WHAT: `abs_div > 10 pm²` is invented. WHY: the acceptance bundle only locks the 30% σ-mapping divergence rule; adding an absolute threshold will almost guarantee `NON_IDENTIFIABLE` because the Stern and imposed-σ fits naturally scale by the σ ratio. DO: either remove `abs_div` as a trigger or label it exploratory, not acceptance-derived.

18. WHAT: The ablation fit with `override_sigma=0.141` and the Stern fit are guaranteed to produce Δβ values differing by ~10^6 for the same ΔpKa. WHY: this means σ-divergence failure is expected, not a risk. DO: state upfront that the locked divergence check will almost certainly flag non-identifiability under local Stern σ, and define what result would actually be surprising.

19. WHAT: `minimize_scalar(..., xatol=0.01*bracket_width)` is too loose in Stern β units. WHY: with a `~4.7e7` width, `xatol` is `~4.7e5 pm²`, which corresponds to ~0.05 pKa units at V_kin before σ feedback. Maybe acceptable, but it is not tied to the stated `1 pp²` loss or ΔpKa resolution. DO: set tolerance in ΔpKa-effect space and convert to β using the active σ scale.

20. WHAT: The wall budget still hides the PDE count. “47-59 solves” means 47-59 full V-resolved evaluations, each with ~27 voltages and two-stage continuation. WHY: calling those “solves” understates runtime and fragility by an order of magnitude. DO: budget in actual PDE Newton solves/rungs and specify warm-start reuse across V and across Δβ.

21. WHAT: v2 still does not resolve the two-stage anchor strategy. WHY: v1 said fresh anchor at every V; v2 recounts evaluations but does not say whether anchors are rebuilt per V, walked across V, or cached per mapping. DO: explicitly choose one continuation topology and make wall/convergence estimates from that topology.

22. WHAT: The `Δ_β=0` byte-equivalence target is underspecified. WHY: “existing v10b A.2 record at the intersection” may not share the same V grid, adaptive refinement, or driver observable definitions. DO: pin exact comparison file, exact V, exact fields, and whether the comparison is one-point `V_kin` or a mini scan.

VERDICT: ISSUES_REMAIN