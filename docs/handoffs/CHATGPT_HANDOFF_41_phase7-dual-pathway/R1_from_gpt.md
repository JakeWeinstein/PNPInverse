1. WHAT: `TestWaterRouteEscapesLevichCap` asks for `|cd| > 10x` the 15.4 µm H+ cap. That is about `>5.8 mA/cm2`, above or equal to the 4e O2 Levich ceiling `~5.7`.
WHY: The test is physically impossible or will only pass through a bug.
WHAT to do: Use a threshold like `>3-5x H+ cap` and also assert `<= O2 4e ceiling`.

2. WHAT: The plan says species Levich-film differences are `<=10%`. False for ions. With repo constants, `D_H+/D_O2 ~4.9`, so `delta_H/delta_O2 ~1.7`; `OH-` is `~1.4`.
WHY: H+/OH transport, local pH, and caps are materially wrong under a single O2 film.
WHAT to do: Add a species-specific Levich sensitivity or stop calling this a 10% error.

3. WHAT: Sulfate/bisulfate buffering is absent. At pH 4 in sulfate electrolyte, inert `SO4^2-` is not enough chemistry.
WHY: A water route generates OH- at one OH per electron. Without buffer, pH will likely run too alkaline, and cation hydrolysis may get blamed for missing bulk buffer physics.
WHAT to do: Add/estimate `HSO4-/SO4^2-` buffer capacity before fitting, or prove it is negligible by flux scale.

4. WHAT: “Water routes share E_eq by thermodynamics” is overstated. The proposed water route has no OH/product factors and is irreversible.
WHY: That is not a thermodynamically consistent alkaline BV law; it is an empirical cathodic Tafel branch.
WHAT to do: Either derive the alkaline BV form in model variables, or explicitly call `E_eq` a formal onset parameter and do not use thermodynamic language.

5. WHAT: “Zero weak-form edits” misses the observable layer. Existing `gross_h2o2_current` returns one reaction index, not `R2e_acid + R2e_water`.
WHY: The fit will silently use the wrong peroxide current unless every driver avoids the helper.
WHAT to do: Add a summed-reaction observable or `h2o2_reaction_indices`, with tests.

6. WHAT: Per-reaction “currents” are ambiguous. `mode="reaction", scale=-I_SCALE` gives a 2e-scaled rate; it is wrong for 4e current.
WHY: Reported R4e current will be off by 2x and per-reaction sums will not match total disk current.
WHAT to do: Output both molar rates and electron-weighted currents: `-I_SCALE * (n_e/2) * R_j`.

7. WHAT: Validation likely still uses an old diffusion limit. `assemble_observable_validated` fails any observable above `I_lim`.
WHY: Water-route success is defined by exceeding the H+ cap, so valid runs can be flagged invalid/dropped.
WHAT to do: Make `I_lim` route/observable aware: O2 4e for total, O2 2e for H2O2, not H+ cap.

8. WHAT: Acid `R2e` remains reversible, but the plan says no surface consumption of free H2O2.
WHY: The net `R_j = cathodic - anodic` can consume H2O2 and distort the anodic flank.
WHAT to do: Either make deck-topology R2e irreversible or expose cathodic-only and anodic-only rates and define the RRDE observable explicitly.

9. WHAT: The proposed mechanism says pH drives 2e-to-4e branching, but water 2e and water 4e are both pH-independent.
WHY: The volcano may come from O2 competition/Tafel slopes, not local pH. That breaks the paper mechanism.
WHAT to do: Add pH-clamped, O2-unlimited, water2e-only, and water4e-only ablations before claiming attribution.

10. WHAT: Hydrolysis continuation is under-specified. Existing infrastructure rejects `lambda_hydrolysis_ladder` combined with `kw_eff_ladder`/`c_s_ladder`.
WHY: Stage 2 cannot just “add k_hyd/lambda” inside the current continuation flow.
WHAT to do: Specify and test the sequence: Kw ramp at `lambda=0`, Stern bump, then separate lambda ramp.

11. WHAT: R3 says `E_eq 1.76 V OCP-shifted`, while the convention says all E_eq values are shifted centrally.
WHY: This invites a double shift for the 5th reaction.
WHAT to do: Store all E_eq values unshifted vs RHE and apply the OCP shift in one place, with an eta-preservation test.

12. WHAT: The fit is underdetermined: 4 water parameters fit one H2O2 curve, while total current and local pH are only “validation.”
WHY: You can fit the volcano with wrong current partition and wrong pH.
WHAT to do: Include disk current and pH-vs-current as hard constraints or objective terms, and report parameter identifiability.

13. WHAT: The coarse sweep varies only k0 factors, then may reject the mechanism if shape is wrong.
WHY: Peak position can be alpha-sensitive; a k0-only reconnaissance can falsely reject a viable branch.
WHAT to do: Add a small alpha grid or sensitivity pass before the “shape wrong” gate.

14. WHAT: Digitization sigma from line thickness is not experimental uncertainty, and dense downsampled curve points are correlated.
WHY: WLS will overweight artwork smoothness and plateau length.
WHAT to do: Fit independent marker/trace bins, separate digitization error from experimental error, or use block/correlated weights.

15. WHAT: Water-route validation in `dispatch.py::build_forms` is too late and too narrow.
WHY: Scripts/tests can build forms through multiple backends, and parser/nondim can already propagate bad configs.
WHAT to do: Centralize reaction validation after parsing, cover both `logc` and `logc_muh`, and test water-route + water-ionization-off failures.

16. WHAT: The plan has no global OH/E balance diagnostic.
WHY: With `E = c_H - c_OH`, acid H+ consumption and water OH- production are algebraically indistinguishable; pH alone cannot prove the route is physically behaving.
WHAT to do: Add integrated boundary source, E-flux, OH production/removal, and acid/water current-share checks at every voltage.

VERDICT: ISSUES_REMAIN