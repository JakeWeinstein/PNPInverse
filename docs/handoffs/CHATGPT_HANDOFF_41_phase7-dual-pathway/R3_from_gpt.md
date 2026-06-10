1. WHAT is wrong: The bisulfate bracket is a stress test, not a physical upper-bound model. Inflating bulk free H+ to `1.1 mol/m3` changes pH, acid BV powers, charge balance, and possibly the OCP-shift consistency.
WHY it matters: If the bracket moves, that proves sensitivity to “extra acid equivalents,” not the magnitude or shape of a real bisulfate closure.
WHAT to do: Label it explicitly as a conservative stress test. Use it only as a trigger for “closure needed or claim downgraded,” not as a quantitative uncertainty band.

2. WHAT is wrong: `produces_h2o2` / `consumes_h2o2` flags can lie unless validated against stoichiometry.
WHY it matters: A typo can silently corrupt the pc observable even though the whole point was to remove index fragility.
WHAT to do: Parser validation: `produces_h2o2=True` requires `stoich[H2O2] > 0`; `consumes_h2o2=True` requires `< 0`; forbid both true.

3. WHAT is wrong: The anodic-share gate is described for the acid-only Phase 0b baseline, but the fitted dual-pathway solution can change `phi_surface` and H2O2 accumulation.
WHY it matters: Passing the baseline does not prove the final fitted pc is not contaminated by reverse R2e.
WHAT to do: Run the same anodic-share check on every accepted sweep corner and final θ; make `<1%` a final reporting gate too.

4. WHAT is wrong: Water-route validation should probably apply only to active water routes.
WHY it matters: A `k0=0` water branch used for ablation/provenance could otherwise force `enable_water_ionization=True` even when it contributes no flux.
WHAT to do: Define “active” as `proton_donor="water" and k0 > 0` unless list omission is mandatory. Put that rule in tests.

5. WHAT is wrong: The L_eff sensitivity only reruns the best two central-film corners.
WHY it matters: If the objective landscape is flat, the best corner at 21.7 or 26.2 µm may not be among the central top two.
WHAT to do: If the top-two losses are close to neighboring corners, rerun the local neighborhood or top 4. This is cheap insurance.

No blocking issues remain after these fixes; the plan now has gates that will either catch the major physics omissions before fitting or force an honest downgrade of the claims.

VERDICT: APPROVED