1. WHAT is wrong: Re point 3, your own bisulfate estimate makes it load-bearing: `0.83 mA/cm2` is bigger than the H2O2 peak target `0.40` and much bigger than the left plateau `0.18`. Calling it “onset/low-current only” is not defensible.
WHY it matters: The model can assign bisulfate-buffer-supported acid-route current to fitted water-route kinetics, shifting onset and peak shape.
WHAT to do: Add at least a cheap buffer bracket before fitting: no-buffer vs bisulfate-proton-equivalent support. If the curve moves materially, v11 needs the closure or must stop short of a calibrated mechanism claim.

2. WHAT is wrong: Re point 2, post-fit `L_eff` bracketing does not fix the single-film bias.
WHY it matters: `k0` and `alpha` can compensate for the wrong H/OH transport during fitting; checking only the final θ understates parameter and peak-position uncertainty.
WHAT to do: Refit or at least repeat the coarse sweep at the OH- and H+ equivalent films, not only rerun best-fit θ.

3. WHAT is wrong: Re point 15, the “dispatch is the single shared entry” defense is factually false. The repo still has direct calls/imports of `build_forms_logc` and `build_forms_logc_muh` in scripts/tests.
WHY it matters: A water route with `enable_water_ionization=False` can bypass a dispatch-only cross-flag check.
WHAT to do: Put the cross-flag validation in a shared helper called by both backend `build_forms_*` functions, and test direct-backend calls as well as dispatcher calls.

4. WHAT is wrong: The new `reaction_indices`/`reaction_sum` plan is still index-fragile.
WHY it matters: Ablations by list omission or adding R3 can silently change what `[0, 1]` means. That is exactly the kind of off-by-one bug this plan is trying to avoid.
WHAT to do: Add stable reaction labels/roles and build peroxide observables from labels like `h2o2_production=True`, not raw indices only.

5. WHAT is wrong: Re point 8, the anodic-share estimate uses a clean deck η, but actual η includes `phi_surface` through the Stern/diffuse solution.
WHY it matters: The `<1%` argument can be wrong precisely where surface fields are large. The diagnostic is good, but the analytic dismissal is not evidence.
WHAT to do: Make the anodic-share diagnostic an early Phase 0/3 gate. If it fails, decide before fitting whether acid R2e reverse is excluded or the topology claim is weakened.

6. WHAT is wrong: Re point 12, the added total-current hinge `[1.5, 5.99]` is too loose to resolve partition non-identifiability, and the pH hinge only applies if the run reaches `|j| ≈ 3`.
WHY it matters: Many parameter sets can satisfy those hinges while giving different 2e/4e splits and different mechanism stories.
WHAT to do: If a same-condition disk-current curve is not digitizable, downgrade Phase 5b from “fit” to “shape calibration” and require profile/sensitivity plots before any mechanism language.

7. WHAT is wrong: Re point 13, probing only `alpha_w4e` can still falsely reject viable cases.
WHY it matters: `alpha_w2e` controls the anodic/onset side of the volcano; a wrong fixed value can make the whole shape look bad.
WHAT to do: Include at least a tiny two-dimensional alpha probe, e.g. `alpha_w2e ∈ {0.45, 0.63}` crossed with `alpha_w4e ∈ {0.35, 0.50, 0.65}` at the best k0 corner.

8. WHAT is wrong: The future HSO4 closure is described as “same algebraic pattern,” but it is not that simple.
WHY it matters: `HSO4-` carries charge, packing, migration, and depends on the sulfate pool; folding it into `E = c_H + c_HSO4 - c_OH` without a sulfate transport/closure story can break electroneutrality and Poisson accounting.
WHAT to do: When documenting future scope, specify charge and sulfate reservoir assumptions, not just the scalar proton condition.

9. WHAT is wrong: The route ledger checks E balance, but not O2 balance.
WHY it matters: The new sanity gate says “no O2-free current,” but the strongest check is integrated O2 consumption versus bulk O2 flux and total electron current.
WHAT to do: Add O2 boundary consumption and bulk O2 flux to the Phase 3 ledger, with `Σ O2 sinks - O2 bulk influx ≈ 0`.

VERDICT: ISSUES_REMAIN