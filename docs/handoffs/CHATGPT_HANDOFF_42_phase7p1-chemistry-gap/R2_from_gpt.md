1. WHAT: Re 15 claims species-specific ionic δ bias has the sign needed to move the pH transition cathodic.

WHY: That sign is not established and is likely backwards. Reducing effective H/OH transport usually makes OH accumulation worse, so the alkaline-to-acid transition should move anodic, not cathodic. If true, your new primary H5 diagnostic cannot rescue H1.

WHAT TO DO: Before any acid refit, run water-only D-scaling and report the pH-midpoint shift in volts. If it moves anodic, H5-as-rescue is dead unless bisulfate/buffer chemistry overcompensates.

2. WHAT: Re 15’s `D_eff,i = D_i·δ_O2/δ_i` is not equivalent to species-specific film thickness in PNP.

WHY: Scaling D also scales electrophoretic mobility and conductivity. A thicker external diffusion layer changes mass-transfer resistance, not intrinsic mobility inside the electrostatic film. You may move the pH transition by changing the wrong physics.

WHAT TO DO: Prefer species-specific Robin/mass-transfer boundary resistances or a boundary-layer operator. If you keep D-scaling, label it an artificial sensitivity, not a film-thickness correction.

3. WHAT: Re 15 mentions scaling `D_OH`, but OH is algebraically closed through `c_OH = Kw/c_H`, not a normal transported species.

WHY: If OH has no independent transport equation, “δ_OH/δ_O2” may not be applied where you think. The diagnostic could be a no-op, partial-op, or inconsistent flux modification.

WHAT TO DO: Point to the exact flux term where `D_OH` enters the proton-condition equation and verify with a unit test that changing `D_OH` changes OH export but not unrelated kinetics.

4. WHAT: The A1 interpretation says the far-cathodic acid current is “protons supplied by Kw closure,” but the deeper problem is the acid Tafel branch being extrapolated until overpotential overwhelms c_H suppression.

WHY: A post-hoc source ledger catches the artifact after the solve, but the invalid acid branch can still steal O2, reshape pc, and contaminate optimizer paths.

WHAT TO DO: Add a pre-solve or in-model admissibility cap: acid current cannot exceed physically supplied proton flux, or use a reversible/mass-action acid model with explicit proton/buffer transport. Do not rely only on after-the-fact flagging.

5. WHAT: The acid acceptance rule “surface pH < 7” is too loose for c_H²/c_H⁴ acid kinetics.

WHY: At pH 6-7, acid-route rates should be orders of magnitude below their pH 4 calibration unless the Tafel exponential is again doing unphysical work. The rule would still credit near-neutral acid artifacts.

WHAT TO DO: Replace the pH<7 rule with a calibration-domain rule, e.g. acid current is credited only where both supply-limited and the c_H-power factor is within an explicitly allowed range relative to pH 4.

6. WHAT: Re 16’s bisulfate bracket is still only a nonphysical upper bound, but the updated synthesis now needs missing bisulfate buffering to move the pH transition.

WHY: A trigger-only upper bound cannot support the central H1/H5 rescue mechanism. If buffer chemistry is load-bearing, it must be modeled, not waved through as a bracket.

WHAT TO DO: Add an explicit HSO4-/SO4 equilibrium or buffer-capacity closure before making any final claim that bisulfate shifts the transition into the bump window.

7. WHAT: Re 29’s defense that Stage B is insensitive to OCP because η is shift-invariant is too narrow.

WHY: The BV exponent may be invariant, but PNP migration, Stern charge, Bikerman crowding, and surface pH depend on the absolute electrostatic boundary convention. Your central pH-transition diagnosis may change with the OCP shift.

WHAT TO DO: Make OCP/reference verification a gate for any conclusion involving pH-transition voltage, not just Stage C interpretation.

8. WHAT: Stage C’s voltage axis is still confused: “vs RHE absolute,” “un-shifted,” and “deck axis [+0.15,+0.40]” are all present.

WHY: A redox/Frumkin transition lives on an absolute electrode-potential scale. If the fitted variable and reported variable differ by the deck shift, Vθ can be misinterpreted by almost a volt.

WHAT TO DO: Define one mapping table: optimizer coordinate, plotted RHE coordinate, shifted internal coordinate, and literature-comparison coordinate. Then state Vθ bounds in exactly one of them.

9. WHAT: Re 28 excludes thresholded-tail bins from the taped objective.

WHY: That removes the only optimization pressure enforcing the zero tail. A model can overproduce peroxide beyond +0.45 and still pass the fitted objective.

WHAT TO DO: Use a differentiable one-sided penalty in the taped objective for flagged bins. Do not defer censoring if the tail affects onset/cliff fits.

10. WHAT: Re 32’s null model acts directly on `pc`, while H3 acts inside the PDE through k0.

WHY: Same parameter count does not make them comparable. The pc-null can violate conservation and may win/lose for reasons unrelated to chemistry.

WHAT TO DO: Use the null only as a descriptive curve-fit baseline. For model comparison, compare PDE-level alternatives with comparable conservation constraints.

11. WHAT: Re 26’s R3 sink diagnostic must change the observable from net production to escape flux.

WHY: Your Re 27 defense says net production equals escape only absent consumption. Once R3 exists, scoring net production against ring-derived flux is wrong.

WHAT TO DO: For every R3 run, score boundary escape flux to the ring-equivalent observable, not net 2e production.

12. WHAT: The soft total-current gate `|cd| at −0.3 within [1.5, 4.5]` is arbitrary and may conflict with the later disk-data gate.

WHY: It can reject a peroxide-shape hypothesis before the actual disk series arrives, or let through a model that still fails the true disk curve shape.

WHAT TO DO: Either make this explicitly advisory until xlsx lands, or derive the range from the deck trace with uncertainty and apply it at the same voltages as the reported mismatch.

13. WHAT: Acid-4e remains in B-ii despite A1 showing it can be pure closure artifact.

WHY: A “hard upper bound” is not enough if the branch’s only demonstrated behavior is O2 theft under laundered protons. It adds identifiability risk and a false 4e partition lever.

WHAT TO DO: Freeze acid-4e out of refits until the proton-source cap/buffer model is implemented and the single-rate pH-scaling test passes.

14. WHAT: The updated plan still allows Stage B acid refits before proving the pH transition can be moved into the bump window.

WHY: A1 already showed acid contribution is zero in the current bump window. Without a successful H5/buffer shift first, Stage B is compute theater.

WHAT TO DO: Gate Stage B on a diagnostic pH-midpoint shift into roughly +0.22..+0.31 with acceptable total current and no proton-ledger violation.

VERDICT: ISSUES_REMAIN