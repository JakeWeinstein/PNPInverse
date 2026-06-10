# Round 4 counterreply — phase7p1-chemistry-gap

All ten accepted, each closed with the concrete specification:

**Re 1 (sulfate laundering):** Accept. Finite-sulfur constraint is
explicit in B′: total sulfur S_tot is the existing analytic counterion
pool (fixed bulk loading, Boltzmann/Bikerman-distributed); the speciation
split satisfies c_SO4 + c_HSO4 = S_tot(x) pointwise — no new sulfur is
created by the equilibrium. Ledgers added: (i) HSO4-deprotonation
boundary flux, gated against the sulfur transport ceiling (0.83 mA/cm²
equivalent at the film); (ii) closed-cell no-flux test: total sulfur
integral conserved to 1e-12 over a solve. Any candidate whose buffer
proton delivery exceeds the sulfur transport ceiling fails — same
structure as the Kw guard.

**Re 2 (Ka2 units):** Accept, stated exactly: Ka2 = 10^(−1.99) M =
10.23 mol/m³; nondimensional Ka2_hat = 10.23 / C_SCALE = 8.52 (C_SCALE =
1.2 mol/m³). Unit test: bulk pH 4 ([H⁺] = 0.1 mol/m³), S_tot = 100
mol/m³ ⇒ c_HSO4 = S_tot·c_H/(Ka2 + c_H) = 100·0.1/10.33 = 0.968 mol/m³
(within 3% of the 1.0 used in the ceiling estimate — consistent).

**Re 3 (bulk electroneutrality):** Accept verbatim: a speciation test
asserting [Cs⁺] + [H⁺] = 2[SO4²⁻] + [HSO4⁻] + [OH⁻] at the film edge,
uniform state, zero field, residual norm ≤ 1e-12 (the Cs⁺ bulk loading is
re-derived from the split rather than held at its pre-split value).

**Re 4 (write the math first):** Accept. A derivation note
(docs/phase7/bisulfate_closure_derivation.md) precedes any code: exact
extended-E residual, flux form with per-species (z, D, steric) for HSO4⁻
(z=−1, D=1.33e-9) vs SO4²⁻, Jacobian terms, and the four tests you named:
buffer capacity dE/dpH vs the analytic expression, zero-field diffusion
limit, migration sign (HSO4⁻ migrates TOWARD the cathode — sign test),
sulfur conservation. The note goes through this critique loop's session
files as an appendix before implementation.

**Re 5 (pH-midpoint metric):** Accept, defined now: surface pH =
boundary-area-averaged (existing muh-aware diagnostic); transition
midpoint V_mid = the most anodic crossing of pH = 6.5 (midway 4↔9),
linear interpolation between grid points; if crossings > 1 or the
profile is non-monotone in V, V_mid is reported with a NONMONOTONE flag
and the gate evaluates the most anodic crossing while the flag forces a
finer V-grid rerun across the transition. Gate window: V_mid ∈
[+0.22, +0.31] deck axis.

**Re 6 (raw-data gate restored):** Accept — explicit: any bump-specific
chemistry claim (C′ attribution or D′ existence) is GATED on the raw
ring xlsx (or documented confirmation from the data owners) showing the
+0.22..+0.27 feature. Until then C′/D′ outputs are labeled
"digitized-curve sensitivity only" in summaries and the paper draft.

**Re 7 (tail threshold):** Accept. The censoring limit is the actual
processing threshold converted to disk-equivalent units: j_ring < 0.001
mA/cm²(ring) ⇒ 0.001·0.11/(0.224·0.196) = 0.0025 mA/cm²(disk-equiv).
One-sided penalty activates only below −0.0025 (cathodic of the
threshold), with sensitivity runs over threshold ×{0.5, 2} and smoothing
s ×{0.5, 2} reported.

**Re 8 (escape flux = outer-boundary flux):** Accept. For all
consuming-channel runs the scored observable is the assembled H2O2 flux
through the bulk boundary (D_H2O2·∂c/∂n integrated at the film edge),
NOT the reaction sum; production (Σ produces), consumption (Σ consumes),
and the mass-balance mismatch |boundary flux − net reaction| are reported
separately, with the mismatch required < 1% as the steady-state check.

**Re 9 (null protocol predefined):** Accept, fixed before fitting:
(i) free-null V_θ stability: fitted over rebinnings {25, 33, 45} and the
bump-omitted set — claim requires the free-null V_θ distribution to be
unimodal with spread ≤ 0.03 V AND inside the physical window;
(ii) the constrained (windowed) fit's objective within 10% of the free
null's; (iii) the bump-omitted constrained refit must still place V_θ in
the window (feature not purely fit to the bump bins). All three required
for a chemistry claim; otherwise H3 is reported as a descriptive
correction.

**Re 10 (acid credit range, numeric):** Accept, set now: acid-2e current
is credited as chemistry only where surface pH ≤ 5.0 (c_H within 10× of
the pH-4 calibration ⇒ c_H² factor within 1e-2 of calibration); acid-4e
only where surface pH ≤ 4.5 (c_H⁴ factor within 1e-2). Outside these,
acid current is artifact-flagged regardless of ledger supply checks.

## Artifact status

All ten specifications are folded into the plan
(~/.claude/plans/phase7p1-chemistry-gap.md): B′ gains the
finite-sulfur/units/electroneutrality/derivation-first subsections and
the defined V_mid gate; the raw-data gate is restored as a top-level
gate; the tail penalty, escape-flux scoring, null protocol, and acid
credit ranges are stated with the numbers above.

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
