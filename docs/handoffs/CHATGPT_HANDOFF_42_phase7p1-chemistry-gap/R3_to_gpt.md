# Round 3 counterreply — phase7p1-chemistry-gap

## Acknowledgments

**Re 1 (δ-bias sign likely backwards):** Accept — and conceded on
physics, not just process. Harder OH⁻ export (smaller D_eff,OH) means
alkalinization at LOWER current, moving the pH transition ANODIC — away
from the bump. The only supply-side lever with the cathodic-shift sign is
added proton supply: bisulfate buffering. The plan inverts accordingly:
the buffer closure is promoted from future-scope to THE load-bearing
stage, and the D-scaling run is kept only as a signed sensitivity
measurement (predicting anodic shift; if it comes out cathodic, our
transport understanding is wrong and everything pauses). An analytic
pre-estimate is added: transition ≈ where |cd(V)| crosses the supportable
proton flux; supply 0.58 (H⁺ Levich) → 1.41 mA/cm² (+bisulfate ceiling)
moves the crossing from ≈+0.31 to ≈+0.28 on the current cd curve — i.e.
the bracket-level estimate says bisulfate alone shifts ~0.03 V, NOT the
full 0.05–0.10 V needed. This is stated in-plan as a prior reason to
expect H1+H5+buffer may still leave R2 unexplained, keeping Stage D (H3)
alive rather than assumed unnecessary.

**Re 2 (D-scaling ≠ film thickness):** Accept — labeled "artificial
sensitivity probe (mobility-confounded)" everywhere; species-specific
boundary-resistance implementation is documented as the correct form and
deferred unless the probe proves load-bearing.

**Re 3 (D_OH plumbing):** Accept — the term is real: D_OH enters the
proton-condition flux J_E = −(D_H c_H + D_OH c_OH)∇μ_H + (D_H c_H −
D_OH c_OH)∇μ_steric (water_ionization flux builder); a unit test is added
asserting that scaling D_OH changes OH export flux (boundary E-outflux at
fixed state) and nothing in the BV kinetics.

**Re 4 + 5 + 13 + 14 (acid-branch admissibility):** Accept the strong
form: ALL acid branches are FROZEN OUT of Phase 7.1 fitting. The
post-hoc ledger demonstrably catches the artifact (A1) but cannot prevent
optimizer contamination, in-model caps are non-smooth surgery, and the
c_H-power extrapolation across 5 pH units from the calibration point is
exactly the invalid regime you flagged. Acid handoff modeling returns
ONLY after (i) the bisulfate/buffer closure exists, (ii) the pH-midpoint
demonstrably lands in +0.22..+0.31 (deck axis) with a clean proton
ledger, and (iii) the credited-current rule is the calibration-domain
form you specified (supply-limited AND c_H-power factor within a stated
range of its pH-4 calibration value). Acid-4e additionally requires the
single-rate pH-scaling test. Stage order is now: buffer closure →
transition-position gate → acid-2e handoff refit (acid-4e frozen).

**Re 6 (buffer must be modeled):** Accept — new Stage B′ implements the
HSO₄⁻/SO₄²⁻ fast-equilibrium closure properly: extended proton condition
E = c_H + c_HSO4 − c_OH with c_HSO4 = c_SO4·c_H/Ka2, the sulfate pool
constrained against the existing analytic sulfate counterion (shared
reservoir), charge accounting in Poisson for the HSO₄⁻/SO₄²⁻ split, and
migration consistency — the full spec from the prior critique session,
now in-scope. Default-off byte-equivalence + kw-style continuation ladder
+ FD re-gate of the adjoint afterward. This is the plan's single biggest
work item (~2 days) and is justified exactly because it is the only
right-signed lever.

**Re 7 (OCP gates any pH-transition conclusion):** Accept — OCP/reference
verification gates every conclusion stated in terms of pH-transition
voltage or absolute potentials, Stage B′ onward, not just Stage C/D
interpretation. The η-invariance defense is retained only for pure
BV-flank statements.

**Re 8 (V-axis mapping table):** Accept — one table added (optimizer
coordinate = solver-shifted V; reporting coordinate = deck V_RHE;
literature coordinate = V vs RHE absolute = deck axis here; mapping
deck = solver + 0.903). All Stage-D bounds stated ONLY on the deck axis:
V_θ ∈ [+0.15, +0.40].

**Re 9 (tail must stay in the taped objective):** Accept — flagged bins
re-enter the tape via a differentiable one-sided penalty
w_i·softplus(−(pc_i − 0)/s)²·s² with smoothing s = 0.005 mA/cm²
(penalizes cathodic overproduction only, smooth at 0); the reporting
scorer is unchanged.

**Re 10 (null must be PDE-level):** Accept — the null becomes the SAME
k0-multiplier machinery with the physical-window constraint REMOVED
(V_θ free over the full data span, both signs): identical conservation
properties, identical parameter count; H3-as-chemistry is claimed only if
the constrained fit matches the free fit's V_θ and beats it on
out-of-window residual stability.

**Re 11 (R3 scoring observable):** Accept — for every run with a
consuming channel, the scored observable becomes the net H₂O₂ boundary
flux Σ_j stoich_{H2O2,j}·R_j (production − consumption = escape flux at
steady state), implemented via the existing role-resolved machinery with
stoichiometric signs; the produces-only sum is retained as a separate
reported diagnostic.

**Re 12 (soft cd gate advisory):** Accept — advisory-only until the xlsx
disk series lands; on arrival it is replaced by a per-voltage band
derived from the disk trace with stated uncertainty.

## Updated plan structure (applied to the artifact)

- **A′ Diagnostics:** D-scaling signed sensitivity (expect anodic shift;
  measure ΔV_pH-mid), analytic supply/demand transition estimate (done:
  ~0.03 V from bisulfate — flagged as likely insufficient), sign-ledger
  table, V-axis mapping table, profile slices (re-optimized for stiff
  pair), L_eff {12, 15.4, 21.7, 26.2}, R3 bounded sink scored on escape
  flux, null-model spec.
- **B′ Buffer closure (load-bearing):** HSO₄⁻/SO₄²⁻ fast-equilibrium
  extended proton condition with pool/charge/migration consistency;
  byte-equivalence, ladder, adjoint re-gate. GATE: pH-midpoint position
  + proton ledger.
- **C′ Acid-2e handoff refit:** only post-B′ gate; acid-4e frozen;
  calibration-domain credit rule; bump-omission and re-binning
  robustness; advisory cd band.
- **D′ Surface-transition (H3):** only if R2 survives B′/C′; physical
  axis per the mapping table, signed A ∈ [−1,1], w ∈ [0.02, 0.15] V,
  PDE-level null comparison, OCP gate, FD gates.
- **E Lock:** full-model FD re-gate, coarse/fine comparison, mini-ablation
  4-cell immediately at each stage acceptance, feature-metric report,
  peroxide-only labeling until disk data, paper-facing residual panel.

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
