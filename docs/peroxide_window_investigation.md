# Peroxide-window accessibility — Investigation Log

**Date:** 2026-05-03
**Question:** Can the production solver compute SS observables at
voltages above E_eq_R1 = +0.68 V (the "peroxide window"
V_RHE ∈ (E_eq_R1, E_eq_R2) = (+0.68, +1.78) V), where R1 reverses sign
(anodic peroxide oxidation) and the SS is set by a different basin
than the cathodic-side mass-transport-limited regime?

**Answer:** No, not with the current production stack. The peroxide
window is structurally inaccessible: both cathodic-side warm-walk and
anodic-side cold-IC approaches fail at the same Newton-conditioning
wall driven by the Boltzmann counterion's contribution to the Poisson
residual.

This investigation grew out of `docs/clip_observable_investigation.md`
when interpreting a small dip in the clip=100 PC curve near V=+0.6 V.
That doc closes the clip-distortion question; this one tracks the
separate solver-architecture question of "what's past +0.6 V?"

---

## 1. Cathodic-side extension across E_eq_R1

Ran `scripts/studies/peroxide_window_extension.py`: extended V grid to
V_RHE = +1.20 V with 0.02 V spacing across E_eq_R1, `exponent_clip=100`,
`n_substeps_warm=8`, `bisect_depth_warm=5` (vs production 4 / 3).
Artifacts: `StudyResults/peroxide_window_extension/`.

### 1.1 Results

16/29 voltages converged. The cathodic side (V_RHE ≤ +0.66) fully
converged. The peroxide window (V_RHE ≥ +0.68) was entirely
inaccessible — all 13 voltages from V=+0.68 to V=+1.20 cold-failed,
and the anodic warm-walk from V=+0.66 → V=+0.68 also failed even with
finer sub-stepping. **E_eq_R1 = +0.68 V is a hard wall** for the C+D
orchestrator with the current strategy.

### 1.2 What we did learn

The PC dip at V≈+0.60 (visible in the clip-threshold sensitivity plot
that motivated this work) is a **local minimum**, not a monotonic
descent toward E_eq_R1:

| V_RHE | PC (mA/cm²) | observation |
|---|---|---|
| +0.40 | +1.544e-05 | diffusion-supply floor |
| +0.50 | +1.182e-05 | starts dropping |
| +0.60 | **+2.695e-09** | deepest dip on grid |
| +0.62 | +4.081e-09 | starts rising |
| +0.64 | +7.229e-09 | rising |
| +0.66 | +1.292e-08 | rising further |
| +0.68 | (FAIL) | E_eq_R1 wall |

The minimum is at V≈+0.60 (PC ≈ 3×10⁻⁹) and PC monotonically climbs
back up by ~3× per 0.02 V step toward V=+0.66. Past E_eq_R1 R1
reverses sign (anodic peroxide oxidation), so PC's behavior beyond is
unverifiable from this experiment.

### 1.3 CD ≈ PC in V ∈ [+0.4, +0.66]

In this voltage range R2 is essentially extinct (c_H2O2 too depleted
for R2 to fire), so:

- CD = -I·(R1 + R2) ≈ -I·R1
- PC = -I·(R1 - R2) ≈ -I·R1

The I-V signal in this kinetics-limited transition is **purely R1**;
R1's Tafel kinetics can be read off the curve directly.

### 1.4 Why V=+0.68 fails (cathodic side)

At η_R1 = 0 (V = E_eq_R1) R1's net rate is exactly zero by
reversibility. The SS basin to the right of +0.68 requires R1 running
anodic (peroxide oxidation, c_H2O2 depleted at SS). The seed snapshot
from V=+0.66 lives in the cathodic-R1 basin; even an 8-substep ×
5-bisect warm walk (effectively up to 256 attempts) can't bridge the
basin change.

This is **not a clip artifact** (we ran at clip=100, fully unclipped
on the production grid). It's a genuine SS-basin transition that the
C+D orchestrator + bulk-IC cold ramp cannot navigate on its own.

## 2. Anodic-cold attempt

Ran `scripts/studies/anodic_cold_start.py` with two IC strategies at
V_RHE = +1.0 V (eta_hat=+38.9 nondim):

- **v2:** `u_H2O2 ∈ {-50, -80, -100}` and `u_H+ ∈ {+30, +35}` at the
  electrode (incorrectly elevated u_H+, expecting double-layer
  accumulation for the positive ion).
- **v3:** `u_H2O2 ∈ {-10, -14, -20}` and `u_H+ ∈ {-30, -35, -40, -45}`
  at the electrode (correctly depleted u_H+ for z=+1 at high anodic
  phi; SS Boltzmann/NP equilibrium gives
  `u_H+_surf ≈ u_H+_bulk - phi`).

Used widened `u_clamp = 200` (default 30 / MMS 100 are too tight to
let the SS u_H+ depletion ~ -40 settle without clipping the bulk PDE
coefficient).

All 11 IC combinations across both versions failed Newton in 1-2
iterations.

### 2.1 Diagnosis

At phi = +38.9 nondim (V=+1.0), the Boltzmann ClO4- contribution to
the Poisson source is `+charge_rhs · 0.2 · exp(phi) ≈ 10^16` per unit
cell volume near the electrode. To zero this residual, c_H+ must
Boltzmann-distribute (depleted, c_H+_surf ~ 10⁻¹⁸) inside a thin
Debye layer ~10⁻⁴ nondim thick.

From any non-Boltzmann IC, the Newton step in u_H+ scales as
`-F/J ≈ 10^16 / (charge_rhs · c_H+)`. With c_H+ at either bulk (0.2)
or depleted (10⁻¹⁸) values:

- diagonal Jacobian entry ∝ c_H+ is tiny
- off-diagonal residual is ~10^16

→ linearized step explodes, Newton aborts.

### 2.2 Why the cathodic-side warm-walk hits the same wall

Same mechanism. At V=+0.68, phi=+26.5 nondim, c_ClO4 ≈ 0.2·exp(26.5)
≈ 10^11 — already 8 OOM beyond anything the cathodic basin had to
handle, and a Newton-step away from the cathodic SS where c_H+ ≈ bulk.

### 2.3 The peroxide-window basin DOES exist

PNP + Boltzmann is a well-defined PDE there. The basin lives at the
end of a thin Debye layer that the global Newton solver cannot
navigate to from a non-Boltzmann initial guess. The PDE has a
solution; the production solver just can't find it.

## 3. What would be required to access the peroxide window

Beyond the scope of this investigation; documented for future work.

**Note (2026-05-03):** I initially considered "switch to the 4sp
DYNAMIC species config (ClO4- as a regular NP species with steric
saturation)" as a workaround, on the theory that steric Bikerman
would cap c_ClO4 at the sterically-allowed limit (~100 nondim) where
3sp+Boltzmann lets it grow to ~10¹⁶. **That is not a viable path.**
The Boltzmann counterion reduction wasn't introduced as a math
identity — it's the *enabling simplification* that makes the
production solver converge in the first place. 4sp dynamic is
structurally **harder** than 3sp+Boltzmann, not easier: it has to
evolve the ClO4- profile through a dynamic NP equation with its own
timescale and basin issues, which is exactly the problem analytic
Boltzmann was added to side-step. Empirically (v1 of
`scripts/studies/peroxide_window_4sp.py`, 2026-05-03) 4sp at
production settings cold-fails at every voltage on the production
grid — even at the cathodic side where 3sp+Boltzmann succeeds.

So the actual direction for "future work" is **more Boltzmann, not
less**, plus better double-layer handling:

1. **Debye-layer-aware initialization.** Compute the linearized
   Debye-Hückel solution analytically, use as IC (rather than a
   linear profile in y).
2. **Mesh refinement at the electrode.** The current graded mesh has
   smallest cell ~10⁻⁷ nondim near y=0; Debye length is ~10⁻⁴
   nondim. The layer is captured (~10 cells) but the IC's coarse
   linear interpolation across the layer is not.
3. **Stern-layer formulation.** The production code already supports
   `bv_stern_capacitance_model` which absorbs the double-layer into a
   capacitive boundary condition; could bypass the explicit Debye
   layer entirely. Most promising single-step fix.
4. **PB-PNP all-Boltzmann.** Treat H+ also as a Boltzmann analytic
   counterion (extending the existing reduction to *both* charged
   ions). Loses the H+ dynamic flux at the electrode (which matters
   for BV's `(c_H+/c_ref)²` factor) but eliminates the basin
   transition. Would need a hybrid where H+ is dynamic only near the
   electrode and Boltzmann elsewhere — non-trivial.
5. **Pseudo-arclength continuation.** Replace simple V-stepping with
   a continuation that allows the solution path to curve through
   E_eq_R1 without needing a direct V-step.

What does **not** work:

- 4sp dynamic (per above). Confirmed to fail at production settings
  even on the production grid where 3sp+B succeeds.
- Wider `u_clamp` and `exponent_clip` alone. The cap on the maximum
  c value isn't the binding constraint; the basin navigation is.
- Anodic-cold IC with manually-set u_H+ depletion (per §2 of this
  log). Newton can't bridge from any non-Boltzmann initial guess.

## 4. What this does and does not affect

**Does not affect** the inverse-pipeline-relevant questions answered
in `docs/clip_observable_investigation.md`:

- CD: clip-neutral on the entire production grid V ∈ [-0.5, +0.6]
  (within 0.2%).
- PC: clip-distorted at V_RHE < -0.1 V (sign flip + 4 OOM at clip=50
  vs clip=100).

The "what's past V=+0.6" follow-up turned out to be about the
E_eq_R1 transition, not the clip — and that transition is
inaccessible regardless of clip setting. The clip-effect
investigation closes cleanly without needing the peroxide window.

**Does affect** future studies that would want PC or CD data in the
peroxide window (e.g., for inverse fitting that uses peroxide
oxidation kinetics, or for multi-experiment FIM screening that
includes voltages above +0.68 V). Those studies would need one of the
solver-architecture changes in §3.

## 5. Pointers

- Cathodic-extension runner: `scripts/studies/peroxide_window_extension.py`
- Anodic-cold runner: `scripts/studies/anodic_cold_start.py`
- 4sp dynamic attempt (failed): `scripts/studies/peroxide_window_4sp.py`
- Data: `StudyResults/peroxide_window_extension/`,
  `StudyResults/anodic_cold_start/`,
  `StudyResults/peroxide_window_4sp/`
- Companion log: `docs/clip_observable_investigation.md`
- Production stack: `Forward/bv_solver/`, `scripts/plot_iv_curve_unified.py`
- Boltzmann reduction (the enabling simplification, "Change 1" in the
  writeup): `Forward/bv_solver/boltzmann.py`,
  `writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`.
- Stern layer hooks (most promising single-step fix):
  `Forward/bv_solver/forms_logc.py:223-226` and
  `bv_stern_capacitance_model` in `_bv_common.py`.
