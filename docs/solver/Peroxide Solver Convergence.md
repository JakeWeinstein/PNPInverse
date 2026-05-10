# Claude Code Handoff: Make BV-PNP Solver Converge Past the V = 0.68 Wall

**Date:** 2026-05-03  
**Goal:** Make the production 3-species log-c + Boltzmann-counterion BV-PNP solver converge for steady-state observables in the peroxide window:

```text
V_RHE ∈ (E_eq_R1, E_eq_R2) = (+0.68, +1.78) V
```

This is needed for the inverse problem because the current production grid reaches only up to about +0.66 V before hitting the `E_eq_R1 = +0.68 V` wall. We need reliable CD/PC data beyond that wall, especially to understand peroxide behavior and parameter identifiability.

---

## 0. Read these context files first

Please read these before implementing:

```text
docs/peroxide_window_investigation.md
docs/clip_observable_investigation.md
bv_solver_unified_api.md
```

Key existing scripts/files referenced by those docs:

```text
scripts/studies/peroxide_window_extension.py
scripts/studies/anodic_cold_start.py
scripts/studies/clip_threshold_sensitivity.py
scripts/plot_iv_curve_unified.py
Forward/bv_solver/forms_logc.py
Forward/bv_solver/boltzmann.py
Forward/bv_solver/grid_per_voltage.py
Forward/bv_solver/solvers.py
scripts/_bv_common.py
```

---

## 1. Current diagnosis

Do **not** treat this as primarily a voltage-step-size problem.

The investigation already tried:

- extending the voltage grid to +1.20 V,
- using `exponent_clip=100`,
- increasing warm-walk substepping,
- increasing bisection depth,
- trying hand-built anodic initial conditions at V = +1.0 V.

Result: voltages through `V_RHE = +0.66` converge, but `V_RHE >= +0.68` fails. The wall occurs exactly at the R1 reversal:

```text
E_eq_R1 = +0.68 V
```

The failure is **not** caused by the eta clip. The peroxide-window investigation used `exponent_clip=100`, which is unclipped over this region.

The working diagnosis is:

1. At anodic voltage, the Boltzmann ClO4- term in Poisson becomes enormous near the electrode.
2. Dynamic H+ must become strongly depleted near the electrode to balance charge.
3. The current initial guesses do not put H+ on that depleted Boltzmann profile.
4. Newton tries to correct the mismatch in one or two steps.
5. The linearized step explodes and SNES aborts.

At V = +1.0 V, the investigation estimated:

```text
phi ≈ +38.9 nondim
c_ClO4 ≈ 0.2 * exp(phi) ≈ 1e16 near electrode
c_H+ must be depleted to roughly exp(-phi)
```

At V = +0.68 V:

```text
phi ≈ +26.5 nondim
c_ClO4 ≈ 0.2 * exp(26.5) ≈ 1e11
```

This is already many orders of magnitude beyond the cathodic-side basin.

The solver is not proving that the peroxide-window steady state does not exist. The PDE should have a solution. The current global Newton path just cannot find the basin.

---

## 2. Do not spend much more time on these

These are unlikely to solve the wall by themselves:

- smaller voltage steps,
- more warm-walk bisection,
- random manual anodic initial guesses,
- changing the eta clip,
- increasing `u_clamp` alone,
- using `clip=50` to force convergence and then trusting PC.

The clip investigation showed that `clip=50` qualitatively distorts peroxide current at cathodic voltages. For peroxide observables, especially PC, use `clip=100` when possible.

---

## 3. Recommended implementation order

The priorities below are ordered by **targetedness of the fix to the diagnosed failure mode**, not by execution order. Cost-vs-probability-of-success suggests a different *execution* order.

Priority list (by physical/numerical leverage on the diagnosed problem):

```text
Priority 1: Nonlinear Poisson-Boltzmann initializer with exponent homotopy
Priority 2: Staged reaction ramp after PB initialization
Priority 3: Pseudo-transient or Gummel-like basin finder before full Newton
Priority 4: Stern-layer formulation test
Priority 5: H+ electrochemical-potential formulation
Priority 6: Pseudo-arclength continuation
```

Recommended execution order (cost-aware):

```text
First:    Diagnostics (Section "Diagnostics to add immediately") — 1 day
Second:   Priority 4 (Stern) — 1 day if hooks are clean
Third:    Priorities 1+2 (PB initializer + staged ramp) — 3-5 days
Fourth:   Priority 5 (muH) — 1-2 weeks, only if 1-4 don't suffice
Defer:    Priorities 3, 6 — only if a working basic path exists to extend
```

The reasoning for putting Stern (Priority 4) first in execution order even though it is fourth in physical leverage:

- It is the cheapest possible attempt — the production code already supports `bv_stern_capacitance_model`, so it requires only a study script, not new solver code.
- If it works, it is the **production answer**, not a fallback. Stern is a standard, physically defensible double-layer model; absorbing the diffuse-layer drop into a capacitive BC dramatically reduces `exp(phi)` magnitudes and is **more complete physics** than the no-Stern Dirichlet treatment, not a workaround.
- If it does not work, you have lost only a day and you proceed to PB+staged with confidence the cheap path was tested.

The aim is to produce a converged state at a small set of target voltages first:

```text
V_TEST = [0.68, 0.70, 0.75, 0.80, 1.00]
```

Only after that works should we run a full peroxide-window grid.

**Hard regression requirement:** any new code path (Stern, PB initializer, staged ramp, muH) must reproduce the existing 3sp+Boltzmann CD/PC on the cathodic-side overlap window V_RHE ∈ [-0.5, +0.1] within the tolerance used by `tests/test_solver_equivalence.py`, before its anodic-regime output is trusted.

**Fallback plan if priorities 1-5 all fail:** accept that the production stack as-formulated is not suitable for the peroxide window. Document the V <= +0.66 ceiling as a permanent constraint of this codebase, and either restrict inverse fits to that range or change the formulation more deeply (e.g. switch to a finite-element electrochemistry framework that natively handles Stern + Frumkin corrections).

---

# Priority 1: Nonlinear Poisson-Boltzmann initializer

## 1.1 Rationale

The current ICs are too far from the anodic double-layer state. A linear potential profile or hand-set electrode values do not give Newton a consistent charge distribution.

Instead, before solving the full BV-PNP system, solve a reduced nonlinear PB-like electrostatic problem at the target voltage. Use it to initialize `phi` and `u_H`.

For H+ at equilibrium:

```text
u_H = log(c_H_bulk) - phi
```

For analytic ClO4-:

```text
c_ClO4 = c_ClO4_bulk * exp(+phi)
```

because ClO4- has z = -1 and the Boltzmann factor is `exp(-z phi)`.

The important idea is that the PB initializer should already balance the huge ClO4- accumulation with H+ depletion before the full mixed Newton solve begins.

## 1.2 Create a new initializer module

Suggested file:

```text
Forward/bv_solver/initializers.py
```

Add functions like:

```python
def solve_pb_initializer(
    ctx,
    params,
    phi_applied_hat: float,
    q_values=None,
    phi_clip: float = 100.0,
    solver_parameters=None,
):
    """Return a scalar Function phi_pb approximating the nonlinear PB double layer.

    q_values ramps the Boltzmann exponent from 0 to 1:
        exp(+q*phi), exp(-q*phi)

    This is an initializer only. It should not change production equations.
    """
```

and:

```python
def assign_pb_initialized_state(ctx, params, phi_pb, h2o2_mode="seed_depleted"):
    """Assign phi, u_H, u_O2, u_H2O2 in the mixed solution from PB state."""
```

Exact access patterns will depend on how `ctx` stores the mixed solution. Follow `set_initial_conditions()` in the current dispatcher/log-c path.

## 1.3 Reuse the existing Boltzmann residual code; do not re-implement

**Stronger than "copy the sign convention":** structure the PB initializer as a *reduced* solve that reuses `Forward/bv_solver/boltzmann.py:add_boltzmann_counterion_residual` directly, with q-ramped `c_bulk` Functions injected (effectively `c_bulk -> q * c_bulk`, or a similar parameter the existing residual exposes). For H+ contribute the analogous Boltzmann source through the same code path by treating H+ as a temporary Boltzmann species during the initializer.

Reasons:

- The production sign and scaling conventions are then **automatic**, not copied.
- Future changes to the Boltzmann residual propagate automatically to the initializer.
- It avoids the entire class of "PB-init converged to a state inconsistent with what the production residual expects" bugs.

If the existing residual API does not allow injecting H+ as a temporary Boltzmann species, the minimal extension is to add a `boltzmann_counterions=[ClO4-, H+_temporary]` mode used only by the initializer. Do **not** write a parallel Poisson assembly.

Conceptually, the PB source still needs to include:

```text
+ dynamic H+ charge density using c_H = c_H_bulk * exp(-q * phi)
+ analytic ClO4- charge density using c_ClO4 = c_ClO4_bulk * exp(+q * phi)
```

but the implementation should route both through the existing residual machinery.

## 1.4 Use exponent homotopy, not a direct q = 1 solve

Do not jump straight to the full Boltzmann factors. Use exponent homotopy:

```python
q_values = [0.0, 0.05, 0.10, 0.15, ..., 1.0]
```

At each q:

```text
c_H    = c_H_bulk    * exp(-q * phi)
c_ClO4 = c_ClO4_bulk * exp(+q * phi)
```

This ramps the nonlinear exponential sensitivity itself, not just a residual multiplier.

The existing `boltzmann_z_scale` ramp is helpful, but it does not fully solve the issue because `exp(phi)` is still the hard part. We need to ramp the exponent.

## 1.5 Potential continuation inside PB initializer

If q-ramping alone fails at high target voltage, add a second continuation dimension:

```text
V: 0.60 -> 0.64 -> 0.66 -> 0.68 -> 0.70 -> ... target
q: 0 -> 1 at each V
```

This PB-only continuation is cheaper and more stable than full BV-PNP continuation because it isolates the electrostatic double layer.

## 1.6 Assign the full initial state

After `phi_pb` is solved:

```text
phi   = phi_pb
u_H   = log(C_H_bulk) - phi_pb
u_O2  = log(C_O2_bulk) or existing O2 bulk IC
u_H2O2 = log(H2O2_seed) plus optional electrode depletion profile
```

For H2O2, start simple:

```text
u_H2O2 = log(H2O2_seed)
```

Then test a depleted profile if needed:

```text
u_H2O2 = log(H2O2_seed) - A * exp(-y / lambda_guess)
```

with A modest at first, not `-80` everywhere. The prior hand-set ICs failed; avoid another arbitrary extreme IC unless guided by residual diagnostics.

---

# Priority 2: Staged reaction ramp

## 2.1 Rationale

Near +0.68, the solver is crossing two transitions at once:

1. the anodic double-layer H+/ClO4- electrostatic transition,
2. the R1 reversal from cathodic peroxide production to anodic peroxide oxidation.

Do not ask Newton to handle both simultaneously.

After PB initialization, ramp the reactions in stages.

## 2.2 Add reaction scale controls

Add optional scale factors to the BV residual:

```text
r1_scale, r2_scale
```

Default both to 1.0 so existing behavior is unchanged.

Implementation options:

### Option A: scale k0 during staged solves

Simpler if k0 is already a `Function` or configurable parameter:

```python
ctx["bv_k0_funcs"][0].assign(r1_scale * K0_R1)
ctx["bv_k0_funcs"][1].assign(r2_scale * K0_R2)
```

This may be easiest, but be careful not to permanently mutate true parameters used later by the inverse code.

### Option B: add explicit reaction scale Functions

Cleaner long term:

```python
ctx["bv_reaction_scales"] = [fd.Function(R).assign(1.0), fd.Function(R).assign(1.0)]
```

Then multiply the net BV rate for reaction i by `ctx["bv_reaction_scales"][i]` in `forms_logc.py`.

Default to 1.0 if missing to preserve backward compatibility.

## 2.3 Staging sequence

Use this sequence at a target voltage:

```text
Stage A: PB initializer only
Stage B: Full PNP transport, BV reactions off
Stage C: Ramp R1 from 0 -> 1, R2 still 0
Stage D: Ramp R2 from 0 -> 1
Stage E: Final full solve with both reactions at 1
```

Concrete ramp:

```python
ramp = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 0.3, 0.6, 1.0]
```

At each ramp value:

```text
solve a few steady or pseudo-transient steps
validate fields
record residual norms and observables
```

If R1 ramp fails before R2 is active, the problem is the R1 reversal/basin. If R1 succeeds and R2 fails, the issue is peroxide coupling.

---

# Priority 3: Pseudo-transient or Gummel-like basin finder

## 3.1 Pseudo-transient continuation

Before full steady Newton, use pseudo-time stepping as a basin finder:

```text
M * (U_new - U_old) / dt + F(U_new) = 0
```

Start with a small pseudo-time step and increase it gradually.

Suggested schedule:

```python
dt_values = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0]
```

Use PB-initialized state as the starting point. At each dt, solve until the update is small or a fixed number of iterations passes.

The purpose is not necessarily to be physically time-accurate. It is to prevent full Newton from taking a basin-destroying step.

## 3.2 Gummel-like block relaxation

If pseudo-transient alone fails, implement a cheap block-relaxed basin finder:

```text
repeat:
    solve/update Poisson with species fixed
    solve/update H+ with phi fixed
    solve/update O2/H2O2 with phi fixed
    under-relax all fields
until residual is small enough for full Newton
```

Use under-relaxation:

```python
U.assign((1 - omega) * U_old + omega * U_candidate)
omega = 0.05 to 0.3
```

This may be more code than pseudo-transient continuation, so treat it as a second-line basin finder.

---

# Priority 4: Stern-layer formulation test

## 4.1 Rationale

The production code already supports:

```text
bv_stern_capacitance_model
```

**This is more complete physics than the current no-Stern Dirichlet treatment, not a numerical workaround.** Stern is the standard double-layer model in electrochemistry: a capacitive boundary layer that absorbs part of the applied voltage before the diffuse layer is reached. Real electrochemical interfaces always have some Stern contribution; the current production code's "no Stern" choice is an idealization, and switching it on moves the model toward (not away from) physical realism.

The numerical consequence is that the diffuse-layer voltage drop is reduced, which directly attacks the diagnosed failure mode: `exp(phi_diffuse)` Boltzmann magnitudes drop dramatically. If a moderate Stern capacitance brings phi_diffuse to within the cathodic-regime range we already converge in, the whole peroxide-window basin-entry problem is resolved.

For these reasons this should be tried **first in execution order** even though it sits at #4 in the physical-leverage list (see Section 3 ordering note).

## 4.2 Minimal test

Create a study script:

```text
scripts/studies/peroxide_window_stern_test.py
```

Run:

```text
V_TEST = [0.66, 0.68, 0.70, 0.75, 0.80, 1.00]
```

For several Stern capacitance settings:

```text
C_stern = [weak, medium, strong]  # use whatever nondim parameterization the code expects
```

Record:

```text
converged?
CD
PC
surface phi
stern voltage drop, if available
diffuse-layer phi drop
surface c_H
surface c_H2O2
surface c_ClO4 analytic
```

Success criterion for this test:

```text
At least V = 0.68, 0.70, 0.75 converge with smooth CD/PC continuation from V = 0.66.
```

If Stern works, it may become the preferred production route for peroxide-window inverse data.

---

# Priority 5: H+ electrochemical-potential formulation

## 5.1 Rationale

The current log-c variable for H+ is:

```text
u_H = log(c_H)
```

But in the anodic double layer, the stable variable is closer to electrochemical potential:

```text
mu_H = log(c_H) + phi
```

At Boltzmann equilibrium:

```text
mu_H ≈ log(c_H_bulk)
```

That means the correct anodic double-layer state is nearly flat in `mu_H`, even though `u_H = log(c_H)` becomes extremely negative near the electrode.

So the current variable choice makes the solution look extreme exactly where the physics is actually simple.

## 5.2 Proposed hybrid formulation

Keep O2 and H2O2 as log concentrations:

```text
u_O2    = log(c_O2)
u_H2O2  = log(c_H2O2)
```

Change H+ only:

```text
mu_H = log(c_H) + phi
c_H  = exp(mu_H - phi)
```

Then the Nernst-Planck flux for H+ becomes:

```text
J_H = -D_H * c_H * grad(mu_H)
```

This should make the H+ equilibrium profile much easier for Newton.

## 5.3 Suggested implementation path

Do not rewrite everything first. Add this as an experimental formulation flag:

```python
formulation="logc_muH"
```

New form file:

```text
Forward/bv_solver/forms_logc_muH.py
```

Reuse as much as possible from `forms_logc.py`, but alter only the H+ concentration and flux definitions.

Validation steps:

1. Verify it reproduces existing log-c results for V <= +0.60 where both converge.
2. Run MMS or a simpler manufactured test for the H+ block.
3. Try the peroxide-window points.
4. Compare CD/PC against the standard log-c formulation on overlapping voltages.

This is a bigger change, but it is likely the cleanest long-term formulation fix.

---

# Priority 6: Pseudo-arclength continuation

## 6.1 Rationale

Voltage may be a poor continuation parameter through the R1 reversal. A pseudo-arclength method could trace the branch even if the solution path bends near `E_eq_R1`.

However, do not do this first. The evidence points to an electrostatic double-layer initialization problem, so pseudo-arclength may still fail unless PB/Stern/muH fixes are in place.

## 6.2 Use after fixing the electrostatic basin

Once PB initialization or Stern-layer modeling can converge at isolated peroxide-window points, implement continuation across the branch:

```text
start at V = +0.60 or +0.66
trace through V = +0.68
continue to +1.0
```

Track solution norm and voltage as continuation variables.

---

# Diagnostics to add immediately

Before implementing major changes, improve failure diagnostics. Every failed solve near +0.68 should print/save:

```text
V_RHE
eta_R1, eta_R2
SNES reason
number of nonlinear iterations
max/min phi
max/min u_H
max/min u_H2O2
max/min c_H = exp(u_H)
max/min c_H2O2 = exp(u_H2O2)
max analytic c_ClO4 = C_ClO4 * exp(phi)
max Poisson residual contribution from ClO4
max Poisson residual contribution from H+
surface averages of phi, c_H, c_H2O2, c_ClO4
CD and PC if available
R1 and R2 observable components separately
```

If possible, save these to JSON per voltage:

```text
StudyResults/peroxide_window_debug/<script_name>_<timestamp>.json
```

Also save PVD/VTK fields for failed and last-success states:

```text
phi
u_H
u_H2O2
c_H
c_H2O2
c_ClO4_boltzmann
charge_density
```

This will make it much easier to tell whether the next failure is still Poisson/H+ dominated or has moved to peroxide chemistry.

---

# Suggested first study script

Create:

```text
scripts/studies/peroxide_window_pb_init_test.py
```

Purpose: test whether PB initialization plus staged reaction ramp can cross the wall.

## Inputs

```python
V_TEST = [0.66, 0.68, 0.70, 0.75, 0.80, 1.00]
CLIP = 100.0
MESH_NY = 200
```

## Algorithm

For each V:

```text
1. Build production params:
   - species = THREE_SPECIES_LOGC_BOLTZMANN
   - formulation = "logc"
   - log_rate = True
   - boltzmann_counterions = [DEFAULT_CLO4_BOLTZMANN_COUNTERION]
   - exponent_clip = 100

2. Build context/forms through dispatcher.

3. Run PB initializer:
   - q ramp from 0 to 1
   - target phi_applied = V / V_T

4. Assign PB-initialized mixed state:
   - phi = phi_pb
   - u_H = log(C_H_bulk) - phi_pb
   - u_O2 = bulk log value
   - u_H2O2 = seed log value initially

5. Solve with BV off.

6. Ramp R1 scale from 0 to 1.

7. Ramp R2 scale from 0 to 1.

8. Final full solve.

9. Extract:
   - converged flag
   - CD
   - PC
   - R1 current component
   - R2 current component
   - surface concentrations
   - debug norms
```

## Output

```text
StudyResults/peroxide_window_pb_init_test/results.json
StudyResults/peroxide_window_pb_init_test/results.csv
StudyResults/peroxide_window_pb_init_test/comparison.png
```

Plot:

```text
CD vs V
PC vs V
surface c_H vs V
surface c_H2O2 vs V
surface c_ClO4 vs V
method/stage reached vs V
```

---

# Success criteria

A fix is promising if:

```text
1. V = +0.68 converges at clip=100.
2. V = +0.70 converges at clip=100.
3. At least one deeper peroxide-window point, e.g. +0.80 or +1.00, converges.
4. CD is continuous across V = +0.68. (PC's value is also continuous, but its
   V-derivative legitimately changes sign at E_eq_R1 because R1 reverses
   direction. A "kink" in PC at +0.68 is expected; a discontinuous jump
   is not.)
5. Surface c_H is depleted near the electrode, not bulk-like.
6. Surface c_ClO4 is large but balanced by charge density / potential structure.
7. Surface c_ClO4 stays within the sterically-allowed range (roughly
   <= 1/A_DEFAULT = 100 nondim). Values much beyond that indicate the
   3sp+Boltzmann model has converged to a non-physical state because the
   analytic-Boltzmann ClO4- residual ignores Bikerman steric saturation.
   See Section "Steric saturation watch" below.
8. The final full residual is genuinely small, not false convergence.
9. Physics validation passes or any validation failure is clearly understood.
10. Reproduces existing 3sp+Boltzmann CD/PC on V_RHE in [-0.5, +0.1] within
    the equivalence-test tolerance (`tests/test_solver_equivalence.py`).
    This is a hard regression requirement on every new code path.
```

A fix is not acceptable if:

```text
1. It converges only by returning to clip=50 artifacts.
2. CD shows a discontinuous jump (not a kink) at +0.68 without a physical reason.
3. H+ remains bulk-like while ClO4 is enormous.
4. It relies on arbitrary hand-set extreme H2O2 or H+ values without a continuation path.
5. It breaks agreement with the existing production solver on V <= +0.60.
6. Surface c_ClO4 grows past ~100 nondim in the converged solution
   (sterically forbidden — the converged answer is non-physical even if
   Newton is happy).
```

## Steric saturation watch

The 3sp+Boltzmann model has one hidden non-physicality at high anodic phi: the analytic Boltzmann residual for ClO4- has **no steric (Bikerman) cap**. The dynamic species in the same model do go through Bikerman with `a_vals_hat = [A_DEFAULT]*3 = [0.01]*3` (saturating at total nondim concentration ~ 1/a = 100), but the Boltzmann counterion residual (`Forward/bv_solver/boltzmann.py`) just sets `c_ClO4 = c_bulk * exp(+phi_clamped)` with no analogous saturation.

Consequence: at V = +1.0 V the Boltzmann formula gives `c_ClO4 ~ 10^16`, but the *physical* c_ClO4 is sterically capped at ~100. A converged peroxide-window solve that lets c_ClO4 grow past ~100 has the same class of artifact as the cathodic-regime clip=50 PC distortion: the equations as solved are not the equations as physically intended.

This is a separate failure mode from "Newton can't converge." Newton might converge happily to a non-physical state. **Validate every converged peroxide-window solve by checking that surface c_ClO4 stays in the sterically-allowed range.** If it does not, the production model itself needs amendment (steric Boltzmann, or switch the counterion to dynamic with steric, or use Stern to keep diffuse-layer phi small enough that the unsaturated Boltzmann is still physically reasonable).

---

# Inverse-problem guidance while this is being fixed

For the inverse problem, current safe/unsafe regions are:

```text
CD at V <= +0.60:
    approximately safe, including cathodic voltages, because CD was nearly clip-neutral.

PC at V < -0.10 with clip=50:
    unsafe, because clip=50 qualitatively distorts peroxide current.

PC at V < -0.10 with clip=100:
    usable as high-fidelity reference where it converges.

V = +0.40 to +0.66:
    useful R1-dominated transition region; CD ≈ PC because R2 is essentially extinct.

V > +0.68:
    unavailable until this handoff succeeds.
```

Do not fit R2 parameters using production `clip=50` cathodic PC data. That data contains a clipping artifact.

---

# Recommended concrete next commit sequence

## Commit 1: diagnostics only (~1 day)

- Add peroxide-window failure diagnostics helper.
- Add JSON/CSV output of failed-stage metrics.
- Re-run existing `peroxide_window_extension.py` for V = [0.66, 0.68, 0.70].

## Commit 2: Stern-layer test (~1 day)  [moved up from old Commit 5]

- Add `scripts/studies/peroxide_window_stern_test.py`.
- Sweep `bv_stern_capacitance_model` over weak/medium/strong values.
- Run V = [0.66, 0.68, 0.70, 0.75, 0.80, 1.00] at each Stern setting.
- Verify equivalence-test regression on V in [-0.5, +0.1] for each Stern setting.
- If at least one Stern setting cleanly traverses +0.68 with continuous CD, smooth PC kink, and surface c_ClO4 within steric range, **stop here**: this is the production answer.
- If no Stern setting works, proceed to Commit 3.

## Commit 3: PB initializer prototype (~2-3 days)

- Add `Forward/bv_solver/initializers.py`.
- Implement scalar PB solve with q-ramp, **reusing the existing
  `add_boltzmann_counterion_residual` machinery** (see Section 1.3).
- Add `scripts/studies/peroxide_window_pb_init_test.py`.
- Test only PB solve and assignment first. Do not turn BV on yet.

## Commit 4: reaction scales (~1 day)

- Add optional `r1_scale`, `r2_scale` mechanism.
- Default both to 1.0.
- Add staged ramp in the study script.
- Verify equivalence-test regression on V in [-0.5, +0.1].

## Commit 5: PB + staged full solve (~1-2 days)

- Try V = [0.66, 0.68, 0.70].
- If +0.68 works, extend to [0.75, 0.80, 1.00].
- Apply all Section "Success criteria" checks, including the
  steric-saturation watch.

## Commit 6: optional muH formulation (~1-2 weeks)

- Only if Commits 1-5 are insufficient.
- Add experimental `formulation="logc_muH"`.
- Validate via MMS on the H+ block.
- Verify equivalence-test regression on V in [-0.5, +0.1] before
  trusting peroxide-window output.

---

# Final note

The current evidence says the wall is a **basin-entry problem caused by the double layer**, not a missing steady state and not an eta-clip issue. The most promising path is to give the full Newton solver a physically consistent anodic double-layer initial state before asking it to solve BV-PNP. The PB exponent-homotopy initializer plus staged reaction ramp is the smallest serious implementation that directly targets that failure mode.
