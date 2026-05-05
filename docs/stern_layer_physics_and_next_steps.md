# Stern layer modeling: physics judgment and Claude Code next steps

**Date:** 2026-05-03

## Bottom line

Turning on a Stern layer is a physics change, not just a numerical trick.
But it is a physics change toward a more realistic electrochemical interface.

The current no-Stern production model is the idealized limiting case where the
solution-side potential at the electrode equals the metal potential:

```text
phi_s = phi_m = phi_applied.
```

That means the compact-layer voltage drop is assumed to be zero. Real
electrode/electrolyte interfaces generally have a compact Stern layer: ions
cannot place charge centers directly on the metal surface, solvent is
structured near the interface, and some of the applied voltage drops before
the diffuse PNP layer begins.

The more physical voltage split is:

```text
phi_m - phi_bulk = Delta_phi_Stern + Delta_phi_Diffuse.
```

The Stern layer is represented as a compact capacitor:

```text
sigma = C_S (phi_m - phi_s),
```

where `C_S` is the Stern capacitance. In the limit

```text
C_S -> infinity,
```

the Stern drop vanishes and the model returns to the current no-Stern
Dirichlet behavior:

```text
phi_m - phi_s -> 0.
```

So the right framing is:

- **No-Stern:** idealized infinite-Stern-capacitance interface.
- **Finite Stern:** more physical compact-layer/Frumkin interface.

Finite Stern is only defensible if `C_S` is physically plausible or treated as
a calibrated model parameter. It should not be selected only because it makes
Newton converge.

## Why this matters for the current solver

The peroxide-window failure is driven by the resolved diffuse-layer potential.
In no-Stern mode, the electrode Dirichlet condition forces:

```text
phi_s = phi_m.
```

At positive voltage this makes the analytic Boltzmann counterion source huge:

```text
c_ClO4,s = c_ClO4_bulk exp(phi_s).
```

At `V_RHE = +0.68 V`, `phi_m ~= 26.5` nondimensional. At `+1.0 V`,
`phi_m ~= 38.9`. Those values imply enormous counterion accumulation unless
the solver simultaneously discovers a matching proton depletion layer.

With finite Stern capacitance, the solution potential `phi_s` can be much
smaller than the metal potential `phi_m`; the missing voltage sits in the
compact Stern layer. This directly reduces `exp(phi_s)` in the PNP domain and
attacks the diagnosed Poisson/Boltzmann wall.

## Comparison with electrochemical-potential formulation

The `mu_H = log(c_H) + phi` formulation is still a good long-term idea.
It makes the proton variable smoother in the Debye layer:

```text
c_H = exp(mu_H - phi),
J_H = -D_H c_H grad(mu_H).
```

But it does **not** remove the analytic counterion source:

```text
c_ClO4 = c_bulk exp(phi).
```

So `mu_H` improves the proton block while leaving the main Poisson stiffness
in place. Stern changes the interfacial electrostatic model so the resolved
diffuse-layer potential itself can be smaller.

Practical recommendation:

1. Test Stern first.
2. Keep `debye_boltzmann` as the high-voltage IC.
3. Implement `logc_muH` only if Stern/PB-initialized paths do not solve the
   peroxide-window wall, or if a cleaner long-term formulation is needed.

## Current code status

The lower-level solver already has Stern hooks:

- `Forward/bv_solver/config.py` parses `bv_bc.stern_capacitance_f_m2`.
- `Forward/bv_solver/nondim.py` nondimensionalizes it into
  `bv_stern_capacitance_model`.
- `Forward/bv_solver/forms_logc.py` switches to the Stern/Frumkin form when
  `bv_stern_capacitance_model > 0`:

```text
eta = phi_applied - phi - E_eq
```

and replaces the electrode Dirichlet BC for `phi` with a Robin term:

```text
F_res -= stern_coeff * (phi_applied - phi) * w * ds(electrode_marker)
```

Important gap:

`scripts/_bv_common.py::make_bv_solver_params()` does **not** currently expose
a `stern_capacitance_f_m2` argument, and `_make_bv_bc_cfg()` does not write it
into `params["bv_bc"]`. Claude Code should add this first.

## Implementation plan for Claude Code

### Step 1 - expose Stern in the shared BV factory

Patch `scripts/_bv_common.py`:

1. Add optional keyword to `_make_bv_bc_cfg()`:

```python
stern_capacitance_f_m2: Optional[float] = None
```

2. If the value is not `None`, write:

```python
cfg["stern_capacitance_f_m2"] = float(stern_capacitance_f_m2)
```

3. Add the same keyword to `make_bv_solver_params()` and pass it through to
   `_make_bv_bc_cfg()`.

4. Update the docstring to say values are physical `F/m^2`.

5. Preserve exact current behavior when the value is `None` or `0.0`.

### Step 2 - add a minimal config/unit test

Add or extend a lightweight test that builds solver params with:

```python
stern_capacitance_f_m2=0.2
```

and verifies:

```text
solver_params[10]["bv_bc"]["stern_capacitance_f_m2"] == 0.2
```

Also verify no key is emitted, or the parsed value is inactive, when
`stern_capacitance_f_m2=None`.

### Step 3 - create the Stern study script

Create:

```text
scripts/studies/peroxide_window_stern_test.py
```

Use the production stack:

```text
THREE_SPECIES_LOGC_BOLTZMANN
DEFAULT_CLO4_BOLTZMANN_COUNTERION
formulation="logc"
log_rate=True
initializer="debye_boltzmann" for high-voltage cold starts
exponent_clip=100 if the script already supports overriding it
```

Run at least:

```text
V_TEST = [0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00]
```

Use physically plausible Stern capacitance values:

```text
C_S = [None, 0.05, 0.10, 0.20, 0.40, 1.00] F/m^2
```

Unit conversion sanity:

```text
1 F/m^2 = 100 uF/cm^2
```

So the range above covers roughly `5` to `100 uF/cm^2`, with
`0.10-0.40 F/m^2` being a reasonable compact-layer scale to inspect.

### Step 4 - record the right diagnostics

For each `(C_S, V)` record:

```text
converged
failure reason / exception
CD
PC
surface phi_s
metal potential phi_m
stern drop phi_m - phi_s
estimated diffuse drop phi_s - phi_bulk
surface c_H
surface c_H2O2
analytic surface c_ClO4 = c_bulk exp(phi_s)
total SNES iterations
steady-state steps
initializer fallback flag and reason
```

Save:

```text
StudyResults/peroxide_window_stern_test/results.csv
StudyResults/peroxide_window_stern_test/results.json
StudyResults/peroxide_window_stern_test/summary.md
```

Optional plot:

```text
CD/PC vs V by C_S
phi_s and phi_m - phi_s vs V by C_S
c_ClO4,s vs V by C_S
```

### Step 5 - validation gates

Separate code regression from physics comparison.

Code-regression gate:

```text
C_S = None or 0.0 must reproduce the current no-Stern behavior.
```

Finite-Stern physics gate:

```text
Finite C_S is allowed to differ from no-Stern because it is a physics branch.
Do not require exact CD/PC equality to no-Stern on the overlap window.
Instead, quantify the difference and check that it is smooth and physically
interpretable.
```

Peroxide-window success gate:

```text
At least V = 0.68, 0.70, and 0.75 converge for one physically plausible C_S.
CD/PC should continue smoothly from V = 0.66.
surface c_ClO4 should stay below the steric scale implied by a=0.01
(order 100 nondim) or the summary must flag the state as physically suspect.
```

Large-capacitance consistency gate:

```text
A very large C_S should approach no-Stern Dirichlet behavior.
Use this only as a sign/implementation sanity check; it may be numerically
stiff.
```

### Step 6 - decide after the sweep

If a plausible `C_S` crosses the `+0.68 V` wall cleanly:

1. Treat Stern as the preferred peroxide-window model branch.
2. Document the selected `C_S` and sensitivity across nearby values.
3. Update inverse/plot scripts to make the model branch explicit:

```text
model = "no_stern"
model = "stern_Cs_0p20_F_m2"
```

Do not silently replace historical no-Stern results.

If Stern does not cross the wall:

1. Keep the Stern study as a negative result.
2. Proceed to PB initializer with exponent homotopy.
3. Only then move to `formulation="logc_muH"` as the larger formulation
   rewrite.

## Suggested Claude Code prompt

Use this as the next task prompt:

```text
Read docs/stern_layer_physics_and_next_steps.md,
docs/ic_refinement_study.md, docs/Peroxide Solver Convergence.md,
and Forward/bv_solver/forms_logc.py. Implement the Stern-layer test path.

First expose stern_capacitance_f_m2 in scripts/_bv_common.py so
make_bv_solver_params can pass it into params["bv_bc"]. Add a lightweight
test for this config wiring.

Then create scripts/studies/peroxide_window_stern_test.py. Sweep
C_S = [None, 0.05, 0.10, 0.20, 0.40, 1.00] F/m^2 over
V_TEST = [0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00] using the production
3sp logc + Boltzmann + log-rate stack, preferably with debye_boltzmann
initialization for high-voltage cold starts.

Save CSV/JSON/summary artifacts under
StudyResults/peroxide_window_stern_test/. Report convergence, CD/PC,
surface phi, Stern drop, surface H+, H2O2, analytic ClO4, SNES work,
and initializer fallback status.

Do not treat finite Stern as a strict regression to no-Stern. It is a
physics branch. Preserve exact no-Stern behavior for C_S=None/0 and quantify
finite-Stern deviations separately.
```
